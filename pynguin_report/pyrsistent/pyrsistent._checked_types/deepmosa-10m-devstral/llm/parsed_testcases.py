####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test__new__with_valid_key_type.
# Failed to parse test__new__with_valid_value_type.
# Partially parsed test__new__with_valid_invariant. Retrieved 2/2 statements.
# Partially parsed test__new__with_default_serializer. Retrieved 1/6 statements.
# Failed to parse test__new__with_custom_serializer.


def test_case_0():
    var_0 = 'not_callable'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = True
    var_1 = (var_0,)

def test_case_0():
    var_0 = True
    var_1 = (var_0,)

def test_case_0():
    var_0 = '__serializer__'

def test_case_0():
    pass



# Parsed testcases at query #2
#--------------------------

# Failed to parse test_checked_pset_constructor_with_empty_initial.
# Partially parsed test_checked_pset_constructor_with_list_initial. Retrieved 5/10 statements.
# Partially parsed test_checked_pset_constructor_with_pmap_initial. Retrieved 9/14 statements.
# Partially parsed test_checked_pset_constructor_with_invalid_type. Retrieved 3/7 statements.
# Partially parsed test_checked_pset_constructor_with_invalid_invariant. Retrieved 4/8 statements.


def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 1
    var_6 = 2
    var_7 = 3

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = True
    var_5 = True
    var_6 = True
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0.pmap(var_7)
    var_9 = 1
    var_10 = 2
    var_11 = 3

def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 'invalid'
    var_2 = [var_1]
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = -1
    var_2 = -2
    var_3 = [var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_checked_pvector_constructor_empty. Retrieved 1/5 statements.
# Partially parsed test_checked_pvector_constructor_valid_elements. Retrieved 5/9 statements.
# Partially parsed test_checked_pvector_constructor_from_python_pvector. Retrieved 6/10 statements.
# Partially parsed test_checked_pvector_constructor_invalid_type. Retrieved 4/8 statements.
# Partially parsed test_checked_pvector_constructor_invalid_invariant. Retrieved 5/9 statements.


def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')

def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

import pyrsistent._pvector as module_0

def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.python_pvector(var_4)

def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = -1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #4
#--------------------------

# Failed to parse test_maybe_parse_user_type_with_preserved_type.
# Failed to parse test_maybe_parse_user_type_with_non_iterable_type.
# Failed to parse test_maybe_parse_user_type_with_iterable_of_types.
# Failed to parse test_maybe_parse_user_type_with_nested_iterable.


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
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 4/7 statements.
# Partially parsed test_check_types_with_invalid_types. Retrieved 4/8 statements.
# Partially parsed test_check_types_with_multiple_valid_types. Retrieved 4/7 statements.
# Partially parsed test_check_types_with_multiple_invalid_types. Retrieved 4/8 statements.
# Partially parsed test_check_types_with_empty_iterable. Retrieved 1/4 statements.
# Partially parsed test_check_types_with_empty_expected_types. Retrieved 5/7 statements.
# Partially parsed test_check_types_with_custom_exception. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2.0
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3.0
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = []

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_serialize_default. Retrieved 5/9 statements.
# Partially parsed test_serialize_with_format. Retrieved 5/9 statements.


def test_case_0():
    var_0 = lambda x: (x >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = lambda x: (x >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_serialize_default_format. Retrieved 5/9 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 7/12 statements.
# Partially parsed test_serialize_empty_vector. Retrieved 1/5 statements.
# Partially parsed test_serialize_mixed_types. Retrieved 5/9 statements.


def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = lambda fmt, v: str(v) if fmt == 'str' else v
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = 'str'

def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')

def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 1
    var_2 = 2.5
    var_3 = 3
    var_4 = [var_1, var_2, var_3]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_check_types_with_empty_expected_types. Retrieved 2/3 statements.


def test_case_0():
    var_0 = []
    var_1 = []



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_all_invariants_pass. Retrieved 1/5 statements.
# Partially parsed test_single_invariant_fails. Retrieved 1/5 statements.
# Partially parsed test_multiple_invariants_mixed. Retrieved 1/7 statements.
# Partially parsed test_multiple_invariants_fail. Retrieved 1/7 statements.


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
    var_0 = 'test'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_3]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 4/8 statements.
# Partially parsed test_check_types_with_invalid_type. Retrieved 4/9 statements.
# Partially parsed test_check_types_with_multiple_valid_types. Retrieved 4/8 statements.
# Partially parsed test_check_types_with_empty_iterable. Retrieved 1/5 statements.
# Partially parsed test_check_types_with_empty_expected_types. Retrieved 5/8 statements.
# Partially parsed test_check_types_with_custom_exception. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2.0
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3.0
    var_3 = [var_0, var_1, var_2]
    var_4 = []

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 6/12 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_data. Retrieved 6/11 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 2/8 statements.
# Partially parsed test_checked_pmap_constructor_with_empty_data. Retrieved 2/8 statements.
# Partially parsed test_checked_pmap_constructor_with_wrong_key_type. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_wrong_value_type. Retrieved 4/9 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = 1.5
    var_4 = 2.25
    var_5 = {var_1: var_3, var_2: var_4}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = 2.5
    var_4 = 2.25
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 2

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = {}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 'a'
    var_2 = 1.5
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = '1.5'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test__check_types_with_empty_iterable. Retrieved 1/3 statements.


def test_case_0():
    var_0 = []



# Parsed testcases at query #14
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
    var_6 = str(var_1)
    assert var_6 == ', invariant_errors=[], missing_fields=[]'

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
    var_9 = str(var_4)
    assert var_9 == ', invariant_errors=[error1, error2], missing_fields=[]'

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
    var_9 = str(var_4)
    assert var_9 == ', invariant_errors=[], missing_fields=[field1, field2]'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'error1'
    var_1 = (var_0,)
    var_2 = 'field1'
    var_3 = (var_2,)
    var_4 = {}
    var_5 = module_0.InvariantException(var_1, var_3, **var_4)
    var_6 = var_5.invariant_errors
    var_7 = bool(var_5.invariant_errors == ('error1',))
    assert var_7 is True
    var_8 = var_5.missing_fields
    var_9 = bool(var_5.missing_fields == ('field1',))
    assert var_9 is True
    var_10 = str(var_5)
    assert var_10 == ', invariant_errors=[error1], missing_fields=[field1]'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'error1'
    var_1 = lambda : var_0
    var_2 = 'error2'
    var_3 = lambda : var_2
    var_4 = (var_1, var_3)
    var_5 = {}
    var_6 = module_0.InvariantException(var_4, **var_5)
    var_7 = var_6.invariant_errors
    var_8 = bool(var_6.invariant_errors == ('error1', 'error2'))
    assert var_8 is True
    var_9 = var_6.missing_fields
    var_10 = bool(var_6.missing_fields == ())
    assert var_10 is True
    var_11 = str(var_6)
    assert var_11 == ', invariant_errors=[error1, error2], missing_fields=[]'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'error1'
    var_1 = 'error2'
    var_2 = lambda : var_1
    var_3 = (var_0, var_2)
    var_4 = {}
    var_5 = module_0.InvariantException(var_3, **var_4)
    var_6 = var_5.invariant_errors
    var_7 = bool(var_5.invariant_errors == ('error1', 'error2'))
    assert var_7 is True
    var_8 = var_5.missing_fields
    var_9 = bool(var_5.missing_fields == ())
    assert var_9 is True
    var_10 = str(var_5)
    assert var_10 == ', invariant_errors=[error1, error2], missing_fields=[]'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_store_invariants_with_valid_invariants. Retrieved 10/17 statements.
# Partially parsed test_store_invariants_with_multiple_invariants. Retrieved 12/24 statements.
# Partially parsed test_store_invariants_with_non_callable_invariant. Retrieved 4/8 statements.
# Partially parsed test_store_invariants_with_no_invariants. Retrieved 5/9 statements.
# Partially parsed test_store_invariants_with_inherited_invariants. Retrieved 8/17 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'wrapped_invariants'
    var_2 = 'invariant'
    var_3 = 'wrapped_invariants'
    var_4 = bool('wrapped_invariants' in var_0)
    assert var_4 is True
    var_5 = var_0[var_1]
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 0
    var_8 = var_0[var_1][var_7]
    var_9 = callable(var_8)
    var_10 = bool(var_9)
    assert var_10 is True
    var_11 = var_0[var_1][var_7]
    var_12 = var_11()
    var_13 = bool(var_12 == (True, 'data'))
    assert var_13 is True

def test_case_0():
    var_0 = {}
    var_1 = 'wrapped_invariants'
    var_2 = 'invariant'
    var_3 = 'wrapped_invariants'
    var_4 = bool('wrapped_invariants' in var_0)
    assert var_4 is True
    var_5 = var_0[var_1]
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = var_0[var_1]
    var_8 = 0
    var_9 = var_0[var_1][var_8]
    var_10 = var_9()
    var_11 = 1
    var_12 = var_0[var_1][var_11]
    var_13 = var_12()
    var_14 = bool(var_10 == (True, 'data1'))
    assert var_14 is True
    var_15 = bool(var_13 == (False, 'data2'))
    assert var_15 is True

def test_case_0():
    var_0 = 'not callable'
    var_1 = {}
    var_2 = 'wrapped_invariants'
    var_3 = 'invariant'
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = {}
    var_1 = 'wrapped_invariants'
    var_2 = 'invariant'
    var_3 = 'wrapped_invariants'
    var_4 = bool('wrapped_invariants' in var_0)
    assert var_4 is True
    var_5 = var_0[var_1]
    var_6 = len(var_5)
    assert var_6 == 0

def test_case_0():
    var_0 = {}
    var_1 = 'wrapped_invariants'
    var_2 = 'invariant'
    var_3 = 'wrapped_invariants'
    var_4 = bool('wrapped_invariants' in var_0)
    assert var_4 is True
    var_5 = var_0[var_1]
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 0
    var_8 = var_0[var_1][var_7]
    var_9 = var_8()
    var_10 = bool(var_9 == (True, 'base_data'))
    assert var_10 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_restore_pickle_creates_instance_with_factory_fields. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = set()



# Parsed testcases at query #17
#--------------------------

# Failed to parse test_checked_pvector_constructor_with_empty_initial.
# Partially parsed test_checked_pvector_constructor_with_list_initial. Retrieved 4/7 statements.
# Partially parsed test_checked_pvector_constructor_with_python_pvector_initial. Retrieved 5/8 statements.
# Partially parsed test_checked_pvector_constructor_with_invalid_type. Retrieved 5/9 statements.
# Partially parsed test_checked_pvector_constructor_with_invalid_invariant. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_3]

import pyrsistent._pvector as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.python_pvector(var_3)
    var_5 = [var_4]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = -1
    var_2 = -2
    var_3 = -3
    var_4 = [var_1, var_2, var_3]



# Parsed testcases at query #18
#--------------------------

# Failed to parse test_checked_pvector_constructor_with_empty_initial.
# Partially parsed test_checked_pvector_constructor_with_list_initial. Retrieved 4/7 statements.
# Partially parsed test_checked_pvector_constructor_with_python_pvector_initial. Retrieved 5/8 statements.
# Partially parsed test_checked_pvector_constructor_with_invalid_type. Retrieved 4/6 statements.
# Partially parsed test_checked_pvector_constructor_with_invalid_invariant. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_3]

import pyrsistent._pvector as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.python_pvector(var_3)
    var_5 = [var_4]

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_3]
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



# Parsed testcases at query #19
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_initial_data. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_size. Retrieved 6/12 statements.
# Partially parsed test_checked_pmap_constructor_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_invalid_invariant. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'one'
    var_3 = 'two'
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'one'
    var_3 = 'two'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 5

def test_case_0():
    var_0 = 'invalid'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 123
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = lambda k, v: (k == len(v), 'Key must equal length of value')
    var_1 = 1
    var_2 = 'invalid'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_invariants_are_callable. Retrieved 12/14 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'inv1'
    var_1 = 'inv2'
    var_2 = True
    var_3 = lambda : var_2
    var_4 = False
    var_5 = lambda : var_4
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = []
    var_8 = 'invariants'
    var_9 = 'inv1'
    var_10 = module_0.store_invariants(var_6, var_7, var_8, var_9)
    var_11 = var_6[var_8]



# Parsed testcases at query #21
#--------------------------

# Partially parsed test__checked_type_create_with_instance_of_cls. Retrieved 1/4 statements.
# Partially parsed test__checked_type_create_with_non_instance_and_no_checked_types. Retrieved 2/5 statements.
# Partially parsed test__checked_type_create_with_checked_type_in_list. Retrieved 4/9 statements.
# Partially parsed test__checked_type_create_with_checked_type_create_method. Retrieved 4/10 statements.
# Partially parsed test__checked_type_create_with_ignore_extra. Retrieved 5/11 statements.


def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = []
    var_1 = 'test_data'

def test_case_0():
    var_0 = 'module.CheckedType'
    var_1 = [var_0]
    var_2 = 'data1'
    var_3 = 'data2'
    var_4 = [var_2, var_3]

def test_case_0():
    var_0 = 'module.CheckedType'
    var_1 = [var_0]
    var_2 = 'data1'
    var_3 = 'data2'
    var_4 = [var_2, var_3]

def test_case_0():
    var_0 = 'module.CheckedType'
    var_1 = [var_0]
    var_2 = 'data1'
    var_3 = 'data2'
    var_4 = [var_2, var_3]
    var_5 = True



# Parsed testcases at query #22
#--------------------------




import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'invariant'
    var_1 = 123
    var_2 = {var_0: var_1}
    var_3 = ()
    var_4 = 'invariants'
    var_5 = 'invariant'
    var_6 = module_0.store_invariants(var_2, var_3, var_4, var_5)
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #23
#--------------------------

# Failed to parse test_is_preserved_evaluates_to_true.




# Parsed testcases at query #24
#--------------------------

# Partially parsed test_invariant_errors_with_invalid_invariant. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'test_elem'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 1/7 statements.
# Failed to parse test_checked_pmap_constructor_empty.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 'a'
    var_1 = 1.0
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = '1.0'
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



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_empty. Retrieved 1/7 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = '1'
    var_2 = 1.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = '1.0'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 5
    var_2 = 1
    var_3 = {var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_check_types_with_empty_expected_types. Retrieved 5/6 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = []



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_store_types_with_single_type. Retrieved 5/6 statements.
# Partially parsed test_store_types_with_multiple_types. Retrieved 5/6 statements.
# Partially parsed test_store_types_with_string_type. Retrieved 5/6 statements.
# Partially parsed test_store_types_with_base_class_type. Retrieved 3/7 statements.
# Partially parsed test_store_types_with_mixed_types. Retrieved 3/8 statements.
# Partially parsed test_store_types_with_invalid_type. Retrieved 5/7 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'types'
    var_3 = 'type'
    var_4 = module_0._store_types(var_0, var_1, var_2, var_3)
    var_5 = var_0[var_2]

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'types'
    var_3 = 'type'
    var_4 = module_0._store_types(var_0, var_1, var_2, var_3)
    var_5 = var_0[var_2]

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'types'
    var_3 = 'type'
    var_4 = module_0._store_types(var_0, var_1, var_2, var_3)
    var_5 = var_0[var_2]
    var_6 = bool(var_0[var_2] == ['int'])
    assert var_6 is True

def test_case_0():
    var_0 = {}
    var_1 = 'types'
    var_2 = 'type'
    var_3 = var_0[var_1]

def test_case_0():
    var_0 = 'type'
    var_1 = 'types'
    var_2 = 'type'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'types'
    var_3 = 'type'
    var_4 = module_0._store_types(var_0, var_1, var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #29
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_items. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_size. Retrieved 1/7 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_invariant. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'one'
    var_3 = 'two'
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'invalid'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 123
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = lambda k, v: (v == k * 2, 'Value must be twice the key')
    var_1 = 1
    var_2 = 3.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_checked_pset_constructor_with_valid_initial_elements. Retrieved 7/11 statements.
# Partially parsed test_checked_pset_constructor_with_invalid_type. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_invalid_invariant. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_pmap_input. Retrieved 11/15 statements.
# Partially parsed test_checked_pset_constructor_empty_initial. Retrieved 2/6 statements.


import pyrsistent._pset as module_0

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = [var_1, var_2, var_3]
    var_6 = module_0.pset(var_5)

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 'a'
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
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = True
    var_5 = True
    var_6 = True
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0.pmap(var_7)
    var_9 = [var_6, var_2, var_3]
    var_10 = module_1.pset(var_9)

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = module_0.pset()



# Parsed testcases at query #31
#--------------------------




import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = True
    var_2 = 'data1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'data2'
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = [var_4, var_7]
    var_9 = module_0._invariant_errors(var_0, var_8)
    var_10 = bool(var_9 == [])
    assert var_10 is True



# Parsed testcases at query #32
#--------------------------

# Failed to parse test_checked_type_create_with_checked_type_instance.




# Parsed testcases at query #33
#--------------------------




import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'source_name'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'dest'
    var_5 = 'source_name'
    var_6 = module_0._store_types(var_2, var_3, var_4, var_5)
    var_7 = bool(var_4 in var_2)
    assert var_7 is True



# Parsed testcases at query #34
#--------------------------

# Failed to parse test_checked_type_create_predicate_false.




# Parsed testcases at query #35
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'not_a_class_instance'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_checked_type_create_predicate_false. Retrieved 2/4 statements.


def test_case_0():
    var_0 = ()
    var_1 = {}



# Parsed testcases at query #37
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_valid_items. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_size. Retrieved 1/7 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 2/7 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_violation. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'one'
    var_3 = 'two'
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'a'
    var_1 = 'one'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = {var_0: var_0}
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = lambda k, v: (k == int(v), 'Key must equal int(value)')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_empty. Retrieved 1/7 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 5
    var_2 = 1
    var_3 = 2
    var_4 = {var_2: var_2, var_3: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 'a'
    var_2 = 1.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 'a'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')



# Parsed testcases at query #39
#--------------------------

# Failed to parse test_predicate_at_line_18.




# Parsed testcases at query #40
#--------------------------

# Failed to parse test_checked_type_instantiation.




# Parsed testcases at query #41
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_initial_data. Retrieved 6/12 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_initial_data. Retrieved 6/11 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 7/13 statements.
# Partially parsed test_checked_pmap_constructor_empty_initial_data. Retrieved 1/7 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = 1.5
    var_4 = 2.25
    var_5 = {var_1: var_3, var_2: var_4}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = 1.5
    var_4 = 3.25
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 5
    var_2 = 1
    var_3 = 2
    var_4 = 1.5
    var_5 = 2.25
    var_6 = {var_2: var_4, var_3: var_5}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_data. Retrieved 6/11 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_empty. Retrieved 1/7 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = 1.5
    var_4 = 2.25
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 5
    var_2 = 1
    var_3 = 2
    var_4 = {var_2: var_2, var_3: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 4/10 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = '1'
    var_2 = 1.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = '1.0'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 5
    var_2 = 1
    var_3 = {var_2: var_2}



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_initial_data. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_empty. Retrieved 1/7 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 'a'
    var_2 = 1.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = '1.0'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 5
    var_2 = 1
    var_3 = {var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'not_a_class_instance'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test__checked_type_create_with_instance_of_cls. Retrieved 1/4 statements.
# Partially parsed test__checked_type_create_with_non_instance_and_no_checked_types. Retrieved 2/5 statements.
# Partially parsed test__checked_type_create_with_checked_type_in_source_data. Retrieved 4/11 statements.
# Partially parsed test__checked_type_create_with_checked_type_not_in_source_data. Retrieved 4/11 statements.
# Partially parsed test__checked_type_create_with_ignore_extra. Retrieved 5/12 statements.


def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = []
    var_1 = 'test_data'

def test_case_0():
    var_0 = 'module.CheckedType'
    var_1 = [var_0]
    var_2 = 'data1'
    var_3 = 'data2'
    var_4 = [var_2, var_3]
    var_5 = bool(var_2)
    assert var_5 is True

def test_case_0():
    var_0 = 'module.CheckedType'
    var_1 = [var_0]
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = bool(var_2)
    assert var_5 is True

def test_case_0():
    var_0 = 'module.CheckedType'
    var_1 = [var_0]
    var_2 = 'data1'
    var_3 = 'data2'
    var_4 = [var_2, var_3]
    var_5 = True
    var_6 = bool(var_3)
    assert var_6 is True



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_no_errors. Retrieved 1/5 statements.
# Partially parsed test_single_error. Retrieved 1/5 statements.
# Partially parsed test_multiple_errors. Retrieved 1/7 statements.
# Partially parsed test_mixed_valid_invalid. Retrieved 1/7 statements.


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
    var_0 = 'test'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = lambda k, v: (k < v, 'Key must be less than value')
    var_1 = 2
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 5



# Parsed testcases at query #49
#--------------------------

# Failed to parse test_is_preserved_evaluates_to_true.




# Parsed testcases at query #50
#--------------------------

# Partially parsed test__checked_type_create_with_instance_of_cls. Retrieved 1/4 statements.
# Partially parsed test__checked_type_create_with_non_instance_and_no_checked_types. Retrieved 2/5 statements.
# Partially parsed test__checked_type_create_with_checked_type_in_list. Retrieved 4/10 statements.
# Partially parsed test__checked_type_create_with_instance_in_checked_types. Retrieved 2/9 statements.
# Partially parsed test__checked_type_create_with_ignore_extra. Retrieved 5/11 statements.


def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = []
    var_1 = 'test_data'

def test_case_0():
    var_0 = 'module.TestCheckedType'
    var_1 = [var_0]
    var_2 = 'data1'
    var_3 = 'data2'
    var_4 = [var_2, var_3]

def test_case_0():
    var_0 = 'module.TestCheckedType'
    var_1 = [var_0]
    var_2 = 'other_data'

def test_case_0():
    var_0 = 'module.TestCheckedType'
    var_1 = [var_0]
    var_2 = 'data1'
    var_3 = 'data2'
    var_4 = [var_2, var_3]
    var_5 = True



# Parsed testcases at query #51
#--------------------------

# Failed to parse test_wrap_invariant_with_single_bool_result.
# Failed to parse test_wrap_invariant_with_false_bool_result.
# Failed to parse test_wrap_invariant_with_multiple_results.
# Failed to parse test_wrap_invariant_with_all_true_results.
# Failed to parse test_wrap_invariant_with_all_false_results.




# Parsed testcases at query #52
#--------------------------

# Failed to parse test_maybe_parse_user_type_with_preserved_type.
# Failed to parse test_maybe_parse_user_type_with_non_iterable_type.
# Failed to parse test_maybe_parse_user_type_with_iterable_of_types.
# Failed to parse test_maybe_parse_user_type_with_nested_iterable.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'str'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(var_1 == ['str'])
    assert var_2 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.maybe_parse_user_type(var_0)



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_restore_pickle_creates_instance_with_factory_fields. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}



# Parsed testcases at query #54
#--------------------------

# Failed to parse test_wrap_invariant_with_non_bool_result.




# Parsed testcases at query #55
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_initial_data. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 4/10 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = '1'
    var_2 = 1.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = '1.0'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 5
    var_2 = 1
    var_3 = {var_2: var_2}



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_3]



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_serialize_default_format. Retrieved 4/8 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 6/10 statements.
# Partially parsed test_serialize_empty_set. Retrieved 1/5 statements.
# Partially parsed test_serialize_with_different_types. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = lambda fmt, x: str(x) if fmt == 'str' else x
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 'str'

def test_case_0():
    var_0 = set()

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = 'b'
    var_4 = [var_0, var_1, var_2, var_3]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_store_invariants_with_valid_invariants. Retrieved 8/16 statements.
# Partially parsed test_store_invariants_with_inherited_invariants. Retrieved 8/15 statements.
# Partially parsed test_store_invariants_with_multiple_inherited_invariants. Retrieved 6/18 statements.
# Partially parsed test_store_invariants_with_non_callable_invariant. Retrieved 4/8 statements.
# Partially parsed test_store_invariants_with_no_invariants. Retrieved 5/9 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'wrapped_invariants'
    var_2 = 'invariant1'
    var_3 = 'wrapped_invariants'
    var_4 = bool('wrapped_invariants' in var_0)
    assert var_4 is True
    var_5 = var_0[var_1]
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 0
    var_8 = var_0[var_1][var_7]
    var_9 = callable(var_8)
    var_10 = bool(var_9)
    assert var_10 is True

def test_case_0():
    var_0 = {}
    var_1 = 'wrapped_invariants'
    var_2 = 'invariant'
    var_3 = 'wrapped_invariants'
    var_4 = bool('wrapped_invariants' in var_0)
    assert var_4 is True
    var_5 = var_0[var_1]
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 0
    var_8 = var_0[var_1][var_7]
    var_9 = callable(var_8)
    var_10 = bool(var_9)
    assert var_10 is True

def test_case_0():
    var_0 = {}
    var_1 = 'wrapped_invariants'
    var_2 = 'invariant'
    var_3 = 'wrapped_invariants'
    var_4 = bool('wrapped_invariants' in var_0)
    assert var_4 is True
    var_5 = var_0[var_1]
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = var_0[var_1]

def test_case_0():
    var_0 = 'not_callable'
    var_1 = {}
    var_2 = 'wrapped_invariants'
    var_3 = 'invariant'
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = {}
    var_1 = 'wrapped_invariants'
    var_2 = 'invariant'
    var_3 = 'wrapped_invariants'
    var_4 = bool('wrapped_invariants' in var_0)
    assert var_4 is True
    var_5 = var_0[var_1]
    var_6 = len(var_5)
    assert var_6 == 0



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_checked_pset_constructor_with_valid_elements. Retrieved 7/11 statements.
# Partially parsed test_checked_pset_constructor_with_invalid_type. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_invalid_invariant. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_pmap_input. Retrieved 11/15 statements.
# Partially parsed test_checked_pset_constructor_empty. Retrieved 2/6 statements.
# Partially parsed test_checked_pset_constructor_with_mixed_valid_types. Retrieved 7/11 statements.


import pyrsistent._pset as module_0

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = [var_1, var_2, var_3]
    var_6 = module_0.pset(var_5)

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 'a'
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
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = True
    var_5 = True
    var_6 = True
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = module_0.pmap(var_7)
    var_9 = [var_6, var_2, var_3]
    var_10 = module_1.pset(var_9)

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = module_0.pset()

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2.5
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = [var_1, var_2, var_3]
    var_6 = module_0.pset(var_5)



# Parsed testcases at query #4
#--------------------------

# Failed to parse test_wrap_invariant_with_single_bool_result.
# Failed to parse test_wrap_invariant_with_multiple_bool_results.
# Failed to parse test_wrap_invariant_with_empty_list.
# Failed to parse test_wrap_invariant_with_all_true_results.
# Failed to parse test_wrap_invariant_with_all_false_results.




# Parsed testcases at query #5
#--------------------------

# Partially parsed test_checked_pset_constructor_with_empty_initial. Retrieved 1/6 statements.
# Partially parsed test_checked_pset_constructor_with_valid_elements. Retrieved 5/10 statements.
# Partially parsed test_checked_pset_constructor_with_invalid_type. Retrieved 4/8 statements.
# Partially parsed test_checked_pset_constructor_with_invalid_invariant. Retrieved 4/8 statements.
# Partially parsed test_checked_pset_constructor_with_pmap_initial. Retrieved 7/12 statements.


def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')

def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 1
    var_6 = 2
    var_7 = 3

def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = -1
    var_2 = -2
    var_3 = [var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 1
    var_2 = 2
    var_3 = True
    var_4 = True
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.pmap(var_5)
    var_7 = 1
    var_8 = 2



# Parsed testcases at query #6
#--------------------------

# Failed to parse test_checked_type_instantiation.




# Parsed testcases at query #7
#--------------------------

# Partially parsed test_store_types_with_single_type. Retrieved 5/6 statements.
# Partially parsed test_store_types_with_string_type. Retrieved 5/6 statements.
# Partially parsed test_store_types_with_iterable_types. Retrieved 5/6 statements.
# Partially parsed test_store_types_with_inherited_types. Retrieved 3/7 statements.
# Partially parsed test_store_types_with_multiple_inherited_types. Retrieved 3/9 statements.
# Partially parsed test_store_types_with_preserved_iterable_type. Retrieved 5/6 statements.
# Partially parsed test_store_types_with_invalid_type. Retrieved 5/7 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = ()
    var_2 = 'types'
    var_3 = 'type'
    var_4 = module_0._store_types(var_0, var_1, var_2, var_3)
    var_5 = var_0[var_2]

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = ()
    var_2 = 'types'
    var_3 = 'type'
    var_4 = module_0._store_types(var_0, var_1, var_2, var_3)
    var_5 = var_0[var_2]
    var_6 = bool(var_0[var_2] == ['int'])
    assert var_6 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = ()
    var_2 = 'types'
    var_3 = 'type'
    var_4 = module_0._store_types(var_0, var_1, var_2, var_3)
    var_5 = var_0[var_2]

def test_case_0():
    var_0 = {}
    var_1 = 'types'
    var_2 = 'type'
    var_3 = var_0[var_1]

def test_case_0():
    var_0 = {}
    var_1 = 'types'
    var_2 = 'type'
    var_3 = var_0[var_1]

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = ()
    var_2 = 'types'
    var_3 = 'type'
    var_4 = module_0._store_types(var_0, var_1, var_2, var_3)
    var_5 = var_0[var_2]

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = ()
    var_2 = 'types'
    var_3 = 'type'
    var_4 = module_0._store_types(var_0, var_1, var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #8
#--------------------------




import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = True
    var_1 = 'data1'
    var_2 = (var_0, var_1)
    var_3 = 'data2'
    var_4 = (var_0, var_3)
    var_5 = [var_2, var_4]
    var_6 = module_0._merge_invariant_results(var_5)
    var_7 = bool(var_6 == (True, ()))
    assert var_7 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = False
    var_1 = 'data1'
    var_2 = (var_0, var_1)
    var_3 = 'data2'
    var_4 = (var_0, var_3)
    var_5 = [var_2, var_4]
    var_6 = module_0._merge_invariant_results(var_5)
    var_7 = bool(var_6 == (False, ('data1', 'data2')))
    assert var_7 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = True
    var_1 = 'data1'
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'data2'
    var_5 = (var_3, var_4)
    var_6 = 'data3'
    var_7 = (var_0, var_6)
    var_8 = [var_2, var_5, var_7]
    var_9 = module_0._merge_invariant_results(var_8)
    var_10 = bool(var_9 == (False, ('data2',)))
    assert var_10 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0._merge_invariant_results(var_0)
    var_2 = bool(var_1 == (True, ()))
    assert var_2 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test__checked_type_create_with_instance_of_cls. Retrieved 1/4 statements.
# Partially parsed test__checked_type_create_with_non_instance_and_no_checked_types. Retrieved 5/8 statements.
# Partially parsed test__checked_type_create_with_checked_type_in_source_data. Retrieved 7/18 statements.
# Partially parsed test__checked_type_create_with_non_checked_type_in_source_data. Retrieved 5/14 statements.


def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = 'module.TestCheckedType'
    var_1 = [var_0]
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = 0

def test_case_0():
    var_0 = 'module.TestCheckedType'
    var_1 = [var_0]
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_checked_pvector_constructor_with_empty_initial. Retrieved 1/5 statements.
# Partially parsed test_checked_pvector_constructor_with_list_initial. Retrieved 5/9 statements.
# Partially parsed test_checked_pvector_constructor_with_pvector_initial. Retrieved 6/10 statements.
# Partially parsed test_checked_pvector_constructor_with_invalid_type. Retrieved 5/9 statements.
# Partially parsed test_checked_pvector_constructor_with_invalid_invariant. Retrieved 5/9 statements.


def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')

def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

import pyrsistent._pvector as module_0

def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.python_pvector(var_4)

def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 1
    var_2 = '2'
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 1
    var_2 = -2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_store_invariants_predicate. Retrieved 8/10 statements.


def test_case_0():
    var_0 = 'source'
    var_1 = lambda x: x
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'destination'
    var_5 = 'source'
    var_6 = var_2[var_5]
    var_7 = [var_6]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 3/9 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 'a'
    var_1 = 1.0
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
    var_2 = 2.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 5
    var_1 = 1
    var_2 = {var_1: var_1}



# Parsed testcases at query #13
#--------------------------

# Failed to parse test_maybe_parse_user_type_with_preserved_type.
# Failed to parse test_maybe_parse_user_type_with_non_iterable_type.
# Failed to parse test_maybe_parse_user_type_with_iterable_of_types.
# Failed to parse test_maybe_parse_user_type_with_nested_iterable.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(var_1 == ['int'])
    assert var_2 is True
    var_3 = 'str'
    var_4 = module_0.maybe_parse_user_type(var_3)
    var_5 = bool(var_4 == ['str'])
    assert var_5 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = None
    var_4 = module_0.maybe_parse_user_type(var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #14
#--------------------------




import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = []
    var_2 = module_0._invariant_errors(var_0, var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = True
    var_2 = 'error'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = [var_4]
    var_6 = module_0._invariant_errors(var_0, var_5)
    var_7 = bool(var_6 == [])
    assert var_7 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = False
    var_2 = 'error'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = [var_4]
    var_6 = module_0._invariant_errors(var_0, var_5)
    var_7 = bool(var_6 == ['error'])
    assert var_7 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = True
    var_2 = 'error1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'error2'
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = [var_4, var_7]
    var_9 = module_0._invariant_errors(var_0, var_8)
    var_10 = bool(var_9 == [])
    assert var_10 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = True
    var_2 = 'error1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = False
    var_6 = 'error2'
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = [var_4, var_8]
    var_10 = module_0._invariant_errors(var_0, var_9)
    var_11 = bool(var_10 == ['error2'])
    assert var_11 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = False
    var_2 = 'error1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'error2'
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = [var_4, var_7]
    var_9 = module_0._invariant_errors(var_0, var_8)
    var_10 = bool(var_9 == ['error1', 'error2'])
    assert var_10 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_restore_pickle_creates_instance_with_factory_fields. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = set()



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_store_invariants_with_valid_invariants. Retrieved 8/16 statements.
# Partially parsed test_store_invariants_with_inherited_invariants. Retrieved 8/16 statements.
# Partially parsed test_store_invariants_with_non_callable_invariant. Retrieved 4/8 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'wrapped_invariants'
    var_2 = 'invariant1'
    var_3 = 'wrapped_invariants'
    var_4 = bool('wrapped_invariants' in var_0)
    assert var_4 is True
    var_5 = var_0[var_1]
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 0
    var_8 = var_0[var_1][var_7]
    var_9 = callable(var_8)
    var_10 = bool(var_9)
    assert var_10 is True

def test_case_0():
    var_0 = {}
    var_1 = 'wrapped_invariants'
    var_2 = 'invariant1'
    var_3 = 'wrapped_invariants'
    var_4 = bool('wrapped_invariants' in var_0)
    assert var_4 is True
    var_5 = var_0[var_1]
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 0
    var_8 = var_0[var_1][var_7]
    var_9 = callable(var_8)
    var_10 = bool(var_9)
    assert var_10 is True

def test_case_0():
    var_0 = 'not callable'
    var_1 = {}
    var_2 = 'wrapped_invariants'
    var_3 = 'invariant1'
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_data. Retrieved 6/11 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 2/8 statements.
# Partially parsed test_checked_pmap_constructor_with_empty_initial. Retrieved 1/7 statements.
# Partially parsed test_checked_pmap_constructor_with_nested_checked_types. Retrieved 3/11 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = 1.5
    var_4 = 2.25
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 10

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0: var_0, var_1: var_1}
    var_3 = [var_2]



# Parsed testcases at query #18
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_initial_data. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_size. Retrieved 1/7 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_valid_invariant. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'one'
    var_3 = 'two'
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'invalid'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 123
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = lambda k, v: (k < v, 'Key must be less than value')
    var_1 = 2
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = lambda k, v: (k < v, 'Key must be less than value')
    var_1 = 1
    var_2 = 3
    var_3 = 2
    var_4 = 4
    var_5 = {var_1: var_3, var_2: var_4}



# Parsed testcases at query #19
#--------------------------




import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = []
    var_2 = module_0._invariant_errors(var_0, var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = 'positive'
    var_3 = lambda x: (x > var_1, var_2)
    var_4 = 10
    var_5 = 'less than 10'
    var_6 = lambda x: (x < var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0._invariant_errors(var_0, var_7)
    var_9 = bool(var_8 == [])
    assert var_9 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = -1
    var_1 = 0
    var_2 = 'non-negative'
    var_3 = lambda x: (x >= var_1, var_2)
    var_4 = [var_3]
    var_5 = module_0._invariant_errors(var_0, var_4)
    var_6 = bool(var_5 == ['non-negative'])
    assert var_6 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 15
    var_1 = 10
    var_2 = 'less than 10'
    var_3 = lambda x: (x < var_1, var_2)
    var_4 = 2
    var_5 = 0
    var_6 = 'even'
    var_7 = lambda x: (x % var_4 == var_5, var_6)
    var_8 = [var_3, var_7]
    var_9 = module_0._invariant_errors(var_0, var_8)
    var_10 = bool(var_9 == ['less than 10', 'even'])
    assert var_10 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 7
    var_1 = 0
    var_2 = 'positive'
    var_3 = lambda x: (x > var_1, var_2)
    var_4 = 5
    var_5 = 'less than 5'
    var_6 = lambda x: (x < var_4, var_5)
    var_7 = 2
    var_8 = 1
    var_9 = 'odd'
    var_10 = lambda x: (x % var_7 == var_8, var_9)
    var_11 = [var_3, var_6, var_10]
    var_12 = module_0._invariant_errors(var_0, var_11)
    var_13 = bool(var_12 == ['less than 5'])
    assert var_13 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_data. Retrieved 6/11 statements.
# Partially parsed test_checked_pmap_constructor_with_size. Retrieved 2/8 statements.
# Partially parsed test_checked_pmap_constructor_empty. Retrieved 1/7 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = 1.5
    var_4 = 2.25
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 5

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_checked_type_create_returns_source_data_when_input_is_instance_of_cls. Retrieved 1/4 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_data. Retrieved 6/11 statements.
# Partially parsed test_checked_pmap_constructor_with_wrong_types. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 4/10 statements.
# Failed to parse test_checked_pmap_constructor_empty.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = 1.5
    var_4 = 2.25
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = '1'
    var_1 = 2
    var_2 = 1.0
    var_3 = '2.0'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 5
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}



# Parsed testcases at query #23
#--------------------------

# Failed to parse test__checked_type_create_with_instance_of_cls.
# Partially parsed test__checked_type_create_with_non_instance_and_no_checked_types. Retrieved 1/6 statements.
# Partially parsed test__checked_type_create_with_checked_type_in_list. Retrieved 5/14 statements.
# Partially parsed test__checked_type_create_with_matching_type_in_list. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'test_data'

def test_case_0():
    var_0 = []
    var_1 = 'module.CheckedType'
    var_2 = [var_1]
    var_3 = 'data1'
    var_4 = 'data2'
    var_5 = [var_3, var_4]

def test_case_0():
    var_0 = []
    var_1 = 'module.CheckedType'
    var_2 = [var_1]
    var_3 = []
    var_4 = []



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_constructor_with_empty_initial. Retrieved 1/6 statements.
# Partially parsed test_constructor_with_list_initial. Retrieved 5/10 statements.
# Partially parsed test_constructor_with_pvector_initial. Retrieved 6/11 statements.
# Partially parsed test_constructor_with_invalid_type. Retrieved 3/7 statements.
# Partially parsed test_constructor_with_invalid_invariant. Retrieved 4/8 statements.


def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')

def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

import pyrsistent._pvector as module_0

def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.python_pvector(var_4)

def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 'invalid'
    var_2 = [var_1]
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = -1
    var_2 = -2
    var_3 = [var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #25
#--------------------------




import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'source_name'
    var_1 = 'not_callable'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'destination'
    var_5 = 'source_name'
    var_6 = module_0.store_invariants(var_2, var_3, var_4, var_5)
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_initial_data. Retrieved 6/12 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 2/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_initial_data. Retrieved 6/11 statements.
# Partially parsed test_checked_pmap_constructor_with_wrong_key_type. Retrieved 6/11 statements.
# Partially parsed test_checked_pmap_constructor_with_wrong_value_type. Retrieved 6/11 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = 1.5
    var_4 = 2.25
    var_5 = {var_1: var_3, var_2: var_4}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 10

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = 2.5
    var_4 = 2.25
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = '1'
    var_2 = 2
    var_3 = 1.5
    var_4 = 2.25
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = '1.5'
    var_4 = 2.25
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 3/9 statements.
# Failed to parse test_checked_pmap_constructor_empty_initial.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 'a'
    var_1 = 1.0
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
    var_2 = 2.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 5
    var_1 = 1
    var_2 = {var_1: var_1}



# Parsed testcases at query #28
#--------------------------




def test_case_0():
    var_0 = 'source_name'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'source_name'
    var_5 = bool(var_4 in var_2)
    assert var_5 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 3/9 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 'a'
    var_1 = 1.0
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
    var_2 = 2.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 5
    var_1 = 1
    var_2 = {var_1: var_1}



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_data. Retrieved 6/11 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 2/8 statements.
# Partially parsed test_checked_pmap_constructor_with_empty_data. Retrieved 2/8 statements.
# Partially parsed test_checked_pmap_constructor_with_type_mismatch. Retrieved 4/9 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = 1.5
    var_4 = 2.25
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 5

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = {}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 'a'
    var_2 = 1.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_empty. Retrieved 1/7 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = '1'
    var_2 = 2
    var_3 = 1.0
    var_4 = {var_1: var_3, var_2: var_2}
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = '1.0'
    var_4 = {var_1: var_3, var_2: var_2}
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_2, var_2: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 4
    var_2 = 1
    var_3 = 2
    var_4 = {var_2: var_2, var_3: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 4/7 statements.
# Partially parsed test_check_types_with_invalid_types. Retrieved 4/8 statements.
# Partially parsed test_check_types_with_empty_iterable. Retrieved 1/4 statements.
# Partially parsed test_check_types_with_multiple_valid_types. Retrieved 4/7 statements.
# Partially parsed test_check_types_with_custom_exception. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = 1
    var_1 = 2.0
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #33
#--------------------------

# Failed to parse test_constructor_with_empty_initial.
# Partially parsed test_constructor_with_list_initial. Retrieved 4/7 statements.
# Partially parsed test_constructor_with_python_pvector_initial. Retrieved 5/8 statements.
# Partially parsed test_constructor_with_checked_pvector_initial. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_3]

import pyrsistent._pvector as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.python_pvector(var_3)
    var_5 = [var_4]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_3]



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



