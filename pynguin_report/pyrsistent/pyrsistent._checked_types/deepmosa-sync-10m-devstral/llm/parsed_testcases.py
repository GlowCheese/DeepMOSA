####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_all_invariants_pass. Retrieved 1/5 statements.
# Partially parsed test_single_invariant_fails. Retrieved 1/5 statements.
# Partially parsed test_multiple_invariants_mixed_results. Retrieved 1/7 statements.


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



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_store_invariants_with_valid_invariants. Retrieved 13/22 statements.
# Partially parsed test_store_invariants_with_non_callable_invariant. Retrieved 4/8 statements.
# Partially parsed test_store_invariants_with_multiple_inherited_invariants. Retrieved 7/20 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant1'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_0[var_1][var_5]
    var_7 = callable(var_6)
    var_8 = bool(var_7)
    assert var_8 is True
    var_9 = 'invariant2'
    var_10 = var_0[var_1]
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = var_0[var_1][var_5]
    var_13 = callable(var_12)
    var_14 = bool(var_13)
    assert var_14 is True

def test_case_0():
    var_0 = 'not_callable'
    var_1 = {}
    var_2 = 'invariants'
    var_3 = 'invariant'
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant1'
    var_3 = 'invariant2'
    var_4 = var_0[var_1]
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = var_0[var_1]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_checked_pvector_constructor_empty. Retrieved 1/6 statements.
# Partially parsed test_checked_pvector_constructor_with_list. Retrieved 5/10 statements.
# Partially parsed test_checked_pvector_constructor_with_pvector. Retrieved 6/11 statements.
# Partially parsed test_checked_pvector_constructor_type_error. Retrieved 5/9 statements.
# Partially parsed test_checked_pvector_constructor_invariant_error. Retrieved 5/9 statements.


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
    var_2 = 2
    var_3 = '3'
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 1
    var_2 = 2
    var_3 = -3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_initial_data. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 2/8 statements.
# Partially parsed test_checked_pmap_constructor_empty_initial_data. Retrieved 1/7 statements.


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
    var_1 = 10

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')



# Parsed testcases at query #5
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



# Parsed testcases at query #6
#--------------------------

# Failed to parse test_checked_pset_constructor_with_empty_initial.
# Partially parsed test_checked_pset_constructor_with_list_initial. Retrieved 5/10 statements.
# Partially parsed test_checked_pset_constructor_with_pmap_initial. Retrieved 9/14 statements.
# Partially parsed test_checked_pset_constructor_with_invalid_type. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_invalid_invariant. Retrieved 5/9 statements.


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
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = -1
    var_2 = -2
    var_3 = -3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_serialize_default_format. Retrieved 4/8 statements.
# Partially parsed test_serialize_with_custom_format. Retrieved 6/10 statements.
# Partially parsed test_serialize_empty_set. Retrieved 1/5 statements.


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



# Parsed testcases at query #8
#--------------------------

# Partially parsed test__checked_type_create_with_already_correct_type. Retrieved 1/4 statements.
# Partially parsed test__checked_type_create_without_checked_types. Retrieved 2/5 statements.
# Partially parsed test__checked_type_create_with_checked_type_in_list. Retrieved 4/10 statements.
# Partially parsed test__checked_type_create_with_matching_type_in_list. Retrieved 2/9 statements.


def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = []
    var_1 = 'test_data'

def test_case_0():
    var_0 = '__main__.CheckedType'
    var_1 = [var_0]
    var_2 = 'data1'
    var_3 = 'data2'
    var_4 = [var_2, var_3]

def test_case_0():
    var_0 = '__main__.CheckedType'
    var_1 = [var_0]
    var_2 = []
    var_3 = 'data'



# Parsed testcases at query #9
#--------------------------

# Failed to parse test_maybe_parse_user_type_preserved.




# Parsed testcases at query #10
#--------------------------




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
    var_8 = str(var_7)
    assert var_8 == ', invariant_errors=[error1, error2], missing_fields=[field1, field2]'



# Parsed testcases at query #11
#--------------------------

# Failed to parse test_maybe_parse_user_type_with_preserved_iterable_type.
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



# Parsed testcases at query #12
#--------------------------




import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'source_name'
    var_1 = 'not_callable'
    var_2 = {var_0: var_1}
    var_3 = ()
    var_4 = 'destination'
    var_5 = 'source_name'
    var_6 = module_0.store_invariants(var_2, var_3, var_4, var_5)
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_checkedpset_constructor_with_valid_elements. Retrieved 7/11 statements.
# Partially parsed test_checkedpset_constructor_with_invalid_type. Retrieved 5/9 statements.
# Partially parsed test_checkedpset_constructor_with_invalid_invariant. Retrieved 5/9 statements.
# Partially parsed test_checkedpset_constructor_with_pmap_input. Retrieved 11/15 statements.
# Partially parsed test_checkedpset_constructor_empty. Retrieved 2/6 statements.
# Partially parsed test_checkedpset_constructor_with_duplicate_elements. Retrieved 7/11 statements.


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
    var_2 = '2'
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
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_2, var_3]
    var_5 = [var_1, var_2, var_3]
    var_6 = module_0.pset(var_5)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 4/7 statements.
# Partially parsed test_check_types_with_invalid_types. Retrieved 4/8 statements.
# Partially parsed test_check_types_with_empty_iterable. Retrieved 1/4 statements.
# Partially parsed test_check_types_with_empty_expected_types. Retrieved 5/7 statements.
# Partially parsed test_check_types_with_string_type_name. Retrieved 6/8 statements.
# Partially parsed test_check_types_with_multiple_valid_types. Retrieved 4/7 statements.
# Partially parsed test_check_types_with_custom_exception. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = '2'
    var_2 = 3
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
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'builtins.int'
    var_5 = [var_4]

def test_case_0():
    var_0 = 1
    var_1 = 2.0
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = '2'
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_store_types_with_single_type_in_dict. Retrieved 3/5 statements.
# Partially parsed test_store_types_with_preserved_type_in_dict. Retrieved 3/5 statements.
# Partially parsed test_store_types_with_iterable_of_types_in_dict. Retrieved 3/6 statements.
# Partially parsed test_store_types_with_nested_iterable_of_types_in_dict. Retrieved 3/7 statements.
# Partially parsed test_store_types_with_multiple_bases. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'source'
    var_1 = []
    var_2 = 'destination'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'source'
    var_1 = 'str'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'destination'
    var_5 = module_0._store_types(var_2, var_3, var_4, var_0)
    var_6 = var_2['destination']
    var_7 = bool(var_2['destination'] == ['str'])
    assert var_7 is True

def test_case_0():
    var_0 = 'source'
    var_1 = []
    var_2 = 'destination'

def test_case_0():
    var_0 = 'source'
    var_1 = []
    var_2 = 'destination'

def test_case_0():
    var_0 = 'source'
    var_1 = []
    var_2 = 'destination'

def test_case_0():
    var_0 = {}
    var_1 = 'Base1'
    var_2 = ()
    var_3 = 'source'
    var_4 = 'Base2'
    var_5 = ()
    var_6 = 'destination'
    var_7 = var_0['destination']

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'source'
    var_1 = 123
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'destination'
    var_5 = 'source'
    var_6 = module_0._store_types(var_2, var_3, var_4, var_5)
    var_7 = bool(False)
    assert var_7 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'destination'
    var_3 = 'source'
    var_4 = module_0._store_types(var_0, var_1, var_2, var_3)
    var_5 = 'destination'
    var_6 = bool('destination' not in var_0)
    assert var_6 is True



# Parsed testcases at query #16
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



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_store_invariants_no_invariant_in_bases. Retrieved 3/7 statements.
# Partially parsed test_store_invariants_multiple_invariants_in_bases. Retrieved 8/14 statements.
# Partially parsed test_store_invariants_inherited_invariants. Retrieved 9/14 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = ()
    var_2 = 'dest'
    var_3 = 'src'
    var_4 = module_0.store_invariants(var_0, var_1, var_2, var_3)
    var_5 = bool(var_0 == {})
    assert var_5 is True

def test_case_0():
    var_0 = {}
    var_1 = 'dest'
    var_2 = 'src'
    var_3 = bool(var_0 == {})
    assert var_3 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'src'
    var_1 = True
    var_2 = 'data'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = {var_0: var_4}
    var_6 = ()
    var_7 = 'dest'
    var_8 = 'src'
    var_9 = module_0.store_invariants(var_5, var_6, var_7, var_8)
    var_10 = var_5[var_7]
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = 0
    var_13 = var_5[var_7][var_12]
    var_14 = callable(var_13)
    var_15 = bool(var_14)
    assert var_15 is True

def test_case_0():
    var_0 = {}
    var_1 = lambda x: (True, 'data1')
    var_2 = lambda x: (False, 'data2')
    var_3 = 'dest'
    var_4 = 'src'
    var_5 = var_0[var_3]
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = var_0[var_3]

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

def test_case_0():
    var_0 = {}
    var_1 = lambda x: (True, 'base_data')
    var_2 = 'dest'
    var_3 = 'src'
    var_4 = var_0[var_2]
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_0[var_2][var_6]
    var_8 = callable(var_7)
    var_9 = bool(var_8)
    assert var_9 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'src'
    var_1 = True
    var_2 = 'data1'
    var_3 = (var_1, var_2)
    var_4 = False
    var_5 = 'data2'
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = lambda x: var_7
    var_9 = {var_0: var_8}
    var_10 = ()
    var_11 = 'dest'
    var_12 = 'src'
    var_13 = module_0.store_invariants(var_9, var_10, var_11, var_12)
    var_14 = var_9[var_11][var_4]
    var_15 = None
    var_16 = var_14(var_15)
    var_17 = bool(var_16 == (False, ('data2',)))
    assert var_17 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_checked_pset_constructor_with_valid_elements. Retrieved 7/11 statements.
# Partially parsed test_checked_pset_constructor_with_pmap. Retrieved 11/15 statements.
# Partially parsed test_checked_pset_constructor_with_invalid_type. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_invalid_invariant. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_empty. Retrieved 2/6 statements.
# Partially parsed test_checked_pset_constructor_with_duplicate_elements. Retrieved 7/11 statements.


import pyrsistent._pset as module_0

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = [var_1, var_2, var_3]
    var_6 = module_0.pset(var_5)

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

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = module_0.pset()

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_2, var_3]
    var_5 = [var_1, var_2, var_3]
    var_6 = module_0.pset(var_5)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_store_invariants_with_valid_callable_invariants. Retrieved 8/15 statements.
# Partially parsed test_store_invariants_with_multiple_inherited_invariants. Retrieved 5/15 statements.
# Partially parsed test_store_invariants_with_non_callable_invariant. Retrieved 4/8 statements.


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
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 2

def test_case_0():
    var_0 = 'not_callable'
    var_1 = {}
    var_2 = 'wrapped_invariants'
    var_3 = 'invariant'
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #2
#--------------------------

# Failed to parse test_maybe_parse_user_type_preserved_type.
# Failed to parse test_maybe_parse_user_type_non_iterable_type.
# Failed to parse test_maybe_parse_user_type_iterable_of_types.
# Failed to parse test_maybe_parse_user_type_nested_iterable_of_types.


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



# Parsed testcases at query #3
#--------------------------

# Failed to parse test_checked_pset_new_with_empty_initial.
# Partially parsed test_checked_pset_new_with_iterable_initial. Retrieved 4/7 statements.
# Partially parsed test_checked_pset_new_with_duplicate_elements. Retrieved 4/7 statements.
# Partially parsed test_checked_pset_new_with_pmap_initial. Retrieved 11/14 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_3]
    var_5 = 1
    var_6 = 2
    var_7 = 3

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_1, var_2]
    var_4 = [var_3]
    var_5 = 1
    var_6 = 2
    var_7 = 3

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = True
    var_2 = (var_0, var_1)
    var_3 = 2
    var_4 = True
    var_5 = (var_3, var_4)
    var_6 = 3
    var_7 = True
    var_8 = (var_6, var_7)
    var_9 = [var_2, var_5, var_8]
    var_10 = module_0.pmap(var_9)
    var_11 = [var_10]
    var_12 = 1
    var_13 = 2
    var_14 = 3



# Parsed testcases at query #4
#--------------------------

# Partially parsed test__checked_type_create_with_instance_of_cls. Retrieved 1/4 statements.
# Partially parsed test__checked_type_create_with_non_instance_and_no_checked_types. Retrieved 2/5 statements.
# Partially parsed test__checked_type_create_with_checked_type_in_list. Retrieved 6/14 statements.
# Partially parsed test__checked_type_create_with_ignore_extra. Retrieved 9/17 statements.


def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = []
    var_1 = 'test_data'

def test_case_0():
    var_0 = 'module.CheckedType'
    var_1 = [var_0]
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = [var_4]
    var_6 = 0

def test_case_0():
    var_0 = 'module.CheckedType'
    var_1 = [var_0]
    var_2 = 'key'
    var_3 = 'extra'
    var_4 = 'value'
    var_5 = 'data'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = [var_6]
    var_8 = True
    var_9 = 0



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_items. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_size. Retrieved 1/7 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_violation. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 5

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



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_all_invariants_pass. Retrieved 1/5 statements.
# Partially parsed test_single_invariant_fails. Retrieved 1/5 statements.
# Partially parsed test_multiple_invariants_mixed. Retrieved 1/7 statements.
# Partially parsed test_multiple_invariants_all_fail. Retrieved 1/7 statements.


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



# Parsed testcases at query #7
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



# Parsed testcases at query #8
#--------------------------

# Failed to parse test_constructor_with_empty_initial.
# Partially parsed test_constructor_with_list_initial. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_python_pvector_initial. Retrieved 5/7 statements.


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



# Parsed testcases at query #9
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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_checked_pset_constructor_with_valid_elements. Retrieved 7/11 statements.
# Partially parsed test_checked_pset_constructor_with_pmap_input. Retrieved 11/15 statements.
# Partially parsed test_checked_pset_constructor_with_invalid_type. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_invalid_invariant. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_empty. Retrieved 2/6 statements.
# Partially parsed test_checked_pset_constructor_with_mixed_valid_invalid_elements. Retrieved 5/9 statements.


import pyrsistent._pset as module_0

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = [var_1, var_2, var_3]
    var_6 = module_0.pset(var_5)

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

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = -1
    var_2 = -2
    var_3 = -3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = module_0.pset()

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = -2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 3/9 statements.
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
    var_1 = 'not a float'
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



# Parsed testcases at query #12
#--------------------------

# Failed to parse test_wrap_invariant_with_single_bool_result.
# Failed to parse test_wrap_invariant_with_multiple_bool_results.
# Failed to parse test_wrap_invariant_with_all_true_results.
# Failed to parse test_wrap_invariant_with_empty_result_list.
# Partially parsed test_wrap_invariant_with_args_and_kwargs. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 4



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 4/7 statements.
# Partially parsed test_check_types_with_invalid_types. Retrieved 4/8 statements.
# Partially parsed test_check_types_with_empty_expected_types. Retrieved 5/7 statements.
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
    var_4 = "Type int can only be used with ('int',), not str"

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = []

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



# Parsed testcases at query #14
#--------------------------

# Failed to parse test_maybe_parse_user_type_with_non_iterable_type.




# Parsed testcases at query #15
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



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_invariants_must_be_callable. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'destination'
    var_1 = 'source'
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_initial_data. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 2/8 statements.
# Partially parsed test_checked_pmap_constructor_empty_initial_data. Retrieved 1/7 statements.


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
    var_1 = 10

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')



# Parsed testcases at query #19
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



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 1/7 statements.


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



# Parsed testcases at query #21
#--------------------------




import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = 'Positive'
    var_3 = lambda x: (x > var_1, var_2)
    var_4 = 10
    var_5 = 'Less than 10'
    var_6 = lambda x: (x < var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0._invariant_errors(var_0, var_7)
    var_9 = bool(var_8 == [])
    assert var_9 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_checked_type_create_with_checked_type_subclass. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_store_types_with_single_type. Retrieved 5/6 statements.
# Partially parsed test_store_types_with_string_type. Retrieved 5/6 statements.
# Partially parsed test_store_types_with_iterable_types. Retrieved 5/6 statements.
# Partially parsed test_store_types_with_mixed_types. Retrieved 6/7 statements.
# Partially parsed test_store_types_with_inherited_types. Retrieved 3/7 statements.
# Partially parsed test_store_types_with_multiple_inherited_types. Retrieved 3/9 statements.
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
    var_6 = bool(var_0[var_2] == ['str'])
    assert var_6 is True

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
    var_4 = 'str'
    var_5 = module_0._store_types(var_0, var_1, var_2, var_3)
    var_6 = var_0[var_2]

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
    var_1 = []
    var_2 = 'types'
    var_3 = 'type'
    var_4 = module_0._store_types(var_0, var_1, var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_checked_pvector_constructor_empty. Retrieved 1/6 statements.
# Partially parsed test_checked_pvector_constructor_with_list. Retrieved 5/10 statements.
# Partially parsed test_checked_pvector_constructor_with_pvector. Retrieved 6/11 statements.
# Partially parsed test_checked_pvector_constructor_type_error. Retrieved 5/9 statements.
# Partially parsed test_checked_pvector_constructor_invariant_error. Retrieved 5/9 statements.


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
    var_2 = 2
    var_3 = '3'
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 1
    var_2 = 2
    var_3 = -3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_checked_pset_constructor_with_valid_elements. Retrieved 7/11 statements.
# Partially parsed test_checked_pset_constructor_with_pmap. Retrieved 11/15 statements.
# Partially parsed test_checked_pset_constructor_with_invalid_type. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_invalid_invariant. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_empty. Retrieved 2/6 statements.
# Partially parsed test_checked_pset_constructor_with_mixed_valid_invalid_elements. Retrieved 5/9 statements.


import pyrsistent._pset as module_0

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = [var_1, var_2, var_3]
    var_6 = module_0.pset(var_5)

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

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = -1
    var_2 = -2
    var_3 = -3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = module_0.pset()

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = -2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_3]



# Parsed testcases at query #27
#--------------------------

# Partially parsed test__check_types_with_valid_types. Retrieved 4/8 statements.
# Partially parsed test__check_types_with_invalid_type. Retrieved 4/9 statements.
# Partially parsed test__check_types_with_multiple_valid_types. Retrieved 4/8 statements.
# Partially parsed test__check_types_with_custom_exception. Retrieved 4/11 statements.
# Partially parsed test__check_types_with_empty_iterable. Retrieved 1/5 statements.
# Partially parsed test__check_types_with_empty_expected_types. Retrieved 5/8 statements.


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
    var_4 = "Type TestClass can only be used with ('int',), not str"

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
    var_4 = "Type TestClass can only be used with ('int',), not str"

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = []



# Parsed testcases at query #28
#--------------------------

# Failed to parse test_checkedpset_constructor_with_empty_initial.
# Partially parsed test_checkedpset_constructor_with_list_initial. Retrieved 5/10 statements.
# Partially parsed test_checkedpset_constructor_with_pmap_initial. Retrieved 9/14 statements.
# Partially parsed test_checkedpset_constructor_with_invalid_type. Retrieved 3/7 statements.
# Partially parsed test_checkedpset_constructor_with_invalid_invariant. Retrieved 5/9 statements.


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
    var_3 = -3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #29
#--------------------------

# Failed to parse test_maybe_parse_user_type_preserved_type.




# Parsed testcases at query #30
#--------------------------

# Failed to parse test_is_preserved_predicate.




# Parsed testcases at query #31
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_isinstance_check. Retrieved 4/5 statements.


def test_case_0():
    var_0 = True
    var_1 = (var_0,)
    var_2 = 0
    var_3 = var_1[var_2]



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_data. Retrieved 6/11 statements.
# Partially parsed test_checked_pmap_constructor_with_empty_data. Retrieved 1/7 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 2/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 4/9 statements.


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

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 5

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



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_store_invariants_with_valid_callable_invariants. Retrieved 8/16 statements.
# Partially parsed test_store_invariants_with_inherited_invariants. Retrieved 7/18 statements.
# Partially parsed test_store_invariants_with_non_callable_invariant. Retrieved 4/8 statements.
# Partially parsed test_store_invariants_with_no_invariants. Retrieved 5/9 statements.
# Partially parsed test_store_invariants_with_multiple_inheritance. Retrieved 7/20 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant1'
    var_3 = 'invariants'
    var_4 = bool('invariants' in var_0)
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
    var_1 = 'invariants'
    var_2 = 'invariant1'
    var_3 = 'invariant2'
    var_4 = var_0[var_1]
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = var_0[var_1]

def test_case_0():
    var_0 = 'not_callable'
    var_1 = {}
    var_2 = 'invariants'
    var_3 = 'invariant1'
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'nonexistent'
    var_3 = 'invariants'
    var_4 = bool('invariants' in var_0)
    assert var_4 is True
    var_5 = var_0[var_1]
    var_6 = len(var_5)
    assert var_6 == 0

def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant1'
    var_3 = 'invariant2'
    var_4 = var_0[var_1]
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = var_0[var_1]



# Parsed testcases at query #35
#--------------------------

# Failed to parse test_checked_type_create_returns_source_data_when_isinstance_of_cls.




# Parsed testcases at query #36
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



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_restore_pickle_creates_instance_with_factory_fields. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = set()



# Parsed testcases at query #38
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



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_store_types_with_single_type. Retrieved 5/6 statements.
# Partially parsed test_store_types_with_string. Retrieved 5/6 statements.
# Partially parsed test_store_types_with_iterable. Retrieved 5/6 statements.
# Partially parsed test_store_types_with_nested_iterable. Retrieved 6/8 statements.
# Partially parsed test_store_types_with_inherited_type. Retrieved 3/7 statements.
# Partially parsed test_store_types_with_inherited_and_own_type. Retrieved 3/8 statements.
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
    var_6 = bool(var_0[var_2] == ['str'])
    assert var_6 is True

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
    var_4 = 'float'
    var_5 = module_0._store_types(var_0, var_1, var_2, var_3)
    var_6 = var_0[var_2]

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



# Parsed testcases at query #40
#--------------------------




import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = []
    var_2 = module_0._invariant_errors(var_0, var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_3]



# Parsed testcases at query #42
#--------------------------




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



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_restore_pickle_creates_instance_with_factory_fields. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = set()



# Parsed testcases at query #44
#--------------------------

# Failed to parse test_checked_type_constructor.




# Parsed testcases at query #45
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



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_type. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_invariant. Retrieved 6/11 statements.
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
    var_3 = 1.5
    var_4 = 2.25
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 10
    var_2 = 1
    var_3 = 2
    var_4 = {var_2: var_2, var_3: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 5/10 statements.
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
    var_1 = 2
    var_2 = 1
    var_3 = {var_2: var_2, var_1: var_1}



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_data. Retrieved 6/11 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 1/7 statements.
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
    var_0 = 5



# Parsed testcases at query #49
#--------------------------

# Failed to parse test_constructor_with_empty_initial.
# Partially parsed test_constructor_with_list_initial. Retrieved 4/7 statements.
# Partially parsed test_constructor_with_python_pvector_initial. Retrieved 5/8 statements.


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



# Parsed testcases at query #50
#--------------------------

# Failed to parse test_checked_type_create_predicate_false.




# Parsed testcases at query #51
#--------------------------

# Failed to parse test_maybe_parse_user_type_preserved.




# Parsed testcases at query #52
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_initial_data. Retrieved 4/11 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_type. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_empty_initial. Retrieved 1/7 statements.


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



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_store_types_with_single_type. Retrieved 5/6 statements.
# Partially parsed test_store_types_with_string_type. Retrieved 5/6 statements.
# Partially parsed test_store_types_with_iterable_types. Retrieved 5/6 statements.
# Partially parsed test_store_types_with_multiple_bases. Retrieved 3/9 statements.
# Partially parsed test_store_types_with_mixed_types. Retrieved 4/8 statements.


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

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
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
    var_1 = 'float'
    var_2 = 'types'
    var_3 = 'type'
    var_4 = var_0[var_2]

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'types'
    var_3 = 'type'
    var_4 = module_0._store_types(var_0, var_1, var_2, var_3)
    var_5 = bool(var_2 not in var_0)
    assert var_5 is True



# Parsed testcases at query #54
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_items. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_size. Retrieved 1/7 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 2/7 statements.
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
    var_0 = lambda k, v: (len(v) == k, 'Length mismatch')
    var_1 = 1
    var_2 = 'one'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 3/9 statements.
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



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_store_invariants_with_valid_callable_invariants. Retrieved 6/16 statements.
# Partially parsed test_store_invariants_with_non_callable_invariant. Retrieved 4/8 statements.
# Partially parsed test_store_invariants_with_multiple_inherited_invariants. Retrieved 5/15 statements.
# Partially parsed test_store_invariants_with_no_invariants. Retrieved 5/9 statements.
# Partially parsed test_store_invariants_with_wrapped_invariants. Retrieved 8/13 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant_1'
    var_3 = 'invariants'
    var_4 = bool('invariants' in var_0)
    assert var_4 is True
    var_5 = var_0[var_1]
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = var_0[var_1]

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = 'invariants'
    var_3 = 'invariant_1'

def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant_1'
    var_3 = 'invariants'
    var_4 = bool('invariants' in var_0)
    assert var_4 is True
    var_5 = var_0[var_1]
    var_6 = len(var_5)
    assert var_6 == 1

def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant_1'
    var_3 = 'invariants'
    var_4 = bool('invariants' in var_0)
    assert var_4 is True
    var_5 = var_0[var_1]
    var_6 = len(var_5)
    assert var_6 == 0

def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant_1'
    var_3 = 'invariants'
    var_4 = bool('invariants' in var_0)
    assert var_4 is True
    var_5 = var_0[var_1]
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 0
    var_8 = var_0[var_1][var_7]
    var_9 = var_8()
    var_10 = bool(var_9 == (False, ('data2',)))
    assert var_10 is True



# Parsed testcases at query #57
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



# Parsed testcases at query #58
#--------------------------

# Failed to parse test_invariant_result_not_merged_when_first_element_is_bool.




# Parsed testcases at query #59
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 3/9 statements.
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
    var_1 = 'not a float'
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



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_checked_pset_constructor_with_empty_initial. Retrieved 1/2 statements.
# Partially parsed test_checked_pset_constructor_with_list_initial. Retrieved 6/7 statements.
# Partially parsed test_checked_pset_constructor_with_pset_initial. Retrieved 7/8 statements.
# Partially parsed test_checked_pset_constructor_with_pmap_initial. Retrieved 10/11 statements.
# Partially parsed test_checked_pset_constructor_with_duplicate_elements. Retrieved 6/7 statements.
# Partially parsed test_checked_pset_constructor_with_invalid_type. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_invalid_invariant. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_valid_types_and_invariants. Retrieved 10/14 statements.


import pyrsistent._pset as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.pset()

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_3]
    var_5 = [var_0, var_1, var_2]
    var_6 = module_0.pset(var_5)

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pset(var_3)
    var_5 = [var_4]
    var_6 = [var_0, var_1, var_2]
    var_7 = module_0.pset(var_6)

import pyrsistent._pmap as module_0
import pyrsistent._pset as module_1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = True
    var_4 = True
    var_5 = True
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = [var_7]
    var_9 = [var_5, var_1, var_2]
    var_10 = module_1.pset(var_9)

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_1, var_2]
    var_4 = [var_3]
    var_5 = [var_0, var_1, var_2]
    var_6 = module_0.pset(var_5)

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 'a'
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = -3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = [var_1, var_2, var_3]
    var_6 = module_0.pset(var_5)
    var_7 = [var_1, var_2, var_3]
    var_8 = [var_1, var_2, var_3]
    var_9 = module_0.pset(var_8)



# Parsed testcases at query #61
#--------------------------

# Failed to parse test_maybe_parse_user_type_with_preserved_type.
# Failed to parse test_maybe_parse_user_type_with_non_iterable_type.
# Failed to parse test_maybe_parse_user_type_with_iterable_of_types.
# Failed to parse test_maybe_parse_user_type_with_nested_iterable_of_types.


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
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #62
#--------------------------

# Failed to parse test__checked_type_create_with_instance_of_cls.
# Partially parsed test__checked_type_create_with_non_instance_and_no_checked_types. Retrieved 1/5 statements.
# Partially parsed test__checked_type_create_with_checked_type_in_source_data. Retrieved 4/7 statements.
# Partially parsed test__checked_type_create_with_checked_type_not_in_source_data. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'some.module.CheckedType'
    var_1 = [var_0]
    var_2 = 'test'
    var_3 = 'data'
    var_4 = [var_2, var_3]

def test_case_0():
    var_0 = 'some.module.CheckedType'
    var_1 = [var_0]
    var_2 = 'test'
    var_3 = 'data'
    var_4 = [var_2, var_3]
    var_5 = bool(var_2)
    assert var_5 is True



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_checked_type_create_with_non_cls_instance. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'not an instance of TestClass'



# Parsed testcases at query #64
#--------------------------

# Failed to parse test_isinstance_predicate.




# Parsed testcases at query #65
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 4/8 statements.
# Partially parsed test_check_types_with_invalid_types. Retrieved 4/9 statements.
# Partially parsed test_check_types_with_multiple_valid_types. Retrieved 4/8 statements.
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



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_initial_data. Retrieved 4/10 statements.
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



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_merge_invariant_results_all_true. Retrieved 6/7 statements.
# Partially parsed test_merge_invariant_results_all_false. Retrieved 6/7 statements.
# Partially parsed test_merge_invariant_results_mixed. Retrieved 9/10 statements.
# Partially parsed test_merge_invariant_results_empty. Retrieved 1/2 statements.


def test_case_0():
    var_0 = True
    var_1 = 'data1'
    var_2 = (var_0, var_1)
    var_3 = 'data2'
    var_4 = (var_0, var_3)
    var_5 = [var_2, var_4]

def test_case_0():
    var_0 = False
    var_1 = 'data1'
    var_2 = (var_0, var_1)
    var_3 = 'data2'
    var_4 = (var_0, var_3)
    var_5 = [var_2, var_4]

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

def test_case_0():
    var_0 = []



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_initial_data. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_empty_initial. Retrieved 1/7 statements.


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
    var_1 = 2
    var_2 = 1
    var_3 = {var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_all_invariants_pass. Retrieved 1/5 statements.
# Partially parsed test_single_invariant_fails. Retrieved 1/5 statements.
# Partially parsed test_multiple_invariants_some_fail. Retrieved 1/7 statements.
# Partially parsed test_multiple_invariants_all_fail. Retrieved 1/8 statements.


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



# Parsed testcases at query #4
#--------------------------

# Failed to parse test_get_type_with_type_instance.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = module_0.get_type(var_0)
    var_2 = 'builtins.str'
    var_3 = module_0.get_type(var_2)
    var_4 = 'builtins.list'
    var_5 = module_0.get_type(var_4)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'collections.OrderedDict'
    var_1 = module_0.get_type(var_0)
    var_2 = 'collections'
    var_3 = 'OrderedDict'
    var_4 = [var_3]
    var_5 = __import__(var_2, fromlist=var_4)
    var_6 = var_5.OrderedDict
    var_7 = bool(var_1 == var_6)
    assert var_7 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_store_invariants_with_valid_invariants. Retrieved 13/22 statements.
# Partially parsed test_store_invariants_with_inherited_invariants. Retrieved 8/15 statements.
# Partially parsed test_store_invariants_with_multiple_inherited_invariants. Retrieved 6/18 statements.
# Partially parsed test_store_invariants_with_non_callable_invariant. Retrieved 4/8 statements.
# Partially parsed test_store_invariants_with_no_invariants. Retrieved 3/7 statements.
# Partially parsed test_store_invariants_with_wrapped_invariant_execution. Retrieved 6/11 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'wrapped_invariants'
    var_2 = 'invariant1'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_0[var_1][var_5]
    var_7 = callable(var_6)
    var_8 = bool(var_7)
    assert var_8 is True
    var_9 = 'invariant2'
    var_10 = var_0[var_1]
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = var_0[var_1][var_5]
    var_13 = callable(var_12)
    var_14 = bool(var_13)
    assert var_14 is True

def test_case_0():
    var_0 = {}
    var_1 = 'wrapped_invariants'
    var_2 = 'invariant'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_0[var_1][var_5]
    var_7 = callable(var_6)
    var_8 = bool(var_7)
    assert var_8 is True

def test_case_0():
    var_0 = {}
    var_1 = 'wrapped_invariants'
    var_2 = 'invariant'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = var_0[var_1]

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
    var_4 = bool('wrapped_invariants' not in var_0)
    assert var_4 is True

def test_case_0():
    var_0 = {}
    var_1 = 'wrapped_invariants'
    var_2 = 'invariant'
    var_3 = 0
    var_4 = var_0[var_1][var_3]
    var_5 = var_4()
    var_6 = bool(var_5 == (False, ('data2',)))
    assert var_6 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_checked_pset_constructor_with_empty_initial. Retrieved 1/6 statements.
# Partially parsed test_checked_pset_constructor_with_valid_elements. Retrieved 5/10 statements.
# Partially parsed test_checked_pset_constructor_with_pmap_initial. Retrieved 9/14 statements.
# Partially parsed test_checked_pset_constructor_with_invalid_type. Retrieved 3/7 statements.
# Partially parsed test_checked_pset_constructor_with_invalid_invariant. Retrieved 4/8 statements.


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



# Parsed testcases at query #7
#--------------------------

# Failed to parse test_checked_type_instantiation.




# Parsed testcases at query #8
#--------------------------

# Partially parsed test_constructor_with_empty_initial. Retrieved 1/6 statements.
# Partially parsed test_constructor_with_list_initial. Retrieved 5/10 statements.
# Partially parsed test_constructor_with_pvector_initial. Retrieved 6/11 statements.
# Partially parsed test_constructor_with_invalid_type. Retrieved 3/7 statements.
# Partially parsed test_constructor_with_invalid_invariant. Retrieved 3/7 statements.


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
    var_2 = [var_1]
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #9
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



# Parsed testcases at query #10
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_initial_data. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_size. Retrieved 1/7 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_valid_invariant. Retrieved 5/11 statements.


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
    var_0 = lambda k, v: (v == k * 2.0, 'Value must be twice the key')
    var_1 = 1
    var_2 = 3.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = lambda k, v: (v == k * 2.0, 'Value must be twice the key')
    var_1 = 1
    var_2 = 2
    var_3 = 4.0
    var_4 = {var_1: var_2, var_2: var_3}



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

# Partially parsed test__check_types_with_correct_types. Retrieved 4/7 statements.
# Partially parsed test__check_types_with_incorrect_types. Retrieved 4/8 statements.
# Partially parsed test__check_types_with_multiple_expected_types. Retrieved 4/7 statements.
# Partially parsed test__check_types_with_empty_iterable. Retrieved 1/4 statements.
# Partially parsed test__check_types_with_empty_expected_types. Retrieved 5/7 statements.
# Partially parsed test__check_types_with_string_type_names. Retrieved 6/8 statements.
# Partially parsed test__check_types_with_custom_exception. Retrieved 4/8 statements.


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
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 1
    var_1 = 2.0
    var_2 = 3
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
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'builtins.int'
    var_5 = [var_4]

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #13
#--------------------------

# Failed to parse test_maybe_parse_user_type_with_preserved_iterable_type.
# Failed to parse test_maybe_parse_user_type_with_non_iterable_type.
# Failed to parse test_maybe_parse_user_type_with_iterable_of_types.
# Failed to parse test_maybe_parse_user_type_with_nested_iterable_of_types.


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
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #14
#--------------------------

# Failed to parse test_predicate_at_line_18.




# Parsed testcases at query #15
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_initial_data. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_empty_initial_data. Retrieved 1/7 statements.


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



# Parsed testcases at query #16
#--------------------------

# Failed to parse test_constructor_with_empty_initial.
# Partially parsed test_constructor_with_list_initial. Retrieved 4/7 statements.
# Partially parsed test_constructor_with_python_pvector_initial. Retrieved 5/8 statements.


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



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_type. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_invariant. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_empty. Retrieved 1/7 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 'a'
    var_2 = 2
    var_3 = 1.0
    var_4 = {var_1: var_3, var_2: var_2}
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = 1.5
    var_4 = {var_1: var_3, var_2: var_2}
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 5
    var_2 = 1
    var_3 = 2
    var_4 = {var_2: var_2, var_3: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')



# Parsed testcases at query #18
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



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_restore_pickle_returns_instance. Retrieved 3/8 statements.
# Partially parsed test_restore_pickle_calls_create_with_data. Retrieved 4/9 statements.
# Partially parsed test_restore_pickle_passes_empty_factory_fields. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = None
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = None
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = set()



# Parsed testcases at query #20
#--------------------------

# Failed to parse test__checked_type_create_with_instance_of_cls.
# Partially parsed test__checked_type_create_with_non_instance_and_no_checked_types. Retrieved 1/6 statements.
# Partially parsed test__checked_type_create_with_checked_type_in_list. Retrieved 5/16 statements.
# Partially parsed test__checked_type_create_with_matching_type_in_list. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'test_data'

def test_case_0():
    var_0 = 'module.SubCheckedType'
    var_1 = [var_0]
    var_2 = 'data1'
    var_3 = 'data2'
    var_4 = [var_2, var_3]
    var_5 = True

def test_case_0():
    var_0 = []
    var_1 = 'other_data'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_store_invariants_with_non_callable_invariant. Retrieved 7/10 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'invariant'
    var_1 = 'not_callable'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'destination'
    var_5 = 'invariant'
    var_6 = module_0.store_invariants(var_2, var_3, var_4, var_5)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #23
#--------------------------

# Failed to parse test_predicate_at_line_18.




# Parsed testcases at query #24
#--------------------------

# Partially parsed test_checked_pmap_new_with_valid_initial_data. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_new_with_invalid_key_type. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_new_with_invalid_value_type. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_new_with_invalid_invariant. Retrieved 6/11 statements.
# Failed to parse test_checked_pmap_new_with_empty_initial_data.
# Partially parsed test_checked_pmap_new_with_predefined_size. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = '1.5'
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = bool(False)
    assert var_5 is True

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
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 16



# Parsed testcases at query #25
#--------------------------

# Failed to parse test_wrap_invariant_with_single_bool_result.
# Failed to parse test_wrap_invariant_with_multiple_bool_results.
# Failed to parse test_wrap_invariant_with_all_true_results.
# Failed to parse test_wrap_invariant_with_all_false_results.




# Parsed testcases at query #26
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 4/7 statements.
# Partially parsed test_check_types_with_invalid_types. Retrieved 4/8 statements.
# Partially parsed test_check_types_with_empty_iterable. Retrieved 1/4 statements.
# Partially parsed test_check_types_with_multiple_valid_types. Retrieved 4/7 statements.
# Partially parsed test_check_types_with_multiple_invalid_types. Retrieved 5/9 statements.
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
    var_4 = bool(False)
    assert var_4 is True
    var_5 = "Type list can only be used with ('int',), not str"

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
    var_2 = 3.0
    var_3 = None
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True
    var_6 = "Type list can only be used with ('int', 'float'), not str"

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True
    var_5 = "Type list can only be used with ('int',), not str"



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_checked_pmap_new_without_size_uses_evolver. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #28
#--------------------------

# Failed to parse test_checked_pset_constructor_with_empty_initial.
# Partially parsed test_checked_pset_constructor_with_list_initial. Retrieved 4/7 statements.
# Partially parsed test_checked_pset_constructor_with_pset_initial. Retrieved 5/8 statements.
# Partially parsed test_checked_pset_constructor_with_pmap_initial. Retrieved 8/11 statements.
# Partially parsed test_checked_pset_constructor_with_invalid_initial_type. Retrieved 1/3 statements.
# Partially parsed test_checked_pset_constructor_with_invalid_element_type. Retrieved 4/8 statements.
# Partially parsed test_checked_pset_constructor_with_invalid_invariant. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_3]
    var_5 = 1
    var_6 = 2
    var_7 = 3

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pset(var_3)
    var_5 = [var_4]
    var_6 = 1
    var_7 = 2
    var_8 = 3

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = True
    var_4 = True
    var_5 = True
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = [var_7]
    var_9 = 1
    var_10 = 2
    var_11 = 3

def test_case_0():
    var_0 = 'invalid'
    var_1 = [var_0]
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'invalid'
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = lambda x: (x >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = -3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test__invariant_errors_returns_empty_list_when_all_invariants_valid. Retrieved 7/9 statements.


def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = (var_1, var_2)
    var_6 = lambda x: var_5
    var_7 = [var_4, var_6]



# Parsed testcases at query #30
#--------------------------

# Failed to parse test_is_preserved_evaluates_to_true.




# Parsed testcases at query #31
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 2/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_invariant. Retrieved 4/9 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 5

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



# Parsed testcases at query #32
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
    var_1 = 5
    var_2 = 1
    var_3 = 2
    var_4 = {var_2: var_2, var_3: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_restore_pickle_returns_instance_with_factory_fields. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = set()



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_data. Retrieved 6/11 statements.
# Partially parsed test_checked_pmap_constructor_with_wrong_types. Retrieved 6/11 statements.
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
    var_1 = 'a'
    var_2 = 2
    var_3 = 1.0
    var_4 = 'b'
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



# Parsed testcases at query #35
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
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 5

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



# Parsed testcases at query #36
#--------------------------

# Failed to parse test_checked_type_create_with_checked_type_subclass.




# Parsed testcases at query #37
#--------------------------

# Partially parsed test_checked_pmap_new_without_size_uses_evolver. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_3]



# Parsed testcases at query #39
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
    var_0 = 5
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
    var_0 = 5
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

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 5
    var_1 = False
    var_2 = 'error1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = True
    var_6 = 'error2'
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = 'error3'
    var_10 = (var_1, var_9)
    var_11 = lambda x: var_10
    var_12 = [var_4, var_8, var_11]
    var_13 = module_0._invariant_errors(var_0, var_12)
    var_14 = bool(var_13 == ['error1', 'error3'])
    assert var_14 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_merge_invariant_results_with_false_verdict. Retrieved 7/8 statements.


def test_case_0():
    var_0 = False
    var_1 = 'data1'
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'data2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_type. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_invariant. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_empty. Retrieved 1/7 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 'a'
    var_2 = 2
    var_3 = 1.0
    var_4 = {var_1: var_3, var_2: var_2}
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = 1.5
    var_4 = {var_1: var_3, var_2: var_2}
    var_5 = bool(False)
    assert var_5 is True

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

# Failed to parse test_is_preserved_predicate.




# Parsed testcases at query #44
#--------------------------

# Partially parsed test_checked_type_create_predicate_false. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #45
#--------------------------

# Failed to parse test_checked_type_constructor.




# Parsed testcases at query #46
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



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 6/12 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_data. Retrieved 6/11 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 7/13 statements.
# Partially parsed test_checked_pmap_constructor_with_empty_data. Retrieved 1/7 statements.
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
    var_1 = 1
    var_2 = 3
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
    var_4 = 1.5
    var_5 = 2.25
    var_6 = {var_2: var_4, var_3: var_5}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 'a'
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



# Parsed testcases at query #48
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
    var_0 = {}
    var_1 = 'types'
    var_2 = 'type'
    var_3 = var_0[var_1]

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'types'
    var_3 = 'type'
    var_4 = module_0._store_types(var_0, var_1, var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_3]



# Parsed testcases at query #50
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



# Parsed testcases at query #51
#--------------------------




def test_case_0():
    var_0 = 'source_name'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'source_name'
    var_5 = bool(var_4 in var_2)
    assert var_5 is True



# Parsed testcases at query #52
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



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_checked_type_create_returns_source_data_when_input_is_instance_of_cls. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 4/7 statements.
# Partially parsed test_check_types_with_invalid_types. Retrieved 4/8 statements.
# Partially parsed test_check_types_with_empty_iterable. Retrieved 1/4 statements.
# Partially parsed test_check_types_with_multiple_valid_types. Retrieved 4/7 statements.
# Partially parsed test_check_types_with_string_type_name. Retrieved 6/8 statements.


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
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = 1
    var_1 = 2.0
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'builtins.int'
    var_5 = [var_4]



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_checked_pmap_new_without_size. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #56
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



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_data. Retrieved 6/11 statements.
# Partially parsed test_checked_pmap_constructor_with_empty_data. Retrieved 2/8 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 2/8 statements.


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
    var_1 = {}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 5



# Parsed testcases at query #58
#--------------------------

# Partially parsed test__check_types_with_empty_iterable. Retrieved 1/3 statements.


def test_case_0():
    var_0 = []



# Parsed testcases at query #59
#--------------------------

# Failed to parse test_checked_type_instantiation.




# Parsed testcases at query #60
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



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 3/9 statements.
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
    var_1 = 1
    var_2 = {var_1: var_1}



# Parsed testcases at query #62
#--------------------------

# Failed to parse test_predicate_at_line_18.




# Parsed testcases at query #63
#--------------------------

# Failed to parse test_constructor_empty_initial.
# Partially parsed test_constructor_with_valid_initial. Retrieved 5/11 statements.
# Partially parsed test_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_invalid_value_type. Retrieved 2/7 statements.
# Partially parsed test_constructor_with_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_size_parameter. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'one'
    var_3 = 'two'
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = '1'
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
    var_0 = lambda k, v: (k < v, 'Key must be less than value')
    var_1 = 1
    var_2 = 0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 10



# Parsed testcases at query #64
#--------------------------

# Partially parsed test__checked_type_create_with_instance_of_cls. Retrieved 1/4 statements.
# Partially parsed test__checked_type_create_with_non_instance_and_no_checked_types. Retrieved 2/5 statements.
# Partially parsed test__checked_type_create_with_checked_type_in_list. Retrieved 4/10 statements.
# Partially parsed test__checked_type_create_with_checked_type_and_matching_instance. Retrieved 2/9 statements.
# Partially parsed test__checked_type_create_with_ignore_extra. Retrieved 5/11 statements.


def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = []
    var_1 = 'test_data'

def test_case_0():
    var_0 = '__main__.CheckedType'
    var_1 = [var_0]
    var_2 = 'data1'
    var_3 = 'data2'
    var_4 = [var_2, var_3]

def test_case_0():
    var_0 = '__main__.CheckedType'
    var_1 = [var_0]
    var_2 = []
    var_3 = 'other_data'

def test_case_0():
    var_0 = '__main__.CheckedType'
    var_1 = [var_0]
    var_2 = 'data1'
    var_3 = 'data2'
    var_4 = [var_2, var_3]
    var_5 = True



# Parsed testcases at query #65
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
    var_7 = bool(var_2[var_4])
    assert var_7 is True



# Parsed testcases at query #66
#--------------------------

# Failed to parse test_checked_type_create_predicate_false.




# Parsed testcases at query #67
#--------------------------

# Failed to parse test_constructor_with_empty_initial.
# Partially parsed test_constructor_with_list_initial. Retrieved 4/7 statements.
# Partially parsed test_constructor_with_python_pvector_initial. Retrieved 5/8 statements.


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



# Parsed testcases at query #68
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



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_check_types_with_empty_expected_types. Retrieved 5/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = []



# Parsed testcases at query #70
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



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_merge_invariant_results_all_true. Retrieved 6/7 statements.
# Partially parsed test_merge_invariant_results_all_false. Retrieved 6/7 statements.
# Partially parsed test_merge_invariant_results_mixed. Retrieved 9/10 statements.
# Partially parsed test_merge_invariant_results_empty. Retrieved 1/2 statements.


def test_case_0():
    var_0 = True
    var_1 = 'data1'
    var_2 = (var_0, var_1)
    var_3 = 'data2'
    var_4 = (var_0, var_3)
    var_5 = [var_2, var_4]

def test_case_0():
    var_0 = False
    var_1 = 'data1'
    var_2 = (var_0, var_1)
    var_3 = 'data2'
    var_4 = (var_0, var_3)
    var_5 = [var_2, var_4]

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

def test_case_0():
    var_0 = []



# Parsed testcases at query #72
#--------------------------

# Failed to parse test_is_preserved_evaluates_to_true.




# Parsed testcases at query #73
#--------------------------

# Partially parsed test__checked_type_create_with_instance_of_cls. Retrieved 1/4 statements.
# Partially parsed test__checked_type_create_with_non_checked_type. Retrieved 5/8 statements.
# Partially parsed test__checked_type_create_with_checked_type. Retrieved 5/11 statements.
# Partially parsed test__checked_type_create_with_mixed_checked_and_non_checked_types. Retrieved 5/15 statements.
# Partially parsed test__checked_type_create_with_ignore_extra. Retrieved 5/11 statements.


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
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]

def test_case_0():
    var_0 = '__main__.CheckedType'
    var_1 = [var_0]
    var_2 = []
    var_3 = 'a'
    var_4 = 'b'
    var_5 = []
    var_6 = 'created_a'
    var_7 = 'created_b'

def test_case_0():
    var_0 = '__main__.CheckedType'
    var_1 = [var_0]
    var_2 = 'a'
    var_3 = 'b'
    var_4 = [var_2, var_3]
    var_5 = True



# Parsed testcases at query #74
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



# Parsed testcases at query #75
#--------------------------

# Failed to parse test_is_preserved_evaluates_to_true.




# Parsed testcases at query #76
#--------------------------

# Partially parsed test_checked_pset_constructor_with_empty_initial. Retrieved 1/6 statements.
# Partially parsed test_checked_pset_constructor_with_valid_elements. Retrieved 5/10 statements.
# Partially parsed test_checked_pset_constructor_with_pmap_initial. Retrieved 9/14 statements.
# Partially parsed test_checked_pset_constructor_with_invalid_type. Retrieved 3/7 statements.
# Partially parsed test_checked_pset_constructor_with_invalid_invariant. Retrieved 5/9 statements.


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
    var_3 = -3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #77
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_initial_data. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_size. Retrieved 2/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_valid_invariant. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = {}
    var_1 = 5

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
    var_0 = lambda k, v: (k < v, 'Key must be less than value')
    var_1 = 1
    var_2 = 3
    var_3 = 2
    var_4 = 4
    var_5 = {var_1: var_3, var_2: var_4}



# Parsed testcases at query #78
#--------------------------

# Failed to parse test_checked_type_instantiation.




