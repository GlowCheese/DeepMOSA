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



