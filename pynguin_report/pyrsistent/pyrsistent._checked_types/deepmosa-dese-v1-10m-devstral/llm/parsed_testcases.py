####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_maybe_parse_user_type_with_preserved_type.
# Failed to parse test_maybe_parse_user_type_with_non_iterable_type.
# Failed to parse test_maybe_parse_user_type_with_iterable_of_types.
# Failed to parse test_maybe_parse_user_type_with_nested_iterable_of_types.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'str'
    var_1 = module_0.maybe_parse_user_type(var_0)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.maybe_parse_user_type(var_0)



# Parsed testcases at query #2
#--------------------------




import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = []
    var_2 = module_0._invariant_errors(var_0, var_1)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = True
    var_1 = 'error1'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = 'error2'
    var_5 = (var_0, var_4)
    var_6 = lambda x: var_5
    var_7 = [var_3, var_6]
    var_8 = 'test'
    var_9 = module_0._invariant_errors(var_8, var_7)

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

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = True
    var_1 = 'error1'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = False
    var_5 = 'error2'
    var_6 = (var_4, var_5)
    var_7 = lambda x: var_6
    var_8 = 'error3'
    var_9 = (var_0, var_8)
    var_10 = lambda x: var_9
    var_11 = [var_3, var_7, var_10]
    var_12 = 'test'
    var_13 = module_0._invariant_errors(var_12, var_11)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test__invariant_errors_all_valid. Retrieved 1/5 statements.
# Partially parsed test__invariant_errors_some_invalid. Retrieved 1/5 statements.
# Partially parsed test__invariant_errors_multiple_invalid. Retrieved 1/7 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 5
    var_1 = []
    var_2 = module_0._invariant_errors(var_0, var_1)

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 5



# Parsed testcases at query #4
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

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = '1.0'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 5
    var_2 = 1
    var_3 = {var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 6/12 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_data. Retrieved 6/11 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 7/13 statements.
# Partially parsed test_checked_pmap_constructor_with_empty_initial. Retrieved 1/7 statements.


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



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 6/12 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 4/10 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = 1.5
    var_4 = 2.25
    var_5 = {var_1: var_3, var_2: var_4}

def test_case_0():
    var_0 = 'a'
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = '1.5'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 5
    var_1 = 1
    var_2 = 1.5
    var_3 = {var_1: var_2}



# Parsed testcases at query #7
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



# Parsed testcases at query #8
#--------------------------




import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = module_0.CheckedType()



# Parsed testcases at query #9
#--------------------------

# Failed to parse test__checked_type_create_with_instance_of_cls.
# Partially parsed test__checked_type_create_with_non_instance_of_cls. Retrieved 2/5 statements.
# Partially parsed test__checked_type_create_with_checked_type_in_source_data. Retrieved 3/12 statements.
# Partially parsed test__checked_type_create_with_ignore_extra. Retrieved 6/11 statements.


def test_case_0():
    var_0 = []
    var_1 = 'data'

def test_case_0():
    var_0 = 'TestCheckedType'
    var_1 = [var_0]
    var_2 = 'other_data'
    var_3 = 0

def test_case_0():
    var_0 = 'TestCheckedType'
    var_1 = [var_0]
    var_2 = 'extra'
    var_3 = 'data'
    var_4 = {var_2: var_3}
    var_5 = [var_4]
    var_6 = True



# Parsed testcases at query #10
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

def test_case_0():
    var_0 = 1
    var_1 = '1.0'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 5
    var_1 = 1
    var_2 = {var_1: var_1}



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_store_types_with_single_type. Retrieved 5/6 statements.
# Partially parsed test_store_types_with_string. Retrieved 5/6 statements.
# Partially parsed test_store_types_with_preserved_iterable. Retrieved 5/6 statements.
# Partially parsed test_store_types_with_iterable_of_types. Retrieved 5/6 statements.
# Partially parsed test_store_types_with_invalid_type. Retrieved 5/7 statements.
# Partially parsed test_store_types_with_base_classes. Retrieved 3/9 statements.
# Partially parsed test_store_types_with_mixed_sources. Retrieved 3/8 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'types'
    var_3 = 'type'
    var_4 = module_0._store_types(var_0, var_1, var_2, var_3)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'types'
    var_3 = 'type'
    var_4 = module_0._store_types(var_0, var_1, var_2, var_3)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'types'
    var_3 = 'type'
    var_4 = module_0._store_types(var_0, var_1, var_2, var_3)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'types'
    var_3 = 'type'
    var_4 = module_0._store_types(var_0, var_1, var_2, var_3)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'types'
    var_3 = 'type'
    var_4 = module_0._store_types(var_0, var_1, var_2, var_3)

def test_case_0():
    var_0 = {}
    var_1 = 'types'
    var_2 = 'type'

def test_case_0():
    var_0 = 'type'
    var_1 = 'types'
    var_2 = 'type'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_checked_pset_constructor_with_empty_initial. Retrieved 2/3 statements.
# Partially parsed test_checked_pset_constructor_with_list_initial. Retrieved 5/10 statements.
# Partially parsed test_checked_pset_constructor_with_pmap_initial. Retrieved 9/14 statements.
# Partially parsed test_checked_pset_constructor_with_invalid_type. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_invalid_invariant. Retrieved 5/9 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = module_0.CheckedPSet()
    var_1 = len(var_0)
    assert var_1 == 0

def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

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

def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 1
    var_2 = 2
    var_3 = '3'
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 1
    var_2 = 2
    var_3 = -3
    var_4 = [var_1, var_2, var_3]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_checkedpset_constructor_with_empty_initial. Retrieved 2/3 statements.
# Partially parsed test_checkedpset_constructor_with_list_initial. Retrieved 5/10 statements.
# Partially parsed test_checkedpset_constructor_with_pmap_initial. Retrieved 9/14 statements.
# Partially parsed test_checkedpset_constructor_with_invalid_type. Retrieved 3/7 statements.
# Partially parsed test_checkedpset_constructor_with_invalid_invariant. Retrieved 5/9 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = module_0.CheckedPSet()
    var_1 = len(var_0)
    assert var_1 == 0

def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

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

def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 'invalid'
    var_2 = [var_1]

def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = -1
    var_2 = -2
    var_3 = -3
    var_4 = [var_1, var_2, var_3]



# Parsed testcases at query #14
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_initial_data. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_size. Retrieved 1/7 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant. Retrieved 6/12 statements.
# Partially parsed test_checked_pmap_constructor_with_invariant_violation. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'one'
    var_3 = 'two'
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'invalid_key'
    var_1 = 'value'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 123
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (k < v, 'Key must be less than value')
    var_1 = 1
    var_2 = 3
    var_3 = 2
    var_4 = 4
    var_5 = {var_1: var_3, var_2: var_4}

def test_case_0():
    var_0 = lambda k, v: (k < v, 'Key must be less than value')
    var_1 = 1
    var_2 = 3
    var_3 = 2
    var_4 = {var_1: var_3, var_2: var_3}



# Parsed testcases at query #15
#--------------------------

# Failed to parse test_maybe_parse_user_type_with_non_iterable_type.




# Parsed testcases at query #16
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 8/14 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 1/7 statements.


def test_case_0():
    var_0 = lambda k, v: (len(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = 'a'
    var_5 = 'bb'
    var_6 = 'ccc'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}

def test_case_0():
    var_0 = 'a'
    var_1 = 'value'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 123
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (len(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 'abc'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 10



# Parsed testcases at query #17
#--------------------------

# Failed to parse test_maybe_parse_user_type_with_non_iterable_type.




# Parsed testcases at query #18
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 3/9 statements.


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

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = '1.0'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 5
    var_1 = 1
    var_2 = {var_1: var_1}



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_merge_invariant_results_all_true. Retrieved 8/9 statements.
# Partially parsed test_merge_invariant_results_all_false. Retrieved 8/9 statements.
# Partially parsed test_merge_invariant_results_mixed. Retrieved 9/10 statements.
# Partially parsed test_merge_invariant_results_empty. Retrieved 1/2 statements.


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
    var_4 = 'data2'
    var_5 = (var_3, var_4)
    var_6 = 'data3'
    var_7 = (var_0, var_6)
    var_8 = [var_2, var_5, var_7]

def test_case_0():
    var_0 = []



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 4/7 statements.
# Partially parsed test_check_types_with_invalid_type. Retrieved 4/8 statements.
# Partially parsed test_check_types_with_multiple_valid_types. Retrieved 4/7 statements.
# Partially parsed test_check_types_with_empty_iterable. Retrieved 1/4 statements.
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
    var_0 = 1
    var_1 = 2.0
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_constructor_with_valid_initial_data. Retrieved 4/10 statements.
# Partially parsed test_constructor_with_invalid_initial_data. Retrieved 6/11 statements.
# Partially parsed test_constructor_with_size_parameter. Retrieved 2/8 statements.
# Partially parsed test_constructor_with_empty_initial_data. Retrieved 1/7 statements.
# Partially parsed test_constructor_with_wrong_key_type. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_wrong_value_type. Retrieved 4/9 statements.


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

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 5

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 'a'
    var_2 = 1.0
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 'a'
    var_3 = {var_1: var_2}



# Parsed testcases at query #22
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
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (k < v, 'Key must be less than value')
    var_1 = 2
    var_2 = 1
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = lambda k, v: (k < v, 'Key must be less than value')
    var_1 = 1
    var_2 = 3
    var_3 = 2
    var_4 = 4
    var_5 = {var_1: var_3, var_2: var_4}



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_checked_pset_constructor_empty. Retrieved 1/6 statements.
# Partially parsed test_checked_pset_constructor_with_valid_elements. Retrieved 5/10 statements.
# Partially parsed test_checked_pset_constructor_with_invalid_type. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_with_invalid_invariant. Retrieved 5/9 statements.
# Partially parsed test_checked_pset_constructor_from_pmap. Retrieved 9/14 statements.


def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')

def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 1
    var_2 = 2
    var_3 = '3'
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 1
    var_2 = 2
    var_3 = -3
    var_4 = [var_1, var_2, var_3]

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



# Parsed testcases at query #25
#--------------------------




import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = module_0.CheckedType()



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_invariant. Retrieved 6/11 statements.
# Partially parsed test_checked_pmap_constructor_with_valid_invariant. Retrieved 6/12 statements.
# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_size. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'c'
    var_2 = 'b'
    var_3 = 'd'
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 3
    var_2 = 2
    var_3 = 4
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = lambda k, v: (k < v, 'Key must be less than value')
    var_1 = 1
    var_2 = 2
    var_3 = 0
    var_4 = 3
    var_5 = {var_1: var_3, var_2: var_4}

def test_case_0():
    var_0 = lambda k, v: (k < v, 'Key must be less than value')
    var_1 = 1
    var_2 = 3
    var_3 = 2
    var_4 = 4
    var_5 = {var_1: var_3, var_2: var_4}

def test_case_0():
    var_0 = 5



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_constructor_with_empty_initial. Retrieved 1/6 statements.
# Partially parsed test_constructor_with_list_initial. Retrieved 5/10 statements.
# Partially parsed test_constructor_with_pvector_initial. Retrieved 6/11 statements.
# Partially parsed test_constructor_with_invalid_type. Retrieved 5/9 statements.
# Partially parsed test_constructor_with_invalid_invariant. Retrieved 5/9 statements.


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

def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 1
    var_2 = -2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_check_types_with_empty_iterable. Retrieved 1/3 statements.


def test_case_0():
    var_0 = []



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_constructor_with_empty_initial. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_list_initial. Retrieved 4/7 statements.
# Partially parsed test_constructor_with_python_pvector_initial. Retrieved 5/8 statements.
# Partially parsed test_constructor_with_invalid_type. Retrieved 4/6 statements.
# Partially parsed test_constructor_with_invalid_invariant. Retrieved 5/9 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = module_0.CheckedPVector()
    var_1 = len(var_0)
    assert var_1 == 0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import pyrsistent._pvector as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
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



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 6/12 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_data. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_empty_data. Retrieved 2/8 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 2/8 statements.


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
    var_3 = 2.25
    var_4 = {var_1: var_1, var_2: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = {}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 10



# Parsed testcases at query #31
#--------------------------

# Failed to parse test_checked_type_create_predicate_false.




# Parsed testcases at query #32
#--------------------------

# Failed to parse test_checked_type_create_with_checked_type_subclass.




# Parsed testcases at query #33
#--------------------------

# Partially parsed test_checked_pvector_constructor_with_empty_initial. Retrieved 2/3 statements.
# Partially parsed test_checked_pvector_constructor_with_list_initial. Retrieved 4/7 statements.
# Partially parsed test_checked_pvector_constructor_with_python_pvector_initial. Retrieved 5/8 statements.
# Partially parsed test_checked_pvector_constructor_with_tuple_initial. Retrieved 4/7 statements.
# Partially parsed test_checked_pvector_constructor_with_generator_initial. Retrieved 4/8 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = module_0.CheckedPVector()
    var_1 = len(var_0)
    assert var_1 == 0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import pyrsistent._pvector as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.python_pvector(var_3)

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



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_constructor_empty_initial. Retrieved 1/6 statements.
# Partially parsed test_constructor_list_initial. Retrieved 5/10 statements.
# Partially parsed test_constructor_pvector_initial. Retrieved 6/11 statements.
# Partially parsed test_constructor_type_error. Retrieved 5/9 statements.
# Partially parsed test_constructor_invariant_error. Retrieved 5/9 statements.


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
    var_2 = 'not an int'
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 1
    var_2 = -2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]



# Parsed testcases at query #35
#--------------------------

# Partially parsed test__check_types_with_valid_types. Retrieved 4/7 statements.
# Partially parsed test__check_types_with_invalid_types. Retrieved 4/8 statements.
# Partially parsed test__check_types_with_multiple_valid_types. Retrieved 4/7 statements.
# Partially parsed test__check_types_with_multiple_invalid_types. Retrieved 4/8 statements.
# Partially parsed test__check_types_with_empty_iterable. Retrieved 1/4 statements.
# Partially parsed test__check_types_with_empty_expected_types. Retrieved 5/7 statements.
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



# Parsed testcases at query #36
#--------------------------




import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'Test that CheckedType cannot be instantiated directly.'
    var_1 = module_0.CheckedType()



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_restore_pickle_creates_instance_with_factory_fields. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = set()



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_maybe_parse_user_type_with_preserved_type.
# Failed to parse test_maybe_parse_user_type_with_non_iterable_type.
# Failed to parse test_maybe_parse_user_type_with_iterable_of_types.
# Partially parsed test_maybe_parse_user_type_with_nested_iterable. Retrieved 1/4 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'str'
    var_1 = module_0.maybe_parse_user_type(var_0)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = 'str'
    var_2 = [var_0, var_1]
    var_3 = module_0.maybe_parse_user_type(var_2)

def test_case_0():
    var_0 = 'float'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.maybe_parse_user_type(var_0)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_violated_invariant. Retrieved 4/9 statements.
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

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = '1.0'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 5
    var_2 = 1
    var_3 = {var_2: var_2}



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_initial_data. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 2/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_empty_initial_data. Retrieved 1/7 statements.
# Partially parsed test_checked_pmap_constructor_with_multiple_invariants. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_multiple_invariants_failure. Retrieved 4/9 statements.


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
    var_1 = '1'
    var_2 = 1.0
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = '1.0'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = lambda k, v: (v > 0, 'Value must be positive')
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = 2
    var_5 = {var_3: var_3, var_4: var_4}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = lambda k, v: (v > 0, 'Value must be positive')
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = -1.0
    var_5 = {var_3: var_4}



# Parsed testcases at query #4
#--------------------------




import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 5
    var_1 = True
    var_2 = 'Error1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'Error2'
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = [var_4, var_7]
    var_9 = module_0._invariant_errors(var_0, var_8)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 5
    var_1 = False
    var_2 = 'Error1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = True
    var_6 = 'Error2'
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = [var_4, var_8]
    var_10 = module_0._invariant_errors(var_0, var_9)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 5
    var_1 = False
    var_2 = 'Error1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'Error2'
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = [var_4, var_7]
    var_9 = module_0._invariant_errors(var_0, var_8)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 5
    var_1 = False
    var_2 = 'Error1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'Error2'
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = 'Error3'
    var_9 = (var_1, var_8)
    var_10 = lambda x: var_9
    var_11 = [var_4, var_7, var_10]
    var_12 = module_0._invariant_errors(var_0, var_11)



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_maybe_parse_user_type_with_preserved_type.
# Failed to parse test_maybe_parse_user_type_with_non_iterable_type.
# Failed to parse test_maybe_parse_user_type_with_iterable_of_types.
# Failed to parse test_maybe_parse_user_type_with_nested_iterable_of_types.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'str'
    var_1 = module_0.maybe_parse_user_type(var_0)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.maybe_parse_user_type(var_0)



# Parsed testcases at query #6
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_initial_data. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_size. Retrieved 1/7 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 2/7 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_invariant. Retrieved 4/9 statements.
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
    var_0 = 'a'
    var_1 = 'one'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = {var_0: var_0}

def test_case_0():
    var_0 = lambda k, v: (v == k * 2, 'Value must be double the key')
    var_1 = 1
    var_2 = 3.0
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = lambda k, v: (v == k * 2, 'Value must be double the key')
    var_1 = 1
    var_2 = 2
    var_3 = 4.0
    var_4 = {var_1: var_2, var_2: var_3}



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_data. Retrieved 6/11 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 1/7 statements.
# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_type_mismatch. Retrieved 3/8 statements.


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

def test_case_0():
    var_0 = 10

def test_case_0():
    var_0 = 'a'
    var_1 = 1.0
    var_2 = {var_0: var_1}



# Parsed testcases at query #8
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_initial_data. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_size. Retrieved 6/12 statements.
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
    var_0 = 1
    var_1 = 2
    var_2 = 'one'
    var_3 = 'two'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 4

def test_case_0():
    var_0 = 'invalid'
    var_1 = 'value'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 123
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (len(v) == k, 'Length mismatch')
    var_1 = 1
    var_2 = 'one'
    var_3 = {var_1: var_2}



# Parsed testcases at query #9
#--------------------------

# Partially parsed test__check_types_with_valid_types. Retrieved 4/7 statements.
# Partially parsed test__check_types_with_invalid_types. Retrieved 4/8 statements.
# Partially parsed test__check_types_with_multiple_valid_types. Retrieved 4/7 statements.
# Partially parsed test__check_types_with_multiple_invalid_types. Retrieved 4/8 statements.
# Partially parsed test__check_types_with_empty_iterable. Retrieved 1/4 statements.
# Partially parsed test__check_types_with_empty_expected_types. Retrieved 5/7 statements.
# Partially parsed test__check_types_with_custom_exception. Retrieved 4/9 statements.


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
    var_1 = 'a'
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = []

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #10
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

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = '1.0'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 5
    var_2 = 1
    var_3 = {var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_checked_type_constructor. Retrieved 1/2 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = module_0.CheckedType()



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_store_invariants_with_valid_invariants. Retrieved 6/16 statements.
# Partially parsed test_store_invariants_with_invalid_invariant. Retrieved 4/8 statements.
# Partially parsed test_store_invariants_with_no_invariants. Retrieved 5/9 statements.
# Partially parsed test_store_invariants_with_inherited_invariants. Retrieved 8/15 statements.
# Partially parsed test_store_invariants_with_multiple_inheritance. Retrieved 6/18 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'destination'
    var_2 = 'invariant'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = var_0[var_1]

def test_case_0():
    var_0 = 'not_callable'
    var_1 = {}
    var_2 = 'destination'
    var_3 = 'invariant'

def test_case_0():
    var_0 = {}
    var_1 = 'destination'
    var_2 = 'invariant'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 0

def test_case_0():
    var_0 = {}
    var_1 = 'destination'
    var_2 = 'invariant'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_0[var_1][var_5]
    var_7 = callable(var_6)

def test_case_0():
    var_0 = {}
    var_1 = 'destination'
    var_2 = 'invariant'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = var_0[var_1]



# Parsed testcases at query #13
#--------------------------

# Failed to parse test_is_preserved_predicate.




# Parsed testcases at query #14
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
    var_1 = 'a'
    var_2 = 2
    var_3 = 1.0
    var_4 = {var_1: var_3, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = '1.0'
    var_4 = {var_1: var_3, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_2, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 5
    var_2 = 1
    var_3 = 2
    var_4 = {var_2: var_2, var_3: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')



# Parsed testcases at query #15
#--------------------------

# Failed to parse test_predicate_at_line_18.




# Parsed testcases at query #16
#--------------------------

# Partially parsed test_constructor_with_empty_initial. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_list_initial. Retrieved 4/7 statements.
# Partially parsed test_constructor_with_python_pvector_initial. Retrieved 5/8 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = module_0.CheckedPVector()
    var_1 = len(var_0)
    assert var_1 == 0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import pyrsistent._pvector as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.python_pvector(var_3)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_checked_pset_constructor_with_empty_initial. Retrieved 2/3 statements.
# Partially parsed test_checked_pset_constructor_with_list_initial. Retrieved 5/10 statements.
# Partially parsed test_checked_pset_constructor_with_pmap_initial. Retrieved 8/11 statements.
# Partially parsed test_checked_pset_constructor_with_invalid_type. Retrieved 3/7 statements.
# Partially parsed test_checked_pset_constructor_with_invalid_invariant. Retrieved 4/8 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = module_0.CheckedPSet()
    var_1 = len(var_0)
    assert var_1 == 0

def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

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

def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 'invalid'
    var_2 = [var_1]

def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = -1
    var_2 = -2
    var_3 = [var_1, var_2]



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_restore_pickle_returns_instance_of_cls. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}



# Parsed testcases at query #19
#--------------------------

# Failed to parse test_wrap_invariant_with_single_bool_result.
# Failed to parse test_wrap_invariant_with_multiple_bool_results.
# Failed to parse test_wrap_invariant_with_all_true_results.
# Failed to parse test_wrap_invariant_with_all_false_results.




# Parsed testcases at query #20
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

def test_case_0():
    var_0 = 1
    var_1 = 'not_a_float'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 5
    var_1 = 1
    var_2 = {var_1: var_1}



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
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

def test_case_0():
    var_0 = 5



# Parsed testcases at query #2
#--------------------------




import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = module_0.CheckedType()



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_checked_pvector_constructor_empty. Retrieved 1/6 statements.
# Partially parsed test_checked_pvector_constructor_from_list. Retrieved 5/10 statements.
# Partially parsed test_checked_pvector_constructor_from_pvector. Retrieved 6/11 statements.


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



# Parsed testcases at query #4
#--------------------------

# Failed to parse test_maybe_parse_user_type_with_preserved_iterable_type.
# Failed to parse test_maybe_parse_user_type_with_non_iterable_type.
# Failed to parse test_maybe_parse_user_type_with_iterable_of_types.
# Failed to parse test_maybe_parse_user_type_with_nested_iterable.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = 'str'
    var_2 = [var_0, var_1]
    var_3 = module_0.maybe_parse_user_type(var_2)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.maybe_parse_user_type(var_0)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_store_invariants_with_valid_invariants. Retrieved 8/15 statements.
# Partially parsed test_store_invariants_with_multiple_inherited_invariants. Retrieved 6/18 statements.
# Partially parsed test_store_invariants_with_non_callable_invariant. Retrieved 4/8 statements.
# Partially parsed test_store_invariants_with_no_invariants. Retrieved 5/9 statements.
# Partially parsed test_store_invariants_with_direct_invariant. Retrieved 5/13 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_0[var_1][var_5]
    var_7 = callable(var_6)

def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = var_0[var_1]

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = 'invariants'
    var_3 = 'invariant'

def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 0

def test_case_0():
    var_0 = 'invariant'
    var_1 = ()
    var_2 = 'invariants'
    var_3 = 'invariant'
    var_4 = 0



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_restore_pickle_creates_instance_with_factory_fields. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}



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
    var_1 = 'a'
    var_2 = 1.0
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = '1.0'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 5
    var_2 = 1
    var_3 = {var_2: var_2}



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_all_invariants_pass. Retrieved 1/5 statements.
# Partially parsed test_single_invariant_fails. Retrieved 1/5 statements.
# Partially parsed test_multiple_invariants_mixed. Retrieved 1/7 statements.
# Partially parsed test_invariant_with_different_data. Retrieved 1/5 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = []
    var_2 = module_0._invariant_errors(var_0, var_1)

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'hi'



# Parsed testcases at query #9
#--------------------------




import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = False
    var_2 = 'Error1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = True
    var_6 = None
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = [var_4, var_8]
    var_10 = module_0._invariant_errors(var_0, var_9)



# Parsed testcases at query #10
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

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = '1.0'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 5
    var_2 = 1
    var_3 = {var_2: var_2}



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_data. Retrieved 6/11 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 1/7 statements.
# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_wrong_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_wrong_value_type. Retrieved 3/8 statements.


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

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'a'
    var_1 = 1.0
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_0: var_1}



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_checked_pvector_constructor_empty. Retrieved 1/5 statements.
# Partially parsed test_checked_pvector_constructor_with_list. Retrieved 5/9 statements.
# Partially parsed test_checked_pvector_constructor_with_pvector. Retrieved 6/10 statements.
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

def test_case_0():
    var_0 = lambda x: (x > 0, 'Non-positive')
    var_1 = 1
    var_2 = 2
    var_3 = -3
    var_4 = [var_1, var_2, var_3]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 4/7 statements.
# Partially parsed test_check_types_with_invalid_types. Retrieved 4/8 statements.
# Partially parsed test_check_types_with_multiple_valid_types. Retrieved 4/7 statements.
# Partially parsed test_check_types_with_empty_iterable. Retrieved 1/4 statements.
# Partially parsed test_check_types_with_empty_expected_types. Retrieved 5/7 statements.
# Partially parsed test_check_types_with_custom_exception. Retrieved 4/9 statements.
# Partially parsed test_check_types_with_string_type_names. Retrieved 6/8 statements.


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
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = []

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'builtins.int'
    var_5 = [var_4]



# Parsed testcases at query #14
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_initial_data. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_size. Retrieved 1/7 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_invariant. Retrieved 4/9 statements.
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

def test_case_0():
    var_0 = 1
    var_1 = 123
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (v == k * 2, 'Value must be double the key')
    var_1 = 1
    var_2 = 3.0
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = lambda k, v: (v == k * 2, 'Value must be double the key')
    var_1 = 1
    var_2 = 2
    var_3 = 4.0
    var_4 = {var_1: var_2, var_2: var_3}



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_store_invariants_with_single_invariant. Retrieved 11/12 statements.
# Partially parsed test_store_invariants_with_multiple_invariants. Retrieved 9/11 statements.
# Partially parsed test_store_invariants_with_inherited_invariants. Retrieved 4/14 statements.
# Partially parsed test_store_invariants_with_non_callable_invariant. Retrieved 5/7 statements.
# Partially parsed test_store_invariants_with_multiple_inherited_invariants. Retrieved 4/16 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = ()
    var_2 = 'invariants'
    var_3 = 'invariant'
    var_4 = module_0.store_invariants(var_0, var_1, var_2, var_3)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = ()
    var_2 = 'invariants'
    var_3 = 'invariant'
    var_4 = True
    var_5 = module_0.store_invariants(var_0, var_1, var_2, var_3)
    var_6 = var_0[var_2]
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 0
    var_9 = var_0[var_2][var_8]
    var_10 = callable(var_9)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = ()
    var_2 = 'invariants'
    var_3 = 'invariant'
    var_4 = True
    var_5 = False
    var_6 = module_0.store_invariants(var_0, var_1, var_2, var_3)
    var_7 = var_0[var_2]
    var_8 = len(var_7)
    assert var_8 == 1

def test_case_0():
    var_0 = lambda self: (True, self)
    var_1 = 'invariants'
    var_2 = 'invariant'
    var_3 = 0

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = ()
    var_2 = 'invariants'
    var_3 = 'invariant'
    var_4 = module_0.store_invariants(var_0, var_1, var_2, var_3)

def test_case_0():
    var_0 = lambda self: (True, self)
    var_1 = lambda self: (False, self)
    var_2 = 'invariants'
    var_3 = 'invariant'



# Parsed testcases at query #16
#--------------------------

# Failed to parse test_wrap_invariant_with_single_bool_result.
# Failed to parse test_wrap_invariant_with_multiple_bool_results.
# Partially parsed test_wrap_invariant_with_all_true_results. Retrieved 3/7 statements.
# Partially parsed test_wrap_invariant_with_empty_result_list. Retrieved 3/7 statements.
# Partially parsed test_wrap_invariant_with_args_and_kwargs. Retrieved 2/6 statements.


def test_case_0():
    var_0 = True
    var_1 = tuple()
    var_2 = (var_0, var_1)

def test_case_0():
    var_0 = True
    var_1 = tuple()
    var_2 = (var_0, var_1)

def test_case_0():
    var_0 = 'test'
    var_1 = True



# Parsed testcases at query #17
#--------------------------




import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = '__invariants__'
    var_1 = True
    var_2 = lambda : var_1
    var_3 = {var_0: var_2}
    var_4 = ()
    var_5 = '__invariants__'
    var_6 = '__invariants__'
    var_7 = module_0.store_invariants(var_3, var_4, var_5, var_6)



# Parsed testcases at query #18
#--------------------------

# Failed to parse test_predicate_at_line_18.




# Parsed testcases at query #19
#--------------------------

# Failed to parse test_is_preserved_predicate.




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

def test_case_0():
    var_0 = 1
    var_1 = 'not a float'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 10



# Parsed testcases at query #21
#--------------------------

# Failed to parse test_checked_pmap_constructor_empty.
# Partially parsed test_checked_pmap_constructor_with_initial_data. Retrieved 5/11 statements.
# Partially parsed test_checked_pmap_constructor_with_size. Retrieved 1/7 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 2/7 statements.
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
    var_0 = 'a'
    var_1 = 'one'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = {var_0: var_0}

def test_case_0():
    var_0 = lambda k, v: (k < v, 'Key must be less than value')
    var_1 = 1
    var_2 = 0
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = lambda k, v: (k < v, 'Key must be less than value')
    var_1 = 1
    var_2 = 3
    var_3 = 2
    var_4 = 4
    var_5 = {var_1: var_3, var_2: var_4}



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_all_valid_invariants. Retrieved 1/5 statements.
# Partially parsed test_single_invalid_invariant. Retrieved 1/5 statements.
# Partially parsed test_mixed_invariants. Retrieved 1/7 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = []
    var_2 = module_0._invariant_errors(var_0, var_1)

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_merge_invariant_results_all_passing. Retrieved 6/7 statements.
# Partially parsed test_merge_invariant_results_some_failing. Retrieved 9/10 statements.
# Partially parsed test_merge_invariant_results_all_failing. Retrieved 6/7 statements.
# Partially parsed test_merge_invariant_results_empty_input. Retrieved 1/2 statements.


def test_case_0():
    var_0 = True
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
    var_0 = False
    var_1 = 'data1'
    var_2 = (var_0, var_1)
    var_3 = 'data2'
    var_4 = (var_0, var_3)
    var_5 = [var_2, var_4]

def test_case_0():
    var_0 = []



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_data. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_key_type. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_value_type. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_invariant. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 5/11 statements.


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

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = '1.0'
    var_4 = {var_1: var_3, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_2, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 5
    var_2 = 1
    var_3 = 2
    var_4 = {var_2: var_2, var_3: var_3}



# Parsed testcases at query #26
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

def test_case_0():
    var_0 = 1
    var_1 = '1.0'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 5
    var_1 = 1
    var_2 = {var_1: var_1}



# Parsed testcases at query #27
#--------------------------

# Partially parsed test__checked_type_create_with_instance_of_cls. Retrieved 1/4 statements.
# Partially parsed test__checked_type_create_with_non_instance_and_no_checked_types. Retrieved 2/5 statements.
# Partially parsed test__checked_type_create_with_checked_type_in_list. Retrieved 4/10 statements.
# Partially parsed test__checked_type_create_with_instance_in_checked_types. Retrieved 6/11 statements.
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

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = '__main__.CheckedType'
    var_1 = [var_0]
    var_2 = module_0.CheckedType()
    var_3 = 'other_data'
    var_4 = [var_2, var_3]
    var_5 = module_0.CheckedType()
    var_6 = [var_5, var_3]

def test_case_0():
    var_0 = '__main__.CheckedType'
    var_1 = [var_0]
    var_2 = 'data1'
    var_3 = 'data2'
    var_4 = [var_2, var_3]
    var_5 = True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_store_types_with_single_type. Retrieved 5/6 statements.
# Partially parsed test_store_types_with_multiple_types. Retrieved 5/6 statements.
# Partially parsed test_store_types_with_string_type. Retrieved 5/6 statements.
# Partially parsed test_store_types_with_inherited_type. Retrieved 3/7 statements.
# Partially parsed test_store_types_with_mixed_types. Retrieved 3/8 statements.
# Partially parsed test_store_types_with_nested_iterable. Retrieved 5/7 statements.
# Partially parsed test_store_types_with_invalid_type. Retrieved 5/7 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'types'
    var_3 = 'type'
    var_4 = module_0._store_types(var_0, var_1, var_2, var_3)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'types'
    var_3 = 'type'
    var_4 = module_0._store_types(var_0, var_1, var_2, var_3)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'types'
    var_3 = 'type'
    var_4 = module_0._store_types(var_0, var_1, var_2, var_3)

def test_case_0():
    var_0 = {}
    var_1 = 'types'
    var_2 = 'type'

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

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'types'
    var_3 = 'type'
    var_4 = module_0._store_types(var_0, var_1, var_2, var_3)



# Parsed testcases at query #29
#--------------------------

# Failed to parse test_checked_type_create_with_checked_type_subclass.




# Parsed testcases at query #30
#--------------------------

# Partially parsed test_checked_pmap_constructor_with_valid_initial_data. Retrieved 4/10 statements.
# Partially parsed test_checked_pmap_constructor_with_invalid_initial_data. Retrieved 5/10 statements.
# Partially parsed test_checked_pmap_constructor_with_size_parameter. Retrieved 2/8 statements.
# Partially parsed test_checked_pmap_constructor_with_empty_initial_data. Retrieved 1/7 statements.
# Partially parsed test_checked_pmap_constructor_with_wrong_key_type. Retrieved 4/9 statements.
# Partially parsed test_checked_pmap_constructor_with_wrong_value_type. Retrieved 4/9 statements.


def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = 3.0
    var_4 = {var_1: var_2, var_2: var_3}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 10

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 'a'
    var_2 = 1.0
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 'a'
    var_3 = {var_1: var_2}



