####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_check_global_invariants_no_errors. Retrieved 5/7 statements.
# Partially parsed test_check_global_invariants_with_errors. Retrieved 9/12 statements.


def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = [var_4]

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = 'ERROR1'
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = True
    var_6 = None
    var_7 = (var_5, var_6)
    var_8 = lambda _: var_7
    var_9 = [var_4, var_8]
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_check_field_parameters_with_valid_types. Retrieved 2/9 statements.
# Partially parsed test_check_field_parameters_with_invalid_type_parameter. Retrieved 3/11 statements.
# Partially parsed test_check_field_parameters_with_invalid_initial_type. Retrieved 2/10 statements.
# Partially parsed test_check_field_parameters_with_valid_initial_type. Retrieved 2/9 statements.
# Partially parsed test_check_field_parameters_with_callable_initial. Retrieved 3/10 statements.
# Partially parsed test_check_field_parameters_with_non_callable_invariant. Retrieved 1/9 statements.
# Partially parsed test_check_field_parameters_with_non_callable_factory. Retrieved 1/9 statements.
# Partially parsed test_check_field_parameters_with_non_callable_serializer. Retrieved 2/10 statements.


def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = 123
    var_1 = True
    var_2 = None

def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = 42
    var_1 = True
    var_2 = None

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = True
    var_1 = None



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint_with_false_ignore_extra. Retrieved 1/3 statements.
# Partially parsed test_is_field_ignore_extra_complaint_with_non_matching_type. Retrieved 1/3 statements.
# Partially parsed test_is_field_ignore_extra_complaint_with_matching_type_and_no_ignore_extra_param. Retrieved 3/5 statements.
# Partially parsed test_is_field_ignore_extra_complaint_with_matching_type_and_ignore_extra_param. Retrieved 3/5 statements.
# Partially parsed test_is_field_ignore_extra_complaint_with_set_type. Retrieved 3/6 statements.
# Partially parsed test_is_field_ignore_extra_complaint_with_empty_tuple_type. Retrieved 4/6 statements.


def test_case_0():
    var_0 = False

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = None
    var_1 = lambda : var_0
    var_2 = True

def test_case_0():
    var_0 = None
    var_1 = lambda *: var_0
    var_2 = True

def test_case_0():
    var_0 = None
    var_1 = lambda *: var_0
    var_2 = True

def test_case_0():
    var_0 = ()
    var_1 = None
    var_2 = lambda *: var_1
    var_3 = True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test__sequence_field_with_checked_class_and_item_type. Retrieved 6/11 statements.
# Partially parsed test__sequence_field_with_optional_true. Retrieved 6/14 statements.
# Partially parsed test__sequence_field_with_item_invariant. Retrieved 8/13 statements.
# Partially parsed test__sequence_field_with_invariant. Retrieved 7/12 statements.
# Partially parsed test__sequence_field_with_optional_type. Retrieved 7/15 statements.


def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = True
    var_1 = 'a'
    var_2 = 'b'
    var_3 = {var_1, var_2}
    var_4 = None
    var_5 = [var_4]
    var_6 = {var_1, var_2}

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 0
    var_6 = lambda x: x > var_5
    var_7 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = False
    var_1 = 'a'
    var_2 = 'b'
    var_3 = {var_1, var_2}
    var_4 = 0
    var_5 = lambda x: len(x) > var_4
    var_6 = {var_1, var_2}

def test_case_0():
    var_0 = True
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = None
    var_6 = [var_5]
    var_7 = [var_1, var_2, var_3]



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_types_to_names_with_single_type.
# Failed to parse test_types_to_names_with_multiple_types.
# Partially parsed test_types_to_names_with_mixed_types_and_strings. Retrieved 1/3 statements.
# Failed to parse test_types_to_names_with_custom_type.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = 'builtins.str'
    var_2 = (var_0, var_1)
    var_3 = module_0._types_to_names(var_2)
    assert var_3 == 'IntStr'

def test_case_0():
    var_0 = 'builtins.str'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ()
    var_1 = module_0._types_to_names(var_0)
    assert var_1 == ''



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_check_field_parameters_initial_valid_type. Retrieved 6/9 statements.


def test_case_0():
    var_0 = 42
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = None
    var_4 = lambda : var_3
    var_5 = lambda x: x



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = 'format'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda fmt, val: f'serialized_{val}'
    var_1 = 'format'
    var_2 = 'value'
    var_3 = module_0.serialize(var_0, var_1, var_2)
    assert var_3 == 'serialized_value'



# Parsed testcases at query #8
#--------------------------

# Failed to parse test_predicate_evaluates_to_false.




# Parsed testcases at query #9
#--------------------------

# Partially parsed test_check_field_parameters_with_valid_field. Retrieved 5/8 statements.
# Partially parsed test_check_field_parameters_with_invalid_type_parameter. Retrieved 7/10 statements.
# Partially parsed test_check_field_parameters_with_invalid_initial_type. Retrieved 6/10 statements.
# Partially parsed test_check_field_parameters_with_non_callable_invariant. Retrieved 4/8 statements.
# Partially parsed test_check_field_parameters_with_non_callable_factory. Retrieved 5/9 statements.
# Partially parsed test_check_field_parameters_with_non_callable_serializer. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 0
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = lambda : var_0
    var_4 = lambda x: x

def test_case_0():
    var_0 = 123
    var_1 = [var_0]
    var_2 = 0
    var_3 = True
    var_4 = lambda x: var_3
    var_5 = lambda : var_2
    var_6 = lambda x: x

def test_case_0():
    var_0 = 'not an int'
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = 0
    var_4 = lambda : var_3
    var_5 = lambda x: x

def test_case_0():
    var_0 = 0
    var_1 = 'not callable'
    var_2 = lambda : var_0
    var_3 = lambda x: x

def test_case_0():
    var_0 = 0
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = 'not callable'
    var_4 = lambda x: x

def test_case_0():
    var_0 = 0
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = lambda : var_0
    var_4 = 'not callable'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 'MockCheckedClass'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = 'b'
    var_6 = 'c'
    var_7 = [var_4, var_5, var_6]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 15/21 statements.


def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'initial'
    var_4 = 'invariant'
    var_5 = 'factory'
    var_6 = 'serializer'
    var_7 = 123
    var_8 = [var_7]
    var_9 = None
    var_10 = True
    var_11 = lambda x: var_10
    var_12 = lambda : var_9
    var_13 = lambda x: x
    var_14 = {var_2: var_8, var_3: var_9, var_4: var_11, var_5: var_12, var_6: var_13}
    var_15 = [var_0, var_1, var_14]
    var_16 = 'Type parameter expected, not'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_check_field_parameters_with_invalid_type_parameter. Retrieved 14/18 statements.
# Partially parsed test_check_field_parameters_with_valid_type_parameters. Retrieved 12/16 statements.
# Partially parsed test_check_field_parameters_with_invalid_initial_type. Retrieved 13/18 statements.
# Partially parsed test_check_field_parameters_with_valid_initial_type. Retrieved 13/17 statements.
# Partially parsed test_check_field_parameters_with_callable_initial. Retrieved 16/18 statements.
# Partially parsed test_check_field_parameters_with_non_callable_invariant. Retrieved 11/16 statements.
# Partially parsed test_check_field_parameters_with_non_callable_factory. Retrieved 11/16 statements.
# Partially parsed test_check_field_parameters_with_non_callable_serializer. Retrieved 12/17 statements.


def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'initial'
    var_4 = 'invariant'
    var_5 = 'factory'
    var_6 = 'serializer'
    var_7 = 123
    var_8 = [var_7]
    var_9 = True
    var_10 = lambda x: var_9
    var_11 = None
    var_12 = lambda : var_11
    var_13 = lambda x: x

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'initial'
    var_4 = 'invariant'
    var_5 = 'factory'
    var_6 = 'serializer'
    var_7 = True
    var_8 = lambda x: var_7
    var_9 = None
    var_10 = lambda : var_9
    var_11 = lambda x: x

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'initial'
    var_4 = 'invariant'
    var_5 = 'factory'
    var_6 = 'serializer'
    var_7 = 'not_an_int'
    var_8 = True
    var_9 = lambda x: var_8
    var_10 = None
    var_11 = lambda : var_10
    var_12 = lambda x: x

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'initial'
    var_4 = 'invariant'
    var_5 = 'factory'
    var_6 = 'serializer'
    var_7 = 42
    var_8 = True
    var_9 = lambda x: var_8
    var_10 = None
    var_11 = lambda : var_10
    var_12 = lambda x: x

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'initial'
    var_4 = 'invariant'
    var_5 = 'factory'
    var_6 = 'serializer'
    var_7 = []
    var_8 = 42
    var_9 = lambda : var_8
    var_10 = True
    var_11 = lambda x: var_10
    var_12 = None
    var_13 = lambda : var_12
    var_14 = lambda x: x
    var_15 = {var_2: var_7, var_3: var_9, var_4: var_11, var_5: var_13, var_6: var_14}
    var_16 = [var_0, var_1, var_15]

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'initial'
    var_4 = 'invariant'
    var_5 = 'factory'
    var_6 = 'serializer'
    var_7 = 'not_callable'
    var_8 = None
    var_9 = lambda : var_8
    var_10 = lambda x: x

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'initial'
    var_4 = 'invariant'
    var_5 = 'factory'
    var_6 = 'serializer'
    var_7 = True
    var_8 = lambda x: var_7
    var_9 = 'not_callable'
    var_10 = lambda x: x

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'initial'
    var_4 = 'invariant'
    var_5 = 'factory'
    var_6 = 'serializer'
    var_7 = True
    var_8 = lambda x: var_7
    var_9 = None
    var_10 = lambda : var_9
    var_11 = 'not_callable'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_check_global_invariants_no_errors. Retrieved 5/7 statements.
# Partially parsed test_check_global_invariants_with_errors. Retrieved 9/12 statements.
# Partially parsed test_check_global_invariants_multiple_errors. Retrieved 8/11 statements.


def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = [var_4]

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = 'ERROR1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = True
    var_6 = None
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = [var_4, var_8]
    var_10 = bool(False)
    assert var_10 is True

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = 'ERROR1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'ERROR2'
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = [var_4, var_7]
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_check_field_parameters_with_valid_field. Retrieved 4/8 statements.
# Partially parsed test_check_field_parameters_with_invalid_type_parameter. Retrieved 4/9 statements.
# Partially parsed test_check_field_parameters_with_invalid_initial_type. Retrieved 4/9 statements.
# Partially parsed test_check_field_parameters_with_non_callable_invariant. Retrieved 4/9 statements.
# Partially parsed test_check_field_parameters_with_non_callable_factory. Retrieved 4/9 statements.
# Partially parsed test_check_field_parameters_with_non_callable_serializer. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 5
    var_1 = lambda x: True
    var_2 = lambda : 0
    var_3 = lambda x: str(x)

def test_case_0():
    var_0 = 5
    var_1 = lambda x: True
    var_2 = lambda : 0
    var_3 = lambda x: str(x)
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 5.5
    var_1 = lambda x: True
    var_2 = lambda : 0
    var_3 = lambda x: str(x)
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 5
    var_1 = 'not callable'
    var_2 = lambda : 0
    var_3 = lambda x: str(x)
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 5
    var_1 = lambda x: True
    var_2 = 'not callable'
    var_3 = lambda x: str(x)
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 5
    var_1 = lambda x: True
    var_2 = lambda : 0
    var_3 = 'not callable'
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test__make_seq_field_type_creates_subclass_with_correct_name. Retrieved 2/5 statements.
# Partially parsed test__make_seq_field_type_creates_subclass_with_correct_type. Retrieved 2/6 statements.
# Partially parsed test__make_seq_field_type_creates_subclass_with_correct_attributes. Retrieved 3/7 statements.
# Partially parsed test__make_seq_field_type_returns_cached_type. Retrieved 3/7 statements.
# Partially parsed test__make_seq_field_type_creates_different_types_for_different_item_types. Retrieved 3/7 statements.
# Partially parsed test__make_seq_field_type_creates_different_types_for_different_checked_classes. Retrieved 3/9 statements.
# Partially parsed test__make_seq_field_type_creates_type_with_correct_reduce. Retrieved 5/10 statements.


def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = 5

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = lambda x: var_0

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = lambda x: var_0

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = lambda x: var_0

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_2, var_3]



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = set()



# Parsed testcases at query #18
#--------------------------

# Failed to parse test_pmap_field_creates_checked_pmap_with_correct_types.
# Partially parsed test_pmap_field_with_optional_creates_checked_pmap_with_none_type. Retrieved 5/15 statements.
# Partially parsed test_pmap_field_with_invariant_preserves_invariant. Retrieved 3/4 statements.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = [var_1]
    var_3 = 'a'
    var_4 = {var_0: var_3}
    var_5 = {var_0: var_3}

def test_case_0():
    var_0 = True
    var_1 = 'test'
    var_2 = (var_0, var_1)

def test_case_0():
    var_0 = True
    var_1 = 'test'
    var_2 = (var_0, var_1)



# Parsed testcases at query #19
#--------------------------

# Failed to parse test_pmap_field_basic.
# Partially parsed test_pmap_field_optional. Retrieved 4/16 statements.
# Partially parsed test_invariant. Retrieved 1/4 statements.
# Partially parsed test_pmap_field_with_invariant. Retrieved 5/4 statements.
# Partially parsed test_invariant. Retrieved 1/4 statements.
# Partially parsed test_pmap_field_optional_with_invariant. Retrieved 7/4 statements.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = [var_1]
    var_3 = 'a'
    var_4 = {var_0: var_3}

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 'a'
    var_3 = {var_1: var_2}
    var_4 = False

def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = 'a'
    var_3 = {var_1: var_2}
    var_4 = False

def test_case_0():
    var_0 = 0
    var_1 = True
    var_2 = None
    var_3 = [var_2]
    var_4 = 'a'
    var_5 = {var_1: var_4}
    var_6 = {var_1: var_4}
    var_7 = False

def test_case_0():
    var_0 = 0
    var_1 = True
    var_2 = None
    var_3 = [var_2]
    var_4 = 'a'
    var_5 = {var_1: var_4}
    var_6 = {var_1: var_4}
    var_7 = False



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_make_seq_field_type_creates_new_type. Retrieved 2/8 statements.
# Partially parsed test_make_seq_field_type_reuses_existing_type. Retrieved 2/7 statements.
# Partially parsed test_make_seq_field_type_sets_correct_name. Retrieved 2/8 statements.
# Partially parsed test_make_seq_field_type_with_string_type. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 'Int'

def test_case_0():
    var_0 = 'builtins.str'
    var_1 = 0
    var_2 = lambda x: len(x) > var_1



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_check_field_parameters_with_non_callable_invariant. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 5
    var_1 = 'not_callable'
    var_2 = None
    var_3 = lambda : var_2
    var_4 = lambda x: x



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_check_global_invariants_with_valid_invariants. Retrieved 5/7 statements.


def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = [var_4]



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = set()



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_check_global_invariants_with_valid_invariants. Retrieved 7/9 statements.


def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = (var_1, var_2)
    var_6 = lambda _: var_5
    var_7 = [var_4, var_6]



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_restore_seq_field_pickle_returns_correct_type. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 'MockCheckedClass'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]



# Parsed testcases at query #26
#--------------------------

# Failed to parse test_pmap_field_basic.
# Partially parsed test_pmap_field_optional. Retrieved 4/17 statements.
# Partially parsed test_pmap_field_with_invariant. Retrieved 4/4 statements.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = [var_1]
    var_3 = 'a'
    var_4 = {var_0: var_3}

def test_case_0():
    var_0 = True
    var_1 = 'OK'
    var_2 = (var_0, var_1)
    var_3 = bool(var_0)
    assert var_3 is True
    var_4 = {var_1}

def test_case_0():
    var_0 = True
    var_1 = 'OK'
    var_2 = (var_0, var_1)
    var_3 = bool(var_0)
    assert var_3 is True
    var_4 = {var_1}



# Parsed testcases at query #27
#--------------------------

# Partially parsed test__make_seq_field_type_creates_subclass_with_correct_name. Retrieved 1/5 statements.
# Partially parsed test__make_seq_field_type_sets_type_and_invariant. Retrieved 1/4 statements.


def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = lambda x: x



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_pmap_field_with_optional_false. Retrieved 1/8 statements.
# Partially parsed test_pmap_field_with_optional_true. Retrieved 5/15 statements.
# Partially parsed test_pmap_field_with_invariant. Retrieved 4/4 statements.


def test_case_0():
    var_0 = False

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = [var_1]
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = {var_3: var_0}

def test_case_0():
    var_0 = True
    var_1 = 'Test'
    var_2 = (var_0, var_1)
    var_3 = False

def test_case_0():
    var_0 = True
    var_1 = 'Test'
    var_2 = (var_0, var_1)
    var_3 = False



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_pmap_field_optional_type_predicate. Retrieved 1/4 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_check_field_parameters_with_invalid_type. Retrieved 1/5 statements.
# Failed to parse test_check_field_parameters_with_invalid_initial_type.
# Failed to parse test_check_field_parameters_with_non_callable_invariant.
# Partially parsed test_check_field_parameters_with_non_callable_factory. Retrieved 1/8 statements.
# Partially parsed test_check_field_parameters_with_non_callable_serializer. Retrieved 2/10 statements.
# Partially parsed test_check_field_parameters_with_valid_parameters. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 123

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = True
    var_1 = 123

def test_case_0():
    var_0 = True
    var_1 = 123



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_set_fields_with_non_empty_bases. Retrieved 6/10 statements.
# Partially parsed test_set_fields_with_pfield_instances. Retrieved 6/14 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'value1'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'new_name'
    var_5 = module_0.set_fields(var_2, var_3, var_4)
    var_6 = bool(var_2 == {'field1': 'value1', 'new_name': {}})
    assert var_6 is True

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'field3'
    var_6 = 'value3'
    var_7 = {var_5: var_6}
    var_8 = 'field4'
    var_9 = 'value4'
    var_10 = {var_8: var_9}
    var_11 = 'new_name'
    var_12 = bool(var_10 == {'field4': 'value4', 'new_name': {'field1': 'value1', 'field2': 'value2', 'field3': 'value3'}})
    assert var_12 is True

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'value2'
    var_3 = []
    var_4 = 'new_name'
    var_5 = 'new_name'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_check_global_invariants_no_errors. Retrieved 7/9 statements.
# Partially parsed test_check_global_invariants_with_errors. Retrieved 9/12 statements.
# Partially parsed test_check_global_invariants_multiple_errors. Retrieved 8/11 statements.


def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = (var_1, var_2)
    var_6 = lambda _: var_5
    var_7 = [var_4, var_6]

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = 'ERROR1'
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = True
    var_6 = None
    var_7 = (var_5, var_6)
    var_8 = lambda _: var_7
    var_9 = [var_4, var_8]
    var_10 = bool(False)
    assert var_10 is True

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = 'ERROR1'
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = 'ERROR2'
    var_6 = (var_1, var_5)
    var_7 = lambda _: var_6
    var_8 = [var_4, var_7]
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_optional_factory_returns_none. Retrieved 2/4 statements.


def test_case_0():
    var_0 = True
    var_1 = None



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_pfield_constructor. Retrieved 7/9 statements.


def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = 'initial'
    var_3 = True
    var_4 = 'factory'
    var_5 = lambda : var_4
    var_6 = lambda x: x



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_pfield_constructor. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 0
    var_3 = True
    var_4 = lambda : var_0



# Parsed testcases at query #36
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_new_class.
# Failed to parse test_make_pmap_field_type_returns_cached_class.
# Failed to parse test_make_pmap_field_type_sets_correct_name.
# Partially parsed test_make_pmap_field_type_reduce_method. Retrieved 3/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'builtins.str'
    var_1 = 'builtins.int'
    var_2 = module_0._make_pmap_field_type(var_0, var_1)
    var_3 = var_2.__key_type__
    var_4 = var_2.__value_type__
    var_5 = var_2.__name__
    assert var_5 == 'StrToIntPMap'

def test_case_0():
    var_0 = 1
    var_1 = 'one'
    var_2 = {var_0: var_1}



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_predicate_at_line_5_evaluates_to_true. Retrieved 4/16 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'field3'
    var_3 = 'field4'
    var_4 = 'field1'
    var_5 = 'field5'
    var_6 = 'value'
    var_7 = 'name'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_check_global_invariants_with_valid_invariants. Retrieved 5/7 statements.


def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = [var_4]



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_set_fields_with_single_base. Retrieved 3/6 statements.
# Partially parsed test_set_fields_with_multiple_bases. Retrieved 4/8 statements.
# Partially parsed test_set_fields_with_pfield_in_dct. Retrieved 8/16 statements.
# Partially parsed test_set_fields_with_pfield_and_overlapping_keys. Retrieved 6/14 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'test_name'
    var_3 = module_0.set_fields(var_0, var_1, var_2)
    var_4 = bool(var_0 == {'test_name': {}})
    assert var_4 is True

def test_case_0():
    var_0 = {}
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = 'test_name'
    var_5 = bool(var_0 == {'test_name': {'key1': 'value1'}})
    assert var_5 is True

def test_case_0():
    var_0 = {}
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = 'key2'
    var_5 = 'value2'
    var_6 = {var_4: var_5}
    var_7 = 'test_name'
    var_8 = bool(var_0 == {'test_name': {'key1': 'value1', 'key2': 'value2'}})
    assert var_8 is True

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'value'
    var_3 = 'key1'
    var_4 = 'value1'
    var_5 = {var_3: var_4}
    var_6 = 'test_name'
    var_7 = 'test_name'
    var_8 = 'key1'
    var_9 = 'value1'

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value'
    var_3 = 'key1'
    var_4 = 'base_value'
    var_5 = {var_3: var_4}
    var_6 = 'test_name'
    var_7 = 'test_name'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = set()



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_pmap_field_optional_factory_returns_none. Retrieved 2/4 statements.


def test_case_0():
    var_0 = True
    var_1 = None



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_pmap_field_optional_type_predicate. Retrieved 2/11 statements.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = [var_1]



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = 'some_format'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test__make_seq_field_type_with_checked_class_and_item_type. Retrieved 2/6 statements.
# Partially parsed test__make_seq_field_type_with_cached_type. Retrieved 2/6 statements.
# Partially parsed test__make_seq_field_type_with_different_item_types. Retrieved 3/6 statements.
# Partially parsed test__make_seq_field_type_with_custom_checked_class. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = lambda x: len(x) > var_0

def test_case_0():
    var_0 = 0.0
    var_1 = lambda x: x >= var_0



# Parsed testcases at query #45
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_subclass_with_correct_name.
# Failed to parse test_make_pmap_field_type_sets_key_and_value_types.
# Failed to parse test_make_pmap_field_type_reuses_existing_type.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = 'builtins.str'
    var_2 = module_0._make_pmap_field_type(var_0, var_1)
    var_3 = var_2.__name__
    assert var_3 == 'IntToStrPMap'
    var_4 = var_2.__key_type__
    var_5 = var_2.__value_type__



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_pfield_initialization_with_factory. Retrieved 4/5 statements.


def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = 0
    var_3 = lambda : var_2



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_predicate_at_line_3_evaluates_to_false. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 1
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = [var_0, var_1, var_5]



# Parsed testcases at query #48
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_new_class.
# Failed to parse test_make_pmap_field_type_reuses_existing_class.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'builtins.str'
    var_1 = 'builtins.int'
    var_2 = module_0._make_pmap_field_type(var_0, var_1)
    var_3 = var_2.__name__
    assert var_3 == 'StrToIntPMap'
    var_4 = var_2.__key_type__
    var_5 = var_2.__value_type__

import pyrsistent._field_common as module_0
import pyrsistent._checked_types as module_1

def test_case_0():
    var_0 = 'my.module.CustomKey'
    var_1 = 'my.module.CustomValue'
    var_2 = module_0._make_pmap_field_type(var_0, var_1)
    var_3 = var_2.__name__
    assert var_3 == 'CustomKeyToCustomValuePMap'
    var_4 = module_1._get_class(var_0)
    var_5 = var_2.__key_type__
    var_6 = bool(var_2.__key_type__ == var_4)
    assert var_6 is True
    var_7 = module_1._get_class(var_1)
    var_8 = var_2.__value_type__
    var_9 = bool(var_2.__value_type__ == var_7)
    assert var_9 is True



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 15/24 statements.


def test_case_0():
    var_0 = 'MockCheckedClass'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = 'MockItemType'
    var_5 = ()
    var_6 = {}
    var_7 = [var_4, var_5, var_6]
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = [var_8, var_9, var_10]
    var_12 = 'MockType'
    var_13 = ()
    var_14 = 'create'
    var_15 = lambda self, data, **kwargs: data
    var_16 = {var_14: var_15}
    var_17 = [var_12, var_13, var_16]



# Parsed testcases at query #50
#--------------------------

# Failed to parse test_check_field_parameters_valid.
# Failed to parse test_check_field_parameters_invalid_type_parameter.
# Failed to parse test_check_field_parameters_invalid_initial_type.
# Failed to parse test_check_field_parameters_non_callable_invariant.
# Failed to parse test_check_field_parameters_non_callable_factory.
# Failed to parse test_check_field_parameters_non_callable_serializer.




# Parsed testcases at query #51
#--------------------------

# Partially parsed test_check_field_parameters_with_invalid_type_parameter. Retrieved 6/13 statements.
# Partially parsed test_check_field_parameters_with_invalid_initial_type. Retrieved 6/13 statements.
# Partially parsed test_check_field_parameters_with_non_callable_invariant. Retrieved 4/11 statements.
# Partially parsed test_check_field_parameters_with_non_callable_factory. Retrieved 4/11 statements.
# Partially parsed test_check_field_parameters_with_non_callable_serializer. Retrieved 5/12 statements.
# Partially parsed test_check_field_parameters_with_valid_parameters. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 123
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = None
    var_4 = lambda : var_3
    var_5 = lambda x: x
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 1.5
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = None
    var_4 = lambda : var_3
    var_5 = lambda x: x
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 123
    var_1 = None
    var_2 = lambda : var_1
    var_3 = lambda x: x
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = 123
    var_3 = lambda x: x
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = None
    var_3 = lambda : var_2
    var_4 = 123
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 5
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = None
    var_4 = lambda : var_3
    var_5 = lambda x: x



# Parsed testcases at query #52
#--------------------------

# Failed to parse test_valid_field_parameters.
# Failed to parse test_invalid_type_parameter.
# Failed to parse test_invalid_initial_type.
# Failed to parse test_non_callable_invariant.
# Failed to parse test_non_callable_factory.
# Failed to parse test_non_callable_serializer.




# Parsed testcases at query #53
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 2/7 statements.


def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = 'format'



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 9/16 statements.


def test_case_0():
    var_0 = 'MockType'
    var_1 = ()
    var_2 = 'create'
    var_3 = lambda self, data, **kwargs: data
    var_4 = {var_2: var_3}
    var_5 = [var_0, var_1, var_4]
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 1/4 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 2/5 statements.


def test_case_0():
    var_0 = []
    var_1 = 'format'

def test_case_0():
    var_0 = 'format'
    var_1 = 'test_value'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_pfield_constructor. Retrieved 6/8 statements.


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 0
    var_3 = True
    var_4 = lambda : var_0
    var_5 = lambda x: str(x)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_restore_pmap_field_pickle_calls_restore_pickle. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_check_type_with_valid_type. Retrieved 2/8 statements.
# Partially parsed test_check_type_with_invalid_type. Retrieved 2/9 statements.
# Partially parsed test_check_type_with_no_type_specified. Retrieved 3/8 statements.
# Partially parsed test_check_type_with_multiple_valid_types. Retrieved 3/11 statements.
# Partially parsed test_check_type_with_string_type_name. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 123

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'not_an_int'
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = None
    var_1 = 'test_field'
    var_2 = 'any_value'

def test_case_0():
    var_0 = 'test_field'
    var_1 = 123
    var_2 = 123.45

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = (var_0,)
    var_2 = 'test_field'
    var_3 = 123



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_new_class.
# Failed to parse test_make_pmap_field_type_reuses_existing_class.
# Failed to parse test_make_pmap_field_type_custom_class_names.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'builtins.str'
    var_1 = 'builtins.int'
    var_2 = module_0._make_pmap_field_type(var_0, var_1)
    var_3 = var_2.__key_type__
    var_4 = var_2.__value_type__
    var_5 = var_2.__name__
    assert var_5 == 'StrToIntPMap'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_valid_field_parameters. Retrieved 4/8 statements.
# Partially parsed test_invalid_type_parameter. Retrieved 4/9 statements.
# Partially parsed test_invalid_initial_type. Retrieved 4/9 statements.
# Partially parsed test_non_callable_invariant. Retrieved 4/9 statements.
# Partially parsed test_non_callable_factory. Retrieved 4/9 statements.
# Partially parsed test_non_callable_serializer. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 5
    var_1 = lambda x: True
    var_2 = lambda : None
    var_3 = lambda x: str(x)

def test_case_0():
    var_0 = 5
    var_1 = lambda x: True
    var_2 = lambda : None
    var_3 = lambda x: str(x)
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 5.5
    var_1 = lambda x: True
    var_2 = lambda : None
    var_3 = lambda x: str(x)
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 5
    var_1 = 'not callable'
    var_2 = lambda : None
    var_3 = lambda x: str(x)
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 5
    var_1 = lambda x: True
    var_2 = 'not callable'
    var_3 = lambda x: str(x)
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 5
    var_1 = lambda x: True
    var_2 = lambda : None
    var_3 = 'not callable'
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_at_line_6_evaluates_to_false. Retrieved 2/9 statements.


def test_case_0():
    var_0 = True
    var_1 = 0



# Parsed testcases at query #8
#--------------------------




import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_type'
    var_1 = 'test_invariant'
    var_2 = 'test_initial'
    var_3 = True
    var_4 = 'test_factory'
    var_5 = 'test_serializer'
    var_6 = module_0._PField(var_0, var_1, var_2, var_3, var_4, var_5)
    var_7 = var_6.type
    assert var_7 == 'test_type'
    var_8 = var_6.invariant
    assert var_8 == 'test_invariant'
    var_9 = var_6.initial
    assert var_9 == 'test_initial'
    var_10 = var_6.mandatory
    assert var_10 is True
    var_11 = var_6._factory
    assert var_11 == 'test_factory'
    var_12 = var_6.serializer
    assert var_12 == 'test_serializer'



# Parsed testcases at query #9
#--------------------------

# Failed to parse test_is_type_cls_with_set.
# Partially parsed test_is_type_cls_with_empty_tuple. Retrieved 1/2 statements.
# Failed to parse test_is_type_cls_with_single_type_in_tuple.
# Failed to parse test_is_type_cls_with_multiple_types_in_tuple.
# Failed to parse test_is_type_cls_with_non_matching_type.
# Failed to parse test_is_type_cls_with_type_object.
# Partially parsed test_is_type_cls_with_string_type_name. Retrieved 1/2 statements.
# Partially parsed test_is_type_cls_with_non_matching_string_type_name. Retrieved 1/2 statements.


def test_case_0():
    var_0 = ()

def test_case_0():
    var_0 = 'builtins.ValueError'

def test_case_0():
    var_0 = 'builtins.Exception'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test__make_seq_field_type_returns_existing_type. Retrieved 2/6 statements.
# Partially parsed test__make_seq_field_type_creates_new_type. Retrieved 2/6 statements.
# Partially parsed test__make_seq_field_type_sets_correct_name. Retrieved 2/5 statements.
# Partially parsed test__make_seq_field_type_with_string_type. Retrieved 2/5 statements.
# Partially parsed test__make_seq_field_type_with_custom_type. Retrieved 3/5 statements.
# Partially parsed test__make_seq_field_type_pickle_support. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: len(x) > var_0

def test_case_0():
    var_0 = 'collections.OrderedDict'
    var_1 = True
    var_2 = lambda x: var_1

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]



# Parsed testcases at query #11
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_new_class.
# Failed to parse test_make_pmap_field_type_reuses_existing_class.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'builtins.str'
    var_1 = 'builtins.int'
    var_2 = module_0._make_pmap_field_type(var_0, var_1)
    var_3 = var_2.__name__
    assert var_3 == 'StrToIntPMap'
    var_4 = var_2.__key_type__
    var_5 = var_2.__value_type__



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_serialize_checked_type_with_no_serializer. Retrieved 2/7 statements.


def test_case_0():
    var_0 = None
    var_1 = []



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_factory_property_returns_no_factory_when_type_is_not_checked_type. Retrieved 3/4 statements.


def test_case_0():
    var_0 = ()
    var_1 = None
    var_2 = False



# Parsed testcases at query #14
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_new_class.
# Failed to parse test_make_pmap_field_type_reuses_existing_class.
# Failed to parse test_make_pmap_field_type_with_custom_types.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = 'builtins.str'
    var_2 = module_0._make_pmap_field_type(var_0, var_1)
    var_3 = var_2.__name__
    assert var_3 == 'IntToStrPMap'
    var_4 = var_2.__key_type__
    var_5 = var_2.__value_type__



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 2/7 statements.


def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = 'test_format'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_check_global_invariants_no_errors. Retrieved 5/7 statements.
# Partially parsed test_check_global_invariants_single_error. Retrieved 5/8 statements.
# Partially parsed test_check_global_invariants_multiple_errors. Retrieved 8/11 statements.
# Partially parsed test_check_global_invariants_mixed_results. Retrieved 11/14 statements.


def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = [var_4]

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = 'E1'
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = [var_4]
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = 'E1'
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = 'E2'
    var_6 = (var_1, var_5)
    var_7 = lambda _: var_6
    var_8 = [var_4, var_7]
    var_9 = bool(False)
    assert var_9 is True

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = False
    var_6 = 'E1'
    var_7 = (var_5, var_6)
    var_8 = lambda _: var_7
    var_9 = (var_1, var_2)
    var_10 = lambda _: var_9
    var_11 = [var_4, var_8, var_10]
    var_12 = bool(False)
    assert var_12 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_pfield_initialization. Retrieved 5/7 statements.


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = True
    var_3 = None
    var_4 = 5



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint_with_ignore_extra_false. Retrieved 4/8 statements.
# Partially parsed test_is_field_ignore_extra_complaint_with_non_matching_type. Retrieved 4/8 statements.
# Partially parsed test_is_field_ignore_extra_complaint_with_matching_type_and_no_ignore_extra_param. Retrieved 4/8 statements.
# Partially parsed test_is_field_ignore_extra_complaint_with_matching_type_and_ignore_extra_param. Retrieved 4/8 statements.
# Partially parsed test_is_field_ignore_extra_complaint_with_set_type. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = lambda : var_1
    var_3 = False

def test_case_0():
    var_0 = 'test'
    var_1 = ''
    var_2 = lambda : var_1
    var_3 = True

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = lambda : var_1
    var_3 = True

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = lambda *: var_1
    var_3 = True

def test_case_0():
    var_0 = 'test'
    var_1 = 0
    var_2 = lambda : var_1
    var_3 = True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_check_global_invariants_raises_exception_when_invariant_fails. Retrieved 5/8 statements.


def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = 'ERROR_CODE'
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = [var_4]
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_check_type_with_valid_type. Retrieved 2/8 statements.
# Partially parsed test_check_type_with_invalid_type. Retrieved 2/9 statements.
# Partially parsed test_check_type_with_no_type_specified. Retrieved 3/8 statements.
# Partially parsed test_check_type_with_multiple_valid_types. Retrieved 3/11 statements.
# Partially parsed test_check_type_with_string_type_name. Retrieved 3/8 statements.
# Partially parsed test_check_type_with_invalid_string_type_name. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 123

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'not_an_int'
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = None
    var_1 = 'test_field'
    var_2 = 'any_value'

def test_case_0():
    var_0 = 'test_field'
    var_1 = 123
    var_2 = 'string'

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = (var_0,)
    var_2 = 'test_field'
    var_3 = 123

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = (var_0,)
    var_2 = 'test_field'
    var_3 = 'not_an_int'
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint_returns_false_when_ignore_extra_is_false. Retrieved 1/3 statements.
# Partially parsed test_is_field_ignore_extra_complaint_returns_false_when_field_type_is_not_a_subset_of_type_cls. Retrieved 1/3 statements.
# Partially parsed test_is_field_ignore_extra_complaint_returns_false_when_factory_has_no_ignore_extra_param. Retrieved 2/6 statements.
# Partially parsed test_is_field_ignore_extra_complaint_returns_true_when_all_conditions_are_met. Retrieved 2/6 statements.


def test_case_0():
    var_0 = []
    var_1 = False

def test_case_0():
    var_0 = []
    var_1 = True

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = True

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test__make_seq_field_type_creates_new_type. Retrieved 3/11 statements.
# Partially parsed test__make_seq_field_type_returns_cached_type. Retrieved 3/11 statements.
# Partially parsed test__make_seq_field_type_with_string_type. Retrieved 4/12 statements.


def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = 5

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = lambda x: var_0

def test_case_0():
    var_0 = 'builtins.str'
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = 'test'
    var_4 = 'builtins.str'
    var_5 = (var_4,)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 7/9 statements.


def test_case_0():
    var_0 = 123
    var_1 = [var_0]
    var_2 = None
    var_3 = True
    var_4 = lambda : var_3
    var_5 = lambda : var_3
    var_6 = lambda : var_3



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_check_type_with_valid_type. Retrieved 2/8 statements.
# Partially parsed test_check_type_with_invalid_type. Retrieved 2/9 statements.
# Partially parsed test_check_type_with_no_type_specified. Retrieved 3/8 statements.
# Partially parsed test_check_type_with_multiple_valid_types. Retrieved 3/11 statements.
# Partially parsed test_check_type_with_string_type_name. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 42

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'not_an_int'
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = None
    var_1 = 'test_field'
    var_2 = 'any_value'

def test_case_0():
    var_0 = 'test_field'
    var_1 = 42
    var_2 = 'string'

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = (var_0,)
    var_2 = 'test_field'
    var_3 = 42



# Parsed testcases at query #25
#--------------------------

# Partially parsed test__make_seq_field_type_with_built_in_type. Retrieved 3/7 statements.
# Partially parsed test__make_seq_field_type_with_string_type. Retrieved 5/8 statements.
# Partially parsed test__make_seq_field_type_caching. Retrieved 3/7 statements.
# Partially parsed test__make_seq_field_type_pickle_support. Retrieved 5/10 statements.


def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = 5

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = (var_0,)
    var_2 = 'builtins.int'
    var_3 = True
    var_4 = lambda x: var_3
    var_5 = 5

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = lambda x: var_0

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_2, var_3]



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_check_global_invariants_no_errors. Retrieved 7/9 statements.
# Partially parsed test_check_global_invariants_single_error. Retrieved 9/12 statements.
# Partially parsed test_check_global_invariants_multiple_errors. Retrieved 8/11 statements.


def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = (var_1, var_2)
    var_6 = lambda _: var_5
    var_7 = [var_4, var_6]

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = 'ERROR1'
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = True
    var_6 = None
    var_7 = (var_5, var_6)
    var_8 = lambda _: var_7
    var_9 = [var_4, var_8]
    var_10 = bool(False)
    assert var_10 is True

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = 'ERROR1'
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = 'ERROR2'
    var_6 = (var_1, var_5)
    var_7 = lambda _: var_6
    var_8 = [var_4, var_7]
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_pfield_initialization. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 0
    var_3 = True
    var_4 = None



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_serialize_checked_type_with_no_serializer. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = 'some_format'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_factory_assignment. Retrieved 4/6 statements.


def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = 0
    var_3 = None



# Parsed testcases at query #30
#--------------------------

# Failed to parse test_pmap_field_basic.
# Partially parsed test_pmap_field_optional. Retrieved 1/8 statements.
# Partially parsed test_pmap_field_with_invariant. Retrieved 5/4 statements.
# Partially parsed test_pmap_field_factory_none. Retrieved 2/4 statements.
# Partially parsed test_pmap_field_factory_with_value. Retrieved 5/9 statements.


def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = True
    var_1 = 'test'
    var_2 = (var_0, var_1)
    var_3 = {var_0}
    var_4 = var_2()

def test_case_0():
    var_0 = True
    var_1 = 'test'
    var_2 = (var_0, var_1)
    var_3 = {var_0}
    var_4 = var_2()

def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = set()



# Parsed testcases at query #32
#--------------------------

# Failed to parse test_pmap_field_basic.
# Partially parsed test_pmap_field_optional. Retrieved 5/15 statements.
# Partially parsed test_pmap_field_with_invariant. Retrieved 3/4 statements.
# Partially parsed test_pmap_field_optional_with_invariant. Retrieved 4/4 statements.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = [var_1]
    var_3 = 'a'
    var_4 = {var_0: var_3}
    var_5 = {var_0: var_3}

def test_case_0():
    var_0 = True
    var_1 = 'test'
    var_2 = (var_0, var_1)

def test_case_0():
    var_0 = True
    var_1 = 'test'
    var_2 = (var_0, var_1)

def test_case_0():
    var_0 = True
    var_1 = 'test'
    var_2 = (var_0, var_1)
    var_3 = True

def test_case_0():
    var_0 = True
    var_1 = 'test'
    var_2 = (var_0, var_1)
    var_3 = True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_check_field_parameters_with_valid_type. Retrieved 12/16 statements.
# Partially parsed test_check_field_parameters_with_invalid_type. Retrieved 14/18 statements.
# Partially parsed test_check_field_parameters_with_valid_initial. Retrieved 13/17 statements.
# Partially parsed test_check_field_parameters_with_invalid_initial. Retrieved 13/18 statements.
# Partially parsed test_check_field_parameters_with_non_callable_invariant. Retrieved 11/16 statements.
# Partially parsed test_check_field_parameters_with_non_callable_factory. Retrieved 11/16 statements.
# Partially parsed test_check_field_parameters_with_non_callable_serializer. Retrieved 12/17 statements.


def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'initial'
    var_4 = 'invariant'
    var_5 = 'factory'
    var_6 = 'serializer'
    var_7 = True
    var_8 = lambda x: var_7
    var_9 = None
    var_10 = lambda : var_9
    var_11 = lambda x: x

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'initial'
    var_4 = 'invariant'
    var_5 = 'factory'
    var_6 = 'serializer'
    var_7 = 123
    var_8 = [var_7]
    var_9 = True
    var_10 = lambda x: var_9
    var_11 = None
    var_12 = lambda : var_11
    var_13 = lambda x: x

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'initial'
    var_4 = 'invariant'
    var_5 = 'factory'
    var_6 = 'serializer'
    var_7 = 5
    var_8 = True
    var_9 = lambda x: var_8
    var_10 = None
    var_11 = lambda : var_10
    var_12 = lambda x: x

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'initial'
    var_4 = 'invariant'
    var_5 = 'factory'
    var_6 = 'serializer'
    var_7 = 'string'
    var_8 = True
    var_9 = lambda x: var_8
    var_10 = None
    var_11 = lambda : var_10
    var_12 = lambda x: x

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'initial'
    var_4 = 'invariant'
    var_5 = 'factory'
    var_6 = 'serializer'
    var_7 = 'not callable'
    var_8 = None
    var_9 = lambda : var_8
    var_10 = lambda x: x

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'initial'
    var_4 = 'invariant'
    var_5 = 'factory'
    var_6 = 'serializer'
    var_7 = True
    var_8 = lambda x: var_7
    var_9 = 'not callable'
    var_10 = lambda x: x

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'initial'
    var_4 = 'invariant'
    var_5 = 'factory'
    var_6 = 'serializer'
    var_7 = True
    var_8 = lambda x: var_7
    var_9 = None
    var_10 = lambda : var_9
    var_11 = 'not callable'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_check_type_with_valid_type. Retrieved 2/8 statements.
# Partially parsed test_check_type_with_invalid_type. Retrieved 2/9 statements.
# Partially parsed test_check_type_with_multiple_valid_types. Retrieved 3/11 statements.
# Partially parsed test_check_type_with_no_type_specified. Retrieved 7/16 statements.
# Partially parsed test_check_type_with_string_type_name. Retrieved 3/8 statements.
# Partially parsed test_check_type_with_invalid_string_type_name. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 42

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'not_an_int'

def test_case_0():
    var_0 = 'test_field'
    var_1 = 42
    var_2 = 'a_string'

def test_case_0():
    var_0 = None
    var_1 = 'test_field'
    var_2 = 42
    var_3 = 'a_string'
    var_4 = 'a'
    var_5 = 'dict'
    var_6 = {var_4: var_5}

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = (var_0,)
    var_2 = 'test_field'
    var_3 = 42

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = (var_0,)
    var_2 = 'test_field'
    var_3 = 'not_an_int'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_check_global_invariants_with_valid_invariants. Retrieved 5/7 statements.


def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = [var_4]



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_check_type_with_valid_type. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 42



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = 'some_format'



# Parsed testcases at query #38
#--------------------------

# Failed to parse test_serialize_with_checked_type_and_pfield_no_serializer.




# Parsed testcases at query #39
#--------------------------

# Failed to parse test_check_field_parameters_with_valid_field.
# Failed to parse test_check_field_parameters_with_invalid_type_parameter.
# Failed to parse test_check_field_parameters_with_invalid_initial_type.
# Failed to parse test_check_field_parameters_with_non_callable_invariant.
# Failed to parse test_check_field_parameters_with_non_callable_factory.
# Failed to parse test_check_field_parameters_with_non_callable_serializer.




# Parsed testcases at query #40
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 12/20 statements.


def test_case_0():
    var_0 = 'MockCheckedClass'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = 'MockType'
    var_9 = ()
    var_10 = 'create'
    var_11 = lambda self, data, **kwargs: data
    var_12 = {var_10: var_11}
    var_13 = [var_8, var_9, var_12]



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_check_field_parameters_with_valid_type. Retrieved 5/8 statements.
# Partially parsed test_check_field_parameters_with_invalid_type. Retrieved 7/10 statements.
# Partially parsed test_check_field_parameters_with_valid_initial. Retrieved 6/9 statements.
# Partially parsed test_check_field_parameters_with_invalid_initial. Retrieved 6/10 statements.
# Partially parsed test_check_field_parameters_with_callable_initial. Retrieved 7/10 statements.
# Partially parsed test_check_field_parameters_with_non_callable_invariant. Retrieved 4/8 statements.
# Partially parsed test_check_field_parameters_with_non_callable_factory. Retrieved 4/8 statements.
# Partially parsed test_check_field_parameters_with_non_callable_serializer. Retrieved 5/9 statements.


def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = None
    var_3 = lambda : var_2
    var_4 = lambda x: x

def test_case_0():
    var_0 = 123
    var_1 = [var_0]
    var_2 = True
    var_3 = lambda x: var_2
    var_4 = None
    var_5 = lambda : var_4
    var_6 = lambda x: x

def test_case_0():
    var_0 = 5
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = None
    var_4 = lambda : var_3
    var_5 = lambda x: x

def test_case_0():
    var_0 = 'not an int'
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = None
    var_4 = lambda : var_3
    var_5 = lambda x: x

def test_case_0():
    var_0 = 5
    var_1 = lambda : var_0
    var_2 = True
    var_3 = lambda x: var_2
    var_4 = None
    var_5 = lambda : var_4
    var_6 = lambda x: x

def test_case_0():
    var_0 = 'not callable'
    var_1 = None
    var_2 = lambda : var_1
    var_3 = lambda x: x

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = 'not callable'
    var_3 = lambda x: x

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = None
    var_3 = lambda : var_2
    var_4 = 'not callable'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint_returns_false_when_ignore_extra_is_false. Retrieved 7/10 statements.
# Partially parsed test_is_field_ignore_extra_complaint_returns_false_when_field_type_is_not_subclass. Retrieved 7/10 statements.
# Partially parsed test_is_field_ignore_extra_complaint_returns_false_when_factory_has_no_ignore_extra_param. Retrieved 7/10 statements.
# Partially parsed test_is_field_ignore_extra_complaint_returns_true_when_all_conditions_are_met. Retrieved 5/10 statements.
# Partially parsed test_is_field_ignore_extra_complaint_works_with_set_type. Retrieved 7/11 statements.
# Partially parsed test_is_field_ignore_extra_complaint_works_with_empty_tuple_type. Retrieved 9/11 statements.


def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'factory'
    var_4 = None
    var_5 = lambda : var_4
    var_6 = False

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'factory'
    var_4 = None
    var_5 = lambda : var_4
    var_6 = True

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'factory'
    var_4 = None
    var_5 = lambda : var_4
    var_6 = True

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'factory'
    var_4 = True

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'factory'
    var_4 = None
    var_5 = lambda : var_4
    var_6 = True

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'factory'
    var_4 = tuple()
    var_5 = None
    var_6 = lambda : var_5
    var_7 = {var_2: var_4, var_3: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = set()



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_restore_pmap_field_pickle_calls_restore_pickle_with_correct_args. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_set_fields_with_non_empty_bases_and_empty_dct. Retrieved 4/8 statements.
# Partially parsed test_set_fields_with_pfield_in_dct. Retrieved 5/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'test_name'
    var_3 = module_0.set_fields(var_0, var_1, var_2)
    var_4 = bool(var_0 == {var_2: {}})
    assert var_4 is True

def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = {var_0: var_1}
    var_3 = 'key2'
    var_4 = 'value2'
    var_5 = {var_3: var_4}
    var_6 = {}
    var_7 = 'test_name'
    var_8 = bool(var_6 == {var_7: {'key1': 'value1', 'key2': 'value2'}})
    assert var_8 is True

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'value2'
    var_3 = []
    var_4 = 'test_name'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 10/17 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'MockType'
    var_6 = ()
    var_7 = 'create'
    var_8 = lambda self, data, **kwargs: data
    var_9 = {var_7: var_8}
    var_10 = [var_5, var_6, var_9]



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = []
    var_4 = 'fields'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_valid_field_parameters. Retrieved 4/8 statements.
# Partially parsed test_invalid_type_parameter. Retrieved 4/9 statements.
# Partially parsed test_invalid_initial_type. Retrieved 4/9 statements.
# Partially parsed test_non_callable_invariant. Retrieved 3/9 statements.
# Partially parsed test_non_callable_factory. Retrieved 3/9 statements.
# Partially parsed test_non_callable_serializer. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 10
    var_1 = lambda x: True
    var_2 = lambda : None
    var_3 = lambda x: x

def test_case_0():
    var_0 = 123
    var_1 = [var_0]
    var_2 = lambda x: True
    var_3 = lambda : None
    var_4 = lambda x: x
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 'not an int'
    var_1 = lambda x: True
    var_2 = lambda : None
    var_3 = lambda x: x
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'not callable'
    var_1 = lambda : None
    var_2 = lambda x: x
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = lambda x: True
    var_1 = 'not callable'
    var_2 = lambda x: x
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = lambda x: True
    var_1 = lambda : None
    var_2 = 'not callable'
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint_returns_false_when_ignore_extra_is_true_but_type_cls_not_compatible. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'invalid.type.name'
    var_1 = lambda : None
    var_2 = True



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_check_global_invariants_with_valid_invariants. Retrieved 5/7 statements.


def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = [var_4]



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_serialize_checked_type_with_no_serializer. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = 'some_format'



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_set_fields_predicate. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = []
    var_4 = 'fields'



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_restore_seq_field_pickle_calls_restore_pickle_with_correct_args. Retrieved 12/20 statements.


def test_case_0():
    var_0 = 'TestClass'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = 'TestType'
    var_9 = ()
    var_10 = 'create'
    var_11 = lambda self, data, _factory_fields: data
    var_12 = {var_10: var_11}
    var_13 = [var_8, var_9, var_12]



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 15/19 statements.


def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'initial'
    var_4 = 'invariant'
    var_5 = 'factory'
    var_6 = 'serializer'
    var_7 = 123
    var_8 = [var_7]
    var_9 = 'value'
    var_10 = True
    var_11 = lambda : var_10
    var_12 = lambda : var_10
    var_13 = lambda : var_10
    var_14 = {var_2: var_8, var_3: var_9, var_4: var_11, var_5: var_12, var_6: var_13}
    var_15 = [var_0, var_1, var_14]



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_check_global_invariants_no_errors. Retrieved 7/9 statements.
# Partially parsed test_check_global_invariants_with_errors. Retrieved 12/15 statements.


def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = (var_1, var_2)
    var_6 = lambda _: var_5
    var_7 = [var_4, var_6]

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = False
    var_6 = 'ERROR1'
    var_7 = (var_5, var_6)
    var_8 = lambda _: var_7
    var_9 = 'ERROR2'
    var_10 = (var_5, var_9)
    var_11 = lambda _: var_10
    var_12 = [var_4, var_8, var_11]
    var_13 = bool(False)
    assert var_13 is True



# Parsed testcases at query #56
#--------------------------

# Failed to parse test_pmap_field_docstring_exists.




# Parsed testcases at query #57
#--------------------------

# Partially parsed test_make_seq_field_type_with_builtin_type. Retrieved 2/7 statements.
# Partially parsed test_make_seq_field_type_with_custom_type. Retrieved 4/8 statements.
# Partially parsed test_make_seq_field_type_caching. Retrieved 2/8 statements.
# Partially parsed test_make_seq_field_type_reduce. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'collections.abc.Sequence'
    var_1 = 0
    var_2 = lambda x: len(x) > var_1
    var_3 = module_0.get_type(var_0)

def test_case_0():
    var_0 = 0
    var_1 = lambda x: len(x) > var_0

def test_case_0():
    var_0 = 0.0
    var_1 = lambda x: x != var_0
    var_2 = 1.0
    var_3 = 2.0
    var_4 = 3.0
    var_5 = [var_2, var_3, var_4]



# Parsed testcases at query #58
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_pmap_field_optional_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = False



# Parsed testcases at query #60
#--------------------------

# Partially parsed test__make_seq_field_type_creates_new_type. Retrieved 1/8 statements.
# Partially parsed test__make_seq_field_type_reuses_existing_type. Retrieved 1/6 statements.


def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = None



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_set_fields_with_single_base. Retrieved 3/6 statements.
# Partially parsed test_set_fields_with_multiple_bases. Retrieved 4/8 statements.
# Partially parsed test_set_fields_with_pfield. Retrieved 5/13 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'test_name'
    var_3 = module_0.set_fields(var_0, var_1, var_2)
    var_4 = bool(var_0 == {var_2: {}})
    assert var_4 is True

def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = 'test_name'
    var_5 = bool(var_3 == {var_4: {'key1': 'value1'}})
    assert var_5 is True

def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = {var_0: var_1}
    var_3 = 'key2'
    var_4 = 'value2'
    var_5 = {var_3: var_4}
    var_6 = {}
    var_7 = 'test_name'
    var_8 = bool(var_6 == {var_7: {'key1': 'value1', 'key2': 'value2'}})
    assert var_8 is True

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'value2'
    var_3 = []
    var_4 = 'test_name'



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_check_global_invariants_no_errors. Retrieved 7/9 statements.
# Partially parsed test_check_global_invariants_with_errors. Retrieved 9/12 statements.


def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = (var_1, var_2)
    var_6 = lambda _: var_5
    var_7 = [var_4, var_6]

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = 'ERROR1'
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = True
    var_6 = None
    var_7 = (var_5, var_6)
    var_8 = lambda _: var_7
    var_9 = [var_4, var_8]
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_isinstance_v_pfield. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = []
    var_2 = 'test_name'



