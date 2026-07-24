####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
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



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_restore_seq_field_pickle_calls_restore_pickle. Retrieved 15/23 statements.


def test_case_0():
    var_0 = 'MockCheckedClass'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = 'MockItemType'
    var_5 = ()
    var_6 = {}
    var_7 = [var_4, var_5, var_6]
    var_8 = 'MockType'
    var_9 = ()
    var_10 = 'create'
    var_11 = lambda self, data, _factory_fields: data
    var_12 = {var_10: var_11}
    var_13 = [var_8, var_9, var_12]
    var_14 = 1
    var_15 = 2
    var_16 = 3
    var_17 = [var_14, var_15, var_16]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test__make_seq_field_type_creates_new_type_with_correct_name. Retrieved 2/5 statements.
# Partially parsed test__make_seq_field_type_reuses_existing_type. Retrieved 3/7 statements.
# Partially parsed test__make_seq_field_type_sets_correct_attributes. Retrieved 3/7 statements.
# Partially parsed test__make_seq_field_type_creates_reduce_method. Retrieved 5/10 statements.


def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = lambda x: var_0

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = 5

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_2, var_3]



# Parsed testcases at query #4
#--------------------------

# Failed to parse test_types_to_names_with_single_type.
# Failed to parse test_types_to_names_with_multiple_types.
# Partially parsed test_types_to_names_with_mixed_type_and_string. Retrieved 1/3 statements.


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



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint_returns_false_when_ignore_extra_is_false. Retrieved 2/3 statements.
# Partially parsed test_is_field_ignore_extra_complaint_returns_false_when_field_type_is_not_subclass. Retrieved 2/6 statements.
# Partially parsed test_is_field_ignore_extra_complaint_returns_false_when_factory_has_no_ignore_extra_param. Retrieved 2/6 statements.
# Partially parsed test_is_field_ignore_extra_complaint_returns_true_when_all_conditions_met. Retrieved 2/6 statements.
# Partially parsed test_is_field_ignore_extra_complaint_works_with_set_type. Retrieved 2/6 statements.
# Partially parsed test_is_field_ignore_extra_complaint_works_with_empty_type_tuple. Retrieved 3/6 statements.


def test_case_0():
    var_0 = None
    var_1 = False

def test_case_0():
    var_0 = lambda : None
    var_1 = True

def test_case_0():
    var_0 = lambda : None
    var_1 = True

def test_case_0():
    var_0 = lambda ignore_extra: None
    var_1 = True

def test_case_0():
    var_0 = lambda ignore_extra: None
    var_1 = True

def test_case_0():
    var_0 = ()
    var_1 = lambda ignore_extra: None
    var_2 = True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 1/4 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 2/5 statements.


def test_case_0():
    var_0 = []
    var_1 = 'format'

def test_case_0():
    var_0 = 'format'
    var_1 = 'test_value'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test__make_seq_field_type_creates_new_type. Retrieved 1/6 statements.
# Partially parsed test__make_seq_field_type_reuses_existing_type. Retrieved 1/5 statements.
# Partially parsed test__make_seq_field_type_sets_type_and_invariant. Retrieved 4/9 statements.
# Partially parsed test__make_seq_field_type_reduce_method. Retrieved 5/10 statements.


def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 5
    var_3 = -1

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_check_field_parameters_with_invalid_type_parameter. Retrieved 2/7 statements.
# Partially parsed test_check_field_parameters_with_invalid_initial_type. Retrieved 1/7 statements.
# Partially parsed test_check_field_parameters_with_non_callable_invariant. Retrieved 1/6 statements.
# Partially parsed test_check_field_parameters_with_non_callable_factory. Retrieved 1/6 statements.
# Partially parsed test_check_field_parameters_with_non_callable_serializer. Retrieved 1/6 statements.
# Partially parsed test_check_field_parameters_with_valid_parameters. Retrieved 6/9 statements.


def test_case_0():
    var_0 = 123
    var_1 = [var_0]
    var_2 = 'Type parameter expected, not'

def test_case_0():
    var_0 = 123.45
    var_1 = 'Initial has invalid type'

def test_case_0():
    var_0 = 123
    var_1 = 'Invariant must be callable'

def test_case_0():
    var_0 = 123
    var_1 = 'Factory must be callable'

def test_case_0():
    var_0 = 123
    var_1 = 'Serializer must be callable'

def test_case_0():
    var_0 = 42
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = 0
    var_4 = lambda : var_3
    var_5 = lambda x: str(x)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_set_fields_single_base_with_empty_dict. Retrieved 7/10 statements.
# Partially parsed test_set_fields_single_base_with_items. Retrieved 9/12 statements.
# Partially parsed test_set_fields_multiple_bases. Retrieved 15/19 statements.
# Partially parsed test_set_fields_with_pfield. Retrieved 3/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'test'
    var_3 = module_0.set_fields(var_0, var_1, var_2)
    var_4 = bool(var_0 == {'test': {}})
    assert var_4 is True

def test_case_0():
    var_0 = {}
    var_1 = 'Base'
    var_2 = ()
    var_3 = 'test'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = [var_1, var_2, var_5]
    var_7 = 'test'
    var_8 = bool(var_0 == {'test': {}})
    assert var_8 is True

def test_case_0():
    var_0 = {}
    var_1 = 'Base'
    var_2 = ()
    var_3 = 'test'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = [var_1, var_2, var_7]
    var_9 = 'test'
    var_10 = bool(var_0 == {'test': {'key': 'value'}})
    assert var_10 is True

def test_case_0():
    var_0 = {}
    var_1 = 'Base1'
    var_2 = ()
    var_3 = 'test'
    var_4 = 'key1'
    var_5 = 'value1'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = [var_1, var_2, var_7]
    var_9 = 'Base2'
    var_10 = ()
    var_11 = 'key2'
    var_12 = 'value2'
    var_13 = {var_11: var_12}
    var_14 = {var_3: var_13}
    var_15 = [var_9, var_10, var_14]
    var_16 = 'test'
    var_17 = bool(var_0 == {'test': {'key1': 'value1', 'key2': 'value2'}})
    assert var_17 is True

def test_case_0():
    var_0 = 'pf'
    var_1 = []
    var_2 = 'test'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_sequence_field_with_checked_class_and_item_type. Retrieved 6/11 statements.
# Partially parsed test_sequence_field_with_optional_true. Retrieved 8/18 statements.
# Partially parsed test_sequence_field_with_invariant. Retrieved 9/14 statements.
# Partially parsed test_sequence_field_with_item_invariant. Retrieved 7/12 statements.


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
    var_3 = [var_1, var_2]
    var_4 = None
    var_5 = [var_4]
    var_6 = [var_1, var_2]
    var_7 = [var_1, var_2]
    var_8 = [var_1, var_2]

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 0
    var_6 = lambda x: x > var_5
    var_7 = module_0.wrap_invariant(var_6)
    var_8 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = False
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = 0
    var_5 = lambda x: len(x) > var_4
    var_6 = [var_1, var_2]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_check_type_with_valid_type. Retrieved 2/8 statements.
# Partially parsed test_check_type_with_invalid_type. Retrieved 2/9 statements.
# Partially parsed test_check_type_with_no_type_specified. Retrieved 3/8 statements.
# Partially parsed test_check_type_with_multiple_valid_types. Retrieved 3/11 statements.
# Partially parsed test_check_type_with_string_type_name. Retrieved 3/8 statements.
# Partially parsed test_check_type_with_string_type_name_invalid. Retrieved 3/9 statements.


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
    var_2 = 'a_string'

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
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #12
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_new_class.
# Failed to parse test_make_pmap_field_type_reuses_existing_class.
# Partially parsed test_make_pmap_field_type_with_string_type_names. Retrieved 3/4 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'builtins.str'
    var_1 = 'builtins.int'
    var_2 = module_0._make_pmap_field_type(var_0, var_1)
    var_3 = var_2.__key_type__
    var_4 = var_2.__value_type__
    var_5 = var_2.__name__
    assert var_5 == 'StrToIntPMap'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_check_field_parameters_with_valid_types. Retrieved 6/9 statements.
# Partially parsed test_check_field_parameters_with_invalid_type_parameter. Retrieved 7/10 statements.
# Partially parsed test_check_field_parameters_with_invalid_initial_type. Retrieved 6/10 statements.
# Partially parsed test_check_field_parameters_with_non_callable_invariant. Retrieved 5/9 statements.
# Partially parsed test_check_field_parameters_with_non_callable_factory. Retrieved 5/9 statements.
# Partially parsed test_check_field_parameters_with_non_callable_serializer. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 42
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = None
    var_4 = lambda : var_3
    var_5 = lambda x: x

def test_case_0():
    var_0 = 42
    var_1 = [var_0]
    var_2 = None
    var_3 = True
    var_4 = lambda x: var_3
    var_5 = lambda : var_2
    var_6 = lambda x: x

def test_case_0():
    var_0 = 'not an int'
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = None
    var_4 = lambda : var_3
    var_5 = lambda x: x

def test_case_0():
    var_0 = 42
    var_1 = 'not callable'
    var_2 = None
    var_3 = lambda : var_2
    var_4 = lambda x: x

def test_case_0():
    var_0 = 42
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = 'not callable'
    var_4 = lambda x: x

def test_case_0():
    var_0 = 42
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = None
    var_4 = lambda : var_3
    var_5 = 'not callable'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_invariant_default_value. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 0
    var_1 = False
    var_2 = []



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_pfield_constructor. Retrieved 6/8 statements.


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 0
    var_3 = True
    var_4 = lambda : var_0
    var_5 = lambda x: str(x)



# Parsed testcases at query #16
#--------------------------

# Failed to parse test_pmap_field_basic.
# Partially parsed test_pmap_field_optional. Retrieved 2/11 statements.
# Failed to parse test_pmap_field_with_invariant.
# Partially parsed test_pmap_field_optional_with_invariant. Retrieved 2/15 statements.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = [var_1]

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = [var_1]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_sequence_field_optional_predicate. Retrieved 2/3 statements.


def test_case_0():
    var_0 = True
    var_1 = []



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_pmap_field_with_optional_true_returns_correct_type. Retrieved 1/4 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_restore_pmap_field_pickle_calls_restore_pickle_with_correct_args. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #20
#--------------------------

# Failed to parse test_serialize_checked_type_with_no_serializer.




# Parsed testcases at query #21
#--------------------------

# Failed to parse test_pmap_field_with_non_optional_and_no_invariant.
# Partially parsed test_pmap_field_with_optional_and_no_invariant. Retrieved 5/15 statements.
# Partially parsed test_pmap_field_with_non_optional_and_invariant. Retrieved 2/12 statements.
# Partially parsed test_pmap_field_with_optional_and_invariant. Retrieved 6/19 statements.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = [var_1]
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = {var_3: var_0}

def test_case_0():
    var_0 = True
    var_1 = 'OK'

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = [var_1]
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = {var_3: var_0}
    var_6 = 'OK'



# Parsed testcases at query #22
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



# Parsed testcases at query #23
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
    var_9 = None
    var_10 = True
    var_11 = lambda : var_10
    var_12 = lambda : var_9
    var_13 = lambda x: x
    var_14 = {var_2: var_8, var_3: var_9, var_4: var_11, var_5: var_12, var_6: var_13}
    var_15 = [var_0, var_1, var_14]
    var_16 = bool(False)
    assert var_16 is True



# Parsed testcases at query #24
#--------------------------




import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda : var_0
    var_2 = module_0._PField(var_0, var_0, var_0, var_0, var_1, var_0)
    var_3 = var_2._factory
    var_4 = bool(var_2._factory is var_1)
    assert var_4 is True



# Parsed testcases at query #25
#--------------------------




import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = True
    var_2 = 'error1'
    var_3 = (var_1, var_2)
    var_4 = lambda s: var_3
    var_5 = 'error2'
    var_6 = (var_1, var_5)
    var_7 = lambda s: var_6
    var_8 = [var_4, var_7]
    var_9 = module_0.check_global_invariants(var_0, var_8)
    assert var_9 is None



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_pmap_field_optional_predicate. Retrieved 1/5 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_check_type_with_valid_type. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'field_name'
    var_1 = 42



# Parsed testcases at query #28
#--------------------------

# Failed to parse test_serialize_with_checked_type_and_pfield_no_serializer.




# Parsed testcases at query #29
#--------------------------

# Partially parsed test_check_global_invariants_no_errors. Retrieved 7/9 statements.
# Partially parsed test_check_global_invariants_single_error. Retrieved 9/12 statements.
# Partially parsed test_check_global_invariants_multiple_errors. Retrieved 8/11 statements.
# Partially parsed test_check_global_invariants_empty_invariants. Retrieved 1/3 statements.


def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = (var_1, var_2)
    var_6 = lambda x: var_5
    var_7 = [var_4, var_6]

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = False
    var_6 = 'ERROR1'
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = [var_4, var_8]

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

def test_case_0():
    var_0 = []
    var_1 = []



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_pfield_constructor_initialization. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = True
    var_4 = lambda : var_0



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_set_fields_predicate. Retrieved 6/15 statements.


def test_case_0():
    var_0 = 'value'
    var_1 = 'another_value'
    var_2 = {}
    var_3 = '__pfields__'
    var_4 = 'field1'
    var_5 = var_2[var_3][var_4]



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_check_global_invariants_with_no_errors. Retrieved 5/7 statements.


def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = [var_4]



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_restore_seq_field_pickle_calls_restore_pickle_with_correct_args. Retrieved 12/20 statements.


def test_case_0():
    var_0 = 'MockCheckedClass'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = 'MockItemType'
    var_5 = ()
    var_6 = {}
    var_7 = [var_4, var_5, var_6]
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = 'MockType'
    var_12 = ()
    var_13 = {}
    var_14 = [var_11, var_12, var_13]



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_pmap_field_optional_factory_returns_none. Retrieved 2/4 statements.


def test_case_0():
    var_0 = True
    var_1 = None



# Parsed testcases at query #35
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
    var_1 = 'not an int'
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = None
    var_1 = 'test_field'
    var_2 = 'any value'

def test_case_0():
    var_0 = 'test_field'
    var_1 = 42
    var_2 = 3.14

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = (var_0,)
    var_2 = 'test_field'
    var_3 = 42



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_check_type_with_valid_type. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 42



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_pfield_initialization. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 0
    var_3 = True
    var_4 = None



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_serialize_checked_type_with_no_serializer. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = 'some_format'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_set_fields_with_single_base. Retrieved 3/6 statements.
# Partially parsed test_set_fields_with_multiple_bases. Retrieved 4/8 statements.
# Partially parsed test_set_fields_with_pfield. Retrieved 6/16 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'test_name'
    var_3 = module_0.set_fields(var_0, var_1, var_2)
    var_4 = bool(var_0 == {'test_name': {}})
    assert var_4 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = 'test_name'
    var_5 = bool(var_3 == {'test_name': {'a': 1}})
    assert var_5 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = {}
    var_7 = 'test_name'
    var_8 = bool(var_6 == {'test_name': {'a': 1, 'b': 2}})
    assert var_8 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'field'
    var_4 = 'test_name'
    var_5 = 'test_name'
    var_6 = 'a'
    var_7 = 1



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_check_field_parameters_with_non_callable_invariant. Retrieved 12/18 statements.


def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'initial'
    var_4 = 'invariant'
    var_5 = 'factory'
    var_6 = 'serializer'
    var_7 = 5
    var_8 = 'not_callable'
    var_9 = None
    var_10 = lambda : var_9
    var_11 = lambda x: x
    var_12 = bool(False)
    assert var_12 is True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_pfield_constructor_with_all_parameters. Retrieved 6/8 statements.
# Partially parsed test_pfield_constructor_with_none_factory. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 0
    var_3 = True
    var_4 = lambda : var_0
    var_5 = lambda x: str(x)

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 0
    var_3 = True
    var_4 = lambda x: str(x)



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_set_fields_with_non_empty_bases. Retrieved 8/12 statements.
# Partially parsed test_set_fields_with_pfield_instances. Retrieved 6/13 statements.
# Partially parsed test_set_fields_with_overlapping_fields_in_bases. Retrieved 6/10 statements.
# Partially parsed test_set_fields_with_no_fields_in_bases. Retrieved 6/12 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = 'fields'
    var_7 = module_0.set_fields(var_4, var_5, var_6)
    var_8 = bool(var_4 == {'a': 1, 'b': 2, 'fields': {}})
    assert var_8 is True

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'y'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = 'a'
    var_7 = 'b'
    var_8 = 1
    var_9 = 2
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 'fields'
    var_12 = bool(var_10 == {'a': 1, 'b': 2, 'fields': {'x': 1, 'y': 2}})
    assert var_12 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 2
    var_3 = []
    var_4 = 'fields'
    var_5 = 'fields'

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'x'
    var_4 = 'y'
    var_5 = 2
    var_6 = 3
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'a'
    var_9 = 1
    var_10 = {var_8: var_9}
    var_11 = 'fields'
    var_12 = bool(var_10 == {'a': 1, 'fields': {'x': 2, 'y': 3}})
    assert var_12 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'fields'
    var_6 = bool(var_4 == {'a': 1, 'b': 2, 'fields': {}})
    assert var_6 is True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 10/19 statements.


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
    var_10 = {}
    var_11 = [var_8, var_9, var_10]



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_pfield_initialization. Retrieved 4/6 statements.


def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = 0
    var_3 = None



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_valid_field_parameters. Retrieved 6/9 statements.
# Partially parsed test_invalid_type_parameter. Retrieved 8/11 statements.
# Partially parsed test_invalid_initial_type. Retrieved 6/10 statements.
# Partially parsed test_non_callable_invariant. Retrieved 5/9 statements.
# Partially parsed test_non_callable_factory. Retrieved 5/9 statements.
# Partially parsed test_non_callable_serializer. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 5
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = 0
    var_4 = lambda : var_3
    var_5 = lambda x: str(x)

def test_case_0():
    var_0 = 123
    var_1 = [var_0]
    var_2 = 5
    var_3 = True
    var_4 = lambda x: var_3
    var_5 = 0
    var_6 = lambda : var_5
    var_7 = lambda x: str(x)

def test_case_0():
    var_0 = 'not an int'
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = 0
    var_4 = lambda : var_3
    var_5 = lambda x: str(x)

def test_case_0():
    var_0 = 5
    var_1 = 'not callable'
    var_2 = 0
    var_3 = lambda : var_2
    var_4 = lambda x: str(x)

def test_case_0():
    var_0 = 5
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = 'not callable'
    var_4 = lambda x: str(x)

def test_case_0():
    var_0 = 5
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = 0
    var_4 = lambda : var_3
    var_5 = 'not callable'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test__check_field_parameters_with_invalid_initial_type. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 'not_an_int'
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = None
    var_4 = lambda : var_3
    var_5 = lambda x: x



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_pmap_field_optional_factory_with_none. Retrieved 2/4 statements.


def test_case_0():
    var_0 = True
    var_1 = None



# Parsed testcases at query #48
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_new_subclass.
# Failed to parse test_make_pmap_field_type_reuses_existing_subclass.
# Partially parsed test_make_pmap_field_type_with_string_type_names. Retrieved 3/4 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = 'builtins.str'
    var_2 = module_0._make_pmap_field_type(var_0, var_1)
    var_3 = var_2.__key_type__
    var_4 = var_2.__value_type__
    var_5 = var_2.__name__
    assert var_5 == 'IntToStrPMap'



# Parsed testcases at query #49
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



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_sequence_field_optional_predicate. Retrieved 5/10 statements.


def test_case_0():
    var_0 = True
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_check_field_parameters_with_non_callable_invariant. Retrieved 12/18 statements.


def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'initial'
    var_4 = 'invariant'
    var_5 = 'factory'
    var_6 = 'serializer'
    var_7 = 0
    var_8 = 123
    var_9 = None
    var_10 = lambda : var_9
    var_11 = lambda x: x



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_check_type_with_valid_type. Retrieved 2/8 statements.
# Partially parsed test_check_type_with_invalid_type. Retrieved 2/9 statements.
# Partially parsed test_check_type_with_no_type_specified. Retrieved 3/8 statements.
# Partially parsed test_check_type_with_multiple_valid_types. Retrieved 3/11 statements.
# Partially parsed test_check_type_with_string_type_name. Retrieved 3/8 statements.
# Partially parsed test_check_type_with_string_type_name_invalid. Retrieved 3/9 statements.


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
    var_2 = 'string_value'

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
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_pfield_constructor. Retrieved 6/8 statements.


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 0
    var_3 = True
    var_4 = lambda : var_0
    var_5 = lambda x: str(x)



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 8/15 statements.


def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = (var_0, var_1)
    var_3 = 2
    var_4 = 'b'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = set()



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint_with_false_ignore_extra. Retrieved 3/11 statements.
# Partially parsed test_is_field_ignore_extra_complaint_with_non_matching_type. Retrieved 3/13 statements.
# Partially parsed test_is_field_ignore_extra_complaint_without_ignore_extra_in_factory. Retrieved 3/11 statements.
# Partially parsed test_is_field_ignore_extra_complaint_with_ignore_extra_in_factory. Retrieved 3/11 statements.
# Partially parsed test_is_field_ignore_extra_complaint_with_set_type. Retrieved 3/12 statements.


def test_case_0():
    var_0 = None
    var_1 = lambda : var_0
    var_2 = False

def test_case_0():
    var_0 = None
    var_1 = lambda : var_0
    var_2 = True

def test_case_0():
    var_0 = None
    var_1 = lambda : var_0
    var_2 = True

def test_case_0():
    var_0 = None
    var_1 = lambda ignore_extra: var_0
    var_2 = True

def test_case_0():
    var_0 = None
    var_1 = lambda ignore_extra: var_0
    var_2 = True



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_all_invariants_pass. Retrieved 1/7 statements.
# Partially parsed test_single_invariant_fails. Retrieved 1/6 statements.
# Partially parsed test_multiple_invariants_fail. Retrieved 1/8 statements.
# Partially parsed test_mixed_invariants. Retrieved 1/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'subject'
    var_1 = []
    var_2 = module_0.check_global_invariants(var_0, var_1)

def test_case_0():
    var_0 = 'subject'

def test_case_0():
    var_0 = 'subject'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 'subject'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 'subject'
    var_1 = bool(False)
    assert var_1 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_pfield_constructor_initialization. Retrieved 7/9 statements.


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 0
    var_3 = True
    var_4 = 42
    var_5 = lambda : var_4
    var_6 = lambda x: str(x)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test__make_seq_field_type_creates_subclass_with_correct_name. Retrieved 2/6 statements.
# Partially parsed test__make_seq_field_type_reuses_existing_type. Retrieved 3/7 statements.
# Partially parsed test__make_seq_field_type_sets_type_and_invariant. Retrieved 4/9 statements.
# Partially parsed test__make_seq_field_type_implements_reduce. Retrieved 4/10 statements.


def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = lambda x: var_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 5.0
    var_3 = -1.0

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = 'a'
    var_3 = 'b'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_check_type_with_valid_type. Retrieved 5/12 statements.
# Partially parsed test_check_type_with_invalid_type. Retrieved 5/13 statements.
# Partially parsed test_check_type_with_no_type_specified. Retrieved 7/12 statements.
# Partially parsed test_check_type_with_string_type_name. Retrieved 9/14 statements.


def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'test_field'
    var_4 = 123

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'test_field'
    var_4 = 123.45

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = None
    var_4 = {var_2: var_3}
    var_5 = [var_0, var_1, var_4]
    var_6 = 'test_field'
    var_7 = 123.45

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'builtins.int'
    var_4 = 'builtins.str'
    var_5 = (var_3, var_4)
    var_6 = {var_2: var_5}
    var_7 = [var_0, var_1, var_6]
    var_8 = 'test_field'
    var_9 = 'test'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_check_field_parameters_with_valid_field. Retrieved 6/9 statements.
# Partially parsed test_check_field_parameters_with_invalid_type_parameter. Retrieved 6/10 statements.
# Partially parsed test_check_field_parameters_with_invalid_initial_type. Retrieved 6/10 statements.
# Partially parsed test_check_field_parameters_with_non_callable_invariant. Retrieved 4/8 statements.
# Partially parsed test_check_field_parameters_with_non_callable_factory. Retrieved 4/8 statements.
# Partially parsed test_check_field_parameters_with_non_callable_serializer. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 42
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = None
    var_4 = lambda : var_3
    var_5 = lambda x: x

def test_case_0():
    var_0 = 42
    var_1 = None
    var_2 = True
    var_3 = lambda x: var_2
    var_4 = lambda : var_1
    var_5 = lambda x: x

def test_case_0():
    var_0 = 42.0
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = None
    var_4 = lambda : var_3
    var_5 = lambda x: x

def test_case_0():
    var_0 = 42
    var_1 = None
    var_2 = lambda : var_1
    var_3 = lambda x: x

def test_case_0():
    var_0 = 42
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = lambda x: x

def test_case_0():
    var_0 = 42
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = None
    var_4 = lambda : var_3



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_check_field_parameters_with_non_callable_invariant. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 123



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint_returns_false_when_ignore_extra_is_false. Retrieved 1/3 statements.
# Partially parsed test_is_field_ignore_extra_complaint_returns_false_when_field_type_is_not_subclass. Retrieved 1/3 statements.
# Partially parsed test_is_field_ignore_extra_complaint_returns_false_when_factory_has_no_ignore_extra_param. Retrieved 2/4 statements.
# Partially parsed test_is_field_ignore_extra_complaint_returns_true_when_all_conditions_met. Retrieved 2/4 statements.
# Partially parsed test_is_field_ignore_extra_complaint_returns_true_for_set_type. Retrieved 1/4 statements.
# Partially parsed test_is_field_ignore_extra_complaint_returns_false_for_empty_tuple_type. Retrieved 2/4 statements.


def test_case_0():
    var_0 = False

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = lambda x: x
    var_1 = True

def test_case_0():
    var_0 = lambda x, ignore_extra=False: x
    var_1 = True

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = ()
    var_1 = True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test__make_seq_field_type_creates_new_type. Retrieved 3/9 statements.
# Partially parsed test__make_seq_field_type_reuses_existing_type. Retrieved 3/7 statements.
# Partially parsed test__make_seq_field_type_with_different_types. Retrieved 2/6 statements.


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



# Parsed testcases at query #8
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



# Parsed testcases at query #9
#--------------------------

# Partially parsed test__check_field_parameters_with_non_callable_invariant. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 123



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_check_field_parameters_with_valid_types. Retrieved 13/18 statements.


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
    var_12 = lambda x: str(x)



# Parsed testcases at query #11
#--------------------------

# Failed to parse test_pmap_field_creates_checked_pmap_field.
# Partially parsed test_pmap_field_with_optional. Retrieved 5/15 statements.
# Partially parsed test_pmap_field_with_invariant. Retrieved 2/7 statements.
# Partially parsed test_pmap_field_with_optional_and_invariant. Retrieved 6/16 statements.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = [var_1]
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = {var_3: var_0}

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = None
    var_3 = [var_2]
    var_4 = 'a'
    var_5 = {var_4: var_0}
    var_6 = {var_4: var_0}



# Parsed testcases at query #12
#--------------------------

# Failed to parse test__make_pmap_field_type_creates_new_class_with_correct_name.
# Failed to parse test__make_pmap_field_type_returns_cached_class.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'builtins.str'
    var_1 = 'builtins.int'
    var_2 = module_0._make_pmap_field_type(var_0, var_1)
    var_3 = var_2.__name__
    assert var_3 == 'StrToIntPMap'
    var_4 = var_2.__key_type__
    var_5 = var_2.__value_type__



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_check_field_parameters_with_non_callable_invariant. Retrieved 11/17 statements.


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



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_set_fields_merges_base_class_fields. Retrieved 4/8 statements.
# Partially parsed test_set_fields_moves_pfield_instances_to_fields_dict. Retrieved 5/12 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'b'
    var_6 = 'c'
    var_7 = 3
    var_8 = 4
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {}
    var_11 = 'fields'
    var_12 = var_10['fields']
    var_13 = bool(var_10['fields'] == {'a': 1, 'b': 3, 'c': 4})
    assert var_13 is True

def test_case_0():
    var_0 = {}
    var_1 = 'x'
    var_2 = 'y'
    var_3 = 'not a field'
    var_4 = 'fields'
    var_5 = 'x'
    var_6 = 'x'
    var_7 = 'y'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_set_fields_predicate_false. Retrieved 6/9 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'a'
    var_5 = var_4 in var_2



# Parsed testcases at query #16
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_new_class.
# Failed to parse test_make_pmap_field_type_reuses_existing_class.
# Failed to parse test_make_pmap_field_type_sets_correct_name.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'builtins.str'
    var_1 = 'builtins.int'
    var_2 = module_0._make_pmap_field_type(var_0, var_1)
    var_3 = var_2.__key_type__
    var_4 = var_2.__value_type__
    var_5 = var_2.__name__
    assert var_5 == 'StrToIntPMap'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_pfield_constructor_initialization. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 0
    var_3 = True
    var_4 = None



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_check_type_with_valid_type. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'field_name'
    var_1 = 42



# Parsed testcases at query #19
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_new_type.
# Failed to parse test_make_pmap_field_type_reuses_existing_type.
# Partially parsed test_make_pmap_field_type_pickle_support. Retrieved 5/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = 'builtins.str'
    var_2 = module_0._make_pmap_field_type(var_0, var_1)
    var_3 = var_2.__name__
    assert var_3 == 'IntToStrPMap'
    var_4 = var_2.__key_type__
    var_5 = var_2.__value_type__

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #20
#--------------------------

# Failed to parse test_pmap_field_docstring_exists.




# Parsed testcases at query #21
#--------------------------

# Partially parsed test_restore_seq_field_pickle_returns_correct_type. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_sequence_field_creates_checked_class. Retrieved 3/6 statements.
# Partially parsed test_sequence_field_with_optional. Retrieved 4/9 statements.
# Partially parsed test_sequence_field_with_initial_none. Retrieved 2/6 statements.
# Partially parsed test_sequence_field_with_invariant. Retrieved 3/8 statements.
# Partially parsed test_sequence_field_with_item_invariant. Retrieved 3/8 statements.
# Partially parsed test_sequence_field_with_checked_pvector. Retrieved 3/6 statements.
# Partially parsed test_sequence_field_with_optional_checked_pvector. Retrieved 4/9 statements.
# Partially parsed test_sequence_field_with_custom_initial. Retrieved 6/9 statements.


def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = None
    var_3 = [var_2]
    var_4 = []

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = [var_1]

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = None
    var_3 = [var_2]
    var_4 = []

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = [var_1, var_2, var_3]



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_check_type_with_valid_type. Retrieved 2/8 statements.
# Partially parsed test_check_type_with_invalid_type. Retrieved 2/9 statements.
# Partially parsed test_check_type_with_no_type_specified. Retrieved 3/8 statements.
# Partially parsed test_check_type_with_multiple_valid_types. Retrieved 2/8 statements.
# Partially parsed test_check_type_with_string_type_name. Retrieved 3/8 statements.
# Partially parsed test_check_type_with_string_type_name_invalid. Retrieved 6/12 statements.


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
    var_1 = 3.14

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = (var_0,)
    var_2 = 'test_field'
    var_3 = 100

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = (var_0,)
    var_2 = 'test_field'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_sequence_field_optional_false. Retrieved 10/13 statements.


def test_case_0():
    var_0 = 'CheckedClass'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = False
    var_5 = 'a'
    var_6 = 'b'
    var_7 = [var_5, var_6]
    var_8 = True
    var_9 = lambda x: var_8
    var_10 = lambda x: var_8



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_sequence_field_optional_false. Retrieved 8/11 statements.


def test_case_0():
    var_0 = 'CheckedClass'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = False
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = [var_5, var_6, var_7]



# Parsed testcases at query #26
#--------------------------




import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = None
    var_1 = 'test_factory'
    var_2 = module_0._PField(var_0, var_0, var_0, var_0, var_1, var_0)
    var_3 = var_2._factory
    assert var_3 == 'test_factory'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 2/5 statements.


def test_case_0():
    var_0 = []
    var_1 = 'serialized'
    var_2 = 'format'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda fmt, val: f'{fmt}:{val}'
    var_1 = 'json'
    var_2 = 'data'
    var_3 = module_0.serialize(var_0, var_1, var_2)
    assert var_3 == 'json:data'



# Parsed testcases at query #28
#--------------------------

# Failed to parse test_pmap_field_basic.
# Partially parsed test_pmap_field_optional. Retrieved 5/16 statements.
# Partially parsed test_pmap_field_with_invariant. Retrieved 5/13 statements.
# Partially parsed test_pmap_field_optional_with_invariant. Retrieved 9/20 statements.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = [var_1]
    var_3 = 'a'
    var_4 = {var_0: var_3}
    var_5 = {var_0: var_3}

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = True
    var_1 = 'OK'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = module_0.wrap_invariant(var_3)

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = True
    var_1 = 'OK'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = None
    var_5 = [var_4]
    var_6 = 'a'
    var_7 = {var_0: var_6}
    var_8 = {var_0: var_6}
    var_9 = module_0.wrap_invariant(var_3)



# Parsed testcases at query #29
#--------------------------




def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'test_name'
    var_3 = bool(not (var_0 and var_1 and var_2))
    assert var_3 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_sequence_field_optional_false. Retrieved 13/16 statements.


def test_case_0():
    var_0 = 'CheckedClass'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = False
    var_5 = 'a'
    var_6 = 'b'
    var_7 = [var_5, var_6]
    var_8 = True
    var_9 = None
    var_10 = (var_8, var_9)
    var_11 = lambda x: var_10
    var_12 = (var_8, var_9)
    var_13 = lambda x: var_12



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_pmap_field_optional_type_predicate. Retrieved 1/7 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint_returns_false_when_ignore_extra_is_false. Retrieved 2/3 statements.
# Partially parsed test_is_field_ignore_extra_complaint_returns_false_when_type_mismatch. Retrieved 2/3 statements.


def test_case_0():
    var_0 = None
    var_1 = False

def test_case_0():
    var_0 = None
    var_1 = True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_is_type_cls_with_set_field_type. Retrieved 1/2 statements.
# Partially parsed test_is_type_cls_with_empty_tuple. Retrieved 1/2 statements.
# Failed to parse test_is_type_cls_with_non_empty_tuple_and_valid_subclass.
# Failed to parse test_is_type_cls_with_non_empty_tuple_and_invalid_subclass.
# Failed to parse test_is_type_cls_with_type_directly.
# Failed to parse test_is_type_cls_with_type_directly_and_invalid_subclass.


def test_case_0():
    var_0 = set()

def test_case_0():
    var_0 = ()



# Parsed testcases at query #34
#--------------------------

# Partially parsed test__make_seq_field_type_creates_new_type. Retrieved 1/4 statements.
# Partially parsed test__make_seq_field_type_reuses_existing_type. Retrieved 1/5 statements.
# Partially parsed test__make_seq_field_type_with_different_item_type. Retrieved 1/5 statements.
# Failed to parse test__make_seq_field_type_with_invariant.
# Partially parsed test__make_seq_field_type_pickle_support. Retrieved 5/14 statements.


def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]



# Parsed testcases at query #35
#--------------------------

# Failed to parse test_pmap_field_basic.
# Partially parsed test_pmap_field_optional. Retrieved 5/15 statements.
# Partially parsed test_pmap_field_with_invariant. Retrieved 2/12 statements.
# Partially parsed test_pmap_field_optional_with_invariant. Retrieved 6/19 statements.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = [var_1]
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = {var_3: var_0}

def test_case_0():
    var_0 = True
    var_1 = 'test'

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = [var_1]
    var_3 = 'a'
    var_4 = {var_3: var_0}
    var_5 = {var_3: var_0}
    var_6 = 'test'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_check_global_invariants_no_errors. Retrieved 7/9 statements.
# Partially parsed test_check_global_invariants_with_errors. Retrieved 12/15 statements.


def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = (var_1, var_2)
    var_6 = lambda x: var_5
    var_7 = [var_4, var_6]

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = 'E1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = True
    var_6 = None
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = 'E2'
    var_10 = (var_1, var_9)
    var_11 = lambda x: var_10
    var_12 = [var_4, var_8, var_11]
    var_13 = bool(False)
    assert var_13 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_restore_pmap_field_pickle_calls_restore_pickle_with_correct_args. Retrieved 10/16 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'MockType'
    var_6 = ()
    var_7 = 'create'
    var_8 = lambda self, data, _factory_fields: data
    var_9 = {var_7: var_8}
    var_10 = [var_5, var_6, var_9]



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint_returns_false_when_ignore_extra_is_false. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = False



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = 'some_format'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_pfield_constructor. Retrieved 6/8 statements.


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 0
    var_3 = True
    var_4 = lambda : var_0
    var_5 = lambda x: x



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_check_global_invariants_with_empty_invariants. Retrieved 1/3 statements.


def test_case_0():
    var_0 = []
    var_1 = []



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #43
#--------------------------




import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = True
    var_2 = 'OK1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'OK2'
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = [var_4, var_7]
    var_9 = module_0.check_global_invariants(var_0, var_8)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = False
    var_2 = 'ERROR1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = True
    var_6 = 'OK2'
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = [var_4, var_8]
    var_10 = module_0.check_global_invariants(var_0, var_9)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = False
    var_2 = 'ERROR1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'ERROR2'
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = [var_4, var_7]
    var_9 = module_0.check_global_invariants(var_0, var_8)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = []
    var_2 = module_0.check_global_invariants(var_0, var_1)



# Parsed testcases at query #44
#--------------------------




import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'valid_subject'
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda s: var_3
    var_5 = [var_4]
    var_6 = module_0.check_global_invariants(var_0, var_5)
    assert var_6 is None



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_set_fields_predicate_false. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = 'new_key'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_check_global_invariants_no_errors. Retrieved 5/7 statements.
# Partially parsed test_check_global_invariants_with_errors. Retrieved 9/12 statements.
# Partially parsed test_check_global_invariants_multiple_errors. Retrieved 8/11 statements.


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



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_sequence_field_optional_predicate. Retrieved 2/3 statements.


def test_case_0():
    var_0 = True
    var_1 = []



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_predicate_at_line_4_evaluates_to_false. Retrieved 4/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'test_name'
    var_3 = module_0.set_fields(var_0, var_1, var_2)



# Parsed testcases at query #49
#--------------------------

# Partially parsed test__PField__init__assigns__factory_to_factory_parameter. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 0
    var_3 = True
    var_4 = lambda : var_0



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_sequence_field_checked_pset. Retrieved 3/5 statements.
# Partially parsed test_sequence_field_checked_pvector. Retrieved 3/5 statements.
# Partially parsed test_sequence_field_optional. Retrieved 2/5 statements.
# Partially parsed test_sequence_field_with_invariant. Retrieved 3/7 statements.
# Partially parsed test_sequence_field_with_item_invariant. Retrieved 3/7 statements.


def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = [var_1]

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = []
    var_4 = 'test'



# Parsed testcases at query #52
#--------------------------

# Failed to parse test_pmap_field_docstring_exists.




# Parsed testcases at query #53
#--------------------------

# Partially parsed test_pmap_field_optional_false. Retrieved 1/4 statements.


def test_case_0():
    var_0 = False



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = set()



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_check_global_invariants_with_valid_invariants. Retrieved 5/7 statements.


def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = [var_4]



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 12/19 statements.


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



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 15/22 statements.


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



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_check_global_invariants_with_no_errors. Retrieved 5/7 statements.


def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = [var_4]



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_check_global_invariants_with_valid_subject. Retrieved 7/9 statements.


def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = (var_1, var_2)
    var_6 = lambda _: var_5
    var_7 = [var_4, var_6]



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 2/5 statements.


def test_case_0():
    var_0 = []
    var_1 = 'serialized'
    var_2 = 'format'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda fmt, val: f'{fmt}:{val}'
    var_1 = 'json'
    var_2 = 'data'
    var_3 = module_0.serialize(var_0, var_1, var_2)
    assert var_3 == 'json:data'



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_sequence_field_creates_correct_field_with_non_optional_type. Retrieved 3/11 statements.
# Partially parsed test_sequence_field_creates_correct_field_with_optional_type. Retrieved 4/12 statements.
# Partially parsed test_sequence_field_with_custom_invariant. Retrieved 5/10 statements.
# Partially parsed test_sequence_field_with_item_invariant. Retrieved 5/12 statements.
# Partially parsed test_sequence_field_with_optional_and_none_initial. Retrieved 2/6 statements.


def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []

def test_case_0():
    var_0 = True
    var_1 = set()
    var_2 = None
    var_3 = [var_2]
    var_4 = set()

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = [var_1, var_2]

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = {var_1, var_2}
    var_4 = {var_1, var_2}

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = [var_1]



# Parsed testcases at query #62
#--------------------------

# Partially parsed test__make_seq_field_type_creates_subclass_with_correct_name. Retrieved 2/5 statements.
# Partially parsed test__make_seq_field_type_stores_item_type_and_invariant. Retrieved 4/9 statements.
# Partially parsed test__make_seq_field_type_caches_created_types. Retrieved 3/7 statements.
# Partially parsed test__make_seq_field_type_preserves_checked_class_behavior. Retrieved 5/12 statements.


def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 5
    var_3 = -1

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



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_set_fields_predicate. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = []
    var_4 = 'fields'



# Parsed testcases at query #64
#--------------------------

# Failed to parse test__make_pmap_field_type_creates_new_type.
# Failed to parse test__make_pmap_field_type_reuses_existing_type.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'collections.abc.Hashable'
    var_1 = 'numbers.Number'
    var_2 = module_0._make_pmap_field_type(var_0, var_1)
    var_3 = var_2.__name__
    assert var_3 == 'HashableToNumberPMap'



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint_with_ignore_extra_false. Retrieved 2/8 statements.


def test_case_0():
    var_0 = lambda : None
    var_1 = False



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = 'some_format'



# Parsed testcases at query #67
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_new_subclass.
# Failed to parse test_make_pmap_field_type_reuses_existing_subclass.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = 'builtins.str'
    var_2 = module_0._make_pmap_field_type(var_0, var_1)
    var_3 = var_2.__name__
    assert var_3 == 'IntToStrPMap'
    var_4 = var_2.__key_type__
    var_5 = var_2.__value_type__

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'collections.OrderedDict'
    var_1 = 'decimal.Decimal'
    var_2 = module_0._make_pmap_field_type(var_0, var_1)
    var_3 = var_2.__name__
    assert var_3 == 'OrdereddictToDecimalPMap'
    var_4 = var_2.__key_type__
    var_5 = var_2.__value_type__



