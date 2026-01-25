####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = 'code1'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = 'code2'
    var_5 = (var_0, var_4)
    var_6 = lambda x: var_5
    var_7 = 'test_subject'
    var_8 = [var_3, var_6]
    var_9 = module_0.check_global_invariants(var_7, var_8)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = 'code1'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = False
    var_5 = 'code2'
    var_6 = (var_4, var_5)
    var_7 = lambda x: var_6
    var_8 = 'test_subject'
    var_9 = [var_3, var_7]
    var_10 = module_0.check_global_invariants(var_8, var_9)
    var_11 = bool(False)
    assert var_11 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = False
    var_1 = 'error1'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = 'error2'
    var_5 = (var_0, var_4)
    var_6 = lambda x: var_5
    var_7 = True
    var_8 = 'code3'
    var_9 = (var_7, var_8)
    var_10 = lambda x: var_9
    var_11 = 'key'
    var_12 = 'value'
    var_13 = {var_11: var_12}
    var_14 = [var_3, var_6, var_10]
    var_15 = module_0.check_global_invariants(var_13, var_14)
    var_16 = bool(False)
    assert var_16 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = []
    var_2 = module_0.check_global_invariants(var_0, var_1)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = 'success_code'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = 42
    var_5 = [var_3]
    var_6 = module_0.check_global_invariants(var_4, var_5)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = False
    var_1 = 'failure_code'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = None
    var_5 = [var_3]
    var_6 = module_0.check_global_invariants(var_4, var_5)
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #2
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_new_pmap_field_type.
# Failed to parse test_make_pmap_field_type_returns_cached_type.
# Failed to parse test_make_pmap_field_type_different_types_creates_different_classes.
# Failed to parse test_make_pmap_field_type_has_reduce_method.
# Failed to parse test_make_pmap_field_type_with_multiple_key_types.




# Parsed testcases at query #3
#--------------------------

# Partially parsed test_check_type_with_valid_single_type. Retrieved 3/12 statements.
# Partially parsed test_check_type_with_valid_multiple_types. Retrieved 3/11 statements.
# Partially parsed test_check_type_with_no_type_constraint. Retrieved 4/11 statements.
# Partially parsed test_check_type_with_invalid_type. Retrieved 3/13 statements.
# Partially parsed test_check_type_with_empty_type_tuple. Retrieved 4/12 statements.
# Partially parsed test_check_type_with_subclass. Retrieved 2/15 statements.
# Partially parsed test_check_type_error_message_format. Retrieved 3/13 statements.


def test_case_0():
    var_0 = 'TestClass'
    var_1 = 'test_field'
    var_2 = 42

def test_case_0():
    var_0 = 'TestClass'
    var_1 = 'test_field'
    var_2 = 'hello'

def test_case_0():
    var_0 = 'TestClass'
    var_1 = None
    var_2 = 'test_field'
    var_3 = 'any_value'

def test_case_0():
    var_0 = 'TestClass'
    var_1 = 'test_field'
    var_2 = 'not_an_int'
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'TestClass'
    var_1 = ()
    var_2 = 'test_field'
    var_3 = 42
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'TestClass'
    var_1 = 'test_field'

def test_case_0():
    var_0 = 'MyClass'
    var_1 = 'my_field'
    var_2 = 3.14
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'MyClass'
    var_5 = 'my_field'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_make_pmap_field_type_creates_new_type. Retrieved 2/6 statements.
# Failed to parse test_make_pmap_field_type_returns_cached_type.
# Failed to parse test_make_pmap_field_type_generates_correct_name.
# Failed to parse test_make_pmap_field_type_different_types_different_classes.
# Partially parsed test_make_pmap_field_type_has_reduce_method. Retrieved 1/6 statements.
# Failed to parse test_make_pmap_field_type_with_bool_types.


def test_case_0():
    var_0 = '__key_type__'
    var_1 = '__value_type__'

def test_case_0():
    var_0 = '__reduce__'



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_checked_pmap_subclass.
# Failed to parse test_make_pmap_field_type_caches_result.
# Failed to parse test_make_pmap_field_type_different_types_creates_different_classes.
# Partially parsed test_make_pmap_field_type_has_reduce_method. Retrieved 1/4 statements.
# Failed to parse test_make_pmap_field_type_with_multiple_key_value_types.


def test_case_0():
    var_0 = '__reduce__'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_check_global_invariants_no_errors. Retrieved 3/7 statements.
# Partially parsed test_check_global_invariants_single_error. Retrieved 3/8 statements.
# Partially parsed test_check_global_invariants_multiple_errors. Retrieved 3/10 statements.
# Partially parsed test_check_global_invariants_mixed_pass_fail. Retrieved 3/10 statements.
# Partially parsed test_check_global_invariants_all_pass. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 'data'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'test'
    var_1 = 'data'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'test'
    var_1 = 'data'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'test'
    var_1 = 'data'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'data'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.check_global_invariants(var_2, var_3)

def test_case_0():
    var_0 = 'test'
    var_1 = 'data'
    var_2 = {var_0: var_1}



# Parsed testcases at query #7
#--------------------------

# Failed to parse test_check_field_parameters_valid_field.
# Failed to parse test_check_field_parameters_invalid_type_parameter.
# Failed to parse test_check_field_parameters_invalid_initial_type.
# Failed to parse test_check_field_parameters_callable_initial.
# Failed to parse test_check_field_parameters_non_callable_invariant.
# Failed to parse test_check_field_parameters_non_callable_factory.
# Failed to parse test_check_field_parameters_non_callable_serializer.
# Failed to parse test_check_field_parameters_string_type.
# Failed to parse test_check_field_parameters_empty_type.




# Parsed testcases at query #8
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #9
#--------------------------

# Failed to parse test_types_to_names_with_single_type.
# Failed to parse test_types_to_names_with_multiple_types.
# Failed to parse test_types_to_names_with_bool_type.
# Failed to parse test_types_to_names_with_list_type.
# Failed to parse test_types_to_names_with_dict_type.
# Failed to parse test_types_to_names_with_multiple_builtin_types.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ()
    var_1 = module_0._types_to_names(var_0)
    assert var_1 == ''



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_check_field_parameters_valid_type_class. Retrieved 4/8 statements.
# Partially parsed test_check_field_parameters_valid_type_string. Retrieved 5/8 statements.
# Partially parsed test_check_field_parameters_invalid_type_parameter. Retrieved 4/9 statements.
# Partially parsed test_check_field_parameters_initial_no_initial. Retrieved 3/9 statements.
# Partially parsed test_check_field_parameters_initial_callable. Retrieved 4/8 statements.
# Partially parsed test_check_field_parameters_initial_valid_type. Retrieved 4/8 statements.
# Partially parsed test_check_field_parameters_initial_invalid_type. Retrieved 4/9 statements.
# Partially parsed test_check_field_parameters_initial_invalid_type_empty_type_list. Retrieved 5/8 statements.
# Partially parsed test_check_field_parameters_non_callable_invariant. Retrieved 4/9 statements.
# Partially parsed test_check_field_parameters_non_callable_factory. Retrieved 4/9 statements.
# Partially parsed test_check_field_parameters_non_callable_serializer. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 5
    var_1 = lambda self: True
    var_2 = lambda : None
    var_3 = lambda x: x

def test_case_0():
    var_0 = 'int'
    var_1 = 'str'
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = lambda self: True
    var_5 = lambda : None
    var_6 = lambda x: x

def test_case_0():
    var_0 = 5
    var_1 = lambda self: True
    var_2 = lambda : None
    var_3 = lambda x: x
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Type parameter expected'

def test_case_0():
    var_0 = []
    var_1 = lambda self: True
    var_2 = lambda : None
    var_3 = lambda x: x

def test_case_0():
    var_0 = lambda : 5
    var_1 = lambda self: True
    var_2 = lambda : None
    var_3 = lambda x: x

def test_case_0():
    var_0 = 'hello'
    var_1 = lambda self: True
    var_2 = lambda : None
    var_3 = lambda x: x

def test_case_0():
    var_0 = 3.14
    var_1 = lambda self: True
    var_2 = lambda : None
    var_3 = lambda x: x
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Initial has invalid type'

def test_case_0():
    var_0 = []
    var_1 = 3.14
    var_2 = lambda self: True
    var_3 = lambda : None
    var_4 = lambda x: x

def test_case_0():
    var_0 = 5
    var_1 = 'not callable'
    var_2 = lambda : None
    var_3 = lambda x: x
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Invariant must be callable'

def test_case_0():
    var_0 = 5
    var_1 = lambda self: True
    var_2 = 'not callable'
    var_3 = lambda x: x
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Factory must be callable'

def test_case_0():
    var_0 = 5
    var_1 = lambda self: True
    var_2 = lambda : None
    var_3 = 'not callable'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Serializer must be callable'



# Parsed testcases at query #11
#--------------------------

# Failed to parse test_field_with_single_type.
# Failed to parse test_field_with_multiple_types_as_tuple.
# Failed to parse test_field_with_multiple_types_as_list.
# Failed to parse test_field_with_multiple_types_as_set.
# Partially parsed test_field_with_initial_value. Retrieved 1/3 statements.
# Partially parsed test_field_with_mandatory_true. Retrieved 1/3 statements.
# Failed to parse test_field_with_callable_invariant.
# Failed to parse test_field_with_callable_factory.
# Failed to parse test_field_with_callable_serializer.
# Partially parsed test_field_initial_with_wrong_type. Retrieved 1/4 statements.
# Partially parsed test_field_non_callable_invariant. Retrieved 1/4 statements.
# Partially parsed test_field_non_callable_factory. Retrieved 1/4 statements.
# Partially parsed test_field_non_callable_serializer. Retrieved 1/4 statements.
# Failed to parse test_field_with_callable_initial.
# Failed to parse test_field_returns_pfield_instance.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'MyType'
    var_1 = module_0.field(var_0)
    var_2 = 'MyType'
    var_3 = bool('MyType' in var_1.type)
    assert var_3 is True

def test_case_0():
    var_0 = 42

def test_case_0():
    var_0 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.field(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Type specifications must be types or strings'

def test_case_0():
    var_0 = 'not an int'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Initial has invalid type'

def test_case_0():
    var_0 = 'not callable'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Invariant must be callable'

def test_case_0():
    var_0 = 'not callable'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Factory must be callable'

def test_case_0():
    var_0 = 'not callable'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Serializer must be callable'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = var_0.type
    var_2 = len(var_1)
    assert var_2 == 0



# Parsed testcases at query #12
#--------------------------

# Failed to parse test_check_field_parameters_predicate_line_3_false.




# Parsed testcases at query #13
#--------------------------

# Failed to parse test_wrap_invariant_called_when_invariant_is_callable_and_not_no_invariant.




# Parsed testcases at query #14
#--------------------------

# Failed to parse test_check_field_parameters_valid_field.
# Failed to parse test_check_field_parameters_invalid_type_parameter.
# Failed to parse test_check_field_parameters_invalid_initial_type.
# Failed to parse test_check_field_parameters_callable_initial.
# Failed to parse test_check_field_parameters_non_callable_invariant.
# Failed to parse test_check_field_parameters_non_callable_factory.
# Failed to parse test_check_field_parameters_non_callable_serializer.
# Failed to parse test_check_field_parameters_string_type.
# Failed to parse test_check_field_parameters_empty_type.




# Parsed testcases at query #15
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 1/9 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 2/5 statements.
# Partially parsed test_serialize_with_non_checked_type_and_no_serializer. Retrieved 2/6 statements.
# Partially parsed test_serialize_with_checked_type_and_custom_serializer. Retrieved 1/10 statements.


def test_case_0():
    var_0 = []
    var_1 = 'json'

def test_case_0():
    var_0 = 'xml'
    var_1 = 'test_value'

def test_case_0():
    var_0 = []
    var_1 = 'json'
    var_2 = 'data'

def test_case_0():
    var_0 = 'json'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_pfield_constructor. Retrieved 2/10 statements.
# Partially parsed test_pfield_constructor_with_none_values. Retrieved 2/4 statements.
# Partially parsed test_pfield_constructor_mandatory_false. Retrieved 6/8 statements.


def test_case_0():
    var_0 = 42
    var_1 = True

def test_case_0():
    var_0 = None
    var_1 = False

def test_case_0():
    var_0 = 0
    var_1 = lambda x: len(x) > var_0
    var_2 = 'default'
    var_3 = False
    var_4 = 'created'
    var_5 = lambda : var_4



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_set_fields. Retrieved 29/52 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'fields'
    var_3 = module_0.set_fields(var_0, var_1, var_2)
    var_4 = 'fields'
    var_5 = bool('fields' in var_0)
    assert var_5 is True
    var_6 = var_0['fields']
    var_7 = bool(var_0['fields'] == {})
    assert var_7 is True
    var_8 = 'field_a'
    var_9 = 'field_b'
    var_10 = 'other'
    var_11 = 'value'
    var_12 = []
    var_13 = 'fields'
    var_14 = 'fields'
    var_15 = 'field_a'
    var_16 = 'field_b'
    var_17 = 'field_a'
    var_18 = 'field_b'
    var_19 = 'other'
    var_20 = 'fields'
    var_21 = 'base_field'
    var_22 = 'base_value'
    var_23 = {var_21: var_22}
    var_24 = {}
    var_25 = 'fields'
    var_26 = 'fields'
    var_27 = bool('fields' in var_24)
    assert var_27 is True
    var_28 = 'base_field'
    var_29 = bool('base_field' in var_24['fields'])
    assert var_29 is True
    var_30 = var_24['fields']['base_field']
    assert var_30 == 'base_value'
    var_31 = 'inherited_field'
    var_32 = 'inherited_value'
    var_33 = {var_31: var_32}
    var_34 = 'new_field'
    var_35 = 'fields'
    var_36 = 'fields'
    var_37 = 'inherited_field'
    var_38 = 'new_field'
    var_39 = 'new_field'
    var_40 = 'field_from_base3'
    var_41 = 'value3'
    var_42 = {var_40: var_41}
    var_43 = 'field_from_base4'
    var_44 = 'value4'
    var_45 = {var_43: var_44}
    var_46 = {}
    var_47 = 'fields'
    var_48 = 'fields'
    var_49 = bool('fields' in var_46)
    assert var_49 is True
    var_50 = 'field_from_base3'
    var_51 = bool('field_from_base3' in var_46['fields'])
    assert var_51 is True
    var_52 = 'field_from_base4'
    var_53 = bool('field_from_base4' in var_46['fields'])
    assert var_53 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_set_fields_isinstance_predicate. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'not_a_field'
    var_3 = 'fields'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 6/15 statements.
# Partially parsed test_restore_pmap_field_pickle_empty. Retrieved 3/13 statements.
# Partially parsed test_restore_pmap_field_pickle_multiple_entries. Retrieved 10/20 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.pmap(var_1)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 'a'
    var_5 = 'b'
    var_6 = 'c'
    var_7 = 'd'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.pmap(var_8)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint_ignore_extra_false. Retrieved 1/11 statements.
# Partially parsed test_is_field_ignore_extra_complaint_ignore_extra_true_wrong_type. Retrieved 1/11 statements.
# Partially parsed test_is_field_ignore_extra_complaint_ignore_extra_true_correct_type_no_param. Retrieved 1/11 statements.
# Partially parsed test_is_field_ignore_extra_complaint_ignore_extra_true_correct_type_with_param. Retrieved 1/11 statements.
# Partially parsed test_is_field_ignore_extra_complaint_empty_type_set. Retrieved 2/11 statements.
# Partially parsed test_is_field_ignore_extra_complaint_type_as_set. Retrieved 1/11 statements.


def test_case_0():
    var_0 = False

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = set()
    var_1 = True

def test_case_0():
    var_0 = True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 7/16 statements.
# Partially parsed test_restore_seq_field_pickle_with_string_items. Retrieved 6/14 statements.
# Partially parsed test_restore_seq_field_pickle_empty_data. Retrieved 3/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = None
    var_6 = '__iter__'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = []
    var_2 = None



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_check_global_invariants_no_errors. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'test_subject'
    var_1 = True
    var_2 = False
    assert var_2 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_check_global_invariants_all_pass. Retrieved 1/7 statements.
# Partially parsed test_check_global_invariants_single_failure. Retrieved 1/8 statements.
# Partially parsed test_check_global_invariants_multiple_failures. Retrieved 1/10 statements.
# Partially parsed test_check_global_invariants_with_none_subject. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'test_subject'

def test_case_0():
    var_0 = 'test_subject'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 'test_subject'
    var_1 = bool(False)
    assert var_1 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = []
    var_2 = module_0.check_global_invariants(var_0, var_1)

def test_case_0():
    var_0 = None



# Parsed testcases at query #24
#--------------------------

# Failed to parse test_check_field_parameters_predicate_line_3_false.




# Parsed testcases at query #25
#--------------------------

# Partially parsed test_check_global_invariants_all_pass. Retrieved 3/9 statements.
# Partially parsed test_check_global_invariants_single_failure. Retrieved 3/8 statements.
# Partially parsed test_check_global_invariants_multiple_failures. Retrieved 3/12 statements.
# Partially parsed test_check_global_invariants_with_different_subjects. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 'data'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'test'
    var_1 = 'data'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'test'
    var_1 = 'data'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = []
    var_1 = 'test'
    var_2 = 'data'
    var_3 = {var_1: var_2}
    var_4 = module_0.check_global_invariants(var_3, var_0)
    assert var_4 is None

def test_case_0():
    var_0 = 'value'
    var_1 = 5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #26
#--------------------------

# Failed to parse test_check_field_parameters_predicate_line_3_false.




# Parsed testcases at query #27
#--------------------------

# Partially parsed test_make_seq_field_type. Retrieved 4/19 statements.


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = '__name__'
    var_3 = '__reduce__'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_pmap_field_basic. Retrieved 1/6 statements.
# Partially parsed test_pmap_field_with_optional_false. Retrieved 2/7 statements.
# Partially parsed test_pmap_field_with_optional_true. Retrieved 2/7 statements.
# Partially parsed test_pmap_field_with_invariant. Retrieved 1/10 statements.
# Partially parsed test_pmap_field_factory_non_optional. Retrieved 6/11 statements.
# Partially parsed test_pmap_field_factory_optional_with_none. Retrieved 2/6 statements.
# Partially parsed test_pmap_field_factory_optional_with_data. Retrieved 5/10 statements.
# Failed to parse test_pmap_field_type_included.
# Partially parsed test_pmap_field_type_optional_included. Retrieved 1/5 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.pmap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.pmap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.pmap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.pmap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = False
    var_1 = 'key'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_1: var_2}
    var_5 = module_0.pmap(var_4)

def test_case_0():
    var_0 = True
    var_1 = None

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = True
    var_1 = 'key'
    var_2 = {var_1: var_0}
    var_3 = {var_1: var_0}
    var_4 = module_0.pmap(var_3)

def test_case_0():
    var_0 = True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_pfield_constructor. Retrieved 2/10 statements.
# Partially parsed test_pfield_constructor_with_none_values. Retrieved 2/4 statements.
# Partially parsed test_pfield_constructor_with_different_types. Retrieved 7/9 statements.


def test_case_0():
    var_0 = 10
    var_1 = True

def test_case_0():
    var_0 = None
    var_1 = False

def test_case_0():
    var_0 = 0
    var_1 = lambda x: len(x) > var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_set_fields. Retrieved 23/60 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'fields'
    var_3 = module_0.set_fields(var_0, var_1, var_2)
    var_4 = bool(var_0 == {'fields': {}})
    assert var_4 is True
    var_5 = 'value1'
    var_6 = 'value2'
    var_7 = 'field1'
    var_8 = 'field2'
    var_9 = 'other'
    var_10 = 'data'
    var_11 = []
    var_12 = 'field1'
    var_13 = 'field2'
    var_14 = 'base_field1'
    var_15 = 'base_value1'
    var_16 = 'base_field2'
    var_17 = 'base_value2'
    var_18 = 'new_field'
    var_19 = 'new_value'
    var_20 = 'base_field1'
    var_21 = 'base_field2'
    var_22 = 'new_field'
    var_23 = 'new_field'
    var_24 = 'pvalue'
    var_25 = 'pfield'
    var_26 = 'normal'
    var_27 = 'number'
    var_28 = 'value'
    var_29 = 42
    var_30 = []
    var_31 = 'pfield'
    var_32 = {}
    var_33 = 'field'
    var_34 = 'val'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 1/12 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'json'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_predicate_isinstance_pfield. Retrieved 3/14 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = []
    var_2 = 'fields'
    var_3 = 'field1'
    var_4 = 'field1'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 1/12 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'json'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_check_global_invariants_raises_exception_when_invariants_fail. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 'test_subject'
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint_ignore_extra_false. Retrieved 2/11 statements.
# Partially parsed test_is_field_ignore_extra_complaint_ignore_extra_true_but_not_type_cls. Retrieved 2/11 statements.
# Partially parsed test_is_field_ignore_extra_complaint_ignore_extra_true_type_cls_but_no_ignore_extra_param. Retrieved 2/11 statements.
# Partially parsed test_is_field_ignore_extra_complaint_all_conditions_met. Retrieved 2/11 statements.
# Partially parsed test_is_field_ignore_extra_complaint_with_type_string. Retrieved 3/12 statements.
# Partially parsed test_is_field_ignore_extra_complaint_with_multiple_types_in_tuple. Retrieved 1/11 statements.


def test_case_0():
    var_0 = set()
    var_1 = False

def test_case_0():
    var_0 = ()
    var_1 = True

def test_case_0():
    var_0 = set()
    var_1 = True

def test_case_0():
    var_0 = set()
    var_1 = True

def test_case_0():
    var_0 = 'builtins.str'
    var_1 = (var_0,)
    var_2 = True

def test_case_0():
    var_0 = True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint_predicate_line_6_false. Retrieved 1/15 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_check_global_invariants_passes_subject_correctly. Retrieved 4/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = (var_0, var_1)
    var_5 = lambda x: var_4
    var_6 = (var_0, var_1)
    var_7 = lambda x: var_6
    var_8 = [var_3, var_5, var_7]
    var_9 = 'test'
    var_10 = module_0.check_global_invariants(var_9, var_8)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = False
    var_5 = 'ERROR_001'
    var_6 = (var_4, var_5)
    var_7 = lambda x: var_6
    var_8 = (var_0, var_1)
    var_9 = lambda x: var_8
    var_10 = [var_3, var_7, var_9]
    var_11 = 'test'
    var_12 = module_0.check_global_invariants(var_11, var_10)
    var_13 = bool(False)
    assert var_13 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = False
    var_1 = 'ERROR_001'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = 'ERROR_002'
    var_5 = (var_0, var_4)
    var_6 = lambda x: var_5
    var_7 = True
    var_8 = None
    var_9 = (var_7, var_8)
    var_10 = lambda x: var_9
    var_11 = [var_3, var_6, var_10]
    var_12 = 'test'
    var_13 = module_0.check_global_invariants(var_12, var_11)
    var_14 = bool(False)
    assert var_14 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = False
    var_1 = 'ERROR_A'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = 'ERROR_B'
    var_5 = (var_0, var_4)
    var_6 = lambda x: var_5
    var_7 = 'ERROR_C'
    var_8 = (var_0, var_7)
    var_9 = lambda x: var_8
    var_10 = [var_3, var_6, var_9]
    var_11 = 'test'
    var_12 = module_0.check_global_invariants(var_11, var_10)
    var_13 = bool(False)
    assert var_13 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = []
    var_1 = 'test'
    var_2 = module_0.check_global_invariants(var_1, var_0)

def test_case_0():
    var_0 = []
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = var_0[0]
    var_5 = bool(var_0[0] == var_3)
    assert var_5 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_check_global_invariants_no_errors. Retrieved 3/7 statements.
# Partially parsed test_check_global_invariants_single_error. Retrieved 3/8 statements.
# Partially parsed test_check_global_invariants_multiple_errors. Retrieved 3/10 statements.
# Partially parsed test_check_global_invariants_mixed_pass_fail. Retrieved 3/10 statements.
# Partially parsed test_check_global_invariants_multiple_invariants_all_pass. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 'data'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'test'
    var_1 = 'data'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'test'
    var_1 = 'data'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'test'
    var_1 = 'data'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'data'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.check_global_invariants(var_2, var_3)

def test_case_0():
    var_0 = 'test'
    var_1 = 'data'
    var_2 = {var_0: var_1}



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 6/13 statements.
# Partially parsed test_restore_pmap_field_pickle_empty. Retrieved 2/8 statements.
# Partially parsed test_restore_pmap_field_pickle_different_types. Retrieved 8/13 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 'one'
    var_4 = 'two'
    var_5 = 'three'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.pmap(var_6)



# Parsed testcases at query #40
#--------------------------

# Failed to parse test_check_field_parameters_valid_field.
# Failed to parse test_check_field_parameters_invalid_type_parameter.
# Failed to parse test_check_field_parameters_invalid_initial_type.
# Failed to parse test_check_field_parameters_callable_initial.
# Failed to parse test_check_field_parameters_non_callable_invariant.
# Failed to parse test_check_field_parameters_non_callable_factory.
# Failed to parse test_check_field_parameters_non_callable_serializer.
# Failed to parse test_check_field_parameters_string_type.
# Failed to parse test_check_field_parameters_empty_type.




# Parsed testcases at query #41
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_pmap_subclass.
# Failed to parse test_make_pmap_field_type_caches_result.
# Failed to parse test_make_pmap_field_type_different_types.
# Partially parsed test_make_pmap_field_type_has_reduce_method. Retrieved 1/6 statements.
# Failed to parse test_make_pmap_field_type_multiple_key_types.


def test_case_0():
    var_0 = '__reduce__'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 4/16 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_set_fields_predicate_isinstance_pfield. Retrieved 3/14 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = []
    var_2 = 'fields'
    var_3 = 'field1'
    var_4 = 'field1'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 6/11 statements.
# Partially parsed test_restore_seq_field_pickle_with_empty_data. Retrieved 2/6 statements.
# Partially parsed test_restore_seq_field_pickle_preserves_data. Retrieved 5/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = '__iter__'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = []

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 1/7 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 2/5 statements.
# Partially parsed test_serialize_with_checked_type_and_custom_serializer. Retrieved 1/8 statements.
# Partially parsed test_serialize_with_non_checked_type. Retrieved 2/5 statements.
# Partially parsed test_serialize_checked_type_with_different_formats. Retrieved 3/11 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'json'

def test_case_0():
    var_0 = 'xml'
    var_1 = 'test_value'

def test_case_0():
    var_0 = []
    var_1 = 'json'

def test_case_0():
    var_0 = 'json'
    var_1 = 42

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'json'
    var_3 = 'xml'
    var_4 = 'csv'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_check_global_invariants_no_errors. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'test_subject'
    var_1 = True
    var_2 = False
    assert var_2 is True



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 8/20 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.pmap(var_6)



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_pmap_field_basic. Retrieved 1/6 statements.
# Partially parsed test_pmap_field_optional_false. Retrieved 2/7 statements.
# Partially parsed test_pmap_field_optional_true. Retrieved 3/9 statements.
# Failed to parse test_pmap_field_with_invariant.
# Partially parsed test_pmap_field_factory_creates_map. Retrieved 6/12 statements.
# Partially parsed test_pmap_field_optional_factory_with_data. Retrieved 6/12 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.pmap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.pmap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.pmap()
    var_2 = None

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap()
    var_6 = [var_5]

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = True
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 2
    var_4 = {var_1: var_0, var_2: var_3}
    var_5 = module_0.pmap()
    var_6 = [var_5]



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_pmap_field_optional_false_predicate. Retrieved 2/5 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.pmap()



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint_predicate_line_6_false. Retrieved 1/11 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_make_seq_field_type_creates_new_type. Retrieved 2/7 statements.
# Partially parsed test_make_seq_field_type_caches_result. Retrieved 1/6 statements.
# Partially parsed test_make_seq_field_type_with_different_item_types. Retrieved 1/5 statements.
# Failed to parse test_make_seq_field_type_with_invariant.
# Partially parsed test_make_seq_field_type_has_reduce_method. Retrieved 5/12 statements.


def test_case_0():
    var_0 = None
    var_1 = '__reduce__'

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



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_pfield_factory_assignment. Retrieved 2/8 statements.


def test_case_0():
    var_0 = None
    var_1 = False



# Parsed testcases at query #53
#--------------------------

# Failed to parse test_make_pmap_field_type.




# Parsed testcases at query #54
#--------------------------

# Partially parsed test_pfield_factory_assignment. Retrieved 3/8 statements.


def test_case_0():
    var_0 = set()
    var_1 = None
    var_2 = False



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 8/18 statements.
# Partially parsed test_restore_pmap_field_pickle_with_empty_data. Retrieved 1/8 statements.
# Partially parsed test_restore_pmap_field_pickle_different_types. Retrieved 5/12 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'items'

def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'one'
    var_3 = 'two'
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_pmap_field_returns_field_with_correct_type. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'type'
    var_1 = 'factory'
    var_2 = 'mandatory'



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_pmap_field_optional_parameter_affects_type. Retrieved 3/13 statements.


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = None
    var_3 = [var_2]



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_make_seq_field_type_creates_new_type. Retrieved 3/9 statements.
# Partially parsed test_make_seq_field_type_caches_result. Retrieved 1/7 statements.
# Partially parsed test_make_seq_field_type_has_reduce_method. Retrieved 2/9 statements.
# Partially parsed test_make_seq_field_type_with_different_item_types. Retrieved 1/6 statements.
# Failed to parse test_make_seq_field_type_with_invariant.


def test_case_0():
    var_0 = None
    var_1 = '__type__'
    var_2 = '__invariant__'

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = None
    var_1 = '__reduce__'

def test_case_0():
    var_0 = None



# Parsed testcases at query #59
#--------------------------

# Failed to parse test_pmap_field_basic.
# Partially parsed test_pmap_field_optional_false. Retrieved 2/5 statements.
# Partially parsed test_pmap_field_optional_true. Retrieved 1/5 statements.
# Failed to parse test_pmap_field_with_invariant.
# Partially parsed test_pmap_field_factory_with_optional_true. Retrieved 2/7 statements.
# Partially parsed test_pmap_field_factory_with_optional_false. Retrieved 3/8 statements.
# Failed to parse test_pmap_field_returns_pfield.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.pmap()

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = True
    var_1 = None

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.pmap()



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_pmap_field_factory_property_returns_no_factory_when_multiple_types. Retrieved 2/7 statements.


def test_case_0():
    var_0 = None
    var_1 = False



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_pmap_field_optional_false_predicate. Retrieved 2/5 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.pmap()



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_make_seq_field_type_creates_new_type. Retrieved 4/14 statements.
# Partially parsed test_make_seq_field_type_caches_result. Retrieved 1/7 statements.
# Failed to parse test_make_seq_field_type_with_invariant.
# Partially parsed test_make_seq_field_type_reduce_method. Retrieved 6/15 statements.
# Partially parsed test_make_seq_field_type_different_item_types. Retrieved 1/6 statements.


def test_case_0():
    var_0 = None
    var_1 = '__type__'
    var_2 = '__invariant__'
    var_3 = '__name__'

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 0

def test_case_0():
    var_0 = None



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint_predicate_line_6_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #64
#--------------------------

# Failed to parse test_check_field_parameters_valid_type_class.
# Failed to parse test_check_field_parameters_valid_type_string.
# Failed to parse test_check_field_parameters_valid_callable_initial.
# Failed to parse test_check_field_parameters_invalid_type_parameter.
# Failed to parse test_check_field_parameters_invalid_initial_type.
# Failed to parse test_check_field_parameters_non_callable_invariant.
# Failed to parse test_check_field_parameters_non_callable_factory.
# Failed to parse test_check_field_parameters_non_callable_serializer.
# Failed to parse test_check_field_parameters_empty_type.
# Failed to parse test_check_field_parameters_no_initial.




# Parsed testcases at query #65
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]
    var_5 = [var_0, var_1, var_2]



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_check_global_invariants_raises_when_invariants_fail. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'test_subject'
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_set_fields_with_pfield_values. Retrieved 8/15 statements.
# Partially parsed test_set_fields_with_base_fields. Retrieved 6/18 statements.
# Partially parsed test_set_fields_multiple_bases. Retrieved 7/22 statements.
# Partially parsed test_set_fields_no_fields_in_bases. Retrieved 4/16 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = {}
    var_1 = ()
    var_2 = 'fields'
    var_3 = module_0.set_fields(var_0, var_1, var_2)
    var_4 = 'fields'
    var_5 = bool('fields' in var_0)
    assert var_5 is True
    var_6 = var_0['fields']
    var_7 = bool(var_0['fields'] == {})
    assert var_7 is True

def test_case_0():
    var_0 = 'value1'
    var_1 = 'value2'
    var_2 = 'field1'
    var_3 = 'field2'
    var_4 = 'other'
    var_5 = 'data'
    var_6 = ()
    var_7 = 'fields'
    var_8 = 'fields'
    var_9 = 'field1'
    var_10 = 'field2'

def test_case_0():
    var_0 = 'base_value'
    var_1 = 'fields'
    var_2 = 'base_field'
    var_3 = 'new_field'
    var_4 = 'new_value'
    var_5 = 'fields'
    var_6 = 'fields'
    var_7 = 'base_field'
    var_8 = 'new_field'
    var_9 = 'new_field'

def test_case_0():
    var_0 = 'fields'
    var_1 = 'field1'
    var_2 = 'value1'
    var_3 = 'field2'
    var_4 = 'value2'
    var_5 = {}
    var_6 = 'fields'
    var_7 = 'fields'
    var_8 = bool('fields' in var_5)
    assert var_8 is True
    var_9 = 'field1'
    var_10 = bool('field1' in var_5['fields'])
    assert var_10 is True
    var_11 = 'field2'
    var_12 = bool('field2' in var_5['fields'])
    assert var_12 is True

def test_case_0():
    var_0 = 'myfield'
    var_1 = 'myvalue'
    var_2 = 'fields'
    var_3 = 'fields'
    var_4 = 'fields'
    var_5 = 'myfield'



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_pmap_field_optional_type_includes_none. Retrieved 3/13 statements.


def test_case_0():
    var_0 = True
    var_1 = 0
    var_2 = None
    var_3 = [var_2]



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_pfield_init_factory_assignment. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = lambda : var_0
    var_2 = None
    var_3 = False



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_make_seq_field_type_creates_subclass_with_correct_name. Retrieved 3/9 statements.
# Partially parsed test_make_seq_field_type_caches_result. Retrieved 2/7 statements.
# Partially parsed test_make_seq_field_type_has_reduce_method. Retrieved 3/10 statements.
# Partially parsed test_make_seq_field_type_different_item_types_create_different_classes. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = '__name__'

def test_case_0():
    var_0 = 0
    var_1 = lambda x: len(x) > var_0

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = '__reduce__'

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 6/20 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_check_global_invariants_no_errors. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'test_subject'
    var_1 = True
    var_2 = False
    assert var_2 is True



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 1/7 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 2/5 statements.
# Partially parsed test_serialize_with_checked_type_and_custom_serializer. Retrieved 1/8 statements.
# Partially parsed test_serialize_with_non_checked_type_and_no_serializer. Retrieved 2/6 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'json'

def test_case_0():
    var_0 = 'xml'
    var_1 = 'test_value'

def test_case_0():
    var_0 = []
    var_1 = 'json'

def test_case_0():
    var_0 = []
    var_1 = 'json'
    var_2 = 'plain_value'



# Parsed testcases at query #74
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_new_type.
# Failed to parse test_make_pmap_field_type_caches_result.
# Failed to parse test_make_pmap_field_type_different_types.
# Partially parsed test_make_pmap_field_type_has_reduce_method. Retrieved 1/6 statements.
# Failed to parse test_make_pmap_field_type_with_float_and_bool.


def test_case_0():
    var_0 = '__reduce__'



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_pmap_field_predicate_line_2_evaluates_to_false. Retrieved 1/5 statements.


def test_case_0():
    var_0 = False
    var_1 = []



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 8/23 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.pmap()
    var_8 = [var_7]



# Parsed testcases at query #77
#--------------------------

# Failed to parse test_check_field_parameters_predicate_line_3_false.




# Parsed testcases at query #78
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 4/15 statements.
# Partially parsed test_restore_seq_field_pickle_empty. Retrieved 1/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = []



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_pfield_factory_assignment. Retrieved 2/8 statements.


def test_case_0():
    var_0 = None
    var_1 = False



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_check_global_invariants_with_failed_invariant. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'test_subject'



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_check_global_invariants_with_no_violations. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 'data'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = False
    assert var_4 is True



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 1/11 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'json'



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint. Retrieved 8/31 statements.


import collections as module_0

def test_case_0():
    var_0 = 'MockField'
    var_1 = 'type'
    var_2 = 'factory'
    var_3 = [var_1, var_2]
    var_4 = module_0.namedtuple(var_0, var_3)
    var_5 = False
    var_6 = ()
    var_7 = True



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_pmap_field_optional_false_creates_field_with_themap_type. Retrieved 2/5 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.pmap()



# Parsed testcases at query #85
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_pmap_subclass.
# Failed to parse test_make_pmap_field_type_generates_correct_name.
# Failed to parse test_make_pmap_field_type_caches_types.
# Failed to parse test_make_pmap_field_type_different_types_not_cached.
# Partially parsed test_make_pmap_field_type_reduce_method. Retrieved 3/7 statements.
# Failed to parse test_make_pmap_field_type_with_builtin_types.
# Failed to parse test_make_pmap_field_type_with_bool_and_list.


def test_case_0():
    var_0 = 'key'
    var_1 = 1
    var_2 = {var_0: var_1}



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_pmap_field_factory_property_with_non_checkedtype. Retrieved 2/7 statements.


def test_case_0():
    var_0 = None
    var_1 = False



# Parsed testcases at query #87
#--------------------------

# Partially parsed test_make_seq_field_type_creates_new_type. Retrieved 1/6 statements.
# Partially parsed test_make_seq_field_type_returns_cached_type. Retrieved 1/7 statements.
# Partially parsed test_make_seq_field_type_has_correct_name. Retrieved 1/6 statements.
# Partially parsed test_make_seq_field_type_with_different_item_types. Retrieved 1/6 statements.
# Partially parsed test_make_seq_field_type_reduce_method. Retrieved 5/12 statements.


def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = None
    var_1 = 'Int'
    var_2 = 'Vector'

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_set_fields_predicate_isinstance_pfield. Retrieved 5/12 statements.


def test_case_0():
    var_0 = 'field1'
    assert var_0 is True
    var_1 = 'field2'
    var_2 = 'value'
    var_3 = []
    var_4 = 'fields'



# Parsed testcases at query #89
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 1/11 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'json'



# Parsed testcases at query #91
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint_predicate_line_6_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #92
#--------------------------

# Partially parsed test_set_fields_predicate_isinstance_pfield. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'not_a_pfield'
    var_3 = 'fields'



# Parsed testcases at query #93
#--------------------------

# Partially parsed test_check_global_invariants_no_errors. Retrieved 3/9 statements.
# Partially parsed test_check_global_invariants_single_error. Retrieved 3/8 statements.
# Partially parsed test_check_global_invariants_multiple_errors. Retrieved 3/12 statements.
# Partially parsed test_check_global_invariants_with_complex_subject. Retrieved 5/11 statements.
# Partially parsed test_check_global_invariants_complex_subject_with_error. Retrieved 5/12 statements.


def test_case_0():
    var_0 = 'value'
    var_1 = 10
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'value'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'value'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.check_global_invariants(var_2, var_3)

def test_case_0():
    var_0 = 'value'
    var_1 = 'name'
    var_2 = 5
    var_3 = 'test'
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'value'
    var_1 = 'name'
    var_2 = -5
    var_3 = 'test'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #94
#--------------------------

# Failed to parse test_check_field_parameters_valid_field.
# Failed to parse test_check_field_parameters_invalid_type_parameter.
# Failed to parse test_check_field_parameters_invalid_initial_type.
# Failed to parse test_check_field_parameters_callable_initial.
# Failed to parse test_check_field_parameters_non_callable_invariant.
# Failed to parse test_check_field_parameters_non_callable_factory.
# Failed to parse test_check_field_parameters_non_callable_serializer.
# Failed to parse test_check_field_parameters_string_type.
# Failed to parse test_check_field_parameters_empty_type_with_initial.




# Parsed testcases at query #95
#--------------------------

# Partially parsed test_pfield_factory_assignment. Retrieved 2/8 statements.


def test_case_0():
    var_0 = None
    var_1 = False



# Parsed testcases at query #96
#--------------------------

# Partially parsed test_make_pmap_field_type. Retrieved 3/14 statements.


def test_case_0():
    var_0 = '__key_type__'
    var_1 = '__value_type__'
    var_2 = '__reduce__'



# Parsed testcases at query #97
#--------------------------

# Failed to parse test_pmap_field_basic.
# Partially parsed test_pmap_field_with_optional_false. Retrieved 1/3 statements.
# Partially parsed test_pmap_field_with_optional_true. Retrieved 1/5 statements.
# Failed to parse test_pmap_field_with_invariant.
# Partially parsed test_pmap_field_factory_with_optional_true. Retrieved 2/5 statements.
# Partially parsed test_pmap_field_factory_callable. Retrieved 1/5 statements.


def test_case_0():
    var_0 = False

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = False



# Parsed testcases at query #98
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 5/16 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #99
#--------------------------

# Partially parsed test_pmap_field_optional_false_predicate. Retrieved 4/8 statements.


def test_case_0():
    var_0 = False
    var_1 = 'key'
    var_2 = 1
    var_3 = {var_1: var_2}



# Parsed testcases at query #100
#--------------------------

# Partially parsed test_check_global_invariants_with_object_subject. Retrieved 10/15 statements.
# Partially parsed test_check_global_invariants_object_fails. Retrieved 10/16 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = (var_0, var_1)
    var_5 = lambda x: var_4
    var_6 = [var_3, var_5]
    var_7 = 'test_subject'
    var_8 = module_0.check_global_invariants(var_7, var_6)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = False
    var_5 = 'ERROR_CODE_1'
    var_6 = (var_4, var_5)
    var_7 = lambda x: var_6
    var_8 = [var_3, var_7]
    var_9 = 'test_subject'
    var_10 = module_0.check_global_invariants(var_9, var_8)
    var_11 = bool(False)
    assert var_11 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = False
    var_1 = 'ERROR_1'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = 'ERROR_2'
    var_5 = (var_0, var_4)
    var_6 = lambda x: var_5
    var_7 = True
    var_8 = None
    var_9 = (var_7, var_8)
    var_10 = lambda x: var_9
    var_11 = [var_3, var_6, var_10]
    var_12 = 'test_subject'
    var_13 = module_0.check_global_invariants(var_12, var_11)
    var_14 = bool(False)
    assert var_14 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = []
    var_1 = 'test_subject'
    var_2 = module_0.check_global_invariants(var_1, var_0)

def test_case_0():
    var_0 = 42
    var_1 = 0
    var_2 = True
    var_3 = None
    var_4 = (var_2, var_3)
    var_5 = False
    var_6 = 'NEGATIVE_VALUE'
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_4 if x.value > var_1 else var_7
    var_9 = [var_8]

def test_case_0():
    var_0 = -5
    var_1 = 0
    var_2 = True
    var_3 = None
    var_4 = (var_2, var_3)
    var_5 = False
    var_6 = 'NEGATIVE_VALUE'
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_4 if x.value > var_1 else var_7
    var_9 = [var_8]
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #101
#--------------------------

# Partially parsed test_invariant. Retrieved 1/3 statements.
# Partially parsed test_make_seq_field_type. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 0

def test_case_0():
    var_0 = 0



# Parsed testcases at query #102
#--------------------------

# Failed to parse test_check_global_invariants_raises_exception_when_error_codes_exist.




# Parsed testcases at query #103
#--------------------------




import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = 'OK1'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = 'OK2'
    var_5 = (var_0, var_4)
    var_6 = lambda x: var_5
    var_7 = [var_3, var_6]
    var_8 = 'test_subject'
    var_9 = module_0.check_global_invariants(var_8, var_7)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = 'OK1'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = False
    var_5 = 'ERROR1'
    var_6 = (var_4, var_5)
    var_7 = lambda x: var_6
    var_8 = [var_3, var_7]
    var_9 = 'test_subject'
    var_10 = module_0.check_global_invariants(var_9, var_8)
    var_11 = bool(False)
    assert var_11 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = False
    var_1 = 'ERROR1'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = 'ERROR2'
    var_5 = (var_0, var_4)
    var_6 = lambda x: var_5
    var_7 = True
    var_8 = 'OK1'
    var_9 = (var_7, var_8)
    var_10 = lambda x: var_9
    var_11 = [var_3, var_6, var_10]
    var_12 = 'test_subject'
    var_13 = module_0.check_global_invariants(var_12, var_11)
    var_14 = bool(False)
    assert var_14 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = []
    var_1 = 'test_subject'
    var_2 = module_0.check_global_invariants(var_1, var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = False
    var_1 = 'FAIL1'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = 'FAIL2'
    var_5 = (var_0, var_4)
    var_6 = lambda x: var_5
    var_7 = [var_3, var_6]
    var_8 = 'test_subject'
    var_9 = module_0.check_global_invariants(var_8, var_7)
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #104
#--------------------------

# Partially parsed test_pfield_constructor. Retrieved 2/9 statements.
# Partially parsed test_pfield_constructor_with_none_values. Retrieved 2/3 statements.
# Partially parsed test_pfield_constructor_with_different_types. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 10
    var_1 = True

def test_case_0():
    var_0 = None
    var_1 = False

def test_case_0():
    var_0 = 0
    var_1 = lambda x: len(x) > var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = True



# Parsed testcases at query #105
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 6/13 statements.
# Partially parsed test_restore_pmap_field_pickle_empty. Retrieved 2/8 statements.
# Partially parsed test_restore_pmap_field_pickle_different_types. Retrieved 8/13 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.pmap(var_0)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.pmap(var_6)



# Parsed testcases at query #106
#--------------------------

# Partially parsed test_make_pmap_field_type_creates_new_type. Retrieved 2/6 statements.
# Failed to parse test_make_pmap_field_type_caches_result.
# Failed to parse test_make_pmap_field_type_different_types.
# Failed to parse test_make_pmap_field_type_name_generation.
# Partially parsed test_make_pmap_field_type_has_reduce_method. Retrieved 1/6 statements.
# Failed to parse test_make_pmap_field_type_reduce_returns_tuple.


def test_case_0():
    var_0 = '__key_type__'
    var_1 = '__value_type__'

def test_case_0():
    var_0 = '__reduce__'



# Parsed testcases at query #107
#--------------------------

# Failed to parse test_check_field_parameters_valid_types.
# Failed to parse test_check_field_parameters_invalid_type_parameter.
# Failed to parse test_check_field_parameters_invalid_initial_type.
# Failed to parse test_check_field_parameters_callable_initial.
# Failed to parse test_check_field_parameters_non_callable_invariant.
# Failed to parse test_check_field_parameters_non_callable_factory.
# Failed to parse test_check_field_parameters_non_callable_serializer.
# Failed to parse test_check_field_parameters_string_type.
# Failed to parse test_check_field_parameters_empty_type.




# Parsed testcases at query #108
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 1/7 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 2/5 statements.
# Partially parsed test_serialize_with_checked_type_and_custom_serializer. Retrieved 1/8 statements.
# Partially parsed test_serialize_with_non_checked_type. Retrieved 2/5 statements.
# Partially parsed test_serialize_checked_type_calls_serialize_method. Retrieved 1/7 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'json'

def test_case_0():
    var_0 = 'xml'
    var_1 = 'test_value'

def test_case_0():
    var_0 = []
    var_1 = 'json'

def test_case_0():
    var_0 = 'csv'
    var_1 = 42

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'yaml'



# Parsed testcases at query #109
#--------------------------

# Failed to parse test_pmap_field_creates_field_with_correct_type.
# Partially parsed test_pmap_field_with_optional_true. Retrieved 1/5 statements.
# Partially parsed test_pmap_field_with_optional_false. Retrieved 1/5 statements.
# Failed to parse test_pmap_field_with_invariant.
# Partially parsed test_pmap_field_factory_handles_none_when_optional. Retrieved 2/5 statements.
# Failed to parse test_pmap_field_different_key_value_types.
# Failed to parse test_pmap_field_returns_pfield_instance.


def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = False

def test_case_0():
    var_0 = True
    var_1 = None



# Parsed testcases at query #110
#--------------------------

# Partially parsed test_set_fields_predicate_isinstance_pfield. Retrieved 4/18 statements.


def test_case_0():
    var_0 = 'field1'
    assert var_0 is True
    assert var_0 is False
    var_1 = 'field2'
    var_2 = 'not_a_pfield'
    var_3 = 'test_name'



# Parsed testcases at query #111
#--------------------------

# Partially parsed test_make_seq_field_type_creates_new_type. Retrieved 4/14 statements.
# Partially parsed test_make_seq_field_type_caches_result. Retrieved 1/7 statements.
# Partially parsed test_make_seq_field_type_with_different_item_types. Retrieved 1/6 statements.
# Partially parsed test_make_seq_field_type_has_reduce_method. Retrieved 2/9 statements.
# Failed to parse test_make_seq_field_type_with_invariant.


def test_case_0():
    var_0 = None
    var_1 = '__type__'
    var_2 = '__invariant__'
    var_3 = '__name__'

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = None
    var_1 = '__reduce__'



# Parsed testcases at query #112
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint_predicate_line_6_false. Retrieved 1/14 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #113
#--------------------------

# Failed to parse test_pmap_field_basic.
# Partially parsed test_pmap_field_optional_false. Retrieved 1/3 statements.
# Partially parsed test_pmap_field_optional_true. Retrieved 1/5 statements.
# Failed to parse test_pmap_field_with_invariant.
# Failed to parse test_pmap_field_initial_value.
# Partially parsed test_pmap_field_factory_with_none_optional. Retrieved 2/5 statements.
# Failed to parse test_pmap_field_different_types.
# Failed to parse test_pmap_field_type_attribute.


def test_case_0():
    var_0 = False

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = True
    var_1 = None



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_pfield_constructor. Retrieved 2/9 statements.
# Partially parsed test_pfield_constructor_with_none_values. Retrieved 2/3 statements.
# Partially parsed test_pfield_constructor_with_different_types. Retrieved 5/6 statements.
# Partially parsed test_pfield_constructor_slots. Retrieved 9/16 statements.


def test_case_0():
    var_0 = 10
    var_1 = True

def test_case_0():
    var_0 = None
    var_1 = False

def test_case_0():
    var_0 = 0
    var_1 = lambda x: len(x) > var_0
    var_2 = []
    var_3 = True
    var_4 = lambda x: repr(x)

def test_case_0():
    var_0 = None
    var_1 = 1.5
    var_2 = True
    var_3 = 'type'
    var_4 = 'invariant'
    var_5 = 'initial'
    var_6 = 'mandatory'
    var_7 = '_factory'
    var_8 = 'serializer'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_check_global_invariants_all_pass. Retrieved 1/7 statements.
# Partially parsed test_check_global_invariants_single_failure. Retrieved 1/8 statements.
# Partially parsed test_check_global_invariants_multiple_failures. Retrieved 1/10 statements.
# Partially parsed test_check_global_invariants_with_different_subject_types. Retrieved 8/16 statements.


def test_case_0():
    var_0 = 'test_subject'

def test_case_0():
    var_0 = 'test_subject'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 'test_subject'
    var_1 = bool(False)
    assert var_1 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = []
    var_2 = module_0.check_global_invariants(var_0, var_1)

def test_case_0():
    var_0 = 42
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_set_fields. Retrieved 38/72 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'regular'
    var_3 = 'value1'
    var_4 = 'value2'
    var_5 = 'data'
    var_6 = []
    var_7 = 'fields'
    var_8 = 'fields'
    var_9 = 'field1'
    var_10 = 'field2'
    var_11 = 'regular'
    var_12 = 'field1'
    var_13 = 'field2'
    var_14 = 'Base1'
    var_15 = ()
    var_16 = {}
    var_17 = [var_14, var_15, var_16]
    var_18 = 'base_field'
    var_19 = 'base_value'
    var_20 = 'new_field'
    var_21 = 'new_value'
    var_22 = 'fields'
    var_23 = 'base_field'
    var_24 = 'new_field'
    var_25 = 'new_field'
    var_26 = {}
    var_27 = []
    var_28 = module_0.set_fields(var_26, var_27, var_7)
    var_29 = 'fields'
    var_30 = bool('fields' in var_26)
    assert var_30 is True
    var_31 = var_26['fields']
    var_32 = bool(var_26['fields'] == {})
    assert var_32 is True
    var_33 = 'Base2'
    var_34 = ()
    var_35 = {}
    var_36 = [var_33, var_34, var_35]
    var_37 = 'inherited1'
    var_38 = 'val1'
    var_39 = 'Base3'
    var_40 = ()
    var_41 = {}
    var_42 = [var_39, var_40, var_41]
    var_43 = 'inherited2'
    var_44 = 'val2'
    var_45 = 'own_field'
    var_46 = 'other'
    var_47 = 'own_value'
    var_48 = 'keep'
    var_49 = 'fields'
    var_50 = 'inherited1'
    var_51 = 'inherited2'
    var_52 = 'own_field'
    var_53 = 'own_field'
    var_54 = 'other'
    var_55 = 'attr1'
    var_56 = 'attr2'
    var_57 = 123
    var_58 = {var_55: var_3, var_56: var_57}
    var_59 = []
    var_60 = module_0.set_fields(var_58, var_59, var_7)
    var_61 = 'fields'
    var_62 = bool('fields' in var_58)
    assert var_62 is True
    var_63 = var_58['fields']
    var_64 = bool(var_58['fields'] == {})
    assert var_64 is True
    var_65 = 'attr1'
    var_66 = bool('attr1' in var_58)
    assert var_66 is True
    var_67 = 'attr2'
    var_68 = bool('attr2' in var_58)
    assert var_68 is True



# Parsed testcases at query #4
#--------------------------

# Failed to parse test_check_field_parameters_valid_field.
# Failed to parse test_check_field_parameters_invalid_type_parameter.
# Failed to parse test_check_field_parameters_invalid_initial_type.
# Failed to parse test_check_field_parameters_callable_initial.
# Failed to parse test_check_field_parameters_non_callable_invariant.
# Failed to parse test_check_field_parameters_non_callable_factory.
# Failed to parse test_check_field_parameters_non_callable_serializer.
# Failed to parse test_check_field_parameters_string_type.
# Failed to parse test_check_field_parameters_empty_type.




# Parsed testcases at query #5
#--------------------------

# Failed to parse test_check_field_parameters_predicate_line_3_false.




# Parsed testcases at query #6
#--------------------------

# Failed to parse test_check_field_parameters_line_1_predicate_false.




# Parsed testcases at query #7
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 12/20 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.pmap()
    var_1 = 'TestPMapField'
    var_2 = 'create'
    var_3 = lambda cls, data, _factory_fields=None: pmap(data)
    var_4 = classmethod(var_3)
    var_5 = {var_2: var_4}
    var_6 = 'key1'
    var_7 = 'key2'
    var_8 = 1
    var_9 = 2
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = module_0.pmap(var_10)



# Parsed testcases at query #8
#--------------------------

# Failed to parse test_types_to_names_with_builtin_types.
# Failed to parse test_types_to_names_with_single_type.
# Failed to parse test_types_to_names_with_bool_type.
# Failed to parse test_types_to_names_with_list_type.
# Failed to parse test_types_to_names_with_dict_type.
# Failed to parse test_types_to_names_with_multiple_types.
# Failed to parse test_types_to_names_with_tuple_type.
# Failed to parse test_types_to_names_with_set_type.




# Parsed testcases at query #9
#--------------------------

# Partially parsed test_sequence_field_creates_field_with_checked_class. Retrieved 6/11 statements.
# Partially parsed test_sequence_field_optional_true_returns_none. Retrieved 5/9 statements.
# Partially parsed test_sequence_field_optional_false_with_value. Retrieved 9/14 statements.
# Partially parsed test_sequence_field_with_invariant. Retrieved 3/8 statements.
# Partially parsed test_sequence_field_with_item_invariant. Retrieved 4/9 statements.
# Partially parsed test_sequence_field_mandatory_is_true. Retrieved 2/5 statements.
# Partially parsed test_sequence_field_initial_value_set. Retrieved 5/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.field()
    var_6 = [var_5]

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = None

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = [var_1]

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]

def test_case_0():
    var_0 = False
    var_1 = []

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint. Retrieved 3/37 statements.


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = set()



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_check_global_invariants_with_failing_invariant. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 'data'
    var_2 = {var_0: var_1}



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_make_seq_field_type_creates_new_type. Retrieved 4/14 statements.
# Partially parsed test_make_seq_field_type_caches_result. Retrieved 1/7 statements.
# Failed to parse test_make_seq_field_type_with_invariant.
# Partially parsed test_make_seq_field_type_has_reduce_method. Retrieved 2/10 statements.
# Partially parsed test_make_seq_field_type_different_types_different_results. Retrieved 1/6 statements.


def test_case_0():
    var_0 = None
    var_1 = '__type__'
    var_2 = '__invariant__'
    var_3 = '__name__'

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = None
    var_1 = '__reduce__'

def test_case_0():
    var_0 = None



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_make_pmap_field_type_creates_pmap_subclass. Retrieved 2/6 statements.
# Failed to parse test_make_pmap_field_type_caches_result.
# Failed to parse test_make_pmap_field_type_different_types_create_different_classes.
# Failed to parse test_make_pmap_field_type_class_name_format.
# Partially parsed test_make_pmap_field_type_has_reduce_method. Retrieved 1/6 statements.
# Failed to parse test_make_pmap_field_type_with_builtin_types.
# Failed to parse test_make_pmap_field_type_with_multiple_types_in_sequence.


def test_case_0():
    var_0 = '__key_type__'
    var_1 = '__value_type__'

def test_case_0():
    var_0 = '__reduce__'



# Parsed testcases at query #14
#--------------------------

# Failed to parse test_is_type_cls_with_set.
# Partially parsed test_is_type_cls_with_empty_tuple. Retrieved 1/3 statements.
# Failed to parse test_is_type_cls_with_single_type_class.
# Partially parsed test_is_type_cls_with_single_type_string. Retrieved 2/4 statements.
# Failed to parse test_is_type_cls_with_subclass.
# Failed to parse test_is_type_cls_with_non_subclass.
# Failed to parse test_is_type_cls_with_multiple_types_first_matches.
# Failed to parse test_is_type_cls_with_multiple_types_first_does_not_match.
# Failed to parse test_is_type_cls_with_list_converted_to_tuple.


def test_case_0():
    var_0 = ()

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = (var_0,)



# Parsed testcases at query #15
#--------------------------




import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ()
    var_1 = module_0._types_to_names(var_0)
    assert var_1 == ''



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_sequence_field_with_checked_pvector. Retrieved 5/11 statements.
# Partially parsed test_sequence_field_optional_with_none. Retrieved 2/6 statements.
# Partially parsed test_sequence_field_optional_with_value. Retrieved 4/8 statements.
# Partially parsed test_sequence_field_with_invariant. Retrieved 2/8 statements.
# Partially parsed test_sequence_field_factory_callable. Retrieved 2/7 statements.
# Partially parsed test_sequence_field_optional_factory_with_none. Retrieved 3/8 statements.
# Partially parsed test_sequence_field_non_optional_factory. Retrieved 6/12 statements.


def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = True
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]

def test_case_0():
    var_0 = False
    var_1 = []

def test_case_0():
    var_0 = False
    var_1 = []

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = None

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = [var_1, var_2, var_4]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_check_global_invariants_no_errors. Retrieved 1/5 statements.
# Partially parsed test_check_global_invariants_single_error. Retrieved 1/6 statements.
# Partially parsed test_check_global_invariants_multiple_errors. Retrieved 1/8 statements.
# Partially parsed test_check_global_invariants_mixed_pass_fail. Retrieved 1/8 statements.
# Partially parsed test_check_global_invariants_exception_message. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'test_subject'

def test_case_0():
    var_0 = 'test_subject'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 'test_subject'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 'test_subject'
    var_1 = bool(False)
    assert var_1 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = []
    var_1 = 'test_subject'
    var_2 = module_0.check_global_invariants(var_1, var_0)
    assert var_2 is None

def test_case_0():
    var_0 = 'test_subject'
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint_predicate_line_6_false. Retrieved 3/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'test'
    var_2 = True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_seq_field_type_caching. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0



# Parsed testcases at query #20
#--------------------------

# Failed to parse test_check_field_parameters_predicate_line_3.




# Parsed testcases at query #21
#--------------------------

# Partially parsed test_make_seq_field_type. Retrieved 4/19 statements.


def test_case_0():
    var_0 = None
    var_1 = '__type__'
    var_2 = '__invariant__'
    var_3 = '__reduce__'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_check_type_with_valid_type. Retrieved 3/12 statements.
# Partially parsed test_check_type_with_invalid_type. Retrieved 3/13 statements.
# Partially parsed test_check_type_with_multiple_valid_types. Retrieved 4/14 statements.
# Partially parsed test_check_type_with_no_type_constraint. Retrieved 5/14 statements.
# Partially parsed test_check_type_with_empty_type_list. Retrieved 4/13 statements.
# Partially parsed test_check_type_error_message_format. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'TestClass'
    var_1 = 'test_field'
    var_2 = 42

def test_case_0():
    var_0 = 'TestClass'
    var_1 = 'test_field'
    var_2 = 'not_an_int'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'Invalid type for field TestClass.test_field'

def test_case_0():
    var_0 = 'TestClass'
    var_1 = 'test_field'
    var_2 = 'string_value'
    var_3 = 42

def test_case_0():
    var_0 = 'TestClass'
    var_1 = None
    var_2 = 'test_field'
    var_3 = 'any_value'
    var_4 = 123

def test_case_0():
    var_0 = 'TestClass'
    var_1 = []
    var_2 = 'test_field'
    var_3 = 42
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Invalid type for field TestClass.test_field'

def test_case_0():
    var_0 = 'MyClass'
    var_1 = 'my_field'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'MyClass'
    var_8 = 'my_field'
    var_9 = 'list'



# Parsed testcases at query #23
#--------------------------

# Failed to parse test_make_pmap_field_type_returns_cached_type.




# Parsed testcases at query #24
#--------------------------

# Partially parsed test_set_fields_predicate_isinstance_pfield. Retrieved 4/14 statements.


def test_case_0():
    var_0 = 'field1'
    assert var_0 is True
    var_1 = 'field2'
    var_2 = 'not_a_pfield'
    var_3 = 'fields'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_pfield_constructor. Retrieved 2/10 statements.
# Partially parsed test_pfield_constructor_with_none_values. Retrieved 2/4 statements.
# Partially parsed test_pfield_constructor_with_multiple_types. Retrieved 6/8 statements.


def test_case_0():
    var_0 = 10
    var_1 = True

def test_case_0():
    var_0 = None
    var_1 = False

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = 5
    var_3 = 0
    var_4 = lambda : var_3
    var_5 = lambda x: x



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 1/7 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 2/5 statements.
# Partially parsed test_serialize_with_checked_type_and_custom_serializer. Retrieved 1/8 statements.
# Partially parsed test_serialize_with_regular_value_and_serializer. Retrieved 2/5 statements.
# Partially parsed test_serialize_checked_type_different_formats. Retrieved 2/9 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'json'

def test_case_0():
    var_0 = 'xml'
    var_1 = 'test_value'

def test_case_0():
    var_0 = []
    var_1 = 'json'

def test_case_0():
    var_0 = 'csv'
    var_1 = 42

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'json'
    var_3 = 'xml'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_check_type_predicate_evaluates_to_true. Retrieved 6/24 statements.


def test_case_0():
    var_0 = 'TestClass'
    var_1 = 'test_field'
    var_2 = 5
    var_3 = 'hello'
    var_4 = 42
    var_5 = 'test'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_check_field_parameters_predicate_line_3_evaluates_to_false. Retrieved 1/20 statements.


def test_case_0():
    var_0 = 'SomeType'



# Parsed testcases at query #29
#--------------------------

# Failed to parse test_make_pmap_field_type_returns_cached_type.




# Parsed testcases at query #30
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_new_type.
# Failed to parse test_make_pmap_field_type_caches_result.
# Failed to parse test_make_pmap_field_type_different_types_different_classes.
# Partially parsed test_make_pmap_field_type_has_reduce_method. Retrieved 1/7 statements.
# Failed to parse test_make_pmap_field_type_name_format.
# Failed to parse test_make_pmap_field_type_with_float_and_bool.


def test_case_0():
    var_0 = '__reduce__'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_check_global_invariants_all_pass. Retrieved 1/8 statements.
# Partially parsed test_check_global_invariants_single_failure. Retrieved 3/11 statements.
# Partially parsed test_check_global_invariants_multiple_failures. Retrieved 3/13 statements.
# Partially parsed test_check_global_invariants_all_failures. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'test_subject'

def test_case_0():
    var_0 = 'test_subject'
    var_1 = 'Should raise InvariantException'
    var_2 = AssertionError(var_1)

def test_case_0():
    var_0 = 'test_subject'
    var_1 = 'Should raise InvariantException'
    var_2 = AssertionError(var_1)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = []
    var_1 = 'test_subject'
    var_2 = module_0.check_global_invariants(var_1, var_0)

def test_case_0():
    var_0 = 'test_subject'
    var_1 = 'Should raise InvariantException'
    var_2 = AssertionError(var_1)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_pmap_field_basic. Retrieved 1/6 statements.
# Partially parsed test_pmap_field_optional_false. Retrieved 2/7 statements.
# Partially parsed test_pmap_field_optional_true. Retrieved 2/7 statements.
# Failed to parse test_pmap_field_with_invariant.
# Partially parsed test_pmap_field_factory_with_optional_true. Retrieved 2/6 statements.
# Partially parsed test_pmap_field_factory_with_optional_false. Retrieved 6/10 statements.
# Failed to parse test_pmap_field_caching.
# Failed to parse test_pmap_field_different_types.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.pmap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.pmap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.pmap()

def test_case_0():
    var_0 = True
    var_1 = None

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = False
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_1: var_2}
    var_5 = module_0.pmap(var_4)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint_predicate_line_6_false. Retrieved 2/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_make_pmap_field_type_creates_new_type. Retrieved 2/6 statements.
# Failed to parse test_make_pmap_field_type_returns_cached_type.
# Failed to parse test_make_pmap_field_type_generates_correct_name.
# Failed to parse test_make_pmap_field_type_different_types.
# Partially parsed test_make_pmap_field_type_has_reduce_method. Retrieved 1/6 statements.
# Failed to parse test_make_pmap_field_type_is_checked_pmap_subclass.


def test_case_0():
    var_0 = '__key_type__'
    var_1 = '__value_type__'

def test_case_0():
    var_0 = '__reduce__'



# Parsed testcases at query #35
#--------------------------

# Failed to parse test_check_field_parameters_predicate_line_6_true.




# Parsed testcases at query #36
#--------------------------

# Partially parsed test_make_seq_field_type_creates_new_type. Retrieved 3/10 statements.
# Partially parsed test_make_seq_field_type_caches_result. Retrieved 1/7 statements.
# Partially parsed test_make_seq_field_type_with_different_item_types. Retrieved 1/6 statements.
# Partially parsed test_make_seq_field_type_has_reduce_method. Retrieved 2/8 statements.
# Partially parsed test_make_seq_field_type_name_generation. Retrieved 2/11 statements.


def test_case_0():
    var_0 = None
    var_1 = '__type__'
    var_2 = '__invariant__'

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = None
    var_1 = '__reduce__'

def test_case_0():
    var_0 = None
    var_1 = '__name__'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_check_global_invariants_raises_exception_when_error_codes_exist. Retrieved 1/17 statements.


def test_case_0():
    var_0 = 'test_subject'
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #38
#--------------------------

# Failed to parse test_check_field_parameters_predicate_line_6_true.




# Parsed testcases at query #39
#--------------------------

# Partially parsed test_pfield_factory_assignment. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'custom'
    var_1 = lambda : var_0
    var_2 = None
    var_3 = False



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_pfield_factory_assignment_not_pfield_no_factory. Retrieved 3/9 statements.


def test_case_0():
    var_0 = set()
    var_1 = None
    var_2 = False



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_make_seq_field_type. Retrieved 5/15 statements.


def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_pfield_factory_assignment_with_none. Retrieved 2/7 statements.


def test_case_0():
    var_0 = None
    var_1 = False



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_pmap_field_returns_field_with_correct_type. Retrieved 1/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = [var_0]



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_pmap_field_basic. Retrieved 1/6 statements.
# Partially parsed test_pmap_field_optional_false. Retrieved 2/7 statements.
# Partially parsed test_pmap_field_optional_true. Retrieved 3/9 statements.
# Failed to parse test_pmap_field_with_invariant.
# Partially parsed test_pmap_field_factory_creates_map. Retrieved 8/12 statements.
# Partially parsed test_pmap_field_optional_factory_with_data. Retrieved 6/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.pmap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.pmap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.pmap()
    var_2 = None

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = False
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_1: var_3, var_2: var_4}
    var_7 = module_0.pmap(var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = True
    var_1 = 'x'
    var_2 = 10
    var_3 = {var_1: var_2}
    var_4 = {var_1: var_2}
    var_5 = module_0.pmap(var_4)



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 7/16 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap()
    var_6 = [var_5]
    var_7 = module_0.pmap(var_4)



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_check_global_invariants_all_pass. Retrieved 1/7 statements.
# Partially parsed test_check_global_invariants_single_failure. Retrieved 1/8 statements.
# Partially parsed test_check_global_invariants_multiple_failures. Retrieved 1/10 statements.
# Partially parsed test_check_global_invariants_with_complex_subject. Retrieved 3/7 statements.
# Partially parsed test_check_global_invariants_complex_subject_failure. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'test_subject'

def test_case_0():
    var_0 = 'test_subject'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 'test_subject'
    var_1 = bool(False)
    assert var_1 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = []
    var_1 = 'test_subject'
    var_2 = module_0.check_global_invariants(var_1, var_0)

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'key'
    var_1 = 'wrong_value'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_set_fields. Retrieved 26/75 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'fields'
    var_3 = module_0.set_fields(var_0, var_1, var_2)
    var_4 = bool(var_0 == {'fields': {}})
    assert var_4 is True
    var_5 = 'field1'
    var_6 = 'field2'
    var_7 = 'value1'
    var_8 = 'value2'
    var_9 = []
    var_10 = 'fields'
    var_11 = 'field1'
    var_12 = 'field2'
    var_13 = 'field1'
    var_14 = 'field2'
    var_15 = 'other'
    var_16 = 'not_a_field'
    var_17 = []
    var_18 = 'fields'
    var_19 = 'field1'
    var_20 = 'other'
    var_21 = 'field1'
    var_22 = 'fields'
    var_23 = 'inherited_field'
    var_24 = 'inherited'
    var_25 = 'new_field'
    var_26 = 'new'
    var_27 = 'fields'
    var_28 = 'new_field'
    var_29 = 'inherited_field'
    var_30 = 'new_field'
    var_31 = 'field_a'
    var_32 = 'a'
    var_33 = 'field_b'
    var_34 = 'b'
    var_35 = {}
    var_36 = 'fields'
    var_37 = 'field_a'
    var_38 = bool('field_a' in var_35[var_36])
    assert var_38 is True
    var_39 = 'field_b'
    var_40 = bool('field_b' in var_35[var_36])
    assert var_40 is True



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_pfield_constructor. Retrieved 2/10 statements.
# Partially parsed test_pfield_constructor_with_none_values. Retrieved 2/4 statements.
# Partially parsed test_pfield_constructor_slots. Retrieved 10/18 statements.


def test_case_0():
    var_0 = 'initial'
    var_1 = True

def test_case_0():
    var_0 = None
    var_1 = False

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 42
    var_3 = True
    var_4 = 'type'
    var_5 = 'invariant'
    var_6 = 'initial'
    var_7 = 'mandatory'
    var_8 = '_factory'
    var_9 = 'serializer'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_set_fields_predicate_isinstance_pfield. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'test_name'



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_pmap_field_optional_predicate_true. Retrieved 1/4 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_pmap_field_type_predicate_line_25_with_optional_true. Retrieved 2/11 statements.
# Partially parsed test_pmap_field_type_predicate_line_25_with_optional_false. Retrieved 2/10 statements.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = [var_1]

def test_case_0():
    var_0 = False
    var_1 = None
    var_2 = [var_1]



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_make_pmap_field_type_creates_new_pmap_class. Retrieved 2/6 statements.
# Failed to parse test_make_pmap_field_type_caches_result.
# Failed to parse test_make_pmap_field_type_generates_correct_name.
# Failed to parse test_make_pmap_field_type_different_types_create_different_classes.
# Partially parsed test_make_pmap_field_type_has_reduce_method. Retrieved 1/6 statements.
# Failed to parse test_make_pmap_field_type_with_multiple_types.


def test_case_0():
    var_0 = '__key_type__'
    var_1 = '__value_type__'

def test_case_0():
    var_0 = '__reduce__'



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_check_global_invariants_all_pass. Retrieved 1/7 statements.
# Partially parsed test_check_global_invariants_single_failure. Retrieved 1/8 statements.
# Partially parsed test_check_global_invariants_multiple_failures. Retrieved 1/10 statements.
# Partially parsed test_check_global_invariants_single_invariant_pass. Retrieved 1/5 statements.
# Partially parsed test_check_global_invariants_with_different_subjects. Retrieved 1/7 statements.
# Partially parsed test_check_global_invariants_subject_fails_first_invariant. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 'test_subject'

def test_case_0():
    var_0 = 'test_subject'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 'test_subject'
    var_1 = bool(False)
    assert var_1 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = []
    var_1 = 'test_subject'
    var_2 = module_0.check_global_invariants(var_1, var_0)

def test_case_0():
    var_0 = 'test_subject'

def test_case_0():
    var_0 = 42

def test_case_0():
    var_0 = 50
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #55
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_new_pmap_field_type.
# Failed to parse test_make_pmap_field_type_caches_result.
# Failed to parse test_make_pmap_field_type_different_types_create_different_classes.
# Partially parsed test_make_pmap_field_type_has_reduce_method. Retrieved 1/6 statements.
# Failed to parse test_make_pmap_field_type_with_builtin_types.


def test_case_0():
    var_0 = '__reduce__'



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_check_global_invariants_no_errors. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'test_subject'
    var_1 = True
    var_2 = False
    assert var_2 is True



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_sequence_field_predicate_line_26. Retrieved 9/16 statements.


def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'type'
    var_5 = False
    var_6 = 'a'
    var_7 = 'b'
    var_8 = [var_6, var_7]



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_sequence_field_with_optional_true. Retrieved 5/12 statements.
# Partially parsed test_sequence_field_with_optional_false. Retrieved 6/11 statements.
# Partially parsed test_sequence_field_factory_with_none_when_optional. Retrieved 3/8 statements.
# Partially parsed test_sequence_field_factory_with_value_when_optional. Retrieved 5/10 statements.
# Partially parsed test_sequence_field_factory_non_optional. Retrieved 5/10 statements.
# Partially parsed test_sequence_field_with_invariant. Retrieved 2/7 statements.
# Partially parsed test_sequence_field_with_item_invariant. Retrieved 3/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.field()
    var_5 = [var_4]

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.field()
    var_6 = [var_5]

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = None

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_2, var_3]

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = False
    var_1 = []

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_check_global_invariants_all_pass. Retrieved 1/7 statements.
# Partially parsed test_check_global_invariants_single_failure. Retrieved 1/8 statements.
# Partially parsed test_check_global_invariants_multiple_failures. Retrieved 1/10 statements.
# Partially parsed test_check_global_invariants_with_different_subject_types. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 'test_subject'

def test_case_0():
    var_0 = 'test_subject'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 'test_subject'
    var_1 = bool(False)
    assert var_1 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = []
    var_2 = module_0.check_global_invariants(var_0, var_1)

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = 42



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_sequence_field_with_optional_true. Retrieved 2/5 statements.
# Partially parsed test_sequence_field_with_optional_false. Retrieved 2/5 statements.
# Partially parsed test_sequence_field_factory_with_none_when_optional. Retrieved 3/7 statements.
# Partially parsed test_sequence_field_factory_with_list_when_optional. Retrieved 5/10 statements.
# Partially parsed test_sequence_field_with_item_invariant. Retrieved 2/7 statements.
# Partially parsed test_sequence_field_with_invariant. Retrieved 2/7 statements.


def test_case_0():
    var_0 = True
    var_1 = []

def test_case_0():
    var_0 = False
    var_1 = []

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = None

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_2, var_3]

def test_case_0():
    var_0 = False
    var_1 = []

def test_case_0():
    var_0 = False
    var_1 = []



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_set_fields_with_pfield_instances. Retrieved 6/12 statements.
# Partially parsed test_set_fields_with_inherited_fields. Retrieved 4/16 statements.
# Partially parsed test_set_fields_with_multiple_bases. Retrieved 5/19 statements.
# Partially parsed test_set_fields_preserves_non_pfield_attributes. Retrieved 7/12 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = {}
    var_1 = ()
    var_2 = 'fields'
    var_3 = module_0.set_fields(var_0, var_1, var_2)
    var_4 = 'fields'
    var_5 = bool('fields' in var_0)
    assert var_5 is True
    var_6 = var_0['fields']
    var_7 = bool(var_0['fields'] == {})
    assert var_7 is True

def test_case_0():
    var_0 = 'attr1'
    var_1 = 'attr2'
    var_2 = 'other'
    var_3 = 'value'
    var_4 = ()
    var_5 = 'fields'
    var_6 = 'fields'
    var_7 = 'attr1'
    var_8 = 'attr2'

def test_case_0():
    var_0 = 'fields'
    var_1 = 'inherited'
    var_2 = 'new_field'
    var_3 = 'fields'
    var_4 = 'fields'
    var_5 = 'inherited'
    var_6 = 'new_field'
    var_7 = 'new_field'

def test_case_0():
    var_0 = 'fields'
    var_1 = 'f1'
    var_2 = 'f2'
    var_3 = {}
    var_4 = 'fields'
    var_5 = 'fields'
    var_6 = bool('fields' in var_3)
    assert var_6 is True
    var_7 = 'f1'
    var_8 = bool('f1' in var_3['fields'])
    assert var_8 is True
    var_9 = 'f2'
    var_10 = bool('f2' in var_3['fields'])
    assert var_10 is True

def test_case_0():
    var_0 = 'field'
    var_1 = 'regular_attr'
    var_2 = 'number'
    var_3 = 'value'
    var_4 = 42
    var_5 = ()
    var_6 = 'fields'
    var_7 = 'field'



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 1/9 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 2/5 statements.
# Partially parsed test_serialize_with_checked_type_and_custom_serializer. Retrieved 2/13 statements.
# Partially parsed test_serialize_with_regular_value_and_serializer. Retrieved 2/5 statements.
# Partially parsed test_serialize_with_dict_value. Retrieved 5/8 statements.


def test_case_0():
    var_0 = []
    var_1 = 'json'

def test_case_0():
    var_0 = 'json'
    var_1 = 'test_value'

def test_case_0():
    var_0 = 'xml'
    var_1 = 'custom_xml_'

def test_case_0():
    var_0 = 'yaml'
    var_1 = 42

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'json'
    var_4 = str(var_2)



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 4/14 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 8/19 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.pmap(var_6)



# Parsed testcases at query #65
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_new_pmap_class.
# Failed to parse test_make_pmap_field_type_returns_cached_type.
# Failed to parse test_make_pmap_field_type_different_types_creates_different_classes.
# Partially parsed test_make_pmap_field_type_has_reduce_method. Retrieved 1/7 statements.
# Failed to parse test_make_pmap_field_type_with_builtin_types.


def test_case_0():
    var_0 = '__reduce__'



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_sequence_field_invariant_parameter_with_pfield_no_invariant. Retrieved 6/10 statements.


def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 'invariant'



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 4/12 statements.
# Partially parsed test_restore_seq_field_pickle_empty. Retrieved 1/9 statements.
# Partially parsed test_restore_seq_field_pickle_with_strings. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_pmap_field_optional_predicate. Retrieved 8/16 statements.


def test_case_0():
    var_0 = True
    var_1 = 'factory'
    var_2 = 'type'
    var_3 = None
    var_4 = 'a'
    var_5 = 'b'
    var_6 = 2
    var_7 = {var_4: var_0, var_5: var_6}



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_pmap_field_optional_type_predicate. Retrieved 2/7 statements.


def test_case_0():
    var_0 = True
    var_1 = False



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_pmap_field_predicate_line_25_optional_true. Retrieved 2/8 statements.
# Partially parsed test_pmap_field_predicate_line_25_optional_false. Retrieved 2/11 statements.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = [var_1]

def test_case_0():
    var_0 = False
    var_1 = None
    var_2 = [var_1]



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_pmap_field_optional_true_creates_factory_that_handles_none. Retrieved 3/9 statements.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = {}



# Parsed testcases at query #72
#--------------------------

# Failed to parse test_check_field_parameters_predicate_line_3_false.




# Parsed testcases at query #73
#--------------------------

# Partially parsed test_pmap_field_basic. Retrieved 1/4 statements.
# Partially parsed test_pmap_field_optional_true. Retrieved 2/5 statements.
# Partially parsed test_pmap_field_optional_false. Retrieved 2/5 statements.
# Partially parsed test_pmap_field_with_invariant. Retrieved 1/6 statements.
# Partially parsed test_pmap_field_factory_none_optional. Retrieved 2/5 statements.
# Partially parsed test_pmap_field_factory_dict_optional. Retrieved 5/8 statements.
# Partially parsed test_pmap_field_factory_dict_not_optional. Retrieved 6/9 statements.
# Failed to parse test_pmap_field_different_types.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.pmap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.pmap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.pmap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.pmap()

def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = True
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 2
    var_4 = {var_1: var_0, var_2: var_3}

def test_case_0():
    var_0 = False
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 1/7 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 2/5 statements.
# Partially parsed test_serialize_with_checked_type_and_custom_serializer. Retrieved 1/8 statements.
# Partially parsed test_serialize_with_non_checked_type_and_no_serializer. Retrieved 2/6 statements.
# Partially parsed test_serialize_with_different_formats. Retrieved 4/8 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'json'

def test_case_0():
    var_0 = 'xml'
    var_1 = 'test_value'

def test_case_0():
    var_0 = []
    var_1 = 'json'

def test_case_0():
    var_0 = []
    var_1 = 'csv'
    var_2 = 'data'

def test_case_0():
    var_0 = 'json'
    var_1 = 'value1'
    var_2 = 'xml'
    var_3 = 'value2'



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_serialize_checked_type_with_no_serializer. Retrieved 1/8 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'json'



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 8/21 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.pmap(var_6)



# Parsed testcases at query #77
#--------------------------

# Failed to parse test_sequence_field_invariant_parameter_default_value.




# Parsed testcases at query #78
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 6/18 statements.
# Partially parsed test_restore_seq_field_pickle_empty. Retrieved 3/14 statements.
# Partially parsed test_restore_seq_field_pickle_with_strings. Retrieved 6/17 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'TestVec'
    var_5 = {}
    var_6 = [var_4, var_1, var_5]

def test_case_0():
    var_0 = []
    var_1 = 'TestVecStr'
    assert var_1 == 0
    var_2 = {}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'TestVecStr2'
    var_5 = {}
    var_6 = [var_4, var_1, var_5]



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_factory_parameter_assignment. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = lambda : var_0
    var_2 = None
    var_3 = False



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_pfield_constructor. Retrieved 2/10 statements.
# Partially parsed test_pfield_constructor_with_none_values. Retrieved 2/4 statements.
# Partially parsed test_pfield_constructor_with_multiple_types. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 10
    var_1 = True

def test_case_0():
    var_0 = None
    var_1 = False

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = 0
    var_3 = lambda : var_2
    var_4 = lambda x: x



# Parsed testcases at query #81
#--------------------------

# Failed to parse test_sequence_field_invariant_parameter_type.




# Parsed testcases at query #82
#--------------------------

# Failed to parse test_check_field_parameters_predicate_line_3_false.




# Parsed testcases at query #83
#--------------------------

# Failed to parse test_pmap_field_basic.
# Partially parsed test_pmap_field_with_optional_true. Retrieved 1/3 statements.
# Partially parsed test_pmap_field_with_optional_false. Retrieved 1/3 statements.
# Failed to parse test_pmap_field_with_invariant.
# Failed to parse test_pmap_field_multiple_calls_same_types.
# Failed to parse test_pmap_field_different_types.
# Partially parsed test_pmap_field_optional_factory_with_none. Retrieved 2/5 statements.
# Partially parsed test_pmap_field_optional_factory_with_dict. Retrieved 3/6 statements.


def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = False

def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = True
    var_1 = 'a'
    var_2 = {var_1: var_0}



# Parsed testcases at query #84
#--------------------------

# Failed to parse test_pmap_field_basic.
# Partially parsed test_pmap_field_optional_true. Retrieved 2/7 statements.
# Partially parsed test_pmap_field_optional_false. Retrieved 1/5 statements.
# Failed to parse test_pmap_field_with_invariant.
# Failed to parse test_pmap_field_type_int_int.
# Failed to parse test_pmap_field_type_float_str.
# Partially parsed test_pmap_field_optional_and_invariant. Retrieved 1/7 statements.


def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = False

def test_case_0():
    var_0 = True



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_make_seq_field_type_creates_new_type. Retrieved 4/12 statements.
# Partially parsed test_make_seq_field_type_caches_result. Retrieved 1/7 statements.
# Partially parsed test_make_seq_field_type_different_item_types. Retrieved 1/6 statements.
# Failed to parse test_make_seq_field_type_with_invariant.
# Partially parsed test_make_seq_field_type_has_reduce_method. Retrieved 2/9 statements.


def test_case_0():
    var_0 = None
    var_1 = '__type__'
    var_2 = '__invariant__'
    var_3 = '__reduce__'

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = None
    var_1 = '__reduce__'



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_pmap_field_optional_false_predicate. Retrieved 2/5 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.pmap()



# Parsed testcases at query #87
#--------------------------

# Partially parsed test_pfield_factory_assignment. Retrieved 2/8 statements.


def test_case_0():
    var_0 = None
    var_1 = False



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_pmap_field_optional_false_predicate. Retrieved 1/5 statements.


def test_case_0():
    var_0 = False
    var_1 = []



# Parsed testcases at query #89
#--------------------------

# Partially parsed test_pmap_field_optional_false_predicate. Retrieved 1/4 statements.


def test_case_0():
    var_0 = False



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_pmap_field_returns_field_with_correct_type. Retrieved 2/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.field()
    var_2 = [var_1]



# Parsed testcases at query #91
#--------------------------

# Partially parsed test_pmap_field_optional_predicate. Retrieved 3/10 statements.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = {}



# Parsed testcases at query #92
#--------------------------

# Partially parsed test_check_field_parameters_predicate_line_3_false. Retrieved 7/19 statements.


def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = lambda : var_0
    var_4 = lambda x: x
    var_5 = True
    var_6 = False
    assert var_6 is True



# Parsed testcases at query #93
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 1/12 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'json'



# Parsed testcases at query #94
#--------------------------

# Partially parsed test_sequence_field_invariant_parameter_has_default_value. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'invariant'



# Parsed testcases at query #95
#--------------------------

# Partially parsed test_pmap_field_type_predicate_optional_true. Retrieved 2/13 statements.
# Partially parsed test_pmap_field_type_predicate_optional_false. Retrieved 1/8 statements.


def test_case_0():
    var_0 = True
    var_1 = 0

def test_case_0():
    var_0 = False



# Parsed testcases at query #96
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_new_type.
# Failed to parse test_make_pmap_field_type_caches_type.
# Failed to parse test_make_pmap_field_type_different_types_different_results.
# Partially parsed test_make_pmap_field_type_has_reduce_method. Retrieved 1/6 statements.
# Failed to parse test_make_pmap_field_type_with_multiple_key_types.
# Failed to parse test_make_pmap_field_type_preserves_checked_pmap_inheritance.


def test_case_0():
    var_0 = '__reduce__'



# Parsed testcases at query #97
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 6/16 statements.
# Partially parsed test_restore_seq_field_pickle_empty. Retrieved 3/13 statements.
# Partially parsed test_restore_seq_field_pickle_with_data. Retrieved 6/16 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 'PVec'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = []
    var_2 = 'PVec'
    assert var_2 == 0

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1.5
    var_2 = 2.5
    var_3 = 3.5
    var_4 = [var_1, var_2, var_3]
    var_5 = 'PVec'



# Parsed testcases at query #98
#--------------------------

# Partially parsed test_make_seq_field_type. Retrieved 4/21 statements.


def test_case_0():
    var_0 = None
    var_1 = '__type__'
    var_2 = '__invariant__'
    var_3 = '__reduce__'



# Parsed testcases at query #99
#--------------------------

# Partially parsed test_check_global_invariants_with_no_violations. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'test_subject'
    var_1 = True
    var_2 = False
    assert var_2 is True



# Parsed testcases at query #100
#--------------------------

# Partially parsed test_set_fields_predicate_isinstance_pfield. Retrieved 4/16 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'not_a_pfield'
    var_3 = 'fields'



# Parsed testcases at query #101
#--------------------------

# Partially parsed test_pfield_factory_assignment. Retrieved 2/8 statements.


def test_case_0():
    var_0 = None
    var_1 = False



# Parsed testcases at query #102
#--------------------------




import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = (var_0, var_1)
    var_5 = lambda x: var_4
    var_6 = [var_3, var_5]
    var_7 = 'test_subject'
    var_8 = module_0.check_global_invariants(var_7, var_6)
    assert var_8 is None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = False
    var_5 = 'ERROR_CODE_1'
    var_6 = (var_4, var_5)
    var_7 = lambda x: var_6
    var_8 = [var_3, var_7]
    var_9 = 'test_subject'
    var_10 = module_0.check_global_invariants(var_9, var_8)
    var_11 = bool(False)
    assert var_11 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = False
    var_1 = 'ERROR_CODE_1'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = 'ERROR_CODE_2'
    var_5 = (var_0, var_4)
    var_6 = lambda x: var_5
    var_7 = True
    var_8 = None
    var_9 = (var_7, var_8)
    var_10 = lambda x: var_9
    var_11 = [var_3, var_6, var_10]
    var_12 = 'test_subject'
    var_13 = module_0.check_global_invariants(var_12, var_11)
    var_14 = bool(False)
    assert var_14 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = []
    var_1 = 'test_subject'
    var_2 = module_0.check_global_invariants(var_1, var_0)
    assert var_2 is None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = False
    var_5 = 'MISSING_KEY'
    var_6 = (var_4, var_5)
    var_7 = lambda x: var_3 if var_0 in x else var_6
    var_8 = [var_7]
    var_9 = 'value'
    var_10 = {var_0: var_9}
    var_11 = module_0.check_global_invariants(var_10, var_8)
    assert var_11 is None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = False
    var_5 = 'MISSING_KEY'
    var_6 = (var_4, var_5)
    var_7 = lambda x: var_3 if var_0 in x else var_6
    var_8 = [var_7]
    var_9 = 'other_key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = module_0.check_global_invariants(var_11, var_8)
    var_13 = bool(False)
    assert var_13 is True



# Parsed testcases at query #103
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 6/14 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)



# Parsed testcases at query #104
#--------------------------

# Failed to parse test_pmap_field_basic.
# Partially parsed test_pmap_field_optional_true. Retrieved 2/7 statements.
# Partially parsed test_pmap_field_optional_false. Retrieved 1/5 statements.
# Failed to parse test_pmap_field_with_invariant.
# Failed to parse test_pmap_field_type_attribute.
# Failed to parse test_pmap_field_factory_callable.
# Partially parsed test_pmap_field_optional_factory_with_dict. Retrieved 3/6 statements.


def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = False

def test_case_0():
    var_0 = True
    var_1 = 'a'
    var_2 = {var_1: var_0}



# Parsed testcases at query #105
#--------------------------

# Partially parsed test_pfield_constructor. Retrieved 2/9 statements.
# Partially parsed test_pfield_constructor_with_none_values. Retrieved 2/3 statements.
# Partially parsed test_pfield_constructor_with_tuple_type. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 'initial_value'
    var_1 = True

def test_case_0():
    var_0 = None
    var_1 = False

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = 42



# Parsed testcases at query #106
#--------------------------

# Failed to parse test_pmap_field_basic.
# Partially parsed test_pmap_field_with_optional_true. Retrieved 1/5 statements.
# Partially parsed test_pmap_field_with_optional_false. Retrieved 1/5 statements.
# Failed to parse test_pmap_field_with_invariant.
# Failed to parse test_pmap_field_initial_value.
# Failed to parse test_pmap_field_type_attribute.
# Partially parsed test_pmap_field_optional_factory_with_none. Retrieved 2/5 statements.
# Partially parsed test_pmap_field_optional_factory_with_dict. Retrieved 3/6 statements.
# Partially parsed test_pmap_field_non_optional_factory. Retrieved 4/7 statements.


def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = False

def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = True
    var_1 = 'key'
    var_2 = {var_1: var_0}

def test_case_0():
    var_0 = False
    var_1 = 'key'
    var_2 = 1
    var_3 = {var_1: var_2}



# Parsed testcases at query #107
#--------------------------

# Partially parsed test_set_fields_predicate_isinstance_pfield. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'not_a_pfield'
    var_3 = 'test_name'



# Parsed testcases at query #108
#--------------------------

# Partially parsed test_sequence_field_non_optional_creates_field_with_correct_type. Retrieved 6/13 statements.
# Partially parsed test_sequence_field_optional_creates_field_with_none_type. Retrieved 5/12 statements.
# Partially parsed test_sequence_field_factory_returns_none_for_optional_with_none_argument. Retrieved 3/7 statements.
# Partially parsed test_sequence_field_factory_creates_checked_instance_for_optional_with_value. Retrieved 5/11 statements.
# Partially parsed test_sequence_field_factory_non_optional_creates_checked_instance. Retrieved 6/11 statements.
# Partially parsed test_sequence_field_initial_value_is_set. Retrieved 5/10 statements.
# Partially parsed test_sequence_field_with_item_invariant. Retrieved 3/8 statements.
# Partially parsed test_sequence_field_with_field_invariant. Retrieved 3/10 statements.
# Partially parsed test_sequence_field_optional_factory_with_factory_fields. Retrieved 6/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.field()
    var_6 = [var_5]

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.field()
    var_5 = [var_4]

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = None

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_2, var_3]

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = [var_1, var_2, var_4]

def test_case_0():
    var_0 = False
    var_1 = 5
    var_2 = 6
    var_3 = 7
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = [var_1]

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = [var_1]

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = 2
    var_3 = [var_0, var_2]
    var_4 = None
    var_5 = False



# Parsed testcases at query #109
#--------------------------

# Partially parsed test_check_global_invariants_raises_exception_when_invariant_fails. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'test_subject'
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #110
#--------------------------

# Partially parsed test_pfield_factory_assignment. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = lambda : var_0
    var_2 = None
    var_3 = False



# Parsed testcases at query #111
#--------------------------

# Partially parsed test_pmap_field_type_predicate_line_25. Retrieved 3/11 statements.


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = None
    var_3 = [var_2]



# Parsed testcases at query #112
#--------------------------

# Partially parsed test_pmap_field_optional_predicate. Retrieved 2/6 statements.


def test_case_0():
    var_0 = True
    var_1 = False



# Parsed testcases at query #113
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_pmap_subclass.
# Failed to parse test_make_pmap_field_type_generates_correct_name.
# Failed to parse test_make_pmap_field_type_caches_types.
# Failed to parse test_make_pmap_field_type_different_types_creates_different_classes.
# Partially parsed test_make_pmap_field_type_has_reduce_method. Retrieved 1/6 statements.
# Failed to parse test_make_pmap_field_type_with_builtin_types.


def test_case_0():
    var_0 = '__reduce__'



# Parsed testcases at query #114
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 1/9 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 2/6 statements.
# Partially parsed test_serialize_with_checked_type_and_custom_serializer. Retrieved 1/11 statements.
# Partially parsed test_serialize_with_non_checked_type. Retrieved 2/6 statements.


def test_case_0():
    var_0 = []
    var_1 = 'json'

def test_case_0():
    var_0 = []
    var_1 = 'xml'
    var_2 = 'test_value'

def test_case_0():
    var_0 = []
    var_1 = 'yaml'

def test_case_0():
    var_0 = []
    var_1 = 'json'
    var_2 = 42



# Parsed testcases at query #115
#--------------------------

# Partially parsed test_make_seq_field_type. Retrieved 8/24 statements.


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = '__name__'
    var_7 = 'int'



