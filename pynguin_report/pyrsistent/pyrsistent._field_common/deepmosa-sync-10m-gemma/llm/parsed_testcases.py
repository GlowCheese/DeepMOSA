####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = None
    var_5 = (var_3, var_4)
    var_6 = lambda x: var_5
    var_7 = 'none'
    var_8 = (var_3, var_7)
    var_9 = lambda x: var_8
    var_10 = [var_6, var_9]
    var_11 = module_0.check_global_invariants(var_2, var_10)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = None
    var_5 = (var_3, var_4)
    var_6 = lambda x: var_5
    var_7 = False
    var_8 = 'ERR_01'
    var_9 = (var_7, var_8)
    var_10 = lambda x: var_9
    var_11 = [var_6, var_10]
    var_12 = module_0.check_global_invariants(var_2, var_11)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = 'ERR_01'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = True
    var_6 = None
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = 'ERR_02'
    var_10 = (var_1, var_9)
    var_11 = lambda x: var_10
    var_12 = [var_4, var_8, var_11]
    var_13 = module_0.check_global_invariants(var_0, var_12)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = module_0.check_global_invariants(var_0, var_1)



# Parsed testcases at query #2
#--------------------------

# Failed to parse test_is_type_cls_returns_true_for_set_field_type.
# Partially parsed test_is_type_cls_returns_false_for_empty_tuple_field_type. Retrieved 1/3 statements.
# Failed to parse test_is_type_cls_returns_true_for_matching_tuple_element.
# Failed to parse test_is_type_cls_returns_false_for_mismatched_tuple_element.
# Failed to parse test_is_type_cls_returns_true_for_subclass_in_tuple.
# Partially parsed test_is_type_cls_handles_string_type_references. Retrieved 2/4 statements.
# Failed to parse test_is_type_cls_returns_false_for_unrelated_class_in_tuple.


def test_case_0():
    var_0 = ()

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = (var_0,)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_set_fields_merging_bases. Retrieved 2/30 statements.
# Partially parsed test_set_fields_with_pfield_migration. Retrieved 3/19 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'target'
    var_2 = var_0['target']
    var_3 = bool(var_0['target'] == {'x': 10, 'y': 20, 'z': 30})
    assert var_3 is True

def test_case_0():
    var_0 = 'target'
    var_1 = 'field_key'
    var_2 = {}
    var_3 = 'field_key'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_check_field_parameters_valid_input. Retrieved 2/12 statements.
# Partially parsed test_check_field_parameters_invalid_type_element. Retrieved 1/12 statements.
# Partially parsed test_check_field_parameters_invalid_initial_type. Retrieved 1/12 statements.
# Partially parsed test_check_field_parameters_non_callable_invariant. Retrieved 1/12 statements.
# Partially parsed test_check_field_parameters_non_callable_factory. Retrieved 1/12 statements.
# Partially parsed test_check_field_parameters_non_callable_serializer. Retrieved 1/12 statements.


import builtins as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.object(*var_0, **var_1)
    var_3 = 'hello'

def test_case_0():
    var_0 = 123

def test_case_0():
    var_0 = 'not an int'

def test_case_0():
    var_0 = 'not callable'

def test_case_0():
    var_0 = 'not callable'

def test_case_0():
    var_0 = 'not callable'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_set_fields_basic_merging. Retrieved 14/41 statements.
# Partially parsed test_set_fields_merging_logic. Retrieved 24/33 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 'b'
    var_3 = 2
    var_4 = 'c'
    var_5 = 3
    var_6 = 'extra'
    var_7 = 'my_field'
    var_8 = 'x'
    var_9 = 10
    var_10 = 'y'
    var_11 = 20
    var_12 = 'z'
    var_13 = 30

import builtins as module_0

def test_case_0():
    var_0 = 'Base1'
    var_1 = ()
    var_2 = 'shared'
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = [var_0, var_1, var_6]
    var_8 = {}
    var_9 = module_0.type(*var_7, **var_8)
    var_10 = 'Base2'
    var_11 = ()
    var_12 = 'other'
    var_13 = 'b'
    var_14 = 2
    var_15 = {var_13: var_14}
    var_16 = 'c'
    var_17 = 3
    var_18 = {var_16: var_17}
    var_19 = {var_2: var_15, var_12: var_18}
    var_20 = [var_10, var_11, var_19]
    var_21 = {}
    var_22 = module_0.type(*var_20, **var_21)
    var_23 = 'field_to_move'
    var_24 = 'unrelated'
    var_25 = 5
    var_26 = [var_9, var_22]
    var_27 = 'shared'
    var_28 = 'field_to_move'
    var_29 = 'extra_data'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_set_fields_predicate_is_false. Retrieved 4/14 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'test_name'
    var_2 = 'some_key'
    var_3 = var_0[var_2]



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_sequence_field_creates_correctly_with_checked_pvector. Retrieved 3/18 statements.
# Partially parsed test_sequence_field_handles_optional_parameter. Retrieved 3/9 statements.
# Partially parsed test_sequence_field_initial_value_is_processed. Retrieved 5/9 statements.


def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = 'create'

def test_case_0():
    var_0 = True
    var_1 = None
    assert var_1 is None
    var_2 = None

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = False



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_pfield_constructor_initialization. Retrieved 6/8 statements.


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = True
    var_4 = lambda x: x
    var_5 = lambda x: str(x)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_make_seq_field_type_returns_cached_type. Retrieved 1/7 statements.
# Partially parsed test_make_seq_field_type_creates_new_subclass. Retrieved 2/10 statements.
# Partially parsed test_make_seq_field_type_name_generation. Retrieved 1/4 statements.


def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = True
    var_1 = False

def test_case_0():
    var_0 = True
    var_1 = 'Int'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_check_field_parameters_predicate_true. Retrieved 1/14 statements.


def test_case_0():
    var_0 = 'string'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_field_valid_single_type. Retrieved 1/3 statements.
# Partially parsed test_field_valid_multiple_types. Retrieved 1/4 statements.
# Partially parsed test_field_valid_string_type. Retrieved 2/4 statements.
# Partially parsed test_field_invalid_type_parameter_raises_error. Retrieved 3/7 statements.
# Partially parsed test_field_invalid_initial_type_raises_error. Retrieved 6/11 statements.
# Partially parsed test_field_non_callable_invariant_raises_error. Retrieved 2/6 statements.
# Partially parsed test_field_non_callable_factory_raises_error. Retrieved 2/6 statements.
# Partially parsed test_field_non_callable_serializer_raises_error. Retrieved 2/6 statements.
# Partially parsed test_field_wrapped_invariant. Retrieved 2/7 statements.


def test_case_0():
    var_0 = False

def test_case_0():
    var_0 = False

def test_case_0():
    var_0 = 'int'
    var_1 = False

def test_case_0():
    var_0 = 123
    var_1 = [var_0]
    var_2 = False

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = 'not an int'
    var_3 = False
    var_4 = lambda x: x
    var_5 = lambda x: x

def test_case_0():
    var_0 = 'not callable'
    var_1 = False

def test_case_0():
    var_0 = False
    var_1 = 'not callable'

def test_case_0():
    var_0 = False
    var_1 = 'not callable'

def test_case_0():
    var_0 = False
    var_1 = 1



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_sequence_field_creates_type_with_correct_parameters. Retrieved 8/16 statements.
# Partially parsed test_sequence_field_optional_true_handles_none. Retrieved 2/8 statements.
# Partially parsed test_sequence_field_initial_value_assignment. Retrieved 3/6 statements.


def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = False
    var_3 = 2
    var_4 = 3
    var_5 = [var_0, var_3, var_4]
    var_6 = 'type'
    var_7 = 'initial'

def test_case_0():
    var_0 = True
    var_1 = []

def test_case_0():
    var_0 = 10
    var_1 = [var_0]
    var_2 = False



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_pfield_constructor_initialization. Retrieved 6/8 statements.


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = True
    var_4 = lambda x: x
    var_5 = lambda x: str(x)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_restore_seq_field_pickle_success. Retrieved 9/20 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'data'
    var_5 = 'fields'
    var_6 = [var_0, var_1, var_2]
    var_7 = set()
    var_8 = {var_4: var_6, var_5: var_7}



# Parsed testcases at query #15
#--------------------------

# Failed to parse test_pmap_field_basic.
# Partially parsed test_pmap_field_optional. Retrieved 3/5 statements.
# Failed to parse test_pmap_field_with_invariant.
# Partially parsed test_pmap_field_factory_logic_for_optional. Retrieved 1/3 statements.
# Partially parsed test_pmap_field_name_generation. Retrieved 1/4 statements.


import builtins as module_0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.type(*var_2, **var_3)

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = 'type'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_check_field_parameters_predicate_false_with_types. Retrieved 1/11 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_sequence_field_factory_assignment_not_optional. Retrieved 1/7 statements.


def test_case_0():
    var_0 = False



# Parsed testcases at query #18
#--------------------------




import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = [var_2]
    var_4 = module_0.check_global_invariants(var_0, var_3)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_pmap_field_not_optional. Retrieved 1/7 statements.


def test_case_0():
    var_0 = False



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_serialize_with_no_serializer_and_checked_type. Retrieved 3/10 statements.
# Partially parsed test_serialize_with_standard_serializer. Retrieved 2/6 statements.
# Partially parsed test_serialize_with_checked_type_and_specific_serializer. Retrieved 2/9 statements.
# Partially parsed test_serialize_with_checked_type_and_no_serializer_bypass. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'NONE'
    var_1 = 'data'
    var_2 = 'json'

def test_case_0():
    var_0 = 'xml'
    var_1 = 'plain_text'

def test_case_0():
    var_0 = 'NONE'
    var_1 = 'json'

def test_case_0():
    var_0 = 'NONE'
    var_1 = 'yaml'



# Parsed testcases at query #21
#--------------------------

# Failed to parse test_pmap_field_basic.
# Partially parsed test_pmap_field_optional. Retrieved 3/5 statements.
# Partially parsed test_pmap_field_factory_logic. Retrieved 1/5 statements.
# Failed to parse test_pmap_field_invariant_pass_through.


import builtins as module_0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.type(*var_2, **var_3)

def test_case_0():
    var_0 = True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint_returns_false_when_ignore_extra_is_false. Retrieved 1/5 statements.
# Partially parsed test_is_field_ignore_extra_complaint_returns_false_when_not_type_cls. Retrieved 1/5 statements.
# Partially parsed test_is_field_ignore_extra_complaint_returns_true_when_param_exists. Retrieved 1/8 statements.
# Partially parsed test_is_field_ignore_extra_complaint_returns_false_when_param_missing. Retrieved 1/8 statements.
# Partially parsed test_is_field_ignore_extra_complaint_with_set_type_and_param_exists. Retrieved 1/8 statements.


def test_case_0():
    var_0 = False

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_check_field_parameters_valid. Retrieved 5/15 statements.
# Partially parsed test_check_field_parameters_invalid_type_element. Retrieved 3/14 statements.
# Partially parsed test_check_field_parameters_invalid_initial_type. Retrieved 1/12 statements.
# Partially parsed test_check_field_parameters_invalid_invariant. Retrieved 1/12 statements.
# Partially parsed test_check_field_parameters_invalid_factory. Retrieved 1/12 statements.
# Partially parsed test_check_field_parameters_invalid_serializer. Retrieved 1/12 statements.


def test_case_0():
    var_0 = 'hello'
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = lambda : var_1
    var_4 = lambda x: x

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'Type parameter expected'

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



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint_predicate_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #25
#--------------------------

# Failed to parse test_check_field_parameters_predicate_false_with_type.
# Partially parsed test_check_field_parameters_predicate_false_with_str. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'string_type'
    var_1 = [var_0]



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_check_type_valid_single_type. Retrieved 2/7 statements.
# Partially parsed test_check_type_valid_multiple_types. Retrieved 2/7 statements.
# Partially parsed test_check_type_no_type_requirement. Retrieved 2/7 statements.
# Partially parsed test_check_type_invalid_type_raises_error. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'my_field'
    var_1 = 10

def test_case_0():
    var_0 = 'my_field'
    var_1 = 'hello'

def test_case_0():
    var_0 = 'my_field'
    var_1 = 123

def test_case_0():
    var_0 = 'my_field'
    var_1 = 'not an int'
    var_2 = 'Invalid type for field MyClass.my_field, was str'



# Parsed testcases at query #27
#--------------------------

# Failed to parse test_pmap_field_basic_creation.
# Partially parsed test_pmap_field_optional. Retrieved 5/10 statements.
# Failed to parse test_pmap_field_with_invariant.
# Failed to parse test_pmap_field_type_name_generation.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 1
    var_3 = 'a'
    var_4 = {var_2: var_3}



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_check_field_parameters_predicate_false. Retrieved 6/16 statements.


def test_case_0():
    var_0 = None
    var_1 = 10
    var_2 = True
    var_3 = lambda x: var_2
    var_4 = lambda : var_1
    var_5 = lambda x: x



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_pfield_init_assigns_factory_to_private_attr. Retrieved 6/9 statements.


def test_case_0():
    var_0 = None
    var_1 = lambda x: x
    var_2 = 'int'
    var_3 = None
    var_4 = 0
    var_5 = True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_check_global_invariants_raises_exception_when_invariant_fails. Retrieved 7/13 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = False
    var_2 = lambda x: var_1
    var_3 = [var_2]
    var_4 = 'error_1'
    var_5 = (var_4,)
    var_6 = module_0.check_global_invariants(var_0, var_3)
    var_7 = bool(True)
    assert var_7 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_check_field_parameters_predicate_true. Retrieved 1/15 statements.


def test_case_0():
    var_0 = 10



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_make_pmap_field_type_new_class_creation. Retrieved 2/8 statements.
# Failed to parse test_make_pmap_field_type_returns_cached_class.


def test_case_0():
    var_0 = '__key_type__'
    var_1 = '__value_type__'
    var_2 = 'StringToIntPMap'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_pfield_constructor_assignment. Retrieved 6/8 statements.


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = True
    var_4 = lambda x: x
    var_5 = lambda x: str(x)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint_returns_false_when_ignore_extra_is_false. Retrieved 1/5 statements.
# Partially parsed test_is_field_ignore_extra_complaint_returns_false_when_type_does_not_match. Retrieved 1/5 statements.
# Partially parsed test_is_field_ignore_extra_complaint_returns_true_when_factory_has_ignore_extra_param. Retrieved 1/8 statements.
# Partially parsed test_is_field_ignore_extra_complaint_returns_false_when_factory_lacks_ignore_extra_param. Retrieved 1/8 statements.
# Partially parsed test_is_field_ignore_extra_complaint_handles_set_type_correctly. Retrieved 1/8 statements.


def test_case_0():
    var_0 = False

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_check_type_valid_single_type. Retrieved 2/4 statements.
# Partially parsed test_check_type_valid_tuple_type. Retrieved 4/8 statements.
# Partially parsed test_check_type_no_type_constraint. Retrieved 6/8 statements.
# Partially parsed test_check_type_invalid_type_raises_error. Retrieved 3/7 statements.
# Partially parsed test_check_type_invalid_tuple_element_raises_error. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'age'
    var_1 = 25

def test_case_0():
    var_0 = 'data'
    var_1 = 'hello'
    var_2 = 'id'
    var_3 = 10

def test_case_0():
    var_0 = None
    var_1 = 'anything'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]

def test_case_0():
    var_0 = 'age'
    var_1 = 'not_an_int'
    var_2 = str(var_0)
    var_3 = 'Invalid type for field MockDest.age, was str'
    var_4 = bool('Invalid type for field MockDest.age, was str' in var_2)
    assert var_4 is True

def test_case_0():
    var_0 = 'value'
    var_1 = 'string_is_wrong'
    var_2 = str(var_1)
    var_3 = 'Invalid type for field MockDest.value, was str'
    var_4 = bool('Invalid type for field MockDest.value, was str' in var_2)
    assert var_4 is True



# Parsed testcases at query #4
#--------------------------




import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = None
    var_5 = (var_3, var_4)
    var_6 = lambda x: var_5
    var_7 = True
    var_8 = 'no error'
    var_9 = (var_7, var_8)
    var_10 = lambda x: var_9
    var_11 = [var_6, var_10]
    var_12 = module_0.check_global_invariants(var_2, var_11)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = 'ERR001'
    var_5 = (var_3, var_4)
    var_6 = lambda x: var_5
    var_7 = True
    var_8 = None
    var_9 = (var_7, var_8)
    var_10 = lambda x: var_9
    var_11 = [var_6, var_10]
    var_12 = module_0.check_global_invariants(var_2, var_11)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = 'ERR001'
    var_5 = (var_3, var_4)
    var_6 = lambda x: var_5
    var_7 = 'ERR002'
    var_8 = (var_3, var_7)
    var_9 = lambda x: var_8
    var_10 = True
    var_11 = None
    var_12 = (var_10, var_11)
    var_13 = lambda x: var_12
    var_14 = [var_6, var_9, var_13]
    var_15 = module_0.check_global_invariants(var_2, var_14)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'id'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.check_global_invariants(var_2, var_3)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_make_seq_field_type_returns_new_class. Retrieved 1/9 statements.
# Partially parsed test_make_seq_field_type_returns_cached_class. Retrieved 1/6 statements.
# Partially parsed test_make_seq_field_type_with_pvector. Retrieved 1/4 statements.


def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = False

def test_case_0():
    var_0 = True
    var_1 = 'Vector'



# Parsed testcases at query #6
#--------------------------

# Failed to parse test_make_pmap_field_type_new_class_creation.
# Failed to parse test_make_pmap_field_type_memoization.
# Failed to parse test_make_pmap_field_type_different_types_produce_different_classes.




# Parsed testcases at query #7
#--------------------------

# Partially parsed test_make_seq_field_type_returns_existing_type_if_cached. Retrieved 3/15 statements.
# Partially parsed test_make_seq_field_type_creates_new_subclass. Retrieved 1/11 statements.
# Partially parsed test_make_seq_field_type_sets_correct_name. Retrieved 1/12 statements.


def test_case_0():
    var_0 = 'CachedType'
    var_1 = {}
    var_2 = True

def test_case_0():
    var_0 = False

def test_case_0():
    var_0 = True



# Parsed testcases at query #8
#--------------------------

# Failed to parse test_types_to_names_with_simple_types.
# Failed to parse test_types_to_names_with_single_type.
# Failed to parse test_types_to_names_with_string_type_references.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ()
    var_1 = module_0._types_to_names(var_0)
    assert var_1 == ''



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_make_seq_field_type_creation. Retrieved 4/15 statements.
# Partially parsed test_make_seq_field_type_naming_logic. Retrieved 1/4 statements.


def test_case_0():
    var_0 = True
    var_1 = 'Vector'
    var_2 = False
    var_3 = 'Set'

def test_case_0():
    var_0 = True
    var_1 = 'Vector'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_check_global_invariants_raises_exception_when_invariant_fails. Retrieved 7/14 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = False
    var_2 = lambda x: var_1
    var_3 = [var_2]
    var_4 = 'error_1'
    var_5 = (var_4,)
    var_6 = module_0.check_global_invariants(var_0, var_3)
    var_7 = bool(True)
    assert var_7 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 9/24 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'result'
    var_5 = 'fields'
    var_6 = [var_0, var_1, var_2]
    var_7 = set()
    var_8 = {var_4: var_6, var_5: var_7}



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 4/12 statements.
# Partially parsed test_serialize_with_checked_type_and_standard_serializer. Retrieved 3/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda fmt, val: f'{fmt}:{val}'
    var_1 = 'json'
    var_2 = 'data'
    var_3 = module_0.serialize(var_0, var_1, var_2)
    assert var_3 == 'json:data'

def test_case_0():
    var_0 = None
    var_1 = 'PFIELD_NO_SERIALIZER'
    var_2 = None
    var_3 = 'xml'

def test_case_0():
    var_0 = lambda fmt, val: f'wrapped_{fmt}_{val.serialize(fmt)}'
    var_1 = None
    var_2 = 'json'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda fmt, val: f'{fmt}_{val}'
    var_1 = 'csv'
    var_2 = 123
    var_3 = module_0.serialize(var_0, var_1, var_2)
    assert var_3 == 'csv_123'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_is_type_cls_with_set_field_type. Retrieved 3/4 statements.
# Failed to parse test_is_type_cls_with_single_type_tuple_match.
# Failed to parse test_is_type_cls_with_single_type_tuple_mismatch.
# Partially parsed test_is_type_cls_with_empty_tuple. Retrieved 1/2 statements.
# Failed to parse test_is_type_cls_with_subclass_match.
# Failed to parse test_is_type_cls_with_bool_inheritance.
# Failed to parse test_is_type_cls_with_builtin_type_direct.


def test_case_0():
    var_0 = 'int'
    var_1 = [var_0]
    var_2 = set(var_1)

def test_case_0():
    var_0 = ()



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_pmap_field_basic. Retrieved 1/6 statements.
# Partially parsed test_pmap_field_optional. Retrieved 3/5 statements.
# Partially parsed test_pmap_field_invariant_passing. Retrieved 2/4 statements.
# Failed to parse test_pmap_field_factory_behavior.


def test_case_0():
    var_0 = 'type'

import builtins as module_0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.type(*var_2, **var_3)

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_is_type_cls_with_set_field_type. Retrieved 3/5 statements.
# Failed to parse test_is_type_cls_with_tuple_containing_matching_type.
# Failed to parse test_is_type_cls_with_tuple_containing_non_matching_type.
# Partially parsed test_is_type_cls_with_empty_tuple. Retrieved 1/3 statements.
# Failed to parse test_is_type_cls_with_subclass_match.
# Partially parsed test_is_type_cls_with_string_type_reference. Retrieved 2/4 statements.
# Failed to parse test_is_type_cls_with_mismatched_subclass.


def test_case_0():
    var_0 = 'int'
    var_1 = [var_0]
    var_2 = set(var_1)

def test_case_0():
    var_0 = ()

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = (var_0,)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_sequence_field_creates_checked_type_with_correct_attributes. Retrieved 8/24 statements.


def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = '__type__'
    var_3 = True
    var_4 = []
    var_5 = None
    var_6 = []
    var_7 = []



# Parsed testcases at query #17
#--------------------------




import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.object(*var_0, **var_1)
    var_3 = 'some_invariant'
    var_4 = 'int'
    var_5 = 0
    var_6 = True
    var_7 = None
    var_8 = module_1._PField(var_4, var_3, var_5, var_6, var_7, var_7)
    var_9 = var_8.invariant
    var_10 = bool(var_8.invariant == var_3)
    assert var_10 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_check_field_parameters_valid. Retrieved 6/16 statements.
# Partially parsed test_check_field_parameters_invalid_type_element. Retrieved 1/7 statements.


def test_case_0():
    var_0 = None
    var_1 = 10
    var_2 = True
    var_3 = lambda x: var_2
    var_4 = lambda : var_2
    var_5 = lambda x: x

def test_case_0():
    var_0 = 123



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_field_valid_single_type. Retrieved 1/5 statements.
# Partially parsed test_field_valid_multiple_types. Retrieved 1/4 statements.
# Partially parsed test_field_valid_string_type. Retrieved 2/4 statements.
# Partially parsed test_field_invalid_type_parameter_raises_error. Retrieved 2/6 statements.
# Partially parsed test_field_invalid_initial_value_type_raises_error. Retrieved 2/5 statements.
# Partially parsed test_field_non_callable_invariant_raises_error. Retrieved 2/5 statements.
# Partially parsed test_field_non_callable_factory_raises_error. Retrieved 2/5 statements.
# Partially parsed test_field_non_callable_serializer_raises_error. Retrieved 2/5 statements.
# Partially parsed test_field_wrapped_invariant. Retrieved 2/7 statements.


def test_case_0():
    var_0 = False

def test_case_0():
    var_0 = False

def test_case_0():
    var_0 = 'int'
    var_1 = False
    var_2 = 'int'

def test_case_0():
    var_0 = 1
    var_1 = False

def test_case_0():
    var_0 = 10
    var_1 = False

def test_case_0():
    var_0 = 'not_callable'
    var_1 = False

def test_case_0():
    var_0 = False
    var_1 = 'not_callable'

def test_case_0():
    var_0 = False
    var_1 = 'not_callable'

def test_case_0():
    var_0 = False
    var_1 = 10



# Parsed testcases at query #20
#--------------------------

# Failed to parse test_check_field_parameters_predicate_false.




# Parsed testcases at query #21
#--------------------------

# Partially parsed test_check_field_parameters_predicate_false_by_matching_type. Retrieved 2/12 statements.


def test_case_0():
    var_0 = None
    var_1 = 10



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_make_seq_field_type_returns_cached_type. Retrieved 1/6 statements.
# Partially parsed test_make_seq_field_type_creates_new_subclass. Retrieved 1/7 statements.


def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = False



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint_evaluates_false_when_ignore_extra_is_false. Retrieved 2/27 statements.


def test_case_0():
    var_0 = True
    var_1 = False



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 2/7 statements.
# Partially parsed test_serialize_with_checked_type_and_standard_serializer. Retrieved 3/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda fmt, val: f'{fmt}:{val}'
    var_1 = 'json'
    var_2 = 'data'
    var_3 = module_0.serialize(var_0, var_1, var_2)
    assert var_3 == 'json:data'

def test_case_0():
    var_0 = 'NO_SERIALIZER'
    var_1 = 'xml'

def test_case_0():
    var_0 = lambda fmt, val: f'{fmt}_{val}'
    var_1 = 'NO_SERIALIZER'
    var_2 = 'csv'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda fmt, val: f'{fmt}_{val}'
    var_1 = 'text'
    var_2 = 123
    var_3 = module_0.serialize(var_0, var_1, var_2)
    assert var_3 == 'text_123'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_check_field_parameters_raises_type_error_on_invalid_type_in_list. Retrieved 2/13 statements.


def test_case_0():
    var_0 = 123
    var_1 = [var_0]



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_restore_seq_field_pickle_success. Retrieved 5/10 statements.
# Partially parsed test_restore_seq_field_pickle_key_error. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_pfield_constructor_initialization. Retrieved 6/8 statements.


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = True
    var_4 = lambda x: x
    var_5 = lambda x: str(x)



# Parsed testcases at query #28
#--------------------------




import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'data'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = None
    var_5 = (var_3, var_4)
    var_6 = lambda s: var_5
    var_7 = True
    var_8 = 'none'
    var_9 = (var_7, var_8)
    var_10 = lambda s: var_9
    var_11 = [var_6, var_10]
    var_12 = module_0.check_global_invariants(var_2, var_11)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'data'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = None
    var_5 = (var_3, var_4)
    var_6 = lambda s: var_5
    var_7 = False
    var_8 = 'ERR_001'
    var_9 = (var_7, var_8)
    var_10 = lambda s: var_9
    var_11 = 'ERR_002'
    var_12 = (var_7, var_11)
    var_13 = lambda s: var_12
    var_14 = [var_6, var_10, var_13]
    var_15 = module_0.check_global_invariants(var_2, var_14)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = None
    var_1 = []
    var_2 = module_0.check_global_invariants(var_0, var_1)



# Parsed testcases at query #29
#--------------------------

# Failed to parse test_check_field_parameters_predicate_false_with_valid_type.




# Parsed testcases at query #30
#--------------------------

# Partially parsed test_check_type_predicate_false_when_value_matches_type. Retrieved 5/18 statements.


import collections as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = 'field1'
    var_2 = [var_1]
    var_3 = module_0.namedtuple(var_0, var_2)
    var_4 = 10



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_check_field_parameters_raises_type_error_on_invalid_type_element. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 123
    var_1 = [var_0]



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_check_field_parameters_valid. Retrieved 6/16 statements.
# Partially parsed test_check_field_parameters_invalid_type_element. Retrieved 1/11 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = None
    var_4 = lambda : var_3
    var_5 = lambda x: x

def test_case_0():
    var_0 = 123



# Parsed testcases at query #33
#--------------------------

# Failed to parse test_make_pmap_field_type_new_class.
# Failed to parse test_make_pmap_field_type_different_types.




# Parsed testcases at query #34
#--------------------------

# Partially parsed test_sequence_field_pvector_mandatory. Retrieved 5/8 statements.
# Partially parsed test_sequence_field_pset_optional. Retrieved 4/7 statements.
# Partially parsed test_sequence_field_with_invariant. Retrieved 4/10 statements.
# Partially parsed test_sequence_field_pvector_with_initial_val. Retrieved 4/5 statements.
# Partially parsed test_sequence_field_pset_with_none_as_optional. Retrieved 4/5 statements.


def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

import builtins as module_0

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = None
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.type(*var_3, **var_4)

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = [var_1]
    var_3 = 'CheckedPVector'

def test_case_0():
    var_0 = False
    var_1 = 10
    var_2 = 20
    var_3 = [var_1, var_2]

import builtins as module_0

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = None
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.type(*var_3, **var_4)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_check_field_parameters_predicate_true. Retrieved 6/17 statements.


def test_case_0():
    var_0 = None
    var_1 = 10
    var_2 = True
    var_3 = lambda x: var_2
    var_4 = lambda : var_1
    var_5 = lambda x: x



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_check_field_parameters_predicate_false. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 10
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = None
    var_4 = lambda : var_3
    var_5 = lambda x: x



# Parsed testcases at query #37
#--------------------------

# Failed to parse test_pmap_field_docstring_exists.




# Parsed testcases at query #38
#--------------------------

# Partially parsed test_check_field_parameters_success. Retrieved 5/15 statements.
# Partially parsed test_check_field_parameters_invalid_type_element. Retrieved 3/14 statements.
# Partially parsed test_check_field_parameters_invalid_initial_type. Retrieved 1/12 statements.
# Partially parsed test_check_field_parameters_invalid_invariant. Retrieved 1/12 statements.
# Partially parsed test_check_field_parameters_invalid_factory. Retrieved 1/12 statements.
# Partially parsed test_check_field_parameters_invalid_serializer. Retrieved 1/12 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = lambda : var_1
    var_4 = lambda x: x

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'Type parameter expected, not'

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Initial has invalid type'

def test_case_0():
    var_0 = 'not_callable'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Invariant must be callable'

def test_case_0():
    var_0 = 'not_callable'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Factory must be callable'

def test_case_0():
    var_0 = 'not_callable'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Serializer must be callable'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_pmap_field_not_optional_logic. Retrieved 1/3 statements.


def test_case_0():
    var_0 = False



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_check_type_predicate_false_when_type_matches. Retrieved 5/13 statements.


import collections as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = 'field1'
    var_2 = [var_1]
    var_3 = module_0.namedtuple(var_0, var_2)
    var_4 = 10



