####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_check_global_invariants_all_pass. Retrieved 1/7 statements.
# Partially parsed test_check_global_invariants_one_fails. Retrieved 1/8 statements.
# Partially parsed test_check_global_invariants_multiple_fail. Retrieved 1/10 statements.
# Partially parsed test_check_global_invariants_with_subject_data. Retrieved 1/7 statements.
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
    var_0 = 50

def test_case_0():
    var_0 = 25
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'VALUE_TOO_SMALL'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 6/16 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint. Retrieved 4/28 statements.


def test_case_0():
    var_0 = lambda x: x
    var_1 = False
    var_2 = lambda x: x
    var_3 = True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_make_pmap_field_type_creates_new_type. Retrieved 2/6 statements.
# Failed to parse test_make_pmap_field_type_caches_result.
# Failed to parse test_make_pmap_field_type_different_types_different_results.
# Partially parsed test_make_pmap_field_type_has_reduce_method. Retrieved 1/6 statements.
# Partially parsed test_make_pmap_field_type_name_format. Retrieved 1/4 statements.
# Failed to parse test_make_pmap_field_type_with_builtin_types.


def test_case_0():
    var_0 = '__key_type__'
    var_1 = '__value_type__'

def test_case_0():
    var_0 = '__reduce__'

def test_case_0():
    var_0 = 'To'
    var_1 = 'PMap'
    var_2 = 'PMap'



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_field_with_single_type.
# Failed to parse test_field_with_multiple_types_list.
# Failed to parse test_field_with_multiple_types_tuple.
# Failed to parse test_field_with_multiple_types_set.
# Partially parsed test_field_with_string_type. Retrieved 2/4 statements.
# Failed to parse test_field_with_invariant.
# Partially parsed test_field_with_initial_value. Retrieved 1/4 statements.
# Failed to parse test_field_with_initial_callable.
# Partially parsed test_field_with_mandatory_true. Retrieved 1/4 statements.
# Failed to parse test_field_with_factory.
# Failed to parse test_field_with_serializer.
# Partially parsed test_field_invalid_initial_type. Retrieved 1/4 statements.
# Partially parsed test_field_invalid_invariant. Retrieved 1/4 statements.
# Partially parsed test_field_invalid_factory. Retrieved 1/4 statements.
# Partially parsed test_field_invalid_serializer. Retrieved 1/4 statements.
# Partially parsed test_field_with_all_parameters. Retrieved 2/14 statements.
# Failed to parse test_field_empty_type.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'MyCustomType'
    var_1 = module_0.field(var_0)
    var_2 = 'MyCustomType'
    var_3 = bool('MyCustomType' in var_1.type)
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
    var_3 = 'Type parameter expected'

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

def test_case_0():
    var_0 = 5
    var_1 = True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_check_field_parameters_valid_type_class. Retrieved 1/10 statements.
# Partially parsed test_check_field_parameters_valid_type_string. Retrieved 1/10 statements.
# Partially parsed test_check_field_parameters_invalid_type_parameter. Retrieved 1/11 statements.
# Partially parsed test_check_field_parameters_invalid_initial_type. Retrieved 1/11 statements.
# Partially parsed test_check_field_parameters_valid_initial_value. Retrieved 1/10 statements.
# Partially parsed test_check_field_parameters_callable_initial. Retrieved 1/10 statements.
# Partially parsed test_check_field_parameters_invalid_invariant. Retrieved 1/11 statements.
# Partially parsed test_check_field_parameters_invalid_factory. Retrieved 1/11 statements.
# Partially parsed test_check_field_parameters_invalid_serializer. Retrieved 1/11 statements.
# Partially parsed test_check_field_parameters_empty_type. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 'PFIELD_NO_INITIAL'

def test_case_0():
    var_0 = 'PFIELD_NO_INITIAL'

def test_case_0():
    var_0 = 'PFIELD_NO_INITIAL'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Type parameter expected'

def test_case_0():
    var_0 = 'PFIELD_NO_INITIAL'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Initial has invalid type'

def test_case_0():
    var_0 = 'PFIELD_NO_INITIAL'

def test_case_0():
    var_0 = 'PFIELD_NO_INITIAL'

def test_case_0():
    var_0 = 'PFIELD_NO_INITIAL'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Invariant must be callable'

def test_case_0():
    var_0 = 'PFIELD_NO_INITIAL'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Factory must be callable'

def test_case_0():
    var_0 = 'PFIELD_NO_INITIAL'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Serializer must be callable'

def test_case_0():
    var_0 = 'PFIELD_NO_INITIAL'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 4/13 statements.
# Partially parsed test_restore_seq_field_pickle_with_empty_data. Retrieved 1/10 statements.
# Partially parsed test_restore_seq_field_pickle_retrieves_correct_type. Retrieved 4/14 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_pfield_constructor. Retrieved 4/11 statements.
# Partially parsed test_pfield_constructor_with_various_types. Retrieved 8/11 statements.


def test_case_0():
    var_0 = 'int'
    var_1 = {var_0}
    var_2 = 10
    var_3 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = set()
    var_1 = None
    var_2 = False
    var_3 = module_0._PField(var_0, var_1, var_1, var_2, var_1, var_1)
    var_4 = set()
    var_5 = var_3.type
    var_6 = bool(var_3.type == var_4)
    assert var_6 is True
    var_7 = var_3.invariant
    assert var_7 is None
    var_8 = var_3.initial
    assert var_8 is None
    var_9 = var_3.mandatory
    assert var_9 is False
    var_10 = var_3._factory
    assert var_10 is None
    var_11 = var_3.serializer
    assert var_11 is None

def test_case_0():
    var_0 = 'str'
    var_1 = 'int'
    var_2 = 'float'
    var_3 = {var_0, var_1, var_2}
    var_4 = 0
    var_5 = lambda x: len(x) > var_4
    var_6 = 'default'
    var_7 = True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_sequence_field_with_pvector_non_optional. Retrieved 4/10 statements.
# Partially parsed test_sequence_field_with_pvector_optional. Retrieved 3/9 statements.
# Partially parsed test_sequence_field_with_pset_non_optional. Retrieved 4/10 statements.
# Partially parsed test_sequence_field_with_invariant. Retrieved 3/11 statements.
# Partially parsed test_sequence_field_with_item_invariant. Retrieved 3/11 statements.
# Partially parsed test_sequence_field_optional_factory_with_none. Retrieved 3/8 statements.
# Partially parsed test_sequence_field_non_optional_factory_with_data. Retrieved 7/13 statements.


def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = 3

def test_case_0():
    var_0 = True
    var_1 = 'a'
    var_2 = 'b'

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = 3

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = None

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = 4
    var_5 = 5
    var_6 = [var_3, var_4, var_5]



# Parsed testcases at query #10
#--------------------------

# Failed to parse test_check_field_parameters_predicate_line_3_false.




# Parsed testcases at query #11
#--------------------------

# Partially parsed test_check_type_valid_single_type. Retrieved 2/12 statements.
# Partially parsed test_check_type_valid_multiple_types. Retrieved 2/12 statements.
# Partially parsed test_check_type_valid_with_string_type. Retrieved 4/13 statements.
# Partially parsed test_check_type_invalid_type_raises_error. Retrieved 2/13 statements.
# Partially parsed test_check_type_no_type_constraint. Retrieved 3/12 statements.
# Partially parsed test_check_type_empty_type_tuple. Retrieved 3/13 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 42

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'hello'

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = (var_0,)
    var_2 = 'test_field'
    var_3 = 42

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'not_an_int'
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Invalid type for field'

def test_case_0():
    var_0 = None
    var_1 = 'test_field'
    var_2 = 'any_value'

def test_case_0():
    var_0 = ()
    var_1 = 'test_field'
    var_2 = 42
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_set_fields. Retrieved 19/50 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'fields'
    var_3 = module_0.set_fields(var_0, var_1, var_2)
    var_4 = bool(var_2 in var_0)
    assert var_4 is True
    var_5 = var_0[var_2]
    var_6 = bool(var_0[var_2] == {})
    assert var_6 is True
    var_7 = 'field1'
    var_8 = 'field2'
    var_9 = 'other'
    var_10 = 'value'
    var_11 = []
    var_12 = 'fields'
    var_13 = 'field1'
    var_14 = 'field2'
    var_15 = 'base_field'
    var_16 = {}
    var_17 = 'fields'
    var_18 = bool(var_17 in var_16)
    assert var_18 is True
    var_19 = var_16[var_17]['base_field']
    var_20 = 'dct_field'
    var_21 = 'fields'
    var_22 = 'dct_field'
    var_23 = 'field_a'
    var_24 = 'field_b'
    var_25 = {}
    var_26 = 'fields'
    var_27 = bool(var_26 in var_25)
    assert var_27 is True
    var_28 = var_25[var_26]['field_a']
    var_29 = var_25[var_26]['field_b']



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_check_global_invariants_raises_exception_when_invariants_fail. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'test_subject'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_make_pmap_field_type_creates_new_pmap_class. Retrieved 2/6 statements.
# Failed to parse test_make_pmap_field_type_sets_correct_name.
# Failed to parse test_make_pmap_field_type_caches_result.
# Failed to parse test_make_pmap_field_type_different_types_different_classes.
# Partially parsed test_make_pmap_field_type_has_reduce_method. Retrieved 1/6 statements.
# Failed to parse test_make_pmap_field_type_with_tuple_types.


def test_case_0():
    var_0 = '__key_type__'
    var_1 = '__value_type__'

def test_case_0():
    var_0 = '__reduce__'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 1/7 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 2/5 statements.
# Partially parsed test_serialize_with_checked_type_and_custom_serializer. Retrieved 1/8 statements.
# Partially parsed test_serialize_with_non_checked_type. Retrieved 2/5 statements.
# Partially parsed test_serialize_with_different_formats. Retrieved 4/9 statements.


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
    var_1 = 123

def test_case_0():
    var_0 = 'json'
    var_1 = 'data'
    var_2 = 'xml'
    var_3 = 'yaml'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_pfield_constructor. Retrieved 2/9 statements.
# Partially parsed test_pfield_constructor_with_none_values. Retrieved 2/3 statements.
# Partially parsed test_pfield_constructor_with_different_types. Retrieved 8/9 statements.


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
    var_7 = lambda x: x



# Parsed testcases at query #17
#--------------------------

# Failed to parse test_types_to_names_with_builtin_types.
# Failed to parse test_types_to_names_with_single_type.
# Failed to parse test_types_to_names_with_bool_type.
# Failed to parse test_types_to_names_with_list_type.
# Failed to parse test_types_to_names_with_multiple_types.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ()
    var_1 = module_0._types_to_names(var_0)
    assert var_1 == ''



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_set_fields_predicate_isinstance_pfield. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'not_a_pfield'
    var_3 = 'fields'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_check_global_invariants_no_errors. Retrieved 2/9 statements.


def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = False
    assert var_2 is True



# Parsed testcases at query #20
#--------------------------

# Failed to parse test_sequence_field_invariant_parameter_is_pfield_no_invariant.




# Parsed testcases at query #21
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 6/13 statements.
# Partially parsed test_restore_pmap_field_pickle_empty. Retrieved 3/11 statements.
# Partially parsed test_restore_pmap_field_pickle_different_types. Retrieved 6/11 statements.


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
    var_2 = '__getitem__'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'one'
    var_3 = 'two'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_sequence_field_creates_field_with_checked_class. Retrieved 1/10 statements.
# Partially parsed test_sequence_field_with_optional_true. Retrieved 1/9 statements.
# Partially parsed test_sequence_field_with_optional_false. Retrieved 1/9 statements.
# Partially parsed test_sequence_field_factory_with_none_optional. Retrieved 2/9 statements.
# Partially parsed test_sequence_field_with_initial_value. Retrieved 4/10 statements.
# Partially parsed test_sequence_field_with_invariant. Retrieved 4/4 statements.
# Partially parsed test_sequence_field_with_item_invariant. Retrieved 1/9 statements.
# Partially parsed test_sequence_field_mandatory_is_true. Retrieved 1/7 statements.


def test_case_0():
    var_0 = False

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = False

def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = False

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = False

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = False

def test_case_0():
    var_0 = False

def test_case_0():
    var_0 = False



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_pmap_field_basic. Retrieved 1/4 statements.
# Partially parsed test_pmap_field_with_optional_true. Retrieved 2/5 statements.
# Partially parsed test_pmap_field_with_optional_false. Retrieved 2/5 statements.
# Partially parsed test_pmap_field_with_invariant. Retrieved 1/6 statements.
# Failed to parse test_pmap_field_factory_callable.
# Partially parsed test_pmap_field_factory_with_optional_true. Retrieved 2/7 statements.
# Partially parsed test_pmap_field_factory_with_optional_creates_map. Retrieved 5/9 statements.


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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = True
    var_1 = 'a'
    var_2 = {var_1: var_0}
    var_3 = {var_1: var_0}
    var_4 = module_0.pmap(var_3)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_pfield_factory_assignment. Retrieved 3/7 statements.


def test_case_0():
    var_0 = None
    var_1 = lambda : var_0
    var_2 = False



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_pmap_field_type_predicate_optional_true. Retrieved 2/6 statements.
# Partially parsed test_pmap_field_type_predicate_optional_false. Retrieved 2/5 statements.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = [var_1]

def test_case_0():
    var_0 = False
    var_1 = None
    var_2 = [var_1]



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_pmap_field_optional_false_predicate. Retrieved 1/6 statements.


def test_case_0():
    var_0 = False



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_check_global_invariants_raises_exception_when_invariants_fail. Retrieved 3/20 statements.


def test_case_0():
    var_0 = 'test_subject'
    var_1 = ()
    var_2 = 'Global invariant failed'
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #28
#--------------------------

# Failed to parse test_invariant_wrapping_when_callable_and_not_no_invariant.




# Parsed testcases at query #29
#--------------------------

# Partially parsed test_invariant_wrapping_when_callable_and_not_no_invariant. Retrieved 1/12 statements.


def test_case_0():
    var_0 = 42



# Parsed testcases at query #30
#--------------------------

# Failed to parse test_check_field_parameters_valid_type_parameter.
# Failed to parse test_check_field_parameters_invalid_type_parameter.
# Failed to parse test_check_field_parameters_invalid_initial_type.
# Failed to parse test_check_field_parameters_callable_initial.
# Failed to parse test_check_field_parameters_no_initial.
# Failed to parse test_check_field_parameters_non_callable_invariant.
# Failed to parse test_check_field_parameters_non_callable_factory.
# Failed to parse test_check_field_parameters_non_callable_serializer.
# Failed to parse test_check_field_parameters_string_type.
# Failed to parse test_check_field_parameters_empty_type.




# Parsed testcases at query #31
#--------------------------

# Partially parsed test_pmap_field_optional_false_predicate. Retrieved 2/5 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.pmap()



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_make_seq_field_type_creates_new_type. Retrieved 4/12 statements.
# Partially parsed test_make_seq_field_type_caches_result. Retrieved 1/7 statements.
# Partially parsed test_make_seq_field_type_returns_cached_type. Retrieved 1/7 statements.
# Partially parsed test_make_seq_field_type_has_reduce_method. Retrieved 2/13 statements.
# Partially parsed test_make_seq_field_type_different_item_types. Retrieved 1/6 statements.
# Failed to parse test_make_seq_field_type_with_invariant.


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
    var_1 = 0

def test_case_0():
    var_0 = None



# Parsed testcases at query #33
#--------------------------

# Failed to parse test_field_with_single_type.
# Failed to parse test_field_with_multiple_types_as_list.
# Failed to parse test_field_with_multiple_types_as_tuple.
# Failed to parse test_field_with_multiple_types_as_set.
# Partially parsed test_field_with_initial_value. Retrieved 1/2 statements.
# Partially parsed test_field_with_mandatory_true. Retrieved 1/2 statements.
# Partially parsed test_field_with_callable_factory. Retrieved 2/3 statements.
# Partially parsed test_field_with_callable_serializer. Retrieved 1/2 statements.
# Partially parsed test_field_with_callable_invariant. Retrieved 4/7 statements.
# Partially parsed test_field_with_invalid_type_parameter. Retrieved 1/3 statements.
# Partially parsed test_field_with_initial_matching_type. Retrieved 1/2 statements.
# Partially parsed test_field_with_initial_not_matching_type. Retrieved 1/3 statements.
# Partially parsed test_field_with_initial_matching_multiple_types. Retrieved 1/3 statements.
# Partially parsed test_field_with_non_callable_factory. Retrieved 1/3 statements.
# Partially parsed test_field_with_non_callable_serializer. Retrieved 1/3 statements.
# Partially parsed test_field_all_parameters. Retrieved 8/10 statements.
# Failed to parse test_field_invariant_wrapping.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'CustomType'
    var_1 = module_0.field(var_0)
    var_2 = var_1.type
    var_3 = bool(var_1.type == {'CustomType'})
    assert var_3 is True

def test_case_0():
    var_0 = 42

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = 'default'
    var_1 = lambda : var_0

def test_case_0():
    var_0 = lambda x: str(x)

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2

def test_case_0():
    var_0 = 123
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Invariant must be callable'

def test_case_0():
    var_0 = 10

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Initial has invalid type'

def test_case_0():
    var_0 = 'hello'

def test_case_0():
    var_0 = 'not_callable'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Factory must be callable'

def test_case_0():
    var_0 = 42
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Serializer must be callable'

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = 0
    var_5 = lambda : var_4
    var_6 = lambda x: str(x)
    var_7 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = set()
    var_2 = var_0.type
    var_3 = bool(var_0.type == var_1)
    assert var_3 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_pfield_constructor. Retrieved 4/10 statements.
# Partially parsed test_pfield_constructor_with_none_values. Retrieved 2/4 statements.
# Partially parsed test_pfield_constructor_with_empty_type. Retrieved 5/6 statements.
# Partially parsed test_pfield_constructor_mandatory_false. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 42
    var_1 = 10
    var_2 = lambda : var_1
    var_3 = True

def test_case_0():
    var_0 = None
    var_1 = False

def test_case_0():
    var_0 = ()
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = 0
    var_4 = lambda : var_3

def test_case_0():
    var_0 = None
    var_1 = 'default'
    var_2 = False



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 4/16 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_check_field_parameters_predicate_line_3_false. Retrieved 1/20 statements.


def test_case_0():
    var_0 = 'int'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_pmap_field_optional_false_predicate. Retrieved 2/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.field()
    var_2 = [var_1]



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_check_global_invariants_all_pass. Retrieved 1/7 statements.
# Partially parsed test_check_global_invariants_single_failure. Retrieved 1/8 statements.
# Partially parsed test_check_global_invariants_multiple_failures. Retrieved 1/10 statements.
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

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = []
    var_1 = 'test_subject'
    var_2 = module_0.check_global_invariants(var_1, var_0)

def test_case_0():
    var_0 = 'test_subject'
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_pfield_factory_assignment. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = lambda : var_0
    var_2 = lambda x: str(x)
    var_3 = None
    var_4 = False



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_serialize_checked_type_with_no_serializer. Retrieved 1/11 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'json'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_pmap_field_returns_field_with_correct_type. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'type'
    var_1 = 'factory'
    var_2 = 'mandatory'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_make_seq_field_type_creates_new_type. Retrieved 4/14 statements.
# Partially parsed test_make_seq_field_type_caches_type. Retrieved 1/7 statements.
# Failed to parse test_make_seq_field_type_with_invariant.
# Partially parsed test_make_seq_field_type_has_reduce_method. Retrieved 2/10 statements.
# Partially parsed test_make_seq_field_type_different_item_types_create_different_classes. Retrieved 1/6 statements.


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



# Parsed testcases at query #43
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



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_make_pmap_field_type_creates_pmap_subclass. Retrieved 2/6 statements.
# Failed to parse test_make_pmap_field_type_caches_result.
# Failed to parse test_make_pmap_field_type_different_types_creates_different_classes.
# Failed to parse test_make_pmap_field_type_generates_correct_class_name.
# Partially parsed test_make_pmap_field_type_has_reduce_method. Retrieved 1/6 statements.
# Failed to parse test_make_pmap_field_type_with_complex_types.


def test_case_0():
    var_0 = '__key_type__'
    var_1 = '__value_type__'

def test_case_0():
    var_0 = '__reduce__'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_check_global_invariants_no_errors. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'test_subject'
    var_1 = True
    var_2 = False
    assert var_2 is True



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_check_global_invariants_all_pass. Retrieved 1/7 statements.
# Partially parsed test_check_global_invariants_single_failure. Retrieved 1/6 statements.
# Partially parsed test_check_global_invariants_multiple_failures. Retrieved 1/10 statements.
# Partially parsed test_check_global_invariants_exception_details. Retrieved 1/6 statements.


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
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #47
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



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 4/11 statements.
# Partially parsed test_restore_seq_field_pickle_empty. Retrieved 1/8 statements.
# Partially parsed test_restore_seq_field_pickle_with_strings. Retrieved 4/11 statements.


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



# Parsed testcases at query #49
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_new_type.
# Failed to parse test_make_pmap_field_type_caches_result.
# Failed to parse test_make_pmap_field_type_different_types_create_different_classes.
# Partially parsed test_make_pmap_field_type_has_reduce_method. Retrieved 1/6 statements.
# Failed to parse test_make_pmap_field_type_inherits_from_checked_pmap.
# Failed to parse test_make_pmap_field_type_with_multiple_types.


def test_case_0():
    var_0 = '__reduce__'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_set_fields_predicate_isinstance_pfield. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'not a pfield'
    var_3 = 'fields'



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 6/18 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)



# Parsed testcases at query #52
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_new_pmap_type.
# Failed to parse test_make_pmap_field_type_caches_result.
# Failed to parse test_make_pmap_field_type_different_types_create_different_classes.
# Partially parsed test_make_pmap_field_type_has_reduce_method. Retrieved 1/4 statements.
# Failed to parse test_make_pmap_field_type_with_multiple_key_value_types.


def test_case_0():
    var_0 = '__reduce__'



# Parsed testcases at query #53
#--------------------------

# Failed to parse test_check_field_parameters_predicate_line_3_false.




# Parsed testcases at query #54
#--------------------------

# Partially parsed test_set_fields. Retrieved 29/59 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'fields'
    var_3 = module_0.set_fields(var_0, var_1, var_2)
    var_4 = bool(var_0 == {'fields': {}})
    assert var_4 is True
    var_5 = 1
    var_6 = 2
    var_7 = 'attr1'
    var_8 = 'attr2'
    var_9 = 'other'
    var_10 = 'value'
    var_11 = []
    var_12 = 'fields'
    var_13 = 'fields'
    var_14 = 'attr1'
    var_15 = 'attr2'
    var_16 = 'base_field'
    var_17 = 0
    var_18 = 'fields'
    var_19 = 'base_attr'
    var_20 = 'child_field'
    var_21 = 3
    var_22 = 'fields'
    var_23 = 'fields'
    var_24 = 'base_attr'
    var_25 = 'child_field'
    var_26 = 'child_field'
    var_27 = 'field1'
    var_28 = 'field2'
    var_29 = 'field3'
    var_30 = 'fields'
    var_31 = 'field1'
    var_32 = 'field2'
    var_33 = 'field3'
    var_34 = 'field3'
    var_35 = 'string'
    var_36 = 123
    var_37 = {var_7: var_35, var_8: var_36}
    var_38 = []
    var_39 = 'fields'
    var_40 = module_0.set_fields(var_37, var_38, var_39)
    var_41 = bool(var_37 == {'fields': {}, 'attr1': 'string', 'attr2': 123})
    assert var_41 is True



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 1/11 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'json'



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 6/15 statements.
# Partially parsed test_restore_seq_field_pickle_with_empty_data. Retrieved 1/10 statements.
# Partially parsed test_restore_seq_field_pickle_with_string_items. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_check_global_invariants_raises_exception_when_invariants_fail. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 'test_subject'



# Parsed testcases at query #58
#--------------------------




def test_case_0():
    var_0 = 'Test that the predicate at line 2 (optional=False) evaluates to False'
    var_1 = False
    var_2 = bool(not var_1)
    assert var_2 is True



# Parsed testcases at query #59
#--------------------------

# Failed to parse test_pmap_field_basic.
# Partially parsed test_pmap_field_with_optional_false. Retrieved 2/5 statements.
# Partially parsed test_pmap_field_with_optional_true. Retrieved 2/5 statements.
# Partially parsed test_pmap_field_with_optional_true_non_none. Retrieved 3/6 statements.
# Partially parsed test_pmap_field_with_invariant. Retrieved 3/4 statements.
# Failed to parse test_pmap_field_initial_value.
# Failed to parse test_pmap_field_type_attribute.


def test_case_0():
    var_0 = False
    var_1 = {}

def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = True
    var_1 = 'key'
    var_2 = {var_1: var_0}

def test_case_0():
    var_0 = True
    var_1 = 'valid'
    var_2 = (var_0, var_1)

def test_case_0():
    var_0 = True
    var_1 = 'valid'
    var_2 = (var_0, var_1)



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_make_seq_field_type. Retrieved 2/14 statements.
# Failed to parse test_make_seq_field_type_with_invariant.
# Partially parsed test_make_seq_field_type_reduce. Retrieved 6/17 statements.
# Partially parsed test_make_seq_field_type_different_item_types. Retrieved 1/6 statements.


def test_case_0():
    var_0 = None
    var_1 = '__name__'

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 0

def test_case_0():
    var_0 = None



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_pmap_field_predicate_line_2. Retrieved 1/5 statements.


def test_case_0():
    var_0 = False
    var_1 = 'Create a checked ``PMap`` field'



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_pmap_field_returns_field_with_correct_type. Retrieved 1/7 statements.


def test_case_0():
    var_0 = False



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_make_seq_field_type_creates_new_type. Retrieved 4/14 statements.
# Partially parsed test_make_seq_field_type_caches_result. Retrieved 1/7 statements.
# Partially parsed test_make_seq_field_type_different_types_different_results. Retrieved 1/6 statements.
# Failed to parse test_make_seq_field_type_with_invariant.
# Partially parsed test_make_seq_field_type_has_reduce_method. Retrieved 2/9 statements.
# Partially parsed test_make_seq_field_type_name_generation. Retrieved 2/12 statements.


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

def test_case_0():
    var_0 = None
    var_1 = 'int'



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_check_global_invariants_with_all_passing_invariants. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'test_subject'
    var_1 = True
    var_2 = False
    assert var_2 is True



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_set_fields_with_pfield. Retrieved 8/16 statements.
# Partially parsed test_set_fields_with_base_fields. Retrieved 10/21 statements.
# Partially parsed test_set_fields_multiple_bases. Retrieved 12/23 statements.


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

def test_case_0():
    var_0 = 'value1'
    var_1 = 'value2'
    var_2 = 'field1'
    var_3 = 'field2'
    var_4 = 'other'
    var_5 = 'data'
    var_6 = []
    var_7 = 'fields'
    var_8 = 'fields'
    var_9 = 'field1'
    var_10 = 'field2'
    var_11 = 'field1'
    var_12 = 'field2'
    var_13 = 'other'

def test_case_0():
    var_0 = 'base1'
    var_1 = 'base2'
    var_2 = 'Base1'
    var_3 = ()
    var_4 = {}
    var_5 = [var_2, var_3, var_4]
    var_6 = 'base_field1'
    var_7 = 'base_field2'
    var_8 = 'field1'
    var_9 = 'value1'
    var_10 = 'fields'
    var_11 = 'fields'
    var_12 = 'base_field1'
    var_13 = 'base_field2'
    var_14 = 'field1'

def test_case_0():
    var_0 = 'base1'
    var_1 = 'base2'
    var_2 = 'Base1'
    var_3 = ()
    var_4 = {}
    var_5 = [var_2, var_3, var_4]
    var_6 = 'base_field1'
    var_7 = 'Base2'
    var_8 = ()
    var_9 = {}
    var_10 = [var_7, var_8, var_9]
    var_11 = 'base_field2'
    var_12 = {}
    var_13 = 'fields'
    var_14 = 'fields'
    var_15 = bool('fields' in var_12)
    assert var_15 is True
    var_16 = 'base_field1'
    var_17 = bool('base_field1' in var_12['fields'])
    assert var_17 is True
    var_18 = 'base_field2'
    var_19 = bool('base_field2' in var_12['fields'])
    assert var_19 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = 'fields'
    var_7 = module_0.set_fields(var_4, var_5, var_6)
    var_8 = var_4['fields']
    var_9 = bool(var_4['fields'] == {})
    assert var_9 is True
    var_10 = var_4['key1']
    assert var_10 == 'value1'
    var_11 = var_4['key2']
    assert var_11 == 'value2'



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 9/19 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_0: var_2, var_1: var_3}
    var_7 = module_0.pmap()
    var_8 = [var_7]
    var_9 = module_0.pmap(var_6)



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 1/12 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'json'



# Parsed testcases at query #68
#--------------------------

# Failed to parse test_check_field_parameters_valid_field.
# Failed to parse test_check_field_parameters_invalid_type_parameter.
# Failed to parse test_check_field_parameters_invalid_initial_type.
# Failed to parse test_check_field_parameters_non_callable_invariant.
# Failed to parse test_check_field_parameters_non_callable_factory.
# Failed to parse test_check_field_parameters_non_callable_serializer.
# Failed to parse test_check_field_parameters_string_type.
# Failed to parse test_check_field_parameters_callable_initial.
# Failed to parse test_check_field_parameters_empty_type.




# Parsed testcases at query #69
#--------------------------

# Partially parsed test_check_global_invariants_raises_exception_when_invariants_fail. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 'test_subject'



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_pfield_factory_assignment. Retrieved 2/8 statements.


def test_case_0():
    var_0 = None
    var_1 = False



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_pmap_field_optional_false_predicate. Retrieved 1/4 statements.


def test_case_0():
    var_0 = False



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_make_seq_field_type_creates_new_type. Retrieved 1/6 statements.
# Partially parsed test_make_seq_field_type_caches_type. Retrieved 1/6 statements.
# Partially parsed test_make_seq_field_type_different_item_types. Retrieved 1/5 statements.
# Failed to parse test_make_seq_field_type_with_invariant.
# Partially parsed test_make_seq_field_type_has_reduce_method. Retrieved 2/8 statements.


def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = None
    var_1 = '__reduce__'



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 5/17 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]



# Parsed testcases at query #74
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_new_pmap_field_type.
# Failed to parse test_make_pmap_field_type_caches_result.
# Failed to parse test_make_pmap_field_type_different_types_creates_different_classes.
# Partially parsed test_make_pmap_field_type_has_reduce_method. Retrieved 1/7 statements.
# Failed to parse test_make_pmap_field_type_reduce_returns_correct_tuple.


def test_case_0():
    var_0 = '__reduce__'



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_pmap_field_optional_factory_returns_none. Retrieved 2/7 statements.


def test_case_0():
    var_0 = True
    var_1 = None



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_check_global_invariants_no_errors. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'test_subject'



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_make_seq_field_type_creates_new_type. Retrieved 4/11 statements.
# Partially parsed test_make_seq_field_type_caches_result. Retrieved 1/7 statements.
# Partially parsed test_make_seq_field_type_different_types_different_results. Retrieved 1/6 statements.
# Failed to parse test_make_seq_field_type_with_invariant.
# Partially parsed test_make_seq_field_type_has_reduce_method. Retrieved 1/6 statements.


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



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_set_fields_pfield_isinstance. Retrieved 5/16 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'not_pfield'
    var_3 = []
    var_4 = 'fields'
    var_5 = 'field1'
    var_6 = 'field1'
    var_7 = 'field2'



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_pfield_factory_assignment. Retrieved 2/8 statements.


def test_case_0():
    var_0 = None
    var_1 = False



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_check_field_parameters_predicate_line_3_false. Retrieved 1/21 statements.


def test_case_0():
    var_0 = 'str'



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_pmap_field_basic. Retrieved 1/4 statements.
# Partially parsed test_pmap_field_with_optional_false. Retrieved 2/5 statements.
# Partially parsed test_pmap_field_with_optional_true. Retrieved 2/5 statements.
# Failed to parse test_pmap_field_with_invariant.
# Partially parsed test_pmap_field_factory_none_optional. Retrieved 2/5 statements.
# Partially parsed test_pmap_field_factory_with_dict_optional. Retrieved 5/8 statements.
# Partially parsed test_pmap_field_factory_without_optional. Retrieved 6/9 statements.


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



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 1/11 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'json'



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_make_pmap_field_type. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'PMap'
    var_1 = '__reduce__'



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 8/21 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {}
    var_7 = module_0.pmap(var_6)
    var_8 = [var_7]



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_check_global_invariants_all_pass. Retrieved 1/7 statements.
# Partially parsed test_check_global_invariants_single_failure. Retrieved 1/8 statements.
# Partially parsed test_check_global_invariants_multiple_failures. Retrieved 1/10 statements.
# Partially parsed test_check_global_invariants_with_subject_data. Retrieved 1/7 statements.
# Partially parsed test_check_global_invariants_subject_validation_fails. Retrieved 1/8 statements.


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
    var_0 = 'valid'

def test_case_0():
    var_0 = 'invalid'
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #87
#--------------------------

# Partially parsed test_pmap_field_creates_field_with_correct_type_when_optional_false. Retrieved 4/10 statements.


def test_case_0():
    var_0 = False
    var_1 = 'type'
    var_2 = 'factory'
    var_3 = 'mandatory'



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_set_fields. Retrieved 24/57 statements.


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
    var_8 = 'field1'
    var_9 = 'field2'
    var_10 = 'other'
    var_11 = 'value'
    var_12 = []
    var_13 = 'fields'
    var_14 = module_0.set_fields(var_0, var_12, var_13)
    var_15 = var_0['fields']
    var_16 = 'field1'
    var_17 = bool('field1' not in var_0)
    assert var_17 is True
    var_18 = 'field2'
    var_19 = bool('field2' not in var_0)
    assert var_19 is True
    var_20 = var_0['other']
    assert var_20 == 'value'
    var_21 = 'fields'
    var_22 = 'base_field'
    var_23 = {}
    var_24 = 'fields'
    var_25 = module_0.set_fields(var_23, var_12, var_24)
    var_26 = 'fields'
    var_27 = bool('fields' in var_23)
    assert var_27 is True
    var_28 = 'base_field'
    var_29 = bool('base_field' in var_23['fields'])
    assert var_29 is True
    var_30 = 'dct_field'
    var_31 = 'fields'
    var_32 = module_0.set_fields(var_23, var_12, var_31)
    var_33 = 'fields'
    var_34 = bool('fields' in var_23)
    assert var_34 is True
    var_35 = 'base_field'
    var_36 = bool('base_field' in var_23['fields'])
    assert var_36 is True
    var_37 = 'dct_field'
    var_38 = bool('dct_field' in var_23['fields'])
    assert var_38 is True
    var_39 = 'field_a'
    var_40 = 'field_b'
    var_41 = {}
    var_42 = 'fields'
    var_43 = module_0.set_fields(var_41, var_12, var_42)
    var_44 = 'fields'
    var_45 = bool('fields' in var_41)
    assert var_45 is True
    var_46 = 'field_a'
    var_47 = bool('field_a' in var_41['fields'])
    assert var_47 is True
    var_48 = 'field_b'
    var_49 = bool('field_b' in var_41['fields'])
    assert var_49 is True



# Parsed testcases at query #89
#--------------------------

# Failed to parse test_check_field_parameters_valid_field.
# Failed to parse test_check_field_parameters_invalid_type_parameter.
# Failed to parse test_check_field_parameters_type_parameter_as_string.
# Failed to parse test_check_field_parameters_invalid_initial_type.
# Failed to parse test_check_field_parameters_callable_initial.
# Failed to parse test_check_field_parameters_non_callable_invariant.
# Failed to parse test_check_field_parameters_non_callable_factory.
# Failed to parse test_check_field_parameters_non_callable_serializer.
# Failed to parse test_check_field_parameters_empty_type_list.




# Parsed testcases at query #90
#--------------------------

# Partially parsed test_pmap_field_optional_type_predicate. Retrieved 3/19 statements.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = [var_1]
    var_3 = False
    var_4 = [var_1]



# Parsed testcases at query #91
#--------------------------

# Partially parsed test_check_global_invariants_with_different_subjects. Retrieved 9/12 statements.
# Partially parsed test_check_global_invariants_invariant_receives_correct_subject. Retrieved 2/7 statements.


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
    var_9 = 'test_subject'
    var_10 = module_0.check_global_invariants(var_9, var_8)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = False
    var_5 = 'error_1'
    var_6 = (var_4, var_5)
    var_7 = lambda x: var_6
    var_8 = (var_0, var_1)
    var_9 = lambda x: var_8
    var_10 = [var_3, var_7, var_9]
    var_11 = 'test_subject'
    var_12 = module_0.check_global_invariants(var_11, var_10)
    var_13 = bool(False)
    assert var_13 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = False
    var_1 = 'error_1'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = 'error_2'
    var_5 = (var_0, var_4)
    var_6 = lambda x: var_5
    var_7 = True
    var_8 = None
    var_9 = (var_7, var_8)
    var_10 = lambda x: var_9
    var_11 = 'error_3'
    var_12 = (var_0, var_11)
    var_13 = lambda x: var_12
    var_14 = [var_3, var_6, var_10, var_13]
    var_15 = 'test_subject'
    var_16 = module_0.check_global_invariants(var_15, var_14)
    var_17 = bool(False)
    assert var_17 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = []
    var_1 = 'test_subject'
    var_2 = module_0.check_global_invariants(var_1, var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = False
    var_1 = 'test_error'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = [var_3]
    var_5 = 'test_subject'
    var_6 = module_0.check_global_invariants(var_5, var_4)
    var_7 = bool(False)
    assert var_7 is True

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = None
    var_5 = (var_3, var_4)
    var_6 = False
    var_7 = 'not_dict'
    var_8 = (var_6, var_7)

def test_case_0():
    var_0 = 'test_value'
    var_1 = []
    var_2 = var_1[0]
    assert var_2 == 'test_value'



# Parsed testcases at query #92
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #93
#--------------------------

# Partially parsed test_pfield_init_factory_assignment. Retrieved 2/8 statements.


def test_case_0():
    var_0 = None
    var_1 = False



# Parsed testcases at query #94
#--------------------------

# Partially parsed test_serialize_checked_type_with_no_serializer. Retrieved 1/8 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'json'



# Parsed testcases at query #95
#--------------------------

# Partially parsed test_check_global_invariants_raises_exception_when_error_codes_exist. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 'test_subject'
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #96
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 6/19 statements.
# Partially parsed test_restore_pmap_field_pickle_empty. Retrieved 1/13 statements.
# Partially parsed test_restore_pmap_field_pickle_with_types. Retrieved 7/19 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = [var_5]

def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 1.5
    var_4 = 2.5
    var_5 = 3.5
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}



# Parsed testcases at query #97
#--------------------------

# Partially parsed test_make_seq_field_type_creates_new_type. Retrieved 4/12 statements.
# Partially parsed test_make_seq_field_type_caches_result. Retrieved 1/7 statements.
# Partially parsed test_make_seq_field_type_with_different_item_types. Retrieved 1/6 statements.
# Failed to parse test_make_seq_field_type_with_invariant.
# Partially parsed test_make_seq_field_type_has_reduce_method. Retrieved 1/7 statements.


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



# Parsed testcases at query #98
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_new_pmap_subclass.
# Failed to parse test_make_pmap_field_type_caches_result.
# Failed to parse test_make_pmap_field_type_different_types_create_different_classes.
# Partially parsed test_make_pmap_field_type_has_reduce_method. Retrieved 1/6 statements.
# Failed to parse test_make_pmap_field_type_with_multiple_key_value_types.


def test_case_0():
    var_0 = '__reduce__'



# Parsed testcases at query #99
#--------------------------

# Partially parsed test_pmap_field_optional_false_predicate. Retrieved 5/11 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = False
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = module_0.pmap(var_3)



# Parsed testcases at query #100
#--------------------------

# Partially parsed test_make_seq_field_type. Retrieved 7/21 statements.


def test_case_0():
    var_0 = None
    var_1 = '__name__'
    var_2 = 'Vector'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = 0



# Parsed testcases at query #101
#--------------------------

# Partially parsed test_check_global_invariants_raises_exception_when_invariants_fail. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 'test_subject'
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #102
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 4/13 statements.
# Partially parsed test_restore_seq_field_pickle_with_strings. Retrieved 4/13 statements.
# Partially parsed test_restore_seq_field_pickle_empty_list. Retrieved 1/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = []



# Parsed testcases at query #103
#--------------------------

# Partially parsed test_set_fields. Retrieved 28/56 statements.


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
    var_8 = 'field1'
    var_9 = 'field2'
    var_10 = 'value1'
    var_11 = 'regular_value'
    var_12 = []
    var_13 = 'fields'
    var_14 = 'field1'
    var_15 = 'field1'
    var_16 = 'field2'
    var_17 = 'base_field1'
    var_18 = 'base_value1'
    var_19 = 'base_field2'
    var_20 = 'base_value2'
    var_21 = 'field3'
    var_22 = 'value3'
    var_23 = 'fields'
    var_24 = 'base_field1'
    var_25 = 'base_field2'
    var_26 = 'field3'
    var_27 = 'field3'
    var_28 = 'pf1'
    var_29 = 'pf2'
    var_30 = 'pf3'
    var_31 = 'regular'
    var_32 = 'pv1'
    var_33 = 'pv2'
    var_34 = 'pv3'
    var_35 = 'value'
    var_36 = []
    var_37 = 'pf1'
    var_38 = 'pf2'
    var_39 = 'pf3'
    var_40 = 'pf1'
    var_41 = 'pf2'
    var_42 = 'pf3'
    var_43 = 'regular'
    var_44 = {}
    var_45 = []
    var_46 = 'my_fields'
    var_47 = module_0.set_fields(var_44, var_45, var_46)
    var_48 = 'my_fields'
    var_49 = bool('my_fields' in var_44)
    assert var_49 is True
    var_50 = var_44['my_fields']
    var_51 = bool(var_44['my_fields'] == {})
    assert var_51 is True



# Parsed testcases at query #104
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 1/11 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'json'



# Parsed testcases at query #105
#--------------------------

# Partially parsed test_make_pmap_field_type. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'PMap'
    var_1 = '__reduce__'
    var_2 = 'PMap'



# Parsed testcases at query #106
#--------------------------

# Failed to parse test_check_field_parameters_predicate_line_3_false.




####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 2/7 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 2/5 statements.
# Partially parsed test_serialize_with_checked_type_and_custom_serializer. Retrieved 2/11 statements.
# Partially parsed test_serialize_with_regular_value_and_serializer. Retrieved 2/5 statements.
# Partially parsed test_serialize_with_checked_type_no_serializer_constant. Retrieved 1/7 statements.


def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = 'json'

def test_case_0():
    var_0 = 'xml'
    var_1 = 'test_value'

def test_case_0():
    var_0 = []
    var_1 = 'json'
    var_2 = 'custom_json_'

def test_case_0():
    var_0 = 'csv'
    var_1 = 42

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'binary'



# Parsed testcases at query #2
#--------------------------

# Failed to parse test_pmap_field_basic.
# Partially parsed test_pmap_field_optional_false. Retrieved 1/5 statements.
# Partially parsed test_pmap_field_optional_true. Retrieved 1/3 statements.
# Failed to parse test_pmap_field_with_invariant.
# Partially parsed test_pmap_field_factory_none_when_optional. Retrieved 2/5 statements.
# Failed to parse test_pmap_field_different_types.
# Failed to parse test_pmap_field_creates_initial_empty_map.
# Partially parsed test_pmap_field_mandatory_always_true. Retrieved 2/5 statements.


def test_case_0():
    var_0 = False

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = False
    var_1 = True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_check_type_with_valid_type. Retrieved 3/13 statements.
# Partially parsed test_check_type_with_multiple_valid_types. Retrieved 3/12 statements.
# Partially parsed test_check_type_with_none_field_type. Retrieved 4/11 statements.
# Partially parsed test_check_type_with_invalid_type. Retrieved 3/13 statements.
# Partially parsed test_check_type_with_empty_type_tuple. Retrieved 4/13 statements.
# Partially parsed test_check_type_with_subclass. Retrieved 3/14 statements.


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
    var_4 = 'TestClass.test_field'

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
    var_2 = 5



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_pfield_constructor. Retrieved 2/9 statements.
# Partially parsed test_pfield_constructor_with_none_values. Retrieved 2/3 statements.
# Partially parsed test_pfield_constructor_with_different_types. Retrieved 8/9 statements.


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
    var_7 = lambda x: repr(x)



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_check_field_parameters_valid_field.
# Failed to parse test_check_field_parameters_invalid_type_parameter.
# Failed to parse test_check_field_parameters_string_type.
# Failed to parse test_check_field_parameters_initial_wrong_type.
# Failed to parse test_check_field_parameters_initial_callable.
# Failed to parse test_check_field_parameters_non_callable_invariant.
# Failed to parse test_check_field_parameters_non_callable_factory.
# Failed to parse test_check_field_parameters_non_callable_serializer.
# Failed to parse test_check_field_parameters_empty_type.




# Parsed testcases at query #6
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_new_pmap_field_type.
# Failed to parse test_make_pmap_field_type_returns_cached_type.
# Failed to parse test_make_pmap_field_type_different_types_create_different_classes.
# Partially parsed test_make_pmap_field_type_has_reduce_method. Retrieved 1/4 statements.
# Failed to parse test_make_pmap_field_type_with_float_and_bool.


def test_case_0():
    var_0 = '__reduce__'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 6/16 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = [var_5]



# Parsed testcases at query #8
#--------------------------

# Failed to parse test_predicate_at_line_6_evaluates_to_false.




# Parsed testcases at query #9
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 1/9 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 2/5 statements.
# Partially parsed test_serialize_with_checked_type_and_custom_serializer. Retrieved 1/10 statements.
# Partially parsed test_serialize_with_non_checked_type. Retrieved 2/6 statements.
# Partially parsed test_serialize_with_different_formats. Retrieved 3/7 statements.


def test_case_0():
    var_0 = []
    var_1 = 'json'

def test_case_0():
    var_0 = 'xml'
    var_1 = 'test_value'

def test_case_0():
    var_0 = 'json'

def test_case_0():
    var_0 = []
    var_1 = 'json'
    var_2 = 'plain_value'

def test_case_0():
    var_0 = 'json'
    var_1 = 'data'
    var_2 = 'xml'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 6/16 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = [var_1, var_2, var_3]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_check_global_invariants_all_pass. Retrieved 1/7 statements.
# Partially parsed test_check_global_invariants_one_fails. Retrieved 1/8 statements.
# Partially parsed test_check_global_invariants_multiple_fail. Retrieved 1/10 statements.
# Partially parsed test_check_global_invariants_with_different_subject_types. Retrieved 9/16 statements.


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
    var_0 = 42
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = None



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_set_fields. Retrieved 23/56 statements.


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
    var_14 = 'base_field'
    var_15 = 'base_value'
    var_16 = 'value3'
    var_17 = 'field3'
    var_18 = 'field3'
    var_19 = 'base1_field'
    var_20 = 'base1_value'
    var_21 = 'base2_field'
    var_22 = 'base2_value'
    var_23 = 'value4'
    var_24 = 'field4'
    var_25 = 'value5'
    var_26 = 'pfield'
    var_27 = 'method'
    var_28 = 'attr'
    var_29 = lambda x: x
    var_30 = 42
    var_31 = []
    var_32 = 'pfield'
    var_33 = 10



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_check_type_predicate_evaluates_to_true. Retrieved 3/16 statements.
# Partially parsed test_check_type_predicate_evaluates_to_false. Retrieved 3/15 statements.
# Partially parsed test_check_type_with_none_field_type. Retrieved 4/12 statements.
# Partially parsed test_check_type_with_multiple_types. Retrieved 3/13 statements.
# Partially parsed test_check_type_with_multiple_types_no_match. Retrieved 3/14 statements.


def test_case_0():
    var_0 = 'MockDestinationClass'
    var_1 = 42
    var_2 = 'test_field'

def test_case_0():
    var_0 = 'MockDestinationClass'
    var_1 = 'not an int'
    var_2 = 'test_field'
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'MockDestinationClass'
    var_1 = None
    var_2 = 'any value'
    var_3 = 'test_field'

def test_case_0():
    var_0 = 'MockDestinationClass'
    var_1 = 'valid string'
    var_2 = 'test_field'

def test_case_0():
    var_0 = 'MockDestinationClass'
    var_1 = []
    var_2 = 'test_field'
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint_returns_false_when_ignore_extra_is_false. Retrieved 1/11 statements.
# Partially parsed test_is_field_ignore_extra_complaint_returns_false_when_type_cls_mismatch. Retrieved 1/10 statements.
# Partially parsed test_is_field_ignore_extra_complaint_returns_false_when_factory_has_no_ignore_extra_param. Retrieved 1/11 statements.
# Partially parsed test_is_field_ignore_extra_complaint_returns_true_when_all_conditions_met. Retrieved 1/11 statements.
# Partially parsed test_is_field_ignore_extra_complaint_with_set_type. Retrieved 2/11 statements.
# Partially parsed test_is_field_ignore_extra_complaint_with_empty_tuple_type. Retrieved 2/11 statements.


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
    var_0 = ()
    var_1 = True



# Parsed testcases at query #15
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
    var_11 = 'test_subject'
    var_12 = [var_3, var_6, var_10]
    var_13 = module_0.check_global_invariants(var_11, var_12)
    var_14 = bool(False)
    assert var_14 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = []
    var_2 = module_0.check_global_invariants(var_0, var_1)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = False
    var_1 = 'error1'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = 'error2'
    var_5 = (var_0, var_4)
    var_6 = lambda x: var_5
    var_7 = 'test_subject'
    var_8 = [var_3, var_6]
    var_9 = module_0.check_global_invariants(var_7, var_8)
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 6/17 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1, var_2]
    var_5 = [var_0, var_1, var_2]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 1/9 statements.
# Partially parsed test_serialize_with_regular_serializer. Retrieved 2/5 statements.
# Partially parsed test_serialize_with_checked_type_and_custom_serializer. Retrieved 1/10 statements.
# Partially parsed test_serialize_with_non_checked_type_and_no_serializer. Retrieved 2/6 statements.


def test_case_0():
    var_0 = []
    var_1 = 'json'

def test_case_0():
    var_0 = 'xml'
    var_1 = 'test_value'

def test_case_0():
    var_0 = 'json'

def test_case_0():
    var_0 = []
    var_1 = 'csv'
    var_2 = 'data'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda fmt, val: f'lambda_{fmt}_{val}'
    var_1 = 'txt'
    var_2 = 'content'
    var_3 = module_0.serialize(var_0, var_1, var_2)
    assert var_3 == 'lambda_txt_content'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint. Retrieved 4/35 statements.


def test_case_0():
    var_0 = False
    var_1 = set()
    var_2 = True
    var_3 = ()



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_set_fields_pfield_instance. Retrieved 5/18 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'not_pfield'
    var_3 = []
    var_4 = 'fields'
    var_5 = 'field1'
    var_6 = 'field1'



# Parsed testcases at query #20
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_pmap_subclass.
# Failed to parse test_make_pmap_field_type_caches_result.
# Failed to parse test_make_pmap_field_type_generates_correct_name.
# Failed to parse test_make_pmap_field_type_with_different_types.
# Failed to parse test_make_pmap_field_type_reduce_method.
# Failed to parse test_make_pmap_field_type_with_list_type.
# Failed to parse test_make_pmap_field_type_multiple_distinct_types.




# Parsed testcases at query #21
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 10/25 statements.
# Partially parsed test_restore_pmap_field_pickle_empty. Retrieved 1/10 statements.
# Partially parsed test_restore_pmap_field_pickle_with_int_keys. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 'TestType'
    var_1 = ()
    var_2 = 'test_field'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = {var_3: var_6, var_4: var_7, var_5: var_8}

def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 1.5
    var_4 = 2.5
    var_5 = 3.5
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_set_fields_predicate_isinstance_pfield. Retrieved 5/16 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'not_a_pfield'
    var_3 = []
    var_4 = 'fields'
    var_5 = 'field1'
    var_6 = 'field1'
    var_7 = 'field2'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_check_global_invariants_no_errors. Retrieved 2/9 statements.


def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = False
    assert var_2 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_pfield_factory_assignment. Retrieved 3/8 statements.


def test_case_0():
    var_0 = set()
    var_1 = None
    var_2 = False



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_sequence_field_with_optional_true. Retrieved 1/7 statements.
# Partially parsed test_sequence_field_with_optional_false. Retrieved 1/7 statements.
# Partially parsed test_sequence_field_factory_with_none_when_optional. Retrieved 2/7 statements.
# Partially parsed test_sequence_field_factory_with_list_when_optional. Retrieved 4/10 statements.
# Partially parsed test_sequence_field_factory_without_optional. Retrieved 5/11 statements.
# Partially parsed test_sequence_field_with_invariant. Retrieved 1/7 statements.
# Partially parsed test_sequence_field_with_item_invariant. Retrieved 1/9 statements.


def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = False

def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = False

def test_case_0():
    var_0 = False



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_pfield_factory_assignment. Retrieved 3/9 statements.


def test_case_0():
    var_0 = lambda x: str(x)
    var_1 = None
    var_2 = False



# Parsed testcases at query #27
#--------------------------

# Failed to parse test_check_field_parameters_valid_field.
# Failed to parse test_check_field_parameters_invalid_type_parameter.
# Failed to parse test_check_field_parameters_type_parameter_as_string.
# Failed to parse test_check_field_parameters_invalid_initial_type.
# Failed to parse test_check_field_parameters_callable_initial.
# Failed to parse test_check_field_parameters_no_initial.
# Failed to parse test_check_field_parameters_non_callable_invariant.
# Failed to parse test_check_field_parameters_non_callable_factory.
# Failed to parse test_check_field_parameters_non_callable_serializer.
# Failed to parse test_check_field_parameters_empty_type_list.




# Parsed testcases at query #28
#--------------------------

# Partially parsed test_set_fields_predicate_isinstance_pfield. Retrieved 4/14 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'not_a_pfield'
    var_3 = 'fields'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_sequence_field_invariant_parameter_evaluated_to_true. Retrieved 5/10 statements.


def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]



# Parsed testcases at query #30
#--------------------------

# Failed to parse test_check_field_parameters_predicate_line_3_false.




# Parsed testcases at query #31
#--------------------------

# Partially parsed test_check_global_invariants_all_pass. Retrieved 1/7 statements.
# Partially parsed test_check_global_invariants_one_fails. Retrieved 1/8 statements.
# Partially parsed test_check_global_invariants_multiple_fail. Retrieved 1/10 statements.
# Partially parsed test_check_global_invariants_with_complex_subject. Retrieved 4/10 statements.
# Partially parsed test_check_global_invariants_complex_subject_fails. Retrieved 3/10 statements.


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
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_make_seq_field_type. Retrieved 5/21 statements.


def test_case_0():
    var_0 = None
    var_1 = '__type__'
    var_2 = '__invariant__'
    var_3 = '__reduce__'
    var_4 = None



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_pfield_constructor. Retrieved 2/9 statements.
# Partially parsed test_pfield_constructor_with_none_values. Retrieved 2/3 statements.
# Partially parsed test_pfield_constructor_with_different_types. Retrieved 8/9 statements.


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
    var_7 = lambda x: repr(x)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint. Retrieved 5/31 statements.


def test_case_0():
    var_0 = False
    var_1 = ()
    var_2 = True
    var_3 = 'non.existent.Class'
    var_4 = {var_3}



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_serialize_checked_type_with_no_serializer. Retrieved 1/8 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'json'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_make_seq_field_type_creates_new_type. Retrieved 4/13 statements.
# Partially parsed test_make_seq_field_type_caches_result. Retrieved 1/8 statements.
# Partially parsed test_make_seq_field_type_with_different_item_types. Retrieved 1/6 statements.
# Failed to parse test_make_seq_field_type_with_invariant.
# Partially parsed test_make_seq_field_type_has_name. Retrieved 2/11 statements.
# Partially parsed test_make_seq_field_type_reduce_method. Retrieved 6/18 statements.


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
    var_1 = '__name__'

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 0



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint_predicate_line_6_evaluates_to_false. Retrieved 1/12 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_make_seq_field_type. Retrieved 5/20 statements.


def test_case_0():
    var_0 = None
    var_1 = '__type__'
    var_2 = '__invariant__'
    var_3 = '__reduce__'
    var_4 = None



# Parsed testcases at query #39
#--------------------------

# Failed to parse test_sequence_field_invariant_parameter_is_pfield_no_invariant_by_default.




# Parsed testcases at query #40
#--------------------------

# Failed to parse test_check_field_parameters_predicate_line_3_false.




# Parsed testcases at query #41
#--------------------------

# Partially parsed test_pmap_field_creates_field_with_checked_pmap_type. Retrieved 1/6 statements.
# Partially parsed test_pmap_field_with_optional_true. Retrieved 2/7 statements.
# Partially parsed test_pmap_field_with_optional_false. Retrieved 2/7 statements.
# Failed to parse test_pmap_field_with_invariant.
# Partially parsed test_pmap_field_factory_with_optional_none_argument. Retrieved 2/5 statements.
# Partially parsed test_pmap_field_factory_with_optional_dict_argument. Retrieved 3/6 statements.
# Partially parsed test_pmap_field_factory_without_optional. Retrieved 4/7 statements.
# Failed to parse test_pmap_field_returns_field_object.
# Failed to parse test_pmap_field_caching_same_type_combination.
# Failed to parse test_pmap_field_different_type_combinations.


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



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_make_seq_field_type_creates_new_type. Retrieved 2/8 statements.
# Partially parsed test_make_seq_field_type_caches_result. Retrieved 2/8 statements.
# Partially parsed test_make_seq_field_type_different_item_types_different_results. Retrieved 2/7 statements.
# Partially parsed test_make_seq_field_type_has_reduce_method. Retrieved 3/10 statements.
# Partially parsed test_make_seq_field_type_name_generation. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: len(x) > var_0

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = '__reduce__'

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 5/14 statements.
# Partially parsed test_restore_seq_field_pickle_with_strings. Retrieved 5/14 statements.
# Partially parsed test_restore_seq_field_pickle_empty_data. Retrieved 2/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = []



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_check_field_parameters_valid_type_as_type. Retrieved 4/8 statements.
# Partially parsed test_check_field_parameters_valid_type_as_string. Retrieved 5/8 statements.
# Partially parsed test_check_field_parameters_invalid_type_parameter. Retrieved 4/9 statements.
# Partially parsed test_check_field_parameters_initial_no_initial. Retrieved 3/9 statements.
# Partially parsed test_check_field_parameters_initial_callable. Retrieved 4/8 statements.
# Partially parsed test_check_field_parameters_initial_valid_type. Retrieved 4/8 statements.
# Partially parsed test_check_field_parameters_initial_invalid_type. Retrieved 4/10 statements.
# Partially parsed test_check_field_parameters_invariant_not_callable. Retrieved 4/9 statements.
# Partially parsed test_check_field_parameters_factory_not_callable. Retrieved 4/9 statements.
# Partially parsed test_check_field_parameters_serializer_not_callable. Retrieved 4/9 statements.
# Partially parsed test_check_field_parameters_empty_type_list. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 5
    var_1 = lambda x: True
    var_2 = lambda : None
    var_3 = lambda x: x

def test_case_0():
    var_0 = 'int'
    var_1 = 'str'
    var_2 = [var_0, var_1]
    var_3 = 5
    var_4 = lambda x: True
    var_5 = lambda : None
    var_6 = lambda x: x

def test_case_0():
    var_0 = 5
    var_1 = lambda x: True
    var_2 = lambda : None
    var_3 = lambda x: x
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Type parameter expected'

def test_case_0():
    var_0 = []
    var_1 = lambda x: True
    var_2 = lambda : None
    var_3 = lambda x: x

def test_case_0():
    var_0 = lambda : 5
    var_1 = lambda x: True
    var_2 = lambda : None
    var_3 = lambda x: x

def test_case_0():
    var_0 = 'hello'
    var_1 = lambda x: True
    var_2 = lambda : None
    var_3 = lambda x: x

def test_case_0():
    var_0 = []
    var_1 = 3.14
    var_2 = lambda x: True
    var_3 = lambda : None
    var_4 = lambda x: x
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Initial has invalid type'

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
    var_1 = lambda x: True
    var_2 = 'not callable'
    var_3 = lambda x: x
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Factory must be callable'

def test_case_0():
    var_0 = 5
    var_1 = lambda x: True
    var_2 = lambda : None
    var_3 = 'not callable'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Serializer must be callable'

def test_case_0():
    var_0 = []
    var_1 = 5
    var_2 = lambda x: True
    var_3 = lambda : None
    var_4 = lambda x: x



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_set_fields_with_parent_fields. Retrieved 7/17 statements.
# Partially parsed test_set_fields_moves_pfield_from_dct. Retrieved 5/10 statements.
# Partially parsed test_set_fields_multiple_bases. Retrieved 11/25 statements.
# Partially parsed test_set_fields_mixed_content. Retrieved 7/12 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'fields'
    var_3 = module_0.set_fields(var_0, var_1, var_2)
    var_4 = bool(var_2 in var_0)
    assert var_4 is True
    var_5 = var_0[var_2]
    var_6 = bool(var_0[var_2] == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'Parent'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = 'fields'
    var_5 = 'parent_field'
    var_6 = {}
    var_7 = 'fields'
    var_8 = bool(var_7 in var_6)
    assert var_8 is True
    var_9 = 'parent_field'
    var_10 = bool('parent_field' in var_6[var_7])
    assert var_10 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'other_value'
    var_2 = 123
    var_3 = []
    var_4 = 'fields'
    var_5 = 'test_field'
    var_6 = 'test_field'
    var_7 = 'other_value'

def test_case_0():
    var_0 = 'Parent1'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = 'fields'
    var_5 = 'field1'
    var_6 = 'Parent2'
    var_7 = ()
    var_8 = {}
    var_9 = [var_6, var_7, var_8]
    var_10 = 'field2'
    var_11 = {}
    var_12 = 'fields'
    var_13 = bool(var_12 in var_11)
    assert var_13 is True
    var_14 = 'field1'
    var_15 = bool('field1' in var_11[var_12])
    assert var_15 is True
    var_16 = 'field2'
    var_17 = bool('field2' in var_11[var_12])
    assert var_17 is True

def test_case_0():
    var_0 = 'pfield'
    var_1 = 'regular'
    var_2 = 'number'
    var_3 = 'value'
    var_4 = 42
    var_5 = []
    var_6 = 'fields'
    var_7 = 'pfield'
    var_8 = 'regular'
    var_9 = 'number'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_check_global_invariants_raises_exception_when_invariants_fail. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 'test_subject'



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_check_global_invariants_no_errors. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'test_subject'
    var_1 = True
    var_2 = False
    assert var_2 is True



# Parsed testcases at query #48
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_new_pmap_subclass.
# Failed to parse test_make_pmap_field_type_caches_result.
# Failed to parse test_make_pmap_field_type_different_types_create_different_classes.
# Partially parsed test_make_pmap_field_type_has_reduce_method. Retrieved 1/6 statements.
# Failed to parse test_make_pmap_field_type_with_multiple_key_value_types.


def test_case_0():
    var_0 = '__reduce__'



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_pmap_field_optional_false_predicate. Retrieved 6/12 statements.


def test_case_0():
    var_0 = False
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_make_pmap_field_type_creates_new_type. Retrieved 2/6 statements.
# Failed to parse test_make_pmap_field_type_caches_result.
# Failed to parse test_make_pmap_field_type_sets_name.
# Failed to parse test_make_pmap_field_type_different_types_different_classes.
# Partially parsed test_make_pmap_field_type_has_reduce_method. Retrieved 1/6 statements.


def test_case_0():
    var_0 = '__key_type__'
    var_1 = '__value_type__'

def test_case_0():
    var_0 = '__reduce__'



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_check_global_invariants_no_errors. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'test_subject'



# Parsed testcases at query #52
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
    var_7 = [var_3, var_6]
    var_8 = 'test_subject'
    var_9 = module_0.check_global_invariants(var_8, var_7)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = 'code1'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = False
    var_5 = 'error_code1'
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
    var_1 = 'error_code1'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = 'error_code2'
    var_5 = (var_0, var_4)
    var_6 = lambda x: var_5
    var_7 = True
    var_8 = 'code3'
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
    var_0 = 'key'
    var_1 = 'nested'
    var_2 = 'value'
    var_3 = 'data'
    var_4 = 123
    var_5 = {var_3: var_4}
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = True
    var_8 = 'valid_structure'
    var_9 = (var_7, var_8)
    var_10 = lambda x: var_9
    var_11 = 'valid_content'
    var_12 = (var_7, var_11)
    var_13 = lambda x: var_12
    var_14 = [var_10, var_13]
    var_15 = module_0.check_global_invariants(var_6, var_14)



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_check_global_invariants_all_pass. Retrieved 1/8 statements.
# Partially parsed test_check_global_invariants_single_failure. Retrieved 1/8 statements.
# Partially parsed test_check_global_invariants_multiple_failures. Retrieved 1/10 statements.
# Partially parsed test_check_global_invariants_all_fail. Retrieved 1/8 statements.
# Partially parsed test_check_global_invariants_with_none_subject. Retrieved 1/6 statements.


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

def test_case_0():
    var_0 = 'test_subject'
    var_1 = bool(False)
    assert var_1 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = []
    var_1 = 'test_subject'
    var_2 = module_0.check_global_invariants(var_1, var_0)
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = None
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 10/20 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'TestSeqField'
    var_5 = ()
    var_6 = 'create'
    var_7 = lambda self, d, _factory_fields=None: PVector(d)
    var_8 = {var_6: var_7}
    var_9 = [var_4, var_5, var_8]
    var_10 = [var_0, var_1, var_2]



# Parsed testcases at query #55
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_checked_pmap_subclass.
# Failed to parse test_make_pmap_field_type_caches_types.
# Failed to parse test_make_pmap_field_type_different_types_create_different_classes.
# Partially parsed test_make_pmap_field_type_has_reduce_method. Retrieved 1/6 statements.
# Partially parsed test_make_pmap_field_type_name_format. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '__reduce__'

def test_case_0():
    var_0 = 'To'
    var_1 = 'PMap'



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_sequence_field_invariant_parameter_default. Retrieved 2/7 statements.


def test_case_0():
    var_0 = False
    var_1 = []



# Parsed testcases at query #57
#--------------------------

# Failed to parse test_pmap_field_basic.
# Partially parsed test_pmap_field_with_optional_true. Retrieved 1/3 statements.
# Partially parsed test_pmap_field_with_optional_false. Retrieved 1/3 statements.
# Failed to parse test_pmap_field_with_invariant.
# Failed to parse test_pmap_field_factory_callable.
# Partially parsed test_pmap_field_factory_with_optional_is_callable. Retrieved 1/5 statements.
# Failed to parse test_pmap_field_type_set.
# Failed to parse test_pmap_field_serializer_callable.


def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = False

def test_case_0():
    var_0 = True



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 6/14 statements.
# Partially parsed test_restore_pmap_field_pickle_empty_data. Retrieved 3/9 statements.
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
    var_1 = {}
    var_2 = module_0.pmap(var_1)

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



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_pmap_field_optional_false_predicate. Retrieved 1/8 statements.


def test_case_0():
    var_0 = False



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 6/15 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_pmap_field_basic. Retrieved 1/6 statements.
# Partially parsed test_pmap_field_optional. Retrieved 3/9 statements.
# Failed to parse test_pmap_field_with_invariant.
# Partially parsed test_pmap_field_factory_non_optional. Retrieved 6/13 statements.
# Partially parsed test_pmap_field_factory_optional_with_none. Retrieved 2/5 statements.
# Partially parsed test_pmap_field_factory_optional_with_data. Retrieved 4/9 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.pmap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.pmap()
    var_2 = None

def test_case_0():
    var_0 = False
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}

def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = True
    var_1 = 'x'
    var_2 = 10
    var_3 = {var_1: var_2}



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_pmap_field_returns_field_with_correct_type. Retrieved 2/9 statements.


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = []



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_pmap_field_optional_false_predicate. Retrieved 2/5 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.pmap()



# Parsed testcases at query #64
#--------------------------

# Failed to parse test_sequence_field_invariant_parameter_is_pfield_no_invariant.




# Parsed testcases at query #65
#--------------------------

# Partially parsed test_serialize_checked_type_with_no_serializer. Retrieved 1/11 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'json'



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_set_fields. Retrieved 24/65 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = []
    var_5 = 'fields'
    var_6 = 'fields'
    var_7 = 'field1'
    var_8 = 'field2'
    var_9 = 'field1'
    var_10 = 'field2'
    var_11 = 'base_field'
    var_12 = 'base_value'
    var_13 = 'fields'
    var_14 = 'base_field'
    var_15 = 'field1'
    var_16 = 'field1'
    var_17 = 'pfield'
    var_18 = 'regular'
    var_19 = 'pfield2'
    var_20 = 'pvalue'
    var_21 = 'regular_value'
    var_22 = 'pvalue2'
    var_23 = []
    var_24 = 'fields'
    var_25 = 'pfield'
    var_26 = 'pfield2'
    var_27 = 'regular'
    var_28 = 'pfield'
    var_29 = 'pfield2'
    var_30 = {}
    var_31 = []
    var_32 = module_0.set_fields(var_30, var_31, var_5)
    var_33 = 'fields'
    var_34 = bool('fields' in var_30)
    assert var_34 is True
    var_35 = var_30['fields']
    var_36 = bool(var_30['fields'] == {})
    assert var_36 is True
    var_37 = 'base1_field'
    var_38 = 'base1_value'
    var_39 = 'base2_field'
    var_40 = 'base2_value'
    var_41 = 'child_field'
    var_42 = 'child_value'
    var_43 = 'fields'
    var_44 = 'base1_field'
    var_45 = 'base2_field'
    var_46 = 'child_field'



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_check_global_invariants_raises_exception_when_invariants_fail. Retrieved 3/18 statements.


def test_case_0():
    var_0 = {}
    var_1 = False
    var_2 = True
    var_3 = bool(var_2)
    assert var_3 is True



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_check_field_parameters_predicate_line_3_false. Retrieved 1/20 statements.


def test_case_0():
    var_0 = 'int'



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 7/18 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_check_global_invariants_with_all_passing_invariants. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'test_subject'
    var_1 = True
    var_2 = False
    assert var_2 is True



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_pmap_field_basic. Retrieved 1/4 statements.
# Partially parsed test_pmap_field_optional_true. Retrieved 2/5 statements.
# Partially parsed test_pmap_field_optional_false. Retrieved 2/5 statements.
# Failed to parse test_pmap_field_with_invariant.
# Partially parsed test_pmap_field_factory_without_optional. Retrieved 1/5 statements.
# Partially parsed test_pmap_field_factory_with_optional. Retrieved 1/5 statements.
# Partially parsed test_pmap_field_factory_none_with_optional. Retrieved 2/5 statements.
# Partially parsed test_pmap_field_factory_creates_pmap. Retrieved 6/9 statements.
# Failed to parse test_pmap_field_multiple_calls_same_types.
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

def test_case_0():
    var_0 = False

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = False
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint_predicate_line_6_false. Retrieved 1/13 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 4/14 statements.
# Partially parsed test_restore_seq_field_pickle_empty. Retrieved 1/11 statements.
# Partially parsed test_restore_seq_field_pickle_with_strings. Retrieved 4/13 statements.


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



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_make_pmap_field_type_creates_new_type. Retrieved 2/6 statements.
# Failed to parse test_make_pmap_field_type_caches_result.
# Failed to parse test_make_pmap_field_type_generates_correct_name.
# Partially parsed test_make_pmap_field_type_has_reduce_method. Retrieved 1/6 statements.
# Failed to parse test_make_pmap_field_type_different_types_create_different_classes.
# Failed to parse test_make_pmap_field_type_with_float_and_bool.


def test_case_0():
    var_0 = '__key_type__'
    var_1 = '__value_type__'

def test_case_0():
    var_0 = '__reduce__'



# Parsed testcases at query #75
#--------------------------




def test_case_0():
    var_0 = False
    var_1 = bool(not var_0)
    assert var_1 is True



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_make_seq_field_type_creates_new_type. Retrieved 2/9 statements.
# Partially parsed test_make_seq_field_type_caches_result. Retrieved 1/7 statements.
# Failed to parse test_make_seq_field_type_with_invariant.
# Partially parsed test_make_seq_field_type_reduce_method. Retrieved 5/12 statements.
# Partially parsed test_make_seq_field_type_name_generation. Retrieved 2/12 statements.


def test_case_0():
    var_0 = None
    var_1 = '__reduce__'

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = None
    var_1 = '__name__'



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = None
    var_5 = [var_4]



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_check_global_invariants_raises_exception_when_invariants_fail. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 'test_subject'
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 1/11 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'json'



# Parsed testcases at query #80
#--------------------------

# Failed to parse test_check_field_parameters_valid_field.
# Failed to parse test_check_field_parameters_invalid_type_parameter.
# Failed to parse test_check_field_parameters_invalid_initial_type.
# Failed to parse test_check_field_parameters_callable_initial.
# Failed to parse test_check_field_parameters_non_callable_invariant.
# Failed to parse test_check_field_parameters_non_callable_factory.
# Failed to parse test_check_field_parameters_non_callable_serializer.
# Failed to parse test_check_field_parameters_string_type.
# Failed to parse test_check_field_parameters_empty_type_list.




# Parsed testcases at query #81
#--------------------------

# Partially parsed test_make_seq_field_type. Retrieved 4/22 statements.


def test_case_0():
    var_0 = None
    var_1 = '__reduce__'
    var_2 = None
    var_3 = '__name__'



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_predicate_isinstance_pfield_evaluates_to_true. Retrieved 3/9 statements.


def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'fields'



# Parsed testcases at query #83
#--------------------------

# Failed to parse test_sequence_field_invariant_parameter_is_pfield_no_invariant.




# Parsed testcases at query #84
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint_returns_false_when_ignore_extra_is_false. Retrieved 1/12 statements.
# Partially parsed test_is_field_ignore_extra_complaint_returns_false_when_type_cls_mismatch. Retrieved 1/11 statements.
# Partially parsed test_is_field_ignore_extra_complaint_returns_false_when_no_ignore_extra_param. Retrieved 1/11 statements.
# Partially parsed test_is_field_ignore_extra_complaint_returns_true_when_all_conditions_met. Retrieved 1/11 statements.
# Partially parsed test_is_field_ignore_extra_complaint_with_set_type. Retrieved 1/11 statements.
# Partially parsed test_is_field_ignore_extra_complaint_with_empty_tuple. Retrieved 2/11 statements.


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

def test_case_0():
    var_0 = ()
    var_1 = True



# Parsed testcases at query #85
#--------------------------




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
    var_9 = 'test_subject'
    var_10 = module_0.check_global_invariants(var_9, var_8)

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
    var_8 = (var_0, var_1)
    var_9 = lambda x: var_8
    var_10 = [var_3, var_7, var_9]
    var_11 = 'test_subject'
    var_12 = module_0.check_global_invariants(var_11, var_10)
    var_13 = bool(False)
    assert var_13 is True

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
    var_11 = 'ERROR_CODE_3'
    var_12 = (var_0, var_11)
    var_13 = lambda x: var_12
    var_14 = [var_3, var_6, var_10, var_13]
    var_15 = 'test_subject'
    var_16 = module_0.check_global_invariants(var_15, var_14)
    var_17 = bool(False)
    assert var_17 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = []
    var_1 = 'test_subject'
    var_2 = module_0.check_global_invariants(var_1, var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = False
    var_1 = 'ERROR_A'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = 'ERROR_B'
    var_5 = (var_0, var_4)
    var_6 = lambda x: var_5
    var_7 = [var_3, var_6]
    var_8 = 'test_subject'
    var_9 = module_0.check_global_invariants(var_8, var_7)
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_make_pmap_field_type. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 'PMap'
    var_1 = '__reduce__'
    var_2 = 'StrToIntPMap'
    var_3 = 'IntToStrPMap'



# Parsed testcases at query #87
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 7/18 statements.
# Partially parsed test_restore_pmap_field_pickle_empty. Retrieved 1/7 statements.
# Partially parsed test_restore_pmap_field_pickle_with_data. Retrieved 5/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = len(var_4)

def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 'name'
    var_1 = 'city'
    var_2 = 'Alice'
    var_3 = 'NYC'
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_pmap_field_optional_false_predicate. Retrieved 1/4 statements.


def test_case_0():
    var_0 = False



# Parsed testcases at query #89
#--------------------------

# Partially parsed test_pmap_field_basic. Retrieved 1/4 statements.
# Partially parsed test_pmap_field_with_optional_true. Retrieved 2/5 statements.
# Partially parsed test_pmap_field_with_optional_false. Retrieved 2/5 statements.
# Failed to parse test_pmap_field_with_invariant.
# Partially parsed test_pmap_field_factory_without_optional. Retrieved 1/6 statements.
# Partially parsed test_pmap_field_factory_with_optional. Retrieved 1/6 statements.
# Failed to parse test_pmap_field_different_key_value_types.


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

def test_case_0():
    var_0 = False

def test_case_0():
    var_0 = True



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_check_global_invariants_raises_exception_when_invariants_fail. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 'test_subject'



# Parsed testcases at query #91
#--------------------------

# Partially parsed test_set_fields_predicate_isinstance_pfield. Retrieved 5/16 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'not_a_pfield'
    var_3 = []
    var_4 = 'fields'
    var_5 = 'field1'
    var_6 = 'field1'
    var_7 = 'field2'



# Parsed testcases at query #92
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 6/17 statements.
# Partially parsed test_restore_seq_field_pickle_with_strings. Retrieved 6/17 statements.
# Partially parsed test_restore_seq_field_pickle_empty_data. Retrieved 3/14 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = [var_1, var_2, var_3]

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = [var_1, var_2, var_3]

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = []
    var_2 = []



# Parsed testcases at query #93
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_checked_pmap_subclass.
# Failed to parse test_make_pmap_field_type_caches_result.
# Failed to parse test_make_pmap_field_type_different_types_create_different_classes.
# Partially parsed test_make_pmap_field_type_has_reduce_method. Retrieved 1/6 statements.
# Failed to parse test_make_pmap_field_type_name_format.


def test_case_0():
    var_0 = '__reduce__'



# Parsed testcases at query #94
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint_predicate_line_6_false. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = True



# Parsed testcases at query #95
#--------------------------

# Partially parsed test_check_field_parameters_predicate_line_3_false. Retrieved 2/11 statements.


def test_case_0():
    var_0 = True
    var_1 = False
    assert var_1 is True



# Parsed testcases at query #96
#--------------------------

# Partially parsed test_sequence_field_with_optional_true. Retrieved 2/7 statements.
# Partially parsed test_sequence_field_with_optional_false. Retrieved 2/5 statements.
# Partially parsed test_sequence_field_factory_with_optional_none. Retrieved 2/6 statements.
# Partially parsed test_sequence_field_factory_with_optional_list. Retrieved 5/10 statements.
# Partially parsed test_sequence_field_has_correct_type. Retrieved 2/7 statements.
# Partially parsed test_sequence_field_factory_callable. Retrieved 2/7 statements.


def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = False
    var_1 = None

def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_2, var_3]

def test_case_0():
    var_0 = False
    var_1 = None

def test_case_0():
    var_0 = False
    var_1 = None



# Parsed testcases at query #97
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 14/25 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'PMapField'
    var_2 = 'create'
    var_3 = lambda cls, data, _factory_fields=None: pmap(data)
    var_4 = classmethod(var_3)
    var_5 = {var_2: var_4}
    var_6 = 'a'
    var_7 = 'b'
    var_8 = 'c'
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = {var_6: var_9, var_7: var_10, var_8: var_11}
    var_13 = bool(var_5)
    assert var_13 is True
    var_14 = module_0.pmap(var_12)



# Parsed testcases at query #98
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 1/12 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'json'



# Parsed testcases at query #99
#--------------------------

# Partially parsed test_make_seq_field_type_creates_new_type. Retrieved 5/13 statements.
# Partially parsed test_make_seq_field_type_caches_result. Retrieved 2/8 statements.
# Partially parsed test_make_seq_field_type_different_types_create_different_classes. Retrieved 2/7 statements.
# Partially parsed test_make_seq_field_type_has_correct_name. Retrieved 3/12 statements.
# Partially parsed test_make_seq_field_type_reduce_method. Retrieved 6/18 statements.


def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = '__type__'
    var_3 = '__invariant__'
    var_4 = '__reduce__'

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = '__name__'

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_2, var_3]
    var_5 = 0



# Parsed testcases at query #100
#--------------------------

# Partially parsed test_check_global_invariants_all_pass. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 'test_subject'



# Parsed testcases at query #101
#--------------------------

# Partially parsed test_pmap_field_optional_false_predicate. Retrieved 2/5 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.pmap()



# Parsed testcases at query #102
#--------------------------

# Failed to parse test_pmap_field_basic.
# Partially parsed test_pmap_field_with_optional. Retrieved 1/3 statements.
# Failed to parse test_pmap_field_with_invariant.
# Partially parsed test_pmap_field_optional_factory_with_none. Retrieved 2/5 statements.
# Partially parsed test_pmap_field_type_parameter. Retrieved 1/5 statements.
# Partially parsed test_pmap_field_type_parameter_optional. Retrieved 1/5 statements.
# Failed to parse test_pmap_field_creates_checked_pmap.
# Failed to parse test_pmap_field_multiple_calls_same_types.
# Failed to parse test_pmap_field_different_types.


def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = False

def test_case_0():
    var_0 = True



# Parsed testcases at query #103
#--------------------------

# Partially parsed test_check_field_parameters_predicate_line_3_false. Retrieved 1/20 statements.


def test_case_0():
    var_0 = 'string'



# Parsed testcases at query #104
#--------------------------

# Partially parsed test_set_fields_pfield_predicate. Retrieved 4/18 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'not_pfield'
    var_3 = 'fields'
    var_4 = 'field1'
    var_5 = 'field1'
    var_6 = 'field2'



# Parsed testcases at query #105
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint_predicate_line_6_false. Retrieved 2/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True



# Parsed testcases at query #106
#--------------------------

# Partially parsed test_make_seq_field_type_creates_new_type. Retrieved 1/7 statements.
# Partially parsed test_make_seq_field_type_caches_result. Retrieved 1/7 statements.
# Partially parsed test_make_seq_field_type_has_correct_name. Retrieved 2/10 statements.
# Failed to parse test_make_seq_field_type_with_invariant.
# Partially parsed test_make_seq_field_type_reduce_method. Retrieved 6/16 statements.


def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = None
    var_1 = '__name__'

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 0



# Parsed testcases at query #107
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 6/17 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap()
    var_6 = [var_5]



# Parsed testcases at query #108
#--------------------------

# Partially parsed test_sequence_field_basic. Retrieved 5/10 statements.
# Partially parsed test_sequence_field_optional_true. Retrieved 5/11 statements.
# Partially parsed test_sequence_field_with_invariant. Retrieved 4/13 statements.
# Partially parsed test_sequence_field_with_item_invariant. Retrieved 3/10 statements.
# Partially parsed test_sequence_field_optional_factory_with_none. Retrieved 2/6 statements.
# Partially parsed test_sequence_field_optional_factory_with_values. Retrieved 8/13 statements.


def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = True
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = None

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]

def test_case_0():
    var_0 = False
    var_1 = 5
    var_2 = [var_1]

def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]



# Parsed testcases at query #109
#--------------------------

# Partially parsed test_check_global_invariants_invariant_receives_subject. Retrieved 2/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = ''
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = (var_0, var_1)
    var_5 = lambda x: var_4
    var_6 = 'test_subject'
    var_7 = [var_3, var_5]
    var_8 = module_0.check_global_invariants(var_6, var_7)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = ''
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = False
    var_5 = 'ERROR_CODE_1'
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
    var_1 = 'ERROR_CODE_1'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = 'ERROR_CODE_2'
    var_5 = (var_0, var_4)
    var_6 = lambda x: var_5
    var_7 = True
    var_8 = ''
    var_9 = (var_7, var_8)
    var_10 = lambda x: var_9
    var_11 = 'test_subject'
    var_12 = [var_3, var_6, var_10]
    var_13 = module_0.check_global_invariants(var_11, var_12)
    var_14 = bool(False)
    assert var_14 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = []
    var_2 = module_0.check_global_invariants(var_0, var_1)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = ''
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = None
    var_5 = [var_3]
    var_6 = module_0.check_global_invariants(var_4, var_5)

def test_case_0():
    var_0 = []
    var_1 = 'test_subject'
    var_2 = var_0[0]
    var_3 = bool(var_0[0] == var_1)
    assert var_3 is True



# Parsed testcases at query #110
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_checked_pmap_subclass.
# Failed to parse test_make_pmap_field_type_caches_types.
# Failed to parse test_make_pmap_field_type_different_key_types.
# Failed to parse test_make_pmap_field_type_different_value_types.
# Partially parsed test_make_pmap_field_type_has_reduce_method. Retrieved 1/7 statements.
# Partially parsed test_make_pmap_field_type_name_format. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '__reduce__'

def test_case_0():
    var_0 = 'To'
    var_1 = 'PMap'
    var_2 = 'Str'
    var_3 = 'Int'



