####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
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
    var_8 = 'error_1'
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
    var_11 = [var_6, var_10]
    var_12 = module_0.check_global_invariants(var_2, var_11)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'data'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = 'ERR_A'
    var_5 = (var_3, var_4)
    var_6 = lambda s: var_5
    var_7 = True
    var_8 = None
    var_9 = (var_7, var_8)
    var_10 = lambda s: var_9
    var_11 = 'ERR_B'
    var_12 = (var_3, var_11)
    var_13 = lambda s: var_12
    var_14 = [var_6, var_10, var_13]
    var_15 = module_0.check_global_invariants(var_2, var_14)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'data'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.check_global_invariants(var_2, var_3)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint_ignore_extra_false. Retrieved 1/9 statements.
# Partially parsed test_is_field_ignore_extra_complaint_not_type_cls. Retrieved 1/9 statements.
# Partially parsed test_is_field_ignore_extra_complaint_no_param_in_factory. Retrieved 1/9 statements.
# Partially parsed test_is_field_ignore_extra_complaint_success. Retrieved 1/10 statements.
# Partially parsed test_is_field_ignore_extra_complaint_with_set_type. Retrieved 1/11 statements.


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

# Partially parsed test_make_seq_field_type_new_class_creation. Retrieved 2/9 statements.
# Partially parsed test_make_seq_field_type_returns_cached_class. Retrieved 2/7 statements.


def test_case_0():
    var_0 = True
    var_1 = True

def test_case_0():
    var_0 = False
    var_1 = False



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_check_field_parameters_valid_inputs. Retrieved 6/16 statements.
# Partially parsed test_check_field_parameters_invalid_type_element. Retrieved 3/13 statements.


def test_case_0():
    var_0 = None
    var_1 = 10
    var_2 = True
    var_3 = lambda x: var_2
    var_4 = lambda : var_2
    var_5 = lambda x: x

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_is_type_cls_with_set_field_type.
# Failed to parse test_is_type_cls_with_tuple_containing_int.
# Failed to parse test_is_type_cls_with_tuple_containing_float.
# Failed to parse test_is_type_cls_with_tuple_containing_subclass.
# Failed to parse test_is_type_cls_with_mismatching_types_in_tuple.
# Partially parsed test_is_type_cls_with_empty_tuple. Retrieved 1/3 statements.
# Failed to parse test_is_type_cls_with_multiple_types_in_tuple_matching_first.
# Failed to parse test_is_type_cls_with_multiple_types_in_tuple_not_matching_first.


def test_case_0():
    var_0 = ()



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'error1'
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = [var_4, var_7]
    var_9 = module_0.check_global_invariants(var_0, var_8)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = False
    var_6 = 'error2'
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = [var_4, var_8]
    var_10 = module_0.check_global_invariants(var_0, var_9)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test'
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
    var_13 = module_0.check_global_invariants(var_0, var_12)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = []
    var_2 = module_0.check_global_invariants(var_0, var_1)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 2/7 statements.
# Partially parsed test_serialize_with_checked_type_and_standard_serializer. Retrieved 4/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda fmt, val: f'{fmt}:{val}'
    var_1 = 'json'
    var_2 = 'data'
    var_3 = module_0.serialize(var_0, var_1, var_2)
    assert var_3 == 'json:data'

def test_case_0():
    var_0 = 'NONE'
    var_1 = 'xml'

def test_case_0():
    var_0 = lambda fmt, val: f'{fmt}_{val.val}'
    var_1 = 'NONE'
    var_2 = 'data'
    var_3 = 'csv'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'format'
    var_1 = 'payload'
    var_2 = lambda fmt, val: {var_0: fmt, var_1: val}
    var_3 = 'msgpack'
    var_4 = 123
    var_5 = module_0.serialize(var_2, var_3, var_4)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_set_fields_merges_bases_and_moves_pfields. Retrieved 28/60 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'p_field_key'
    var_1 = 'existing_val'
    var_2 = module_0._PField()
    var_3 = 10
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'new_meta'
    var_6 = 'extra'
    var_7 = {}
    var_8 = 'target'
    var_9 = 'key1'
    var_10 = 'val1'
    var_11 = {var_9: var_10}
    var_12 = 'RealBase'
    var_13 = ()
    var_14 = 'key2'
    var_15 = 'val2'
    var_16 = {var_14: var_15}
    var_17 = {var_8: var_16}
    var_18 = 'a'
    var_19 = 1
    var_20 = {var_18: var_19}
    var_21 = 'b'
    var_22 = 2
    var_23 = {var_21: var_22}
    var_24 = 'other'
    var_25 = module_0._PField()
    var_26 = 5
    var_27 = {var_0: var_25, var_24: var_26}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'old'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'new'
    var_5 = module_0.set_fields(var_2, var_3, var_4)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_check_field_parameters_valid. Retrieved 6/16 statements.
# Partially parsed test_check_field_parameters_invalid_type_element. Retrieved 1/8 statements.
# Partially parsed test_check_field_parameters_invalid_initial_type. Retrieved 2/13 statements.
# Partially parsed test_check_field_parameters_invalid_invariant. Retrieved 3/14 statements.
# Partially parsed test_check_field_parameters_invalid_factory. Retrieved 3/14 statements.
# Partially parsed test_check_field_parameters_invalid_serializer. Retrieved 3/14 statements.


def test_case_0():
    var_0 = None
    var_1 = 'hello'
    var_2 = True
    var_3 = lambda x: var_2
    var_4 = lambda : var_2
    var_5 = lambda x: x

def test_case_0():
    var_0 = 123

def test_case_0():
    var_0 = None
    var_1 = 'not an int'

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 'not callable'

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 'not callable'

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 'not callable'



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_make_pmap_field_type_new_class_creation.
# Failed to parse test_make_pmap_field_type_memoization.
# Failed to parse test_make_pmap_field_type_different_types.




# Parsed testcases at query #6
#--------------------------

# Partially parsed test_check_global_invariants_raises_exception_when_invariants_fail. Retrieved 11/20 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = False
    var_2 = 'ERR_001'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = True
    var_6 = None
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = [var_4, var_8]
    var_10 = module_0.check_global_invariants(var_0, var_9)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_set_fields_merges_bases_and_moves_pfields. Retrieved 6/20 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0._PField()
    var_1 = 'attr1'
    var_2 = 'other'
    var_3 = 'value'
    var_4 = {var_1: var_0, var_2: var_3}
    var_5 = 'merged_attr'



# Parsed testcases at query #8
#--------------------------

# Failed to parse test_types_to_names_with_simple_types.
# Failed to parse test_types_to_names_with_single_type.
# Failed to parse test_types_to_names_with_complex_objects.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ()
    var_1 = module_0._types_to_names(var_0)
    assert var_1 == ''

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = 'builtins.str'
    var_2 = (var_0, var_1)
    var_3 = module_0._types_to_names(var_2)
    assert var_3 == 'IntStr'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_check_field_parameters_predicate_false_by_matching_type. Retrieved 1/11 statements.


def test_case_0():
    var_0 = 10



# Parsed testcases at query #10
#--------------------------




import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = False
    var_2 = 'ERR001'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = [var_4]
    var_6 = module_0.check_global_invariants(var_0, var_5)



# Parsed testcases at query #11
#--------------------------

# Failed to parse test_make_pmap_field_type_new_class_creation.
# Partially parsed test_make_pmap_field_type_returns_cached_class. Retrieved 3/13 statements.


def test_case_0():
    var_0 = 'ExistingClass'
    var_1 = ()
    var_2 = {}



# Parsed testcases at query #12
#--------------------------




import pyrsistent._checked_types as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = 'no_serializer'
    var_1 = module_0.CheckedType()
    var_2 = 'json'
    var_3 = var_0
    var_4 = module_1.serialize(var_3, var_2, var_1)
    assert var_4 == 'serialized_json'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_make_seq_field_type_returns_cached_type. Retrieved 1/7 statements.
# Partially parsed test_make_seq_field_type_creates_new_subclass_with_correct_name. Retrieved 1/9 statements.


def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = False



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_sequence_field_creates_checked_type_with_correct_factory. Retrieved 13/43 statements.
# Partially parsed test_sequence_field_initialization_value. Retrieved 6/20 statements.


def test_case_0():
    var_0 = 'pyrsistent._field_common'
    var_1 = None
    var_2 = '_vec'
    var_3 = '_v'
    var_4 = False
    var_5 = 1
    var_6 = 2
    var_7 = [var_5, var_6]
    var_8 = True
    var_9 = [var_8, var_6]
    var_10 = 3
    var_11 = 4
    var_12 = [var_10, var_11]

def test_case_0():
    var_0 = '_v'
    var_1 = None
    var_2 = False
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_check_field_parameters_predicate_false_with_valid_types. Retrieved 1/11 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #16
#--------------------------




import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'data'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = None
    var_5 = (var_3, var_4)
    var_6 = lambda x: var_5
    var_7 = True
    var_8 = 'error_1'
    var_9 = (var_7, var_8)
    var_10 = lambda x: var_9
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
    var_6 = lambda x: var_5
    var_7 = False
    var_8 = 'error_1'
    var_9 = (var_7, var_8)
    var_10 = lambda x: var_9
    var_11 = 'error_2'
    var_12 = (var_7, var_11)
    var_13 = lambda x: var_12
    var_14 = [var_6, var_10, var_13]
    var_15 = module_0.check_global_invariants(var_2, var_14)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'data'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0.check_global_invariants(var_2, var_3)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_check_type_valid_single_type. Retrieved 2/7 statements.
# Partially parsed test_check_type_valid_multiple_types. Retrieved 2/7 statements.
# Partially parsed test_check_type_no_type_restriction. Retrieved 2/7 statements.
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



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 3/12 statements.


def test_case_0():
    var_0 = 'NO_SERIALIZER_SENTINEL'
    var_1 = 'json'
    var_2 = var_0



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 3/11 statements.
# Partially parsed test_serialize_with_standard_serializer_and_checked_type. Retrieved 3/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda fmt, val: f'{fmt}:{val}'
    var_1 = 'json'
    var_2 = 'data'
    var_3 = module_0.serialize(var_0, var_1, var_2)
    assert var_3 == 'json:data'

def test_case_0():
    var_0 = 'PFIELD_NO_SERIALIZER'
    var_1 = 'SENTINEL'
    var_2 = 'xml'

def test_case_0():
    var_0 = lambda fmt, val: f'wrapped_{fmt}_{val}'
    var_1 = 'SENTINEL'
    var_2 = 'json'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_check_type_valid_single_type. Retrieved 2/5 statements.
# Partially parsed test_check_type_valid_multiple_types. Retrieved 3/7 statements.
# Partially parsed test_check_type_no_type_constraint. Retrieved 6/8 statements.
# Partially parsed test_check_type_invalid_type_raises_error. Retrieved 3/7 statements.
# Partially parsed test_check_type_invalid_type_in_tuple_raises_error. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'age'
    var_1 = 10

def test_case_0():
    var_0 = 'data'
    var_1 = 'hello'
    var_2 = 123

def test_case_0():
    var_0 = None
    var_1 = 'anything'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]

def test_case_0():
    var_0 = 'Invalid type for field MockDest.age, was str'
    var_1 = 'age'
    var_2 = 'not_an_int'

def test_case_0():
    var_0 = 'name'
    var_1 = 1.5



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_pmap_field_basic_creation. Retrieved 1/8 statements.
# Partially parsed test_pmap_field_optional. Retrieved 2/5 statements.
# Failed to parse test_pmap_field_with_invariant.
# Failed to parse test_pmap_field_name_generation.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.PMap()

def test_case_0():
    var_0 = True
    var_1 = None



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_check_type_predicate_evaluates_to_false_when_value_matches_field_type. Retrieved 2/16 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 10



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_pfield_constructor_initializes_all_attributes. Retrieved 7/9 statements.


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = True
    var_4 = None
    var_5 = lambda : var_4
    var_6 = lambda x: str(x)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_sequence_field_creates_correct_type_and_functionality. Retrieved 4/15 statements.
# Partially parsed test_sequence_field_with_invariants. Retrieved 3/7 statements.
# Partially parsed test_sequence_field_item_invariant. Retrieved 3/7 statements.


def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = True
    var_3 = None

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = [var_1]

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = [var_1]



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_pfield_constructor_initialization. Retrieved 6/8 statements.


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = True
    var_4 = lambda x: x
    var_5 = lambda x: str(x)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 2/7 statements.
# Partially parsed test_serialize_with_checked_type_and_standard_serializer. Retrieved 4/9 statements.


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
    var_0 = lambda fmt, val: f'{fmt}_{val.val}'
    var_1 = 'NO_SERIALIZER'
    var_2 = 'data'
    var_3 = 'csv'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda fmt, val: val
    var_1 = lambda f, v: v
    var_2 = 'text'
    var_3 = 123
    var_4 = module_0.serialize(var_1, var_2, var_3)
    assert var_4 == 123



