####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'data'
    var_1 = 123
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = None
    var_5 = (var_3, var_4)
    var_6 = lambda x: var_5
    var_7 = 'No error'
    var_8 = (var_3, var_7)
    var_9 = lambda x: var_8
    var_10 = [var_6, var_9]
    var_11 = module_0.check_global_invariants(var_2, var_10)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'data'
    var_1 = 123
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = None
    var_5 = (var_3, var_4)
    var_6 = lambda x: var_5
    var_7 = False
    var_8 = 'ERR_001'
    var_9 = (var_7, var_8)
    var_10 = lambda x: var_9
    var_11 = [var_6, var_10]
    var_12 = module_0.check_global_invariants(var_2, var_11)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'data'
    var_1 = 123
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = 'ERR_001'
    var_5 = (var_3, var_4)
    var_6 = lambda x: var_5
    var_7 = True
    var_8 = None
    var_9 = (var_7, var_8)
    var_10 = lambda x: var_9
    var_11 = 'ERR_002'
    var_12 = (var_3, var_11)
    var_13 = lambda x: var_12
    var_14 = [var_6, var_10, var_13]
    var_15 = module_0.check_global_invariants(var_2, var_14)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_check_field_parameters_valid. Retrieved 6/16 statements.
# Partially parsed test_check_field_parameters_invalid_type_element. Retrieved 1/11 statements.


def test_case_0():
    var_0 = 10
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = None
    var_4 = lambda : var_3
    var_5 = lambda x: x

def test_case_0():
    var_0 = 123



# Parsed testcases at query #3
#--------------------------

# Failed to parse test_types_to_names_simple_types.
# Failed to parse test_types_to_names_single_type.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ()
    var_1 = module_0._types_to_names(var_0)
    assert var_1 == ''

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'builtins.str'
    var_1 = 'builtins.int'
    var_2 = (var_0, var_1)
    var_3 = module_0._types_to_names(var_2)
    assert var_3 == 'StrInt'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_pmap_field_basic_creation. Retrieved 2/6 statements.
# Partially parsed test_pmap_field_optional_creation. Retrieved 3/6 statements.
# Partially parsed test_pmap_field_with_invariant. Retrieved 1/6 statements.
# Failed to parse test_pmap_field_type_naming.
# Failed to parse test_pmap_field_multiple_instances_distinct_classes.


import pyrsistent._pmap as module_0
import builtins as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.PMap(*var_0, **var_1)
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_1.type(*var_3, **var_4)

import builtins as module_0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.type(*var_2, **var_3)

def test_case_0():
    var_0 = {}



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_set_fields_merges_dicts_and_removes_pfields. Retrieved 2/23 statements.


def test_case_0():
    var_0 = 'field_to_move'
    var_1 = 'merged_attr'
    var_2 = 'field_to_move'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_sequence_field_creation_with_optional_true. Retrieved 3/5 statements.
# Partially parsed test_sequence_field_creation_with_optional_false. Retrieved 2/7 statements.


def test_case_0():
    var_0 = True
    var_1 = set()
    var_2 = set()

def test_case_0():
    var_0 = False
    var_1 = []



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_field_valid_single_type. Retrieved 5/6 statements.
# Partially parsed test_field_valid_multiple_types. Retrieved 5/7 statements.
# Partially parsed test_field_invalid_initial_type_raises_error. Retrieved 5/7 statements.
# Partially parsed test_field_non_callable_invariant_raises_error. Retrieved 5/7 statements.
# Partially parsed test_field_non_callable_factory_raises_error. Retrieved 5/7 statements.
# Partially parsed test_field_non_callable_serializer_raises_error. Retrieved 5/7 statements.
# Partially parsed test_field_wrapped_invariant. Retrieved 5/9 statements.


def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = 0
    var_3 = lambda x: x
    var_4 = lambda x: x

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = 0
    var_3 = lambda x: x
    var_4 = lambda x: x

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = 0
    var_4 = lambda x: x
    var_5 = lambda x: x
    var_6 = module_0.field(var_0, var_2, var_3, var_1, var_4, var_5)
    var_7 = var_6.type
    var_8 = bool(var_6.type == {'int'})
    assert var_8 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 123
    var_1 = [var_0]
    var_2 = True
    var_3 = lambda x: var_2
    var_4 = 0
    var_5 = lambda x: x
    var_6 = lambda x: x
    var_7 = module_0.field(var_1, var_3, var_4, var_2, var_5, var_6)
    var_8 = 'Type parameter expected'

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = 10
    var_3 = lambda x: x
    var_4 = lambda x: x
    var_5 = 'Initial has invalid type'

def test_case_0():
    var_0 = 'not_callable'
    var_1 = 0
    var_2 = True
    var_3 = lambda x: x
    var_4 = lambda x: x
    var_5 = 'Invariant must be callable'

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = 0
    var_3 = 'not_callable'
    var_4 = lambda x: x
    var_5 = 'Factory must be callable'

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = 0
    var_3 = lambda x: x
    var_4 = 'not_callable'
    var_5 = 'Serializer must be callable'

def test_case_0():
    var_0 = 0
    var_1 = True
    var_2 = lambda x: x
    var_3 = lambda x: x
    var_4 = 5

def test_case_0():
    pass



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 3/10 statements.
# Partially parsed test_serialize_with_checked_type_and_standard_serializer. Retrieved 4/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda f, v: f'{f}:{v}'
    var_1 = 'json'
    var_2 = 'data'
    var_3 = module_0.serialize(var_0, var_1, var_2)
    assert var_3 == 'json:data'

def test_case_0():
    var_0 = 'NO_SERIALIZER'
    var_1 = 'test'
    var_2 = 'xml'

def test_case_0():
    var_0 = lambda f, v: f'wrapped_{f}_{v.val}'
    var_1 = 'NO_SERIALIZER'
    var_2 = 'test'
    var_3 = 'json'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda f, v: f'{v}_{f}'
    var_1 = 'simple_value'
    var_2 = 'csv'
    var_3 = module_0.serialize(var_0, var_2, var_1)
    assert var_3 == 'simple_value_csv'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_pmap_field_basic_creation. Retrieved 1/7 statements.
# Partially parsed test_pmap_field_optional. Retrieved 6/11 statements.
# Failed to parse test_pmap_field_with_invariant.


def test_case_0():
    var_0 = dict()

import builtins as module_0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.type(*var_2, **var_3)
    var_5 = None
    var_6 = 'a'
    var_7 = {var_6: var_0}



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_check_type_valid_single_type. Retrieved 2/7 statements.
# Partially parsed test_check_type_valid_multiple_types. Retrieved 2/7 statements.
# Partially parsed test_check_type_no_type_restriction. Retrieved 2/7 statements.
# Partially parsed test_check_type_invalid_type_raises_error. Retrieved 2/10 statements.


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



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_make_seq_field_type_returns_cached_type. Retrieved 4/13 statements.
# Partially parsed test_make_seq_field_type_creates_new_subclass_with_correct_attributes. Retrieved 2/14 statements.


def test_case_0():
    var_0 = 0
    var_1 = 1
    var_2 = True
    var_3 = True

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'String'
    var_3 = 'Int'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_check_type_valid_type_passes. Retrieved 4/18 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 10
    var_2 = 'not_an_int'
    var_3 = 'string'



