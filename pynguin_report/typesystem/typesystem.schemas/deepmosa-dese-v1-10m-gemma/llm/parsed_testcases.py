####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'test'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = None
    var_4 = var_2.serialize(var_3)
    assert var_4 is None

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'test'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.serialize(var_5)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_schema_constructor_initializes_fields_and_required_list. Retrieved 6/17 statements.
# Partially parsed test_schema_constructor_inherits_field_properties. Retrieved 3/12 statements.


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'val'
    var_3 = 'required_field'
    var_4 = 'read_only_field'
    var_5 = 'defaulted_field'

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)

def test_case_0():
    var_0 = 'key'
    var_1 = 'SchemaTitle'
    var_2 = 'SchemaDesc'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_nested_error_propagation. Retrieved 11/15 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = {}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = {}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = {}
    var_2 = module_1.Schema(var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = var_2.validate(var_6)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = {}
    var_2 = module_1.Schema(var_1)
    var_3 = 123
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Field(read_only=var_0)
    var_2 = 'name'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'default_val'
    var_1 = module_0.Field(default=var_0)
    var_2 = 'name'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'name'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = 'actual_val'
    var_5 = {var_1: var_4}
    var_6 = var_3.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(read_only=var_0)
    var_2 = 'name'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'sub'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = 'data'
    var_5 = {var_4: var_3}
    var_6 = module_1.Schema(var_5)
    var_7 = 'data'
    var_8 = 'not_a_dict'
    var_9 = {var_7: var_8}
    var_10 = var_6.validate(var_9)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_validate_success. Retrieved 6/34 statements.
# Partially parsed test_validate_missing_required_field. Retrieved 4/27 statements.
# Partially parsed test_validate_type_error. Retrieved 5/25 statements.
# Partially parsed test_validate_invalid_key_type. Retrieved 4/27 statements.


def test_case_0():
    var_0 = 'default'
    var_1 = 'age'
    var_2 = 'name'
    var_3 = 25
    var_4 = 'John'
    var_5 = {var_1: var_3, var_2: var_4}

def test_case_0():
    var_0 = 'id'
    var_1 = 'name'
    var_2 = 'John'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'id'
    var_1 = 'not'
    var_2 = 'a'
    var_3 = 'dict'
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = {}
    var_1 = 123
    var_2 = 'value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_validate_returns_none_when_value_is_none_and_allow_null_is_true. Retrieved 5/12 statements.


import typesystem.schemas as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Schema(var_1)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_validate_returns_none_when_value_is_none_and_allow_null_is_true. Retrieved 9/13 statements.
# Partially parsed test_validate_raises_error_when_value_is_none_and_allow_null_is_false. Retrieved 9/14 statements.
# Partially parsed test_validate_calls_target_validate_with_correct_value. Retrieved 2/9 statements.
# Partially parsed test_validate_accesses_correct_definition_key. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'target_key'
    var_1 = 'Mock'
    var_2 = ()
    var_3 = 'validate'
    var_4 = lambda self, x: x
    var_5 = {var_3: var_4}
    var_6 = type(var_1, var_2, var_5)
    var_7 = True
    var_8 = None

def test_case_0():
    var_0 = 'target_key'
    var_1 = 'Mock'
    var_2 = ()
    var_3 = 'validate'
    var_4 = lambda self, x: x
    var_5 = {var_3: var_4}
    var_6 = type(var_1, var_2, var_5)
    var_7 = False
    var_8 = None

def test_case_0():
    var_0 = 'target_key'
    var_1 = 'some_input'

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'test'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_validate_returns_none_when_value_is_none_and_allow_null_is_true. Retrieved 3/21 statements.


def test_case_0():
    var_0 = 'target_key'
    var_1 = True
    var_2 = None



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validate_skips_error_branch_when_child_is_valid. Retrieved 3/20 statements.


def test_case_0():
    var_0 = 'valid_value'
    var_1 = 'test_key'
    var_2 = {var_1: var_0}



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_validate_returns_none_when_value_is_none_and_allow_null_is_true. Retrieved 4/12 statements.


import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_validate_child_schema_with_error. Retrieved 11/43 statements.


def test_case_0():
    var_0 = 'Error'
    var_1 = ()
    var_2 = 'messages'
    var_3 = lambda self, add_prefix: [f'{add_prefix}_err']
    var_4 = {var_2: var_3}
    var_5 = type(var_0, var_1, var_4)
    var_6 = None
    var_7 = 'test_key'
    var_8 = 'test_key'
    var_9 = 'some_value'
    var_10 = {var_8: var_9}



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_validate_returns_none_when_value_is_none_and_allow_null_is_true. Retrieved 3/15 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = True
    var_2 = None



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_validate_returns_none_when_value_is_none_and_allow_null_is_true. Retrieved 9/13 statements.
# Partially parsed test_validate_raises_error_when_value_is_none_and_allow_null_is_false. Retrieved 9/14 statements.
# Partially parsed test_validate_calls_target_validate_with_provided_value. Retrieved 2/9 statements.
# Partially parsed test_validate_retrieves_correct_target_from_definitions. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'target_key'
    var_1 = 'Mock'
    var_2 = ()
    var_3 = 'validate'
    var_4 = lambda self, x: x
    var_5 = {var_3: var_4}
    var_6 = type(var_1, var_2, var_5)
    var_7 = True
    var_8 = None

def test_case_0():
    var_0 = 'target_key'
    var_1 = 'Mock'
    var_2 = ()
    var_3 = 'validate'
    var_4 = lambda self, x: x
    var_5 = {var_3: var_4}
    var_6 = type(var_1, var_2, var_5)
    var_7 = False
    var_8 = None

def test_case_0():
    var_0 = 'target_key'
    var_1 = 'some_value'

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'test'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_validate_schema_child_field_with_error. Retrieved 21/31 statements.


def test_case_0():
    var_0 = 'MockField'
    var_1 = ()
    var_2 = 'read_only'
    var_3 = 'has_default'
    var_4 = 'get_default_value'
    var_5 = 'validate_or_error'
    var_6 = False
    var_7 = lambda : var_6
    var_8 = None
    var_9 = lambda : var_8
    var_10 = 'MockError'
    var_11 = ()
    var_12 = 'messages'
    var_13 = []
    var_14 = lambda self, add_prefix: var_13
    var_15 = {var_12: var_14}
    var_16 = type(var_10, var_11, var_15)
    var_17 = 'test_key'
    var_18 = 'test_key'
    var_19 = 'some_value'
    var_20 = {var_18: var_19}



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_validate_child_field_with_error. Retrieved 4/32 statements.


def test_case_0():
    var_0 = 'test_key'
    var_1 = 'test_key'
    var_2 = 'trigger_error'
    var_3 = {var_1: var_2}



# Parsed testcases at query #15
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_validate_returns_none_when_value_is_none_and_allow_null_is_true. Retrieved 5/10 statements.


import typesystem.schemas as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Schema(var_1)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_schema_validate_valid_dict. Retrieved 3/28 statements.
# Partially parsed test_schema_validate_null_not_allowed. Retrieved 2/25 statements.
# Partially parsed test_schema_validate_type_error. Retrieved 5/28 statements.
# Partially parsed test_schema_validate_required_field_missing. Retrieved 4/37 statements.
# Partially parsed test_schema_validate_default_value_applied. Retrieved 3/28 statements.
# Partially parsed test_schema_validate_invalid_key_type. Retrieved 4/37 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'a'
    var_1 = None

def test_case_0():
    var_0 = 'a'
    var_1 = 'not'
    var_2 = 'a'
    var_3 = 'dict'
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = 'missing'
    var_1 = 'other'
    var_2 = 1
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'age'
    var_1 = 25
    var_2 = {}

def test_case_0():
    var_0 = 'a'
    var_1 = 123
    var_2 = 'value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_validate_raises_error_on_null_when_not_allowed. Retrieved 3/10 statements.
# Partially parsed test_validate_returns_none_on_null_when_allowed. Retrieved 3/7 statements.
# Partially parsed test_validate_calls_target_validate_with_value. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'target_key'
    var_1 = False
    var_2 = None

def test_case_0():
    var_0 = 'target_key'
    var_1 = True
    var_2 = None

def test_case_0():
    var_0 = 'target_key'
    var_1 = False
    var_2 = 'some_data'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_schema_validate_success. Retrieved 3/23 statements.
# Partially parsed test_schema_validate_null_error. Retrieved 2/17 statements.
# Partially parsed test_schema_validate_type_error. Retrieved 5/20 statements.
# Partially parsed test_schema_validate_invalid_key. Retrieved 4/19 statements.
# Partially parsed test_schema_validate_required_field_missing. Retrieved 4/22 statements.
# Partially parsed test_schema_validate_with_defaults. Retrieved 3/22 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 123
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = {}
    var_1 = None

def test_case_0():
    var_0 = {}
    var_1 = 'not'
    var_2 = 'a'
    var_3 = 'dict'
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = {}
    var_1 = 123
    var_2 = 'value'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'required_key'
    var_1 = 'other_key'
    var_2 = 1
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'default_val'
    var_1 = 'a'
    var_2 = {}



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_validate_returns_none_when_value_is_none_and_allow_null_is_true. Retrieved 3/28 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = True
    var_2 = None



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_validate_skips_assignment_when_error_exists. Retrieved 3/22 statements.


def test_case_0():
    var_0 = 'test_key'
    var_1 = 'trigger_error'
    var_2 = {var_0: var_1}



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_validate_raises_error_on_null_when_not_allowed. Retrieved 9/14 statements.
# Partially parsed test_validate_returns_none_when_null_is_allowed. Retrieved 9/13 statements.
# Partially parsed test_validate_calls_target_validate_with_value. Retrieved 3/10 statements.
# Partially parsed test_validate_uses_correct_target_from_definitions. Retrieved 3/13 statements.


def test_case_0():
    var_0 = 'target_key'
    var_1 = 'Mock'
    var_2 = ()
    var_3 = 'validate'
    var_4 = lambda x: x
    var_5 = {var_3: var_4}
    var_6 = type(var_1, var_2, var_5)
    var_7 = False
    var_8 = None

def test_case_0():
    var_0 = 'target_key'
    var_1 = 'Mock'
    var_2 = ()
    var_3 = 'validate'
    var_4 = lambda x: x
    var_5 = {var_3: var_4}
    var_6 = type(var_1, var_2, var_5)
    var_7 = True
    var_8 = None

def test_case_0():
    var_0 = 'target_key'
    var_1 = False
    var_2 = 'some_value'

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'val'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_validate_success. Retrieved 6/11 statements.
# Partially parsed test_validate_default_value. Retrieved 3/7 statements.
# Partially parsed test_validate_null_error. Retrieved 3/8 statements.
# Partially parsed test_validate_type_error. Retrieved 5/10 statements.
# Partially parsed test_validate_invalid_key_error. Retrieved 4/9 statements.
# Partially parsed test_validate_required_error. Retrieved 2/7 statements.
# Partially parsed test_validate_nested_error_propagation. Retrieved 6/16 statements.
# Partially parsed test_validate_allow_null_success. Retrieved 3/7 statements.
# Partially parsed test_validate_readonly_skips_default. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 10
    var_1 = 'name'
    var_2 = 'age'
    var_3 = 'John'
    var_4 = 25
    var_5 = {var_1: var_3, var_2: var_4}

def test_case_0():
    var_0 = 42
    var_1 = 'score'
    var_2 = {}

def test_case_0():
    var_0 = 'name'
    var_1 = False
    var_2 = None

def test_case_0():
    var_0 = 'name'
    var_1 = 'not'
    var_2 = 'a'
    var_3 = 'dict'
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = 'name'
    var_1 = 123
    var_2 = 'value'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'name'
    var_1 = {}

def test_case_0():
    var_0 = 'sub'
    var_1 = 'data'
    var_2 = 123
    var_3 = {var_0: var_2}
    var_4 = {var_1: var_3}
    var_5 = 'type'

def test_case_0():
    var_0 = 'name'
    var_1 = True
    var_2 = None

def test_case_0():
    var_0 = 10
    var_1 = True
    var_2 = 'age'
    var_3 = {}



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_validate_success_with_defaults. Retrieved 5/29 statements.
# Partially parsed test_validate_error_type_not_dict. Retrieved 5/26 statements.
# Partially parsed test_validate_error_null_not_allowed. Retrieved 2/23 statements.
# Partially parsed test_validate_error_required_field. Retrieved 4/34 statements.


def test_case_0():
    var_0 = 'hello'
    var_1 = 'name'
    var_2 = 'age'
    var_3 = 25
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'not'
    var_2 = 'a'
    var_3 = 'dict'
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = 'a'
    var_1 = None

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = 123
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)

def test_case_0():
    var_0 = 'missing_key'
    var_1 = 'other'
    var_2 = 1
    var_3 = {var_1: var_2}



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_validate_child_schema_with_error. Retrieved 3/28 statements.


def test_case_0():
    var_0 = 'child'
    var_1 = 'some_value'
    var_2 = {var_0: var_1}



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_validate_returns_none_when_value_is_none_and_allow_null_is_true. Retrieved 2/13 statements.


def test_case_0():
    var_0 = True
    var_1 = None



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_validate_returns_none_when_value_is_none_and_allow_null_is_true. Retrieved 3/15 statements.


def test_case_0():
    var_0 = 'item'
    var_1 = True
    var_2 = None



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_serialize_returns_none_when_obj_is_none. Retrieved 8/12 statements.
# Partially parsed test_serialize_returns_dict_with_serialized_values_from_mapping. Retrieved 18/23 statements.
# Partially parsed test_serialize_returns_dict_with_serialized_values_from_object. Retrieved 9/17 statements.
# Partially parsed test_serialize_skips_missing_keys. Retrieved 15/20 statements.
# Partially parsed test_serialize_skips_attributes_not_present_on_object. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'serialize'
    var_3 = lambda self, v: v
    var_4 = {var_2: var_3}
    var_5 = type(var_0, var_1, var_4)
    var_6 = 'name'
    var_7 = None

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'serialize'
    var_3 = lambda self, v: str(v)
    var_4 = {var_2: var_3}
    var_5 = type(var_0, var_1, var_4)
    var_6 = ()
    var_7 = lambda self, v: int(v)
    var_8 = {var_2: var_7}
    var_9 = type(var_0, var_6, var_8)
    var_10 = 'name'
    var_11 = 'age'
    var_12 = 'extra'
    var_13 = 'Alice'
    var_14 = 30
    var_15 = 'ignored'
    var_16 = {var_10: var_13, var_11: var_14, var_12: var_15}
    var_17 = {var_10: var_13, var_11: var_14}

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'serialize'
    var_3 = lambda self, v: str(v)
    var_4 = {var_2: var_3}
    var_5 = type(var_0, var_1, var_4)
    var_6 = 'name'
    var_7 = 'Bob'
    var_8 = {var_6: var_7}

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'serialize'
    var_3 = lambda self, v: v
    var_4 = {var_2: var_3}
    var_5 = type(var_0, var_1, var_4)
    var_6 = ()
    var_7 = lambda self, v: v
    var_8 = {var_2: var_7}
    var_9 = type(var_0, var_6, var_8)
    var_10 = 'present'
    var_11 = 'missing'
    var_12 = 'exists'
    var_13 = {var_10: var_12}
    var_14 = {var_10: var_12}

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'serialize'
    var_3 = lambda self, v: v
    var_4 = {var_2: var_3}
    var_5 = type(var_0, var_1, var_4)
    var_6 = 'name'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_serialize_predicate_true_with_dict. Retrieved 3/10 statements.
# Partially parsed test_serialize_predicate_true_with_object. Retrieved 2/13 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'test_value'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'name'
    var_1 = 'test_value'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_success. Retrieved 3/7 statements.
# Partially parsed test_validate_null_allowed. Retrieved 4/8 statements.
# Partially parsed test_validate_null_not_allowed_raises_error. Retrieved 5/10 statements.
# Partially parsed test_validate_wrong_type_raises_error. Retrieved 5/10 statements.
# Partially parsed test_validate_invalid_key_raises_error. Retrieved 4/9 statements.
# Partially parsed test_validate_required_field_missing_raises_error. Retrieved 2/7 statements.
# Partially parsed test_validate_default_value_applied. Retrieved 3/7 statements.
# Partially parsed test_validate_readonly_field_skipped. Retrieved 4/8 statements.
# Partially parsed test_validate_nested_schema_error_propagation. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = True
    var_1 = 'name'
    var_2 = None
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = False
    var_1 = 'name'
    var_2 = 'name'
    var_3 = None
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'name'
    var_1 = 'not'
    var_2 = 'a'
    var_3 = 'dict'
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = 'name'
    var_1 = 123
    var_2 = 'value'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'name'
    var_1 = {}

def test_case_0():
    var_0 = 'default_val'
    var_1 = 'name'
    var_2 = {}

def test_case_0():
    var_0 = True
    var_1 = 'name'
    var_2 = 'should_be_ignored'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'child'
    var_1 = 'child'
    var_2 = 123
    var_3 = {var_1: var_2}



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_validate_iterates_over_fields_at_line_27. Retrieved 27/33 statements.


def test_case_0():
    var_0 = 'MockField'
    var_1 = ()
    var_2 = 'fields'
    var_3 = 'allow_null'
    var_4 = 'validation_error'
    var_5 = 'get_error_text'
    var_6 = 'validate'
    var_7 = 'test_key'
    var_8 = ()
    var_9 = 'read_only'
    var_10 = 'has_default'
    var_11 = 'get_default_value'
    var_12 = 'validate_or_error'
    var_13 = False
    var_14 = lambda : var_13
    var_15 = None
    var_16 = lambda : var_15
    var_17 = lambda x: (x, var_15)
    var_18 = {var_9: var_13, var_10: var_14, var_11: var_16, var_12: var_17}
    var_19 = type(var_0, var_8, var_18)
    var_20 = True
    var_21 = lambda self, x: Exception(x)
    var_22 = lambda self, x: x
    var_23 = 'value'
    var_24 = {var_7: var_23}
    var_25 = lambda self, value: var_24
    var_26 = {var_7: var_23}



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_schema_constructor_initializes_fields_and_required_keys. Retrieved 5/16 statements.
# Partially parsed test_schema_constructor_with_kwargs. Retrieved 4/12 statements.


def test_case_0():
    var_0 = True
    var_1 = 'val'
    var_2 = 'req'
    var_3 = 'ro'
    var_4 = 'def'

def test_case_0():
    var_0 = 'test'
    var_1 = 'Test Schema'
    var_2 = 'Test Description'
    var_3 = True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_serialize_with_dict_obj_ensures_loop_executes. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'test_value'
    var_2 = {var_0: var_1}



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_validate_returns_none_when_value_is_none_and_allow_null_is_true. Retrieved 9/13 statements.
# Partially parsed test_validate_raises_error_when_value_is_none_and_allow_null_is_false. Retrieved 9/14 statements.
# Partially parsed test_validate_calls_target_validate_with_correct_value. Retrieved 2/9 statements.
# Partially parsed test_validate_uses_correct_target_from_definitions. Retrieved 10/20 statements.


def test_case_0():
    var_0 = 'target_key'
    var_1 = 'Mock'
    var_2 = ()
    var_3 = 'validate'
    var_4 = lambda x: x
    var_5 = {var_3: var_4}
    var_6 = type(var_1, var_2, var_5)
    var_7 = True
    var_8 = None

def test_case_0():
    var_0 = 'target_key'
    var_1 = 'Mock'
    var_2 = ()
    var_3 = 'validate'
    var_4 = lambda x: x
    var_5 = {var_3: var_4}
    var_6 = type(var_1, var_2, var_5)
    var_7 = False
    var_8 = None

def test_case_0():
    var_0 = 'target_key'
    var_1 = 'some_value'

def test_case_0():
    var_0 = 'key_a'
    var_1 = 'key_b'
    var_2 = 'Mock'
    var_3 = ()
    var_4 = 'validate'
    var_5 = 'wrong'
    var_6 = lambda x: var_5
    var_7 = {var_4: var_6}
    var_8 = type(var_2, var_3, var_7)
    var_9 = 'val'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_schema_serialize_returns_none_when_obj_is_none. Retrieved 8/12 statements.
# Partially parsed test_schema_serialize_returns_dict_with_serialized_values. Retrieved 8/16 statements.
# Partially parsed test_schema_serialize_works_with_object_attributes. Retrieved 2/13 statements.
# Partially parsed test_schema_serialize_skips_missing_keys_in_dict. Retrieved 4/12 statements.
# Partially parsed test_schema_serialize_skips_missing_attributes_in_object. Retrieved 2/14 statements.


def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'serialize'
    var_3 = lambda self, v: v
    var_4 = {var_2: var_3}
    var_5 = type(var_0, var_1, var_4)
    var_6 = 'name'
    var_7 = None

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'Alice'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'serialized_Alice'
    var_6 = 'serialized_30'
    var_7 = {var_0: var_5, var_1: var_6}

def test_case_0():
    var_0 = 'name'
    var_1 = 'Bob'

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'Alice'
    var_3 = {var_0: var_2}

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_validate_iterates_over_value_keys. Retrieved 6/12 statements.


import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = 'test_key'
    var_3 = 123
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_validate_returns_valid_dict. Retrieved 30/35 statements.
# Partially parsed test_validate_raises_type_error_for_non_dict. Retrieved 23/28 statements.
# Partially parsed test_validate_raises_null_error. Retrieved 20/25 statements.
# Partially parsed test_validate_returns_none_if_allow_null_is_true. Retrieved 18/23 statements.
# Partially parsed test_validate_raises_invalid_key_error. Retrieved 23/52 statements.
# Partially parsed test_validate_raises_required_error. Retrieved 23/40 statements.
# Partially parsed test_validate_applies_defaults. Retrieved 23/27 statements.
# Partially parsed test_validate_skips_read_only_fields. Retrieved 20/24 statements.


import builtins as module_0

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'read_only'
    var_3 = 'has_default'
    var_4 = 'validate_or_error'
    var_5 = 'get_error_text'
    var_6 = 'validation_error'
    var_7 = False
    var_8 = lambda : var_7
    var_9 = None
    var_10 = lambda self, x: (x, var_9)
    var_11 = ''
    var_12 = lambda self, e: var_11
    var_13 = module_0.Exception()
    var_14 = lambda self, e: var_13
    var_15 = {var_2: var_7, var_3: var_8, var_4: var_10, var_5: var_12, var_6: var_14}
    var_16 = type(var_0, var_1, var_15)
    var_17 = ()
    var_18 = lambda : var_7
    var_19 = lambda self, x: (x, var_9)
    var_20 = lambda self, e: var_11
    var_21 = module_0.Exception()
    var_22 = lambda self, e: var_21
    var_23 = {var_2: var_7, var_3: var_18, var_4: var_19, var_5: var_20, var_6: var_22}
    var_24 = type(var_0, var_17, var_23)
    var_25 = 'a'
    var_26 = 'b'
    var_27 = 1
    var_28 = 2
    var_29 = {var_25: var_27, var_26: var_28}

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'read_only'
    var_3 = 'has_default'
    var_4 = 'validate_or_error'
    var_5 = 'get_error_text'
    var_6 = 'validation_error'
    var_7 = False
    var_8 = lambda : var_7
    var_9 = None
    var_10 = lambda self, x: (x, var_9)
    var_11 = ''
    var_12 = lambda self, e: var_11
    var_13 = 'Must be an object.'
    var_14 = ValueError(var_13)
    var_15 = lambda self, e: var_14
    var_16 = {var_2: var_7, var_3: var_8, var_4: var_10, var_5: var_12, var_6: var_15}
    var_17 = type(var_0, var_1, var_16)
    var_18 = 'a'
    var_19 = 1
    var_20 = 2
    var_21 = 3
    var_22 = [var_19, var_20, var_21]

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'read_only'
    var_3 = 'has_default'
    var_4 = 'validate_or_error'
    var_5 = 'get_error_text'
    var_6 = 'validation_error'
    var_7 = False
    var_8 = lambda : var_7
    var_9 = None
    var_10 = lambda self, x: (x, var_9)
    var_11 = ''
    var_12 = lambda self, e: var_11
    var_13 = 'May not be null.'
    var_14 = ValueError(var_13)
    var_15 = lambda self, e: var_14
    var_16 = {var_2: var_7, var_3: var_8, var_4: var_10, var_5: var_12, var_6: var_15}
    var_17 = type(var_0, var_1, var_16)
    var_18 = 'a'
    var_19 = None

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'read_only'
    var_3 = 'has_default'
    var_4 = 'validate_or_error'
    var_5 = 'get_error_text'
    var_6 = 'validation_error'
    var_7 = False
    var_8 = lambda : var_7
    var_9 = None
    var_10 = lambda self, x: (x, var_9)
    var_11 = ''
    var_12 = lambda self, e: var_11
    var_13 = ValueError()
    var_14 = lambda self, e: var_13
    var_15 = {var_2: var_7, var_3: var_8, var_4: var_10, var_5: var_12, var_6: var_14}
    var_16 = type(var_0, var_1, var_15)
    var_17 = 'a'

import builtins as module_0

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'read_only'
    var_3 = 'has_default'
    var_4 = 'validate_or_error'
    var_5 = 'get_error_text'
    var_6 = 'validation_error'
    var_7 = False
    var_8 = lambda : var_7
    var_9 = None
    var_10 = lambda self, x: (x, var_9)
    var_11 = 'All object keys must be strings.'
    var_12 = lambda self, e: var_11
    var_13 = module_0.Exception()
    var_14 = lambda self, e: var_13
    var_15 = {var_2: var_7, var_3: var_8, var_4: var_10, var_5: var_12, var_6: var_14}
    var_16 = type(var_0, var_1, var_15)
    var_17 = 'a'
    var_18 = 'Message'
    var_19 = 'ValidationError'
    var_20 = 123
    var_21 = 'value'
    var_22 = {var_20: var_21}

import builtins as module_0

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'read_only'
    var_3 = 'has_default'
    var_4 = 'validate_or_error'
    var_5 = 'get_error_text'
    var_6 = 'validation_error'
    var_7 = False
    var_8 = lambda : var_7
    var_9 = None
    var_10 = lambda self, x: (x, var_9)
    var_11 = 'This field is required.'
    var_12 = lambda self, e: var_11
    var_13 = module_0.Exception()
    var_14 = lambda self, e: var_13
    var_15 = {var_2: var_7, var_3: var_8, var_4: var_10, var_5: var_12, var_6: var_14}
    var_16 = type(var_0, var_1, var_15)
    var_17 = 'Message'
    var_18 = 'ValidationError'
    var_19 = 'required_key'
    var_20 = 'other_key'
    var_21 = 1
    var_22 = {var_20: var_21}

import builtins as module_0

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'read_only'
    var_3 = 'has_default'
    var_4 = 'get_default_value'
    var_5 = 'validate_or_error'
    var_6 = 'get_error_text'
    var_7 = 'validation_error'
    var_8 = False
    var_9 = True
    var_10 = lambda self: var_9
    var_11 = 'default'
    var_12 = lambda self: var_11
    var_13 = None
    var_14 = lambda self, x: (x, var_13)
    var_15 = ''
    var_16 = lambda self, e: var_15
    var_17 = module_0.Exception()
    var_18 = lambda self, e: var_17
    var_19 = {var_2: var_8, var_3: var_10, var_4: var_12, var_5: var_14, var_6: var_16, var_7: var_18}
    var_20 = type(var_0, var_1, var_19)
    var_21 = 'a'
    var_22 = {}

import builtins as module_0

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'read_only'
    var_3 = 'has_default'
    var_4 = 'validate_or_error'
    var_5 = 'get_error_text'
    var_6 = 'validation_error'
    var_7 = True
    var_8 = False
    var_9 = lambda self: var_8
    var_10 = None
    var_11 = lambda self, x: (x, var_10)
    var_12 = ''
    var_13 = lambda self, e: var_12
    var_14 = module_0.Exception()
    var_15 = lambda self, e: var_14
    var_16 = {var_2: var_7, var_3: var_9, var_4: var_11, var_5: var_13, var_6: var_15}
    var_17 = type(var_0, var_1, var_16)
    var_18 = 'a'
    var_19 = {}



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_validate_key_exists_in_value. Retrieved 3/14 statements.


def test_case_0():
    var_0 = 'test_key'
    var_1 = 'some_value'
    var_2 = {var_0: var_1}



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_schema_validate_success. Retrieved 7/35 statements.
# Partially parsed test_schema_validate_null_error. Retrieved 3/21 statements.
# Partially parsed test_schema_validate_type_error. Retrieved 6/21 statements.
# Partially parsed test_schema_validate_invalid_key. Retrieved 2/29 statements.
# Partially parsed test_schema_validate_required_error. Retrieved 2/27 statements.
# Partially parsed test_schema_validate_default_value_application. Retrieved 2/21 statements.


def test_case_0():
    var_0 = 'val'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = False
    var_4 = 'new'
    var_5 = 123
    var_6 = {var_1: var_4, var_2: var_5}

def test_case_0():
    var_0 = {}
    var_1 = False
    var_2 = None

def test_case_0():
    var_0 = {}
    var_1 = False
    var_2 = 'not'
    var_3 = 'a'
    var_4 = 'dict'
    var_5 = [var_2, var_3, var_4]

def test_case_0():
    var_0 = {}
    var_1 = False

def test_case_0():
    var_0 = 'req'
    var_1 = {}

def test_case_0():
    var_0 = 'opt'
    var_1 = {}



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_validate_success. Retrieved 3/7 statements.
# Partially parsed test_validate_null_error. Retrieved 3/8 statements.
# Partially parsed test_validate_null_success. Retrieved 3/7 statements.
# Partially parsed test_validate_type_error. Retrieved 5/10 statements.
# Partially parsed test_validate_invalid_key_error. Retrieved 4/9 statements.
# Partially parsed test_validate_required_error. Retrieved 2/7 statements.
# Partially parsed test_validate_with_default_value. Retrieved 3/7 statements.
# Partially parsed test_validate_read_only_skip. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'name'
    var_1 = False
    var_2 = None

def test_case_0():
    var_0 = 'name'
    var_1 = True
    var_2 = None

def test_case_0():
    var_0 = 'name'
    var_1 = 'not'
    var_2 = 'a'
    var_3 = 'dict'
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = 'name'
    var_1 = 123
    var_2 = 'value'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'name'
    var_1 = {}

def test_case_0():
    var_0 = 'default_val'
    var_1 = 'name'
    var_2 = {}

def test_case_0():
    var_0 = True
    var_1 = 'name'
    var_2 = 'original'
    var_3 = {var_1: var_2}



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_validate_returns_none_when_value_is_none_and_allow_null_is_true. Retrieved 3/7 statements.
# Partially parsed test_validate_raises_error_when_value_is_none_and_allow_null_is_false. Retrieved 3/10 statements.
# Partially parsed test_validate_calls_target_validate_with_correct_value. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'target_key'
    var_1 = True
    var_2 = None

def test_case_0():
    var_0 = 'target_key'
    var_1 = False
    var_2 = None

def test_case_0():
    var_0 = 'target_key'
    var_1 = 'some_value'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_validate_returns_none_when_value_is_none_and_allow_null_is_true. Retrieved 4/10 statements.


import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_schema_validate_null_allowed. Retrieved 2/11 statements.
# Partially parsed test_schema_validate_null_not_allowed. Retrieved 3/13 statements.
# Partially parsed test_schema_validate_invalid_type. Retrieved 6/16 statements.
# Partially parsed test_schema_validate_invalid_key_type. Retrieved 7/17 statements.
# Partially parsed test_schema_validate_required_field_missing. Retrieved 5/14 statements.
# Partially parsed test_schema_validate_success_with_defaults. Retrieved 4/11 statements.
# Partially parsed test_schema_validate_success_with_provided_value. Retrieved 5/12 statements.
# Partially parsed test_schema_validate_child_error_propagation. Retrieved 8/21 statements.


def test_case_0():
    var_0 = None
    var_1 = 'test'

def test_case_0():
    var_0 = 'test'
    var_1 = None
    var_2 = 'null'

def test_case_0():
    var_0 = 'test'
    var_1 = 'not'
    var_2 = 'a'
    var_3 = 'dict'
    var_4 = [var_1, var_2, var_3]
    var_5 = 'type'

def test_case_0():
    var_0 = 'val'
    var_1 = None
    var_2 = 'a'
    var_3 = 'Invalid key error'
    var_4 = 123
    var_5 = 'value'
    var_6 = {var_4: var_5}

def test_case_0():
    var_0 = 'required_key'
    var_1 = 'Required error'
    var_2 = 'other_key'
    var_3 = 'val'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'optional_key'
    var_1 = 'other_key'
    var_2 = 'val'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'valid_value'
    var_1 = None
    var_2 = 'key'
    var_3 = 'actual_value'
    var_4 = {var_2: var_3}

import typesystem.base as module_0

def test_case_0():
    var_0 = 'Child error'
    var_1 = 'child_err'
    var_2 = []
    var_3 = module_0.Message(text=var_0, code=var_1, index=var_2)
    var_4 = None
    var_5 = 'key'
    var_6 = 'bad_value'
    var_7 = {var_5: var_6}



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_validate_child_schema_with_error. Retrieved 3/46 statements.


def test_case_0():
    var_0 = 'test_key'
    var_1 = 'some_value'
    var_2 = {var_0: var_1}



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_validate_returns_none_when_allow_null_is_true. Retrieved 4/14 statements.
# Partially parsed test_validate_raises_error_when_value_is_none_and_not_allow_null. Retrieved 4/15 statements.
# Partially parsed test_validate_raises_error_on_missing_required_field. Retrieved 2/22 statements.
# Partially parsed test_validate_applies_default_values. Retrieved 4/24 statements.
# Partially parsed test_validate_skips_read_only_fields. Retrieved 3/19 statements.
# Partially parsed test_validate_propagates_child_errors. Retrieved 4/20 statements.


import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = None
    var_3 = var_1.validate(var_2)

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = 'not'
    var_3 = 'a'
    var_4 = 'dict'
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.validate(var_5)

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = 123
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)

def test_case_0():
    var_0 = 'username'
    var_1 = {}

def test_case_0():
    var_0 = 'age'
    var_1 = True
    var_2 = 25
    var_3 = {}

def test_case_0():
    var_0 = 'id'
    var_1 = True
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'child'
    var_1 = 'child'
    var_2 = 'some_value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_validate_skips_missing_key_with_default. Retrieved 3/19 statements.


def test_case_0():
    var_0 = True
    var_1 = 'test_key'
    var_2 = {}



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_validate_returns_none_when_value_is_none_and_allow_null_is_true. Retrieved 3/28 statements.


def test_case_0():
    var_0 = 'test_key'
    var_1 = True
    var_2 = None



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_validate_raises_error_on_null_when_not_allowed. Retrieved 2/9 statements.
# Partially parsed test_validate_returns_none_when_null_is_allowed. Retrieved 2/7 statements.
# Partially parsed test_validate_calls_target_validate_with_value. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'target_key'
    var_1 = None

def test_case_0():
    var_0 = 'target_key'
    var_1 = None

def test_case_0():
    var_0 = 'target_key'
    var_1 = 'some_input'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_validate_schema_skips_missing_key_with_default. Retrieved 4/16 statements.


def test_case_0():
    var_0 = 'test_key'
    var_1 = True
    var_2 = 'default'
    var_3 = {}



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_schema_validate_success. Retrieved 6/33 statements.
# Partially parsed test_schema_validate_default_values. Retrieved 5/25 statements.
# Partially parsed test_schema_validate_type_error. Retrieved 5/31 statements.
# Partially parsed test_schema_validate_required_error. Retrieved 4/30 statements.
# Partially parsed test_schema_validate_invalid_key. Retrieved 4/30 statements.


def test_case_0():
    var_0 = 'default'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 'hello'
    var_5 = {var_1: var_3, var_2: var_4}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'def'
    var_3 = 1
    var_4 = {var_0: var_3}

def test_case_0():
    var_0 = {}
    var_1 = 'not'
    var_2 = 'a'
    var_3 = 'dict'
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = 'required_field'
    var_1 = 'other_field'
    var_2 = 1
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = {}
    var_1 = 123
    var_2 = 'value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_validate_skips_missing_key_when_no_default_exists. Retrieved 5/30 statements.


def test_case_0():
    var_0 = False
    var_1 = 'missing_key'
    var_2 = 'other_key'
    var_3 = 'some_value'
    var_4 = {var_2: var_3}



# Parsed testcases at query #25
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_validate_returns_none_when_value_is_none_and_allow_null_is_true. Retrieved 3/22 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = True
    var_2 = None



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_schema_validate_valid_object. Retrieved 3/24 statements.
# Partially parsed test_schema_validate_null_error. Retrieved 2/19 statements.
# Partially parsed test_schema_validate_type_error. Retrieved 5/24 statements.
# Partially parsed test_schema_validate_invalid_key. Retrieved 4/21 statements.
# Partially parsed test_schema_validate_required_field_missing. Retrieved 4/23 statements.
# Partially parsed test_schema_validate_with_defaults. Retrieved 3/28 statements.


def test_case_0():
    var_0 = 'age'
    var_1 = 25
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = {}
    var_1 = None

def test_case_0():
    var_0 = {}
    var_1 = 'not'
    var_2 = 'a'
    var_3 = 'dict'
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = {}
    var_1 = 123
    var_2 = 'value'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'must_exist'
    var_1 = 'other'
    var_2 = 1
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'count'
    var_1 = 5
    var_2 = {}



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_validate_returns_none_when_value_is_none_and_allow_null_is_true. Retrieved 9/13 statements.


def test_case_0():
    var_0 = 'target_key'
    var_1 = 'Mock'
    var_2 = ()
    var_3 = 'validate'
    var_4 = lambda x: x
    var_5 = {var_3: var_4}
    var_6 = type(var_1, var_2, var_5)
    var_7 = True
    var_8 = None



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_validate_returns_none_when_value_is_none_and_allow_null_is_true. Retrieved 5/12 statements.


import typesystem.schemas as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Schema(var_1)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_validate_returns_none_when_value_is_none_and_allow_null_is_true. Retrieved 5/12 statements.


import typesystem.schemas as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Schema(var_1)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_validate_returns_none_when_value_is_none_and_allow_null_is_true. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'target_key'
    var_1 = True
    var_2 = None



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_validate_skips_missing_key_with_no_default. Retrieved 5/32 statements.


def test_case_0():
    var_0 = False
    var_1 = 'missing_key'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_validate_returns_none_when_value_is_none_and_allow_null_is_true. Retrieved 3/15 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = True
    var_2 = None



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_validate_allows_null_when_configured. Retrieved 3/12 statements.


def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = None



