####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_calls_target_validate_with_valid_value. Retrieved 2/9 statements.
# Partially parsed test_validate_returns_target_result_for_non_null_value. Retrieved 2/9 statements.


import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'target_key'
    var_1 = 'MockTarget'
    var_2 = ()
    var_3 = 'validate'
    var_4 = lambda x: x
    var_5 = {var_3: var_4}
    var_6 = type(var_1, var_2, var_5)
    var_7 = var_6()
    var_8 = {var_0: var_7}
    var_9 = True
    var_10 = 'allow_null'
    var_11 = {var_10: var_9}
    var_12 = module_0.Reference(var_0, var_8, **var_11)
    var_13 = None
    var_14 = var_12.validate(var_13)
    assert var_14 is None

import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'target_key'
    var_1 = 'MockTarget'
    var_2 = ()
    var_3 = 'validate'
    var_4 = lambda x: x
    var_5 = {var_3: var_4}
    var_6 = type(var_1, var_2, var_5)
    var_7 = var_6()
    var_8 = {var_0: var_7}
    var_9 = False
    var_10 = 'allow_null'
    var_11 = {var_10: var_9}
    var_12 = module_0.Reference(var_0, var_8, **var_11)
    var_13 = None
    var_14 = var_12.validate(var_13)

def test_case_0():
    var_0 = 'target_key'
    var_1 = 'some_value'

def test_case_0():
    var_0 = 'target_key'
    var_1 = 'data'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_schema_serialize_returns_none_when_obj_is_none. Retrieved 2/9 statements.
# Partially parsed test_schema_serialize_returns_correct_values_for_dict_input. Retrieved 8/16 statements.
# Partially parsed test_schema_serialize_returns_correct_values_for_object_input. Retrieved 6/19 statements.
# Partially parsed test_schema_serialize_skips_missing_keys_in_dict. Retrieved 5/13 statements.
# Partially parsed test_schema_serialize_skips_missing_attributes_in_object. Retrieved 4/16 statements.
# Partially parsed test_schema_serialize_handles_nested_serialization. Retrieved 11/21 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = None

def test_case_0():
    var_0 = 'id'
    var_1 = 'name'
    var_2 = 1
    var_3 = 'test'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'val_1'
    var_6 = 'val_test'
    var_7 = {var_0: var_5, var_1: var_6}

def test_case_0():
    var_0 = 'id'
    var_1 = 'name'
    var_2 = 10
    var_3 = 'hello'
    var_4 = '10'
    var_5 = {var_0: var_4, var_1: var_3}

def test_case_0():
    var_0 = 'present'
    var_1 = 'missing'
    var_2 = True
    var_3 = {var_0: var_2}
    var_4 = {var_0: var_2}

def test_case_0():
    var_0 = 'present'
    var_1 = 'missing'
    var_2 = True
    var_3 = {var_0: var_2}

def test_case_0():
    var_0 = 'data'
    var_1 = 'simple'
    var_2 = 'inner'
    var_3 = lambda v: {var_2: v}
    var_4 = lambda v: v.upper()
    var_5 = 123
    var_6 = 'abc'
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = {var_2: var_5}
    var_9 = 'ABC'
    var_10 = {var_0: var_8, var_1: var_9}



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_schema_validate_success. Retrieved 6/28 statements.
# Partially parsed test_schema_validate_null_error. Retrieved 4/21 statements.
# Partially parsed test_schema_validate_type_error. Retrieved 5/19 statements.
# Partially parsed test_schema_validate_invalid_key. Retrieved 4/28 statements.
# Partially parsed test_schema_validate_required_error. Retrieved 4/31 statements.
# Partially parsed test_schema_validate_default_value_applied. Retrieved 3/17 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 0
    var_3 = 'John'
    var_4 = 30
    var_5 = {var_0: var_3, var_1: var_4}

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 == 'null'

def test_case_0():
    var_0 = {}
    var_1 = 'not'
    var_2 = 'a'
    assert var_2 == 'type'
    var_3 = 'dict'
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = {}
    var_1 = 123
    var_2 = 'value'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'required_field'
    var_1 = 'other_field'
    var_2 = 'val'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'opt'
    var_1 = 'present'
    var_2 = {}



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_schema_validate_success. Retrieved 8/13 statements.
# Partially parsed test_schema_validate_default_value. Retrieved 4/8 statements.
# Partially parsed test_schema_validate_null_error. Retrieved 4/11 statements.
# Partially parsed test_schema_validate_type_error. Retrieved 7/13 statements.
# Partially parsed test_schema_validate_invalid_key_error. Retrieved 6/14 statements.
# Partially parsed test_schema_validate_required_error. Retrieved 7/15 statements.
# Partially parsed test_schema_validate_nested_error_propagation. Retrieved 6/16 statements.
# Partially parsed test_schema_validate_allow_null_success. Retrieved 5/9 statements.
# Partially parsed test_schema_validate_ignores_readonly_fields. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'string'
    var_1 = 'integer'
    var_2 = 10
    var_3 = 'name'
    var_4 = 'age'
    var_5 = 'John'
    var_6 = 25
    var_7 = {var_3: var_5, var_4: var_6}

def test_case_0():
    var_0 = 'integer'
    var_1 = 10
    var_2 = 'count'
    var_3 = {}

def test_case_0():
    var_0 = 'string'
    var_1 = False
    var_2 = 'name'
    var_3 = None
    var_4 = 'null'

def test_case_0():
    var_0 = 'string'
    var_1 = 'name'
    var_2 = 'not'
    var_3 = 'a'
    var_4 = 'dict'
    var_5 = [var_2, var_3, var_4]
    var_6 = str(var_5)
    var_7 = 'type'
    var_8 = bool('type' in var_6)
    assert var_8 is True

def test_case_0():
    var_0 = 'string'
    var_1 = 'name'
    var_2 = 123
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'invalid_key'

def test_case_0():
    var_0 = 'string'
    var_1 = 'name'
    var_2 = 'other'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'required'
    var_6 = [var_1]

def test_case_0():
    var_0 = 'string'
    var_1 = 'child'
    var_2 = 'parent'
    var_3 = 123
    var_4 = {var_1: var_3}
    var_5 = {var_2: var_4}

def test_case_0():
    var_0 = 'string'
    var_1 = True
    var_2 = 'name'
    var_3 = None
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'string'
    var_1 = True
    var_2 = 'id'
    var_3 = '123'
    var_4 = {var_2: var_3}
    var_5 = 'id'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_validate_required_fields_loop_executes. Retrieved 2/25 statements.


def test_case_0():
    var_0 = 'req_field'
    var_1 = {}



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_schema_constructor_initializes_fields_and_required_keys. Retrieved 5/17 statements.
# Partially parsed test_schema_constructor_passes_kwargs_to_super. Retrieved 2/10 statements.


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'required'
    var_3 = 'readonly'
    var_4 = 'defaulted'
    var_5 = 'required'
    var_6 = 'readonly'
    var_7 = 'defaulted'

def test_case_0():
    var_0 = 'test'
    var_1 = 'Test Schema'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_schema_validate_success. Retrieved 7/38 statements.
# Partially parsed test_schema_validate_null_error. Retrieved 4/26 statements.
# Partially parsed test_schema_validate_type_error. Retrieved 3/22 statements.
# Partially parsed test_schema_validate_invalid_key. Retrieved 7/34 statements.
# Partially parsed test_schema_validate_required_field_missing. Retrieved 6/28 statements.


def test_case_0():
    var_0 = 10
    var_1 = 'name'
    var_2 = 'age'
    var_3 = 'John'
    var_4 = 25
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_1: var_3, var_2: var_4}
    var_7 = bool(var_6 == {'name': 'John', 'age': 25})
    assert var_7 is True

def test_case_0():
    var_0 = 'name'
    var_1 = None
    var_2 = 'null'
    var_3 = ValueError(var_2)

def test_case_0():
    var_0 = 'name'
    var_1 = 'not a dict'
    var_2 = 'type'
    assert var_2 == 'type'

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 'val'
    var_3 = {var_1: var_2}
    var_4 = []
    var_5 = 'invalid_key'
    var_6 = len(var_4)
    assert var_6 == 1
    var_7 = var_4[0].code
    assert var_7 == 'invalid_key'
    var_8 = var_4[0].index
    var_9 = bool(var_4[0].index == [1])
    assert var_9 is True

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 20
    var_3 = {var_1: var_2}
    var_4 = []
    var_5 = 'required'
    var_6 = 'err'
    var_7 = bool('err' in var_4)
    assert var_7 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validate_success. Retrieved 6/11 statements.
# Partially parsed test_validate_success_with_defaults. Retrieved 5/10 statements.
# Partially parsed test_validate_error_null. Retrieved 4/12 statements.
# Partially parsed test_validate_error_type. Retrieved 6/14 statements.
# Partially parsed test_validate_error_invalid_key. Retrieved 6/14 statements.
# Partially parsed test_validate_error_required. Retrieved 6/14 statements.
# Partially parsed test_validate_success_allow_null. Retrieved 4/8 statements.
# Partially parsed test_validate_nested_schema_error. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 10
    var_1 = 'name'
    var_2 = 'age'
    var_3 = 'John'
    var_4 = 25
    var_5 = {var_1: var_3, var_2: var_4}

def test_case_0():
    var_0 = 10
    var_1 = 'name'
    var_2 = 'age'
    var_3 = 'John'
    var_4 = {var_1: var_3}

def test_case_0():
    var_0 = False
    var_1 = 'name'
    var_2 = None
    var_3 = 'null'

def test_case_0():
    var_0 = 'name'
    var_1 = 'not'
    var_2 = 'a'
    var_3 = 'dict'
    var_4 = [var_1, var_2, var_3]
    var_5 = 'type'

def test_case_0():
    var_0 = 'name'
    var_1 = 123
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'invalid_key'
    var_5 = any(var_1)
    var_6 = bool(var_5)
    assert var_6 is True

def test_case_0():
    var_0 = 'name'
    var_1 = 'other'
    var_2 = 'data'
    var_3 = {var_1: var_2}
    var_4 = 'required'
    var_5 = [var_0]

def test_case_0():
    var_0 = True
    var_1 = 'name'
    var_2 = None
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'sub'
    var_1 = 'data'
    var_2 = 123
    var_3 = {var_0: var_2}
    var_4 = {var_1: var_3}
    var_5 = [var_0]



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_validate_skips_read_only_fields. Retrieved 7/19 statements.


def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'readonly'
    var_3 = 'writable'
    var_4 = 'old_value'
    var_5 = 'new_value'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'writable'
    var_8 = 'readonly'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_validate_skips_missing_key_with_default. Retrieved 3/18 statements.


def test_case_0():
    var_0 = True
    var_1 = 'test_key'
    var_2 = {}



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_validate_success. Retrieved 3/7 statements.
# Partially parsed test_validate_null_not_allowed. Retrieved 3/8 statements.
# Partially parsed test_validate_null_allowed. Retrieved 3/7 statements.
# Partially parsed test_validate_type_error. Retrieved 5/10 statements.
# Partially parsed test_validate_invalid_key_type. Retrieved 4/9 statements.
# Partially parsed test_validate_required_field_missing. Retrieved 2/7 statements.
# Partially parsed test_validate_with_default_value. Retrieved 2/11 statements.
# Partially parsed test_validate_nested_error_propagation. Retrieved 7/16 statements.


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
    var_0 = 'name'
    var_1 = {}

def test_case_0():
    var_0 = 'child'
    var_1 = 'key'
    var_2 = 'child'
    var_3 = 'key'
    var_4 = 123
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_validate_returns_none_when_value_is_none_and_allow_null_is_true. Retrieved 5/12 statements.


import typesystem.schemas as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = {}
    var_3 = module_0.Schema(var_1, **var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None



# Parsed testcases at query #13
#--------------------------




import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'Required'
    var_1 = module_0.Field(title=var_0)
    var_2 = 'Default'
    var_3 = 'some_value'
    var_4 = module_0.Field(title=var_2, default=var_3)
    var_5 = 'ReadOnly'
    var_6 = True
    var_7 = module_0.Field(title=var_5, read_only=var_6)
    var_8 = 'req'
    var_9 = 'def'
    var_10 = 'ro'
    var_11 = {var_8: var_1, var_9: var_4, var_10: var_7}
    var_12 = {}
    var_13 = module_1.Schema(var_11, **var_12)
    var_14 = 'req'
    var_15 = bool('req' in var_13.required)
    assert var_15 is True
    var_16 = 'def'
    var_17 = bool('def' not in var_13.required)
    assert var_17 is True
    var_18 = 'ro'
    var_19 = bool('ro' not in var_13.required)
    assert var_19 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_validate_skips_missing_key_with_no_default. Retrieved 3/16 statements.


def test_case_0():
    var_0 = 'test_key'
    var_1 = False
    var_2 = {}
    var_3 = 'test_key'



# Parsed testcases at query #15
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.Schema(var_0, **var_3)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_validate_predicate_at_line_37_evaluates_to_true. Retrieved 5/36 statements.


def test_case_0():
    var_0 = 'some_value'
    var_1 = 'key'
    var_2 = 'key'
    var_3 = 'some_value'
    var_4 = {var_2: var_3}



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_validate_skips_missing_key_with_no_default. Retrieved 5/18 statements.


def test_case_0():
    var_0 = 'test_key'
    var_1 = False
    var_2 = 'other_key'
    var_3 = 123
    var_4 = {var_2: var_3}
    var_5 = 'test_key'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_validate_skips_error_when_child_is_valid. Retrieved 3/17 statements.


def test_case_0():
    var_0 = 'test_key'
    var_1 = 'valid_data'
    var_2 = {var_0: var_1}



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_validate_success. Retrieved 8/13 statements.
# Partially parsed test_validate_missing_required_field. Retrieved 3/8 statements.
# Partially parsed test_validate_null_not_allowed. Retrieved 6/11 statements.
# Partially parsed test_validate_null_allowed. Retrieved 5/9 statements.
# Partially parsed test_validate_invalid_type_input. Retrieved 6/11 statements.
# Partially parsed test_validate_invalid_key_type. Retrieved 5/10 statements.
# Partially parsed test_validate_uses_default_value. Retrieved 4/8 statements.
# Partially parsed test_validate_skips_read_only_fields_in_processing. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'string'
    var_1 = 'integer'
    var_2 = 10
    var_3 = 'name'
    var_4 = 'age'
    var_5 = 'John'
    var_6 = 25
    var_7 = {var_3: var_5, var_4: var_6}

def test_case_0():
    var_0 = 'string'
    var_1 = 'name'
    var_2 = {}

def test_case_0():
    var_0 = 'string'
    var_1 = False
    var_2 = 'name'
    var_3 = 'name'
    var_4 = None
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'string'
    var_1 = True
    var_2 = 'name'
    var_3 = None
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'string'
    var_1 = 'name'
    var_2 = 'not'
    var_3 = 'a'
    var_4 = 'dict'
    var_5 = [var_2, var_3, var_4]

def test_case_0():
    var_0 = 'string'
    var_1 = 'name'
    var_2 = 123
    var_3 = 'value'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'integer'
    var_1 = 42
    var_2 = 'age'
    var_3 = {}

def test_case_0():
    var_0 = 'string'
    var_1 = True
    var_2 = 'name'
    var_3 = 'John'
    var_4 = {var_2: var_3}
    var_5 = 'name'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_validate_trigger_error_on_child_field. Retrieved 3/25 statements.


def test_case_0():
    var_0 = 'test_key'
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = 'test_key_error'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_validate_skips_error_when_child_is_valid. Retrieved 4/15 statements.


def test_case_0():
    var_0 = False
    var_1 = 'test_key'
    var_2 = 'valid_value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_validate_success. Retrieved 6/21 statements.
# Partially parsed test_validate_null_error. Retrieved 3/18 statements.
# Partially parsed test_validate_type_error. Retrieved 5/19 statements.
# Partially parsed test_validate_required_field_missing. Retrieved 5/34 statements.
# Partially parsed test_validate_default_value_assignment. Retrieved 4/19 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 0
    var_3 = 'John'
    var_4 = 30
    var_5 = {var_0: var_3, var_1: var_4}

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
    pass

def test_case_0():
    var_0 = 'mock_module'
    var_1 = 'name'
    var_2 = 'age'
    var_3 = 25
    var_4 = {var_2: var_3}
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = {var_0: var_2}



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_validate_success. Retrieved 3/7 statements.
# Partially parsed test_validate_null_error. Retrieved 3/8 statements.
# Partially parsed test_validate_null_success. Retrieved 3/7 statements.
# Partially parsed test_validate_type_error. Retrieved 5/10 statements.
# Partially parsed test_validate_invalid_key_error. Retrieved 4/9 statements.
# Partially parsed test_validate_required_error. Retrieved 2/7 statements.
# Partially parsed test_validate_default_value. Retrieved 3/7 statements.
# Partially parsed test_validate_read_only_ignored. Retrieved 4/8 statements.


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
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = 'name'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_validate_success. Retrieved 4/8 statements.
# Partially parsed test_validate_null_allowed. Retrieved 5/9 statements.
# Partially parsed test_validate_null_not_allowed. Retrieved 6/11 statements.
# Partially parsed test_validate_wrong_type. Retrieved 6/11 statements.
# Partially parsed test_validate_invalid_key_type. Retrieved 5/10 statements.
# Partially parsed test_validate_required_field_missing. Retrieved 3/8 statements.
# Partially parsed test_validate_with_default_value. Retrieved 4/8 statements.
# Partially parsed test_validate_read_only_ignored_in_required. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'integer'
    var_1 = 'age'
    var_2 = 25
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'integer'
    var_1 = True
    var_2 = 'age'
    var_3 = None
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'integer'
    var_1 = False
    var_2 = 'age'
    var_3 = 'age'
    var_4 = None
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 'integer'
    var_1 = 'age'
    var_2 = 'not'
    var_3 = 'a'
    var_4 = 'dict'
    var_5 = [var_2, var_3, var_4]

def test_case_0():
    var_0 = 'integer'
    var_1 = 'age'
    var_2 = 123
    var_3 = 'value'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'integer'
    var_1 = 'age'
    var_2 = {}

def test_case_0():
    var_0 = 'integer'
    var_1 = 10
    var_2 = 'age'
    var_3 = {}

def test_case_0():
    var_0 = 'integer'
    var_1 = True
    var_2 = 'age'
    var_3 = {}
    var_4 = 'age'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_schema_validate_success. Retrieved 3/7 statements.
# Partially parsed test_schema_validate_null_error. Retrieved 3/8 statements.
# Partially parsed test_schema_validate_type_error. Retrieved 5/10 statements.
# Partially parsed test_schema_validate_invalid_key. Retrieved 4/9 statements.
# Partially parsed test_schema_validate_required_field_missing. Retrieved 2/7 statements.
# Partially parsed test_schema_validate_with_default_value. Retrieved 3/7 statements.
# Partially parsed test_schema_validate_readonly_field_ignored. Retrieved 4/8 statements.
# Partially parsed test_schema_validate_nested_error_propagation. Retrieved 7/14 statements.


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
    var_0 = 'default'
    var_1 = 'name'
    var_2 = {}

def test_case_0():
    var_0 = True
    var_1 = 'name'
    var_2 = 'new_value'
    var_3 = {var_1: var_2}
    var_4 = 'name'

def test_case_0():
    var_0 = 'sub'
    var_1 = 'data'
    var_2 = 'data'
    var_3 = 'sub'
    var_4 = 123
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_validate_skips_missing_field_without_default. Retrieved 5/21 statements.


def test_case_0():
    var_0 = False
    var_1 = 'missing_key'
    var_2 = 'other_key'
    var_3 = 'some_value'
    var_4 = {var_2: var_3}
    var_5 = 'missing_key'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_validate_skips_error_when_child_validation_is_successful. Retrieved 4/18 statements.


def test_case_0():
    var_0 = 'valid_content'
    var_1 = None
    var_2 = 'test_key'
    var_3 = {var_2: var_0}



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_schema_validate_trigger_line_32. Retrieved 4/28 statements.


def test_case_0():
    var_0 = 'default'
    var_1 = True
    var_2 = 'test_key'
    var_3 = {}



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_validate_success. Retrieved 6/29 statements.
# Partially parsed test_validate_required_field. Retrieved 9/35 statements.
# Partially parsed test_validate_default_value_injection. Retrieved 2/18 statements.


def test_case_0():
    var_0 = 'default'
    var_1 = 'age'
    var_2 = 'name'
    var_3 = 25
    var_4 = 'John'
    var_5 = {var_1: var_3, var_2: var_4}

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = var_2.validate(var_6)

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = None
    var_4 = var_2.validate(var_3)

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 123
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = 0
    var_8 = e.messages()[var_7]
    var_9 = var_8.code
    assert var_9 == 'invalid_key'

def test_case_0():
    var_0 = 'must_exist'
    var_1 = 'other'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 0
    var_5 = e.messages()[var_4]
    var_6 = var_5.code
    assert var_6 == 'required'
    var_7 = e.messages()[var_4]
    var_8 = var_7.index
    var_9 = bool(var_8 == ['must_exist'])
    assert var_9 is True

def test_case_0():
    var_0 = 'opt'
    var_1 = {}



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_schema_constructor_initializes_fields_and_required_list. Retrieved 5/16 statements.
# Partially parsed test_schema_constructor_inherits_field_properties. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'required_field'
    var_1 = 'optional_field'
    var_2 = 'read_only_field'
    var_3 = 'something'
    var_4 = True
    var_5 = 'required_field'
    var_6 = 'optional_field'
    var_7 = 'read_only_field'

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = var_2.fields
    var_4 = bool(var_2.fields == {})
    assert var_4 is True
    var_5 = var_2.required
    var_6 = bool(var_2.required == [])
    assert var_6 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 'Schema Title'
    var_2 = 'Schema Desc'



# Parsed testcases at query #2
#--------------------------




import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'Name'
    var_2 = module_0.Field(title=var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = var_5.fields
    var_7 = bool(var_5.fields == var_3)
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'required_field'
    var_1 = 'readonly_field'
    var_2 = 'default_field'
    var_3 = 'nullable_field'
    var_4 = 'Required'
    var_5 = module_0.Field(title=var_4)
    var_6 = 'Read Only'
    var_7 = True
    var_8 = module_0.Field(title=var_6, read_only=var_7)
    var_9 = 'Default'
    var_10 = 'some_value'
    var_11 = module_0.Field(title=var_9, default=var_10)
    var_12 = 'Nullable'
    var_13 = module_0.Field(title=var_12, allow_null=var_7)
    var_14 = {var_0: var_5, var_1: var_8, var_2: var_11, var_3: var_13}
    var_15 = {}
    var_16 = module_1.Schema(var_14, **var_15)
    var_17 = 'required_field'
    var_18 = bool('required_field' in var_16.required)
    assert var_18 is True
    var_19 = 'nullable_field'
    var_20 = bool('nullable_field' in var_16.required)
    assert var_20 is True
    var_21 = 'readonly_field'
    var_22 = bool('readonly_field' not in var_16.required)
    assert var_22 is True
    var_23 = 'default_field'
    var_24 = bool('default_field' not in var_16.required)
    assert var_24 is True

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = var_2.fields
    var_4 = bool(var_2.fields == {})
    assert var_4 is True
    var_5 = var_2.required
    var_6 = bool(var_2.required == [])
    assert var_6 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_success. Retrieved 6/26 statements.
# Partially parsed test_validate_missing_required_field. Retrieved 7/30 statements.
# Partially parsed test_validate_invalid_type. Retrieved 6/24 statements.
# Partially parsed test_validate_null_not_allowed. Retrieved 3/24 statements.
# Partially parsed test_validate_invalid_key_type. Retrieved 6/24 statements.
# Partially parsed test_validate_apply_defaults. Retrieved 5/24 statements.


def test_case_0():
    var_0 = 'def'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}

def test_case_0():
    var_0 = 'required_key'
    var_1 = 'other'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = 'required'
    var_5 = 'required_key'
    var_6 = [var_5]

def test_case_0():
    var_0 = {}
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 'type'

def test_case_0():
    var_0 = {}
    var_1 = None
    var_2 = 'null'

def test_case_0():
    var_0 = {}
    var_1 = 123
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'invalid_key'
    var_5 = [var_1]

def test_case_0():
    var_0 = 'b'
    var_1 = 'val'
    var_2 = 'a'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = 'b'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_validate_skips_read_only_fields. Retrieved 6/23 statements.


def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'readonly'
    var_3 = 'writable'
    var_4 = 'new_value'
    var_5 = {var_3: var_4}
    var_6 = 'readonly'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_schema_validate_success. Retrieved 5/34 statements.
# Partially parsed test_schema_validate_null_error. Retrieved 2/17 statements.
# Partially parsed test_schema_validate_type_error. Retrieved 5/18 statements.
# Partially parsed test_schema_validate_invalid_key. Retrieved 4/22 statements.
# Partially parsed test_schema_validate_required_error. Retrieved 4/24 statements.
# Partially parsed test_schema_validate_child_error. Retrieved 4/24 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 20
    var_3 = 'John'
    var_4 = {var_0: var_3}

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
    var_0 = 'required_field'
    var_1 = 'other'
    var_2 = 'val'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'child'
    var_1 = 'child'
    var_2 = 'bad'
    var_3 = {var_1: var_2}
    var_4 = 'child_err'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_validate_raises_error_when_null_and_not_allowed. Retrieved 3/27 statements.
# Partially parsed test_validate_returns_none_when_null_and_allowed. Retrieved 3/21 statements.
# Partially parsed test_validate_calls_target_validate_with_value. Retrieved 3/27 statements.


def test_case_0():
    var_0 = 'target'
    var_1 = False
    var_2 = None

def test_case_0():
    var_0 = 'target'
    var_1 = True
    var_2 = None

def test_case_0():
    var_0 = 'target'
    var_1 = False
    var_2 = 'some_value'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_validate_with_child_error. Retrieved 20/33 statements.


import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'validate_or_error'
    var_3 = 'read_only'
    var_4 = 'has_default'
    var_5 = 'get_default_value'
    var_6 = 'allow_null'
    var_7 = 'validation_error'
    var_8 = 'get_error_text'
    var_9 = None
    var_10 = lambda self, x: (x, var_9)
    var_11 = False
    var_12 = lambda self: var_11
    var_13 = lambda self: var_9
    var_14 = lambda self, e: Exception(e)
    var_15 = 'err'
    var_16 = lambda self, e: var_15
    var_17 = {var_2: var_10, var_3: var_11, var_4: var_12, var_5: var_13, var_6: var_11, var_7: var_14, var_8: var_16}
    var_18 = type(var_0, var_1, var_17)
    var_19 = var_18()
    var_20 = 'name'
    var_21 = {var_20: var_19}
    var_22 = {}
    var_23 = module_0.Schema(var_21, **var_22)
    var_24 = 'test'
    var_25 = {var_20: var_24}
    var_26 = var_23.validate(var_25)
    var_27 = bool(var_26 == {'name': 'test'})
    assert var_27 is True

import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'validate_or_error'
    var_3 = 'read_only'
    var_4 = 'has_default'
    var_5 = 'get_default_value'
    var_6 = 'allow_null'
    var_7 = 'validation_error'
    var_8 = 'get_error_text'
    var_9 = None
    var_10 = lambda self, x: (x, var_9)
    var_11 = False
    var_12 = lambda self: var_11
    var_13 = lambda self: var_9
    var_14 = lambda self, e: ValueError(e)
    var_15 = 'err'
    var_16 = lambda self, e: var_15
    var_17 = {var_2: var_10, var_3: var_11, var_4: var_12, var_5: var_13, var_6: var_11, var_7: var_14, var_8: var_16}
    var_18 = type(var_0, var_1, var_17)
    var_19 = var_18()
    var_20 = 'name'
    var_21 = {var_20: var_19}
    var_22 = {}
    var_23 = module_0.Schema(var_21, **var_22)
    var_24 = None
    var_25 = var_23.validate(var_24)

import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'validate_or_error'
    var_3 = 'read_only'
    var_4 = 'has_default'
    var_5 = 'get_default_value'
    var_6 = 'allow_null'
    var_7 = 'validation_error'
    var_8 = 'get_error_text'
    var_9 = None
    var_10 = lambda self, x: (x, var_9)
    var_11 = False
    var_12 = lambda self: var_11
    var_13 = lambda self: var_9
    var_14 = lambda self, e: ValueError(e)
    var_15 = 'err'
    var_16 = lambda self, e: var_15
    var_17 = {var_2: var_10, var_3: var_11, var_4: var_12, var_5: var_13, var_6: var_11, var_7: var_14, var_8: var_16}
    var_18 = type(var_0, var_1, var_17)
    var_19 = var_18()
    var_20 = 'name'
    var_21 = {var_20: var_19}
    var_22 = {}
    var_23 = module_0.Schema(var_21, **var_22)
    var_24 = 'not'
    var_25 = 'a'
    var_26 = 'dict'
    var_27 = [var_24, var_25, var_26]
    var_28 = var_23.validate(var_27)

import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'validate_or_error'
    var_3 = 'read_only'
    var_4 = 'has_default'
    var_5 = 'get_default_value'
    var_6 = 'allow_null'
    var_7 = 'validation_error'
    var_8 = 'get_error_text'
    var_9 = None
    var_10 = lambda self, x: (x, var_9)
    var_11 = False
    var_12 = lambda self: var_11
    var_13 = lambda self: var_9
    var_14 = lambda self, e: ValueError(e)
    var_15 = 'err'
    var_16 = lambda self, e: var_15
    var_17 = {var_2: var_10, var_3: var_11, var_4: var_12, var_5: var_13, var_6: var_11, var_7: var_14, var_8: var_16}
    var_18 = type(var_0, var_1, var_17)
    var_19 = var_18()
    var_20 = 'name'
    var_21 = {var_20: var_19}
    var_22 = {}
    var_23 = module_0.Schema(var_21, **var_22)
    var_24 = 123
    var_25 = 'value'
    var_26 = {var_24: var_25}
    var_27 = var_23.validate(var_26)

import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'validate_or_error'
    var_3 = 'read_only'
    var_4 = 'has_default'
    var_5 = 'get_default_value'
    var_6 = 'allow_null'
    var_7 = 'validation_error'
    var_8 = 'get_error_text'
    var_9 = None
    var_10 = lambda self, x: (x, var_9)
    var_11 = False
    var_12 = lambda self: var_11
    var_13 = lambda self: var_9
    var_14 = lambda self, e: ValueError(e)
    var_15 = 'err'
    var_16 = lambda self, e: var_15
    var_17 = {var_2: var_10, var_3: var_11, var_4: var_12, var_5: var_13, var_6: var_11, var_7: var_14, var_8: var_16}
    var_18 = type(var_0, var_1, var_17)
    var_19 = var_18()
    var_20 = 'name'
    var_21 = {var_20: var_19}
    var_22 = {}
    var_23 = module_0.Schema(var_21, **var_22)
    var_24 = {}
    var_25 = var_23.validate(var_24)

import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'validate_or_error'
    var_3 = 'read_only'
    var_4 = 'has_default'
    var_5 = 'get_default_value'
    var_6 = 'allow_null'
    var_7 = 'validation_error'
    var_8 = 'get_error_text'
    var_9 = None
    var_10 = lambda self, x: (x, var_9)
    var_11 = False
    var_12 = True
    var_13 = lambda self: var_12
    var_14 = 'default'
    var_15 = lambda self: var_14
    var_16 = lambda self, e: ValueError(e)
    var_17 = 'err'
    var_18 = lambda self, e: var_17
    var_19 = {var_2: var_10, var_3: var_11, var_4: var_13, var_5: var_15, var_6: var_11, var_7: var_16, var_8: var_18}
    var_20 = type(var_0, var_1, var_19)
    var_21 = var_20()
    var_22 = 'name'
    var_23 = {var_22: var_21}
    var_24 = {}
    var_25 = module_0.Schema(var_23, **var_24)
    var_26 = {}
    var_27 = var_25.validate(var_26)
    var_28 = bool(var_27 == {'name': 'default'})
    assert var_28 is True

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'validate_or_error'
    var_3 = 'read_only'
    var_4 = 'has_default'
    var_5 = 'get_default_value'
    var_6 = 'allow_null'
    var_7 = 'validation_error'
    var_8 = 'get_error_text'
    var_9 = None
    var_10 = False
    var_11 = lambda self: var_10
    var_12 = lambda self: var_9
    var_13 = lambda self, e: ValueError(e)
    var_14 = 'err'
    var_15 = lambda self, e: var_14
    var_16 = 'name'
    var_17 = 'name'
    var_18 = 'val'
    var_19 = {var_17: var_18}



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validate_skips_missing_key_with_no_default. Retrieved 5/28 statements.


def test_case_0():
    var_0 = False
    var_1 = 'missing_key'
    var_2 = 'existing_key'
    var_3 = 'some_value'
    var_4 = {var_2: var_3}
    var_5 = 'missing_key'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_validate_returns_none_when_value_is_none_and_allow_null_is_true. Retrieved 3/20 statements.


def test_case_0():
    var_0 = 'target'
    var_1 = True
    var_2 = None



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_validate_raises_error_when_null_and_not_allowed. Retrieved 3/10 statements.
# Partially parsed test_validate_returns_none_when_null_and_allowed. Retrieved 3/7 statements.
# Partially parsed test_validate_calls_target_validate_with_value. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'target_key'
    var_1 = False
    var_2 = None
    var_3 = 'May not be null.'

def test_case_0():
    var_0 = 'target_key'
    var_1 = True
    var_2 = None

def test_case_0():
    var_0 = 'target_key'
    var_1 = False
    var_2 = 'some_input'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_schema_validate_success. Retrieved 6/36 statements.
# Partially parsed test_schema_validate_null_error. Retrieved 2/22 statements.
# Partially parsed test_schema_validate_type_error. Retrieved 2/20 statements.
# Partially parsed test_schema_validate_invalid_key. Retrieved 4/29 statements.
# Partially parsed test_schema_validate_required_error. Retrieved 4/31 statements.
# Partially parsed test_schema_validate_default_values. Retrieved 3/25 statements.


def test_case_0():
    var_0 = 10
    var_1 = 'name'
    var_2 = 'age'
    var_3 = 'John'
    var_4 = 25
    var_5 = {var_1: var_3, var_2: var_4}

def test_case_0():
    var_0 = {}
    var_1 = None
    var_2 = 'null'

def test_case_0():
    var_0 = {}
    var_1 = 'not a dict'
    var_2 = 'type'

def test_case_0():
    var_0 = {}
    var_1 = 123
    var_2 = 'value'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'required_field'
    var_1 = 'other_field'
    var_2 = 'value'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'default_field'
    var_1 = 'fallback'
    var_2 = {}



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_validate_returns_none_when_value_is_none_and_allow_null_is_true. Retrieved 3/23 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = True
    var_2 = None



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_validate_predicate_at_line_37_is_false. Retrieved 3/33 statements.


def test_case_0():
    var_0 = 'test_key'
    var_1 = 'some_value'
    var_2 = {var_0: var_1}



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_validate_predicate_true. Retrieved 5/23 statements.


def test_case_0():
    var_0 = 'valid_data'
    var_1 = None
    var_2 = 'test_key'
    var_3 = 'input_data'
    var_4 = {var_2: var_3}



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_validate_returns_none_when_value_is_none_and_allow_null_is_true. Retrieved 5/10 statements.


import typesystem.schemas as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = {}
    var_3 = module_0.Schema(var_1, **var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_schema_validate_success. Retrieved 8/13 statements.
# Partially parsed test_schema_validate_use_default. Retrieved 4/8 statements.
# Partially parsed test_schema_validate_null_error. Retrieved 4/9 statements.
# Partially parsed test_schema_validate_type_error. Retrieved 6/11 statements.
# Partially parsed test_schema_validate_invalid_key_error. Retrieved 7/14 statements.
# Partially parsed test_schema_validate_required_error. Retrieved 8/16 statements.
# Partially parsed test_schema_validate_allow_null_success. Retrieved 5/9 statements.
# Partially parsed test_schema_validate_skip_read_only. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'integer'
    var_1 = 'string'
    var_2 = 'default'
    var_3 = 'age'
    var_4 = 'name'
    var_5 = 25
    var_6 = 'John'
    var_7 = {var_3: var_5, var_4: var_6}

def test_case_0():
    var_0 = 'string'
    var_1 = 'default'
    var_2 = 'name'
    var_3 = {}

def test_case_0():
    var_0 = 'integer'
    var_1 = False
    var_2 = 'age'
    var_3 = None

def test_case_0():
    var_0 = 'integer'
    var_1 = 'age'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]

def test_case_0():
    var_0 = 'integer'
    var_1 = 'age'
    var_2 = 123
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'invalid_key'
    var_6 = any(var_2)
    var_7 = bool(var_6)
    assert var_7 is True

def test_case_0():
    var_0 = 'integer'
    var_1 = 'age'
    var_2 = 'name'
    var_3 = 'John'
    var_4 = {var_2: var_3}
    var_5 = 'required'
    var_6 = 'age'
    var_7 = [var_6]

def test_case_0():
    var_0 = 'integer'
    var_1 = True
    var_2 = 'age'
    var_3 = None
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'string'
    var_1 = True
    var_2 = 'name'
    var_3 = 'John'
    var_4 = {var_2: var_3}
    var_5 = 'name'



# Parsed testcases at query #17
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.Schema(var_0, **var_3)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_validate_predicate_true. Retrieved 4/21 statements.


def test_case_0():
    var_0 = 'success'
    var_1 = 'test_key'
    var_2 = 'valid_value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_validate_skips_error_assignment_when_child_validation_fails. Retrieved 3/30 statements.
# Partially parsed test_validate_line_37_evaluates_to_false. Retrieved 3/27 statements.


def test_case_0():
    var_0 = 'test_key'
    var_1 = 'some_value'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'key'
    var_1 = 'data'
    var_2 = {var_0: var_1}
    var_3 = 'key_error'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_validate_ensure_error_is_present_at_line_37. Retrieved 3/28 statements.


def test_case_0():
    var_0 = 'test_key'
    var_1 = 'some_value'
    var_2 = {var_0: var_1}
    var_3 = 'test_key_error'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_validate_skips_error_block_when_child_validation_succeeds. Retrieved 3/17 statements.


def test_case_0():
    var_0 = 'test_key'
    var_1 = 'valid_value'
    var_2 = {var_0: var_1}



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_validate_success. Retrieved 3/7 statements.
# Partially parsed test_validate_type_error. Retrieved 6/13 statements.
# Partially parsed test_validate_null_error. Retrieved 2/10 statements.
# Partially parsed test_validate_required_field_missing. Retrieved 2/10 statements.
# Partially parsed test_validate_invalid_key_type. Retrieved 4/12 statements.
# Partially parsed test_validate_default_value_application. Retrieved 3/7 statements.
# Partially parsed test_validate_nested_schema. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'name'
    var_1 = 'not'
    var_2 = 'a'
    var_3 = 'dict'
    var_4 = [var_1, var_2, var_3]
    var_5 = str(var_4)
    var_6 = 'type'
    var_7 = bool('type' in var_5)
    assert var_7 is True

def test_case_0():
    var_0 = 'name'
    var_1 = None
    var_2 = 'null'

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = 'required'

def test_case_0():
    var_0 = 'name'
    var_1 = 123
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'invalid_key'

def test_case_0():
    var_0 = 'default'
    var_1 = 'name'
    var_2 = {}

def test_case_0():
    var_0 = 'age'
    var_1 = 'user'
    var_2 = '25'
    var_3 = {var_0: var_2}
    var_4 = {var_1: var_3}



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_validate_returns_none_when_value_is_none_and_allow_null_is_true. Retrieved 3/7 statements.
# Partially parsed test_validate_raises_error_when_value_is_none_and_allow_null_is_false. Retrieved 3/8 statements.
# Partially parsed test_validate_calls_target_validate_when_value_is_not_none. Retrieved 2/8 statements.


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



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_validate_child_field_no_error. Retrieved 4/18 statements.


def test_case_0():
    var_0 = 'valid_value'
    var_1 = 'test_key'
    var_2 = 'input_value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_validate_success. Retrieved 8/13 statements.
# Partially parsed test_validate_success_with_defaults. Retrieved 7/12 statements.
# Partially parsed test_validate_error_type_not_dict. Retrieved 3/10 statements.
# Partially parsed test_validate_error_null_not_allowed. Retrieved 4/11 statements.
# Partially parsed test_validate_error_null_allowed. Retrieved 4/8 statements.
# Partially parsed test_validate_error_invalid_key_type. Retrieved 6/14 statements.
# Partially parsed test_validate_error_required_field_missing. Retrieved 7/15 statements.
# Partially parsed test_validate_error_nested_validation_failure. Retrieved 6/16 statements.
# Partially parsed test_validate_ignores_read_only_fields_in_logic. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'string'
    var_1 = 'integer'
    var_2 = 10
    var_3 = 'name'
    var_4 = 'age'
    var_5 = 'John'
    var_6 = 25
    var_7 = {var_3: var_5, var_4: var_6}

def test_case_0():
    var_0 = 'string'
    var_1 = 'integer'
    var_2 = 10
    var_3 = 'name'
    var_4 = 'age'
    var_5 = 'John'
    var_6 = {var_3: var_5}

def test_case_0():
    var_0 = 'string'
    var_1 = 'name'
    var_2 = 'not a dict'
    var_3 = 'Must be an object.'

def test_case_0():
    var_0 = 'string'
    var_1 = False
    var_2 = 'name'
    var_3 = None
    var_4 = 'May not be null.'

def test_case_0():
    var_0 = 'string'
    var_1 = True
    var_2 = 'name'
    var_3 = None

def test_case_0():
    var_0 = 'string'
    var_1 = 'name'
    var_2 = 123
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'invalid_key'

def test_case_0():
    var_0 = 'string'
    var_1 = 'name'
    var_2 = 'not_name'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'required'
    var_6 = [var_1]

def test_case_0():
    var_0 = 'integer'
    var_1 = 'sub'
    var_2 = 'parent'
    var_3 = 'not an integer'
    var_4 = {var_1: var_3}
    var_5 = {var_2: var_4}

def test_case_0():
    var_0 = 'string'
    var_1 = True
    var_2 = 'name'
    var_3 = 'John'
    var_4 = {var_2: var_3}
    var_5 = 'name'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_schema_validate_type_error. Retrieved 9/20 statements.
# Partially parsed test_schema_validate_null_error. Retrieved 5/12 statements.
# Partially parsed test_schema_validate_invalid_key. Retrieved 8/18 statements.
# Partially parsed test_schema_validate_required_field. Retrieved 3/23 statements.
# Partially parsed test_schema_validate_success_with_defaults. Retrieved 2/18 statements.
# Partially parsed test_schema_validate_success_with_provided_value. Retrieved 3/17 statements.


import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 'not'
    var_4 = 'a'
    var_5 = 'dict'
    var_6 = [var_3, var_4, var_5]
    var_7 = var_2.validate(var_6)
    var_8 = 'Must be an object.'
    var_9 = any(var_6)
    var_10 = bool(var_9)
    assert var_10 is True

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = None
    var_4 = var_2.validate(var_3)
    var_5 = str(var_4)
    var_6 = 'null'
    var_7 = bool('null' in var_5)
    assert var_7 is True

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 1
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = 'invalid_key'
    var_8 = any(var_6)
    var_9 = bool(var_8)
    assert var_9 is True

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = 'required'

def test_case_0():
    var_0 = 'name'
    var_1 = {}

def test_case_0():
    var_0 = 'name'
    var_1 = 'real_value'
    var_2 = {var_0: var_1}



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_validate_returns_none_when_value_is_none_and_allow_null_is_true. Retrieved 3/21 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = True
    var_2 = None



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_validate_predicate_at_line_37_is_false. Retrieved 4/33 statements.


def test_case_0():
    var_0 = True
    var_1 = 'test_key'
    var_2 = 'some_value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_validate_child_field_success. Retrieved 3/26 statements.


def test_case_0():
    var_0 = 'test_key'
    var_1 = 'valid_value'
    var_2 = {var_0: var_1}



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_validate_skips_adding_to_validated_when_error_exists. Retrieved 4/30 statements.


def test_case_0():
    var_0 = 'test_key'
    var_1 = 'some_value'
    var_2 = {var_0: var_1}
    var_3 = 'caught'
    var_4 = 'test_key'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_validate_returns_none_when_value_is_none_and_allow_null_is_true. Retrieved 3/29 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = True
    var_2 = None



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_validate_returns_none_when_value_is_none_and_allow_null_is_true. Retrieved 3/26 statements.


def test_case_0():
    var_0 = 'some_key'
    var_1 = True
    var_2 = None



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_validate_skips_error_block_when_child_validation_is_successful. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'valid_value'
    var_1 = None
    var_2 = 'test_key'
    var_3 = False
    var_4 = 'some_data'
    var_5 = {var_2: var_4}



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_validate_skips_error_when_child_field_is_valid. Retrieved 5/18 statements.


def test_case_0():
    var_0 = 'valid_data'
    var_1 = None
    var_2 = 'test_key'
    var_3 = 'actual_input'
    var_4 = {var_2: var_3}



