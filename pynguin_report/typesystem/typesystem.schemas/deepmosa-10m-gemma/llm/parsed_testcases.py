####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_schema_validate_success_with_defaults. Retrieved 5/28 statements.
# Partially parsed test_schema_validate_raises_type_error. Retrieved 10/27 statements.
# Partially parsed test_schema_validate_raises_null_error. Retrieved 6/23 statements.
# Partially parsed test_schema_validate_raises_required_error. Retrieved 7/27 statements.
# Partially parsed test_schema_validate_raises_invalid_key_error. Retrieved 10/31 statements.
# Partially parsed test_schema_validate_skips_read_only_fields. Retrieved 4/20 statements.


def test_case_0():
    var_0 = 'default_val'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 123
    var_4 = {var_2: var_3}

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
    var_8 = None
    var_9 = 'type'
    var_10 = any(var_7)
    var_11 = bool(var_10)
    assert var_11 is True

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = None
    var_4 = var_2.validate(var_3)
    var_5 = None
    var_6 = 'null'

def test_case_0():
    var_0 = 'required_key'
    var_1 = 'other_key'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = None
    var_5 = 'required'
    var_6 = [var_1]

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 123
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = None
    var_8 = 'invalid_key'
    var_9 = 123
    var_10 = [var_9]

def test_case_0():
    var_0 = 'readonly'
    var_1 = True
    var_2 = 'new_value'
    var_3 = {var_0: var_2}
    var_4 = 'readonly'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_schema_validate_key_missing_with_default_value_triggers_line_32. Retrieved 4/28 statements.


def test_case_0():
    var_0 = 'default_val'
    var_1 = True
    var_2 = 'test_key'
    var_3 = {}



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_skips_key_if_not_in_value_and_has_no_default. Retrieved 5/15 statements.


def test_case_0():
    var_0 = False
    var_1 = 'missing_key'
    var_2 = 'existing_key'
    var_3 = 123
    var_4 = {var_2: var_3}
    var_5 = 'missing_key'



# Parsed testcases at query #4
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



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_schema_constructor_initializes_fields_and_required_correctly. Retrieved 9/24 statements.
# Partially parsed test_schema_constructor_with_no_required_fields. Retrieved 2/11 statements.


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'val'
    var_3 = 'required_key'
    var_4 = 'readonly_key'
    var_5 = 'default_key'
    var_6 = 'default_readonly_key'
    var_7 = 'Test Schema'
    var_8 = 'Test Desc'
    var_9 = 'required_key'
    var_10 = 'readonly_key'
    var_11 = 'default_key'
    var_12 = 'default_readonly_key'

def test_case_0():
    var_0 = 'opt'
    var_1 = True



# Parsed testcases at query #6
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
    var_1 = 'default_field'
    var_2 = 'read_only_field'
    var_3 = 'Required'
    var_4 = module_0.Field(title=var_3)
    var_5 = 'Default'
    var_6 = 'something'
    var_7 = module_0.Field(title=var_5, default=var_6)
    var_8 = 'Read Only'
    var_9 = True
    var_10 = module_0.Field(title=var_8, read_only=var_9)
    var_11 = {var_0: var_4, var_1: var_7, var_2: var_10}
    var_12 = {}
    var_13 = module_1.Schema(var_11, **var_12)
    var_14 = var_13.required
    var_15 = bool(var_13.required == ['required_field'])
    assert var_15 is True

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



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_validate_skips_error_when_child_is_valid. Retrieved 5/23 statements.


def test_case_0():
    var_0 = 'valid_data'
    var_1 = None
    var_2 = 'test_key'
    var_3 = 'input_data'
    var_4 = {var_2: var_3}



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validate_child_schema_with_error. Retrieved 5/35 statements.


def test_case_0():
    var_0 = 'invalid value'
    var_1 = 'test_key'
    var_2 = 'test_key'
    var_3 = 'some_value'
    var_4 = {var_2: var_3}
    var_5 = 'test_key: invalid value'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_schema_validate_skips_read_only_fields. Retrieved 7/25 statements.


def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'readonly_key'
    var_3 = 'writable_key'
    var_4 = 'old_value'
    var_5 = 'new_value'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'writable_key'
    var_8 = 'readonly_key'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_validate_raises_error_when_value_is_none_and_allow_null_is_false. Retrieved 13/16 statements.
# Partially parsed test_validate_calls_target_validate_with_correct_value. Retrieved 2/9 statements.
# Partially parsed test_validate_uses_correct_definition_lookup. Retrieved 2/10 statements.


import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'target_key'
    var_1 = 'Mock'
    var_2 = ()
    var_3 = 'validate'
    var_4 = lambda self, x: x
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
    var_15 = 'May not be null.'

import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'target_key'
    var_1 = 'Mock'
    var_2 = ()
    var_3 = 'validate'
    var_4 = lambda self, x: x
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

def test_case_0():
    var_0 = 'target_key'
    var_1 = 'some_value'

def test_case_0():
    var_0 = 'key_a'
    var_1 = 'key_b'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_schema_validate_success. Retrieved 3/7 statements.
# Partially parsed test_schema_validate_null_error. Retrieved 2/7 statements.
# Partially parsed test_schema_validate_type_error. Retrieved 5/10 statements.
# Partially parsed test_schema_validate_invalid_key_error. Retrieved 4/9 statements.
# Partially parsed test_schema_validate_required_error. Retrieved 2/7 statements.
# Partially parsed test_schema_validate_with_defaults. Retrieved 3/7 statements.
# Partially parsed test_schema_validate_skips_read_only_on_missing. Retrieved 3/7 statements.
# Partially parsed test_schema_validate_nested_error_propagation. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'name'
    var_1 = None

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
    var_2 = {}
    var_3 = 'name'

def test_case_0():
    var_0 = 'sub'
    var_1 = 'parent'
    var_2 = 'parent'
    var_3 = 'sub'
    var_4 = 123
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_validate_child_schema_success. Retrieved 3/26 statements.


def test_case_0():
    var_0 = 'test_key'
    var_1 = 'valid_value'
    var_2 = {var_0: var_1}



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_schema_validate_skips_read_only_fields. Retrieved 6/24 statements.


def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'readonly'
    var_3 = 'writable'
    var_4 = 'correct_value'
    var_5 = {var_3: var_4}
    var_6 = 'writable'
    var_7 = 'readonly'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_validate_success_with_defaults. Retrieved 7/60 statements.
# Partially parsed test_validate_raises_type_error. Retrieved 6/30 statements.
# Partially parsed test_validate_raises_null_error. Retrieved 2/27 statements.
# Partially parsed test_validate_raises_required_error. Retrieved 4/31 statements.
# Partially parsed test_validate_invalid_key_type. Retrieved 4/32 statements.


def test_case_0():
    var_0 = 'default_val'
    var_1 = True
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'actual_val'
    var_5 = 'ignored_val'
    var_6 = {var_2: var_4, var_3: var_5}

def test_case_0():
    var_0 = 'a'
    var_1 = 'not'
    var_2 = 'a'
    var_3 = 'dict'
    var_4 = [var_1, var_2, var_3]
    var_5 = str(var_4)
    var_6 = 'type'
    var_7 = bool('type' in var_5)
    assert var_7 is True

def test_case_0():
    var_0 = 'a'
    var_1 = None
    var_2 = 'null'

def test_case_0():
    var_0 = 'missing_key'
    var_1 = 'other_key'
    var_2 = 123
    var_3 = {var_1: var_2}
    var_4 = 'required'

def test_case_0():
    var_0 = 'a'
    var_1 = 123
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'invalid_key'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_validate_key_not_in_value_with_default_triggering_line_32. Retrieved 4/19 statements.


def test_case_0():
    var_0 = 'test_key'
    var_1 = True
    var_2 = 'default_val'
    var_3 = {}



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_validate_success_with_defaults. Retrieved 5/56 statements.
# Partially parsed test_validate_error_null. Retrieved 2/23 statements.
# Partially parsed test_validate_error_type. Retrieved 5/24 statements.
# Partially parsed test_validate_error_invalid_key. Retrieved 4/26 statements.
# Partially parsed test_validate_error_required. Retrieved 4/29 statements.


def test_case_0():
    var_0 = 10
    var_1 = 'name'
    var_2 = 'age'
    var_3 = 'Alice'
    var_4 = {var_1: var_3}

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
    var_0 = 'id'
    var_1 = 'name'
    var_2 = 'Alice'
    var_3 = {var_1: var_2}



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_validate_child_schema_success. Retrieved 3/26 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'valid_value'
    var_2 = {var_0: var_1}



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_validate_success_on_child_field. Retrieved 3/17 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'test_value'
    var_2 = {var_0: var_1}



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_validate_skips_error_branch_when_child_validation_succeeds. Retrieved 5/18 statements.


def test_case_0():
    var_0 = 'success'
    var_1 = None
    var_2 = 'name'
    var_3 = 'test_value'
    var_4 = {var_2: var_3}



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_validate_error_at_line_37. Retrieved 3/30 statements.


def test_case_0():
    var_0 = 'test_key'
    var_1 = 'some_value'
    var_2 = {var_0: var_1}
    var_3 = 'test_key: error'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_validate_success_with_defaults. Retrieved 6/23 statements.
# Partially parsed test_validate_success_with_missing_optional_and_defaults. Retrieved 3/19 statements.
# Partially parsed test_validate_error_null_not_allowed. Retrieved 3/21 statements.
# Partially parsed test_validate_error_type_mismatch. Retrieved 5/19 statements.
# Partially parsed test_validate_error_required_field_missing. Retrieved 4/18 statements.
# Partially parsed test_validate_error_child_field_validation_failure. Retrieved 4/21 statements.


def test_case_0():
    var_0 = 'hello'
    var_1 = 'name'
    var_2 = 'age'
    var_3 = 'world'
    var_4 = 25
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'name'
    var_7 = 'age'

def test_case_0():
    var_0 = 'opt'
    var_1 = 'default_val'
    var_2 = {}

def test_case_0():
    var_0 = 'data'
    var_1 = False
    var_2 = None

def test_case_0():
    var_0 = 'data'
    var_1 = 'not'
    var_2 = 'a'
    var_3 = 'dict'
    var_4 = [var_1, var_2, var_3]

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 123
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)

def test_case_0():
    var_0 = 'required_key'
    var_1 = 'other_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'child'
    var_1 = 'child'
    var_2 = 'bad_value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_schema_validate_skips_read_only_fields. Retrieved 6/24 statements.


def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'readonly'
    var_3 = 'writable'
    var_4 = 'data'
    var_5 = {var_3: var_4}
    var_6 = 'writable'
    var_7 = 'readonly'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_schema_validate_key_missing_with_default_value_triggers_line_32. Retrieved 4/19 statements.


def test_case_0():
    var_0 = 'default'
    var_1 = True
    var_2 = 'test_key'
    var_3 = {}



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_validate_null_error. Retrieved 13/16 statements.


import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'item'
    var_1 = 'Mock'
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
    var_15 = 'May not be null.'

import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'item'
    var_1 = 'Mock'
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
    var_0 = 'Mock'
    var_1 = ()
    var_2 = 'validate'
    var_3 = 1
    var_4 = lambda self, x: x + var_3
    var_5 = {var_2: var_4}
    var_6 = type(var_0, var_1, var_5)
    var_7 = var_6()
    var_8 = 'item'
    var_9 = {var_8: var_7}
    var_10 = {}
    var_11 = module_0.Reference(var_8, var_9, **var_10)
    var_12 = 10
    var_13 = var_11.validate(var_12)
    assert var_13 == 11

import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'Mock'
    var_1 = ()
    var_2 = 'validate'
    var_3 = lambda self, x: x
    var_4 = {var_2: var_3}
    var_5 = type(var_0, var_1, var_4)
    var_6 = var_5()
    var_7 = 'item'
    var_8 = {var_7: var_6}
    var_9 = {}
    var_10 = module_0.Reference(var_7, var_8, **var_9)
    var_11 = var_10.target
    var_12 = bool(var_10.target == var_6)
    assert var_12 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_schema_validate_success. Retrieved 7/12 statements.
# Partially parsed test_schema_validate_null_error. Retrieved 4/9 statements.
# Partially parsed test_schema_validate_null_success. Retrieved 4/8 statements.
# Partially parsed test_schema_validate_type_error. Retrieved 6/11 statements.
# Partially parsed test_schema_validate_invalid_key_error. Retrieved 5/10 statements.
# Partially parsed test_schema_validate_required_error. Retrieved 5/10 statements.
# Partially parsed test_schema_validate_default_value_injection. Retrieved 4/8 statements.
# Partially parsed test_schema_validate_readonly_skips_logic. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'integer'
    var_1 = 'string'
    var_2 = 'age'
    var_3 = 'name'
    var_4 = 25
    var_5 = 'John'
    var_6 = {var_2: var_4, var_3: var_5}

def test_case_0():
    var_0 = 'integer'
    var_1 = False
    var_2 = 'age'
    var_3 = None

def test_case_0():
    var_0 = 'integer'
    var_1 = True
    var_2 = 'age'
    var_3 = None

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
    var_3 = 'invalid key type'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'integer'
    var_1 = 'age'
    var_2 = 'name'
    var_3 = 'John'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'integer'
    var_1 = 10
    var_2 = 'age'
    var_3 = {}

def test_case_0():
    var_0 = 'integer'
    var_1 = True
    var_2 = 'age'
    var_3 = 25
    var_4 = {var_2: var_3}
    var_5 = 'age'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_validate_success_with_defaults. Retrieved 7/29 statements.
# Partially parsed test_validate_error_type_not_object. Retrieved 5/25 statements.
# Partially parsed test_validate_error_null_not_allowed. Retrieved 3/23 statements.
# Partially parsed test_validate_error_invalid_key_type. Retrieved 4/32 statements.
# Partially parsed test_validate_error_required_field_missing. Retrieved 4/32 statements.
# Partially parsed test_validate_child_field_error_propagation. Retrieved 4/34 statements.


def test_case_0():
    var_0 = 10
    var_1 = True
    var_2 = 'age'
    var_3 = 'name'
    var_4 = 25
    var_5 = 'John'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'age'
    var_8 = 'name'

def test_case_0():
    var_0 = 'test'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = False
    var_1 = 'test'
    var_2 = None

def test_case_0():
    var_0 = {}
    var_1 = 1
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = bool(var_1)
    assert var_4 is True

def test_case_0():
    var_0 = 'must_exist'
    var_1 = 'other'
    var_2 = 123
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'child'
    var_1 = 'child'
    var_2 = 'some_val'
    var_3 = {var_1: var_2}



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_schema_validate_type_error. Retrieved 5/10 statements.
# Partially parsed test_schema_validate_null_error. Retrieved 2/7 statements.
# Partially parsed test_schema_validate_null_allowed. Retrieved 3/7 statements.
# Partially parsed test_schema_validate_invalid_key_type. Retrieved 4/9 statements.
# Partially parsed test_schema_validate_required_field_missing. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'not'
    var_2 = 'a'
    var_3 = 'dict'
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = 'name'
    var_1 = None

def test_case_0():
    var_0 = True
    var_1 = 'name'
    var_2 = None

def test_case_0():
    var_0 = 'name'
    var_1 = 123
    var_2 = 'value'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'name'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_validate_success. Retrieved 6/11 statements.
# Partially parsed test_validate_success_with_defaults. Retrieved 5/10 statements.
# Partially parsed test_validate_error_required_field_missing. Retrieved 2/7 statements.
# Partially parsed test_validate_error_nested_validation_failure. Retrieved 5/15 statements.
# Partially parsed test_validate_ignores_read_only_fields_in_required_check. Retrieved 3/7 statements.


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

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 'not a dict'
    var_4 = var_2.validate(var_3)

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.Schema(var_0, **var_3)
    var_5 = None
    var_6 = var_4.validate(var_5)

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

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 123
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)

def test_case_0():
    var_0 = 'name'
    var_1 = {}

def test_case_0():
    var_0 = 'child'
    var_1 = 'parent'
    var_2 = 123
    var_3 = {var_0: var_2}
    var_4 = {var_1: var_3}

def test_case_0():
    var_0 = True
    var_1 = 'name'
    var_2 = {}
    var_3 = 'name'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_schema_validate_skips_read_only_fields. Retrieved 7/25 statements.


def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'readonly'
    var_3 = 'writable'
    var_4 = 'should_be_ignored'
    var_5 = 'should_be_kept'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'writable'
    var_8 = 'readonly'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validate_skips_missing_key_with_no_default. Retrieved 5/17 statements.


def test_case_0():
    var_0 = 'test_key'
    var_1 = False
    var_2 = 'other_key'
    var_3 = 123
    var_4 = {var_2: var_3}
    var_5 = 'test_key'



