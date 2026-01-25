####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_multiple_errors. Retrieved 10/11 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = {}
    var_3 = 'allow_null'
    var_4 = {var_3: var_0}
    var_5 = module_1.Schema(var_2, **var_4)
    var_6 = None
    var_7 = var_5.validate(var_6)
    assert var_7 is None

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.Schema(var_0, **var_3)
    var_5 = None
    var_6 = var_4.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 'not a dict'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'valid'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 1
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'required_field'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(read_only=var_0)
    var_2 = 'read_only'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'ignored'
    var_7 = {var_2: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = bool(var_8 == {})
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = module_0.Field(default=var_0)
    var_2 = 'with_default'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = {}
    var_7 = var_5.validate(var_6)
    var_8 = bool(var_7 == {'with_default': 'default_value'})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'test'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'test'
    var_6 = None
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'name'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'John'
    var_6 = {var_1: var_5}
    var_7 = var_4.validate(var_6)
    var_8 = bool(var_7 == {'name': 'John'})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = 'req1'
    var_3 = 'req2'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = 'invalid_key'
    var_8 = 123
    var_9 = {var_7: var_8}
    var_10 = var_6.validate(var_9)
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'invalid_key'
    var_13 = 'required'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_validate_allows_null_when_allow_null_is_true. Retrieved 3/7 statements.
# Partially parsed test_validate_raises_error_for_null_when_allow_null_is_false. Retrieved 3/8 statements.
# Partially parsed test_validate_calls_target_validate_with_value. Retrieved 3/10 statements.
# Partially parsed test_validate_passes_through_target_validation_error. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'target_name'
    var_1 = True
    var_2 = None

def test_case_0():
    var_0 = 'target_name'
    var_1 = False
    var_2 = None
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'target_name'
    var_1 = 'validated_value'
    var_2 = 'input_value'

def test_case_0():
    var_0 = 'target_name'
    var_1 = 'Target error'
    var_2 = 'input_value'
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_allows_null_when_allow_null_is_true. Retrieved 3/7 statements.
# Partially parsed test_validate_raises_error_for_null_when_allow_null_is_false. Retrieved 3/8 statements.
# Partially parsed test_validate_calls_target_validate_with_value. Retrieved 5/9 statements.
# Partially parsed test_validate_returns_target_validate_result. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'target'
    var_1 = True
    var_2 = None

def test_case_0():
    var_0 = 'target'
    var_1 = False
    var_2 = None
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'target'
    var_1 = False
    var_2 = 'id'
    var_3 = 1
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'target'
    var_1 = False
    var_2 = 'id'
    var_3 = 1
    var_4 = {var_2: var_3}



# Parsed testcases at query #4
#--------------------------

# Failed to parse test_setitem_adds_new_key.
# Failed to parse test_setitem_raises_assertion_on_duplicate_key.
# Failed to parse test_setitem_works_with_empty_dict.
# Failed to parse test_setitem_multiple_unique_keys.
# Partially parsed test_setitem_after_deletion. Retrieved 3/7 statements.
# Failed to parse test_setitem_with_none_value.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = [var_2]



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_validate_allow_null. Retrieved 3/7 statements.
# Partially parsed test_validate_not_allow_null. Retrieved 3/8 statements.
# Partially parsed test_validate_with_value. Retrieved 5/12 statements.
# Partially parsed test_validate_calls_target_validate. Retrieved 5/12 statements.


def test_case_0():
    var_0 = 'target'
    var_1 = True
    var_2 = None

def test_case_0():
    var_0 = 'target'
    var_1 = False
    var_2 = None
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'target'
    var_1 = False
    var_2 = 'id'
    var_3 = 1
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'target'
    var_1 = False
    var_2 = 'id'
    var_3 = 1
    var_4 = {var_2: var_3}



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_validate_with_no_error_from_child_schema. Retrieved 4/15 statements.


def test_case_0():
    var_0 = False
    var_1 = 'field1'
    var_2 = 'some_value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_validate_with_valid_child_schema. Retrieved 4/13 statements.


def test_case_0():
    var_0 = False
    var_1 = 'key'
    var_2 = 'valid'
    var_3 = {var_1: var_2}



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validate_with_error_in_child_schema. Retrieved 16/20 statements.


import typesystem.fields as module_0
import typesystem.base as module_1
import typesystem.schemas as module_2

def test_case_0():
    var_0 = module_0.Field()
    var_1 = None
    var_2 = 'Invalid'
    var_3 = 'invalid'
    var_4 = []
    var_5 = module_1.Message(text=var_2, code=var_3, index=var_4)
    var_6 = [var_5]
    var_7 = module_1.ValidationError(messages=var_6)
    var_8 = (var_1, var_7)
    var_9 = False
    var_10 = 'key'
    var_11 = {var_10: var_0}
    var_12 = {}
    var_13 = module_2.Schema(var_11, **var_12)
    var_14 = 'invalid_value'
    var_15 = {var_10: var_14}
    var_16 = var_13.validate(var_15)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_validate_allow_null. Retrieved 3/7 statements.
# Partially parsed test_validate_not_allow_null. Retrieved 3/8 statements.
# Partially parsed test_validate_delegates_to_target. Retrieved 5/9 statements.
# Partially parsed test_validate_target_validation_error. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 'target'
    var_1 = True
    var_2 = None

def test_case_0():
    var_0 = 'target'
    var_1 = False
    var_2 = None
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'target'
    var_1 = False
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'target'
    var_1 = True
    var_2 = False
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_validate_with_missing_required_field. Retrieved 7/9 statements.
# Partially parsed test_validate_with_field_has_default_and_missing. Retrieved 10/13 statements.
# Partially parsed test_validate_with_field_validation_error. Retrieved 16/19 statements.
# Partially parsed test_validate_successful_with_all_fields. Retrieved 9/11 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = {}
    var_3 = 'allow_null'
    var_4 = {var_3: var_0}
    var_5 = module_1.Schema(var_2, **var_4)
    var_6 = None
    var_7 = var_5.validate(var_6)
    assert var_7 is None

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.Schema(var_0, **var_3)
    var_5 = None
    var_6 = var_4.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 'not a dict'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 1
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Field(read_only=var_0)
    var_2 = 'required_key'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = {}
    var_7 = var_5.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(read_only=var_0)
    var_2 = 'read_only_key'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'some_value'
    var_7 = {var_2: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = 'read_only_key'
    var_10 = bool('read_only_key' not in var_8)
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Field(read_only=var_0)
    var_2 = True
    var_3 = 'default_value'
    var_4 = None
    var_5 = 'key_with_default'
    var_6 = {var_5: var_1}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = {}
    var_10 = var_8.validate(var_9)
    var_11 = var_10['key_with_default']
    assert var_11 == 'default_value'

import typesystem.fields as module_0
import typesystem.base as module_1
import typesystem.schemas as module_2

def test_case_0():
    var_0 = False
    var_1 = module_0.Field(read_only=var_0)
    var_2 = 'error'
    var_3 = []
    var_4 = module_1.Message(text=var_2, code=var_2, index=var_3)
    var_5 = [var_4]
    var_6 = module_1.ValidationError(messages=var_5)
    var_7 = None
    var_8 = (var_7, var_6)
    var_9 = 'problem_key'
    var_10 = {var_9: var_1}
    var_11 = {}
    var_12 = module_2.Schema(var_10, **var_11)
    var_13 = 'problem_key'
    var_14 = 'value'
    var_15 = {var_13: var_14}
    var_16 = var_12.validate(var_15)
    var_17 = bool(False)
    assert var_17 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Field(read_only=var_0)
    var_2 = None
    var_3 = 'valid_key'
    var_4 = {var_3: var_1}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = 'valid_value'
    var_8 = {var_3: var_7}
    var_9 = var_6.validate(var_8)
    var_10 = bool(var_9 == {'valid_key': 'valid_value'})
    assert var_10 is True



# Parsed testcases at query #11
#--------------------------




import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = {}
    var_3 = 'allow_null'
    var_4 = {var_3: var_0}
    var_5 = module_1.Schema(var_2, **var_4)
    var_6 = None
    var_7 = var_5.validate(var_6)
    assert var_7 is None

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.Schema(var_0, **var_3)
    var_5 = None
    var_6 = var_4.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 'not a dict'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 1
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'required_field'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(read_only=var_0)
    var_2 = 'read_only_field'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = {}
    var_7 = var_5.validate(var_6)
    var_8 = bool(var_7 == {})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = module_0.Field(default=var_0)
    var_2 = 'field_with_default'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = {}
    var_7 = var_5.validate(var_6)
    var_8 = bool(var_7 == {'field_with_default': 'default_value'})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'child'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'child_value'
    var_6 = {var_1: var_5}
    var_7 = var_4.validate(var_6)
    var_8 = bool(var_7 == {'child': 'child_value'})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'child'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'child'
    var_6 = None
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = 'child1'
    var_3 = 'child2'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = 'child1'
    var_8 = 'child2'
    var_9 = None
    var_10 = {var_7: var_9, var_8: var_9}
    var_11 = var_6.validate(var_10)
    var_12 = bool(False)
    assert var_12 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'defined'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'extra'
    var_6 = 'value'
    var_7 = 'ignored'
    var_8 = {var_1: var_6, var_5: var_7}
    var_9 = var_4.validate(var_8)
    var_10 = bool(var_9 == {'defined': 'value'})
    assert var_10 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_validate_with_valid_child_schema. Retrieved 4/13 statements.


def test_case_0():
    var_0 = False
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_validate_with_error_in_child_schema. Retrieved 4/29 statements.


def test_case_0():
    var_0 = True
    var_1 = 'test_key'
    var_2 = 'some_value'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_validate_when_value_is_none_and_allow_null_is_true. Retrieved 7/10 statements.
# Partially parsed test_validate_when_value_is_none_and_allow_null_is_false. Retrieved 8/14 statements.
# Partially parsed test_validate_when_value_is_not_none. Retrieved 7/11 statements.


import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'validated'
    var_2 = 'some_ref'
    var_3 = True
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Reference(var_2, var_0, **var_5)
    var_7 = None
    var_8 = var_6.validate(var_7)
    assert var_8 is None

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'validated'
    var_2 = 'some_ref'
    var_3 = False
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Reference(var_2, var_0, **var_5)
    var_7 = 'null error'
    var_8 = [var_7]
    var_9 = None
    var_10 = var_6.validate(var_9)
    var_11 = bool(False)
    assert var_11 is True

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'validated'
    var_2 = 'some_ref'
    var_3 = False
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Reference(var_2, var_0, **var_5)
    var_7 = 'some_value'
    var_8 = var_6.validate(var_7)
    assert var_8 == 'validated'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_validate_child_schema_error. Retrieved 4/24 statements.


def test_case_0():
    var_0 = False
    var_1 = 'key'
    var_2 = 'some_value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_validate_child_schema_error. Retrieved 4/17 statements.


def test_case_0():
    var_0 = False
    var_1 = 'key'
    var_2 = 'invalid'
    var_3 = {var_1: var_2}



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_validate_with_child_schema_error. Retrieved 6/24 statements.


def test_case_0():
    var_0 = False
    var_1 = 'key'
    var_2 = None
    var_3 = []
    var_4 = 'invalid'
    var_5 = {var_1: var_4}



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_validate_with_error_in_child_schema. Retrieved 4/14 statements.


def test_case_0():
    var_0 = False
    var_1 = 'field'
    var_2 = 'invalid'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_collects_errors_from_child_fields. Retrieved 15/19 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = {}
    var_3 = 'allow_null'
    var_4 = {var_3: var_0}
    var_5 = module_1.Schema(var_2, **var_4)
    var_6 = None
    var_7 = var_5.validate(var_6)
    assert var_7 is None

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.Schema(var_0, **var_3)
    var_5 = None
    var_6 = var_4.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 'not a dict'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 123
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'required_field'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = module_0.Field(default=var_0)
    var_2 = 'field_with_default'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = {}
    var_7 = var_5.validate(var_6)
    var_8 = bool(var_7 == {'field_with_default': 'default_value'})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(read_only=var_0)
    var_2 = 'read_only_field'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = {}
    var_7 = var_5.validate(var_6)
    var_8 = bool(var_7 == {})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'valid_field'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'some_value'
    var_6 = {var_1: var_5}
    var_7 = var_4.validate(var_6)
    var_8 = bool(var_7 == {'valid_field': 'some_value'})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.base as module_1
import typesystem.schemas as module_2

def test_case_0():
    var_0 = module_0.Field()
    var_1 = ()
    var_2 = 'Child error'
    var_3 = 'child_error'
    var_4 = []
    var_5 = module_1.Message(text=var_2, code=var_3, index=var_4)
    var_6 = [var_5]
    var_7 = module_1.ValidationError(messages=var_6)
    var_8 = 'problem_field'
    var_9 = {var_8: var_0}
    var_10 = {}
    var_11 = module_2.Schema(var_9, **var_10)
    var_12 = 'problem_field'
    var_13 = 'bad_value'
    var_14 = {var_12: var_13}
    var_15 = var_11.validate(var_14)
    var_16 = bool(False)
    assert var_16 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'default2'
    var_2 = module_0.Field(default=var_1)
    var_3 = 'field1'
    var_4 = 'field2'
    var_5 = {var_3: var_0, var_4: var_2}
    var_6 = {}
    var_7 = module_1.Schema(var_5, **var_6)
    var_8 = 'value1'
    var_9 = {var_3: var_8}
    var_10 = var_7.validate(var_9)
    var_11 = bool(var_10 == {'field1': 'value1', 'field2': 'default2'})
    assert var_11 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_validate_allows_null_when_allow_null_is_true. Retrieved 7/10 statements.
# Partially parsed test_validate_calls_target_validate_with_value. Retrieved 6/11 statements.


import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'validated_value'
    var_2 = 'some_key'
    var_3 = True
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Reference(var_2, var_0, **var_5)
    var_7 = None
    var_8 = var_6.validate(var_7)
    assert var_8 is None

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'some_key'
    var_2 = False
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.Reference(var_1, var_0, **var_4)
    var_6 = None
    var_7 = var_5.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'validated_value'
    var_2 = 'some_key'
    var_3 = {}
    var_4 = module_0.Reference(var_2, var_0, **var_3)
    var_5 = 'input_value'
    var_6 = var_4.validate(var_5)
    assert var_6 == 'validated_value'



# Parsed testcases at query #3
#--------------------------

# Failed to parse test_setitem_adds_new_key.
# Failed to parse test_setitem_raises_assertion_on_duplicate_key.
# Failed to parse test_setitem_works_with_empty_dict.
# Partially parsed test_setitem_works_with_prefilled_dict. Retrieved 3/5 statements.
# Failed to parse test_setitem_key_not_in_definitions.
# Failed to parse test_setitem_value_stored_correctly.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = [var_2]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_validate_with_child_schema_error. Retrieved 9/32 statements.


import typesystem.base as module_0

def test_case_0():
    var_0 = 'Invalid'
    var_1 = 'invalid'
    var_2 = []
    var_3 = module_0.Message(text=var_0, code=var_1, index=var_2)
    var_4 = [var_3]
    var_5 = module_0.ValidationError(messages=var_4)
    var_6 = 'key'
    var_7 = 'some_value'
    var_8 = {var_6: var_7}



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_serialize_with_none_object. Retrieved 2/7 statements.
# Partially parsed test_serialize_with_dict_object. Retrieved 5/11 statements.
# Partially parsed test_serialize_with_object_attributes. Retrieved 4/15 statements.
# Partially parsed test_serialize_missing_key_in_dict. Retrieved 4/10 statements.
# Partially parsed test_serialize_missing_attribute_in_object. Retrieved 3/13 statements.
# Partially parsed test_serialize_with_nested_field_serialization. Retrieved 8/15 statements.
# Partially parsed test_serialize_with_read_only_field_in_dict. Retrieved 6/16 statements.
# Partially parsed test_serialize_with_read_only_field_in_object. Retrieved 5/20 statements.
# Partially parsed test_serialize_empty_dict. Retrieved 2/7 statements.
# Partially parsed test_serialize_empty_object. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = None

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = {var_0: var_2}

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'

def test_case_0():
    var_0 = 'name'
    var_1 = 'scores'
    var_2 = 'John'
    var_3 = 85
    var_4 = 90
    var_5 = 78
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_0: var_2, var_1: var_6}

def test_case_0():
    var_0 = True
    var_1 = 'name'
    var_2 = 'id'
    var_3 = 'John'
    var_4 = 123
    var_5 = {var_1: var_3, var_2: var_4}

def test_case_0():
    var_0 = True
    var_1 = 'name'
    var_2 = 'id'
    var_3 = 'John'
    var_4 = 123

def test_case_0():
    var_0 = 'name'
    var_1 = {}

def test_case_0():
    var_0 = 'name'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_serialize_with_non_mapping_object_and_missing_attribute. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = []



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_serialize_with_non_mapping_object_without_attribute. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'test_key'
    var_1 = []



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validate_with_error_in_child_schema. Retrieved 4/19 statements.


def test_case_0():
    var_0 = False
    var_1 = 'key'
    var_2 = 'invalid'
    var_3 = {var_1: var_2}



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_serialize_handles_object_with_attributes. Retrieved 5/8 statements.
# Partially parsed test_serialize_ignores_missing_attributes_in_object. Retrieved 5/8 statements.
# Partially parsed test_serialize_calls_field_serialize_for_each_key. Retrieved 7/8 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'test'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = None
    var_6 = var_4.serialize(var_5)
    assert var_6 is None

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'test'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'value'
    var_6 = {var_1: var_5}
    var_7 = var_4.serialize(var_6)
    var_8 = bool(var_7 == {'test': None})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'test'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'value'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'test'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'other'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.serialize(var_7)
    var_9 = bool(var_8 == {})
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'test'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'value'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'test'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'value'
    var_6 = {var_1: var_5}
    var_7 = var_4.serialize(var_6)
    var_8 = bool(var_7 == {'test': 'serialized_value'})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = 'field1'
    var_3 = 'field2'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = 'value1'
    var_8 = 'value2'
    var_9 = {var_2: var_7, var_3: var_8}
    var_10 = var_6.serialize(var_9)
    var_11 = bool(var_10 == {'field1': None, 'field2': None})
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = 'field1'
    var_3 = 'field2'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = 'value1'
    var_8 = {var_2: var_7}
    var_9 = var_6.serialize(var_8)
    var_10 = bool(var_9 == {'field1': None})
    assert var_10 is True



# Parsed testcases at query #10
#--------------------------

# Failed to parse test_setitem_raises_assertion_error_when_key_exists.




# Parsed testcases at query #11
#--------------------------

# Partially parsed test_validate_collects_child_field_validation_errors. Retrieved 15/19 statements.
# Partially parsed test_validate_handles_mapping_subclass. Retrieved 6/10 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = {}
    var_3 = 'allow_null'
    var_4 = {var_3: var_0}
    var_5 = module_1.Schema(var_2, **var_4)
    var_6 = None
    var_7 = var_5.validate(var_6)
    assert var_7 is None

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.Schema(var_0, **var_3)
    var_5 = None
    var_6 = var_4.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 'not a dict'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 1
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Field(read_only=var_0)
    var_2 = 'required_field'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = {}
    var_7 = var_5.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = False
    var_1 = 'default_value'
    var_2 = module_0.Field(default=var_1, read_only=var_0)
    var_3 = 'field_with_default'
    var_4 = {var_3: var_2}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = {}
    var_8 = var_6.validate(var_7)
    var_9 = bool(var_8 == {'field_with_default': 'default_value'})
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(read_only=var_0)
    var_2 = 'read_only_field'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'some_value'
    var_7 = {var_2: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = bool(var_8 == {})
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'child'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'child_value'
    var_6 = {var_1: var_5}
    var_7 = var_4.validate(var_6)
    var_8 = bool(var_7 == {'child': 'child_value'})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.base as module_1
import typesystem.schemas as module_2

def test_case_0():
    var_0 = module_0.Field()
    var_1 = ()
    var_2 = 'Child error'
    var_3 = 'child_error'
    var_4 = []
    var_5 = module_1.Message(text=var_2, code=var_3, index=var_4)
    var_6 = [var_5]
    var_7 = module_1.ValidationError(messages=var_6)
    var_8 = 'child'
    var_9 = {var_8: var_0}
    var_10 = {}
    var_11 = module_2.Schema(var_9, **var_10)
    var_12 = 'child'
    var_13 = 'bad_value'
    var_14 = {var_12: var_13}
    var_15 = var_11.validate(var_14)
    var_16 = bool(False)
    assert var_16 is True

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = {}
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == {})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'test'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'data'
    var_6 = {var_1: var_5}



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_validate_with_valid_child_schema. Retrieved 4/13 statements.


def test_case_0():
    var_0 = False
    var_1 = 'key'
    var_2 = 'valid'
    var_3 = {var_1: var_2}



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_validate_multiple_errors. Retrieved 11/12 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = {}
    var_3 = 'allow_null'
    var_4 = {var_3: var_0}
    var_5 = module_1.Schema(var_2, **var_4)
    var_6 = None
    var_7 = var_5.validate(var_6)
    assert var_7 is None

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.Schema(var_0, **var_3)
    var_5 = None
    var_6 = var_4.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 'not a dict'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 1
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = 'required_field'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = {}
    var_7 = var_5.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(read_only=var_0)
    var_2 = 'read_only_field'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'value'
    var_7 = {var_2: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = 'read_only_field'
    var_10 = bool('read_only_field' not in var_8)
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = module_0.Field(default=var_0)
    var_2 = 'field_with_default'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = {}
    var_7 = var_5.validate(var_6)
    var_8 = var_7['field_with_default']
    assert var_8 == 'default_value'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = 'child'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'child'
    var_7 = None
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'child'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'value'
    var_6 = {var_1: var_5}
    var_7 = var_4.validate(var_6)
    var_8 = bool(var_7 == {'child': 'value'})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = 'required'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 1
    var_7 = 'required'
    var_8 = 'invalid'
    var_9 = None
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = var_5.validate(var_10)
    var_12 = bool(False)
    assert var_12 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_validate_with_error_in_child_schema. Retrieved 4/27 statements.


def test_case_0():
    var_0 = True
    var_1 = 'key'
    var_2 = 'some_value'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_setitem_raises_assertion_error_when_key_exists. Retrieved 2/5 statements.


def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = True
    var_3 = bool(var_2)
    assert var_3 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_validate_with_valid_child_value. Retrieved 4/13 statements.


def test_case_0():
    var_0 = False
    var_1 = 'key'
    var_2 = 'valid'
    var_3 = {var_1: var_2}



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_validate_collects_child_field_validation_errors. Retrieved 15/19 statements.
# Partially parsed test_validate_aggregates_multiple_errors. Retrieved 8/9 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = {}
    var_3 = 'allow_null'
    var_4 = {var_3: var_0}
    var_5 = module_1.Schema(var_2, **var_4)
    var_6 = None
    var_7 = var_5.validate(var_6)
    assert var_7 is None

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.Schema(var_0, **var_3)
    var_5 = None
    var_6 = var_4.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 'not a dict'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 123
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'required_key'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = module_0.Field(default=var_0)
    var_2 = 'key_with_default'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = {}
    var_7 = var_5.validate(var_6)
    var_8 = bool(var_7 == {'key_with_default': 'default_value'})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(read_only=var_0)
    var_2 = 'read_only_key'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'value'
    var_7 = {var_2: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = bool(var_8 == {})
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'child'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'child_value'
    var_6 = {var_1: var_5}
    var_7 = var_4.validate(var_6)
    var_8 = bool(var_7 == {'child': 'child_value'})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.base as module_1
import typesystem.schemas as module_2

def test_case_0():
    var_0 = module_0.Field()
    var_1 = ()
    var_2 = 'Child error'
    var_3 = 'child_error'
    var_4 = []
    var_5 = module_1.Message(text=var_2, code=var_3, index=var_4)
    var_6 = [var_5]
    var_7 = module_1.ValidationError(messages=var_6)
    var_8 = 'child'
    var_9 = {var_8: var_0}
    var_10 = {}
    var_11 = module_2.Schema(var_9, **var_10)
    var_12 = 'child'
    var_13 = 'bad_value'
    var_14 = {var_12: var_13}
    var_15 = var_11.validate(var_14)
    var_16 = bool(False)
    assert var_16 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'default2'
    var_2 = module_0.Field(default=var_1)
    var_3 = True
    var_4 = module_0.Field(read_only=var_3)
    var_5 = 'key1'
    var_6 = 'key2'
    var_7 = 'key3'
    var_8 = {var_5: var_0, var_6: var_2, var_7: var_4}
    var_9 = {}
    var_10 = module_1.Schema(var_8, **var_9)
    var_11 = 'value1'
    var_12 = 'value3'
    var_13 = {var_5: var_11, var_7: var_12}
    var_14 = var_10.validate(var_13)
    var_15 = bool(var_14 == {'key1': 'value1', 'key2': 'default2'})
    assert var_15 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'required'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'invalid_key'
    var_6 = 123
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #18
#--------------------------

# Failed to parse test_setitem_raises_assertion_error_on_duplicate_key.




# Parsed testcases at query #19
#--------------------------

# Failed to parse test_setitem_adds_new_key.
# Failed to parse test_setitem_raises_assertion_on_duplicate_key.
# Partially parsed test_setitem_works_with_existing_dict. Retrieved 5/7 statements.
# Partially parsed test_setitem_preserves_other_keys. Retrieved 3/5 statements.
# Partially parsed test_setitem_after_delitem. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_4]

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = [var_2]

def test_case_0():
    var_0 = 'temp'
    var_1 = 5
    var_2 = {var_0: var_1}
    var_3 = [var_2]



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_validate_with_error_in_child_schema. Retrieved 4/17 statements.


def test_case_0():
    var_0 = False
    var_1 = 'key'
    var_2 = 'invalid'
    var_3 = {var_1: var_2}



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_validate_with_no_error_from_child_schema. Retrieved 3/22 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'valid'
    var_2 = {var_0: var_1}



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_validate_with_no_error_from_child_schema. Retrieved 4/23 statements.


def test_case_0():
    var_0 = False
    var_1 = 'key'
    var_2 = 'valid_value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_validate_with_error_in_child_schema. Retrieved 4/14 statements.


def test_case_0():
    var_0 = False
    var_1 = 'field'
    var_2 = 'invalid'
    var_3 = {var_1: var_2}



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_setitem_raises_assertion_error_when_key_exists. Retrieved 2/5 statements.


def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = True
    var_3 = bool(var_2)
    assert var_3 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_validate_combines_multiple_errors. Retrieved 8/9 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = {}
    var_3 = 'allow_null'
    var_4 = {var_3: var_0}
    var_5 = module_1.Schema(var_2, **var_4)
    var_6 = None
    var_7 = var_5.validate(var_6)
    assert var_7 is None

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.Schema(var_0, **var_3)
    var_5 = None
    var_6 = var_4.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 'not a dict'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 1
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'required_field'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(read_only=var_0)
    var_2 = 'read_only_field'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = {}
    var_7 = var_5.validate(var_6)
    var_8 = bool(var_7 == {})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = module_0.Field(default=var_0)
    var_2 = 'field_with_default'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = {}
    var_7 = var_5.validate(var_6)
    var_8 = bool(var_7 == {'field_with_default': 'default_value'})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'child'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'valid'
    var_6 = {var_1: var_5}
    var_7 = var_4.validate(var_6)
    var_8 = bool(var_7 == {'child': 'valid'})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'child'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'child'
    var_6 = None
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'required'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 1
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'invalid_key'
    var_11 = 'required'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_validate_multiple_errors. Retrieved 8/9 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = {}
    var_3 = 'allow_null'
    var_4 = {var_3: var_0}
    var_5 = module_1.Schema(var_2, **var_4)
    var_6 = None
    var_7 = var_5.validate(var_6)
    assert var_7 is None

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = None
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 'not a dict'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'valid'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 123
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'required_field'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(read_only=var_0)
    var_2 = 'read_only'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'ignored'
    var_7 = {var_2: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = bool(var_8 == {})
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = module_0.Field(default=var_0)
    var_2 = 'with_default'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = {}
    var_7 = var_5.validate(var_6)
    var_8 = bool(var_7 == {'with_default': 'default_value'})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'test'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'test'
    var_6 = None
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'name'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'John'
    var_6 = {var_1: var_5}
    var_7 = var_4.validate(var_6)
    var_8 = bool(var_7 == {'name': 'John'})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'required'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 123
    var_6 = 'invalid'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'invalid_key'
    var_11 = 'required'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_validate_child_schema_no_error. Retrieved 5/25 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_validate_with_child_schema_error. Retrieved 15/19 statements.


import typesystem.fields as module_0
import typesystem.base as module_1
import typesystem.schemas as module_2

def test_case_0():
    var_0 = module_0.Field()
    var_1 = False
    var_2 = None
    var_3 = 'error'
    var_4 = []
    var_5 = module_1.Message(text=var_3, code=var_3, index=var_4)
    var_6 = [var_5]
    var_7 = module_1.ValidationError(messages=var_6)
    var_8 = (var_2, var_7)
    var_9 = 'key'
    var_10 = {var_9: var_0}
    var_11 = {}
    var_12 = module_2.Schema(var_10, **var_11)
    var_13 = 'value'
    var_14 = {var_9: var_13}
    var_15 = var_12.validate(var_14)



