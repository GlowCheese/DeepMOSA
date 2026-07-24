####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_returns_none_when_value_is_none_and_allow_null_is_true. Retrieved 3/7 statements.
# Partially parsed test_validate_raises_error_when_value_is_none_and_allow_null_is_false. Retrieved 3/8 statements.
# Partially parsed test_validate_calls_target_validate_when_value_is_not_none. Retrieved 5/11 statements.
# Partially parsed test_validate_works_with_allow_null_true_and_non_none_value. Retrieved 4/10 statements.


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
    var_1 = True
    var_2 = 'id'
    var_3 = {var_2: var_1}



# Parsed testcases at query #2
#--------------------------

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


def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 'not a dict'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


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


def test_case_0():
    var_0 = True
    var_1 = module_0.Field(read_only=var_0)
    var_2 = 'read_only_field'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'ignored'
    var_7 = {var_2: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = bool(var_8 == {})
    assert var_9 is True


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


def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'default2'
    var_2 = module_0.Field(default=var_1)
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_0, var_4: var_2}
    var_6 = {}
    var_7 = module_1.Schema(var_5, **var_6)
    var_8 = 'value1'
    var_9 = {var_3: var_8}
    var_10 = var_7.validate(var_9)
    var_11 = bool(var_10 == {'key1': 'value1', 'key2': 'default2'})
    assert var_11 is True


def test_case_0():
    var_0 = module_0.Field()
    var_1 = False
    var_2 = module_0.Field(allow_null=var_1)
    var_3 = 'valid'
    var_4 = 'invalid'
    var_5 = {var_3: var_0, var_4: var_2}
    var_6 = {}
    var_7 = module_1.Schema(var_5, **var_6)
    var_8 = 'valid'
    var_9 = 'invalid'
    var_10 = 'ok'
    var_11 = None
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = var_7.validate(var_12)
    var_14 = bool(False)
    assert var_14 is True


def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'required'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 123
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_with_error_in_child_schema. Retrieved 4/16 statements.


def test_case_0():
    var_0 = False
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_validate_allows_null_when_allow_null_is_true. Retrieved 3/7 statements.
# Partially parsed test_validate_raises_error_for_null_when_allow_null_is_false. Retrieved 3/8 statements.
# Partially parsed test_validate_delegates_to_target_validate_for_non_null_value. Retrieved 5/11 statements.
# Partially parsed test_validate_delegates_to_target_validate_when_allow_null_is_true_but_value_not_null. Retrieved 5/11 statements.


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
    var_1 = False
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'target_name'
    var_1 = True
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_validate_with_error_in_child_schema. Retrieved 4/17 statements.


def test_case_0():
    var_0 = False
    var_1 = 'key'
    var_2 = 'invalid'
    var_3 = {var_1: var_2}



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_validate_with_none_and_allow_null_true. Retrieved 7/10 statements.
# Partially parsed test_validate_with_none_and_allow_null_false. Retrieved 8/14 statements.
# Partially parsed test_validate_with_non_none_value. Retrieved 7/11 statements.


import typesystem.schemas as module_0


def test_case_0():
    var_0 = {}
    var_1 = 'validated_value'
    var_2 = 'some_ref'
    var_3 = True
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Reference(var_2, var_0, **var_5)
    var_7 = None
    var_8 = var_6.validate(var_7)
    assert var_8 is None


def test_case_0():
    var_0 = {}
    var_1 = 'validated_value'
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


def test_case_0():
    var_0 = {}
    var_1 = 'validated_value'
    var_2 = 'some_ref'
    var_3 = False
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Reference(var_2, var_0, **var_5)
    var_7 = 'some_value'
    var_8 = var_6.validate(var_7)
    assert var_8 == 'validated_value'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_validate_with_child_schema_error. Retrieved 3/14 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'invalid'
    var_2 = {var_0: var_1}



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validate_with_child_schema_error. Retrieved 4/16 statements.


def test_case_0():
    var_0 = False
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_validate_with_child_schema_error. Retrieved 10/31 statements.


import typesystem.base as module_0


def test_case_0():
    var_0 = None
    var_1 = 'error'
    var_2 = []
    var_3 = module_0.Message(text=var_1, code=var_1, index=var_2)
    var_4 = [var_3]
    var_5 = module_0.ValidationError(messages=var_4)
    var_6 = (var_0, var_5)
    var_7 = 'key'
    var_8 = 'invalid'
    var_9 = {var_7: var_8}



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_validate_with_multiple_errors. Retrieved 12/13 statements.


import typesystem.fields as module_0


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


def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 'not a dict'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


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
    var_9 = var_8['field_with_default']
    assert var_9 == 'default_value'


def test_case_0():
    var_0 = False
    var_1 = module_0.Field(read_only=var_0)
    var_2 = 'problem_field'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'problem_field'
    var_7 = None
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True


def test_case_0():
    var_0 = False
    var_1 = module_0.Field(read_only=var_0)
    var_2 = 'valid_field'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'valid_value'
    var_7 = {var_2: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = bool(var_8 == {'valid_field': 'valid_value'})
    assert var_9 is True


def test_case_0():
    var_0 = False
    var_1 = module_0.Field(read_only=var_0)
    var_2 = module_0.Field(read_only=var_0)
    var_3 = 'field1'
    var_4 = 'field2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = {}
    var_7 = module_1.Schema(var_5, **var_6)
    var_8 = 'field1'
    var_9 = 'field2'
    var_10 = None
    var_11 = {var_8: var_10, var_9: var_10}
    var_12 = var_7.validate(var_11)
    var_13 = bool(False)
    assert var_13 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_validate_with_error_in_child_schema. Retrieved 4/19 statements.


def test_case_0():
    var_0 = False
    var_1 = 'key'
    var_2 = 'invalid'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_validate_with_error_in_child_schema. Retrieved 15/19 statements.


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
    var_13 = 'invalid'
    var_14 = {var_9: var_13}
    var_15 = var_12.validate(var_14)



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_multiple_errors. Retrieved 8/9 statements.


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


def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 'not a dict'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


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


def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'test_field'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'test_field'
    var_6 = None
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True


def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'test_field'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'valid'
    var_6 = {var_1: var_5}
    var_7 = var_4.validate(var_6)
    var_8 = bool(var_7 == {'test_field': 'valid'})
    assert var_8 is True


def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'required'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 1
    var_6 = 'invalid'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'invalid_key'
    var_11 = 'required'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_validate_with_error_in_child_schema. Retrieved 4/17 statements.


def test_case_0():
    var_0 = False
    var_1 = 'key'
    var_2 = 'invalid'
    var_3 = {var_1: var_2}



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_allows_null_when_allow_null_is_true. Retrieved 7/10 statements.
# Partially parsed test_validate_calls_target_validate_for_non_null_value. Retrieved 6/11 statements.


import typesystem.schemas as module_0


def test_case_0():
    var_0 = {}
    var_1 = 'validated_value'
    var_2 = 'some_target'
    var_3 = True
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Reference(var_2, var_0, **var_5)
    var_7 = None
    var_8 = var_6.validate(var_7)
    assert var_8 is None


def test_case_0():
    var_0 = {}
    var_1 = 'some_target'
    var_2 = False
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.Reference(var_1, var_0, **var_4)
    var_6 = None
    var_7 = var_5.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True


def test_case_0():
    var_0 = {}
    var_1 = 'validated_value'
    var_2 = 'some_target'
    var_3 = {}
    var_4 = module_0.Reference(var_2, var_0, **var_3)
    var_5 = 'some_value'
    var_6 = var_4.validate(var_5)
    assert var_6 == 'validated_value'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_validate_with_multiple_errors. Retrieved 12/13 statements.


import typesystem.fields as module_0


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


def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 'not a dict'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


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


def test_case_0():
    var_0 = True
    var_1 = module_0.Field(read_only=var_0)
    var_2 = 'read_only_field'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'some value'
    var_7 = {var_2: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = 'read_only_field'
    var_10 = bool('read_only_field' not in var_8)
    assert var_10 is True


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
    var_9 = var_8['field_with_default']
    assert var_9 == 'default_value'


def test_case_0():
    var_0 = False
    var_1 = module_0.Field(read_only=var_0)
    var_2 = 'problem_field'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'problem_field'
    var_7 = 'invalid'
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True


def test_case_0():
    var_0 = False
    var_1 = module_0.Field(read_only=var_0)
    var_2 = 'valid_field'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'valid_value'
    var_7 = {var_2: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = var_8['valid_field']
    assert var_9 == 'valid_value'


def test_case_0():
    var_0 = False
    var_1 = module_0.Field(read_only=var_0)
    var_2 = module_0.Field(read_only=var_0)
    var_3 = 'field1'
    var_4 = 'field2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = {}
    var_7 = module_1.Schema(var_5, **var_6)
    var_8 = 'field1'
    var_9 = 'field2'
    var_10 = 'invalid'
    var_11 = {var_8: var_10, var_9: var_10}
    var_12 = var_7.validate(var_11)
    var_13 = bool(False)
    assert var_13 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_validate_with_multiple_errors. Retrieved 8/9 statements.



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


def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 'not a dict'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


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


def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'child'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'valid'
    var_6 = {var_1: var_5}
    var_7 = var_4.validate(var_6)
    var_8 = var_7['child']
    assert var_8 == 'valid'


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


def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = 'req1'
    var_3 = 'req2'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = {}
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True


def test_case_0():
    var_0 = 'nested_field'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'nested'
    var_6 = {var_5: var_4}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = 'value'
    var_10 = {var_0: var_9}
    var_11 = {var_5: var_10}
    var_12 = var_8.validate(var_11)
    var_13 = var_12['nested']['nested_field']
    assert var_13 == 'value'


def test_case_0():
    var_0 = 'nested_field'
    var_1 = False
    var_2 = module_0.Field(allow_null=var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'nested'
    var_7 = {var_6: var_5}
    var_8 = {}
    var_9 = module_1.Schema(var_7, **var_8)
    var_10 = 'nested'
    var_11 = 'nested_field'
    var_12 = None
    var_13 = {var_11: var_12}
    var_14 = {var_10: var_13}
    var_15 = var_9.validate(var_14)
    var_16 = bool(False)
    assert var_16 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_validate_raises_required_error_when_required_key_is_missing. Retrieved 7/9 statements.
# Partially parsed test_validate_uses_default_value_when_key_is_missing_and_field_has_default. Retrieved 10/13 statements.
# Partially parsed test_validate_includes_validated_child_values. Retrieved 9/11 statements.
# Partially parsed test_validate_collects_child_validation_errors. Retrieved 17/20 statements.
# Partially parsed test_validate_returns_validated_dict_with_multiple_fields. Retrieved 14/19 statements.



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


def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 'not a dict'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


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


def test_case_0():
    var_0 = False
    var_1 = module_0.Field(read_only=var_0)
    var_2 = True
    var_3 = 'default'
    var_4 = None
    var_5 = 'key_with_default'
    var_6 = {var_5: var_1}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = {}
    var_10 = var_8.validate(var_9)
    var_11 = bool(var_10 == {'key_with_default': 'default'})
    assert var_11 is True


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


def test_case_0():
    var_0 = False
    var_1 = module_0.Field(read_only=var_0)
    var_2 = None
    var_3 = 'child_key'
    var_4 = {var_3: var_1}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = 'value'
    var_8 = {var_3: var_7}
    var_9 = var_6.validate(var_8)
    var_10 = bool(var_9 == {'child_key': 'VALUE'})
    assert var_10 is True

import typesystem.base as module_1


def test_case_0():
    var_0 = False
    var_1 = module_0.Field(read_only=var_0)
    var_2 = 'Child error'
    var_3 = 'child_error'
    var_4 = []
    var_5 = module_1.Message(text=var_2, code=var_3, index=var_4)
    var_6 = [var_5]
    var_7 = module_1.ValidationError(messages=var_6)
    var_8 = None
    var_9 = (var_8, var_7)
    var_10 = 'child_key'
    var_11 = {var_10: var_1}
    var_12 = {}
    var_13 = module_2.Schema(var_11, **var_12)
    var_14 = 'child_key'
    var_15 = 'value'
    var_16 = {var_14: var_15}
    var_17 = var_13.validate(var_16)
    var_18 = bool(False)
    assert var_18 is True

import typesystem.schemas as module_1


def test_case_0():
    var_0 = False
    var_1 = module_0.Field(read_only=var_0)
    var_2 = 1
    var_3 = None
    var_4 = module_0.Field(read_only=var_0)
    var_5 = True
    var_6 = 100
    var_7 = 'field1'
    var_8 = 'field2'
    var_9 = {var_7: var_1, var_8: var_4}
    var_10 = {}
    var_11 = module_1.Schema(var_9, **var_10)
    var_12 = 5
    var_13 = {var_7: var_12}
    var_14 = var_11.validate(var_13)
    var_15 = bool(var_14 == {'field1': 6, 'field2': 100})
    assert var_15 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_validate_with_child_schema_error. Retrieved 5/29 statements.


import typesystem.base as module_0


def test_case_0():
    var_0 = []
    var_1 = module_0.ValidationError(messages=var_0)
    var_2 = 'key'
    var_3 = 'some_value'
    var_4 = {var_2: var_3}



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validate_collects_child_field_validation_errors. Retrieved 14/18 statements.
# Partially parsed test_validate_handles_nested_errors_with_prefix. Retrieved 15/19 statements.


import typesystem.fields as module_0


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


def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 'not a dict'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


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


def test_case_0():
    var_0 = True
    var_1 = module_0.Field(read_only=var_0)
    var_2 = 'read_only_field'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'ignored'
    var_7 = {var_2: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = bool(var_8 == {})
    assert var_9 is True


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

import typesystem.base as module_1


def test_case_0():
    var_0 = module_0.Field()
    var_1 = ()
    var_2 = 'error'
    var_3 = []
    var_4 = module_1.Message(text=var_2, code=var_2, index=var_3)
    var_5 = [var_4]
    var_6 = module_1.ValidationError(messages=var_5)
    var_7 = 'child'
    var_8 = {var_7: var_0}
    var_9 = {}
    var_10 = module_2.Schema(var_8, **var_9)
    var_11 = 'child'
    var_12 = 'value'
    var_13 = {var_11: var_12}
    var_14 = var_10.validate(var_13)
    var_15 = bool(False)
    assert var_15 is True

import typesystem.schemas as module_1


def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'default2'
    var_2 = module_0.Field(default=var_1)
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_0, var_4: var_2}
    var_6 = {}
    var_7 = module_1.Schema(var_5, **var_6)
    var_8 = 'value1'
    var_9 = {var_3: var_8}
    var_10 = var_7.validate(var_9)
    var_11 = bool(var_10 == {'key1': 'value1', 'key2': 'default2'})
    assert var_11 is True

import typesystem.base as module_1


def test_case_0():
    var_0 = module_0.Field()
    var_1 = ()
    var_2 = 'child error'
    var_3 = 'child_error'
    var_4 = []
    var_5 = module_1.Message(text=var_2, code=var_3, index=var_4)
    var_6 = [var_5]
    var_7 = module_1.ValidationError(messages=var_6)
    var_8 = 'nested'
    var_9 = {var_8: var_0}
    var_10 = {}
    var_11 = module_2.Schema(var_9, **var_10)
    var_12 = 'nested'
    var_13 = 'value'
    var_14 = {var_12: var_13}
    var_15 = var_11.validate(var_14)
    var_16 = bool(False)
    assert var_16 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_validate_with_valid_child_field. Retrieved 8/9 statements.


import typesystem.schemas as module_1


def test_case_0():
    var_0 = module_0.Field()
    var_1 = None
    var_2 = 'key'
    var_3 = {var_2: var_0}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'valid'
    var_7 = {var_2: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = bool(var_8 == {'key': 'valid'})
    assert var_9 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_validate_with_child_schema_no_error. Retrieved 4/13 statements.


def test_case_0():
    var_0 = False
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_validate_with_child_schema_no_error. Retrieved 8/9 statements.



def test_case_0():
    var_0 = module_0.Field()
    var_1 = None
    var_2 = 'key'
    var_3 = {var_2: var_0}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'value'
    var_7 = {var_2: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = bool(var_8 == {'key': 'value'})
    assert var_9 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_validate_collects_child_field_validation_errors. Retrieved 14/18 statements.
# Partially parsed test_validate_combines_multiple_errors. Retrieved 8/9 statements.



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


def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 'not a dict'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


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


def test_case_0():
    var_0 = 'default_value'
    var_1 = module_0.Field(default=var_0)
    var_2 = 'optional_key'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = {}
    var_7 = var_5.validate(var_6)
    var_8 = bool(var_7 == {'optional_key': 'default_value'})
    assert var_8 is True


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


def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'child_key'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'child_value'
    var_6 = {var_1: var_5}
    var_7 = var_4.validate(var_6)
    var_8 = bool(var_7 == {'child_key': 'child_value'})
    assert var_8 is True

import typesystem.base as module_1


def test_case_0():
    var_0 = module_0.Field()
    var_1 = ()
    var_2 = 'error'
    var_3 = []
    var_4 = module_1.Message(text=var_2, code=var_2, index=var_3)
    var_5 = [var_4]
    var_6 = module_1.ValidationError(messages=var_5)
    var_7 = 'child_key'
    var_8 = {var_7: var_0}
    var_9 = {}
    var_10 = module_2.Schema(var_8, **var_9)
    var_11 = 'child_key'
    var_12 = 'invalid'
    var_13 = {var_11: var_12}
    var_14 = var_10.validate(var_13)
    var_15 = bool(False)
    assert var_15 is True

import typesystem.schemas as module_1


def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'default'
    var_2 = module_0.Field(default=var_1)
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_0, var_4: var_2}
    var_6 = {}
    var_7 = module_1.Schema(var_5, **var_6)
    var_8 = 'value1'
    var_9 = {var_3: var_8}
    var_10 = var_7.validate(var_9)
    var_11 = bool(var_10 == {'key1': 'value1', 'key2': 'default'})
    assert var_11 is True


def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'required_key'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 123
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'invalid_key'
    var_11 = 'required'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_validate_with_error_in_child_schema. Retrieved 16/20 statements.


import typesystem.base as module_1


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



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_validate_collects_child_field_validation_errors. Retrieved 15/19 statements.
# Partially parsed test_validate_aggregates_multiple_errors. Retrieved 8/9 statements.


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


def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 'not a dict'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


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
    var_9 = bool(var_8 == {})
    assert var_9 is True


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

import typesystem.base as module_1


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

import typesystem.schemas as module_1


def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'default2'
    var_2 = module_0.Field(default=var_1)
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_0, var_4: var_2}
    var_6 = {}
    var_7 = module_1.Schema(var_5, **var_6)
    var_8 = 'value1'
    var_9 = {var_3: var_8}
    var_10 = var_7.validate(var_9)
    var_11 = bool(var_10 == {'key1': 'value1', 'key2': 'default2'})
    assert var_11 is True


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
    var_10 = 'invalid_key'
    var_11 = 'required'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_validate_with_no_error_from_child_schema. Retrieved 4/15 statements.


def test_case_0():
    var_0 = False
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_validate_with_child_schema_error. Retrieved 4/19 statements.


def test_case_0():
    var_0 = False
    var_1 = 'key'
    var_2 = 'invalid'
    var_3 = {var_1: var_2}



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_validate_with_multiple_errors. Retrieved 8/9 statements.



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


def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 'not a dict'
    var_4 = var_2.validate(var_3)
    var_5 = bool(False)
    assert var_5 is True


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


def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = 'req1'
    var_3 = 'req2'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = {}
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'required'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_validate_with_error_in_child_schema. Retrieved 4/14 statements.


def test_case_0():
    var_0 = False
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}



