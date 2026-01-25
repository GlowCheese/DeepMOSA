####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'target'
    var_1 = True
    var_2 = module_0.Field(allow_null=var_1)
    var_3 = {var_0: var_2}
    var_4 = module_1.Reference(var_0, var_3)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'target'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Reference(var_0, var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'target'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Reference(var_0, var_2)
    var_4 = 'value'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'value'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'target'
    var_1 = False
    var_2 = module_0.Field(allow_null=var_1)
    var_3 = {var_0: var_2}
    var_4 = module_1.Reference(var_0, var_3)
    var_5 = None
    var_6 = var_4.validate(var_5)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_validate_null. Retrieved 4/5 statements.
# Partially parsed test_validate_null_not_allowed. Retrieved 4/6 statements.
# Partially parsed test_validate_read_only_field. Retrieved 7/8 statements.
# Partially parsed test_validate_field_with_default. Retrieved 6/7 statements.
# Partially parsed test_validate_field_with_validation_error. Retrieved 8/12 statements.


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
    var_2 = 'not a dict'
    var_3 = var_1.validate(var_2)

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = 1
    var_3 = 'invalid key'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'required_field'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = {}
    var_5 = var_3.validate(var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'required_field'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = 'value'
    var_5 = {var_1: var_4}
    var_6 = var_3.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'read_only_field'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = 'value'
    var_5 = {var_1: var_4}
    var_6 = var_3.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'field_with_default'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = {}
    var_5 = var_3.validate(var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'field_with_error'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = 'field_with_error'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = 'field1'
    var_3 = 'field2'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_1.Schema(var_4)
    var_6 = 1
    var_7 = 'field2'
    var_8 = 'invalid key'
    var_9 = 'value'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = var_5.validate(var_10)



# Parsed testcases at query #3
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'example'
    var_2 = True
    var_3 = module_0.Reference(var_1, var_0)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None



# Parsed testcases at query #4
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None



# Parsed testcases at query #5
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'target'
    var_2 = True
    var_3 = module_0.Reference(var_1, var_0)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_predicate_at_line_28_evaluates_to_true. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = True
    var_2 = 'value1'
    var_3 = {var_0: var_2}



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_validate_allows_null_when_allow_null_is_true. Retrieved 3/7 statements.
# Partially parsed test_validate_raises_error_when_value_is_null_and_allow_null_is_false. Retrieved 3/8 statements.
# Partially parsed test_validate_delegates_to_target_when_value_is_not_null. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'target'
    var_1 = True
    var_2 = None

def test_case_0():
    var_0 = 'target'
    var_1 = False
    var_2 = None

def test_case_0():
    var_0 = 'target'
    var_1 = 'value'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validate_with_nested_errors. Retrieved 13/17 statements.


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

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = False
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = 'invalid'
    var_3 = var_1.validate(var_2)

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = 1
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'field'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = {}
    var_5 = var_3.validate(var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'default'
    var_1 = module_0.Field(default=var_0)
    var_2 = 'field'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(read_only=var_0)
    var_2 = 'field'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)
    var_5 = 'value'
    var_6 = {var_2: var_5}
    var_7 = var_4.validate(var_6)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'field'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = 'value'
    var_5 = {var_1: var_4}
    var_6 = var_3.validate(var_5)

import typesystem.fields as module_0
import typesystem.base as module_1
import typesystem.schemas as module_2

def test_case_0():
    var_0 = module_0.Field()
    var_1 = ()
    var_2 = 'error'
    var_3 = module_1.Message(text=var_2, code=var_2)
    var_4 = [var_3]
    var_5 = module_1.ValidationError(messages=var_4)
    var_6 = 'field'
    var_7 = {var_6: var_0}
    var_8 = module_2.Schema(var_7)
    var_9 = 'field'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = var_8.validate(var_11)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_validate_delegates_to_target_validation. Retrieved 2/9 statements.


import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'target'
    var_1 = 'some_definition'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.Reference(var_0, var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None

import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'target'
    var_1 = 'some_definition'
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = module_0.Reference(var_0, var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)

def test_case_0():
    var_0 = 'target'
    var_1 = 'test_value'



# Parsed testcases at query #10
#--------------------------




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
    var_0 = False
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = {}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = {}
    var_2 = module_1.Schema(var_1)
    var_3 = 123
    var_4 = var_2.validate(var_3)

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
    var_0 = module_0.Field()
    var_1 = 'required_field'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = {}
    var_5 = var_3.validate(var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'field'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = 'value'
    var_5 = {var_1: var_4}
    var_6 = var_3.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(read_only=var_0)
    var_2 = 'read_only_field'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)
    var_5 = 'value'
    var_6 = {var_2: var_5}
    var_7 = var_4.validate(var_6)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = module_0.Field(default=var_0)
    var_2 = 'field_with_default'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'nested_field'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = 'nested'
    var_5 = {var_4: var_3}
    var_6 = module_1.Schema(var_5)
    var_7 = 'nested'
    var_8 = {}
    var_9 = {var_7: var_8}
    var_10 = var_6.validate(var_9)



# Parsed testcases at query #11
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = False
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = 'not a dict'
    var_3 = var_1.validate(var_2)

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = 1
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field()
    var_2 = 'test'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'default'
    var_1 = module_0.Field(default=var_0)
    var_2 = 'test'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(read_only=var_0)
    var_2 = 'test'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)
    var_5 = 'value'
    var_6 = {var_2: var_5}
    var_7 = var_4.validate(var_6)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'test'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = 'value'
    var_5 = {var_1: var_4}
    var_6 = var_3.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field()
    var_2 = 'test'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_validate_with_error. Retrieved 14/16 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'test_field'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = 'invalid_value'
    var_5 = {var_1: var_4}
    var_6 = None
    var_7 = 'Invalid value'
    var_8 = [var_1]
    var_9 = module_2.Message(text=var_7, code=var_4, index=var_8)
    var_10 = [var_9]
    var_11 = module_2.ValidationError(messages=var_10)
    var_12 = (var_6, var_11)
    var_13 = var_3.validate(var_5)



# Parsed testcases at query #13
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_validate_with_nested_errors. Retrieved 14/16 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field1'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_1.Schema(var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field1'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = module_1.Schema(var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field1'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'not_a_dict'
    var_5 = var_3.validate(var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field1'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 1
    var_5 = 'invalid_key'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field1'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = {}
    var_5 = var_3.validate(var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field1'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'value1'
    var_5 = {var_0: var_4}
    var_6 = var_3.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field1'
    var_1 = 'default_value'
    var_2 = module_0.Field(default=var_1)
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field1'
    var_1 = True
    var_2 = module_0.Field(read_only=var_1)
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = 'value1'
    var_6 = {var_0: var_5}
    var_7 = var_4.validate(var_6)

import typesystem.fields as module_0
import typesystem.base as module_1
import typesystem.schemas as module_2

def test_case_0():
    var_0 = module_0.Field()
    var_1 = None
    var_2 = 'nested_error'
    var_3 = module_1.Message(text=var_2, code=var_2)
    var_4 = [var_3]
    var_5 = module_1.ValidationError(messages=var_4)
    var_6 = (var_1, var_5)
    var_7 = 'field1'
    var_8 = {var_7: var_0}
    var_9 = module_2.Schema(var_8)
    var_10 = 'field1'
    var_11 = 'value1'
    var_12 = {var_10: var_11}
    var_13 = var_9.validate(var_12)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_validate_with_error_in_child_schema. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'invalid_value'
    var_2 = {var_0: var_1}



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_validate_returns_none_when_value_is_none_and_allow_null_is_true. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = True
    var_2 = None



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_validate_with_error_in_child_schema. Retrieved 4/21 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = True
    var_2 = 'test_value'
    var_3 = {var_0: var_2}



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_validate_returns_target_validate_result_when_value_is_not_null. Retrieved 7/8 statements.


import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'to'
    var_1 = 'definition'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.Reference(var_0, var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None

import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'to'
    var_1 = 'definition'
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = module_0.Reference(var_0, var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)

import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'to'
    var_1 = 'definition'
    var_2 = {var_0: var_1}
    var_3 = lambda value: value
    var_4 = module_0.Reference(var_0, var_2)
    var_5 = 'some_value'
    var_6 = var_4.validate(var_5)
    assert var_6 == 'some_value'



# Parsed testcases at query #19
#--------------------------




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



# Parsed testcases at query #20
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = False
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = 'not a dict'
    var_3 = var_1.validate(var_2)

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = 123
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = {}
    var_5 = var_3.validate(var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = module_0.Field(default=var_0)
    var_2 = 'name'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(read_only=var_0)
    var_2 = 'name'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)
    var_5 = 'value'
    var_6 = {var_2: var_5}
    var_7 = var_4.validate(var_6)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'name'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = 'value'
    var_5 = {var_1: var_4}
    var_6 = var_3.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'inner'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'nested'
    var_5 = {var_4: var_3}
    var_6 = module_1.Schema(var_5)
    var_7 = 'nested'
    var_8 = {}
    var_9 = {var_7: var_8}
    var_10 = var_6.validate(var_9)



# Parsed testcases at query #21
#--------------------------




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



# Parsed testcases at query #22
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'target'
    var_2 = True
    var_3 = module_0.Reference(var_1, var_0)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_validate_raises_error_when_child_field_validation_fails. Retrieved 4/12 statements.


import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = False
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = []
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 1
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = {}
    var_5 = var_3.validate(var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field'
    var_1 = 'default_value'
    var_2 = module_0.Field(default=var_1)
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'value'
    var_5 = {var_0: var_4}
    var_6 = var_3.validate(var_5)

def test_case_0():
    var_0 = 'field'
    var_1 = 'field'
    var_2 = 'value'
    var_3 = {var_1: var_2}

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field'
    var_1 = True
    var_2 = module_0.Field(read_only=var_1)
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = 'value'
    var_6 = {var_0: var_5}
    var_7 = var_4.validate(var_6)



# Parsed testcases at query #24
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_validate_allows_null_when_allow_null_is_true. Retrieved 7/10 statements.
# Partially parsed test_validate_raises_error_when_null_and_allow_null_is_false. Retrieved 7/11 statements.
# Partially parsed test_validate_delegates_to_target_validate. Retrieved 6/9 statements.
# Partially parsed test_validate_raises_error_from_target_validate. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 'example'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = True
    var_6 = None

def test_case_0():
    var_0 = 'example'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = False
    var_6 = None

def test_case_0():
    var_0 = 'example'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test'

def test_case_0():
    var_0 = 'example'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 123



# Parsed testcases at query #26
#--------------------------




import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'key'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'invalid_value'
    var_5 = {var_0: var_4}
    var_6 = var_3.validate(var_5)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_validate_with_nested_field_validation_error. Retrieved 15/17 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_1.Schema(var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = module_1.Schema(var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'not a dict'
    var_5 = var_3.validate(var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 1
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = {}
    var_5 = var_3.validate(var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'John'
    var_5 = {var_0: var_4}
    var_6 = var_3.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'Default Name'
    var_2 = module_0.Field(default=var_1)
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = True
    var_2 = module_0.Field(read_only=var_1)
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = 'John'
    var_6 = {var_0: var_5}
    var_7 = var_4.validate(var_6)

import typesystem.fields as module_0
import typesystem.base as module_1
import typesystem.schemas as module_2

def test_case_0():
    var_0 = module_0.Field()
    var_1 = None
    var_2 = 'Invalid nested field'
    var_3 = 'invalid_nested'
    var_4 = module_1.Message(text=var_2, code=var_3)
    var_5 = [var_4]
    var_6 = module_1.ValidationError(messages=var_5)
    var_7 = (var_1, var_6)
    var_8 = 'nested'
    var_9 = {var_8: var_0}
    var_10 = module_2.Schema(var_9)
    var_11 = 'nested'
    var_12 = 'value'
    var_13 = {var_11: var_12}
    var_14 = var_10.validate(var_13)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.Field()
    var_3 = module_0.Field()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_1.Schema(var_4)
    var_6 = 'age'
    var_7 = 25
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_validate_none_with_allow_null. Retrieved 9/13 statements.
# Partially parsed test_validate_none_without_allow_null. Retrieved 9/14 statements.
# Partially parsed test_validate_non_none_value. Retrieved 9/13 statements.


def test_case_0():
    var_0 = 'target'
    var_1 = 'MockField'
    var_2 = ()
    var_3 = 'validate'
    var_4 = lambda self, value: value
    var_5 = {var_3: var_4}
    var_6 = type(var_1, var_2, var_5)
    var_7 = True
    var_8 = None

def test_case_0():
    var_0 = 'target'
    var_1 = 'MockField'
    var_2 = ()
    var_3 = 'validate'
    var_4 = lambda self, value: value
    var_5 = {var_3: var_4}
    var_6 = type(var_1, var_2, var_5)
    var_7 = False
    var_8 = None

def test_case_0():
    var_0 = 'target'
    var_1 = 'MockField'
    var_2 = ()
    var_3 = 'validate'
    var_4 = lambda self, value: value
    var_5 = {var_3: var_4}
    var_6 = type(var_1, var_2, var_5)
    var_7 = False
    var_8 = 'test'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_allow_null. Retrieved 3/7 statements.
# Partially parsed test_validate_not_allow_null. Retrieved 3/8 statements.
# Partially parsed test_validate_valid_value. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'target'
    var_1 = True
    var_2 = None

def test_case_0():
    var_0 = 'target'
    var_1 = False
    var_2 = None

def test_case_0():
    var_0 = 'target'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #2
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = False
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = 'not a dict'
    var_3 = var_1.validate(var_2)

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = 1
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = {}
    var_5 = var_3.validate(var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field'
    var_1 = 'default_value'
    var_2 = module_0.Field(default=var_1)
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field'
    var_1 = True
    var_2 = module_0.Field(read_only=var_1)
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = 'value'
    var_6 = {var_0: var_5}
    var_7 = var_4.validate(var_6)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'value'
    var_5 = {var_0: var_4}
    var_6 = var_3.validate(var_5)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_serialize_handles_non_dict_object. Retrieved 5/8 statements.
# Partially parsed test_serialize_ignores_missing_attributes_in_object. Retrieved 5/8 statements.


import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = None
    var_3 = var_1.serialize(var_2)
    assert var_3 is None

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'name'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = 'test'
    var_5 = {var_1: var_4}
    var_6 = var_3.serialize(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'name'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = 'test'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'name'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = 'other'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.serialize(var_6)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'name'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = 'value'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = 'name'
    var_3 = 'age'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_1.Schema(var_4)
    var_6 = 'test'
    var_7 = 20
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = var_5.serialize(var_8)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'nested_name'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'nested'
    var_5 = {var_4: var_3}
    var_6 = module_1.Schema(var_5)
    var_7 = 'test'
    var_8 = {var_0: var_7}
    var_9 = {var_4: var_8}
    var_10 = var_6.serialize(var_9)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_serialize_predicate_evaluates_to_false. Retrieved 2/13 statements.


def test_case_0():
    var_0 = 'attr'
    var_1 = 'test_value'



# Parsed testcases at query #5
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'some_key'
    var_2 = True
    var_3 = module_0.Reference(var_1, var_0)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_validate_or_error_returns_error. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'invalid_value'
    var_2 = {var_0: var_1}



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_serialize_returns_dict_for_object_input. Retrieved 4/9 statements.


import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = None
    var_3 = var_1.serialize(var_2)
    assert var_3 is None

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field1'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'value1'
    var_5 = {var_0: var_4}
    var_6 = var_3.serialize(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field1'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = module_0.Field()
    var_3 = module_0.Field()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_1.Schema(var_4)
    var_6 = 'value1'
    var_7 = {var_0: var_6}
    var_8 = var_5.serialize(var_7)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(read_only=var_0)
    var_2 = 'field1'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)
    var_5 = 'value1'
    var_6 = {var_2: var_5}
    var_7 = var_4.serialize(var_6)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_serialize_predicate_evaluates_to_false. Retrieved 4/9 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'test_field'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)



# Parsed testcases at query #9
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_validate_with_child_validation_error. Retrieved 8/12 statements.


import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = False
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = 'not a dict'
    var_3 = var_1.validate(var_2)

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = 1
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'required_field'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = {}
    var_5 = var_3.validate(var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = module_0.Field(default=var_0)
    var_2 = 'optional_field'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(read_only=var_0)
    var_2 = 'read_only_field'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)
    var_5 = 'value'
    var_6 = {var_2: var_5}
    var_7 = var_4.validate(var_6)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'child_field'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = 'child_field'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'valid_field'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = 'value'
    var_5 = {var_1: var_4}
    var_6 = var_3.validate(var_5)



# Parsed testcases at query #11
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = False
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = 'not_a_dict'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 1
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = {}
    var_5 = var_3.validate(var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'value'
    var_5 = {var_0: var_4}
    var_6 = var_3.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field'
    var_1 = 'default_value'
    var_2 = module_0.Field(default=var_1)
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field'
    var_1 = True
    var_2 = module_0.Field(read_only=var_1)
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = 'value'
    var_6 = {var_0: var_5}
    var_7 = var_4.validate(var_6)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field'
    var_1 = False
    var_2 = module_0.Field(allow_null=var_1)
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = 'field'
    var_6 = None
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)



# Parsed testcases at query #12
#--------------------------




import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_1.Schema(var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = module_1.Schema(var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'not a dict'
    var_5 = var_3.validate(var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 1
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'other_field'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'value'
    var_5 = {var_0: var_4}
    var_6 = var_3.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field'
    var_1 = 'default_value'
    var_2 = module_0.Field(default=var_1)
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field'
    var_1 = True
    var_2 = module_0.Field(read_only=var_1)
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = 'value'
    var_6 = {var_0: var_5}
    var_7 = var_4.validate(var_6)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'nested_field'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'field'
    var_5 = {var_4: var_3}
    var_6 = module_1.Schema(var_5)
    var_7 = 'field'
    var_8 = 'nested_field'
    var_9 = None
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = var_6.validate(var_11)



# Parsed testcases at query #13
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = False
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = 'not_a_dict'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'key'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 1
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'key'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = {}
    var_5 = var_3.validate(var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'key'
    var_1 = 'default_value'
    var_2 = module_0.Field(default=var_1)
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'key'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'value'
    var_5 = {var_0: var_4}
    var_6 = var_3.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'key'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'key'
    var_5 = None
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)



# Parsed testcases at query #14
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'some_target'
    var_2 = True
    var_3 = module_0.Reference(var_1, var_0)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None



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

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = False
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = 'not a dict'
    var_3 = var_1.validate(var_2)

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = 1
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field()
    var_2 = 'test'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'default'
    var_1 = module_0.Field(default=var_0)
    var_2 = 'test'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(read_only=var_0)
    var_2 = 'test'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)
    var_5 = 'value'
    var_6 = {var_2: var_5}
    var_7 = var_4.validate(var_6)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'test'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = 'value'
    var_5 = {var_1: var_4}
    var_6 = var_3.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field()
    var_2 = 'test'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)
    var_5 = 'test'
    var_6 = None
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)



# Parsed testcases at query #16
#--------------------------




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



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_validate_delegates_to_target_validation. Retrieved 2/9 statements.


import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'target'
    var_1 = 'some_definition'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.Reference(var_0, var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None

import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'target'
    var_1 = 'some_definition'
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = module_0.Reference(var_0, var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)

def test_case_0():
    var_0 = 'target'
    var_1 = 'value'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_validate_delegates_to_target_validation. Retrieved 2/9 statements.


import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'target'
    var_1 = 'some_definition'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.Reference(var_0, var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None

import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'target'
    var_1 = 'some_definition'
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = module_0.Reference(var_0, var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)

def test_case_0():
    var_0 = 'target'
    var_1 = 'value'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_predicate_at_line_37_evaluates_to_false. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'invalid_value'
    var_2 = {var_0: var_1}



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'valid_value'
    var_2 = {var_0: var_1}



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_validate_with_valid_child_value. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'any_value'
    var_2 = {var_0: var_1}



# Parsed testcases at query #23
#--------------------------




import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'value'
    var_5 = {var_0: var_4}
    var_6 = var_3.validate(var_5)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_validate_with_valid_child_schema. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'any_value'
    var_2 = {var_0: var_1}



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




import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'test'
    var_2 = True
    var_3 = module_0.Reference(var_1, var_0)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'test'
    var_2 = False
    var_3 = module_0.Reference(var_1, var_0)
    var_4 = None
    var_5 = var_3.validate(var_4)



# Parsed testcases at query #27
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_predicate_at_line_37_evaluates_to_false. Retrieved 3/13 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'invalid_value'
    var_2 = {var_0: var_1}



# Parsed testcases at query #29
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'test'
    var_2 = True
    var_3 = module_0.Reference(var_1, var_0)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None



# Parsed testcases at query #30
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = False
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = 'not a dict'
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'valid_key'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 123
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'required_key'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'other_key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'key_with_default'
    var_1 = 'default_value'
    var_2 = module_0.Field(default=var_1)
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'read_only_key'
    var_1 = True
    var_2 = module_0.Field(read_only=var_1)
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = 'value'
    var_6 = {var_0: var_5}
    var_7 = var_4.validate(var_6)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'child_key'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = 'valid_value'
    var_5 = {var_1: var_4}
    var_6 = var_3.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'child_key'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = 'child_key'
    var_5 = 'invalid_value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'key'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'value'
    var_5 = {var_0: var_4}
    var_6 = var_3.validate(var_5)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_validate_calls_target_validate_when_value_is_not_none. Retrieved 2/8 statements.


import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'test'
    var_2 = True
    var_3 = module_0.Reference(var_1, var_0)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'test'
    var_2 = False
    var_3 = module_0.Reference(var_1, var_0)
    var_4 = None
    var_5 = var_3.validate(var_4)

def test_case_0():
    var_0 = 'test'
    var_1 = 'some_value'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_validate_with_valid_child_value. Retrieved 8/9 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = None
    var_2 = 'test'
    var_3 = {var_2: var_0}
    var_4 = module_1.Schema(var_3)
    var_5 = 'valid'
    var_6 = {var_2: var_5}
    var_7 = var_4.validate(var_6)



# Parsed testcases at query #33
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None



# Parsed testcases at query #34
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = False
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = 'not a dict'
    var_3 = var_1.validate(var_2)

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = 1
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'required_field'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = {}
    var_5 = var_3.validate(var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = module_0.Field(default=var_0)
    var_2 = 'optional_field'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(read_only=var_0)
    var_2 = 'read_only_field'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)
    var_5 = 'value'
    var_6 = {var_2: var_5}
    var_7 = var_4.validate(var_6)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'nested_field'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = 'nested_field'
    var_5 = 'invalid'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'valid_field'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = 'valid_value'
    var_5 = {var_1: var_4}
    var_6 = var_3.validate(var_5)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_validate_nested_field_error. Retrieved 10/12 statements.


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

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = False
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = 'not a dict'
    var_3 = var_1.validate(var_2)

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = 1
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'required_field'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = {}
    var_5 = var_3.validate(var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = module_0.Field(default=var_0)
    var_2 = 'field_with_default'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'type'
    var_2 = 'Nested error'
    var_3 = 'nested'
    var_4 = {var_3: var_0}
    var_5 = module_1.Schema(var_4)
    var_6 = 'nested'
    var_7 = 'invalid'
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'valid_field'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = 'valid_value'
    var_5 = {var_1: var_4}
    var_6 = var_3.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(read_only=var_0)
    var_2 = 'read_only_field'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)
    var_5 = 'ignored'
    var_6 = {var_2: var_5}
    var_7 = var_4.validate(var_6)



# Parsed testcases at query #36
#--------------------------




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



# Parsed testcases at query #37
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'test'
    var_2 = True
    var_3 = module_0.Reference(var_1, var_0)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'test'
    var_2 = False
    var_3 = module_0.Reference(var_1, var_0)
    var_4 = None
    var_5 = var_3.validate(var_4)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_validate_with_valid_child_value. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'valid_value'
    var_2 = {var_0: var_1}



