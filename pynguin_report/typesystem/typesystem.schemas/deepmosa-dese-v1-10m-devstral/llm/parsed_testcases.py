####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
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
    var_2 = 'required_field'
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
    var_5 = 'other_field'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)

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
    var_1 = 'field'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = 'value'
    var_5 = {var_1: var_4}
    var_6 = var_3.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'child_field'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = 'child_field'
    var_5 = 'invalid_value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_validate_with_child_validation_error. Retrieved 14/16 statements.


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
    var_0 = True
    var_1 = module_0.Field()
    var_2 = 'required_field'
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
    var_1 = None
    var_2 = 'error'
    var_3 = module_1.Message(text=var_2, code=var_2)
    var_4 = [var_3]
    var_5 = module_1.ValidationError(messages=var_4)
    var_6 = (var_1, var_5)
    var_7 = 'child_field'
    var_8 = {var_7: var_0}
    var_9 = module_2.Schema(var_8)
    var_10 = 'child_field'
    var_11 = 'invalid'
    var_12 = {var_10: var_11}
    var_13 = var_9.validate(var_12)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_with_child_validation_error. Retrieved 14/16 statements.


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
    var_0 = True
    var_1 = module_0.Field()
    var_2 = 'required_field'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

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
    var_5 = {}
    var_6 = var_4.validate(var_5)

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
import typesystem.base as module_1
import typesystem.schemas as module_2

def test_case_0():
    var_0 = module_0.Field()
    var_1 = None
    var_2 = 'error'
    var_3 = module_1.Message(text=var_2, code=var_2)
    var_4 = [var_3]
    var_5 = module_1.ValidationError(messages=var_4)
    var_6 = (var_1, var_5)
    var_7 = 'child_field'
    var_8 = {var_7: var_0}
    var_9 = module_2.Schema(var_8)
    var_10 = 'child_field'
    var_11 = 'invalid'
    var_12 = {var_10: var_11}
    var_13 = var_9.validate(var_12)



# Parsed testcases at query #4
#--------------------------




import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 123
    var_5 = {var_0: var_4}
    var_6 = var_3.validate(var_5)



# Parsed testcases at query #5
#--------------------------




import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_1.Reference(var_0, var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = module_1.Reference(var_0, var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Reference(var_0, var_2)
    var_4 = 'valid_value'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'valid_value'



# Parsed testcases at query #6
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_validate_with_child_validation_error. Retrieved 14/16 statements.


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
    var_2 = 'required_field'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

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
    var_5 = {}
    var_6 = var_4.validate(var_5)

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
import typesystem.base as module_1
import typesystem.schemas as module_2

def test_case_0():
    var_0 = module_0.Field()
    var_1 = None
    var_2 = 'error'
    var_3 = module_1.Message(text=var_2, code=var_2)
    var_4 = [var_3]
    var_5 = module_1.ValidationError(messages=var_4)
    var_6 = (var_1, var_5)
    var_7 = 'child_field'
    var_8 = {var_7: var_0}
    var_9 = module_2.Schema(var_8)
    var_10 = 'child_field'
    var_11 = 'value'
    var_12 = {var_10: var_11}
    var_13 = var_9.validate(var_12)



# Parsed testcases at query #8
#--------------------------




import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = 'key'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)
    var_5 = None
    var_6 = {var_2: var_5}
    var_7 = var_4.validate(var_6)



# Parsed testcases at query #9
#--------------------------




import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_1.Reference(var_0, var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = module_1.Reference(var_0, var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = module_1.Reference(var_0, var_2)
    var_5 = 'valid_value'
    var_6 = var_4.validate(var_5)
    assert var_6 == 'valid_value'



# Parsed testcases at query #10
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = True
    var_3 = module_0.Reference(var_0, var_1)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None



# Parsed testcases at query #11
#--------------------------




import typesystem.schemas as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'Error'
    var_6 = 'error'
    var_7 = [var_2]
    var_8 = module_1.Message(text=var_5, code=var_6, index=var_7)
    var_9 = [var_8]
    var_10 = module_1.ValidationError(messages=var_9)
    var_11 = var_1.validate(var_4)
    assert var_11 is None



# Parsed testcases at query #12
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = True
    var_3 = module_0.Reference(var_0, var_1)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None



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
    var_0 = True
    var_1 = module_0.Field()
    var_2 = 'required_field'
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
    var_1 = 'field'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = 'value'
    var_5 = {var_1: var_4}
    var_6 = var_3.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'child_field'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = 'child_field'
    var_5 = 'invalid_value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_validate_error_raises_validation_error. Retrieved 11/13 statements.


import typesystem.schemas as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'error'
    var_6 = [var_2]
    var_7 = module_1.Message(text=var_5, code=var_5, index=var_6)
    var_8 = [var_7]
    var_9 = module_1.ValidationError(messages=var_8)
    var_10 = var_1.validate



# Parsed testcases at query #15
#--------------------------




import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_1.Reference(var_0, var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = module_1.Reference(var_0, var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Reference(var_0, var_2)
    var_4 = 'test_value'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'test_value'



# Parsed testcases at query #16
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
    var_0 = True
    var_1 = module_0.Field()
    var_2 = 'required_field'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

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
    var_5 = {}
    var_6 = var_4.validate(var_5)

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
    var_1 = 'child_field'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = 'child_field'
    var_5 = 'invalid_value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)



# Parsed testcases at query #17
#--------------------------




import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_1.Reference(var_0, var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = module_1.Reference(var_0, var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Reference(var_0, var_2)
    var_4 = 'valid_value'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'valid_value'



# Parsed testcases at query #18
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None



# Parsed testcases at query #19
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = True
    var_3 = module_0.Reference(var_0, var_1)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None



# Parsed testcases at query #20
#--------------------------




import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'key'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'key'
    var_5 = 'invalid_value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)



# Parsed testcases at query #21
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None



# Parsed testcases at query #22
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = True
    var_3 = module_0.Reference(var_0, var_1)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None



# Parsed testcases at query #23
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_validate_error_handling. Retrieved 10/11 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'key'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'invalid_value'
    var_5 = {var_0: var_4}
    var_6 = None
    var_7 = 'error_message'
    var_8 = (var_6, var_7)
    var_9 = var_3.validate(var_5)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
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
    var_0 = True
    var_1 = module_0.Field()
    var_2 = 'required_field'
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
    var_1 = 'field'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = 'value'
    var_5 = {var_1: var_4}
    var_6 = var_3.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'child_field'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = 'child_field'
    var_5 = 'invalid_value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)



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
    var_0 = 'required_field'
    var_1 = True
    var_2 = module_0.Field()
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'read_only_field'
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
    var_0 = 'default_field'
    var_1 = 'default'
    var_2 = module_0.Field(default=var_1)
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

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
    var_7 = 'value2'
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = var_5.validate(var_8)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_with_child_validation_error. Retrieved 15/17 statements.


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
    var_0 = True
    var_1 = module_0.Field()
    var_2 = 'required_field'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

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
    var_5 = {}
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = module_0.Field(default=var_0)
    var_2 = 'field'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.base as module_1
import typesystem.schemas as module_2

def test_case_0():
    var_0 = module_0.Field()
    var_1 = None
    var_2 = 'error'
    var_3 = 'child_error'
    var_4 = module_1.Message(text=var_2, code=var_3)
    var_5 = [var_4]
    var_6 = module_1.ValidationError(messages=var_5)
    var_7 = (var_1, var_6)
    var_8 = 'child_field'
    var_9 = {var_8: var_0}
    var_10 = module_2.Schema(var_9)
    var_11 = 'child_field'
    var_12 = 'invalid'
    var_13 = {var_11: var_12}
    var_14 = var_10.validate(var_13)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_validate_error_case. Retrieved 8/9 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'key'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = 'invalid_value'
    var_5 = {var_1: var_4}
    var_6 = var_3.fields[var_1]
    var_7 = var_5[var_1]



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_validate_with_valid_value. Retrieved 2/7 statements.


import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = None
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.Reference(var_0, var_2)
    var_5 = var_4.validate(var_1)
    assert var_5 is None

import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = None
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = module_0.Reference(var_0, var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_value'



# Parsed testcases at query #6
#--------------------------




import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_1.Reference(var_0, var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = module_1.Reference(var_0, var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Reference(var_0, var_2)
    var_4 = 'valid_value'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'valid_value'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_validate_with_child_validation_error. Retrieved 14/16 statements.


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
    var_0 = 'required_field'
    var_1 = True
    var_2 = module_0.Field()
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

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
    var_7 = 'value2'
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = var_5.validate(var_8)

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
    var_2 = 'error'
    var_3 = module_1.Message(text=var_2, code=var_2)
    var_4 = [var_3]
    var_5 = module_1.ValidationError(messages=var_4)
    var_6 = (var_1, var_5)
    var_7 = 'child_field'
    var_8 = {var_7: var_0}
    var_9 = module_2.Schema(var_8)
    var_10 = 'child_field'
    var_11 = 'invalid'
    var_12 = {var_10: var_11}
    var_13 = var_9.validate(var_12)



# Parsed testcases at query #8
#--------------------------




import typesystem.schemas as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'error'
    var_6 = []
    var_7 = module_1.Message(text=var_5, code=var_5, index=var_6)
    var_8 = [var_7]
    var_9 = module_1.ValidationError(messages=var_8)
    var_10 = var_1.validate(var_4)



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
    var_0 = True
    var_1 = module_0.Field()
    var_2 = 'required_field'
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
    var_1 = 'field'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = 'value'
    var_5 = {var_1: var_4}
    var_6 = var_3.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'child_field'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    var_4 = 'child_field'
    var_5 = 'invalid_value'
    var_6 = {var_4: var_5}
    var_7 = var_3.validate(var_6)



# Parsed testcases at query #10
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None



# Parsed testcases at query #11
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = True
    var_3 = module_0.Reference(var_0, var_1)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None



# Parsed testcases at query #12
#--------------------------




import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'test'
    var_5 = {var_0: var_4}
    var_6 = var_3.validate(var_5)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_validate_error_path_with_error. Retrieved 13/15 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = 'key'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'invalid_value'
    var_5 = {var_0: var_4}
    var_6 = None
    var_7 = 'error'
    var_8 = module_2.Message(text=var_7, code=var_7)
    var_9 = [var_8]
    var_10 = module_2.ValidationError(messages=var_9)
    var_11 = (var_6, var_10)
    var_12 = var_3.validate(var_5)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_validate_with_non_none_value. Retrieved 6/7 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'some_type'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_1.Reference(var_0, var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'some_type'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = module_1.Reference(var_0, var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'some_type'
    var_2 = {var_1: var_0}
    var_3 = module_1.Reference(var_1, var_2)
    var_4 = 'test'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'TEST'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_validate_with_child_validation_error. Retrieved 15/17 statements.


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
    var_0 = True
    var_1 = module_0.Field()
    var_2 = 'required_field'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'key'
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
    var_2 = 'read_only'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = module_0.Field(default=var_0)
    var_2 = 'default_field'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.base as module_1
import typesystem.schemas as module_2

def test_case_0():
    var_0 = module_0.Field()
    var_1 = None
    var_2 = 'error'
    var_3 = 'child_error'
    var_4 = module_1.Message(text=var_2, code=var_3)
    var_5 = [var_4]
    var_6 = module_1.ValidationError(messages=var_5)
    var_7 = (var_1, var_6)
    var_8 = 'child'
    var_9 = {var_8: var_0}
    var_10 = module_2.Schema(var_9)
    var_11 = 'child'
    var_12 = 'invalid'
    var_13 = {var_11: var_12}
    var_14 = var_10.validate(var_13)



# Parsed testcases at query #16
#--------------------------




import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'key'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'valid_value'
    var_5 = {var_0: var_4}
    var_6 = var_3.validate(var_5)



# Parsed testcases at query #17
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



# Parsed testcases at query #18
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_validate_with_valid_child_schema. Retrieved 4/18 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = False
    var_2 = 'valid_value'
    var_3 = {var_0: var_2}



# Parsed testcases at query #20
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = True
    var_3 = module_0.Reference(var_0, var_1)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_validate_with_child_validation_error. Retrieved 14/16 statements.


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
    var_0 = 'required_field'
    var_1 = True
    var_2 = module_0.Field()
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'read_only_field'
    var_1 = True
    var_2 = module_0.Field(read_only=var_1)
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'default_field'
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
    var_1 = 'field2'
    var_2 = module_0.Field()
    var_3 = module_0.Field()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_1.Schema(var_4)
    var_6 = 'value1'
    var_7 = 'value2'
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = var_5.validate(var_8)

import typesystem.fields as module_0
import typesystem.base as module_1
import typesystem.schemas as module_2

def test_case_0():
    var_0 = module_0.Field()
    var_1 = None
    var_2 = 'error'
    var_3 = module_1.Message(text=var_2, code=var_2)
    var_4 = [var_3]
    var_5 = module_1.ValidationError(messages=var_4)
    var_6 = (var_1, var_5)
    var_7 = 'child_field'
    var_8 = {var_7: var_0}
    var_9 = module_2.Schema(var_8)
    var_10 = 'child_field'
    var_11 = 'invalid'
    var_12 = {var_10: var_11}
    var_13 = var_9.validate(var_12)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_validate_with_child_validation_error. Retrieved 15/17 statements.


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
    var_0 = 'required_field'
    var_1 = True
    var_2 = module_0.Field()
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = module_0.Field()
    var_3 = module_0.Field()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_1.Schema(var_4)
    var_6 = 'value1'
    var_7 = 'value2'
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = var_5.validate(var_8)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'read_only_field'
    var_1 = True
    var_2 = module_0.Field(read_only=var_1)
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = 'other_field'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field_with_default'
    var_1 = 'default_value'
    var_2 = module_0.Field(default=var_1)
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.base as module_1
import typesystem.schemas as module_2

def test_case_0():
    var_0 = module_0.Field()
    var_1 = None
    var_2 = 'error'
    var_3 = 'child_error'
    var_4 = module_1.Message(text=var_2, code=var_3)
    var_5 = [var_4]
    var_6 = module_1.ValidationError(messages=var_5)
    var_7 = (var_1, var_6)
    var_8 = 'child_field'
    var_9 = {var_8: var_0}
    var_10 = module_2.Schema(var_9)
    var_11 = 'child_field'
    var_12 = 'invalid'
    var_13 = {var_11: var_12}
    var_14 = var_10.validate(var_13)



# Parsed testcases at query #23
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_validate_with_error_in_child_schema. Retrieved 14/16 statements.


import typesystem.fields as module_0
import typesystem.base as module_1
import typesystem.schemas as module_2

def test_case_0():
    var_0 = module_0.Field()
    var_1 = None
    var_2 = 'error'
    var_3 = module_1.Message(text=var_2, code=var_2)
    var_4 = [var_3]
    var_5 = module_1.ValidationError(messages=var_4)
    var_6 = (var_1, var_5)
    var_7 = 'key'
    var_8 = {var_7: var_0}
    var_9 = module_2.Schema(var_8)
    var_10 = 'key'
    var_11 = 'value'
    var_12 = {var_10: var_11}
    var_13 = var_9.validate(var_12)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_validate_returns_none_when_value_is_none_and_allow_null_is_true. Retrieved 6/7 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Reference(var_0, var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_validate_with_non_null_value. Retrieved 2/8 statements.


import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = None
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.Reference(var_0, var_2)
    var_5 = var_4.validate(var_1)
    assert var_5 is None

import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = None
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = module_0.Reference(var_0, var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)

def test_case_0():
    var_0 = 'test'
    var_1 = 'value'



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

# Partially parsed test_validate_no_error_adds_validated_value. Retrieved 11/12 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field1'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'valid_value'
    var_5 = {var_0: var_4}
    var_6 = var_3.fields[var_0]
    var_7 = 'validated_value'
    var_8 = None
    var_9 = (var_7, var_8)
    var_10 = var_3.validate(var_5)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_validate_error_case. Retrieved 13/15 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = 'key'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'invalid_value'
    var_5 = {var_0: var_4}
    var_6 = None
    var_7 = 'error'
    var_8 = module_2.Message(text=var_7, code=var_7)
    var_9 = [var_8]
    var_10 = module_2.ValidationError(messages=var_9)
    var_11 = (var_6, var_10)
    var_12 = var_3.validate(var_5)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_validate_error_path. Retrieved 10/12 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'key'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = 'invalid_value'
    var_5 = {var_0: var_4}
    var_6 = None
    var_7 = 'error'
    var_8 = (var_6, var_7)
    var_9 = var_3.validate(var_5)



# Parsed testcases at query #31
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None



# Parsed testcases at query #32
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {}
    var_2 = True
    var_3 = module_0.Reference(var_0, var_1)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_validate_with_child_validation_error. Retrieved 14/16 statements.


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
    var_0 = 'required_field'
    var_1 = True
    var_2 = module_0.Field()
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = {}
    var_6 = var_4.validate(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'read_only_field'
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
    var_0 = 'default_field'
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
    var_1 = 'field2'
    var_2 = module_0.Field()
    var_3 = module_0.Field()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_1.Schema(var_4)
    var_6 = 'value1'
    var_7 = 'value2'
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = var_5.validate(var_8)

import typesystem.fields as module_0
import typesystem.base as module_1
import typesystem.schemas as module_2

def test_case_0():
    var_0 = module_0.Field()
    var_1 = None
    var_2 = 'error'
    var_3 = module_1.Message(text=var_2, code=var_2)
    var_4 = [var_3]
    var_5 = module_1.ValidationError(messages=var_4)
    var_6 = (var_1, var_5)
    var_7 = 'child_field'
    var_8 = {var_7: var_0}
    var_9 = module_2.Schema(var_8)
    var_10 = 'child_field'
    var_11 = 'invalid'
    var_12 = {var_10: var_11}
    var_13 = var_9.validate(var_12)



