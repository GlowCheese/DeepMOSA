####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_missing_required_field. Retrieved 3/8 statements.


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

def test_case_0():
    var_0 = True
    var_1 = 'required_field'
    var_2 = {}
    var_3 = bool(False)
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = module_0.Field(default=var_0)
    var_2 = 'optional_field'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = {}
    var_7 = var_5.validate(var_6)
    var_8 = bool(var_7 == {'optional_field': 'default_value'})
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
    var_9 = bool(var_8 == {})
    assert var_9 is True

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
    var_10 = var_6.validate(var_9)
    var_11 = bool(var_10 == {'field1': 'value1', 'field2': 'value2'})
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'child_field'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'child_field'
    var_6 = 'invalid_value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #2
#--------------------------




import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_1.Reference(var_0, var_2, **var_5)
    var_7 = None
    var_8 = var_6.validate(var_7)
    assert var_8 is None

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_1.Reference(var_0, var_2, **var_5)
    var_7 = None
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_1.Reference(var_0, var_2, **var_3)
    var_5 = 'test_value'
    var_6 = var_4.validate(var_5)
    assert var_6 == 'test_value'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_error_path. Retrieved 13/15 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = 'key'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'invalid_value'
    var_6 = {var_0: var_5}
    var_7 = None
    var_8 = 'error'
    var_9 = module_2.Message(text=var_8, code=var_8)
    var_10 = [var_9]
    var_11 = module_2.ValidationError(messages=var_10)
    var_12 = (var_7, var_11)
    var_13 = var_4.validate(var_6)
    var_14 = len(e.messages())
    var_15 = bool(len(e.messages()) > 0)
    assert var_15 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_validate_with_missing_required_field. Retrieved 3/8 statements.
# Partially parsed test_validate_with_child_validation_error. Retrieved 14/16 statements.


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

def test_case_0():
    var_0 = True
    var_1 = 'required_field'
    var_2 = {}
    var_3 = bool(False)
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(read_only=var_0)
    var_2 = 'read_only_field'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'other_field'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = 'read_only_field'
    var_11 = bool('read_only_field' not in var_9)
    assert var_11 is True

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
    var_1 = 'valid_field'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'value'
    var_6 = {var_1: var_5}
    var_7 = var_4.validate(var_6)
    var_8 = bool(var_7 == {'valid_field': 'value'})
    assert var_8 is True

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
    var_9 = {}
    var_10 = module_2.Schema(var_8, **var_9)
    var_11 = 'child_field'
    var_12 = 'invalid'
    var_13 = {var_11: var_12}
    var_14 = var_10.validate(var_13)
    var_15 = bool(False)
    assert var_15 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_validate_with_missing_required_field. Retrieved 3/8 statements.


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
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.Schema(var_0, **var_3)
    var_5 = None
    var_6 = var_4.validate(var_5)

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
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 123
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)

def test_case_0():
    var_0 = 'required_field'
    var_1 = True
    var_2 = {}

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'read_only_field'
    var_1 = True
    var_2 = module_0.Field(read_only=var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = {}
    var_7 = var_5.validate(var_6)
    var_8 = bool(var_7 == {})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'default_field'
    var_1 = 'default_value'
    var_2 = module_0.Field(default=var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = {}
    var_7 = var_5.validate(var_6)
    var_8 = bool(var_7 == {'default_field': 'default_value'})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = module_0.Field()
    var_3 = module_0.Field()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = 'value1'
    var_8 = 'value2'
    var_9 = {var_0: var_7, var_1: var_8}
    var_10 = var_6.validate(var_9)
    var_11 = bool(var_10 == {'field1': 'value1', 'field2': 'value2'})
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'child_field'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'child_field'
    var_6 = 'invalid_value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_validate_error_path. Retrieved 3/4 statements.


import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = {}



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_validate_error_handling. Retrieved 10/11 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'invalid_value'
    var_6 = {var_0: var_5}
    var_7 = None
    var_8 = 'error'
    var_9 = (var_7, var_8)
    var_10 = var_4.validate(var_6)
    assert var_10 is None



# Parsed testcases at query #8
#--------------------------




import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'key'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'invalid_value'
    var_6 = {var_0: var_5}
    var_7 = var_4.validate(var_6)



# Parsed testcases at query #9
#--------------------------




import typesystem.schemas as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = 'Error'
    var_7 = 'error'
    var_8 = []
    var_9 = module_1.Message(text=var_6, code=var_7, index=var_8)
    var_10 = [var_9]
    var_11 = module_1.ValidationError(messages=var_10)
    var_12 = var_2.validate(var_5)
    var_13 = bool(var_12 == {})
    assert var_13 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_validate_with_missing_required_field. Retrieved 3/8 statements.
# Partially parsed test_validate_with_child_validation_error. Retrieved 14/16 statements.


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

def test_case_0():
    var_0 = True
    var_1 = 'required_field'
    var_2 = {}
    var_3 = bool(False)
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'field'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'value'
    var_6 = {var_1: var_5}
    var_7 = var_4.validate(var_6)
    var_8 = bool(var_7 == {'field': 'value'})
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
    var_6 = 'other_field'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = bool(var_9 == {})
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = module_0.Field(default=var_0)
    var_2 = 'field'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = {}
    var_7 = var_5.validate(var_6)
    var_8 = bool(var_7 == {'field': 'default_value'})
    assert var_8 is True

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
    var_9 = {}
    var_10 = module_2.Schema(var_8, **var_9)
    var_11 = 'child_field'
    var_12 = 'invalid'
    var_13 = {var_11: var_12}
    var_14 = var_10.validate(var_13)
    var_15 = bool(False)
    assert var_15 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_with_missing_required_field. Retrieved 3/8 statements.
# Partially parsed test_validate_with_multiple_errors. Retrieved 7/12 statements.


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
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.Schema(var_0, **var_3)
    var_5 = None
    var_6 = var_4.validate(var_5)

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
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 1
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)

def test_case_0():
    var_0 = True
    var_1 = 'required_field'
    var_2 = {}

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
    var_0 = module_0.Field()
    var_1 = 'field'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'value'
    var_6 = {var_1: var_5}
    var_7 = var_4.validate(var_6)
    var_8 = var_7['field']
    assert var_8 == 'value'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'child_field'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'child_field'
    var_6 = 'invalid_value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Field()
    var_2 = 'required_field'
    var_3 = 'child_field'
    var_4 = 'child_field'
    var_5 = 'invalid_value'
    var_6 = {var_4: var_5}



# Parsed testcases at query #2
#--------------------------




import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_1.Reference(var_0, var_2, **var_5)
    var_7 = None
    var_8 = var_6.validate(var_7)
    assert var_8 is None

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_1.Reference(var_0, var_2, **var_5)
    var_7 = None
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_1.Reference(var_0, var_2, **var_3)
    var_5 = 'valid_value'
    var_6 = var_4.validate(var_5)
    assert var_6 == 'valid_value'



# Parsed testcases at query #3
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = None
    var_4 = var_2.serialize(var_3)
    assert var_4 is None

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = {}
    var_4 = var_2.serialize(var_3)
    var_5 = bool(var_4 == {})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'key'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'value'
    var_6 = {var_1: var_5}
    var_7 = var_4.serialize(var_6)
    var_8 = var_0.serialize(var_5)
    var_9 = {var_1: var_8}
    var_10 = bool(var_7 == var_9)
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'attr'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'Obj'
    var_6 = ()
    var_7 = 'value'
    var_8 = {var_1: var_7}
    var_9 = type(var_5, var_6, var_8)
    var_10 = var_9()
    var_11 = var_4.serialize(var_10)
    var_12 = var_0.serialize(var_7)
    var_13 = {var_1: var_12}
    var_14 = bool(var_11 == var_13)
    assert var_14 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'key'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = {}
    var_6 = var_4.serialize(var_5)
    var_7 = bool(var_6 == {})
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'attr'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'Obj'
    var_6 = ()
    var_7 = {}
    var_8 = type(var_5, var_6, var_7)
    var_9 = var_8()
    var_10 = var_4.serialize(var_9)
    var_11 = bool(var_10 == {})
    assert var_11 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_validate_error_case. Retrieved 10/12 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'key'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'invalid_value'
    var_6 = {var_0: var_5}
    var_7 = None
    var_8 = 'error'
    var_9 = (var_7, var_8)
    var_10 = var_4.validate(var_6)
    var_11 = bool(False)
    assert var_11 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_serialize_with_object_input. Retrieved 6/12 statements.


import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = None
    var_4 = var_2.serialize(var_3)
    assert var_4 is None

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = {}
    var_4 = var_2.serialize(var_3)
    var_5 = bool(var_4 == {})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.Field()
    var_3 = module_0.Field()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = 'Alice'
    var_8 = 30
    var_9 = {var_0: var_7, var_1: var_8}
    var_10 = var_6.serialize(var_9)
    var_11 = bool(var_10 == {'name': 'Alice', 'age': 30})
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.Field()
    var_3 = module_0.Field()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.Field()
    var_3 = module_0.Field()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = 'Charlie'
    var_8 = {var_0: var_7}
    var_9 = var_6.serialize(var_8)
    var_10 = bool(var_9 == {'name': 'Charlie'})
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'street'
    var_1 = 'city'
    var_2 = module_0.Field()
    var_3 = module_0.Field()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = 'name'
    var_8 = 'address'
    var_9 = module_0.Field()
    var_10 = {var_7: var_9, var_8: var_6}
    var_11 = {}
    var_12 = module_1.Schema(var_10, **var_11)
    var_13 = 'Dave'
    var_14 = '123 Main'
    var_15 = 'Metropolis'
    var_16 = {var_0: var_14, var_1: var_15}
    var_17 = {var_7: var_13, var_8: var_16}
    var_18 = var_12.serialize(var_17)
    var_19 = bool(var_18 == {'name': 'Dave', 'address': {'street': '123 Main', 'city': 'Metropolis'}})
    assert var_19 is True



# Parsed testcases at query #6
#--------------------------




import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'test'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'valid_value'
    var_6 = {var_1: var_5}
    var_7 = var_4.validate(var_6)
    var_8 = bool(var_7 == {'test': 'valid_value'})
    assert var_8 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_validate_without_error_adds_validated_value. Retrieved 8/9 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'key'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = None
    var_6 = 'value'
    var_7 = {var_0: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(var_8 == {'key': 'value'})
    assert var_9 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_validate_child_schema_without_error. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_validate_with_missing_required_field. Retrieved 3/8 statements.


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
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.Schema(var_0, **var_3)
    var_5 = None
    var_6 = var_4.validate(var_5)

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
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 123
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)

def test_case_0():
    var_0 = True
    var_1 = 'required_field'
    var_2 = {}

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
    var_9 = bool(var_8 == {})
    assert var_9 is True

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
    var_1 = 'field'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'value'
    var_6 = {var_1: var_5}
    var_7 = var_4.validate(var_6)
    var_8 = bool(var_7 == {'field': 'value'})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'child_field'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'child_field'
    var_6 = 'invalid_value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)



# Parsed testcases at query #10
#--------------------------




import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'key'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'value'
    var_6 = {var_0: var_5}
    var_7 = var_4.serialize(var_6)
    var_8 = bool(var_7 == {'key': 'value'})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'key'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'invalid_key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.serialize(var_7)
    var_9 = bool(var_8 == {})
    assert var_9 is True



# Parsed testcases at query #11
#--------------------------




import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'key1'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'key2'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.serialize(var_7)
    var_9 = bool(var_8 == {})
    assert var_9 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_validate_with_missing_required_field. Retrieved 3/8 statements.
# Partially parsed test_validate_with_child_validation_error. Retrieved 14/16 statements.


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

def test_case_0():
    var_0 = True
    var_1 = 'required_field'
    var_2 = {}
    var_3 = bool(False)
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(read_only=var_0)
    var_2 = 'read_only_field'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'other_field'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = 'read_only_field'
    var_11 = bool('read_only_field' not in var_9)
    assert var_11 is True

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
    var_0 = module_0.Field()
    var_1 = 'field'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'value'
    var_6 = {var_1: var_5}
    var_7 = var_4.validate(var_6)
    var_8 = var_7['field']
    assert var_8 == 'value'

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
    var_9 = {}
    var_10 = module_2.Schema(var_8, **var_9)
    var_11 = 'child_field'
    var_12 = 'invalid'
    var_13 = {var_11: var_12}
    var_14 = var_10.validate(var_13)
    var_15 = bool(False)
    assert var_15 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_validate_with_missing_required_field. Retrieved 3/8 statements.


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
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.Schema(var_0, **var_3)
    var_5 = None
    var_6 = var_4.validate(var_5)

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
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 1
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)

def test_case_0():
    var_0 = True
    var_1 = 'required_field'
    var_2 = {}

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
    var_0 = 'default'
    var_1 = module_0.Field(default=var_0)
    var_2 = 'default_field'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = {}
    var_7 = var_5.validate(var_6)
    var_8 = var_7['default_field']
    assert var_8 == 'default'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'field'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'value'
    var_6 = {var_1: var_5}
    var_7 = var_4.validate(var_6)
    var_8 = var_7['field']
    assert var_8 == 'value'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'child_field'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'child_field'
    var_6 = 'invalid'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_validate_with_error_in_child_schema. Retrieved 11/13 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = None
    var_2 = True
    var_3 = (var_1, var_2)
    var_4 = 'key'
    var_5 = {var_4: var_0}
    var_6 = {}
    var_7 = module_1.Schema(var_5, **var_6)
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = var_7.validate(var_10)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_serialize_with_non_mapping_object_missing_attribute. Retrieved 4/6 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = []



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_validate_with_missing_required_field. Retrieved 3/8 statements.
# Partially parsed test_validate_with_child_validation_error. Retrieved 14/16 statements.


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

def test_case_0():
    var_0 = True
    var_1 = 'required_field'
    var_2 = {}
    var_3 = bool(False)
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'key'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'value'
    var_6 = {var_1: var_5}
    var_7 = var_4.validate(var_6)
    var_8 = bool(var_7 == {'key': 'value'})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(read_only=var_0)
    var_2 = 'read_only'
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
    var_2 = 'default_field'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = {}
    var_7 = var_5.validate(var_6)
    var_8 = bool(var_7 == {'default_field': 'default_value'})
    assert var_8 is True

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
    var_7 = 'failing_field'
    var_8 = {var_7: var_0}
    var_9 = {}
    var_10 = module_2.Schema(var_8, **var_9)
    var_11 = 'failing_field'
    var_12 = 'value'
    var_13 = {var_11: var_12}
    var_14 = var_10.validate(var_13)
    var_15 = bool(False)
    assert var_15 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_validate_with_missing_required_field. Retrieved 3/8 statements.


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

def test_case_0():
    var_0 = True
    var_1 = 'required_field'
    var_2 = {}
    var_3 = bool(False)
    assert var_3 is True

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
    var_0 = module_0.Field()
    var_1 = 'field'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'value'
    var_6 = {var_1: var_5}
    var_7 = var_4.validate(var_6)
    var_8 = var_7['field']
    assert var_8 == 'value'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'child_field'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'child_field'
    var_6 = 'invalid_value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_validate_adds_validated_child_value_when_no_error. Retrieved 3/17 statements.


def test_case_0():
    var_0 = 'test_key'
    var_1 = 'test_value'
    var_2 = {var_0: var_1}



# Parsed testcases at query #19
#--------------------------




import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'key'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'invalid_value'
    var_6 = {var_0: var_5}
    var_7 = var_4.validate(var_6)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_validate_with_error_in_child_schema. Retrieved 14/16 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = 'key'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'invalid_value'
    var_6 = {var_0: var_5}
    var_7 = None
    var_8 = 'Error'
    var_9 = 'error'
    var_10 = module_2.Message(text=var_8, code=var_9)
    var_11 = [var_10]
    var_12 = module_2.ValidationError(messages=var_11)
    var_13 = (var_7, var_12)
    var_14 = var_4.validate(var_6)
    var_15 = bool(False)
    assert var_15 is True
    var_16 = len(e.messages())
    assert var_16 == 1
    var_17 = e.messages()[0].code
    assert var_17 == 'error'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_validate_with_missing_required_field. Retrieved 3/8 statements.


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

def test_case_0():
    var_0 = True
    var_1 = 'required_field'
    var_2 = {}
    var_3 = bool(False)
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'field'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'value'
    var_6 = {var_1: var_5}
    var_7 = var_4.validate(var_6)
    var_8 = bool(var_7 == {'field': 'value'})
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
    var_1 = 'child_field'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'child_field'
    var_6 = 'invalid_value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_validate_with_missing_required_field. Retrieved 3/8 statements.
# Partially parsed test_validate_with_child_validation_error. Retrieved 15/17 statements.


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

def test_case_0():
    var_0 = True
    var_1 = 'required_field'
    var_2 = {}
    var_3 = bool(False)
    assert var_3 is True

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
    var_2 = 'default_field'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = {}
    var_7 = var_5.validate(var_6)
    var_8 = var_7['default_field']
    assert var_8 == 'default_value'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'field'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'value'
    var_6 = {var_1: var_5}
    var_7 = var_4.validate(var_6)
    var_8 = var_7['field']
    assert var_8 == 'value'

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
    var_10 = {}
    var_11 = module_2.Schema(var_9, **var_10)
    var_12 = 'child_field'
    var_13 = 'invalid'
    var_14 = {var_12: var_13}
    var_15 = var_11.validate(var_14)
    var_16 = bool(False)
    assert var_16 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_validate_predicate_false. Retrieved 10/12 statements.


import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_1.Field()
    var_7 = None
    var_8 = 'error'
    var_9 = (var_7, var_8)
    var_10 = var_2.validate(var_5)
    var_11 = bool(var_10 == {})
    assert var_11 is True



