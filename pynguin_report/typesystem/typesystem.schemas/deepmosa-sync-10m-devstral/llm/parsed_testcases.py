####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
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



# Parsed testcases at query #2
#--------------------------




import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = var_8.fields
    var_10 = bool(var_8.fields == var_6)
    assert var_10 is True
    var_11 = var_8.required
    var_12 = bool(var_8.required == ['name', 'age'])
    assert var_12 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = ''
    var_3 = 'default'
    var_4 = {var_3: var_2}
    var_5 = module_0.String(**var_4)
    var_6 = 0
    var_7 = 'default'
    var_8 = {var_7: var_6}
    var_9 = module_0.Integer(**var_8)
    var_10 = {var_0: var_5, var_1: var_9}
    var_11 = {}
    var_12 = module_1.Schema(var_10, **var_11)
    var_13 = var_12.fields
    var_14 = bool(var_12.fields == var_10)
    assert var_14 is True
    var_15 = var_12.required
    var_16 = bool(var_12.required == [])
    assert var_16 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = True
    var_3 = 'read_only'
    var_4 = {var_3: var_2}
    var_5 = module_0.String(**var_4)
    var_6 = {}
    var_7 = module_0.Integer(**var_6)
    var_8 = {var_0: var_5, var_1: var_7}
    var_9 = {}
    var_10 = module_1.Schema(var_8, **var_9)
    var_11 = var_10.fields
    var_12 = bool(var_10.fields == var_8)
    assert var_12 is True
    var_13 = var_10.required
    var_14 = bool(var_10.required == ['age'])
    assert var_14 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = True
    var_8 = 'allow_null'
    var_9 = {var_8: var_7}
    var_10 = module_1.Schema(var_6, **var_9)
    var_11 = var_10.fields
    var_12 = bool(var_10.fields == var_6)
    assert var_12 is True
    var_13 = var_10.allow_null
    assert var_13 is True
    var_14 = var_10.required
    var_15 = bool(var_10.required == ['name', 'age'])
    assert var_15 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = 'Test Schema'
    var_8 = 'A test schema'
    var_9 = 'title'
    var_10 = 'description'
    var_11 = {var_9: var_7, var_10: var_8}
    var_12 = module_1.Schema(var_6, **var_11)
    var_13 = var_12.fields
    var_14 = bool(var_12.fields == var_6)
    assert var_14 is True
    var_15 = var_12.title
    assert var_15 == 'Test Schema'
    var_16 = var_12.description
    assert var_16 == 'A test schema'



# Parsed testcases at query #3
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
    var_8 = bool(var_7 == {'field': 'value'})
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
    var_12 = 'value'
    var_13 = {var_11: var_12}
    var_14 = var_10.validate(var_13)
    var_15 = bool(False)
    assert var_15 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_validate_missing_required_key. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'required_field'
    var_1 = True
    var_2 = {}
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #5
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)



# Parsed testcases at query #6
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



# Parsed testcases at query #7
#--------------------------

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
    var_2 = 'optional'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = {}
    var_7 = var_5.validate(var_6)
    var_8 = bool(var_7 == {'optional': 'default_value'})
    assert var_8 is True

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
    var_10 = {}
    var_11 = module_2.Schema(var_9, **var_10)
    var_12 = 'child'
    var_13 = 'invalid'
    var_14 = {var_12: var_13}
    var_15 = var_11.validate(var_14)
    var_16 = bool(False)
    assert var_16 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_serialize_with_dict_input. Retrieved 14/16 statements.
# Partially parsed test_serialize_with_object_input. Retrieved 10/18 statements.
# Partially parsed test_serialize_with_missing_keys. Retrieved 12/14 statements.


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
    var_1 = module_0.Field()
    var_2 = 2
    var_3 = 'name'
    var_4 = 'age'
    var_5 = {var_3: var_0, var_4: var_1}
    var_6 = {}
    var_7 = module_1.Schema(var_5, **var_6)
    var_8 = 'alice'
    var_9 = 25
    var_10 = {var_3: var_8, var_4: var_9}
    var_11 = 'ALICE'
    var_12 = 50
    var_13 = {var_3: var_11, var_4: var_12}
    var_14 = var_7.serialize(var_10)
    var_15 = bool(var_14 == var_13)
    assert var_15 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = 2
    var_3 = 'name'
    var_4 = 'age'
    var_5 = {var_3: var_0, var_4: var_1}
    var_6 = {}
    var_7 = module_1.Schema(var_5, **var_6)
    var_8 = 'BOB'
    var_9 = 60
    var_10 = {var_3: var_8, var_4: var_9}

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = 2
    var_3 = 'name'
    var_4 = 'age'
    var_5 = {var_3: var_0, var_4: var_1}
    var_6 = {}
    var_7 = module_1.Schema(var_5, **var_6)
    var_8 = 'charlie'
    var_9 = {var_3: var_8}
    var_10 = 'CHARLIE'
    var_11 = {var_3: var_10}
    var_12 = var_7.serialize(var_9)
    var_13 = bool(var_12 == var_11)
    assert var_13 is True



# Parsed testcases at query #9
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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_validate_error_case. Retrieved 10/11 statements.


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
    assert var_10 is None



# Parsed testcases at query #11
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



# Parsed testcases at query #12
#--------------------------




import typesystem.schemas as module_0
import typesystem.base as module_1

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = []
    var_4 = module_1.ValidationError(messages=var_3)
    var_5 = bool(not var_4)
    assert var_5 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_validate_predicate_line_37_false. Retrieved 17/18 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'key'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'invalid_value'
    var_6 = {var_1: var_5}
    var_7 = None
    var_8 = 'error'
    var_9 = module_2.Message(text=var_8, code=var_8)
    var_10 = [var_9]
    var_11 = module_2.ValidationError(messages=var_10)
    var_12 = (var_7, var_11)
    var_13 = var_4.validate(var_6)
    var_14 = [var_1]
    var_15 = module_2.Message(text=var_8, code=var_8, index=var_14)
    var_16 = [var_15]
    var_17 = module_2.ValidationError(messages=var_16)
    var_18 = bool(var_13 == var_17)
    assert var_18 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_validate_error_is_none. Retrieved 3/4 statements.


import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = {}



# Parsed testcases at query #15
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
    var_6 = []
    var_7 = module_1.ValidationError(messages=var_6)
    var_8 = var_2.validate(var_5)
    var_9 = bool(var_8 == var_5)
    assert var_9 is True
    var_10 = bool(not var_7)
    assert var_10 is True



# Parsed testcases at query #16
#--------------------------




import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'key'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'valid_value'
    var_6 = {var_0: var_5}
    var_7 = var_4.validate(var_6)
    var_8 = 'key'
    var_9 = bool('key' in var_7)
    assert var_9 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_validate_error_case. Retrieved 17/18 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'key'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'invalid_value'
    var_6 = {var_1: var_5}
    var_7 = None
    var_8 = 'error'
    var_9 = module_2.Message(text=var_8, code=var_8)
    var_10 = [var_9]
    var_11 = module_2.ValidationError(messages=var_10)
    var_12 = (var_7, var_11)
    var_13 = var_4.validate(var_6)
    var_14 = [var_1]
    var_15 = module_2.Message(text=var_8, code=var_8, index=var_14)
    var_16 = [var_15]
    var_17 = module_2.ValidationError(messages=var_16)
    var_18 = bool(var_13 == var_17)
    assert var_18 is True



# Parsed testcases at query #18
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
    var_1 = 'test_field'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'test_value'
    var_6 = {var_1: var_5}
    var_7 = var_4.validate(var_6)
    var_8 = bool(var_7 == {'test_field': 'test_value'})
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



# Parsed testcases at query #19
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



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_validate_child_schema_error. Retrieved 14/15 statements.


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
    var_9 = (var_7, var_8)
    var_10 = var_4.validate(var_6)
    var_11 = [var_0]
    var_12 = module_2.Message(text=var_8, code=var_8, index=var_11)
    var_13 = [var_12]
    var_14 = module_2.ValidationError(messages=var_13)
    var_15 = bool(var_10 == var_14)
    assert var_15 is True



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




import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'key'
    var_1 = False
    var_2 = module_0.Field(allow_null=var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = None
    var_7 = {var_0: var_6}
    var_8 = var_5.validate(var_7)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
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
    var_0 = 'required_field'
    var_1 = True
    var_2 = {}
    var_3 = bool(False)
    assert var_3 is True

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
    var_0 = 'field_with_default'
    var_1 = 'default_value'
    var_2 = module_0.Field(default=var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = {}
    var_7 = var_5.validate(var_6)
    var_8 = bool(var_7 == {'field_with_default': 'default_value'})
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



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_serialize_with_object_input. Retrieved 4/9 statements.
# Partially parsed test_serialize_with_missing_attribute. Retrieved 4/8 statements.


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
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'test'
    var_6 = {var_0: var_5}
    var_7 = var_4.serialize(var_6)
    var_8 = bool(var_7 == {'name': 'test'})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
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
    var_0 = 'name'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'age'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'user'
    var_6 = {var_5: var_4}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = 25
    var_10 = {var_0: var_9}
    var_11 = {var_5: var_10}
    var_12 = {var_0: var_9}
    var_13 = {var_5: var_12}
    var_14 = var_8.serialize(var_11)
    var_15 = bool(var_14 == var_13)
    assert var_15 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_error_handling. Retrieved 10/11 statements.


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
    var_11 = bool(var_10 == {'key': None})
    assert var_11 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_validate_error_handling. Retrieved 10/11 statements.


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
    var_11 = bool(var_10 == {})
    assert var_11 is True



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_setitem_adds_new_key_value_pair.
# Partially parsed test_setitem_raises_assertion_error_for_existing_key. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_validate_with_null_value_and_allow_null_false. Retrieved 7/10 statements.


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
    var_3 = {}
    var_4 = module_1.Reference(var_0, var_2, **var_3)
    var_5 = 'valid_value'
    var_6 = var_4.validate(var_5)
    assert var_6 == 'valid_value'



# Parsed testcases at query #7
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
    var_6 = None
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_serialize_with_object_input. Retrieved 7/12 statements.
# Partially parsed test_serialize_with_missing_attribute_in_object. Retrieved 4/8 statements.


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
    var_1 = 'key'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'value'
    var_6 = var_0.serialize(var_5)
    var_7 = {var_1: var_6}

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'key'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'other_key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.serialize(var_7)
    var_9 = bool(var_8 == {})
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'key'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = 'key1'
    var_3 = 'key2'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = 'value1'
    var_8 = 'value2'
    var_9 = {var_2: var_7, var_3: var_8}
    var_10 = var_6.serialize(var_9)
    var_11 = var_0.serialize(var_7)
    var_12 = var_1.serialize(var_8)
    var_13 = {var_2: var_11, var_3: var_12}
    var_14 = bool(var_10 == var_13)
    assert var_14 is True



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
    var_6 = None
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)



# Parsed testcases at query #10
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.serialize(var_5)
    var_7 = bool(var_6 == {})
    assert var_7 is True



# Parsed testcases at query #11
#--------------------------

# Failed to parse test_setitem_adds_new_key_value_pair.
# Partially parsed test_setitem_raises_assertion_error_when_key_exists. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #12
#--------------------------

# Failed to parse test_setitem_adds_new_key_value_pair.
# Partially parsed test_setitem_raises_assertion_error_for_existing_key. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = bool(False)
    assert var_4 is True



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
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #14
#--------------------------

# Failed to parse test_setitem_raises_assertion_error_when_key_exists.




# Parsed testcases at query #15
#--------------------------

# Partially parsed test_validate_child_schema_error. Retrieved 10/11 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'key'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = None
    var_6 = True
    var_7 = (var_5, var_6)
    var_8 = 'value'
    var_9 = {var_0: var_8}
    var_10 = var_4.validate(var_9)
    var_11 = bool(var_10 == {'key': 'value'})
    assert var_11 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_serialize_with_dict_input. Retrieved 9/10 statements.
# Partially parsed test_serialize_with_object_input. Retrieved 6/13 statements.
# Partially parsed test_serialize_with_missing_keys_in_dict. Retrieved 9/10 statements.
# Partially parsed test_serialize_with_missing_attributes_in_object. Retrieved 6/12 statements.


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
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_0}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = 1
    var_8 = {var_2: var_7, var_3: var_1}
    var_9 = var_6.serialize(var_8)
    var_10 = bool(var_9 == {'a': 2, 'b': 4})
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_0}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_0}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = 1
    var_8 = {var_2: var_7}
    var_9 = var_6.serialize(var_8)
    var_10 = bool(var_9 == {'a': 2})
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2: var_0, var_3: var_0}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_validate_adds_validated_child_value_when_no_error. Retrieved 3/17 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_serialize_with_non_mapping_input. Retrieved 5/11 statements.


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
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = {var_1: var_0, var_2: var_0}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'value1'
    var_7 = 'value2'
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = var_5.serialize(var_8)
    var_10 = bool(var_9 == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = {var_1: var_0, var_2: var_0}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = {var_1: var_0, var_2: var_0}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'value1'
    var_7 = {var_1: var_6}
    var_8 = var_5.serialize(var_7)
    var_9 = bool(var_8 == {'key1': 'value1'})
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'inner_key'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'outer_key'
    var_6 = {var_5: var_4}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = 'inner_value'
    var_10 = {var_1: var_9}
    var_11 = {var_5: var_10}
    var_12 = var_8.serialize(var_11)
    var_13 = bool(var_12 == {'outer_key': {'inner_key': 'inner_value'}})
    assert var_13 is True



# Parsed testcases at query #19
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
    var_3 = 1
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)
    var_7 = bool(False)
    assert var_7 is True

def test_case_0():
    var_0 = 'required_field'
    var_1 = True
    var_2 = {}
    var_3 = bool(False)
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'read_only_field'
    var_1 = True
    var_2 = module_0.Field(read_only=var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'value'
    var_7 = {var_0: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = 'read_only_field'
    var_10 = bool('read_only_field' not in var_8)
    assert var_10 is True

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
    var_8 = var_7['default_field']
    assert var_8 == 'default_value'

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
    var_13 = 'invalid_value'
    var_14 = {var_12: var_13}
    var_15 = var_11.validate(var_14)
    var_16 = bool(False)
    assert var_16 is True



# Parsed testcases at query #20
#--------------------------

# Failed to parse test_setitem_adds_new_key_value_pair.
# Partially parsed test_setitem_raises_assertion_error_for_existing_key. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_validate_without_error_adds_value_to_validated. Retrieved 7/9 statements.


import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = None
    var_4 = 'test'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = var_2.validate(var_6)
    var_8 = bool(var_7 == {'test': 'value'})
    assert var_8 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_serialize_with_mapping_input. Retrieved 5/10 statements.
# Partially parsed test_serialize_with_non_mapping_input. Retrieved 2/12 statements.
# Partially parsed test_serialize_with_missing_keys. Retrieved 4/9 statements.
# Partially parsed test_serialize_with_extra_keys. Retrieved 5/9 statements.
# Partially parsed test_serialize_with_nested_schema. Retrieved 9/17 statements.


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

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'Alice'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'Charlie'
    var_3 = {var_0: var_2}

def test_case_0():
    var_0 = 'name'
    var_1 = 'extra'
    var_2 = 'Dave'
    var_3 = 'ignored'
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'street'
    var_1 = 'city'
    var_2 = 'name'
    var_3 = 'address'
    var_4 = 'Eve'
    var_5 = '123 Main St'
    var_6 = 'Metropolis'
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = {var_2: var_4, var_3: var_7}



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_validate_error_case. Retrieved 10/11 statements.


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
    assert var_10 is None



# Parsed testcases at query #24
#--------------------------

# Failed to parse test_setitem_raises_assertion_error_when_key_exists.




# Parsed testcases at query #25
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = None
    var_4 = var_2.serialize(var_3)
    assert var_4 is None



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_validate_error_path. Retrieved 14/15 statements.


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
    var_9 = (var_7, var_8)
    var_10 = var_4.validate(var_6)
    var_11 = [var_0]
    var_12 = module_2.Message(text=var_8, code=var_7, index=var_11)
    var_13 = [var_12]
    var_14 = module_2.ValidationError(messages=var_13)
    var_15 = bool(var_10 == var_14)
    assert var_15 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_validate_adds_validated_child_value_when_no_error. Retrieved 8/9 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'test'
    var_6 = {var_0: var_5}
    var_7 = None
    var_8 = var_4.validate(var_6)
    var_9 = bool(var_8 == {'name': 'test'})
    assert var_9 is True



# Parsed testcases at query #28
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
    var_0 = module_0.Field()
    var_1 = 'child_field'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = 'child_field'
    var_6 = None
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #29
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = None
    var_4 = var_2.serialize(var_3)
    assert var_4 is None



# Parsed testcases at query #30
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
    var_7 = var_4.validate(var_6)
    var_8 = bool(var_7 == {'key': 'value'})
    assert var_8 is True



# Parsed testcases at query #31
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
    var_3 = 'child_error'
    var_4 = module_1.Message(text=var_2, code=var_3)
    var_5 = [var_4]
    var_6 = module_1.ValidationError(messages=var_5)
    var_7 = (var_1, var_6)
    var_8 = 'child'
    var_9 = {var_8: var_0}
    var_10 = {}
    var_11 = module_2.Schema(var_9, **var_10)
    var_12 = 'child'
    var_13 = 'invalid'
    var_14 = {var_12: var_13}
    var_15 = var_11.validate(var_14)
    var_16 = bool(False)
    assert var_16 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_validate_error_path. Retrieved 10/11 statements.


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
    var_11 = bool(var_10 == {'key': 'invalid_value'})
    assert var_11 is True



# Parsed testcases at query #33
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = None
    var_4 = var_2.serialize(var_3)
    assert var_4 is None



