####################################################################
#    TEST GENERATION BEGINS (DEEPMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_with_non_string_keys. Retrieved 8/10 statements.
# Partially parsed test_validate_with_missing_required_field. Retrieved 6/8 statements.
# Partially parsed test_validate_with_nested_validation_error. Retrieved 12/14 statements.
# Partially parsed test_validate_with_multiple_errors. Retrieved 10/12 statements.
# Partially parsed test_validate_with_mapping_type. Retrieved 7/11 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = True
    var_5 = 'allow_null'
    var_6 = {var_5: var_4}
    var_7 = module_1.Schema(var_3, **var_6)
    var_8 = None
    var_9 = var_7.validate(var_8)
    assert var_9 is None

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = None
    var_7 = var_5.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = len(e.messages())
    var_10 = bool(len(e.messages()) > 0)
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'not a dict'
    var_7 = var_5.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = len(e.messages())
    var_10 = bool(len(e.messages()) > 0)
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 1
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = {}
    var_7 = var_5.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'John'
    var_7 = {var_0: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = bool(var_8 == {'name': 'John'})
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'extra'
    var_7 = 'John'
    var_8 = 'field'
    var_9 = {var_0: var_7, var_6: var_8}
    var_10 = var_5.validate(var_9)
    var_11 = bool(var_10 == {'name': 'John'})
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'Unknown'
    var_2 = 'default'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(**var_3)
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = module_1.Schema(var_5, **var_6)
    var_8 = {}
    var_9 = var_7.validate(var_8)
    var_10 = bool(var_9 == {'name': 'Unknown'})
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'id'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = True
    var_5 = 'read_only'
    var_6 = {var_5: var_4}
    var_7 = module_0.String(**var_6)
    var_8 = {var_0: var_3, var_1: var_7}
    var_9 = {}
    var_10 = module_1.Schema(var_8, **var_9)
    var_11 = 'John'
    var_12 = '123'
    var_13 = {var_0: var_11, var_1: var_12}
    var_14 = var_10.validate(var_13)
    var_15 = bool(var_14 == {'name': 'John'})
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
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = 'name'
    var_10 = 'age'
    var_11 = 'John'
    var_12 = 'not an integer'
    var_13 = {var_9: var_11, var_10: var_12}
    var_14 = var_8.validate(var_13)
    var_15 = bool(False)
    assert var_15 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'email'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.String(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = 1
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = var_8.validate(var_11)
    var_13 = bool(False)
    assert var_13 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'John'
    var_7 = (var_0, var_6)
    var_8 = [var_7]

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'nickname'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = True
    var_5 = 'allow_null'
    var_6 = {var_5: var_4}
    var_7 = module_0.String(**var_6)
    var_8 = {var_0: var_3, var_1: var_7}
    var_9 = {}
    var_10 = module_1.Schema(var_8, **var_9)
    var_11 = 'John'
    var_12 = {var_0: var_11}
    var_13 = var_10.validate(var_12)
    var_14 = 'nickname'
    var_15 = bool('nickname' not in var_13)
    assert var_15 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_serialize_with_object. Retrieved 8/15 statements.
# Partially parsed test_serialize_with_missing_attributes. Retrieved 7/13 statements.


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
    var_9 = 'John'
    var_10 = 30
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = var_8.serialize(var_11)
    var_13 = bool(var_12 == {'name': 'John', 'age': 30})
    assert var_13 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = None
    var_7 = var_5.serialize(var_6)
    assert var_7 is None

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
    var_9 = 'Jane'
    var_10 = 25

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'email'
    var_3 = {}
    var_4 = module_0.String(**var_3)
    var_5 = {}
    var_6 = module_0.Integer(**var_5)
    var_7 = {}
    var_8 = module_0.String(**var_7)
    var_9 = {var_0: var_4, var_1: var_6, var_2: var_8}
    var_10 = {}
    var_11 = module_1.Schema(var_9, **var_10)
    var_12 = 'Bob'
    var_13 = 35
    var_14 = {var_0: var_12, var_1: var_13}
    var_15 = var_11.serialize(var_14)
    var_16 = bool(var_15 == {'name': 'Bob', 'age': 35})
    assert var_16 is True

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
    var_9 = 'Alice'

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 'name'
    var_4 = 'Test'
    var_5 = {var_3: var_4}
    var_6 = var_2.serialize(var_5)
    var_7 = bool(var_6 == {})
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'street'
    var_1 = 'city'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.String(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = 'name'
    var_10 = 'address'
    var_11 = {}
    var_12 = module_0.String(**var_11)
    var_13 = {var_9: var_12, var_10: var_8}
    var_14 = {}
    var_15 = module_1.Schema(var_13, **var_14)
    var_16 = 'John'
    var_17 = '123 Main St'
    var_18 = 'NYC'
    var_19 = {var_0: var_17, var_1: var_18}
    var_20 = {var_9: var_16, var_10: var_19}
    var_21 = var_15.serialize(var_20)
    var_22 = bool(var_21 == {'name': 'John', 'address': {'street': '123 Main St', 'city': 'NYC'}})
    assert var_22 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_serialize_exception_handler_attribute_error. Retrieved 2/14 statements.


def test_case_0():
    var_0 = 'existing_attr'
    var_1 = 'missing_attr'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_serialize_with_object. Retrieved 8/15 statements.
# Partially parsed test_serialize_missing_attributes. Retrieved 7/13 statements.


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
    var_9 = 'John'
    var_10 = 30
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = var_8.serialize(var_11)
    var_13 = bool(var_12 == {'name': 'John', 'age': 30})
    assert var_13 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = None
    var_7 = var_5.serialize(var_6)
    assert var_7 is None

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
    var_9 = 'Jane'
    var_10 = 25

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'email'
    var_3 = {}
    var_4 = module_0.String(**var_3)
    var_5 = {}
    var_6 = module_0.Integer(**var_5)
    var_7 = {}
    var_8 = module_0.String(**var_7)
    var_9 = {var_0: var_4, var_1: var_6, var_2: var_8}
    var_10 = {}
    var_11 = module_1.Schema(var_9, **var_10)
    var_12 = 'Bob'
    var_13 = 35
    var_14 = {var_0: var_12, var_1: var_13}
    var_15 = var_11.serialize(var_14)
    var_16 = bool(var_15 == {'name': 'Bob', 'age': 35})
    assert var_16 is True

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
    var_9 = 'Alice'

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = 'name'
    var_4 = 'age'
    var_5 = 'Tom'
    var_6 = 40
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = var_2.serialize(var_7)
    var_9 = bool(var_8 == {})
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'street'
    var_1 = 'city'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.String(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = 'name'
    var_10 = 'address'
    var_11 = {}
    var_12 = module_0.String(**var_11)
    var_13 = {var_9: var_12, var_10: var_8}
    var_14 = {}
    var_15 = module_1.Schema(var_13, **var_14)
    var_16 = 'Charlie'
    var_17 = 'Main St'
    var_18 = 'NYC'
    var_19 = {var_0: var_17, var_1: var_18}
    var_20 = {var_9: var_16, var_10: var_19}
    var_21 = var_15.serialize(var_20)
    var_22 = var_21['name']
    assert var_22 == 'Charlie'
    var_23 = var_21['address']
    var_24 = bool(var_21['address'] == {'street': 'Main St', 'city': 'NYC'})
    assert var_24 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_serialize_attribute_error_exception_handling. Retrieved 2/14 statements.


def test_case_0():
    var_0 = 'existing_attr'
    var_1 = 'missing_attr'
    var_2 = 'missing_attr'



# Parsed testcases at query #6
#--------------------------




import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'User'
    var_1 = None
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Reference(var_0, var_2, **var_5)
    var_7 = var_6.validate(var_1)
    assert var_7 is None

import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'User'
    var_1 = None
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Reference(var_0, var_2, **var_5)
    var_7 = None
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'May not be null'

import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'MockTarget'
    var_1 = ()
    var_2 = 'validate'
    var_3 = 2
    var_4 = lambda self, v: v * var_3
    var_5 = {var_2: var_4}
    var_6 = type(var_0, var_1, var_5)
    var_7 = var_6()
    var_8 = 'User'
    var_9 = {var_8: var_7}
    var_10 = False
    var_11 = 'allow_null'
    var_12 = {var_11: var_10}
    var_13 = module_0.Reference(var_8, var_9, **var_12)
    var_14 = 5
    var_15 = var_13.validate(var_14)
    assert var_15 == 10

import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'MockTarget'
    var_1 = ()
    var_2 = 'validate'
    var_3 = 'id'
    var_4 = lambda self, v: {var_3: v}
    var_5 = {var_2: var_4}
    var_6 = type(var_0, var_1, var_5)
    var_7 = var_6()
    var_8 = 'User'
    var_9 = {var_8: var_7}
    var_10 = False
    var_11 = 'allow_null'
    var_12 = {var_11: var_10}
    var_13 = module_0.Reference(var_8, var_9, **var_12)
    var_14 = 123
    var_15 = var_13.validate(var_14)
    var_16 = bool(var_15 == {'id': 123})
    assert var_16 is True

import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'MockTarget'
    var_1 = ()
    var_2 = 'validate'
    var_3 = lambda self, v: v.upper()
    var_4 = {var_2: var_3}
    var_5 = type(var_0, var_1, var_4)
    var_6 = var_5()
    var_7 = 'Status'
    var_8 = {var_7: var_6}
    var_9 = False
    var_10 = 'allow_null'
    var_11 = {var_10: var_9}
    var_12 = module_0.Reference(var_7, var_8, **var_11)
    var_13 = 'active'
    var_14 = var_12.validate(var_13)
    assert var_14 == 'ACTIVE'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_serialize_predicate_line_13_evaluates_to_false. Retrieved 1/12 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'name'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_serialize_with_object. Retrieved 6/13 statements.
# Partially parsed test_serialize_missing_attributes. Retrieved 6/12 statements.
# Partially parsed test_serialize_with_custom_field_serialization. Retrieved 3/11 statements.


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
    var_11 = bool(var_10 == {'key1': 'value1', 'key2': 'value2'})
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'key1'
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
    var_1 = module_0.Field()
    var_2 = 'key1'
    var_3 = 'key2'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)

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
    var_8 = {var_2: var_7}
    var_9 = var_6.serialize(var_8)
    var_10 = bool(var_9 == {'key1': 'value1'})
    assert var_10 is True

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

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'key1'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_1.Schema(var_2, **var_3)
    var_5 = {}
    var_6 = var_4.serialize(var_5)
    var_7 = bool(var_6 == {})
    assert var_7 is True

def test_case_0():
    var_0 = 'key1'
    var_1 = 'hello'
    var_2 = {var_0: var_1}



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_serialize_with_object. Retrieved 8/15 statements.
# Partially parsed test_serialize_missing_attribute. Retrieved 7/13 statements.


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
    var_9 = 'John'
    var_10 = 30
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = var_8.serialize(var_11)
    var_13 = bool(var_12 == {'name': 'John', 'age': 30})
    assert var_13 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = None
    var_7 = var_5.serialize(var_6)
    assert var_7 is None

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
    var_9 = 'Jane'
    var_10 = 25

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'email'
    var_3 = {}
    var_4 = module_0.String(**var_3)
    var_5 = {}
    var_6 = module_0.Integer(**var_5)
    var_7 = {}
    var_8 = module_0.String(**var_7)
    var_9 = {var_0: var_4, var_1: var_6, var_2: var_8}
    var_10 = {}
    var_11 = module_1.Schema(var_9, **var_10)
    var_12 = 'Bob'
    var_13 = 35
    var_14 = {var_0: var_12, var_1: var_13}
    var_15 = var_11.serialize(var_14)
    var_16 = bool(var_15 == {'name': 'Bob', 'age': 35})
    assert var_16 is True
    var_17 = 'email'
    var_18 = bool('email' not in var_15)
    assert var_18 is True

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
    var_9 = 'Alice'
    var_10 = 'age'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = {}
    var_7 = var_5.serialize(var_6)
    var_8 = bool(var_7 == {})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'count'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = 'test'
    var_10 = 42
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = var_8.serialize(var_11)
    var_13 = var_12['name']
    assert var_13 == 'test'
    var_14 = var_12['count']
    assert var_14 == 42



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_serialize_attribute_error_exception_handling. Retrieved 2/14 statements.


def test_case_0():
    var_0 = 'existing_attr'
    var_1 = 'missing_attr'
    var_2 = 'missing_attr'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_serialize_missing_attribute_continues. Retrieved 2/14 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'field2'



####################################################################
#    TEST GENERATION BEGINS (DEEPMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_with_mapping_object. Retrieved 6/10 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = True
    var_5 = 'allow_null'
    var_6 = {var_5: var_4}
    var_7 = module_1.Schema(var_3, **var_6)
    var_8 = None
    var_9 = var_7.validate(var_8)
    assert var_9 is None

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = None
    var_7 = var_5.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'not a dict'
    var_7 = var_5.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 1
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = bool(True)
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = {}
    var_7 = var_5.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'John'
    var_7 = {var_0: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = bool(var_8 == {'name': 'John'})
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(**var_3)
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = module_1.Schema(var_5, **var_6)
    var_8 = {}
    var_9 = var_7.validate(var_8)
    var_10 = bool(var_9 == {})
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'DefaultName'
    var_2 = 'default'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(**var_3)
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = module_1.Schema(var_5, **var_6)
    var_8 = {}
    var_9 = var_7.validate(var_8)
    var_10 = bool(var_9 == {'name': 'DefaultName'})
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'id'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = True
    var_5 = 'read_only'
    var_6 = {var_5: var_4}
    var_7 = module_0.String(**var_6)
    var_8 = {var_0: var_3, var_1: var_7}
    var_9 = {}
    var_10 = module_1.Schema(var_8, **var_9)
    var_11 = 'John'
    var_12 = '123'
    var_13 = {var_0: var_11, var_1: var_12}
    var_14 = var_10.validate(var_13)
    var_15 = bool(var_14 == {'name': 'John'})
    assert var_15 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'extra'
    var_7 = 'John'
    var_8 = 'field'
    var_9 = {var_0: var_7, var_6: var_8}
    var_10 = var_5.validate(var_9)
    var_11 = bool(var_10 == {'name': 'John'})
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'age'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'name'
    var_7 = 'details'
    var_8 = {}
    var_9 = module_0.String(**var_8)
    var_10 = {var_6: var_9, var_7: var_5}
    var_11 = {}
    var_12 = module_1.Schema(var_10, **var_11)
    var_13 = 'John'
    var_14 = '30'
    var_15 = {var_0: var_14}
    var_16 = {var_6: var_13, var_7: var_15}
    var_17 = var_12.validate(var_16)
    var_18 = bool(var_17 == {'name': 'John', 'details': {'age': '30'}})
    assert var_18 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'John'
    var_7 = {var_0: var_6}

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'age'
    var_1 = {}
    var_2 = module_0.Integer(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'name'
    var_7 = 'details'
    var_8 = {}
    var_9 = module_0.String(**var_8)
    var_10 = {var_6: var_9, var_7: var_5}
    var_11 = {}
    var_12 = module_1.Schema(var_10, **var_11)
    var_13 = 'name'
    var_14 = 'details'
    var_15 = 'John'
    var_16 = 'age'
    var_17 = 'not_an_int'
    var_18 = {var_16: var_17}
    var_19 = {var_13: var_15, var_14: var_18}
    var_20 = var_12.validate(var_19)
    var_21 = bool(False)
    assert var_21 is True
    var_22 = bool(True)
    assert var_22 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'email'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.String(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = 'John'
    var_10 = 'john@example.com'
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = var_8.validate(var_11)
    var_13 = bool(var_12 == {'name': 'John', 'email': 'john@example.com'})
    assert var_13 is True

import typesystem.schemas as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.Schema(var_0, **var_1)
    var_3 = {}
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == {})
    assert var_5 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_validate_with_mapping_type. Retrieved 7/11 statements.
# Partially parsed test_validate_with_multiple_errors. Retrieved 8/10 statements.


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = True
    var_5 = 'allow_null'
    var_6 = {var_5: var_4}
    var_7 = module_1.Schema(var_3, **var_6)
    var_8 = None
    var_9 = var_7.validate(var_8)
    assert var_9 is None

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = None
    var_7 = var_5.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'not a dict'
    var_7 = var_5.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 1
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = {}
    var_7 = var_5.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'John'
    var_7 = {var_0: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = bool(var_8 == {'name': 'John'})
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'id'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = True
    var_5 = 'read_only'
    var_6 = {var_5: var_4}
    var_7 = module_0.String(**var_6)
    var_8 = {var_0: var_3, var_1: var_7}
    var_9 = {}
    var_10 = module_1.Schema(var_8, **var_9)
    var_11 = 'John'
    var_12 = '123'
    var_13 = {var_0: var_11, var_1: var_12}
    var_14 = var_10.validate(var_13)
    var_15 = bool(var_14 == {'name': 'John'})
    assert var_15 is True
    var_16 = 'id'
    var_17 = bool('id' not in var_14)
    assert var_17 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'status'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = 'active'
    var_5 = 'default'
    var_6 = {var_5: var_4}
    var_7 = module_0.String(**var_6)
    var_8 = {var_0: var_3, var_1: var_7}
    var_9 = {}
    var_10 = module_1.Schema(var_8, **var_9)
    var_11 = 'John'
    var_12 = {var_0: var_11}
    var_13 = var_10.validate(var_12)
    var_14 = bool(var_13 == {'name': 'John', 'status': 'active'})
    assert var_14 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'age'
    var_1 = {}
    var_2 = module_0.Integer(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'age'
    var_7 = 'not an integer'
    var_8 = {var_6: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = bool(False)
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'John'
    var_7 = (var_0, var_6)
    var_8 = [var_7]

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'email'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.String(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = {}
    var_10 = var_8.validate(var_9)
    var_11 = bool(False)
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'extra'
    var_7 = 'John'
    var_8 = 'field'
    var_9 = {var_0: var_7, var_6: var_8}
    var_10 = var_5.validate(var_9)
    var_11 = bool(var_10 == {'name': 'John'})
    assert var_11 is True
    var_12 = 'extra'
    var_13 = bool('extra' not in var_10)
    assert var_13 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_validate_with_valid_value. Retrieved 2/9 statements.
# Partially parsed test_validate_delegates_to_target. Retrieved 2/11 statements.
# Partially parsed test_validate_with_complex_object. Retrieved 6/13 statements.


import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'User'
    var_1 = None
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Reference(var_0, var_2, **var_5)
    var_7 = var_6.validate(var_1)
    assert var_7 is None

import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'User'
    var_1 = None
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Reference(var_0, var_2, **var_5)
    var_7 = None
    var_8 = var_6.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'May not be null'

def test_case_0():
    var_0 = 'User'
    var_1 = 'test'

def test_case_0():
    var_0 = 'Number'
    var_1 = 5

def test_case_0():
    var_0 = 'User'
    var_1 = 'id'
    var_2 = 'name'
    var_3 = 1
    var_4 = 'John'
    var_5 = {var_1: var_3, var_2: var_4}



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_schema_serialize_with_object. Retrieved 8/15 statements.
# Partially parsed test_schema_serialize_missing_attributes. Retrieved 7/13 statements.


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
    var_9 = 'John'
    var_10 = 30
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = var_8.serialize(var_11)
    var_13 = bool(var_12 == {'name': 'John', 'age': 30})
    assert var_13 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = None
    var_7 = var_5.serialize(var_6)
    assert var_7 is None

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
    var_9 = 'Alice'
    var_10 = 25

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'email'
    var_3 = {}
    var_4 = module_0.String(**var_3)
    var_5 = {}
    var_6 = module_0.Integer(**var_5)
    var_7 = {}
    var_8 = module_0.String(**var_7)
    var_9 = {var_0: var_4, var_1: var_6, var_2: var_8}
    var_10 = {}
    var_11 = module_1.Schema(var_9, **var_10)
    var_12 = 'Bob'
    var_13 = 35
    var_14 = {var_0: var_12, var_1: var_13}
    var_15 = var_11.serialize(var_14)
    var_16 = bool(var_15 == {'name': 'Bob', 'age': 35})
    assert var_16 is True

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
    var_9 = 'Charlie'

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = {}
    var_7 = var_5.serialize(var_6)
    var_8 = bool(var_7 == {})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'street'
    var_1 = 'city'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.String(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = {}
    var_8 = module_1.Schema(var_6, **var_7)
    var_9 = 'name'
    var_10 = 'address'
    var_11 = {}
    var_12 = module_0.String(**var_11)
    var_13 = {var_9: var_12, var_10: var_8}
    var_14 = {}
    var_15 = module_1.Schema(var_13, **var_14)
    var_16 = 'David'
    var_17 = '123 Main St'
    var_18 = 'NYC'
    var_19 = {var_0: var_17, var_1: var_18}
    var_20 = {var_9: var_16, var_10: var_19}
    var_21 = var_15.serialize(var_20)
    var_22 = bool(var_21 == {'name': 'David', 'address': {'street': '123 Main St', 'city': 'NYC'}})
    assert var_22 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_serialize_attribute_error_handling. Retrieved 2/14 statements.


def test_case_0():
    var_0 = 'existing_field'
    var_1 = 'missing_field'
    var_2 = 'missing_field'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_serialize_predicate_line_13_evaluates_to_false. Retrieved 2/17 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'field1'
    var_3 = 'field2'



