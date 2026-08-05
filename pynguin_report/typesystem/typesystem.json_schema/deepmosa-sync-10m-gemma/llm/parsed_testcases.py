####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_all_of_from_json_schema. Retrieved 16/17 statements.
# Partially parsed test_all_of_from_json_schema_empty. Retrieved 7/8 statements.
# Partially parsed test_all_of_from_json_schema_with_ref. Retrieved 10/13 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Definitions(*var_0, **var_1)
    var_3 = 'allOf'
    var_4 = 'default'
    var_5 = 'type'
    var_6 = 'string'
    var_7 = {var_5: var_6}
    var_8 = 'enum'
    var_9 = 'a'
    var_10 = 'b'
    var_11 = [var_9, var_10]
    var_12 = {var_8: var_11}
    var_13 = [var_7, var_12]
    var_14 = {var_3: var_13, var_4: var_9}
    var_15 = module_1.all_of_from_json_schema(var_14, var_2)
    var_16 = var_15.all_of
    var_17 = len(var_16)
    assert var_17 == 2
    var_18 = var_15.default
    assert var_18 == 'a'

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Definitions(*var_0, **var_1)
    var_3 = 'allOf'
    var_4 = []
    var_5 = {var_3: var_4}
    var_6 = module_1.all_of_from_json_schema(var_5, var_2)
    var_7 = var_6.all_of
    var_8 = len(var_7)
    assert var_8 == 0

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Definitions(*var_0, **var_1)
    var_3 = 'allOf'
    var_4 = '$ref'
    var_5 = '#/components/schemas/test'
    var_6 = {var_4: var_5}
    var_7 = [var_6]
    var_8 = {var_3: var_7}
    var_9 = module_1.all_of_from_json_schema(var_8, var_2)
    var_10 = 0
    var_11 = var_9.all_of[var_10]
    var_12 = var_9.all_of[0].to
    assert var_12 == '#/components/schemas/test'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_from_json_schema_type_number. Retrieved 11/12 statements.
# Partially parsed test_from_json_schema_type_integer. Retrieved 9/10 statements.
# Partially parsed test_from_json_schema_type_string. Retrieved 13/14 statements.
# Partially parsed test_from_json_schema_type_boolean. Retrieved 7/8 statements.
# Partially parsed test_from_json_schema_type_array. Retrieved 15/17 statements.
# Partially parsed test_from_json_schema_type_object. Retrieved 18/21 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'minimum'
    var_1 = 'maximum'
    var_2 = 'default'
    var_3 = 1
    var_4 = 10
    var_5 = 5
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = []
    var_8 = {}
    var_9 = module_0.Definitions(*var_7, **var_8)
    var_10 = 'number'
    var_11 = False
    var_12 = module_1.from_json_schema_type(var_6, var_10, var_11, var_9)
    var_13 = var_12.minimum
    assert var_13 == 1
    var_14 = var_12.maximum
    assert var_14 == 10
    var_15 = var_12.default
    assert var_15 == 5

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'exclusiveMinimum'
    var_1 = 'multipleOf'
    var_2 = 0
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = {}
    var_7 = module_0.Definitions(*var_5, **var_6)
    var_8 = 'integer'
    var_9 = True
    var_10 = module_1.from_json_schema_type(var_4, var_8, var_9, var_7)
    var_11 = var_10.exclusive_minimum
    assert var_11 == 0
    var_12 = var_10.multiple_of
    assert var_12 == 2
    var_13 = var_10.allow_null
    assert var_13 is True

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'minLength'
    var_1 = 'maxLength'
    var_2 = 'pattern'
    var_3 = 'format'
    var_4 = 5
    var_5 = 10
    var_6 = '^abc'
    var_7 = 'email'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = []
    var_10 = {}
    var_11 = module_0.Definitions(*var_9, **var_10)
    var_12 = 'string'
    var_13 = False
    var_14 = module_1.from_json_schema_type(var_8, var_12, var_13, var_11)
    var_15 = var_14.min_length
    assert var_15 == 5
    var_16 = var_14.max_length
    assert var_16 == 10
    var_17 = var_14.pattern
    assert var_17 == '^abc'
    var_18 = var_14.format
    assert var_18 == 'email'

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'default'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = {}
    var_5 = module_0.Definitions(*var_3, **var_4)
    var_6 = 'boolean'
    var_7 = False
    var_8 = module_1.from_json_schema_type(var_2, var_6, var_7, var_5)
    var_9 = var_8.default
    assert var_9 is True

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'items'
    var_1 = 'additionalItems'
    var_2 = 'minItems'
    var_3 = 'uniqueItems'
    var_4 = 'type'
    var_5 = 'string'
    var_6 = {var_4: var_5}
    var_7 = False
    var_8 = 1
    var_9 = True
    var_10 = {var_0: var_6, var_1: var_7, var_2: var_8, var_3: var_9}
    var_11 = []
    var_12 = {}
    var_13 = module_0.Definitions(*var_11, **var_12)
    var_14 = 'array'
    var_15 = module_1.from_json_schema_type(var_10, var_14, var_7, var_13)
    var_16 = var_15.items
    var_17 = var_15.additional_items
    assert var_17 is False
    var_18 = var_15.min_items
    assert var_18 == 1
    var_19 = var_15.unique_items
    assert var_19 is True

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'properties'
    var_1 = 'required'
    var_2 = 'additionalProperties'
    var_3 = 'name'
    var_4 = 'type'
    var_5 = 'string'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = [var_3]
    var_9 = 'integer'
    var_10 = {var_4: var_9}
    var_11 = {var_0: var_7, var_1: var_8, var_2: var_10}
    var_12 = []
    var_13 = {}
    var_14 = module_0.Definitions(*var_12, **var_13)
    var_15 = 'object'
    var_16 = False
    var_17 = module_1.from_json_schema_type(var_11, var_15, var_16, var_14)
    var_18 = var_17.properties[var_3]
    var_19 = 'name'
    var_20 = bool('name' in var_17.required)
    assert var_20 is True
    var_21 = var_17.additional_properties



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_to_json_schema_reference. Retrieved 2/17 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 5
    var_2 = 'allow_null'
    var_3 = {var_2: var_0}
    var_4 = module_0.String(min_length=var_1, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['type']
    var_7 = bool(var_5['type'] == ['string', 'null'])
    assert var_7 is True
    var_8 = var_5['minLength']
    assert var_8 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = 'default'
    var_3 = {var_2: var_0}
    var_4 = module_0.Integer(minimum=var_1, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['type']
    assert var_6 == 'integer'
    var_7 = var_5['default']
    assert var_7 == 10
    var_8 = var_5['minimum']
    assert var_8 == 0

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['boolean', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 1
    var_3 = {}
    var_4 = module_0.Array(var_1, min_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['type']
    assert var_6 == 'array'
    var_7 = var_5['minItems']
    assert var_7 == 1
    var_8 = var_5['items']
    var_9 = bool(var_5['items'] == {'type': 'string', 'minLength': 1})
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = [var_0]
    var_8 = {}
    var_9 = module_0.Object(properties=var_6, required=var_7, **var_8)
    var_10 = module_1.to_json_schema(var_9)
    var_11 = var_10['type']
    assert var_11 == 'object'
    var_12 = var_10['properties']['name']
    var_13 = bool(var_10['properties']['name'] == {'type': 'string', 'minLength': 1})
    assert var_13 is True
    var_14 = var_10['properties']['age']
    var_15 = bool(var_10['properties']['age'] == {'type': 'integer'})
    assert var_15 is True
    var_16 = var_10['required']
    var_17 = bool(var_10['required'] == ['name'])
    assert var_17 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'anyOf'
    var_9 = bool('anyOf' in var_7)
    assert var_9 is True
    var_10 = 'anyOf'
    var_11 = var_7[var_10]
    var_12 = len(var_11)
    assert var_12 == 2
    var_13 = var_7['anyOf'][0]['type']
    assert var_13 == 'string'
    var_14 = var_7['anyOf'][1]['type']
    assert var_14 == 'integer'

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'User'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_1.Definitions(*var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = 'components'
    var_9 = bool('components' in var_7)
    assert var_9 is True
    var_10 = 'schemas'
    var_11 = bool('schemas' in var_7['components'])
    assert var_11 is True
    var_12 = var_7['components']['schemas']['User']
    var_13 = bool(var_7['components']['schemas']['User'] == {'type': 'string', 'minLength': 1})
    assert var_13 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'User'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = 'target'
    var_4 = {var_3: var_2}



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_type_from_json_schema_single_type_integer. Retrieved 7/8 statements.
# Partially parsed test_type_from_json_schema_single_type_string. Retrieved 9/10 statements.
# Partially parsed test_type_from_json_schema_with_null. Retrieved 11/13 statements.
# Partially parsed test_type_from_json_schema_number_discards_integer. Retrieved 11/13 statements.
# Partially parsed test_type_from_json_schema_empty_types_with_null. Retrieved 6/7 statements.
# Partially parsed test_type_from_json_schema_boolean_type. Retrieved 7/8 statements.
# Partially parsed test_type_from_json_schema_array_type. Retrieved 7/8 statements.
# Partially parsed test_type_from_json_schema_object_type. Retrieved 8/9 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'minimum'
    var_2 = 'integer'
    var_3 = 5
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = {}
    var_7 = module_0.Definitions(*var_5, **var_6)
    var_8 = module_1.type_from_json_schema(var_4, var_7)
    var_9 = var_8.allow_null
    assert var_9 is False
    var_10 = var_8.minimum
    assert var_10 == 5

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'minLength'
    var_2 = 'maxLength'
    var_3 = 'string'
    var_4 = 3
    var_5 = 10
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = []
    var_8 = {}
    var_9 = module_0.Definitions(*var_7, **var_8)
    var_10 = module_1.type_from_json_schema(var_6, var_9)
    var_11 = var_10.min_length
    assert var_11 == 3
    var_12 = var_10.max_length
    assert var_12 == 10

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = 'null'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = []
    var_6 = {}
    var_7 = module_0.Definitions(*var_5, **var_6)
    var_8 = module_1.type_from_json_schema(var_4, var_7)
    var_9 = var_8.allow_null
    assert var_9 is True
    var_10 = var_8.any_of
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = 0
    var_13 = var_8.any_of[var_12]

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'number'
    var_2 = 'integer'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = []
    var_6 = {}
    var_7 = module_0.Definitions(*var_5, **var_6)
    var_8 = module_1.type_from_json_schema(var_4, var_7)
    var_9 = var_8.any_of
    var_10 = len(var_9)
    assert var_10 == 1
    var_11 = 0
    var_12 = var_8.any_of[var_11]

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'null'
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = []
    var_5 = {}
    var_6 = module_0.Definitions(*var_4, **var_5)
    var_7 = module_1.type_from_json_schema(var_3, var_6)
    var_8 = var_7.const
    assert var_8 is None

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'default'
    var_2 = 'boolean'
    var_3 = True
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = {}
    var_7 = module_0.Definitions(*var_5, **var_6)
    var_8 = module_1.type_from_json_schema(var_4, var_7)
    var_9 = var_8.default
    assert var_9 is True

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'minItems'
    var_2 = 'array'
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = {}
    var_7 = module_0.Definitions(*var_5, **var_6)
    var_8 = module_1.type_from_json_schema(var_4, var_7)
    var_9 = var_8.min_items
    assert var_9 == 2

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'required'
    var_2 = 'object'
    var_3 = 'name'
    var_4 = [var_3]
    var_5 = {var_0: var_2, var_1: var_4}
    var_6 = []
    var_7 = {}
    var_8 = module_0.Definitions(*var_6, **var_7)
    var_9 = module_1.type_from_json_schema(var_5, var_8)
    var_10 = var_9.required
    var_11 = bool(var_9.required == ['name'])
    assert var_11 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_to_json_schema_array_items_is_list. Retrieved 8/33 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = 'mock_module'
    var_5 = {}
    var_6 = {}
    var_7 = module_1.to_json_schema(var_3)
    var_8 = 'items'
    var_9 = bool('items' in var_7)
    assert var_9 is True
    var_10 = 'items'
    var_11 = var_7[var_10]



# Parsed testcases at query #6
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = None
    var_3 = {}
    var_4 = module_1.IfThenElse(var_1, var_2, var_2, **var_3)
    var_5 = module_2.to_json_schema(var_4)
    var_6 = 'if'
    var_7 = bool('if' in var_5)
    assert var_7 is True
    var_8 = 'then'
    var_9 = bool('then' not in var_5)
    assert var_9 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_to_json_schema_reference_predicate_true. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'MySchema'
    var_1 = {}
    var_2 = '$ref'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_to_json_schema_additional_properties_is_not_bool. Retrieved 10/11 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = {}
    var_6 = None
    var_7 = []
    var_8 = {}
    var_9 = module_0.Object(properties=var_4, pattern_properties=var_5, additional_properties=var_3, property_names=var_6, min_properties=var_6, max_properties=var_6, required=var_7, **var_8)
    var_10 = module_1.to_json_schema(var_9)
    var_11 = 'additionalProperties'
    var_12 = bool('additionalProperties' in var_10)
    assert var_12 is True
    var_13 = 'additionalProperties'
    var_14 = var_10[var_13]



# Parsed testcases at query #9
#--------------------------




import re as module_0
import typesystem.fields as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = module_0.compile(var_0)
    var_2 = 'pattern_regex'
    var_3 = {var_2: var_1}
    var_4 = module_1.String(**var_3)
    var_5 = True
    var_6 = 'allow_null'
    var_7 = {var_6: var_5}
    var_8 = module_1.String(**var_7)
    var_9 = '^[0-9]+$'
    var_10 = {var_9: var_8}
    var_11 = {}
    var_12 = module_1.Object(pattern_properties=var_10, **var_11)
    var_13 = module_2.to_json_schema(var_12)
    var_14 = 'patternProperties'
    var_15 = bool('patternProperties' in var_13)
    assert var_15 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_to_json_schema_evaluates_object_branch. Retrieved 7/8 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = {}
    var_3 = None
    var_4 = []
    var_5 = 'allow_null'
    var_6 = {var_5: var_0}
    var_7 = module_0.Object(properties=var_1, pattern_properties=var_2, additional_properties=var_3, property_names=var_3, min_properties=var_3, max_properties=var_3, required=var_4, **var_6)
    var_8 = module_1.to_json_schema(var_7)



# Parsed testcases at query #11
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'date-time'
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(format=var_0, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['format']
    assert var_6 == 'date-time'



# Parsed testcases at query #12
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = False
    var_5 = 'allow_null'
    var_6 = {var_5: var_4}
    var_7 = module_0.Boolean(**var_6)
    var_8 = None
    var_9 = {}
    var_10 = module_1.IfThenElse(var_3, var_7, var_8, **var_9)
    var_11 = module_2.to_json_schema(var_10)
    var_12 = 'else'
    var_13 = bool('else' not in var_11)
    assert var_13 is True



# Parsed testcases at query #13
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True



# Parsed testcases at query #14
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 != True)
    assert var_5 is True



# Parsed testcases at query #15
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = {}
    var_7 = module_0.Boolean(**var_6)
    var_8 = {}
    var_9 = module_1.IfThenElse(var_3, var_5, var_7, **var_8)
    var_10 = module_2.to_json_schema(var_9)
    var_11 = 'else'
    var_12 = bool('else' in var_10)
    assert var_12 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_ref_from_json_schema_success. Retrieved 3/12 statements.
# Partially parsed test_ref_from_json_schema_invalid_ref_prefix. Retrieved 3/11 statements.
# Partially parsed test_ref_from_json_schema_missing_ref_key. Retrieved 3/11 statements.


def test_case_0():
    var_0 = '#/user'
    var_1 = '$ref'
    var_2 = {var_1: var_0}

def test_case_0():
    var_0 = '$ref'
    var_1 = 'user'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'not_a_ref'
    var_1 = '#/user'
    var_2 = {var_0: var_1}



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_from_json_schema_bool_true. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_bool_false. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_ref. Retrieved 5/6 statements.
# Partially parsed test_from_json_schema_enum. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_const. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_all_of. Retrieved 11/12 statements.
# Partially parsed test_from_json_schema_any_of. Retrieved 11/12 statements.
# Partially parsed test_from_json_schema_one_of. Retrieved 11/12 statements.
# Partially parsed test_from_json_schema_not. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_if_then_else. Retrieved 12/13 statements.
# Partially parsed test_from_json_schema_components_definitions. Retrieved 13/14 statements.
# Partially parsed test_from_json_schema_any_fallback. Retrieved 4/5 statements.


import typesystem.json_schema as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.from_json_schema(var_0)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Definitions(*var_0, **var_1)
    var_3 = '$ref'
    var_4 = '#/components/schemas/MyType'
    var_5 = {var_3: var_4}
    var_6 = module_1.from_json_schema(var_5, var_2)
    var_7 = var_6.to
    assert var_7 == '#/components/schemas/MyType'

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'enum'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.from_json_schema(var_4)
    var_6 = var_5.choices
    var_7 = bool(var_5.choices == [('a', 'a'), ('b', 'b')])
    assert var_7 is True

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'const'
    var_1 = 123
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)
    var_4 = var_3.const
    assert var_4 == 123

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'allOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'integer'
    var_5 = {var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = {var_0: var_6}
    var_8 = module_0.from_json_schema(var_7)
    var_9 = var_8.all_of
    var_10 = len(var_9)
    assert var_10 == 2

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'anyOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'integer'
    var_5 = {var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = {var_0: var_6}
    var_8 = module_0.from_json_schema(var_7)
    var_9 = var_8.any_of
    var_10 = len(var_9)
    assert var_10 == 2

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'oneOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'integer'
    var_5 = {var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = {var_0: var_6}
    var_8 = module_0.from_json_schema(var_7)
    var_9 = var_8.one_of
    var_10 = len(var_9)
    assert var_10 == 2

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'not'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.from_json_schema(var_4)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'if'
    var_1 = 'then'
    var_2 = 'else'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 'integer'
    var_7 = {var_3: var_6}
    var_8 = 'boolean'
    var_9 = {var_3: var_8}
    var_10 = {var_0: var_5, var_1: var_7, var_2: var_9}
    var_11 = module_0.from_json_schema(var_10)
    var_12 = var_11.if_clause
    var_13 = bool(var_11.if_clause is not None)
    assert var_13 is True
    var_14 = var_11.then_clause
    var_15 = bool(var_11.then_clause is not None)
    assert var_15 is True
    var_16 = var_11.else_clause
    var_17 = bool(var_11.else_clause is not None)
    assert var_17 is True

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'components'
    var_1 = 'schemas'
    var_2 = 'User'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = []
    var_10 = {}
    var_11 = module_0.Definitions(*var_9, **var_10)
    var_12 = module_1.from_json_schema(var_8, var_11)
    var_13 = '#/components/schemas/User'
    var_14 = bool('#/components/schemas/User' in var_11)
    assert var_14 is True
    var_15 = '#/components/schemas/User'
    var_16 = var_11[var_15]

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'unsupported_key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)



# Parsed testcases at query #18
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'maxLength'
    var_5 = bool('maxLength' in var_3)
    assert var_5 is True
    var_6 = var_3['maxLength']
    assert var_6 == 10



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_to_json_schema_const. Retrieved 1/4 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'string', 'minLength': 1})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': ['string', 'null'], 'minLength': 1})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = {}
    var_3 = module_0.Integer(minimum=var_0, maximum=var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'integer', 'minimum': 0, 'maximum': 10})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'boolean'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'array', 'items': {'type': 'string', 'minLength': 1}, 'uniqueItems': True})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = {}
    var_8 = module_0.Object(properties=var_6, **var_7)
    var_9 = 'type'
    var_10 = 'properties'
    var_11 = 'object'
    var_12 = 'minLength'
    var_13 = 'string'
    var_14 = 1
    var_15 = {var_9: var_13, var_12: var_14}
    var_16 = 'integer'
    var_17 = {var_9: var_16}
    var_18 = {var_0: var_15, var_1: var_17}
    var_19 = {var_9: var_11, var_10: var_18}
    var_20 = module_1.to_json_schema(var_8)
    var_21 = bool(var_20 == var_19)
    assert var_21 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'User'
    var_1 = 'id'
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_0.Object(properties=var_4, **var_5)
    var_7 = {var_0: var_6}
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_1.Definitions(*var_8, **var_9)
    var_11 = module_2.to_json_schema(var_10)
    var_12 = 'components'
    var_13 = bool('components' in var_11)
    assert var_13 is True
    var_14 = 'schemas'
    var_15 = bool('schemas' in var_11['components'])
    assert var_15 is True
    var_16 = var_11['components']['schemas']['User']
    var_17 = bool(var_11['components']['schemas']['User'] == {'type': 'object', 'properties': {'id': {'type': 'integer'}}})
    assert var_17 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'anyOf'
    var_9 = bool('anyOf' in var_7)
    assert var_9 is True
    var_10 = 'type'
    var_11 = 'minLength'
    var_12 = 'string'
    var_13 = 1
    var_14 = {var_10: var_12, var_11: var_13}
    var_15 = bool({'type': 'string', 'minLength': 1} in var_7['anyOf'])
    assert var_15 is True
    var_16 = 'type'
    var_17 = 'integer'
    var_18 = {var_16: var_17}
    var_19 = bool({'type': 'integer'} in var_7['anyOf'])
    assert var_19 is True

def test_case_0():
    var_0 = 'fixed'
    var_1 = 'value'
    var_2 = {var_1: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = (var_3, var_1)
    var_5 = [var_2, var_4]
    var_6 = {}
    var_7 = module_0.Choice(choices=var_5, **var_6)
    var_8 = module_1.to_json_schema(var_7)
    var_9 = bool(var_8 == {'enum': ['a', 'b']})
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'hello'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = 'default'
    var_5 = to_json_schema(var_3)[var_4]
    assert var_5 == 'hello'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_from_json_schema_type_number. Retrieved 11/12 statements.
# Partially parsed test_from_json_schema_type_integer. Retrieved 9/10 statements.
# Partially parsed test_from_json_schema_type_string. Retrieved 12/14 statements.
# Partially parsed test_from_json_schema_type_boolean. Retrieved 7/8 statements.
# Partially parsed test_from_json_schema_type_array_simple. Retrieved 10/12 statements.
# Partially parsed test_from_json_schema_type_array_complex. Retrieved 21/25 statements.
# Partially parsed test_from_json_schema_type_object_simple. Retrieved 14/16 statements.
# Partially parsed test_from_json_schema_type_object_advanced. Retrieved 23/25 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'minimum'
    var_1 = 'maximum'
    var_2 = 'default'
    var_3 = 1
    var_4 = 10
    var_5 = 5
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = []
    var_8 = {}
    var_9 = module_0.Definitions(*var_7, **var_8)
    var_10 = 'number'
    var_11 = False
    var_12 = module_1.from_json_schema_type(var_6, var_10, var_11, var_9)
    var_13 = var_12.minimum
    assert var_13 == 1
    var_14 = var_12.maximum
    assert var_14 == 10
    var_15 = var_12.default
    assert var_15 == 5

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'minimum'
    var_1 = 'exclusiveMinimum'
    var_2 = 0
    var_3 = 1
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = {}
    var_7 = module_0.Definitions(*var_5, **var_6)
    var_8 = 'integer'
    var_9 = True
    var_10 = module_1.from_json_schema_type(var_4, var_8, var_9, var_7)
    var_11 = var_10.minimum
    assert var_11 == 0
    var_12 = var_10.exclusive_minimum
    assert var_12 == 1
    var_13 = var_10.allow_null
    assert var_13 is True

import typesystem.schemas as module_0

def test_case_0():
    var_0 = 'minLength'
    var_1 = 'maxLength'
    var_2 = 'pattern'
    var_3 = 'format'
    var_4 = 5
    var_5 = 10
    var_6 = '^abc'
    var_7 = 'email'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = []
    var_10 = {}
    var_11 = module_0.Definitions(*var_9, **var_10)
    var_12 = 'string'
    var_13 = False

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'minLength'
    var_1 = 0
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = {}
    var_5 = module_0.Definitions(*var_3, **var_4)
    var_6 = 'string'
    var_7 = False
    var_8 = module_1.from_json_schema_type(var_2, var_6, var_7, var_5)
    var_9 = var_8.allow_blank
    assert var_9 is True

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'default'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = {}
    var_5 = module_0.Definitions(*var_3, **var_4)
    var_6 = 'boolean'
    var_7 = False
    var_8 = module_1.from_json_schema_type(var_2, var_6, var_7, var_5)
    var_9 = var_8.default
    assert var_9 is True

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'items'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = []
    var_6 = {}
    var_7 = module_0.Definitions(*var_5, **var_6)
    var_8 = 'array'
    var_9 = False
    var_10 = module_1.from_json_schema_type(var_4, var_8, var_9, var_7)
    var_11 = var_10.items

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'items'
    var_1 = 'additionalItems'
    var_2 = 'minItems'
    var_3 = 'type'
    var_4 = 'integer'
    var_5 = {var_3: var_4}
    var_6 = 'boolean'
    var_7 = {var_3: var_6}
    var_8 = [var_5, var_7]
    var_9 = 'string'
    var_10 = {var_3: var_9}
    var_11 = 1
    var_12 = {var_0: var_8, var_1: var_10, var_2: var_11}
    var_13 = []
    var_14 = {}
    var_15 = module_0.Definitions(*var_13, **var_14)
    var_16 = 'array'
    var_17 = False
    var_18 = module_1.from_json_schema_type(var_12, var_16, var_17, var_15)
    var_19 = var_18.items
    var_20 = var_18.items[var_17]
    var_21 = var_18.items[var_11]
    var_22 = var_18.additional_items
    var_23 = var_18.min_items
    assert var_23 == 1

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'properties'
    var_1 = 'required'
    var_2 = 'name'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = [var_2]
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = []
    var_10 = {}
    var_11 = module_0.Definitions(*var_9, **var_10)
    var_12 = 'object'
    var_13 = False
    var_14 = module_1.from_json_schema_type(var_8, var_12, var_13, var_11)
    var_15 = 'name'
    var_16 = bool('name' in var_14.properties)
    assert var_16 is True
    var_17 = var_14.properties[var_2]
    var_18 = 'name'
    var_19 = bool('name' in var_14.required)
    assert var_19 is True

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'properties'
    var_1 = 'patternProperties'
    var_2 = 'additionalProperties'
    var_3 = 'propertyNames'
    var_4 = 'minProperties'
    var_5 = 'age'
    var_6 = 'type'
    var_7 = 'integer'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = '^id_'
    var_11 = 'string'
    var_12 = {var_6: var_11}
    var_13 = {var_10: var_12}
    var_14 = False
    var_15 = {var_6: var_11}
    var_16 = 1
    var_17 = {var_0: var_9, var_1: var_13, var_2: var_14, var_3: var_15, var_4: var_16}
    var_18 = []
    var_19 = {}
    var_20 = module_0.Definitions(*var_18, **var_19)
    var_21 = 'object'
    var_22 = module_1.from_json_schema_type(var_17, var_21, var_14, var_20)
    var_23 = var_22.properties[var_5]
    var_24 = '^id_'
    var_25 = bool('^id_' in var_22.pattern_properties)
    assert var_25 is True
    var_26 = var_22.additional_properties
    assert var_26 is False
    var_27 = var_22.property_names
    var_28 = var_22.min_properties
    assert var_28 == 1



# Parsed testcases at query #21
#--------------------------

# Failed to parse test_to_json_schema_error_unsupported_type.
# Partially parsed test_to_json_schema_regex_invalid_flags. Retrieved 1/8 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'string'
    var_4 = var_2['minLength']
    assert var_4 == 1

import re as module_0
import typesystem.fields as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = '^[a-z]+$'
    var_3 = module_0.compile(var_2)
    var_4 = 'pattern_regex'
    var_5 = {var_4: var_3}
    var_6 = module_1.String(max_length=var_1, min_length=var_0, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = var_7['minLength']
    assert var_8 == 5
    var_9 = var_7['maxLength']
    assert var_9 == 10
    var_10 = var_7['pattern']
    assert var_10 == '^[a-z]+$'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['string', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = 100
    var_2 = {}
    var_3 = module_0.Integer(minimum=var_0, maximum=var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    assert var_5 == 'integer'
    var_6 = var_4['minimum']
    assert var_6 == 0
    var_7 = var_4['maximum']
    assert var_7 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    assert var_5 == 'boolean'
    var_6 = var_4['default']
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 1
    var_3 = 5
    var_4 = {}
    var_5 = module_0.Array(var_1, min_items=var_2, max_items=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['type']
    assert var_7 == 'array'
    var_8 = var_6['minItems']
    assert var_8 == 1
    var_9 = var_6['maxItems']
    assert var_9 == 5
    var_10 = var_6['items']
    var_11 = bool(var_6['items'] == {'type': 'string', 'minLength': 1})
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = [var_0]
    var_8 = {}
    var_9 = module_0.Object(properties=var_6, required=var_7, **var_8)
    var_10 = module_1.to_json_schema(var_9)
    var_11 = var_10['type']
    assert var_11 == 'object'
    var_12 = 'name'
    var_13 = bool('name' in var_10['properties'])
    assert var_13 is True
    var_14 = 'age'
    var_15 = bool('age' in var_10['properties'])
    assert var_15 is True
    var_16 = 'name'
    var_17 = bool('name' in var_10['required'])
    assert var_17 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'anyOf'
    var_9 = bool('anyOf' in var_7)
    assert var_9 is True
    var_10 = 'anyOf'
    var_11 = var_7[var_10]
    var_12 = len(var_11)
    assert var_12 == 2

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'User'
    var_1 = 'id'
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_0.Object(properties=var_4, **var_5)
    var_7 = {var_0: var_6}
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_1.Definitions(*var_8, **var_9)
    var_11 = module_2.to_json_schema(var_10)
    var_12 = 'components'
    var_13 = bool('components' in var_11)
    assert var_13 is True
    var_14 = 'schemas'
    var_15 = bool('schemas' in var_11['components'])
    assert var_15 is True
    var_16 = 'User'
    var_17 = bool('User' in var_11['components']['schemas'])
    assert var_17 is True

def test_case_0():
    var_0 = 'abc'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['const']
    assert var_4 == 'fixed_value'



# Parsed testcases at query #22
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Integer(exclusive_maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['exclusiveMaximum']
    assert var_4 == 10



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_if_then_else_from_json_schema_all_clauses. Retrieved 15/16 statements.
# Partially parsed test_if_then_else_from_json_schema_only_if. Retrieved 11/14 statements.
# Partially parsed test_if_then_else_from_json_schema_if_and_then. Retrieved 13/16 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Definitions(*var_0, **var_1)
    var_3 = 'if'
    var_4 = 'then'
    var_5 = 'else'
    var_6 = 'default'
    var_7 = 'type'
    var_8 = 'string'
    var_9 = {var_7: var_8}
    var_10 = 'integer'
    var_11 = {var_7: var_10}
    var_12 = 'boolean'
    var_13 = {var_7: var_12}
    var_14 = 123
    var_15 = {var_3: var_9, var_4: var_11, var_5: var_13, var_6: var_14}
    var_16 = module_1.if_then_else_from_json_schema(var_15, var_2)
    var_17 = var_16.if_clause.numeric_type
    assert var_17 is None
    var_18 = var_16.then_clause.numeric_type
    var_19 = var_16.else_clause.numeric_type
    var_20 = var_16.default
    assert var_20 == 123

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Definitions(*var_0, **var_1)
    var_3 = 'if'
    var_4 = 'default'
    var_5 = 'type'
    var_6 = 'number'
    var_7 = {var_5: var_6}
    var_8 = None
    var_9 = {var_3: var_7, var_4: var_8}
    var_10 = module_1.if_then_else_from_json_schema(var_9, var_2)
    var_11 = var_10.if_clause.numeric_type
    var_12 = var_10.then_clause
    var_13 = var_10.else_clause

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Definitions(*var_0, **var_1)
    var_3 = 'if'
    var_4 = 'then'
    var_5 = 'type'
    var_6 = 'string'
    var_7 = {var_5: var_6}
    var_8 = 'boolean'
    var_9 = {var_5: var_8}
    var_10 = {var_3: var_7, var_4: var_9}
    var_11 = module_1.if_then_else_from_json_schema(var_10, var_2)
    var_12 = var_11.if_clause
    var_13 = var_11.then_clause
    var_14 = var_11.else_clause



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_from_json_schema_bool_true. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_bool_false. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_ref. Retrieved 5/6 statements.
# Partially parsed test_from_json_schema_not. Retrieved 7/8 statements.
# Partially parsed test_from_json_schema_any. Retrieved 4/5 statements.


import typesystem.json_schema as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.from_json_schema(var_0)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'const'
    var_1 = 123
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)
    var_4 = var_3.const
    assert var_4 == 123

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'enum'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.from_json_schema(var_4)
    var_6 = var_5.choices
    var_7 = bool(var_5.choices == [('a', 'a'), ('b', 'b')])
    assert var_7 is True

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Definitions(*var_0, **var_1)
    var_3 = '$ref'
    var_4 = '#/components/schemas/User'
    var_5 = {var_3: var_4}
    var_6 = module_1.from_json_schema(var_5, var_2)
    var_7 = var_6.to
    assert var_7 == '#/components/schemas/User'
    var_8 = var_6.target
    var_9 = bool(var_6.target == var_2['#/components/schema/User'])
    assert var_9 is True

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'allOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'const'
    var_5 = 'foo'
    var_6 = {var_4: var_5}
    var_7 = [var_3, var_6]
    var_8 = {var_0: var_7}
    var_9 = module_0.from_json_schema(var_8)
    var_10 = var_9.all_of
    var_11 = len(var_10)
    assert var_11 == 2

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'anyOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'number'
    var_5 = {var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = {var_0: var_6}
    var_8 = module_0.from_json_schema(var_7)
    var_9 = var_8.any_of
    var_10 = len(var_9)
    assert var_10 == 2

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'oneOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'number'
    var_5 = {var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = {var_0: var_6}
    var_8 = module_0.from_json_schema(var_7)
    var_9 = var_8.one_of
    var_10 = len(var_9)
    assert var_10 == 2

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'not'
    var_1 = 'const'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.from_json_schema(var_4)
    var_6 = var_5.negated

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'components'
    var_1 = 'schemas'
    var_2 = 'Item'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = module_0.from_json_schema(var_8)
    var_10 = bool(True)
    assert var_10 is True

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'if'
    var_1 = 'then'
    var_2 = 'else'
    var_3 = 'type'
    var_4 = 'number'
    var_5 = {var_3: var_4}
    var_6 = 'const'
    var_7 = 1
    var_8 = {var_6: var_7}
    var_9 = 0
    var_10 = {var_6: var_9}
    var_11 = {var_0: var_5, var_1: var_8, var_2: var_10}
    var_12 = module_0.from_json_schema(var_11)
    var_13 = var_12.if_clause
    var_14 = bool(var_12.if_clause is not None)
    assert var_14 is True
    var_15 = var_12.then_clause
    var_16 = bool(var_12.then_clause is not None)
    assert var_16 is True
    var_17 = var_12.else_clause
    var_18 = bool(var_12.else_clause is not None)
    assert var_18 is True

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'other'
    var_1 = 'unrecognized'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)



# Parsed testcases at query #25
#--------------------------

# Failed to parse test_to_json_schema_unsupported_type.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'string', 'minLength': 1})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'hello'
    var_2 = 'allow_null'
    var_3 = 'default'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.String(**var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = bool(var_6 == {'type': ['string', 'null'], 'default': 'hello', 'minLength': 1})
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.Integer(minimum=var_0, maximum=var_1, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = bool(var_6 == {'type': ['integer', 'null'], 'minimum': 0, 'maximum': 10})
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'boolean'})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 1
    var_3 = {}
    var_4 = module_0.Array(var_1, min_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = bool(var_5 == {'type': 'array', 'minItems': 1, 'items': {'type': 'string', 'minLength': 1}, 'uniqueItems': True})
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = {}
    var_8 = module_0.Object(properties=var_6, **var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = var_9['type']
    assert var_10 == 'object'
    var_11 = var_9['properties']['name']
    var_12 = bool(var_9['properties']['name'] == {'type': 'string', 'minLength': 1})
    assert var_12 is True
    var_13 = var_9['properties']['age']
    var_14 = bool(var_9['properties']['age'] == {'type': 'integer', 'minLength': 1})
    assert var_14 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'anyOf'
    var_9 = bool('anyOf' in var_7)
    assert var_9 is True
    var_10 = var_7['anyOf'][0]
    var_11 = bool(var_7['anyOf'][0] == {'type': 'string', 'minLength': 1})
    assert var_11 is True
    var_12 = var_7['anyOf'][1]
    var_13 = bool(var_7['anyOf'][1] == {'type': 'integer', 'minLength': 1})
    assert var_13 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'User'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_1.Definitions(*var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = 'components'
    var_9 = bool('components' in var_7)
    assert var_9 is True
    var_10 = 'schemas'
    var_11 = bool('schemas' in var_7['components'])
    assert var_11 is True
    var_12 = var_7['components']['schemas']['User']
    var_13 = bool(var_7['components']['schemas']['User'] == {'type': 'string', 'minLength': 1})
    assert var_13 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['const']
    assert var_4 == 'fixed_value'

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.to_json_schema(var_0)
    var_2 = bool(var_1 == {})
    assert var_2 is True



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_to_json_schema_unsupported_type.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.String(max_length=var_1, min_length=var_0, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = bool(var_6 == {'type': ['string', 'null'], 'minLength': 5, 'maxLength': 10})
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 42
    var_1 = 0
    var_2 = 'default'
    var_3 = {var_2: var_0}
    var_4 = module_0.Integer(minimum=var_1, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = bool(var_5 == {'type': 'integer', 'default': 42, 'minimum': 0})
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': ['boolean', 'null']})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 1
    var_3 = {}
    var_4 = module_0.Array(var_1, min_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = bool(var_5 == {'type': 'array', 'minItems': 1, 'items': {'type': 'string', 'minLength': 1}})
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = [var_0]
    var_8 = {}
    var_9 = module_0.Object(properties=var_6, required=var_7, **var_8)
    var_10 = module_1.to_json_schema(var_9)
    var_11 = bool(var_10 == {'type': 'object', 'properties': {'name': {'type': 'string', 'minLength': 1}, 'age': {'type': 'integer'}}, 'required': ['name']})
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = bool(var_7 == {'anyOf': [{'type': 'string', 'minLength': 1}, {'type': 'integer'}]})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'User'
    var_1 = 'id'
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_0.Object(properties=var_4, **var_5)
    var_7 = {var_0: var_6}
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_1.Definitions(*var_8, **var_9)
    var_11 = module_2.to_json_schema(var_10)
    var_12 = 'components'
    var_13 = bool('components' in var_11)
    assert var_13 is True
    var_14 = 'schemas'
    var_15 = bool('schemas' in var_11['components'])
    assert var_15 is True
    var_16 = 'User'
    var_17 = bool('User' in var_11['components']['schemas'])
    assert var_17 is True
    var_18 = var_11['components']['schemas']['User']['type']
    assert var_18 == 'object'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'const': 'fixed'})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = (var_3, var_1)
    var_5 = [var_2, var_4]
    var_6 = {}
    var_7 = module_0.Choice(choices=var_5, **var_6)
    var_8 = module_1.to_json_schema(var_7)
    var_9 = var_8['enum']
    var_10 = bool(var_8['enum'] == ['a', 'b'])
    assert var_10 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_from_json_schema_bool_true. Retrieved 2/5 statements.
# Partially parsed test_from_json_schema_bool_false. Retrieved 2/5 statements.
# Partially parsed test_from_json_schema_ref. Retrieved 5/8 statements.
# Partially parsed test_from_json_schema_enum. Retrieved 6/9 statements.
# Partially parsed test_from_json_schema_const. Retrieved 4/7 statements.
# Partially parsed test_from_json_schema_all_of. Retrieved 12/15 statements.
# Partially parsed test_from_json_schema_any_of. Retrieved 11/14 statements.
# Partially parsed test_from_json_schema_one_of. Retrieved 11/14 statements.
# Partially parsed test_from_json_schema_not. Retrieved 6/9 statements.
# Partially parsed test_from_json_schema_if_then_else. Retrieved 13/17 statements.


import typesystem.json_schema as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.from_json_schema(var_0)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Definitions(*var_0, **var_1)
    var_3 = '$ref'
    var_4 = '#/components/schemas/MyType'
    var_5 = {var_3: var_4}
    var_6 = module_1.from_json_schema(var_5, var_2)
    var_7 = var_6.to
    assert var_7 == '#/components/schemas/MyType'

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'enum'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.from_json_schema(var_4)
    var_6 = 'a'
    var_7 = 'a'
    var_8 = (var_6, var_7)
    var_9 = bool(('a', 'a') in var_5.choices)
    assert var_9 is True

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'const'
    var_1 = 42
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)
    var_4 = var_3.const
    assert var_4 == 42

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'allOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'const'
    var_5 = 'fixed'
    var_6 = {var_4: var_5}
    var_7 = [var_3, var_6]
    var_8 = {var_0: var_7}
    var_9 = module_0.from_json_schema(var_8)
    var_10 = var_9.all_of
    var_11 = len(var_10)
    assert var_11 == 2

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'anyOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'integer'
    var_5 = {var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = {var_0: var_6}
    var_8 = module_0.from_json_schema(var_7)
    var_9 = var_8.any_of
    var_10 = len(var_9)
    assert var_10 == 2

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'oneOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'integer'
    var_5 = {var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = {var_0: var_6}
    var_8 = module_0.from_json_schema(var_7)
    var_9 = var_8.one_of
    var_10 = len(var_9)
    assert var_10 == 2

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'not'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.from_json_schema(var_4)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Definitions(*var_0, **var_1)
    var_3 = 'if'
    var_4 = 'then'
    var_5 = 'else'
    var_6 = 'type'
    var_7 = 'string'
    var_8 = {var_6: var_7}
    var_9 = 'integer'
    var_10 = {var_6: var_9}
    var_11 = 'boolean'
    var_12 = {var_6: var_11}
    var_13 = {var_3: var_8, var_4: var_10, var_5: var_12}
    var_14 = module_1.from_json_schema(var_13, var_2)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1
import builtins as module_2

def test_case_0():
    var_0 = 'components'
    var_1 = '$ref'
    var_2 = 'schemas'
    var_3 = 'User'
    var_4 = 'type'
    var_5 = 'properties'
    var_6 = 'object'
    var_7 = 'id'
    var_8 = 'integer'
    var_9 = {var_4: var_8}
    var_10 = {var_7: var_9}
    var_11 = {var_4: var_6, var_5: var_10}
    var_12 = {var_3: var_11}
    var_13 = {var_2: var_12}
    var_14 = '#/components/schemas/User'
    var_15 = {var_0: var_13, var_1: var_14}
    var_16 = []
    var_17 = {}
    var_18 = module_0.Definitions(*var_16, **var_17)
    var_19 = module_1.from_json_schema(var_15, var_18)
    var_20 = '#/components/schemas/User'
    var_21 = bool('#/components/schemas/User' in var_18)
    assert var_21 is True
    var_22 = var_18[var_14]
    var_23 = var_22.properties[var_7]
    var_24 = None
    var_25 = [var_24]
    var_26 = {}
    var_27 = module_2.type(*var_25, **var_26)
    var_28 = isinstance(var_23, var_27)
    var_29 = bool(var_28)
    assert var_29 is True



# Parsed testcases at query #3
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'minItems'
    var_5 = bool('minItems' in var_3)
    assert var_5 is True
    var_6 = var_3['minItems']
    assert var_6 == 5



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_from_json_schema_with_definitions_is_none. Retrieved 5/7 statements.


import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0.from_json_schema(var_2, var_3)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_from_json_schema_type_number. Retrieved 11/12 statements.
# Partially parsed test_from_json_schema_type_integer. Retrieved 9/10 statements.
# Partially parsed test_from_json_schema_type_string. Retrieved 13/14 statements.
# Partially parsed test_from_json_schema_type_boolean. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_type_array. Retrieved 15/16 statements.
# Partially parsed test_from_json_schema_type_object. Retrieved 19/21 statements.


import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'minimum'
    var_1 = 'maximum'
    var_2 = 'default'
    var_3 = 0
    var_4 = 10
    var_5 = 5
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'number'
    var_8 = True
    var_9 = None
    var_10 = module_0.from_json_schema_type(var_6, var_7, var_8, var_9)
    var_11 = var_10.min_properties
    assert var_11 is None
    var_12 = var_10.allow_null
    assert var_12 is True

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'minimum'
    var_1 = 'maximum'
    var_2 = 1
    var_3 = 5
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'integer'
    var_6 = False
    var_7 = None
    var_8 = module_0.from_json_schema_type(var_4, var_5, var_6, var_7)
    var_9 = var_8.allow_null
    assert var_9 is False

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'minLength'
    var_1 = 'maxLength'
    var_2 = 'pattern'
    var_3 = 'format'
    var_4 = 5
    var_5 = 10
    var_6 = '^abc'
    var_7 = 'email'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = 'string'
    var_10 = False
    var_11 = None
    var_12 = module_0.from_json_schema_type(var_8, var_9, var_10, var_11)
    var_13 = var_12.min_length
    assert var_13 == 5
    var_14 = var_12.max_length
    assert var_14 == 10
    var_15 = var_12.pattern
    assert var_15 == '^abc'
    var_16 = var_12.format
    assert var_16 == 'email'

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'default'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'boolean'
    var_4 = None
    var_5 = module_0.from_json_schema_type(var_2, var_3, var_1, var_4)
    var_6 = var_5.allow_null
    assert var_6 is True

def test_case_0():
    var_0 = 'items'
    var_1 = 'additionalItems'
    var_2 = 'minItems'
    var_3 = 'maxItems'
    var_4 = 'uniqueItems'
    var_5 = 'type'
    var_6 = 'string'
    var_7 = {var_5: var_6}
    var_8 = False
    var_9 = 1
    var_10 = 5
    var_11 = True
    var_12 = {var_0: var_7, var_1: var_8, var_2: var_9, var_3: var_10, var_4: var_11}
    var_13 = 'array'
    var_14 = None

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'properties'
    var_1 = 'required'
    var_2 = 'additionalProperties'
    var_3 = 'minProperties'
    var_4 = 'name'
    var_5 = 'type'
    var_6 = 'string'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = [var_4]
    var_10 = 'integer'
    var_11 = {var_5: var_10}
    var_12 = 1
    var_13 = {var_0: var_8, var_1: var_9, var_2: var_11, var_3: var_12}
    var_14 = 'object'
    var_15 = False
    var_16 = None
    var_17 = module_0.from_json_schema_type(var_13, var_14, var_15, var_16)
    var_18 = 'name'
    var_19 = bool('name' in var_17.properties)
    assert var_19 is True
    var_20 = 'name'
    var_21 = bool('name' in var_17.required)
    assert var_21 is True
    var_22 = var_17.additional_properties
    var_23 = var_17.min_properties
    assert var_23 == 1



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_to_json_schema_predicate_false. Retrieved 2/3 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)



# Parsed testcases at query #7
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 != True)
    assert var_3 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_from_json_schema_type_array_with_items. Retrieved 9/13 statements.


import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'items'
    var_2 = 'array'
    var_3 = 'integer'
    var_4 = {var_0: var_3}
    var_5 = {var_0: var_2, var_1: var_4}
    var_6 = {}
    var_7 = True
    var_8 = module_0.from_json_schema_type(var_5, var_2, var_7, var_6)



# Parsed testcases at query #9
#--------------------------

# Failed to parse test_to_json_schema_error_on_unsupported.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'string', 'minLength': 1})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.String(max_length=var_1, min_length=var_0, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = bool(var_6 == {'type': ['string', 'null'], 'minLength': 5, 'maxLength': 10})
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = 100
    var_2 = 1
    var_3 = 'default'
    var_4 = {var_3: var_2}
    var_5 = module_0.Integer(minimum=var_0, maximum=var_1, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = bool(var_6 == {'type': 'integer', 'default': 1, 'minimum': 0, 'maximum': 100})
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': ['boolean', 'null']})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 1
    var_3 = True
    var_4 = {}
    var_5 = module_0.Array(var_1, min_items=var_2, unique_items=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = bool(var_6 == {'type': 'array', 'minItems': 1, 'items': {'type': 'string', 'minLength': 1}, 'uniqueItems': True})
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = {}
    var_8 = module_0.Object(properties=var_6, **var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = var_9['type']
    assert var_10 == 'object'
    var_11 = var_9['properties']['name']
    var_12 = bool(var_9['properties']['name'] == {'type': 'string', 'minLength': 1})
    assert var_12 is True
    var_13 = var_9['properties']['age']
    var_14 = bool(var_9['properties']['age'] == {'type': 'integer', 'minLength': 1})
    assert var_14 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'anyOf'
    var_9 = bool('anyOf' in var_7)
    assert var_9 is True
    var_10 = 'anyOf'
    var_11 = var_7[var_10]
    var_12 = len(var_11)
    assert var_12 == 2

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'MyString'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_1.Definitions(*var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = 'components'
    var_9 = bool('components' in var_7)
    assert var_9 is True
    var_10 = 'schemas'
    var_11 = bool('schemas' in var_7['components'])
    assert var_11 is True
    var_12 = var_7['components']['schemas']['MyString']
    var_13 = bool(var_7['components']['schemas']['MyString'] == {'type': 'string', 'minLength': 1})
    assert var_13 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['const']
    assert var_4 == 'fixed_value'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = (var_3, var_1)
    var_5 = [var_2, var_4]
    var_6 = {}
    var_7 = module_0.Choice(choices=var_5, **var_6)
    var_8 = module_1.to_json_schema(var_7)
    var_9 = 'enum'
    var_10 = bool('enum' in var_8)
    assert var_10 is True
    var_11 = 'a'
    var_12 = bool('a' in var_8['enum'])
    assert var_12 is True
    var_13 = 'b'
    var_14 = bool('b' in var_8['enum'])
    assert var_14 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_from_json_schema_type_number. Retrieved 17/18 statements.
# Partially parsed test_from_json_schema_type_integer. Retrieved 9/10 statements.
# Partially parsed test_from_json_schema_type_boolean. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_type_array_simple. Retrieved 17/18 statements.
# Partially parsed test_from_json_schema_type_array_list_items. Retrieved 15/16 statements.
# Partially parsed test_from_json_schema_type_object_complex. Retrieved 32/34 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'minimum'
    var_1 = 'maximum'
    var_2 = 'exclusiveMinimum'
    var_3 = 'exclusiveMaximum'
    var_4 = 'multipleOf'
    var_5 = 'default'
    var_6 = 1.0
    var_7 = 10.0
    var_8 = 0.5
    var_9 = 10.5
    var_10 = 2.0
    var_11 = 5.0
    var_12 = {var_0: var_6, var_1: var_7, var_2: var_8, var_3: var_9, var_4: var_10, var_5: var_11}
    var_13 = []
    var_14 = {}
    var_15 = module_0.Definitions(*var_13, **var_14)
    var_16 = 'number'
    var_17 = True
    var_18 = module_1.from_json_schema_type(var_12, var_16, var_17, var_15)
    var_19 = var_18.allow_null
    assert var_19 is True
    var_20 = var_18.minimum
    var_21 = bool(var_18.minimum == 1.0)
    assert var_21 is True
    var_22 = var_18.maximum
    var_23 = bool(var_18.maximum == 10.0)
    assert var_23 is True
    var_24 = var_18.exclusive_minimum
    var_25 = bool(var_18.exclusive_minimum == 0.5)
    assert var_25 is True
    var_26 = var_18.exclusive_maximum
    var_27 = bool(var_18.exclusive_maximum == 10.5)
    assert var_27 is True
    var_28 = var_18.multiple_of
    var_29 = bool(var_18.multiple_of == 2.0)
    assert var_29 is True

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'minimum'
    var_1 = 'default'
    var_2 = 1
    var_3 = 5
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = {}
    var_7 = module_0.Definitions(*var_5, **var_6)
    var_8 = 'integer'
    var_9 = False
    var_10 = module_1.from_json_schema_type(var_4, var_8, var_9, var_7)
    var_11 = var_10.allow_null
    assert var_11 is False
    var_12 = var_10.minimum
    assert var_12 == 1
    var_13 = var_10.default
    assert var_13 == 5

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'minLength'
    var_1 = 'maxLength'
    var_2 = 'format'
    var_3 = 'pattern'
    var_4 = 'default'
    var_5 = 5
    var_6 = 10
    var_7 = 'email'
    var_8 = '^[a-z]+$'
    var_9 = 'hello'
    var_10 = {var_0: var_5, var_1: var_6, var_2: var_7, var_3: var_8, var_4: var_9}
    var_11 = []
    var_12 = {}
    var_13 = module_0.Definitions(*var_11, **var_12)
    var_14 = 'string'
    var_15 = True
    var_16 = module_1.from_json_schema_type(var_10, var_14, var_15, var_13)
    var_17 = var_16.min_length
    assert var_17 == 5
    var_18 = var_16.max_length
    assert var_18 == 10
    var_19 = var_16.format
    assert var_19 == 'email'
    var_20 = var_16.pattern
    assert var_20 == '^[a-z]+$'
    var_21 = var_16.default
    assert var_21 == 'hello'

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'default'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = {}
    var_5 = module_0.Definitions(*var_3, **var_4)
    var_6 = 'boolean'
    var_7 = module_1.from_json_schema_type(var_2, var_6, var_1, var_5)
    var_8 = var_7.default
    assert var_8 is True

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'items'
    var_1 = 'minItems'
    var_2 = 'maxItems'
    var_3 = 'uniqueItems'
    var_4 = 'additionalItems'
    var_5 = 'type'
    var_6 = 'string'
    var_7 = {var_5: var_6}
    var_8 = 1
    var_9 = 5
    var_10 = True
    var_11 = False
    var_12 = {var_0: var_7, var_1: var_8, var_2: var_9, var_3: var_10, var_4: var_11}
    var_13 = []
    var_14 = {}
    var_15 = module_0.Definitions(*var_13, **var_14)
    var_16 = 'array'
    var_17 = True
    var_18 = module_1.from_json_schema_type(var_12, var_16, var_17, var_15)
    var_19 = var_18.min_items
    assert var_19 == 1
    var_20 = var_18.max_items
    assert var_20 == 5
    var_21 = var_18.unique_items
    assert var_21 is True
    var_22 = var_18.additional_items
    assert var_22 is False

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'items'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'integer'
    var_5 = {var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = {var_0: var_6}
    var_8 = []
    var_9 = {}
    var_10 = module_0.Definitions(*var_8, **var_9)
    var_11 = 'array'
    var_12 = True
    var_13 = module_1.from_json_schema_type(var_7, var_11, var_12, var_10)
    var_14 = var_13.items
    var_15 = var_13.items
    var_16 = len(var_15)
    assert var_16 == 2

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'properties'
    var_1 = 'patternProperties'
    var_2 = 'additionalProperties'
    var_3 = 'propertyNames'
    var_4 = 'minProperties'
    var_5 = 'maxProperties'
    var_6 = 'required'
    var_7 = 'default'
    var_8 = 'name'
    var_9 = 'age'
    var_10 = 'type'
    var_11 = 'string'
    var_12 = {var_10: var_11}
    var_13 = 'integer'
    var_14 = {var_10: var_13}
    var_15 = {var_8: var_12, var_9: var_14}
    var_16 = '^attr_'
    var_17 = {var_10: var_11}
    var_18 = {var_16: var_17}
    var_19 = 'boolean'
    var_20 = {var_10: var_19}
    var_21 = {var_10: var_11}
    var_22 = 1
    var_23 = 5
    var_24 = [var_8]
    var_25 = {}
    var_26 = {var_0: var_15, var_1: var_18, var_2: var_20, var_3: var_21, var_4: var_22, var_5: var_23, var_6: var_24, var_7: var_25}
    var_27 = []
    var_28 = {}
    var_29 = module_0.Definitions(*var_27, **var_28)
    var_30 = 'object'
    var_31 = False
    var_32 = module_1.from_json_schema_type(var_26, var_30, var_31, var_29)
    var_33 = 'name'
    var_34 = bool('name' in var_32.properties)
    assert var_34 is True
    var_35 = '^attr_'
    var_36 = bool('^attr_' in var_32.pattern_properties)
    assert var_36 is True
    var_37 = var_32.additional_properties
    var_38 = var_32.min_properties
    assert var_38 == 1
    var_39 = var_32.max_properties
    assert var_39 == 5
    var_40 = 'name'
    var_41 = bool('name' in var_32.required)
    assert var_41 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_to_json_schema_reference. Retrieved 2/9 statements.
# Partially parsed test_to_json_schema_error_on_unsupported_type. Retrieved 2/11 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = True
    var_3 = 'test'
    var_4 = 'allow_null'
    var_5 = 'default'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_0.String(max_length=var_1, min_length=var_0, **var_6)
    var_8 = module_1.to_json_schema(var_7)
    var_9 = var_8['type']
    var_10 = bool(var_8['type'] == ['string', 'null'])
    assert var_10 is True
    var_11 = var_8['minLength']
    assert var_11 == 5
    var_12 = var_8['maxLength']
    assert var_12 == 10
    var_13 = var_8['default']
    assert var_13 == 'test'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = 100
    var_2 = False
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.Integer(minimum=var_0, maximum=var_1, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['type']
    assert var_7 == 'integer'
    var_8 = var_6['minimum']
    assert var_8 == 0
    var_9 = var_6['maximum']
    assert var_9 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    assert var_5 == 'boolean'
    var_6 = var_4['default']
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = 5
    var_4 = {}
    var_5 = module_0.Array(var_2, min_items=var_0, max_items=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['type']
    assert var_7 == 'array'
    var_8 = var_6['minItems']
    assert var_8 == 1
    var_9 = var_6['maxItems']
    assert var_9 == 5
    var_10 = var_6['items']
    var_11 = bool(var_6['items'] == {'type': ['string', 'null']})
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = [var_0]
    var_8 = {}
    var_9 = module_0.Object(properties=var_6, required=var_7, **var_8)
    var_10 = module_1.to_json_schema(var_9)
    var_11 = var_10['type']
    assert var_11 == 'object'
    var_12 = 'name'
    var_13 = bool('name' in var_10['properties'])
    assert var_13 is True
    var_14 = 'age'
    var_15 = bool('age' in var_10['properties'])
    assert var_15 is True
    var_16 = var_10['required']
    var_17 = bool(var_10['required'] == ['name'])
    assert var_17 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'anyOf'
    var_9 = bool('anyOf' in var_7)
    assert var_9 is True
    var_10 = 'anyOf'
    var_11 = var_7[var_10]
    var_12 = len(var_11)
    assert var_12 == 2

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'User'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_1.Definitions(*var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = 'components'
    var_9 = bool('components' in var_7)
    assert var_9 is True
    var_10 = 'schemas'
    var_11 = bool('schemas' in var_7['components'])
    assert var_11 is True
    var_12 = 'User'
    var_13 = bool('User' in var_7['components']['schemas'])
    assert var_13 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'User'
    var_1 = {}
    var_2 = module_0.String(**var_1)

def test_case_0():
    var_0 = 'Cannot convert field type'
    var_1 = AssertionError()



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_type_from_json_schema_number. Retrieved 7/8 statements.
# Partially parsed test_type_from_json_schema_integer. Retrieved 7/8 statements.
# Partially parsed test_type_from_json_schema_string. Retrieved 9/10 statements.
# Partially parsed test_type_from_json_schema_boolean. Retrieved 7/8 statements.
# Partially parsed test_type_from_json_schema_array. Retrieved 11/13 statements.
# Partially parsed test_type_from_json_schema_object. Retrieved 13/15 statements.
# Partially parsed test_type_from_json_schema_union_with_null. Retrieved 7/8 statements.
# Partially parsed test_type_from_json_schema_union_multiple_types. Retrieved 7/8 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'minimum'
    var_2 = 'number'
    var_3 = 0
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = {}
    var_7 = module_0.Definitions(*var_5, **var_6)
    var_8 = module_1.type_from_json_schema(var_4, var_7)
    var_9 = var_8.allow_null
    assert var_9 is False
    var_10 = var_8.minimum
    assert var_10 == 0

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'maximum'
    var_2 = 'integer'
    var_3 = 10
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = {}
    var_7 = module_0.Definitions(*var_5, **var_6)
    var_8 = module_1.type_from_json_schema(var_4, var_7)
    var_9 = var_8.allow_null
    assert var_9 is False
    var_10 = var_8.maximum
    assert var_10 == 10

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'minLength'
    var_2 = 'maxLength'
    var_3 = 'string'
    var_4 = 5
    var_5 = 10
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = []
    var_8 = {}
    var_9 = module_0.Definitions(*var_7, **var_8)
    var_10 = module_1.type_from_json_schema(var_6, var_9)
    var_11 = var_10.min_length
    assert var_11 == 5
    var_12 = var_10.max_length
    assert var_12 == 10

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'default'
    var_2 = 'boolean'
    var_3 = True
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = {}
    var_7 = module_0.Definitions(*var_5, **var_6)
    var_8 = module_1.type_from_json_schema(var_4, var_7)
    var_9 = var_8.allow_null
    assert var_9 is False
    var_10 = var_8.default
    assert var_10 is True

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'items'
    var_2 = 'minItems'
    var_3 = 'array'
    var_4 = 'string'
    var_5 = {var_0: var_4}
    var_6 = 1
    var_7 = {var_0: var_3, var_1: var_5, var_2: var_6}
    var_8 = []
    var_9 = {}
    var_10 = module_0.Definitions(*var_8, **var_9)
    var_11 = module_1.type_from_json_schema(var_7, var_10)
    var_12 = var_11.min_items
    assert var_12 == 1
    var_13 = var_11.items

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'properties'
    var_2 = 'required'
    var_3 = 'object'
    var_4 = 'name'
    var_5 = 'string'
    var_6 = {var_0: var_5}
    var_7 = {var_4: var_6}
    var_8 = [var_4]
    var_9 = {var_0: var_3, var_1: var_7, var_2: var_8}
    var_10 = []
    var_11 = {}
    var_12 = module_0.Definitions(*var_10, **var_11)
    var_13 = module_1.type_from_json_schema(var_9, var_12)
    var_14 = var_13.properties[var_4]
    var_15 = var_13.required
    var_16 = bool(var_13.required == ['name'])
    assert var_16 is True

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = 'null'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = []
    var_6 = {}
    var_7 = module_0.Definitions(*var_5, **var_6)
    var_8 = module_1.type_from_json_schema(var_4, var_7)
    var_9 = var_8.allow_null
    assert var_9 is True

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = 'integer'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = []
    var_6 = {}
    var_7 = module_0.Definitions(*var_5, **var_6)
    var_8 = module_1.type_from_json_schema(var_4, var_7)
    var_9 = var_8.allow_null
    assert var_9 is False



# Parsed testcases at query #13
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.String(max_length=var_1, min_length=var_0, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = bool(var_6 == {'type': ['string', 'null'], 'minLength': 5, 'maxLength': 10})
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = 100
    var_2 = 10
    var_3 = 'default'
    var_4 = {var_3: var_2}
    var_5 = module_0.Integer(minimum=var_0, maximum=var_1, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = bool(var_6 == {'type': 'integer', 'default': 10, 'minimum': 0, 'maximum': 100})
    assert var_7 is True



# Parsed testcases at query #14
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'string', 'minLength': 1})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.String(max_length=var_1, min_length=var_0, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = bool(var_6 == {'type': ['string', 'null'], 'minLength': 5, 'maxLength': 10})
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = 100
    var_2 = {}
    var_3 = module_0.Integer(minimum=var_0, maximum=var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'integer', 'minimum': 0, 'maximum': 100})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': ['boolean', 'null']})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 1
    var_3 = {}
    var_4 = module_0.Array(var_1, min_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = bool(var_5 == {'type': 'array', 'minItems': 1, 'items': {'type': 'string', 'minLength': 1}})
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = {}
    var_8 = module_0.Object(properties=var_6, **var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = var_9['type']
    assert var_10 == 'object'
    var_11 = var_9['properties']['name']
    var_12 = bool(var_9['properties']['name'] == {'type': 'string', 'minLength': 1})
    assert var_12 is True
    var_13 = var_9['properties']['age']
    var_14 = bool(var_9['properties']['age'] == {'type': 'integer'})
    assert var_14 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'anyOf'
    var_9 = bool('anyOf' in var_7)
    assert var_9 is True
    var_10 = 'type'
    var_11 = 'minLength'
    var_12 = 'string'
    var_13 = 1
    var_14 = {var_10: var_12, var_11: var_13}
    var_15 = bool({'type': 'string', 'minLength': 1} in var_7['anyOf'])
    assert var_15 is True
    var_16 = 'type'
    var_17 = 'integer'
    var_18 = {var_16: var_17}
    var_19 = bool({'type': 'integer'} in var_7['anyOf'])
    assert var_19 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'MyString'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_1.Definitions(*var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = 'components'
    var_9 = bool('components' in var_7)
    assert var_9 is True
    var_10 = 'schemas'
    var_11 = bool('schemas' in var_7['components'])
    assert var_11 is True
    var_12 = var_7['components']['schemas']['MyString']
    var_13 = bool(var_7['components']['schemas']['MyString'] == {'type': 'string', 'minLength': 1})
    assert var_13 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['default']
    assert var_5 == 'hello'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['const']
    assert var_4 == 'fixed_value'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = (var_3, var_1)
    var_5 = [var_2, var_4]
    var_6 = {}
    var_7 = module_0.Choice(choices=var_5, **var_6)
    var_8 = module_1.to_json_schema(var_7)
    var_9 = var_8['enum']
    var_10 = bool(var_8['enum'] == ['a', 'b'])
    assert var_10 is True



# Parsed testcases at query #15
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Integer(multiple_of=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['multipleOf']
    assert var_4 == 5



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_to_json_schema_evaluates_schema_branch. Retrieved 1/3 statements.


def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_from_json_schema_multiple_constraints. Retrieved 9/11 statements.


import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'enum'
    var_2 = 'const'
    var_3 = 'string'
    var_4 = 'a'
    var_5 = 'b'
    var_6 = [var_4, var_5]
    var_7 = {var_0: var_3, var_1: var_6, var_2: var_4}
    var_8 = module_0.from_json_schema(var_7)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_to_json_schema_evaluates_object_predicate. Retrieved 7/8 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = {}
    var_3 = None
    var_4 = []
    var_5 = 'allow_null'
    var_6 = {var_5: var_0}
    var_7 = module_0.Object(properties=var_1, pattern_properties=var_2, additional_properties=var_3, property_names=var_3, min_properties=var_3, max_properties=var_3, required=var_4, **var_6)
    var_8 = module_1.to_json_schema(var_7)



# Parsed testcases at query #19
#--------------------------




import re as module_0
import typesystem.fields as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = module_0.compile(var_0)
    var_2 = 'pattern_regex'
    var_3 = {var_2: var_1}
    var_4 = module_1.String(**var_3)
    var_5 = module_2.to_json_schema(var_4)
    var_6 = 'pattern'
    var_7 = bool('pattern' in var_5)
    assert var_7 is True
    var_8 = var_5['pattern']
    assert var_8 == '^[a-z]+$'



# Parsed testcases at query #20
#--------------------------




import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'prop'
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(**var_3)
    var_5 = {var_0: var_4}
    var_6 = False
    var_7 = [var_0]
    var_8 = 'allow_null'
    var_9 = 'required'
    var_10 = {var_8: var_6, var_9: var_7}
    var_11 = module_1.Schema(var_5, **var_10)
    var_12 = module_2.to_json_schema(var_11)
    var_13 = 'properties'
    var_14 = bool('properties' in var_12)
    assert var_14 is True



# Parsed testcases at query #21
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_to_json_schema_evaluates_schema_predicate. Retrieved 1/3 statements.


def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}



# Parsed testcases at query #23
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = 'allow_null'
    var_5 = {var_4: var_0}
    var_6 = module_0.String(**var_5)
    var_7 = None
    var_8 = {}
    var_9 = module_1.IfThenElse(var_3, var_6, var_7, **var_8)
    var_10 = module_2.to_json_schema(var_9)
    var_11 = 'then'
    var_12 = bool('then' in var_10)
    assert var_12 is True
    var_13 = 'else'
    var_14 = bool('else' not in var_10)
    assert var_14 is True



# Parsed testcases at query #24
#--------------------------

# Failed to parse test_to_json_schema_error_unsupported_type.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 5
    var_2 = 'allow_null'
    var_3 = {var_2: var_0}
    var_4 = module_0.String(min_length=var_1, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['type']
    var_7 = bool(var_5['type'] == ['string', 'null'])
    assert var_7 is True
    var_8 = var_5['minLength']
    assert var_8 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = 1
    var_3 = 'default'
    var_4 = {var_3: var_2}
    var_5 = module_0.Integer(minimum=var_0, maximum=var_1, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['type']
    assert var_7 == 'integer'
    var_8 = var_6['minimum']
    assert var_8 == 0
    var_9 = var_6['maximum']
    assert var_9 == 10
    var_10 = var_6['default']
    assert var_10 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['boolean', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 1
    var_3 = True
    var_4 = {}
    var_5 = module_0.Array(var_1, min_items=var_2, unique_items=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['type']
    assert var_7 == 'array'
    var_8 = var_6['minItems']
    assert var_8 == 1
    var_9 = var_6['uniqueItems']
    assert var_9 is True
    var_10 = var_6['items']
    var_11 = bool(var_6['items'] == {'type': 'string', 'minLength': 1})
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = [var_0]
    var_8 = {}
    var_9 = module_0.Object(properties=var_6, required=var_7, **var_8)
    var_10 = module_1.to_json_schema(var_9)
    var_11 = var_10['type']
    assert var_11 == 'object'
    var_12 = var_10['properties']['name']
    var_13 = bool(var_10['properties']['name'] == {'type': 'string', 'minLength': 1})
    assert var_13 is True
    var_14 = var_10['properties']['age']
    var_15 = bool(var_10['properties']['age'] == {'type': 'integer', 'minLength': 1})
    assert var_15 is True
    var_16 = var_10['required']
    var_17 = bool(var_10['required'] == ['name'])
    assert var_17 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'anyOf'
    var_9 = bool('anyOf' in var_7)
    assert var_9 is True
    var_10 = 'anyOf'
    var_11 = var_7[var_10]
    var_12 = len(var_11)
    assert var_12 == 2
    var_13 = var_7['anyOf'][0]
    var_14 = bool(var_7['anyOf'][0] == {'type': 'string', 'minLength': 1})
    assert var_14 is True
    var_15 = var_7['anyOf'][1]
    var_16 = bool(var_7['anyOf'][1] == {'type': 'integer', 'minLength': 1})
    assert var_16 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'User'
    var_1 = 'id'
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_0.Object(properties=var_4, **var_5)
    var_7 = {var_0: var_6}
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_1.Definitions(*var_8, **var_9)
    var_11 = module_2.to_json_schema(var_10)
    var_12 = 'components'
    var_13 = bool('components' in var_11)
    assert var_13 is True
    var_14 = 'schemas'
    var_15 = bool('schemas' in var_11['components'])
    assert var_15 is True
    var_16 = 'User'
    var_17 = bool('User' in var_11['components']['schemas'])
    assert var_17 is True
    var_18 = var_11['components']['schemas']['User']['type']
    assert var_18 == 'object'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['const']
    assert var_4 == 'fixed'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'A'
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'B'
    var_4 = (var_3, var_1)
    var_5 = [var_2, var_4]
    var_6 = {}
    var_7 = module_0.Choice(choices=var_5, **var_6)
    var_8 = module_1.to_json_schema(var_7)
    var_9 = var_8['enum']
    var_10 = bool(var_8['enum'] == ['A', 'B'])
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_1.Not(var_1, **var_2)
    var_4 = module_2.to_json_schema(var_3)
    var_5 = 'not'
    var_6 = bool('not' in var_4)
    assert var_6 is True
    var_7 = var_4['not']['type']
    assert var_7 == 'string'

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = {}
    var_5 = module_1.IfThenElse(var_1, var_3, **var_4)
    var_6 = module_2.to_json_schema(var_5)
    var_7 = var_6['if']
    var_8 = bool(var_6['if'] == {'type': 'string', 'minLength': 1})
    assert var_8 is True
    var_9 = var_6['then']
    var_10 = bool(var_6['then'] == {'type': 'integer', 'minLength': 1})
    assert var_10 is True



# Parsed testcases at query #25
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Integer(exclusive_minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['exclusiveMinimum']
    assert var_4 == 5



