####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import typesystem.json_schema as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.get_valid_types(var_0)
    var_2 = bool(var_1 == ({'boolean', 'object', 'array', 'number', 'string'}, True))
    assert var_2 is True

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_valid_types(var_2)
    var_4 = bool(var_3 == ({'string'}, False))
    assert var_4 is True

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = 'boolean'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.get_valid_types(var_4)
    var_6 = bool(var_5 == ({'string', 'boolean'}, False))
    assert var_6 is True

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = 'null'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.get_valid_types(var_4)
    var_6 = bool(var_5 == ({'string'}, True))
    assert var_6 is True

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'number'
    var_2 = 'integer'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.get_valid_types(var_4)
    var_6 = bool(var_5 == ({'number'}, False))
    assert var_6 is True

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'null'
    var_2 = 'number'
    var_3 = 'object'
    var_4 = 'integer'
    var_5 = [var_1, var_2, var_3, var_4]
    var_6 = {var_0: var_5}
    var_7 = module_0.get_valid_types(var_6)
    var_8 = bool(var_7 == ({'number', 'object'}, True))
    assert var_8 is True

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'other'
    var_1 = 'data'
    var_2 = {var_0: var_1}
    var_3 = module_0.get_valid_types(var_2)
    var_4 = bool(var_3 == ({'boolean', 'object', 'array', 'number', 'string'}, True))
    assert var_4 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_to_json_schema_const. Retrieved 1/6 statements.


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
    var_0 = 10
    var_1 = 0
    var_2 = 'default'
    var_3 = {var_2: var_0}
    var_4 = module_0.Integer(minimum=var_1, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = bool(var_5 == {'type': 'integer', 'default': 10, 'minimum': 0})
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
    var_10 = var_7['anyOf'][0]
    var_11 = bool(var_7['anyOf'][0] == {'type': 'string', 'minLength': 1})
    assert var_11 is True
    var_12 = var_7['anyOf'][1]
    var_13 = bool(var_7['anyOf'][1] == {'type': 'integer'})
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

def test_case_0():
    var_0 = 'fixed'
    var_1 = 'value'
    var_2 = {var_1: var_0}



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_to_json_schema_definitions_and_reference. Retrieved 5/11 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 5
    var_2 = 10
    var_3 = 'allow_null'
    var_4 = {var_3: var_0}
    var_5 = module_0.String(max_length=var_2, min_length=var_1, **var_4)
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
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'boolean', 'default': True})
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
    var_10 = var_9['properties']['name']
    var_11 = bool(var_9['properties']['name'] == {'type': 'string', 'minLength': 1})
    assert var_11 is True
    var_12 = var_9['properties']['age']
    var_13 = bool(var_9['properties']['age'] == {'type': 'integer'})
    assert var_13 is True

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
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'User'
    var_3 = 'target'
    var_4 = {var_3: var_1}
    var_5 = {var_2: var_1}
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_1.Definitions(*var_6, **var_7)
    var_9 = module_2.to_json_schema(var_8)
    var_10 = bool('$ref' in var_9['components']['schemas']['User'] or 'User' in var_9['components']['schemas'])
    assert var_10 is True
    var_11 = 'components'
    var_12 = bool('components' in var_9)
    assert var_12 is True
    var_13 = 'User'
    var_14 = bool('User' in var_9['components']['schemas'])
    assert var_14 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'const': 'fixed_value'})
    assert var_4 is True

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



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_from_json_schema_type_number. Retrieved 17/22 statements.
# Partially parsed test_from_json_schema_type_string. Retrieved 19/23 statements.
# Partially parsed test_from_json_schema_type_boolean. Retrieved 6/10 statements.
# Partially parsed test_from_json_schema_type_array. Retrieved 30/39 statements.
# Partially parsed test_from_json_schema_type_object. Retrieved 34/43 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

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
    var_9 = []
    var_10 = {}
    var_11 = module_0.Definitions(*var_9, **var_10)
    var_12 = module_1.from_json_schema_type(var_6, var_7, var_8, var_11)
    var_13 = var_12.allow_null
    assert var_13 is True
    var_14 = var_12.minimum
    assert var_14 == 0
    var_15 = var_12.maximum
    assert var_15 == 10
    var_16 = 'exclusiveMinimum'
    var_17 = {var_0: var_8, var_16: var_3}
    var_18 = 'integer'
    var_19 = False
    var_20 = []
    var_21 = {}
    var_22 = module_0.Definitions(*var_20, **var_21)
    var_23 = module_1.from_json_schema_type(var_17, var_18, var_19, var_22)
    var_24 = var_23.allow_null
    assert var_24 is False
    var_25 = var_23.minimum
    assert var_25 == 1
    var_26 = var_23.exclusive_minimum
    assert var_26 == 0

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'minLength'
    var_1 = 'maxLength'
    var_2 = 'pattern'
    var_3 = 'format'
    var_4 = 'default'
    var_5 = 5
    var_6 = 10
    var_7 = '^abc'
    var_8 = 'email'
    var_9 = 'test'
    var_10 = {var_0: var_5, var_1: var_6, var_2: var_7, var_3: var_8, var_4: var_9}
    var_11 = 'string'
    var_12 = False
    var_13 = []
    var_14 = {}
    var_15 = module_0.Definitions(*var_13, **var_14)
    var_16 = module_1.from_json_schema_type(var_10, var_11, var_12, var_15)
    var_17 = var_16.min_length
    assert var_17 == 5
    var_18 = var_16.max_length
    assert var_18 == 10
    var_19 = var_16.pattern
    assert var_19 == '^abc'
    var_20 = var_16.format
    assert var_20 == 'email'
    var_21 = var_16.default
    assert var_21 == 'test'
    var_22 = {var_0: var_12}
    var_23 = True
    var_24 = []
    var_25 = {}
    var_26 = module_0.Definitions(*var_24, **var_25)
    var_27 = module_1.from_json_schema_type(var_22, var_11, var_23, var_26)
    var_28 = var_27.allow_blank
    assert var_28 is True

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'default'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'boolean'
    var_4 = []
    var_5 = {}
    var_6 = module_0.Definitions(*var_4, **var_5)
    var_7 = module_1.from_json_schema_type(var_2, var_3, var_1, var_6)
    var_8 = var_7.allow_null
    assert var_8 is True
    var_9 = var_7.default
    assert var_9 is True

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'items'
    var_1 = 'additionalItems'
    var_2 = 'minItems'
    var_3 = 'maxItems'
    var_4 = 'uniqueItems'
    var_5 = 'type'
    var_6 = 'string'
    var_7 = {var_5: var_6}
    var_8 = 'integer'
    var_9 = {var_5: var_8}
    var_10 = 1
    var_11 = 5
    var_12 = True
    var_13 = {var_0: var_7, var_1: var_9, var_2: var_10, var_3: var_11, var_4: var_12}
    var_14 = 'array'
    var_15 = False
    var_16 = []
    var_17 = {}
    var_18 = module_0.Definitions(*var_16, **var_17)
    var_19 = module_1.from_json_schema_type(var_13, var_14, var_15, var_18)
    var_20 = var_19.items
    var_21 = var_19.additional_items
    var_22 = var_19.min_items
    assert var_22 == 1
    var_23 = var_19.max_items
    assert var_23 == 5
    var_24 = var_19.unique_items
    assert var_24 is True
    var_25 = {var_5: var_6}
    var_26 = {var_5: var_8}
    var_27 = [var_25, var_26]
    var_28 = {var_0: var_27}
    var_29 = True
    var_30 = []
    var_31 = {}
    var_32 = module_0.Definitions(*var_30, **var_31)
    var_33 = module_1.from_json_schema_type(var_28, var_14, var_29, var_32)
    var_34 = var_33.items
    var_35 = var_33.items[var_15]
    var_36 = var_33.items[var_29]

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
    var_7 = 'name'
    var_8 = 'age'
    var_9 = 'type'
    var_10 = 'string'
    var_11 = {var_9: var_10}
    var_12 = 'integer'
    var_13 = {var_9: var_12}
    var_14 = {var_7: var_11, var_8: var_13}
    var_15 = '^id_'
    var_16 = {var_9: var_10}
    var_17 = {var_15: var_16}
    var_18 = 'boolean'
    var_19 = {var_9: var_18}
    var_20 = {var_9: var_10}
    var_21 = 1
    var_22 = 10
    var_23 = [var_7]
    var_24 = {var_0: var_14, var_1: var_17, var_2: var_19, var_3: var_20, var_4: var_21, var_5: var_22, var_6: var_23}
    var_25 = 'object'
    var_26 = False
    var_27 = []
    var_28 = {}
    var_29 = module_0.Definitions(*var_27, **var_28)
    var_30 = module_1.from_json_schema_type(var_24, var_25, var_26, var_29)
    var_31 = var_30.properties[var_7]
    var_32 = var_30.properties[var_8]
    var_33 = var_30.pattern_properties[var_15]
    var_34 = var_30.additional_properties
    var_35 = var_30.property_names
    var_36 = var_30.min_properties
    assert var_36 == 1
    var_37 = var_30.max_properties
    assert var_37 == 10
    var_38 = var_30.required
    var_39 = bool(var_30.required == ['name'])
    assert var_39 is True



# Parsed testcases at query #5
#--------------------------




import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Definitions(*var_0, **var_1)
    var_3 = 'allOf'
    var_4 = 'default'
    var_5 = 'type'
    var_6 = 'integer'
    var_7 = {var_5: var_6}
    var_8 = 'string'
    var_9 = {var_5: var_8}
    var_10 = [var_7, var_9]
    var_11 = 10
    var_12 = {var_3: var_10, var_4: var_11}
    var_13 = module_1.all_of_from_json_schema(var_12, var_2)
    var_14 = 'all_of'
    var_15 = hasattr(var_13, var_14)
    var_16 = bool(var_15)
    assert var_16 is True
    var_17 = var_13.all_of
    var_18 = len(var_17)
    assert var_18 == 2
    var_19 = var_13.default
    assert var_19 == 10



# Parsed testcases at query #6
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'email'
    var_1 = False
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_1}
    var_5 = module_0.String(allow_blank=var_2, format=var_0, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = var_6['format']
    assert var_7 == 'email'



# Parsed testcases at query #7
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
    var_7 = module_0.Integer(**var_6)
    var_8 = None
    var_9 = {}
    var_10 = module_1.IfThenElse(var_3, var_7, var_8, **var_9)
    var_11 = module_2.to_json_schema(var_10)
    var_12 = 'else'
    var_13 = bool('else' not in var_11)
    assert var_13 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_ref_from_json_schema_valid. Retrieved 11/13 statements.


import typesystem.schemas as module_0
import builtins as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Definitions(*var_0, **var_1)
    var_3 = 'MockField'
    var_4 = ()
    var_5 = 'validate'
    var_6 = lambda self, x: x
    var_7 = {var_5: var_6}
    var_8 = [var_3, var_4, var_7]
    var_9 = {}
    var_10 = module_1.type(*var_8, **var_9)
    var_11 = '$ref'
    var_12 = '#/user'
    var_13 = {var_11: var_12}
    var_14 = module_2.ref_from_json_schema(var_13, var_2)
    var_15 = var_14.to
    assert var_15 == '#/user'
    var_16 = var_14.target
    var_17 = bool(var_14.target == var_2['#/user'])
    assert var_17 is True

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Definitions(*var_0, **var_1)
    var_3 = '$ref'
    var_4 = 'user'
    var_5 = {var_3: var_4}
    var_6 = module_1.ref_from_json_schema(var_5, var_2)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Definitions(*var_0, **var_1)
    var_3 = 'not_a_ref'
    var_4 = '#/user'
    var_5 = {var_3: var_4}
    var_6 = module_1.ref_from_json_schema(var_5, var_2)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_one_of_from_json_schema_valid_input. Retrieved 14/18 statements.
# Partially parsed test_one_of_from_json_schema_no_default. Retrieved 10/14 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Definitions(*var_0, **var_1)
    var_3 = 'oneOf'
    var_4 = 'default'
    var_5 = 'type'
    var_6 = 'string'
    var_7 = {var_5: var_6}
    var_8 = 'number'
    var_9 = {var_5: var_8}
    var_10 = [var_7, var_9]
    var_11 = 'some_default'
    var_12 = {var_3: var_10, var_4: var_11}
    var_13 = module_1.one_of_from_json_schema(var_12, var_2)
    var_14 = var_13.one_of
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = var_13.default
    assert var_16 == 'some_default'

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Definitions(*var_0, **var_1)
    var_3 = 'oneOf'
    var_4 = 'type'
    var_5 = 'string'
    var_6 = {var_4: var_5}
    var_7 = [var_6]
    var_8 = {var_3: var_7}
    var_9 = module_1.one_of_from_json_schema(var_8, var_2)
    var_10 = var_9.one_of
    var_11 = len(var_10)
    assert var_11 == 1



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_type_from_json_schema_single_type_string. Retrieved 5/9 statements.
# Partially parsed test_type_from_json_schema_multiple_types_union. Retrieved 9/13 statements.
# Partially parsed test_type_from_json_schema_empty_type_defaults_to_all. Retrieved 12/16 statements.
# Partially parsed test_type_from_json_schema_integer_with_constraints. Retrieved 9/13 statements.
# Partially parsed test_type_from_json_schema_string_with_constraints. Retrieved 13/17 statements.
# Partially parsed test_type_from_json_schema_boolean. Retrieved 7/11 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'integer'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = {}
    var_5 = module_0.Definitions(*var_3, **var_4)
    var_6 = module_1.type_from_json_schema(var_2, var_5)
    var_7 = var_6.allow_null
    assert var_7 is False

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'null'
    var_2 = 'string'
    var_3 = True
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_2, var_1]
    var_6 = {var_0: var_5}
    var_7 = []
    var_8 = {}
    var_9 = module_0.Definitions(*var_7, **var_8)
    var_10 = module_1.type_from_json_schema(var_6, var_9)
    var_11 = var_10.allow_null
    assert var_11 is True

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'integer'
    var_2 = 'string'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = []
    var_6 = {}
    var_7 = module_0.Definitions(*var_5, **var_6)
    var_8 = module_1.type_from_json_schema(var_4, var_7)
    var_9 = var_8.any_of
    var_10 = len(var_9)
    assert var_10 == 2

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = {}
    var_3 = module_0.Definitions(*var_1, **var_2)
    var_4 = module_1.type_from_json_schema(var_0, var_3)
    var_5 = 'string'
    var_6 = var_4.any_of
    var_7 = 'type_string'
    var_8 = [t.type_string for t in var_6 if hasattr(t, var_7)]
    var_9 = var_5 in var_8
    var_10 = var_4.any_of
    var_11 = len(var_10)
    var_12 = 1
    var_13 = var_11 > var_12
    var_14 = bool(var_9 or var_13)
    assert var_14 is True

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'integer'
    var_4 = 5
    var_5 = 10
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = []
    var_8 = {}
    var_9 = module_0.Definitions(*var_7, **var_8)
    var_10 = module_1.type_from_json_schema(var_6, var_9)
    var_11 = var_10.minimum
    assert var_11 == 5
    var_12 = var_10.maximum
    assert var_12 == 10

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'minLength'
    var_2 = 'maxLength'
    var_3 = 'pattern'
    var_4 = 'format'
    var_5 = 'string'
    var_6 = 2
    var_7 = 5
    var_8 = '^abc'
    var_9 = 'email'
    var_10 = {var_0: var_5, var_1: var_6, var_2: var_7, var_3: var_8, var_4: var_9}
    var_11 = []
    var_12 = {}
    var_13 = module_0.Definitions(*var_11, **var_12)
    var_14 = module_1.type_from_json_schema(var_10, var_13)
    var_15 = var_14.min_length
    assert var_15 == 2
    var_16 = var_14.max_length
    assert var_16 == 5
    var_17 = var_14.pattern
    assert var_17 == '^abc'
    var_18 = var_14.format
    assert var_18 == 'email'

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



# Parsed testcases at query #11
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = 'allow_null'
    var_5 = {var_4: var_0}
    var_6 = module_0.Integer(**var_5)
    var_7 = None
    var_8 = {}
    var_9 = module_1.IfThenElse(var_3, var_6, var_7, **var_8)
    var_10 = module_2.to_json_schema(var_9)
    var_11 = 'if'
    var_12 = bool('if' in var_10)
    assert var_12 is True
    var_13 = 'then'
    var_14 = bool('then' in var_10)
    assert var_14 is True
    var_15 = 'else'
    var_16 = bool('else' not in var_10)
    assert var_16 is True



# Parsed testcases at query #12
#--------------------------




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
    var_10 = var_7['anyOf'][0]
    var_11 = bool(var_7['anyOf'][0] == {'type': 'string', 'minLength': 1})
    assert var_11 is True
    var_12 = var_7['anyOf'][1]
    var_13 = bool(var_7['anyOf'][1] == {'type': 'integer'})
    assert var_13 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'MyString'
    var_1 = 2
    var_2 = {}
    var_3 = module_0.String(min_length=var_1, **var_2)
    var_4 = {var_0: var_3}
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_1.Definitions(*var_5, **var_6)
    var_8 = module_2.to_json_schema(var_7)
    var_9 = 'components'
    var_10 = bool('components' in var_8)
    assert var_10 is True
    var_11 = 'schemas'
    var_12 = bool('schemas' in var_8['components'])
    assert var_12 is True
    var_13 = var_8['components']['schemas']['MyString']
    var_14 = bool(var_8['components']['schemas']['MyString'] == {'type': 'string', 'minLength': 2})
    assert var_14 is True

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
    var_0 = 'A'
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'B'
    var_4 = (var_3, var_1)
    var_5 = [var_2, var_4]
    var_6 = {}
    var_7 = module_0.Choice(choices=var_5, **var_6)
    var_8 = module_1.to_json_schema(var_7)
    var_9 = 'A'
    var_10 = bool('A' in var_8['enum'])
    assert var_10 is True
    var_11 = 'B'
    var_12 = bool('B' in var_8['enum'])
    assert var_12 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_from_json_schema_bool_true. Retrieved 2/5 statements.
# Partially parsed test_from_json_schema_bool_false. Retrieved 2/5 statements.
# Partially parsed test_from_json_schema_any. Retrieved 2/5 statements.
# Partially parsed test_from_json_schema_const. Retrieved 7/10 statements.
# Partially parsed test_from_json_schema_enum. Retrieved 9/12 statements.
# Partially parsed test_from_json_schema_ref_with_definitions. Retrieved 16/21 statements.
# Partially parsed test_from_json_schema_allOf. Retrieved 12/16 statements.
# Partially parsed test_from_json_schema_anyOf. Retrieved 13/16 statements.
# Partially parsed test_from_json_schema_type_string_single. Retrieved 4/7 statements.
# Partially parsed test_from_json_schema_type_string_multiple. Retrieved 6/9 statements.


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
    var_0 = {}
    var_1 = module_0.from_json_schema(var_0)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'const'
    var_1 = 123
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)
    var_4 = {var_0: var_1}
    var_5 = module_0.from_json_schema(var_4)
    var_6 = var_5.validate(var_1)
    assert var_6 == 123

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'enum'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.from_json_schema(var_4)
    var_6 = var_5.validate(var_1)
    assert var_6 == 'a'
    var_7 = 'c'
    var_8 = var_5.validate(var_7)
    assert var_8 is False

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'components'
    var_1 = '$ref'
    var_2 = 'schemas'
    var_3 = 'MyInt'
    var_4 = 'type'
    var_5 = 'integer'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {var_2: var_7}
    var_9 = '#/components/schemas/MyInt'
    var_10 = {var_0: var_8, var_1: var_9}
    var_11 = []
    var_12 = {}
    var_13 = module_0.Definitions(*var_11, **var_12)
    var_14 = module_1.from_json_schema(var_10, var_13)
    var_15 = var_14.target
    var_16 = 5
    var_17 = var_14.validate(var_16)
    assert var_17 == 5

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'allOf'
    var_1 = 'type'
    var_2 = 'integer'
    var_3 = {var_1: var_2}
    var_4 = 'const'
    var_5 = 10
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
    var_2 = 'integer'
    var_3 = {var_1: var_2}
    var_4 = 'string'
    var_5 = {var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = {var_0: var_6}
    var_8 = module_0.from_json_schema(var_7)
    var_9 = 10
    var_10 = var_8.validate(var_9)
    assert var_10 == 10
    var_11 = 'hello'
    var_12 = var_8.validate(var_11)
    assert var_12 == 'hello'

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'oneOf'
    var_1 = 'type'
    var_2 = 'integer'
    var_3 = {var_1: var_2}
    var_4 = 'string'
    var_5 = {var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = {var_0: var_6}
    var_8 = module_0.from_json_schema(var_7)
    var_9 = bool(var_8 is not None)
    assert var_9 is True

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'not'
    var_1 = 'type'
    var_2 = 'integer'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.from_json_schema(var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'if'
    var_1 = 'then'
    var_2 = 'else'
    var_3 = 'type'
    var_4 = 'integer'
    var_5 = {var_3: var_4}
    var_6 = 'const'
    var_7 = 1
    var_8 = {var_6: var_7}
    var_9 = 2
    var_10 = {var_6: var_9}
    var_11 = {var_0: var_5, var_1: var_8, var_2: var_10}
    var_12 = module_0.from_json_schema(var_11)
    var_13 = bool(var_12 is not None)
    assert var_13 is True

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'integer'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'integer'
    var_2 = 'string'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.from_json_schema(var_4)



# Parsed testcases at query #14
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = {}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, property_names=var_2, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = 'propertyNames'
    var_8 = bool('propertyNames' in var_6['properties'])
    assert var_8 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_to_json_schema_array_items_as_list. Retrieved 9/10 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = [var_3, var_3]
    var_5 = {}
    var_6 = module_0.Array(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'items'
    var_9 = bool('items' in var_7)
    assert var_9 is True
    var_10 = 'items'
    var_11 = var_7[var_10]
    var_12 = var_7[var_10]
    var_13 = len(var_12)
    assert var_13 == 2



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_to_json_schema_error_unsupported_type.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'default_val'
    var_2 = 'allow_null'
    var_3 = 'default'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.String(**var_4)
    var_6 = 0
    var_7 = 10
    var_8 = {}
    var_9 = module_0.Integer(minimum=var_6, maximum=var_7, **var_8)
    var_10 = {}
    var_11 = module_0.Boolean(**var_10)
    var_12 = 0.5
    var_13 = {}
    var_14 = module_0.Float(multiple_of=var_12, **var_13)
    var_15 = module_1.to_json_schema(var_5)
    var_16 = module_1.to_json_schema(var_9)
    var_17 = module_1.to_json_schema(var_11)
    var_18 = module_1.to_json_schema(var_14)
    var_19 = var_15['type']
    var_20 = bool(var_15['type'] == ['string', 'null'])
    assert var_20 is True
    var_21 = var_15['default']
    assert var_21 == 'default_val'
    var_22 = var_16['type']
    assert var_22 == 'integer'
    var_23 = var_16['minimum']
    assert var_23 == 0
    var_24 = var_16['maximum']
    assert var_24 == 10
    var_25 = var_17['type']
    assert var_25 == 'boolean'
    var_26 = var_18['type']
    assert var_26 == 'number'
    var_27 = var_18['multipleOf']
    var_28 = bool(var_18['multipleOf'] == 0.5)
    assert var_28 is True

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
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 1
    var_3 = {}
    var_4 = module_0.Array(var_1, min_items=var_2, **var_3)
    var_5 = 'name'
    var_6 = 'age'
    var_7 = {}
    var_8 = module_0.String(**var_7)
    var_9 = {}
    var_10 = module_0.Integer(**var_9)
    var_11 = {var_5: var_8, var_6: var_10}
    var_12 = [var_5]
    var_13 = {}
    var_14 = module_0.Object(properties=var_11, required=var_12, **var_13)
    var_15 = module_1.to_json_schema(var_4)
    var_16 = module_1.to_json_schema(var_14)
    var_17 = var_15['type']
    assert var_17 == 'array'
    var_18 = var_15['items']
    var_19 = bool(var_15['items'] == {'type': 'string', 'minLength': 1})
    assert var_19 is True
    var_20 = var_15['minItems']
    assert var_20 == 1
    var_21 = var_16['type']
    assert var_21 == 'object'
    var_22 = var_16['properties']['name']
    var_23 = bool(var_16['properties']['name'] == {'type': 'string', 'minLength': 1})
    assert var_23 is True
    var_24 = var_16['properties']['age']
    var_25 = bool(var_16['properties']['age'] == {'type': 'integer', 'minLength': 1})
    assert var_25 is True
    var_26 = var_16['required']
    var_27 = bool(var_16['required'] == ['name'])
    assert var_27 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'type'
    var_1 = 'properties'
    var_2 = 'object'
    var_3 = 'id'
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = {var_3: var_5}
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = 'string'
    var_9 = {var_0: var_8}
    var_10 = []
    var_11 = 'User'
    var_12 = 'Tag'
    var_13 = {var_11: var_7, var_12: var_9}
    var_14 = module_1.Definitions(*var_10, **var_13)
    var_15 = module_2.to_json_schema(var_14)
    var_16 = 'components'
    var_17 = bool('components' in var_15)
    assert var_17 is True
    var_18 = 'schemas'
    var_19 = bool('schemas' in var_15['components'])
    assert var_19 is True
    var_20 = 'User'
    var_21 = bool('User' in var_15['components']['schemas'])
    assert var_21 is True
    var_22 = var_15['components']['schemas']['User']['properties']['id']['type']
    assert var_22 == 'integer'
    var_23 = var_15['components']['schemas']['Tag']['type']
    assert var_23 == 'string'

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = {}
    var_8 = module_0.String(**var_7)
    var_9 = {}
    var_10 = module_1.Not(var_8, **var_9)
    var_11 = module_2.to_json_schema(var_6)
    var_12 = module_2.to_json_schema(var_10)
    var_13 = 'anyOf'
    var_14 = bool('anyOf' in var_11)
    assert var_14 is True
    var_15 = var_11['anyOf'][0]['type']
    assert var_15 == 'string'
    var_16 = var_11['anyOf'][1]['type']
    assert var_16 == 'integer'
    var_17 = 'not'
    var_18 = bool('not' in var_12)
    assert var_18 is True
    var_19 = var_12['not']['type']
    assert var_19 == 'string'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_from_json_schema_type_string. Retrieved 16/17 statements.
# Partially parsed test_from_json_schema_type_number. Retrieved 12/13 statements.
# Partially parsed test_from_json_schema_type_integer. Retrieved 10/11 statements.
# Partially parsed test_from_json_schema_type_boolean. Retrieved 8/9 statements.
# Partially parsed test_from_json_schema_type_array. Retrieved 20/23 statements.
# Partially parsed test_from_json_schema_type_object. Retrieved 19/20 statements.


import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'minLength'
    var_2 = 'maxLength'
    var_3 = 'pattern'
    var_4 = 'format'
    var_5 = 'default'
    var_6 = 'string'
    var_7 = 5
    var_8 = 10
    var_9 = '^abc'
    var_10 = 'email'
    var_11 = 'test'
    var_12 = {var_0: var_6, var_1: var_7, var_2: var_8, var_3: var_9, var_4: var_10, var_5: var_11}
    var_13 = False
    var_14 = None
    var_15 = module_0.from_json_schema_type(var_12, var_6, var_13, var_14)
    var_16 = var_15.min_length
    assert var_16 == 5
    var_17 = var_15.max_length
    assert var_17 == 10
    var_18 = var_15.pattern
    assert var_18 == '^abc'
    var_19 = var_15.format
    assert var_19 == 'email'
    var_20 = var_15.default
    assert var_20 == 'test'
    var_21 = var_15.allow_blank
    assert var_21 is False

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'default'
    var_4 = 'number'
    var_5 = 0
    var_6 = 100
    var_7 = 50.5
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = True
    var_10 = None
    var_11 = module_0.from_json_schema_type(var_8, var_4, var_9, var_10)
    var_12 = var_11.minimum
    assert var_12 == 0
    var_13 = var_11.maximum
    assert var_13 == 100
    var_14 = var_11.default
    var_15 = bool(var_11.default == 50.5)
    assert var_15 is True
    var_16 = var_11.allow_null
    assert var_16 is True

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'minimum'
    var_2 = 'exclusiveMinimum'
    var_3 = 'integer'
    var_4 = 1
    var_5 = 0
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = False
    var_8 = None
    var_9 = module_0.from_json_schema_type(var_6, var_3, var_7, var_8)
    var_10 = var_9.minimum
    assert var_10 == 1
    var_11 = var_9.exclusive_minimum
    assert var_11 == 0
    var_12 = var_9.allow_null
    assert var_12 is False

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'default'
    var_2 = 'boolean'
    var_3 = True
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = False
    var_6 = None
    var_7 = module_0.from_json_schema_type(var_4, var_2, var_5, var_6)
    var_8 = var_7.default
    assert var_8 is True
    var_9 = var_7.allow_null
    assert var_9 is False

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'items'
    var_2 = 'additionalItems'
    var_3 = 'minItems'
    var_4 = 'maxItems'
    var_5 = 'uniqueItems'
    var_6 = 'array'
    var_7 = 'string'
    var_8 = {var_0: var_7}
    var_9 = 'integer'
    var_10 = {var_0: var_9}
    var_11 = 1
    var_12 = 5
    var_13 = True
    var_14 = {var_0: var_6, var_1: var_8, var_2: var_10, var_3: var_11, var_4: var_12, var_5: var_13}
    var_15 = False
    var_16 = None
    var_17 = module_0.from_json_schema_type(var_14, var_6, var_15, var_16)
    var_18 = var_17.min_items
    assert var_18 == 1
    var_19 = var_17.max_items
    assert var_19 == 5
    var_20 = var_17.unique_items
    assert var_20 is True
    var_21 = var_17.items
    var_22 = var_17.additional_items

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'properties'
    var_2 = 'required'
    var_3 = 'additionalProperties'
    var_4 = 'minProperties'
    var_5 = 'object'
    var_6 = 'name'
    var_7 = 'age'
    var_8 = 'string'
    var_9 = {var_0: var_8}
    var_10 = 'integer'
    var_11 = {var_0: var_10}
    var_12 = {var_6: var_9, var_7: var_11}
    var_13 = [var_6]
    var_14 = False
    var_15 = 1
    var_16 = {var_0: var_5, var_1: var_12, var_2: var_13, var_3: var_14, var_4: var_15}
    var_17 = None
    var_18 = module_0.from_json_schema_type(var_16, var_5, var_14, var_17)
    var_19 = 'name'
    var_20 = bool('name' in var_18.properties)
    assert var_20 is True
    var_21 = 'age'
    var_22 = bool('age' in var_18.properties)
    assert var_22 is True
    var_23 = var_18.required
    var_24 = bool(var_18.required == ['name'])
    assert var_24 is True
    var_25 = var_18.additional_properties
    assert var_25 is False
    var_26 = var_18.min_properties
    assert var_26 == 1



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_ref_from_json_schema_success. Retrieved 11/13 statements.


import typesystem.schemas as module_0
import builtins as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Definitions(*var_0, **var_1)
    var_3 = 'MockField'
    var_4 = ()
    var_5 = 'validate'
    var_6 = lambda x: x
    var_7 = {var_5: var_6}
    var_8 = [var_3, var_4, var_7]
    var_9 = {}
    var_10 = module_1.type(*var_8, **var_9)
    var_11 = '$ref'
    var_12 = '#/user'
    var_13 = {var_11: var_12}
    var_14 = module_2.ref_from_json_schema(var_13, var_2)
    var_15 = var_14.to
    assert var_15 == '#/user'
    var_16 = var_14.target
    var_17 = bool(var_14.target == var_2['#/user'])
    assert var_17 is True

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Definitions(*var_0, **var_1)
    var_3 = '$ref'
    var_4 = 'user'
    var_5 = {var_3: var_4}
    var_6 = module_1.ref_from_json_schema(var_5, var_2)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Definitions(*var_0, **var_1)
    var_3 = 'not_a_ref'
    var_4 = '#/user'
    var_5 = {var_3: var_4}
    var_6 = module_1.ref_from_json_schema(var_5, var_2)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_to_json_schema_array_items_is_list_evaluates_to_true. Retrieved 7/8 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Array(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'items'
    var_9 = var_7[var_8]



# Parsed testcases at query #5
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'email'
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(format=var_0, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = var_5['format']
    assert var_6 == 'email'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_to_json_schema_definitions. Retrieved 3/8 statements.
# Partially parsed test_to_json_schema_union_and_const. Retrieved 9/12 statements.
# Failed to parse test_to_json_schema_error_on_unsupported_type.


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
    var_9 = 10
    var_10 = 0
    var_11 = 'default'
    var_12 = {var_11: var_9}
    var_13 = module_0.Integer(minimum=var_10, **var_12)
    var_14 = module_1.to_json_schema(var_13)
    var_15 = var_14['type']
    assert var_15 == 'integer'
    var_16 = var_14['default']
    assert var_16 == 10
    var_17 = var_14['minimum']
    assert var_17 == 0
    var_18 = 'allow_null'
    var_19 = {var_18: var_0}
    var_20 = module_0.Boolean(**var_19)
    var_21 = module_1.to_json_schema(var_20)
    var_22 = var_21['type']
    var_23 = bool(var_21['type'] == ['boolean', 'null'])
    assert var_23 is True

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
    var_7 = var_5['items']
    var_8 = bool(var_5['items'] == {'type': 'string', 'minLength': 1})
    assert var_8 is True
    var_9 = var_5['minItems']
    assert var_9 == 1
    var_10 = 'name'
    var_11 = 'age'
    var_12 = {}
    var_13 = module_0.String(**var_12)
    var_14 = {}
    var_15 = module_0.Integer(**var_14)
    var_16 = {var_10: var_13, var_11: var_15}
    var_17 = [var_10]
    var_18 = {}
    var_19 = module_0.Object(properties=var_16, required=var_17, **var_18)
    var_20 = module_1.to_json_schema(var_19)
    var_21 = var_20['type']
    assert var_21 == 'object'
    var_22 = var_20['properties']['name']
    var_23 = bool(var_20['properties']['name'] == {'type': 'string', 'minLength': 1})
    assert var_23 is True
    var_24 = var_20['properties']['age']
    var_25 = bool(var_20['properties']['age'] == {'type': 'integer'})
    assert var_25 is True
    var_26 = var_20['required']
    var_27 = bool(var_20['required'] == ['name'])
    assert var_27 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = 'components'
    var_5 = 'schemas'
    var_6 = 'User'

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
    var_13 = 'fixed'
    var_14 = 'value'
    var_15 = {var_14: var_13}



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_from_json_schema_boolean_true. Retrieved 2/5 statements.
# Partially parsed test_from_json_schema_boolean_false. Retrieved 2/5 statements.
# Partially parsed test_from_json_schema_any_type. Retrieved 2/5 statements.
# Partially parsed test_from_json_schema_const. Retrieved 4/7 statements.
# Partially parsed test_from_json_schema_enum. Retrieved 6/9 statements.
# Partially parsed test_from_json_schema_ref. Retrieved 5/9 statements.
# Partially parsed test_from_json_schema_components_parsing. Retrieved 13/17 statements.
# Partially parsed test_from_json_schema_all_of. Retrieved 10/14 statements.
# Partially parsed test_from_json_schema_any_of. Retrieved 11/15 statements.
# Partially parsed test_from_json_schema_one_of. Retrieved 11/15 statements.
# Partially parsed test_from_json_schema_not. Retrieved 7/12 statements.
# Partially parsed test_from_json_schema_if_then_else. Retrieved 15/22 statements.


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
    var_0 = {}
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
    var_6 = 'a'
    var_7 = 'a'
    var_8 = (var_6, var_7)
    var_9 = bool(('a', 'a') in var_5.choices)
    assert var_9 is True
    var_10 = 'b'
    var_11 = 'b'
    var_12 = (var_10, var_11)
    var_13 = bool(('b', 'b') in var_5.choices)
    assert var_13 is True

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

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'components'
    var_1 = 'schemas'
    var_2 = 'MyType'
    var_3 = 'const'
    var_4 = 'hello'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = []
    var_10 = {}
    var_11 = module_0.Definitions(*var_9, **var_10)
    var_12 = module_1.from_json_schema(var_8, var_11)
    var_13 = '#/components/schemas/MyType'
    var_14 = bool(var_13 in var_11)
    assert var_14 is True
    var_15 = var_11[var_13]
    var_16 = var_11[var_13].const
    assert var_16 == 'hello'

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'allOf'
    var_1 = 'const'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_1: var_2}
    var_5 = [var_3, var_4]
    var_6 = {var_0: var_5}
    var_7 = module_0.from_json_schema(var_6)
    var_8 = var_7.all_of
    var_9 = len(var_8)
    assert var_9 == 2

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'anyOf'
    var_1 = 'const'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = 2
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
    var_1 = 'const'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = 2
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
    var_0 = 'if'
    var_1 = 'then'
    var_2 = 'else'
    var_3 = 'const'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = 2
    var_7 = {var_3: var_6}
    var_8 = 3
    var_9 = {var_3: var_8}
    var_10 = {var_0: var_5, var_1: var_7, var_2: var_9}
    var_11 = module_0.from_json_schema(var_10)
    var_12 = var_11.if_clause
    var_13 = var_11.then_clause
    var_14 = var_11.else_clause



# Parsed testcases at query #8
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['integer', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Float(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    assert var_5 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Decimal(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    assert var_5 == 'number'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_to_json_schema_root_with_definitions_evaluates_true_at_line_172. Retrieved 9/23 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'MySchema'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = True
    var_6 = 'allow_null'
    var_7 = {var_6: var_5}
    var_8 = module_0.String(**var_7)
    var_9 = 'User'
    var_10 = 'target'
    var_11 = {var_10: var_8}
    var_12 = None
    var_13 = 'components'
    var_14 = 'User'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_from_json_schema_iterates_over_components_schemas. Retrieved 10/14 statements.


import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'components'
    var_1 = 'schemas'
    var_2 = 'MySchema'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = module_0.from_json_schema(var_8)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_to_json_schema_array_items_is_list_evaluates_to_true. Retrieved 7/8 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Array(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'items'
    var_9 = bool('items' in var_7)
    assert var_9 is True
    var_10 = 'items'
    var_11 = var_7[var_10]



# Parsed testcases at query #12
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = '^abc_'
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(**var_3)
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = module_0.Object(pattern_properties=var_5, **var_6)
    var_8 = module_1.to_json_schema(var_7)
    var_9 = 'patternProperties'
    var_10 = bool('patternProperties' in var_8)
    assert var_10 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_from_json_schema_type_number. Retrieved 9/13 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'minimum'
    var_1 = 'maximum'
    var_2 = 0
    var_3 = 10
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'number'
    var_6 = True
    var_7 = []
    var_8 = {}
    var_9 = module_0.Definitions(*var_7, **var_8)
    var_10 = module_1.from_json_schema_type(var_4, var_5, var_6, var_9)
    var_11 = var_10.allow_null
    assert var_11 is True
    var_12 = var_10.minimum
    assert var_12 == 0
    var_13 = var_10.maximum
    assert var_13 == 10



# Parsed testcases at query #14
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(**var_3)
    var_5 = 'allow_null'
    var_6 = {var_5: var_0}
    var_7 = module_0.Array(var_4, **var_6)
    var_8 = module_1.to_json_schema(var_7)
    var_9 = var_8['type']
    assert var_9 == 'array'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_type_from_json_schema_string. Retrieved 9/10 statements.
# Partially parsed test_type_from_json_schema_integer. Retrieved 7/8 statements.
# Partially parsed test_type_from_json_schema_number_strips_integer. Retrieved 11/13 statements.
# Partially parsed test_type_from_json_schema_boolean. Retrieved 7/8 statements.
# Partially parsed test_type_from_json_schema_with_null. Retrieved 9/11 statements.
# Partially parsed test_type_from_json_schema_empty_type_returns_const_none. Retrieved 5/6 statements.


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
    var_13 = var_10.allow_null
    assert var_13 is False

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'minimum'
    var_2 = 'integer'
    var_3 = 0
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = {}
    var_7 = module_0.Definitions(*var_5, **var_6)
    var_8 = module_1.type_from_json_schema(var_4, var_7)
    var_9 = var_8.minimum
    assert var_9 == 0
    var_10 = var_8.allow_null
    assert var_10 is False

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
    var_10 = 0
    var_11 = var_8.any_of[var_10]

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'properties'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = {}
    var_5 = module_0.Definitions(*var_3, **var_4)
    var_6 = module_1.type_from_json_schema(var_2, var_5)
    var_7 = var_6.const
    assert var_7 is None



# Parsed testcases at query #16
#--------------------------




import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'User'
    var_1 = 'type'
    var_2 = 'object'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = True
    var_6 = 'allow_null'
    var_7 = {var_6: var_5}
    var_8 = module_0.String(**var_7)
    var_9 = {var_0: var_8}
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_1.Definitions(*var_10, **var_11)
    var_13 = module_2.to_json_schema(var_12, var_4)
    var_14 = 'components'
    var_15 = bool('components' in var_13)
    assert var_15 is True
    var_16 = var_13['components']['schemas']['User']
    var_17 = bool(var_13['components']['schemas']['User'] == {'type': ['string', 'null']})
    assert var_17 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_to_json_schema_reference_predicate_true. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'MySchema'
    var_1 = {}



# Parsed testcases at query #18
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True



