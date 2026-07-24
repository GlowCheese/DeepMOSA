####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_from_json_schema_with_bool_true. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_with_bool_false. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_with_ref. Retrieved 5/7 statements.
# Partially parsed test_from_json_schema_with_type_constraint. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_enum. Retrieved 7/8 statements.
# Partially parsed test_from_json_schema_with_const. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_all_of. Retrieved 11/12 statements.
# Partially parsed test_from_json_schema_with_any_of. Retrieved 11/12 statements.
# Partially parsed test_from_json_schema_with_one_of. Retrieved 11/12 statements.
# Partially parsed test_from_json_schema_with_not. Retrieved 7/9 statements.
# Partially parsed test_from_json_schema_with_if_then_else. Retrieved 15/19 statements.
# Partially parsed test_from_json_schema_with_multiple_constraints. Retrieved 14/17 statements.
# Partially parsed test_from_json_schema_with_no_constraints. Retrieved 2/3 statements.


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
    var_4 = '#/components/schemas/test'
    var_5 = {var_3: var_4}
    var_6 = {}
    var_7 = module_1.from_json_schema(var_5, var_2)
    var_8 = var_7.to
    assert var_8 == '#/components/schemas/test'

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'enum'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = module_0.from_json_schema(var_5)
    var_7 = var_6.choices
    var_8 = bool(var_6.choices == [('a', 'a'), ('b', 'b'), ('c', 'c')])
    assert var_8 is True

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'const'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)
    var_4 = var_3.const
    assert var_4 == 'test'

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'allOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'number'
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
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.from_json_schema(var_4)
    var_6 = var_5.negated

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'if'
    var_1 = 'then'
    var_2 = 'else'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 'number'
    var_7 = {var_3: var_6}
    var_8 = 'boolean'
    var_9 = {var_3: var_8}
    var_10 = {var_0: var_5, var_1: var_7, var_2: var_9}
    var_11 = module_0.from_json_schema(var_10)
    var_12 = var_11.if_clause
    var_13 = var_11.then_clause
    var_14 = var_11.else_clause

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'enum'
    var_2 = 'string'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = {var_0: var_2, var_1: var_5}
    var_7 = module_0.from_json_schema(var_6)
    var_8 = var_7.all_of
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = 0
    var_11 = var_7.all_of[var_10]
    var_12 = 1
    var_13 = var_7.all_of[var_12]

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.from_json_schema(var_0)



# Parsed testcases at query #2
#--------------------------




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
    var_3 = bool(var_2 == {'type': 'string'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': ['string', 'null']})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'string', 'minLength': 5})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'string', 'maxLength': 10})
    assert var_4 is True

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
    var_6 = bool(var_5 == {'type': 'string', 'pattern': '^[a-z]+$'})
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'string', 'format': 'email'})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'integer'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': ['integer', 'null']})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'integer', 'minimum': 0})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'integer', 'maximum': 100})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'number'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Float(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': ['number', 'null']})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'number'})
    assert var_3 is True

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
    var_1 = module_0.Array(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'array'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': ['array', 'null']})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'array', 'minItems': 1})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'array', 'maxItems': 10})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'array', 'items': {'type': 'string'}})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = {}
    var_3 = module_0.Array(additional_items=var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'array', 'additionalItems': {'type': 'boolean'}})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'array', 'uniqueItems': True})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'object'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': ['object', 'null']})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = bool(var_6 == {'type': 'object', 'properties': {'name': {'type': 'string'}}})
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = {}
    var_2 = module_0.Integer(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(pattern_properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = bool(var_6 == {'type': 'object', 'patternProperties': {'^[a-z]+$': {'type': 'integer'}}})
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = {}
    var_3 = module_0.Object(additional_properties=var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'object', 'additionalProperties': {'type': 'boolean'}})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Object(property_names=var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'object', 'propertyNames': {'type': 'string'}})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Object(max_properties=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'object', 'maxProperties': 5})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'object', 'minProperties': 1})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Object(required=var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'object', 'required': ['name']})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = module_2.to_json_schema(var_5)
    var_7 = bool(var_6 == {'type': 'object', 'properties': {'name': {'type': 'string'}}})
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = True
    var_1 = 'name'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = 'allow_null'
    var_6 = {var_5: var_0}
    var_7 = module_1.Schema(var_4, **var_6)
    var_8 = module_2.to_json_schema(var_7)
    var_9 = bool(var_8 == {'type': ['object', 'null'], 'properties': {'name': {'type': 'string'}}})
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = 'required'
    var_6 = {var_5: var_4}
    var_7 = module_1.Schema(var_3, **var_6)
    var_8 = module_2.to_json_schema(var_7)
    var_9 = bool(var_8 == {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']})
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = bool(var_9 == {'enum': ['a', 'b']})
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'const': 'value'})
    assert var_4 is True

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
    var_8 = bool(var_7 == {'anyOf': [{'type': 'string'}, {'type': 'integer'}]})
    assert var_8 is True

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
    var_6 = module_1.OneOf(var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = bool(var_7 == {'oneOf': [{'type': 'string'}, {'type': 'integer'}]})
    assert var_8 is True

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
    var_6 = module_1.AllOf(var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = bool(var_7 == {'allOf': [{'type': 'string'}, {'type': 'integer'}]})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = {}
    var_5 = module_0.Boolean(**var_4)
    var_6 = {}
    var_7 = module_1.IfThenElse(var_1, var_3, var_5, **var_6)
    var_8 = module_2.to_json_schema(var_7)
    var_9 = bool(var_8 == {'if': {'type': 'string'}, 'then': {'type': 'integer'}, 'else': {'type': 'boolean'}})
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_1.Not(var_1, **var_2)
    var_4 = module_2.to_json_schema(var_3)
    var_5 = bool(var_4 == {'not': {'type': 'string'}})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'string'
    var_1 = 'integer'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_1.Definitions(*var_7, **var_8)
    var_10 = module_2.to_json_schema(var_9)
    var_11 = bool(var_10 == {'components': {'schemas': {'string': {'type': 'string'}, 'integer': {'type': 'integer'}}}})
    assert var_11 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_ifthenelse_field_type_check. Retrieved 3/4 statements.


import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = None
    var_3 = {}
    var_4 = module_1.IfThenElse(var_1, var_2, var_2, **var_3)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_array_field_with_list_items. Retrieved 9/10 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Array(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'items'
    var_9 = var_7[var_8]
    var_10 = var_7[var_8]
    var_11 = len(var_10)
    assert var_11 == 2



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_from_json_schema_type_number. Retrieved 12/13 statements.
# Partially parsed test_from_json_schema_type_integer. Retrieved 12/13 statements.
# Partially parsed test_from_json_schema_type_string. Retrieved 14/15 statements.
# Partially parsed test_from_json_schema_type_boolean. Retrieved 8/9 statements.
# Partially parsed test_from_json_schema_type_array. Retrieved 17/19 statements.
# Partially parsed test_from_json_schema_type_object. Retrieved 17/19 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'default'
    var_4 = 'number'
    var_5 = 0
    var_6 = 100
    var_7 = 50
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = False
    var_10 = []
    var_11 = {}
    var_12 = module_0.Definitions(*var_10, **var_11)
    var_13 = module_1.from_json_schema_type(var_8, var_4, var_9, var_12)
    var_14 = var_13.minimum
    assert var_14 == 0
    var_15 = var_13.maximum
    assert var_15 == 100
    var_16 = var_13.default
    assert var_16 == 50
    var_17 = var_13.allow_null
    assert var_17 is False

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'default'
    var_4 = 'integer'
    var_5 = 0
    var_6 = 100
    var_7 = 50
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = False
    var_10 = []
    var_11 = {}
    var_12 = module_0.Definitions(*var_10, **var_11)
    var_13 = module_1.from_json_schema_type(var_8, var_4, var_9, var_12)
    var_14 = var_13.minimum
    assert var_14 == 0
    var_15 = var_13.maximum
    assert var_15 == 100
    var_16 = var_13.default
    assert var_16 == 50
    var_17 = var_13.allow_null
    assert var_17 is False

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'minLength'
    var_2 = 'maxLength'
    var_3 = 'format'
    var_4 = 'default'
    var_5 = 'string'
    var_6 = 5
    var_7 = 10
    var_8 = 'email'
    var_9 = 'test@example.com'
    var_10 = {var_0: var_5, var_1: var_6, var_2: var_7, var_3: var_8, var_4: var_9}
    var_11 = False
    var_12 = []
    var_13 = {}
    var_14 = module_0.Definitions(*var_12, **var_13)
    var_15 = module_1.from_json_schema_type(var_10, var_5, var_11, var_14)
    var_16 = var_15.min_length
    assert var_16 == 5
    var_17 = var_15.max_length
    assert var_17 == 10
    var_18 = var_15.format
    assert var_18 == 'email'
    var_19 = var_15.default
    assert var_19 == 'test@example.com'
    var_20 = var_15.allow_null
    assert var_20 is False

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'default'
    var_2 = 'boolean'
    var_3 = True
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = False
    var_6 = []
    var_7 = {}
    var_8 = module_0.Definitions(*var_6, **var_7)
    var_9 = module_1.from_json_schema_type(var_4, var_2, var_5, var_8)
    var_10 = var_9.default
    assert var_10 is True
    var_11 = var_9.allow_null
    assert var_11 is False

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'items'
    var_2 = 'minItems'
    var_3 = 'maxItems'
    var_4 = 'default'
    var_5 = 'array'
    var_6 = 'string'
    var_7 = {var_0: var_6}
    var_8 = 1
    var_9 = 5
    var_10 = 'test'
    var_11 = [var_10]
    var_12 = {var_0: var_5, var_1: var_7, var_2: var_8, var_3: var_9, var_4: var_11}
    var_13 = False
    var_14 = []
    var_15 = {}
    var_16 = module_0.Definitions(*var_14, **var_15)
    var_17 = module_1.from_json_schema_type(var_12, var_5, var_13, var_16)
    var_18 = var_17.items
    var_19 = var_17.min_items
    assert var_19 == 1
    var_20 = var_17.max_items
    assert var_20 == 5
    var_21 = var_17.default
    var_22 = bool(var_17.default == ['test'])
    assert var_22 is True
    var_23 = var_17.allow_null
    assert var_23 is False

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'properties'
    var_2 = 'required'
    var_3 = 'default'
    var_4 = 'object'
    var_5 = 'name'
    var_6 = 'string'
    var_7 = {var_0: var_6}
    var_8 = {var_5: var_7}
    var_9 = [var_5]
    var_10 = 'test'
    var_11 = {var_5: var_10}
    var_12 = {var_0: var_4, var_1: var_8, var_2: var_9, var_3: var_11}
    var_13 = False
    var_14 = []
    var_15 = {}
    var_16 = module_0.Definitions(*var_14, **var_15)
    var_17 = module_1.from_json_schema_type(var_12, var_4, var_13, var_16)
    var_18 = var_17.properties[var_5]
    var_19 = var_17.required
    var_20 = bool(var_17.required == ['name'])
    assert var_20 is True
    var_21 = var_17.default
    var_22 = bool(var_17.default == {'name': 'test'})
    assert var_22 is True
    var_23 = var_17.allow_null
    assert var_23 is False



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_type_from_json_schema_with_single_type. Retrieved 5/6 statements.
# Partially parsed test_type_from_json_schema_with_nullable_single_type. Retrieved 7/8 statements.
# Partially parsed test_type_from_json_schema_with_multiple_types. Retrieved 13/16 statements.
# Partially parsed test_type_from_json_schema_with_nullable_multiple_types. Retrieved 14/17 statements.
# Partially parsed test_type_from_json_schema_with_no_type. Retrieved 3/4 statements.
# Partially parsed test_type_from_json_schema_with_nullable_no_type. Retrieved 6/7 statements.
# Partially parsed test_type_from_json_schema_with_integer_type. Retrieved 5/6 statements.
# Partially parsed test_type_from_json_schema_with_boolean_type. Retrieved 5/6 statements.
# Partially parsed test_type_from_json_schema_with_array_type. Retrieved 5/6 statements.
# Partially parsed test_type_from_json_schema_with_object_type. Retrieved 5/6 statements.
# Partially parsed test_type_from_json_schema_with_number_type. Retrieved 5/6 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
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
    var_2 = 'number'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = []
    var_6 = {}
    var_7 = module_0.Definitions(*var_5, **var_6)
    var_8 = module_1.type_from_json_schema(var_4, var_7)
    var_9 = var_8.any_of
    var_10 = len(var_9)
    assert var_10 == 2
    var_11 = 0
    var_12 = var_8.any_of[var_11]
    var_13 = 1
    var_14 = var_8.any_of[var_13]
    var_15 = var_8.allow_null
    assert var_15 is False

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = 'number'
    var_3 = 'null'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = []
    var_7 = {}
    var_8 = module_0.Definitions(*var_6, **var_7)
    var_9 = module_1.type_from_json_schema(var_5, var_8)
    var_10 = var_9.any_of
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = 0
    var_13 = var_9.any_of[var_12]
    var_14 = 1
    var_15 = var_9.any_of[var_14]
    var_16 = var_9.allow_null
    assert var_16 is True

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = {}
    var_3 = module_0.Definitions(*var_1, **var_2)
    var_4 = module_1.type_from_json_schema(var_0, var_3)

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
    var_1 = 'boolean'
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
    var_1 = 'array'
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
    var_1 = 'object'
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
    var_1 = 'number'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = {}
    var_5 = module_0.Definitions(*var_3, **var_4)
    var_6 = module_1.type_from_json_schema(var_2, var_5)
    var_7 = var_6.allow_null
    assert var_7 is False



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_from_json_schema_type_number. Retrieved 12/13 statements.
# Partially parsed test_from_json_schema_type_integer. Retrieved 12/13 statements.
# Partially parsed test_from_json_schema_type_string. Retrieved 14/15 statements.
# Partially parsed test_from_json_schema_type_string_with_min_length_zero. Retrieved 10/11 statements.
# Partially parsed test_from_json_schema_type_boolean. Retrieved 8/9 statements.
# Partially parsed test_from_json_schema_type_array. Retrieved 18/20 statements.
# Partially parsed test_from_json_schema_type_array_with_list_items. Retrieved 21/25 statements.
# Partially parsed test_from_json_schema_type_object. Retrieved 22/25 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'default'
    var_4 = 'number'
    var_5 = 0
    var_6 = 100
    var_7 = 50
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = False
    var_10 = []
    var_11 = {}
    var_12 = module_0.Definitions(*var_10, **var_11)
    var_13 = module_1.from_json_schema_type(var_8, var_4, var_9, var_12)
    var_14 = var_13.minimum
    assert var_14 == 0
    var_15 = var_13.maximum
    assert var_15 == 100
    var_16 = var_13.default
    assert var_16 == 50
    var_17 = var_13.allow_null
    assert var_17 is False
    var_18 = var_13.coerce_types
    assert var_18 is False

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'default'
    var_4 = 'integer'
    var_5 = 1
    var_6 = 10
    var_7 = 5
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = True
    var_10 = []
    var_11 = {}
    var_12 = module_0.Definitions(*var_10, **var_11)
    var_13 = module_1.from_json_schema_type(var_8, var_4, var_9, var_12)
    var_14 = var_13.minimum
    assert var_14 == 1
    var_15 = var_13.maximum
    assert var_15 == 10
    var_16 = var_13.default
    assert var_16 == 5
    var_17 = var_13.allow_null
    assert var_17 is True
    var_18 = var_13.coerce_types
    assert var_18 is False

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'minLength'
    var_2 = 'maxLength'
    var_3 = 'format'
    var_4 = 'default'
    var_5 = 'string'
    var_6 = 5
    var_7 = 10
    var_8 = 'email'
    var_9 = 'test@example.com'
    var_10 = {var_0: var_5, var_1: var_6, var_2: var_7, var_3: var_8, var_4: var_9}
    var_11 = False
    var_12 = []
    var_13 = {}
    var_14 = module_0.Definitions(*var_12, **var_13)
    var_15 = module_1.from_json_schema_type(var_10, var_5, var_11, var_14)
    var_16 = var_15.min_length
    assert var_16 == 5
    var_17 = var_15.max_length
    assert var_17 == 10
    var_18 = var_15.format
    assert var_18 == 'email'
    var_19 = var_15.default
    assert var_19 == 'test@example.com'
    var_20 = var_15.allow_null
    assert var_20 is False
    var_21 = var_15.allow_blank
    assert var_21 is False
    var_22 = var_15.coerce_types
    assert var_22 is False

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'minLength'
    var_2 = 'default'
    var_3 = 'string'
    var_4 = 0
    var_5 = ''
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = True
    var_8 = []
    var_9 = {}
    var_10 = module_0.Definitions(*var_8, **var_9)
    var_11 = module_1.from_json_schema_type(var_6, var_3, var_7, var_10)
    var_12 = var_11.min_length
    assert var_12 is None
    var_13 = var_11.default
    assert var_13 == ''
    var_14 = var_11.allow_null
    assert var_14 is True
    var_15 = var_11.allow_blank
    assert var_15 is True
    var_16 = var_11.coerce_types
    assert var_16 is False

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'default'
    var_2 = 'boolean'
    var_3 = True
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = False
    var_6 = []
    var_7 = {}
    var_8 = module_0.Definitions(*var_6, **var_7)
    var_9 = module_1.from_json_schema_type(var_4, var_2, var_5, var_8)
    var_10 = var_9.default
    assert var_10 is True
    var_11 = var_9.allow_null
    assert var_11 is False
    var_12 = var_9.coerce_types
    assert var_12 is False

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'items'
    var_2 = 'minItems'
    var_3 = 'maxItems'
    var_4 = 'default'
    var_5 = 'array'
    var_6 = 'string'
    var_7 = {var_0: var_6}
    var_8 = 1
    var_9 = 5
    var_10 = 'a'
    var_11 = 'b'
    var_12 = [var_10, var_11]
    var_13 = {var_0: var_5, var_1: var_7, var_2: var_8, var_3: var_9, var_4: var_12}
    var_14 = True
    var_15 = []
    var_16 = {}
    var_17 = module_0.Definitions(*var_15, **var_16)
    var_18 = module_1.from_json_schema_type(var_13, var_5, var_14, var_17)
    var_19 = var_18.items
    var_20 = var_18.min_items
    assert var_20 == 1
    var_21 = var_18.max_items
    assert var_21 == 5
    var_22 = var_18.default
    var_23 = bool(var_18.default == ['a', 'b'])
    assert var_23 is True
    var_24 = var_18.allow_null
    assert var_24 is True
    var_25 = var_18.additional_items
    assert var_25 is True
    var_26 = var_18.unique_items
    assert var_26 is False

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'items'
    var_2 = 'default'
    var_3 = 'array'
    var_4 = 'string'
    var_5 = {var_0: var_4}
    var_6 = 'number'
    var_7 = {var_0: var_6}
    var_8 = [var_5, var_7]
    var_9 = 'a'
    var_10 = 1
    var_11 = [var_9, var_10]
    var_12 = {var_0: var_3, var_1: var_8, var_2: var_11}
    var_13 = False
    var_14 = []
    var_15 = {}
    var_16 = module_0.Definitions(*var_14, **var_15)
    var_17 = module_1.from_json_schema_type(var_12, var_3, var_13, var_16)
    var_18 = var_17.items
    var_19 = var_17.items
    var_20 = len(var_19)
    assert var_20 == 2
    var_21 = var_17.items[var_13]
    var_22 = var_17.items[var_10]
    var_23 = var_17.default
    var_24 = bool(var_17.default == ['a', 1])
    assert var_24 is True
    var_25 = var_17.allow_null
    assert var_25 is False
    var_26 = var_17.min_items
    assert var_26 == 2
    var_27 = var_17.max_items
    assert var_27 == 2

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'properties'
    var_2 = 'required'
    var_3 = 'default'
    var_4 = 'object'
    var_5 = 'name'
    var_6 = 'age'
    var_7 = 'string'
    var_8 = {var_0: var_7}
    var_9 = 'integer'
    var_10 = {var_0: var_9}
    var_11 = {var_5: var_8, var_6: var_10}
    var_12 = [var_5]
    var_13 = 'John'
    var_14 = 30
    var_15 = {var_5: var_13, var_6: var_14}
    var_16 = {var_0: var_4, var_1: var_11, var_2: var_12, var_3: var_15}
    var_17 = True
    var_18 = []
    var_19 = {}
    var_20 = module_0.Definitions(*var_18, **var_19)
    var_21 = module_1.from_json_schema_type(var_16, var_4, var_17, var_20)
    var_22 = 'name'
    var_23 = bool('name' in var_21.properties)
    assert var_23 is True
    var_24 = 'age'
    var_25 = bool('age' in var_21.properties)
    assert var_25 is True
    var_26 = var_21.properties[var_5]
    var_27 = var_21.properties[var_6]
    var_28 = var_21.required
    var_29 = bool(var_21.required == ['name'])
    assert var_29 is True
    var_30 = var_21.default
    var_31 = bool(var_21.default == {'name': 'John', 'age': 30})
    assert var_31 is True
    var_32 = var_21.allow_null
    assert var_32 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_to_json_schema_with_reference_field. Retrieved 2/6 statements.


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
    var_0 = 1
    var_1 = 10
    var_2 = False
    var_3 = {}
    var_4 = module_0.String(allow_blank=var_2, max_length=var_1, min_length=var_0, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = bool(var_5 == {'type': 'string', 'minLength': 1, 'maxLength': 10})
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': ['string', 'null']})
    assert var_5 is True

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
    var_0 = 0.5
    var_1 = {}
    var_2 = module_0.Float(multiple_of=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'number', 'multipleOf': 0.5})
    assert var_4 is True

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
    var_2 = 1
    var_3 = 10
    var_4 = {}
    var_5 = module_0.Array(var_1, min_items=var_2, max_items=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = bool(var_6 == {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1, 'maxItems': 10})
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = bool(var_6 == {'type': 'object', 'properties': {'name': {'type': 'string'}}})
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'Option A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'Option B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = bool(var_9 == {'enum': ['a', 'b']})
    assert var_10 is True

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
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = bool(var_7 == {'anyOf': [{'type': 'string'}, {'type': 'integer'}]})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'string_field'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_1.Definitions(*var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = bool(var_7 == {'components': {'schemas': {'string_field': {'type': 'string'}}}})
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'string_ref'
    var_3 = 'target'
    var_4 = {var_3: var_1}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'string', 'default': 'default_value'})
    assert var_5 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_to_json_schema_with_reference_field. Retrieved 2/4 statements.


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
    var_3 = bool(var_2 == {'type': 'string'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': ['string', 'null']})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'string', 'minLength': 5})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'string', 'maxLength': 10})
    assert var_4 is True

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
    var_6 = bool(var_5 == {'type': 'string', 'pattern': '^[a-z]+$'})
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'string', 'format': 'email'})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'integer'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': ['integer', 'null']})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'number'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Float(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': ['number', 'null']})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'number'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Decimal(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': ['number', 'null']})
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
    var_1 = module_0.Array(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'array'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': ['array', 'null']})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'array', 'items': {'type': 'string'}})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.Array(additional_items=var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'array', 'additionalItems': {'type': 'integer'}})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'array', 'uniqueItems': True})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'object'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': ['object', 'null']})
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
    var_9 = module_1.to_json_schema(var_8)
    var_10 = bool(var_9 == {'type': 'object', 'properties': {'name': {'type': 'string'}, 'age': {'type': 'integer'}}})
    assert var_10 is True

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
    var_11 = bool(var_10 == {'type': 'object', 'properties': {'name': {'type': 'string'}, 'age': {'type': 'integer'}}, 'required': ['name']})
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

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
    var_9 = module_2.to_json_schema(var_8)
    var_10 = bool(var_9 == {'type': 'object', 'properties': {'name': {'type': 'string'}, 'age': {'type': 'integer'}}})
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

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
    var_11 = module_2.to_json_schema(var_10)
    var_12 = bool(var_11 == {'type': ['object', 'null'], 'properties': {'name': {'type': 'string'}, 'age': {'type': 'integer'}}})
    assert var_12 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'Option A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'Option B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = bool(var_9 == {'enum': ['a', 'b']})
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'constant_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'const': 'constant_value'})
    assert var_4 is True

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
    var_8 = bool(var_7 == {'anyOf': [{'type': 'string'}, {'type': 'integer'}]})
    assert var_8 is True

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
    var_6 = module_1.OneOf(var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = bool(var_7 == {'oneOf': [{'type': 'string'}, {'type': 'integer'}]})
    assert var_8 is True

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
    var_6 = module_1.AllOf(var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = bool(var_7 == {'allOf': [{'type': 'string'}, {'type': 'integer'}]})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = {}
    var_5 = module_0.Boolean(**var_4)
    var_6 = {}
    var_7 = module_1.IfThenElse(var_1, var_3, var_5, **var_6)
    var_8 = module_2.to_json_schema(var_7)
    var_9 = bool(var_8 == {'if': {'type': 'string'}, 'then': {'type': 'integer'}, 'else': {'type': 'boolean'}})
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_1.Not(var_1, **var_2)
    var_4 = module_2.to_json_schema(var_3)
    var_5 = bool(var_4 == {'not': {'type': 'string'}})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'string_field'
    var_1 = 'integer_field'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_1.Definitions(*var_7, **var_8)
    var_10 = module_2.to_json_schema(var_9)
    var_11 = bool(var_10 == {'components': {'schemas': {'string_field': {'type': 'string'}, 'integer_field': {'type': 'integer'}}}})
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'string_field'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = 'target'
    var_4 = {var_3: var_2}



# Parsed testcases at query #10
#--------------------------




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
    var_3 = bool(var_2 == {'type': 'string'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': ['string', 'null']})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'string', 'default': 'default_value'})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'string', 'minLength': 5})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'string', 'maxLength': 10})
    assert var_4 is True

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
    var_6 = bool(var_5 == {'type': 'string', 'pattern': '^[a-z]+$'})
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'string', 'format': 'email'})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'integer'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': ['integer', 'null']})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'integer', 'minimum': 0})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'integer', 'maximum': 100})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'number'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'number'})
    assert var_3 is True

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
    var_1 = module_0.Array(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'array'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': ['array', 'null']})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'array', 'minItems': 1})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'array', 'maxItems': 10})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'array', 'items': {'type': 'string'}})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Array(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = bool(var_7 == {'type': 'array', 'items': [{'type': 'string'}, {'type': 'integer'}]})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = {}
    var_3 = module_0.Array(additional_items=var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'array', 'additionalItems': {'type': 'boolean'}})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'array', 'uniqueItems': True})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'object'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': ['object', 'null']})
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
    var_9 = module_1.to_json_schema(var_8)
    var_10 = bool(var_9 == {'type': 'object', 'properties': {'name': {'type': 'string'}, 'age': {'type': 'integer'}}})
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = '^S_'
    var_1 = '^I_'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = {}
    var_8 = module_0.Object(pattern_properties=var_6, **var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = bool(var_9 == {'type': 'object', 'patternProperties': {'^S_': {'type': 'string'}, '^I_': {'type': 'integer'}}})
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = {}
    var_3 = module_0.Object(additional_properties=var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'object', 'additionalProperties': {'type': 'boolean'}})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Object(property_names=var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'object', 'propertyNames': {'type': 'string'}})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Object(max_properties=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'object', 'maxProperties': 10})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'object', 'minProperties': 1})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Object(required=var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'object', 'required': ['name']})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

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
    var_9 = module_2.to_json_schema(var_8)
    var_10 = bool(var_9 == {'type': 'object', 'properties': {'name': {'type': 'string'}, 'age': {'type': 'integer'}}})
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = [var_0]
    var_8 = 'required'
    var_9 = {var_8: var_7}
    var_10 = module_1.Schema(var_6, **var_9)
    var_11 = module_2.to_json_schema(var_10)
    var_12 = bool(var_11 == {'type': 'object', 'properties': {'name': {'type': 'string'}, 'age': {'type': 'integer'}}, 'required': ['name']})
    assert var_12 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = bool(var_9 == {'enum': ['a', 'b']})
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'constant_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'const': 'constant_value'})
    assert var_4 is True

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
    var_8 = bool(var_7 == {'anyOf': [{'type': 'string'}, {'type': 'integer'}]})
    assert var_8 is True

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
    var_6 = module_1.OneOf(var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = bool(var_7 == {'oneOf': [{'type': 'string'}, {'type': 'integer'}]})
    assert var_8 is True

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
    var_6 = module_1.AllOf(var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = bool(var_7 == {'allOf': [{'type': 'string'}, {'type': 'integer'}]})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = {}
    var_5 = module_0.Boolean(**var_4)
    var_6 = {}
    var_7 = module_1.IfThenElse(var_1, var_3, var_5, **var_6)
    var_8 = module_2.to_json_schema(var_7)
    var_9 = bool(var_8 == {'if': {'type': 'string'}, 'then': {'type': 'integer'}, 'else': {'type': 'boolean'}})
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_1.Not(var_1, **var_2)
    var_4 = module_2.to_json_schema(var_3)
    var_5 = bool(var_4 == {'not': {'type': 'string'}})
    assert var_5 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_string_field_with_pattern_regex. Retrieved 1/5 statements.


def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = 'pattern'



# Parsed testcases at query #12
#--------------------------




import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'array'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = {}
    var_5 = module_0.Definitions(*var_3, **var_4)
    var_6 = False
    var_7 = module_1.from_json_schema_type(var_2, var_1, var_6, var_5)
    var_8 = var_7.additional_items
    assert var_8 is True



# Parsed testcases at query #13
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = None
    var_1 = {}
    var_2 = module_0.Array(additional_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'additionalItems'
    var_5 = bool('additionalItems' not in var_3)
    assert var_5 is True



# Parsed testcases at query #14
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = None
    var_3 = {}
    var_4 = module_1.IfThenElse(var_1, var_2, var_2, **var_3)
    var_5 = bool(not var_4.else_clause)
    assert var_5 is True



# Parsed testcases at query #15
#--------------------------




import re as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.compile(var_0)
    var_2 = 'pattern_regex'
    var_3 = {var_2: var_1}
    var_4 = module_1.String(**var_3)
    var_5 = var_4.pattern_regex
    var_6 = bool(var_4.pattern_regex is not None)
    assert var_6 is True



# Parsed testcases at query #16
#--------------------------




import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'object'
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = {}
    var_5 = module_0.from_json_schema_type(var_2, var_1, var_3, var_4)
    var_6 = var_5.additional_properties
    assert var_6 is None



# Parsed testcases at query #17
#--------------------------




import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'object'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = {}
    var_5 = module_0.Definitions(*var_3, **var_4)
    var_6 = False
    var_7 = module_1.from_json_schema_type(var_2, var_1, var_6, var_5)
    var_8 = var_7.additional_properties
    assert var_8 is None



# Parsed testcases at query #18
#--------------------------

# Failed to parse test_to_json_schema_with_schema_field.
# Partially parsed test_to_json_schema_with_definitions. Retrieved 2/3 statements.
# Partially parsed test_to_json_schema_with_reference_field. Retrieved 2/4 statements.


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
    var_3 = bool(var_2 == {'type': 'string'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': ['string', 'null']})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'string', 'minLength': 5})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'string', 'maxLength': 10})
    assert var_4 is True

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
    var_6 = bool(var_5 == {'type': 'string', 'pattern': '^[a-z]+$'})
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'string', 'format': 'email'})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'integer'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'number'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'number'})
    assert var_3 is True

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
    var_1 = module_0.Array(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'array'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'object'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = bool(var_9 == {'enum': ['a', 'b']})
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'constant'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'const': 'constant'})
    assert var_4 is True

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
    var_8 = bool(var_7 == {'anyOf': [{'type': 'string'}, {'type': 'integer'}]})
    assert var_8 is True

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
    var_6 = module_1.OneOf(var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = bool(var_7 == {'oneOf': [{'type': 'string'}, {'type': 'integer'}]})
    assert var_8 is True

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
    var_6 = module_1.AllOf(var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = bool(var_7 == {'allOf': [{'type': 'string'}, {'type': 'integer'}]})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = {}
    var_5 = module_0.Boolean(**var_4)
    var_6 = {}
    var_7 = module_1.IfThenElse(var_1, var_3, var_5, **var_6)
    var_8 = module_2.to_json_schema(var_7)
    var_9 = bool(var_8 == {'if': {'type': 'string'}, 'then': {'type': 'integer'}, 'else': {'type': 'boolean'}})
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_1.Not(var_1, **var_2)
    var_4 = module_2.to_json_schema(var_3)
    var_5 = bool(var_4 == {'not': {'type': 'string'}})
    assert var_5 is True

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Definitions(*var_0, **var_1)
    var_3 = {}
    var_4 = module_1.to_json_schema(var_2)
    var_5 = bool(var_4 == {'components': {'schemas': {'string_field': {'type': 'string'}}}})
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'string_ref'
    var_3 = 'target'
    var_4 = {var_3: var_1}



# Parsed testcases at query #19
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Integer(exclusive_minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'exclusiveMinimum'
    var_5 = bool('exclusiveMinimum' in var_3)
    assert var_5 is True
    var_6 = var_3['exclusiveMinimum']
    assert var_6 is True



# Parsed testcases at query #20
#--------------------------




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
    var_3 = bool(var_2 == {'type': 'string'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': ['string', 'null']})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'string', 'default': 'default_value'})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'string', 'minLength': 5})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'string', 'maxLength': 10})
    assert var_4 is True

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
    var_6 = bool(var_5 == {'type': 'string', 'pattern': '^[a-z]+$'})
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'string', 'format': 'email'})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'integer'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': ['integer', 'null']})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'integer', 'minimum': 0})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'integer', 'maximum': 100})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(exclusive_minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'integer', 'exclusiveMinimum': 0})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(exclusive_maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'integer', 'exclusiveMaximum': 100})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Integer(multiple_of=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'integer', 'multipleOf': 2})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'number'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Float(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': ['number', 'null']})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'number'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Decimal(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': ['number', 'null']})
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
    var_1 = module_0.Array(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'array'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': ['array', 'null']})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'array', 'minItems': 1})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'array', 'maxItems': 10})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'array', 'items': {'type': 'string'}})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = {}
    var_3 = module_0.Array(additional_items=var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'array', 'additionalItems': {'type': 'boolean'}})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'array', 'uniqueItems': True})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'object'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': ['object', 'null']})
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
    var_9 = module_1.to_json_schema(var_8)
    var_10 = bool(var_9 == {'type': 'object', 'properties': {'name': {'type': 'string'}, 'age': {'type': 'integer'}}})
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = '^S_'
    var_1 = '^I_'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = {}
    var_8 = module_0.Object(pattern_properties=var_6, **var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = bool(var_9 == {'type': 'object', 'patternProperties': {'^S_': {'type': 'string'}, '^I_': {'type': 'integer'}}})
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = {}
    var_3 = module_0.Object(additional_properties=var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'object', 'additionalProperties': {'type': 'boolean'}})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Object(property_names=var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'object', 'propertyNames': {'type': 'string'}})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Object(max_properties=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'object', 'maxProperties': 10})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'object', 'minProperties': 1})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Object(required=var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'object', 'required': ['name']})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

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
    var_9 = module_2.to_json_schema(var_8)
    var_10 = bool(var_9 == {'type': 'object', 'properties': {'name': {'type': 'string'}, 'age': {'type': 'integer'}}})
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = True
    var_1 = 'name'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = 'allow_null'
    var_6 = {var_5: var_0}
    var_7 = module_1.Schema(var_4, **var_6)
    var_8 = module_2.to_json_schema(var_7)
    var_9 = bool(var_8 == {'type': ['object', 'null'], 'properties': {'name': {'type': 'string'}}})
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = 'required'
    var_6 = {var_5: var_4}
    var_7 = module_1.Schema(var_3, **var_6)
    var_8 = module_2.to_json_schema(var_7)
    var_9 = bool(var_8 == {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']})
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'Option A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'Option B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = bool(var_9 == {'enum': ['a', 'b']})
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'constant_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'const': 'constant_value'})
    assert var_4 is True

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



# Parsed testcases at query #21
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = None
    var_3 = {}
    var_4 = module_0.Integer(**var_3)
    var_5 = {}
    var_6 = module_1.IfThenElse(var_1, var_2, var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = 'then'
    var_9 = bool('then' not in var_7)
    assert var_9 is True



# Parsed testcases at query #22
#--------------------------




import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'object'
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = {}
    var_5 = module_0.from_json_schema_type(var_2, var_1, var_3, var_4)
    var_6 = var_5.pattern_properties
    assert var_6 is None



# Parsed testcases at query #23
#--------------------------




import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'object'
    var_2 = {var_0: var_1}
    var_3 = 'object'
    var_4 = False
    var_5 = []
    var_6 = {}
    var_7 = module_0.Definitions(*var_5, **var_6)
    var_8 = module_1.from_json_schema_type(var_2, var_3, var_4, var_7)
    var_9 = var_8.pattern_properties
    assert var_9 is None



# Parsed testcases at query #24
#--------------------------




import typesystem.json_schema as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'object'
    var_2 = False
    var_3 = {}
    var_4 = module_0.from_json_schema_type(var_0, var_1, var_2, var_3)
    var_5 = var_4.property_names
    assert var_5 is None



# Parsed testcases at query #25
#--------------------------




import typesystem.json_schema as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'object'
    var_2 = False
    var_3 = {}
    var_4 = module_0.from_json_schema_type(var_0, var_1, var_2, var_3)
    var_5 = var_4.property_names
    assert var_5 is None



# Parsed testcases at query #26
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = None
    var_3 = {}
    var_4 = module_0.Integer(**var_3)
    var_5 = {}
    var_6 = module_1.IfThenElse(var_1, var_2, var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = 'then'
    var_9 = bool('then' not in var_7)
    assert var_9 is True



# Parsed testcases at query #27
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = None
    var_3 = {}
    var_4 = module_0.Array(var_1, var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = 'additionalItems'
    var_7 = bool('additionalItems' not in var_5)
    assert var_7 is True



# Parsed testcases at query #28
#--------------------------




import re as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.compile(var_0)
    var_2 = 'pattern_regex'
    var_3 = {var_2: var_1}
    var_4 = module_1.String(**var_3)
    var_5 = var_4.pattern_regex
    var_6 = bool(var_4.pattern_regex is not None)
    assert var_6 is True



# Parsed testcases at query #29
#--------------------------




import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'additionalProperties'
    var_2 = 'object'
    var_3 = None
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = {}
    var_7 = module_0.Definitions(*var_5, **var_6)
    var_8 = False
    var_9 = module_1.from_json_schema_type(var_4, var_2, var_8, var_7)
    var_10 = var_9.additional_properties
    assert var_10 is None



# Parsed testcases at query #30
#--------------------------




import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'object'
    var_2 = {var_0: var_1}
    var_3 = 'object'
    var_4 = False
    var_5 = {}
    var_6 = module_0.from_json_schema_type(var_2, var_3, var_4, var_5)
    var_7 = var_6.additional_properties
    assert var_7 is None



# Parsed testcases at query #31
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
    var_6 = 'then'
    var_7 = bool('then' not in var_5)
    assert var_7 is True



# Parsed testcases at query #32
#--------------------------




import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'array'
    var_2 = {var_0: var_1}
    var_3 = 'array'
    var_4 = False
    var_5 = []
    var_6 = {}
    var_7 = module_0.Definitions(*var_5, **var_6)
    var_8 = module_1.from_json_schema_type(var_2, var_3, var_4, var_7)
    var_9 = var_8.additional_items
    assert var_9 is True



# Parsed testcases at query #33
#--------------------------




import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = 'invalid_type'
    var_2 = False
    var_3 = []
    var_4 = {}
    var_5 = module_0.Definitions(*var_3, **var_4)
    var_6 = module_1.from_json_schema_type(var_0, var_1, var_2, var_5)
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #34
#--------------------------




import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'array'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = {}
    var_5 = module_0.Definitions(*var_3, **var_4)
    var_6 = False
    var_7 = module_1.from_json_schema_type(var_2, var_1, var_6, var_5)
    var_8 = var_7.additional_items
    assert var_8 is True



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_from_json_schema_type_with_additional_items_as_bool. Retrieved 7/8 statements.


import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'additionalItems'
    var_2 = 'array'
    var_3 = False
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {}
    var_6 = module_0.from_json_schema_type(var_4, var_2, var_3, var_5)
    var_7 = var_6.additional_items
    assert var_7 is False



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_pattern_regex_predicate. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'test'



# Parsed testcases at query #37
#--------------------------




import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = 'object'
    var_2 = False
    var_3 = []
    var_4 = {}
    var_5 = module_0.Definitions(*var_3, **var_4)
    var_6 = module_1.from_json_schema_type(var_0, var_1, var_2, var_5)
    var_7 = var_6.pattern_properties
    assert var_7 is None



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_from_json_schema_type_number. Retrieved 10/11 statements.
# Partially parsed test_from_json_schema_type_integer. Retrieved 10/11 statements.
# Partially parsed test_from_json_schema_type_string. Retrieved 10/11 statements.
# Partially parsed test_from_json_schema_type_boolean. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_type_array. Retrieved 10/11 statements.
# Partially parsed test_from_json_schema_type_object. Retrieved 10/11 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'number'
    var_4 = 0
    var_5 = 100
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = []
    var_8 = {}
    var_9 = module_0.Definitions(*var_7, **var_8)
    var_10 = False
    var_11 = module_1.from_json_schema_type(var_6, var_3, var_10, var_9)
    var_12 = var_11.minimum
    assert var_12 == 0
    var_13 = var_11.maximum
    assert var_13 == 100
    var_14 = var_11.allow_null
    assert var_14 is False

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'integer'
    var_4 = 0
    var_5 = 100
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = []
    var_8 = {}
    var_9 = module_0.Definitions(*var_7, **var_8)
    var_10 = False
    var_11 = module_1.from_json_schema_type(var_6, var_3, var_10, var_9)
    var_12 = var_11.minimum
    assert var_12 == 0
    var_13 = var_11.maximum
    assert var_13 == 100
    var_14 = var_11.allow_null
    assert var_14 is False

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
    var_10 = False
    var_11 = module_1.from_json_schema_type(var_6, var_3, var_10, var_9)
    var_12 = var_11.min_length
    assert var_12 == 5
    var_13 = var_11.max_length
    assert var_13 == 10
    var_14 = var_11.allow_null
    assert var_14 is False

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'boolean'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = {}
    var_5 = module_0.Definitions(*var_3, **var_4)
    var_6 = False
    var_7 = module_1.from_json_schema_type(var_2, var_1, var_6, var_5)
    var_8 = var_7.allow_null
    assert var_8 is False

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'minItems'
    var_2 = 'maxItems'
    var_3 = 'array'
    var_4 = 1
    var_5 = 10
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = []
    var_8 = {}
    var_9 = module_0.Definitions(*var_7, **var_8)
    var_10 = False
    var_11 = module_1.from_json_schema_type(var_6, var_3, var_10, var_9)
    var_12 = var_11.min_items
    assert var_12 == 1
    var_13 = var_11.max_items
    assert var_13 == 10
    var_14 = var_11.allow_null
    assert var_14 is False

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'minProperties'
    var_2 = 'maxProperties'
    var_3 = 'object'
    var_4 = 1
    var_5 = 10
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = []
    var_8 = {}
    var_9 = module_0.Definitions(*var_7, **var_8)
    var_10 = False
    var_11 = module_1.from_json_schema_type(var_6, var_3, var_10, var_9)
    var_12 = var_11.min_properties
    assert var_12 == 1
    var_13 = var_11.max_properties
    assert var_13 == 10
    var_14 = var_11.allow_null
    assert var_14 is False

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'default'
    var_2 = 'number'
    var_3 = 42
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = {}
    var_7 = module_0.Definitions(*var_5, **var_6)
    var_8 = False
    var_9 = module_1.from_json_schema_type(var_4, var_2, var_8, var_7)
    var_10 = var_9.get_default_value()
    assert var_10 == 42

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'number'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = {}
    var_5 = module_0.Definitions(*var_3, **var_4)
    var_6 = True
    var_7 = module_1.from_json_schema_type(var_2, var_1, var_6, var_5)
    var_8 = var_7.allow_null
    assert var_8 is True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_from_json_schema_type_number. Retrieved 11/12 statements.
# Partially parsed test_from_json_schema_type_integer. Retrieved 11/12 statements.
# Partially parsed test_from_json_schema_type_string. Retrieved 15/16 statements.
# Partially parsed test_from_json_schema_type_boolean. Retrieved 7/8 statements.
# Partially parsed test_from_json_schema_type_array. Retrieved 13/14 statements.
# Partially parsed test_from_json_schema_type_object. Retrieved 11/12 statements.
# Partially parsed test_from_json_schema_type_array_with_items. Retrieved 11/13 statements.
# Partially parsed test_from_json_schema_type_object_with_properties. Retrieved 13/15 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'minimum'
    var_1 = 'maximum'
    var_2 = 'default'
    var_3 = 0
    var_4 = 100
    var_5 = 50
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'number'
    var_8 = False
    var_9 = []
    var_10 = {}
    var_11 = module_0.Definitions(*var_9, **var_10)
    var_12 = module_1.from_json_schema_type(var_6, var_7, var_8, var_11)
    var_13 = var_12.allow_null
    assert var_13 is False
    var_14 = var_12.minimum
    assert var_14 == 0
    var_15 = var_12.maximum
    assert var_15 == 100
    var_16 = var_12.default
    assert var_16 == 50
    var_17 = var_12.coerce_types
    assert var_17 is False

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'minimum'
    var_1 = 'maximum'
    var_2 = 'default'
    var_3 = 0
    var_4 = 100
    var_5 = 50
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'integer'
    var_8 = False
    var_9 = []
    var_10 = {}
    var_11 = module_0.Definitions(*var_9, **var_10)
    var_12 = module_1.from_json_schema_type(var_6, var_7, var_8, var_11)
    var_13 = var_12.allow_null
    assert var_13 is False
    var_14 = var_12.minimum
    assert var_14 == 0
    var_15 = var_12.maximum
    assert var_15 == 100
    var_16 = var_12.default
    assert var_16 == 50
    var_17 = var_12.coerce_types
    assert var_17 is False

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
    var_8 = '^[a-zA-Z0-9]+$'
    var_9 = 'test'
    var_10 = {var_0: var_5, var_1: var_6, var_2: var_7, var_3: var_8, var_4: var_9}
    var_11 = 'string'
    var_12 = False
    var_13 = []
    var_14 = {}
    var_15 = module_0.Definitions(*var_13, **var_14)
    var_16 = module_1.from_json_schema_type(var_10, var_11, var_12, var_15)
    var_17 = var_16.allow_null
    assert var_17 is False
    var_18 = var_16.allow_blank
    assert var_18 is False
    var_19 = var_16.min_length
    assert var_19 == 5
    var_20 = var_16.max_length
    assert var_20 == 10
    var_21 = var_16.format
    assert var_21 == 'email'
    var_22 = var_16.pattern
    assert var_22 == '^[a-zA-Z0-9]+$'
    var_23 = var_16.default
    assert var_23 == 'test'
    var_24 = var_16.coerce_types
    assert var_24 is False

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'default'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'boolean'
    var_4 = False
    var_5 = []
    var_6 = {}
    var_7 = module_0.Definitions(*var_5, **var_6)
    var_8 = module_1.from_json_schema_type(var_2, var_3, var_4, var_7)
    var_9 = var_8.allow_null
    assert var_9 is False
    var_10 = var_8.default
    assert var_10 is True
    var_11 = var_8.coerce_types
    assert var_11 is False

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'minItems'
    var_1 = 'maxItems'
    var_2 = 'uniqueItems'
    var_3 = 'default'
    var_4 = 1
    var_5 = 5
    var_6 = True
    var_7 = []
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = 'array'
    var_10 = False
    var_11 = []
    var_12 = {}
    var_13 = module_0.Definitions(*var_11, **var_12)
    var_14 = module_1.from_json_schema_type(var_8, var_9, var_10, var_13)
    var_15 = var_14.allow_null
    assert var_15 is False
    var_16 = var_14.min_items
    assert var_16 == 1
    var_17 = var_14.max_items
    assert var_17 == 5
    var_18 = var_14.unique_items
    assert var_18 is True
    var_19 = var_14.default
    var_20 = bool(var_14.default == [])
    assert var_20 is True

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'minProperties'
    var_1 = 'maxProperties'
    var_2 = 'default'
    var_3 = 1
    var_4 = 5
    var_5 = {}
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'object'
    var_8 = False
    var_9 = []
    var_10 = {}
    var_11 = module_0.Definitions(*var_9, **var_10)
    var_12 = module_1.from_json_schema_type(var_6, var_7, var_8, var_11)
    var_13 = var_12.allow_null
    assert var_13 is False
    var_14 = var_12.min_properties
    assert var_14 == 1
    var_15 = var_12.max_properties
    assert var_15 == 5
    var_16 = var_12.default
    var_17 = bool(var_12.default == {})
    assert var_17 is True

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'items'
    var_1 = 'additionalItems'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = False
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = []
    var_8 = {}
    var_9 = module_0.Definitions(*var_7, **var_8)
    var_10 = 'array'
    var_11 = module_1.from_json_schema_type(var_6, var_10, var_5, var_9)
    var_12 = var_11.items
    var_13 = var_11.additional_items
    assert var_13 is False

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'properties'
    var_1 = 'additionalProperties'
    var_2 = 'name'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = False
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = []
    var_10 = {}
    var_11 = module_0.Definitions(*var_9, **var_10)
    var_12 = 'object'
    var_13 = module_1.from_json_schema_type(var_8, var_12, var_7, var_11)
    var_14 = var_13.properties[var_2]
    var_15 = var_13.additional_properties
    assert var_15 is False



# Parsed testcases at query #40
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = None
    var_1 = {}
    var_2 = module_0.Array(additional_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'additionalItems'
    var_5 = bool('additionalItems' not in var_3)
    assert var_5 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 2/4 statements.


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
    var_3 = bool(var_2 == {'type': 'string'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': ['string', 'null']})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'string', 'minLength': 5})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'string', 'maxLength': 10})
    assert var_4 is True

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
    var_6 = bool(var_5 == {'type': 'string', 'pattern': '^[a-z]+$'})
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'string', 'format': 'email'})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'integer'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': ['integer', 'null']})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'integer', 'minimum': 0})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'integer', 'maximum': 100})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'number'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Float(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': ['number', 'null']})
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
    var_1 = module_0.Array(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'array'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': ['array', 'null']})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'array', 'minItems': 1})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'array', 'maxItems': 10})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'array', 'items': {'type': 'string'}})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = {}
    var_3 = module_0.Array(additional_items=var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'array', 'additionalItems': {'type': 'boolean'}})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'array', 'uniqueItems': True})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'object'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': ['object', 'null']})
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
    var_9 = module_1.to_json_schema(var_8)
    var_10 = bool(var_9 == {'type': 'object', 'properties': {'name': {'type': 'string'}, 'age': {'type': 'integer'}}})
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(pattern_properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = bool(var_6 == {'type': 'object', 'patternProperties': {'^[a-z]+$': {'type': 'string'}}})
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = {}
    var_3 = module_0.Object(additional_properties=var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'object', 'additionalProperties': {'type': 'boolean'}})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Object(property_names=var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'object', 'propertyNames': {'type': 'string'}})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Object(max_properties=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'object', 'maxProperties': 10})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'object', 'minProperties': 1})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Object(required=var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'object', 'required': ['name']})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

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
    var_9 = module_2.to_json_schema(var_8)
    var_10 = bool(var_9 == {'type': 'object', 'properties': {'name': {'type': 'string'}, 'age': {'type': 'integer'}}})
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = True
    var_1 = 'name'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = 'allow_null'
    var_6 = {var_5: var_0}
    var_7 = module_1.Schema(var_4, **var_6)
    var_8 = module_2.to_json_schema(var_7)
    var_9 = bool(var_8 == {'type': ['object', 'null'], 'properties': {'name': {'type': 'string'}}})
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {var_0: var_3}
    var_5 = 'required'
    var_6 = {var_5: var_1}
    var_7 = module_1.Schema(var_4, **var_6)
    var_8 = module_2.to_json_schema(var_7)
    var_9 = bool(var_8 == {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']})
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'Option A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'Option B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = bool(var_9 == {'enum': ['a', 'b']})
    assert var_10 is True

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
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = bool(var_7 == {'anyOf': [{'type': 'string'}, {'type': 'integer'}]})
    assert var_8 is True

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
    var_6 = module_1.OneOf(var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = bool(var_7 == {'oneOf': [{'type': 'string'}, {'type': 'integer'}]})
    assert var_8 is True

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
    var_6 = module_1.AllOf(var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = bool(var_7 == {'allOf': [{'type': 'string'}, {'type': 'integer'}]})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = {}
    var_5 = module_0.Boolean(**var_4)
    var_6 = {}
    var_7 = module_1.IfThenElse(var_1, var_3, var_5, **var_6)
    var_8 = module_2.to_json_schema(var_7)
    var_9 = bool(var_8 == {'if': {'type': 'string'}, 'then': {'type': 'integer'}, 'else': {'type': 'boolean'}})
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_1.Not(var_1, **var_2)
    var_4 = module_2.to_json_schema(var_3)
    var_5 = bool(var_4 == {'not': {'type': 'string'}})
    assert var_5 is True

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Definitions(*var_0, **var_1)
    var_3 = {}
    var_4 = {}
    var_5 = module_1.to_json_schema(var_2)
    var_6 = bool(var_5 == {'components': {'schemas': {'string_field': {'type': 'string'}, 'integer_field': {'type': 'integer'}}}})
    assert var_6 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_from_json_schema_type_number. Retrieved 10/11 statements.
# Partially parsed test_from_json_schema_type_integer. Retrieved 10/11 statements.
# Partially parsed test_from_json_schema_type_string. Retrieved 10/11 statements.
# Partially parsed test_from_json_schema_type_boolean. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_type_array. Retrieved 10/11 statements.
# Partially parsed test_from_json_schema_type_object. Retrieved 10/11 statements.
# Partially parsed test_from_json_schema_type_with_allow_null. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_type_with_default. Retrieved 8/9 statements.
# Partially parsed test_from_json_schema_type_with_items. Retrieved 10/12 statements.
# Partially parsed test_from_json_schema_type_with_additional_items. Retrieved 10/12 statements.
# Partially parsed test_from_json_schema_type_with_properties. Retrieved 12/14 statements.
# Partially parsed test_from_json_schema_type_with_pattern_properties. Retrieved 12/14 statements.
# Partially parsed test_from_json_schema_type_with_additional_properties. Retrieved 10/12 statements.
# Partially parsed test_from_json_schema_type_with_property_names. Retrieved 10/12 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'number'
    var_4 = 0
    var_5 = 100
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = False
    var_8 = []
    var_9 = {}
    var_10 = module_0.Definitions(*var_8, **var_9)
    var_11 = module_1.from_json_schema_type(var_6, var_3, var_7, var_10)
    var_12 = var_11.minimum
    assert var_12 == 0
    var_13 = var_11.maximum
    assert var_13 == 100
    var_14 = var_11.allow_null
    assert var_14 is False

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'integer'
    var_4 = 0
    var_5 = 100
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = False
    var_8 = []
    var_9 = {}
    var_10 = module_0.Definitions(*var_8, **var_9)
    var_11 = module_1.from_json_schema_type(var_6, var_3, var_7, var_10)
    var_12 = var_11.minimum
    assert var_12 == 0
    var_13 = var_11.maximum
    assert var_13 == 100
    var_14 = var_11.allow_null
    assert var_14 is False

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
    var_7 = False
    var_8 = []
    var_9 = {}
    var_10 = module_0.Definitions(*var_8, **var_9)
    var_11 = module_1.from_json_schema_type(var_6, var_3, var_7, var_10)
    var_12 = var_11.min_length
    assert var_12 == 5
    var_13 = var_11.max_length
    assert var_13 == 10
    var_14 = var_11.allow_null
    assert var_14 is False

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'boolean'
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = []
    var_5 = {}
    var_6 = module_0.Definitions(*var_4, **var_5)
    var_7 = module_1.from_json_schema_type(var_2, var_1, var_3, var_6)
    var_8 = var_7.allow_null
    assert var_8 is False

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'minItems'
    var_2 = 'maxItems'
    var_3 = 'array'
    var_4 = 1
    var_5 = 10
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = False
    var_8 = []
    var_9 = {}
    var_10 = module_0.Definitions(*var_8, **var_9)
    var_11 = module_1.from_json_schema_type(var_6, var_3, var_7, var_10)
    var_12 = var_11.min_items
    assert var_12 == 1
    var_13 = var_11.max_items
    assert var_13 == 10
    var_14 = var_11.allow_null
    assert var_14 is False

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'minProperties'
    var_2 = 'maxProperties'
    var_3 = 'object'
    var_4 = 1
    var_5 = 10
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = False
    var_8 = []
    var_9 = {}
    var_10 = module_0.Definitions(*var_8, **var_9)
    var_11 = module_1.from_json_schema_type(var_6, var_3, var_7, var_10)
    var_12 = var_11.min_properties
    assert var_12 == 1
    var_13 = var_11.max_properties
    assert var_13 == 10
    var_14 = var_11.allow_null
    assert var_14 is False

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'number'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = []
    var_5 = {}
    var_6 = module_0.Definitions(*var_4, **var_5)
    var_7 = module_1.from_json_schema_type(var_2, var_1, var_3, var_6)
    var_8 = var_7.allow_null
    assert var_8 is True

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'default'
    var_2 = 'number'
    var_3 = 42
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = False
    var_6 = []
    var_7 = {}
    var_8 = module_0.Definitions(*var_6, **var_7)
    var_9 = module_1.from_json_schema_type(var_4, var_2, var_5, var_8)
    var_10 = var_9.default
    assert var_10 == 42

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'items'
    var_2 = 'array'
    var_3 = 'string'
    var_4 = {var_0: var_3}
    var_5 = {var_0: var_2, var_1: var_4}
    var_6 = []
    var_7 = {}
    var_8 = module_0.Definitions(*var_6, **var_7)
    var_9 = False
    var_10 = module_1.from_json_schema_type(var_5, var_2, var_9, var_8)
    var_11 = var_10.items

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'additionalItems'
    var_2 = 'array'
    var_3 = 'number'
    var_4 = {var_0: var_3}
    var_5 = {var_0: var_2, var_1: var_4}
    var_6 = []
    var_7 = {}
    var_8 = module_0.Definitions(*var_6, **var_7)
    var_9 = False
    var_10 = module_1.from_json_schema_type(var_5, var_2, var_9, var_8)
    var_11 = var_10.additional_items

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'properties'
    var_2 = 'object'
    var_3 = 'name'
    var_4 = 'string'
    var_5 = {var_0: var_4}
    var_6 = {var_3: var_5}
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = []
    var_9 = {}
    var_10 = module_0.Definitions(*var_8, **var_9)
    var_11 = False
    var_12 = module_1.from_json_schema_type(var_7, var_2, var_11, var_10)
    var_13 = var_12.properties[var_3]

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'patternProperties'
    var_2 = 'object'
    var_3 = '^S_'
    var_4 = 'string'
    var_5 = {var_0: var_4}
    var_6 = {var_3: var_5}
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = []
    var_9 = {}
    var_10 = module_0.Definitions(*var_8, **var_9)
    var_11 = False
    var_12 = module_1.from_json_schema_type(var_7, var_2, var_11, var_10)
    var_13 = var_12.pattern_properties[var_3]

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'additionalProperties'
    var_2 = 'object'
    var_3 = 'number'
    var_4 = {var_0: var_3}
    var_5 = {var_0: var_2, var_1: var_4}
    var_6 = []
    var_7 = {}
    var_8 = module_0.Definitions(*var_6, **var_7)
    var_9 = False
    var_10 = module_1.from_json_schema_type(var_5, var_2, var_9, var_8)
    var_11 = var_10.additional_properties

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'propertyNames'
    var_2 = 'object'
    var_3 = 'string'
    var_4 = {var_0: var_3}
    var_5 = {var_0: var_2, var_1: var_4}
    var_6 = []
    var_7 = {}
    var_8 = module_0.Definitions(*var_6, **var_7)
    var_9 = False
    var_10 = module_1.from_json_schema_type(var_5, var_2, var_9, var_8)
    var_11 = var_10.property_names



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 2/3 statements.
# Partially parsed test_to_json_schema_with_reference_field. Retrieved 2/4 statements.


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

import re as module_0
import typesystem.fields as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = '^[a-z]+$'
    var_3 = module_0.compile(var_2)
    var_4 = 'email'
    var_5 = 'pattern_regex'
    var_6 = {var_5: var_3}
    var_7 = module_1.String(max_length=var_1, min_length=var_0, format=var_4, **var_6)
    var_8 = module_2.to_json_schema(var_7)
    var_9 = bool(var_8 == {'type': 'string', 'minLength': 5, 'maxLength': 10, 'pattern': '^[a-z]+$', 'format': 'email'})
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = 100
    var_2 = True
    var_3 = 2
    var_4 = {}
    var_5 = module_0.Integer(minimum=var_0, maximum=var_1, exclusive_minimum=var_2, exclusive_maximum=var_2, multiple_of=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = bool(var_6 == {'type': 'integer', 'minimum': 0, 'maximum': 100, 'exclusiveMinimum': True, 'exclusiveMaximum': True, 'multipleOf': 2})
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0.0
    var_1 = 100.0
    var_2 = {}
    var_3 = module_0.Float(minimum=var_0, maximum=var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'number', 'minimum': 0.0, 'maximum': 100.0})
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
    var_0 = 1
    var_1 = 10
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = False
    var_5 = True
    var_6 = {}
    var_7 = module_0.Array(var_3, var_4, var_0, var_1, unique_items=var_5, **var_6)
    var_8 = module_1.to_json_schema(var_7)
    var_9 = bool(var_8 == {'type': 'array', 'minItems': 1, 'maxItems': 10, 'items': {'type': 'string'}, 'additionalItems': False, 'uniqueItems': True})
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = '^S_'
    var_5 = {}
    var_6 = module_0.String(**var_5)
    var_7 = {var_4: var_6}
    var_8 = False
    var_9 = {}
    var_10 = module_0.String(**var_9)
    var_11 = 10
    var_12 = 1
    var_13 = [var_0]
    var_14 = {}
    var_15 = module_0.Object(properties=var_3, pattern_properties=var_7, additional_properties=var_8, property_names=var_10, min_properties=var_12, max_properties=var_11, required=var_13, **var_14)
    var_16 = module_1.to_json_schema(var_15)
    var_17 = bool(var_16 == {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'patternProperties': {'^S_': {'type': 'string'}}, 'additionalProperties': False, 'propertyNames': {'type': 'string'}, 'maxProperties': 10, 'minProperties': 1, 'required': ['name']})
    assert var_17 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = 'required'
    var_6 = {var_5: var_4}
    var_7 = module_1.Schema(var_3, **var_6)
    var_8 = module_2.to_json_schema(var_7)
    var_9 = bool(var_8 == {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']})
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'Option A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'Option B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = bool(var_9 == {'enum': ['a', 'b']})
    assert var_10 is True

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
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = bool(var_7 == {'anyOf': [{'type': 'string'}, {'type': 'integer'}]})
    assert var_8 is True

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
    var_6 = module_1.OneOf(var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = bool(var_7 == {'oneOf': [{'type': 'string'}, {'type': 'integer'}]})
    assert var_8 is True

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
    var_6 = module_1.AllOf(var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = bool(var_7 == {'allOf': [{'type': 'string'}, {'type': 'integer'}]})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = {}
    var_5 = module_0.Boolean(**var_4)
    var_6 = {}
    var_7 = module_1.IfThenElse(var_1, var_3, var_5, **var_6)
    var_8 = module_2.to_json_schema(var_7)
    var_9 = bool(var_8 == {'if': {'type': 'string'}, 'then': {'type': 'integer'}, 'else': {'type': 'boolean'}})
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_1.Not(var_1, **var_2)
    var_4 = module_2.to_json_schema(var_3)
    var_5 = bool(var_4 == {'not': {'type': 'string'}})
    assert var_5 is True

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Definitions(*var_0, **var_1)
    var_3 = {}
    var_4 = module_1.to_json_schema(var_2)
    var_5 = bool(var_4 == {'components': {'schemas': {'string_field': {'type': 'string'}}}})
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'string_ref'
    var_3 = 'target'
    var_4 = {var_3: var_1}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': ['string', 'null']})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'string', 'default': 'default_value'})
    assert var_5 is True

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'invalid_field_type'
    var_1 = module_0.to_json_schema(var_0)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_type_from_json_schema_single_type. Retrieved 5/6 statements.
# Partially parsed test_type_from_json_schema_multiple_types. Retrieved 13/16 statements.
# Partially parsed test_type_from_json_schema_with_null. Retrieved 7/8 statements.
# Partially parsed test_type_from_json_schema_no_type. Retrieved 3/4 statements.
# Partially parsed test_type_from_json_schema_null_only. Retrieved 5/6 statements.
# Partially parsed test_type_from_json_schema_integer. Retrieved 9/10 statements.
# Partially parsed test_type_from_json_schema_number. Retrieved 7/8 statements.
# Partially parsed test_type_from_json_schema_boolean. Retrieved 5/6 statements.
# Partially parsed test_type_from_json_schema_array. Retrieved 9/11 statements.
# Partially parsed test_type_from_json_schema_object. Retrieved 11/13 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
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
    var_1 = 'string'
    var_2 = 'integer'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = []
    var_6 = {}
    var_7 = module_0.Definitions(*var_5, **var_6)
    var_8 = module_1.type_from_json_schema(var_4, var_7)
    var_9 = var_8.any_of
    var_10 = len(var_9)
    assert var_10 == 2
    var_11 = 0
    var_12 = var_8.any_of[var_11]
    var_13 = 1
    var_14 = var_8.any_of[var_13]
    var_15 = var_8.allow_null
    assert var_15 is False

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
    var_0 = {}
    var_1 = []
    var_2 = {}
    var_3 = module_0.Definitions(*var_1, **var_2)
    var_4 = module_1.type_from_json_schema(var_0, var_3)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'null'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = {}
    var_5 = module_0.Definitions(*var_3, **var_4)
    var_6 = module_1.type_from_json_schema(var_2, var_5)
    var_7 = var_6.const
    assert var_7 is None

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'integer'
    var_4 = 0
    var_5 = 100
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = []
    var_8 = {}
    var_9 = module_0.Definitions(*var_7, **var_8)
    var_10 = module_1.type_from_json_schema(var_6, var_9)
    var_11 = var_10.minimum
    assert var_11 == 0
    var_12 = var_10.maximum
    assert var_12 == 100

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'multipleOf'
    var_2 = 'number'
    var_3 = 0.5
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = {}
    var_7 = module_0.Definitions(*var_5, **var_6)
    var_8 = module_1.type_from_json_schema(var_4, var_7)
    var_9 = var_8.multiple_of
    var_10 = bool(var_8.multiple_of == 0.5)
    assert var_10 is True

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'boolean'
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
    var_1 = 'items'
    var_2 = 'array'
    var_3 = 'string'
    var_4 = {var_0: var_3}
    var_5 = {var_0: var_2, var_1: var_4}
    var_6 = []
    var_7 = {}
    var_8 = module_0.Definitions(*var_6, **var_7)
    var_9 = module_1.type_from_json_schema(var_5, var_8)
    var_10 = var_9.items

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'properties'
    var_2 = 'object'
    var_3 = 'name'
    var_4 = 'string'
    var_5 = {var_0: var_4}
    var_6 = {var_3: var_5}
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = []
    var_9 = {}
    var_10 = module_0.Definitions(*var_8, **var_9)
    var_11 = module_1.type_from_json_schema(var_7, var_10)
    var_12 = 'name'
    var_13 = bool('name' in var_11.properties)
    assert var_13 is True
    var_14 = var_11.properties[var_3]



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_isinstance_items_list. Retrieved 4/5 statements.


def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = [var_2]



# Parsed testcases at query #6
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = None
    var_5 = {}
    var_6 = module_1.IfThenElse(var_1, var_3, var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = 'else'
    var_9 = bool('else' not in var_7)
    assert var_9 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_if_then_else_from_json_schema_with_all_clauses. Retrieved 18/22 statements.
# Partially parsed test_if_then_else_from_json_schema_without_then_clause. Retrieved 15/19 statements.
# Partially parsed test_if_then_else_from_json_schema_without_else_clause. Retrieved 15/19 statements.
# Partially parsed test_if_then_else_from_json_schema_without_default. Retrieved 16/20 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'if'
    var_1 = 'then'
    var_2 = 'else'
    var_3 = 'default'
    var_4 = 'type'
    var_5 = 'string'
    var_6 = {var_4: var_5}
    var_7 = 'number'
    var_8 = {var_4: var_7}
    var_9 = 'boolean'
    var_10 = {var_4: var_9}
    var_11 = 42
    var_12 = {var_0: var_6, var_1: var_8, var_2: var_10, var_3: var_11}
    var_13 = []
    var_14 = {}
    var_15 = module_0.Definitions(*var_13, **var_14)
    var_16 = module_1.if_then_else_from_json_schema(var_12, var_15)
    var_17 = var_16.if_clause
    var_18 = var_16.then_clause
    var_19 = var_16.else_clause
    var_20 = var_16.default
    assert var_20 == 42

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'if'
    var_1 = 'else'
    var_2 = 'default'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 'boolean'
    var_7 = {var_3: var_6}
    var_8 = 42
    var_9 = {var_0: var_5, var_1: var_7, var_2: var_8}
    var_10 = []
    var_11 = {}
    var_12 = module_0.Definitions(*var_10, **var_11)
    var_13 = module_1.if_then_else_from_json_schema(var_9, var_12)
    var_14 = var_13.if_clause
    var_15 = var_13.then_clause
    var_16 = var_13.else_clause
    var_17 = var_13.default
    assert var_17 == 42

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'if'
    var_1 = 'then'
    var_2 = 'default'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 'number'
    var_7 = {var_3: var_6}
    var_8 = 42
    var_9 = {var_0: var_5, var_1: var_7, var_2: var_8}
    var_10 = []
    var_11 = {}
    var_12 = module_0.Definitions(*var_10, **var_11)
    var_13 = module_1.if_then_else_from_json_schema(var_9, var_12)
    var_14 = var_13.if_clause
    var_15 = var_13.then_clause
    var_16 = var_13.else_clause
    var_17 = var_13.default
    assert var_17 == 42

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'if'
    var_1 = 'then'
    var_2 = 'else'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 'number'
    var_7 = {var_3: var_6}
    var_8 = 'boolean'
    var_9 = {var_3: var_8}
    var_10 = {var_0: var_5, var_1: var_7, var_2: var_9}
    var_11 = []
    var_12 = {}
    var_13 = module_0.Definitions(*var_11, **var_12)
    var_14 = module_1.if_then_else_from_json_schema(var_10, var_13)
    var_15 = var_14.if_clause
    var_16 = var_14.then_clause
    var_17 = var_14.else_clause
    var_18 = var_14.default



# Parsed testcases at query #8
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = None
    var_3 = {}
    var_4 = module_0.Array(var_1, var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = 'additionalItems'
    var_7 = bool('additionalItems' not in var_5)
    assert var_7 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_from_json_schema_type_object_additional_properties_bool. Retrieved 8/9 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'additionalProperties'
    var_2 = 'object'
    var_3 = True
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = {}
    var_7 = module_0.Definitions(*var_5, **var_6)
    var_8 = False
    var_9 = module_1.from_json_schema_type(var_4, var_2, var_8, var_7)
    var_10 = var_9.additional_properties
    assert var_10 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_from_json_schema_with_bool_true. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_with_bool_false. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_with_ref. Retrieved 5/7 statements.
# Partially parsed test_from_json_schema_with_type_constraint. Retrieved 8/10 statements.
# Partially parsed test_from_json_schema_with_enum. Retrieved 9/10 statements.
# Partially parsed test_from_json_schema_with_const. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_all_of. Retrieved 11/12 statements.
# Partially parsed test_from_json_schema_with_any_of. Retrieved 11/12 statements.
# Partially parsed test_from_json_schema_with_one_of. Retrieved 11/12 statements.
# Partially parsed test_from_json_schema_with_not. Retrieved 7/9 statements.
# Partially parsed test_from_json_schema_with_if_then_else. Retrieved 15/19 statements.
# Partially parsed test_from_json_schema_with_multiple_constraints. Retrieved 11/12 statements.
# Partially parsed test_from_json_schema_with_no_constraints. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_with_definitions. Retrieved 10/11 statements.


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
    var_4 = '#/components/schemas/test'
    var_5 = {var_3: var_4}
    var_6 = module_1.from_json_schema(var_5, var_2)
    var_7 = var_6.to
    assert var_7 == '#/components/schemas/test'

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)
    var_4 = var_3.any_of
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_3.any_of[var_6]

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'enum'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = module_0.from_json_schema(var_5)
    var_7 = var_6.choices
    var_8 = len(var_7)
    assert var_8 == 3

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'const'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)
    var_4 = var_3.const
    assert var_4 == 'test'

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
    var_6 = var_5.negated

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
    var_13 = var_11.then_clause
    var_14 = var_11.else_clause

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'enum'
    var_2 = 'string'
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = module_0.from_json_schema(var_7)
    var_9 = var_8.all_of
    var_10 = len(var_9)
    assert var_10 == 2

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.from_json_schema(var_0)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'components'
    var_1 = 'schemas'
    var_2 = 'test'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = module_0.from_json_schema(var_8)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_to_json_schema_with_reference_field. Retrieved 2/4 statements.


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

import re as module_0
import typesystem.fields as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = '^[a-z]+$'
    var_3 = module_0.compile(var_2)
    var_4 = 'email'
    var_5 = 'pattern_regex'
    var_6 = {var_5: var_3}
    var_7 = module_1.String(max_length=var_1, min_length=var_0, format=var_4, **var_6)
    var_8 = module_2.to_json_schema(var_7)
    var_9 = bool(var_8 == {'type': 'string', 'minLength': 1, 'maxLength': 10, 'pattern': '^[a-z]+$', 'format': 'email'})
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = 100
    var_2 = True
    var_3 = 2
    var_4 = {}
    var_5 = module_0.Integer(minimum=var_0, maximum=var_1, exclusive_minimum=var_2, exclusive_maximum=var_2, multiple_of=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = bool(var_6 == {'type': 'integer', 'minimum': 0, 'maximum': 100, 'exclusiveMinimum': True, 'exclusiveMaximum': True, 'multipleOf': 2})
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0.0
    var_1 = 100.0
    var_2 = True
    var_3 = 0.5
    var_4 = {}
    var_5 = module_0.Float(minimum=var_0, maximum=var_1, exclusive_minimum=var_2, exclusive_maximum=var_2, multiple_of=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = bool(var_6 == {'type': 'number', 'minimum': 0.0, 'maximum': 100.0, 'exclusiveMinimum': True, 'exclusiveMaximum': True, 'multipleOf': 0.5})
    assert var_7 is True

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
    var_0 = 1
    var_1 = 10
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = False
    var_5 = True
    var_6 = {}
    var_7 = module_0.Array(var_3, var_4, var_0, var_1, unique_items=var_5, **var_6)
    var_8 = module_1.to_json_schema(var_7)
    var_9 = bool(var_8 == {'type': 'array', 'minItems': 1, 'maxItems': 10, 'items': {'type': 'string'}, 'additionalItems': False, 'uniqueItems': True})
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = '^S_'
    var_5 = {}
    var_6 = module_0.String(**var_5)
    var_7 = {var_4: var_6}
    var_8 = False
    var_9 = {}
    var_10 = module_0.String(**var_9)
    var_11 = 10
    var_12 = 1
    var_13 = [var_0]
    var_14 = {}
    var_15 = module_0.Object(properties=var_3, pattern_properties=var_7, additional_properties=var_8, property_names=var_10, min_properties=var_12, max_properties=var_11, required=var_13, **var_14)
    var_16 = module_1.to_json_schema(var_15)
    var_17 = bool(var_16 == {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'patternProperties': {'^S_': {'type': 'string'}}, 'additionalProperties': False, 'propertyNames': {'type': 'string'}, 'maxProperties': 10, 'minProperties': 1, 'required': ['name']})
    assert var_17 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = 'required'
    var_6 = {var_5: var_4}
    var_7 = module_1.Schema(var_3, **var_6)
    var_8 = module_2.to_json_schema(var_7)
    var_9 = bool(var_8 == {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']})
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = bool(var_9 == {'enum': ['a', 'b']})
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'constant_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'const': 'constant_value'})
    assert var_4 is True

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
    var_8 = bool(var_7 == {'anyOf': [{'type': 'string'}, {'type': 'integer'}]})
    assert var_8 is True

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
    var_6 = module_1.OneOf(var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = bool(var_7 == {'oneOf': [{'type': 'string'}, {'type': 'integer'}]})
    assert var_8 is True

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
    var_6 = module_1.AllOf(var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = bool(var_7 == {'allOf': [{'type': 'string'}, {'type': 'integer'}]})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = {}
    var_5 = module_0.Boolean(**var_4)
    var_6 = {}
    var_7 = module_1.IfThenElse(var_1, var_3, var_5, **var_6)
    var_8 = module_2.to_json_schema(var_7)
    var_9 = bool(var_8 == {'if': {'type': 'string'}, 'then': {'type': 'integer'}, 'else': {'type': 'boolean'}})
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_1.Not(var_1, **var_2)
    var_4 = module_2.to_json_schema(var_3)
    var_5 = bool(var_4 == {'not': {'type': 'string'}})
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'test'
    var_3 = 'target'
    var_4 = {var_3: var_1}

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_1.Definitions(*var_7, **var_8)
    var_10 = module_2.to_json_schema(var_9)
    var_11 = bool(var_10 == {'components': {'schemas': {'field1': {'type': 'string'}, 'field2': {'type': 'integer'}}}})
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'string', 'default': 'default_value'})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': ['string', 'null']})
    assert var_5 is True



# Parsed testcases at query #12
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = None
    var_3 = {}
    var_4 = module_0.Integer(**var_3)
    var_5 = {}
    var_6 = module_1.IfThenElse(var_1, var_2, var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = 'then'
    var_9 = bool('then' not in var_7)
    assert var_9 is True



# Parsed testcases at query #13
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = None
    var_3 = {}
    var_4 = module_0.Integer(**var_3)
    var_5 = {}
    var_6 = module_1.IfThenElse(var_1, var_2, var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = 'then'
    var_9 = bool('then' not in var_7)
    assert var_9 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_pattern_regex_is_not_none. Retrieved 1/4 statements.


def test_case_0():
    var_0 = '^test$'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_string_field_with_pattern_regex. Retrieved 1/5 statements.


def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = 'pattern'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_array_items_is_list_or_tuple. Retrieved 7/8 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Array(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'items'
    var_9 = var_7[var_8]



# Parsed testcases at query #17
#--------------------------




import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = 'invalid_type'
    var_2 = False
    var_3 = []
    var_4 = {}
    var_5 = module_0.Definitions(*var_3, **var_4)
    var_6 = module_1.from_json_schema_type(var_0, var_1, var_2, var_5)
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #18
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = None
    var_1 = {}
    var_2 = module_0.Array(additional_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'additionalItems'
    var_5 = bool('additionalItems' not in var_3)
    assert var_5 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_to_json_schema_with_decimal_field_with_default. Retrieved 1/4 statements.
# Partially parsed test_to_json_schema_with_schema_field. Retrieved 6/7 statements.


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
    var_3 = bool(var_2 == {'type': 'string'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': ['string', 'null']})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'string', 'default': 'default_value'})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'string', 'minLength': 5})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'string', 'maxLength': 10})
    assert var_4 is True

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
    var_6 = bool(var_5 == {'type': 'string', 'pattern': '^[a-z]+$'})
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'string', 'format': 'email'})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'integer'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': ['integer', 'null']})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 42
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'integer', 'default': 42})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'integer', 'minimum': 0})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'integer', 'maximum': 100})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(exclusive_minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'integer', 'exclusiveMinimum': 0})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(exclusive_maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'integer', 'exclusiveMaximum': 100})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Integer(multiple_of=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'integer', 'multipleOf': 2})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'number'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Float(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': ['number', 'null']})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 3.14
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.Float(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'number', 'default': 3.14})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'number'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Decimal(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': ['number', 'null']})
    assert var_5 is True

def test_case_0():
    var_0 = '3.14'

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
    var_1 = module_0.Array(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'array'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': ['array', 'null']})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = []
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'array', 'default': []})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'array', 'minItems': 1})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'array', 'maxItems': 10})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'array', 'items': {'type': 'string'}})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Boolean(**var_5)
    var_7 = {}
    var_8 = module_0.Array(var_4, var_6, **var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = bool(var_9 == {'type': 'array', 'items': [{'type': 'string'}, {'type': 'integer'}], 'additionalItems': {'type': 'boolean'}})
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'array', 'uniqueItems': True})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'object'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': ['object', 'null']})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'object', 'default': {}})
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
    var_9 = module_1.to_json_schema(var_8)
    var_10 = bool(var_9 == {'type': 'object', 'properties': {'name': {'type': 'string'}, 'age': {'type': 'integer'}}})
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = '^S_'
    var_1 = '^I_'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = {}
    var_8 = module_0.Object(pattern_properties=var_6, **var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = bool(var_9 == {'type': 'object', 'patternProperties': {'^S_': {'type': 'string'}, '^I_': {'type': 'integer'}}})
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = {}
    var_3 = module_0.Object(additional_properties=var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'object', 'additionalProperties': {'type': 'boolean'}})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Object(property_names=var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'object', 'propertyNames': {'type': 'string'}})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Object(max_properties=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'object', 'maxProperties': 10})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'object', 'minProperties': 1})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Object(required=var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'object', 'required': ['name']})
    assert var_5 is True

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



# Parsed testcases at query #20
#--------------------------

# Failed to parse test_to_json_schema_with_schema_field.
# Partially parsed test_to_json_schema_with_definitions. Retrieved 2/4 statements.
# Partially parsed test_to_json_schema_with_reference_field. Retrieved 2/4 statements.


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
    var_3 = bool(var_2 == {'type': 'string'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': ['string', 'null']})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = bool(var_4 == {'type': 'string', 'default': 'default_value'})
    assert var_5 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'string', 'minLength': 5})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'string', 'maxLength': 10})
    assert var_4 is True

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
    var_6 = bool(var_5 == {'type': 'string', 'pattern': '^[a-z]+$'})
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'type': 'string', 'format': 'email'})
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'integer'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'number'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'number'})
    assert var_3 is True

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
    var_1 = module_0.Array(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'array'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = bool(var_2 == {'type': 'object'})
    assert var_3 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'Option A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'Option B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = bool(var_9 == {'enum': ['a', 'b']})
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'constant_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = bool(var_3 == {'const': 'constant_value'})
    assert var_4 is True

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
    var_8 = bool(var_7 == {'anyOf': [{'type': 'string'}, {'type': 'integer'}]})
    assert var_8 is True

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Definitions(*var_0, **var_1)
    var_3 = {}
    var_4 = {}
    var_5 = module_1.to_json_schema(var_2)
    var_6 = bool(var_5 == {'components': {'schemas': {'string_field': {'type': 'string'}, 'integer_field': {'type': 'integer'}}}})
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'string_field'
    var_3 = 'target'
    var_4 = {var_3: var_1}

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
    var_6 = module_1.OneOf(var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = bool(var_7 == {'oneOf': [{'type': 'string'}, {'type': 'integer'}]})
    assert var_8 is True

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
    var_6 = module_1.AllOf(var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = bool(var_7 == {'allOf': [{'type': 'string'}, {'type': 'integer'}]})
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = {}
    var_5 = module_0.Boolean(**var_4)
    var_6 = {}
    var_7 = module_1.IfThenElse(var_1, var_3, var_5, **var_6)
    var_8 = module_2.to_json_schema(var_7)
    var_9 = bool(var_8 == {'if': {'type': 'string'}, 'then': {'type': 'integer'}, 'else': {'type': 'boolean'}})
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_1.Not(var_1, **var_2)
    var_4 = module_2.to_json_schema(var_3)
    var_5 = bool(var_4 == {'not': {'type': 'string'}})
    assert var_5 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_predicate_at_line_34. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'test'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_from_json_schema_type_number. Retrieved 10/11 statements.
# Partially parsed test_from_json_schema_type_integer. Retrieved 10/11 statements.
# Partially parsed test_from_json_schema_type_string. Retrieved 10/11 statements.
# Partially parsed test_from_json_schema_type_boolean. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_type_array. Retrieved 10/11 statements.
# Partially parsed test_from_json_schema_type_object. Retrieved 10/11 statements.
# Partially parsed test_from_json_schema_type_array_with_items. Retrieved 10/12 statements.
# Partially parsed test_from_json_schema_type_array_with_additional_items. Retrieved 13/15 statements.
# Partially parsed test_from_json_schema_type_object_with_properties. Retrieved 12/14 statements.
# Partially parsed test_from_json_schema_type_object_with_pattern_properties. Retrieved 12/14 statements.
# Partially parsed test_from_json_schema_type_object_with_additional_properties. Retrieved 10/12 statements.
# Partially parsed test_from_json_schema_type_object_with_property_names. Retrieved 10/12 statements.
# Partially parsed test_from_json_schema_type_object_with_required. Retrieved 9/10 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'number'
    var_4 = 0
    var_5 = 100
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = False
    var_8 = []
    var_9 = {}
    var_10 = module_0.Definitions(*var_8, **var_9)
    var_11 = module_1.from_json_schema_type(var_6, var_3, var_7, var_10)
    var_12 = var_11.minimum
    assert var_12 == 0
    var_13 = var_11.maximum
    assert var_13 == 100
    var_14 = var_11.allow_null
    assert var_14 is False

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'integer'
    var_4 = 0
    var_5 = 100
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = False
    var_8 = []
    var_9 = {}
    var_10 = module_0.Definitions(*var_8, **var_9)
    var_11 = module_1.from_json_schema_type(var_6, var_3, var_7, var_10)
    var_12 = var_11.minimum
    assert var_12 == 0
    var_13 = var_11.maximum
    assert var_13 == 100
    var_14 = var_11.allow_null
    assert var_14 is False

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
    var_7 = False
    var_8 = []
    var_9 = {}
    var_10 = module_0.Definitions(*var_8, **var_9)
    var_11 = module_1.from_json_schema_type(var_6, var_3, var_7, var_10)
    var_12 = var_11.min_length
    assert var_12 == 5
    var_13 = var_11.max_length
    assert var_13 == 10
    var_14 = var_11.allow_null
    assert var_14 is False

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'boolean'
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = []
    var_5 = {}
    var_6 = module_0.Definitions(*var_4, **var_5)
    var_7 = module_1.from_json_schema_type(var_2, var_1, var_3, var_6)
    var_8 = var_7.allow_null
    assert var_8 is False

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'minItems'
    var_2 = 'maxItems'
    var_3 = 'array'
    var_4 = 1
    var_5 = 5
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = False
    var_8 = []
    var_9 = {}
    var_10 = module_0.Definitions(*var_8, **var_9)
    var_11 = module_1.from_json_schema_type(var_6, var_3, var_7, var_10)
    var_12 = var_11.min_items
    assert var_12 == 1
    var_13 = var_11.max_items
    assert var_13 == 5
    var_14 = var_11.allow_null
    assert var_14 is False

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'minProperties'
    var_2 = 'maxProperties'
    var_3 = 'object'
    var_4 = 1
    var_5 = 5
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = False
    var_8 = []
    var_9 = {}
    var_10 = module_0.Definitions(*var_8, **var_9)
    var_11 = module_1.from_json_schema_type(var_6, var_3, var_7, var_10)
    var_12 = var_11.min_properties
    assert var_12 == 1
    var_13 = var_11.max_properties
    assert var_13 == 5
    var_14 = var_11.allow_null
    assert var_14 is False

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'default'
    var_2 = 'string'
    var_3 = 'test'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = False
    var_6 = []
    var_7 = {}
    var_8 = module_0.Definitions(*var_6, **var_7)
    var_9 = module_1.from_json_schema_type(var_4, var_2, var_5, var_8)
    var_10 = var_9.default
    assert var_10 == 'test'

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = []
    var_5 = {}
    var_6 = module_0.Definitions(*var_4, **var_5)
    var_7 = module_1.from_json_schema_type(var_2, var_1, var_3, var_6)
    var_8 = var_7.allow_null
    assert var_8 is True

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'items'
    var_2 = 'array'
    var_3 = 'string'
    var_4 = {var_0: var_3}
    var_5 = {var_0: var_2, var_1: var_4}
    var_6 = []
    var_7 = {}
    var_8 = module_0.Definitions(*var_6, **var_7)
    var_9 = False
    var_10 = module_1.from_json_schema_type(var_5, var_2, var_9, var_8)
    var_11 = var_10.items

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'items'
    var_2 = 'additionalItems'
    var_3 = 'array'
    var_4 = 'string'
    var_5 = {var_0: var_4}
    var_6 = 'number'
    var_7 = {var_0: var_6}
    var_8 = {var_0: var_3, var_1: var_5, var_2: var_7}
    var_9 = []
    var_10 = {}
    var_11 = module_0.Definitions(*var_9, **var_10)
    var_12 = False
    var_13 = module_1.from_json_schema_type(var_8, var_3, var_12, var_11)
    var_14 = var_13.additional_items

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'properties'
    var_2 = 'object'
    var_3 = 'name'
    var_4 = 'string'
    var_5 = {var_0: var_4}
    var_6 = {var_3: var_5}
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = []
    var_9 = {}
    var_10 = module_0.Definitions(*var_8, **var_9)
    var_11 = False
    var_12 = module_1.from_json_schema_type(var_7, var_2, var_11, var_10)
    var_13 = var_12.properties[var_3]

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'patternProperties'
    var_2 = 'object'
    var_3 = '^S_'
    var_4 = 'string'
    var_5 = {var_0: var_4}
    var_6 = {var_3: var_5}
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = []
    var_9 = {}
    var_10 = module_0.Definitions(*var_8, **var_9)
    var_11 = False
    var_12 = module_1.from_json_schema_type(var_7, var_2, var_11, var_10)
    var_13 = var_12.pattern_properties[var_3]

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'additionalProperties'
    var_2 = 'object'
    var_3 = 'string'
    var_4 = {var_0: var_3}
    var_5 = {var_0: var_2, var_1: var_4}
    var_6 = []
    var_7 = {}
    var_8 = module_0.Definitions(*var_6, **var_7)
    var_9 = False
    var_10 = module_1.from_json_schema_type(var_5, var_2, var_9, var_8)
    var_11 = var_10.additional_properties

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'propertyNames'
    var_2 = 'object'
    var_3 = 'string'
    var_4 = {var_0: var_3}
    var_5 = {var_0: var_2, var_1: var_4}
    var_6 = []
    var_7 = {}
    var_8 = module_0.Definitions(*var_6, **var_7)
    var_9 = False
    var_10 = module_1.from_json_schema_type(var_5, var_2, var_9, var_8)
    var_11 = var_10.property_names

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'required'
    var_2 = 'object'
    var_3 = 'name'
    var_4 = [var_3]
    var_5 = {var_0: var_2, var_1: var_4}
    var_6 = False
    var_7 = []
    var_8 = {}
    var_9 = module_0.Definitions(*var_7, **var_8)
    var_10 = module_1.from_json_schema_type(var_5, var_2, var_6, var_9)
    var_11 = var_10.required
    var_12 = bool(var_10.required == ['name'])
    assert var_12 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_pattern_regex_unicode_flag. Retrieved 1/5 statements.


def test_case_0():
    var_0 = '^test$'



