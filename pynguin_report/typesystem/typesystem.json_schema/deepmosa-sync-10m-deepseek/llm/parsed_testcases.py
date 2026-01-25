####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 9/14 statements.
# Partially parsed test_to_json_schema_with_union_and_definitions. Retrieved 16/21 statements.
# Partially parsed test_to_json_schema_with_union_and_reference. Retrieved 14/20 statements.
# Partially parsed test_to_json_schema_with_union_and_nested_definitions. Retrieved 16/21 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = module_0.Field()
    var_3 = [var_1, var_2]
    var_4 = {}
    var_5 = module_0.Union(var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = 'anyOf'
    var_8 = 'components'
    var_9 = 'type'
    var_10 = 'null'
    var_11 = [var_10]
    var_12 = {var_9: var_11}
    var_13 = {}
    var_14 = [var_12, var_13]
    var_15 = 'schemas'
    var_16 = {}
    var_17 = {var_15: var_16}
    var_18 = {var_7: var_14, var_8: var_17}
    var_19 = bool(var_6 == var_18)
    assert var_19 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = 'components'
    var_5 = 'schemas'
    var_6 = {}
    var_7 = {var_0: var_6}
    var_8 = {var_5: var_7}
    var_9 = {var_4: var_8}

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Union(var_2, **var_3)
    var_5 = 'UnionField'
    var_6 = {var_5: var_4}
    var_7 = [var_6]
    var_8 = 'components'
    var_9 = 'schemas'
    var_10 = 'anyOf'
    var_11 = {}
    var_12 = {}
    var_13 = [var_11, var_12]
    var_14 = {var_10: var_13}
    var_15 = {var_5: var_14}
    var_16 = {var_9: var_15}
    var_17 = {var_8: var_16}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = False
    var_3 = module_0.Field(allow_null=var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'anyOf'
    var_9 = 'components'
    var_10 = 'type'
    var_11 = 'null'
    var_12 = [var_11]
    var_13 = {var_10: var_12}
    var_14 = {}
    var_15 = [var_13, var_14]
    var_16 = 'schemas'
    var_17 = {}
    var_18 = {var_16: var_17}
    var_19 = {var_8: var_15, var_9: var_18}
    var_20 = bool(var_7 == var_19)
    assert var_20 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'default1'
    var_1 = module_0.Field(default=var_0)
    var_2 = 'default2'
    var_3 = module_0.Field(default=var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'anyOf'
    var_9 = 'components'
    var_10 = 'default'
    var_11 = {var_10: var_0}
    var_12 = {var_10: var_2}
    var_13 = [var_11, var_12]
    var_14 = 'schemas'
    var_15 = {}
    var_16 = {var_14: var_15}
    var_17 = {var_8: var_13, var_9: var_16}
    var_18 = bool(var_7 == var_17)
    assert var_18 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'Target'
    var_2 = {}
    var_3 = 'anyOf'
    var_4 = 'components'
    var_5 = {}
    var_6 = '$ref'
    var_7 = '#/components/schemas/Target'
    var_8 = {var_6: var_7}
    var_9 = [var_5, var_8]
    var_10 = 'schemas'
    var_11 = {}
    var_12 = {var_1: var_11}
    var_13 = {var_10: var_12}
    var_14 = {var_3: var_9, var_4: var_13}

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Union(var_1, **var_2)
    var_4 = 'Inner'
    var_5 = 'Union'
    var_6 = {var_4: var_0, var_5: var_3}
    var_7 = [var_6]
    var_8 = 'components'
    var_9 = 'schemas'
    var_10 = {}
    var_11 = 'anyOf'
    var_12 = {}
    var_13 = [var_12]
    var_14 = {var_11: var_13}
    var_15 = {var_4: var_10, var_5: var_14}
    var_16 = {var_9: var_15}
    var_17 = {var_8: var_16}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = {}
    var_3 = module_0.String(max_length=var_1, min_length=var_0, **var_2)
    var_4 = 0
    var_5 = 100
    var_6 = {}
    var_7 = module_0.Integer(minimum=var_4, maximum=var_5, **var_6)
    var_8 = [var_3, var_7]
    var_9 = {}
    var_10 = module_0.Union(var_8, **var_9)
    var_11 = module_1.to_json_schema(var_10)
    var_12 = 'anyOf'
    var_13 = 'components'
    var_14 = 'type'
    var_15 = 'minLength'
    var_16 = 'maxLength'
    var_17 = 'string'
    var_18 = {var_14: var_17, var_15: var_0, var_16: var_1}
    var_19 = 'minimum'
    var_20 = 'maximum'
    var_21 = 'integer'
    var_22 = {var_14: var_21, var_19: var_4, var_20: var_5}
    var_23 = [var_18, var_22]
    var_24 = 'schemas'
    var_25 = {}
    var_26 = {var_24: var_25}
    var_27 = {var_12: var_23, var_13: var_26}
    var_28 = bool(var_11 == var_27)
    assert var_28 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_0.Field()
    var_5 = [var_3, var_4]
    var_6 = {}
    var_7 = module_0.Union(var_5, **var_6)
    var_8 = module_1.to_json_schema(var_7)
    var_9 = 'anyOf'
    var_10 = 'components'
    var_11 = 'type'
    var_12 = 'string'
    var_13 = 'null'
    var_14 = [var_12, var_13]
    var_15 = {var_11: var_14}
    var_16 = {}
    var_17 = [var_15, var_16]
    var_18 = 'schemas'
    var_19 = {}
    var_20 = {var_18: var_19}
    var_21 = {var_9: var_17, var_10: var_20}
    var_22 = bool(var_8 == var_21)
    assert var_22 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = module_0.Field(allow_null=var_0)
    var_3 = [var_1, var_2]
    var_4 = {}
    var_5 = module_0.Union(var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = 'anyOf'
    var_8 = 'components'
    var_9 = 'type'
    var_10 = 'null'
    var_11 = [var_10]
    var_12 = {var_9: var_11}
    var_13 = [var_10]
    var_14 = {var_9: var_13}
    var_15 = [var_12, var_14]
    var_16 = 'schemas'
    var_17 = {}
    var_18 = {var_16: var_17}
    var_19 = {var_7: var_15, var_8: var_18}
    var_20 = bool(var_6 == var_19)
    assert var_20 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Field(allow_null=var_0)
    var_2 = module_0.Field(allow_null=var_0)
    var_3 = [var_1, var_2]
    var_4 = {}
    var_5 = module_0.Union(var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = 'anyOf'
    var_8 = 'components'
    var_9 = {}
    var_10 = {}
    var_11 = [var_9, var_10]
    var_12 = 'schemas'
    var_13 = {}
    var_14 = {var_12: var_13}
    var_15 = {var_7: var_11, var_8: var_14}
    var_16 = bool(var_6 == var_15)
    assert var_16 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 15/20 statements.
# Partially parsed test_to_json_schema_with_reference_field. Retrieved 12/16 statements.
# Failed to parse test_to_json_schema_with_unknown_field_type.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = var_1 | var_3
    var_5 = module_1.to_json_schema(var_4)
    var_6 = 'anyOf'
    var_7 = 'type'
    var_8 = 'string'
    var_9 = {var_7: var_8}
    var_10 = 'integer'
    var_11 = {var_7: var_10}
    var_12 = [var_9, var_11]
    var_13 = {var_6: var_12}
    var_14 = bool(var_5 == var_13)
    assert var_14 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = var_3 | var_5
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'anyOf'
    var_9 = 'type'
    var_10 = 'string'
    var_11 = 'null'
    var_12 = [var_10, var_11]
    var_13 = {var_9: var_12}
    var_14 = 'integer'
    var_15 = {var_9: var_14}
    var_16 = [var_13, var_15]
    var_17 = {var_8: var_16}
    var_18 = bool(var_7 == var_17)
    assert var_18 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'default'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = var_3 | var_5
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'anyOf'
    var_9 = 'type'
    var_10 = 'string'
    var_11 = {var_9: var_10, var_0: var_0}
    var_12 = 'integer'
    var_13 = {var_9: var_12}
    var_14 = [var_11, var_13]
    var_15 = {var_8: var_14}
    var_16 = bool(var_7 == var_15)
    assert var_16 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'StringField'
    var_1 = 'IntegerField'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = [var_6]
    var_8 = 'components'
    var_9 = 'schemas'
    var_10 = 'type'
    var_11 = 'string'
    var_12 = {var_10: var_11}
    var_13 = 'integer'
    var_14 = {var_10: var_13}
    var_15 = {var_0: var_12, var_1: var_14}
    var_16 = {var_9: var_15}
    var_17 = {var_8: var_16}

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'TargetSchema'
    var_3 = 'target'
    var_4 = {var_3: var_1}
    var_5 = '$ref'
    var_6 = 'components'
    var_7 = '#/components/schemas/TargetSchema'
    var_8 = 'schemas'
    var_9 = 'type'
    var_10 = 'string'
    var_11 = {var_9: var_10}
    var_12 = {var_2: var_11}
    var_13 = {var_8: var_12}
    var_14 = {var_5: var_7, var_6: var_13}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = '^[a-z]+$'
    var_3 = 'email'
    var_4 = 'pattern_regex'
    var_5 = {var_4: var_2}
    var_6 = module_0.String(max_length=var_1, min_length=var_0, format=var_3, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'type'
    var_9 = 'minLength'
    var_10 = 'maxLength'
    var_11 = 'pattern'
    var_12 = 'format'
    var_13 = 'string'
    var_14 = {var_8: var_13, var_9: var_0, var_10: var_1, var_11: var_2, var_12: var_3}
    var_15 = bool(var_7 == var_14)
    assert var_15 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = 100
    var_2 = 5
    var_3 = {}
    var_4 = module_0.Integer(minimum=var_0, maximum=var_1, exclusive_minimum=var_0, exclusive_maximum=var_1, multiple_of=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = 'type'
    var_7 = 'minimum'
    var_8 = 'maximum'
    var_9 = 'exclusiveMinimum'
    var_10 = 'exclusiveMaximum'
    var_11 = 'multipleOf'
    var_12 = 'integer'
    var_13 = {var_6: var_12, var_7: var_0, var_8: var_1, var_9: var_0, var_10: var_1, var_11: var_2}
    var_14 = bool(var_5 == var_13)
    assert var_14 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'type'
    var_6 = 'default'
    var_7 = 'boolean'
    var_8 = {var_5: var_7, var_6: var_0}
    var_9 = bool(var_4 == var_8)
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 1
    var_3 = 10
    var_4 = True
    var_5 = {}
    var_6 = module_0.Array(var_1, min_items=var_2, max_items=var_3, unique_items=var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'type'
    var_9 = 'items'
    var_10 = 'minItems'
    var_11 = 'maxItems'
    var_12 = 'uniqueItems'
    var_13 = 'array'
    var_14 = 'string'
    var_15 = {var_8: var_14}
    var_16 = True
    var_17 = {var_8: var_13, var_9: var_15, var_10: var_4, var_11: var_3, var_12: var_16}
    var_18 = bool(var_7 == var_17)
    assert var_18 is True

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
    var_8 = 1
    var_9 = 2
    var_10 = {}
    var_11 = module_0.Object(properties=var_6, min_properties=var_8, max_properties=var_9, required=var_7, **var_10)
    var_12 = module_1.to_json_schema(var_11)
    var_13 = 'type'
    var_14 = 'properties'
    var_15 = 'required'
    var_16 = 'minProperties'
    var_17 = 'maxProperties'
    var_18 = 'object'
    var_19 = 'string'
    var_20 = {var_13: var_19}
    var_21 = 'integer'
    var_22 = {var_13: var_21}
    var_23 = {var_0: var_20, var_1: var_22}
    var_24 = [var_0]
    var_25 = {var_13: var_18, var_14: var_23, var_15: var_24, var_16: var_8, var_17: var_9}
    var_26 = bool(var_12 == var_25)
    assert var_26 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'option1'
    var_1 = 'Option 1'
    var_2 = (var_0, var_1)
    var_3 = 'option2'
    var_4 = 'Option 2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = 'enum'
    var_11 = [var_0, var_3]
    var_12 = {var_10: var_11}
    var_13 = bool(var_9 == var_12)
    assert var_13 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'constant_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'const'
    var_5 = {var_4: var_0}
    var_6 = bool(var_3 == var_5)
    assert var_6 is True

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
    var_8 = 'oneOf'
    var_9 = 'type'
    var_10 = 'string'
    var_11 = {var_9: var_10}
    var_12 = 'integer'
    var_13 = {var_9: var_12}
    var_14 = [var_11, var_13]
    var_15 = {var_8: var_14}
    var_16 = bool(var_7 == var_15)
    assert var_16 is True

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
    var_8 = 'allOf'
    var_9 = 'type'
    var_10 = 'string'
    var_11 = {var_9: var_10}
    var_12 = 'integer'
    var_13 = {var_9: var_12}
    var_14 = [var_11, var_13]
    var_15 = {var_8: var_14}
    var_16 = bool(var_7 == var_15)
    assert var_16 is True

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
    var_7 = 'if'
    var_8 = 'then'
    var_9 = 'type'
    var_10 = 'string'
    var_11 = {var_9: var_10}
    var_12 = 'integer'
    var_13 = {var_9: var_12}
    var_14 = {var_7: var_11, var_8: var_13}
    var_15 = bool(var_6 == var_14)
    assert var_15 is True

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
    var_6 = 'type'
    var_7 = 'string'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = bool(var_4 == var_9)
    assert var_10 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_from_json_schema_with_boolean_true. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_with_boolean_false. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_with_ref. Retrieved 4/8 statements.
# Partially parsed test_from_json_schema_with_type_string. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_type_array. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_with_enum. Retrieved 7/8 statements.
# Partially parsed test_from_json_schema_with_const. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_allOf. Retrieved 10/11 statements.
# Partially parsed test_from_json_schema_with_anyOf. Retrieved 9/10 statements.
# Partially parsed test_from_json_schema_with_oneOf. Retrieved 9/10 statements.
# Partially parsed test_from_json_schema_with_not. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_with_if_then. Retrieved 10/11 statements.
# Partially parsed test_from_json_schema_with_if_then_else. Retrieved 13/14 statements.
# Partially parsed test_from_json_schema_with_multiple_constraints. Retrieved 10/11 statements.
# Partially parsed test_from_json_schema_with_no_constraints. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_with_null_type. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_null_in_union. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_with_components. Retrieved 11/14 statements.
# Partially parsed test_from_json_schema_with_number_type_includes_integer. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_integer_type_excludes_number. Retrieved 4/5 statements.


import typesystem.json_schema as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.from_json_schema(var_0)

def test_case_0():
    var_0 = []
    var_1 = 'test'
    var_2 = {}
    var_3 = '$ref'
    var_4 = '#/components/schemas/Test'
    var_5 = {var_3: var_4}

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = 'number'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.from_json_schema(var_4)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'enum'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = module_0.from_json_schema(var_5)

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
    var_4 = 'maxLength'
    var_5 = 10
    var_6 = {var_4: var_5}
    var_7 = [var_3, var_6]
    var_8 = {var_0: var_7}
    var_9 = module_0.from_json_schema(var_8)

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
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = 'maxLength'
    var_6 = 10
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = module_0.from_json_schema(var_8)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'if'
    var_1 = 'then'
    var_2 = 'else'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 'maxLength'
    var_7 = 10
    var_8 = {var_6: var_7}
    var_9 = 'number'
    var_10 = {var_3: var_9}
    var_11 = {var_0: var_5, var_1: var_8, var_2: var_10}
    var_12 = module_0.from_json_schema(var_11)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'enum'
    var_2 = 'maxLength'
    var_3 = 'string'
    var_4 = 'a'
    var_5 = 'b'
    var_6 = [var_4, var_5]
    var_7 = 5
    var_8 = {var_0: var_3, var_1: var_6, var_2: var_7}
    var_9 = module_0.from_json_schema(var_8)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.from_json_schema(var_0)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'null'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)
    var_4 = var_3.const
    assert var_4 is None

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'null'
    var_2 = 'string'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.from_json_schema(var_4)
    var_6 = var_5.allow_null
    assert var_6 is True

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'components'
    var_1 = 'schemas'
    var_2 = 'Test'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = module_0.from_json_schema(var_8)
    var_10 = []
    var_11 = {var_3: var_4}

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'default'
    var_2 = 'string'
    var_3 = 'hello'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.from_json_schema(var_4)
    var_6 = var_5.default
    assert var_6 == 'hello'

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'number'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'integer'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)



# Parsed testcases at query #4
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['additionalProperties']
    assert var_4 is False



# Parsed testcases at query #5
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = None
    var_3 = {}
    var_4 = module_0.String(**var_3)
    var_5 = {}
    var_6 = module_1.IfThenElse(var_1, var_2, var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = 'then'
    var_9 = bool('then' not in var_7)
    assert var_9 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_from_json_schema_type_number. Retrieved 13/16 statements.
# Partially parsed test_from_json_schema_type_integer. Retrieved 13/16 statements.
# Partially parsed test_from_json_schema_type_string. Retrieved 13/16 statements.
# Partially parsed test_from_json_schema_type_boolean. Retrieved 4/7 statements.
# Partially parsed test_from_json_schema_type_array. Retrieved 13/16 statements.
# Partially parsed test_from_json_schema_type_object. Retrieved 11/14 statements.


def test_case_0():
    var_0 = 'minimum'
    var_1 = 'maximum'
    var_2 = 'exclusiveMinimum'
    var_3 = 'exclusiveMaximum'
    var_4 = 'multipleOf'
    var_5 = 'default'
    var_6 = 0
    var_7 = 10
    var_8 = 2
    var_9 = 5.0
    var_10 = {var_0: var_6, var_1: var_7, var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_9}
    var_11 = []
    var_12 = 'number'
    var_13 = False

def test_case_0():
    var_0 = 'minimum'
    var_1 = 'maximum'
    var_2 = 'exclusiveMinimum'
    var_3 = 'exclusiveMaximum'
    var_4 = 'multipleOf'
    var_5 = 'default'
    var_6 = 0
    var_7 = 10
    var_8 = 2
    var_9 = 5
    var_10 = {var_0: var_6, var_1: var_7, var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_9}
    var_11 = []
    var_12 = 'integer'
    var_13 = True

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
    var_12 = 'string'
    var_13 = False

def test_case_0():
    var_0 = 'default'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'boolean'

def test_case_0():
    var_0 = 'minItems'
    var_1 = 'maxItems'
    var_2 = 'uniqueItems'
    var_3 = 'default'
    var_4 = 1
    var_5 = 10
    var_6 = True
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]
    var_10 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_9}
    var_11 = []
    var_12 = 'array'
    var_13 = False

def test_case_0():
    var_0 = 'minProperties'
    var_1 = 'maxProperties'
    var_2 = 'default'
    var_3 = 1
    var_4 = 10
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_3, var_1: var_4, var_2: var_7}
    var_9 = []
    var_10 = 'object'
    var_11 = True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_if_then_else_from_json_schema_basic. Retrieved 15/20 statements.
# Partially parsed test_if_then_else_from_json_schema_without_then. Retrieved 10/14 statements.
# Partially parsed test_if_then_else_from_json_schema_without_else. Retrieved 11/15 statements.
# Partially parsed test_if_then_else_from_json_schema_with_nested_schemas. Retrieved 18/22 statements.
# Partially parsed test_if_then_else_from_json_schema_with_const. Retrieved 13/17 statements.
# Partially parsed test_if_then_else_from_json_schema_with_ref_in_if. Retrieved 15/20 statements.
# Partially parsed test_if_then_else_from_json_schema_with_all_of_in_then. Retrieved 20/24 statements.
# Partially parsed test_if_then_else_from_json_schema_with_any_of_in_else. Retrieved 18/23 statements.
# Partially parsed test_if_then_else_from_json_schema_with_one_of_in_if. Retrieved 18/23 statements.
# Partially parsed test_if_then_else_from_json_schema_with_not_in_then. Retrieved 16/20 statements.
# Partially parsed test_if_then_else_from_json_schema_default_handling. Retrieved 12/16 statements.
# Partially parsed test_if_then_else_from_json_schema_complex_nesting. Retrieved 20/24 statements.
# Partially parsed test_if_then_else_from_json_schema_empty_then_and_else. Retrieved 7/11 statements.
# Partially parsed test_if_then_else_from_json_schema_with_enum_in_if. Retrieved 15/20 statements.


def test_case_0():
    var_0 = []
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'else'
    var_4 = 'type'
    var_5 = 'string'
    var_6 = {var_4: var_5}
    var_7 = 'minLength'
    var_8 = 5
    var_9 = {var_4: var_5, var_7: var_8}
    var_10 = 'integer'
    var_11 = {var_4: var_10}
    var_12 = {var_1: var_6, var_2: var_9, var_3: var_11}
    var_13 = 'hello'
    var_14 = 'hi'
    var_15 = 123

def test_case_0():
    var_0 = []
    var_1 = 'if'
    var_2 = 'else'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 'integer'
    var_7 = {var_3: var_6}
    var_8 = {var_1: var_5, var_2: var_7}
    var_9 = 'test'
    var_10 = 42

def test_case_0():
    var_0 = []
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 'maxLength'
    var_7 = 3
    var_8 = {var_3: var_4, var_6: var_7}
    var_9 = {var_1: var_5, var_2: var_8}
    var_10 = 'abc'
    var_11 = 100

def test_case_0():
    var_0 = []
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'else'
    var_4 = 'type'
    var_5 = 'array'
    var_6 = {var_4: var_5}
    var_7 = 'items'
    var_8 = 'number'
    var_9 = {var_4: var_8}
    var_10 = {var_7: var_9}
    var_11 = 'boolean'
    var_12 = {var_4: var_11}
    var_13 = {var_1: var_6, var_2: var_10, var_3: var_12}
    var_14 = 1
    var_15 = 2
    var_16 = 3
    var_17 = [var_14, var_15, var_16]
    var_18 = True

def test_case_0():
    var_0 = []
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'else'
    var_4 = 'const'
    var_5 = 'special'
    var_6 = {var_4: var_5}
    var_7 = 'type'
    var_8 = 'string'
    var_9 = {var_7: var_8}
    var_10 = 'number'
    var_11 = {var_7: var_10}
    var_12 = {var_1: var_6, var_2: var_9, var_3: var_11}
    var_13 = 99

def test_case_0():
    var_0 = []
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'if'
    var_5 = 'then'
    var_6 = 'else'
    var_7 = '$ref'
    var_8 = '#/components/schemas/MyString'
    var_9 = {var_7: var_8}
    var_10 = {var_1: var_2}
    var_11 = 'integer'
    var_12 = {var_1: var_11}
    var_13 = {var_4: var_9, var_5: var_10, var_6: var_12}
    var_14 = 'text'
    var_15 = 456

def test_case_0():
    var_0 = []
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'else'
    var_4 = 'type'
    var_5 = 'object'
    var_6 = {var_4: var_5}
    var_7 = 'allOf'
    var_8 = {var_4: var_5}
    var_9 = 'required'
    var_10 = 'id'
    var_11 = [var_10]
    var_12 = {var_9: var_11}
    var_13 = [var_8, var_12]
    var_14 = {var_7: var_13}
    var_15 = 'null'
    var_16 = {var_4: var_15}
    var_17 = {var_1: var_6, var_2: var_14, var_3: var_16}
    var_18 = 1
    var_19 = {var_10: var_18}
    var_20 = None

def test_case_0():
    var_0 = []
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'else'
    var_4 = 'type'
    var_5 = 'number'
    var_6 = {var_4: var_5}
    var_7 = {var_4: var_5}
    var_8 = 'anyOf'
    var_9 = 'string'
    var_10 = {var_4: var_9}
    var_11 = 'boolean'
    var_12 = {var_4: var_11}
    var_13 = [var_10, var_12]
    var_14 = {var_8: var_13}
    var_15 = {var_1: var_6, var_2: var_7, var_3: var_14}
    var_16 = 3.14
    var_17 = 'text'
    var_18 = False

def test_case_0():
    var_0 = []
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'else'
    var_4 = 'oneOf'
    var_5 = 'type'
    var_6 = 'string'
    var_7 = {var_5: var_6}
    var_8 = 'number'
    var_9 = {var_5: var_8}
    var_10 = [var_7, var_9]
    var_11 = {var_4: var_10}
    var_12 = {var_5: var_6}
    var_13 = 'boolean'
    var_14 = {var_5: var_13}
    var_15 = {var_1: var_11, var_2: var_12, var_3: var_14}
    var_16 = 'test'
    var_17 = 42
    var_18 = True

def test_case_0():
    var_0 = []
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'else'
    var_4 = 'type'
    var_5 = 'array'
    var_6 = {var_4: var_5}
    var_7 = 'not'
    var_8 = 'string'
    var_9 = {var_4: var_8}
    var_10 = {var_7: var_9}
    var_11 = {var_4: var_8}
    var_12 = {var_1: var_6, var_2: var_10, var_3: var_11}
    var_13 = 1
    var_14 = 2
    var_15 = [var_13, var_14]
    var_16 = 'allowed'

def test_case_0():
    var_0 = []
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'else'
    var_4 = 'default'
    var_5 = 'type'
    var_6 = 'string'
    var_7 = {var_5: var_6}
    var_8 = {var_5: var_6}
    var_9 = 'integer'
    var_10 = {var_5: var_9}
    var_11 = 'default_value'
    var_12 = {var_1: var_7, var_2: var_8, var_3: var_10, var_4: var_11}

def test_case_0():
    var_0 = []
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'else'
    var_4 = 'type'
    var_5 = 'properties'
    var_6 = 'object'
    var_7 = 'active'
    var_8 = 'boolean'
    var_9 = {var_4: var_8}
    var_10 = {var_7: var_9}
    var_11 = {var_4: var_6, var_5: var_10}
    var_12 = 'required'
    var_13 = [var_7]
    var_14 = {var_4: var_6, var_12: var_13}
    var_15 = 'null'
    var_16 = {var_4: var_15}
    var_17 = {var_1: var_11, var_2: var_14, var_3: var_16}
    var_18 = True
    var_19 = {var_7: var_18}
    var_20 = None

def test_case_0():
    var_0 = []
    var_1 = 'if'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'anything'
    var_7 = 123

def test_case_0():
    var_0 = []
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'else'
    var_4 = 'enum'
    var_5 = 'yes'
    var_6 = 'no'
    var_7 = [var_5, var_6]
    var_8 = {var_4: var_7}
    var_9 = 'type'
    var_10 = 'string'
    var_11 = {var_9: var_10}
    var_12 = 'integer'
    var_13 = {var_9: var_12}
    var_14 = {var_1: var_8, var_2: var_11, var_3: var_13}
    var_15 = 0



# Parsed testcases at query #8
#--------------------------






# Parsed testcases at query #9
#--------------------------

# Partially parsed test_property_names_is_none. Retrieved 3/5 statements.


def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'object'
    var_3 = False



# Parsed testcases at query #10
#--------------------------






# Parsed testcases at query #11
#--------------------------

# Partially parsed test_additional_items_is_not_bool. Retrieved 5/6 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(additional_items=var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'additionalItems'
    var_6 = var_4[var_5]



# Parsed testcases at query #12
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = None
    var_5 = {}
    var_6 = module_1.IfThenElse(var_1, var_3, var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = 'else'
    var_9 = bool('else' not in var_7)
    assert var_9 is True



# Parsed testcases at query #13
#--------------------------




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
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    assert var_5 == 'boolean'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_additional_properties_none_returns_none. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'additionalProperties'
    var_1 = None
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'object'
    var_5 = False



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_to_json_schema_with_definitions_loops_through_items. Retrieved 6/8 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = {}
    var_4 = module_0.String(**var_3)
    var_5 = {}
    var_6 = module_0.Integer(**var_5)
    var_7 = {var_1: var_4, var_2: var_6}
    var_8 = [var_7]
    var_9 = 'key1'
    var_10 = bool('key1' in var_0)
    assert var_10 is True
    var_11 = 'key2'
    var_12 = bool('key2' in var_0)
    assert var_12 is True
    var_13 = var_0['key1']['type']
    assert var_13 == 'string'
    var_14 = var_0['key2']['type']
    assert var_14 == 'integer'



# Parsed testcases at query #16
#--------------------------






# Parsed testcases at query #17
#--------------------------

# Partially parsed test_from_json_schema_object_with_properties. Retrieved 10/13 statements.


import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'properties'
    var_2 = 'object'
    var_3 = 'name'
    var_4 = 'string'
    var_5 = {var_0: var_4}
    var_6 = {var_3: var_5}
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = {}
    var_9 = module_0.from_json_schema(var_7, var_8)
    var_10 = var_9.properties
    var_11 = bool(var_9.properties is not None)
    assert var_11 is True
    var_12 = 'name'
    var_13 = bool('name' in var_9.properties)
    assert var_13 is True



# Parsed testcases at query #18
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['object', 'null'])
    assert var_6 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_predicate_at_line_71_evaluates_to_true. Retrieved 9/13 statements.


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
    var_9 = {}
    var_10 = module_0.String(**var_9)
    var_11 = {}
    var_12 = module_0.Array(var_10, **var_11)
    var_13 = module_1.to_json_schema(var_12)



# Parsed testcases at query #20
#--------------------------






# Parsed testcases at query #21
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['array', 'null'])
    assert var_6 is True



# Parsed testcases at query #22
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    assert var_5 == 'integer'

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
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Float(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['number', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Decimal(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['number', 'null'])
    assert var_6 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_predicate_at_line_120_evaluates_to_true. Retrieved 2/7 statements.


def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = False
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_to_json_schema_with_definitions_and_union. Retrieved 18/23 statements.
# Partially parsed test_to_json_schema_with_reference_in_union. Retrieved 14/21 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = var_1 | var_3
    var_5 = module_1.to_json_schema(var_4)
    var_6 = 'anyOf'
    var_7 = 'type'
    var_8 = 'string'
    var_9 = {var_7: var_8}
    var_10 = 'integer'
    var_11 = {var_7: var_10}
    var_12 = [var_9, var_11]
    var_13 = {var_6: var_12}
    var_14 = bool(var_5 == var_13)
    assert var_14 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = var_3 | var_5
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'anyOf'
    var_9 = 'type'
    var_10 = 'string'
    var_11 = 'null'
    var_12 = [var_10, var_11]
    var_13 = {var_9: var_12}
    var_14 = 'integer'
    var_15 = {var_9: var_14}
    var_16 = [var_13, var_15]
    var_17 = {var_8: var_16}
    var_18 = bool(var_7 == var_17)
    assert var_18 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = 42
    var_5 = 'default'
    var_6 = {var_5: var_4}
    var_7 = module_0.Integer(**var_6)
    var_8 = var_3 | var_7
    var_9 = module_1.to_json_schema(var_8)
    var_10 = 'anyOf'
    var_11 = 'type'
    var_12 = 'default'
    var_13 = 'string'
    var_14 = {var_11: var_13, var_12: var_0}
    var_15 = 'integer'
    var_16 = {var_11: var_15, var_12: var_4}
    var_17 = [var_14, var_16]
    var_18 = {var_10: var_17}
    var_19 = bool(var_9 == var_18)
    assert var_19 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = var_1 | var_3
    var_5 = {}
    var_6 = module_0.Boolean(**var_5)
    var_7 = var_4 | var_6
    var_8 = module_1.to_json_schema(var_7)
    var_9 = 'anyOf'
    var_10 = 'type'
    var_11 = 'string'
    var_12 = {var_10: var_11}
    var_13 = 'integer'
    var_14 = {var_10: var_13}
    var_15 = 'boolean'
    var_16 = {var_10: var_15}
    var_17 = [var_12, var_14, var_16]
    var_18 = {var_9: var_17}
    var_19 = bool(var_8 == var_18)
    assert var_19 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'MyUnion'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {}
    var_4 = module_0.Integer(**var_3)
    var_5 = var_2 | var_4
    var_6 = {var_0: var_5}
    var_7 = [var_6]
    var_8 = 'components'
    var_9 = 'schemas'
    var_10 = 'anyOf'
    var_11 = 'type'
    var_12 = 'string'
    var_13 = {var_11: var_12}
    var_14 = 'integer'
    var_15 = {var_11: var_14}
    var_16 = [var_13, var_15]
    var_17 = {var_10: var_16}
    var_18 = {var_0: var_17}
    var_19 = {var_9: var_18}
    var_20 = {var_8: var_19}

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'MyString'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.String(**var_5)
    var_7 = 'target'
    var_8 = {var_7: var_6}
    var_9 = {}
    var_10 = module_0.String(**var_9)
    var_11 = 'anyOf'
    var_12 = '$ref'
    var_13 = '#/components/schemas/MyString'
    var_14 = {var_12: var_13}
    var_15 = 'type'
    var_16 = 'string'
    var_17 = {var_15: var_16}
    var_18 = [var_14, var_17]
    var_19 = {var_11: var_18}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = var_3 | var_5
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'anyOf'
    var_9 = 'type'
    var_10 = 'items'
    var_11 = 'array'
    var_12 = 'string'
    var_13 = {var_9: var_12}
    var_14 = {var_9: var_11, var_10: var_13}
    var_15 = 'integer'
    var_16 = {var_9: var_15}
    var_17 = [var_14, var_16]
    var_18 = {var_8: var_17}
    var_19 = bool(var_7 == var_18)
    assert var_19 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = {}
    var_7 = module_0.Integer(**var_6)
    var_8 = var_5 | var_7
    var_9 = module_1.to_json_schema(var_8)
    var_10 = 'anyOf'
    var_11 = 'type'
    var_12 = 'properties'
    var_13 = 'object'
    var_14 = 'string'
    var_15 = {var_11: var_14}
    var_16 = {var_0: var_15}
    var_17 = {var_11: var_13, var_12: var_16}
    var_18 = 'integer'
    var_19 = {var_11: var_18}
    var_20 = [var_17, var_19]
    var_21 = {var_10: var_20}
    var_22 = bool(var_9 == var_21)
    assert var_22 is True

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
    var_9 = {}
    var_10 = module_0.Integer(**var_9)
    var_11 = var_8 | var_10
    var_12 = module_1.to_json_schema(var_11)
    var_13 = 'anyOf'
    var_14 = 'enum'
    var_15 = [var_0, var_3]
    var_16 = {var_14: var_15}
    var_17 = 'type'
    var_18 = 'integer'
    var_19 = {var_17: var_18}
    var_20 = [var_16, var_19]
    var_21 = {var_13: var_20}
    var_22 = bool(var_12 == var_21)
    assert var_22 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Integer(**var_3)
    var_5 = var_2 | var_4
    var_6 = module_1.to_json_schema(var_5)
    var_7 = 'anyOf'
    var_8 = 'const'
    var_9 = {var_8: var_0}
    var_10 = 'type'
    var_11 = 'integer'
    var_12 = {var_10: var_11}
    var_13 = [var_9, var_12]
    var_14 = {var_7: var_13}
    var_15 = bool(var_6 == var_14)
    assert var_15 is True

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 10
    var_3 = {}
    var_4 = module_0.String(max_length=var_2, **var_3)
    var_5 = [var_1, var_4]
    var_6 = {}
    var_7 = module_1.AllOf(var_5, **var_6)
    var_8 = {}
    var_9 = module_0.Integer(**var_8)
    var_10 = var_7 | var_9
    var_11 = module_2.to_json_schema(var_10)
    var_12 = 'anyOf'
    var_13 = 'allOf'
    var_14 = 'type'
    var_15 = 'string'
    var_16 = {var_14: var_15}
    var_17 = 'maxLength'
    var_18 = {var_14: var_15, var_17: var_2}
    var_19 = [var_16, var_18]
    var_20 = {var_13: var_19}
    var_21 = 'integer'
    var_22 = {var_14: var_21}
    var_23 = [var_20, var_22]
    var_24 = {var_12: var_23}
    var_25 = bool(var_11 == var_24)
    assert var_25 is True

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
    var_7 = {}
    var_8 = module_0.String(**var_7)
    var_9 = var_6 | var_8
    var_10 = module_2.to_json_schema(var_9)
    var_11 = 'anyOf'
    var_12 = 'oneOf'
    var_13 = 'type'
    var_14 = 'string'
    var_15 = {var_13: var_14}
    var_16 = 'integer'
    var_17 = {var_13: var_16}
    var_18 = [var_15, var_17]
    var_19 = {var_12: var_18}
    var_20 = {var_13: var_14}
    var_21 = [var_19, var_20]
    var_22 = {var_11: var_21}
    var_23 = bool(var_10 == var_22)
    assert var_23 is True

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
    var_6 = {}
    var_7 = module_0.String(**var_6)
    var_8 = var_5 | var_7
    var_9 = module_2.to_json_schema(var_8)
    var_10 = 'anyOf'
    var_11 = 'if'
    var_12 = 'then'
    var_13 = 'type'
    var_14 = 'string'
    var_15 = {var_13: var_14}
    var_16 = 'integer'
    var_17 = {var_13: var_16}
    var_18 = {var_11: var_15, var_12: var_17}
    var_19 = {var_13: var_14}
    var_20 = [var_18, var_19]
    var_21 = {var_10: var_20}
    var_22 = bool(var_9 == var_21)
    assert var_22 is True

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_1.Not(var_1, **var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = var_3 | var_5
    var_7 = module_2.to_json_schema(var_6)
    var_8 = 'anyOf'
    var_9 = 'not'
    var_10 = 'type'
    var_11 = 'string'
    var_12 = {var_10: var_11}
    var_13 = {var_9: var_12}
    var_14 = 'integer'
    var_15 = {var_10: var_14}
    var_16 = [var_13, var_15]
    var_17 = {var_8: var_16}
    var_18 = bool(var_7 == var_17)
    assert var_18 is True

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
    var_6 = {}
    var_7 = module_0.Integer(**var_6)
    var_8 = var_5 | var_7
    var_9 = module_2.to_json_schema(var_8)
    var_10 = 'anyOf'
    var_11 = 'type'
    var_12 = 'properties'
    var_13 = 'object'
    var_14 = 'string'
    var_15 = {var_11: var_14}
    var_16 = {var_0: var_15}
    var_17 = {var_11: var_13, var_12: var_16}
    var_18 = 'integer'
    var_19 = {var_11: var_18}
    var_20 = [var_17, var_19]
    var_21 = {var_10: var_20}
    var_22 = bool(var_9 == var_21)
    assert var_22 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = var_0 | var_2
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'anyOf'
    var_6 = True
    var_7 = 'type'
    var_8 = 'string'
    var_9 = {var_7: var_8}
    var_10 = [var_6, var_9]
    var_11 = {var_5: var_10}
    var_12 = bool(var_4 == var_11)
    assert var_12 is True

import typesystem.composites as module_0
import typesystem.fields as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = {}
    var_3 = module_1.String(**var_2)
    var_4 = var_1 | var_3
    var_5 = module_2.to_json_schema(var_4)
    var_6 = 'anyOf'
    var_7 = False
    var_8 = 'type'
    var_9 = 'string'
    var_10 = {var_8: var_9}
    var_11 = [var_7, var_10]
    var_12 = {var_6: var_11}
    var_13 = bool(var_5 == var_12)
    assert var_13 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = 'allow_null'
    var_5 = {var_4: var_0}
    var_6 = module_0.Integer(**var_5)
    var_7 = var_3 | var_6
    var_8 = module_1.to_json_schema(var_7)
    var_9 = 'anyOf'
    var_10 = 'type'
    var_11 = 'string'
    var_12 = 'null'
    var_13 = [var_11, var_12]
    var_14 = {var_10: var_13}
    var_15 = 'integer'
    var_16 = [var_15, var_12]
    var_17 = {var_10: var_16}
    var_18 = [var_14, var_17]
    var_19 = {var_9: var_18}
    var_20 = bool(var_8 == var_19)
    assert var_20 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = var_3 | var_5
    var_7 = 'allow_null'
    var_8 = {var_7: var_0}
    var_9 = module_0.Boolean(**var_8)
    var_10 = var_6 | var_9
    var_11 = module_1.to_json_schema(var_10)
    var_12 = 'anyOf'
    var_13 = 'type'
    var_14 = 'string'
    var_15 = 'null'
    var_16 = [var_14, var_15]
    var_17 = {var_13: var_16}
    var_18 = 'integer'
    var_19 = {var_13: var_18}
    var_20 = 'boolean'
    var_21 = [var_20, var_15]
    var_22 = {var_13: var_21}
    var_23 = [var_17, var_19, var_22]
    var_24 = {var_12: var_23}
    var_25 = bool(var_11 == var_24)
    assert var_25 is True



# Parsed testcases at query #25
#--------------------------




import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False



# Parsed testcases at query #26
#--------------------------






# Parsed testcases at query #27
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['format']
    assert var_4 == 'email'



# Parsed testcases at query #28
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = None
    var_3 = {}
    var_4 = module_0.String(**var_3)
    var_5 = {}
    var_6 = module_1.IfThenElse(var_1, var_2, var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = 'then'
    var_9 = bool('then' not in var_7)
    assert var_9 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_to_json_schema_root_with_definitions. Retrieved 2/4 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'TestRef'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = 'target'
    var_4 = {var_3: var_2}
    var_5 = 'components'
    var_6 = 'schemas'
    var_7 = 'TestRef'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_to_json_schema_with_union_field_and_definitions. Retrieved 5/15 statements.
# Partially parsed test_to_json_schema_with_union_field_root_definitions. Retrieved 4/14 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
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
    assert var_13 == 'integer'
    var_14 = var_7['anyOf'][1]['type']
    assert var_14 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = {}
    var_5 = module_0.String(**var_4)
    var_6 = [var_3, var_5]
    var_7 = {}
    var_8 = module_0.Union(var_6, **var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = 'anyOf'
    var_11 = bool('anyOf' in var_9)
    assert var_11 is True
    var_12 = 'anyOf'
    var_13 = var_9[var_12]
    var_14 = len(var_13)
    assert var_14 == 2
    var_15 = var_9['anyOf'][0]['type']
    var_16 = bool(var_9['anyOf'][0]['type'] == ['integer', 'null'])
    assert var_16 is True
    var_17 = var_9['anyOf'][1]['type']
    assert var_17 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 42
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = {}
    var_5 = module_0.String(**var_4)
    var_6 = [var_3, var_5]
    var_7 = {}
    var_8 = module_0.Union(var_6, **var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = 'anyOf'
    var_11 = bool('anyOf' in var_9)
    assert var_11 is True
    var_12 = 'default'
    var_13 = bool('default' in var_9)
    assert var_13 is True
    var_14 = var_9['default']
    assert var_14 == 42

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'MyRef'
    var_1 = {}
    var_2 = module_0.Integer(**var_1)
    var_3 = 'target'
    var_4 = {var_3: var_2}
    var_5 = {}
    var_6 = module_0.Integer(**var_5)
    var_7 = 'target'
    var_8 = {var_7: var_6}
    var_9 = {}
    var_10 = module_0.String(**var_9)
    var_11 = {}
    var_12 = 'anyOf'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'MyRef'
    var_1 = {}
    var_2 = module_0.Integer(**var_1)
    var_3 = 'target'
    var_4 = {var_3: var_2}
    var_5 = {}
    var_6 = module_0.Integer(**var_5)
    var_7 = 'target'
    var_8 = {var_7: var_6}
    var_9 = {}
    var_10 = module_0.String(**var_9)
    var_11 = 'anyOf'
    var_12 = 'components'
    var_13 = 'schemas'
    var_14 = 'MyRef'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Boolean(**var_4)
    var_6 = [var_1, var_3, var_5]
    var_7 = {}
    var_8 = module_0.Union(var_6, **var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = 'anyOf'
    var_11 = bool('anyOf' in var_9)
    assert var_11 is True
    var_12 = 'anyOf'
    var_13 = var_9[var_12]
    var_14 = len(var_13)
    assert var_14 == 3
    var_15 = var_9['anyOf'][0]['type']
    assert var_15 == 'integer'
    var_16 = var_9['anyOf'][1]['type']
    assert var_16 == 'string'
    var_17 = var_9['anyOf'][2]['type']
    assert var_17 == 'boolean'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = {}
    var_8 = module_0.String(**var_7)
    var_9 = [var_6, var_8]
    var_10 = {}
    var_11 = module_0.Union(var_9, **var_10)
    var_12 = module_1.to_json_schema(var_11)
    var_13 = 'anyOf'
    var_14 = bool('anyOf' in var_12)
    assert var_14 is True
    var_15 = 'anyOf'
    var_16 = var_12[var_15]
    var_17 = len(var_16)
    assert var_17 == 2
    var_18 = 'anyOf'
    var_19 = bool('anyOf' in var_12['anyOf'][0])
    assert var_19 is True
    var_20 = var_12['anyOf'][1]['type']
    assert var_20 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = 'allow_null'
    var_5 = {var_4: var_0}
    var_6 = module_0.String(**var_5)
    var_7 = [var_3, var_6]
    var_8 = {}
    var_9 = module_0.Union(var_7, **var_8)
    var_10 = module_1.to_json_schema(var_9)
    var_11 = 'anyOf'
    var_12 = bool('anyOf' in var_10)
    assert var_12 is True
    var_13 = 'anyOf'
    var_14 = var_10[var_13]
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = var_10['anyOf'][0]['type']
    var_17 = bool(var_10['anyOf'][0]['type'] == ['integer', 'null'])
    assert var_17 is True
    var_18 = var_10['anyOf'][1]['type']
    var_19 = bool(var_10['anyOf'][1]['type'] == ['string', 'null'])
    assert var_19 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Union(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'anyOf'
    var_5 = bool('anyOf' in var_3)
    assert var_5 is True
    var_6 = var_3['anyOf']
    var_7 = bool(var_3['anyOf'] == [])
    assert var_7 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = {}
    var_5 = module_0.String(**var_4)
    var_6 = [var_3, var_5]
    var_7 = {}
    var_8 = module_0.Union(var_6, **var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = 'anyOf'
    var_11 = bool('anyOf' in var_9)
    assert var_11 is True
    var_12 = 'anyOf'
    var_13 = var_9[var_12]
    var_14 = len(var_13)
    assert var_14 == 2
    var_15 = var_9['anyOf'][0]['type']
    assert var_15 == 'array'
    var_16 = 'items'
    var_17 = bool('items' in var_9['anyOf'][0])
    assert var_17 is True
    var_18 = var_9['anyOf'][0]['items']['type']
    assert var_18 == 'integer'
    var_19 = var_9['anyOf'][1]['type']
    assert var_19 == 'string'



# Parsed testcases at query #31
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = '^test.*'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(pattern_properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = 'patternProperties'
    var_8 = bool('patternProperties' in var_6)
    assert var_8 is True
    var_9 = '^test.*'
    var_10 = bool('^test.*' in var_6['patternProperties'])
    assert var_10 is True
    var_11 = var_6['patternProperties']['^test.*']['type']
    assert var_11 == 'string'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_from_json_schema_object_without_pattern_properties. Retrieved 6/9 statements.


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



# Parsed testcases at query #33
#--------------------------






####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 12/22 statements.
# Partially parsed test_to_json_schema_with_unknown_field_type. Retrieved 1/9 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.String(**var_4)
    var_6 = {}
    var_7 = module_0.Integer(**var_6)
    var_8 = [var_5, var_7]
    var_9 = {}
    var_10 = module_0.Union(var_8, **var_9)
    var_11 = module_1.to_json_schema(var_10)
    var_12 = 'anyOf'
    var_13 = 'type'
    var_14 = 'string'
    var_15 = 'null'
    var_16 = [var_14, var_15]
    var_17 = {var_13: var_16}
    var_18 = 'integer'
    var_19 = {var_13: var_18}
    var_20 = [var_17, var_19]
    var_21 = {var_12: var_20}
    var_22 = bool(var_11 == var_21)
    assert var_22 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'MyString'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = [var_4]
    var_6 = 'components'
    var_7 = 'schemas'
    var_8 = 'type'
    var_9 = 'string'
    var_10 = {var_8: var_9}
    var_11 = {var_1: var_10}
    var_12 = {var_7: var_11}
    var_13 = {var_6: var_12}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = 'hello'
    var_2 = 'default'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(**var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = 'type'
    var_7 = 'default'
    var_8 = 'string'
    var_9 = {var_6: var_8, var_7: var_1}
    var_10 = bool(var_5 == var_9)
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(**var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = 'type'
    var_7 = 'string'
    var_8 = 'null'
    var_9 = [var_7, var_8]
    var_10 = {var_6: var_9}
    var_11 = bool(var_5 == var_10)
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = 0
    var_2 = 100
    var_3 = 5
    var_4 = {}
    var_5 = module_0.Integer(minimum=var_1, maximum=var_2, exclusive_minimum=var_1, exclusive_maximum=var_2, multiple_of=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = 'type'
    var_8 = 'minimum'
    var_9 = 'maximum'
    var_10 = 'exclusiveMinimum'
    var_11 = 'exclusiveMaximum'
    var_12 = 'multipleOf'
    var_13 = 'integer'
    var_14 = {var_7: var_13, var_8: var_1, var_9: var_2, var_10: var_1, var_11: var_2, var_12: var_3}
    var_15 = bool(var_6 == var_14)
    assert var_15 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = 1
    var_5 = 10
    var_6 = True
    var_7 = {}
    var_8 = module_0.Array(var_3, min_items=var_4, max_items=var_5, unique_items=var_6, **var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = 'type'
    var_11 = 'items'
    var_12 = 'minItems'
    var_13 = 'maxItems'
    var_14 = 'uniqueItems'
    var_15 = 'array'
    var_16 = 'string'
    var_17 = {var_10: var_16}
    var_18 = True
    var_19 = {var_10: var_15, var_11: var_17, var_12: var_6, var_13: var_5, var_14: var_18}
    var_20 = bool(var_9 == var_19)
    assert var_20 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = {}
    var_3 = 'name'
    var_4 = 'age'
    var_5 = {}
    var_6 = module_0.String(**var_5)
    var_7 = {}
    var_8 = module_0.Integer(**var_7)
    var_9 = {var_3: var_6, var_4: var_8}
    var_10 = [var_3]
    var_11 = {}
    var_12 = module_0.Object(properties=var_9, required=var_10, **var_11)
    var_13 = module_1.to_json_schema(var_12)
    var_14 = 'type'
    var_15 = 'properties'
    var_16 = 'required'
    var_17 = 'object'
    var_18 = 'string'
    var_19 = {var_14: var_18}
    var_20 = 'integer'
    var_21 = {var_14: var_20}
    var_22 = {var_3: var_19, var_4: var_21}
    var_23 = [var_3]
    var_24 = {var_14: var_17, var_15: var_22, var_16: var_23}
    var_25 = bool(var_13 == var_24)
    assert var_25 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = 'a'
    var_2 = 'A'
    var_3 = (var_1, var_2)
    var_4 = 'b'
    var_5 = 'B'
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = {}
    var_9 = module_0.Choice(choices=var_7, **var_8)
    var_10 = module_1.to_json_schema(var_9)
    var_11 = 'enum'
    var_12 = [var_1, var_4]
    var_13 = {var_11: var_12}
    var_14 = bool(var_10 == var_13)
    assert var_14 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = 42
    var_2 = {}
    var_3 = module_0.Const(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'const'
    var_6 = {var_5: var_1}
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

def test_case_0():
    var_0 = {}
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Cannot convert field type'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 18/20 statements.
# Partially parsed test_to_json_schema_with_reference_field. Retrieved 8/10 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = bool(var_2 == var_5)
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'type'
    var_6 = 'string'
    var_7 = 'null'
    var_8 = [var_6, var_7]
    var_9 = {var_5: var_8}
    var_10 = bool(var_4 == var_9)
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'type'
    var_6 = 'default'
    var_7 = 'string'
    var_8 = {var_5: var_7, var_6: var_0}
    var_9 = bool(var_4 == var_8)
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'integer'
    var_5 = {var_3: var_4}
    var_6 = bool(var_2 == var_5)
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'type'
    var_6 = 'integer'
    var_7 = 'null'
    var_8 = [var_6, var_7]
    var_9 = {var_5: var_8}
    var_10 = bool(var_4 == var_9)
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = 100
    var_2 = {}
    var_3 = module_0.Integer(minimum=var_0, maximum=var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'type'
    var_6 = 'minimum'
    var_7 = 'maximum'
    var_8 = 'integer'
    var_9 = {var_5: var_8, var_6: var_0, var_7: var_1}
    var_10 = bool(var_4 == var_9)
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'boolean'
    var_5 = {var_3: var_4}
    var_6 = bool(var_2 == var_5)
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'type'
    var_6 = 'default'
    var_7 = 'boolean'
    var_8 = {var_5: var_7, var_6: var_0}
    var_9 = bool(var_4 == var_8)
    assert var_9 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'type'
    var_6 = 'items'
    var_7 = 'array'
    var_8 = 'string'
    var_9 = {var_5: var_8}
    var_10 = {var_5: var_7, var_6: var_9}
    var_11 = bool(var_4 == var_10)
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = 1
    var_3 = 10
    var_4 = {}
    var_5 = module_0.Array(var_1, min_items=var_2, max_items=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = 'type'
    var_8 = 'items'
    var_9 = 'minItems'
    var_10 = 'maxItems'
    var_11 = 'array'
    var_12 = 'integer'
    var_13 = {var_7: var_12}
    var_14 = {var_7: var_11, var_8: var_13, var_9: var_2, var_10: var_3}
    var_15 = bool(var_6 == var_14)
    assert var_15 is True

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
    var_7 = 'type'
    var_8 = 'properties'
    var_9 = 'object'
    var_10 = 'string'
    var_11 = {var_7: var_10}
    var_12 = {var_0: var_11}
    var_13 = {var_7: var_9, var_8: var_12}
    var_14 = bool(var_6 == var_13)
    assert var_14 is True

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
    var_11 = 'type'
    var_12 = 'properties'
    var_13 = 'required'
    var_14 = 'object'
    var_15 = 'string'
    var_16 = {var_11: var_15}
    var_17 = 'integer'
    var_18 = {var_11: var_17}
    var_19 = {var_0: var_16, var_1: var_18}
    var_20 = [var_0]
    var_21 = {var_11: var_14, var_12: var_19, var_13: var_20}
    var_22 = bool(var_10 == var_21)
    assert var_22 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = var_1 | var_3
    var_5 = module_1.to_json_schema(var_4)
    var_6 = 'anyOf'
    var_7 = 'type'
    var_8 = 'string'
    var_9 = {var_7: var_8}
    var_10 = 'integer'
    var_11 = {var_7: var_10}
    var_12 = [var_9, var_11]
    var_13 = {var_6: var_12}
    var_14 = bool(var_5 == var_13)
    assert var_14 is True

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
    var_10 = 'enum'
    var_11 = [var_0, var_3]
    var_12 = {var_10: var_11}
    var_13 = bool(var_9 == var_12)
    assert var_13 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'const'
    var_5 = {var_4: var_0}
    var_6 = bool(var_3 == var_5)
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'title'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = module_2.to_json_schema(var_5)
    var_7 = 'type'
    var_8 = 'properties'
    var_9 = 'object'
    var_10 = 'string'
    var_11 = {var_7: var_10}
    var_12 = {var_0: var_11}
    var_13 = {var_7: var_9, var_8: var_12}
    var_14 = bool(var_6 == var_13)
    assert var_14 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'User'
    var_1 = 'name'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_1.Schema(var_4, **var_5)
    var_7 = {var_0: var_6}
    var_8 = [var_7]
    var_9 = 'components'
    var_10 = 'schemas'
    var_11 = 'type'
    var_12 = 'properties'
    var_13 = 'object'
    var_14 = 'string'
    var_15 = {var_11: var_14}
    var_16 = {var_1: var_15}
    var_17 = {var_11: var_13, var_12: var_16}
    var_18 = {var_0: var_17}
    var_19 = {var_10: var_18}
    var_20 = {var_9: var_19}

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = 'User'
    var_7 = 'target'
    var_8 = {var_7: var_5}
    var_9 = '$ref'
    var_10 = '#/components/schemas/User'
    var_11 = {var_9: var_10}

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
    var_8 = 'oneOf'
    var_9 = 'type'
    var_10 = 'string'
    var_11 = {var_9: var_10}
    var_12 = 'integer'
    var_13 = {var_9: var_12}
    var_14 = [var_11, var_13]
    var_15 = {var_8: var_14}
    var_16 = bool(var_7 == var_15)
    assert var_16 is True

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = 10
    var_4 = {}
    var_5 = module_0.String(max_length=var_3, **var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_1.AllOf(var_6, **var_7)
    var_9 = module_2.to_json_schema(var_8)
    var_10 = 'allOf'
    var_11 = 'type'
    var_12 = 'minLength'
    var_13 = 'string'
    var_14 = {var_11: var_13, var_12: var_0}
    var_15 = 'maxLength'
    var_16 = {var_11: var_13, var_15: var_3}
    var_17 = [var_14, var_16]
    var_18 = {var_10: var_17}
    var_19 = bool(var_9 == var_18)
    assert var_19 is True

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
    var_9 = 'if'
    var_10 = 'then'
    var_11 = 'else'
    var_12 = 'type'
    var_13 = 'string'
    var_14 = {var_12: var_13}
    var_15 = 'integer'
    var_16 = {var_12: var_15}
    var_17 = 'boolean'
    var_18 = {var_12: var_17}
    var_19 = {var_9: var_14, var_10: var_16, var_11: var_18}
    var_20 = bool(var_8 == var_19)
    assert var_20 is True

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
    var_6 = 'type'
    var_7 = 'string'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = bool(var_4 == var_9)
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = {}
    var_4 = module_0.Array(var_1, unique_items=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = 'type'
    var_7 = 'items'
    var_8 = 'uniqueItems'
    var_9 = 'array'
    var_10 = 'string'
    var_11 = {var_6: var_10}
    var_12 = {var_6: var_9, var_7: var_11, var_8: var_2}
    var_13 = bool(var_5 == var_12)
    assert var_13 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = False
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, additional_properties=var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'type'
    var_9 = 'properties'
    var_10 = 'additionalProperties'
    var_11 = 'object'
    var_12 = 'string'
    var_13 = {var_8: var_12}
    var_14 = {var_0: var_13}
    var_15 = {var_8: var_11, var_9: var_14, var_10: var_4}
    var_16 = bool(var_7 == var_15)
    assert var_16 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = {}
    var_7 = module_0.Object(properties=var_3, additional_properties=var_5, **var_6)
    var_8 = module_1.to_json_schema(var_7)
    var_9 = 'type'
    var_10 = 'properties'
    var_11 = 'additionalProperties'
    var_12 = 'object'
    var_13 = 'string'
    var_14 = {var_9: var_13}
    var_15 = {var_0: var_14}
    var_16 = 'integer'
    var_17 = {var_9: var_16}
    var_18 = {var_9: var_12, var_10: var_15, var_11: var_17}
    var_19 = bool(var_8 == var_18)
    assert var_19 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'type'
    var_5 = 'minLength'
    var_6 = 'string'
    var_7 = {var_4: var_6, var_5: var_0}
    var_8 = bool(var_3 == var_7)
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'type'
    var_5 = 'maxLength'
    var_6 = 'string'
    var_7 = {var_4: var_6, var_5: var_0}
    var_8 = bool(var_3 == var_7)
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'type'
    var_5 = 'pattern'
    var_6 = 'string'
    var_7 = {var_4: var_6, var_5: var_0}
    var_8 = bool(var_3 == var_7)
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'type'
    var_5 = 'format'
    var_6 = 'string'
    var_7 = {var_4: var_6, var_5: var_0}
    var_8 = bool(var_3 == var_7)
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'number'
    var_5 = {var_3: var_4}
    var_6 = bool(var_2 == var_5)
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'number'
    var_5 = {var_3: var_4}
    var_6 = bool(var_2 == var_5)
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Integer(multiple_of=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'type'
    var_5 = 'multipleOf'
    var_6 = 'integer'
    var_7 = {var_4: var_6, var_5: var_0}
    var_8 = bool(var_3 == var_7)
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = 100
    var_2 = {}
    var_3 = module_0.Integer(exclusive_minimum=var_0, exclusive_maximum=var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'type'
    var_6 = 'exclusiveMinimum'
    var_7 = 'exclusiveMaximum'
    var_8 = 'integer'
    var_9 = {var_5: var_8, var_6: var_0, var_7: var_1}
    var_10 = bool(var_4 == var_9)
    assert var_10 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_from_json_schema_boolean_true. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_boolean_false. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_ref. Retrieved 4/8 statements.
# Partially parsed test_from_json_schema_type_string. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_type_array. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_enum. Retrieved 7/8 statements.
# Partially parsed test_from_json_schema_const. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_allOf. Retrieved 10/11 statements.
# Partially parsed test_from_json_schema_anyOf. Retrieved 9/10 statements.
# Partially parsed test_from_json_schema_oneOf. Retrieved 9/10 statements.
# Partially parsed test_from_json_schema_not. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_if_then_else. Retrieved 10/11 statements.
# Partially parsed test_from_json_schema_multiple_constraints. Retrieved 8/9 statements.
# Partially parsed test_from_json_schema_no_constraints. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_with_null_type. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_with_components. Retrieved 10/11 statements.
# Partially parsed test_from_json_schema_type_empty_with_null. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_type_empty_without_null. Retrieved 2/3 statements.


import typesystem.json_schema as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.from_json_schema(var_0)

def test_case_0():
    var_0 = '$ref'
    var_1 = '#/components/schemas/Test'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'test'
    var_5 = {}

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = 'number'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.from_json_schema(var_4)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'enum'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = module_0.from_json_schema(var_5)

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
    var_4 = 'maxLength'
    var_5 = 10
    var_6 = {var_4: var_5}
    var_7 = [var_3, var_6]
    var_8 = {var_0: var_7}
    var_9 = module_0.from_json_schema(var_8)

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
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = 'maxLength'
    var_6 = 5
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = module_0.from_json_schema(var_8)

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

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.from_json_schema(var_0)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = 'null'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.from_json_schema(var_4)
    var_6 = var_5.allow_null
    assert var_6 is True

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'components'
    var_1 = 'schemas'
    var_2 = 'Test'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = {var_0: var_7}
    var_9 = module_0.from_json_schema(var_8)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = []
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)
    var_4 = var_3.const
    assert var_4 is None

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.from_json_schema(var_0)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_pattern_regex_flags_unicode. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 'pattern'



# Parsed testcases at query #5
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = bool(var_2 == var_5)
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'type'
    var_6 = 'string'
    var_7 = 'null'
    var_8 = [var_6, var_7]
    var_9 = {var_5: var_8}
    var_10 = bool(var_4 == var_9)
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'type'
    var_5 = 'minLength'
    var_6 = 'string'
    var_7 = {var_4: var_6, var_5: var_0}
    var_8 = bool(var_3 == var_7)
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'type'
    var_5 = 'maxLength'
    var_6 = 'string'
    var_7 = {var_4: var_6, var_5: var_0}
    var_8 = bool(var_3 == var_7)
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'type'
    var_5 = 'pattern'
    var_6 = 'string'
    var_7 = {var_4: var_6, var_5: var_0}
    var_8 = bool(var_3 == var_7)
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'type'
    var_5 = 'format'
    var_6 = 'string'
    var_7 = {var_4: var_6, var_5: var_0}
    var_8 = bool(var_3 == var_7)
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'integer'
    var_5 = {var_3: var_4}
    var_6 = bool(var_2 == var_5)
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Integer(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'type'
    var_6 = 'integer'
    var_7 = 'null'
    var_8 = [var_6, var_7]
    var_9 = {var_5: var_8}
    var_10 = bool(var_4 == var_9)
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'type'
    var_5 = 'minimum'
    var_6 = 'integer'
    var_7 = {var_4: var_6, var_5: var_0}
    var_8 = bool(var_3 == var_7)
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'type'
    var_5 = 'maximum'
    var_6 = 'integer'
    var_7 = {var_4: var_6, var_5: var_0}
    var_8 = bool(var_3 == var_7)
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(exclusive_minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'type'
    var_5 = 'exclusiveMinimum'
    var_6 = 'integer'
    var_7 = {var_4: var_6, var_5: var_0}
    var_8 = bool(var_3 == var_7)
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(exclusive_maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'type'
    var_5 = 'exclusiveMaximum'
    var_6 = 'integer'
    var_7 = {var_4: var_6, var_5: var_0}
    var_8 = bool(var_3 == var_7)
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Integer(multiple_of=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'type'
    var_5 = 'multipleOf'
    var_6 = 'integer'
    var_7 = {var_4: var_6, var_5: var_0}
    var_8 = bool(var_3 == var_7)
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'number'
    var_5 = {var_3: var_4}
    var_6 = bool(var_2 == var_5)
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Float(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'type'
    var_6 = 'number'
    var_7 = 'null'
    var_8 = [var_6, var_7]
    var_9 = {var_5: var_8}
    var_10 = bool(var_4 == var_9)
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'boolean'
    var_5 = {var_3: var_4}
    var_6 = bool(var_2 == var_5)
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'type'
    var_6 = 'boolean'
    var_7 = 'null'
    var_8 = [var_6, var_7]
    var_9 = {var_5: var_8}
    var_10 = bool(var_4 == var_9)
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'array'
    var_5 = {var_3: var_4}
    var_6 = bool(var_2 == var_5)
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'type'
    var_6 = 'array'
    var_7 = 'null'
    var_8 = [var_6, var_7]
    var_9 = {var_5: var_8}
    var_10 = bool(var_4 == var_9)
    assert var_10 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'type'
    var_5 = 'minItems'
    var_6 = 'array'
    var_7 = {var_4: var_6, var_5: var_0}
    var_8 = bool(var_3 == var_7)
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'type'
    var_5 = 'maxItems'
    var_6 = 'array'
    var_7 = {var_4: var_6, var_5: var_0}
    var_8 = bool(var_3 == var_7)
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'type'
    var_6 = 'items'
    var_7 = 'array'
    var_8 = 'string'
    var_9 = {var_5: var_8}
    var_10 = {var_5: var_7, var_6: var_9}
    var_11 = bool(var_4 == var_10)
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
    var_6 = module_0.Array(var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'type'
    var_9 = 'items'
    var_10 = 'array'
    var_11 = 'string'
    var_12 = {var_8: var_11}
    var_13 = 'integer'
    var_14 = {var_8: var_13}
    var_15 = [var_12, var_14]
    var_16 = {var_8: var_10, var_9: var_15}
    var_17 = bool(var_7 == var_16)
    assert var_17 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Array(additional_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'type'
    var_5 = 'additionalItems'
    var_6 = 'array'
    var_7 = {var_4: var_6, var_5: var_0}
    var_8 = bool(var_3 == var_7)
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(additional_items=var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'type'
    var_6 = 'additionalItems'
    var_7 = 'array'
    var_8 = 'string'
    var_9 = {var_5: var_8}
    var_10 = {var_5: var_7, var_6: var_9}
    var_11 = bool(var_4 == var_10)
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'type'
    var_5 = 'uniqueItems'
    var_6 = 'array'
    var_7 = {var_4: var_6, var_5: var_0}
    var_8 = bool(var_3 == var_7)
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'object'
    var_5 = {var_3: var_4}
    var_6 = bool(var_2 == var_5)
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'type'
    var_6 = 'object'
    var_7 = 'null'
    var_8 = [var_6, var_7]
    var_9 = {var_5: var_8}
    var_10 = bool(var_4 == var_9)
    assert var_10 is True

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
    var_7 = 'type'
    var_8 = 'properties'
    var_9 = 'object'
    var_10 = 'string'
    var_11 = {var_7: var_10}
    var_12 = {var_0: var_11}
    var_13 = {var_7: var_9, var_8: var_12}
    var_14 = bool(var_6 == var_13)
    assert var_14 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = {}
    var_2 = module_0.Integer(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(pattern_properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = 'type'
    var_8 = 'patternProperties'
    var_9 = 'object'
    var_10 = 'integer'
    var_11 = {var_7: var_10}
    var_12 = {var_0: var_11}
    var_13 = {var_7: var_9, var_8: var_12}
    var_14 = bool(var_6 == var_13)
    assert var_14 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'type'
    var_5 = 'additionalProperties'
    var_6 = 'object'
    var_7 = {var_4: var_6, var_5: var_0}
    var_8 = bool(var_3 == var_7)
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Object(additional_properties=var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'type'
    var_6 = 'additionalProperties'
    var_7 = 'object'
    var_8 = 'string'
    var_9 = {var_5: var_8}
    var_10 = {var_5: var_7, var_6: var_9}
    var_11 = bool(var_4 == var_10)
    assert var_11 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Object(property_names=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = 'type'
    var_7 = 'propertyNames'
    var_8 = 'object'
    var_9 = 'pattern'
    var_10 = 'string'
    var_11 = {var_6: var_10, var_9: var_0}
    var_12 = {var_6: var_8, var_7: var_11}
    var_13 = bool(var_5 == var_12)
    assert var_13 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Object(max_properties=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'type'
    var_5 = 'maxProperties'
    var_6 = 'object'
    var_7 = {var_4: var_6, var_5: var_0}
    var_8 = bool(var_3 == var_7)
    assert var_8 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'type'
    var_5 = 'minProperties'
    var_6 = 'object'
    var_7 = {var_4: var_6, var_5: var_0}
    var_8 = bool(var_3 == var_7)
    assert var_8 is True

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
    var_11 = 'type'
    var_12 = 'properties'
    var_13 = 'required'
    var_14 = 'object'
    var_15 = 'string'
    var_16 = {var_11: var_15}
    var_17 = 'integer'
    var_18 = {var_11: var_17}
    var_19 = {var_0: var_16, var_1: var_18}
    var_20 = [var_0]
    var_21 = {var_11: var_14, var_12: var_19, var_13: var_20}
    var_22 = bool(var_10 == var_21)
    assert var_22 is True

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
    var_7 = 'type'
    var_8 = 'properties'
    var_9 = 'object'
    var_10 = 'string'
    var_11 = {var_7: var_10}
    var_12 = {var_0: var_11}
    var_13 = {var_7: var_9, var_8: var_12}
    var_14 = bool(var_6 == var_13)
    assert var_14 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_from_json_schema_type_number. Retrieved 11/14 statements.
# Partially parsed test_from_json_schema_type_integer. Retrieved 10/13 statements.
# Partially parsed test_from_json_schema_type_string. Retrieved 11/14 statements.
# Partially parsed test_from_json_schema_type_boolean. Retrieved 3/6 statements.
# Partially parsed test_from_json_schema_type_array. Retrieved 14/19 statements.
# Partially parsed test_from_json_schema_type_object. Retrieved 17/22 statements.


def test_case_0():
    var_0 = 'minimum'
    var_1 = 'maximum'
    var_2 = 'exclusiveMinimum'
    var_3 = 'exclusiveMaximum'
    var_4 = 'multipleOf'
    var_5 = 0
    var_6 = 10
    var_7 = 2
    var_8 = {var_0: var_5, var_1: var_6, var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = []
    var_10 = 'number'
    var_11 = False

def test_case_0():
    var_0 = 'minimum'
    var_1 = 'maximum'
    var_2 = 'exclusiveMinimum'
    var_3 = 'exclusiveMaximum'
    var_4 = 'multipleOf'
    var_5 = 5
    var_6 = 15
    var_7 = {var_0: var_5, var_1: var_6, var_2: var_5, var_3: var_6, var_4: var_5}
    var_8 = []
    var_9 = 'integer'
    var_10 = True

def test_case_0():
    var_0 = 'minLength'
    var_1 = 'maxLength'
    var_2 = 'format'
    var_3 = 'pattern'
    var_4 = 1
    var_5 = 10
    var_6 = 'email'
    var_7 = '^[a-z]+$'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = []
    var_10 = 'string'
    var_11 = False

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'boolean'
    var_3 = True

def test_case_0():
    var_0 = 'minItems'
    var_1 = 'maxItems'
    var_2 = 'uniqueItems'
    var_3 = 'items'
    var_4 = 'additionalItems'
    var_5 = 1
    var_6 = 5
    var_7 = True
    var_8 = 'type'
    var_9 = 'string'
    var_10 = {var_8: var_9}
    var_11 = False
    var_12 = {var_0: var_5, var_1: var_6, var_2: var_7, var_3: var_10, var_4: var_11}
    var_13 = []
    var_14 = 'array'

def test_case_0():
    var_0 = 'minProperties'
    var_1 = 'maxProperties'
    var_2 = 'required'
    var_3 = 'properties'
    var_4 = 'additionalProperties'
    var_5 = 1
    var_6 = 3
    var_7 = 'id'
    var_8 = [var_7]
    var_9 = 'type'
    var_10 = 'string'
    var_11 = {var_9: var_10}
    var_12 = {var_7: var_11}
    var_13 = False
    var_14 = {var_0: var_5, var_1: var_6, var_2: var_8, var_3: var_12, var_4: var_13}
    var_15 = []
    var_16 = 'object'
    var_17 = True



# Parsed testcases at query #7
#--------------------------




import typesystem.json_schema as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'object'
    var_2 = False
    var_3 = {}
    var_4 = module_0.from_json_schema_type(var_0, var_1, var_2, var_3)
    var_5 = var_4.additional_properties
    assert var_5 is None



# Parsed testcases at query #8
#--------------------------






# Parsed testcases at query #9
#--------------------------

# Partially parsed test_ref_from_json_schema_creates_reference_with_correct_to. Retrieved 3/5 statements.
# Partially parsed test_ref_from_json_schema_raises_assertion_error_for_non_hash_ref. Retrieved 3/6 statements.


def test_case_0():
    var_0 = []
    var_1 = '$ref'
    var_2 = '#/components/schemas/User'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = []
    var_1 = '$ref'
    var_2 = 'http://example.com/schema'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_predicate_at_line_120_evaluates_to_true. Retrieved 2/6 statements.


def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = False
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_if_then_else_from_json_schema_basic. Retrieved 13/19 statements.
# Partially parsed test_if_then_else_from_json_schema_without_then. Retrieved 10/14 statements.
# Partially parsed test_if_then_else_from_json_schema_without_else. Retrieved 10/15 statements.
# Partially parsed test_if_then_else_from_json_schema_with_default. Retrieved 13/17 statements.
# Partially parsed test_if_then_else_from_json_schema_with_nested_schemas. Retrieved 22/28 statements.
# Partially parsed test_if_then_else_from_json_schema_with_ref_in_if. Retrieved 13/19 statements.
# Partially parsed test_if_then_else_from_json_schema_complex_condition. Retrieved 19/24 statements.
# Partially parsed test_if_then_else_from_json_schema_with_boolean_schemas. Retrieved 10/18 statements.
# Partially parsed test_if_then_else_from_json_schema_empty_object_defaults. Retrieved 5/9 statements.
# Partially parsed test_if_then_else_from_json_schema_with_enum_in_then. Retrieved 17/21 statements.


def test_case_0():
    var_0 = []
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'else'
    var_4 = 'type'
    var_5 = 'string'
    var_6 = {var_4: var_5}
    var_7 = 'integer'
    var_8 = {var_4: var_7}
    var_9 = 'boolean'
    var_10 = {var_4: var_9}
    var_11 = {var_1: var_6, var_2: var_8, var_3: var_10}
    var_12 = 'hello'
    var_13 = False

def test_case_0():
    var_0 = []
    var_1 = 'if'
    var_2 = 'else'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 'integer'
    var_7 = {var_3: var_6}
    var_8 = {var_1: var_5, var_2: var_7}
    var_9 = 'test'
    var_10 = 5

def test_case_0():
    var_0 = []
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 'integer'
    var_7 = {var_3: var_6}
    var_8 = {var_1: var_5, var_2: var_7}
    var_9 = 'hello'
    var_10 = 3.14

def test_case_0():
    var_0 = []
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'else'
    var_4 = 'default'
    var_5 = 'type'
    var_6 = 'string'
    var_7 = {var_5: var_6}
    var_8 = 'integer'
    var_9 = {var_5: var_8}
    var_10 = 'boolean'
    var_11 = {var_5: var_10}
    var_12 = 42
    var_13 = {var_1: var_7, var_2: var_9, var_3: var_11, var_4: var_12}

def test_case_0():
    var_0 = []
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'else'
    var_4 = 'type'
    var_5 = 'properties'
    var_6 = 'object'
    var_7 = 'x'
    var_8 = 'integer'
    var_9 = {var_4: var_8}
    var_10 = {var_7: var_9}
    var_11 = {var_4: var_6, var_5: var_10}
    var_12 = 'string'
    var_13 = {var_4: var_12}
    var_14 = 'array'
    var_15 = {var_4: var_14}
    var_16 = {var_1: var_11, var_2: var_13, var_3: var_15}
    var_17 = 5
    var_18 = {var_7: var_17}
    var_19 = 1
    var_20 = 2
    var_21 = 3
    var_22 = [var_19, var_20, var_21]

def test_case_0():
    var_0 = []
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'if'
    var_5 = 'then'
    var_6 = '$ref'
    var_7 = '#/components/schemas/MyString'
    var_8 = {var_6: var_7}
    var_9 = 'integer'
    var_10 = {var_1: var_9}
    var_11 = {var_4: var_8, var_5: var_10}
    var_12 = 'hello'
    var_13 = 123

def test_case_0():
    var_0 = []
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'else'
    var_4 = 'allOf'
    var_5 = 'type'
    var_6 = 'number'
    var_7 = {var_5: var_6}
    var_8 = 'minimum'
    var_9 = 0
    var_10 = {var_8: var_9}
    var_11 = [var_7, var_10]
    var_12 = {var_4: var_11}
    var_13 = 'string'
    var_14 = {var_5: var_13}
    var_15 = 'null'
    var_16 = {var_5: var_15}
    var_17 = {var_1: var_12, var_2: var_14, var_3: var_16}
    var_18 = 10
    var_19 = -5

import typesystem.fields as module_0

def test_case_0():
    var_0 = []
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'else'
    var_4 = True
    var_5 = False
    var_6 = {var_1: var_4, var_2: var_5, var_3: var_4}
    var_7 = 'anything'
    var_8 = module_0.Any()
    var_9 = [var_8]
    var_10 = 'other'
    var_11 = module_0.Any()
    var_12 = [var_11]

def test_case_0():
    var_0 = []
    var_1 = 'if'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = 'value'
    var_5 = None

def test_case_0():
    var_0 = []
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'else'
    var_4 = 'type'
    var_5 = 'string'
    var_6 = {var_4: var_5}
    var_7 = 'enum'
    var_8 = 'a'
    var_9 = 'b'
    var_10 = 'c'
    var_11 = [var_8, var_9, var_10]
    var_12 = {var_7: var_11}
    var_13 = 'number'
    var_14 = {var_4: var_13}
    var_15 = {var_1: var_6, var_2: var_12, var_3: var_14}
    var_16 = 'hello'
    var_17 = 42



# Parsed testcases at query #12
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Array(additional_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['additionalItems']
    assert var_4 is False



# Parsed testcases at query #13
#--------------------------






# Parsed testcases at query #14
#--------------------------

# Partially parsed test_property_names_is_none. Retrieved 3/5 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'object'
    var_2 = False
    var_3 = []



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_pattern_regex_flags_unicode. Retrieved 1/7 statements.


def test_case_0():
    var_0 = '^test$'
    var_1 = 'pattern'



# Parsed testcases at query #16
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = None
    var_5 = {}
    var_6 = module_1.IfThenElse(var_1, var_3, var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = 'else'
    var_9 = bool('else' not in var_7)
    assert var_9 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_from_json_schema_type_number. Retrieved 13/16 statements.
# Partially parsed test_from_json_schema_type_integer. Retrieved 13/16 statements.
# Partially parsed test_from_json_schema_type_string. Retrieved 13/16 statements.
# Partially parsed test_from_json_schema_type_boolean. Retrieved 5/8 statements.
# Partially parsed test_from_json_schema_type_array. Retrieved 16/21 statements.
# Partially parsed test_from_json_schema_type_object. Retrieved 18/23 statements.
# Partially parsed test_from_json_schema_type_with_allow_null. Retrieved 3/6 statements.
# Partially parsed test_from_json_schema_type_without_optional_fields. Retrieved 3/6 statements.
# Partially parsed test_from_json_schema_type_array_with_list_items. Retrieved 12/19 statements.
# Partially parsed test_from_json_schema_type_object_with_pattern_properties. Retrieved 9/14 statements.
# Partially parsed test_from_json_schema_type_object_with_additional_properties_field. Retrieved 7/12 statements.
# Partially parsed test_from_json_schema_type_object_with_property_names. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 'minimum'
    var_1 = 'maximum'
    var_2 = 'exclusiveMinimum'
    var_3 = 'exclusiveMaximum'
    var_4 = 'multipleOf'
    var_5 = 'default'
    var_6 = 0
    var_7 = 10
    var_8 = 2
    var_9 = 4
    var_10 = {var_0: var_6, var_1: var_7, var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_9}
    var_11 = []
    var_12 = 'number'
    var_13 = False

def test_case_0():
    var_0 = 'minimum'
    var_1 = 'maximum'
    var_2 = 'exclusiveMinimum'
    var_3 = 'exclusiveMaximum'
    var_4 = 'multipleOf'
    var_5 = 'default'
    var_6 = 0
    var_7 = 10
    var_8 = 2
    var_9 = 4
    var_10 = {var_0: var_6, var_1: var_7, var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_9}
    var_11 = []
    var_12 = 'integer'
    var_13 = False

def test_case_0():
    var_0 = 'minLength'
    var_1 = 'maxLength'
    var_2 = 'format'
    var_3 = 'pattern'
    var_4 = 'default'
    var_5 = 1
    var_6 = 10
    var_7 = 'email'
    var_8 = '^a.*z$'
    var_9 = 'test'
    var_10 = {var_0: var_5, var_1: var_6, var_2: var_7, var_3: var_8, var_4: var_9}
    var_11 = []
    var_12 = 'string'
    var_13 = False

def test_case_0():
    var_0 = 'default'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'boolean'
    var_5 = False

def test_case_0():
    var_0 = 'items'
    var_1 = 'minItems'
    var_2 = 'maxItems'
    var_3 = 'uniqueItems'
    var_4 = 'default'
    var_5 = 'type'
    var_6 = 'string'
    var_7 = {var_5: var_6}
    var_8 = 1
    var_9 = 10
    var_10 = True
    var_11 = 'a'
    var_12 = [var_11]
    var_13 = {var_0: var_7, var_1: var_8, var_2: var_9, var_3: var_10, var_4: var_12}
    var_14 = []
    var_15 = 'array'
    var_16 = False

def test_case_0():
    var_0 = 'properties'
    var_1 = 'required'
    var_2 = 'minProperties'
    var_3 = 'maxProperties'
    var_4 = 'default'
    var_5 = 'name'
    var_6 = 'type'
    var_7 = 'string'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = [var_5]
    var_11 = 1
    var_12 = 2
    var_13 = 'John'
    var_14 = {var_5: var_13}
    var_15 = {var_0: var_9, var_1: var_10, var_2: var_11, var_3: var_12, var_4: var_14}
    var_16 = []
    var_17 = 'object'
    var_18 = False

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'string'
    var_3 = True

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'number'
    var_3 = False

def test_case_0():
    var_0 = 'items'
    var_1 = 'additionalItems'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = 'integer'
    var_6 = {var_2: var_5}
    var_7 = [var_4, var_6]
    var_8 = False
    var_9 = {var_0: var_7, var_1: var_8}
    var_10 = []
    var_11 = 'array'
    var_12 = 1

def test_case_0():
    var_0 = 'patternProperties'
    var_1 = '^a.*$'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = []
    var_8 = 'object'
    var_9 = False

def test_case_0():
    var_0 = 'additionalProperties'
    var_1 = 'type'
    var_2 = 'integer'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = []
    var_6 = 'object'
    var_7 = False

def test_case_0():
    var_0 = 'propertyNames'
    var_1 = 'pattern'
    var_2 = '^[a-z]+$'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = []
    var_6 = 'object'
    var_7 = False



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_ref_from_json_schema_with_valid_ref. Retrieved 3/6 statements.
# Partially parsed test_ref_from_json_schema_raises_assertion_error_for_non_hash_ref. Retrieved 3/6 statements.
# Partially parsed test_ref_from_json_schema_raises_key_error_for_missing_ref_key. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = '$ref'
    var_2 = '#/components/schemas/User'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = []
    var_1 = '$ref'
    var_2 = 'http://example.com/schema'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_line_172_predicate_true_with_definitions. Retrieved 2/4 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'TestRef'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = 'target'
    var_4 = {var_3: var_2}
    var_5 = 'components'
    var_6 = 'schemas'
    var_7 = 'TestRef'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_if_then_else_from_json_schema_with_else_key_present. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 'if'
    var_1 = 'else'
    var_2 = {}
    var_3 = {}
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []



# Parsed testcases at query #21
#--------------------------






# Parsed testcases at query #22
#--------------------------

# Failed to parse test_to_json_schema_with_definitions.
# Partially parsed test_to_json_schema_array_field_with_items_list. Retrieved 7/8 statements.
# Partially parsed test_to_json_schema_union_field. Retrieved 6/7 statements.
# Partially parsed test_to_json_schema_one_of_field. Retrieved 7/8 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'string'

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
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minLength']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxLength']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['pattern']
    assert var_4 == '^\\d+$'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['format']
    assert var_4 == 'email'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'integer'

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
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minimum']
    assert var_4 == 0

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maximum']
    assert var_4 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = {}
    var_2 = module_0.Integer(exclusive_minimum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['exclusiveMinimum']
    assert var_4 == 0

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = {}
    var_2 = module_0.Integer(exclusive_maximum=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['exclusiveMaximum']
    assert var_4 == 100

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Integer(multiple_of=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['multipleOf']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'number'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'boolean'

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
    var_1 = module_0.Array(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'array'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['array', 'null'])
    assert var_6 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minItems']
    assert var_4 == 1

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxItems']
    assert var_4 == 10

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'items'
    var_6 = bool('items' in var_4)
    assert var_6 is True
    var_7 = var_4['items']['type']
    assert var_7 == 'string'

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
    var_9 = bool('items' in var_7)
    assert var_9 is True
    var_10 = 'items'
    var_11 = var_7[var_10]
    var_12 = var_7['items'][0]['type']
    assert var_12 == 'string'
    var_13 = var_7['items'][1]['type']
    assert var_13 == 'integer'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Array(additional_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['additionalItems']
    assert var_4 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['uniqueItems']
    assert var_4 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = var_2['type']
    assert var_3 == 'object'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['object', 'null'])
    assert var_6 is True

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
    var_7 = 'properties'
    var_8 = bool('properties' in var_6)
    assert var_8 is True
    var_9 = 'name'
    var_10 = bool('name' in var_6['properties'])
    assert var_10 is True
    var_11 = var_6['properties']['name']['type']
    assert var_11 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(pattern_properties=var_3, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = 'patternProperties'
    var_8 = bool('patternProperties' in var_6)
    assert var_8 is True
    var_9 = '^\\d+$'
    var_10 = bool('^\\d+$' in var_6['patternProperties'])
    assert var_10 is True
    var_11 = var_6['patternProperties']['^\\d+$']['type']
    assert var_11 == 'string'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['additionalProperties']
    assert var_4 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Object(property_names=var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = 'propertyNames'
    var_7 = bool('propertyNames' in var_5)
    assert var_7 is True
    var_8 = var_5['propertyNames']['type']
    assert var_8 == 'string'
    var_9 = var_5['propertyNames']['pattern']
    assert var_9 == '^[a-z]+$'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Object(max_properties=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['maxProperties']
    assert var_4 == 5

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = var_3['minProperties']
    assert var_4 == 1

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
    var_11 = 'required'
    var_12 = bool('required' in var_10)
    assert var_12 is True
    var_13 = var_10['required']
    var_14 = bool(var_10['required'] == ['name'])
    assert var_14 is True

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
    var_7 = var_6['type']
    assert var_7 == 'object'
    var_8 = 'properties'
    var_9 = bool('properties' in var_6)
    assert var_9 is True
    var_10 = 'name'
    var_11 = bool('name' in var_6['properties'])
    assert var_11 is True
    var_12 = var_6['properties']['name']['type']
    assert var_12 == 'string'

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
    var_9 = 'required'
    var_10 = bool('required' in var_8)
    assert var_10 is True
    var_11 = var_8['required']
    var_12 = bool(var_8['required'] == ['name'])
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
    var_10 = 'enum'
    var_11 = bool('enum' in var_9)
    assert var_11 is True
    var_12 = var_9['enum']
    var_13 = bool(var_9['enum'] == ['a', 'b'])
    assert var_13 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'const'
    var_5 = bool('const' in var_3)
    assert var_5 is True
    var_6 = var_3['const']
    assert var_6 == 'fixed_value'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = var_1 | var_3
    var_5 = module_1.to_json_schema(var_4)
    var_6 = 'anyOf'
    var_7 = bool('anyOf' in var_5)
    assert var_7 is True
    var_8 = 'anyOf'
    var_9 = var_5[var_8]
    var_10 = var_5['anyOf'][0]['type']
    assert var_10 == 'string'
    var_11 = var_5['anyOf'][1]['type']
    assert var_11 == 'integer'

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
    var_8 = 'oneOf'
    var_9 = bool('oneOf' in var_7)
    assert var_9 is True
    var_10 = 'oneOf'
    var_11 = var_7[var_10]



