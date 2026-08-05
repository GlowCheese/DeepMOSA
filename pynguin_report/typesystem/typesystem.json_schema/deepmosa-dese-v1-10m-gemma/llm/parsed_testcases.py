####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_from_json_schema_bool_true. Retrieved 2/5 statements.
# Partially parsed test_from_json_schema_bool_false. Retrieved 2/5 statements.
# Partially parsed test_from_json_schema_ref. Retrieved 5/8 statements.
# Partially parsed test_from_json_schema_enum. Retrieved 7/10 statements.


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
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = '#/components/schemas/MyType'
    var_3 = {var_1: var_2}
    var_4 = module_1.from_json_schema(var_3, var_0)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'enum'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.Definitions()
    var_6 = module_1.from_json_schema(var_4, var_5)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_from_json_schema_type_number. Retrieved 11/12 statements.
# Partially parsed test_from_json_schema_type_integer. Retrieved 9/10 statements.
# Partially parsed test_from_json_schema_type_string. Retrieved 13/14 statements.
# Partially parsed test_from_json_schema_type_boolean. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_type_array. Retrieved 20/23 statements.
# Partially parsed test_from_json_schema_type_object. Retrieved 30/35 statements.


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
    var_7 = module_0.Definitions()
    var_8 = 'number'
    var_9 = True
    var_10 = module_1.from_json_schema_type(var_6, var_8, var_9, var_7)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'minimum'
    var_1 = 'exclusiveMinimum'
    var_2 = 1
    var_3 = 0
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Definitions()
    var_6 = 'integer'
    var_7 = False
    var_8 = module_1.from_json_schema_type(var_4, var_6, var_7, var_5)

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
    var_9 = module_0.Definitions()
    var_10 = 'string'
    var_11 = False
    var_12 = module_1.from_json_schema_type(var_8, var_10, var_11, var_9)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'default'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = module_0.Definitions()
    var_4 = 'boolean'
    var_5 = module_1.from_json_schema_type(var_2, var_4, var_1, var_3)

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
    var_14 = module_0.Definitions()
    var_15 = 'array'
    var_16 = False
    var_17 = module_1.from_json_schema_type(var_13, var_15, var_16, var_14)
    var_18 = var_17.items
    var_19 = var_17.additional_items

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'properties'
    var_1 = 'patternProperties'
    var_2 = 'additionalProperties'
    var_3 = 'propertyNames'
    var_4 = 'required'
    var_5 = 'minProperties'
    var_6 = 'name'
    var_7 = 'age'
    var_8 = 'type'
    var_9 = 'string'
    var_10 = {var_8: var_9}
    var_11 = 'integer'
    var_12 = {var_8: var_11}
    var_13 = {var_6: var_10, var_7: var_12}
    var_14 = '^id_'
    var_15 = {var_8: var_11}
    var_16 = {var_14: var_15}
    var_17 = False
    var_18 = {var_8: var_9}
    var_19 = [var_6]
    var_20 = 1
    var_21 = {var_0: var_13, var_1: var_16, var_2: var_17, var_3: var_18, var_4: var_19, var_5: var_20}
    var_22 = module_0.Definitions()
    var_23 = 'object'
    var_24 = True
    var_25 = module_1.from_json_schema_type(var_21, var_23, var_24, var_22)
    var_26 = var_25.properties[var_6]
    var_27 = var_25.properties[var_7]
    var_28 = var_25.pattern_properties[var_14]
    var_29 = var_25.property_names



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_to_json_schema_definitions. Retrieved 3/7 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'not_actually_default'
    var_2 = module_0.Boolean()
    var_3 = module_1.to_json_schema(var_2)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = False
    var_3 = module_0.Integer(minimum=var_0, maximum=var_1)
    var_4 = module_1.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = module_0.String(max_length=var_1, min_length=var_0)
    var_3 = module_1.to_json_schema(var_2)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = 1
    var_2 = module_0.Array(var_0, min_items=var_1)
    var_3 = module_1.to_json_schema(var_2)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_0]
    var_6 = module_0.Object(properties=var_4, required=var_5)
    var_7 = module_1.to_json_schema(var_6)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'anyOf'
    var_6 = var_4[var_5]
    var_7 = len(var_6)
    assert var_7 == 2

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'MyString'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed_value'
    var_1 = module_0.Const(var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'A'
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'B'
    var_4 = (var_3, var_1)
    var_5 = [var_2, var_4]
    var_6 = module_0.Choice(choices=var_5)
    var_7 = module_1.to_json_schema(var_6)



# Parsed testcases at query #4
#--------------------------




import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'test_field'
    var_1 = True
    var_2 = module_0.String()
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = module_2.to_json_schema(var_4)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_from_json_schema_type_number. Retrieved 11/12 statements.
# Partially parsed test_from_json_schema_type_integer. Retrieved 8/9 statements.
# Partially parsed test_from_json_schema_type_string. Retrieved 13/14 statements.
# Partially parsed test_from_json_schema_type_boolean. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_type_array. Retrieved 20/23 statements.
# Partially parsed test_from_json_schema_type_object. Retrieved 23/26 statements.


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
    var_7 = 'number'
    var_8 = False
    var_9 = module_0.Definitions()
    var_10 = module_1.from_json_schema_type(var_6, var_7, var_8, var_9)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'minimum'
    var_1 = 'exclusiveMinimum'
    var_2 = 0
    var_3 = {var_0: var_2, var_1: var_2}
    var_4 = 'integer'
    var_5 = True
    var_6 = module_0.Definitions()
    var_7 = module_1.from_json_schema_type(var_3, var_4, var_5, var_6)

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
    var_9 = 'string'
    var_10 = False
    var_11 = module_0.Definitions()
    var_12 = module_1.from_json_schema_type(var_8, var_9, var_10, var_11)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'minLength'
    var_1 = 0
    var_2 = {var_0: var_1}
    var_3 = 'string'
    var_4 = False
    var_5 = module_0.Definitions()
    var_6 = module_1.from_json_schema_type(var_2, var_3, var_4, var_5)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'default'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'boolean'
    var_4 = module_0.Definitions()
    var_5 = module_1.from_json_schema_type(var_2, var_3, var_1, var_4)

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
    var_16 = module_0.Definitions()
    var_17 = module_1.from_json_schema_type(var_13, var_14, var_15, var_16)
    var_18 = var_17.items
    var_19 = var_17.additional_items

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'properties'
    var_1 = 'patternProperties'
    var_2 = 'additionalProperties'
    var_3 = 'required'
    var_4 = 'minProperties'
    var_5 = 'name'
    var_6 = 'type'
    var_7 = 'string'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = '^id_'
    var_11 = 'integer'
    var_12 = {var_6: var_11}
    var_13 = {var_10: var_12}
    var_14 = False
    var_15 = [var_5]
    var_16 = 1
    var_17 = {var_0: var_9, var_1: var_13, var_2: var_14, var_3: var_15, var_4: var_16}
    var_18 = module_0.Definitions()
    var_19 = 'object'
    var_20 = module_1.from_json_schema_type(var_17, var_19, var_14, var_18)
    var_21 = var_20.properties[var_5]
    var_22 = var_20.pattern_properties[var_10]



# Parsed testcases at query #6
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = module_0.Integer(exclusive_minimum=var_0)
    var_2 = module_1.to_json_schema(var_1)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_test_type_from_json_schema_string. Retrieved 9/10 statements.
# Partially parsed test_test_type_from_json_schema_integer. Retrieved 7/8 statements.
# Partially parsed test_test_type_from_json_schema_union. Retrieved 9/10 statements.
# Partially parsed test_test_type_from_json_schema_null_allowed. Retrieved 7/8 statements.
# Partially parsed test_test_type_from_json_schema_empty_type_list. Retrieved 5/6 statements.
# Partially parsed test_test_type_from_json_schema_const_none. Retrieved 8/9 statements.


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
    var_7 = module_0.Definitions()
    var_8 = module_1.type_from_json_schema(var_6, var_7)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'minimum'
    var_2 = 'integer'
    var_3 = 0
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Definitions()
    var_6 = module_1.type_from_json_schema(var_4, var_5)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = 'integer'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.Definitions()
    var_6 = module_1.type_from_json_schema(var_4, var_5)
    var_7 = var_6.any_of
    var_8 = len(var_7)
    assert var_8 == 2

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = 'null'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.Definitions()
    var_6 = module_1.type_from_json_schema(var_4, var_5)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = []
    var_2 = {var_0: var_1}
    var_3 = module_0.Definitions()
    var_4 = module_1.type_from_json_schema(var_2, var_3)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = []
    var_2 = {var_0: var_1}
    var_3 = 'null'
    var_4 = [var_3]
    var_5 = {var_0: var_4}
    var_6 = module_0.Definitions()
    var_7 = module_1.type_from_json_schema(var_5, var_6)



# Parsed testcases at query #8
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = '^abc$'
    var_1 = True
    var_2 = module_0.String()
    var_3 = {var_0: var_2}
    var_4 = module_0.Object(pattern_properties=var_3)
    var_5 = module_1.to_json_schema(var_4)



# Parsed testcases at query #9
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Integer()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Integer()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Float()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Decimal()
    var_2 = module_1.to_json_schema(var_1)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_to_json_schema_array. Retrieved 3/6 statements.
# Partially parsed test_to_json_schema_definitions_handling. Retrieved 3/6 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'hello'
    var_2 = module_0.String()
    var_3 = module_1.to_json_schema(var_2)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = False
    var_3 = module_0.Integer(minimum=var_0, maximum=var_1)
    var_4 = module_1.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean()
    var_2 = module_1.to_json_schema(var_1)

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = True

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_0]
    var_6 = module_0.Object(properties=var_4, required=var_5)
    var_7 = module_1.to_json_schema(var_6)

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
    var_7 = module_0.Choice(choices=var_6)
    var_8 = module_1.to_json_schema(var_7)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'static'
    var_1 = module_0.Const()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'anyOf'
    var_6 = var_4[var_5]
    var_7 = len(var_6)
    assert var_7 == 2

import typesystem.fields as module_0
import typesystem.json_schema as module_1
import typesystem.composites as module_2

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True
    var_2 = module_2.NeverMatch()
    var_3 = module_1.to_json_schema(var_2)
    assert var_3 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'User'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}



# Parsed testcases at query #11
#--------------------------






# Parsed testcases at query #12
#--------------------------

# Partially parsed test_to_json_schema_definitions. Retrieved 3/7 statements.
# Failed to parse test_to_json_schema_error_on_unknown_type.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 5
    var_2 = module_0.String(min_length=var_1)
    var_3 = module_1.to_json_schema(var_2)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = module_0.Integer(minimum=var_1)
    var_3 = module_1.to_json_schema(var_2)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = 1
    var_2 = module_0.Array(var_0, min_items=var_1)
    var_3 = module_1.to_json_schema(var_2)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Object(properties=var_4)
    var_6 = module_1.to_json_schema(var_5)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'anyOf'
    var_6 = var_4[var_5]
    var_7 = len(var_6)
    assert var_7 == 2

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'User'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed'
    var_1 = module_0.Const(var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = (var_3, var_1)
    var_5 = [var_2, var_4]
    var_6 = module_0.Choice(choices=var_5)
    var_7 = module_1.to_json_schema(var_6)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1.5
    var_1 = module_0.Float(maximum=var_0)
    var_2 = module_1.to_json_schema(var_1)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_if_then_else_from_json_schema_with_all_clauses. Retrieved 17/23 statements.
# Partially parsed test_if_then_else_from_json_schema_with_only_if_and_then. Retrieved 11/15 statements.
# Partially parsed test_if_then_else_from_json_schema_with_only_if_and_else. Retrieved 11/15 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'number'
    var_5 = {var_1: var_4}
    var_6 = 'boolean'
    var_7 = {var_1: var_6}
    var_8 = 'if'
    var_9 = 'then'
    var_10 = 'else'
    var_11 = 'default'
    var_12 = 123
    var_13 = {var_8: var_3, var_9: var_5, var_10: var_7, var_11: var_12}
    var_14 = module_1.if_then_else_from_json_schema(var_13, var_0)
    var_15 = 'test'
    var_16 = True

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'number'
    var_5 = {var_1: var_4}
    var_6 = 'if'
    var_7 = 'then'
    var_8 = {var_6: var_3, var_7: var_5}
    var_9 = module_1.if_then_else_from_json_schema(var_8, var_0)
    var_10 = 123

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'boolean'
    var_5 = {var_1: var_4}
    var_6 = 'if'
    var_7 = 'else'
    var_8 = {var_6: var_3, var_7: var_5}
    var_9 = module_1.if_then_else_from_json_schema(var_8, var_0)
    var_10 = 555



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_from_json_schema_with_components_schemas_dict. Retrieved 11/15 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

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
    var_9 = module_0.Definitions()
    var_10 = module_1.from_json_schema(var_8, var_9)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_from_json_schema_type_number. Retrieved 11/12 statements.
# Partially parsed test_from_json_schema_type_integer. Retrieved 9/10 statements.
# Partially parsed test_from_json_schema_type_string. Retrieved 13/14 statements.
# Partially parsed test_from_json_schema_type_boolean. Retrieved 7/8 statements.
# Partially parsed test_from_json_schema_type_array_simple. Retrieved 13/14 statements.
# Partially parsed test_from_json_schema_type_array_list_items. Retrieved 17/20 statements.
# Partially parsed test_from_json_schema_type_array_additional_items. Retrieved 13/14 statements.
# Partially parsed test_from_json_schema_type_object_properties. Retrieved 18/21 statements.


import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'minimum'
    var_1 = 'maximum'
    var_2 = 'default'
    var_3 = 5
    var_4 = 10
    var_5 = 7
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'number'
    var_8 = True
    var_9 = None
    var_10 = module_0.from_json_schema_type(var_6, var_7, var_8, var_9)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'minimum'
    var_1 = 'exclusiveMinimum'
    var_2 = 1
    var_3 = 0
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'integer'
    var_6 = False
    var_7 = None
    var_8 = module_0.from_json_schema_type(var_4, var_5, var_6, var_7)

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
    var_10 = True
    var_11 = None
    var_12 = module_0.from_json_schema_type(var_8, var_9, var_10, var_11)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'default'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'boolean'
    var_4 = False
    var_5 = None
    var_6 = module_0.from_json_schema_type(var_2, var_3, var_4, var_5)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'items'
    var_1 = 'minItems'
    var_2 = 'uniqueItems'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 1
    var_7 = True
    var_8 = {var_0: var_5, var_1: var_6, var_2: var_7}
    var_9 = 'array'
    var_10 = False
    var_11 = None
    var_12 = module_0.from_json_schema_type(var_8, var_9, var_10, var_11)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'items'
    var_1 = 'type'
    var_2 = 'integer'
    var_3 = {var_1: var_2}
    var_4 = 'string'
    var_5 = {var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = {var_0: var_6}
    var_8 = 'array'
    var_9 = False
    var_10 = None
    var_11 = module_0.from_json_schema_type(var_7, var_8, var_9, var_10)
    var_12 = var_11.items
    var_13 = len(var_12)
    assert var_13 == 2
    var_14 = var_11.items[var_9]
    var_15 = 1
    var_16 = var_11.items[var_15]

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'items'
    var_1 = 'additionalItems'
    var_2 = 'type'
    var_3 = 'integer'
    var_4 = {var_2: var_3}
    var_5 = 'string'
    var_6 = {var_2: var_5}
    var_7 = {var_0: var_4, var_1: var_6}
    var_8 = 'array'
    var_9 = False
    var_10 = None
    var_11 = module_0.from_json_schema_type(var_7, var_8, var_9, var_10)
    var_12 = var_11.additional_items

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'properties'
    var_1 = 'required'
    var_2 = 'name'
    var_3 = 'age'
    var_4 = 'type'
    var_5 = 'string'
    var_6 = {var_4: var_5}
    var_7 = 'integer'
    var_8 = {var_4: var_7}
    var_9 = {var_2: var_6, var_3: var_8}
    var_10 = [var_2]
    var_11 = {var_0: var_9, var_1: var_10}
    var_12 = 'object'
    var_13 = False
    var_14 = None
    var_15 = module_0.from_json_schema_type(var_11, var_12, var_13, var_14)
    var_16 = var_15.properties[var_2]
    var_17 = var_15.properties[var_3]



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_to_json_schema_definitions_and_ref. Retrieved 2/22 statements.
# Failed to parse test_to_json_schema_error_unsupported_type.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 'hello'
    var_2 = module_0.String()
    var_3 = module_1.to_json_schema(var_2)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = 10
    var_2 = False
    var_3 = module_0.Integer(minimum=var_0, maximum=var_1)
    var_4 = module_1.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = 1
    var_2 = module_0.Array(var_0, min_items=var_1)
    var_3 = module_1.to_json_schema(var_2)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Object(properties=var_4)
    var_6 = module_1.to_json_schema(var_5)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'anyOf'
    var_6 = var_4[var_5]
    var_7 = len(var_6)
    assert var_7 == 2

def test_case_0():
    var_0 = 'MySchema'
    var_1 = 'Test'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed_value'
    var_1 = module_0.Const(var_0)
    var_2 = module_1.to_json_schema(var_1)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_all_of_from_json_schema_valid_input. Retrieved 14/15 statements.
# Partially parsed test_all_of_from_json_schema_empty_allOf. Retrieved 7/8 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'allOf'
    var_2 = 'default'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 'integer'
    var_7 = {var_3: var_6}
    var_8 = [var_5, var_7]
    var_9 = 123
    var_10 = {var_1: var_8, var_2: var_9}
    var_11 = module_1.all_of_from_json_schema(var_10, var_0)
    var_12 = var_11.all_of
    var_13 = len(var_12)
    assert var_13 == 2

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'allOf'
    var_2 = []
    var_3 = {var_1: var_2}
    var_4 = module_1.all_of_from_json_schema(var_3, var_0)
    var_5 = var_4.all_of
    var_6 = len(var_5)
    assert var_6 == 0

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'allOf'
    var_2 = 'default'
    var_3 = 'type'
    var_4 = 'boolean'
    var_5 = {var_3: var_4}
    var_6 = [var_5]
    var_7 = True
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = module_1.all_of_from_json_schema(var_8, var_0)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_to_json_schema_string_field_evaluates_true. Retrieved 8/9 statements.


import re as module_0
import typesystem.fields as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 10
    var_3 = '^[a-z]+$'
    var_4 = module_0.compile(var_3)
    var_5 = 'email'
    var_6 = module_1.String(allow_blank=var_1, max_length=var_2, min_length=var_0, format=var_5)
    var_7 = module_2.to_json_schema(var_6)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_from_json_schema_multiple_constraints_returns_all_of. Retrieved 7/11 statements.


import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'enum'
    var_1 = 'const'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = [var_2, var_3]
    var_5 = {var_0: var_4, var_1: var_2}
    var_6 = module_0.from_json_schema(var_5)



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_from_json_schema_type_number. Retrieved 17/18 statements.
# Partially parsed test_from_json_schema_type_integer. Retrieved 11/12 statements.
# Partially parsed test_from_json_schema_type_string. Retrieved 15/16 statements.
# Partially parsed test_from_json_schema_type_boolean. Retrieved 7/8 statements.
# Partially parsed test_from_json_schema_type_array_simple. Retrieved 15/16 statements.
# Partially parsed test_from_json_schema_type_array_complex. Retrieved 21/23 statements.
# Partially parsed test_from_json_schema_type_object_simple. Retrieved 15/16 statements.


import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'minimum'
    var_1 = 'maximum'
    var_2 = 'exclusiveMinimum'
    var_3 = 'exclusiveMaximum'
    var_4 = 'multipleOf'
    var_5 = 'default'
    var_6 = 10
    var_7 = 20
    var_8 = 5
    var_9 = 25
    var_10 = 2
    var_11 = 15
    var_12 = {var_0: var_6, var_1: var_7, var_2: var_8, var_3: var_9, var_4: var_10, var_5: var_11}
    var_13 = 'number'
    var_14 = True
    var_15 = None
    var_16 = module_0.from_json_schema_type(var_12, var_13, var_14, var_15)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'minimum'
    var_1 = 'maximum'
    var_2 = 'default'
    var_3 = 1
    var_4 = 5
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'integer'
    var_8 = False
    var_9 = None
    var_10 = module_0.from_json_schema_type(var_6, var_7, var_8, var_9)

import typesystem.json_schema as module_0

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
    var_9 = 'test'
    var_10 = {var_0: var_5, var_1: var_6, var_2: var_7, var_3: var_8, var_4: var_9}
    var_11 = 'string'
    var_12 = False
    var_13 = None
    var_14 = module_0.from_json_schema_type(var_10, var_11, var_12, var_13)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'minLength'
    var_1 = 0
    var_2 = {var_0: var_1}
    var_3 = 'string'
    var_4 = True
    var_5 = None
    var_6 = module_0.from_json_schema_type(var_2, var_3, var_4, var_5)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'default'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'boolean'
    var_4 = False
    var_5 = None
    var_6 = module_0.from_json_schema_type(var_2, var_3, var_4, var_5)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'items'
    var_1 = 'minItems'
    var_2 = 'maxItems'
    var_3 = 'uniqueItems'
    var_4 = 'type'
    var_5 = 'string'
    var_6 = {var_4: var_5}
    var_7 = 1
    var_8 = 5
    var_9 = True
    var_10 = {var_0: var_6, var_1: var_7, var_2: var_8, var_3: var_9}
    var_11 = 'array'
    var_12 = False
    var_13 = None
    var_14 = module_0.from_json_schema_type(var_10, var_11, var_12, var_13)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'items'
    var_1 = 'additionalItems'
    var_2 = 'minItems'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 'integer'
    var_7 = {var_3: var_6}
    var_8 = [var_5, var_7]
    var_9 = 'boolean'
    var_10 = {var_3: var_9}
    var_11 = 2
    var_12 = {var_0: var_8, var_1: var_10, var_2: var_11}
    var_13 = 'array'
    var_14 = False
    var_15 = None
    var_16 = module_0.from_json_schema_type(var_12, var_13, var_14, var_15)
    var_17 = var_16.items
    var_18 = var_16.items
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = var_16.additional_items

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'properties'
    var_1 = 'required'
    var_2 = 'minProperties'
    var_3 = 'name'
    var_4 = 'type'
    var_5 = 'string'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = [var_3]
    var_9 = 1
    var_10 = {var_0: var_7, var_1: var_8, var_2: var_9}
    var_11 = 'object'
    var_12 = False
    var_13 = None
    var_14 = module_0.from_json_schema_type(var_10, var_11, var_12, var_13)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'properties'
    var_1 = 'patternProperties'
    var_2 = 'additionalProperties'
    var_3 = 'propertyNames'
    var_4 = 'id'
    var_5 = 'type'
    var_6 = 'integer'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = '^prefix_'
    var_10 = 'string'
    var_11 = {var_5: var_10}
    var_12 = {var_9: var_11}
    var_13 = False
    var_14 = 'minLength'
    var_15 = 3
    var_16 = {var_14: var_15}
    var_17 = {var_0: var_8, var_1: var_12, var_2: var_13, var_3: var_16}
    var_18 = 'object'
    var_19 = True
    var_20 = None
    var_21 = module_0.from_json_schema_type(var_17, var_18, var_19, var_20)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_to_json_schema_definitions. Retrieved 3/7 statements.
# Failed to parse test_to_json_schema_error_unsupported_type.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = True
    var_3 = module_0.String(max_length=var_1, min_length=var_0)
    var_4 = module_1.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = 100
    var_2 = 10
    var_3 = module_0.Integer(minimum=var_0, maximum=var_1)
    var_4 = module_1.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = 1
    var_2 = True
    var_3 = module_0.Array(var_0, min_items=var_1, unique_items=var_2)
    var_4 = module_1.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_0]
    var_6 = module_0.Object(properties=var_4, required=var_5)
    var_7 = module_1.to_json_schema(var_6)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = module_1.to_json_schema(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'User'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed'
    var_1 = module_0.Const()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = (var_3, var_1)
    var_5 = [var_2, var_4]
    var_6 = module_0.Choice(choices=var_5)
    var_7 = module_1.to_json_schema(var_6)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_to_json_schema_definitions. Retrieved 5/8 statements.
# Partially parsed test_to_json_schema_root_with_definitions. Retrieved 7/9 statements.
# Failed to parse test_to_json_schema_error_on_unsupported_type.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = True
    var_3 = module_0.String(max_length=var_1, min_length=var_0)
    var_4 = module_1.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = 100
    var_2 = 10
    var_3 = module_0.Integer(minimum=var_0, maximum=var_1)
    var_4 = module_1.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = 1
    var_2 = 5
    var_3 = True
    var_4 = module_0.Array(var_0, min_items=var_1, max_items=var_2, unique_items=var_3)
    var_5 = module_1.to_json_schema(var_4)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'Alpha'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'Beta'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = module_0.Choice(choices=var_6)
    var_8 = module_1.to_json_schema(var_7)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed'
    var_1 = module_0.Const()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = module_1.to_json_schema(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'MyString'
    var_1 = 'MyInt'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'Shared'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.String()
    var_4 = module_0.String()
    var_5 = {var_0: var_4}
    var_6 = module_1.to_json_schema(var_3, var_5)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 42
    var_1 = module_0.Integer()
    var_2 = module_1.get_standard_properties(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_1.get_standard_properties(var_0)



# Parsed testcases at query #4
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'email'
    var_1 = False
    var_2 = True
    var_3 = module_0.String(allow_blank=var_2, format=var_0)
    var_4 = module_1.to_json_schema(var_3)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_to_json_schema_object_predicate_evaluates_to_true. Retrieved 7/8 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = {}
    var_3 = None
    var_4 = []
    var_5 = module_0.Object(properties=var_1, pattern_properties=var_2, additional_properties=var_3, property_names=var_3, min_properties=var_3, max_properties=var_3, required=var_4)
    var_6 = module_1.to_json_schema(var_5)



# Parsed testcases at query #6
#--------------------------




import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = []
    var_3 = module_0.Schema(var_1)
    var_4 = module_1.to_json_schema(var_3)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_if_then_else_from_json_schema_full_clauses. Retrieved 16/26 statements.
# Partially parsed test_if_then_else_from_json_schema_no_else. Retrieved 10/16 statements.
# Partially parsed test_if_then_else_from_json_schema_no_then. Retrieved 10/16 statements.


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
    var_7 = 'const'
    var_8 = 'yes'
    var_9 = {var_7: var_8}
    var_10 = 'no'
    var_11 = {var_7: var_10}
    var_12 = 'maybe'
    var_13 = {var_0: var_6, var_1: var_9, var_2: var_11, var_3: var_12}
    var_14 = module_0.Definitions()
    var_15 = module_1.if_then_else_from_json_schema(var_13, var_14)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'if'
    var_1 = 'then'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = {var_2: var_3}
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = module_0.Definitions()
    var_8 = module_1.if_then_else_from_json_schema(var_6, var_7)
    var_9 = None

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'if'
    var_1 = 'else'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = {var_2: var_3}
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = module_0.Definitions()
    var_8 = module_1.if_then_else_from_json_schema(var_6, var_7)
    var_9 = 'anything'



# Parsed testcases at query #8
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Object()
    var_2 = module_1.to_json_schema(var_1)



# Parsed testcases at query #9
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = True
    var_1 = module_0.String()
    var_2 = None
    var_3 = module_1.IfThenElse(var_1, var_2, var_2)
    var_4 = module_2.to_json_schema(var_3)



# Parsed testcases at query #10
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = module_0.Object(min_properties=var_0)
    var_2 = module_1.to_json_schema(var_1)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_from_json_schema_bool_true. Retrieved 3/4 statements.
# Partially parsed test_from_json_schema_bool_false. Retrieved 3/4 statements.
# Partially parsed test_from_json_schema_simple_string. Retrieved 7/8 statements.
# Partially parsed test_from_json_schema_simple_integer. Retrieved 7/8 statements.
# Partially parsed test_from_json_schema_enum. Retrieved 9/10 statements.
# Partially parsed test_from_json_schema_const. Retrieved 5/6 statements.
# Partially parsed test_from_json_schema_all_of. Retrieved 13/14 statements.
# Partially parsed test_from_json_schema_any_of. Retrieved 10/11 statements.
# Partially parsed test_from_json_schema_ref. Retrieved 5/6 statements.
# Partially parsed test_from_json_schema_with_components. Retrieved 18/19 statements.
# Partially parsed test_from_json_schema_not. Retrieved 7/8 statements.
# Partially parsed test_from_json_schema_if_then_else. Retrieved 13/14 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = True
    var_2 = module_1.from_json_schema(var_1, var_0)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = False
    var_2 = module_1.from_json_schema(var_1, var_0)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'type'
    var_2 = 'minLength'
    var_3 = 'string'
    var_4 = 5
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_1.from_json_schema(var_5, var_0)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'type'
    var_2 = 'minimum'
    var_3 = 'integer'
    var_4 = 10
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_1.from_json_schema(var_5, var_0)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'type'
    var_2 = 'enum'
    var_3 = 'string'
    var_4 = 'a'
    var_5 = 'b'
    var_6 = [var_4, var_5]
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = module_1.from_json_schema(var_7, var_0)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'const'
    var_2 = 123
    var_3 = {var_1: var_2}
    var_4 = module_1.from_json_schema(var_3, var_0)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'allOf'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = 'const'
    var_6 = 'foo'
    var_7 = {var_5: var_6}
    var_8 = [var_4, var_7]
    var_9 = {var_1: var_8}
    var_10 = module_1.from_json_schema(var_9, var_0)
    var_11 = var_10.all_of
    var_12 = len(var_11)
    assert var_12 == 2

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'anyOf'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = 'integer'
    var_6 = {var_2: var_5}
    var_7 = [var_4, var_6]
    var_8 = {var_1: var_7}
    var_9 = module_1.from_json_schema(var_8, var_0)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = '#/components/schemas/MyType'
    var_3 = {var_1: var_2}
    var_4 = module_1.from_json_schema(var_3, var_0)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

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
    var_16 = module_0.Definitions()
    var_17 = module_1.from_json_schema(var_15, var_16)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'not'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = module_1.from_json_schema(var_5, var_0)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
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
    var_12 = module_1.from_json_schema(var_11, var_0)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_to_json_schema_definitions_and_reference. Retrieved 4/16 statements.
# Failed to parse test_to_json_schema_error_unsupported_type.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = module_0.String(max_length=var_1, min_length=var_0)
    var_3 = module_1.to_json_schema(var_2)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.String()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = 100
    var_2 = 5
    var_3 = module_0.Integer(minimum=var_0, maximum=var_1, exclusive_minimum=var_2)
    var_4 = module_1.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = 1
    var_2 = True
    var_3 = module_0.Array(var_0, min_items=var_1, unique_items=var_2)
    var_4 = module_1.to_json_schema(var_3)

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
    var_7 = module_0.Choice(choices=var_6)
    var_8 = module_1.to_json_schema(var_7)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'anyOf'
    var_6 = var_4[var_5]
    var_7 = len(var_6)
    assert var_7 == 2

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'User'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.String()

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = module_0.Integer()
    var_2 = module_1.get_standard_properties(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'static_value'
    var_1 = module_0.Const(var_0)
    var_2 = module_1.to_json_schema(var_1)



# Parsed testcases at query #13
#--------------------------

# Failed to parse test_to_json_schema_evaluates_true_for_any_type.




# Parsed testcases at query #14
#--------------------------

# Partially parsed test_to_json_schema_array_items_as_tuple. Retrieved 7/8 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = (var_0, var_1)
    var_3 = module_0.Array(var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'items'
    var_6 = var_4[var_5]



# Parsed testcases at query #15
#--------------------------




import re as module_0
import typesystem.fields as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = module_0.compile(var_0)
    var_2 = module_1.String()
    var_3 = module_2.to_json_schema(var_2)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_to_json_schema_definitions_and_reference. Retrieved 2/10 statements.
# Partially parsed test_to_json_schema_choice. Retrieved 6/9 statements.
# Failed to parse test_to_json_schema_error_on_unsupported.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = module_0.String(max_length=var_1, min_length=var_0)
    var_3 = module_1.to_json_schema(var_2)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.String()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = 100
    var_2 = 5
    var_3 = module_0.Integer(minimum=var_0, maximum=var_1, exclusive_minimum=var_2)
    var_4 = module_1.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Array(var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Object(properties=var_4)
    var_6 = module_1.to_json_schema(var_5)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'anyOf'
    var_6 = var_4[var_5]
    var_7 = len(var_6)
    assert var_7 == 2

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'User'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed'
    var_1 = module_0.Const()
    var_2 = module_1.to_json_schema(var_1)

def test_case_0():
    var_0 = 'A'
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'B'
    var_4 = (var_3, var_1)
    var_5 = [var_2, var_4]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_one_of_from_json_schema. Retrieved 14/15 statements.
# Partially parsed test_one_of_from_json_schema_with_empty_list. Retrieved 7/8 statements.
# Partially parsed test_one_of_from_json_schema_with_complex_items. Retrieved 18/21 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'oneOf'
    var_2 = 'default'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 'number'
    var_7 = {var_3: var_6}
    var_8 = [var_5, var_7]
    var_9 = 'some_value'
    var_10 = {var_1: var_8, var_2: var_9}
    var_11 = module_1.one_of_from_json_schema(var_10, var_0)
    var_12 = var_11.one_of
    var_13 = len(var_12)
    assert var_13 == 2

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'oneOf'
    var_2 = []
    var_3 = {var_1: var_2}
    var_4 = module_1.one_of_from_json_schema(var_3, var_0)
    var_5 = var_4.one_of
    var_6 = len(var_5)
    assert var_6 == 0

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'oneOf'
    var_2 = 'type'
    var_3 = 'enum'
    var_4 = 'integer'
    var_5 = 1
    var_6 = 2
    var_7 = [var_5, var_6]
    var_8 = {var_2: var_4, var_3: var_7}
    var_9 = 'const'
    var_10 = 'fixed'
    var_11 = {var_9: var_10}
    var_12 = [var_8, var_11]
    var_13 = {var_1: var_12}
    var_14 = module_1.one_of_from_json_schema(var_13, var_0)
    var_15 = 0
    var_16 = var_14.one_of[var_15]
    var_17 = var_14.one_of[var_5]



# Parsed testcases at query #18
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = True
    var_3 = None
    var_4 = 5
    var_5 = []
    var_6 = module_0.Object(properties=var_0, pattern_properties=var_1, additional_properties=var_2, property_names=var_3, min_properties=var_3, max_properties=var_4, required=var_5)
    var_7 = module_1.to_json_schema(var_6)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_ref_from_json_schema_valid. Retrieved 10/13 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'MockField'
    var_2 = ()
    var_3 = 'validate'
    var_4 = lambda x: x
    var_5 = {var_3: var_4}
    var_6 = '$ref'
    var_7 = '#/user'
    var_8 = {var_6: var_7}
    var_9 = module_1.ref_from_json_schema(var_8, var_0)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = 'user'
    var_3 = {var_1: var_2}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'not_a_ref'
    var_2 = '#/user'
    var_3 = {var_1: var_2}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)



# Parsed testcases at query #20
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 5
    var_2 = 10
    var_3 = module_0.String(max_length=var_2, min_length=var_1)
    var_4 = module_1.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = 0
    var_2 = module_0.Integer(minimum=var_1)
    var_3 = module_1.to_json_schema(var_2)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = 1
    var_2 = module_0.Array(var_0, min_items=var_1)
    var_3 = module_1.to_json_schema(var_2)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Object(properties=var_4)
    var_6 = module_1.to_json_schema(var_5)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'anyOf'
    var_6 = var_4[var_5]
    var_7 = len(var_6)
    assert var_7 == 2

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.Object(properties=var_2)
    var_4 = module_1.Definitions()
    var_5 = module_2.to_json_schema(var_4)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed'
    var_1 = module_0.Const()
    var_2 = module_1.to_json_schema(var_1)

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
    var_7 = module_0.Choice(choices=var_6)
    var_8 = module_1.to_json_schema(var_7)



# Parsed testcases at query #21
#--------------------------




import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = []
    var_3 = module_0.Schema(var_1)
    var_4 = module_1.to_json_schema(var_3)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_to_json_schema_definitions. Retrieved 6/8 statements.
# Partially parsed test_to_json_schema_regex_error. Retrieved 1/7 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1
import typesystem.composites as module_2

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True
    var_2 = module_2.NeverMatch()
    var_3 = module_1.to_json_schema(var_2)
    assert var_3 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = True
    var_3 = module_0.String(max_length=var_1, min_length=var_0)
    var_4 = module_1.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = 100
    var_2 = 10
    var_3 = module_0.Integer(minimum=var_0, maximum=var_1)
    var_4 = module_1.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0.5
    var_1 = module_0.Float(exclusive_minimum=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = 1
    var_2 = True
    var_3 = module_0.Array(var_0, min_items=var_1, unique_items=var_2)
    var_4 = module_1.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_0]
    var_6 = module_0.Object(properties=var_4, required=var_5)
    var_7 = module_1.to_json_schema(var_6)

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
    var_7 = module_0.Choice(choices=var_6)
    var_8 = module_1.to_json_schema(var_7)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed'
    var_1 = module_0.Const(var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'anyOf'
    var_6 = var_4[var_5]
    var_7 = len(var_6)
    assert var_7 == 2

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'User'
    var_1 = 'id'
    var_2 = module_0.Integer()
    var_3 = {var_1: var_2}
    var_4 = module_0.Object(properties=var_3)
    var_5 = {var_0: var_4}

def test_case_0():
    var_0 = 'abc'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_to_json_schema_definitions_and_reference. Retrieved 5/10 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = module_0.String(max_length=var_1, min_length=var_0)
    var_3 = module_1.to_json_schema(var_2)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.String()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = 100
    var_2 = 5
    var_3 = module_0.Integer(minimum=var_0, maximum=var_1, exclusive_minimum=var_2)
    var_4 = module_1.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.Boolean()
    var_3 = module_1.to_json_schema(var_2)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = 1
    var_2 = 5
    var_3 = module_0.Array(var_0, min_items=var_1, max_items=var_2)
    var_4 = module_1.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_0]
    var_6 = module_0.Object(properties=var_4, required=var_5)
    var_7 = module_1.to_json_schema(var_6)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'anyOf'
    var_6 = var_4[var_5]
    var_7 = len(var_6)
    assert var_7 == 2

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'User'
    var_2 = {var_1: var_0}
    var_3 = module_1.Reference(var_1)
    var_4 = {}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed_value'
    var_1 = module_0.Const(var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'A'
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'B'
    var_4 = (var_3, var_1)
    var_5 = [var_2, var_4]
    var_6 = module_0.Choice(choices=var_5)
    var_7 = module_1.to_json_schema(var_6)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.to_json_schema(var_0)



