####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 5/7 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.String()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = module_0.String(min_length=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = module_0.String(max_length=var_0)
    var_2 = module_1.to_json_schema(var_1)

import re as module_0
import typesystem.fields as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = module_0.compile(var_0)
    var_2 = module_1.String()
    var_3 = module_2.to_json_schema(var_2)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'email'
    var_1 = module_0.String(format=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Integer()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Float()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Decimal()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Array()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Array()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 2
    var_1 = module_0.Array(min_items=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = module_0.Array(max_items=var_0)
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
    var_0 = module_0.Boolean()
    var_1 = module_0.Array(additional_items=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Array(unique_items=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Object()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Object()
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
    var_0 = '^S_'
    var_1 = '^I_'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Object(pattern_properties=var_4)
    var_6 = module_1.to_json_schema(var_5)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = module_0.Object(additional_properties=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Object(property_names=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = module_0.Object(max_properties=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 2
    var_1 = module_0.Object(min_properties=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = [var_0]
    var_2 = module_0.Object(required=var_1)
    var_3 = module_1.to_json_schema(var_2)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_1.Schema(var_4)
    var_6 = module_2.to_json_schema(var_5)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = True
    var_1 = 'name'
    var_2 = module_0.String()
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = module_2.to_json_schema(var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = [var_0]
    var_4 = module_1.Schema(var_2)
    var_5 = module_2.to_json_schema(var_4)

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
    var_7 = module_0.Choice(choices=var_6)
    var_8 = module_1.to_json_schema(var_7)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed_value'
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

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = module_2.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = module_2.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = module_0.Boolean()
    var_3 = module_1.IfThenElse(var_0, var_1, var_2)
    var_4 = module_2.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Not(var_0)
    var_2 = module_2.to_json_schema(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'string_field'
    var_1 = 'integer_field'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'string_ref'
    var_2 = module_1.Reference(var_1)
    var_3 = module_2.to_json_schema(var_2)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_from_json_schema_type_number. Retrieved 12/13 statements.
# Partially parsed test_from_json_schema_type_integer. Retrieved 12/13 statements.
# Partially parsed test_from_json_schema_type_string. Retrieved 12/13 statements.
# Partially parsed test_from_json_schema_type_boolean. Retrieved 8/9 statements.
# Partially parsed test_from_json_schema_type_array. Retrieved 18/20 statements.
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
    var_9 = module_0.Definitions()
    var_10 = False
    var_11 = module_1.from_json_schema_type(var_8, var_4, var_10, var_9)

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
    var_9 = module_0.Definitions()
    var_10 = False
    var_11 = module_1.from_json_schema_type(var_8, var_4, var_10, var_9)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'minLength'
    var_2 = 'maxLength'
    var_3 = 'default'
    var_4 = 'string'
    var_5 = 5
    var_6 = 10
    var_7 = 'hello'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.Definitions()
    var_10 = False
    var_11 = module_1.from_json_schema_type(var_8, var_4, var_10, var_9)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'default'
    var_2 = 'boolean'
    var_3 = True
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Definitions()
    var_6 = False
    var_7 = module_1.from_json_schema_type(var_4, var_2, var_6, var_5)

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
    var_14 = module_0.Definitions()
    var_15 = False
    var_16 = module_1.from_json_schema_type(var_13, var_5, var_15, var_14)
    var_17 = var_16.items

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
    var_13 = module_0.Definitions()
    var_14 = False
    var_15 = module_1.from_json_schema_type(var_12, var_4, var_14, var_13)
    var_16 = var_15.properties[var_5]



# Parsed testcases at query #3
#--------------------------




import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Schema()
    var_2 = module_1.to_json_schema(var_1)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_if_then_else_from_json_schema_with_all_clauses. Retrieved 16/20 statements.
# Partially parsed test_if_then_else_from_json_schema_without_then_and_else. Retrieved 10/14 statements.
# Partially parsed test_if_then_else_from_json_schema_without_else. Retrieved 13/17 statements.
# Partially parsed test_if_then_else_from_json_schema_without_then. Retrieved 13/17 statements.


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
    var_11 = module_0.Definitions()
    var_12 = module_1.if_then_else_from_json_schema(var_10, var_11)
    var_13 = var_12.if_clause
    var_14 = var_12.then_clause
    var_15 = var_12.else_clause

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'if'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.Definitions()
    var_6 = module_1.if_then_else_from_json_schema(var_4, var_5)
    var_7 = var_6.if_clause
    var_8 = var_6.then_clause
    var_9 = var_6.else_clause

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'if'
    var_1 = 'then'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = 'number'
    var_6 = {var_2: var_5}
    var_7 = {var_0: var_4, var_1: var_6}
    var_8 = module_0.Definitions()
    var_9 = module_1.if_then_else_from_json_schema(var_7, var_8)
    var_10 = var_9.if_clause
    var_11 = var_9.then_clause
    var_12 = var_9.else_clause

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'if'
    var_1 = 'else'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = 'boolean'
    var_6 = {var_2: var_5}
    var_7 = {var_0: var_4, var_1: var_6}
    var_8 = module_0.Definitions()
    var_9 = module_1.if_then_else_from_json_schema(var_7, var_8)
    var_10 = var_9.if_clause
    var_11 = var_9.then_clause
    var_12 = var_9.else_clause

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
    var_11 = 'default_value'
    var_12 = {var_0: var_6, var_1: var_8, var_2: var_10, var_3: var_11}
    var_13 = module_0.Definitions()
    var_14 = module_1.if_then_else_from_json_schema(var_12, var_13)



# Parsed testcases at query #5
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = module_0.Integer(multiple_of=var_0)
    var_2 = module_1.to_json_schema(var_1)



# Parsed testcases at query #6
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.String()
    var_2 = module_1.to_json_schema(var_1)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_field_items_is_list_or_tuple. Retrieved 5/7 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Array(var_2)
    var_4 = var_3.items



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_one_of_from_json_schema. Retrieved 16/19 statements.
# Partially parsed test_one_of_from_json_schema_with_default. Retrieved 10/11 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'oneOf'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = 'number'
    var_6 = {var_2: var_5}
    var_7 = [var_4, var_6]
    var_8 = {var_1: var_7}
    var_9 = module_1.one_of_from_json_schema(var_8, var_0)
    var_10 = var_9.one_of
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = 0
    var_13 = var_9.one_of[var_12]
    var_14 = 1
    var_15 = var_9.one_of[var_14]

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'oneOf'
    var_2 = 'default'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = [var_5]
    var_7 = 'test'
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = module_1.one_of_from_json_schema(var_8, var_0)



# Parsed testcases at query #9
#--------------------------




import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = 'object'
    var_2 = False
    var_3 = module_0.Definitions()
    var_4 = module_1.from_json_schema_type(var_0, var_1, var_2, var_3)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_from_json_schema_with_bool_true. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_bool_false. Retrieved 4/6 statements.
# Partially parsed test_from_json_schema_with_ref. Retrieved 7/9 statements.
# Partially parsed test_from_json_schema_with_type_constraint. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_with_enum. Retrieved 10/12 statements.
# Partially parsed test_from_json_schema_with_const. Retrieved 7/9 statements.
# Partially parsed test_from_json_schema_with_all_of. Retrieved 14/16 statements.
# Partially parsed test_from_json_schema_with_any_of. Retrieved 15/17 statements.
# Partially parsed test_from_json_schema_with_one_of. Retrieved 13/15 statements.
# Partially parsed test_from_json_schema_with_not. Retrieved 10/12 statements.
# Partially parsed test_from_json_schema_with_if_then_else. Retrieved 19/21 statements.
# Partially parsed test_from_json_schema_with_multiple_constraints. Retrieved 12/14 statements.
# Partially parsed test_from_json_schema_with_no_constraints. Retrieved 4/5 statements.


import typesystem.json_schema as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.from_json_schema(var_0)
    var_2 = None
    var_3 = var_1.validate(var_2)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = '$ref'
    var_1 = '#/components/schemas/test'
    var_2 = {var_0: var_1}
    var_3 = module_0.Definitions()
    var_4 = module_1.from_json_schema(var_2, var_3)
    var_5 = 'test'
    var_6 = var_4.validate(var_5)
    assert var_6 == 'test'

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)
    var_4 = 'test'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'test'

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'enum'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = module_0.from_json_schema(var_5)
    var_7 = var_6.validate(var_1)
    assert var_7 == 'a'
    var_8 = 'd'
    var_9 = var_6.validate(var_8)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'const'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)
    var_4 = var_3.validate(var_1)
    assert var_4 == 'test'
    var_5 = 'other'
    var_6 = var_3.validate(var_5)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'allOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'minLength'
    var_5 = 5
    var_6 = {var_1: var_2, var_4: var_5}
    var_7 = [var_3, var_6]
    var_8 = {var_0: var_7}
    var_9 = module_0.from_json_schema(var_8)
    var_10 = 'test'
    var_11 = var_9.validate(var_10)
    assert var_11 == 'test'
    var_12 = 'a'
    var_13 = var_9.validate(var_12)

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
    var_9 = 'test'
    var_10 = var_8.validate(var_9)
    assert var_10 == 'test'
    var_11 = 1
    var_12 = var_8.validate(var_11)
    assert var_12 == 1
    var_13 = None
    var_14 = var_8.validate(var_13)

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
    var_9 = 'test'
    var_10 = var_8.validate(var_9)
    assert var_10 == 'test'
    var_11 = 1
    var_12 = var_8.validate(var_11)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'not'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.from_json_schema(var_4)
    var_6 = 1
    var_7 = var_5.validate(var_6)
    assert var_7 == 1
    var_8 = 'test'
    var_9 = var_5.validate(var_8)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'if'
    var_1 = 'then'
    var_2 = 'else'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 'minLength'
    var_7 = 5
    var_8 = {var_3: var_4, var_6: var_7}
    var_9 = 'number'
    var_10 = {var_3: var_9}
    var_11 = {var_0: var_5, var_1: var_8, var_2: var_10}
    var_12 = module_0.from_json_schema(var_11)
    var_13 = 'test'
    var_14 = var_12.validate(var_13)
    assert var_14 == 'test'
    var_15 = 1
    var_16 = var_12.validate(var_15)
    assert var_16 == 1
    var_17 = 'a'
    var_18 = var_12.validate(var_17)

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
    var_9 = var_8.validate(var_3)
    assert var_9 == 'a'
    var_10 = 'd'
    var_11 = var_8.validate(var_10)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.from_json_schema(var_0)
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None



# Parsed testcases at query #11
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Array()
    var_2 = module_1.to_json_schema(var_1)



# Parsed testcases at query #12
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = None
    var_3 = module_1.IfThenElse(var_0, var_1, var_2)
    var_4 = module_2.to_json_schema(var_3)



# Parsed testcases at query #13
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
    var_7 = module_0.Definitions()
    var_8 = False
    var_9 = module_1.from_json_schema_type(var_6, var_3, var_8, var_7)

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
    var_7 = module_0.Definitions()
    var_8 = False
    var_9 = module_1.from_json_schema_type(var_6, var_3, var_8, var_7)

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
    var_8 = False
    var_9 = module_1.from_json_schema_type(var_6, var_3, var_8, var_7)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'boolean'
    var_2 = {var_0: var_1}
    var_3 = module_0.Definitions()
    var_4 = False
    var_5 = module_1.from_json_schema_type(var_2, var_1, var_4, var_3)

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
    var_7 = module_0.Definitions()
    var_8 = False
    var_9 = module_1.from_json_schema_type(var_6, var_3, var_8, var_7)

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
    var_7 = module_0.Definitions()
    var_8 = False
    var_9 = module_1.from_json_schema_type(var_6, var_3, var_8, var_7)



# Parsed testcases at query #14
#--------------------------




import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = 'object'
    var_2 = False
    var_3 = module_0.Definitions()
    var_4 = module_1.from_json_schema_type(var_0, var_1, var_2, var_3)



# Parsed testcases at query #15
#--------------------------




import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'array'
    var_2 = {var_0: var_1}
    var_3 = module_0.Definitions()
    var_4 = False
    var_5 = module_1.from_json_schema_type(var_2, var_1, var_4, var_3)



# Parsed testcases at query #16
#--------------------------




import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'propertyNames'
    var_2 = 'object'
    var_3 = 'string'
    var_4 = {var_0: var_3}
    var_5 = {var_0: var_2, var_1: var_4}
    var_6 = {}
    var_7 = False
    var_8 = module_0.from_json_schema_type(var_5, var_2, var_7, var_6)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_from_json_schema_type_with_list_items. Retrieved 14/16 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'items'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = {var_0: var_4}
    var_6 = module_0.Definitions()
    var_7 = 'array'
    var_8 = False
    var_9 = module_1.from_json_schema_type(var_5, var_7, var_8, var_6)
    var_10 = var_9.items
    var_11 = var_9.items
    var_12 = len(var_11)
    assert var_12 == 1
    var_13 = var_9.items[var_8]



# Parsed testcases at query #18
#--------------------------




import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is False



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 2/4 statements.
# Failed to parse test_to_json_schema_with_unknown_field_type.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is False

import re as module_0
import typesystem.fields as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = '^[a-z]+$'
    var_3 = module_0.compile(var_2)
    var_4 = 'email'
    var_5 = module_1.String(max_length=var_1, min_length=var_0, format=var_4)
    var_6 = module_2.to_json_schema(var_5)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = 100
    var_2 = True
    var_3 = 2
    var_4 = module_0.Integer(minimum=var_0, maximum=var_1, exclusive_minimum=var_2, exclusive_maximum=var_2, multiple_of=var_3)
    var_5 = module_1.to_json_schema(var_4)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0.0
    var_1 = 100.0
    var_2 = True
    var_3 = 0.5
    var_4 = module_0.Float(minimum=var_0, maximum=var_1, exclusive_minimum=var_2, exclusive_maximum=var_2, multiple_of=var_3)
    var_5 = module_1.to_json_schema(var_4)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = module_0.String()
    var_3 = False
    var_4 = True
    var_5 = module_0.Array(var_2, var_3, var_0, var_1, unique_items=var_4)
    var_6 = module_1.to_json_schema(var_5)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = '^S_'
    var_4 = module_0.String()
    var_5 = {var_3: var_4}
    var_6 = False
    var_7 = module_0.String()
    var_8 = 10
    var_9 = 1
    var_10 = [var_0]
    var_11 = module_0.Object(properties=var_2, pattern_properties=var_5, additional_properties=var_6, property_names=var_7, min_properties=var_9, max_properties=var_8, required=var_10)
    var_12 = module_1.to_json_schema(var_11)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = [var_0]
    var_4 = module_1.Schema(var_2)
    var_5 = module_2.to_json_schema(var_4)

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
    var_0 = 'constant_value'
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

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = module_2.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = module_2.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = module_0.Boolean()
    var_3 = module_1.IfThenElse(var_0, var_1, var_2)
    var_4 = module_2.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Not(var_0)
    var_2 = module_2.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.String()
    var_2 = module_1.Reference(var_0)
    var_3 = module_2.to_json_schema(var_2)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.String()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = module_0.String()
    var_2 = module_1.to_json_schema(var_1)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 2/3 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is False

import re as module_0
import typesystem.fields as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = '^[a-z]+$'
    var_3 = module_0.compile(var_2)
    var_4 = 'email'
    var_5 = module_1.String(max_length=var_1, min_length=var_0, format=var_4)
    var_6 = module_2.to_json_schema(var_5)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = 100
    var_2 = True
    var_3 = 2
    var_4 = module_0.Integer(minimum=var_0, maximum=var_1, exclusive_minimum=var_2, exclusive_maximum=var_2, multiple_of=var_3)
    var_5 = module_1.to_json_schema(var_4)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0.0
    var_1 = 100.0
    var_2 = True
    var_3 = 0.5
    var_4 = module_0.Float(minimum=var_0, maximum=var_1, exclusive_minimum=var_2, exclusive_maximum=var_2, multiple_of=var_3)
    var_5 = module_1.to_json_schema(var_4)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = module_0.String()
    var_3 = False
    var_4 = True
    var_5 = module_0.Array(var_2, var_3, var_0, var_1, unique_items=var_4)
    var_6 = module_1.to_json_schema(var_5)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = '^S_'
    var_4 = module_0.String()
    var_5 = {var_3: var_4}
    var_6 = False
    var_7 = module_0.String()
    var_8 = 10
    var_9 = 1
    var_10 = [var_0]
    var_11 = module_0.Object(properties=var_2, pattern_properties=var_5, additional_properties=var_6, property_names=var_7, min_properties=var_9, max_properties=var_8, required=var_10)
    var_12 = module_1.to_json_schema(var_11)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = [var_0]
    var_4 = module_1.Schema(var_2)
    var_5 = module_2.to_json_schema(var_4)

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
    var_7 = module_0.Choice(choices=var_6)
    var_8 = module_1.to_json_schema(var_7)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed_value'
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

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = module_2.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = module_2.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = module_0.Boolean()
    var_3 = module_1.IfThenElse(var_0, var_1, var_2)
    var_4 = module_2.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Not(var_0)
    var_2 = module_2.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'test'
    var_2 = module_1.Reference(var_1)
    var_3 = module_2.to_json_schema(var_2)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.String()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = module_0.String()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'invalid_field_type'
    var_1 = module_0.to_json_schema(var_0)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_definitions_iteration. Retrieved 7/8 statements.


import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0.to_json_schema(var_4, var_5)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_ref_from_json_schema_with_valid_reference. Retrieved 6/9 statements.
# Partially parsed test_ref_from_json_schema_with_invalid_reference. Retrieved 6/9 statements.


def test_case_0():
    var_0 = '$ref'
    var_1 = '#/definitions/valid'
    var_2 = {var_0: var_1}
    var_3 = 'valid'
    var_4 = 'dummy'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = '$ref'
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = 'valid'
    var_4 = 'dummy'
    var_5 = {var_3: var_4}



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_type_from_json_schema_with_single_type. Retrieved 5/6 statements.
# Partially parsed test_type_from_json_schema_with_nullable_single_type. Retrieved 7/8 statements.
# Partially parsed test_type_from_json_schema_with_multiple_types. Retrieved 13/16 statements.
# Partially parsed test_type_from_json_schema_with_nullable_multiple_types. Retrieved 14/17 statements.
# Partially parsed test_type_from_json_schema_with_no_type. Retrieved 3/4 statements.
# Partially parsed test_type_from_json_schema_with_nullable_no_type. Retrieved 6/7 statements.
# Partially parsed test_type_from_json_schema_with_object_type. Retrieved 11/13 statements.
# Partially parsed test_type_from_json_schema_with_array_type. Retrieved 9/11 statements.
# Partially parsed test_type_from_json_schema_with_integer_type. Retrieved 5/6 statements.
# Partially parsed test_type_from_json_schema_with_boolean_type. Retrieved 5/6 statements.
# Partially parsed test_type_from_json_schema_with_number_type. Retrieved 5/6 statements.
# Partially parsed test_type_from_json_schema_with_pattern_properties. Retrieved 11/13 statements.
# Partially parsed test_type_from_json_schema_with_additional_properties. Retrieved 9/11 statements.
# Partially parsed test_type_from_json_schema_with_property_names. Retrieved 9/11 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = module_0.Definitions()
    var_4 = module_1.type_from_json_schema(var_2, var_3)

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
    var_1 = 'string'
    var_2 = 'number'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.Definitions()
    var_6 = module_1.type_from_json_schema(var_4, var_5)
    var_7 = var_6.any_of
    var_8 = len(var_7)
    assert var_8 == 2
    var_9 = 0
    var_10 = var_6.any_of[var_9]
    var_11 = 1
    var_12 = var_6.any_of[var_11]

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = 'number'
    var_3 = 'null'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = module_0.Definitions()
    var_7 = module_1.type_from_json_schema(var_5, var_6)
    var_8 = var_7.any_of
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = 0
    var_11 = var_7.any_of[var_10]
    var_12 = 1
    var_13 = var_7.any_of[var_12]

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Definitions()
    var_2 = module_1.type_from_json_schema(var_0, var_1)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'null'
    var_2 = [var_1]
    var_3 = {var_0: var_2}
    var_4 = module_0.Definitions()
    var_5 = module_1.type_from_json_schema(var_3, var_4)

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
    var_8 = module_0.Definitions()
    var_9 = module_1.type_from_json_schema(var_7, var_8)
    var_10 = var_9.properties[var_3]

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'items'
    var_2 = 'array'
    var_3 = 'string'
    var_4 = {var_0: var_3}
    var_5 = {var_0: var_2, var_1: var_4}
    var_6 = module_0.Definitions()
    var_7 = module_1.type_from_json_schema(var_5, var_6)
    var_8 = var_7.items

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'integer'
    var_2 = {var_0: var_1}
    var_3 = module_0.Definitions()
    var_4 = module_1.type_from_json_schema(var_2, var_3)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'boolean'
    var_2 = {var_0: var_1}
    var_3 = module_0.Definitions()
    var_4 = module_1.type_from_json_schema(var_2, var_3)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'number'
    var_2 = {var_0: var_1}
    var_3 = module_0.Definitions()
    var_4 = module_1.type_from_json_schema(var_2, var_3)

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
    var_8 = module_0.Definitions()
    var_9 = module_1.type_from_json_schema(var_7, var_8)
    var_10 = var_9.pattern_properties[var_3]

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'additionalProperties'
    var_2 = 'object'
    var_3 = 'string'
    var_4 = {var_0: var_3}
    var_5 = {var_0: var_2, var_1: var_4}
    var_6 = module_0.Definitions()
    var_7 = module_1.type_from_json_schema(var_5, var_6)
    var_8 = var_7.additional_properties

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'propertyNames'
    var_2 = 'object'
    var_3 = 'string'
    var_4 = {var_0: var_3}
    var_5 = {var_0: var_2, var_1: var_4}
    var_6 = module_0.Definitions()
    var_7 = module_1.type_from_json_schema(var_5, var_6)
    var_8 = var_7.property_names



# Parsed testcases at query #24
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = None
    var_2 = module_0.Integer()
    var_3 = module_1.IfThenElse(var_0, var_1, var_2)
    var_4 = module_2.to_json_schema(var_3)



# Parsed testcases at query #25
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
    var_0 = module_0.NeverMatch()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.String()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = module_0.String()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = module_0.String(min_length=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = module_0.String(max_length=var_0)
    var_2 = module_1.to_json_schema(var_1)

import re as module_0
import typesystem.fields as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = module_0.compile(var_0)
    var_2 = module_1.String()
    var_3 = module_2.to_json_schema(var_2)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'email'
    var_1 = module_0.String(format=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Integer()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = module_0.Integer(minimum=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = module_0.Integer(maximum=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = module_0.Integer(exclusive_minimum=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = module_0.Integer(exclusive_maximum=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 2
    var_1 = module_0.Integer(multiple_of=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Float()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Float()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Decimal()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Decimal()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Array()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Array()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = module_0.Array(min_items=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = module_0.Array(max_items=var_0)
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
    var_0 = module_0.String()
    var_1 = module_0.Array(additional_items=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Array(unique_items=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Object()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Object()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.Object(properties=var_2)
    var_4 = module_1.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.Object(pattern_properties=var_2)
    var_4 = module_1.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Object(additional_properties=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Object(property_names=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = module_0.Object(max_properties=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = module_0.Object(min_properties=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = [var_0]
    var_2 = module_0.Object(required=var_1)
    var_3 = module_1.to_json_schema(var_2)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = module_2.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = True
    var_1 = 'name'
    var_2 = module_0.String()
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = module_2.to_json_schema(var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = [var_0]
    var_4 = module_1.Schema(var_2)
    var_5 = module_2.to_json_schema(var_4)

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
    var_7 = module_0.Choice(choices=var_6)
    var_8 = module_1.to_json_schema(var_7)

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
    var_7 = module_0.Choice(choices=var_6)
    var_8 = module_1.to_json_schema(var_7)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'constant_value'
    var_1 = module_0.Const(var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'constant_value'
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

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = 'default'
    var_4 = module_0.Union(var_2)
    var_5 = module_1.to_json_schema(var_4)

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = module_2.to_json_schema(var_3)



# Parsed testcases at query #26
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = None
    var_3 = module_1.IfThenElse(var_0, var_1, var_2)
    var_4 = module_2.to_json_schema(var_3)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 2/3 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.String()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = module_0.String(min_length=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = module_0.String(max_length=var_0)
    var_2 = module_1.to_json_schema(var_1)

import re as module_0
import typesystem.fields as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = module_0.compile(var_0)
    var_2 = module_1.String()
    var_3 = module_2.to_json_schema(var_2)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'email'
    var_1 = module_0.String(format=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Integer()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = module_0.Integer(minimum=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = module_0.Integer(maximum=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Float()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Float()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Array()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Array()
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
    var_0 = module_0.Object()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Object()
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
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_1.Schema(var_4)
    var_6 = module_2.to_json_schema(var_5)

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
    var_7 = module_0.Choice(choices=var_6)
    var_8 = module_1.to_json_schema(var_7)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'constant_value'
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

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = module_2.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = module_2.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = module_0.Boolean()
    var_3 = module_1.IfThenElse(var_0, var_1, var_2)
    var_4 = module_2.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Not(var_0)
    var_2 = module_2.to_json_schema(var_1)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.to_json_schema(var_0)



# Parsed testcases at query #28
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'test'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = module_1.to_json_schema(var_0, var_5)



# Parsed testcases at query #29
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Array()
    var_2 = module_1.to_json_schema(var_1)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_from_json_schema_type_number. Retrieved 17/18 statements.
# Partially parsed test_from_json_schema_type_integer. Retrieved 17/18 statements.
# Partially parsed test_from_json_schema_type_string. Retrieved 16/17 statements.
# Partially parsed test_from_json_schema_type_boolean. Retrieved 8/9 statements.
# Partially parsed test_from_json_schema_type_array. Retrieved 20/22 statements.
# Partially parsed test_from_json_schema_type_object. Retrieved 30/34 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'exclusiveMinimum'
    var_4 = 'exclusiveMaximum'
    var_5 = 'multipleOf'
    var_6 = 'default'
    var_7 = 'number'
    var_8 = 0
    var_9 = 100
    var_10 = True
    var_11 = 2
    var_12 = 50
    var_13 = {var_0: var_7, var_1: var_8, var_2: var_9, var_3: var_10, var_4: var_10, var_5: var_11, var_6: var_12}
    var_14 = module_0.Definitions()
    var_15 = False
    var_16 = module_1.from_json_schema_type(var_13, var_7, var_15, var_14)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'exclusiveMinimum'
    var_4 = 'exclusiveMaximum'
    var_5 = 'multipleOf'
    var_6 = 'default'
    var_7 = 'integer'
    var_8 = 0
    var_9 = 100
    var_10 = True
    var_11 = 2
    var_12 = 50
    var_13 = {var_0: var_7, var_1: var_8, var_2: var_9, var_3: var_10, var_4: var_10, var_5: var_11, var_6: var_12}
    var_14 = module_0.Definitions()
    var_15 = False
    var_16 = module_1.from_json_schema_type(var_13, var_7, var_15, var_14)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'minLength'
    var_2 = 'maxLength'
    var_3 = 'format'
    var_4 = 'pattern'
    var_5 = 'default'
    var_6 = 'string'
    var_7 = 5
    var_8 = 10
    var_9 = 'email'
    var_10 = '^[a-zA-Z0-9]+$'
    var_11 = 'test'
    var_12 = {var_0: var_6, var_1: var_7, var_2: var_8, var_3: var_9, var_4: var_10, var_5: var_11}
    var_13 = module_0.Definitions()
    var_14 = False
    var_15 = module_1.from_json_schema_type(var_12, var_6, var_14, var_13)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'default'
    var_2 = 'boolean'
    var_3 = True
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Definitions()
    var_6 = False
    var_7 = module_1.from_json_schema_type(var_4, var_2, var_6, var_5)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'items'
    var_2 = 'additionalItems'
    var_3 = 'minItems'
    var_4 = 'maxItems'
    var_5 = 'uniqueItems'
    var_6 = 'default'
    var_7 = 'array'
    var_8 = 'string'
    var_9 = {var_0: var_8}
    var_10 = False
    var_11 = 1
    var_12 = 10
    var_13 = True
    var_14 = 'test'
    var_15 = [var_14]
    var_16 = {var_0: var_7, var_1: var_9, var_2: var_10, var_3: var_11, var_4: var_12, var_5: var_13, var_6: var_15}
    var_17 = module_0.Definitions()
    var_18 = module_1.from_json_schema_type(var_16, var_7, var_10, var_17)
    var_19 = var_18.items

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'properties'
    var_2 = 'patternProperties'
    var_3 = 'additionalProperties'
    var_4 = 'propertyNames'
    var_5 = 'minProperties'
    var_6 = 'maxProperties'
    var_7 = 'required'
    var_8 = 'default'
    var_9 = 'object'
    var_10 = 'name'
    var_11 = 'string'
    var_12 = {var_0: var_11}
    var_13 = {var_10: var_12}
    var_14 = '^S_'
    var_15 = {var_0: var_11}
    var_16 = {var_14: var_15}
    var_17 = False
    var_18 = {var_0: var_11}
    var_19 = 1
    var_20 = 10
    var_21 = [var_10]
    var_22 = 'test'
    var_23 = {var_10: var_22}
    var_24 = {var_0: var_9, var_1: var_13, var_2: var_16, var_3: var_17, var_4: var_18, var_5: var_19, var_6: var_20, var_7: var_21, var_8: var_23}
    var_25 = module_0.Definitions()
    var_26 = module_1.from_json_schema_type(var_24, var_9, var_17, var_25)
    var_27 = var_26.properties[var_10]
    var_28 = var_26.pattern_properties[var_14]
    var_29 = var_26.property_names

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Definitions()
    var_2 = 'invalid'
    var_3 = False
    var_4 = module_1.from_json_schema_type(var_0, var_2, var_3, var_1)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 5/7 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.String()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = module_0.String()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = module_0.String(min_length=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = module_0.String(max_length=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Integer()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Float()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Array()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Array()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Object()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Object()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Schema()
    var_1 = module_1.to_json_schema(var_0)

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
    var_0 = 'constant_value'
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

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = module_2.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = module_2.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = module_0.Boolean()
    var_3 = module_1.IfThenElse(var_0, var_1, var_2)
    var_4 = module_2.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Not(var_0)
    var_2 = module_2.to_json_schema(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'string_field'
    var_1 = 'integer_field'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_from_json_schema_type_number. Retrieved 12/13 statements.
# Partially parsed test_from_json_schema_type_integer. Retrieved 12/13 statements.
# Partially parsed test_from_json_schema_type_string. Retrieved 14/15 statements.
# Partially parsed test_from_json_schema_type_boolean. Retrieved 8/9 statements.
# Partially parsed test_from_json_schema_type_array. Retrieved 19/21 statements.
# Partially parsed test_from_json_schema_type_object. Retrieved 21/22 statements.
# Partially parsed test_from_json_schema_type_allow_null. Retrieved 8/9 statements.
# Partially parsed test_from_json_schema_type_array_with_additional_items. Retrieved 19/22 statements.
# Partially parsed test_from_json_schema_type_object_with_pattern_properties. Retrieved 15/16 statements.
# Partially parsed test_from_json_schema_type_object_with_property_names. Retrieved 14/16 statements.


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
    var_10 = module_0.Definitions()
    var_11 = module_1.from_json_schema_type(var_8, var_4, var_9, var_10)

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
    var_10 = module_0.Definitions()
    var_11 = module_1.from_json_schema_type(var_8, var_4, var_9, var_10)

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
    var_12 = module_0.Definitions()
    var_13 = module_1.from_json_schema_type(var_10, var_5, var_11, var_12)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'default'
    var_2 = 'boolean'
    var_3 = True
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = False
    var_6 = module_0.Definitions()
    var_7 = module_1.from_json_schema_type(var_4, var_2, var_5, var_6)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'items'
    var_2 = 'minItems'
    var_3 = 'maxItems'
    var_4 = 'uniqueItems'
    var_5 = 'default'
    var_6 = 'array'
    var_7 = 'string'
    var_8 = {var_0: var_7}
    var_9 = 1
    var_10 = 5
    var_11 = True
    var_12 = 'test'
    var_13 = [var_12]
    var_14 = {var_0: var_6, var_1: var_8, var_2: var_9, var_3: var_10, var_4: var_11, var_5: var_13}
    var_15 = False
    var_16 = module_0.Definitions()
    var_17 = module_1.from_json_schema_type(var_14, var_6, var_15, var_16)
    var_18 = var_17.items

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'properties'
    var_2 = 'additionalProperties'
    var_3 = 'required'
    var_4 = 'default'
    var_5 = 'object'
    var_6 = 'name'
    var_7 = 'age'
    var_8 = 'string'
    var_9 = {var_0: var_8}
    var_10 = 'integer'
    var_11 = {var_0: var_10}
    var_12 = {var_6: var_9, var_7: var_11}
    var_13 = False
    var_14 = [var_6]
    var_15 = 'John'
    var_16 = 30
    var_17 = {var_6: var_15, var_7: var_16}
    var_18 = {var_0: var_5, var_1: var_12, var_2: var_13, var_3: var_14, var_4: var_17}
    var_19 = module_0.Definitions()
    var_20 = module_1.from_json_schema_type(var_18, var_5, var_13, var_19)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'minLength'
    var_2 = 'string'
    var_3 = 0
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True
    var_6 = module_0.Definitions()
    var_7 = module_1.from_json_schema_type(var_4, var_2, var_5, var_6)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'items'
    var_2 = 'additionalItems'
    var_3 = 'array'
    var_4 = 'string'
    var_5 = {var_0: var_4}
    var_6 = 'integer'
    var_7 = {var_0: var_6}
    var_8 = [var_5, var_7]
    var_9 = 'boolean'
    var_10 = {var_0: var_9}
    var_11 = {var_0: var_3, var_1: var_8, var_2: var_10}
    var_12 = False
    var_13 = module_0.Definitions()
    var_14 = module_1.from_json_schema_type(var_11, var_3, var_12, var_13)
    var_15 = var_14.items
    var_16 = var_14.items
    var_17 = len(var_16)
    assert var_17 == 2
    var_18 = var_14.additional_items

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'patternProperties'
    var_2 = 'additionalProperties'
    var_3 = 'object'
    var_4 = '^S_'
    var_5 = '^I_'
    var_6 = 'string'
    var_7 = {var_0: var_6}
    var_8 = 'integer'
    var_9 = {var_0: var_8}
    var_10 = {var_4: var_7, var_5: var_9}
    var_11 = False
    var_12 = {var_0: var_3, var_1: var_10, var_2: var_11}
    var_13 = module_0.Definitions()
    var_14 = module_1.from_json_schema_type(var_12, var_3, var_11, var_13)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'propertyNames'
    var_2 = 'additionalProperties'
    var_3 = 'object'
    var_4 = 'pattern'
    var_5 = 'string'
    var_6 = '^[A-Za-z_][A-Za-z0-9_]*$'
    var_7 = {var_0: var_5, var_4: var_6}
    var_8 = True
    var_9 = {var_0: var_3, var_1: var_7, var_2: var_8}
    var_10 = False
    var_11 = module_0.Definitions()
    var_12 = module_1.from_json_schema_type(var_9, var_3, var_10, var_11)
    var_13 = var_12.property_names



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_all_of_from_json_schema_basic. Retrieved 16/19 statements.
# Partially parsed test_all_of_from_json_schema_with_default. Retrieved 10/11 statements.
# Partially parsed test_all_of_from_json_schema_empty. Retrieved 7/8 statements.
# Partially parsed test_all_of_from_json_schema_with_definitions. Retrieved 12/15 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'allOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'number'
    var_5 = {var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = {var_0: var_6}
    var_8 = module_0.Definitions()
    var_9 = module_1.all_of_from_json_schema(var_7, var_8)
    var_10 = var_9.all_of
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = 0
    var_13 = var_9.all_of[var_12]
    var_14 = 1
    var_15 = var_9.all_of[var_14]

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'allOf'
    var_1 = 'default'
    var_2 = 'type'
    var_3 = 'integer'
    var_4 = {var_2: var_3}
    var_5 = [var_4]
    var_6 = 42
    var_7 = {var_0: var_5, var_1: var_6}
    var_8 = module_0.Definitions()
    var_9 = module_1.all_of_from_json_schema(var_7, var_8)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'allOf'
    var_1 = []
    var_2 = {var_0: var_1}
    var_3 = module_0.Definitions()
    var_4 = module_1.all_of_from_json_schema(var_2, var_3)
    var_5 = var_4.all_of
    var_6 = len(var_5)
    assert var_6 == 0

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'allOf'
    var_1 = '$ref'
    var_2 = '#/components/schemas/test'
    var_3 = {var_1: var_2}
    var_4 = [var_3]
    var_5 = {var_0: var_4}
    var_6 = module_0.Definitions()
    var_7 = module_1.all_of_from_json_schema(var_5, var_6)
    var_8 = var_7.all_of
    var_9 = len(var_8)
    assert var_9 == 1
    var_10 = 0
    var_11 = var_7.all_of[var_10]



# Parsed testcases at query #3
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
    var_0 = module_0.NeverMatch()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.String()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = module_0.String(min_length=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = module_0.String(max_length=var_0)
    var_2 = module_1.to_json_schema(var_1)

import re as module_0
import typesystem.fields as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = module_0.compile(var_0)
    var_2 = module_1.String()
    var_3 = module_2.to_json_schema(var_2)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'email'
    var_1 = module_0.String(format=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Integer()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = module_0.Integer(minimum=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = module_0.Integer(maximum=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Float()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Float()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Array()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Array()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = module_0.Array(min_items=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = module_0.Array(max_items=var_0)
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
    var_0 = module_0.Integer()
    var_1 = module_0.Array(additional_items=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Array(unique_items=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Object()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Object()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.Object(properties=var_2)
    var_4 = module_1.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = '^S_'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.Object(pattern_properties=var_2)
    var_4 = module_1.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.Object(additional_properties=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Object(property_names=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = module_0.Object(max_properties=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = module_0.Object(min_properties=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = [var_0]
    var_2 = module_0.Object(required=var_1)
    var_3 = module_1.to_json_schema(var_2)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = module_2.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = True
    var_1 = 'name'
    var_2 = module_0.String()
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = module_2.to_json_schema(var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = [var_0]
    var_4 = module_1.Schema(var_2)
    var_5 = module_2.to_json_schema(var_4)

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
    var_0 = 'value'
    var_1 = module_0.Const(var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'value'
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

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = 'default'
    var_4 = module_0.Union(var_2)
    var_5 = module_1.to_json_schema(var_4)

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = module_2.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = module_2.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = module_0.Boolean()
    var_3 = module_1.IfThenElse(var_0, var_1, var_2)
    var_4 = module_2.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Not(var_0)
    var_2 = module_2.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'ref'
    var_2 = module_1.Reference(var_1)
    var_3 = module_2.to_json_schema(var_2)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 5/7 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.String()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = module_0.String()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = module_0.String(min_length=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = module_0.String(max_length=var_0)
    var_2 = module_1.to_json_schema(var_1)

import re as module_0
import typesystem.fields as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = module_0.compile(var_0)
    var_2 = module_1.String()
    var_3 = module_2.to_json_schema(var_2)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'email'
    var_1 = module_0.String(format=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Integer()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = module_0.Integer(minimum=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = module_0.Integer(maximum=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Float()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Array()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Array()
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
    var_0 = module_0.Object()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Object()
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

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Schema()
    var_1 = module_1.to_json_schema(var_0)

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
    var_7 = module_0.Choice(choices=var_6)
    var_8 = module_1.to_json_schema(var_7)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'constant_value'
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

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = module_2.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = module_2.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = module_0.Boolean()
    var_3 = module_1.IfThenElse(var_0, var_1, var_2)
    var_4 = module_2.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Not(var_0)
    var_2 = module_2.to_json_schema(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'string_field'
    var_1 = 'integer_field'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #5
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
    var_0 = module_0.NeverMatch()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.String()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = module_0.String()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = module_0.String(min_length=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = module_0.String(max_length=var_0)
    var_2 = module_1.to_json_schema(var_1)

import re as module_0
import typesystem.fields as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = module_0.compile(var_0)
    var_2 = module_1.String()
    var_3 = module_2.to_json_schema(var_2)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'email'
    var_1 = module_0.String(format=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Integer()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = module_0.Integer(minimum=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = module_0.Integer(maximum=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = module_0.Integer(exclusive_minimum=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = module_0.Integer(exclusive_maximum=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 2
    var_1 = module_0.Integer(multiple_of=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Float()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Decimal()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Array()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Array()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = module_0.Array(min_items=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = module_0.Array(max_items=var_0)
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
    var_0 = module_0.String()
    var_1 = module_0.Array(additional_items=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Array(unique_items=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Object()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Object()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.Object(properties=var_2)
    var_4 = module_1.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = '^S_'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.Object(pattern_properties=var_2)
    var_4 = module_1.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Object(additional_properties=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Object(property_names=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = module_0.Object(max_properties=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = module_0.Object(min_properties=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = [var_0]
    var_2 = module_0.Object(required=var_1)
    var_3 = module_1.to_json_schema(var_2)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = module_2.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = True
    var_1 = 'name'
    var_2 = module_0.String()
    var_3 = {var_1: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = module_2.to_json_schema(var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = [var_0]
    var_4 = module_1.Schema(var_2)
    var_5 = module_2.to_json_schema(var_4)

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
    var_7 = module_0.Choice(choices=var_6)
    var_8 = module_1.to_json_schema(var_7)

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
    var_7 = module_0.Choice(choices=var_6)
    var_8 = module_1.to_json_schema(var_7)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'constant_value'
    var_1 = module_0.Const(var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'constant_value'
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

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = 'default'
    var_4 = module_0.Union(var_2)
    var_5 = module_1.to_json_schema(var_4)

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = module_2.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = module_2.to_json_schema(var_3)



# Parsed testcases at query #6
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Integer()
    var_2 = module_1.to_json_schema(var_1)



# Parsed testcases at query #7
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.String()
    var_2 = module_1.to_json_schema(var_1)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_if_then_else_from_json_schema. Retrieved 18/22 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'else'
    var_4 = 'default'
    var_5 = 'type'
    var_6 = 'string'
    var_7 = {var_5: var_6}
    var_8 = 'number'
    var_9 = {var_5: var_8}
    var_10 = 'boolean'
    var_11 = {var_5: var_10}
    var_12 = 42
    var_13 = {var_1: var_7, var_2: var_9, var_3: var_11, var_4: var_12}
    var_14 = module_1.if_then_else_from_json_schema(var_13, var_0)
    var_15 = var_14.if_clause
    var_16 = var_14.then_clause
    var_17 = var_14.else_clause



# Parsed testcases at query #9
#--------------------------




import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Schema()
    var_2 = module_1.to_json_schema(var_1)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_enum_from_json_schema_with_valid_data. Retrieved 9/10 statements.
# Partially parsed test_enum_from_json_schema_without_default. Retrieved 7/8 statements.
# Partially parsed test_enum_from_json_schema_empty_enum. Retrieved 7/8 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'enum'
    var_1 = 'default'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_0: var_5, var_1: var_2}
    var_7 = module_0.Definitions()
    var_8 = module_1.enum_from_json_schema(var_6, var_7)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'enum'
    var_1 = 'x'
    var_2 = 'y'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.Definitions()
    var_6 = module_1.enum_from_json_schema(var_4, var_5)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'enum'
    var_1 = 'default'
    var_2 = []
    var_3 = 'z'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Definitions()
    var_6 = module_1.enum_from_json_schema(var_4, var_5)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_type_from_json_schema_with_single_type. Retrieved 5/6 statements.
# Partially parsed test_type_from_json_schema_with_multiple_types. Retrieved 13/16 statements.
# Partially parsed test_type_from_json_schema_with_null_type. Retrieved 5/6 statements.
# Partially parsed test_type_from_json_schema_with_allow_null. Retrieved 7/8 statements.
# Partially parsed test_type_from_json_schema_with_integer_type. Retrieved 9/10 statements.
# Partially parsed test_type_from_json_schema_with_boolean_type. Retrieved 7/8 statements.
# Partially parsed test_type_from_json_schema_with_array_type. Retrieved 9/11 statements.
# Partially parsed test_type_from_json_schema_with_object_type. Retrieved 11/13 statements.
# Partially parsed test_type_from_json_schema_with_no_type. Retrieved 3/4 statements.
# Partially parsed test_type_from_json_schema_with_null_in_multiple_types. Retrieved 7/8 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = module_0.Definitions()
    var_4 = module_1.type_from_json_schema(var_2, var_3)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = 'number'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.Definitions()
    var_6 = module_1.type_from_json_schema(var_4, var_5)
    var_7 = var_6.any_of
    var_8 = len(var_7)
    assert var_8 == 2
    var_9 = 0
    var_10 = var_6.any_of[var_9]
    var_11 = 1
    var_12 = var_6.any_of[var_11]

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'null'
    var_2 = {var_0: var_1}
    var_3 = module_0.Definitions()
    var_4 = module_1.type_from_json_schema(var_2, var_3)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'default'
    var_2 = 'string'
    var_3 = None
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Definitions()
    var_6 = module_1.type_from_json_schema(var_4, var_5)

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
    var_7 = module_0.Definitions()
    var_8 = module_1.type_from_json_schema(var_6, var_7)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'default'
    var_2 = 'boolean'
    var_3 = True
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Definitions()
    var_6 = module_1.type_from_json_schema(var_4, var_5)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'items'
    var_2 = 'array'
    var_3 = 'string'
    var_4 = {var_0: var_3}
    var_5 = {var_0: var_2, var_1: var_4}
    var_6 = module_0.Definitions()
    var_7 = module_1.type_from_json_schema(var_5, var_6)
    var_8 = var_7.items

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
    var_8 = module_0.Definitions()
    var_9 = module_1.type_from_json_schema(var_7, var_8)
    var_10 = var_9.properties[var_3]

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Definitions()
    var_2 = module_1.type_from_json_schema(var_0, var_1)

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



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 3/5 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.NeverMatch()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.String()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'default_value'
    var_1 = module_0.String()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = module_0.String(min_length=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = module_0.String(max_length=var_0)
    var_2 = module_1.to_json_schema(var_1)

import re as module_0
import typesystem.fields as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = module_0.compile(var_0)
    var_2 = module_1.String()
    var_3 = module_2.to_json_schema(var_2)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'email'
    var_1 = module_0.String(format=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Integer()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = module_0.Integer(minimum=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = module_0.Integer(maximum=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Float()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Decimal()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Array()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Array(var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = module_0.Array(min_items=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = module_0.Array(max_items=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Object()
    var_1 = module_1.to_json_schema(var_0)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.Object(properties=var_2)
    var_4 = module_1.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = [var_0]
    var_4 = module_0.Object(properties=var_2, required=var_3)
    var_5 = module_1.to_json_schema(var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = module_2.to_json_schema(var_3)

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
    var_7 = module_0.Choice(choices=var_6)
    var_8 = module_1.to_json_schema(var_7)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'constant_value'
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

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = module_2.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = module_2.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = module_1.IfThenElse(var_0, var_1)
    var_3 = module_2.to_json_schema(var_2)

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Not(var_0)
    var_2 = module_2.to_json_schema(var_1)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'string_field'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}



# Parsed testcases at query #13
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Array()
    var_2 = module_1.to_json_schema(var_1)



# Parsed testcases at query #14
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Integer()
    var_2 = module_1.to_json_schema(var_1)



# Parsed testcases at query #15
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = None
    var_2 = module_1.IfThenElse(var_0, var_1, var_1)
    var_3 = module_2.to_json_schema(var_2)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_array_field_items_is_list_or_tuple. Retrieved 5/7 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Array(var_2)
    var_4 = var_3.items



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_from_json_schema_with_bool_true. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_with_bool_false. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_with_ref. Retrieved 5/7 statements.
# Partially parsed test_from_json_schema_with_type_constraint. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_enum_constraint. Retrieved 7/8 statements.
# Partially parsed test_from_json_schema_with_const_constraint. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_all_of_constraint. Retrieved 11/12 statements.
# Partially parsed test_from_json_schema_with_any_of_constraint. Retrieved 11/12 statements.
# Partially parsed test_from_json_schema_with_one_of_constraint. Retrieved 11/12 statements.
# Partially parsed test_from_json_schema_with_not_constraint. Retrieved 7/9 statements.
# Partially parsed test_from_json_schema_with_if_then_else_constraint. Retrieved 15/19 statements.
# Partially parsed test_from_json_schema_with_multiple_constraints. Retrieved 11/12 statements.
# Partially parsed test_from_json_schema_with_no_constraints. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_with_components_schemas. Retrieved 10/11 statements.


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
    var_0 = '$ref'
    var_1 = '#/components/schemas/test'
    var_2 = {var_0: var_1}
    var_3 = module_0.Definitions()
    var_4 = module_1.from_json_schema(var_2, var_3)

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

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'const'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)

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



# Parsed testcases at query #18
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = {}
    var_2 = True



