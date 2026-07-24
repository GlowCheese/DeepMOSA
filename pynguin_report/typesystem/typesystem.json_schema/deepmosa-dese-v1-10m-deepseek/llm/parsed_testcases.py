####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.to_json_schema(var_0)
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.String()
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'string'
    var_5 = 'null'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = module_0.String(min_length=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'minLength'
    var_5 = 'string'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = module_0.String(max_length=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'maxLength'
    var_5 = 'string'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.String(pattern=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'pattern'
    var_5 = 'string'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'email'
    var_1 = module_0.String(format=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'format'
    var_5 = 'string'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_1.to_json_schema(var_0)
    var_2 = 'type'
    var_3 = 'integer'
    var_4 = {var_2: var_3}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Integer()
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'integer'
    var_5 = 'null'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = module_0.Integer(minimum=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'minimum'
    var_5 = 'integer'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = module_0.Integer(maximum=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'maximum'
    var_5 = 'integer'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = module_0.Integer(exclusive_minimum=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'exclusiveMinimum'
    var_5 = 'integer'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = module_0.Integer(exclusive_maximum=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'exclusiveMaximum'
    var_5 = 'integer'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = module_0.Integer(multiple_of=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'multipleOf'
    var_5 = 'integer'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Float()
    var_1 = module_1.to_json_schema(var_0)
    var_2 = 'type'
    var_3 = 'number'
    var_4 = {var_2: var_3}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Float()
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'number'
    var_5 = 'null'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = module_1.to_json_schema(var_0)
    var_2 = 'type'
    var_3 = 'boolean'
    var_4 = {var_2: var_3}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean()
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'boolean'
    var_5 = 'null'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Array()
    var_1 = module_1.to_json_schema(var_0)
    var_2 = 'type'
    var_3 = 'array'
    var_4 = {var_2: var_3}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Array()
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'array'
    var_5 = 'null'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = module_0.Array(min_items=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'minItems'
    var_5 = 'array'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = module_0.Array(max_items=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'maxItems'
    var_5 = 'array'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Array(var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'items'
    var_5 = 'array'
    var_6 = 'string'
    var_7 = {var_3: var_6}
    var_8 = {var_3: var_5, var_4: var_7}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Array(var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'type'
    var_6 = 'items'
    var_7 = 'array'
    var_8 = 'string'
    var_9 = {var_5: var_8}
    var_10 = 'integer'
    var_11 = {var_5: var_10}
    var_12 = [var_9, var_11]
    var_13 = {var_5: var_7, var_6: var_12}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Array(unique_items=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'uniqueItems'
    var_5 = 'array'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Object()
    var_1 = module_1.to_json_schema(var_0)
    var_2 = 'type'
    var_3 = 'object'
    var_4 = {var_2: var_3}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Object()
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'object'
    var_5 = 'null'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.Object(properties=var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'type'
    var_6 = 'properties'
    var_7 = 'object'
    var_8 = 'string'
    var_9 = {var_5: var_8}
    var_10 = {var_0: var_9}
    var_11 = {var_5: var_7, var_6: var_10}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = [var_0]
    var_4 = module_0.Object(properties=var_2, required=var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = 'type'
    var_7 = 'properties'
    var_8 = 'required'
    var_9 = 'object'
    var_10 = 'string'
    var_11 = {var_6: var_10}
    var_12 = {var_0: var_11}
    var_13 = [var_0]
    var_14 = {var_6: var_9, var_7: var_12, var_8: var_13}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = module_0.Object(min_properties=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'minProperties'
    var_5 = 'object'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = module_0.Object(max_properties=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'maxProperties'
    var_5 = 'object'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = module_2.to_json_schema(var_3)
    var_5 = 'type'
    var_6 = 'properties'
    var_7 = 'object'
    var_8 = 'string'
    var_9 = {var_5: var_8}
    var_10 = {var_0: var_9}
    var_11 = {var_5: var_7, var_6: var_10}

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
    var_6 = 'type'
    var_7 = 'properties'
    var_8 = 'required'
    var_9 = 'object'
    var_10 = 'string'
    var_11 = {var_6: var_10}
    var_12 = {var_0: var_11}
    var_13 = [var_0]
    var_14 = {var_6: var_9, var_7: var_12, var_8: var_13}

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
    var_9 = 'enum'
    var_10 = [var_0, var_3]
    var_11 = {var_9: var_10}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed'
    var_1 = module_0.Const(var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'const'
    var_4 = {var_3: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'anyOf'
    var_6 = 'type'
    var_7 = 'string'
    var_8 = {var_6: var_7}
    var_9 = 'integer'
    var_10 = {var_6: var_9}
    var_11 = [var_8, var_10]
    var_12 = {var_5: var_11}

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = module_2.to_json_schema(var_3)
    var_5 = 'oneOf'
    var_6 = 'type'
    var_7 = 'string'
    var_8 = {var_6: var_7}
    var_9 = 'integer'
    var_10 = {var_6: var_9}
    var_11 = [var_8, var_10]
    var_12 = {var_5: var_11}

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = module_2.to_json_schema(var_3)
    var_5 = 'allOf'
    var_6 = 'type'
    var_7 = 'string'
    var_8 = {var_6: var_7}
    var_9 = 'integer'
    var_10 = {var_6: var_9}
    var_11 = [var_8, var_10]
    var_12 = {var_5: var_11}

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = module_1.IfThenElse(var_0, var_1)
    var_3 = module_2.to_json_schema(var_2)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_additional_properties_is_not_bool. Retrieved 4/6 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Object(additional_properties=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'additionalProperties'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_additional_items_is_not_bool. Retrieved 6/8 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.String()
    var_2 = module_0.Array(var_0, var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'additionalItems'
    var_5 = var_3[var_4]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_predicate_at_line_17_is_true_for_definitions_instance. Retrieved 3/8 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}



# Parsed testcases at query #5
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Object()
    var_2 = module_1.to_json_schema(var_1)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_ref_from_json_schema_creates_reference_with_correct_to. Retrieved 5/6 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = '#/components/schemas/User'
    var_3 = {var_1: var_2}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = 'http://example.com/schema'
    var_3 = {var_1: var_2}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = {}
    var_2 = module_1.ref_from_json_schema(var_1, var_0)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_additional_items_is_not_bool. Retrieved 6/8 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.String()
    var_2 = module_0.Array(var_0, var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'additionalItems'
    var_5 = var_3[var_4]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_from_json_schema_boolean_true. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_boolean_false. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_empty_object. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_with_ref. Retrieved 5/7 statements.
# Partially parsed test_from_json_schema_with_type_string. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_type_integer. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_type_number. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_type_boolean. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_type_array. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_type_object. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_enum. Retrieved 7/8 statements.
# Partially parsed test_from_json_schema_with_const. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_allOf. Retrieved 10/11 statements.
# Partially parsed test_from_json_schema_with_anyOf. Retrieved 9/10 statements.
# Partially parsed test_from_json_schema_with_oneOf. Retrieved 9/10 statements.
# Partially parsed test_from_json_schema_with_not. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_with_if_then. Retrieved 10/11 statements.
# Partially parsed test_from_json_schema_with_if_then_else. Retrieved 13/14 statements.
# Partially parsed test_from_json_schema_multiple_constraints. Retrieved 8/9 statements.
# Partially parsed test_from_json_schema_with_components. Retrieved 10/11 statements.
# Partially parsed test_from_json_schema_nullable_type. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_multiple_types. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_no_type_but_nullable. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_no_type_not_nullable. Retrieved 2/3 statements.


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

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = '#/components/schemas/Test'
    var_3 = {var_1: var_2}
    var_4 = module_1.from_json_schema(var_3, var_0)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'integer'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'number'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'boolean'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'array'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'object'
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
    var_1 = 42
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'allOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'minLength'
    var_5 = 1
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
    var_4 = 'integer'
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
    var_4 = 'integer'
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
    var_5 = 'minLength'
    var_6 = 1
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
    var_6 = 'minLength'
    var_7 = 1
    var_8 = {var_6: var_7}
    var_9 = 'integer'
    var_10 = {var_3: var_9}
    var_11 = {var_0: var_5, var_1: var_8, var_2: var_10}
    var_12 = module_0.from_json_schema(var_11)

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
    var_1 = 'string'
    var_2 = 'null'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.from_json_schema(var_4)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = 'integer'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.from_json_schema(var_4)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'null'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.from_json_schema(var_0)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_additional_items_is_false. Retrieved 4/5 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Array(additional_items=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'additionalItems'



# Parsed testcases at query #10
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
    var_0 = True
    var_1 = module_0.Float()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Decimal()
    var_2 = module_1.to_json_schema(var_1)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 18/20 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.to_json_schema(var_0)
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.String()
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'string'
    var_5 = 'null'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = module_0.String(min_length=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'minLength'
    var_5 = 'string'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = module_0.String(max_length=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'maxLength'
    var_5 = 'string'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.String(pattern=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'pattern'
    var_5 = 'string'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'email'
    var_1 = module_0.String(format=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'format'
    var_5 = 'string'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_1.to_json_schema(var_0)
    var_2 = 'type'
    var_3 = 'integer'
    var_4 = {var_2: var_3}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Integer()
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'integer'
    var_5 = 'null'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = module_0.Integer(minimum=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'minimum'
    var_5 = 'integer'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = module_0.Integer(maximum=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'maximum'
    var_5 = 'integer'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = module_0.Integer(exclusive_minimum=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'exclusiveMinimum'
    var_5 = 'integer'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = module_0.Integer(exclusive_maximum=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'exclusiveMaximum'
    var_5 = 'integer'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 2
    var_1 = module_0.Integer(multiple_of=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'multipleOf'
    var_5 = 'integer'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Float()
    var_1 = module_1.to_json_schema(var_0)
    var_2 = 'type'
    var_3 = 'number'
    var_4 = {var_2: var_3}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Float()
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'number'
    var_5 = 'null'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = module_1.to_json_schema(var_0)
    var_2 = 'type'
    var_3 = 'boolean'
    var_4 = {var_2: var_3}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean()
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'boolean'
    var_5 = 'null'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Array()
    var_1 = module_1.to_json_schema(var_0)
    var_2 = 'type'
    var_3 = 'array'
    var_4 = {var_2: var_3}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Array()
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'array'
    var_5 = 'null'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = module_0.Array(min_items=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'minItems'
    var_5 = 'array'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = module_0.Array(max_items=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'maxItems'
    var_5 = 'array'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Array(var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'items'
    var_5 = 'array'
    var_6 = 'string'
    var_7 = {var_3: var_6}
    var_8 = {var_3: var_5, var_4: var_7}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Array(unique_items=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'uniqueItems'
    var_5 = 'array'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Object()
    var_1 = module_1.to_json_schema(var_0)
    var_2 = 'type'
    var_3 = 'object'
    var_4 = {var_2: var_3}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Object()
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'object'
    var_5 = 'null'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.Object(properties=var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'type'
    var_6 = 'properties'
    var_7 = 'object'
    var_8 = 'string'
    var_9 = {var_5: var_8}
    var_10 = {var_0: var_9}
    var_11 = {var_5: var_7, var_6: var_10}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Object(additional_properties=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'additionalProperties'
    var_5 = 'object'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = [var_0]
    var_4 = module_0.Object(properties=var_2, required=var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = 'type'
    var_7 = 'properties'
    var_8 = 'required'
    var_9 = 'object'
    var_10 = 'string'
    var_11 = {var_6: var_10}
    var_12 = {var_0: var_11}
    var_13 = [var_0]
    var_14 = {var_6: var_9, var_7: var_12, var_8: var_13}

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
    var_9 = 'enum'
    var_10 = [var_0, var_3]
    var_11 = {var_9: var_10}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed'
    var_1 = module_0.Const(var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'const'
    var_4 = {var_3: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'anyOf'
    var_6 = 'type'
    var_7 = 'string'
    var_8 = {var_6: var_7}
    var_9 = 'integer'
    var_10 = {var_6: var_9}
    var_11 = [var_8, var_10]
    var_12 = {var_5: var_11}

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = module_2.to_json_schema(var_3)
    var_5 = 'oneOf'
    var_6 = 'type'
    var_7 = 'string'
    var_8 = {var_6: var_7}
    var_9 = 'integer'
    var_10 = {var_6: var_9}
    var_11 = [var_8, var_10]
    var_12 = {var_5: var_11}

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = module_2.to_json_schema(var_3)
    var_5 = 'allOf'
    var_6 = 'type'
    var_7 = 'string'
    var_8 = {var_6: var_7}
    var_9 = 'integer'
    var_10 = {var_6: var_9}
    var_11 = [var_8, var_10]
    var_12 = {var_5: var_11}

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = module_1.IfThenElse(var_0, var_1)
    var_3 = module_2.to_json_schema(var_2)
    var_4 = 'if'
    var_5 = 'then'
    var_6 = 'type'
    var_7 = 'string'
    var_8 = {var_6: var_7}
    var_9 = 'integer'
    var_10 = {var_6: var_9}
    var_11 = {var_4: var_8, var_5: var_10}

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Not(var_0)
    var_2 = module_2.to_json_schema(var_1)
    var_3 = 'not'
    var_4 = 'type'
    var_5 = 'string'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'default'
    var_1 = module_0.String()
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4, var_0: var_0}

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'User'
    var_1 = 'name'
    var_2 = module_0.String()
    var_3 = {var_1: var_2}
    var_4 = module_0.Object(properties=var_3)
    var_5 = {var_0: var_4}
    var_6 = 'components'
    var_7 = 'schemas'
    var_8 = 'type'
    var_9 = 'properties'
    var_10 = 'object'
    var_11 = 'string'
    var_12 = {var_8: var_11}
    var_13 = {var_1: var_12}
    var_14 = {var_8: var_10, var_9: var_13}
    var_15 = {var_0: var_14}
    var_16 = {var_7: var_15}
    var_17 = {var_6: var_16}

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'User'
    var_1 = 'name'
    var_2 = module_0.String()
    var_3 = {var_1: var_2}
    var_4 = module_0.Object(properties=var_3)
    var_5 = module_1.Reference(var_0)
    var_6 = module_2.to_json_schema(var_5)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_predicate_at_line_17_evaluates_to_true. Retrieved 1/2 statements.


import typesystem.schemas as module_0

def test_case_0():
    var_0 = module_0.Definitions()



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_from_json_schema_type_array. Retrieved 19/20 statements.
# Partially parsed test_from_json_schema_type_array_list_items. Retrieved 22/25 statements.
# Partially parsed test_from_json_schema_type_array_additional_items_field. Retrieved 14/16 statements.
# Partially parsed test_from_json_schema_type_object. Retrieved 39/44 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'minimum'
    var_1 = 'maximum'
    var_2 = 'exclusiveMinimum'
    var_3 = 'exclusiveMaximum'
    var_4 = 'multipleOf'
    var_5 = 'default'
    var_6 = 0
    var_7 = 10
    var_8 = -1
    var_9 = 11
    var_10 = 2
    var_11 = 5.0
    var_12 = {var_0: var_6, var_1: var_7, var_2: var_8, var_3: var_9, var_4: var_10, var_5: var_11}
    var_13 = 'number'
    var_14 = False
    var_15 = module_0.Definitions()
    var_16 = module_1.from_json_schema_type(var_12, var_13, var_14, var_15)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'minimum'
    var_1 = 'maximum'
    var_2 = 'exclusiveMinimum'
    var_3 = 'exclusiveMaximum'
    var_4 = 'multipleOf'
    var_5 = 'default'
    var_6 = 0
    var_7 = 10
    var_8 = -1
    var_9 = 11
    var_10 = 2
    var_11 = 5
    var_12 = {var_0: var_6, var_1: var_7, var_2: var_8, var_3: var_9, var_4: var_10, var_5: var_11}
    var_13 = 'integer'
    var_14 = True
    var_15 = module_0.Definitions()
    var_16 = module_1.from_json_schema_type(var_12, var_13, var_14, var_15)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'minLength'
    var_1 = 'maxLength'
    var_2 = 'format'
    var_3 = 'pattern'
    var_4 = 'default'
    var_5 = 3
    var_6 = 10
    var_7 = 'email'
    var_8 = '^[a-z]+$'
    var_9 = 'abc'
    var_10 = {var_0: var_5, var_1: var_6, var_2: var_7, var_3: var_8, var_4: var_9}
    var_11 = 'string'
    var_12 = False
    var_13 = module_0.Definitions()
    var_14 = module_1.from_json_schema_type(var_10, var_11, var_12, var_13)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'minLength'
    var_1 = 'maxLength'
    var_2 = 'default'
    var_3 = 0
    var_4 = 10
    var_5 = ''
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'string'
    var_8 = True
    var_9 = module_0.Definitions()
    var_10 = module_1.from_json_schema_type(var_6, var_7, var_8, var_9)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'default'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'boolean'
    var_4 = False
    var_5 = module_0.Definitions()
    var_6 = module_1.from_json_schema_type(var_2, var_3, var_4, var_5)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = 'items'
    var_4 = 'minItems'
    var_5 = 'maxItems'
    var_6 = 'uniqueItems'
    var_7 = 'default'
    var_8 = 1
    var_9 = 5
    var_10 = True
    var_11 = 'a'
    var_12 = [var_11]
    var_13 = {var_3: var_2, var_4: var_8, var_5: var_9, var_6: var_10, var_7: var_12}
    var_14 = 'array'
    var_15 = False
    var_16 = module_0.Definitions()
    var_17 = module_1.from_json_schema_type(var_13, var_14, var_15, var_16)
    var_18 = var_17.items

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = 'integer'
    var_4 = {var_0: var_3}
    var_5 = [var_2, var_4]
    var_6 = 'items'
    var_7 = 'additionalItems'
    var_8 = 'minItems'
    var_9 = 'maxItems'
    var_10 = False
    var_11 = 2
    var_12 = {var_6: var_5, var_7: var_10, var_8: var_11, var_9: var_11}
    var_13 = 'array'
    var_14 = True
    var_15 = module_0.Definitions()
    var_16 = module_1.from_json_schema_type(var_12, var_13, var_14, var_15)
    var_17 = var_16.items
    var_18 = var_16.items
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = var_16.items[var_10]
    var_21 = var_16.items[var_14]

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = 'integer'
    var_4 = {var_0: var_3}
    var_5 = 'items'
    var_6 = 'additionalItems'
    var_7 = {var_5: var_2, var_6: var_4}
    var_8 = 'array'
    var_9 = False
    var_10 = module_0.Definitions()
    var_11 = module_1.from_json_schema_type(var_7, var_8, var_9, var_10)
    var_12 = var_11.items
    var_13 = var_11.additional_items

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = 'integer'
    var_6 = {var_2: var_5}
    var_7 = {var_0: var_4, var_1: var_6}
    var_8 = '^s_'
    var_9 = {var_2: var_3}
    var_10 = {var_8: var_9}
    var_11 = 'boolean'
    var_12 = {var_2: var_11}
    var_13 = 'pattern'
    var_14 = '^[a-z]+$'
    var_15 = {var_13: var_14}
    var_16 = 'properties'
    var_17 = 'patternProperties'
    var_18 = 'additionalProperties'
    var_19 = 'propertyNames'
    var_20 = 'minProperties'
    var_21 = 'maxProperties'
    var_22 = 'required'
    var_23 = 'default'
    var_24 = 1
    var_25 = 5
    var_26 = [var_0]
    var_27 = 'test'
    var_28 = {var_0: var_27}
    var_29 = {var_16: var_7, var_17: var_10, var_18: var_12, var_19: var_15, var_20: var_24, var_21: var_25, var_22: var_26, var_23: var_28}
    var_30 = 'object'
    var_31 = False
    var_32 = module_0.Definitions()
    var_33 = module_1.from_json_schema_type(var_29, var_30, var_31, var_32)
    var_34 = var_33.properties[var_0]
    var_35 = var_33.properties[var_1]
    var_36 = var_33.pattern_properties[var_8]
    var_37 = var_33.additional_properties
    var_38 = var_33.property_names

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'additionalProperties'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'object'
    var_4 = True
    var_5 = module_0.Definitions()
    var_6 = module_1.from_json_schema_type(var_2, var_3, var_4, var_5)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = 'invalid'
    var_2 = False
    var_3 = module_0.Definitions()
    var_4 = module_1.from_json_schema_type(var_0, var_1, var_2, var_3)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_pattern_regex_flags_unicode. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'test'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 18/20 statements.
# Failed to parse test_to_json_schema_with_unknown_field_type.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = 'test desc'
    var_2 = True
    var_3 = 'default'
    var_4 = module_0.String()
    var_5 = module_1.to_json_schema(var_4)
    var_6 = 'type'
    var_7 = 'title'
    var_8 = 'description'
    var_9 = 'string'
    var_10 = 'null'
    var_11 = [var_9, var_10]
    var_12 = {var_6: var_11, var_3: var_3, var_7: var_0, var_8: var_1}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = 100
    var_2 = False
    var_3 = module_0.Integer(minimum=var_0, maximum=var_1)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'type'
    var_6 = 'minimum'
    var_7 = 'maximum'
    var_8 = 'integer'
    var_9 = {var_5: var_8, var_6: var_2, var_7: var_1}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean()
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'default'
    var_5 = 'boolean'
    var_6 = 'null'
    var_7 = [var_5, var_6]
    var_8 = {var_3: var_7, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = 1
    var_2 = 10
    var_3 = False
    var_4 = module_0.Array(var_0, min_items=var_1, max_items=var_2)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = 'type'
    var_7 = 'minItems'
    var_8 = 'maxItems'
    var_9 = 'items'
    var_10 = 'array'
    var_11 = 'string'
    var_12 = {var_6: var_11}
    var_13 = {var_6: var_10, var_7: var_1, var_8: var_2, var_9: var_12}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = [var_0]
    var_4 = True
    var_5 = module_0.Object(properties=var_2, required=var_3)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = 'type'
    var_8 = 'properties'
    var_9 = 'required'
    var_10 = 'object'
    var_11 = 'null'
    var_12 = [var_10, var_11]
    var_13 = 'string'
    var_14 = {var_7: var_13}
    var_15 = {var_0: var_14}
    var_16 = [var_0]
    var_17 = {var_7: var_12, var_8: var_15, var_9: var_16}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = module_0.Union(var_2)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = 'type'
    var_7 = 'anyOf'
    var_8 = 'string'
    var_9 = 'null'
    var_10 = [var_8, var_9]
    var_11 = {var_6: var_8}
    var_12 = 'integer'
    var_13 = {var_6: var_12}
    var_14 = [var_11, var_13]
    var_15 = {var_6: var_10, var_7: var_14}

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'User'
    var_1 = 'name'
    var_2 = module_0.String()
    var_3 = {var_1: var_2}
    var_4 = module_0.Object(properties=var_3)
    var_5 = {var_0: var_4}
    var_6 = 'components'
    var_7 = 'schemas'
    var_8 = 'type'
    var_9 = 'properties'
    var_10 = 'object'
    var_11 = 'string'
    var_12 = {var_8: var_11}
    var_13 = {var_1: var_12}
    var_14 = {var_8: var_10, var_9: var_13}
    var_15 = {var_0: var_14}
    var_16 = {var_7: var_15}
    var_17 = {var_6: var_16}

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.Object(properties=var_2)
    var_4 = 'User'
    var_5 = module_1.Reference(var_4)
    var_6 = module_2.to_json_schema(var_5)
    var_7 = '$ref'
    var_8 = 'components'
    var_9 = '#/components/schemas/User'
    var_10 = 'schemas'
    var_11 = 'type'
    var_12 = 'properties'
    var_13 = 'object'
    var_14 = 'string'
    var_15 = {var_11: var_14}
    var_16 = {var_0: var_15}
    var_17 = {var_11: var_13, var_12: var_16}
    var_18 = {var_4: var_17}
    var_19 = {var_10: var_18}
    var_20 = {var_7: var_9, var_8: var_19}

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
    var_9 = 'enum'
    var_10 = 'default'
    var_11 = [var_0, var_3]
    var_12 = {var_9: var_11, var_10: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed_value'
    var_1 = False
    var_2 = module_0.Const(var_0)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'const'
    var_5 = {var_4: var_0}

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 1
    var_1 = module_0.String(min_length=var_0)
    var_2 = 10
    var_3 = module_0.String(max_length=var_2)
    var_4 = [var_1, var_3]
    var_5 = False
    var_6 = module_1.AllOf(var_4)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = 'allOf'
    var_9 = 'type'
    var_10 = 'minLength'
    var_11 = 'string'
    var_12 = {var_9: var_11, var_10: var_0}
    var_13 = 'maxLength'
    var_14 = {var_9: var_11, var_13: var_2}
    var_15 = [var_12, var_14]
    var_16 = {var_8: var_15}

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = module_1.OneOf(var_2)
    var_5 = module_2.to_json_schema(var_4)
    var_6 = 'type'
    var_7 = 'oneOf'
    var_8 = 'string'
    var_9 = 'null'
    var_10 = [var_8, var_9]
    var_11 = {var_6: var_8}
    var_12 = 'integer'
    var_13 = {var_6: var_12}
    var_14 = [var_11, var_13]
    var_15 = {var_6: var_10, var_7: var_14}

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 5
    var_1 = module_0.String(min_length=var_0)
    var_2 = module_0.Integer()
    var_3 = module_0.Boolean()
    var_4 = False
    var_5 = module_1.IfThenElse(var_1, var_2, var_3)
    var_6 = module_2.to_json_schema(var_5)
    var_7 = 'if'
    var_8 = 'then'
    var_9 = 'else'
    var_10 = 'type'
    var_11 = 'minLength'
    var_12 = 'string'
    var_13 = {var_10: var_12, var_11: var_0}
    var_14 = 'integer'
    var_15 = {var_10: var_14}
    var_16 = 'boolean'
    var_17 = {var_10: var_16}
    var_18 = {var_7: var_13, var_8: var_15, var_9: var_17}

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = False
    var_2 = module_1.Not(var_0)
    var_3 = module_2.to_json_schema(var_2)
    var_4 = 'not'
    var_5 = 'type'
    var_6 = 'string'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}

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
    var_0 = 'dynamic_default'
    var_1 = lambda : var_0
    var_2 = False
    var_3 = module_0.String()
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'type'
    var_6 = 'default'
    var_7 = 'string'
    var_8 = {var_5: var_7, var_6: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.String()
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}

import re as module_0
import typesystem.fields as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = module_0.compile(var_0)
    var_2 = False
    var_3 = module_1.String()
    var_4 = module_2.to_json_schema(var_3)
    var_5 = 'type'
    var_6 = 'pattern'
    var_7 = 'string'
    var_8 = {var_5: var_7, var_6: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'email'
    var_1 = False
    var_2 = module_0.String(format=var_0)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'type'
    var_5 = 'format'
    var_6 = 'string'
    var_7 = {var_4: var_6, var_5: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = 100
    var_2 = False
    var_3 = module_0.Integer(exclusive_minimum=var_0, exclusive_maximum=var_1)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'type'
    var_6 = 'exclusiveMinimum'
    var_7 = 'exclusiveMaximum'
    var_8 = 'integer'
    var_9 = {var_5: var_8, var_6: var_2, var_7: var_1}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = False
    var_2 = module_0.Integer(multiple_of=var_0)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'type'
    var_5 = 'multipleOf'
    var_6 = 'integer'
    var_7 = {var_4: var_6, var_5: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Boolean()
    var_2 = False
    var_3 = module_0.Array(var_0, var_1)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'type'
    var_6 = 'items'
    var_7 = 'additionalItems'
    var_8 = 'array'
    var_9 = 'string'
    var_10 = {var_5: var_9}
    var_11 = 'boolean'
    var_12 = {var_5: var_11}
    var_13 = {var_5: var_8, var_6: var_10, var_7: var_12}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = True
    var_2 = False
    var_3 = module_0.Array(var_0, unique_items=var_1)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'type'
    var_6 = 'items'
    var_7 = 'uniqueItems'
    var_8 = 'array'
    var_9 = 'string'
    var_10 = {var_5: var_9}
    var_11 = {var_5: var_8, var_6: var_10, var_7: var_1}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = '^test_'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = module_0.Object(pattern_properties=var_2)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = 'type'
    var_7 = 'patternProperties'
    var_8 = 'object'
    var_9 = 'string'
    var_10 = {var_6: var_9}
    var_11 = {var_0: var_10}
    var_12 = {var_6: var_8, var_7: var_11}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = False
    var_2 = module_0.Object(additional_properties=var_0)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'type'
    var_5 = 'additionalProperties'
    var_6 = 'object'
    var_7 = 'integer'
    var_8 = {var_4: var_7}
    var_9 = {var_4: var_6, var_5: var_8}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = module_0.String(min_length=var_0)
    var_2 = False
    var_3 = module_0.Object(property_names=var_1)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'type'
    var_6 = 'propertyNames'
    var_7 = 'object'
    var_8 = 'minLength'
    var_9 = 'string'
    var_10 = {var_5: var_9, var_8: var_0}
    var_11 = {var_5: var_7, var_6: var_10}

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = [var_0]
    var_4 = False
    var_5 = module_1.Schema(var_2)
    var_6 = module_2.to_json_schema(var_5)
    var_7 = 'type'
    var_8 = 'properties'
    var_9 = 'required'
    var_10 = 'object'
    var_11 = 'string'
    var_12 = {var_7: var_11}
    var_13 = {var_0: var_12}
    var_14 = [var_0]
    var_15 = {var_7: var_10, var_8: var_13, var_9: var_14}



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 11/12 statements.
# Partially parsed test_to_json_schema_with_definitions_nested. Retrieved 18/19 statements.
# Partially parsed test_to_json_schema_with_definitions_multiple. Retrieved 14/16 statements.
# Partially parsed test_to_json_schema_with_definitions_and_field. Retrieved 7/8 statements.
# Partially parsed test_to_json_schema_with_definitions_duplicate. Retrieved 1/4 statements.
# Partially parsed test_to_json_schema_with_definitions_and_root_field. Retrieved 21/22 statements.
# Partially parsed test_to_json_schema_with_definitions_reference_chain. Retrieved 30/33 statements.
# Partially parsed test_to_json_schema_with_definitions_circular. Retrieved 19/20 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.to_json_schema(var_0)
    var_2 = 'components'
    var_3 = 'schemas'
    var_4 = 'TestField'
    var_5 = 'type'
    var_6 = 'string'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = {var_3: var_8}
    var_10 = {var_2: var_9}

import typesystem.schemas as module_0
import typesystem.fields as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'inner'
    var_2 = module_1.String()
    var_3 = {var_1: var_2}
    var_4 = module_2.to_json_schema(var_0)
    var_5 = 'components'
    var_6 = 'schemas'
    var_7 = 'Nested'
    var_8 = 'type'
    var_9 = 'properties'
    var_10 = 'object'
    var_11 = 'string'
    var_12 = {var_8: var_11}
    var_13 = {var_1: var_12}
    var_14 = {var_8: var_10, var_9: var_13}
    var_15 = {var_7: var_14}
    var_16 = {var_6: var_15}
    var_17 = {var_5: var_16}

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.to_json_schema(var_0)
    var_2 = 'components'
    var_3 = 'schemas'
    var_4 = 'Field1'
    var_5 = 'Field2'
    var_6 = 'type'
    var_7 = 'string'
    var_8 = {var_6: var_7}
    var_9 = 'integer'
    var_10 = {var_6: var_9}
    var_11 = {var_4: var_8, var_5: var_10}
    var_12 = {var_3: var_11}
    var_13 = {var_2: var_12}

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'RefField'
    var_2 = module_0.Reference(var_1)
    var_3 = module_1.to_json_schema(var_2, var_0)
    var_4 = '$ref'
    var_5 = '#/components/schemas/RefField'
    var_6 = {var_4: var_5}

import typesystem.schemas as module_0

def test_case_0():
    var_0 = module_0.Definitions()

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.to_json_schema(var_0)
    var_2 = 'components'
    var_3 = 'schemas'
    var_4 = {}
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

import typesystem.schemas as module_0
import typesystem.fields as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'ref'
    var_2 = 'Ref'
    var_3 = module_0.Reference(var_2)
    var_4 = {var_1: var_3}
    var_5 = module_1.Object(properties=var_4)
    var_6 = module_2.to_json_schema(var_5, var_0)
    var_7 = 'type'
    var_8 = 'properties'
    var_9 = 'components'
    var_10 = 'object'
    var_11 = '$ref'
    var_12 = '#/components/schemas/Ref'
    var_13 = {var_11: var_12}
    var_14 = {var_1: var_13}
    var_15 = 'schemas'
    var_16 = 'string'
    var_17 = {var_7: var_16}
    var_18 = {var_2: var_17}
    var_19 = {var_15: var_18}
    var_20 = {var_7: var_10, var_8: var_14, var_9: var_19}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = {}
    var_1 = 'name'
    var_2 = module_0.String()
    var_3 = {var_1: var_2}
    var_4 = module_0.Object(properties=var_3)
    var_5 = module_1.to_json_schema(var_4, var_0)
    var_6 = 'age'
    var_7 = module_0.Integer()
    var_8 = {var_6: var_7}
    var_9 = module_0.Object(properties=var_8)
    var_10 = module_1.to_json_schema(var_9, var_0)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'b'
    var_2 = 'B'
    var_3 = module_0.Reference(var_2)
    var_4 = {var_1: var_3}
    var_5 = 'c'
    var_6 = 'C'
    var_7 = module_0.Reference(var_6)
    var_8 = {var_5: var_7}
    var_9 = module_1.to_json_schema(var_0)
    var_10 = 'components'
    var_11 = 'schemas'
    var_12 = 'A'
    var_13 = 'type'
    var_14 = 'properties'
    var_15 = 'object'
    var_16 = '$ref'
    var_17 = '#/components/schemas/B'
    var_18 = {var_16: var_17}
    var_19 = {var_1: var_18}
    var_20 = {var_13: var_15, var_14: var_19}
    var_21 = '#/components/schemas/C'
    var_22 = {var_16: var_21}
    var_23 = {var_5: var_22}
    var_24 = {var_13: var_15, var_14: var_23}
    var_25 = 'string'
    var_26 = {var_13: var_25}
    var_27 = {var_12: var_20, var_2: var_24, var_6: var_26}
    var_28 = {var_11: var_27}
    var_29 = {var_10: var_28}

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'next'
    var_2 = 'Node'
    var_3 = module_0.Reference(var_2)
    var_4 = {var_1: var_3}
    var_5 = module_1.to_json_schema(var_0)
    var_6 = 'components'
    var_7 = 'schemas'
    var_8 = 'type'
    var_9 = 'properties'
    var_10 = 'object'
    var_11 = '$ref'
    var_12 = '#/components/schemas/Node'
    var_13 = {var_11: var_12}
    var_14 = {var_1: var_13}
    var_15 = {var_8: var_10, var_9: var_14}
    var_16 = {var_2: var_15}
    var_17 = {var_7: var_16}
    var_18 = {var_6: var_17}



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 11/13 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.to_json_schema(var_0)
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.String()
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'string'
    var_5 = 'null'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = module_0.String(min_length=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'minLength'
    var_5 = 'string'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = module_0.String(max_length=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'maxLength'
    var_5 = 'string'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.String(pattern=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'pattern'
    var_5 = 'string'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'email'
    var_1 = module_0.String(format=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'format'
    var_5 = 'string'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_1.to_json_schema(var_0)
    var_2 = 'type'
    var_3 = 'integer'
    var_4 = {var_2: var_3}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Integer()
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'integer'
    var_5 = 'null'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = 100
    var_2 = module_0.Integer(minimum=var_0, maximum=var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'type'
    var_5 = 'minimum'
    var_6 = 'maximum'
    var_7 = 'integer'
    var_8 = {var_4: var_7, var_5: var_0, var_6: var_1}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = 100
    var_2 = module_0.Integer(exclusive_minimum=var_0, exclusive_maximum=var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'type'
    var_5 = 'exclusiveMinimum'
    var_6 = 'exclusiveMaximum'
    var_7 = 'integer'
    var_8 = {var_4: var_7, var_5: var_0, var_6: var_1}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = module_0.Integer(multiple_of=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'multipleOf'
    var_5 = 'integer'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Float()
    var_1 = module_1.to_json_schema(var_0)
    var_2 = 'type'
    var_3 = 'number'
    var_4 = {var_2: var_3}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Float()
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'number'
    var_5 = 'null'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = module_1.to_json_schema(var_0)
    var_2 = 'type'
    var_3 = 'boolean'
    var_4 = {var_2: var_3}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean()
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'boolean'
    var_5 = 'null'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Array()
    var_1 = module_1.to_json_schema(var_0)
    var_2 = 'type'
    var_3 = 'array'
    var_4 = {var_2: var_3}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Array()
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'array'
    var_5 = 'null'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = module_0.Array(min_items=var_0, max_items=var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'type'
    var_5 = 'minItems'
    var_6 = 'maxItems'
    var_7 = 'array'
    var_8 = {var_4: var_7, var_5: var_0, var_6: var_1}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Array(var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'items'
    var_5 = 'array'
    var_6 = 'string'
    var_7 = {var_3: var_6}
    var_8 = {var_3: var_5, var_4: var_7}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Array(additional_items=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'additionalItems'
    var_5 = 'array'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Array(unique_items=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'uniqueItems'
    var_5 = 'array'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Object()
    var_1 = module_1.to_json_schema(var_0)
    var_2 = 'type'
    var_3 = 'object'
    var_4 = {var_2: var_3}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Object()
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'object'
    var_5 = 'null'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.Object(properties=var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'type'
    var_6 = 'properties'
    var_7 = 'object'
    var_8 = 'string'
    var_9 = {var_5: var_8}
    var_10 = {var_0: var_9}
    var_11 = {var_5: var_7, var_6: var_10}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Object(additional_properties=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'additionalProperties'
    var_5 = 'object'
    var_6 = {var_3: var_5, var_4: var_0}

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
    var_8 = 'type'
    var_9 = 'properties'
    var_10 = 'required'
    var_11 = 'object'
    var_12 = 'string'
    var_13 = {var_8: var_12}
    var_14 = 'integer'
    var_15 = {var_8: var_14}
    var_16 = {var_0: var_13, var_1: var_15}
    var_17 = [var_0]
    var_18 = {var_8: var_11, var_9: var_16, var_10: var_17}

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = module_2.to_json_schema(var_3)
    var_5 = 'type'
    var_6 = 'properties'
    var_7 = 'object'
    var_8 = 'string'
    var_9 = {var_5: var_8}
    var_10 = {var_0: var_9}
    var_11 = {var_5: var_7, var_6: var_10}

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
    var_9 = 'enum'
    var_10 = [var_0, var_3]
    var_11 = {var_9: var_10}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed'
    var_1 = module_0.Const(var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'const'
    var_4 = {var_3: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'anyOf'
    var_6 = 'type'
    var_7 = 'string'
    var_8 = {var_6: var_7}
    var_9 = 'integer'
    var_10 = {var_6: var_9}
    var_11 = [var_8, var_10]
    var_12 = {var_5: var_11}

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = module_2.to_json_schema(var_3)
    var_5 = 'oneOf'
    var_6 = 'type'
    var_7 = 'string'
    var_8 = {var_6: var_7}
    var_9 = 'integer'
    var_10 = {var_6: var_9}
    var_11 = [var_8, var_10]
    var_12 = {var_5: var_11}

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = module_2.to_json_schema(var_3)
    var_5 = 'allOf'
    var_6 = 'type'
    var_7 = 'string'
    var_8 = {var_6: var_7}
    var_9 = 'integer'
    var_10 = {var_6: var_9}
    var_11 = [var_8, var_10]
    var_12 = {var_5: var_11}

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = module_1.IfThenElse(var_0, var_1)
    var_3 = module_2.to_json_schema(var_2)
    var_4 = 'if'
    var_5 = 'then'
    var_6 = 'type'
    var_7 = 'string'
    var_8 = {var_6: var_7}
    var_9 = 'integer'
    var_10 = {var_6: var_9}
    var_11 = {var_4: var_8, var_5: var_10}

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Not(var_0)
    var_2 = module_2.to_json_schema(var_1)
    var_3 = 'not'
    var_4 = 'type'
    var_5 = 'string'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'MyString'
    var_2 = module_1.Reference(var_1)
    var_3 = module_2.to_json_schema(var_2)
    var_4 = '$ref'
    var_5 = 'components'
    var_6 = '#/components/schemas/MyString'
    var_7 = 'schemas'
    var_8 = 'type'
    var_9 = 'string'
    var_10 = {var_8: var_9}
    var_11 = {var_1: var_10}
    var_12 = {var_7: var_11}
    var_13 = {var_4: var_6, var_5: var_12}

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'MyString'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = 'components'
    var_4 = 'schemas'
    var_5 = 'type'
    var_6 = 'string'
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_7}
    var_9 = {var_4: var_8}
    var_10 = {var_3: var_9}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'hello'
    var_1 = module_0.String()
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'default'
    var_5 = 'string'
    var_6 = {var_3: var_5, var_4: var_0}



# Parsed testcases at query #18
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.String()
    var_2 = module_1.to_json_schema(var_1)



# Parsed testcases at query #19
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.to_json_schema(var_0)
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.String()
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'string'
    var_5 = 'null'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = module_0.String(min_length=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'minLength'
    var_5 = 'string'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = module_0.String(max_length=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'maxLength'
    var_5 = 'string'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.String(pattern=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'pattern'
    var_5 = 'string'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'email'
    var_1 = module_0.String(format=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'format'
    var_5 = 'string'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_1.to_json_schema(var_0)
    var_2 = 'type'
    var_3 = 'integer'
    var_4 = {var_2: var_3}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Integer()
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'integer'
    var_5 = 'null'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = 100
    var_2 = module_0.Integer(minimum=var_0, maximum=var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'type'
    var_5 = 'minimum'
    var_6 = 'maximum'
    var_7 = 'integer'
    var_8 = {var_4: var_7, var_5: var_0, var_6: var_1}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = 100
    var_2 = module_0.Integer(exclusive_minimum=var_0, exclusive_maximum=var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'type'
    var_5 = 'exclusiveMinimum'
    var_6 = 'exclusiveMaximum'
    var_7 = 'integer'
    var_8 = {var_4: var_7, var_5: var_0, var_6: var_1}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = module_0.Integer(multiple_of=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'multipleOf'
    var_5 = 'integer'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Float()
    var_1 = module_1.to_json_schema(var_0)
    var_2 = 'type'
    var_3 = 'number'
    var_4 = {var_2: var_3}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Float()
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'number'
    var_5 = 'null'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = module_1.to_json_schema(var_0)
    var_2 = 'type'
    var_3 = 'boolean'
    var_4 = {var_2: var_3}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean()
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'boolean'
    var_5 = 'null'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Array()
    var_1 = module_1.to_json_schema(var_0)
    var_2 = 'type'
    var_3 = 'array'
    var_4 = {var_2: var_3}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Array()
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'array'
    var_5 = 'null'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = module_0.Array(min_items=var_0, max_items=var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'type'
    var_5 = 'minItems'
    var_6 = 'maxItems'
    var_7 = 'array'
    var_8 = {var_4: var_7, var_5: var_0, var_6: var_1}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Array(var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'items'
    var_5 = 'array'
    var_6 = 'string'
    var_7 = {var_3: var_6}
    var_8 = {var_3: var_5, var_4: var_7}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Array(var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'type'
    var_6 = 'items'
    var_7 = 'array'
    var_8 = 'string'
    var_9 = {var_5: var_8}
    var_10 = 'integer'
    var_11 = {var_5: var_10}
    var_12 = [var_9, var_11]
    var_13 = {var_5: var_7, var_6: var_12}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Array(additional_items=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'additionalItems'
    var_5 = 'array'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Array(unique_items=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'uniqueItems'
    var_5 = 'array'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Object()
    var_1 = module_1.to_json_schema(var_0)
    var_2 = 'type'
    var_3 = 'object'
    var_4 = {var_2: var_3}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Object()
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'object'
    var_5 = 'null'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}

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
    var_7 = 'type'
    var_8 = 'properties'
    var_9 = 'object'
    var_10 = 'string'
    var_11 = {var_7: var_10}
    var_12 = 'integer'
    var_13 = {var_7: var_12}
    var_14 = {var_0: var_11, var_1: var_13}
    var_15 = {var_7: var_9, var_8: var_14}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Object(additional_properties=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'additionalProperties'
    var_5 = 'object'
    var_6 = {var_3: var_5, var_4: var_0}

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
    var_8 = 'type'
    var_9 = 'properties'
    var_10 = 'required'
    var_11 = 'object'
    var_12 = 'string'
    var_13 = {var_8: var_12}
    var_14 = 'integer'
    var_15 = {var_8: var_14}
    var_16 = {var_0: var_13, var_1: var_15}
    var_17 = [var_0]
    var_18 = {var_8: var_11, var_9: var_16, var_10: var_17}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = module_0.Object(min_properties=var_0, max_properties=var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'type'
    var_5 = 'minProperties'
    var_6 = 'maxProperties'
    var_7 = 'object'
    var_8 = {var_4: var_7, var_5: var_0, var_6: var_1}

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
    var_7 = 'type'
    var_8 = 'properties'
    var_9 = 'object'
    var_10 = 'string'
    var_11 = {var_7: var_10}
    var_12 = 'integer'
    var_13 = {var_7: var_12}
    var_14 = {var_0: var_11, var_1: var_13}
    var_15 = {var_7: var_9, var_8: var_14}

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_0]
    var_6 = module_1.Schema(var_4)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = 'type'
    var_9 = 'properties'
    var_10 = 'required'
    var_11 = 'object'
    var_12 = 'string'
    var_13 = {var_8: var_12}
    var_14 = 'integer'
    var_15 = {var_8: var_14}
    var_16 = {var_0: var_13, var_1: var_15}
    var_17 = [var_0]
    var_18 = {var_8: var_11, var_9: var_16, var_10: var_17}

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
    var_9 = 'enum'
    var_10 = [var_0, var_3]
    var_11 = {var_9: var_10}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed'
    var_1 = module_0.Const(var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'const'
    var_4 = {var_3: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'anyOf'
    var_6 = 'type'
    var_7 = 'string'
    var_8 = {var_6: var_7}
    var_9 = 'integer'
    var_10 = {var_6: var_9}
    var_11 = [var_8, var_10]
    var_12 = {var_5: var_11}

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = module_2.to_json_schema(var_3)
    var_5 = 'oneOf'
    var_6 = 'type'
    var_7 = 'string'
    var_8 = {var_6: var_7}
    var_9 = 'integer'
    var_10 = {var_6: var_9}
    var_11 = [var_8, var_10]
    var_12 = {var_5: var_11}

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 1
    var_1 = module_0.String(min_length=var_0)
    var_2 = 10
    var_3 = module_0.String(max_length=var_2)
    var_4 = [var_1, var_3]
    var_5 = module_1.AllOf(var_4)
    var_6 = module_2.to_json_schema(var_5)
    var_7 = 'allOf'
    var_8 = 'type'
    var_9 = 'minLength'
    var_10 = 'string'
    var_11 = {var_8: var_10, var_9: var_0}
    var_12 = 'maxLength'
    var_13 = {var_8: var_10, var_12: var_2}
    var_14 = [var_11, var_13]
    var_15 = {var_7: var_14}



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 2/3 statements.
# Partially parsed test_to_json_schema_reference_field. Retrieved 5/6 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.to_json_schema(var_0)

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
    var_0 = 'hello'
    var_1 = module_0.String()
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
    var_0 = module_0.String()
    var_1 = module_0.Array(var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = True
    var_2 = module_0.Array(var_0)
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
    var_3 = True
    var_4 = module_0.Object(properties=var_2)
    var_5 = module_1.to_json_schema(var_4)

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
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = var_0 | var_1
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'anyOf'
    var_5 = var_3[var_4]
    var_6 = len(var_5)
    assert var_6 == 2

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.String()
    var_2 = module_0.Integer()
    var_3 = var_1 | var_2
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'anyOf'
    var_6 = var_4[var_5]
    var_7 = len(var_6)
    assert var_7 == 2

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
    var_0 = 'fixed'
    var_1 = module_0.Const(var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.schemas as module_0
import typesystem.fields as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'User'
    var_2 = module_1.String()
    var_3 = module_0.Reference(var_1)
    var_4 = module_2.to_json_schema(var_3, var_0)

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
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = [var_0]
    var_4 = module_1.Schema(var_2)
    var_5 = module_2.to_json_schema(var_4)

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
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = module_2.to_json_schema(var_3)
    var_5 = 'allOf'
    var_6 = var_4[var_5]
    var_7 = len(var_6)
    assert var_7 == 2

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = module_2.to_json_schema(var_3)
    var_5 = 'oneOf'
    var_6 = var_4[var_5]
    var_7 = len(var_6)
    assert var_7 == 2

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
    var_0 = 'email'
    var_1 = module_0.String(format=var_0)
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.String(pattern=var_0)
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



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_from_json_schema_with_boolean_true. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_with_boolean_false. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_with_ref. Retrieved 5/7 statements.
# Partially parsed test_from_json_schema_with_type_string. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_type_integer. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_type_number. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_type_boolean. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_type_array. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_type_object. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_nullable_type. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_with_enum. Retrieved 7/8 statements.
# Partially parsed test_from_json_schema_with_const. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_allOf. Retrieved 12/13 statements.
# Partially parsed test_from_json_schema_with_anyOf. Retrieved 11/12 statements.
# Partially parsed test_from_json_schema_with_oneOf. Retrieved 11/12 statements.
# Partially parsed test_from_json_schema_with_not. Retrieved 7/9 statements.
# Partially parsed test_from_json_schema_with_if_then_else. Retrieved 16/20 statements.
# Partially parsed test_from_json_schema_with_multiple_constraints. Retrieved 11/12 statements.
# Partially parsed test_from_json_schema_with_no_constraints. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_with_components_schemas. Retrieved 11/13 statements.
# Partially parsed test_from_json_schema_with_allow_null_true. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_with_allow_null_false. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_with_array_items. Retrieved 8/10 statements.
# Partially parsed test_from_json_schema_with_object_properties. Retrieved 10/12 statements.
# Partially parsed test_from_json_schema_with_union_types. Retrieved 12/15 statements.


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
    var_2 = '#/components/schemas/Test'
    var_3 = {var_1: var_2}
    var_4 = module_1.from_json_schema(var_3, var_0)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'integer'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'number'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'boolean'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'array'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'object'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = 'null'
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

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'allOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'minLength'
    var_5 = 5
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
    var_6 = var_5.negated

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
    var_8 = {var_6: var_7}
    var_9 = 'integer'
    var_10 = {var_3: var_9}
    var_11 = {var_0: var_5, var_1: var_8, var_2: var_10}
    var_12 = module_0.from_json_schema(var_11)
    var_13 = var_12.if_clause
    var_14 = var_12.then_clause
    var_15 = var_12.else_clause

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
    var_9 = var_8.all_of
    var_10 = len(var_9)
    assert var_10 == 3

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.from_json_schema(var_0)

import typesystem.json_schema as module_0
import typesystem.schemas as module_1

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
    var_10 = module_1.Definitions()

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'nullable'
    var_2 = 'string'
    var_3 = True
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.from_json_schema(var_4)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'nullable'
    var_2 = 'string'
    var_3 = False
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.from_json_schema(var_4)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'default'
    var_2 = 'string'
    var_3 = 'hello'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.from_json_schema(var_4)

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'items'
    var_2 = 'array'
    var_3 = 'string'
    var_4 = {var_0: var_3}
    var_5 = {var_0: var_2, var_1: var_4}
    var_6 = module_0.from_json_schema(var_5)
    var_7 = var_6.items

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
    var_8 = module_0.from_json_schema(var_7)
    var_9 = var_8.properties[var_3]

import typesystem.json_schema as module_0

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = 'integer'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.from_json_schema(var_4)
    var_6 = var_5.any_of
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = 0
    var_9 = var_5.any_of[var_8]
    var_10 = 1
    var_11 = var_5.any_of[var_10]



# Parsed testcases at query #3
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = '^test.*'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.Object(pattern_properties=var_2)
    var_4 = module_1.to_json_schema(var_3)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_if_then_else_from_json_schema_with_ref. Retrieved 22/23 statements.


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
    var_12 = module_1.if_then_else_from_json_schema(var_11, var_0)
    var_13 = 'hello'
    var_14 = var_12.validate(var_13)
    assert var_14 == 'hello'
    var_15 = 123
    var_16 = var_12.validate(var_15)
    assert var_16 == 123
    var_17 = True
    var_18 = var_12.validate(var_17)
    assert var_18 is True

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'if'
    var_2 = 'else'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 'boolean'
    var_7 = {var_3: var_6}
    var_8 = {var_1: var_5, var_2: var_7}
    var_9 = module_1.if_then_else_from_json_schema(var_8, var_0)
    var_10 = 'hello'
    var_11 = var_9.validate(var_10)
    assert var_11 == 'hello'
    var_12 = True
    var_13 = var_9.validate(var_12)
    assert var_13 is True

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 'integer'
    var_7 = {var_3: var_6}
    var_8 = {var_1: var_5, var_2: var_7}
    var_9 = module_1.if_then_else_from_json_schema(var_8, var_0)
    var_10 = 'hello'
    var_11 = var_9.validate(var_10)
    assert var_11 == 'hello'
    var_12 = 123
    var_13 = var_9.validate(var_12)
    assert var_13 == 123

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
    var_8 = 'integer'
    var_9 = {var_5: var_8}
    var_10 = 'boolean'
    var_11 = {var_5: var_10}
    var_12 = 42
    var_13 = {var_1: var_7, var_2: var_9, var_3: var_11, var_4: var_12}
    var_14 = module_1.if_then_else_from_json_schema(var_13, var_0)
    var_15 = var_14.has_default()

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'if'
    var_5 = 'then'
    var_6 = 'else'
    var_7 = '$ref'
    var_8 = '#/components/schemas/StringType'
    var_9 = {var_7: var_8}
    var_10 = 'integer'
    var_11 = {var_1: var_10}
    var_12 = 'boolean'
    var_13 = {var_1: var_12}
    var_14 = {var_4: var_9, var_5: var_11, var_6: var_13}
    var_15 = module_1.if_then_else_from_json_schema(var_14, var_0)
    var_16 = 'hello'
    var_17 = var_15.validate(var_16)
    assert var_17 == 'hello'
    var_18 = 123
    var_19 = var_15.validate(var_18)
    assert var_19 == 123
    var_20 = True
    var_21 = var_15.validate(var_20)
    assert var_21 is True

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'else'
    var_4 = 'const'
    var_5 = 'yes'
    var_6 = {var_4: var_5}
    var_7 = 'type'
    var_8 = 'integer'
    var_9 = {var_7: var_8}
    var_10 = 'boolean'
    var_11 = {var_7: var_10}
    var_12 = {var_1: var_6, var_2: var_9, var_3: var_11}
    var_13 = module_1.if_then_else_from_json_schema(var_12, var_0)
    var_14 = var_13.validate(var_5)
    assert var_14 == 'yes'
    var_15 = 'no'
    var_16 = var_13.validate(var_15)
    assert var_16 is True

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'else'
    var_4 = 'enum'
    var_5 = 'yes'
    var_6 = 'no'
    var_7 = [var_5, var_6]
    var_8 = {var_4: var_7}
    var_9 = 'type'
    var_10 = 'integer'
    var_11 = {var_9: var_10}
    var_12 = 'boolean'
    var_13 = {var_9: var_12}
    var_14 = {var_1: var_8, var_2: var_11, var_3: var_13}
    var_15 = module_1.if_then_else_from_json_schema(var_14, var_0)
    var_16 = var_15.validate(var_5)
    assert var_16 == 'yes'
    var_17 = 'maybe'
    var_18 = var_15.validate(var_17)
    assert var_18 is True

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'else'
    var_4 = 'allOf'
    var_5 = 'type'
    var_6 = 'string'
    var_7 = {var_5: var_6}
    var_8 = 'maxLength'
    var_9 = 5
    var_10 = {var_8: var_9}
    var_11 = [var_7, var_10]
    var_12 = {var_4: var_11}
    var_13 = 'integer'
    var_14 = {var_5: var_13}
    var_15 = 'boolean'
    var_16 = {var_5: var_15}
    var_17 = {var_1: var_12, var_2: var_14, var_3: var_16}
    var_18 = module_1.if_then_else_from_json_schema(var_17, var_0)
    var_19 = 'hello'
    var_20 = var_18.validate(var_19)
    assert var_20 == 'hello'
    var_21 = 'toolong'
    var_22 = var_18.validate(var_21)
    assert var_22 is True

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'else'
    var_4 = 'anyOf'
    var_5 = 'type'
    var_6 = 'string'
    var_7 = {var_5: var_6}
    var_8 = 'integer'
    var_9 = {var_5: var_8}
    var_10 = [var_7, var_9]
    var_11 = {var_4: var_10}
    var_12 = 'boolean'
    var_13 = {var_5: var_12}
    var_14 = 'null'
    var_15 = {var_5: var_14}
    var_16 = {var_1: var_11, var_2: var_13, var_3: var_15}
    var_17 = module_1.if_then_else_from_json_schema(var_16, var_0)
    var_18 = 'hello'
    var_19 = var_17.validate(var_18)
    assert var_19 is True
    var_20 = 123
    var_21 = var_17.validate(var_20)
    assert var_21 is True
    var_22 = True
    var_23 = var_17.validate(var_22)
    assert var_23 is True

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'else'
    var_4 = 'oneOf'
    var_5 = 'type'
    var_6 = 'string'
    var_7 = {var_5: var_6}
    var_8 = 'integer'
    var_9 = {var_5: var_8}
    var_10 = [var_7, var_9]
    var_11 = {var_4: var_10}
    var_12 = 'boolean'
    var_13 = {var_5: var_12}
    var_14 = 'null'
    var_15 = {var_5: var_14}
    var_16 = {var_1: var_11, var_2: var_13, var_3: var_15}
    var_17 = module_1.if_then_else_from_json_schema(var_16, var_0)
    var_18 = 'hello'
    var_19 = var_17.validate(var_18)
    assert var_19 is True
    var_20 = 123
    var_21 = var_17.validate(var_20)
    assert var_21 is True
    var_22 = True
    var_23 = var_17.validate(var_22)
    assert var_23 is True

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'else'
    var_4 = 'not'
    var_5 = 'type'
    var_6 = 'string'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = 'integer'
    var_10 = {var_5: var_9}
    var_11 = 'boolean'
    var_12 = {var_5: var_11}
    var_13 = {var_1: var_8, var_2: var_10, var_3: var_12}
    var_14 = module_1.if_then_else_from_json_schema(var_13, var_0)
    var_15 = 123
    var_16 = var_14.validate(var_15)
    assert var_16 == 123
    var_17 = 'hello'
    var_18 = var_14.validate(var_17)
    assert var_18 is True

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
    var_7 = 'maxLength'
    var_8 = 5
    var_9 = {var_7: var_8}
    var_10 = 'integer'
    var_11 = {var_4: var_10}
    var_12 = 'boolean'
    var_13 = {var_4: var_12}
    var_14 = {var_1: var_9, var_2: var_11, var_3: var_13}
    var_15 = 'null'
    var_16 = {var_4: var_15}
    var_17 = {var_1: var_6, var_2: var_14, var_3: var_16}
    var_18 = module_1.if_then_else_from_json_schema(var_17, var_0)
    var_19 = 'hello'
    var_20 = var_18.validate(var_19)
    assert var_20 == 'hello'
    var_21 = 'toolong'
    var_22 = var_18.validate(var_21)
    assert var_22 is True
    var_23 = 123
    var_24 = var_18.validate(var_23)
    assert var_24 == 123



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_from_json_schema_type_number. Retrieved 15/16 statements.
# Partially parsed test_from_json_schema_type_integer. Retrieved 15/16 statements.
# Partially parsed test_from_json_schema_type_string. Retrieved 15/16 statements.
# Partially parsed test_from_json_schema_type_string_allow_blank. Retrieved 15/16 statements.
# Partially parsed test_from_json_schema_type_boolean. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_type_array. Retrieved 20/22 statements.
# Partially parsed test_from_json_schema_type_array_with_list_items. Retrieved 26/30 statements.
# Partially parsed test_from_json_schema_type_array_with_additional_items. Retrieved 23/26 statements.
# Partially parsed test_from_json_schema_type_array_with_additional_items_bool. Retrieved 21/23 statements.
# Partially parsed test_from_json_schema_type_object. Retrieved 26/29 statements.
# Partially parsed test_from_json_schema_type_object_with_pattern_properties. Retrieved 20/22 statements.
# Partially parsed test_from_json_schema_type_object_with_additional_properties. Retrieved 18/20 statements.
# Partially parsed test_from_json_schema_type_object_with_additional_properties_bool. Retrieved 12/13 statements.
# Partially parsed test_from_json_schema_type_object_with_property_names. Retrieved 20/22 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

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
    var_11 = 'number'
    var_12 = False
    var_13 = module_0.Definitions()
    var_14 = module_1.from_json_schema_type(var_10, var_11, var_12, var_13)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

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
    var_11 = 'integer'
    var_12 = True
    var_13 = module_0.Definitions()
    var_14 = module_1.from_json_schema_type(var_10, var_11, var_12, var_13)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

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
    var_11 = 'string'
    var_12 = False
    var_13 = module_0.Definitions()
    var_14 = module_1.from_json_schema_type(var_10, var_11, var_12, var_13)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'minLength'
    var_1 = 'maxLength'
    var_2 = 'format'
    var_3 = 'pattern'
    var_4 = 'default'
    var_5 = 0
    var_6 = 10
    var_7 = 'email'
    var_8 = '^a.*z$'
    var_9 = ''
    var_10 = {var_0: var_5, var_1: var_6, var_2: var_7, var_3: var_8, var_4: var_9}
    var_11 = 'string'
    var_12 = False
    var_13 = module_0.Definitions()
    var_14 = module_1.from_json_schema_type(var_10, var_11, var_12, var_13)

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
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = 'items'
    var_4 = 'minItems'
    var_5 = 'maxItems'
    var_6 = 'uniqueItems'
    var_7 = 'default'
    var_8 = 1
    var_9 = 10
    var_10 = True
    var_11 = 'a'
    var_12 = 'b'
    var_13 = [var_11, var_12]
    var_14 = {var_3: var_2, var_4: var_8, var_5: var_9, var_6: var_10, var_7: var_13}
    var_15 = 'array'
    var_16 = False
    var_17 = module_0.Definitions()
    var_18 = module_1.from_json_schema_type(var_14, var_15, var_16, var_17)
    var_19 = var_18.items

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = 'integer'
    var_4 = {var_0: var_3}
    var_5 = [var_2, var_4]
    var_6 = 'items'
    var_7 = 'minItems'
    var_8 = 'maxItems'
    var_9 = 'uniqueItems'
    var_10 = 'default'
    var_11 = 2
    var_12 = False
    var_13 = 'a'
    var_14 = 1
    var_15 = [var_13, var_14]
    var_16 = {var_6: var_5, var_7: var_11, var_8: var_11, var_9: var_12, var_10: var_15}
    var_17 = 'array'
    var_18 = True
    var_19 = module_0.Definitions()
    var_20 = module_1.from_json_schema_type(var_16, var_17, var_18, var_19)
    var_21 = var_20.items
    var_22 = var_20.items
    var_23 = len(var_22)
    assert var_23 == 2
    var_24 = var_20.items[var_12]
    var_25 = var_20.items[var_18]

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = 'integer'
    var_4 = {var_0: var_3}
    var_5 = 'items'
    var_6 = 'additionalItems'
    var_7 = 'minItems'
    var_8 = 'maxItems'
    var_9 = 'uniqueItems'
    var_10 = 'default'
    var_11 = 1
    var_12 = 5
    var_13 = False
    var_14 = 'a'
    var_15 = 2
    var_16 = [var_14, var_11, var_15]
    var_17 = {var_5: var_2, var_6: var_4, var_7: var_11, var_8: var_12, var_9: var_13, var_10: var_16}
    var_18 = 'array'
    var_19 = module_0.Definitions()
    var_20 = module_1.from_json_schema_type(var_17, var_18, var_13, var_19)
    var_21 = var_20.items
    var_22 = var_20.additional_items

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = 'items'
    var_4 = 'additionalItems'
    var_5 = 'minItems'
    var_6 = 'maxItems'
    var_7 = 'uniqueItems'
    var_8 = 'default'
    var_9 = False
    var_10 = 1
    var_11 = 5
    var_12 = 'a'
    var_13 = 'b'
    var_14 = [var_12, var_13]
    var_15 = {var_3: var_2, var_4: var_9, var_5: var_10, var_6: var_11, var_7: var_9, var_8: var_14}
    var_16 = 'array'
    var_17 = True
    var_18 = module_0.Definitions()
    var_19 = module_1.from_json_schema_type(var_15, var_16, var_17, var_18)
    var_20 = var_19.items

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = 'integer'
    var_6 = {var_2: var_5}
    var_7 = {var_0: var_4, var_1: var_6}
    var_8 = 'properties'
    var_9 = 'minProperties'
    var_10 = 'maxProperties'
    var_11 = 'required'
    var_12 = 'default'
    var_13 = 1
    var_14 = 2
    var_15 = [var_0]
    var_16 = 'John'
    var_17 = 30
    var_18 = {var_0: var_16, var_1: var_17}
    var_19 = {var_8: var_7, var_9: var_13, var_10: var_14, var_11: var_15, var_12: var_18}
    var_20 = 'object'
    var_21 = False
    var_22 = module_0.Definitions()
    var_23 = module_1.from_json_schema_type(var_19, var_20, var_21, var_22)
    var_24 = var_23.properties[var_0]
    var_25 = var_23.properties[var_1]

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = '^S_'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'patternProperties'
    var_6 = 'minProperties'
    var_7 = 'maxProperties'
    var_8 = 'default'
    var_9 = 0
    var_10 = 5
    var_11 = 'S_key'
    var_12 = 'value'
    var_13 = {var_11: var_12}
    var_14 = {var_5: var_4, var_6: var_9, var_7: var_10, var_8: var_13}
    var_15 = 'object'
    var_16 = True
    var_17 = module_0.Definitions()
    var_18 = module_1.from_json_schema_type(var_14, var_15, var_16, var_17)
    var_19 = var_18.pattern_properties[var_0]

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'boolean'
    var_2 = {var_0: var_1}
    var_3 = 'additionalProperties'
    var_4 = 'minProperties'
    var_5 = 'maxProperties'
    var_6 = 'default'
    var_7 = 0
    var_8 = 5
    var_9 = 'extra'
    var_10 = True
    var_11 = {var_9: var_10}
    var_12 = {var_3: var_2, var_4: var_7, var_5: var_8, var_6: var_11}
    var_13 = 'object'
    var_14 = False
    var_15 = module_0.Definitions()
    var_16 = module_1.from_json_schema_type(var_12, var_13, var_14, var_15)
    var_17 = var_16.additional_properties

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'additionalProperties'
    var_1 = 'minProperties'
    var_2 = 'maxProperties'
    var_3 = 'default'
    var_4 = False
    var_5 = 5
    var_6 = {}
    var_7 = {var_0: var_4, var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = 'object'
    var_9 = True
    var_10 = module_0.Definitions()
    var_11 = module_1.from_json_schema_type(var_7, var_8, var_9, var_10)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'pattern'
    var_2 = 'string'
    var_3 = '^[a-z]+$'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'propertyNames'
    var_6 = 'minProperties'
    var_7 = 'maxProperties'
    var_8 = 'default'
    var_9 = 0
    var_10 = 5
    var_11 = 'key'
    var_12 = 'value'
    var_13 = {var_11: var_12}
    var_14 = {var_5: var_4, var_6: var_9, var_7: var_10, var_8: var_13}
    var_15 = 'object'
    var_16 = False
    var_17 = module_0.Definitions()
    var_18 = module_1.from_json_schema_type(var_14, var_15, var_16, var_17)
    var_19 = var_18.property_names



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_ref_from_json_schema_creates_reference_with_correct_to. Retrieved 5/6 statements.
# Partially parsed test_ref_from_json_schema_works_with_minimal_ref. Retrieved 5/6 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = '#/components/schemas/User'
    var_3 = {var_1: var_2}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = 'http://example.com/schema'
    var_3 = {var_1: var_2}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = '#/User'
    var_3 = {var_1: var_2}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 11/13 statements.
# Partially parsed test_to_json_schema_with_definitions_nested. Retrieved 15/19 statements.
# Partially parsed test_to_json_schema_with_definitions_duplicate_key. Retrieved 11/14 statements.
# Partially parsed test_to_json_schema_with_definitions_and_field. Retrieved 7/9 statements.
# Partially parsed test_to_json_schema_with_definitions_root. Retrieved 3/5 statements.
# Partially parsed test_to_json_schema_with_definitions_non_root. Retrieved 4/6 statements.
# Partially parsed test_to_json_schema_with_definitions_multiple_keys. Retrieved 15/17 statements.
# Partially parsed test_to_json_schema_with_definitions_reference. Retrieved 11/13 statements.
# Partially parsed test_to_json_schema_with_definitions_complex_field. Retrieved 19/21 statements.
# Partially parsed test_to_json_schema_with_definitions_and_standard_properties. Retrieved 13/15 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'User'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = 'components'
    var_4 = 'schemas'
    var_5 = 'type'
    var_6 = 'string'
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_7}
    var_9 = {var_4: var_8}
    var_10 = {var_3: var_9}

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Inner'
    var_1 = module_0.Integer()
    var_2 = {var_0: var_1}
    var_3 = 'Outer'
    var_4 = 'components'
    var_5 = 'schemas'
    var_6 = 'type'
    var_7 = 'integer'
    var_8 = {var_6: var_7}
    var_9 = {var_0: var_8}
    var_10 = {var_5: var_9}
    var_11 = {var_4: var_10}
    var_12 = {var_3: var_11}
    var_13 = {var_5: var_12}
    var_14 = {var_4: var_13}

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'User'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = 'components'
    var_4 = 'schemas'
    var_5 = 'type'
    var_6 = 'integer'
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_7}
    var_9 = {var_4: var_8}
    var_10 = {var_3: var_9}

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.to_json_schema(var_0)
    var_2 = {}

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'User'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.Integer()
    var_4 = 'type'
    var_5 = 'integer'
    var_6 = {var_4: var_5}

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'User'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'User'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = {}

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'User'
    var_1 = 'Post'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'components'
    var_6 = 'schemas'
    var_7 = 'type'
    var_8 = 'string'
    var_9 = {var_7: var_8}
    var_10 = 'integer'
    var_11 = {var_7: var_10}
    var_12 = {var_0: var_9, var_1: var_11}
    var_13 = {var_6: var_12}
    var_14 = {var_5: var_13}

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'User'
    var_2 = {var_1: var_0}
    var_3 = 'components'
    var_4 = 'schemas'
    var_5 = 'type'
    var_6 = 'string'
    var_7 = {var_5: var_6}
    var_8 = {var_1: var_7}
    var_9 = {var_4: var_8}
    var_10 = {var_3: var_9}

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = 'Complex'
    var_5 = {var_4: var_3}
    var_6 = 'components'
    var_7 = 'schemas'
    var_8 = 'anyOf'
    var_9 = 'type'
    var_10 = 'string'
    var_11 = {var_9: var_10}
    var_12 = 'integer'
    var_13 = {var_9: var_12}
    var_14 = [var_11, var_13]
    var_15 = {var_8: var_14}
    var_16 = {var_4: var_15}
    var_17 = {var_7: var_16}
    var_18 = {var_6: var_17}

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.String()
    var_2 = 'Field'
    var_3 = {var_2: var_1}
    var_4 = 'components'
    var_5 = 'schemas'
    var_6 = 'type'
    var_7 = 'default'
    var_8 = 'string'
    var_9 = {var_6: var_8, var_7: var_0}
    var_10 = {var_2: var_9}
    var_11 = {var_5: var_10}
    var_12 = {var_4: var_11}



# Parsed testcases at query #8
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.String()
    var_2 = None
    var_3 = module_1.IfThenElse(var_0, var_1, var_2)
    var_4 = module_2.to_json_schema(var_3)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_predicate_at_line_120_evaluates_to_true. Retrieved 8/10 statements.


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Schema(var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = False
    var_5 = {}
    var_6 = module_0.Schema(var_5)
    var_7 = module_1.to_json_schema(var_6)



# Parsed testcases at query #10
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean()
    var_2 = module_1.to_json_schema(var_1)



# Parsed testcases at query #11
#--------------------------




import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = None
    var_2 = module_1.IfThenElse(var_0, var_1, var_1)
    var_3 = module_2.to_json_schema(var_2)



# Parsed testcases at query #12
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Object()
    var_2 = module_1.to_json_schema(var_1)



# Parsed testcases at query #13
#--------------------------




import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Object()
    var_2 = module_1.to_json_schema(var_1)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 11/13 statements.
# Partially parsed test_to_json_schema_with_definitions_nested. Retrieved 18/20 statements.
# Partially parsed test_to_json_schema_with_definitions_multiple. Retrieved 15/17 statements.
# Partially parsed test_to_json_schema_with_definitions_and_field. Retrieved 7/9 statements.
# Partially parsed test_to_json_schema_with_definitions_reference. Retrieved 10/12 statements.
# Partially parsed test_to_json_schema_with_definitions_union. Retrieved 15/17 statements.
# Partially parsed test_to_json_schema_with_definitions_object. Retrieved 16/18 statements.
# Partially parsed test_to_json_schema_with_definitions_array. Retrieved 13/15 statements.
# Partially parsed test_to_json_schema_with_definitions_choice. Retrieved 14/16 statements.
# Partially parsed test_to_json_schema_with_definitions_const. Retrieved 7/9 statements.
# Partially parsed test_to_json_schema_with_definitions_one_of. Retrieved 15/17 statements.
# Partially parsed test_to_json_schema_with_definitions_all_of. Retrieved 16/18 statements.
# Partially parsed test_to_json_schema_with_definitions_if_then_else. Retrieved 18/20 statements.
# Partially parsed test_to_json_schema_with_definitions_not. Retrieved 10/12 statements.
# Partially parsed test_to_json_schema_with_definitions_schema. Retrieved 14/16 statements.
# Partially parsed test_to_json_schema_with_definitions_allow_null. Retrieved 14/16 statements.
# Partially parsed test_to_json_schema_with_definitions_default. Retrieved 13/15 statements.
# Partially parsed test_to_json_schema_with_definitions_min_length. Retrieved 13/15 statements.
# Partially parsed test_to_json_schema_with_definitions_max_length. Retrieved 13/15 statements.
# Partially parsed test_to_json_schema_with_definitions_pattern. Retrieved 13/15 statements.
# Partially parsed test_to_json_schema_with_definitions_format. Retrieved 13/15 statements.
# Partially parsed test_to_json_schema_with_definitions_integer. Retrieved 15/17 statements.
# Partially parsed test_to_json_schema_with_definitions_float. Retrieved 15/17 statements.
# Partially parsed test_to_json_schema_with_definitions_boolean. Retrieved 13/15 statements.
# Partially parsed test_to_json_schema_with_definitions_array_min_max. Retrieved 15/17 statements.
# Partially parsed test_to_json_schema_with_definitions_array_unique. Retrieved 13/15 statements.
# Partially parsed test_to_json_schema_with_definitions_object_additional_props. Retrieved 4/6 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'User'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = 'components'
    var_4 = 'schemas'
    var_5 = 'type'
    var_6 = 'string'
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_7}
    var_9 = {var_4: var_8}
    var_10 = {var_3: var_9}

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'User'
    var_1 = 'name'
    var_2 = module_0.String()
    var_3 = {var_1: var_2}
    var_4 = module_0.Object(properties=var_3)
    var_5 = {var_0: var_4}
    var_6 = 'components'
    var_7 = 'schemas'
    var_8 = 'type'
    var_9 = 'properties'
    var_10 = 'object'
    var_11 = 'string'
    var_12 = {var_8: var_11}
    var_13 = {var_1: var_12}
    var_14 = {var_8: var_10, var_9: var_13}
    var_15 = {var_0: var_14}
    var_16 = {var_7: var_15}
    var_17 = {var_6: var_16}

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'User'
    var_1 = 'Age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'components'
    var_6 = 'schemas'
    var_7 = 'type'
    var_8 = 'string'
    var_9 = {var_7: var_8}
    var_10 = 'integer'
    var_11 = {var_7: var_10}
    var_12 = {var_0: var_9, var_1: var_11}
    var_13 = {var_6: var_12}
    var_14 = {var_5: var_13}

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = module_1.to_json_schema(var_0)
    var_2 = {}

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'User'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.Integer()
    var_4 = 'type'
    var_5 = 'integer'
    var_6 = {var_4: var_5}

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.Object(properties=var_2)
    var_4 = 'User'
    var_5 = {var_4: var_3}
    var_6 = module_1.Reference(var_4)
    var_7 = '$ref'
    var_8 = '#/components/schemas/User'
    var_9 = {var_7: var_8}

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'User'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.String()
    var_4 = module_0.Integer()
    var_5 = [var_3, var_4]
    var_6 = module_0.Union(var_5)
    var_7 = 'anyOf'
    var_8 = 'type'
    var_9 = 'string'
    var_10 = {var_8: var_9}
    var_11 = 'integer'
    var_12 = {var_8: var_11}
    var_13 = [var_10, var_12]
    var_14 = {var_7: var_13}

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'User'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = 'user'
    var_4 = module_0.String()
    var_5 = module_1.Reference(var_0)
    var_6 = {var_3: var_5}
    var_7 = module_0.Object(properties=var_6)
    var_8 = 'type'
    var_9 = 'properties'
    var_10 = 'object'
    var_11 = '$ref'
    var_12 = '#/components/schemas/User'
    var_13 = {var_11: var_12}
    var_14 = {var_3: var_13}
    var_15 = {var_8: var_10, var_9: var_14}

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'User'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.String()
    var_4 = module_1.Reference(var_0)
    var_5 = module_0.Array(var_4)
    var_6 = 'type'
    var_7 = 'items'
    var_8 = 'array'
    var_9 = '$ref'
    var_10 = '#/components/schemas/User'
    var_11 = {var_9: var_10}
    var_12 = {var_6: var_8, var_7: var_11}

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'User'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = 'A'
    var_5 = (var_3, var_4)
    var_6 = 'b'
    var_7 = 'B'
    var_8 = (var_6, var_7)
    var_9 = [var_5, var_8]
    var_10 = module_0.Choice(choices=var_9)
    var_11 = 'enum'
    var_12 = [var_3, var_6]
    var_13 = {var_11: var_12}

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'User'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = 'fixed'
    var_4 = module_0.Const(var_3)
    var_5 = 'const'
    var_6 = {var_5: var_3}

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'User'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.String()
    var_4 = module_0.Integer()
    var_5 = [var_3, var_4]
    var_6 = module_1.OneOf(var_5)
    var_7 = 'oneOf'
    var_8 = 'type'
    var_9 = 'string'
    var_10 = {var_8: var_9}
    var_11 = 'integer'
    var_12 = {var_8: var_11}
    var_13 = [var_10, var_12]
    var_14 = {var_7: var_13}

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'User'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.String()
    var_4 = 'test'
    var_5 = module_0.Const(var_4)
    var_6 = [var_3, var_5]
    var_7 = module_1.AllOf(var_6)
    var_8 = 'allOf'
    var_9 = 'type'
    var_10 = 'string'
    var_11 = {var_9: var_10}
    var_12 = 'const'
    var_13 = {var_12: var_4}
    var_14 = [var_11, var_13]
    var_15 = {var_8: var_14}

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'User'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.String()
    var_4 = module_0.Integer()
    var_5 = module_0.Boolean()
    var_6 = module_1.IfThenElse(var_3, var_4, var_5)
    var_7 = 'if'
    var_8 = 'then'
    var_9 = 'else'
    var_10 = 'type'
    var_11 = 'string'
    var_12 = {var_10: var_11}
    var_13 = 'integer'
    var_14 = {var_10: var_13}
    var_15 = 'boolean'
    var_16 = {var_10: var_15}
    var_17 = {var_7: var_12, var_8: var_14, var_9: var_16}

import typesystem.fields as module_0
import typesystem.composites as module_1

def test_case_0():
    var_0 = 'User'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.String()
    var_4 = module_1.Not(var_3)
    var_5 = 'not'
    var_6 = 'type'
    var_7 = 'string'
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}

import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'User'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = 'name'
    var_4 = module_0.String()
    var_5 = {var_3: var_4}
    var_6 = module_1.Schema(var_5)
    var_7 = 'type'
    var_8 = 'properties'
    var_9 = 'object'
    var_10 = 'string'
    var_11 = {var_7: var_10}
    var_12 = {var_3: var_11}
    var_13 = {var_7: var_9, var_8: var_12}

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'User'
    var_1 = True
    var_2 = module_0.String()
    var_3 = {var_0: var_2}
    var_4 = 'components'
    var_5 = 'schemas'
    var_6 = 'type'
    var_7 = 'string'
    var_8 = 'null'
    var_9 = [var_7, var_8]
    var_10 = {var_6: var_9}
    var_11 = {var_0: var_10}
    var_12 = {var_5: var_11}
    var_13 = {var_4: var_12}

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'User'
    var_1 = 'guest'
    var_2 = module_0.String()
    var_3 = {var_0: var_2}
    var_4 = 'components'
    var_5 = 'schemas'
    var_6 = 'type'
    var_7 = 'default'
    var_8 = 'string'
    var_9 = {var_6: var_8, var_7: var_1}
    var_10 = {var_0: var_9}
    var_11 = {var_5: var_10}
    var_12 = {var_4: var_11}

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'User'
    var_1 = 5
    var_2 = module_0.String(min_length=var_1)
    var_3 = {var_0: var_2}
    var_4 = 'components'
    var_5 = 'schemas'
    var_6 = 'type'
    var_7 = 'minLength'
    var_8 = 'string'
    var_9 = {var_6: var_8, var_7: var_1}
    var_10 = {var_0: var_9}
    var_11 = {var_5: var_10}
    var_12 = {var_4: var_11}

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'User'
    var_1 = 10
    var_2 = module_0.String(max_length=var_1)
    var_3 = {var_0: var_2}
    var_4 = 'components'
    var_5 = 'schemas'
    var_6 = 'type'
    var_7 = 'maxLength'
    var_8 = 'string'
    var_9 = {var_6: var_8, var_7: var_1}
    var_10 = {var_0: var_9}
    var_11 = {var_5: var_10}
    var_12 = {var_4: var_11}

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'User'
    var_1 = '^\\d+$'
    var_2 = module_0.String(pattern=var_1)
    var_3 = {var_0: var_2}
    var_4 = 'components'
    var_5 = 'schemas'
    var_6 = 'type'
    var_7 = 'pattern'
    var_8 = 'string'
    var_9 = {var_6: var_8, var_7: var_1}
    var_10 = {var_0: var_9}
    var_11 = {var_5: var_10}
    var_12 = {var_4: var_11}

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'User'
    var_1 = 'email'
    var_2 = module_0.String(format=var_1)
    var_3 = {var_0: var_2}
    var_4 = 'components'
    var_5 = 'schemas'
    var_6 = 'type'
    var_7 = 'format'
    var_8 = 'string'
    var_9 = {var_6: var_8, var_7: var_1}
    var_10 = {var_0: var_9}
    var_11 = {var_5: var_10}
    var_12 = {var_4: var_11}

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Age'
    var_1 = 0
    var_2 = 120
    var_3 = module_0.Integer(minimum=var_1, maximum=var_2)
    var_4 = {var_0: var_3}
    var_5 = 'components'
    var_6 = 'schemas'
    var_7 = 'type'
    var_8 = 'minimum'
    var_9 = 'maximum'
    var_10 = 'integer'
    var_11 = {var_7: var_10, var_8: var_1, var_9: var_2}
    var_12 = {var_0: var_11}
    var_13 = {var_6: var_12}
    var_14 = {var_5: var_13}

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Price'
    var_1 = 0.0
    var_2 = 100.0
    var_3 = module_0.Float(minimum=var_1, exclusive_maximum=var_2)
    var_4 = {var_0: var_3}
    var_5 = 'components'
    var_6 = 'schemas'
    var_7 = 'type'
    var_8 = 'minimum'
    var_9 = 'exclusiveMaximum'
    var_10 = 'number'
    var_11 = {var_7: var_10, var_8: var_1, var_9: var_2}
    var_12 = {var_0: var_11}
    var_13 = {var_6: var_12}
    var_14 = {var_5: var_13}

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Active'
    var_1 = True
    var_2 = module_0.Boolean()
    var_3 = {var_0: var_2}
    var_4 = 'components'
    var_5 = 'schemas'
    var_6 = 'type'
    var_7 = 'default'
    var_8 = 'boolean'
    var_9 = {var_6: var_8, var_7: var_1}
    var_10 = {var_0: var_9}
    var_11 = {var_5: var_10}
    var_12 = {var_4: var_11}

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Tags'
    var_1 = 1
    var_2 = 5
    var_3 = module_0.Array(min_items=var_1, max_items=var_2)
    var_4 = {var_0: var_3}
    var_5 = 'components'
    var_6 = 'schemas'
    var_7 = 'type'
    var_8 = 'minItems'
    var_9 = 'maxItems'
    var_10 = 'array'
    var_11 = {var_7: var_10, var_8: var_1, var_9: var_2}
    var_12 = {var_0: var_11}
    var_13 = {var_6: var_12}
    var_14 = {var_5: var_13}

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Tags'
    var_1 = True
    var_2 = module_0.Array(unique_items=var_1)
    var_3 = {var_0: var_2}
    var_4 = 'components'
    var_5 = 'schemas'
    var_6 = 'type'
    var_7 = 'uniqueItems'
    var_8 = 'array'
    var_9 = {var_6: var_8, var_7: var_1}
    var_10 = {var_0: var_9}
    var_11 = {var_5: var_10}
    var_12 = {var_4: var_11}

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Meta'
    var_1 = True
    var_2 = module_0.Object(additional_properties=var_1)
    var_3 = {var_0: var_2}



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 6/8 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.to_json_schema(var_0)
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.String()
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'string'
    var_5 = 'null'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = module_0.String(min_length=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'minLength'
    var_5 = 'string'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = module_0.String(max_length=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'maxLength'
    var_5 = 'string'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = module_0.String(pattern=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'pattern'
    var_5 = 'string'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'email'
    var_1 = module_0.String(format=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'format'
    var_5 = 'string'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_1.to_json_schema(var_0)
    var_2 = 'type'
    var_3 = 'integer'
    var_4 = {var_2: var_3}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Integer()
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'integer'
    var_5 = 'null'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = module_0.Integer(minimum=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'minimum'
    var_5 = 'integer'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = module_0.Integer(maximum=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'maximum'
    var_5 = 'integer'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 0
    var_1 = module_0.Integer(exclusive_minimum=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'exclusiveMinimum'
    var_5 = 'integer'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 100
    var_1 = module_0.Integer(exclusive_maximum=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'exclusiveMaximum'
    var_5 = 'integer'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 5
    var_1 = module_0.Integer(multiple_of=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'multipleOf'
    var_5 = 'integer'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Float()
    var_1 = module_1.to_json_schema(var_0)
    var_2 = 'type'
    var_3 = 'number'
    var_4 = {var_2: var_3}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Float()
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'number'
    var_5 = 'null'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = module_1.to_json_schema(var_0)
    var_2 = 'type'
    var_3 = 'boolean'
    var_4 = {var_2: var_3}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean()
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'boolean'
    var_5 = 'null'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Array()
    var_1 = module_1.to_json_schema(var_0)
    var_2 = 'type'
    var_3 = 'array'
    var_4 = {var_2: var_3}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Array()
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'array'
    var_5 = 'null'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = module_0.Array(min_items=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'minItems'
    var_5 = 'array'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = module_0.Array(max_items=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'maxItems'
    var_5 = 'array'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Array(var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'items'
    var_5 = 'array'
    var_6 = 'string'
    var_7 = {var_3: var_6}
    var_8 = {var_3: var_5, var_4: var_7}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Array(unique_items=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'uniqueItems'
    var_5 = 'array'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Object()
    var_1 = module_1.to_json_schema(var_0)
    var_2 = 'type'
    var_3 = 'object'
    var_4 = {var_2: var_3}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Object()
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'object'
    var_5 = 'null'
    var_6 = [var_4, var_5]
    var_7 = {var_3: var_6}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.Object(properties=var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'type'
    var_6 = 'properties'
    var_7 = 'object'
    var_8 = 'string'
    var_9 = {var_5: var_8}
    var_10 = {var_0: var_9}
    var_11 = {var_5: var_7, var_6: var_10}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = [var_0]
    var_4 = module_0.Object(properties=var_2, required=var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = 'type'
    var_7 = 'properties'
    var_8 = 'required'
    var_9 = 'object'
    var_10 = 'string'
    var_11 = {var_6: var_10}
    var_12 = {var_0: var_11}
    var_13 = [var_0]
    var_14 = {var_6: var_9, var_7: var_12, var_8: var_13}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Object(additional_properties=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'additionalProperties'
    var_5 = 'object'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 1
    var_1 = module_0.Object(min_properties=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'minProperties'
    var_5 = 'object'
    var_6 = {var_3: var_5, var_4: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 10
    var_1 = module_0.Object(max_properties=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'maxProperties'
    var_5 = 'object'
    var_6 = {var_3: var_5, var_4: var_0}

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
    var_9 = 'enum'
    var_10 = [var_0, var_3]
    var_11 = {var_9: var_10}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'fixed'
    var_1 = module_0.Const(var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'const'
    var_4 = {var_3: var_0}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'anyOf'
    var_6 = 'type'
    var_7 = 'string'
    var_8 = {var_6: var_7}
    var_9 = 'integer'
    var_10 = {var_6: var_9}
    var_11 = [var_8, var_10]
    var_12 = {var_5: var_11}

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.OneOf(var_2)
    var_4 = module_2.to_json_schema(var_3)
    var_5 = 'oneOf'
    var_6 = 'type'
    var_7 = 'string'
    var_8 = {var_6: var_7}
    var_9 = 'integer'
    var_10 = {var_6: var_9}
    var_11 = [var_8, var_10]
    var_12 = {var_5: var_11}

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_1.AllOf(var_2)
    var_4 = module_2.to_json_schema(var_3)
    var_5 = 'allOf'
    var_6 = 'type'
    var_7 = 'string'
    var_8 = {var_6: var_7}
    var_9 = 'integer'
    var_10 = {var_6: var_9}
    var_11 = [var_8, var_10]
    var_12 = {var_5: var_11}

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = module_1.IfThenElse(var_0, var_1)
    var_3 = module_2.to_json_schema(var_2)
    var_4 = 'if'
    var_5 = 'then'
    var_6 = 'type'
    var_7 = 'string'
    var_8 = {var_6: var_7}
    var_9 = 'integer'
    var_10 = {var_6: var_9}
    var_11 = {var_4: var_8, var_5: var_10}

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.Not(var_0)
    var_2 = module_2.to_json_schema(var_1)
    var_3 = 'not'
    var_4 = 'type'
    var_5 = 'string'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'default'
    var_1 = module_0.String()
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4, var_0: var_0}

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'User'
    var_1 = 'name'
    var_2 = module_0.String()
    var_3 = {var_1: var_2}
    var_4 = module_0.Object(properties=var_3)
    var_5 = {var_0: var_4}



