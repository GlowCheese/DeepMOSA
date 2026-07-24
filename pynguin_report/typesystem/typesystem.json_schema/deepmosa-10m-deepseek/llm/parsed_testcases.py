####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_from_json_schema_boolean_true. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_boolean_false. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_ref. Retrieved 3/7 statements.
# Partially parsed test_from_json_schema_type_string. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_type_integer. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_type_number. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_type_boolean. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_type_array. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_type_object. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_type_null. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_type_multiple. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_enum. Retrieved 7/8 statements.
# Partially parsed test_from_json_schema_const. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_allOf. Retrieved 10/11 statements.
# Partially parsed test_from_json_schema_anyOf. Retrieved 9/10 statements.
# Partially parsed test_from_json_schema_oneOf. Retrieved 9/10 statements.
# Partially parsed test_from_json_schema_not. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_if_then_else. Retrieved 10/11 statements.
# Partially parsed test_from_json_schema_multiple_constraints. Retrieved 8/9 statements.
# Partially parsed test_from_json_schema_no_constraints. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_with_components. Retrieved 11/14 statements.
# Partially parsed test_from_json_schema_allow_null. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_type_only_null. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_type_no_valid_types. Retrieved 2/3 statements.


import typesystem.json_schema as module_0


def test_case_0():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)


def test_case_0():
    var_0 = False
    var_1 = module_0.from_json_schema(var_0)

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = '$ref'
    var_3 = '#/components/schemas/Test'
    var_4 = {var_2: var_3}


def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)


def test_case_0():
    var_0 = 'type'
    var_1 = 'integer'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)


def test_case_0():
    var_0 = 'type'
    var_1 = 'number'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)


def test_case_0():
    var_0 = 'type'
    var_1 = 'boolean'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)


def test_case_0():
    var_0 = 'type'
    var_1 = 'array'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)


def test_case_0():
    var_0 = 'type'
    var_1 = 'object'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)


def test_case_0():
    var_0 = 'type'
    var_1 = 'null'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)
    var_4 = var_3.const
    assert var_4 is None


def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = 'integer'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.from_json_schema(var_4)


def test_case_0():
    var_0 = 'enum'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = module_0.from_json_schema(var_5)


def test_case_0():
    var_0 = 'const'
    var_1 = 42
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)
    var_4 = var_3.const
    assert var_4 == 42


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


def test_case_0():
    var_0 = 'not'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.from_json_schema(var_4)


def test_case_0():
    var_0 = 'if'
    var_1 = 'then'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = 'minLength'
    var_6 = 5
    var_7 = {var_5: var_6}
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = module_0.from_json_schema(var_8)


def test_case_0():
    var_0 = 'type'
    var_1 = 'minLength'
    var_2 = 'maxLength'
    var_3 = 'string'
    var_4 = 5
    var_5 = 10
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.from_json_schema(var_6)


def test_case_0():
    var_0 = {}
    var_1 = module_0.from_json_schema(var_0)


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


def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = 'null'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.from_json_schema(var_4)
    var_6 = var_5.allow_null
    assert var_6 is True


def test_case_0():
    var_0 = 'type'
    var_1 = 'null'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)
    var_4 = var_3.const
    assert var_4 is None


def test_case_0():
    var_0 = {}
    var_1 = module_0.from_json_schema(var_0)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_if_then_else_from_json_schema_basic. Retrieved 15/21 statements.
# Partially parsed test_if_then_else_from_json_schema_without_then. Retrieved 10/14 statements.
# Partially parsed test_if_then_else_from_json_schema_without_else. Retrieved 11/15 statements.
# Partially parsed test_if_then_else_from_json_schema_with_default. Retrieved 13/17 statements.
# Partially parsed test_if_then_else_from_json_schema_nested. Retrieved 23/27 statements.
# Partially parsed test_if_then_else_from_json_schema_with_ref_in_if. Retrieved 17/22 statements.
# Partially parsed test_if_then_else_from_json_schema_complex_condition. Retrieved 20/24 statements.
# Partially parsed test_if_then_else_from_json_schema_boolean_schema. Retrieved 11/16 statements.
# Partially parsed test_if_then_else_from_json_schema_empty_then_and_else. Retrieved 7/11 statements.
# Partially parsed test_if_then_else_from_json_schema_with_any_of_in_if. Retrieved 19/23 statements.


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
    var_9 = 'integer'
    var_10 = {var_3: var_9}
    var_11 = {var_0: var_5, var_1: var_8, var_2: var_10}
    var_12 = []
    var_13 = 'hello'
    var_14 = 'hi'
    var_15 = bool(False)
    assert var_15 is True
    var_16 = bool(True)
    assert var_16 is True
    var_17 = 42

def test_case_0():
    var_0 = 'if'
    var_1 = 'else'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = 'integer'
    var_6 = {var_2: var_5}
    var_7 = {var_0: var_4, var_1: var_6}
    var_8 = []
    var_9 = 'any_string'
    var_10 = 123

def test_case_0():
    var_0 = 'if'
    var_1 = 'then'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = 'maxLength'
    var_6 = 10
    var_7 = {var_2: var_3, var_5: var_6}
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = []
    var_10 = 'short'
    var_11 = 3.14

def test_case_0():
    var_0 = 'if'
    var_1 = 'then'
    var_2 = 'else'
    var_3 = 'default'
    var_4 = 'type'
    var_5 = 'boolean'
    var_6 = {var_4: var_5}
    var_7 = 'const'
    var_8 = True
    var_9 = {var_7: var_8}
    var_10 = False
    var_11 = {var_7: var_10}
    var_12 = {var_0: var_6, var_1: var_9, var_2: var_11, var_3: var_10}
    var_13 = []

def test_case_0():
    var_0 = 'if'
    var_1 = 'then'
    var_2 = 'else'
    var_3 = 'type'
    var_4 = 'properties'
    var_5 = 'object'
    var_6 = 'x'
    var_7 = 'integer'
    var_8 = {var_3: var_7}
    var_9 = {var_6: var_8}
    var_10 = {var_3: var_5, var_4: var_9}
    var_11 = 'required'
    var_12 = [var_6]
    var_13 = {var_11: var_12}
    var_14 = 'array'
    var_15 = {var_3: var_14}
    var_16 = {var_0: var_10, var_1: var_13, var_2: var_15}
    var_17 = []
    var_18 = 5
    var_19 = {var_6: var_18}
    var_20 = 1
    var_21 = 2
    var_22 = 3
    var_23 = [var_20, var_21, var_22]

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
    var_10 = 'minLength'
    var_11 = 1
    var_12 = {var_10: var_11}
    var_13 = 'number'
    var_14 = {var_1: var_13}
    var_15 = {var_4: var_9, var_5: var_12, var_6: var_14}
    var_16 = 'a'
    var_17 = 100

def test_case_0():
    var_0 = 'if'
    var_1 = 'then'
    var_2 = 'else'
    var_3 = 'allOf'
    var_4 = 'type'
    var_5 = 'string'
    var_6 = {var_4: var_5}
    var_7 = 'pattern'
    var_8 = '^[A-Z]+$'
    var_9 = {var_7: var_8}
    var_10 = [var_6, var_9]
    var_11 = {var_3: var_10}
    var_12 = 'maxLength'
    var_13 = 10
    var_14 = {var_4: var_5, var_12: var_13}
    var_15 = 'null'
    var_16 = {var_4: var_15}
    var_17 = {var_0: var_11, var_1: var_14, var_2: var_16}
    var_18 = []
    var_19 = 'HELLO'
    var_20 = None

def test_case_0():
    var_0 = 'if'
    var_1 = 'then'
    var_2 = 'else'
    var_3 = True
    var_4 = 'type'
    var_5 = 'string'
    var_6 = {var_4: var_5}
    var_7 = False
    var_8 = {var_0: var_3, var_1: var_6, var_2: var_7}
    var_9 = []
    var_10 = 'test'
    var_11 = 123
    var_12 = bool(False)
    assert var_12 is True
    var_13 = bool(True)
    assert var_13 is True

def test_case_0():
    var_0 = 'if'
    var_1 = 'type'
    var_2 = 'boolean'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = []
    var_6 = True
    var_7 = 'anything'

def test_case_0():
    var_0 = 'if'
    var_1 = 'then'
    var_2 = 'else'
    var_3 = 'anyOf'
    var_4 = 'type'
    var_5 = 'string'
    var_6 = {var_4: var_5}
    var_7 = 'number'
    var_8 = {var_4: var_7}
    var_9 = [var_6, var_8]
    var_10 = {var_3: var_9}
    var_11 = {var_4: var_5}
    var_12 = 'array'
    var_13 = {var_4: var_12}
    var_14 = {var_0: var_10, var_1: var_11, var_2: var_13}
    var_15 = []
    var_16 = 'text'
    var_17 = 1
    var_18 = 2
    var_19 = [var_17, var_18]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_to_json_schema_with_definitions. Retrieved 18/20 statements.
# Partially parsed test_to_json_schema_with_reference_field. Retrieved 19/21 statements.


import typesystem.fields as module_0
import typesystem.json_schema as module_1


def test_case_0():
    var_0 = 'Name'
    var_1 = 'A name'
    var_2 = True
    var_3 = 'John'
    var_4 = 'title'
    var_5 = 'description'
    var_6 = 'allow_null'
    var_7 = 'default'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_2, var_7: var_3}
    var_9 = module_0.String(**var_8)
    var_10 = module_1.to_json_schema(var_9)
    var_11 = 'type'
    var_12 = 'default'
    var_13 = 'title'
    var_14 = 'description'
    var_15 = 'string'
    var_16 = 'null'
    var_17 = [var_15, var_16]
    var_18 = {var_11: var_17, var_12: var_3, var_13: var_0, var_14: var_1}
    var_19 = bool(var_10 == var_18)
    assert var_19 is True


def test_case_0():
    var_0 = 0
    var_1 = 100
    var_2 = False
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.Integer(minimum=var_0, maximum=var_1, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = 'type'
    var_8 = 'minimum'
    var_9 = 'maximum'
    var_10 = 'integer'
    var_11 = {var_7: var_10, var_8: var_2, var_9: var_1}
    var_12 = bool(var_6 == var_11)
    assert var_12 is True


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'default'
    var_3 = 'allow_null'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Boolean(**var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = 'type'
    var_8 = 'default'
    var_9 = 'boolean'
    var_10 = 'null'
    var_11 = [var_9, var_10]
    var_12 = {var_7: var_11, var_8: var_0}
    var_13 = bool(var_6 == var_12)
    assert var_13 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 1
    var_3 = 10
    var_4 = False
    var_5 = 'allow_null'
    var_6 = {var_5: var_4}
    var_7 = module_0.Array(var_1, min_items=var_2, max_items=var_3, **var_6)
    var_8 = module_1.to_json_schema(var_7)
    var_9 = 'type'
    var_10 = 'minItems'
    var_11 = 'maxItems'
    var_12 = 'items'
    var_13 = 'array'
    var_14 = 'string'
    var_15 = {var_9: var_14}
    var_16 = {var_9: var_13, var_10: var_2, var_11: var_3, var_12: var_15}
    var_17 = bool(var_8 == var_16)
    assert var_17 is True


def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = True
    var_6 = 'allow_null'
    var_7 = {var_6: var_5}
    var_8 = module_0.Object(properties=var_3, required=var_4, **var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = 'type'
    var_11 = 'properties'
    var_12 = 'required'
    var_13 = 'object'
    var_14 = 'null'
    var_15 = [var_13, var_14]
    var_16 = 'string'
    var_17 = {var_10: var_16}
    var_18 = {var_0: var_17}
    var_19 = [var_0]
    var_20 = {var_10: var_15, var_11: var_18, var_12: var_19}
    var_21 = bool(var_9 == var_20)
    assert var_21 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = False
    var_6 = 'allow_null'
    var_7 = {var_6: var_5}
    var_8 = module_0.Union(var_4, **var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = 'anyOf'
    var_11 = 'type'
    var_12 = 'string'
    var_13 = {var_11: var_12}
    var_14 = 'integer'
    var_15 = {var_11: var_14}
    var_16 = [var_13, var_15]
    var_17 = {var_10: var_16}
    var_18 = bool(var_9 == var_17)
    assert var_18 is True


def test_case_0():
    var_0 = 'User'
    var_1 = 'name'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = module_0.Object(properties=var_4, **var_5)
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


def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = 'User'
    var_7 = 'target'
    var_8 = {var_7: var_5}
    var_9 = '$ref'
    var_10 = 'components'
    var_11 = '#/components/schemas/User'
    var_12 = 'schemas'
    var_13 = 'type'
    var_14 = 'properties'
    var_15 = 'object'
    var_16 = 'string'
    var_17 = {var_13: var_16}
    var_18 = {var_0: var_17}
    var_19 = {var_13: var_15, var_14: var_18}
    var_20 = {var_6: var_19}
    var_21 = {var_12: var_20}
    var_22 = {var_9: var_11, var_10: var_21}


def test_case_0():
    var_0 = 'a'
    var_1 = 'Option A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'Option B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 'default'
    var_8 = {var_7: var_0}
    var_9 = module_0.Choice(choices=var_6, **var_8)
    var_10 = module_1.to_json_schema(var_9)
    var_11 = 'enum'
    var_12 = 'default'
    var_13 = [var_0, var_3]
    var_14 = {var_11: var_13, var_12: var_0}
    var_15 = bool(var_10 == var_14)
    assert var_15 is True


def test_case_0():
    var_0 = 42
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.Const(var_0, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = 'const'
    var_7 = {var_6: var_0}
    var_8 = bool(var_5 == var_7)
    assert var_8 is True

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
    var_7 = False
    var_8 = 'allow_null'
    var_9 = {var_8: var_7}
    var_10 = module_1.AllOf(var_6, **var_9)
    var_11 = module_2.to_json_schema(var_10)
    var_12 = 'allOf'
    var_13 = 'type'
    var_14 = 'minLength'
    var_15 = 'string'
    var_16 = {var_13: var_15, var_14: var_0}
    var_17 = 'maxLength'
    var_18 = {var_13: var_15, var_17: var_3}
    var_19 = [var_16, var_18]
    var_20 = {var_12: var_19}
    var_21 = bool(var_11 == var_20)
    assert var_21 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = False
    var_6 = 'allow_null'
    var_7 = {var_6: var_5}
    var_8 = module_1.OneOf(var_4, **var_7)
    var_9 = module_2.to_json_schema(var_8)
    var_10 = 'oneOf'
    var_11 = 'type'
    var_12 = 'string'
    var_13 = {var_11: var_12}
    var_14 = 'integer'
    var_15 = {var_11: var_14}
    var_16 = [var_13, var_15]
    var_17 = {var_10: var_16}
    var_18 = bool(var_9 == var_17)
    assert var_18 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = {}
    var_5 = module_0.Boolean(**var_4)
    var_6 = False
    var_7 = 'allow_null'
    var_8 = {var_7: var_6}
    var_9 = module_1.IfThenElse(var_1, var_3, var_5, **var_8)
    var_10 = module_2.to_json_schema(var_9)
    var_11 = 'if'
    var_12 = 'then'
    var_13 = 'else'
    var_14 = 'type'
    var_15 = 'string'
    var_16 = {var_14: var_15}
    var_17 = 'integer'
    var_18 = {var_14: var_17}
    var_19 = 'boolean'
    var_20 = {var_14: var_19}
    var_21 = {var_11: var_16, var_12: var_18, var_13: var_20}
    var_22 = bool(var_10 == var_21)
    assert var_22 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = False
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_1.Not(var_1, **var_4)
    var_6 = module_2.to_json_schema(var_5)
    var_7 = 'not'
    var_8 = 'type'
    var_9 = 'string'
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = bool(var_6 == var_11)
    assert var_12 is True

import typesystem.json_schema as module_1


def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True

import typesystem.composites as module_0


def test_case_0():
    var_0 = {}
    var_1 = module_0.NeverMatch(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is False

import typesystem.fields as module_0


def test_case_0():
    var_0 = 0.0
    var_1 = 1.0
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.Decimal(minimum=var_0, maximum=var_1, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = 'type'
    var_8 = 'minimum'
    var_9 = 'maximum'
    var_10 = 'number'
    var_11 = 'null'
    var_12 = [var_10, var_11]
    var_13 = {var_7: var_12, var_8: var_0, var_9: var_2}
    var_14 = bool(var_6 == var_13)
    assert var_14 is True


def test_case_0():
    var_0 = 0.0
    var_1 = 10.0
    var_2 = False
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.Float(exclusive_minimum=var_0, exclusive_maximum=var_1, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = 'type'
    var_8 = 'exclusiveMinimum'
    var_9 = 'exclusiveMaximum'
    var_10 = 'number'
    var_11 = {var_7: var_10, var_8: var_2, var_9: var_1}
    var_12 = bool(var_6 == var_11)
    assert var_12 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = False
    var_5 = 'allow_null'
    var_6 = {var_5: var_4}
    var_7 = module_0.Array(var_1, var_3, **var_6)
    var_8 = module_1.to_json_schema(var_7)
    var_9 = 'type'
    var_10 = 'items'
    var_11 = 'additionalItems'
    var_12 = 'array'
    var_13 = 'string'
    var_14 = {var_9: var_13}
    var_15 = 'integer'
    var_16 = {var_9: var_15}
    var_17 = {var_9: var_12, var_10: var_14, var_11: var_16}
    var_18 = bool(var_8 == var_17)
    assert var_18 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = False
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.Object(additional_properties=var_1, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = 'type'
    var_8 = 'additionalProperties'
    var_9 = 'object'
    var_10 = 'string'
    var_11 = {var_7: var_10}
    var_12 = {var_7: var_9, var_8: var_11}
    var_13 = bool(var_6 == var_12)
    assert var_13 is True


def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = False
    var_5 = 'allow_null'
    var_6 = {var_5: var_4}
    var_7 = module_0.Object(pattern_properties=var_3, **var_6)
    var_8 = module_1.to_json_schema(var_7)
    var_9 = 'type'
    var_10 = 'patternProperties'
    var_11 = 'object'
    var_12 = 'string'
    var_13 = {var_9: var_12}
    var_14 = {var_0: var_13}
    var_15 = {var_9: var_11, var_10: var_14}
    var_16 = bool(var_8 == var_15)
    assert var_16 is True

import re as module_0

import typesystem.fields as module_1


def test_case_0():
    var_0 = '^[a-z]+$'
    var_1 = module_0.compile(var_0)
    var_2 = 'pattern_regex'
    var_3 = {var_2: var_1}
    var_4 = module_1.String(**var_3)
    var_5 = False
    var_6 = 'allow_null'
    var_7 = {var_6: var_5}
    var_8 = module_1.Object(property_names=var_4, **var_7)
    var_9 = module_2.to_json_schema(var_8)
    var_10 = 'type'
    var_11 = 'propertyNames'
    var_12 = 'object'
    var_13 = 'pattern'
    var_14 = 'string'
    var_15 = {var_10: var_14, var_13: var_0}
    var_16 = {var_10: var_12, var_11: var_15}
    var_17 = bool(var_9 == var_16)
    assert var_17 is True

import typesystem.fields as module_0
import typesystem.schemas as module_1


def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = True
    var_6 = 'required'
    var_7 = 'allow_null'
    var_8 = {var_6: var_4, var_7: var_5}
    var_9 = module_1.Schema(var_3, **var_8)
    var_10 = module_2.to_json_schema(var_9)
    var_11 = 'type'
    var_12 = 'properties'
    var_13 = 'required'
    var_14 = 'object'
    var_15 = 'null'
    var_16 = [var_14, var_15]
    var_17 = 'string'
    var_18 = {var_11: var_17}
    var_19 = {var_0: var_18}
    var_20 = [var_0]
    var_21 = {var_11: var_16, var_12: var_19, var_13: var_20}
    var_22 = bool(var_10 == var_21)
    assert var_22 is True

import re as module_0

import typesystem.fields as module_1


def test_case_0():
    var_0 = '^[A-Z]+$'
    var_1 = module_0.compile(var_0)
    var_2 = False
    var_3 = 'pattern_regex'
    var_4 = 'allow_null'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_1.String(**var_5)
    var_7 = module_2.to_json_schema(var_6)
    var_8 = 'type'
    var_9 = 'pattern'
    var_10 = 'string'
    var_11 = {var_8: var_10, var_9: var_0}
    var_12 = bool(var_7 == var_11)
    assert var_12 is True

import typesystem.fields as module_0
import typesystem.json_schema as module_1


def test_case_0():
    var_0 = 'email'
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(format=var_0, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = 'type'
    var_7 = 'format'
    var_8 = 'string'
    var_9 = {var_6: var_8, var_7: var_0}
    var_10 = bool(var_5 == var_9)
    assert var_10 is True


def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = False
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.String(max_length=var_1, min_length=var_0, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = 'type'
    var_8 = 'minLength'
    var_9 = 'maxLength'
    var_10 = 'string'
    var_11 = {var_7: var_10, var_8: var_0, var_9: var_1}
    var_12 = bool(var_6 == var_11)
    assert var_12 is True


def test_case_0():
    var_0 = 2
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.Integer(multiple_of=var_0, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = 'type'
    var_7 = 'multipleOf'
    var_8 = 'integer'
    var_9 = {var_6: var_8, var_7: var_0}
    var_10 = bool(var_5 == var_9)
    assert var_10 is True


def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.Array(unique_items=var_0, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = 'type'
    var_7 = 'uniqueItems'
    var_8 = 'array'
    var_9 = {var_6: var_8, var_7: var_0}
    var_10 = bool(var_5 == var_9)
    assert var_10 is True


def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = False
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.Object(min_properties=var_0, max_properties=var_1, **var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = 'type'
    var_8 = 'minProperties'
    var_9 = 'maxProperties'
    var_10 = 'object'
    var_11 = {var_7: var_10, var_8: var_0, var_9: var_1}
    var_12 = bool(var_6 == var_11)
    assert var_12 is True


def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = [var_3, var_5]
    var_7 = False
    var_8 = 'allow_null'
    var_9 = {var_8: var_7}
    var_10 = module_0.Union(var_6, **var_9)



# Parsed testcases at query #4
#--------------------------





def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = var_4['type']
    var_6 = bool(var_4['type'] == ['array', 'null'])
    assert var_6 is True



# Parsed testcases at query #5
#--------------------------




import typesystem.composites as module_1


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
# Partially parsed test_from_json_schema_type_array_with_items_list. Retrieved 18/25 statements.
# Partially parsed test_from_json_schema_type_array_with_items_single. Retrieved 17/22 statements.
# Partially parsed test_from_json_schema_type_array_without_items. Retrieved 13/16 statements.
# Partially parsed test_from_json_schema_type_object_with_properties. Retrieved 26/35 statements.
# Partially parsed test_from_json_schema_type_object_without_properties. Retrieved 12/15 statements.
# Partially parsed test_from_json_schema_type_object_with_additional_properties_field. Retrieved 11/16 statements.


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
    var_0 = []
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'items'
    var_5 = 'additionalItems'
    var_6 = 'minItems'
    var_7 = 'maxItems'
    var_8 = 'uniqueItems'
    var_9 = 'default'
    var_10 = [var_3, var_3]
    var_11 = False
    var_12 = 2
    var_13 = True
    var_14 = 'a'
    var_15 = 'b'
    var_16 = [var_14, var_15]
    var_17 = {var_4: var_10, var_5: var_11, var_6: var_12, var_7: var_12, var_8: var_13, var_9: var_16}
    var_18 = 'array'

def test_case_0():
    var_0 = []
    var_1 = 'type'
    var_2 = 'integer'
    var_3 = {var_1: var_2}
    var_4 = 'items'
    var_5 = 'additionalItems'
    var_6 = 'minItems'
    var_7 = 'maxItems'
    var_8 = 'uniqueItems'
    var_9 = 'default'
    var_10 = True
    var_11 = 0
    var_12 = 10
    var_13 = False
    var_14 = 2
    var_15 = [var_10, var_14]
    var_16 = {var_4: var_3, var_5: var_10, var_6: var_11, var_7: var_12, var_8: var_13, var_9: var_15}
    var_17 = 'array'

def test_case_0():
    var_0 = 'minItems'
    var_1 = 'maxItems'
    var_2 = 'additionalItems'
    var_3 = 'uniqueItems'
    var_4 = 'default'
    var_5 = 0
    var_6 = None
    var_7 = True
    var_8 = False
    var_9 = []
    var_10 = {var_0: var_5, var_1: var_6, var_2: var_7, var_3: var_8, var_4: var_9}
    var_11 = []
    var_12 = 'array'
    var_13 = False

def test_case_0():
    var_0 = []
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'properties'
    var_5 = 'patternProperties'
    var_6 = 'additionalProperties'
    var_7 = 'propertyNames'
    var_8 = 'minProperties'
    var_9 = 'maxProperties'
    var_10 = 'required'
    var_11 = 'default'
    var_12 = 'name'
    var_13 = {var_12: var_3}
    var_14 = '^test_'
    var_15 = {var_14: var_3}
    var_16 = False
    var_17 = 'pattern'
    var_18 = '^[a-z]+$'
    var_19 = {var_17: var_18}
    var_20 = 1
    var_21 = 5
    var_22 = [var_12]
    var_23 = 'john'
    var_24 = {var_12: var_23}
    var_25 = {var_4: var_13, var_5: var_15, var_6: var_16, var_7: var_19, var_8: var_20, var_9: var_21, var_10: var_22, var_11: var_24}
    var_26 = 'object'
    var_27 = 'name'
    var_28 = '^test_'

def test_case_0():
    var_0 = 'minProperties'
    var_1 = 'maxProperties'
    var_2 = 'additionalProperties'
    var_3 = 'required'
    var_4 = 'default'
    var_5 = 0
    var_6 = None
    var_7 = True
    var_8 = []
    var_9 = {}
    var_10 = {var_0: var_5, var_1: var_6, var_2: var_7, var_3: var_8, var_4: var_9}
    var_11 = []
    var_12 = 'object'

def test_case_0():
    var_0 = []
    var_1 = 'type'
    var_2 = 'number'
    var_3 = {var_1: var_2}
    var_4 = 'additionalProperties'
    var_5 = 'default'
    var_6 = 'extra'
    var_7 = 42
    var_8 = {var_6: var_7}
    var_9 = {var_4: var_3, var_5: var_8}
    var_10 = 'object'
    var_11 = False



# Parsed testcases at query #7
#--------------------------






# Parsed testcases at query #8
#--------------------------

# Partially parsed test_ref_from_json_schema_with_valid_ref. Retrieved 3/6 statements.
# Partially parsed test_ref_from_json_schema_raises_assertion_for_non_hash_ref. Retrieved 3/6 statements.
# Partially parsed test_ref_from_json_schema_creates_reference_with_correct_target. Retrieved 4/7 statements.
# Partially parsed test_ref_from_json_schema_passes_definitions_to_reference. Retrieved 5/7 statements.


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
    var_1 = 'MockField'
    var_2 = '$ref'
    var_3 = '#/definitions/Item'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = '#/definitions/Test'
    var_1 = 'TestSchema'
    var_2 = {var_0: var_1}
    var_3 = [var_2]
    var_4 = '$ref'
    var_5 = {var_4: var_0}



# Parsed testcases at query #9
#--------------------------






# Parsed testcases at query #10
#--------------------------

# Partially parsed test_array_field_with_list_items. Retrieved 10/13 statements.


import typesystem.json_schema as module_1


def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.String(**var_4)
    var_6 = [var_1, var_5]
    var_7 = {}
    var_8 = module_0.Array(var_6, **var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = 'items'
    var_11 = bool('items' in var_9)
    assert var_11 is True
    var_12 = 'items'
    var_13 = var_9[var_12]
    var_14 = var_9[var_12]
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = var_9['items'][0]
    var_17 = bool(var_9['items'][0] == {'type': 'string'})
    assert var_17 is True
    var_18 = var_9['items'][1]
    var_19 = bool(var_9['items'][1] == {'type': ['string', 'null']})
    assert var_19 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_pattern_regex_flags_unicode. Retrieved 1/5 statements.
# Partially parsed test_pattern_regex_flags_non_unicode_raises. Retrieved 1/6 statements.


def test_case_0():
    var_0 = '^test$'
    var_1 = 'pattern'

def test_case_0():
    var_0 = '^test$'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'non-standard flags'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_from_json_schema_type_number. Retrieved 13/16 statements.
# Partially parsed test_from_json_schema_type_number_nullable. Retrieved 3/6 statements.
# Partially parsed test_from_json_schema_type_integer. Retrieved 13/16 statements.
# Partially parsed test_from_json_schema_type_string. Retrieved 13/16 statements.
# Partially parsed test_from_json_schema_type_string_allow_blank. Retrieved 5/8 statements.
# Partially parsed test_from_json_schema_type_boolean. Retrieved 5/8 statements.
# Partially parsed test_from_json_schema_type_array_no_items. Retrieved 11/14 statements.
# Partially parsed test_from_json_schema_type_array_with_items_single. Retrieved 7/12 statements.
# Partially parsed test_from_json_schema_type_array_with_items_list. Retrieved 10/17 statements.
# Partially parsed test_from_json_schema_type_array_additional_items_bool. Retrieved 4/7 statements.
# Partially parsed test_from_json_schema_type_array_additional_items_field. Retrieved 7/12 statements.
# Partially parsed test_from_json_schema_type_object_no_properties. Retrieved 12/15 statements.
# Partially parsed test_from_json_schema_type_object_with_properties. Retrieved 9/14 statements.
# Partially parsed test_from_json_schema_type_object_with_pattern_properties. Retrieved 9/14 statements.
# Partially parsed test_from_json_schema_type_object_additional_properties_bool. Retrieved 4/7 statements.
# Partially parsed test_from_json_schema_type_object_additional_properties_field. Retrieved 7/12 statements.
# Partially parsed test_from_json_schema_type_object_with_property_names. Retrieved 7/12 statements.
# Partially parsed test_from_json_schema_type_invalid_type_string. Retrieved 3/6 statements.


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
    var_11 = 'number'
    var_12 = False
    var_13 = []

def test_case_0():
    var_0 = {}
    var_1 = 'number'
    var_2 = True
    var_3 = []

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
    var_11 = 'integer'
    var_12 = False
    var_13 = []

def test_case_0():
    var_0 = 'minLength'
    var_1 = 'maxLength'
    var_2 = 'format'
    var_3 = 'pattern'
    var_4 = 'default'
    var_5 = 5
    var_6 = 10
    var_7 = 'email'
    var_8 = '^a.*z$'
    var_9 = 'abc'
    var_10 = {var_0: var_5, var_1: var_6, var_2: var_7, var_3: var_8, var_4: var_9}
    var_11 = 'string'
    var_12 = False
    var_13 = []

def test_case_0():
    var_0 = 'minLength'
    var_1 = 0
    var_2 = {var_0: var_1}
    var_3 = 'string'
    var_4 = False
    var_5 = []

def test_case_0():
    var_0 = 'default'
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 'boolean'
    var_4 = False
    var_5 = []

def test_case_0():
    var_0 = 'minItems'
    var_1 = 'maxItems'
    var_2 = 'uniqueItems'
    var_3 = 'default'
    var_4 = 0
    var_5 = 10
    var_6 = True
    var_7 = []
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = 'array'
    var_10 = False
    var_11 = []

def test_case_0():
    var_0 = []
    var_1 = 'items'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'array'
    var_7 = False

def test_case_0():
    var_0 = []
    var_1 = 'items'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = 'number'
    var_6 = {var_2: var_5}
    var_7 = [var_4, var_6]
    var_8 = {var_1: var_7}
    var_9 = 'array'
    var_10 = False

def test_case_0():
    var_0 = 'additionalItems'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'array'
    var_4 = []

def test_case_0():
    var_0 = []
    var_1 = 'additionalItems'
    var_2 = 'type'
    var_3 = 'integer'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'array'
    var_7 = False

def test_case_0():
    var_0 = 'minProperties'
    var_1 = 'maxProperties'
    var_2 = 'required'
    var_3 = 'default'
    var_4 = 0
    var_5 = 10
    var_6 = 'id'
    var_7 = [var_6]
    var_8 = {}
    var_9 = {var_0: var_4, var_1: var_5, var_2: var_7, var_3: var_8}
    var_10 = 'object'
    var_11 = False
    var_12 = []

def test_case_0():
    var_0 = []
    var_1 = 'properties'
    var_2 = 'name'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = 'object'
    var_9 = False
    var_10 = 'name'

def test_case_0():
    var_0 = []
    var_1 = 'patternProperties'
    var_2 = '^x_'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {var_1: var_6}
    var_8 = 'object'
    var_9 = False
    var_10 = '^x_'

def test_case_0():
    var_0 = 'additionalProperties'
    var_1 = False
    var_2 = {var_0: var_1}
    var_3 = 'object'
    var_4 = []

def test_case_0():
    var_0 = []
    var_1 = 'additionalProperties'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'object'
    var_7 = False

def test_case_0():
    var_0 = []
    var_1 = 'propertyNames'
    var_2 = 'pattern'
    var_3 = '^[a-z]+$'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'object'
    var_7 = False

def test_case_0():
    var_0 = {}
    var_1 = 'invalid'
    var_2 = False
    var_3 = []
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_from_json_schema_boolean_true. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_boolean_false. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_ref. Retrieved 3/7 statements.
# Partially parsed test_from_json_schema_type_string. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_type_integer. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_type_number. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_type_boolean. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_type_array. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_type_object. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_enum. Retrieved 7/8 statements.
# Partially parsed test_from_json_schema_const. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_allOf. Retrieved 10/11 statements.
# Partially parsed test_from_json_schema_anyOf. Retrieved 9/10 statements.
# Partially parsed test_from_json_schema_oneOf. Retrieved 9/10 statements.
# Partially parsed test_from_json_schema_not. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_if_then_else. Retrieved 10/11 statements.
# Partially parsed test_from_json_schema_multiple_constraints. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_no_constraints. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_nullable_type. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_multiple_types. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_components_definitions. Retrieved 11/14 statements.
# Partially parsed test_from_json_schema_default_definitions. Retrieved 12/13 statements.


import typesystem.json_schema as module_0


def test_case_0():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)


def test_case_0():
    var_0 = False
    var_1 = module_0.from_json_schema(var_0)

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = '$ref'
    var_3 = '#/components/schemas/Test'
    var_4 = {var_2: var_3}


def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)
    var_4 = var_3.allow_null
    assert var_4 is False


def test_case_0():
    var_0 = 'type'
    var_1 = 'integer'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)


def test_case_0():
    var_0 = 'type'
    var_1 = 'number'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)


def test_case_0():
    var_0 = 'type'
    var_1 = 'boolean'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)


def test_case_0():
    var_0 = 'type'
    var_1 = 'array'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)


def test_case_0():
    var_0 = 'type'
    var_1 = 'object'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)


def test_case_0():
    var_0 = 'enum'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = module_0.from_json_schema(var_5)


def test_case_0():
    var_0 = 'const'
    var_1 = 'fixed'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)
    var_4 = var_3.const
    assert var_4 == 'fixed'


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


def test_case_0():
    var_0 = 'not'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.from_json_schema(var_4)


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


def test_case_0():
    var_0 = 'type'
    var_1 = 'minLength'
    var_2 = 'string'
    var_3 = 5
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.from_json_schema(var_4)


def test_case_0():
    var_0 = {}
    var_1 = module_0.from_json_schema(var_0)


def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = 'null'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.from_json_schema(var_4)
    var_6 = var_5.allow_null
    assert var_6 is True


def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = 'integer'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.from_json_schema(var_4)
    var_6 = var_5.allow_null
    assert var_6 is False


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


def test_case_0():
    var_0 = '$ref'
    var_1 = 'components'
    var_2 = '#/components/schemas/Test'
    var_3 = 'schemas'
    var_4 = 'Test'
    var_5 = 'type'
    var_6 = 'string'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = {var_3: var_8}
    var_10 = {var_0: var_2, var_1: var_9}
    var_11 = module_0.from_json_schema(var_10)



# Parsed testcases at query #14
#--------------------------




import typesystem.composites as module_1
import typesystem.fields as module_0


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



# Parsed testcases at query #15
#--------------------------






# Parsed testcases at query #16
#--------------------------




import typesystem.json_schema as module_1


def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Array(additional_items=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'additionalItems'
    var_5 = bool('additionalItems' not in var_3)
    assert var_5 is True



# Parsed testcases at query #17
#--------------------------






# Parsed testcases at query #18
#--------------------------

# Partially parsed test_to_json_schema_with_union_field_and_definitions. Retrieved 9/15 statements.
# Partially parsed test_to_json_schema_with_union_field_and_nested_definitions. Retrieved 10/16 statements.
# Partially parsed test_to_json_schema_with_union_field_and_reference. Retrieved 4/12 statements.



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


def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
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
    var_16 = bool(var_9['anyOf'][0]['type'] == ['string', 'null'])
    assert var_16 is True
    var_17 = var_9['anyOf'][1]['type']
    assert var_17 == 'integer'


def test_case_0():
    var_0 = 'test'
    var_1 = 'default'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = 42
    var_5 = 'default'
    var_6 = {var_5: var_4}
    var_7 = module_0.Integer(**var_6)
    var_8 = [var_3, var_7]
    var_9 = {}
    var_10 = module_0.Union(var_8, **var_9)
    var_11 = module_1.to_json_schema(var_10)
    var_12 = 'anyOf'
    var_13 = bool('anyOf' in var_11)
    assert var_13 is True
    var_14 = 'anyOf'
    var_15 = var_11[var_14]
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = var_11['anyOf'][0]['type']
    assert var_17 == 'string'
    var_18 = var_11['anyOf'][0]['default']
    assert var_18 == 'test'
    var_19 = var_11['anyOf'][1]['type']
    assert var_19 == 'integer'
    var_20 = var_11['anyOf'][1]['default']
    assert var_20 == 42


def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = 'MyUnion'
    var_8 = {var_7: var_6}
    var_9 = [var_8]
    var_10 = 'components'
    var_11 = 'schemas'
    var_12 = 'MyUnion'
    var_13 = 'anyOf'
    var_14 = 'anyOf'
    var_15 = 'schemas'
    var_16 = 'components'


def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Union(var_4, **var_5)
    var_7 = 'MyString'
    var_8 = 'MyUnion'
    var_9 = {var_7: var_1, var_8: var_6}
    var_10 = [var_9]
    var_11 = 'components'
    var_12 = 'schemas'
    var_13 = 'MyString'
    var_14 = 'MyUnion'
    var_15 = 'anyOf'
    var_16 = 'anyOf'
    var_17 = 'schemas'
    var_18 = 'components'


def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = 'MyString'
    var_5 = {}
    var_6 = 'MyUnion'
    var_7 = 'components'
    var_8 = 'schemas'
    var_9 = 'MyString'
    var_10 = 'MyUnion'


def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
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
    var_16 = bool(var_9['anyOf'][0]['type'] == ['string', 'null'])
    assert var_16 is True
    var_17 = var_9['anyOf'][1]['type']
    assert var_17 == 'integer'


def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = 'allow_null'
    var_5 = {var_4: var_0}
    var_6 = module_0.Integer(**var_5)
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
    var_17 = bool(var_10['anyOf'][0]['type'] == ['string', 'null'])
    assert var_17 is True
    var_18 = var_10['anyOf'][1]['type']
    var_19 = bool(var_10['anyOf'][1]['type'] == ['integer', 'null'])
    assert var_19 is True


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


def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Union(var_2, **var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = 'anyOf'
    var_7 = bool('anyOf' in var_5)
    assert var_7 is True
    var_8 = 'anyOf'
    var_9 = var_5[var_8]
    var_10 = len(var_9)
    assert var_10 == 1
    var_11 = var_5['anyOf'][0]['type']
    assert var_11 == 'string'


def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
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
    assert var_15 == 'string'
    var_16 = var_9['anyOf'][1]['type']
    assert var_16 == 'integer'
    var_17 = var_9['anyOf'][2]['type']
    assert var_17 == 'boolean'


def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = {}
    var_5 = module_0.String(**var_4)
    var_6 = {}
    var_7 = module_0.Integer(**var_6)
    var_8 = [var_5, var_7, var_3]
    var_9 = {}
    var_10 = module_0.Union(var_8, **var_9)
    var_11 = module_1.to_json_schema(var_10)
    var_12 = 'anyOf'
    var_13 = bool('anyOf' in var_11)
    assert var_13 is True
    var_14 = 'anyOf'
    var_15 = var_11[var_14]
    var_16 = len(var_15)
    assert var_16 == 3
    var_17 = var_11['anyOf'][0]['type']
    assert var_17 == 'string'
    var_18 = var_11['anyOf'][1]['type']
    assert var_18 == 'integer'
    var_19 = var_11['anyOf'][2]['type']
    assert var_19 == 'array'
    var_20 = var_11['anyOf'][2]['items']['type']
    assert var_20 == 'string'


def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = {}
    var_7 = module_0.String(**var_6)
    var_8 = [var_7, var_5]
    var_9 = {}
    var_10 = module_0.Union(var_8, **var_9)
    var_11 = module_1.to_json_schema(var_10)
    var_12 = 'anyOf'
    var_13 = bool('anyOf' in var_11)
    assert var_13 is True
    var_14 = 'anyOf'
    var_15 = var_11[var_14]
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = var_11['anyOf'][0]['type']
    assert var_17 == 'string'
    var_18 = var_11['anyOf'][1]['type']
    assert var_18 == 'object'
    var_19 = 'properties'
    var_20 = bool('properties' in var_11['anyOf'][1])
    assert var_20 is True
    var_21 = 'name'
    var_22 = bool('name' in var_11['anyOf'][1]['properties'])
    assert var_22 is True

import typesystem.schemas as module_1


def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_1.Schema(var_3, **var_4)
    var_6 = {}
    var_7 = module_0.String(**var_6)
    var_8 = [var_7, var_5]
    var_9 = {}
    var_10 = module_0.Union(var_8, **var_9)
    var_11 = module_2.to_json_schema(var_10)
    var_12 = 'anyOf'
    var_13 = bool('anyOf' in var_11)
    assert var_13 is True
    var_14 = 'anyOf'
    var_15 = var_11[var_14]
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = var_11['anyOf'][0]['type']
    assert var_17 == 'string'
    var_18 = var_11['anyOf'][1]['type']
    assert var_18 == 'object'
    var_19 = 'properties'
    var_20 = bool('properties' in var_11['anyOf'][1])
    assert var_20 is True
    var_21 = 'name'
    var_22 = bool('name' in var_11['anyOf'][1]['properties'])
    assert var_22 is True

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
    var_10 = module_0.String(**var_9)
    var_11 = [var_10, var_8]
    var_12 = {}
    var_13 = module_0.Union(var_11, **var_12)
    var_14 = module_1.to_json_schema(var_13)
    var_15 = 'anyOf'
    var_16 = bool('anyOf' in var_14)
    assert var_16 is True
    var_17 = 'anyOf'
    var_18 = var_14[var_17]
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = var_14['anyOf'][0]['type']
    assert var_20 == 'string'
    var_21 = 'enum'
    var_22 = bool('enum' in var_14['anyOf'][1])
    assert var_22 is True
    var_23 = var_14['anyOf'][1]['enum']
    var_24 = bool(var_14['anyOf'][1]['enum'] == ['a', 'b'])
    assert var_24 is True

def test_case_0():
    pass



# Parsed testcases at query #19
#--------------------------





def test_case_0():
    var_0 = 'test_default'
    var_1 = module_0.Field(default=var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'default'
    var_4 = {var_3: var_0}
    var_5 = bool(var_2 == var_4)
    assert var_5 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = bool(var_2 == var_5)
    assert var_6 is True


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


def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'integer'
    var_5 = {var_3: var_4}
    var_6 = bool(var_2 == var_5)
    assert var_6 is True


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


def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'number'
    var_5 = {var_3: var_4}
    var_6 = bool(var_2 == var_5)
    assert var_6 is True


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


def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'boolean'
    var_5 = {var_3: var_4}
    var_6 = bool(var_2 == var_5)
    assert var_6 is True


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


def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'array'
    var_5 = {var_3: var_4}
    var_6 = bool(var_2 == var_5)
    assert var_6 is True


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


def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'object'
    var_5 = {var_3: var_4}
    var_6 = bool(var_2 == var_5)
    assert var_6 is True


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


def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Object(max_properties=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'type'
    var_5 = 'maxProperties'
    var_6 = 'object'
    var_7 = {var_4: var_6, var_5: var_0}
    var_8 = bool(var_3 == var_7)
    assert var_8 is True


def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, required=var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'type'
    var_9 = 'properties'
    var_10 = 'required'
    var_11 = 'object'
    var_12 = 'string'
    var_13 = {var_8: var_12}
    var_14 = {var_0: var_13}
    var_15 = [var_0]
    var_16 = {var_8: var_11, var_9: var_14, var_10: var_15}
    var_17 = bool(var_7 == var_16)
    assert var_17 is True

import typesystem.schemas as module_1


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
    var_9 = 'type'
    var_10 = 'properties'
    var_11 = 'required'
    var_12 = 'object'
    var_13 = 'string'
    var_14 = {var_9: var_13}
    var_15 = {var_0: var_14}
    var_16 = [var_0]
    var_17 = {var_9: var_12, var_10: var_15, var_11: var_16}
    var_18 = bool(var_8 == var_17)
    assert var_18 is True

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


def test_case_0():
    var_0 = 'fixed_value'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'const'
    var_5 = {var_4: var_0}
    var_6 = bool(var_3 == var_5)
    assert var_6 is True


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
    var_9 = 'type'
    var_10 = 'string'
    var_11 = {var_9: var_10}
    var_12 = 'integer'
    var_13 = {var_9: var_12}
    var_14 = [var_11, var_13]
    var_15 = {var_8: var_14}
    var_16 = bool(var_7 == var_15)
    assert var_16 is True



# Parsed testcases at query #20
#--------------------------






# Parsed testcases at query #21
#--------------------------

# Partially parsed test_pattern_regex_flags_unicode. Retrieved 1/7 statements.


def test_case_0():
    var_0 = '^test$'
    var_1 = 'pattern'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_from_json_schema_type_array_with_items_list. Retrieved 10/18 statements.


def test_case_0():
    var_0 = []
    var_1 = 'items'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = 'integer'
    var_6 = {var_2: var_5}
    var_7 = [var_4, var_6]
    var_8 = {var_1: var_7}
    var_9 = 'array'
    var_10 = False



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




import typesystem.composites as module_1


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



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_type_from_json_schema_with_single_type. Retrieved 4/7 statements.
# Partially parsed test_type_from_json_schema_with_nullable_single_type. Retrieved 6/9 statements.
# Partially parsed test_type_from_json_schema_with_multiple_types. Retrieved 7/11 statements.
# Partially parsed test_type_from_json_schema_with_nullable_multiple_types. Retrieved 9/14 statements.
# Partially parsed test_type_from_json_schema_with_number_and_integer. Retrieved 7/11 statements.
# Partially parsed test_type_from_json_schema_with_no_type_specified. Retrieved 7/15 statements.
# Partially parsed test_type_from_json_schema_with_only_null. Retrieved 4/7 statements.
# Partially parsed test_type_from_json_schema_with_only_null_and_allow_null_false. Retrieved 4/8 statements.
# Partially parsed test_type_from_json_schema_with_constraints. Retrieved 8/11 statements.
# Partially parsed test_type_from_json_schema_with_invalid_constraint. Retrieved 8/12 statements.
# Partially parsed test_type_from_json_schema_with_array_type. Retrieved 9/12 statements.
# Partially parsed test_type_from_json_schema_with_object_type. Retrieved 10/13 statements.
# Partially parsed test_type_from_json_schema_with_boolean_type. Retrieved 5/9 statements.
# Partially parsed test_type_from_json_schema_with_number_type. Retrieved 8/11 statements.
# Partially parsed test_type_from_json_schema_with_string_type. Retrieved 8/11 statements.
# Partially parsed test_type_from_json_schema_with_empty_type_strings. Retrieved 9/17 statements.
# Partially parsed test_type_from_json_schema_with_type_null_in_union. Retrieved 7/11 statements.
# Partially parsed test_type_from_json_schema_with_type_removal_of_integer_when_number_present. Retrieved 7/11 statements.
# Partially parsed test_type_from_json_schema_with_invalid_type_string. Retrieved 3/6 statements.


def test_case_0():
    var_0 = []
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'hello'

def test_case_0():
    var_0 = []
    var_1 = 'type'
    var_2 = 'string'
    var_3 = 'null'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = None

def test_case_0():
    var_0 = []
    var_1 = 'type'
    var_2 = 'string'
    var_3 = 'integer'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = 'hello'
    var_7 = 123

def test_case_0():
    var_0 = []
    var_1 = 'type'
    var_2 = 'string'
    var_3 = 'integer'
    var_4 = 'null'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = None
    var_8 = 'hello'
    var_9 = 123

def test_case_0():
    var_0 = []
    var_1 = 'type'
    var_2 = 'number'
    var_3 = 'integer'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = 3.14
    var_7 = 42

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = None
    var_3 = True
    var_4 = {}
    var_5 = []
    var_6 = 3.14
    var_7 = 'hello'

def test_case_0():
    var_0 = []
    var_1 = 'type'
    var_2 = 'null'
    var_3 = {var_1: var_2}
    var_4 = None

def test_case_0():
    var_0 = []
    var_1 = 'type'
    var_2 = 'null'
    var_3 = {var_1: var_2}
    var_4 = 'not null'
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = []
    var_1 = 'type'
    var_2 = 'minimum'
    var_3 = 'maximum'
    var_4 = 'integer'
    var_5 = 0
    var_6 = 10
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = 5

def test_case_0():
    var_0 = []
    var_1 = 'type'
    var_2 = 'minimum'
    var_3 = 'maximum'
    var_4 = 'integer'
    var_5 = 0
    var_6 = 10
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = 15
    var_9 = bool(False)
    assert var_9 is True

def test_case_0():
    var_0 = []
    var_1 = 'type'
    var_2 = 'items'
    var_3 = 'array'
    var_4 = 'string'
    var_5 = {var_1: var_4}
    var_6 = {var_1: var_3, var_2: var_5}
    var_7 = 'hello'
    var_8 = 'world'
    var_9 = [var_7, var_8]

def test_case_0():
    var_0 = []
    var_1 = 'type'
    var_2 = 'properties'
    var_3 = 'object'
    var_4 = 'name'
    var_5 = 'string'
    var_6 = {var_1: var_5}
    var_7 = {var_4: var_6}
    var_8 = {var_1: var_3, var_2: var_7}
    var_9 = 'Alice'
    var_10 = {var_4: var_9}

def test_case_0():
    var_0 = []
    var_1 = 'type'
    var_2 = 'boolean'
    var_3 = {var_1: var_2}
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = []
    var_1 = 'type'
    var_2 = 'minimum'
    var_3 = 'maximum'
    var_4 = 'number'
    var_5 = 0.0
    var_6 = 1.0
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = 0.5

def test_case_0():
    var_0 = []
    var_1 = 'type'
    var_2 = 'minLength'
    var_3 = 'maxLength'
    var_4 = 'string'
    var_5 = 1
    var_6 = 10
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = 'hello'

def test_case_0():
    var_0 = []
    var_1 = 'type'
    var_2 = []
    var_3 = {var_1: var_2}
    var_4 = None
    var_5 = True
    var_6 = {}
    var_7 = []
    var_8 = 3.14
    var_9 = 'hello'

def test_case_0():
    var_0 = []
    var_1 = 'type'
    var_2 = 'null'
    var_3 = 'string'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = None
    var_7 = 'hello'

def test_case_0():
    var_0 = []
    var_1 = 'type'
    var_2 = 'number'
    var_3 = 'integer'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = 3.14
    var_7 = 42

def test_case_0():
    var_0 = []
    var_1 = 'type'
    var_2 = 'invalid'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #2
#--------------------------




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


def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'integer'
    var_5 = {var_3: var_4}
    var_6 = bool(var_2 == var_5)
    assert var_6 is True


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


def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'number'
    var_5 = {var_3: var_4}
    var_6 = bool(var_2 == var_5)
    assert var_6 is True


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


def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'boolean'
    var_5 = {var_3: var_4}
    var_6 = bool(var_2 == var_5)
    assert var_6 is True


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


def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'array'
    var_5 = {var_3: var_4}
    var_6 = bool(var_2 == var_5)
    assert var_6 is True


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


def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'object'
    var_5 = {var_3: var_4}
    var_6 = bool(var_2 == var_5)
    assert var_6 is True


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


def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, required=var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'type'
    var_9 = 'properties'
    var_10 = 'required'
    var_11 = 'object'
    var_12 = 'string'
    var_13 = {var_8: var_12}
    var_14 = {var_0: var_13}
    var_15 = [var_0]
    var_16 = {var_8: var_11, var_9: var_14, var_10: var_15}
    var_17 = bool(var_7 == var_16)
    assert var_17 is True

import typesystem.schemas as module_1


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
    var_9 = 'type'
    var_10 = 'properties'
    var_11 = 'required'
    var_12 = 'object'
    var_13 = 'string'
    var_14 = {var_9: var_13}
    var_15 = {var_0: var_14}
    var_16 = [var_0]
    var_17 = {var_9: var_12, var_10: var_15, var_11: var_16}
    var_18 = bool(var_8 == var_17)
    assert var_18 is True

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


def test_case_0():
    var_0 = 'fixed'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'const'
    var_5 = {var_4: var_0}
    var_6 = bool(var_3 == var_5)
    assert var_6 is True


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
    var_9 = 'type'
    var_10 = 'string'
    var_11 = {var_9: var_10}
    var_12 = 'integer'
    var_13 = {var_9: var_12}
    var_14 = [var_11, var_13]
    var_15 = {var_8: var_14}
    var_16 = bool(var_7 == var_15)
    assert var_16 is True

import typesystem.composites as module_1


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


def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'test'
    var_3 = {}
    var_4 = module_0.Const(var_2, **var_3)
    var_5 = [var_1, var_4]
    var_6 = {}
    var_7 = module_1.AllOf(var_5, **var_6)
    var_8 = module_2.to_json_schema(var_7)
    var_9 = 'allOf'
    var_10 = 'type'
    var_11 = 'string'
    var_12 = {var_10: var_11}
    var_13 = 'const'
    var_14 = {var_13: var_2}
    var_15 = [var_12, var_14]
    var_16 = {var_9: var_15}
    var_17 = bool(var_8 == var_16)
    assert var_17 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'valid'
    var_3 = {}
    var_4 = module_0.Const(var_2, **var_3)
    var_5 = {}
    var_6 = module_1.IfThenElse(var_1, var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_from_json_schema_type_number. Retrieved 13/16 statements.
# Partially parsed test_from_json_schema_type_integer. Retrieved 13/16 statements.
# Partially parsed test_from_json_schema_type_string. Retrieved 13/16 statements.
# Partially parsed test_from_json_schema_type_boolean. Retrieved 5/8 statements.
# Partially parsed test_from_json_schema_type_object. Retrieved 28/37 statements.
# Partially parsed test_from_json_schema_type_array. Retrieved 18/23 statements.
# Partially parsed test_from_json_schema_type_with_allow_null. Retrieved 3/6 statements.
# Partially parsed test_from_json_schema_type_with_no_default. Retrieved 3/7 statements.
# Partially parsed test_from_json_schema_type_with_min_length_zero. Retrieved 5/8 statements.
# Partially parsed test_from_json_schema_type_with_min_length_one. Retrieved 5/8 statements.
# Partially parsed test_from_json_schema_type_with_min_length_greater_than_one. Retrieved 5/8 statements.
# Partially parsed test_from_json_schema_type_with_additional_properties_field. Retrieved 7/12 statements.
# Partially parsed test_from_json_schema_type_with_additional_properties_bool. Retrieved 5/8 statements.
# Partially parsed test_from_json_schema_type_with_additional_properties_none. Retrieved 5/8 statements.
# Partially parsed test_from_json_schema_type_with_items_list. Retrieved 11/20 statements.
# Partially parsed test_from_json_schema_type_with_items_single. Retrieved 7/12 statements.
# Partially parsed test_from_json_schema_type_with_additional_items_field. Retrieved 7/12 statements.
# Partially parsed test_from_json_schema_type_with_additional_items_bool. Retrieved 4/7 statements.
# Partially parsed test_from_json_schema_type_with_no_items. Retrieved 3/6 statements.
# Partially parsed test_from_json_schema_type_with_no_additional_items. Retrieved 3/6 statements.


def test_case_0():
    var_0 = []
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'exclusiveMinimum'
    var_4 = 'exclusiveMaximum'
    var_5 = 'multipleOf'
    var_6 = 'default'
    var_7 = 0
    var_8 = 10
    var_9 = 2
    var_10 = 5.0
    var_11 = {var_1: var_7, var_2: var_8, var_3: var_7, var_4: var_8, var_5: var_9, var_6: var_10}
    var_12 = 'number'
    var_13 = False

def test_case_0():
    var_0 = []
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'exclusiveMinimum'
    var_4 = 'exclusiveMaximum'
    var_5 = 'multipleOf'
    var_6 = 'default'
    var_7 = 0
    var_8 = 10
    var_9 = 2
    var_10 = 5
    var_11 = {var_1: var_7, var_2: var_8, var_3: var_7, var_4: var_8, var_5: var_9, var_6: var_10}
    var_12 = 'integer'
    var_13 = False

def test_case_0():
    var_0 = []
    var_1 = 'minLength'
    var_2 = 'maxLength'
    var_3 = 'format'
    var_4 = 'pattern'
    var_5 = 'default'
    var_6 = 5
    var_7 = 10
    var_8 = 'email'
    var_9 = '^[a-z]+$'
    var_10 = 'hello'
    var_11 = {var_1: var_6, var_2: var_7, var_3: var_8, var_4: var_9, var_5: var_10}
    var_12 = 'string'
    var_13 = False

def test_case_0():
    var_0 = []
    var_1 = 'default'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = 'boolean'
    var_5 = False

def test_case_0():
    var_0 = []
    var_1 = 'properties'
    var_2 = 'patternProperties'
    var_3 = 'additionalProperties'
    var_4 = 'propertyNames'
    var_5 = 'minProperties'
    var_6 = 'maxProperties'
    var_7 = 'required'
    var_8 = 'default'
    var_9 = 'name'
    var_10 = 'type'
    var_11 = 'string'
    var_12 = {var_10: var_11}
    var_13 = {var_9: var_12}
    var_14 = '^x_'
    var_15 = 'integer'
    var_16 = {var_10: var_15}
    var_17 = {var_14: var_16}
    var_18 = False
    var_19 = 'pattern'
    var_20 = '^[a-z]+$'
    var_21 = {var_19: var_20}
    var_22 = 1
    var_23 = 5
    var_24 = [var_9]
    var_25 = 'test'
    var_26 = {var_9: var_25}
    var_27 = {var_1: var_13, var_2: var_17, var_3: var_18, var_4: var_21, var_5: var_22, var_6: var_23, var_7: var_24, var_8: var_26}
    var_28 = 'object'

def test_case_0():
    var_0 = []
    var_1 = 'items'
    var_2 = 'additionalItems'
    var_3 = 'minItems'
    var_4 = 'maxItems'
    var_5 = 'uniqueItems'
    var_6 = 'default'
    var_7 = 'type'
    var_8 = 'string'
    var_9 = {var_7: var_8}
    var_10 = False
    var_11 = 1
    var_12 = 5
    var_13 = True
    var_14 = 'a'
    var_15 = 'b'
    var_16 = [var_14, var_15]
    var_17 = {var_1: var_9, var_2: var_10, var_3: var_11, var_4: var_12, var_5: var_13, var_6: var_16}
    var_18 = 'array'

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = 'string'
    var_3 = True

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = 'string'
    var_3 = False

def test_case_0():
    var_0 = []
    var_1 = 'minLength'
    var_2 = 0
    var_3 = {var_1: var_2}
    var_4 = 'string'
    var_5 = False

def test_case_0():
    var_0 = []
    var_1 = 'minLength'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = 'string'
    var_5 = False

def test_case_0():
    var_0 = []
    var_1 = 'minLength'
    var_2 = 5
    var_3 = {var_1: var_2}
    var_4 = 'string'
    var_5 = False

def test_case_0():
    var_0 = []
    var_1 = 'additionalProperties'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'object'
    var_7 = False

def test_case_0():
    var_0 = []
    var_1 = 'additionalProperties'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = 'object'
    var_5 = False

def test_case_0():
    var_0 = []
    var_1 = 'additionalProperties'
    var_2 = None
    var_3 = {var_1: var_2}
    var_4 = 'object'
    var_5 = False

def test_case_0():
    var_0 = []
    var_1 = 'items'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = 'integer'
    var_6 = {var_2: var_5}
    var_7 = [var_4, var_6]
    var_8 = {var_1: var_7}
    var_9 = 'array'
    var_10 = False
    var_11 = 1

def test_case_0():
    var_0 = []
    var_1 = 'items'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'array'
    var_7 = False

def test_case_0():
    var_0 = []
    var_1 = 'additionalItems'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'array'
    var_7 = False

def test_case_0():
    var_0 = []
    var_1 = 'additionalItems'
    var_2 = False
    var_3 = {var_1: var_2}
    var_4 = 'array'

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = 'array'
    var_3 = False

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = 'array'
    var_3 = False



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_from_json_schema_with_boolean_true. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_with_boolean_false. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_with_ref. Retrieved 4/8 statements.
# Partially parsed test_from_json_schema_with_const. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_enum. Retrieved 7/8 statements.
# Partially parsed test_from_json_schema_with_allOf. Retrieved 12/13 statements.
# Partially parsed test_from_json_schema_with_anyOf. Retrieved 11/12 statements.
# Partially parsed test_from_json_schema_with_oneOf. Retrieved 11/12 statements.
# Partially parsed test_from_json_schema_with_not. Retrieved 6/7 statements.
# Partially parsed test_from_json_schema_with_if_then_else. Retrieved 13/14 statements.
# Partially parsed test_from_json_schema_with_type_string. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_type_number. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_type_integer. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_type_boolean. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_type_array. Retrieved 7/8 statements.
# Partially parsed test_from_json_schema_with_type_object. Retrieved 9/10 statements.
# Partially parsed test_from_json_schema_with_multiple_types. Retrieved 8/9 statements.
# Partially parsed test_from_json_schema_with_null_type. Retrieved 4/5 statements.
# Partially parsed test_from_json_schema_with_null_in_multiple_types. Retrieved 8/9 statements.
# Partially parsed test_from_json_schema_with_no_type. Retrieved 2/3 statements.
# Partially parsed test_from_json_schema_with_multiple_constraints. Retrieved 12/13 statements.
# Partially parsed test_from_json_schema_with_components_schemas. Retrieved 11/13 statements.


import typesystem.json_schema as module_0


def test_case_0():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)


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


def test_case_0():
    var_0 = 'const'
    var_1 = 42
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)
    var_4 = var_3.const
    assert var_4 == 42


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


def test_case_0():
    var_0 = 'allOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'maxLength'
    var_5 = 5
    var_6 = {var_1: var_2, var_4: var_5}
    var_7 = [var_3, var_6]
    var_8 = {var_0: var_7}
    var_9 = module_0.from_json_schema(var_8)
    var_10 = var_9.all_of
    var_11 = len(var_10)
    assert var_11 == 2


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


def test_case_0():
    var_0 = 'not'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = module_0.from_json_schema(var_4)


def test_case_0():
    var_0 = 'if'
    var_1 = 'then'
    var_2 = 'else'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 'maxLength'
    var_7 = 5
    var_8 = {var_3: var_4, var_6: var_7}
    var_9 = 'number'
    var_10 = {var_3: var_9}
    var_11 = {var_0: var_5, var_1: var_8, var_2: var_10}
    var_12 = module_0.from_json_schema(var_11)


def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)


def test_case_0():
    var_0 = 'type'
    var_1 = 'number'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)


def test_case_0():
    var_0 = 'type'
    var_1 = 'integer'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)


def test_case_0():
    var_0 = 'type'
    var_1 = 'boolean'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)


def test_case_0():
    var_0 = 'type'
    var_1 = 'items'
    var_2 = 'array'
    var_3 = 'string'
    var_4 = {var_0: var_3}
    var_5 = {var_0: var_2, var_1: var_4}
    var_6 = module_0.from_json_schema(var_5)


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


def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = 'number'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.from_json_schema(var_4)
    var_6 = var_5.any_of
    var_7 = len(var_6)
    assert var_7 == 2


def test_case_0():
    var_0 = 'type'
    var_1 = 'null'
    var_2 = {var_0: var_1}
    var_3 = module_0.from_json_schema(var_2)
    var_4 = var_3.const
    assert var_4 is None


def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = 'null'
    var_3 = [var_1, var_2]
    var_4 = {var_0: var_3}
    var_5 = module_0.from_json_schema(var_4)
    var_6 = var_5.allow_null
    assert var_6 is True
    var_7 = var_5.any_of
    var_8 = len(var_7)
    assert var_8 == 1


def test_case_0():
    var_0 = {}
    var_1 = module_0.from_json_schema(var_0)


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
    var_10 = var_9.all_of
    var_11 = len(var_10)
    assert var_11 == 2


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
    var_12 = '#/components/schemas/Test'


def test_case_0():
    var_0 = 'type'
    var_1 = 'default'
    var_2 = 'string'
    var_3 = 'hello'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.from_json_schema(var_4)
    var_6 = var_5.default
    assert var_6 == 'hello'



# Parsed testcases at query #5
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



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_additional_items_is_none_so_additional_items_argument_is_true. Retrieved 7/9 statements.


def test_case_0():
    var_0 = 'items'
    var_1 = 'additionalItems'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = None
    var_6 = {var_0: var_4, var_1: var_5}
    var_7 = []



# Parsed testcases at query #7
#--------------------------





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



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_predicate_at_line_22_evaluates_to_true. Retrieved 2/7 statements.



def test_case_0():
    var_0 = 'test_ref'
    var_1 = module_0.Field()
    var_2 = 'target'
    var_3 = {var_2: var_1}



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_from_json_schema_type_number. Retrieved 13/16 statements.
# Partially parsed test_from_json_schema_type_integer. Retrieved 13/16 statements.
# Partially parsed test_from_json_schema_type_string. Retrieved 13/16 statements.
# Partially parsed test_from_json_schema_type_string_allow_blank. Retrieved 5/8 statements.
# Partially parsed test_from_json_schema_type_boolean. Retrieved 4/7 statements.
# Partially parsed test_from_json_schema_type_object. Retrieved 27/40 statements.
# Partially parsed test_from_json_schema_type_object_no_properties. Retrieved 3/6 statements.
# Partially parsed test_from_json_schema_type_array. Retrieved 18/23 statements.
# Partially parsed test_from_json_schema_type_array_items_list. Retrieved 11/22 statements.
# Partially parsed test_from_json_schema_type_array_no_items. Retrieved 3/6 statements.
# Partially parsed test_from_json_schema_type_invalid_type_string. Retrieved 3/6 statements.


def test_case_0():
    var_0 = []
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'exclusiveMinimum'
    var_4 = 'exclusiveMaximum'
    var_5 = 'multipleOf'
    var_6 = 'default'
    var_7 = 0.0
    var_8 = 10.0
    var_9 = 2.0
    var_10 = 4.0
    var_11 = {var_1: var_7, var_2: var_8, var_3: var_7, var_4: var_8, var_5: var_9, var_6: var_10}
    var_12 = 'number'
    var_13 = False

def test_case_0():
    var_0 = []
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'exclusiveMinimum'
    var_4 = 'exclusiveMaximum'
    var_5 = 'multipleOf'
    var_6 = 'default'
    var_7 = 0
    var_8 = 10
    var_9 = 2
    var_10 = 4
    var_11 = {var_1: var_7, var_2: var_8, var_3: var_7, var_4: var_8, var_5: var_9, var_6: var_10}
    var_12 = 'integer'
    var_13 = True

def test_case_0():
    var_0 = []
    var_1 = 'minLength'
    var_2 = 'maxLength'
    var_3 = 'format'
    var_4 = 'pattern'
    var_5 = 'default'
    var_6 = 5
    var_7 = 10
    var_8 = 'email'
    var_9 = '^[a-z]+$'
    var_10 = 'hello'
    var_11 = {var_1: var_6, var_2: var_7, var_3: var_8, var_4: var_9, var_5: var_10}
    var_12 = 'string'
    var_13 = False

def test_case_0():
    var_0 = []
    var_1 = 'minLength'
    var_2 = 0
    var_3 = {var_1: var_2}
    var_4 = 'string'
    var_5 = False

def test_case_0():
    var_0 = []
    var_1 = 'default'
    var_2 = True
    var_3 = {var_1: var_2}
    var_4 = 'boolean'

def test_case_0():
    var_0 = []
    var_1 = 'properties'
    var_2 = 'patternProperties'
    var_3 = 'additionalProperties'
    var_4 = 'propertyNames'
    var_5 = 'minProperties'
    var_6 = 'maxProperties'
    var_7 = 'required'
    var_8 = 'default'
    var_9 = 'name'
    var_10 = 'type'
    var_11 = 'string'
    var_12 = {var_10: var_11}
    var_13 = {var_9: var_12}
    var_14 = '^x-'
    var_15 = {var_10: var_11}
    var_16 = {var_14: var_15}
    var_17 = False
    var_18 = 'pattern'
    var_19 = '^[a-z]+$'
    var_20 = {var_18: var_19}
    var_21 = 1
    var_22 = 5
    var_23 = [var_9]
    var_24 = 'test'
    var_25 = {var_9: var_24}
    var_26 = {var_1: var_13, var_2: var_16, var_3: var_17, var_4: var_20, var_5: var_21, var_6: var_22, var_7: var_23, var_8: var_25}
    var_27 = 'object'
    var_28 = 'name'
    var_29 = '^x-'

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = 'object'
    var_3 = True

def test_case_0():
    var_0 = []
    var_1 = 'items'
    var_2 = 'additionalItems'
    var_3 = 'minItems'
    var_4 = 'maxItems'
    var_5 = 'uniqueItems'
    var_6 = 'default'
    var_7 = 'type'
    var_8 = 'string'
    var_9 = {var_7: var_8}
    var_10 = False
    var_11 = 1
    var_12 = 10
    var_13 = True
    var_14 = 'a'
    var_15 = 'b'
    var_16 = [var_14, var_15]
    var_17 = {var_1: var_9, var_2: var_10, var_3: var_11, var_4: var_12, var_5: var_13, var_6: var_16}
    var_18 = 'array'

def test_case_0():
    var_0 = []
    var_1 = 'items'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = 'integer'
    var_6 = {var_2: var_5}
    var_7 = [var_4, var_6]
    var_8 = {var_1: var_7}
    var_9 = 'array'
    var_10 = True
    var_11 = 0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = 'array'
    var_3 = False

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = 'invalid'
    var_3 = False
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #10
#--------------------------





def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = bool(var_2 == var_5)
    assert var_6 is True


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


def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'integer'
    var_5 = {var_3: var_4}
    var_6 = bool(var_2 == var_5)
    assert var_6 is True


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


def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'number'
    var_5 = {var_3: var_4}
    var_6 = bool(var_2 == var_5)
    assert var_6 is True


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


def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'boolean'
    var_5 = {var_3: var_4}
    var_6 = bool(var_2 == var_5)
    assert var_6 is True


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


def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'array'
    var_5 = {var_3: var_4}
    var_6 = bool(var_2 == var_5)
    assert var_6 is True


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


def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'object'
    var_5 = {var_3: var_4}
    var_6 = bool(var_2 == var_5)
    assert var_6 is True


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


def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, required=var_4, **var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 'type'
    var_9 = 'properties'
    var_10 = 'required'
    var_11 = 'object'
    var_12 = 'string'
    var_13 = {var_8: var_12}
    var_14 = {var_0: var_13}
    var_15 = [var_0]
    var_16 = {var_8: var_11, var_9: var_14, var_10: var_15}
    var_17 = bool(var_7 == var_16)
    assert var_17 is True

import typesystem.schemas as module_1


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
    var_9 = 'type'
    var_10 = 'properties'
    var_11 = 'required'
    var_12 = 'object'
    var_13 = 'string'
    var_14 = {var_9: var_13}
    var_15 = {var_0: var_14}
    var_16 = [var_0]
    var_17 = {var_9: var_12, var_10: var_15, var_11: var_16}
    var_18 = bool(var_8 == var_17)
    assert var_18 is True

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


def test_case_0():
    var_0 = 'fixed'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'const'
    var_5 = {var_4: var_0}
    var_6 = bool(var_3 == var_5)
    assert var_6 is True


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
    var_9 = 'type'
    var_10 = 'string'
    var_11 = {var_9: var_10}
    var_12 = 'integer'
    var_13 = {var_9: var_12}
    var_14 = [var_11, var_13]
    var_15 = {var_8: var_14}
    var_16 = bool(var_7 == var_15)
    assert var_16 is True

import typesystem.composites as module_1


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


def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'test'
    var_3 = {}
    var_4 = module_0.Const(var_2, **var_3)
    var_5 = [var_1, var_4]
    var_6 = {}
    var_7 = module_1.AllOf(var_5, **var_6)
    var_8 = module_2.to_json_schema(var_7)
    var_9 = 'allOf'
    var_10 = 'type'
    var_11 = 'string'
    var_12 = {var_10: var_11}
    var_13 = 'const'
    var_14 = {var_13: var_2}
    var_15 = [var_12, var_14]
    var_16 = {var_9: var_15}
    var_17 = bool(var_8 == var_16)
    assert var_17 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'valid'
    var_3 = {}
    var_4 = module_0.Const(var_2, **var_3)
    var_5 = {}
    var_6 = module_1.IfThenElse(var_1, var_4, **var_5)
    var_7 = module_2.to_json_schema(var_6)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_additional_items_is_not_bool. Retrieved 7/12 statements.


def test_case_0():
    var_0 = []
    var_1 = 'additionalItems'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'array'
    var_7 = False



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_from_json_schema_type_array_with_items_list. Retrieved 10/18 statements.


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
    var_9 = 'array'
    var_10 = False



# Parsed testcases at query #13
#--------------------------




import typesystem.json_schema as module_1


def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Object(max_properties=var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'maxProperties'
    var_5 = bool('maxProperties' in var_3)
    assert var_5 is True
    var_6 = var_3['maxProperties']
    assert var_6 == 5



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_additional_items_is_not_bool_and_not_none. Retrieved 4/6 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(additional_items=var_1, **var_2)
    var_4 = module_1.to_json_schema(var_3)
    var_5 = 'additionalItems'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_to_json_schema_with_ifthenelse_field. Retrieved 3/4 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = bool(var_2 == var_5)
    assert var_6 is True


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


def test_case_0():
    var_0 = {}
    var_1 = module_0.Integer(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'integer'
    var_5 = {var_3: var_4}
    var_6 = bool(var_2 == var_5)
    assert var_6 is True


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


def test_case_0():
    var_0 = {}
    var_1 = module_0.Float(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'number'
    var_5 = {var_3: var_4}
    var_6 = bool(var_2 == var_5)
    assert var_6 is True


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


def test_case_0():
    var_0 = {}
    var_1 = module_0.Boolean(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'boolean'
    var_5 = {var_3: var_4}
    var_6 = bool(var_2 == var_5)
    assert var_6 is True


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


def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'array'
    var_5 = {var_3: var_4}
    var_6 = bool(var_2 == var_5)
    assert var_6 is True


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


def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = module_1.to_json_schema(var_1)
    var_3 = 'type'
    var_4 = 'object'
    var_5 = {var_3: var_4}
    var_6 = bool(var_2 == var_5)
    assert var_6 is True


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

import typesystem.schemas as module_1


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
    var_9 = 'type'
    var_10 = 'properties'
    var_11 = 'required'
    var_12 = 'object'
    var_13 = 'string'
    var_14 = {var_9: var_13}
    var_15 = {var_0: var_14}
    var_16 = [var_0]
    var_17 = {var_9: var_12, var_10: var_15, var_11: var_16}
    var_18 = bool(var_8 == var_17)
    assert var_18 is True

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


def test_case_0():
    var_0 = 'fixed'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = 'const'
    var_5 = {var_4: var_0}
    var_6 = bool(var_3 == var_5)
    assert var_6 is True


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
    var_9 = 'type'
    var_10 = 'string'
    var_11 = {var_9: var_10}
    var_12 = 'integer'
    var_13 = {var_9: var_12}
    var_14 = [var_11, var_13]
    var_15 = {var_8: var_14}
    var_16 = bool(var_7 == var_15)
    assert var_16 is True

import typesystem.composites as module_1


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


def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'test'
    var_3 = {}
    var_4 = module_0.Const(var_2, **var_3)
    var_5 = [var_1, var_4]
    var_6 = {}
    var_7 = module_1.AllOf(var_5, **var_6)
    var_8 = module_2.to_json_schema(var_7)
    var_9 = 'allOf'
    var_10 = 'type'
    var_11 = 'string'
    var_12 = {var_10: var_11}
    var_13 = 'const'
    var_14 = {var_13: var_2}
    var_15 = [var_12, var_14]
    var_16 = {var_9: var_15}
    var_17 = bool(var_8 == var_16)
    assert var_17 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = {}
    var_5 = module_1.IfThenElse(var_1, var_3, **var_4)



# Parsed testcases at query #16
#--------------------------





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



# Parsed testcases at query #17
#--------------------------






# Parsed testcases at query #18
#--------------------------






