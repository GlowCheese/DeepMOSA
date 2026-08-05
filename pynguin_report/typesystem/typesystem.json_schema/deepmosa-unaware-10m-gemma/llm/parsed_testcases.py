####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.fields as module_0
import typesystem.json_schema as module_1
import typesystem.composites as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = True
    var_1 = 5
    var_2 = 10
    var_3 = 'email'
    var_4 = module_0.String(max_length=var_2, min_length=var_1, format=var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = False
    var_7 = 100
    var_8 = module_0.Integer(minimum=var_6, maximum=var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = module_0.Boolean()
    var_11 = module_1.to_json_schema(var_10)
    var_12 = module_0.String(min_length=var_0)
    var_13 = module_0.Array(var_12, min_items=var_0, unique_items=var_0)
    var_14 = module_1.to_json_schema(var_13)
    var_15 = 'name'
    var_16 = module_0.String()
    var_17 = {var_15: var_16}
    var_18 = [var_15]
    var_19 = module_0.Object(properties=var_17, required=var_18)
    var_20 = module_1.to_json_schema(var_19)
    var_21 = 'red'
    var_22 = (var_21, var_21)
    var_23 = 'blue'
    var_24 = (var_23, var_23)
    var_25 = [var_22, var_24]
    var_26 = module_0.Choice(choices=var_25)
    var_27 = module_1.to_json_schema(var_26)
    var_28 = 'fixed_value'
    var_29 = module_0.Const(var_28)
    var_30 = module_1.to_json_schema(var_29)
    var_31 = module_0.String()
    var_32 = module_0.Integer()
    var_33 = [var_31, var_32]
    var_34 = module_0.Union(var_33)
    var_35 = module_1.to_json_schema(var_34)
    var_36 = 'anyOf'
    var_37 = var_35[var_36]
    var_38 = len(var_37)
    assert var_38 == 2
    var_39 = module_0.String()
    var_40 = module_0.Integer()
    var_41 = [var_39, var_40]
    var_42 = module_2.AllOf(var_41)
    var_43 = module_1.to_json_schema(var_42)
    var_44 = module_0.String()
    var_45 = module_2.Not(var_44)
    var_46 = module_1.to_json_schema(var_45)
    var_47 = module_0.Integer()
    var_48 = module_0.Boolean()
    var_49 = module_2.IfThenElse(var_47, var_48)
    var_50 = module_1.to_json_schema(var_49)
    var_51 = module_0.Any()
    var_52 = module_1.to_json_schema(var_51)
    assert var_52 is True
    var_53 = module_2.NeverMatch()
    var_54 = module_1.to_json_schema(var_53)
    assert var_54 is False
    var_55 = 'User'
    var_56 = module_0.String()
    var_57 = {var_55: var_56}
    var_58 = module_3.Reference(var_55, var_57)
    var_59 = {}
    var_60 = module_1.to_json_schema(var_58, var_59)



# Parsed testcases at query #2
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'allOf'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = 'integer'
    var_6 = {var_2: var_5}
    var_7 = [var_4, var_6]
    var_8 = {var_1: var_7}
    var_9 = module_1.all_of_from_json_schema(var_8, var_0)
    var_10 = var_9.all_of
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = 0
    var_13 = var_9.all_of[var_12]
    var_14 = 1
    var_15 = var_9.all_of[var_14]
    var_16 = 'default'
    var_17 = {var_2: var_3}
    var_18 = [var_17]
    var_19 = 'hello'
    var_20 = {var_1: var_18, var_16: var_19}
    var_21 = module_1.all_of_from_json_schema(var_20, var_0)
    var_22 = {var_2: var_5}
    var_23 = 'minimum'
    var_24 = 10
    var_25 = {var_23: var_24}
    var_26 = [var_22, var_25]
    var_27 = {var_1: var_26}
    var_28 = 'number'
    var_29 = {var_2: var_28}
    var_30 = [var_27, var_29]
    var_31 = {var_1: var_30}
    var_32 = module_1.all_of_from_json_schema(var_31, var_0)
    var_33 = var_32.all_of
    var_34 = len(var_33)
    assert var_34 == 2
    var_35 = var_32.all_of[var_12]
    var_36 = var_32.all_of[var_14]
    var_37 = []
    var_38 = {var_1: var_37}
    var_39 = module_1.all_of_from_json_schema(var_38, var_0)
    var_40 = var_39.all_of
    var_41 = len(var_40)
    assert var_41 == 0



# Parsed testcases at query #3
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'enum'
    var_2 = 'default'
    var_3 = 'apple'
    var_4 = 'banana'
    var_5 = 'cherry'
    var_6 = [var_3, var_4, var_5]
    var_7 = {var_1: var_6, var_2: var_3}
    var_8 = module_1.enum_from_json_schema(var_7, var_0)
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = [var_9, var_10, var_11]
    var_13 = {var_1: var_12}
    var_14 = module_1.enum_from_json_schema(var_13, var_0)
    var_15 = 'a'
    var_16 = True
    var_17 = [var_9, var_15, var_16]
    var_18 = {var_1: var_17}
    var_19 = module_1.enum_from_json_schema(var_18, var_0)
    var_20 = 'id'
    var_21 = {var_20: var_16}
    var_22 = {var_20: var_10}
    var_23 = [var_21, var_22]
    var_24 = {var_1: var_23}
    var_25 = module_1.enum_from_json_schema(var_24, var_0)



# Parsed testcases at query #4
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = module_1.type_from_json_schema(var_3, var_0)
    var_5 = 'integer'
    var_6 = {var_1: var_5}
    var_7 = module_1.type_from_json_schema(var_6, var_0)
    var_8 = 'boolean'
    var_9 = {var_1: var_8}
    var_10 = module_1.type_from_json_schema(var_9, var_0)
    var_11 = 'number'
    var_12 = {var_1: var_11}
    var_13 = module_1.type_from_json_schema(var_12, var_0)
    var_14 = [var_2, var_5]
    var_15 = {var_1: var_14}
    var_16 = module_1.type_from_json_schema(var_15, var_0)
    var_17 = 'minLength'
    var_18 = 5
    var_19 = {var_1: var_2, var_17: var_18}
    var_20 = module_1.type_from_json_schema(var_19, var_0)
    var_21 = {}
    var_22 = module_1.type_from_json_schema(var_21, var_0)
    var_23 = 'items'
    var_24 = 'array'
    var_25 = {var_1: var_2}
    var_26 = {var_1: var_24, var_23: var_25}
    var_27 = module_1.type_from_json_schema(var_26, var_0)



# Parsed testcases at query #5
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = '#/definitions/MySchema'
    var_3 = {var_1: var_2}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)
    var_5 = '#/components/schemas/User'
    var_6 = {var_1: var_5}
    var_7 = module_1.ref_from_json_schema(var_6, var_0)
    var_8 = 'https://example.com/schema.json'
    var_9 = {var_1: var_8}
    var_10 = module_1.ref_from_json_schema(var_9, var_0)
    var_11 = 'type'
    var_12 = 'string'
    var_13 = {var_11: var_12}
    var_14 = module_1.ref_from_json_schema(var_13, var_0)



# Parsed testcases at query #6
#--------------------------


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
    var_12 = 123
    var_13 = {var_1: var_7, var_2: var_9, var_3: var_11, var_4: var_12}
    var_14 = module_1.if_then_else_from_json_schema(var_13, var_0)
    var_15 = var_14.if_clause
    var_16 = var_14.then_clause
    var_17 = var_14.else_clause
    var_18 = 'number'
    var_19 = {var_5: var_18}
    var_20 = 'float'
    var_21 = {var_5: var_20}
    var_22 = {var_1: var_19, var_2: var_21}
    var_23 = module_1.if_then_else_from_json_schema(var_22, var_0)
    var_24 = var_23.if_clause
    var_25 = {var_5: var_10}
    var_26 = {var_5: var_6}
    var_27 = {var_1: var_25, var_3: var_26}
    var_28 = module_1.if_then_else_from_json_schema(var_27, var_0)
    var_29 = var_28.if_clause
    var_30 = var_28.else_clause
    var_31 = {var_5: var_6}
    var_32 = {var_5: var_6}
    var_33 = {var_1: var_31, var_2: var_32}
    var_34 = module_1.if_then_else_from_json_schema(var_33, var_0)



# Parsed testcases at query #7
#--------------------------


import typesystem.json_schema as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)
    var_2 = False
    var_3 = module_0.from_json_schema(var_2)
    var_4 = 'type'
    var_5 = 'minLength'
    var_6 = 'string'
    var_7 = 5
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = module_0.from_json_schema(var_8)
    var_10 = 'minimum'
    var_11 = 'integer'
    var_12 = 10
    var_13 = {var_4: var_11, var_10: var_12}
    var_14 = module_0.from_json_schema(var_13)
    var_15 = 'enum'
    var_16 = 'a'
    var_17 = 'b'
    var_18 = 'c'
    var_19 = [var_16, var_17, var_18]
    var_20 = {var_15: var_19}
    var_21 = module_0.from_json_schema(var_20)
    var_22 = 'const'
    var_23 = 42
    var_24 = {var_22: var_23}
    var_25 = module_0.from_json_schema(var_24)
    var_26 = 'items'
    var_27 = 'array'
    var_28 = {var_4: var_6}
    var_29 = {var_4: var_27, var_26: var_28}
    var_30 = module_0.from_json_schema(var_29)
    var_31 = var_30.constraints[var_26]
    var_32 = 'properties'
    var_33 = 'required'
    var_34 = 'object'
    var_35 = 'name'
    var_36 = 'age'
    var_37 = {var_4: var_6}
    var_38 = {var_4: var_11}
    var_39 = {var_35: var_37, var_36: var_38}
    var_40 = [var_35]
    var_41 = {var_4: var_34, var_32: var_39, var_33: var_40}
    var_42 = module_0.from_json_schema(var_41)
    var_43 = var_42.constraints[var_32][var_35]
    var_44 = 'pattern'
    var_45 = '^abc'
    var_46 = {var_4: var_6, var_5: var_7, var_44: var_45}
    var_47 = module_0.from_json_schema(var_46)
    var_48 = {}
    var_49 = module_0.from_json_schema(var_48)
    var_50 = 'components'
    var_51 = 'schemas'
    var_52 = 'User'
    var_53 = {var_4: var_6}
    var_54 = {var_52: var_53}
    var_55 = {var_51: var_54}
    var_56 = {var_50: var_55}
    var_57 = module_0.from_json_schema(var_56)
    var_58 = 'maximum'
    var_59 = 'number'
    var_60 = 100
    var_61 = {var_4: var_59, var_58: var_60}
    var_62 = module_0.from_json_schema(var_61)



# Parsed testcases at query #8
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = module_1.type_from_json_schema(var_3, var_0)
    var_5 = 'integer'
    var_6 = {var_1: var_5}
    var_7 = module_1.type_from_json_schema(var_6, var_0)
    var_8 = 'number'
    var_9 = {var_1: var_8}
    var_10 = module_1.type_from_json_schema(var_9, var_0)
    var_11 = 'boolean'
    var_12 = {var_1: var_11}
    var_13 = module_1.type_from_json_schema(var_12, var_0)
    var_14 = [var_2, var_5]
    var_15 = {var_1: var_14}
    var_16 = module_1.type_from_json_schema(var_15, var_0)
    var_17 = {}
    var_18 = module_1.type_from_json_schema(var_17, var_0)
    var_19 = 'pattern'
    var_20 = '^abc$'
    var_21 = {var_1: var_2, var_19: var_20}
    var_22 = module_1.type_from_json_schema(var_21, var_0)
    var_23 = 'minimum'
    var_24 = 10
    var_25 = {var_1: var_5, var_23: var_24}
    var_26 = module_1.type_from_json_schema(var_25, var_0)



# Parsed testcases at query #9
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'oneOf'
    var_2 = 'default'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 'integer'
    var_7 = {var_3: var_6}
    var_8 = [var_5, var_7]
    var_9 = 'some_default'
    var_10 = {var_1: var_8, var_2: var_9}
    var_11 = module_1.one_of_from_json_schema(var_10, var_0)
    var_12 = 0
    var_13 = var_11.one_of[var_12]
    var_14 = 1
    var_15 = var_11.one_of[var_14]
    var_16 = 'items'
    var_17 = 'array'
    var_18 = {var_3: var_4}
    var_19 = {var_3: var_17, var_16: var_18}
    var_20 = 'properties'
    var_21 = 'object'
    var_22 = 'name'
    var_23 = {var_3: var_4}
    var_24 = {var_22: var_23}
    var_25 = {var_3: var_21, var_20: var_24}
    var_26 = [var_19, var_25]
    var_27 = {var_1: var_26}
    var_28 = module_1.one_of_from_json_schema(var_27, var_0)
    var_29 = var_28.one_of[var_12]
    var_30 = var_28.one_of[var_14]
    var_31 = 'boolean'
    var_32 = {var_3: var_31}
    var_33 = [var_32]
    var_34 = {var_1: var_33}
    var_35 = module_1.one_of_from_json_schema(var_34, var_0)
    var_36 = 'number'
    var_37 = {var_3: var_36}
    var_38 = [var_37]
    var_39 = {var_1: var_38}
    var_40 = module_1.one_of_from_json_schema(var_39, var_0)
    var_41 = var_40.one_of[var_12]



# Parsed testcases at query #10
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1
import typesystem.fields as module_2

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
    var_13 = module_2.String()
    var_14 = module_2.Integer()
    var_15 = module_2.Boolean()
    var_16 = {var_4: var_5}
    var_17 = {var_4: var_7}
    var_18 = {var_1: var_16, var_2: var_17}
    var_19 = module_1.if_then_else_from_json_schema(var_18, var_0)
    var_20 = module_2.String()
    var_21 = module_2.Integer()
    var_22 = {var_4: var_5}
    var_23 = {var_4: var_9}
    var_24 = {var_1: var_22, var_3: var_23}
    var_25 = module_1.if_then_else_from_json_schema(var_24, var_0)
    var_26 = module_2.String()
    var_27 = module_2.Boolean()
    var_28 = 'default'
    var_29 = {var_4: var_5}
    var_30 = 'some_default'
    var_31 = {var_1: var_29, var_28: var_30}
    var_32 = module_1.if_then_else_from_json_schema(var_31, var_0)
    var_33 = module_2.String()



# Parsed testcases at query #11
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'type'
    var_2 = 'minimum'
    var_3 = 'maximum'
    var_4 = 'multipleOf'
    var_5 = 'number'
    var_6 = 0
    var_7 = 10
    var_8 = 2
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = False
    var_11 = module_1.from_json_schema_type(var_9, var_5, var_10, var_0)
    var_12 = 'exclusiveMinimum'
    var_13 = 'integer'
    var_14 = 1
    var_15 = {var_1: var_13, var_2: var_14, var_12: var_10}
    var_16 = False
    var_17 = module_1.from_json_schema_type(var_15, var_13, var_16, var_0)
    var_18 = 'minLength'
    var_19 = 'maxLength'
    var_20 = 'pattern'
    var_21 = 'format'
    var_22 = 'string'
    var_23 = 5
    var_24 = '^abc'
    var_25 = 'email'
    var_26 = {var_1: var_22, var_18: var_23, var_19: var_7, var_20: var_24, var_21: var_25}
    var_27 = True
    var_28 = module_1.from_json_schema_type(var_26, var_22, var_27, var_0)
    var_29 = 'default'
    var_30 = 'boolean'
    var_31 = True
    var_32 = {var_1: var_30, var_29: var_31}
    var_33 = False
    var_34 = module_1.from_json_schema_type(var_32, var_30, var_33, var_0)
    var_35 = 'items'
    var_36 = 'minItems'
    var_37 = 'maxItems'
    var_38 = 'uniqueItems'
    var_39 = 'array'
    var_40 = {var_1: var_22}
    var_41 = True
    var_42 = {var_1: var_39, var_35: var_40, var_36: var_31, var_37: var_23, var_38: var_41}
    var_43 = False
    var_44 = module_1.from_json_schema_type(var_42, var_39, var_43, var_0)
    var_45 = var_44.items
    var_46 = 'additionalItems'
    var_47 = False
    var_48 = {var_1: var_39, var_46: var_47}
    var_49 = False
    var_50 = module_1.from_json_schema_type(var_48, var_39, var_49, var_0)
    var_51 = 'properties'
    var_52 = 'required'
    var_53 = 'minProperties'
    var_54 = 'additionalProperties'
    var_55 = 'object'
    var_56 = 'name'
    var_57 = 'age'
    var_58 = {var_1: var_22}
    var_59 = {var_1: var_13}
    var_60 = {var_56: var_58, var_57: var_59}
    var_61 = [var_56]
    var_62 = True
    var_63 = {var_1: var_55, var_51: var_60, var_52: var_61, var_53: var_41, var_54: var_62}
    var_64 = False
    var_65 = module_1.from_json_schema_type(var_63, var_55, var_64, var_0)
    var_66 = var_65.properties[var_56]
    var_67 = var_65.properties[var_57]
    var_68 = 'patternProperties'
    var_69 = '^prop_'
    var_70 = {var_1: var_22}
    var_71 = {var_69: var_70}
    var_72 = {var_1: var_55, var_68: var_71}
    var_73 = False
    var_74 = module_1.from_json_schema_type(var_72, var_55, var_73, var_0)
    var_75 = var_74.pattern_properties[var_69]
    var_76 = 'null'
    var_77 = True
    var_78 = {var_1: var_5, var_76: var_77}
    var_79 = True
    var_80 = module_1.from_json_schema_type(var_78, var_5, var_79, var_0)



# Parsed testcases at query #12
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'User'
    var_1 = 'id'
    var_2 = False
    var_3 = module_0.Integer()
    var_4 = {var_1: var_3}
    var_5 = module_0.Object(properties=var_4)
    var_6 = {var_0: var_5}
    var_7 = module_0.Integer()
    var_8 = {var_1: var_7}
    var_9 = module_0.Object(properties=var_8)
    var_10 = module_1.Reference(var_0, var_6)
    var_11 = {}
    var_12 = module_2.to_json_schema(var_10, var_11)

def test_case_0():
    var_0 = '[a-z]'

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.String()
    var_2 = module_0.Integer()
    var_3 = [var_1, var_2]
    var_4 = True
    var_5 = module_0.String()
    var_6 = module_0.Array(var_3, var_5)
    var_7 = module_1.to_json_schema(var_6)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = '.*'
    var_1 = False
    var_2 = module_0.String()
    var_3 = {var_0: var_2}
    var_4 = module_0.Object(pattern_properties=var_3, additional_properties=var_1)
    var_5 = module_1.to_json_schema(var_4)



# Parsed testcases at query #13
#--------------------------


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.String()
    var_2 = True
    var_3 = module_0.Integer()
    var_4 = 'id'
    var_5 = 'username'
    var_6 = {var_4: var_3, var_5: var_1}
    var_7 = module_0.Object(properties=var_6, additional_properties=var_0)
    var_8 = 'type'
    var_9 = 'properties'
    var_10 = 'additionalProperties'
    var_11 = 'object'
    var_12 = 'integer'
    var_13 = 'null'
    var_14 = [var_12, var_13]
    var_15 = {var_8: var_14}
    var_16 = 'string'
    var_17 = {var_8: var_16}
    var_18 = {var_4: var_15, var_5: var_17}
    var_19 = {var_8: var_11, var_9: var_18, var_10: var_0}
    var_20 = module_1.to_json_schema(var_7)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'User'
    var_1 = {}
    var_2 = module_0.Reference(var_0, var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = '$ref'
    var_5 = var_3[var_4]
    var_6 = '#/components/schemas/'

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = module_0.String()
    var_2 = module_0.Integer()
    var_3 = module_1.IfThenElse(var_0, var_1, var_2)
    var_4 = module_2.to_json_schema(var_3)

def test_case_0():
    pass



# Parsed testcases at query #14
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = module_1.type_from_json_schema(var_3, var_0)
    var_5 = 'integer'
    var_6 = {var_1: var_5}
    var_7 = module_1.type_from_json_schema(var_6, var_0)
    var_8 = 'boolean'
    var_9 = {var_1: var_8}
    var_10 = module_1.type_from_json_schema(var_9, var_0)
    var_11 = 'number'
    var_12 = {var_1: var_11}
    var_13 = module_1.type_from_json_schema(var_12, var_0)
    var_14 = 'items'
    var_15 = 'array'
    var_16 = {var_1: var_2}
    var_17 = {var_1: var_15, var_14: var_16}
    var_18 = module_1.type_from_json_schema(var_17, var_0)
    var_19 = var_18.items
    var_20 = 'properties'
    var_21 = 'object'
    var_22 = 'name'
    var_23 = {var_1: var_2}
    var_24 = {var_22: var_23}
    var_25 = {var_1: var_21, var_20: var_24}
    var_26 = module_1.type_from_json_schema(var_25, var_0)
    var_27 = var_26.properties[var_22]
    var_28 = [var_2, var_5]
    var_29 = {var_1: var_28}
    var_30 = module_1.type_from_json_schema(var_29, var_0)
    var_31 = var_30.any_of
    var_32 = [t for t in var_31]
    var_33 = 'null'
    var_34 = [var_2, var_33]
    var_35 = {var_1: var_34}
    var_36 = module_1.type_from_json_schema(var_35, var_0)
    var_37 = []
    var_38 = {var_1: var_37}
    var_39 = module_1.type_from_json_schema(var_38, var_0)



# Parsed testcases at query #15
#--------------------------


import typesystem.json_schema as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)
    var_2 = False
    var_3 = module_0.from_json_schema(var_2)
    var_4 = 'type'
    var_5 = 'string'
    var_6 = {var_4: var_5}
    var_7 = module_0.from_json_schema(var_6)
    var_8 = 'minimum'
    var_9 = 'integer'
    var_10 = {var_4: var_9, var_8: var_2}
    var_11 = module_0.from_json_schema(var_10)
    var_12 = 'pattern'
    var_13 = '^[a-z]+$'
    var_14 = {var_4: var_5, var_12: var_13}
    var_15 = module_0.from_json_schema(var_14)
    var_16 = 'enum'
    var_17 = 'a'
    var_18 = 'b'
    var_19 = 'c'
    var_20 = [var_17, var_18, var_19]
    var_21 = {var_16: var_20}
    var_22 = module_0.from_json_schema(var_21)
    var_23 = 'items'
    var_24 = 'array'
    var_25 = {var_4: var_9}
    var_26 = {var_4: var_24, var_23: var_25}
    var_27 = module_0.from_json_schema(var_26)
    var_28 = var_27.items
    var_29 = 'properties'
    var_30 = 'required'
    var_31 = 'object'
    var_32 = 'name'
    var_33 = 'age'
    var_34 = {var_4: var_5}
    var_35 = {var_4: var_9}
    var_36 = {var_32: var_34, var_33: var_35}
    var_37 = [var_32]
    var_38 = {var_4: var_31, var_29: var_36, var_30: var_37}
    var_39 = module_0.from_json_schema(var_38)
    var_40 = var_39.properties[var_32]
    var_41 = 'minLength'
    var_42 = 5
    var_43 = {var_4: var_5, var_41: var_42}
    var_44 = module_0.from_json_schema(var_43)
    var_45 = {}
    var_46 = module_0.from_json_schema(var_45)
    var_47 = 'components'
    var_48 = 'schemas'
    var_49 = 'User'
    var_50 = 'id'
    var_51 = {var_4: var_9}
    var_52 = {var_50: var_51}
    var_53 = {var_4: var_31, var_29: var_52}
    var_54 = {var_49: var_53}
    var_55 = {var_48: var_54}
    var_56 = {var_47: var_55}
    var_57 = module_1.Definitions()
    var_58 = module_0.from_json_schema(var_56, var_57)
    var_59 = '#/components/schemas/User'
    var_60 = var_57[var_59]
    var_61 = '$ref'
    var_62 = '#/definitions/User'
    var_63 = {var_61: var_62}
    var_64 = module_0.from_json_schema(var_63, var_57)



# Parsed testcases at query #16
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = module_1.type_from_json_schema(var_3, var_0)
    var_5 = 'hello'
    var_6 = var_4.validate(var_5)
    assert var_6 is None
    var_7 = 123
    var_8 = var_4.validate(var_7)
    var_9 = 'integer'
    var_10 = {var_7: var_9}
    var_11 = module_1.type_from_json_schema(var_10, var_0)
    var_12 = 42
    var_13 = var_11.validate(var_12)
    assert var_13 is None
    var_14 = '42'
    var_15 = var_11.validate(var_14)
    var_16 = 'boolean'
    var_17 = {var_14: var_16}
    var_18 = module_1.type_from_json_schema(var_17, var_0)
    var_19 = True
    var_20 = var_18.validate(var_19)
    assert var_20 is None
    var_21 = False
    var_22 = var_18.validate(var_21)
    assert var_22 is None
    var_23 = [var_15, var_9]
    var_24 = {var_14: var_23}
    var_25 = module_1.type_from_json_schema(var_24, var_0)
    var_26 = '123'
    var_27 = var_25.validate(var_26)
    assert var_27 is None
    var_28 = 123
    var_29 = var_25.validate(var_28)
    assert var_29 is None
    var_30 = True
    var_31 = var_25.validate(var_30)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'type'
    var_2 = 'minLength'
    var_3 = 'string'
    var_4 = 5
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_1.type_from_json_schema(var_5, var_0)
    var_7 = 'minimum'
    var_8 = 'number'
    var_9 = 10
    var_10 = {var_1: var_8, var_7: var_9}
    var_11 = module_1.type_from_json_schema(var_10, var_0)

import typesystem.schemas as module_0

def test_case_0():
    var_0 = module_0.Definitions()



# Parsed testcases at query #17
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = module_1.type_from_json_schema(var_3, var_0)
    var_5 = 'integer'
    var_6 = {var_1: var_5}
    var_7 = module_1.type_from_json_schema(var_6, var_0)
    var_8 = 'number'
    var_9 = {var_1: var_8}
    var_10 = module_1.type_from_json_schema(var_9, var_0)
    var_11 = 'boolean'
    var_12 = {var_1: var_11}
    var_13 = module_1.type_from_json_schema(var_12, var_0)
    var_14 = 'items'
    var_15 = 'array'
    var_16 = {var_1: var_2}
    var_17 = {var_1: var_15, var_14: var_16}
    var_18 = module_1.type_from_json_schema(var_17, var_0)
    var_19 = var_18.items
    var_20 = 'properties'
    var_21 = 'object'
    var_22 = 'name'
    var_23 = {var_1: var_2}
    var_24 = {var_22: var_23}
    var_25 = {var_1: var_21, var_20: var_24}
    var_26 = module_1.type_from_json_schema(var_25, var_0)
    var_27 = var_26.properties[var_22]
    var_28 = [var_2, var_5]
    var_29 = {var_1: var_28}
    var_30 = module_1.type_from_json_schema(var_29, var_0)
    var_31 = var_30.any_of
    var_32 = [t for t in var_31]
    var_33 = 'null'
    var_34 = [var_2, var_33]
    var_35 = {var_1: var_34}
    var_36 = module_1.type_from_json_schema(var_35, var_0)
    var_37 = var_36.any_of
    var_38 = any(var_2)
    var_39 = []
    var_40 = {var_37: var_39}
    var_41 = module_1.type_from_json_schema(var_40, var_0)
    var_42 = 'minLength'
    var_43 = 5
    var_44 = {var_37: var_2, var_42: var_43}
    var_45 = module_1.type_from_json_schema(var_44, var_0)
    var_46 = 'age'
    var_47 = 'tags'
    var_48 = 'minimum'
    var_49 = 18
    var_50 = {var_37: var_5, var_48: var_49}
    var_51 = {var_37: var_2}
    var_52 = {var_37: var_15, var_14: var_51}
    var_53 = {var_46: var_50, var_47: var_52}
    var_54 = {var_37: var_21, var_20: var_53}
    var_55 = module_1.type_from_json_schema(var_54, var_0)
    var_56 = var_55.properties[var_46]
    var_57 = var_55.properties[var_47]



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'allOf'
    var_2 = 'default'
    var_3 = 'type'
    var_4 = 'minLength'
    var_5 = 'string'
    var_6 = 5
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'minimum'
    var_9 = 'integer'
    var_10 = 10
    var_11 = {var_3: var_9, var_8: var_10}
    var_12 = [var_7, var_11]
    var_13 = 'some_default'
    var_14 = {var_1: var_12, var_2: var_13}
    var_15 = module_1.all_of_from_json_schema(var_14, var_0)
    var_16 = var_15.all_of
    var_17 = len(var_16)
    assert var_17 == 2
    var_18 = 0
    var_19 = var_15.all_of[var_18]
    var_20 = 1
    var_21 = var_15.all_of[var_20]
    var_22 = 'boolean'
    var_23 = {var_3: var_22}
    var_24 = [var_23]
    var_25 = {var_1: var_24}
    var_26 = module_1.all_of_from_json_schema(var_25, var_0)
    var_27 = var_26.all_of
    var_28 = len(var_27)
    assert var_28 == 1
    var_29 = var_26.all_of[var_18]
    var_30 = 'items'
    var_31 = 'array'
    var_32 = 'number'
    var_33 = {var_3: var_32}
    var_34 = {var_3: var_31, var_30: var_33}
    var_35 = 'properties'
    var_36 = 'object'
    var_37 = 'name'
    var_38 = {var_3: var_5}
    var_39 = {var_37: var_38}
    var_40 = {var_3: var_36, var_35: var_39}
    var_41 = [var_34, var_40]
    var_42 = {var_1: var_41}
    var_43 = module_1.all_of_from_json_schema(var_42, var_0)
    var_44 = var_43.all_of[var_18]
    var_45 = var_43.all_of[var_20]
    var_46 = {var_3: var_5}
    var_47 = [var_46]
    var_48 = {var_1: var_47}
    var_49 = module_1.all_of_from_json_schema(var_48, var_0)
    var_50 = 'pattern'
    var_51 = '^abc'
    var_52 = {var_3: var_5, var_50: var_51}
    var_53 = 'enum'
    var_54 = 'val1'
    var_55 = 'val2'
    var_56 = [var_54, var_55]
    var_57 = {var_53: var_56}
    var_58 = [var_52, var_57]
    var_59 = {var_1: var_58}
    var_60 = module_1.all_of_from_json_schema(var_59, var_0)
    var_61 = var_60.all_of[var_18]
    var_62 = var_60.all_of[var_20]



# Parsed testcases at query #2
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = '#/definitions/MyType'
    var_3 = {var_1: var_2}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)
    var_5 = 'https://example.com/schema'
    var_6 = {var_1: var_5}
    var_7 = module_1.ref_from_json_schema(var_6, var_0)
    var_8 = 'type'
    var_9 = 'string'
    var_10 = {var_8: var_9}
    var_11 = module_1.ref_from_json_schema(var_10, var_0)



# Parsed testcases at query #3
#--------------------------


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
    var_12 = 'some_default'
    var_13 = {var_1: var_7, var_2: var_9, var_3: var_11, var_4: var_12}
    var_14 = module_1.if_then_else_from_json_schema(var_13, var_0)
    var_15 = 'number'
    var_16 = {var_5: var_15}
    var_17 = {var_5: var_6}
    var_18 = {var_1: var_16, var_2: var_17}
    var_19 = module_1.if_then_else_from_json_schema(var_18, var_0)
    var_20 = {var_5: var_10}
    var_21 = 'items'
    var_22 = 'array'
    var_23 = {var_5: var_6}
    var_24 = {var_5: var_22, var_21: var_23}
    var_25 = {var_1: var_20, var_3: var_24}
    var_26 = module_1.if_then_else_from_json_schema(var_25, var_0)
    var_27 = 'const'
    var_28 = 123
    var_29 = {var_27: var_28}
    var_30 = {var_1: var_29}
    var_31 = module_1.if_then_else_from_json_schema(var_30, var_0)
    var_32 = {var_5: var_6}
    var_33 = 42
    var_34 = {var_1: var_32, var_4: var_33}
    var_35 = module_1.if_then_else_from_json_schema(var_34, var_0)



# Parsed testcases at query #4
#--------------------------


import typesystem.fields as module_0
import typesystem.json_schema as module_1
import typesystem.composites as module_2
import re as module_3
import typesystem.schemas as module_4

def test_case_0():
    var_0 = {}
    var_1 = 'NO_DEFAULT'
    var_2 = module_0.Any()
    var_3 = module_1.to_json_schema(var_2)
    assert var_3 is True
    var_4 = module_2.NeverMatch()
    var_5 = module_1.to_json_schema(var_4)
    assert var_5 is False
    var_6 = False
    var_7 = True
    var_8 = module_0.String(allow_blank=var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = 5
    var_11 = 10
    var_12 = '^[a-z]+$'
    var_13 = module_3.compile(var_12)
    var_14 = 'email'
    var_15 = module_0.String(max_length=var_11, min_length=var_10, format=var_14)
    var_16 = module_1.to_json_schema(var_15)
    var_17 = 100
    var_18 = module_0.Integer(minimum=var_6, maximum=var_17)
    var_19 = module_1.to_json_schema(var_18)
    var_20 = 0.5
    var_21 = module_0.Float(exclusive_minimum=var_20)
    var_22 = module_1.to_json_schema(var_21)
    var_23 = module_0.Boolean()
    var_24 = module_1.to_json_schema(var_23)
    var_25 = module_0.String()
    var_26 = module_0.Array(var_25, min_items=var_7)
    var_27 = module_1.to_json_schema(var_26)
    var_28 = 'name'
    var_29 = module_0.String()
    var_30 = {var_28: var_29}
    var_31 = [var_28]
    var_32 = module_0.Object(properties=var_30, required=var_31)
    var_33 = module_1.to_json_schema(var_32)
    var_34 = 'A'
    var_35 = (var_34, var_34)
    var_36 = 'B'
    var_37 = (var_36, var_36)
    var_38 = [var_35, var_37]
    var_39 = module_0.Choice(choices=var_38)
    var_40 = module_1.to_json_schema(var_39)
    var_41 = 'fixed_value'
    var_42 = module_0.Const(var_41)
    var_43 = module_1.to_json_schema(var_42)
    var_44 = module_0.String()
    var_45 = module_0.Integer()
    var_46 = [var_44, var_45]
    var_47 = module_0.Union(var_46)
    var_48 = module_1.to_json_schema(var_47)
    var_49 = 'anyOf'
    var_50 = var_48[var_49]
    var_51 = len(var_50)
    assert var_51 == 2
    var_52 = module_0.String()
    var_53 = 'val'
    var_54 = module_0.Const(var_53)
    var_55 = [var_52, var_54]
    var_56 = module_2.AllOf(var_55)
    var_57 = module_1.to_json_schema(var_56)
    var_58 = 'allOf'
    var_59 = var_57[var_58]
    var_60 = len(var_59)
    assert var_60 == 2
    var_61 = 'User'
    var_62 = {}
    var_63 = module_4.Reference(var_61, var_62)
    var_64 = module_0.String()
    var_65 = {var_61: var_64}
    var_66 = module_1.to_json_schema(var_63, var_65)
    var_67 = module_1.to_json_schema(var_2)



# Parsed testcases at query #5
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1
import typesystem.fields as module_2

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
    var_12 = 'default_value'
    var_13 = {var_1: var_7, var_2: var_9, var_3: var_11, var_4: var_12}
    var_14 = module_1.if_then_else_from_json_schema(var_13, var_0)
    var_15 = module_2.String()
    var_16 = module_2.Integer()
    var_17 = module_2.Boolean()
    var_18 = {var_5: var_6}
    var_19 = {var_5: var_8}
    var_20 = {var_1: var_18, var_2: var_19}
    var_21 = module_1.if_then_else_from_json_schema(var_20, var_0)
    var_22 = module_2.String()
    var_23 = module_2.Integer()
    var_24 = {var_5: var_6}
    var_25 = {var_5: var_10}
    var_26 = {var_1: var_24, var_3: var_25}
    var_27 = module_1.if_then_else_from_json_schema(var_26, var_0)
    var_28 = module_2.String()
    var_29 = module_2.Boolean()
    var_30 = {var_5: var_6}
    var_31 = {var_5: var_8}
    var_32 = {var_1: var_30, var_2: var_31}
    var_33 = module_1.if_then_else_from_json_schema(var_32, var_0)



# Parsed testcases at query #6
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = module_1.type_from_json_schema(var_3, var_0)
    var_5 = 'integer'
    var_6 = {var_1: var_5}
    var_7 = module_1.type_from_json_schema(var_6, var_0)
    var_8 = 'boolean'
    var_9 = {var_1: var_8}
    var_10 = module_1.type_from_json_schema(var_9, var_0)
    var_11 = 'minLength'
    var_12 = 5
    var_13 = {var_1: var_2, var_11: var_12}
    var_14 = module_1.type_from_json_schema(var_13, var_0)
    var_15 = [var_2, var_5]
    var_16 = {var_1: var_15}
    var_17 = module_1.type_from_json_schema(var_16, var_0)
    var_18 = var_17.any_of
    var_19 = [t for t in var_18]
    var_20 = 'null'
    var_21 = [var_2, var_20]
    var_22 = {var_1: var_21}
    var_23 = module_1.type_from_json_schema(var_22, var_0)
    var_24 = var_23.any_of
    var_25 = []
    var_26 = {var_1: var_25}
    var_27 = module_1.type_from_json_schema(var_26, var_0)
    var_28 = isinstance(var_27, var_1)
    var_29 = 'number'
    var_30 = {var_1: var_29}
    var_31 = module_1.type_from_json_schema(var_30, var_0)
    var_32 = 'multipleOf'
    var_33 = 0.5
    var_34 = {var_1: var_29, var_32: var_33}
    var_35 = module_1.type_from_json_schema(var_34, var_0)



# Parsed testcases at query #7
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'oneOf'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = 'integer'
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
    var_16 = 'default'
    var_17 = 'boolean'
    var_18 = {var_2: var_17}
    var_19 = [var_18]
    var_20 = True
    var_21 = {var_1: var_19, var_16: var_20}
    var_22 = module_1.one_of_from_json_schema(var_21, var_0)
    var_23 = 'items'
    var_24 = 'array'
    var_25 = 'properties'
    var_26 = 'object'
    var_27 = 'name'
    var_28 = {var_2: var_3}
    var_29 = {var_27: var_28}
    var_30 = {var_2: var_26, var_25: var_29}
    var_31 = {var_2: var_24, var_23: var_30}
    var_32 = [var_31]
    var_33 = {var_1: var_32}
    var_34 = module_1.one_of_from_json_schema(var_33, var_0)
    var_35 = var_34.one_of[var_12]
    var_36 = var_34.one_of[var_12]
    var_37 = var_36.items
    var_38 = []
    var_39 = {var_1: var_38}
    var_40 = module_1.one_of_from_json_schema(var_39, var_0)
    var_41 = var_40.one_of
    var_42 = len(var_41)
    assert var_42 == 0



# Parsed testcases at query #8
#--------------------------


import typesystem.json_schema as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)
    var_2 = False
    var_3 = module_0.from_json_schema(var_2)
    var_4 = 'type'
    var_5 = 'minLength'
    var_6 = 'string'
    var_7 = 5
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = module_0.from_json_schema(var_8)
    var_10 = 'minimum'
    var_11 = 'integer'
    var_12 = 10
    var_13 = {var_4: var_11, var_10: var_12}
    var_14 = module_0.from_json_schema(var_13)
    var_15 = 'enum'
    var_16 = 'a'
    var_17 = 'b'
    var_18 = 'c'
    var_19 = [var_16, var_17, var_18]
    var_20 = {var_15: var_19}
    var_21 = module_0.from_json_schema(var_20)
    var_22 = 'const'
    var_23 = 42
    var_24 = {var_22: var_23}
    var_25 = module_0.from_json_schema(var_24)
    var_26 = 'items'
    var_27 = 'minItems'
    var_28 = 'array'
    var_29 = {var_4: var_6}
    var_30 = {var_4: var_28, var_26: var_29, var_27: var_0}
    var_31 = module_0.from_json_schema(var_30)
    var_32 = var_31.items
    var_33 = 'properties'
    var_34 = 'required'
    var_35 = 'object'
    var_36 = 'name'
    var_37 = 'age'
    var_38 = {var_4: var_6}
    var_39 = {var_4: var_11}
    var_40 = {var_36: var_38, var_37: var_39}
    var_41 = [var_36]
    var_42 = {var_4: var_35, var_33: var_40, var_34: var_41}
    var_43 = module_0.from_json_schema(var_42)
    var_44 = var_43.properties[var_36]
    var_45 = 'allOf'
    var_46 = {var_4: var_6}
    var_47 = {var_5: var_12}
    var_48 = [var_46, var_47]
    var_49 = {var_45: var_48}
    var_50 = module_0.from_json_schema(var_49)
    var_51 = 'components'
    var_52 = 'schemas'
    var_53 = 'User'
    var_54 = 'id'
    var_55 = {var_4: var_11}
    var_56 = {var_54: var_55}
    var_57 = {var_4: var_35, var_33: var_56}
    var_58 = {var_53: var_57}
    var_59 = {var_52: var_58}
    var_60 = {var_51: var_59}
    var_61 = module_0.from_json_schema(var_60)
    var_62 = {}
    var_63 = module_0.from_json_schema(var_62)



# Parsed testcases at query #9
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = module_1.type_from_json_schema(var_3, var_0)
    var_5 = 'integer'
    var_6 = {var_1: var_5}
    var_7 = module_1.type_from_json_schema(var_6, var_0)
    var_8 = 'boolean'
    var_9 = {var_1: var_8}
    var_10 = module_1.type_from_json_schema(var_9, var_0)
    var_11 = 'number'
    var_12 = {var_1: var_11}
    var_13 = module_1.type_from_json_schema(var_12, var_0)
    var_14 = [var_2, var_5]
    var_15 = {var_1: var_14}
    var_16 = module_1.type_from_json_schema(var_15, var_0)
    var_17 = 'null'
    var_18 = [var_2, var_17]
    var_19 = {var_1: var_18}
    var_20 = module_1.type_from_json_schema(var_19, var_0)
    var_21 = []
    var_22 = {var_1: var_21}
    var_23 = module_1.type_from_json_schema(var_22, var_0)
    var_24 = isinstance(var_23, var_1)
    var_25 = 'minLength'
    var_26 = 5
    var_27 = {var_1: var_24, var_25: var_26}
    var_28 = module_1.type_from_json_schema(var_27, var_0)



# Parsed testcases at query #10
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'oneOf'
    var_2 = 'default'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 'integer'
    var_7 = {var_3: var_6}
    var_8 = [var_5, var_7]
    var_9 = 'some_default'
    var_10 = {var_1: var_8, var_2: var_9}
    var_11 = module_1.one_of_from_json_schema(var_10, var_0)
    var_12 = 0
    var_13 = var_11.one_of[var_12]
    var_14 = 1
    var_15 = var_11.one_of[var_14]
    var_16 = 'items'
    var_17 = 'array'
    var_18 = 'properties'
    var_19 = 'object'
    var_20 = 'name'
    var_21 = {var_3: var_4}
    var_22 = {var_20: var_21}
    var_23 = {var_3: var_19, var_18: var_22}
    var_24 = {var_3: var_17, var_16: var_23}
    var_25 = [var_24]
    var_26 = {var_1: var_25}
    var_27 = module_1.one_of_from_json_schema(var_26, var_0)
    var_28 = var_27.one_of[var_12]
    var_29 = var_27.one_of[var_12]
    var_30 = var_29.items
    var_31 = 'boolean'
    var_32 = {var_3: var_31}
    var_33 = [var_32]
    var_34 = {var_1: var_33}
    var_35 = module_1.one_of_from_json_schema(var_34, var_0)
    var_36 = 'minLength'
    var_37 = 5
    var_38 = {var_3: var_4, var_36: var_37}
    var_39 = 'maximum'
    var_40 = 'number'
    var_41 = 10
    var_42 = {var_3: var_40, var_39: var_41}
    var_43 = [var_38, var_42]
    var_44 = {var_1: var_43}
    var_45 = module_1.one_of_from_json_schema(var_44, var_0)
    var_46 = var_45.one_of
    var_47 = len(var_46)
    assert var_47 == 2



# Parsed testcases at query #11
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'exclusiveMinimum'
    var_4 = 'exclusiveMaximum'
    var_5 = 'multipleOf'
    var_6 = 'default'
    var_7 = 0
    var_8 = 10
    var_9 = 1
    var_10 = 9
    var_11 = 2
    var_12 = 5
    var_13 = {var_1: var_7, var_2: var_8, var_3: var_9, var_4: var_10, var_5: var_11, var_6: var_12}
    var_14 = 'number'
    var_15 = False
    var_16 = module_1.from_json_schema_type(var_13, var_14, var_15, var_0)
    var_17 = {var_1: var_9, var_2: var_12, var_5: var_9}
    var_18 = 'integer'
    var_19 = True
    var_20 = module_1.from_json_schema_type(var_17, var_18, var_19, var_0)
    var_21 = 'minLength'
    var_22 = 'maxLength'
    var_23 = 'pattern'
    var_24 = 'format'
    var_25 = '^abc'
    var_26 = 'email'
    var_27 = 'test'
    var_28 = {var_21: var_12, var_22: var_8, var_23: var_25, var_24: var_26, var_6: var_27}
    var_29 = 'string'
    var_30 = False
    var_31 = module_1.from_json_schema_type(var_28, var_29, var_30, var_0)
    var_32 = True
    var_33 = {var_6: var_32}
    var_34 = 'boolean'
    var_35 = False
    var_36 = module_1.from_json_schema_type(var_33, var_34, var_35, var_0)
    var_37 = 'items'
    var_38 = 'minItems'
    var_39 = 'maxItems'
    var_40 = 'uniqueItems'
    var_41 = 'additionalItems'
    var_42 = 'type'
    var_43 = {var_42: var_29}
    var_44 = True
    var_45 = False
    var_46 = {var_37: var_43, var_38: var_32, var_39: var_12, var_40: var_44, var_41: var_45}
    var_47 = 'array'
    var_48 = False
    var_49 = module_1.from_json_schema_type(var_46, var_47, var_48, var_0)
    var_50 = var_49.items
    var_51 = 'properties'
    var_52 = 'required'
    var_53 = 'additionalProperties'
    var_54 = 'minProperties'
    var_55 = 'name'
    var_56 = 'age'
    var_57 = {var_42: var_29}
    var_58 = {var_42: var_18}
    var_59 = {var_55: var_57, var_56: var_58}
    var_60 = [var_55]
    var_61 = True
    var_62 = {var_51: var_59, var_52: var_60, var_53: var_61, var_54: var_61}
    var_63 = 'object'
    var_64 = False
    var_65 = module_1.from_json_schema_type(var_62, var_63, var_64, var_0)
    var_66 = var_65.properties[var_55]
    var_67 = var_65.properties[var_56]
    var_68 = {var_42: var_29}
    var_69 = True
    var_70 = module_1.from_json_schema_type(var_68, var_29, var_69, var_0)



# Parsed testcases at query #12
#--------------------------


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 5
    var_2 = 10
    var_3 = 'email'
    var_4 = module_0.String(max_length=var_2, min_length=var_1, format=var_3)
    var_5 = module_1.to_json_schema(var_4)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = 100
    var_2 = 1
    var_3 = module_0.Integer(minimum=var_0, maximum=var_1, multiple_of=var_2)
    var_4 = module_1.to_json_schema(var_3)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = 0.5
    var_2 = module_0.Float(exclusive_minimum=var_1)
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
    var_0 = False
    var_1 = module_0.String()
    var_2 = 1
    var_3 = 5
    var_4 = True
    var_5 = module_0.Array(var_1, var_0, var_2, var_3, unique_items=var_4)
    var_6 = module_1.to_json_schema(var_5)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.String()
    var_2 = 'name'
    var_3 = {var_2: var_1}
    var_4 = [var_2]
    var_5 = 1
    var_6 = True
    var_7 = module_0.Object(properties=var_3, additional_properties=var_6, min_properties=var_5, required=var_4)
    var_8 = module_1.to_json_schema(var_7)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'A'
    var_1 = (var_0, var_0)
    var_2 = 'B'
    var_3 = (var_2, var_2)
    var_4 = [var_1, var_3]
    var_5 = module_0.Choice(choices=var_4)
    var_6 = module_1.to_json_schema(var_5)

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
    var_5 = 'anyOf'
    var_6 = var_4[var_5]
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = var_4[var_5]
    var_9 = 'type'
    var_10 = {item[var_9] for item in var_8}

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
    var_1 = module_1.Not(var_0)
    var_2 = module_2.to_json_schema(var_1)

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
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'MySchema'
    var_2 = {}
    var_3 = module_1.Reference(var_1, var_2)
    var_4 = module_2.to_json_schema(var_3)

def test_case_0():
    var_0 = 'abc'

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'MyType'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_1.Reference(var_0, var_2)
    var_4 = module_2.to_json_schema(var_2)



# Parsed testcases at query #13
#--------------------------


import typesystem.fields as module_0
import typesystem.json_schema as module_1
import typesystem.composites as module_2

def test_case_0():
    var_0 = False
    var_1 = module_0.String()
    var_2 = module_0.Integer()
    var_3 = [var_1, var_2]
    var_4 = module_0.Union(var_3)
    var_5 = 'anyOf'
    var_6 = 'type'
    var_7 = 'string'
    var_8 = {var_6: var_7}
    var_9 = 'integer'
    var_10 = {var_6: var_9}
    var_11 = [var_8, var_10]
    var_12 = {var_5: var_11}
    var_13 = module_1.to_json_schema(var_4)
    var_14 = module_0.String()
    var_15 = 'foo'
    var_16 = module_0.Const(var_15)
    var_17 = [var_14, var_16]
    var_18 = module_2.AllOf(var_17)
    var_19 = 'allOf'
    var_20 = {var_6: var_7}
    var_21 = 'const'
    var_22 = {var_21: var_15}
    var_23 = [var_20, var_22]
    var_24 = {var_19: var_23}
    var_25 = module_1.to_json_schema(var_18)
    var_26 = module_0.String()
    var_27 = module_2.Not(var_26)
    var_28 = 'not'
    var_29 = {var_6: var_7}
    var_30 = {var_28: var_29}
    var_31 = module_1.to_json_schema(var_27)
    var_32 = module_0.Integer()
    var_33 = module_0.Boolean()
    var_34 = module_0.String()
    var_35 = module_2.IfThenElse(var_32, var_33, var_34)
    var_36 = 'if'
    var_37 = 'then'
    var_38 = 'else'
    var_39 = {var_6: var_9}
    var_40 = 'boolean'
    var_41 = {var_6: var_40}
    var_42 = {var_6: var_7}
    var_43 = {var_36: var_39, var_37: var_41, var_38: var_42}
    var_44 = module_1.to_json_schema(var_35)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = False
    var_1 = module_0.String()
    var_2 = 'MyString'
    var_3 = {var_2: var_1}
    var_4 = module_1.Reference(var_2, var_3)
    var_5 = None
    var_6 = module_2.to_json_schema(var_4, var_5)

def test_case_0():
    var_0 = '\\w'
    var_1 = False

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.String()
    var_2 = module_0.Integer()
    var_3 = [var_1, var_2]
    var_4 = module_0.Array(var_3)
    var_5 = module_1.to_json_schema(var_4)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = '.*'
    var_1 = False
    var_2 = module_0.String()
    var_3 = {var_0: var_2}
    var_4 = module_0.Boolean()
    var_5 = module_0.Object(pattern_properties=var_3, additional_properties=var_4)
    var_6 = module_1.to_json_schema(var_5)



# Parsed testcases at query #14
#--------------------------


import typesystem.fields as module_0
import typesystem.json_schema as module_1
import re as module_2

def test_case_0():
    var_0 = True
    var_1 = 5
    var_2 = 10
    var_3 = 'email'
    var_4 = module_0.String(max_length=var_2, min_length=var_1, format=var_3)
    var_5 = module_1.to_json_schema(var_4)
    var_6 = '^[a-z]+$'
    var_7 = module_2.compile(var_6)
    var_8 = module_0.String()
    var_9 = module_1.to_json_schema(var_8)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = 100
    var_2 = module_0.Integer(minimum=var_0, maximum=var_1)
    var_3 = module_1.to_json_schema(var_2)
    var_4 = True
    var_5 = 5.5
    var_6 = module_0.Float(exclusive_minimum=var_5)
    var_7 = module_1.to_json_schema(var_6)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean()
    var_2 = module_1.to_json_schema(var_1)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.String()
    var_2 = 1
    var_3 = True
    var_4 = module_0.Array(var_1, min_items=var_2, unique_items=var_3)
    var_5 = module_1.to_json_schema(var_4)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.String()
    var_2 = 'name'
    var_3 = {var_2: var_1}
    var_4 = [var_2]
    var_5 = module_0.Object(properties=var_3, required=var_4)
    var_6 = module_1.to_json_schema(var_5)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'red'
    var_1 = (var_0, var_0)
    var_2 = 'blue'
    var_3 = (var_2, var_2)
    var_4 = [var_1, var_3]
    var_5 = module_0.Choice(choices=var_4)
    var_6 = module_1.to_json_schema(var_5)
    var_7 = 'fixed_value'
    var_8 = module_0.Const(var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = module_0.String()
    var_11 = module_0.Integer()
    var_12 = [var_10, var_11]
    var_13 = module_0.Union(var_12)
    var_14 = module_1.to_json_schema(var_13)
    var_15 = 'anyOf'
    var_16 = var_14[var_15]
    var_17 = len(var_16)
    assert var_17 == 2

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = False
    var_1 = module_0.String()
    var_2 = 'User'
    var_3 = {}
    var_4 = module_1.Reference(var_2, var_3)
    var_5 = {}
    var_6 = module_2.to_json_schema(var_4, var_5)

def test_case_0():
    var_0 = 'abc'



# Parsed testcases at query #15
#--------------------------


import typesystem.json_schema as module_0
import typesystem.fields as module_1
import typesystem.composites as module_2

def test_case_0():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)
    var_2 = module_1.Any()
    var_3 = False
    var_4 = module_0.from_json_schema(var_3)
    var_5 = module_2.NeverMatch()
    var_6 = 'type'
    var_7 = 'string'
    var_8 = {var_6: var_7}
    var_9 = module_0.from_json_schema(var_8)
    var_10 = 'integer'
    var_11 = {var_6: var_10}
    var_12 = module_0.from_json_schema(var_11)
    var_13 = 'enum'
    var_14 = 'a'
    var_15 = 'b'
    var_16 = [var_14, var_15]
    var_17 = {var_6: var_7, var_13: var_16}
    var_18 = module_0.from_json_schema(var_17)
    var_19 = 'const'
    var_20 = 123
    var_21 = {var_19: var_20}
    var_22 = module_0.from_json_schema(var_21)
    var_23 = 'minLength'
    var_24 = 'maxLength'
    var_25 = 5
    var_26 = 10
    var_27 = {var_6: var_7, var_23: var_25, var_24: var_26}
    var_28 = module_0.from_json_schema(var_27)
    var_29 = var_28.constraints
    var_30 = [c for c in var_29]
    var_31 = 'foo'
    var_32 = 'bar'
    var_33 = {var_31: var_32}
    var_34 = module_0.from_json_schema(var_33)
    var_35 = module_1.Any()
    var_36 = 'components'
    var_37 = '$ref'
    var_38 = 'schemas'
    var_39 = 'User'
    var_40 = {var_6: var_7}
    var_41 = {var_39: var_40}
    var_42 = {var_38: var_41}
    var_43 = '#/components/schemas/User'
    var_44 = {var_36: var_42, var_37: var_43}
    var_45 = module_0.from_json_schema(var_44)
    var_46 = 'items'
    var_47 = 'array'
    var_48 = {var_6: var_7}
    var_49 = {var_6: var_47, var_46: var_48}
    var_50 = module_0.from_json_schema(var_49)
    var_51 = var_50.items
    var_52 = 'minimum'
    var_53 = 'maximum'
    var_54 = 'number'
    var_55 = 100
    var_56 = {var_6: var_54, var_52: var_3, var_53: var_55}
    var_57 = module_0.from_json_schema(var_56)



# Parsed testcases at query #16
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'exclusiveMinimum'
    var_4 = 'exclusiveMaximum'
    var_5 = 'multipleOf'
    var_6 = 'default'
    var_7 = 0
    var_8 = 10
    var_9 = 2
    var_10 = 8
    var_11 = 0.5
    var_12 = 5.0
    var_13 = {var_1: var_7, var_2: var_8, var_3: var_9, var_4: var_10, var_5: var_11, var_6: var_12}
    var_14 = 'number'
    var_15 = False
    var_16 = module_1.from_json_schema_type(var_13, var_14, var_15, var_0)
    var_17 = 1
    var_18 = {var_1: var_17, var_2: var_12, var_5: var_9}
    var_19 = 'integer'
    var_20 = True
    var_21 = module_1.from_json_schema_type(var_18, var_19, var_20, var_0)
    var_22 = 'minLength'
    var_23 = 'maxLength'
    var_24 = 'pattern'
    var_25 = 'format'
    var_26 = '^abc'
    var_27 = 'email'
    var_28 = 'hello'
    var_29 = {var_22: var_12, var_23: var_8, var_24: var_26, var_25: var_27, var_6: var_28}
    var_30 = 'string'
    var_31 = False
    var_32 = module_1.from_json_schema_type(var_29, var_30, var_31, var_0)
    var_33 = True
    var_34 = {var_6: var_33}
    var_35 = 'boolean'
    var_36 = False
    var_37 = module_1.from_json_schema_type(var_34, var_35, var_36, var_0)
    var_38 = 'items'
    var_39 = 'minItems'
    var_40 = 'maxItems'
    var_41 = 'uniqueItems'
    var_42 = 'additionalItems'
    var_43 = 'type'
    var_44 = {var_43: var_30}
    var_45 = True
    var_46 = False
    var_47 = {var_38: var_44, var_39: var_33, var_40: var_12, var_41: var_45, var_42: var_46}
    var_48 = 'array'
    var_49 = False
    var_50 = module_1.from_json_schema_type(var_47, var_48, var_49, var_0)
    var_51 = 'properties'
    var_52 = 'required'
    var_53 = 'minProperties'
    var_54 = 'additionalProperties'
    var_55 = 'name'
    var_56 = 'age'
    var_57 = {var_43: var_30}
    var_58 = {var_43: var_19}
    var_59 = {var_55: var_57, var_56: var_58}
    var_60 = [var_55]
    var_61 = {var_43: var_35}
    var_62 = {var_51: var_59, var_52: var_60, var_53: var_45, var_54: var_61}
    var_63 = 'object'
    var_64 = False
    var_65 = module_1.from_json_schema_type(var_62, var_63, var_64, var_0)
    var_66 = var_65.properties[var_55]
    var_67 = var_65.properties[var_56]
    var_68 = var_65.additional_properties
    var_69 = {}
    var_70 = 'invalid_type'
    var_71 = False
    var_72 = module_1.from_json_schema_type(var_69, var_70, var_71, var_0)



# Parsed testcases at query #17
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'exclusiveMinimum'
    var_4 = 'exclusiveMaximum'
    var_5 = 'multipleOf'
    var_6 = 'default'
    var_7 = 0
    var_8 = 10
    var_9 = 1
    var_10 = 9
    var_11 = 2
    var_12 = 5
    var_13 = {var_1: var_7, var_2: var_8, var_3: var_9, var_4: var_10, var_5: var_11, var_6: var_12}
    var_14 = 'number'
    var_15 = False
    var_16 = module_1.from_json_schema_type(var_13, var_14, var_15, var_0)
    var_17 = {var_1: var_15, var_2: var_8, var_5: var_11, var_6: var_12}
    var_18 = 'integer'
    var_19 = True
    var_20 = module_1.from_json_schema_type(var_17, var_18, var_19, var_0)
    var_21 = 'minLength'
    var_22 = 'maxLength'
    var_23 = 'format'
    var_24 = 'pattern'
    var_25 = 'email'
    var_26 = '^[a-z]+$'
    var_27 = 'test'
    var_28 = {var_21: var_12, var_22: var_8, var_23: var_25, var_24: var_26, var_6: var_27}
    var_29 = 'string'
    var_30 = False
    var_31 = module_1.from_json_schema_type(var_28, var_29, var_30, var_0)
    var_32 = True
    var_33 = {var_6: var_32}
    var_34 = 'boolean'
    var_35 = False
    var_36 = module_1.from_json_schema_type(var_33, var_34, var_35, var_0)
    var_37 = 'items'
    var_38 = 'minItems'
    var_39 = 'maxItems'
    var_40 = 'uniqueItems'
    var_41 = 'additionalItems'
    var_42 = 'type'
    var_43 = {var_42: var_29}
    var_44 = True
    var_45 = False
    var_46 = {var_37: var_43, var_38: var_32, var_39: var_12, var_40: var_44, var_41: var_45}
    var_47 = 'array'
    var_48 = False
    var_49 = module_1.from_json_schema_type(var_46, var_47, var_48, var_0)
    var_50 = 'properties'
    var_51 = 'required'
    var_52 = 'minProperties'
    var_53 = 'additionalProperties'
    var_54 = 'name'
    var_55 = 'age'
    var_56 = {var_42: var_29}
    var_57 = {var_42: var_18}
    var_58 = {var_54: var_56, var_55: var_57}
    var_59 = [var_54]
    var_60 = {var_42: var_34}
    var_61 = {var_50: var_58, var_51: var_59, var_52: var_44, var_53: var_60}
    var_62 = 'object'
    var_63 = False
    var_64 = module_1.from_json_schema_type(var_61, var_62, var_63, var_0)
    var_65 = var_64.properties[var_54]
    var_66 = var_64.properties[var_55]
    var_67 = var_64.additional_properties



