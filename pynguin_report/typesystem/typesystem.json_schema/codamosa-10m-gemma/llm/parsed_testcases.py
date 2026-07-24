####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
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
    var_8 = 'boolean'
    var_9 = {var_5: var_8}
    var_10 = 'integer'
    var_11 = {var_5: var_10}
    var_12 = 'some_default'
    var_13 = {var_1: var_7, var_2: var_9, var_3: var_11, var_4: var_12}
    var_14 = module_1.if_then_else_from_json_schema(var_13, var_0)
    var_15 = module_2.String()
    var_16 = module_2.Boolean()
    var_17 = module_2.Integer()
    var_18 = {var_5: var_6}
    var_19 = {var_5: var_8}
    var_20 = {var_1: var_18, var_2: var_19}
    var_21 = module_1.if_then_else_from_json_schema(var_20, var_0)
    var_22 = module_2.String()
    var_23 = module_2.Boolean()
    var_24 = {var_5: var_6}
    var_25 = {var_5: var_10}
    var_26 = {var_1: var_24, var_3: var_25}
    var_27 = module_1.if_then_else_from_json_schema(var_26, var_0)
    var_28 = module_2.String()
    var_29 = module_2.Integer()
    var_30 = {var_5: var_6}
    var_31 = {var_1: var_30}
    var_32 = module_1.if_then_else_from_json_schema(var_31, var_0)
    var_33 = module_2.String()



# Parsed testcases at query #2
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'enum'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = module_1.enum_from_json_schema(var_6, var_0)
    var_8 = 'default'
    var_9 = 1
    var_10 = 2
    var_11 = [var_9, var_10]
    var_12 = {var_1: var_11, var_8: var_9}
    var_13 = module_1.enum_from_json_schema(var_12, var_0)
    var_14 = 'apple'
    var_15 = True
    var_16 = [var_9, var_14, var_15]
    var_17 = {var_1: var_16}
    var_18 = module_1.enum_from_json_schema(var_17, var_0)
    var_19 = {}
    var_20 = module_1.enum_from_json_schema(var_19, var_0)



# Parsed testcases at query #3
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1
import typesystem.fields as module_2

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'type'
    var_2 = 'minLength'
    var_3 = 'maxLength'
    var_4 = 'pattern'
    var_5 = 'format'
    var_6 = 'string'
    var_7 = 5
    var_8 = 10
    var_9 = '^a'
    var_10 = 'email'
    var_11 = {var_1: var_6, var_2: var_7, var_3: var_8, var_4: var_9, var_5: var_10}
    var_12 = False
    var_13 = module_1.from_json_schema_type(var_11, var_6, var_12, var_0)
    var_14 = 'minimum'
    var_15 = 'maximum'
    var_16 = 'multipleOf'
    var_17 = 'integer'
    var_18 = 100
    var_19 = 2
    var_20 = {var_1: var_17, var_14: var_12, var_15: var_18, var_16: var_19}
    var_21 = True
    var_22 = module_1.from_json_schema_type(var_20, var_17, var_21, var_0)
    var_23 = 'exclusiveMinimum'
    var_24 = 'number'
    var_25 = 1.5
    var_26 = {var_1: var_24, var_23: var_25}
    var_27 = module_1.from_json_schema_type(var_26, var_24, var_12, var_0)
    var_28 = 'default'
    var_29 = 'boolean'
    var_30 = {var_1: var_29, var_28: var_21}
    var_31 = module_1.from_json_schema_type(var_30, var_29, var_12, var_0)
    var_32 = 'items'
    var_33 = 'minItems'
    var_34 = 'uniqueItems'
    var_35 = 'additionalItems'
    var_36 = 'array'
    var_37 = {var_1: var_6}
    var_38 = {var_1: var_36, var_32: var_37, var_33: var_21, var_34: var_21, var_35: var_12}
    var_39 = module_1.from_json_schema_type(var_38, var_36, var_12, var_0)
    var_40 = var_39.items
    var_41 = 'properties'
    var_42 = 'required'
    var_43 = 'additionalProperties'
    var_44 = 'object'
    var_45 = 'name'
    var_46 = 'age'
    var_47 = {var_1: var_6}
    var_48 = {var_1: var_17}
    var_49 = {var_45: var_47, var_46: var_48}
    var_50 = [var_45]
    var_51 = {var_1: var_44, var_41: var_49, var_42: var_50, var_43: var_21}
    var_52 = module_1.from_json_schema_type(var_51, var_44, var_12, var_0)
    var_53 = var_52.properties[var_45]
    var_54 = var_52.properties[var_46]
    var_55 = [var_6, var_24]
    var_56 = {var_1: var_55}
    var_57 = module_1.from_json_schema_type(var_56, var_6, var_12, var_0)
    var_58 = module_1.from_json_schema_type(var_56, var_24, var_12, var_0)
    var_59 = [var_57, var_58]
    var_60 = module_2.Union(var_59)
    var_61 = {var_1: var_6}
    var_62 = module_1.from_json_schema_type(var_61, var_6, var_21, var_0)



# Parsed testcases at query #4
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = '#/definitions/MyType'
    var_3 = {var_1: var_2}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)
    var_5 = '#/components/schemas/User'
    var_6 = {var_1: var_5}
    var_7 = module_1.ref_from_json_schema(var_6, var_0)
    var_8 = 'external_file.json#/Type'
    var_9 = {var_1: var_8}
    var_10 = module_1.ref_from_json_schema(var_9, var_0)
    var_11 = {}
    var_12 = module_1.ref_from_json_schema(var_11, var_0)



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'if'
    var_1 = 'then'
    var_2 = 'else'
    var_3 = 'default'
    var_4 = 'type'
    var_5 = 'string'
    var_6 = {var_4: var_5}
    var_7 = 'integer'
    var_8 = {var_4: var_7}
    var_9 = 'boolean'
    var_10 = {var_4: var_9}
    var_11 = 'some_default'
    var_12 = {var_0: var_6, var_1: var_8, var_2: var_10, var_3: var_11}
    var_13 = 'number'
    var_14 = {var_4: var_13}
    var_15 = {var_4: var_5}
    var_16 = {var_0: var_14, var_1: var_15}
    var_17 = {var_4: var_9}
    var_18 = 'object'
    var_19 = {var_4: var_18}
    var_20 = {var_0: var_17, var_2: var_19}
    var_21 = 'array'
    var_22 = {var_4: var_21}
    var_23 = {var_0: var_22}



# Parsed testcases at query #6
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'allOf'
    var_2 = 'default'
    var_3 = 'type'
    var_4 = 'minimum'
    var_5 = 'integer'
    var_6 = 10
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'minLength'
    var_9 = 'string'
    var_10 = 5
    var_11 = {var_3: var_9, var_8: var_10}
    var_12 = [var_7, var_11]
    var_13 = 123
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
    var_25 = True
    var_26 = {var_1: var_24, var_2: var_25}
    var_27 = module_1.all_of_from_json_schema(var_26, var_0)
    var_28 = var_27.all_of
    var_29 = len(var_28)
    assert var_29 == 1
    var_30 = var_27.all_of[var_18]
    var_31 = {var_3: var_5}
    var_32 = [var_31]
    var_33 = {var_1: var_32}
    var_34 = module_1.all_of_from_json_schema(var_33, var_0)
    var_35 = 'properties'
    var_36 = 'object'
    var_37 = 'name'
    var_38 = {var_3: var_9}
    var_39 = {var_37: var_38}
    var_40 = {var_3: var_36, var_35: var_39}
    var_41 = {var_3: var_22}
    var_42 = [var_40, var_41]
    var_43 = {var_1: var_42}
    var_44 = module_1.all_of_from_json_schema(var_43, var_0)
    var_45 = var_44.all_of[var_18]
    var_46 = var_44.all_of[var_25]



# Parsed testcases at query #7
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
    var_12 = var_11.one_of
    var_13 = len(var_12)
    assert var_13 == 2
    var_14 = 0
    var_15 = var_11.one_of[var_14]
    var_16 = 1
    var_17 = var_11.one_of[var_16]
    var_18 = 'boolean'
    var_19 = {var_3: var_18}
    var_20 = [var_19]
    var_21 = {var_1: var_20}
    var_22 = module_1.one_of_from_json_schema(var_21, var_0)
    var_23 = var_22.one_of[var_14]
    var_24 = 'anyOf'
    var_25 = {var_3: var_4}
    var_26 = {var_3: var_6}
    var_27 = [var_25, var_26]
    var_28 = {var_24: var_27}
    var_29 = 'allOf'
    var_30 = {var_3: var_4}
    var_31 = 'minLength'
    var_32 = 5
    var_33 = {var_31: var_32}
    var_34 = [var_30, var_33]
    var_35 = {var_29: var_34}
    var_36 = [var_28, var_35]
    var_37 = {var_1: var_36}
    var_38 = module_1.one_of_from_json_schema(var_37, var_0)
    var_39 = var_38.one_of[var_14]
    var_40 = var_38.one_of[var_16]
    var_41 = {var_3: var_4}
    var_42 = [var_41]
    var_43 = 123
    var_44 = {var_1: var_42, var_2: var_43}
    var_45 = module_1.one_of_from_json_schema(var_44, var_0)



# Parsed testcases at query #8
#--------------------------


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.String()
    var_2 = module_0.Integer()
    var_3 = 'name'
    var_4 = 'age'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = [var_3]
    var_7 = module_0.Object(properties=var_5, additional_properties=var_0, required=var_6)
    var_8 = module_1.to_json_schema(var_7)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'User'
    var_1 = False
    var_2 = module_0.String()
    var_3 = {var_0: var_2}
    var_4 = module_1.Reference(var_0, var_3)
    var_5 = module_2.to_json_schema(var_3)

import re as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = 'abc'
    var_1 = False
    var_2 = '^[a-z]+$'
    var_3 = module_0.compile(var_2)
    var_4 = module_1.String()
    var_5 = 'pattern'
    var_6 = to_json_schema(var_4)[var_5]
    assert var_6 == '^[a-z]+$'

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
    var_0 = '.*suffix'
    var_1 = False
    var_2 = module_0.String()
    var_3 = {var_0: var_2}
    var_4 = module_0.Object(pattern_properties=var_3)
    var_5 = module_1.to_json_schema(var_4)



# Parsed testcases at query #9
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'allOf'
    var_2 = 'default'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = [var_5]
    var_7 = 'default_val'
    var_8 = {var_1: var_6, var_2: var_7}
    var_9 = module_1.all_of_from_json_schema(var_8, var_0)
    var_10 = var_9.all_of
    var_11 = len(var_10)
    assert var_11 == 1
    var_12 = 0
    var_13 = var_9.all_of[var_12]
    var_14 = 'minLength'
    var_15 = 5
    var_16 = {var_3: var_4, var_14: var_15}
    var_17 = 'minimum'
    var_18 = 'integer'
    var_19 = 10
    var_20 = {var_3: var_18, var_17: var_19}
    var_21 = 'boolean'
    var_22 = {var_3: var_21}
    var_23 = [var_16, var_20, var_22]
    var_24 = {var_1: var_23}
    var_25 = module_1.all_of_from_json_schema(var_24, var_0)
    var_26 = var_25.all_of
    var_27 = len(var_26)
    assert var_27 == 3
    var_28 = var_25.all_of[var_12]
    var_29 = 1
    var_30 = var_25.all_of[var_29]
    var_31 = 2
    var_32 = var_25.all_of[var_31]
    var_33 = 'properties'
    var_34 = 'object'
    var_35 = 'name'
    var_36 = {var_3: var_4}
    var_37 = {var_35: var_36}
    var_38 = {var_3: var_34, var_33: var_37}
    var_39 = 'items'
    var_40 = 'array'
    var_41 = {var_3: var_18}
    var_42 = {var_3: var_40, var_39: var_41}
    var_43 = [var_38, var_42]
    var_44 = None
    var_45 = {var_1: var_43, var_2: var_44}
    var_46 = module_1.all_of_from_json_schema(var_45, var_0)
    var_47 = var_46.all_of[var_12]
    var_48 = var_46.all_of[var_29]



# Parsed testcases at query #10
#--------------------------


import typesystem.json_schema as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)
    var_2 = False
    var_3 = module_0.from_json_schema(var_2)
    var_4 = 'type'
    var_5 = 'string'
    var_6 = {var_4: var_5}
    var_7 = module_0.from_json_schema(var_6)
    var_8 = 'integer'
    var_9 = {var_4: var_8}
    var_10 = module_0.from_json_schema(var_9)
    var_11 = 'enum'
    var_12 = 'a'
    var_13 = 'b'
    var_14 = 'c'
    var_15 = [var_12, var_13, var_14]
    var_16 = {var_11: var_15}
    var_17 = module_0.from_json_schema(var_16)
    var_18 = 'const'
    var_19 = 42
    var_20 = {var_18: var_19}
    var_21 = module_0.from_json_schema(var_20)
    var_22 = 'val1'
    var_23 = [var_22]
    var_24 = {var_4: var_5, var_11: var_23}
    var_25 = module_0.from_json_schema(var_24)
    var_26 = {}
    var_27 = module_0.from_json_schema(var_26)
    var_28 = '$ref'
    var_29 = '#/definitions/MyType'
    var_30 = {var_28: var_29}
    var_31 = module_0.from_json_schema(var_30)
    var_32 = 'components'
    var_33 = 'schemas'
    var_34 = 'User'
    var_35 = {var_4: var_5}
    var_36 = {var_34: var_35}
    var_37 = {var_33: var_36}
    var_38 = {var_32: var_37}
    var_39 = module_0.from_json_schema(var_38)
    var_40 = 'minimum'
    var_41 = 'number'
    var_42 = 10
    var_43 = {var_4: var_41, var_40: var_42}
    var_44 = module_0.from_json_schema(var_43)



# Parsed testcases at query #11
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'type'
    var_2 = 'minimum'
    var_3 = 'maximum'
    var_4 = 'exclusiveMinimum'
    var_5 = 'exclusiveMaximum'
    var_6 = 'multipleOf'
    var_7 = 'default'
    var_8 = 'number'
    var_9 = 0
    var_10 = 10
    var_11 = 1
    var_12 = 9
    var_13 = 2
    var_14 = 5.5
    var_15 = {var_1: var_8, var_2: var_9, var_3: var_10, var_4: var_11, var_5: var_12, var_6: var_13, var_7: var_14}
    var_16 = False
    var_17 = module_1.from_json_schema_type(var_15, var_8, var_16, var_0)
    var_18 = 'integer'
    var_19 = 5
    var_20 = 3
    var_21 = {var_1: var_18, var_2: var_11, var_3: var_19, var_7: var_20}
    var_22 = True
    var_23 = module_1.from_json_schema_type(var_21, var_18, var_22, var_0)
    var_24 = 'minLength'
    var_25 = 'maxLength'
    var_26 = 'pattern'
    var_27 = 'format'
    var_28 = 'string'
    var_29 = '^abc'
    var_30 = 'email'
    var_31 = {var_1: var_28, var_24: var_19, var_25: var_10, var_26: var_29, var_27: var_30}
    var_32 = False
    var_33 = module_1.from_json_schema_type(var_31, var_28, var_32, var_0)
    var_34 = 'boolean'
    var_35 = True
    var_36 = {var_1: var_34, var_7: var_35}
    var_37 = False
    var_38 = module_1.from_json_schema_type(var_36, var_34, var_37, var_0)
    var_39 = 'items'
    var_40 = 'minItems'
    var_41 = 'maxItems'
    var_42 = 'uniqueItems'
    var_43 = 'additionalItems'
    var_44 = 'array'
    var_45 = {var_1: var_28}
    var_46 = True
    var_47 = False
    var_48 = {var_1: var_44, var_39: var_45, var_40: var_35, var_41: var_19, var_42: var_46, var_43: var_47}
    var_49 = False
    var_50 = module_1.from_json_schema_type(var_48, var_44, var_49, var_0)
    var_51 = var_50.items
    var_52 = 'properties'
    var_53 = 'required'
    var_54 = 'additionalProperties'
    var_55 = 'minProperties'
    var_56 = 'object'
    var_57 = 'name'
    var_58 = 'age'
    var_59 = {var_1: var_28}
    var_60 = {var_1: var_18}
    var_61 = {var_57: var_59, var_58: var_60}
    var_62 = [var_57]
    var_63 = True
    var_64 = {var_1: var_56, var_52: var_61, var_53: var_62, var_54: var_63, var_55: var_63}
    var_65 = False
    var_66 = module_1.from_json_schema_type(var_64, var_56, var_65, var_0)
    var_67 = var_66.properties[var_57]
    var_68 = var_66.properties[var_58]
    var_69 = {var_1: var_28, var_24: var_13}
    var_70 = True
    var_71 = module_1.from_json_schema_type(var_69, var_28, var_70, var_0)



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
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
    var_12 = 'some_default'
    var_13 = {var_1: var_7, var_2: var_9, var_3: var_11, var_4: var_12}
    var_14 = module_1.if_then_else_from_json_schema(var_13, var_0)
    var_15 = module_2.String()
    var_16 = module_2.Integer()
    var_17 = module_2.Boolean()
    var_18 = {var_5: var_10}
    var_19 = {var_5: var_6}
    var_20 = {var_1: var_18, var_2: var_19}
    var_21 = module_1.if_then_else_from_json_schema(var_20, var_0)
    var_22 = module_2.Boolean()
    var_23 = module_2.String()
    var_24 = 'number'
    var_25 = {var_5: var_24}
    var_26 = 'items'
    var_27 = 'array'
    var_28 = {var_5: var_6}
    var_29 = {var_5: var_27, var_26: var_28}
    var_30 = {var_1: var_25, var_3: var_29}
    var_31 = module_1.if_then_else_from_json_schema(var_30, var_0)
    var_32 = module_2.Float()
    var_33 = var_31.else_clause
    var_34 = module_2.String()



# Parsed testcases at query #2
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'enum'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 'c'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = module_1.enum_from_json_schema(var_6, var_0)
    var_8 = 'default'
    var_9 = 1
    var_10 = 2
    var_11 = [var_9, var_10]
    var_12 = {var_1: var_11, var_8: var_9}
    var_13 = module_1.enum_from_json_schema(var_12, var_0)
    var_14 = 10
    var_15 = 20
    var_16 = [var_14, var_15]
    var_17 = {var_1: var_16}
    var_18 = module_1.enum_from_json_schema(var_17, var_0)
    var_19 = 'id'
    var_20 = {var_19: var_9}
    var_21 = {var_19: var_10}
    var_22 = [var_20, var_21]
    var_23 = {var_1: var_22}
    var_24 = module_1.enum_from_json_schema(var_23, var_0)
    var_25 = {}
    var_26 = module_1.enum_from_json_schema(var_25, var_0)



# Parsed testcases at query #3
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1
import typesystem.fields as module_2

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'type'
    var_2 = 'minLength'
    var_3 = 'maxLength'
    var_4 = 'pattern'
    var_5 = 'default'
    var_6 = 'string'
    var_7 = 5
    var_8 = 10
    var_9 = '^abc'
    var_10 = 'def'
    var_11 = {var_1: var_6, var_2: var_7, var_3: var_8, var_4: var_9, var_5: var_10}
    var_12 = False
    var_13 = module_1.from_json_schema_type(var_11, var_6, var_12, var_0)
    var_14 = 'null'
    var_15 = True
    var_16 = {var_1: var_6, var_14: var_15}
    var_17 = module_1.from_json_schema_type(var_16, var_6, var_15, var_0)
    var_18 = 'minimum'
    var_19 = 'maximum'
    var_20 = 'multipleOf'
    var_21 = 'integer'
    var_22 = 2
    var_23 = {var_1: var_21, var_18: var_12, var_19: var_8, var_20: var_22}
    var_24 = module_1.from_json_schema_type(var_23, var_21, var_12, var_0)
    var_25 = 'exclusiveMinimum'
    var_26 = 'number'
    var_27 = 0.5
    var_28 = 0.1
    var_29 = {var_1: var_26, var_18: var_27, var_25: var_28}
    var_30 = module_1.from_json_schema_type(var_29, var_26, var_12, var_0)
    var_31 = 'boolean'
    var_32 = {var_1: var_31, var_5: var_15}
    var_33 = module_1.from_json_schema_type(var_32, var_31, var_12, var_0)
    var_34 = 'items'
    var_35 = 'minItems'
    var_36 = 'maxItems'
    var_37 = 'uniqueItems'
    var_38 = 'additionalItems'
    var_39 = 'array'
    var_40 = {var_1: var_6}
    var_41 = {var_1: var_39, var_34: var_40, var_35: var_15, var_36: var_7, var_37: var_15, var_38: var_12}
    var_42 = module_1.from_json_schema_type(var_41, var_39, var_12, var_0)
    var_43 = var_42.items
    var_44 = 'properties'
    var_45 = 'required'
    var_46 = 'additionalProperties'
    var_47 = 'minProperties'
    var_48 = 'object'
    var_49 = 'name'
    var_50 = {var_1: var_6}
    var_51 = {var_49: var_50}
    var_52 = [var_49]
    var_53 = {var_1: var_21}
    var_54 = {var_1: var_48, var_44: var_51, var_45: var_52, var_46: var_53, var_47: var_15}
    var_55 = module_1.from_json_schema_type(var_54, var_48, var_12, var_0)
    var_56 = var_55.properties[var_49]
    var_57 = var_55.additional_properties
    var_58 = [var_6, var_26]
    var_59 = {var_1: var_58}
    var_60 = {var_1: var_6}
    var_61 = module_1.from_json_schema_type(var_60, var_6, var_12, var_0)
    var_62 = {var_1: var_26}
    var_63 = module_1.from_json_schema_type(var_62, var_26, var_12, var_0)
    var_64 = [var_61, var_63]
    var_65 = module_2.Union(var_64)
    var_66 = {var_1: var_14}
    var_67 = {var_1: var_14}
    var_68 = module_1.from_json_schema_type(var_67, var_14, var_15, var_0)
    var_69 = {var_1: var_14}
    var_70 = module_1.from_json_schema_type(var_69, var_14, var_15, var_0)
    var_71 = 'type'
    var_72 = 'null'
    var_73 = {var_71: var_72}
    var_74 = True
    var_75 = module_1.from_json_schema_type(var_73, var_72, var_74, var_0)



# Parsed testcases at query #4
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
    var_22 = 'properties'
    var_23 = 'object'
    var_24 = 'name'
    var_25 = {var_2: var_3}
    var_26 = {var_24: var_25}
    var_27 = {var_2: var_23, var_22: var_26}
    var_28 = 'age'
    var_29 = {var_2: var_5}
    var_30 = {var_28: var_29}
    var_31 = {var_2: var_23, var_22: var_30}
    var_32 = [var_27, var_31]
    var_33 = {var_1: var_32}
    var_34 = module_1.all_of_from_json_schema(var_33, var_0)
    var_35 = var_34.all_of
    var_36 = len(var_35)
    assert var_36 == 2



# Parsed testcases at query #5
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'allOf'
    var_2 = 'type'
    var_3 = 'minimum'
    var_4 = 'integer'
    var_5 = 10
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'minLength'
    var_8 = 'string'
    var_9 = 5
    var_10 = {var_2: var_8, var_7: var_9}
    var_11 = [var_6, var_10]
    var_12 = {var_1: var_11}
    var_13 = module_1.all_of_from_json_schema(var_12, var_0)
    var_14 = var_13.all_of
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = 0
    var_17 = var_13.all_of[var_16]
    var_18 = 1
    var_19 = var_13.all_of[var_18]
    var_20 = 'default'
    var_21 = {var_2: var_4}
    var_22 = [var_21]
    var_23 = 42
    var_24 = {var_1: var_22, var_20: var_23}
    var_25 = module_1.all_of_from_json_schema(var_24, var_0)
    var_26 = 'boolean'
    var_27 = {var_2: var_26}
    var_28 = [var_27]
    var_29 = {var_1: var_28}
    var_30 = module_1.all_of_from_json_schema(var_29, var_0)
    var_31 = var_30.all_of
    var_32 = len(var_31)
    assert var_32 == 1
    var_33 = var_30.all_of[var_16]
    var_34 = {var_2: var_4}
    var_35 = 'number'
    var_36 = {var_2: var_35}
    var_37 = [var_34, var_36]
    var_38 = {var_1: var_37}
    var_39 = {var_2: var_8}
    var_40 = [var_38, var_39]
    var_41 = {var_1: var_40}
    var_42 = module_1.all_of_from_json_schema(var_41, var_0)
    var_43 = var_42.all_of[var_16]
    var_44 = var_42.all_of[var_18]



# Parsed testcases at query #6
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
    var_17 = 'const'
    var_18 = 'A'
    var_19 = {var_17: var_18}
    var_20 = 'B'
    var_21 = {var_17: var_20}
    var_22 = [var_19, var_21]
    var_23 = {var_1: var_22, var_16: var_18}
    var_24 = module_1.one_of_from_json_schema(var_23, var_0)
    var_25 = var_24.one_of[var_12]
    var_26 = 'properties'
    var_27 = 'object'
    var_28 = 'name'
    var_29 = {var_2: var_3}
    var_30 = {var_28: var_29}
    var_31 = {var_2: var_27, var_26: var_30}
    var_32 = 'items'
    var_33 = 'array'
    var_34 = {var_2: var_5}
    var_35 = {var_2: var_33, var_32: var_34}
    var_36 = [var_31, var_35]
    var_37 = {var_1: var_36}
    var_38 = module_1.one_of_from_json_schema(var_37, var_0)
    var_39 = var_38.one_of[var_12]
    var_40 = var_38.one_of[var_14]
    var_41 = []
    var_42 = {var_1: var_41}
    var_43 = module_1.one_of_from_json_schema(var_42, var_0)
    var_44 = var_43.one_of
    var_45 = len(var_44)
    assert var_45 == 0



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'abc'

import typesystem.schemas as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = 'User'
    var_1 = {}
    var_2 = module_0.Reference(var_0, var_1)
    var_3 = 'id'
    var_4 = module_1.Integer()
    var_5 = {var_3: var_4}
    var_6 = module_1.Object(properties=var_5)
    var_7 = {var_0: var_6}

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = module_0.Array(var_0, var_1)
    var_3 = module_1.to_json_schema(var_2)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Object(additional_properties=var_0)
    var_2 = module_1.to_json_schema(var_1)



# Parsed testcases at query #8
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = '#/definitions/MyType'
    var_3 = {var_1: var_2}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)
    var_5 = 'definitions/MyType'
    var_6 = {var_1: var_5}
    var_7 = module_1.ref_from_json_schema(var_6, var_0)
    var_8 = 'type'
    var_9 = 'string'
    var_10 = {var_8: var_9}
    var_11 = module_1.ref_from_json_schema(var_10, var_0)



# Parsed testcases at query #9
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = '#/definitions/MySchema'
    var_3 = {var_1: var_2}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)
    var_5 = 'definitions/MySchema'
    var_6 = {var_1: var_5}
    var_7 = module_1.ref_from_json_schema(var_6, var_0)
    var_8 = {}
    var_9 = module_1.ref_from_json_schema(var_8, var_0)



# Parsed testcases at query #10
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = '#/definitions/MySchema'
    var_3 = {var_1: var_2}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)
    var_5 = 'definitions/MySchema'
    var_6 = {var_1: var_5}
    var_7 = module_1.ref_from_json_schema(var_6, var_0)
    var_8 = {}
    var_9 = module_1.ref_from_json_schema(var_8, var_0)



# Parsed testcases at query #11
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
    var_2 = 5
    var_3 = module_0.Integer(minimum=var_0, maximum=var_1, multiple_of=var_2)
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
    var_0 = False
    var_1 = module_0.String()
    var_2 = 1
    var_3 = 5
    var_4 = True
    var_5 = module_0.Array(var_1, min_items=var_2, max_items=var_3, unique_items=var_4)
    var_6 = module_1.to_json_schema(var_5)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Integer()
    var_2 = 'age'
    var_3 = {var_2: var_1}
    var_4 = [var_2]
    var_5 = 1
    var_6 = module_0.Object(properties=var_3, min_properties=var_5, required=var_4)
    var_7 = module_1.to_json_schema(var_6)

import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = (var_0, var_0)
    var_2 = 'b'
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
    var_0 = False
    var_1 = module_0.String()
    var_2 = module_0.Integer()
    var_3 = [var_1, var_2]
    var_4 = module_0.Union(var_3)
    var_5 = module_1.to_json_schema(var_4)

import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'User'
    var_1 = {}
    var_2 = module_0.Object(properties=var_1)
    var_3 = module_1.Reference(var_0)
    var_4 = {}
    var_5 = module_0.Object(properties=var_4)
    var_6 = {var_0: var_5}
    var_7 = module_2.to_json_schema(var_3, var_6)

def test_case_0():
    var_0 = False

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = False
    var_1 = module_0.String()
    var_2 = module_1.Not(var_1)
    var_3 = module_2.to_json_schema(var_2)

import typesystem.fields as module_0
import typesystem.composites as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = False
    var_1 = module_0.Boolean()
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = module_1.IfThenElse(var_1, var_2, var_3)
    var_5 = module_2.to_json_schema(var_4)



