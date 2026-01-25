####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
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



# Parsed testcases at query #2
#--------------------------


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
    var_7 = 'number'
    var_8 = {var_4: var_7}
    var_9 = 'boolean'
    var_10 = {var_4: var_9}
    var_11 = {var_1: var_6, var_2: var_8, var_3: var_10}
    var_12 = module_1.if_then_else_from_json_schema(var_11, var_0)
    var_13 = var_12.if_clause
    var_14 = var_12.then_clause
    var_15 = var_12.else_clause
    var_16 = {var_4: var_5}
    var_17 = {var_4: var_7}
    var_18 = {var_1: var_16, var_2: var_17}
    var_19 = module_1.if_then_else_from_json_schema(var_18, var_0)
    var_20 = var_19.if_clause
    var_21 = var_19.then_clause
    var_22 = {var_4: var_5}
    var_23 = {var_4: var_9}
    var_24 = {var_1: var_22, var_3: var_23}
    var_25 = module_1.if_then_else_from_json_schema(var_24, var_0)
    var_26 = var_25.if_clause
    var_27 = var_25.else_clause



# Parsed testcases at query #3
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'oneOf'
    var_1 = 'type'
    var_2 = 'integer'
    var_3 = {var_1: var_2}
    var_4 = 'string'
    var_5 = {var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = {var_0: var_6}
    var_8 = module_0.Definitions()
    var_9 = module_1.one_of_from_json_schema(var_7, var_8)
    var_10 = var_9.one_of
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = 0
    var_13 = var_9.one_of[var_12]
    var_14 = 1
    var_15 = var_9.one_of[var_14]
    var_16 = 'default'
    var_17 = {var_1: var_2}
    var_18 = {var_1: var_4}
    var_19 = [var_17, var_18]
    var_20 = 42
    var_21 = {var_0: var_19, var_16: var_20}
    var_22 = module_0.Definitions()
    var_23 = module_1.one_of_from_json_schema(var_21, var_22)
    var_24 = []
    var_25 = {var_0: var_24}
    var_26 = module_0.Definitions()
    var_27 = module_1.one_of_from_json_schema(var_25, var_26)
    var_28 = 'invalid'
    var_29 = module_0.Definitions()
    var_30 = module_1.one_of_from_json_schema(var_28, var_29)



# Parsed testcases at query #4
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'type'
    var_2 = 'minLength'
    var_3 = 'maxLength'
    var_4 = 'format'
    var_5 = 'pattern'
    var_6 = 'default'
    var_7 = 'string'
    var_8 = 5
    var_9 = 10
    var_10 = 'email'
    var_11 = '^[A-Za-z0-9]+$'
    var_12 = 'example@example.com'
    var_13 = {var_1: var_7, var_2: var_8, var_3: var_9, var_4: var_10, var_5: var_11, var_6: var_12}
    var_14 = False
    var_15 = module_1.from_json_schema_type(var_13, var_7, var_14, var_0)
    var_16 = 'minimum'
    var_17 = 'maximum'
    var_18 = 'exclusiveMinimum'
    var_19 = 'exclusiveMaximum'
    var_20 = 'multipleOf'
    var_21 = 'integer'
    var_22 = 1
    var_23 = 100
    var_24 = 101
    var_25 = 2
    var_26 = {var_1: var_21, var_16: var_22, var_17: var_23, var_18: var_14, var_19: var_24, var_20: var_25, var_6: var_25}
    var_27 = module_1.from_json_schema_type(var_26, var_21, var_14, var_0)
    var_28 = 'number'
    var_29 = {var_1: var_28, var_16: var_22, var_17: var_23, var_18: var_14, var_19: var_24, var_20: var_25, var_6: var_25}
    var_30 = module_1.from_json_schema_type(var_29, var_28, var_14, var_0)
    var_31 = 'boolean'
    var_32 = True
    var_33 = {var_1: var_31, var_6: var_32}
    var_34 = module_1.from_json_schema_type(var_33, var_31, var_14, var_0)
    var_35 = 'items'
    var_36 = 'minItems'
    var_37 = 'maxItems'
    var_38 = 'uniqueItems'
    var_39 = 'array'
    var_40 = {var_1: var_7}
    var_41 = True
    var_42 = 'example'
    var_43 = [var_42]
    var_44 = {var_1: var_39, var_35: var_40, var_36: var_32, var_37: var_9, var_38: var_41, var_6: var_43}
    var_45 = module_1.from_json_schema_type(var_44, var_39, var_14, var_0)
    var_46 = var_45.items
    var_47 = 'properties'
    var_48 = 'minProperties'
    var_49 = 'maxProperties'
    var_50 = 'required'
    var_51 = 'object'
    var_52 = 'name'
    var_53 = {var_1: var_7}
    var_54 = {var_52: var_53}
    var_55 = [var_52]
    var_56 = {var_52: var_42}
    var_57 = {var_1: var_51, var_47: var_54, var_48: var_41, var_49: var_25, var_50: var_55, var_6: var_56}
    var_58 = module_1.from_json_schema_type(var_57, var_51, var_14, var_0)
    var_59 = var_58.properties[var_52]



# Parsed testcases at query #5
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'allOf'
    var_2 = 'type'
    var_3 = 'minLength'
    var_4 = 'string'
    var_5 = 5
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'maxLength'
    var_8 = 10
    var_9 = {var_2: var_4, var_7: var_8}
    var_10 = [var_6, var_9]
    var_11 = {var_1: var_10}
    var_12 = module_1.all_of_from_json_schema(var_11, var_0)
    var_13 = var_12.all_of
    var_14 = len(var_13)
    assert var_14 == 2
    var_15 = 0
    var_16 = var_12.all_of[var_15]
    var_17 = 1
    var_18 = var_12.all_of[var_17]



# Parsed testcases at query #6
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
    var_8 = 'integer'
    var_9 = {var_4: var_8}
    var_10 = module_0.from_json_schema(var_9)
    var_11 = 'number'
    var_12 = {var_4: var_11}
    var_13 = module_0.from_json_schema(var_12)
    var_14 = 'boolean'
    var_15 = {var_4: var_14}
    var_16 = module_0.from_json_schema(var_15)
    var_17 = 'items'
    var_18 = 'array'
    var_19 = {var_4: var_5}
    var_20 = {var_4: var_18, var_17: var_19}
    var_21 = module_0.from_json_schema(var_20)
    var_22 = 'properties'
    var_23 = 'object'
    var_24 = 'name'
    var_25 = {var_4: var_5}
    var_26 = {var_24: var_25}
    var_27 = {var_4: var_23, var_22: var_26}
    var_28 = module_0.from_json_schema(var_27)
    var_29 = 'enum'
    var_30 = 'red'
    var_31 = 'green'
    var_32 = 'blue'
    var_33 = [var_30, var_31, var_32]
    var_34 = {var_29: var_33}
    var_35 = module_0.from_json_schema(var_34)
    var_36 = 'const'
    var_37 = 'constant_value'
    var_38 = {var_36: var_37}
    var_39 = module_0.from_json_schema(var_38)
    var_40 = 'allOf'
    var_41 = {var_4: var_5}
    var_42 = 'minLength'
    var_43 = 5
    var_44 = {var_42: var_43}
    var_45 = [var_41, var_44]
    var_46 = {var_40: var_45}
    var_47 = module_0.from_json_schema(var_46)
    var_48 = 'anyOf'
    var_49 = {var_4: var_5}
    var_50 = {var_4: var_11}
    var_51 = [var_49, var_50]
    var_52 = {var_48: var_51}
    var_53 = module_0.from_json_schema(var_52)
    var_54 = 'oneOf'
    var_55 = {var_4: var_5}
    var_56 = {var_4: var_11}
    var_57 = [var_55, var_56]
    var_58 = {var_54: var_57}
    var_59 = module_0.from_json_schema(var_58)
    var_60 = 'not'
    var_61 = {var_4: var_5}
    var_62 = {var_60: var_61}
    var_63 = module_0.from_json_schema(var_62)
    var_64 = 'if'
    var_65 = 'then'
    var_66 = {var_4: var_5}
    var_67 = {var_42: var_43}
    var_68 = {var_64: var_66, var_65: var_67}
    var_69 = module_0.from_json_schema(var_68)
    var_70 = module_1.Definitions()
    var_71 = '$ref'
    var_72 = '#/components/schemas/Example'
    var_73 = {var_71: var_72}
    var_74 = module_0.from_json_schema(var_73, var_70)
    var_75 = {var_71: var_72}
    var_76 = {var_24: var_75}
    var_77 = {var_4: var_23, var_22: var_76}
    var_78 = module_0.from_json_schema(var_77, var_70)
    var_79 = 'required'
    var_80 = 'age'
    var_81 = {var_4: var_5}
    var_82 = 'minimum'
    var_83 = {var_4: var_8, var_82: var_2}
    var_84 = {var_24: var_81, var_80: var_83}
    var_85 = [var_24]
    var_86 = {var_4: var_23, var_22: var_84, var_79: var_85}
    var_87 = module_0.from_json_schema(var_86)
    var_88 = 'additionalProperties'
    var_89 = {var_4: var_5}
    var_90 = {var_4: var_23, var_88: var_89}
    var_91 = module_0.from_json_schema(var_90)
    var_92 = 'patternProperties'
    var_93 = '^[a-z]+$'
    var_94 = {var_4: var_5}
    var_95 = {var_93: var_94}
    var_96 = {var_4: var_23, var_92: var_95}
    var_97 = module_0.from_json_schema(var_96)
    var_98 = 'minItems'
    var_99 = {var_4: var_5}
    var_100 = {var_4: var_18, var_17: var_99, var_98: var_0}
    var_101 = module_0.from_json_schema(var_100)
    var_102 = 'uniqueItems'
    var_103 = {var_4: var_5}
    var_104 = {var_4: var_18, var_17: var_103, var_102: var_0}
    var_105 = module_0.from_json_schema(var_104)
    var_106 = 'format'
    var_107 = 'email'
    var_108 = {var_4: var_5, var_106: var_107}
    var_109 = module_0.from_json_schema(var_108)
    var_110 = 'maxLength'
    var_111 = 10
    var_112 = {var_4: var_5, var_42: var_43, var_110: var_111}
    var_113 = module_0.from_json_schema(var_112)
    var_114 = [var_5, var_11]
    var_115 = {var_4: var_114}
    var_116 = module_0.from_json_schema(var_115)
    var_117 = {var_4: var_5}
    var_118 = module_0.from_json_schema(var_117)
    var_119 = module_1.Definitions()
    var_120 = {var_71: var_72}
    var_121 = module_0.from_json_schema(var_120, var_119)
    var_122 = {}
    var_123 = module_0.from_json_schema(var_122)



# Parsed testcases at query #7
#--------------------------


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'test_default'
    var_1 = module_0.Field(default=var_0)
    var_2 = module_1.get_standard_properties(var_1)
    var_3 = module_0.Field()
    var_4 = module_1.get_standard_properties(var_3)
    var_5 = None
    var_6 = module_0.Field(default=var_5)
    var_7 = module_1.get_standard_properties(var_6)
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = module_0.Field(default=var_10)
    var_12 = module_1.get_standard_properties(var_11)
    var_13 = 'Test Field'
    var_14 = module_0.Field(description=var_13, default=var_0)
    var_15 = module_1.get_standard_properties(var_14)



# Parsed testcases at query #8
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'enum'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = module_0.Definitions()
    var_7 = module_1.enum_from_json_schema(var_5, var_6)
    var_8 = 'default'
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = [var_9, var_10, var_11]
    var_13 = {var_0: var_12, var_8: var_10}
    var_14 = module_0.Definitions()
    var_15 = module_1.enum_from_json_schema(var_13, var_14)
    var_16 = []
    var_17 = {var_0: var_16}
    var_18 = module_0.Definitions()
    var_19 = module_1.enum_from_json_schema(var_17, var_18)
    var_20 = None
    var_21 = [var_20, var_1, var_2]
    var_22 = {var_18: var_21}
    var_23 = module_0.Definitions()
    var_24 = module_1.enum_from_json_schema(var_22, var_23)
    var_25 = True
    var_26 = [var_9, var_1, var_25]
    var_27 = {var_18: var_26}
    var_28 = module_0.Definitions()
    var_29 = module_1.enum_from_json_schema(var_27, var_28)



# Parsed testcases at query #9
#--------------------------


import typesystem.fields as module_0
import typesystem.json_schema as module_1
import typesystem.composites as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True
    var_2 = module_2.NeverMatch()
    var_3 = module_1.to_json_schema(var_2)
    assert var_3 is False
    var_4 = True
    var_5 = 10
    var_6 = '^[a-z]+$'
    var_7 = 'email'
    var_8 = module_0.String(max_length=var_5, min_length=var_4, pattern=var_6, format=var_7)
    var_9 = 'type'
    var_10 = 'minLength'
    var_11 = 'maxLength'
    var_12 = 'pattern'
    var_13 = 'format'
    var_14 = 'string'
    var_15 = 'null'
    var_16 = [var_14, var_15]
    var_17 = {var_9: var_16, var_10: var_4, var_11: var_5, var_12: var_6, var_13: var_7}
    var_18 = module_1.to_json_schema(var_8)
    var_19 = 0
    var_20 = 100
    var_21 = 5
    var_22 = module_0.Integer(minimum=var_19, maximum=var_20, multiple_of=var_21)
    var_23 = 'minimum'
    var_24 = 'maximum'
    var_25 = 'multipleOf'
    var_26 = 'integer'
    var_27 = [var_26, var_15]
    var_28 = {var_9: var_27, var_23: var_19, var_24: var_20, var_25: var_21}
    var_29 = module_1.to_json_schema(var_22)
    var_30 = module_0.Boolean()
    var_31 = 'boolean'
    var_32 = [var_31, var_15]
    var_33 = {var_9: var_32}
    var_34 = module_1.to_json_schema(var_30)
    var_35 = module_0.String()
    var_36 = module_0.Array(var_35, min_items=var_4, max_items=var_5, unique_items=var_4)
    var_37 = 'minItems'
    var_38 = 'maxItems'
    var_39 = 'items'
    var_40 = 'uniqueItems'
    var_41 = 'array'
    var_42 = [var_41, var_15]
    var_43 = {var_9: var_14}
    var_44 = {var_9: var_42, var_37: var_4, var_38: var_5, var_39: var_43, var_40: var_4}
    var_45 = module_1.to_json_schema(var_36)
    var_46 = 'name'
    var_47 = module_0.String()
    var_48 = {var_46: var_47}
    var_49 = [var_46]
    var_50 = module_0.Object(properties=var_48, required=var_49)
    var_51 = 'properties'
    var_52 = 'required'
    var_53 = 'object'
    var_54 = [var_53, var_15]
    var_55 = {var_9: var_14}
    var_56 = {var_46: var_55}
    var_57 = [var_46]
    var_58 = {var_9: var_54, var_51: var_56, var_52: var_57}
    var_59 = module_1.to_json_schema(var_50)
    var_60 = 'Person'
    var_61 = module_0.String()
    var_62 = {var_46: var_61}
    var_63 = module_0.Object(properties=var_62)
    var_64 = {var_60: var_63}
    var_65 = module_3.Reference(var_60, var_64)
    var_66 = '$ref'
    var_67 = 'components'
    var_68 = '#/components/schemas/Person'
    var_69 = 'schemas'
    var_70 = {var_9: var_14}
    var_71 = {var_46: var_70}
    var_72 = {var_9: var_53, var_51: var_71}
    var_73 = {var_60: var_72}
    var_74 = {var_69: var_73}
    var_75 = {var_66: var_68, var_67: var_74}
    var_76 = module_1.to_json_schema(var_65)
    var_77 = module_0.String()
    var_78 = module_0.Integer()
    var_79 = [var_77, var_78]
    var_80 = module_0.Union(var_79)
    var_81 = 'anyOf'
    var_82 = {var_9: var_14}
    var_83 = {var_9: var_26}
    var_84 = [var_82, var_83]
    var_85 = {var_81: var_84}
    var_86 = module_1.to_json_schema(var_80)
    var_87 = module_0.String(min_length=var_4)
    var_88 = module_0.String(max_length=var_5)
    var_89 = [var_87, var_88]
    var_90 = module_2.AllOf(var_89)
    var_91 = 'allOf'
    var_92 = {var_9: var_14, var_10: var_4}
    var_93 = {var_9: var_14, var_11: var_5}
    var_94 = [var_92, var_93]
    var_95 = {var_91: var_94}
    var_96 = module_1.to_json_schema(var_90)
    var_97 = module_0.String(min_length=var_21)
    var_98 = module_0.String(max_length=var_5)
    var_99 = module_0.Integer()
    var_100 = module_2.IfThenElse(var_97, var_98, var_99)
    var_101 = 'if'
    var_102 = 'then'
    var_103 = 'else'
    var_104 = {var_9: var_14, var_10: var_21}
    var_105 = {var_9: var_14, var_11: var_5}
    var_106 = {var_9: var_26}
    var_107 = {var_101: var_104, var_102: var_105, var_103: var_106}
    var_108 = module_1.to_json_schema(var_100)
    var_109 = 'All tests passed!'
    var_110 = print(var_109)



# Parsed testcases at query #10
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
    var_10 = '$ref'
    var_11 = '#/definitions/Example'
    var_12 = {var_10: var_11}
    var_13 = module_0.from_json_schema(var_12)
    var_14 = 'enum'
    var_15 = 'value1'
    var_16 = 'value2'
    var_17 = [var_15, var_16]
    var_18 = {var_14: var_17}
    var_19 = module_0.from_json_schema(var_18)
    var_20 = 'const'
    var_21 = 'value'
    var_22 = {var_20: var_21}
    var_23 = module_0.from_json_schema(var_22)
    var_24 = 'allOf'
    var_25 = {var_4: var_6}
    var_26 = {var_5: var_7}
    var_27 = [var_25, var_26]
    var_28 = {var_24: var_27}
    var_29 = module_0.from_json_schema(var_28)
    var_30 = 'anyOf'
    var_31 = {var_4: var_6}
    var_32 = 'number'
    var_33 = {var_4: var_32}
    var_34 = [var_31, var_33]
    var_35 = {var_30: var_34}
    var_36 = module_0.from_json_schema(var_35)
    var_37 = 'oneOf'
    var_38 = {var_4: var_6}
    var_39 = {var_4: var_32}
    var_40 = [var_38, var_39]
    var_41 = {var_37: var_40}
    var_42 = module_0.from_json_schema(var_41)
    var_43 = 'not'
    var_44 = {var_4: var_6}
    var_45 = {var_43: var_44}
    var_46 = module_0.from_json_schema(var_45)
    var_47 = 'if'
    var_48 = 'then'
    var_49 = 'else'
    var_50 = {var_4: var_6}
    var_51 = {var_5: var_7}
    var_52 = {var_4: var_32}
    var_53 = {var_47: var_50, var_48: var_51, var_49: var_52}
    var_54 = module_0.from_json_schema(var_53)
    var_55 = 'maxLength'
    var_56 = 10
    var_57 = {var_4: var_6, var_5: var_7, var_55: var_56}
    var_58 = module_0.from_json_schema(var_57)



# Parsed testcases at query #11
#--------------------------


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = '#/components/schemas/MySchema'
    var_1 = 1
    var_2 = module_0.String(min_length=var_1)
    var_3 = {var_0: var_2}
    var_4 = '$ref'
    var_5 = {var_4: var_0}
    var_6 = module_1.ref_from_json_schema(var_5, var_3)



# Parsed testcases at query #12
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = '#/components/schemas/User'
    var_3 = {var_1: var_2}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)
    var_5 = '$ref'
    var_6 = 'http://example.com'
    var_7 = {var_5: var_6}
    var_8 = module_1.ref_from_json_schema(var_7, var_0)



# Parsed testcases at query #13
#--------------------------


import typesystem.fields as module_0
import typesystem.json_schema as module_1
import typesystem.composites as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True
    var_2 = module_2.NeverMatch()
    var_3 = module_1.to_json_schema(var_2)
    assert var_3 is False
    var_4 = True
    var_5 = 10
    var_6 = '^[a-z]+$'
    var_7 = 'email'
    var_8 = 'test@example.com'
    var_9 = module_0.String(max_length=var_5, min_length=var_4, pattern=var_6, format=var_7)
    var_10 = 'type'
    var_11 = 'minLength'
    var_12 = 'maxLength'
    var_13 = 'pattern'
    var_14 = 'format'
    var_15 = 'default'
    var_16 = 'string'
    var_17 = 'null'
    var_18 = [var_16, var_17]
    var_19 = {var_10: var_18, var_11: var_4, var_12: var_5, var_13: var_6, var_14: var_7, var_15: var_8}
    var_20 = module_1.to_json_schema(var_9)
    var_21 = 100
    var_22 = 0
    var_23 = 101
    var_24 = 5
    var_25 = 50
    var_26 = module_0.Integer(minimum=var_4, maximum=var_21, exclusive_minimum=var_22, exclusive_maximum=var_23, multiple_of=var_24)
    var_27 = 'minimum'
    var_28 = 'maximum'
    var_29 = 'exclusiveMinimum'
    var_30 = 'exclusiveMaximum'
    var_31 = 'multipleOf'
    var_32 = 'integer'
    var_33 = [var_32, var_17]
    var_34 = {var_10: var_33, var_27: var_4, var_28: var_21, var_29: var_22, var_30: var_23, var_31: var_24, var_15: var_25}
    var_35 = module_1.to_json_schema(var_26)
    var_36 = module_0.Boolean()
    var_37 = 'boolean'
    var_38 = [var_37, var_17]
    var_39 = {var_10: var_38, var_15: var_4}
    var_40 = module_1.to_json_schema(var_36)
    var_41 = module_0.Integer()
    var_42 = 2
    var_43 = 3
    var_44 = [var_4, var_42, var_43]
    var_45 = module_0.Array(var_41, var_4, var_4, var_5, unique_items=var_4)
    var_46 = 'minItems'
    var_47 = 'maxItems'
    var_48 = 'items'
    var_49 = 'additionalItems'
    var_50 = 'uniqueItems'
    var_51 = 'array'
    var_52 = [var_51, var_17]
    var_53 = {var_10: var_32}
    var_54 = [var_4, var_42, var_43]
    var_55 = {var_10: var_52, var_46: var_4, var_47: var_5, var_48: var_53, var_49: var_4, var_50: var_4, var_15: var_54}
    var_56 = module_1.to_json_schema(var_45)
    var_57 = 'name'
    var_58 = module_0.String()
    var_59 = {var_57: var_58}
    var_60 = module_0.Integer()
    var_61 = {var_6: var_60}
    var_62 = module_0.String()
    var_63 = [var_57]
    var_64 = 'John'
    var_65 = {var_57: var_64}
    var_66 = module_0.Object(properties=var_59, pattern_properties=var_61, additional_properties=var_4, property_names=var_62, min_properties=var_4, max_properties=var_5, required=var_63)
    var_67 = 'properties'
    var_68 = 'patternProperties'
    var_69 = 'additionalProperties'
    var_70 = 'propertyNames'
    var_71 = 'minProperties'
    var_72 = 'maxProperties'
    var_73 = 'required'
    var_74 = 'object'
    var_75 = [var_74, var_17]
    var_76 = {var_10: var_16}
    var_77 = {var_57: var_76}
    var_78 = {var_10: var_32}
    var_79 = {var_6: var_78}
    var_80 = {var_10: var_16}
    var_81 = [var_57]
    var_82 = {var_57: var_64}
    var_83 = {var_10: var_75, var_67: var_77, var_68: var_79, var_69: var_4, var_70: var_80, var_71: var_4, var_72: var_5, var_73: var_81, var_15: var_82}
    var_84 = module_1.to_json_schema(var_66)
    var_85 = 'age'
    var_86 = module_0.String()
    var_87 = module_0.Integer()
    var_88 = {var_57: var_86, var_85: var_87}
    var_89 = [var_57]
    var_90 = 30
    var_91 = {var_57: var_64, var_85: var_90}
    var_92 = module_3.Schema(var_88)
    var_93 = [var_74, var_17]
    var_94 = {var_10: var_16}
    var_95 = {var_10: var_32}
    var_96 = {var_57: var_94, var_85: var_95}
    var_97 = [var_57]
    var_98 = {var_57: var_64, var_85: var_90}
    var_99 = {var_10: var_93, var_67: var_96, var_73: var_97, var_15: var_98}
    var_100 = module_1.to_json_schema(var_92)
    var_101 = 'male'
    var_102 = 'Male'
    var_103 = (var_101, var_102)
    var_104 = 'female'
    var_105 = 'Female'
    var_106 = (var_104, var_105)
    var_107 = [var_103, var_106]
    var_108 = module_0.Choice(choices=var_107)
    var_109 = 'enum'
    var_110 = [var_101, var_104]
    var_111 = {var_109: var_110, var_15: var_101}
    var_112 = module_1.to_json_schema(var_108)
    var_113 = 42
    var_114 = module_0.Const(var_113)
    var_115 = 'const'
    var_116 = {var_115: var_113, var_15: var_113}
    var_117 = module_1.to_json_schema(var_114)
    var_118 = module_0.Integer()
    var_119 = module_0.String()
    var_120 = [var_118, var_119]
    var_121 = module_0.Union(var_120)
    var_122 = 'anyOf'
    var_123 = {var_10: var_32}
    var_124 = {var_10: var_16}
    var_125 = [var_123, var_124]
    var_126 = {var_122: var_125, var_15: var_5}
    var_127 = module_1.to_json_schema(var_121)
    var_128 = module_0.Integer()
    var_129 = module_0.String()
    var_130 = [var_128, var_129]
    var_131 = module_2.OneOf(var_130)
    var_132 = 'oneOf'
    var_133 = {var_10: var_32}
    var_134 = {var_10: var_16}
    var_135 = [var_133, var_134]
    var_136 = {var_132: var_135, var_15: var_5}
    var_137 = module_1.to_json_schema(var_131)
    var_138 = module_0.Integer()
    var_139 = module_0.Integer(minimum=var_5)
    var_140 = [var_138, var_139]
    var_141 = module_2.AllOf(var_140)
    var_142 = 'allOf'
    var_143 = {var_10: var_32}
    var_144 = {var_10: var_32, var_27: var_5}
    var_145 = [var_143, var_144]
    var_146 = {var_142: var_145, var_15: var_5}
    var_147 = module_1.to_json_schema(var_141)
    var_148 = module_0.Integer(minimum=var_5)
    var_149 = module_0.String()
    var_150 = module_0.Integer()
    var_151 = module_2.IfThenElse(var_148, var_149, var_150)
    var_152 = 'if'
    var_153 = 'then'
    var_154 = 'else'
    var_155 = {var_10: var_32, var_27: var_5}
    var_156 = {var_10: var_16}
    var_157 = {var_10: var_32}
    var_158 = {var_152: var_155, var_153: var_156, var_154: var_157, var_15: var_5}
    var_159 = module_1.to_json_schema(var_151)
    var_160 = module_0.Integer(minimum=var_5)
    var_161 = module_2.Not(var_160)
    var_162 = 'not'
    var_163 = {var_10: var_32, var_27: var_5}
    var_164 = {var_162: var_163, var_15: var_5}
    var_165 = module_1.to_json_schema(var_161)
    var_166 = 'Person'
    var_167 = module_0.String()
    var_168 = module_0.Integer()
    var_169 = {var_57: var_167, var_85: var_168}
    var_170 = module_3.Schema(var_169)
    var_171 = {var_166: var_170}
    var_172 = 'components'
    var_173 = 'schemas'
    var_174 = {var_10: var_16}
    var_175 = {var_10: var_32}
    var_176 = {var_57: var_174, var_85: var_175}
    var_177 = {var_10: var_74, var_67: var_176}
    var_178 = {var_166: var_177}
    var_179 = {var_173: var_178}
    var_180 = {var_172: var_179}
    var_181 = module_0.String()
    var_182 = {var_57: var_181}
    var_183 = module_3.Schema(var_182)
    var_184 = {var_166: var_183}
    var_185 = '$ref'
    var_186 = '#/components/schemas/Person'
    var_187 = {var_185: var_186}
    var_188 = 'invalid_field'
    var_189 = module_1.to_json_schema(var_188)



# Parsed testcases at query #14
#--------------------------


import typesystem.fields as module_0
import typesystem.json_schema as module_1
import typesystem.composites as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True
    var_2 = module_2.NeverMatch()
    var_3 = module_1.to_json_schema(var_2)
    assert var_3 is False
    var_4 = module_0.String()
    var_5 = module_1.to_json_schema(var_4)
    var_6 = module_0.Integer()
    var_7 = module_1.to_json_schema(var_6)
    var_8 = module_0.Float()
    var_9 = module_1.to_json_schema(var_8)
    var_10 = module_0.Boolean()
    var_11 = module_1.to_json_schema(var_10)
    var_12 = module_0.Array()
    var_13 = module_1.to_json_schema(var_12)
    var_14 = module_0.Object()
    var_15 = module_1.to_json_schema(var_14)
    var_16 = module_3.Schema()
    var_17 = module_1.to_json_schema(var_16)
    var_18 = 'a'
    var_19 = 'A'
    var_20 = (var_18, var_19)
    var_21 = [var_20]
    var_22 = module_0.Choice(choices=var_21)
    var_23 = module_1.to_json_schema(var_22)
    var_24 = True
    var_25 = module_0.Const(var_24)
    var_26 = module_1.to_json_schema(var_25)
    var_27 = module_0.String()
    var_28 = module_0.Integer()
    var_29 = [var_27, var_28]
    var_30 = module_0.Union(var_29)
    var_31 = module_1.to_json_schema(var_30)
    var_32 = module_0.String()
    var_33 = module_0.Integer()
    var_34 = [var_32, var_33]
    var_35 = module_2.OneOf(var_34)
    var_36 = module_1.to_json_schema(var_35)
    var_37 = module_0.String()
    var_38 = module_0.Integer()
    var_39 = [var_37, var_38]
    var_40 = module_2.AllOf(var_39)
    var_41 = module_1.to_json_schema(var_40)
    var_42 = module_0.String()
    var_43 = module_0.Integer()
    var_44 = module_0.Boolean()
    var_45 = module_2.IfThenElse(var_42, var_43, var_44)
    var_46 = module_1.to_json_schema(var_45)
    var_47 = module_0.String()
    var_48 = module_2.Not(var_47)
    var_49 = module_1.to_json_schema(var_48)



# Parsed testcases at query #15
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = '#/definitions/Example'
    var_3 = {var_1: var_2}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)



# Parsed testcases at query #16
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = module_0.Definitions()
    var_4 = module_1.type_from_json_schema(var_2, var_3)
    var_5 = 'number'
    var_6 = [var_1, var_5]
    var_7 = {var_0: var_6}
    var_8 = module_1.type_from_json_schema(var_7, var_3)
    var_9 = var_8.any_of
    var_10 = len(var_9)
    assert var_10 == 2
    var_11 = 0
    var_12 = var_8.any_of[var_11]
    var_13 = 1
    var_14 = var_8.any_of[var_13]
    var_15 = 'nullable'
    var_16 = True
    var_17 = {var_0: var_1, var_15: var_16}
    var_18 = module_1.type_from_json_schema(var_17, var_3)
    var_19 = True
    var_20 = {var_15: var_19}
    var_21 = module_1.type_from_json_schema(var_20, var_3)
    var_22 = {}
    var_23 = module_1.type_from_json_schema(var_22, var_3)



# Parsed testcases at query #17
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'type'
    var_2 = 'minLength'
    var_3 = 'string'
    var_4 = 1
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_1.type_from_json_schema(var_5, var_0)
    var_7 = 'minimum'
    var_8 = 'number'
    var_9 = 0
    var_10 = {var_1: var_8, var_7: var_9}
    var_11 = module_1.type_from_json_schema(var_10, var_0)
    var_12 = 'multipleOf'
    var_13 = 'integer'
    var_14 = 2
    var_15 = {var_1: var_13, var_12: var_14}
    var_16 = module_1.type_from_json_schema(var_15, var_0)
    var_17 = 'boolean'
    var_18 = {var_1: var_17}
    var_19 = module_1.type_from_json_schema(var_18, var_0)
    var_20 = 'null'
    var_21 = {var_1: var_20}
    var_22 = module_1.type_from_json_schema(var_21, var_0)
    var_23 = [var_3, var_20]
    var_24 = {var_1: var_23}
    var_25 = module_1.type_from_json_schema(var_24, var_0)
    var_26 = [var_3, var_8]
    var_27 = {var_1: var_26}
    var_28 = module_1.type_from_json_schema(var_27, var_0)
    var_29 = var_28.any_of
    var_30 = len(var_29)
    assert var_30 == 2
    var_31 = var_28.any_of[var_9]
    var_32 = var_28.any_of[var_4]
    var_33 = []
    var_34 = {var_1: var_33}
    var_35 = module_1.type_from_json_schema(var_34, var_0)
    var_36 = 'allow_null'
    var_37 = True
    var_38 = {var_1: var_20, var_36: var_37}
    var_39 = module_1.type_from_json_schema(var_38, var_0)



# Parsed testcases at query #18
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = module_0.Definitions()
    var_4 = module_1.type_from_json_schema(var_2, var_3)
    var_5 = 'number'
    var_6 = [var_1, var_5]
    var_7 = {var_0: var_6}
    var_8 = module_0.Definitions()
    var_9 = module_1.type_from_json_schema(var_7, var_8)
    var_10 = var_9.any_of
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = 0
    var_13 = var_9.any_of[var_12]
    var_14 = 1
    var_15 = var_9.any_of[var_14]
    var_16 = 'null'
    var_17 = {var_0: var_16}
    var_18 = module_0.Definitions()
    var_19 = module_1.type_from_json_schema(var_17, var_18)
    var_20 = [var_1, var_16]
    var_21 = {var_0: var_20}
    var_22 = module_0.Definitions()
    var_23 = module_1.type_from_json_schema(var_21, var_22)
    var_24 = {}
    var_25 = module_0.Definitions()
    var_26 = module_1.type_from_json_schema(var_24, var_25)
    var_27 = 'minLength'
    var_28 = 5
    var_29 = {var_0: var_1, var_27: var_28}
    var_30 = module_0.Definitions()
    var_31 = module_1.type_from_json_schema(var_29, var_30)



# Parsed testcases at query #19
#--------------------------


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
    var_8 = 1
    var_9 = 10
    var_10 = 0
    var_11 = 11
    var_12 = 2
    var_13 = 5.5
    var_14 = {var_0: var_7, var_1: var_8, var_2: var_9, var_3: var_10, var_4: var_11, var_5: var_12, var_6: var_13}
    var_15 = False
    var_16 = module_0.Definitions()
    var_17 = module_1.from_json_schema_type(var_14, var_7, var_15, var_16)
    var_18 = 'integer'
    var_19 = 5
    var_20 = {var_0: var_18, var_1: var_8, var_2: var_9, var_3: var_15, var_4: var_11, var_5: var_12, var_6: var_19}
    var_21 = False
    var_22 = module_0.Definitions()
    var_23 = module_1.from_json_schema_type(var_20, var_18, var_21, var_22)
    var_24 = 'minLength'
    var_25 = 'maxLength'
    var_26 = 'format'
    var_27 = 'pattern'
    var_28 = 'string'
    var_29 = 'email'
    var_30 = '^[a-zA-Z0-9]+$'
    var_31 = 'test'
    var_32 = {var_0: var_28, var_24: var_8, var_25: var_9, var_26: var_29, var_27: var_30, var_6: var_31}
    var_33 = False
    var_34 = module_0.Definitions()
    var_35 = module_1.from_json_schema_type(var_32, var_28, var_33, var_34)
    var_36 = 'boolean'
    var_37 = True
    var_38 = {var_0: var_36, var_6: var_37}
    var_39 = False
    var_40 = module_0.Definitions()
    var_41 = module_1.from_json_schema_type(var_38, var_36, var_39, var_40)
    var_42 = 'items'
    var_43 = 'minItems'
    var_44 = 'maxItems'
    var_45 = 'uniqueItems'
    var_46 = 'array'
    var_47 = {var_0: var_28}
    var_48 = True
    var_49 = [var_31]
    var_50 = {var_0: var_46, var_42: var_47, var_43: var_37, var_44: var_9, var_45: var_48, var_6: var_49}
    var_51 = False
    var_52 = module_0.Definitions()
    var_53 = module_1.from_json_schema_type(var_50, var_46, var_51, var_52)
    var_54 = 'properties'
    var_55 = 'minProperties'
    var_56 = 'maxProperties'
    var_57 = 'required'
    var_58 = 'object'
    var_59 = {var_0: var_28}
    var_60 = {var_31: var_59}
    var_61 = [var_31]
    var_62 = 'value'
    var_63 = {var_31: var_62}
    var_64 = {var_0: var_58, var_54: var_60, var_55: var_48, var_56: var_9, var_57: var_61, var_6: var_63}
    var_65 = False
    var_66 = module_0.Definitions()
    var_67 = module_1.from_json_schema_type(var_64, var_58, var_65, var_66)
    var_68 = {}
    var_69 = 'invalid'
    var_70 = False
    var_71 = module_0.Definitions()
    var_72 = module_1.from_json_schema_type(var_68, var_69, var_70, var_71)



# Parsed testcases at query #20
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = module_1.type_from_json_schema(var_3, var_0)
    var_5 = 'null'
    var_6 = [var_2, var_5]
    var_7 = {var_1: var_6}
    var_8 = module_1.type_from_json_schema(var_7, var_0)
    var_9 = 'integer'
    var_10 = [var_2, var_9]
    var_11 = {var_1: var_10}
    var_12 = module_1.type_from_json_schema(var_11, var_0)
    var_13 = var_12.any_of
    var_14 = len(var_13)
    assert var_14 == 2
    var_15 = 0
    var_16 = var_12.any_of[var_15]
    var_17 = 1
    var_18 = var_12.any_of[var_17]
    var_19 = [var_2, var_9, var_5]
    var_20 = {var_1: var_19}
    var_21 = module_1.type_from_json_schema(var_20, var_0)
    var_22 = var_21.any_of
    var_23 = len(var_22)
    assert var_23 == 2
    var_24 = var_21.any_of[var_15]
    var_25 = var_21.any_of[var_17]
    var_26 = [var_5]
    var_27 = {var_1: var_26}
    var_28 = module_1.type_from_json_schema(var_27, var_0)
    var_29 = []
    var_30 = {var_1: var_29}
    var_31 = module_1.type_from_json_schema(var_30, var_0)



# Parsed testcases at query #21
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1
import re as module_2

def test_case_0():
    var_0 = 'type'
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'exclusiveMinimum'
    var_4 = 'exclusiveMaximum'
    var_5 = 'multipleOf'
    var_6 = 'default'
    var_7 = 'number'
    var_8 = 1
    var_9 = 10
    var_10 = 0
    var_11 = 11
    var_12 = 2
    var_13 = 5.0
    var_14 = {var_0: var_7, var_1: var_8, var_2: var_9, var_3: var_10, var_4: var_11, var_5: var_12, var_6: var_13}
    var_15 = False
    var_16 = module_0.Definitions()
    var_17 = module_1.from_json_schema_type(var_14, var_7, var_15, var_16)
    var_18 = 'integer'
    var_19 = {var_0: var_18, var_1: var_8, var_2: var_9, var_3: var_15, var_4: var_11, var_5: var_12, var_6: var_13}
    var_20 = False
    var_21 = module_0.Definitions()
    var_22 = module_1.from_json_schema_type(var_19, var_18, var_20, var_21)
    var_23 = 'minLength'
    var_24 = 'maxLength'
    var_25 = 'format'
    var_26 = 'pattern'
    var_27 = 'string'
    var_28 = 'email'
    var_29 = '^[a-z]+$'
    var_30 = 'test'
    var_31 = {var_0: var_27, var_23: var_8, var_24: var_9, var_25: var_28, var_26: var_29, var_6: var_30}
    var_32 = False
    var_33 = module_0.Definitions()
    var_34 = module_1.from_json_schema_type(var_31, var_27, var_32, var_33)
    var_35 = module_2.compile(var_29)
    var_36 = 'boolean'
    var_37 = True
    var_38 = {var_0: var_36, var_6: var_37}
    var_39 = False
    var_40 = module_0.Definitions()
    var_41 = module_1.from_json_schema_type(var_38, var_36, var_39, var_40)
    var_42 = 'items'
    var_43 = 'minItems'
    var_44 = 'maxItems'
    var_45 = 'uniqueItems'
    var_46 = 'array'
    var_47 = {var_0: var_27}
    var_48 = True
    var_49 = [var_30]
    var_50 = {var_0: var_46, var_42: var_47, var_43: var_37, var_44: var_9, var_45: var_48, var_6: var_49}
    var_51 = False
    var_52 = module_0.Definitions()
    var_53 = module_1.from_json_schema_type(var_50, var_46, var_51, var_52)
    var_54 = var_53.items
    var_55 = 'properties'
    var_56 = 'minProperties'
    var_57 = 'maxProperties'
    var_58 = 'required'
    var_59 = 'object'
    var_60 = 'name'
    var_61 = {var_0: var_27}
    var_62 = {var_60: var_61}
    var_63 = [var_60]
    var_64 = {var_60: var_30}
    var_65 = {var_0: var_59, var_55: var_62, var_56: var_48, var_57: var_12, var_58: var_63, var_6: var_64}
    var_66 = False
    var_67 = module_0.Definitions()
    var_68 = module_1.from_json_schema_type(var_65, var_59, var_66, var_67)
    var_69 = var_68.properties[var_60]
    var_70 = None
    var_71 = {var_0: var_27, var_6: var_70}
    var_72 = True
    var_73 = module_0.Definitions()
    var_74 = module_1.from_json_schema_type(var_71, var_27, var_72, var_73)
    var_75 = {}
    var_76 = 'invalid'
    var_77 = False
    var_78 = module_0.Definitions()
    var_79 = module_1.from_json_schema_type(var_75, var_76, var_77, var_78)



# Parsed testcases at query #22
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1
import re as module_2

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'type'
    var_2 = 'minLength'
    var_3 = 'maxLength'
    var_4 = 'pattern'
    var_5 = 'format'
    var_6 = 'default'
    var_7 = 'string'
    var_8 = 1
    var_9 = 10
    var_10 = '^[a-z]+$'
    var_11 = 'email'
    var_12 = 'test@example.com'
    var_13 = {var_1: var_7, var_2: var_8, var_3: var_9, var_4: var_10, var_5: var_11, var_6: var_12}
    var_14 = False
    var_15 = module_1.from_json_schema_type(var_13, var_7, var_14, var_0)
    var_16 = module_2.compile(var_10)
    var_17 = 'minimum'
    var_18 = 'maximum'
    var_19 = 'exclusiveMinimum'
    var_20 = 'exclusiveMaximum'
    var_21 = 'multipleOf'
    var_22 = 'integer'
    var_23 = 11
    var_24 = 2
    var_25 = {var_1: var_22, var_17: var_8, var_18: var_9, var_19: var_14, var_20: var_23, var_21: var_24, var_6: var_24}
    var_26 = module_1.from_json_schema_type(var_25, var_22, var_14, var_0)
    var_27 = 'number'
    var_28 = {var_1: var_27, var_17: var_8, var_18: var_9, var_19: var_14, var_20: var_23, var_21: var_24, var_6: var_24}
    var_29 = module_1.from_json_schema_type(var_28, var_27, var_14, var_0)
    var_30 = 'boolean'
    var_31 = True
    var_32 = {var_1: var_30, var_6: var_31}
    var_33 = module_1.from_json_schema_type(var_32, var_30, var_14, var_0)
    var_34 = 'items'
    var_35 = 'minItems'
    var_36 = 'maxItems'
    var_37 = 'uniqueItems'
    var_38 = 'array'
    var_39 = {var_1: var_7}
    var_40 = True
    var_41 = 'test'
    var_42 = [var_41]
    var_43 = {var_1: var_38, var_34: var_39, var_35: var_31, var_36: var_9, var_37: var_40, var_6: var_42}
    var_44 = module_1.from_json_schema_type(var_43, var_38, var_14, var_0)
    var_45 = var_44.items
    var_46 = 'properties'
    var_47 = 'required'
    var_48 = 'minProperties'
    var_49 = 'maxProperties'
    var_50 = 'object'
    var_51 = 'name'
    var_52 = {var_1: var_7}
    var_53 = {var_51: var_52}
    var_54 = [var_51]
    var_55 = {var_51: var_41}
    var_56 = {var_1: var_50, var_46: var_53, var_47: var_54, var_48: var_40, var_49: var_24, var_6: var_55}
    var_57 = module_1.from_json_schema_type(var_56, var_50, var_14, var_0)
    var_58 = var_57.properties[var_51]



# Parsed testcases at query #23
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1
import re as module_2

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
    var_10 = 2
    var_11 = 10
    var_12 = {var_0: var_7, var_1: var_8, var_2: var_9, var_3: var_8, var_4: var_9, var_5: var_10, var_6: var_11}
    var_13 = False
    var_14 = module_0.Definitions()
    var_15 = module_1.from_json_schema_type(var_12, var_7, var_13, var_14)
    var_16 = 'integer'
    var_17 = {var_0: var_16, var_1: var_13, var_2: var_9, var_3: var_13, var_4: var_9, var_5: var_10, var_6: var_11}
    var_18 = False
    var_19 = module_0.Definitions()
    var_20 = module_1.from_json_schema_type(var_17, var_16, var_18, var_19)
    var_21 = 'minLength'
    var_22 = 'maxLength'
    var_23 = 'format'
    var_24 = 'pattern'
    var_25 = 'string'
    var_26 = 1
    var_27 = 'email'
    var_28 = '^[a-z]+$'
    var_29 = 'test'
    var_30 = {var_0: var_25, var_21: var_26, var_22: var_11, var_23: var_27, var_24: var_28, var_6: var_29}
    var_31 = False
    var_32 = module_0.Definitions()
    var_33 = module_1.from_json_schema_type(var_30, var_25, var_31, var_32)
    var_34 = module_2.compile(var_28)
    var_35 = 'boolean'
    var_36 = True
    var_37 = {var_0: var_35, var_6: var_36}
    var_38 = False
    var_39 = module_0.Definitions()
    var_40 = module_1.from_json_schema_type(var_37, var_35, var_38, var_39)
    var_41 = 'items'
    var_42 = 'minItems'
    var_43 = 'maxItems'
    var_44 = 'uniqueItems'
    var_45 = 'array'
    var_46 = {var_0: var_25}
    var_47 = True
    var_48 = [var_29]
    var_49 = {var_0: var_45, var_41: var_46, var_42: var_36, var_43: var_11, var_44: var_47, var_6: var_48}
    var_50 = False
    var_51 = module_0.Definitions()
    var_52 = module_1.from_json_schema_type(var_49, var_45, var_50, var_51)
    var_53 = var_52.items
    var_54 = 'properties'
    var_55 = 'minProperties'
    var_56 = 'maxProperties'
    var_57 = 'required'
    var_58 = 'object'
    var_59 = 'name'
    var_60 = {var_0: var_25}
    var_61 = {var_59: var_60}
    var_62 = [var_59]
    var_63 = {var_59: var_29}
    var_64 = {var_0: var_58, var_54: var_61, var_55: var_47, var_56: var_11, var_57: var_62, var_6: var_63}
    var_65 = False
    var_66 = module_0.Definitions()
    var_67 = module_1.from_json_schema_type(var_64, var_58, var_65, var_66)
    var_68 = var_67.properties[var_59]



# Parsed testcases at query #24
#--------------------------


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
    var_8 = 1
    var_9 = 10
    var_10 = 0
    var_11 = 11
    var_12 = 2
    var_13 = 3
    var_14 = {var_0: var_7, var_1: var_8, var_2: var_9, var_3: var_10, var_4: var_11, var_5: var_12, var_6: var_13}
    var_15 = False
    var_16 = module_0.Definitions()
    var_17 = module_1.from_json_schema_type(var_14, var_7, var_15, var_16)
    var_18 = 'integer'
    var_19 = {var_0: var_18, var_1: var_8, var_2: var_9, var_3: var_15, var_4: var_11, var_5: var_12, var_6: var_13}
    var_20 = False
    var_21 = module_0.Definitions()
    var_22 = module_1.from_json_schema_type(var_19, var_18, var_20, var_21)
    var_23 = 'minLength'
    var_24 = 'maxLength'
    var_25 = 'format'
    var_26 = 'pattern'
    var_27 = 'string'
    var_28 = 'email'
    var_29 = '^[a-z]+$'
    var_30 = 'test'
    var_31 = {var_0: var_27, var_23: var_8, var_24: var_9, var_25: var_28, var_26: var_29, var_6: var_30}
    var_32 = False
    var_33 = module_0.Definitions()
    var_34 = module_1.from_json_schema_type(var_31, var_27, var_32, var_33)
    var_35 = 'boolean'
    var_36 = True
    var_37 = {var_0: var_35, var_6: var_36}
    var_38 = False
    var_39 = module_0.Definitions()
    var_40 = module_1.from_json_schema_type(var_37, var_35, var_38, var_39)
    var_41 = 'items'
    var_42 = 'minItems'
    var_43 = 'maxItems'
    var_44 = 'uniqueItems'
    var_45 = 'array'
    var_46 = {var_0: var_27}
    var_47 = True
    var_48 = [var_30]
    var_49 = {var_0: var_45, var_41: var_46, var_42: var_36, var_43: var_9, var_44: var_47, var_6: var_48}
    var_50 = False
    var_51 = module_0.Definitions()
    var_52 = module_1.from_json_schema_type(var_49, var_45, var_50, var_51)
    var_53 = var_52.items
    var_54 = 'properties'
    var_55 = 'minProperties'
    var_56 = 'maxProperties'
    var_57 = 'required'
    var_58 = 'object'
    var_59 = 'name'
    var_60 = {var_0: var_27}
    var_61 = {var_59: var_60}
    var_62 = [var_59]
    var_63 = {var_59: var_30}
    var_64 = {var_0: var_58, var_54: var_61, var_55: var_47, var_56: var_9, var_57: var_62, var_6: var_63}
    var_65 = False
    var_66 = module_0.Definitions()
    var_67 = module_1.from_json_schema_type(var_64, var_58, var_65, var_66)
    var_68 = var_67.properties[var_59]



# Parsed testcases at query #25
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1
import re as module_2

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'type'
    var_2 = 'minLength'
    var_3 = 'maxLength'
    var_4 = 'pattern'
    var_5 = 'default'
    var_6 = 'string'
    var_7 = 1
    var_8 = 10
    var_9 = '^[a-z]+$'
    var_10 = 'test'
    var_11 = {var_1: var_6, var_2: var_7, var_3: var_8, var_4: var_9, var_5: var_10}
    var_12 = False
    var_13 = module_1.from_json_schema_type(var_11, var_6, var_12, var_0)
    var_14 = module_2.compile(var_9)
    var_15 = 'minimum'
    var_16 = 'maximum'
    var_17 = 'exclusiveMinimum'
    var_18 = 'exclusiveMaximum'
    var_19 = 'multipleOf'
    var_20 = 'number'
    var_21 = 11
    var_22 = 2
    var_23 = 4
    var_24 = {var_1: var_20, var_15: var_7, var_16: var_8, var_17: var_12, var_18: var_21, var_19: var_22, var_5: var_23}
    var_25 = module_1.from_json_schema_type(var_24, var_20, var_12, var_0)
    var_26 = 'integer'
    var_27 = {var_1: var_26, var_15: var_7, var_16: var_8, var_17: var_12, var_18: var_21, var_19: var_22, var_5: var_23}
    var_28 = module_1.from_json_schema_type(var_27, var_26, var_12, var_0)
    var_29 = 'boolean'
    var_30 = True
    var_31 = {var_1: var_29, var_5: var_30}
    var_32 = module_1.from_json_schema_type(var_31, var_29, var_12, var_0)
    var_33 = 'items'
    var_34 = 'minItems'
    var_35 = 'maxItems'
    var_36 = 'uniqueItems'
    var_37 = 'array'
    var_38 = {var_1: var_6}
    var_39 = True
    var_40 = [var_10]
    var_41 = {var_1: var_37, var_33: var_38, var_34: var_30, var_35: var_8, var_36: var_39, var_5: var_40}
    var_42 = module_1.from_json_schema_type(var_41, var_37, var_12, var_0)
    var_43 = var_42.items
    var_44 = 'properties'
    var_45 = 'required'
    var_46 = 'object'
    var_47 = 'name'
    var_48 = {var_1: var_6}
    var_49 = {var_47: var_48}
    var_50 = [var_47]
    var_51 = {var_47: var_10}
    var_52 = {var_1: var_46, var_44: var_49, var_45: var_50, var_5: var_51}
    var_53 = module_1.from_json_schema_type(var_52, var_46, var_12, var_0)
    var_54 = var_53.properties[var_47]



# Parsed testcases at query #26
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1
import re as module_2

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
    var_9 = 1
    var_10 = 10
    var_11 = 0.5
    var_12 = 10.5
    var_13 = 2
    var_14 = 5
    var_15 = {var_1: var_8, var_2: var_9, var_3: var_10, var_4: var_11, var_5: var_12, var_6: var_13, var_7: var_14}
    var_16 = False
    var_17 = module_1.from_json_schema_type(var_15, var_8, var_16, var_0)
    var_18 = 'integer'
    var_19 = 11
    var_20 = {var_1: var_18, var_2: var_9, var_3: var_10, var_4: var_16, var_5: var_19, var_6: var_13, var_7: var_14}
    var_21 = module_1.from_json_schema_type(var_20, var_18, var_16, var_0)
    var_22 = 'minLength'
    var_23 = 'maxLength'
    var_24 = 'format'
    var_25 = 'pattern'
    var_26 = 'string'
    var_27 = 'email'
    var_28 = '^[a-z]+$'
    var_29 = 'test'
    var_30 = {var_1: var_26, var_22: var_9, var_23: var_10, var_24: var_27, var_25: var_28, var_7: var_29}
    var_31 = module_1.from_json_schema_type(var_30, var_26, var_16, var_0)
    var_32 = module_2.compile(var_28)
    var_33 = 'boolean'
    var_34 = True
    var_35 = {var_1: var_33, var_7: var_34}
    var_36 = module_1.from_json_schema_type(var_35, var_33, var_16, var_0)
    var_37 = 'items'
    var_38 = 'minItems'
    var_39 = 'maxItems'
    var_40 = 'uniqueItems'
    var_41 = 'array'
    var_42 = {var_1: var_26}
    var_43 = True
    var_44 = [var_29]
    var_45 = {var_1: var_41, var_37: var_42, var_38: var_34, var_39: var_10, var_40: var_43, var_7: var_44}
    var_46 = module_1.from_json_schema_type(var_45, var_41, var_16, var_0)
    var_47 = var_46.items
    var_48 = 'properties'
    var_49 = 'minProperties'
    var_50 = 'maxProperties'
    var_51 = 'required'
    var_52 = 'object'
    var_53 = 'name'
    var_54 = {var_1: var_26}
    var_55 = {var_53: var_54}
    var_56 = [var_53]
    var_57 = {var_53: var_29}
    var_58 = {var_1: var_52, var_48: var_55, var_49: var_43, var_50: var_10, var_51: var_56, var_7: var_57}
    var_59 = module_1.from_json_schema_type(var_58, var_52, var_16, var_0)
    var_60 = var_59.properties[var_53]



# Parsed testcases at query #27
#--------------------------


import typesystem.fields as module_0
import typesystem.json_schema as module_1
import typesystem.composites as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True
    var_2 = module_2.NeverMatch()
    var_3 = module_1.to_json_schema(var_2)
    assert var_3 is False
    var_4 = True
    var_5 = 10
    var_6 = '^[a-zA-Z]+$'
    var_7 = 'email'
    var_8 = module_0.String(max_length=var_5, min_length=var_4, pattern=var_6, format=var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = 0
    var_11 = 11
    var_12 = 2
    var_13 = module_0.Integer(minimum=var_4, maximum=var_5, exclusive_minimum=var_10, exclusive_maximum=var_11, multiple_of=var_12)
    var_14 = module_1.to_json_schema(var_13)
    var_15 = module_0.Boolean()
    var_16 = module_1.to_json_schema(var_15)
    var_17 = module_0.String()
    var_18 = module_0.Array(var_17, var_4, var_4, var_5, unique_items=var_4)
    var_19 = module_1.to_json_schema(var_18)
    var_20 = 'name'
    var_21 = module_0.String()
    var_22 = {var_20: var_21}
    var_23 = '^[a-z]+$'
    var_24 = module_0.Integer()
    var_25 = {var_23: var_24}
    var_26 = module_0.String()
    var_27 = [var_20]
    var_28 = module_0.Object(properties=var_22, pattern_properties=var_25, additional_properties=var_4, property_names=var_26, min_properties=var_4, max_properties=var_5, required=var_27)
    var_29 = module_1.to_json_schema(var_28)
    var_30 = module_0.String()
    var_31 = {var_20: var_30}
    var_32 = [var_20]
    var_33 = module_3.Schema(var_31)
    var_34 = module_1.to_json_schema(var_33)
    var_35 = 'a'
    var_36 = 'A'
    var_37 = (var_35, var_36)
    var_38 = 'b'
    var_39 = 'B'
    var_40 = (var_38, var_39)
    var_41 = [var_37, var_40]
    var_42 = module_0.Choice(choices=var_41)
    var_43 = module_1.to_json_schema(var_42)
    var_44 = 'constant'
    var_45 = module_0.Const(var_44)
    var_46 = module_1.to_json_schema(var_45)
    var_47 = module_0.String()
    var_48 = module_0.Integer()
    var_49 = [var_47, var_48]
    var_50 = module_0.Union(var_49)
    var_51 = module_1.to_json_schema(var_50)
    var_52 = module_0.String()
    var_53 = module_0.Integer()
    var_54 = [var_52, var_53]
    var_55 = module_2.OneOf(var_54)
    var_56 = module_1.to_json_schema(var_55)
    var_57 = module_0.String()
    var_58 = module_0.Integer()
    var_59 = [var_57, var_58]
    var_60 = module_2.AllOf(var_59)
    var_61 = module_1.to_json_schema(var_60)
    var_62 = module_0.String()
    var_63 = module_0.Integer()
    var_64 = module_0.Boolean()
    var_65 = module_2.IfThenElse(var_62, var_63, var_64)
    var_66 = module_1.to_json_schema(var_65)
    var_67 = module_0.String()
    var_68 = module_2.Not(var_67)
    var_69 = module_1.to_json_schema(var_68)
    var_70 = 'MySchema'
    var_71 = module_0.String()
    var_72 = {var_20: var_71}
    var_73 = module_3.Schema(var_72)
    var_74 = {var_70: var_73}



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.fields as module_0
import typesystem.json_schema as module_1
import typesystem.composites as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True
    var_2 = module_2.NeverMatch()
    var_3 = module_1.to_json_schema(var_2)
    assert var_3 is False
    var_4 = True
    var_5 = module_0.String()
    var_6 = module_1.to_json_schema(var_5)
    var_7 = module_0.Integer()
    var_8 = module_1.to_json_schema(var_7)
    var_9 = module_0.Float()
    var_10 = module_1.to_json_schema(var_9)
    var_11 = module_0.Boolean()
    var_12 = module_1.to_json_schema(var_11)
    var_13 = module_0.Array()
    var_14 = module_1.to_json_schema(var_13)
    var_15 = module_0.Object()
    var_16 = module_1.to_json_schema(var_15)
    var_17 = module_3.Schema()
    var_18 = module_1.to_json_schema(var_17)
    var_19 = 'a'
    var_20 = (var_19, var_19)
    var_21 = [var_20]
    var_22 = module_0.Choice(choices=var_21)
    var_23 = module_1.to_json_schema(var_22)
    var_24 = module_0.Const(var_19)
    var_25 = module_1.to_json_schema(var_24)
    var_26 = module_0.String()
    var_27 = module_0.Integer()
    var_28 = [var_26, var_27]
    var_29 = module_0.Union(var_28)
    var_30 = module_1.to_json_schema(var_29)
    var_31 = module_0.String()
    var_32 = module_0.Integer()
    var_33 = [var_31, var_32]
    var_34 = module_2.OneOf(var_33)
    var_35 = module_1.to_json_schema(var_34)
    var_36 = module_0.String()
    var_37 = module_0.Integer()
    var_38 = [var_36, var_37]
    var_39 = module_2.AllOf(var_38)
    var_40 = module_1.to_json_schema(var_39)
    var_41 = module_0.String()
    var_42 = module_0.Integer()
    var_43 = module_2.IfThenElse(var_41, var_42)
    var_44 = module_1.to_json_schema(var_43)
    var_45 = module_0.String()
    var_46 = module_2.Not(var_45)
    var_47 = module_1.to_json_schema(var_46)
    var_48 = module_0.String()
    var_49 = {var_19: var_48}
    var_50 = module_3.Reference(var_19, var_49)
    var_51 = module_1.to_json_schema(var_50)
    var_52 = module_0.String()
    var_53 = {var_19: var_52}



# Parsed testcases at query #2
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = '#/components/schemas/User'
    var_3 = {var_1: var_2}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)



# Parsed testcases at query #3
#--------------------------


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
    var_5 = 'null'
    var_6 = [var_2, var_5]
    var_7 = {var_1: var_6}
    var_8 = module_1.type_from_json_schema(var_7, var_0)
    var_9 = []
    var_10 = {var_1: var_9}
    var_11 = module_1.type_from_json_schema(var_10, var_0)
    var_12 = [var_5]
    var_13 = {var_1: var_12}
    var_14 = module_1.type_from_json_schema(var_13, var_0)
    var_15 = 'invalid'
    var_16 = {var_1: var_15}
    var_17 = module_1.type_from_json_schema(var_16, var_0)



# Parsed testcases at query #5
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
    var_11 = 'number'
    var_12 = {var_4: var_11}
    var_13 = module_0.from_json_schema(var_12)
    var_14 = 'boolean'
    var_15 = {var_4: var_14}
    var_16 = module_0.from_json_schema(var_15)
    var_17 = 'items'
    var_18 = 'array'
    var_19 = {var_4: var_5}
    var_20 = {var_4: var_18, var_17: var_19}
    var_21 = module_0.from_json_schema(var_20)
    var_22 = 'properties'
    var_23 = 'object'
    var_24 = 'name'
    var_25 = {var_4: var_5}
    var_26 = {var_24: var_25}
    var_27 = {var_4: var_23, var_22: var_26}
    var_28 = module_0.from_json_schema(var_27)
    var_29 = '$ref'
    var_30 = '#/components/schemas/Person'
    var_31 = {var_29: var_30}
    var_32 = module_0.from_json_schema(var_31)
    var_33 = 'enum'
    var_34 = 'red'
    var_35 = 'green'
    var_36 = 'blue'
    var_37 = [var_34, var_35, var_36]
    var_38 = {var_33: var_37}
    var_39 = module_0.from_json_schema(var_38)
    var_40 = 'const'
    var_41 = {var_40: var_34}
    var_42 = module_0.from_json_schema(var_41)
    var_43 = 'allOf'
    var_44 = {var_4: var_5}
    var_45 = 'minLength'
    var_46 = 5
    var_47 = {var_45: var_46}
    var_48 = [var_44, var_47]
    var_49 = {var_43: var_48}
    var_50 = module_0.from_json_schema(var_49)
    var_51 = 'anyOf'
    var_52 = {var_4: var_5}
    var_53 = {var_4: var_11}
    var_54 = [var_52, var_53]
    var_55 = {var_51: var_54}
    var_56 = module_0.from_json_schema(var_55)
    var_57 = 'oneOf'
    var_58 = {var_4: var_5}
    var_59 = {var_4: var_11}
    var_60 = [var_58, var_59]
    var_61 = {var_57: var_60}
    var_62 = module_0.from_json_schema(var_61)
    var_63 = 'not'
    var_64 = {var_4: var_5}
    var_65 = {var_63: var_64}
    var_66 = module_0.from_json_schema(var_65)
    var_67 = 'if'
    var_68 = 'then'
    var_69 = 'else'
    var_70 = {var_4: var_5}
    var_71 = {var_45: var_46}
    var_72 = {var_4: var_11}
    var_73 = {var_67: var_70, var_68: var_71, var_69: var_72}
    var_74 = module_0.from_json_schema(var_73)



# Parsed testcases at query #6
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '#/components/schemas/test_schema'
    var_2 = '$ref'
    var_3 = {var_2: var_1}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)



# Parsed testcases at query #7
#--------------------------


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
    var_16 = {var_3: var_4}
    var_17 = {var_3: var_6}
    var_18 = {var_0: var_16, var_1: var_17}
    var_19 = module_0.Definitions()
    var_20 = module_1.if_then_else_from_json_schema(var_18, var_19)
    var_21 = var_20.if_clause
    var_22 = var_20.then_clause
    var_23 = {var_3: var_4}
    var_24 = {var_3: var_8}
    var_25 = {var_0: var_23, var_2: var_24}
    var_26 = module_0.Definitions()
    var_27 = module_1.if_then_else_from_json_schema(var_25, var_26)
    var_28 = var_27.if_clause
    var_29 = var_27.else_clause



# Parsed testcases at query #8
#--------------------------


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
    var_7 = 'number'
    var_8 = {var_4: var_7}
    var_9 = 'boolean'
    var_10 = {var_4: var_9}
    var_11 = {var_1: var_6, var_2: var_8, var_3: var_10}
    var_12 = module_1.if_then_else_from_json_schema(var_11, var_0)
    var_13 = var_12.if_clause
    var_14 = var_12.then_clause
    var_15 = var_12.else_clause



# Parsed testcases at query #9
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = '#/components/schemas/User'
    var_3 = {var_1: var_2}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)
    var_5 = '#/definitions/User'
    var_6 = {var_1: var_5}
    var_7 = module_1.ref_from_json_schema(var_6, var_0)
    var_8 = '#/invalid/path'
    var_9 = {var_1: var_8}
    var_10 = module_1.ref_from_json_schema(var_9, var_0)



# Parsed testcases at query #10
#--------------------------


import typesystem.fields as module_0
import typesystem.json_schema as module_1
import typesystem.composites as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True
    var_2 = module_2.NeverMatch()
    var_3 = module_1.to_json_schema(var_2)
    assert var_3 is False
    var_4 = True
    var_5 = 10
    var_6 = '^[a-z]+$'
    var_7 = 'email'
    var_8 = module_0.String(max_length=var_5, min_length=var_4, pattern=var_6, format=var_7)
    var_9 = 'type'
    var_10 = 'minLength'
    var_11 = 'maxLength'
    var_12 = 'pattern'
    var_13 = 'format'
    var_14 = 'string'
    var_15 = 'null'
    var_16 = [var_14, var_15]
    var_17 = {var_9: var_16, var_10: var_4, var_11: var_5, var_12: var_6, var_13: var_7}
    var_18 = module_1.to_json_schema(var_8)
    var_19 = 0
    var_20 = 100
    var_21 = 5
    var_22 = module_0.Integer(minimum=var_19, maximum=var_20, multiple_of=var_21)
    var_23 = 'minimum'
    var_24 = 'maximum'
    var_25 = 'multipleOf'
    var_26 = 'integer'
    var_27 = [var_26, var_15]
    var_28 = {var_9: var_27, var_23: var_19, var_24: var_20, var_25: var_21}
    var_29 = module_1.to_json_schema(var_22)
    var_30 = module_0.Boolean()
    var_31 = 'boolean'
    var_32 = [var_31, var_15]
    var_33 = {var_9: var_32}
    var_34 = module_1.to_json_schema(var_30)
    var_35 = module_0.String()
    var_36 = False
    var_37 = module_0.Array(var_35, var_36, var_4, var_5, unique_items=var_4)
    var_38 = 'minItems'
    var_39 = 'maxItems'
    var_40 = 'items'
    var_41 = 'additionalItems'
    var_42 = 'uniqueItems'
    var_43 = 'array'
    var_44 = [var_43, var_15]
    var_45 = {var_9: var_14}
    var_46 = False
    var_47 = {var_9: var_44, var_38: var_4, var_39: var_5, var_40: var_45, var_41: var_46, var_42: var_4}
    var_48 = module_1.to_json_schema(var_37)
    var_49 = 'name'
    var_50 = module_0.String()
    var_51 = {var_49: var_50}
    var_52 = [var_49]
    var_53 = False
    var_54 = module_0.Object(properties=var_51, additional_properties=var_53, required=var_52)
    var_55 = 'properties'
    var_56 = 'required'
    var_57 = 'additionalProperties'
    var_58 = 'object'
    var_59 = [var_58, var_15]
    var_60 = {var_9: var_14}
    var_61 = {var_49: var_60}
    var_62 = [var_49]
    var_63 = False
    var_64 = {var_9: var_59, var_55: var_61, var_56: var_62, var_57: var_63}
    var_65 = module_1.to_json_schema(var_54)
    var_66 = module_3.Definitions()
    var_67 = module_0.String()
    var_68 = {var_49: var_67}
    var_69 = 'Person'
    var_70 = module_3.Reference(var_69, var_66)
    var_71 = '$ref'
    var_72 = 'components'
    var_73 = '#/components/schemas/Person'
    var_74 = 'schemas'
    var_75 = {var_9: var_14}
    var_76 = {var_49: var_75}
    var_77 = {var_9: var_58, var_55: var_76}
    var_78 = {var_69: var_77}
    var_79 = {var_74: var_78}
    var_80 = {var_71: var_73, var_72: var_79}
    var_81 = module_1.to_json_schema(var_70)
    var_82 = 'All tests passed!'
    var_83 = print(var_82)



# Parsed testcases at query #11
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = '#/components/schemas/User'
    var_3 = {var_1: var_2}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)



# Parsed testcases at query #12
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'type'
    var_2 = 'minLength'
    var_3 = 'maxLength'
    var_4 = 'string'
    var_5 = 5
    var_6 = 10
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = False
    var_9 = module_1.from_json_schema_type(var_7, var_4, var_8, var_0)
    var_10 = 'minimum'
    var_11 = 'maximum'
    var_12 = 'integer'
    var_13 = 1
    var_14 = 100
    var_15 = {var_1: var_12, var_10: var_13, var_11: var_14}
    var_16 = module_1.from_json_schema_type(var_15, var_12, var_8, var_0)
    var_17 = 'number'
    var_18 = {var_1: var_17, var_10: var_13, var_11: var_14}
    var_19 = module_1.from_json_schema_type(var_18, var_17, var_8, var_0)
    var_20 = 'boolean'
    var_21 = {var_1: var_20}
    var_22 = module_1.from_json_schema_type(var_21, var_20, var_8, var_0)
    var_23 = 'items'
    var_24 = 'minItems'
    var_25 = 'maxItems'
    var_26 = 'array'
    var_27 = {var_1: var_4}
    var_28 = {var_1: var_26, var_23: var_27, var_24: var_13, var_25: var_6}
    var_29 = module_1.from_json_schema_type(var_28, var_26, var_8, var_0)
    var_30 = var_29.items
    var_31 = 'properties'
    var_32 = 'object'
    var_33 = 'name'
    var_34 = {var_1: var_4}
    var_35 = {var_33: var_34}
    var_36 = {var_1: var_32, var_31: var_35}
    var_37 = module_1.from_json_schema_type(var_36, var_32, var_8, var_0)
    var_38 = var_37.properties[var_33]



# Parsed testcases at query #13
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = '#/components/schemas/User'
    var_3 = {var_1: var_2}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)
    var_5 = '$ref'
    var_6 = 'http://example.com/schema#/User'
    var_7 = {var_5: var_6}
    var_8 = module_1.ref_from_json_schema(var_7, var_0)



# Parsed testcases at query #14
#--------------------------


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
    var_4 = module_0.String()
    var_5 = module_1.to_json_schema(var_4)
    var_6 = module_0.Integer()
    var_7 = module_1.to_json_schema(var_6)
    var_8 = module_0.Float()
    var_9 = module_1.to_json_schema(var_8)
    var_10 = module_0.Boolean()
    var_11 = module_1.to_json_schema(var_10)
    var_12 = module_0.String()
    var_13 = module_0.Array(var_12)
    var_14 = module_1.to_json_schema(var_13)
    var_15 = 'name'
    var_16 = module_0.String()
    var_17 = {var_15: var_16}
    var_18 = module_0.Object(properties=var_17)
    var_19 = module_1.to_json_schema(var_18)
    var_20 = 'a'
    var_21 = (var_20, var_20)
    var_22 = [var_21]
    var_23 = module_0.Choice(choices=var_22)
    var_24 = module_1.to_json_schema(var_23)
    var_25 = module_0.Const(var_20)
    var_26 = module_1.to_json_schema(var_25)
    var_27 = module_0.String()
    var_28 = module_0.Integer()
    var_29 = [var_27, var_28]
    var_30 = module_0.Union(var_29)
    var_31 = module_1.to_json_schema(var_30)
    var_32 = module_0.String()
    var_33 = module_0.Integer()
    var_34 = [var_32, var_33]
    var_35 = module_2.OneOf(var_34)
    var_36 = module_1.to_json_schema(var_35)
    var_37 = module_0.String()
    var_38 = module_0.Integer()
    var_39 = [var_37, var_38]
    var_40 = module_2.AllOf(var_39)
    var_41 = module_1.to_json_schema(var_40)
    var_42 = module_0.String()
    var_43 = module_0.Integer()
    var_44 = module_2.IfThenElse(var_42, var_43)
    var_45 = module_1.to_json_schema(var_44)
    var_46 = module_0.String()
    var_47 = module_2.Not(var_46)
    var_48 = module_1.to_json_schema(var_47)



# Parsed testcases at query #15
#--------------------------


import typesystem.json_schema as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)
    var_2 = False
    var_3 = module_0.from_json_schema(var_2)
    var_4 = '$ref'
    var_5 = '#/components/schemas/Example'
    var_6 = {var_4: var_5}
    var_7 = module_0.from_json_schema(var_6)
    var_8 = 'type'
    var_9 = 'minLength'
    var_10 = 'string'
    var_11 = 5
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = module_0.from_json_schema(var_12)
    var_14 = 'enum'
    var_15 = 'red'
    var_16 = 'green'
    var_17 = 'blue'
    var_18 = [var_15, var_16, var_17]
    var_19 = {var_14: var_18}
    var_20 = module_0.from_json_schema(var_19)
    var_21 = 'const'
    var_22 = 'example'
    var_23 = {var_21: var_22}
    var_24 = module_0.from_json_schema(var_23)
    var_25 = 'allOf'
    var_26 = {var_8: var_10}
    var_27 = 3
    var_28 = {var_9: var_27}
    var_29 = [var_26, var_28]
    var_30 = {var_25: var_29}
    var_31 = module_0.from_json_schema(var_30)
    var_32 = 'anyOf'
    var_33 = {var_8: var_10}
    var_34 = 'number'
    var_35 = {var_8: var_34}
    var_36 = [var_33, var_35]
    var_37 = {var_32: var_36}
    var_38 = module_0.from_json_schema(var_37)
    var_39 = 'oneOf'
    var_40 = {var_8: var_10}
    var_41 = {var_8: var_34}
    var_42 = [var_40, var_41]
    var_43 = {var_39: var_42}
    var_44 = module_0.from_json_schema(var_43)
    var_45 = 'not'
    var_46 = {var_8: var_10}
    var_47 = {var_45: var_46}
    var_48 = module_0.from_json_schema(var_47)
    var_49 = 'if'
    var_50 = 'then'
    var_51 = 'else'
    var_52 = {var_8: var_10}
    var_53 = {var_9: var_11}
    var_54 = {var_8: var_34}
    var_55 = {var_49: var_52, var_50: var_53, var_51: var_54}
    var_56 = module_0.from_json_schema(var_55)
    var_57 = 'hello'
    var_58 = 'world'
    var_59 = [var_57, var_58]
    var_60 = {var_8: var_10, var_9: var_11, var_14: var_59}
    var_61 = module_0.from_json_schema(var_60)



# Parsed testcases at query #16
#--------------------------


import typesystem.fields as module_0
import typesystem.json_schema as module_1
import typesystem.composites as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = True
    var_1 = module_0.Any()
    var_2 = module_1.to_json_schema(var_1)
    var_3 = False
    var_4 = module_2.NeverMatch()
    var_5 = module_1.to_json_schema(var_4)
    var_6 = 'type'
    var_7 = 'minLength'
    var_8 = 'string'
    var_9 = {var_6: var_8, var_7: var_0}
    var_10 = module_0.String(min_length=var_0)
    var_11 = module_1.to_json_schema(var_10)
    var_12 = 'minimum'
    var_13 = 'maximum'
    var_14 = 'number'
    var_15 = 100
    var_16 = {var_6: var_14, var_12: var_3, var_13: var_15}
    var_17 = module_0.Float(minimum=var_3, maximum=var_15)
    var_18 = module_1.to_json_schema(var_17)
    var_19 = 'integer'
    var_20 = {var_6: var_19, var_12: var_3, var_13: var_15}
    var_21 = module_0.Integer(minimum=var_3, maximum=var_15)
    var_22 = module_1.to_json_schema(var_21)
    var_23 = 'boolean'
    var_24 = {var_6: var_23}
    var_25 = module_0.Boolean()
    var_26 = module_1.to_json_schema(var_25)
    var_27 = 'items'
    var_28 = 'array'
    var_29 = {var_6: var_8}
    var_30 = {var_6: var_28, var_27: var_29}
    var_31 = module_0.String()
    var_32 = module_0.Array(var_31)
    var_33 = module_1.to_json_schema(var_32)
    var_34 = 'properties'
    var_35 = 'object'
    var_36 = 'name'
    var_37 = {var_6: var_8}
    var_38 = {var_36: var_37}
    var_39 = {var_6: var_35, var_34: var_38}
    var_40 = module_0.String()
    var_41 = {var_36: var_40}
    var_42 = module_0.Object(properties=var_41)
    var_43 = module_1.to_json_schema(var_42)
    var_44 = 'enum'
    var_45 = 'red'
    var_46 = 'green'
    var_47 = 'blue'
    var_48 = [var_45, var_46, var_47]
    var_49 = {var_44: var_48}
    var_50 = (var_45, var_45)
    var_51 = (var_46, var_46)
    var_52 = (var_47, var_47)
    var_53 = [var_50, var_51, var_52]
    var_54 = module_0.Choice(choices=var_53)
    var_55 = module_1.to_json_schema(var_54)
    var_56 = 'const'
    var_57 = {var_56: var_45}
    var_58 = module_0.Const(var_45)
    var_59 = module_1.to_json_schema(var_58)
    var_60 = 'anyOf'
    var_61 = {var_6: var_8}
    var_62 = {var_6: var_14}
    var_63 = [var_61, var_62]
    var_64 = {var_60: var_63}
    var_65 = module_0.String()
    var_66 = module_0.Float()
    var_67 = [var_65, var_66]
    var_68 = module_0.Union(var_67)
    var_69 = module_1.to_json_schema(var_68)
    var_70 = 'oneOf'
    var_71 = {var_6: var_8}
    var_72 = {var_6: var_14}
    var_73 = [var_71, var_72]
    var_74 = {var_70: var_73}
    var_75 = module_0.String()
    var_76 = module_0.Float()
    var_77 = [var_75, var_76]
    var_78 = module_2.OneOf(var_77)
    var_79 = module_1.to_json_schema(var_78)
    var_80 = 'allOf'
    var_81 = {var_6: var_8}
    var_82 = {var_6: var_14}
    var_83 = [var_81, var_82]
    var_84 = {var_80: var_83}
    var_85 = module_0.String()
    var_86 = module_0.Float()
    var_87 = [var_85, var_86]
    var_88 = module_2.AllOf(var_87)
    var_89 = module_1.to_json_schema(var_88)
    var_90 = 'if'
    var_91 = 'then'
    var_92 = {var_6: var_8}
    var_93 = {var_6: var_14}
    var_94 = {var_90: var_92, var_91: var_93}
    var_95 = module_0.String()
    var_96 = module_0.Float()
    var_97 = module_2.IfThenElse(var_95, var_96)
    var_98 = module_1.to_json_schema(var_97)
    var_99 = 'not'
    var_100 = {var_6: var_8}
    var_101 = {var_99: var_100}
    var_102 = module_0.String()
    var_103 = module_2.Not(var_102)
    var_104 = module_1.to_json_schema(var_103)
    var_105 = 'components'
    var_106 = 'schemas'
    var_107 = 'Person'
    var_108 = {var_6: var_8}
    var_109 = {var_36: var_108}
    var_110 = {var_6: var_35, var_34: var_109}
    var_111 = {var_107: var_110}
    var_112 = {var_106: var_111}
    var_113 = {var_105: var_112}
    var_114 = module_0.String()
    var_115 = {var_36: var_114}
    var_116 = module_0.Object(properties=var_115)
    var_117 = {var_107: var_116}
    var_118 = '$ref'
    var_119 = '#/components/schemas/Person'
    var_120 = {var_6: var_8}
    var_121 = {var_36: var_120}
    var_122 = {var_6: var_35, var_34: var_121}
    var_123 = {var_107: var_122}
    var_124 = {var_106: var_123}
    var_125 = {var_118: var_119, var_105: var_124}
    var_126 = module_0.String()
    var_127 = {var_36: var_126}
    var_128 = module_0.Object(properties=var_127)
    var_129 = module_3.Reference(var_107)
    var_130 = module_1.to_json_schema(var_129)



# Parsed testcases at query #17
#--------------------------


import typesystem.fields as module_0
import typesystem.json_schema as module_1
import typesystem.composites as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True
    var_2 = module_2.NeverMatch()
    var_3 = module_1.to_json_schema(var_2)
    assert var_3 is False
    var_4 = True
    var_5 = 10
    var_6 = '^[a-z]*$'
    var_7 = 'email'
    var_8 = module_0.String(max_length=var_5, min_length=var_4, pattern=var_6, format=var_7)
    var_9 = 'type'
    var_10 = 'minLength'
    var_11 = 'maxLength'
    var_12 = 'pattern'
    var_13 = 'format'
    var_14 = 'string'
    var_15 = 'null'
    var_16 = [var_14, var_15]
    var_17 = {var_9: var_16, var_10: var_4, var_11: var_5, var_12: var_6, var_13: var_7}
    var_18 = module_1.to_json_schema(var_8)
    var_19 = 100
    var_20 = 2
    var_21 = module_0.Integer(minimum=var_4, maximum=var_19, multiple_of=var_20)
    var_22 = 'minimum'
    var_23 = 'maximum'
    var_24 = 'multipleOf'
    var_25 = 'integer'
    var_26 = [var_25, var_15]
    var_27 = {var_9: var_26, var_22: var_4, var_23: var_19, var_24: var_20}
    var_28 = module_1.to_json_schema(var_21)
    var_29 = module_0.Boolean()
    var_30 = 'boolean'
    var_31 = [var_30, var_15]
    var_32 = {var_9: var_31}
    var_33 = module_1.to_json_schema(var_29)
    var_34 = module_0.String()
    var_35 = False
    var_36 = module_0.Array(var_34, var_35, var_4, var_5, unique_items=var_4)
    var_37 = 'minItems'
    var_38 = 'maxItems'
    var_39 = 'items'
    var_40 = 'additionalItems'
    var_41 = 'uniqueItems'
    var_42 = 'array'
    var_43 = [var_42, var_15]
    var_44 = {var_9: var_14}
    var_45 = {var_9: var_43, var_37: var_4, var_38: var_5, var_39: var_44, var_40: var_35, var_41: var_4}
    var_46 = module_1.to_json_schema(var_36)
    var_47 = 'name'
    var_48 = module_0.String()
    var_49 = {var_47: var_48}
    var_50 = [var_47]
    var_51 = module_0.Object(properties=var_49, min_properties=var_4, max_properties=var_20, required=var_50)
    var_52 = 'properties'
    var_53 = 'required'
    var_54 = 'minProperties'
    var_55 = 'maxProperties'
    var_56 = 'object'
    var_57 = [var_56, var_15]
    var_58 = {var_9: var_14}
    var_59 = {var_47: var_58}
    var_60 = [var_47]
    var_61 = {var_9: var_57, var_52: var_59, var_53: var_60, var_54: var_4, var_55: var_20}
    var_62 = module_1.to_json_schema(var_51)
    var_63 = 'a'
    var_64 = 'A'
    var_65 = (var_63, var_64)
    var_66 = 'b'
    var_67 = 'B'
    var_68 = (var_66, var_67)
    var_69 = [var_65, var_68]
    var_70 = module_0.Choice(choices=var_69)
    var_71 = 'enum'
    var_72 = [var_63, var_66]
    var_73 = {var_71: var_72}
    var_74 = module_1.to_json_schema(var_70)
    var_75 = 'test'
    var_76 = module_0.Const(var_75)
    var_77 = 'const'
    var_78 = {var_77: var_75}
    var_79 = module_1.to_json_schema(var_76)
    var_80 = module_0.String()
    var_81 = module_0.Integer()
    var_82 = [var_80, var_81]
    var_83 = module_0.Union(var_82)
    var_84 = 'anyOf'
    var_85 = {var_9: var_14}
    var_86 = {var_9: var_25}
    var_87 = [var_85, var_86]
    var_88 = {var_84: var_87}
    var_89 = module_1.to_json_schema(var_83)
    var_90 = module_0.String(min_length=var_4)
    var_91 = module_0.String(max_length=var_5)
    var_92 = [var_90, var_91]
    var_93 = module_2.AllOf(var_92)
    var_94 = 'allOf'
    var_95 = {var_9: var_14, var_10: var_4}
    var_96 = {var_9: var_14, var_11: var_5}
    var_97 = [var_95, var_96]
    var_98 = {var_94: var_97}
    var_99 = module_1.to_json_schema(var_93)
    var_100 = module_0.String()
    var_101 = module_0.Integer()
    var_102 = module_0.Boolean()
    var_103 = module_2.IfThenElse(var_100, var_101, var_102)
    var_104 = 'if'
    var_105 = 'then'
    var_106 = 'else'
    var_107 = {var_9: var_14}
    var_108 = {var_9: var_25}
    var_109 = {var_9: var_30}
    var_110 = {var_104: var_107, var_105: var_108, var_106: var_109}
    var_111 = module_1.to_json_schema(var_103)
    var_112 = module_0.String()
    var_113 = module_2.Not(var_112)
    var_114 = 'not'
    var_115 = {var_9: var_14}
    var_116 = {var_114: var_115}
    var_117 = module_1.to_json_schema(var_113)
    var_118 = 'Test'
    var_119 = module_0.String()
    var_120 = {var_118: var_119}
    var_121 = module_3.Reference(var_118, var_120)
    var_122 = '$ref'
    var_123 = 'components'
    var_124 = '#/components/schemas/Test'
    var_125 = 'schemas'
    var_126 = {var_9: var_14}
    var_127 = {var_118: var_126}
    var_128 = {var_125: var_127}
    var_129 = {var_122: var_124, var_123: var_128}
    var_130 = module_1.to_json_schema(var_121, var_120)
    var_131 = 'All tests passed!'
    var_132 = print(var_131)



# Parsed testcases at query #18
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'type'
    var_2 = 'minLength'
    var_3 = 'maxLength'
    var_4 = 'pattern'
    var_5 = 'format'
    var_6 = 'default'
    var_7 = 'string'
    var_8 = 5
    var_9 = 10
    var_10 = '^[a-zA-Z]+$'
    var_11 = 'email'
    var_12 = 'example@example.com'
    var_13 = {var_1: var_7, var_2: var_8, var_3: var_9, var_4: var_10, var_5: var_11, var_6: var_12}
    var_14 = False
    var_15 = module_1.from_json_schema_type(var_13, var_7, var_14, var_0)
    var_16 = 'minimum'
    var_17 = 'maximum'
    var_18 = 'exclusiveMinimum'
    var_19 = 'exclusiveMaximum'
    var_20 = 'multipleOf'
    var_21 = 'integer'
    var_22 = 1
    var_23 = 100
    var_24 = 101
    var_25 = 2
    var_26 = 50
    var_27 = {var_1: var_21, var_16: var_22, var_17: var_23, var_18: var_14, var_19: var_24, var_20: var_25, var_6: var_26}
    var_28 = module_1.from_json_schema_type(var_27, var_21, var_14, var_0)
    var_29 = 'boolean'
    var_30 = True
    var_31 = {var_1: var_29, var_6: var_30}
    var_32 = module_1.from_json_schema_type(var_31, var_29, var_14, var_0)
    var_33 = 'items'
    var_34 = 'minItems'
    var_35 = 'maxItems'
    var_36 = 'uniqueItems'
    var_37 = 'array'
    var_38 = {var_1: var_7}
    var_39 = True
    var_40 = 'item1'
    var_41 = [var_40]
    var_42 = {var_1: var_37, var_33: var_38, var_34: var_30, var_35: var_8, var_36: var_39, var_6: var_41}
    var_43 = module_1.from_json_schema_type(var_42, var_37, var_14, var_0)
    var_44 = 'properties'
    var_45 = 'minProperties'
    var_46 = 'maxProperties'
    var_47 = 'required'
    var_48 = 'object'
    var_49 = 'name'
    var_50 = 'age'
    var_51 = {var_1: var_7}
    var_52 = {var_1: var_21}
    var_53 = {var_49: var_51, var_50: var_52}
    var_54 = [var_49]
    var_55 = 'John'
    var_56 = 30
    var_57 = {var_49: var_55, var_50: var_56}
    var_58 = {var_1: var_48, var_44: var_53, var_45: var_39, var_46: var_25, var_47: var_54, var_6: var_57}
    var_59 = module_1.from_json_schema_type(var_58, var_48, var_14, var_0)



# Parsed testcases at query #19
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1
import re as module_2

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
    var_9 = 10
    var_10 = 2
    var_11 = 4
    var_12 = {var_0: var_7, var_1: var_8, var_2: var_9, var_3: var_8, var_4: var_9, var_5: var_10, var_6: var_11}
    var_13 = False
    var_14 = module_0.Definitions()
    var_15 = module_1.from_json_schema_type(var_12, var_7, var_13, var_14)
    var_16 = 'integer'
    var_17 = {var_0: var_16, var_1: var_13, var_2: var_9, var_3: var_13, var_4: var_9, var_5: var_10, var_6: var_11}
    var_18 = False
    var_19 = module_0.Definitions()
    var_20 = module_1.from_json_schema_type(var_17, var_16, var_18, var_19)
    var_21 = 'minLength'
    var_22 = 'maxLength'
    var_23 = 'format'
    var_24 = 'pattern'
    var_25 = 'string'
    var_26 = 1
    var_27 = 'email'
    var_28 = '^[a-z]+$'
    var_29 = 'test'
    var_30 = {var_0: var_25, var_21: var_26, var_22: var_9, var_23: var_27, var_24: var_28, var_6: var_29}
    var_31 = False
    var_32 = module_0.Definitions()
    var_33 = module_1.from_json_schema_type(var_30, var_25, var_31, var_32)
    var_34 = module_2.compile(var_28)
    var_35 = 'boolean'
    var_36 = True
    var_37 = {var_0: var_35, var_6: var_36}
    var_38 = False
    var_39 = module_0.Definitions()
    var_40 = module_1.from_json_schema_type(var_37, var_35, var_38, var_39)
    var_41 = 'items'
    var_42 = 'minItems'
    var_43 = 'maxItems'
    var_44 = 'uniqueItems'
    var_45 = 'array'
    var_46 = {var_0: var_25}
    var_47 = True
    var_48 = [var_29]
    var_49 = {var_0: var_45, var_41: var_46, var_42: var_36, var_43: var_9, var_44: var_47, var_6: var_48}
    var_50 = False
    var_51 = module_0.Definitions()
    var_52 = module_1.from_json_schema_type(var_49, var_45, var_50, var_51)
    var_53 = var_52.items
    var_54 = 'properties'
    var_55 = 'minProperties'
    var_56 = 'maxProperties'
    var_57 = 'required'
    var_58 = 'object'
    var_59 = 'name'
    var_60 = {var_0: var_25}
    var_61 = {var_59: var_60}
    var_62 = [var_59]
    var_63 = {var_59: var_29}
    var_64 = {var_0: var_58, var_54: var_61, var_55: var_47, var_56: var_9, var_57: var_62, var_6: var_63}
    var_65 = False
    var_66 = module_0.Definitions()
    var_67 = module_1.from_json_schema_type(var_64, var_58, var_65, var_66)
    var_68 = var_67.properties[var_59]



# Parsed testcases at query #20
#--------------------------


import typesystem.fields as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.String()
    var_2 = module_1.to_json_schema(var_1)



# Parsed testcases at query #21
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
    var_11 = 2
    var_12 = 5
    var_13 = {var_1: var_8, var_2: var_9, var_3: var_10, var_4: var_9, var_5: var_10, var_6: var_11, var_7: var_12}
    var_14 = False
    var_15 = module_1.from_json_schema_type(var_13, var_8, var_14, var_0)
    var_16 = 'integer'
    var_17 = {var_1: var_16, var_2: var_14, var_3: var_10, var_4: var_14, var_5: var_10, var_6: var_11, var_7: var_12}
    var_18 = False
    var_19 = module_1.from_json_schema_type(var_17, var_16, var_18, var_0)
    var_20 = 'minLength'
    var_21 = 'maxLength'
    var_22 = 'format'
    var_23 = 'pattern'
    var_24 = 'string'
    var_25 = 1
    var_26 = 'email'
    var_27 = '^[a-zA-Z0-9]+$'
    var_28 = 'test'
    var_29 = {var_1: var_24, var_20: var_25, var_21: var_10, var_22: var_26, var_23: var_27, var_7: var_28}
    var_30 = False
    var_31 = module_1.from_json_schema_type(var_29, var_24, var_30, var_0)
    var_32 = 'boolean'
    var_33 = True
    var_34 = {var_1: var_32, var_7: var_33}
    var_35 = False
    var_36 = module_1.from_json_schema_type(var_34, var_32, var_35, var_0)
    var_37 = 'items'
    var_38 = 'minItems'
    var_39 = 'maxItems'
    var_40 = 'uniqueItems'
    var_41 = 'array'
    var_42 = {var_1: var_24}
    var_43 = True
    var_44 = [var_28]
    var_45 = {var_1: var_41, var_37: var_42, var_38: var_33, var_39: var_10, var_40: var_43, var_7: var_44}
    var_46 = False
    var_47 = module_1.from_json_schema_type(var_45, var_41, var_46, var_0)
    var_48 = var_47.items
    var_49 = 'properties'
    var_50 = 'minProperties'
    var_51 = 'maxProperties'
    var_52 = 'required'
    var_53 = 'object'
    var_54 = 'name'
    var_55 = {var_1: var_24}
    var_56 = {var_54: var_55}
    var_57 = [var_54]
    var_58 = {var_54: var_28}
    var_59 = {var_1: var_53, var_49: var_56, var_50: var_43, var_51: var_10, var_52: var_57, var_7: var_58}
    var_60 = False
    var_61 = module_1.from_json_schema_type(var_59, var_53, var_60, var_0)



# Parsed testcases at query #22
#--------------------------


import typesystem.fields as module_0
import typesystem.json_schema as module_1
import typesystem.composites as module_2
import typesystem.schemas as module_3

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_1.to_json_schema(var_0)
    var_2 = True
    var_3 = module_0.String()
    var_4 = module_1.to_json_schema(var_3)
    var_5 = module_0.Integer()
    var_6 = module_1.to_json_schema(var_5)
    var_7 = module_0.Float()
    var_8 = module_1.to_json_schema(var_7)
    var_9 = module_0.Boolean()
    var_10 = module_1.to_json_schema(var_9)
    var_11 = module_0.String()
    var_12 = module_0.Array(var_11)
    var_13 = module_1.to_json_schema(var_12)
    var_14 = 'name'
    var_15 = module_0.String()
    var_16 = {var_14: var_15}
    var_17 = module_0.Object(properties=var_16)
    var_18 = module_1.to_json_schema(var_17)
    var_19 = 'a'
    var_20 = 'A'
    var_21 = (var_19, var_20)
    var_22 = 'b'
    var_23 = 'B'
    var_24 = (var_22, var_23)
    var_25 = [var_21, var_24]
    var_26 = module_0.Choice(choices=var_25)
    var_27 = module_1.to_json_schema(var_26)
    var_28 = 'test'
    var_29 = module_0.Const(var_28)
    var_30 = module_1.to_json_schema(var_29)
    var_31 = module_0.String()
    var_32 = module_0.Integer()
    var_33 = [var_31, var_32]
    var_34 = module_0.Union(var_33)
    var_35 = module_1.to_json_schema(var_34)
    var_36 = module_0.String()
    var_37 = module_0.Integer()
    var_38 = [var_36, var_37]
    var_39 = module_2.OneOf(var_38)
    var_40 = module_1.to_json_schema(var_39)
    var_41 = module_0.String()
    var_42 = module_0.Integer()
    var_43 = [var_41, var_42]
    var_44 = module_2.AllOf(var_43)
    var_45 = module_1.to_json_schema(var_44)
    var_46 = module_0.String()
    var_47 = module_0.Integer()
    var_48 = module_2.IfThenElse(var_46, var_47)
    var_49 = module_1.to_json_schema(var_48)
    var_50 = module_0.String()
    var_51 = module_2.Not(var_50)
    var_52 = module_1.to_json_schema(var_51)
    var_53 = 'Person'
    var_54 = module_0.String()
    var_55 = {var_14: var_54}
    var_56 = module_0.Object(properties=var_55)
    var_57 = {var_53: var_56}
    var_58 = module_3.Reference(var_53, var_57)
    var_59 = module_1.to_json_schema(var_58)
    var_60 = module_0.String()
    var_61 = {var_14: var_60}
    var_62 = [var_14]
    var_63 = module_3.Schema(var_61)
    var_64 = module_1.to_json_schema(var_63)
    var_65 = module_2.NeverMatch()
    var_66 = module_1.to_json_schema(var_65)
    assert var_66 is False
    var_67 = module_0.Any()
    var_68 = module_1.to_json_schema(var_67)
    assert var_68 is True



# Parsed testcases at query #23
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = 'type'
    var_1 = 'minLength'
    var_2 = 'maxLength'
    var_3 = 'pattern'
    var_4 = 'format'
    var_5 = 'default'
    var_6 = 'string'
    var_7 = 1
    var_8 = 10
    var_9 = '^[a-zA-Z0-9_]+$'
    var_10 = 'email'
    var_11 = 'test@example.com'
    var_12 = {var_0: var_6, var_1: var_7, var_2: var_8, var_3: var_9, var_4: var_10, var_5: var_11}
    var_13 = False
    var_14 = module_0.String(allow_blank=var_13, max_length=var_8, min_length=var_7, pattern=var_9, format=var_10, coerce_types=var_13)
    var_15 = module_1.Definitions()
    var_16 = module_2.from_json_schema_type(var_12, var_6, var_13, var_15)
    var_17 = 'minimum'
    var_18 = 'maximum'
    var_19 = 'exclusiveMinimum'
    var_20 = 'exclusiveMaximum'
    var_21 = 'multipleOf'
    var_22 = 'integer'
    var_23 = 100
    var_24 = 101
    var_25 = 2
    var_26 = 50
    var_27 = {var_0: var_22, var_17: var_7, var_18: var_23, var_19: var_13, var_20: var_24, var_21: var_25, var_5: var_26}
    var_28 = module_0.Integer(minimum=var_7, maximum=var_23, exclusive_minimum=var_13, exclusive_maximum=var_24, multiple_of=var_25, coerce_types=var_13)
    var_29 = module_1.Definitions()
    var_30 = module_2.from_json_schema_type(var_27, var_22, var_13, var_29)
    var_31 = 'items'
    var_32 = 'minItems'
    var_33 = 'maxItems'
    var_34 = 'additionalItems'
    var_35 = 'uniqueItems'
    var_36 = 'array'
    var_37 = {var_0: var_6}
    var_38 = True
    var_39 = True
    var_40 = 'item1'
    var_41 = [var_40]
    var_42 = {var_0: var_36, var_31: var_37, var_32: var_7, var_33: var_8, var_34: var_38, var_35: var_39, var_5: var_41}
    var_43 = True
    var_44 = module_0.String(coerce_types=var_13)
    var_45 = True
    var_46 = [var_40]
    var_47 = module_0.Array(var_44, var_43, var_39, var_8, unique_items=var_45)
    var_48 = module_1.Definitions()
    var_49 = module_2.from_json_schema_type(var_42, var_36, var_13, var_48)
    var_50 = 'properties'
    var_51 = 'minProperties'
    var_52 = 'maxProperties'
    var_53 = 'additionalProperties'
    var_54 = 'required'
    var_55 = 'object'
    var_56 = 'name'
    var_57 = 'age'
    var_58 = {var_0: var_6}
    var_59 = {var_0: var_22}
    var_60 = {var_56: var_58, var_57: var_59}
    var_61 = [var_56]
    var_62 = 'John'
    var_63 = 30
    var_64 = {var_56: var_62, var_57: var_63}
    var_65 = {var_0: var_55, var_50: var_60, var_51: var_45, var_52: var_25, var_53: var_13, var_54: var_61, var_5: var_64}
    var_66 = module_0.String(coerce_types=var_13)
    var_67 = module_0.Integer(coerce_types=var_13)
    var_68 = {var_56: var_66, var_57: var_67}
    var_69 = [var_56]
    var_70 = {var_56: var_62, var_57: var_63}
    var_71 = module_0.Object(properties=var_68, additional_properties=var_13, min_properties=var_45, max_properties=var_25, required=var_69)
    var_72 = module_1.Definitions()
    var_73 = module_2.from_json_schema_type(var_65, var_55, var_13, var_72)



# Parsed testcases at query #24
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1
import re as module_2

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
    var_11 = 2
    var_12 = 4
    var_13 = {var_1: var_8, var_2: var_9, var_3: var_10, var_4: var_9, var_5: var_10, var_6: var_11, var_7: var_12}
    var_14 = False
    var_15 = module_1.from_json_schema_type(var_13, var_8, var_14, var_0)
    var_16 = 'integer'
    var_17 = {var_1: var_16, var_2: var_14, var_3: var_10, var_4: var_14, var_5: var_10, var_6: var_11, var_7: var_12}
    var_18 = False
    var_19 = module_1.from_json_schema_type(var_17, var_16, var_18, var_0)
    var_20 = 'minLength'
    var_21 = 'maxLength'
    var_22 = 'pattern'
    var_23 = 'format'
    var_24 = 'string'
    var_25 = 1
    var_26 = '^[a-z]+$'
    var_27 = 'email'
    var_28 = 'test'
    var_29 = {var_1: var_24, var_20: var_25, var_21: var_10, var_22: var_26, var_23: var_27, var_7: var_28}
    var_30 = False
    var_31 = module_1.from_json_schema_type(var_29, var_24, var_30, var_0)
    var_32 = module_2.compile(var_26)
    var_33 = 'boolean'
    var_34 = True
    var_35 = {var_1: var_33, var_7: var_34}
    var_36 = False
    var_37 = module_1.from_json_schema_type(var_35, var_33, var_36, var_0)
    var_38 = 'items'
    var_39 = 'minItems'
    var_40 = 'maxItems'
    var_41 = 'uniqueItems'
    var_42 = 'array'
    var_43 = {var_1: var_24}
    var_44 = True
    var_45 = [var_28]
    var_46 = {var_1: var_42, var_38: var_43, var_39: var_34, var_40: var_10, var_41: var_44, var_7: var_45}
    var_47 = False
    var_48 = module_1.from_json_schema_type(var_46, var_42, var_47, var_0)
    var_49 = var_48.items
    var_50 = 'properties'
    var_51 = 'minProperties'
    var_52 = 'maxProperties'
    var_53 = 'required'
    var_54 = 'object'
    var_55 = 'name'
    var_56 = {var_1: var_24}
    var_57 = {var_55: var_56}
    var_58 = [var_55]
    var_59 = {var_55: var_28}
    var_60 = {var_1: var_54, var_50: var_57, var_51: var_44, var_52: var_10, var_53: var_58, var_7: var_59}
    var_61 = False
    var_62 = module_1.from_json_schema_type(var_60, var_54, var_61, var_0)
    var_63 = var_62.properties[var_55]
    var_64 = {var_1: var_24}
    var_65 = True
    var_66 = module_1.from_json_schema_type(var_64, var_24, var_65, var_0)



