####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 'minLength'
    var_7 = 5
    var_8 = {var_3: var_4, var_6: var_7}
    var_9 = {var_1: var_5, var_2: var_8}
    var_10 = module_1.if_then_else_from_json_schema(var_9, var_0)
    var_11 = var_10.if_clause
    var_12 = var_10.then_clause
    var_13 = 'else'
    var_14 = {var_3: var_4}
    var_15 = {var_3: var_4, var_6: var_7}
    var_16 = 'minimum'
    var_17 = 'integer'
    var_18 = 0
    var_19 = {var_3: var_17, var_16: var_18}
    var_20 = {var_1: var_14, var_2: var_15, var_13: var_19}
    var_21 = module_1.if_then_else_from_json_schema(var_20, var_0)
    var_22 = var_21.if_clause
    var_23 = var_21.then_clause
    var_24 = var_21.else_clause
    var_25 = 'default'
    var_26 = 'boolean'
    var_27 = {var_3: var_26}
    var_28 = {var_3: var_26}
    var_29 = 'null'
    var_30 = {var_3: var_29}
    var_31 = True
    var_32 = {var_1: var_27, var_2: var_28, var_13: var_30, var_25: var_31}
    var_33 = module_1.if_then_else_from_json_schema(var_32, var_0)
    var_34 = 'properties'
    var_35 = 'object'
    var_36 = 'x'
    var_37 = {var_3: var_17}
    var_38 = {var_36: var_37}
    var_39 = {var_3: var_35, var_34: var_38}
    var_40 = 'required'
    var_41 = [var_36]
    var_42 = {var_3: var_35, var_40: var_41}
    var_43 = 'y'
    var_44 = {var_3: var_4}
    var_45 = {var_43: var_44}
    var_46 = {var_3: var_35, var_34: var_45}
    var_47 = {var_1: var_39, var_2: var_42, var_13: var_46}
    var_48 = module_1.if_then_else_from_json_schema(var_47, var_0)
    var_49 = var_48.if_clause
    var_50 = var_48.then_clause
    var_51 = var_48.else_clause
    var_52 = 'minItems'
    var_53 = 'array'
    var_54 = {var_3: var_53, var_52: var_31}
    var_55 = 'uniqueItems'
    var_56 = {var_3: var_53, var_55: var_31}
    var_57 = {var_1: var_54, var_2: var_56}
    var_58 = module_1.if_then_else_from_json_schema(var_57, var_0)
    var_59 = var_58.if_clause
    var_60 = var_58.then_clause
    var_61 = 'number'
    var_62 = {var_3: var_61, var_16: var_18}
    var_63 = 'maximum'
    var_64 = 100
    var_65 = {var_3: var_61, var_63: var_64}
    var_66 = -100
    var_67 = {var_3: var_61, var_16: var_66}
    var_68 = {var_1: var_62, var_2: var_65, var_13: var_67}
    var_69 = module_1.if_then_else_from_json_schema(var_68, var_0)
    var_70 = var_69.if_clause
    var_71 = var_69.then_clause
    var_72 = var_69.else_clause



# Parsed testcases at query #2
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'enum'
    var_2 = 'red'
    var_3 = 'green'
    var_4 = 'blue'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = module_1.enum_from_json_schema(var_6, var_0)
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = [var_8, var_9, var_10]
    var_12 = {var_1: var_11}
    var_13 = module_1.enum_from_json_schema(var_12, var_0)
    var_14 = 'default'
    var_15 = 'a'
    var_16 = 'b'
    var_17 = 'c'
    var_18 = [var_15, var_16, var_17]
    var_19 = {var_1: var_18, var_14: var_16}
    var_20 = module_1.enum_from_json_schema(var_19, var_0)
    var_21 = 'key'
    var_22 = 'value'
    var_23 = {var_21: var_22}
    var_24 = [var_8, var_9, var_10]
    var_25 = True
    var_26 = [var_23, var_24, var_25]
    var_27 = {var_1: var_26}
    var_28 = module_1.enum_from_json_schema(var_27, var_0)
    var_29 = None
    var_30 = [var_29, var_22]
    var_31 = {var_1: var_30}
    var_32 = module_1.enum_from_json_schema(var_31, var_0)
    var_33 = 'only'
    var_34 = [var_33]
    var_35 = {var_1: var_34}
    var_36 = module_1.enum_from_json_schema(var_35, var_0)



# Parsed testcases at query #3
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = '#/components/schemas/User'
    var_3 = {var_1: var_2}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)
    var_5 = '#/definitions/Address'
    var_6 = {var_1: var_5}
    var_7 = module_1.ref_from_json_schema(var_6, var_0)
    var_8 = 'http://example.com/schema.json'
    var_9 = {var_1: var_8}
    var_10 = module_1.ref_from_json_schema(var_9, var_0)
    var_11 = module_0.Definitions()
    var_12 = '#/components/schemas/Test'
    var_13 = {var_10: var_12}
    var_14 = module_1.ref_from_json_schema(var_13, var_11)



# Parsed testcases at query #4
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = '#/components/schemas/User'
    var_3 = {var_1: var_2}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)
    var_5 = '#/definitions/Address'
    var_6 = {var_1: var_5}
    var_7 = module_1.ref_from_json_schema(var_6, var_0)
    var_8 = '$ref'
    var_9 = 'http://example.com/schema'
    var_10 = {var_8: var_9}
    var_11 = module_1.ref_from_json_schema(var_10, var_0)
    var_12 = module_0.Definitions()
    var_13 = '#/components/schemas/Test'
    var_14 = {var_8: var_13}
    var_15 = module_1.ref_from_json_schema(var_14, var_12)



# Parsed testcases at query #5
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1
import typesystem.fields as module_2

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
    var_17 = {var_2: var_3}
    var_18 = {var_2: var_5}
    var_19 = [var_17, var_18]
    var_20 = 'test'
    var_21 = {var_1: var_19, var_16: var_20}
    var_22 = module_1.one_of_from_json_schema(var_21, var_0)
    var_23 = 'properties'
    var_24 = 'object'
    var_25 = 'name'
    var_26 = {var_2: var_3}
    var_27 = {var_25: var_26}
    var_28 = {var_2: var_24, var_23: var_27}
    var_29 = 'items'
    var_30 = 'array'
    var_31 = 'number'
    var_32 = {var_2: var_31}
    var_33 = {var_2: var_30, var_29: var_32}
    var_34 = [var_28, var_33]
    var_35 = {var_1: var_34}
    var_36 = module_1.one_of_from_json_schema(var_35, var_0)
    var_37 = var_36.one_of
    var_38 = len(var_37)
    assert var_38 == 2
    var_39 = var_36.one_of[var_12]
    var_40 = var_36.one_of[var_14]
    var_41 = module_2.String()
    var_42 = {var_25: var_41}
    var_43 = '$ref'
    var_44 = '#/components/schemas/Person'
    var_45 = {var_43: var_44}
    var_46 = {var_2: var_3}
    var_47 = [var_45, var_46]
    var_48 = {var_1: var_47}
    var_49 = module_1.one_of_from_json_schema(var_48, var_0)
    var_50 = var_49.one_of[var_12]
    var_51 = var_49.one_of[var_14]
    var_52 = 'boolean'
    var_53 = {var_2: var_52}
    var_54 = [var_53]
    var_55 = {var_1: var_54}
    var_56 = module_1.one_of_from_json_schema(var_55, var_0)
    var_57 = var_56.one_of
    var_58 = len(var_57)
    assert var_58 == 1
    var_59 = var_56.one_of[var_12]



# Parsed testcases at query #6
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 'minLength'
    var_7 = 5
    var_8 = {var_3: var_4, var_6: var_7}
    var_9 = {var_1: var_5, var_2: var_8}
    var_10 = module_1.if_then_else_from_json_schema(var_9, var_0)
    var_11 = var_10.if_clause
    var_12 = var_10.then_clause
    var_13 = 'else'
    var_14 = 'minimum'
    var_15 = 'number'
    var_16 = 0
    var_17 = {var_3: var_15, var_14: var_16}
    var_18 = 'maximum'
    var_19 = 100
    var_20 = {var_3: var_15, var_14: var_16, var_18: var_19}
    var_21 = -1
    var_22 = {var_3: var_15, var_18: var_21}
    var_23 = {var_1: var_17, var_2: var_20, var_13: var_22}
    var_24 = module_1.if_then_else_from_json_schema(var_23, var_0)
    var_25 = var_24.if_clause
    var_26 = var_24.then_clause
    var_27 = var_24.else_clause
    var_28 = '$ref'
    var_29 = '#/definitions/Positive'
    var_30 = {var_28: var_29}
    var_31 = {var_3: var_4}
    var_32 = 'boolean'
    var_33 = {var_3: var_32}
    var_34 = {var_1: var_30, var_2: var_31, var_13: var_33}
    var_35 = module_1.if_then_else_from_json_schema(var_34, var_0)
    var_36 = var_35.if_clause
    var_37 = var_35.then_clause
    var_38 = var_35.else_clause
    var_39 = 'default'
    var_40 = 'array'
    var_41 = {var_3: var_40}
    var_42 = 'minItems'
    var_43 = 2
    var_44 = {var_3: var_40, var_42: var_43}
    var_45 = 'maxItems'
    var_46 = 1
    var_47 = {var_3: var_40, var_45: var_46}
    var_48 = []
    var_49 = {var_1: var_41, var_2: var_44, var_13: var_47, var_39: var_48}
    var_50 = module_1.if_then_else_from_json_schema(var_49, var_0)
    var_51 = 'allOf'
    var_52 = 'object'
    var_53 = {var_3: var_52}
    var_54 = 'required'
    var_55 = 'status'
    var_56 = [var_55]
    var_57 = {var_54: var_56}
    var_58 = [var_53, var_57]
    var_59 = {var_51: var_58}
    var_60 = 'properties'
    var_61 = {var_3: var_4}
    var_62 = {var_55: var_61}
    var_63 = {var_3: var_52, var_60: var_62}
    var_64 = {var_1: var_59, var_2: var_63}
    var_65 = module_1.if_then_else_from_json_schema(var_64, var_0)
    var_66 = var_65.if_clause
    var_67 = var_65.then_clause



# Parsed testcases at query #7
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
    var_4 = False
    var_5 = 1
    var_6 = 10
    var_7 = '^[a-z]+$'
    var_8 = 'email'
    var_9 = module_0.String(max_length=var_6, min_length=var_5, pattern=var_7, format=var_8)
    var_10 = module_1.to_json_schema(var_9)
    var_11 = True
    var_12 = module_0.String()
    var_13 = module_1.to_json_schema(var_12)
    var_14 = 100
    var_15 = 2
    var_16 = module_0.Integer(minimum=var_4, maximum=var_14, exclusive_minimum=var_4, exclusive_maximum=var_14, multiple_of=var_15)
    var_17 = module_1.to_json_schema(var_16)
    var_18 = True
    var_19 = module_0.Float(minimum=var_4, maximum=var_18)
    var_20 = module_1.to_json_schema(var_19)
    var_21 = True
    var_22 = module_0.Boolean()
    var_23 = module_1.to_json_schema(var_22)
    var_24 = module_0.String()
    var_25 = True
    var_26 = module_0.Array(var_24, min_items=var_21, max_items=var_6, unique_items=var_25)
    var_27 = module_1.to_json_schema(var_26)
    var_28 = module_0.String()
    var_29 = module_0.Integer()
    var_30 = [var_28, var_29]
    var_31 = module_0.Array(var_30)
    var_32 = module_1.to_json_schema(var_31)
    var_33 = 'items'
    var_34 = var_32[var_33]
    var_35 = var_32[var_33]
    var_36 = len(var_35)
    assert var_36 == 2
    var_37 = module_0.Array(additional_items=var_4)
    var_38 = module_1.to_json_schema(var_37)
    var_39 = module_0.String()
    var_40 = module_0.Array(additional_items=var_39)
    var_41 = module_1.to_json_schema(var_40)
    var_42 = 'additionalItems'
    var_43 = var_41[var_42]
    var_44 = True
    var_45 = 'name'
    var_46 = 'age'
    var_47 = module_0.String()
    var_48 = module_0.Integer()
    var_49 = {var_45: var_47, var_46: var_48}
    var_50 = '^test_'
    var_51 = module_0.Boolean()
    var_52 = {var_50: var_51}
    var_53 = [var_45]
    var_54 = module_0.Object(properties=var_49, pattern_properties=var_52, additional_properties=var_4, min_properties=var_44, max_properties=var_6, required=var_53)
    var_55 = module_1.to_json_schema(var_54)
    var_56 = module_0.String()
    var_57 = module_0.Object(additional_properties=var_56)
    var_58 = module_1.to_json_schema(var_57)
    var_59 = 'additionalProperties'
    var_60 = var_58[var_59]
    var_61 = module_0.String(pattern=var_7)
    var_62 = module_0.Object(property_names=var_61)
    var_63 = module_1.to_json_schema(var_62)
    var_64 = module_0.String()
    var_65 = {var_45: var_64}
    var_66 = [var_45]
    var_67 = module_3.Schema(var_65)
    var_68 = module_1.to_json_schema(var_67)
    var_69 = 'a'
    var_70 = 'A'
    var_71 = (var_69, var_70)
    var_72 = 'b'
    var_73 = 'B'
    var_74 = (var_72, var_73)
    var_75 = [var_71, var_74]
    var_76 = module_0.Choice(choices=var_75)
    var_77 = module_1.to_json_schema(var_76)
    var_78 = 'fixed_value'
    var_79 = module_0.Const(var_78)
    var_80 = module_1.to_json_schema(var_79)
    var_81 = module_0.String()
    var_82 = module_0.Integer()
    var_83 = [var_81, var_82]
    var_84 = module_0.Union(var_83)
    var_85 = module_1.to_json_schema(var_84)
    var_86 = 'anyOf'
    var_87 = var_85[var_86]
    var_88 = len(var_87)
    assert var_88 == 2
    var_89 = module_0.String()
    var_90 = module_0.Integer()
    var_91 = [var_89, var_90]
    var_92 = module_2.OneOf(var_91)
    var_93 = module_1.to_json_schema(var_92)
    var_94 = 'oneOf'
    var_95 = var_93[var_94]
    var_96 = len(var_95)
    assert var_96 == 2
    var_97 = module_0.String(min_length=var_44)
    var_98 = module_0.String(max_length=var_6)
    var_99 = [var_97, var_98]
    var_100 = module_2.AllOf(var_99)
    var_101 = module_1.to_json_schema(var_100)
    var_102 = 'allOf'
    var_103 = var_101[var_102]
    var_104 = len(var_103)
    assert var_104 == 2
    var_105 = 5
    var_106 = module_0.String(min_length=var_105)
    var_107 = 20
    var_108 = module_0.String(max_length=var_107)
    var_109 = module_0.Integer()
    var_110 = module_2.IfThenElse(var_106, var_108, var_109)
    var_111 = module_1.to_json_schema(var_110)
    var_112 = module_0.String()
    var_113 = module_2.IfThenElse(var_112)
    var_114 = module_1.to_json_schema(var_113)
    var_115 = module_0.String()
    var_116 = module_2.Not(var_115)
    var_117 = module_1.to_json_schema(var_116)
    var_118 = module_3.Definitions()
    var_119 = module_0.String()
    var_120 = {var_45: var_119}
    var_121 = 'User'
    var_122 = module_3.Reference(var_121, var_118)
    var_123 = module_1.to_json_schema(var_122)
    var_124 = module_3.Definitions()
    var_125 = module_0.String()
    var_126 = {var_45: var_125}
    var_127 = module_1.to_json_schema(var_124)
    var_128 = 'test'
    var_129 = module_0.String()
    var_130 = module_1.to_json_schema(var_129)
    var_131 = module_0.String()
    var_132 = module_1.to_json_schema(var_131)
    var_133 = module_0.Decimal(minimum=var_4)
    var_134 = module_1.to_json_schema(var_133)
    var_135 = module_1.to_json_schema(var_133)
    var_136 = '^test$'
    var_137 = module_1.to_json_schema(var_133)



# Parsed testcases at query #8
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'number'
    var_4 = 0
    var_5 = 10
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = False
    var_8 = module_0.Definitions()
    var_9 = module_1.from_json_schema_type(var_6, var_3, var_7, var_8)
    var_10 = 'exclusiveMinimum'
    var_11 = 'multipleOf'
    var_12 = 'integer'
    var_13 = 5
    var_14 = 2
    var_15 = {var_0: var_12, var_10: var_13, var_11: var_14}
    var_16 = True
    var_17 = module_0.Definitions()
    var_18 = module_1.from_json_schema_type(var_15, var_12, var_16, var_17)
    var_19 = 'minLength'
    var_20 = 'maxLength'
    var_21 = 'pattern'
    var_22 = 'string'
    var_23 = 3
    var_24 = '^a.*z$'
    var_25 = {var_0: var_22, var_19: var_23, var_20: var_5, var_21: var_24}
    var_26 = False
    var_27 = module_0.Definitions()
    var_28 = module_1.from_json_schema_type(var_25, var_22, var_26, var_27)
    var_29 = 'default'
    var_30 = 'boolean'
    var_31 = {var_0: var_30, var_29: var_16}
    var_32 = module_0.Definitions()
    var_33 = module_1.from_json_schema_type(var_31, var_30, var_16, var_32)
    var_34 = 'items'
    var_35 = 'minItems'
    var_36 = 'array'
    var_37 = {var_0: var_22}
    var_38 = {var_0: var_36, var_34: var_37, var_35: var_16}
    var_39 = False
    var_40 = module_0.Definitions()
    var_41 = module_1.from_json_schema_type(var_38, var_36, var_39, var_40)
    var_42 = var_41.items
    var_43 = {var_0: var_22}
    var_44 = {var_0: var_3}
    var_45 = [var_43, var_44]
    var_46 = {var_0: var_36, var_34: var_45}
    var_47 = False
    var_48 = module_0.Definitions()
    var_49 = module_1.from_json_schema_type(var_46, var_36, var_47, var_48)
    var_50 = var_49.items
    var_51 = len(var_50)
    assert var_51 == 2
    var_52 = var_49.items[var_47]
    var_53 = var_49.items[var_16]
    var_54 = 'properties'
    var_55 = 'object'
    var_56 = 'name'
    var_57 = {var_0: var_22}
    var_58 = {var_56: var_57}
    var_59 = {var_0: var_55, var_54: var_58}
    var_60 = False
    var_61 = module_0.Definitions()
    var_62 = module_1.from_json_schema_type(var_59, var_55, var_60, var_61)
    var_63 = var_62.properties[var_56]
    var_64 = 'patternProperties'
    var_65 = '^S_'
    var_66 = {var_0: var_22}
    var_67 = {var_65: var_66}
    var_68 = {var_0: var_55, var_64: var_67}
    var_69 = module_0.Definitions()
    var_70 = module_1.from_json_schema_type(var_68, var_55, var_16, var_69)
    var_71 = var_70.pattern_properties[var_65]
    var_72 = 'additionalProperties'
    var_73 = False
    var_74 = {var_0: var_55, var_72: var_73}
    var_75 = False
    var_76 = module_0.Definitions()
    var_77 = module_1.from_json_schema_type(var_74, var_55, var_75, var_76)
    var_78 = 'propertyNames'
    var_79 = '^[a-z]+$'
    var_80 = {var_21: var_79}
    var_81 = {var_0: var_55, var_78: var_80}
    var_82 = False
    var_83 = module_0.Definitions()
    var_84 = module_1.from_json_schema_type(var_81, var_55, var_82, var_83)
    var_85 = var_84.property_names
    var_86 = 'test'
    var_87 = {var_0: var_22, var_29: var_86}
    var_88 = False
    var_89 = module_0.Definitions()
    var_90 = module_1.from_json_schema_type(var_87, var_22, var_88, var_89)
    var_91 = {var_0: var_3}
    var_92 = False
    var_93 = module_0.Definitions()
    var_94 = module_1.from_json_schema_type(var_91, var_3, var_92, var_93)
    var_95 = {}
    var_96 = 'invalid'
    var_97 = False
    var_98 = module_0.Definitions()
    var_99 = module_1.from_json_schema_type(var_95, var_96, var_97, var_98)



# Parsed testcases at query #9
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1
import typesystem.fields as module_2

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'allOf'
    var_2 = 'type'
    var_3 = 'minLength'
    var_4 = 'string'
    var_5 = 1
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
    var_15 = var_12.all_of
    var_16 = {var_2: var_4}
    var_17 = 'pattern'
    var_18 = '^[A-Z]+$'
    var_19 = {var_2: var_4, var_17: var_18}
    var_20 = [var_16, var_19]
    var_21 = {var_1: var_20}
    var_22 = module_1.all_of_from_json_schema(var_21, var_0)
    var_23 = var_22.all_of
    var_24 = len(var_23)
    assert var_24 == 2
    var_25 = var_22.all_of
    var_26 = 'default'
    var_27 = 'integer'
    var_28 = {var_2: var_27}
    var_29 = 'minimum'
    var_30 = 0
    var_31 = {var_2: var_27, var_29: var_30}
    var_32 = [var_28, var_31]
    var_33 = 5
    var_34 = {var_1: var_32, var_26: var_33}
    var_35 = module_1.all_of_from_json_schema(var_34, var_0)
    var_36 = 'properties'
    var_37 = 'object'
    var_38 = 'name'
    var_39 = {var_2: var_4}
    var_40 = {var_38: var_39}
    var_41 = {var_2: var_37, var_36: var_40}
    var_42 = 'required'
    var_43 = [var_38]
    var_44 = {var_2: var_37, var_42: var_43}
    var_45 = [var_41, var_44]
    var_46 = {var_1: var_45}
    var_47 = module_1.all_of_from_json_schema(var_46, var_0)
    var_48 = var_47.all_of
    var_49 = len(var_48)
    assert var_49 == 2
    var_50 = var_47.all_of
    var_51 = module_2.String()
    var_52 = {var_38: var_51}
    var_53 = '$ref'
    var_54 = '#/components/schemas/Person'
    var_55 = {var_53: var_54}
    var_56 = 'age'
    var_57 = {var_2: var_27}
    var_58 = {var_56: var_57}
    var_59 = {var_2: var_37, var_36: var_58}
    var_60 = [var_55, var_59]
    var_61 = {var_1: var_60}
    var_62 = module_1.all_of_from_json_schema(var_61, var_0)
    var_63 = var_62.all_of
    var_64 = len(var_63)
    assert var_64 == 2
    var_65 = var_62.all_of[var_30]
    var_66 = var_62.all_of[var_5]
    var_67 = 'items'
    var_68 = 'array'
    var_69 = {var_2: var_4}
    var_70 = {var_2: var_68, var_67: var_69}
    var_71 = 'minItems'
    var_72 = {var_2: var_68, var_71: var_5}
    var_73 = [var_70, var_72]
    var_74 = {var_1: var_73}
    var_75 = module_1.all_of_from_json_schema(var_74, var_0)
    var_76 = var_75.all_of
    var_77 = len(var_76)
    assert var_77 == 2
    var_78 = var_75.all_of
    var_79 = 'number'
    var_80 = {var_2: var_79, var_29: var_30}
    var_81 = 'maximum'
    var_82 = 100
    var_83 = {var_2: var_79, var_81: var_82}
    var_84 = [var_80, var_83]
    var_85 = {var_1: var_84}
    var_86 = module_1.all_of_from_json_schema(var_85, var_0)
    var_87 = var_86.all_of
    var_88 = len(var_87)
    assert var_88 == 2
    var_89 = var_86.all_of
    var_90 = 'boolean'
    var_91 = {var_2: var_90}
    var_92 = [var_91]
    var_93 = {var_1: var_92}
    var_94 = module_1.all_of_from_json_schema(var_93, var_0)
    var_95 = var_94.all_of
    var_96 = len(var_95)
    assert var_96 == 1
    var_97 = var_94.all_of[var_30]
    var_98 = []
    var_99 = {var_1: var_98}
    var_100 = module_1.all_of_from_json_schema(var_99, var_0)
    var_101 = var_100.all_of
    var_102 = len(var_101)
    assert var_102 == 0



# Parsed testcases at query #10
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1
import typesystem.fields as module_2

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'allOf'
    var_2 = 'type'
    var_3 = 'minLength'
    var_4 = 'string'
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'maxLength'
    var_8 = 5
    var_9 = {var_2: var_4, var_7: var_8}
    var_10 = [var_6, var_9]
    var_11 = {var_1: var_10}
    var_12 = module_1.all_of_from_json_schema(var_11, var_0)
    var_13 = var_12.all_of
    var_14 = len(var_13)
    assert var_14 == 2
    var_15 = var_12.all_of
    var_16 = 'default'
    var_17 = 'minimum'
    var_18 = 'integer'
    var_19 = 0
    var_20 = {var_2: var_18, var_17: var_19}
    var_21 = 'maximum'
    var_22 = 100
    var_23 = {var_2: var_18, var_21: var_22}
    var_24 = [var_20, var_23]
    var_25 = 50
    var_26 = {var_1: var_24, var_16: var_25}
    var_27 = module_1.all_of_from_json_schema(var_26, var_0)
    var_28 = 'properties'
    var_29 = 'object'
    var_30 = 'name'
    var_31 = {var_2: var_4}
    var_32 = {var_30: var_31}
    var_33 = {var_2: var_29, var_28: var_32}
    var_34 = 'required'
    var_35 = [var_30]
    var_36 = {var_2: var_29, var_34: var_35}
    var_37 = [var_33, var_36]
    var_38 = {var_1: var_37}
    var_39 = module_1.all_of_from_json_schema(var_38, var_0)
    var_40 = var_39.all_of
    var_41 = len(var_40)
    assert var_41 == 2
    var_42 = var_39.all_of[var_19]
    var_43 = 1
    var_44 = var_39.all_of[var_43]
    var_45 = module_2.String()
    var_46 = {var_30: var_45}
    var_47 = '$ref'
    var_48 = '#/components/schemas/Person'
    var_49 = {var_47: var_48}
    var_50 = 'age'
    var_51 = {var_2: var_18}
    var_52 = {var_50: var_51}
    var_53 = {var_2: var_29, var_28: var_52}
    var_54 = [var_49, var_53]
    var_55 = {var_1: var_54}
    var_56 = module_1.all_of_from_json_schema(var_55, var_0)
    var_57 = var_56.all_of[var_19]
    var_58 = var_56.all_of[var_43]
    var_59 = 'minItems'
    var_60 = 'array'
    var_61 = {var_2: var_60, var_59: var_43}
    var_62 = 'maxItems'
    var_63 = 10
    var_64 = {var_2: var_60, var_62: var_63}
    var_65 = [var_61, var_64]
    var_66 = {var_1: var_65}
    var_67 = module_1.all_of_from_json_schema(var_66, var_0)
    var_68 = var_67.all_of
    var_69 = 'boolean'
    var_70 = {var_2: var_69}
    var_71 = 'const'
    var_72 = True
    var_73 = {var_71: var_72}
    var_74 = [var_70, var_73]
    var_75 = {var_1: var_74}
    var_76 = module_1.all_of_from_json_schema(var_75, var_0)
    var_77 = var_76.all_of[var_19]
    var_78 = var_76.all_of[var_72]



# Parsed testcases at query #11
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
    var_17 = 'array'
    var_18 = {var_4: var_17}
    var_19 = module_0.from_json_schema(var_18)
    var_20 = 'object'
    var_21 = {var_4: var_20}
    var_22 = module_0.from_json_schema(var_21)
    var_23 = 'enum'
    var_24 = 'a'
    var_25 = 'b'
    var_26 = 'c'
    var_27 = [var_24, var_25, var_26]
    var_28 = {var_23: var_27}
    var_29 = module_0.from_json_schema(var_28)
    var_30 = 'const'
    var_31 = 'fixed_value'
    var_32 = {var_30: var_31}
    var_33 = module_0.from_json_schema(var_32)
    var_34 = 'allOf'
    var_35 = {var_4: var_5}
    var_36 = 'minLength'
    var_37 = 5
    var_38 = {var_36: var_37}
    var_39 = [var_35, var_38]
    var_40 = {var_34: var_39}
    var_41 = module_0.from_json_schema(var_40)
    var_42 = 'anyOf'
    var_43 = {var_4: var_5}
    var_44 = {var_4: var_11}
    var_45 = [var_43, var_44]
    var_46 = {var_42: var_45}
    var_47 = module_0.from_json_schema(var_46)
    var_48 = 'oneOf'
    var_49 = {var_4: var_5}
    var_50 = {var_4: var_11}
    var_51 = [var_49, var_50]
    var_52 = {var_48: var_51}
    var_53 = module_0.from_json_schema(var_52)
    var_54 = 'not'
    var_55 = {var_4: var_5}
    var_56 = {var_54: var_55}
    var_57 = module_0.from_json_schema(var_56)
    var_58 = 'if'
    var_59 = 'then'
    var_60 = 'else'
    var_61 = {var_4: var_5}
    var_62 = {var_36: var_37}
    var_63 = {var_4: var_11}
    var_64 = {var_58: var_61, var_59: var_62, var_60: var_63}
    var_65 = module_0.from_json_schema(var_64)
    var_66 = '$ref'
    var_67 = '#/components/schemas/User'
    var_68 = {var_66: var_67}
    var_69 = module_0.from_json_schema(var_68)
    var_70 = 'maxLength'
    var_71 = 'pattern'
    var_72 = 10
    var_73 = '^[a-z]+$'
    var_74 = {var_4: var_5, var_36: var_37, var_70: var_72, var_71: var_73}
    var_75 = module_0.from_json_schema(var_74)
    var_76 = {}
    var_77 = module_0.from_json_schema(var_76)
    var_78 = module_1.Definitions()
    var_79 = 'properties'
    var_80 = 'user'
    var_81 = {var_66: var_67}
    var_82 = {var_80: var_81}
    var_83 = {var_4: var_20, var_79: var_82}
    var_84 = module_0.from_json_schema(var_83, var_78)
    var_85 = 'items'
    var_86 = {var_4: var_5}
    var_87 = {var_4: var_17, var_85: var_86}
    var_88 = module_0.from_json_schema(var_87)
    var_89 = var_88.items
    var_90 = 'name'
    var_91 = 'age'
    var_92 = {var_4: var_5}
    var_93 = {var_4: var_8}
    var_94 = {var_90: var_92, var_91: var_93}
    var_95 = {var_4: var_20, var_79: var_94}
    var_96 = module_0.from_json_schema(var_95)
    var_97 = 'format'
    var_98 = 'email'
    var_99 = {var_4: var_5, var_97: var_98}
    var_100 = module_0.from_json_schema(var_99)
    var_101 = 'minimum'
    var_102 = 'maximum'
    var_103 = 'exclusiveMinimum'
    var_104 = 'exclusiveMaximum'
    var_105 = 'multipleOf'
    var_106 = 100
    var_107 = {var_4: var_11, var_101: var_2, var_102: var_106, var_103: var_2, var_104: var_106, var_105: var_37}
    var_108 = module_0.from_json_schema(var_107)
    var_109 = [var_5, var_11]
    var_110 = {var_4: var_109}
    var_111 = module_0.from_json_schema(var_110)
    var_112 = {var_4: var_5}
    var_113 = {var_4: var_11}
    var_114 = [var_112, var_113]
    var_115 = {var_4: var_17, var_85: var_114}
    var_116 = module_0.from_json_schema(var_115)
    var_117 = var_116.items
    var_118 = var_116.items
    var_119 = len(var_118)
    assert var_119 == 2
    var_120 = 'additionalProperties'
    var_121 = {var_4: var_5}
    var_122 = {var_4: var_20, var_120: var_121}
    var_123 = module_0.from_json_schema(var_122)
    var_124 = 'required'
    var_125 = {var_4: var_5}
    var_126 = {var_4: var_8}
    var_127 = {var_90: var_125, var_91: var_126}
    var_128 = [var_90]
    var_129 = {var_4: var_20, var_79: var_127, var_124: var_128}
    var_130 = module_0.from_json_schema(var_129)
    var_131 = 'uniqueItems'
    var_132 = {var_4: var_17, var_131: var_0}
    var_133 = module_0.from_json_schema(var_132)



# Parsed testcases at query #12
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
    var_4 = False
    var_5 = 1
    var_6 = 10
    var_7 = '^[a-z]+$'
    var_8 = 'email'
    var_9 = module_0.String(max_length=var_6, min_length=var_5, pattern=var_7, format=var_8)
    var_10 = module_1.to_json_schema(var_9)
    var_11 = True
    var_12 = True
    var_13 = module_0.String(allow_blank=var_12, min_length=var_4)
    var_14 = module_1.to_json_schema(var_13)
    var_15 = 100
    var_16 = 2
    var_17 = module_0.Integer(minimum=var_4, maximum=var_15, exclusive_minimum=var_4, exclusive_maximum=var_15, multiple_of=var_16)
    var_18 = module_1.to_json_schema(var_17)
    var_19 = True
    var_20 = module_0.Float(minimum=var_4, maximum=var_19)
    var_21 = module_1.to_json_schema(var_20)
    var_22 = True
    var_23 = module_0.Boolean()
    var_24 = module_1.to_json_schema(var_23)
    var_25 = module_0.String()
    var_26 = True
    var_27 = module_0.Array(var_25, min_items=var_22, max_items=var_6, unique_items=var_26)
    var_28 = module_1.to_json_schema(var_27)
    var_29 = module_0.String()
    var_30 = module_0.Integer()
    var_31 = [var_29, var_30]
    var_32 = module_0.Array(var_31, var_4)
    var_33 = module_1.to_json_schema(var_32)
    var_34 = 'items'
    var_35 = var_33[var_34]
    var_36 = var_33[var_34]
    var_37 = len(var_36)
    assert var_37 == 2
    var_38 = True
    var_39 = 'name'
    var_40 = 'age'
    var_41 = module_0.String()
    var_42 = module_0.Integer()
    var_43 = {var_39: var_41, var_40: var_42}
    var_44 = [var_39]
    var_45 = module_0.Object(properties=var_43, additional_properties=var_4, min_properties=var_38, max_properties=var_16, required=var_44)
    var_46 = module_1.to_json_schema(var_45)
    var_47 = module_0.String()
    var_48 = {var_7: var_47}
    var_49 = module_0.Object(pattern_properties=var_48)
    var_50 = module_1.to_json_schema(var_49)
    var_51 = 'A'
    var_52 = 'Option A'
    var_53 = (var_51, var_52)
    var_54 = 'B'
    var_55 = 'Option B'
    var_56 = (var_54, var_55)
    var_57 = [var_53, var_56]
    var_58 = module_0.Choice(choices=var_57)
    var_59 = module_1.to_json_schema(var_58)
    var_60 = 'fixed_value'
    var_61 = module_0.Const(var_60)
    var_62 = module_1.to_json_schema(var_61)
    var_63 = module_0.String()
    var_64 = module_0.Integer()
    var_65 = [var_63, var_64]
    var_66 = module_0.Union(var_65)
    var_67 = module_1.to_json_schema(var_66)
    var_68 = 'anyOf'
    var_69 = var_67[var_68]
    var_70 = len(var_69)
    assert var_70 == 2
    var_71 = module_0.String()
    var_72 = module_0.Integer()
    var_73 = [var_71, var_72]
    var_74 = module_2.OneOf(var_73)
    var_75 = module_1.to_json_schema(var_74)
    var_76 = 'oneOf'
    var_77 = var_75[var_76]
    var_78 = len(var_77)
    assert var_78 == 2
    var_79 = module_0.String(min_length=var_38)
    var_80 = module_0.String(max_length=var_6)
    var_81 = [var_79, var_80]
    var_82 = module_2.AllOf(var_81)
    var_83 = module_1.to_json_schema(var_82)
    var_84 = 'allOf'
    var_85 = var_83[var_84]
    var_86 = len(var_85)
    assert var_86 == 2
    var_87 = 5
    var_88 = module_0.String(min_length=var_87)
    var_89 = 20
    var_90 = module_0.String(max_length=var_89)
    var_91 = module_0.Integer()
    var_92 = module_2.IfThenElse(var_88, var_90, var_91)
    var_93 = module_1.to_json_schema(var_92)
    var_94 = module_0.String()
    var_95 = module_2.Not(var_94)
    var_96 = module_1.to_json_schema(var_95)
    var_97 = 'User'
    var_98 = module_0.String()
    var_99 = {var_39: var_98}
    var_100 = module_0.Object(properties=var_99)
    var_101 = {var_97: var_100}
    var_102 = {}
    var_103 = module_1.to_json_schema(var_95, var_102)
    var_104 = module_0.String()
    var_105 = {var_39: var_104}
    var_106 = [var_39]
    var_107 = module_3.Schema(var_105)
    var_108 = module_1.to_json_schema(var_107)
    var_109 = 'default_value'
    var_110 = module_0.String()
    var_111 = module_1.to_json_schema(var_110)
    var_112 = module_1.to_json_schema(var_110)
    var_113 = module_1.to_json_schema(var_110)



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
    var_5 = 5
    var_6 = 10
    var_7 = '^test.*$'
    var_8 = 'email'
    var_9 = module_0.String(max_length=var_6, min_length=var_5, pattern=var_7, format=var_8)
    var_10 = module_1.to_json_schema(var_9)
    var_11 = False
    var_12 = 100
    var_13 = module_0.Integer(minimum=var_11, maximum=var_12, exclusive_minimum=var_11, exclusive_maximum=var_12, multiple_of=var_5)
    var_14 = module_1.to_json_schema(var_13)
    var_15 = 0.1
    var_16 = module_0.Float(minimum=var_11, maximum=var_4, multiple_of=var_15)
    var_17 = module_1.to_json_schema(var_16)
    var_18 = module_0.Boolean()
    var_19 = module_1.to_json_schema(var_18)
    var_20 = module_0.String()
    var_21 = module_0.Array(var_20, var_11, var_4, var_6, unique_items=var_4)
    var_22 = module_1.to_json_schema(var_21)
    var_23 = module_0.String()
    var_24 = module_0.Integer()
    var_25 = [var_23, var_24]
    var_26 = module_0.Boolean()
    var_27 = module_0.Array(var_25, var_26)
    var_28 = module_1.to_json_schema(var_27)
    var_29 = 'items'
    var_30 = var_28[var_29]
    var_31 = var_28[var_29]
    var_32 = len(var_31)
    assert var_32 == 2
    var_33 = 'name'
    var_34 = 'age'
    var_35 = module_0.String()
    var_36 = module_0.Integer()
    var_37 = {var_33: var_35, var_34: var_36}
    var_38 = module_0.Boolean()
    var_39 = {var_7: var_38}
    var_40 = '^[a-z]+$'
    var_41 = module_0.String(pattern=var_40)
    var_42 = [var_33]
    var_43 = module_0.Object(properties=var_37, pattern_properties=var_39, additional_properties=var_11, property_names=var_41, min_properties=var_4, max_properties=var_5, required=var_42)
    var_44 = module_1.to_json_schema(var_43)
    var_45 = module_0.String()
    var_46 = module_0.Integer()
    var_47 = {var_33: var_45, var_34: var_46}
    var_48 = [var_33]
    var_49 = module_3.Schema(var_47)
    var_50 = module_1.to_json_schema(var_49)
    var_51 = 'A'
    var_52 = (var_51, var_51)
    var_53 = 'B'
    var_54 = (var_53, var_53)
    var_55 = 'C'
    var_56 = (var_55, var_55)
    var_57 = [var_52, var_54, var_56]
    var_58 = module_0.Choice(choices=var_57)
    var_59 = module_1.to_json_schema(var_58)
    var_60 = 'fixed_value'
    var_61 = module_0.Const(var_60)
    var_62 = module_1.to_json_schema(var_61)
    var_63 = module_0.String()
    var_64 = module_0.Integer()
    var_65 = module_0.Boolean()
    var_66 = [var_63, var_64, var_65]
    var_67 = module_0.Union(var_66)
    var_68 = module_1.to_json_schema(var_67)
    var_69 = 'anyOf'
    var_70 = var_68[var_69]
    var_71 = len(var_70)
    assert var_71 == 3
    var_72 = module_0.String()
    var_73 = module_0.Integer()
    var_74 = [var_72, var_73]
    var_75 = module_2.OneOf(var_74)
    var_76 = module_1.to_json_schema(var_75)
    var_77 = 'oneOf'
    var_78 = var_76[var_77]
    var_79 = len(var_78)
    assert var_79 == 2
    var_80 = module_0.String(min_length=var_5)
    var_81 = module_0.String(max_length=var_6)
    var_82 = [var_80, var_81]
    var_83 = module_2.AllOf(var_82)
    var_84 = module_1.to_json_schema(var_83)
    var_85 = 'allOf'
    var_86 = var_84[var_85]
    var_87 = len(var_86)
    assert var_87 == 2
    var_88 = module_0.String(pattern=var_7)
    var_89 = module_0.Integer(minimum=var_11)
    var_90 = module_0.Boolean()
    var_91 = module_2.IfThenElse(var_88, var_89, var_90)
    var_92 = module_1.to_json_schema(var_91)
    var_93 = module_0.String(pattern=var_7)
    var_94 = module_2.Not(var_93)
    var_95 = module_1.to_json_schema(var_94)
    var_96 = module_3.Definitions()
    var_97 = module_0.String()
    var_98 = {var_33: var_97}
    var_99 = 'Person'
    var_100 = module_3.Reference(var_99, var_96)
    var_101 = module_1.to_json_schema(var_100)
    var_102 = module_3.Reference(var_99, var_96)
    var_103 = {}
    var_104 = module_1.to_json_schema(var_102, var_103)
    var_105 = 'default_value'
    var_106 = module_0.String()
    var_107 = module_1.to_json_schema(var_106)
    var_108 = module_1.to_json_schema(var_106)
    var_109 = module_1.to_json_schema(var_106)



# Parsed testcases at query #14
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'number'
    var_4 = 0
    var_5 = 10
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = False
    var_8 = module_0.Definitions()
    var_9 = module_1.from_json_schema_type(var_6, var_3, var_7, var_8)
    var_10 = 'exclusiveMinimum'
    var_11 = 'multipleOf'
    var_12 = 'integer'
    var_13 = 5
    var_14 = 2
    var_15 = {var_0: var_12, var_10: var_13, var_11: var_14}
    var_16 = True
    var_17 = module_0.Definitions()
    var_18 = module_1.from_json_schema_type(var_15, var_12, var_16, var_17)
    var_19 = 'minLength'
    var_20 = 'maxLength'
    var_21 = 'pattern'
    var_22 = 'string'
    var_23 = 3
    var_24 = '^a.*z$'
    var_25 = {var_0: var_22, var_19: var_23, var_20: var_5, var_21: var_24}
    var_26 = False
    var_27 = module_0.Definitions()
    var_28 = module_1.from_json_schema_type(var_25, var_22, var_26, var_27)
    var_29 = 'default'
    var_30 = 'boolean'
    var_31 = {var_0: var_30, var_29: var_16}
    var_32 = module_0.Definitions()
    var_33 = module_1.from_json_schema_type(var_31, var_30, var_16, var_32)
    var_34 = 'items'
    var_35 = 'minItems'
    var_36 = 'array'
    var_37 = {var_0: var_22}
    var_38 = {var_0: var_36, var_34: var_37, var_35: var_16}
    var_39 = False
    var_40 = module_0.Definitions()
    var_41 = module_1.from_json_schema_type(var_38, var_36, var_39, var_40)
    var_42 = var_41.items
    var_43 = {var_0: var_22}
    var_44 = {var_0: var_3}
    var_45 = [var_43, var_44]
    var_46 = {var_0: var_36, var_34: var_45}
    var_47 = False
    var_48 = module_0.Definitions()
    var_49 = module_1.from_json_schema_type(var_46, var_36, var_47, var_48)
    var_50 = var_49.items
    var_51 = var_49.items
    var_52 = len(var_51)
    assert var_52 == 2
    var_53 = var_49.items[var_47]
    var_54 = var_49.items[var_16]
    var_55 = 'properties'
    var_56 = 'required'
    var_57 = 'object'
    var_58 = 'name'
    var_59 = 'age'
    var_60 = {var_0: var_22}
    var_61 = {var_0: var_12}
    var_62 = {var_58: var_60, var_59: var_61}
    var_63 = [var_58]
    var_64 = {var_0: var_57, var_55: var_62, var_56: var_63}
    var_65 = False
    var_66 = module_0.Definitions()
    var_67 = module_1.from_json_schema_type(var_64, var_57, var_65, var_66)
    var_68 = var_67.properties[var_58]
    var_69 = var_67.properties[var_59]
    var_70 = 'patternProperties'
    var_71 = '^S_'
    var_72 = {var_0: var_22}
    var_73 = {var_71: var_72}
    var_74 = {var_0: var_57, var_70: var_73}
    var_75 = False
    var_76 = module_0.Definitions()
    var_77 = module_1.from_json_schema_type(var_74, var_57, var_75, var_76)
    var_78 = var_77.pattern_properties[var_71]
    var_79 = 'additionalProperties'
    var_80 = False
    var_81 = {var_0: var_57, var_79: var_80}
    var_82 = False
    var_83 = module_0.Definitions()
    var_84 = module_1.from_json_schema_type(var_81, var_57, var_82, var_83)
    var_85 = 'propertyNames'
    var_86 = '^[a-z]+$'
    var_87 = {var_21: var_86}
    var_88 = {var_0: var_57, var_85: var_87}
    var_89 = False
    var_90 = module_0.Definitions()
    var_91 = module_1.from_json_schema_type(var_88, var_57, var_89, var_90)
    var_92 = var_91.property_names
    var_93 = 'test'
    var_94 = {var_0: var_22, var_29: var_93}
    var_95 = False
    var_96 = module_0.Definitions()
    var_97 = module_1.from_json_schema_type(var_94, var_22, var_95, var_96)
    var_98 = {var_0: var_3}
    var_99 = module_0.Definitions()
    var_100 = module_1.from_json_schema_type(var_98, var_3, var_16, var_99)
    var_101 = 'additionalItems'
    var_102 = {var_0: var_22}
    var_103 = {var_0: var_36, var_101: var_102}
    var_104 = False
    var_105 = module_0.Definitions()
    var_106 = module_1.from_json_schema_type(var_103, var_36, var_104, var_105)
    var_107 = var_106.additional_items
    var_108 = {var_0: var_3}
    var_109 = {var_0: var_57, var_79: var_108}
    var_110 = False
    var_111 = module_0.Definitions()
    var_112 = module_1.from_json_schema_type(var_109, var_57, var_110, var_111)
    var_113 = var_112.additional_properties



# Parsed testcases at query #15
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'number'
    var_4 = 0
    var_5 = 10
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = False
    var_8 = module_0.Definitions()
    var_9 = module_1.from_json_schema_type(var_6, var_3, var_7, var_8)
    var_10 = 'exclusiveMinimum'
    var_11 = 'multipleOf'
    var_12 = 'integer'
    var_13 = 5
    var_14 = 2
    var_15 = {var_0: var_12, var_10: var_13, var_11: var_14}
    var_16 = True
    var_17 = module_0.Definitions()
    var_18 = module_1.from_json_schema_type(var_15, var_12, var_16, var_17)
    var_19 = 'minLength'
    var_20 = 'maxLength'
    var_21 = 'pattern'
    var_22 = 'string'
    var_23 = 3
    var_24 = '^a.*z$'
    var_25 = {var_0: var_22, var_19: var_23, var_20: var_5, var_21: var_24}
    var_26 = False
    var_27 = module_0.Definitions()
    var_28 = module_1.from_json_schema_type(var_25, var_22, var_26, var_27)
    var_29 = {var_0: var_22, var_19: var_26}
    var_30 = False
    var_31 = module_0.Definitions()
    var_32 = module_1.from_json_schema_type(var_29, var_22, var_30, var_31)
    var_33 = 'default'
    var_34 = 'boolean'
    var_35 = {var_0: var_34, var_33: var_16}
    var_36 = module_0.Definitions()
    var_37 = module_1.from_json_schema_type(var_35, var_34, var_16, var_36)
    var_38 = 'items'
    var_39 = 'minItems'
    var_40 = 'array'
    var_41 = {var_0: var_22}
    var_42 = {var_0: var_40, var_38: var_41, var_39: var_16}
    var_43 = False
    var_44 = module_0.Definitions()
    var_45 = module_1.from_json_schema_type(var_42, var_40, var_43, var_44)
    var_46 = var_45.items
    var_47 = {var_0: var_22}
    var_48 = {var_0: var_3}
    var_49 = [var_47, var_48]
    var_50 = {var_0: var_40, var_38: var_49}
    var_51 = False
    var_52 = module_0.Definitions()
    var_53 = module_1.from_json_schema_type(var_50, var_40, var_51, var_52)
    var_54 = var_53.items
    var_55 = var_53.items
    var_56 = len(var_55)
    assert var_56 == 2
    var_57 = var_53.items[var_51]
    var_58 = var_53.items[var_16]
    var_59 = 'additionalItems'
    var_60 = False
    var_61 = {var_0: var_40, var_59: var_60}
    var_62 = False
    var_63 = module_0.Definitions()
    var_64 = module_1.from_json_schema_type(var_61, var_40, var_62, var_63)
    var_65 = {var_0: var_12}
    var_66 = {var_0: var_40, var_59: var_65}
    var_67 = False
    var_68 = module_0.Definitions()
    var_69 = module_1.from_json_schema_type(var_66, var_40, var_67, var_68)
    var_70 = var_69.additional_items
    var_71 = 'properties'
    var_72 = 'object'
    var_73 = 'name'
    var_74 = {var_0: var_22}
    var_75 = {var_73: var_74}
    var_76 = {var_0: var_72, var_71: var_75}
    var_77 = False
    var_78 = module_0.Definitions()
    var_79 = module_1.from_json_schema_type(var_76, var_72, var_77, var_78)
    var_80 = var_79.properties[var_73]
    var_81 = 'patternProperties'
    var_82 = '^x_'
    var_83 = {var_0: var_3}
    var_84 = {var_82: var_83}
    var_85 = {var_0: var_72, var_81: var_84}
    var_86 = False
    var_87 = module_0.Definitions()
    var_88 = module_1.from_json_schema_type(var_85, var_72, var_86, var_87)
    var_89 = var_88.pattern_properties[var_82]
    var_90 = 'additionalProperties'
    var_91 = False
    var_92 = {var_0: var_72, var_90: var_91}
    var_93 = False
    var_94 = module_0.Definitions()
    var_95 = module_1.from_json_schema_type(var_92, var_72, var_93, var_94)
    var_96 = {var_0: var_34}
    var_97 = {var_0: var_72, var_90: var_96}
    var_98 = False
    var_99 = module_0.Definitions()
    var_100 = module_1.from_json_schema_type(var_97, var_72, var_98, var_99)
    var_101 = var_100.additional_properties
    var_102 = 'propertyNames'
    var_103 = '^[a-z]+$'
    var_104 = {var_21: var_103}
    var_105 = {var_0: var_72, var_102: var_104}
    var_106 = False
    var_107 = module_0.Definitions()
    var_108 = module_1.from_json_schema_type(var_105, var_72, var_106, var_107)
    var_109 = var_108.property_names
    var_110 = 'required'
    var_111 = 'id'
    var_112 = [var_111, var_73]
    var_113 = {var_0: var_72, var_110: var_112}
    var_114 = False
    var_115 = module_0.Definitions()
    var_116 = module_1.from_json_schema_type(var_113, var_72, var_114, var_115)
    var_117 = 'test'
    var_118 = {var_0: var_22, var_33: var_117}
    var_119 = False
    var_120 = module_0.Definitions()
    var_121 = module_1.from_json_schema_type(var_118, var_22, var_119, var_120)
    var_122 = {var_0: var_12}
    var_123 = False
    var_124 = module_0.Definitions()
    var_125 = module_1.from_json_schema_type(var_122, var_12, var_123, var_124)



# Parsed testcases at query #16
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
    var_17 = 'array'
    var_18 = {var_4: var_17}
    var_19 = module_0.from_json_schema(var_18)
    var_20 = 'object'
    var_21 = {var_4: var_20}
    var_22 = module_0.from_json_schema(var_21)
    var_23 = 'enum'
    var_24 = 'a'
    var_25 = 'b'
    var_26 = 'c'
    var_27 = [var_24, var_25, var_26]
    var_28 = {var_23: var_27}
    var_29 = module_0.from_json_schema(var_28)
    var_30 = 'const'
    var_31 = 'fixed_value'
    var_32 = {var_30: var_31}
    var_33 = module_0.from_json_schema(var_32)
    var_34 = 'allOf'
    var_35 = {var_4: var_5}
    var_36 = 'minLength'
    var_37 = 3
    var_38 = {var_36: var_37}
    var_39 = [var_35, var_38]
    var_40 = {var_34: var_39}
    var_41 = module_0.from_json_schema(var_40)
    var_42 = 'anyOf'
    var_43 = {var_4: var_5}
    var_44 = {var_4: var_8}
    var_45 = [var_43, var_44]
    var_46 = {var_42: var_45}
    var_47 = module_0.from_json_schema(var_46)
    var_48 = 'oneOf'
    var_49 = {var_4: var_5}
    var_50 = {var_4: var_8}
    var_51 = [var_49, var_50]
    var_52 = {var_48: var_51}
    var_53 = module_0.from_json_schema(var_52)
    var_54 = 'not'
    var_55 = {var_4: var_5}
    var_56 = {var_54: var_55}
    var_57 = module_0.from_json_schema(var_56)
    var_58 = 'if'
    var_59 = 'then'
    var_60 = 'else'
    var_61 = {var_4: var_5}
    var_62 = 5
    var_63 = {var_36: var_62}
    var_64 = {var_4: var_8}
    var_65 = {var_58: var_61, var_59: var_63, var_60: var_64}
    var_66 = module_0.from_json_schema(var_65)
    var_67 = '$ref'
    var_68 = '#/components/schemas/User'
    var_69 = {var_67: var_68}
    var_70 = module_0.from_json_schema(var_69)
    var_71 = 'maxLength'
    var_72 = 'pattern'
    var_73 = 10
    var_74 = '^[a-z]+$'
    var_75 = {var_4: var_5, var_36: var_37, var_71: var_73, var_72: var_74}
    var_76 = module_0.from_json_schema(var_75)
    var_77 = {}
    var_78 = module_0.from_json_schema(var_77)
    var_79 = module_1.Definitions()
    var_80 = {var_4: var_5}
    var_81 = module_0.from_json_schema(var_80, var_79)
    var_82 = 'components'
    var_83 = 'schemas'
    var_84 = 'User'
    var_85 = {var_4: var_5}
    var_86 = {var_84: var_85}
    var_87 = {var_83: var_86}
    var_88 = {var_82: var_87}
    var_89 = module_0.from_json_schema(var_88)



# Parsed testcases at query #17
#--------------------------


import typesystem.json_schema as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)
    var_2 = False
    var_3 = module_0.from_json_schema(var_2)
    var_4 = module_1.Definitions()
    var_5 = '$ref'
    var_6 = '#/components/schemas/Test'
    var_7 = {var_5: var_6}
    var_8 = module_0.from_json_schema(var_7, var_4)
    var_9 = 'type'
    var_10 = 'string'
    var_11 = {var_9: var_10}
    var_12 = module_0.from_json_schema(var_11)
    var_13 = 'minLength'
    var_14 = 5
    var_15 = {var_9: var_10, var_13: var_14}
    var_16 = module_0.from_json_schema(var_15)
    var_17 = 'integer'
    var_18 = {var_9: var_17}
    var_19 = module_0.from_json_schema(var_18)
    var_20 = 'minimum'
    var_21 = {var_9: var_17, var_20: var_2}
    var_22 = module_0.from_json_schema(var_21)
    var_23 = 'number'
    var_24 = {var_9: var_23}
    var_25 = module_0.from_json_schema(var_24)
    var_26 = 'maximum'
    var_27 = 100
    var_28 = {var_9: var_23, var_26: var_27}
    var_29 = module_0.from_json_schema(var_28)
    var_30 = 'boolean'
    var_31 = {var_9: var_30}
    var_32 = module_0.from_json_schema(var_31)
    var_33 = 'array'
    var_34 = {var_9: var_33}
    var_35 = module_0.from_json_schema(var_34)
    var_36 = 'items'
    var_37 = {var_9: var_10}
    var_38 = {var_9: var_33, var_36: var_37}
    var_39 = module_0.from_json_schema(var_38)
    var_40 = var_39.items
    var_41 = 'object'
    var_42 = {var_9: var_41}
    var_43 = module_0.from_json_schema(var_42)
    var_44 = 'properties'
    var_45 = 'name'
    var_46 = {var_9: var_10}
    var_47 = {var_45: var_46}
    var_48 = {var_9: var_41, var_44: var_47}
    var_49 = module_0.from_json_schema(var_48)
    var_50 = var_49.properties[var_45]
    var_51 = 'enum'
    var_52 = 'a'
    var_53 = 'b'
    var_54 = 'c'
    var_55 = [var_52, var_53, var_54]
    var_56 = {var_51: var_55}
    var_57 = module_0.from_json_schema(var_56)
    var_58 = 'const'
    var_59 = 'fixed_value'
    var_60 = {var_58: var_59}
    var_61 = module_0.from_json_schema(var_60)
    var_62 = 'allOf'
    var_63 = {var_9: var_10, var_13: var_14}
    var_64 = 'maxLength'
    var_65 = 10
    var_66 = {var_9: var_10, var_64: var_65}
    var_67 = [var_63, var_66]
    var_68 = {var_62: var_67}
    var_69 = module_0.from_json_schema(var_68)
    var_70 = var_69.fields
    var_71 = len(var_70)
    assert var_71 == 2
    var_72 = 'anyOf'
    var_73 = {var_9: var_10}
    var_74 = {var_9: var_17}
    var_75 = [var_73, var_74]
    var_76 = {var_72: var_75}
    var_77 = module_0.from_json_schema(var_76)
    var_78 = var_77.any_of
    var_79 = len(var_78)
    assert var_79 == 2
    var_80 = 'oneOf'
    var_81 = {var_9: var_10}
    var_82 = {var_9: var_17}
    var_83 = [var_81, var_82]
    var_84 = {var_80: var_83}
    var_85 = module_0.from_json_schema(var_84)
    var_86 = var_85.one_of
    var_87 = len(var_86)
    assert var_87 == 2
    var_88 = 'not'
    var_89 = {var_9: var_10}
    var_90 = {var_88: var_89}
    var_91 = module_0.from_json_schema(var_90)
    var_92 = var_91.negated
    var_93 = 'if'
    var_94 = 'then'
    var_95 = 'else'
    var_96 = {var_9: var_10}
    var_97 = {var_13: var_14}
    var_98 = {var_9: var_17}
    var_99 = {var_93: var_96, var_94: var_97, var_95: var_98}
    var_100 = module_0.from_json_schema(var_99)
    var_101 = var_100.if_clause
    var_102 = var_100.then_clause
    var_103 = var_100.else_clause
    var_104 = [var_52, var_53, var_54]
    var_105 = {var_9: var_10, var_51: var_104, var_13: var_0}
    var_106 = module_0.from_json_schema(var_105)
    var_107 = var_106.fields
    var_108 = len(var_107)
    assert var_108 == 2
    var_109 = {}
    var_110 = module_0.from_json_schema(var_109)
    var_111 = 'components'
    var_112 = 'schemas'
    var_113 = 'TestSchema'
    var_114 = {var_9: var_10, var_13: var_14}
    var_115 = {var_113: var_114}
    var_116 = {var_112: var_115}
    var_117 = {var_111: var_116}
    var_118 = module_0.from_json_schema(var_117)



# Parsed testcases at query #18
#--------------------------


import typesystem.json_schema as module_0
import re as module_1
import typesystem.schemas as module_2

def test_case_0():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)
    var_2 = False
    var_3 = module_0.from_json_schema(var_2)
    var_4 = 'type'
    var_5 = 'string'
    var_6 = {var_4: var_5}
    var_7 = module_0.from_json_schema(var_6)
    var_8 = 'number'
    var_9 = {var_4: var_8}
    var_10 = module_0.from_json_schema(var_9)
    var_11 = 'integer'
    var_12 = {var_4: var_11}
    var_13 = module_0.from_json_schema(var_12)
    var_14 = 'boolean'
    var_15 = {var_4: var_14}
    var_16 = module_0.from_json_schema(var_15)
    var_17 = 'array'
    var_18 = {var_4: var_17}
    var_19 = module_0.from_json_schema(var_18)
    var_20 = 'object'
    var_21 = {var_4: var_20}
    var_22 = module_0.from_json_schema(var_21)
    var_23 = 'minLength'
    var_24 = 5
    var_25 = {var_4: var_5, var_23: var_24}
    var_26 = module_0.from_json_schema(var_25)
    var_27 = 'enum'
    var_28 = 'a'
    var_29 = 'b'
    var_30 = 'c'
    var_31 = [var_28, var_29, var_30]
    var_32 = {var_27: var_31}
    var_33 = module_0.from_json_schema(var_32)
    var_34 = 'const'
    var_35 = 'fixed_value'
    var_36 = {var_34: var_35}
    var_37 = module_0.from_json_schema(var_36)
    var_38 = 'allOf'
    var_39 = 3
    var_40 = {var_4: var_5, var_23: var_39}
    var_41 = 'maxLength'
    var_42 = 10
    var_43 = {var_4: var_5, var_41: var_42}
    var_44 = [var_40, var_43]
    var_45 = {var_38: var_44}
    var_46 = module_0.from_json_schema(var_45)
    var_47 = 'anyOf'
    var_48 = {var_4: var_5}
    var_49 = {var_4: var_8}
    var_50 = [var_48, var_49]
    var_51 = {var_47: var_50}
    var_52 = module_0.from_json_schema(var_51)
    var_53 = 'oneOf'
    var_54 = {var_4: var_5}
    var_55 = {var_4: var_11}
    var_56 = [var_54, var_55]
    var_57 = {var_53: var_56}
    var_58 = module_0.from_json_schema(var_57)
    var_59 = 'not'
    var_60 = {var_4: var_5}
    var_61 = {var_59: var_60}
    var_62 = module_0.from_json_schema(var_61)
    var_63 = 'if'
    var_64 = 'then'
    var_65 = 'else'
    var_66 = {var_4: var_5}
    var_67 = {var_23: var_24}
    var_68 = {var_4: var_8}
    var_69 = {var_63: var_66, var_64: var_67, var_65: var_68}
    var_70 = module_0.from_json_schema(var_69)
    var_71 = '$ref'
    var_72 = '#/components/schemas/User'
    var_73 = {var_71: var_72}
    var_74 = module_0.from_json_schema(var_73)
    var_75 = 'pattern'
    var_76 = '^[a-z]+$'
    var_77 = {var_4: var_5, var_23: var_39, var_41: var_42, var_75: var_76}
    var_78 = module_0.from_json_schema(var_77)
    var_79 = module_1.compile(var_76)
    var_80 = {}
    var_81 = module_0.from_json_schema(var_80)
    var_82 = module_2.Definitions()
    var_83 = 'components'
    var_84 = 'schemas'
    var_85 = 'User'
    var_86 = 'properties'
    var_87 = 'name'
    var_88 = {var_4: var_5}
    var_89 = {var_87: var_88}
    var_90 = {var_4: var_20, var_86: var_89}
    var_91 = {var_85: var_90}
    var_92 = {var_84: var_91}
    var_93 = {var_83: var_92}
    var_94 = module_0.from_json_schema(var_93, var_82)



# Parsed testcases at query #19
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
    var_4 = False
    var_5 = 1
    var_6 = 10
    var_7 = module_0.String(max_length=var_6, min_length=var_5)
    var_8 = module_1.to_json_schema(var_7)
    var_9 = True
    var_10 = module_0.String()
    var_11 = module_1.to_json_schema(var_10)
    var_12 = 100
    var_13 = module_0.Integer(minimum=var_4, maximum=var_12)
    var_14 = module_1.to_json_schema(var_13)
    var_15 = module_0.Float(minimum=var_4, maximum=var_9)
    var_16 = module_1.to_json_schema(var_15)
    var_17 = module_0.Boolean()
    var_18 = module_1.to_json_schema(var_17)
    var_19 = module_0.String()
    var_20 = module_0.Array(var_19, min_items=var_9, max_items=var_6)
    var_21 = module_1.to_json_schema(var_20)
    var_22 = 'name'
    var_23 = module_0.String()
    var_24 = {var_22: var_23}
    var_25 = [var_22]
    var_26 = 5
    var_27 = module_0.Object(properties=var_24, min_properties=var_9, max_properties=var_26, required=var_25)
    var_28 = module_1.to_json_schema(var_27)
    var_29 = 'a'
    var_30 = 'A'
    var_31 = (var_29, var_30)
    var_32 = 'b'
    var_33 = 'B'
    var_34 = (var_32, var_33)
    var_35 = [var_31, var_34]
    var_36 = module_0.Choice(choices=var_35)
    var_37 = module_1.to_json_schema(var_36)
    var_38 = 'fixed_value'
    var_39 = module_0.Const(var_38)
    var_40 = module_1.to_json_schema(var_39)
    var_41 = module_0.String()
    var_42 = module_0.Integer()
    var_43 = [var_41, var_42]
    var_44 = module_0.Union(var_43)
    var_45 = module_1.to_json_schema(var_44)
    var_46 = 'anyOf'
    var_47 = var_45[var_46]
    var_48 = len(var_47)
    assert var_48 == 2
    var_49 = module_0.String(min_length=var_9)
    var_50 = module_0.String(max_length=var_6)
    var_51 = [var_49, var_50]
    var_52 = module_2.AllOf(var_51)
    var_53 = module_1.to_json_schema(var_52)
    var_54 = 'allOf'
    var_55 = var_53[var_54]
    var_56 = len(var_55)
    assert var_56 == 2
    var_57 = module_0.String()
    var_58 = module_0.Integer()
    var_59 = [var_57, var_58]
    var_60 = module_2.OneOf(var_59)
    var_61 = module_1.to_json_schema(var_60)
    var_62 = 'oneOf'
    var_63 = var_61[var_62]
    var_64 = len(var_63)
    assert var_64 == 2
    var_65 = module_3.Definitions()
    var_66 = module_0.String()
    var_67 = {var_22: var_66}
    var_68 = 'Person'
    var_69 = module_3.Reference(var_68, var_65)
    var_70 = module_1.to_json_schema(var_69)
    var_71 = module_3.Definitions()
    var_72 = 'id'
    var_73 = module_0.Integer()
    var_74 = {var_72: var_73}
    var_75 = 'user'
    var_76 = 'User'
    var_77 = module_3.Reference(var_76, var_71)
    var_78 = {var_75: var_77}
    var_79 = module_1.to_json_schema(var_71)
    var_80 = 'default_value'
    var_81 = module_0.String()
    var_82 = module_1.to_json_schema(var_81)
    var_83 = '^test_'
    var_84 = module_0.String()
    var_85 = {var_83: var_84}
    var_86 = module_0.Object(pattern_properties=var_85)
    var_87 = module_1.to_json_schema(var_86)
    var_88 = module_0.Object(additional_properties=var_4)
    var_89 = module_1.to_json_schema(var_88)
    var_90 = '^[a-z]+$'
    var_91 = module_0.String(pattern=var_90)
    var_92 = module_0.Object(property_names=var_91)
    var_93 = module_1.to_json_schema(var_92)
    var_94 = module_0.Float(exclusive_minimum=var_4, exclusive_maximum=var_12)
    var_95 = module_1.to_json_schema(var_94)
    var_96 = 2
    var_97 = module_0.Integer(multiple_of=var_96)
    var_98 = module_1.to_json_schema(var_97)
    var_99 = module_0.String()
    var_100 = True
    var_101 = module_0.Array(var_99, unique_items=var_100)
    var_102 = module_1.to_json_schema(var_101)
    var_103 = 'email'
    var_104 = module_0.String(format=var_103)
    var_105 = module_1.to_json_schema(var_104)
    var_106 = module_0.String()
    var_107 = module_0.Integer()
    var_108 = module_0.Boolean()
    var_109 = module_2.IfThenElse(var_106, var_107, var_108)
    var_110 = module_1.to_json_schema(var_109)
    var_111 = module_0.String()
    var_112 = module_2.Not(var_111)
    var_113 = module_1.to_json_schema(var_112)
    var_114 = module_0.Integer()
    var_115 = {var_72: var_114}
    var_116 = [var_72]
    var_117 = module_3.Schema(var_115)
    var_118 = module_1.to_json_schema(var_117)
    var_119 = module_0.Decimal(minimum=var_4, maximum=var_12)
    var_120 = module_1.to_json_schema(var_119)
    var_121 = module_3.Definitions()
    var_122 = 'inner'
    var_123 = 'Inner'
    var_124 = module_3.Reference(var_123, var_121)
    var_125 = {var_122: var_124}
    var_126 = module_0.Object(properties=var_125)
    var_127 = module_1.to_json_schema(var_126, var_121)
    var_128 = 'components'
    var_129 = {}
    var_130 = 'schemas'
    var_131 = {}



# Parsed testcases at query #20
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
    var_17 = 'array'
    var_18 = {var_4: var_17}
    var_19 = module_0.from_json_schema(var_18)
    var_20 = 'object'
    var_21 = {var_4: var_20}
    var_22 = module_0.from_json_schema(var_21)
    var_23 = 'enum'
    var_24 = 'a'
    var_25 = 'b'
    var_26 = 'c'
    var_27 = [var_24, var_25, var_26]
    var_28 = {var_23: var_27}
    var_29 = module_0.from_json_schema(var_28)
    var_30 = 'const'
    var_31 = 'fixed_value'
    var_32 = {var_30: var_31}
    var_33 = module_0.from_json_schema(var_32)
    var_34 = 'allOf'
    var_35 = {var_4: var_5}
    var_36 = 'minLength'
    var_37 = {var_36: var_0}
    var_38 = [var_35, var_37]
    var_39 = {var_34: var_38}
    var_40 = module_0.from_json_schema(var_39)
    var_41 = 'anyOf'
    var_42 = {var_4: var_5}
    var_43 = {var_4: var_8}
    var_44 = [var_42, var_43]
    var_45 = {var_41: var_44}
    var_46 = module_0.from_json_schema(var_45)
    var_47 = 'oneOf'
    var_48 = {var_4: var_5}
    var_49 = {var_4: var_8}
    var_50 = [var_48, var_49]
    var_51 = {var_47: var_50}
    var_52 = module_0.from_json_schema(var_51)
    var_53 = 'not'
    var_54 = {var_4: var_5}
    var_55 = {var_53: var_54}
    var_56 = module_0.from_json_schema(var_55)
    var_57 = 'if'
    var_58 = 'then'
    var_59 = 'else'
    var_60 = {var_4: var_5}
    var_61 = {var_36: var_0}
    var_62 = {var_4: var_8}
    var_63 = {var_57: var_60, var_58: var_61, var_59: var_62}
    var_64 = module_0.from_json_schema(var_63)
    var_65 = 'maxLength'
    var_66 = 10
    var_67 = {var_4: var_5, var_36: var_0, var_65: var_66}
    var_68 = module_0.from_json_schema(var_67)
    var_69 = {}
    var_70 = module_0.from_json_schema(var_69)
    var_71 = module_1.Definitions()
    var_72 = '$ref'
    var_73 = '#/definitions/MyType'
    var_74 = {var_72: var_73}
    var_75 = module_0.from_json_schema(var_74, var_71)
    var_76 = 'components'
    var_77 = 'schemas'
    var_78 = 'User'
    var_79 = 'properties'
    var_80 = 'name'
    var_81 = {var_4: var_5}
    var_82 = {var_80: var_81}
    var_83 = {var_4: var_20, var_79: var_82}
    var_84 = {var_78: var_83}
    var_85 = {var_77: var_84}
    var_86 = {var_76: var_85}
    var_87 = module_0.from_json_schema(var_86)



# Parsed testcases at query #21
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
    var_4 = False
    var_5 = 1
    var_6 = 10
    var_7 = '^[a-z]+$'
    var_8 = module_0.String(max_length=var_6, min_length=var_5, pattern=var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = True
    var_11 = True
    var_12 = module_0.String(allow_blank=var_11)
    var_13 = module_1.to_json_schema(var_12)
    var_14 = 100
    var_15 = True
    var_16 = module_0.Integer(minimum=var_4, maximum=var_14, exclusive_minimum=var_15, exclusive_maximum=var_4)
    var_17 = module_1.to_json_schema(var_16)
    var_18 = 0.5
    var_19 = module_0.Float(multiple_of=var_18)
    var_20 = module_1.to_json_schema(var_19)
    var_21 = True
    var_22 = module_0.Boolean()
    var_23 = module_1.to_json_schema(var_22)
    var_24 = module_0.String()
    var_25 = True
    var_26 = module_0.Array(var_24, min_items=var_21, max_items=var_6, unique_items=var_25)
    var_27 = module_1.to_json_schema(var_26)
    var_28 = module_0.String()
    var_29 = module_0.Integer()
    var_30 = [var_28, var_29]
    var_31 = module_0.Array(var_30)
    var_32 = module_1.to_json_schema(var_31)
    var_33 = 'items'
    var_34 = var_32[var_33]
    var_35 = var_32[var_33]
    var_36 = len(var_35)
    assert var_36 == 2
    var_37 = 'name'
    var_38 = 'age'
    var_39 = module_0.String()
    var_40 = module_0.Integer()
    var_41 = {var_37: var_39, var_38: var_40}
    var_42 = [var_37]
    var_43 = 2
    var_44 = module_0.Object(properties=var_41, min_properties=var_25, max_properties=var_43, required=var_42)
    var_45 = module_1.to_json_schema(var_44)
    var_46 = module_0.Integer()
    var_47 = {var_7: var_46}
    var_48 = module_0.Object(pattern_properties=var_47)
    var_49 = module_1.to_json_schema(var_48)
    var_50 = 'A'
    var_51 = 'Option A'
    var_52 = (var_50, var_51)
    var_53 = 'B'
    var_54 = 'Option B'
    var_55 = (var_53, var_54)
    var_56 = [var_52, var_55]
    var_57 = module_0.Choice(choices=var_56)
    var_58 = module_1.to_json_schema(var_57)
    var_59 = 'fixed_value'
    var_60 = module_0.Const(var_59)
    var_61 = module_1.to_json_schema(var_60)
    var_62 = module_0.String()
    var_63 = module_0.Integer()
    var_64 = [var_62, var_63]
    var_65 = module_0.Union(var_64)
    var_66 = module_1.to_json_schema(var_65)
    var_67 = 'anyOf'
    var_68 = var_66[var_67]
    var_69 = len(var_68)
    assert var_69 == 2
    var_70 = module_0.String()
    var_71 = module_0.Integer()
    var_72 = [var_70, var_71]
    var_73 = module_2.OneOf(var_72)
    var_74 = module_1.to_json_schema(var_73)
    var_75 = 'oneOf'
    var_76 = var_74[var_75]
    var_77 = len(var_76)
    assert var_77 == 2
    var_78 = module_0.String(min_length=var_25)
    var_79 = module_0.String(max_length=var_6)
    var_80 = [var_78, var_79]
    var_81 = module_2.AllOf(var_80)
    var_82 = module_1.to_json_schema(var_81)
    var_83 = 'allOf'
    var_84 = var_82[var_83]
    var_85 = len(var_84)
    assert var_85 == 2
    var_86 = 5
    var_87 = module_0.String(min_length=var_86)
    var_88 = module_0.Integer(minimum=var_6)
    var_89 = module_0.Boolean()
    var_90 = module_2.IfThenElse(var_87, var_88, var_89)
    var_91 = module_1.to_json_schema(var_90)
    var_92 = module_0.String()
    var_93 = module_2.Not(var_92)
    var_94 = module_1.to_json_schema(var_93)
    var_95 = module_3.Definitions()
    var_96 = module_0.String()
    var_97 = {var_37: var_96}
    var_98 = 'Person'
    var_99 = module_3.Reference(var_98, var_95)
    var_100 = module_1.to_json_schema(var_99)
    var_101 = module_0.String()
    var_102 = {var_37: var_101}
    var_103 = module_0.Object(properties=var_102)
    var_104 = {var_98: var_103}
    var_105 = module_1.to_json_schema(var_103, var_104)
    var_106 = module_1.to_json_schema(var_95)
    var_107 = 'test'
    var_108 = module_0.String()
    var_109 = module_1.to_json_schema(var_108)
    var_110 = module_0.String()
    var_111 = module_0.Integer()
    var_112 = module_1.to_json_schema(var_108)
    var_113 = module_0.Decimal(minimum=var_4, maximum=var_14)
    var_114 = module_1.to_json_schema(var_113)
    var_115 = module_1.to_json_schema(var_113)



# Parsed testcases at query #22
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
    var_4 = False
    var_5 = 1
    var_6 = 10
    var_7 = '^[a-z]+$'
    var_8 = module_0.String(max_length=var_6, min_length=var_5, pattern=var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = True
    var_11 = module_0.String()
    var_12 = module_1.to_json_schema(var_11)
    var_13 = 100
    var_14 = True
    var_15 = module_0.Integer(minimum=var_4, maximum=var_13, exclusive_minimum=var_14, exclusive_maximum=var_4)
    var_16 = module_1.to_json_schema(var_15)
    var_17 = 0.5
    var_18 = module_0.Float(multiple_of=var_17)
    var_19 = module_1.to_json_schema(var_18)
    var_20 = True
    var_21 = module_0.Boolean()
    var_22 = module_1.to_json_schema(var_21)
    var_23 = module_0.String()
    var_24 = True
    var_25 = module_0.Array(var_23, var_4, var_20, var_6, unique_items=var_24)
    var_26 = module_1.to_json_schema(var_25)
    var_27 = 'name'
    var_28 = 'age'
    var_29 = module_0.String()
    var_30 = module_0.Integer()
    var_31 = {var_27: var_29, var_28: var_30}
    var_32 = [var_27]
    var_33 = 5
    var_34 = module_0.Object(properties=var_31, min_properties=var_24, max_properties=var_33, required=var_32)
    var_35 = module_1.to_json_schema(var_34)
    var_36 = '^test_'
    var_37 = module_0.String()
    var_38 = {var_36: var_37}
    var_39 = module_0.Object(pattern_properties=var_38, additional_properties=var_4)
    var_40 = module_1.to_json_schema(var_39)
    var_41 = 'A'
    var_42 = 'Option A'
    var_43 = (var_41, var_42)
    var_44 = 'B'
    var_45 = 'Option B'
    var_46 = (var_44, var_45)
    var_47 = [var_43, var_46]
    var_48 = module_0.Choice(choices=var_47)
    var_49 = module_1.to_json_schema(var_48)
    var_50 = 'fixed_value'
    var_51 = module_0.Const(var_50)
    var_52 = module_1.to_json_schema(var_51)
    var_53 = module_0.String()
    var_54 = module_0.Integer()
    var_55 = [var_53, var_54]
    var_56 = module_0.Union(var_55)
    var_57 = module_1.to_json_schema(var_56)
    var_58 = 'anyOf'
    var_59 = var_57[var_58]
    var_60 = len(var_59)
    assert var_60 == 2
    var_61 = module_0.String()
    var_62 = module_0.Integer()
    var_63 = [var_61, var_62]
    var_64 = module_2.OneOf(var_63)
    var_65 = module_1.to_json_schema(var_64)
    var_66 = 'oneOf'
    var_67 = var_65[var_66]
    var_68 = len(var_67)
    assert var_68 == 2
    var_69 = module_0.String(min_length=var_24)
    var_70 = module_0.String(max_length=var_6)
    var_71 = [var_69, var_70]
    var_72 = module_2.AllOf(var_71)
    var_73 = module_1.to_json_schema(var_72)
    var_74 = 'allOf'
    var_75 = var_73[var_74]
    var_76 = len(var_75)
    assert var_76 == 2
    var_77 = module_0.String()
    var_78 = module_0.Integer()
    var_79 = module_0.Boolean()
    var_80 = module_2.IfThenElse(var_77, var_78, var_79)
    var_81 = module_1.to_json_schema(var_80)
    var_82 = module_0.String()
    var_83 = module_2.Not(var_82)
    var_84 = module_1.to_json_schema(var_83)
    var_85 = 'User'
    var_86 = module_0.String()
    var_87 = {var_27: var_86}
    var_88 = module_0.Object(properties=var_87)
    var_89 = {var_85: var_88}
    var_90 = module_0.String()
    var_91 = {var_27: var_90}
    var_92 = [var_27]
    var_93 = module_3.Schema(var_91)
    var_94 = module_1.to_json_schema(var_93)
    var_95 = 'default_value'
    var_96 = module_0.String()
    var_97 = module_1.to_json_schema(var_96)
    var_98 = module_0.String()
    var_99 = {var_27: var_98}
    var_100 = module_0.Object(properties=var_99)
    var_101 = {var_85: var_100}
    var_102 = module_1.to_json_schema(var_100, var_101)



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = '#/components/schemas/User'
    var_3 = {var_1: var_2}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)
    var_5 = '#/definitions/Address'
    var_6 = {var_1: var_5}
    var_7 = module_1.ref_from_json_schema(var_6, var_0)
    var_8 = '$ref'
    var_9 = 'http://example.com/schema.json'
    var_10 = {var_8: var_9}
    var_11 = module_1.ref_from_json_schema(var_10, var_0)
    var_12 = module_0.Definitions()
    var_13 = '#/components/schemas/Product'
    var_14 = {var_8: var_13}
    var_15 = module_1.ref_from_json_schema(var_14, var_12)



# Parsed testcases at query #2
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1
import typesystem.fields as module_2

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'type'
    var_4 = 'minLength'
    var_5 = 'string'
    var_6 = 5
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'pattern'
    var_9 = '^[A-Z].*'
    var_10 = {var_3: var_5, var_8: var_9}
    var_11 = {var_1: var_7, var_2: var_10}
    var_12 = module_1.if_then_else_from_json_schema(var_11, var_0)
    var_13 = var_12.if_clause
    var_14 = var_12.then_clause
    var_15 = 'else'
    var_16 = 'minimum'
    var_17 = 'number'
    var_18 = 0
    var_19 = {var_3: var_17, var_16: var_18}
    var_20 = 'maximum'
    var_21 = 100
    var_22 = {var_3: var_17, var_20: var_21}
    var_23 = -1
    var_24 = {var_3: var_17, var_20: var_23}
    var_25 = {var_1: var_19, var_2: var_22, var_15: var_24}
    var_26 = module_1.if_then_else_from_json_schema(var_25, var_0)
    var_27 = var_26.if_clause
    var_28 = var_26.then_clause
    var_29 = var_26.else_clause
    var_30 = 'allOf'
    var_31 = 'object'
    var_32 = {var_3: var_31}
    var_33 = 'required'
    var_34 = 'active'
    var_35 = [var_34]
    var_36 = {var_33: var_35}
    var_37 = [var_32, var_36]
    var_38 = {var_30: var_37}
    var_39 = 'properties'
    var_40 = 'status'
    var_41 = {var_3: var_5}
    var_42 = {var_40: var_41}
    var_43 = {var_3: var_31, var_39: var_42}
    var_44 = 'error'
    var_45 = {var_3: var_5}
    var_46 = {var_44: var_45}
    var_47 = {var_3: var_31, var_39: var_46}
    var_48 = {var_1: var_38, var_2: var_43, var_15: var_47}
    var_49 = module_1.if_then_else_from_json_schema(var_48, var_0)
    var_50 = var_49.if_clause
    var_51 = var_49.then_clause
    var_52 = var_49.else_clause
    var_53 = 'default'
    var_54 = 'boolean'
    var_55 = {var_3: var_54}
    var_56 = 'const'
    var_57 = 'yes'
    var_58 = {var_3: var_5, var_56: var_57}
    var_59 = 'no'
    var_60 = {var_3: var_5, var_56: var_59}
    var_61 = 'maybe'
    var_62 = {var_1: var_55, var_2: var_58, var_15: var_60, var_53: var_61}
    var_63 = module_1.if_then_else_from_json_schema(var_62, var_0)
    var_64 = 'minItems'
    var_65 = 'array'
    var_66 = 1
    var_67 = {var_3: var_65, var_64: var_66}
    var_68 = {var_1: var_67}
    var_69 = module_1.if_then_else_from_json_schema(var_68, var_0)
    var_70 = var_69.if_clause
    var_71 = 'anyOf'
    var_72 = {var_3: var_5}
    var_73 = {var_3: var_17}
    var_74 = [var_72, var_73]
    var_75 = {var_71: var_74}
    var_76 = 'oneOf'
    var_77 = {var_3: var_5}
    var_78 = 'integer'
    var_79 = {var_3: var_78}
    var_80 = [var_77, var_79]
    var_81 = {var_76: var_80}
    var_82 = 'not'
    var_83 = {var_3: var_54}
    var_84 = {var_82: var_83}
    var_85 = {var_1: var_75, var_2: var_81, var_15: var_84}
    var_86 = module_1.if_then_else_from_json_schema(var_85, var_0)
    var_87 = var_86.if_clause
    var_88 = var_86.then_clause
    var_89 = var_86.else_clause
    var_90 = 'name'
    var_91 = module_2.String()
    var_92 = {var_90: var_91}
    var_93 = '$ref'
    var_94 = '#/components/schemas/User'
    var_95 = {var_93: var_94}
    var_96 = {var_3: var_5}
    var_97 = {var_3: var_17}
    var_98 = {var_1: var_95, var_2: var_96, var_15: var_97}
    var_99 = module_1.if_then_else_from_json_schema(var_98, var_0)
    var_100 = var_99.if_clause
    var_101 = var_99.then_clause
    var_102 = var_99.else_clause



# Parsed testcases at query #3
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
    var_17 = 'array'
    var_18 = {var_4: var_17}
    var_19 = module_0.from_json_schema(var_18)
    var_20 = 'object'
    var_21 = {var_4: var_20}
    var_22 = module_0.from_json_schema(var_21)
    var_23 = 'enum'
    var_24 = 'a'
    var_25 = 'b'
    var_26 = 'c'
    var_27 = [var_24, var_25, var_26]
    var_28 = {var_23: var_27}
    var_29 = module_0.from_json_schema(var_28)
    var_30 = 'const'
    var_31 = 'fixed_value'
    var_32 = {var_30: var_31}
    var_33 = module_0.from_json_schema(var_32)
    var_34 = 'allOf'
    var_35 = {var_4: var_5}
    var_36 = 'minLength'
    var_37 = 5
    var_38 = {var_36: var_37}
    var_39 = [var_35, var_38]
    var_40 = {var_34: var_39}
    var_41 = module_0.from_json_schema(var_40)
    var_42 = 'anyOf'
    var_43 = {var_4: var_5}
    var_44 = {var_4: var_8}
    var_45 = [var_43, var_44]
    var_46 = {var_42: var_45}
    var_47 = module_0.from_json_schema(var_46)
    var_48 = 'oneOf'
    var_49 = {var_4: var_5}
    var_50 = {var_4: var_8}
    var_51 = [var_49, var_50]
    var_52 = {var_48: var_51}
    var_53 = module_0.from_json_schema(var_52)
    var_54 = 'not'
    var_55 = {var_4: var_5}
    var_56 = {var_54: var_55}
    var_57 = module_0.from_json_schema(var_56)
    var_58 = 'if'
    var_59 = 'then'
    var_60 = 'else'
    var_61 = {var_4: var_5}
    var_62 = {var_36: var_37}
    var_63 = {var_4: var_8}
    var_64 = {var_58: var_61, var_59: var_62, var_60: var_63}
    var_65 = module_0.from_json_schema(var_64)
    var_66 = 'maxLength'
    var_67 = 10
    var_68 = {var_4: var_5, var_36: var_37, var_66: var_67}
    var_69 = module_0.from_json_schema(var_68)
    var_70 = {}
    var_71 = module_0.from_json_schema(var_70)
    var_72 = '$ref'
    var_73 = 'components'
    var_74 = '#/components/schemas/User'
    var_75 = 'schemas'
    var_76 = 'User'
    var_77 = {var_4: var_5}
    var_78 = {var_76: var_77}
    var_79 = {var_75: var_78}
    var_80 = {var_72: var_74, var_73: var_79}
    var_81 = module_0.from_json_schema(var_80)
    var_82 = 'items'
    var_83 = {var_4: var_5}
    var_84 = {var_4: var_17, var_82: var_83}
    var_85 = module_0.from_json_schema(var_84)
    var_86 = var_85.items
    var_87 = 'properties'
    var_88 = 'name'
    var_89 = 'age'
    var_90 = {var_4: var_5}
    var_91 = {var_4: var_8}
    var_92 = {var_88: var_90, var_89: var_91}
    var_93 = {var_4: var_20, var_87: var_92}
    var_94 = module_0.from_json_schema(var_93)
    var_95 = 'pattern'
    var_96 = '^[a-z]+$'
    var_97 = {var_4: var_5, var_95: var_96}
    var_98 = module_0.from_json_schema(var_97)
    var_99 = var_98.fields
    var_100 = 'minimum'
    var_101 = 'maximum'
    var_102 = 'multipleOf'
    var_103 = 100
    var_104 = {var_4: var_11, var_100: var_2, var_101: var_103, var_102: var_37}
    var_105 = module_0.from_json_schema(var_104)
    var_106 = 'minItems'
    var_107 = 'maxItems'
    var_108 = 'uniqueItems'
    var_109 = {var_4: var_17, var_106: var_0, var_107: var_67, var_108: var_0}
    var_110 = module_0.from_json_schema(var_109)
    var_111 = 'minProperties'
    var_112 = 'maxProperties'
    var_113 = 'required'
    var_114 = 'id'
    var_115 = [var_114]
    var_116 = {var_4: var_20, var_111: var_0, var_112: var_37, var_113: var_115}
    var_117 = module_0.from_json_schema(var_116)



# Parsed testcases at query #4
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
    var_17 = 'array'
    var_18 = {var_4: var_17}
    var_19 = module_0.from_json_schema(var_18)
    var_20 = 'object'
    var_21 = {var_4: var_20}
    var_22 = module_0.from_json_schema(var_21)
    var_23 = 'enum'
    var_24 = 'a'
    var_25 = 'b'
    var_26 = 'c'
    var_27 = [var_24, var_25, var_26]
    var_28 = {var_23: var_27}
    var_29 = module_0.from_json_schema(var_28)
    var_30 = 'const'
    var_31 = 'fixed_value'
    var_32 = {var_30: var_31}
    var_33 = module_0.from_json_schema(var_32)
    var_34 = 'allOf'
    var_35 = {var_4: var_5}
    var_36 = 'minLength'
    var_37 = 3
    var_38 = {var_36: var_37}
    var_39 = [var_35, var_38]
    var_40 = {var_34: var_39}
    var_41 = module_0.from_json_schema(var_40)
    var_42 = 'anyOf'
    var_43 = {var_4: var_5}
    var_44 = {var_4: var_8}
    var_45 = [var_43, var_44]
    var_46 = {var_42: var_45}
    var_47 = module_0.from_json_schema(var_46)
    var_48 = 'oneOf'
    var_49 = {var_4: var_5}
    var_50 = {var_4: var_8}
    var_51 = [var_49, var_50]
    var_52 = {var_48: var_51}
    var_53 = module_0.from_json_schema(var_52)
    var_54 = 'not'
    var_55 = {var_4: var_5}
    var_56 = {var_54: var_55}
    var_57 = module_0.from_json_schema(var_56)
    var_58 = 'if'
    var_59 = 'then'
    var_60 = 'else'
    var_61 = {var_4: var_5}
    var_62 = 5
    var_63 = {var_36: var_62}
    var_64 = {var_4: var_8}
    var_65 = {var_58: var_61, var_59: var_63, var_60: var_64}
    var_66 = module_0.from_json_schema(var_65)
    var_67 = '$ref'
    var_68 = '#/components/schemas/User'
    var_69 = {var_67: var_68}
    var_70 = module_0.from_json_schema(var_69)
    var_71 = 'maxLength'
    var_72 = 'pattern'
    var_73 = 10
    var_74 = '^[a-z]+$'
    var_75 = {var_4: var_5, var_36: var_37, var_71: var_73, var_72: var_74}
    var_76 = module_0.from_json_schema(var_75)
    var_77 = {}
    var_78 = module_0.from_json_schema(var_77)
    var_79 = module_1.Definitions()
    var_80 = 'properties'
    var_81 = 'name'
    var_82 = {var_4: var_5}
    var_83 = {var_81: var_82}
    var_84 = {var_4: var_20, var_80: var_83}
    var_85 = module_0.from_json_schema(var_84, var_79)
    var_86 = 'items'
    var_87 = {var_4: var_5}
    var_88 = {var_4: var_17, var_86: var_87}
    var_89 = module_0.from_json_schema(var_88)
    var_90 = var_89.items
    var_91 = 'required'
    var_92 = 'age'
    var_93 = {var_4: var_5}
    var_94 = {var_4: var_8}
    var_95 = {var_81: var_93, var_92: var_94}
    var_96 = [var_81]
    var_97 = {var_4: var_20, var_80: var_95, var_91: var_96}
    var_98 = module_0.from_json_schema(var_97)
    var_99 = 'minimum'
    var_100 = 'maximum'
    var_101 = 'exclusiveMinimum'
    var_102 = 'exclusiveMaximum'
    var_103 = 'multipleOf'
    var_104 = 100
    var_105 = 2
    var_106 = {var_4: var_11, var_99: var_2, var_100: var_104, var_101: var_2, var_102: var_104, var_103: var_105}
    var_107 = module_0.from_json_schema(var_106)
    var_108 = 'format'
    var_109 = 'email'
    var_110 = {var_4: var_5, var_108: var_109}
    var_111 = module_0.from_json_schema(var_110)
    var_112 = {var_4: var_5}
    var_113 = {var_4: var_8}
    var_114 = [var_112, var_113]
    var_115 = {var_4: var_17, var_86: var_114}
    var_116 = module_0.from_json_schema(var_115)
    var_117 = var_116.items
    var_118 = var_116.items
    var_119 = len(var_118)
    assert var_119 == 2
    var_120 = var_116.items[var_2]
    var_121 = var_116.items[var_0]



# Parsed testcases at query #5
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'enum'
    var_2 = 'red'
    var_3 = 'green'
    var_4 = 'blue'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = module_1.enum_from_json_schema(var_6, var_0)
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = [var_8, var_9, var_10]
    var_12 = {var_1: var_11}
    var_13 = module_1.enum_from_json_schema(var_12, var_0)
    var_14 = 'text'
    var_15 = 42
    var_16 = True
    var_17 = None
    var_18 = [var_14, var_15, var_16, var_17]
    var_19 = {var_1: var_18}
    var_20 = module_1.enum_from_json_schema(var_19, var_0)
    var_21 = 'default'
    var_22 = 'a'
    var_23 = 'b'
    var_24 = 'c'
    var_25 = [var_22, var_23, var_24]
    var_26 = {var_1: var_25, var_21: var_23}
    var_27 = module_1.enum_from_json_schema(var_26, var_0)
    var_28 = []
    var_29 = {var_1: var_28}
    var_30 = module_1.enum_from_json_schema(var_29, var_0)
    var_31 = 'only'
    var_32 = [var_31]
    var_33 = {var_1: var_32}
    var_34 = module_1.enum_from_json_schema(var_33, var_0)



# Parsed testcases at query #6
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'minimum'
    var_2 = 'maximum'
    var_3 = 'number'
    var_4 = 0
    var_5 = 10
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = False
    var_8 = module_0.Definitions()
    var_9 = module_1.from_json_schema_type(var_6, var_3, var_7, var_8)
    var_10 = 'exclusiveMinimum'
    var_11 = 'multipleOf'
    var_12 = 'integer'
    var_13 = 5
    var_14 = 2
    var_15 = {var_0: var_12, var_10: var_13, var_11: var_14}
    var_16 = True
    var_17 = module_0.Definitions()
    var_18 = module_1.from_json_schema_type(var_15, var_12, var_16, var_17)
    var_19 = 'minLength'
    var_20 = 'pattern'
    var_21 = 'string'
    var_22 = 3
    var_23 = '^[A-Z]+$'
    var_24 = {var_0: var_21, var_19: var_22, var_20: var_23}
    var_25 = False
    var_26 = module_0.Definitions()
    var_27 = module_1.from_json_schema_type(var_24, var_21, var_25, var_26)
    var_28 = 'default'
    var_29 = 'boolean'
    var_30 = {var_0: var_29, var_28: var_16}
    var_31 = module_0.Definitions()
    var_32 = module_1.from_json_schema_type(var_30, var_29, var_16, var_31)
    var_33 = module_0.Definitions()
    var_34 = 'items'
    var_35 = 'array'
    var_36 = {var_0: var_21}
    var_37 = {var_0: var_3}
    var_38 = [var_36, var_37]
    var_39 = {var_0: var_35, var_34: var_38}
    var_40 = False
    var_41 = module_1.from_json_schema_type(var_39, var_35, var_40, var_33)
    var_42 = var_41.items
    var_43 = var_41.items
    var_44 = len(var_43)
    assert var_44 == 2
    var_45 = var_41.items[var_40]
    var_46 = var_41.items[var_16]
    var_47 = {var_0: var_12}
    var_48 = {var_0: var_35, var_34: var_47}
    var_49 = False
    var_50 = module_1.from_json_schema_type(var_48, var_35, var_49, var_33)
    var_51 = var_50.items
    var_52 = 'properties'
    var_53 = 'required'
    var_54 = 'object'
    var_55 = 'name'
    var_56 = 'age'
    var_57 = {var_0: var_21}
    var_58 = {var_0: var_12}
    var_59 = {var_55: var_57, var_56: var_58}
    var_60 = [var_55]
    var_61 = {var_0: var_54, var_52: var_59, var_53: var_60}
    var_62 = False
    var_63 = module_1.from_json_schema_type(var_61, var_54, var_62, var_33)
    var_64 = var_63.properties[var_55]
    var_65 = var_63.properties[var_56]
    var_66 = 'patternProperties'
    var_67 = '^S_'
    var_68 = {var_0: var_21}
    var_69 = {var_67: var_68}
    var_70 = {var_0: var_54, var_66: var_69}
    var_71 = False
    var_72 = module_1.from_json_schema_type(var_70, var_54, var_71, var_33)
    var_73 = var_72.pattern_properties[var_67]
    var_74 = 'additionalProperties'
    var_75 = False
    var_76 = {var_0: var_54, var_74: var_75}
    var_77 = False
    var_78 = module_1.from_json_schema_type(var_76, var_54, var_77, var_33)
    var_79 = {var_0: var_3}
    var_80 = {var_0: var_54, var_74: var_79}
    var_81 = False
    var_82 = module_1.from_json_schema_type(var_80, var_54, var_81, var_33)
    var_83 = var_82.additional_properties
    var_84 = {var_0: var_21}
    var_85 = module_1.from_json_schema_type(var_84, var_21, var_16, var_33)
    var_86 = 42
    var_87 = {var_0: var_12, var_28: var_86}
    var_88 = False
    var_89 = module_1.from_json_schema_type(var_87, var_12, var_88, var_33)
    var_90 = 'additionalItems'
    var_91 = False
    var_92 = {var_0: var_35, var_90: var_91}
    var_93 = False
    var_94 = module_1.from_json_schema_type(var_92, var_35, var_93, var_33)
    var_95 = {var_0: var_29}
    var_96 = {var_0: var_35, var_90: var_95}
    var_97 = False
    var_98 = module_1.from_json_schema_type(var_96, var_35, var_97, var_33)
    var_99 = var_98.additional_items
    var_100 = 'propertyNames'
    var_101 = '^[a-z]+$'
    var_102 = {var_20: var_101}
    var_103 = {var_0: var_54, var_100: var_102}
    var_104 = False
    var_105 = module_1.from_json_schema_type(var_103, var_54, var_104, var_33)
    var_106 = var_105.property_names



# Parsed testcases at query #7
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
    var_4 = False
    var_5 = 1
    var_6 = 10
    var_7 = '^[a-z]+$'
    var_8 = 'email'
    var_9 = module_0.String(max_length=var_6, min_length=var_5, pattern=var_7, format=var_8)
    var_10 = module_1.to_json_schema(var_9)
    var_11 = True
    var_12 = module_0.String()
    var_13 = module_1.to_json_schema(var_12)
    var_14 = 100
    var_15 = 2
    var_16 = module_0.Integer(minimum=var_4, maximum=var_14, exclusive_minimum=var_4, exclusive_maximum=var_14, multiple_of=var_15)
    var_17 = module_1.to_json_schema(var_16)
    var_18 = True
    var_19 = module_0.Float(minimum=var_4, maximum=var_18)
    var_20 = module_1.to_json_schema(var_19)
    var_21 = module_0.Boolean()
    var_22 = module_1.to_json_schema(var_21)
    var_23 = module_0.String()
    var_24 = True
    var_25 = module_0.Array(var_23, min_items=var_18, max_items=var_6, unique_items=var_24)
    var_26 = module_1.to_json_schema(var_25)
    var_27 = module_0.String()
    var_28 = module_0.Integer()
    var_29 = [var_27, var_28]
    var_30 = module_0.Array(var_29, var_4)
    var_31 = module_1.to_json_schema(var_30)
    var_32 = 'items'
    var_33 = var_31[var_32]
    var_34 = var_31[var_32]
    var_35 = len(var_34)
    assert var_35 == 2
    var_36 = 'name'
    var_37 = 'age'
    var_38 = module_0.String()
    var_39 = module_0.Integer()
    var_40 = {var_36: var_38, var_37: var_39}
    var_41 = [var_36]
    var_42 = module_0.Object(properties=var_40, additional_properties=var_4, required=var_41)
    var_43 = module_1.to_json_schema(var_42)
    var_44 = module_0.Integer()
    var_45 = {var_7: var_44}
    var_46 = module_0.Object(pattern_properties=var_45)
    var_47 = module_1.to_json_schema(var_46)
    var_48 = 'A'
    var_49 = (var_48, var_48)
    var_50 = 'B'
    var_51 = (var_50, var_50)
    var_52 = [var_49, var_51]
    var_53 = module_0.Choice(choices=var_52)
    var_54 = module_1.to_json_schema(var_53)
    var_55 = 'fixed_value'
    var_56 = module_0.Const(var_55)
    var_57 = module_1.to_json_schema(var_56)
    var_58 = module_0.String()
    var_59 = module_0.Integer()
    var_60 = [var_58, var_59]
    var_61 = module_0.Union(var_60)
    var_62 = module_1.to_json_schema(var_61)
    var_63 = 'anyOf'
    var_64 = var_62[var_63]
    var_65 = len(var_64)
    assert var_65 == 2
    var_66 = module_0.String()
    var_67 = module_0.Integer()
    var_68 = [var_66, var_67]
    var_69 = module_2.OneOf(var_68)
    var_70 = module_1.to_json_schema(var_69)
    var_71 = 'oneOf'
    var_72 = var_70[var_71]
    var_73 = len(var_72)
    assert var_73 == 2
    var_74 = module_0.String(min_length=var_24)
    var_75 = module_0.String(max_length=var_6)
    var_76 = [var_74, var_75]
    var_77 = module_2.AllOf(var_76)
    var_78 = module_1.to_json_schema(var_77)
    var_79 = 'allOf'
    var_80 = var_78[var_79]
    var_81 = len(var_80)
    assert var_81 == 2
    var_82 = module_0.String(pattern=var_7)
    var_83 = module_0.Integer(minimum=var_4)
    var_84 = module_0.Boolean()
    var_85 = module_2.IfThenElse(var_82, var_83, var_84)
    var_86 = module_1.to_json_schema(var_85)
    var_87 = '^[0-9]+$'
    var_88 = module_0.String(pattern=var_87)
    var_89 = module_2.Not(var_88)
    var_90 = module_1.to_json_schema(var_89)
    var_91 = module_0.String()
    var_92 = 'User'
    var_93 = {var_92: var_91}
    var_94 = {}
    var_95 = module_1.to_json_schema(var_89, var_94)
    var_96 = 'Group'
    var_97 = module_0.String()
    var_98 = {var_36: var_97}
    var_99 = module_0.Object(properties=var_98)
    var_100 = 'users'
    var_101 = {}
    var_102 = module_3.Reference(var_92, var_101)
    var_103 = module_0.Array(var_102)
    var_104 = {var_100: var_103}
    var_105 = module_0.Object(properties=var_104)
    var_106 = {var_92: var_99, var_96: var_105}
    var_107 = 'default_value'
    var_108 = module_0.String()
    var_109 = module_1.to_json_schema(var_108)
    var_110 = module_0.String()
    var_111 = module_0.Integer()
    var_112 = module_1.to_json_schema(var_108)
    var_113 = module_1.to_json_schema(var_108)



# Parsed testcases at query #8
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
    var_15 = 'null'
    var_16 = [var_1, var_15]
    var_17 = {var_0: var_16}
    var_18 = module_1.type_from_json_schema(var_17, var_3)
    var_19 = {var_0: var_15}
    var_20 = module_1.type_from_json_schema(var_19, var_3)
    var_21 = []
    var_22 = {var_0: var_21}
    var_23 = module_1.type_from_json_schema(var_22, var_3)
    var_24 = [var_15]
    var_25 = {var_0: var_24}
    var_26 = module_1.type_from_json_schema(var_25, var_3)
    var_27 = 'integer'
    var_28 = {var_0: var_27}
    var_29 = module_1.type_from_json_schema(var_28, var_3)
    var_30 = {var_0: var_5}
    var_31 = module_1.type_from_json_schema(var_30, var_3)
    var_32 = 'boolean'
    var_33 = {var_0: var_32}
    var_34 = module_1.type_from_json_schema(var_33, var_3)
    var_35 = 'array'
    var_36 = {var_0: var_35}
    var_37 = module_1.type_from_json_schema(var_36, var_3)
    var_38 = 'object'
    var_39 = {var_0: var_38}
    var_40 = module_1.type_from_json_schema(var_39, var_3)
    var_41 = [var_1, var_27, var_15]
    var_42 = {var_0: var_41}
    var_43 = module_1.type_from_json_schema(var_42, var_3)
    var_44 = var_43.any_of
    var_45 = len(var_44)
    assert var_45 == 2
    var_46 = var_43.any_of[var_11]
    var_47 = var_43.any_of[var_13]



# Parsed testcases at query #9
#--------------------------


import typesystem.fields as module_0
import typesystem.json_schema as module_1
import typesystem.composites as module_2
import typesystem.schemas as module_3
import re as module_4

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True
    var_2 = module_2.NeverMatch()
    var_3 = module_1.to_json_schema(var_2)
    assert var_3 is False
    var_4 = False
    var_5 = 1
    var_6 = 10
    var_7 = module_0.String(max_length=var_6, min_length=var_5)
    var_8 = module_1.to_json_schema(var_7)
    var_9 = True
    var_10 = module_0.String()
    var_11 = module_1.to_json_schema(var_10)
    var_12 = 100
    var_13 = module_0.Integer(minimum=var_4, maximum=var_12)
    var_14 = module_1.to_json_schema(var_13)
    var_15 = module_0.Float(minimum=var_4, maximum=var_9)
    var_16 = module_1.to_json_schema(var_15)
    var_17 = True
    var_18 = module_0.Boolean()
    var_19 = module_1.to_json_schema(var_18)
    var_20 = module_0.String()
    var_21 = module_0.Array(var_20, min_items=var_17, max_items=var_6)
    var_22 = module_1.to_json_schema(var_21)
    var_23 = 'name'
    var_24 = module_0.String()
    var_25 = {var_23: var_24}
    var_26 = [var_23]
    var_27 = module_0.Object(properties=var_25, required=var_26)
    var_28 = module_1.to_json_schema(var_27)
    var_29 = 'a'
    var_30 = 'A'
    var_31 = (var_29, var_30)
    var_32 = 'b'
    var_33 = 'B'
    var_34 = (var_32, var_33)
    var_35 = [var_31, var_34]
    var_36 = module_0.Choice(choices=var_35)
    var_37 = module_1.to_json_schema(var_36)
    var_38 = 'fixed_value'
    var_39 = module_0.Const(var_38)
    var_40 = module_1.to_json_schema(var_39)
    var_41 = module_0.String()
    var_42 = module_0.Integer()
    var_43 = [var_41, var_42]
    var_44 = module_0.Union(var_43)
    var_45 = module_1.to_json_schema(var_44)
    var_46 = 'anyOf'
    var_47 = var_45[var_46]
    var_48 = len(var_47)
    assert var_48 == 2
    var_49 = module_0.String(min_length=var_17)
    var_50 = module_0.String(max_length=var_6)
    var_51 = [var_49, var_50]
    var_52 = module_2.AllOf(var_51)
    var_53 = module_1.to_json_schema(var_52)
    var_54 = 'allOf'
    var_55 = var_53[var_54]
    var_56 = len(var_55)
    assert var_56 == 2
    var_57 = module_0.String()
    var_58 = module_0.Integer()
    var_59 = [var_57, var_58]
    var_60 = module_2.OneOf(var_59)
    var_61 = module_1.to_json_schema(var_60)
    var_62 = 'oneOf'
    var_63 = var_61[var_62]
    var_64 = len(var_63)
    assert var_64 == 2
    var_65 = module_0.String()
    var_66 = module_0.Integer()
    var_67 = module_2.IfThenElse(var_65, var_66)
    var_68 = module_1.to_json_schema(var_67)
    var_69 = module_0.String()
    var_70 = module_2.Not(var_69)
    var_71 = module_1.to_json_schema(var_70)
    var_72 = 'MySchema'
    var_73 = module_0.String()
    var_74 = {var_72: var_73}
    var_75 = module_3.Reference(var_72, var_74)
    var_76 = module_1.to_json_schema(var_75)
    var_77 = module_0.String()
    var_78 = {var_72: var_77}
    var_79 = module_1.to_json_schema(var_78)
    var_80 = 'default_value'
    var_81 = module_0.String()
    var_82 = module_1.to_json_schema(var_81)
    var_83 = '^\\d+$'
    var_84 = module_4.compile(var_83)
    var_85 = module_0.String()
    var_86 = module_1.to_json_schema(var_85)
    var_87 = module_0.Integer(exclusive_minimum=var_4, exclusive_maximum=var_12)
    var_88 = module_1.to_json_schema(var_87)
    var_89 = 2
    var_90 = module_0.Integer(multiple_of=var_89)
    var_91 = module_1.to_json_schema(var_90)
    var_92 = True
    var_93 = module_0.Array(unique_items=var_92)
    var_94 = module_1.to_json_schema(var_93)
    var_95 = module_0.Object(additional_properties=var_4)
    var_96 = module_1.to_json_schema(var_95)
    var_97 = module_0.String()
    var_98 = module_0.Object(additional_properties=var_97)
    var_99 = module_1.to_json_schema(var_98)
    var_100 = 'additionalProperties'
    var_101 = var_99[var_100]
    var_102 = module_0.String(min_length=var_92)
    var_103 = module_0.Object(property_names=var_102)
    var_104 = module_1.to_json_schema(var_103)
    var_105 = module_0.Integer()
    var_106 = {var_83: var_105}
    var_107 = module_0.Object(pattern_properties=var_106)
    var_108 = module_1.to_json_schema(var_107)
    var_109 = module_0.Object(min_properties=var_92, max_properties=var_6)
    var_110 = module_1.to_json_schema(var_109)
    var_111 = module_0.String()
    var_112 = {var_23: var_111}
    var_113 = [var_23]
    var_114 = module_3.Schema(var_112)
    var_115 = module_1.to_json_schema(var_114)
    var_116 = module_0.Decimal(minimum=var_4, maximum=var_92)
    var_117 = module_1.to_json_schema(var_116)



# Parsed testcases at query #10
#--------------------------


import typesystem.fields as module_0
import typesystem.json_schema as module_1
import typesystem.composites as module_2
import typesystem.schemas as module_3
import re as module_4

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True
    var_2 = module_2.NeverMatch()
    var_3 = module_1.to_json_schema(var_2)
    assert var_3 is False
    var_4 = False
    var_5 = 1
    var_6 = 10
    var_7 = module_0.String(max_length=var_6, min_length=var_5)
    var_8 = module_1.to_json_schema(var_7)
    var_9 = True
    var_10 = module_0.String()
    var_11 = module_1.to_json_schema(var_10)
    var_12 = 100
    var_13 = module_0.Integer(minimum=var_4, maximum=var_12)
    var_14 = module_1.to_json_schema(var_13)
    var_15 = module_0.Float(minimum=var_4, maximum=var_9)
    var_16 = module_1.to_json_schema(var_15)
    var_17 = True
    var_18 = module_0.Boolean()
    var_19 = module_1.to_json_schema(var_18)
    var_20 = module_0.String()
    var_21 = 5
    var_22 = module_0.Array(var_20, min_items=var_17, max_items=var_21)
    var_23 = module_1.to_json_schema(var_22)
    var_24 = module_0.String()
    var_25 = module_0.Integer()
    var_26 = [var_24, var_25]
    var_27 = module_0.Array(var_26)
    var_28 = module_1.to_json_schema(var_27)
    var_29 = 'items'
    var_30 = var_28[var_29]
    var_31 = var_28[var_29]
    var_32 = len(var_31)
    assert var_32 == 2
    var_33 = 'name'
    var_34 = 'age'
    var_35 = module_0.String()
    var_36 = module_0.Integer()
    var_37 = {var_33: var_35, var_34: var_36}
    var_38 = [var_33]
    var_39 = module_0.Object(properties=var_37, required=var_38)
    var_40 = module_1.to_json_schema(var_39)
    var_41 = '\\d+'
    var_42 = module_0.Integer()
    var_43 = {var_41: var_42}
    var_44 = module_0.Object(pattern_properties=var_43)
    var_45 = module_1.to_json_schema(var_44)
    var_46 = 'A'
    var_47 = (var_46, var_46)
    var_48 = 'B'
    var_49 = (var_48, var_48)
    var_50 = [var_47, var_49]
    var_51 = module_0.Choice(choices=var_50)
    var_52 = module_1.to_json_schema(var_51)
    var_53 = 'fixed_value'
    var_54 = module_0.Const(var_53)
    var_55 = module_1.to_json_schema(var_54)
    var_56 = module_0.String()
    var_57 = module_0.Integer()
    var_58 = [var_56, var_57]
    var_59 = module_0.Union(var_58)
    var_60 = module_1.to_json_schema(var_59)
    var_61 = 'anyOf'
    var_62 = var_60[var_61]
    var_63 = len(var_62)
    assert var_63 == 2
    var_64 = module_0.String()
    var_65 = module_0.Integer()
    var_66 = [var_64, var_65]
    var_67 = module_2.OneOf(var_66)
    var_68 = module_1.to_json_schema(var_67)
    var_69 = 'oneOf'
    var_70 = var_68[var_69]
    var_71 = len(var_70)
    assert var_71 == 2
    var_72 = module_0.String(min_length=var_17)
    var_73 = module_0.String(max_length=var_6)
    var_74 = [var_72, var_73]
    var_75 = module_2.AllOf(var_74)
    var_76 = module_1.to_json_schema(var_75)
    var_77 = 'allOf'
    var_78 = var_76[var_77]
    var_79 = len(var_78)
    assert var_79 == 2
    var_80 = module_0.String()
    var_81 = module_0.Integer()
    var_82 = module_0.Boolean()
    var_83 = module_2.IfThenElse(var_80, var_81, var_82)
    var_84 = module_1.to_json_schema(var_83)
    var_85 = module_0.String()
    var_86 = module_2.Not(var_85)
    var_87 = module_1.to_json_schema(var_86)
    var_88 = 'MySchema'
    var_89 = module_0.String()
    var_90 = {var_88: var_89}
    var_91 = 'Person'
    var_92 = 'Address'
    var_93 = module_0.String()
    var_94 = {var_33: var_93}
    var_95 = module_0.Object(properties=var_94)
    var_96 = 'street'
    var_97 = module_0.String()
    var_98 = {var_96: var_97}
    var_99 = module_0.Object(properties=var_98)
    var_100 = {var_91: var_95, var_92: var_99}
    var_101 = 'default_value'
    var_102 = module_0.String()
    var_103 = module_1.to_json_schema(var_102)
    var_104 = module_0.String()
    var_105 = {var_33: var_104}
    var_106 = [var_33]
    var_107 = module_3.Schema(var_105)
    var_108 = module_1.to_json_schema(var_107)
    var_109 = '^\\d+$'
    var_110 = module_4.compile(var_109)
    var_111 = module_0.String()
    var_112 = module_1.to_json_schema(var_111)
    var_113 = True
    var_114 = module_0.Array(unique_items=var_113)
    var_115 = module_1.to_json_schema(var_114)
    var_116 = module_0.Object(additional_properties=var_4)
    var_117 = module_1.to_json_schema(var_116)
    var_118 = module_0.String()
    var_119 = module_0.Object(additional_properties=var_118)
    var_120 = module_1.to_json_schema(var_119)
    var_121 = 'additionalProperties'
    var_122 = var_120[var_121]
    var_123 = module_0.String(min_length=var_113)
    var_124 = module_0.Object(property_names=var_123)
    var_125 = module_1.to_json_schema(var_124)
    var_126 = module_0.Decimal()
    var_127 = module_1.to_json_schema(var_126)



# Parsed testcases at query #11
#--------------------------


import typesystem.fields as module_0
import typesystem.json_schema as module_1
import typesystem.composites as module_2
import typesystem.schemas as module_3
import re as module_4

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True
    var_2 = module_2.NeverMatch()
    var_3 = module_1.to_json_schema(var_2)
    assert var_3 is False
    var_4 = False
    var_5 = 1
    var_6 = 10
    var_7 = '^[a-z]+$'
    var_8 = module_0.String(max_length=var_6, min_length=var_5, pattern=var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = True
    var_11 = module_0.String()
    var_12 = module_1.to_json_schema(var_11)
    var_13 = 100
    var_14 = True
    var_15 = True
    var_16 = module_0.Integer(minimum=var_4, maximum=var_13, exclusive_minimum=var_14, exclusive_maximum=var_15)
    var_17 = module_1.to_json_schema(var_16)
    var_18 = 0.1
    var_19 = module_0.Float(minimum=var_4, maximum=var_15, multiple_of=var_18)
    var_20 = module_1.to_json_schema(var_19)
    var_21 = True
    var_22 = module_0.Boolean()
    var_23 = module_1.to_json_schema(var_22)
    var_24 = module_0.String()
    var_25 = True
    var_26 = module_0.Array(var_24, var_4, var_21, var_6, unique_items=var_25)
    var_27 = module_1.to_json_schema(var_26)
    var_28 = 'name'
    var_29 = 'age'
    var_30 = module_0.String()
    var_31 = module_0.Integer()
    var_32 = {var_28: var_30, var_29: var_31}
    var_33 = [var_28]
    var_34 = 5
    var_35 = module_0.Object(properties=var_32, additional_properties=var_4, min_properties=var_25, max_properties=var_34, required=var_33)
    var_36 = module_1.to_json_schema(var_35)
    var_37 = 'a'
    var_38 = 'A'
    var_39 = (var_37, var_38)
    var_40 = 'b'
    var_41 = 'B'
    var_42 = (var_40, var_41)
    var_43 = [var_39, var_42]
    var_44 = module_0.Choice(choices=var_43)
    var_45 = module_1.to_json_schema(var_44)
    var_46 = 'fixed_value'
    var_47 = module_0.Const(var_46)
    var_48 = module_1.to_json_schema(var_47)
    var_49 = module_0.String()
    var_50 = module_0.Integer()
    var_51 = [var_49, var_50]
    var_52 = module_0.Union(var_51)
    var_53 = module_1.to_json_schema(var_52)
    var_54 = 'anyOf'
    var_55 = var_53[var_54]
    var_56 = len(var_55)
    assert var_56 == 2
    var_57 = module_0.String()
    var_58 = module_0.Integer()
    var_59 = [var_57, var_58]
    var_60 = module_2.OneOf(var_59)
    var_61 = module_1.to_json_schema(var_60)
    var_62 = 'oneOf'
    var_63 = var_61[var_62]
    var_64 = len(var_63)
    assert var_64 == 2
    var_65 = module_0.String(min_length=var_25)
    var_66 = module_0.String(max_length=var_6)
    var_67 = [var_65, var_66]
    var_68 = module_2.AllOf(var_67)
    var_69 = module_1.to_json_schema(var_68)
    var_70 = 'allOf'
    var_71 = var_69[var_70]
    var_72 = len(var_71)
    assert var_72 == 2
    var_73 = module_0.String(min_length=var_25)
    var_74 = module_0.Integer(minimum=var_4)
    var_75 = module_0.String(max_length=var_6)
    var_76 = module_2.IfThenElse(var_73, var_74, var_75)
    var_77 = module_1.to_json_schema(var_76)
    var_78 = module_0.String()
    var_79 = module_2.Not(var_78)
    var_80 = module_1.to_json_schema(var_79)
    var_81 = module_3.Definitions()
    var_82 = module_0.String()
    var_83 = {var_28: var_82}
    var_84 = 'Person'
    var_85 = module_3.Reference(var_84, var_81)
    var_86 = module_1.to_json_schema(var_85)
    var_87 = module_0.String()
    var_88 = {var_28: var_87}
    var_89 = [var_28]
    var_90 = module_3.Schema(var_88)
    var_91 = module_1.to_json_schema(var_90)
    var_92 = 'default_value'
    var_93 = module_0.String()
    var_94 = module_1.to_json_schema(var_93)
    var_95 = module_0.String()
    var_96 = {var_7: var_95}
    var_97 = module_0.Object(pattern_properties=var_96)
    var_98 = module_1.to_json_schema(var_97)
    var_99 = module_0.String(pattern=var_7)
    var_100 = module_0.Object(property_names=var_99)
    var_101 = module_1.to_json_schema(var_100)
    var_102 = module_0.String()
    var_103 = module_0.Integer()
    var_104 = [var_102, var_103]
    var_105 = module_0.String()
    var_106 = module_0.Array(var_104, var_105)
    var_107 = module_1.to_json_schema(var_106)
    var_108 = 'items'
    var_109 = var_107[var_108]
    var_110 = var_107[var_108]
    var_111 = len(var_110)
    assert var_111 == 2
    var_112 = '^[a-z]+$'
    var_113 = module_4.compile(var_112, var_1)
    var_114 = module_0.String(pattern=var_113)
    var_115 = module_1.to_json_schema(var_114)



# Parsed testcases at query #12
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
    var_4 = False
    var_5 = 1
    var_6 = 10
    var_7 = '^[a-z]+$'
    var_8 = 'email'
    var_9 = module_0.String(max_length=var_6, min_length=var_5, pattern=var_7, format=var_8)
    var_10 = module_1.to_json_schema(var_9)
    var_11 = True
    var_12 = True
    var_13 = module_0.String(allow_blank=var_12, min_length=var_4)
    var_14 = module_1.to_json_schema(var_13)
    var_15 = 100
    var_16 = 2
    var_17 = module_0.Integer(minimum=var_4, maximum=var_15, exclusive_minimum=var_4, exclusive_maximum=var_15, multiple_of=var_16)
    var_18 = module_1.to_json_schema(var_17)
    var_19 = True
    var_20 = module_0.Float(minimum=var_4, maximum=var_19)
    var_21 = module_1.to_json_schema(var_20)
    var_22 = True
    var_23 = module_0.Boolean()
    var_24 = module_1.to_json_schema(var_23)
    var_25 = module_0.String()
    var_26 = True
    var_27 = module_0.Array(var_25, min_items=var_22, max_items=var_6, unique_items=var_26)
    var_28 = module_1.to_json_schema(var_27)
    var_29 = module_0.String()
    var_30 = module_0.Integer()
    var_31 = [var_29, var_30]
    var_32 = module_0.Array(var_31, var_4)
    var_33 = module_1.to_json_schema(var_32)
    var_34 = 'items'
    var_35 = var_33[var_34]
    var_36 = var_33[var_34]
    var_37 = len(var_36)
    assert var_37 == 2
    var_38 = 'name'
    var_39 = 'age'
    var_40 = module_0.String()
    var_41 = module_0.Integer()
    var_42 = {var_38: var_40, var_39: var_41}
    var_43 = True
    var_44 = [var_38]
    var_45 = module_0.Object(properties=var_42, additional_properties=var_4, required=var_44)
    var_46 = module_1.to_json_schema(var_45)
    var_47 = '^test_'
    var_48 = module_0.String()
    var_49 = {var_47: var_48}
    var_50 = module_0.Object(pattern_properties=var_49)
    var_51 = module_1.to_json_schema(var_50)
    var_52 = 'A'
    var_53 = 'Option A'
    var_54 = (var_52, var_53)
    var_55 = 'B'
    var_56 = 'Option B'
    var_57 = (var_55, var_56)
    var_58 = [var_54, var_57]
    var_59 = module_0.Choice(choices=var_58)
    var_60 = module_1.to_json_schema(var_59)
    var_61 = 'fixed_value'
    var_62 = module_0.Const(var_61)
    var_63 = module_1.to_json_schema(var_62)
    var_64 = module_0.String()
    var_65 = module_0.Integer()
    var_66 = [var_64, var_65]
    var_67 = module_0.Union(var_66)
    var_68 = module_1.to_json_schema(var_67)
    var_69 = 'anyOf'
    var_70 = var_68[var_69]
    var_71 = len(var_70)
    assert var_71 == 2
    var_72 = module_0.String()
    var_73 = module_0.Integer()
    var_74 = [var_72, var_73]
    var_75 = module_2.OneOf(var_74)
    var_76 = module_1.to_json_schema(var_75)
    var_77 = 'oneOf'
    var_78 = var_76[var_77]
    var_79 = len(var_78)
    assert var_79 == 2
    var_80 = module_0.String(min_length=var_43)
    var_81 = module_0.String(max_length=var_6)
    var_82 = [var_80, var_81]
    var_83 = module_2.AllOf(var_82)
    var_84 = module_1.to_json_schema(var_83)
    var_85 = 'allOf'
    var_86 = var_84[var_85]
    var_87 = len(var_86)
    assert var_87 == 2
    var_88 = module_0.String()
    var_89 = module_0.Integer()
    var_90 = module_0.Boolean()
    var_91 = module_2.IfThenElse(var_88, var_89, var_90)
    var_92 = module_1.to_json_schema(var_91)
    var_93 = module_0.String()
    var_94 = module_2.Not(var_93)
    var_95 = module_1.to_json_schema(var_94)
    var_96 = 'User'
    var_97 = module_0.String()
    var_98 = {var_38: var_97}
    var_99 = module_0.Object(properties=var_98)
    var_100 = {var_96: var_99}
    var_101 = module_1.to_json_schema(var_94)
    var_102 = module_0.String()
    var_103 = {var_38: var_102}
    var_104 = module_0.Object(properties=var_103)
    var_105 = module_1.to_json_schema(var_104)
    var_106 = module_0.String()
    var_107 = {var_38: var_106}
    var_108 = [var_38]
    var_109 = module_3.Schema(var_107)
    var_110 = module_1.to_json_schema(var_109)
    var_111 = 'default_value'
    var_112 = module_0.String()
    var_113 = module_1.to_json_schema(var_112)
    var_114 = True
    var_115 = module_0.Decimal(minimum=var_4, maximum=var_15)
    var_116 = module_1.to_json_schema(var_115)
    var_117 = module_1.to_json_schema(var_115)



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
    var_4 = False
    var_5 = 1
    var_6 = 10
    var_7 = '^[a-z]+$'
    var_8 = module_0.String(max_length=var_6, min_length=var_5, pattern=var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = True
    var_11 = module_0.String()
    var_12 = module_1.to_json_schema(var_11)
    var_13 = 100
    var_14 = module_0.Integer(minimum=var_4, maximum=var_13, exclusive_minimum=var_4, exclusive_maximum=var_4)
    var_15 = module_1.to_json_schema(var_14)
    var_16 = 0.1
    var_17 = module_0.Float(minimum=var_4, maximum=var_10, multiple_of=var_16)
    var_18 = module_1.to_json_schema(var_17)
    var_19 = True
    var_20 = module_0.Boolean()
    var_21 = module_1.to_json_schema(var_20)
    var_22 = module_0.String()
    var_23 = True
    var_24 = module_0.Array(var_22, min_items=var_19, max_items=var_6, unique_items=var_23)
    var_25 = module_1.to_json_schema(var_24)
    var_26 = module_0.String()
    var_27 = module_0.Integer()
    var_28 = [var_26, var_27]
    var_29 = module_0.Array(var_28)
    var_30 = module_1.to_json_schema(var_29)
    var_31 = 'items'
    var_32 = var_30[var_31]
    var_33 = var_30[var_31]
    var_34 = len(var_33)
    assert var_34 == 2
    var_35 = 'name'
    var_36 = 'age'
    var_37 = module_0.String()
    var_38 = module_0.Integer()
    var_39 = {var_35: var_37, var_36: var_38}
    var_40 = [var_35]
    var_41 = module_0.Object(properties=var_39, additional_properties=var_4, required=var_40)
    var_42 = module_1.to_json_schema(var_41)
    var_43 = '^test_'
    var_44 = module_0.String()
    var_45 = {var_43: var_44}
    var_46 = module_0.Object(pattern_properties=var_45)
    var_47 = module_1.to_json_schema(var_46)
    var_48 = 'A'
    var_49 = 'Option A'
    var_50 = (var_48, var_49)
    var_51 = 'B'
    var_52 = 'Option B'
    var_53 = (var_51, var_52)
    var_54 = [var_50, var_53]
    var_55 = module_0.Choice(choices=var_54)
    var_56 = module_1.to_json_schema(var_55)
    var_57 = 'fixed_value'
    var_58 = module_0.Const(var_57)
    var_59 = module_1.to_json_schema(var_58)
    var_60 = module_0.String()
    var_61 = module_0.Integer()
    var_62 = [var_60, var_61]
    var_63 = module_0.Union(var_62)
    var_64 = module_1.to_json_schema(var_63)
    var_65 = 'anyOf'
    var_66 = var_64[var_65]
    var_67 = len(var_66)
    assert var_67 == 2
    var_68 = module_0.String()
    var_69 = module_0.Integer()
    var_70 = [var_68, var_69]
    var_71 = module_2.OneOf(var_70)
    var_72 = module_1.to_json_schema(var_71)
    var_73 = 'oneOf'
    var_74 = var_72[var_73]
    var_75 = len(var_74)
    assert var_75 == 2
    var_76 = module_0.String(min_length=var_23)
    var_77 = module_0.String(max_length=var_6)
    var_78 = [var_76, var_77]
    var_79 = module_2.AllOf(var_78)
    var_80 = module_1.to_json_schema(var_79)
    var_81 = 'allOf'
    var_82 = var_80[var_81]
    var_83 = len(var_82)
    assert var_83 == 2
    var_84 = module_0.String()
    var_85 = module_0.Integer()
    var_86 = module_0.Boolean()
    var_87 = module_2.IfThenElse(var_84, var_85, var_86)
    var_88 = module_1.to_json_schema(var_87)
    var_89 = module_0.String()
    var_90 = module_2.Not(var_89)
    var_91 = module_1.to_json_schema(var_90)
    var_92 = module_3.Definitions()
    var_93 = module_0.String()
    var_94 = {var_35: var_93}
    var_95 = 'Person'
    var_96 = module_3.Reference(var_95, var_92)
    var_97 = module_1.to_json_schema(var_96)
    var_98 = 'person'
    var_99 = module_3.Reference(var_95, var_92)
    var_100 = {var_98: var_99}
    var_101 = module_0.Object(properties=var_100)
    var_102 = module_1.to_json_schema(var_101)
    var_103 = 'default_value'
    var_104 = module_0.String()
    var_105 = module_1.to_json_schema(var_104)
    var_106 = module_0.String()
    var_107 = {var_35: var_106}
    var_108 = [var_35]
    var_109 = module_3.Schema(var_107)
    var_110 = module_1.to_json_schema(var_109)
    var_111 = module_0.Decimal(minimum=var_4, maximum=var_13)
    var_112 = module_1.to_json_schema(var_111)



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
    var_4 = False
    var_5 = 1
    var_6 = 10
    var_7 = '^[a-z]+$'
    var_8 = module_0.String(max_length=var_6, min_length=var_5, pattern=var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = True
    var_11 = module_0.String()
    var_12 = module_1.to_json_schema(var_11)
    var_13 = 100
    var_14 = 2
    var_15 = module_0.Integer(minimum=var_4, maximum=var_13, multiple_of=var_14)
    var_16 = module_1.to_json_schema(var_15)
    var_17 = True
    var_18 = module_0.Float(exclusive_minimum=var_4, exclusive_maximum=var_17)
    var_19 = module_1.to_json_schema(var_18)
    var_20 = module_0.Boolean()
    var_21 = module_1.to_json_schema(var_20)
    var_22 = module_0.String()
    var_23 = True
    var_24 = module_0.Array(var_22, var_4, var_17, var_6, unique_items=var_23)
    var_25 = module_1.to_json_schema(var_24)
    var_26 = True
    var_27 = 'name'
    var_28 = 'age'
    var_29 = module_0.String()
    var_30 = module_0.Integer()
    var_31 = {var_27: var_29, var_28: var_30}
    var_32 = [var_27]
    var_33 = module_0.Object(properties=var_31, min_properties=var_26, max_properties=var_14, required=var_32)
    var_34 = module_1.to_json_schema(var_33)
    var_35 = 'A'
    var_36 = 'Option A'
    var_37 = (var_35, var_36)
    var_38 = 'B'
    var_39 = 'Option B'
    var_40 = (var_38, var_39)
    var_41 = [var_37, var_40]
    var_42 = module_0.Choice(choices=var_41)
    var_43 = module_1.to_json_schema(var_42)
    var_44 = 'fixed_value'
    var_45 = module_0.Const(var_44)
    var_46 = module_1.to_json_schema(var_45)
    var_47 = module_0.String()
    var_48 = module_0.Integer()
    var_49 = [var_47, var_48]
    var_50 = module_0.Union(var_49)
    var_51 = module_1.to_json_schema(var_50)
    var_52 = 'anyOf'
    var_53 = var_51[var_52]
    var_54 = len(var_53)
    assert var_54 == 2
    var_55 = module_0.String()
    var_56 = module_0.Integer()
    var_57 = [var_55, var_56]
    var_58 = module_2.OneOf(var_57)
    var_59 = module_1.to_json_schema(var_58)
    var_60 = 'oneOf'
    var_61 = var_59[var_60]
    var_62 = len(var_61)
    assert var_62 == 2
    var_63 = module_0.String(min_length=var_26)
    var_64 = module_0.String(max_length=var_6)
    var_65 = [var_63, var_64]
    var_66 = module_2.AllOf(var_65)
    var_67 = module_1.to_json_schema(var_66)
    var_68 = 'allOf'
    var_69 = var_67[var_68]
    var_70 = len(var_69)
    assert var_70 == 2
    var_71 = module_3.Definitions()
    var_72 = module_0.String()
    var_73 = {var_27: var_72}
    var_74 = 'Person'
    var_75 = module_3.Reference(var_74, var_71)
    var_76 = module_1.to_json_schema(var_75)
    var_77 = module_0.String()
    var_78 = module_0.Integer()
    var_79 = module_0.Boolean()
    var_80 = module_2.IfThenElse(var_77, var_78, var_79)
    var_81 = module_1.to_json_schema(var_80)
    var_82 = module_0.String()
    var_83 = module_2.Not(var_82)
    var_84 = module_1.to_json_schema(var_83)
    var_85 = module_0.String()
    var_86 = {var_27: var_85}
    var_87 = module_0.Object(properties=var_86)
    var_88 = {var_74: var_87}
    var_89 = module_1.to_json_schema(var_87, var_88)
    var_90 = module_1.to_json_schema(var_87)
    var_91 = 'default_value'
    var_92 = module_0.String()
    var_93 = module_1.to_json_schema(var_92)



# Parsed testcases at query #15
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
    var_4 = False
    var_5 = 1
    var_6 = 10
    var_7 = '^[a-z]+$'
    var_8 = 'email'
    var_9 = module_0.String(max_length=var_6, min_length=var_5, pattern=var_7, format=var_8)
    var_10 = module_1.to_json_schema(var_9)
    var_11 = True
    var_12 = module_0.String(allow_blank=var_4)
    var_13 = module_1.to_json_schema(var_12)
    var_14 = 100
    var_15 = 5
    var_16 = module_0.Integer(minimum=var_4, maximum=var_14, multiple_of=var_15)
    var_17 = module_1.to_json_schema(var_16)
    var_18 = True
    var_19 = module_0.Float(exclusive_minimum=var_4, exclusive_maximum=var_18)
    var_20 = module_1.to_json_schema(var_19)
    var_21 = module_0.Boolean()
    var_22 = module_1.to_json_schema(var_21)
    var_23 = module_0.String()
    var_24 = True
    var_25 = module_0.Array(var_23, var_4, var_18, var_6, unique_items=var_24)
    var_26 = module_1.to_json_schema(var_25)
    var_27 = 'name'
    var_28 = 'age'
    var_29 = module_0.String()
    var_30 = module_0.Integer()
    var_31 = {var_27: var_29, var_28: var_30}
    var_32 = [var_27]
    var_33 = 2
    var_34 = module_0.Object(properties=var_31, additional_properties=var_4, min_properties=var_24, max_properties=var_33, required=var_32)
    var_35 = module_1.to_json_schema(var_34)
    var_36 = 'a'
    var_37 = 'A'
    var_38 = (var_36, var_37)
    var_39 = 'b'
    var_40 = 'B'
    var_41 = (var_39, var_40)
    var_42 = [var_38, var_41]
    var_43 = module_0.Choice(choices=var_42)
    var_44 = module_1.to_json_schema(var_43)
    var_45 = 'fixed_value'
    var_46 = module_0.Const(var_45)
    var_47 = module_1.to_json_schema(var_46)
    var_48 = module_0.String()
    var_49 = module_0.Integer()
    var_50 = [var_48, var_49]
    var_51 = module_0.Union(var_50)
    var_52 = module_1.to_json_schema(var_51)
    var_53 = 'anyOf'
    var_54 = var_52[var_53]
    var_55 = len(var_54)
    assert var_55 == 2
    var_56 = module_0.String()
    var_57 = module_0.Integer()
    var_58 = [var_56, var_57]
    var_59 = module_2.OneOf(var_58)
    var_60 = module_1.to_json_schema(var_59)
    var_61 = 'oneOf'
    var_62 = var_60[var_61]
    var_63 = len(var_62)
    assert var_63 == 2
    var_64 = module_0.String(min_length=var_24)
    var_65 = module_0.String(max_length=var_6)
    var_66 = [var_64, var_65]
    var_67 = module_2.AllOf(var_66)
    var_68 = module_1.to_json_schema(var_67)
    var_69 = 'allOf'
    var_70 = var_68[var_69]
    var_71 = len(var_70)
    assert var_71 == 2
    var_72 = module_0.String(min_length=var_15)
    var_73 = module_0.String(max_length=var_6)
    var_74 = module_0.Integer()
    var_75 = module_2.IfThenElse(var_72, var_73, var_74)
    var_76 = module_1.to_json_schema(var_75)
    var_77 = module_0.String()
    var_78 = module_2.Not(var_77)
    var_79 = module_1.to_json_schema(var_78)
    var_80 = module_3.Definitions()
    var_81 = module_0.String()
    var_82 = {var_27: var_81}
    var_83 = 'Person'
    var_84 = module_3.Reference(var_83, var_80)
    var_85 = module_1.to_json_schema(var_84)
    var_86 = 'person'
    var_87 = module_3.Reference(var_83, var_80)
    var_88 = {var_86: var_87}
    var_89 = module_0.Object(properties=var_88)
    var_90 = {}
    var_91 = module_1.to_json_schema(var_89, var_90)
    var_92 = 'default_value'
    var_93 = module_0.String()
    var_94 = module_1.to_json_schema(var_93)
    var_95 = module_0.String()
    var_96 = module_0.Integer()
    var_97 = module_1.to_json_schema(var_93)



# Parsed testcases at query #16
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
    var_5 = 5
    var_6 = 10
    var_7 = '^test.*$'
    var_8 = 'email'
    var_9 = module_0.String(max_length=var_6, min_length=var_5, pattern=var_7, format=var_8)
    var_10 = module_1.to_json_schema(var_9)
    var_11 = False
    var_12 = 100
    var_13 = module_0.Integer(minimum=var_11, maximum=var_12, exclusive_minimum=var_11, exclusive_maximum=var_12, multiple_of=var_5)
    var_14 = module_1.to_json_schema(var_13)
    var_15 = module_0.Float(minimum=var_11, maximum=var_4)
    var_16 = module_1.to_json_schema(var_15)
    var_17 = module_0.Boolean()
    var_18 = module_1.to_json_schema(var_17)
    var_19 = module_0.String()
    var_20 = module_0.Array(var_19, var_11, var_4, var_6, unique_items=var_4)
    var_21 = module_1.to_json_schema(var_20)
    var_22 = 'name'
    var_23 = 'age'
    var_24 = module_0.String()
    var_25 = module_0.Integer()
    var_26 = {var_22: var_24, var_23: var_25}
    var_27 = module_0.String()
    var_28 = {var_7: var_27}
    var_29 = '^[a-z]+$'
    var_30 = module_0.String(pattern=var_29)
    var_31 = [var_22]
    var_32 = module_0.Object(properties=var_26, pattern_properties=var_28, additional_properties=var_11, property_names=var_30, min_properties=var_4, max_properties=var_5, required=var_31)
    var_33 = module_1.to_json_schema(var_32)
    var_34 = module_0.String()
    var_35 = module_0.Integer()
    var_36 = {var_22: var_34, var_23: var_35}
    var_37 = [var_22]
    var_38 = module_3.Schema(var_36)
    var_39 = module_1.to_json_schema(var_38)
    var_40 = 'A'
    var_41 = (var_40, var_40)
    var_42 = 'B'
    var_43 = (var_42, var_42)
    var_44 = [var_41, var_43]
    var_45 = module_0.Choice(choices=var_44)
    var_46 = module_1.to_json_schema(var_45)
    var_47 = 'fixed_value'
    var_48 = module_0.Const(var_47)
    var_49 = module_1.to_json_schema(var_48)
    var_50 = module_0.String()
    var_51 = module_0.Integer()
    var_52 = [var_50, var_51]
    var_53 = module_0.Union(var_52)
    var_54 = module_1.to_json_schema(var_53)
    var_55 = 'anyOf'
    var_56 = var_54[var_55]
    var_57 = len(var_56)
    assert var_57 == 2
    var_58 = module_0.String()
    var_59 = module_0.Integer()
    var_60 = [var_58, var_59]
    var_61 = module_2.OneOf(var_60)
    var_62 = module_1.to_json_schema(var_61)
    var_63 = 'oneOf'
    var_64 = var_62[var_63]
    var_65 = len(var_64)
    assert var_65 == 2
    var_66 = module_0.String(min_length=var_5)
    var_67 = module_0.String(max_length=var_6)
    var_68 = [var_66, var_67]
    var_69 = module_2.AllOf(var_68)
    var_70 = module_1.to_json_schema(var_69)
    var_71 = 'allOf'
    var_72 = var_70[var_71]
    var_73 = len(var_72)
    assert var_73 == 2
    var_74 = module_0.String(pattern=var_7)
    var_75 = module_0.String(min_length=var_6)
    var_76 = module_0.String(max_length=var_5)
    var_77 = module_2.IfThenElse(var_74, var_75, var_76)
    var_78 = module_1.to_json_schema(var_77)
    var_79 = module_0.String(pattern=var_7)
    var_80 = module_2.Not(var_79)
    var_81 = module_1.to_json_schema(var_80)
    var_82 = module_3.Definitions()
    var_83 = module_0.String()
    var_84 = {var_22: var_83}
    var_85 = 'Person'
    var_86 = module_3.Reference(var_85, var_82)
    var_87 = module_1.to_json_schema(var_86)
    var_88 = 'person'
    var_89 = module_3.Reference(var_85, var_82)
    var_90 = {var_88: var_89}
    var_91 = module_0.Object(properties=var_90)
    var_92 = module_1.to_json_schema(var_91)
    var_93 = 'default_value'
    var_94 = module_0.String()
    var_95 = module_1.to_json_schema(var_94)
    var_96 = module_1.to_json_schema(var_94)



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
    var_4 = False
    var_5 = 1
    var_6 = 10
    var_7 = '^[a-z]+$'
    var_8 = module_0.String(max_length=var_6, min_length=var_5, pattern=var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = True
    var_11 = module_0.String()
    var_12 = module_1.to_json_schema(var_11)
    var_13 = 100
    var_14 = True
    var_15 = module_0.Integer(minimum=var_4, maximum=var_13, exclusive_minimum=var_14, exclusive_maximum=var_4)
    var_16 = module_1.to_json_schema(var_15)
    var_17 = 0.5
    var_18 = module_0.Float(multiple_of=var_17)
    var_19 = module_1.to_json_schema(var_18)
    var_20 = True
    var_21 = module_0.Boolean()
    var_22 = module_1.to_json_schema(var_21)
    var_23 = module_0.String()
    var_24 = True
    var_25 = module_0.Array(var_23, min_items=var_20, max_items=var_6, unique_items=var_24)
    var_26 = module_1.to_json_schema(var_25)
    var_27 = module_0.String()
    var_28 = module_0.Integer()
    var_29 = [var_27, var_28]
    var_30 = module_0.Array(var_29, var_4)
    var_31 = module_1.to_json_schema(var_30)
    var_32 = 'items'
    var_33 = var_31[var_32]
    var_34 = var_31[var_32]
    var_35 = len(var_34)
    assert var_35 == 2
    var_36 = 'name'
    var_37 = 'age'
    var_38 = module_0.String()
    var_39 = module_0.Integer()
    var_40 = {var_36: var_38, var_37: var_39}
    var_41 = [var_36]
    var_42 = module_0.Object(properties=var_40, additional_properties=var_4, required=var_41)
    var_43 = module_1.to_json_schema(var_42)
    var_44 = '^test_'
    var_45 = module_0.String()
    var_46 = {var_44: var_45}
    var_47 = module_0.Object(pattern_properties=var_46)
    var_48 = module_1.to_json_schema(var_47)
    var_49 = 'a'
    var_50 = 'A'
    var_51 = (var_49, var_50)
    var_52 = 'b'
    var_53 = 'B'
    var_54 = (var_52, var_53)
    var_55 = [var_51, var_54]
    var_56 = module_0.Choice(choices=var_55)
    var_57 = module_1.to_json_schema(var_56)
    var_58 = 42
    var_59 = module_0.Const(var_58)
    var_60 = module_1.to_json_schema(var_59)
    var_61 = module_0.String()
    var_62 = module_0.Integer()
    var_63 = [var_61, var_62]
    var_64 = module_0.Union(var_63)
    var_65 = module_1.to_json_schema(var_64)
    var_66 = 'anyOf'
    var_67 = var_65[var_66]
    var_68 = len(var_67)
    assert var_68 == 2
    var_69 = module_0.String()
    var_70 = module_0.Integer()
    var_71 = [var_69, var_70]
    var_72 = module_2.OneOf(var_71)
    var_73 = module_1.to_json_schema(var_72)
    var_74 = 'oneOf'
    var_75 = var_73[var_74]
    var_76 = len(var_75)
    assert var_76 == 2
    var_77 = module_0.String(min_length=var_24)
    var_78 = module_0.String(max_length=var_6)
    var_79 = [var_77, var_78]
    var_80 = module_2.AllOf(var_79)
    var_81 = module_1.to_json_schema(var_80)
    var_82 = 'allOf'
    var_83 = var_81[var_82]
    var_84 = len(var_83)
    assert var_84 == 2
    var_85 = module_0.String()
    var_86 = module_0.Integer()
    var_87 = module_0.Boolean()
    var_88 = module_2.IfThenElse(var_85, var_86, var_87)
    var_89 = module_1.to_json_schema(var_88)
    var_90 = module_0.String()
    var_91 = module_2.Not(var_90)
    var_92 = module_1.to_json_schema(var_91)
    var_93 = module_3.Definitions()
    var_94 = module_0.String()
    var_95 = {var_36: var_94}
    var_96 = 'Person'
    var_97 = module_3.Reference(var_96, var_93)
    var_98 = module_1.to_json_schema(var_97)
    var_99 = 'person'
    var_100 = {var_99: var_97}
    var_101 = module_0.Object(properties=var_100)
    var_102 = module_1.to_json_schema(var_101)
    var_103 = 'hello'
    var_104 = module_0.String()
    var_105 = module_1.to_json_schema(var_104)
    var_106 = module_0.String()
    var_107 = {var_36: var_106}
    var_108 = [var_36]
    var_109 = module_3.Schema(var_107)
    var_110 = module_1.to_json_schema(var_109)



# Parsed testcases at query #18
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
    var_5 = 5
    var_6 = 10
    var_7 = '^test.*$'
    var_8 = 'email'
    var_9 = module_0.String(max_length=var_6, min_length=var_5, pattern=var_7, format=var_8)
    var_10 = module_1.to_json_schema(var_9)
    var_11 = False
    var_12 = 100
    var_13 = module_0.Integer(minimum=var_11, maximum=var_12, exclusive_minimum=var_11, exclusive_maximum=var_12, multiple_of=var_5)
    var_14 = module_1.to_json_schema(var_13)
    var_15 = module_0.Float(minimum=var_11, maximum=var_4)
    var_16 = module_1.to_json_schema(var_15)
    var_17 = module_0.Boolean()
    var_18 = module_1.to_json_schema(var_17)
    var_19 = module_0.String()
    var_20 = module_0.Array(var_19, var_11, var_4, var_6, unique_items=var_4)
    var_21 = module_1.to_json_schema(var_20)
    var_22 = 'name'
    var_23 = 'age'
    var_24 = module_0.String()
    var_25 = module_0.Integer()
    var_26 = {var_22: var_24, var_23: var_25}
    var_27 = [var_22]
    var_28 = module_0.Object(properties=var_26, additional_properties=var_11, min_properties=var_4, max_properties=var_5, required=var_27)
    var_29 = module_1.to_json_schema(var_28)
    var_30 = 'A'
    var_31 = 'Option A'
    var_32 = (var_30, var_31)
    var_33 = 'B'
    var_34 = 'Option B'
    var_35 = (var_33, var_34)
    var_36 = [var_32, var_35]
    var_37 = module_0.Choice(choices=var_36)
    var_38 = module_1.to_json_schema(var_37)
    var_39 = 'fixed_value'
    var_40 = module_0.Const(var_39)
    var_41 = module_1.to_json_schema(var_40)
    var_42 = module_0.String()
    var_43 = module_0.Integer()
    var_44 = [var_42, var_43]
    var_45 = module_0.Union(var_44)
    var_46 = module_1.to_json_schema(var_45)
    var_47 = 'anyOf'
    var_48 = var_46[var_47]
    var_49 = len(var_48)
    assert var_49 == 2
    var_50 = module_0.String()
    var_51 = module_0.Integer()
    var_52 = [var_50, var_51]
    var_53 = module_2.OneOf(var_52)
    var_54 = module_1.to_json_schema(var_53)
    var_55 = 'oneOf'
    var_56 = var_54[var_55]
    var_57 = len(var_56)
    assert var_57 == 2
    var_58 = module_0.String(min_length=var_5)
    var_59 = module_0.String(max_length=var_6)
    var_60 = [var_58, var_59]
    var_61 = module_2.AllOf(var_60)
    var_62 = module_1.to_json_schema(var_61)
    var_63 = 'allOf'
    var_64 = var_62[var_63]
    var_65 = len(var_64)
    assert var_65 == 2
    var_66 = module_0.String(min_length=var_5)
    var_67 = module_0.Integer(minimum=var_6)
    var_68 = module_0.Boolean()
    var_69 = module_2.IfThenElse(var_66, var_67, var_68)
    var_70 = module_1.to_json_schema(var_69)
    var_71 = module_0.String()
    var_72 = module_2.Not(var_71)
    var_73 = module_1.to_json_schema(var_72)
    var_74 = module_3.Definitions()
    var_75 = module_0.String()
    var_76 = {var_22: var_75}
    var_77 = 'Person'
    var_78 = module_3.Reference(var_77, var_74)
    var_79 = module_1.to_json_schema(var_78)
    var_80 = module_3.Reference(var_77, var_74)
    var_81 = {}
    var_82 = module_1.to_json_schema(var_80, var_81)
    var_83 = module_0.String()
    var_84 = {var_22: var_83}
    var_85 = [var_22]
    var_86 = module_3.Schema(var_84)
    var_87 = module_1.to_json_schema(var_86)
    var_88 = 'default_value'
    var_89 = module_0.String()
    var_90 = module_1.to_json_schema(var_89)
    var_91 = module_1.to_json_schema(var_89)
    var_92 = module_1.to_json_schema(var_89)



# Parsed testcases at query #19
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
    var_4 = False
    var_5 = 1
    var_6 = 10
    var_7 = '^[a-z]+$'
    var_8 = 'email'
    var_9 = module_0.String(max_length=var_6, min_length=var_5, pattern=var_7, format=var_8)
    var_10 = module_1.to_json_schema(var_9)
    var_11 = True
    var_12 = module_0.String()
    var_13 = module_1.to_json_schema(var_12)
    var_14 = 100
    var_15 = 2
    var_16 = module_0.Integer(minimum=var_4, maximum=var_14, exclusive_minimum=var_4, exclusive_maximum=var_14, multiple_of=var_15)
    var_17 = module_1.to_json_schema(var_16)
    var_18 = True
    var_19 = module_0.Float(minimum=var_4, maximum=var_18)
    var_20 = module_1.to_json_schema(var_19)
    var_21 = module_0.Boolean()
    var_22 = module_1.to_json_schema(var_21)
    var_23 = module_0.String()
    var_24 = True
    var_25 = module_0.Array(var_23, var_4, var_18, var_6, unique_items=var_24)
    var_26 = module_1.to_json_schema(var_25)
    var_27 = module_0.String()
    var_28 = module_0.Integer()
    var_29 = [var_27, var_28]
    var_30 = module_0.Array(var_29)
    var_31 = module_1.to_json_schema(var_30)
    var_32 = 'items'
    var_33 = var_31[var_32]
    var_34 = len(var_33)
    assert var_34 == 2
    var_35 = 'name'
    var_36 = 'age'
    var_37 = module_0.String()
    var_38 = module_0.Integer()
    var_39 = {var_35: var_37, var_36: var_38}
    var_40 = [var_35]
    var_41 = module_0.Object(properties=var_39, additional_properties=var_4, min_properties=var_24, max_properties=var_15, required=var_40)
    var_42 = module_1.to_json_schema(var_41)
    var_43 = '^test_'
    var_44 = module_0.String()
    var_45 = {var_43: var_44}
    var_46 = module_0.Object(pattern_properties=var_45)
    var_47 = module_1.to_json_schema(var_46)
    var_48 = 'A'
    var_49 = (var_48, var_48)
    var_50 = 'B'
    var_51 = (var_50, var_50)
    var_52 = [var_49, var_51]
    var_53 = module_0.Choice(choices=var_52)
    var_54 = module_1.to_json_schema(var_53)
    var_55 = 'fixed_value'
    var_56 = module_0.Const(var_55)
    var_57 = module_1.to_json_schema(var_56)
    var_58 = module_0.String()
    var_59 = module_0.Integer()
    var_60 = [var_58, var_59]
    var_61 = module_0.Union(var_60)
    var_62 = module_1.to_json_schema(var_61)
    var_63 = 'anyOf'
    var_64 = var_62[var_63]
    var_65 = len(var_64)
    assert var_65 == 2
    var_66 = module_2.OneOf(var_60)
    var_67 = module_1.to_json_schema(var_66)
    var_68 = 'oneOf'
    var_69 = var_67[var_68]
    var_70 = len(var_69)
    assert var_70 == 2
    var_71 = module_2.AllOf(var_60)
    var_72 = module_1.to_json_schema(var_71)
    var_73 = 'allOf'
    var_74 = var_72[var_73]
    var_75 = len(var_74)
    assert var_75 == 2
    var_76 = module_0.Boolean()
    var_77 = module_0.String()
    var_78 = module_0.Integer()
    var_79 = module_2.IfThenElse(var_76, var_77, var_78)
    var_80 = module_1.to_json_schema(var_79)
    var_81 = module_0.String()
    var_82 = module_2.Not(var_81)
    var_83 = module_1.to_json_schema(var_82)
    var_84 = module_0.String()
    var_85 = 'TestSchema'
    var_86 = {var_85: var_84}
    var_87 = module_1.to_json_schema(var_82)
    var_88 = {}
    var_89 = module_1.to_json_schema(var_82, var_88)
    var_90 = module_0.String()
    var_91 = module_0.Integer()
    var_92 = {var_35: var_90, var_36: var_91}
    var_93 = [var_35]
    var_94 = module_3.Schema(var_92)
    var_95 = module_1.to_json_schema(var_94)
    var_96 = 'default_value'
    var_97 = module_0.String()
    var_98 = module_1.to_json_schema(var_97)
    var_99 = module_0.Decimal(minimum=var_4, maximum=var_6)
    var_100 = module_1.to_json_schema(var_99)
    var_101 = module_1.to_json_schema(var_99)



# Parsed testcases at query #20
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
    var_4 = False
    var_5 = 1
    var_6 = 10
    var_7 = '^[a-z]+$'
    var_8 = module_0.String(max_length=var_6, min_length=var_5, pattern=var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = True
    var_11 = module_0.String()
    var_12 = module_1.to_json_schema(var_11)
    var_13 = 100
    var_14 = True
    var_15 = module_0.Integer(minimum=var_4, maximum=var_13, exclusive_minimum=var_14, exclusive_maximum=var_4)
    var_16 = module_1.to_json_schema(var_15)
    var_17 = 0.5
    var_18 = module_0.Float(multiple_of=var_17)
    var_19 = module_1.to_json_schema(var_18)
    var_20 = True
    var_21 = module_0.Boolean()
    var_22 = module_1.to_json_schema(var_21)
    var_23 = module_0.String()
    var_24 = 5
    var_25 = True
    var_26 = module_0.Array(var_23, var_4, var_20, var_24, unique_items=var_25)
    var_27 = module_1.to_json_schema(var_26)
    var_28 = 'name'
    var_29 = 'age'
    var_30 = module_0.String()
    var_31 = module_0.Integer()
    var_32 = {var_28: var_30, var_29: var_31}
    var_33 = [var_28]
    var_34 = module_0.Object(properties=var_32, additional_properties=var_4, required=var_33)
    var_35 = module_1.to_json_schema(var_34)
    var_36 = 'id'
    var_37 = module_0.Integer()
    var_38 = {var_36: var_37}
    var_39 = [var_36]
    var_40 = module_3.Schema(var_38)
    var_41 = module_1.to_json_schema(var_40)
    var_42 = 'A'
    var_43 = 'Option A'
    var_44 = (var_42, var_43)
    var_45 = 'B'
    var_46 = 'Option B'
    var_47 = (var_45, var_46)
    var_48 = [var_44, var_47]
    var_49 = module_0.Choice(choices=var_48)
    var_50 = module_1.to_json_schema(var_49)
    var_51 = 'fixed_value'
    var_52 = module_0.Const(var_51)
    var_53 = module_1.to_json_schema(var_52)
    var_54 = module_0.String()
    var_55 = module_0.Integer()
    var_56 = [var_54, var_55]
    var_57 = module_0.Union(var_56)
    var_58 = module_1.to_json_schema(var_57)
    var_59 = 'anyOf'
    var_60 = var_58[var_59]
    var_61 = len(var_60)
    assert var_61 == 2
    var_62 = module_0.String()
    var_63 = module_0.Integer()
    var_64 = [var_62, var_63]
    var_65 = module_2.OneOf(var_64)
    var_66 = module_1.to_json_schema(var_65)
    var_67 = 'oneOf'
    var_68 = var_66[var_67]
    var_69 = len(var_68)
    assert var_69 == 2
    var_70 = module_0.String(min_length=var_25)
    var_71 = module_0.String(max_length=var_6)
    var_72 = [var_70, var_71]
    var_73 = module_2.AllOf(var_72)
    var_74 = module_1.to_json_schema(var_73)
    var_75 = 'allOf'
    var_76 = var_74[var_75]
    var_77 = len(var_76)
    assert var_77 == 2
    var_78 = module_0.String(min_length=var_24)
    var_79 = 20
    var_80 = module_0.String(max_length=var_79)
    var_81 = module_0.Integer()
    var_82 = module_2.IfThenElse(var_78, var_80, var_81)
    var_83 = module_1.to_json_schema(var_82)
    var_84 = module_0.String()
    var_85 = module_2.Not(var_84)
    var_86 = module_1.to_json_schema(var_85)
    var_87 = 'User'
    var_88 = module_0.String()
    var_89 = {var_28: var_88}
    var_90 = module_0.Object(properties=var_89)
    var_91 = {var_87: var_90}
    var_92 = module_0.String()
    var_93 = {var_28: var_92}
    var_94 = module_0.Object(properties=var_93)
    var_95 = {var_87: var_94}
    var_96 = module_1.to_json_schema(var_94, var_95)
    var_97 = 'default_value'
    var_98 = module_0.String()
    var_99 = module_1.to_json_schema(var_98)



