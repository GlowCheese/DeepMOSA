####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + devstral-2512 t=0.8)      #
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
    var_8 = 'maxLength'
    var_9 = 10
    var_10 = {var_3: var_5, var_8: var_9}
    var_11 = [var_7, var_10]
    var_12 = 'test'
    var_13 = {var_1: var_11, var_2: var_12}
    var_14 = module_1.all_of_from_json_schema(var_13, var_0)
    var_15 = var_14.all_of
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = 0
    var_18 = var_14.all_of[var_17]
    var_19 = 1
    var_20 = var_14.all_of[var_19]
    var_21 = 'properties'
    var_22 = 'object'
    var_23 = 'name'
    var_24 = {var_3: var_5}
    var_25 = {var_23: var_24}
    var_26 = {var_3: var_22, var_21: var_25}
    var_27 = 'age'
    var_28 = 'integer'
    var_29 = {var_3: var_28}
    var_30 = {var_27: var_29}
    var_31 = {var_3: var_22, var_21: var_30}
    var_32 = [var_31]
    var_33 = {var_1: var_32}
    var_34 = [var_26, var_33]
    var_35 = {var_1: var_34}
    var_36 = module_1.all_of_from_json_schema(var_35, var_0)
    var_37 = var_36.all_of
    var_38 = len(var_37)
    assert var_38 == 2
    var_39 = var_36.all_of[var_17]
    var_40 = var_36.all_of[var_19]



# Parsed testcases at query #2
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
    var_42 = var_41.schemas
    var_43 = len(var_42)
    assert var_43 == 2
    var_44 = 'anyOf'
    var_45 = {var_4: var_5}
    var_46 = {var_4: var_8}
    var_47 = [var_45, var_46]
    var_48 = {var_44: var_47}
    var_49 = module_0.from_json_schema(var_48)
    var_50 = var_49.schemas
    var_51 = len(var_50)
    assert var_51 == 2
    var_52 = 'oneOf'
    var_53 = {var_4: var_5}
    var_54 = {var_4: var_8}
    var_55 = [var_53, var_54]
    var_56 = {var_52: var_55}
    var_57 = module_0.from_json_schema(var_56)
    var_58 = var_57.schemas
    var_59 = len(var_58)
    assert var_59 == 2
    var_60 = 'not'
    var_61 = {var_4: var_5}
    var_62 = {var_60: var_61}
    var_63 = module_0.from_json_schema(var_62)
    var_64 = var_63.schema
    var_65 = 'if'
    var_66 = 'then'
    var_67 = 'else'
    var_68 = {var_4: var_5}
    var_69 = {var_36: var_37}
    var_70 = {var_4: var_8}
    var_71 = {var_65: var_68, var_66: var_69, var_67: var_70}
    var_72 = module_0.from_json_schema(var_71)
    var_73 = var_72.if_schema
    var_74 = var_72.then_schema
    var_75 = var_72.else_schema
    var_76 = module_1.Definitions()
    var_77 = '$ref'
    var_78 = '#/components/schemas/Test'
    var_79 = {var_77: var_78}
    var_80 = module_0.from_json_schema(var_79, var_76)
    var_81 = 'maxLength'
    var_82 = 'pattern'
    var_83 = 10
    var_84 = '^[a-zA-Z]+$'
    var_85 = {var_4: var_5, var_36: var_37, var_81: var_83, var_82: var_84}
    var_86 = module_0.from_json_schema(var_85)
    var_87 = var_86.schemas
    var_88 = len(var_87)
    assert var_88 == 4



# Parsed testcases at query #3
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = '$ref'
    var_1 = '#/components/schemas/User'
    var_2 = {var_0: var_1}
    var_3 = module_0.Definitions()
    var_4 = module_1.ref_from_json_schema(var_2, var_3)
    var_5 = 'components/schemas/User'
    var_6 = {var_0: var_5}
    var_7 = module_0.Definitions()
    var_8 = module_1.ref_from_json_schema(var_6, var_7)



# Parsed testcases at query #4
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
    var_23 = {var_1: var_22}
    var_24 = module_1.if_then_else_from_json_schema(var_23, var_0)
    var_25 = var_24.if_clause
    var_26 = 'default'
    var_27 = {var_4: var_5}
    var_28 = {var_4: var_7}
    var_29 = {var_4: var_9}
    var_30 = 42
    var_31 = {var_1: var_27, var_2: var_28, var_3: var_29, var_26: var_30}
    var_32 = module_1.if_then_else_from_json_schema(var_31, var_0)



# Parsed testcases at query #5
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
    var_6 = 'number'
    var_7 = {var_3: var_6}
    var_8 = [var_5, var_7]
    var_9 = 'default_value'
    var_10 = {var_1: var_8, var_2: var_9}
    var_11 = module_1.one_of_from_json_schema(var_10, var_0)
    var_12 = var_11.one_of
    var_13 = len(var_12)
    assert var_13 == 2
    var_14 = 0
    var_15 = var_11.one_of[var_14]
    var_16 = 1
    var_17 = var_11.one_of[var_16]



# Parsed testcases at query #6
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
    var_8 = 0
    var_9 = 100
    var_10 = 2
    var_11 = 50
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
    var_26 = 5
    var_27 = 'email'
    var_28 = '^[a-zA-Z0-9]+$'
    var_29 = 'test'
    var_30 = {var_0: var_25, var_21: var_26, var_22: var_9, var_23: var_27, var_24: var_28, var_6: var_29}
    var_31 = False
    var_32 = module_0.Definitions()
    var_33 = module_1.from_json_schema_type(var_30, var_25, var_31, var_32)
    var_34 = 'boolean'
    var_35 = True
    var_36 = {var_0: var_34, var_6: var_35}
    var_37 = False
    var_38 = module_0.Definitions()
    var_39 = module_1.from_json_schema_type(var_36, var_34, var_37, var_38)
    var_40 = 'items'
    var_41 = 'minItems'
    var_42 = 'maxItems'
    var_43 = 'uniqueItems'
    var_44 = 'array'
    var_45 = {var_0: var_25}
    var_46 = 10
    var_47 = [var_29]
    var_48 = {var_0: var_44, var_40: var_45, var_41: var_35, var_42: var_46, var_43: var_35, var_6: var_47}
    var_49 = False
    var_50 = module_0.Definitions()
    var_51 = module_1.from_json_schema_type(var_48, var_44, var_49, var_50)
    var_52 = var_51.items
    var_53 = 'properties'
    var_54 = 'required'
    var_55 = 'minProperties'
    var_56 = 'maxProperties'
    var_57 = 'object'
    var_58 = 'name'
    var_59 = 'age'
    var_60 = {var_0: var_25}
    var_61 = {var_0: var_16}
    var_62 = {var_58: var_60, var_59: var_61}
    var_63 = [var_58]
    var_64 = 25
    var_65 = {var_58: var_29, var_59: var_64}
    var_66 = {var_0: var_57, var_53: var_62, var_54: var_63, var_55: var_35, var_56: var_46, var_6: var_65}
    var_67 = False
    var_68 = module_0.Definitions()
    var_69 = module_1.from_json_schema_type(var_66, var_57, var_67, var_68)
    var_70 = var_69.properties[var_58]
    var_71 = var_69.properties[var_59]



# Parsed testcases at query #7
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
    var_12 = var_9.any_of
    var_13 = var_9.any_of
    var_14 = 'nullable'
    var_15 = True
    var_16 = {var_0: var_1, var_14: var_15}
    var_17 = module_0.Definitions()
    var_18 = module_1.type_from_json_schema(var_16, var_17)
    var_19 = {var_14: var_15}
    var_20 = module_0.Definitions()
    var_21 = module_1.type_from_json_schema(var_19, var_20)
    var_22 = {}
    var_23 = module_0.Definitions()
    var_24 = module_1.type_from_json_schema(var_22, var_23)
    var_25 = 'minLength'
    var_26 = 'maxLength'
    var_27 = 5
    var_28 = 10
    var_29 = {var_0: var_1, var_25: var_27, var_26: var_28}
    var_30 = module_0.Definitions()
    var_31 = module_1.type_from_json_schema(var_29, var_30)
    var_32 = 'items'
    var_33 = 'array'
    var_34 = {var_0: var_1}
    var_35 = {var_0: var_33, var_32: var_34}
    var_36 = module_0.Definitions()
    var_37 = module_1.type_from_json_schema(var_35, var_36)
    var_38 = var_37.items
    var_39 = 'properties'
    var_40 = 'object'
    var_41 = 'name'
    var_42 = {var_0: var_1}
    var_43 = {var_41: var_42}
    var_44 = {var_0: var_40, var_39: var_43}
    var_45 = module_0.Definitions()
    var_46 = module_1.type_from_json_schema(var_44, var_45)
    var_47 = var_46.properties[var_41]
    var_48 = 'minimum'
    var_49 = 'maximum'
    var_50 = 0
    var_51 = 100
    var_52 = {var_0: var_5, var_48: var_50, var_49: var_51}
    var_53 = module_0.Definitions()
    var_54 = module_1.type_from_json_schema(var_52, var_53)
    var_55 = 'boolean'
    var_56 = {var_0: var_55}
    var_57 = module_0.Definitions()
    var_58 = module_1.type_from_json_schema(var_56, var_57)
    var_59 = 'integer'
    var_60 = {var_0: var_59}
    var_61 = module_0.Definitions()
    var_62 = module_1.type_from_json_schema(var_60, var_61)



# Parsed testcases at query #8
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
    var_4 = 1
    var_5 = 10
    var_6 = '[a-z]+'
    var_7 = 'email'
    var_8 = module_0.String(max_length=var_5, min_length=var_4, pattern=var_6, format=var_7)
    var_9 = 'type'
    var_10 = 'minLength'
    var_11 = 'maxLength'
    var_12 = 'pattern'
    var_13 = 'format'
    var_14 = 'string'
    var_15 = {var_9: var_14, var_10: var_4, var_11: var_5, var_12: var_6, var_13: var_7}
    var_16 = module_1.to_json_schema(var_8)
    var_17 = 0
    var_18 = 100
    var_19 = True
    var_20 = module_0.Integer(minimum=var_17, maximum=var_18, exclusive_maximum=var_19)
    var_21 = 'minimum'
    var_22 = 'maximum'
    var_23 = 'exclusiveMaximum'
    var_24 = 'integer'
    var_25 = True
    var_26 = {var_9: var_24, var_21: var_17, var_22: var_18, var_23: var_25}
    var_27 = module_1.to_json_schema(var_20)
    var_28 = 0.5
    var_29 = module_0.Float(multiple_of=var_28)
    var_30 = 'multipleOf'
    var_31 = 'number'
    var_32 = {var_9: var_31, var_30: var_28}
    var_33 = module_1.to_json_schema(var_29)
    var_34 = module_0.Boolean()
    var_35 = 'boolean'
    var_36 = {var_9: var_35}
    var_37 = module_1.to_json_schema(var_34)
    var_38 = module_0.String()
    var_39 = 5
    var_40 = True
    var_41 = module_0.Array(var_38, min_items=var_25, max_items=var_39, unique_items=var_40)
    var_42 = 'items'
    var_43 = 'minItems'
    var_44 = 'maxItems'
    var_45 = 'uniqueItems'
    var_46 = 'array'
    var_47 = {var_9: var_14}
    var_48 = True
    var_49 = {var_9: var_46, var_42: var_47, var_43: var_40, var_44: var_39, var_45: var_48}
    var_50 = module_1.to_json_schema(var_41)
    var_51 = 'name'
    var_52 = module_0.String()
    var_53 = {var_51: var_52}
    var_54 = [var_51]
    var_55 = module_0.Object(properties=var_53, min_properties=var_48, max_properties=var_39, required=var_54)
    var_56 = 'properties'
    var_57 = 'required'
    var_58 = 'minProperties'
    var_59 = 'maxProperties'
    var_60 = 'object'
    var_61 = {var_9: var_14}
    var_62 = {var_51: var_61}
    var_63 = [var_51]
    var_64 = {var_9: var_60, var_56: var_62, var_57: var_63, var_58: var_48, var_59: var_39}
    var_65 = module_1.to_json_schema(var_55)
    var_66 = 'a'
    var_67 = (var_66, var_66)
    var_68 = 'b'
    var_69 = (var_68, var_68)
    var_70 = [var_67, var_69]
    var_71 = module_0.Choice(choices=var_70)
    var_72 = 'enum'
    var_73 = [var_66, var_68]
    var_74 = {var_72: var_73}
    var_75 = module_1.to_json_schema(var_71)
    var_76 = 'fixed_value'
    var_77 = module_0.Const(var_76)
    var_78 = 'const'
    var_79 = {var_78: var_76}
    var_80 = module_1.to_json_schema(var_77)
    var_81 = module_0.String()
    var_82 = module_0.Integer()
    var_83 = [var_81, var_82]
    var_84 = module_0.Union(var_83)
    var_85 = 'anyOf'
    var_86 = {var_9: var_14}
    var_87 = {var_9: var_24}
    var_88 = [var_86, var_87]
    var_89 = {var_85: var_88}
    var_90 = module_1.to_json_schema(var_84)
    var_91 = module_0.String()
    var_92 = module_0.Integer()
    var_93 = [var_91, var_92]
    var_94 = module_2.OneOf(var_93)
    var_95 = 'oneOf'
    var_96 = {var_9: var_14}
    var_97 = {var_9: var_24}
    var_98 = [var_96, var_97]
    var_99 = {var_95: var_98}
    var_100 = module_1.to_json_schema(var_94)
    var_101 = module_0.String()
    var_102 = module_0.Integer()
    var_103 = [var_101, var_102]
    var_104 = module_2.AllOf(var_103)
    var_105 = 'allOf'
    var_106 = {var_9: var_14}
    var_107 = {var_9: var_24}
    var_108 = [var_106, var_107]
    var_109 = {var_105: var_108}
    var_110 = module_1.to_json_schema(var_104)
    var_111 = module_0.String()
    var_112 = module_0.Integer()
    var_113 = module_0.Boolean()
    var_114 = module_2.IfThenElse(var_111, var_112, var_113)
    var_115 = 'if'
    var_116 = 'then'
    var_117 = 'else'
    var_118 = {var_9: var_14}
    var_119 = {var_9: var_24}
    var_120 = {var_9: var_35}
    var_121 = {var_115: var_118, var_116: var_119, var_117: var_120}
    var_122 = module_1.to_json_schema(var_114)
    var_123 = module_0.String()
    var_124 = module_2.Not(var_123)
    var_125 = 'not'
    var_126 = {var_9: var_14}
    var_127 = {var_125: var_126}
    var_128 = module_1.to_json_schema(var_124)
    var_129 = 'test'
    var_130 = module_0.String()
    var_131 = module_0.String()
    var_132 = {var_129: var_131}
    var_133 = module_3.Reference(var_129, var_132)
    var_134 = '$ref'
    var_135 = 'components'
    var_136 = '#/components/schemas/test'
    var_137 = 'schemas'
    var_138 = {var_9: var_14}
    var_139 = {var_129: var_138}
    var_140 = {var_137: var_139}
    var_141 = {var_134: var_136, var_135: var_140}
    var_142 = module_1.to_json_schema(var_133)
    var_143 = module_0.String()
    var_144 = {var_51: var_143}
    var_145 = [var_51]
    var_146 = module_3.Schema(var_144)
    var_147 = {var_9: var_14}
    var_148 = {var_51: var_147}
    var_149 = [var_51]
    var_150 = {var_9: var_60, var_56: var_148, var_57: var_149}
    var_151 = module_1.to_json_schema(var_146)
    var_152 = 'field1'
    var_153 = 'field2'
    var_154 = module_0.String()
    var_155 = module_0.Integer()
    var_156 = {var_152: var_154, var_153: var_155}
    var_157 = {var_9: var_14}
    var_158 = {var_9: var_24}
    var_159 = {var_152: var_157, var_153: var_158}
    var_160 = {var_137: var_159}
    var_161 = {var_135: var_160}
    var_162 = True
    var_163 = module_0.String()
    var_164 = 'null'
    var_165 = [var_14, var_164]
    var_166 = {var_9: var_165}
    var_167 = module_1.to_json_schema(var_163)
    var_168 = 'default_value'
    var_169 = module_0.String()
    var_170 = 'default'
    var_171 = {var_9: var_14, var_170: var_168}
    var_172 = module_1.to_json_schema(var_169)



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
    var_4 = 1
    var_5 = 10
    var_6 = '[a-z]+'
    var_7 = 'email'
    var_8 = module_0.String(max_length=var_5, min_length=var_4, pattern=var_6, format=var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = 0
    var_11 = 100
    var_12 = module_0.Integer(minimum=var_10, maximum=var_11)
    var_13 = module_1.to_json_schema(var_12)
    var_14 = module_0.Float(minimum=var_10, maximum=var_4)
    var_15 = module_1.to_json_schema(var_14)
    var_16 = module_0.Boolean()
    var_17 = module_1.to_json_schema(var_16)
    var_18 = module_0.String()
    var_19 = module_0.Array(var_18, min_items=var_4, max_items=var_5)
    var_20 = module_1.to_json_schema(var_19)
    var_21 = 'name'
    var_22 = module_0.String()
    var_23 = {var_21: var_22}
    var_24 = [var_21]
    var_25 = module_0.Object(properties=var_23, required=var_24)
    var_26 = module_1.to_json_schema(var_25)
    var_27 = 'a'
    var_28 = (var_27, var_27)
    var_29 = 'b'
    var_30 = (var_29, var_29)
    var_31 = [var_28, var_30]
    var_32 = module_0.Choice(choices=var_31)
    var_33 = module_1.to_json_schema(var_32)
    var_34 = 'fixed'
    var_35 = module_0.Const(var_34)
    var_36 = module_1.to_json_schema(var_35)
    var_37 = module_0.String()
    var_38 = module_0.Integer()
    var_39 = [var_37, var_38]
    var_40 = module_0.Union(var_39)
    var_41 = module_1.to_json_schema(var_40)
    var_42 = module_0.String()
    var_43 = module_0.Integer()
    var_44 = [var_42, var_43]
    var_45 = module_2.OneOf(var_44)
    var_46 = module_1.to_json_schema(var_45)
    var_47 = module_0.String()
    var_48 = module_0.Integer()
    var_49 = [var_47, var_48]
    var_50 = module_2.AllOf(var_49)
    var_51 = module_1.to_json_schema(var_50)
    var_52 = module_0.String()
    var_53 = module_0.Integer()
    var_54 = module_0.Boolean()
    var_55 = module_2.IfThenElse(var_52, var_53, var_54)
    var_56 = module_1.to_json_schema(var_55)
    var_57 = module_0.String()
    var_58 = module_2.Not(var_57)
    var_59 = module_1.to_json_schema(var_58)
    var_60 = module_3.Definitions()
    var_61 = 'test'
    var_62 = module_3.Reference(var_61, var_60)
    var_63 = module_1.to_json_schema(var_62)
    var_64 = module_0.String()
    var_65 = {var_21: var_64}
    var_66 = [var_21]
    var_67 = module_3.Schema(var_65)
    var_68 = module_1.to_json_schema(var_67)
    var_69 = module_3.Definitions()
    var_70 = module_1.to_json_schema(var_69)



# Parsed testcases at query #10
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1
import typesystem.fields as module_2

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
    var_12 = var_9.any_of
    var_13 = var_9.any_of
    var_14 = 'nullable'
    var_15 = True
    var_16 = {var_0: var_1, var_14: var_15}
    var_17 = module_0.Definitions()
    var_18 = module_1.type_from_json_schema(var_16, var_17)
    var_19 = {var_14: var_15}
    var_20 = module_0.Definitions()
    var_21 = module_1.type_from_json_schema(var_19, var_20)
    var_22 = {}
    var_23 = module_0.Definitions()
    var_24 = module_1.type_from_json_schema(var_22, var_23)
    var_25 = module_0.Definitions()
    var_26 = 'name'
    var_27 = module_2.String()
    var_28 = {var_26: var_27}
    var_29 = '$ref'
    var_30 = '#/components/schemas/Test'
    var_31 = {var_29: var_30}
    var_32 = module_1.type_from_json_schema(var_31, var_25)



# Parsed testcases at query #11
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
    var_4 = 1
    var_5 = 10
    var_6 = '[a-z]+'
    var_7 = 'email'
    var_8 = module_0.String(max_length=var_5, min_length=var_4, pattern=var_6, format=var_7)
    var_9 = 'type'
    var_10 = 'minLength'
    var_11 = 'maxLength'
    var_12 = 'pattern'
    var_13 = 'format'
    var_14 = 'string'
    var_15 = {var_9: var_14, var_10: var_4, var_11: var_5, var_12: var_6, var_13: var_7}
    var_16 = module_1.to_json_schema(var_8)
    var_17 = 0
    var_18 = 100
    var_19 = True
    var_20 = module_0.Integer(minimum=var_17, maximum=var_18, exclusive_minimum=var_19)
    var_21 = 'minimum'
    var_22 = 'maximum'
    var_23 = 'exclusiveMinimum'
    var_24 = 'integer'
    var_25 = True
    var_26 = {var_9: var_24, var_21: var_17, var_22: var_18, var_23: var_25}
    var_27 = module_1.to_json_schema(var_20)
    var_28 = 0.5
    var_29 = module_0.Float(multiple_of=var_28)
    var_30 = 'multipleOf'
    var_31 = 'number'
    var_32 = {var_9: var_31, var_30: var_28}
    var_33 = module_1.to_json_schema(var_29)
    var_34 = module_0.Boolean()
    var_35 = 'boolean'
    var_36 = {var_9: var_35}
    var_37 = module_1.to_json_schema(var_34)
    var_38 = module_0.String()
    var_39 = 5
    var_40 = True
    var_41 = module_0.Array(var_38, min_items=var_25, max_items=var_39, unique_items=var_40)
    var_42 = 'items'
    var_43 = 'minItems'
    var_44 = 'maxItems'
    var_45 = 'uniqueItems'
    var_46 = 'array'
    var_47 = {var_9: var_14}
    var_48 = True
    var_49 = {var_9: var_46, var_42: var_47, var_43: var_40, var_44: var_39, var_45: var_48}
    var_50 = module_1.to_json_schema(var_41)
    var_51 = 'name'
    var_52 = module_0.String()
    var_53 = {var_51: var_52}
    var_54 = [var_51]
    var_55 = module_0.Object(properties=var_53, min_properties=var_48, max_properties=var_39, required=var_54)
    var_56 = 'properties'
    var_57 = 'required'
    var_58 = 'minProperties'
    var_59 = 'maxProperties'
    var_60 = 'object'
    var_61 = {var_9: var_14}
    var_62 = {var_51: var_61}
    var_63 = [var_51]
    var_64 = {var_9: var_60, var_56: var_62, var_57: var_63, var_58: var_48, var_59: var_39}
    var_65 = module_1.to_json_schema(var_55)
    var_66 = 'a'
    var_67 = (var_66, var_66)
    var_68 = 'b'
    var_69 = (var_68, var_68)
    var_70 = [var_67, var_69]
    var_71 = module_0.Choice(choices=var_70)
    var_72 = 'enum'
    var_73 = [var_66, var_68]
    var_74 = {var_72: var_73}
    var_75 = module_1.to_json_schema(var_71)
    var_76 = 'fixed_value'
    var_77 = module_0.Const(var_76)
    var_78 = 'const'
    var_79 = {var_78: var_76}
    var_80 = module_1.to_json_schema(var_77)
    var_81 = module_0.String()
    var_82 = module_0.Integer()
    var_83 = [var_81, var_82]
    var_84 = module_0.Union(var_83)
    var_85 = 'anyOf'
    var_86 = {var_9: var_14}
    var_87 = {var_9: var_24}
    var_88 = [var_86, var_87]
    var_89 = {var_85: var_88}
    var_90 = module_1.to_json_schema(var_84)
    var_91 = module_0.String()
    var_92 = module_0.Integer()
    var_93 = [var_91, var_92]
    var_94 = module_2.AllOf(var_93)
    var_95 = 'allOf'
    var_96 = {var_9: var_14}
    var_97 = {var_9: var_24}
    var_98 = [var_96, var_97]
    var_99 = {var_95: var_98}
    var_100 = module_1.to_json_schema(var_94)
    var_101 = module_0.String()
    var_102 = module_0.Integer()
    var_103 = [var_101, var_102]
    var_104 = module_2.OneOf(var_103)
    var_105 = 'oneOf'
    var_106 = {var_9: var_14}
    var_107 = {var_9: var_24}
    var_108 = [var_106, var_107]
    var_109 = {var_105: var_108}
    var_110 = module_1.to_json_schema(var_104)
    var_111 = module_0.String()
    var_112 = module_2.Not(var_111)
    var_113 = 'not'
    var_114 = {var_9: var_14}
    var_115 = {var_113: var_114}
    var_116 = module_1.to_json_schema(var_112)
    var_117 = module_0.String()
    var_118 = module_0.Integer()
    var_119 = module_0.Boolean()
    var_120 = module_2.IfThenElse(var_117, var_118, var_119)
    var_121 = 'if'
    var_122 = 'then'
    var_123 = 'else'
    var_124 = {var_9: var_14}
    var_125 = {var_9: var_24}
    var_126 = {var_9: var_35}
    var_127 = {var_121: var_124, var_122: var_125, var_123: var_126}
    var_128 = module_1.to_json_schema(var_120)
    var_129 = module_3.Definitions()
    var_130 = 'test_ref'
    var_131 = module_3.Reference(var_130, var_129)
    var_132 = '$ref'
    var_133 = 'components'
    var_134 = '#/components/schemas/test_ref'
    var_135 = 'schemas'
    var_136 = {}
    var_137 = {var_135: var_136}
    var_138 = {var_132: var_134, var_133: var_137}
    var_139 = module_1.to_json_schema(var_131)
    var_140 = module_0.String()
    var_141 = {var_51: var_140}
    var_142 = [var_51]
    var_143 = module_3.Schema(var_141)
    var_144 = {var_9: var_14}
    var_145 = {var_51: var_144}
    var_146 = [var_51]
    var_147 = {var_9: var_60, var_56: var_145, var_57: var_146}
    var_148 = module_1.to_json_schema(var_143)
    var_149 = True
    var_150 = module_0.String()
    var_151 = 'null'
    var_152 = [var_14, var_151]
    var_153 = {var_9: var_152}
    var_154 = module_1.to_json_schema(var_150)
    var_155 = 'default_value'
    var_156 = module_0.String()
    var_157 = 'default'
    var_158 = {var_9: var_14, var_157: var_155}
    var_159 = module_1.to_json_schema(var_156)



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
    var_4 = True
    var_5 = 10
    var_6 = '[a-z]+'
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
    var_19 = False
    var_20 = 100
    var_21 = module_0.Integer(minimum=var_19, maximum=var_20, exclusive_minimum=var_4)
    var_22 = 'minimum'
    var_23 = 'maximum'
    var_24 = 'exclusiveMinimum'
    var_25 = 'integer'
    var_26 = {var_9: var_25, var_22: var_19, var_23: var_20, var_24: var_4}
    var_27 = module_1.to_json_schema(var_21)
    var_28 = 0.5
    var_29 = module_0.Float(multiple_of=var_28)
    var_30 = 'multipleOf'
    var_31 = 'number'
    var_32 = [var_31, var_15]
    var_33 = {var_9: var_32, var_30: var_28}
    var_34 = module_1.to_json_schema(var_29)
    var_35 = module_0.Boolean()
    var_36 = 'boolean'
    var_37 = {var_9: var_36}
    var_38 = module_1.to_json_schema(var_35)
    var_39 = module_0.String()
    var_40 = 5
    var_41 = module_0.Array(var_39, var_19, var_4, var_40, unique_items=var_4)
    var_42 = 'items'
    var_43 = 'additionalItems'
    var_44 = 'minItems'
    var_45 = 'maxItems'
    var_46 = 'uniqueItems'
    var_47 = 'array'
    var_48 = [var_47, var_15]
    var_49 = {var_9: var_14}
    var_50 = {var_9: var_48, var_42: var_49, var_43: var_19, var_44: var_4, var_45: var_40, var_46: var_4}
    var_51 = module_1.to_json_schema(var_41)
    var_52 = 'name'
    var_53 = module_0.String()
    var_54 = {var_52: var_53}
    var_55 = [var_52]
    var_56 = module_0.Object(properties=var_54, additional_properties=var_19, min_properties=var_4, max_properties=var_5, required=var_55)
    var_57 = 'properties'
    var_58 = 'additionalProperties'
    var_59 = 'minProperties'
    var_60 = 'maxProperties'
    var_61 = 'required'
    var_62 = 'object'
    var_63 = {var_9: var_14}
    var_64 = {var_52: var_63}
    var_65 = [var_52]
    var_66 = {var_9: var_62, var_57: var_64, var_58: var_19, var_59: var_4, var_60: var_5, var_61: var_65}
    var_67 = module_1.to_json_schema(var_56)
    var_68 = 'a'
    var_69 = (var_68, var_68)
    var_70 = 'b'
    var_71 = (var_70, var_70)
    var_72 = [var_69, var_71]
    var_73 = module_0.Choice(choices=var_72)
    var_74 = 'enum'
    var_75 = 'default'
    var_76 = [var_68, var_70]
    var_77 = {var_74: var_76, var_75: var_68}
    var_78 = module_1.to_json_schema(var_73)
    var_79 = 42
    var_80 = module_0.Const(var_79)
    var_81 = 'const'
    var_82 = {var_81: var_79, var_75: var_79}
    var_83 = module_1.to_json_schema(var_80)
    var_84 = module_0.String()
    var_85 = module_0.Integer()
    var_86 = [var_84, var_85]
    var_87 = module_0.Union(var_86)
    var_88 = 'anyOf'
    var_89 = {var_9: var_14}
    var_90 = {var_9: var_25}
    var_91 = [var_89, var_90]
    var_92 = {var_88: var_91}
    var_93 = module_1.to_json_schema(var_87)
    var_94 = module_0.String()
    var_95 = 'test'
    var_96 = module_0.Const(var_95)
    var_97 = [var_94, var_96]
    var_98 = module_2.AllOf(var_97)
    var_99 = 'allOf'
    var_100 = {var_9: var_14}
    var_101 = {var_81: var_95}
    var_102 = [var_100, var_101]
    var_103 = {var_99: var_102}
    var_104 = module_1.to_json_schema(var_98)
    var_105 = 'Test'
    var_106 = module_0.String()
    var_107 = {var_105: var_106}
    var_108 = '$ref'
    var_109 = 'components'
    var_110 = '#/components/schemas/Test'
    var_111 = 'schemas'
    var_112 = {var_9: var_14}
    var_113 = {var_105: var_112}
    var_114 = {var_111: var_113}
    var_115 = {var_108: var_110, var_109: var_114}
    var_116 = module_0.Const(var_4)
    var_117 = module_0.String()
    var_118 = module_0.Integer()
    var_119 = module_2.IfThenElse(var_116, var_117, var_118)
    var_120 = 'if'
    var_121 = 'then'
    var_122 = 'else'
    var_123 = {var_81: var_4}
    var_124 = {var_9: var_14}
    var_125 = {var_9: var_25}
    var_126 = {var_120: var_123, var_121: var_124, var_122: var_125}
    var_127 = module_1.to_json_schema(var_119)
    var_128 = module_0.String()
    var_129 = module_2.Not(var_128)
    var_130 = 'not'
    var_131 = {var_9: var_14}
    var_132 = {var_130: var_131}
    var_133 = module_1.to_json_schema(var_129)
    var_134 = module_0.String()
    var_135 = {var_52: var_134}
    var_136 = [var_52]
    var_137 = module_3.Schema(var_135)
    var_138 = {var_9: var_14}
    var_139 = {var_52: var_138}
    var_140 = [var_52]
    var_141 = {var_9: var_62, var_57: var_139, var_61: var_140}
    var_142 = module_1.to_json_schema(var_137)



# Parsed testcases at query #13
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
    var_30 = module_0.from_json_schema(var_28)
    var_31 = var_30.choices
    var_32 = 'const'
    var_33 = 'value'
    var_34 = {var_32: var_33}
    var_35 = module_0.from_json_schema(var_34)
    var_36 = module_0.from_json_schema(var_34)
    var_37 = var_36.value
    assert var_37 == 'value'
    var_38 = 'allOf'
    var_39 = {var_4: var_5}
    var_40 = 'minLength'
    var_41 = 5
    var_42 = {var_40: var_41}
    var_43 = [var_39, var_42]
    var_44 = {var_38: var_43}
    var_45 = module_0.from_json_schema(var_44)
    var_46 = var_45.schemas
    var_47 = len(var_46)
    assert var_47 == 2
    var_48 = 'anyOf'
    var_49 = {var_4: var_5}
    var_50 = {var_4: var_8}
    var_51 = [var_49, var_50]
    var_52 = {var_48: var_51}
    var_53 = module_0.from_json_schema(var_52)
    var_54 = var_53.schemas
    var_55 = len(var_54)
    assert var_55 == 2
    var_56 = 'oneOf'
    var_57 = {var_4: var_5}
    var_58 = {var_4: var_8}
    var_59 = [var_57, var_58]
    var_60 = {var_56: var_59}
    var_61 = module_0.from_json_schema(var_60)
    var_62 = var_61.schemas
    var_63 = len(var_62)
    assert var_63 == 2
    var_64 = 'not'
    var_65 = {var_4: var_5}
    var_66 = {var_64: var_65}
    var_67 = module_0.from_json_schema(var_66)
    var_68 = var_67.schema
    var_69 = 'if'
    var_70 = 'then'
    var_71 = 'else'
    var_72 = {var_4: var_5}
    var_73 = {var_40: var_41}
    var_74 = {var_4: var_8}
    var_75 = {var_69: var_72, var_70: var_73, var_71: var_74}
    var_76 = module_0.from_json_schema(var_75)
    var_77 = var_76.if_schema
    var_78 = var_76.then_schema
    var_79 = var_76.else_schema
    var_80 = module_1.Definitions()
    var_81 = '$ref'
    var_82 = '#/components/schemas/Test'
    var_83 = {var_81: var_82}
    var_84 = module_0.from_json_schema(var_83, var_80)
    var_85 = 'maxLength'
    var_86 = 'pattern'
    var_87 = 10
    var_88 = '^[a-zA-Z]+$'
    var_89 = {var_4: var_5, var_40: var_41, var_85: var_87, var_86: var_88}
    var_90 = module_0.from_json_schema(var_89)
    var_91 = var_90.schemas
    var_92 = len(var_91)
    assert var_92 == 2
    var_93 = {}
    var_94 = module_0.from_json_schema(var_93)



# Parsed testcases at query #14
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
    var_8 = 0
    var_9 = 100
    var_10 = 2
    var_11 = 50
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
    var_26 = 5
    var_27 = 'email'
    var_28 = '^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$'
    var_29 = 'test@example.com'
    var_30 = {var_0: var_25, var_21: var_26, var_22: var_9, var_23: var_27, var_24: var_28, var_6: var_29}
    var_31 = False
    var_32 = module_0.Definitions()
    var_33 = module_1.from_json_schema_type(var_30, var_25, var_31, var_32)
    var_34 = 'boolean'
    var_35 = True
    var_36 = {var_0: var_34, var_6: var_35}
    var_37 = False
    var_38 = module_0.Definitions()
    var_39 = module_1.from_json_schema_type(var_36, var_34, var_37, var_38)
    var_40 = 'items'
    var_41 = 'minItems'
    var_42 = 'maxItems'
    var_43 = 'uniqueItems'
    var_44 = 'array'
    var_45 = {var_0: var_25}
    var_46 = 10
    var_47 = 'test'
    var_48 = [var_47]
    var_49 = {var_0: var_44, var_40: var_45, var_41: var_35, var_42: var_46, var_43: var_35, var_6: var_48}
    var_50 = False
    var_51 = module_0.Definitions()
    var_52 = module_1.from_json_schema_type(var_49, var_44, var_50, var_51)
    var_53 = var_52.items
    var_54 = 'properties'
    var_55 = 'required'
    var_56 = 'object'
    var_57 = 'name'
    var_58 = 'age'
    var_59 = {var_0: var_25}
    var_60 = {var_0: var_16}
    var_61 = {var_57: var_59, var_58: var_60}
    var_62 = [var_57]
    var_63 = 'John'
    var_64 = 30
    var_65 = {var_57: var_63, var_58: var_64}
    var_66 = {var_0: var_56, var_54: var_61, var_55: var_62, var_6: var_65}
    var_67 = False
    var_68 = module_0.Definitions()
    var_69 = module_1.from_json_schema_type(var_66, var_56, var_67, var_68)
    var_70 = var_69.properties[var_57]
    var_71 = var_69.properties[var_58]
    var_72 = {var_0: var_25}
    var_73 = module_0.Definitions()
    var_74 = module_1.from_json_schema_type(var_72, var_25, var_35, var_73)



# Parsed testcases at query #15
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
    var_8 = 0
    var_9 = 100
    var_10 = 2
    var_11 = 50
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
    var_26 = 5
    var_27 = 10
    var_28 = 'email'
    var_29 = '^[a-zA-Z0-9]+$'
    var_30 = 'test'
    var_31 = {var_0: var_25, var_21: var_26, var_22: var_27, var_23: var_28, var_24: var_29, var_6: var_30}
    var_32 = False
    var_33 = module_0.Definitions()
    var_34 = module_1.from_json_schema_type(var_31, var_25, var_32, var_33)
    var_35 = 'boolean'
    var_36 = True
    var_37 = {var_0: var_35, var_6: var_36}
    var_38 = False
    var_39 = module_0.Definitions()
    var_40 = module_1.from_json_schema_type(var_37, var_35, var_38, var_39)
    var_41 = 'items'
    var_42 = 'additionalItems'
    var_43 = 'minItems'
    var_44 = 'maxItems'
    var_45 = 'uniqueItems'
    var_46 = 'array'
    var_47 = {var_0: var_25}
    var_48 = False
    var_49 = [var_30]
    var_50 = {var_0: var_46, var_41: var_47, var_42: var_48, var_43: var_36, var_44: var_27, var_45: var_36, var_6: var_49}
    var_51 = False
    var_52 = module_0.Definitions()
    var_53 = module_1.from_json_schema_type(var_50, var_46, var_51, var_52)
    var_54 = var_53.items
    var_55 = 'properties'
    var_56 = 'patternProperties'
    var_57 = 'additionalProperties'
    var_58 = 'propertyNames'
    var_59 = 'minProperties'
    var_60 = 'maxProperties'
    var_61 = 'required'
    var_62 = 'object'
    var_63 = 'name'
    var_64 = 'age'
    var_65 = {var_0: var_25}
    var_66 = {var_0: var_16}
    var_67 = {var_63: var_65, var_64: var_66}
    var_68 = '^S_'
    var_69 = '^I_'
    var_70 = {var_0: var_25}
    var_71 = {var_0: var_16}
    var_72 = {var_68: var_70, var_69: var_71}
    var_73 = False
    var_74 = {var_0: var_25}
    var_75 = [var_63]
    var_76 = 25
    var_77 = {var_63: var_30, var_64: var_76}
    var_78 = {var_0: var_62, var_55: var_67, var_56: var_72, var_57: var_73, var_58: var_74, var_59: var_36, var_60: var_27, var_61: var_75, var_6: var_77}
    var_79 = False
    var_80 = module_0.Definitions()
    var_81 = module_1.from_json_schema_type(var_78, var_62, var_79, var_80)
    var_82 = var_81.properties[var_63]
    var_83 = var_81.properties[var_64]
    var_84 = var_81.pattern_properties[var_68]
    var_85 = var_81.pattern_properties[var_69]
    var_86 = var_81.property_names



# Parsed testcases at query #16
#--------------------------


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
    var_12 = 'integer'
    var_13 = {var_0: var_12, var_1: var_9, var_2: var_6, var_3: var_7}
    var_14 = False
    var_15 = module_0.Definitions()
    var_16 = module_1.from_json_schema_type(var_13, var_12, var_14, var_15)
    var_17 = 'minLength'
    var_18 = 'maxLength'
    var_19 = 'pattern'
    var_20 = 'string'
    var_21 = 5
    var_22 = '^[a-zA-Z]+$'
    var_23 = 'test'
    var_24 = {var_0: var_20, var_17: var_21, var_18: var_6, var_19: var_22, var_3: var_23}
    var_25 = False
    var_26 = module_0.Definitions()
    var_27 = module_1.from_json_schema_type(var_24, var_20, var_25, var_26)
    var_28 = 'boolean'
    var_29 = True
    var_30 = {var_0: var_28, var_3: var_29}
    var_31 = False
    var_32 = module_0.Definitions()
    var_33 = module_1.from_json_schema_type(var_30, var_28, var_31, var_32)
    var_34 = 'items'
    var_35 = 'minItems'
    var_36 = 'maxItems'
    var_37 = 'uniqueItems'
    var_38 = 'array'
    var_39 = {var_0: var_20}
    var_40 = 10
    var_41 = [var_23]
    var_42 = {var_0: var_38, var_34: var_39, var_35: var_29, var_36: var_40, var_37: var_29, var_3: var_41}
    var_43 = False
    var_44 = module_0.Definitions()
    var_45 = module_1.from_json_schema_type(var_42, var_38, var_43, var_44)
    var_46 = var_45.items
    var_47 = 'properties'
    var_48 = 'required'
    var_49 = 'object'
    var_50 = 'name'
    var_51 = 'age'
    var_52 = {var_0: var_20}
    var_53 = {var_0: var_12}
    var_54 = {var_50: var_52, var_51: var_53}
    var_55 = [var_50]
    var_56 = 25
    var_57 = {var_50: var_23, var_51: var_56}
    var_58 = {var_0: var_49, var_47: var_54, var_48: var_55, var_3: var_57}
    var_59 = False
    var_60 = module_0.Definitions()
    var_61 = module_1.from_json_schema_type(var_58, var_49, var_59, var_60)
    var_62 = var_61.properties[var_50]
    var_63 = var_61.properties[var_51]
    var_64 = {var_0: var_20}
    var_65 = module_0.Definitions()
    var_66 = module_1.from_json_schema_type(var_64, var_20, var_29, var_65)



# Parsed testcases at query #17
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = module_0.Definitions()
    var_4 = module_1.type_from_json_schema(var_2, var_3)
    var_5 = 'integer'
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
    var_17 = [var_1, var_16]
    var_18 = {var_0: var_17}
    var_19 = module_0.Definitions()
    var_20 = module_1.type_from_json_schema(var_18, var_19)
    var_21 = []
    var_22 = {var_0: var_21}
    var_23 = module_0.Definitions()
    var_24 = module_1.type_from_json_schema(var_22, var_23)
    var_25 = [var_16]
    var_26 = {var_0: var_25}
    var_27 = module_0.Definitions()
    var_28 = module_1.type_from_json_schema(var_26, var_27)
    var_29 = 'minLength'
    var_30 = 'maxLength'
    var_31 = 5
    var_32 = 10
    var_33 = {var_0: var_1, var_29: var_31, var_30: var_32}
    var_34 = module_0.Definitions()
    var_35 = module_1.type_from_json_schema(var_33, var_34)
    var_36 = 'minimum'
    var_37 = 'maximum'
    var_38 = 'number'
    var_39 = 100
    var_40 = {var_0: var_38, var_36: var_12, var_37: var_39}
    var_41 = module_0.Definitions()
    var_42 = module_1.type_from_json_schema(var_40, var_41)
    var_43 = 'minItems'
    var_44 = 'maxItems'
    var_45 = 'array'
    var_46 = {var_0: var_45, var_43: var_14, var_44: var_32}
    var_47 = module_0.Definitions()
    var_48 = module_1.type_from_json_schema(var_46, var_47)
    var_49 = 'minProperties'
    var_50 = 'maxProperties'
    var_51 = 'object'
    var_52 = {var_0: var_51, var_49: var_14, var_50: var_32}
    var_53 = module_0.Definitions()
    var_54 = module_1.type_from_json_schema(var_52, var_53)



# Parsed testcases at query #18
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
    var_8 = 0
    var_9 = 100
    var_10 = 2
    var_11 = 50
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
    var_26 = 5
    var_27 = 'email'
    var_28 = '^[a-zA-Z0-9]+$'
    var_29 = 'test'
    var_30 = {var_0: var_25, var_21: var_26, var_22: var_9, var_23: var_27, var_24: var_28, var_6: var_29}
    var_31 = False
    var_32 = module_0.Definitions()
    var_33 = module_1.from_json_schema_type(var_30, var_25, var_31, var_32)
    var_34 = 'boolean'
    var_35 = True
    var_36 = {var_0: var_34, var_6: var_35}
    var_37 = False
    var_38 = module_0.Definitions()
    var_39 = module_1.from_json_schema_type(var_36, var_34, var_37, var_38)
    var_40 = 'items'
    var_41 = 'minItems'
    var_42 = 'maxItems'
    var_43 = 'uniqueItems'
    var_44 = 'array'
    var_45 = {var_0: var_25}
    var_46 = 10
    var_47 = [var_29]
    var_48 = {var_0: var_44, var_40: var_45, var_41: var_35, var_42: var_46, var_43: var_35, var_6: var_47}
    var_49 = False
    var_50 = module_0.Definitions()
    var_51 = module_1.from_json_schema_type(var_48, var_44, var_49, var_50)
    var_52 = var_51.items
    var_53 = 'properties'
    var_54 = 'required'
    var_55 = 'minProperties'
    var_56 = 'maxProperties'
    var_57 = 'object'
    var_58 = 'name'
    var_59 = 'age'
    var_60 = {var_0: var_25}
    var_61 = {var_0: var_16}
    var_62 = {var_58: var_60, var_59: var_61}
    var_63 = [var_58]
    var_64 = 25
    var_65 = {var_58: var_29, var_59: var_64}
    var_66 = {var_0: var_57, var_53: var_62, var_54: var_63, var_55: var_35, var_56: var_10, var_6: var_65}
    var_67 = False
    var_68 = module_0.Definitions()
    var_69 = module_1.from_json_schema_type(var_66, var_57, var_67, var_68)
    var_70 = var_69.properties[var_58]
    var_71 = var_69.properties[var_59]
    var_72 = {}
    var_73 = 'invalid'
    var_74 = False
    var_75 = module_0.Definitions()
    var_76 = module_1.from_json_schema_type(var_72, var_73, var_74, var_75)



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
    var_4 = 1
    var_5 = 10
    var_6 = '[a-z]+'
    var_7 = 'email'
    var_8 = module_0.String(max_length=var_5, min_length=var_4, pattern=var_6, format=var_7)
    var_9 = 'type'
    var_10 = 'minLength'
    var_11 = 'maxLength'
    var_12 = 'pattern'
    var_13 = 'format'
    var_14 = 'string'
    var_15 = {var_9: var_14, var_10: var_4, var_11: var_5, var_12: var_6, var_13: var_7}
    var_16 = module_1.to_json_schema(var_8)
    var_17 = 0
    var_18 = 100
    var_19 = True
    var_20 = module_0.Integer(minimum=var_17, maximum=var_18, exclusive_maximum=var_19)
    var_21 = 'minimum'
    var_22 = 'maximum'
    var_23 = 'exclusiveMaximum'
    var_24 = 'integer'
    var_25 = True
    var_26 = {var_9: var_24, var_21: var_17, var_22: var_18, var_23: var_25}
    var_27 = module_1.to_json_schema(var_20)
    var_28 = 0.5
    var_29 = module_0.Float(multiple_of=var_28)
    var_30 = 'multipleOf'
    var_31 = 'number'
    var_32 = {var_9: var_31, var_30: var_28}
    var_33 = module_1.to_json_schema(var_29)
    var_34 = module_0.Boolean()
    var_35 = 'boolean'
    var_36 = {var_9: var_35}
    var_37 = module_1.to_json_schema(var_34)
    var_38 = module_0.String()
    var_39 = 5
    var_40 = True
    var_41 = module_0.Array(var_38, min_items=var_25, max_items=var_39, unique_items=var_40)
    var_42 = 'items'
    var_43 = 'minItems'
    var_44 = 'maxItems'
    var_45 = 'uniqueItems'
    var_46 = 'array'
    var_47 = {var_9: var_14}
    var_48 = True
    var_49 = {var_9: var_46, var_42: var_47, var_43: var_40, var_44: var_39, var_45: var_48}
    var_50 = module_1.to_json_schema(var_41)
    var_51 = 'name'
    var_52 = module_0.String()
    var_53 = {var_51: var_52}
    var_54 = [var_51]
    var_55 = False
    var_56 = module_0.Object(properties=var_53, additional_properties=var_55, required=var_54)
    var_57 = 'properties'
    var_58 = 'required'
    var_59 = 'additionalProperties'
    var_60 = 'object'
    var_61 = {var_9: var_14}
    var_62 = {var_51: var_61}
    var_63 = [var_51]
    var_64 = False
    var_65 = {var_9: var_60, var_57: var_62, var_58: var_63, var_59: var_64}
    var_66 = module_1.to_json_schema(var_56)
    var_67 = 'a'
    var_68 = (var_67, var_67)
    var_69 = 'b'
    var_70 = (var_69, var_69)
    var_71 = [var_68, var_70]
    var_72 = module_0.Choice(choices=var_71)
    var_73 = 'enum'
    var_74 = [var_67, var_69]
    var_75 = {var_73: var_74}
    var_76 = module_1.to_json_schema(var_72)
    var_77 = 'fixed_value'
    var_78 = module_0.Const(var_77)
    var_79 = 'const'
    var_80 = {var_79: var_77}
    var_81 = module_1.to_json_schema(var_78)
    var_82 = module_0.String()
    var_83 = module_0.Integer()
    var_84 = [var_82, var_83]
    var_85 = module_0.Union(var_84)
    var_86 = 'anyOf'
    var_87 = {var_9: var_14}
    var_88 = {var_9: var_24}
    var_89 = [var_87, var_88]
    var_90 = {var_86: var_89}
    var_91 = module_1.to_json_schema(var_85)
    var_92 = module_0.String()
    var_93 = 'test'
    var_94 = module_0.Const(var_93)
    var_95 = [var_92, var_94]
    var_96 = module_2.AllOf(var_95)
    var_97 = 'allOf'
    var_98 = {var_9: var_14}
    var_99 = {var_79: var_93}
    var_100 = [var_98, var_99]
    var_101 = {var_97: var_100}
    var_102 = module_1.to_json_schema(var_96)
    var_103 = 'TestSchema'
    var_104 = module_0.String()
    var_105 = {var_103: var_104}
    var_106 = '$ref'
    var_107 = 'components'
    var_108 = '#/components/schemas/TestSchema'
    var_109 = 'schemas'
    var_110 = {var_9: var_14}
    var_111 = {var_103: var_110}
    var_112 = {var_109: var_111}
    var_113 = {var_106: var_108, var_107: var_112}
    var_114 = module_0.String()
    var_115 = {var_51: var_114}
    var_116 = [var_51]
    var_117 = module_3.Schema(var_115)
    var_118 = {var_9: var_14}
    var_119 = {var_51: var_118}
    var_120 = [var_51]
    var_121 = {var_9: var_60, var_57: var_119, var_58: var_120}
    var_122 = module_1.to_json_schema(var_117)
    var_123 = module_0.String()
    var_124 = module_0.Integer()
    var_125 = module_0.Boolean()
    var_126 = module_2.IfThenElse(var_123, var_124, var_125)
    var_127 = 'if'
    var_128 = 'then'
    var_129 = 'else'
    var_130 = {var_9: var_14}
    var_131 = {var_9: var_24}
    var_132 = {var_9: var_35}
    var_133 = {var_127: var_130, var_128: var_131, var_129: var_132}
    var_134 = module_1.to_json_schema(var_126)
    var_135 = module_0.String()
    var_136 = module_2.Not(var_135)
    var_137 = 'not'
    var_138 = {var_9: var_14}
    var_139 = {var_137: var_138}
    var_140 = module_1.to_json_schema(var_136)



# Parsed testcases at query #20
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
    var_8 = 0
    var_9 = 100
    var_10 = 2
    var_11 = 50
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
    var_26 = 5
    var_27 = 'email'
    var_28 = '^[a-zA-Z0-9]+$'
    var_29 = 'test'
    var_30 = {var_0: var_25, var_21: var_26, var_22: var_9, var_23: var_27, var_24: var_28, var_6: var_29}
    var_31 = False
    var_32 = module_0.Definitions()
    var_33 = module_1.from_json_schema_type(var_30, var_25, var_31, var_32)
    var_34 = 'boolean'
    var_35 = True
    var_36 = {var_0: var_34, var_6: var_35}
    var_37 = False
    var_38 = module_0.Definitions()
    var_39 = module_1.from_json_schema_type(var_36, var_34, var_37, var_38)
    var_40 = 'items'
    var_41 = 'additionalItems'
    var_42 = 'minItems'
    var_43 = 'maxItems'
    var_44 = 'uniqueItems'
    var_45 = 'array'
    var_46 = {var_0: var_25}
    var_47 = False
    var_48 = 10
    var_49 = [var_29]
    var_50 = {var_0: var_45, var_40: var_46, var_41: var_47, var_42: var_35, var_43: var_48, var_44: var_35, var_6: var_49}
    var_51 = False
    var_52 = module_0.Definitions()
    var_53 = module_1.from_json_schema_type(var_50, var_45, var_51, var_52)
    var_54 = var_53.items
    var_55 = 'properties'
    var_56 = 'patternProperties'
    var_57 = 'additionalProperties'
    var_58 = 'propertyNames'
    var_59 = 'minProperties'
    var_60 = 'maxProperties'
    var_61 = 'required'
    var_62 = 'object'
    var_63 = 'name'
    var_64 = 'age'
    var_65 = {var_0: var_25}
    var_66 = {var_0: var_16}
    var_67 = {var_63: var_65, var_64: var_66}
    var_68 = '^S_'
    var_69 = '^I_'
    var_70 = {var_0: var_25}
    var_71 = {var_0: var_16}
    var_72 = {var_68: var_70, var_69: var_71}
    var_73 = False
    var_74 = {var_0: var_25}
    var_75 = [var_63]
    var_76 = 30
    var_77 = {var_63: var_29, var_64: var_76}
    var_78 = {var_0: var_62, var_55: var_67, var_56: var_72, var_57: var_73, var_58: var_74, var_59: var_35, var_60: var_48, var_61: var_75, var_6: var_77}
    var_79 = False
    var_80 = module_0.Definitions()
    var_81 = module_1.from_json_schema_type(var_78, var_62, var_79, var_80)
    var_82 = var_81.properties[var_63]
    var_83 = var_81.properties[var_64]
    var_84 = var_81.pattern_properties[var_68]
    var_85 = var_81.pattern_properties[var_69]
    var_86 = var_81.property_names
    var_87 = {var_0: var_25}
    var_88 = module_0.Definitions()
    var_89 = module_1.from_json_schema_type(var_87, var_25, var_35, var_88)



# Parsed testcases at query #21
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
    var_4 = True
    var_5 = 10
    var_6 = '[a-z]+'
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
    var_19 = False
    var_20 = 100
    var_21 = module_0.Integer(minimum=var_19, maximum=var_20, exclusive_minimum=var_4, exclusive_maximum=var_4)
    var_22 = 'minimum'
    var_23 = 'maximum'
    var_24 = 'exclusiveMinimum'
    var_25 = 'exclusiveMaximum'
    var_26 = 'integer'
    var_27 = {var_9: var_26, var_22: var_19, var_23: var_20, var_24: var_4, var_25: var_4}
    var_28 = module_1.to_json_schema(var_21)
    var_29 = 0.1
    var_30 = module_0.Float(minimum=var_19, maximum=var_4, multiple_of=var_29)
    var_31 = 'multipleOf'
    var_32 = 'number'
    var_33 = [var_32, var_15]
    var_34 = {var_9: var_33, var_22: var_19, var_23: var_4, var_31: var_29}
    var_35 = module_1.to_json_schema(var_30)
    var_36 = module_0.Boolean()
    var_37 = 'boolean'
    var_38 = [var_37, var_15]
    var_39 = {var_9: var_38}
    var_40 = module_1.to_json_schema(var_36)
    var_41 = module_0.String()
    var_42 = module_0.Array(var_41, var_19, var_4, var_5)
    var_43 = 'items'
    var_44 = 'additionalItems'
    var_45 = 'minItems'
    var_46 = 'maxItems'
    var_47 = 'array'
    var_48 = {var_9: var_14}
    var_49 = {var_9: var_47, var_43: var_48, var_44: var_19, var_45: var_4, var_46: var_5}
    var_50 = module_1.to_json_schema(var_42)
    var_51 = 'name'
    var_52 = module_0.String()
    var_53 = {var_51: var_52}
    var_54 = [var_51]
    var_55 = module_0.Object(properties=var_53, additional_properties=var_19, required=var_54)
    var_56 = 'properties'
    var_57 = 'additionalProperties'
    var_58 = 'required'
    var_59 = 'object'
    var_60 = [var_59, var_15]
    var_61 = {var_9: var_14}
    var_62 = {var_51: var_61}
    var_63 = [var_51]
    var_64 = {var_9: var_60, var_56: var_62, var_57: var_19, var_58: var_63}
    var_65 = module_1.to_json_schema(var_55)
    var_66 = 'a'
    var_67 = (var_66, var_66)
    var_68 = 'b'
    var_69 = (var_68, var_68)
    var_70 = [var_67, var_69]
    var_71 = module_0.Choice(choices=var_70)
    var_72 = 'enum'
    var_73 = [var_66, var_68]
    var_74 = {var_72: var_73}
    var_75 = module_1.to_json_schema(var_71)
    var_76 = 'fixed_value'
    var_77 = module_0.Const(var_76)
    var_78 = 'const'
    var_79 = {var_78: var_76}
    var_80 = module_1.to_json_schema(var_77)
    var_81 = module_0.String()
    var_82 = module_0.Integer()
    var_83 = [var_81, var_82]
    var_84 = module_0.Union(var_83)
    var_85 = 'anyOf'
    var_86 = {var_9: var_14}
    var_87 = {var_9: var_26}
    var_88 = [var_86, var_87]
    var_89 = {var_85: var_88}
    var_90 = module_1.to_json_schema(var_84)
    var_91 = module_0.String()
    var_92 = 'test'
    var_93 = module_0.Const(var_92)
    var_94 = [var_91, var_93]
    var_95 = module_2.AllOf(var_94)
    var_96 = 'allOf'
    var_97 = {var_9: var_14}
    var_98 = {var_78: var_92}
    var_99 = [var_97, var_98]
    var_100 = {var_96: var_99}
    var_101 = module_1.to_json_schema(var_95)
    var_102 = module_0.String()
    var_103 = module_0.Integer()
    var_104 = [var_102, var_103]
    var_105 = module_2.OneOf(var_104)
    var_106 = 'oneOf'
    var_107 = {var_9: var_14}
    var_108 = {var_9: var_26}
    var_109 = [var_107, var_108]
    var_110 = {var_106: var_109}
    var_111 = module_1.to_json_schema(var_105)
    var_112 = module_0.String()
    var_113 = module_0.Integer()
    var_114 = module_0.Boolean()
    var_115 = module_2.IfThenElse(var_112, var_113, var_114)
    var_116 = 'if'
    var_117 = 'then'
    var_118 = 'else'
    var_119 = {var_9: var_14}
    var_120 = {var_9: var_26}
    var_121 = {var_9: var_37}
    var_122 = {var_116: var_119, var_117: var_120, var_118: var_121}
    var_123 = module_1.to_json_schema(var_115)
    var_124 = module_0.String()
    var_125 = module_2.Not(var_124)
    var_126 = 'not'
    var_127 = {var_9: var_14}
    var_128 = {var_126: var_127}
    var_129 = module_1.to_json_schema(var_125)
    var_130 = 'Test'
    var_131 = module_0.String()
    var_132 = {var_130: var_131}
    var_133 = '$ref'
    var_134 = 'components'
    var_135 = '#/components/schemas/Test'
    var_136 = 'schemas'
    var_137 = {var_9: var_14}
    var_138 = {var_130: var_137}
    var_139 = {var_136: var_138}
    var_140 = {var_133: var_135, var_134: var_139}



# Parsed testcases at query #22
#--------------------------


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
    var_7 = 50.5
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = False
    var_10 = module_0.Definitions()
    var_11 = module_1.from_json_schema_type(var_8, var_4, var_9, var_10)
    var_12 = 'integer'
    var_13 = 50
    var_14 = {var_0: var_12, var_1: var_9, var_2: var_6, var_3: var_13}
    var_15 = False
    var_16 = module_0.Definitions()
    var_17 = module_1.from_json_schema_type(var_14, var_12, var_15, var_16)
    var_18 = 'minLength'
    var_19 = 'maxLength'
    var_20 = 'pattern'
    var_21 = 'string'
    var_22 = 5
    var_23 = '^[A-Za-z]+$'
    var_24 = 'hello'
    var_25 = {var_0: var_21, var_18: var_22, var_19: var_6, var_20: var_23, var_3: var_24}
    var_26 = False
    var_27 = module_0.Definitions()
    var_28 = module_1.from_json_schema_type(var_25, var_21, var_26, var_27)
    var_29 = 'boolean'
    var_30 = True
    var_31 = {var_0: var_29, var_3: var_30}
    var_32 = False
    var_33 = module_0.Definitions()
    var_34 = module_1.from_json_schema_type(var_31, var_29, var_32, var_33)
    var_35 = 'items'
    var_36 = 'minItems'
    var_37 = 'maxItems'
    var_38 = 'uniqueItems'
    var_39 = 'array'
    var_40 = {var_0: var_21}
    var_41 = 10
    var_42 = 'item1'
    var_43 = 'item2'
    var_44 = [var_42, var_43]
    var_45 = {var_0: var_39, var_35: var_40, var_36: var_30, var_37: var_41, var_38: var_30, var_3: var_44}
    var_46 = False
    var_47 = module_0.Definitions()
    var_48 = module_1.from_json_schema_type(var_45, var_39, var_46, var_47)
    var_49 = var_48.items
    var_50 = 'properties'
    var_51 = 'required'
    var_52 = 'object'
    var_53 = 'name'
    var_54 = 'age'
    var_55 = {var_0: var_21}
    var_56 = {var_0: var_12}
    var_57 = {var_53: var_55, var_54: var_56}
    var_58 = [var_53]
    var_59 = 'John'
    var_60 = 30
    var_61 = {var_53: var_59, var_54: var_60}
    var_62 = {var_0: var_52, var_50: var_57, var_51: var_58, var_3: var_61}
    var_63 = False
    var_64 = module_0.Definitions()
    var_65 = module_1.from_json_schema_type(var_62, var_52, var_63, var_64)
    var_66 = var_65.properties[var_53]
    var_67 = var_65.properties[var_54]
    var_68 = {var_0: var_21}
    var_69 = module_0.Definitions()
    var_70 = module_1.from_json_schema_type(var_68, var_21, var_30, var_69)



# Parsed testcases at query #23
#--------------------------


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
    var_7 = 50.5
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = False
    var_10 = module_0.Definitions()
    var_11 = module_1.from_json_schema_type(var_8, var_4, var_9, var_10)
    var_12 = 'integer'
    var_13 = 50
    var_14 = {var_0: var_12, var_1: var_9, var_2: var_6, var_3: var_13}
    var_15 = False
    var_16 = module_0.Definitions()
    var_17 = module_1.from_json_schema_type(var_14, var_12, var_15, var_16)
    var_18 = 'minLength'
    var_19 = 'maxLength'
    var_20 = 'format'
    var_21 = 'string'
    var_22 = 5
    var_23 = 10
    var_24 = 'email'
    var_25 = 'test@example.com'
    var_26 = {var_0: var_21, var_18: var_22, var_19: var_23, var_20: var_24, var_3: var_25}
    var_27 = False
    var_28 = module_0.Definitions()
    var_29 = module_1.from_json_schema_type(var_26, var_21, var_27, var_28)
    var_30 = 'boolean'
    var_31 = True
    var_32 = {var_0: var_30, var_3: var_31}
    var_33 = False
    var_34 = module_0.Definitions()
    var_35 = module_1.from_json_schema_type(var_32, var_30, var_33, var_34)
    var_36 = 'items'
    var_37 = 'minItems'
    var_38 = 'maxItems'
    var_39 = 'array'
    var_40 = {var_0: var_21}
    var_41 = 'item1'
    var_42 = 'item2'
    var_43 = [var_41, var_42]
    var_44 = {var_0: var_39, var_36: var_40, var_37: var_31, var_38: var_22, var_3: var_43}
    var_45 = False
    var_46 = module_0.Definitions()
    var_47 = module_1.from_json_schema_type(var_44, var_39, var_45, var_46)
    var_48 = var_47.items
    var_49 = 'properties'
    var_50 = 'required'
    var_51 = 'object'
    var_52 = 'name'
    var_53 = 'age'
    var_54 = {var_0: var_21}
    var_55 = {var_0: var_12}
    var_56 = {var_52: var_54, var_53: var_55}
    var_57 = [var_52]
    var_58 = 'John'
    var_59 = 30
    var_60 = {var_52: var_58, var_53: var_59}
    var_61 = {var_0: var_51, var_49: var_56, var_50: var_57, var_3: var_60}
    var_62 = False
    var_63 = module_0.Definitions()
    var_64 = module_1.from_json_schema_type(var_61, var_51, var_62, var_63)
    var_65 = {var_0: var_21}
    var_66 = module_0.Definitions()
    var_67 = module_1.from_json_schema_type(var_65, var_21, var_31, var_66)



# Parsed testcases at query #24
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
    var_35 = 'minLength'
    var_36 = 5
    var_37 = {var_4: var_5, var_35: var_36}
    var_38 = 'maxLength'
    var_39 = 10
    var_40 = {var_4: var_5, var_38: var_39}
    var_41 = [var_37, var_40]
    var_42 = {var_34: var_41}
    var_43 = module_0.from_json_schema(var_42)
    var_44 = var_43.schemas
    var_45 = len(var_44)
    assert var_45 == 2
    var_46 = 'anyOf'
    var_47 = {var_4: var_5}
    var_48 = {var_4: var_8}
    var_49 = [var_47, var_48]
    var_50 = {var_46: var_49}
    var_51 = module_0.from_json_schema(var_50)
    var_52 = var_51.schemas
    var_53 = len(var_52)
    assert var_53 == 2
    var_54 = 'oneOf'
    var_55 = {var_4: var_5}
    var_56 = {var_4: var_8}
    var_57 = [var_55, var_56]
    var_58 = {var_54: var_57}
    var_59 = module_0.from_json_schema(var_58)
    var_60 = var_59.schemas
    var_61 = len(var_60)
    assert var_61 == 2
    var_62 = 'not'
    var_63 = {var_4: var_5}
    var_64 = {var_62: var_63}
    var_65 = module_0.from_json_schema(var_64)
    var_66 = 'if'
    var_67 = 'then'
    var_68 = 'else'
    var_69 = {var_4: var_5}
    var_70 = {var_35: var_36}
    var_71 = {var_35: var_39}
    var_72 = {var_66: var_69, var_67: var_70, var_68: var_71}
    var_73 = module_0.from_json_schema(var_72)
    var_74 = module_1.Definitions()
    var_75 = '$ref'
    var_76 = '#/components/schemas/Test'
    var_77 = {var_75: var_76}
    var_78 = module_0.from_json_schema(var_77, var_74)
    var_79 = 'pattern'
    var_80 = '^[a-z]+$'
    var_81 = {var_4: var_5, var_35: var_36, var_38: var_39, var_79: var_80}
    var_82 = module_0.from_json_schema(var_81)
    var_83 = var_82.schemas
    var_84 = len(var_83)
    assert var_84 == 2
    var_85 = {}
    var_86 = module_0.from_json_schema(var_85)



# Parsed testcases at query #25
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
    var_6 = '[a-z]+'
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
    var_19 = False
    var_20 = 100
    var_21 = module_0.Integer(minimum=var_19, maximum=var_20, exclusive_minimum=var_4)
    var_22 = 'minimum'
    var_23 = 'maximum'
    var_24 = 'exclusiveMinimum'
    var_25 = 'integer'
    var_26 = {var_9: var_25, var_22: var_19, var_23: var_20, var_24: var_4}
    var_27 = module_1.to_json_schema(var_21)
    var_28 = 0.5
    var_29 = module_0.Float(multiple_of=var_28)
    var_30 = 'multipleOf'
    var_31 = 'number'
    var_32 = [var_31, var_15]
    var_33 = {var_9: var_32, var_30: var_28}
    var_34 = module_1.to_json_schema(var_29)
    var_35 = module_0.Boolean()
    var_36 = 'boolean'
    var_37 = {var_9: var_36}
    var_38 = module_1.to_json_schema(var_35)
    var_39 = module_0.String()
    var_40 = 5
    var_41 = module_0.Array(var_39, min_items=var_4, max_items=var_40)
    var_42 = 'items'
    var_43 = 'minItems'
    var_44 = 'maxItems'
    var_45 = 'array'
    var_46 = [var_45, var_15]
    var_47 = {var_9: var_14}
    var_48 = {var_9: var_46, var_42: var_47, var_43: var_4, var_44: var_40}
    var_49 = module_1.to_json_schema(var_41)
    var_50 = 'name'
    var_51 = module_0.String()
    var_52 = {var_50: var_51}
    var_53 = [var_50]
    var_54 = module_0.Object(properties=var_52, min_properties=var_4, max_properties=var_40, required=var_53)
    var_55 = 'properties'
    var_56 = 'required'
    var_57 = 'minProperties'
    var_58 = 'maxProperties'
    var_59 = 'object'
    var_60 = {var_9: var_14}
    var_61 = {var_50: var_60}
    var_62 = [var_50]
    var_63 = {var_9: var_59, var_55: var_61, var_56: var_62, var_57: var_4, var_58: var_40}
    var_64 = module_1.to_json_schema(var_54)
    var_65 = 'a'
    var_66 = (var_65, var_65)
    var_67 = 'b'
    var_68 = (var_67, var_67)
    var_69 = [var_66, var_68]
    var_70 = module_0.Choice(choices=var_69)
    var_71 = 'enum'
    var_72 = [var_65, var_67]
    var_73 = {var_71: var_72}
    var_74 = module_1.to_json_schema(var_70)
    var_75 = 'fixed_value'
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
    var_90 = module_0.String()
    var_91 = 'test'
    var_92 = module_0.Const(var_91)
    var_93 = [var_90, var_92]
    var_94 = module_2.AllOf(var_93)
    var_95 = 'allOf'
    var_96 = {var_9: var_14}
    var_97 = {var_77: var_91}
    var_98 = [var_96, var_97]
    var_99 = {var_95: var_98}
    var_100 = module_1.to_json_schema(var_94)
    var_101 = module_0.String()
    var_102 = module_0.Integer()
    var_103 = [var_101, var_102]
    var_104 = module_2.OneOf(var_103)
    var_105 = 'oneOf'
    var_106 = {var_9: var_14}
    var_107 = {var_9: var_25}
    var_108 = [var_106, var_107]
    var_109 = {var_105: var_108}
    var_110 = module_1.to_json_schema(var_104)
    var_111 = module_0.String()
    var_112 = module_2.Not(var_111)
    var_113 = 'not'
    var_114 = {var_9: var_14}
    var_115 = {var_113: var_114}
    var_116 = module_1.to_json_schema(var_112)
    var_117 = module_0.String()
    var_118 = module_0.Integer()
    var_119 = module_0.Boolean()
    var_120 = module_2.IfThenElse(var_117, var_118, var_119)
    var_121 = 'if'
    var_122 = 'then'
    var_123 = 'else'
    var_124 = {var_9: var_14}
    var_125 = {var_9: var_25}
    var_126 = {var_9: var_36}
    var_127 = {var_121: var_124, var_122: var_125, var_123: var_126}
    var_128 = module_1.to_json_schema(var_120)
    var_129 = 'Test'
    var_130 = module_0.String()
    var_131 = module_3.Reference(var_129)
    var_132 = '$ref'
    var_133 = 'components'
    var_134 = '#/components/schemas/Test'
    var_135 = 'schemas'
    var_136 = {var_9: var_14}
    var_137 = {var_129: var_136}
    var_138 = {var_135: var_137}
    var_139 = {var_132: var_134, var_133: var_138}
    var_140 = module_1.to_json_schema(var_131)
    var_141 = module_0.String()
    var_142 = {var_50: var_141}
    var_143 = [var_50]
    var_144 = module_3.Schema(var_142)
    var_145 = {var_9: var_14}
    var_146 = {var_50: var_145}
    var_147 = [var_50]
    var_148 = {var_9: var_59, var_55: var_146, var_56: var_147}
    var_149 = module_1.to_json_schema(var_144)
    var_150 = 'StringField'
    var_151 = 'IntField'
    var_152 = module_0.String()
    var_153 = module_0.Integer()
    var_154 = {var_150: var_152, var_151: var_153}
    var_155 = {var_9: var_14}
    var_156 = {var_9: var_25}
    var_157 = {var_150: var_155, var_151: var_156}
    var_158 = {var_135: var_157}
    var_159 = {var_133: var_158}



# Parsed testcases at query #26
#--------------------------


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
    var_12 = 'integer'
    var_13 = {var_0: var_12, var_1: var_9, var_2: var_6, var_3: var_7}
    var_14 = False
    var_15 = module_0.Definitions()
    var_16 = module_1.from_json_schema_type(var_13, var_12, var_14, var_15)
    var_17 = 'minLength'
    var_18 = 'maxLength'
    var_19 = 'pattern'
    var_20 = 'string'
    var_21 = 5
    var_22 = 10
    var_23 = '^[A-Za-z]+$'
    var_24 = 'test'
    var_25 = {var_0: var_20, var_17: var_21, var_18: var_22, var_19: var_23, var_3: var_24}
    var_26 = False
    var_27 = module_0.Definitions()
    var_28 = module_1.from_json_schema_type(var_25, var_20, var_26, var_27)
    var_29 = 'boolean'
    var_30 = True
    var_31 = {var_0: var_29, var_3: var_30}
    var_32 = False
    var_33 = module_0.Definitions()
    var_34 = module_1.from_json_schema_type(var_31, var_29, var_32, var_33)
    var_35 = 'items'
    var_36 = 'minItems'
    var_37 = 'maxItems'
    var_38 = 'uniqueItems'
    var_39 = 'array'
    var_40 = {var_0: var_20}
    var_41 = [var_24]
    var_42 = {var_0: var_39, var_35: var_40, var_36: var_30, var_37: var_22, var_38: var_30, var_3: var_41}
    var_43 = False
    var_44 = module_0.Definitions()
    var_45 = module_1.from_json_schema_type(var_42, var_39, var_43, var_44)
    var_46 = var_45.items
    var_47 = 'properties'
    var_48 = 'required'
    var_49 = 'object'
    var_50 = 'name'
    var_51 = 'age'
    var_52 = {var_0: var_20}
    var_53 = {var_0: var_12}
    var_54 = {var_50: var_52, var_51: var_53}
    var_55 = [var_50]
    var_56 = 25
    var_57 = {var_50: var_24, var_51: var_56}
    var_58 = {var_0: var_49, var_47: var_54, var_48: var_55, var_3: var_57}
    var_59 = False
    var_60 = module_0.Definitions()
    var_61 = module_1.from_json_schema_type(var_58, var_49, var_59, var_60)
    var_62 = var_61.properties[var_50]
    var_63 = var_61.properties[var_51]
    var_64 = {var_0: var_20}
    var_65 = module_0.Definitions()
    var_66 = module_1.from_json_schema_type(var_64, var_20, var_30, var_65)



# Parsed testcases at query #27
#--------------------------


import typesystem.json_schema as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)
    var_2 = False
    var_3 = module_0.from_json_schema(var_2)
    var_4 = 'components'
    var_5 = 'schemas'
    var_6 = 'test_schema'
    var_7 = 'type'
    var_8 = 'string'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = {var_5: var_10}
    var_12 = {var_4: var_11}
    var_13 = module_1.Definitions()
    var_14 = module_0.from_json_schema(var_12, var_13)
    var_15 = '$ref'
    var_16 = '#/components/schemas/test_schema'
    var_17 = {var_15: var_16}
    var_18 = module_0.from_json_schema(var_17, var_13)
    var_19 = 'minLength'
    var_20 = {var_7: var_8, var_19: var_0}
    var_21 = module_0.from_json_schema(var_20)
    var_22 = 'enum'
    var_23 = 'a'
    var_24 = 'b'
    var_25 = 'c'
    var_26 = [var_23, var_24, var_25]
    var_27 = {var_22: var_26}
    var_28 = module_0.from_json_schema(var_27)
    var_29 = 'const'
    var_30 = 'test'
    var_31 = {var_29: var_30}
    var_32 = module_0.from_json_schema(var_31)
    var_33 = 'allOf'
    var_34 = {var_7: var_8}
    var_35 = {var_19: var_0}
    var_36 = [var_34, var_35]
    var_37 = {var_33: var_36}
    var_38 = module_0.from_json_schema(var_37)
    var_39 = 'anyOf'
    var_40 = {var_7: var_8}
    var_41 = 'number'
    var_42 = {var_7: var_41}
    var_43 = [var_40, var_42]
    var_44 = {var_39: var_43}
    var_45 = module_0.from_json_schema(var_44)
    var_46 = 'oneOf'
    var_47 = {var_7: var_8}
    var_48 = {var_7: var_41}
    var_49 = [var_47, var_48]
    var_50 = {var_46: var_49}
    var_51 = module_0.from_json_schema(var_50)
    var_52 = 'not'
    var_53 = {var_7: var_8}
    var_54 = {var_52: var_53}
    var_55 = module_0.from_json_schema(var_54)
    var_56 = 'if'
    var_57 = 'then'
    var_58 = 'else'
    var_59 = {var_7: var_8}
    var_60 = {var_19: var_0}
    var_61 = {var_19: var_2}
    var_62 = {var_56: var_59, var_57: var_60, var_58: var_61}
    var_63 = module_0.from_json_schema(var_62)
    var_64 = [var_23, var_24, var_25]
    var_65 = {var_7: var_8, var_19: var_0, var_22: var_64}
    var_66 = module_0.from_json_schema(var_65)
    var_67 = {}
    var_68 = module_0.from_json_schema(var_67)



# Parsed testcases at query #28
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
    var_8 = 0
    var_9 = 100
    var_10 = 2
    var_11 = 50
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
    var_26 = 5
    var_27 = 'email'
    var_28 = '^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$'
    var_29 = 'test@example.com'
    var_30 = {var_0: var_25, var_21: var_26, var_22: var_9, var_23: var_27, var_24: var_28, var_6: var_29}
    var_31 = False
    var_32 = module_0.Definitions()
    var_33 = module_1.from_json_schema_type(var_30, var_25, var_31, var_32)
    var_34 = 'boolean'
    var_35 = True
    var_36 = {var_0: var_34, var_6: var_35}
    var_37 = False
    var_38 = module_0.Definitions()
    var_39 = module_1.from_json_schema_type(var_36, var_34, var_37, var_38)
    var_40 = 'items'
    var_41 = 'additionalItems'
    var_42 = 'minItems'
    var_43 = 'maxItems'
    var_44 = 'uniqueItems'
    var_45 = 'array'
    var_46 = {var_0: var_25}
    var_47 = False
    var_48 = 10
    var_49 = 'item1'
    var_50 = 'item2'
    var_51 = [var_49, var_50]
    var_52 = {var_0: var_45, var_40: var_46, var_41: var_47, var_42: var_35, var_43: var_48, var_44: var_35, var_6: var_51}
    var_53 = False
    var_54 = module_0.Definitions()
    var_55 = module_1.from_json_schema_type(var_52, var_45, var_53, var_54)
    var_56 = var_55.items
    var_57 = 'properties'
    var_58 = 'patternProperties'
    var_59 = 'additionalProperties'
    var_60 = 'propertyNames'
    var_61 = 'minProperties'
    var_62 = 'maxProperties'
    var_63 = 'required'
    var_64 = 'object'
    var_65 = 'name'
    var_66 = 'age'
    var_67 = {var_0: var_25}
    var_68 = {var_0: var_16}
    var_69 = {var_65: var_67, var_66: var_68}
    var_70 = '^S_'
    var_71 = '^I_'
    var_72 = {var_0: var_25}
    var_73 = {var_0: var_16}
    var_74 = {var_70: var_72, var_71: var_73}
    var_75 = False
    var_76 = {var_0: var_25}
    var_77 = [var_65]
    var_78 = 'John'
    var_79 = 30
    var_80 = {var_65: var_78, var_66: var_79}
    var_81 = {var_0: var_64, var_57: var_69, var_58: var_74, var_59: var_75, var_60: var_76, var_61: var_35, var_62: var_48, var_63: var_77, var_6: var_80}
    var_82 = False
    var_83 = module_0.Definitions()
    var_84 = module_1.from_json_schema_type(var_81, var_64, var_82, var_83)
    var_85 = var_84.properties[var_65]
    var_86 = var_84.properties[var_66]
    var_87 = var_84.pattern_properties[var_70]
    var_88 = var_84.pattern_properties[var_71]
    var_89 = var_84.property_names
    var_90 = 'type'
    var_91 = 'invalid'
    var_92 = {var_90: var_91}
    var_93 = False
    var_94 = module_0.Definitions()
    var_95 = module_1.from_json_schema_type(var_92, var_91, var_93, var_94)



# Parsed testcases at query #29
#--------------------------


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
    var_12 = 'integer'
    var_13 = {var_0: var_12, var_1: var_9, var_2: var_6, var_3: var_7}
    var_14 = False
    var_15 = module_0.Definitions()
    var_16 = module_1.from_json_schema_type(var_13, var_12, var_14, var_15)
    var_17 = 'minLength'
    var_18 = 'maxLength'
    var_19 = 'pattern'
    var_20 = 'string'
    var_21 = 5
    var_22 = 10
    var_23 = '^[A-Za-z]+$'
    var_24 = 'hello'
    var_25 = {var_0: var_20, var_17: var_21, var_18: var_22, var_19: var_23, var_3: var_24}
    var_26 = False
    var_27 = module_0.Definitions()
    var_28 = module_1.from_json_schema_type(var_25, var_20, var_26, var_27)
    var_29 = 'boolean'
    var_30 = True
    var_31 = {var_0: var_29, var_3: var_30}
    var_32 = False
    var_33 = module_0.Definitions()
    var_34 = module_1.from_json_schema_type(var_31, var_29, var_32, var_33)
    var_35 = 'items'
    var_36 = 'minItems'
    var_37 = 'maxItems'
    var_38 = 'uniqueItems'
    var_39 = 'array'
    var_40 = {var_0: var_20}
    var_41 = [var_24]
    var_42 = {var_0: var_39, var_35: var_40, var_36: var_30, var_37: var_21, var_38: var_30, var_3: var_41}
    var_43 = False
    var_44 = module_0.Definitions()
    var_45 = module_1.from_json_schema_type(var_42, var_39, var_43, var_44)
    var_46 = var_45.items
    var_47 = 'properties'
    var_48 = 'required'
    var_49 = 'object'
    var_50 = 'name'
    var_51 = 'age'
    var_52 = {var_0: var_20}
    var_53 = {var_0: var_12}
    var_54 = {var_50: var_52, var_51: var_53}
    var_55 = [var_50]
    var_56 = 'John'
    var_57 = 30
    var_58 = {var_50: var_56, var_51: var_57}
    var_59 = {var_0: var_49, var_47: var_54, var_48: var_55, var_3: var_58}
    var_60 = False
    var_61 = module_0.Definitions()
    var_62 = module_1.from_json_schema_type(var_59, var_49, var_60, var_61)
    var_63 = var_62.properties[var_50]
    var_64 = var_62.properties[var_51]



# Parsed testcases at query #30
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
    var_4 = 1
    var_5 = 10
    var_6 = '[a-z]+'
    var_7 = 'email'
    var_8 = module_0.String(max_length=var_5, min_length=var_4, pattern=var_6, format=var_7)
    var_9 = 'type'
    var_10 = 'minLength'
    var_11 = 'maxLength'
    var_12 = 'pattern'
    var_13 = 'format'
    var_14 = 'string'
    var_15 = {var_9: var_14, var_10: var_4, var_11: var_5, var_12: var_6, var_13: var_7}
    var_16 = module_1.to_json_schema(var_8)
    var_17 = 0
    var_18 = 100
    var_19 = True
    var_20 = module_0.Integer(minimum=var_17, maximum=var_18, exclusive_minimum=var_19)
    var_21 = 'minimum'
    var_22 = 'maximum'
    var_23 = 'exclusiveMinimum'
    var_24 = 'integer'
    var_25 = True
    var_26 = {var_9: var_24, var_21: var_17, var_22: var_18, var_23: var_25}
    var_27 = module_1.to_json_schema(var_20)
    var_28 = 0.5
    var_29 = module_0.Float(multiple_of=var_28)
    var_30 = 'multipleOf'
    var_31 = 'number'
    var_32 = {var_9: var_31, var_30: var_28}
    var_33 = module_1.to_json_schema(var_29)
    var_34 = module_0.Boolean()
    var_35 = 'boolean'
    var_36 = {var_9: var_35}
    var_37 = module_1.to_json_schema(var_34)
    var_38 = module_0.String()
    var_39 = 5
    var_40 = True
    var_41 = module_0.Array(var_38, min_items=var_25, max_items=var_39, unique_items=var_40)
    var_42 = 'items'
    var_43 = 'minItems'
    var_44 = 'maxItems'
    var_45 = 'uniqueItems'
    var_46 = 'array'
    var_47 = {var_9: var_14}
    var_48 = True
    var_49 = {var_9: var_46, var_42: var_47, var_43: var_40, var_44: var_39, var_45: var_48}
    var_50 = module_1.to_json_schema(var_41)
    var_51 = 'name'
    var_52 = module_0.String()
    var_53 = {var_51: var_52}
    var_54 = [var_51]
    var_55 = 3
    var_56 = module_0.Object(properties=var_53, min_properties=var_48, max_properties=var_55, required=var_54)
    var_57 = 'properties'
    var_58 = 'required'
    var_59 = 'minProperties'
    var_60 = 'maxProperties'
    var_61 = 'object'
    var_62 = {var_9: var_14}
    var_63 = {var_51: var_62}
    var_64 = [var_51]
    var_65 = {var_9: var_61, var_57: var_63, var_58: var_64, var_59: var_48, var_60: var_55}
    var_66 = module_1.to_json_schema(var_56)
    var_67 = 'a'
    var_68 = (var_67, var_67)
    var_69 = 'b'
    var_70 = (var_69, var_69)
    var_71 = [var_68, var_70]
    var_72 = module_0.Choice(choices=var_71)
    var_73 = 'enum'
    var_74 = [var_67, var_69]
    var_75 = {var_73: var_74}
    var_76 = module_1.to_json_schema(var_72)
    var_77 = 'fixed'
    var_78 = module_0.Const(var_77)
    var_79 = 'const'
    var_80 = {var_79: var_77}
    var_81 = module_1.to_json_schema(var_78)
    var_82 = module_0.String()
    var_83 = module_0.Integer()
    var_84 = [var_82, var_83]
    var_85 = module_0.Union(var_84)
    var_86 = 'anyOf'
    var_87 = {var_9: var_14}
    var_88 = {var_9: var_24}
    var_89 = [var_87, var_88]
    var_90 = {var_86: var_89}
    var_91 = module_1.to_json_schema(var_85)
    var_92 = module_0.String()
    var_93 = module_0.Integer()
    var_94 = [var_92, var_93]
    var_95 = module_2.AllOf(var_94)
    var_96 = 'allOf'
    var_97 = {var_9: var_14}
    var_98 = {var_9: var_24}
    var_99 = [var_97, var_98]
    var_100 = {var_96: var_99}
    var_101 = module_1.to_json_schema(var_95)
    var_102 = module_0.String()
    var_103 = module_0.Integer()
    var_104 = [var_102, var_103]
    var_105 = module_2.OneOf(var_104)
    var_106 = 'oneOf'
    var_107 = {var_9: var_14}
    var_108 = {var_9: var_24}
    var_109 = [var_107, var_108]
    var_110 = {var_106: var_109}
    var_111 = module_1.to_json_schema(var_105)
    var_112 = module_0.String()
    var_113 = module_2.Not(var_112)
    var_114 = 'not'
    var_115 = {var_9: var_14}
    var_116 = {var_114: var_115}
    var_117 = module_1.to_json_schema(var_113)
    var_118 = module_0.String()
    var_119 = module_0.Integer()
    var_120 = module_0.Boolean()
    var_121 = module_2.IfThenElse(var_118, var_119, var_120)
    var_122 = 'if'
    var_123 = 'then'
    var_124 = 'else'
    var_125 = {var_9: var_14}
    var_126 = {var_9: var_24}
    var_127 = {var_9: var_35}
    var_128 = {var_122: var_125, var_123: var_126, var_124: var_127}
    var_129 = module_1.to_json_schema(var_121)
    var_130 = 'Test'
    var_131 = module_0.String()
    var_132 = module_3.Reference(var_130)
    var_133 = '$ref'
    var_134 = 'components'
    var_135 = '#/components/schemas/Test'
    var_136 = 'schemas'
    var_137 = {var_9: var_14}
    var_138 = {var_130: var_137}
    var_139 = {var_136: var_138}
    var_140 = {var_133: var_135, var_134: var_139}
    var_141 = module_1.to_json_schema(var_132)
    var_142 = module_0.String()
    var_143 = {var_51: var_142}
    var_144 = [var_51]
    var_145 = module_3.Schema(var_143)
    var_146 = {var_9: var_14}
    var_147 = {var_51: var_146}
    var_148 = [var_51]
    var_149 = {var_9: var_61, var_57: var_147, var_58: var_148}
    var_150 = module_1.to_json_schema(var_145)
    var_151 = module_0.String()
    var_152 = {var_130: var_151}
    var_153 = {var_9: var_14}
    var_154 = {var_130: var_153}
    var_155 = {var_136: var_154}
    var_156 = {var_134: var_155}



####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
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



# Parsed testcases at query #2
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
    var_4 = 1
    var_5 = 10
    var_6 = '[a-z]+'
    var_7 = 'email'
    var_8 = module_0.String(max_length=var_5, min_length=var_4, pattern=var_6, format=var_7)
    var_9 = 'type'
    var_10 = 'minLength'
    var_11 = 'maxLength'
    var_12 = 'pattern'
    var_13 = 'format'
    var_14 = 'string'
    var_15 = {var_9: var_14, var_10: var_4, var_11: var_5, var_12: var_6, var_13: var_7}
    var_16 = module_1.to_json_schema(var_8)
    var_17 = 0
    var_18 = 100
    var_19 = True
    var_20 = True
    var_21 = module_0.Integer(minimum=var_17, maximum=var_18, exclusive_minimum=var_19, exclusive_maximum=var_20)
    var_22 = 'minimum'
    var_23 = 'maximum'
    var_24 = 'exclusiveMinimum'
    var_25 = 'exclusiveMaximum'
    var_26 = 'integer'
    var_27 = True
    var_28 = True
    var_29 = {var_9: var_26, var_22: var_17, var_23: var_18, var_24: var_27, var_25: var_28}
    var_30 = module_1.to_json_schema(var_21)
    var_31 = 0.1
    var_32 = module_0.Float(minimum=var_17, maximum=var_28, multiple_of=var_31)
    var_33 = 'multipleOf'
    var_34 = 'number'
    var_35 = {var_9: var_34, var_22: var_17, var_23: var_28, var_33: var_31}
    var_36 = module_1.to_json_schema(var_32)
    var_37 = module_0.Boolean()
    var_38 = 'boolean'
    var_39 = {var_9: var_38}
    var_40 = module_1.to_json_schema(var_37)
    var_41 = module_0.String()
    var_42 = True
    var_43 = module_0.Array(var_41, min_items=var_28, max_items=var_5, unique_items=var_42)
    var_44 = 'items'
    var_45 = 'minItems'
    var_46 = 'maxItems'
    var_47 = 'uniqueItems'
    var_48 = 'array'
    var_49 = {var_9: var_14}
    var_50 = True
    var_51 = {var_9: var_48, var_44: var_49, var_45: var_42, var_46: var_5, var_47: var_50}
    var_52 = module_1.to_json_schema(var_43)
    var_53 = 'name'
    var_54 = module_0.String()
    var_55 = {var_53: var_54}
    var_56 = [var_53]
    var_57 = module_0.Object(properties=var_55, min_properties=var_50, max_properties=var_5, required=var_56)
    var_58 = 'properties'
    var_59 = 'required'
    var_60 = 'minProperties'
    var_61 = 'maxProperties'
    var_62 = 'object'
    var_63 = {var_9: var_14}
    var_64 = {var_53: var_63}
    var_65 = [var_53]
    var_66 = {var_9: var_62, var_58: var_64, var_59: var_65, var_60: var_50, var_61: var_5}
    var_67 = module_1.to_json_schema(var_57)
    var_68 = 'a'
    var_69 = (var_68, var_68)
    var_70 = 'b'
    var_71 = (var_70, var_70)
    var_72 = [var_69, var_71]
    var_73 = module_0.Choice(choices=var_72)
    var_74 = 'enum'
    var_75 = [var_68, var_70]
    var_76 = {var_74: var_75}
    var_77 = module_1.to_json_schema(var_73)
    var_78 = 'fixed_value'
    var_79 = module_0.Const(var_78)
    var_80 = 'const'
    var_81 = {var_80: var_78}
    var_82 = module_1.to_json_schema(var_79)
    var_83 = module_0.String()
    var_84 = module_0.Integer()
    var_85 = [var_83, var_84]
    var_86 = module_0.Union(var_85)
    var_87 = 'anyOf'
    var_88 = {var_9: var_14}
    var_89 = {var_9: var_26}
    var_90 = [var_88, var_89]
    var_91 = {var_87: var_90}
    var_92 = module_1.to_json_schema(var_86)
    var_93 = module_0.String()
    var_94 = module_0.Integer()
    var_95 = [var_93, var_94]
    var_96 = module_2.OneOf(var_95)
    var_97 = 'oneOf'
    var_98 = {var_9: var_14}
    var_99 = {var_9: var_26}
    var_100 = [var_98, var_99]
    var_101 = {var_97: var_100}
    var_102 = module_1.to_json_schema(var_96)
    var_103 = module_0.String()
    var_104 = 'test'
    var_105 = module_0.Const(var_104)
    var_106 = [var_103, var_105]
    var_107 = module_2.AllOf(var_106)
    var_108 = 'allOf'
    var_109 = {var_9: var_14}
    var_110 = {var_80: var_104}
    var_111 = [var_109, var_110]
    var_112 = {var_108: var_111}
    var_113 = module_1.to_json_schema(var_107)
    var_114 = module_0.String()
    var_115 = module_0.Integer()
    var_116 = module_0.Boolean()
    var_117 = module_2.IfThenElse(var_114, var_115, var_116)
    var_118 = 'if'
    var_119 = 'then'
    var_120 = 'else'
    var_121 = {var_9: var_14}
    var_122 = {var_9: var_26}
    var_123 = {var_9: var_38}
    var_124 = {var_118: var_121, var_119: var_122, var_120: var_123}
    var_125 = module_1.to_json_schema(var_117)
    var_126 = module_0.String()
    var_127 = module_2.Not(var_126)
    var_128 = 'not'
    var_129 = {var_9: var_14}
    var_130 = {var_128: var_129}
    var_131 = module_1.to_json_schema(var_127)
    var_132 = module_3.Definitions()
    var_133 = module_3.Reference(var_104, var_132)
    var_134 = '$ref'
    var_135 = '#/components/schemas/test'
    var_136 = {var_134: var_135}
    var_137 = module_1.to_json_schema(var_133)
    var_138 = module_0.String()
    var_139 = {var_53: var_138}
    var_140 = [var_53]
    var_141 = module_3.Schema(var_139)
    var_142 = {var_9: var_14}
    var_143 = {var_53: var_142}
    var_144 = [var_53]
    var_145 = {var_9: var_62, var_58: var_143, var_59: var_144}
    var_146 = module_1.to_json_schema(var_141)
    var_147 = module_3.Definitions()
    var_148 = 'components'
    var_149 = 'schemas'
    var_150 = 'string_field'
    var_151 = 'int_field'
    var_152 = {var_9: var_14}
    var_153 = {var_9: var_26}
    var_154 = {var_150: var_152, var_151: var_153}
    var_155 = {var_149: var_154}
    var_156 = {var_148: var_155}
    var_157 = module_1.to_json_schema(var_147)



# Parsed testcases at query #3
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
    var_4 = 1
    var_5 = 10
    var_6 = '[a-z]+'
    var_7 = 'email'
    var_8 = module_0.String(max_length=var_5, min_length=var_4, pattern=var_6, format=var_7)
    var_9 = 'type'
    var_10 = 'minLength'
    var_11 = 'maxLength'
    var_12 = 'pattern'
    var_13 = 'format'
    var_14 = 'string'
    var_15 = {var_9: var_14, var_10: var_4, var_11: var_5, var_12: var_6, var_13: var_7}
    var_16 = module_1.to_json_schema(var_8)
    var_17 = 0
    var_18 = 100
    var_19 = module_0.Integer(minimum=var_17, maximum=var_18)
    var_20 = 'minimum'
    var_21 = 'maximum'
    var_22 = 'integer'
    var_23 = {var_9: var_22, var_20: var_17, var_21: var_18}
    var_24 = module_1.to_json_schema(var_19)
    var_25 = 0.5
    var_26 = module_0.Float(multiple_of=var_25)
    var_27 = 'multipleOf'
    var_28 = 'number'
    var_29 = {var_9: var_28, var_27: var_25}
    var_30 = module_1.to_json_schema(var_26)
    var_31 = module_0.Boolean()
    var_32 = 'boolean'
    var_33 = {var_9: var_32}
    var_34 = module_1.to_json_schema(var_31)
    var_35 = module_0.String()
    var_36 = 5
    var_37 = module_0.Array(var_35, min_items=var_4, max_items=var_36)
    var_38 = 'minItems'
    var_39 = 'maxItems'
    var_40 = 'items'
    var_41 = 'array'
    var_42 = {var_9: var_14}
    var_43 = {var_9: var_41, var_38: var_4, var_39: var_36, var_40: var_42}
    var_44 = module_1.to_json_schema(var_37)
    var_45 = 'name'
    var_46 = module_0.String()
    var_47 = {var_45: var_46}
    var_48 = [var_45]
    var_49 = module_0.Object(properties=var_47, required=var_48)
    var_50 = 'properties'
    var_51 = 'required'
    var_52 = 'object'
    var_53 = {var_9: var_14}
    var_54 = {var_45: var_53}
    var_55 = [var_45]
    var_56 = {var_9: var_52, var_50: var_54, var_51: var_55}
    var_57 = module_1.to_json_schema(var_49)
    var_58 = 'a'
    var_59 = (var_58, var_58)
    var_60 = 'b'
    var_61 = (var_60, var_60)
    var_62 = [var_59, var_61]
    var_63 = module_0.Choice(choices=var_62)
    var_64 = 'enum'
    var_65 = [var_58, var_60]
    var_66 = {var_64: var_65}
    var_67 = module_1.to_json_schema(var_63)
    var_68 = 'fixed'
    var_69 = module_0.Const(var_68)
    var_70 = 'const'
    var_71 = {var_70: var_68}
    var_72 = module_1.to_json_schema(var_69)
    var_73 = module_0.String()
    var_74 = module_0.Integer()
    var_75 = [var_73, var_74]
    var_76 = module_0.Union(var_75)
    var_77 = 'anyOf'
    var_78 = {var_9: var_14}
    var_79 = {var_9: var_22}
    var_80 = [var_78, var_79]
    var_81 = {var_77: var_80}
    var_82 = module_1.to_json_schema(var_76)
    var_83 = module_0.String()
    var_84 = 'test'
    var_85 = module_0.Const(var_84)
    var_86 = [var_83, var_85]
    var_87 = module_2.AllOf(var_86)
    var_88 = 'allOf'
    var_89 = {var_9: var_14}
    var_90 = {var_70: var_84}
    var_91 = [var_89, var_90]
    var_92 = {var_88: var_91}
    var_93 = module_1.to_json_schema(var_87)
    var_94 = 'Test'
    var_95 = module_0.String()
    var_96 = {var_94: var_95}
    var_97 = '$ref'
    var_98 = 'components'
    var_99 = '#/components/schemas/Test'
    var_100 = 'schemas'
    var_101 = {var_9: var_14}
    var_102 = {var_94: var_101}
    var_103 = {var_100: var_102}
    var_104 = {var_97: var_99, var_98: var_103}
    var_105 = module_0.String()
    var_106 = {var_45: var_105}
    var_107 = module_3.Schema(var_106)
    var_108 = {var_9: var_14}
    var_109 = {var_45: var_108}
    var_110 = {var_9: var_52, var_50: var_109}
    var_111 = module_1.to_json_schema(var_107)
    var_112 = module_0.String()
    var_113 = module_0.Integer()
    var_114 = module_0.Boolean()
    var_115 = module_2.IfThenElse(var_112, var_113, var_114)
    var_116 = 'if'
    var_117 = 'then'
    var_118 = 'else'
    var_119 = {var_9: var_14}
    var_120 = {var_9: var_22}
    var_121 = {var_9: var_32}
    var_122 = {var_116: var_119, var_117: var_120, var_118: var_121}
    var_123 = module_1.to_json_schema(var_115)
    var_124 = module_0.String()
    var_125 = module_2.Not(var_124)
    var_126 = 'not'
    var_127 = {var_9: var_14}
    var_128 = {var_126: var_127}
    var_129 = module_1.to_json_schema(var_125)



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
    var_35 = 'minLength'
    var_36 = 5
    var_37 = {var_4: var_5, var_35: var_36}
    var_38 = 'maxLength'
    var_39 = 10
    var_40 = {var_4: var_5, var_38: var_39}
    var_41 = [var_37, var_40]
    var_42 = {var_34: var_41}
    var_43 = module_0.from_json_schema(var_42)
    var_44 = var_43.schemas
    var_45 = len(var_44)
    assert var_45 == 2
    var_46 = 'anyOf'
    var_47 = {var_4: var_5}
    var_48 = {var_4: var_8}
    var_49 = [var_47, var_48]
    var_50 = {var_46: var_49}
    var_51 = module_0.from_json_schema(var_50)
    var_52 = var_51.schemas
    var_53 = len(var_52)
    assert var_53 == 2
    var_54 = 'oneOf'
    var_55 = {var_4: var_5}
    var_56 = {var_4: var_8}
    var_57 = [var_55, var_56]
    var_58 = {var_54: var_57}
    var_59 = module_0.from_json_schema(var_58)
    var_60 = var_59.schemas
    var_61 = len(var_60)
    assert var_61 == 2
    var_62 = 'not'
    var_63 = {var_4: var_5}
    var_64 = {var_62: var_63}
    var_65 = module_0.from_json_schema(var_64)
    var_66 = var_65.schema
    var_67 = 'if'
    var_68 = 'then'
    var_69 = 'else'
    var_70 = {var_4: var_5}
    var_71 = {var_35: var_36}
    var_72 = {var_4: var_8}
    var_73 = {var_67: var_70, var_68: var_71, var_69: var_72}
    var_74 = module_0.from_json_schema(var_73)
    var_75 = var_74.if_schema
    var_76 = var_74.then_schema
    var_77 = var_74.else_schema
    var_78 = module_1.Definitions()
    var_79 = '$ref'
    var_80 = '#/components/schemas/Test'
    var_81 = {var_79: var_80}
    var_82 = module_0.from_json_schema(var_81, var_78)
    var_83 = 'pattern'
    var_84 = '^[a-z]+$'
    var_85 = {var_4: var_5, var_35: var_36, var_38: var_39, var_83: var_84}
    var_86 = module_0.from_json_schema(var_85)
    var_87 = var_86.schemas
    var_88 = len(var_87)
    assert var_88 == 4
    var_89 = {}
    var_90 = module_0.from_json_schema(var_89)



# Parsed testcases at query #5
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
    var_16 = 'properties'
    var_17 = 'object'
    var_18 = 'name'
    var_19 = {var_2: var_3}
    var_20 = {var_18: var_19}
    var_21 = {var_2: var_17, var_16: var_20}
    var_22 = 'items'
    var_23 = 'array'
    var_24 = 'integer'
    var_25 = {var_2: var_24}
    var_26 = {var_2: var_23, var_22: var_25}
    var_27 = [var_21, var_26]
    var_28 = {var_1: var_27}
    var_29 = module_1.one_of_from_json_schema(var_28, var_0)
    var_30 = var_29.one_of
    var_31 = len(var_30)
    assert var_31 == 2
    var_32 = var_29.one_of[var_12]
    var_33 = var_29.one_of[var_14]
    var_34 = 'default'
    var_35 = 'boolean'
    var_36 = {var_2: var_35}
    var_37 = 'null'
    var_38 = {var_2: var_37}
    var_39 = [var_36, var_38]
    var_40 = True
    var_41 = {var_1: var_39, var_34: var_40}
    var_42 = module_1.one_of_from_json_schema(var_41, var_0)
    var_43 = '$ref'
    var_44 = '#/components/schemas/Test'
    var_45 = {var_43: var_44}
    var_46 = {var_2: var_24}
    var_47 = [var_45, var_46]
    var_48 = {var_1: var_47}
    var_49 = module_1.one_of_from_json_schema(var_48, var_0)
    var_50 = var_49.one_of[var_12]
    var_51 = var_49.one_of[var_40]



# Parsed testcases at query #6
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
    var_7 = 'minLength'
    var_8 = 5
    var_9 = {var_4: var_5, var_7: var_8}
    var_10 = 'number'
    var_11 = {var_4: var_10}
    var_12 = {var_1: var_6, var_2: var_9, var_3: var_11}
    var_13 = module_1.if_then_else_from_json_schema(var_12, var_0)
    var_14 = var_13.if_clause
    var_15 = var_13.then_clause
    var_16 = var_13.else_clause
    var_17 = {var_4: var_5}
    var_18 = {var_4: var_5, var_7: var_8}
    var_19 = {var_1: var_17, var_2: var_18}
    var_20 = module_1.if_then_else_from_json_schema(var_19, var_0)
    var_21 = var_20.if_clause
    var_22 = var_20.then_clause
    var_23 = {var_4: var_5}
    var_24 = {var_4: var_10}
    var_25 = {var_1: var_23, var_3: var_24}
    var_26 = module_1.if_then_else_from_json_schema(var_25, var_0)
    var_27 = var_26.if_clause
    var_28 = var_26.else_clause
    var_29 = 'default'
    var_30 = {var_4: var_5}
    var_31 = {var_4: var_5, var_7: var_8}
    var_32 = {var_4: var_10}
    var_33 = 'default_value'
    var_34 = {var_1: var_30, var_2: var_31, var_3: var_32, var_29: var_33}
    var_35 = module_1.if_then_else_from_json_schema(var_34, var_0)



# Parsed testcases at query #7
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = '#/components/schemas/TestSchema'
    var_3 = {var_1: var_2}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)
    var_5 = 'external/schema'
    var_6 = {var_1: var_5}
    var_7 = module_1.ref_from_json_schema(var_6, var_0)



# Parsed testcases at query #8
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
    var_4 = 1
    var_5 = 10
    var_6 = '[a-z]+'
    var_7 = 'email'
    var_8 = module_0.String(max_length=var_5, min_length=var_4, pattern=var_6, format=var_7)
    var_9 = 'type'
    var_10 = 'minLength'
    var_11 = 'maxLength'
    var_12 = 'pattern'
    var_13 = 'format'
    var_14 = 'string'
    var_15 = {var_9: var_14, var_10: var_4, var_11: var_5, var_12: var_6, var_13: var_7}
    var_16 = module_1.to_json_schema(var_8)
    var_17 = 0
    var_18 = 100
    var_19 = module_0.Integer(minimum=var_17, maximum=var_18)
    var_20 = 'minimum'
    var_21 = 'maximum'
    var_22 = 'integer'
    var_23 = {var_9: var_22, var_20: var_17, var_21: var_18}
    var_24 = module_1.to_json_schema(var_19)
    var_25 = 0.5
    var_26 = module_0.Float(multiple_of=var_25)
    var_27 = 'multipleOf'
    var_28 = 'number'
    var_29 = {var_9: var_28, var_27: var_25}
    var_30 = module_1.to_json_schema(var_26)
    var_31 = module_0.Boolean()
    var_32 = 'boolean'
    var_33 = {var_9: var_32}
    var_34 = module_1.to_json_schema(var_31)
    var_35 = module_0.String()
    var_36 = 5
    var_37 = module_0.Array(var_35, min_items=var_4, max_items=var_36)
    var_38 = 'minItems'
    var_39 = 'maxItems'
    var_40 = 'items'
    var_41 = 'array'
    var_42 = {var_9: var_14}
    var_43 = {var_9: var_41, var_38: var_4, var_39: var_36, var_40: var_42}
    var_44 = module_1.to_json_schema(var_37)
    var_45 = 'name'
    var_46 = 'age'
    var_47 = module_0.String()
    var_48 = module_0.Integer()
    var_49 = {var_45: var_47, var_46: var_48}
    var_50 = [var_45]
    var_51 = module_0.Object(properties=var_49, required=var_50)
    var_52 = 'properties'
    var_53 = 'required'
    var_54 = 'object'
    var_55 = {var_9: var_14}
    var_56 = {var_9: var_22}
    var_57 = {var_45: var_55, var_46: var_56}
    var_58 = [var_45]
    var_59 = {var_9: var_54, var_52: var_57, var_53: var_58}
    var_60 = module_1.to_json_schema(var_51)
    var_61 = 'a'
    var_62 = (var_61, var_61)
    var_63 = 'b'
    var_64 = (var_63, var_63)
    var_65 = [var_62, var_64]
    var_66 = module_0.Choice(choices=var_65)
    var_67 = 'enum'
    var_68 = [var_61, var_63]
    var_69 = {var_67: var_68}
    var_70 = module_1.to_json_schema(var_66)
    var_71 = 'fixed_value'
    var_72 = module_0.Const(var_71)
    var_73 = 'const'
    var_74 = {var_73: var_71}
    var_75 = module_1.to_json_schema(var_72)
    var_76 = module_0.String()
    var_77 = module_0.Integer()
    var_78 = [var_76, var_77]
    var_79 = module_0.Union(var_78)
    var_80 = 'anyOf'
    var_81 = {var_9: var_14}
    var_82 = {var_9: var_22}
    var_83 = [var_81, var_82]
    var_84 = {var_80: var_83}
    var_85 = module_1.to_json_schema(var_79)
    var_86 = module_0.String()
    var_87 = 'test'
    var_88 = module_0.Const(var_87)
    var_89 = [var_86, var_88]
    var_90 = module_2.AllOf(var_89)
    var_91 = 'allOf'
    var_92 = {var_9: var_14}
    var_93 = {var_73: var_87}
    var_94 = [var_92, var_93]
    var_95 = {var_91: var_94}
    var_96 = module_1.to_json_schema(var_90)
    var_97 = module_0.String()
    var_98 = module_0.Integer()
    var_99 = module_0.Boolean()
    var_100 = module_2.IfThenElse(var_97, var_98, var_99)
    var_101 = 'if'
    var_102 = 'then'
    var_103 = 'else'
    var_104 = {var_9: var_14}
    var_105 = {var_9: var_22}
    var_106 = {var_9: var_32}
    var_107 = {var_101: var_104, var_102: var_105, var_103: var_106}
    var_108 = module_1.to_json_schema(var_100)
    var_109 = module_0.String()
    var_110 = module_2.Not(var_109)
    var_111 = 'not'
    var_112 = {var_9: var_14}
    var_113 = {var_111: var_112}
    var_114 = module_1.to_json_schema(var_110)
    var_115 = module_0.String()
    var_116 = module_3.Reference(var_87)
    var_117 = '$ref'
    var_118 = 'components'
    var_119 = '#/components/schemas/test'
    var_120 = 'schemas'
    var_121 = {var_9: var_14}
    var_122 = {var_87: var_121}
    var_123 = {var_120: var_122}
    var_124 = {var_117: var_119, var_118: var_123}
    var_125 = module_1.to_json_schema(var_116)
    var_126 = module_0.String()
    var_127 = {var_45: var_126}
    var_128 = [var_45]
    var_129 = module_3.Schema(var_127)
    var_130 = {var_9: var_14}
    var_131 = {var_45: var_130}
    var_132 = [var_45]
    var_133 = {var_9: var_54, var_52: var_131, var_53: var_132}
    var_134 = module_1.to_json_schema(var_129)
    var_135 = 'field1'
    var_136 = 'field2'
    var_137 = module_0.String()
    var_138 = module_0.Integer()
    var_139 = {var_135: var_137, var_136: var_138}
    var_140 = {var_9: var_14}
    var_141 = {var_9: var_22}
    var_142 = {var_135: var_140, var_136: var_141}
    var_143 = {var_120: var_142}
    var_144 = {var_118: var_143}



# Parsed testcases at query #9
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'anyOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'number'
    var_5 = {var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = {var_0: var_6}
    var_8 = module_0.Definitions()
    var_9 = module_1.any_of_from_json_schema(var_7, var_8)
    var_10 = var_9.any_of
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = 0
    var_13 = var_9.any_of[var_12]
    var_14 = 1
    var_15 = var_9.any_of[var_14]
    var_16 = 'minLength'
    var_17 = {var_1: var_2, var_16: var_14}
    var_18 = 'minimum'
    var_19 = {var_1: var_4, var_18: var_12}
    var_20 = 'boolean'
    var_21 = {var_1: var_20}
    var_22 = [var_17, var_19, var_21]
    var_23 = {var_0: var_22}
    var_24 = module_0.Definitions()
    var_25 = module_1.any_of_from_json_schema(var_23, var_24)
    var_26 = var_25.any_of
    var_27 = len(var_26)
    assert var_27 == 3
    var_28 = var_25.any_of[var_12]
    var_29 = var_25.any_of[var_14]
    var_30 = 2
    var_31 = var_25.any_of[var_30]
    var_32 = 'default'
    var_33 = {var_1: var_2}
    var_34 = {var_1: var_4}
    var_35 = [var_33, var_34]
    var_36 = 'test'
    var_37 = {var_0: var_35, var_32: var_36}
    var_38 = module_0.Definitions()
    var_39 = module_1.any_of_from_json_schema(var_37, var_38)
    var_40 = module_0.Definitions()
    var_41 = '$ref'
    var_42 = '#/components/schemas/Test'
    var_43 = {var_41: var_42}
    var_44 = {var_1: var_4}
    var_45 = [var_43, var_44]
    var_46 = {var_0: var_45}
    var_47 = module_1.any_of_from_json_schema(var_46, var_40)
    var_48 = var_47.any_of
    var_49 = len(var_48)
    assert var_49 == 2
    var_50 = var_47.any_of[var_12]
    var_51 = var_47.any_of[var_14]



# Parsed testcases at query #10
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
    var_7 = 'minLength'
    var_8 = 5
    var_9 = {var_4: var_5, var_7: var_8}
    var_10 = 'number'
    var_11 = {var_4: var_10}
    var_12 = {var_1: var_6, var_2: var_9, var_3: var_11}
    var_13 = module_1.if_then_else_from_json_schema(var_12, var_0)
    var_14 = var_13.if_clause
    var_15 = var_13.then_clause
    var_16 = var_13.else_clause
    var_17 = {var_4: var_5}
    var_18 = {var_4: var_5, var_7: var_8}
    var_19 = {var_1: var_17, var_2: var_18}
    var_20 = module_1.if_then_else_from_json_schema(var_19, var_0)
    var_21 = var_20.if_clause
    var_22 = var_20.then_clause
    var_23 = {var_4: var_5}
    var_24 = {var_4: var_10}
    var_25 = {var_1: var_23, var_3: var_24}
    var_26 = module_1.if_then_else_from_json_schema(var_25, var_0)
    var_27 = var_26.if_clause
    var_28 = var_26.else_clause
    var_29 = {var_4: var_5}
    var_30 = {var_1: var_29}
    var_31 = module_1.if_then_else_from_json_schema(var_30, var_0)
    var_32 = var_31.if_clause
    var_33 = 'default'
    var_34 = {var_4: var_5}
    var_35 = {var_4: var_5, var_7: var_8}
    var_36 = {var_4: var_10}
    var_37 = 'default_value'
    var_38 = {var_1: var_34, var_2: var_35, var_3: var_36, var_33: var_37}
    var_39 = module_1.if_then_else_from_json_schema(var_38, var_0)



# Parsed testcases at query #11
#--------------------------


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
    var_12 = 'integer'
    var_13 = {var_0: var_12, var_1: var_9, var_2: var_6, var_3: var_7}
    var_14 = False
    var_15 = module_0.Definitions()
    var_16 = module_1.from_json_schema_type(var_13, var_12, var_14, var_15)
    var_17 = 'minLength'
    var_18 = 'maxLength'
    var_19 = 'pattern'
    var_20 = 'string'
    var_21 = 5
    var_22 = 10
    var_23 = '^[a-zA-Z]+$'
    var_24 = 'hello'
    var_25 = {var_0: var_20, var_17: var_21, var_18: var_22, var_19: var_23, var_3: var_24}
    var_26 = False
    var_27 = module_0.Definitions()
    var_28 = module_1.from_json_schema_type(var_25, var_20, var_26, var_27)
    var_29 = 'boolean'
    var_30 = True
    var_31 = {var_0: var_29, var_3: var_30}
    var_32 = False
    var_33 = module_0.Definitions()
    var_34 = module_1.from_json_schema_type(var_31, var_29, var_32, var_33)
    var_35 = 'items'
    var_36 = 'minItems'
    var_37 = 'maxItems'
    var_38 = 'uniqueItems'
    var_39 = 'array'
    var_40 = {var_0: var_20}
    var_41 = [var_24]
    var_42 = {var_0: var_39, var_35: var_40, var_36: var_30, var_37: var_21, var_38: var_30, var_3: var_41}
    var_43 = False
    var_44 = module_0.Definitions()
    var_45 = module_1.from_json_schema_type(var_42, var_39, var_43, var_44)
    var_46 = var_45.items
    var_47 = 'properties'
    var_48 = 'required'
    var_49 = 'object'
    var_50 = 'name'
    var_51 = 'age'
    var_52 = {var_0: var_20}
    var_53 = {var_0: var_12}
    var_54 = {var_50: var_52, var_51: var_53}
    var_55 = [var_50]
    var_56 = 'John'
    var_57 = 30
    var_58 = {var_50: var_56, var_51: var_57}
    var_59 = {var_0: var_49, var_47: var_54, var_48: var_55, var_3: var_58}
    var_60 = False
    var_61 = module_0.Definitions()
    var_62 = module_1.from_json_schema_type(var_59, var_49, var_60, var_61)
    var_63 = var_62.properties[var_50]
    var_64 = var_62.properties[var_51]
    var_65 = {var_0: var_20}
    var_66 = module_0.Definitions()
    var_67 = module_1.from_json_schema_type(var_65, var_20, var_30, var_66)



# Parsed testcases at query #12
#--------------------------


import typesystem.json_schema as module_0
import typesystem.fields as module_1
import typesystem.composites as module_2
import typesystem.schemas as module_3

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
    var_10 = module_1.String()
    var_11 = 'integer'
    var_12 = {var_6: var_11}
    var_13 = module_0.from_json_schema(var_12)
    var_14 = module_1.Integer()
    var_15 = 'number'
    var_16 = {var_6: var_15}
    var_17 = module_0.from_json_schema(var_16)
    var_18 = module_1.Number()
    var_19 = 'boolean'
    var_20 = {var_6: var_19}
    var_21 = module_0.from_json_schema(var_20)
    var_22 = module_1.Boolean()
    var_23 = 'array'
    var_24 = {var_6: var_23}
    var_25 = module_0.from_json_schema(var_24)
    var_26 = module_1.Any()
    var_27 = module_1.Array(var_26)
    var_28 = 'object'
    var_29 = {var_6: var_28}
    var_30 = module_0.from_json_schema(var_29)
    var_31 = module_1.Object()
    var_32 = 'enum'
    var_33 = 'a'
    var_34 = 'b'
    var_35 = 'c'
    var_36 = [var_33, var_34, var_35]
    var_37 = {var_32: var_36}
    var_38 = module_0.from_json_schema(var_37)
    var_39 = [var_33, var_34, var_35]
    var_40 = module_1.Choice(choices=var_39)
    var_41 = 'const'
    var_42 = 'value'
    var_43 = {var_41: var_42}
    var_44 = module_0.from_json_schema(var_43)
    var_45 = module_1.Const()
    var_46 = 'allOf'
    var_47 = {var_6: var_7}
    var_48 = 'minLength'
    var_49 = 5
    var_50 = {var_48: var_49}
    var_51 = [var_47, var_50]
    var_52 = {var_46: var_51}
    var_53 = module_0.from_json_schema(var_52)
    var_54 = module_1.String()
    var_55 = module_1.String(min_length=var_49)
    var_56 = [var_54, var_55]
    var_57 = module_2.AllOf(var_56)
    var_58 = 'anyOf'
    var_59 = {var_6: var_7}
    var_60 = {var_6: var_11}
    var_61 = [var_59, var_60]
    var_62 = {var_58: var_61}
    var_63 = module_0.from_json_schema(var_62)
    var_64 = module_1.String()
    var_65 = module_1.Integer()
    var_66 = [var_64, var_65]
    var_67 = module_2.OneOf(var_66)
    var_68 = 'oneOf'
    var_69 = {var_6: var_7}
    var_70 = {var_6: var_11}
    var_71 = [var_69, var_70]
    var_72 = {var_68: var_71}
    var_73 = module_0.from_json_schema(var_72)
    var_74 = module_1.String()
    var_75 = module_1.Integer()
    var_76 = [var_74, var_75]
    var_77 = module_2.OneOf(var_76)
    var_78 = 'not'
    var_79 = {var_6: var_7}
    var_80 = {var_78: var_79}
    var_81 = module_0.from_json_schema(var_80)
    var_82 = module_1.String()
    var_83 = module_2.Not(var_82)
    var_84 = 'if'
    var_85 = 'then'
    var_86 = 'else'
    var_87 = {var_6: var_7}
    var_88 = {var_48: var_49}
    var_89 = {var_6: var_11}
    var_90 = {var_84: var_87, var_85: var_88, var_86: var_89}
    var_91 = module_0.from_json_schema(var_90)
    var_92 = module_1.String()
    var_93 = module_1.String(min_length=var_49)
    var_94 = module_1.Integer()
    var_95 = module_2.IfThenElse()
    var_96 = module_3.Definitions()
    var_97 = '$ref'
    var_98 = '#/components/schemas/Test'
    var_99 = {var_97: var_98}
    var_100 = module_0.from_json_schema(var_99, var_96)
    var_101 = 'Test'
    var_102 = module_3.Reference(var_101, var_96)
    var_103 = 'maxLength'
    var_104 = 'pattern'
    var_105 = 10
    var_106 = '^[a-z]+$'
    var_107 = {var_6: var_7, var_48: var_49, var_103: var_105, var_104: var_106}
    var_108 = module_0.from_json_schema(var_107)



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
    var_4 = 1
    var_5 = 10
    var_6 = '[a-z]+'
    var_7 = 'email'
    var_8 = module_0.String(max_length=var_5, min_length=var_4, pattern=var_6, format=var_7)
    var_9 = 'type'
    var_10 = 'minLength'
    var_11 = 'maxLength'
    var_12 = 'pattern'
    var_13 = 'format'
    var_14 = 'string'
    var_15 = {var_9: var_14, var_10: var_4, var_11: var_5, var_12: var_6, var_13: var_7}
    var_16 = module_1.to_json_schema(var_8)
    var_17 = 0
    var_18 = 100
    var_19 = module_0.Integer(minimum=var_17, maximum=var_18)
    var_20 = 'minimum'
    var_21 = 'maximum'
    var_22 = 'integer'
    var_23 = {var_9: var_22, var_20: var_17, var_21: var_18}
    var_24 = module_1.to_json_schema(var_19)
    var_25 = 0.5
    var_26 = module_0.Float(multiple_of=var_25)
    var_27 = 'multipleOf'
    var_28 = 'number'
    var_29 = {var_9: var_28, var_27: var_25}
    var_30 = module_1.to_json_schema(var_26)
    var_31 = module_0.Boolean()
    var_32 = 'boolean'
    var_33 = {var_9: var_32}
    var_34 = module_1.to_json_schema(var_31)
    var_35 = module_0.String()
    var_36 = 5
    var_37 = module_0.Array(var_35, min_items=var_4, max_items=var_36)
    var_38 = 'items'
    var_39 = 'minItems'
    var_40 = 'maxItems'
    var_41 = 'array'
    var_42 = {var_9: var_14}
    var_43 = {var_9: var_41, var_38: var_42, var_39: var_4, var_40: var_36}
    var_44 = module_1.to_json_schema(var_37)
    var_45 = 'name'
    var_46 = module_0.String()
    var_47 = {var_45: var_46}
    var_48 = [var_45]
    var_49 = module_0.Object(properties=var_47, required=var_48)
    var_50 = 'properties'
    var_51 = 'required'
    var_52 = 'object'
    var_53 = {var_9: var_14}
    var_54 = {var_45: var_53}
    var_55 = [var_45]
    var_56 = {var_9: var_52, var_50: var_54, var_51: var_55}
    var_57 = module_1.to_json_schema(var_49)
    var_58 = 'a'
    var_59 = (var_58, var_58)
    var_60 = 'b'
    var_61 = (var_60, var_60)
    var_62 = [var_59, var_61]
    var_63 = module_0.Choice(choices=var_62)
    var_64 = 'enum'
    var_65 = [var_58, var_60]
    var_66 = {var_64: var_65}
    var_67 = module_1.to_json_schema(var_63)
    var_68 = 'fixed'
    var_69 = module_0.Const(var_68)
    var_70 = 'const'
    var_71 = {var_70: var_68}
    var_72 = module_1.to_json_schema(var_69)
    var_73 = module_0.String()
    var_74 = module_0.Integer()
    var_75 = [var_73, var_74]
    var_76 = module_0.Union(var_75)
    var_77 = 'anyOf'
    var_78 = {var_9: var_14}
    var_79 = {var_9: var_22}
    var_80 = [var_78, var_79]
    var_81 = {var_77: var_80}
    var_82 = module_1.to_json_schema(var_76)
    var_83 = module_0.String()
    var_84 = 'test'
    var_85 = module_0.Const(var_84)
    var_86 = [var_83, var_85]
    var_87 = module_2.AllOf(var_86)
    var_88 = 'allOf'
    var_89 = {var_9: var_14}
    var_90 = {var_70: var_84}
    var_91 = [var_89, var_90]
    var_92 = {var_88: var_91}
    var_93 = module_1.to_json_schema(var_87)
    var_94 = 'Test'
    var_95 = module_0.String()
    var_96 = {var_94: var_95}
    var_97 = '$ref'
    var_98 = 'components'
    var_99 = '#/components/schemas/Test'
    var_100 = 'schemas'
    var_101 = {var_9: var_14}
    var_102 = {var_94: var_101}
    var_103 = {var_100: var_102}
    var_104 = {var_97: var_99, var_98: var_103}
    var_105 = module_0.String()
    var_106 = {var_45: var_105}
    var_107 = module_3.Schema(var_106)
    var_108 = {var_9: var_14}
    var_109 = {var_45: var_108}
    var_110 = {var_9: var_52, var_50: var_109}
    var_111 = module_1.to_json_schema(var_107)
    var_112 = module_0.String()
    var_113 = module_0.Integer()
    var_114 = module_2.IfThenElse(var_112, var_113)
    var_115 = 'if'
    var_116 = 'then'
    var_117 = {var_9: var_14}
    var_118 = {var_9: var_22}
    var_119 = {var_115: var_117, var_116: var_118}
    var_120 = module_1.to_json_schema(var_114)
    var_121 = module_0.String()
    var_122 = module_2.Not(var_121)
    var_123 = 'not'
    var_124 = {var_9: var_14}
    var_125 = {var_123: var_124}
    var_126 = module_1.to_json_schema(var_122)
    var_127 = True
    var_128 = module_0.String()
    var_129 = 'null'
    var_130 = [var_14, var_129]
    var_131 = {var_9: var_130}
    var_132 = module_1.to_json_schema(var_128)
    var_133 = 'default'
    var_134 = module_0.String()
    var_135 = {var_9: var_14, var_133: var_133}
    var_136 = module_1.to_json_schema(var_134)



# Parsed testcases at query #14
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
    var_8 = 'null'
    var_9 = [var_5, var_8]
    var_10 = {var_4: var_9}
    var_11 = module_0.from_json_schema(var_10)
    var_12 = 'enum'
    var_13 = 'a'
    var_14 = 'b'
    var_15 = 'c'
    var_16 = [var_13, var_14, var_15]
    var_17 = {var_12: var_16}
    var_18 = module_0.from_json_schema(var_17)
    var_19 = 'const'
    var_20 = 'fixed_value'
    var_21 = {var_19: var_20}
    var_22 = module_0.from_json_schema(var_21)
    var_23 = 'allOf'
    var_24 = 'minLength'
    var_25 = 5
    var_26 = {var_4: var_5, var_24: var_25}
    var_27 = 'maxLength'
    var_28 = 10
    var_29 = {var_4: var_5, var_27: var_28}
    var_30 = [var_26, var_29]
    var_31 = {var_23: var_30}
    var_32 = module_0.from_json_schema(var_31)
    var_33 = var_32.schemas
    var_34 = len(var_33)
    assert var_34 == 2
    var_35 = 'anyOf'
    var_36 = {var_4: var_5}
    var_37 = 'number'
    var_38 = {var_4: var_37}
    var_39 = [var_36, var_38]
    var_40 = {var_35: var_39}
    var_41 = module_0.from_json_schema(var_40)
    var_42 = var_41.schemas
    var_43 = len(var_42)
    assert var_43 == 2
    var_44 = 'oneOf'
    var_45 = {var_4: var_5}
    var_46 = {var_4: var_37}
    var_47 = [var_45, var_46]
    var_48 = {var_44: var_47}
    var_49 = module_0.from_json_schema(var_48)
    var_50 = var_49.schemas
    var_51 = len(var_50)
    assert var_51 == 2
    var_52 = 'not'
    var_53 = {var_4: var_5}
    var_54 = {var_52: var_53}
    var_55 = module_0.from_json_schema(var_54)
    var_56 = var_55.schema
    var_57 = 'if'
    var_58 = 'then'
    var_59 = 'else'
    var_60 = {var_4: var_5}
    var_61 = {var_24: var_25}
    var_62 = {var_4: var_37}
    var_63 = {var_57: var_60, var_58: var_61, var_59: var_62}
    var_64 = module_0.from_json_schema(var_63)
    var_65 = var_64.if_schema
    var_66 = var_64.then_schema
    var_67 = var_64.else_schema
    var_68 = module_1.Definitions()
    var_69 = 'properties'
    var_70 = 'object'
    var_71 = 'name'
    var_72 = {var_4: var_5}
    var_73 = {var_71: var_72}
    var_74 = {var_4: var_70, var_69: var_73}
    var_75 = '$ref'
    var_76 = '#/components/schemas/Person'
    var_77 = {var_75: var_76}
    var_78 = module_0.from_json_schema(var_77, var_68)
    var_79 = 'pattern'
    var_80 = '^[a-z]+$'
    var_81 = {var_4: var_5, var_24: var_25, var_27: var_28, var_79: var_80}
    var_82 = module_0.from_json_schema(var_81)
    var_83 = var_82.schemas
    var_84 = len(var_83)
    assert var_84 == 2
    var_85 = {}
    var_86 = module_0.from_json_schema(var_85)



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
    var_4 = 1
    var_5 = 10
    var_6 = '[a-z]+'
    var_7 = 'email'
    var_8 = module_0.String(max_length=var_5, min_length=var_4, pattern=var_6, format=var_7)
    var_9 = 'type'
    var_10 = 'minLength'
    var_11 = 'maxLength'
    var_12 = 'pattern'
    var_13 = 'format'
    var_14 = 'string'
    var_15 = {var_9: var_14, var_10: var_4, var_11: var_5, var_12: var_6, var_13: var_7}
    var_16 = module_1.to_json_schema(var_8)
    var_17 = 0
    var_18 = 100
    var_19 = module_0.Integer(minimum=var_17, maximum=var_18)
    var_20 = 'minimum'
    var_21 = 'maximum'
    var_22 = 'integer'
    var_23 = {var_9: var_22, var_20: var_17, var_21: var_18}
    var_24 = module_1.to_json_schema(var_19)
    var_25 = 0.5
    var_26 = module_0.Float(multiple_of=var_25)
    var_27 = 'multipleOf'
    var_28 = 'number'
    var_29 = {var_9: var_28, var_27: var_25}
    var_30 = module_1.to_json_schema(var_26)
    var_31 = module_0.Boolean()
    var_32 = 'boolean'
    var_33 = {var_9: var_32}
    var_34 = module_1.to_json_schema(var_31)
    var_35 = module_0.String()
    var_36 = 5
    var_37 = True
    var_38 = module_0.Array(var_35, min_items=var_4, max_items=var_36, unique_items=var_37)
    var_39 = 'minItems'
    var_40 = 'maxItems'
    var_41 = 'items'
    var_42 = 'uniqueItems'
    var_43 = 'array'
    var_44 = {var_9: var_14}
    var_45 = True
    var_46 = {var_9: var_43, var_39: var_37, var_40: var_36, var_41: var_44, var_42: var_45}
    var_47 = module_1.to_json_schema(var_38)
    var_48 = 'name'
    var_49 = module_0.String()
    var_50 = {var_48: var_49}
    var_51 = [var_48]
    var_52 = 2
    var_53 = module_0.Object(properties=var_50, min_properties=var_45, max_properties=var_52, required=var_51)
    var_54 = 'properties'
    var_55 = 'required'
    var_56 = 'minProperties'
    var_57 = 'maxProperties'
    var_58 = 'object'
    var_59 = {var_9: var_14}
    var_60 = {var_48: var_59}
    var_61 = [var_48]
    var_62 = {var_9: var_58, var_54: var_60, var_55: var_61, var_56: var_45, var_57: var_52}
    var_63 = module_1.to_json_schema(var_53)
    var_64 = 'a'
    var_65 = (var_64, var_64)
    var_66 = 'b'
    var_67 = (var_66, var_66)
    var_68 = [var_65, var_67]
    var_69 = module_0.Choice(choices=var_68)
    var_70 = 'enum'
    var_71 = [var_64, var_66]
    var_72 = {var_70: var_71}
    var_73 = module_1.to_json_schema(var_69)
    var_74 = 'fixed_value'
    var_75 = module_0.Const(var_74)
    var_76 = 'const'
    var_77 = {var_76: var_74}
    var_78 = module_1.to_json_schema(var_75)
    var_79 = module_0.String()
    var_80 = module_0.Integer()
    var_81 = [var_79, var_80]
    var_82 = module_0.Union(var_81)
    var_83 = 'anyOf'
    var_84 = {var_9: var_14}
    var_85 = {var_9: var_22}
    var_86 = [var_84, var_85]
    var_87 = {var_83: var_86}
    var_88 = module_1.to_json_schema(var_82)
    var_89 = module_0.String()
    var_90 = module_0.Integer()
    var_91 = [var_89, var_90]
    var_92 = module_2.AllOf(var_91)
    var_93 = 'allOf'
    var_94 = {var_9: var_14}
    var_95 = {var_9: var_22}
    var_96 = [var_94, var_95]
    var_97 = {var_93: var_96}
    var_98 = module_1.to_json_schema(var_92)
    var_99 = module_0.String()
    var_100 = module_0.Integer()
    var_101 = [var_99, var_100]
    var_102 = module_2.OneOf(var_101)
    var_103 = 'oneOf'
    var_104 = {var_9: var_14}
    var_105 = {var_9: var_22}
    var_106 = [var_104, var_105]
    var_107 = {var_103: var_106}
    var_108 = module_1.to_json_schema(var_102)
    var_109 = module_0.String()
    var_110 = module_0.Integer()
    var_111 = module_0.Boolean()
    var_112 = module_2.IfThenElse(var_109, var_110, var_111)
    var_113 = 'if'
    var_114 = 'then'
    var_115 = 'else'
    var_116 = {var_9: var_14}
    var_117 = {var_9: var_22}
    var_118 = {var_9: var_32}
    var_119 = {var_113: var_116, var_114: var_117, var_115: var_118}
    var_120 = module_1.to_json_schema(var_112)
    var_121 = module_0.String()
    var_122 = module_2.Not(var_121)
    var_123 = 'not'
    var_124 = {var_9: var_14}
    var_125 = {var_123: var_124}
    var_126 = module_1.to_json_schema(var_122)
    var_127 = 'TestSchema'
    var_128 = module_0.String()
    var_129 = {var_127: var_128}
    var_130 = '$ref'
    var_131 = 'components'
    var_132 = '#/components/schemas/TestSchema'
    var_133 = 'schemas'
    var_134 = {var_9: var_14}
    var_135 = {var_127: var_134}
    var_136 = {var_133: var_135}
    var_137 = {var_130: var_132, var_131: var_136}
    var_138 = module_0.String()
    var_139 = {var_48: var_138}
    var_140 = [var_48]
    var_141 = module_3.Schema(var_139)
    var_142 = {var_9: var_14}
    var_143 = {var_48: var_142}
    var_144 = [var_48]
    var_145 = {var_9: var_58, var_54: var_143, var_55: var_144}
    var_146 = module_1.to_json_schema(var_141)



# Parsed testcases at query #16
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
    var_8 = 0
    var_9 = 100
    var_10 = 2
    var_11 = 50
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
    var_26 = 5
    var_27 = 'email'
    var_28 = '^[a-zA-Z0-9]+$'
    var_29 = 'test'
    var_30 = {var_0: var_25, var_21: var_26, var_22: var_9, var_23: var_27, var_24: var_28, var_6: var_29}
    var_31 = False
    var_32 = module_0.Definitions()
    var_33 = module_1.from_json_schema_type(var_30, var_25, var_31, var_32)
    var_34 = 'boolean'
    var_35 = True
    var_36 = {var_0: var_34, var_6: var_35}
    var_37 = False
    var_38 = module_0.Definitions()
    var_39 = module_1.from_json_schema_type(var_36, var_34, var_37, var_38)
    var_40 = 'items'
    var_41 = 'additionalItems'
    var_42 = 'minItems'
    var_43 = 'maxItems'
    var_44 = 'uniqueItems'
    var_45 = 'array'
    var_46 = {var_0: var_25}
    var_47 = False
    var_48 = 10
    var_49 = [var_29]
    var_50 = {var_0: var_45, var_40: var_46, var_41: var_47, var_42: var_35, var_43: var_48, var_44: var_35, var_6: var_49}
    var_51 = False
    var_52 = module_0.Definitions()
    var_53 = module_1.from_json_schema_type(var_50, var_45, var_51, var_52)
    var_54 = var_53.items
    var_55 = 'properties'
    var_56 = 'patternProperties'
    var_57 = 'additionalProperties'
    var_58 = 'propertyNames'
    var_59 = 'minProperties'
    var_60 = 'maxProperties'
    var_61 = 'required'
    var_62 = 'object'
    var_63 = 'name'
    var_64 = 'age'
    var_65 = {var_0: var_25}
    var_66 = {var_0: var_16}
    var_67 = {var_63: var_65, var_64: var_66}
    var_68 = '^S_'
    var_69 = '^I_'
    var_70 = {var_0: var_25}
    var_71 = {var_0: var_16}
    var_72 = {var_68: var_70, var_69: var_71}
    var_73 = False
    var_74 = {var_0: var_25}
    var_75 = [var_63]
    var_76 = 30
    var_77 = {var_63: var_29, var_64: var_76}
    var_78 = {var_0: var_62, var_55: var_67, var_56: var_72, var_57: var_73, var_58: var_74, var_59: var_35, var_60: var_48, var_61: var_75, var_6: var_77}
    var_79 = False
    var_80 = module_0.Definitions()
    var_81 = module_1.from_json_schema_type(var_78, var_62, var_79, var_80)
    var_82 = var_81.properties[var_63]
    var_83 = var_81.properties[var_64]
    var_84 = var_81.pattern_properties[var_68]
    var_85 = var_81.pattern_properties[var_69]
    var_86 = var_81.property_names
    var_87 = {}
    var_88 = 'invalid'
    var_89 = False
    var_90 = module_0.Definitions()
    var_91 = module_1.from_json_schema_type(var_87, var_88, var_89, var_90)



# Parsed testcases at query #17
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
    var_8 = 0
    var_9 = 100
    var_10 = 2
    var_11 = 50
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
    var_26 = 5
    var_27 = 'email'
    var_28 = '^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$'
    var_29 = 'test@example.com'
    var_30 = {var_0: var_25, var_21: var_26, var_22: var_9, var_23: var_27, var_24: var_28, var_6: var_29}
    var_31 = False
    var_32 = module_0.Definitions()
    var_33 = module_1.from_json_schema_type(var_30, var_25, var_31, var_32)
    var_34 = 'boolean'
    var_35 = True
    var_36 = {var_0: var_34, var_6: var_35}
    var_37 = False
    var_38 = module_0.Definitions()
    var_39 = module_1.from_json_schema_type(var_36, var_34, var_37, var_38)
    var_40 = 'items'
    var_41 = 'minItems'
    var_42 = 'maxItems'
    var_43 = 'uniqueItems'
    var_44 = 'array'
    var_45 = {var_0: var_25}
    var_46 = 10
    var_47 = 'item1'
    var_48 = 'item2'
    var_49 = [var_47, var_48]
    var_50 = {var_0: var_44, var_40: var_45, var_41: var_35, var_42: var_46, var_43: var_35, var_6: var_49}
    var_51 = False
    var_52 = module_0.Definitions()
    var_53 = module_1.from_json_schema_type(var_50, var_44, var_51, var_52)
    var_54 = var_53.items
    var_55 = 'properties'
    var_56 = 'required'
    var_57 = 'minProperties'
    var_58 = 'maxProperties'
    var_59 = 'object'
    var_60 = 'name'
    var_61 = 'age'
    var_62 = {var_0: var_25}
    var_63 = {var_0: var_16}
    var_64 = {var_60: var_62, var_61: var_63}
    var_65 = [var_60]
    var_66 = 'John'
    var_67 = 30
    var_68 = {var_60: var_66, var_61: var_67}
    var_69 = {var_0: var_59, var_55: var_64, var_56: var_65, var_57: var_35, var_58: var_26, var_6: var_68}
    var_70 = False
    var_71 = module_0.Definitions()
    var_72 = module_1.from_json_schema_type(var_69, var_59, var_70, var_71)
    var_73 = var_72.properties[var_60]
    var_74 = var_72.properties[var_61]
    var_75 = {var_0: var_25}
    var_76 = module_0.Definitions()
    var_77 = module_1.from_json_schema_type(var_75, var_25, var_35, var_76)



# Parsed testcases at query #18
#--------------------------


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
    var_12 = 'integer'
    var_13 = {var_0: var_12, var_1: var_9, var_2: var_6, var_3: var_7}
    var_14 = False
    var_15 = module_0.Definitions()
    var_16 = module_1.from_json_schema_type(var_13, var_12, var_14, var_15)
    var_17 = 'minLength'
    var_18 = 'maxLength'
    var_19 = 'pattern'
    var_20 = 'string'
    var_21 = 5
    var_22 = '^[a-zA-Z0-9]+$'
    var_23 = 'test'
    var_24 = {var_0: var_20, var_17: var_21, var_18: var_6, var_19: var_22, var_3: var_23}
    var_25 = False
    var_26 = module_0.Definitions()
    var_27 = module_1.from_json_schema_type(var_24, var_20, var_25, var_26)
    var_28 = 'boolean'
    var_29 = True
    var_30 = {var_0: var_28, var_3: var_29}
    var_31 = False
    var_32 = module_0.Definitions()
    var_33 = module_1.from_json_schema_type(var_30, var_28, var_31, var_32)
    var_34 = 'items'
    var_35 = 'minItems'
    var_36 = 'maxItems'
    var_37 = 'uniqueItems'
    var_38 = 'array'
    var_39 = {var_0: var_20}
    var_40 = 10
    var_41 = [var_23]
    var_42 = {var_0: var_38, var_34: var_39, var_35: var_29, var_36: var_40, var_37: var_29, var_3: var_41}
    var_43 = False
    var_44 = module_0.Definitions()
    var_45 = module_1.from_json_schema_type(var_42, var_38, var_43, var_44)
    var_46 = var_45.items
    var_47 = 'properties'
    var_48 = 'required'
    var_49 = 'object'
    var_50 = 'name'
    var_51 = 'age'
    var_52 = {var_0: var_20}
    var_53 = {var_0: var_12}
    var_54 = {var_50: var_52, var_51: var_53}
    var_55 = [var_50]
    var_56 = 25
    var_57 = {var_50: var_23, var_51: var_56}
    var_58 = {var_0: var_49, var_47: var_54, var_48: var_55, var_3: var_57}
    var_59 = False
    var_60 = module_0.Definitions()
    var_61 = module_1.from_json_schema_type(var_58, var_49, var_59, var_60)
    var_62 = var_61.properties[var_50]
    var_63 = var_61.properties[var_51]
    var_64 = {var_0: var_20}
    var_65 = module_0.Definitions()
    var_66 = module_1.from_json_schema_type(var_64, var_20, var_29, var_65)



# Parsed testcases at query #19
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'type'
    var_2 = 'minimum'
    var_3 = 'maximum'
    var_4 = 'default'
    var_5 = 'number'
    var_6 = 0
    var_7 = 100
    var_8 = 50
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = False
    var_11 = module_1.from_json_schema_type(var_9, var_5, var_10, var_0)
    var_12 = 'integer'
    var_13 = {var_1: var_12, var_2: var_10, var_3: var_7, var_4: var_8}
    var_14 = False
    var_15 = module_1.from_json_schema_type(var_13, var_12, var_14, var_0)
    var_16 = 'minLength'
    var_17 = 'maxLength'
    var_18 = 'pattern'
    var_19 = 'string'
    var_20 = 5
    var_21 = 10
    var_22 = '^[A-Za-z]+$'
    var_23 = 'hello'
    var_24 = {var_1: var_19, var_16: var_20, var_17: var_21, var_18: var_22, var_4: var_23}
    var_25 = False
    var_26 = module_1.from_json_schema_type(var_24, var_19, var_25, var_0)
    var_27 = 'boolean'
    var_28 = True
    var_29 = {var_1: var_27, var_4: var_28}
    var_30 = False
    var_31 = module_1.from_json_schema_type(var_29, var_27, var_30, var_0)
    var_32 = 'items'
    var_33 = 'minItems'
    var_34 = 'maxItems'
    var_35 = 'uniqueItems'
    var_36 = 'array'
    var_37 = {var_1: var_19}
    var_38 = [var_23]
    var_39 = {var_1: var_36, var_32: var_37, var_33: var_28, var_34: var_21, var_35: var_28, var_4: var_38}
    var_40 = False
    var_41 = module_1.from_json_schema_type(var_39, var_36, var_40, var_0)
    var_42 = var_41.items
    var_43 = 'properties'
    var_44 = 'required'
    var_45 = 'object'
    var_46 = 'name'
    var_47 = 'age'
    var_48 = {var_1: var_19}
    var_49 = {var_1: var_12}
    var_50 = {var_46: var_48, var_47: var_49}
    var_51 = [var_46]
    var_52 = 'John'
    var_53 = 30
    var_54 = {var_46: var_52, var_47: var_53}
    var_55 = {var_1: var_45, var_43: var_50, var_44: var_51, var_4: var_54}
    var_56 = False
    var_57 = module_1.from_json_schema_type(var_55, var_45, var_56, var_0)
    var_58 = var_57.properties[var_46]
    var_59 = var_57.properties[var_47]



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
    var_4 = 1
    var_5 = 10
    var_6 = '[a-z]+'
    var_7 = 'email'
    var_8 = module_0.String(max_length=var_5, min_length=var_4, pattern=var_6, format=var_7)
    var_9 = 'type'
    var_10 = 'minLength'
    var_11 = 'maxLength'
    var_12 = 'pattern'
    var_13 = 'format'
    var_14 = 'string'
    var_15 = {var_9: var_14, var_10: var_4, var_11: var_5, var_12: var_6, var_13: var_7}
    var_16 = module_1.to_json_schema(var_8)
    var_17 = 0
    var_18 = 100
    var_19 = module_0.Integer(minimum=var_17, maximum=var_18)
    var_20 = 'minimum'
    var_21 = 'maximum'
    var_22 = 'integer'
    var_23 = {var_9: var_22, var_20: var_17, var_21: var_18}
    var_24 = module_1.to_json_schema(var_19)
    var_25 = 0.5
    var_26 = module_0.Float(multiple_of=var_25)
    var_27 = 'multipleOf'
    var_28 = 'number'
    var_29 = {var_9: var_28, var_27: var_25}
    var_30 = module_1.to_json_schema(var_26)
    var_31 = module_0.Boolean()
    var_32 = 'boolean'
    var_33 = {var_9: var_32}
    var_34 = module_1.to_json_schema(var_31)
    var_35 = module_0.String()
    var_36 = 5
    var_37 = module_0.Array(var_35, min_items=var_4, max_items=var_36)
    var_38 = 'minItems'
    var_39 = 'maxItems'
    var_40 = 'items'
    var_41 = 'array'
    var_42 = {var_9: var_14}
    var_43 = {var_9: var_41, var_38: var_4, var_39: var_36, var_40: var_42}
    var_44 = module_1.to_json_schema(var_37)
    var_45 = 'name'
    var_46 = 'age'
    var_47 = module_0.String()
    var_48 = module_0.Integer()
    var_49 = {var_45: var_47, var_46: var_48}
    var_50 = [var_45]
    var_51 = module_0.Object(properties=var_49, required=var_50)
    var_52 = 'properties'
    var_53 = 'required'
    var_54 = 'object'
    var_55 = {var_9: var_14}
    var_56 = {var_9: var_22}
    var_57 = {var_45: var_55, var_46: var_56}
    var_58 = [var_45]
    var_59 = {var_9: var_54, var_52: var_57, var_53: var_58}
    var_60 = module_1.to_json_schema(var_51)
    var_61 = 'a'
    var_62 = 'A'
    var_63 = (var_61, var_62)
    var_64 = 'b'
    var_65 = 'B'
    var_66 = (var_64, var_65)
    var_67 = [var_63, var_66]
    var_68 = module_0.Choice(choices=var_67)
    var_69 = 'enum'
    var_70 = [var_61, var_64]
    var_71 = {var_69: var_70}
    var_72 = module_1.to_json_schema(var_68)
    var_73 = 'fixed_value'
    var_74 = module_0.Const(var_73)
    var_75 = 'const'
    var_76 = {var_75: var_73}
    var_77 = module_1.to_json_schema(var_74)
    var_78 = module_0.String()
    var_79 = module_0.Integer()
    var_80 = [var_78, var_79]
    var_81 = module_0.Union(var_80)
    var_82 = 'anyOf'
    var_83 = {var_9: var_14}
    var_84 = {var_9: var_22}
    var_85 = [var_83, var_84]
    var_86 = {var_82: var_85}
    var_87 = module_1.to_json_schema(var_81)
    var_88 = module_0.String()
    var_89 = module_0.Integer()
    var_90 = [var_88, var_89]
    var_91 = module_2.OneOf(var_90)
    var_92 = 'oneOf'
    var_93 = {var_9: var_14}
    var_94 = {var_9: var_22}
    var_95 = [var_93, var_94]
    var_96 = {var_92: var_95}
    var_97 = module_1.to_json_schema(var_91)
    var_98 = module_0.String()
    var_99 = module_0.Integer()
    var_100 = [var_98, var_99]
    var_101 = module_2.AllOf(var_100)
    var_102 = 'allOf'
    var_103 = {var_9: var_14}
    var_104 = {var_9: var_22}
    var_105 = [var_103, var_104]
    var_106 = {var_102: var_105}
    var_107 = module_1.to_json_schema(var_101)
    var_108 = module_0.String()
    var_109 = module_0.Integer()
    var_110 = module_0.Boolean()
    var_111 = module_2.IfThenElse(var_108, var_109, var_110)
    var_112 = 'if'
    var_113 = 'then'
    var_114 = 'else'
    var_115 = {var_9: var_14}
    var_116 = {var_9: var_22}
    var_117 = {var_9: var_32}
    var_118 = {var_112: var_115, var_113: var_116, var_114: var_117}
    var_119 = module_1.to_json_schema(var_111)
    var_120 = module_0.String()
    var_121 = module_2.Not(var_120)
    var_122 = 'not'
    var_123 = {var_9: var_14}
    var_124 = {var_122: var_123}
    var_125 = module_1.to_json_schema(var_121)
    var_126 = 'Person'
    var_127 = module_0.String()
    var_128 = {var_45: var_127}
    var_129 = module_0.Object(properties=var_128)
    var_130 = {var_126: var_129}
    var_131 = '$ref'
    var_132 = 'components'
    var_133 = '#/components/schemas/Person'
    var_134 = 'schemas'
    var_135 = {var_9: var_14}
    var_136 = {var_45: var_135}
    var_137 = {var_9: var_54, var_52: var_136}
    var_138 = {var_126: var_137}
    var_139 = {var_134: var_138}
    var_140 = {var_131: var_133, var_132: var_139}
    var_141 = module_0.String()
    var_142 = module_0.Integer()
    var_143 = {var_45: var_141, var_46: var_142}
    var_144 = [var_45]
    var_145 = module_3.Schema(var_143)
    var_146 = {var_9: var_14}
    var_147 = {var_9: var_22}
    var_148 = {var_45: var_146, var_46: var_147}
    var_149 = [var_45]
    var_150 = {var_9: var_54, var_52: var_148, var_53: var_149}
    var_151 = module_1.to_json_schema(var_145)



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
    var_4 = 1
    var_5 = 10
    var_6 = '[a-z]+'
    var_7 = 'email'
    var_8 = module_0.String(max_length=var_5, min_length=var_4, pattern=var_6, format=var_7)
    var_9 = 'type'
    var_10 = 'minLength'
    var_11 = 'maxLength'
    var_12 = 'pattern'
    var_13 = 'format'
    var_14 = 'string'
    var_15 = {var_9: var_14, var_10: var_4, var_11: var_5, var_12: var_6, var_13: var_7}
    var_16 = module_1.to_json_schema(var_8)
    var_17 = 0
    var_18 = 100
    var_19 = module_0.Integer(minimum=var_17, maximum=var_18)
    var_20 = 'minimum'
    var_21 = 'maximum'
    var_22 = 'integer'
    var_23 = {var_9: var_22, var_20: var_17, var_21: var_18}
    var_24 = module_1.to_json_schema(var_19)
    var_25 = 0.1
    var_26 = module_0.Float(minimum=var_17, maximum=var_4, multiple_of=var_25)
    var_27 = 'multipleOf'
    var_28 = 'number'
    var_29 = {var_9: var_28, var_20: var_17, var_21: var_4, var_27: var_25}
    var_30 = module_1.to_json_schema(var_26)
    var_31 = module_0.Boolean()
    var_32 = 'boolean'
    var_33 = {var_9: var_32}
    var_34 = module_1.to_json_schema(var_31)
    var_35 = module_0.String()
    var_36 = module_0.Array(var_35, min_items=var_4, max_items=var_5)
    var_37 = 'items'
    var_38 = 'minItems'
    var_39 = 'maxItems'
    var_40 = 'array'
    var_41 = {var_9: var_14}
    var_42 = {var_9: var_40, var_37: var_41, var_38: var_4, var_39: var_5}
    var_43 = module_1.to_json_schema(var_36)
    var_44 = 'name'
    var_45 = module_0.String()
    var_46 = {var_44: var_45}
    var_47 = [var_44]
    var_48 = module_0.Object(properties=var_46, required=var_47)
    var_49 = 'properties'
    var_50 = 'required'
    var_51 = 'object'
    var_52 = {var_9: var_14}
    var_53 = {var_44: var_52}
    var_54 = [var_44]
    var_55 = {var_9: var_51, var_49: var_53, var_50: var_54}
    var_56 = module_1.to_json_schema(var_48)
    var_57 = 'a'
    var_58 = (var_57, var_57)
    var_59 = 'b'
    var_60 = (var_59, var_59)
    var_61 = [var_58, var_60]
    var_62 = module_0.Choice(choices=var_61)
    var_63 = 'enum'
    var_64 = [var_57, var_59]
    var_65 = {var_63: var_64}
    var_66 = module_1.to_json_schema(var_62)
    var_67 = 'value'
    var_68 = module_0.Const(var_67)
    var_69 = 'const'
    var_70 = {var_69: var_67}
    var_71 = module_1.to_json_schema(var_68)
    var_72 = module_0.String()
    var_73 = module_0.Integer()
    var_74 = [var_72, var_73]
    var_75 = module_0.Union(var_74)
    var_76 = 'anyOf'
    var_77 = {var_9: var_14}
    var_78 = {var_9: var_22}
    var_79 = [var_77, var_78]
    var_80 = {var_76: var_79}
    var_81 = module_1.to_json_schema(var_75)
    var_82 = module_0.String()
    var_83 = module_0.Const(var_67)
    var_84 = [var_82, var_83]
    var_85 = module_2.AllOf(var_84)
    var_86 = 'allOf'
    var_87 = {var_9: var_14}
    var_88 = {var_69: var_67}
    var_89 = [var_87, var_88]
    var_90 = {var_86: var_89}
    var_91 = module_1.to_json_schema(var_85)
    var_92 = module_0.String()
    var_93 = module_0.Integer()
    var_94 = module_0.Boolean()
    var_95 = module_2.IfThenElse(var_92, var_93, var_94)
    var_96 = 'if'
    var_97 = 'then'
    var_98 = 'else'
    var_99 = {var_9: var_14}
    var_100 = {var_9: var_22}
    var_101 = {var_9: var_32}
    var_102 = {var_96: var_99, var_97: var_100, var_98: var_101}
    var_103 = module_1.to_json_schema(var_95)
    var_104 = module_0.String()
    var_105 = module_2.Not(var_104)
    var_106 = 'not'
    var_107 = {var_9: var_14}
    var_108 = {var_106: var_107}
    var_109 = module_1.to_json_schema(var_105)
    var_110 = module_3.Definitions()
    var_111 = 'test'
    var_112 = module_3.Reference(var_111, var_110)
    var_113 = '$ref'
    var_114 = 'components'
    var_115 = '#/components/schemas/test'
    var_116 = 'schemas'
    var_117 = {}
    var_118 = {var_116: var_117}
    var_119 = {var_113: var_115, var_114: var_118}
    var_120 = module_1.to_json_schema(var_112)



# Parsed testcases at query #22
#--------------------------


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
    var_12 = 'integer'
    var_13 = {var_0: var_12, var_1: var_9, var_2: var_6, var_3: var_7}
    var_14 = False
    var_15 = module_0.Definitions()
    var_16 = module_1.from_json_schema_type(var_13, var_12, var_14, var_15)
    var_17 = 'minLength'
    var_18 = 'maxLength'
    var_19 = 'string'
    var_20 = 5
    var_21 = 10
    var_22 = 'hello'
    var_23 = {var_0: var_19, var_17: var_20, var_18: var_21, var_3: var_22}
    var_24 = False
    var_25 = module_0.Definitions()
    var_26 = module_1.from_json_schema_type(var_23, var_19, var_24, var_25)
    var_27 = 'boolean'
    var_28 = True
    var_29 = {var_0: var_27, var_3: var_28}
    var_30 = False
    var_31 = module_0.Definitions()
    var_32 = module_1.from_json_schema_type(var_29, var_27, var_30, var_31)
    var_33 = 'items'
    var_34 = 'minItems'
    var_35 = 'maxItems'
    var_36 = 'array'
    var_37 = {var_0: var_19}
    var_38 = [var_22]
    var_39 = {var_0: var_36, var_33: var_37, var_34: var_28, var_35: var_20, var_3: var_38}
    var_40 = False
    var_41 = module_0.Definitions()
    var_42 = module_1.from_json_schema_type(var_39, var_36, var_40, var_41)
    var_43 = var_42.items
    var_44 = 'properties'
    var_45 = 'required'
    var_46 = 'object'
    var_47 = 'name'
    var_48 = 'age'
    var_49 = {var_0: var_19}
    var_50 = {var_0: var_12}
    var_51 = {var_47: var_49, var_48: var_50}
    var_52 = [var_47]
    var_53 = 'John'
    var_54 = 30
    var_55 = {var_47: var_53, var_48: var_54}
    var_56 = {var_0: var_46, var_44: var_51, var_45: var_52, var_3: var_55}
    var_57 = False
    var_58 = module_0.Definitions()
    var_59 = module_1.from_json_schema_type(var_56, var_46, var_57, var_58)
    var_60 = {var_0: var_19}
    var_61 = module_0.Definitions()
    var_62 = module_1.from_json_schema_type(var_60, var_19, var_28, var_61)



# Parsed testcases at query #23
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
    var_23 = [var_5, var_8]
    var_24 = {var_4: var_23}
    var_25 = module_0.from_json_schema(var_24)
    var_26 = 'enum'
    var_27 = 'a'
    var_28 = 'b'
    var_29 = 'c'
    var_30 = [var_27, var_28, var_29]
    var_31 = {var_26: var_30}
    var_32 = module_0.from_json_schema(var_31)
    var_33 = 'const'
    var_34 = 'value'
    var_35 = {var_33: var_34}
    var_36 = module_0.from_json_schema(var_35)
    var_37 = 'allOf'
    var_38 = {var_4: var_5}
    var_39 = 'minLength'
    var_40 = 5
    var_41 = {var_39: var_40}
    var_42 = [var_38, var_41]
    var_43 = {var_37: var_42}
    var_44 = module_0.from_json_schema(var_43)
    var_45 = var_44.constraints
    var_46 = len(var_45)
    assert var_46 == 2
    var_47 = 'anyOf'
    var_48 = {var_4: var_5}
    var_49 = {var_4: var_8}
    var_50 = [var_48, var_49]
    var_51 = {var_47: var_50}
    var_52 = module_0.from_json_schema(var_51)
    var_53 = var_52.options
    var_54 = len(var_53)
    assert var_54 == 2
    var_55 = 'oneOf'
    var_56 = {var_4: var_5}
    var_57 = {var_4: var_8}
    var_58 = [var_56, var_57]
    var_59 = {var_55: var_58}
    var_60 = module_0.from_json_schema(var_59)
    var_61 = var_60.options
    var_62 = len(var_61)
    assert var_62 == 2
    var_63 = 'not'
    var_64 = {var_4: var_5}
    var_65 = {var_63: var_64}
    var_66 = module_0.from_json_schema(var_65)
    var_67 = var_66.constraint
    var_68 = 'if'
    var_69 = 'then'
    var_70 = 'else'
    var_71 = {var_4: var_5}
    var_72 = {var_39: var_40}
    var_73 = {var_4: var_8}
    var_74 = {var_68: var_71, var_69: var_72, var_70: var_73}
    var_75 = module_0.from_json_schema(var_74)
    var_76 = var_75.if_constraint
    var_77 = var_75.then_constraint
    var_78 = var_75.else_constraint
    var_79 = module_1.Definitions()
    var_80 = '$ref'
    var_81 = '#/components/schemas/Test'
    var_82 = {var_80: var_81}
    var_83 = module_0.from_json_schema(var_82, var_79)
    var_84 = 'maxLength'
    var_85 = 'pattern'
    var_86 = 10
    var_87 = '^[a-z]+$'
    var_88 = {var_4: var_5, var_39: var_40, var_84: var_86, var_85: var_87}
    var_89 = module_0.from_json_schema(var_88)
    var_90 = var_89.constraints
    var_91 = len(var_90)
    assert var_91 == 4
    var_92 = {}
    var_93 = module_0.from_json_schema(var_92)



# Parsed testcases at query #24
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
    var_35 = 'minLength'
    var_36 = 5
    var_37 = {var_4: var_5, var_35: var_36}
    var_38 = 'maxLength'
    var_39 = 10
    var_40 = {var_4: var_5, var_38: var_39}
    var_41 = [var_37, var_40]
    var_42 = {var_34: var_41}
    var_43 = module_0.from_json_schema(var_42)
    var_44 = var_43.schemas
    var_45 = len(var_44)
    assert var_45 == 2
    var_46 = 'anyOf'
    var_47 = {var_4: var_5}
    var_48 = {var_4: var_8}
    var_49 = [var_47, var_48]
    var_50 = {var_46: var_49}
    var_51 = module_0.from_json_schema(var_50)
    var_52 = var_51.schemas
    var_53 = len(var_52)
    assert var_53 == 2
    var_54 = 'oneOf'
    var_55 = {var_4: var_5}
    var_56 = {var_4: var_8}
    var_57 = [var_55, var_56]
    var_58 = {var_54: var_57}
    var_59 = module_0.from_json_schema(var_58)
    var_60 = var_59.schemas
    var_61 = len(var_60)
    assert var_61 == 2
    var_62 = 'not'
    var_63 = {var_4: var_5}
    var_64 = {var_62: var_63}
    var_65 = module_0.from_json_schema(var_64)
    var_66 = var_65.schema
    var_67 = 'if'
    var_68 = 'then'
    var_69 = 'else'
    var_70 = {var_4: var_5}
    var_71 = {var_35: var_36}
    var_72 = {var_4: var_8}
    var_73 = {var_67: var_70, var_68: var_71, var_69: var_72}
    var_74 = module_0.from_json_schema(var_73)
    var_75 = var_74.if_schema
    var_76 = var_74.then_schema
    var_77 = var_74.else_schema
    var_78 = module_1.Definitions()
    var_79 = 'properties'
    var_80 = 'name'
    var_81 = {var_4: var_5}
    var_82 = {var_80: var_81}
    var_83 = '$ref'
    var_84 = '#/components/schemas/Person'
    var_85 = {var_83: var_84}
    var_86 = module_0.from_json_schema(var_85, var_78)
    var_87 = 'pattern'
    var_88 = '^[a-z]+$'
    var_89 = {var_4: var_5, var_35: var_36, var_38: var_39, var_87: var_88}
    var_90 = module_0.from_json_schema(var_89)
    var_91 = var_90.schemas
    var_92 = len(var_91)
    assert var_92 == 2



# Parsed testcases at query #25
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
    var_42 = var_41.schemas
    var_43 = len(var_42)
    assert var_43 == 2
    var_44 = 'anyOf'
    var_45 = {var_4: var_5}
    var_46 = {var_4: var_8}
    var_47 = [var_45, var_46]
    var_48 = {var_44: var_47}
    var_49 = module_0.from_json_schema(var_48)
    var_50 = var_49.schemas
    var_51 = len(var_50)
    assert var_51 == 2
    var_52 = 'oneOf'
    var_53 = {var_4: var_5}
    var_54 = {var_4: var_8}
    var_55 = [var_53, var_54]
    var_56 = {var_52: var_55}
    var_57 = module_0.from_json_schema(var_56)
    var_58 = var_57.schemas
    var_59 = len(var_58)
    assert var_59 == 2
    var_60 = 'not'
    var_61 = {var_4: var_5}
    var_62 = {var_60: var_61}
    var_63 = module_0.from_json_schema(var_62)
    var_64 = 'if'
    var_65 = 'then'
    var_66 = 'else'
    var_67 = {var_4: var_5}
    var_68 = {var_36: var_37}
    var_69 = {var_4: var_8}
    var_70 = {var_64: var_67, var_65: var_68, var_66: var_69}
    var_71 = module_0.from_json_schema(var_70)
    var_72 = module_1.Definitions()
    var_73 = '$ref'
    var_74 = '#/components/schemas/Person'
    var_75 = {var_73: var_74}
    var_76 = module_0.from_json_schema(var_75, var_72)
    var_77 = 'maxLength'
    var_78 = 10
    var_79 = {var_4: var_5, var_36: var_37, var_77: var_78}
    var_80 = module_0.from_json_schema(var_79)
    var_81 = var_80.schemas
    var_82 = len(var_81)
    assert var_82 == 2



# Parsed testcases at query #26
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
    var_23 = [var_5, var_8]
    var_24 = {var_4: var_23}
    var_25 = module_0.from_json_schema(var_24)
    var_26 = var_25.one_of
    var_27 = len(var_26)
    assert var_27 == 2
    var_28 = 'enum'
    var_29 = 'a'
    var_30 = 'b'
    var_31 = 'c'
    var_32 = [var_29, var_30, var_31]
    var_33 = {var_28: var_32}
    var_34 = module_0.from_json_schema(var_33)
    var_35 = 'const'
    var_36 = 'value'
    var_37 = {var_35: var_36}
    var_38 = module_0.from_json_schema(var_37)
    var_39 = 'allOf'
    var_40 = {var_4: var_5}
    var_41 = 'minLength'
    var_42 = 5
    var_43 = {var_41: var_42}
    var_44 = [var_40, var_43]
    var_45 = {var_39: var_44}
    var_46 = module_0.from_json_schema(var_45)
    var_47 = var_46.all_of
    var_48 = len(var_47)
    assert var_48 == 2
    var_49 = 'anyOf'
    var_50 = {var_4: var_5}
    var_51 = {var_4: var_8}
    var_52 = [var_50, var_51]
    var_53 = {var_49: var_52}
    var_54 = module_0.from_json_schema(var_53)
    var_55 = var_54.one_of
    var_56 = len(var_55)
    assert var_56 == 2
    var_57 = 'oneOf'
    var_58 = {var_4: var_5}
    var_59 = {var_4: var_8}
    var_60 = [var_58, var_59]
    var_61 = {var_57: var_60}
    var_62 = module_0.from_json_schema(var_61)
    var_63 = var_62.one_of
    var_64 = len(var_63)
    assert var_64 == 2
    var_65 = 'not'
    var_66 = {var_4: var_5}
    var_67 = {var_65: var_66}
    var_68 = module_0.from_json_schema(var_67)
    var_69 = var_68.not_
    var_70 = 'if'
    var_71 = 'then'
    var_72 = 'else'
    var_73 = {var_4: var_5}
    var_74 = {var_41: var_42}
    var_75 = {var_4: var_8}
    var_76 = {var_70: var_73, var_71: var_74, var_72: var_75}
    var_77 = module_0.from_json_schema(var_76)
    var_78 = module_1.Definitions()
    var_79 = '$ref'
    var_80 = '#/components/schemas/Test'
    var_81 = {var_79: var_80}
    var_82 = module_0.from_json_schema(var_81, var_78)
    var_83 = 'maxLength'
    var_84 = 'pattern'
    var_85 = 10
    var_86 = '^[a-z]+$'
    var_87 = {var_4: var_5, var_41: var_42, var_83: var_85, var_84: var_86}
    var_88 = module_0.from_json_schema(var_87)
    var_89 = var_88.all_of
    var_90 = len(var_89)
    assert var_90 == 4
    var_91 = 'components'
    var_92 = 'properties'
    var_93 = 'schemas'
    var_94 = 'Name'
    var_95 = 'Age'
    var_96 = {var_4: var_5}
    var_97 = {var_4: var_11}
    var_98 = {var_94: var_96, var_95: var_97}
    var_99 = {var_93: var_98}
    var_100 = 'name'
    var_101 = 'age'
    var_102 = '#/components/schemas/Name'
    var_103 = {var_79: var_102}
    var_104 = '#/components/schemas/Age'
    var_105 = {var_79: var_104}
    var_106 = {var_100: var_103, var_101: var_105}
    var_107 = {var_91: var_99, var_4: var_20, var_92: var_106}
    var_108 = module_0.from_json_schema(var_107)
    var_109 = var_108.properties
    var_110 = len(var_109)
    assert var_110 == 2



# Parsed testcases at query #27
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
    var_31 = 'value'
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
    var_42 = var_41.constraints
    var_43 = len(var_42)
    assert var_43 == 2
    var_44 = 'anyOf'
    var_45 = {var_4: var_5}
    var_46 = {var_4: var_8}
    var_47 = [var_45, var_46]
    var_48 = {var_44: var_47}
    var_49 = module_0.from_json_schema(var_48)
    var_50 = var_49.options
    var_51 = len(var_50)
    assert var_51 == 2
    var_52 = 'oneOf'
    var_53 = {var_4: var_5}
    var_54 = {var_4: var_8}
    var_55 = [var_53, var_54]
    var_56 = {var_52: var_55}
    var_57 = module_0.from_json_schema(var_56)
    var_58 = var_57.options
    var_59 = len(var_58)
    assert var_59 == 2
    var_60 = 'not'
    var_61 = {var_4: var_5}
    var_62 = {var_60: var_61}
    var_63 = module_0.from_json_schema(var_62)
    var_64 = var_63.schema
    var_65 = 'if'
    var_66 = 'then'
    var_67 = 'else'
    var_68 = {var_4: var_5}
    var_69 = {var_36: var_37}
    var_70 = {var_4: var_8}
    var_71 = {var_65: var_68, var_66: var_69, var_67: var_70}
    var_72 = module_0.from_json_schema(var_71)
    var_73 = var_72.if_schema
    var_74 = var_72.then_schema
    var_75 = var_72.else_schema
    var_76 = module_1.Definitions()
    var_77 = {var_4: var_5}
    var_78 = '$ref'
    var_79 = '#/components/schemas/Test'
    var_80 = {var_78: var_79}
    var_81 = module_0.from_json_schema(var_80, var_76)
    var_82 = 'maxLength'
    var_83 = 'pattern'
    var_84 = 10
    var_85 = '^[a-z]+$'
    var_86 = {var_4: var_5, var_36: var_37, var_82: var_84, var_83: var_85}
    var_87 = module_0.from_json_schema(var_86)
    var_88 = var_87.constraints
    var_89 = len(var_88)
    assert var_89 == 4



# Parsed testcases at query #28
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
    var_4 = 1
    var_5 = 10
    var_6 = '[a-z]+'
    var_7 = 'email'
    var_8 = module_0.String(max_length=var_5, min_length=var_4, pattern=var_6, format=var_7)
    var_9 = 'type'
    var_10 = 'minLength'
    var_11 = 'maxLength'
    var_12 = 'pattern'
    var_13 = 'format'
    var_14 = 'string'
    var_15 = {var_9: var_14, var_10: var_4, var_11: var_5, var_12: var_6, var_13: var_7}
    var_16 = module_1.to_json_schema(var_8)
    var_17 = 0
    var_18 = 100
    var_19 = True
    var_20 = True
    var_21 = module_0.Integer(minimum=var_17, maximum=var_18, exclusive_minimum=var_19, exclusive_maximum=var_20)
    var_22 = 'minimum'
    var_23 = 'maximum'
    var_24 = 'exclusiveMinimum'
    var_25 = 'exclusiveMaximum'
    var_26 = 'integer'
    var_27 = True
    var_28 = True
    var_29 = {var_9: var_26, var_22: var_17, var_23: var_18, var_24: var_27, var_25: var_28}
    var_30 = module_1.to_json_schema(var_21)
    var_31 = 0.1
    var_32 = module_0.Float(minimum=var_17, maximum=var_28, multiple_of=var_31)
    var_33 = 'multipleOf'
    var_34 = 'number'
    var_35 = {var_9: var_34, var_22: var_17, var_23: var_28, var_33: var_31}
    var_36 = module_1.to_json_schema(var_32)
    var_37 = module_0.Boolean()
    var_38 = 'boolean'
    var_39 = {var_9: var_38}
    var_40 = module_1.to_json_schema(var_37)
    var_41 = module_0.String()
    var_42 = True
    var_43 = module_0.Array(var_41, min_items=var_28, max_items=var_5, unique_items=var_42)
    var_44 = 'items'
    var_45 = 'minItems'
    var_46 = 'maxItems'
    var_47 = 'uniqueItems'
    var_48 = 'array'
    var_49 = {var_9: var_14}
    var_50 = True
    var_51 = {var_9: var_48, var_44: var_49, var_45: var_42, var_46: var_5, var_47: var_50}
    var_52 = module_1.to_json_schema(var_43)
    var_53 = 'name'
    var_54 = module_0.String()
    var_55 = {var_53: var_54}
    var_56 = [var_53]
    var_57 = module_0.Object(properties=var_55, min_properties=var_50, max_properties=var_5, required=var_56)
    var_58 = 'properties'
    var_59 = 'required'
    var_60 = 'minProperties'
    var_61 = 'maxProperties'
    var_62 = 'object'
    var_63 = {var_9: var_14}
    var_64 = {var_53: var_63}
    var_65 = [var_53]
    var_66 = {var_9: var_62, var_58: var_64, var_59: var_65, var_60: var_50, var_61: var_5}
    var_67 = module_1.to_json_schema(var_57)
    var_68 = 'a'
    var_69 = (var_68, var_68)
    var_70 = 'b'
    var_71 = (var_70, var_70)
    var_72 = [var_69, var_71]
    var_73 = module_0.Choice(choices=var_72)
    var_74 = 'enum'
    var_75 = [var_68, var_70]
    var_76 = {var_74: var_75}
    var_77 = module_1.to_json_schema(var_73)
    var_78 = 'fixed_value'
    var_79 = module_0.Const(var_78)
    var_80 = 'const'
    var_81 = {var_80: var_78}
    var_82 = module_1.to_json_schema(var_79)
    var_83 = module_0.String()
    var_84 = module_0.Integer()
    var_85 = [var_83, var_84]
    var_86 = module_0.Union(var_85)
    var_87 = 'anyOf'
    var_88 = {var_9: var_14}
    var_89 = {var_9: var_26}
    var_90 = [var_88, var_89]
    var_91 = {var_87: var_90}
    var_92 = module_1.to_json_schema(var_86)
    var_93 = module_0.String()
    var_94 = module_0.Integer()
    var_95 = [var_93, var_94]
    var_96 = module_2.OneOf(var_95)
    var_97 = 'oneOf'
    var_98 = {var_9: var_14}
    var_99 = {var_9: var_26}
    var_100 = [var_98, var_99]
    var_101 = {var_97: var_100}
    var_102 = module_1.to_json_schema(var_96)
    var_103 = module_0.String()
    var_104 = module_0.Integer()
    var_105 = [var_103, var_104]
    var_106 = module_2.AllOf(var_105)
    var_107 = 'allOf'
    var_108 = {var_9: var_14}
    var_109 = {var_9: var_26}
    var_110 = [var_108, var_109]
    var_111 = {var_107: var_110}
    var_112 = module_1.to_json_schema(var_106)
    var_113 = module_0.String()
    var_114 = module_0.Integer()
    var_115 = module_0.Boolean()
    var_116 = module_2.IfThenElse(var_113, var_114, var_115)
    var_117 = 'if'
    var_118 = 'then'
    var_119 = 'else'
    var_120 = {var_9: var_14}
    var_121 = {var_9: var_26}
    var_122 = {var_9: var_38}
    var_123 = {var_117: var_120, var_118: var_121, var_119: var_122}
    var_124 = module_1.to_json_schema(var_116)
    var_125 = module_0.String()
    var_126 = module_2.Not(var_125)
    var_127 = 'not'
    var_128 = {var_9: var_14}
    var_129 = {var_127: var_128}
    var_130 = module_1.to_json_schema(var_126)
    var_131 = 'Test'
    var_132 = module_0.String()
    var_133 = {var_131: var_132}
    var_134 = '$ref'
    var_135 = 'components'
    var_136 = '#/components/schemas/Test'
    var_137 = 'schemas'
    var_138 = {var_9: var_14}
    var_139 = {var_131: var_138}
    var_140 = {var_137: var_139}
    var_141 = {var_134: var_136, var_135: var_140}
    var_142 = module_0.String()
    var_143 = {var_53: var_142}
    var_144 = [var_53]
    var_145 = module_3.Schema(var_143)
    var_146 = {var_9: var_14}
    var_147 = {var_53: var_146}
    var_148 = [var_53]
    var_149 = {var_9: var_62, var_58: var_147, var_59: var_148}
    var_150 = module_1.to_json_schema(var_145)
    var_151 = True
    var_152 = module_0.String()
    var_153 = 'null'
    var_154 = [var_14, var_153]
    var_155 = {var_9: var_154}
    var_156 = module_1.to_json_schema(var_152)



# Parsed testcases at query #29
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
    var_8 = 0
    var_9 = 100
    var_10 = True
    var_11 = 2
    var_12 = 50
    var_13 = {var_0: var_7, var_1: var_8, var_2: var_9, var_3: var_10, var_4: var_10, var_5: var_11, var_6: var_12}
    var_14 = False
    var_15 = module_0.Definitions()
    var_16 = module_1.from_json_schema_type(var_13, var_7, var_14, var_15)
    var_17 = 'integer'
    var_18 = {var_0: var_17, var_1: var_14, var_2: var_9, var_3: var_10, var_4: var_10, var_5: var_11, var_6: var_12}
    var_19 = False
    var_20 = module_0.Definitions()
    var_21 = module_1.from_json_schema_type(var_18, var_17, var_19, var_20)
    var_22 = 'minLength'
    var_23 = 'maxLength'
    var_24 = 'format'
    var_25 = 'pattern'
    var_26 = 'string'
    var_27 = 5
    var_28 = 'email'
    var_29 = '^[a-zA-Z0-9]+$'
    var_30 = 'test'
    var_31 = {var_0: var_26, var_22: var_27, var_23: var_9, var_24: var_28, var_25: var_29, var_6: var_30}
    var_32 = False
    var_33 = module_0.Definitions()
    var_34 = module_1.from_json_schema_type(var_31, var_26, var_32, var_33)
    var_35 = 'boolean'
    var_36 = {var_0: var_35, var_6: var_10}
    var_37 = False
    var_38 = module_0.Definitions()
    var_39 = module_1.from_json_schema_type(var_36, var_35, var_37, var_38)
    var_40 = 'items'
    var_41 = 'minItems'
    var_42 = 'maxItems'
    var_43 = 'uniqueItems'
    var_44 = 'array'
    var_45 = {var_0: var_26}
    var_46 = 10
    var_47 = [var_30]
    var_48 = {var_0: var_44, var_40: var_45, var_41: var_10, var_42: var_46, var_43: var_10, var_6: var_47}
    var_49 = False
    var_50 = module_0.Definitions()
    var_51 = module_1.from_json_schema_type(var_48, var_44, var_49, var_50)
    var_52 = var_51.items
    var_53 = 'properties'
    var_54 = 'required'
    var_55 = 'minProperties'
    var_56 = 'maxProperties'
    var_57 = 'object'
    var_58 = 'name'
    var_59 = 'age'
    var_60 = {var_0: var_26}
    var_61 = {var_0: var_17}
    var_62 = {var_58: var_60, var_59: var_61}
    var_63 = [var_58]
    var_64 = 30
    var_65 = {var_58: var_30, var_59: var_64}
    var_66 = {var_0: var_57, var_53: var_62, var_54: var_63, var_55: var_10, var_56: var_46, var_6: var_65}
    var_67 = False
    var_68 = module_0.Definitions()
    var_69 = module_1.from_json_schema_type(var_66, var_57, var_67, var_68)
    var_70 = var_69.properties[var_58]
    var_71 = var_69.properties[var_59]



# Parsed testcases at query #30
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
    var_24 = 2
    var_25 = 3
    var_26 = [var_0, var_24, var_25]
    var_27 = {var_23: var_26}
    var_28 = module_0.from_json_schema(var_27)
    var_29 = 'const'
    var_30 = 'test'
    var_31 = {var_29: var_30}
    var_32 = module_0.from_json_schema(var_31)
    var_33 = 'allOf'
    var_34 = {var_4: var_5}
    var_35 = 'minLength'
    var_36 = {var_35: var_0}
    var_37 = [var_34, var_36]
    var_38 = {var_33: var_37}
    var_39 = module_0.from_json_schema(var_38)
    var_40 = var_39.schemas
    var_41 = len(var_40)
    assert var_41 == 2
    var_42 = 'anyOf'
    var_43 = {var_4: var_5}
    var_44 = {var_4: var_8}
    var_45 = [var_43, var_44]
    var_46 = {var_42: var_45}
    var_47 = module_0.from_json_schema(var_46)
    var_48 = var_47.schemas
    var_49 = len(var_48)
    assert var_49 == 2
    var_50 = 'oneOf'
    var_51 = {var_4: var_5}
    var_52 = {var_4: var_8}
    var_53 = [var_51, var_52]
    var_54 = {var_50: var_53}
    var_55 = module_0.from_json_schema(var_54)
    var_56 = var_55.schemas
    var_57 = len(var_56)
    assert var_57 == 2
    var_58 = 'not'
    var_59 = {var_4: var_5}
    var_60 = {var_58: var_59}
    var_61 = module_0.from_json_schema(var_60)
    var_62 = var_61.schema
    var_63 = 'if'
    var_64 = 'then'
    var_65 = 'else'
    var_66 = {var_4: var_5}
    var_67 = {var_35: var_0}
    var_68 = {var_4: var_8}
    var_69 = {var_63: var_66, var_64: var_67, var_65: var_68}
    var_70 = module_0.from_json_schema(var_69)
    var_71 = var_70.if_schema
    var_72 = var_70.then_schema
    var_73 = var_70.else_schema
    var_74 = module_1.Definitions()
    var_75 = '$ref'
    var_76 = '#/components/schemas/Test'
    var_77 = {var_75: var_76}
    var_78 = module_0.from_json_schema(var_77, var_74)
    var_79 = 'maxLength'
    var_80 = 10
    var_81 = {var_4: var_5, var_35: var_0, var_79: var_80}
    var_82 = module_0.from_json_schema(var_81)
    var_83 = var_82.schemas
    var_84 = len(var_83)
    assert var_84 == 3
    var_85 = {}
    var_86 = module_0.from_json_schema(var_85)



# Parsed testcases at query #31
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
    var_42 = var_41.schemas
    var_43 = len(var_42)
    assert var_43 == 2
    var_44 = 'anyOf'
    var_45 = {var_4: var_5}
    var_46 = {var_4: var_8}
    var_47 = [var_45, var_46]
    var_48 = {var_44: var_47}
    var_49 = module_0.from_json_schema(var_48)
    var_50 = var_49.schemas
    var_51 = len(var_50)
    assert var_51 == 2
    var_52 = 'oneOf'
    var_53 = {var_4: var_5}
    var_54 = {var_4: var_8}
    var_55 = [var_53, var_54]
    var_56 = {var_52: var_55}
    var_57 = module_0.from_json_schema(var_56)
    var_58 = var_57.schemas
    var_59 = len(var_58)
    assert var_59 == 2
    var_60 = 'not'
    var_61 = {var_4: var_5}
    var_62 = {var_60: var_61}
    var_63 = module_0.from_json_schema(var_62)
    var_64 = var_63.schema
    var_65 = 'if'
    var_66 = 'then'
    var_67 = 'else'
    var_68 = {var_4: var_5}
    var_69 = {var_36: var_37}
    var_70 = {var_4: var_8}
    var_71 = {var_65: var_68, var_66: var_69, var_67: var_70}
    var_72 = module_0.from_json_schema(var_71)
    var_73 = var_72.if_schema
    var_74 = var_72.then_schema
    var_75 = var_72.else_schema
    var_76 = module_1.Definitions()
    var_77 = '$ref'
    var_78 = '#/components/schemas/Test'
    var_79 = {var_77: var_78}
    var_80 = module_0.from_json_schema(var_79, var_76)
    var_81 = 'maxLength'
    var_82 = 10
    var_83 = {var_4: var_5, var_36: var_37, var_81: var_82}
    var_84 = module_0.from_json_schema(var_83)
    var_85 = var_84.schemas
    var_86 = len(var_85)
    assert var_86 == 2
    var_87 = {}
    var_88 = module_0.from_json_schema(var_87)



# Parsed testcases at query #32
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
    var_6 = '[a-z]+'
    var_7 = 'email'
    var_8 = module_0.String(max_length=var_5, min_length=var_4, pattern=var_6, format=var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = False
    var_11 = 100
    var_12 = 2
    var_13 = module_0.Integer(minimum=var_10, maximum=var_11, exclusive_minimum=var_4, multiple_of=var_12)
    var_14 = module_1.to_json_schema(var_13)
    var_15 = module_0.Float(minimum=var_10, maximum=var_4)
    var_16 = module_1.to_json_schema(var_15)
    var_17 = module_0.Boolean()
    var_18 = module_1.to_json_schema(var_17)
    var_19 = module_0.String()
    var_20 = module_0.Array(var_19, var_10, var_4, var_5)
    var_21 = module_1.to_json_schema(var_20)
    var_22 = 'name'
    var_23 = module_0.String()
    var_24 = {var_22: var_23}
    var_25 = [var_22]
    var_26 = module_0.Object(properties=var_24, additional_properties=var_10, required=var_25)
    var_27 = module_1.to_json_schema(var_26)
    var_28 = 'a'
    var_29 = (var_28, var_28)
    var_30 = 'b'
    var_31 = (var_30, var_30)
    var_32 = [var_29, var_31]
    var_33 = module_0.Choice(choices=var_32)
    var_34 = module_1.to_json_schema(var_33)
    var_35 = 'fixed_value'
    var_36 = module_0.Const(var_35)
    var_37 = module_1.to_json_schema(var_36)
    var_38 = module_0.String()
    var_39 = module_0.Integer()
    var_40 = [var_38, var_39]
    var_41 = module_0.Union(var_40)
    var_42 = module_1.to_json_schema(var_41)
    var_43 = module_0.String()
    var_44 = module_0.Integer()
    var_45 = [var_43, var_44]
    var_46 = module_2.OneOf(var_45)
    var_47 = module_1.to_json_schema(var_46)
    var_48 = module_0.String()
    var_49 = 'test'
    var_50 = module_0.Const(var_49)
    var_51 = [var_48, var_50]
    var_52 = module_2.AllOf(var_51)
    var_53 = module_1.to_json_schema(var_52)
    var_54 = module_0.String()
    var_55 = module_0.Integer()
    var_56 = module_0.Boolean()
    var_57 = module_2.IfThenElse(var_54, var_55, var_56)
    var_58 = module_1.to_json_schema(var_57)
    var_59 = module_0.String()
    var_60 = module_2.Not(var_59)
    var_61 = module_1.to_json_schema(var_60)
    var_62 = 'Test'
    var_63 = module_0.String()
    var_64 = {var_62: var_63}
    var_65 = module_1.to_json_schema(var_60)
    var_66 = module_0.String()
    var_67 = {var_22: var_66}
    var_68 = [var_22]
    var_69 = module_3.Schema(var_67)
    var_70 = module_1.to_json_schema(var_69)



# Parsed testcases at query #33
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
    var_8 = 0
    var_9 = 100
    var_10 = 2
    var_11 = 50
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
    var_26 = 5
    var_27 = 'email'
    var_28 = '^[a-zA-Z0-9]+$'
    var_29 = 'test'
    var_30 = {var_0: var_25, var_21: var_26, var_22: var_9, var_23: var_27, var_24: var_28, var_6: var_29}
    var_31 = False
    var_32 = module_0.Definitions()
    var_33 = module_1.from_json_schema_type(var_30, var_25, var_31, var_32)
    var_34 = 'boolean'
    var_35 = True
    var_36 = {var_0: var_34, var_6: var_35}
    var_37 = False
    var_38 = module_0.Definitions()
    var_39 = module_1.from_json_schema_type(var_36, var_34, var_37, var_38)
    var_40 = 'items'
    var_41 = 'additionalItems'
    var_42 = 'minItems'
    var_43 = 'maxItems'
    var_44 = 'uniqueItems'
    var_45 = 'array'
    var_46 = {var_0: var_25}
    var_47 = False
    var_48 = 10
    var_49 = [var_29]
    var_50 = {var_0: var_45, var_40: var_46, var_41: var_47, var_42: var_35, var_43: var_48, var_44: var_35, var_6: var_49}
    var_51 = False
    var_52 = module_0.Definitions()
    var_53 = module_1.from_json_schema_type(var_50, var_45, var_51, var_52)
    var_54 = var_53.items
    var_55 = 'properties'
    var_56 = 'patternProperties'
    var_57 = 'additionalProperties'
    var_58 = 'propertyNames'
    var_59 = 'minProperties'
    var_60 = 'maxProperties'
    var_61 = 'required'
    var_62 = 'object'
    var_63 = 'name'
    var_64 = 'age'
    var_65 = {var_0: var_25}
    var_66 = {var_0: var_16}
    var_67 = {var_63: var_65, var_64: var_66}
    var_68 = '^S_'
    var_69 = '^I_'
    var_70 = {var_0: var_25}
    var_71 = {var_0: var_16}
    var_72 = {var_68: var_70, var_69: var_71}
    var_73 = False
    var_74 = {var_0: var_25}
    var_75 = [var_63]
    var_76 = 30
    var_77 = {var_63: var_29, var_64: var_76}
    var_78 = {var_0: var_62, var_55: var_67, var_56: var_72, var_57: var_73, var_58: var_74, var_59: var_35, var_60: var_48, var_61: var_75, var_6: var_77}
    var_79 = False
    var_80 = module_0.Definitions()
    var_81 = module_1.from_json_schema_type(var_78, var_62, var_79, var_80)
    var_82 = var_81.properties[var_63]
    var_83 = var_81.properties[var_64]
    var_84 = var_81.pattern_properties[var_68]
    var_85 = var_81.pattern_properties[var_69]
    var_86 = var_81.property_names



# Parsed testcases at query #34
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
    var_6 = '[a-z]+'
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
    var_19 = False
    var_20 = 100
    var_21 = 2
    var_22 = module_0.Integer(minimum=var_19, maximum=var_20, multiple_of=var_21)
    var_23 = 'minimum'
    var_24 = 'maximum'
    var_25 = 'multipleOf'
    var_26 = 'integer'
    var_27 = {var_9: var_26, var_23: var_19, var_24: var_20, var_25: var_21}
    var_28 = module_1.to_json_schema(var_22)
    var_29 = module_0.Float(exclusive_minimum=var_19, exclusive_maximum=var_4)
    var_30 = 'exclusiveMinimum'
    var_31 = 'exclusiveMaximum'
    var_32 = 'number'
    var_33 = [var_32, var_15]
    var_34 = {var_9: var_33, var_30: var_19, var_31: var_4}
    var_35 = module_1.to_json_schema(var_29)
    var_36 = module_0.Boolean()
    var_37 = 'boolean'
    var_38 = {var_9: var_37}
    var_39 = module_1.to_json_schema(var_36)
    var_40 = module_0.String()
    var_41 = 5
    var_42 = module_0.Array(var_40, var_19, var_4, var_41, unique_items=var_4)
    var_43 = 'items'
    var_44 = 'additionalItems'
    var_45 = 'minItems'
    var_46 = 'maxItems'
    var_47 = 'uniqueItems'
    var_48 = 'array'
    var_49 = [var_48, var_15]
    var_50 = {var_9: var_14}
    var_51 = {var_9: var_49, var_43: var_50, var_44: var_19, var_45: var_4, var_46: var_41, var_47: var_4}
    var_52 = module_1.to_json_schema(var_42)
    var_53 = 'name'
    var_54 = module_0.String()
    var_55 = {var_53: var_54}
    var_56 = [var_53]
    var_57 = module_0.Object(properties=var_55, additional_properties=var_19, required=var_56)
    var_58 = 'properties'
    var_59 = 'additionalProperties'
    var_60 = 'required'
    var_61 = 'object'
    var_62 = {var_9: var_14}
    var_63 = {var_53: var_62}
    var_64 = [var_53]
    var_65 = {var_9: var_61, var_58: var_63, var_59: var_19, var_60: var_64}
    var_66 = module_1.to_json_schema(var_57)
    var_67 = 'a'
    var_68 = (var_67, var_67)
    var_69 = 'b'
    var_70 = (var_69, var_69)
    var_71 = [var_68, var_70]
    var_72 = module_0.Choice(choices=var_71)
    var_73 = 'enum'
    var_74 = [var_67, var_69]
    var_75 = {var_73: var_74}
    var_76 = module_1.to_json_schema(var_72)
    var_77 = 'value'
    var_78 = module_0.Const(var_77)
    var_79 = 'const'
    var_80 = {var_79: var_77}
    var_81 = module_1.to_json_schema(var_78)
    var_82 = module_0.String()
    var_83 = module_0.Integer()
    var_84 = [var_82, var_83]
    var_85 = module_0.Union(var_84)
    var_86 = 'anyOf'
    var_87 = {var_9: var_14}
    var_88 = {var_9: var_26}
    var_89 = [var_87, var_88]
    var_90 = {var_86: var_89}
    var_91 = module_1.to_json_schema(var_85)
    var_92 = module_0.String()
    var_93 = 'test'
    var_94 = module_0.Const(var_93)
    var_95 = [var_92, var_94]
    var_96 = module_2.AllOf(var_95)
    var_97 = 'allOf'
    var_98 = {var_9: var_14}
    var_99 = {var_79: var_93}
    var_100 = [var_98, var_99]
    var_101 = {var_97: var_100}
    var_102 = module_1.to_json_schema(var_96)
    var_103 = module_3.Definitions()
    var_104 = module_0.String()
    var_105 = module_3.Reference(var_93, var_103)
    var_106 = '$ref'
    var_107 = 'components'
    var_108 = '#/components/schemas/test'
    var_109 = 'schemas'
    var_110 = {var_9: var_14}
    var_111 = {var_93: var_110}
    var_112 = {var_109: var_111}
    var_113 = {var_106: var_108, var_107: var_112}
    var_114 = module_1.to_json_schema(var_105)
    var_115 = module_0.String()
    var_116 = module_0.Integer()
    var_117 = module_0.Boolean()
    var_118 = module_2.IfThenElse(var_115, var_116, var_117)
    var_119 = 'if'
    var_120 = 'then'
    var_121 = 'else'
    var_122 = {var_9: var_14}
    var_123 = {var_9: var_26}
    var_124 = {var_9: var_37}
    var_125 = {var_119: var_122, var_120: var_123, var_121: var_124}
    var_126 = module_1.to_json_schema(var_118)
    var_127 = module_0.String()
    var_128 = module_2.Not(var_127)
    var_129 = 'not'
    var_130 = {var_9: var_14}
    var_131 = {var_129: var_130}
    var_132 = module_1.to_json_schema(var_128)



# Parsed testcases at query #35
#--------------------------


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
    var_8 = module_0.Definitions()
    var_9 = module_1.from_json_schema_type(var_6, var_3, var_7, var_8)
    var_10 = 'integer'
    var_11 = {var_0: var_10, var_1: var_7, var_2: var_5}
    var_12 = False
    var_13 = module_0.Definitions()
    var_14 = module_1.from_json_schema_type(var_11, var_10, var_12, var_13)
    var_15 = 'minLength'
    var_16 = 'maxLength'
    var_17 = 'string'
    var_18 = 1
    var_19 = {var_0: var_17, var_15: var_18, var_16: var_5}
    var_20 = False
    var_21 = module_0.Definitions()
    var_22 = module_1.from_json_schema_type(var_19, var_17, var_20, var_21)
    var_23 = 'boolean'
    var_24 = {var_0: var_23}
    var_25 = False
    var_26 = module_0.Definitions()
    var_27 = module_1.from_json_schema_type(var_24, var_23, var_25, var_26)
    var_28 = 'items'
    var_29 = 'array'
    var_30 = {var_0: var_17}
    var_31 = {var_0: var_29, var_28: var_30}
    var_32 = False
    var_33 = module_0.Definitions()
    var_34 = module_1.from_json_schema_type(var_31, var_29, var_32, var_33)
    var_35 = var_34.items
    var_36 = 'properties'
    var_37 = 'object'
    var_38 = 'name'
    var_39 = {var_0: var_17}
    var_40 = {var_38: var_39}
    var_41 = {var_0: var_37, var_36: var_40}
    var_42 = False
    var_43 = module_0.Definitions()
    var_44 = module_1.from_json_schema_type(var_41, var_37, var_42, var_43)
    var_45 = var_44.properties[var_38]
    var_46 = 'type'
    var_47 = 'invalid'
    var_48 = {var_46: var_47}
    var_49 = False
    var_50 = module_0.Definitions()
    var_51 = module_1.from_json_schema_type(var_48, var_47, var_49, var_50)



