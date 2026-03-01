####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = '#/components/schemas/Test'
    var_3 = {var_1: var_2}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)



# Parsed testcases at query #2
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



# Parsed testcases at query #3
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
    var_64 = var_63.negated
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
    var_83 = 10
    var_84 = {var_4: var_5, var_36: var_37, var_82: var_83}
    var_85 = module_0.from_json_schema(var_84)
    var_86 = var_85.constraints
    var_87 = len(var_86)
    assert var_87 == 3
    var_88 = {}
    var_89 = module_0.from_json_schema(var_88)



# Parsed testcases at query #4
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = '#/components/schemas/TestSchema'
    var_3 = {var_1: var_2}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)



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
    var_5 = 2
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
    var_19 = 'properties'
    var_20 = 'object'
    var_21 = 'name'
    var_22 = {var_2: var_4}
    var_23 = {var_21: var_22}
    var_24 = {var_2: var_20, var_19: var_23}
    var_25 = 'age'
    var_26 = 'integer'
    var_27 = {var_2: var_26}
    var_28 = {var_25: var_27}
    var_29 = {var_2: var_20, var_19: var_28}
    var_30 = [var_24, var_29]
    var_31 = {var_1: var_30}
    var_32 = module_1.all_of_from_json_schema(var_31, var_0)
    var_33 = var_32.all_of
    var_34 = len(var_33)
    assert var_34 == 2
    var_35 = var_32.all_of[var_15]
    var_36 = var_32.all_of[var_17]
    var_37 = 'default'
    var_38 = {var_2: var_4}
    var_39 = [var_38]
    var_40 = 'test'
    var_41 = {var_1: var_39, var_37: var_40}
    var_42 = module_1.all_of_from_json_schema(var_41, var_0)
    var_43 = '$ref'
    var_44 = '#/components/schemas/Test'
    var_45 = {var_43: var_44}
    var_46 = {var_2: var_4, var_3: var_17}
    var_47 = [var_45, var_46]
    var_48 = {var_1: var_47}
    var_49 = module_1.all_of_from_json_schema(var_48, var_0)
    var_50 = var_49.all_of
    var_51 = len(var_50)
    assert var_51 == 2
    var_52 = var_49.all_of[var_15]
    var_53 = var_49.all_of[var_17]



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
    var_24 = {var_1: var_23}
    var_25 = module_1.if_then_else_from_json_schema(var_24, var_0)
    var_26 = var_25.if_clause
    var_27 = 'default'
    var_28 = {var_4: var_5}
    var_29 = {var_4: var_5, var_7: var_8}
    var_30 = {var_4: var_10}
    var_31 = 'default_value'
    var_32 = {var_1: var_28, var_2: var_29, var_3: var_30, var_27: var_31}
    var_33 = module_1.if_then_else_from_json_schema(var_32, var_0)



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
    var_55 = module_0.Object(properties=var_53, required=var_54)
    var_56 = 'properties'
    var_57 = 'required'
    var_58 = 'object'
    var_59 = {var_9: var_14}
    var_60 = {var_51: var_59}
    var_61 = [var_51]
    var_62 = {var_9: var_58, var_56: var_60, var_57: var_61}
    var_63 = module_1.to_json_schema(var_55)
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
    var_74 = 'fixed'
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
    var_85 = {var_9: var_24}
    var_86 = [var_84, var_85]
    var_87 = {var_83: var_86}
    var_88 = module_1.to_json_schema(var_82)
    var_89 = module_0.String()
    var_90 = 'test'
    var_91 = module_0.Const(var_90)
    var_92 = [var_89, var_91]
    var_93 = module_2.AllOf(var_92)
    var_94 = 'allOf'
    var_95 = {var_9: var_14}
    var_96 = {var_76: var_90}
    var_97 = [var_95, var_96]
    var_98 = {var_94: var_97}
    var_99 = module_1.to_json_schema(var_93)
    var_100 = module_0.String()
    var_101 = module_0.Integer()
    var_102 = [var_100, var_101]
    var_103 = module_2.OneOf(var_102)
    var_104 = 'oneOf'
    var_105 = {var_9: var_14}
    var_106 = {var_9: var_24}
    var_107 = [var_105, var_106]
    var_108 = {var_104: var_107}
    var_109 = module_1.to_json_schema(var_103)
    var_110 = module_0.String()
    var_111 = module_2.Not(var_110)
    var_112 = 'not'
    var_113 = {var_9: var_14}
    var_114 = {var_112: var_113}
    var_115 = module_1.to_json_schema(var_111)
    var_116 = module_0.String()
    var_117 = module_0.Integer()
    var_118 = module_2.IfThenElse(var_116, var_117)
    var_119 = 'if'
    var_120 = 'then'
    var_121 = {var_9: var_14}
    var_122 = {var_9: var_24}
    var_123 = {var_119: var_121, var_120: var_122}
    var_124 = module_1.to_json_schema(var_118)
    var_125 = 'Test'
    var_126 = module_0.String()
    var_127 = {var_125: var_126}
    var_128 = '$ref'
    var_129 = 'components'
    var_130 = '#/components/schemas/Test'
    var_131 = 'schemas'
    var_132 = {var_9: var_14}
    var_133 = {var_125: var_132}
    var_134 = {var_131: var_133}
    var_135 = {var_128: var_130, var_129: var_134}
    var_136 = module_1.to_json_schema(var_118)
    var_137 = module_0.String()
    var_138 = {var_51: var_137}
    var_139 = [var_51]
    var_140 = module_3.Schema(var_138)
    var_141 = {var_9: var_14}
    var_142 = {var_51: var_141}
    var_143 = [var_51]
    var_144 = {var_9: var_58, var_56: var_142, var_57: var_143}
    var_145 = module_1.to_json_schema(var_140)



# Parsed testcases at query #8
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = '#/test'
    var_3 = {var_1: var_2}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)



# Parsed testcases at query #9
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
    var_16 = 'nullable'
    var_17 = True
    var_18 = {var_0: var_1, var_16: var_17}
    var_19 = module_0.Definitions()
    var_20 = module_1.type_from_json_schema(var_18, var_19)
    var_21 = [var_1, var_5]
    var_22 = True
    var_23 = {var_0: var_21, var_16: var_22}
    var_24 = module_0.Definitions()
    var_25 = module_1.type_from_json_schema(var_23, var_24)
    var_26 = True
    var_27 = {var_16: var_26}
    var_28 = module_0.Definitions()
    var_29 = module_1.type_from_json_schema(var_27, var_28)
    var_30 = {}
    var_31 = module_0.Definitions()
    var_32 = module_1.type_from_json_schema(var_30, var_31)
    var_33 = 'minLength'
    var_34 = 5
    var_35 = {var_0: var_1, var_33: var_34}
    var_36 = module_0.Definitions()
    var_37 = module_1.type_from_json_schema(var_35, var_36)
    var_38 = 'items'
    var_39 = 'array'
    var_40 = {var_0: var_1}
    var_41 = {var_0: var_39, var_38: var_40}
    var_42 = module_0.Definitions()
    var_43 = module_1.type_from_json_schema(var_41, var_42)
    var_44 = var_43.items
    var_45 = 'properties'
    var_46 = 'object'
    var_47 = 'name'
    var_48 = {var_0: var_1}
    var_49 = {var_47: var_48}
    var_50 = {var_0: var_46, var_45: var_49}
    var_51 = module_0.Definitions()
    var_52 = module_1.type_from_json_schema(var_50, var_51)
    var_53 = var_52.properties[var_47]



# Parsed testcases at query #10
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = '#/components/schemas/Test'
    var_3 = {var_1: var_2}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)



# Parsed testcases at query #11
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
    var_90 = {var_0: var_25}
    var_91 = module_0.Definitions()
    var_92 = module_1.from_json_schema_type(var_90, var_25, var_35, var_91)



# Parsed testcases at query #12
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = '#/components/schemas/Test'
    var_3 = {var_1: var_2}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)



# Parsed testcases at query #13
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = '#/components/schemas/TestSchema'
    var_3 = {var_1: var_2}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)



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
    var_41 = 'additionalItems'
    var_42 = 'minItems'
    var_43 = 'maxItems'
    var_44 = 'uniqueItems'
    var_45 = 'array'
    var_46 = {var_0: var_26}
    var_47 = False
    var_48 = 10
    var_49 = [var_30]
    var_50 = {var_0: var_45, var_40: var_46, var_41: var_47, var_42: var_10, var_43: var_48, var_44: var_10, var_6: var_49}
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
    var_65 = {var_0: var_26}
    var_66 = {var_0: var_17}
    var_67 = {var_63: var_65, var_64: var_66}
    var_68 = '^S_'
    var_69 = '^I_'
    var_70 = {var_0: var_26}
    var_71 = {var_0: var_17}
    var_72 = {var_68: var_70, var_69: var_71}
    var_73 = False
    var_74 = {var_0: var_26}
    var_75 = [var_63]
    var_76 = {var_63: var_30}
    var_77 = {var_0: var_62, var_55: var_67, var_56: var_72, var_57: var_73, var_58: var_74, var_59: var_10, var_60: var_48, var_61: var_75, var_6: var_76}
    var_78 = False
    var_79 = module_0.Definitions()
    var_80 = module_1.from_json_schema_type(var_77, var_62, var_78, var_79)
    var_81 = var_80.properties[var_63]
    var_82 = var_80.properties[var_64]
    var_83 = var_80.pattern_properties[var_68]
    var_84 = var_80.pattern_properties[var_69]
    var_85 = var_80.property_names
    var_86 = {var_0: var_26}
    var_87 = module_0.Definitions()
    var_88 = module_1.from_json_schema_type(var_86, var_26, var_10, var_87)



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
    var_18 = 5
    var_19 = 10
    var_20 = {var_0: var_17, var_15: var_18, var_16: var_19}
    var_21 = False
    var_22 = module_0.Definitions()
    var_23 = module_1.from_json_schema_type(var_20, var_17, var_21, var_22)
    var_24 = 'boolean'
    var_25 = {var_0: var_24}
    var_26 = False
    var_27 = module_0.Definitions()
    var_28 = module_1.from_json_schema_type(var_25, var_24, var_26, var_27)
    var_29 = 'items'
    var_30 = 'minItems'
    var_31 = 'maxItems'
    var_32 = 'array'
    var_33 = {var_0: var_17}
    var_34 = 1
    var_35 = {var_0: var_32, var_29: var_33, var_30: var_34, var_31: var_18}
    var_36 = False
    var_37 = module_0.Definitions()
    var_38 = module_1.from_json_schema_type(var_35, var_32, var_36, var_37)
    var_39 = var_38.items
    var_40 = 'properties'
    var_41 = 'required'
    var_42 = 'object'
    var_43 = 'name'
    var_44 = {var_0: var_17}
    var_45 = {var_43: var_44}
    var_46 = [var_43]
    var_47 = {var_0: var_42, var_40: var_45, var_41: var_46}
    var_48 = False
    var_49 = module_0.Definitions()
    var_50 = module_1.from_json_schema_type(var_47, var_42, var_48, var_49)
    var_51 = var_50.properties[var_43]
    var_52 = {var_0: var_17}
    var_53 = True
    var_54 = module_0.Definitions()
    var_55 = module_1.from_json_schema_type(var_52, var_17, var_53, var_54)



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
    var_46 = 'age'
    var_47 = module_0.String()
    var_48 = module_0.Integer()
    var_49 = {var_45: var_47, var_46: var_48}
    var_50 = module_0.Object(properties=var_49)
    var_51 = 'properties'
    var_52 = 'object'
    var_53 = {var_9: var_14}
    var_54 = {var_9: var_22}
    var_55 = {var_45: var_53, var_46: var_54}
    var_56 = {var_9: var_52, var_51: var_55}
    var_57 = module_1.to_json_schema(var_50)
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
    var_68 = 'fixed_value'
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
    var_94 = module_0.String()
    var_95 = module_0.Integer()
    var_96 = [var_94, var_95]
    var_97 = module_2.OneOf(var_96)
    var_98 = 'oneOf'
    var_99 = {var_9: var_14}
    var_100 = {var_9: var_22}
    var_101 = [var_99, var_100]
    var_102 = {var_98: var_101}
    var_103 = module_1.to_json_schema(var_97)
    var_104 = module_0.String()
    var_105 = module_2.Not(var_104)
    var_106 = 'not'
    var_107 = {var_9: var_14}
    var_108 = {var_106: var_107}
    var_109 = module_1.to_json_schema(var_105)
    var_110 = module_0.String()
    var_111 = module_0.Integer()
    var_112 = module_2.IfThenElse(var_110, var_111)
    var_113 = 'if'
    var_114 = 'then'
    var_115 = {var_9: var_14}
    var_116 = {var_9: var_22}
    var_117 = {var_113: var_115, var_114: var_116}
    var_118 = module_1.to_json_schema(var_112)
    var_119 = module_3.Definitions()
    var_120 = module_3.Reference(var_84, var_119)
    var_121 = '$ref'
    var_122 = '#/components/schemas/test'
    var_123 = {var_121: var_122}
    var_124 = module_1.to_json_schema(var_120)
    var_125 = module_0.String()
    var_126 = {var_45: var_125}
    var_127 = module_3.Schema(var_126)
    var_128 = {var_9: var_14}
    var_129 = {var_45: var_128}
    var_130 = {var_9: var_52, var_51: var_129}
    var_131 = module_1.to_json_schema(var_127)



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
    var_96 = module_3.Reference(var_94)
    var_97 = '$ref'
    var_98 = 'components'
    var_99 = '#/components/schemas/Test'
    var_100 = 'schemas'
    var_101 = {var_9: var_14}
    var_102 = {var_94: var_101}
    var_103 = {var_100: var_102}
    var_104 = {var_97: var_99, var_98: var_103}
    var_105 = module_1.to_json_schema(var_96)
    var_106 = module_0.String()
    var_107 = {var_45: var_106}
    var_108 = module_3.Schema(var_107)
    var_109 = {var_9: var_14}
    var_110 = {var_45: var_109}
    var_111 = {var_9: var_52, var_50: var_110}
    var_112 = module_1.to_json_schema(var_108)
    var_113 = module_0.String()
    var_114 = {var_94: var_113}
    var_115 = {var_9: var_14}
    var_116 = {var_94: var_115}
    var_117 = {var_100: var_116}
    var_118 = {var_98: var_117}



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
    var_22 = module_0.Integer(minimum=var_19, maximum=var_20, exclusive_minimum=var_4, multiple_of=var_21)
    var_23 = 'minimum'
    var_24 = 'maximum'
    var_25 = 'exclusiveMinimum'
    var_26 = 'multipleOf'
    var_27 = 'integer'
    var_28 = {var_9: var_27, var_23: var_19, var_24: var_20, var_25: var_4, var_26: var_21}
    var_29 = module_1.to_json_schema(var_22)
    var_30 = module_0.Float(minimum=var_19, maximum=var_4)
    var_31 = 'number'
    var_32 = [var_31, var_15]
    var_33 = {var_9: var_32, var_23: var_19, var_24: var_4}
    var_34 = module_1.to_json_schema(var_30)
    var_35 = module_0.Boolean()
    var_36 = 'boolean'
    var_37 = {var_9: var_36}
    var_38 = module_1.to_json_schema(var_35)
    var_39 = 5
    var_40 = module_0.String()
    var_41 = module_0.Array(var_40, var_19, var_4, var_39, unique_items=var_4)
    var_42 = 'minItems'
    var_43 = 'maxItems'
    var_44 = 'items'
    var_45 = 'additionalItems'
    var_46 = 'uniqueItems'
    var_47 = 'array'
    var_48 = [var_47, var_15]
    var_49 = {var_9: var_14}
    var_50 = {var_9: var_48, var_42: var_4, var_43: var_39, var_44: var_49, var_45: var_19, var_46: var_4}
    var_51 = module_1.to_json_schema(var_41)
    var_52 = 'name'
    var_53 = module_0.String()
    var_54 = {var_52: var_53}
    var_55 = '^S_'
    var_56 = module_0.String()
    var_57 = {var_55: var_56}
    var_58 = '[A-Z]+'
    var_59 = module_0.String(pattern=var_58)
    var_60 = [var_52]
    var_61 = module_0.Object(properties=var_54, pattern_properties=var_57, additional_properties=var_19, property_names=var_59, min_properties=var_4, max_properties=var_39, required=var_60)
    var_62 = 'properties'
    var_63 = 'patternProperties'
    var_64 = 'additionalProperties'
    var_65 = 'propertyNames'
    var_66 = 'minProperties'
    var_67 = 'maxProperties'
    var_68 = 'required'
    var_69 = 'object'
    var_70 = {var_9: var_14}
    var_71 = {var_52: var_70}
    var_72 = {var_9: var_14}
    var_73 = {var_55: var_72}
    var_74 = {var_9: var_14, var_12: var_58}
    var_75 = [var_52]
    var_76 = {var_9: var_69, var_62: var_71, var_63: var_73, var_64: var_19, var_65: var_74, var_66: var_4, var_67: var_39, var_68: var_75}
    var_77 = module_1.to_json_schema(var_61)
    var_78 = 'a'
    var_79 = (var_78, var_78)
    var_80 = 'b'
    var_81 = (var_80, var_80)
    var_82 = [var_79, var_81]
    var_83 = module_0.Choice(choices=var_82)
    var_84 = 'enum'
    var_85 = 'default'
    var_86 = [var_78, var_80]
    var_87 = {var_84: var_86, var_85: var_78}
    var_88 = module_1.to_json_schema(var_83)
    var_89 = 'fixed'
    var_90 = module_0.Const(var_89)
    var_91 = 'const'
    var_92 = {var_91: var_89, var_85: var_89}
    var_93 = module_1.to_json_schema(var_90)
    var_94 = module_0.String()
    var_95 = module_0.Integer()
    var_96 = [var_94, var_95]
    var_97 = module_0.Union(var_96)
    var_98 = 'anyOf'
    var_99 = {var_9: var_14}
    var_100 = {var_9: var_27}
    var_101 = [var_99, var_100]
    var_102 = {var_98: var_101}
    var_103 = module_1.to_json_schema(var_97)
    var_104 = module_0.String()
    var_105 = module_0.Integer()
    var_106 = [var_104, var_105]
    var_107 = module_2.OneOf(var_106)
    var_108 = 'oneOf'
    var_109 = {var_9: var_14}
    var_110 = {var_9: var_27}
    var_111 = [var_109, var_110]
    var_112 = {var_108: var_111}
    var_113 = module_1.to_json_schema(var_107)
    var_114 = module_0.String()
    var_115 = 'test'
    var_116 = module_0.Const(var_115)
    var_117 = [var_114, var_116]
    var_118 = module_2.AllOf(var_117)
    var_119 = 'allOf'
    var_120 = {var_9: var_14}
    var_121 = {var_91: var_115}
    var_122 = [var_120, var_121]
    var_123 = {var_119: var_122}
    var_124 = module_1.to_json_schema(var_118)
    var_125 = module_0.String()
    var_126 = module_0.Integer()
    var_127 = module_0.Boolean()
    var_128 = module_2.IfThenElse(var_125, var_126, var_127)
    var_129 = 'if'
    var_130 = 'then'
    var_131 = 'else'
    var_132 = {var_9: var_14}
    var_133 = {var_9: var_27}
    var_134 = {var_9: var_36}
    var_135 = {var_129: var_132, var_130: var_133, var_131: var_134}
    var_136 = module_1.to_json_schema(var_128)
    var_137 = module_0.String()
    var_138 = module_2.Not(var_137)
    var_139 = 'not'
    var_140 = {var_9: var_14}
    var_141 = {var_139: var_140}
    var_142 = module_1.to_json_schema(var_138)
    var_143 = 'Test'
    var_144 = module_0.String()
    var_145 = {var_143: var_144}
    var_146 = '$ref'
    var_147 = 'components'
    var_148 = '#/components/schemas/Test'
    var_149 = 'schemas'
    var_150 = {var_9: var_14}
    var_151 = {var_143: var_150}
    var_152 = {var_149: var_151}
    var_153 = {var_146: var_148, var_147: var_152}
    var_154 = module_0.String()
    var_155 = {var_52: var_154}
    var_156 = [var_52]
    var_157 = module_3.Schema(var_155)
    var_158 = {var_9: var_14}
    var_159 = {var_52: var_158}
    var_160 = [var_52]
    var_161 = {var_9: var_69, var_62: var_159, var_68: var_160}
    var_162 = module_1.to_json_schema(var_157)
    var_163 = 'StringField'
    var_164 = 'IntField'
    var_165 = module_0.String()
    var_166 = module_0.Integer()
    var_167 = {var_163: var_165, var_164: var_166}
    var_168 = {var_9: var_14}
    var_169 = {var_9: var_27}
    var_170 = {var_163: var_168, var_164: var_169}
    var_171 = {var_149: var_170}
    var_172 = {var_147: var_171}



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
    var_28 = '^[a-zA-Z0-9_]+$'
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
    var_64 = {var_58: var_29}
    var_65 = {var_0: var_57, var_53: var_62, var_54: var_63, var_55: var_35, var_56: var_46, var_6: var_64}
    var_66 = False
    var_67 = module_0.Definitions()
    var_68 = module_1.from_json_schema_type(var_65, var_57, var_66, var_67)
    var_69 = var_68.properties[var_58]
    var_70 = var_68.properties[var_59]
    var_71 = {var_0: var_25}
    var_72 = module_0.Definitions()
    var_73 = module_1.from_json_schema_type(var_71, var_25, var_35, var_72)



# Parsed testcases at query #21
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
    var_14 = 'null'
    var_15 = [var_1, var_14]
    var_16 = {var_0: var_15}
    var_17 = module_0.Definitions()
    var_18 = module_1.type_from_json_schema(var_16, var_17)
    var_19 = {}
    var_20 = module_0.Definitions()
    var_21 = module_1.type_from_json_schema(var_19, var_20)
    var_22 = [var_14]
    var_23 = {var_0: var_22}
    var_24 = module_0.Definitions()
    var_25 = module_1.type_from_json_schema(var_23, var_24)
    var_26 = 'minLength'
    var_27 = 'maxLength'
    var_28 = 'pattern'
    var_29 = 5
    var_30 = 10
    var_31 = '^[A-Za-z]+$'
    var_32 = {var_0: var_1, var_26: var_29, var_27: var_30, var_28: var_31}
    var_33 = module_0.Definitions()
    var_34 = module_1.type_from_json_schema(var_32, var_33)
    var_35 = 'properties'
    var_36 = 'required'
    var_37 = 'object'
    var_38 = 'name'
    var_39 = 'age'
    var_40 = {var_0: var_1}
    var_41 = {var_0: var_5}
    var_42 = {var_38: var_40, var_39: var_41}
    var_43 = [var_38]
    var_44 = {var_0: var_37, var_35: var_42, var_36: var_43}
    var_45 = module_0.Definitions()
    var_46 = module_1.type_from_json_schema(var_44, var_45)
    var_47 = 'items'
    var_48 = 'minItems'
    var_49 = 'maxItems'
    var_50 = 'array'
    var_51 = {var_0: var_1}
    var_52 = 1
    var_53 = {var_0: var_50, var_47: var_51, var_48: var_52, var_49: var_29}
    var_54 = module_0.Definitions()
    var_55 = module_1.type_from_json_schema(var_53, var_54)
    var_56 = var_55.items



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
    var_23 = 10
    var_24 = '^[A-Za-z]+$'
    var_25 = 'hello'
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
    var_39 = 'uniqueItems'
    var_40 = 'array'
    var_41 = {var_0: var_21}
    var_42 = 'a'
    var_43 = 'b'
    var_44 = [var_42, var_43]
    var_45 = {var_0: var_40, var_36: var_41, var_37: var_31, var_38: var_22, var_39: var_31, var_3: var_44}
    var_46 = False
    var_47 = module_0.Definitions()
    var_48 = module_1.from_json_schema_type(var_45, var_40, var_46, var_47)
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
    var_70 = module_1.from_json_schema_type(var_68, var_21, var_31, var_69)



# Parsed testcases at query #23
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
    var_12 = 'nullable'
    var_13 = True
    var_14 = {var_0: var_1, var_12: var_13}
    var_15 = module_0.Definitions()
    var_16 = module_1.type_from_json_schema(var_14, var_15)
    var_17 = var_16.any_of
    var_18 = var_16.any_of
    var_19 = None
    var_20 = {var_12: var_13}
    var_21 = module_0.Definitions()
    var_22 = module_1.type_from_json_schema(var_20, var_21)
    var_23 = False
    var_24 = {var_12: var_23}
    var_25 = module_0.Definitions()
    var_26 = module_1.type_from_json_schema(var_24, var_25)
    var_27 = 'array'
    var_28 = {var_0: var_27}
    var_29 = module_0.Definitions()
    var_30 = module_1.type_from_json_schema(var_28, var_29)
    var_31 = 'object'
    var_32 = {var_0: var_31}
    var_33 = module_0.Definitions()
    var_34 = module_1.type_from_json_schema(var_32, var_33)
    var_35 = {var_0: var_5}
    var_36 = module_0.Definitions()
    var_37 = module_1.type_from_json_schema(var_35, var_36)
    var_38 = 'integer'
    var_39 = {var_0: var_38}
    var_40 = module_0.Definitions()
    var_41 = module_1.type_from_json_schema(var_39, var_40)
    var_42 = 'boolean'
    var_43 = {var_0: var_42}
    var_44 = module_0.Definitions()
    var_45 = module_1.type_from_json_schema(var_43, var_44)



# Parsed testcases at query #24
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
    var_26 = 5
    var_27 = {var_0: var_1, var_25: var_26}
    var_28 = module_0.Definitions()
    var_29 = module_1.type_from_json_schema(var_27, var_28)



# Parsed testcases at query #25
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
    var_26 = 1
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
    var_49 = True
    var_50 = [var_29]
    var_51 = {var_0: var_45, var_40: var_46, var_41: var_47, var_42: var_35, var_43: var_48, var_44: var_49, var_6: var_50}
    var_52 = False
    var_53 = module_0.Definitions()
    var_54 = module_1.from_json_schema_type(var_51, var_45, var_52, var_53)
    var_55 = var_54.items
    var_56 = 'properties'
    var_57 = 'patternProperties'
    var_58 = 'additionalProperties'
    var_59 = 'propertyNames'
    var_60 = 'minProperties'
    var_61 = 'maxProperties'
    var_62 = 'required'
    var_63 = 'object'
    var_64 = 'name'
    var_65 = {var_0: var_25}
    var_66 = {var_64: var_65}
    var_67 = '^S_'
    var_68 = {var_0: var_25}
    var_69 = {var_67: var_68}
    var_70 = False
    var_71 = {var_0: var_25}
    var_72 = [var_64]
    var_73 = {var_64: var_29}
    var_74 = {var_0: var_63, var_56: var_66, var_57: var_69, var_58: var_70, var_59: var_71, var_60: var_49, var_61: var_48, var_62: var_72, var_6: var_73}
    var_75 = False
    var_76 = module_0.Definitions()
    var_77 = module_1.from_json_schema_type(var_74, var_63, var_75, var_76)
    var_78 = var_77.properties[var_64]
    var_79 = var_77.pattern_properties[var_67]
    var_80 = var_77.property_names
    var_81 = {var_0: var_25}
    var_82 = True
    var_83 = module_0.Definitions()
    var_84 = module_1.from_json_schema_type(var_81, var_25, var_82, var_83)



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
    var_85 = [var_24, var_25]
    var_86 = {var_4: var_5, var_23: var_85, var_36: var_0}
    var_87 = module_0.from_json_schema(var_86)
    var_88 = var_87.schemas
    var_89 = len(var_88)
    assert var_89 == 2



# Parsed testcases at query #27
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
    var_18 = 5
    var_19 = 10
    var_20 = {var_0: var_17, var_15: var_18, var_16: var_19}
    var_21 = False
    var_22 = module_0.Definitions()
    var_23 = module_1.from_json_schema_type(var_20, var_17, var_21, var_22)
    var_24 = 'boolean'
    var_25 = {var_0: var_24}
    var_26 = False
    var_27 = module_0.Definitions()
    var_28 = module_1.from_json_schema_type(var_25, var_24, var_26, var_27)
    var_29 = 'items'
    var_30 = 'array'
    var_31 = {var_0: var_17}
    var_32 = {var_0: var_30, var_29: var_31}
    var_33 = False
    var_34 = module_0.Definitions()
    var_35 = module_1.from_json_schema_type(var_32, var_30, var_33, var_34)
    var_36 = var_35.items
    var_37 = 'properties'
    var_38 = 'required'
    var_39 = 'object'
    var_40 = 'name'
    var_41 = {var_0: var_17}
    var_42 = {var_40: var_41}
    var_43 = [var_40]
    var_44 = {var_0: var_39, var_37: var_42, var_38: var_43}
    var_45 = False
    var_46 = module_0.Definitions()
    var_47 = module_1.from_json_schema_type(var_44, var_39, var_45, var_46)
    var_48 = var_47.properties[var_40]
    var_49 = {var_0: var_17}
    var_50 = True
    var_51 = module_0.Definitions()
    var_52 = module_1.from_json_schema_type(var_49, var_17, var_50, var_51)



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
    var_6 = True
    var_7 = module_0.String(max_length=var_5, min_length=var_4)
    var_8 = 'type'
    var_9 = 'minLength'
    var_10 = 'maxLength'
    var_11 = 'string'
    var_12 = 'null'
    var_13 = [var_11, var_12]
    var_14 = {var_8: var_13, var_9: var_6, var_10: var_5}
    var_15 = module_1.to_json_schema(var_7)
    var_16 = 0
    var_17 = 100
    var_18 = False
    var_19 = module_0.Integer(minimum=var_16, maximum=var_17)
    var_20 = 'minimum'
    var_21 = 'maximum'
    var_22 = 'integer'
    var_23 = {var_8: var_22, var_20: var_18, var_21: var_17}
    var_24 = module_1.to_json_schema(var_19)
    var_25 = 0.5
    var_26 = True
    var_27 = module_0.Float(multiple_of=var_25)
    var_28 = 'multipleOf'
    var_29 = 'number'
    var_30 = [var_29, var_12]
    var_31 = {var_8: var_30, var_28: var_25}
    var_32 = module_1.to_json_schema(var_27)
    var_33 = True
    var_34 = module_0.Boolean()
    var_35 = 'boolean'
    var_36 = [var_35, var_12]
    var_37 = {var_8: var_36}
    var_38 = module_1.to_json_schema(var_34)
    var_39 = module_0.String()
    var_40 = False
    var_41 = module_0.Array(var_39, min_items=var_33)
    var_42 = 'minItems'
    var_43 = 'items'
    var_44 = 'array'
    var_45 = {var_8: var_11}
    var_46 = {var_8: var_44, var_42: var_33, var_43: var_45}
    var_47 = module_1.to_json_schema(var_41)
    var_48 = 'name'
    var_49 = module_0.String()
    var_50 = {var_48: var_49}
    var_51 = [var_48]
    var_52 = True
    var_53 = module_0.Object(properties=var_50, required=var_51)
    var_54 = 'properties'
    var_55 = 'required'
    var_56 = 'object'
    var_57 = [var_56, var_12]
    var_58 = {var_8: var_11}
    var_59 = {var_48: var_58}
    var_60 = [var_48]
    var_61 = {var_8: var_57, var_54: var_59, var_55: var_60}
    var_62 = module_1.to_json_schema(var_53)
    var_63 = 'a'
    var_64 = (var_63, var_63)
    var_65 = 'b'
    var_66 = (var_65, var_65)
    var_67 = [var_64, var_66]
    var_68 = module_0.Choice(choices=var_67)
    var_69 = 'enum'
    var_70 = [var_63, var_65]
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
    var_83 = {var_8: var_11}
    var_84 = {var_8: var_22}
    var_85 = [var_83, var_84]
    var_86 = {var_82: var_85}
    var_87 = module_1.to_json_schema(var_81)
    var_88 = module_0.String()
    var_89 = 'test'
    var_90 = module_0.Const(var_89)
    var_91 = [var_88, var_90]
    var_92 = module_2.AllOf(var_91)
    var_93 = 'allOf'
    var_94 = {var_8: var_11}
    var_95 = {var_75: var_89}
    var_96 = [var_94, var_95]
    var_97 = {var_93: var_96}
    var_98 = module_1.to_json_schema(var_92)
    var_99 = 'Test'
    var_100 = module_0.String()
    var_101 = {var_99: var_100}
    var_102 = '$ref'
    var_103 = 'components'
    var_104 = '#/components/schemas/Test'
    var_105 = 'schemas'
    var_106 = {var_8: var_11}
    var_107 = {var_99: var_106}
    var_108 = {var_105: var_107}
    var_109 = {var_102: var_104, var_103: var_108}
    var_110 = module_0.String()
    var_111 = module_0.Integer()
    var_112 = module_0.Boolean()
    var_113 = module_2.IfThenElse(var_110, var_111, var_112)
    var_114 = 'if'
    var_115 = 'then'
    var_116 = 'else'
    var_117 = {var_8: var_11}
    var_118 = {var_8: var_22}
    var_119 = {var_8: var_35}
    var_120 = {var_114: var_117, var_115: var_118, var_116: var_119}
    var_121 = module_1.to_json_schema(var_113)
    var_122 = module_0.String()
    var_123 = module_2.Not(var_122)
    var_124 = 'not'
    var_125 = {var_8: var_11}
    var_126 = {var_124: var_125}
    var_127 = module_1.to_json_schema(var_123)
    var_128 = module_0.String()
    var_129 = {var_48: var_128}
    var_130 = [var_48]
    var_131 = True
    var_132 = module_3.Schema(var_129)
    var_133 = [var_56, var_12]
    var_134 = {var_8: var_11}
    var_135 = {var_48: var_134}
    var_136 = [var_48]
    var_137 = {var_8: var_133, var_54: var_135, var_55: var_136}
    var_138 = module_1.to_json_schema(var_132)



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
    var_23 = 10
    var_24 = '^[A-Za-z]+$'
    var_25 = 'hello'
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
    var_39 = 'uniqueItems'
    var_40 = 'array'
    var_41 = {var_0: var_21}
    var_42 = 'a'
    var_43 = 'b'
    var_44 = [var_42, var_43]
    var_45 = {var_0: var_40, var_36: var_41, var_37: var_31, var_38: var_22, var_39: var_31, var_3: var_44}
    var_46 = False
    var_47 = module_0.Definitions()
    var_48 = module_1.from_json_schema_type(var_45, var_40, var_46, var_47)
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



# Parsed testcases at query #30
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
    var_65 = {var_0: var_20}
    var_66 = module_0.Definitions()
    var_67 = module_1.from_json_schema_type(var_65, var_20, var_30, var_66)



# Parsed testcases at query #31
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
    var_102 = 'Test'
    var_103 = module_0.String()
    var_104 = {var_102: var_103}
    var_105 = '$ref'
    var_106 = 'components'
    var_107 = '#/components/schemas/Test'
    var_108 = 'schemas'
    var_109 = {var_9: var_14}
    var_110 = {var_102: var_109}
    var_111 = {var_108: var_110}
    var_112 = {var_105: var_107, var_106: var_111}
    var_113 = module_0.String()
    var_114 = {var_51: var_113}
    var_115 = [var_51]
    var_116 = module_3.Schema(var_114)
    var_117 = {var_9: var_14}
    var_118 = {var_51: var_117}
    var_119 = [var_51]
    var_120 = {var_9: var_60, var_56: var_118, var_57: var_119}
    var_121 = module_1.to_json_schema(var_116)



# Parsed testcases at query #32
#--------------------------


import typesystem.json_schema as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)
    var_2 = False
    var_3 = module_0.from_json_schema(var_2)
    var_4 = module_1.Definitions()
    var_5 = 'type'
    var_6 = 'string'
    var_7 = '$ref'
    var_8 = '#/components/schemas/Test'
    var_9 = {var_7: var_8}
    var_10 = module_0.from_json_schema(var_9, var_4)
    var_11 = {var_5: var_6}
    var_12 = module_0.from_json_schema(var_11)
    var_13 = 'integer'
    var_14 = {var_5: var_13}
    var_15 = module_0.from_json_schema(var_14)
    var_16 = 'number'
    var_17 = {var_5: var_16}
    var_18 = module_0.from_json_schema(var_17)
    var_19 = 'boolean'
    var_20 = {var_5: var_19}
    var_21 = module_0.from_json_schema(var_20)
    var_22 = 'array'
    var_23 = {var_5: var_22}
    var_24 = module_0.from_json_schema(var_23)
    var_25 = 'object'
    var_26 = {var_5: var_25}
    var_27 = module_0.from_json_schema(var_26)
    var_28 = 'enum'
    var_29 = 'a'
    var_30 = 'b'
    var_31 = 'c'
    var_32 = [var_29, var_30, var_31]
    var_33 = {var_28: var_32}
    var_34 = module_0.from_json_schema(var_33)
    var_35 = module_0.from_json_schema(var_33)
    var_36 = var_35.choices
    var_37 = 'const'
    var_38 = 'test'
    var_39 = {var_37: var_38}
    var_40 = module_0.from_json_schema(var_39)
    var_41 = module_0.from_json_schema(var_39)
    var_42 = var_41.value
    assert var_42 == 'test'
    var_43 = 'allOf'
    var_44 = {var_5: var_6}
    var_45 = 'minLength'
    var_46 = 5
    var_47 = {var_45: var_46}
    var_48 = [var_44, var_47]
    var_49 = {var_43: var_48}
    var_50 = module_0.from_json_schema(var_49)
    var_51 = var_50.schemas
    var_52 = len(var_51)
    assert var_52 == 2
    var_53 = 'anyOf'
    var_54 = {var_5: var_6}
    var_55 = {var_5: var_13}
    var_56 = [var_54, var_55]
    var_57 = {var_53: var_56}
    var_58 = module_0.from_json_schema(var_57)
    var_59 = var_58.schemas
    var_60 = len(var_59)
    assert var_60 == 2
    var_61 = 'oneOf'
    var_62 = {var_5: var_6}
    var_63 = {var_5: var_13}
    var_64 = [var_62, var_63]
    var_65 = {var_61: var_64}
    var_66 = module_0.from_json_schema(var_65)
    var_67 = var_66.schemas
    var_68 = len(var_67)
    assert var_68 == 2
    var_69 = 'not'
    var_70 = {var_5: var_6}
    var_71 = {var_69: var_70}
    var_72 = module_0.from_json_schema(var_71)
    var_73 = var_72.schema
    var_74 = 'if'
    var_75 = 'then'
    var_76 = 'else'
    var_77 = {var_5: var_6}
    var_78 = {var_45: var_46}
    var_79 = {var_5: var_13}
    var_80 = {var_74: var_77, var_75: var_78, var_76: var_79}
    var_81 = module_0.from_json_schema(var_80)
    var_82 = var_81.if_schema
    var_83 = var_81.then_schema
    var_84 = var_81.else_schema
    var_85 = 'maxLength'
    var_86 = 'pattern'
    var_87 = 10
    var_88 = '^[a-z]+$'
    var_89 = {var_5: var_6, var_45: var_46, var_85: var_87, var_86: var_88}
    var_90 = module_0.from_json_schema(var_89)
    var_91 = {}
    var_92 = module_0.from_json_schema(var_91)



# Parsed testcases at query #33
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
    var_22 = 'exclusiveMinimum'
    var_23 = 'maximum'
    var_24 = 'integer'
    var_25 = True
    var_26 = {var_9: var_24, var_21: var_17, var_22: var_25, var_23: var_18}
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
    var_101 = 'Person'
    var_102 = module_0.String()
    var_103 = {var_51: var_102}
    var_104 = module_0.Object(properties=var_103)
    var_105 = {var_101: var_104}
    var_106 = '$ref'
    var_107 = 'components'
    var_108 = '#/components/schemas/Person'
    var_109 = 'schemas'
    var_110 = {var_9: var_14}
    var_111 = {var_51: var_110}
    var_112 = {var_9: var_60, var_56: var_111}
    var_113 = {var_101: var_112}
    var_114 = {var_109: var_113}
    var_115 = {var_106: var_108, var_107: var_114}
    var_116 = module_0.String()
    var_117 = {var_51: var_116}
    var_118 = [var_51]
    var_119 = module_3.Schema(var_117)
    var_120 = {var_9: var_14}
    var_121 = {var_51: var_120}
    var_122 = [var_51]
    var_123 = {var_9: var_60, var_56: var_121, var_57: var_122}
    var_124 = module_1.to_json_schema(var_119)



# Parsed testcases at query #34
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
    var_31 = 'value'
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



# Parsed testcases at query #35
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
    var_40 = {var_39: var_0}
    var_41 = [var_38, var_40]
    var_42 = {var_37: var_41}
    var_43 = module_0.from_json_schema(var_42)
    var_44 = 'anyOf'
    var_45 = {var_4: var_5}
    var_46 = {var_4: var_8}
    var_47 = [var_45, var_46]
    var_48 = {var_44: var_47}
    var_49 = module_0.from_json_schema(var_48)
    var_50 = 'oneOf'
    var_51 = {var_4: var_5}
    var_52 = {var_4: var_8}
    var_53 = [var_51, var_52]
    var_54 = {var_50: var_53}
    var_55 = module_0.from_json_schema(var_54)
    var_56 = 'not'
    var_57 = {var_4: var_5}
    var_58 = {var_56: var_57}
    var_59 = module_0.from_json_schema(var_58)
    var_60 = 'if'
    var_61 = 'then'
    var_62 = 'else'
    var_63 = {var_4: var_5}
    var_64 = {var_39: var_0}
    var_65 = {var_39: var_2}
    var_66 = {var_60: var_63, var_61: var_64, var_62: var_65}
    var_67 = module_0.from_json_schema(var_66)
    var_68 = module_1.Definitions()
    var_69 = {var_4: var_5}
    var_70 = '$ref'
    var_71 = '#/components/schemas/Test'
    var_72 = {var_70: var_71}
    var_73 = module_0.from_json_schema(var_72, var_68)
    var_74 = [var_27, var_28, var_29]
    var_75 = {var_4: var_5, var_26: var_74}
    var_76 = module_0.from_json_schema(var_75)
    var_77 = {}
    var_78 = module_0.from_json_schema(var_77)



# Parsed testcases at query #36
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
    var_21 = 2
    var_22 = module_0.Integer(minimum=var_19, maximum=var_20, exclusive_minimum=var_4, multiple_of=var_21)
    var_23 = 'minimum'
    var_24 = 'maximum'
    var_25 = 'exclusiveMinimum'
    var_26 = 'multipleOf'
    var_27 = 'integer'
    var_28 = {var_9: var_27, var_23: var_19, var_24: var_20, var_25: var_4, var_26: var_21}
    var_29 = module_1.to_json_schema(var_22)
    var_30 = module_0.Float(minimum=var_19, maximum=var_4)
    var_31 = 'number'
    var_32 = [var_31, var_15]
    var_33 = {var_9: var_32, var_23: var_19, var_24: var_4}
    var_34 = module_1.to_json_schema(var_30)
    var_35 = module_0.Boolean()
    var_36 = 'boolean'
    var_37 = {var_9: var_36}
    var_38 = module_1.to_json_schema(var_35)
    var_39 = module_0.String()
    var_40 = module_0.Array(var_39, min_items=var_4, max_items=var_5, unique_items=var_4)
    var_41 = 'items'
    var_42 = 'minItems'
    var_43 = 'maxItems'
    var_44 = 'uniqueItems'
    var_45 = 'array'
    var_46 = [var_45, var_15]
    var_47 = {var_9: var_14}
    var_48 = {var_9: var_46, var_41: var_47, var_42: var_4, var_43: var_5, var_44: var_4}
    var_49 = module_1.to_json_schema(var_40)
    var_50 = 'name'
    var_51 = module_0.String()
    var_52 = {var_50: var_51}
    var_53 = [var_50]
    var_54 = module_0.Object(properties=var_52, min_properties=var_4, max_properties=var_5, required=var_53)
    var_55 = 'properties'
    var_56 = 'required'
    var_57 = 'minProperties'
    var_58 = 'maxProperties'
    var_59 = 'object'
    var_60 = {var_9: var_14}
    var_61 = {var_50: var_60}
    var_62 = [var_50]
    var_63 = {var_9: var_59, var_55: var_61, var_56: var_62, var_57: var_4, var_58: var_5}
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
    var_86 = {var_9: var_27}
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
    var_101 = 'Test'
    var_102 = module_0.String()
    var_103 = {var_101: var_102}
    var_104 = '$ref'
    var_105 = 'components'
    var_106 = '#/components/schemas/Test'
    var_107 = 'schemas'
    var_108 = {var_9: var_14}
    var_109 = {var_101: var_108}
    var_110 = {var_107: var_109}
    var_111 = {var_104: var_106, var_105: var_110}
    var_112 = module_1.to_json_schema(var_94)
    var_113 = module_0.String()
    var_114 = module_0.Integer()
    var_115 = module_0.Boolean()
    var_116 = module_2.IfThenElse(var_113, var_114, var_115)
    var_117 = 'if'
    var_118 = 'then'
    var_119 = 'else'
    var_120 = {var_9: var_14}
    var_121 = {var_9: var_27}
    var_122 = {var_9: var_36}
    var_123 = {var_117: var_120, var_118: var_121, var_119: var_122}
    var_124 = module_1.to_json_schema(var_116)
    var_125 = module_0.String()
    var_126 = module_2.Not(var_125)
    var_127 = 'not'
    var_128 = {var_9: var_14}
    var_129 = {var_127: var_128}
    var_130 = module_1.to_json_schema(var_126)



# Parsed testcases at query #37
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
    var_36 = 5
    var_37 = True
    var_38 = module_0.Array(var_35, min_items=var_4, max_items=var_36, unique_items=var_37)
    var_39 = 'items'
    var_40 = 'minItems'
    var_41 = 'maxItems'
    var_42 = 'uniqueItems'
    var_43 = 'array'
    var_44 = {var_9: var_14}
    var_45 = True
    var_46 = {var_9: var_43, var_39: var_44, var_40: var_37, var_41: var_36, var_42: var_45}
    var_47 = module_1.to_json_schema(var_38)
    var_48 = 'name'
    var_49 = module_0.String()
    var_50 = {var_48: var_49}
    var_51 = False
    var_52 = [var_48]
    var_53 = module_0.Object(properties=var_50, additional_properties=var_51, required=var_52)
    var_54 = 'properties'
    var_55 = 'additionalProperties'
    var_56 = 'required'
    var_57 = 'object'
    var_58 = {var_9: var_14}
    var_59 = {var_48: var_58}
    var_60 = False
    var_61 = [var_48]
    var_62 = {var_9: var_57, var_54: var_59, var_55: var_60, var_56: var_61}
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
    var_90 = 'test'
    var_91 = module_0.Const(var_90)
    var_92 = [var_89, var_91]
    var_93 = module_2.AllOf(var_92)
    var_94 = 'allOf'
    var_95 = {var_9: var_14}
    var_96 = {var_76: var_90}
    var_97 = [var_95, var_96]
    var_98 = {var_94: var_97}
    var_99 = module_1.to_json_schema(var_93)
    var_100 = 'Test'
    var_101 = module_0.String()
    var_102 = {var_100: var_101}
    var_103 = '$ref'
    var_104 = 'components'
    var_105 = '#/components/schemas/Test'
    var_106 = 'schemas'
    var_107 = {var_9: var_14}
    var_108 = {var_100: var_107}
    var_109 = {var_106: var_108}
    var_110 = {var_103: var_105, var_104: var_109}
    var_111 = module_0.String()
    var_112 = {var_48: var_111}
    var_113 = [var_48]
    var_114 = module_3.Schema(var_112)
    var_115 = {var_9: var_14}
    var_116 = {var_48: var_115}
    var_117 = [var_48]
    var_118 = {var_9: var_57, var_54: var_116, var_56: var_117}
    var_119 = module_1.to_json_schema(var_114)
    var_120 = module_0.String()
    var_121 = module_0.Integer()
    var_122 = module_0.Boolean()
    var_123 = module_2.IfThenElse(var_120, var_121, var_122)
    var_124 = 'if'
    var_125 = 'then'
    var_126 = 'else'
    var_127 = {var_9: var_14}
    var_128 = {var_9: var_22}
    var_129 = {var_9: var_32}
    var_130 = {var_124: var_127, var_125: var_128, var_126: var_129}
    var_131 = module_1.to_json_schema(var_123)
    var_132 = module_0.String()
    var_133 = module_2.Not(var_132)
    var_134 = 'not'
    var_135 = {var_9: var_14}
    var_136 = {var_134: var_135}
    var_137 = module_1.to_json_schema(var_133)
    var_138 = True
    var_139 = module_0.String()
    var_140 = 'null'
    var_141 = [var_14, var_140]
    var_142 = {var_9: var_141}
    var_143 = module_1.to_json_schema(var_139)



# Parsed testcases at query #38
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
    var_29 = module_0.Boolean()
    var_30 = 'boolean'
    var_31 = [var_30, var_15]
    var_32 = {var_9: var_31}
    var_33 = module_1.to_json_schema(var_29)
    var_34 = module_0.String()
    var_35 = 5
    var_36 = module_0.Array(var_34, min_items=var_4, max_items=var_35)
    var_37 = 'items'
    var_38 = 'minItems'
    var_39 = 'maxItems'
    var_40 = 'array'
    var_41 = {var_9: var_14}
    var_42 = {var_9: var_40, var_37: var_41, var_38: var_4, var_39: var_35}
    var_43 = module_1.to_json_schema(var_36)
    var_44 = 'name'
    var_45 = module_0.String()
    var_46 = {var_44: var_45}
    var_47 = [var_44]
    var_48 = module_0.Object(properties=var_46, required=var_47)
    var_49 = 'properties'
    var_50 = 'required'
    var_51 = 'object'
    var_52 = [var_51, var_15]
    var_53 = {var_9: var_14}
    var_54 = {var_44: var_53}
    var_55 = [var_44]
    var_56 = {var_9: var_52, var_49: var_54, var_50: var_55}
    var_57 = module_1.to_json_schema(var_48)
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
    var_68 = 'value'
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
    var_79 = {var_9: var_26}
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
    var_94 = module_3.Definitions()
    var_95 = module_3.Reference(var_84, var_94)
    var_96 = '$ref'
    var_97 = '#/components/schemas/test'
    var_98 = {var_96: var_97}
    var_99 = module_1.to_json_schema(var_95)
    var_100 = module_0.String()
    var_101 = {var_44: var_100}
    var_102 = module_3.Schema(var_101)
    var_103 = {var_9: var_14}
    var_104 = {var_44: var_103}
    var_105 = {var_9: var_51, var_49: var_104}
    var_106 = module_1.to_json_schema(var_102)



# Parsed testcases at query #39
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
    var_40 = 5
    var_41 = module_0.String()
    var_42 = module_0.Array(var_41, var_19, var_4, var_40)
    var_43 = 'minItems'
    var_44 = 'maxItems'
    var_45 = 'items'
    var_46 = 'additionalItems'
    var_47 = 'uniqueItems'
    var_48 = 'array'
    var_49 = [var_48, var_15]
    var_50 = {var_9: var_14}
    var_51 = {var_9: var_49, var_43: var_4, var_44: var_40, var_45: var_50, var_46: var_19, var_47: var_4}
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
    var_103 = module_0.String()
    var_104 = module_0.Integer()
    var_105 = [var_103, var_104]
    var_106 = module_2.OneOf(var_105)
    var_107 = 'oneOf'
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
    var_122 = {var_9: var_37}
    var_123 = {var_117: var_120, var_118: var_121, var_119: var_122}
    var_124 = module_1.to_json_schema(var_116)
    var_125 = module_0.String()
    var_126 = module_2.Not(var_125)
    var_127 = 'not'
    var_128 = {var_9: var_14}
    var_129 = {var_127: var_128}
    var_130 = module_1.to_json_schema(var_126)
    var_131 = module_0.String()
    var_132 = {var_93: var_131}
    var_133 = module_3.Reference(var_93, var_132)
    var_134 = '$ref'
    var_135 = 'components'
    var_136 = '#/components/schemas/test'
    var_137 = 'schemas'
    var_138 = {var_9: var_14}
    var_139 = {var_93: var_138}
    var_140 = {var_137: var_139}
    var_141 = {var_134: var_136, var_135: var_140}
    var_142 = module_1.to_json_schema(var_133)
    var_143 = module_0.String()
    var_144 = {var_53: var_143}
    var_145 = [var_53]
    var_146 = module_3.Schema(var_144)
    var_147 = {var_9: var_14}
    var_148 = {var_53: var_147}
    var_149 = [var_53]
    var_150 = {var_9: var_61, var_58: var_148, var_60: var_149}
    var_151 = module_1.to_json_schema(var_146)
    var_152 = 'field1'
    var_153 = 'field2'
    var_154 = module_0.String()
    var_155 = module_0.Integer()
    var_156 = {var_152: var_154, var_153: var_155}
    var_157 = {var_9: var_14}
    var_158 = {var_9: var_26}
    var_159 = {var_152: var_157, var_153: var_158}
    var_160 = {var_137: var_159}
    var_161 = {var_135: var_160}



# Parsed testcases at query #40
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
    var_56 = 'John'
    var_57 = 30
    var_58 = {var_50: var_56, var_51: var_57}
    var_59 = {var_0: var_49, var_47: var_54, var_48: var_55, var_3: var_58}
    var_60 = False
    var_61 = module_0.Definitions()
    var_62 = module_1.from_json_schema_type(var_59, var_49, var_60, var_61)
    var_63 = var_62.properties[var_50]
    var_64 = var_62.properties[var_51]



# Parsed testcases at query #41
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
    var_58 = 'additionalProperties'
    var_59 = 'minProperties'
    var_60 = 'maxProperties'
    var_61 = 'required'
    var_62 = 'object'
    var_63 = 'name'
    var_64 = 'age'
    var_65 = {var_0: var_25}
    var_66 = {var_0: var_16}
    var_67 = {var_63: var_65, var_64: var_66}
    var_68 = False
    var_69 = [var_63]
    var_70 = 'John'
    var_71 = 30
    var_72 = {var_63: var_70, var_64: var_71}
    var_73 = {var_0: var_62, var_57: var_67, var_58: var_68, var_59: var_35, var_60: var_48, var_61: var_69, var_6: var_72}
    var_74 = False
    var_75 = module_0.Definitions()
    var_76 = module_1.from_json_schema_type(var_73, var_62, var_74, var_75)
    var_77 = var_76.properties[var_63]
    var_78 = var_76.properties[var_64]
    var_79 = {var_0: var_25}
    var_80 = module_0.Definitions()
    var_81 = module_1.from_json_schema_type(var_79, var_25, var_35, var_80)



# Parsed testcases at query #42
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
    var_31 = 'test'
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
    var_42 = var_41.all_of
    var_43 = len(var_42)
    assert var_43 == 2
    var_44 = 'anyOf'
    var_45 = {var_4: var_5}
    var_46 = {var_4: var_8}
    var_47 = [var_45, var_46]
    var_48 = {var_44: var_47}
    var_49 = module_0.from_json_schema(var_48)
    var_50 = var_49.one_of
    var_51 = len(var_50)
    assert var_51 == 2
    var_52 = 'oneOf'
    var_53 = {var_4: var_5}
    var_54 = {var_4: var_8}
    var_55 = [var_53, var_54]
    var_56 = {var_52: var_55}
    var_57 = module_0.from_json_schema(var_56)
    var_58 = var_57.one_of
    var_59 = len(var_58)
    assert var_59 == 2
    var_60 = 'not'
    var_61 = {var_4: var_5}
    var_62 = {var_60: var_61}
    var_63 = module_0.from_json_schema(var_62)
    var_64 = var_63.not_
    var_65 = 'if'
    var_66 = 'then'
    var_67 = 'else'
    var_68 = {var_4: var_5}
    var_69 = {var_36: var_37}
    var_70 = {var_4: var_8}
    var_71 = {var_65: var_68, var_66: var_69, var_67: var_70}
    var_72 = module_0.from_json_schema(var_71)
    var_73 = var_72.if_
    var_74 = var_72.then
    var_75 = var_72.else_
    var_76 = module_1.Definitions()
    var_77 = '$ref'
    var_78 = '#/components/schemas/test'
    var_79 = {var_77: var_78}
    var_80 = module_0.from_json_schema(var_79, var_76)
    var_81 = 'maxLength'
    var_82 = 10
    var_83 = {var_4: var_5, var_36: var_37, var_81: var_82}
    var_84 = module_0.from_json_schema(var_83)
    var_85 = var_84.all_of
    var_86 = len(var_85)
    assert var_86 == 2
    var_87 = {}
    var_88 = module_0.from_json_schema(var_87)



# Parsed testcases at query #43
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
    var_18 = 5
    var_19 = 10
    var_20 = {var_0: var_17, var_15: var_18, var_16: var_19}
    var_21 = False
    var_22 = module_0.Definitions()
    var_23 = module_1.from_json_schema_type(var_20, var_17, var_21, var_22)
    var_24 = 'boolean'
    var_25 = {var_0: var_24}
    var_26 = False
    var_27 = module_0.Definitions()
    var_28 = module_1.from_json_schema_type(var_25, var_24, var_26, var_27)
    var_29 = 'items'
    var_30 = 'minItems'
    var_31 = 'maxItems'
    var_32 = 'array'
    var_33 = {var_0: var_17}
    var_34 = 1
    var_35 = {var_0: var_32, var_29: var_33, var_30: var_34, var_31: var_18}
    var_36 = False
    var_37 = module_0.Definitions()
    var_38 = module_1.from_json_schema_type(var_35, var_32, var_36, var_37)
    var_39 = var_38.items
    var_40 = 'properties'
    var_41 = 'required'
    var_42 = 'object'
    var_43 = 'name'
    var_44 = {var_0: var_17}
    var_45 = {var_43: var_44}
    var_46 = [var_43]
    var_47 = {var_0: var_42, var_40: var_45, var_41: var_46}
    var_48 = False
    var_49 = module_0.Definitions()
    var_50 = module_1.from_json_schema_type(var_47, var_42, var_48, var_49)
    var_51 = var_50.properties[var_43]
    var_52 = {var_0: var_17}
    var_53 = True
    var_54 = module_0.Definitions()
    var_55 = module_1.from_json_schema_type(var_52, var_17, var_53, var_54)



# Parsed testcases at query #44
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
    var_10 = 'integer'
    var_11 = {var_0: var_10, var_1: var_7, var_2: var_5}
    var_12 = False
    var_13 = module_0.Definitions()
    var_14 = module_1.from_json_schema_type(var_11, var_10, var_12, var_13)
    var_15 = 'minLength'
    var_16 = 'maxLength'
    var_17 = 'string'
    var_18 = 5
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



# Parsed testcases at query #45
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
    var_64 = {var_0: var_25}
    var_65 = {var_63: var_64}
    var_66 = '^S_'
    var_67 = {var_0: var_25}
    var_68 = {var_66: var_67}
    var_69 = False
    var_70 = {var_0: var_25}
    var_71 = [var_63]
    var_72 = {var_63: var_29}
    var_73 = {var_0: var_62, var_55: var_65, var_56: var_68, var_57: var_69, var_58: var_70, var_59: var_35, var_60: var_48, var_61: var_71, var_6: var_72}
    var_74 = False
    var_75 = module_0.Definitions()
    var_76 = module_1.from_json_schema_type(var_73, var_62, var_74, var_75)
    var_77 = var_76.properties[var_63]
    var_78 = var_76.pattern_properties[var_66]
    var_79 = var_76.property_names



# Parsed testcases at query #46
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
    var_54 = 'minProperties'
    var_55 = 'maxProperties'
    var_56 = 'required'
    var_57 = 'object'
    var_58 = 'name'
    var_59 = 'age'
    var_60 = {var_0: var_25}
    var_61 = {var_0: var_16}
    var_62 = {var_58: var_60, var_59: var_61}
    var_63 = [var_58]
    var_64 = 25
    var_65 = {var_58: var_29, var_59: var_64}
    var_66 = {var_0: var_57, var_53: var_62, var_54: var_35, var_55: var_10, var_56: var_63, var_6: var_65}
    var_67 = False
    var_68 = module_0.Definitions()
    var_69 = module_1.from_json_schema_type(var_66, var_57, var_67, var_68)
    var_70 = var_69.properties[var_58]
    var_71 = var_69.properties[var_59]



# Parsed testcases at query #47
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
    var_43 = 'additionalItems'
    var_44 = 'uniqueItems'
    var_45 = 'array'
    var_46 = {var_0: var_26}
    var_47 = 10
    var_48 = False
    var_49 = [var_30]
    var_50 = {var_0: var_45, var_40: var_46, var_41: var_10, var_42: var_47, var_43: var_48, var_44: var_10, var_6: var_49}
    var_51 = False
    var_52 = module_0.Definitions()
    var_53 = module_1.from_json_schema_type(var_50, var_45, var_51, var_52)
    var_54 = var_53.items
    var_55 = 'properties'
    var_56 = 'patternProperties'
    var_57 = 'additionalProperties'
    var_58 = 'minProperties'
    var_59 = 'maxProperties'
    var_60 = 'required'
    var_61 = 'object'
    var_62 = 'name'
    var_63 = 'age'
    var_64 = {var_0: var_26}
    var_65 = {var_0: var_17}
    var_66 = {var_62: var_64, var_63: var_65}
    var_67 = '^S_'
    var_68 = '^I_'
    var_69 = {var_0: var_26}
    var_70 = {var_0: var_17}
    var_71 = {var_67: var_69, var_68: var_70}
    var_72 = False
    var_73 = [var_62]
    var_74 = 30
    var_75 = {var_62: var_30, var_63: var_74}
    var_76 = {var_0: var_61, var_55: var_66, var_56: var_71, var_57: var_72, var_58: var_10, var_59: var_47, var_60: var_73, var_6: var_75}
    var_77 = False
    var_78 = module_0.Definitions()
    var_79 = module_1.from_json_schema_type(var_76, var_61, var_77, var_78)
    var_80 = var_79.properties[var_62]
    var_81 = var_79.properties[var_63]
    var_82 = var_79.pattern_properties[var_67]
    var_83 = var_79.pattern_properties[var_68]
    var_84 = {var_0: var_26}
    var_85 = module_0.Definitions()
    var_86 = module_1.from_json_schema_type(var_84, var_26, var_10, var_85)



# Parsed testcases at query #48
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
    var_102 = module_3.Definitions()
    var_103 = 'Test'
    var_104 = module_0.String()
    var_105 = module_3.Reference(var_103, var_102)
    var_106 = '$ref'
    var_107 = 'components'
    var_108 = '#/components/schemas/Test'
    var_109 = 'schemas'
    var_110 = {var_9: var_14}
    var_111 = {var_103: var_110}
    var_112 = {var_109: var_111}
    var_113 = {var_106: var_108, var_107: var_112}
    var_114 = module_1.to_json_schema(var_105)



# Parsed testcases at query #49
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
    var_38 = 'minLength'
    var_39 = 5
    var_40 = {var_4: var_5, var_38: var_39}
    var_41 = 'maxLength'
    var_42 = 10
    var_43 = {var_4: var_5, var_41: var_42}
    var_44 = [var_40, var_43]
    var_45 = {var_37: var_44}
    var_46 = module_0.from_json_schema(var_45)
    var_47 = var_46.schemas
    var_48 = len(var_47)
    assert var_48 == 2
    var_49 = 'anyOf'
    var_50 = {var_4: var_5}
    var_51 = {var_4: var_8}
    var_52 = [var_50, var_51]
    var_53 = {var_49: var_52}
    var_54 = module_0.from_json_schema(var_53)
    var_55 = var_54.schemas
    var_56 = len(var_55)
    assert var_56 == 2
    var_57 = 'oneOf'
    var_58 = {var_4: var_5}
    var_59 = {var_4: var_8}
    var_60 = [var_58, var_59]
    var_61 = {var_57: var_60}
    var_62 = module_0.from_json_schema(var_61)
    var_63 = var_62.schemas
    var_64 = len(var_63)
    assert var_64 == 2
    var_65 = 'not'
    var_66 = {var_4: var_5}
    var_67 = {var_65: var_66}
    var_68 = module_0.from_json_schema(var_67)
    var_69 = var_68.schema
    var_70 = 'if'
    var_71 = 'then'
    var_72 = 'else'
    var_73 = {var_4: var_5}
    var_74 = {var_38: var_39}
    var_75 = {var_4: var_8}
    var_76 = {var_70: var_73, var_71: var_74, var_72: var_75}
    var_77 = module_0.from_json_schema(var_76)
    var_78 = var_77.if_schema
    var_79 = var_77.then_schema
    var_80 = var_77.else_schema
    var_81 = module_1.Definitions()
    var_82 = '$ref'
    var_83 = '#/components/schemas/Test'
    var_84 = {var_82: var_83}
    var_85 = module_0.from_json_schema(var_84, var_81)
    var_86 = 'pattern'
    var_87 = '^[a-z]+$'
    var_88 = {var_4: var_5, var_38: var_39, var_41: var_42, var_86: var_87}
    var_89 = module_0.from_json_schema(var_88)
    var_90 = var_89.schemas
    var_91 = len(var_90)
    assert var_91 == 4
    var_92 = {}
    var_93 = module_0.from_json_schema(var_92)



# Parsed testcases at query #50
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
    var_84 = '^[a-z]+$'
    var_85 = {var_4: var_5, var_36: var_37, var_81: var_83, var_82: var_84}
    var_86 = module_0.from_json_schema(var_85)
    var_87 = {}
    var_88 = module_0.from_json_schema(var_87)



# Parsed testcases at query #51
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
    var_36 = 5
    var_37 = True
    var_38 = module_0.Array(var_35, min_items=var_4, max_items=var_36, unique_items=var_37)
    var_39 = 'items'
    var_40 = 'minItems'
    var_41 = 'maxItems'
    var_42 = 'uniqueItems'
    var_43 = 'array'
    var_44 = {var_9: var_14}
    var_45 = True
    var_46 = {var_9: var_43, var_39: var_44, var_40: var_37, var_41: var_36, var_42: var_45}
    var_47 = module_1.to_json_schema(var_38)
    var_48 = 'name'
    var_49 = module_0.String()
    var_50 = {var_48: var_49}
    var_51 = [var_48]
    var_52 = False
    var_53 = module_0.Object(properties=var_50, additional_properties=var_52, required=var_51)
    var_54 = 'properties'
    var_55 = 'required'
    var_56 = 'additionalProperties'
    var_57 = 'object'
    var_58 = {var_9: var_14}
    var_59 = {var_48: var_58}
    var_60 = [var_48]
    var_61 = False
    var_62 = {var_9: var_57, var_54: var_59, var_55: var_60, var_56: var_61}
    var_63 = module_1.to_json_schema(var_53)
    var_64 = 'a'
    var_65 = 'A'
    var_66 = (var_64, var_65)
    var_67 = 'b'
    var_68 = 'B'
    var_69 = (var_67, var_68)
    var_70 = [var_66, var_69]
    var_71 = module_0.Choice(choices=var_70)
    var_72 = 'enum'
    var_73 = [var_64, var_67]
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
    var_87 = {var_9: var_22}
    var_88 = [var_86, var_87]
    var_89 = {var_85: var_88}
    var_90 = module_1.to_json_schema(var_84)
    var_91 = module_0.String()
    var_92 = module_0.Integer()
    var_93 = [var_91, var_92]
    var_94 = module_2.OneOf(var_93)
    var_95 = 'oneOf'
    var_96 = {var_9: var_14}
    var_97 = {var_9: var_22}
    var_98 = [var_96, var_97]
    var_99 = {var_95: var_98}
    var_100 = module_1.to_json_schema(var_94)
    var_101 = module_0.String()
    var_102 = module_0.Integer()
    var_103 = [var_101, var_102]
    var_104 = module_2.AllOf(var_103)
    var_105 = 'allOf'
    var_106 = {var_9: var_14}
    var_107 = {var_9: var_22}
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
    var_119 = {var_9: var_22}
    var_120 = {var_9: var_32}
    var_121 = {var_115: var_118, var_116: var_119, var_117: var_120}
    var_122 = module_1.to_json_schema(var_114)
    var_123 = module_0.String()
    var_124 = module_2.Not(var_123)
    var_125 = 'not'
    var_126 = {var_9: var_14}
    var_127 = {var_125: var_126}
    var_128 = module_1.to_json_schema(var_124)
    var_129 = module_3.Definitions()
    var_130 = 'TestRef'
    var_131 = module_3.Reference(var_130, var_129)
    var_132 = '$ref'
    var_133 = '#/components/schemas/TestRef'
    var_134 = {var_132: var_133}
    var_135 = module_1.to_json_schema(var_131)
    var_136 = module_0.String()
    var_137 = {var_48: var_136}
    var_138 = [var_48]
    var_139 = module_3.Schema(var_137)
    var_140 = {var_9: var_14}
    var_141 = {var_48: var_140}
    var_142 = [var_48]
    var_143 = {var_9: var_57, var_54: var_141, var_55: var_142}
    var_144 = module_1.to_json_schema(var_139)
    var_145 = module_3.Definitions()
    var_146 = module_1.to_json_schema(var_145)



# Parsed testcases at query #52
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



# Parsed testcases at query #53
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
    var_102 = 'Test'
    var_103 = module_0.String()
    var_104 = {var_102: var_103}
    var_105 = '$ref'
    var_106 = 'components'
    var_107 = '#/components/schemas/Test'
    var_108 = 'schemas'
    var_109 = {var_9: var_14}
    var_110 = {var_102: var_109}
    var_111 = {var_108: var_110}
    var_112 = {var_105: var_107, var_106: var_111}
    var_113 = module_0.String()
    var_114 = {var_51: var_113}
    var_115 = [var_51]
    var_116 = module_3.Schema(var_114)
    var_117 = {var_9: var_14}
    var_118 = {var_51: var_117}
    var_119 = [var_51]
    var_120 = {var_9: var_60, var_56: var_118, var_57: var_119}
    var_121 = module_1.to_json_schema(var_116)
    var_122 = module_0.String()
    var_123 = module_0.Integer()
    var_124 = module_0.Boolean()
    var_125 = module_2.IfThenElse(var_122, var_123, var_124)
    var_126 = 'if'
    var_127 = 'then'
    var_128 = 'else'
    var_129 = {var_9: var_14}
    var_130 = {var_9: var_24}
    var_131 = {var_9: var_35}
    var_132 = {var_126: var_129, var_127: var_130, var_128: var_131}
    var_133 = module_1.to_json_schema(var_125)
    var_134 = module_0.String()
    var_135 = module_2.Not(var_134)
    var_136 = 'not'
    var_137 = {var_9: var_14}
    var_138 = {var_136: var_137}
    var_139 = module_1.to_json_schema(var_135)
    var_140 = 'StringField'
    var_141 = 'IntField'
    var_142 = module_0.String()
    var_143 = module_0.Integer()
    var_144 = {var_140: var_142, var_141: var_143}
    var_145 = {var_9: var_14}
    var_146 = {var_9: var_24}
    var_147 = {var_140: var_145, var_141: var_146}
    var_148 = {var_108: var_147}
    var_149 = {var_106: var_148}



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
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
    var_16 = {var_1: var_2}
    var_17 = {var_1: var_4}
    var_18 = 'integer'
    var_19 = {var_1: var_18}
    var_20 = [var_17, var_19]
    var_21 = {var_0: var_20}
    var_22 = [var_16, var_21]
    var_23 = {var_0: var_22}
    var_24 = module_0.Definitions()
    var_25 = module_1.any_of_from_json_schema(var_23, var_24)
    var_26 = var_25.any_of
    var_27 = len(var_26)
    assert var_27 == 2
    var_28 = var_25.any_of[var_12]
    var_29 = var_25.any_of[var_14]
    var_30 = var_25.any_of[var_14]
    var_31 = var_30.any_of
    var_32 = len(var_31)
    assert var_32 == 2
    var_33 = var_25.any_of[var_14]
    var_34 = var_33.any_of[var_12]
    var_35 = var_25.any_of[var_14]
    var_36 = var_35.any_of[var_14]
    var_37 = 'default'
    var_38 = {var_1: var_2}
    var_39 = {var_1: var_4}
    var_40 = [var_38, var_39]
    var_41 = 'test'
    var_42 = {var_0: var_40, var_37: var_41}
    var_43 = module_0.Definitions()
    var_44 = module_1.any_of_from_json_schema(var_42, var_43)
    var_45 = []
    var_46 = {var_0: var_45}
    var_47 = module_0.Definitions()
    var_48 = module_1.any_of_from_json_schema(var_46, var_47)
    var_49 = var_48.any_of
    var_50 = len(var_49)
    assert var_50 == 0



# Parsed testcases at query #2
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
    var_16 = True
    var_17 = 'text'
    var_18 = 123
    var_19 = [var_16, var_17, var_18]
    var_20 = {var_0: var_19}
    var_21 = module_0.Definitions()
    var_22 = module_1.enum_from_json_schema(var_20, var_21)



# Parsed testcases at query #3
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
    var_17 = [var_1, var_16]
    var_18 = {var_0: var_17}
    var_19 = module_0.Definitions()
    var_20 = module_1.type_from_json_schema(var_18, var_19)
    var_21 = {}
    var_22 = module_0.Definitions()
    var_23 = module_1.type_from_json_schema(var_21, var_22)
    var_24 = [var_16]
    var_25 = {var_0: var_24}
    var_26 = module_0.Definitions()
    var_27 = module_1.type_from_json_schema(var_25, var_26)
    var_28 = 'minLength'
    var_29 = 'maxLength'
    var_30 = 'pattern'
    var_31 = 5
    var_32 = 10
    var_33 = '^[A-Za-z]+$'
    var_34 = {var_0: var_1, var_28: var_31, var_29: var_32, var_30: var_33}
    var_35 = module_0.Definitions()
    var_36 = module_1.type_from_json_schema(var_34, var_35)
    var_37 = 'minimum'
    var_38 = 'maximum'
    var_39 = 'multipleOf'
    var_40 = 100
    var_41 = {var_0: var_5, var_37: var_12, var_38: var_40, var_39: var_31}
    var_42 = module_0.Definitions()
    var_43 = module_1.type_from_json_schema(var_41, var_42)
    var_44 = 'properties'
    var_45 = 'required'
    var_46 = 'object'
    var_47 = 'name'
    var_48 = 'age'
    var_49 = {var_0: var_1}
    var_50 = {var_0: var_5}
    var_51 = {var_47: var_49, var_48: var_50}
    var_52 = [var_47]
    var_53 = {var_0: var_46, var_44: var_51, var_45: var_52}
    var_54 = module_0.Definitions()
    var_55 = module_1.type_from_json_schema(var_53, var_54)
    var_56 = var_55.properties[var_47]
    var_57 = var_55.properties[var_48]
    var_58 = 'items'
    var_59 = 'minItems'
    var_60 = 'maxItems'
    var_61 = 'uniqueItems'
    var_62 = 'array'
    var_63 = {var_0: var_1}
    var_64 = True
    var_65 = {var_0: var_62, var_58: var_63, var_59: var_14, var_60: var_32, var_61: var_64}
    var_66 = module_0.Definitions()
    var_67 = module_1.type_from_json_schema(var_65, var_66)
    var_68 = var_67.items



# Parsed testcases at query #4
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



# Parsed testcases at query #5
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
    var_16 = True
    var_17 = False
    var_18 = None
    var_19 = [var_16, var_17, var_18]
    var_20 = {var_0: var_19}
    var_21 = module_0.Definitions()
    var_22 = module_1.enum_from_json_schema(var_20, var_21)



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
    var_28 = {var_4: var_5}
    var_29 = {var_1: var_28}
    var_30 = module_1.if_then_else_from_json_schema(var_29, var_0)
    var_31 = var_30.if_clause
    var_32 = 'default'
    var_33 = {var_4: var_5}
    var_34 = {var_4: var_7}
    var_35 = {var_4: var_9}
    var_36 = 42
    var_37 = {var_1: var_33, var_2: var_34, var_3: var_35, var_32: var_36}
    var_38 = module_1.if_then_else_from_json_schema(var_37, var_0)



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
    var_103 = 'Test'
    var_104 = module_0.String()
    var_105 = module_0.String()
    var_106 = {var_103: var_105}
    var_107 = module_3.Reference(var_103, var_106)
    var_108 = '$ref'
    var_109 = 'components'
    var_110 = '#/components/schemas/Test'
    var_111 = 'schemas'
    var_112 = {var_9: var_14}
    var_113 = {var_103: var_112}
    var_114 = {var_111: var_113}
    var_115 = {var_108: var_110, var_109: var_114}
    var_116 = module_1.to_json_schema(var_107)
    var_117 = module_0.String()
    var_118 = {var_51: var_117}
    var_119 = [var_51]
    var_120 = module_3.Schema(var_118)
    var_121 = {var_9: var_14}
    var_122 = {var_51: var_121}
    var_123 = [var_51]
    var_124 = {var_9: var_60, var_57: var_122, var_58: var_123}
    var_125 = module_1.to_json_schema(var_120)
    var_126 = 'StringField'
    var_127 = 'IntField'
    var_128 = module_0.String()
    var_129 = module_0.Integer()
    var_130 = {var_126: var_128, var_127: var_129}
    var_131 = {var_9: var_14}
    var_132 = {var_9: var_24}
    var_133 = {var_126: var_131, var_127: var_132}
    var_134 = {var_111: var_133}
    var_135 = {var_109: var_134}



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
    var_28 = {var_4: var_5}
    var_29 = {var_1: var_28}
    var_30 = module_1.if_then_else_from_json_schema(var_29, var_0)
    var_31 = var_30.if_clause
    var_32 = 'default'
    var_33 = {var_4: var_5}
    var_34 = {var_4: var_7}
    var_35 = {var_4: var_9}
    var_36 = 42
    var_37 = {var_1: var_33, var_2: var_34, var_3: var_35, var_32: var_36}
    var_38 = module_1.if_then_else_from_json_schema(var_37, var_0)



# Parsed testcases at query #9
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = '#/test'
    var_3 = {var_1: var_2}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)
    var_5 = '#/another'
    var_6 = {var_1: var_5}
    var_7 = module_1.ref_from_json_schema(var_6, var_0)



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
    var_16 = 'default'
    var_17 = 'minLength'
    var_18 = {var_2: var_3, var_17: var_14}
    var_19 = 'minimum'
    var_20 = {var_2: var_5, var_19: var_12}
    var_21 = 'boolean'
    var_22 = {var_2: var_21}
    var_23 = [var_18, var_20, var_22]
    var_24 = 'default_value'
    var_25 = {var_1: var_23, var_16: var_24}
    var_26 = module_1.one_of_from_json_schema(var_25, var_0)
    var_27 = var_26.one_of
    var_28 = len(var_27)
    assert var_28 == 3
    var_29 = var_26.one_of[var_12]
    var_30 = var_26.one_of[var_14]
    var_31 = 2
    var_32 = var_26.one_of[var_31]
    var_33 = {var_2: var_3}
    var_34 = '$ref'
    var_35 = '#/components/schemas/TestRef'
    var_36 = {var_34: var_35}
    var_37 = {var_2: var_5}
    var_38 = [var_36, var_37]
    var_39 = {var_1: var_38}
    var_40 = module_1.one_of_from_json_schema(var_39, var_0)
    var_41 = var_40.one_of
    var_42 = len(var_41)
    assert var_42 == 2
    var_43 = var_40.one_of[var_12]
    var_44 = var_40.one_of[var_14]



# Parsed testcases at query #12
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'oneOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'number'
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
    var_16 = module_0.Definitions()
    var_17 = '$ref'
    var_18 = '#/components/schemas/Test'
    var_19 = {var_17: var_18}
    var_20 = {var_1: var_4}
    var_21 = [var_19, var_20]
    var_22 = {var_0: var_21}
    var_23 = module_1.one_of_from_json_schema(var_22, var_16)
    var_24 = var_23.one_of
    var_25 = len(var_24)
    assert var_25 == 2
    var_26 = var_23.one_of[var_12]
    var_27 = var_23.one_of[var_14]
    var_28 = 'default'
    var_29 = {var_1: var_2}
    var_30 = {var_1: var_4}
    var_31 = [var_29, var_30]
    var_32 = 'test'
    var_33 = {var_0: var_31, var_28: var_32}
    var_34 = module_0.Definitions()
    var_35 = module_1.one_of_from_json_schema(var_33, var_34)



# Parsed testcases at query #13
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'oneOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'number'
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
    var_16 = 'properties'
    var_17 = 'object'
    var_18 = 'a'
    var_19 = {var_1: var_2}
    var_20 = {var_18: var_19}
    var_21 = {var_1: var_17, var_16: var_20}
    var_22 = 'b'
    var_23 = {var_1: var_4}
    var_24 = {var_22: var_23}
    var_25 = {var_1: var_17, var_16: var_24}
    var_26 = [var_21, var_25]
    var_27 = {var_0: var_26}
    var_28 = module_0.Definitions()
    var_29 = module_1.one_of_from_json_schema(var_27, var_28)
    var_30 = var_29.one_of
    var_31 = len(var_30)
    assert var_31 == 2
    var_32 = var_29.one_of[var_12]
    var_33 = var_29.one_of[var_14]
    var_34 = 'default'
    var_35 = {var_1: var_2}
    var_36 = {var_1: var_4}
    var_37 = [var_35, var_36]
    var_38 = 'test'
    var_39 = {var_0: var_37, var_34: var_38}
    var_40 = module_0.Definitions()
    var_41 = module_1.one_of_from_json_schema(var_39, var_40)
    var_42 = 'name'
    var_43 = 'age'
    var_44 = {var_1: var_2}
    var_45 = 'integer'
    var_46 = {var_1: var_45}
    var_47 = {var_42: var_44, var_43: var_46}
    var_48 = {var_1: var_17, var_16: var_47}
    var_49 = 'items'
    var_50 = 'array'
    var_51 = {var_1: var_2}
    var_52 = {var_1: var_50, var_49: var_51}
    var_53 = [var_48, var_52]
    var_54 = {var_0: var_53}
    var_55 = module_0.Definitions()
    var_56 = module_1.one_of_from_json_schema(var_54, var_55)
    var_57 = var_56.one_of
    var_58 = len(var_57)
    assert var_58 == 2
    var_59 = var_56.one_of[var_12]
    var_60 = var_56.one_of[var_14]



# Parsed testcases at query #14
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'oneOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'number'
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
    var_20 = 'test'
    var_21 = {var_0: var_19, var_16: var_20}
    var_22 = module_0.Definitions()
    var_23 = module_1.one_of_from_json_schema(var_21, var_22)
    var_24 = 'properties'
    var_25 = 'object'
    var_26 = 'name'
    var_27 = {var_1: var_2}
    var_28 = {var_26: var_27}
    var_29 = {var_1: var_25, var_24: var_28}
    var_30 = 'items'
    var_31 = 'array'
    var_32 = 'integer'
    var_33 = {var_1: var_32}
    var_34 = {var_1: var_31, var_30: var_33}
    var_35 = [var_29, var_34]
    var_36 = {var_0: var_35}
    var_37 = module_0.Definitions()
    var_38 = module_1.one_of_from_json_schema(var_36, var_37)
    var_39 = var_38.one_of
    var_40 = len(var_39)
    assert var_40 == 2
    var_41 = var_38.one_of[var_12]
    var_42 = var_38.one_of[var_14]
    var_43 = module_0.Definitions()
    var_44 = {var_1: var_2}
    var_45 = {var_26: var_44}
    var_46 = {var_1: var_25, var_24: var_45}
    var_47 = '$ref'
    var_48 = '#/components/schemas/Person'
    var_49 = {var_47: var_48}
    var_50 = {var_1: var_2}
    var_51 = [var_49, var_50]
    var_52 = {var_0: var_51}
    var_53 = module_1.one_of_from_json_schema(var_52, var_43)
    var_54 = var_53.one_of
    var_55 = len(var_54)
    assert var_55 == 2
    var_56 = var_53.one_of[var_12]
    var_57 = var_53.one_of[var_14]



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
    var_10 = 10
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
    var_46 = [var_29]
    var_47 = {var_0: var_44, var_40: var_45, var_41: var_35, var_42: var_10, var_43: var_35, var_6: var_46}
    var_48 = False
    var_49 = module_0.Definitions()
    var_50 = module_1.from_json_schema_type(var_47, var_44, var_48, var_49)
    var_51 = var_50.items
    var_52 = 'properties'
    var_53 = 'minProperties'
    var_54 = 'maxProperties'
    var_55 = 'required'
    var_56 = 'object'
    var_57 = 'name'
    var_58 = 'age'
    var_59 = {var_0: var_25}
    var_60 = {var_0: var_16}
    var_61 = {var_57: var_59, var_58: var_60}
    var_62 = [var_57]
    var_63 = 30
    var_64 = {var_57: var_29, var_58: var_63}
    var_65 = {var_0: var_56, var_52: var_61, var_53: var_35, var_54: var_10, var_55: var_62, var_6: var_64}
    var_66 = False
    var_67 = module_0.Definitions()
    var_68 = module_1.from_json_schema_type(var_65, var_56, var_66, var_67)
    var_69 = var_68.properties[var_57]
    var_70 = var_68.properties[var_58]



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
    var_72 = {var_0: var_25}
    var_73 = module_0.Definitions()
    var_74 = module_1.from_json_schema_type(var_72, var_25, var_35, var_73)



# Parsed testcases at query #17
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
    var_82 = 'pattern'
    var_83 = 10
    var_84 = '^[a-z]+$'
    var_85 = {var_4: var_5, var_36: var_37, var_81: var_83, var_82: var_84}
    var_86 = module_0.from_json_schema(var_85)
    var_87 = var_86.schemas
    var_88 = len(var_87)
    assert var_88 == 4
    var_89 = {}
    var_90 = module_0.from_json_schema(var_89)



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
    var_25 = module_0.Float(minimum=var_17, maximum=var_4)
    var_26 = 'number'
    var_27 = {var_9: var_26, var_20: var_17, var_21: var_4}
    var_28 = module_1.to_json_schema(var_25)
    var_29 = module_0.Boolean()
    var_30 = 'boolean'
    var_31 = {var_9: var_30}
    var_32 = module_1.to_json_schema(var_29)
    var_33 = module_0.String()
    var_34 = module_0.Array(var_33, min_items=var_4, max_items=var_5)
    var_35 = 'items'
    var_36 = 'minItems'
    var_37 = 'maxItems'
    var_38 = 'array'
    var_39 = {var_9: var_14}
    var_40 = {var_9: var_38, var_35: var_39, var_36: var_4, var_37: var_5}
    var_41 = module_1.to_json_schema(var_34)
    var_42 = 'name'
    var_43 = 'age'
    var_44 = module_0.String()
    var_45 = module_0.Integer()
    var_46 = {var_42: var_44, var_43: var_45}
    var_47 = module_0.Object(properties=var_46)
    var_48 = 'properties'
    var_49 = 'object'
    var_50 = {var_9: var_14}
    var_51 = {var_9: var_22}
    var_52 = {var_42: var_50, var_43: var_51}
    var_53 = {var_9: var_49, var_48: var_52}
    var_54 = module_1.to_json_schema(var_47)
    var_55 = 'a'
    var_56 = (var_55, var_55)
    var_57 = 'b'
    var_58 = (var_57, var_57)
    var_59 = [var_56, var_58]
    var_60 = module_0.Choice(choices=var_59)
    var_61 = 'enum'
    var_62 = [var_55, var_57]
    var_63 = {var_61: var_62}
    var_64 = module_1.to_json_schema(var_60)
    var_65 = 'fixed_value'
    var_66 = module_0.Const(var_65)
    var_67 = 'const'
    var_68 = {var_67: var_65}
    var_69 = module_1.to_json_schema(var_66)
    var_70 = module_0.String()
    var_71 = module_0.Integer()
    var_72 = [var_70, var_71]
    var_73 = module_0.Union(var_72)
    var_74 = 'anyOf'
    var_75 = {var_9: var_14}
    var_76 = {var_9: var_22}
    var_77 = [var_75, var_76]
    var_78 = {var_74: var_77}
    var_79 = module_1.to_json_schema(var_73)
    var_80 = module_0.String()
    var_81 = module_0.Integer()
    var_82 = [var_80, var_81]
    var_83 = module_2.AllOf(var_82)
    var_84 = 'allOf'
    var_85 = {var_9: var_14}
    var_86 = {var_9: var_22}
    var_87 = [var_85, var_86]
    var_88 = {var_84: var_87}
    var_89 = module_1.to_json_schema(var_83)
    var_90 = module_0.String()
    var_91 = module_0.Integer()
    var_92 = [var_90, var_91]
    var_93 = module_2.OneOf(var_92)
    var_94 = 'oneOf'
    var_95 = {var_9: var_14}
    var_96 = {var_9: var_22}
    var_97 = [var_95, var_96]
    var_98 = {var_94: var_97}
    var_99 = module_1.to_json_schema(var_93)
    var_100 = module_0.String()
    var_101 = module_2.Not(var_100)
    var_102 = 'not'
    var_103 = {var_9: var_14}
    var_104 = {var_102: var_103}
    var_105 = module_1.to_json_schema(var_101)
    var_106 = module_0.String()
    var_107 = module_0.Integer()
    var_108 = module_0.Boolean()
    var_109 = module_2.IfThenElse(var_106, var_107, var_108)
    var_110 = 'if'
    var_111 = 'then'
    var_112 = 'else'
    var_113 = {var_9: var_14}
    var_114 = {var_9: var_22}
    var_115 = {var_9: var_30}
    var_116 = {var_110: var_113, var_111: var_114, var_112: var_115}
    var_117 = module_1.to_json_schema(var_109)
    var_118 = 'Person'
    var_119 = module_0.String()
    var_120 = {var_42: var_119}
    var_121 = module_0.Object(properties=var_120)
    var_122 = {var_118: var_121}
    var_123 = '$ref'
    var_124 = 'components'
    var_125 = '#/components/schemas/Person'
    var_126 = 'schemas'
    var_127 = {var_9: var_14}
    var_128 = {var_42: var_127}
    var_129 = {var_9: var_49, var_48: var_128}
    var_130 = {var_118: var_129}
    var_131 = {var_126: var_130}
    var_132 = {var_123: var_125, var_124: var_131}
    var_133 = module_0.String()
    var_134 = module_0.Integer()
    var_135 = {var_42: var_133, var_43: var_134}
    var_136 = module_3.Schema(var_135)
    var_137 = {var_9: var_14}
    var_138 = {var_9: var_22}
    var_139 = {var_42: var_137, var_43: var_138}
    var_140 = {var_9: var_49, var_48: var_139}
    var_141 = module_1.to_json_schema(var_136)



# Parsed testcases at query #19
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
    var_73 = module_1.Definitions()
    var_74 = {var_4: var_5}
    var_75 = '$ref'
    var_76 = '#/components/schemas/Test'
    var_77 = {var_75: var_76}
    var_78 = module_0.from_json_schema(var_77, var_73)
    var_79 = 'maxLength'
    var_80 = 10
    var_81 = {var_4: var_5, var_36: var_37, var_79: var_80}
    var_82 = module_0.from_json_schema(var_81)
    var_83 = var_82.constraints
    var_84 = len(var_83)
    assert var_84 == 2
    var_85 = {}
    var_86 = module_0.from_json_schema(var_85)



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
    var_6 = True
    var_7 = module_0.String(max_length=var_5, min_length=var_4)
    var_8 = 'type'
    var_9 = 'minLength'
    var_10 = 'maxLength'
    var_11 = 'string'
    var_12 = 'null'
    var_13 = [var_11, var_12]
    var_14 = {var_8: var_13, var_9: var_6, var_10: var_5}
    var_15 = module_1.to_json_schema(var_7)
    var_16 = 0
    var_17 = 100
    var_18 = False
    var_19 = module_0.Integer(minimum=var_16, maximum=var_17)
    var_20 = 'minimum'
    var_21 = 'maximum'
    var_22 = 'integer'
    var_23 = {var_8: var_22, var_20: var_18, var_21: var_17}
    var_24 = module_1.to_json_schema(var_19)
    var_25 = 0.5
    var_26 = True
    var_27 = module_0.Float(multiple_of=var_25)
    var_28 = 'multipleOf'
    var_29 = 'number'
    var_30 = [var_29, var_12]
    var_31 = {var_8: var_30, var_28: var_25}
    var_32 = module_1.to_json_schema(var_27)
    var_33 = True
    var_34 = module_0.Boolean()
    var_35 = 'boolean'
    var_36 = [var_35, var_12]
    var_37 = {var_8: var_36}
    var_38 = module_1.to_json_schema(var_34)
    var_39 = module_0.String()
    var_40 = False
    var_41 = module_0.Array(var_39, min_items=var_33)
    var_42 = 'minItems'
    var_43 = 'items'
    var_44 = 'array'
    var_45 = {var_8: var_11}
    var_46 = {var_8: var_44, var_42: var_33, var_43: var_45}
    var_47 = module_1.to_json_schema(var_41)
    var_48 = 'name'
    var_49 = module_0.String()
    var_50 = {var_48: var_49}
    var_51 = True
    var_52 = module_0.Object(properties=var_50)
    var_53 = 'properties'
    var_54 = 'object'
    var_55 = [var_54, var_12]
    var_56 = {var_8: var_11}
    var_57 = {var_48: var_56}
    var_58 = {var_8: var_55, var_53: var_57}
    var_59 = module_1.to_json_schema(var_52)
    var_60 = 'a'
    var_61 = (var_60, var_60)
    var_62 = 'b'
    var_63 = (var_62, var_62)
    var_64 = [var_61, var_63]
    var_65 = module_0.Choice(choices=var_64)
    var_66 = 'enum'
    var_67 = [var_60, var_62]
    var_68 = {var_66: var_67}
    var_69 = module_1.to_json_schema(var_65)
    var_70 = 'fixed'
    var_71 = module_0.Const(var_70)
    var_72 = 'const'
    var_73 = {var_72: var_70}
    var_74 = module_1.to_json_schema(var_71)
    var_75 = module_0.String()
    var_76 = module_0.Integer()
    var_77 = [var_75, var_76]
    var_78 = module_0.Union(var_77)
    var_79 = 'anyOf'
    var_80 = {var_8: var_11}
    var_81 = {var_8: var_22}
    var_82 = [var_80, var_81]
    var_83 = {var_79: var_82}
    var_84 = module_1.to_json_schema(var_78)
    var_85 = module_0.String()
    var_86 = 'test'
    var_87 = module_0.Const(var_86)
    var_88 = [var_85, var_87]
    var_89 = module_2.AllOf(var_88)
    var_90 = 'allOf'
    var_91 = {var_8: var_11}
    var_92 = {var_72: var_86}
    var_93 = [var_91, var_92]
    var_94 = {var_90: var_93}
    var_95 = module_1.to_json_schema(var_89)
    var_96 = module_3.Definitions()
    var_97 = module_3.Reference(var_86, var_96)
    var_98 = '$ref'
    var_99 = 'components'
    var_100 = '#/components/schemas/test'
    var_101 = 'schemas'
    var_102 = {}
    var_103 = {var_101: var_102}
    var_104 = {var_98: var_100, var_99: var_103}
    var_105 = module_1.to_json_schema(var_97)
    var_106 = module_0.String()
    var_107 = {var_48: var_106}
    var_108 = True
    var_109 = module_3.Schema(var_107)
    var_110 = [var_54, var_12]
    var_111 = {var_8: var_11}
    var_112 = {var_48: var_111}
    var_113 = {var_8: var_110, var_53: var_112}
    var_114 = module_1.to_json_schema(var_109)
    var_115 = module_0.String()
    var_116 = module_0.Integer()
    var_117 = module_2.IfThenElse(var_115, var_116)
    var_118 = 'if'
    var_119 = 'then'
    var_120 = {var_8: var_11}
    var_121 = {var_8: var_22}
    var_122 = {var_118: var_120, var_119: var_121}
    var_123 = module_1.to_json_schema(var_117)
    var_124 = module_0.String()
    var_125 = module_2.Not(var_124)
    var_126 = 'not'
    var_127 = {var_8: var_11}
    var_128 = {var_126: var_127}
    var_129 = module_1.to_json_schema(var_125)



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
    var_39 = 'items'
    var_40 = 'minItems'
    var_41 = 'maxItems'
    var_42 = 'uniqueItems'
    var_43 = 'array'
    var_44 = {var_9: var_14}
    var_45 = True
    var_46 = {var_9: var_43, var_39: var_44, var_40: var_37, var_41: var_36, var_42: var_45}
    var_47 = module_1.to_json_schema(var_38)
    var_48 = 'name'
    var_49 = module_0.String()
    var_50 = {var_48: var_49}
    var_51 = [var_48]
    var_52 = module_0.Object(properties=var_50, required=var_51)
    var_53 = 'properties'
    var_54 = 'required'
    var_55 = 'object'
    var_56 = {var_9: var_14}
    var_57 = {var_48: var_56}
    var_58 = [var_48]
    var_59 = {var_9: var_55, var_53: var_57, var_54: var_58}
    var_60 = module_1.to_json_schema(var_52)
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
    var_71 = 'value'
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
    var_87 = module_0.Integer()
    var_88 = [var_86, var_87]
    var_89 = module_2.OneOf(var_88)
    var_90 = 'oneOf'
    var_91 = {var_9: var_14}
    var_92 = {var_9: var_22}
    var_93 = [var_91, var_92]
    var_94 = {var_90: var_93}
    var_95 = module_1.to_json_schema(var_89)
    var_96 = module_0.String()
    var_97 = module_0.Integer()
    var_98 = [var_96, var_97]
    var_99 = module_2.AllOf(var_98)
    var_100 = 'allOf'
    var_101 = {var_9: var_14}
    var_102 = {var_9: var_22}
    var_103 = [var_101, var_102]
    var_104 = {var_100: var_103}
    var_105 = module_1.to_json_schema(var_99)
    var_106 = module_0.String()
    var_107 = module_0.Integer()
    var_108 = module_0.Boolean()
    var_109 = module_2.IfThenElse(var_106, var_107, var_108)
    var_110 = 'if'
    var_111 = 'then'
    var_112 = 'else'
    var_113 = {var_9: var_14}
    var_114 = {var_9: var_22}
    var_115 = {var_9: var_32}
    var_116 = {var_110: var_113, var_111: var_114, var_112: var_115}
    var_117 = module_1.to_json_schema(var_109)
    var_118 = module_0.String()
    var_119 = module_2.Not(var_118)
    var_120 = 'not'
    var_121 = {var_9: var_14}
    var_122 = {var_120: var_121}
    var_123 = module_1.to_json_schema(var_119)
    var_124 = module_3.Definitions()
    var_125 = 'test'
    var_126 = module_0.String()
    var_127 = module_3.Reference(var_125, var_124)
    var_128 = '$ref'
    var_129 = 'components'
    var_130 = '#/components/schemas/test'
    var_131 = 'schemas'
    var_132 = {var_9: var_14}
    var_133 = {var_125: var_132}
    var_134 = {var_131: var_133}
    var_135 = {var_128: var_130, var_129: var_134}
    var_136 = module_1.to_json_schema(var_127)
    var_137 = module_0.String()
    var_138 = {var_48: var_137}
    var_139 = [var_48]
    var_140 = module_3.Schema(var_138)
    var_141 = {var_9: var_14}
    var_142 = {var_48: var_141}
    var_143 = [var_48]
    var_144 = {var_9: var_55, var_53: var_142, var_54: var_143}
    var_145 = module_1.to_json_schema(var_140)



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
    var_102 = 'Test'
    var_103 = module_0.String()
    var_104 = {var_102: var_103}
    var_105 = '$ref'
    var_106 = 'components'
    var_107 = '#/components/schemas/Test'
    var_108 = 'schemas'
    var_109 = {var_9: var_14}
    var_110 = {var_102: var_109}
    var_111 = {var_108: var_110}
    var_112 = {var_105: var_107, var_106: var_111}
    var_113 = module_0.String()
    var_114 = {var_51: var_113}
    var_115 = [var_51]
    var_116 = module_3.Schema(var_114)
    var_117 = {var_9: var_14}
    var_118 = {var_51: var_117}
    var_119 = [var_51]
    var_120 = {var_9: var_60, var_56: var_118, var_57: var_119}
    var_121 = module_1.to_json_schema(var_116)
    var_122 = module_0.String()
    var_123 = module_0.Integer()
    var_124 = module_0.Boolean()
    var_125 = module_2.IfThenElse(var_122, var_123, var_124)
    var_126 = 'if'
    var_127 = 'then'
    var_128 = 'else'
    var_129 = {var_9: var_14}
    var_130 = {var_9: var_24}
    var_131 = {var_9: var_35}
    var_132 = {var_126: var_129, var_127: var_130, var_128: var_131}
    var_133 = module_1.to_json_schema(var_125)
    var_134 = module_0.String()
    var_135 = module_2.Not(var_134)
    var_136 = 'not'
    var_137 = {var_9: var_14}
    var_138 = {var_136: var_137}
    var_139 = module_1.to_json_schema(var_135)
    var_140 = 'Person'
    var_141 = module_0.String()
    var_142 = {var_51: var_141}
    var_143 = module_0.Object(properties=var_142)
    var_144 = {var_140: var_143}
    var_145 = {var_9: var_14}
    var_146 = {var_51: var_145}
    var_147 = {var_9: var_60, var_56: var_146}
    var_148 = {var_140: var_147}
    var_149 = {var_108: var_148}
    var_150 = {var_106: var_149}



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
    var_72 = {var_38: var_39}
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
    var_84 = '^[a-zA-Z]+$'
    var_85 = {var_4: var_5, var_35: var_36, var_38: var_39, var_83: var_84}
    var_86 = module_0.from_json_schema(var_85)
    var_87 = var_86.schemas
    var_88 = len(var_87)
    assert var_88 == 4
    var_89 = {}
    var_90 = module_0.from_json_schema(var_89)



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
    var_42 = 'minItems'
    var_43 = 'maxItems'
    var_44 = 'uniqueItems'
    var_45 = 'array'
    var_46 = {var_0: var_25}
    var_47 = [var_30]
    var_48 = {var_0: var_45, var_41: var_46, var_42: var_36, var_43: var_26, var_44: var_36, var_6: var_47}
    var_49 = False
    var_50 = module_0.Definitions()
    var_51 = module_1.from_json_schema_type(var_48, var_45, var_49, var_50)
    var_52 = var_51.items
    var_53 = 'properties'
    var_54 = 'minProperties'
    var_55 = 'maxProperties'
    var_56 = 'required'
    var_57 = 'object'
    var_58 = 'name'
    var_59 = 'age'
    var_60 = {var_0: var_25}
    var_61 = {var_0: var_16}
    var_62 = {var_58: var_60, var_59: var_61}
    var_63 = [var_58]
    var_64 = 25
    var_65 = {var_58: var_30, var_59: var_64}
    var_66 = {var_0: var_57, var_53: var_62, var_54: var_36, var_55: var_10, var_56: var_63, var_6: var_65}
    var_67 = False
    var_68 = module_0.Definitions()
    var_69 = module_1.from_json_schema_type(var_66, var_57, var_67, var_68)
    var_70 = var_69.properties[var_58]
    var_71 = var_69.properties[var_59]



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
    var_130 = 'Test'
    var_131 = module_3.Reference(var_130, var_129)
    var_132 = 'components'
    var_133 = '$ref'
    var_134 = 'schemas'
    var_135 = {var_9: var_14}
    var_136 = {var_130: var_135}
    var_137 = {var_134: var_136}
    var_138 = '#/components/schemas/Test'
    var_139 = {var_132: var_137, var_133: var_138}
    var_140 = module_1.to_json_schema(var_131)
    var_141 = module_0.String()
    var_142 = {var_51: var_141}
    var_143 = [var_51]
    var_144 = module_3.Schema(var_142)
    var_145 = {var_9: var_14}
    var_146 = {var_51: var_145}
    var_147 = [var_51]
    var_148 = {var_9: var_60, var_56: var_146, var_57: var_147}
    var_149 = module_1.to_json_schema(var_144)



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
    var_42 = 'a'
    var_43 = 'b'
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



# Parsed testcases at query #27
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
    var_18 = 5
    var_19 = 10
    var_20 = {var_0: var_17, var_15: var_18, var_16: var_19}
    var_21 = False
    var_22 = module_0.Definitions()
    var_23 = module_1.from_json_schema_type(var_20, var_17, var_21, var_22)
    var_24 = 'boolean'
    var_25 = {var_0: var_24}
    var_26 = False
    var_27 = module_0.Definitions()
    var_28 = module_1.from_json_schema_type(var_25, var_24, var_26, var_27)
    var_29 = 'items'
    var_30 = 'minItems'
    var_31 = 'maxItems'
    var_32 = 'array'
    var_33 = {var_0: var_17}
    var_34 = 1
    var_35 = {var_0: var_32, var_29: var_33, var_30: var_34, var_31: var_18}
    var_36 = False
    var_37 = module_0.Definitions()
    var_38 = module_1.from_json_schema_type(var_35, var_32, var_36, var_37)
    var_39 = var_38.items
    var_40 = 'properties'
    var_41 = 'required'
    var_42 = 'object'
    var_43 = 'name'
    var_44 = {var_0: var_17}
    var_45 = {var_43: var_44}
    var_46 = [var_43]
    var_47 = {var_0: var_42, var_40: var_45, var_41: var_46}
    var_48 = False
    var_49 = module_0.Definitions()
    var_50 = module_1.from_json_schema_type(var_47, var_42, var_48, var_49)
    var_51 = var_50.properties[var_43]
    var_52 = {var_0: var_17}
    var_53 = True
    var_54 = module_0.Definitions()
    var_55 = module_1.from_json_schema_type(var_52, var_17, var_53, var_54)
    var_56 = 'default'
    var_57 = 'test'
    var_58 = {var_0: var_17, var_56: var_57}
    var_59 = False
    var_60 = module_0.Definitions()
    var_61 = module_1.from_json_schema_type(var_58, var_17, var_59, var_60)



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
    var_20 = module_0.Integer(minimum=var_17, maximum=var_18, exclusive_minimum=var_19)
    var_21 = 'minimum'
    var_22 = 'exclusiveMinimum'
    var_23 = 'maximum'
    var_24 = 'integer'
    var_25 = True
    var_26 = {var_9: var_24, var_21: var_17, var_22: var_25, var_23: var_18}
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
    var_76 = 'value'
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
    var_92 = module_0.Const(var_76)
    var_93 = [var_91, var_92]
    var_94 = module_2.AllOf(var_93)
    var_95 = 'allOf'
    var_96 = {var_9: var_14}
    var_97 = {var_78: var_76}
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
    var_129 = 'Test'
    var_130 = module_0.String()
    var_131 = {var_129: var_130}
    var_132 = '$ref'
    var_133 = 'components'
    var_134 = '#/components/schemas/Test'
    var_135 = 'schemas'
    var_136 = {var_9: var_14}
    var_137 = {var_129: var_136}
    var_138 = {var_135: var_137}
    var_139 = {var_132: var_134, var_133: var_138}
    var_140 = module_1.to_json_schema(var_120)
    var_141 = module_0.String()
    var_142 = {var_51: var_141}
    var_143 = [var_51]
    var_144 = module_3.Schema(var_142)
    var_145 = {var_9: var_14}
    var_146 = {var_51: var_145}
    var_147 = [var_51]
    var_148 = {var_9: var_60, var_56: var_146, var_57: var_147}
    var_149 = module_1.to_json_schema(var_144)
    var_150 = 'StringField'
    var_151 = 'IntegerField'
    var_152 = module_0.String()
    var_153 = module_0.Integer()
    var_154 = {var_150: var_152, var_151: var_153}
    var_155 = {var_9: var_14}
    var_156 = {var_9: var_24}
    var_157 = {var_150: var_155, var_151: var_156}
    var_158 = {var_135: var_157}
    var_159 = {var_133: var_158}



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
    var_69 = {var_0: var_59, var_55: var_64, var_56: var_65, var_57: var_35, var_58: var_46, var_6: var_68}
    var_70 = False
    var_71 = module_0.Definitions()
    var_72 = module_1.from_json_schema_type(var_69, var_59, var_70, var_71)
    var_73 = var_72.properties[var_60]
    var_74 = var_72.properties[var_61]
    var_75 = {var_0: var_25}
    var_76 = module_0.Definitions()
    var_77 = module_1.from_json_schema_type(var_75, var_25, var_35, var_76)



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
    var_22 = module_0.Integer(minimum=var_19, maximum=var_20, exclusive_minimum=var_4, multiple_of=var_21)
    var_23 = 'minimum'
    var_24 = 'maximum'
    var_25 = 'exclusiveMinimum'
    var_26 = 'multipleOf'
    var_27 = 'integer'
    var_28 = {var_9: var_27, var_23: var_19, var_24: var_20, var_25: var_4, var_26: var_21}
    var_29 = module_1.to_json_schema(var_22)
    var_30 = module_0.Float(minimum=var_19, maximum=var_4, exclusive_maximum=var_4)
    var_31 = 'exclusiveMaximum'
    var_32 = 'number'
    var_33 = [var_32, var_15]
    var_34 = {var_9: var_33, var_23: var_19, var_24: var_4, var_31: var_4}
    var_35 = module_1.to_json_schema(var_30)
    var_36 = module_0.Boolean()
    var_37 = 'boolean'
    var_38 = {var_9: var_37}
    var_39 = module_1.to_json_schema(var_36)
    var_40 = 5
    var_41 = module_0.String()
    var_42 = module_0.Array(var_41, var_19, var_4, var_40, unique_items=var_4)
    var_43 = 'minItems'
    var_44 = 'maxItems'
    var_45 = 'items'
    var_46 = 'additionalItems'
    var_47 = 'uniqueItems'
    var_48 = 'array'
    var_49 = [var_48, var_15]
    var_50 = {var_9: var_14}
    var_51 = {var_9: var_49, var_43: var_4, var_44: var_40, var_45: var_50, var_46: var_19, var_47: var_4}
    var_52 = module_1.to_json_schema(var_42)
    var_53 = 'name'
    var_54 = module_0.String()
    var_55 = {var_53: var_54}
    var_56 = '^S_'
    var_57 = module_0.String()
    var_58 = {var_56: var_57}
    var_59 = module_0.String()
    var_60 = [var_53]
    var_61 = module_0.Object(properties=var_55, pattern_properties=var_58, additional_properties=var_4, property_names=var_59, min_properties=var_4, max_properties=var_5, required=var_60)
    var_62 = 'properties'
    var_63 = 'patternProperties'
    var_64 = 'additionalProperties'
    var_65 = 'propertyNames'
    var_66 = 'minProperties'
    var_67 = 'maxProperties'
    var_68 = 'required'
    var_69 = 'object'
    var_70 = {var_9: var_14}
    var_71 = {var_53: var_70}
    var_72 = {var_9: var_14}
    var_73 = {var_56: var_72}
    var_74 = {var_9: var_14}
    var_75 = [var_53]
    var_76 = {var_9: var_69, var_62: var_71, var_63: var_73, var_64: var_4, var_65: var_74, var_66: var_4, var_67: var_5, var_68: var_75}
    var_77 = module_1.to_json_schema(var_61)
    var_78 = 'a'
    var_79 = (var_78, var_78)
    var_80 = 'b'
    var_81 = (var_80, var_80)
    var_82 = [var_79, var_81]
    var_83 = module_0.Choice(choices=var_82)
    var_84 = 'enum'
    var_85 = 'default'
    var_86 = [var_78, var_80]
    var_87 = {var_84: var_86, var_85: var_78}
    var_88 = module_1.to_json_schema(var_83)
    var_89 = 'fixed_value'
    var_90 = module_0.Const(var_89)
    var_91 = 'const'
    var_92 = {var_91: var_89, var_85: var_89}
    var_93 = module_1.to_json_schema(var_90)
    var_94 = module_0.String()
    var_95 = module_0.Integer()
    var_96 = [var_94, var_95]
    var_97 = module_0.Union(var_96)
    var_98 = 'anyOf'
    var_99 = {var_9: var_14}
    var_100 = {var_9: var_27}
    var_101 = [var_99, var_100]
    var_102 = {var_98: var_101}
    var_103 = module_1.to_json_schema(var_97)
    var_104 = module_0.String()
    var_105 = module_0.Integer()
    var_106 = [var_104, var_105]
    var_107 = module_2.OneOf(var_106)
    var_108 = 'oneOf'
    var_109 = {var_9: var_14}
    var_110 = {var_9: var_27}
    var_111 = [var_109, var_110]
    var_112 = {var_108: var_111}
    var_113 = module_1.to_json_schema(var_107)
    var_114 = module_0.String()
    var_115 = module_0.Integer()
    var_116 = [var_114, var_115]
    var_117 = module_2.AllOf(var_116)
    var_118 = 'allOf'
    var_119 = {var_9: var_14}
    var_120 = {var_9: var_27}
    var_121 = [var_119, var_120]
    var_122 = {var_118: var_121}
    var_123 = module_1.to_json_schema(var_117)
    var_124 = module_0.String()
    var_125 = module_0.Integer()
    var_126 = module_0.Boolean()
    var_127 = module_2.IfThenElse(var_124, var_125, var_126)
    var_128 = 'if'
    var_129 = 'then'
    var_130 = 'else'
    var_131 = {var_9: var_14}
    var_132 = {var_9: var_27}
    var_133 = {var_9: var_37}
    var_134 = {var_128: var_131, var_129: var_132, var_130: var_133}
    var_135 = module_1.to_json_schema(var_127)
    var_136 = module_0.String()
    var_137 = module_2.Not(var_136)
    var_138 = 'not'
    var_139 = {var_9: var_14}
    var_140 = {var_138: var_139}
    var_141 = module_1.to_json_schema(var_137)
    var_142 = 'Person'
    var_143 = module_0.String()
    var_144 = {var_53: var_143}
    var_145 = module_0.Object(properties=var_144)
    var_146 = {var_142: var_145}
    var_147 = '$ref'
    var_148 = 'components'
    var_149 = '#/components/schemas/Person'
    var_150 = 'schemas'
    var_151 = {var_9: var_14}
    var_152 = {var_53: var_151}
    var_153 = {var_9: var_69, var_62: var_152}
    var_154 = {var_142: var_153}
    var_155 = {var_150: var_154}
    var_156 = {var_147: var_149, var_148: var_155}
    var_157 = module_0.String()
    var_158 = {var_53: var_157}
    var_159 = [var_53]
    var_160 = module_3.Schema(var_158)
    var_161 = {var_9: var_14}
    var_162 = {var_53: var_161}
    var_163 = [var_53]
    var_164 = {var_9: var_69, var_62: var_162, var_68: var_163}
    var_165 = module_1.to_json_schema(var_160)



# Parsed testcases at query #31
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
    var_23 = 10
    var_24 = '^[A-Za-z]+$'
    var_25 = 'hello'
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
    var_39 = 'uniqueItems'
    var_40 = 'array'
    var_41 = {var_0: var_21}
    var_42 = 'a'
    var_43 = 'b'
    var_44 = [var_42, var_43]
    var_45 = {var_0: var_40, var_36: var_41, var_37: var_31, var_38: var_22, var_39: var_31, var_3: var_44}
    var_46 = False
    var_47 = module_0.Definitions()
    var_48 = module_1.from_json_schema_type(var_45, var_40, var_46, var_47)
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
    var_70 = module_1.from_json_schema_type(var_68, var_21, var_31, var_69)



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
    var_39 = 'items'
    var_40 = 'minItems'
    var_41 = 'maxItems'
    var_42 = 'uniqueItems'
    var_43 = 'array'
    var_44 = {var_9: var_14}
    var_45 = True
    var_46 = {var_9: var_43, var_39: var_44, var_40: var_37, var_41: var_36, var_42: var_45}
    var_47 = module_1.to_json_schema(var_38)
    var_48 = 'name'
    var_49 = module_0.String()
    var_50 = {var_48: var_49}
    var_51 = [var_48]
    var_52 = module_0.Object(properties=var_50, min_properties=var_45, max_properties=var_36, required=var_51)
    var_53 = 'properties'
    var_54 = 'required'
    var_55 = 'minProperties'
    var_56 = 'maxProperties'
    var_57 = 'object'
    var_58 = {var_9: var_14}
    var_59 = {var_48: var_58}
    var_60 = [var_48]
    var_61 = {var_9: var_57, var_53: var_59, var_54: var_60, var_55: var_45, var_56: var_36}
    var_62 = module_1.to_json_schema(var_52)
    var_63 = 'a'
    var_64 = (var_63, var_63)
    var_65 = 'b'
    var_66 = (var_65, var_65)
    var_67 = [var_64, var_66]
    var_68 = module_0.Choice(choices=var_67)
    var_69 = 'enum'
    var_70 = [var_63, var_65]
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
    var_89 = 'test'
    var_90 = module_0.Const(var_89)
    var_91 = [var_88, var_90]
    var_92 = module_2.AllOf(var_91)
    var_93 = 'allOf'
    var_94 = {var_9: var_14}
    var_95 = {var_75: var_89}
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
    var_110 = module_2.Not(var_109)
    var_111 = 'not'
    var_112 = {var_9: var_14}
    var_113 = {var_111: var_112}
    var_114 = module_1.to_json_schema(var_110)
    var_115 = module_0.String()
    var_116 = module_0.Integer()
    var_117 = module_0.Boolean()
    var_118 = module_2.IfThenElse(var_115, var_116, var_117)
    var_119 = 'if'
    var_120 = 'then'
    var_121 = 'else'
    var_122 = {var_9: var_14}
    var_123 = {var_9: var_22}
    var_124 = {var_9: var_32}
    var_125 = {var_119: var_122, var_120: var_123, var_121: var_124}
    var_126 = module_1.to_json_schema(var_118)
    var_127 = module_3.Definitions()
    var_128 = module_3.Reference(var_89, var_127)
    var_129 = '$ref'
    var_130 = 'components'
    var_131 = '#/components/schemas/test'
    var_132 = 'schemas'
    var_133 = {var_9: var_14}
    var_134 = {var_89: var_133}
    var_135 = {var_132: var_134}
    var_136 = {var_129: var_131, var_130: var_135}
    var_137 = module_1.to_json_schema(var_128)
    var_138 = module_0.String()
    var_139 = {var_48: var_138}
    var_140 = [var_48]
    var_141 = module_3.Schema(var_139)
    var_142 = {var_9: var_14}
    var_143 = {var_48: var_142}
    var_144 = [var_48]
    var_145 = {var_9: var_57, var_53: var_143, var_54: var_144}
    var_146 = module_1.to_json_schema(var_141)



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
    var_76 = 30
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



# Parsed testcases at query #34
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
    var_18 = 5
    var_19 = 10
    var_20 = {var_0: var_17, var_15: var_18, var_16: var_19}
    var_21 = False
    var_22 = module_0.Definitions()
    var_23 = module_1.from_json_schema_type(var_20, var_17, var_21, var_22)
    var_24 = 'boolean'
    var_25 = {var_0: var_24}
    var_26 = False
    var_27 = module_0.Definitions()
    var_28 = module_1.from_json_schema_type(var_25, var_24, var_26, var_27)
    var_29 = 'items'
    var_30 = 'minItems'
    var_31 = 'array'
    var_32 = {var_0: var_17}
    var_33 = 1
    var_34 = {var_0: var_31, var_29: var_32, var_30: var_33}
    var_35 = False
    var_36 = module_0.Definitions()
    var_37 = module_1.from_json_schema_type(var_34, var_31, var_35, var_36)
    var_38 = var_37.items
    var_39 = 'properties'
    var_40 = 'required'
    var_41 = 'object'
    var_42 = 'name'
    var_43 = {var_0: var_17}
    var_44 = {var_42: var_43}
    var_45 = [var_42]
    var_46 = {var_0: var_41, var_39: var_44, var_40: var_45}
    var_47 = False
    var_48 = module_0.Definitions()
    var_49 = module_1.from_json_schema_type(var_46, var_41, var_47, var_48)
    var_50 = var_49.properties[var_42]
    var_51 = {var_0: var_17}
    var_52 = True
    var_53 = module_0.Definitions()
    var_54 = module_1.from_json_schema_type(var_51, var_17, var_52, var_53)
    var_55 = 'default'
    var_56 = 'test'
    var_57 = {var_0: var_17, var_55: var_56}
    var_58 = False
    var_59 = module_0.Definitions()
    var_60 = module_1.from_json_schema_type(var_57, var_17, var_58, var_59)
    var_61 = 'type'
    var_62 = 'invalid'
    var_63 = {var_61: var_62}
    var_64 = False
    var_65 = module_0.Definitions()
    var_66 = module_1.from_json_schema_type(var_63, var_62, var_64, var_65)



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
    var_18 = 5
    var_19 = 10
    var_20 = {var_0: var_17, var_15: var_18, var_16: var_19}
    var_21 = False
    var_22 = module_0.Definitions()
    var_23 = module_1.from_json_schema_type(var_20, var_17, var_21, var_22)
    var_24 = 'boolean'
    var_25 = {var_0: var_24}
    var_26 = False
    var_27 = module_0.Definitions()
    var_28 = module_1.from_json_schema_type(var_25, var_24, var_26, var_27)
    var_29 = 'items'
    var_30 = 'minItems'
    var_31 = 'maxItems'
    var_32 = 'array'
    var_33 = {var_0: var_17}
    var_34 = 1
    var_35 = {var_0: var_32, var_29: var_33, var_30: var_34, var_31: var_18}
    var_36 = False
    var_37 = module_0.Definitions()
    var_38 = module_1.from_json_schema_type(var_35, var_32, var_36, var_37)
    var_39 = var_38.items
    var_40 = 'properties'
    var_41 = 'required'
    var_42 = 'object'
    var_43 = 'name'
    var_44 = {var_0: var_17}
    var_45 = {var_43: var_44}
    var_46 = [var_43]
    var_47 = {var_0: var_42, var_40: var_45, var_41: var_46}
    var_48 = False
    var_49 = module_0.Definitions()
    var_50 = module_1.from_json_schema_type(var_47, var_42, var_48, var_49)
    var_51 = var_50.properties[var_43]
    var_52 = {var_0: var_17}
    var_53 = True
    var_54 = module_0.Definitions()
    var_55 = module_1.from_json_schema_type(var_52, var_17, var_53, var_54)



# Parsed testcases at query #36
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
    var_54 = 'minProperties'
    var_55 = 'maxProperties'
    var_56 = 'required'
    var_57 = 'object'
    var_58 = 'name'
    var_59 = {var_0: var_25}
    var_60 = {var_58: var_59}
    var_61 = [var_58]
    var_62 = {var_58: var_29}
    var_63 = {var_0: var_57, var_53: var_60, var_54: var_35, var_55: var_46, var_56: var_61, var_6: var_62}
    var_64 = False
    var_65 = module_0.Definitions()
    var_66 = module_1.from_json_schema_type(var_63, var_57, var_64, var_65)
    var_67 = var_66.properties[var_58]
    var_68 = {var_0: var_25}
    var_69 = module_0.Definitions()
    var_70 = module_1.from_json_schema_type(var_68, var_25, var_35, var_69)



# Parsed testcases at query #37
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
    var_129 = 'Test'
    var_130 = module_0.String()
    var_131 = {var_129: var_130}
    var_132 = '$ref'
    var_133 = 'components'
    var_134 = '#/components/schemas/Test'
    var_135 = 'schemas'
    var_136 = {var_9: var_14}
    var_137 = {var_129: var_136}
    var_138 = {var_135: var_137}
    var_139 = {var_132: var_134, var_133: var_138}
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



# Parsed testcases at query #38
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
    var_76 = 'fixed'
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
    var_91 = module_0.String(min_length=var_48)
    var_92 = module_0.String(max_length=var_5)
    var_93 = [var_91, var_92]
    var_94 = module_2.AllOf(var_93)
    var_95 = 'allOf'
    var_96 = {var_9: var_14, var_10: var_48}
    var_97 = {var_9: var_14, var_11: var_5}
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
    var_129 = 'Test'
    var_130 = module_0.String()
    var_131 = {var_129: var_130}
    var_132 = '$ref'
    var_133 = 'components'
    var_134 = '#/components/schemas/Test'
    var_135 = 'schemas'
    var_136 = {var_9: var_14}
    var_137 = {var_129: var_136}
    var_138 = {var_135: var_137}
    var_139 = {var_132: var_134, var_133: var_138}
    var_140 = module_1.to_json_schema(var_120)
    var_141 = module_0.String()
    var_142 = {var_51: var_141}
    var_143 = [var_51]
    var_144 = module_3.Schema(var_142)
    var_145 = {var_9: var_14}
    var_146 = {var_51: var_145}
    var_147 = [var_51]
    var_148 = {var_9: var_60, var_56: var_146, var_57: var_147}
    var_149 = module_1.to_json_schema(var_144)
    var_150 = True
    var_151 = module_0.String()
    var_152 = 'null'
    var_153 = [var_14, var_152]
    var_154 = {var_9: var_153}
    var_155 = module_1.to_json_schema(var_151)



# Parsed testcases at query #39
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
    var_43 = 'additionalItems'
    var_44 = 'uniqueItems'
    var_45 = 'array'
    var_46 = {var_0: var_26}
    var_47 = 10
    var_48 = False
    var_49 = [var_30]
    var_50 = {var_0: var_45, var_40: var_46, var_41: var_10, var_42: var_47, var_43: var_48, var_44: var_10, var_6: var_49}
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
    var_65 = {var_0: var_26}
    var_66 = {var_0: var_17}
    var_67 = {var_63: var_65, var_64: var_66}
    var_68 = '^S_'
    var_69 = '^I_'
    var_70 = {var_0: var_26}
    var_71 = {var_0: var_17}
    var_72 = {var_68: var_70, var_69: var_71}
    var_73 = False
    var_74 = {var_0: var_26}
    var_75 = [var_63]
    var_76 = 25
    var_77 = {var_63: var_30, var_64: var_76}
    var_78 = {var_0: var_62, var_55: var_67, var_56: var_72, var_57: var_73, var_58: var_74, var_59: var_10, var_60: var_47, var_61: var_75, var_6: var_77}
    var_79 = False
    var_80 = module_0.Definitions()
    var_81 = module_1.from_json_schema_type(var_78, var_62, var_79, var_80)
    var_82 = var_81.properties[var_63]
    var_83 = var_81.properties[var_64]
    var_84 = var_81.pattern_properties[var_68]
    var_85 = var_81.pattern_properties[var_69]
    var_86 = var_81.property_names
    var_87 = {var_0: var_26}
    var_88 = module_0.Definitions()
    var_89 = module_1.from_json_schema_type(var_87, var_26, var_10, var_88)



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = 'number'
    var_4 = [var_1, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'null'
    var_7 = [var_1, var_6]
    var_8 = {var_0: var_7}
    var_9 = {}
    var_10 = 'integer'
    var_11 = [var_10, var_3]
    var_12 = {var_0: var_11}



# Parsed testcases at query #2
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
    var_16 = True
    var_17 = False
    var_18 = None
    var_19 = [var_16, var_17, var_18]
    var_20 = {var_0: var_19}
    var_21 = module_0.Definitions()
    var_22 = module_1.enum_from_json_schema(var_20, var_21)



# Parsed testcases at query #3
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
    var_16 = True
    var_17 = 'text'
    var_18 = 123
    var_19 = [var_16, var_17, var_18]
    var_20 = {var_0: var_19}
    var_21 = module_0.Definitions()
    var_22 = module_1.enum_from_json_schema(var_20, var_21)



# Parsed testcases at query #4
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
    var_16 = 'default'
    var_17 = 'minLength'
    var_18 = {var_1: var_2, var_17: var_14}
    var_19 = 'minimum'
    var_20 = {var_1: var_4, var_19: var_12}
    var_21 = [var_18, var_20]
    var_22 = 'test'
    var_23 = {var_0: var_21, var_16: var_22}
    var_24 = module_0.Definitions()
    var_25 = module_1.any_of_from_json_schema(var_23, var_24)
    var_26 = var_25.any_of
    var_27 = len(var_26)
    assert var_27 == 2
    var_28 = module_0.Definitions()
    var_29 = '$ref'
    var_30 = '#/components/schemas/Test'
    var_31 = {var_29: var_30}
    var_32 = {var_1: var_4}
    var_33 = [var_31, var_32]
    var_34 = {var_0: var_33}
    var_35 = module_1.any_of_from_json_schema(var_34, var_28)
    var_36 = var_35.any_of
    var_37 = len(var_36)
    assert var_37 == 2
    var_38 = var_35.any_of[var_12]
    var_39 = var_35.any_of[var_14]
    var_40 = []
    var_41 = {var_0: var_40}
    var_42 = module_0.Definitions()
    var_43 = module_1.any_of_from_json_schema(var_41, var_42)
    var_44 = var_43.any_of
    var_45 = len(var_44)
    assert var_45 == 0



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
    var_19 = 'properties'
    var_20 = 'object'
    var_21 = 'name'
    var_22 = {var_2: var_4}
    var_23 = {var_21: var_22}
    var_24 = {var_2: var_20, var_19: var_23}
    var_25 = 'age'
    var_26 = 'integer'
    var_27 = {var_2: var_26}
    var_28 = {var_25: var_27}
    var_29 = {var_2: var_20, var_19: var_28}
    var_30 = [var_24, var_29]
    var_31 = {var_1: var_30}
    var_32 = module_1.all_of_from_json_schema(var_31, var_0)
    var_33 = var_32.all_of
    var_34 = len(var_33)
    assert var_34 == 2
    var_35 = var_32.all_of[var_15]
    var_36 = var_32.all_of[var_17]
    var_37 = 'default'
    var_38 = {var_2: var_4}
    var_39 = [var_38]
    var_40 = 'test'
    var_41 = {var_1: var_39, var_37: var_40}
    var_42 = module_1.all_of_from_json_schema(var_41, var_0)
    var_43 = '$ref'
    var_44 = '#/components/schemas/Test'
    var_45 = {var_43: var_44}
    var_46 = {var_2: var_4, var_3: var_17}
    var_47 = [var_45, var_46]
    var_48 = {var_1: var_47}
    var_49 = module_1.all_of_from_json_schema(var_48, var_0)
    var_50 = var_49.all_of[var_15]
    var_51 = var_49.all_of[var_17]



# Parsed testcases at query #6
#--------------------------


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
    var_11 = 42
    var_12 = {var_0: var_6, var_1: var_8, var_2: var_10, var_3: var_11}
    var_13 = module_0.Definitions()
    var_14 = module_1.if_then_else_from_json_schema(var_12, var_13)
    var_15 = var_14.if_clause
    var_16 = var_14.then_clause
    var_17 = var_14.else_clause
    var_18 = {var_4: var_5}
    var_19 = {var_4: var_7}
    var_20 = 3.14
    var_21 = {var_0: var_18, var_1: var_19, var_3: var_20}
    var_22 = module_1.if_then_else_from_json_schema(var_21, var_13)
    var_23 = var_22.if_clause
    var_24 = var_22.then_clause
    var_25 = {var_4: var_5}
    var_26 = {var_4: var_9}
    var_27 = True
    var_28 = {var_0: var_25, var_2: var_26, var_3: var_27}
    var_29 = module_1.if_then_else_from_json_schema(var_28, var_13)
    var_30 = var_29.if_clause
    var_31 = var_29.else_clause
    var_32 = {var_4: var_5}
    var_33 = 'test'
    var_34 = {var_0: var_32, var_3: var_33}
    var_35 = module_1.if_then_else_from_json_schema(var_34, var_13)
    var_36 = var_35.if_clause



# Parsed testcases at query #7
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'oneOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'number'
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
    var_20 = 'default_value'
    var_21 = {var_0: var_19, var_16: var_20}
    var_22 = module_0.Definitions()
    var_23 = module_1.one_of_from_json_schema(var_21, var_22)
    var_24 = 'properties'
    var_25 = 'object'
    var_26 = 'name'
    var_27 = {var_1: var_2}
    var_28 = {var_26: var_27}
    var_29 = {var_1: var_25, var_24: var_28}
    var_30 = 'items'
    var_31 = 'array'
    var_32 = 'integer'
    var_33 = {var_1: var_32}
    var_34 = {var_1: var_31, var_30: var_33}
    var_35 = [var_29, var_34]
    var_36 = {var_0: var_35}
    var_37 = module_0.Definitions()
    var_38 = module_1.one_of_from_json_schema(var_36, var_37)
    var_39 = var_38.one_of
    var_40 = len(var_39)
    assert var_40 == 2
    var_41 = var_38.one_of[var_12]
    var_42 = var_38.one_of[var_14]
    var_43 = module_0.Definitions()
    var_44 = '$ref'
    var_45 = '#/components/schemas/StringSchema'
    var_46 = {var_44: var_45}
    var_47 = {var_1: var_4}
    var_48 = [var_46, var_47]
    var_49 = {var_0: var_48}
    var_50 = module_1.one_of_from_json_schema(var_49, var_43)
    var_51 = var_50.one_of
    var_52 = len(var_51)
    assert var_52 == 2
    var_53 = var_50.one_of[var_12]
    var_54 = var_50.one_of[var_14]



# Parsed testcases at query #8
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = '#/components/schemas/Test'
    var_3 = {var_1: var_2}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)



# Parsed testcases at query #9
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
    var_19 = module_1.if_then_else_from_json_schema(var_18, var_11)
    var_20 = var_19.if_clause
    var_21 = var_19.then_clause
    var_22 = {var_3: var_4}
    var_23 = {var_3: var_8}
    var_24 = {var_0: var_22, var_2: var_23}
    var_25 = module_1.if_then_else_from_json_schema(var_24, var_11)
    var_26 = var_25.if_clause
    var_27 = var_25.else_clause
    var_28 = {var_3: var_4}
    var_29 = {var_0: var_28}
    var_30 = module_1.if_then_else_from_json_schema(var_29, var_11)
    var_31 = var_30.if_clause
    var_32 = 'default'
    var_33 = {var_3: var_4}
    var_34 = {var_3: var_6}
    var_35 = {var_3: var_8}
    var_36 = 42
    var_37 = {var_0: var_33, var_1: var_34, var_2: var_35, var_32: var_36}
    var_38 = module_1.if_then_else_from_json_schema(var_37, var_11)



# Parsed testcases at query #10
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
    var_16 = 'default'
    var_17 = 'minLength'
    var_18 = {var_2: var_3, var_17: var_14}
    var_19 = 'minimum'
    var_20 = {var_2: var_5, var_19: var_12}
    var_21 = 'boolean'
    var_22 = {var_2: var_21}
    var_23 = [var_18, var_20, var_22]
    var_24 = 'default_value'
    var_25 = {var_1: var_23, var_16: var_24}
    var_26 = module_1.one_of_from_json_schema(var_25, var_0)
    var_27 = var_26.one_of
    var_28 = len(var_27)
    assert var_28 == 3
    var_29 = var_26.one_of[var_12]
    var_30 = var_26.one_of[var_14]
    var_31 = 2
    var_32 = var_26.one_of[var_31]
    var_33 = '$ref'
    var_34 = '#/components/schemas/TestRef'
    var_35 = {var_33: var_34}
    var_36 = {var_2: var_5}
    var_37 = [var_35, var_36]
    var_38 = {var_1: var_37}
    var_39 = module_1.one_of_from_json_schema(var_38, var_0)
    var_40 = var_39.one_of
    var_41 = len(var_40)
    assert var_41 == 2
    var_42 = var_39.one_of[var_12]
    var_43 = var_39.one_of[var_14]
    var_44 = []
    var_45 = {var_1: var_44}
    var_46 = module_1.one_of_from_json_schema(var_45, var_0)
    var_47 = var_46.one_of
    var_48 = len(var_47)
    assert var_48 == 0



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
    var_42 = 'a'
    var_43 = 'b'
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



# Parsed testcases at query #12
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
    var_8 = 50.5
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = False
    var_11 = module_1.from_json_schema_type(var_9, var_5, var_10, var_0)
    var_12 = 'integer'
    var_13 = 50
    var_14 = {var_1: var_12, var_2: var_10, var_3: var_7, var_4: var_13}
    var_15 = False
    var_16 = module_1.from_json_schema_type(var_14, var_12, var_15, var_0)
    var_17 = 'minLength'
    var_18 = 'maxLength'
    var_19 = 'pattern'
    var_20 = 'string'
    var_21 = 5
    var_22 = 10
    var_23 = '^[A-Za-z]+$'
    var_24 = 'hello'
    var_25 = {var_1: var_20, var_17: var_21, var_18: var_22, var_19: var_23, var_4: var_24}
    var_26 = False
    var_27 = module_1.from_json_schema_type(var_25, var_20, var_26, var_0)
    var_28 = 'boolean'
    var_29 = True
    var_30 = {var_1: var_28, var_4: var_29}
    var_31 = False
    var_32 = module_1.from_json_schema_type(var_30, var_28, var_31, var_0)
    var_33 = 'items'
    var_34 = 'minItems'
    var_35 = 'maxItems'
    var_36 = 'uniqueItems'
    var_37 = 'array'
    var_38 = {var_1: var_20}
    var_39 = 'item1'
    var_40 = [var_39]
    var_41 = {var_1: var_37, var_33: var_38, var_34: var_29, var_35: var_21, var_36: var_29, var_4: var_40}
    var_42 = False
    var_43 = module_1.from_json_schema_type(var_41, var_37, var_42, var_0)
    var_44 = var_43.items
    var_45 = 'properties'
    var_46 = 'required'
    var_47 = 'object'
    var_48 = 'name'
    var_49 = 'age'
    var_50 = {var_1: var_20}
    var_51 = {var_1: var_12}
    var_52 = {var_48: var_50, var_49: var_51}
    var_53 = [var_48]
    var_54 = 'John'
    var_55 = 30
    var_56 = {var_48: var_54, var_49: var_55}
    var_57 = {var_1: var_47, var_45: var_52, var_46: var_53, var_4: var_56}
    var_58 = False
    var_59 = module_1.from_json_schema_type(var_57, var_47, var_58, var_0)
    var_60 = var_59.properties[var_48]
    var_61 = var_59.properties[var_49]
    var_62 = {var_1: var_20}
    var_63 = module_1.from_json_schema_type(var_62, var_20, var_29, var_0)



# Parsed testcases at query #13
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
    var_12 = 'nullable'
    var_13 = True
    var_14 = {var_0: var_1, var_12: var_13}
    var_15 = module_0.Definitions()
    var_16 = module_1.type_from_json_schema(var_14, var_15)
    var_17 = var_16.any_of
    var_18 = var_16.any_of
    var_19 = None
    var_20 = {var_12: var_13}
    var_21 = module_0.Definitions()
    var_22 = module_1.type_from_json_schema(var_20, var_21)
    var_23 = {}
    var_24 = module_0.Definitions()
    var_25 = module_1.type_from_json_schema(var_23, var_24)
    var_26 = 'properties'
    var_27 = 'object'
    var_28 = 'name'
    var_29 = 'age'
    var_30 = {var_0: var_1}
    var_31 = {var_0: var_5}
    var_32 = {var_28: var_30, var_29: var_31}
    var_33 = {var_0: var_27, var_26: var_32}
    var_34 = module_0.Definitions()
    var_35 = module_1.type_from_json_schema(var_33, var_34)
    var_36 = 'items'
    var_37 = 'array'
    var_38 = {var_0: var_1}
    var_39 = {var_0: var_37, var_36: var_38}
    var_40 = module_0.Definitions()
    var_41 = module_1.type_from_json_schema(var_39, var_40)
    var_42 = var_41.items



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
    var_77 = {var_4: var_5}
    var_78 = '$ref'
    var_79 = '#/components/schemas/Test'
    var_80 = {var_78: var_79}
    var_81 = module_0.from_json_schema(var_80, var_76)
    var_82 = 'maxLength'
    var_83 = 10
    var_84 = {var_4: var_5, var_36: var_37, var_82: var_83}
    var_85 = module_0.from_json_schema(var_84)
    var_86 = [var_24, var_25]
    var_87 = {var_4: var_5, var_23: var_86, var_30: var_24}
    var_88 = module_0.from_json_schema(var_87)
    var_89 = var_88.schemas
    var_90 = len(var_89)
    assert var_90 == 3



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
    var_76 = 25
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
    var_31 = 'value'
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
    var_44 = var_43.constraints
    var_45 = len(var_44)
    assert var_45 == 2
    var_46 = 'anyOf'
    var_47 = {var_4: var_5}
    var_48 = {var_4: var_8}
    var_49 = [var_47, var_48]
    var_50 = {var_46: var_49}
    var_51 = module_0.from_json_schema(var_50)
    var_52 = var_51.options
    var_53 = len(var_52)
    assert var_53 == 2
    var_54 = 'oneOf'
    var_55 = {var_4: var_5}
    var_56 = {var_4: var_8}
    var_57 = [var_55, var_56]
    var_58 = {var_54: var_57}
    var_59 = module_0.from_json_schema(var_58)
    var_60 = var_59.options
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
    var_71 = {var_4: var_5, var_35: var_36}
    var_72 = {var_4: var_8}
    var_73 = {var_67: var_70, var_68: var_71, var_69: var_72}
    var_74 = module_0.from_json_schema(var_73)
    var_75 = var_74.if_schema
    var_76 = var_74.then_schema
    var_77 = var_74.else_schema
    var_78 = module_1.Definitions()
    var_79 = {var_4: var_5}
    var_80 = '$ref'
    var_81 = '#/components/schemas/Test'
    var_82 = {var_80: var_81}
    var_83 = module_0.from_json_schema(var_82, var_78)
    var_84 = 'pattern'
    var_85 = '^[a-z]+$'
    var_86 = {var_4: var_5, var_35: var_36, var_38: var_39, var_84: var_85}
    var_87 = module_0.from_json_schema(var_86)
    var_88 = var_87.constraints
    var_89 = len(var_88)
    assert var_89 == 4
    var_90 = {}
    var_91 = module_0.from_json_schema(var_90)



# Parsed testcases at query #17
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
    var_17 = 'format'
    var_18 = 'string'
    var_19 = 5
    var_20 = 10
    var_21 = 'email'
    var_22 = {var_0: var_18, var_15: var_19, var_16: var_20, var_17: var_21}
    var_23 = False
    var_24 = module_0.Definitions()
    var_25 = module_1.from_json_schema_type(var_22, var_18, var_23, var_24)
    var_26 = 'boolean'
    var_27 = {var_0: var_26}
    var_28 = False
    var_29 = module_0.Definitions()
    var_30 = module_1.from_json_schema_type(var_27, var_26, var_28, var_29)
    var_31 = 'items'
    var_32 = 'minItems'
    var_33 = 'maxItems'
    var_34 = 'array'
    var_35 = {var_0: var_18}
    var_36 = 1
    var_37 = {var_0: var_34, var_31: var_35, var_32: var_36, var_33: var_19}
    var_38 = False
    var_39 = module_0.Definitions()
    var_40 = module_1.from_json_schema_type(var_37, var_34, var_38, var_39)
    var_41 = var_40.items
    var_42 = 'properties'
    var_43 = 'required'
    var_44 = 'minProperties'
    var_45 = 'maxProperties'
    var_46 = 'object'
    var_47 = 'name'
    var_48 = {var_0: var_18}
    var_49 = {var_47: var_48}
    var_50 = [var_47]
    var_51 = {var_0: var_46, var_42: var_49, var_43: var_50, var_44: var_36, var_45: var_19}
    var_52 = False
    var_53 = module_0.Definitions()
    var_54 = module_1.from_json_schema_type(var_51, var_46, var_52, var_53)
    var_55 = var_54.properties[var_47]
    var_56 = {var_0: var_18}
    var_57 = True
    var_58 = module_0.Definitions()
    var_59 = module_1.from_json_schema_type(var_56, var_18, var_57, var_58)



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
    var_68 = 'Option A'
    var_69 = (var_67, var_68)
    var_70 = 'b'
    var_71 = 'Option B'
    var_72 = (var_70, var_71)
    var_73 = [var_69, var_72]
    var_74 = module_0.Choice(choices=var_73)
    var_75 = 'enum'
    var_76 = [var_67, var_70]
    var_77 = {var_75: var_76}
    var_78 = module_1.to_json_schema(var_74)
    var_79 = 'fixed_value'
    var_80 = module_0.Const(var_79)
    var_81 = 'const'
    var_82 = {var_81: var_79}
    var_83 = module_1.to_json_schema(var_80)
    var_84 = module_0.String()
    var_85 = module_0.Integer()
    var_86 = [var_84, var_85]
    var_87 = module_0.Union(var_86)
    var_88 = 'anyOf'
    var_89 = {var_9: var_14}
    var_90 = {var_9: var_24}
    var_91 = [var_89, var_90]
    var_92 = {var_88: var_91}
    var_93 = module_1.to_json_schema(var_87)
    var_94 = module_0.String(min_length=var_48)
    var_95 = module_0.String(max_length=var_5)
    var_96 = [var_94, var_95]
    var_97 = module_2.AllOf(var_96)
    var_98 = 'allOf'
    var_99 = {var_9: var_14, var_10: var_48}
    var_100 = {var_9: var_14, var_11: var_5}
    var_101 = [var_99, var_100]
    var_102 = {var_98: var_101}
    var_103 = module_1.to_json_schema(var_97)
    var_104 = module_0.String()
    var_105 = module_0.Integer()
    var_106 = [var_104, var_105]
    var_107 = module_2.OneOf(var_106)
    var_108 = 'oneOf'
    var_109 = {var_9: var_14}
    var_110 = {var_9: var_24}
    var_111 = [var_109, var_110]
    var_112 = {var_108: var_111}
    var_113 = module_1.to_json_schema(var_107)
    var_114 = module_0.String()
    var_115 = module_2.Not(var_114)
    var_116 = 'not'
    var_117 = {var_9: var_14}
    var_118 = {var_116: var_117}
    var_119 = module_1.to_json_schema(var_115)
    var_120 = module_0.String()
    var_121 = module_0.Integer()
    var_122 = module_0.Boolean()
    var_123 = module_2.IfThenElse(var_120, var_121, var_122)
    var_124 = 'if'
    var_125 = 'then'
    var_126 = 'else'
    var_127 = {var_9: var_14}
    var_128 = {var_9: var_24}
    var_129 = {var_9: var_35}
    var_130 = {var_124: var_127, var_125: var_128, var_126: var_129}
    var_131 = module_1.to_json_schema(var_123)
    var_132 = 'Person'
    var_133 = module_0.String()
    var_134 = {var_51: var_133}
    var_135 = module_0.Object(properties=var_134)
    var_136 = {var_132: var_135}
    var_137 = '$ref'
    var_138 = 'components'
    var_139 = '#/components/schemas/Person'
    var_140 = 'schemas'
    var_141 = {var_9: var_14}
    var_142 = {var_51: var_141}
    var_143 = {var_9: var_60, var_57: var_142}
    var_144 = {var_132: var_143}
    var_145 = {var_140: var_144}
    var_146 = {var_137: var_139, var_138: var_145}
    var_147 = module_0.String()
    var_148 = {var_51: var_147}
    var_149 = [var_51]
    var_150 = module_3.Schema(var_148)
    var_151 = {var_9: var_14}
    var_152 = {var_51: var_151}
    var_153 = [var_51]
    var_154 = {var_9: var_60, var_57: var_152, var_58: var_153}
    var_155 = module_1.to_json_schema(var_150)
    var_156 = 'Address'
    var_157 = module_0.String()
    var_158 = {var_51: var_157}
    var_159 = module_0.Object(properties=var_158)
    var_160 = 'street'
    var_161 = module_0.String()
    var_162 = {var_160: var_161}
    var_163 = module_0.Object(properties=var_162)
    var_164 = {var_132: var_159, var_156: var_163}
    var_165 = {var_9: var_14}
    var_166 = {var_51: var_165}
    var_167 = {var_9: var_60, var_57: var_166}
    var_168 = {var_9: var_14}
    var_169 = {var_160: var_168}
    var_170 = {var_9: var_60, var_57: var_169}
    var_171 = {var_132: var_167, var_156: var_170}
    var_172 = {var_140: var_171}
    var_173 = {var_138: var_172}



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
    var_19 = False
    var_20 = 100
    var_21 = module_0.Integer(minimum=var_19, maximum=var_20)
    var_22 = 'minimum'
    var_23 = 'maximum'
    var_24 = 'integer'
    var_25 = {var_9: var_24, var_22: var_19, var_23: var_20}
    var_26 = module_1.to_json_schema(var_21)
    var_27 = 0.5
    var_28 = module_0.Float(multiple_of=var_27)
    var_29 = 'multipleOf'
    var_30 = 'number'
    var_31 = [var_30, var_15]
    var_32 = {var_9: var_31, var_29: var_27}
    var_33 = module_1.to_json_schema(var_28)
    var_34 = module_0.Boolean()
    var_35 = 'boolean'
    var_36 = {var_9: var_35}
    var_37 = module_1.to_json_schema(var_34)
    var_38 = module_0.String()
    var_39 = 5
    var_40 = module_0.Array(var_38, min_items=var_4, max_items=var_39)
    var_41 = 'minItems'
    var_42 = 'maxItems'
    var_43 = 'items'
    var_44 = 'array'
    var_45 = [var_44, var_15]
    var_46 = {var_9: var_14}
    var_47 = {var_9: var_45, var_41: var_4, var_42: var_39, var_43: var_46}
    var_48 = module_1.to_json_schema(var_40)
    var_49 = 'name'
    var_50 = module_0.String()
    var_51 = {var_49: var_50}
    var_52 = [var_49]
    var_53 = module_0.Object(properties=var_51, required=var_52)
    var_54 = 'properties'
    var_55 = 'required'
    var_56 = 'object'
    var_57 = {var_9: var_14}
    var_58 = {var_49: var_57}
    var_59 = [var_49]
    var_60 = {var_9: var_56, var_54: var_58, var_55: var_59}
    var_61 = module_1.to_json_schema(var_53)
    var_62 = 'a'
    var_63 = (var_62, var_62)
    var_64 = 'b'
    var_65 = (var_64, var_64)
    var_66 = [var_63, var_65]
    var_67 = module_0.Choice(choices=var_66)
    var_68 = 'enum'
    var_69 = [var_62, var_64]
    var_70 = {var_68: var_69}
    var_71 = module_1.to_json_schema(var_67)
    var_72 = 'fixed_value'
    var_73 = module_0.Const(var_72)
    var_74 = 'const'
    var_75 = {var_74: var_72}
    var_76 = module_1.to_json_schema(var_73)
    var_77 = module_0.String()
    var_78 = module_0.Integer()
    var_79 = [var_77, var_78]
    var_80 = module_0.Union(var_79)
    var_81 = 'anyOf'
    var_82 = {var_9: var_14}
    var_83 = {var_9: var_24}
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
    var_97 = module_3.Definitions()
    var_98 = 'TestSchema'
    var_99 = module_3.Reference(var_98, var_97)
    var_100 = '$ref'
    var_101 = '#/components/schemas/TestSchema'
    var_102 = {var_100: var_101}
    var_103 = module_1.to_json_schema(var_99)
    var_104 = module_0.String()
    var_105 = {var_49: var_104}
    var_106 = module_3.Schema(var_105)
    var_107 = {var_98: var_106}
    var_108 = 'components'
    var_109 = [var_56, var_15]
    var_110 = {var_9: var_14}
    var_111 = {var_49: var_110}
    var_112 = 'schemas'
    var_113 = [var_56, var_15]
    var_114 = {var_9: var_14}
    var_115 = {var_49: var_114}
    var_116 = {var_9: var_113, var_54: var_115}
    var_117 = {var_98: var_116}
    var_118 = {var_112: var_117}
    var_119 = {var_9: var_109, var_54: var_111, var_108: var_118}
    var_120 = module_1.to_json_schema(var_107)



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



# Parsed testcases at query #21
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
    var_17 = [var_1, var_16]
    var_18 = {var_0: var_17}
    var_19 = module_0.Definitions()
    var_20 = module_1.type_from_json_schema(var_18, var_19)
    var_21 = {}
    var_22 = module_0.Definitions()
    var_23 = module_1.type_from_json_schema(var_21, var_22)
    var_24 = []
    var_25 = {var_0: var_24}
    var_26 = module_0.Definitions()
    var_27 = module_1.type_from_json_schema(var_25, var_26)
    var_28 = 'properties'
    var_29 = 'object'
    var_30 = 'name'
    var_31 = 'age'
    var_32 = {var_0: var_1}
    var_33 = {var_0: var_5}
    var_34 = {var_30: var_32, var_31: var_33}
    var_35 = {var_0: var_29, var_28: var_34}
    var_36 = module_0.Definitions()
    var_37 = module_1.type_from_json_schema(var_35, var_36)
    var_38 = var_37.properties[var_30]
    var_39 = var_37.properties[var_31]
    var_40 = 'items'
    var_41 = 'array'
    var_42 = {var_0: var_1}
    var_43 = {var_0: var_41, var_40: var_42}
    var_44 = module_0.Definitions()
    var_45 = module_1.type_from_json_schema(var_43, var_44)
    var_46 = var_45.items
    var_47 = 'minLength'
    var_48 = 'maxLength'
    var_49 = 5
    var_50 = 10
    var_51 = {var_0: var_1, var_47: var_49, var_48: var_50}
    var_52 = module_0.Definitions()
    var_53 = module_1.type_from_json_schema(var_51, var_52)



# Parsed testcases at query #22
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
    var_25 = module_0.Definitions()
    var_26 = 'integer'
    var_27 = '$ref'
    var_28 = '#/components/schemas/Test'
    var_29 = {var_27: var_28}
    var_30 = module_1.type_from_json_schema(var_29, var_25)



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
    var_22 = '^[A-Za-z]+$'
    var_23 = 'hello'
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
    var_41 = 'item1'
    var_42 = [var_41]
    var_43 = {var_0: var_38, var_34: var_39, var_35: var_29, var_36: var_40, var_37: var_29, var_3: var_42}
    var_44 = False
    var_45 = module_0.Definitions()
    var_46 = module_1.from_json_schema_type(var_43, var_38, var_44, var_45)
    var_47 = var_46.items
    var_48 = 'properties'
    var_49 = 'required'
    var_50 = 'object'
    var_51 = 'name'
    var_52 = 'age'
    var_53 = {var_0: var_20}
    var_54 = {var_0: var_12}
    var_55 = {var_51: var_53, var_52: var_54}
    var_56 = [var_51]
    var_57 = 'John'
    var_58 = {var_51: var_57}
    var_59 = {var_0: var_50, var_48: var_55, var_49: var_56, var_3: var_58}
    var_60 = False
    var_61 = module_0.Definitions()
    var_62 = module_1.from_json_schema_type(var_59, var_50, var_60, var_61)
    var_63 = var_62.properties[var_51]
    var_64 = var_62.properties[var_52]
    var_65 = {var_0: var_20}
    var_66 = module_0.Definitions()
    var_67 = module_1.from_json_schema_type(var_65, var_20, var_29, var_66)



# Parsed testcases at query #24
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
    var_12 = 'null'
    var_13 = [var_1, var_12]
    var_14 = {var_0: var_13}
    var_15 = module_0.Definitions()
    var_16 = module_1.type_from_json_schema(var_14, var_15)
    var_17 = {}
    var_18 = module_0.Definitions()
    var_19 = module_1.type_from_json_schema(var_17, var_18)
    var_20 = {var_0: var_12}
    var_21 = module_0.Definitions()
    var_22 = module_1.type_from_json_schema(var_20, var_21)
    var_23 = 'minLength'
    var_24 = 5
    var_25 = {var_0: var_1, var_23: var_24}
    var_26 = module_0.Definitions()
    var_27 = module_1.type_from_json_schema(var_25, var_26)
    var_28 = 'minimum'
    var_29 = 'maximum'
    var_30 = 0
    var_31 = 100
    var_32 = {var_0: var_5, var_28: var_30, var_29: var_31}
    var_33 = module_0.Definitions()
    var_34 = module_1.type_from_json_schema(var_32, var_33)



# Parsed testcases at query #25
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
    var_42 = 'minItems'
    var_43 = 'maxItems'
    var_44 = 'uniqueItems'
    var_45 = 'array'
    var_46 = {var_0: var_25}
    var_47 = [var_30]
    var_48 = {var_0: var_45, var_41: var_46, var_42: var_36, var_43: var_27, var_44: var_36, var_6: var_47}
    var_49 = False
    var_50 = module_0.Definitions()
    var_51 = module_1.from_json_schema_type(var_48, var_45, var_49, var_50)
    var_52 = var_51.items
    var_53 = 'properties'
    var_54 = 'minProperties'
    var_55 = 'maxProperties'
    var_56 = 'required'
    var_57 = 'object'
    var_58 = 'name'
    var_59 = 'age'
    var_60 = {var_0: var_25}
    var_61 = {var_0: var_16}
    var_62 = {var_58: var_60, var_59: var_61}
    var_63 = [var_58]
    var_64 = 30
    var_65 = {var_58: var_30, var_59: var_64}
    var_66 = {var_0: var_57, var_53: var_62, var_54: var_36, var_55: var_10, var_56: var_63, var_6: var_65}
    var_67 = False
    var_68 = module_0.Definitions()
    var_69 = module_1.from_json_schema_type(var_66, var_57, var_67, var_68)
    var_70 = var_69.properties[var_58]
    var_71 = var_69.properties[var_59]



# Parsed testcases at query #26
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
    var_54 = False
    var_55 = [var_51]
    var_56 = module_0.Object(properties=var_53, additional_properties=var_54, required=var_55)
    var_57 = 'properties'
    var_58 = 'additionalProperties'
    var_59 = 'required'
    var_60 = 'object'
    var_61 = {var_9: var_14}
    var_62 = {var_51: var_61}
    var_63 = False
    var_64 = [var_51]
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
    var_92 = module_0.String(min_length=var_48)
    var_93 = module_0.String(max_length=var_5)
    var_94 = [var_92, var_93]
    var_95 = module_2.AllOf(var_94)
    var_96 = 'allOf'
    var_97 = {var_9: var_14, var_10: var_48}
    var_98 = {var_9: var_14, var_11: var_5}
    var_99 = [var_97, var_98]
    var_100 = {var_96: var_99}
    var_101 = module_1.to_json_schema(var_95)
    var_102 = 'Person'
    var_103 = module_0.String()
    var_104 = {var_51: var_103}
    var_105 = module_0.Object(properties=var_104)
    var_106 = {var_102: var_105}
    var_107 = '$ref'
    var_108 = 'components'
    var_109 = '#/components/schemas/Person'
    var_110 = 'schemas'
    var_111 = {var_9: var_14}
    var_112 = {var_51: var_111}
    var_113 = {var_9: var_60, var_57: var_112}
    var_114 = {var_102: var_113}
    var_115 = {var_110: var_114}
    var_116 = {var_107: var_109, var_108: var_115}
    var_117 = module_0.String(min_length=var_48)
    var_118 = module_0.Integer()
    var_119 = module_0.Boolean()
    var_120 = module_2.IfThenElse(var_117, var_118, var_119)
    var_121 = 'if'
    var_122 = 'then'
    var_123 = 'else'
    var_124 = {var_9: var_14, var_10: var_48}
    var_125 = {var_9: var_24}
    var_126 = {var_9: var_35}
    var_127 = {var_121: var_124, var_122: var_125, var_123: var_126}
    var_128 = module_1.to_json_schema(var_120)
    var_129 = module_0.String()
    var_130 = module_2.Not(var_129)
    var_131 = 'not'
    var_132 = {var_9: var_14}
    var_133 = {var_131: var_132}
    var_134 = module_1.to_json_schema(var_130)



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
    var_22 = 'exclusiveMinimum'
    var_23 = 'maximum'
    var_24 = 'integer'
    var_25 = True
    var_26 = {var_9: var_24, var_21: var_17, var_22: var_25, var_23: var_18}
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
    var_149 = module_3.Definitions()
    var_150 = module_1.to_json_schema(var_149)
    var_151 = 'test'
    var_152 = {var_9: var_14}
    var_153 = {var_151: var_152}
    var_154 = {var_135: var_153}
    var_155 = {var_133: var_154}



# Parsed testcases at query #28
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
    var_64 = var_63.constraint
    var_65 = 'if'
    var_66 = 'then'
    var_67 = 'else'
    var_68 = {var_4: var_5}
    var_69 = {var_36: var_37}
    var_70 = {var_4: var_8}
    var_71 = {var_65: var_68, var_66: var_69, var_67: var_70}
    var_72 = module_0.from_json_schema(var_71)
    var_73 = var_72.if_constraint
    var_74 = var_72.then_constraint
    var_75 = var_72.else_constraint
    var_76 = module_1.Definitions()
    var_77 = '$ref'
    var_78 = '#/components/schemas/Test'
    var_79 = {var_77: var_78}
    var_80 = module_0.from_json_schema(var_79, var_76)
    var_81 = 'maxLength'
    var_82 = 10
    var_83 = {var_4: var_5, var_36: var_37, var_81: var_82}
    var_84 = module_0.from_json_schema(var_83)
    var_85 = [var_24, var_25]
    var_86 = {var_4: var_5, var_23: var_85, var_36: var_0}
    var_87 = module_0.from_json_schema(var_86)
    var_88 = var_87.constraints
    var_89 = len(var_88)
    assert var_89 == 2



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
    var_41 = 'additionalItems'
    var_42 = 'minItems'
    var_43 = 'maxItems'
    var_44 = 'uniqueItems'
    var_45 = 'array'
    var_46 = {var_0: var_26}
    var_47 = False
    var_48 = 10
    var_49 = [var_30]
    var_50 = {var_0: var_45, var_40: var_46, var_41: var_47, var_42: var_10, var_43: var_48, var_44: var_10, var_6: var_49}
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
    var_65 = {var_0: var_26}
    var_66 = {var_0: var_17}
    var_67 = {var_63: var_65, var_64: var_66}
    var_68 = '^S_'
    var_69 = '^I_'
    var_70 = {var_0: var_26}
    var_71 = {var_0: var_17}
    var_72 = {var_68: var_70, var_69: var_71}
    var_73 = False
    var_74 = {var_0: var_26}
    var_75 = [var_63]
    var_76 = 30
    var_77 = {var_63: var_30, var_64: var_76}
    var_78 = {var_0: var_62, var_55: var_67, var_56: var_72, var_57: var_73, var_58: var_74, var_59: var_10, var_60: var_48, var_61: var_75, var_6: var_77}
    var_79 = False
    var_80 = module_0.Definitions()
    var_81 = module_1.from_json_schema_type(var_78, var_62, var_79, var_80)
    var_82 = var_81.properties[var_63]
    var_83 = var_81.properties[var_64]
    var_84 = var_81.pattern_properties[var_68]
    var_85 = var_81.pattern_properties[var_69]
    var_86 = var_81.property_names
    var_87 = {var_0: var_26}
    var_88 = module_0.Definitions()
    var_89 = module_1.from_json_schema_type(var_87, var_26, var_10, var_88)



# Parsed testcases at query #30
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
    var_29 = '^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$'
    var_30 = 'test@example.com'
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
    var_47 = 'item1'
    var_48 = 'item2'
    var_49 = [var_47, var_48]
    var_50 = {var_0: var_44, var_40: var_45, var_41: var_10, var_42: var_46, var_43: var_10, var_6: var_49}
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
    var_62 = {var_0: var_26}
    var_63 = {var_0: var_17}
    var_64 = {var_60: var_62, var_61: var_63}
    var_65 = [var_60]
    var_66 = 'John'
    var_67 = 30
    var_68 = {var_60: var_66, var_61: var_67}
    var_69 = {var_0: var_59, var_55: var_64, var_56: var_65, var_57: var_10, var_58: var_27, var_6: var_68}
    var_70 = False
    var_71 = module_0.Definitions()
    var_72 = module_1.from_json_schema_type(var_69, var_59, var_70, var_71)
    var_73 = var_72.properties[var_60]
    var_74 = var_72.properties[var_61]



# Parsed testcases at query #31
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
    var_47 = {var_0: var_7}
    var_48 = 10
    var_49 = [var_29]
    var_50 = {var_0: var_45, var_40: var_46, var_41: var_47, var_42: var_35, var_43: var_48, var_44: var_35, var_6: var_49}
    var_51 = False
    var_52 = module_0.Definitions()
    var_53 = module_1.from_json_schema_type(var_50, var_45, var_51, var_52)
    var_54 = var_53.items
    var_55 = var_53.additional_items
    var_56 = 'properties'
    var_57 = 'patternProperties'
    var_58 = 'additionalProperties'
    var_59 = 'propertyNames'
    var_60 = 'minProperties'
    var_61 = 'maxProperties'
    var_62 = 'required'
    var_63 = 'object'
    var_64 = 'name'
    var_65 = 'age'
    var_66 = {var_0: var_25}
    var_67 = {var_0: var_16}
    var_68 = {var_64: var_66, var_65: var_67}
    var_69 = '^S_'
    var_70 = '^I_'
    var_71 = {var_0: var_25}
    var_72 = {var_0: var_16}
    var_73 = {var_69: var_71, var_70: var_72}
    var_74 = {var_0: var_34}
    var_75 = {var_0: var_25}
    var_76 = [var_64]
    var_77 = 30
    var_78 = {var_64: var_29, var_65: var_77}
    var_79 = {var_0: var_63, var_56: var_68, var_57: var_73, var_58: var_74, var_59: var_75, var_60: var_35, var_61: var_48, var_62: var_76, var_6: var_78}
    var_80 = False
    var_81 = module_0.Definitions()
    var_82 = module_1.from_json_schema_type(var_79, var_63, var_80, var_81)
    var_83 = var_82.properties[var_64]
    var_84 = var_82.properties[var_65]
    var_85 = var_82.pattern_properties[var_69]
    var_86 = var_82.pattern_properties[var_70]
    var_87 = var_82.additional_properties
    var_88 = var_82.property_names
    var_89 = {}
    var_90 = 'invalid'
    var_91 = False
    var_92 = module_0.Definitions()
    var_93 = module_1.from_json_schema_type(var_89, var_90, var_91, var_92)



# Parsed testcases at query #32
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
    var_39 = 'items'
    var_40 = 'minItems'
    var_41 = 'maxItems'
    var_42 = 'uniqueItems'
    var_43 = 'array'
    var_44 = {var_9: var_14}
    var_45 = True
    var_46 = {var_9: var_43, var_39: var_44, var_40: var_37, var_41: var_36, var_42: var_45}
    var_47 = module_1.to_json_schema(var_38)
    var_48 = 'name'
    var_49 = module_0.String()
    var_50 = {var_48: var_49}
    var_51 = [var_48]
    var_52 = module_0.Object(properties=var_50, min_properties=var_45, max_properties=var_36, required=var_51)
    var_53 = 'properties'
    var_54 = 'required'
    var_55 = 'minProperties'
    var_56 = 'maxProperties'
    var_57 = 'object'
    var_58 = {var_9: var_14}
    var_59 = {var_48: var_58}
    var_60 = [var_48]
    var_61 = {var_9: var_57, var_53: var_59, var_54: var_60, var_55: var_45, var_56: var_36}
    var_62 = module_1.to_json_schema(var_52)
    var_63 = 'a'
    var_64 = (var_63, var_63)
    var_65 = 'b'
    var_66 = (var_65, var_65)
    var_67 = [var_64, var_66]
    var_68 = module_0.Choice(choices=var_67)
    var_69 = 'enum'
    var_70 = [var_63, var_65]
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
    var_89 = 'test'
    var_90 = module_0.Const(var_89)
    var_91 = [var_88, var_90]
    var_92 = module_2.AllOf(var_91)
    var_93 = 'allOf'
    var_94 = {var_9: var_14}
    var_95 = {var_75: var_89}
    var_96 = [var_94, var_95]
    var_97 = {var_93: var_96}
    var_98 = module_1.to_json_schema(var_92)
    var_99 = 'Test'
    var_100 = module_0.String()
    var_101 = {var_99: var_100}
    var_102 = '$ref'
    var_103 = 'components'
    var_104 = '#/components/schemas/Test'
    var_105 = 'schemas'
    var_106 = {var_9: var_14}
    var_107 = {var_99: var_106}
    var_108 = {var_105: var_107}
    var_109 = {var_102: var_104, var_103: var_108}
    var_110 = module_0.String()
    var_111 = module_0.Integer()
    var_112 = module_0.Boolean()
    var_113 = module_2.IfThenElse(var_110, var_111, var_112)
    var_114 = 'if'
    var_115 = 'then'
    var_116 = 'else'
    var_117 = {var_9: var_14}
    var_118 = {var_9: var_22}
    var_119 = {var_9: var_32}
    var_120 = {var_114: var_117, var_115: var_118, var_116: var_119}
    var_121 = module_1.to_json_schema(var_113)
    var_122 = module_0.String()
    var_123 = module_2.Not(var_122)
    var_124 = 'not'
    var_125 = {var_9: var_14}
    var_126 = {var_124: var_125}
    var_127 = module_1.to_json_schema(var_123)



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
    var_87 = {}
    var_88 = 'invalid'
    var_89 = False
    var_90 = module_0.Definitions()
    var_91 = module_1.from_json_schema_type(var_87, var_88, var_89, var_90)



# Parsed testcases at query #34
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
    var_64 = {var_0: var_25}
    var_65 = {var_63: var_64}
    var_66 = '^S_'
    var_67 = {var_0: var_25}
    var_68 = {var_66: var_67}
    var_69 = False
    var_70 = {var_0: var_25}
    var_71 = [var_63]
    var_72 = {var_63: var_29}
    var_73 = {var_0: var_62, var_55: var_65, var_56: var_68, var_57: var_69, var_58: var_70, var_59: var_35, var_60: var_48, var_61: var_71, var_6: var_72}
    var_74 = False
    var_75 = module_0.Definitions()
    var_76 = module_1.from_json_schema_type(var_73, var_62, var_74, var_75)
    var_77 = var_76.properties[var_63]
    var_78 = var_76.pattern_properties[var_66]
    var_79 = var_76.property_names
    var_80 = {}
    var_81 = 'invalid'
    var_82 = False
    var_83 = module_0.Definitions()
    var_84 = module_1.from_json_schema_type(var_80, var_81, var_82, var_83)



# Parsed testcases at query #35
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
    var_57 = False
    var_58 = module_0.Object(properties=var_55, additional_properties=var_57, required=var_56)
    var_59 = 'properties'
    var_60 = 'required'
    var_61 = 'additionalProperties'
    var_62 = 'object'
    var_63 = {var_9: var_14}
    var_64 = {var_53: var_63}
    var_65 = [var_53]
    var_66 = False
    var_67 = {var_9: var_62, var_59: var_64, var_60: var_65, var_61: var_66}
    var_68 = module_1.to_json_schema(var_58)
    var_69 = 'a'
    var_70 = (var_69, var_69)
    var_71 = 'b'
    var_72 = (var_71, var_71)
    var_73 = [var_70, var_72]
    var_74 = module_0.Choice(choices=var_73)
    var_75 = 'enum'
    var_76 = [var_69, var_71]
    var_77 = {var_75: var_76}
    var_78 = module_1.to_json_schema(var_74)
    var_79 = 'fixed_value'
    var_80 = module_0.Const(var_79)
    var_81 = 'const'
    var_82 = {var_81: var_79}
    var_83 = module_1.to_json_schema(var_80)
    var_84 = module_0.String()
    var_85 = module_0.Integer()
    var_86 = [var_84, var_85]
    var_87 = module_0.Union(var_86)
    var_88 = 'anyOf'
    var_89 = {var_9: var_14}
    var_90 = {var_9: var_26}
    var_91 = [var_89, var_90]
    var_92 = {var_88: var_91}
    var_93 = module_1.to_json_schema(var_87)
    var_94 = module_0.String()
    var_95 = module_0.Integer()
    var_96 = [var_94, var_95]
    var_97 = module_2.AllOf(var_96)
    var_98 = 'allOf'
    var_99 = {var_9: var_14}
    var_100 = {var_9: var_26}
    var_101 = [var_99, var_100]
    var_102 = {var_98: var_101}
    var_103 = module_1.to_json_schema(var_97)
    var_104 = module_0.String()
    var_105 = module_0.Integer()
    var_106 = [var_104, var_105]
    var_107 = module_2.OneOf(var_106)
    var_108 = 'oneOf'
    var_109 = {var_9: var_14}
    var_110 = {var_9: var_26}
    var_111 = [var_109, var_110]
    var_112 = {var_108: var_111}
    var_113 = module_1.to_json_schema(var_107)
    var_114 = module_0.String()
    var_115 = module_2.Not(var_114)
    var_116 = 'not'
    var_117 = {var_9: var_14}
    var_118 = {var_116: var_117}
    var_119 = module_1.to_json_schema(var_115)
    var_120 = module_0.String()
    var_121 = module_0.Integer()
    var_122 = module_0.Boolean()
    var_123 = module_2.IfThenElse(var_120, var_121, var_122)
    var_124 = 'if'
    var_125 = 'then'
    var_126 = 'else'
    var_127 = {var_9: var_14}
    var_128 = {var_9: var_26}
    var_129 = {var_9: var_38}
    var_130 = {var_124: var_127, var_125: var_128, var_126: var_129}
    var_131 = module_1.to_json_schema(var_123)
    var_132 = 'Test'
    var_133 = module_0.String()
    var_134 = {var_132: var_133}
    var_135 = '$ref'
    var_136 = '#/components/schemas/Test'
    var_137 = {var_135: var_136}
    var_138 = module_0.String()
    var_139 = {var_53: var_138}
    var_140 = [var_53]
    var_141 = module_3.Schema(var_139)
    var_142 = {var_9: var_14}
    var_143 = {var_53: var_142}
    var_144 = [var_53]
    var_145 = {var_9: var_62, var_59: var_143, var_60: var_144}
    var_146 = module_1.to_json_schema(var_141)
    var_147 = module_0.String()
    var_148 = {var_132: var_147}
    var_149 = 'components'
    var_150 = 'schemas'
    var_151 = {var_9: var_14}
    var_152 = {var_132: var_151}
    var_153 = {var_150: var_152}
    var_154 = {var_149: var_153}



# Parsed testcases at query #36
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
    var_42 = module_0.Array(var_40, var_19, var_4, var_41)
    var_43 = 'items'
    var_44 = 'additionalItems'
    var_45 = 'minItems'
    var_46 = 'maxItems'
    var_47 = 'array'
    var_48 = [var_47, var_15]
    var_49 = {var_9: var_14}
    var_50 = {var_9: var_48, var_43: var_49, var_44: var_19, var_45: var_4, var_46: var_41}
    var_51 = module_1.to_json_schema(var_42)
    var_52 = 'name'
    var_53 = module_0.String()
    var_54 = {var_52: var_53}
    var_55 = [var_52]
    var_56 = module_0.Object(properties=var_54, additional_properties=var_19, required=var_55)
    var_57 = 'properties'
    var_58 = 'additionalProperties'
    var_59 = 'required'
    var_60 = 'object'
    var_61 = {var_9: var_14}
    var_62 = {var_52: var_61}
    var_63 = [var_52]
    var_64 = {var_9: var_60, var_57: var_62, var_58: var_19, var_59: var_63}
    var_65 = module_1.to_json_schema(var_56)
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
    var_102 = 'Test'
    var_103 = module_0.String()
    var_104 = {var_102: var_103}
    var_105 = '$ref'
    var_106 = '#/components/schemas/Test'
    var_107 = {var_105: var_106}
    var_108 = module_0.String()
    var_109 = {var_52: var_108}
    var_110 = [var_52]
    var_111 = module_3.Schema(var_109)
    var_112 = {var_9: var_14}
    var_113 = {var_52: var_112}
    var_114 = [var_52]
    var_115 = {var_9: var_60, var_57: var_113, var_59: var_114}
    var_116 = module_1.to_json_schema(var_111)
    var_117 = 'StringField'
    var_118 = 'IntField'
    var_119 = module_0.String()
    var_120 = module_0.Integer()
    var_121 = {var_117: var_119, var_118: var_120}
    var_122 = 'components'
    var_123 = 'schemas'
    var_124 = {var_9: var_14}
    var_125 = {var_9: var_26}
    var_126 = {var_117: var_124, var_118: var_125}
    var_127 = {var_123: var_126}
    var_128 = {var_122: var_127}



# Parsed testcases at query #37
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
    var_12 = 50.5
    var_13 = {var_0: var_7, var_1: var_8, var_2: var_9, var_3: var_10, var_4: var_10, var_5: var_11, var_6: var_12}
    var_14 = False
    var_15 = module_0.Definitions()
    var_16 = module_1.from_json_schema_type(var_13, var_7, var_14, var_15)
    var_17 = 'integer'
    var_18 = 50
    var_19 = {var_0: var_17, var_1: var_14, var_2: var_9, var_3: var_10, var_4: var_10, var_5: var_11, var_6: var_18}
    var_20 = False
    var_21 = module_0.Definitions()
    var_22 = module_1.from_json_schema_type(var_19, var_17, var_20, var_21)
    var_23 = 'minLength'
    var_24 = 'maxLength'
    var_25 = 'format'
    var_26 = 'pattern'
    var_27 = 'string'
    var_28 = 5
    var_29 = 'email'
    var_30 = '^[a-zA-Z0-9]+$'
    var_31 = 'test@example.com'
    var_32 = {var_0: var_27, var_23: var_28, var_24: var_9, var_25: var_29, var_26: var_30, var_6: var_31}
    var_33 = False
    var_34 = module_0.Definitions()
    var_35 = module_1.from_json_schema_type(var_32, var_27, var_33, var_34)
    var_36 = 'boolean'
    var_37 = {var_0: var_36, var_6: var_10}
    var_38 = False
    var_39 = module_0.Definitions()
    var_40 = module_1.from_json_schema_type(var_37, var_36, var_38, var_39)
    var_41 = 'items'
    var_42 = 'additionalItems'
    var_43 = 'minItems'
    var_44 = 'maxItems'
    var_45 = 'uniqueItems'
    var_46 = 'array'
    var_47 = {var_0: var_27}
    var_48 = False
    var_49 = 10
    var_50 = 'item1'
    var_51 = 'item2'
    var_52 = [var_50, var_51]
    var_53 = {var_0: var_46, var_41: var_47, var_42: var_48, var_43: var_10, var_44: var_49, var_45: var_10, var_6: var_52}
    var_54 = False
    var_55 = module_0.Definitions()
    var_56 = module_1.from_json_schema_type(var_53, var_46, var_54, var_55)
    var_57 = var_56.items
    var_58 = 'properties'
    var_59 = 'patternProperties'
    var_60 = 'additionalProperties'
    var_61 = 'propertyNames'
    var_62 = 'minProperties'
    var_63 = 'maxProperties'
    var_64 = 'required'
    var_65 = 'object'
    var_66 = 'name'
    var_67 = 'age'
    var_68 = {var_0: var_27}
    var_69 = {var_0: var_17}
    var_70 = {var_66: var_68, var_67: var_69}
    var_71 = '^S_'
    var_72 = '^I_'
    var_73 = {var_0: var_27}
    var_74 = {var_0: var_17}
    var_75 = {var_71: var_73, var_72: var_74}
    var_76 = False
    var_77 = {var_0: var_27}
    var_78 = [var_66]
    var_79 = 'John'
    var_80 = 30
    var_81 = {var_66: var_79, var_67: var_80}
    var_82 = {var_0: var_65, var_58: var_70, var_59: var_75, var_60: var_76, var_61: var_77, var_62: var_10, var_63: var_49, var_64: var_78, var_6: var_81}
    var_83 = False
    var_84 = module_0.Definitions()
    var_85 = module_1.from_json_schema_type(var_82, var_65, var_83, var_84)
    var_86 = var_85.properties[var_66]
    var_87 = var_85.properties[var_67]
    var_88 = var_85.pattern_properties[var_71]
    var_89 = var_85.pattern_properties[var_72]
    var_90 = var_85.property_names
    var_91 = {var_0: var_27}
    var_92 = module_0.Definitions()
    var_93 = module_1.from_json_schema_type(var_91, var_27, var_10, var_92)



# Parsed testcases at query #38
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
    var_37 = 'minItems'
    var_38 = 'maxItems'
    var_39 = 'items'
    var_40 = 'array'
    var_41 = {var_9: var_14}
    var_42 = {var_9: var_40, var_37: var_4, var_38: var_5, var_39: var_41}
    var_43 = module_1.to_json_schema(var_36)
    var_44 = 'name'
    var_45 = module_0.String()
    var_46 = {var_44: var_45}
    var_47 = [var_44]
    var_48 = False
    var_49 = module_0.Object(properties=var_46, additional_properties=var_48, required=var_47)
    var_50 = 'properties'
    var_51 = 'required'
    var_52 = 'additionalProperties'
    var_53 = 'object'
    var_54 = {var_9: var_14}
    var_55 = {var_44: var_54}
    var_56 = [var_44]
    var_57 = False
    var_58 = {var_9: var_53, var_50: var_55, var_51: var_56, var_52: var_57}
    var_59 = module_1.to_json_schema(var_49)
    var_60 = 'a'
    var_61 = (var_60, var_60)
    var_62 = 'b'
    var_63 = (var_62, var_62)
    var_64 = [var_61, var_63]
    var_65 = module_0.Choice(choices=var_64)
    var_66 = 'enum'
    var_67 = [var_60, var_62]
    var_68 = {var_66: var_67}
    var_69 = module_1.to_json_schema(var_65)
    var_70 = 'fixed_value'
    var_71 = module_0.Const(var_70)
    var_72 = 'const'
    var_73 = {var_72: var_70}
    var_74 = module_1.to_json_schema(var_71)
    var_75 = module_0.String()
    var_76 = module_0.Integer()
    var_77 = [var_75, var_76]
    var_78 = module_0.Union(var_77)
    var_79 = 'anyOf'
    var_80 = {var_9: var_14}
    var_81 = {var_9: var_22}
    var_82 = [var_80, var_81]
    var_83 = {var_79: var_82}
    var_84 = module_1.to_json_schema(var_78)
    var_85 = module_0.String()
    var_86 = 'test'
    var_87 = module_0.Const(var_86)
    var_88 = [var_85, var_87]
    var_89 = module_2.AllOf(var_88)
    var_90 = 'allOf'
    var_91 = {var_9: var_14}
    var_92 = {var_72: var_86}
    var_93 = [var_91, var_92]
    var_94 = {var_90: var_93}
    var_95 = module_1.to_json_schema(var_89)
    var_96 = module_3.Definitions()
    var_97 = 'Test'
    var_98 = module_3.Reference(var_97, var_96)
    var_99 = '$ref'
    var_100 = 'components'
    var_101 = '#/components/schemas/Test'
    var_102 = 'schemas'
    var_103 = {var_9: var_14}
    var_104 = {var_97: var_103}
    var_105 = {var_102: var_104}
    var_106 = {var_99: var_101, var_100: var_105}
    var_107 = module_1.to_json_schema(var_98)
    var_108 = module_0.String()
    var_109 = {var_44: var_108}
    var_110 = [var_44]
    var_111 = module_3.Schema(var_109)
    var_112 = {var_9: var_14}
    var_113 = {var_44: var_112}
    var_114 = [var_44]
    var_115 = {var_9: var_53, var_50: var_113, var_51: var_114}
    var_116 = module_1.to_json_schema(var_111)
    var_117 = True
    var_118 = module_0.Const(var_117)
    var_119 = module_0.String()
    var_120 = module_0.Integer()
    var_121 = module_2.IfThenElse(var_118, var_119, var_120)
    var_122 = 'if'
    var_123 = 'then'
    var_124 = 'else'
    var_125 = True
    var_126 = {var_72: var_125}
    var_127 = {var_9: var_14}
    var_128 = {var_9: var_22}
    var_129 = {var_122: var_126, var_123: var_127, var_124: var_128}
    var_130 = module_1.to_json_schema(var_121)
    var_131 = module_0.String()
    var_132 = module_2.Not(var_131)
    var_133 = 'not'
    var_134 = {var_9: var_14}
    var_135 = {var_133: var_134}
    var_136 = module_1.to_json_schema(var_132)



# Parsed testcases at query #39
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
    var_83 = {var_4: var_20, var_79: var_82}
    var_84 = '$ref'
    var_85 = '#/components/schemas/Person'
    var_86 = {var_84: var_85}
    var_87 = module_0.from_json_schema(var_86, var_78)
    var_88 = 'pattern'
    var_89 = '^[a-zA-Z]+$'
    var_90 = {var_4: var_5, var_35: var_36, var_38: var_39, var_88: var_89}
    var_91 = module_0.from_json_schema(var_90)
    var_92 = var_91.schemas
    var_93 = len(var_92)
    assert var_93 == 2
    var_94 = {}
    var_95 = module_0.from_json_schema(var_94)



# Parsed testcases at query #40
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
    var_102 = 'Test'
    var_103 = module_0.String()
    var_104 = {var_102: var_103}
    var_105 = '$ref'
    var_106 = 'components'
    var_107 = '#/components/schemas/Test'
    var_108 = 'schemas'
    var_109 = {var_9: var_14}
    var_110 = {var_102: var_109}
    var_111 = {var_108: var_110}
    var_112 = {var_105: var_107, var_106: var_111}
    var_113 = True
    var_114 = module_0.Const(var_113)
    var_115 = module_0.String()
    var_116 = module_0.Integer()
    var_117 = module_2.IfThenElse(var_114, var_115, var_116)
    var_118 = 'if'
    var_119 = 'then'
    var_120 = 'else'
    var_121 = True
    var_122 = {var_78: var_121}
    var_123 = {var_9: var_14}
    var_124 = {var_9: var_24}
    var_125 = {var_118: var_122, var_119: var_123, var_120: var_124}
    var_126 = module_1.to_json_schema(var_117)
    var_127 = module_0.String()
    var_128 = module_2.Not(var_127)
    var_129 = 'not'
    var_130 = {var_9: var_14}
    var_131 = {var_129: var_130}
    var_132 = module_1.to_json_schema(var_128)



# Parsed testcases at query #41
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
    var_22 = 'exclusiveMinimum'
    var_23 = 'maximum'
    var_24 = 'integer'
    var_25 = True
    var_26 = {var_9: var_24, var_21: var_17, var_22: var_25, var_23: var_18}
    var_27 = module_1.to_json_schema(var_20)
    var_28 = 0.5
    var_29 = 1.5
    var_30 = module_0.Float(multiple_of=var_28)
    var_31 = 'multipleOf'
    var_32 = 'default'
    var_33 = 'number'
    var_34 = {var_9: var_33, var_31: var_28, var_32: var_29}
    var_35 = module_1.to_json_schema(var_30)
    var_36 = False
    var_37 = module_0.Boolean()
    var_38 = 'boolean'
    var_39 = False
    var_40 = {var_9: var_38, var_32: var_39}
    var_41 = module_1.to_json_schema(var_37)
    var_42 = module_0.String()
    var_43 = 5
    var_44 = True
    var_45 = False
    var_46 = module_0.Array(var_42, var_45, var_25, var_43, unique_items=var_44)
    var_47 = 'items'
    var_48 = 'minItems'
    var_49 = 'maxItems'
    var_50 = 'uniqueItems'
    var_51 = 'additionalItems'
    var_52 = 'array'
    var_53 = {var_9: var_14}
    var_54 = True
    var_55 = False
    var_56 = {var_9: var_52, var_47: var_53, var_48: var_44, var_49: var_43, var_50: var_54, var_51: var_55}
    var_57 = module_1.to_json_schema(var_46)
    var_58 = 'name'
    var_59 = 'age'
    var_60 = module_0.String()
    var_61 = module_0.Integer()
    var_62 = {var_58: var_60, var_59: var_61}
    var_63 = [var_58]
    var_64 = 3
    var_65 = module_0.Object(properties=var_62, min_properties=var_54, max_properties=var_64, required=var_63)
    var_66 = 'properties'
    var_67 = 'required'
    var_68 = 'minProperties'
    var_69 = 'maxProperties'
    var_70 = 'object'
    var_71 = {var_9: var_14}
    var_72 = {var_9: var_24}
    var_73 = {var_58: var_71, var_59: var_72}
    var_74 = [var_58]
    var_75 = {var_9: var_70, var_66: var_73, var_67: var_74, var_68: var_54, var_69: var_64}
    var_76 = module_1.to_json_schema(var_65)
    var_77 = 'a'
    var_78 = (var_77, var_77)
    var_79 = 'b'
    var_80 = (var_79, var_79)
    var_81 = [var_78, var_80]
    var_82 = module_0.Choice(choices=var_81)
    var_83 = 'enum'
    var_84 = [var_77, var_79]
    var_85 = {var_83: var_84, var_32: var_77}
    var_86 = module_1.to_json_schema(var_82)
    var_87 = 'fixed'
    var_88 = module_0.Const(var_87)
    var_89 = 'const'
    var_90 = {var_89: var_87, var_32: var_87}
    var_91 = module_1.to_json_schema(var_88)
    var_92 = module_0.String()
    var_93 = module_0.Integer()
    var_94 = [var_92, var_93]
    var_95 = module_0.Union(var_94)
    var_96 = 'anyOf'
    var_97 = {var_9: var_14}
    var_98 = {var_9: var_24}
    var_99 = [var_97, var_98]
    var_100 = {var_96: var_99}
    var_101 = module_1.to_json_schema(var_95)
    var_102 = module_0.String(min_length=var_54)
    var_103 = module_0.String(max_length=var_5)
    var_104 = [var_102, var_103]
    var_105 = module_2.AllOf(var_104)
    var_106 = 'allOf'
    var_107 = {var_9: var_14, var_10: var_54}
    var_108 = {var_9: var_14, var_11: var_5}
    var_109 = [var_107, var_108]
    var_110 = {var_106: var_109}
    var_111 = module_1.to_json_schema(var_105)
    var_112 = 'Person'
    var_113 = module_0.String()
    var_114 = {var_58: var_113}
    var_115 = module_0.Object(properties=var_114)
    var_116 = {var_112: var_115}
    var_117 = '$ref'
    var_118 = 'components'
    var_119 = '#/components/schemas/Person'
    var_120 = 'schemas'
    var_121 = {var_9: var_14}
    var_122 = {var_58: var_121}
    var_123 = {var_9: var_70, var_66: var_122}
    var_124 = {var_112: var_123}
    var_125 = {var_120: var_124}
    var_126 = {var_117: var_119, var_118: var_125}
    var_127 = module_0.String()
    var_128 = module_0.Integer()
    var_129 = module_0.Boolean()
    var_130 = module_2.IfThenElse(var_127, var_128, var_129)
    var_131 = 'if'
    var_132 = 'then'
    var_133 = 'else'
    var_134 = {var_9: var_14}
    var_135 = {var_9: var_24}
    var_136 = {var_9: var_38}
    var_137 = {var_131: var_134, var_132: var_135, var_133: var_136}
    var_138 = module_1.to_json_schema(var_130)
    var_139 = module_0.String()
    var_140 = module_2.Not(var_139)
    var_141 = 'not'
    var_142 = {var_9: var_14}
    var_143 = {var_141: var_142}
    var_144 = module_1.to_json_schema(var_140)



# Parsed testcases at query #42
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
    var_12 = True
    var_13 = True
    var_14 = module_0.Integer(minimum=var_10, maximum=var_11, exclusive_minimum=var_12, exclusive_maximum=var_13)
    var_15 = module_1.to_json_schema(var_14)
    var_16 = 0.1
    var_17 = module_0.Float(minimum=var_10, maximum=var_13, multiple_of=var_16)
    var_18 = module_1.to_json_schema(var_17)
    var_19 = module_0.Boolean()
    var_20 = module_1.to_json_schema(var_19)
    var_21 = module_0.String()
    var_22 = True
    var_23 = module_0.Array(var_21, min_items=var_13, max_items=var_5, unique_items=var_22)
    var_24 = module_1.to_json_schema(var_23)
    var_25 = 'name'
    var_26 = module_0.String()
    var_27 = {var_25: var_26}
    var_28 = [var_25]
    var_29 = module_0.Object(properties=var_27, min_properties=var_22, max_properties=var_5, required=var_28)
    var_30 = module_1.to_json_schema(var_29)
    var_31 = 'a'
    var_32 = (var_31, var_31)
    var_33 = 'b'
    var_34 = (var_33, var_33)
    var_35 = [var_32, var_34]
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
    var_46 = module_0.String()
    var_47 = module_0.Integer()
    var_48 = [var_46, var_47]
    var_49 = module_2.OneOf(var_48)
    var_50 = module_1.to_json_schema(var_49)
    var_51 = module_0.String()
    var_52 = 'test'
    var_53 = module_0.Const(var_52)
    var_54 = [var_51, var_53]
    var_55 = module_2.AllOf(var_54)
    var_56 = module_1.to_json_schema(var_55)
    var_57 = module_0.String()
    var_58 = module_2.Not(var_57)
    var_59 = module_1.to_json_schema(var_58)
    var_60 = module_0.String()
    var_61 = module_0.Integer()
    var_62 = module_0.Boolean()
    var_63 = module_2.IfThenElse(var_60, var_61, var_62)
    var_64 = module_1.to_json_schema(var_63)
    var_65 = module_3.Definitions()
    var_66 = 'Test'
    var_67 = module_3.Reference(var_66, var_65)
    var_68 = module_1.to_json_schema(var_67)
    var_69 = module_0.String()
    var_70 = {var_25: var_69}
    var_71 = [var_25]
    var_72 = module_3.Schema(var_70)
    var_73 = module_1.to_json_schema(var_72)
    var_74 = module_3.Definitions()
    var_75 = module_1.to_json_schema(var_74)



# Parsed testcases at query #43
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
    var_40 = 5
    var_41 = module_0.String()
    var_42 = module_0.Array(var_41, var_19, var_4, var_40)
    var_43 = 'minItems'
    var_44 = 'maxItems'
    var_45 = 'items'
    var_46 = 'additionalItems'
    var_47 = 'uniqueItems'
    var_48 = 'array'
    var_49 = [var_48, var_15]
    var_50 = {var_9: var_14}
    var_51 = {var_9: var_49, var_43: var_4, var_44: var_40, var_45: var_50, var_46: var_19, var_47: var_4}
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
    var_88 = {var_9: var_26}
    var_89 = [var_87, var_88]
    var_90 = {var_86: var_89}
    var_91 = module_1.to_json_schema(var_85)
    var_92 = module_0.String(min_length=var_4)
    var_93 = module_0.String(max_length=var_5)
    var_94 = [var_92, var_93]
    var_95 = module_2.AllOf(var_94)
    var_96 = 'allOf'
    var_97 = {var_9: var_14, var_10: var_4}
    var_98 = {var_9: var_14, var_11: var_5}
    var_99 = [var_97, var_98]
    var_100 = {var_96: var_99}
    var_101 = module_1.to_json_schema(var_95)
    var_102 = 'Person'
    var_103 = module_0.String()
    var_104 = {var_53: var_103}
    var_105 = module_0.Object(properties=var_104)
    var_106 = {var_102: var_105}
    var_107 = '$ref'
    var_108 = 'components'
    var_109 = '#/components/schemas/Person'
    var_110 = 'schemas'
    var_111 = {var_9: var_14}
    var_112 = {var_53: var_111}
    var_113 = {var_9: var_61, var_58: var_112}
    var_114 = {var_102: var_113}
    var_115 = {var_110: var_114}
    var_116 = {var_107: var_109, var_108: var_115}
    var_117 = module_0.String()
    var_118 = {var_53: var_117}
    var_119 = [var_53]
    var_120 = module_3.Schema(var_118)
    var_121 = {var_9: var_14}
    var_122 = {var_53: var_121}
    var_123 = [var_53]
    var_124 = {var_9: var_61, var_58: var_122, var_60: var_123}
    var_125 = module_1.to_json_schema(var_120)



# Parsed testcases at query #44
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
    var_94 = module_0.String()
    var_95 = module_0.Integer()
    var_96 = module_2.IfThenElse(var_94, var_95)
    var_97 = 'if'
    var_98 = 'then'
    var_99 = {var_9: var_14}
    var_100 = {var_9: var_22}
    var_101 = {var_97: var_99, var_98: var_100}
    var_102 = module_1.to_json_schema(var_96)
    var_103 = module_0.String()
    var_104 = module_2.Not(var_103)
    var_105 = 'not'
    var_106 = {var_9: var_14}
    var_107 = {var_105: var_106}
    var_108 = module_1.to_json_schema(var_104)
    var_109 = 'Test'
    var_110 = module_0.String()
    var_111 = {var_109: var_110}
    var_112 = '$ref'
    var_113 = 'components'
    var_114 = '#/components/schemas/Test'
    var_115 = 'schemas'
    var_116 = {var_9: var_14}
    var_117 = {var_109: var_116}
    var_118 = {var_115: var_117}
    var_119 = {var_112: var_114, var_113: var_118}
    var_120 = module_0.String()
    var_121 = {var_45: var_120}
    var_122 = [var_45]
    var_123 = module_3.Schema(var_121)
    var_124 = {var_9: var_14}
    var_125 = {var_45: var_124}
    var_126 = [var_45]
    var_127 = {var_9: var_52, var_50: var_125, var_51: var_126}
    var_128 = module_1.to_json_schema(var_123)



# Parsed testcases at query #45
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
    var_46 = 'age'
    var_47 = module_0.String()
    var_48 = module_0.Integer()
    var_49 = {var_45: var_47, var_46: var_48}
    var_50 = module_0.Object(properties=var_49)
    var_51 = 'properties'
    var_52 = 'object'
    var_53 = {var_9: var_14}
    var_54 = {var_9: var_22}
    var_55 = {var_45: var_53, var_46: var_54}
    var_56 = {var_9: var_52, var_51: var_55}
    var_57 = module_1.to_json_schema(var_50)
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
    var_68 = 'fixed_value'
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
    var_94 = module_0.String()
    var_95 = module_0.Integer()
    var_96 = module_2.IfThenElse(var_94, var_95)
    var_97 = 'if'
    var_98 = 'then'
    var_99 = {var_9: var_14}
    var_100 = {var_9: var_22}
    var_101 = {var_97: var_99, var_98: var_100}
    var_102 = module_1.to_json_schema(var_96)
    var_103 = module_0.String()
    var_104 = module_2.Not(var_103)
    var_105 = 'not'
    var_106 = {var_9: var_14}
    var_107 = {var_105: var_106}
    var_108 = module_1.to_json_schema(var_104)
    var_109 = 'Test'
    var_110 = module_0.String()
    var_111 = {var_109: var_110}
    var_112 = '$ref'
    var_113 = 'components'
    var_114 = '#/components/schemas/Test'
    var_115 = 'schemas'
    var_116 = {var_9: var_14}
    var_117 = {var_109: var_116}
    var_118 = {var_115: var_117}
    var_119 = {var_112: var_114, var_113: var_118}
    var_120 = module_0.String()
    var_121 = {var_45: var_120}
    var_122 = module_3.Schema(var_121)
    var_123 = {var_9: var_14}
    var_124 = {var_45: var_123}
    var_125 = {var_9: var_52, var_51: var_124}
    var_126 = module_1.to_json_schema(var_122)



# Parsed testcases at query #46
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
    var_8 = True
    var_9 = module_0.String(max_length=var_5, min_length=var_4, pattern=var_6, format=var_7)
    var_10 = 'type'
    var_11 = 'minLength'
    var_12 = 'maxLength'
    var_13 = 'pattern'
    var_14 = 'format'
    var_15 = 'string'
    var_16 = 'null'
    var_17 = [var_15, var_16]
    var_18 = {var_10: var_17, var_11: var_8, var_12: var_5, var_13: var_6, var_14: var_7}
    var_19 = module_1.to_json_schema(var_9)
    var_20 = 0
    var_21 = 100
    var_22 = True
    var_23 = module_0.Integer(minimum=var_20, maximum=var_21)
    var_24 = 'minimum'
    var_25 = 'maximum'
    var_26 = 'integer'
    var_27 = [var_26, var_16]
    var_28 = {var_10: var_27, var_24: var_20, var_25: var_21}
    var_29 = module_1.to_json_schema(var_23)
    var_30 = 0.5
    var_31 = True
    var_32 = module_0.Float(multiple_of=var_30)
    var_33 = 'multipleOf'
    var_34 = 'number'
    var_35 = [var_34, var_16]
    var_36 = {var_10: var_35, var_33: var_30}
    var_37 = module_1.to_json_schema(var_32)
    var_38 = True
    var_39 = module_0.Boolean()
    var_40 = 'boolean'
    var_41 = [var_40, var_16]
    var_42 = {var_10: var_41}
    var_43 = module_1.to_json_schema(var_39)
    var_44 = module_0.String()
    var_45 = True
    var_46 = module_0.Array(var_44, min_items=var_38, max_items=var_5)
    var_47 = 'minItems'
    var_48 = 'maxItems'
    var_49 = 'items'
    var_50 = 'array'
    var_51 = [var_50, var_16]
    var_52 = {var_10: var_15}
    var_53 = {var_10: var_51, var_47: var_45, var_48: var_5, var_49: var_52}
    var_54 = module_1.to_json_schema(var_46)
    var_55 = 'name'
    var_56 = module_0.String()
    var_57 = {var_55: var_56}
    var_58 = [var_55]
    var_59 = True
    var_60 = module_0.Object(properties=var_57, required=var_58)
    var_61 = 'properties'
    var_62 = 'required'
    var_63 = 'object'
    var_64 = [var_63, var_16]
    var_65 = {var_10: var_15}
    var_66 = {var_55: var_65}
    var_67 = [var_55]
    var_68 = {var_10: var_64, var_61: var_66, var_62: var_67}
    var_69 = module_1.to_json_schema(var_60)
    var_70 = 'a'
    var_71 = (var_70, var_70)
    var_72 = 'b'
    var_73 = (var_72, var_72)
    var_74 = [var_71, var_73]
    var_75 = module_0.Choice(choices=var_74)
    var_76 = 'enum'
    var_77 = [var_70, var_72]
    var_78 = {var_76: var_77}
    var_79 = module_1.to_json_schema(var_75)
    var_80 = 'fixed_value'
    var_81 = module_0.Const(var_80)
    var_82 = 'const'
    var_83 = {var_82: var_80}
    var_84 = module_1.to_json_schema(var_81)
    var_85 = module_0.String()
    var_86 = module_0.Integer()
    var_87 = [var_85, var_86]
    var_88 = module_0.Union(var_87)
    var_89 = 'anyOf'
    var_90 = {var_10: var_15}
    var_91 = {var_10: var_26}
    var_92 = [var_90, var_91]
    var_93 = {var_89: var_92}
    var_94 = module_1.to_json_schema(var_88)
    var_95 = module_0.String()
    var_96 = module_0.Integer()
    var_97 = [var_95, var_96]
    var_98 = module_2.OneOf(var_97)
    var_99 = 'oneOf'
    var_100 = {var_10: var_15}
    var_101 = {var_10: var_26}
    var_102 = [var_100, var_101]
    var_103 = {var_99: var_102}
    var_104 = module_1.to_json_schema(var_98)
    var_105 = module_0.String()
    var_106 = module_0.Integer()
    var_107 = [var_105, var_106]
    var_108 = module_2.AllOf(var_107)
    var_109 = 'allOf'
    var_110 = {var_10: var_15}
    var_111 = {var_10: var_26}
    var_112 = [var_110, var_111]
    var_113 = {var_109: var_112}
    var_114 = module_1.to_json_schema(var_108)
    var_115 = module_0.String()
    var_116 = module_0.Integer()
    var_117 = module_0.Boolean()
    var_118 = module_2.IfThenElse(var_115, var_116, var_117)
    var_119 = 'if'
    var_120 = 'then'
    var_121 = 'else'
    var_122 = {var_10: var_15}
    var_123 = {var_10: var_26}
    var_124 = {var_10: var_40}
    var_125 = {var_119: var_122, var_120: var_123, var_121: var_124}
    var_126 = module_1.to_json_schema(var_118)
    var_127 = module_0.String()
    var_128 = module_2.Not(var_127)
    var_129 = 'not'
    var_130 = {var_10: var_15}
    var_131 = {var_129: var_130}
    var_132 = module_1.to_json_schema(var_128)
    var_133 = module_3.Definitions()
    var_134 = 'test'
    var_135 = module_3.Reference(var_134, var_133)
    var_136 = '$ref'
    var_137 = 'components'
    var_138 = '#/components/schemas/test'
    var_139 = 'schemas'
    var_140 = {}
    var_141 = {var_139: var_140}
    var_142 = {var_136: var_138, var_137: var_141}
    var_143 = module_1.to_json_schema(var_135)
    var_144 = module_0.String()
    var_145 = {var_55: var_144}
    var_146 = [var_55]
    var_147 = module_3.Schema(var_145)
    var_148 = {var_10: var_15}
    var_149 = {var_55: var_148}
    var_150 = [var_55]
    var_151 = {var_10: var_63, var_61: var_149, var_62: var_150}
    var_152 = module_1.to_json_schema(var_147)



# Parsed testcases at query #47
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
    var_40 = 5
    var_41 = module_0.String()
    var_42 = module_0.Array(var_41, var_19, var_4, var_40)
    var_43 = 'minItems'
    var_44 = 'maxItems'
    var_45 = 'items'
    var_46 = 'additionalItems'
    var_47 = 'uniqueItems'
    var_48 = 'array'
    var_49 = [var_48, var_15]
    var_50 = {var_9: var_14}
    var_51 = {var_9: var_49, var_43: var_4, var_44: var_40, var_45: var_50, var_46: var_19, var_47: var_4}
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
    var_88 = {var_9: var_26}
    var_89 = [var_87, var_88]
    var_90 = {var_86: var_89}
    var_91 = module_1.to_json_schema(var_85)
    var_92 = module_0.String(min_length=var_4)
    var_93 = module_0.String(max_length=var_5)
    var_94 = [var_92, var_93]
    var_95 = module_2.AllOf(var_94)
    var_96 = 'allOf'
    var_97 = {var_9: var_14, var_10: var_4}
    var_98 = {var_9: var_14, var_11: var_5}
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
    var_130 = 'Person'
    var_131 = module_0.String()
    var_132 = {var_53: var_131}
    var_133 = module_0.Object(properties=var_132)
    var_134 = {var_130: var_133}
    var_135 = '$ref'
    var_136 = 'components'
    var_137 = '#/components/schemas/Person'
    var_138 = 'schemas'
    var_139 = {var_9: var_14}
    var_140 = {var_53: var_139}
    var_141 = {var_9: var_61, var_58: var_140}
    var_142 = {var_130: var_141}
    var_143 = {var_138: var_142}
    var_144 = {var_135: var_137, var_136: var_143}
    var_145 = module_0.String()
    var_146 = {var_53: var_145}
    var_147 = [var_53]
    var_148 = module_3.Schema(var_146)
    var_149 = {var_9: var_14}
    var_150 = {var_53: var_149}
    var_151 = [var_53]
    var_152 = {var_9: var_61, var_58: var_150, var_60: var_151}
    var_153 = module_1.to_json_schema(var_148)



# Parsed testcases at query #48
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
    var_23 = 10
    var_24 = '^[A-Za-z]+$'
    var_25 = 'hello'
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
    var_39 = 'uniqueItems'
    var_40 = 'array'
    var_41 = {var_0: var_21}
    var_42 = 'a'
    var_43 = 'b'
    var_44 = [var_42, var_43]
    var_45 = {var_0: var_40, var_36: var_41, var_37: var_31, var_38: var_22, var_39: var_31, var_3: var_44}
    var_46 = False
    var_47 = module_0.Definitions()
    var_48 = module_1.from_json_schema_type(var_45, var_40, var_46, var_47)
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
    var_70 = module_1.from_json_schema_type(var_68, var_21, var_31, var_69)



# Parsed testcases at query #49
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
    var_88 = {var_9: var_26}
    var_89 = [var_87, var_88]
    var_90 = {var_86: var_89}
    var_91 = module_1.to_json_schema(var_85)
    var_92 = module_0.String(min_length=var_4)
    var_93 = module_0.String(max_length=var_5)
    var_94 = [var_92, var_93]
    var_95 = module_2.AllOf(var_94)
    var_96 = 'allOf'
    var_97 = {var_9: var_14, var_10: var_4}
    var_98 = {var_9: var_14, var_11: var_5}
    var_99 = [var_97, var_98]
    var_100 = {var_96: var_99}
    var_101 = module_1.to_json_schema(var_95)
    var_102 = 'Person'
    var_103 = module_0.String()
    var_104 = {var_53: var_103}
    var_105 = module_0.Object(properties=var_104)
    var_106 = {var_102: var_105}
    var_107 = '$ref'
    var_108 = 'components'
    var_109 = '#/components/schemas/Person'
    var_110 = 'schemas'
    var_111 = {var_9: var_14}
    var_112 = {var_53: var_111}
    var_113 = {var_9: var_61, var_58: var_112}
    var_114 = {var_102: var_113}
    var_115 = {var_110: var_114}
    var_116 = {var_107: var_109, var_108: var_115}
    var_117 = module_0.String(min_length=var_4)
    var_118 = module_0.Integer()
    var_119 = module_0.Boolean()
    var_120 = module_2.IfThenElse(var_117, var_118, var_119)
    var_121 = 'if'
    var_122 = 'then'
    var_123 = 'else'
    var_124 = {var_9: var_14, var_10: var_4}
    var_125 = {var_9: var_26}
    var_126 = {var_9: var_37}
    var_127 = {var_121: var_124, var_122: var_125, var_123: var_126}
    var_128 = module_1.to_json_schema(var_120)
    var_129 = module_0.String()
    var_130 = module_2.Not(var_129)
    var_131 = 'not'
    var_132 = {var_9: var_14}
    var_133 = {var_131: var_132}
    var_134 = module_1.to_json_schema(var_130)



# Parsed testcases at query #50
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
    var_18 = 5
    var_19 = 10
    var_20 = {var_0: var_17, var_15: var_18, var_16: var_19}
    var_21 = False
    var_22 = module_0.Definitions()
    var_23 = module_1.from_json_schema_type(var_20, var_17, var_21, var_22)
    var_24 = 'boolean'
    var_25 = {var_0: var_24}
    var_26 = False
    var_27 = module_0.Definitions()
    var_28 = module_1.from_json_schema_type(var_25, var_24, var_26, var_27)
    var_29 = 'items'
    var_30 = 'minItems'
    var_31 = 'array'
    var_32 = {var_0: var_17}
    var_33 = 1
    var_34 = {var_0: var_31, var_29: var_32, var_30: var_33}
    var_35 = False
    var_36 = module_0.Definitions()
    var_37 = module_1.from_json_schema_type(var_34, var_31, var_35, var_36)
    var_38 = var_37.items
    var_39 = 'properties'
    var_40 = 'required'
    var_41 = 'object'
    var_42 = 'name'
    var_43 = {var_0: var_17}
    var_44 = {var_42: var_43}
    var_45 = [var_42]
    var_46 = {var_0: var_41, var_39: var_44, var_40: var_45}
    var_47 = False
    var_48 = module_0.Definitions()
    var_49 = module_1.from_json_schema_type(var_46, var_41, var_47, var_48)
    var_50 = var_49.properties[var_42]
    var_51 = {var_0: var_17}
    var_52 = True
    var_53 = module_0.Definitions()
    var_54 = module_1.from_json_schema_type(var_51, var_17, var_52, var_53)



# Parsed testcases at query #51
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
    var_90 = 'invalid'
    var_91 = {var_0: var_90}
    var_92 = 'invalid'
    var_93 = False
    var_94 = module_0.Definitions()
    var_95 = module_1.from_json_schema_type(var_91, var_92, var_93, var_94)



