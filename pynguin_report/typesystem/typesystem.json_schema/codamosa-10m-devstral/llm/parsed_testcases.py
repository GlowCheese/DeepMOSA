####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
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
    var_17 = var_14.all_of
    var_18 = 'properties'
    var_19 = 'object'
    var_20 = 'name'
    var_21 = {var_3: var_5}
    var_22 = {var_20: var_21}
    var_23 = {var_3: var_19, var_18: var_22}
    var_24 = 'age'
    var_25 = 'integer'
    var_26 = {var_3: var_25}
    var_27 = {var_24: var_26}
    var_28 = {var_3: var_19, var_18: var_27}
    var_29 = [var_28]
    var_30 = {var_1: var_29}
    var_31 = [var_23, var_30]
    var_32 = {var_1: var_31}
    var_33 = module_1.all_of_from_json_schema(var_32, var_0)
    var_34 = var_33.all_of
    var_35 = len(var_34)
    assert var_35 == 2
    var_36 = 0
    var_37 = var_33.all_of[var_36]
    var_38 = 1
    var_39 = var_33.all_of[var_38]
    var_40 = []
    var_41 = {var_1: var_40}
    var_42 = module_1.all_of_from_json_schema(var_41, var_0)
    var_43 = var_42.all_of
    var_44 = len(var_43)
    assert var_44 == 0



# Parsed testcases at query #2
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
    var_6 = 'minLength'
    var_7 = 5
    var_8 = {var_3: var_4, var_6: var_7}
    var_9 = 'number'
    var_10 = {var_3: var_9}
    var_11 = {var_0: var_5, var_1: var_8, var_2: var_10}
    var_12 = module_0.Definitions()
    var_13 = module_1.if_then_else_from_json_schema(var_11, var_12)
    var_14 = var_13.if_clause
    var_15 = var_13.then_clause
    var_16 = var_13.else_clause
    var_17 = {var_3: var_4}
    var_18 = {var_3: var_4, var_6: var_7}
    var_19 = {var_0: var_17, var_1: var_18}
    var_20 = module_1.if_then_else_from_json_schema(var_19, var_12)
    var_21 = var_20.if_clause
    var_22 = var_20.then_clause
    var_23 = {var_3: var_4}
    var_24 = {var_3: var_9}
    var_25 = {var_0: var_23, var_2: var_24}
    var_26 = module_1.if_then_else_from_json_schema(var_25, var_12)
    var_27 = var_26.if_clause
    var_28 = var_26.else_clause
    var_29 = {var_3: var_4}
    var_30 = {var_0: var_29}
    var_31 = module_1.if_then_else_from_json_schema(var_30, var_12)
    var_32 = var_31.if_clause
    var_33 = 'default'
    var_34 = {var_3: var_4}
    var_35 = {var_3: var_4, var_6: var_7}
    var_36 = {var_3: var_9}
    var_37 = 'default_value'
    var_38 = {var_0: var_34, var_1: var_35, var_2: var_36, var_33: var_37}
    var_39 = module_1.if_then_else_from_json_schema(var_38, var_12)



# Parsed testcases at query #3
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
    var_113 = 'nested'
    var_114 = 'Nested'
    var_115 = module_0.Integer()
    var_116 = {var_114: var_115}
    var_117 = '#/components/schemas/Nested'
    var_118 = {var_105: var_117}
    var_119 = {var_113: var_118}
    var_120 = {var_9: var_24}
    var_121 = {var_114: var_120}
    var_122 = {var_108: var_121}
    var_123 = {var_9: var_60, var_56: var_119, var_106: var_122}



# Parsed testcases at query #4
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



# Parsed testcases at query #5
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'allOf'
    var_1 = 'type'
    var_2 = 'minLength'
    var_3 = 'string'
    var_4 = 5
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'maxLength'
    var_7 = 10
    var_8 = {var_1: var_3, var_6: var_7}
    var_9 = [var_5, var_8]
    var_10 = {var_0: var_9}
    var_11 = module_0.Definitions()
    var_12 = module_1.all_of_from_json_schema(var_10, var_11)
    var_13 = var_12.all_of
    var_14 = len(var_13)
    assert var_14 == 2
    var_15 = var_12.all_of
    var_16 = 'minimum'
    var_17 = 'integer'
    var_18 = 0
    var_19 = {var_1: var_17, var_16: var_18}
    var_20 = 'maximum'
    var_21 = 100
    var_22 = {var_1: var_17, var_20: var_21}
    var_23 = [var_19, var_22]
    var_24 = {var_0: var_23}
    var_25 = 'multipleOf'
    var_26 = 2
    var_27 = {var_1: var_17, var_25: var_26}
    var_28 = [var_24, var_27]
    var_29 = {var_0: var_28}
    var_30 = module_0.Definitions()
    var_31 = module_1.all_of_from_json_schema(var_29, var_30)
    var_32 = var_31.all_of
    var_33 = len(var_32)
    assert var_33 == 2
    var_34 = var_31.all_of[var_18]
    var_35 = 1
    var_36 = var_31.all_of[var_35]
    var_37 = 'default'
    var_38 = 'boolean'
    var_39 = {var_1: var_38}
    var_40 = [var_39]
    var_41 = True
    var_42 = {var_0: var_40, var_37: var_41}
    var_43 = module_0.Definitions()
    var_44 = module_1.all_of_from_json_schema(var_42, var_43)
    var_45 = module_0.Definitions()
    var_46 = '$ref'
    var_47 = '#/components/schemas/Test'
    var_48 = {var_46: var_47}
    var_49 = {var_1: var_3, var_6: var_7}
    var_50 = [var_48, var_49]
    var_51 = {var_0: var_50}
    var_52 = module_1.all_of_from_json_schema(var_51, var_45)
    var_53 = var_52.all_of
    var_54 = len(var_53)
    assert var_54 == 2
    var_55 = var_52.all_of[var_18]
    var_56 = var_52.all_of[var_41]



# Parsed testcases at query #6
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = '#/components/schemas/Test'
    var_3 = {var_1: var_2}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)
    var_5 = '$ref'
    var_6 = 'unsupported_ref'
    var_7 = {var_5: var_6}
    var_8 = module_1.ref_from_json_schema(var_7, var_0)



# Parsed testcases at query #7
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
    var_17 = 'properties'
    var_18 = 'object'
    var_19 = 'name'
    var_20 = {var_2: var_3}
    var_21 = {var_19: var_20}
    var_22 = {var_2: var_18, var_17: var_21}
    var_23 = 'items'
    var_24 = 'array'
    var_25 = 'integer'
    var_26 = {var_2: var_25}
    var_27 = {var_2: var_24, var_23: var_26}
    var_28 = [var_22, var_27]
    var_29 = {var_19: var_16}
    var_30 = {var_1: var_28, var_16: var_29}
    var_31 = module_1.one_of_from_json_schema(var_30, var_0)
    var_32 = var_31.one_of
    var_33 = len(var_32)
    assert var_33 == 2
    var_34 = var_31.one_of[var_12]
    var_35 = var_31.one_of[var_14]
    var_36 = module_2.String()
    var_37 = {var_19: var_36}
    var_38 = '$ref'
    var_39 = '#/components/schemas/Person'
    var_40 = {var_38: var_39}
    var_41 = {var_2: var_3}
    var_42 = [var_40, var_41]
    var_43 = {var_1: var_42}
    var_44 = module_1.one_of_from_json_schema(var_43, var_0)
    var_45 = var_44.one_of
    var_46 = len(var_45)
    assert var_46 == 2
    var_47 = var_44.one_of[var_12]
    var_48 = var_44.one_of[var_14]



# Parsed testcases at query #8
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
    var_22 = 'age'
    var_23 = {var_2: var_5}
    var_24 = {var_22: var_23}
    var_25 = {var_2: var_17, var_16: var_24}
    var_26 = [var_21, var_25]
    var_27 = {var_1: var_26}
    var_28 = module_1.one_of_from_json_schema(var_27, var_0)
    var_29 = var_28.one_of
    var_30 = len(var_29)
    assert var_30 == 2
    var_31 = var_28.one_of[var_12]
    var_32 = var_28.one_of[var_14]
    var_33 = 'default'
    var_34 = {var_2: var_3}
    var_35 = {var_2: var_5}
    var_36 = [var_34, var_35]
    var_37 = 'default_value'
    var_38 = {var_1: var_36, var_33: var_37}
    var_39 = module_1.one_of_from_json_schema(var_38, var_0)
    var_40 = '$ref'
    var_41 = '#/components/schemas/Test'
    var_42 = {var_40: var_41}
    var_43 = {var_2: var_5}
    var_44 = [var_42, var_43]
    var_45 = {var_1: var_44}
    var_46 = module_1.one_of_from_json_schema(var_45, var_0)
    var_47 = var_46.one_of[var_12]



# Parsed testcases at query #9
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
    var_17 = {var_1: var_2}
    var_18 = '$ref'
    var_19 = '#/components/schemas/Test'
    var_20 = {var_18: var_19}
    var_21 = {var_1: var_4}
    var_22 = [var_20, var_21]
    var_23 = {var_0: var_22}
    var_24 = module_1.one_of_from_json_schema(var_23, var_16)
    var_25 = var_24.one_of
    var_26 = len(var_25)
    assert var_26 == 2
    var_27 = var_24.one_of[var_12]
    var_28 = var_24.one_of[var_14]
    var_29 = 'default'
    var_30 = {var_1: var_2}
    var_31 = {var_1: var_4}
    var_32 = [var_30, var_31]
    var_33 = 'test'
    var_34 = {var_0: var_32, var_29: var_33}
    var_35 = module_0.Definitions()
    var_36 = module_1.one_of_from_json_schema(var_34, var_35)
    var_37 = []
    var_38 = {var_0: var_37}
    var_39 = module_0.Definitions()
    var_40 = module_1.one_of_from_json_schema(var_38, var_39)
    var_41 = var_40.one_of
    var_42 = len(var_41)
    assert var_42 == 0



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
    var_19 = module_0.Integer(minimum=var_17, maximum=var_18, exclusive_minimum=var_17, exclusive_maximum=var_18)
    var_20 = 'minimum'
    var_21 = 'maximum'
    var_22 = 'exclusiveMinimum'
    var_23 = 'exclusiveMaximum'
    var_24 = 'integer'
    var_25 = {var_9: var_24, var_20: var_17, var_21: var_18, var_22: var_17, var_23: var_18}
    var_26 = module_1.to_json_schema(var_19)
    var_27 = 0.1
    var_28 = module_0.Float(minimum=var_17, maximum=var_4, multiple_of=var_27)
    var_29 = 'multipleOf'
    var_30 = 'number'
    var_31 = {var_9: var_30, var_20: var_17, var_21: var_4, var_29: var_27}
    var_32 = module_1.to_json_schema(var_28)
    var_33 = module_0.Boolean()
    var_34 = 'boolean'
    var_35 = {var_9: var_34}
    var_36 = module_1.to_json_schema(var_33)
    var_37 = module_0.String()
    var_38 = True
    var_39 = module_0.Array(var_37, min_items=var_4, max_items=var_5, unique_items=var_38)
    var_40 = 'items'
    var_41 = 'minItems'
    var_42 = 'maxItems'
    var_43 = 'uniqueItems'
    var_44 = 'array'
    var_45 = {var_9: var_14}
    var_46 = True
    var_47 = {var_9: var_44, var_40: var_45, var_41: var_38, var_42: var_5, var_43: var_46}
    var_48 = module_1.to_json_schema(var_39)
    var_49 = 'name'
    var_50 = module_0.String()
    var_51 = {var_49: var_50}
    var_52 = [var_49]
    var_53 = module_0.Object(properties=var_51, min_properties=var_46, max_properties=var_5, required=var_52)
    var_54 = 'properties'
    var_55 = 'required'
    var_56 = 'minProperties'
    var_57 = 'maxProperties'
    var_58 = 'object'
    var_59 = {var_9: var_14}
    var_60 = {var_49: var_59}
    var_61 = [var_49]
    var_62 = {var_9: var_58, var_54: var_60, var_55: var_61, var_56: var_46, var_57: var_5}
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
    var_85 = {var_9: var_24}
    var_86 = [var_84, var_85]
    var_87 = {var_83: var_86}
    var_88 = module_1.to_json_schema(var_82)
    var_89 = module_0.String()
    var_90 = module_0.Integer()
    var_91 = [var_89, var_90]
    var_92 = module_2.AllOf(var_91)
    var_93 = 'allOf'
    var_94 = {var_9: var_14}
    var_95 = {var_9: var_24}
    var_96 = [var_94, var_95]
    var_97 = {var_93: var_96}
    var_98 = module_1.to_json_schema(var_92)
    var_99 = module_0.String()
    var_100 = module_0.Integer()
    var_101 = [var_99, var_100]
    var_102 = module_2.OneOf(var_101)
    var_103 = 'oneOf'
    var_104 = {var_9: var_14}
    var_105 = {var_9: var_24}
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
    var_123 = {var_9: var_24}
    var_124 = {var_9: var_34}
    var_125 = {var_119: var_122, var_120: var_123, var_121: var_124}
    var_126 = module_1.to_json_schema(var_118)
    var_127 = 'Test'
    var_128 = module_0.String()
    var_129 = {}
    var_130 = module_3.Reference(var_127, var_129)
    var_131 = '$ref'
    var_132 = 'components'
    var_133 = '#/components/schemas/Test'
    var_134 = 'schemas'
    var_135 = {var_9: var_14}
    var_136 = {var_127: var_135}
    var_137 = {var_134: var_136}
    var_138 = {var_131: var_133, var_132: var_137}
    var_139 = module_1.to_json_schema(var_130)



# Parsed testcases at query #11
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
    var_17 = []
    var_18 = {var_0: var_17}
    var_19 = module_0.Definitions()
    var_20 = module_1.type_from_json_schema(var_18, var_19)
    var_21 = 'allow_null'
    var_22 = []
    var_23 = True
    var_24 = {var_0: var_22, var_21: var_23}
    var_25 = module_0.Definitions()
    var_26 = module_1.type_from_json_schema(var_24, var_25)
    var_27 = 'properties'
    var_28 = 'object'
    var_29 = 'name'
    var_30 = 'age'
    var_31 = {var_0: var_1}
    var_32 = {var_0: var_5}
    var_33 = {var_29: var_31, var_30: var_32}
    var_34 = {var_0: var_28, var_27: var_33}
    var_35 = module_0.Definitions()
    var_36 = module_1.type_from_json_schema(var_34, var_35)



# Parsed testcases at query #12
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
    var_34 = 5
    var_35 = {var_0: var_32, var_29: var_33, var_30: var_18, var_31: var_34}
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



# Parsed testcases at query #13
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
    var_9 = [var_1, var_2, var_3]
    var_10 = {var_0: var_9, var_8: var_2}
    var_11 = module_0.Definitions()
    var_12 = module_1.enum_from_json_schema(var_10, var_11)
    var_13 = 1
    var_14 = 2
    var_15 = 3
    var_16 = [var_13, var_14, var_15]
    var_17 = {var_0: var_16}
    var_18 = module_0.Definitions()
    var_19 = module_1.enum_from_json_schema(var_17, var_18)
    var_20 = True
    var_21 = [var_13, var_1, var_20]
    var_22 = {var_0: var_21}
    var_23 = module_0.Definitions()
    var_24 = module_1.enum_from_json_schema(var_22, var_23)



# Parsed testcases at query #14
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
    var_13 = 'number'
    var_14 = {var_5: var_13}
    var_15 = module_0.from_json_schema(var_14)
    var_16 = 'integer'
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
    var_35 = 'const'
    var_36 = 'fixed_value'
    var_37 = {var_35: var_36}
    var_38 = module_0.from_json_schema(var_37)
    var_39 = 'allOf'
    var_40 = 'minLength'
    var_41 = 5
    var_42 = {var_5: var_6, var_40: var_41}
    var_43 = 'maxLength'
    var_44 = 10
    var_45 = {var_5: var_6, var_43: var_44}
    var_46 = [var_42, var_45]
    var_47 = {var_39: var_46}
    var_48 = module_0.from_json_schema(var_47)
    var_49 = var_48.schemas
    var_50 = len(var_49)
    assert var_50 == 2
    var_51 = 'anyOf'
    var_52 = {var_5: var_6}
    var_53 = {var_5: var_13}
    var_54 = [var_52, var_53]
    var_55 = {var_51: var_54}
    var_56 = module_0.from_json_schema(var_55)
    var_57 = var_56.schemas
    var_58 = len(var_57)
    assert var_58 == 2
    var_59 = 'oneOf'
    var_60 = {var_5: var_6}
    var_61 = {var_5: var_13}
    var_62 = [var_60, var_61]
    var_63 = {var_59: var_62}
    var_64 = module_0.from_json_schema(var_63)
    var_65 = var_64.schemas
    var_66 = len(var_65)
    assert var_66 == 2
    var_67 = 'not'
    var_68 = {var_5: var_6}
    var_69 = {var_67: var_68}
    var_70 = module_0.from_json_schema(var_69)
    var_71 = var_70.schema
    var_72 = 'if'
    var_73 = 'then'
    var_74 = 'else'
    var_75 = {var_5: var_6}
    var_76 = {var_40: var_41}
    var_77 = {var_5: var_13}
    var_78 = {var_72: var_75, var_73: var_76, var_74: var_77}
    var_79 = module_0.from_json_schema(var_78)
    var_80 = var_79.if_schema
    var_81 = var_79.then_schema
    var_82 = var_79.else_schema
    var_83 = 'pattern'
    var_84 = '^[a-z]+$'
    var_85 = {var_5: var_6, var_40: var_41, var_43: var_44, var_83: var_84}
    var_86 = module_0.from_json_schema(var_85)
    var_87 = var_86.schemas
    var_88 = len(var_87)
    assert var_88 == 4
    var_89 = {}
    var_90 = module_0.from_json_schema(var_89)



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
    assert var_88 == 2
    var_89 = {}
    var_90 = module_0.from_json_schema(var_89)



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
    var_140 = True
    var_141 = module_0.String()
    var_142 = 'null'
    var_143 = [var_14, var_142]
    var_144 = {var_9: var_143}
    var_145 = module_1.to_json_schema(var_141)



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
    var_69 = {var_0: var_59, var_55: var_64, var_56: var_65, var_57: var_35, var_58: var_10, var_6: var_68}
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
    var_30 = module_0.Boolean()
    var_31 = 'boolean'
    var_32 = [var_31, var_15]
    var_33 = {var_9: var_32}
    var_34 = module_1.to_json_schema(var_30)
    var_35 = 5
    var_36 = module_0.String()
    var_37 = module_0.Array(var_36, var_19, var_4, var_35)
    var_38 = 'minItems'
    var_39 = 'maxItems'
    var_40 = 'items'
    var_41 = 'additionalItems'
    var_42 = 'uniqueItems'
    var_43 = 'array'
    var_44 = {var_9: var_14}
    var_45 = {var_9: var_43, var_38: var_4, var_39: var_35, var_40: var_44, var_41: var_19, var_42: var_4}
    var_46 = module_1.to_json_schema(var_37)
    var_47 = 'name'
    var_48 = module_0.String()
    var_49 = {var_47: var_48}
    var_50 = module_0.Object(properties=var_49, additional_properties=var_19, min_properties=var_4, max_properties=var_35)
    var_51 = 'properties'
    var_52 = 'additionalProperties'
    var_53 = 'minProperties'
    var_54 = 'maxProperties'
    var_55 = 'object'
    var_56 = [var_55, var_15]
    var_57 = {var_9: var_14}
    var_58 = {var_47: var_57}
    var_59 = {var_9: var_56, var_51: var_58, var_52: var_19, var_53: var_4, var_54: var_35}
    var_60 = module_1.to_json_schema(var_50)
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
    var_82 = {var_9: var_27}
    var_83 = [var_81, var_82]
    var_84 = {var_80: var_83}
    var_85 = module_1.to_json_schema(var_79)
    var_86 = module_0.String()
    var_87 = module_0.Integer()
    var_88 = [var_86, var_87]
    var_89 = module_2.AllOf(var_88)
    var_90 = 'allOf'
    var_91 = {var_9: var_14}
    var_92 = {var_9: var_27}
    var_93 = [var_91, var_92]
    var_94 = {var_90: var_93}
    var_95 = module_1.to_json_schema(var_89)
    var_96 = 'Test'
    var_97 = module_0.String()
    var_98 = {var_96: var_97}
    var_99 = '$ref'
    var_100 = 'components'
    var_101 = '#/components/schemas/Test'
    var_102 = 'schemas'
    var_103 = {var_9: var_14}
    var_104 = {var_96: var_103}
    var_105 = {var_102: var_104}
    var_106 = {var_99: var_101, var_100: var_105}
    var_107 = module_0.String()
    var_108 = module_0.Integer()
    var_109 = module_0.Boolean()
    var_110 = module_2.IfThenElse(var_107, var_108, var_109)
    var_111 = 'if'
    var_112 = 'then'
    var_113 = 'else'
    var_114 = {var_9: var_14}
    var_115 = {var_9: var_27}
    var_116 = {var_9: var_31}
    var_117 = {var_111: var_114, var_112: var_115, var_113: var_116}
    var_118 = module_1.to_json_schema(var_110)
    var_119 = module_0.String()
    var_120 = module_2.Not(var_119)
    var_121 = 'not'
    var_122 = {var_9: var_14}
    var_123 = {var_121: var_122}
    var_124 = module_1.to_json_schema(var_120)



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = 'number'
    var_4 = [var_1, var_3]
    var_5 = {var_0: var_4}
    var_6 = 0
    var_7 = 1
    var_8 = 'nullable'
    var_9 = True
    var_10 = {var_0: var_1, var_8: var_9}
    var_11 = True
    var_12 = {var_8: var_11}
    var_13 = {}
    var_14 = 'properties'
    var_15 = 'object'
    var_16 = 'name'
    var_17 = 'age'
    var_18 = {var_0: var_1}
    var_19 = {var_0: var_3}
    var_20 = {var_16: var_18, var_17: var_19}
    var_21 = {var_0: var_15, var_14: var_20}
    var_22 = 'items'
    var_23 = 'array'
    var_24 = {var_0: var_1}
    var_25 = {var_0: var_23, var_22: var_24}
    var_26 = 'boolean'
    var_27 = {var_0: var_26}
    var_28 = 'integer'
    var_29 = {var_0: var_28}
    var_30 = {var_0: var_3}
    var_31 = 'minLength'
    var_32 = 'maxLength'
    var_33 = 'pattern'
    var_34 = 5
    var_35 = 10
    var_36 = '^[A-Za-z]+$'
    var_37 = {var_0: var_1, var_31: var_34, var_32: var_35, var_33: var_36}



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
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
    var_38 = 5
    var_39 = module_0.String()
    var_40 = module_0.Array(var_39, min_items=var_4, max_items=var_38)
    var_41 = 'minItems'
    var_42 = 'maxItems'
    var_43 = 'items'
    var_44 = 'array'
    var_45 = [var_44, var_15]
    var_46 = {var_9: var_14}
    var_47 = {var_9: var_45, var_41: var_4, var_42: var_38, var_43: var_46}
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
    var_97 = module_0.String()
    var_98 = module_0.Integer()
    var_99 = [var_97, var_98]
    var_100 = module_2.OneOf(var_99)
    var_101 = 'oneOf'
    var_102 = {var_9: var_14}
    var_103 = {var_9: var_24}
    var_104 = [var_102, var_103]
    var_105 = {var_101: var_104}
    var_106 = module_1.to_json_schema(var_100)
    var_107 = module_0.String()
    var_108 = module_2.Not(var_107)
    var_109 = 'not'
    var_110 = {var_9: var_14}
    var_111 = {var_109: var_110}
    var_112 = module_1.to_json_schema(var_108)
    var_113 = module_0.String()
    var_114 = module_0.Integer()
    var_115 = module_2.IfThenElse(var_113, var_114)
    var_116 = 'if'
    var_117 = 'then'
    var_118 = {var_9: var_14}
    var_119 = {var_9: var_24}
    var_120 = {var_116: var_118, var_117: var_119}
    var_121 = module_1.to_json_schema(var_115)
    var_122 = 'Test'
    var_123 = module_0.String()
    var_124 = {var_122: var_123}
    var_125 = '$ref'
    var_126 = 'components'
    var_127 = '#/components/schemas/Test'
    var_128 = 'schemas'
    var_129 = {var_9: var_14}
    var_130 = {var_122: var_129}
    var_131 = {var_128: var_130}
    var_132 = {var_125: var_127, var_126: var_131}
    var_133 = module_0.String()
    var_134 = {var_49: var_133}
    var_135 = [var_49]
    var_136 = module_3.Schema(var_134)
    var_137 = {var_9: var_14}
    var_138 = {var_49: var_137}
    var_139 = [var_49]
    var_140 = {var_9: var_56, var_54: var_138, var_55: var_139}
    var_141 = module_1.to_json_schema(var_136)
    var_142 = 'Person'
    var_143 = 'Address'
    var_144 = module_0.String()
    var_145 = {var_49: var_144}
    var_146 = module_0.Object(properties=var_145)
    var_147 = 'street'
    var_148 = module_0.String()
    var_149 = {var_147: var_148}
    var_150 = module_0.Object(properties=var_149)
    var_151 = {var_142: var_146, var_143: var_150}
    var_152 = {var_9: var_14}
    var_153 = {var_49: var_152}
    var_154 = {var_9: var_56, var_54: var_153}
    var_155 = {var_9: var_14}
    var_156 = {var_147: var_155}
    var_157 = {var_9: var_56, var_54: var_156}
    var_158 = {var_142: var_154, var_143: var_157}
    var_159 = {var_128: var_158}
    var_160 = {var_126: var_159}



# Parsed testcases at query #2
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
    var_74 = {var_24: var_29}
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



# Parsed testcases at query #3
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
    var_11 = 3
    var_12 = [var_9, var_10, var_11]
    var_13 = {var_1: var_12, var_8: var_10}
    var_14 = module_1.enum_from_json_schema(var_13, var_0)



# Parsed testcases at query #4
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
    var_9 = 'x'
    var_10 = 'y'
    var_11 = 'z'
    var_12 = [var_9, var_10, var_11]
    var_13 = {var_0: var_12, var_8: var_10}
    var_14 = module_0.Definitions()
    var_15 = module_1.enum_from_json_schema(var_13, var_14)
    var_16 = 1
    var_17 = 2
    var_18 = 3
    var_19 = [var_16, var_17, var_18]
    var_20 = {var_0: var_19}
    var_21 = module_0.Definitions()
    var_22 = module_1.enum_from_json_schema(var_20, var_21)
    var_23 = True
    var_24 = False
    var_25 = None
    var_26 = [var_23, var_24, var_25]
    var_27 = {var_0: var_26}
    var_28 = module_0.Definitions()
    var_29 = module_1.enum_from_json_schema(var_27, var_28)



# Parsed testcases at query #5
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



# Parsed testcases at query #6
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
    var_65 = var_64.properties[var_52]
    var_66 = var_64.properties[var_53]
    var_67 = {var_0: var_21}
    var_68 = module_0.Definitions()
    var_69 = module_1.from_json_schema_type(var_67, var_21, var_31, var_68)



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



# Parsed testcases at query #8
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
    var_21 = [var_18, var_20]
    var_22 = 'default_value'
    var_23 = {var_1: var_21, var_16: var_22}
    var_24 = module_1.one_of_from_json_schema(var_23, var_0)
    var_25 = var_24.one_of
    var_26 = len(var_25)
    assert var_26 == 2
    var_27 = '$ref'
    var_28 = '#/components/schemas/Test'
    var_29 = {var_27: var_28}
    var_30 = {var_2: var_5}
    var_31 = [var_29, var_30]
    var_32 = {var_1: var_31}
    var_33 = module_1.one_of_from_json_schema(var_32, var_0)
    var_34 = var_33.one_of
    var_35 = len(var_34)
    assert var_35 == 2
    var_36 = var_33.one_of[var_12]
    var_37 = var_33.one_of[var_14]



# Parsed testcases at query #9
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'allOf'
    var_1 = 'type'
    var_2 = 'minLength'
    var_3 = 'string'
    var_4 = 5
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'maxLength'
    var_7 = 10
    var_8 = {var_1: var_3, var_6: var_7}
    var_9 = [var_5, var_8]
    var_10 = {var_0: var_9}
    var_11 = module_0.Definitions()
    var_12 = module_1.all_of_from_json_schema(var_10, var_11)
    var_13 = var_12.all_of
    var_14 = len(var_13)
    assert var_14 == 2
    var_15 = 0
    var_16 = var_12.all_of[var_15]
    var_17 = 1
    var_18 = var_12.all_of[var_17]
    var_19 = module_0.Definitions()
    var_20 = 'integer'
    var_21 = {var_1: var_20}
    var_22 = '$ref'
    var_23 = '#/components/schemas/Test'
    var_24 = {var_22: var_23}
    var_25 = 'minimum'
    var_26 = {var_1: var_20, var_25: var_15}
    var_27 = [var_24, var_26]
    var_28 = {var_0: var_27}
    var_29 = module_1.all_of_from_json_schema(var_28, var_19)
    var_30 = var_29.all_of
    var_31 = len(var_30)
    assert var_31 == 2
    var_32 = var_29.all_of[var_15]
    var_33 = var_29.all_of[var_17]
    var_34 = 'default'
    var_35 = {var_1: var_3}
    var_36 = 'pattern'
    var_37 = '^[A-Z]+'
    var_38 = {var_1: var_3, var_36: var_37}
    var_39 = [var_35, var_38]
    var_40 = 'TEST'
    var_41 = {var_0: var_39, var_34: var_40}
    var_42 = module_0.Definitions()
    var_43 = module_1.all_of_from_json_schema(var_41, var_42)
    var_44 = []
    var_45 = {var_0: var_44}
    var_46 = module_0.Definitions()
    var_47 = module_1.all_of_from_json_schema(var_45, var_46)
    var_48 = var_47.all_of
    var_49 = len(var_48)
    assert var_49 == 0



