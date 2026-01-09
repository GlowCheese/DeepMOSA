####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------




# Parsed testcases at query #2
#--------------------------




# Parsed testcases at query #3
#--------------------------


import typesystem.json_schema as module_1
import typesystem.schemas as module_0


def test_case_0():
    var_0 = 'anyOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'integer'
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
    var_17 = {var_1: var_2}
    var_18 = {var_1: var_4}
    var_19 = [var_17, var_18]
    var_20 = 'test'
    var_21 = {var_0: var_19, var_16: var_20}
    var_22 = module_1.any_of_from_json_schema(var_21, var_8)
    var_23 = {var_1: var_2}
    var_24 = [var_23]
    var_25 = {var_0: var_24}
    var_26 = {var_1: var_4}
    var_27 = [var_25, var_26]
    var_28 = {var_0: var_27}
    var_29 = module_1.any_of_from_json_schema(var_28, var_8)
    var_30 = var_29.any_of
    var_31 = len(var_30)
    assert var_31 == 2
    var_32 = var_29.any_of[var_12]
    var_33 = var_29.any_of[var_14]
    var_34 = {var_1: var_2}
    var_35 = 'number'
    var_36 = {var_1: var_35}
    var_37 = 'boolean'
    var_38 = {var_1: var_37}
    var_39 = [var_34, var_36, var_38]
    var_40 = {var_0: var_39}
    var_41 = module_1.any_of_from_json_schema(var_40, var_8)
    var_42 = var_41.any_of
    var_43 = len(var_42)
    assert var_43 == 3
    var_44 = var_41.any_of[var_12]
    var_45 = var_41.any_of[var_14]
    var_46 = 2
    var_47 = var_41.any_of[var_46]
    var_48 = 'minLength'
    var_49 = 5
    var_50 = {var_1: var_2, var_48: var_49}
    var_51 = 'minimum'
    var_52 = {var_1: var_4, var_51: var_12}
    var_53 = [var_50, var_52]
    var_54 = {var_0: var_53}
    var_55 = module_1.any_of_from_json_schema(var_54, var_8)
    var_56 = var_55.any_of[var_12]
    var_57 = var_55.any_of[var_14]
    var_58 = 'All tests passed!'
    var_59 = print(var_58)



# Parsed testcases at query #4
#--------------------------




# Parsed testcases at query #5
#--------------------------




# Parsed testcases at query #6
#--------------------------




# Parsed testcases at query #7
#--------------------------




# Parsed testcases at query #8
#--------------------------




# Parsed testcases at query #9
#--------------------------


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
    var_11 = 5
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
    var_63 = {var_0: var_57, var_53: var_60, var_54: var_47, var_55: var_9, var_56: var_61, var_6: var_62}
    var_64 = False
    var_65 = module_0.Definitions()
    var_66 = module_1.from_json_schema_type(var_63, var_57, var_64, var_65)
    var_67 = 'null'
    var_68 = {var_0: var_67}
    var_69 = True
    var_70 = module_0.Definitions()
    var_71 = module_1.from_json_schema_type(var_68, var_67, var_69, var_70)
    var_72 = {}
    var_73 = 'invalid'
    var_74 = False
    var_75 = module_0.Definitions()
    var_76 = module_1.from_json_schema_type(var_72, var_73, var_74, var_75)
    var_77 = 'All tests passed.'
    var_78 = print(var_77)



# Parsed testcases at query #10
#--------------------------




# Parsed testcases at query #11
#--------------------------




# Parsed testcases at query #12
#--------------------------




# Parsed testcases at query #13
#--------------------------




# Parsed testcases at query #14
#--------------------------




# Parsed testcases at query #15
#--------------------------




# Parsed testcases at query #16
#--------------------------


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
    var_11 = 'number'
    var_12 = False
    var_13 = {var_0: var_12, var_1: var_7, var_2: var_12, var_3: var_7, var_4: var_8, var_5: var_9}
    var_14 = 'integer'
    var_15 = False
    var_16 = 'minLength'
    var_17 = 'maxLength'
    var_18 = 'format'
    var_19 = 'pattern'
    var_20 = 1
    var_21 = 'email'
    var_22 = '^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$'
    var_23 = 'test@example.com'
    var_24 = {var_16: var_20, var_17: var_7, var_18: var_21, var_19: var_22, var_5: var_23}
    var_25 = 'string'
    var_26 = False
    var_27 = True
    var_28 = {var_5: var_27}
    var_29 = 'boolean'
    var_30 = False
    var_31 = 'items'
    var_32 = 'minItems'
    var_33 = 'maxItems'
    var_34 = 'uniqueItems'
    var_35 = 'type'
    var_36 = {var_35: var_25}
    var_37 = True
    var_38 = 'item1'
    var_39 = 'item2'
    var_40 = [var_38, var_39]
    var_41 = {var_31: var_36, var_32: var_27, var_33: var_7, var_34: var_37, var_5: var_40}
    var_42 = 'array'
    var_43 = False
    var_44 = 'properties'
    var_45 = 'minProperties'
    var_46 = 'maxProperties'
    var_47 = 'required'
    var_48 = 'name'
    var_49 = 'age'
    var_50 = {var_35: var_25}
    var_51 = {var_35: var_14}
    var_52 = {var_48: var_50, var_49: var_51}
    var_53 = [var_48]
    var_54 = 'John'
    var_55 = 30
    var_56 = {var_48: var_54, var_49: var_55}
    var_57 = {var_44: var_52, var_45: var_37, var_46: var_8, var_47: var_53, var_5: var_56}
    var_58 = 'object'
    var_59 = False
    var_60 = {}
    var_61 = 'null'
    var_62 = True
    var_63 = {}
    var_64 = 'unknown'
    var_65 = False
    var_66 = 'All test cases pass'
    var_67 = print(var_66)



# Parsed testcases at query #17
#--------------------------




# Parsed testcases at query #18
#--------------------------




# Parsed testcases at query #19
#--------------------------




# Parsed testcases at query #20
#--------------------------




# Parsed testcases at query #21
#--------------------------




# Parsed testcases at query #22
#--------------------------




# Parsed testcases at query #23
#--------------------------




# Parsed testcases at query #24
#--------------------------




# Parsed testcases at query #25
#--------------------------




####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------




# Parsed testcases at query #2
#--------------------------




# Parsed testcases at query #3
#--------------------------




# Parsed testcases at query #4
#--------------------------




# Parsed testcases at query #5
#--------------------------




# Parsed testcases at query #6
#--------------------------




# Parsed testcases at query #7
#--------------------------




# Parsed testcases at query #8
#--------------------------




# Parsed testcases at query #9
#--------------------------




# Parsed testcases at query #10
#--------------------------




# Parsed testcases at query #11
#--------------------------




# Parsed testcases at query #12
#--------------------------



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
    var_17 = 'boolean'
    var_18 = {var_3: var_17}
    var_19 = 'null'
    var_20 = {var_3: var_19}
    var_21 = {var_0: var_18, var_2: var_20}
    var_22 = module_1.if_then_else_from_json_schema(var_21, var_12)
    var_23 = var_22.if_clause
    var_24 = var_22.else_clause
    var_25 = 'integer'
    var_26 = {var_3: var_25}
    var_27 = 'minimum'
    var_28 = 0
    var_29 = {var_3: var_25, var_27: var_28}
    var_30 = {var_0: var_26, var_1: var_29}
    var_31 = module_1.if_then_else_from_json_schema(var_30, var_12)
    var_32 = var_31.if_clause
    var_33 = var_31.then_clause
    var_34 = 'object'
    var_35 = {var_3: var_34}
    var_36 = 'array'
    var_37 = {var_3: var_36}
    var_38 = 'minItems'
    var_39 = 2
    var_40 = {var_3: var_36, var_38: var_39}
    var_41 = {var_3: var_4}
    var_42 = {var_0: var_37, var_1: var_40, var_2: var_41}
    var_43 = {var_3: var_17}
    var_44 = {var_0: var_35, var_1: var_42, var_2: var_43}
    var_45 = module_1.if_then_else_from_json_schema(var_44, var_12)
    var_46 = var_45.if_clause
    var_47 = var_45.then_clause
    var_48 = var_45.then_clause.if_clause
    var_49 = var_45.then_clause.then_clause
    var_50 = var_45.then_clause.else_clause
    var_51 = var_45.else_clause
    var_52 = {var_3: var_4}
    var_53 = 'default'
    var_54 = 'default_then'
    var_55 = {var_3: var_4, var_53: var_54}
    var_56 = 42
    var_57 = {var_3: var_9, var_53: var_56}
    var_58 = {var_0: var_52, var_1: var_55, var_2: var_57}
    var_59 = module_1.if_then_else_from_json_schema(var_58, var_12)
    var_60 = 'allOf'
    var_61 = {var_3: var_4}
    var_62 = 3
    var_63 = {var_6: var_62}
    var_64 = [var_61, var_63]
    var_65 = {var_60: var_64}
    var_66 = 'pattern'
    var_67 = '^[A-Z]+$'
    var_68 = {var_3: var_4, var_66: var_67}
    var_69 = '^[a-z]+$'
    var_70 = {var_3: var_4, var_66: var_69}
    var_71 = {var_0: var_65, var_1: var_68, var_2: var_70}
    var_72 = module_1.if_then_else_from_json_schema(var_71, var_12)
    var_73 = var_72.if_clause
    var_74 = var_72.if_clause.all_of
    var_75 = len(var_74)
    assert var_75 == 2
    var_76 = var_72.then_clause
    var_77 = var_72.else_clause
    var_78 = 'All tests passed!'
    var_79 = print(var_78)



# Parsed testcases at query #13
#--------------------------




# Parsed testcases at query #14
#--------------------------



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
    var_10 = 'integer'
    var_11 = {var_4: var_10}
    var_12 = {var_1: var_6, var_2: var_9, var_3: var_11}
    var_13 = module_1.if_then_else_from_json_schema(var_12, var_0)
    var_14 = var_13.if_clause
    var_15 = var_13.then_clause
    var_16 = var_13.else_clause



# Parsed testcases at query #15
#--------------------------




# Parsed testcases at query #16
#--------------------------



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
    var_13 = 100
    var_14 = {var_1: var_12, var_10: var_8, var_11: var_13}
    var_15 = module_1.from_json_schema_type(var_14, var_12, var_8, var_0)
    var_16 = 'exclusiveMinimum'
    var_17 = 'exclusiveMaximum'
    var_18 = 'number'
    var_19 = {var_1: var_18, var_16: var_8, var_17: var_13}
    var_20 = module_1.from_json_schema_type(var_19, var_18, var_8, var_0)
    var_21 = 'default'
    var_22 = 'boolean'
    var_23 = True
    var_24 = {var_1: var_22, var_21: var_23}
    var_25 = module_1.from_json_schema_type(var_24, var_22, var_8, var_0)
    var_26 = 'items'
    var_27 = 'minItems'
    var_28 = 'maxItems'
    var_29 = 'array'
    var_30 = {var_1: var_4}
    var_31 = {var_1: var_29, var_26: var_30, var_27: var_23, var_28: var_5}
    var_32 = module_1.from_json_schema_type(var_31, var_29, var_8, var_0)
    var_33 = var_32.items
    var_34 = 'properties'
    var_35 = 'required'
    var_36 = 'object'
    var_37 = 'name'
    var_38 = {var_1: var_4}
    var_39 = {var_37: var_38}
    var_40 = [var_37]
    var_41 = {var_1: var_36, var_34: var_39, var_35: var_40}
    var_42 = module_1.from_json_schema_type(var_41, var_36, var_8, var_0)
    var_43 = var_42.properties[var_37]
    var_44 = 'allow_null'
    var_45 = {var_1: var_4, var_44: var_23}
    var_46 = module_1.from_json_schema_type(var_45, var_4, var_23, var_0)
    var_47 = 'pattern'
    var_48 = '^[a-z]+$'
    var_49 = {var_1: var_4, var_47: var_48}
    var_50 = module_1.from_json_schema_type(var_49, var_4, var_8, var_0)
    var_51 = 'additionalItems'
    var_52 = {var_1: var_4}
    var_53 = {var_1: var_12}
    var_54 = [var_52, var_53]
    var_55 = {var_1: var_29, var_26: var_54, var_51: var_8}
    var_56 = module_1.from_json_schema_type(var_55, var_29, var_8, var_0)
    var_57 = var_56.items
    var_58 = var_56.items
    var_59 = len(var_58)
    assert var_59 == 2
    var_60 = var_56.items[var_8]
    var_61 = var_56.items[var_23]
    var_62 = 'patternProperties'
    var_63 = {var_1: var_4}
    var_64 = {var_48: var_63}
    var_65 = {var_1: var_36, var_62: var_64}
    var_66 = module_1.from_json_schema_type(var_65, var_36, var_8, var_0)
    var_67 = var_66.pattern_properties[var_48]
    var_68 = 'propertyNames'
    var_69 = {var_47: var_48}
    var_70 = {var_1: var_36, var_68: var_69}
    var_71 = module_1.from_json_schema_type(var_70, var_36, var_8, var_0)
    var_72 = var_71.property_names
    var_73 = 'additionalProperties'
    var_74 = {var_1: var_36, var_73: var_8}
    var_75 = module_1.from_json_schema_type(var_74, var_36, var_8, var_0)
    var_76 = {var_1: var_4}
    var_77 = {var_1: var_36, var_73: var_76}
    var_78 = module_1.from_json_schema_type(var_77, var_36, var_8, var_0)
    var_79 = var_78.additional_properties
    var_80 = 'uniqueItems'
    var_81 = {var_1: var_29, var_80: var_23}
    var_82 = module_1.from_json_schema_type(var_81, var_29, var_8, var_0)
    var_83 = 'format'
    var_84 = 'email'
    var_85 = {var_1: var_4, var_83: var_84}
    var_86 = module_1.from_json_schema_type(var_85, var_4, var_8, var_0)
    var_87 = 'multipleOf'
    var_88 = 2
    var_89 = {var_1: var_12, var_87: var_88}
    var_90 = module_1.from_json_schema_type(var_89, var_12, var_8, var_0)
    var_91 = 0.5
    var_92 = {var_1: var_18, var_87: var_91}
    var_93 = module_1.from_json_schema_type(var_92, var_18, var_8, var_0)
    var_94 = 'hello'
    var_95 = {var_1: var_4, var_21: var_94}
    var_96 = module_1.from_json_schema_type(var_95, var_4, var_8, var_0)
    var_97 = 42
    var_98 = {var_1: var_12, var_21: var_97}
    var_99 = module_1.from_json_schema_type(var_98, var_12, var_8, var_0)
    var_100 = {var_1: var_22, var_21: var_8}
    var_101 = module_1.from_json_schema_type(var_100, var_22, var_8, var_0)
    var_102 = []
    var_103 = {var_1: var_29, var_21: var_102}
    var_104 = module_1.from_json_schema_type(var_103, var_29, var_8, var_0)
    var_105 = {}
    var_106 = {var_1: var_36, var_21: var_105}
    var_107 = module_1.from_json_schema_type(var_106, var_36, var_8, var_0)
    var_108 = 'allow_blank'
    var_109 = {var_1: var_4, var_108: var_23}
    var_110 = module_1.from_json_schema_type(var_109, var_4, var_8, var_0)
    var_111 = {var_1: var_4, var_108: var_8}
    var_112 = module_1.from_json_schema_type(var_111, var_4, var_8, var_0)
    var_113 = 'min_length'
    var_114 = {var_1: var_4, var_113: var_8}
    var_115 = module_1.from_json_schema_type(var_114, var_4, var_8, var_0)
    var_116 = {var_1: var_4, var_113: var_23}
    var_117 = module_1.from_json_schema_type(var_116, var_4, var_8, var_0)
    var_118 = 'coerce_types'
    var_119 = {var_1: var_4, var_118: var_8}
    var_120 = module_1.from_json_schema_type(var_119, var_4, var_8, var_0)
    var_121 = {var_1: var_12, var_118: var_8}
    var_122 = module_1.from_json_schema_type(var_121, var_12, var_8, var_0)
    var_123 = {var_1: var_18, var_118: var_8}
    var_124 = module_1.from_json_schema_type(var_123, var_18, var_8, var_0)
    var_125 = {var_1: var_22, var_118: var_8}
    var_126 = module_1.from_json_schema_type(var_125, var_22, var_8, var_0)
    var_127 = {var_1: var_29, var_118: var_8}
    var_128 = module_1.from_json_schema_type(var_127, var_29, var_8, var_0)
    var_129 = {var_1: var_36, var_118: var_8}
    var_130 = module_1.from_json_schema_type(var_129, var_36, var_8, var_0)
    var_131 = 'null'
    var_132 = {var_1: var_131}
    var_133 = module_1.from_json_schema_type(var_132, var_131, var_23, var_0)
    var_134 = [var_4, var_12]
    var_135 = {var_1: var_134}
    var_136 = module_1.from_json_schema(var_135, var_0)
    var_137 = var_136.any_of
    var_138 = len(var_137)
    assert var_138 == 2
    var_139 = var_136.any_of[var_8]
    var_140 = var_136.any_of[var_23]
    var_141 = {}
    var_142 = module_1.from_json_schema(var_141, var_0)
    var_143 = 'enum'
    var_144 = 'red'
    var_145 = 'green'
    var_146 = 'blue'
    var_147 = [var_144, var_145, var_146]
    var_148 = {var_143: var_147}
    var_149 = module_1.from_json_schema(var_148, var_0)
    var_150 = 'const'
    var_151 = 'fixed'
    var_152 = {var_150: var_151}
    var_153 = module_1.from_json_schema(var_152, var_0)
    var_154 = 'allOf'
    var_155 = {var_1: var_4}
    var_156 = {var_2: var_5}
    var_157 = [var_155, var_156]
    var_158 = {var_154: var_157}
    var_159 = module_1.from_json_schema(var_158, var_0)
    var_160 = var_159.all_of
    var_161 = len(var_160)
    assert var_161 == 2
    var_162 = 'anyOf'
    var_163 = {var_1: var_4}
    var_164 = {var_1: var_12}
    var_165 = [var_163, var_164]
    var_166 = {var_162: var_165}
    var_167 = module_1.from_json_schema(var_166, var_0)
    var_168 = var_167.any_of
    var_169 = len(var_168)
    assert var_169 == 2



