####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'oneOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'integer'
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
    var_18 = 'number'
    var_19 = {var_1: var_18}
    var_20 = [var_17, var_19]
    var_21 = 'test_value'
    var_22 = {var_0: var_20, var_16: var_21}
    var_23 = module_0.Definitions()
    var_24 = module_1.one_of_from_json_schema(var_22, var_23)
    var_25 = 'boolean'
    var_26 = {var_1: var_25}
    var_27 = [var_26]
    var_28 = {var_0: var_27}
    var_29 = module_0.Definitions()
    var_30 = module_1.one_of_from_json_schema(var_28, var_29)
    var_31 = var_30.one_of
    var_32 = len(var_31)
    assert var_32 == 1
    var_33 = var_30.one_of[var_12]
    var_34 = 'properties'
    var_35 = 'object'
    var_36 = 'name'
    var_37 = {var_1: var_2}
    var_38 = {var_36: var_37}
    var_39 = {var_1: var_35, var_34: var_38}
    var_40 = 'items'
    var_41 = 'array'
    var_42 = {var_1: var_4}
    var_43 = {var_1: var_41, var_40: var_42}
    var_44 = [var_39, var_43]
    var_45 = {var_0: var_44}
    var_46 = module_0.Definitions()
    var_47 = module_1.one_of_from_json_schema(var_45, var_46)
    var_48 = var_47.one_of
    var_49 = len(var_48)
    assert var_49 == 2
    var_50 = var_47.one_of[var_12]
    var_51 = var_47.one_of[var_14]
    var_52 = {var_1: var_2}
    var_53 = {var_1: var_4}
    var_54 = [var_52, var_53]
    var_55 = {var_0: var_54}
    var_56 = module_0.Definitions()
    var_57 = module_1.one_of_from_json_schema(var_55, var_56)
    var_58 = module_0.Definitions()
    var_59 = '$ref'
    var_60 = '#/definitions/StringType'
    var_61 = {var_59: var_60}
    var_62 = {var_1: var_4}
    var_63 = [var_61, var_62]
    var_64 = {var_0: var_63}
    var_65 = module_1.one_of_from_json_schema(var_64, var_58)
    var_66 = var_65.one_of
    var_67 = len(var_66)
    assert var_67 == 2
    var_68 = var_65.one_of[var_12]
    var_69 = var_65.one_of[var_14]



# Parsed testcases at query #2
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'oneOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'integer'
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
    var_18 = 'number'
    var_19 = {var_1: var_18}
    var_20 = [var_17, var_19]
    var_21 = 'test'
    var_22 = {var_0: var_20, var_16: var_21}
    var_23 = module_0.Definitions()
    var_24 = module_1.one_of_from_json_schema(var_22, var_23)
    var_25 = 'properties'
    var_26 = 'object'
    var_27 = 'name'
    var_28 = {var_1: var_2}
    var_29 = {var_27: var_28}
    var_30 = {var_1: var_26, var_25: var_29}
    var_31 = 'items'
    var_32 = 'array'
    var_33 = {var_1: var_4}
    var_34 = {var_1: var_32, var_31: var_33}
    var_35 = [var_30, var_34]
    var_36 = {var_0: var_35}
    var_37 = module_0.Definitions()
    var_38 = module_1.one_of_from_json_schema(var_36, var_37)
    var_39 = var_38.one_of
    var_40 = len(var_39)
    assert var_40 == 2
    var_41 = var_38.one_of[var_12]
    var_42 = var_38.one_of[var_14]
    var_43 = 'boolean'
    var_44 = {var_1: var_43}
    var_45 = [var_44]
    var_46 = {var_0: var_45}
    var_47 = module_0.Definitions()
    var_48 = module_1.one_of_from_json_schema(var_46, var_47)
    var_49 = var_48.one_of
    var_50 = len(var_49)
    assert var_50 == 1
    var_51 = var_48.one_of[var_12]
    var_52 = {var_1: var_2}
    var_53 = {var_1: var_4}
    var_54 = [var_52, var_53]
    var_55 = {var_0: var_54}
    var_56 = module_0.Definitions()
    var_57 = module_1.one_of_from_json_schema(var_55, var_56)
    var_58 = module_0.Definitions()
    var_59 = '$ref'
    var_60 = '#/definitions/StringType'
    var_61 = {var_59: var_60}
    var_62 = {var_1: var_4}
    var_63 = [var_61, var_62]
    var_64 = {var_0: var_63}
    var_65 = module_1.one_of_from_json_schema(var_64, var_58)
    var_66 = var_65.one_of
    var_67 = len(var_66)
    assert var_67 == 2
    var_68 = var_65.one_of[var_12]
    var_69 = var_65.one_of[var_14]



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = 'number'
    var_4 = [var_1, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'null'
    var_7 = {var_0: var_6}
    var_8 = set()
    var_9 = [var_1, var_6]
    var_10 = {var_0: var_9}
    var_11 = {}
    var_12 = [var_6]
    var_13 = {var_0: var_12}
    var_14 = set()
    var_15 = 'integer'
    var_16 = [var_15, var_3]
    var_17 = {var_0: var_16}
    var_18 = {var_0: var_15}
    var_19 = 'boolean'
    var_20 = {var_0: var_19}
    var_21 = 'object'
    var_22 = {var_0: var_21}
    var_23 = 'array'
    var_24 = {var_0: var_23}
    var_25 = [var_1, var_3, var_6, var_21]
    var_26 = {var_0: var_25}
    var_27 = [var_15, var_3, var_6]
    var_28 = {var_0: var_27}
    var_29 = [var_1, var_3, var_15, var_19, var_21, var_23, var_6]
    var_30 = {var_0: var_29}



# Parsed testcases at query #4
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'oneOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'integer'
    var_5 = {var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = {var_0: var_6}
    var_8 = 0
    var_9 = 1
    var_10 = 'default'
    var_11 = {var_1: var_2}
    var_12 = 'number'
    var_13 = {var_1: var_12}
    var_14 = [var_11, var_13]
    var_15 = 'test'
    var_16 = {var_0: var_14, var_10: var_15}
    var_17 = 'properties'
    var_18 = 'object'
    var_19 = 'name'
    var_20 = {var_1: var_2}
    var_21 = {var_19: var_20}
    var_22 = {var_1: var_18, var_17: var_21}
    var_23 = 'items'
    var_24 = 'array'
    var_25 = {var_1: var_4}
    var_26 = {var_1: var_24, var_23: var_25}
    var_27 = [var_22, var_26]
    var_28 = {var_0: var_27}
    var_29 = 'boolean'
    var_30 = {var_1: var_29}
    var_31 = [var_30]
    var_32 = {var_0: var_31}
    var_33 = {var_1: var_2}
    var_34 = {var_1: var_4}
    var_35 = [var_33, var_34]
    var_36 = {var_0: var_35}
    var_37 = {var_1: var_2}
    var_38 = {var_1: var_12}
    var_39 = [var_37, var_38]
    var_40 = {var_0: var_39}
    var_41 = {var_1: var_29}
    var_42 = [var_40, var_41]
    var_43 = {var_0: var_42}
    var_44 = module_0.Definitions()
    var_45 = '$ref'
    var_46 = '#/components/schemas/StringSchema'
    var_47 = {var_45: var_46}
    var_48 = '#/components/schemas/IntSchema'
    var_49 = {var_45: var_48}
    var_50 = [var_47, var_49]
    var_51 = {var_0: var_50}
    var_52 = module_1.one_of_from_json_schema(var_51, var_44)
    var_53 = var_52.one_of
    var_54 = len(var_53)
    assert var_54 == 2
    var_55 = var_52.one_of[var_8]
    var_56 = var_52.one_of[var_9]



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
    var_11 = 'test_default'
    var_12 = {var_0: var_6, var_1: var_8, var_2: var_10, var_3: var_11}
    var_13 = {var_4: var_5}
    var_14 = {var_4: var_7}
    var_15 = {var_0: var_13, var_1: var_14}
    var_16 = {var_4: var_5}
    var_17 = {var_4: var_9}
    var_18 = {var_0: var_16, var_2: var_17}
    var_19 = 'object'
    var_20 = {var_4: var_19}
    var_21 = {var_0: var_20}
    var_22 = 'items'
    var_23 = 'array'
    var_24 = {var_4: var_5}
    var_25 = {var_4: var_23, var_22: var_24}
    var_26 = 'properties'
    var_27 = 'name'
    var_28 = {var_4: var_5}
    var_29 = {var_27: var_28}
    var_30 = {var_4: var_19, var_26: var_29}
    var_31 = 'enum'
    var_32 = 1
    var_33 = 2
    var_34 = 3
    var_35 = [var_32, var_33, var_34]
    var_36 = {var_31: var_35}
    var_37 = 42
    var_38 = {var_0: var_25, var_1: var_30, var_2: var_36, var_3: var_37}
    var_39 = True
    var_40 = False
    var_41 = True
    var_42 = {var_0: var_39, var_1: var_40, var_2: var_41}



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'enum'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'red'
    var_7 = 'green'
    var_8 = 'blue'
    var_9 = [var_6, var_7, var_8]
    var_10 = {var_0: var_9}
    var_11 = 'two'
    var_12 = True
    var_13 = None
    var_14 = [var_1, var_11, var_3, var_12, var_13]
    var_15 = {var_0: var_14}
    var_16 = 'default'
    var_17 = 'a'
    var_18 = 'b'
    var_19 = 'c'
    var_20 = [var_17, var_18, var_19]
    var_21 = {var_0: var_20, var_16: var_18}
    var_22 = 10
    var_23 = 20
    var_24 = 30
    var_25 = [var_22, var_23, var_24]
    var_26 = {var_0: var_25}
    var_27 = 'only'
    var_28 = [var_27]
    var_29 = {var_0: var_28}



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'allOf'
    var_1 = 'type'
    var_2 = 'minLength'
    var_3 = 'string'
    var_4 = 1
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'maxLength'
    var_7 = 10
    var_8 = {var_1: var_3, var_6: var_7}
    var_9 = [var_5, var_8]
    var_10 = {var_0: var_9}
    var_11 = 0
    var_12 = 'default'
    var_13 = 'minimum'
    var_14 = 'integer'
    var_15 = {var_1: var_14, var_13: var_11}
    var_16 = 'maximum'
    var_17 = 100
    var_18 = {var_1: var_14, var_16: var_17}
    var_19 = [var_15, var_18]
    var_20 = 50
    var_21 = {var_0: var_19, var_12: var_20}
    var_22 = 'boolean'
    var_23 = {var_1: var_22}
    var_24 = 'const'
    var_25 = True
    var_26 = {var_24: var_25}
    var_27 = [var_23, var_26]
    var_28 = {var_0: var_27}
    var_29 = 'properties'
    var_30 = 'object'
    var_31 = 'name'
    var_32 = {var_1: var_3}
    var_33 = {var_31: var_32}
    var_34 = {var_1: var_30, var_29: var_33}
    var_35 = 'required'
    var_36 = [var_31]
    var_37 = {var_1: var_30, var_35: var_36}
    var_38 = [var_34, var_37]
    var_39 = {var_0: var_38}
    var_40 = {var_1: var_3}
    var_41 = [var_40]
    var_42 = {var_0: var_41}
    var_43 = 'items'
    var_44 = 'array'
    var_45 = {var_1: var_3}
    var_46 = {var_1: var_44, var_43: var_45}
    var_47 = 'minItems'
    var_48 = {var_47: var_25}
    var_49 = [var_46, var_48]
    var_50 = {var_0: var_49}



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'type'
    var_1 = 'minLength'
    var_2 = 'string'
    var_3 = 1
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'number'
    var_6 = [var_2, var_5]
    var_7 = {var_0: var_6}
    var_8 = 'null'
    var_9 = [var_2, var_8]
    var_10 = {var_0: var_9}
    var_11 = {var_0: var_8}
    var_12 = {var_0: var_8}
    var_13 = 'minimum'
    var_14 = 'integer'
    var_15 = 0
    var_16 = {var_0: var_14, var_13: var_15}
    var_17 = 'boolean'
    var_18 = {var_0: var_17}
    var_19 = 'items'
    var_20 = 'array'
    var_21 = {var_0: var_2}
    var_22 = {var_0: var_20, var_19: var_21}
    var_23 = 'properties'
    var_24 = 'object'
    var_25 = 'name'
    var_26 = {var_0: var_2}
    var_27 = {var_25: var_26}
    var_28 = {var_0: var_24, var_23: var_27}
    var_29 = {var_0: var_5, var_13: var_15}
    var_30 = [var_2, var_14, var_8]
    var_31 = {var_0: var_30}
    var_32 = {}



# Parsed testcases at query #9
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
    var_30 = 'value'
    var_31 = {var_29: var_30}
    var_32 = module_0.from_json_schema(var_31)
    var_33 = 'minLength'
    var_34 = 'maxLength'
    var_35 = 5
    var_36 = 10
    var_37 = {var_4: var_5, var_33: var_35, var_34: var_36}
    var_38 = module_0.from_json_schema(var_37)
    var_39 = 'minimum'
    var_40 = 'maximum'
    var_41 = 100
    var_42 = {var_4: var_11, var_39: var_2, var_40: var_41}
    var_43 = module_0.from_json_schema(var_42)
    var_44 = 'minItems'
    var_45 = 'maxItems'
    var_46 = {var_4: var_17, var_44: var_0, var_45: var_35}
    var_47 = module_0.from_json_schema(var_46)
    var_48 = 'properties'
    var_49 = 'name'
    var_50 = 'age'
    var_51 = {var_4: var_5}
    var_52 = {var_4: var_8}
    var_53 = {var_49: var_51, var_50: var_52}
    var_54 = {var_4: var_20, var_48: var_53}
    var_55 = module_0.from_json_schema(var_54)
    var_56 = 'allOf'
    var_57 = {var_4: var_5}
    var_58 = {var_33: var_35}
    var_59 = [var_57, var_58]
    var_60 = {var_56: var_59}
    var_61 = module_0.from_json_schema(var_60)
    var_62 = 'anyOf'
    var_63 = {var_4: var_5}
    var_64 = {var_4: var_8}
    var_65 = [var_63, var_64]
    var_66 = {var_62: var_65}
    var_67 = module_0.from_json_schema(var_66)
    var_68 = 'oneOf'
    var_69 = {var_4: var_5}
    var_70 = {var_4: var_8}
    var_71 = [var_69, var_70]
    var_72 = {var_68: var_71}
    var_73 = module_0.from_json_schema(var_72)
    var_74 = 'not'
    var_75 = {var_4: var_5}
    var_76 = {var_74: var_75}
    var_77 = module_0.from_json_schema(var_76)
    var_78 = 'if'
    var_79 = 'then'
    var_80 = 'else'
    var_81 = {var_4: var_5}
    var_82 = {var_33: var_35}
    var_83 = {var_4: var_8}
    var_84 = {var_78: var_81, var_79: var_82, var_80: var_83}
    var_85 = module_0.from_json_schema(var_84)
    var_86 = module_1.Definitions()
    var_87 = '$ref'
    var_88 = '#/components/schemas/User'
    var_89 = {var_87: var_88}
    var_90 = module_0.from_json_schema(var_89, var_86)
    var_91 = 'a'
    var_92 = 'b'
    var_93 = 'c'
    var_94 = [var_91, var_92, var_93]
    var_95 = {var_4: var_5, var_23: var_94, var_33: var_0}
    var_96 = module_0.from_json_schema(var_95)
    var_97 = {}
    var_98 = module_0.from_json_schema(var_97)
    var_99 = 'pattern'
    var_100 = '^[a-z]+$'
    var_101 = {var_4: var_5, var_99: var_100}
    var_102 = module_0.from_json_schema(var_101)
    var_103 = 'components'
    var_104 = 'schemas'
    var_105 = 'User'
    var_106 = {var_4: var_5}
    var_107 = {var_105: var_106}
    var_108 = {var_104: var_107}
    var_109 = {var_4: var_20, var_103: var_108}
    var_110 = module_0.from_json_schema(var_109)



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
    var_4 = module_0.String()
    var_5 = module_1.to_json_schema(var_4)
    var_6 = 'minLength'
    var_7 = 'maxLength'
    var_8 = True
    var_9 = module_0.String()
    var_10 = module_1.to_json_schema(var_9)
    var_11 = 5
    var_12 = 10
    var_13 = module_0.String(max_length=var_12, min_length=var_11)
    var_14 = module_1.to_json_schema(var_13)
    var_15 = '^[a-z]+$'
    var_16 = module_0.String(pattern=var_15)
    var_17 = module_1.to_json_schema(var_16)
    var_18 = 'email'
    var_19 = module_0.String(format=var_18)
    var_20 = module_1.to_json_schema(var_19)
    var_21 = module_0.Integer()
    var_22 = module_1.to_json_schema(var_21)
    var_23 = module_0.Integer()
    var_24 = module_1.to_json_schema(var_23)
    var_25 = 0
    var_26 = 100
    var_27 = module_0.Integer(minimum=var_25, maximum=var_26)
    var_28 = module_1.to_json_schema(var_27)
    var_29 = module_0.Integer(exclusive_minimum=var_25, exclusive_maximum=var_26)
    var_30 = module_1.to_json_schema(var_29)
    var_31 = module_0.Integer(multiple_of=var_11)
    var_32 = module_1.to_json_schema(var_31)
    var_33 = module_0.Float()
    var_34 = module_1.to_json_schema(var_33)
    var_35 = module_0.Float()
    var_36 = module_1.to_json_schema(var_35)
    var_37 = module_0.Boolean()
    var_38 = module_1.to_json_schema(var_37)
    var_39 = module_0.Boolean()
    var_40 = module_1.to_json_schema(var_39)
    var_41 = module_0.Array()
    var_42 = module_1.to_json_schema(var_41)
    var_43 = module_0.String()
    var_44 = module_0.Array(var_43)
    var_45 = module_1.to_json_schema(var_44)
    var_46 = module_0.Array(min_items=var_8, max_items=var_12)
    var_47 = module_1.to_json_schema(var_46)
    var_48 = module_0.Array(unique_items=var_8)
    var_49 = module_1.to_json_schema(var_48)
    var_50 = False
    var_51 = module_0.Array(additional_items=var_50)
    var_52 = module_1.to_json_schema(var_51)
    var_53 = module_0.String()
    var_54 = module_0.Array(additional_items=var_53)
    var_55 = module_1.to_json_schema(var_54)
    var_56 = module_0.Object()
    var_57 = module_1.to_json_schema(var_56)
    var_58 = 'name'
    var_59 = 'age'
    var_60 = module_0.String()
    var_61 = module_0.Integer()
    var_62 = {var_58: var_60, var_59: var_61}
    var_63 = module_0.Object(properties=var_62)
    var_64 = module_1.to_json_schema(var_63)
    var_65 = module_0.String()
    var_66 = {var_58: var_65}
    var_67 = [var_58]
    var_68 = module_0.Object(properties=var_66, required=var_67)
    var_69 = module_1.to_json_schema(var_68)
    var_70 = False
    var_71 = module_0.Object(additional_properties=var_70)
    var_72 = module_1.to_json_schema(var_71)
    var_73 = module_0.String()
    var_74 = module_0.Object(additional_properties=var_73)
    var_75 = module_1.to_json_schema(var_74)
    var_76 = module_0.String(pattern=var_15)
    var_77 = module_0.Object(property_names=var_76)
    var_78 = module_1.to_json_schema(var_77)
    var_79 = module_0.Object(min_properties=var_8, max_properties=var_11)
    var_80 = module_1.to_json_schema(var_79)
    var_81 = 'red'
    var_82 = (var_81, var_81)
    var_83 = 'green'
    var_84 = (var_83, var_83)
    var_85 = 'blue'
    var_86 = (var_85, var_85)
    var_87 = [var_82, var_84, var_86]
    var_88 = module_0.Choice(choices=var_87)
    var_89 = module_1.to_json_schema(var_88)
    var_90 = 'constant_value'
    var_91 = module_0.Const(var_90)
    var_92 = module_1.to_json_schema(var_91)
    var_93 = module_0.String()
    var_94 = module_0.Integer()
    var_95 = [var_93, var_94]
    var_96 = module_0.Union(var_95)
    var_97 = module_1.to_json_schema(var_96)
    var_98 = 'anyOf'
    var_99 = var_97[var_98]
    var_100 = len(var_99)
    assert var_100 == 2
    var_101 = module_0.String()
    var_102 = module_0.Integer()
    var_103 = [var_101, var_102]
    var_104 = module_2.OneOf(var_103)
    var_105 = module_1.to_json_schema(var_104)
    var_106 = 'oneOf'
    var_107 = var_105[var_106]
    var_108 = len(var_107)
    assert var_108 == 2
    var_109 = module_0.String()
    var_110 = module_0.String(max_length=var_12)
    var_111 = [var_109, var_110]
    var_112 = module_2.AllOf(var_111)
    var_113 = module_1.to_json_schema(var_112)
    var_114 = 'allOf'
    var_115 = var_113[var_114]
    var_116 = len(var_115)
    assert var_116 == 2
    var_117 = module_0.String()
    var_118 = module_2.Not(var_117)
    var_119 = module_1.to_json_schema(var_118)
    var_120 = module_0.String()
    var_121 = module_0.Integer()
    var_122 = module_0.Boolean()
    var_123 = module_2.IfThenElse(var_120, var_121, var_122)
    var_124 = module_1.to_json_schema(var_123)
    var_125 = module_0.String()
    var_126 = module_0.Integer()
    var_127 = module_2.IfThenElse(var_125, var_126)
    var_128 = module_1.to_json_schema(var_127)
    var_129 = 'StringDef'
    var_130 = 'IntegerDef'
    var_131 = module_0.String()
    var_132 = module_0.Integer()
    var_133 = {var_129: var_131, var_130: var_132}
    var_134 = 'User'
    var_135 = module_0.String()
    var_136 = {var_58: var_135}
    var_137 = module_0.Object(properties=var_136)
    var_138 = {var_134: var_137}
    var_139 = module_0.String()
    var_140 = module_0.Integer()
    var_141 = {var_58: var_139, var_59: var_140}
    var_142 = module_3.Schema(var_141)
    var_143 = module_1.to_json_schema(var_142)



# Parsed testcases at query #11
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
    var_8 = {var_6: var_7}
    var_9 = 'number'
    var_10 = {var_3: var_9}
    var_11 = {var_0: var_5, var_1: var_8, var_2: var_10}
    var_12 = 'integer'
    var_13 = {var_3: var_12}
    var_14 = 'minimum'
    var_15 = 0
    var_16 = {var_14: var_15}
    var_17 = {var_0: var_13, var_1: var_16}
    var_18 = 'boolean'
    var_19 = {var_3: var_18}
    var_20 = {var_3: var_4}
    var_21 = {var_0: var_19, var_2: var_20}
    var_22 = 'array'
    var_23 = {var_3: var_22}
    var_24 = {var_0: var_23}
    var_25 = 'default'
    var_26 = {var_3: var_4}
    var_27 = 1
    var_28 = {var_6: var_27}
    var_29 = 'null'
    var_30 = {var_3: var_29}
    var_31 = 'test_default'
    var_32 = {var_0: var_26, var_1: var_28, var_2: var_30, var_25: var_31}
    var_33 = 'properties'
    var_34 = 'enum'
    var_35 = 'A'
    var_36 = 'B'
    var_37 = [var_35, var_36]
    var_38 = {var_34: var_37}
    var_39 = {var_3: var_38}
    var_40 = {var_33: var_39}
    var_41 = 'value'
    var_42 = {var_3: var_4}
    var_43 = {var_41: var_42}
    var_44 = {var_33: var_43}
    var_45 = {var_3: var_9}
    var_46 = {var_41: var_45}
    var_47 = {var_33: var_46}
    var_48 = {var_0: var_40, var_1: var_44, var_2: var_47}
    var_49 = 'object'
    var_50 = {var_3: var_49}
    var_51 = 'minProperties'
    var_52 = {var_51: var_27}
    var_53 = {var_0: var_50, var_1: var_52}



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'Test type_from_json_schema function with various JSON schema types.'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'integer'
    var_5 = {var_1: var_4}
    var_6 = 'number'
    var_7 = {var_1: var_6}
    var_8 = 'boolean'
    var_9 = {var_1: var_8}
    var_10 = 'array'
    var_11 = {var_1: var_10}
    var_12 = 'object'
    var_13 = {var_1: var_12}
    var_14 = [var_2, var_4]
    var_15 = {var_1: var_14}
    var_16 = 'null'
    var_17 = {var_1: var_16}
    var_18 = [var_2, var_16]
    var_19 = {var_1: var_18}
    var_20 = [var_2, var_4, var_16]
    var_21 = {var_1: var_20}
    var_22 = 'minLength'
    var_23 = 'maxLength'
    var_24 = 1
    var_25 = 10
    var_26 = {var_1: var_2, var_22: var_24, var_23: var_25}
    var_27 = 'minimum'
    var_28 = 'maximum'
    var_29 = 0
    var_30 = 100
    var_31 = {var_1: var_4, var_27: var_29, var_28: var_30}
    var_32 = 'pattern'
    var_33 = '^[a-z]+$'
    var_34 = {var_1: var_2, var_32: var_33}
    var_35 = 'items'
    var_36 = {var_1: var_2}
    var_37 = {var_1: var_10, var_35: var_36}
    var_38 = 'properties'
    var_39 = 'name'
    var_40 = {var_1: var_2}
    var_41 = {var_39: var_40}
    var_42 = {var_1: var_12, var_38: var_41}
    var_43 = {}
    var_44 = []
    var_45 = {var_1: var_44}



# Parsed testcases at query #13
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'Test from_json_schema_type function with various type strings and data.'
    var_1 = module_0.Definitions()
    var_2 = 'minimum'
    var_3 = 'maximum'
    var_4 = 'exclusiveMinimum'
    var_5 = 'exclusiveMaximum'
    var_6 = 'multipleOf'
    var_7 = 'default'
    var_8 = 0
    var_9 = 100
    var_10 = 10
    var_11 = 90
    var_12 = 5
    var_13 = 50
    var_14 = {var_2: var_8, var_3: var_9, var_4: var_10, var_5: var_11, var_6: var_12, var_7: var_13}
    var_15 = 'number'
    var_16 = False
    var_17 = module_1.from_json_schema_type(var_14, var_15, var_16, var_1)
    var_18 = True
    var_19 = module_1.from_json_schema_type(var_14, var_15, var_18, var_1)
    var_20 = {var_2: var_18, var_3: var_10, var_7: var_12}
    var_21 = 'integer'
    var_22 = False
    var_23 = module_1.from_json_schema_type(var_20, var_21, var_22, var_1)
    var_24 = 'minLength'
    var_25 = 'maxLength'
    var_26 = 'pattern'
    var_27 = 'format'
    var_28 = 2
    var_29 = '^[a-z]+$'
    var_30 = 'email'
    var_31 = 'test'
    var_32 = {var_24: var_28, var_25: var_13, var_26: var_29, var_27: var_30, var_7: var_31}
    var_33 = 'string'
    var_34 = False
    var_35 = module_1.from_json_schema_type(var_32, var_33, var_34, var_1)
    var_36 = {var_24: var_34}
    var_37 = False
    var_38 = module_1.from_json_schema_type(var_36, var_33, var_37, var_1)
    var_39 = {var_24: var_18}
    var_40 = False
    var_41 = module_1.from_json_schema_type(var_39, var_33, var_40, var_1)
    var_42 = {var_7: var_18}
    var_43 = 'boolean'
    var_44 = False
    var_45 = module_1.from_json_schema_type(var_42, var_43, var_44, var_1)
    var_46 = module_1.from_json_schema_type(var_42, var_43, var_18, var_1)
    var_47 = {}
    var_48 = 'array'
    var_49 = False
    var_50 = module_1.from_json_schema_type(var_47, var_48, var_49, var_1)
    var_51 = 'items'
    var_52 = 'minItems'
    var_53 = 'maxItems'
    var_54 = 'uniqueItems'
    var_55 = 'type'
    var_56 = {var_55: var_33}
    var_57 = 'a'
    var_58 = 'b'
    var_59 = [var_57, var_58]
    var_60 = {var_51: var_56, var_52: var_18, var_53: var_10, var_54: var_18, var_7: var_59}
    var_61 = False
    var_62 = module_1.from_json_schema_type(var_60, var_48, var_61, var_1)
    var_63 = {var_55: var_33}
    var_64 = {var_55: var_21}
    var_65 = [var_63, var_64]
    var_66 = {var_51: var_65}
    var_67 = False
    var_68 = module_1.from_json_schema_type(var_66, var_48, var_67, var_1)
    var_69 = var_68.items
    var_70 = var_68.items
    var_71 = len(var_70)
    assert var_71 == 2
    var_72 = 'additionalItems'
    var_73 = False
    var_74 = {var_72: var_73}
    var_75 = False
    var_76 = module_1.from_json_schema_type(var_74, var_48, var_75, var_1)
    var_77 = {var_55: var_15}
    var_78 = {var_72: var_77}
    var_79 = False
    var_80 = module_1.from_json_schema_type(var_78, var_48, var_79, var_1)
    var_81 = var_80.additional_items
    var_82 = {}
    var_83 = 'object'
    var_84 = False
    var_85 = module_1.from_json_schema_type(var_82, var_83, var_84, var_1)
    var_86 = 'properties'
    var_87 = 'required'
    var_88 = 'minProperties'
    var_89 = 'maxProperties'
    var_90 = 'name'
    var_91 = 'age'
    var_92 = {var_55: var_33}
    var_93 = {var_55: var_21}
    var_94 = {var_90: var_92, var_91: var_93}
    var_95 = [var_90]
    var_96 = {var_86: var_94, var_87: var_95, var_88: var_18, var_89: var_12}
    var_97 = False
    var_98 = module_1.from_json_schema_type(var_96, var_83, var_97, var_1)
    var_99 = 'patternProperties'
    var_100 = '^S_'
    var_101 = '^I_'
    var_102 = {var_55: var_33}
    var_103 = {var_55: var_21}
    var_104 = {var_100: var_102, var_101: var_103}
    var_105 = {var_99: var_104}
    var_106 = False
    var_107 = module_1.from_json_schema_type(var_105, var_83, var_106, var_1)
    var_108 = 'additionalProperties'
    var_109 = False
    var_110 = {var_108: var_109}
    var_111 = False
    var_112 = module_1.from_json_schema_type(var_110, var_83, var_111, var_1)
    var_113 = {var_55: var_33}
    var_114 = {var_108: var_113}
    var_115 = False
    var_116 = module_1.from_json_schema_type(var_114, var_83, var_115, var_1)
    var_117 = var_116.additional_properties
    var_118 = 'propertyNames'
    var_119 = {var_26: var_29}
    var_120 = {var_118: var_119}
    var_121 = False
    var_122 = module_1.from_json_schema_type(var_120, var_83, var_121, var_1)
    var_123 = var_122.property_names
    var_124 = {}
    var_125 = module_1.from_json_schema_type(var_124, var_83, var_18, var_1)



# Parsed testcases at query #14
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = '$ref'
    var_1 = '#/components/schemas/User'
    var_2 = {var_0: var_1}
    var_3 = module_0.Definitions()
    var_4 = module_1.ref_from_json_schema(var_2, var_3)
    var_5 = '#/definitions/Product'
    var_6 = {var_0: var_5}
    var_7 = module_0.Definitions()
    var_8 = module_1.ref_from_json_schema(var_6, var_7)
    var_9 = '#/components/schemas/nested/Item'
    var_10 = {var_0: var_9}
    var_11 = module_0.Definitions()
    var_12 = module_1.ref_from_json_schema(var_10, var_11)
    var_13 = 'components/schemas/User'
    var_14 = {var_0: var_13}
    var_15 = module_0.Definitions()
    var_16 = module_1.ref_from_json_schema(var_14, var_15)
    var_17 = module_0.Definitions()
    var_18 = '#/definitions/Test'
    var_19 = {var_16: var_18}
    var_20 = module_1.ref_from_json_schema(var_19, var_17)



# Parsed testcases at query #15
#--------------------------


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
    var_2 = '#/definitions/Address'
    var_3 = {var_1: var_2}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = '#/components/schemas/Test'
    var_3 = {var_1: var_2}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = 'schemas/User'
    var_3 = {var_1: var_2}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = '#/components/schemas/Empty'
    var_3 = {var_1: var_2}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)
    var_5 = len(var_0)
    assert var_5 == 0

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = '#/components/schemas/models/v1/User'
    var_3 = {var_1: var_2}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'if'
    var_1 = 'then'
    var_2 = 'else'
    var_3 = 'default'
    var_4 = 'type'
    var_5 = 'string'
    var_6 = {var_4: var_5}
    var_7 = 'minLength'
    var_8 = 5
    var_9 = {var_7: var_8}
    var_10 = 'integer'
    var_11 = {var_4: var_10}
    var_12 = 'test'
    var_13 = {var_0: var_6, var_1: var_9, var_2: var_11, var_3: var_12}
    var_14 = {var_4: var_5}
    var_15 = 3
    var_16 = {var_7: var_15}
    var_17 = {var_0: var_14, var_1: var_16}
    var_18 = 'number'
    var_19 = {var_4: var_18}
    var_20 = 'boolean'
    var_21 = {var_4: var_20}
    var_22 = {var_0: var_19, var_2: var_21}
    var_23 = 'array'
    var_24 = {var_4: var_23}
    var_25 = {var_0: var_24}
    var_26 = 'properties'
    var_27 = 'const'
    var_28 = 'object'
    var_29 = {var_27: var_28}
    var_30 = {var_4: var_29}
    var_31 = {var_26: var_30}
    var_32 = 'required'
    var_33 = 'name'
    var_34 = 'age'
    var_35 = [var_33, var_34]
    var_36 = {var_32: var_35}
    var_37 = 'null'
    var_38 = {var_4: var_37}
    var_39 = None
    var_40 = {var_0: var_31, var_1: var_36, var_2: var_38, var_3: var_39}
    var_41 = {var_4: var_5}
    var_42 = 'pattern'
    var_43 = '^[a-z]+$'
    var_44 = {var_42: var_43}
    var_45 = {var_0: var_41, var_1: var_44}



# Parsed testcases at query #17
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
    var_11 = 'test_default'
    var_12 = {var_0: var_6, var_1: var_8, var_2: var_10, var_3: var_11}
    var_13 = {var_4: var_5}
    var_14 = {var_4: var_7}
    var_15 = {var_0: var_13, var_1: var_14}
    var_16 = {var_4: var_5}
    var_17 = {var_4: var_9}
    var_18 = {var_0: var_16, var_2: var_17}
    var_19 = {var_4: var_5}
    var_20 = {var_0: var_19}
    var_21 = 'properties'
    var_22 = 'object'
    var_23 = 'name'
    var_24 = {var_4: var_5}
    var_25 = {var_23: var_24}
    var_26 = {var_4: var_22, var_21: var_25}
    var_27 = 'items'
    var_28 = 'array'
    var_29 = {var_4: var_7}
    var_30 = {var_4: var_28, var_27: var_29}
    var_31 = 'number'
    var_32 = {var_4: var_31}
    var_33 = {var_0: var_26, var_1: var_30, var_2: var_32}
    var_34 = {var_4: var_5}
    var_35 = {var_4: var_7}
    var_36 = {var_4: var_9}
    var_37 = {var_0: var_34, var_1: var_35, var_2: var_36}



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
    var_4 = False
    var_5 = True
    var_6 = module_0.String(allow_blank=var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = module_0.String()
    var_9 = module_1.to_json_schema(var_8)
    var_10 = 5
    var_11 = module_0.String(min_length=var_10)
    var_12 = module_1.to_json_schema(var_11)
    var_13 = 10
    var_14 = module_0.String(max_length=var_13)
    var_15 = module_1.to_json_schema(var_14)
    var_16 = module_0.Integer()
    var_17 = module_1.to_json_schema(var_16)
    var_18 = module_0.Float()
    var_19 = module_1.to_json_schema(var_18)
    var_20 = 100
    var_21 = module_0.Integer(minimum=var_4, maximum=var_20)
    var_22 = module_1.to_json_schema(var_21)
    var_23 = module_0.Boolean()
    var_24 = module_1.to_json_schema(var_23)
    var_25 = module_0.Array()
    var_26 = module_1.to_json_schema(var_25)
    var_27 = module_0.String()
    var_28 = module_0.Array(var_27)
    var_29 = module_1.to_json_schema(var_28)
    var_30 = module_0.Array(min_items=var_5, max_items=var_10)
    var_31 = module_1.to_json_schema(var_30)
    var_32 = module_0.Object()
    var_33 = module_1.to_json_schema(var_32)
    var_34 = 'name'
    var_35 = 'age'
    var_36 = module_0.String()
    var_37 = module_0.Integer()
    var_38 = {var_34: var_36, var_35: var_37}
    var_39 = module_0.Object(properties=var_38)
    var_40 = module_1.to_json_schema(var_39)
    var_41 = [var_34]
    var_42 = module_0.Object(required=var_41)
    var_43 = module_1.to_json_schema(var_42)
    var_44 = 'a'
    var_45 = (var_44, var_5)
    var_46 = 'b'
    var_47 = 2
    var_48 = (var_46, var_47)
    var_49 = [var_45, var_48]
    var_50 = module_0.Choice(choices=var_49)
    var_51 = module_1.to_json_schema(var_50)
    var_52 = 'constant_value'
    var_53 = module_0.Const(var_52)
    var_54 = module_1.to_json_schema(var_53)
    var_55 = module_0.String()
    var_56 = module_0.Integer()
    var_57 = [var_55, var_56]
    var_58 = module_0.Union(var_57)
    var_59 = module_1.to_json_schema(var_58)
    var_60 = 'anyOf'
    var_61 = var_59[var_60]
    var_62 = len(var_61)
    assert var_62 == 2
    var_63 = module_0.String()
    var_64 = module_0.Integer()
    var_65 = [var_63, var_64]
    var_66 = module_2.OneOf(var_65)
    var_67 = module_1.to_json_schema(var_66)
    var_68 = 'oneOf'
    var_69 = var_67[var_68]
    var_70 = len(var_69)
    assert var_70 == 2
    var_71 = module_0.String()
    var_72 = module_0.String()
    var_73 = [var_71, var_72]
    var_74 = module_2.AllOf(var_73)
    var_75 = module_1.to_json_schema(var_74)
    var_76 = 'allOf'
    var_77 = var_75[var_76]
    var_78 = len(var_77)
    assert var_78 == 2
    var_79 = module_0.String()
    var_80 = module_2.Not(var_79)
    var_81 = module_1.to_json_schema(var_80)
    var_82 = module_0.String()
    var_83 = module_0.Integer()
    var_84 = module_0.Boolean()
    var_85 = module_2.IfThenElse(var_82, var_83, var_84)
    var_86 = module_1.to_json_schema(var_85)
    var_87 = module_0.String()
    var_88 = module_0.Integer()
    var_89 = module_2.IfThenElse(var_87, var_88)
    var_90 = module_1.to_json_schema(var_89)
    var_91 = 'string_def'
    var_92 = module_0.String()
    var_93 = {var_91: var_92}
    var_94 = 'nested'
    var_95 = 'field'
    var_96 = module_0.String()
    var_97 = {var_95: var_96}
    var_98 = module_0.Object(properties=var_97)
    var_99 = {var_94: var_98}
    var_100 = module_0.Object(properties=var_99)
    var_101 = module_1.to_json_schema(var_100)
    var_102 = 'default_value'
    var_103 = module_0.String()
    var_104 = module_1.to_json_schema(var_103)
    var_105 = 'default'
    var_106 = '^[a-z]+$'
    var_107 = module_0.String(pattern=var_106)
    var_108 = module_1.to_json_schema(var_107)
    var_109 = 'email'
    var_110 = module_0.String(format=var_109)
    var_111 = module_1.to_json_schema(var_110)
    var_112 = module_0.Array(unique_items=var_5)
    var_113 = module_1.to_json_schema(var_112)
    var_114 = module_0.Object(additional_properties=var_4)
    var_115 = module_1.to_json_schema(var_114)
    var_116 = module_0.String()
    var_117 = module_0.Object(additional_properties=var_116)
    var_118 = module_1.to_json_schema(var_117)
    var_119 = 'additionalProperties'
    var_120 = var_118[var_119]



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = 'null'
    var_4 = [var_1, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'integer'
    var_7 = [var_1, var_6]
    var_8 = {var_0: var_7}
    var_9 = [var_1, var_6, var_3]
    var_10 = {var_0: var_9}
    var_11 = {var_0: var_3}
    var_12 = {}
    var_13 = 'minLength'
    var_14 = 'maxLength'
    var_15 = 1
    var_16 = 10
    var_17 = {var_0: var_1, var_13: var_15, var_14: var_16}
    var_18 = 'minimum'
    var_19 = 'maximum'
    var_20 = 0
    var_21 = 100
    var_22 = {var_0: var_6, var_18: var_20, var_19: var_21}
    var_23 = 'exclusiveMaximum'
    var_24 = 'number'
    var_25 = 0.5
    var_26 = 10.5
    var_27 = {var_0: var_24, var_18: var_25, var_23: var_26}
    var_28 = 'items'
    var_29 = 'array'
    var_30 = {var_0: var_1}
    var_31 = {var_0: var_29, var_28: var_30}
    var_32 = 'properties'
    var_33 = 'required'
    var_34 = 'object'
    var_35 = 'name'
    var_36 = {var_0: var_1}
    var_37 = {var_35: var_36}
    var_38 = [var_35]
    var_39 = {var_0: var_34, var_32: var_37, var_33: var_38}
    var_40 = 'boolean'
    var_41 = {var_0: var_40}
    var_42 = [var_1, var_24, var_3]
    var_43 = {var_0: var_42}



# Parsed testcases at query #20
#--------------------------


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
    var_11 = {var_3: var_4}
    var_12 = {var_3: var_6}
    var_13 = {var_0: var_11, var_1: var_12}
    var_14 = {var_3: var_4}
    var_15 = {var_3: var_8}
    var_16 = {var_0: var_14, var_2: var_15}
    var_17 = {var_3: var_4}
    var_18 = {var_0: var_17}
    var_19 = 'default'
    var_20 = {var_3: var_4}
    var_21 = {var_3: var_6}
    var_22 = 42
    var_23 = {var_0: var_20, var_1: var_21, var_19: var_22}
    var_24 = 'properties'
    var_25 = 'object'
    var_26 = 'name'
    var_27 = {var_3: var_4}
    var_28 = {var_26: var_27}
    var_29 = {var_3: var_25, var_24: var_28}
    var_30 = 'items'
    var_31 = 'array'
    var_32 = {var_3: var_6}
    var_33 = {var_3: var_31, var_30: var_32}
    var_34 = 'minLength'
    var_35 = 5
    var_36 = {var_3: var_4, var_34: var_35}
    var_37 = {var_0: var_29, var_1: var_33, var_2: var_36}



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'Test if_then_else_from_json_schema function.'
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
    var_12 = {var_4: var_5}
    var_13 = {var_4: var_7}
    var_14 = {var_1: var_12, var_2: var_13}
    var_15 = {var_4: var_5}
    var_16 = {var_4: var_9}
    var_17 = {var_1: var_15, var_3: var_16}
    var_18 = {var_4: var_5}
    var_19 = {var_1: var_18}
    var_20 = 'default'
    var_21 = {var_4: var_5}
    var_22 = {var_4: var_7}
    var_23 = 'default_value'
    var_24 = {var_1: var_21, var_2: var_22, var_20: var_23}
    var_25 = 'properties'
    var_26 = 'object'
    var_27 = 'name'
    var_28 = {var_4: var_5}
    var_29 = {var_27: var_28}
    var_30 = {var_4: var_26, var_25: var_29}
    var_31 = 'items'
    var_32 = 'array'
    var_33 = 'number'
    var_34 = {var_4: var_33}
    var_35 = {var_4: var_32, var_31: var_34}
    var_36 = 'enum'
    var_37 = 1
    var_38 = 2
    var_39 = 3
    var_40 = [var_37, var_38, var_39]
    var_41 = {var_36: var_40}
    var_42 = {var_1: var_30, var_2: var_35, var_3: var_41}
    var_43 = {var_4: var_5}
    var_44 = {var_4: var_7}
    var_45 = {var_1: var_43, var_2: var_44}



# Parsed testcases at query #22
#--------------------------


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
    var_9 = 'number'
    var_10 = {var_3: var_9}
    var_11 = {var_0: var_5, var_1: var_8, var_2: var_10}
    var_12 = 'object'
    var_13 = {var_3: var_12}
    var_14 = 'minProperties'
    var_15 = {var_14: var_7}
    var_16 = {var_0: var_13, var_1: var_15}
    var_17 = 'array'
    var_18 = {var_3: var_17}
    var_19 = 'minItems'
    var_20 = 0
    var_21 = {var_19: var_20}
    var_22 = {var_0: var_18, var_2: var_21}
    var_23 = 'boolean'
    var_24 = {var_3: var_23}
    var_25 = {var_0: var_24}
    var_26 = 'default'
    var_27 = {var_3: var_4}
    var_28 = {var_3: var_4}
    var_29 = 'test_default'
    var_30 = {var_0: var_27, var_1: var_28, var_26: var_29}
    var_31 = 'properties'
    var_32 = 'name'
    var_33 = {var_3: var_4}
    var_34 = {var_32: var_33}
    var_35 = {var_31: var_34}
    var_36 = 'required'
    var_37 = [var_32]
    var_38 = {var_36: var_37}
    var_39 = 'additionalProperties'
    var_40 = False
    var_41 = {var_39: var_40}
    var_42 = {var_0: var_35, var_1: var_38, var_2: var_41}



# Parsed testcases at query #23
#--------------------------


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
    var_9 = 'number'
    var_10 = {var_3: var_9}
    var_11 = {var_0: var_5, var_1: var_8, var_2: var_10}
    var_12 = {var_3: var_4}
    var_13 = {var_6: var_7}
    var_14 = {var_0: var_12, var_1: var_13}
    var_15 = {var_3: var_4}
    var_16 = {var_3: var_9}
    var_17 = {var_0: var_15, var_2: var_16}
    var_18 = {var_3: var_4}
    var_19 = {var_0: var_18}
    var_20 = 'default'
    var_21 = {var_3: var_4}
    var_22 = {var_6: var_7}
    var_23 = {var_3: var_9}
    var_24 = 'test_default'
    var_25 = {var_0: var_21, var_1: var_22, var_2: var_23, var_20: var_24}
    var_26 = 'properties'
    var_27 = 'object'
    var_28 = 'name'
    var_29 = {var_3: var_4}
    var_30 = {var_28: var_29}
    var_31 = {var_3: var_27, var_26: var_30}
    var_32 = 'required'
    var_33 = [var_28]
    var_34 = {var_32: var_33}
    var_35 = 'array'
    var_36 = {var_3: var_35}
    var_37 = {var_0: var_31, var_1: var_34, var_2: var_36}



# Parsed testcases at query #24
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'Test if_then_else_from_json_schema function.'
    var_1 = module_0.Definitions()
    var_2 = 'if'
    var_3 = 'then'
    var_4 = 'else'
    var_5 = 'type'
    var_6 = 'string'
    var_7 = {var_5: var_6}
    var_8 = 'integer'
    var_9 = {var_5: var_8}
    var_10 = 'boolean'
    var_11 = {var_5: var_10}
    var_12 = {var_2: var_7, var_3: var_9, var_4: var_11}
    var_13 = module_1.if_then_else_from_json_schema(var_12, var_1)
    var_14 = {var_5: var_6}
    var_15 = {var_5: var_8}
    var_16 = {var_2: var_14, var_3: var_15}
    var_17 = module_1.if_then_else_from_json_schema(var_16, var_1)
    var_18 = {var_5: var_6}
    var_19 = {var_5: var_10}
    var_20 = {var_2: var_18, var_4: var_19}
    var_21 = module_1.if_then_else_from_json_schema(var_20, var_1)
    var_22 = {var_5: var_6}
    var_23 = {var_2: var_22}
    var_24 = module_1.if_then_else_from_json_schema(var_23, var_1)
    var_25 = 'default'
    var_26 = {var_5: var_6}
    var_27 = {var_5: var_8}
    var_28 = 'test_default'
    var_29 = {var_2: var_26, var_3: var_27, var_25: var_28}
    var_30 = module_1.if_then_else_from_json_schema(var_29, var_1)
    var_31 = {var_5: var_6}
    var_32 = {var_2: var_31}
    var_33 = module_1.if_then_else_from_json_schema(var_32, var_1)
    var_34 = 'properties'
    var_35 = 'object'
    var_36 = 'name'
    var_37 = {var_5: var_6}
    var_38 = {var_36: var_37}
    var_39 = {var_5: var_35, var_34: var_38}
    var_40 = 'items'
    var_41 = 'array'
    var_42 = {var_5: var_8}
    var_43 = {var_5: var_41, var_40: var_42}
    var_44 = 'enum'
    var_45 = 1
    var_46 = 2
    var_47 = 3
    var_48 = [var_45, var_46, var_47]
    var_49 = {var_44: var_48}
    var_50 = {var_2: var_39, var_3: var_43, var_4: var_49}
    var_51 = module_1.if_then_else_from_json_schema(var_50, var_1)



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'Test if_then_else_from_json_schema function'
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'else'
    var_4 = 'type'
    var_5 = 'string'
    var_6 = {var_4: var_5}
    var_7 = 'minLength'
    var_8 = 1
    var_9 = {var_4: var_5, var_7: var_8}
    var_10 = 'number'
    var_11 = {var_4: var_10}
    var_12 = {var_1: var_6, var_2: var_9, var_3: var_11}
    var_13 = {var_4: var_5}
    var_14 = {var_4: var_5, var_7: var_8}
    var_15 = {var_1: var_13, var_2: var_14}
    var_16 = {var_4: var_5}
    var_17 = {var_4: var_10}
    var_18 = {var_1: var_16, var_3: var_17}
    var_19 = {var_4: var_5}
    var_20 = {var_1: var_19}
    var_21 = 'default'
    var_22 = {var_4: var_5}
    var_23 = {var_4: var_5}
    var_24 = {var_4: var_10}
    var_25 = 'test'
    var_26 = {var_1: var_22, var_2: var_23, var_3: var_24, var_21: var_25}
    var_27 = 'properties'
    var_28 = 'object'
    var_29 = 'name'
    var_30 = {var_4: var_5}
    var_31 = {var_29: var_30}
    var_32 = {var_4: var_28, var_27: var_31}
    var_33 = 'required'
    var_34 = [var_29]
    var_35 = {var_4: var_28, var_33: var_34}
    var_36 = 'items'
    var_37 = 'array'
    var_38 = {var_4: var_5}
    var_39 = {var_4: var_37, var_36: var_38}
    var_40 = {var_1: var_32, var_2: var_35, var_3: var_39}



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 'type'
    var_1 = 'minLength'
    var_2 = 'string'
    var_3 = 1
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'integer'
    var_6 = [var_2, var_5]
    var_7 = {var_0: var_6}
    var_8 = 'null'
    var_9 = [var_2, var_8]
    var_10 = {var_0: var_9}
    var_11 = [var_5, var_8]
    var_12 = {var_0: var_11}
    var_13 = {var_0: var_8}
    var_14 = {var_1: var_3}
    var_15 = 'minimum'
    var_16 = 'maximum'
    var_17 = 'number'
    var_18 = 0
    var_19 = 100
    var_20 = {var_0: var_17, var_15: var_18, var_16: var_19}
    var_21 = 'boolean'
    var_22 = {var_0: var_21}
    var_23 = 'items'
    var_24 = 'array'
    var_25 = {var_0: var_2}
    var_26 = {var_0: var_24, var_23: var_25}
    var_27 = 'properties'
    var_28 = 'object'
    var_29 = 'name'
    var_30 = {var_0: var_2}
    var_31 = {var_29: var_30}
    var_32 = {var_0: var_28, var_27: var_31}
    var_33 = [var_2, var_17, var_8]
    var_34 = {var_0: var_33}



# Parsed testcases at query #27
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
    var_4 = False
    var_5 = module_0.String()
    var_6 = module_1.to_json_schema(var_5)
    var_7 = True
    var_8 = module_0.String()
    var_9 = module_1.to_json_schema(var_8)
    var_10 = 5
    var_11 = 10
    var_12 = '^[a-z]+$'
    var_13 = module_0.String(max_length=var_11, min_length=var_10, pattern=var_12)
    var_14 = module_1.to_json_schema(var_13)
    var_15 = module_0.String(allow_blank=var_4)
    var_16 = module_1.to_json_schema(var_15)
    var_17 = module_0.Integer()
    var_18 = module_1.to_json_schema(var_17)
    var_19 = 100
    var_20 = module_0.Integer(minimum=var_4, maximum=var_19, multiple_of=var_10)
    var_21 = module_1.to_json_schema(var_20)
    var_22 = module_0.Float()
    var_23 = module_1.to_json_schema(var_22)
    var_24 = module_0.Float(exclusive_minimum=var_4, exclusive_maximum=var_7)
    var_25 = module_1.to_json_schema(var_24)
    var_26 = module_0.Boolean()
    var_27 = module_1.to_json_schema(var_26)
    var_28 = module_0.Boolean()
    var_29 = module_1.to_json_schema(var_28)
    var_30 = module_0.String()
    var_31 = module_0.Array(var_30)
    var_32 = module_1.to_json_schema(var_31)
    var_33 = module_0.Array(min_items=var_7, max_items=var_11, unique_items=var_7)
    var_34 = module_1.to_json_schema(var_33)
    var_35 = module_0.String()
    var_36 = module_0.Integer()
    var_37 = [var_35, var_36]
    var_38 = module_0.Array(var_37)
    var_39 = module_1.to_json_schema(var_38)
    var_40 = 'items'
    var_41 = var_39[var_40]
    var_42 = var_39[var_40]
    var_43 = len(var_42)
    assert var_43 == 2
    var_44 = module_0.Array(additional_items=var_4)
    var_45 = module_1.to_json_schema(var_44)
    var_46 = module_0.String()
    var_47 = module_0.Array(additional_items=var_46)
    var_48 = module_1.to_json_schema(var_47)
    var_49 = 'additionalItems'
    var_50 = var_48[var_49]
    var_51 = 'name'
    var_52 = 'age'
    var_53 = module_0.String()
    var_54 = module_0.Integer()
    var_55 = {var_51: var_53, var_52: var_54}
    var_56 = module_0.Object(properties=var_55)
    var_57 = module_1.to_json_schema(var_56)
    var_58 = [var_51]
    var_59 = module_0.Object(min_properties=var_7, max_properties=var_10, required=var_58)
    var_60 = module_1.to_json_schema(var_59)
    var_61 = '^S_'
    var_62 = module_0.String()
    var_63 = {var_61: var_62}
    var_64 = module_0.Object(pattern_properties=var_63)
    var_65 = module_1.to_json_schema(var_64)
    var_66 = module_0.Object(additional_properties=var_4)
    var_67 = module_1.to_json_schema(var_66)
    var_68 = module_0.String()
    var_69 = module_0.Object(additional_properties=var_68)
    var_70 = module_1.to_json_schema(var_69)
    var_71 = 'additionalProperties'
    var_72 = var_70[var_71]
    var_73 = module_0.String(pattern=var_12)
    var_74 = module_0.Object(property_names=var_73)
    var_75 = module_1.to_json_schema(var_74)
    var_76 = 'a'
    var_77 = (var_76, var_7)
    var_78 = 'b'
    var_79 = 2
    var_80 = (var_78, var_79)
    var_81 = [var_77, var_80]
    var_82 = module_0.Choice(choices=var_81)
    var_83 = module_1.to_json_schema(var_82)
    var_84 = 42
    var_85 = module_0.Const(var_84)
    var_86 = module_1.to_json_schema(var_85)
    var_87 = module_0.String()
    var_88 = module_0.Integer()
    var_89 = [var_87, var_88]
    var_90 = module_0.Union(var_89)
    var_91 = module_1.to_json_schema(var_90)
    var_92 = 'anyOf'
    var_93 = var_91[var_92]
    var_94 = len(var_93)
    assert var_94 == 2
    var_95 = module_0.String()
    var_96 = module_0.Integer()
    var_97 = [var_95, var_96]
    var_98 = module_2.OneOf(var_97)
    var_99 = module_1.to_json_schema(var_98)
    var_100 = 'oneOf'
    var_101 = var_99[var_100]
    var_102 = len(var_101)
    assert var_102 == 2
    var_103 = module_0.String()
    var_104 = 'test'
    var_105 = module_0.Const(var_104)
    var_106 = [var_103, var_105]
    var_107 = module_2.AllOf(var_106)
    var_108 = module_1.to_json_schema(var_107)
    var_109 = 'allOf'
    var_110 = var_108[var_109]
    var_111 = len(var_110)
    assert var_111 == 2
    var_112 = module_0.String()
    var_113 = module_2.Not(var_112)
    var_114 = module_1.to_json_schema(var_113)
    var_115 = module_0.String()
    var_116 = module_0.Integer()
    var_117 = module_0.Boolean()
    var_118 = module_2.IfThenElse(var_115, var_116, var_117)
    var_119 = module_1.to_json_schema(var_118)
    var_120 = module_0.String()
    var_121 = module_0.Integer()
    var_122 = module_2.IfThenElse(var_120, var_121)
    var_123 = module_1.to_json_schema(var_122)
    var_124 = 'StringDef'
    var_125 = 'IntDef'
    var_126 = module_0.String()
    var_127 = module_0.Integer()
    var_128 = {var_124: var_126, var_125: var_127}
    var_129 = module_0.String()
    var_130 = {var_124: var_129}



# Parsed testcases at query #28
#--------------------------


import typesystem.fields as module_0
import typesystem.json_schema as module_1
import typesystem.composites as module_2

def test_case_0():
    var_0 = 'Test to_json_schema conversion for various field types.'
    var_1 = module_0.Any()
    var_2 = module_1.to_json_schema(var_1)
    assert var_2 is True
    var_3 = module_2.NeverMatch()
    var_4 = module_1.to_json_schema(var_3)
    assert var_4 is False
    var_5 = False
    var_6 = 1
    var_7 = 10
    var_8 = '^[a-z]+$'
    var_9 = 'email'
    var_10 = module_0.String(max_length=var_7, min_length=var_6, pattern=var_8, format=var_9)
    var_11 = module_1.to_json_schema(var_10)
    var_12 = True
    var_13 = module_0.String()
    var_14 = module_1.to_json_schema(var_13)
    var_15 = True
    var_16 = module_0.String(allow_blank=var_15)
    var_17 = module_1.to_json_schema(var_16)
    var_18 = 100
    var_19 = 5
    var_20 = module_0.Integer(minimum=var_5, maximum=var_18, multiple_of=var_19)
    var_21 = module_1.to_json_schema(var_20)
    var_22 = True
    var_23 = module_0.Integer()
    var_24 = module_1.to_json_schema(var_23)
    var_25 = module_0.Float(minimum=var_5, maximum=var_22, exclusive_minimum=var_5, exclusive_maximum=var_22)
    var_26 = module_1.to_json_schema(var_25)
    var_27 = module_0.Boolean()
    var_28 = module_1.to_json_schema(var_27)
    var_29 = True
    var_30 = module_0.Boolean()
    var_31 = module_1.to_json_schema(var_30)
    var_32 = module_0.String()
    var_33 = True
    var_34 = module_0.Array(var_32, min_items=var_29, max_items=var_7, unique_items=var_33)
    var_35 = module_1.to_json_schema(var_34)
    var_36 = True
    var_37 = module_0.Integer()
    var_38 = module_0.Array(var_37)
    var_39 = module_1.to_json_schema(var_38)
    var_40 = module_0.String()
    var_41 = module_0.Integer()
    var_42 = [var_40, var_41]
    var_43 = module_0.Array(var_42)
    var_44 = module_1.to_json_schema(var_43)
    var_45 = 'items'
    var_46 = var_44[var_45]
    var_47 = var_44[var_45]
    var_48 = len(var_47)
    assert var_48 == 2
    var_49 = module_0.String()
    var_50 = module_0.Array(var_49, var_5)
    var_51 = module_1.to_json_schema(var_50)
    var_52 = 'name'
    var_53 = 'age'
    var_54 = module_0.String()
    var_55 = module_0.Integer()
    var_56 = {var_52: var_54, var_53: var_55}
    var_57 = [var_52]
    var_58 = module_0.Object(properties=var_56, min_properties=var_36, max_properties=var_7, required=var_57)
    var_59 = module_1.to_json_schema(var_58)
    var_60 = True
    var_61 = 'id'
    var_62 = module_0.Integer()
    var_63 = {var_61: var_62}
    var_64 = module_0.Object(properties=var_63)
    var_65 = module_1.to_json_schema(var_64)
    var_66 = '^S_'
    var_67 = module_0.String()
    var_68 = {var_66: var_67}
    var_69 = module_0.Object(pattern_properties=var_68)
    var_70 = module_1.to_json_schema(var_69)
    var_71 = module_0.Object(additional_properties=var_5)
    var_72 = module_1.to_json_schema(var_71)
    var_73 = module_0.String(pattern=var_8)
    var_74 = module_0.Object(property_names=var_73)
    var_75 = module_1.to_json_schema(var_74)
    var_76 = 'a'
    var_77 = (var_76, var_60)
    var_78 = 'b'
    var_79 = 2
    var_80 = (var_78, var_79)
    var_81 = 'c'
    var_82 = 3
    var_83 = (var_81, var_82)
    var_84 = [var_77, var_80, var_83]
    var_85 = module_0.Choice(choices=var_84)
    var_86 = module_1.to_json_schema(var_85)
    var_87 = 'constant_value'
    var_88 = module_0.Const(var_87)
    var_89 = module_1.to_json_schema(var_88)
    var_90 = module_0.String()
    var_91 = module_0.Integer()
    var_92 = [var_90, var_91]
    var_93 = module_0.Union(var_92)
    var_94 = module_1.to_json_schema(var_93)
    var_95 = 'anyOf'
    var_96 = var_94[var_95]
    var_97 = len(var_96)
    assert var_97 == 2
    var_98 = module_0.String()
    var_99 = module_0.Integer()
    var_100 = [var_98, var_99]
    var_101 = module_2.OneOf(var_100)
    var_102 = module_1.to_json_schema(var_101)
    var_103 = 'oneOf'
    var_104 = var_102[var_103]
    var_105 = len(var_104)
    assert var_105 == 2
    var_106 = module_0.String()
    var_107 = module_0.String(min_length=var_60)
    var_108 = [var_106, var_107]
    var_109 = module_2.AllOf(var_108)
    var_110 = module_1.to_json_schema(var_109)
    var_111 = 'allOf'
    var_112 = var_110[var_111]
    var_113 = len(var_112)
    assert var_113 == 2
    var_114 = module_0.String()
    var_115 = module_2.Not(var_114)
    var_116 = module_1.to_json_schema(var_115)
    var_117 = module_0.String()
    var_118 = module_0.Integer()
    var_119 = module_0.Boolean()
    var_120 = module_2.IfThenElse(var_117, var_118, var_119)
    var_121 = module_1.to_json_schema(var_120)
    var_122 = module_0.String()
    var_123 = module_0.Integer()
    var_124 = module_2.IfThenElse(var_122, var_123)
    var_125 = module_1.to_json_schema(var_124)
    var_126 = 'default_value'
    var_127 = module_0.String()
    var_128 = module_1.to_json_schema(var_127)
    var_129 = 'MyString'
    var_130 = 'MyInt'
    var_131 = module_0.String()
    var_132 = module_0.Integer()
    var_133 = {var_129: var_131, var_130: var_132}



# Parsed testcases at query #29
#--------------------------


def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = 'minimum'
    var_4 = 'integer'
    var_5 = 0
    var_6 = {var_0: var_4, var_3: var_5}
    var_7 = [var_1, var_4]
    var_8 = {var_0: var_7}
    var_9 = 'null'
    var_10 = [var_1, var_9]
    var_11 = {var_0: var_10}
    var_12 = {var_0: var_9}
    var_13 = 'number'
    var_14 = [var_1, var_13, var_9]
    var_15 = {var_0: var_14}
    var_16 = {var_0: var_13}
    var_17 = 'boolean'
    var_18 = {var_0: var_17}
    var_19 = 'items'
    var_20 = 'array'
    var_21 = {var_0: var_1}
    var_22 = {var_0: var_20, var_19: var_21}
    var_23 = 'properties'
    var_24 = 'object'
    var_25 = 'name'
    var_26 = {var_0: var_1}
    var_27 = {var_25: var_26}
    var_28 = {var_0: var_24, var_23: var_27}
    var_29 = {}
    var_30 = []
    var_31 = {var_0: var_30}
    var_32 = 'pattern'
    var_33 = '^[a-z]+$'
    var_34 = {var_0: var_1, var_32: var_33}
    var_35 = [var_1, var_17, var_4]
    var_36 = {var_0: var_35}
    var_37 = {var_0: var_13}



# Parsed testcases at query #30
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
    var_4 = False
    var_5 = 2
    var_6 = 10
    var_7 = '^[a-z]+$'
    var_8 = 'email'
    var_9 = module_0.String(max_length=var_6, min_length=var_5, pattern=var_7, format=var_8)
    var_10 = module_1.to_json_schema(var_9)
    var_11 = True
    var_12 = module_0.String()
    var_13 = module_1.to_json_schema(var_12)
    var_14 = module_0.String(allow_blank=var_4)
    var_15 = module_1.to_json_schema(var_14)
    var_16 = 100
    var_17 = 5
    var_18 = module_0.Integer(minimum=var_4, maximum=var_16, multiple_of=var_17)
    var_19 = module_1.to_json_schema(var_18)
    var_20 = module_0.Float(exclusive_minimum=var_4, exclusive_maximum=var_11)
    var_21 = module_1.to_json_schema(var_20)
    var_22 = module_0.Boolean()
    var_23 = module_1.to_json_schema(var_22)
    var_24 = module_0.Boolean()
    var_25 = module_1.to_json_schema(var_24)
    var_26 = module_0.String()
    var_27 = module_0.Array(var_26, min_items=var_11, max_items=var_6, unique_items=var_11)
    var_28 = module_1.to_json_schema(var_27)
    var_29 = module_0.String()
    var_30 = module_0.Integer()
    var_31 = [var_29, var_30]
    var_32 = module_0.Array(var_31)
    var_33 = module_1.to_json_schema(var_32)
    var_34 = 'items'
    var_35 = var_33[var_34]
    var_36 = var_33[var_34]
    var_37 = len(var_36)
    assert var_37 == 2
    var_38 = module_0.String()
    var_39 = module_0.Array(var_38, var_4)
    var_40 = module_1.to_json_schema(var_39)
    var_41 = 'name'
    var_42 = 'age'
    var_43 = module_0.String()
    var_44 = module_0.Integer()
    var_45 = {var_41: var_43, var_42: var_44}
    var_46 = [var_41]
    var_47 = module_0.Object(properties=var_45, min_properties=var_11, max_properties=var_6, required=var_46)
    var_48 = module_1.to_json_schema(var_47)
    var_49 = '^S_'
    var_50 = module_0.String()
    var_51 = {var_49: var_50}
    var_52 = module_0.Object(pattern_properties=var_51)
    var_53 = module_1.to_json_schema(var_52)
    var_54 = module_0.Object(additional_properties=var_4)
    var_55 = module_1.to_json_schema(var_54)
    var_56 = module_0.String(pattern=var_7)
    var_57 = module_0.Object(property_names=var_56)
    var_58 = module_1.to_json_schema(var_57)
    var_59 = 'a'
    var_60 = 'Option A'
    var_61 = (var_59, var_60)
    var_62 = 'b'
    var_63 = 'Option B'
    var_64 = (var_62, var_63)
    var_65 = [var_61, var_64]
    var_66 = module_0.Choice(choices=var_65)
    var_67 = module_1.to_json_schema(var_66)
    var_68 = 'fixed_value'
    var_69 = module_0.Const(var_68)
    var_70 = module_1.to_json_schema(var_69)
    var_71 = module_0.String()
    var_72 = module_0.Integer()
    var_73 = [var_71, var_72]
    var_74 = module_0.Union(var_73)
    var_75 = module_1.to_json_schema(var_74)
    var_76 = 'anyOf'
    var_77 = var_75[var_76]
    var_78 = len(var_77)
    assert var_78 == 2
    var_79 = module_0.String()
    var_80 = module_0.Integer()
    var_81 = [var_79, var_80]
    var_82 = module_2.OneOf(var_81)
    var_83 = module_1.to_json_schema(var_82)
    var_84 = 'oneOf'
    var_85 = var_83[var_84]
    var_86 = len(var_85)
    assert var_86 == 2
    var_87 = module_0.String()
    var_88 = module_0.Object()
    var_89 = [var_87, var_88]
    var_90 = module_2.AllOf(var_89)
    var_91 = module_1.to_json_schema(var_90)
    var_92 = 'allOf'
    var_93 = var_91[var_92]
    var_94 = len(var_93)
    assert var_94 == 2
    var_95 = module_0.String()
    var_96 = module_2.Not(var_95)
    var_97 = module_1.to_json_schema(var_96)
    var_98 = module_0.String()
    var_99 = module_0.Integer()
    var_100 = module_0.Boolean()
    var_101 = module_2.IfThenElse(var_98, var_99, var_100)
    var_102 = module_1.to_json_schema(var_101)
    var_103 = module_0.String()
    var_104 = module_2.IfThenElse(var_103)
    var_105 = module_1.to_json_schema(var_104)
    var_106 = 'User'
    var_107 = module_0.String()
    var_108 = {var_41: var_107}
    var_109 = module_0.Object(properties=var_108)
    var_110 = {var_106: var_109}
    var_111 = module_0.Object()
    var_112 = {var_106: var_111}
    var_113 = 'test_default'
    var_114 = module_0.String()
    var_115 = module_1.to_json_schema(var_114)
    var_116 = module_1.to_json_schema(var_0)



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = 'null'
    var_4 = [var_1, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'integer'
    var_7 = [var_1, var_6]
    var_8 = {var_0: var_7}
    var_9 = [var_1, var_6, var_3]
    var_10 = {var_0: var_9}
    var_11 = {var_0: var_3}
    var_12 = [var_3]
    var_13 = {var_0: var_12}
    var_14 = {var_0: var_6}
    var_15 = 'number'
    var_16 = {var_0: var_15}
    var_17 = 'boolean'
    var_18 = {var_0: var_17}
    var_19 = 'array'
    var_20 = {var_0: var_19}
    var_21 = 'object'
    var_22 = {var_0: var_21}
    var_23 = 'minLength'
    var_24 = 5
    var_25 = {var_0: var_1, var_23: var_24}
    var_26 = 'minimum'
    var_27 = 10
    var_28 = {var_0: var_6, var_26: var_27}
    var_29 = {}
    var_30 = [var_1, var_15]
    var_31 = 0
    var_32 = {var_0: var_30, var_26: var_31}



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = 'type'
    var_1 = 'minLength'
    var_2 = 'string'
    var_3 = 1
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'integer'
    var_6 = [var_2, var_5]
    var_7 = {var_0: var_6}
    var_8 = 'null'
    var_9 = [var_2, var_8]
    var_10 = {var_0: var_9}
    var_11 = {var_0: var_8}
    var_12 = {}
    var_13 = 'minimum'
    var_14 = 0
    var_15 = {var_0: var_5, var_13: var_14}
    var_16 = 'maximum'
    var_17 = 'number'
    var_18 = 100
    var_19 = {var_0: var_17, var_16: var_18}
    var_20 = 'boolean'
    var_21 = {var_0: var_20}
    var_22 = 'items'
    var_23 = 'array'
    var_24 = {var_0: var_2}
    var_25 = {var_0: var_23, var_22: var_24}
    var_26 = 'properties'
    var_27 = 'object'
    var_28 = 'name'
    var_29 = {var_0: var_2}
    var_30 = {var_28: var_29}
    var_31 = {var_0: var_27, var_26: var_30}
    var_32 = [var_2, var_8]
    var_33 = {var_0: var_32}
    var_34 = [var_2, var_5, var_8]
    var_35 = {var_0: var_34}



# Parsed testcases at query #33
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
    var_14 = True
    var_15 = module_0.String(allow_blank=var_14)
    var_16 = module_1.to_json_schema(var_15)
    var_17 = 100
    var_18 = 5
    var_19 = 95
    var_20 = module_0.Integer(minimum=var_4, maximum=var_17, exclusive_minimum=var_18, exclusive_maximum=var_19, multiple_of=var_18)
    var_21 = module_1.to_json_schema(var_20)
    var_22 = True
    var_23 = module_0.Integer()
    var_24 = module_1.to_json_schema(var_23)
    var_25 = module_0.Float(minimum=var_4, maximum=var_22)
    var_26 = module_1.to_json_schema(var_25)
    var_27 = module_0.Boolean()
    var_28 = module_1.to_json_schema(var_27)
    var_29 = True
    var_30 = module_0.Boolean()
    var_31 = module_1.to_json_schema(var_30)
    var_32 = module_0.String()
    var_33 = True
    var_34 = module_0.Array(var_32, min_items=var_29, max_items=var_18, unique_items=var_33)
    var_35 = module_1.to_json_schema(var_34)
    var_36 = True
    var_37 = module_0.Array()
    var_38 = module_1.to_json_schema(var_37)
    var_39 = module_0.String()
    var_40 = module_0.Integer()
    var_41 = [var_39, var_40]
    var_42 = module_0.Array(var_41)
    var_43 = module_1.to_json_schema(var_42)
    var_44 = 'items'
    var_45 = var_43[var_44]
    var_46 = var_43[var_44]
    var_47 = len(var_46)
    assert var_47 == 2
    var_48 = module_0.Array(additional_items=var_4)
    var_49 = module_1.to_json_schema(var_48)
    var_50 = module_0.String()
    var_51 = module_0.Array(additional_items=var_50)
    var_52 = module_1.to_json_schema(var_51)
    var_53 = 'additionalItems'
    var_54 = var_52[var_53]
    var_55 = 'name'
    var_56 = 'age'
    var_57 = module_0.String()
    var_58 = module_0.Integer()
    var_59 = {var_55: var_57, var_56: var_58}
    var_60 = [var_55]
    var_61 = module_0.Object(properties=var_59, min_properties=var_36, max_properties=var_6, required=var_60)
    var_62 = module_1.to_json_schema(var_61)
    var_63 = True
    var_64 = module_0.Object()
    var_65 = module_1.to_json_schema(var_64)
    var_66 = '^S_'
    var_67 = module_0.String()
    var_68 = {var_66: var_67}
    var_69 = module_0.Object(pattern_properties=var_68)
    var_70 = module_1.to_json_schema(var_69)
    var_71 = module_0.Object(additional_properties=var_4)
    var_72 = module_1.to_json_schema(var_71)
    var_73 = module_0.String()
    var_74 = module_0.Object(additional_properties=var_73)
    var_75 = module_1.to_json_schema(var_74)
    var_76 = 'additionalProperties'
    var_77 = var_75[var_76]
    var_78 = module_0.String(pattern=var_7)
    var_79 = module_0.Object(property_names=var_78)
    var_80 = module_1.to_json_schema(var_79)
    var_81 = 'a'
    var_82 = (var_81, var_63)
    var_83 = 'b'
    var_84 = 2
    var_85 = (var_83, var_84)
    var_86 = [var_82, var_85]
    var_87 = module_0.Choice(choices=var_86)
    var_88 = module_1.to_json_schema(var_87)
    var_89 = 'fixed_value'
    var_90 = module_0.Const(var_89)
    var_91 = module_1.to_json_schema(var_90)
    var_92 = module_0.String()
    var_93 = module_0.Integer()
    var_94 = [var_92, var_93]
    var_95 = module_0.Union(var_94)
    var_96 = module_1.to_json_schema(var_95)
    var_97 = 'anyOf'
    var_98 = var_96[var_97]
    var_99 = len(var_98)
    assert var_99 == 2
    var_100 = module_0.String()
    var_101 = module_0.Integer()
    var_102 = [var_100, var_101]
    var_103 = module_2.OneOf(var_102)
    var_104 = module_1.to_json_schema(var_103)
    var_105 = 'oneOf'
    var_106 = var_104[var_105]
    var_107 = len(var_106)
    assert var_107 == 2
    var_108 = module_0.String()
    var_109 = module_0.String(min_length=var_18)
    var_110 = [var_108, var_109]
    var_111 = module_2.AllOf(var_110)
    var_112 = module_1.to_json_schema(var_111)
    var_113 = 'allOf'
    var_114 = var_112[var_113]
    var_115 = len(var_114)
    assert var_115 == 2
    var_116 = module_0.String()
    var_117 = module_0.Integer()
    var_118 = module_0.Boolean()
    var_119 = module_2.IfThenElse(var_116, var_117, var_118)
    var_120 = module_1.to_json_schema(var_119)
    var_121 = module_0.String()
    var_122 = module_2.IfThenElse(var_121)
    var_123 = module_1.to_json_schema(var_122)
    var_124 = module_0.String()
    var_125 = module_2.Not(var_124)
    var_126 = module_1.to_json_schema(var_125)
    var_127 = 'StringDef'
    var_128 = 'IntDef'
    var_129 = module_0.String()
    var_130 = module_0.Integer()
    var_131 = {var_127: var_129, var_128: var_130}



# Parsed testcases at query #34
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
    var_6 = {var_0: var_5}
    var_7 = module_0.Definitions()
    var_8 = module_1.type_from_json_schema(var_6, var_7)
    var_9 = 'number'
    var_10 = {var_0: var_9}
    var_11 = module_0.Definitions()
    var_12 = module_1.type_from_json_schema(var_10, var_11)
    var_13 = 'boolean'
    var_14 = {var_0: var_13}
    var_15 = module_0.Definitions()
    var_16 = module_1.type_from_json_schema(var_14, var_15)
    var_17 = 'array'
    var_18 = {var_0: var_17}
    var_19 = module_0.Definitions()
    var_20 = module_1.type_from_json_schema(var_18, var_19)
    var_21 = 'object'
    var_22 = {var_0: var_21}
    var_23 = module_0.Definitions()
    var_24 = module_1.type_from_json_schema(var_22, var_23)
    var_25 = [var_1, var_5]
    var_26 = {var_0: var_25}
    var_27 = module_0.Definitions()
    var_28 = module_1.type_from_json_schema(var_26, var_27)
    var_29 = 'null'
    var_30 = [var_1, var_29]
    var_31 = {var_0: var_30}
    var_32 = module_0.Definitions()
    var_33 = module_1.type_from_json_schema(var_31, var_32)
    var_34 = {var_0: var_29}
    var_35 = module_0.Definitions()
    var_36 = module_1.type_from_json_schema(var_34, var_35)
    var_37 = {}
    var_38 = module_0.Definitions()
    var_39 = module_1.type_from_json_schema(var_37, var_38)
    var_40 = 'minLength'
    var_41 = 5
    var_42 = {var_0: var_1, var_40: var_41}
    var_43 = module_0.Definitions()
    var_44 = module_1.type_from_json_schema(var_42, var_43)
    var_45 = 'minimum'
    var_46 = 10
    var_47 = {var_0: var_5, var_45: var_46}
    var_48 = module_0.Definitions()
    var_49 = module_1.type_from_json_schema(var_47, var_48)
    var_50 = 'items'
    var_51 = {var_0: var_1}
    var_52 = {var_0: var_17, var_50: var_51}
    var_53 = module_0.Definitions()
    var_54 = module_1.type_from_json_schema(var_52, var_53)
    var_55 = 'properties'
    var_56 = 'name'
    var_57 = {var_0: var_1}
    var_58 = {var_56: var_57}
    var_59 = {var_0: var_21, var_55: var_58}
    var_60 = module_0.Definitions()
    var_61 = module_1.type_from_json_schema(var_59, var_60)
    var_62 = [var_1, var_5]
    var_63 = 1
    var_64 = {var_0: var_62, var_40: var_63}
    var_65 = module_0.Definitions()
    var_66 = module_1.type_from_json_schema(var_64, var_65)
    var_67 = 'pattern'
    var_68 = '^[a-z]+$'
    var_69 = {var_0: var_1, var_67: var_68}
    var_70 = module_0.Definitions()
    var_71 = module_1.type_from_json_schema(var_69, var_70)
    var_72 = 'multipleOf'
    var_73 = 0.5
    var_74 = {var_0: var_9, var_72: var_73}
    var_75 = module_0.Definitions()
    var_76 = module_1.type_from_json_schema(var_74, var_75)
    var_77 = 'exclusiveMinimum'
    var_78 = 'exclusiveMaximum'
    var_79 = 0
    var_80 = 100
    var_81 = {var_0: var_5, var_77: var_79, var_78: var_80}
    var_82 = module_0.Definitions()
    var_83 = module_1.type_from_json_schema(var_81, var_82)



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = 'type'
    var_1 = 'minLength'
    var_2 = 'string'
    var_3 = 1
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'minimum'
    var_6 = 'integer'
    var_7 = 0
    var_8 = {var_0: var_6, var_5: var_7}
    var_9 = 'boolean'
    var_10 = {var_0: var_9}
    var_11 = 'number'
    var_12 = {var_0: var_11}
    var_13 = 'items'
    var_14 = 'array'
    var_15 = {var_0: var_2}
    var_16 = {var_0: var_14, var_13: var_15}
    var_17 = 'properties'
    var_18 = 'object'
    var_19 = 'name'
    var_20 = {var_0: var_2}
    var_21 = {var_19: var_20}
    var_22 = {var_0: var_18, var_17: var_21}
    var_23 = [var_2, var_6]
    var_24 = {var_0: var_23}
    var_25 = 'null'
    var_26 = [var_2, var_25]
    var_27 = {var_0: var_26}
    var_28 = {var_0: var_25}
    var_29 = [var_25]
    var_30 = {var_0: var_29}
    var_31 = [var_2, var_11, var_25]
    var_32 = {var_0: var_31}
    var_33 = 'pattern'
    var_34 = 'maxLength'
    var_35 = '^[a-z]+$'
    var_36 = 10
    var_37 = {var_0: var_2, var_33: var_35, var_34: var_36}
    var_38 = 'maximum'
    var_39 = 100
    var_40 = {var_0: var_11, var_5: var_7, var_38: var_39}
    var_41 = {}
    var_42 = [var_2, var_6, var_9]
    var_43 = {var_0: var_42}



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
    var_4 = False
    var_5 = True
    var_6 = module_0.String(allow_blank=var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = module_0.String()
    var_9 = module_1.to_json_schema(var_8)
    var_10 = 2
    var_11 = 10
    var_12 = '^[a-z]+$'
    var_13 = module_0.String(max_length=var_11, min_length=var_10, pattern=var_12)
    var_14 = module_1.to_json_schema(var_13)
    var_15 = module_0.Integer()
    var_16 = module_1.to_json_schema(var_15)
    var_17 = 100
    var_18 = 5
    var_19 = module_0.Integer(minimum=var_4, maximum=var_17, multiple_of=var_18)
    var_20 = module_1.to_json_schema(var_19)
    var_21 = module_0.Float()
    var_22 = module_1.to_json_schema(var_21)
    var_23 = module_0.Float(exclusive_minimum=var_4, exclusive_maximum=var_17)
    var_24 = module_1.to_json_schema(var_23)
    var_25 = module_0.Boolean()
    var_26 = module_1.to_json_schema(var_25)
    var_27 = module_0.Boolean()
    var_28 = module_1.to_json_schema(var_27)
    var_29 = module_0.String()
    var_30 = module_0.Array(var_29, min_items=var_5, max_items=var_18, unique_items=var_5)
    var_31 = module_1.to_json_schema(var_30)
    var_32 = module_0.String()
    var_33 = module_0.Array(additional_items=var_32)
    var_34 = module_1.to_json_schema(var_33)
    var_35 = 'name'
    var_36 = 'age'
    var_37 = module_0.String()
    var_38 = module_0.Integer()
    var_39 = {var_35: var_37, var_36: var_38}
    var_40 = [var_35]
    var_41 = module_0.Object(properties=var_39, required=var_40)
    var_42 = module_1.to_json_schema(var_41)
    var_43 = '^S_'
    var_44 = module_0.String()
    var_45 = {var_43: var_44}
    var_46 = module_0.Object(pattern_properties=var_45)
    var_47 = module_1.to_json_schema(var_46)
    var_48 = module_0.Object(additional_properties=var_4)
    var_49 = module_1.to_json_schema(var_48)
    var_50 = module_0.String()
    var_51 = module_0.Object(additional_properties=var_50)
    var_52 = module_1.to_json_schema(var_51)
    var_53 = 'additionalProperties'
    var_54 = var_52[var_53]
    var_55 = module_0.String(pattern=var_12)
    var_56 = module_0.Object(property_names=var_55)
    var_57 = module_1.to_json_schema(var_56)
    var_58 = module_0.Object(min_properties=var_5, max_properties=var_11)
    var_59 = module_1.to_json_schema(var_58)
    var_60 = 'a'
    var_61 = 'Apple'
    var_62 = (var_60, var_61)
    var_63 = 'b'
    var_64 = 'Banana'
    var_65 = (var_63, var_64)
    var_66 = [var_62, var_65]
    var_67 = module_0.Choice(choices=var_66)
    var_68 = module_1.to_json_schema(var_67)
    var_69 = 'fixed_value'
    var_70 = module_0.Const(var_69)
    var_71 = module_1.to_json_schema(var_70)
    var_72 = module_0.String()
    var_73 = module_0.Integer()
    var_74 = [var_72, var_73]
    var_75 = module_0.Union(var_74)
    var_76 = module_1.to_json_schema(var_75)
    var_77 = 'anyOf'
    var_78 = var_76[var_77]
    var_79 = len(var_78)
    assert var_79 == 2
    var_80 = module_0.String()
    var_81 = module_0.Integer()
    var_82 = [var_80, var_81]
    var_83 = module_2.OneOf(var_82)
    var_84 = module_1.to_json_schema(var_83)
    var_85 = 'oneOf'
    var_86 = var_84[var_85]
    var_87 = len(var_86)
    assert var_87 == 2
    var_88 = module_0.String()
    var_89 = 'A'
    var_90 = (var_60, var_89)
    var_91 = [var_90]
    var_92 = module_0.Choice(choices=var_91)
    var_93 = [var_88, var_92]
    var_94 = module_2.AllOf(var_93)
    var_95 = module_1.to_json_schema(var_94)
    var_96 = 'allOf'
    var_97 = var_95[var_96]
    var_98 = len(var_97)
    assert var_98 == 2
    var_99 = module_0.String()
    var_100 = module_2.Not(var_99)
    var_101 = module_1.to_json_schema(var_100)
    var_102 = (var_60, var_89)
    var_103 = [var_102]
    var_104 = module_0.Choice(choices=var_103)
    var_105 = module_0.String()
    var_106 = module_0.Integer()
    var_107 = module_2.IfThenElse(var_104, var_105, var_106)
    var_108 = module_1.to_json_schema(var_107)
    var_109 = (var_60, var_89)
    var_110 = [var_109]
    var_111 = module_0.Choice(choices=var_110)
    var_112 = module_0.String()
    var_113 = module_2.IfThenElse(var_111, var_112)
    var_114 = module_1.to_json_schema(var_113)
    var_115 = 'hello'
    var_116 = module_0.String()
    var_117 = module_1.to_json_schema(var_116)
    var_118 = 'default'
    var_119 = 'User'
    var_120 = module_0.String()
    var_121 = {var_35: var_120}
    var_122 = module_0.Object(properties=var_121)
    var_123 = {var_119: var_122}
    var_124 = module_0.String()
    var_125 = {var_35: var_124}
    var_126 = module_0.Object(properties=var_125)
    var_127 = {var_119: var_126}



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
    var_15 = 5
    var_16 = module_0.Integer(minimum=var_4, maximum=var_14, multiple_of=var_15)
    var_17 = module_1.to_json_schema(var_16)
    var_18 = module_0.Float(minimum=var_4, maximum=var_11, exclusive_minimum=var_4, exclusive_maximum=var_11)
    var_19 = module_1.to_json_schema(var_18)
    var_20 = module_0.Boolean()
    var_21 = module_1.to_json_schema(var_20)
    var_22 = module_0.String()
    var_23 = True
    var_24 = module_0.Array(var_22, min_items=var_11, max_items=var_15, unique_items=var_23)
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
    var_41 = module_0.Object(properties=var_39, min_properties=var_23, max_properties=var_6, required=var_40)
    var_42 = module_1.to_json_schema(var_41)
    var_43 = '^S_'
    var_44 = module_0.String()
    var_45 = {var_43: var_44}
    var_46 = module_0.Object(pattern_properties=var_45)
    var_47 = module_1.to_json_schema(var_46)
    var_48 = module_0.String()
    var_49 = module_0.Object(additional_properties=var_48)
    var_50 = module_1.to_json_schema(var_49)
    var_51 = 'additionalProperties'
    var_52 = var_50[var_51]
    var_53 = 'a'
    var_54 = 'A'
    var_55 = (var_53, var_54)
    var_56 = 'b'
    var_57 = 'B'
    var_58 = (var_56, var_57)
    var_59 = [var_55, var_58]
    var_60 = module_0.Choice(choices=var_59)
    var_61 = module_1.to_json_schema(var_60)
    var_62 = 'fixed_value'
    var_63 = module_0.Const(var_62)
    var_64 = module_1.to_json_schema(var_63)
    var_65 = module_0.String()
    var_66 = module_0.Integer()
    var_67 = [var_65, var_66]
    var_68 = module_0.Union(var_67)
    var_69 = module_1.to_json_schema(var_68)
    var_70 = 'anyOf'
    var_71 = var_69[var_70]
    var_72 = len(var_71)
    assert var_72 == 2
    var_73 = module_0.String()
    var_74 = module_0.Integer()
    var_75 = [var_73, var_74]
    var_76 = module_2.OneOf(var_75)
    var_77 = module_1.to_json_schema(var_76)
    var_78 = 'oneOf'
    var_79 = var_77[var_78]
    var_80 = len(var_79)
    assert var_80 == 2
    var_81 = module_0.String()
    var_82 = (var_53, var_54)
    var_83 = [var_82]
    var_84 = module_0.Choice(choices=var_83)
    var_85 = [var_81, var_84]
    var_86 = module_2.AllOf(var_85)
    var_87 = module_1.to_json_schema(var_86)
    var_88 = 'allOf'
    var_89 = var_87[var_88]
    var_90 = len(var_89)
    assert var_90 == 2
    var_91 = module_0.String()
    var_92 = module_2.Not(var_91)
    var_93 = module_1.to_json_schema(var_92)
    var_94 = (var_53, var_54)
    var_95 = [var_94]
    var_96 = module_0.Choice(choices=var_95)
    var_97 = module_0.String()
    var_98 = module_0.Integer()
    var_99 = module_2.IfThenElse(var_96, var_97, var_98)
    var_100 = module_1.to_json_schema(var_99)
    var_101 = (var_53, var_54)
    var_102 = [var_101]
    var_103 = module_0.Choice(choices=var_102)
    var_104 = module_0.String()
    var_105 = module_2.IfThenElse(var_103, var_104)
    var_106 = module_1.to_json_schema(var_105)
    var_107 = module_3.Definitions()
    var_108 = 'MySchema'
    var_109 = module_3.Reference(var_108, var_107)
    var_110 = module_1.to_json_schema(var_109)
    var_111 = module_3.Definitions()
    var_112 = module_1.to_json_schema(var_111)
    var_113 = 'hello'
    var_114 = module_0.String()
    var_115 = module_1.to_json_schema(var_114)
    var_116 = module_0.String(allow_blank=var_4)
    var_117 = module_1.to_json_schema(var_116)
    var_118 = module_0.Array(additional_items=var_4)
    var_119 = module_1.to_json_schema(var_118)
    var_120 = module_0.String()
    var_121 = module_0.Array(additional_items=var_120)
    var_122 = module_1.to_json_schema(var_121)
    var_123 = 'additionalItems'
    var_124 = var_122[var_123]
    var_125 = module_0.String(pattern=var_7)
    var_126 = module_0.Object(property_names=var_125)
    var_127 = module_1.to_json_schema(var_126)
    var_128 = module_0.String()
    var_129 = {var_35: var_128}
    var_130 = [var_35]
    var_131 = module_3.Schema(var_129)
    var_132 = module_1.to_json_schema(var_131)



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
    var_4 = False
    var_5 = 2
    var_6 = 10
    var_7 = '^[a-z]+$'
    var_8 = 'email'
    var_9 = module_0.String(max_length=var_6, min_length=var_5, pattern=var_7, format=var_8)
    var_10 = module_1.to_json_schema(var_9)
    var_11 = True
    var_12 = module_0.String()
    var_13 = module_1.to_json_schema(var_12)
    var_14 = 100
    var_15 = 101
    var_16 = module_0.Integer(minimum=var_11, maximum=var_14, exclusive_minimum=var_4, exclusive_maximum=var_15)
    var_17 = module_1.to_json_schema(var_16)
    var_18 = 0.5
    var_19 = 99.5
    var_20 = 0.1
    var_21 = module_0.Float(minimum=var_18, maximum=var_19, multiple_of=var_20)
    var_22 = module_1.to_json_schema(var_21)
    var_23 = module_0.Boolean()
    var_24 = module_1.to_json_schema(var_23)
    var_25 = module_0.Boolean()
    var_26 = module_1.to_json_schema(var_25)
    var_27 = module_0.String()
    var_28 = 5
    var_29 = module_0.Array(var_27, min_items=var_11, max_items=var_28, unique_items=var_11)
    var_30 = module_1.to_json_schema(var_29)
    var_31 = module_0.String()
    var_32 = module_0.Integer()
    var_33 = [var_31, var_32]
    var_34 = module_0.Array(var_33)
    var_35 = module_1.to_json_schema(var_34)
    var_36 = 'items'
    var_37 = var_35[var_36]
    var_38 = len(var_37)
    assert var_38 == 2
    var_39 = module_0.String()
    var_40 = module_0.Array(var_39, var_4)
    var_41 = module_1.to_json_schema(var_40)
    var_42 = module_0.String()
    var_43 = module_0.Integer()
    var_44 = module_0.Array(var_42, var_43)
    var_45 = module_1.to_json_schema(var_44)
    var_46 = 'name'
    var_47 = 'age'
    var_48 = module_0.String()
    var_49 = module_0.Integer()
    var_50 = {var_46: var_48, var_47: var_49}
    var_51 = [var_46]
    var_52 = module_0.Object(properties=var_50, min_properties=var_11, max_properties=var_6, required=var_51)
    var_53 = module_1.to_json_schema(var_52)
    var_54 = '^S_'
    var_55 = '^I_'
    var_56 = module_0.String()
    var_57 = module_0.Integer()
    var_58 = {var_54: var_56, var_55: var_57}
    var_59 = module_0.Object(pattern_properties=var_58)
    var_60 = module_1.to_json_schema(var_59)
    var_61 = module_0.Object(additional_properties=var_4)
    var_62 = module_1.to_json_schema(var_61)
    var_63 = module_0.String()
    var_64 = module_0.Object(additional_properties=var_63)
    var_65 = module_1.to_json_schema(var_64)
    var_66 = module_0.String(pattern=var_7)
    var_67 = module_0.Object(property_names=var_66)
    var_68 = module_1.to_json_schema(var_67)
    var_69 = module_0.String()
    var_70 = module_0.Integer()
    var_71 = {var_46: var_69, var_47: var_70}
    var_72 = [var_46]
    var_73 = module_3.Schema(var_71)
    var_74 = module_1.to_json_schema(var_73)
    var_75 = 'a'
    var_76 = 'A'
    var_77 = (var_75, var_76)
    var_78 = 'b'
    var_79 = 'B'
    var_80 = (var_78, var_79)
    var_81 = [var_77, var_80]
    var_82 = module_0.Choice(choices=var_81)
    var_83 = module_1.to_json_schema(var_82)
    var_84 = 'constant_value'
    var_85 = module_0.Const(var_84)
    var_86 = module_1.to_json_schema(var_85)
    var_87 = module_0.String()
    var_88 = module_0.Integer()
    var_89 = [var_87, var_88]
    var_90 = module_0.Union(var_89)
    var_91 = module_1.to_json_schema(var_90)
    var_92 = 'anyOf'
    var_93 = var_91[var_92]
    var_94 = len(var_93)
    assert var_94 == 2
    var_95 = module_0.String()
    var_96 = module_0.Integer()
    var_97 = [var_95, var_96]
    var_98 = module_2.OneOf(var_97)
    var_99 = module_1.to_json_schema(var_98)
    var_100 = 'oneOf'
    var_101 = var_99[var_100]
    var_102 = len(var_101)
    assert var_102 == 2
    var_103 = module_0.String()
    var_104 = 'test'
    var_105 = module_0.Const(var_104)
    var_106 = [var_103, var_105]
    var_107 = module_2.AllOf(var_106)
    var_108 = module_1.to_json_schema(var_107)
    var_109 = 'allOf'
    var_110 = var_108[var_109]
    var_111 = len(var_110)
    assert var_111 == 2
    var_112 = module_0.String()
    var_113 = module_0.Integer()
    var_114 = module_0.Boolean()
    var_115 = module_2.IfThenElse(var_112, var_113, var_114)
    var_116 = module_1.to_json_schema(var_115)
    var_117 = module_0.String()
    var_118 = module_0.Integer()
    var_119 = module_2.IfThenElse(var_117, var_118)
    var_120 = module_1.to_json_schema(var_119)
    var_121 = module_0.String()
    var_122 = module_2.Not(var_121)
    var_123 = module_1.to_json_schema(var_122)
    var_124 = module_3.Definitions()
    var_125 = 'id'
    var_126 = module_0.Integer()
    var_127 = {var_125: var_126}
    var_128 = 'TestSchema'
    var_129 = module_3.Reference(var_128, var_124)
    var_130 = module_1.to_json_schema(var_129)



# Parsed testcases at query #39
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
    var_14 = True
    var_15 = module_0.String(allow_blank=var_14)
    var_16 = module_1.to_json_schema(var_15)
    var_17 = 100
    var_18 = 5
    var_19 = module_0.Integer(minimum=var_4, maximum=var_17, multiple_of=var_18)
    var_20 = module_1.to_json_schema(var_19)
    var_21 = True
    var_22 = module_0.Integer()
    var_23 = module_1.to_json_schema(var_22)
    var_24 = module_0.Float(minimum=var_4, maximum=var_21)
    var_25 = module_1.to_json_schema(var_24)
    var_26 = module_0.Float(exclusive_minimum=var_4, exclusive_maximum=var_21)
    var_27 = module_1.to_json_schema(var_26)
    var_28 = module_0.Boolean()
    var_29 = module_1.to_json_schema(var_28)
    var_30 = True
    var_31 = module_0.Boolean()
    var_32 = module_1.to_json_schema(var_31)
    var_33 = True
    var_34 = module_0.Array(min_items=var_30, max_items=var_6, unique_items=var_33)
    var_35 = module_1.to_json_schema(var_34)
    var_36 = module_0.String()
    var_37 = module_0.Array(var_36)
    var_38 = module_1.to_json_schema(var_37)
    var_39 = module_0.String()
    var_40 = module_0.Integer()
    var_41 = [var_39, var_40]
    var_42 = module_0.Array(var_41)
    var_43 = module_1.to_json_schema(var_42)
    var_44 = 'items'
    var_45 = var_43[var_44]
    var_46 = var_43[var_44]
    var_47 = len(var_46)
    assert var_47 == 2
    var_48 = module_0.Array(additional_items=var_4)
    var_49 = module_1.to_json_schema(var_48)
    var_50 = module_0.String()
    var_51 = module_0.Array(additional_items=var_50)
    var_52 = module_1.to_json_schema(var_51)
    var_53 = module_0.Object(min_properties=var_33, max_properties=var_6)
    var_54 = module_1.to_json_schema(var_53)
    var_55 = 'name'
    var_56 = 'age'
    var_57 = module_0.String()
    var_58 = module_0.Integer()
    var_59 = {var_55: var_57, var_56: var_58}
    var_60 = module_0.Object(properties=var_59)
    var_61 = module_1.to_json_schema(var_60)
    var_62 = '^S_'
    var_63 = module_0.String()
    var_64 = {var_62: var_63}
    var_65 = module_0.Object(pattern_properties=var_64)
    var_66 = module_1.to_json_schema(var_65)
    var_67 = module_0.Object(additional_properties=var_4)
    var_68 = module_1.to_json_schema(var_67)
    var_69 = module_0.String()
    var_70 = module_0.Object(additional_properties=var_69)
    var_71 = module_1.to_json_schema(var_70)
    var_72 = module_0.String(pattern=var_7)
    var_73 = module_0.Object(property_names=var_72)
    var_74 = module_1.to_json_schema(var_73)
    var_75 = [var_55, var_56]
    var_76 = module_0.Object(required=var_75)
    var_77 = module_1.to_json_schema(var_76)
    var_78 = 'a'
    var_79 = 'Option A'
    var_80 = (var_78, var_79)
    var_81 = 'b'
    var_82 = 'Option B'
    var_83 = (var_81, var_82)
    var_84 = [var_80, var_83]
    var_85 = module_0.Choice(choices=var_84)
    var_86 = module_1.to_json_schema(var_85)
    var_87 = 'constant_value'
    var_88 = module_0.Const(var_87)
    var_89 = module_1.to_json_schema(var_88)
    var_90 = module_0.String()
    var_91 = module_0.Integer()
    var_92 = [var_90, var_91]
    var_93 = module_0.Union(var_92)
    var_94 = module_1.to_json_schema(var_93)
    var_95 = 'anyOf'
    var_96 = var_94[var_95]
    var_97 = len(var_96)
    assert var_97 == 2
    var_98 = module_0.String()
    var_99 = module_0.Integer()
    var_100 = [var_98, var_99]
    var_101 = module_2.OneOf(var_100)
    var_102 = module_1.to_json_schema(var_101)
    var_103 = 'oneOf'
    var_104 = var_102[var_103]
    var_105 = len(var_104)
    assert var_105 == 2
    var_106 = module_0.String()
    var_107 = 'test'
    var_108 = module_0.Const(var_107)
    var_109 = [var_106, var_108]
    var_110 = module_2.AllOf(var_109)
    var_111 = module_1.to_json_schema(var_110)
    var_112 = 'allOf'
    var_113 = var_111[var_112]
    var_114 = len(var_113)
    assert var_114 == 2
    var_115 = module_0.String()
    var_116 = module_0.Integer()
    var_117 = module_0.Boolean()
    var_118 = module_2.IfThenElse(var_115, var_116, var_117)
    var_119 = module_1.to_json_schema(var_118)
    var_120 = module_0.String()
    var_121 = module_0.Integer()
    var_122 = module_2.IfThenElse(var_120, var_121)
    var_123 = module_1.to_json_schema(var_122)
    var_124 = module_0.String()
    var_125 = module_2.Not(var_124)
    var_126 = module_1.to_json_schema(var_125)
    var_127 = 'TestRef'
    var_128 = module_0.String()
    var_129 = {var_127: var_128}
    var_130 = module_1.to_json_schema(var_125)



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
    var_4 = False
    var_5 = module_0.String()
    var_6 = module_1.to_json_schema(var_5)
    var_7 = True
    var_8 = module_0.String()
    var_9 = module_1.to_json_schema(var_8)
    var_10 = 5
    var_11 = 10
    var_12 = '^[a-z]+$'
    var_13 = module_0.String(max_length=var_11, min_length=var_10, pattern=var_12)
    var_14 = module_1.to_json_schema(var_13)
    var_15 = module_0.Integer()
    var_16 = module_1.to_json_schema(var_15)
    var_17 = 100
    var_18 = module_0.Integer(minimum=var_4, maximum=var_17, multiple_of=var_10)
    var_19 = module_1.to_json_schema(var_18)
    var_20 = module_0.Float()
    var_21 = module_1.to_json_schema(var_20)
    var_22 = module_0.Float(exclusive_minimum=var_4, exclusive_maximum=var_7)
    var_23 = module_1.to_json_schema(var_22)
    var_24 = module_0.Boolean()
    var_25 = module_1.to_json_schema(var_24)
    var_26 = module_0.Boolean()
    var_27 = module_1.to_json_schema(var_26)
    var_28 = module_0.String()
    var_29 = module_0.Array(var_28)
    var_30 = module_1.to_json_schema(var_29)
    var_31 = module_0.Array(min_items=var_7, max_items=var_11, unique_items=var_7)
    var_32 = module_1.to_json_schema(var_31)
    var_33 = module_0.String()
    var_34 = module_0.Integer()
    var_35 = [var_33, var_34]
    var_36 = module_0.Array(var_35)
    var_37 = module_1.to_json_schema(var_36)
    var_38 = 'items'
    var_39 = var_37[var_38]
    var_40 = var_37[var_38]
    var_41 = len(var_40)
    assert var_41 == 2
    var_42 = 'name'
    var_43 = 'age'
    var_44 = module_0.String()
    var_45 = module_0.Integer()
    var_46 = {var_42: var_44, var_43: var_45}
    var_47 = module_0.Object(properties=var_46)
    var_48 = module_1.to_json_schema(var_47)
    var_49 = module_0.String()
    var_50 = {var_42: var_49}
    var_51 = [var_42]
    var_52 = module_0.Object(properties=var_50, required=var_51)
    var_53 = module_1.to_json_schema(var_52)
    var_54 = '^S_'
    var_55 = module_0.String()
    var_56 = {var_54: var_55}
    var_57 = module_0.Object(pattern_properties=var_56)
    var_58 = module_1.to_json_schema(var_57)
    var_59 = module_0.String()
    var_60 = module_0.Object(additional_properties=var_59)
    var_61 = module_1.to_json_schema(var_60)
    var_62 = 'a'
    var_63 = 'Option A'
    var_64 = (var_62, var_63)
    var_65 = 'b'
    var_66 = 'Option B'
    var_67 = (var_65, var_66)
    var_68 = [var_64, var_67]
    var_69 = module_0.Choice(choices=var_68)
    var_70 = module_1.to_json_schema(var_69)
    var_71 = 'fixed_value'
    var_72 = module_0.Const(var_71)
    var_73 = module_1.to_json_schema(var_72)
    var_74 = module_0.String()
    var_75 = module_0.Integer()
    var_76 = [var_74, var_75]
    var_77 = module_0.Union(var_76)
    var_78 = module_1.to_json_schema(var_77)
    var_79 = 'anyOf'
    var_80 = var_78[var_79]
    var_81 = len(var_80)
    assert var_81 == 2
    var_82 = module_0.String()
    var_83 = module_0.Integer()
    var_84 = [var_82, var_83]
    var_85 = module_2.OneOf(var_84)
    var_86 = module_1.to_json_schema(var_85)
    var_87 = 'oneOf'
    var_88 = var_86[var_87]
    var_89 = len(var_88)
    assert var_89 == 2
    var_90 = module_0.String()
    var_91 = module_0.String()
    var_92 = [var_90, var_91]
    var_93 = module_2.AllOf(var_92)
    var_94 = module_1.to_json_schema(var_93)
    var_95 = 'allOf'
    var_96 = var_94[var_95]
    var_97 = len(var_96)
    assert var_97 == 2
    var_98 = module_0.String()
    var_99 = module_2.Not(var_98)
    var_100 = module_1.to_json_schema(var_99)
    var_101 = module_0.String()
    var_102 = module_0.Integer()
    var_103 = module_0.Boolean()
    var_104 = module_2.IfThenElse(var_101, var_102, var_103)
    var_105 = module_1.to_json_schema(var_104)
    var_106 = module_0.String()
    var_107 = module_0.Integer()
    var_108 = module_2.IfThenElse(var_106, var_107)
    var_109 = module_1.to_json_schema(var_108)
    var_110 = 'TestSchema'
    var_111 = module_0.String()
    var_112 = {var_110: var_111}
    var_113 = module_0.String()
    var_114 = {var_110: var_113}
    var_115 = module_0.String(allow_blank=var_4)
    var_116 = module_1.to_json_schema(var_115)
    var_117 = module_0.Array(additional_items=var_4)
    var_118 = module_1.to_json_schema(var_117)
    var_119 = module_0.String()
    var_120 = module_0.Array(additional_items=var_119)
    var_121 = module_1.to_json_schema(var_120)
    var_122 = module_0.String(pattern=var_12)
    var_123 = module_0.Object(property_names=var_122)
    var_124 = module_1.to_json_schema(var_123)
    var_125 = module_0.Object(min_properties=var_7, max_properties=var_10)
    var_126 = module_1.to_json_schema(var_125)



# Parsed testcases at query #41
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
    var_14 = True
    var_15 = None
    var_16 = module_0.String(allow_blank=var_14, min_length=var_15)
    var_17 = module_1.to_json_schema(var_16)
    var_18 = 100
    var_19 = 99
    var_20 = 5
    var_21 = module_0.Integer(minimum=var_4, maximum=var_18, exclusive_minimum=var_14, exclusive_maximum=var_19, multiple_of=var_20)
    var_22 = module_1.to_json_schema(var_21)
    var_23 = module_0.Float(minimum=var_4, maximum=var_14)
    var_24 = module_1.to_json_schema(var_23)
    var_25 = module_0.Boolean()
    var_26 = module_1.to_json_schema(var_25)
    var_27 = True
    var_28 = module_0.Boolean()
    var_29 = module_1.to_json_schema(var_28)
    var_30 = module_0.String()
    var_31 = True
    var_32 = module_0.Array(var_30, min_items=var_27, max_items=var_20, unique_items=var_31)
    var_33 = module_1.to_json_schema(var_32)
    var_34 = module_0.String()
    var_35 = module_0.Integer()
    var_36 = [var_34, var_35]
    var_37 = module_0.Array(var_36)
    var_38 = module_1.to_json_schema(var_37)
    var_39 = 'items'
    var_40 = var_38[var_39]
    var_41 = var_38[var_39]
    var_42 = len(var_41)
    assert var_42 == 2
    var_43 = module_0.Array(additional_items=var_4)
    var_44 = module_1.to_json_schema(var_43)
    var_45 = 'name'
    var_46 = 'age'
    var_47 = module_0.String()
    var_48 = module_0.Integer()
    var_49 = {var_45: var_47, var_46: var_48}
    var_50 = [var_45]
    var_51 = module_0.Object(properties=var_49, required=var_50)
    var_52 = module_1.to_json_schema(var_51)
    var_53 = '^S_'
    var_54 = '^I_'
    var_55 = module_0.String()
    var_56 = module_0.Integer()
    var_57 = {var_53: var_55, var_54: var_56}
    var_58 = module_0.Object(pattern_properties=var_57)
    var_59 = module_1.to_json_schema(var_58)
    var_60 = True
    var_61 = module_0.Object(additional_properties=var_60)
    var_62 = module_1.to_json_schema(var_61)
    var_63 = module_0.String(pattern=var_7)
    var_64 = module_0.Object(property_names=var_63)
    var_65 = module_1.to_json_schema(var_64)
    var_66 = 'a'
    var_67 = (var_66, var_60)
    var_68 = 'b'
    var_69 = 2
    var_70 = (var_68, var_69)
    var_71 = [var_67, var_70]
    var_72 = module_0.Choice(choices=var_71)
    var_73 = module_1.to_json_schema(var_72)
    var_74 = 'fixed_value'
    var_75 = module_0.Const(var_74)
    var_76 = module_1.to_json_schema(var_75)
    var_77 = module_0.String()
    var_78 = module_0.Integer()
    var_79 = [var_77, var_78]
    var_80 = module_0.Union(var_79)
    var_81 = module_1.to_json_schema(var_80)
    var_82 = 'anyOf'
    var_83 = var_81[var_82]
    var_84 = len(var_83)
    assert var_84 == 2
    var_85 = module_0.String()
    var_86 = module_0.Integer()
    var_87 = [var_85, var_86]
    var_88 = module_2.OneOf(var_87)
    var_89 = module_1.to_json_schema(var_88)
    var_90 = 'oneOf'
    var_91 = var_89[var_90]
    var_92 = len(var_91)
    assert var_92 == 2
    var_93 = module_0.String()
    var_94 = module_0.String(min_length=var_20)
    var_95 = [var_93, var_94]
    var_96 = module_2.AllOf(var_95)
    var_97 = module_1.to_json_schema(var_96)
    var_98 = 'allOf'
    var_99 = var_97[var_98]
    var_100 = len(var_99)
    assert var_100 == 2
    var_101 = module_0.String()
    var_102 = module_0.Integer()
    var_103 = module_0.Boolean()
    var_104 = module_2.IfThenElse(var_101, var_102, var_103)
    var_105 = module_1.to_json_schema(var_104)
    var_106 = module_0.String()
    var_107 = module_0.Integer()
    var_108 = module_2.IfThenElse(var_106, var_107)
    var_109 = module_1.to_json_schema(var_108)
    var_110 = module_0.String()
    var_111 = module_2.Not(var_110)
    var_112 = module_1.to_json_schema(var_111)
    var_113 = module_3.Definitions()
    var_114 = module_1.to_json_schema(var_113)
    var_115 = 'CustomType'
    var_116 = module_3.Definitions()
    var_117 = module_3.Reference(var_115, var_116)
    var_118 = module_1.to_json_schema(var_117)
    var_119 = module_0.String()
    var_120 = module_0.Integer()
    var_121 = {var_45: var_119, var_46: var_120}
    var_122 = [var_45]
    var_123 = module_3.Schema(var_121)
    var_124 = module_1.to_json_schema(var_123)
    var_125 = 'test_default'
    var_126 = module_0.String()
    var_127 = module_1.to_json_schema(var_126)
    var_128 = 'default'
    var_129 = 'user'
    var_130 = module_0.String()
    var_131 = module_0.String()
    var_132 = {var_45: var_130, var_8: var_131}
    var_133 = module_0.Object(properties=var_132)
    var_134 = 'id'
    var_135 = 'value'
    var_136 = module_0.Integer()
    var_137 = module_0.String()
    var_138 = {var_134: var_136, var_135: var_137}
    var_139 = module_0.Object(properties=var_138)
    var_140 = module_0.Array(var_139)
    var_141 = {var_129: var_133, var_39: var_140}
    var_142 = module_0.Object(properties=var_141)



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
    var_4 = False
    var_5 = module_0.String()
    var_6 = module_1.to_json_schema(var_5)
    var_7 = True
    var_8 = module_0.String()
    var_9 = module_1.to_json_schema(var_8)
    var_10 = 2
    var_11 = 10
    var_12 = module_0.String(allow_blank=var_4, max_length=var_11, min_length=var_10)
    var_13 = module_1.to_json_schema(var_12)
    var_14 = module_0.Integer()
    var_15 = module_1.to_json_schema(var_14)
    var_16 = 100
    var_17 = module_0.Integer(minimum=var_4, maximum=var_16)
    var_18 = module_1.to_json_schema(var_17)
    var_19 = module_0.Float()
    var_20 = module_1.to_json_schema(var_19)
    var_21 = module_0.Boolean()
    var_22 = module_1.to_json_schema(var_21)
    var_23 = module_0.Boolean()
    var_24 = module_1.to_json_schema(var_23)
    var_25 = module_0.String()
    var_26 = module_0.Array(var_25)
    var_27 = module_1.to_json_schema(var_26)
    var_28 = 5
    var_29 = module_0.Array(min_items=var_7, max_items=var_28)
    var_30 = module_1.to_json_schema(var_29)
    var_31 = 'name'
    var_32 = module_0.String()
    var_33 = {var_31: var_32}
    var_34 = module_0.Object(properties=var_33)
    var_35 = module_1.to_json_schema(var_34)
    var_36 = module_0.String()
    var_37 = {var_31: var_36}
    var_38 = [var_31]
    var_39 = module_0.Object(properties=var_37, required=var_38)
    var_40 = module_1.to_json_schema(var_39)
    var_41 = 'a'
    var_42 = 'Option A'
    var_43 = (var_41, var_42)
    var_44 = 'b'
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
    var_69 = module_0.String()
    var_70 = 'A'
    var_71 = (var_41, var_70)
    var_72 = [var_71]
    var_73 = module_0.Choice(choices=var_72)
    var_74 = [var_69, var_73]
    var_75 = module_2.AllOf(var_74)
    var_76 = module_1.to_json_schema(var_75)
    var_77 = 'allOf'
    var_78 = var_76[var_77]
    var_79 = len(var_78)
    assert var_79 == 2
    var_80 = module_0.String()
    var_81 = module_2.Not(var_80)
    var_82 = module_1.to_json_schema(var_81)
    var_83 = module_0.String()
    var_84 = module_0.Integer()
    var_85 = module_0.Boolean()
    var_86 = module_2.IfThenElse(var_83, var_84, var_85)
    var_87 = module_1.to_json_schema(var_86)
    var_88 = module_0.String()
    var_89 = module_0.Integer()
    var_90 = module_2.IfThenElse(var_88, var_89)
    var_91 = module_1.to_json_schema(var_90)
    var_92 = module_3.Definitions()
    var_93 = 'MyType'
    var_94 = module_3.Reference(var_93, var_92)
    var_95 = module_1.to_json_schema(var_94)
    var_96 = module_3.Definitions()
    var_97 = module_1.to_json_schema(var_96)
    var_98 = 'user'
    var_99 = 'tags'
    var_100 = 'age'
    var_101 = module_0.String()
    var_102 = module_0.Integer()
    var_103 = {var_31: var_101, var_100: var_102}
    var_104 = module_0.Object(properties=var_103)
    var_105 = module_0.String()
    var_106 = module_0.Array(var_105)
    var_107 = {var_98: var_104, var_99: var_106}
    var_108 = module_0.Object(properties=var_107)
    var_109 = module_1.to_json_schema(var_108)
    var_110 = module_0.String()
    var_111 = module_0.Array(var_110, unique_items=var_7)
    var_112 = module_1.to_json_schema(var_111)
    var_113 = '^S_'
    var_114 = module_0.String()
    var_115 = {var_113: var_114}
    var_116 = module_0.Object(pattern_properties=var_115)
    var_117 = module_1.to_json_schema(var_116)
    var_118 = module_0.String()
    var_119 = module_0.Object(additional_properties=var_118)
    var_120 = module_1.to_json_schema(var_119)
    var_121 = '^[a-z]+$'
    var_122 = module_0.String(pattern=var_121)
    var_123 = module_1.to_json_schema(var_122)
    var_124 = 'email'
    var_125 = module_0.String(format=var_124)
    var_126 = module_1.to_json_schema(var_125)
    var_127 = module_0.Float(exclusive_minimum=var_4, exclusive_maximum=var_16)
    var_128 = module_1.to_json_schema(var_127)
    var_129 = module_0.Integer(multiple_of=var_28)
    var_130 = module_1.to_json_schema(var_129)



# Parsed testcases at query #43
#--------------------------


import typesystem.fields as module_0
import typesystem.json_schema as module_1
import typesystem.composites as module_2
import re as module_3
import typesystem.schemas as module_4

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True
    var_2 = module_2.NeverMatch()
    var_3 = module_1.to_json_schema(var_2)
    assert var_3 is False
    var_4 = module_0.String()
    var_5 = module_1.to_json_schema(var_4)
    var_6 = True
    var_7 = module_0.String()
    var_8 = module_1.to_json_schema(var_7)
    var_9 = 2
    var_10 = 10
    var_11 = module_0.String(max_length=var_10, min_length=var_9)
    var_12 = module_1.to_json_schema(var_11)
    var_13 = '^\\d+$'
    var_14 = module_3.compile(var_13)
    var_15 = module_0.String(pattern=var_14)
    var_16 = module_1.to_json_schema(var_15)
    var_17 = 'email'
    var_18 = module_0.String(format=var_17)
    var_19 = module_1.to_json_schema(var_18)
    var_20 = module_0.Integer()
    var_21 = module_1.to_json_schema(var_20)
    var_22 = 0
    var_23 = 100
    var_24 = 5
    var_25 = module_0.Integer(minimum=var_22, maximum=var_23, multiple_of=var_24)
    var_26 = module_1.to_json_schema(var_25)
    var_27 = module_0.Float()
    var_28 = module_1.to_json_schema(var_27)
    var_29 = module_0.Boolean()
    var_30 = module_1.to_json_schema(var_29)
    var_31 = module_0.Boolean()
    var_32 = module_1.to_json_schema(var_31)
    var_33 = module_0.String()
    var_34 = module_0.Array(var_33)
    var_35 = module_1.to_json_schema(var_34)
    var_36 = module_0.Integer()
    var_37 = module_0.Array(var_36, min_items=var_6, max_items=var_10)
    var_38 = module_1.to_json_schema(var_37)
    var_39 = module_0.String()
    var_40 = module_0.Array(var_39, unique_items=var_6)
    var_41 = module_1.to_json_schema(var_40)
    var_42 = 'name'
    var_43 = 'age'
    var_44 = module_0.String()
    var_45 = module_0.Integer()
    var_46 = {var_42: var_44, var_43: var_45}
    var_47 = module_0.Object(properties=var_46)
    var_48 = module_1.to_json_schema(var_47)
    var_49 = module_0.String()
    var_50 = module_0.Integer()
    var_51 = {var_42: var_49, var_43: var_50}
    var_52 = [var_42]
    var_53 = module_0.Object(properties=var_51, required=var_52)
    var_54 = module_1.to_json_schema(var_53)
    var_55 = '^S_'
    var_56 = module_0.String()
    var_57 = {var_55: var_56}
    var_58 = module_0.Object(pattern_properties=var_57)
    var_59 = module_1.to_json_schema(var_58)
    var_60 = 'a'
    var_61 = 'Option A'
    var_62 = (var_60, var_61)
    var_63 = 'b'
    var_64 = 'Option B'
    var_65 = (var_63, var_64)
    var_66 = [var_62, var_65]
    var_67 = module_0.Choice(choices=var_66)
    var_68 = module_1.to_json_schema(var_67)
    var_69 = 'fixed_value'
    var_70 = module_0.Const(var_69)
    var_71 = module_1.to_json_schema(var_70)
    var_72 = module_0.String()
    var_73 = module_0.Integer()
    var_74 = [var_72, var_73]
    var_75 = module_0.Union(var_74)
    var_76 = module_1.to_json_schema(var_75)
    var_77 = 'anyOf'
    var_78 = var_76[var_77]
    var_79 = len(var_78)
    assert var_79 == 2
    var_80 = module_0.String()
    var_81 = module_0.Integer()
    var_82 = [var_80, var_81]
    var_83 = module_2.OneOf(var_82)
    var_84 = module_1.to_json_schema(var_83)
    var_85 = 'oneOf'
    var_86 = var_84[var_85]
    var_87 = len(var_86)
    assert var_87 == 2
    var_88 = module_0.String()
    var_89 = module_0.String(min_length=var_6)
    var_90 = [var_88, var_89]
    var_91 = module_2.AllOf(var_90)
    var_92 = module_1.to_json_schema(var_91)
    var_93 = 'allOf'
    var_94 = var_92[var_93]
    var_95 = len(var_94)
    assert var_95 == 2
    var_96 = module_0.String()
    var_97 = module_2.Not(var_96)
    var_98 = module_1.to_json_schema(var_97)
    var_99 = 'A'
    var_100 = (var_60, var_99)
    var_101 = 'B'
    var_102 = (var_63, var_101)
    var_103 = [var_100, var_102]
    var_104 = module_0.Choice(choices=var_103)
    var_105 = module_0.String()
    var_106 = module_0.Integer()
    var_107 = module_2.IfThenElse(var_104, var_105, var_106)
    var_108 = module_1.to_json_schema(var_107)
    var_109 = module_4.Definitions()
    var_110 = 'CustomSchema'
    var_111 = module_4.Reference(var_110, var_109)
    var_112 = module_1.to_json_schema(var_111)
    var_113 = module_4.Definitions()
    var_114 = module_1.to_json_schema(var_113)
    var_115 = False
    var_116 = module_0.String(allow_blank=var_115)
    var_117 = module_1.to_json_schema(var_116)
    var_118 = module_0.String()
    var_119 = module_0.Integer()
    var_120 = [var_118, var_119]
    var_121 = module_0.Array(var_120)
    var_122 = module_1.to_json_schema(var_121)
    var_123 = 'items'
    var_124 = var_122[var_123]
    var_125 = var_122[var_123]
    var_126 = len(var_125)
    assert var_126 == 2
    var_127 = module_0.String()
    var_128 = False
    var_129 = module_0.Array(var_127, var_128)
    var_130 = module_1.to_json_schema(var_129)
    var_131 = module_0.String()
    var_132 = module_0.Integer()
    var_133 = module_0.Array(var_131, var_132)
    var_134 = module_1.to_json_schema(var_133)
    var_135 = 'additionalItems'
    var_136 = var_134[var_135]
    var_137 = False
    var_138 = module_0.Object(additional_properties=var_137)
    var_139 = module_1.to_json_schema(var_138)
    var_140 = module_0.String()
    var_141 = module_0.Object(additional_properties=var_140)
    var_142 = module_1.to_json_schema(var_141)
    var_143 = 'additionalProperties'
    var_144 = var_142[var_143]



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
    var_4 = False
    var_5 = True
    var_6 = None
    var_7 = module_0.String(allow_blank=var_5, max_length=var_6, min_length=var_6, pattern=var_6, format=var_6)
    var_8 = module_1.to_json_schema(var_7)
    var_9 = module_0.String(allow_blank=var_5, max_length=var_6, min_length=var_6, pattern=var_6, format=var_6)
    var_10 = module_1.to_json_schema(var_9)
    var_11 = 2
    var_12 = 10
    var_13 = 'email'
    var_14 = '^[a-z]+$'
    var_15 = module_0.String(allow_blank=var_4, max_length=var_12, min_length=var_11, pattern=var_14, format=var_13)
    var_16 = module_1.to_json_schema(var_15)
    var_17 = 100
    var_18 = module_0.Integer(minimum=var_4, maximum=var_17, exclusive_minimum=var_6, exclusive_maximum=var_6, multiple_of=var_6)
    var_19 = module_1.to_json_schema(var_18)
    var_20 = module_0.Float(minimum=var_4, maximum=var_5, exclusive_minimum=var_6, exclusive_maximum=var_6, multiple_of=var_6)
    var_21 = module_1.to_json_schema(var_20)
    var_22 = module_0.Boolean()
    var_23 = module_1.to_json_schema(var_22)
    var_24 = module_0.Boolean()
    var_25 = module_1.to_json_schema(var_24)
    var_26 = module_0.String(allow_blank=var_5, max_length=var_6, min_length=var_6, pattern=var_6, format=var_6)
    var_27 = module_0.Array(var_26, var_6, var_4, var_6, unique_items=var_4)
    var_28 = module_1.to_json_schema(var_27)
    var_29 = module_0.Integer(minimum=var_6, maximum=var_6, exclusive_minimum=var_6, exclusive_maximum=var_6, multiple_of=var_6)
    var_30 = 5
    var_31 = module_0.Array(var_29, var_6, var_5, var_30, unique_items=var_5)
    var_32 = module_1.to_json_schema(var_31)
    var_33 = 'name'
    var_34 = module_0.String(allow_blank=var_5, max_length=var_6, min_length=var_6, pattern=var_6, format=var_6)
    var_35 = {var_33: var_34}
    var_36 = [var_33]
    var_37 = module_0.Object(properties=var_35, pattern_properties=var_6, additional_properties=var_6, property_names=var_6, min_properties=var_6, max_properties=var_6, required=var_36)
    var_38 = module_1.to_json_schema(var_37)
    var_39 = 'a'
    var_40 = (var_39, var_39)
    var_41 = 'b'
    var_42 = (var_41, var_41)
    var_43 = [var_40, var_42]
    var_44 = module_0.Choice(choices=var_43)
    var_45 = module_1.to_json_schema(var_44)
    var_46 = 'fixed_value'
    var_47 = module_0.Const(var_46)
    var_48 = module_1.to_json_schema(var_47)
    var_49 = module_0.String(allow_blank=var_5, max_length=var_6, min_length=var_6, pattern=var_6, format=var_6)
    var_50 = module_0.Integer(minimum=var_6, maximum=var_6, exclusive_minimum=var_6, exclusive_maximum=var_6, multiple_of=var_6)
    var_51 = [var_49, var_50]
    var_52 = module_0.Union(var_51)
    var_53 = module_1.to_json_schema(var_52)
    var_54 = 'anyOf'
    var_55 = var_53[var_54]
    var_56 = len(var_55)
    assert var_56 == 2
    var_57 = module_0.String(allow_blank=var_5, max_length=var_6, min_length=var_6, pattern=var_6, format=var_6)
    var_58 = module_0.Integer(minimum=var_6, maximum=var_6, exclusive_minimum=var_6, exclusive_maximum=var_6, multiple_of=var_6)
    var_59 = [var_57, var_58]
    var_60 = module_2.OneOf(var_59)
    var_61 = module_1.to_json_schema(var_60)
    var_62 = 'oneOf'
    var_63 = var_61[var_62]
    var_64 = len(var_63)
    assert var_64 == 2
    var_65 = module_0.String(allow_blank=var_5, max_length=var_6, min_length=var_6, pattern=var_6, format=var_6)
    var_66 = module_0.Integer(minimum=var_6, maximum=var_6, exclusive_minimum=var_6, exclusive_maximum=var_6, multiple_of=var_6)
    var_67 = [var_65, var_66]
    var_68 = module_2.AllOf(var_67)
    var_69 = module_1.to_json_schema(var_68)
    var_70 = 'allOf'
    var_71 = var_69[var_70]
    var_72 = len(var_71)
    assert var_72 == 2
    var_73 = module_0.String(allow_blank=var_5, max_length=var_6, min_length=var_6, pattern=var_6, format=var_6)
    var_74 = module_2.Not(var_73)
    var_75 = module_1.to_json_schema(var_74)
    var_76 = module_0.Boolean()
    var_77 = module_0.String(allow_blank=var_5, max_length=var_6, min_length=var_6, pattern=var_6, format=var_6)
    var_78 = module_0.Integer(minimum=var_6, maximum=var_6, exclusive_minimum=var_6, exclusive_maximum=var_6, multiple_of=var_6)
    var_79 = module_2.IfThenElse(var_76, var_77, var_78)
    var_80 = module_1.to_json_schema(var_79)
    var_81 = module_3.Definitions()
    var_82 = module_1.to_json_schema(var_81)
    var_83 = module_3.Definitions()
    var_84 = module_0.String(allow_blank=var_5, max_length=var_6, min_length=var_6, pattern=var_6, format=var_6)
    var_85 = {var_33: var_84}
    var_86 = 'User'
    var_87 = module_3.Reference(var_86, var_83)
    var_88 = module_1.to_json_schema(var_87)



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
    var_14 = True
    var_15 = module_0.String(allow_blank=var_14)
    var_16 = module_1.to_json_schema(var_15)
    var_17 = 100
    var_18 = 99
    var_19 = 5
    var_20 = module_0.Integer(minimum=var_4, maximum=var_17, exclusive_minimum=var_14, exclusive_maximum=var_18, multiple_of=var_19)
    var_21 = module_1.to_json_schema(var_20)
    var_22 = True
    var_23 = module_0.Integer()
    var_24 = module_1.to_json_schema(var_23)
    var_25 = module_0.Float(minimum=var_4, maximum=var_17)
    var_26 = module_1.to_json_schema(var_25)
    var_27 = module_0.Boolean()
    var_28 = module_1.to_json_schema(var_27)
    var_29 = True
    var_30 = module_0.Boolean()
    var_31 = module_1.to_json_schema(var_30)
    var_32 = module_0.String()
    var_33 = True
    var_34 = module_0.Array(var_32, min_items=var_29, max_items=var_6, unique_items=var_33)
    var_35 = module_1.to_json_schema(var_34)
    var_36 = module_0.String()
    var_37 = module_0.Integer()
    var_38 = [var_36, var_37]
    var_39 = module_0.Array(var_38)
    var_40 = module_1.to_json_schema(var_39)
    var_41 = 'items'
    var_42 = var_40[var_41]
    var_43 = var_40[var_41]
    var_44 = len(var_43)
    assert var_44 == 2
    var_45 = module_0.Array(additional_items=var_4)
    var_46 = module_1.to_json_schema(var_45)
    var_47 = 'name'
    var_48 = 'age'
    var_49 = module_0.String()
    var_50 = module_0.Integer()
    var_51 = {var_47: var_49, var_48: var_50}
    var_52 = '^S_'
    var_53 = module_0.String()
    var_54 = {var_52: var_53}
    var_55 = [var_47]
    var_56 = module_0.Object(properties=var_51, pattern_properties=var_54, additional_properties=var_4, min_properties=var_33, max_properties=var_6, required=var_55)
    var_57 = module_1.to_json_schema(var_56)
    var_58 = True
    var_59 = module_0.Object()
    var_60 = module_1.to_json_schema(var_59)
    var_61 = 'a'
    var_62 = 'A'
    var_63 = (var_61, var_62)
    var_64 = 'b'
    var_65 = 'B'
    var_66 = (var_64, var_65)
    var_67 = [var_63, var_66]
    var_68 = module_0.Choice(choices=var_67)
    var_69 = module_1.to_json_schema(var_68)
    var_70 = 'fixed_value'
    var_71 = module_0.Const(var_70)
    var_72 = module_1.to_json_schema(var_71)
    var_73 = module_0.String()
    var_74 = module_0.Integer()
    var_75 = [var_73, var_74]
    var_76 = module_0.Union(var_75)
    var_77 = module_1.to_json_schema(var_76)
    var_78 = 'anyOf'
    var_79 = var_77[var_78]
    var_80 = len(var_79)
    assert var_80 == 2
    var_81 = module_0.String()
    var_82 = module_0.Integer()
    var_83 = [var_81, var_82]
    var_84 = module_2.OneOf(var_83)
    var_85 = module_1.to_json_schema(var_84)
    var_86 = 'oneOf'
    var_87 = var_85[var_86]
    var_88 = len(var_87)
    assert var_88 == 2
    var_89 = module_0.String()
    var_90 = module_0.String(min_length=var_19)
    var_91 = [var_89, var_90]
    var_92 = module_2.AllOf(var_91)
    var_93 = module_1.to_json_schema(var_92)
    var_94 = 'allOf'
    var_95 = var_93[var_94]
    var_96 = len(var_95)
    assert var_96 == 2
    var_97 = module_0.String()
    var_98 = module_0.Integer()
    var_99 = module_0.Boolean()
    var_100 = module_2.IfThenElse(var_97, var_98, var_99)
    var_101 = module_1.to_json_schema(var_100)
    var_102 = module_0.String()
    var_103 = module_0.Integer()
    var_104 = module_2.IfThenElse(var_102, var_103)
    var_105 = module_1.to_json_schema(var_104)
    var_106 = module_0.String()
    var_107 = module_2.Not(var_106)
    var_108 = module_1.to_json_schema(var_107)
    var_109 = 'default_value'
    var_110 = module_0.String()
    var_111 = module_1.to_json_schema(var_110)
    var_112 = 'User'
    var_113 = module_0.String()
    var_114 = {var_47: var_113}
    var_115 = module_0.Object(properties=var_114)
    var_116 = {var_112: var_115}
    var_117 = module_0.Object()
    var_118 = {var_112: var_117}
    var_119 = module_0.String()
    var_120 = module_0.Integer()
    var_121 = {var_47: var_119, var_48: var_120}
    var_122 = [var_47]
    var_123 = module_3.Schema(var_121)
    var_124 = module_1.to_json_schema(var_123)



# Parsed testcases at query #46
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
    var_4 = False
    var_5 = True
    var_6 = None
    var_7 = 'default'
    var_8 = 2
    var_9 = 10
    var_10 = 'email'
    var_11 = 100
    var_12 = 5
    var_13 = 'name'
    var_14 = 'age'
    var_15 = [var_13]
    var_16 = 'a'
    var_17 = 'Option A'
    var_18 = (var_16, var_17)
    var_19 = 'b'
    var_20 = 'Option B'
    var_21 = (var_19, var_20)
    var_22 = [var_18, var_21]
    var_23 = 'constant_value'
    var_24 = 'anyOf'
    var_25 = 'oneOf'



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
    var_4 = module_0.String()
    var_5 = module_1.to_json_schema(var_4)
    var_6 = True
    var_7 = module_0.String()
    var_8 = module_1.to_json_schema(var_7)
    var_9 = 5
    var_10 = 10
    var_11 = False
    var_12 = module_0.String(allow_blank=var_11, max_length=var_10, min_length=var_9)
    var_13 = module_1.to_json_schema(var_12)
    var_14 = '^[a-z]+$'
    var_15 = module_0.String(pattern=var_14)
    var_16 = module_1.to_json_schema(var_15)
    var_17 = 'email'
    var_18 = module_0.String(format=var_17)
    var_19 = module_1.to_json_schema(var_18)
    var_20 = module_0.Integer()
    var_21 = module_1.to_json_schema(var_20)
    var_22 = 100
    var_23 = module_0.Integer(minimum=var_11, maximum=var_22)
    var_24 = module_1.to_json_schema(var_23)
    var_25 = module_0.Float()
    var_26 = module_1.to_json_schema(var_25)
    var_27 = module_0.Float(exclusive_minimum=var_11, exclusive_maximum=var_6)
    var_28 = module_1.to_json_schema(var_27)
    var_29 = module_0.Boolean()
    var_30 = module_1.to_json_schema(var_29)
    var_31 = module_0.Boolean()
    var_32 = module_1.to_json_schema(var_31)
    var_33 = module_0.Array()
    var_34 = module_1.to_json_schema(var_33)
    var_35 = module_0.String()
    var_36 = module_0.Array(var_35)
    var_37 = module_1.to_json_schema(var_36)
    var_38 = module_0.Array(min_items=var_6, max_items=var_10)
    var_39 = module_1.to_json_schema(var_38)
    var_40 = module_0.Array(unique_items=var_6)
    var_41 = module_1.to_json_schema(var_40)
    var_42 = module_0.Object()
    var_43 = module_1.to_json_schema(var_42)
    var_44 = 'name'
    var_45 = 'age'
    var_46 = module_0.String()
    var_47 = module_0.Integer()
    var_48 = {var_44: var_46, var_45: var_47}
    var_49 = module_0.Object(properties=var_48)
    var_50 = module_1.to_json_schema(var_49)
    var_51 = [var_44]
    var_52 = module_0.Object(required=var_51)
    var_53 = module_1.to_json_schema(var_52)
    var_54 = module_0.Object(min_properties=var_6, max_properties=var_9)
    var_55 = module_1.to_json_schema(var_54)
    var_56 = 'a'
    var_57 = 'A'
    var_58 = (var_56, var_57)
    var_59 = 'b'
    var_60 = 'B'
    var_61 = (var_59, var_60)
    var_62 = [var_58, var_61]
    var_63 = module_0.Choice(choices=var_62)
    var_64 = module_1.to_json_schema(var_63)
    var_65 = 'fixed_value'
    var_66 = module_0.Const(var_65)
    var_67 = module_1.to_json_schema(var_66)
    var_68 = module_0.String()
    var_69 = module_0.Integer()
    var_70 = [var_68, var_69]
    var_71 = module_0.Union(var_70)
    var_72 = module_1.to_json_schema(var_71)
    var_73 = 'anyOf'
    var_74 = var_72[var_73]
    var_75 = len(var_74)
    assert var_75 == 2
    var_76 = module_0.String()
    var_77 = module_0.Integer()
    var_78 = [var_76, var_77]
    var_79 = module_2.OneOf(var_78)
    var_80 = module_1.to_json_schema(var_79)
    var_81 = 'oneOf'
    var_82 = var_80[var_81]
    var_83 = len(var_82)
    assert var_83 == 2
    var_84 = module_0.String()
    var_85 = module_0.String(min_length=var_9)
    var_86 = [var_84, var_85]
    var_87 = module_2.AllOf(var_86)
    var_88 = module_1.to_json_schema(var_87)
    var_89 = 'allOf'
    var_90 = var_88[var_89]
    var_91 = len(var_90)
    assert var_91 == 2
    var_92 = module_0.String()
    var_93 = module_2.Not(var_92)
    var_94 = module_1.to_json_schema(var_93)
    var_95 = module_0.String()
    var_96 = module_0.Integer()
    var_97 = module_0.Boolean()
    var_98 = module_2.IfThenElse(var_95, var_96, var_97)
    var_99 = module_1.to_json_schema(var_98)
    var_100 = module_0.String()
    var_101 = module_2.IfThenElse(var_100)
    var_102 = module_1.to_json_schema(var_101)
    var_103 = module_3.Definitions()
    var_104 = 'User'
    var_105 = module_3.Reference(var_104, var_103)
    var_106 = module_1.to_json_schema(var_105)
    var_107 = module_0.String()
    var_108 = {var_104: var_107}
    var_109 = module_0.String()
    var_110 = {var_44: var_109}
    var_111 = module_0.Object(properties=var_110)
    var_112 = None
    var_113 = module_1.to_json_schema(var_111, var_112)
    var_114 = module_0.String()
    var_115 = module_0.Integer()
    var_116 = [var_114, var_115]
    var_117 = module_0.Array(var_116)
    var_118 = module_1.to_json_schema(var_117)
    var_119 = 'items'
    var_120 = var_118[var_119]
    var_121 = var_118[var_119]
    var_122 = len(var_121)
    assert var_122 == 2
    var_123 = module_0.Array(additional_items=var_11)
    var_124 = module_1.to_json_schema(var_123)
    var_125 = '^S_'
    var_126 = module_0.String()
    var_127 = {var_125: var_126}
    var_128 = module_0.Object(pattern_properties=var_127)
    var_129 = module_1.to_json_schema(var_128)
    var_130 = module_0.Object(additional_properties=var_11)
    var_131 = module_1.to_json_schema(var_130)
    var_132 = '^[a-z_]+$'
    var_133 = module_0.String(pattern=var_132)
    var_134 = module_0.Object(property_names=var_133)
    var_135 = module_1.to_json_schema(var_134)
    var_136 = module_0.String(allow_blank=var_11)
    var_137 = module_1.to_json_schema(var_136)
    var_138 = module_0.Integer(multiple_of=var_9)
    var_139 = module_1.to_json_schema(var_138)
    var_140 = module_1.to_json_schema(var_0)



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
    var_14 = True
    var_15 = None
    var_16 = module_0.String(allow_blank=var_14, min_length=var_15)
    var_17 = module_1.to_json_schema(var_16)
    var_18 = 100
    var_19 = 99
    var_20 = 5
    var_21 = module_0.Integer(minimum=var_4, maximum=var_18, exclusive_minimum=var_14, exclusive_maximum=var_19, multiple_of=var_20)
    var_22 = module_1.to_json_schema(var_21)
    var_23 = True
    var_24 = module_0.Integer()
    var_25 = module_1.to_json_schema(var_24)
    var_26 = module_0.Float(minimum=var_4, maximum=var_23)
    var_27 = module_1.to_json_schema(var_26)
    var_28 = module_0.Boolean()
    var_29 = module_1.to_json_schema(var_28)
    var_30 = True
    var_31 = module_0.Boolean()
    var_32 = module_1.to_json_schema(var_31)
    var_33 = module_0.String()
    var_34 = True
    var_35 = module_0.Array(var_33, min_items=var_30, max_items=var_6, unique_items=var_34)
    var_36 = module_1.to_json_schema(var_35)
    var_37 = module_0.String()
    var_38 = module_0.Integer()
    var_39 = [var_37, var_38]
    var_40 = module_0.Array(var_39)
    var_41 = module_1.to_json_schema(var_40)
    var_42 = 'items'
    var_43 = var_41[var_42]
    var_44 = var_41[var_42]
    var_45 = len(var_44)
    assert var_45 == 2
    var_46 = module_0.Array(additional_items=var_4)
    var_47 = module_1.to_json_schema(var_46)
    var_48 = module_0.String()
    var_49 = module_0.Array(additional_items=var_48)
    var_50 = module_1.to_json_schema(var_49)
    var_51 = 'additionalItems'
    var_52 = var_50[var_51]
    var_53 = 'name'
    var_54 = 'age'
    var_55 = module_0.String()
    var_56 = module_0.Integer()
    var_57 = {var_53: var_55, var_54: var_56}
    var_58 = [var_53]
    var_59 = module_0.Object(properties=var_57, min_properties=var_34, max_properties=var_6, required=var_58)
    var_60 = module_1.to_json_schema(var_59)
    var_61 = '^S_'
    var_62 = '^I_'
    var_63 = module_0.String()
    var_64 = module_0.Integer()
    var_65 = {var_61: var_63, var_62: var_64}
    var_66 = module_0.Object(pattern_properties=var_65)
    var_67 = module_1.to_json_schema(var_66)
    var_68 = module_0.Object(additional_properties=var_4)
    var_69 = module_1.to_json_schema(var_68)
    var_70 = module_0.String()
    var_71 = module_0.Object(additional_properties=var_70)
    var_72 = module_1.to_json_schema(var_71)
    var_73 = 'additionalProperties'
    var_74 = var_72[var_73]
    var_75 = module_0.String(pattern=var_7)
    var_76 = module_0.Object(property_names=var_75)
    var_77 = module_1.to_json_schema(var_76)
    var_78 = 'a'
    var_79 = 'Option A'
    var_80 = (var_78, var_79)
    var_81 = 'b'
    var_82 = 'Option B'
    var_83 = (var_81, var_82)
    var_84 = [var_80, var_83]
    var_85 = module_0.Choice(choices=var_84)
    var_86 = module_1.to_json_schema(var_85)
    var_87 = 'fixed_value'
    var_88 = module_0.Const(var_87)
    var_89 = module_1.to_json_schema(var_88)
    var_90 = module_0.String()
    var_91 = module_0.Integer()
    var_92 = [var_90, var_91]
    var_93 = module_0.Union(var_92)
    var_94 = module_1.to_json_schema(var_93)
    var_95 = 'anyOf'
    var_96 = var_94[var_95]
    var_97 = len(var_96)
    assert var_97 == 2
    var_98 = module_0.String()
    var_99 = module_0.Integer()
    var_100 = [var_98, var_99]
    var_101 = module_2.OneOf(var_100)
    var_102 = module_1.to_json_schema(var_101)
    var_103 = 'oneOf'
    var_104 = var_102[var_103]
    var_105 = len(var_104)
    assert var_105 == 2
    var_106 = module_0.String()
    var_107 = module_0.Object()
    var_108 = [var_106, var_107]
    var_109 = module_2.AllOf(var_108)
    var_110 = module_1.to_json_schema(var_109)
    var_111 = 'allOf'
    var_112 = var_110[var_111]
    var_113 = len(var_112)
    assert var_113 == 2
    var_114 = module_0.String()
    var_115 = module_2.Not(var_114)
    var_116 = module_1.to_json_schema(var_115)
    var_117 = module_0.String()
    var_118 = module_0.Integer()
    var_119 = module_0.Boolean()
    var_120 = module_2.IfThenElse(var_117, var_118, var_119)
    var_121 = module_1.to_json_schema(var_120)
    var_122 = module_0.String()
    var_123 = module_2.IfThenElse(var_122)
    var_124 = module_1.to_json_schema(var_123)
    var_125 = module_3.Definitions()
    var_126 = module_1.to_json_schema(var_125)



# Parsed testcases at query #49
#--------------------------


import typesystem.fields as module_0
import typesystem.json_schema as module_1
import typesystem.composites as module_2
import re as module_3

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True
    var_2 = module_2.NeverMatch()
    var_3 = module_1.to_json_schema(var_2)
    assert var_3 is False
    var_4 = False
    var_5 = module_0.String()
    var_6 = module_1.to_json_schema(var_5)
    var_7 = True
    var_8 = module_0.String()
    var_9 = module_1.to_json_schema(var_8)
    var_10 = 5
    var_11 = module_0.String(min_length=var_10)
    var_12 = module_1.to_json_schema(var_11)
    var_13 = 10
    var_14 = module_0.String(max_length=var_13)
    var_15 = module_1.to_json_schema(var_14)
    var_16 = '^[a-z]+$'
    var_17 = module_3.compile(var_16)
    var_18 = module_0.String(pattern=var_17)
    var_19 = module_1.to_json_schema(var_18)
    var_20 = 'email'
    var_21 = module_0.String(format=var_20)
    var_22 = module_1.to_json_schema(var_21)
    var_23 = module_0.Integer()
    var_24 = module_1.to_json_schema(var_23)
    var_25 = module_0.Integer()
    var_26 = module_1.to_json_schema(var_25)
    var_27 = module_0.Integer(minimum=var_4)
    var_28 = module_1.to_json_schema(var_27)
    var_29 = 100
    var_30 = module_0.Integer(maximum=var_29)
    var_31 = module_1.to_json_schema(var_30)
    var_32 = 95
    var_33 = module_0.Integer(exclusive_minimum=var_10, exclusive_maximum=var_32)
    var_34 = module_1.to_json_schema(var_33)
    var_35 = module_0.Integer(multiple_of=var_10)
    var_36 = module_1.to_json_schema(var_35)
    var_37 = module_0.Float()
    var_38 = module_1.to_json_schema(var_37)
    var_39 = module_0.Float()
    var_40 = module_1.to_json_schema(var_39)
    var_41 = module_0.Boolean()
    var_42 = module_1.to_json_schema(var_41)
    var_43 = module_0.Boolean()
    var_44 = module_1.to_json_schema(var_43)
    var_45 = module_0.String()
    var_46 = module_0.Array(var_45)
    var_47 = module_1.to_json_schema(var_46)
    var_48 = module_0.String()
    var_49 = module_0.Array(var_48)
    var_50 = module_1.to_json_schema(var_49)
    var_51 = module_0.String()
    var_52 = module_0.Array(var_51, min_items=var_7, max_items=var_13)
    var_53 = module_1.to_json_schema(var_52)
    var_54 = module_0.String()
    var_55 = module_0.Array(var_54, unique_items=var_7)
    var_56 = module_1.to_json_schema(var_55)
    var_57 = module_0.String()
    var_58 = module_0.Integer()
    var_59 = [var_57, var_58]
    var_60 = module_0.Array(var_59)
    var_61 = module_1.to_json_schema(var_60)
    var_62 = 'items'
    var_63 = var_61[var_62]
    var_64 = var_61[var_62]
    var_65 = len(var_64)
    assert var_65 == 2
    var_66 = module_0.String()
    var_67 = module_0.Array(var_66, var_4)
    var_68 = module_1.to_json_schema(var_67)
    var_69 = module_0.String()
    var_70 = module_0.Integer()
    var_71 = module_0.Array(var_69, var_70)
    var_72 = module_1.to_json_schema(var_71)
    var_73 = 'additionalItems'
    var_74 = var_72[var_73]
    var_75 = 'name'
    var_76 = module_0.String()
    var_77 = {var_75: var_76}
    var_78 = module_0.Object(properties=var_77)
    var_79 = module_1.to_json_schema(var_78)
    var_80 = module_0.String()
    var_81 = {var_75: var_80}
    var_82 = module_0.Object(properties=var_81)
    var_83 = module_1.to_json_schema(var_82)
    var_84 = '^S_'
    var_85 = '^I_'
    var_86 = module_0.String()
    var_87 = module_0.Integer()
    var_88 = {var_84: var_86, var_85: var_87}
    var_89 = module_0.Object(pattern_properties=var_88)
    var_90 = module_1.to_json_schema(var_89)
    var_91 = module_0.String()
    var_92 = {var_75: var_91}
    var_93 = module_0.Object(properties=var_92, additional_properties=var_4)
    var_94 = module_1.to_json_schema(var_93)
    var_95 = module_0.String()
    var_96 = {var_75: var_95}
    var_97 = module_0.String()
    var_98 = module_0.Object(properties=var_96, additional_properties=var_97)
    var_99 = module_1.to_json_schema(var_98)
    var_100 = 'additionalProperties'
    var_101 = var_99[var_100]
    var_102 = module_0.String()
    var_103 = {var_75: var_102}
    var_104 = module_3.compile(var_16)
    var_105 = module_0.String(pattern=var_104)
    var_106 = module_0.Object(properties=var_103, property_names=var_105)
    var_107 = module_1.to_json_schema(var_106)
    var_108 = module_0.String()
    var_109 = {var_75: var_108}
    var_110 = module_0.Object(properties=var_109, min_properties=var_7, max_properties=var_10)
    var_111 = module_1.to_json_schema(var_110)
    var_112 = 'age'
    var_113 = module_0.String()
    var_114 = module_0.Integer()
    var_115 = {var_75: var_113, var_112: var_114}
    var_116 = [var_75]
    var_117 = module_0.Object(properties=var_115, required=var_116)
    var_118 = module_1.to_json_schema(var_117)
    var_119 = 'a'
    var_120 = 'Option A'
    var_121 = (var_119, var_120)
    var_122 = 'b'
    var_123 = 'Option B'
    var_124 = (var_122, var_123)
    var_125 = [var_121, var_124]
    var_126 = module_0.Choice(choices=var_125)
    var_127 = module_1.to_json_schema(var_126)
    var_128 = 'fixed_value'
    var_129 = module_0.Const(var_128)
    var_130 = module_1.to_json_schema(var_129)



# Parsed testcases at query #50
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
    var_6 = True
    var_7 = module_0.String()
    var_8 = module_1.to_json_schema(var_7)
    var_9 = 5
    var_10 = 10
    var_11 = '^[a-z]+$'
    var_12 = module_0.String(max_length=var_10, min_length=var_9, pattern=var_11)
    var_13 = module_1.to_json_schema(var_12)
    var_14 = module_0.Integer()
    var_15 = module_1.to_json_schema(var_14)
    var_16 = 0
    var_17 = 100
    var_18 = module_0.Integer(minimum=var_16, maximum=var_17)
    var_19 = module_1.to_json_schema(var_18)
    var_20 = module_0.Float()
    var_21 = module_1.to_json_schema(var_20)
    var_22 = module_0.Boolean()
    var_23 = module_1.to_json_schema(var_22)
    var_24 = module_0.String()
    var_25 = module_0.Array(var_24)
    var_26 = module_1.to_json_schema(var_25)
    var_27 = module_0.Integer()
    var_28 = module_0.Array(var_27, min_items=var_6, max_items=var_10, unique_items=var_6)
    var_29 = module_1.to_json_schema(var_28)
    var_30 = 'name'
    var_31 = 'age'
    var_32 = module_0.String()
    var_33 = module_0.Integer()
    var_34 = {var_30: var_32, var_31: var_33}
    var_35 = module_0.Object(properties=var_34)
    var_36 = module_1.to_json_schema(var_35)
    var_37 = module_0.String()
    var_38 = module_0.Integer()
    var_39 = {var_30: var_37, var_31: var_38}
    var_40 = [var_30]
    var_41 = module_0.Object(properties=var_39, required=var_40)
    var_42 = module_1.to_json_schema(var_41)
    var_43 = 'a'
    var_44 = 'Option A'
    var_45 = (var_43, var_44)
    var_46 = 'b'
    var_47 = 'Option B'
    var_48 = (var_46, var_47)
    var_49 = [var_45, var_48]
    var_50 = module_0.Choice(choices=var_49)
    var_51 = module_1.to_json_schema(var_50)
    var_52 = 'fixed_value'
    var_53 = module_0.Const(var_52)
    var_54 = module_1.to_json_schema(var_53)
    var_55 = module_0.String()
    var_56 = module_0.Integer()
    var_57 = [var_55, var_56]
    var_58 = module_0.Union(var_57)
    var_59 = module_1.to_json_schema(var_58)
    var_60 = 'anyOf'
    var_61 = var_59[var_60]
    var_62 = len(var_61)
    assert var_62 == 2
    var_63 = module_0.String()
    var_64 = module_0.Integer()
    var_65 = [var_63, var_64]
    var_66 = module_2.OneOf(var_65)
    var_67 = module_1.to_json_schema(var_66)
    var_68 = 'oneOf'
    var_69 = var_67[var_68]
    var_70 = len(var_69)
    assert var_70 == 2
    var_71 = module_0.String()
    var_72 = 'key'
    var_73 = module_0.String()
    var_74 = {var_72: var_73}
    var_75 = module_0.Object(properties=var_74)
    var_76 = [var_71, var_75]
    var_77 = module_2.AllOf(var_76)
    var_78 = module_1.to_json_schema(var_77)
    var_79 = 'allOf'
    var_80 = var_78[var_79]
    var_81 = len(var_80)
    assert var_81 == 2
    var_82 = module_0.String()
    var_83 = module_2.Not(var_82)
    var_84 = module_1.to_json_schema(var_83)
    var_85 = module_0.String()
    var_86 = module_0.Integer()
    var_87 = module_0.Boolean()
    var_88 = module_2.IfThenElse(var_85, var_86, var_87)
    var_89 = module_1.to_json_schema(var_88)
    var_90 = 'MyType'
    var_91 = module_0.String()
    var_92 = {var_90: var_91}
    var_93 = 'default_value'
    var_94 = module_0.String()
    var_95 = module_1.to_json_schema(var_94)
    var_96 = module_0.String()
    var_97 = module_0.Integer()
    var_98 = [var_96, var_97]
    var_99 = module_0.Array(var_98)
    var_100 = module_1.to_json_schema(var_99)
    var_101 = 'items'
    var_102 = var_100[var_101]
    var_103 = var_100[var_101]
    var_104 = len(var_103)
    assert var_104 == 2
    var_105 = '^S_'
    var_106 = module_0.String()
    var_107 = {var_105: var_106}
    var_108 = module_0.Object(pattern_properties=var_107)
    var_109 = module_1.to_json_schema(var_108)
    var_110 = module_0.String()
    var_111 = module_0.Object(additional_properties=var_110)
    var_112 = module_1.to_json_schema(var_111)
    var_113 = False
    var_114 = module_0.String(allow_blank=var_113)
    var_115 = module_1.to_json_schema(var_114)
    var_116 = module_0.Float(exclusive_minimum=var_113, exclusive_maximum=var_17)
    var_117 = module_1.to_json_schema(var_116)
    var_118 = 'Type1'
    var_119 = 'Type2'
    var_120 = module_0.String()
    var_121 = module_0.Integer()
    var_122 = {var_118: var_120, var_119: var_121}
    var_123 = module_0.String()
    var_124 = False
    var_125 = module_0.Array(var_123, var_124)
    var_126 = module_1.to_json_schema(var_125)
    var_127 = module_0.String()
    var_128 = module_0.Integer()
    var_129 = module_0.Array(var_127, var_128)
    var_130 = module_1.to_json_schema(var_129)
    var_131 = module_0.String(pattern=var_11)
    var_132 = module_0.Object(property_names=var_131)
    var_133 = module_1.to_json_schema(var_132)
    var_134 = module_0.Object(min_properties=var_6, max_properties=var_9)
    var_135 = module_1.to_json_schema(var_134)



# Parsed testcases at query #51
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
    var_4 = False
    var_5 = None
    var_6 = 10
    var_7 = module_0.String(max_length=var_6, min_length=var_5, pattern=var_5, format=var_5)
    var_8 = module_1.to_json_schema(var_7)
    var_9 = True
    var_10 = module_0.String(max_length=var_5, min_length=var_5, pattern=var_5, format=var_5)
    var_11 = module_1.to_json_schema(var_10)
    var_12 = 5
    var_13 = module_0.String(max_length=var_5, min_length=var_12, pattern=var_5, format=var_5)
    var_14 = module_1.to_json_schema(var_13)
    var_15 = module_0.String(allow_blank=var_4, max_length=var_5, min_length=var_5, pattern=var_5, format=var_5)
    var_16 = module_1.to_json_schema(var_15)
    var_17 = 100
    var_18 = module_0.Integer(minimum=var_4, maximum=var_17, exclusive_minimum=var_5, exclusive_maximum=var_5, multiple_of=var_5)
    var_19 = module_1.to_json_schema(var_18)
    var_20 = module_0.Integer(minimum=var_5, maximum=var_5, exclusive_minimum=var_5, exclusive_maximum=var_5, multiple_of=var_5)
    var_21 = module_1.to_json_schema(var_20)
    var_22 = 1.5
    var_23 = 9.5
    var_24 = 0.5
    var_25 = module_0.Float(minimum=var_22, maximum=var_23, exclusive_minimum=var_5, exclusive_maximum=var_5, multiple_of=var_24)
    var_26 = module_1.to_json_schema(var_25)
    var_27 = module_0.Boolean()
    var_28 = module_1.to_json_schema(var_27)
    var_29 = module_0.Boolean()
    var_30 = module_1.to_json_schema(var_29)
    var_31 = module_0.String()
    var_32 = module_0.Array(var_31, min_items=var_9, max_items=var_6, unique_items=var_4)
    var_33 = module_1.to_json_schema(var_32)
    var_34 = module_0.Array(var_5, min_items=var_5, max_items=var_5, unique_items=var_4)
    var_35 = module_1.to_json_schema(var_34)
    var_36 = module_0.Array(var_5, min_items=var_5, max_items=var_5, unique_items=var_9)
    var_37 = module_1.to_json_schema(var_36)
    var_38 = 'name'
    var_39 = module_0.String()
    var_40 = {var_38: var_39}
    var_41 = module_0.Object(properties=var_40, min_properties=var_5, max_properties=var_5, required=var_5)
    var_42 = module_1.to_json_schema(var_41)
    var_43 = module_0.Object(properties=var_5, min_properties=var_5, max_properties=var_5, required=var_5)
    var_44 = module_1.to_json_schema(var_43)
    var_45 = 'id'
    var_46 = [var_45, var_38]
    var_47 = module_0.Object(properties=var_5, min_properties=var_5, max_properties=var_5, required=var_46)
    var_48 = module_1.to_json_schema(var_47)
    var_49 = 'a'
    var_50 = 'Option A'
    var_51 = (var_49, var_50)
    var_52 = 'b'
    var_53 = 'Option B'
    var_54 = (var_52, var_53)
    var_55 = [var_51, var_54]
    var_56 = module_0.Choice(choices=var_55)
    var_57 = module_1.to_json_schema(var_56)
    var_58 = 'fixed_value'
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
    var_77 = module_0.String()
    var_78 = module_0.Object()
    var_79 = [var_77, var_78]
    var_80 = module_2.AllOf(var_79)
    var_81 = module_1.to_json_schema(var_80)
    var_82 = 'allOf'
    var_83 = var_81[var_82]
    var_84 = len(var_83)
    assert var_84 == 2
    var_85 = module_0.String()
    var_86 = module_2.Not(var_85)
    var_87 = module_1.to_json_schema(var_86)
    var_88 = module_0.String()
    var_89 = module_0.Integer()
    var_90 = module_0.Boolean()
    var_91 = module_2.IfThenElse(var_88, var_89, var_90)
    var_92 = module_1.to_json_schema(var_91)
    var_93 = module_0.String()
    var_94 = module_2.IfThenElse(var_93, var_5, var_5)
    var_95 = module_1.to_json_schema(var_94)
    var_96 = 'User'
    var_97 = module_0.String()
    var_98 = {var_38: var_97}
    var_99 = module_0.Object(properties=var_98)
    var_100 = {var_96: var_99}
    var_101 = module_0.String()
    var_102 = {var_38: var_101}
    var_103 = module_0.Object(properties=var_102)
    var_104 = {var_96: var_103}
    var_105 = module_1.to_json_schema(var_0)



# Parsed testcases at query #52
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
    var_14 = 5
    var_15 = module_0.Integer(minimum=var_4, maximum=var_13, multiple_of=var_14)
    var_16 = module_1.to_json_schema(var_15)
    var_17 = module_0.Float(exclusive_minimum=var_4, exclusive_maximum=var_10)
    var_18 = module_1.to_json_schema(var_17)
    var_19 = module_0.Boolean()
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
    var_43 = module_0.Object(properties=var_41, required=var_42)
    var_44 = module_1.to_json_schema(var_43)
    var_45 = '^S_'
    var_46 = module_0.String()
    var_47 = {var_45: var_46}
    var_48 = module_0.Object(pattern_properties=var_47, additional_properties=var_4)
    var_49 = module_1.to_json_schema(var_48)
    var_50 = 'a'
    var_51 = (var_50, var_25)
    var_52 = 'b'
    var_53 = 2
    var_54 = (var_52, var_53)
    var_55 = [var_51, var_54]
    var_56 = module_0.Choice(choices=var_55)
    var_57 = module_1.to_json_schema(var_56)
    var_58 = 'constant_value'
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
    var_77 = module_0.String(min_length=var_25)
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
    var_91 = module_0.Integer()
    var_92 = module_2.IfThenElse(var_90, var_91)
    var_93 = module_1.to_json_schema(var_92)
    var_94 = module_0.String()
    var_95 = module_2.Not(var_94)
    var_96 = module_1.to_json_schema(var_95)
    var_97 = module_0.String()
    var_98 = module_0.Integer()
    var_99 = {var_37: var_97, var_38: var_98}
    var_100 = [var_37]
    var_101 = module_3.Schema(var_99)
    var_102 = module_1.to_json_schema(var_101)
    var_103 = 'StringType'
    var_104 = 'IntType'
    var_105 = module_0.String()
    var_106 = module_0.Integer()
    var_107 = {var_103: var_105, var_104: var_106}
    var_108 = 'MyString'
    var_109 = module_0.String()
    var_110 = {var_108: var_109}
    var_111 = module_0.String()
    var_112 = module_0.Array(var_111, var_4)
    var_113 = module_1.to_json_schema(var_112)
    var_114 = module_0.String()
    var_115 = module_0.Integer()
    var_116 = module_0.Array(var_114, var_115)
    var_117 = module_1.to_json_schema(var_116)
    var_118 = 'additionalItems'
    var_119 = var_117[var_118]
    var_120 = module_0.String(pattern=var_7)
    var_121 = module_0.Object(property_names=var_120)
    var_122 = module_1.to_json_schema(var_121)
    var_123 = module_0.Object(min_properties=var_25, max_properties=var_14)
    var_124 = module_1.to_json_schema(var_123)
    var_125 = 'email'
    var_126 = module_0.String(format=var_125)



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
    var_14 = True
    var_15 = module_0.String(allow_blank=var_14)
    var_16 = module_1.to_json_schema(var_15)
    var_17 = 100
    var_18 = 5
    var_19 = 95
    var_20 = module_0.Integer(minimum=var_4, maximum=var_17, exclusive_minimum=var_18, exclusive_maximum=var_19, multiple_of=var_18)
    var_21 = module_1.to_json_schema(var_20)
    var_22 = module_0.Float()
    var_23 = module_1.to_json_schema(var_22)
    var_24 = module_0.Boolean()
    var_25 = module_1.to_json_schema(var_24)
    var_26 = True
    var_27 = module_0.Boolean()
    var_28 = module_1.to_json_schema(var_27)
    var_29 = module_0.String()
    var_30 = True
    var_31 = module_0.Array(var_29, min_items=var_26, max_items=var_6, unique_items=var_30)
    var_32 = module_1.to_json_schema(var_31)
    var_33 = module_0.String()
    var_34 = module_0.Array(var_33, var_4)
    var_35 = module_1.to_json_schema(var_34)
    var_36 = module_0.String()
    var_37 = module_0.Integer()
    var_38 = [var_36, var_37]
    var_39 = module_0.Array(var_38)
    var_40 = module_1.to_json_schema(var_39)
    var_41 = 'items'
    var_42 = var_40[var_41]
    var_43 = var_40[var_41]
    var_44 = len(var_43)
    assert var_44 == 2
    var_45 = 'name'
    var_46 = 'age'
    var_47 = module_0.String()
    var_48 = module_0.Integer()
    var_49 = {var_45: var_47, var_46: var_48}
    var_50 = [var_45]
    var_51 = module_0.Object(properties=var_49, min_properties=var_30, max_properties=var_6, required=var_50)
    var_52 = module_1.to_json_schema(var_51)
    var_53 = '^S_'
    var_54 = module_0.String()
    var_55 = {var_53: var_54}
    var_56 = module_0.Object(pattern_properties=var_55)
    var_57 = module_1.to_json_schema(var_56)
    var_58 = module_0.Object(additional_properties=var_4)
    var_59 = module_1.to_json_schema(var_58)
    var_60 = module_0.String(pattern=var_7)
    var_61 = module_0.Object(property_names=var_60)
    var_62 = module_1.to_json_schema(var_61)
    var_63 = 'a'
    var_64 = 'Option A'
    var_65 = (var_63, var_64)
    var_66 = 'b'
    var_67 = 'Option B'
    var_68 = (var_66, var_67)
    var_69 = [var_65, var_68]
    var_70 = module_0.Choice(choices=var_69)
    var_71 = module_1.to_json_schema(var_70)
    var_72 = 'constant_value'
    var_73 = module_0.Const(var_72)
    var_74 = module_1.to_json_schema(var_73)
    var_75 = module_0.String()
    var_76 = module_0.Integer()
    var_77 = [var_75, var_76]
    var_78 = module_0.Union(var_77)
    var_79 = module_1.to_json_schema(var_78)
    var_80 = 'anyOf'
    var_81 = var_79[var_80]
    var_82 = len(var_81)
    assert var_82 == 2
    var_83 = module_0.String()
    var_84 = module_0.Integer()
    var_85 = [var_83, var_84]
    var_86 = module_2.OneOf(var_85)
    var_87 = module_1.to_json_schema(var_86)
    var_88 = 'oneOf'
    var_89 = var_87[var_88]
    var_90 = len(var_89)
    assert var_90 == 2
    var_91 = module_0.String()
    var_92 = 'test'
    var_93 = module_0.Const(var_92)
    var_94 = [var_91, var_93]
    var_95 = module_2.AllOf(var_94)
    var_96 = module_1.to_json_schema(var_95)
    var_97 = 'allOf'
    var_98 = var_96[var_97]
    var_99 = len(var_98)
    assert var_99 == 2
    var_100 = module_0.String()
    var_101 = module_2.Not(var_100)
    var_102 = module_1.to_json_schema(var_101)
    var_103 = module_0.Boolean()
    var_104 = module_0.String()
    var_105 = module_0.Integer()
    var_106 = module_2.IfThenElse(var_103, var_104, var_105)
    var_107 = module_1.to_json_schema(var_106)
    var_108 = module_0.Boolean()
    var_109 = module_2.IfThenElse(var_108)
    var_110 = module_1.to_json_schema(var_109)
    var_111 = 'default_value'
    var_112 = module_0.String()
    var_113 = module_1.to_json_schema(var_112)
    var_114 = 'default'
    var_115 = 'MyString'
    var_116 = module_0.String()
    var_117 = {var_115: var_116}
    var_118 = 'MyType'
    var_119 = module_0.String()
    var_120 = {var_118: var_119}
    var_121 = module_0.String()
    var_122 = module_0.Integer()
    var_123 = {var_45: var_121, var_46: var_122}
    var_124 = [var_45]
    var_125 = module_3.Schema(var_123)
    var_126 = module_1.to_json_schema(var_125)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
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
    var_4 = False
    var_5 = module_0.String()
    var_6 = module_1.to_json_schema(var_5)
    var_7 = 'type'
    var_8 = var_6[var_7]
    var_9 = str(var_8)
    var_10 = True
    var_11 = module_0.String()
    var_12 = module_1.to_json_schema(var_11)
    var_13 = 5
    var_14 = 10
    var_15 = module_0.String(max_length=var_14, min_length=var_13)
    var_16 = module_1.to_json_schema(var_15)
    var_17 = '^[a-z]+$'
    var_18 = module_0.String(pattern=var_17)
    var_19 = module_1.to_json_schema(var_18)
    var_20 = 'email'
    var_21 = module_0.String(format=var_20)
    var_22 = module_1.to_json_schema(var_21)
    var_23 = module_0.Integer()
    var_24 = module_1.to_json_schema(var_23)
    var_25 = 100
    var_26 = module_0.Integer(minimum=var_4, maximum=var_25, multiple_of=var_13)
    var_27 = module_1.to_json_schema(var_26)
    var_28 = module_0.Float()
    var_29 = module_1.to_json_schema(var_28)
    var_30 = module_0.Float(exclusive_minimum=var_4, exclusive_maximum=var_25)
    var_31 = module_1.to_json_schema(var_30)
    var_32 = module_0.Boolean()
    var_33 = module_1.to_json_schema(var_32)
    var_34 = module_0.Boolean()
    var_35 = module_1.to_json_schema(var_34)
    var_36 = module_0.String()
    var_37 = module_0.Array(var_36)
    var_38 = module_1.to_json_schema(var_37)
    var_39 = module_0.Array(min_items=var_10, max_items=var_14, unique_items=var_10)
    var_40 = module_1.to_json_schema(var_39)
    var_41 = module_0.String()
    var_42 = module_0.Integer()
    var_43 = [var_41, var_42]
    var_44 = module_0.Array(var_43)
    var_45 = module_1.to_json_schema(var_44)
    var_46 = 'items'
    var_47 = var_45[var_46]
    var_48 = var_45[var_46]
    var_49 = len(var_48)
    assert var_49 == 2
    var_50 = 'name'
    var_51 = 'age'
    var_52 = module_0.String()
    var_53 = module_0.Integer()
    var_54 = {var_50: var_52, var_51: var_53}
    var_55 = module_0.Object(properties=var_54)
    var_56 = module_1.to_json_schema(var_55)
    var_57 = [var_50, var_51]
    var_58 = module_0.Object(required=var_57)
    var_59 = module_1.to_json_schema(var_58)
    var_60 = '^S_'
    var_61 = module_0.String()
    var_62 = {var_60: var_61}
    var_63 = module_0.Object(pattern_properties=var_62)
    var_64 = module_1.to_json_schema(var_63)
    var_65 = module_0.Object(additional_properties=var_4)
    var_66 = module_1.to_json_schema(var_65)
    var_67 = module_0.String()
    var_68 = module_0.Object(additional_properties=var_67)
    var_69 = module_1.to_json_schema(var_68)
    var_70 = 'additionalProperties'
    var_71 = var_69[var_70]
    var_72 = module_0.String(pattern=var_17)
    var_73 = module_0.Object(property_names=var_72)
    var_74 = module_1.to_json_schema(var_73)
    var_75 = 'red'
    var_76 = 'Red'
    var_77 = (var_75, var_76)
    var_78 = 'blue'
    var_79 = 'Blue'
    var_80 = (var_78, var_79)
    var_81 = [var_77, var_80]
    var_82 = module_0.Choice(choices=var_81)
    var_83 = module_1.to_json_schema(var_82)
    var_84 = 'fixed_value'
    var_85 = module_0.Const(var_84)
    var_86 = module_1.to_json_schema(var_85)
    var_87 = module_0.String()
    var_88 = module_0.Integer()
    var_89 = [var_87, var_88]
    var_90 = module_0.Union(var_89)
    var_91 = module_1.to_json_schema(var_90)
    var_92 = 'anyOf'
    var_93 = var_91[var_92]
    var_94 = len(var_93)
    assert var_94 == 2
    var_95 = module_0.String()
    var_96 = module_0.Integer()
    var_97 = [var_95, var_96]
    var_98 = module_2.OneOf(var_97)
    var_99 = module_1.to_json_schema(var_98)
    var_100 = 'oneOf'
    var_101 = var_99[var_100]
    var_102 = len(var_101)
    assert var_102 == 2
    var_103 = module_0.String()
    var_104 = module_0.Object()
    var_105 = [var_103, var_104]
    var_106 = module_2.AllOf(var_105)
    var_107 = module_1.to_json_schema(var_106)
    var_108 = 'allOf'
    var_109 = var_107[var_108]
    var_110 = len(var_109)
    assert var_110 == 2
    var_111 = module_0.String()
    var_112 = module_0.Integer()
    var_113 = module_0.Boolean()
    var_114 = module_2.IfThenElse(var_111, var_112, var_113)
    var_115 = module_1.to_json_schema(var_114)
    var_116 = module_0.String()
    var_117 = module_0.Integer()
    var_118 = module_2.IfThenElse(var_116, var_117)
    var_119 = module_1.to_json_schema(var_118)
    var_120 = module_0.String()
    var_121 = module_2.Not(var_120)
    var_122 = module_1.to_json_schema(var_121)
    var_123 = module_3.Definitions()
    var_124 = module_0.String()
    var_125 = {var_50: var_124}
    var_126 = module_1.to_json_schema(var_123)
    var_127 = module_3.Definitions()
    var_128 = 'User'
    var_129 = module_3.Reference(var_128, var_127)
    var_130 = module_1.to_json_schema(var_129)
    var_131 = module_0.String()
    var_132 = module_0.Integer()
    var_133 = {var_50: var_131, var_51: var_132}
    var_134 = module_3.Schema(var_133)
    var_135 = module_1.to_json_schema(var_134)
    var_136 = 'default_value'
    var_137 = module_0.String()
    var_138 = module_1.to_json_schema(var_137)
    var_139 = 'user'
    var_140 = 'tags'
    var_141 = module_0.String()
    var_142 = module_0.Integer()
    var_143 = {var_50: var_141, var_51: var_142}
    var_144 = module_0.Object(properties=var_143)
    var_145 = module_0.String()
    var_146 = module_0.Array(var_145)
    var_147 = {var_139: var_144, var_140: var_146}
    var_148 = module_0.Object(properties=var_147)
    var_149 = module_1.to_json_schema(var_148)
    var_150 = module_0.String(allow_blank=var_4)
    var_151 = module_1.to_json_schema(var_150)
    var_152 = module_0.Array(additional_items=var_4)
    var_153 = module_1.to_json_schema(var_152)
    var_154 = module_0.String()
    var_155 = module_0.Array(additional_items=var_154)
    var_156 = module_1.to_json_schema(var_155)
    var_157 = 'additionalItems'
    var_158 = var_156[var_157]



# Parsed testcases at query #2
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = '$ref'
    var_1 = '#/components/schemas/User'
    var_2 = {var_0: var_1}
    var_3 = module_0.Definitions()
    var_4 = module_1.ref_from_json_schema(var_2, var_3)
    var_5 = '#/definitions/Address'
    var_6 = {var_0: var_5}
    var_7 = module_0.Definitions()
    var_8 = module_1.ref_from_json_schema(var_6, var_7)
    var_9 = '#/components/schemas/nested/Model'
    var_10 = {var_0: var_9}
    var_11 = module_0.Definitions()
    var_12 = module_1.ref_from_json_schema(var_10, var_11)
    var_13 = 'external.json#/definitions/Model'
    var_14 = {var_0: var_13}
    var_15 = module_0.Definitions()
    var_16 = module_1.ref_from_json_schema(var_14, var_15)
    var_17 = 'definitions/Model'
    var_18 = {var_16: var_17}
    var_19 = module_0.Definitions()
    var_20 = module_1.ref_from_json_schema(var_18, var_19)
    var_21 = '#/definitions/Test'
    var_22 = {var_20: var_21}
    var_23 = module_0.Definitions()
    var_24 = module_1.ref_from_json_schema(var_22, var_23)



# Parsed testcases at query #3
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'Test ref_from_json_schema function.'
    var_1 = module_0.Definitions()
    var_2 = '$ref'
    var_3 = '#/components/schemas/User'
    var_4 = {var_2: var_3}
    var_5 = module_1.ref_from_json_schema(var_4, var_1)
    var_6 = '#/definitions/Address'
    var_7 = {var_2: var_6}
    var_8 = module_1.ref_from_json_schema(var_7, var_1)
    var_9 = '#/components/schemas/models/Product'
    var_10 = {var_2: var_9}
    var_11 = module_1.ref_from_json_schema(var_10, var_1)
    var_12 = 'external.json#/definitions/Item'
    var_13 = {var_2: var_12}
    var_14 = module_1.ref_from_json_schema(var_13, var_1)
    var_15 = 'schemas/User'
    var_16 = {var_2: var_15}
    var_17 = module_1.ref_from_json_schema(var_16, var_1)
    var_18 = module_0.Definitions()
    var_19 = '#/definitions/Custom'
    var_20 = {var_2: var_19}
    var_21 = module_1.ref_from_json_schema(var_20, var_18)



# Parsed testcases at query #4
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'Test one_of_from_json_schema function.'
    var_1 = 'oneOf'
    var_2 = 'type'
    var_3 = 'string'
    var_4 = {var_2: var_3}
    var_5 = 'integer'
    var_6 = {var_2: var_5}
    var_7 = [var_4, var_6]
    var_8 = {var_1: var_7}
    var_9 = module_0.Definitions()
    var_10 = module_1.one_of_from_json_schema(var_8, var_9)
    var_11 = var_10.one_of
    var_12 = len(var_11)
    assert var_12 == 2
    var_13 = 0
    var_14 = var_10.one_of[var_13]
    var_15 = 1
    var_16 = var_10.one_of[var_15]
    var_17 = 'default'
    var_18 = {var_2: var_3}
    var_19 = 'boolean'
    var_20 = {var_2: var_19}
    var_21 = [var_18, var_20]
    var_22 = 'test'
    var_23 = {var_1: var_21, var_17: var_22}
    var_24 = module_1.one_of_from_json_schema(var_23, var_9)
    var_25 = var_24.one_of
    var_26 = len(var_25)
    assert var_26 == 2
    var_27 = 'properties'
    var_28 = 'object'
    var_29 = 'name'
    var_30 = {var_2: var_3}
    var_31 = {var_29: var_30}
    var_32 = {var_2: var_28, var_27: var_31}
    var_33 = 'items'
    var_34 = 'array'
    var_35 = 'number'
    var_36 = {var_2: var_35}
    var_37 = {var_2: var_34, var_33: var_36}
    var_38 = [var_32, var_37]
    var_39 = {var_1: var_38}
    var_40 = module_1.one_of_from_json_schema(var_39, var_9)
    var_41 = var_40.one_of
    var_42 = len(var_41)
    assert var_42 == 2
    var_43 = var_40.one_of[var_13]
    var_44 = var_40.one_of[var_15]
    var_45 = {var_2: var_3}
    var_46 = {var_2: var_35}
    var_47 = [var_45, var_46]
    var_48 = {var_1: var_47}
    var_49 = module_1.one_of_from_json_schema(var_48, var_9)
    var_50 = var_49.one_of
    var_51 = len(var_50)
    assert var_51 == 2
    var_52 = {var_2: var_3}
    var_53 = [var_52]
    var_54 = {var_1: var_53}
    var_55 = module_1.one_of_from_json_schema(var_54, var_9)
    var_56 = var_55.one_of
    var_57 = len(var_56)
    assert var_57 == 1
    var_58 = var_55.one_of[var_13]
    var_59 = module_0.Definitions()
    var_60 = '$ref'
    var_61 = '#/components/schemas/StringType'
    var_62 = {var_60: var_61}
    var_63 = {var_2: var_5}
    var_64 = [var_62, var_63]
    var_65 = {var_1: var_64}
    var_66 = module_1.one_of_from_json_schema(var_65, var_59)
    var_67 = var_66.one_of
    var_68 = len(var_67)
    assert var_68 == 2
    var_69 = var_66.one_of[var_13]
    var_70 = var_66.one_of[var_15]



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'enum'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'red'
    var_7 = 'green'
    var_8 = 'blue'
    var_9 = [var_6, var_7, var_8]
    var_10 = {var_0: var_9}
    var_11 = 'two'
    var_12 = None
    var_13 = True
    var_14 = [var_1, var_11, var_12, var_13]
    var_15 = {var_0: var_14}
    var_16 = 'default'
    var_17 = 'a'
    var_18 = 'b'
    var_19 = 'c'
    var_20 = [var_17, var_18, var_19]
    var_21 = {var_0: var_20, var_16: var_18}
    var_22 = 42
    var_23 = [var_22]
    var_24 = {var_0: var_23}
    var_25 = True
    var_26 = False
    var_27 = [var_25, var_26]
    var_28 = {var_0: var_27}
    var_29 = 'value'
    var_30 = [var_12, var_29]
    var_31 = {var_0: var_30}



# Parsed testcases at query #6
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
    var_11 = 0
    var_12 = 1
    var_13 = 'default'
    var_14 = 'minimum'
    var_15 = 'integer'
    var_16 = {var_1: var_15, var_14: var_11}
    var_17 = 'maximum'
    var_18 = 100
    var_19 = {var_1: var_15, var_17: var_18}
    var_20 = [var_16, var_19]
    var_21 = 50
    var_22 = {var_0: var_20, var_13: var_21}
    var_23 = 'boolean'
    var_24 = {var_1: var_23}
    var_25 = [var_24]
    var_26 = {var_0: var_25}
    var_27 = 'properties'
    var_28 = 'object'
    var_29 = 'name'
    var_30 = {var_1: var_3}
    var_31 = {var_29: var_30}
    var_32 = {var_1: var_28, var_27: var_31}
    var_33 = 'age'
    var_34 = {var_1: var_15}
    var_35 = {var_33: var_34}
    var_36 = {var_1: var_28, var_27: var_35}
    var_37 = [var_32, var_36]
    var_38 = {var_0: var_37}
    var_39 = module_0.Definitions()
    var_40 = '$ref'
    var_41 = '#/definitions/StringField'
    var_42 = {var_40: var_41}
    var_43 = {var_1: var_3, var_6: var_21}
    var_44 = [var_42, var_43]
    var_45 = {var_0: var_44}
    var_46 = module_1.all_of_from_json_schema(var_45, var_39)
    var_47 = var_46.all_of
    var_48 = len(var_47)
    assert var_48 == 2
    var_49 = var_46.all_of[var_11]
    var_50 = var_46.all_of[var_12]



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'allOf'
    var_1 = 'type'
    var_2 = 'minLength'
    var_3 = 'string'
    var_4 = 1
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'maxLength'
    var_7 = 10
    var_8 = {var_1: var_3, var_6: var_7}
    var_9 = [var_5, var_8]
    var_10 = {var_0: var_9}
    var_11 = 0
    var_12 = 'default'
    var_13 = 'minimum'
    var_14 = 'integer'
    var_15 = {var_1: var_14, var_13: var_11}
    var_16 = 'maximum'
    var_17 = 100
    var_18 = {var_1: var_14, var_16: var_17}
    var_19 = [var_15, var_18]
    var_20 = 50
    var_21 = {var_0: var_19, var_12: var_20}
    var_22 = 'boolean'
    var_23 = {var_1: var_22}
    var_24 = 'const'
    var_25 = True
    var_26 = {var_24: var_25}
    var_27 = [var_23, var_26]
    var_28 = {var_0: var_27}
    var_29 = 'properties'
    var_30 = 'object'
    var_31 = 'name'
    var_32 = {var_1: var_3}
    var_33 = {var_31: var_32}
    var_34 = {var_1: var_30, var_29: var_33}
    var_35 = 'required'
    var_36 = [var_31]
    var_37 = {var_1: var_30, var_35: var_36}
    var_38 = [var_34, var_37]
    var_39 = {var_0: var_38}
    var_40 = {var_1: var_3}
    var_41 = [var_40]
    var_42 = {var_0: var_41}
    var_43 = []
    var_44 = {var_0: var_43}



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = 'integer'
    var_4 = {var_0: var_3}
    var_5 = 'number'
    var_6 = {var_0: var_5}
    var_7 = 'boolean'
    var_8 = {var_0: var_7}
    var_9 = 'array'
    var_10 = {var_0: var_9}
    var_11 = 'object'
    var_12 = {var_0: var_11}
    var_13 = [var_1, var_3]
    var_14 = {var_0: var_13}
    var_15 = 'null'
    var_16 = [var_1, var_15]
    var_17 = {var_0: var_16}
    var_18 = {var_0: var_15}
    var_19 = {}
    var_20 = 'minLength'
    var_21 = 5
    var_22 = {var_0: var_1, var_20: var_21}
    var_23 = 'minimum'
    var_24 = 10
    var_25 = {var_0: var_5, var_23: var_24}
    var_26 = [var_1, var_3, var_15]
    var_27 = {var_0: var_26}
    var_28 = 'items'
    var_29 = {var_0: var_1}
    var_30 = {var_0: var_9, var_28: var_29}
    var_31 = 'properties'
    var_32 = 'name'
    var_33 = {var_0: var_1}
    var_34 = {var_32: var_33}
    var_35 = {var_0: var_11, var_31: var_34}



# Parsed testcases at query #9
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
    var_30 = 'fixed_value'
    var_31 = {var_29: var_30}
    var_32 = module_0.from_json_schema(var_31)
    var_33 = 'minLength'
    var_34 = 'maxLength'
    var_35 = 'pattern'
    var_36 = 10
    var_37 = '^[a-z]+$'
    var_38 = {var_4: var_5, var_33: var_0, var_34: var_36, var_35: var_37}
    var_39 = module_0.from_json_schema(var_38)
    var_40 = 'minimum'
    var_41 = 'maximum'
    var_42 = 'multipleOf'
    var_43 = 100
    var_44 = 5
    var_45 = {var_4: var_11, var_40: var_2, var_41: var_43, var_42: var_44}
    var_46 = module_0.from_json_schema(var_45)
    var_47 = 'items'
    var_48 = 'minItems'
    var_49 = 'maxItems'
    var_50 = 'uniqueItems'
    var_51 = {var_4: var_5}
    var_52 = {var_4: var_17, var_47: var_51, var_48: var_0, var_49: var_36, var_50: var_0}
    var_53 = module_0.from_json_schema(var_52)
    var_54 = 'properties'
    var_55 = 'required'
    var_56 = 'name'
    var_57 = 'age'
    var_58 = {var_4: var_5}
    var_59 = {var_4: var_8}
    var_60 = {var_56: var_58, var_57: var_59}
    var_61 = [var_56]
    var_62 = {var_4: var_20, var_54: var_60, var_55: var_61}
    var_63 = module_0.from_json_schema(var_62)
    var_64 = 'allOf'
    var_65 = {var_4: var_5}
    var_66 = {var_33: var_44}
    var_67 = [var_65, var_66]
    var_68 = {var_64: var_67}
    var_69 = module_0.from_json_schema(var_68)
    var_70 = 'anyOf'
    var_71 = {var_4: var_5}
    var_72 = {var_4: var_8}
    var_73 = [var_71, var_72]
    var_74 = {var_70: var_73}
    var_75 = module_0.from_json_schema(var_74)
    var_76 = 'oneOf'
    var_77 = {var_4: var_5}
    var_78 = {var_4: var_8}
    var_79 = [var_77, var_78]
    var_80 = {var_76: var_79}
    var_81 = module_0.from_json_schema(var_80)
    var_82 = 'not'
    var_83 = 'null'
    var_84 = {var_4: var_83}
    var_85 = {var_82: var_84}
    var_86 = module_0.from_json_schema(var_85)
    var_87 = 'if'
    var_88 = 'then'
    var_89 = 'else'
    var_90 = {var_4: var_5}
    var_91 = {var_33: var_44}
    var_92 = {var_4: var_8}
    var_93 = {var_87: var_90, var_88: var_91, var_89: var_92}
    var_94 = module_0.from_json_schema(var_93)
    var_95 = 'red'
    var_96 = 'green'
    var_97 = 'blue'
    var_98 = [var_95, var_96, var_97]
    var_99 = {var_4: var_5, var_23: var_98, var_33: var_25}
    var_100 = module_0.from_json_schema(var_99)
    var_101 = module_1.Definitions()
    var_102 = '$ref'
    var_103 = '#/components/schemas/StringType'
    var_104 = {var_102: var_103}
    var_105 = module_0.from_json_schema(var_104, var_101)
    var_106 = 'components'
    var_107 = 'schemas'
    var_108 = 'MyString'
    var_109 = {var_4: var_5}
    var_110 = {var_108: var_109}
    var_111 = {var_107: var_110}
    var_112 = {var_4: var_20, var_106: var_111}
    var_113 = module_0.from_json_schema(var_112)
    var_114 = {}
    var_115 = module_0.from_json_schema(var_114)
    var_116 = 'additionalProperties'
    var_117 = {var_4: var_5}
    var_118 = {var_4: var_20, var_116: var_117}
    var_119 = module_0.from_json_schema(var_118)
    var_120 = 'patternProperties'
    var_121 = {var_4: var_5}
    var_122 = {var_37: var_121}
    var_123 = {var_4: var_20, var_120: var_122}
    var_124 = module_0.from_json_schema(var_123)



# Parsed testcases at query #10
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
    var_15 = 'default'
    var_16 = 'minimum'
    var_17 = 'integer'
    var_18 = 0
    var_19 = {var_1: var_17, var_16: var_18}
    var_20 = 'maximum'
    var_21 = 100
    var_22 = {var_1: var_17, var_20: var_21}
    var_23 = [var_19, var_22]
    var_24 = 50
    var_25 = {var_0: var_23, var_15: var_24}
    var_26 = module_0.Definitions()
    var_27 = module_1.all_of_from_json_schema(var_25, var_26)
    var_28 = 'boolean'
    var_29 = {var_1: var_28}
    var_30 = 'const'
    var_31 = True
    var_32 = {var_30: var_31}
    var_33 = [var_29, var_32]
    var_34 = {var_0: var_33}
    var_35 = module_0.Definitions()
    var_36 = module_1.all_of_from_json_schema(var_34, var_35)
    var_37 = 'number'
    var_38 = {var_1: var_37, var_16: var_18}
    var_39 = [var_38]
    var_40 = {var_0: var_39}
    var_41 = module_0.Definitions()
    var_42 = module_1.all_of_from_json_schema(var_40, var_41)
    var_43 = var_42.all_of
    var_44 = len(var_43)
    assert var_44 == 1
    var_45 = module_0.Definitions()
    var_46 = '$ref'
    var_47 = '#/definitions/StringField'
    var_48 = {var_46: var_47}
    var_49 = {var_2: var_31}
    var_50 = [var_48, var_49]
    var_51 = {var_0: var_50}
    var_52 = module_1.all_of_from_json_schema(var_51, var_45)
    var_53 = var_52.all_of
    var_54 = len(var_53)
    assert var_54 == 2
    var_55 = 'properties'
    var_56 = 'object'
    var_57 = 'name'
    var_58 = {var_1: var_3}
    var_59 = {var_57: var_58}
    var_60 = {var_1: var_56, var_55: var_59}
    var_61 = 'required'
    var_62 = [var_57]
    var_63 = {var_61: var_62}
    var_64 = [var_60, var_63]
    var_65 = {var_0: var_64}
    var_66 = module_0.Definitions()
    var_67 = module_1.all_of_from_json_schema(var_65, var_66)
    var_68 = var_67.all_of
    var_69 = len(var_68)
    assert var_69 == 2



# Parsed testcases at query #11
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
    var_8 = {var_6: var_7}
    var_9 = 'number'
    var_10 = {var_3: var_9}
    var_11 = {var_0: var_5, var_1: var_8, var_2: var_10}
    var_12 = 'object'
    var_13 = {var_3: var_12}
    var_14 = 'minProperties'
    var_15 = 1
    var_16 = {var_14: var_15}
    var_17 = {var_0: var_13, var_1: var_16}
    var_18 = 'array'
    var_19 = {var_3: var_18}
    var_20 = 'maxItems'
    var_21 = 10
    var_22 = {var_20: var_21}
    var_23 = {var_0: var_19, var_2: var_22}
    var_24 = 'default'
    var_25 = 'boolean'
    var_26 = {var_3: var_25}
    var_27 = 'const'
    var_28 = True
    var_29 = {var_27: var_28}
    var_30 = False
    var_31 = {var_27: var_30}
    var_32 = {var_0: var_26, var_1: var_29, var_2: var_31, var_24: var_30}
    var_33 = 'properties'
    var_34 = {var_27: var_4}
    var_35 = {var_3: var_34}
    var_36 = {var_33: var_35}
    var_37 = 'value'
    var_38 = {var_3: var_4}
    var_39 = {var_37: var_38}
    var_40 = {var_33: var_39}
    var_41 = {var_3: var_9}
    var_42 = {var_37: var_41}
    var_43 = {var_33: var_42}
    var_44 = {var_0: var_36, var_1: var_40, var_2: var_43}
    var_45 = 'integer'
    var_46 = {var_3: var_45}
    var_47 = {var_0: var_46}



# Parsed testcases at query #12
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'Test from_json_schema_type function with various type strings.'
    var_1 = module_0.Definitions()
    var_2 = 'minimum'
    var_3 = 'maximum'
    var_4 = 'multipleOf'
    var_5 = 0
    var_6 = 100
    var_7 = 5
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = 'number'
    var_10 = False
    var_11 = module_1.from_json_schema_type(var_8, var_9, var_10, var_1)
    var_12 = -10
    var_13 = 10
    var_14 = {var_2: var_12, var_3: var_13}
    var_15 = True
    var_16 = module_1.from_json_schema_type(var_14, var_9, var_15, var_1)
    var_17 = {var_2: var_15, var_3: var_13}
    var_18 = 'integer'
    var_19 = False
    var_20 = module_1.from_json_schema_type(var_17, var_18, var_19, var_1)
    var_21 = 'exclusiveMinimum'
    var_22 = 'exclusiveMaximum'
    var_23 = {var_21: var_19, var_22: var_6}
    var_24 = False
    var_25 = module_1.from_json_schema_type(var_23, var_18, var_24, var_1)
    var_26 = 'minLength'
    var_27 = 'maxLength'
    var_28 = 'pattern'
    var_29 = 50
    var_30 = '^[a-z]+$'
    var_31 = {var_26: var_15, var_27: var_29, var_28: var_30}
    var_32 = 'string'
    var_33 = False
    var_34 = module_1.from_json_schema_type(var_31, var_32, var_33, var_1)
    var_35 = {var_26: var_33}
    var_36 = False
    var_37 = module_1.from_json_schema_type(var_35, var_32, var_36, var_1)
    var_38 = 'format'
    var_39 = 'email'
    var_40 = {var_38: var_39}
    var_41 = False
    var_42 = module_1.from_json_schema_type(var_40, var_32, var_41, var_1)
    var_43 = {}
    var_44 = 'boolean'
    var_45 = False
    var_46 = module_1.from_json_schema_type(var_43, var_44, var_45, var_1)
    var_47 = {}
    var_48 = module_1.from_json_schema_type(var_47, var_44, var_15, var_1)
    var_49 = 'minItems'
    var_50 = 'maxItems'
    var_51 = 'uniqueItems'
    var_52 = {var_49: var_15, var_50: var_13, var_51: var_15}
    var_53 = 'array'
    var_54 = False
    var_55 = module_1.from_json_schema_type(var_52, var_53, var_54, var_1)
    var_56 = 'items'
    var_57 = 'type'
    var_58 = {var_57: var_32}
    var_59 = {var_56: var_58}
    var_60 = False
    var_61 = module_1.from_json_schema_type(var_59, var_53, var_60, var_1)
    var_62 = var_61.items
    var_63 = {var_57: var_32}
    var_64 = {var_57: var_18}
    var_65 = [var_63, var_64]
    var_66 = {var_56: var_65}
    var_67 = False
    var_68 = module_1.from_json_schema_type(var_66, var_53, var_67, var_1)
    var_69 = var_68.items
    var_70 = var_68.items
    var_71 = len(var_70)
    assert var_71 == 2
    var_72 = 'additionalItems'
    var_73 = {var_57: var_9}
    var_74 = {var_72: var_73}
    var_75 = False
    var_76 = module_1.from_json_schema_type(var_74, var_53, var_75, var_1)
    var_77 = var_76.additional_items
    var_78 = 'properties'
    var_79 = 'required'
    var_80 = 'name'
    var_81 = {var_57: var_32}
    var_82 = {var_80: var_81}
    var_83 = [var_80]
    var_84 = {var_78: var_82, var_79: var_83}
    var_85 = 'object'
    var_86 = False
    var_87 = module_1.from_json_schema_type(var_84, var_85, var_86, var_1)
    var_88 = var_87.properties[var_80]
    var_89 = 'patternProperties'
    var_90 = '^S_'
    var_91 = {var_57: var_32}
    var_92 = {var_90: var_91}
    var_93 = {var_89: var_92}
    var_94 = False
    var_95 = module_1.from_json_schema_type(var_93, var_85, var_94, var_1)
    var_96 = 'additionalProperties'
    var_97 = False
    var_98 = {var_96: var_97}
    var_99 = False
    var_100 = module_1.from_json_schema_type(var_98, var_85, var_99, var_1)
    var_101 = {var_57: var_32}
    var_102 = {var_96: var_101}
    var_103 = False
    var_104 = module_1.from_json_schema_type(var_102, var_85, var_103, var_1)
    var_105 = var_104.additional_properties
    var_106 = 'propertyNames'
    var_107 = {var_28: var_30}
    var_108 = {var_106: var_107}
    var_109 = False
    var_110 = module_1.from_json_schema_type(var_108, var_85, var_109, var_1)
    var_111 = var_110.property_names
    var_112 = 'minProperties'
    var_113 = 'maxProperties'
    var_114 = {var_112: var_15, var_113: var_7}
    var_115 = False
    var_116 = module_1.from_json_schema_type(var_114, var_85, var_115, var_1)
    var_117 = 'default'
    var_118 = 42
    var_119 = {var_117: var_118}
    var_120 = False
    var_121 = module_1.from_json_schema_type(var_119, var_18, var_120, var_1)



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'Test if_then_else_from_json_schema function.'
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'else'
    var_4 = 'type'
    var_5 = 'string'
    var_6 = {var_4: var_5}
    var_7 = 'minLength'
    var_8 = 5
    var_9 = {var_7: var_8}
    var_10 = 'number'
    var_11 = {var_4: var_10}
    var_12 = {var_1: var_6, var_2: var_9, var_3: var_11}
    var_13 = {var_4: var_5}
    var_14 = {var_7: var_8}
    var_15 = {var_1: var_13, var_2: var_14}
    var_16 = {var_4: var_5}
    var_17 = {var_4: var_10}
    var_18 = {var_1: var_16, var_3: var_17}
    var_19 = {var_4: var_5}
    var_20 = {var_1: var_19}
    var_21 = 'default'
    var_22 = {var_4: var_5}
    var_23 = {var_7: var_8}
    var_24 = {var_4: var_10}
    var_25 = 'test_default'
    var_26 = {var_1: var_22, var_2: var_23, var_3: var_24, var_21: var_25}
    var_27 = 'properties'
    var_28 = 'object'
    var_29 = 'name'
    var_30 = {var_4: var_5}
    var_31 = {var_29: var_30}
    var_32 = {var_4: var_28, var_27: var_31}
    var_33 = 'age'
    var_34 = 'integer'
    var_35 = {var_4: var_34}
    var_36 = {var_33: var_35}
    var_37 = {var_27: var_36}
    var_38 = 'items'
    var_39 = 'array'
    var_40 = {var_4: var_5}
    var_41 = {var_4: var_39, var_38: var_40}
    var_42 = {var_1: var_32, var_2: var_37, var_3: var_41}



# Parsed testcases at query #14
#--------------------------


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
    var_7 = 100
    var_8 = 10
    var_9 = 90
    var_10 = 5
    var_11 = 50
    var_12 = {var_0: var_6, var_1: var_7, var_2: var_8, var_3: var_9, var_4: var_10, var_5: var_11}
    var_13 = 'number'
    var_14 = False
    var_15 = module_0.Definitions()
    var_16 = module_1.from_json_schema_type(var_12, var_13, var_14, var_15)
    var_17 = {var_0: var_14}
    var_18 = True
    var_19 = module_0.Definitions()
    var_20 = module_1.from_json_schema_type(var_17, var_13, var_18, var_19)
    var_21 = 11
    var_22 = 2
    var_23 = {var_0: var_18, var_1: var_8, var_2: var_14, var_3: var_21, var_4: var_22, var_5: var_10}
    var_24 = 'integer'
    var_25 = False
    var_26 = module_0.Definitions()
    var_27 = module_1.from_json_schema_type(var_23, var_24, var_25, var_26)
    var_28 = 'minLength'
    var_29 = 'maxLength'
    var_30 = 'pattern'
    var_31 = 'format'
    var_32 = 20
    var_33 = '^[a-z]+$'
    var_34 = 'email'
    var_35 = 'test'
    var_36 = {var_28: var_10, var_29: var_32, var_30: var_33, var_31: var_34, var_5: var_35}
    var_37 = 'string'
    var_38 = False
    var_39 = module_0.Definitions()
    var_40 = module_1.from_json_schema_type(var_36, var_37, var_38, var_39)
    var_41 = {var_28: var_38, var_29: var_8}
    var_42 = False
    var_43 = module_0.Definitions()
    var_44 = module_1.from_json_schema_type(var_41, var_37, var_42, var_43)
    var_45 = {var_28: var_18}
    var_46 = False
    var_47 = module_0.Definitions()
    var_48 = module_1.from_json_schema_type(var_45, var_37, var_46, var_47)
    var_49 = {var_5: var_18}
    var_50 = 'boolean'
    var_51 = False
    var_52 = module_0.Definitions()
    var_53 = module_1.from_json_schema_type(var_49, var_50, var_51, var_52)
    var_54 = {}
    var_55 = module_0.Definitions()
    var_56 = module_1.from_json_schema_type(var_54, var_50, var_18, var_55)
    var_57 = 'minItems'
    var_58 = 'maxItems'
    var_59 = 'uniqueItems'
    var_60 = {var_57: var_18, var_58: var_8, var_59: var_18}
    var_61 = 'array'
    var_62 = False
    var_63 = module_0.Definitions()
    var_64 = module_1.from_json_schema_type(var_60, var_61, var_62, var_63)
    var_65 = 'items'
    var_66 = 'type'
    var_67 = {var_66: var_37}
    var_68 = {var_65: var_67}
    var_69 = False
    var_70 = module_0.Definitions()
    var_71 = module_1.from_json_schema_type(var_68, var_61, var_69, var_70)
    var_72 = var_71.items
    var_73 = {var_66: var_37}
    var_74 = {var_66: var_24}
    var_75 = [var_73, var_74]
    var_76 = {var_65: var_75}
    var_77 = False
    var_78 = module_0.Definitions()
    var_79 = module_1.from_json_schema_type(var_76, var_61, var_77, var_78)
    var_80 = var_79.items
    var_81 = var_79.items
    var_82 = len(var_81)
    assert var_82 == 2
    var_83 = 'additionalItems'
    var_84 = False
    var_85 = {var_83: var_84}
    var_86 = False
    var_87 = module_0.Definitions()
    var_88 = module_1.from_json_schema_type(var_85, var_61, var_86, var_87)
    var_89 = {var_66: var_13}
    var_90 = {var_83: var_89}
    var_91 = False
    var_92 = module_0.Definitions()
    var_93 = module_1.from_json_schema_type(var_90, var_61, var_91, var_92)
    var_94 = var_93.additional_items
    var_95 = {}
    var_96 = 'object'
    var_97 = False
    var_98 = module_0.Definitions()
    var_99 = module_1.from_json_schema_type(var_95, var_96, var_97, var_98)
    var_100 = 'properties'
    var_101 = 'name'
    var_102 = 'age'
    var_103 = {var_66: var_37}
    var_104 = {var_66: var_24}
    var_105 = {var_101: var_103, var_102: var_104}
    var_106 = {var_100: var_105}
    var_107 = False
    var_108 = module_0.Definitions()
    var_109 = module_1.from_json_schema_type(var_106, var_96, var_107, var_108)
    var_110 = var_109.properties
    var_111 = 'patternProperties'
    var_112 = '^S_'
    var_113 = {var_66: var_37}
    var_114 = {var_112: var_113}
    var_115 = {var_111: var_114}
    var_116 = False
    var_117 = module_0.Definitions()
    var_118 = module_1.from_json_schema_type(var_115, var_96, var_116, var_117)
    var_119 = var_118.pattern_properties
    var_120 = 'additionalProperties'
    var_121 = False
    var_122 = {var_120: var_121}
    var_123 = False
    var_124 = module_0.Definitions()
    var_125 = module_1.from_json_schema_type(var_122, var_96, var_123, var_124)
    var_126 = {var_66: var_37}
    var_127 = {var_120: var_126}
    var_128 = False
    var_129 = module_0.Definitions()
    var_130 = module_1.from_json_schema_type(var_127, var_96, var_128, var_129)
    var_131 = var_130.additional_properties
    var_132 = 'propertyNames'
    var_133 = {var_30: var_33}
    var_134 = {var_132: var_133}
    var_135 = False
    var_136 = module_0.Definitions()
    var_137 = module_1.from_json_schema_type(var_134, var_96, var_135, var_136)
    var_138 = var_137.property_names
    var_139 = 'required'
    var_140 = 'minProperties'
    var_141 = 'maxProperties'
    var_142 = [var_101]
    var_143 = {var_139: var_142, var_140: var_18, var_141: var_10}
    var_144 = False
    var_145 = module_0.Definitions()
    var_146 = module_1.from_json_schema_type(var_143, var_96, var_144, var_145)



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'Test if_then_else_from_json_schema function.'
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 'minLength'
    var_7 = 5
    var_8 = {var_6: var_7}
    var_9 = {var_1: var_5, var_2: var_8}
    var_10 = 'else'
    var_11 = 'integer'
    var_12 = {var_3: var_11}
    var_13 = 'minimum'
    var_14 = 0
    var_15 = {var_13: var_14}
    var_16 = 'maximum'
    var_17 = {var_16: var_14}
    var_18 = {var_1: var_12, var_2: var_15, var_10: var_17}
    var_19 = 'boolean'
    var_20 = {var_3: var_19}
    var_21 = {var_1: var_20}
    var_22 = 'default'
    var_23 = {var_3: var_4}
    var_24 = 1
    var_25 = {var_6: var_24}
    var_26 = 'test'
    var_27 = {var_1: var_23, var_2: var_25, var_22: var_26}
    var_28 = 'properties'
    var_29 = 'enum'
    var_30 = 'A'
    var_31 = [var_30]
    var_32 = {var_29: var_31}
    var_33 = {var_3: var_32}
    var_34 = {var_28: var_33}
    var_35 = 'required'
    var_36 = 'fieldA'
    var_37 = [var_36]
    var_38 = {var_35: var_37}
    var_39 = 'fieldB'
    var_40 = [var_39]
    var_41 = {var_35: var_40}
    var_42 = {var_1: var_34, var_2: var_38, var_10: var_41}



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'if'
    var_1 = 'then'
    var_2 = 'else'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 'integer'
    var_7 = {var_3: var_6}
    var_8 = 'boolean'
    var_9 = {var_3: var_8}
    var_10 = {var_0: var_5, var_1: var_7, var_2: var_9}
    var_11 = {var_3: var_4}
    var_12 = {var_3: var_6}
    var_13 = {var_0: var_11, var_1: var_12}
    var_14 = {var_3: var_4}
    var_15 = {var_3: var_8}
    var_16 = {var_0: var_14, var_2: var_15}
    var_17 = {var_3: var_4}
    var_18 = {var_0: var_17}
    var_19 = 'default'
    var_20 = {var_3: var_4}
    var_21 = {var_3: var_6}
    var_22 = 42
    var_23 = {var_0: var_20, var_1: var_21, var_19: var_22}
    var_24 = 'properties'
    var_25 = 'object'
    var_26 = 'name'
    var_27 = {var_3: var_4}
    var_28 = {var_26: var_27}
    var_29 = {var_3: var_25, var_24: var_28}
    var_30 = 'items'
    var_31 = 'array'
    var_32 = {var_3: var_6}
    var_33 = {var_3: var_31, var_30: var_32}
    var_34 = 'enum'
    var_35 = None
    var_36 = 'unknown'
    var_37 = [var_35, var_36]
    var_38 = {var_34: var_37}
    var_39 = {var_0: var_29, var_1: var_33, var_2: var_38}



# Parsed testcases at query #17
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'Test if_then_else_from_json_schema with various configurations.'
    var_1 = module_0.Definitions()
    var_2 = 'if'
    var_3 = 'then'
    var_4 = 'else'
    var_5 = 'type'
    var_6 = 'string'
    var_7 = {var_5: var_6}
    var_8 = 'integer'
    var_9 = {var_5: var_8}
    var_10 = 'boolean'
    var_11 = {var_5: var_10}
    var_12 = {var_2: var_7, var_3: var_9, var_4: var_11}
    var_13 = module_1.if_then_else_from_json_schema(var_12, var_1)
    var_14 = {var_5: var_6}
    var_15 = {var_2: var_14}
    var_16 = module_1.if_then_else_from_json_schema(var_15, var_1)
    var_17 = 'array'
    var_18 = {var_5: var_17}
    var_19 = 'object'
    var_20 = {var_5: var_19}
    var_21 = {var_2: var_18, var_3: var_20}
    var_22 = module_1.if_then_else_from_json_schema(var_21, var_1)
    var_23 = 'number'
    var_24 = {var_5: var_23}
    var_25 = {var_5: var_6}
    var_26 = {var_2: var_24, var_4: var_25}
    var_27 = module_1.if_then_else_from_json_schema(var_26, var_1)
    var_28 = 'default'
    var_29 = {var_5: var_10}
    var_30 = {var_5: var_6}
    var_31 = {var_5: var_8}
    var_32 = 'test_default'
    var_33 = {var_2: var_29, var_3: var_30, var_4: var_31, var_28: var_32}
    var_34 = module_1.if_then_else_from_json_schema(var_33, var_1)
    var_35 = 'properties'
    var_36 = 'name'
    var_37 = {var_5: var_6}
    var_38 = {var_36: var_37}
    var_39 = {var_5: var_19, var_35: var_38}
    var_40 = 'items'
    var_41 = {var_5: var_6}
    var_42 = {var_5: var_17, var_40: var_41}
    var_43 = 'enum'
    var_44 = 1
    var_45 = 2
    var_46 = 3
    var_47 = [var_44, var_45, var_46]
    var_48 = {var_43: var_47}
    var_49 = {var_2: var_39, var_3: var_42, var_4: var_48}
    var_50 = module_1.if_then_else_from_json_schema(var_49, var_1)



# Parsed testcases at query #18
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'Test if_then_else_from_json_schema function.'
    var_1 = module_0.Definitions()
    var_2 = 'if'
    var_3 = 'then'
    var_4 = 'else'
    var_5 = 'type'
    var_6 = 'string'
    var_7 = {var_5: var_6}
    var_8 = 'integer'
    var_9 = {var_5: var_8}
    var_10 = 'boolean'
    var_11 = {var_5: var_10}
    var_12 = {var_2: var_7, var_3: var_9, var_4: var_11}
    var_13 = module_1.if_then_else_from_json_schema(var_12, var_1)
    var_14 = var_13.if_clause
    var_15 = var_13.then_clause
    var_16 = var_13.else_clause
    var_17 = {var_5: var_6}
    var_18 = {var_5: var_8}
    var_19 = {var_2: var_17, var_3: var_18}
    var_20 = module_1.if_then_else_from_json_schema(var_19, var_1)
    var_21 = var_20.if_clause
    var_22 = var_20.then_clause
    var_23 = {var_5: var_6}
    var_24 = {var_5: var_10}
    var_25 = {var_2: var_23, var_4: var_24}
    var_26 = module_1.if_then_else_from_json_schema(var_25, var_1)
    var_27 = var_26.if_clause
    var_28 = var_26.else_clause
    var_29 = {var_5: var_6}
    var_30 = {var_2: var_29}
    var_31 = module_1.if_then_else_from_json_schema(var_30, var_1)
    var_32 = var_31.if_clause
    var_33 = 'default'
    var_34 = {var_5: var_6}
    var_35 = {var_5: var_8}
    var_36 = {var_5: var_10}
    var_37 = 'test_default'
    var_38 = {var_2: var_34, var_3: var_35, var_4: var_36, var_33: var_37}
    var_39 = module_1.if_then_else_from_json_schema(var_38, var_1)
    var_40 = 'properties'
    var_41 = 'object'
    var_42 = 'name'
    var_43 = {var_5: var_6}
    var_44 = {var_42: var_43}
    var_45 = {var_5: var_41, var_40: var_44}
    var_46 = 'items'
    var_47 = 'array'
    var_48 = {var_5: var_8}
    var_49 = {var_5: var_47, var_46: var_48}
    var_50 = 'number'
    var_51 = {var_5: var_50}
    var_52 = {var_2: var_45, var_3: var_49, var_4: var_51}
    var_53 = module_1.if_then_else_from_json_schema(var_52, var_1)
    var_54 = var_53.if_clause
    var_55 = var_53.then_clause
    var_56 = var_53.else_clause



# Parsed testcases at query #19
#--------------------------


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
    var_9 = 'else'
    var_10 = 'number'
    var_11 = {var_2: var_10}
    var_12 = 'minimum'
    var_13 = 0
    var_14 = {var_12: var_13}
    var_15 = 'maximum'
    var_16 = {var_15: var_13}
    var_17 = {var_0: var_11, var_1: var_14, var_9: var_16}
    var_18 = 'boolean'
    var_19 = {var_2: var_18}
    var_20 = {var_0: var_19}
    var_21 = 'default'
    var_22 = 'integer'
    var_23 = {var_2: var_22}
    var_24 = 10
    var_25 = {var_12: var_24}
    var_26 = 42
    var_27 = {var_0: var_23, var_1: var_25, var_21: var_26}
    var_28 = 'properties'
    var_29 = 'object'
    var_30 = 'name'
    var_31 = {var_2: var_3}
    var_32 = {var_30: var_31}
    var_33 = {var_2: var_29, var_28: var_32}
    var_34 = 'required'
    var_35 = [var_30]
    var_36 = {var_34: var_35}
    var_37 = 'id'
    var_38 = {var_2: var_22}
    var_39 = {var_37: var_38}
    var_40 = {var_28: var_39}
    var_41 = {var_0: var_33, var_1: var_36, var_9: var_40}



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
    var_8 = 'email'
    var_9 = module_0.String(max_length=var_6, min_length=var_5, pattern=var_7, format=var_8)
    var_10 = module_1.to_json_schema(var_9)
    var_11 = True
    var_12 = module_0.String()
    var_13 = module_1.to_json_schema(var_12)
    var_14 = True
    var_15 = module_0.String(allow_blank=var_14)
    var_16 = module_1.to_json_schema(var_15)
    var_17 = 100
    var_18 = 5
    var_19 = module_0.Integer(minimum=var_4, maximum=var_17, multiple_of=var_18)
    var_20 = module_1.to_json_schema(var_19)
    var_21 = True
    var_22 = module_0.Integer()
    var_23 = module_1.to_json_schema(var_22)
    var_24 = module_0.Float(exclusive_minimum=var_4, exclusive_maximum=var_17)
    var_25 = module_1.to_json_schema(var_24)
    var_26 = module_0.Boolean()
    var_27 = module_1.to_json_schema(var_26)
    var_28 = True
    var_29 = module_0.Boolean()
    var_30 = module_1.to_json_schema(var_29)
    var_31 = module_0.String()
    var_32 = True
    var_33 = module_0.Array(var_31, min_items=var_28, max_items=var_6, unique_items=var_32)
    var_34 = module_1.to_json_schema(var_33)
    var_35 = module_0.String()
    var_36 = module_0.Integer()
    var_37 = [var_35, var_36]
    var_38 = module_0.Array(var_37)
    var_39 = module_1.to_json_schema(var_38)
    var_40 = 'items'
    var_41 = var_39[var_40]
    var_42 = var_39[var_40]
    var_43 = len(var_42)
    assert var_43 == 2
    var_44 = 'name'
    var_45 = 'age'
    var_46 = module_0.String()
    var_47 = module_0.Integer()
    var_48 = {var_44: var_46, var_45: var_47}
    var_49 = [var_44]
    var_50 = module_0.Object(properties=var_48, required=var_49)
    var_51 = module_1.to_json_schema(var_50)
    var_52 = '^S_'
    var_53 = module_0.String()
    var_54 = {var_52: var_53}
    var_55 = module_0.Object(pattern_properties=var_54)
    var_56 = module_1.to_json_schema(var_55)
    var_57 = module_0.Object(min_properties=var_32, max_properties=var_18)
    var_58 = module_1.to_json_schema(var_57)
    var_59 = 'a'
    var_60 = 'Option A'
    var_61 = (var_59, var_60)
    var_62 = 'b'
    var_63 = 'Option B'
    var_64 = (var_62, var_63)
    var_65 = [var_61, var_64]
    var_66 = module_0.Choice(choices=var_65)
    var_67 = module_1.to_json_schema(var_66)
    var_68 = 'fixed_value'
    var_69 = module_0.Const(var_68)
    var_70 = module_1.to_json_schema(var_69)
    var_71 = module_0.String()
    var_72 = module_0.Integer()
    var_73 = [var_71, var_72]
    var_74 = module_0.Union(var_73)
    var_75 = module_1.to_json_schema(var_74)
    var_76 = 'anyOf'
    var_77 = var_75[var_76]
    var_78 = len(var_77)
    assert var_78 == 2
    var_79 = module_0.String()
    var_80 = module_0.Integer()
    var_81 = [var_79, var_80]
    var_82 = module_2.OneOf(var_81)
    var_83 = module_1.to_json_schema(var_82)
    var_84 = 'oneOf'
    var_85 = var_83[var_84]
    var_86 = len(var_85)
    assert var_86 == 2
    var_87 = module_0.String(min_length=var_32)
    var_88 = module_0.String(max_length=var_17)
    var_89 = [var_87, var_88]
    var_90 = module_2.AllOf(var_89)
    var_91 = module_1.to_json_schema(var_90)
    var_92 = 'allOf'
    var_93 = var_91[var_92]
    var_94 = len(var_93)
    assert var_94 == 2
    var_95 = module_0.String()
    var_96 = module_2.Not(var_95)
    var_97 = module_1.to_json_schema(var_96)
    var_98 = module_0.String()
    var_99 = module_0.Integer()
    var_100 = module_0.Boolean()
    var_101 = module_2.IfThenElse(var_98, var_99, var_100)
    var_102 = module_1.to_json_schema(var_101)
    var_103 = module_0.String()
    var_104 = module_0.Integer()
    var_105 = module_2.IfThenElse(var_103, var_104)
    var_106 = module_1.to_json_schema(var_105)
    var_107 = 'StringDef'
    var_108 = 'IntDef'
    var_109 = module_0.String()
    var_110 = module_0.Integer()
    var_111 = {var_107: var_109, var_108: var_110}
    var_112 = module_0.String()
    var_113 = {var_107: var_112}
    var_114 = 'default_value'
    var_115 = module_0.String()
    var_116 = module_1.to_json_schema(var_115)
    var_117 = module_0.Object(additional_properties=var_4)
    var_118 = module_1.to_json_schema(var_117)
    var_119 = module_0.String()
    var_120 = module_0.Object(additional_properties=var_119)
    var_121 = module_1.to_json_schema(var_120)
    var_122 = 'additionalProperties'
    var_123 = var_121[var_122]
    var_124 = module_0.Array(additional_items=var_4)
    var_125 = module_1.to_json_schema(var_124)
    var_126 = module_0.String()
    var_127 = module_0.Array(additional_items=var_126)
    var_128 = module_1.to_json_schema(var_127)
    var_129 = 'additionalItems'
    var_130 = var_128[var_129]
    var_131 = module_0.String()
    var_132 = module_0.Integer()
    var_133 = {var_44: var_131, var_45: var_132}
    var_134 = [var_44]
    var_135 = module_3.Schema(var_133)



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
    var_14 = True
    var_15 = module_0.String(allow_blank=var_14)
    var_16 = module_1.to_json_schema(var_15)
    var_17 = 'minLength'
    var_18 = None
    var_19 = 100
    var_20 = 99
    var_21 = 5
    var_22 = module_0.Integer(minimum=var_4, maximum=var_19, exclusive_minimum=var_14, exclusive_maximum=var_20, multiple_of=var_21)
    var_23 = module_1.to_json_schema(var_22)
    var_24 = module_0.Float(minimum=var_4, maximum=var_14)
    var_25 = module_1.to_json_schema(var_24)
    var_26 = module_0.Boolean()
    var_27 = module_1.to_json_schema(var_26)
    var_28 = True
    var_29 = module_0.Boolean()
    var_30 = module_1.to_json_schema(var_29)
    var_31 = module_0.String()
    var_32 = True
    var_33 = module_0.Array(var_31, min_items=var_28, max_items=var_21, unique_items=var_32)
    var_34 = module_1.to_json_schema(var_33)
    var_35 = module_0.String()
    var_36 = module_0.Array(var_35, var_4)
    var_37 = module_1.to_json_schema(var_36)
    var_38 = module_0.String()
    var_39 = module_0.Integer()
    var_40 = module_0.Array(var_38, var_39)
    var_41 = module_1.to_json_schema(var_40)
    var_42 = 'name'
    var_43 = 'age'
    var_44 = module_0.String()
    var_45 = module_0.Integer()
    var_46 = {var_42: var_44, var_43: var_45}
    var_47 = [var_42]
    var_48 = module_0.Object(properties=var_46, min_properties=var_32, max_properties=var_6, required=var_47)
    var_49 = module_1.to_json_schema(var_48)
    var_50 = '^S_'
    var_51 = '^I_'
    var_52 = module_0.String()
    var_53 = module_0.Integer()
    var_54 = {var_50: var_52, var_51: var_53}
    var_55 = module_0.Object(pattern_properties=var_54)
    var_56 = module_1.to_json_schema(var_55)
    var_57 = module_0.Object(additional_properties=var_4)
    var_58 = module_1.to_json_schema(var_57)
    var_59 = module_0.String()
    var_60 = module_0.Object(additional_properties=var_59)
    var_61 = module_1.to_json_schema(var_60)
    var_62 = module_0.String(pattern=var_7)
    var_63 = module_0.Object(property_names=var_62)
    var_64 = module_1.to_json_schema(var_63)
    var_65 = 'a'
    var_66 = 'Option A'
    var_67 = (var_65, var_66)
    var_68 = 'b'
    var_69 = 'Option B'
    var_70 = (var_68, var_69)
    var_71 = [var_67, var_70]
    var_72 = module_0.Choice(choices=var_71)
    var_73 = module_1.to_json_schema(var_72)
    var_74 = 'constant_value'
    var_75 = module_0.Const(var_74)
    var_76 = module_1.to_json_schema(var_75)
    var_77 = module_0.String()
    var_78 = module_0.Integer()
    var_79 = [var_77, var_78]
    var_80 = module_0.Union(var_79)
    var_81 = module_1.to_json_schema(var_80)
    var_82 = 'anyOf'
    var_83 = var_81[var_82]
    var_84 = len(var_83)
    assert var_84 == 2
    var_85 = module_0.String()
    var_86 = module_0.Integer()
    var_87 = [var_85, var_86]
    var_88 = module_2.OneOf(var_87)
    var_89 = module_1.to_json_schema(var_88)
    var_90 = 'oneOf'
    var_91 = var_89[var_90]
    var_92 = len(var_91)
    assert var_92 == 2
    var_93 = module_0.String()
    var_94 = module_0.String(min_length=var_21)
    var_95 = [var_93, var_94]
    var_96 = module_2.AllOf(var_95)
    var_97 = module_1.to_json_schema(var_96)
    var_98 = 'allOf'
    var_99 = var_97[var_98]
    var_100 = len(var_99)
    assert var_100 == 2
    var_101 = module_0.String()
    var_102 = module_2.Not(var_101)
    var_103 = module_1.to_json_schema(var_102)
    var_104 = module_0.String()
    var_105 = module_0.Integer()
    var_106 = module_0.Boolean()
    var_107 = module_2.IfThenElse(var_104, var_105, var_106)
    var_108 = module_1.to_json_schema(var_107)
    var_109 = module_0.String()
    var_110 = module_0.Integer()
    var_111 = module_2.IfThenElse(var_109, var_110, var_18)
    var_112 = module_1.to_json_schema(var_111)
    var_113 = 'StringDef'
    var_114 = 'IntegerDef'
    var_115 = module_0.String()
    var_116 = module_0.Integer()
    var_117 = {var_113: var_115, var_114: var_116}
    var_118 = 'MyString'
    var_119 = module_0.String()
    var_120 = {var_118: var_119}



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
    var_8 = 'email'
    var_9 = module_0.String(max_length=var_6, min_length=var_5, pattern=var_7, format=var_8)
    var_10 = module_1.to_json_schema(var_9)
    var_11 = True
    var_12 = module_0.String()
    var_13 = module_1.to_json_schema(var_12)
    var_14 = True
    var_15 = module_0.String(allow_blank=var_14)
    var_16 = module_1.to_json_schema(var_15)
    var_17 = 100
    var_18 = 5
    var_19 = 95
    var_20 = module_0.Integer(minimum=var_4, maximum=var_17, exclusive_minimum=var_18, exclusive_maximum=var_19, multiple_of=var_18)
    var_21 = module_1.to_json_schema(var_20)
    var_22 = True
    var_23 = module_0.Integer()
    var_24 = module_1.to_json_schema(var_23)
    var_25 = module_0.Float(minimum=var_4, maximum=var_22)
    var_26 = module_1.to_json_schema(var_25)
    var_27 = module_0.Boolean()
    var_28 = module_1.to_json_schema(var_27)
    var_29 = True
    var_30 = module_0.Boolean()
    var_31 = module_1.to_json_schema(var_30)
    var_32 = module_0.String()
    var_33 = True
    var_34 = module_0.Array(var_32, min_items=var_29, max_items=var_6, unique_items=var_33)
    var_35 = module_1.to_json_schema(var_34)
    var_36 = True
    var_37 = module_0.Array()
    var_38 = module_1.to_json_schema(var_37)
    var_39 = 'name'
    var_40 = 'age'
    var_41 = module_0.String()
    var_42 = module_0.Integer()
    var_43 = {var_39: var_41, var_40: var_42}
    var_44 = [var_39]
    var_45 = module_0.Object(properties=var_43, required=var_44)
    var_46 = module_1.to_json_schema(var_45)
    var_47 = True
    var_48 = module_0.Object()
    var_49 = module_1.to_json_schema(var_48)
    var_50 = 'a'
    var_51 = 'A'
    var_52 = (var_50, var_51)
    var_53 = 'b'
    var_54 = 'B'
    var_55 = (var_53, var_54)
    var_56 = [var_52, var_55]
    var_57 = module_0.Choice(choices=var_56)
    var_58 = module_1.to_json_schema(var_57)
    var_59 = 'constant_value'
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
    var_78 = module_0.String()
    var_79 = module_0.String(min_length=var_18)
    var_80 = [var_78, var_79]
    var_81 = module_2.AllOf(var_80)
    var_82 = module_1.to_json_schema(var_81)
    var_83 = 'allOf'
    var_84 = var_82[var_83]
    var_85 = len(var_84)
    assert var_85 == 2
    var_86 = module_0.String()
    var_87 = module_2.Not(var_86)
    var_88 = module_1.to_json_schema(var_87)
    var_89 = module_0.String()
    var_90 = module_0.Integer()
    var_91 = module_0.Boolean()
    var_92 = module_2.IfThenElse(var_89, var_90, var_91)
    var_93 = module_1.to_json_schema(var_92)
    var_94 = module_0.String()
    var_95 = module_0.Integer()
    var_96 = module_2.IfThenElse(var_94, var_95)
    var_97 = module_1.to_json_schema(var_96)
    var_98 = 'User'
    var_99 = module_0.String()
    var_100 = {var_39: var_99}
    var_101 = module_0.Object(properties=var_100)
    var_102 = {var_98: var_101}
    var_103 = module_3.Reference(var_98, var_102)
    var_104 = module_1.to_json_schema(var_103)
    var_105 = 'Post'
    var_106 = module_0.String()
    var_107 = {var_39: var_106}
    var_108 = module_0.Object(properties=var_107)
    var_109 = 'title'
    var_110 = module_0.String()
    var_111 = {var_109: var_110}
    var_112 = module_0.Object(properties=var_111)
    var_113 = {var_98: var_108, var_105: var_112}
    var_114 = 'test'
    var_115 = module_0.String()
    var_116 = module_1.to_json_schema(var_115)
    var_117 = module_0.String()
    var_118 = module_0.Integer()
    var_119 = [var_117, var_118]
    var_120 = module_0.Array(var_119)
    var_121 = module_1.to_json_schema(var_120)
    var_122 = 'items'
    var_123 = var_121[var_122]
    var_124 = var_121[var_122]
    var_125 = len(var_124)
    assert var_125 == 2
    var_126 = module_0.Array(additional_items=var_4)
    var_127 = module_1.to_json_schema(var_126)
    var_128 = module_0.String()
    var_129 = module_0.Array(additional_items=var_128)
    var_130 = module_1.to_json_schema(var_129)
    var_131 = 'additionalItems'
    var_132 = var_130[var_131]
    var_133 = '^S_'
    var_134 = module_0.String()
    var_135 = {var_133: var_134}
    var_136 = module_0.Object(pattern_properties=var_135)
    var_137 = module_1.to_json_schema(var_136)
    var_138 = module_0.Object(additional_properties=var_4)
    var_139 = module_1.to_json_schema(var_138)



# Parsed testcases at query #23
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
    var_5 = module_0.String()
    var_6 = module_1.to_json_schema(var_5)
    var_7 = True
    var_8 = module_0.String()
    var_9 = module_1.to_json_schema(var_8)
    var_10 = 2
    var_11 = 10
    var_12 = '^[a-z]+$'
    var_13 = module_0.String(max_length=var_11, min_length=var_10, pattern=var_12)
    var_14 = module_1.to_json_schema(var_13)
    var_15 = 100
    var_16 = module_0.Integer(minimum=var_4, maximum=var_15)
    var_17 = module_1.to_json_schema(var_16)
    var_18 = module_0.Float()
    var_19 = module_1.to_json_schema(var_18)
    var_20 = module_0.Boolean()
    var_21 = module_1.to_json_schema(var_20)
    var_22 = module_0.String()
    var_23 = 5
    var_24 = module_0.Array(var_22, min_items=var_7, max_items=var_23)
    var_25 = module_1.to_json_schema(var_24)
    var_26 = 'name'
    var_27 = 'age'
    var_28 = module_0.String()
    var_29 = module_0.Integer()
    var_30 = {var_26: var_28, var_27: var_29}
    var_31 = [var_26]
    var_32 = module_0.Object(properties=var_30, required=var_31)
    var_33 = module_1.to_json_schema(var_32)
    var_34 = 'a'
    var_35 = 'A'
    var_36 = (var_34, var_35)
    var_37 = 'b'
    var_38 = 'B'
    var_39 = (var_37, var_38)
    var_40 = [var_36, var_39]
    var_41 = module_0.Choice(choices=var_40)
    var_42 = module_1.to_json_schema(var_41)
    var_43 = 42
    var_44 = module_0.Const(var_43)
    var_45 = module_1.to_json_schema(var_44)
    var_46 = module_0.String()
    var_47 = module_0.Integer()
    var_48 = [var_46, var_47]
    var_49 = module_0.Union(var_48)
    var_50 = module_1.to_json_schema(var_49)
    var_51 = 'anyOf'
    var_52 = var_50[var_51]
    var_53 = len(var_52)
    assert var_53 == 2
    var_54 = module_0.String()
    var_55 = module_0.Integer()
    var_56 = [var_54, var_55]
    var_57 = module_2.OneOf(var_56)
    var_58 = module_1.to_json_schema(var_57)
    var_59 = 'oneOf'
    var_60 = var_58[var_59]
    var_61 = len(var_60)
    assert var_61 == 2
    var_62 = module_0.String()
    var_63 = module_0.String(min_length=var_7)
    var_64 = [var_62, var_63]
    var_65 = module_2.AllOf(var_64)
    var_66 = module_1.to_json_schema(var_65)
    var_67 = 'allOf'
    var_68 = var_66[var_67]
    var_69 = len(var_68)
    assert var_69 == 2
    var_70 = module_0.String()
    var_71 = module_2.Not(var_70)
    var_72 = module_1.to_json_schema(var_71)
    var_73 = module_0.String()
    var_74 = module_0.Integer()
    var_75 = module_0.Boolean()
    var_76 = module_2.IfThenElse(var_73, var_74, var_75)
    var_77 = module_1.to_json_schema(var_76)
    var_78 = module_0.String()
    var_79 = module_0.Integer()
    var_80 = module_2.IfThenElse(var_78, var_79)
    var_81 = module_1.to_json_schema(var_80)
    var_82 = module_3.Definitions()
    var_83 = 'MySchema'
    var_84 = module_3.Reference(var_83, var_82)
    var_85 = module_1.to_json_schema(var_84)
    var_86 = module_0.String()
    var_87 = module_0.Integer()
    var_88 = {var_26: var_86, var_27: var_87}
    var_89 = [var_26]
    var_90 = module_3.Schema(var_88)
    var_91 = module_1.to_json_schema(var_90)
    var_92 = module_0.String()
    var_93 = module_0.Array(var_92, unique_items=var_7)
    var_94 = module_1.to_json_schema(var_93)
    var_95 = module_0.String()
    var_96 = module_0.Integer()
    var_97 = [var_95, var_96]
    var_98 = module_0.Boolean()
    var_99 = module_0.Array(var_97, var_98)
    var_100 = module_1.to_json_schema(var_99)
    var_101 = 'items'
    var_102 = var_100[var_101]
    var_103 = '^S_'
    var_104 = '^I_'
    var_105 = module_0.String()
    var_106 = module_0.Integer()
    var_107 = {var_103: var_105, var_104: var_106}
    var_108 = module_0.Object(pattern_properties=var_107)
    var_109 = module_1.to_json_schema(var_108)
    var_110 = module_0.String()
    var_111 = module_0.Object(additional_properties=var_110)
    var_112 = module_1.to_json_schema(var_111)
    var_113 = module_0.String(pattern=var_12)
    var_114 = module_0.Object(property_names=var_113)
    var_115 = module_1.to_json_schema(var_114)
    var_116 = module_3.Definitions()
    var_117 = module_1.to_json_schema(var_116)
    var_118 = module_0.String(allow_blank=var_7, min_length=var_4)
    var_119 = module_1.to_json_schema(var_118)
    var_120 = 'minLength'
    var_121 = module_0.String(allow_blank=var_4)
    var_122 = module_1.to_json_schema(var_121)
    var_123 = 0.5
    var_124 = 99.9
    var_125 = 0.1
    var_126 = module_0.Float(minimum=var_123, maximum=var_124, exclusive_minimum=var_125, exclusive_maximum=var_15, multiple_of=var_123)
    var_127 = module_1.to_json_schema(var_126)



# Parsed testcases at query #24
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
    var_14 = True
    var_15 = module_0.String(allow_blank=var_14)
    var_16 = module_1.to_json_schema(var_15)
    var_17 = 100
    var_18 = 5
    var_19 = module_0.Integer(minimum=var_4, maximum=var_17, multiple_of=var_18)
    var_20 = module_1.to_json_schema(var_19)
    var_21 = True
    var_22 = module_0.Integer()
    var_23 = module_1.to_json_schema(var_22)
    var_24 = 0.1
    var_25 = 99.9
    var_26 = module_0.Float(minimum=var_4, maximum=var_17, exclusive_minimum=var_24, exclusive_maximum=var_25)
    var_27 = module_1.to_json_schema(var_26)
    var_28 = module_0.Boolean()
    var_29 = module_1.to_json_schema(var_28)
    var_30 = True
    var_31 = module_0.Boolean()
    var_32 = module_1.to_json_schema(var_31)
    var_33 = module_0.String()
    var_34 = True
    var_35 = module_0.Array(var_33, min_items=var_30, max_items=var_6, unique_items=var_34)
    var_36 = module_1.to_json_schema(var_35)
    var_37 = module_0.String()
    var_38 = module_0.Integer()
    var_39 = [var_37, var_38]
    var_40 = module_0.Array(var_39)
    var_41 = module_1.to_json_schema(var_40)
    var_42 = 'items'
    var_43 = var_41[var_42]
    var_44 = var_41[var_42]
    var_45 = len(var_44)
    assert var_45 == 2
    var_46 = module_0.Array(additional_items=var_4)
    var_47 = module_1.to_json_schema(var_46)
    var_48 = module_0.String()
    var_49 = module_0.Array(additional_items=var_48)
    var_50 = module_1.to_json_schema(var_49)
    var_51 = 'name'
    var_52 = 'age'
    var_53 = module_0.String()
    var_54 = module_0.Integer()
    var_55 = {var_51: var_53, var_52: var_54}
    var_56 = [var_51]
    var_57 = module_0.Object(properties=var_55, min_properties=var_34, max_properties=var_6, required=var_56)
    var_58 = module_1.to_json_schema(var_57)
    var_59 = '^S_'
    var_60 = '^I_'
    var_61 = module_0.String()
    var_62 = module_0.Integer()
    var_63 = {var_59: var_61, var_60: var_62}
    var_64 = module_0.Object(pattern_properties=var_63)
    var_65 = module_1.to_json_schema(var_64)
    var_66 = module_0.Object(additional_properties=var_4)
    var_67 = module_1.to_json_schema(var_66)
    var_68 = module_0.String()
    var_69 = module_0.Object(additional_properties=var_68)
    var_70 = module_1.to_json_schema(var_69)
    var_71 = module_0.String(pattern=var_7)
    var_72 = module_0.Object(property_names=var_71)
    var_73 = module_1.to_json_schema(var_72)
    var_74 = 'a'
    var_75 = 'Option A'
    var_76 = (var_74, var_75)
    var_77 = 'b'
    var_78 = 'Option B'
    var_79 = (var_77, var_78)
    var_80 = [var_76, var_79]
    var_81 = module_0.Choice(choices=var_80)
    var_82 = module_1.to_json_schema(var_81)
    var_83 = 'constant_value'
    var_84 = module_0.Const(var_83)
    var_85 = module_1.to_json_schema(var_84)
    var_86 = module_0.String()
    var_87 = module_0.Integer()
    var_88 = [var_86, var_87]
    var_89 = module_0.Union(var_88)
    var_90 = module_1.to_json_schema(var_89)
    var_91 = 'anyOf'
    var_92 = var_90[var_91]
    var_93 = len(var_92)
    assert var_93 == 2
    var_94 = module_0.String()
    var_95 = module_0.Integer()
    var_96 = [var_94, var_95]
    var_97 = module_2.OneOf(var_96)
    var_98 = module_1.to_json_schema(var_97)
    var_99 = 'oneOf'
    var_100 = var_98[var_99]
    var_101 = len(var_100)
    assert var_101 == 2
    var_102 = module_0.String()
    var_103 = 'A'
    var_104 = (var_74, var_103)
    var_105 = [var_104]
    var_106 = module_0.Choice(choices=var_105)
    var_107 = [var_102, var_106]
    var_108 = module_2.AllOf(var_107)
    var_109 = module_1.to_json_schema(var_108)
    var_110 = 'allOf'
    var_111 = var_109[var_110]
    var_112 = len(var_111)
    assert var_112 == 2
    var_113 = module_0.String()
    var_114 = module_2.Not(var_113)
    var_115 = module_1.to_json_schema(var_114)
    var_116 = (var_74, var_103)
    var_117 = [var_116]
    var_118 = module_0.Choice(choices=var_117)
    var_119 = module_0.String()
    var_120 = module_0.Integer()
    var_121 = module_2.IfThenElse(var_118, var_119, var_120)
    var_122 = module_1.to_json_schema(var_121)
    var_123 = (var_74, var_103)
    var_124 = [var_123]
    var_125 = module_0.Choice(choices=var_124)
    var_126 = module_0.String()
    var_127 = module_2.IfThenElse(var_125, var_126)
    var_128 = module_1.to_json_schema(var_127)
    var_129 = module_3.Definitions()
    var_130 = 'TestSchema'
    var_131 = module_3.Reference(var_130, var_129)
    var_132 = module_1.to_json_schema(var_131)



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
    var_4 = False
    var_5 = 1
    var_6 = 10
    var_7 = '^[a-z]+$'
    var_8 = 'email'
    var_9 = module_0.String(max_length=var_6, min_length=var_5, pattern=var_7, format=var_8)
    var_10 = module_1.to_json_schema(var_9)
    var_11 = True
    var_12 = module_0.String()
    var_13 = 'type'
    var_14 = to_json_schema(var_12)[var_13]
    var_15 = 100
    var_16 = 5
    var_17 = module_0.Integer(minimum=var_4, maximum=var_15, multiple_of=var_16)
    var_18 = module_1.to_json_schema(var_17)
    var_19 = module_0.Float(exclusive_minimum=var_4, exclusive_maximum=var_11)
    var_20 = module_1.to_json_schema(var_19)
    var_21 = module_0.Boolean()
    var_22 = module_1.to_json_schema(var_21)
    var_23 = True
    var_24 = module_0.Boolean()
    var_25 = to_json_schema(var_24)[var_13]
    var_26 = module_0.String()
    var_27 = True
    var_28 = module_0.Array(var_26, min_items=var_23, max_items=var_6, unique_items=var_27)
    var_29 = module_1.to_json_schema(var_28)
    var_30 = module_0.String()
    var_31 = module_0.Integer()
    var_32 = [var_30, var_31]
    var_33 = module_0.Array(var_32)
    var_34 = module_1.to_json_schema(var_33)
    var_35 = 'items'
    var_36 = var_34[var_35]
    var_37 = var_34[var_35]
    var_38 = len(var_37)
    assert var_38 == 2
    var_39 = 'name'
    var_40 = 'age'
    var_41 = module_0.String()
    var_42 = module_0.Integer()
    var_43 = {var_39: var_41, var_40: var_42}
    var_44 = [var_39]
    var_45 = module_0.Object(properties=var_43, min_properties=var_27, max_properties=var_6, required=var_44)
    var_46 = module_1.to_json_schema(var_45)
    var_47 = '^S_'
    var_48 = '^I_'
    var_49 = module_0.String()
    var_50 = module_0.Integer()
    var_51 = {var_47: var_49, var_48: var_50}
    var_52 = module_0.Object(pattern_properties=var_51)
    var_53 = module_1.to_json_schema(var_52)
    var_54 = module_0.Object(additional_properties=var_4)
    var_55 = module_1.to_json_schema(var_54)
    var_56 = module_0.String()
    var_57 = module_0.Object(additional_properties=var_56)
    var_58 = module_1.to_json_schema(var_57)
    var_59 = 'additionalProperties'
    var_60 = var_58[var_59]
    var_61 = module_0.String(pattern=var_7)
    var_62 = module_0.Object(property_names=var_61)
    var_63 = module_1.to_json_schema(var_62)
    var_64 = 'a'
    var_65 = 'Option A'
    var_66 = (var_64, var_65)
    var_67 = 'b'
    var_68 = 'Option B'
    var_69 = (var_67, var_68)
    var_70 = [var_66, var_69]
    var_71 = module_0.Choice(choices=var_70)
    var_72 = module_1.to_json_schema(var_71)
    var_73 = 'fixed_value'
    var_74 = module_0.Const(var_73)
    var_75 = module_1.to_json_schema(var_74)
    var_76 = module_0.String()
    var_77 = module_0.Integer()
    var_78 = [var_76, var_77]
    var_79 = module_0.Union(var_78)
    var_80 = module_1.to_json_schema(var_79)
    var_81 = 'anyOf'
    var_82 = var_80[var_81]
    var_83 = len(var_82)
    assert var_83 == 2
    var_84 = module_0.String()
    var_85 = module_0.Integer()
    var_86 = [var_84, var_85]
    var_87 = module_2.OneOf(var_86)
    var_88 = module_1.to_json_schema(var_87)
    var_89 = 'oneOf'
    var_90 = var_88[var_89]
    var_91 = len(var_90)
    assert var_91 == 2
    var_92 = module_0.String(min_length=var_27)
    var_93 = module_0.String(max_length=var_6)
    var_94 = [var_92, var_93]
    var_95 = module_2.AllOf(var_94)
    var_96 = module_1.to_json_schema(var_95)
    var_97 = 'allOf'
    var_98 = var_96[var_97]
    var_99 = len(var_98)
    assert var_99 == 2
    var_100 = module_0.String()
    var_101 = module_2.Not(var_100)
    var_102 = module_1.to_json_schema(var_101)
    var_103 = module_0.String()
    var_104 = module_0.Integer()
    var_105 = module_0.Boolean()
    var_106 = module_2.IfThenElse(var_103, var_104, var_105)
    var_107 = module_1.to_json_schema(var_106)
    var_108 = module_0.String()
    var_109 = module_2.IfThenElse(var_108)
    var_110 = module_1.to_json_schema(var_109)
    var_111 = 'User'
    var_112 = module_0.String()
    var_113 = {var_39: var_112}
    var_114 = module_0.Object(properties=var_113)
    var_115 = {var_111: var_114}
    var_116 = module_0.String()
    var_117 = module_0.Integer()
    var_118 = {var_39: var_116, var_40: var_117}
    var_119 = [var_39]
    var_120 = module_3.Schema(var_118)
    var_121 = module_1.to_json_schema(var_120)
    var_122 = module_0.String()
    var_123 = [var_122]
    var_124 = module_0.Array(var_123, var_4)
    var_125 = module_1.to_json_schema(var_124)
    var_126 = module_0.String()
    var_127 = [var_126]
    var_128 = module_0.Integer()
    var_129 = module_0.Array(var_127, var_128)
    var_130 = module_1.to_json_schema(var_129)
    var_131 = 'additionalItems'
    var_132 = var_130[var_131]
    var_133 = 'test_default'
    var_134 = module_0.String()
    var_135 = module_1.to_json_schema(var_134)



# Parsed testcases at query #26
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
    var_5 = module_0.String()
    var_6 = module_1.to_json_schema(var_5)
    var_7 = True
    var_8 = module_0.String()
    var_9 = module_1.to_json_schema(var_8)
    var_10 = 5
    var_11 = 10
    var_12 = '^[a-z]+$'
    var_13 = module_0.String(max_length=var_11, min_length=var_10, pattern=var_12)
    var_14 = module_1.to_json_schema(var_13)
    var_15 = 100
    var_16 = module_0.Integer(minimum=var_4, maximum=var_15)
    var_17 = module_1.to_json_schema(var_16)
    var_18 = module_0.Float(exclusive_minimum=var_4, exclusive_maximum=var_7)
    var_19 = module_1.to_json_schema(var_18)
    var_20 = module_0.Boolean()
    var_21 = module_1.to_json_schema(var_20)
    var_22 = module_0.Boolean()
    var_23 = module_1.to_json_schema(var_22)
    var_24 = module_0.String()
    var_25 = module_0.Array(var_24, min_items=var_7, max_items=var_10)
    var_26 = module_1.to_json_schema(var_25)
    var_27 = module_0.String()
    var_28 = module_0.Integer()
    var_29 = [var_27, var_28]
    var_30 = module_0.Array(var_29)
    var_31 = module_1.to_json_schema(var_30)
    var_32 = 'items'
    var_33 = var_31[var_32]
    var_34 = var_31[var_32]
    var_35 = len(var_34)
    assert var_35 == 2
    var_36 = module_0.String()
    var_37 = module_0.Array(var_36, unique_items=var_7)
    var_38 = module_1.to_json_schema(var_37)
    var_39 = 'name'
    var_40 = 'age'
    var_41 = module_0.String()
    var_42 = module_0.Integer()
    var_43 = {var_39: var_41, var_40: var_42}
    var_44 = [var_39]
    var_45 = module_0.Object(properties=var_43, required=var_44)
    var_46 = module_1.to_json_schema(var_45)
    var_47 = '^S_'
    var_48 = module_0.String()
    var_49 = {var_47: var_48}
    var_50 = module_0.Object(pattern_properties=var_49)
    var_51 = module_1.to_json_schema(var_50)
    var_52 = module_0.String()
    var_53 = module_0.Object(additional_properties=var_52)
    var_54 = module_1.to_json_schema(var_53)
    var_55 = module_0.String(pattern=var_12)
    var_56 = module_0.Object(property_names=var_55)
    var_57 = module_1.to_json_schema(var_56)
    var_58 = 'a'
    var_59 = 'A'
    var_60 = (var_58, var_59)
    var_61 = 'b'
    var_62 = 'B'
    var_63 = (var_61, var_62)
    var_64 = [var_60, var_63]
    var_65 = module_0.Choice(choices=var_64)
    var_66 = module_1.to_json_schema(var_65)
    var_67 = 'fixed_value'
    var_68 = module_0.Const(var_67)
    var_69 = module_1.to_json_schema(var_68)
    var_70 = module_0.String()
    var_71 = module_0.Integer()
    var_72 = [var_70, var_71]
    var_73 = module_0.Union(var_72)
    var_74 = module_1.to_json_schema(var_73)
    var_75 = 'anyOf'
    var_76 = var_74[var_75]
    var_77 = len(var_76)
    assert var_77 == 2
    var_78 = module_0.String()
    var_79 = module_0.Integer()
    var_80 = [var_78, var_79]
    var_81 = module_2.OneOf(var_80)
    var_82 = module_1.to_json_schema(var_81)
    var_83 = 'oneOf'
    var_84 = var_82[var_83]
    var_85 = len(var_84)
    assert var_85 == 2
    var_86 = module_0.String()
    var_87 = (var_58, var_59)
    var_88 = [var_87]
    var_89 = module_0.Choice(choices=var_88)
    var_90 = [var_86, var_89]
    var_91 = module_2.AllOf(var_90)
    var_92 = module_1.to_json_schema(var_91)
    var_93 = 'allOf'
    var_94 = var_92[var_93]
    var_95 = len(var_94)
    assert var_95 == 2
    var_96 = module_0.String()
    var_97 = module_2.Not(var_96)
    var_98 = module_1.to_json_schema(var_97)
    var_99 = (var_58, var_59)
    var_100 = [var_99]
    var_101 = module_0.Choice(choices=var_100)
    var_102 = module_0.String()
    var_103 = module_0.Integer()
    var_104 = module_2.IfThenElse(var_101, var_102, var_103)
    var_105 = module_1.to_json_schema(var_104)
    var_106 = (var_58, var_59)
    var_107 = [var_106]
    var_108 = module_0.Choice(choices=var_107)
    var_109 = module_0.String()
    var_110 = module_2.IfThenElse(var_108, var_109)
    var_111 = module_1.to_json_schema(var_110)
    var_112 = 'StringType'
    var_113 = module_0.String()
    var_114 = {var_112: var_113}
    var_115 = 'MySchema'
    var_116 = module_0.String()
    var_117 = {var_115: var_116}
    var_118 = module_0.Object(min_properties=var_7, max_properties=var_10)
    var_119 = module_1.to_json_schema(var_118)
    var_120 = module_0.String()
    var_121 = module_0.Integer()
    var_122 = {var_39: var_120, var_40: var_121}
    var_123 = [var_39]
    var_124 = module_3.Schema(var_122)
    var_125 = module_1.to_json_schema(var_124)
    var_126 = module_0.String()
    var_127 = module_0.Array(var_126, var_4)
    var_128 = module_1.to_json_schema(var_127)
    var_129 = module_0.String()
    var_130 = module_0.Integer()
    var_131 = module_0.Array(var_129, var_130)
    var_132 = module_1.to_json_schema(var_131)
    var_133 = 'additionalItems'
    var_134 = var_132[var_133]
    var_135 = module_0.Object(additional_properties=var_7)
    var_136 = module_1.to_json_schema(var_135)



# Parsed testcases at query #27
#--------------------------


import typesystem.fields as module_0
import typesystem.json_schema as module_1
import typesystem.composites as module_2
import re as module_3

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
    var_12 = '^[a-z]+$'
    var_13 = module_3.compile(var_12)
    var_14 = module_0.String()
    var_15 = module_1.to_json_schema(var_14)
    var_16 = 'email'
    var_17 = module_0.String(format=var_16)
    var_18 = module_1.to_json_schema(var_17)
    var_19 = 100
    var_20 = module_0.Integer(minimum=var_4, maximum=var_19)
    var_21 = module_1.to_json_schema(var_20)
    var_22 = True
    var_23 = module_0.Integer()
    var_24 = module_1.to_json_schema(var_23)
    var_25 = module_0.Float(exclusive_minimum=var_4, exclusive_maximum=var_22)
    var_26 = module_1.to_json_schema(var_25)
    var_27 = 0.5
    var_28 = module_0.Float(multiple_of=var_27)
    var_29 = module_1.to_json_schema(var_28)
    var_30 = module_0.Boolean()
    var_31 = module_1.to_json_schema(var_30)
    var_32 = True
    var_33 = module_0.Boolean()
    var_34 = module_1.to_json_schema(var_33)
    var_35 = module_0.String()
    var_36 = module_0.Array(var_35, min_items=var_32, max_items=var_6)
    var_37 = module_1.to_json_schema(var_36)
    var_38 = True
    var_39 = module_0.Integer()
    var_40 = module_0.Array(var_39, unique_items=var_38)
    var_41 = module_1.to_json_schema(var_40)
    var_42 = module_0.String()
    var_43 = module_0.Integer()
    var_44 = [var_42, var_43]
    var_45 = module_0.Array(var_44)
    var_46 = module_1.to_json_schema(var_45)
    var_47 = 'items'
    var_48 = var_46[var_47]
    var_49 = var_46[var_47]
    var_50 = len(var_49)
    assert var_50 == 2
    var_51 = module_0.String()
    var_52 = module_0.Array(var_51, var_4)
    var_53 = module_1.to_json_schema(var_52)
    var_54 = module_0.Integer()
    var_55 = module_0.String()
    var_56 = module_0.Array(var_55, var_54)
    var_57 = module_1.to_json_schema(var_56)
    var_58 = 'name'
    var_59 = 'age'
    var_60 = module_0.String()
    var_61 = module_0.Integer()
    var_62 = {var_58: var_60, var_59: var_61}
    var_63 = [var_58]
    var_64 = module_0.Object(properties=var_62, required=var_63)
    var_65 = module_1.to_json_schema(var_64)
    var_66 = '^S_'
    var_67 = module_0.String()
    var_68 = {var_66: var_67}
    var_69 = module_0.Object(pattern_properties=var_68)
    var_70 = module_1.to_json_schema(var_69)
    var_71 = module_0.Object(additional_properties=var_4)
    var_72 = module_1.to_json_schema(var_71)
    var_73 = module_0.String()
    var_74 = module_0.Object(additional_properties=var_73)
    var_75 = module_1.to_json_schema(var_74)
    var_76 = '^[a-z]+'
    var_77 = module_3.compile(var_76)
    var_78 = module_0.String()
    var_79 = module_0.Object(property_names=var_78)
    var_80 = module_1.to_json_schema(var_79)
    var_81 = 5
    var_82 = module_0.Object(min_properties=var_38, max_properties=var_81)
    var_83 = module_1.to_json_schema(var_82)
    var_84 = 'a'
    var_85 = 'A'
    var_86 = (var_84, var_85)
    var_87 = 'b'
    var_88 = 'B'
    var_89 = (var_87, var_88)
    var_90 = [var_86, var_89]
    var_91 = module_0.Choice(choices=var_90)
    var_92 = module_1.to_json_schema(var_91)
    var_93 = 'constant_value'
    var_94 = module_0.Const(var_93)
    var_95 = module_1.to_json_schema(var_94)
    var_96 = module_0.String()
    var_97 = module_0.Integer()
    var_98 = [var_96, var_97]
    var_99 = module_0.Union(var_98)
    var_100 = module_1.to_json_schema(var_99)
    var_101 = 'anyOf'
    var_102 = var_100[var_101]
    var_103 = len(var_102)
    assert var_103 == 2
    var_104 = module_0.String()
    var_105 = module_0.Integer()
    var_106 = [var_104, var_105]
    var_107 = module_2.OneOf(var_106)
    var_108 = module_1.to_json_schema(var_107)
    var_109 = 'oneOf'
    var_110 = var_108[var_109]
    var_111 = len(var_110)
    assert var_111 == 2
    var_112 = module_0.String(min_length=var_38)
    var_113 = module_0.String(max_length=var_6)
    var_114 = [var_112, var_113]
    var_115 = module_2.AllOf(var_114)
    var_116 = module_1.to_json_schema(var_115)
    var_117 = 'allOf'
    var_118 = var_116[var_117]
    var_119 = len(var_118)
    assert var_119 == 2
    var_120 = module_0.String()
    var_121 = module_2.Not(var_120)
    var_122 = module_1.to_json_schema(var_121)
    var_123 = 'type'
    var_124 = (var_84, var_85)
    var_125 = [var_124]
    var_126 = module_0.Choice(choices=var_125)
    var_127 = {var_123: var_126}
    var_128 = module_0.Object(properties=var_127)
    var_129 = module_0.String()
    var_130 = module_0.Integer()
    var_131 = module_2.IfThenElse(var_128, var_129, var_130)
    var_132 = module_1.to_json_schema(var_131)
    var_133 = module_0.Boolean()
    var_134 = module_0.String()
    var_135 = module_2.IfThenElse(var_133, var_134)



# Parsed testcases at query #28
#--------------------------


import typesystem.fields as module_0
import typesystem.json_schema as module_1
import typesystem.composites as module_2
import typesystem.schemas as module_3
import builtins as module_4

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
    var_14 = True
    var_15 = module_0.String(allow_blank=var_14)
    var_16 = module_1.to_json_schema(var_15)
    var_17 = 100
    var_18 = 5
    var_19 = module_0.Integer(minimum=var_4, maximum=var_17, multiple_of=var_18)
    var_20 = module_1.to_json_schema(var_19)
    var_21 = module_0.Float(minimum=var_4, maximum=var_14, exclusive_minimum=var_4, exclusive_maximum=var_14)
    var_22 = module_1.to_json_schema(var_21)
    var_23 = module_0.Boolean()
    var_24 = module_1.to_json_schema(var_23)
    var_25 = True
    var_26 = module_0.Boolean()
    var_27 = module_1.to_json_schema(var_26)
    var_28 = module_0.String()
    var_29 = True
    var_30 = module_0.Array(var_28, min_items=var_25, max_items=var_6, unique_items=var_29)
    var_31 = module_1.to_json_schema(var_30)
    var_32 = module_0.String()
    var_33 = module_0.Integer()
    var_34 = [var_32, var_33]
    var_35 = module_0.Array(var_34)
    var_36 = module_1.to_json_schema(var_35)
    var_37 = 'items'
    var_38 = var_36[var_37]
    var_39 = var_36[var_37]
    var_40 = len(var_39)
    assert var_40 == 2
    var_41 = module_0.Array(additional_items=var_4)
    var_42 = module_1.to_json_schema(var_41)
    var_43 = 'name'
    var_44 = 'age'
    var_45 = module_0.String()
    var_46 = module_0.Integer()
    var_47 = {var_43: var_45, var_44: var_46}
    var_48 = [var_43]
    var_49 = module_0.Object(properties=var_47, required=var_48)
    var_50 = module_1.to_json_schema(var_49)
    var_51 = '^S_'
    var_52 = module_0.String()
    var_53 = {var_51: var_52}
    var_54 = module_0.Object(pattern_properties=var_53)
    var_55 = module_1.to_json_schema(var_54)
    var_56 = module_0.Object(additional_properties=var_4)
    var_57 = module_1.to_json_schema(var_56)
    var_58 = module_0.String(pattern=var_7)
    var_59 = module_0.Object(property_names=var_58)
    var_60 = module_1.to_json_schema(var_59)
    var_61 = 'a'
    var_62 = 'Option A'
    var_63 = (var_61, var_62)
    var_64 = 'b'
    var_65 = 'Option B'
    var_66 = (var_64, var_65)
    var_67 = [var_63, var_66]
    var_68 = module_0.Choice(choices=var_67)
    var_69 = module_1.to_json_schema(var_68)
    var_70 = 'fixed_value'
    var_71 = module_0.Const(var_70)
    var_72 = module_1.to_json_schema(var_71)
    var_73 = module_0.String()
    var_74 = module_0.Integer()
    var_75 = [var_73, var_74]
    var_76 = module_0.Union(var_75)
    var_77 = module_1.to_json_schema(var_76)
    var_78 = 'anyOf'
    var_79 = var_77[var_78]
    var_80 = len(var_79)
    assert var_80 == 2
    var_81 = module_0.String()
    var_82 = module_0.Integer()
    var_83 = [var_81, var_82]
    var_84 = module_2.OneOf(var_83)
    var_85 = module_1.to_json_schema(var_84)
    var_86 = 'oneOf'
    var_87 = var_85[var_86]
    var_88 = len(var_87)
    assert var_88 == 2
    var_89 = module_0.String()
    var_90 = 'test'
    var_91 = module_0.Const(var_90)
    var_92 = [var_89, var_91]
    var_93 = module_2.AllOf(var_92)
    var_94 = module_1.to_json_schema(var_93)
    var_95 = 'allOf'
    var_96 = var_94[var_95]
    var_97 = len(var_96)
    assert var_97 == 2
    var_98 = module_0.String()
    var_99 = module_2.Not(var_98)
    var_100 = module_1.to_json_schema(var_99)
    var_101 = module_0.String()
    var_102 = module_0.Integer()
    var_103 = module_0.Boolean()
    var_104 = module_2.IfThenElse(var_101, var_102, var_103)
    var_105 = module_1.to_json_schema(var_104)
    var_106 = module_0.String()
    var_107 = module_2.IfThenElse(var_106)
    var_108 = module_1.to_json_schema(var_107)
    var_109 = 'TestSchema'
    var_110 = module_0.String()
    var_111 = {var_109: var_110}
    var_112 = module_0.String()
    var_113 = {var_109: var_112}
    var_114 = module_0.String()
    var_115 = module_0.Integer()
    var_116 = {var_43: var_114, var_44: var_115}
    var_117 = [var_43]
    var_118 = module_3.Schema(var_116)
    var_119 = module_1.to_json_schema(var_118)
    var_120 = module_4.object()
    var_121 = module_1.to_json_schema(var_120)



# Parsed testcases at query #29
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
    var_14 = True
    var_15 = module_0.String(allow_blank=var_14)
    var_16 = module_1.to_json_schema(var_15)
    var_17 = 100
    var_18 = 5
    var_19 = 95
    var_20 = module_0.Integer(minimum=var_4, maximum=var_17, exclusive_minimum=var_18, exclusive_maximum=var_19, multiple_of=var_18)
    var_21 = module_1.to_json_schema(var_20)
    var_22 = True
    var_23 = module_0.Integer()
    var_24 = module_1.to_json_schema(var_23)
    var_25 = module_0.Float(minimum=var_4, maximum=var_17)
    var_26 = module_1.to_json_schema(var_25)
    var_27 = module_0.Boolean()
    var_28 = module_1.to_json_schema(var_27)
    var_29 = True
    var_30 = module_0.Boolean()
    var_31 = module_1.to_json_schema(var_30)
    var_32 = module_0.String()
    var_33 = True
    var_34 = module_0.Array(var_32, min_items=var_29, max_items=var_6, unique_items=var_33)
    var_35 = module_1.to_json_schema(var_34)
    var_36 = module_0.String()
    var_37 = module_0.Integer()
    var_38 = [var_36, var_37]
    var_39 = module_0.Array(var_38)
    var_40 = module_1.to_json_schema(var_39)
    var_41 = 'items'
    var_42 = var_40[var_41]
    var_43 = var_40[var_41]
    var_44 = len(var_43)
    assert var_44 == 2
    var_45 = module_0.Array(additional_items=var_4)
    var_46 = module_1.to_json_schema(var_45)
    var_47 = module_0.String()
    var_48 = module_0.Array(additional_items=var_47)
    var_49 = module_1.to_json_schema(var_48)
    var_50 = 'additionalItems'
    var_51 = var_49[var_50]
    var_52 = 'name'
    var_53 = 'age'
    var_54 = module_0.String()
    var_55 = module_0.Integer()
    var_56 = {var_52: var_54, var_53: var_55}
    var_57 = [var_52]
    var_58 = module_0.Object(properties=var_56, min_properties=var_33, max_properties=var_6, required=var_57)
    var_59 = module_1.to_json_schema(var_58)
    var_60 = module_0.String()
    var_61 = {var_7: var_60}
    var_62 = module_0.Object(pattern_properties=var_61)
    var_63 = module_1.to_json_schema(var_62)
    var_64 = module_0.Object(additional_properties=var_4)
    var_65 = module_1.to_json_schema(var_64)
    var_66 = module_0.String()
    var_67 = module_0.Object(additional_properties=var_66)
    var_68 = module_1.to_json_schema(var_67)
    var_69 = 'additionalProperties'
    var_70 = var_68[var_69]
    var_71 = module_0.String(pattern=var_7)
    var_72 = module_0.Object(property_names=var_71)
    var_73 = module_1.to_json_schema(var_72)
    var_74 = 'a'
    var_75 = 'Option A'
    var_76 = (var_74, var_75)
    var_77 = 'b'
    var_78 = 'Option B'
    var_79 = (var_77, var_78)
    var_80 = [var_76, var_79]
    var_81 = module_0.Choice(choices=var_80)
    var_82 = module_1.to_json_schema(var_81)
    var_83 = 'constant_value'
    var_84 = module_0.Const(var_83)
    var_85 = module_1.to_json_schema(var_84)
    var_86 = module_0.String()
    var_87 = module_0.Integer()
    var_88 = [var_86, var_87]
    var_89 = module_0.Union(var_88)
    var_90 = module_1.to_json_schema(var_89)
    var_91 = 'anyOf'
    var_92 = var_90[var_91]
    var_93 = len(var_92)
    assert var_93 == 2
    var_94 = module_0.String()
    var_95 = module_0.Integer()
    var_96 = [var_94, var_95]
    var_97 = module_2.OneOf(var_96)
    var_98 = module_1.to_json_schema(var_97)
    var_99 = 'oneOf'
    var_100 = var_98[var_99]
    var_101 = len(var_100)
    assert var_101 == 2
    var_102 = module_0.String()
    var_103 = module_0.String(min_length=var_18)
    var_104 = [var_102, var_103]
    var_105 = module_2.AllOf(var_104)
    var_106 = module_1.to_json_schema(var_105)
    var_107 = 'allOf'
    var_108 = var_106[var_107]
    var_109 = len(var_108)
    assert var_109 == 2
    var_110 = module_0.String()
    var_111 = module_0.Integer()
    var_112 = module_0.Boolean()
    var_113 = module_2.IfThenElse(var_110, var_111, var_112)
    var_114 = module_1.to_json_schema(var_113)
    var_115 = module_0.String()
    var_116 = module_2.IfThenElse(var_115)
    var_117 = module_1.to_json_schema(var_116)
    var_118 = module_0.String()
    var_119 = module_2.Not(var_118)
    var_120 = module_1.to_json_schema(var_119)
    var_121 = 'TestSchema'
    var_122 = module_0.String()
    var_123 = {var_121: var_122}
    var_124 = module_0.String()
    var_125 = {var_121: var_124}



# Parsed testcases at query #30
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
    var_6 = True
    var_7 = module_0.String()
    var_8 = module_1.to_json_schema(var_7)
    var_9 = 2
    var_10 = 10
    var_11 = '^[a-z]+$'
    var_12 = module_0.String(max_length=var_10, min_length=var_9, pattern=var_11)
    var_13 = module_1.to_json_schema(var_12)
    var_14 = module_0.Integer()
    var_15 = module_1.to_json_schema(var_14)
    var_16 = 0
    var_17 = 100
    var_18 = 5
    var_19 = module_0.Integer(minimum=var_16, maximum=var_17, multiple_of=var_18)
    var_20 = module_1.to_json_schema(var_19)
    var_21 = module_0.Float()
    var_22 = module_1.to_json_schema(var_21)
    var_23 = module_0.Float(exclusive_minimum=var_16, exclusive_maximum=var_17)
    var_24 = module_1.to_json_schema(var_23)
    var_25 = module_0.Boolean()
    var_26 = module_1.to_json_schema(var_25)
    var_27 = module_0.Boolean()
    var_28 = module_1.to_json_schema(var_27)
    var_29 = module_0.Array()
    var_30 = module_1.to_json_schema(var_29)
    var_31 = module_0.String()
    var_32 = module_0.Array(var_31)
    var_33 = module_1.to_json_schema(var_32)
    var_34 = module_0.Array(min_items=var_6, max_items=var_18, unique_items=var_6)
    var_35 = module_1.to_json_schema(var_34)
    var_36 = module_0.Object()
    var_37 = module_1.to_json_schema(var_36)
    var_38 = 'name'
    var_39 = 'age'
    var_40 = module_0.String()
    var_41 = module_0.Integer()
    var_42 = {var_38: var_40, var_39: var_41}
    var_43 = module_0.Object(properties=var_42)
    var_44 = module_1.to_json_schema(var_43)
    var_45 = module_0.String()
    var_46 = {var_38: var_45}
    var_47 = [var_38]
    var_48 = module_0.Object(properties=var_46, required=var_47)
    var_49 = module_1.to_json_schema(var_48)
    var_50 = '^S_'
    var_51 = module_0.String()
    var_52 = {var_50: var_51}
    var_53 = module_0.Object(pattern_properties=var_52)
    var_54 = module_1.to_json_schema(var_53)
    var_55 = module_0.String()
    var_56 = module_0.Object(additional_properties=var_55)
    var_57 = module_1.to_json_schema(var_56)
    var_58 = 'a'
    var_59 = 'Option A'
    var_60 = (var_58, var_59)
    var_61 = 'b'
    var_62 = 'Option B'
    var_63 = (var_61, var_62)
    var_64 = [var_60, var_63]
    var_65 = module_0.Choice(choices=var_64)
    var_66 = module_1.to_json_schema(var_65)
    var_67 = 'fixed_value'
    var_68 = module_0.Const(var_67)
    var_69 = module_1.to_json_schema(var_68)
    var_70 = module_0.String()
    var_71 = module_0.Integer()
    var_72 = [var_70, var_71]
    var_73 = module_0.Union(var_72)
    var_74 = module_1.to_json_schema(var_73)
    var_75 = 'anyOf'
    var_76 = var_74[var_75]
    var_77 = len(var_76)
    assert var_77 == 2
    var_78 = module_0.String()
    var_79 = module_0.Integer()
    var_80 = [var_78, var_79]
    var_81 = module_2.OneOf(var_80)
    var_82 = module_1.to_json_schema(var_81)
    var_83 = 'oneOf'
    var_84 = var_82[var_83]
    var_85 = len(var_84)
    assert var_85 == 2
    var_86 = module_0.String()
    var_87 = module_0.String(min_length=var_18)
    var_88 = [var_86, var_87]
    var_89 = module_2.AllOf(var_88)
    var_90 = module_1.to_json_schema(var_89)
    var_91 = 'allOf'
    var_92 = var_90[var_91]
    var_93 = len(var_92)
    assert var_93 == 2
    var_94 = module_0.String()
    var_95 = module_2.Not(var_94)
    var_96 = module_1.to_json_schema(var_95)
    var_97 = module_0.String()
    var_98 = module_0.Integer()
    var_99 = module_0.Boolean()
    var_100 = module_2.IfThenElse(var_97, var_98, var_99)
    var_101 = module_1.to_json_schema(var_100)
    var_102 = module_0.String()
    var_103 = module_0.Integer()
    var_104 = module_2.IfThenElse(var_102, var_103)
    var_105 = module_1.to_json_schema(var_104)
    var_106 = 'default_value'
    var_107 = module_0.String()
    var_108 = module_1.to_json_schema(var_107)
    var_109 = 'email'
    var_110 = module_0.String(format=var_109)
    var_111 = module_1.to_json_schema(var_110)
    var_112 = module_0.String()
    var_113 = module_0.Integer()
    var_114 = [var_112, var_113]
    var_115 = module_0.Array(var_114)
    var_116 = module_1.to_json_schema(var_115)
    var_117 = 'items'
    var_118 = var_116[var_117]
    var_119 = var_116[var_117]
    var_120 = len(var_119)
    assert var_120 == 2
    var_121 = False
    var_122 = module_0.Array(additional_items=var_121)
    var_123 = module_1.to_json_schema(var_122)
    var_124 = module_0.String()
    var_125 = module_0.Array(additional_items=var_124)
    var_126 = module_1.to_json_schema(var_125)
    var_127 = module_0.String(pattern=var_11)
    var_128 = module_0.Object(property_names=var_127)



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
    var_4 = False
    var_5 = 1
    var_6 = 10
    var_7 = '^test'
    var_8 = module_0.String(max_length=var_6, min_length=var_5, pattern=var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = True
    var_11 = module_0.String()
    var_12 = module_1.to_json_schema(var_11)
    var_13 = 100
    var_14 = module_0.Integer(minimum=var_4, maximum=var_13)
    var_15 = module_1.to_json_schema(var_14)
    var_16 = 0.1
    var_17 = module_0.Float(minimum=var_4, maximum=var_10, multiple_of=var_16)
    var_18 = module_1.to_json_schema(var_17)
    var_19 = module_0.Boolean()
    var_20 = module_1.to_json_schema(var_19)
    var_21 = True
    var_22 = module_0.Boolean()
    var_23 = module_1.to_json_schema(var_22)
    var_24 = module_0.String()
    var_25 = 5
    var_26 = module_0.Array(var_24, min_items=var_21, max_items=var_25)
    var_27 = module_1.to_json_schema(var_26)
    var_28 = module_0.Integer()
    var_29 = True
    var_30 = module_0.Array(var_28, unique_items=var_29)
    var_31 = module_1.to_json_schema(var_30)
    var_32 = 'name'
    var_33 = 'age'
    var_34 = module_0.String()
    var_35 = module_0.Integer()
    var_36 = {var_32: var_34, var_33: var_35}
    var_37 = [var_32]
    var_38 = module_0.Object(properties=var_36, required=var_37)
    var_39 = module_1.to_json_schema(var_38)
    var_40 = 'id'
    var_41 = module_0.Integer()
    var_42 = {var_40: var_41}
    var_43 = True
    var_44 = module_0.Object(properties=var_42, additional_properties=var_43)
    var_45 = module_1.to_json_schema(var_44)
    var_46 = 'a'
    var_47 = 'A'
    var_48 = (var_46, var_47)
    var_49 = 'b'
    var_50 = 'B'
    var_51 = (var_49, var_50)
    var_52 = [var_48, var_51]
    var_53 = module_0.Choice(choices=var_52)
    var_54 = module_1.to_json_schema(var_53)
    var_55 = 'constant_value'
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
    var_74 = module_0.String(min_length=var_43)
    var_75 = module_0.String(max_length=var_13)
    var_76 = [var_74, var_75]
    var_77 = module_2.AllOf(var_76)
    var_78 = module_1.to_json_schema(var_77)
    var_79 = 'allOf'
    var_80 = var_78[var_79]
    var_81 = len(var_80)
    assert var_81 == 2
    var_82 = module_0.String()
    var_83 = module_2.Not(var_82)
    var_84 = module_1.to_json_schema(var_83)
    var_85 = (var_46, var_47)
    var_86 = [var_85]
    var_87 = module_0.Choice(choices=var_86)
    var_88 = module_0.String()
    var_89 = module_0.Integer()
    var_90 = module_2.IfThenElse(var_87, var_88, var_89)
    var_91 = module_1.to_json_schema(var_90)
    var_92 = (var_46, var_47)
    var_93 = [var_92]
    var_94 = module_0.Choice(choices=var_93)
    var_95 = module_0.String()
    var_96 = module_2.IfThenElse(var_94, var_95)
    var_97 = module_1.to_json_schema(var_96)
    var_98 = 'test_default'
    var_99 = module_0.String()
    var_100 = module_1.to_json_schema(var_99)
    var_101 = 'default'
    var_102 = 'TestSchema'
    var_103 = module_0.String()
    var_104 = {var_102: var_103}
    var_105 = module_3.Reference(var_102, var_104)
    var_106 = module_1.to_json_schema(var_105)
    var_107 = module_0.String()
    var_108 = module_0.Integer()
    var_109 = {var_32: var_107, var_33: var_108}
    var_110 = [var_32]
    var_111 = module_3.Schema(var_109)
    var_112 = module_1.to_json_schema(var_111)
    var_113 = 'StringDef'
    var_114 = 'IntegerDef'
    var_115 = module_0.String()
    var_116 = module_0.Integer()
    var_117 = {var_113: var_115, var_114: var_116}
    var_118 = module_1.to_json_schema(var_117)
    var_119 = module_0.String()
    var_120 = module_0.Integer()
    var_121 = [var_119, var_120]
    var_122 = module_0.Array(var_121)
    var_123 = module_1.to_json_schema(var_122)
    var_124 = 'items'
    var_125 = var_123[var_124]
    var_126 = var_123[var_124]
    var_127 = len(var_126)
    assert var_127 == 2
    var_128 = '^S_'
    var_129 = '^I_'
    var_130 = module_0.String()
    var_131 = module_0.Integer()
    var_132 = {var_128: var_130, var_129: var_131}
    var_133 = module_0.Object(pattern_properties=var_132)
    var_134 = module_1.to_json_schema(var_133)
    var_135 = '^[a-z]+$'
    var_136 = module_0.String(pattern=var_135)
    var_137 = module_0.Object(property_names=var_136)
    var_138 = module_1.to_json_schema(var_137)
    var_139 = module_0.Float(exclusive_minimum=var_4, exclusive_maximum=var_13)
    var_140 = module_1.to_json_schema(var_139)



####################################################################
#    TEST GENERATION BEGINS (CODAMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# Parsed testcases at query #1
#--------------------------


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
    var_7 = 100
    var_8 = 0.5
    var_9 = 99.5
    var_10 = 5
    var_11 = 50
    var_12 = {var_0: var_6, var_1: var_7, var_2: var_8, var_3: var_9, var_4: var_10, var_5: var_11}
    var_13 = 'number'
    var_14 = False
    var_15 = module_0.Definitions()
    var_16 = module_1.from_json_schema_type(var_12, var_13, var_14, var_15)
    var_17 = 10
    var_18 = {var_5: var_17}
    var_19 = True
    var_20 = module_0.Definitions()
    var_21 = module_1.from_json_schema_type(var_18, var_13, var_19, var_20)
    var_22 = 11
    var_23 = 2
    var_24 = {var_0: var_19, var_1: var_17, var_2: var_14, var_3: var_22, var_4: var_23, var_5: var_10}
    var_25 = 'integer'
    var_26 = False
    var_27 = module_0.Definitions()
    var_28 = module_1.from_json_schema_type(var_24, var_25, var_26, var_27)
    var_29 = 'minLength'
    var_30 = 'maxLength'
    var_31 = 'pattern'
    var_32 = 'format'
    var_33 = '^[a-z]+$'
    var_34 = 'email'
    var_35 = 'test'
    var_36 = {var_29: var_23, var_30: var_11, var_31: var_33, var_32: var_34, var_5: var_35}
    var_37 = 'string'
    var_38 = False
    var_39 = module_0.Definitions()
    var_40 = module_1.from_json_schema_type(var_36, var_37, var_38, var_39)
    var_41 = {var_29: var_38}
    var_42 = False
    var_43 = module_0.Definitions()
    var_44 = module_1.from_json_schema_type(var_41, var_37, var_42, var_43)
    var_45 = {var_29: var_19}
    var_46 = False
    var_47 = module_0.Definitions()
    var_48 = module_1.from_json_schema_type(var_45, var_37, var_46, var_47)
    var_49 = {var_5: var_19}
    var_50 = 'boolean'
    var_51 = False
    var_52 = module_0.Definitions()
    var_53 = module_1.from_json_schema_type(var_49, var_50, var_51, var_52)
    var_54 = {}
    var_55 = module_0.Definitions()
    var_56 = module_1.from_json_schema_type(var_54, var_50, var_19, var_55)
    var_57 = 'minItems'
    var_58 = 'maxItems'
    var_59 = 'uniqueItems'
    var_60 = {var_57: var_51, var_58: var_17, var_59: var_19}
    var_61 = 'array'
    var_62 = False
    var_63 = module_0.Definitions()
    var_64 = module_1.from_json_schema_type(var_60, var_61, var_62, var_63)
    var_65 = 'items'
    var_66 = 'type'
    var_67 = {var_66: var_37}
    var_68 = {var_65: var_67, var_57: var_19}
    var_69 = False
    var_70 = module_0.Definitions()
    var_71 = module_1.from_json_schema_type(var_68, var_61, var_69, var_70)
    var_72 = var_71.items
    var_73 = {var_66: var_37}
    var_74 = {var_66: var_13}
    var_75 = [var_73, var_74]
    var_76 = {var_65: var_75}
    var_77 = False
    var_78 = module_0.Definitions()
    var_79 = module_1.from_json_schema_type(var_76, var_61, var_77, var_78)
    var_80 = var_79.items
    var_81 = var_79.items
    var_82 = len(var_81)
    assert var_82 == 2
    var_83 = 'additionalItems'
    var_84 = False
    var_85 = {var_83: var_84}
    var_86 = False
    var_87 = module_0.Definitions()
    var_88 = module_1.from_json_schema_type(var_85, var_61, var_86, var_87)
    var_89 = {var_66: var_37}
    var_90 = {var_83: var_89}
    var_91 = False
    var_92 = module_0.Definitions()
    var_93 = module_1.from_json_schema_type(var_90, var_61, var_91, var_92)
    var_94 = var_93.additional_items
    var_95 = {}
    var_96 = 'object'
    var_97 = False
    var_98 = module_0.Definitions()
    var_99 = module_1.from_json_schema_type(var_95, var_96, var_97, var_98)
    var_100 = 'properties'
    var_101 = 'name'
    var_102 = 'age'
    var_103 = {var_66: var_37}
    var_104 = {var_66: var_25}
    var_105 = {var_101: var_103, var_102: var_104}
    var_106 = {var_100: var_105}
    var_107 = False
    var_108 = module_0.Definitions()
    var_109 = module_1.from_json_schema_type(var_106, var_96, var_107, var_108)
    var_110 = var_109.properties
    var_111 = 'patternProperties'
    var_112 = '^S_'
    var_113 = '^I_'
    var_114 = {var_66: var_37}
    var_115 = {var_66: var_25}
    var_116 = {var_112: var_114, var_113: var_115}
    var_117 = {var_111: var_116}
    var_118 = False
    var_119 = module_0.Definitions()
    var_120 = module_1.from_json_schema_type(var_117, var_96, var_118, var_119)
    var_121 = var_120.pattern_properties
    var_122 = 'additionalProperties'
    var_123 = False
    var_124 = {var_122: var_123}
    var_125 = False
    var_126 = module_0.Definitions()
    var_127 = module_1.from_json_schema_type(var_124, var_96, var_125, var_126)
    var_128 = {var_66: var_37}
    var_129 = {var_122: var_128}
    var_130 = False
    var_131 = module_0.Definitions()
    var_132 = module_1.from_json_schema_type(var_129, var_96, var_130, var_131)
    var_133 = var_132.additional_properties
    var_134 = 'propertyNames'
    var_135 = {var_31: var_33}
    var_136 = {var_134: var_135}
    var_137 = False
    var_138 = module_0.Definitions()
    var_139 = module_1.from_json_schema_type(var_136, var_96, var_137, var_138)
    var_140 = var_139.property_names



# Parsed testcases at query #2
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = '#/components/schemas/MySchema'
    var_3 = {var_1: var_2}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)

import typesystem.schemas as module_0
import typesystem.fields as module_1
import typesystem.json_schema as module_2

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.String()
    var_3 = {var_1: var_2}
    var_4 = '$ref'
    var_5 = '#/components/schemas/User'
    var_6 = {var_4: var_5}
    var_7 = module_2.ref_from_json_schema(var_6, var_0)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = 'external.json#/definitions/MySchema'
    var_3 = {var_1: var_2}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = '#/definitions/components/schemas/ComplexSchema'
    var_3 = {var_1: var_2}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)

import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = '$ref'
    var_2 = '#/definitions/String'
    var_3 = {var_1: var_2}
    var_4 = module_1.ref_from_json_schema(var_3, var_0)



# Parsed testcases at query #3
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'minimum'
    var_1 = 'maximum'
    var_2 = 'multipleOf'
    var_3 = 0
    var_4 = 100
    var_5 = 5
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'number'
    var_8 = False
    var_9 = module_0.Definitions()
    var_10 = module_1.from_json_schema_type(var_6, var_7, var_8, var_9)
    var_11 = -10
    var_12 = 10
    var_13 = {var_0: var_11, var_1: var_12}
    var_14 = True
    var_15 = module_0.Definitions()
    var_16 = module_1.from_json_schema_type(var_13, var_7, var_14, var_15)
    var_17 = {var_0: var_14, var_1: var_4}
    var_18 = 'integer'
    var_19 = False
    var_20 = module_0.Definitions()
    var_21 = module_1.from_json_schema_type(var_17, var_18, var_19, var_20)
    var_22 = 'exclusiveMinimum'
    var_23 = 'exclusiveMaximum'
    var_24 = {var_22: var_19, var_23: var_4}
    var_25 = False
    var_26 = module_0.Definitions()
    var_27 = module_1.from_json_schema_type(var_24, var_18, var_25, var_26)
    var_28 = 'minLength'
    var_29 = 'maxLength'
    var_30 = 'pattern'
    var_31 = 50
    var_32 = '^[a-z]+$'
    var_33 = {var_28: var_14, var_29: var_31, var_30: var_32}
    var_34 = 'string'
    var_35 = False
    var_36 = module_0.Definitions()
    var_37 = module_1.from_json_schema_type(var_33, var_34, var_35, var_36)
    var_38 = {var_28: var_35}
    var_39 = False
    var_40 = module_0.Definitions()
    var_41 = module_1.from_json_schema_type(var_38, var_34, var_39, var_40)
    var_42 = 'format'
    var_43 = 'email'
    var_44 = {var_42: var_43}
    var_45 = False
    var_46 = module_0.Definitions()
    var_47 = module_1.from_json_schema_type(var_44, var_34, var_45, var_46)
    var_48 = {}
    var_49 = 'boolean'
    var_50 = False
    var_51 = module_0.Definitions()
    var_52 = module_1.from_json_schema_type(var_48, var_49, var_50, var_51)
    var_53 = {}
    var_54 = module_0.Definitions()
    var_55 = module_1.from_json_schema_type(var_53, var_49, var_14, var_54)
    var_56 = {}
    var_57 = 'array'
    var_58 = False
    var_59 = module_0.Definitions()
    var_60 = module_1.from_json_schema_type(var_56, var_57, var_58, var_59)
    var_61 = 'items'
    var_62 = 'minItems'
    var_63 = 'maxItems'
    var_64 = 'uniqueItems'
    var_65 = 'type'
    var_66 = {var_65: var_34}
    var_67 = {var_61: var_66, var_62: var_14, var_63: var_12, var_64: var_14}
    var_68 = False
    var_69 = module_0.Definitions()
    var_70 = module_1.from_json_schema_type(var_67, var_57, var_68, var_69)
    var_71 = var_70.items
    var_72 = {var_65: var_34}
    var_73 = {var_65: var_18}
    var_74 = [var_72, var_73]
    var_75 = {var_61: var_74}
    var_76 = False
    var_77 = module_0.Definitions()
    var_78 = module_1.from_json_schema_type(var_75, var_57, var_76, var_77)
    var_79 = var_78.items
    var_80 = var_78.items
    var_81 = len(var_80)
    assert var_81 == 2
    var_82 = 'additionalItems'
    var_83 = False
    var_84 = {var_82: var_83}
    var_85 = False
    var_86 = module_0.Definitions()
    var_87 = module_1.from_json_schema_type(var_84, var_57, var_85, var_86)
    var_88 = {var_65: var_7}
    var_89 = {var_82: var_88}
    var_90 = False
    var_91 = module_0.Definitions()
    var_92 = module_1.from_json_schema_type(var_89, var_57, var_90, var_91)
    var_93 = var_92.additional_items
    var_94 = {}
    var_95 = 'object'
    var_96 = False
    var_97 = module_0.Definitions()
    var_98 = module_1.from_json_schema_type(var_94, var_95, var_96, var_97)
    var_99 = 'properties'
    var_100 = 'name'
    var_101 = 'age'
    var_102 = {var_65: var_34}
    var_103 = {var_65: var_18}
    var_104 = {var_100: var_102, var_101: var_103}
    var_105 = {var_99: var_104}
    var_106 = False
    var_107 = module_0.Definitions()
    var_108 = module_1.from_json_schema_type(var_105, var_95, var_106, var_107)
    var_109 = 'patternProperties'
    var_110 = '^S_'
    var_111 = {var_65: var_34}
    var_112 = {var_110: var_111}
    var_113 = {var_109: var_112}
    var_114 = False
    var_115 = module_0.Definitions()
    var_116 = module_1.from_json_schema_type(var_113, var_95, var_114, var_115)
    var_117 = 'additionalProperties'
    var_118 = False
    var_119 = {var_117: var_118}
    var_120 = False
    var_121 = module_0.Definitions()
    var_122 = module_1.from_json_schema_type(var_119, var_95, var_120, var_121)
    var_123 = {var_65: var_34}
    var_124 = {var_117: var_123}
    var_125 = False
    var_126 = module_0.Definitions()
    var_127 = module_1.from_json_schema_type(var_124, var_95, var_125, var_126)
    var_128 = var_127.additional_properties
    var_129 = 'propertyNames'
    var_130 = '^[a-z_]+$'
    var_131 = {var_30: var_130}
    var_132 = {var_129: var_131}
    var_133 = False
    var_134 = module_0.Definitions()
    var_135 = module_1.from_json_schema_type(var_132, var_95, var_133, var_134)
    var_136 = 'minProperties'
    var_137 = 'maxProperties'
    var_138 = {var_136: var_14, var_137: var_5}
    var_139 = False
    var_140 = module_0.Definitions()
    var_141 = module_1.from_json_schema_type(var_138, var_95, var_139, var_140)
    var_142 = 'required'
    var_143 = [var_100, var_101]
    var_144 = {var_142: var_143}
    var_145 = False
    var_146 = module_0.Definitions()
    var_147 = module_1.from_json_schema_type(var_144, var_95, var_145, var_146)
    var_148 = 'default'
    var_149 = 'test_value'
    var_150 = {var_148: var_149}
    var_151 = False
    var_152 = module_0.Definitions()
    var_153 = module_1.from_json_schema_type(var_150, var_34, var_151, var_152)



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'enum'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'red'
    var_7 = 'green'
    var_8 = 'blue'
    var_9 = [var_6, var_7, var_8]
    var_10 = {var_0: var_9}
    var_11 = 'two'
    var_12 = None
    var_13 = True
    var_14 = [var_1, var_11, var_12, var_13]
    var_15 = {var_0: var_14}
    var_16 = 'default'
    var_17 = 10
    var_18 = 20
    var_19 = 30
    var_20 = [var_17, var_18, var_19]
    var_21 = {var_0: var_20, var_16: var_18}
    var_22 = 'only'
    var_23 = [var_22]
    var_24 = {var_0: var_23}
    var_25 = ''
    var_26 = 'a'
    var_27 = 'b'
    var_28 = [var_25, var_26, var_27]
    var_29 = {var_0: var_28}
    var_30 = 0
    var_31 = [var_30, var_13, var_2]
    var_32 = {var_0: var_31}
    var_33 = False
    var_34 = True
    var_35 = [var_33, var_34]
    var_36 = {var_0: var_35}
    var_37 = 1.5
    var_38 = 2.5
    var_39 = 3.5
    var_40 = [var_37, var_38, var_39]
    var_41 = {var_0: var_40}



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'type'
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = 'integer'
    var_4 = [var_1, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'null'
    var_7 = [var_1, var_6]
    var_8 = {var_0: var_7}
    var_9 = {var_0: var_6}
    var_10 = set()
    var_11 = {}
    var_12 = 'number'
    var_13 = [var_12, var_3]
    var_14 = {var_0: var_13}
    var_15 = [var_12, var_3, var_6]
    var_16 = {var_0: var_15}
    var_17 = []
    var_18 = {var_0: var_17}
    var_19 = 'object'
    var_20 = {var_0: var_19}
    var_21 = 'array'
    var_22 = [var_21, var_6]
    var_23 = {var_0: var_22}
    var_24 = 'boolean'
    var_25 = {var_0: var_24}



# Parsed testcases at query #6
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

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
    var_18 = 'number'
    var_19 = {var_1: var_18}
    var_20 = [var_17, var_19]
    var_21 = 'test_default'
    var_22 = {var_0: var_20, var_16: var_21}
    var_23 = module_0.Definitions()
    var_24 = module_1.any_of_from_json_schema(var_22, var_23)
    var_25 = 'properties'
    var_26 = 'object'
    var_27 = 'name'
    var_28 = {var_1: var_2}
    var_29 = {var_27: var_28}
    var_30 = {var_1: var_26, var_25: var_29}
    var_31 = 'items'
    var_32 = 'array'
    var_33 = {var_1: var_4}
    var_34 = {var_1: var_32, var_31: var_33}
    var_35 = [var_30, var_34]
    var_36 = {var_0: var_35}
    var_37 = module_0.Definitions()
    var_38 = module_1.any_of_from_json_schema(var_36, var_37)
    var_39 = var_38.any_of
    var_40 = len(var_39)
    assert var_40 == 2
    var_41 = var_38.any_of[var_12]
    var_42 = var_38.any_of[var_14]
    var_43 = 'boolean'
    var_44 = {var_1: var_43}
    var_45 = [var_44]
    var_46 = {var_0: var_45}
    var_47 = module_0.Definitions()
    var_48 = module_1.any_of_from_json_schema(var_46, var_47)
    var_49 = var_48.any_of
    var_50 = len(var_49)
    assert var_50 == 1
    var_51 = var_48.any_of[var_12]
    var_52 = {var_1: var_2}
    var_53 = 'null'
    var_54 = {var_1: var_53}
    var_55 = [var_52, var_54]
    var_56 = {var_0: var_55}
    var_57 = module_0.Definitions()
    var_58 = module_1.any_of_from_json_schema(var_56, var_57)



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'enum'
    var_1 = 'red'
    var_2 = 'green'
    var_3 = 'blue'
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]
    var_10 = {var_0: var_9}
    var_11 = 'a'
    var_12 = True
    var_13 = None
    var_14 = [var_11, var_6, var_12, var_13]
    var_15 = {var_0: var_14}
    var_16 = 'default'
    var_17 = 'option1'
    var_18 = 'option2'
    var_19 = 'option3'
    var_20 = [var_17, var_18, var_19]
    var_21 = {var_0: var_20, var_16: var_18}
    var_22 = 'only_option'
    var_23 = [var_22]
    var_24 = {var_0: var_23}
    var_25 = True
    var_26 = False
    var_27 = [var_25, var_26]
    var_28 = {var_0: var_27}
    var_29 = 1.5
    var_30 = 2.5
    var_31 = 3.5
    var_32 = [var_29, var_30, var_31]
    var_33 = {var_0: var_32}
    var_34 = 'value'
    var_35 = [var_13, var_34]
    var_36 = {var_0: var_35, var_16: var_13}



# Parsed testcases at query #8
#--------------------------


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
    var_11 = 'test_default'
    var_12 = {var_0: var_6, var_1: var_8, var_2: var_10, var_3: var_11}
    var_13 = {var_4: var_5}
    var_14 = {var_4: var_7}
    var_15 = {var_0: var_13, var_1: var_14}
    var_16 = {var_4: var_5}
    var_17 = {var_4: var_9}
    var_18 = {var_0: var_16, var_2: var_17}
    var_19 = {var_4: var_5}
    var_20 = {var_0: var_19}
    var_21 = {var_4: var_7}
    var_22 = {var_4: var_5}
    var_23 = {var_4: var_9}
    var_24 = {var_0: var_21, var_1: var_22, var_2: var_23}
    var_25 = 'properties'
    var_26 = 'object'
    var_27 = 'name'
    var_28 = {var_4: var_5}
    var_29 = {var_27: var_28}
    var_30 = {var_4: var_26, var_25: var_29}
    var_31 = 'items'
    var_32 = 'array'
    var_33 = {var_4: var_7}
    var_34 = {var_4: var_32, var_31: var_33}
    var_35 = 'enum'
    var_36 = 1
    var_37 = 2
    var_38 = 3
    var_39 = [var_36, var_37, var_38]
    var_40 = {var_35: var_39}
    var_41 = 42
    var_42 = {var_0: var_30, var_1: var_34, var_2: var_40, var_3: var_41}



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'enum'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_0: var_4}
    var_6 = 'red'
    var_7 = 'green'
    var_8 = 'blue'
    var_9 = [var_6, var_7, var_8]
    var_10 = {var_0: var_9}
    var_11 = 'two'
    var_12 = True
    var_13 = None
    var_14 = [var_1, var_11, var_3, var_12, var_13]
    var_15 = {var_0: var_14}
    var_16 = 'default'
    var_17 = 'a'
    var_18 = 'b'
    var_19 = 'c'
    var_20 = [var_17, var_18, var_19]
    var_21 = {var_0: var_20, var_16: var_18}
    var_22 = 42
    var_23 = [var_22]
    var_24 = {var_0: var_23}
    var_25 = True
    var_26 = False
    var_27 = [var_25, var_26]
    var_28 = {var_0: var_27}
    var_29 = '1'
    var_30 = '2'
    var_31 = '3'
    var_32 = [var_29, var_30, var_31]
    var_33 = {var_0: var_32}
    var_34 = 10
    var_35 = 20
    var_36 = 30
    var_37 = [var_34, var_35, var_36]
    var_38 = {var_0: var_37, var_16: var_35}
    var_39 = 'value'
    var_40 = [var_13, var_39]
    var_41 = {var_0: var_40}



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'allOf'
    var_1 = 'type'
    var_2 = 'minLength'
    var_3 = 'string'
    var_4 = 1
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'maxLength'
    var_7 = 10
    var_8 = {var_1: var_3, var_6: var_7}
    var_9 = [var_5, var_8]
    var_10 = {var_0: var_9}
    var_11 = 0
    var_12 = 'default'
    var_13 = 'minimum'
    var_14 = 'integer'
    var_15 = {var_1: var_14, var_13: var_11}
    var_16 = 'maximum'
    var_17 = 100
    var_18 = {var_1: var_14, var_16: var_17}
    var_19 = [var_15, var_18]
    var_20 = 50
    var_21 = {var_0: var_19, var_12: var_20}
    var_22 = 'multipleOf'
    var_23 = 'number'
    var_24 = 2
    var_25 = {var_1: var_23, var_22: var_24}
    var_26 = [var_25]
    var_27 = {var_0: var_26}
    var_28 = 'properties'
    var_29 = 'object'
    var_30 = 'name'
    var_31 = {var_1: var_3}
    var_32 = {var_30: var_31}
    var_33 = {var_1: var_29, var_28: var_32}
    var_34 = 'required'
    var_35 = [var_30]
    var_36 = {var_1: var_29, var_34: var_35}
    var_37 = [var_33, var_36]
    var_38 = {var_0: var_37}
    var_39 = 'boolean'
    var_40 = {var_1: var_39}
    var_41 = [var_40]
    var_42 = {var_0: var_41}
    var_43 = 'items'
    var_44 = 'array'
    var_45 = {var_1: var_3}
    var_46 = {var_1: var_44, var_43: var_45}
    var_47 = 'minItems'
    var_48 = {var_47: var_4}
    var_49 = [var_46, var_48]
    var_50 = {var_0: var_49}



# Parsed testcases at query #11
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'allOf'
    var_1 = 'type'
    var_2 = 'minLength'
    var_3 = 'string'
    var_4 = 1
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'maxLength'
    var_7 = 10
    var_8 = {var_6: var_7}
    var_9 = [var_5, var_8]
    var_10 = {var_0: var_9}
    var_11 = 'default'
    var_12 = 'minimum'
    var_13 = 'integer'
    var_14 = 0
    var_15 = {var_1: var_13, var_12: var_14}
    var_16 = 'maximum'
    var_17 = 100
    var_18 = {var_16: var_17}
    var_19 = [var_15, var_18]
    var_20 = 50
    var_21 = {var_0: var_19, var_11: var_20}
    var_22 = 'boolean'
    var_23 = {var_1: var_22}
    var_24 = [var_23]
    var_25 = {var_0: var_24}
    var_26 = 'properties'
    var_27 = 'object'
    var_28 = 'name'
    var_29 = {var_1: var_3}
    var_30 = {var_28: var_29}
    var_31 = {var_1: var_27, var_26: var_30}
    var_32 = 'required'
    var_33 = [var_28]
    var_34 = {var_32: var_33}
    var_35 = 'maxProperties'
    var_36 = 5
    var_37 = {var_35: var_36}
    var_38 = [var_31, var_34, var_37]
    var_39 = {var_0: var_38}
    var_40 = 'number'
    var_41 = {var_1: var_40}
    var_42 = {var_12: var_14}
    var_43 = [var_41, var_42]
    var_44 = {var_0: var_43}
    var_45 = module_0.Definitions()
    var_46 = '$ref'
    var_47 = '#/components/schemas/StringType'
    var_48 = {var_46: var_47}
    var_49 = {var_2: var_36}
    var_50 = [var_48, var_49]
    var_51 = {var_0: var_50}
    var_52 = module_1.all_of_from_json_schema(var_51, var_45)
    var_53 = var_52.all_of
    var_54 = len(var_53)
    assert var_54 == 2



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
    var_12 = module_0.String()
    var_13 = module_1.to_json_schema(var_12)
    var_14 = True
    var_15 = None
    var_16 = module_0.String(allow_blank=var_14, min_length=var_15)
    var_17 = module_1.to_json_schema(var_16)
    var_18 = 100
    var_19 = 99
    var_20 = 5
    var_21 = module_0.Integer(minimum=var_4, maximum=var_18, exclusive_minimum=var_14, exclusive_maximum=var_19, multiple_of=var_20)
    var_22 = module_1.to_json_schema(var_21)
    var_23 = module_0.Float(minimum=var_4, maximum=var_14)
    var_24 = module_1.to_json_schema(var_23)
    var_25 = module_0.Boolean()
    var_26 = module_1.to_json_schema(var_25)
    var_27 = True
    var_28 = module_0.Boolean()
    var_29 = module_1.to_json_schema(var_28)
    var_30 = module_0.String()
    var_31 = True
    var_32 = module_0.Array(var_30, min_items=var_27, max_items=var_6, unique_items=var_31)
    var_33 = module_1.to_json_schema(var_32)
    var_34 = module_0.String()
    var_35 = module_0.Integer()
    var_36 = [var_34, var_35]
    var_37 = module_0.Array(var_36)
    var_38 = module_1.to_json_schema(var_37)
    var_39 = 'items'
    var_40 = var_38[var_39]
    var_41 = var_38[var_39]
    var_42 = len(var_41)
    assert var_42 == 2
    var_43 = module_0.Array(additional_items=var_4)
    var_44 = module_1.to_json_schema(var_43)
    var_45 = module_0.String()
    var_46 = module_0.Array(additional_items=var_45)
    var_47 = module_1.to_json_schema(var_46)
    var_48 = 'additionalItems'
    var_49 = var_47[var_48]
    var_50 = 'name'
    var_51 = 'age'
    var_52 = module_0.String()
    var_53 = module_0.Integer()
    var_54 = {var_50: var_52, var_51: var_53}
    var_55 = [var_50]
    var_56 = module_0.Object(properties=var_54, min_properties=var_31, max_properties=var_6, required=var_55)
    var_57 = module_1.to_json_schema(var_56)
    var_58 = '^S_'
    var_59 = '^I_'
    var_60 = module_0.String()
    var_61 = module_0.Integer()
    var_62 = {var_58: var_60, var_59: var_61}
    var_63 = module_0.Object(pattern_properties=var_62)
    var_64 = module_1.to_json_schema(var_63)
    var_65 = module_0.Object(additional_properties=var_4)
    var_66 = module_1.to_json_schema(var_65)
    var_67 = module_0.String()
    var_68 = module_0.Object(additional_properties=var_67)
    var_69 = module_1.to_json_schema(var_68)
    var_70 = 'additionalProperties'
    var_71 = var_69[var_70]
    var_72 = module_0.String(pattern=var_7)
    var_73 = module_0.Object(property_names=var_72)
    var_74 = module_1.to_json_schema(var_73)
    var_75 = 'a'
    var_76 = 'Option A'
    var_77 = (var_75, var_76)
    var_78 = 'b'
    var_79 = 'Option B'
    var_80 = (var_78, var_79)
    var_81 = [var_77, var_80]
    var_82 = module_0.Choice(choices=var_81)
    var_83 = module_1.to_json_schema(var_82)
    var_84 = 'fixed_value'
    var_85 = module_0.Const(var_84)
    var_86 = module_1.to_json_schema(var_85)
    var_87 = module_0.String()
    var_88 = module_0.Integer()
    var_89 = [var_87, var_88]
    var_90 = module_0.Union(var_89)
    var_91 = module_1.to_json_schema(var_90)
    var_92 = 'anyOf'
    var_93 = var_91[var_92]
    var_94 = len(var_93)
    assert var_94 == 2
    var_95 = module_0.String()
    var_96 = module_0.Integer()
    var_97 = [var_95, var_96]
    var_98 = module_2.OneOf(var_97)
    var_99 = module_1.to_json_schema(var_98)
    var_100 = 'oneOf'
    var_101 = var_99[var_100]
    var_102 = len(var_101)
    assert var_102 == 2
    var_103 = module_0.String()
    var_104 = module_0.String(min_length=var_20)
    var_105 = [var_103, var_104]
    var_106 = module_2.AllOf(var_105)
    var_107 = module_1.to_json_schema(var_106)
    var_108 = 'allOf'
    var_109 = var_107[var_108]
    var_110 = len(var_109)
    assert var_110 == 2
    var_111 = module_0.String()
    var_112 = module_2.Not(var_111)
    var_113 = module_1.to_json_schema(var_112)
    var_114 = module_0.String()
    var_115 = module_0.Integer()
    var_116 = module_0.Boolean()
    var_117 = module_2.IfThenElse(var_114, var_115, var_116)
    var_118 = module_1.to_json_schema(var_117)
    var_119 = module_0.String()
    var_120 = module_0.Integer()
    var_121 = module_2.IfThenElse(var_119, var_120, var_15)
    var_122 = module_1.to_json_schema(var_121)
    var_123 = module_0.String()
    var_124 = module_0.Integer()
    var_125 = {var_50: var_123, var_51: var_124}
    var_126 = [var_50]
    var_127 = module_3.Schema(var_125)
    var_128 = module_1.to_json_schema(var_127)



# Parsed testcases at query #13
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'oneOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'integer'
    var_5 = {var_1: var_4}
    var_6 = [var_3, var_5]
    var_7 = {var_0: var_6}
    var_8 = 0
    var_9 = 1
    var_10 = 'default'
    var_11 = {var_1: var_2}
    var_12 = 'number'
    var_13 = {var_1: var_12}
    var_14 = [var_11, var_13]
    var_15 = 'test'
    var_16 = {var_0: var_14, var_10: var_15}
    var_17 = 'properties'
    var_18 = 'object'
    var_19 = 'name'
    var_20 = {var_1: var_2}
    var_21 = {var_19: var_20}
    var_22 = {var_1: var_18, var_17: var_21}
    var_23 = 'items'
    var_24 = 'array'
    var_25 = {var_1: var_4}
    var_26 = {var_1: var_24, var_23: var_25}
    var_27 = [var_22, var_26]
    var_28 = {var_0: var_27}
    var_29 = 'boolean'
    var_30 = {var_1: var_29}
    var_31 = [var_30]
    var_32 = {var_0: var_31}
    var_33 = module_0.Definitions()
    var_34 = '$ref'
    var_35 = '#/definitions/StringType'
    var_36 = {var_34: var_35}
    var_37 = 'null'
    var_38 = {var_1: var_37}
    var_39 = [var_36, var_38]
    var_40 = {var_0: var_39}
    var_41 = module_1.one_of_from_json_schema(var_40, var_33)
    var_42 = var_41.one_of
    var_43 = len(var_42)
    assert var_43 == 2
    var_44 = var_41.one_of[var_8]
    var_45 = {var_1: var_2}
    var_46 = {var_1: var_4}
    var_47 = [var_45, var_46]
    var_48 = {var_0: var_47}



# Parsed testcases at query #14
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'oneOf'
    var_1 = 'type'
    var_2 = 'string'
    var_3 = {var_1: var_2}
    var_4 = 'integer'
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
    var_18 = 'number'
    var_19 = {var_1: var_18}
    var_20 = [var_17, var_19]
    var_21 = 'test'
    var_22 = {var_0: var_20, var_16: var_21}
    var_23 = module_0.Definitions()
    var_24 = module_1.one_of_from_json_schema(var_22, var_23)
    var_25 = 'properties'
    var_26 = 'object'
    var_27 = 'name'
    var_28 = {var_1: var_2}
    var_29 = {var_27: var_28}
    var_30 = {var_1: var_26, var_25: var_29}
    var_31 = 'items'
    var_32 = 'array'
    var_33 = {var_1: var_4}
    var_34 = {var_1: var_32, var_31: var_33}
    var_35 = [var_30, var_34]
    var_36 = {var_0: var_35}
    var_37 = module_0.Definitions()
    var_38 = module_1.one_of_from_json_schema(var_36, var_37)
    var_39 = var_38.one_of
    var_40 = len(var_39)
    assert var_40 == 2
    var_41 = var_38.one_of[var_12]
    var_42 = var_38.one_of[var_14]
    var_43 = 'boolean'
    var_44 = {var_1: var_43}
    var_45 = [var_44]
    var_46 = {var_0: var_45}
    var_47 = module_0.Definitions()
    var_48 = module_1.one_of_from_json_schema(var_46, var_47)
    var_49 = var_48.one_of
    var_50 = len(var_49)
    assert var_50 == 1
    var_51 = var_48.one_of[var_12]
    var_52 = module_0.Definitions()
    var_53 = '$ref'
    var_54 = '#/definitions/StringType'
    var_55 = {var_53: var_54}
    var_56 = {var_1: var_4}
    var_57 = [var_55, var_56]
    var_58 = {var_0: var_57}
    var_59 = module_1.one_of_from_json_schema(var_58, var_52)
    var_60 = var_59.one_of
    var_61 = len(var_60)
    assert var_61 == 2
    var_62 = var_59.one_of[var_12]
    var_63 = var_59.one_of[var_14]
    var_64 = {var_1: var_2}
    var_65 = {var_1: var_18}
    var_66 = [var_64, var_65]
    var_67 = {var_0: var_66}
    var_68 = module_0.Definitions()
    var_69 = module_1.one_of_from_json_schema(var_67, var_68)
    var_70 = 'enum'
    var_71 = 2
    var_72 = 3
    var_73 = [var_14, var_71, var_72]
    var_74 = {var_70: var_73}
    var_75 = 'const'
    var_76 = 'fixed'
    var_77 = {var_75: var_76}
    var_78 = [var_74, var_77]
    var_79 = {var_0: var_78}
    var_80 = module_0.Definitions()
    var_81 = module_1.one_of_from_json_schema(var_79, var_80)
    var_82 = var_81.one_of
    var_83 = len(var_82)
    assert var_83 == 2
    var_84 = var_81.one_of[var_12]
    var_85 = var_81.one_of[var_14]



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'if'
    var_1 = 'then'
    var_2 = 'else'
    var_3 = 'type'
    var_4 = 'string'
    var_5 = {var_3: var_4}
    var_6 = 'integer'
    var_7 = {var_3: var_6}
    var_8 = 'boolean'
    var_9 = {var_3: var_8}
    var_10 = {var_0: var_5, var_1: var_7, var_2: var_9}
    var_11 = {var_3: var_4}
    var_12 = {var_3: var_6}
    var_13 = {var_0: var_11, var_1: var_12}
    var_14 = {var_3: var_4}
    var_15 = {var_3: var_8}
    var_16 = {var_0: var_14, var_2: var_15}
    var_17 = {var_3: var_4}
    var_18 = {var_0: var_17}
    var_19 = 'default'
    var_20 = {var_3: var_4}
    var_21 = {var_3: var_6}
    var_22 = {var_3: var_8}
    var_23 = 'test_default'
    var_24 = {var_0: var_20, var_1: var_21, var_2: var_22, var_19: var_23}
    var_25 = 'properties'
    var_26 = 'object'
    var_27 = 'name'
    var_28 = {var_3: var_4}
    var_29 = {var_27: var_28}
    var_30 = {var_3: var_26, var_25: var_29}
    var_31 = 'items'
    var_32 = 'array'
    var_33 = {var_3: var_6}
    var_34 = {var_3: var_32, var_31: var_33}
    var_35 = 'enum'
    var_36 = 1
    var_37 = 2
    var_38 = 3
    var_39 = [var_36, var_37, var_38]
    var_40 = {var_35: var_39}
    var_41 = {var_0: var_30, var_1: var_34, var_2: var_40}



# Parsed testcases at query #16
#--------------------------


import typesystem.schemas as module_0
import typesystem.json_schema as module_1

def test_case_0():
    var_0 = 'Test if_then_else_from_json_schema function.'
    var_1 = module_0.Definitions()
    var_2 = 'if'
    var_3 = 'then'
    var_4 = 'else'
    var_5 = 'type'
    var_6 = 'string'
    var_7 = {var_5: var_6}
    var_8 = 'integer'
    var_9 = {var_5: var_8}
    var_10 = 'boolean'
    var_11 = {var_5: var_10}
    var_12 = {var_2: var_7, var_3: var_9, var_4: var_11}
    var_13 = module_1.if_then_else_from_json_schema(var_12, var_1)
    var_14 = {var_5: var_6}
    var_15 = {var_5: var_8}
    var_16 = {var_2: var_14, var_3: var_15}
    var_17 = module_1.if_then_else_from_json_schema(var_16, var_1)
    var_18 = {var_5: var_6}
    var_19 = {var_5: var_10}
    var_20 = {var_2: var_18, var_4: var_19}
    var_21 = module_1.if_then_else_from_json_schema(var_20, var_1)
    var_22 = {var_5: var_6}
    var_23 = {var_2: var_22}
    var_24 = module_1.if_then_else_from_json_schema(var_23, var_1)
    var_25 = 'default'
    var_26 = {var_5: var_6}
    var_27 = {var_5: var_8}
    var_28 = 42
    var_29 = {var_2: var_26, var_3: var_27, var_25: var_28}
    var_30 = module_1.if_then_else_from_json_schema(var_29, var_1)
    var_31 = 'properties'
    var_32 = 'object'
    var_33 = 'name'
    var_34 = {var_5: var_6}
    var_35 = {var_33: var_34}
    var_36 = {var_5: var_32, var_31: var_35}
    var_37 = 'items'
    var_38 = 'array'
    var_39 = {var_5: var_8}
    var_40 = {var_5: var_38, var_37: var_39}
    var_41 = 'enum'
    var_42 = None
    var_43 = 'unknown'
    var_44 = [var_42, var_43]
    var_45 = {var_41: var_44}
    var_46 = {var_2: var_36, var_3: var_40, var_4: var_45}
    var_47 = module_1.if_then_else_from_json_schema(var_46, var_1)



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'Test if_then_else_from_json_schema function.'
    var_1 = 'if'
    var_2 = 'then'
    var_3 = 'else'
    var_4 = 'type'
    var_5 = 'string'
    var_6 = {var_4: var_5}
    var_7 = 'minLength'
    var_8 = 5
    var_9 = {var_7: var_8}
    var_10 = 'number'
    var_11 = {var_4: var_10}
    var_12 = {var_1: var_6, var_2: var_9, var_3: var_11}
    var_13 = {var_4: var_5}
    var_14 = {var_7: var_8}
    var_15 = {var_1: var_13, var_2: var_14}
    var_16 = {var_4: var_5}
    var_17 = {var_4: var_10}
    var_18 = {var_1: var_16, var_3: var_17}
    var_19 = {var_4: var_5}
    var_20 = {var_1: var_19}
    var_21 = 'default'
    var_22 = {var_4: var_5}
    var_23 = {var_7: var_8}
    var_24 = {var_4: var_10}
    var_25 = 'test_default'
    var_26 = {var_1: var_22, var_2: var_23, var_3: var_24, var_21: var_25}
    var_27 = 'properties'
    var_28 = 'object'
    var_29 = 'name'
    var_30 = {var_4: var_5}
    var_31 = {var_29: var_30}
    var_32 = {var_4: var_28, var_27: var_31}
    var_33 = 'age'
    var_34 = 'integer'
    var_35 = {var_4: var_34}
    var_36 = {var_33: var_35}
    var_37 = {var_27: var_36}
    var_38 = 'items'
    var_39 = 'array'
    var_40 = {var_4: var_5}
    var_41 = {var_4: var_39, var_38: var_40}
    var_42 = {var_1: var_32, var_2: var_37, var_3: var_41}
    var_43 = True
    var_44 = False
    var_45 = {var_1: var_43, var_2: var_44, var_3: var_43}



# Parsed testcases at query #18
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
    var_7 = 'minLength'
    var_8 = 5
    var_9 = {var_7: var_8}
    var_10 = 'number'
    var_11 = {var_4: var_10}
    var_12 = 'test'
    var_13 = {var_0: var_6, var_1: var_9, var_2: var_11, var_3: var_12}
    var_14 = 'object'
    var_15 = {var_4: var_14}
    var_16 = 'properties'
    var_17 = 'name'
    var_18 = {var_4: var_5}
    var_19 = {var_17: var_18}
    var_20 = {var_16: var_19}
    var_21 = {var_0: var_15, var_1: var_20}
    var_22 = 'array'
    var_23 = {var_4: var_22}
    var_24 = {var_4: var_5}
    var_25 = {var_0: var_23, var_2: var_24}
    var_26 = 'boolean'
    var_27 = {var_4: var_26}
    var_28 = {var_0: var_27}
    var_29 = 'const'
    var_30 = 'admin'
    var_31 = {var_29: var_30}
    var_32 = {var_4: var_31}
    var_33 = {var_16: var_32}
    var_34 = 'required'
    var_35 = 'permissions'
    var_36 = [var_35]
    var_37 = {var_34: var_36}
    var_38 = 'username'
    var_39 = [var_38]
    var_40 = {var_34: var_39}
    var_41 = None
    var_42 = {var_0: var_33, var_1: var_37, var_2: var_40, var_3: var_41}
    var_43 = module_0.Definitions()
    var_44 = '$ref'
    var_45 = '#/definitions/StringType'
    var_46 = {var_44: var_45}
    var_47 = '#/definitions/LongString'
    var_48 = {var_44: var_47}
    var_49 = '#/definitions/NumberType'
    var_50 = {var_44: var_49}
    var_51 = {var_0: var_46, var_1: var_48, var_2: var_50}
    var_52 = module_1.if_then_else_from_json_schema(var_51, var_43)



# Parsed testcases at query #19
#--------------------------


import typesystem.json_schema as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.from_json_schema(var_0)
    var_2 = False
    var_3 = module_0.from_json_schema(var_2)
    var_4 = {}
    var_5 = module_0.from_json_schema(var_4)
    var_6 = 'type'
    var_7 = 'string'
    var_8 = {var_6: var_7}
    var_9 = module_0.from_json_schema(var_8)
    var_10 = 'integer'
    var_11 = {var_6: var_10}
    var_12 = module_0.from_json_schema(var_11)
    var_13 = 'number'
    var_14 = {var_6: var_13}
    var_15 = module_0.from_json_schema(var_14)
    var_16 = 'boolean'
    var_17 = {var_6: var_16}
    var_18 = module_0.from_json_schema(var_17)
    var_19 = 'array'
    var_20 = {var_6: var_19}
    var_21 = module_0.from_json_schema(var_20)
    var_22 = 'object'
    var_23 = {var_6: var_22}
    var_24 = module_0.from_json_schema(var_23)
    var_25 = 'enum'
    var_26 = 2
    var_27 = 3
    var_28 = [var_0, var_26, var_27]
    var_29 = {var_25: var_28}
    var_30 = module_0.from_json_schema(var_29)
    var_31 = 'const'
    var_32 = 'value'
    var_33 = {var_31: var_32}
    var_34 = module_0.from_json_schema(var_33)
    var_35 = 'allOf'
    var_36 = {var_6: var_7}
    var_37 = 'minLength'
    var_38 = {var_37: var_0}
    var_39 = [var_36, var_38]
    var_40 = {var_35: var_39}
    var_41 = module_0.from_json_schema(var_40)
    var_42 = 'anyOf'
    var_43 = {var_6: var_7}
    var_44 = {var_6: var_10}
    var_45 = [var_43, var_44]
    var_46 = {var_42: var_45}
    var_47 = module_0.from_json_schema(var_46)
    var_48 = 'oneOf'
    var_49 = {var_6: var_7}
    var_50 = {var_6: var_10}
    var_51 = [var_49, var_50]
    var_52 = {var_48: var_51}
    var_53 = module_0.from_json_schema(var_52)
    var_54 = 'not'
    var_55 = {var_6: var_7}
    var_56 = {var_54: var_55}
    var_57 = module_0.from_json_schema(var_56)
    var_58 = 'if'
    var_59 = 'then'
    var_60 = {var_6: var_7}
    var_61 = {var_37: var_0}
    var_62 = {var_58: var_60, var_59: var_61}
    var_63 = module_0.from_json_schema(var_62)
    var_64 = module_1.Definitions()
    var_65 = '$ref'
    var_66 = '#/components/schemas/MySchema'
    var_67 = {var_65: var_66}
    var_68 = module_0.from_json_schema(var_67, var_64)
    var_69 = 'maxLength'
    var_70 = 'pattern'
    var_71 = 100
    var_72 = '^[a-z]+$'
    var_73 = {var_6: var_7, var_37: var_0, var_69: var_71, var_70: var_72}
    var_74 = module_0.from_json_schema(var_73)
    var_75 = 'components'
    var_76 = 'schemas'
    var_77 = 'MyType'
    var_78 = {var_6: var_7}
    var_79 = {var_77: var_78}
    var_80 = {var_76: var_79}
    var_81 = {var_75: var_80}
    var_82 = module_0.from_json_schema(var_81)
    var_83 = 5
    var_84 = 10
    var_85 = '^test'
    var_86 = {var_6: var_7, var_37: var_83, var_69: var_84, var_70: var_85}
    var_87 = module_0.from_json_schema(var_86)
    var_88 = 'minimum'
    var_89 = 'maximum'
    var_90 = 'multipleOf'
    var_91 = {var_6: var_13, var_88: var_2, var_89: var_71, var_90: var_83}
    var_92 = module_0.from_json_schema(var_91)
    var_93 = 'items'
    var_94 = 'minItems'
    var_95 = 'maxItems'
    var_96 = 'uniqueItems'
    var_97 = {var_6: var_7}
    var_98 = {var_6: var_19, var_93: var_97, var_94: var_0, var_95: var_84, var_96: var_0}
    var_99 = module_0.from_json_schema(var_98)
    var_100 = 'properties'
    var_101 = 'required'
    var_102 = 'name'
    var_103 = 'age'
    var_104 = {var_6: var_7}
    var_105 = {var_6: var_10}
    var_106 = {var_102: var_104, var_103: var_105}
    var_107 = [var_102]
    var_108 = {var_6: var_22, var_100: var_106, var_101: var_107}
    var_109 = module_0.from_json_schema(var_108)



# Parsed testcases at query #20
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
    var_11 = 'test_default'
    var_12 = {var_0: var_6, var_1: var_8, var_2: var_10, var_3: var_11}
    var_13 = 'object'
    var_14 = {var_4: var_13}
    var_15 = 'array'
    var_16 = {var_4: var_15}
    var_17 = {var_0: var_14, var_1: var_16}
    var_18 = 'number'
    var_19 = {var_4: var_18}
    var_20 = {var_4: var_5}
    var_21 = {var_0: var_19, var_2: var_20}
    var_22 = {var_4: var_9}
    var_23 = {var_0: var_22}
    var_24 = {var_4: var_5}
    var_25 = {var_4: var_7}
    var_26 = {var_0: var_24, var_1: var_25}
    var_27 = 'properties'
    var_28 = 'name'
    var_29 = {var_4: var_5}
    var_30 = {var_28: var_29}
    var_31 = {var_4: var_13, var_27: var_30}
    var_32 = 'items'
    var_33 = {var_4: var_7}
    var_34 = {var_4: var_15, var_32: var_33}
    var_35 = 'enum'
    var_36 = 1
    var_37 = 2
    var_38 = 3
    var_39 = [var_36, var_37, var_38]
    var_40 = {var_35: var_39}
    var_41 = None
    var_42 = {var_0: var_31, var_1: var_34, var_2: var_40, var_3: var_41}



# Parsed testcases at query #21
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
    var_30 = 'fixed_value'
    var_31 = {var_29: var_30}
    var_32 = module_0.from_json_schema(var_31)
    var_33 = 'minLength'
    var_34 = 'maxLength'
    var_35 = 10
    var_36 = {var_4: var_5, var_33: var_0, var_34: var_35}
    var_37 = module_0.from_json_schema(var_36)
    var_38 = 'minimum'
    var_39 = 'maximum'
    var_40 = 100
    var_41 = {var_4: var_11, var_38: var_2, var_39: var_40}
    var_42 = module_0.from_json_schema(var_41)
    var_43 = 'minItems'
    var_44 = 'maxItems'
    var_45 = 5
    var_46 = {var_4: var_17, var_43: var_0, var_44: var_45}
    var_47 = module_0.from_json_schema(var_46)
    var_48 = 'minProperties'
    var_49 = {var_4: var_20, var_48: var_0}
    var_50 = module_0.from_json_schema(var_49)
    var_51 = 'pattern'
    var_52 = '^[a-z]+$'
    var_53 = {var_4: var_5, var_51: var_52}
    var_54 = module_0.from_json_schema(var_53)
    var_55 = 'allOf'
    var_56 = {var_4: var_5}
    var_57 = {var_33: var_0}
    var_58 = [var_56, var_57]
    var_59 = {var_55: var_58}
    var_60 = module_0.from_json_schema(var_59)
    var_61 = 'oneOf'
    var_62 = {var_4: var_5}
    var_63 = {var_4: var_8}
    var_64 = [var_62, var_63]
    var_65 = {var_61: var_64}
    var_66 = module_0.from_json_schema(var_65)
    var_67 = 'not'
    var_68 = 'null'
    var_69 = {var_4: var_68}
    var_70 = {var_67: var_69}
    var_71 = module_0.from_json_schema(var_70)
    var_72 = 'if'
    var_73 = 'then'
    var_74 = {var_4: var_5}
    var_75 = {var_33: var_0}
    var_76 = {var_72: var_74, var_73: var_75}
    var_77 = module_0.from_json_schema(var_76)
    var_78 = {}
    var_79 = module_0.from_json_schema(var_78)
    var_80 = module_1.Definitions()
    var_81 = '$ref'
    var_82 = '#/components/schemas/Test'
    var_83 = {var_81: var_82}
    var_84 = module_0.from_json_schema(var_83, var_80)
    var_85 = 'components'
    var_86 = 'schemas'
    var_87 = 'TestSchema'
    var_88 = {var_4: var_5}
    var_89 = {var_87: var_88}
    var_90 = {var_86: var_89}
    var_91 = {var_4: var_20, var_85: var_90}
    var_92 = module_0.from_json_schema(var_91)
    var_93 = 'abc'
    var_94 = 'def'
    var_95 = [var_93, var_94]
    var_96 = {var_4: var_5, var_33: var_0, var_51: var_52, var_23: var_95}
    var_97 = module_0.from_json_schema(var_96)
    var_98 = 'properties'
    var_99 = 'name'
    var_100 = 'age'
    var_101 = {var_4: var_5}
    var_102 = {var_4: var_8}
    var_103 = {var_99: var_101, var_100: var_102}
    var_104 = {var_4: var_20, var_98: var_103}
    var_105 = module_0.from_json_schema(var_104)
    var_106 = 'items'
    var_107 = {var_4: var_5}
    var_108 = {var_4: var_17, var_106: var_107}
    var_109 = module_0.from_json_schema(var_108)
    var_110 = 'anyOf'
    var_111 = {var_4: var_5}
    var_112 = {var_4: var_8}
    var_113 = [var_111, var_112]
    var_114 = {var_110: var_113}
    var_115 = module_0.from_json_schema(var_114)



# Parsed testcases at query #22
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
    var_8 = 'enum'
    var_9 = 2
    var_10 = 3
    var_11 = [var_0, var_9, var_10]
    var_12 = {var_8: var_11}
    var_13 = module_0.from_json_schema(var_12)
    var_14 = 'const'
    var_15 = 'value'
    var_16 = {var_14: var_15}
    var_17 = module_0.from_json_schema(var_16)
    var_18 = 'a'
    var_19 = 'b'
    var_20 = [var_18, var_19]
    var_21 = {var_4: var_5, var_8: var_20}
    var_22 = module_0.from_json_schema(var_21)
    var_23 = module_1.Definitions()
    var_24 = '$ref'
    var_25 = '#/components/schemas/TestSchema'
    var_26 = {var_24: var_25}
    var_27 = module_0.from_json_schema(var_26, var_23)
    var_28 = 'allOf'
    var_29 = {var_4: var_5}
    var_30 = 'minLength'
    var_31 = {var_30: var_0}
    var_32 = [var_29, var_31]
    var_33 = {var_28: var_32}
    var_34 = module_0.from_json_schema(var_33)
    var_35 = 'anyOf'
    var_36 = {var_4: var_5}
    var_37 = 'integer'
    var_38 = {var_4: var_37}
    var_39 = [var_36, var_38]
    var_40 = {var_35: var_39}
    var_41 = module_0.from_json_schema(var_40)
    var_42 = 'oneOf'
    var_43 = {var_4: var_5}
    var_44 = {var_4: var_37}
    var_45 = [var_43, var_44]
    var_46 = {var_42: var_45}
    var_47 = module_0.from_json_schema(var_46)
    var_48 = 'not'
    var_49 = {var_4: var_5}
    var_50 = {var_48: var_49}
    var_51 = module_0.from_json_schema(var_50)
    var_52 = 'if'
    var_53 = 'then'
    var_54 = 'else'
    var_55 = {var_4: var_5}
    var_56 = {var_30: var_0}
    var_57 = {var_4: var_37}
    var_58 = {var_52: var_55, var_53: var_56, var_54: var_57}
    var_59 = module_0.from_json_schema(var_58)
    var_60 = {}
    var_61 = module_0.from_json_schema(var_60)
    var_62 = 'components'
    var_63 = 'schemas'
    var_64 = 'StringType'
    var_65 = 'IntType'
    var_66 = {var_4: var_5}
    var_67 = {var_4: var_37}
    var_68 = {var_64: var_66, var_65: var_67}
    var_69 = {var_63: var_68}
    var_70 = {var_62: var_69}
    var_71 = module_0.from_json_schema(var_70)
    var_72 = 'properties'
    var_73 = 'object'
    var_74 = 'name'
    var_75 = {var_4: var_5}
    var_76 = {var_74: var_75}
    var_77 = {var_4: var_73, var_72: var_76}
    var_78 = module_0.from_json_schema(var_77)
    var_79 = 'items'
    var_80 = 'array'
    var_81 = {var_4: var_5}
    var_82 = {var_4: var_80, var_79: var_81}
    var_83 = module_0.from_json_schema(var_82)
    var_84 = 'maxLength'
    var_85 = 'pattern'
    var_86 = 100
    var_87 = '^[a-z]+$'
    var_88 = {var_4: var_5, var_30: var_0, var_84: var_86, var_85: var_87}
    var_89 = module_0.from_json_schema(var_88)
    var_90 = 'minimum'
    var_91 = 'maximum'
    var_92 = 'multipleOf'
    var_93 = 'number'
    var_94 = 5
    var_95 = {var_4: var_93, var_90: var_2, var_91: var_86, var_92: var_94}
    var_96 = module_0.from_json_schema(var_95)
    var_97 = module_1.Definitions()
    var_98 = 'CustomRef'
    var_99 = {var_24: var_98}
    var_100 = module_0.from_json_schema(var_99, var_97)



# Parsed testcases at query #23
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
    var_4 = False
    var_5 = module_0.String()
    var_6 = module_1.to_json_schema(var_5)
    var_7 = True
    var_8 = module_0.String()
    var_9 = module_1.to_json_schema(var_8)
    var_10 = 5
    var_11 = 10
    var_12 = '^[a-z]+$'
    var_13 = module_0.String(max_length=var_11, min_length=var_10, pattern=var_12)
    var_14 = module_1.to_json_schema(var_13)
    var_15 = module_0.String(allow_blank=var_4)
    var_16 = module_1.to_json_schema(var_15)
    var_17 = module_0.Integer()
    var_18 = module_1.to_json_schema(var_17)
    var_19 = 100
    var_20 = module_0.Integer(minimum=var_4, maximum=var_19, multiple_of=var_10)
    var_21 = module_1.to_json_schema(var_20)
    var_22 = module_0.Float()
    var_23 = module_1.to_json_schema(var_22)
    var_24 = 0.5
    var_25 = 99.5
    var_26 = module_0.Float(exclusive_minimum=var_24, exclusive_maximum=var_25)
    var_27 = module_1.to_json_schema(var_26)
    var_28 = module_0.Boolean()
    var_29 = module_1.to_json_schema(var_28)
    var_30 = module_0.Boolean()
    var_31 = module_1.to_json_schema(var_30)
    var_32 = module_0.String()
    var_33 = module_0.Array(var_32, min_items=var_7, max_items=var_11)
    var_34 = module_1.to_json_schema(var_33)
    var_35 = module_0.Integer()
    var_36 = module_0.Array(var_35, unique_items=var_7)
    var_37 = module_1.to_json_schema(var_36)
    var_38 = module_0.String()
    var_39 = module_0.Integer()
    var_40 = [var_38, var_39]
    var_41 = module_0.Array(var_40)
    var_42 = module_1.to_json_schema(var_41)
    var_43 = 'items'
    var_44 = var_42[var_43]
    var_45 = var_42[var_43]
    var_46 = len(var_45)
    assert var_46 == 2
    var_47 = module_0.Array(additional_items=var_4)
    var_48 = module_1.to_json_schema(var_47)
    var_49 = module_0.String()
    var_50 = module_0.Array(additional_items=var_49)
    var_51 = module_1.to_json_schema(var_50)
    var_52 = 'name'
    var_53 = module_0.String()
    var_54 = {var_52: var_53}
    var_55 = module_0.Object(properties=var_54)
    var_56 = module_1.to_json_schema(var_55)
    var_57 = 'id'
    var_58 = module_0.Integer()
    var_59 = {var_57: var_58}
    var_60 = [var_57]
    var_61 = module_0.Object(properties=var_59, required=var_60)
    var_62 = module_1.to_json_schema(var_61)
    var_63 = '^S_'
    var_64 = module_0.String()
    var_65 = {var_63: var_64}
    var_66 = module_0.Object(pattern_properties=var_65)
    var_67 = module_1.to_json_schema(var_66)
    var_68 = module_0.Object(additional_properties=var_4)
    var_69 = module_1.to_json_schema(var_68)
    var_70 = module_0.String()
    var_71 = module_0.Object(additional_properties=var_70)
    var_72 = module_1.to_json_schema(var_71)
    var_73 = module_0.String(pattern=var_12)
    var_74 = module_0.Object(property_names=var_73)
    var_75 = module_1.to_json_schema(var_74)
    var_76 = module_0.Object(min_properties=var_7, max_properties=var_10)
    var_77 = module_1.to_json_schema(var_76)
    var_78 = 'a'
    var_79 = (var_78, var_78)
    var_80 = 'b'
    var_81 = (var_80, var_80)
    var_82 = [var_79, var_81]
    var_83 = module_0.Choice(choices=var_82)
    var_84 = module_1.to_json_schema(var_83)
    var_85 = 'fixed'
    var_86 = module_0.Const(var_85)
    var_87 = module_1.to_json_schema(var_86)
    var_88 = module_0.String()
    var_89 = module_0.Integer()
    var_90 = [var_88, var_89]
    var_91 = module_0.Union(var_90)
    var_92 = module_1.to_json_schema(var_91)
    var_93 = 'anyOf'
    var_94 = var_92[var_93]
    var_95 = len(var_94)
    assert var_95 == 2
    var_96 = module_0.String()
    var_97 = module_0.Integer()
    var_98 = [var_96, var_97]
    var_99 = module_2.OneOf(var_98)
    var_100 = module_1.to_json_schema(var_99)
    var_101 = 'oneOf'
    var_102 = var_100[var_101]
    var_103 = len(var_102)
    assert var_103 == 2
    var_104 = module_0.String()
    var_105 = 'test'
    var_106 = module_0.Const(var_105)
    var_107 = [var_104, var_106]
    var_108 = module_2.AllOf(var_107)
    var_109 = module_1.to_json_schema(var_108)
    var_110 = 'allOf'
    var_111 = var_109[var_110]
    var_112 = len(var_111)
    assert var_112 == 2
    var_113 = module_0.String()
    var_114 = module_2.Not(var_113)
    var_115 = module_1.to_json_schema(var_114)
    var_116 = module_0.String()
    var_117 = module_0.Integer()
    var_118 = module_0.Boolean()
    var_119 = module_2.IfThenElse(var_116, var_117, var_118)
    var_120 = module_1.to_json_schema(var_119)
    var_121 = module_0.String()
    var_122 = module_0.Integer()
    var_123 = module_2.IfThenElse(var_121, var_122)
    var_124 = module_1.to_json_schema(var_123)
    var_125 = 'MyString'
    var_126 = module_0.String()
    var_127 = {var_125: var_126}
    var_128 = module_0.String()
    var_129 = {var_125: var_128}
    var_130 = module_0.String()
    var_131 = module_1.to_json_schema(var_130)
    var_132 = False
    var_133 = module_1.to_json_schema(var_0)
    var_134 = True



# Parsed testcases at query #24
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
    var_15 = 5
    var_16 = module_0.Integer(minimum=var_4, maximum=var_14, multiple_of=var_15)
    var_17 = module_1.to_json_schema(var_16)
    var_18 = module_0.Float(exclusive_minimum=var_4, exclusive_maximum=var_11)
    var_19 = module_1.to_json_schema(var_18)
    var_20 = module_0.Boolean()
    var_21 = module_1.to_json_schema(var_20)
    var_22 = True
    var_23 = module_0.Boolean()
    var_24 = module_1.to_json_schema(var_23)
    var_25 = module_0.String()
    var_26 = True
    var_27 = module_0.Array(var_25, min_items=var_22, max_items=var_15, unique_items=var_26)
    var_28 = module_1.to_json_schema(var_27)
    var_29 = module_0.String()
    var_30 = module_0.Integer()
    var_31 = [var_29, var_30]
    var_32 = module_0.Array(var_31)
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
    var_43 = [var_38]
    var_44 = module_0.Object(properties=var_42, min_properties=var_26, max_properties=var_15, required=var_43)
    var_45 = module_1.to_json_schema(var_44)
    var_46 = '^S_'
    var_47 = module_0.String()
    var_48 = {var_46: var_47}
    var_49 = module_0.Object(pattern_properties=var_48)
    var_50 = module_1.to_json_schema(var_49)
    var_51 = module_0.Object(additional_properties=var_4)
    var_52 = module_1.to_json_schema(var_51)
    var_53 = module_0.String()
    var_54 = module_0.Object(additional_properties=var_53)
    var_55 = module_1.to_json_schema(var_54)
    var_56 = 'additionalProperties'
    var_57 = var_55[var_56]
    var_58 = module_0.String(pattern=var_7)
    var_59 = module_0.Object(property_names=var_58)
    var_60 = module_1.to_json_schema(var_59)
    var_61 = 'a'
    var_62 = 'Option A'
    var_63 = (var_61, var_62)
    var_64 = 'b'
    var_65 = 'Option B'
    var_66 = (var_64, var_65)
    var_67 = [var_63, var_66]
    var_68 = module_0.Choice(choices=var_67)
    var_69 = module_1.to_json_schema(var_68)
    var_70 = 'constant_value'
    var_71 = module_0.Const(var_70)
    var_72 = module_1.to_json_schema(var_71)
    var_73 = module_0.String()
    var_74 = module_0.Integer()
    var_75 = [var_73, var_74]
    var_76 = module_0.Union(var_75)
    var_77 = module_1.to_json_schema(var_76)
    var_78 = 'anyOf'
    var_79 = var_77[var_78]
    var_80 = len(var_79)
    assert var_80 == 2
    var_81 = module_0.String()
    var_82 = module_0.Integer()
    var_83 = [var_81, var_82]
    var_84 = module_2.OneOf(var_83)
    var_85 = module_1.to_json_schema(var_84)
    var_86 = 'oneOf'
    var_87 = var_85[var_86]
    var_88 = len(var_87)
    assert var_88 == 2
    var_89 = module_0.String(min_length=var_26)
    var_90 = module_0.String(max_length=var_6)
    var_91 = [var_89, var_90]
    var_92 = module_2.AllOf(var_91)
    var_93 = module_1.to_json_schema(var_92)
    var_94 = 'allOf'
    var_95 = var_93[var_94]
    var_96 = len(var_95)
    assert var_96 == 2
    var_97 = module_0.String()
    var_98 = module_2.Not(var_97)
    var_99 = module_1.to_json_schema(var_98)
    var_100 = module_0.String()
    var_101 = module_0.Integer()
    var_102 = module_0.Boolean()
    var_103 = module_2.IfThenElse(var_100, var_101, var_102)
    var_104 = module_1.to_json_schema(var_103)
    var_105 = module_0.String()
    var_106 = module_0.Integer()
    var_107 = module_2.IfThenElse(var_105, var_106)
    var_108 = module_1.to_json_schema(var_107)
    var_109 = 'StringSchema'
    var_110 = 'IntSchema'
    var_111 = module_0.String()
    var_112 = module_0.Integer()
    var_113 = {var_109: var_111, var_110: var_112}
    var_114 = module_0.String()
    var_115 = {var_109: var_114}
    var_116 = module_1.to_json_schema(var_0)
    var_117 = module_0.Array(additional_items=var_4)
    var_118 = module_1.to_json_schema(var_117)
    var_119 = module_0.String()
    var_120 = module_0.Array(additional_items=var_119)
    var_121 = module_1.to_json_schema(var_120)
    var_122 = 'additionalItems'
    var_123 = var_121[var_122]
    var_124 = True
    var_125 = module_0.String(allow_blank=var_124)
    var_126 = module_1.to_json_schema(var_125)
    var_127 = 'minLength'
    var_128 = None



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
    var_14 = True
    var_15 = module_0.String(allow_blank=var_14)
    var_16 = module_1.to_json_schema(var_15)
    var_17 = 100
    var_18 = 99
    var_19 = 5
    var_20 = module_0.Integer(minimum=var_4, maximum=var_17, exclusive_minimum=var_14, exclusive_maximum=var_18, multiple_of=var_19)
    var_21 = module_1.to_json_schema(var_20)
    var_22 = True
    var_23 = module_0.Integer()
    var_24 = module_1.to_json_schema(var_23)
    var_25 = module_0.Float(minimum=var_4, maximum=var_17)
    var_26 = module_1.to_json_schema(var_25)
    var_27 = module_0.Decimal()
    var_28 = module_1.to_json_schema(var_27)
    var_29 = module_0.Boolean()
    var_30 = module_1.to_json_schema(var_29)
    var_31 = True
    var_32 = module_0.Boolean()
    var_33 = module_1.to_json_schema(var_32)
    var_34 = module_0.String()
    var_35 = True
    var_36 = module_0.Array(var_34, min_items=var_31, max_items=var_6, unique_items=var_35)
    var_37 = module_1.to_json_schema(var_36)
    var_38 = module_0.String()
    var_39 = module_0.Integer()
    var_40 = [var_38, var_39]
    var_41 = module_0.Array(var_40)
    var_42 = module_1.to_json_schema(var_41)
    var_43 = 'items'
    var_44 = var_42[var_43]
    var_45 = var_42[var_43]
    var_46 = len(var_45)
    assert var_46 == 2
    var_47 = module_0.Array(additional_items=var_4)
    var_48 = module_1.to_json_schema(var_47)
    var_49 = module_0.String()
    var_50 = module_0.Array(additional_items=var_49)
    var_51 = module_1.to_json_schema(var_50)
    var_52 = 'additionalItems'
    var_53 = var_51[var_52]
    var_54 = 'name'
    var_55 = 'age'
    var_56 = module_0.String()
    var_57 = module_0.Integer()
    var_58 = {var_54: var_56, var_55: var_57}
    var_59 = [var_54]
    var_60 = module_0.Object(properties=var_58, min_properties=var_35, max_properties=var_6, required=var_59)
    var_61 = module_1.to_json_schema(var_60)
    var_62 = '^S_'
    var_63 = module_0.String()
    var_64 = {var_62: var_63}
    var_65 = module_0.Object(pattern_properties=var_64)
    var_66 = module_1.to_json_schema(var_65)
    var_67 = module_0.Object(additional_properties=var_4)
    var_68 = module_1.to_json_schema(var_67)
    var_69 = module_0.String()
    var_70 = module_0.Object(additional_properties=var_69)
    var_71 = module_1.to_json_schema(var_70)
    var_72 = 'additionalProperties'
    var_73 = var_71[var_72]
    var_74 = module_0.String(pattern=var_7)
    var_75 = module_0.Object(property_names=var_74)
    var_76 = module_1.to_json_schema(var_75)
    var_77 = module_0.String()
    var_78 = module_0.Integer()
    var_79 = {var_54: var_77, var_55: var_78}
    var_80 = [var_54]
    var_81 = module_3.Schema(var_79)
    var_82 = module_1.to_json_schema(var_81)
    var_83 = 'a'
    var_84 = 'Option A'
    var_85 = (var_83, var_84)
    var_86 = 'b'
    var_87 = 'Option B'
    var_88 = (var_86, var_87)
    var_89 = [var_85, var_88]
    var_90 = module_0.Choice(choices=var_89)
    var_91 = module_1.to_json_schema(var_90)
    var_92 = 'constant_value'
    var_93 = module_0.Const(var_92)
    var_94 = module_1.to_json_schema(var_93)
    var_95 = module_0.String()
    var_96 = module_0.Integer()
    var_97 = [var_95, var_96]
    var_98 = module_0.Union(var_97)
    var_99 = module_1.to_json_schema(var_98)
    var_100 = 'anyOf'
    var_101 = var_99[var_100]
    var_102 = len(var_101)
    assert var_102 == 2
    var_103 = module_0.String()
    var_104 = module_0.Integer()
    var_105 = [var_103, var_104]
    var_106 = module_2.OneOf(var_105)
    var_107 = module_1.to_json_schema(var_106)
    var_108 = 'oneOf'
    var_109 = var_107[var_108]
    var_110 = len(var_109)
    assert var_110 == 2
    var_111 = module_0.String()
    var_112 = 'A'
    var_113 = (var_83, var_112)
    var_114 = [var_113]
    var_115 = module_0.Choice(choices=var_114)
    var_116 = [var_111, var_115]
    var_117 = module_2.AllOf(var_116)
    var_118 = module_1.to_json_schema(var_117)
    var_119 = 'allOf'
    var_120 = var_118[var_119]
    var_121 = len(var_120)
    assert var_121 == 2
    var_122 = module_0.String()
    var_123 = module_2.Not(var_122)
    var_124 = module_1.to_json_schema(var_123)
    var_125 = (var_83, var_112)
    var_126 = [var_125]
    var_127 = module_0.Choice(choices=var_126)
    var_128 = module_0.String()
    var_129 = module_0.Integer()
    var_130 = module_2.IfThenElse(var_127, var_128, var_129)
    var_131 = module_1.to_json_schema(var_130)
    var_132 = (var_83, var_112)
    var_133 = [var_132]
    var_134 = module_0.Choice(choices=var_133)
    var_135 = module_2.IfThenElse(var_134)
    var_136 = module_1.to_json_schema(var_135)



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
    var_14 = True
    var_15 = module_0.String(allow_blank=var_14)
    var_16 = module_1.to_json_schema(var_15)
    var_17 = 100
    var_18 = 5
    var_19 = module_0.Integer(minimum=var_4, maximum=var_17, multiple_of=var_18)
    var_20 = module_1.to_json_schema(var_19)
    var_21 = 0.5
    var_22 = 99.5
    var_23 = module_0.Float(exclusive_minimum=var_21, exclusive_maximum=var_22)
    var_24 = module_1.to_json_schema(var_23)
    var_25 = module_0.Boolean()
    var_26 = module_1.to_json_schema(var_25)
    var_27 = True
    var_28 = module_0.Boolean()
    var_29 = module_1.to_json_schema(var_28)
    var_30 = module_0.String()
    var_31 = True
    var_32 = module_0.Array(var_30, min_items=var_27, max_items=var_18, unique_items=var_31)
    var_33 = module_1.to_json_schema(var_32)
    var_34 = 'items'
    var_35 = var_33[var_34]
    var_36 = module_0.String()
    var_37 = module_0.Integer()
    var_38 = [var_36, var_37]
    var_39 = module_0.Array(var_38)
    var_40 = module_1.to_json_schema(var_39)
    var_41 = var_40[var_34]
    var_42 = var_40[var_34]
    var_43 = len(var_42)
    assert var_43 == 2
    var_44 = module_0.Array(additional_items=var_4)
    var_45 = module_1.to_json_schema(var_44)
    var_46 = module_0.String()
    var_47 = module_0.Array(additional_items=var_46)
    var_48 = module_1.to_json_schema(var_47)
    var_49 = 'additionalItems'
    var_50 = var_48[var_49]
    var_51 = 'name'
    var_52 = 'age'
    var_53 = module_0.String()
    var_54 = module_0.Integer()
    var_55 = {var_51: var_53, var_52: var_54}
    var_56 = [var_51]
    var_57 = module_0.Object(properties=var_55, min_properties=var_31, max_properties=var_6, required=var_56)
    var_58 = module_1.to_json_schema(var_57)
    var_59 = '^S_'
    var_60 = '^I_'
    var_61 = module_0.String()
    var_62 = module_0.Integer()
    var_63 = {var_59: var_61, var_60: var_62}
    var_64 = module_0.Object(pattern_properties=var_63)
    var_65 = module_1.to_json_schema(var_64)
    var_66 = module_0.Object(additional_properties=var_4)
    var_67 = module_1.to_json_schema(var_66)
    var_68 = module_0.String()
    var_69 = module_0.Object(additional_properties=var_68)
    var_70 = module_1.to_json_schema(var_69)
    var_71 = 'additionalProperties'
    var_72 = var_70[var_71]
    var_73 = module_0.String(pattern=var_7)
    var_74 = module_0.Object(property_names=var_73)
    var_75 = module_1.to_json_schema(var_74)
    var_76 = 'a'
    var_77 = 'Option A'
    var_78 = (var_76, var_77)
    var_79 = 'b'
    var_80 = 'Option B'
    var_81 = (var_79, var_80)
    var_82 = [var_78, var_81]
    var_83 = module_0.Choice(choices=var_82)
    var_84 = module_1.to_json_schema(var_83)
    var_85 = 'constant_value'
    var_86 = module_0.Const(var_85)
    var_87 = module_1.to_json_schema(var_86)
    var_88 = module_0.String()
    var_89 = module_0.Integer()
    var_90 = [var_88, var_89]
    var_91 = module_0.Union(var_90)
    var_92 = module_1.to_json_schema(var_91)
    var_93 = 'anyOf'
    var_94 = var_92[var_93]
    var_95 = len(var_94)
    assert var_95 == 2
    var_96 = module_0.String()
    var_97 = module_0.Integer()
    var_98 = [var_96, var_97]
    var_99 = module_2.OneOf(var_98)
    var_100 = module_1.to_json_schema(var_99)
    var_101 = 'oneOf'
    var_102 = var_100[var_101]
    var_103 = len(var_102)
    assert var_103 == 2
    var_104 = module_0.String(min_length=var_31)
    var_105 = module_0.String(max_length=var_17)
    var_106 = [var_104, var_105]
    var_107 = module_2.AllOf(var_106)
    var_108 = module_1.to_json_schema(var_107)
    var_109 = 'allOf'
    var_110 = var_108[var_109]
    var_111 = len(var_110)
    assert var_111 == 2
    var_112 = module_0.String()
    var_113 = module_2.Not(var_112)
    var_114 = module_1.to_json_schema(var_113)
    var_115 = module_0.String()
    var_116 = module_0.Integer()
    var_117 = module_0.Boolean()
    var_118 = module_2.IfThenElse(var_115, var_116, var_117)
    var_119 = module_1.to_json_schema(var_118)
    var_120 = module_0.String()
    var_121 = module_2.IfThenElse(var_120)
    var_122 = module_1.to_json_schema(var_121)
    var_123 = 'MyType'
    var_124 = module_0.String()
    var_125 = {var_123: var_124}
    var_126 = 'StringType'
    var_127 = 'IntType'
    var_128 = module_0.String()
    var_129 = module_0.Integer()
    var_130 = {var_126: var_128, var_127: var_129}



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
    var_14 = True
    var_15 = None
    var_16 = module_0.String(allow_blank=var_14, min_length=var_15)
    var_17 = module_1.to_json_schema(var_16)
    var_18 = 100
    var_19 = 5
    var_20 = module_0.Integer(minimum=var_14, maximum=var_18, multiple_of=var_19)
    var_21 = module_1.to_json_schema(var_20)
    var_22 = module_0.Integer(exclusive_minimum=var_4, exclusive_maximum=var_6)
    var_23 = module_1.to_json_schema(var_22)
    var_24 = 0.5
    var_25 = 99.9
    var_26 = module_0.Float(minimum=var_24, maximum=var_25)
    var_27 = module_1.to_json_schema(var_26)
    var_28 = module_0.Boolean()
    var_29 = module_1.to_json_schema(var_28)
    var_30 = True
    var_31 = module_0.Boolean()
    var_32 = module_1.to_json_schema(var_31)
    var_33 = module_0.String()
    var_34 = True
    var_35 = module_0.Array(var_33, min_items=var_30, max_items=var_19, unique_items=var_34)
    var_36 = module_1.to_json_schema(var_35)
    var_37 = module_0.String()
    var_38 = module_0.Integer()
    var_39 = [var_37, var_38]
    var_40 = module_0.Array(var_39)
    var_41 = module_1.to_json_schema(var_40)
    var_42 = 'items'
    var_43 = var_41[var_42]
    var_44 = var_41[var_42]
    var_45 = len(var_44)
    assert var_45 == 2
    var_46 = module_0.Array(additional_items=var_4)
    var_47 = module_1.to_json_schema(var_46)
    var_48 = module_0.String()
    var_49 = module_0.Array(additional_items=var_48)
    var_50 = module_1.to_json_schema(var_49)
    var_51 = 'additionalItems'
    var_52 = var_50[var_51]
    var_53 = 'name'
    var_54 = 'age'
    var_55 = module_0.String()
    var_56 = module_0.Integer()
    var_57 = {var_53: var_55, var_54: var_56}
    var_58 = [var_53]
    var_59 = module_0.Object(properties=var_57, min_properties=var_34, max_properties=var_6, required=var_58)
    var_60 = module_1.to_json_schema(var_59)
    var_61 = '^S_'
    var_62 = module_0.String()
    var_63 = {var_61: var_62}
    var_64 = module_0.Object(pattern_properties=var_63)
    var_65 = module_1.to_json_schema(var_64)
    var_66 = module_0.Object(additional_properties=var_4)
    var_67 = module_1.to_json_schema(var_66)
    var_68 = module_0.String()
    var_69 = module_0.Object(additional_properties=var_68)
    var_70 = module_1.to_json_schema(var_69)
    var_71 = 'additionalProperties'
    var_72 = var_70[var_71]
    var_73 = module_0.String(pattern=var_7)
    var_74 = module_0.Object(property_names=var_73)
    var_75 = module_1.to_json_schema(var_74)
    var_76 = 'a'
    var_77 = 'Option A'
    var_78 = (var_76, var_77)
    var_79 = 'b'
    var_80 = 'Option B'
    var_81 = (var_79, var_80)
    var_82 = [var_78, var_81]
    var_83 = module_0.Choice(choices=var_82)
    var_84 = module_1.to_json_schema(var_83)
    var_85 = 'constant_value'
    var_86 = module_0.Const(var_85)
    var_87 = module_1.to_json_schema(var_86)
    var_88 = module_0.String()
    var_89 = module_0.Integer()
    var_90 = [var_88, var_89]
    var_91 = module_0.Union(var_90)
    var_92 = module_1.to_json_schema(var_91)
    var_93 = 'anyOf'
    var_94 = var_92[var_93]
    var_95 = len(var_94)
    assert var_95 == 2
    var_96 = module_0.String()
    var_97 = module_0.Integer()
    var_98 = [var_96, var_97]
    var_99 = module_2.OneOf(var_98)
    var_100 = module_1.to_json_schema(var_99)
    var_101 = 'oneOf'
    var_102 = var_100[var_101]
    var_103 = len(var_102)
    assert var_103 == 2
    var_104 = module_0.String(min_length=var_34)
    var_105 = module_0.String(max_length=var_6)
    var_106 = [var_104, var_105]
    var_107 = module_2.AllOf(var_106)
    var_108 = module_1.to_json_schema(var_107)
    var_109 = 'allOf'
    var_110 = var_108[var_109]
    var_111 = len(var_110)
    assert var_111 == 2
    var_112 = module_0.String()
    var_113 = module_2.Not(var_112)
    var_114 = module_1.to_json_schema(var_113)
    var_115 = module_0.String()
    var_116 = module_0.Integer()
    var_117 = module_0.Boolean()
    var_118 = module_2.IfThenElse(var_115, var_116, var_117)
    var_119 = module_1.to_json_schema(var_118)
    var_120 = module_0.String()
    var_121 = module_2.IfThenElse(var_120)
    var_122 = module_1.to_json_schema(var_121)
    var_123 = 'default_value'
    var_124 = module_0.String()
    var_125 = module_1.to_json_schema(var_124)
    var_126 = module_3.Definitions()
    var_127 = 'MySchema'
    var_128 = module_3.Reference(var_127, var_126)
    var_129 = module_1.to_json_schema(var_128)
    var_130 = module_3.Definitions()
    var_131 = module_1.to_json_schema(var_130)
    var_132 = module_1.to_json_schema(var_0)



# Parsed testcases at query #28
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
    var_14 = True
    var_15 = module_0.String(allow_blank=var_14)
    var_16 = module_1.to_json_schema(var_15)
    var_17 = 'minLength'
    var_18 = 100
    var_19 = 5
    var_20 = module_0.Integer(minimum=var_4, maximum=var_18, multiple_of=var_19)
    var_21 = module_1.to_json_schema(var_20)
    var_22 = module_0.Integer(exclusive_minimum=var_4, exclusive_maximum=var_18)
    var_23 = module_1.to_json_schema(var_22)
    var_24 = module_0.Float(minimum=var_4, maximum=var_14)
    var_25 = module_1.to_json_schema(var_24)
    var_26 = module_0.Boolean()
    var_27 = module_1.to_json_schema(var_26)
    var_28 = True
    var_29 = module_0.Boolean()
    var_30 = module_1.to_json_schema(var_29)
    var_31 = module_0.String()
    var_32 = True
    var_33 = module_0.Array(var_31, min_items=var_28, max_items=var_6, unique_items=var_32)
    var_34 = module_1.to_json_schema(var_33)
    var_35 = module_0.String()
    var_36 = module_0.Integer()
    var_37 = [var_35, var_36]
    var_38 = module_0.Array(var_37)
    var_39 = module_1.to_json_schema(var_38)
    var_40 = 'items'
    var_41 = var_39[var_40]
    var_42 = var_39[var_40]
    var_43 = len(var_42)
    assert var_43 == 2
    var_44 = module_0.Array(additional_items=var_4)
    var_45 = module_1.to_json_schema(var_44)
    var_46 = 'name'
    var_47 = 'age'
    var_48 = module_0.String()
    var_49 = module_0.Integer()
    var_50 = {var_46: var_48, var_47: var_49}
    var_51 = [var_46]
    var_52 = module_0.Object(properties=var_50, min_properties=var_32, max_properties=var_6, required=var_51)
    var_53 = module_1.to_json_schema(var_52)
    var_54 = '^S_'
    var_55 = module_0.String()
    var_56 = {var_54: var_55}
    var_57 = module_0.Object(pattern_properties=var_56)
    var_58 = module_1.to_json_schema(var_57)
    var_59 = module_0.Object(additional_properties=var_4)
    var_60 = module_1.to_json_schema(var_59)
    var_61 = 'a'
    var_62 = 'Option A'
    var_63 = (var_61, var_62)
    var_64 = 'b'
    var_65 = 'Option B'
    var_66 = (var_64, var_65)
    var_67 = [var_63, var_66]
    var_68 = module_0.Choice(choices=var_67)
    var_69 = module_1.to_json_schema(var_68)
    var_70 = 'constant_value'
    var_71 = module_0.Const(var_70)
    var_72 = module_1.to_json_schema(var_71)
    var_73 = module_0.String()
    var_74 = module_0.Integer()
    var_75 = [var_73, var_74]
    var_76 = module_0.Union(var_75)
    var_77 = module_1.to_json_schema(var_76)
    var_78 = 'anyOf'
    var_79 = var_77[var_78]
    var_80 = len(var_79)
    assert var_80 == 2
    var_81 = module_0.String()
    var_82 = module_0.Integer()
    var_83 = [var_81, var_82]
    var_84 = module_2.OneOf(var_83)
    var_85 = module_1.to_json_schema(var_84)
    var_86 = 'oneOf'
    var_87 = var_85[var_86]
    var_88 = len(var_87)
    assert var_88 == 2
    var_89 = module_0.String()
    var_90 = 'test'
    var_91 = module_0.Const(var_90)
    var_92 = [var_89, var_91]
    var_93 = module_2.AllOf(var_92)
    var_94 = module_1.to_json_schema(var_93)
    var_95 = 'allOf'
    var_96 = var_94[var_95]
    var_97 = len(var_96)
    assert var_97 == 2
    var_98 = module_0.String()
    var_99 = module_2.Not(var_98)
    var_100 = module_1.to_json_schema(var_99)
    var_101 = 'A'
    var_102 = (var_61, var_101)
    var_103 = 'B'
    var_104 = (var_64, var_103)
    var_105 = [var_102, var_104]
    var_106 = module_0.Choice(choices=var_105)
    var_107 = module_0.String()
    var_108 = module_0.Integer()
    var_109 = module_2.IfThenElse(var_106, var_107, var_108)
    var_110 = module_1.to_json_schema(var_109)
    var_111 = (var_61, var_101)
    var_112 = [var_111]
    var_113 = module_0.Choice(choices=var_112)
    var_114 = module_0.String()
    var_115 = module_2.IfThenElse(var_113, var_114)
    var_116 = module_1.to_json_schema(var_115)
    var_117 = 'default_value'
    var_118 = module_0.String()
    var_119 = module_1.to_json_schema(var_118)
    var_120 = 'default'
    var_121 = 'User'
    var_122 = module_0.String()
    var_123 = {var_46: var_122}
    var_124 = module_0.Object(properties=var_123)
    var_125 = {var_121: var_124}
    var_126 = module_0.Object()
    var_127 = {var_121: var_126}



# Parsed testcases at query #29
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
    var_4 = False
    var_5 = 1
    var_6 = 10
    var_7 = '\\d+'
    var_8 = 'email'
    var_9 = module_0.String(max_length=var_6, min_length=var_5, pattern=var_7, format=var_8)
    var_10 = module_1.to_json_schema(var_9)
    var_11 = True
    var_12 = module_0.String()
    var_13 = module_1.to_json_schema(var_12)
    var_14 = True
    var_15 = None
    var_16 = module_0.String(allow_blank=var_14, min_length=var_15)
    var_17 = module_1.to_json_schema(var_16)
    var_18 = 100
    var_19 = 5
    var_20 = 95
    var_21 = module_0.Integer(minimum=var_4, maximum=var_18, exclusive_minimum=var_19, exclusive_maximum=var_20, multiple_of=var_19)
    var_22 = module_1.to_json_schema(var_21)
    var_23 = module_0.Float(minimum=var_4, maximum=var_14)
    var_24 = module_1.to_json_schema(var_23)
    var_25 = module_0.Boolean()
    var_26 = module_1.to_json_schema(var_25)
    var_27 = True
    var_28 = module_0.Boolean()
    var_29 = module_1.to_json_schema(var_28)
    var_30 = module_0.String()
    var_31 = True
    var_32 = module_0.Array(var_30, min_items=var_27, max_items=var_6, unique_items=var_31)
    var_33 = module_1.to_json_schema(var_32)
    var_34 = module_0.String()
    var_35 = module_0.Integer()
    var_36 = [var_34, var_35]
    var_37 = module_0.Array(var_36)
    var_38 = module_1.to_json_schema(var_37)
    var_39 = 'items'
    var_40 = var_38[var_39]
    var_41 = var_38[var_39]
    var_42 = len(var_41)
    assert var_42 == 2
    var_43 = module_0.String()
    var_44 = module_0.Array(var_43, var_4)
    var_45 = module_1.to_json_schema(var_44)
    var_46 = module_0.String()
    var_47 = module_0.Integer()
    var_48 = module_0.Array(var_46, var_47)
    var_49 = module_1.to_json_schema(var_48)
    var_50 = 'name'
    var_51 = 'age'
    var_52 = module_0.String()
    var_53 = module_0.Integer()
    var_54 = {var_50: var_52, var_51: var_53}
    var_55 = [var_50]
    var_56 = module_0.Object(properties=var_54, min_properties=var_31, max_properties=var_6, required=var_55)
    var_57 = module_1.to_json_schema(var_56)
    var_58 = '^S_'
    var_59 = module_0.String()
    var_60 = {var_58: var_59}
    var_61 = module_0.Object(pattern_properties=var_60)
    var_62 = module_1.to_json_schema(var_61)
    var_63 = module_0.Object(additional_properties=var_4)
    var_64 = module_1.to_json_schema(var_63)
    var_65 = module_0.String()
    var_66 = module_0.Object(additional_properties=var_65)
    var_67 = module_1.to_json_schema(var_66)
    var_68 = module_0.String(min_length=var_31)
    var_69 = module_0.Object(property_names=var_68)
    var_70 = module_1.to_json_schema(var_69)
    var_71 = 'a'
    var_72 = 'option_a'
    var_73 = (var_71, var_72)
    var_74 = 'b'
    var_75 = 'option_b'
    var_76 = (var_74, var_75)
    var_77 = [var_73, var_76]
    var_78 = module_0.Choice(choices=var_77)
    var_79 = module_1.to_json_schema(var_78)
    var_80 = 'fixed_value'
    var_81 = module_0.Const(var_80)
    var_82 = module_1.to_json_schema(var_81)
    var_83 = module_0.String()
    var_84 = module_0.Integer()
    var_85 = [var_83, var_84]
    var_86 = module_0.Union(var_85)
    var_87 = module_1.to_json_schema(var_86)
    var_88 = 'anyOf'
    var_89 = var_87[var_88]
    var_90 = len(var_89)
    assert var_90 == 2
    var_91 = module_0.String()
    var_92 = module_0.Integer()
    var_93 = [var_91, var_92]
    var_94 = module_2.OneOf(var_93)
    var_95 = module_1.to_json_schema(var_94)
    var_96 = 'oneOf'
    var_97 = var_95[var_96]
    var_98 = len(var_97)
    assert var_98 == 2
    var_99 = module_0.String()
    var_100 = module_0.Object()
    var_101 = [var_99, var_100]
    var_102 = module_2.AllOf(var_101)
    var_103 = module_1.to_json_schema(var_102)
    var_104 = 'allOf'
    var_105 = var_103[var_104]
    var_106 = len(var_105)
    assert var_106 == 2
    var_107 = module_0.String()
    var_108 = module_2.Not(var_107)
    var_109 = module_1.to_json_schema(var_108)
    var_110 = 'type'
    var_111 = (var_71, var_71)
    var_112 = [var_111]
    var_113 = module_0.Choice(choices=var_112)
    var_114 = {var_110: var_113}
    var_115 = module_0.Object(properties=var_114)
    var_116 = module_0.String()
    var_117 = module_0.Integer()
    var_118 = module_2.IfThenElse(var_115, var_116, var_117)
    var_119 = module_1.to_json_schema(var_118)
    var_120 = module_0.String()
    var_121 = module_2.IfThenElse(var_120)
    var_122 = module_1.to_json_schema(var_121)
    var_123 = 'MyString'
    var_124 = module_0.String()
    var_125 = {var_123: var_124}



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
    var_4 = False
    var_5 = 1
    var_6 = 10
    var_7 = '[a-z]+'
    var_8 = module_0.String(max_length=var_6, min_length=var_5, pattern=var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = True
    var_11 = module_0.String()
    var_12 = module_1.to_json_schema(var_11)
    var_13 = 100
    var_14 = module_0.Integer(minimum=var_4, maximum=var_13)
    var_15 = module_1.to_json_schema(var_14)
    var_16 = 0.5
    var_17 = 99.9
    var_18 = module_0.Float(exclusive_minimum=var_16, exclusive_maximum=var_17)
    var_19 = module_1.to_json_schema(var_18)
    var_20 = module_0.Boolean()
    var_21 = module_1.to_json_schema(var_20)
    var_22 = True
    var_23 = module_0.Boolean()
    var_24 = module_1.to_json_schema(var_23)
    var_25 = module_0.String()
    var_26 = 5
    var_27 = module_0.Array(var_25, min_items=var_22, max_items=var_26)
    var_28 = module_1.to_json_schema(var_27)
    var_29 = module_0.Integer()
    var_30 = True
    var_31 = module_0.Array(var_29, unique_items=var_30)
    var_32 = module_1.to_json_schema(var_31)
    var_33 = 'name'
    var_34 = 'age'
    var_35 = module_0.String()
    var_36 = module_0.Integer()
    var_37 = {var_33: var_35, var_34: var_36}
    var_38 = [var_33]
    var_39 = module_0.Object(properties=var_37, required=var_38)
    var_40 = module_1.to_json_schema(var_39)
    var_41 = module_0.Object(additional_properties=var_4)
    var_42 = module_1.to_json_schema(var_41)
    var_43 = 'a'
    var_44 = 'Option A'
    var_45 = (var_43, var_44)
    var_46 = 'b'
    var_47 = 'Option B'
    var_48 = (var_46, var_47)
    var_49 = [var_45, var_48]
    var_50 = module_0.Choice(choices=var_49)
    var_51 = module_1.to_json_schema(var_50)
    var_52 = 'fixed_value'
    var_53 = module_0.Const(var_52)
    var_54 = module_1.to_json_schema(var_53)
    var_55 = module_0.String()
    var_56 = module_0.Integer()
    var_57 = [var_55, var_56]
    var_58 = module_0.Union(var_57)
    var_59 = module_1.to_json_schema(var_58)
    var_60 = 'anyOf'
    var_61 = var_59[var_60]
    var_62 = len(var_61)
    assert var_62 == 2
    var_63 = module_0.String()
    var_64 = module_0.Integer()
    var_65 = [var_63, var_64]
    var_66 = module_2.OneOf(var_65)
    var_67 = module_1.to_json_schema(var_66)
    var_68 = 'oneOf'
    var_69 = var_67[var_68]
    var_70 = len(var_69)
    assert var_70 == 2
    var_71 = module_0.String()
    var_72 = module_0.Object()
    var_73 = [var_71, var_72]
    var_74 = module_2.AllOf(var_73)
    var_75 = module_1.to_json_schema(var_74)
    var_76 = 'allOf'
    var_77 = var_75[var_76]
    var_78 = len(var_77)
    assert var_78 == 2
    var_79 = module_0.String()
    var_80 = module_2.Not(var_79)
    var_81 = module_1.to_json_schema(var_80)
    var_82 = module_0.String()
    var_83 = module_0.Integer()
    var_84 = module_0.Boolean()
    var_85 = module_2.IfThenElse(var_82, var_83, var_84)
    var_86 = module_1.to_json_schema(var_85)
    var_87 = module_0.String()
    var_88 = module_0.Integer()
    var_89 = module_2.IfThenElse(var_87, var_88)
    var_90 = module_1.to_json_schema(var_89)
    var_91 = 'CustomType'
    var_92 = module_3.Definitions()
    var_93 = module_3.Reference(var_91, var_92)
    var_94 = module_1.to_json_schema(var_93)
    var_95 = 'test_value'
    var_96 = module_0.String()
    var_97 = module_1.to_json_schema(var_96)
    var_98 = 'default'
    var_99 = module_0.String()
    var_100 = module_0.Integer()
    var_101 = {var_33: var_99, var_34: var_100}
    var_102 = [var_33]
    var_103 = module_3.Schema(var_101)
    var_104 = module_1.to_json_schema(var_103)
    var_105 = module_3.Definitions()
    var_106 = module_0.String()
    var_107 = {var_33: var_106}
    var_108 = module_1.to_json_schema(var_105)



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
    var_14 = True
    var_15 = module_0.String(allow_blank=var_14)
    var_16 = module_1.to_json_schema(var_15)
    var_17 = 100
    var_18 = 5
    var_19 = 95
    var_20 = module_0.Integer(minimum=var_4, maximum=var_17, exclusive_minimum=var_18, exclusive_maximum=var_19, multiple_of=var_18)
    var_21 = module_1.to_json_schema(var_20)
    var_22 = module_0.Float(minimum=var_4, maximum=var_14)
    var_23 = module_1.to_json_schema(var_22)
    var_24 = module_0.Boolean()
    var_25 = module_1.to_json_schema(var_24)
    var_26 = True
    var_27 = module_0.Boolean()
    var_28 = module_1.to_json_schema(var_27)
    var_29 = module_0.String()
    var_30 = True
    var_31 = module_0.Array(var_29, min_items=var_26, max_items=var_6, unique_items=var_30)
    var_32 = module_1.to_json_schema(var_31)
    var_33 = module_0.String()
    var_34 = module_0.Integer()
    var_35 = [var_33, var_34]
    var_36 = module_0.Array(var_35)
    var_37 = module_1.to_json_schema(var_36)
    var_38 = 'items'
    var_39 = var_37[var_38]
    var_40 = var_37[var_38]
    var_41 = len(var_40)
    assert var_41 == 2
    var_42 = module_0.Array(additional_items=var_4)
    var_43 = module_1.to_json_schema(var_42)
    var_44 = module_0.String()
    var_45 = module_0.Array(additional_items=var_44)
    var_46 = module_1.to_json_schema(var_45)
    var_47 = 'additionalItems'
    var_48 = var_46[var_47]
    var_49 = 'name'
    var_50 = 'age'
    var_51 = module_0.String()
    var_52 = module_0.Integer()
    var_53 = {var_49: var_51, var_50: var_52}
    var_54 = [var_49]
    var_55 = module_0.Object(properties=var_53, min_properties=var_30, max_properties=var_6, required=var_54)
    var_56 = module_1.to_json_schema(var_55)
    var_57 = '^S_'
    var_58 = module_0.String()
    var_59 = {var_57: var_58}
    var_60 = module_0.Object(pattern_properties=var_59)
    var_61 = module_1.to_json_schema(var_60)
    var_62 = module_0.Object(additional_properties=var_4)
    var_63 = module_1.to_json_schema(var_62)
    var_64 = module_0.String()
    var_65 = module_0.Object(additional_properties=var_64)
    var_66 = module_1.to_json_schema(var_65)
    var_67 = 'additionalProperties'
    var_68 = var_66[var_67]
    var_69 = module_0.String(pattern=var_7)
    var_70 = module_0.Object(property_names=var_69)
    var_71 = module_1.to_json_schema(var_70)
    var_72 = 'a'
    var_73 = 'Option A'
    var_74 = (var_72, var_73)
    var_75 = 'b'
    var_76 = 'Option B'
    var_77 = (var_75, var_76)
    var_78 = [var_74, var_77]
    var_79 = module_0.Choice(choices=var_78)
    var_80 = module_1.to_json_schema(var_79)
    var_81 = 'constant_value'
    var_82 = module_0.Const(var_81)
    var_83 = module_1.to_json_schema(var_82)
    var_84 = module_0.String()
    var_85 = module_0.Integer()
    var_86 = [var_84, var_85]
    var_87 = module_0.Union(var_86)
    var_88 = module_1.to_json_schema(var_87)
    var_89 = 'anyOf'
    var_90 = var_88[var_89]
    var_91 = len(var_90)
    assert var_91 == 2
    var_92 = module_0.String()
    var_93 = module_0.Integer()
    var_94 = [var_92, var_93]
    var_95 = module_2.OneOf(var_94)
    var_96 = module_1.to_json_schema(var_95)
    var_97 = 'oneOf'
    var_98 = var_96[var_97]
    var_99 = len(var_98)
    assert var_99 == 2
    var_100 = module_0.String()
    var_101 = 'A'
    var_102 = (var_72, var_101)
    var_103 = 'B'
    var_104 = (var_75, var_103)
    var_105 = [var_102, var_104]
    var_106 = module_0.Choice(choices=var_105)
    var_107 = [var_100, var_106]
    var_108 = module_2.AllOf(var_107)
    var_109 = module_1.to_json_schema(var_108)
    var_110 = 'allOf'
    var_111 = var_109[var_110]
    var_112 = len(var_111)
    assert var_112 == 2
    var_113 = module_0.String()
    var_114 = module_2.Not(var_113)
    var_115 = module_1.to_json_schema(var_114)
    var_116 = (var_72, var_101)
    var_117 = [var_116]
    var_118 = module_0.Choice(choices=var_117)
    var_119 = module_0.String()
    var_120 = module_0.Integer()
    var_121 = module_2.IfThenElse(var_118, var_119, var_120)
    var_122 = module_1.to_json_schema(var_121)
    var_123 = (var_72, var_101)
    var_124 = [var_123]
    var_125 = module_0.Choice(choices=var_124)
    var_126 = module_0.String()
    var_127 = module_2.IfThenElse(var_125, var_126)
    var_128 = module_1.to_json_schema(var_127)
    var_129 = 'User'
    var_130 = module_0.String()
    var_131 = {var_49: var_130}
    var_132 = module_0.Object(properties=var_131)
    var_133 = {var_129: var_132}
    var_134 = module_3.Reference(var_129, var_133)
    var_135 = module_1.to_json_schema(var_134)



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
    var_14 = True
    var_15 = module_0.String(allow_blank=var_14)
    var_16 = module_1.to_json_schema(var_15)
    var_17 = 'minLength'
    var_18 = 100
    var_19 = 99
    var_20 = 5
    var_21 = module_0.Integer(minimum=var_4, maximum=var_18, exclusive_minimum=var_14, exclusive_maximum=var_19, multiple_of=var_20)
    var_22 = module_1.to_json_schema(var_21)
    var_23 = True
    var_24 = module_0.Integer()
    var_25 = module_1.to_json_schema(var_24)
    var_26 = module_0.Float(minimum=var_4, maximum=var_23)
    var_27 = module_1.to_json_schema(var_26)
    var_28 = True
    var_29 = module_0.Float()
    var_30 = module_1.to_json_schema(var_29)
    var_31 = module_0.Boolean()
    var_32 = module_1.to_json_schema(var_31)
    var_33 = True
    var_34 = module_0.Boolean()
    var_35 = module_1.to_json_schema(var_34)
    var_36 = module_0.String()
    var_37 = True
    var_38 = module_0.Array(var_36, min_items=var_33, max_items=var_6, unique_items=var_37)
    var_39 = module_1.to_json_schema(var_38)
    var_40 = True
    var_41 = module_0.Array()
    var_42 = module_1.to_json_schema(var_41)
    var_43 = module_0.String()
    var_44 = module_0.Integer()
    var_45 = [var_43, var_44]
    var_46 = module_0.Array(var_45)
    var_47 = module_1.to_json_schema(var_46)
    var_48 = 'items'
    var_49 = var_47[var_48]
    var_50 = var_47[var_48]
    var_51 = len(var_50)
    assert var_51 == 2
    var_52 = module_0.Array(additional_items=var_4)
    var_53 = module_1.to_json_schema(var_52)
    var_54 = module_0.String()
    var_55 = module_0.Array(additional_items=var_54)
    var_56 = module_1.to_json_schema(var_55)
    var_57 = 'additionalItems'
    var_58 = var_56[var_57]
    var_59 = 'name'
    var_60 = 'age'
    var_61 = module_0.String()
    var_62 = module_0.Integer()
    var_63 = {var_59: var_61, var_60: var_62}
    var_64 = [var_59]
    var_65 = module_0.Object(properties=var_63, min_properties=var_40, max_properties=var_6, required=var_64)
    var_66 = module_1.to_json_schema(var_65)
    var_67 = True
    var_68 = module_0.Object()
    var_69 = module_1.to_json_schema(var_68)
    var_70 = '^S_'
    var_71 = module_0.String()
    var_72 = {var_70: var_71}
    var_73 = module_0.Object(pattern_properties=var_72)
    var_74 = module_1.to_json_schema(var_73)
    var_75 = module_0.Object(additional_properties=var_4)
    var_76 = module_1.to_json_schema(var_75)
    var_77 = module_0.String()
    var_78 = module_0.Object(additional_properties=var_77)
    var_79 = module_1.to_json_schema(var_78)
    var_80 = 'additionalProperties'
    var_81 = var_79[var_80]
    var_82 = module_0.String(pattern=var_7)
    var_83 = module_0.Object(property_names=var_82)
    var_84 = module_1.to_json_schema(var_83)
    var_85 = module_0.String()
    var_86 = module_0.Integer()
    var_87 = {var_59: var_85, var_60: var_86}
    var_88 = [var_59]
    var_89 = module_3.Schema(var_87)
    var_90 = module_1.to_json_schema(var_89)
    var_91 = 'a'
    var_92 = 'A'
    var_93 = (var_91, var_92)
    var_94 = 'b'
    var_95 = 'B'
    var_96 = (var_94, var_95)
    var_97 = [var_93, var_96]
    var_98 = module_0.Choice(choices=var_97)
    var_99 = module_1.to_json_schema(var_98)
    var_100 = 'constant_value'
    var_101 = module_0.Const(var_100)
    var_102 = module_1.to_json_schema(var_101)
    var_103 = module_0.String()
    var_104 = module_0.Integer()
    var_105 = [var_103, var_104]
    var_106 = module_0.Union(var_105)
    var_107 = module_1.to_json_schema(var_106)
    var_108 = 'anyOf'
    var_109 = var_107[var_108]
    var_110 = len(var_109)
    assert var_110 == 2
    var_111 = module_0.String()
    var_112 = module_0.Integer()
    var_113 = [var_111, var_112]
    var_114 = module_2.OneOf(var_113)
    var_115 = module_1.to_json_schema(var_114)
    var_116 = 'oneOf'
    var_117 = var_115[var_116]
    var_118 = len(var_117)
    assert var_118 == 2
    var_119 = module_0.String()
    var_120 = module_0.Object()
    var_121 = [var_119, var_120]
    var_122 = module_2.AllOf(var_121)
    var_123 = module_1.to_json_schema(var_122)
    var_124 = 'allOf'
    var_125 = var_123[var_124]
    var_126 = len(var_125)
    assert var_126 == 2
    var_127 = module_0.String()
    var_128 = module_2.Not(var_127)
    var_129 = module_1.to_json_schema(var_128)
    var_130 = module_0.String()
    var_131 = module_0.Integer()
    var_132 = module_0.Boolean()
    var_133 = module_2.IfThenElse(var_130, var_131, var_132)
    var_134 = module_1.to_json_schema(var_133)



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
    var_6 = '^[a-z]+$'
    var_7 = 'email'
    var_8 = module_0.String(max_length=var_5, min_length=var_4, pattern=var_6, format=var_7)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = True
    var_11 = module_0.String()
    var_12 = module_1.to_json_schema(var_11)
    var_13 = True
    var_14 = module_0.String(allow_blank=var_13)
    var_15 = module_1.to_json_schema(var_14)
    var_16 = 0
    var_17 = 100
    var_18 = 5
    var_19 = 95
    var_20 = module_0.Integer(minimum=var_16, maximum=var_17, exclusive_minimum=var_18, exclusive_maximum=var_19, multiple_of=var_18)
    var_21 = module_1.to_json_schema(var_20)
    var_22 = True
    var_23 = module_0.Float()
    var_24 = module_1.to_json_schema(var_23)
    var_25 = module_0.Boolean()
    var_26 = module_1.to_json_schema(var_25)
    var_27 = True
    var_28 = module_0.Boolean()
    var_29 = module_1.to_json_schema(var_28)
    var_30 = module_0.String()
    var_31 = True
    var_32 = module_0.Array(var_30, min_items=var_27, max_items=var_5, unique_items=var_31)
    var_33 = module_1.to_json_schema(var_32)
    var_34 = module_0.String()
    var_35 = module_0.Integer()
    var_36 = [var_34, var_35]
    var_37 = module_0.Array(var_36)
    var_38 = module_1.to_json_schema(var_37)
    var_39 = 'items'
    var_40 = var_38[var_39]
    var_41 = var_38[var_39]
    var_42 = len(var_41)
    assert var_42 == 2
    var_43 = module_0.String()
    var_44 = True
    var_45 = module_0.Array(var_43, var_44)
    var_46 = module_1.to_json_schema(var_45)
    var_47 = module_0.String()
    var_48 = module_0.Integer()
    var_49 = module_0.Array(var_47, var_48)
    var_50 = module_1.to_json_schema(var_49)
    var_51 = 'additionalItems'
    var_52 = var_50[var_51]
    var_53 = 'name'
    var_54 = 'age'
    var_55 = module_0.String()
    var_56 = module_0.Integer()
    var_57 = {var_53: var_55, var_54: var_56}
    var_58 = [var_53]
    var_59 = module_0.Object(properties=var_57, min_properties=var_44, max_properties=var_18, required=var_58)
    var_60 = module_1.to_json_schema(var_59)
    var_61 = '^S_'
    var_62 = module_0.String()
    var_63 = {var_61: var_62}
    var_64 = module_0.Object(pattern_properties=var_63)
    var_65 = module_1.to_json_schema(var_64)
    var_66 = True
    var_67 = module_0.Object(additional_properties=var_66)
    var_68 = module_1.to_json_schema(var_67)
    var_69 = module_0.String(pattern=var_6)
    var_70 = module_0.Object(property_names=var_69)
    var_71 = module_1.to_json_schema(var_70)
    var_72 = 'a'
    var_73 = 'Option A'
    var_74 = (var_72, var_73)
    var_75 = 'b'
    var_76 = 'Option B'
    var_77 = (var_75, var_76)
    var_78 = [var_74, var_77]
    var_79 = module_0.Choice(choices=var_78)
    var_80 = module_1.to_json_schema(var_79)
    var_81 = 'constant_value'
    var_82 = module_0.Const(var_81)
    var_83 = module_1.to_json_schema(var_82)
    var_84 = module_0.String()
    var_85 = module_0.Integer()
    var_86 = [var_84, var_85]
    var_87 = module_0.Union(var_86)
    var_88 = module_1.to_json_schema(var_87)
    var_89 = 'anyOf'
    var_90 = var_88[var_89]
    var_91 = len(var_90)
    assert var_91 == 2
    var_92 = module_0.String()
    var_93 = module_0.Integer()
    var_94 = [var_92, var_93]
    var_95 = module_2.OneOf(var_94)
    var_96 = module_1.to_json_schema(var_95)
    var_97 = 'oneOf'
    var_98 = var_96[var_97]
    var_99 = len(var_98)
    assert var_99 == 2
    var_100 = module_0.String(min_length=var_66)
    var_101 = module_0.String(max_length=var_5)
    var_102 = [var_100, var_101]
    var_103 = module_2.AllOf(var_102)
    var_104 = module_1.to_json_schema(var_103)
    var_105 = 'allOf'
    var_106 = var_104[var_105]
    var_107 = len(var_106)
    assert var_107 == 2
    var_108 = module_0.String()
    var_109 = module_2.Not(var_108)
    var_110 = module_1.to_json_schema(var_109)
    var_111 = 'A'
    var_112 = (var_72, var_111)
    var_113 = [var_112]
    var_114 = module_0.Choice(choices=var_113)
    var_115 = module_0.String()
    var_116 = module_0.Integer()
    var_117 = module_2.IfThenElse(var_114, var_115, var_116)
    var_118 = module_1.to_json_schema(var_117)
    var_119 = (var_72, var_111)
    var_120 = [var_119]
    var_121 = module_0.Choice(choices=var_120)
    var_122 = module_2.IfThenElse(var_121)
    var_123 = module_1.to_json_schema(var_122)
    var_124 = module_0.String()
    var_125 = module_0.Integer()
    var_126 = {var_53: var_124, var_54: var_125}
    var_127 = [var_53]
    var_128 = module_3.Schema(var_126)
    var_129 = module_1.to_json_schema(var_128)
    var_130 = 'User'
    var_131 = 'Product'
    var_132 = module_0.String()
    var_133 = {var_53: var_132}
    var_134 = module_0.Object(properties=var_133)
    var_135 = 'title'
    var_136 = module_0.String()
    var_137 = {var_135: var_136}
    var_138 = module_0.Object(properties=var_137)
    var_139 = {var_130: var_134, var_131: var_138}
    var_140 = 'test'
    var_141 = module_0.String()
    var_142 = module_1.to_json_schema(var_141)
    var_143 = module_1.to_json_schema(var_0)



# Parsed testcases at query #34
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
    var_4 = False
    var_5 = module_0.String()
    var_6 = module_1.to_json_schema(var_5)
    var_7 = True
    var_8 = module_0.String()
    var_9 = module_1.to_json_schema(var_8)
    var_10 = 2
    var_11 = 10
    var_12 = '^[a-z]+$'
    var_13 = 'email'
    var_14 = module_0.String(max_length=var_11, min_length=var_10, pattern=var_12, format=var_13)
    var_15 = module_1.to_json_schema(var_14)
    var_16 = module_0.Integer()
    var_17 = module_1.to_json_schema(var_16)
    var_18 = 100
    var_19 = 5
    var_20 = module_0.Integer(minimum=var_4, maximum=var_18, multiple_of=var_19)
    var_21 = module_1.to_json_schema(var_20)
    var_22 = module_0.Float()
    var_23 = module_1.to_json_schema(var_22)
    var_24 = module_0.Boolean()
    var_25 = module_1.to_json_schema(var_24)
    var_26 = module_0.Boolean()
    var_27 = module_1.to_json_schema(var_26)
    var_28 = module_0.String()
    var_29 = module_0.Array(var_28)
    var_30 = module_1.to_json_schema(var_29)
    var_31 = module_0.Integer()
    var_32 = module_0.Array(var_31, min_items=var_7, max_items=var_11, unique_items=var_7)
    var_33 = module_1.to_json_schema(var_32)
    var_34 = 'name'
    var_35 = 'age'
    var_36 = module_0.String()
    var_37 = module_0.Integer()
    var_38 = {var_34: var_36, var_35: var_37}
    var_39 = [var_34]
    var_40 = module_0.Object(properties=var_38, required=var_39)
    var_41 = module_1.to_json_schema(var_40)
    var_42 = 'id'
    var_43 = module_0.Integer()
    var_44 = {var_42: var_43}
    var_45 = module_0.String()
    var_46 = module_0.Object(properties=var_44, additional_properties=var_45)
    var_47 = module_1.to_json_schema(var_46)
    var_48 = 'a'
    var_49 = 'option_a'
    var_50 = (var_48, var_49)
    var_51 = 'b'
    var_52 = 'option_b'
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
    var_76 = module_0.String()
    var_77 = module_0.Object()
    var_78 = [var_76, var_77]
    var_79 = module_2.AllOf(var_78)
    var_80 = module_1.to_json_schema(var_79)
    var_81 = 'allOf'
    var_82 = var_80[var_81]
    var_83 = len(var_82)
    assert var_83 == 2
    var_84 = module_0.String()
    var_85 = module_2.Not(var_84)
    var_86 = module_1.to_json_schema(var_85)
    var_87 = module_0.String()
    var_88 = module_0.Integer()
    var_89 = module_0.Boolean()
    var_90 = module_2.IfThenElse(var_87, var_88, var_89)
    var_91 = module_1.to_json_schema(var_90)
    var_92 = module_0.String()
    var_93 = module_0.Integer()
    var_94 = module_2.IfThenElse(var_92, var_93)
    var_95 = module_1.to_json_schema(var_94)
    var_96 = 'StringDef'
    var_97 = 'IntegerDef'
    var_98 = module_0.String()
    var_99 = module_0.Integer()
    var_100 = {var_96: var_98, var_97: var_99}
    var_101 = 'default_value'
    var_102 = module_0.String()
    var_103 = module_1.to_json_schema(var_102)
    var_104 = 'items'
    var_105 = 'status'
    var_106 = module_0.String(min_length=var_7)
    var_107 = module_0.Integer()
    var_108 = {var_42: var_107}
    var_109 = module_0.Object(properties=var_108)
    var_110 = module_0.Array(var_109)
    var_111 = 'active'
    var_112 = 'Active'
    var_113 = (var_111, var_112)
    var_114 = 'inactive'
    var_115 = 'Inactive'
    var_116 = (var_114, var_115)
    var_117 = [var_113, var_116]
    var_118 = module_0.Choice(choices=var_117)
    var_119 = {var_34: var_106, var_104: var_110, var_105: var_118}
    var_120 = [var_34, var_104]
    var_121 = module_0.Object(properties=var_119, required=var_120)
    var_122 = module_1.to_json_schema(var_121)



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
    var_4 = False
    var_5 = True
    var_6 = None
    var_7 = 10
    var_8 = module_0.String(allow_blank=var_5, max_length=var_7, min_length=var_6, format=var_6)
    var_9 = module_1.to_json_schema(var_8)
    var_10 = module_0.String(allow_blank=var_5, max_length=var_6, min_length=var_6, format=var_6)
    var_11 = module_1.to_json_schema(var_10)
    var_12 = 5
    var_13 = module_0.String(allow_blank=var_4, max_length=var_6, min_length=var_12, format=var_6)
    var_14 = module_1.to_json_schema(var_13)
    var_15 = 100
    var_16 = module_0.Integer(minimum=var_4, maximum=var_15, exclusive_minimum=var_6, exclusive_maximum=var_6, multiple_of=var_6)
    var_17 = module_1.to_json_schema(var_16)
    var_18 = module_0.Float(minimum=var_4, maximum=var_5, exclusive_minimum=var_6, exclusive_maximum=var_6, multiple_of=var_6)
    var_19 = module_1.to_json_schema(var_18)
    var_20 = module_0.Boolean()
    var_21 = module_1.to_json_schema(var_20)
    var_22 = module_0.Boolean()
    var_23 = module_1.to_json_schema(var_22)
    var_24 = module_0.Array(var_6, var_5, var_4, var_6, unique_items=var_4)
    var_25 = module_1.to_json_schema(var_24)
    var_26 = module_0.String(allow_blank=var_5, max_length=var_6, min_length=var_6, format=var_6)
    var_27 = module_0.Array(var_26, var_5, var_5, var_7, unique_items=var_4)
    var_28 = module_1.to_json_schema(var_27)
    var_29 = module_0.Object(properties=var_6, pattern_properties=var_6, additional_properties=var_6, property_names=var_6, min_properties=var_6, max_properties=var_6, required=var_6)
    var_30 = module_1.to_json_schema(var_29)
    var_31 = 'name'
    var_32 = 'age'
    var_33 = module_0.String(allow_blank=var_5, max_length=var_6, min_length=var_6, format=var_6)
    var_34 = module_0.Integer(minimum=var_6, maximum=var_6, exclusive_minimum=var_6, exclusive_maximum=var_6, multiple_of=var_6)
    var_35 = {var_31: var_33, var_32: var_34}
    var_36 = [var_31]
    var_37 = module_0.Object(properties=var_35, pattern_properties=var_6, additional_properties=var_6, property_names=var_6, min_properties=var_6, max_properties=var_6, required=var_36)
    var_38 = module_1.to_json_schema(var_37)
    var_39 = 'option1'
    var_40 = (var_39, var_39)
    var_41 = 'option2'
    var_42 = (var_41, var_41)
    var_43 = [var_40, var_42]
    var_44 = module_0.Choice(choices=var_43)
    var_45 = module_1.to_json_schema(var_44)
    var_46 = 42
    var_47 = module_0.Const(var_46)
    var_48 = module_1.to_json_schema(var_47)
    var_49 = module_0.String(allow_blank=var_5, max_length=var_6, min_length=var_6, format=var_6)
    var_50 = module_0.Integer(minimum=var_6, maximum=var_6, exclusive_minimum=var_6, exclusive_maximum=var_6, multiple_of=var_6)
    var_51 = [var_49, var_50]
    var_52 = module_0.Union(var_51)
    var_53 = module_1.to_json_schema(var_52)
    var_54 = 'anyOf'
    var_55 = var_53[var_54]
    var_56 = len(var_55)
    assert var_56 == 2
    var_57 = module_0.String(allow_blank=var_5, max_length=var_6, min_length=var_6, format=var_6)
    var_58 = module_0.Integer(minimum=var_6, maximum=var_6, exclusive_minimum=var_6, exclusive_maximum=var_6, multiple_of=var_6)
    var_59 = [var_57, var_58]
    var_60 = module_2.OneOf(var_59)
    var_61 = module_1.to_json_schema(var_60)
    var_62 = 'oneOf'
    var_63 = var_61[var_62]
    var_64 = len(var_63)
    assert var_64 == 2
    var_65 = module_0.String(allow_blank=var_5, max_length=var_6, min_length=var_6, format=var_6)
    var_66 = module_0.Object(properties=var_6, pattern_properties=var_6, additional_properties=var_6, property_names=var_6, min_properties=var_6, max_properties=var_6, required=var_6)
    var_67 = [var_65, var_66]
    var_68 = module_2.AllOf(var_67)
    var_69 = module_1.to_json_schema(var_68)
    var_70 = 'allOf'
    var_71 = var_69[var_70]
    var_72 = len(var_71)
    assert var_72 == 2
    var_73 = module_0.String(allow_blank=var_5, max_length=var_6, min_length=var_6, format=var_6)
    var_74 = module_2.Not(var_73)
    var_75 = module_1.to_json_schema(var_74)
    var_76 = module_0.String(allow_blank=var_5, max_length=var_6, min_length=var_6, format=var_6)
    var_77 = module_0.Integer(minimum=var_6, maximum=var_6, exclusive_minimum=var_6, exclusive_maximum=var_6, multiple_of=var_6)
    var_78 = module_2.IfThenElse(var_76, var_77, var_6)
    var_79 = module_1.to_json_schema(var_78)
    var_80 = module_3.Definitions()
    var_81 = module_1.to_json_schema(var_80)



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
    var_14 = True
    var_15 = module_0.String(allow_blank=var_14)
    var_16 = module_1.to_json_schema(var_15)
    var_17 = 100
    var_18 = 5
    var_19 = 95
    var_20 = module_0.Integer(minimum=var_4, maximum=var_17, exclusive_minimum=var_18, exclusive_maximum=var_19, multiple_of=var_18)
    var_21 = module_1.to_json_schema(var_20)
    var_22 = module_0.Float(minimum=var_4, maximum=var_14)
    var_23 = module_1.to_json_schema(var_22)
    var_24 = module_0.Boolean()
    var_25 = module_1.to_json_schema(var_24)
    var_26 = True
    var_27 = module_0.Boolean()
    var_28 = module_1.to_json_schema(var_27)
    var_29 = module_0.String()
    var_30 = True
    var_31 = module_0.Array(var_29, min_items=var_26, max_items=var_6, unique_items=var_30)
    var_32 = module_1.to_json_schema(var_31)
    var_33 = module_0.String()
    var_34 = module_0.Array(var_33, var_4)
    var_35 = module_1.to_json_schema(var_34)
    var_36 = 'name'
    var_37 = 'age'
    var_38 = module_0.String()
    var_39 = module_0.Integer()
    var_40 = {var_36: var_38, var_37: var_39}
    var_41 = [var_36]
    var_42 = module_0.Object(properties=var_40, min_properties=var_30, max_properties=var_6, required=var_41)
    var_43 = module_1.to_json_schema(var_42)
    var_44 = module_0.String()
    var_45 = {var_7: var_44}
    var_46 = module_0.Object(pattern_properties=var_45)
    var_47 = module_1.to_json_schema(var_46)
    var_48 = module_0.Object(additional_properties=var_4)
    var_49 = module_1.to_json_schema(var_48)
    var_50 = module_0.String(pattern=var_7)
    var_51 = module_0.Object(property_names=var_50)
    var_52 = module_1.to_json_schema(var_51)
    var_53 = 'a'
    var_54 = 'Option A'
    var_55 = (var_53, var_54)
    var_56 = 'b'
    var_57 = 'Option B'
    var_58 = (var_56, var_57)
    var_59 = [var_55, var_58]
    var_60 = module_0.Choice(choices=var_59)
    var_61 = module_1.to_json_schema(var_60)
    var_62 = 'constant_value'
    var_63 = module_0.Const(var_62)
    var_64 = module_1.to_json_schema(var_63)
    var_65 = module_0.String()
    var_66 = module_0.Integer()
    var_67 = [var_65, var_66]
    var_68 = module_0.Union(var_67)
    var_69 = module_1.to_json_schema(var_68)
    var_70 = 'anyOf'
    var_71 = var_69[var_70]
    var_72 = len(var_71)
    assert var_72 == 2
    var_73 = module_0.String()
    var_74 = module_0.Integer()
    var_75 = [var_73, var_74]
    var_76 = module_2.OneOf(var_75)
    var_77 = module_1.to_json_schema(var_76)
    var_78 = 'oneOf'
    var_79 = var_77[var_78]
    var_80 = len(var_79)
    assert var_80 == 2
    var_81 = module_0.String()
    var_82 = 'test'
    var_83 = module_0.Const(var_82)
    var_84 = [var_81, var_83]
    var_85 = module_2.AllOf(var_84)
    var_86 = module_1.to_json_schema(var_85)
    var_87 = 'allOf'
    var_88 = var_86[var_87]
    var_89 = len(var_88)
    assert var_89 == 2
    var_90 = 'A'
    var_91 = (var_53, var_90)
    var_92 = 'B'
    var_93 = (var_56, var_92)
    var_94 = [var_91, var_93]
    var_95 = module_0.Choice(choices=var_94)
    var_96 = module_0.String()
    var_97 = module_0.Integer()
    var_98 = module_2.IfThenElse(var_95, var_96, var_97)
    var_99 = module_1.to_json_schema(var_98)
    var_100 = module_0.String()
    var_101 = module_2.IfThenElse(var_100)
    var_102 = module_1.to_json_schema(var_101)
    var_103 = module_0.String()
    var_104 = module_2.Not(var_103)
    var_105 = module_1.to_json_schema(var_104)
    var_106 = 'User'
    var_107 = {}
    var_108 = module_3.Reference(var_106, var_107)
    var_109 = module_1.to_json_schema(var_108)
    var_110 = module_0.String()
    var_111 = {var_36: var_110}
    var_112 = module_0.Object(properties=var_111)
    var_113 = {var_106: var_112}
    var_114 = module_0.String()
    var_115 = module_0.Integer()
    var_116 = {var_36: var_114, var_37: var_115}
    var_117 = [var_36]
    var_118 = module_3.Schema(var_116)
    var_119 = module_1.to_json_schema(var_118)
    var_120 = module_1.to_json_schema(var_0)



# Parsed testcases at query #37
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
    var_6 = True
    var_7 = module_0.String()
    var_8 = module_1.to_json_schema(var_7)
    var_9 = 5
    var_10 = 10
    var_11 = '^[a-z]+$'
    var_12 = module_0.String(max_length=var_10, min_length=var_9, pattern=var_11)
    var_13 = module_1.to_json_schema(var_12)
    var_14 = 'email'
    var_15 = module_0.String(format=var_14)
    var_16 = module_1.to_json_schema(var_15)
    var_17 = module_0.Integer()
    var_18 = module_1.to_json_schema(var_17)
    var_19 = 0
    var_20 = 100
    var_21 = module_0.Integer(minimum=var_19, maximum=var_20, multiple_of=var_9)
    var_22 = module_1.to_json_schema(var_21)
    var_23 = module_0.Float()
    var_24 = module_1.to_json_schema(var_23)
    var_25 = module_0.Float(exclusive_minimum=var_19, exclusive_maximum=var_20)
    var_26 = module_1.to_json_schema(var_25)
    var_27 = module_0.Boolean()
    var_28 = module_1.to_json_schema(var_27)
    var_29 = module_0.Boolean()
    var_30 = module_1.to_json_schema(var_29)
    var_31 = module_0.String()
    var_32 = module_0.Array(var_31)
    var_33 = module_1.to_json_schema(var_32)
    var_34 = module_0.Integer()
    var_35 = module_0.Array(var_34, min_items=var_6, max_items=var_10, unique_items=var_6)
    var_36 = module_1.to_json_schema(var_35)
    var_37 = module_0.String()
    var_38 = module_0.Integer()
    var_39 = [var_37, var_38]
    var_40 = module_0.Array(var_39)
    var_41 = module_1.to_json_schema(var_40)
    var_42 = 'items'
    var_43 = var_41[var_42]
    var_44 = var_41[var_42]
    var_45 = len(var_44)
    assert var_45 == 2
    var_46 = module_0.String()
    var_47 = module_0.Boolean()
    var_48 = module_0.Array(var_46, var_47)
    var_49 = module_1.to_json_schema(var_48)
    var_50 = module_0.String()
    var_51 = False
    var_52 = module_0.Array(var_50, var_51)
    var_53 = module_1.to_json_schema(var_52)
    var_54 = 'name'
    var_55 = 'age'
    var_56 = module_0.String()
    var_57 = module_0.Integer()
    var_58 = {var_54: var_56, var_55: var_57}
    var_59 = module_0.Object(properties=var_58)
    var_60 = module_1.to_json_schema(var_59)
    var_61 = module_0.String()
    var_62 = {var_54: var_61}
    var_63 = [var_54]
    var_64 = module_0.Object(properties=var_62, min_properties=var_6, max_properties=var_10, required=var_63)
    var_65 = module_1.to_json_schema(var_64)
    var_66 = '^S_'
    var_67 = module_0.String()
    var_68 = {var_66: var_67}
    var_69 = module_0.Object(pattern_properties=var_68)
    var_70 = module_1.to_json_schema(var_69)
    var_71 = module_0.String()
    var_72 = module_0.Object(additional_properties=var_71)
    var_73 = module_1.to_json_schema(var_72)
    var_74 = module_0.String(pattern=var_11)
    var_75 = module_0.Object(property_names=var_74)
    var_76 = module_1.to_json_schema(var_75)
    var_77 = 'a'
    var_78 = 'Option A'
    var_79 = (var_77, var_78)
    var_80 = 'b'
    var_81 = 'Option B'
    var_82 = (var_80, var_81)
    var_83 = [var_79, var_82]
    var_84 = module_0.Choice(choices=var_83)
    var_85 = module_1.to_json_schema(var_84)
    var_86 = 'constant_value'
    var_87 = module_0.Const(var_86)
    var_88 = module_1.to_json_schema(var_87)
    var_89 = module_0.String()
    var_90 = module_0.Integer()
    var_91 = [var_89, var_90]
    var_92 = module_0.Union(var_91)
    var_93 = module_1.to_json_schema(var_92)
    var_94 = 'anyOf'
    var_95 = var_93[var_94]
    var_96 = len(var_95)
    assert var_96 == 2
    var_97 = module_0.String()
    var_98 = module_0.Integer()
    var_99 = [var_97, var_98]
    var_100 = module_2.OneOf(var_99)
    var_101 = module_1.to_json_schema(var_100)
    var_102 = 'oneOf'
    var_103 = var_101[var_102]
    var_104 = len(var_103)
    assert var_104 == 2
    var_105 = module_0.String(min_length=var_6)
    var_106 = module_0.String(max_length=var_20)
    var_107 = [var_105, var_106]
    var_108 = module_2.AllOf(var_107)
    var_109 = module_1.to_json_schema(var_108)
    var_110 = 'allOf'
    var_111 = var_109[var_110]
    var_112 = len(var_111)
    assert var_112 == 2
    var_113 = module_0.String()
    var_114 = module_2.Not(var_113)
    var_115 = module_1.to_json_schema(var_114)
    var_116 = module_0.String()
    var_117 = module_0.Integer()
    var_118 = module_0.Boolean()
    var_119 = module_2.IfThenElse(var_116, var_117, var_118)
    var_120 = module_1.to_json_schema(var_119)
    var_121 = module_0.String()
    var_122 = module_0.Integer()
    var_123 = module_2.IfThenElse(var_121, var_122)
    var_124 = module_1.to_json_schema(var_123)
    var_125 = 'default_value'
    var_126 = module_0.String()
    var_127 = module_1.to_json_schema(var_126)
    var_128 = 'MyString'
    var_129 = 'MyInt'
    var_130 = module_0.String()
    var_131 = module_0.Integer()
    var_132 = {var_128: var_130, var_129: var_131}



# Parsed testcases at query #38
#--------------------------


import typesystem.fields as module_0
import typesystem.json_schema as module_1
import typesystem.composites as module_2
import re as module_3

def test_case_0():
    var_0 = module_0.Any()
    var_1 = module_1.to_json_schema(var_0)
    assert var_1 is True
    var_2 = module_2.NeverMatch()
    var_3 = module_1.to_json_schema(var_2)
    assert var_3 is False
    var_4 = False
    var_5 = True
    var_6 = module_0.String(allow_blank=var_5)
    var_7 = module_1.to_json_schema(var_6)
    var_8 = 5
    var_9 = module_0.String(allow_blank=var_4, min_length=var_8)
    var_10 = module_1.to_json_schema(var_9)
    var_11 = 10
    var_12 = module_0.String(max_length=var_11)
    var_13 = module_1.to_json_schema(var_12)
    var_14 = module_0.String()
    var_15 = module_1.to_json_schema(var_14)
    var_16 = 100
    var_17 = module_0.Integer(minimum=var_4, maximum=var_16)
    var_18 = module_1.to_json_schema(var_17)
    var_19 = module_0.Float()
    var_20 = module_1.to_json_schema(var_19)
    var_21 = module_0.Boolean()
    var_22 = module_1.to_json_schema(var_21)
    var_23 = module_0.Boolean()
    var_24 = module_1.to_json_schema(var_23)
    var_25 = module_0.String()
    var_26 = module_0.Array(var_25)
    var_27 = module_1.to_json_schema(var_26)
    var_28 = module_0.Array(min_items=var_5, max_items=var_8)
    var_29 = module_1.to_json_schema(var_28)
    var_30 = module_0.Array(unique_items=var_5)
    var_31 = module_1.to_json_schema(var_30)
    var_32 = 'name'
    var_33 = 'age'
    var_34 = module_0.String()
    var_35 = module_0.Integer()
    var_36 = {var_32: var_34, var_33: var_35}
    var_37 = module_0.Object(properties=var_36)
    var_38 = module_1.to_json_schema(var_37)
    var_39 = module_0.String()
    var_40 = {var_32: var_39}
    var_41 = [var_32]
    var_42 = module_0.Object(properties=var_40, required=var_41)
    var_43 = module_1.to_json_schema(var_42)
    var_44 = 'a'
    var_45 = 'Option A'
    var_46 = (var_44, var_45)
    var_47 = 'b'
    var_48 = 'Option B'
    var_49 = (var_47, var_48)
    var_50 = [var_46, var_49]
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
    var_72 = module_0.String(min_length=var_5)
    var_73 = module_0.String(max_length=var_11)
    var_74 = [var_72, var_73]
    var_75 = module_2.AllOf(var_74)
    var_76 = module_1.to_json_schema(var_75)
    var_77 = 'allOf'
    var_78 = var_76[var_77]
    var_79 = len(var_78)
    assert var_79 == 2
    var_80 = module_0.String()
    var_81 = module_2.Not(var_80)
    var_82 = module_1.to_json_schema(var_81)
    var_83 = module_0.String()
    var_84 = module_0.Integer()
    var_85 = module_0.Boolean()
    var_86 = module_2.IfThenElse(var_83, var_84, var_85)
    var_87 = module_1.to_json_schema(var_86)
    var_88 = module_0.String()
    var_89 = module_2.IfThenElse(var_88)
    var_90 = module_1.to_json_schema(var_89)
    var_91 = 'User'
    var_92 = module_0.String()
    var_93 = {var_32: var_92}
    var_94 = module_0.Object(properties=var_93)
    var_95 = {var_91: var_94}
    var_96 = 'default_value'
    var_97 = module_0.String()
    var_98 = module_1.to_json_schema(var_97)
    var_99 = 'default'
    var_100 = '^[a-z]+$'
    var_101 = module_3.compile(var_100)
    var_102 = module_0.String(pattern=var_101)
    var_103 = module_1.to_json_schema(var_102)
    var_104 = 'email'
    var_105 = module_0.String(format=var_104)
    var_106 = module_1.to_json_schema(var_105)
    var_107 = module_0.Integer(exclusive_minimum=var_4, exclusive_maximum=var_16)
    var_108 = module_1.to_json_schema(var_107)
    var_109 = 0.5
    var_110 = module_0.Float(multiple_of=var_109)
    var_111 = module_1.to_json_schema(var_110)
    var_112 = module_0.Array(additional_items=var_4)
    var_113 = module_1.to_json_schema(var_112)
    var_114 = module_0.String()
    var_115 = module_0.Array(additional_items=var_114)
    var_116 = module_1.to_json_schema(var_115)
    var_117 = 'additionalItems'
    var_118 = var_116[var_117]
    var_119 = '^S_'
    var_120 = '^I_'
    var_121 = module_0.String()
    var_122 = module_0.Integer()
    var_123 = {var_119: var_121, var_120: var_122}
    var_124 = module_0.Object(pattern_properties=var_123)
    var_125 = module_1.to_json_schema(var_124)
    var_126 = module_3.compile(var_100)
    var_127 = module_0.String(pattern=var_126)
    var_128 = module_0.Object(property_names=var_127)



