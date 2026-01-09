####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.tokenize.tokens as module_0


def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Token(var_4)
    var_6 = {var_0: var_2}
    var_7 = module_0.Token(var_6)
    var_8 = 123
    var_9 = 'twenty'
    var_10 = {var_0: var_8, var_1: var_9}
    var_11 = module_0.Token(var_10)
    var_12 = 'person'
    var_13 = {var_0: var_2, var_1: var_3}
    var_14 = {var_12: var_13}
    var_15 = module_0.Token(var_14)
    var_16 = {var_0: var_2}
    var_17 = {var_12: var_16}
    var_18 = module_0.Token(var_17)
    var_19 = 'All test cases pass'
    var_20 = print(var_19)



# Parsed testcases at query #2
#--------------------------


import typesystem.fields as module_1
import typesystem.tokenize.positional_validation as module_2


def test_case_0():
    var_0 = 'name'
    var_1 = 'John Doe'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0.Token(var_2)
    var_5 = 10
    var_6 = var_1 * var_5
    var_7 = {var_0: var_6}
    var_8 = module_0.Token(var_7)
    var_9 = {}
    var_10 = module_0.Token(var_9)
    var_11 = 'age'
    var_12 = 30
    var_13 = {var_0: var_1, var_11: var_12}
    var_14 = module_0.Token(var_13)
    var_15 = {var_0: var_1, var_11: var_12}
    var_16 = module_0.Token(var_15)
    var_17 = module_1.String(max_length=var_5)
    var_18 = module_2.validate_with_positions(token=var_16, validator=var_17)
    var_19 = module_0.Token(var_1)
    var_20 = module_1.String(max_length=var_5)
    var_21 = module_2.validate_with_positions(token=var_19, validator=var_20)
    assert var_21 == 'John Doe'
    var_22 = var_1 * var_5
    var_23 = module_0.Token(var_22)
    var_24 = module_1.String(max_length=var_5)
    var_25 = module_2.validate_with_positions(token=var_23, validator=var_24)
    var_26 = module_0.Token(var_3)
    var_27 = module_1.String(max_length=var_5)
    var_28 = module_2.validate_with_positions(token=var_26, validator=var_27)
    var_29 = {var_28: var_1, var_11: var_12}
    var_30 = module_0.Token(var_29)
    var_31 = module_1.String(max_length=var_5)
    var_32 = module_2.validate_with_positions(token=var_30, validator=var_31)
    var_33 = {var_32: var_1, var_11: var_12}
    var_34 = module_0.Token(var_33)
    var_35 = module_2.validate_with_positions(token=var_34, validator=var_31)
    var_36 = {var_32: var_1, var_11: var_12}
    var_37 = module_0.Token(var_36)
    var_38 = module_2.validate_with_positions(token=var_37, validator=var_31)
    var_39 = {var_32: var_1, var_11: var_12}
    var_40 = module_0.Token(var_39)
    var_41 = module_2.validate_with_positions(token=var_40, validator=var_31)
    var_42 = {var_32: var_1, var_11: var_12}
    var_43 = module_0.Token(var_42)
    var_44 = module_2.validate_with_positions(token=var_43, validator=var_31)
    var_45 = {var_32: var_1, var_11: var_12}
    var_46 = module_0.Token(var_45)
    var_47 = module_2.validate_with_positions(token=var_46, validator=var_31)
    var_48 = {var_32: var_1, var_11: var_12}
    var_49 = module_0.Token(var_48)
    var_50 = module_2.validate_with_positions(token=var_49, validator=var_31)
    var_51 = {var_32: var_1, var_11: var_12}
    var_52 = module_0.Token(var_51)
    var_53 = module_2.validate_with_positions(token=var_52, validator=var_31)
    var_54 = {var_32: var_1, var_11: var_12}
    var_55 = module_0.Token(var_54)
    var_56 = module_2.validate_with_positions(token=var_55, validator=var_31)
    var_57 = {var_32: var_1, var_11: var_12}
    var_58 = module_0.Token(var_57)
    var_59 = module_2.validate_with_positions(token=var_58, validator=var_31)
    var_60 = {var_32: var_1, var_11: var_12}
    var_61 = module_0.Token(var_60)
    var_62 = module_2.validate_with_positions(token=var_61, validator=var_31)
    var_63 = {var_32: var_1, var_11: var_12}
    var_64 = module_0.Token(var_63)
    var_65 = module_2.validate_with_positions(token=var_64, validator=var_31)
    var_66 = {var_32: var_1, var_11: var_12}
    var_67 = module_0.Token(var_66)
    var_68 = module_2.validate_with_positions(token=var_67, validator=var_31)
    var_69 = {var_32: var_1, var_11: var_12}
    var_70 = module_0.Token(var_69)
    var_71 = module_2.validate_with_positions(token=var_70, validator=var_31)
    var_72 = {var_32: var_1, var_11: var_12}
    var_73 = module_0.Token(var_72)
    var_74 = module_2.validate_with_positions(token=var_73, validator=var_31)
    var_75 = {var_32: var_1, var_11: var_12}
    var_76 = module_0.Token(var_75)
    var_77 = module_2.validate_with_positions(token=var_76, validator=var_31)
    var_78 = {var_32: var_1, var_11: var_12}
    var_79 = module_0.Token(var_78)
    var_80 = module_2.validate_with_positions(token=var_79, validator=var_31)
    var_81 = {var_32: var_1, var_11: var_12}
    var_82 = module_0.Token(var_81)
    var_83 = module_2.validate_with_positions(token=var_82, validator=var_31)
    var_84 = {var_32: var_1, var_11: var_12}
    var_85 = module_0.Token(var_84)
    var_86 = module_2.validate_with_positions(token=var_85, validator=var_31)
    var_87 = {var_32: var_1, var_11: var_12}
    var_88 = module_0.Token(var_87)
    var_89 = module_2.validate_with_positions(token=var_88, validator=var_31)
    var_90 = {var_32: var_1, var_11: var_12}
    var_91 = module_0.Token(var_90)
    var_92 = module_2.validate_with_positions(token=var_91, validator=var_31)
    var_93 = {var_32: var_1, var_11: var_12}
    var_94 = module_0.Token(var_93)



# Parsed testcases at query #3
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0.Token(var_2)
    var_5 = {}
    var_6 = module_0.Token(var_5)
    var_7 = 'John Doe Smith'
    var_8 = {var_0: var_7}
    var_9 = module_0.Token(var_8)
    var_10 = module_1.String()
    var_11 = 'nested'
    var_12 = 'age'
    var_13 = 25
    var_14 = {var_12: var_13}
    var_15 = {var_11: var_14}
    var_16 = module_0.Token(var_15)
    var_17 = 'All tests passed!'
    var_18 = print(var_17)



# Parsed testcases at query #4
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Token(var_4)
    var_6 = {var_0: var_2}
    var_7 = module_0.Token(var_6)
    var_8 = 123
    var_9 = 'twenty'
    var_10 = {var_0: var_8, var_1: var_9}
    var_11 = module_0.Token(var_10)
    var_12 = 'person'
    var_13 = {var_0: var_8}
    var_14 = {var_12: var_13}
    var_15 = module_0.Token(var_14)
    var_16 = {var_0: var_2}
    var_17 = module_0.Token(var_16)
    var_18 = 'required'
    var_19 = 'Name is required.'
    var_20 = {var_18: var_19}
    var_21 = 'All test cases pass'
    var_22 = print(var_21)



# Parsed testcases at query #5
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Token(var_4)
    var_6 = 'object'
    var_7 = 'string'
    var_8 = module_1.Field()
    var_9 = 'integer'
    var_10 = module_1.Field()
    var_11 = {var_0: var_8, var_1: var_10}
    var_12 = module_1.Field()
    var_13 = module_2.validate_with_positions(token=var_5, validator=var_12)
    var_14 = {var_0: var_2}
    var_15 = module_0.Token(var_14)
    var_16 = module_1.Field()
    var_17 = True
    var_18 = module_1.Field()
    var_19 = {var_0: var_16, var_1: var_18}
    var_20 = module_1.Field()
    var_21 = module_2.validate_with_positions(token=var_15, validator=var_20)
    var_22 = 123
    var_23 = 'twenty'
    var_24 = {var_21: var_22, var_1: var_23}
    var_25 = module_0.Token(var_24)
    var_26 = module_1.Field()
    var_27 = module_1.Field()
    var_28 = {var_21: var_26, var_1: var_27}
    var_29 = module_1.Field()
    var_30 = module_2.validate_with_positions(token=var_25, validator=var_29)
    var_31 = 'person'
    var_32 = {var_30: var_2, var_1: var_3}
    var_33 = {var_31: var_32}
    var_34 = module_0.Token(var_33)
    var_35 = module_1.Field()
    var_36 = module_1.Field()
    var_37 = {var_30: var_35, var_1: var_36}
    var_38 = module_1.Field()
    var_39 = {var_31: var_38}
    var_40 = module_1.Field()
    var_41 = module_2.validate_with_positions(token=var_34, validator=var_40)
    var_42 = {var_30: var_2, var_1: var_23}
    var_43 = {var_31: var_42}
    var_44 = module_0.Token(var_43)
    var_45 = module_1.Field()
    var_46 = module_1.Field()
    var_47 = {var_30: var_45, var_1: var_46}
    var_48 = module_1.Field()
    var_49 = {var_31: var_48}
    var_50 = module_1.Field()
    var_51 = module_2.validate_with_positions(token=var_44, validator=var_50)
    var_52 = 'All test cases passed!'
    var_53 = print(var_52)



# Parsed testcases at query #6
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Token(var_4)
    var_6 = {var_0: var_2}
    var_7 = module_0.Token(var_6)
    var_8 = 'twenty-five'
    var_9 = {var_0: var_2, var_1: var_8}
    var_10 = module_0.Token(var_9)
    var_11 = 123
    var_12 = {var_0: var_11, var_1: var_8}
    var_13 = module_0.Token(var_12)
    var_14 = 'person'
    var_15 = {var_0: var_2, var_1: var_3}
    var_16 = {var_14: var_15}
    var_17 = module_0.Token(var_16)
    var_18 = {var_0: var_2, var_1: var_8}
    var_19 = {var_14: var_18}
    var_20 = module_0.Token(var_19)
    var_21 = {var_0: var_2, var_1: var_3}
    var_22 = 'Jane'
    var_23 = 30
    var_24 = {var_0: var_22, var_1: var_23}
    var_25 = [var_21, var_24]
    var_26 = module_0.Token(var_25)
    var_27 = 'array'
    var_28 = {var_0: var_2, var_1: var_3}
    var_29 = 'thirty'
    var_30 = {var_0: var_22, var_1: var_29}
    var_31 = [var_28, var_30]
    var_32 = module_0.Token(var_31)
    var_33 = 'people'
    var_34 = {var_0: var_2, var_1: var_3}
    var_35 = {var_0: var_22, var_1: var_23}
    var_36 = [var_34, var_35]
    var_37 = {var_33: var_36}
    var_38 = module_0.Token(var_37)
    var_39 = {var_0: var_2, var_1: var_3}
    var_40 = {var_0: var_22, var_1: var_29}
    var_41 = [var_39, var_40]
    var_42 = {var_33: var_41}
    var_43 = module_0.Token(var_42)
    var_44 = 'All test cases passed!'
    var_45 = print(var_44)



# Parsed testcases at query #7
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Token(var_4)
    var_6 = {var_0: var_2}
    var_7 = module_0.Token(var_6)
    var_8 = 'twenty-five'
    var_9 = {var_0: var_2, var_1: var_8}
    var_10 = module_0.Token(var_9)
    var_11 = 123
    var_12 = {var_0: var_11, var_1: var_8}
    var_13 = module_0.Token(var_12)
    var_14 = 'person'
    var_15 = {var_0: var_2, var_1: var_3}
    var_16 = {var_14: var_15}
    var_17 = module_0.Token(var_16)
    var_18 = {var_0: var_2, var_1: var_8}
    var_19 = {var_14: var_18}
    var_20 = module_0.Token(var_19)
    var_21 = 'All test cases passed!'
    var_22 = print(var_21)



# Parsed testcases at query #8
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'John Doe'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0.Token(var_2)
    var_5 = 2
    var_6 = var_1 * var_5
    var_7 = {var_0: var_6}
    var_8 = module_0.Token(var_7)
    var_9 = {}
    var_10 = module_0.Token(var_9)
    var_11 = {var_0: var_1}
    var_12 = module_0.Token(var_11)
    var_13 = 10
    var_14 = module_1.String(max_length=var_13)
    var_15 = module_2.validate_with_positions(token=var_12, validator=var_14)
    assert var_15 == 'John Doe'
    var_16 = var_1 * var_5
    var_17 = {var_0: var_16}
    var_18 = module_0.Token(var_17)
    var_19 = module_2.validate_with_positions(token=var_18, validator=var_14)
    var_20 = {}
    var_21 = module_0.Token(var_20)
    var_22 = module_2.validate_with_positions(token=var_21, validator=var_14)
    var_23 = {var_22: var_1}
    var_24 = module_0.Token(var_23)
    var_25 = module_1.String(max_length=var_13)
    var_26 = module_2.validate_with_positions(token=var_24, validator=var_25)
    assert var_26 == 'John Doe'
    var_27 = var_1 * var_5
    var_28 = {var_22: var_27}
    var_29 = module_0.Token(var_28)
    var_30 = module_2.validate_with_positions(token=var_29, validator=var_25)
    var_31 = {}
    var_32 = module_0.Token(var_31)
    var_33 = module_2.validate_with_positions(token=var_32, validator=var_25)
    var_34 = {var_33: var_1}
    var_35 = module_0.Token(var_34)
    var_36 = module_1.String(max_length=var_13)
    var_37 = module_2.validate_with_positions(token=var_35, validator=var_36)
    assert var_37 == 'John Doe'
    var_38 = var_1 * var_5
    var_39 = {var_33: var_38}
    var_40 = module_0.Token(var_39)
    var_41 = module_2.validate_with_positions(token=var_40, validator=var_36)
    var_42 = {}
    var_43 = module_0.Token(var_42)
    var_44 = module_2.validate_with_positions(token=var_43, validator=var_36)
    var_45 = {var_44: var_1}
    var_46 = module_0.Token(var_45)
    var_47 = module_1.String(max_length=var_13)
    var_48 = module_2.validate_with_positions(token=var_46, validator=var_47)
    assert var_48 == 'John Doe'
    var_49 = var_1 * var_5
    var_50 = {var_44: var_49}
    var_51 = module_0.Token(var_50)
    var_52 = module_2.validate_with_positions(token=var_51, validator=var_47)
    var_53 = {}
    var_54 = module_0.Token(var_53)
    var_55 = module_2.validate_with_positions(token=var_54, validator=var_47)
    var_56 = {var_55: var_1}
    var_57 = module_0.Token(var_56)
    var_58 = module_1.String(max_length=var_13)
    var_59 = module_2.validate_with_positions(token=var_57, validator=var_58)
    assert var_59 == 'John Doe'
    var_60 = var_1 * var_5
    var_61 = {var_55: var_60}
    var_62 = module_0.Token(var_61)
    var_63 = module_2.validate_with_positions(token=var_62, validator=var_58)
    var_64 = {}
    var_65 = module_0.Token(var_64)
    var_66 = module_2.validate_with_positions(token=var_65, validator=var_58)
    var_67 = {var_66: var_1}
    var_68 = module_0.Token(var_67)
    var_69 = module_1.String(max_length=var_13)
    var_70 = module_2.validate_with_positions(token=var_68, validator=var_69)
    assert var_70 == 'John Doe'
    var_71 = var_1 * var_5
    var_72 = {var_66: var_71}
    var_73 = module_0.Token(var_72)
    var_74 = module_2.validate_with_positions(token=var_73, validator=var_69)
    var_75 = {}
    var_76 = module_0.Token(var_75)
    var_77 = module_2.validate_with_positions(token=var_76, validator=var_69)
    var_78 = {var_77: var_1}
    var_79 = module_0.Token(var_78)
    var_80 = module_1.String(max_length=var_13)
    var_81 = module_2.validate_with_positions(token=var_79, validator=var_80)
    assert var_81 == 'John Doe'
    var_82 = var_1 * var_5
    var_83 = {var_77: var_82}
    var_84 = module_0.Token(var_83)



# Parsed testcases at query #9
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'John Doe'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0.Token(var_2)
    var_5 = 2
    var_6 = var_1 * var_5
    var_7 = {var_0: var_6}
    var_8 = module_0.Token(var_7)
    var_9 = {}
    var_10 = module_0.Token(var_9)
    var_11 = 'age'
    var_12 = 30
    var_13 = {var_0: var_1, var_11: var_12}
    var_14 = module_0.Token(var_13)
    var_15 = {var_0: var_1, var_11: var_12}
    var_16 = module_0.Token(var_15)
    var_17 = 10
    var_18 = module_1.String(max_length=var_17)
    var_19 = module_2.validate_with_positions(token=var_16, validator=var_18)
    var_20 = var_1 * var_5
    var_21 = {var_0: var_20, var_11: var_12}
    var_22 = module_0.Token(var_21)
    var_23 = module_1.String(max_length=var_17)
    var_24 = module_2.validate_with_positions(token=var_22, validator=var_23)
    var_25 = {}
    var_26 = module_0.Token(var_25)
    var_27 = module_1.String(max_length=var_17)
    var_28 = module_2.validate_with_positions(token=var_26, validator=var_27)
    var_29 = {var_24: var_1, var_11: var_12}
    var_30 = module_0.Token(var_29)
    var_31 = True
    var_32 = module_1.String(max_length=var_17)
    var_33 = module_2.validate_with_positions(token=var_30, validator=var_32)
    var_34 = {var_33: var_1, var_11: var_12}
    var_35 = module_0.Token(var_34)
    var_36 = False
    var_37 = module_1.String(max_length=var_17)
    var_38 = module_2.validate_with_positions(token=var_35, validator=var_37)
    var_39 = {var_33: var_1, var_11: var_12}
    var_40 = module_0.Token(var_39)
    var_41 = module_1.String(max_length=var_17)
    var_42 = module_2.validate_with_positions(token=var_40, validator=var_41)
    var_43 = {var_33: var_3, var_11: var_12}
    var_44 = module_0.Token(var_43)
    var_45 = module_1.String(max_length=var_17)
    var_46 = module_2.validate_with_positions(token=var_44, validator=var_45)
    var_47 = {var_33: var_3, var_11: var_12}
    var_48 = module_0.Token(var_47)
    var_49 = module_1.String(max_length=var_17)
    var_50 = module_2.validate_with_positions(token=var_48, validator=var_49)
    var_51 = {var_50: var_1, var_11: var_12}
    var_52 = module_0.Token(var_51)
    var_53 = module_1.String(max_length=var_17)
    var_54 = module_2.validate_with_positions(token=var_52, validator=var_53)
    var_55 = {var_50: var_1, var_11: var_12}
    var_56 = module_0.Token(var_55)
    var_57 = ''
    var_58 = module_1.String(max_length=var_17)
    var_59 = module_2.validate_with_positions(token=var_56, validator=var_58)
    var_60 = {var_11: var_12}
    var_61 = module_0.Token(var_60)
    var_62 = module_1.String(max_length=var_17)
    var_63 = module_2.validate_with_positions(token=var_61, validator=var_62)
    var_64 = {var_11: var_12}
    var_65 = module_0.Token(var_64)
    var_66 = module_1.String(max_length=var_17)
    var_67 = module_2.validate_with_positions(token=var_65, validator=var_66)
    var_68 = {var_11: var_12}
    var_69 = module_0.Token(var_68)
    var_70 = module_1.String(max_length=var_17)
    var_71 = module_2.validate_with_positions(token=var_69, validator=var_70)
    var_72 = {var_11: var_12}
    var_73 = module_0.Token(var_72)
    var_74 = module_1.String(max_length=var_17)
    var_75 = module_2.validate_with_positions(token=var_73, validator=var_74)
    var_76 = {var_11: var_12}
    var_77 = module_0.Token(var_76)
    var_78 = module_1.String(max_length=var_17)
    var_79 = module_2.validate_with_positions(token=var_77, validator=var_78)
    var_80 = {var_11: var_12}
    var_81 = module_0.Token(var_80)
    var_82 = module_1.String(max_length=var_17)
    var_83 = module_2.validate_with_positions(token=var_81, validator=var_82)
    var_84 = {var_11: var_12}
    var_85 = module_0.Token(var_84)
    var_86 = module_1.String(max_length=var_17)
    var_87 = module_2.validate_with_positions(token=var_85, validator=var_86)
    var_88 = {var_11: var_12}
    var_89 = module_0.Token(var_88)
    var_90 = module_1.String(max_length=var_17)
    var_91 = module_2.validate_with_positions(token=var_89, validator=var_90)



# Parsed testcases at query #10
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Token(var_4)
    var_6 = {var_0: var_2}
    var_7 = module_0.Token(var_6)
    var_8 = 'twenty-five'
    var_9 = {var_0: var_2, var_1: var_8}
    var_10 = module_0.Token(var_9)
    var_11 = 123
    var_12 = {var_0: var_11, var_1: var_8}
    var_13 = module_0.Token(var_12)
    var_14 = 'person'
    var_15 = {var_0: var_2, var_1: var_3}
    var_16 = {var_14: var_15}
    var_17 = module_0.Token(var_16)
    var_18 = {var_0: var_2, var_1: var_8}
    var_19 = {var_14: var_18}
    var_20 = module_0.Token(var_19)
    var_21 = {}
    var_22 = module_0.Token(var_21)
    var_23 = {var_0: var_2, var_1: var_3}
    var_24 = {var_14: var_23}
    var_25 = module_0.Token(var_24)
    var_26 = {var_0: var_2, var_1: var_8}
    var_27 = {var_14: var_26}
    var_28 = module_0.Token(var_27)
    var_29 = {var_0: var_2, var_1: var_3}
    var_30 = {var_14: var_29}
    var_31 = module_0.Token(var_30)
    var_32 = {var_0: var_2, var_1: var_8}
    var_33 = {var_14: var_32}
    var_34 = module_0.Token(var_33)
    var_35 = 'people'
    var_36 = {var_0: var_2, var_1: var_3}
    var_37 = 'Jane'
    var_38 = 30
    var_39 = {var_0: var_37, var_1: var_38}
    var_40 = [var_36, var_39]
    var_41 = {var_35: var_40}
    var_42 = module_0.Token(var_41)
    var_43 = {var_0: var_2, var_1: var_3}
    var_44 = 'thirty'
    var_45 = {var_0: var_37, var_1: var_44}
    var_46 = [var_43, var_45]
    var_47 = {var_35: var_46}
    var_48 = module_0.Token(var_47)
    var_49 = 'data'
    var_50 = {var_0: var_2, var_1: var_3}
    var_51 = {var_14: var_50}
    var_52 = {var_49: var_51}
    var_53 = module_0.Token(var_52)
    var_54 = {var_0: var_2, var_1: var_8}
    var_55 = {var_14: var_54}
    var_56 = {var_49: var_55}
    var_57 = module_0.Token(var_56)



# Parsed testcases at query #11
#--------------------------


import typesystem.fields as module_0
import typesystem.tokenize.tokens as module_1


def test_case_0():
    var_0 = 5
    var_1 = module_0.String(max_length=var_0)
    var_2 = 'hello world'
    var_3 = None
    var_4 = module_1.Token(var_2)
    var_5 = module_2.validate_with_positions(token=var_4, validator=var_1)
    var_6 = True
    var_7 = module_0.String()
    var_8 = module_1.Token(var_3)
    var_9 = module_2.validate_with_positions(token=var_8, validator=var_7)
    var_10 = 'name'
    var_11 = 'age'
    var_12 = -1
    var_13 = {var_10: var_2, var_11: var_12}
    var_14 = module_1.Token(var_13)



# Parsed testcases at query #12
#--------------------------


import typesystem.tokenize.tokens as module_0


def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Token(var_4)
    var_6 = {var_0: var_2}
    var_7 = module_0.Token(var_6)
    var_8 = error.messages()[0]
    var_9 = 'twenty'
    var_10 = {var_0: var_2, var_1: var_9}
    var_11 = module_0.Token(var_10)
    var_12 = error.messages()[0]
    var_13 = 123
    var_14 = {var_0: var_13, var_1: var_9}
    var_15 = module_0.Token(var_14)
    var_16 = sorted(error.messages(), key=lambda m: m.start_position.char_index)
    var_17 = var_16[0]
    var_18 = var_16[1]
    var_19 = 'person'
    var_20 = {var_0: var_2, var_1: var_3}
    var_21 = {var_19: var_20}
    var_22 = module_0.Token(var_21)
    var_23 = {var_0: var_2, var_1: var_9}
    var_24 = {var_19: var_23}
    var_25 = module_0.Token(var_24)
    var_26 = error.messages()[0]
    var_27 = 1
    var_28 = 2
    var_29 = 3
    var_30 = [var_27, var_28, var_29]
    var_31 = module_0.Token(var_30)
    var_32 = 'array'
    var_33 = 'two'
    var_34 = [var_27, var_33, var_29]
    var_35 = module_0.Token(var_34)
    var_36 = error.messages()[0]
    var_37 = 'numbers'
    var_38 = [var_27, var_28, var_29]
    var_39 = {var_37: var_38}
    var_40 = module_0.Token(var_39)
    var_41 = [var_27, var_33, var_29]
    var_42 = {var_37: var_41}
    var_43 = module_0.Token(var_42)
    var_44 = error.messages()[0]



# Parsed testcases at query #13
#--------------------------


import typesystem.fields as module_1


def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0.Token(var_2)
    var_5 = {}
    var_6 = module_0.Token(var_5)
    var_7 = 'John Doe Smith'
    var_8 = {var_0: var_7}
    var_9 = module_0.Token(var_8)
    var_10 = 'nested'
    var_11 = 'value'
    var_12 = 'Too Long'
    var_13 = {var_11: var_12}
    var_14 = {var_10: var_13}
    var_15 = module_0.Token(var_14)
    var_16 = 5
    var_17 = module_1.String(max_length=var_16)
    var_18 = module_0.Token(var_12)
    var_19 = module_2.validate_with_positions(token=var_18, validator=var_17)
    var_20 = 'All tests passed!'
    var_21 = print(var_20)



# Parsed testcases at query #14
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = module_0.Token(var_4)
    var_7 = {var_0: var_2}
    var_8 = module_0.Token(var_7)
    var_9 = 'thirty'
    var_10 = {var_0: var_2, var_1: var_9}
    var_11 = module_0.Token(var_10)
    var_12 = 'All tests passed.'
    var_13 = print(var_12)



# Parsed testcases at query #15
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Token(var_4)
    var_6 = 'object'
    var_7 = 'string'
    var_8 = module_1.Field()
    var_9 = 'integer'
    var_10 = module_1.Field()
    var_11 = {var_0: var_8, var_1: var_10}
    var_12 = module_1.Field()
    var_13 = module_2.validate_with_positions(token=var_5, validator=var_12)
    var_14 = {var_0: var_2}
    var_15 = module_0.Token(var_14)
    var_16 = module_1.Field()
    var_17 = module_1.Field()
    var_18 = {var_0: var_16, var_1: var_17}
    var_19 = module_1.Field()
    var_20 = module_2.validate_with_positions(token=var_15, validator=var_19)
    var_21 = 'twenty-five'
    var_22 = {var_20: var_2, var_1: var_21}
    var_23 = module_0.Token(var_22)
    var_24 = module_1.Field()
    var_25 = module_1.Field()
    var_26 = {var_20: var_24, var_1: var_25}
    var_27 = module_1.Field()
    var_28 = module_2.validate_with_positions(token=var_23, validator=var_27)
    var_29 = 123
    var_30 = {var_28: var_29, var_1: var_21}
    var_31 = module_0.Token(var_30)
    var_32 = module_1.Field()
    var_33 = module_1.Field()
    var_34 = {var_28: var_32, var_1: var_33}
    var_35 = module_1.Field()
    var_36 = module_2.validate_with_positions(token=var_31, validator=var_35)
    var_37 = 'person'
    var_38 = {var_36: var_2, var_1: var_3}
    var_39 = {var_37: var_38}
    var_40 = module_0.Token(var_39)
    var_41 = module_1.Field()
    var_42 = module_1.Field()
    var_43 = {var_36: var_41, var_1: var_42}
    var_44 = module_1.Field()
    var_45 = {var_37: var_44}
    var_46 = module_1.Field()
    var_47 = module_2.validate_with_positions(token=var_40, validator=var_46)
    var_48 = {var_36: var_2, var_1: var_21}
    var_49 = {var_37: var_48}
    var_50 = module_0.Token(var_49)
    var_51 = module_1.Field()
    var_52 = module_1.Field()
    var_53 = {var_36: var_51, var_1: var_52}
    var_54 = module_1.Field()
    var_55 = {var_37: var_54}
    var_56 = module_1.Field()
    var_57 = module_2.validate_with_positions(token=var_50, validator=var_56)
    var_58 = 'numbers'
    var_59 = 1
    var_60 = 2
    var_61 = 3
    var_62 = [var_59, var_60, var_61]
    var_63 = {var_58: var_62}
    var_64 = module_0.Token(var_63)
    var_65 = 'array'
    var_66 = module_1.Field()
    var_67 = module_1.Field()
    var_68 = {var_58: var_67}
    var_69 = module_1.Field()
    var_70 = module_2.validate_with_positions(token=var_64, validator=var_69)
    var_71 = 'two'
    var_72 = [var_59, var_71, var_61]
    var_73 = {var_58: var_72}
    var_74 = module_0.Token(var_73)
    var_75 = module_1.Field()
    var_76 = module_1.Field()
    var_77 = {var_58: var_76}
    var_78 = module_1.Field()
    var_79 = module_2.validate_with_positions(token=var_74, validator=var_78)
    var_80 = 'matrix'
    var_81 = [var_59, var_60]
    var_82 = 4
    var_83 = [var_61, var_82]
    var_84 = [var_81, var_83]
    var_85 = {var_80: var_84}
    var_86 = module_0.Token(var_85)
    var_87 = module_1.Field()
    var_88 = module_1.Field()
    var_89 = module_1.Field()
    var_90 = {var_80: var_89}
    var_91 = module_1.Field()
    var_92 = module_2.validate_with_positions(token=var_86, validator=var_91)
    var_93 = [var_59, var_60]
    var_94 = 'four'
    var_95 = [var_61, var_94]
    var_96 = [var_93, var_95]
    var_97 = {var_80: var_96}
    var_98 = module_0.Token(var_97)
    var_99 = module_1.Field()
    var_100 = module_1.Field()
    var_101 = module_1.Field()
    var_102 = {var_80: var_101}
    var_103 = module_1.Field()
    var_104 = module_2.validate_with_positions(token=var_98, validator=var_103)
    var_105 = 'All test cases pass'
    var_106 = print(var_105)



# Parsed testcases at query #16
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0.Token(var_2)
    var_5 = {}
    var_6 = module_0.Token(var_5)
    var_7 = 'John Doe Smith'
    var_8 = {var_0: var_7}
    var_9 = module_0.Token(var_8)
    var_10 = module_1.String()
    var_11 = 'nested'
    var_12 = 'age'
    var_13 = 25
    var_14 = {var_12: var_13}
    var_15 = {var_11: var_14}
    var_16 = module_0.Token(var_15)
    var_17 = 5
    var_18 = module_1.String(max_length=var_17)
    var_19 = 'Hello World'
    var_20 = module_0.Token(var_19)
    var_21 = module_2.validate_with_positions(token=var_20, validator=var_18)
    var_22 = 'All tests passed!'
    var_23 = print(var_22)



# Parsed testcases at query #17
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Token(var_4)
    var_6 = {var_0: var_2}
    var_7 = module_0.Token(var_6)
    var_8 = 'twenty-five'
    var_9 = {var_0: var_2, var_1: var_8}
    var_10 = module_0.Token(var_9)
    var_11 = 123
    var_12 = {var_0: var_11, var_1: var_8}
    var_13 = module_0.Token(var_12)
    var_14 = 'person'
    var_15 = {var_0: var_2, var_1: var_3}
    var_16 = {var_14: var_15}
    var_17 = module_0.Token(var_16)
    var_18 = {var_0: var_2, var_1: var_8}
    var_19 = {var_14: var_18}
    var_20 = module_0.Token(var_19)
    var_21 = {var_0: var_2}
    var_22 = {var_14: var_21}
    var_23 = module_0.Token(var_22)
    var_24 = {var_0: var_11, var_1: var_8}
    var_25 = {var_14: var_24}
    var_26 = module_0.Token(var_25)
    var_27 = {}
    var_28 = module_0.Token(var_27)
    var_29 = 'city'
    var_30 = 'New York'
    var_31 = {var_0: var_2, var_1: var_3, var_29: var_30}
    var_32 = module_0.Token(var_31)
    var_33 = 'All test cases passed!'
    var_34 = print(var_33)



####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0.Token(var_2)
    var_5 = {}
    var_6 = module_0.Token(var_5)
    var_7 = 'John Doe Smith'
    var_8 = {var_0: var_7}
    var_9 = module_0.Token(var_8)
    var_10 = module_1.String()
    var_11 = 'nested'
    var_12 = 'age'
    var_13 = 25
    var_14 = {var_12: var_13}
    var_15 = {var_11: var_14}
    var_16 = module_0.Token(var_15)
    var_17 = ''
    var_18 = 'twenty'
    var_19 = {var_0: var_17, var_12: var_18}
    var_20 = module_0.Token(var_19)
    var_21 = 'All tests passed!'
    var_22 = print(var_21)



# Parsed testcases at query #2
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Token(var_4)
    var_6 = 'object'
    var_7 = 'string'
    var_8 = module_1.Field()
    var_9 = 'integer'
    var_10 = module_1.Field()
    var_11 = {var_0: var_8, var_1: var_10}
    var_12 = module_1.Field()
    var_13 = module_2.validate_with_positions(token=var_5, validator=var_12)
    var_14 = {var_0: var_2}
    var_15 = module_0.Token(var_14)
    var_16 = module_1.Field()
    var_17 = True
    var_18 = module_1.Field()
    var_19 = {var_0: var_16, var_1: var_18}
    var_20 = module_1.Field()
    var_21 = module_2.validate_with_positions(token=var_15, validator=var_20)
    var_22 = 123
    var_23 = 'twenty'
    var_24 = {var_21: var_22, var_1: var_23}
    var_25 = module_0.Token(var_24)
    var_26 = module_1.Field()
    var_27 = module_1.Field()
    var_28 = {var_21: var_26, var_1: var_27}
    var_29 = module_1.Field()
    var_30 = module_2.validate_with_positions(token=var_25, validator=var_29)
    var_31 = 'address'
    var_32 = 'street'
    var_33 = 'city'
    var_34 = '123 Main St'
    var_35 = 'New York'
    var_36 = {var_32: var_34, var_33: var_35}
    var_37 = {var_30: var_2, var_31: var_36}
    var_38 = module_0.Token(var_37)
    var_39 = module_1.Field()
    var_40 = module_1.Field()
    var_41 = module_1.Field()
    var_42 = {var_32: var_40, var_33: var_41}
    var_43 = module_1.Field()
    var_44 = {var_30: var_39, var_31: var_43}
    var_45 = module_1.Field()
    var_46 = module_2.validate_with_positions(token=var_38, validator=var_45)
    var_47 = {var_32: var_22, var_33: var_35}
    var_48 = {var_30: var_2, var_31: var_47}
    var_49 = module_0.Token(var_48)
    var_50 = module_1.Field()
    var_51 = module_1.Field()
    var_52 = module_1.Field()
    var_53 = {var_32: var_51, var_33: var_52}
    var_54 = module_1.Field()
    var_55 = {var_30: var_50, var_31: var_54}
    var_56 = module_1.Field()
    var_57 = module_2.validate_with_positions(token=var_49, validator=var_56)
    var_58 = 'All test cases pass'
    var_59 = print(var_58)



# Parsed testcases at query #3
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Token(var_4)
    var_6 = {var_0: var_2}
    var_7 = module_0.Token(var_6)
    var_8 = 'twenty-five'
    var_9 = {var_0: var_2, var_1: var_8}
    var_10 = module_0.Token(var_9)
    var_11 = 123
    var_12 = {var_0: var_11, var_1: var_8}
    var_13 = module_0.Token(var_12)
    var_14 = 'person'
    var_15 = {var_0: var_2, var_1: var_3}
    var_16 = {var_14: var_15}
    var_17 = module_0.Token(var_16)
    var_18 = {var_0: var_2, var_1: var_8}
    var_19 = {var_14: var_18}
    var_20 = module_0.Token(var_19)
    var_21 = 'All test cases pass'
    var_22 = print(var_21)



# Parsed testcases at query #4
#--------------------------


import typesystem.schemas as module_3


def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Token(var_4)
    var_6 = 'object'
    var_7 = 'string'
    var_8 = module_1.Field()
    var_9 = 'integer'
    var_10 = module_1.Field()
    var_11 = {var_0: var_8, var_1: var_10}
    var_12 = module_1.Field()
    var_13 = module_2.validate_with_positions(token=var_5, validator=var_12)
    var_14 = {var_0: var_2}
    var_15 = module_0.Token(var_14)
    var_16 = module_1.Field()
    var_17 = module_1.Field()
    var_18 = {var_0: var_16, var_1: var_17}
    var_19 = module_1.Field()
    var_20 = module_2.validate_with_positions(token=var_15, validator=var_19)
    var_21 = 'twenty-five'
    var_22 = {var_20: var_2, var_1: var_21}
    var_23 = module_0.Token(var_22)
    var_24 = module_1.Field()
    var_25 = module_1.Field()
    var_26 = {var_20: var_24, var_1: var_25}
    var_27 = module_1.Field()
    var_28 = module_2.validate_with_positions(token=var_23, validator=var_27)
    var_29 = 123
    var_30 = {var_28: var_29, var_1: var_21}
    var_31 = module_0.Token(var_30)
    var_32 = module_1.Field()
    var_33 = module_1.Field()
    var_34 = {var_28: var_32, var_1: var_33}
    var_35 = module_1.Field()
    var_36 = module_2.validate_with_positions(token=var_31, validator=var_35)
    var_37 = 'person'
    var_38 = {var_36: var_2, var_1: var_3}
    var_39 = {var_37: var_38}
    var_40 = module_0.Token(var_39)
    var_41 = module_1.Field()
    var_42 = module_1.Field()
    var_43 = {var_36: var_41, var_1: var_42}
    var_44 = module_1.Field()
    var_45 = {var_37: var_44}
    var_46 = module_1.Field()
    var_47 = module_2.validate_with_positions(token=var_40, validator=var_46)
    var_48 = {var_36: var_2}
    var_49 = {var_37: var_48}
    var_50 = module_0.Token(var_49)
    var_51 = module_1.Field()
    var_52 = module_1.Field()
    var_53 = {var_36: var_51, var_1: var_52}
    var_54 = module_1.Field()
    var_55 = {var_37: var_54}
    var_56 = module_1.Field()
    var_57 = module_2.validate_with_positions(token=var_50, validator=var_56)
    var_58 = {var_57: var_2, var_1: var_21}
    var_59 = {var_37: var_58}
    var_60 = module_0.Token(var_59)
    var_61 = module_1.Field()
    var_62 = module_1.Field()
    var_63 = {var_57: var_61, var_1: var_62}
    var_64 = module_1.Field()
    var_65 = {var_37: var_64}
    var_66 = module_1.Field()
    var_67 = module_2.validate_with_positions(token=var_60, validator=var_66)
    var_68 = {var_67: var_29, var_1: var_21}
    var_69 = {var_37: var_68}
    var_70 = module_0.Token(var_69)
    var_71 = module_1.Field()
    var_72 = module_1.Field()
    var_73 = {var_67: var_71, var_1: var_72}
    var_74 = module_1.Field()
    var_75 = {var_37: var_74}
    var_76 = module_1.Field()
    var_77 = module_2.validate_with_positions(token=var_70, validator=var_76)
    var_78 = 'numbers'
    var_79 = 1
    var_80 = 2
    var_81 = 3
    var_82 = [var_79, var_80, var_81]
    var_83 = {var_78: var_82}
    var_84 = module_0.Token(var_83)
    var_85 = 'array'
    var_86 = module_1.Field()
    var_87 = module_1.Field()
    var_88 = {var_78: var_87}
    var_89 = module_1.Field()
    var_90 = module_2.validate_with_positions(token=var_84, validator=var_89)
    var_91 = 'two'
    var_92 = [var_79, var_91, var_81]
    var_93 = {var_78: var_92}
    var_94 = module_0.Token(var_93)
    var_95 = module_1.Field()
    var_96 = module_1.Field()
    var_97 = {var_78: var_96}
    var_98 = module_1.Field()
    var_99 = module_2.validate_with_positions(token=var_94, validator=var_98)
    var_100 = 'matrix'
    var_101 = [var_79, var_80]
    var_102 = 4
    var_103 = [var_81, var_102]
    var_104 = [var_101, var_103]
    var_105 = {var_100: var_104}
    var_106 = module_0.Token(var_105)
    var_107 = module_1.Field()
    var_108 = module_1.Field()
    var_109 = module_1.Field()
    var_110 = {var_100: var_109}
    var_111 = module_1.Field()
    var_112 = module_2.validate_with_positions(token=var_106, validator=var_111)
    var_113 = [var_79, var_80]
    var_114 = 'three'
    var_115 = [var_114, var_102]
    var_116 = [var_113, var_115]
    var_117 = {var_100: var_116}
    var_118 = module_0.Token(var_117)
    var_119 = module_1.Field()
    var_120 = module_1.Field()
    var_121 = module_1.Field()
    var_122 = {var_100: var_121}
    var_123 = module_1.Field()
    var_124 = module_2.validate_with_positions(token=var_118, validator=var_123)
    var_125 = {var_124: var_2, var_1: var_3}
    var_126 = module_0.Token(var_125)
    var_127 = module_1.Field()
    var_128 = module_1.Field()
    var_129 = {var_124: var_127, var_1: var_128}
    var_130 = module_3.Schema(var_129)
    var_131 = module_2.validate_with_positions(token=var_126, validator=var_130)
    var_132 = {var_124: var_2}
    var_133 = module_0.Token(var_132)
    var_134 = module_1.Field()
    var_135 = module_1.Field()
    var_136 = {var_124: var_134, var_1: var_135}
    var_137 = module_3.Schema(var_136)
    var_138 = module_2.validate_with_positions(token=var_133, validator=var_137)



# Parsed testcases at query #5
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Token(var_4)
    var_6 = {var_0: var_2}
    var_7 = module_0.Token(var_6)
    var_8 = 'twenty-five'
    var_9 = {var_0: var_2, var_1: var_8}
    var_10 = module_0.Token(var_9)
    var_11 = 123
    var_12 = {var_0: var_11, var_1: var_8}
    var_13 = module_0.Token(var_12)
    var_14 = 'person'
    var_15 = {var_0: var_2, var_1: var_3}
    var_16 = {var_14: var_15}
    var_17 = module_0.Token(var_16)
    var_18 = {var_0: var_2}
    var_19 = {var_14: var_18}
    var_20 = module_0.Token(var_19)
    var_21 = {var_0: var_2, var_1: var_8}
    var_22 = {var_14: var_21}
    var_23 = module_0.Token(var_22)
    var_24 = 'numbers'
    var_25 = 1
    var_26 = 2
    var_27 = 3
    var_28 = [var_25, var_26, var_27]
    var_29 = {var_24: var_28}
    var_30 = module_0.Token(var_29)
    var_31 = 'two'
    var_32 = [var_25, var_31, var_27]
    var_33 = {var_24: var_32}
    var_34 = module_0.Token(var_33)
    var_35 = 'matrix'
    var_36 = [var_25, var_26]
    var_37 = 4
    var_38 = [var_27, var_37]
    var_39 = [var_36, var_38]
    var_40 = {var_35: var_39}
    var_41 = module_0.Token(var_40)
    var_42 = [var_25, var_26]
    var_43 = 'four'
    var_44 = [var_27, var_43]
    var_45 = [var_42, var_44]
    var_46 = {var_35: var_45}
    var_47 = module_0.Token(var_46)
    var_48 = {var_0: var_2}
    var_49 = module_0.Token(var_48)
    var_50 = False
    var_51 = {var_0: var_2}
    var_52 = module_0.Token(var_51)
    var_53 = None
    var_54 = {var_0: var_2, var_1: var_53}
    var_55 = module_0.Token(var_54)
    var_56 = True
    var_57 = {var_0: var_2, var_1: var_8}
    var_58 = module_0.Token(var_57)
    var_59 = True
    var_60 = 'number'
    var_61 = {var_60: var_37}
    var_62 = module_0.Token(var_61)
    var_63 = {var_60: var_27}
    var_64 = module_0.Token(var_63)



# Parsed testcases at query #6
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'John'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0.Token(var_2)
    var_5 = {}
    var_6 = module_0.Token(var_5)
    var_7 = 'John Doe Smith'
    var_8 = {var_0: var_7}
    var_9 = module_0.Token(var_8)
    var_10 = module_1.String()
    var_11 = 'nested'
    var_12 = 'age'
    var_13 = '25'
    var_14 = {var_12: var_13}
    var_15 = {var_11: var_14}
    var_16 = module_0.Token(var_15)
    var_17 = {}
    var_18 = {var_11: var_17}
    var_19 = module_0.Token(var_18)
    var_20 = 5
    var_21 = module_1.String(max_length=var_20)
    var_22 = 'Hello'
    var_23 = module_0.Token(var_22)
    var_24 = module_2.validate_with_positions(token=var_23, validator=var_21)
    assert var_24 == 'Hello'
    var_25 = 'Hello World'
    var_26 = module_0.Token(var_25)
    var_27 = module_2.validate_with_positions(token=var_26, validator=var_21)
    var_28 = 'All tests passed!'
    var_29 = print(var_28)



# Parsed testcases at query #7
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Token(var_4)
    var_6 = {var_0: var_2}
    var_7 = module_0.Token(var_6)
    var_8 = 'twenty'
    var_9 = {var_0: var_2, var_1: var_8}
    var_10 = module_0.Token(var_9)
    var_11 = 123
    var_12 = {var_0: var_11, var_1: var_8}
    var_13 = module_0.Token(var_12)
    var_14 = 'person'
    var_15 = {var_0: var_2, var_1: var_3}
    var_16 = {var_14: var_15}
    var_17 = module_0.Token(var_16)
    var_18 = {var_0: var_2}
    var_19 = {var_14: var_18}
    var_20 = module_0.Token(var_19)
    var_21 = {var_0: var_2, var_1: var_8}
    var_22 = {var_14: var_21}
    var_23 = module_0.Token(var_22)
    var_24 = 'numbers'
    var_25 = 1
    var_26 = 2
    var_27 = 3
    var_28 = [var_25, var_26, var_27]
    var_29 = {var_24: var_28}
    var_30 = module_0.Token(var_29)
    var_31 = 'two'
    var_32 = [var_25, var_31, var_27]
    var_33 = {var_24: var_32}
    var_34 = module_0.Token(var_33)
    var_35 = 'matrix'
    var_36 = [var_25, var_26]
    var_37 = 4
    var_38 = [var_27, var_37]
    var_39 = [var_36, var_38]
    var_40 = {var_35: var_39}
    var_41 = module_0.Token(var_40)
    var_42 = [var_25, var_26]
    var_43 = 'four'
    var_44 = [var_27, var_43]
    var_45 = [var_42, var_44]
    var_46 = {var_35: var_45}
    var_47 = module_0.Token(var_46)
    var_48 = 'All test cases passed!'
    var_49 = print(var_48)



# Parsed testcases at query #8
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Token(var_4)
    var_6 = 'object'
    var_7 = 'string'
    var_8 = module_1.Field()
    var_9 = 'integer'
    var_10 = module_1.Field()
    var_11 = {var_0: var_8, var_1: var_10}
    var_12 = module_1.Field()
    var_13 = module_2.validate_with_positions(token=var_5, validator=var_12)
    var_14 = {var_0: var_2}
    var_15 = module_0.Token(var_14)
    var_16 = module_1.Field()
    var_17 = True
    var_18 = module_1.Field()
    var_19 = {var_0: var_16, var_1: var_18}
    var_20 = module_1.Field()
    var_21 = module_2.validate_with_positions(token=var_15, validator=var_20)
    var_22 = 123
    var_23 = 'twenty'
    var_24 = {var_21: var_22, var_1: var_23}
    var_25 = module_0.Token(var_24)
    var_26 = module_1.Field()
    var_27 = module_1.Field()
    var_28 = {var_21: var_26, var_1: var_27}
    var_29 = module_1.Field()
    var_30 = module_2.validate_with_positions(token=var_25, validator=var_29)
    var_31 = 'person'
    var_32 = {var_30: var_2, var_1: var_3}
    var_33 = {var_31: var_32}
    var_34 = module_0.Token(var_33)
    var_35 = module_1.Field()
    var_36 = module_1.Field()
    var_37 = {var_30: var_35, var_1: var_36}
    var_38 = module_1.Field()
    var_39 = {var_31: var_38}
    var_40 = module_1.Field()
    var_41 = module_2.validate_with_positions(token=var_34, validator=var_40)
    var_42 = {var_30: var_2, var_1: var_23}
    var_43 = {var_31: var_42}
    var_44 = module_0.Token(var_43)
    var_45 = module_1.Field()
    var_46 = module_1.Field()
    var_47 = {var_30: var_45, var_1: var_46}
    var_48 = module_1.Field()
    var_49 = {var_31: var_48}
    var_50 = module_1.Field()
    var_51 = module_2.validate_with_positions(token=var_44, validator=var_50)
    var_52 = 'All test cases passed!'
    var_53 = print(var_52)



# Parsed testcases at query #9
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Token(var_4)
    var_6 = {var_0: var_2}
    var_7 = module_0.Token(var_6)
    var_8 = 'twenty'
    var_9 = {var_0: var_2, var_1: var_8}
    var_10 = module_0.Token(var_9)
    var_11 = 123
    var_12 = {var_0: var_11, var_1: var_8}
    var_13 = module_0.Token(var_12)
    var_14 = 'person'
    var_15 = {var_0: var_2, var_1: var_3}
    var_16 = {var_14: var_15}
    var_17 = module_0.Token(var_16)
    var_18 = {var_0: var_2}
    var_19 = {var_14: var_18}
    var_20 = module_0.Token(var_19)
    var_21 = {}
    var_22 = module_0.Token(var_21)
    var_23 = {}
    var_24 = {var_14: var_23}
    var_25 = module_0.Token(var_24)
    var_26 = {var_0: var_2, var_1: var_8}
    var_27 = {var_14: var_26}
    var_28 = module_0.Token(var_27)
    var_29 = {var_0: var_11, var_1: var_8}
    var_30 = {var_14: var_29}
    var_31 = module_0.Token(var_30)



# Parsed testcases at query #10
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Token(var_4)
    var_6 = 'object'
    var_7 = 'string'
    var_8 = module_1.Field()
    var_9 = 'integer'
    var_10 = module_1.Field()
    var_11 = {var_0: var_8, var_1: var_10}
    var_12 = module_1.Field()
    var_13 = module_2.validate_with_positions(token=var_5, validator=var_12)
    var_14 = {var_0: var_2}
    var_15 = module_0.Token(var_14)
    var_16 = module_1.Field()
    var_17 = True
    var_18 = module_1.Field()
    var_19 = {var_0: var_16, var_1: var_18}
    var_20 = module_1.Field()
    var_21 = module_2.validate_with_positions(token=var_15, validator=var_20)
    var_22 = 'twenty-five'
    var_23 = {var_21: var_2, var_1: var_22}
    var_24 = module_0.Token(var_23)
    var_25 = module_1.Field()
    var_26 = module_1.Field()
    var_27 = {var_21: var_25, var_1: var_26}
    var_28 = module_1.Field()
    var_29 = module_2.validate_with_positions(token=var_24, validator=var_28)
    var_30 = 123
    var_31 = {var_29: var_30, var_1: var_22}
    var_32 = module_0.Token(var_31)
    var_33 = module_1.Field()
    var_34 = module_1.Field()
    var_35 = {var_29: var_33, var_1: var_34}
    var_36 = module_1.Field()
    var_37 = module_2.validate_with_positions(token=var_32, validator=var_36)
    var_38 = 'person'
    var_39 = {var_37: var_2, var_1: var_3}
    var_40 = {var_38: var_39}
    var_41 = module_0.Token(var_40)
    var_42 = module_1.Field()
    var_43 = module_1.Field()
    var_44 = {var_37: var_42, var_1: var_43}
    var_45 = module_1.Field()
    var_46 = {var_38: var_45}
    var_47 = module_1.Field()
    var_48 = module_2.validate_with_positions(token=var_41, validator=var_47)
    var_49 = {var_37: var_2}
    var_50 = {var_38: var_49}
    var_51 = module_0.Token(var_50)
    var_52 = module_1.Field()
    var_53 = module_1.Field()
    var_54 = {var_37: var_52, var_1: var_53}
    var_55 = module_1.Field()
    var_56 = {var_38: var_55}
    var_57 = module_1.Field()
    var_58 = module_2.validate_with_positions(token=var_51, validator=var_57)
    var_59 = {var_58: var_2, var_1: var_22}
    var_60 = {var_38: var_59}
    var_61 = module_0.Token(var_60)
    var_62 = module_1.Field()
    var_63 = module_1.Field()
    var_64 = {var_58: var_62, var_1: var_63}
    var_65 = module_1.Field()
    var_66 = {var_38: var_65}
    var_67 = module_1.Field()
    var_68 = module_2.validate_with_positions(token=var_61, validator=var_67)
    var_69 = {var_68: var_30, var_1: var_22}
    var_70 = {var_38: var_69}
    var_71 = module_0.Token(var_70)
    var_72 = module_1.Field()
    var_73 = module_1.Field()
    var_74 = {var_68: var_72, var_1: var_73}
    var_75 = module_1.Field()
    var_76 = {var_38: var_75}
    var_77 = module_1.Field()
    var_78 = module_2.validate_with_positions(token=var_71, validator=var_77)
    var_79 = 'numbers'
    var_80 = 2
    var_81 = 3
    var_82 = [var_17, var_80, var_81]
    var_83 = {var_79: var_82}
    var_84 = module_0.Token(var_83)
    var_85 = 'array'
    var_86 = module_1.Field()
    var_87 = module_1.Field()
    var_88 = {var_79: var_87}
    var_89 = module_1.Field()
    var_90 = module_2.validate_with_positions(token=var_84, validator=var_89)
    var_91 = 'two'
    var_92 = [var_17, var_91, var_81]
    var_93 = {var_79: var_92}
    var_94 = module_0.Token(var_93)
    var_95 = module_1.Field()
    var_96 = module_1.Field()
    var_97 = {var_79: var_96}
    var_98 = module_1.Field()
    var_99 = module_2.validate_with_positions(token=var_94, validator=var_98)
    var_100 = 'matrix'
    var_101 = [var_17, var_80]
    var_102 = 4
    var_103 = [var_81, var_102]
    var_104 = [var_101, var_103]
    var_105 = {var_100: var_104}
    var_106 = module_0.Token(var_105)
    var_107 = module_1.Field()
    var_108 = module_1.Field()
    var_109 = module_1.Field()
    var_110 = {var_100: var_109}
    var_111 = module_1.Field()
    var_112 = module_2.validate_with_positions(token=var_106, validator=var_111)
    var_113 = [var_17, var_80]
    var_114 = 'three'
    var_115 = [var_114, var_102]
    var_116 = [var_113, var_115]
    var_117 = {var_100: var_116}
    var_118 = module_0.Token(var_117)
    var_119 = module_1.Field()
    var_120 = module_1.Field()
    var_121 = module_1.Field()
    var_122 = {var_100: var_121}
    var_123 = module_1.Field()
    var_124 = module_2.validate_with_positions(token=var_118, validator=var_123)
    var_125 = 'value'
    var_126 = 'hello'
    var_127 = {var_125: var_126}
    var_128 = module_0.Token(var_127)
    var_129 = 'union'
    var_130 = module_1.Field()
    var_131 = module_1.Field()
    var_132 = [var_130, var_131]
    var_133 = module_1.Field()
    var_134 = {var_125: var_133}
    var_135 = module_1.Field()
    var_136 = module_2.validate_with_positions(token=var_128, validator=var_135)
    var_137 = 3.14
    var_138 = {var_125: var_137}
    var_139 = module_0.Token(var_138)
    var_140 = module_1.Field()
    var_141 = module_1.Field()
    var_142 = [var_140, var_141]
    var_143 = module_1.Field()
    var_144 = {var_125: var_143}
    var_145 = module_1.Field()
    var_146 = module_2.validate_with_positions(token=var_139, validator=var_145)



# Parsed testcases at query #11
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Token(var_4)
    var_6 = 'object'
    var_7 = 'string'
    var_8 = module_1.Field()
    var_9 = 'integer'
    var_10 = module_1.Field()
    var_11 = {var_0: var_8, var_1: var_10}
    var_12 = module_1.Field()
    var_13 = module_2.validate_with_positions(token=var_5, validator=var_12)
    var_14 = {var_0: var_2}
    var_15 = module_0.Token(var_14)
    var_16 = module_1.Field()
    var_17 = True
    var_18 = module_1.Field()
    var_19 = {var_0: var_16, var_1: var_18}
    var_20 = module_1.Field()
    var_21 = module_2.validate_with_positions(token=var_15, validator=var_20)
    var_22 = 123
    var_23 = 'twenty'
    var_24 = {var_21: var_22, var_1: var_23}
    var_25 = module_0.Token(var_24)
    var_26 = module_1.Field()
    var_27 = module_1.Field()
    var_28 = {var_21: var_26, var_1: var_27}
    var_29 = module_1.Field()
    var_30 = module_2.validate_with_positions(token=var_25, validator=var_29)
    var_31 = 'person'
    var_32 = {var_30: var_2, var_1: var_3}
    var_33 = {var_31: var_32}
    var_34 = module_0.Token(var_33)
    var_35 = module_1.Field()
    var_36 = module_1.Field()
    var_37 = {var_30: var_35, var_1: var_36}
    var_38 = module_1.Field()
    var_39 = {var_31: var_38}
    var_40 = module_1.Field()
    var_41 = module_2.validate_with_positions(token=var_34, validator=var_40)
    var_42 = {var_30: var_2, var_1: var_23}
    var_43 = {var_31: var_42}
    var_44 = module_0.Token(var_43)
    var_45 = module_1.Field()
    var_46 = module_1.Field()
    var_47 = {var_30: var_45, var_1: var_46}
    var_48 = module_1.Field()
    var_49 = {var_31: var_48}
    var_50 = module_1.Field()
    var_51 = module_2.validate_with_positions(token=var_44, validator=var_50)
    var_52 = 'numbers'
    var_53 = 2
    var_54 = 3
    var_55 = [var_17, var_53, var_54]
    var_56 = {var_52: var_55}
    var_57 = module_0.Token(var_56)
    var_58 = 'array'
    var_59 = module_1.Field()
    var_60 = module_1.Field()
    var_61 = {var_52: var_60}
    var_62 = module_1.Field()
    var_63 = module_2.validate_with_positions(token=var_57, validator=var_62)
    var_64 = 'two'
    var_65 = [var_17, var_64, var_54]
    var_66 = {var_52: var_65}
    var_67 = module_0.Token(var_66)
    var_68 = module_1.Field()
    var_69 = module_1.Field()
    var_70 = {var_52: var_69}
    var_71 = module_1.Field()
    var_72 = module_2.validate_with_positions(token=var_67, validator=var_71)
    var_73 = 'matrix'
    var_74 = [var_17, var_53]
    var_75 = 4
    var_76 = [var_54, var_75]
    var_77 = [var_74, var_76]
    var_78 = {var_73: var_77}
    var_79 = module_0.Token(var_78)
    var_80 = module_1.Field()
    var_81 = module_1.Field()
    var_82 = module_1.Field()
    var_83 = {var_73: var_82}
    var_84 = module_1.Field()
    var_85 = module_2.validate_with_positions(token=var_79, validator=var_84)
    var_86 = [var_17, var_53]
    var_87 = 'four'
    var_88 = [var_54, var_87]
    var_89 = [var_86, var_88]
    var_90 = {var_73: var_89}
    var_91 = module_0.Token(var_90)
    var_92 = module_1.Field()
    var_93 = module_1.Field()
    var_94 = module_1.Field()
    var_95 = {var_73: var_94}
    var_96 = module_1.Field()
    var_97 = module_2.validate_with_positions(token=var_91, validator=var_96)
    var_98 = {var_97: var_2, var_1: var_3}
    var_99 = module_0.Token(var_98)
    var_100 = module_1.Field()
    var_101 = module_1.Field()
    var_102 = {var_97: var_100, var_1: var_101}
    var_103 = module_3.Schema(var_102)
    var_104 = module_2.validate_with_positions(token=var_99, validator=var_103)
    var_105 = {var_97: var_2, var_1: var_23}
    var_106 = module_0.Token(var_105)
    var_107 = module_1.Field()
    var_108 = module_1.Field()
    var_109 = {var_97: var_107, var_1: var_108}
    var_110 = module_3.Schema(var_109)
    var_111 = module_2.validate_with_positions(token=var_106, validator=var_110)
    var_112 = {var_111: var_2, var_1: var_3}
    var_113 = {var_31: var_112}
    var_114 = module_0.Token(var_113)
    var_115 = module_1.Field()
    var_116 = module_1.Field()
    var_117 = {var_111: var_115, var_1: var_116}
    var_118 = module_1.Field()
    var_119 = {var_31: var_118}
    var_120 = module_3.Schema(var_119)
    var_121 = module_2.validate_with_positions(token=var_114, validator=var_120)
    var_122 = {var_111: var_2, var_1: var_23}
    var_123 = {var_31: var_122}
    var_124 = module_0.Token(var_123)
    var_125 = module_1.Field()
    var_126 = module_1.Field()
    var_127 = {var_111: var_125, var_1: var_126}
    var_128 = module_1.Field()
    var_129 = {var_31: var_128}
    var_130 = module_3.Schema(var_129)
    var_131 = module_2.validate_with_positions(token=var_124, validator=var_130)
    var_132 = [var_17, var_53, var_54]
    var_133 = {var_52: var_132}
    var_134 = module_0.Token(var_133)
    var_135 = module_1.Field()
    var_136 = module_1.Field()
    var_137 = {var_52: var_136}
    var_138 = module_3.Schema(var_137)
    var_139 = module_2.validate_with_positions(token=var_134, validator=var_138)
    var_140 = [var_17, var_64, var_54]
    var_141 = {var_52: var_140}
    var_142 = module_0.Token(var_141)
    var_143 = module_1.Field()
    var_144 = module_1.Field()
    var_145 = {var_52: var_144}
    var_146 = module_3.Schema(var_145)
    var_147 = module_2.validate_with_positions(token=var_142, validator=var_146)
    var_148 = [var_17, var_53]
    var_149 = [var_54, var_75]
    var_150 = [var_148, var_149]
    var_151 = {var_73: var_150}
    var_152 = module_0.Token(var_151)
    var_153 = module_1.Field()
    var_154 = module_1.Field()
    var_155 = module_1.Field()
    var_156 = {var_73: var_155}
    var_157 = module_3.Schema(var_156)
    var_158 = module_2.validate_with_positions(token=var_152, validator=var_157)



# Parsed testcases at query #12
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Token(var_4)
    var_6 = 'object'
    var_7 = 'string'
    var_8 = module_1.Field()
    var_9 = 'integer'
    var_10 = module_1.Field()
    var_11 = {var_0: var_8, var_1: var_10}
    var_12 = module_1.Field()
    var_13 = module_2.validate_with_positions(token=var_5, validator=var_12)
    var_14 = {var_0: var_2}
    var_15 = module_0.Token(var_14)
    var_16 = module_1.Field()
    var_17 = True
    var_18 = module_1.Field()
    var_19 = {var_0: var_16, var_1: var_18}
    var_20 = module_1.Field()
    var_21 = module_2.validate_with_positions(token=var_15, validator=var_20)
    var_22 = 'twenty-five'
    var_23 = {var_21: var_2, var_1: var_22}
    var_24 = module_0.Token(var_23)
    var_25 = module_1.Field()
    var_26 = module_1.Field()
    var_27 = {var_21: var_25, var_1: var_26}
    var_28 = module_1.Field()
    var_29 = module_2.validate_with_positions(token=var_24, validator=var_28)
    var_30 = 123
    var_31 = {var_29: var_30, var_1: var_22}
    var_32 = module_0.Token(var_31)
    var_33 = module_1.Field()
    var_34 = module_1.Field()
    var_35 = {var_29: var_33, var_1: var_34}
    var_36 = module_1.Field()
    var_37 = module_2.validate_with_positions(token=var_32, validator=var_36)
    var_38 = 'person'
    var_39 = {var_37: var_2, var_1: var_3}
    var_40 = {var_38: var_39}
    var_41 = module_0.Token(var_40)
    var_42 = module_1.Field()
    var_43 = module_1.Field()
    var_44 = {var_37: var_42, var_1: var_43}
    var_45 = module_1.Field()
    var_46 = {var_38: var_45}
    var_47 = module_1.Field()
    var_48 = module_2.validate_with_positions(token=var_41, validator=var_47)
    var_49 = {var_37: var_2}
    var_50 = {var_38: var_49}
    var_51 = module_0.Token(var_50)
    var_52 = module_1.Field()
    var_53 = module_1.Field()
    var_54 = {var_37: var_52, var_1: var_53}
    var_55 = module_1.Field()
    var_56 = {var_38: var_55}
    var_57 = module_1.Field()
    var_58 = module_2.validate_with_positions(token=var_51, validator=var_57)
    var_59 = {var_58: var_2, var_1: var_22}
    var_60 = {var_38: var_59}
    var_61 = module_0.Token(var_60)
    var_62 = module_1.Field()
    var_63 = module_1.Field()
    var_64 = {var_58: var_62, var_1: var_63}
    var_65 = module_1.Field()
    var_66 = {var_38: var_65}
    var_67 = module_1.Field()
    var_68 = module_2.validate_with_positions(token=var_61, validator=var_67)
    var_69 = {var_68: var_30, var_1: var_22}
    var_70 = {var_38: var_69}
    var_71 = module_0.Token(var_70)
    var_72 = module_1.Field()
    var_73 = module_1.Field()
    var_74 = {var_68: var_72, var_1: var_73}
    var_75 = module_1.Field()
    var_76 = {var_38: var_75}
    var_77 = module_1.Field()
    var_78 = module_2.validate_with_positions(token=var_71, validator=var_77)
    var_79 = 'numbers'
    var_80 = 2
    var_81 = 3
    var_82 = [var_17, var_80, var_81]
    var_83 = {var_79: var_82}
    var_84 = module_0.Token(var_83)
    var_85 = 'array'
    var_86 = module_1.Field()
    var_87 = module_1.Field()
    var_88 = {var_79: var_87}
    var_89 = module_1.Field()
    var_90 = module_2.validate_with_positions(token=var_84, validator=var_89)
    var_91 = 'two'
    var_92 = [var_17, var_91, var_81]
    var_93 = {var_79: var_92}
    var_94 = module_0.Token(var_93)
    var_95 = module_1.Field()
    var_96 = module_1.Field()
    var_97 = {var_79: var_96}
    var_98 = module_1.Field()
    var_99 = module_2.validate_with_positions(token=var_94, validator=var_98)
    var_100 = 'matrix'
    var_101 = [var_17, var_80]
    var_102 = 4
    var_103 = [var_81, var_102]
    var_104 = [var_101, var_103]
    var_105 = {var_100: var_104}
    var_106 = module_0.Token(var_105)
    var_107 = module_1.Field()
    var_108 = module_1.Field()
    var_109 = module_1.Field()
    var_110 = {var_100: var_109}
    var_111 = module_1.Field()
    var_112 = module_2.validate_with_positions(token=var_106, validator=var_111)
    var_113 = [var_17, var_80]
    var_114 = 'three'
    var_115 = [var_114, var_102]
    var_116 = [var_113, var_115]
    var_117 = {var_100: var_116}
    var_118 = module_0.Token(var_117)
    var_119 = module_1.Field()
    var_120 = module_1.Field()
    var_121 = module_1.Field()
    var_122 = {var_100: var_121}
    var_123 = module_1.Field()
    var_124 = module_2.validate_with_positions(token=var_118, validator=var_123)
    var_125 = {var_124: var_2, var_1: var_3}
    var_126 = module_0.Token(var_125)
    var_127 = module_1.Field()
    var_128 = module_1.Field()
    var_129 = {var_124: var_127, var_1: var_128}
    var_130 = module_3.Schema(var_129)
    var_131 = module_2.validate_with_positions(token=var_126, validator=var_130)
    var_132 = {var_124: var_2}
    var_133 = module_0.Token(var_132)
    var_134 = module_1.Field()
    var_135 = module_1.Field()
    var_136 = {var_124: var_134, var_1: var_135}
    var_137 = module_3.Schema(var_136)



# Parsed testcases at query #13
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'John Doe'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0.Token(var_2)
    var_5 = 2
    var_6 = var_1 * var_5
    var_7 = {var_0: var_6}
    var_8 = module_0.Token(var_7)
    var_9 = {}
    var_10 = module_0.Token(var_9)
    var_11 = 'age'
    var_12 = 25
    var_13 = {var_0: var_1, var_11: var_12}
    var_14 = module_0.Token(var_13)
    var_15 = {var_0: var_1, var_11: var_12}
    var_16 = module_0.Token(var_15)
    var_17 = 10
    var_18 = module_1.String(max_length=var_17)
    var_19 = module_2.validate_with_positions(token=var_16, validator=var_18)
    assert var_19 == 'John Doe'
    var_20 = var_1 * var_5
    var_21 = {var_0: var_20}
    var_22 = module_0.Token(var_21)
    var_23 = module_1.String(max_length=var_17)
    var_24 = module_2.validate_with_positions(token=var_22, validator=var_23)
    var_25 = {}
    var_26 = module_0.Token(var_25)
    var_27 = module_1.String(max_length=var_17)
    var_28 = module_2.validate_with_positions(token=var_26, validator=var_27)
    var_29 = {var_28: var_1, var_11: var_12}
    var_30 = module_0.Token(var_29)
    var_31 = module_1.String(max_length=var_17)
    var_32 = module_2.validate_with_positions(token=var_30, validator=var_31)
    assert var_32 == 'John Doe'
    var_33 = {var_28: var_1, var_11: var_12}
    var_34 = module_0.Token(var_33)
    var_35 = module_1.String(max_length=var_17)
    var_36 = module_2.validate_with_positions(token=var_34, validator=var_35)
    assert var_36 == 'John Doe'
    var_37 = {var_28: var_1, var_11: var_12}
    var_38 = module_0.Token(var_37)
    var_39 = module_1.String(max_length=var_17)
    var_40 = module_2.validate_with_positions(token=var_38, validator=var_39)
    assert var_40 == 'John Doe'
    var_41 = {var_28: var_1, var_11: var_12}
    var_42 = module_0.Token(var_41)
    var_43 = module_1.String(max_length=var_17)
    var_44 = module_2.validate_with_positions(token=var_42, validator=var_43)
    assert var_44 == 'John Doe'
    var_45 = {var_28: var_1, var_11: var_12}
    var_46 = module_0.Token(var_45)
    var_47 = module_1.String(max_length=var_17)
    var_48 = module_2.validate_with_positions(token=var_46, validator=var_47)
    assert var_48 == 'John Doe'
    var_49 = {var_28: var_1, var_11: var_12}
    var_50 = module_0.Token(var_49)
    var_51 = module_1.String(max_length=var_17)
    var_52 = module_2.validate_with_positions(token=var_50, validator=var_51)
    assert var_52 == 'John Doe'
    var_53 = {var_28: var_1, var_11: var_12}
    var_54 = module_0.Token(var_53)
    var_55 = module_1.String(max_length=var_17)
    var_56 = module_2.validate_with_positions(token=var_54, validator=var_55)
    assert var_56 == 'John Doe'
    var_57 = {var_28: var_1, var_11: var_12}
    var_58 = module_0.Token(var_57)
    var_59 = module_1.String(max_length=var_17)
    var_60 = module_2.validate_with_positions(token=var_58, validator=var_59)
    assert var_60 == 'John Doe'
    var_61 = {var_28: var_1, var_11: var_12}
    var_62 = module_0.Token(var_61)
    var_63 = module_1.String(max_length=var_17)
    var_64 = module_2.validate_with_positions(token=var_62, validator=var_63)
    assert var_64 == 'John Doe'
    var_65 = {var_28: var_1, var_11: var_12}
    var_66 = module_0.Token(var_65)
    var_67 = module_1.String(max_length=var_17)
    var_68 = module_2.validate_with_positions(token=var_66, validator=var_67)
    assert var_68 == 'John Doe'
    var_69 = {var_28: var_1, var_11: var_12}
    var_70 = module_0.Token(var_69)
    var_71 = module_1.String(max_length=var_17)
    var_72 = module_2.validate_with_positions(token=var_70, validator=var_71)
    assert var_72 == 'John Doe'
    var_73 = {var_28: var_1, var_11: var_12}
    var_74 = module_0.Token(var_73)
    var_75 = module_1.String(max_length=var_17)
    var_76 = module_2.validate_with_positions(token=var_74, validator=var_75)
    assert var_76 == 'John Doe'
    var_77 = {var_28: var_1, var_11: var_12}
    var_78 = module_0.Token(var_77)
    var_79 = module_1.String(max_length=var_17)
    var_80 = module_2.validate_with_positions(token=var_78, validator=var_79)
    assert var_80 == 'John Doe'
    var_81 = {var_28: var_1, var_11: var_12}
    var_82 = module_0.Token(var_81)
    var_83 = module_1.String(max_length=var_17)
    var_84 = module_2.validate_with_positions(token=var_82, validator=var_83)
    assert var_84 == 'John Doe'
    var_85 = {var_28: var_1, var_11: var_12}
    var_86 = module_0.Token(var_85)
    var_87 = module_1.String(max_length=var_17)
    var_88 = module_2.validate_with_positions(token=var_86, validator=var_87)
    assert var_88 == 'John Doe'
    var_89 = {var_28: var_1, var_11: var_12}
    var_90 = module_0.Token(var_89)
    var_91 = module_1.String(max_length=var_17)
    var_92 = module_2.validate_with_positions(token=var_90, validator=var_91)
    assert var_92 == 'John Doe'
    var_93 = {var_28: var_1, var_11: var_12}
    var_94 = module_0.Token(var_93)
    var_95 = module_1.String(max_length=var_17)
    var_96 = module_2.validate_with_positions(token=var_94, validator=var_95)
    assert var_96 == 'John Doe'
    var_97 = {var_28: var_1, var_11: var_12}
    var_98 = module_0.Token(var_97)
    var_99 = module_1.String(max_length=var_17)
    var_100 = module_2.validate_with_positions(token=var_98, validator=var_99)
    assert var_100 == 'John Doe'
    var_101 = {var_28: var_1, var_11: var_12}
    var_102 = module_0.Token(var_101)
    var_103 = module_1.String(max_length=var_17)
    var_104 = module_2.validate_with_positions(token=var_102, validator=var_103)
    assert var_104 == 'John Doe'
    var_105 = {var_28: var_1, var_11: var_12}
    var_106 = module_0.Token(var_105)
    var_107 = module_1.String(max_length=var_17)
    var_108 = module_2.validate_with_positions(token=var_106, validator=var_107)
    assert var_108 == 'John Doe'
    var_109 = {var_28: var_1, var_11: var_12}
    var_110 = module_0.Token(var_109)
    var_111 = module_1.String(max_length=var_17)
    var_112 = module_2.validate_with_positions(token=var_110, validator=var_111)
    assert var_112 == 'John Doe'
    var_113 = {var_28: var_1, var_11: var_12}
    var_114 = module_0.Token(var_113)
    var_115 = module_1.String(max_length=var_17)
    var_116 = module_2.validate_with_positions(token=var_114, validator=var_115)
    assert var_116 == 'John Doe'
    var_117 = {var_28: var_1, var_11: var_12}
    var_118 = module_0.Token(var_117)
    var_119 = module_1.String(max_length=var_17)
    var_120 = module_2.validate_with_positions(token=var_118, validator=var_119)
    assert var_120 == 'John Doe'
    var_121 = {var_28: var_1, var_11: var_12}
    var_122 = module_0.Token(var_121)
    var_123 = module_1.String(max_length=var_17)
    var_124 = module_2.validate_with_positions(token=var_122, validator=var_123)
    assert var_124 == 'John Doe'
    var_125 = {var_28: var_1, var_11: var_12}
    var_126 = module_0.Token(var_125)
    var_127 = module_1.String(max_length=var_17)



# Parsed testcases at query #14
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'John'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Token(var_4)
    var_6 = {var_0: var_2}
    var_7 = module_0.Token(var_6)
    var_8 = 'twenty-five'
    var_9 = {var_0: var_2, var_1: var_8}
    var_10 = module_0.Token(var_9)
    var_11 = 123
    var_12 = {var_0: var_11, var_1: var_8}
    var_13 = module_0.Token(var_12)
    var_14 = 'person'
    var_15 = {var_0: var_2, var_1: var_3}
    var_16 = {var_14: var_15}
    var_17 = module_0.Token(var_16)
    var_18 = {var_0: var_2, var_1: var_8}
    var_19 = {var_14: var_18}
    var_20 = module_0.Token(var_19)
    var_21 = 'numbers'
    var_22 = 1
    var_23 = 2
    var_24 = 3
    var_25 = [var_22, var_23, var_24]
    var_26 = {var_21: var_25}
    var_27 = module_0.Token(var_26)
    var_28 = 'two'
    var_29 = [var_22, var_28, var_24]
    var_30 = {var_21: var_29}
    var_31 = module_0.Token(var_30)
    var_32 = 'matrix'
    var_33 = [var_22, var_23]
    var_34 = 4
    var_35 = [var_24, var_34]
    var_36 = [var_33, var_35]
    var_37 = {var_32: var_36}
    var_38 = module_0.Token(var_37)
    var_39 = [var_22, var_23]
    var_40 = 'four'
    var_41 = [var_24, var_40]
    var_42 = [var_39, var_41]
    var_43 = {var_32: var_42}
    var_44 = module_0.Token(var_43)



