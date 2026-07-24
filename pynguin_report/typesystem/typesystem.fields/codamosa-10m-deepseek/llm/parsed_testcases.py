####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = module_0.Choice(choices=var_6)
    var_8 = var_7.validate(var_0)
    assert var_8 == 'a'
    var_9 = 'c'
    var_10 = var_7.validate(var_9)
    var_11 = (var_9, var_10)
    var_12 = (var_3, var_4)
    var_13 = [var_11, var_12]
    var_14 = True
    var_15 = module_0.Choice(choices=var_13)
    var_16 = None
    var_17 = var_15.validate(var_16)
    assert var_17 is None
    var_18 = (var_9, var_10)
    var_19 = (var_3, var_4)
    var_20 = [var_18, var_19]
    var_21 = module_0.Choice(choices=var_20)
    var_22 = None
    var_23 = var_21.validate(var_22)
    var_24 = (var_22, var_23)
    var_25 = (var_3, var_4)
    var_26 = [var_24, var_25]
    var_27 = module_0.Choice(choices=var_26)
    var_28 = ''
    var_29 = var_27.validate(var_28)
    assert var_29 is None
    var_30 = (var_22, var_23)
    var_31 = (var_3, var_4)
    var_32 = [var_30, var_31]
    var_33 = module_0.Choice(choices=var_32)
    var_34 = ''
    var_35 = var_33.validate(var_34)



# Parsed testcases at query #2
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = module_0.Choice(choices=var_6)
    var_8 = var_7.validate(var_0)
    assert var_8 == 'a'
    var_9 = var_7.validate(var_3)
    assert var_9 == 'b'
    var_10 = 'c'
    var_11 = var_7.validate(var_10)
    var_12 = (var_10, var_11)
    var_13 = (var_3, var_4)
    var_14 = [var_12, var_13]
    var_15 = True
    var_16 = module_0.Choice(choices=var_14)
    var_17 = None
    var_18 = var_16.validate(var_17)
    assert var_18 is None
    var_19 = 'a'
    var_20 = 'A'
    var_21 = (var_19, var_20)
    var_22 = 'b'
    var_23 = 'B'
    var_24 = (var_22, var_23)
    var_25 = [var_21, var_24]
    var_26 = False
    var_27 = module_0.Choice(choices=var_25)
    var_28 = None
    var_29 = var_27.validate(var_28)
    var_30 = (var_19, var_20)
    var_31 = (var_22, var_23)
    var_32 = [var_30, var_31]
    var_33 = module_0.Choice(choices=var_32, coerce_types=var_15)
    var_34 = ''
    var_35 = var_33.validate(var_34)
    assert var_35 is None
    var_36 = 'a'
    var_37 = 'A'
    var_38 = (var_36, var_37)
    var_39 = 'b'
    var_40 = 'B'
    var_41 = (var_39, var_40)
    var_42 = [var_38, var_41]
    var_43 = False
    var_44 = module_0.Choice(choices=var_42)
    var_45 = ''
    var_46 = var_44.validate(var_45)
    var_47 = 'a'
    var_48 = 'A'
    var_49 = (var_47, var_48)
    var_50 = 'b'
    var_51 = 'B'
    var_52 = (var_50, var_51)
    var_53 = [var_49, var_52]
    var_54 = True
    var_55 = False
    var_56 = module_0.Choice(choices=var_53, coerce_types=var_55)
    var_57 = ''
    var_58 = var_56.validate(var_57)



# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = module_0.Choice(choices=var_6)
    var_8 = var_7.validate(var_0)
    assert var_8 == 'a'
    var_9 = 'c'
    var_10 = var_7.validate(var_9)
    var_11 = (var_9, var_10)
    var_12 = (var_3, var_4)
    var_13 = [var_11, var_12]
    var_14 = True
    var_15 = module_0.Choice(choices=var_13)
    var_16 = None
    var_17 = var_15.validate(var_16)
    assert var_17 is None
    var_18 = 'a'
    var_19 = 'A'
    var_20 = (var_18, var_19)
    var_21 = 'b'
    var_22 = 'B'
    var_23 = (var_21, var_22)
    var_24 = [var_20, var_23]
    var_25 = False
    var_26 = module_0.Choice(choices=var_24)
    var_27 = None
    var_28 = var_26.validate(var_27)
    var_29 = (var_18, var_19)
    var_30 = (var_21, var_22)
    var_31 = [var_29, var_30]
    var_32 = module_0.Choice(choices=var_31, coerce_types=var_14)
    var_33 = ''
    var_34 = var_32.validate(var_33)
    assert var_34 is None
    var_35 = 'a'
    var_36 = 'A'
    var_37 = (var_35, var_36)
    var_38 = 'b'
    var_39 = 'B'
    var_40 = (var_38, var_39)
    var_41 = [var_37, var_40]
    var_42 = False
    var_43 = module_0.Choice(choices=var_41)
    var_44 = ''
    var_45 = var_43.validate(var_44)



# Parsed testcases at query #4
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Const(var_0)
    var_2 = 'test'
    var_3 = module_0.Const(var_2)
    var_4 = 123
    var_5 = module_0.Const(var_4)
    var_6 = True
    var_7 = module_0.Const(var_6)
    var_8 = 2
    var_9 = 3
    var_10 = [var_6, var_8, var_9]
    var_11 = module_0.Const(var_10)
    var_12 = 'key'
    var_13 = 'value'
    var_14 = {var_12: var_13}
    var_15 = module_0.Const(var_14)



# Parsed testcases at query #5
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = 5
    var_2 = var_0.validate(var_1)
    assert var_2 == 5
    var_3 = 5.5
    var_4 = var_0.validate(var_3)
    var_5 = None
    var_6 = var_0.validate(var_5)
    assert var_6 is None
    var_7 = '5'
    var_8 = var_0.validate(var_7)
    assert var_8 == 5
    var_9 = '5.5'
    var_10 = var_0.validate(var_9)
    var_11 = ''
    var_12 = var_0.validate(var_11)
    assert var_12 is None
    var_13 = 'abc'
    var_14 = var_0.validate(var_13)
    assert var_14 is None
    var_15 = 'inf'
    var_16 = var_0.validate(var_15)
    assert var_16 is None
    var_17 = '-inf'
    var_18 = var_0.validate(var_17)
    assert var_18 is None
    var_19 = 'nan'
    var_20 = var_0.validate(var_19)
    assert var_20 is None
    var_21 = '5.5.5'
    var_22 = var_0.validate(var_21)
    assert var_22 is None
    var_23 = '5.5.5.5'
    var_24 = var_0.validate(var_23)
    assert var_24 is None
    var_25 = '5.5.5.5.5'
    var_26 = var_0.validate(var_25)
    assert var_26 is None
    var_27 = '5.5.5.5.5.5'
    var_28 = var_0.validate(var_27)
    assert var_28 is None
    var_29 = '5.5.5.5.5.5.5'
    var_30 = var_0.validate(var_29)
    assert var_30 is None
    var_31 = '5.5.5.5.5.5.5.5'
    var_32 = var_0.validate(var_31)
    assert var_32 is None
    var_33 = '5.5.5.5.5.5.5.5.5'
    var_34 = var_0.validate(var_33)
    assert var_34 is None
    var_35 = '5.5.5.5.5.5.5.5.5.5'
    var_36 = var_0.validate(var_35)
    assert var_36 is None
    var_37 = '5.5.5.5.5.5.5.5.5.5.5'
    var_38 = var_0.validate(var_37)
    assert var_38 is None
    var_39 = '5.5.5.5.5.5.5.5.5.5.5.5'
    var_40 = var_0.validate(var_39)
    assert var_40 is None
    var_41 = '5.5.5.5.5.5.5.5.5.5.5.5.5'
    var_42 = var_0.validate(var_41)
    assert var_42 is None
    var_43 = '5.5.5.5.5.5.5.5.5.5.5.5.5.5'
    var_44 = var_0.validate(var_43)
    assert var_44 is None
    var_45 = '5.5.5.5.5.5.5.5.5.5.5.5.5.5.5'
    var_46 = var_0.validate(var_45)
    assert var_46 is None
    var_47 = '5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5'
    var_48 = var_0.validate(var_47)
    assert var_48 is None
    var_49 = '5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5'
    var_50 = var_0.validate(var_49)
    assert var_50 is None
    var_51 = '5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5'
    var_52 = var_0.validate(var_51)
    assert var_52 is None
    var_53 = '5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5'
    var_54 = var_0.validate(var_53)
    assert var_54 is None
    var_55 = '5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5'
    var_56 = var_0.validate(var_55)
    assert var_56 is None
    var_57 = '5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5'
    var_58 = var_0.validate(var_57)
    assert var_58 is None
    var_59 = '5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5'
    var_60 = var_0.validate(var_59)
    assert var_60 is None
    var_61 = '5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5'
    var_62 = var_0.validate(var_61)
    assert var_62 is None
    var_63 = '5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5'
    var_64 = var_0.validate(var_63)
    assert var_64 is None
    var_65 = '5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5'
    var_66 = var_0.validate(var_65)
    assert var_66 is None
    var_67 = '5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5'
    var_68 = var_0.validate(var_67)
    assert var_68 is None
    var_69 = '5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5'
    var_70 = var_0.validate(var_69)
    assert var_70 is None
    var_71 = '5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5'
    var_72 = var_0.validate(var_71)
    assert var_72 is None
    var_73 = '5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5'
    var_74 = var_0.validate(var_73)
    assert var_74 is None
    var_75 = '5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5'
    var_76 = var_0.validate(var_75)
    assert var_76 is None
    var_77 = '5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5'
    var_78 = var_0.validate(var_77)
    assert var_78 is None
    var_79 = '5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5'
    var_80 = var_0.validate(var_79)
    assert var_80 is None
    var_81 = '5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5'
    var_82 = var_0.validate(var_81)
    assert var_82 is None
    var_83 = '5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5'
    var_84 = var_0.validate(var_83)
    assert var_84 is None
    var_85 = '5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5'
    var_86 = var_0.validate(var_85)
    assert var_86 is None
    var_87 = '5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5'
    var_88 = var_0.validate(var_87)
    assert var_88 is None
    var_89 = '5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5'
    var_90 = var_0.validate(var_89)
    assert var_90 is None
    var_91 = '5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5'
    var_92 = var_0.validate(var_91)
    assert var_92 is None
    var_93 = '5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5.5'
    var_94 = var_0.validate(var_93)
    assert var_94 is None



# Parsed testcases at query #6
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'hello'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'hello'
    var_3 = True
    var_4 = module_0.String()
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None
    var_7 = False
    var_8 = module_0.String()
    var_9 = None
    var_10 = var_8.validate(var_9)
    var_11 = module_0.String(allow_blank=var_3)
    var_12 = ''
    var_13 = var_11.validate(var_12)
    assert var_13 == ''
    var_14 = module_0.String(allow_blank=var_7)
    var_15 = ''
    var_16 = var_14.validate(var_15)
    var_17 = 5
    var_18 = module_0.String(max_length=var_17)
    var_19 = 'hello world'
    var_20 = var_18.validate(var_19)
    var_21 = 10
    var_22 = module_0.String(min_length=var_21)
    var_23 = 'hello'
    var_24 = var_22.validate(var_23)
    var_25 = '^\\d+$'
    var_26 = module_0.String(pattern=var_25)
    var_27 = '123'
    var_28 = var_26.validate(var_27)
    assert var_28 == '123'
    var_29 = module_0.String(pattern=var_25)
    var_30 = 'abc'
    var_31 = var_29.validate(var_30)
    var_32 = 'email'
    var_33 = module_0.String(format=var_32)
    var_34 = 'test@example.com'
    var_35 = var_33.validate(var_34)
    assert var_35 == 'test@example.com'
    var_36 = module_0.String(format=var_32)
    var_37 = 'invalid-email'
    var_38 = var_36.validate(var_37)
    var_39 = module_0.String(coerce_types=var_3)
    var_40 = 123
    var_41 = var_39.validate(var_40)
    var_42 = module_0.String(coerce_types=var_7)
    var_43 = 123
    var_44 = var_42.validate(var_43)



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'valid'
    var_1 = 'invalid'



# Parsed testcases at query #8
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'test'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'test'
    var_3 = 3
    var_4 = module_0.String(max_length=var_3)
    var_5 = 'test'
    var_6 = var_4.validate(var_5)
    var_7 = 5
    var_8 = module_0.String(min_length=var_7)
    var_9 = 'test'
    var_10 = var_8.validate(var_9)
    var_11 = '^[a-z]+$'
    var_12 = module_0.String(pattern=var_11)
    var_13 = 'test123'
    var_14 = var_12.validate(var_13)
    var_15 = 'email'
    var_16 = module_0.String(format=var_15)
    var_17 = 'test'
    var_18 = var_16.validate(var_17)
    var_19 = True
    var_20 = module_0.String()
    var_21 = None
    var_22 = var_20.validate(var_21)
    assert var_22 is None
    var_23 = False
    var_24 = module_0.String()
    var_25 = None
    var_26 = var_24.validate(var_25)
    var_27 = module_0.String(allow_blank=var_19)
    var_28 = ''
    var_29 = var_27.validate(var_28)
    assert var_29 == ''
    var_30 = module_0.String(allow_blank=var_23)
    var_31 = ''
    var_32 = var_30.validate(var_31)
    var_33 = module_0.String(trim_whitespace=var_19)
    var_34 = '  test  '
    var_35 = var_33.validate(var_34)
    assert var_35 == 'test'
    var_36 = module_0.String(trim_whitespace=var_23)
    var_37 = var_36.validate(var_34)
    assert var_37 == '  test  '
    var_38 = module_0.String(coerce_types=var_19)
    var_39 = 123
    var_40 = var_38.validate(var_39)
    assert var_40 == '123'
    var_41 = module_0.String(coerce_types=var_23)
    var_42 = 123
    var_43 = var_41.validate(var_42)



# Parsed testcases at query #9
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.Array(var_0)
    var_3 = var_2.serialize(var_0)
    assert var_3 is None
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = [var_4, var_5]
    var_7 = module_0.Array(var_6)
    var_8 = 'test'
    var_9 = 123
    var_10 = [var_8, var_9]
    var_11 = var_7.serialize(var_10)
    var_12 = module_0.String()
    var_13 = module_0.Array(var_12)
    var_14 = 'test1'
    var_15 = 'test2'
    var_16 = [var_14, var_15]
    var_17 = var_13.serialize(var_16)
    var_18 = module_0.Array(var_0)
    var_19 = [var_1, var_8, var_1]
    var_20 = var_18.serialize(var_19)



# Parsed testcases at query #10
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = 42
    var_2 = var_0.validate(var_1)
    assert var_2 == 42
    var_3 = module_0.Number()
    var_4 = 3.14
    var_5 = var_3.validate(var_4)
    var_6 = True
    var_7 = module_0.Number(coerce_types=var_6)
    var_8 = '42'
    var_9 = var_7.validate(var_8)
    assert var_9 == 42
    var_10 = False
    var_11 = module_0.Number(coerce_types=var_10)
    var_12 = '42'
    var_13 = var_11.validate(var_12)
    var_14 = module_0.Number()
    var_15 = None
    var_16 = var_14.validate(var_15)
    assert var_16 is None
    var_17 = module_0.Number()
    var_18 = None
    var_19 = var_17.validate(var_18)
    var_20 = module_0.Number(coerce_types=var_6)
    var_21 = ''
    var_22 = var_20.validate(var_21)
    assert var_22 is None
    var_23 = module_0.Number(coerce_types=var_6)
    var_24 = ''
    var_25 = var_23.validate(var_24)
    var_26 = module_0.Number()
    var_27 = True
    var_28 = var_26.validate(var_27)
    var_29 = module_0.Integer()
    var_30 = 3.14
    var_31 = module_0.Number()
    var_32 = 'inf'
    var_33 = float(var_32)
    var_34 = var_31.validate(var_33)
    var_35 = module_0.Number()
    var_36 = 'nan'
    var_37 = float(var_36)
    var_38 = var_35.validate(var_37)
    var_39 = 10
    var_40 = module_0.Number(minimum=var_39)
    var_41 = var_40.validate(var_39)
    assert var_41 == 10
    var_42 = 11
    var_43 = var_40.validate(var_42)
    assert var_43 == 11
    var_44 = 9
    var_45 = var_40.validate(var_44)
    var_46 = module_0.Number(exclusive_minimum=var_39)
    var_47 = var_46.validate(var_42)
    assert var_47 == 11
    var_48 = 10
    var_49 = var_46.validate(var_48)
    var_50 = module_0.Number(maximum=var_39)
    var_51 = var_50.validate(var_39)
    assert var_51 == 10
    var_52 = 9
    var_53 = var_50.validate(var_52)
    assert var_53 == 9
    var_54 = 11
    var_55 = var_50.validate(var_54)
    var_56 = module_0.Number(exclusive_maximum=var_39)
    var_57 = var_56.validate(var_52)
    assert var_57 == 9
    var_58 = 10
    var_59 = var_56.validate(var_58)
    var_60 = 2
    var_61 = module_0.Number(multiple_of=var_60)
    var_62 = 4
    var_63 = var_61.validate(var_62)
    assert var_63 == 4
    var_64 = 3
    var_65 = var_61.validate(var_64)
    var_66 = 0.5
    var_67 = module_0.Number(multiple_of=var_66)
    var_68 = var_67.validate(var_6)
    var_69 = 1.5
    var_70 = var_67.validate(var_69)
    var_71 = 1.3
    var_72 = var_67.validate(var_71)
    var_73 = '0.01'
    var_74 = module_0.Number(precision=var_73)
    var_75 = 1.234
    var_76 = var_74.validate(var_75)
    var_77 = 1.235
    var_78 = var_74.validate(var_77)
    var_79 = 'All tests passed successfully!'
    var_80 = print(var_79)



# Parsed testcases at query #11
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = module_0.Field()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)



# Parsed testcases at query #12
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Array()
    var_1 = module_0.Integer()
    var_2 = [var_1]
    var_3 = module_0.String()
    var_4 = 1
    var_5 = 10
    var_6 = True
    var_7 = module_0.Array(var_2, var_3, var_4, var_5, unique_items=var_6)
    var_8 = 5
    var_9 = module_0.Array(exact_items=var_8)



# Parsed testcases at query #13
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Field(default=var_0)
    var_2 = var_1.get_default_value()
    assert var_2 == 42
    var_3 = 100
    var_4 = lambda : var_3
    var_5 = module_0.Field(default=var_4)
    var_6 = var_5.get_default_value()
    assert var_6 == 100
    var_7 = module_0.Field()
    var_8 = 'default'
    var_9 = hasattr(var_7, var_8)
    assert var_9 is False



# Parsed testcases at query #14
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = 42
    var_2 = var_0.validate(var_1)
    assert var_2 == 42
    var_3 = module_0.Number()
    var_4 = 3.14
    var_5 = var_3.validate(var_4)
    var_6 = True
    var_7 = module_0.Number()
    var_8 = None
    var_9 = var_7.validate(var_8)
    assert var_9 is None
    var_10 = module_0.Number()
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = module_0.Number()
    var_14 = True
    var_15 = var_13.validate(var_14)
    var_16 = False
    var_17 = module_0.Number(coerce_types=var_16)
    var_18 = 'not a number'
    var_19 = var_17.validate(var_18)
    var_20 = module_0.Number(coerce_types=var_6)
    var_21 = '42'
    var_22 = var_20.validate(var_21)
    assert var_22 == 42
    var_23 = module_0.Number()
    var_24 = 'inf'
    var_25 = float(var_24)
    var_26 = var_23.validate(var_25)
    var_27 = module_0.Number()
    var_28 = 'nan'
    var_29 = float(var_28)
    var_30 = var_27.validate(var_29)
    var_31 = 10
    var_32 = module_0.Number(minimum=var_31)
    var_33 = 5
    var_34 = var_32.validate(var_33)
    var_35 = module_0.Number(maximum=var_31)
    var_36 = 15
    var_37 = var_35.validate(var_36)
    var_38 = 5
    var_39 = module_0.Number(multiple_of=var_38)
    var_40 = 7
    var_41 = var_39.validate(var_40)
    var_42 = module_0.Number(multiple_of=var_38)
    var_43 = var_42.validate(var_31)
    assert var_43 == 10
    var_44 = module_0.Number(exclusive_minimum=var_31)
    var_45 = 10
    var_46 = var_44.validate(var_45)
    var_47 = module_0.Number(exclusive_maximum=var_31)
    var_48 = 10
    var_49 = var_47.validate(var_48)
    var_50 = '0.01'
    var_51 = module_0.Number(precision=var_50)
    var_52 = 3.14159
    var_53 = var_51.validate(var_52)



# Parsed testcases at query #15
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = 42
    var_2 = var_0.validate(var_1)
    assert var_2 == 42
    var_3 = module_0.Number()
    var_4 = 3.14
    var_5 = var_3.validate(var_4)
    var_6 = True
    var_7 = module_0.Number()
    var_8 = None
    var_9 = var_7.validate(var_8)
    assert var_9 is None
    var_10 = False
    var_11 = module_0.Number()
    var_12 = None
    var_13 = var_11.validate(var_12)
    var_14 = module_0.Number(coerce_types=var_6)
    var_15 = 'not a number'
    var_16 = var_14.validate(var_15)
    var_17 = module_0.Number(coerce_types=var_10)
    var_18 = 'not a number'
    var_19 = var_17.validate(var_18)
    var_20 = module_0.Number()
    var_21 = 'inf'
    var_22 = float(var_21)
    var_23 = var_20.validate(var_22)
    var_24 = module_0.Number()
    var_25 = 'nan'
    var_26 = float(var_25)
    var_27 = var_24.validate(var_26)
    var_28 = 10
    var_29 = module_0.Number(minimum=var_28)
    var_30 = 5
    var_31 = var_29.validate(var_30)
    var_32 = 20
    var_33 = module_0.Number(maximum=var_32)
    var_34 = 25
    var_35 = var_33.validate(var_34)
    var_36 = 5
    var_37 = module_0.Number(multiple_of=var_36)
    var_38 = 7
    var_39 = var_37.validate(var_38)
    var_40 = module_0.Number(multiple_of=var_36)
    var_41 = var_40.validate(var_28)
    assert var_41 == 10
    var_42 = module_0.Number(exclusive_minimum=var_28)
    var_43 = 10
    var_44 = var_42.validate(var_43)
    var_45 = module_0.Number(exclusive_maximum=var_32)
    var_46 = 20
    var_47 = var_45.validate(var_46)
    var_48 = '0.01'
    var_49 = module_0.Number(precision=var_48)
    var_50 = 3.14159
    var_51 = var_49.validate(var_50)
    var_52 = module_0.Number(precision=var_48, coerce_types=var_6)
    var_53 = '3.14159'
    var_54 = var_52.validate(var_53)
    var_55 = module_0.Number(precision=var_48, coerce_types=var_10)
    var_56 = var_55.validate(var_50)
    var_57 = '1'
    var_58 = var_55.validate(var_50)
    assert var_58 == 3
    var_59 = var_55.validate(var_50)
    var_60 = var_55.validate(var_50)
    var_61 = '3.14'



# Parsed testcases at query #16
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None
    var_4 = False
    var_5 = module_0.Array()
    var_6 = None
    var_7 = var_5.validate(var_6)
    var_8 = module_0.Array()
    var_9 = 'not a list'
    var_10 = var_8.validate(var_9)
    var_11 = module_0.Array(min_items=var_9)
    var_12 = []
    var_13 = var_11.validate(var_12)
    var_14 = 2
    var_15 = module_0.Array(exact_items=var_14)
    var_16 = 1
    var_17 = [var_16]
    var_18 = var_15.validate(var_17)
    var_19 = module_0.Array(unique_items=var_16)
    var_20 = 1
    var_21 = [var_20, var_20]
    var_22 = var_19.validate(var_21)
    var_23 = module_0.Integer()
    var_24 = module_0.Array(var_23)
    var_25 = 'not an integer'
    var_26 = [var_25]
    var_27 = var_24.validate(var_26)
    var_28 = module_0.Integer()
    var_29 = [var_28]
    var_30 = module_0.Array(var_29, var_4)
    var_31 = 1
    var_32 = 2
    var_33 = [var_31, var_32]
    var_34 = var_30.validate(var_33)
    var_35 = module_0.Integer()
    var_36 = module_0.Array(var_35)
    var_37 = [var_31, var_14]
    var_38 = var_36.validate(var_37)



# Parsed testcases at query #17
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Array()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = var_0.validate(var_4)
    var_6 = module_0.Array(min_items=var_2)
    var_7 = [var_1, var_2]
    var_8 = var_6.validate(var_7)
    var_9 = module_0.Array(max_items=var_3)
    var_10 = [var_1, var_2, var_3]
    var_11 = var_9.validate(var_10)
    var_12 = module_0.Array(min_items=var_3)
    var_13 = 1
    var_14 = 2
    var_15 = [var_13, var_14]
    var_16 = var_12.validate(var_15)
    var_17 = module_0.Array(max_items=var_14)
    var_18 = 1
    var_19 = 2
    var_20 = 3
    var_21 = [var_18, var_19, var_20]
    var_22 = var_17.validate(var_21)
    var_23 = module_0.Array(exact_items=var_19)
    var_24 = 1
    var_25 = [var_24]
    var_26 = var_23.validate(var_25)
    var_27 = True
    var_28 = module_0.Array(unique_items=var_27)
    var_29 = [var_27, var_25, var_26]
    var_30 = var_28.validate(var_29)
    var_31 = True
    var_32 = module_0.Array(unique_items=var_31)
    var_33 = 1
    var_34 = 2
    var_35 = [var_33, var_33, var_34]
    var_36 = var_32.validate(var_35)
    var_37 = module_0.Integer()
    var_38 = module_0.Array(var_37)
    var_39 = [var_31, var_34, var_35]
    var_40 = var_38.validate(var_39)
    var_41 = module_0.Integer()
    var_42 = module_0.Array(var_41)
    var_43 = 1
    var_44 = 'a'
    var_45 = 3
    var_46 = [var_43, var_44, var_45]
    var_47 = var_42.validate(var_46)
    var_48 = module_0.Integer()
    var_49 = module_0.Integer()
    var_50 = [var_48, var_49]
    var_51 = True
    var_52 = module_0.Array(var_50, var_51)
    var_53 = [var_51, var_44, var_45]
    var_54 = var_52.validate(var_53)
    var_55 = module_0.Integer()
    var_56 = module_0.Integer()
    var_57 = [var_55, var_56]
    var_58 = False
    var_59 = module_0.Array(var_57, var_58)
    var_60 = 1
    var_61 = 2
    var_62 = 3
    var_63 = [var_60, var_61, var_62]
    var_64 = var_59.validate(var_63)
    var_65 = module_0.Integer()
    var_66 = module_0.Integer()
    var_67 = [var_65, var_66]
    var_68 = module_0.Float()
    var_69 = module_0.Array(var_67, var_68)
    var_70 = [var_51, var_61, var_62]
    var_71 = var_69.validate(var_70)
    var_72 = module_0.Integer()
    var_73 = module_0.Integer()
    var_74 = [var_72, var_73]
    var_75 = module_0.Float()
    var_76 = module_0.Array(var_74, var_75)
    var_77 = 1
    var_78 = 2
    var_79 = 'a'
    var_80 = [var_77, var_78, var_79]
    var_81 = var_76.validate(var_80)
    var_82 = True
    var_83 = module_0.Array()
    var_84 = None
    var_85 = var_83.validate(var_84)
    assert var_85 is None
    var_86 = module_0.Array()
    var_87 = None
    var_88 = var_86.validate(var_87)
    var_89 = module_0.Array()
    var_90 = 'not a list'
    var_91 = var_89.validate(var_90)



# Parsed testcases at query #18
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None
    var_4 = False
    var_5 = module_0.Array()
    var_6 = None
    var_7 = var_5.validate(var_6)
    var_8 = module_0.Array()
    var_9 = 'not a list'
    var_10 = var_8.validate(var_9)
    var_11 = module_0.Array(min_items=var_9)
    var_12 = []
    var_13 = var_11.validate(var_12)
    var_14 = 2
    var_15 = module_0.Array(min_items=var_14)
    var_16 = 1
    var_17 = [var_16]
    var_18 = var_15.validate(var_17)
    var_19 = module_0.Array(max_items=var_14)
    var_20 = 1
    var_21 = 2
    var_22 = 3
    var_23 = [var_20, var_21, var_22]
    var_24 = var_19.validate(var_23)
    var_25 = module_0.Array(exact_items=var_24)
    var_26 = 1
    var_27 = [var_26]
    var_28 = var_25.validate(var_27)
    var_29 = module_0.Array(unique_items=var_26)
    var_30 = 1
    var_31 = [var_30, var_30]
    var_32 = var_29.validate(var_31)
    var_33 = module_0.Integer()
    var_34 = module_0.Array(var_33)
    var_35 = 'not an integer'
    var_36 = [var_35]
    var_37 = var_34.validate(var_36)
    var_38 = module_0.Integer()
    var_39 = [var_38]
    var_40 = module_0.Array(var_39, var_23)
    var_41 = 1
    var_42 = 2
    var_43 = [var_41, var_42]
    var_44 = var_40.validate(var_43)
    var_45 = module_0.Integer()
    var_46 = [var_45]
    var_47 = module_0.String()
    var_48 = module_0.Array(var_46, var_47)
    var_49 = 1
    var_50 = 2
    var_51 = [var_49, var_50]
    var_52 = var_48.validate(var_51)
    var_53 = module_0.Integer()
    var_54 = module_0.Array(var_53)
    var_55 = [var_49, var_24]
    var_56 = var_54.validate(var_55)
    var_57 = module_0.Integer()
    var_58 = [var_57]
    var_59 = module_0.String()
    var_60 = module_0.Array(var_58, var_59)
    var_61 = 'valid'
    var_62 = [var_49, var_61]
    var_63 = var_60.validate(var_62)
    var_64 = module_0.Integer()
    var_65 = [var_64]
    var_66 = module_0.Array(var_65, var_49)
    var_67 = [var_49, var_61]
    var_68 = var_66.validate(var_67)



# Parsed testcases at query #19
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Number()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None
    var_4 = False
    var_5 = module_0.Number()
    var_6 = None
    var_7 = var_5.validate(var_6)
    var_8 = module_0.Number(coerce_types=var_6)
    var_9 = ''
    var_10 = var_8.validate(var_9)
    assert var_10 is None
    var_11 = module_0.Number(coerce_types=var_4)
    var_12 = ''
    var_13 = var_11.validate(var_12)
    var_14 = module_0.Number()
    var_15 = True
    var_16 = var_14.validate(var_15)
    var_17 = 1.5
    var_18 = var_14.validate(var_17)
    var_19 = module_0.Number()
    var_20 = 'abc'
    var_21 = var_19.validate(var_20)
    var_22 = module_0.Number()
    var_23 = 'inf'
    var_24 = float(var_23)
    var_25 = var_22.validate(var_24)
    var_26 = '0.01'
    var_27 = module_0.Number(precision=var_26)
    var_28 = 1.234
    var_29 = var_27.validate(var_28)
    var_30 = 10
    var_31 = module_0.Number(minimum=var_30)
    var_32 = 5
    var_33 = var_31.validate(var_32)
    var_34 = module_0.Number(exclusive_minimum=var_30)
    var_35 = 10
    var_36 = var_34.validate(var_35)
    var_37 = module_0.Number(maximum=var_30)
    var_38 = 15
    var_39 = var_37.validate(var_38)
    var_40 = module_0.Number(exclusive_maximum=var_30)
    var_41 = 10
    var_42 = var_40.validate(var_41)
    var_43 = 5
    var_44 = module_0.Number(multiple_of=var_43)
    var_45 = 7
    var_46 = var_44.validate(var_45)
    var_47 = 0.5
    var_48 = module_0.Number(multiple_of=var_47)
    var_49 = 0.7
    var_50 = var_48.validate(var_49)



# Parsed testcases at query #20
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Array()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = var_0.validate(var_4)
    var_6 = True
    var_7 = module_0.Array()
    var_8 = None
    var_9 = var_7.validate(var_8)
    assert var_9 is None
    var_10 = False
    var_11 = module_0.Array()
    var_12 = None
    var_13 = var_11.validate(var_12)
    var_14 = module_0.Array()
    var_15 = 'not a list'
    var_16 = var_14.validate(var_15)
    var_17 = module_0.Array(min_items=var_6)
    var_18 = []
    var_19 = var_17.validate(var_18)
    var_20 = module_0.Array(exact_items=var_19)
    var_21 = 1
    var_22 = [var_21]
    var_23 = var_20.validate(var_22)
    var_24 = True
    var_25 = module_0.Array(unique_items=var_24)
    var_26 = 1
    var_27 = [var_26, var_26]
    var_28 = var_25.validate(var_27)
    var_29 = module_0.Integer()
    var_30 = module_0.Integer()
    var_31 = [var_29, var_30]
    var_32 = module_0.Array(var_31, var_10)
    var_33 = 1
    var_34 = 2
    var_35 = 3
    var_36 = [var_33, var_34, var_35]
    var_37 = var_32.validate(var_36)
    var_38 = module_0.Integer()
    var_39 = module_0.Integer()
    var_40 = [var_38, var_39]
    var_41 = module_0.Float()
    var_42 = module_0.Array(var_40, var_41)
    var_43 = 3.5
    var_44 = [var_24, var_34, var_43]
    var_45 = var_42.validate(var_44)
    var_46 = module_0.Integer()
    var_47 = module_0.Array(var_46)
    var_48 = [var_24, var_34, var_35]
    var_49 = var_47.validate(var_48)



# Parsed testcases at query #21
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'apple'
    var_1 = 'Apple'
    var_2 = (var_0, var_1)
    var_3 = 'banana'
    var_4 = 'Banana'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = module_0.Choice(choices=var_6)
    var_8 = var_7.validate(var_0)
    assert var_8 == 'apple'
    var_9 = var_7.validate(var_3)
    assert var_9 == 'banana'
    var_10 = 'cherry'
    var_11 = var_7.validate(var_10)
    var_12 = None
    var_13 = var_7.validate(var_12)
    var_14 = (var_12, var_13)
    var_15 = (var_3, var_4)
    var_16 = [var_14, var_15]
    var_17 = True
    var_18 = module_0.Choice(choices=var_16)
    var_19 = None
    var_20 = var_18.validate(var_19)
    assert var_20 is None
    var_21 = ''
    var_22 = var_7.validate(var_21)
    var_23 = ''
    var_24 = var_18.validate(var_23)
    assert var_24 is None



# Parsed testcases at query #22
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Const(var_0)
    var_2 = None
    var_3 = module_0.Const(var_2)
    var_4 = 'test'
    var_5 = module_0.Const(var_4)



# Parsed testcases at query #23
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Const(var_0)
    var_2 = None
    var_3 = module_0.Const(var_2)



# Parsed testcases at query #24
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'test'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'test'
    var_3 = True
    var_4 = module_0.String()
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None
    var_7 = False
    var_8 = module_0.String()
    var_9 = None
    var_10 = var_8.validate(var_9)
    var_11 = module_0.String()
    var_12 = 123
    var_13 = var_11.validate(var_12)
    var_14 = module_0.String(allow_blank=var_7)
    var_15 = ''
    var_16 = var_14.validate(var_15)
    var_17 = module_0.String(allow_blank=var_3)
    var_18 = ''
    var_19 = var_17.validate(var_18)
    assert var_19 == ''
    var_20 = 3
    var_21 = module_0.String(max_length=var_20)
    var_22 = 'test'
    var_23 = var_21.validate(var_22)
    var_24 = 5
    var_25 = module_0.String(min_length=var_24)
    var_26 = 'test'
    var_27 = var_25.validate(var_26)
    var_28 = '^[A-Z]+$'
    var_29 = module_0.String(pattern=var_28)
    var_30 = 'test'
    var_31 = var_29.validate(var_30)
    var_32 = 'email'
    var_33 = module_0.String(format=var_32)
    var_34 = 'test@example.com'
    var_35 = var_33.validate(var_34)
    assert var_35 == 'test@example.com'



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Choice(choices=var_3)
    var_5 = 'A'
    var_6 = (var_0, var_5)
    var_7 = 'B'
    var_8 = (var_1, var_7)
    var_9 = [var_6, var_8]
    var_10 = module_0.Choice(choices=var_9)



# Parsed testcases at query #2
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Const(var_0)



# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'hello'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'hello'
    var_3 = True
    var_4 = module_0.String()
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None
    var_7 = module_0.String()
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = module_0.String(allow_blank=var_3)
    var_11 = ''
    var_12 = var_10.validate(var_11)
    assert var_12 == ''
    var_13 = module_0.String()
    var_14 = ''
    var_15 = var_13.validate(var_14)
    var_16 = module_0.String(trim_whitespace=var_3)
    var_17 = ' hello '
    var_18 = var_16.validate(var_17)
    assert var_18 == 'hello'
    var_19 = False
    var_20 = module_0.String(trim_whitespace=var_19)
    var_21 = var_20.validate(var_17)
    assert var_21 == ' hello '
    var_22 = 3
    var_23 = module_0.String(min_length=var_22)
    var_24 = var_23.validate(var_14)
    assert var_24 == 'hello'
    var_25 = 6
    var_26 = module_0.String(min_length=var_25)
    var_27 = 'hello'
    var_28 = var_26.validate(var_27)
    var_29 = 5
    var_30 = module_0.String(max_length=var_29)
    var_31 = var_30.validate(var_27)
    assert var_31 == 'hello'
    var_32 = 4
    var_33 = module_0.String(max_length=var_32)
    var_34 = 'hello'
    var_35 = var_33.validate(var_34)
    var_36 = '^[a-z]+$'
    var_37 = module_0.String(pattern=var_36)
    var_38 = var_37.validate(var_34)
    assert var_38 == 'hello'
    var_39 = '^[0-9]+$'
    var_40 = module_0.String(pattern=var_39)
    var_41 = 'hello'
    var_42 = var_40.validate(var_41)
    var_43 = 'email'
    var_44 = module_0.String(format=var_43)
    var_45 = 'test@example.com'
    var_46 = var_44.validate(var_45)
    assert var_46 == 'test@example.com'
    var_47 = module_0.String(format=var_43)
    var_48 = 'invalid-email'
    var_49 = var_47.validate(var_48)



# Parsed testcases at query #4
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Any()
    var_1 = None
    var_2 = var_0.validate(var_1)
    assert var_2 is None
    var_3 = 1
    var_4 = var_0.validate(var_3)
    assert var_4 == 1
    var_5 = 'string'
    var_6 = var_0.validate(var_5)
    assert var_6 == 'string'
    var_7 = True
    var_8 = var_0.validate(var_7)
    assert var_8 is True
    var_9 = 2
    var_10 = 3
    var_11 = [var_7, var_9, var_10]
    var_12 = var_0.validate(var_11)
    var_13 = 'key'
    var_14 = 'value'
    var_15 = {var_13: var_14}
    var_16 = var_0.validate(var_15)



# Parsed testcases at query #5
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = 123
    var_5 = var_3.validate(var_4)
    assert var_5 == 123
    var_6 = 'abc'
    var_7 = var_3.validate(var_6)
    assert var_7 == 'abc'
    var_8 = True
    var_9 = var_3.validate(var_8)
    var_10 = None
    var_11 = var_3.validate(var_10)
    var_12 = None
    var_13 = var_3.validate(var_12)
    assert var_13 is None
    var_14 = True
    var_15 = module_0.String()
    var_16 = [var_0, var_15]
    var_17 = module_0.Union(var_16)
    var_18 = var_17.validate(var_12)
    assert var_18 is None



# Parsed testcases at query #6
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'A'
    var_1 = 'Option A'
    var_2 = (var_0, var_1)
    var_3 = 'B'
    var_4 = 'Option B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = module_0.Choice(choices=var_6)
    var_8 = var_7.validate(var_0)
    assert var_8 == 'A'
    var_9 = var_7.validate(var_3)
    assert var_9 == 'B'
    var_10 = 'C'
    var_11 = var_7.validate(var_10)
    var_12 = (var_10, var_11)
    var_13 = (var_3, var_4)
    var_14 = [var_12, var_13]
    var_15 = True
    var_16 = module_0.Choice(choices=var_14)
    var_17 = None
    var_18 = var_16.validate(var_17)
    assert var_18 is None
    var_19 = None
    var_20 = var_7.validate(var_19)
    var_21 = ''
    var_22 = var_16.validate(var_21)
    assert var_22 is None
    var_23 = ''
    var_24 = var_7.validate(var_23)



# Parsed testcases at query #7
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = 42
    var_2 = var_0.validate(var_1)
    assert var_2 == 42
    var_3 = module_0.Number()
    var_4 = 3.14
    var_5 = var_3.validate(var_4)
    var_6 = True
    var_7 = module_0.Number(coerce_types=var_6)
    var_8 = '42'
    var_9 = var_7.validate(var_8)
    assert var_9 == 42
    var_10 = module_0.Number()
    var_11 = None
    var_12 = var_10.validate(var_11)
    assert var_12 is None
    var_13 = False
    var_14 = module_0.Number()
    var_15 = None
    var_16 = var_14.validate(var_15)
    var_17 = module_0.Number(coerce_types=var_6)
    var_18 = 'not a number'
    var_19 = var_17.validate(var_18)
    var_20 = module_0.Number()
    var_21 = True
    var_22 = var_20.validate(var_21)
    var_23 = 10
    var_24 = module_0.Number(minimum=var_23)
    var_25 = var_24.validate(var_23)
    assert var_25 == 10
    var_26 = 15
    var_27 = var_24.validate(var_26)
    assert var_27 == 15
    var_28 = 5
    var_29 = var_24.validate(var_28)
    var_30 = module_0.Number(maximum=var_23)
    var_31 = var_30.validate(var_23)
    assert var_31 == 10
    var_32 = 5
    var_33 = var_30.validate(var_32)
    assert var_33 == 5
    var_34 = 15
    var_35 = var_30.validate(var_34)
    var_36 = module_0.Number(multiple_of=var_32)
    var_37 = var_36.validate(var_23)
    assert var_37 == 10
    var_38 = var_36.validate(var_26)
    assert var_38 == 15
    var_39 = 7
    var_40 = var_36.validate(var_39)



# Parsed testcases at query #8
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.Const(var_0)
    var_2 = None
    var_3 = True
    var_4 = module_0.Const(var_2)
    var_5 = 'test'
    var_6 = False
    var_7 = module_0.Const(var_5)



# Parsed testcases at query #9
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None
    var_4 = False
    var_5 = module_0.String()
    var_6 = None
    var_7 = var_5.validate(var_6)
    var_8 = module_0.String(allow_blank=var_6, coerce_types=var_6)
    var_9 = var_8.validate(var_7)
    assert var_9 == ''
    var_10 = module_0.String(allow_blank=var_4, coerce_types=var_6)
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = module_0.String()
    var_14 = 123
    var_15 = var_13.validate(var_14)
    var_16 = module_0.String()
    var_17 = 'test'
    var_18 = var_16.validate(var_17)
    assert var_18 == 'test'
    var_19 = module_0.String(trim_whitespace=var_14)
    var_20 = '  test  '
    var_21 = var_19.validate(var_20)
    assert var_21 == 'test'
    var_22 = module_0.String(trim_whitespace=var_4)
    var_23 = var_22.validate(var_20)
    assert var_23 == '  test  '
    var_24 = module_0.String(allow_blank=var_4)
    var_25 = ''
    var_26 = var_24.validate(var_25)
    var_27 = module_0.String(allow_blank=var_25)
    var_28 = ''
    var_29 = var_27.validate(var_28)
    assert var_29 == ''
    var_30 = 3
    var_31 = module_0.String(min_length=var_30)
    var_32 = 'ab'
    var_33 = var_31.validate(var_32)
    var_34 = module_0.String(max_length=var_30)
    var_35 = 'abcd'
    var_36 = var_34.validate(var_35)
    var_37 = '^[a-z]+$'
    var_38 = module_0.String(pattern=var_37)
    var_39 = '123'
    var_40 = var_38.validate(var_39)
    var_41 = 'email'
    var_42 = module_0.String(format=var_41)
    var_43 = 'invalid-email'
    var_44 = var_42.validate(var_43)



# Parsed testcases at query #10
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = module_0.Choice(choices=var_6)
    var_8 = var_7.validate(var_0)
    assert var_8 == 'a'
    var_9 = var_7.validate(var_3)
    assert var_9 == 'b'
    var_10 = 'c'
    var_11 = var_7.validate(var_10)
    var_12 = (var_10, var_11)
    var_13 = (var_3, var_4)
    var_14 = [var_12, var_13]
    var_15 = True
    var_16 = module_0.Choice(choices=var_14)
    var_17 = None
    var_18 = var_16.validate(var_17)
    assert var_18 is None
    var_19 = None
    var_20 = var_7.validate(var_19)
    var_21 = (var_19, var_20)
    var_22 = (var_3, var_4)
    var_23 = [var_21, var_22]
    var_24 = module_0.Choice(choices=var_23, coerce_types=var_15)
    var_25 = ''
    var_26 = var_24.validate(var_25)
    assert var_26 is None
    var_27 = ''
    var_28 = var_7.validate(var_27)



# Parsed testcases at query #11
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.Array(var_0)
    var_3 = var_2.serialize(var_0)
    assert var_3 is None
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = [var_4, var_5]
    var_7 = module_0.Array(var_6)
    var_8 = 'test'
    var_9 = 123
    var_10 = [var_8, var_9]
    var_11 = var_7.serialize(var_10)
    var_12 = module_0.String()
    var_13 = module_0.Array(var_12)
    var_14 = 'test1'
    var_15 = 'test2'
    var_16 = [var_14, var_15]
    var_17 = var_13.serialize(var_16)
    var_18 = module_0.Array(var_0)
    var_19 = [var_8, var_9]
    var_20 = var_18.serialize(var_19)



# Parsed testcases at query #12
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = var_3.any_of
    var_5 = len(var_4)
    assert var_5 == 2



# Parsed testcases at query #13
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = True
    var_8 = module_0.Choice(choices=var_6)
    var_9 = None
    var_10 = var_8.validate(var_9)
    assert var_10 is None
    var_11 = (var_0, var_1)
    var_12 = (var_3, var_4)
    var_13 = [var_11, var_12]
    var_14 = False
    var_15 = module_0.Choice(choices=var_13)
    var_16 = None
    var_17 = var_15.validate(var_16)
    var_18 = (var_16, var_17)
    var_19 = (var_3, var_4)
    var_20 = [var_18, var_19]
    var_21 = module_0.Choice(choices=var_20)
    var_22 = var_21.validate(var_16)
    assert var_22 == 'a'
    var_23 = (var_16, var_17)
    var_24 = (var_3, var_4)
    var_25 = [var_23, var_24]
    var_26 = module_0.Choice(choices=var_25)
    var_27 = 'c'
    var_28 = var_26.validate(var_27)
    var_29 = (var_27, var_28)
    var_30 = (var_3, var_4)
    var_31 = [var_29, var_30]
    var_32 = module_0.Choice(choices=var_31)
    var_33 = ''
    var_34 = var_32.validate(var_33)
    var_35 = (var_33, var_34)
    var_36 = (var_3, var_4)
    var_37 = [var_35, var_36]
    var_38 = module_0.Choice(choices=var_37)
    var_39 = ''
    var_40 = var_38.validate(var_39)



# Parsed testcases at query #14
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = module_0.Choice(choices=var_6)
    var_8 = [var_0, var_3]
    var_9 = True
    var_10 = 'Title'
    var_11 = 'Description'
    var_12 = False
    var_13 = module_0.Choice(choices=var_8, coerce_types=var_12)



# Parsed testcases at query #15
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.Float()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = 1
    var_5 = var_3.validate(var_4)
    assert var_5 == 1
    var_6 = var_3.validate(var_4)
    var_7 = None
    var_8 = var_3.validate(var_7)
    assert var_8 is None
    var_9 = 'abc'
    var_10 = var_3.validate(var_9)
    var_11 = True
    var_12 = var_3.validate(var_11)



# Parsed testcases at query #16
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Const(var_0)
    var_2 = None
    var_3 = module_0.Const(var_2)



# Parsed testcases at query #17
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = 1
    var_2 = 3
    var_3 = module_0.Array(var_0, min_items=var_1, max_items=var_2)
    var_4 = 2
    var_5 = [var_1, var_4, var_2]
    var_6 = var_3.validate(var_5)
    var_7 = module_0.Integer()
    var_8 = module_0.Array(var_7, min_items=var_1)
    var_9 = []
    var_10 = var_8.validate(var_9)
    var_11 = module_0.Integer()
    var_12 = module_0.Array(var_11, max_items=var_4)
    var_13 = 1
    var_14 = 2
    var_15 = 3
    var_16 = [var_13, var_14, var_15]
    var_17 = var_12.validate(var_16)
    var_18 = module_0.Integer()
    var_19 = True
    var_20 = module_0.Array(var_18, unique_items=var_19)
    var_21 = 1
    var_22 = 2
    var_23 = [var_21, var_21, var_22]
    var_24 = var_20.validate(var_23)
    var_25 = module_0.Integer()
    var_26 = module_0.Integer()
    var_27 = [var_25, var_26]
    var_28 = False
    var_29 = module_0.Array(var_27, var_28)
    var_30 = 1
    var_31 = 2
    var_32 = 3
    var_33 = [var_30, var_31, var_32]
    var_34 = var_29.validate(var_33)
    var_35 = module_0.Integer()
    var_36 = module_0.Integer()
    var_37 = [var_35, var_36]
    var_38 = module_0.String()
    var_39 = module_0.Array(var_37, var_38)
    var_40 = 'three'
    var_41 = [var_19, var_33, var_40]
    var_42 = var_39.validate(var_41)
    var_43 = module_0.Integer()
    var_44 = module_0.Integer()
    var_45 = [var_43, var_44]
    var_46 = module_0.Array(var_45)
    var_47 = []
    var_48 = var_46.validate(var_47)
    var_49 = module_0.Integer()
    var_50 = True
    var_51 = module_0.Array(var_49)
    var_52 = None
    var_53 = var_51.validate(var_52)
    assert var_53 is None
    var_54 = module_0.Integer()
    var_55 = module_0.Array(var_54)
    var_56 = None
    var_57 = var_55.validate(var_56)
    var_58 = module_0.Integer()
    var_59 = module_0.Array(var_58)
    var_60 = 'not an array'
    var_61 = var_59.validate(var_60)
    var_62 = module_0.Integer()
    var_63 = module_0.Array(var_62, exact_items=var_32)
    var_64 = 1
    var_65 = 2
    var_66 = [var_64, var_65]
    var_67 = var_63.validate(var_66)



# Parsed testcases at query #18
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.Field(default=var_0)
    var_2 = var_1.get_default_value()
    assert var_2 == 5
    var_3 = 10
    var_4 = lambda : var_3
    var_5 = module_0.Field(default=var_4)
    var_6 = var_5.get_default_value()
    assert var_6 == 10
    var_7 = module_0.Field()
    var_8 = var_7.get_default_value()



# Parsed testcases at query #19
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = True
    var_2 = var_0.validate(var_1)
    assert var_2 is True
    var_3 = False
    var_4 = var_0.validate(var_3)
    assert var_4 is False
    var_5 = 'true'
    var_6 = var_0.validate(var_5)
    assert var_6 is True
    var_7 = 'false'
    var_8 = var_0.validate(var_7)
    assert var_8 is False
    var_9 = 'on'
    var_10 = var_0.validate(var_9)
    assert var_10 is True
    var_11 = 'off'
    var_12 = var_0.validate(var_11)
    assert var_12 is False
    var_13 = '1'
    var_14 = var_0.validate(var_13)
    assert var_14 is True
    var_15 = '0'
    var_16 = var_0.validate(var_15)
    assert var_16 is False
    var_17 = var_0.validate(var_1)
    assert var_17 is True
    var_18 = var_0.validate(var_3)
    assert var_18 is False
    var_19 = ''
    var_20 = var_0.validate(var_19)
    assert var_20 is False
    var_21 = 'null'
    var_22 = var_0.validate(var_21)
    assert var_22 is None
    var_23 = 'none'
    var_24 = var_0.validate(var_23)
    assert var_24 is None
    var_25 = 'invalid'
    var_26 = var_0.validate(var_25)
    var_27 = 2
    var_28 = var_0.validate(var_27)
    var_29 = None
    var_30 = var_0.validate(var_29)
    var_31 = module_0.Boolean()
    var_32 = None
    var_33 = var_31.validate(var_32)
    assert var_33 is None
    var_34 = var_31.validate(var_21)
    assert var_34 is None
    var_35 = var_31.validate(var_23)
    assert var_35 is None
    var_36 = var_31.validate(var_19)
    assert var_36 is False
    var_37 = var_31.validate(var_3)
    assert var_37 is False
    var_38 = var_31.validate(var_29)
    assert var_38 is True
    var_39 = module_0.Boolean(coerce_types=var_3)
    var_40 = var_39.validate(var_29)
    assert var_40 is True
    var_41 = var_39.validate(var_3)
    assert var_41 is False
    var_42 = 'true'
    var_43 = var_39.validate(var_42)
    var_44 = 'false'
    var_45 = var_39.validate(var_44)
    var_46 = 'on'
    var_47 = var_39.validate(var_46)
    var_48 = 'off'
    var_49 = var_39.validate(var_48)
    var_50 = '1'
    var_51 = var_39.validate(var_50)
    var_52 = '0'
    var_53 = var_39.validate(var_52)
    var_54 = 1
    var_55 = var_39.validate(var_54)
    var_56 = 0
    var_57 = var_39.validate(var_56)
    var_58 = ''
    var_59 = var_39.validate(var_58)
    var_60 = 'null'
    var_61 = var_39.validate(var_60)
    var_62 = 'none'
    var_63 = var_39.validate(var_62)
    var_64 = 'invalid'
    var_65 = var_39.validate(var_64)
    var_66 = 2
    var_67 = var_39.validate(var_66)
    var_68 = None
    var_69 = var_39.validate(var_68)
    var_70 = module_0.Boolean(coerce_types=var_3)
    var_71 = var_70.validate(var_32)
    assert var_71 is None
    var_72 = 'null'
    var_73 = var_70.validate(var_72)
    var_74 = 'none'
    var_75 = var_70.validate(var_74)
    var_76 = ''
    var_77 = var_70.validate(var_76)
    var_78 = var_70.validate(var_3)
    assert var_78 is False
    var_79 = var_70.validate(var_76)
    assert var_79 is True



# Parsed testcases at query #20
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = True
    var_2 = var_0.validate(var_1)
    assert var_2 is True
    var_3 = False
    var_4 = var_0.validate(var_3)
    assert var_4 is False
    var_5 = None
    var_6 = var_0.validate(var_5)
    var_7 = module_0.Boolean()
    var_8 = None
    var_9 = var_7.validate(var_8)
    assert var_9 is None
    var_10 = 'invalid'
    var_11 = var_7.validate(var_10)
    var_12 = module_0.Boolean(coerce_types=var_3)
    var_13 = 'true'
    var_14 = var_12.validate(var_13)
    var_15 = var_12.validate(var_13)
    assert var_15 is True
    var_16 = module_0.Boolean(coerce_types=var_13)
    var_17 = 'true'
    var_18 = var_16.validate(var_17)
    assert var_18 is True
    var_19 = 'false'
    var_20 = var_16.validate(var_19)
    assert var_20 is False
    var_21 = 'on'
    var_22 = var_16.validate(var_21)
    assert var_22 is True
    var_23 = 'off'
    var_24 = var_16.validate(var_23)
    assert var_24 is False
    var_25 = '1'
    var_26 = var_16.validate(var_25)
    assert var_26 is True
    var_27 = '0'
    var_28 = var_16.validate(var_27)
    assert var_28 is False
    var_29 = var_16.validate(var_13)
    assert var_29 is True
    var_30 = var_16.validate(var_3)
    assert var_30 is False
    var_31 = module_0.Boolean(coerce_types=var_13)
    var_32 = ''
    var_33 = var_31.validate(var_32)
    assert var_33 is None
    var_34 = 'null'
    var_35 = var_31.validate(var_34)
    assert var_35 is None
    var_36 = 'none'
    var_37 = var_31.validate(var_36)
    assert var_37 is None



# Parsed testcases at query #21
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Array(var_0)
    var_2 = var_1.items
    var_3 = module_0.String()
    var_4 = module_0.Integer()
    var_5 = [var_3, var_4]
    var_6 = module_0.Array(var_5)
    var_7 = var_6.items
    var_8 = var_6.items
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = var_6.items
    var_11 = module_0.String()
    var_12 = module_0.Integer()
    var_13 = module_0.Array(var_11, var_12)
    var_14 = var_13.items
    var_15 = var_13.additional_items
    var_16 = module_0.String()
    var_17 = 1
    var_18 = 10
    var_19 = module_0.Array(var_16, min_items=var_17, max_items=var_18)
    var_20 = module_0.String()
    var_21 = 5
    var_22 = module_0.Array(var_20, exact_items=var_21)
    var_23 = module_0.String()
    var_24 = True
    var_25 = module_0.Array(var_23, unique_items=var_24)
    var_26 = None
    var_27 = module_0.Array(var_26)



# Parsed testcases at query #22
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = True
    var_2 = var_0.validate(var_1)
    assert var_2 is True
    var_3 = False
    var_4 = var_0.validate(var_3)
    assert var_4 is False
    var_5 = 'true'
    var_6 = var_0.validate(var_5)
    assert var_6 is True
    var_7 = 'false'
    var_8 = var_0.validate(var_7)
    assert var_8 is False
    var_9 = 'on'
    var_10 = var_0.validate(var_9)
    assert var_10 is True
    var_11 = 'off'
    var_12 = var_0.validate(var_11)
    assert var_12 is False
    var_13 = '1'
    var_14 = var_0.validate(var_13)
    assert var_14 is True
    var_15 = '0'
    var_16 = var_0.validate(var_15)
    assert var_16 is False
    var_17 = ''
    var_18 = var_0.validate(var_17)
    assert var_18 is False
    var_19 = var_0.validate(var_1)
    assert var_19 is True
    var_20 = var_0.validate(var_3)
    assert var_20 is False
    var_21 = None
    var_22 = var_0.validate(var_21)
    assert var_22 is None
    var_23 = 'null'
    var_24 = var_0.validate(var_23)
    assert var_24 is None
    var_25 = 'none'
    var_26 = var_0.validate(var_25)
    assert var_26 is None
    var_27 = var_0.validate(var_17)
    assert var_27 is False
    var_28 = 'invalid'
    var_29 = var_0.validate(var_28)
    assert var_29 is False
    var_30 = 2
    var_31 = var_0.validate(var_30)
    assert var_31 is False
    var_32 = var_0.validate(var_1)
    assert var_32 is True
    var_33 = var_0.validate(var_3)
    assert var_33 is False
    var_34 = 1.1
    var_35 = var_0.validate(var_34)
    assert var_35 is False
    var_36 = 0.1
    var_37 = var_0.validate(var_36)
    assert var_37 is False
    var_38 = var_0.validate(var_34)
    assert var_38 is False
    var_39 = var_0.validate(var_36)
    assert var_39 is False
    var_40 = var_0.validate(var_34)
    assert var_40 is False
    var_41 = var_0.validate(var_36)
    assert var_41 is False
    var_42 = var_0.validate(var_34)
    assert var_42 is False
    var_43 = var_0.validate(var_36)
    assert var_43 is False
    var_44 = var_0.validate(var_34)
    assert var_44 is False
    var_45 = var_0.validate(var_36)
    assert var_45 is False
    var_46 = var_0.validate(var_34)
    assert var_46 is False
    var_47 = var_0.validate(var_36)
    assert var_47 is False
    var_48 = var_0.validate(var_34)
    assert var_48 is False
    var_49 = var_0.validate(var_36)
    assert var_49 is False
    var_50 = var_0.validate(var_34)
    assert var_50 is False
    var_51 = var_0.validate(var_36)
    assert var_51 is False
    var_52 = var_0.validate(var_34)
    assert var_52 is False
    var_53 = var_0.validate(var_36)
    assert var_53 is False
    var_54 = var_0.validate(var_34)
    assert var_54 is False
    var_55 = var_0.validate(var_36)
    assert var_55 is False
    var_56 = var_0.validate(var_34)
    assert var_56 is False
    var_57 = var_0.validate(var_36)
    assert var_57 is False
    var_58 = var_0.validate(var_34)
    assert var_58 is False
    var_59 = var_0.validate(var_36)
    assert var_59 is False
    var_60 = var_0.validate(var_34)
    assert var_60 is False
    var_61 = var_0.validate(var_36)
    assert var_61 is False
    var_62 = var_0.validate(var_34)
    assert var_62 is False
    var_63 = var_0.validate(var_36)
    assert var_63 is False
    var_64 = var_0.validate(var_34)
    assert var_64 is False
    var_65 = var_0.validate(var_36)
    assert var_65 is False
    var_66 = var_0.validate(var_34)
    assert var_66 is False
    var_67 = var_0.validate(var_36)
    assert var_67 is False
    var_68 = var_0.validate(var_34)
    assert var_68 is False
    var_69 = var_0.validate(var_36)
    assert var_69 is False
    var_70 = var_0.validate(var_34)
    assert var_70 is False
    var_71 = var_0.validate(var_36)
    assert var_71 is False
    var_72 = var_0.validate(var_34)
    assert var_72 is False
    var_73 = var_0.validate(var_36)
    assert var_73 is False
    var_74 = var_0.validate(var_34)
    assert var_74 is False
    var_75 = var_0.validate(var_36)
    assert var_75 is False
    var_76 = var_0.validate(var_34)
    assert var_76 is False
    var_77 = var_0.validate(var_36)
    assert var_77 is False
    var_78 = var_0.validate(var_34)
    assert var_78 is False
    var_79 = var_0.validate(var_36)
    assert var_79 is False
    var_80 = var_0.validate(var_34)
    assert var_80 is False
    var_81 = var_0.validate(var_36)
    assert var_81 is False
    var_82 = var_0.validate(var_34)
    assert var_82 is False
    var_83 = var_0.validate(var_36)
    assert var_83 is False
    var_84 = var_0.validate(var_34)
    assert var_84 is False
    var_85 = var_0.validate(var_36)
    assert var_85 is False
    var_86 = var_0.validate(var_34)
    assert var_86 is False
    var_87 = var_0.validate(var_36)
    assert var_87 is False
    var_88 = var_0.validate(var_34)
    assert var_88 is False
    var_89 = var_0.validate(var_36)
    assert var_89 is False
    var_90 = var_0.validate(var_34)
    assert var_90 is False
    var_91 = var_0.validate(var_36)
    assert var_91 is False
    var_92 = var_0.validate(var_34)
    assert var_92 is False
    var_93 = var_0.validate(var_36)
    assert var_93 is False
    var_94 = var_0.validate(var_34)
    assert var_94 is False
    var_95 = var_0.validate(var_36)
    assert var_95 is False
    var_96 = var_0.validate(var_34)
    assert var_96 is False
    var_97 = var_0.validate(var_36)
    assert var_97 is False
    var_98 = var_0.validate(var_34)
    assert var_98 is False
    var_99 = var_0.validate(var_36)
    assert var_99 is False
    var_100 = var_0.validate(var_34)
    assert var_100 is False
    var_101 = var_0.validate(var_36)
    assert var_101 is False
    var_102 = var_0.validate(var_34)
    assert var_102 is False
    var_103 = var_0.validate(var_36)
    assert var_103 is False
    var_104 = var_0.validate(var_34)
    assert var_104 is False
    var_105 = var_0.validate(var_36)
    assert var_105 is False
    var_106 = var_0.validate(var_34)
    assert var_106 is False
    var_107 = var_0.validate(var_36)
    assert var_107 is False
    var_108 = var_0.validate(var_34)
    assert var_108 is False
    var_109 = var_0.validate(var_36)
    assert var_109 is False
    var_110 = var_0.validate(var_34)
    assert var_110 is False
    var_111 = var_0.validate(var_36)
    assert var_111 is False
    var_112 = var_0.validate(var_34)
    assert var_112 is False
    var_113 = var_0.validate(var_36)
    assert var_113 is False
    var_114 = var_0.validate(var_34)
    assert var_114 is False
    var_115 = var_0.validate(var_36)
    assert var_115 is False
    var_116 = var_0.validate(var_34)
    assert var_116 is False
    var_117 = var_0.validate(var_36)
    assert var_117 is False
    var_118 = var_0.validate(var_34)
    assert var_118 is False
    var_119 = var_0.validate(var_36)
    assert var_119 is False
    var_120 = var_0.validate(var_34)
    assert var_120 is False
    var_121 = var_0.validate(var_36)
    assert var_121 is False
    var_122 = var_0.validate(var_34)
    assert var_122 is False
    var_123 = var_0.validate(var_36)
    assert var_123 is False
    var_124 = var_0.validate(var_34)
    assert var_124 is False
    var_125 = var_0.validate(var_36)
    assert var_125 is False
    var_126 = var_0.validate(var_34)
    assert var_126 is False
    var_127 = var_0.validate(var_36)
    assert var_127 is False
    var_128 = var_0.validate(var_34)
    assert var_128 is False
    var_129 = var_0.validate(var_36)
    assert var_129 is False
    var_130 = var_0.validate(var_34)
    assert var_130 is False
    var_131 = var_0.validate(var_36)
    assert var_131 is False
    var_132 = var_0.validate(var_34)
    assert var_132 is False
    var_133 = var_0.validate(var_36)
    assert var_133 is False
    var_134 = var_0.validate(var_34)
    assert var_134 is False
    var_135 = var_0.validate(var_36)
    assert var_135 is False
    var_136 = var_0.validate(var_34)
    assert var_136 is False
    var_137 = var_0.validate(var_36)
    assert var_137 is False
    var_138 = var_0.validate(var_34)
    assert var_138 is False
    var_139 = var_0.validate(var_36)
    assert var_139 is False
    var_140 = var_0.validate(var_34)
    assert var_140 is False
    var_141 = var_0.validate(var_36)
    assert var_141 is False
    var_142 = var_0.validate(var_34)
    assert var_142 is False
    var_143 = var_0.validate(var_36)
    assert var_143 is False
    var_144 = var_0.validate(var_34)
    assert var_144 is False
    var_145 = var_0.validate(var_36)
    assert var_145 is False
    var_146 = var_0.validate(var_34)
    assert var_146 is False
    var_147 = var_0.validate(var_36)
    assert var_147 is False
    var_148 = var_0.validate(var_34)
    assert var_148 is False
    var_149 = var_0.validate(var_36)
    assert var_149 is False
    var_150 = var_0.validate(var_34)
    assert var_150 is False
    var_151 = var_0.validate(var_36)
    assert var_151 is False
    var_152 = var_0.validate(var_34)
    assert var_152 is False
    var_153 = var_0.validate(var_36)
    assert var_153 is False
    var_154 = var_0.validate(var_34)
    assert var_154 is False
    var_155 = var_0.validate(var_36)
    assert var_155 is False
    var_156 = var_0.validate(var_34)
    assert var_156 is False
    var_157 = var_0.validate(var_36)
    assert var_157 is False
    var_158 = var_0.validate(var_34)
    assert var_158 is False
    var_159 = var_0.validate(var_36)
    assert var_159 is False
    var_160 = var_0.validate(var_34)
    assert var_160 is False
    var_161 = var_0.validate(var_36)
    assert var_161 is False
    var_162 = var_0.validate(var_34)
    assert var_162 is False
    var_163 = var_0.validate(var_36)
    assert var_163 is False
    var_164 = var_0.validate(var_34)
    assert var_164 is False
    var_165 = var_0.validate(var_36)
    assert var_165 is False
    var_166 = var_0.validate(var_34)
    assert var_166 is False
    var_167 = var_0.validate(var_36)
    assert var_167 is False
    var_168 = var_0.validate(var_34)
    assert var_168 is False
    var_169 = var_0.validate(var_36)
    assert var_169 is False
    var_170 = var_0.validate(var_34)
    assert var_170 is False
    var_171 = var_0.validate(var_36)
    assert var_171 is False
    var_172 = var_0.validate(var_34)
    assert var_172 is False
    var_173 = var_0.validate(var_36)
    assert var_173 is False
    var_174 = var_0.validate(var_34)
    assert var_174 is False
    var_175 = var_0.validate(var_36)
    assert var_175 is False
    var_176 = var_0.validate(var_34)
    assert var_176 is False
    var_177 = var_0.validate(var_36)
    assert var_177 is False
    var_178 = var_0.validate(var_34)
    assert var_178 is False
    var_179 = var_0.validate(var_36)
    assert var_179 is False
    var_180 = var_0.validate(var_34)
    assert var_180 is False
    var_181 = var_0.validate(var_36)
    assert var_181 is False
    var_182 = var_0.validate(var_34)
    assert var_182 is False
    var_183 = var_0.validate(var_36)
    assert var_183 is False
    var_184 = var_0.validate(var_34)
    assert var_184 is False
    var_185 = var_0.validate(var_36)
    assert var_185 is False
    var_186 = var_0.validate(var_34)
    assert var_186 is False
    var_187 = var_0.validate(var_36)
    assert var_187 is False
    var_188 = var_0.validate(var_34)
    assert var_188 is False
    var_189 = var_0.validate(var_36)
    assert var_189 is False



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Const(var_0)
    var_2 = None
    var_3 = True
    var_4 = module_0.Const(var_2)
    var_5 = None
    var_6 = False
    var_7 = module_0.Const(var_5)



# Parsed testcases at query #2
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.Field(default=var_0)
    var_2 = var_1.get_default_value()
    assert var_2 == 10
    var_3 = 20
    var_4 = lambda : var_3
    var_5 = module_0.Field(default=var_4)
    var_6 = var_5.get_default_value()
    assert var_6 == 20
    var_7 = module_0.Field()
    var_8 = 'default'
    var_9 = hasattr(var_7, var_8)
    var_10 = var_7.get_default_value()
    assert var_10 is None



# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = module_0.Boolean()
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Union(var_3)
    var_5 = 123
    var_6 = var_4.validate(var_5)
    assert var_6 == 123
    var_7 = 'hello'
    var_8 = var_4.validate(var_7)
    assert var_8 == 'hello'
    var_9 = True
    var_10 = var_4.validate(var_9)
    assert var_10 is True
    var_11 = module_0.Integer()
    var_12 = module_0.String()
    var_13 = [var_11, var_12]
    var_14 = module_0.Union(var_13)
    var_15 = None
    var_16 = var_14.validate(var_15)
    assert var_16 is None
    var_17 = module_0.Integer()
    var_18 = module_0.String()
    var_19 = [var_17, var_18]
    var_20 = module_0.Union(var_19)
    var_21 = None
    var_22 = var_20.validate(var_21)
    var_23 = 1.23
    var_24 = var_20.validate(var_23)
    var_25 = 10
    var_26 = module_0.Integer(minimum=var_25)
    var_27 = module_0.String()
    var_28 = [var_26, var_27]
    var_29 = module_0.Union(var_28)
    var_30 = 5
    var_31 = var_29.validate(var_30)



# Parsed testcases at query #4
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None
    var_4 = False
    var_5 = module_0.Array()
    var_6 = None
    var_7 = var_5.validate(var_6)
    var_8 = module_0.Array()
    var_9 = 'not a list'
    var_10 = var_8.validate(var_9)
    var_11 = module_0.Array(min_items=var_9)
    var_12 = []
    var_13 = var_11.validate(var_12)
    var_14 = 2
    var_15 = module_0.Array(min_items=var_14)
    var_16 = 1
    var_17 = [var_16]
    var_18 = var_15.validate(var_17)
    var_19 = module_0.Array(max_items=var_14)
    var_20 = 1
    var_21 = 2
    var_22 = 3
    var_23 = [var_20, var_21, var_22]
    var_24 = var_19.validate(var_23)
    var_25 = module_0.Array(exact_items=var_24)
    var_26 = 1
    var_27 = [var_26]
    var_28 = var_25.validate(var_27)
    var_29 = module_0.Array(unique_items=var_26)
    var_30 = 1
    var_31 = [var_30, var_30]
    var_32 = var_29.validate(var_31)
    var_33 = module_0.Array()
    var_34 = 3
    var_35 = [var_30, var_24, var_34]
    var_36 = var_33.validate(var_35)
    var_37 = module_0.Integer()
    var_38 = module_0.Array(var_37)
    var_39 = '1'
    var_40 = '2'
    var_41 = [var_39, var_40]
    var_42 = var_38.validate(var_41)
    var_43 = module_0.Integer()
    var_44 = module_0.Array(var_43)
    var_45 = 'not an integer'
    var_46 = [var_45]
    var_47 = var_44.validate(var_46)
    var_48 = module_0.Integer()
    var_49 = [var_48]
    var_50 = module_0.Array(var_49, var_23)
    var_51 = 1
    var_52 = 2
    var_53 = [var_51, var_52]
    var_54 = var_50.validate(var_53)
    var_55 = module_0.Integer()
    var_56 = [var_55]
    var_57 = module_0.String()
    var_58 = module_0.Array(var_56, var_57)
    var_59 = 'extra'
    var_60 = [var_51, var_59]
    var_61 = var_58.validate(var_60)
    var_62 = 'All Array.validate tests passed'
    var_63 = print(var_62)



# Parsed testcases at query #5
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = 1
    var_2 = 3
    var_3 = module_0.Array(var_0, min_items=var_1, max_items=var_2)
    var_4 = 2
    var_5 = [var_1, var_4, var_2]
    var_6 = var_3.validate(var_5)
    var_7 = [var_1]
    var_8 = var_3.validate(var_7)
    var_9 = []
    var_10 = var_3.validate(var_9)
    var_11 = 1
    var_12 = 2
    var_13 = 3
    var_14 = 4
    var_15 = [var_11, var_12, var_13, var_14]
    var_16 = var_3.validate(var_15)
    var_17 = 'not an integer'
    var_18 = [var_17]
    var_19 = var_3.validate(var_18)
    var_20 = module_0.Integer()
    var_21 = module_0.String()
    var_22 = [var_20, var_21]
    var_23 = False
    var_24 = module_0.Array(var_22, var_23)
    var_25 = 'a'
    var_26 = [var_18, var_25]
    var_27 = var_24.validate(var_26)
    var_28 = 1
    var_29 = 'a'
    var_30 = 'extra'
    var_31 = [var_28, var_29, var_30]
    var_32 = var_24.validate(var_31)
    var_33 = module_0.Integer()
    var_34 = module_0.String()
    var_35 = [var_33, var_34]
    var_36 = module_0.Integer()
    var_37 = module_0.Array(var_35, var_36)
    var_38 = [var_29, var_25, var_31]
    var_39 = var_37.validate(var_38)
    var_40 = 1
    var_41 = 'a'
    var_42 = 'not an integer'
    var_43 = [var_40, var_41, var_42]
    var_44 = var_37.validate(var_43)
    var_45 = True
    var_46 = module_0.Array(unique_items=var_45)
    var_47 = [var_45, var_43, var_42]
    var_48 = var_46.validate(var_47)
    var_49 = 1
    var_50 = 2
    var_51 = [var_49, var_49, var_50]
    var_52 = var_46.validate(var_51)
    var_53 = True
    var_54 = module_0.Array()
    var_55 = None
    var_56 = var_54.validate(var_55)
    assert var_56 is None
    var_57 = False
    var_58 = module_0.Array()
    var_59 = None
    var_60 = var_58.validate(var_59)
    var_61 = module_0.Array()
    var_62 = 'key'
    var_63 = 'value'
    var_64 = {var_62: var_63}
    var_65 = [var_53, var_25, var_64]
    var_66 = var_61.validate(var_65)
    var_67 = module_0.Array(exact_items=var_52)
    var_68 = [var_53, var_52]
    var_69 = var_67.validate(var_68)
    var_70 = 1
    var_71 = [var_70]
    var_72 = var_67.validate(var_71)
    var_73 = 1
    var_74 = 2
    var_75 = 3
    var_76 = [var_73, var_74, var_75]
    var_77 = var_67.validate(var_76)



# Parsed testcases at query #6
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.Const(var_0)
    var_2 = None
    var_3 = module_0.Const(var_2)
    var_4 = 'test'
    var_5 = module_0.Const(var_4)
    var_6 = True
    var_7 = module_0.Const(var_6)



# Parsed testcases at query #7
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Array()
    var_1 = module_0.String()
    var_2 = module_0.Integer()
    var_3 = [var_1, var_2]
    var_4 = module_0.Array(var_3)
    var_5 = len(var_3)
    var_6 = len(var_3)
    var_7 = module_0.String()
    var_8 = module_0.Array(var_3, var_7)
    var_9 = len(var_3)
    var_10 = 1
    var_11 = 10
    var_12 = module_0.Array(min_items=var_10, max_items=var_11)
    var_13 = 5
    var_14 = module_0.Array(exact_items=var_13)
    var_15 = True
    var_16 = module_0.Array(unique_items=var_15)
    var_17 = True
    var_18 = module_0.Array()
    var_19 = False
    var_20 = module_0.Array()



# Parsed testcases at query #8
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'hello'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'hello'
    var_3 = True
    var_4 = module_0.String()
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None
    var_7 = False
    var_8 = module_0.String()
    var_9 = None
    var_10 = var_8.validate(var_9)
    var_11 = module_0.String(allow_blank=var_3)
    var_12 = ''
    var_13 = var_11.validate(var_12)
    assert var_13 == ''
    var_14 = module_0.String(allow_blank=var_7)
    var_15 = ''
    var_16 = var_14.validate(var_15)
    var_17 = 5
    var_18 = module_0.String(min_length=var_17)
    var_19 = var_18.validate(var_15)
    assert var_19 == 'hello'
    var_20 = 'hi'
    var_21 = var_18.validate(var_20)
    var_22 = module_0.String(max_length=var_17)
    var_23 = var_22.validate(var_20)
    assert var_23 == 'hello'
    var_24 = 'hello world'
    var_25 = var_22.validate(var_24)
    var_26 = '^[a-z]+$'
    var_27 = module_0.String(pattern=var_26)
    var_28 = var_27.validate(var_24)
    assert var_28 == 'hello'
    var_29 = 'hello123'
    var_30 = var_27.validate(var_29)
    var_31 = 'email'
    var_32 = module_0.String(format=var_31)
    var_33 = 'test@example.com'
    var_34 = var_32.validate(var_33)
    assert var_34 == 'test@example.com'
    var_35 = 'invalid-email'
    var_36 = var_32.validate(var_35)
    var_37 = module_0.String(trim_whitespace=var_3)
    var_38 = '  hello  '
    var_39 = var_37.validate(var_38)
    assert var_39 == 'hello'
    var_40 = module_0.String(trim_whitespace=var_7)
    var_41 = var_40.validate(var_38)
    assert var_41 == '  hello  '
    var_42 = module_0.String(allow_blank=var_3, coerce_types=var_3)
    var_43 = var_42.validate(var_5)
    assert var_43 == ''
    var_44 = module_0.String(allow_blank=var_3, coerce_types=var_7)
    var_45 = None
    var_46 = var_44.validate(var_45)
    var_47 = module_0.String(coerce_types=var_3)
    var_48 = var_47.validate(var_5)
    assert var_48 is None
    var_49 = module_0.String(coerce_types=var_7)
    var_50 = var_49.validate(var_5)
    assert var_50 is None
    var_51 = module_0.String()
    var_52 = 'hello\x00world'
    var_53 = var_51.validate(var_52)
    var_54 = 'uuid'
    var_55 = module_0.String(format=var_54)
    var_56 = '123e4567-e89b-12d3-a456-426614174000'
    var_57 = var_55.validate(var_56)
    var_58 = module_0.String(format=var_54)
    var_59 = 'invalid-uuid'
    var_60 = var_58.validate(var_59)
    var_61 = module_0.String(format=var_54)
    var_62 = 123
    var_63 = var_61.validate(var_62)
    var_64 = module_0.String(format=var_54)
    var_65 = '123e4567-e89b-12d3-a456-426614174000'
    var_66 = var_64.serialize(var_65)



# Parsed testcases at query #9
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 0
    var_3 = 11
    var_4 = '0.01'
    var_5 = 2
    var_6 = True
    var_7 = module_0.Number(minimum=var_0, maximum=var_1, exclusive_minimum=var_2, exclusive_maximum=var_3, precision=var_4, multiple_of=var_5, coerce_types=var_6)
    var_8 = var_7.validate(var_5)
    assert var_8 == 2
    var_9 = 8
    var_10 = var_7.validate(var_9)
    assert var_10 == 8
    var_11 = var_7.validate(var_1)
    assert var_11 == 10
    var_12 = 12
    var_13 = var_7.validate(var_12)
    var_14 = 0
    var_15 = var_7.validate(var_14)
    var_16 = 1.23
    var_17 = var_7.validate(var_16)
    var_18 = 'not_a_number'
    var_19 = var_7.validate(var_18)



# Parsed testcases at query #10
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = module_0.Boolean()
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Union(var_3)
    var_5 = 123
    var_6 = var_4.validate(var_5)
    assert var_6 == 123
    var_7 = 'abc'
    var_8 = var_4.validate(var_7)
    assert var_8 == 'abc'
    var_9 = True
    var_10 = var_4.validate(var_9)
    assert var_10 is True
    var_11 = 1.23
    var_12 = var_4.validate(var_11)
    var_13 = None
    var_14 = var_4.validate(var_13)
    var_15 = module_0.Integer()
    var_16 = module_0.String()
    var_17 = module_0.Boolean()
    var_18 = [var_15, var_16, var_17]
    var_19 = module_0.Union(var_18)
    var_20 = None
    var_21 = var_19.validate(var_20)
    assert var_21 is None



# Parsed testcases at query #11
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = [var_0]
    var_6 = False
    var_7 = module_0.Object(properties=var_4, additional_properties=var_6, required=var_5)
    var_8 = 'John'
    var_9 = 30
    var_10 = {var_0: var_8, var_1: var_9}
    var_11 = var_7.validate(var_10)
    var_12 = {var_0: var_8}
    var_13 = var_7.validate(var_12)
    var_14 = 'age'
    var_15 = 30
    var_16 = {var_14: var_15}
    var_17 = var_7.validate(var_16)
    var_18 = 'name'
    var_19 = 'age'
    var_20 = 'John'
    var_21 = 'thirty'
    var_22 = {var_18: var_20, var_19: var_21}
    var_23 = var_7.validate(var_22)
    var_24 = 'name'
    var_25 = 'height'
    var_26 = 'John'
    var_27 = 180
    var_28 = {var_24: var_26, var_25: var_27}
    var_29 = var_7.validate(var_28)
    var_30 = None
    var_31 = var_7.validate(var_30)
    var_32 = 'not an object'
    var_33 = var_7.validate(var_32)



# Parsed testcases at query #12
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.Array(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.serialize(var_5)
    var_7 = []
    var_8 = var_1.serialize(var_7)
    var_9 = None
    var_10 = var_1.serialize(var_9)
    assert var_10 is None
    var_11 = module_0.Integer()
    var_12 = module_0.String()
    var_13 = [var_11, var_12]
    var_14 = module_0.Array(var_13)
    var_15 = 'hello'
    var_16 = [var_2, var_15]
    var_17 = var_14.serialize(var_16)
    var_18 = 'world'
    var_19 = [var_3, var_18]
    var_20 = var_14.serialize(var_19)
    var_21 = var_14.serialize(var_9)
    assert var_21 is None
    var_22 = []
    var_23 = module_0.Array(var_22)
    var_24 = [var_2, var_3, var_4]
    var_25 = var_23.serialize(var_24)
    var_26 = []
    var_27 = var_23.serialize(var_26)
    var_28 = var_23.serialize(var_9)
    assert var_28 is None



# Parsed testcases at query #13
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.Const(var_0)



# Parsed testcases at query #14
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.Const(var_0)
    var_2 = None
    var_3 = module_0.Const(var_2)
    var_4 = 'test'
    var_5 = module_0.Const(var_4)



# Parsed testcases at query #15
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Decimal()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    var_3 = '1.5'
    var_4 = '2.0'



# Parsed testcases at query #16
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)



# Parsed testcases at query #17
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_0.Object(properties=var_2)
    var_4 = 'test'
    var_5 = {var_0: var_4}
    var_6 = var_3.validate(var_5)
    var_7 = module_0.String()
    var_8 = {var_0: var_7}
    var_9 = True
    var_10 = module_0.Object(properties=var_8)
    var_11 = None
    var_12 = var_10.validate(var_11)
    assert var_12 is None
    var_13 = module_0.String()
    var_14 = {var_0: var_13}
    var_15 = False
    var_16 = module_0.Object(properties=var_14)
    var_17 = None
    var_18 = var_16.validate(var_17)
    var_19 = module_0.String()
    var_20 = {var_17: var_19}
    var_21 = module_0.Object(properties=var_20)
    var_22 = 'not an object'
    var_23 = var_21.validate(var_22)
    var_24 = module_0.String()
    var_25 = {var_22: var_24}
    var_26 = [var_22]
    var_27 = module_0.Object(properties=var_25, required=var_26)
    var_28 = {}
    var_29 = var_27.validate(var_28)
    var_30 = '^test_'
    var_31 = module_0.String()
    var_32 = {var_30: var_31}
    var_33 = module_0.Object(pattern_properties=var_32)
    var_34 = 'test_name'
    var_35 = {var_34: var_4}
    var_36 = var_33.validate(var_35)
    var_37 = module_0.String()
    var_38 = {var_28: var_37}
    var_39 = module_0.Object(properties=var_38, additional_properties=var_15)
    var_40 = 'name'
    var_41 = 'extra'
    var_42 = 'test'
    var_43 = 'field'
    var_44 = {var_40: var_42, var_41: var_43}
    var_45 = var_39.validate(var_44)
    var_46 = module_0.String()
    var_47 = {var_40: var_46}
    var_48 = module_0.Integer()
    var_49 = module_0.Object(properties=var_47, additional_properties=var_48)
    var_50 = 'extra'
    var_51 = 123
    var_52 = {var_40: var_43, var_50: var_51}
    var_53 = var_49.validate(var_52)
    var_54 = module_0.Object(min_properties=var_9)
    var_55 = {}
    var_56 = var_54.validate(var_55)
    var_57 = module_0.Object(max_properties=var_9)
    var_58 = 'a'
    var_59 = 'b'
    var_60 = 1
    var_61 = 2
    var_62 = {var_58: var_60, var_59: var_61}
    var_63 = var_57.validate(var_62)
    var_64 = '^[a-z]+$'
    var_65 = module_0.String(pattern=var_64)
    var_66 = module_0.Object(property_names=var_65)
    var_67 = '123'
    var_68 = 'test'
    var_69 = {var_67: var_68}
    var_70 = var_66.validate(var_69)



# Parsed testcases at query #18
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = 42
    var_2 = var_0.validate(var_1)
    assert var_2 == 42
    var_3 = module_0.Number()
    var_4 = 3.14
    var_5 = var_3.validate(var_4)
    var_6 = True
    var_7 = module_0.Number(coerce_types=var_6)
    var_8 = '42'
    var_9 = var_7.validate(var_8)
    assert var_9 == 42
    var_10 = False
    var_11 = module_0.Number(coerce_types=var_10)
    var_12 = '42'
    var_13 = var_11.validate(var_12)
    var_14 = module_0.Number()
    var_15 = None
    var_16 = var_14.validate(var_15)
    assert var_16 is None
    var_17 = module_0.Number()
    var_18 = None
    var_19 = var_17.validate(var_18)
    var_20 = module_0.Number(coerce_types=var_6)
    var_21 = ''
    var_22 = var_20.validate(var_21)
    assert var_22 is None
    var_23 = module_0.Number(coerce_types=var_6)
    var_24 = ''
    var_25 = var_23.validate(var_24)
    var_26 = module_0.Number()
    var_27 = True
    var_28 = var_26.validate(var_27)
    var_29 = module_0.Number()
    var_30 = 'inf'
    var_31 = float(var_30)
    var_32 = var_29.validate(var_31)
    var_33 = 10
    var_34 = module_0.Number(minimum=var_33)
    var_35 = var_34.validate(var_33)
    assert var_35 == 10
    var_36 = 15
    var_37 = var_34.validate(var_36)
    assert var_37 == 15
    var_38 = 5
    var_39 = var_34.validate(var_38)
    var_40 = module_0.Number(exclusive_minimum=var_33)
    var_41 = 11
    var_42 = var_40.validate(var_41)
    assert var_42 == 11
    var_43 = 10
    var_44 = var_40.validate(var_43)
    var_45 = module_0.Number(maximum=var_33)
    var_46 = var_45.validate(var_33)
    assert var_46 == 10
    var_47 = 5
    var_48 = var_45.validate(var_47)
    assert var_48 == 5
    var_49 = 15
    var_50 = var_45.validate(var_49)
    var_51 = module_0.Number(exclusive_maximum=var_33)
    var_52 = 9
    var_53 = var_51.validate(var_52)
    assert var_53 == 9
    var_54 = 10
    var_55 = var_51.validate(var_54)
    var_56 = module_0.Number(multiple_of=var_47)
    var_57 = var_56.validate(var_33)
    assert var_57 == 10
    var_58 = var_56.validate(var_36)
    assert var_58 == 15
    var_59 = 12
    var_60 = var_56.validate(var_59)
    var_61 = 0.5
    var_62 = module_0.Number(multiple_of=var_61)
    var_63 = var_62.validate(var_6)
    var_64 = 1.5
    var_65 = var_62.validate(var_64)
    var_66 = 1.2
    var_67 = var_62.validate(var_66)
    var_68 = '0.01'
    var_69 = module_0.Number(precision=var_68)
    var_70 = 1.234
    var_71 = var_69.validate(var_70)
    var_72 = 1.235
    var_73 = var_69.validate(var_72)
    var_74 = 'All tests passed!'
    var_75 = print(var_74)



# Parsed testcases at query #19
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Object()
    var_1 = 'key'
    var_2 = module_0.String()
    var_3 = {var_1: var_2}
    var_4 = 'pattern'
    var_5 = module_0.String()
    var_6 = {var_4: var_5}
    var_7 = False
    var_8 = module_0.String()
    var_9 = 1
    var_10 = 10
    var_11 = [var_1]
    var_12 = module_0.Object(properties=var_3, pattern_properties=var_6, additional_properties=var_7, property_names=var_8, min_properties=var_9, max_properties=var_10, required=var_11)
    var_13 = module_0.String()
    var_14 = {var_1: var_13}
    var_15 = module_0.String()
    var_16 = {var_4: var_15}
    var_17 = var_12.property_names
    var_18 = module_0.String()
    var_19 = module_0.Object(additional_properties=var_18)
    var_20 = var_19.additional_properties



# Parsed testcases at query #20
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Const(var_0)



# Parsed testcases at query #21
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'B'
    var_5 = (var_3, var_4)
    var_6 = 'c'
    var_7 = 'C'
    var_8 = (var_6, var_7)
    var_9 = [var_2, var_5, var_8]
    var_10 = module_0.Choice(choices=var_9)
    var_11 = var_10.validate(var_0)
    assert var_11 == 'a'
    var_12 = var_10.validate(var_3)
    assert var_12 == 'b'
    var_13 = var_10.validate(var_6)
    assert var_13 == 'c'
    var_14 = 'd'
    var_15 = var_10.validate(var_14)
    var_16 = True
    var_17 = module_0.Choice(choices=var_9)
    var_18 = None
    var_19 = var_17.validate(var_18)
    assert var_19 is None
    var_20 = None
    var_21 = var_10.validate(var_20)
    var_22 = ''
    var_23 = var_17.validate(var_22)
    assert var_23 is None
    var_24 = ''
    var_25 = var_10.validate(var_24)



# Parsed testcases at query #22
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = 1
    var_2 = 3
    var_3 = module_0.Array(var_0, min_items=var_1, max_items=var_2)
    var_4 = 2
    var_5 = [var_1, var_4, var_2]
    var_6 = var_3.validate(var_5)
    var_7 = [var_1]
    var_8 = var_3.validate(var_7)
    var_9 = []
    var_10 = var_3.validate(var_9)
    var_11 = None
    var_12 = var_3.validate(var_11)
    assert var_12 is None
    var_13 = module_0.Integer()
    var_14 = False
    var_15 = module_0.Array(var_13, min_items=var_1, max_items=var_2)
    var_16 = None
    var_17 = var_15.validate(var_16)
    var_18 = module_0.Integer()
    var_19 = module_0.Array(var_18, min_items=var_17, max_items=var_2)
    var_20 = 'not an integer'
    var_21 = [var_20]
    var_22 = var_19.validate(var_21)
    var_23 = module_0.Integer()
    var_24 = module_0.Array(var_23, min_items=var_4, max_items=var_4)
    var_25 = [var_21, var_4]
    var_26 = var_24.validate(var_25)
    var_27 = 1
    var_28 = [var_27]
    var_29 = var_24.validate(var_28)
    var_30 = module_0.Integer()
    var_31 = True
    var_32 = module_0.Array(var_30, unique_items=var_31)
    var_33 = [var_31, var_4, var_29]
    var_34 = var_32.validate(var_33)
    var_35 = 1
    var_36 = [var_35, var_35]
    var_37 = var_32.validate(var_36)
    var_38 = module_0.Integer()
    var_39 = module_0.String()
    var_40 = [var_38, var_39]
    var_41 = module_0.Array(var_40, var_14)
    var_42 = 'a'
    var_43 = [var_31, var_42]
    var_44 = var_41.validate(var_43)
    var_45 = 1
    var_46 = 'a'
    var_47 = 'extra'
    var_48 = [var_45, var_46, var_47]
    var_49 = var_41.validate(var_48)
    var_50 = module_0.Integer()
    var_51 = module_0.String()
    var_52 = [var_50, var_51]
    var_53 = module_0.Integer()
    var_54 = module_0.Array(var_52, var_53)
    var_55 = [var_31, var_42, var_48]
    var_56 = var_54.validate(var_55)
    var_57 = 1
    var_58 = 'a'
    var_59 = 'not an integer'
    var_60 = [var_57, var_58, var_59]
    var_61 = var_54.validate(var_60)



# Parsed testcases at query #23
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = module_0.Integer()
    var_5 = True
    var_6 = module_0.String()
    var_7 = [var_4, var_6]
    var_8 = module_0.Union(var_7)
    var_9 = module_0.Integer()
    var_10 = False
    var_11 = module_0.String()
    var_12 = [var_9, var_11]
    var_13 = module_0.Union(var_12)
    var_14 = module_0.Integer()
    var_15 = module_0.String()
    var_16 = [var_14, var_15]
    var_17 = module_0.Union(var_16)
    var_18 = module_0.Integer()
    var_19 = module_0.String()
    var_20 = module_0.Boolean()
    var_21 = [var_18, var_19, var_20]
    var_22 = module_0.Union(var_21)
    var_23 = module_0.Integer()
    var_24 = module_0.String()
    var_25 = module_0.Boolean()
    var_26 = [var_23, var_24, var_25]
    var_27 = module_0.Union(var_26)
    var_28 = module_0.Integer()
    var_29 = module_0.String()
    var_30 = module_0.Boolean()
    var_31 = [var_28, var_29, var_30]
    var_32 = module_0.Union(var_31)
    var_33 = module_0.Integer()
    var_34 = module_0.String()
    var_35 = module_0.Boolean()
    var_36 = [var_33, var_34, var_35]
    var_37 = module_0.Union(var_36)
    var_38 = module_0.Integer()
    var_39 = module_0.String()
    var_40 = module_0.Boolean()
    var_41 = [var_38, var_39, var_40]
    var_42 = module_0.Union(var_41)
    var_43 = module_0.Integer()
    var_44 = module_0.String()
    var_45 = module_0.Boolean()
    var_46 = [var_43, var_44, var_45]
    var_47 = module_0.Union(var_46)
    var_48 = module_0.Integer()
    var_49 = module_0.String()
    var_50 = module_0.Boolean()
    var_51 = [var_48, var_49, var_50]
    var_52 = module_0.Union(var_51)
    var_53 = module_0.Integer()
    var_54 = module_0.String()
    var_55 = module_0.Boolean()
    var_56 = [var_53, var_54, var_55]
    var_57 = module_0.Union(var_56)
    var_58 = module_0.Integer()
    var_59 = module_0.String()
    var_60 = module_0.Boolean()
    var_61 = [var_58, var_59, var_60]
    var_62 = module_0.Union(var_61)
    var_63 = module_0.Integer()
    var_64 = module_0.String()
    var_65 = module_0.Boolean()
    var_66 = [var_63, var_64, var_65]
    var_67 = module_0.Union(var_66)
    var_68 = module_0.Integer()
    var_69 = module_0.String()
    var_70 = module_0.Boolean()
    var_71 = [var_68, var_69, var_70]
    var_72 = module_0.Union(var_71)
    var_73 = module_0.Integer()
    var_74 = module_0.String()
    var_75 = module_0.Boolean()
    var_76 = [var_73, var_74, var_75]
    var_77 = module_0.Union(var_76)
    var_78 = module_0.Integer()
    var_79 = module_0.String()
    var_80 = module_0.Boolean()
    var_81 = [var_78, var_79, var_80]
    var_82 = module_0.Union(var_81)
    var_83 = module_0.Integer()
    var_84 = module_0.String()
    var_85 = module_0.Boolean()
    var_86 = [var_83, var_84, var_85]
    var_87 = module_0.Union(var_86)
    var_88 = module_0.Integer()
    var_89 = module_0.String()
    var_90 = module_0.Boolean()
    var_91 = [var_88, var_89, var_90]
    var_92 = module_0.Union(var_91)
    var_93 = module_0.Integer()
    var_94 = module_0.String()
    var_95 = module_0.Boolean()
    var_96 = [var_93, var_94, var_95]
    var_97 = module_0.Union(var_96)
    var_98 = module_0.Integer()
    var_99 = module_0.String()
    var_100 = module_0.Boolean()
    var_101 = [var_98, var_99, var_100]
    var_102 = module_0.Union(var_101)
    var_103 = module_0.Integer()
    var_104 = module_0.String()
    var_105 = module_0.Boolean()
    var_106 = [var_103, var_104, var_105]
    var_107 = module_0.Union(var_106)
    var_108 = module_0.Integer()
    var_109 = module_0.String()
    var_110 = module_0.Boolean()
    var_111 = [var_108, var_109, var_110]
    var_112 = module_0.Union(var_111)
    var_113 = module_0.Integer()
    var_114 = module_0.String()
    var_115 = module_0.Boolean()
    var_116 = [var_113, var_114, var_115]
    var_117 = module_0.Union(var_116)
    var_118 = module_0.Integer()
    var_119 = module_0.String()
    var_120 = module_0.Boolean()
    var_121 = [var_118, var_119, var_120]
    var_122 = module_0.Union(var_121)
    var_123 = module_0.Integer()
    var_124 = module_0.String()
    var_125 = module_0.Boolean()
    var_126 = [var_123, var_124, var_125]
    var_127 = module_0.Union(var_126)
    var_128 = module_0.Integer()
    var_129 = module_0.String()
    var_130 = module_0.Boolean()
    var_131 = [var_128, var_129, var_130]
    var_132 = module_0.Union(var_131)
    var_133 = module_0.Integer()
    var_134 = module_0.String()
    var_135 = module_0.Boolean()
    var_136 = [var_133, var_134, var_135]
    var_137 = module_0.Union(var_136)
    var_138 = module_0.Integer()
    var_139 = module_0.String()
    var_140 = module_0.Boolean()
    var_141 = [var_138, var_139, var_140]
    var_142 = module_0.Union(var_141)
    var_143 = module_0.Integer()
    var_144 = module_0.String()
    var_145 = module_0.Boolean()
    var_146 = [var_143, var_144, var_145]
    var_147 = module_0.Union(var_146)
    var_148 = module_0.Integer()
    var_149 = module_0.String()
    var_150 = module_0.Boolean()
    var_151 = [var_148, var_149, var_150]
    var_152 = module_0.Union(var_151)
    var_153 = module_0.Integer()
    var_154 = module_0.String()
    var_155 = module_0.Boolean()
    var_156 = [var_153, var_154, var_155]
    var_157 = module_0.Union(var_156)
    var_158 = module_0.Integer()
    var_159 = module_0.String()
    var_160 = module_0.Boolean()
    var_161 = [var_158, var_159, var_160]
    var_162 = module_0.Union(var_161)
    var_163 = module_0.Integer()
    var_164 = module_0.String()
    var_165 = module_0.Boolean()
    var_166 = [var_163, var_164, var_165]
    var_167 = module_0.Union(var_166)
    var_168 = module_0.Integer()
    var_169 = module_0.String()
    var_170 = module_0.Boolean()
    var_171 = [var_168, var_169, var_170]
    var_172 = module_0.Union(var_171)
    var_173 = module_0.Integer()
    var_174 = module_0.String()
    var_175 = module_0.Boolean()



# Parsed testcases at query #24
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Object()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None
    var_4 = False
    var_5 = module_0.Object()
    var_6 = None
    var_7 = var_5.validate(var_6)
    var_8 = module_0.Object()
    var_9 = 'not a dict'
    var_10 = var_8.validate(var_9)
    var_11 = 'key'
    var_12 = module_0.String()
    var_13 = {var_11: var_12}
    var_14 = module_0.Object(properties=var_13)
    var_15 = 'value'
    var_16 = {var_11: var_15}
    var_17 = var_14.validate(var_16)
    var_18 = module_0.String()
    var_19 = {var_11: var_18}
    var_20 = [var_11]
    var_21 = module_0.Object(properties=var_19, required=var_20)
    var_22 = {}
    var_23 = var_21.validate(var_22)
    var_24 = module_0.Object(additional_properties=var_4)
    var_25 = 'extra'
    var_26 = 'value'
    var_27 = {var_25: var_26}
    var_28 = var_24.validate(var_27)
    var_29 = module_0.String()
    var_30 = module_0.Object(additional_properties=var_29)
    var_31 = 'extra'
    var_32 = {var_31: var_15}
    var_33 = var_30.validate(var_32)
    var_34 = '^x-'
    var_35 = module_0.String()
    var_36 = {var_34: var_35}
    var_37 = module_0.Object(pattern_properties=var_36)
    var_38 = 'x-header'
    var_39 = {var_38: var_15}
    var_40 = var_37.validate(var_39)
    var_41 = module_0.Object(min_properties=var_25)
    var_42 = {}
    var_43 = var_41.validate(var_42)
    var_44 = module_0.Object(max_properties=var_42)
    var_45 = 'key1'
    var_46 = 'key2'
    var_47 = 'value1'
    var_48 = 'value2'
    var_49 = {var_45: var_47, var_46: var_48}
    var_50 = var_44.validate(var_49)
    var_51 = '^[a-z]+$'
    var_52 = module_0.String(pattern=var_51)
    var_53 = module_0.Object(property_names=var_52)
    var_54 = '123'
    var_55 = 'value'
    var_56 = {var_54: var_55}
    var_57 = var_53.validate(var_56)



# Parsed testcases at query #25
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None
    var_4 = False
    var_5 = module_0.Boolean()
    var_6 = None
    var_7 = var_5.validate(var_6)
    var_8 = module_0.Boolean()
    var_9 = var_8.validate(var_6)
    assert var_9 is True
    var_10 = module_0.Boolean(coerce_types=var_6)
    var_11 = 'true'
    var_12 = var_10.validate(var_11)
    assert var_12 is True
    var_13 = module_0.Boolean(coerce_types=var_4)
    var_14 = 'true'
    var_15 = var_13.validate(var_14)
    var_16 = module_0.Boolean(coerce_types=var_14)
    var_17 = 'null'
    var_18 = var_16.validate(var_17)
    assert var_18 is None
    var_19 = module_0.Boolean(coerce_types=var_14)
    var_20 = 'null'
    var_21 = var_19.validate(var_20)



# Parsed testcases at query #26
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Object()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None
    var_4 = False
    var_5 = module_0.Object()
    var_6 = None
    var_7 = var_5.validate(var_6)
    var_8 = module_0.Object()
    var_9 = 'not a dict'
    var_10 = var_8.validate(var_9)
    var_11 = 'name'
    var_12 = module_0.String()
    var_13 = {var_11: var_12}
    var_14 = [var_11]
    var_15 = module_0.Object(properties=var_13, required=var_14)
    var_16 = 'age'
    var_17 = 25
    var_18 = {var_16: var_17}
    var_19 = var_15.validate(var_18)
    var_20 = module_0.String()
    var_21 = {var_11: var_20}
    var_22 = [var_11]
    var_23 = module_0.Object(properties=var_21, required=var_22)
    var_24 = 'John'
    var_25 = {var_11: var_24}
    var_26 = var_23.validate(var_25)
    var_27 = module_0.String()
    var_28 = {var_11: var_27}
    var_29 = module_0.Object(properties=var_28, additional_properties=var_19)
    var_30 = 'name'
    var_31 = 'age'
    var_32 = 'John'
    var_33 = 25
    var_34 = {var_30: var_32, var_31: var_33}
    var_35 = var_29.validate(var_34)
    var_36 = module_0.String()
    var_37 = {var_34: var_36}
    var_38 = module_0.Object(properties=var_37, additional_properties=var_30)
    var_39 = 'age'
    var_40 = 25
    var_41 = {var_34: var_24, var_39: var_40}
    var_42 = var_38.validate(var_41)
    var_43 = module_0.String()
    var_44 = {var_34: var_43}
    var_45 = module_0.Integer()
    var_46 = module_0.Object(properties=var_44, additional_properties=var_45)
    var_47 = {var_34: var_24, var_39: var_40}
    var_48 = var_46.validate(var_47)
    var_49 = module_0.String()
    var_50 = {var_34: var_49}
    var_51 = module_0.Integer()
    var_52 = module_0.Object(properties=var_50, additional_properties=var_51)
    var_53 = 'name'
    var_54 = 'age'
    var_55 = 'John'
    var_56 = 'twenty-five'
    var_57 = {var_53: var_55, var_54: var_56}
    var_58 = var_52.validate(var_57)
    var_59 = '^[a-z]+$'
    var_60 = module_0.Integer()
    var_61 = {var_59: var_60}
    var_62 = module_0.Object(pattern_properties=var_61)
    var_63 = {var_39: var_40}
    var_64 = var_62.validate(var_63)
    var_65 = 'Age'
    var_66 = 25
    var_67 = {var_65: var_66}
    var_68 = var_62.validate(var_67)
    var_69 = module_0.String(pattern=var_59)
    var_70 = module_0.Object(property_names=var_69)
    var_71 = {var_39: var_40}
    var_72 = var_70.validate(var_71)
    var_73 = 'Age'
    var_74 = 25
    var_75 = {var_73: var_74}
    var_76 = var_70.validate(var_75)
    var_77 = module_0.Object(min_properties=var_73)
    var_78 = {}
    var_79 = var_77.validate(var_78)
    var_80 = module_0.Object(max_properties=var_78)
    var_81 = 'a'
    var_82 = 'b'
    var_83 = 1
    var_84 = 2
    var_85 = {var_81: var_83, var_82: var_84}
    var_86 = var_80.validate(var_85)



# Parsed testcases at query #27
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'A'
    var_1 = 'B'
    var_2 = 'C'
    var_3 = (var_0, var_1, var_2)
    var_4 = module_0.Choice(choices=var_3)
    var_5 = var_4.validate(var_0)
    assert var_5 == 'A'
    var_6 = var_4.validate(var_1)
    assert var_6 == 'B'
    var_7 = var_4.validate(var_2)
    assert var_7 == 'C'
    var_8 = 'D'
    var_9 = var_4.validate(var_8)
    var_10 = None
    var_11 = var_4.validate(var_10)
    var_12 = (var_10, var_11, var_2)
    var_13 = True
    var_14 = module_0.Choice(choices=var_12)
    var_15 = None
    var_16 = var_14.validate(var_15)
    assert var_16 is None
    var_17 = '1'
    var_18 = (var_17, var_10)
    var_19 = '2'
    var_20 = (var_19, var_11)
    var_21 = '3'
    var_22 = (var_21, var_2)
    var_23 = (var_18, var_20, var_22)
    var_24 = module_0.Choice(choices=var_23)
    var_25 = var_24.validate(var_17)
    assert var_25 == '1'
    var_26 = var_24.validate(var_19)
    assert var_26 == '2'
    var_27 = var_24.validate(var_21)
    assert var_27 == '3'
    var_28 = '4'
    var_29 = var_24.validate(var_28)
    var_30 = None
    var_31 = var_24.validate(var_30)
    var_32 = (var_17, var_30)
    var_33 = (var_19, var_31)
    var_34 = (var_21, var_2)
    var_35 = (var_32, var_33, var_34)
    var_36 = module_0.Choice(choices=var_35)
    var_37 = var_36.validate(var_15)
    assert var_37 is None



# Parsed testcases at query #28
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = 'test'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'test'
    var_6 = 123
    var_7 = var_3.validate(var_6)
    assert var_7 == 123
    var_8 = True
    var_9 = var_3.validate(var_8)
    var_10 = None
    var_11 = var_3.validate(var_10)
    assert var_11 is None
    var_12 = None
    var_13 = var_3.validate(var_12)



# Parsed testcases at query #29
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Const(var_0)
    var_2 = None
    var_3 = module_0.Const(var_2)
    var_4 = 'test'
    var_5 = True
    var_6 = module_0.Const(var_4)



# Parsed testcases at query #30
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_0.Integer()
    var_4 = 1
    var_5 = 3
    var_6 = module_0.Array(var_2, var_3, var_4, var_5)
    var_7 = 'test'
    var_8 = [var_4, var_7, var_5]
    var_9 = var_6.validate(var_8)
    var_10 = [var_4, var_7]
    var_11 = var_6.validate(var_10)
    var_12 = [var_4]
    var_13 = var_6.validate(var_12)
    var_14 = []
    var_15 = var_6.validate(var_14)
    var_16 = 1
    var_17 = 'test'
    var_18 = 3
    var_19 = 4
    var_20 = [var_16, var_17, var_18, var_19]
    var_21 = var_6.validate(var_20)
    var_22 = 1
    var_23 = 'test'
    var_24 = [var_22, var_23, var_23]
    var_25 = var_6.validate(var_24)
    var_26 = 1
    var_27 = 'test'
    var_28 = 3
    var_29 = [var_26, var_27, var_28, var_27]
    var_30 = var_6.validate(var_29)
    var_31 = 1
    var_32 = 'test'
    var_33 = 3
    var_34 = 4
    var_35 = 5
    var_36 = [var_31, var_32, var_33, var_34, var_35]
    var_37 = var_6.validate(var_36)
    var_38 = 1
    var_39 = 'test'
    var_40 = 3
    var_41 = 4
    var_42 = 5
    var_43 = 6
    var_44 = [var_38, var_39, var_40, var_41, var_42, var_43]
    var_45 = var_6.validate(var_44)
    var_46 = 1
    var_47 = 'test'
    var_48 = 3
    var_49 = 4
    var_50 = 5
    var_51 = 6
    var_52 = 7
    var_53 = [var_46, var_47, var_48, var_49, var_50, var_51, var_52]
    var_54 = var_6.validate(var_53)
    var_55 = 1
    var_56 = 'test'
    var_57 = 3
    var_58 = 4
    var_59 = 5
    var_60 = 6
    var_61 = 7
    var_62 = 8
    var_63 = [var_55, var_56, var_57, var_58, var_59, var_60, var_61, var_62]
    var_64 = var_6.validate(var_63)
    var_65 = 1
    var_66 = 'test'
    var_67 = 3
    var_68 = 4
    var_69 = 5
    var_70 = 6
    var_71 = 7
    var_72 = 8
    var_73 = 9
    var_74 = [var_65, var_66, var_67, var_68, var_69, var_70, var_71, var_72, var_73]
    var_75 = var_6.validate(var_74)
    var_76 = 1
    var_77 = 'test'
    var_78 = 3
    var_79 = 4
    var_80 = 5
    var_81 = 6
    var_82 = 7
    var_83 = 8
    var_84 = 9
    var_85 = 10
    var_86 = [var_76, var_77, var_78, var_79, var_80, var_81, var_82, var_83, var_84, var_85]
    var_87 = var_6.validate(var_86)
    var_88 = 1
    var_89 = 'test'
    var_90 = 3
    var_91 = 4
    var_92 = 5
    var_93 = 6
    var_94 = 7
    var_95 = 8
    var_96 = 9
    var_97 = 10
    var_98 = 11
    var_99 = [var_88, var_89, var_90, var_91, var_92, var_93, var_94, var_95, var_96, var_97, var_98]
    var_100 = var_6.validate(var_99)
    var_101 = 1
    var_102 = 'test'
    var_103 = 3
    var_104 = 4
    var_105 = 5
    var_106 = 6
    var_107 = 7
    var_108 = 8
    var_109 = 9
    var_110 = 10
    var_111 = 11
    var_112 = 12
    var_113 = [var_101, var_102, var_103, var_104, var_105, var_106, var_107, var_108, var_109, var_110, var_111, var_112]
    var_114 = var_6.validate(var_113)
    var_115 = 1
    var_116 = 'test'
    var_117 = 3
    var_118 = 4
    var_119 = 5
    var_120 = 6
    var_121 = 7
    var_122 = 8
    var_123 = 9
    var_124 = 10
    var_125 = 11
    var_126 = 12
    var_127 = 13
    var_128 = [var_115, var_116, var_117, var_118, var_119, var_120, var_121, var_122, var_123, var_124, var_125, var_126, var_127]
    var_129 = var_6.validate(var_128)
    var_130 = 1
    var_131 = 'test'
    var_132 = 3
    var_133 = 4
    var_134 = 5
    var_135 = 6
    var_136 = 7
    var_137 = 8
    var_138 = 9
    var_139 = 10
    var_140 = 11
    var_141 = 12
    var_142 = 13
    var_143 = 14
    var_144 = [var_130, var_131, var_132, var_133, var_134, var_135, var_136, var_137, var_138, var_139, var_140, var_141, var_142, var_143]
    var_145 = var_6.validate(var_144)
    var_146 = 1
    var_147 = 'test'
    var_148 = 3
    var_149 = 4
    var_150 = 5
    var_151 = 6
    var_152 = 7
    var_153 = 8
    var_154 = 9
    var_155 = 10
    var_156 = 11
    var_157 = 12
    var_158 = 13
    var_159 = 14
    var_160 = 15
    var_161 = [var_146, var_147, var_148, var_149, var_150, var_151, var_152, var_153, var_154, var_155, var_156, var_157, var_158, var_159, var_160]
    var_162 = var_6.validate(var_161)
    var_163 = 1
    var_164 = 'test'
    var_165 = 3
    var_166 = 4
    var_167 = 5
    var_168 = 6
    var_169 = 7
    var_170 = 8
    var_171 = 9
    var_172 = 10
    var_173 = 11
    var_174 = 12
    var_175 = 13
    var_176 = 14
    var_177 = 15
    var_178 = 16
    var_179 = [var_163, var_164, var_165, var_166, var_167, var_168, var_169, var_170, var_171, var_172, var_173, var_174, var_175, var_176, var_177, var_178]
    var_180 = var_6.validate(var_179)
    var_181 = 1
    var_182 = 'test'
    var_183 = 3
    var_184 = 4
    var_185 = 5
    var_186 = 6
    var_187 = 7
    var_188 = 8
    var_189 = 9
    var_190 = 10
    var_191 = 11
    var_192 = 12
    var_193 = 13
    var_194 = 14
    var_195 = 15
    var_196 = 16
    var_197 = 17
    var_198 = [var_181, var_182, var_183, var_184, var_185, var_186, var_187, var_188, var_189, var_190, var_191, var_192, var_193, var_194, var_195, var_196, var_197]
    var_199 = var_6.validate(var_198)
    var_200 = 1
    var_201 = 'test'
    var_202 = 3
    var_203 = 4
    var_204 = 5
    var_205 = 6
    var_206 = 7
    var_207 = 8
    var_208 = 9
    var_209 = 10
    var_210 = 11
    var_211 = 12
    var_212 = 13
    var_213 = 14
    var_214 = 15
    var_215 = 16
    var_216 = 17
    var_217 = 18
    var_218 = [var_200, var_201, var_202, var_203, var_204, var_205, var_206, var_207, var_208, var_209, var_210, var_211, var_212, var_213, var_214, var_215, var_216, var_217]
    var_219 = var_6.validate(var_218)
    var_220 = 1
    var_221 = 'test'
    var_222 = 3
    var_223 = 4
    var_224 = 5
    var_225 = 6
    var_226 = 7
    var_227 = 8
    var_228 = 9
    var_229 = 10
    var_230 = 11
    var_231 = 12
    var_232 = 13
    var_233 = 14
    var_234 = 15
    var_235 = 16
    var_236 = 17
    var_237 = 18
    var_238 = 19
    var_239 = [var_220, var_221, var_222, var_223, var_224, var_225, var_226, var_227, var_228, var_229, var_230, var_231, var_232, var_233, var_234, var_235, var_236, var_237, var_238]
    var_240 = var_6.validate(var_239)
    var_241 = 1
    var_242 = 'test'
    var_243 = 3
    var_244 = 4
    var_245 = 5
    var_246 = 6
    var_247 = 7
    var_248 = 8
    var_249 = 9
    var_250 = 10
    var_251 = 11
    var_252 = 12
    var_253 = 13
    var_254 = 14
    var_255 = 15
    var_256 = 16
    var_257 = 17
    var_258 = 18
    var_259 = 19
    var_260 = 20
    var_261 = [var_241, var_242, var_243, var_244, var_245, var_246, var_247, var_248, var_249, var_250, var_251, var_252, var_253, var_254, var_255, var_256, var_257, var_258, var_259, var_260]
    var_262 = var_6.validate(var_261)
    var_263 = 1
    var_264 = 'test'
    var_265 = 3
    var_266 = 4
    var_267 = 5
    var_268 = 6
    var_269 = 7
    var_270 = 8
    var_271 = 9
    var_272 = 10
    var_273 = 11
    var_274 = 12
    var_275 = 13
    var_276 = 14
    var_277 = 15
    var_278 = 16
    var_279 = 17
    var_280 = 18
    var_281 = 19
    var_282 = 20
    var_283 = 21
    var_284 = [var_263, var_264, var_265, var_266, var_267, var_268, var_269, var_270, var_271, var_272, var_273, var_274, var_275, var_276, var_277, var_278, var_279, var_280, var_281, var_282, var_283]
    var_285 = var_6.validate(var_284)
    var_286 = 1
    var_287 = 'test'
    var_288 = 3
    var_289 = 4
    var_290 = 5
    var_291 = 6
    var_292 = 7
    var_293 = 8
    var_294 = 9
    var_295 = 10
    var_296 = 11
    var_297 = 12
    var_298 = 13
    var_299 = 14
    var_300 = 15
    var_301 = 16
    var_302 = 17
    var_303 = 18
    var_304 = 19
    var_305 = 20
    var_306 = 21
    var_307 = 22
    var_308 = [var_286, var_287, var_288, var_289, var_290, var_291, var_292, var_293, var_294, var_295, var_296, var_297, var_298, var_299, var_300, var_301, var_302, var_303, var_304, var_305, var_306, var_307]
    var_309 = var_6.validate(var_308)
    var_310 = 1
    var_311 = 'test'
    var_312 = 3
    var_313 = 4
    var_314 = 5
    var_315 = 6
    var_316 = 7
    var_317 = 8
    var_318 = 9
    var_319 = 10
    var_320 = 11
    var_321 = 12
    var_322 = 13
    var_323 = 14
    var_324 = 15
    var_325 = 16
    var_326 = 17
    var_327 = 18
    var_328 = 19
    var_329 = 20
    var_330 = 21
    var_331 = 22
    var_332 = 23
    var_333 = [var_310, var_311, var_312, var_313, var_314, var_315, var_316, var_317, var_318, var_319, var_320, var_321, var_322, var_323, var_324, var_325, var_326, var_327, var_328, var_329, var_330, var_331, var_332]
    var_334 = var_6.validate(var_333)
    var_335 = 1
    var_336 = 'test'
    var_337 = 3
    var_338 = 4
    var_339 = 5
    var_340 = 6
    var_341 = 7
    var_342 = 8
    var_343 = 9
    var_344 = 10
    var_345 = 11
    var_346 = 12
    var_347 = 13
    var_348 = 14
    var_349 = 15
    var_350 = 16
    var_351 = 17
    var_352 = 18
    var_353 = 19
    var_354 = 20
    var_355 = 21
    var_356 = 22
    var_357 = 23
    var_358 = 24
    var_359 = [var_335, var_336, var_337, var_338, var_339, var_340, var_341, var_342, var_343, var_344, var_345, var_346, var_347, var_348, var_349, var_350, var_351, var_352, var_353, var_354, var_355, var_356, var_357, var_358]
    var_360 = var_6.validate(var_359)
    var_361 = 1
    var_362 = 'test'
    var_363 = 3
    var_364 = 4
    var_365 = 5
    var_366 = 6
    var_367 = 7
    var_368 = 8
    var_369 = 9
    var_370 = 10
    var_371 = 11
    var_372 = 12
    var_373 = 13
    var_374 = 14
    var_375 = 15
    var_376 = 16
    var_377 = 17
    var_378 = 18
    var_379 = 19
    var_380 = 20
    var_381 = 21
    var_382 = 22
    var_383 = 23
    var_384 = 24
    var_385 = 25
    var_386 = [var_361, var_362, var_363, var_364, var_365, var_366, var_367, var_368, var_369, var_370, var_371, var_372, var_373, var_374, var_375, var_376, var_377, var_378, var_379, var_380, var_381, var_382, var_383, var_384, var_385]
    var_387 = var_6.validate(var_386)
    var_388 = 1
    var_389 = 'test'
    var_390 = 3
    var_391 = 4
    var_392 = 5
    var_393 = 6
    var_394 = 7
    var_395 = 8
    var_396 = 9
    var_397 = 10
    var_398 = 11
    var_399 = 12
    var_400 = 13
    var_401 = 14
    var_402 = 15
    var_403 = 16
    var_404 = 17
    var_405 = 18
    var_406 = 19
    var_407 = 20
    var_408 = 21
    var_409 = 22
    var_410 = 23
    var_411 = 24
    var_412 = 25
    var_413 = 26
    var_414 = [var_388, var_389, var_390, var_391, var_392, var_393, var_394, var_395, var_396, var_397, var_398, var_399, var_400, var_401, var_402, var_403, var_404, var_405, var_406, var_407, var_408, var_409, var_410, var_411, var_412, var_413]
    var_415 = var_6.validate(var_414)
    var_416 = 1
    var_417 = 'test'
    var_418 = 3
    var_419 = 4
    var_420 = 5
    var_421 = 6
    var_422 = 7
    var_423 = 8
    var_424 = 9
    var_425 = 10
    var_426 = 11
    var_427 = 12
    var_428 = 13
    var_429 = 14
    var_430 = 15
    var_431 = 16
    var_432 = 17
    var_433 = 18
    var_434 = 19
    var_435 = 20
    var_436 = 21
    var_437 = 22
    var_438 = 23
    var_439 = 24
    var_440 = 25
    var_441 = 26
    var_442 = 27
    var_443 = [var_416, var_417, var_418, var_419, var_420, var_421, var_422, var_423, var_424, var_425, var_426, var_427, var_428, var_429, var_430, var_431, var_432, var_433, var_434, var_435, var_436, var_437, var_438, var_439, var_440, var_441, var_442]
    var_444 = var_6.validate(var_443)
    var_445 = 1
    var_446 = 'test'
    var_447 = 3
    var_448 = 4
    var_449 = 5
    var_450 = 6
    var_451 = 7
    var_452 = 8
    var_453 = 9
    var_454 = 10
    var_455 = 11
    var_456 = 12
    var_457 = 13
    var_458 = 14
    var_459 = 15
    var_460 = 16
    var_461 = 17
    var_462 = 18
    var_463 = 19
    var_464 = 20
    var_465 = 21
    var_466 = 22
    var_467 = 23
    var_468 = 24
    var_469 = 25
    var_470 = 26
    var_471 = 27
    var_472 = 28
    var_473 = [var_445, var_446, var_447, var_448, var_449, var_450, var_451, var_452, var_453, var_454, var_455, var_456, var_457, var_458, var_459, var_460, var_461, var_462, var_463, var_464, var_465, var_466, var_467, var_468, var_469, var_470, var_471, var_472]
    var_474 = var_6.validate(var_473)
    var_475 = 1
    var_476 = 'test'
    var_477 = 3
    var_478 = 4
    var_479 = 5
    var_480 = 6
    var_481 = 7
    var_482 = 8
    var_483 = 9
    var_484 = 10
    var_485 = 11
    var_486 = 12
    var_487 = 13
    var_488 = 14
    var_489 = 15
    var_490 = 16
    var_491 = 17
    var_492 = 18
    var_493 = 19
    var_494 = 20
    var_495 = 21
    var_496 = 22
    var_497 = 23
    var_498 = 24
    var_499 = 25
    var_500 = 26
    var_501 = 27
    var_502 = 28
    var_503 = 29
    var_504 = [var_475, var_476, var_477, var_478, var_479, var_480, var_481, var_482, var_483, var_484, var_485, var_486, var_487, var_488, var_489, var_490, var_491, var_492, var_493, var_494, var_495, var_496, var_497, var_498, var_499, var_500, var_501, var_502, var_503]
    var_505 = var_6.validate(var_504)
    var_506 = 1
    var_507 = 'test'
    var_508 = 3
    var_509 = 4
    var_510 = 5
    var_511 = 6
    var_512 = 7
    var_513 = 8
    var_514 = 9
    var_515 = 10
    var_516 = 11
    var_517 = 12
    var_518 = 13
    var_519 = 14
    var_520 = 15
    var_521 = 16
    var_522 = 17
    var_523 = 18
    var_524 = 19
    var_525 = 20
    var_526 = 21
    var_527 = 22
    var_528 = 23
    var_529 = 24
    var_530 = 25
    var_531 = 26
    var_532 = 27
    var_533 = 28
    var_534 = 29
    var_535 = 30
    var_536 = [var_506, var_507, var_508, var_509, var_510, var_511, var_512, var_513, var_514, var_515, var_516, var_517, var_518, var_519, var_520, var_521, var_522, var_523, var_524, var_525, var_526, var_527, var_528, var_529, var_530, var_531, var_532, var_533, var_534, var_535]
    var_537 = var_6.validate(var_536)



