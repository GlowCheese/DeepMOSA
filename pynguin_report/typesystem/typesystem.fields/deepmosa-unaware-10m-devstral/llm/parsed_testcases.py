####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = module_0.Field(default=var_0)
    var_2 = var_1.get_default_value()
    assert var_2 == 'test_value'
    var_3 = 'callable_value'
    var_4 = lambda : var_3
    var_5 = module_0.Field(default=var_4)
    var_6 = var_5.get_default_value()
    assert var_6 == 'callable_value'
    var_7 = module_0.Field()
    var_8 = var_7.get_default_value()
    assert var_8 is None
    var_9 = None
    var_10 = module_0.Field(default=var_9)
    var_11 = var_10.get_default_value()
    assert var_11 is None



# Parsed testcases at query #2
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'Option A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'Option B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = module_0.Choice(choices=var_6)
    var_8 = var_7.validate(var_0)
    assert var_8 == 'a'
    var_9 = var_7.validate(var_3)
    assert var_9 == 'b'
    var_10 = (var_0, var_1)
    var_11 = [var_10]
    var_12 = True
    var_13 = module_0.Choice(choices=var_11)
    var_14 = None
    var_15 = var_13.validate(var_14)
    assert var_15 is None
    var_16 = 'c'
    var_17 = var_7.validate(var_16)
    var_18 = ''
    var_19 = var_7.validate(var_18)
    var_20 = (var_18, var_19)
    var_21 = [var_20]
    var_22 = module_0.Choice(choices=var_21, coerce_types=var_12)
    var_23 = ''
    var_24 = var_22.validate(var_23)
    assert var_24 is None
    var_25 = (var_18, var_19)
    var_26 = [var_25]
    var_27 = False
    var_28 = module_0.Choice(choices=var_26, coerce_types=var_27)
    var_29 = 'c'
    var_30 = var_28.validate(var_29)
    var_31 = None
    var_32 = var_7.validate(var_31)
    var_33 = (var_31, var_32)
    var_34 = (var_3, var_4)
    var_35 = [var_33, var_34]
    var_36 = module_0.Choice(choices=var_35)
    var_37 = var_36.validate(var_31)
    assert var_37 == 'a'
    var_38 = var_36.validate(var_3)
    assert var_38 == 'b'
    var_39 = [var_31, var_32]
    var_40 = [var_3, var_4]
    var_41 = [var_39, var_40]
    var_42 = module_0.Choice(choices=var_41)
    var_43 = var_42.validate(var_31)
    assert var_43 == 'a'
    var_44 = var_42.validate(var_3)
    assert var_44 == 'b'



# Parsed testcases at query #3
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
    var_21 = module_0.Boolean(coerce_types=var_3)
    var_22 = 'true'
    var_23 = var_21.validate(var_22)
    var_24 = 1
    var_25 = var_21.validate(var_24)
    var_26 = module_0.Boolean()
    var_27 = None
    var_28 = var_26.validate(var_27)
    assert var_28 is None
    var_29 = 'null'
    var_30 = var_26.validate(var_29)
    assert var_30 is None
    var_31 = 'none'
    var_32 = var_26.validate(var_31)
    assert var_32 is None
    var_33 = None
    var_34 = var_0.validate(var_33)
    var_35 = 'invalid'
    var_36 = var_0.validate(var_35)
    var_37 = 2
    var_38 = var_0.validate(var_37)



# Parsed testcases at query #4
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
    var_7 = module_0.String(allow_blank=var_3)
    var_8 = ''
    var_9 = var_7.validate(var_8)
    assert var_9 == ''
    var_10 = module_0.String(trim_whitespace=var_3)
    var_11 = '  hello  '
    var_12 = var_10.validate(var_11)
    assert var_12 == 'hello'
    var_13 = 3
    var_14 = module_0.String(min_length=var_13)
    var_15 = var_14.validate(var_1)
    assert var_15 == 'hello'
    var_16 = 'hi'
    var_17 = var_14.validate(var_16)
    var_18 = 5
    var_19 = module_0.String(max_length=var_18)
    var_20 = var_19.validate(var_16)
    assert var_20 == 'hello'
    var_21 = 'hello world'
    var_22 = var_19.validate(var_21)
    var_23 = '^[a-z]+$'
    var_24 = module_0.String(pattern=var_23)
    var_25 = var_24.validate(var_21)
    assert var_25 == 'hello'
    var_26 = 'Hello'
    var_27 = var_24.validate(var_26)
    var_28 = 'email'
    var_29 = module_0.String(format=var_28)
    var_30 = 'test@example.com'
    var_31 = var_29.validate(var_30)
    assert var_31 == 'test@example.com'
    var_32 = 'not an email'
    var_33 = var_29.validate(var_32)
    var_34 = module_0.String()
    var_35 = 'hello\x00world'
    var_36 = var_34.validate(var_35)
    assert var_36 == 'helloworld'
    var_37 = module_0.String()
    var_38 = 123
    var_39 = var_37.validate(var_38)
    var_40 = module_0.String()
    var_41 = None
    var_42 = var_40.validate(var_41)
    var_43 = module_0.String()
    var_44 = ''
    var_45 = var_43.validate(var_44)
    var_46 = module_0.String(coerce_types=var_3)
    var_47 = var_46.validate(var_8)
    assert var_47 is None
    var_48 = module_0.String(allow_blank=var_3, coerce_types=var_3)
    var_49 = var_48.validate(var_5)
    assert var_49 == ''



# Parsed testcases at query #5
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
    var_14 = 3
    var_15 = module_0.Array(exact_items=var_14)
    var_16 = 2
    var_17 = [var_12, var_16, var_14]
    var_18 = var_15.validate(var_17)
    var_19 = 1
    var_20 = 2
    var_21 = [var_19, var_20]
    var_22 = var_15.validate(var_21)
    var_23 = module_0.Array(min_items=var_16)
    var_24 = [var_19, var_16, var_14]
    var_25 = var_23.validate(var_24)
    var_26 = 1
    var_27 = [var_26]
    var_28 = var_23.validate(var_27)
    var_29 = module_0.Array(max_items=var_16)
    var_30 = [var_26, var_16]
    var_31 = var_29.validate(var_30)
    var_32 = 1
    var_33 = 2
    var_34 = 3
    var_35 = [var_32, var_33, var_34]
    var_36 = var_29.validate(var_35)
    var_37 = module_0.Array(unique_items=var_32)
    var_38 = [var_32, var_16, var_36]
    var_39 = var_37.validate(var_38)
    var_40 = 1
    var_41 = 2
    var_42 = [var_40, var_41, var_41]
    var_43 = var_37.validate(var_42)
    var_44 = module_0.Integer()
    var_45 = module_0.Array(var_44)
    var_46 = '1'
    var_47 = '2'
    var_48 = '3'
    var_49 = [var_46, var_47, var_48]
    var_50 = var_45.validate(var_49)
    var_51 = '1'
    var_52 = 'two'
    var_53 = '3'
    var_54 = [var_51, var_52, var_53]
    var_55 = var_45.validate(var_54)
    var_56 = module_0.Integer()
    var_57 = module_0.String()
    var_58 = module_0.Boolean()
    var_59 = [var_56, var_57, var_58]
    var_60 = module_0.Array(var_59)
    var_61 = 'two'
    var_62 = 'true'
    var_63 = [var_46, var_61, var_62]
    var_64 = var_60.validate(var_63)
    var_65 = '1'
    var_66 = 'two'
    var_67 = 'not a bool'
    var_68 = [var_65, var_66, var_67]
    var_69 = var_60.validate(var_68)
    var_70 = module_0.Integer()
    var_71 = module_0.String()
    var_72 = [var_70, var_71]
    var_73 = module_0.Array(var_72, var_68)
    var_74 = [var_65, var_61]
    var_75 = var_73.validate(var_74)
    var_76 = 1
    var_77 = 'two'
    var_78 = 3
    var_79 = [var_76, var_77, var_78]
    var_80 = var_73.validate(var_79)
    var_81 = module_0.Integer()
    var_82 = module_0.String()
    var_83 = [var_81, var_82]
    var_84 = module_0.Boolean()
    var_85 = module_0.Array(var_83, var_84)
    var_86 = [var_76, var_61, var_76]
    var_87 = var_85.validate(var_86)
    var_88 = 1
    var_89 = 'two'
    var_90 = 'not a bool'
    var_91 = [var_88, var_89, var_90]
    var_92 = var_85.validate(var_91)
    var_93 = module_0.Integer()
    var_94 = module_0.Array(var_93)
    var_95 = module_0.Array(var_94)
    var_96 = [var_46, var_47]
    var_97 = '4'
    var_98 = [var_48, var_97]
    var_99 = [var_96, var_98]
    var_100 = var_95.validate(var_99)
    var_101 = '1'
    var_102 = 'two'
    var_103 = [var_101, var_102]
    var_104 = '3'
    var_105 = '4'
    var_106 = [var_104, var_105]
    var_107 = [var_103, var_106]
    var_108 = var_95.validate(var_107)



# Parsed testcases at query #6
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = module_0.Const(var_0)
    var_2 = None
    var_3 = module_0.Const(var_2)
    var_4 = 42
    var_5 = module_0.Const(var_4)
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]
    var_10 = module_0.Const(var_9)
    var_11 = 'test'
    var_12 = True
    var_13 = module_0.Const(var_11)



# Parsed testcases at query #7
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
    var_15 = 'not an array'
    var_16 = var_14.validate(var_15)
    var_17 = module_0.Array(min_items=var_16)
    var_18 = [var_6, var_16]
    var_19 = var_17.validate(var_18)
    var_20 = 1
    var_21 = [var_20]
    var_22 = var_17.validate(var_21)
    var_23 = module_0.Array(max_items=var_21)
    var_24 = [var_6, var_21]
    var_25 = var_23.validate(var_24)
    var_26 = 1
    var_27 = 2
    var_28 = 3
    var_29 = [var_26, var_27, var_28]
    var_30 = var_23.validate(var_29)
    var_31 = module_0.Array(exact_items=var_27)
    var_32 = [var_6, var_27]
    var_33 = var_31.validate(var_32)
    var_34 = 1
    var_35 = [var_34]
    var_36 = var_31.validate(var_35)
    var_37 = 1
    var_38 = 2
    var_39 = 3
    var_40 = [var_37, var_38, var_39]
    var_41 = var_31.validate(var_40)
    var_42 = True
    var_43 = module_0.Array(unique_items=var_42)
    var_44 = [var_42, var_38, var_39]
    var_45 = var_43.validate(var_44)
    var_46 = 1
    var_47 = 2
    var_48 = [var_46, var_47, var_47]
    var_49 = var_43.validate(var_48)
    var_50 = module_0.Integer()
    var_51 = module_0.Array(var_50)
    var_52 = [var_42, var_47, var_48]
    var_53 = var_51.validate(var_52)
    var_54 = 1
    var_55 = 'not an integer'
    var_56 = 3
    var_57 = [var_54, var_55, var_56]
    var_58 = var_51.validate(var_57)
    var_59 = module_0.Integer()
    var_60 = module_0.String()
    var_61 = [var_59, var_60]
    var_62 = module_0.Array(var_61)
    var_63 = 'two'
    var_64 = [var_42, var_63]
    var_65 = var_62.validate(var_64)
    var_66 = 1
    var_67 = 2
    var_68 = [var_66, var_67]
    var_69 = var_62.validate(var_68)
    var_70 = 1
    var_71 = 'two'
    var_72 = 'extra'
    var_73 = [var_70, var_71, var_72]
    var_74 = var_62.validate(var_73)
    var_75 = module_0.Integer()
    var_76 = module_0.String()
    var_77 = [var_75, var_76]
    var_78 = module_0.Array(var_77, var_10)
    var_79 = [var_42, var_63]
    var_80 = var_78.validate(var_79)
    var_81 = 1
    var_82 = 'two'
    var_83 = 'extra'
    var_84 = [var_81, var_82, var_83]
    var_85 = var_78.validate(var_84)
    var_86 = module_0.Integer()
    var_87 = module_0.String()
    var_88 = [var_86, var_87]
    var_89 = module_0.String()
    var_90 = module_0.Array(var_88, var_89)
    var_91 = 'extra'
    var_92 = [var_42, var_63, var_91]
    var_93 = var_90.validate(var_92)
    var_94 = 1
    var_95 = 'two'
    var_96 = 3
    var_97 = [var_94, var_95, var_96]
    var_98 = var_90.validate(var_97)
    var_99 = module_0.Array(min_items=var_42)
    var_100 = []
    var_101 = var_99.validate(var_100)
    var_102 = module_0.Array()
    var_103 = []
    var_104 = var_102.validate(var_103)



# Parsed testcases at query #8
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Array()
    var_1 = module_0.Field()
    var_2 = module_0.Array(var_1)
    var_3 = module_0.Field()
    var_4 = module_0.Field()
    var_5 = [var_3, var_4]
    var_6 = module_0.Array(var_5)
    var_7 = module_0.Field()
    var_8 = module_0.Array(additional_items=var_7)
    var_9 = True
    var_10 = module_0.Array(additional_items=var_9)
    var_11 = 10
    var_12 = module_0.Array(min_items=var_9, max_items=var_11)
    var_13 = 5
    var_14 = module_0.Array(exact_items=var_13)
    var_15 = module_0.Array(unique_items=var_9)
    var_16 = module_0.Field()
    var_17 = module_0.Field()
    var_18 = module_0.Field()
    var_19 = [var_17, var_18]
    var_20 = module_0.Array(var_19, var_16, var_9, var_11, unique_items=var_9)



# Parsed testcases at query #9
#--------------------------


import typesystem.fields as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'Test Title'
    var_2 = 'Test Description'
    var_3 = 'default_value'
    var_4 = True
    var_5 = False
    var_6 = 100
    var_7 = 10
    var_8 = '^[a-z]+$'
    var_9 = 'email'
    var_10 = module_0.String(allow_blank=var_4, trim_whitespace=var_5, max_length=var_6, min_length=var_7, pattern=var_8, format=var_9, coerce_types=var_5)
    var_11 = var_10.pattern_regex
    var_12 = module_1.compile(var_8)
    var_13 = module_0.String(pattern=var_12)
    var_14 = module_0.String(allow_blank=var_4)
    var_15 = module_0.String()
    var_16 = 'invalid'
    var_17 = module_0.String(max_length=var_16)
    var_18 = 'invalid'
    var_19 = module_0.String(min_length=var_18)
    var_20 = 123
    var_21 = module_0.String(pattern=var_20)
    var_22 = 123
    var_23 = module_0.String(format=var_22)



# Parsed testcases at query #10
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
    var_9 = module_0.String()
    var_10 = module_0.Integer()
    var_11 = [var_9, var_10]
    var_12 = module_0.Union(var_11)
    var_13 = None
    var_14 = var_12.validate(var_13)
    assert var_14 is None
    var_15 = False
    var_16 = module_0.String()
    var_17 = module_0.Integer()
    var_18 = [var_16, var_17]
    var_19 = module_0.Union(var_18)
    var_20 = None
    var_21 = var_19.validate(var_20)
    var_22 = module_0.String()
    var_23 = module_0.Integer()
    var_24 = [var_22, var_23]
    var_25 = module_0.Union(var_24)
    var_26 = []
    var_27 = var_25.validate(var_26)
    var_28 = 5
    var_29 = module_0.String(min_length=var_28)
    var_30 = module_0.Integer()
    var_31 = [var_29, var_30]
    var_32 = module_0.Union(var_31)
    var_33 = 'abc'
    var_34 = var_32.validate(var_33)
    var_35 = exc_info.value.messages()[var_15]
    var_36 = var_35.code
    assert var_36 == 'min_length'
    var_37 = module_0.String(min_length=var_28)
    var_38 = 10
    var_39 = module_0.Integer(minimum=var_38)
    var_40 = [var_37, var_39]
    var_41 = module_0.Union(var_40)
    var_42 = 'abc'
    var_43 = var_41.validate(var_42)
    var_44 = exc_info.value.messages()[var_15]
    var_45 = var_44.code
    assert var_45 == 'union'



# Parsed testcases at query #11
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.Array(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.validate(var_5)
    var_7 = True
    var_8 = module_0.Array()
    var_9 = None
    var_10 = var_8.validate(var_9)
    assert var_10 is None
    var_11 = False
    var_12 = module_0.Array()
    var_13 = None
    var_14 = var_12.validate(var_13)
    var_15 = module_0.Array()
    var_16 = 'not an array'
    var_17 = var_15.validate(var_16)
    var_18 = module_0.Array(min_items=var_3)
    var_19 = 1
    var_20 = [var_19]
    var_21 = var_18.validate(var_20)
    var_22 = module_0.Array(max_items=var_21)
    var_23 = 1
    var_24 = 2
    var_25 = 3
    var_26 = [var_23, var_24, var_25]
    var_27 = var_22.validate(var_26)
    var_28 = module_0.Array(exact_items=var_25)
    var_29 = 1
    var_30 = [var_29]
    var_31 = var_28.validate(var_30)
    var_32 = 1
    var_33 = 2
    var_34 = 3
    var_35 = [var_32, var_33, var_34]
    var_36 = var_28.validate(var_35)
    var_37 = [var_7, var_34]
    var_38 = var_28.validate(var_37)
    var_39 = True
    var_40 = module_0.Array(unique_items=var_39)
    var_41 = 1
    var_42 = 2
    var_43 = [var_41, var_42, var_42]
    var_44 = var_40.validate(var_43)
    var_45 = [var_39, var_43, var_44]
    var_46 = var_40.validate(var_45)
    var_47 = module_0.Integer()
    var_48 = module_0.Array(var_47)
    var_49 = 1
    var_50 = 'not an integer'
    var_51 = 3
    var_52 = [var_49, var_50, var_51]
    var_53 = var_48.validate(var_52)
    var_54 = module_0.Integer()
    var_55 = module_0.Integer()
    var_56 = [var_54, var_55]
    var_57 = module_0.String()
    var_58 = module_0.Array(var_56, var_57)
    var_59 = 'three'
    var_60 = [var_39, var_51, var_59]
    var_61 = var_58.validate(var_60)
    var_62 = module_0.Integer()
    var_63 = module_0.Integer()
    var_64 = [var_62, var_63]
    var_65 = module_0.Array(var_64, var_11)
    var_66 = 1
    var_67 = 2
    var_68 = 3
    var_69 = [var_66, var_67, var_68]
    var_70 = var_65.validate(var_69)
    var_71 = module_0.Array(min_items=var_39)
    var_72 = []
    var_73 = var_71.validate(var_72)
    var_74 = module_0.Array(min_items=var_11)
    var_75 = []
    var_76 = var_74.validate(var_75)
    var_77 = module_0.Integer()
    var_78 = module_0.Array(var_77)
    var_79 = [var_39, var_68, var_69]
    var_80 = var_78.serialize(var_79)
    var_81 = var_78.serialize(var_9)
    assert var_81 is None
    var_82 = module_0.Integer()
    var_83 = module_0.String()
    var_84 = [var_82, var_83]
    var_85 = module_0.Array(var_84)
    var_86 = 'two'
    var_87 = [var_39, var_86]
    var_88 = var_85.serialize(var_87)



# Parsed testcases at query #12
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
    var_17 = module_0.Array(min_items=var_16)
    var_18 = [var_6, var_16]
    var_19 = var_17.validate(var_18)
    var_20 = 1
    var_21 = [var_20]
    var_22 = var_17.validate(var_21)
    var_23 = module_0.Array(max_items=var_21)
    var_24 = [var_6, var_21]
    var_25 = var_23.validate(var_24)
    var_26 = 1
    var_27 = 2
    var_28 = 3
    var_29 = [var_26, var_27, var_28]
    var_30 = var_23.validate(var_29)
    var_31 = module_0.Array(exact_items=var_27)
    var_32 = [var_6, var_27]
    var_33 = var_31.validate(var_32)
    var_34 = 1
    var_35 = [var_34]
    var_36 = var_31.validate(var_35)
    var_37 = 1
    var_38 = 2
    var_39 = 3
    var_40 = [var_37, var_38, var_39]
    var_41 = var_31.validate(var_40)
    var_42 = module_0.Array(min_items=var_6)
    var_43 = []
    var_44 = var_42.validate(var_43)
    var_45 = module_0.Integer()
    var_46 = module_0.Array(var_45)
    var_47 = [var_6, var_44, var_39]
    var_48 = var_46.validate(var_47)
    var_49 = 1
    var_50 = 'two'
    var_51 = 3
    var_52 = [var_49, var_50, var_51]
    var_53 = var_46.validate(var_52)
    var_54 = module_0.Integer()
    var_55 = module_0.String()
    var_56 = module_0.Boolean()
    var_57 = [var_54, var_55, var_56]
    var_58 = module_0.Array(var_57)
    var_59 = 'two'
    var_60 = True
    var_61 = [var_6, var_59, var_60]
    var_62 = var_58.validate(var_61)
    var_63 = 1
    var_64 = 'two'
    var_65 = 'three'
    var_66 = [var_63, var_64, var_65]
    var_67 = var_58.validate(var_66)
    var_68 = module_0.Integer()
    var_69 = module_0.String()
    var_70 = [var_68, var_69]
    var_71 = module_0.Boolean()
    var_72 = module_0.Array(var_70, var_71)
    var_73 = True
    var_74 = [var_60, var_59, var_73, var_10]
    var_75 = var_72.validate(var_74)
    var_76 = 1
    var_77 = 'two'
    var_78 = 'three'
    var_79 = 'four'
    var_80 = [var_76, var_77, var_78, var_79]
    var_81 = var_72.validate(var_80)
    var_82 = module_0.Integer()
    var_83 = module_0.String()
    var_84 = [var_82, var_83]
    var_85 = module_0.Array(var_84, var_10)
    var_86 = [var_73, var_59]
    var_87 = var_85.validate(var_86)
    var_88 = 1
    var_89 = 'two'
    var_90 = 3
    var_91 = [var_88, var_89, var_90]
    var_92 = var_85.validate(var_91)
    var_93 = True
    var_94 = module_0.Array(unique_items=var_93)
    var_95 = [var_93, var_89, var_90]
    var_96 = var_94.validate(var_95)
    var_97 = 1
    var_98 = 2
    var_99 = [var_97, var_98, var_98]
    var_100 = var_94.validate(var_99)
    var_101 = module_0.Integer()
    var_102 = module_0.Array(var_101)
    var_103 = [var_93, var_98, var_99]
    var_104 = var_102.serialize(var_103)
    var_105 = var_102.serialize(var_8)
    assert var_105 is None
    var_106 = module_0.Integer()
    var_107 = module_0.String()
    var_108 = module_0.Boolean()
    var_109 = [var_106, var_107, var_108]
    var_110 = module_0.Array(var_109)
    var_111 = True
    var_112 = [var_93, var_59, var_111]
    var_113 = var_110.serialize(var_112)



# Parsed testcases at query #13
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Number()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None
    var_4 = module_0.Number()
    var_5 = None
    var_6 = var_4.validate(var_5)
    var_7 = module_0.Number(coerce_types=var_5)
    var_8 = ''
    var_9 = var_7.validate(var_8)
    assert var_9 is None
    var_10 = module_0.Number()
    var_11 = True
    var_12 = var_10.validate(var_11)
    var_13 = 1.5
    var_14 = var_10.validate(var_13)
    var_15 = False
    var_16 = module_0.Number(coerce_types=var_15)
    var_17 = '123'
    var_18 = var_16.validate(var_17)
    var_19 = module_0.Number(coerce_types=var_17)
    var_20 = '123'
    var_21 = var_19.validate(var_20)
    assert var_21 == 123
    var_22 = module_0.Number()
    var_23 = 'inf'
    var_24 = float(var_23)
    var_25 = var_22.validate(var_24)
    var_26 = '0.01'
    var_27 = module_0.Number(precision=var_26)
    var_28 = '1.234'
    var_29 = var_27.validate(var_28)
    var_30 = 5
    var_31 = module_0.Number(minimum=var_30)
    var_32 = var_31.validate(var_30)
    assert var_32 == 5
    var_33 = 4
    var_34 = var_31.validate(var_33)
    var_35 = module_0.Number(exclusive_minimum=var_30)
    var_36 = 6
    var_37 = var_35.validate(var_36)
    assert var_37 == 6
    var_38 = 5
    var_39 = var_35.validate(var_38)
    var_40 = 10
    var_41 = module_0.Number(maximum=var_40)
    var_42 = var_41.validate(var_40)
    assert var_42 == 10
    var_43 = 11
    var_44 = var_41.validate(var_43)
    var_45 = module_0.Number(exclusive_maximum=var_40)
    var_46 = 9
    var_47 = var_45.validate(var_46)
    assert var_47 == 9
    var_48 = 10
    var_49 = var_45.validate(var_48)
    var_50 = 3
    var_51 = module_0.Number(multiple_of=var_50)
    var_52 = var_51.validate(var_36)
    assert var_52 == 6
    var_53 = 7
    var_54 = var_51.validate(var_53)
    var_55 = 0.5
    var_56 = module_0.Number(multiple_of=var_55)
    var_57 = var_56.validate(var_53)
    var_58 = 1.1
    var_59 = var_56.validate(var_58)



# Parsed testcases at query #14
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'valid string'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'valid string'
    var_3 = True
    var_4 = module_0.String()
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None
    var_7 = module_0.String(allow_blank=var_3)
    var_8 = var_7.validate(var_5)
    assert var_8 == ''
    var_9 = module_0.String()
    var_10 = None
    var_11 = var_9.validate(var_10)
    var_12 = module_0.String()
    var_13 = 123
    var_14 = var_12.validate(var_13)
    var_15 = module_0.String()
    var_16 = ''
    var_17 = var_15.validate(var_16)
    var_18 = module_0.String(allow_blank=var_3)
    var_19 = ''
    var_20 = var_18.validate(var_19)
    assert var_20 == ''
    var_21 = module_0.String()
    var_22 = '   '
    var_23 = var_21.validate(var_22)
    assert var_23 is None
    var_24 = 5
    var_25 = module_0.String(min_length=var_24)
    var_26 = 'short'
    var_27 = var_25.validate(var_26)
    var_28 = 'valid length'
    var_29 = var_25.validate(var_28)
    assert var_29 == 'valid length'
    var_30 = module_0.String(max_length=var_24)
    var_31 = 'toolong'
    var_32 = var_30.validate(var_31)
    var_33 = 'short'
    var_34 = var_30.validate(var_33)
    assert var_34 == 'short'
    var_35 = '^[a-z]+$'
    var_36 = module_0.String(pattern=var_35)
    var_37 = 'Invalid123'
    var_38 = var_36.validate(var_37)
    var_39 = 'validpattern'
    var_40 = var_36.validate(var_39)
    assert var_40 == 'validpattern'
    var_41 = 'email'
    var_42 = module_0.String(format=var_41)
    var_43 = 'invalid-email'
    var_44 = var_42.validate(var_43)
    var_45 = 'valid@example.com'
    var_46 = var_42.validate(var_45)
    assert var_46 == 'valid@example.com'
    var_47 = False
    var_48 = module_0.String(trim_whitespace=var_47)
    var_49 = '  spaces  '
    var_50 = var_48.validate(var_49)
    assert var_50 == '  spaces  '
    var_51 = module_0.String(format=var_41)
    var_52 = var_51.validate(var_45)
    assert var_52 == 'valid@example.com'
    var_53 = module_0.String()
    var_54 = 'valid\x00string'
    var_55 = var_53.validate(var_54)
    assert var_55 == 'validstring'



# Parsed testcases at query #15
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'Option A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'Option B'
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
    var_13 = [var_12]
    var_14 = True
    var_15 = module_0.Choice(choices=var_13)
    var_16 = None
    var_17 = var_15.validate(var_16)
    assert var_17 is None
    var_18 = None
    var_19 = var_7.validate(var_18)
    var_20 = (var_18, var_19)
    var_21 = [var_20]
    var_22 = module_0.Choice(choices=var_21, coerce_types=var_14)
    var_23 = ''
    var_24 = var_22.validate(var_23)
    assert var_24 is None
    var_25 = ''
    var_26 = var_7.validate(var_25)
    var_27 = (var_25, var_26)
    var_28 = (var_3, var_4)
    var_29 = [var_27, var_28]
    var_30 = module_0.Choice(choices=var_29)
    var_31 = var_30.validate(var_25)
    assert var_31 == 'a'
    var_32 = var_30.validate(var_3)
    assert var_32 == 'b'
    var_33 = [var_25, var_26]
    var_34 = [var_3, var_4]
    var_35 = [var_33, var_34]
    var_36 = module_0.Choice(choices=var_35)
    var_37 = var_36.validate(var_25)
    assert var_37 == 'a'
    var_38 = var_36.validate(var_3)
    assert var_38 == 'b'



# Parsed testcases at query #16
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
    var_10 = None
    var_11 = var_0.validate(var_10)
    var_12 = 'not an array'
    var_13 = var_0.validate(var_12)
    var_14 = module_0.Array(min_items=var_13)
    var_15 = [var_6, var_13]
    var_16 = var_14.validate(var_15)
    var_17 = 1
    var_18 = [var_17]
    var_19 = var_14.validate(var_18)
    var_20 = module_0.Array(max_items=var_18)
    var_21 = [var_6, var_18]
    var_22 = var_20.validate(var_21)
    var_23 = 1
    var_24 = 2
    var_25 = 3
    var_26 = [var_23, var_24, var_25]
    var_27 = var_20.validate(var_26)
    var_28 = module_0.Array(exact_items=var_24)
    var_29 = [var_6, var_24]
    var_30 = var_28.validate(var_29)
    var_31 = 1
    var_32 = [var_31]
    var_33 = var_28.validate(var_32)
    var_34 = 1
    var_35 = 2
    var_36 = 3
    var_37 = [var_34, var_35, var_36]
    var_38 = var_28.validate(var_37)
    var_39 = True
    var_40 = module_0.Array(unique_items=var_39)
    var_41 = [var_39, var_35, var_36]
    var_42 = var_40.validate(var_41)
    var_43 = 1
    var_44 = 2
    var_45 = [var_43, var_44, var_44]
    var_46 = var_40.validate(var_45)
    var_47 = module_0.Integer()
    var_48 = module_0.Array(var_47)
    var_49 = '1'
    var_50 = '2'
    var_51 = '3'
    var_52 = [var_49, var_50, var_51]
    var_53 = var_48.validate(var_52)
    var_54 = '1'
    var_55 = 'two'
    var_56 = '3'
    var_57 = [var_54, var_55, var_56]
    var_58 = var_48.validate(var_57)
    var_59 = module_0.Integer()
    var_60 = module_0.Integer()
    var_61 = [var_59, var_60]
    var_62 = False
    var_63 = module_0.Array(var_61, var_62)
    var_64 = [var_39, var_55]
    var_65 = var_63.validate(var_64)
    var_66 = 1
    var_67 = 2
    var_68 = 3
    var_69 = [var_66, var_67, var_68]
    var_70 = var_63.validate(var_69)
    var_71 = module_0.Integer()
    var_72 = module_0.Integer()
    var_73 = [var_71, var_72]
    var_74 = True
    var_75 = module_0.Array(var_73, var_74)
    var_76 = [var_74, var_67, var_68]
    var_77 = var_75.validate(var_76)
    var_78 = module_0.Integer()
    var_79 = module_0.Integer()
    var_80 = [var_78, var_79]
    var_81 = module_0.String()
    var_82 = module_0.Array(var_80, var_81)
    var_83 = 'three'
    var_84 = [var_74, var_67, var_83]
    var_85 = var_82.validate(var_84)
    var_86 = 1
    var_87 = 2
    var_88 = 3
    var_89 = [var_86, var_87, var_88]
    var_90 = var_82.validate(var_89)
    var_91 = []
    var_92 = var_14.validate(var_91)
    var_93 = module_0.Array(min_items=var_62)
    var_94 = []
    var_95 = var_93.validate(var_94)
    var_96 = [var_74, var_92, var_88]
    var_97 = var_0.serialize(var_96)
    var_98 = [var_74, var_92, var_88]
    var_99 = var_48.serialize(var_98)



# Parsed testcases at query #17
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None
    var_4 = module_0.String(allow_blank=var_0)
    var_5 = var_4.validate(var_2)
    assert var_5 == ''
    var_6 = module_0.String()
    var_7 = None
    var_8 = var_6.validate(var_7)
    var_9 = module_0.String()
    var_10 = 123
    var_11 = var_9.validate(var_10)
    var_12 = module_0.String()
    var_13 = 'a\x00b'
    var_14 = var_12.validate(var_13)
    assert var_14 == 'ab'
    var_15 = module_0.String(trim_whitespace=var_10)
    var_16 = '  hello  '
    var_17 = var_15.validate(var_16)
    assert var_17 == 'hello'
    var_18 = False
    var_19 = module_0.String(allow_blank=var_18)
    var_20 = ''
    var_21 = var_19.validate(var_20)
    var_22 = module_0.String(allow_blank=var_18)
    var_23 = ''
    var_24 = var_22.validate(var_23)
    assert var_24 is None
    var_25 = 3
    var_26 = module_0.String(min_length=var_25)
    var_27 = 'ab'
    var_28 = var_26.validate(var_27)
    var_29 = module_0.String(max_length=var_25)
    var_30 = 'abcd'
    var_31 = var_29.validate(var_30)
    var_32 = '^[a-z]+$'
    var_33 = module_0.String(pattern=var_32)
    var_34 = '123'
    var_35 = var_33.validate(var_34)
    var_36 = module_0.String(pattern=var_32)
    var_37 = 'abc'
    var_38 = var_36.validate(var_37)
    assert var_38 == 'abc'
    var_39 = 'email'
    var_40 = module_0.String(format=var_39)
    var_41 = 'invalid-email'
    var_42 = var_40.validate(var_41)
    var_43 = module_0.String(format=var_39)
    var_44 = 'test@example.com'
    var_45 = var_43.validate(var_44)
    assert var_45 == 'test@example.com'
    var_46 = module_0.String(format=var_39)
    var_47 = var_46.validate(var_44)
    assert var_47 == 'test@example.com'



# Parsed testcases at query #18
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Const(var_0)
    var_2 = None
    var_3 = module_0.Const(var_2)
    var_4 = 42
    var_5 = True
    var_6 = module_0.Const(var_4)



# Parsed testcases at query #19
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
    var_10 = module_0.Array()
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = 'not a list'
    var_14 = var_10.validate(var_13)
    var_15 = module_0.Array(min_items=var_14)
    var_16 = [var_6, var_14]
    var_17 = var_15.validate(var_16)
    var_18 = 1
    var_19 = [var_18]
    var_20 = var_15.validate(var_19)
    var_21 = module_0.Array(max_items=var_19)
    var_22 = [var_6, var_19]
    var_23 = var_21.validate(var_22)
    var_24 = 1
    var_25 = 2
    var_26 = 3
    var_27 = [var_24, var_25, var_26]
    var_28 = var_21.validate(var_27)
    var_29 = module_0.Array(exact_items=var_25)
    var_30 = [var_6, var_25]
    var_31 = var_29.validate(var_30)
    var_32 = 1
    var_33 = [var_32]
    var_34 = var_29.validate(var_33)
    var_35 = 1
    var_36 = 2
    var_37 = 3
    var_38 = [var_35, var_36, var_37]
    var_39 = var_29.validate(var_38)
    var_40 = True
    var_41 = module_0.Array(unique_items=var_40)
    var_42 = [var_40, var_36, var_37]
    var_43 = var_41.validate(var_42)
    var_44 = 1
    var_45 = 2
    var_46 = [var_44, var_45, var_45]
    var_47 = var_41.validate(var_46)
    var_48 = module_0.Integer()
    var_49 = module_0.Array(var_48)
    var_50 = [var_40, var_45, var_46]
    var_51 = var_49.validate(var_50)
    var_52 = 1
    var_53 = 'two'
    var_54 = 3
    var_55 = [var_52, var_53, var_54]
    var_56 = var_49.validate(var_55)
    var_57 = module_0.Integer()
    var_58 = module_0.Integer()
    var_59 = [var_57, var_58]
    var_60 = False
    var_61 = module_0.Array(var_59, var_60)
    var_62 = [var_40, var_53]
    var_63 = var_61.validate(var_62)
    var_64 = 1
    var_65 = 2
    var_66 = 3
    var_67 = [var_64, var_65, var_66]
    var_68 = var_61.validate(var_67)
    var_69 = module_0.Integer()
    var_70 = module_0.Integer()
    var_71 = [var_69, var_70]
    var_72 = module_0.String()
    var_73 = module_0.Array(var_71, var_72)
    var_74 = 'three'
    var_75 = [var_40, var_65, var_74]
    var_76 = var_73.validate(var_75)
    var_77 = 1
    var_78 = 2
    var_79 = 3
    var_80 = [var_77, var_78, var_79]
    var_81 = var_73.validate(var_80)
    var_82 = [var_40, var_78, var_79]
    var_83 = var_10.serialize(var_82)
    var_84 = [var_40, var_78, var_79]
    var_85 = var_49.serialize(var_84)



# Parsed testcases at query #20
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Array()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    var_3 = module_0.Array(var_1)
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = var_3.serialize(var_7)
    var_9 = module_0.Integer()
    var_10 = module_0.Array(var_9)
    var_11 = [var_4, var_5, var_6]
    var_12 = var_10.serialize(var_11)
    var_13 = '1'
    var_14 = '2'
    var_15 = '3'
    var_16 = [var_13, var_14, var_15]
    var_17 = var_10.serialize(var_16)
    var_18 = module_0.Integer()
    var_19 = module_0.String()
    var_20 = [var_18, var_19]
    var_21 = module_0.Array(var_20)
    var_22 = 'hello'
    var_23 = [var_4, var_22]
    var_24 = var_21.serialize(var_23)
    var_25 = [var_13, var_22]
    var_26 = var_21.serialize(var_25)
    var_27 = module_0.Integer()
    var_28 = [var_27]
    var_29 = True
    var_30 = module_0.Array(var_28, var_29)
    var_31 = [var_29, var_5, var_6]
    var_32 = var_30.serialize(var_31)
    var_33 = module_0.Integer()
    var_34 = module_0.String()
    var_35 = [var_33]
    var_36 = module_0.Array(var_35, var_34)
    var_37 = 'world'
    var_38 = [var_29, var_22, var_37]
    var_39 = var_36.serialize(var_38)



# Parsed testcases at query #21
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'Option A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'Option B'
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
    var_13 = [var_12]
    var_14 = True
    var_15 = module_0.Choice(choices=var_13)
    var_16 = None
    var_17 = var_15.validate(var_16)
    assert var_17 is None
    var_18 = None
    var_19 = var_7.validate(var_18)
    var_20 = (var_18, var_19)
    var_21 = [var_20]
    var_22 = module_0.Choice(choices=var_21, coerce_types=var_14)
    var_23 = ''
    var_24 = var_22.validate(var_23)
    assert var_24 is None
    var_25 = ''
    var_26 = var_7.validate(var_25)
    var_27 = (var_25, var_26)
    var_28 = (var_3, var_4)
    var_29 = [var_27, var_28]
    var_30 = module_0.Choice(choices=var_29)
    var_31 = var_30.validate(var_25)
    assert var_31 == 'a'
    var_32 = var_30.validate(var_3)
    assert var_32 == 'b'
    var_33 = [var_25, var_26]
    var_34 = [var_3, var_4]
    var_35 = [var_33, var_34]
    var_36 = module_0.Choice(choices=var_35)
    var_37 = var_36.validate(var_25)
    assert var_37 == 'a'
    var_38 = var_36.validate(var_3)
    assert var_38 == 'b'



# Parsed testcases at query #22
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Array()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None
    var_4 = module_0.Array()
    var_5 = None
    var_6 = var_4.validate(var_5)
    var_7 = module_0.Array()
    var_8 = 'not a list'
    var_9 = var_7.validate(var_8)
    var_10 = module_0.Array()
    var_11 = []
    var_12 = var_10.validate(var_11)
    var_13 = module_0.Array(min_items=var_8)
    var_14 = []
    var_15 = var_13.validate(var_14)
    var_16 = 2
    var_17 = module_0.Array(exact_items=var_16)
    var_18 = [var_14, var_16]
    var_19 = var_17.validate(var_18)
    var_20 = 1
    var_21 = [var_20]
    var_22 = var_17.validate(var_21)
    var_23 = module_0.Array(max_items=var_16)
    var_24 = [var_20, var_16]
    var_25 = var_23.validate(var_24)
    var_26 = 1
    var_27 = 2
    var_28 = 3
    var_29 = [var_26, var_27, var_28]
    var_30 = var_23.validate(var_29)
    var_31 = module_0.Integer()
    var_32 = module_0.Array(var_31)
    var_33 = '1'
    var_34 = '2'
    var_35 = [var_33, var_34]
    var_36 = var_32.validate(var_35)
    var_37 = 'a'
    var_38 = 'b'
    var_39 = [var_37, var_38]
    var_40 = var_32.validate(var_39)
    var_41 = module_0.Integer()
    var_42 = module_0.Integer()
    var_43 = [var_41, var_42]
    var_44 = False
    var_45 = module_0.Array(var_43, var_44)
    var_46 = [var_37, var_16]
    var_47 = var_45.validate(var_46)
    var_48 = 1
    var_49 = 2
    var_50 = 3
    var_51 = [var_48, var_49, var_50]
    var_52 = var_45.validate(var_51)
    var_53 = module_0.Array(unique_items=var_48)
    var_54 = 3
    var_55 = [var_48, var_16, var_54]
    var_56 = var_53.validate(var_55)
    var_57 = 1
    var_58 = 2
    var_59 = [var_57, var_58, var_57]
    var_60 = var_53.validate(var_59)
    var_61 = 'a'
    var_62 = module_0.Integer()
    var_63 = {var_61: var_62}
    var_64 = module_0.Object(properties=var_63)
    var_65 = module_0.Array(var_64)
    var_66 = {var_61: var_33}
    var_67 = {var_61: var_34}
    var_68 = [var_66, var_67]
    var_69 = var_65.validate(var_68)
    var_70 = 'a'
    var_71 = '1'
    var_72 = {var_70: var_71}
    var_73 = 'b'
    var_74 = {var_70: var_73}
    var_75 = [var_72, var_74]
    var_76 = var_65.validate(var_75)
    var_77 = module_0.Integer()
    var_78 = module_0.Array(var_77)
    var_79 = [var_33, var_34]
    var_80 = var_78.serialize(var_79)
    var_81 = var_78.serialize(var_71)
    assert var_81 is None



# Parsed testcases at query #23
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = module_0.Const(var_0)
    var_2 = None
    var_3 = module_0.Const(var_2)
    var_4 = 42
    var_5 = module_0.Const(var_4)
    var_6 = 3.14
    var_7 = module_0.Const(var_6)
    var_8 = True
    var_9 = module_0.Const(var_8)
    var_10 = 2
    var_11 = 3
    var_12 = [var_8, var_10, var_11]
    var_13 = module_0.Const(var_12)
    var_14 = 'test_value'
    var_15 = True
    var_16 = module_0.Const(var_14)



# Parsed testcases at query #24
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Const(var_0)
    var_2 = None
    var_3 = module_0.Const(var_2)
    var_4 = 'test'
    var_5 = module_0.Const(var_4)
    var_6 = 42
    var_7 = True
    var_8 = module_0.Const(var_6)
    var_9 = 'Test field'
    var_10 = module_0.Const(var_6)



# Parsed testcases at query #25
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
    var_15 = module_0.Array(max_items=var_14)
    var_16 = 1
    var_17 = 2
    var_18 = 3
    var_19 = [var_16, var_17, var_18]
    var_20 = var_15.validate(var_19)
    var_21 = 3
    var_22 = module_0.Array(exact_items=var_21)
    var_23 = [var_16, var_20, var_21]
    var_24 = var_22.validate(var_23)
    var_25 = 1
    var_26 = 2
    var_27 = [var_25, var_26]
    var_28 = var_22.validate(var_27)
    var_29 = module_0.Array(unique_items=var_25)
    var_30 = [var_25, var_20, var_21]
    var_31 = var_29.validate(var_30)
    var_32 = 1
    var_33 = 2
    var_34 = [var_32, var_33, var_33]
    var_35 = var_29.validate(var_34)
    var_36 = module_0.Integer()
    var_37 = module_0.Array(var_36)
    var_38 = '1'
    var_39 = '2'
    var_40 = '3'
    var_41 = [var_38, var_39, var_40]
    var_42 = var_37.validate(var_41)
    var_43 = '1'
    var_44 = 'two'
    var_45 = '3'
    var_46 = [var_43, var_44, var_45]
    var_47 = var_37.validate(var_46)
    var_48 = module_0.Integer()
    var_49 = module_0.Integer()
    var_50 = [var_48, var_49]
    var_51 = module_0.Array(var_50, var_46)
    var_52 = [var_43, var_47]
    var_53 = var_51.validate(var_52)
    var_54 = 1
    var_55 = 2
    var_56 = 3
    var_57 = [var_54, var_55, var_56]
    var_58 = var_51.validate(var_57)
    var_59 = module_0.Integer()
    var_60 = module_0.Integer()
    var_61 = [var_59, var_60]
    var_62 = module_0.String()
    var_63 = module_0.Array(var_61, var_62)
    var_64 = 'three'
    var_65 = [var_54, var_58, var_64]
    var_66 = var_63.validate(var_65)
    var_67 = 1
    var_68 = 2
    var_69 = 3
    var_70 = [var_67, var_68, var_69]
    var_71 = var_63.validate(var_70)
    var_72 = module_0.Integer()
    var_73 = module_0.String()
    var_74 = module_0.Boolean()
    var_75 = [var_72, var_73, var_74]
    var_76 = module_0.Array(var_75)
    var_77 = 'two'
    var_78 = [var_38, var_77, var_67]
    var_79 = var_76.validate(var_78)
    var_80 = '1'
    var_81 = 2
    var_82 = True
    var_83 = [var_80, var_81, var_82]
    var_84 = var_76.validate(var_83)
    var_85 = module_0.Integer()
    var_86 = module_0.Array(var_85)
    var_87 = module_0.Array(var_86)
    var_88 = [var_38, var_39]
    var_89 = '4'
    var_90 = [var_40, var_89]
    var_91 = [var_88, var_90]
    var_92 = var_87.validate(var_91)
    var_93 = '1'
    var_94 = 'two'
    var_95 = [var_93, var_94]
    var_96 = '3'
    var_97 = '4'
    var_98 = [var_96, var_97]
    var_99 = [var_95, var_98]
    var_100 = var_87.validate(var_99)
    var_101 = module_0.Integer()
    var_102 = module_0.Array(var_101)
    var_103 = [var_93, var_94, var_98]
    var_104 = var_102.validate(var_103)
    var_105 = module_0.Boolean(coerce_types=var_93)
    var_106 = module_0.Array(var_105)
    var_107 = 'true'
    var_108 = 'false'
    var_109 = '0'
    var_110 = [var_107, var_108, var_38, var_109]
    var_111 = var_106.validate(var_110)



# Parsed testcases at query #26
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Object()
    var_1 = {}
    var_2 = var_0.validate(var_1)
    var_3 = True
    var_4 = module_0.Object()
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None
    var_7 = module_0.Object()
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = module_0.Object()
    var_11 = 'not a dict'
    var_12 = var_10.validate(var_11)
    var_13 = module_0.Object()
    var_14 = 1
    var_15 = 'value'
    var_16 = {var_14: var_15}
    var_17 = var_13.validate(var_16)
    var_18 = module_0.Object(min_properties=var_16)
    var_19 = {}
    var_20 = var_18.validate(var_19)
    var_21 = module_0.Object(max_properties=var_16)
    var_22 = 'a'
    var_23 = 'b'
    var_24 = 1
    var_25 = 2
    var_26 = {var_22: var_24, var_23: var_25}
    var_27 = var_21.validate(var_26)
    var_28 = 'a'
    var_29 = [var_28]
    var_30 = module_0.Object(required=var_29)
    var_31 = 'b'
    var_32 = 1
    var_33 = {var_31: var_32}
    var_34 = var_30.validate(var_33)
    var_35 = module_0.Integer()
    var_36 = {var_28: var_35}
    var_37 = module_0.Object(properties=var_36)
    var_38 = '1'
    var_39 = {var_28: var_38}
    var_40 = var_37.validate(var_39)
    var_41 = 'a'
    var_42 = 'not a number'
    var_43 = {var_41: var_42}
    var_44 = var_37.validate(var_43)
    var_45 = module_0.Integer()
    var_46 = {var_28: var_45}
    var_47 = False
    var_48 = module_0.Object(properties=var_46, additional_properties=var_47)
    var_49 = 'a'
    var_50 = 'b'
    var_51 = 1
    var_52 = 2
    var_53 = {var_49: var_51, var_50: var_52}
    var_54 = var_48.validate(var_53)
    var_55 = module_0.Integer()
    var_56 = {var_54: var_55}
    var_57 = module_0.String()
    var_58 = module_0.Object(properties=var_56, additional_properties=var_57)
    var_59 = 'b'
    var_60 = 'test'
    var_61 = {var_54: var_51, var_59: var_60}
    var_62 = var_58.validate(var_61)
    var_63 = 'a'
    var_64 = 'b'
    var_65 = 1
    var_66 = 123
    var_67 = {var_63: var_65, var_64: var_66}
    var_68 = var_58.validate(var_67)
    var_69 = 5
    var_70 = module_0.String(max_length=var_69)
    var_71 = module_0.Object(property_names=var_70)
    var_72 = 'abc'
    var_73 = {var_72: var_65}
    var_74 = var_71.validate(var_73)
    var_75 = 'abcdef'
    var_76 = 1
    var_77 = {var_75: var_76}
    var_78 = var_71.validate(var_77)
    var_79 = '^test_'
    var_80 = module_0.String()
    var_81 = {var_79: var_80}
    var_82 = module_0.Object(pattern_properties=var_81)
    var_83 = 'test_a'
    var_84 = 'value'
    var_85 = {var_83: var_84}
    var_86 = var_82.validate(var_85)
    var_87 = 'test_a'
    var_88 = 123
    var_89 = {var_87: var_88}
    var_90 = var_82.validate(var_89)
    var_91 = module_0.Integer()
    var_92 = {var_68: var_91}
    var_93 = module_0.Object(properties=var_92)
    var_94 = {}
    var_95 = var_93.validate(var_94)



# Parsed testcases at query #27
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Const(var_0)
    var_2 = 'hello'
    var_3 = module_0.Const(var_2)
    var_4 = None
    var_5 = module_0.Const(var_4)
    var_6 = 42
    var_7 = True
    var_8 = module_0.Const(var_6)



# Parsed testcases at query #28
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'Option A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'Option B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = module_0.Choice(choices=var_6)
    var_8 = 'x'
    var_9 = 'Option X'
    var_10 = (var_8, var_9)
    var_11 = [var_10]
    var_12 = True
    var_13 = False
    var_14 = module_0.Choice(choices=var_11, coerce_types=var_13)
    var_15 = 'single'
    var_16 = [var_15]
    var_17 = module_0.Choice(choices=var_16)
    var_18 = []
    var_19 = module_0.Choice(choices=var_18)
    var_20 = '1'
    var_21 = 'First'
    var_22 = (var_20, var_21)
    var_23 = [var_22]
    var_24 = 'Test Choice'
    var_25 = 'A test choice field'
    var_26 = module_0.Choice(choices=var_23)
    var_27 = 'key1'
    var_28 = 'Value 1'
    var_29 = (var_27, var_28)
    var_30 = 'key2'
    var_31 = 'Value 2'
    var_32 = (var_30, var_31)
    var_33 = [var_29, var_32]
    var_34 = module_0.Choice(choices=var_33)
    var_35 = [var_27, var_28]
    var_36 = [var_30, var_31]
    var_37 = [var_35, var_36]
    var_38 = module_0.Choice(choices=var_37)



# Parsed testcases at query #29
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
    var_17 = module_0.Array(min_items=var_16)
    var_18 = [var_6, var_16]
    var_19 = var_17.validate(var_18)
    var_20 = 1
    var_21 = [var_20]
    var_22 = var_17.validate(var_21)
    var_23 = module_0.Array(max_items=var_21)
    var_24 = [var_6, var_21]
    var_25 = var_23.validate(var_24)
    var_26 = 1
    var_27 = 2
    var_28 = 3
    var_29 = [var_26, var_27, var_28]
    var_30 = var_23.validate(var_29)
    var_31 = module_0.Array(exact_items=var_27)
    var_32 = [var_6, var_27]
    var_33 = var_31.validate(var_32)
    var_34 = 1
    var_35 = [var_34]
    var_36 = var_31.validate(var_35)
    var_37 = 1
    var_38 = 2
    var_39 = 3
    var_40 = [var_37, var_38, var_39]
    var_41 = var_31.validate(var_40)
    var_42 = True
    var_43 = module_0.Array(unique_items=var_42)
    var_44 = [var_42, var_38, var_39]
    var_45 = var_43.validate(var_44)
    var_46 = 1
    var_47 = 2
    var_48 = [var_46, var_47, var_47]
    var_49 = var_43.validate(var_48)
    var_50 = module_0.Integer()
    var_51 = module_0.Array(var_50)
    var_52 = [var_42, var_47, var_48]
    var_53 = var_51.validate(var_52)
    var_54 = 1
    var_55 = 'two'
    var_56 = 3
    var_57 = [var_54, var_55, var_56]
    var_58 = var_51.validate(var_57)
    var_59 = module_0.Integer()
    var_60 = module_0.String()
    var_61 = [var_59, var_60]
    var_62 = module_0.Array(var_61)
    var_63 = 'two'
    var_64 = [var_42, var_63]
    var_65 = var_62.validate(var_64)
    var_66 = 1
    var_67 = 2
    var_68 = [var_66, var_67]
    var_69 = var_62.validate(var_68)
    var_70 = module_0.Integer()
    var_71 = [var_70]
    var_72 = module_0.Array(var_71, var_10)
    var_73 = [var_42]
    var_74 = var_72.validate(var_73)
    var_75 = 1
    var_76 = 2
    var_77 = [var_75, var_76]
    var_78 = var_72.validate(var_77)
    var_79 = module_0.Integer()
    var_80 = [var_79]
    var_81 = module_0.String()
    var_82 = module_0.Array(var_80, var_81)
    var_83 = [var_42, var_63]
    var_84 = var_82.validate(var_83)
    var_85 = 1
    var_86 = 2
    var_87 = [var_85, var_86]
    var_88 = var_82.validate(var_87)
    var_89 = module_0.Array(min_items=var_42)
    var_90 = []
    var_91 = var_89.validate(var_90)



# Parsed testcases at query #30
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
    var_10 = module_0.Array()
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = 'not a list'
    var_14 = var_10.validate(var_13)
    var_15 = module_0.Array(min_items=var_14)
    var_16 = [var_6, var_14]
    var_17 = var_15.validate(var_16)
    var_18 = 1
    var_19 = [var_18]
    var_20 = var_15.validate(var_19)
    var_21 = module_0.Array(max_items=var_19)
    var_22 = [var_6, var_19]
    var_23 = var_21.validate(var_22)
    var_24 = 1
    var_25 = 2
    var_26 = 3
    var_27 = [var_24, var_25, var_26]
    var_28 = var_21.validate(var_27)
    var_29 = module_0.Array(exact_items=var_25)
    var_30 = [var_6, var_25]
    var_31 = var_29.validate(var_30)
    var_32 = 1
    var_33 = [var_32]
    var_34 = var_29.validate(var_33)
    var_35 = 1
    var_36 = 2
    var_37 = 3
    var_38 = [var_35, var_36, var_37]
    var_39 = var_29.validate(var_38)
    var_40 = module_0.Integer()
    var_41 = module_0.Array(var_40)
    var_42 = [var_6, var_36, var_37]
    var_43 = var_41.validate(var_42)
    var_44 = 1
    var_45 = 'two'
    var_46 = 3
    var_47 = [var_44, var_45, var_46]
    var_48 = var_41.validate(var_47)
    var_49 = module_0.Integer()
    var_50 = module_0.String()
    var_51 = [var_49, var_50]
    var_52 = module_0.Array(var_51)
    var_53 = 'two'
    var_54 = [var_6, var_53]
    var_55 = var_52.validate(var_54)
    var_56 = 1
    var_57 = 'two'
    var_58 = 3
    var_59 = [var_56, var_57, var_58]
    var_60 = var_52.validate(var_59)
    var_61 = module_0.Integer()
    var_62 = module_0.String()
    var_63 = [var_61, var_62]
    var_64 = False
    var_65 = module_0.Array(var_63, var_64)
    var_66 = [var_6, var_53]
    var_67 = var_65.validate(var_66)
    var_68 = 1
    var_69 = 'two'
    var_70 = 3
    var_71 = [var_68, var_69, var_70]
    var_72 = var_65.validate(var_71)
    var_73 = True
    var_74 = module_0.Array(unique_items=var_73)
    var_75 = [var_73, var_69, var_70]
    var_76 = var_74.validate(var_75)
    var_77 = 1
    var_78 = 2
    var_79 = [var_77, var_78, var_78]
    var_80 = var_74.validate(var_79)
    var_81 = module_0.Integer()
    var_82 = module_0.Array(var_81)
    var_83 = [var_73, var_78, var_79]
    var_84 = var_82.serialize(var_83)
    var_85 = var_82.serialize(var_8)
    assert var_85 is None
    var_86 = module_0.Integer()
    var_87 = module_0.String()
    var_88 = [var_86, var_87]
    var_89 = module_0.Array(var_88)
    var_90 = [var_73, var_53]
    var_91 = var_89.serialize(var_90)



# Parsed testcases at query #31
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'Option A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'Option B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = module_0.Choice(choices=var_6)
    var_8 = var_7.validate(var_0)
    assert var_8 == 'a'
    var_9 = var_7.validate(var_3)
    assert var_9 == 'b'
    var_10 = (var_0, var_1)
    var_11 = (var_3, var_4)
    var_12 = [var_10, var_11]
    var_13 = True
    var_14 = module_0.Choice(choices=var_12)
    var_15 = None
    var_16 = var_14.validate(var_15)
    assert var_16 is None
    var_17 = (var_0, var_1)
    var_18 = (var_3, var_4)
    var_19 = [var_17, var_18]
    var_20 = module_0.Choice(choices=var_19)
    var_21 = 'c'
    var_22 = var_20.validate(var_21)
    var_23 = (var_21, var_22)
    var_24 = (var_3, var_4)
    var_25 = [var_23, var_24]
    var_26 = module_0.Choice(choices=var_25, coerce_types=var_13)
    var_27 = ''
    var_28 = var_26.validate(var_27)
    assert var_28 is None
    var_29 = (var_21, var_22)
    var_30 = (var_3, var_4)
    var_31 = [var_29, var_30]
    var_32 = module_0.Choice(choices=var_31)
    var_33 = ''
    var_34 = var_32.validate(var_33)
    var_35 = (var_33, var_34)
    var_36 = (var_3, var_4)
    var_37 = [var_35, var_36]
    var_38 = module_0.Choice(choices=var_37)
    var_39 = None
    var_40 = var_38.validate(var_39)



# Parsed testcases at query #32
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Object(properties=var_4)
    var_6 = 'Alice'
    var_7 = 30
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = module_0.String()
    var_11 = {var_0: var_10}
    var_12 = [var_0]
    var_13 = module_0.Object(properties=var_11, required=var_12)
    var_14 = {}
    var_15 = var_13.validate(var_14)
    var_16 = module_0.String()
    var_17 = {var_14: var_16}
    var_18 = True
    var_19 = module_0.Object(properties=var_17, additional_properties=var_18)
    var_20 = 'extra'
    var_21 = 'value'
    var_22 = {var_14: var_6, var_20: var_21}
    var_23 = var_19.validate(var_22)
    var_24 = module_0.String()
    var_25 = {var_14: var_24}
    var_26 = False
    var_27 = module_0.Object(properties=var_25, additional_properties=var_26)
    var_28 = 'name'
    var_29 = 'extra'
    var_30 = 'Alice'
    var_31 = 'value'
    var_32 = {var_28: var_30, var_29: var_31}
    var_33 = var_27.validate(var_32)
    var_34 = module_0.String()
    var_35 = {var_28: var_34}
    var_36 = module_0.String()
    var_37 = module_0.Object(properties=var_35, additional_properties=var_36)
    var_38 = {var_28: var_33, var_20: var_21}
    var_39 = var_37.validate(var_38)
    var_40 = module_0.Object(min_properties=var_18)
    var_41 = {var_28: var_33}
    var_42 = var_40.validate(var_41)
    var_43 = {}
    var_44 = var_40.validate(var_43)
    var_45 = module_0.Object(max_properties=var_18)
    var_46 = {var_43: var_33}
    var_47 = var_45.validate(var_46)
    var_48 = 'name'
    var_49 = 'age'
    var_50 = 'Alice'
    var_51 = 30
    var_52 = {var_48: var_50, var_49: var_51}
    var_53 = var_45.validate(var_52)
    var_54 = 3
    var_55 = module_0.String(min_length=var_54)
    var_56 = module_0.Object(property_names=var_55)
    var_57 = {var_48: var_53}
    var_58 = var_56.validate(var_57)
    var_59 = 'na'
    var_60 = 'Alice'
    var_61 = {var_59: var_60}
    var_62 = var_56.validate(var_61)
    var_63 = '^S_'
    var_64 = '^I_'
    var_65 = module_0.String()
    var_66 = module_0.Integer()
    var_67 = {var_63: var_65, var_64: var_66}
    var_68 = module_0.Object(pattern_properties=var_67)
    var_69 = 'S_name'
    var_70 = 'I_age'
    var_71 = {var_69: var_53, var_70: var_7}
    var_72 = var_68.validate(var_71)
    var_73 = module_0.Object()
    var_74 = None
    var_75 = var_73.validate(var_74)
    assert var_75 is None
    var_76 = module_0.Object()
    var_77 = 123
    var_78 = 'value'
    var_79 = {var_77: var_78}
    var_80 = var_76.validate(var_79)
    var_81 = 'address'
    var_82 = 'city'
    var_83 = module_0.String()
    var_84 = {var_82: var_83}
    var_85 = module_0.Object(properties=var_84)
    var_86 = {var_81: var_85}
    var_87 = module_0.Object(properties=var_86)
    var_88 = 'New York'
    var_89 = {var_82: var_88}
    var_90 = {var_81: var_89}
    var_91 = var_87.validate(var_90)



# Parsed testcases at query #33
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'Option A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'Option B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = module_0.Choice(choices=var_6)
    var_8 = var_7.validate(var_0)
    assert var_8 == 'a'
    var_9 = var_7.validate(var_3)
    assert var_9 == 'b'
    var_10 = (var_0, var_1)
    var_11 = [var_10]
    var_12 = True
    var_13 = module_0.Choice(choices=var_11)
    var_14 = None
    var_15 = var_13.validate(var_14)
    assert var_15 is None
    var_16 = (var_0, var_1)
    var_17 = [var_16]
    var_18 = module_0.Choice(choices=var_17)
    var_19 = 'invalid'
    var_20 = var_18.validate(var_19)
    var_21 = (var_19, var_20)
    var_22 = [var_21]
    var_23 = module_0.Choice(choices=var_22, coerce_types=var_12)
    var_24 = ''
    var_25 = var_23.validate(var_24)
    assert var_25 is None
    var_26 = (var_19, var_20)
    var_27 = [var_26]
    var_28 = False
    var_29 = module_0.Choice(choices=var_27, coerce_types=var_28)
    var_30 = ''
    var_31 = var_29.validate(var_30)
    var_32 = (var_30, var_31)
    var_33 = [var_32]
    var_34 = module_0.Choice(choices=var_33)
    var_35 = None
    var_36 = var_34.validate(var_35)



# Parsed testcases at query #34
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Choice(choices=var_3)
    var_5 = 'Option A'
    var_6 = (var_0, var_5)
    var_7 = 'Option B'
    var_8 = (var_1, var_7)
    var_9 = [var_6, var_8]
    var_10 = module_0.Choice(choices=var_9)
    var_11 = (var_1, var_7)
    var_12 = 'Option C'
    var_13 = (var_2, var_12)
    var_14 = [var_0, var_11, var_13]
    var_15 = module_0.Choice(choices=var_14)
    var_16 = []
    var_17 = module_0.Choice(choices=var_16)
    var_18 = [var_0, var_1]
    var_19 = False
    var_20 = module_0.Choice(choices=var_18, coerce_types=var_19)
    var_21 = [var_0, var_1]
    var_22 = 'Test Choice'
    var_23 = 'A test choice field'
    var_24 = True
    var_25 = module_0.Choice(choices=var_21)
    var_26 = 'a'
    var_27 = 'b'
    var_28 = 'Option B'
    var_29 = 'extra'
    var_30 = (var_27, var_28, var_29)
    var_31 = [var_26, var_30]
    var_32 = module_0.Choice(choices=var_31)



# Parsed testcases at query #35
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.Float()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = True
    var_5 = module_0.Integer()
    var_6 = [var_5, var_1]
    var_7 = module_0.Union(var_6)
    var_8 = [var_0, var_1]
    var_9 = 'Test union field'
    var_10 = module_0.Union(var_8)



# Parsed testcases at query #36
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = 42
    var_5 = var_3.validate(var_4)
    assert var_5 == 42
    var_6 = 'hello'
    var_7 = var_3.validate(var_6)
    assert var_7 == 'hello'
    var_8 = True
    var_9 = module_0.Integer()
    var_10 = module_0.String()
    var_11 = [var_9, var_10]
    var_12 = module_0.Union(var_11)
    var_13 = None
    var_14 = var_12.validate(var_13)
    assert var_14 is None
    var_15 = 3.14
    var_16 = var_3.validate(var_15)
    var_17 = 0
    var_18 = module_0.Integer(minimum=var_17)
    var_19 = 5
    var_20 = module_0.String(min_length=var_19)
    var_21 = [var_18, var_20]
    var_22 = module_0.Union(var_21)
    var_23 = -1
    var_24 = var_22.validate(var_23)
    var_25 = exc_info.value.messages()[var_17]
    var_26 = var_25.code
    assert var_26 == 'minimum'
    var_27 = module_0.Integer()
    var_28 = module_0.Boolean()
    var_29 = [var_27, var_28]
    var_30 = module_0.Union(var_29)
    var_31 = 'not_a_number_or_boolean'
    var_32 = var_30.validate(var_31)
    var_33 = exc_info.value.messages()[var_17]
    var_34 = var_33.code
    assert var_34 == 'union'
    var_35 = module_0.Integer(coerce_types=var_8)
    var_36 = module_0.String()
    var_37 = [var_35, var_36]
    var_38 = module_0.Union(var_37)
    var_39 = '42'
    var_40 = var_38.validate(var_39)
    assert var_40 == 42
    var_41 = 'not_a_number'
    var_42 = var_38.validate(var_41)
    var_43 = module_0.Integer()
    var_44 = module_0.Float()
    var_45 = [var_43, var_44]
    var_46 = module_0.Union(var_45)
    var_47 = module_0.String()
    var_48 = [var_46, var_47]
    var_49 = module_0.Union(var_48)
    var_50 = var_49.validate(var_4)
    assert var_50 == 42
    var_51 = 3.14
    var_52 = var_49.validate(var_51)
    var_53 = var_49.validate(var_6)
    assert var_53 == 'hello'
    var_54 = 'not_a_number_or_string'
    var_55 = [var_54]
    var_56 = var_49.validate(var_55)



# Parsed testcases at query #37
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Object(properties=var_4)
    var_6 = 'John'
    var_7 = 30
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = True
    var_11 = module_0.Object()
    var_12 = None
    var_13 = var_11.validate(var_12)
    assert var_13 is None
    var_14 = module_0.String()
    var_15 = {var_0: var_14}
    var_16 = [var_0]
    var_17 = module_0.Object(properties=var_15, required=var_16)
    var_18 = 'age'
    var_19 = 30
    var_20 = {var_18: var_19}
    var_21 = var_17.validate(var_20)
    var_22 = module_0.Object()
    var_23 = 123
    var_24 = 'value'
    var_25 = {var_23: var_24}
    var_26 = var_22.validate(var_25)
    var_27 = 2
    var_28 = module_0.Object(min_properties=var_27)
    var_29 = 'a'
    var_30 = 1
    var_31 = {var_29: var_30}
    var_32 = var_28.validate(var_31)
    var_33 = module_0.Object(max_properties=var_27)
    var_34 = 'a'
    var_35 = 'b'
    var_36 = 'c'
    var_37 = 1
    var_38 = 2
    var_39 = 3
    var_40 = {var_34: var_37, var_35: var_38, var_36: var_39}
    var_41 = var_33.validate(var_40)
    var_42 = 0
    var_43 = module_0.Integer(minimum=var_42)
    var_44 = {var_35: var_43}
    var_45 = module_0.Object(properties=var_44)
    var_46 = 'age'
    var_47 = -5
    var_48 = {var_46: var_47}
    var_49 = var_45.validate(var_48)
    var_50 = '^S_'
    var_51 = '^I_'
    var_52 = module_0.String()
    var_53 = module_0.Integer()
    var_54 = {var_50: var_52, var_51: var_53}
    var_55 = module_0.Object(pattern_properties=var_54)
    var_56 = 'S_name'
    var_57 = 'I_age'
    var_58 = {var_56: var_39, var_57: var_40}
    var_59 = var_55.validate(var_58)
    var_60 = module_0.String()
    var_61 = {var_46: var_60}
    var_62 = False
    var_63 = module_0.Object(properties=var_61, additional_properties=var_62)
    var_64 = 'name'
    var_65 = 'age'
    var_66 = 'John'
    var_67 = 30
    var_68 = {var_64: var_66, var_65: var_67}
    var_69 = var_63.validate(var_68)
    var_70 = module_0.String()
    var_71 = {var_64: var_70}
    var_72 = module_0.Integer()
    var_73 = module_0.Object(properties=var_71, additional_properties=var_72)
    var_74 = {var_64: var_69, var_65: var_40}
    var_75 = var_73.validate(var_74)
    var_76 = '^[a-z]+$'
    var_77 = module_0.String(pattern=var_76)
    var_78 = module_0.Object(property_names=var_77)
    var_79 = 'Name'
    var_80 = 'John'
    var_81 = {var_79: var_80}
    var_82 = var_78.validate(var_81)
    var_83 = 'Anonymous'
    var_84 = module_0.String()
    var_85 = {var_79: var_84}
    var_86 = module_0.Object(properties=var_85)
    var_87 = {}
    var_88 = var_86.validate(var_87)
    var_89 = 'address'
    var_90 = 'street'
    var_91 = 'city'
    var_92 = module_0.String()
    var_93 = module_0.String()
    var_94 = {var_90: var_92, var_91: var_93}
    var_95 = module_0.Object(properties=var_94)
    var_96 = {var_89: var_95}
    var_97 = module_0.Object(properties=var_96)
    var_98 = '123 Main'
    var_99 = 'Springfield'
    var_100 = {var_90: var_98, var_91: var_99}
    var_101 = {var_89: var_100}
    var_102 = var_97.validate(var_101)



# Parsed testcases at query #38
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
    var_10 = module_0.Array()
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = 'not a list'
    var_14 = var_10.validate(var_13)
    var_15 = module_0.Array(min_items=var_14)
    var_16 = [var_6, var_14]
    var_17 = var_15.validate(var_16)
    var_18 = 1
    var_19 = [var_18]
    var_20 = var_15.validate(var_19)
    var_21 = module_0.Array(max_items=var_19)
    var_22 = [var_6, var_19]
    var_23 = var_21.validate(var_22)
    var_24 = 1
    var_25 = 2
    var_26 = 3
    var_27 = [var_24, var_25, var_26]
    var_28 = var_21.validate(var_27)
    var_29 = module_0.Array(exact_items=var_25)
    var_30 = [var_6, var_25]
    var_31 = var_29.validate(var_30)
    var_32 = 1
    var_33 = [var_32]
    var_34 = var_29.validate(var_33)
    var_35 = 1
    var_36 = 2
    var_37 = 3
    var_38 = [var_35, var_36, var_37]
    var_39 = var_29.validate(var_38)
    var_40 = True
    var_41 = module_0.Array(unique_items=var_40)
    var_42 = [var_40, var_36, var_37]
    var_43 = var_41.validate(var_42)
    var_44 = 1
    var_45 = 2
    var_46 = [var_44, var_45, var_45]
    var_47 = var_41.validate(var_46)
    var_48 = module_0.Integer()
    var_49 = module_0.Array(var_48)
    var_50 = [var_40, var_45, var_46]
    var_51 = var_49.validate(var_50)
    var_52 = 1
    var_53 = 'two'
    var_54 = 3
    var_55 = [var_52, var_53, var_54]
    var_56 = var_49.validate(var_55)
    var_57 = module_0.Integer()
    var_58 = module_0.Integer()
    var_59 = [var_57, var_58]
    var_60 = False
    var_61 = module_0.Array(var_59, var_60)
    var_62 = [var_40, var_53]
    var_63 = var_61.validate(var_62)
    var_64 = 1
    var_65 = 2
    var_66 = 3
    var_67 = [var_64, var_65, var_66]
    var_68 = var_61.validate(var_67)
    var_69 = module_0.Integer()
    var_70 = module_0.Integer()
    var_71 = [var_69, var_70]
    var_72 = module_0.String()
    var_73 = module_0.Array(var_71, var_72)
    var_74 = 'three'
    var_75 = [var_40, var_65, var_74]
    var_76 = var_73.validate(var_75)
    var_77 = 1
    var_78 = 2
    var_79 = 3
    var_80 = [var_77, var_78, var_79]
    var_81 = var_73.validate(var_80)
    var_82 = [var_40, var_78, var_79]
    var_83 = var_10.serialize(var_82)
    var_84 = [var_40, var_78, var_79]
    var_85 = var_49.serialize(var_84)



# Parsed testcases at query #39
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Number()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None
    var_4 = module_0.Number()
    var_5 = None
    var_6 = var_4.validate(var_5)
    var_7 = module_0.Number(coerce_types=var_5)
    var_8 = ''
    var_9 = var_7.validate(var_8)
    assert var_9 is None
    var_10 = module_0.Number()
    var_11 = True
    var_12 = var_10.validate(var_11)
    var_13 = 1.5
    var_14 = var_10.validate(var_13)
    var_15 = False
    var_16 = module_0.Number(coerce_types=var_15)
    var_17 = '123'
    var_18 = var_16.validate(var_17)
    var_19 = module_0.Number(coerce_types=var_17)
    var_20 = '123'
    var_21 = var_19.validate(var_20)
    assert var_21 == 123
    var_22 = module_0.Number()
    var_23 = 'inf'
    var_24 = float(var_23)
    var_25 = var_22.validate(var_24)
    var_26 = '-inf'
    var_27 = float(var_26)
    var_28 = var_22.validate(var_27)
    var_29 = 'nan'
    var_30 = float(var_29)
    var_31 = var_22.validate(var_30)
    var_32 = '0.01'
    var_33 = module_0.Number(precision=var_32)
    var_34 = '1.234'
    var_35 = var_33.validate(var_34)
    var_36 = 5
    var_37 = module_0.Number(minimum=var_36)
    var_38 = var_37.validate(var_36)
    assert var_38 == 5
    var_39 = 4
    var_40 = var_37.validate(var_39)
    var_41 = module_0.Number(exclusive_minimum=var_36)
    var_42 = 6
    var_43 = var_41.validate(var_42)
    assert var_43 == 6
    var_44 = 5
    var_45 = var_41.validate(var_44)
    var_46 = 10
    var_47 = module_0.Number(maximum=var_46)
    var_48 = var_47.validate(var_46)
    assert var_48 == 10
    var_49 = 11
    var_50 = var_47.validate(var_49)
    var_51 = module_0.Number(exclusive_maximum=var_46)
    var_52 = 9
    var_53 = var_51.validate(var_52)
    assert var_53 == 9
    var_54 = 10
    var_55 = var_51.validate(var_54)
    var_56 = 3
    var_57 = module_0.Number(multiple_of=var_56)
    var_58 = var_57.validate(var_42)
    assert var_58 == 6
    var_59 = 7
    var_60 = var_57.validate(var_59)
    var_61 = 0.5
    var_62 = module_0.Number(multiple_of=var_61)
    var_63 = var_62.validate(var_59)
    var_64 = 1.1
    var_65 = var_62.validate(var_64)
    var_66 = module_0.Number()
    var_67 = 123
    var_68 = var_66.validate(var_67)
    assert var_68 == 123
    var_69 = 12.3
    var_70 = var_66.validate(var_69)



# Parsed testcases at query #40
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
    var_5 = module_0.Boolean()
    var_6 = None
    var_7 = var_5.validate(var_6)
    assert var_7 is None
    var_8 = module_0.Boolean()
    var_9 = None
    var_10 = var_8.validate(var_9)
    var_11 = module_0.Boolean(coerce_types=var_9)
    var_12 = 'true'
    var_13 = var_11.validate(var_12)
    assert var_13 is True
    var_14 = 'false'
    var_15 = var_11.validate(var_14)
    assert var_15 is False
    var_16 = 'on'
    var_17 = var_11.validate(var_16)
    assert var_17 is True
    var_18 = 'off'
    var_19 = var_11.validate(var_18)
    assert var_19 is False
    var_20 = '1'
    var_21 = var_11.validate(var_20)
    assert var_21 is True
    var_22 = '0'
    var_23 = var_11.validate(var_22)
    assert var_23 is False
    var_24 = ''
    var_25 = var_11.validate(var_24)
    assert var_25 is False
    var_26 = var_11.validate(var_9)
    assert var_26 is True
    var_27 = var_11.validate(var_3)
    assert var_27 is False
    var_28 = module_0.Boolean(coerce_types=var_9)
    var_29 = var_28.validate(var_24)
    assert var_29 is None
    var_30 = 'null'
    var_31 = var_28.validate(var_30)
    assert var_31 is None
    var_32 = 'none'
    var_33 = var_28.validate(var_32)
    assert var_33 is None
    var_34 = module_0.Boolean(coerce_types=var_3)
    var_35 = 'true'
    var_36 = var_34.validate(var_35)
    var_37 = 1
    var_38 = var_34.validate(var_37)



# Parsed testcases at query #41
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Const(var_0)
    var_2 = None
    var_3 = module_0.Const(var_2)
    var_4 = 42
    var_5 = True
    var_6 = module_0.Const(var_4)



# Parsed testcases at query #42
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
    var_17 = module_0.Array(min_items=var_16)
    var_18 = [var_6, var_16]
    var_19 = var_17.validate(var_18)
    var_20 = 1
    var_21 = [var_20]
    var_22 = var_17.validate(var_21)
    var_23 = module_0.Array(max_items=var_21)
    var_24 = [var_6, var_21]
    var_25 = var_23.validate(var_24)
    var_26 = 1
    var_27 = 2
    var_28 = 3
    var_29 = [var_26, var_27, var_28]
    var_30 = var_23.validate(var_29)
    var_31 = module_0.Array(exact_items=var_27)
    var_32 = [var_6, var_27]
    var_33 = var_31.validate(var_32)
    var_34 = 1
    var_35 = [var_34]
    var_36 = var_31.validate(var_35)
    var_37 = 1
    var_38 = 2
    var_39 = 3
    var_40 = [var_37, var_38, var_39]
    var_41 = var_31.validate(var_40)
    var_42 = True
    var_43 = module_0.Array(unique_items=var_42)
    var_44 = [var_42, var_38, var_39]
    var_45 = var_43.validate(var_44)
    var_46 = 1
    var_47 = 2
    var_48 = [var_46, var_47, var_47]
    var_49 = var_43.validate(var_48)
    var_50 = module_0.Integer()
    var_51 = module_0.Array(var_50)
    var_52 = [var_42, var_47, var_48]
    var_53 = var_51.validate(var_52)
    var_54 = 1
    var_55 = 'two'
    var_56 = 3
    var_57 = [var_54, var_55, var_56]
    var_58 = var_51.validate(var_57)
    var_59 = module_0.Integer()
    var_60 = module_0.String()
    var_61 = module_0.Boolean()
    var_62 = [var_59, var_60, var_61]
    var_63 = module_0.Array(var_62)
    var_64 = 'two'
    var_65 = True
    var_66 = [var_42, var_64, var_65]
    var_67 = var_63.validate(var_66)
    var_68 = 1
    var_69 = 2
    var_70 = True
    var_71 = [var_68, var_69, var_70]
    var_72 = var_63.validate(var_71)
    var_73 = module_0.Integer()
    var_74 = module_0.String()
    var_75 = [var_73, var_74]
    var_76 = module_0.Array(var_75, var_10)
    var_77 = [var_65, var_64]
    var_78 = var_76.validate(var_77)
    var_79 = 1
    var_80 = 'two'
    var_81 = True
    var_82 = [var_79, var_80, var_81]
    var_83 = var_76.validate(var_82)
    var_84 = module_0.Integer()
    var_85 = module_0.String()
    var_86 = [var_84, var_85]
    var_87 = module_0.Boolean()
    var_88 = module_0.Array(var_86, var_87)
    var_89 = True
    var_90 = [var_65, var_64, var_89]
    var_91 = var_88.validate(var_90)
    var_92 = 1
    var_93 = 'two'
    var_94 = 'three'
    var_95 = [var_92, var_93, var_94]
    var_96 = var_88.validate(var_95)
    var_97 = module_0.Array(min_items=var_89)
    var_98 = []
    var_99 = var_97.validate(var_98)
    var_100 = module_0.Array(min_items=var_10)
    var_101 = []
    var_102 = var_100.validate(var_101)
    var_103 = module_0.Integer()
    var_104 = module_0.Array(var_103)
    var_105 = [var_89, var_99, var_94]
    var_106 = var_104.serialize(var_105)
    var_107 = module_0.Integer()
    var_108 = module_0.String()
    var_109 = module_0.Boolean()
    var_110 = [var_107, var_108, var_109]
    var_111 = module_0.Array(var_110)
    var_112 = True
    var_113 = [var_89, var_64, var_112]
    var_114 = var_111.serialize(var_113)
    var_115 = module_0.Array()
    var_116 = var_115.serialize(var_8)
    assert var_116 is None



# Parsed testcases at query #43
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'Option A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'Option B'
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
    var_13 = [var_12]
    var_14 = True
    var_15 = module_0.Choice(choices=var_13)
    var_16 = None
    var_17 = var_15.validate(var_16)
    assert var_17 is None
    var_18 = None
    var_19 = var_7.validate(var_18)
    var_20 = (var_18, var_19)
    var_21 = [var_20]
    var_22 = module_0.Choice(choices=var_21, coerce_types=var_14)
    var_23 = ''
    var_24 = var_22.validate(var_23)
    assert var_24 is None
    var_25 = ''
    var_26 = var_7.validate(var_25)
    var_27 = (var_25, var_26)
    var_28 = (var_3, var_4)
    var_29 = [var_27, var_28]
    var_30 = module_0.Choice(choices=var_29)
    var_31 = var_30.validate(var_25)
    assert var_31 == 'a'
    var_32 = var_30.validate(var_3)
    assert var_32 == 'b'
    var_33 = [var_25, var_26]
    var_34 = [var_3, var_4]
    var_35 = [var_33, var_34]
    var_36 = module_0.Choice(choices=var_35)
    var_37 = var_36.validate(var_25)
    assert var_37 == 'a'
    var_38 = var_36.validate(var_3)
    assert var_38 == 'b'



# Parsed testcases at query #44
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'Option A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'Option B'
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
    var_13 = [var_12]
    var_14 = True
    var_15 = module_0.Choice(choices=var_13)
    var_16 = None
    var_17 = var_15.validate(var_16)
    assert var_17 is None
    var_18 = None
    var_19 = var_7.validate(var_18)
    var_20 = (var_18, var_19)
    var_21 = [var_20]
    var_22 = module_0.Choice(choices=var_21, coerce_types=var_14)
    var_23 = ''
    var_24 = var_22.validate(var_23)
    assert var_24 is None
    var_25 = ''
    var_26 = var_7.validate(var_25)
    var_27 = (var_25, var_26)
    var_28 = (var_3, var_4)
    var_29 = [var_27, var_28]
    var_30 = module_0.Choice(choices=var_29)
    var_31 = var_30.validate(var_25)
    assert var_31 == 'a'
    var_32 = [var_25, var_26]
    var_33 = [var_3, var_4]
    var_34 = [var_32, var_33]
    var_35 = module_0.Choice(choices=var_34)
    var_36 = var_35.validate(var_25)
    assert var_36 == 'a'



# Parsed testcases at query #45
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = 42
    var_5 = var_3.validate(var_4)
    assert var_5 == 42
    var_6 = 'hello'
    var_7 = var_3.validate(var_6)
    assert var_7 == 'hello'
    var_8 = module_0.Integer()
    var_9 = module_0.Float()
    var_10 = [var_8, var_9]
    var_11 = module_0.Union(var_10)
    var_12 = var_11.validate(var_4)
    assert var_12 == 42
    var_13 = module_0.Integer()
    var_14 = module_0.Float()
    var_15 = [var_13, var_14]
    var_16 = module_0.Union(var_15)
    var_17 = 'not a number'
    var_18 = var_16.validate(var_17)
    var_19 = True
    var_20 = module_0.Integer()
    var_21 = module_0.String()
    var_22 = [var_20, var_21]
    var_23 = module_0.Union(var_22)
    var_24 = None
    var_25 = var_23.validate(var_24)
    assert var_25 is None
    var_26 = module_0.Integer()
    var_27 = module_0.String()
    var_28 = [var_26, var_27]
    var_29 = module_0.Union(var_28)
    var_30 = None
    var_31 = var_29.validate(var_30)
    var_32 = module_0.Integer()
    var_33 = module_0.Float()
    var_34 = [var_32, var_33]
    var_35 = module_0.Union(var_34)
    var_36 = module_0.String()
    var_37 = [var_35, var_36]
    var_38 = module_0.Union(var_37)
    var_39 = var_38.validate(var_4)
    assert var_39 == 42
    var_40 = var_38.validate(var_6)
    assert var_40 == 'hello'
    var_41 = 'not'
    var_42 = 'valid'
    var_43 = [var_41, var_42]
    var_44 = var_38.validate(var_43)
    var_45 = 'a'
    var_46 = module_0.Integer()
    var_47 = {var_45: var_46}
    var_48 = module_0.Object(properties=var_47)
    var_49 = module_0.String()
    var_50 = module_0.Array(var_49)
    var_51 = [var_48, var_50]
    var_52 = module_0.Union(var_51)
    var_53 = {var_45: var_19}
    var_54 = var_52.validate(var_53)
    var_55 = 'world'
    var_56 = [var_6, var_55]
    var_57 = var_52.validate(var_56)
    var_58 = 'b'
    var_59 = 1
    var_60 = {var_58: var_59}
    var_61 = var_52.validate(var_60)
    var_62 = 0
    var_63 = module_0.Integer(minimum=var_62)
    var_64 = module_0.Integer(maximum=var_62)
    var_65 = [var_63, var_64]
    var_66 = module_0.Union(var_65)
    var_67 = -1
    var_68 = var_66.validate(var_67)
    var_69 = exc_info.value.messages()[var_62]
    var_70 = var_69.code
    assert var_70 == 'minimum'
    var_71 = module_0.Integer(minimum=var_62)
    var_72 = -1
    var_73 = module_0.Integer(maximum=var_72)
    var_74 = [var_71, var_73]
    var_75 = module_0.Union(var_74)
    var_76 = -1
    var_77 = var_75.validate(var_76)
    var_78 = exc_info.value.messages()[var_62]
    var_79 = var_78.code
    assert var_79 == 'union'



# Parsed testcases at query #46
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'Option A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'Option B'
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
    var_13 = [var_12]
    var_14 = True
    var_15 = module_0.Choice(choices=var_13)
    var_16 = None
    var_17 = var_15.validate(var_16)
    assert var_17 is None
    var_18 = None
    var_19 = var_7.validate(var_18)
    var_20 = (var_18, var_19)
    var_21 = [var_20]
    var_22 = module_0.Choice(choices=var_21, coerce_types=var_14)
    var_23 = ''
    var_24 = var_22.validate(var_23)
    assert var_24 is None
    var_25 = ''
    var_26 = var_7.validate(var_25)
    var_27 = (var_25, var_26)
    var_28 = (var_3, var_4)
    var_29 = [var_27, var_28]
    var_30 = module_0.Choice(choices=var_29)
    var_31 = var_30.validate(var_25)
    assert var_31 == 'a'
    var_32 = var_30.validate(var_3)
    assert var_32 == 'b'
    var_33 = [var_25, var_26]
    var_34 = [var_3, var_4]
    var_35 = [var_33, var_34]
    var_36 = module_0.Choice(choices=var_35)
    var_37 = var_36.validate(var_25)
    assert var_37 == 'a'
    var_38 = var_36.validate(var_3)
    assert var_38 == 'b'



# Parsed testcases at query #47
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = True
    var_5 = module_0.String()
    var_6 = module_0.Integer()
    var_7 = [var_5, var_6]
    var_8 = module_0.Union(var_7)
    var_9 = []
    var_10 = module_0.Union(var_9)



# Parsed testcases at query #48
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = 42
    var_5 = var_3.validate(var_4)
    assert var_5 == 42
    var_6 = 'hello'
    var_7 = var_3.validate(var_6)
    assert var_7 == 'hello'
    var_8 = module_0.Integer()
    var_9 = module_0.Boolean()
    var_10 = [var_8, var_9]
    var_11 = module_0.Union(var_10)
    var_12 = 'not_a_number_or_boolean'
    var_13 = var_11.validate(var_12)
    var_14 = True
    var_15 = module_0.Integer()
    var_16 = module_0.String()
    var_17 = [var_15, var_16]
    var_18 = module_0.Union(var_17)
    var_19 = None
    var_20 = var_18.validate(var_19)
    assert var_20 is None
    var_21 = False
    var_22 = module_0.Integer()
    var_23 = module_0.String()
    var_24 = [var_22, var_23]
    var_25 = module_0.Union(var_24)
    var_26 = None
    var_27 = var_25.validate(var_26)
    var_28 = module_0.Integer()
    var_29 = module_0.Float()
    var_30 = [var_28, var_29]
    var_31 = module_0.Union(var_30)
    var_32 = var_31.validate(var_4)
    assert var_32 == 42
    var_33 = 3.14
    var_34 = var_31.validate(var_33)
    var_35 = module_0.Integer()
    var_36 = module_0.String()
    var_37 = [var_35, var_36]
    var_38 = module_0.Union(var_37)
    var_39 = 3.14
    var_40 = var_38.validate(var_39)
    var_41 = exc_info.value.messages()[var_21]
    var_42 = var_41.code
    assert var_42 == 'type'
    var_43 = module_0.Integer(minimum=var_21)
    var_44 = 5
    var_45 = module_0.String(min_length=var_44)
    var_46 = [var_43, var_45]
    var_47 = module_0.Union(var_46)
    var_48 = -1
    var_49 = var_47.validate(var_48)
    var_50 = exc_info.value.messages()[var_21]
    var_51 = var_50.code
    assert var_51 == 'minimum'
    var_52 = []
    var_53 = module_0.Union(var_52)
    var_54 = 'anything'
    var_55 = var_53.validate(var_54)
    var_56 = exc_info.value.messages()[var_21]
    var_57 = var_56.code
    assert var_57 == 'union'
    var_58 = 'name'
    var_59 = module_0.String()
    var_60 = {var_58: var_59}
    var_61 = module_0.Object(properties=var_60)
    var_62 = module_0.Integer()
    var_63 = module_0.Array(var_62)
    var_64 = [var_61, var_63]
    var_65 = module_0.Union(var_64)
    var_66 = 'test'
    var_67 = {var_58: var_66}
    var_68 = var_65.validate(var_67)
    var_69 = 2
    var_70 = 3
    var_71 = [var_14, var_69, var_70]
    var_72 = var_65.validate(var_71)
    var_73 = 'invalid'
    var_74 = var_65.validate(var_73)



# Parsed testcases at query #49
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Object(properties=var_4)
    var_6 = 'John'
    var_7 = 30
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = True
    var_11 = module_0.Object()
    var_12 = None
    var_13 = var_11.validate(var_12)
    assert var_13 is None
    var_14 = module_0.Object()
    var_15 = None
    var_16 = var_14.validate(var_15)
    var_17 = 'not a dict'
    var_18 = var_14.validate(var_17)
    var_19 = 123
    var_20 = 'value'
    var_21 = {var_19: var_20}
    var_22 = var_14.validate(var_21)
    var_23 = 3
    var_24 = module_0.String(min_length=var_23)
    var_25 = {var_19: var_24}
    var_26 = module_0.Object(properties=var_25)
    var_27 = 'name'
    var_28 = 'ab'
    var_29 = {var_27: var_28}
    var_30 = var_26.validate(var_29)
    var_31 = module_0.String()
    var_32 = {var_27: var_31}
    var_33 = [var_27]
    var_34 = module_0.Object(properties=var_32, required=var_33)
    var_35 = {}
    var_36 = var_34.validate(var_35)
    var_37 = 2
    var_38 = module_0.Object(min_properties=var_37)
    var_39 = 'a'
    var_40 = 1
    var_41 = {var_39: var_40}
    var_42 = var_38.validate(var_41)
    var_43 = module_0.Object(max_properties=var_37)
    var_44 = 'a'
    var_45 = 'b'
    var_46 = 'c'
    var_47 = 1
    var_48 = 2
    var_49 = 3
    var_50 = {var_44: var_47, var_45: var_48, var_46: var_49}
    var_51 = var_43.validate(var_50)
    var_52 = module_0.String()
    var_53 = {var_44: var_52}
    var_54 = False
    var_55 = module_0.Object(properties=var_53, additional_properties=var_54)
    var_56 = 'name'
    var_57 = 'age'
    var_58 = 'John'
    var_59 = 30
    var_60 = {var_56: var_58, var_57: var_59}
    var_61 = var_55.validate(var_60)
    var_62 = module_0.String()
    var_63 = {var_56: var_62}
    var_64 = module_0.Integer()
    var_65 = module_0.Object(properties=var_63, additional_properties=var_64)
    var_66 = {var_56: var_61, var_57: var_50}
    var_67 = var_65.validate(var_66)
    var_68 = 'name'
    var_69 = 'age'
    var_70 = 'John'
    var_71 = 'thirty'
    var_72 = {var_68: var_70, var_69: var_71}
    var_73 = var_65.validate(var_72)
    var_74 = '^[a-z]+$'
    var_75 = module_0.String(pattern=var_74)
    var_76 = module_0.Object(property_names=var_75)
    var_77 = 'Name'
    var_78 = 'John'
    var_79 = {var_77: var_78}
    var_80 = var_76.validate(var_79)
    var_81 = module_0.String()
    var_82 = {var_77: var_81}
    var_83 = '^age_'
    var_84 = module_0.Integer()
    var_85 = {var_83: var_84}
    var_86 = module_0.Object(properties=var_82, pattern_properties=var_85)
    var_87 = 'age_1'
    var_88 = {var_77: var_73, var_87: var_50}
    var_89 = var_86.validate(var_88)
    var_90 = 'Anonymous'
    var_91 = module_0.String()
    var_92 = {var_77: var_91}
    var_93 = module_0.Object(properties=var_92)
    var_94 = {}
    var_95 = var_93.validate(var_94)
    var_96 = 'user'
    var_97 = module_0.String()
    var_98 = {var_77: var_97}
    var_99 = module_0.Object(properties=var_98)
    var_100 = {var_96: var_99}
    var_101 = module_0.Object(properties=var_100)
    var_102 = {var_77: var_73}
    var_103 = {var_96: var_102}
    var_104 = var_101.validate(var_103)



# Parsed testcases at query #50
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Object(properties=var_4)
    var_6 = 'John'
    var_7 = 30
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = True
    var_11 = module_0.Object()
    var_12 = None
    var_13 = var_11.validate(var_12)
    assert var_13 is None
    var_14 = module_0.Object()
    var_15 = None
    var_16 = var_14.validate(var_15)
    var_17 = 'not an object'
    var_18 = var_14.validate(var_17)
    var_19 = 123
    var_20 = 'value'
    var_21 = {var_19: var_20}
    var_22 = var_14.validate(var_21)
    var_23 = module_0.Object(min_properties=var_10)
    var_24 = {}
    var_25 = var_23.validate(var_24)
    var_26 = 2
    var_27 = module_0.Object(max_properties=var_26)
    var_28 = 'a'
    var_29 = 'b'
    var_30 = 'c'
    var_31 = 1
    var_32 = 2
    var_33 = 3
    var_34 = {var_28: var_31, var_29: var_32, var_30: var_33}
    var_35 = var_27.validate(var_34)
    var_36 = module_0.String()
    var_37 = {var_28: var_36}
    var_38 = [var_28]
    var_39 = module_0.Object(properties=var_37, required=var_38)
    var_40 = {}
    var_41 = var_39.validate(var_40)
    var_42 = module_0.Integer()
    var_43 = {var_41: var_42}
    var_44 = module_0.Object(properties=var_43)
    var_45 = 'age'
    var_46 = 'not an integer'
    var_47 = {var_45: var_46}
    var_48 = var_44.validate(var_47)
    var_49 = module_0.String()
    var_50 = {var_45: var_49}
    var_51 = False
    var_52 = module_0.Object(properties=var_50, additional_properties=var_51)
    var_53 = 'name'
    var_54 = 'age'
    var_55 = 'John'
    var_56 = 30
    var_57 = {var_53: var_55, var_54: var_56}
    var_58 = var_52.validate(var_57)
    var_59 = module_0.String()
    var_60 = {var_53: var_59}
    var_61 = module_0.Integer()
    var_62 = module_0.Object(properties=var_60, additional_properties=var_61)
    var_63 = {var_53: var_58, var_54: var_34}
    var_64 = var_62.validate(var_63)
    var_65 = 'name'
    var_66 = 'age'
    var_67 = 'John'
    var_68 = 'not an integer'
    var_69 = {var_65: var_67, var_66: var_68}
    var_70 = var_62.validate(var_69)
    var_71 = module_0.String()
    var_72 = {var_65: var_71}
    var_73 = '^[a-z]+$'
    var_74 = module_0.String(pattern=var_73)
    var_75 = module_0.Object(properties=var_72, property_names=var_74)
    var_76 = {var_65: var_70}
    var_77 = var_75.validate(var_76)
    var_78 = 'Name'
    var_79 = 'John'
    var_80 = {var_78: var_79}
    var_81 = var_75.validate(var_80)
    var_82 = module_0.String()
    var_83 = {var_78: var_82}
    var_84 = '^age_'
    var_85 = module_0.Integer()
    var_86 = {var_84: var_85}
    var_87 = module_0.Object(properties=var_83, pattern_properties=var_86)
    var_88 = 'age_1'
    var_89 = {var_78: var_70, var_88: var_34}
    var_90 = var_87.validate(var_89)
    var_91 = 'name'
    var_92 = 'age_1'
    var_93 = 'John'
    var_94 = 'not an integer'
    var_95 = {var_91: var_93, var_92: var_94}
    var_96 = var_87.validate(var_95)
    var_97 = 'Default'
    var_98 = module_0.String()
    var_99 = {var_91: var_98}
    var_100 = module_0.Object(properties=var_99)
    var_101 = {}
    var_102 = var_100.validate(var_101)



# Parsed testcases at query #51
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'Option A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'Option B'
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
    var_13 = [var_12]
    var_14 = True
    var_15 = module_0.Choice(choices=var_13)
    var_16 = None
    var_17 = var_15.validate(var_16)
    assert var_17 is None
    var_18 = None
    var_19 = var_7.validate(var_18)
    var_20 = ''
    var_21 = var_7.validate(var_20)
    var_22 = (var_20, var_21)
    var_23 = [var_22]
    var_24 = module_0.Choice(choices=var_23, coerce_types=var_14)
    var_25 = ''
    var_26 = var_24.validate(var_25)
    assert var_26 is None
    var_27 = (var_20, var_21)
    var_28 = (var_3, var_4)
    var_29 = [var_27, var_28]
    var_30 = module_0.Choice(choices=var_29)
    var_31 = var_30.validate(var_20)
    assert var_31 == 'a'
    var_32 = [var_20, var_21]
    var_33 = [var_3, var_4]
    var_34 = [var_32, var_33]
    var_35 = module_0.Choice(choices=var_34)
    var_36 = var_35.validate(var_20)
    assert var_36 == 'a'



# Parsed testcases at query #52
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = True
    var_5 = module_0.String()
    var_6 = module_0.Integer()
    var_7 = [var_5, var_6]
    var_8 = module_0.Union(var_7)
    var_9 = module_0.String()
    var_10 = module_0.Integer()
    var_11 = [var_9, var_10]
    var_12 = module_0.Union(var_11)
    var_13 = [var_0, var_1]
    var_14 = module_0.Union(var_13)



# Parsed testcases at query #53
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Object()
    var_1 = {}
    var_2 = var_0.validate(var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_0.validate(var_5)
    var_7 = True
    var_8 = module_0.Object()
    var_9 = None
    var_10 = var_8.validate(var_9)
    assert var_10 is None
    var_11 = False
    var_12 = module_0.Object()
    var_13 = None
    var_14 = var_12.validate(var_13)
    var_15 = 'not a dict'
    var_16 = var_0.validate(var_15)
    var_17 = 'name'
    var_18 = 'age'
    var_19 = module_0.String()
    var_20 = module_0.Integer()
    var_21 = {var_17: var_19, var_18: var_20}
    var_22 = module_0.Object(properties=var_21)
    var_23 = 'Alice'
    var_24 = 30
    var_25 = {var_17: var_23, var_18: var_24}
    var_26 = var_22.validate(var_25)
    var_27 = 'name'
    var_28 = 'age'
    var_29 = 'Alice'
    var_30 = 'not an integer'
    var_31 = {var_27: var_29, var_28: var_30}
    var_32 = var_22.validate(var_31)
    var_33 = module_0.String()
    var_34 = {var_17: var_33}
    var_35 = [var_17]
    var_36 = module_0.Object(properties=var_34, required=var_35)
    var_37 = 'Bob'
    var_38 = {var_17: var_37}
    var_39 = var_36.validate(var_38)
    var_40 = 'other'
    var_41 = 'value'
    var_42 = {var_40: var_41}
    var_43 = var_36.validate(var_42)
    var_44 = 2
    var_45 = module_0.Object(min_properties=var_7, max_properties=var_44)
    var_46 = 'a'
    var_47 = {var_46: var_7}
    var_48 = var_45.validate(var_47)
    var_49 = 'b'
    var_50 = {var_46: var_7, var_49: var_44}
    var_51 = var_45.validate(var_50)
    var_52 = {}
    var_53 = var_45.validate(var_52)
    var_54 = 'a'
    var_55 = 'b'
    var_56 = 'c'
    var_57 = 1
    var_58 = 2
    var_59 = 3
    var_60 = {var_54: var_57, var_55: var_58, var_56: var_59}
    var_61 = var_45.validate(var_60)
    var_62 = module_0.String()
    var_63 = {var_17: var_62}
    var_64 = module_0.Object(properties=var_63, additional_properties=var_11)
    var_65 = 'Charlie'
    var_66 = {var_17: var_65}
    var_67 = var_64.validate(var_66)
    var_68 = 'name'
    var_69 = 'extra'
    var_70 = 'Charlie'
    var_71 = 'value'
    var_72 = {var_68: var_70, var_69: var_71}
    var_73 = var_64.validate(var_72)
    var_74 = module_0.String()
    var_75 = {var_17: var_74}
    var_76 = module_0.Integer()
    var_77 = module_0.Object(properties=var_75, additional_properties=var_76)
    var_78 = 'Dave'
    var_79 = 25
    var_80 = {var_17: var_78, var_18: var_79}
    var_81 = var_77.validate(var_80)
    var_82 = 'name'
    var_83 = 'age'
    var_84 = 'Dave'
    var_85 = 'not an integer'
    var_86 = {var_82: var_84, var_83: var_85}
    var_87 = var_77.validate(var_86)
    var_88 = '^[a-z]+$'
    var_89 = module_0.String(pattern=var_88)
    var_90 = module_0.Object(property_names=var_89)
    var_91 = 'valid'
    var_92 = {var_91: var_85}
    var_93 = var_90.validate(var_92)
    var_94 = 'invalid_key'
    var_95 = 'value'
    var_96 = {var_94: var_95}
    var_97 = var_90.validate(var_96)
    var_98 = '^num_'
    var_99 = module_0.Integer()
    var_100 = {var_98: var_99}
    var_101 = module_0.Object(pattern_properties=var_100)
    var_102 = 'num_1'
    var_103 = 'num_2'
    var_104 = 10
    var_105 = 20
    var_106 = {var_102: var_104, var_103: var_105}
    var_107 = var_101.validate(var_106)
    var_108 = 'num_1'
    var_109 = 'not an integer'
    var_110 = {var_108: var_109}
    var_111 = var_101.validate(var_110)
    var_112 = 'Unknown'
    var_113 = module_0.String()
    var_114 = module_0.Integer()
    var_115 = {var_17: var_113, var_18: var_114}
    var_116 = module_0.Object(properties=var_115)
    var_117 = {}
    var_118 = var_116.validate(var_117)
    var_119 = 'Eve'
    var_120 = {var_17: var_119}
    var_121 = var_116.validate(var_120)



# Parsed testcases at query #54
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
    var_17 = module_0.Array(min_items=var_16)
    var_18 = [var_6, var_16]
    var_19 = var_17.validate(var_18)
    var_20 = 1
    var_21 = [var_20]
    var_22 = var_17.validate(var_21)
    var_23 = module_0.Array(max_items=var_21)
    var_24 = [var_6, var_21]
    var_25 = var_23.validate(var_24)
    var_26 = 1
    var_27 = 2
    var_28 = 3
    var_29 = [var_26, var_27, var_28]
    var_30 = var_23.validate(var_29)
    var_31 = module_0.Array(exact_items=var_27)
    var_32 = [var_6, var_27]
    var_33 = var_31.validate(var_32)
    var_34 = 1
    var_35 = [var_34]
    var_36 = var_31.validate(var_35)
    var_37 = 1
    var_38 = 2
    var_39 = 3
    var_40 = [var_37, var_38, var_39]
    var_41 = var_31.validate(var_40)
    var_42 = True
    var_43 = module_0.Array(unique_items=var_42)
    var_44 = [var_42, var_38, var_39]
    var_45 = var_43.validate(var_44)
    var_46 = 1
    var_47 = 2
    var_48 = [var_46, var_47, var_47]
    var_49 = var_43.validate(var_48)
    var_50 = module_0.Integer()
    var_51 = module_0.Array(var_50)
    var_52 = [var_42, var_47, var_48]
    var_53 = var_51.validate(var_52)
    var_54 = 1
    var_55 = 'two'
    var_56 = 3
    var_57 = [var_54, var_55, var_56]
    var_58 = var_51.validate(var_57)
    var_59 = module_0.Integer()
    var_60 = module_0.Integer()
    var_61 = [var_59, var_60]
    var_62 = module_0.Array(var_61, var_10)
    var_63 = [var_42, var_55]
    var_64 = var_62.validate(var_63)
    var_65 = 1
    var_66 = 2
    var_67 = 3
    var_68 = [var_65, var_66, var_67]
    var_69 = var_62.validate(var_68)
    var_70 = module_0.String()
    var_71 = module_0.Integer()
    var_72 = module_0.Integer()
    var_73 = [var_71, var_72]
    var_74 = module_0.Array(var_73, var_70)
    var_75 = 'three'
    var_76 = [var_42, var_66, var_75]
    var_77 = var_74.validate(var_76)
    var_78 = 1
    var_79 = 2
    var_80 = 3
    var_81 = [var_78, var_79, var_80]
    var_82 = var_74.validate(var_81)
    var_83 = module_0.Array(min_items=var_42)
    var_84 = []
    var_85 = var_83.validate(var_84)
    var_86 = module_0.Array(min_items=var_10)
    var_87 = []
    var_88 = var_86.validate(var_87)
    var_89 = module_0.Integer()
    var_90 = module_0.Array(var_89)
    var_91 = module_0.Array(var_90)
    var_92 = [var_42, var_85]
    var_93 = 4
    var_94 = [var_80, var_93]
    var_95 = [var_92, var_94]
    var_96 = var_91.validate(var_95)
    var_97 = 1
    var_98 = 2
    var_99 = [var_97, var_98]
    var_100 = 'three'
    var_101 = 4
    var_102 = [var_100, var_101]
    var_103 = [var_99, var_102]
    var_104 = var_91.validate(var_103)



# Parsed testcases at query #55
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
    var_7 = module_0.String(allow_blank=var_3)
    var_8 = ''
    var_9 = var_7.validate(var_8)
    assert var_9 == ''
    var_10 = module_0.String(trim_whitespace=var_3)
    var_11 = '  hello  '
    var_12 = var_10.validate(var_11)
    assert var_12 == 'hello'
    var_13 = 5
    var_14 = module_0.String(max_length=var_13)
    var_15 = var_14.validate(var_1)
    assert var_15 == 'hello'
    var_16 = 'hello world'
    var_17 = var_14.validate(var_16)
    var_18 = module_0.String(min_length=var_13)
    var_19 = 'hello world'
    var_20 = var_18.validate(var_19)
    assert var_20 == 'hello world'
    var_21 = 'hi'
    var_22 = var_18.validate(var_21)
    var_23 = '^[a-z]+$'
    var_24 = module_0.String(pattern=var_23)
    var_25 = var_24.validate(var_21)
    assert var_25 == 'hello'
    var_26 = 'Hello123'
    var_27 = var_24.validate(var_26)
    var_28 = 'email'
    var_29 = module_0.String(format=var_28)
    var_30 = 'test@example.com'
    var_31 = var_29.validate(var_30)
    assert var_31 == 'test@example.com'
    var_32 = 'invalid-email'
    var_33 = var_29.validate(var_32)
    var_34 = module_0.String(allow_blank=var_3, coerce_types=var_3)
    var_35 = var_34.validate(var_5)
    assert var_35 == ''
    var_36 = module_0.String()
    var_37 = 'hello\x00world'
    var_38 = var_36.validate(var_37)
    assert var_38 == 'helloworld'
    var_39 = module_0.String()
    var_40 = 123
    var_41 = var_39.validate(var_40)
    var_42 = False
    var_43 = module_0.String(allow_blank=var_42)
    var_44 = ''
    var_45 = var_43.validate(var_44)
    var_46 = module_0.String(coerce_types=var_3)
    var_47 = var_46.validate(var_8)
    assert var_47 is None



# Parsed testcases at query #56
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Const(var_0)
    var_2 = None
    var_3 = module_0.Const(var_2)
    var_4 = 'test'
    var_5 = module_0.Const(var_4)
    var_6 = 42
    var_7 = True
    var_8 = module_0.Const(var_6)



# Parsed testcases at query #57
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
    var_16 = [var_12, var_14]
    var_17 = var_15.validate(var_16)
    var_18 = 1
    var_19 = [var_18]
    var_20 = var_15.validate(var_19)
    var_21 = 4
    var_22 = module_0.Array(min_items=var_14, max_items=var_21)
    var_23 = [var_18, var_14]
    var_24 = var_22.validate(var_23)
    var_25 = 3
    var_26 = [var_18, var_14, var_25, var_21]
    var_27 = var_22.validate(var_26)
    var_28 = 1
    var_29 = [var_28]
    var_30 = var_22.validate(var_29)
    var_31 = 1
    var_32 = 2
    var_33 = 3
    var_34 = 4
    var_35 = 5
    var_36 = [var_31, var_32, var_33, var_34, var_35]
    var_37 = var_22.validate(var_36)
    var_38 = module_0.Integer()
    var_39 = module_0.Array(var_38)
    var_40 = [var_31, var_35, var_25]
    var_41 = var_39.validate(var_40)
    var_42 = 1
    var_43 = 'not an integer'
    var_44 = 3
    var_45 = [var_42, var_43, var_44]
    var_46 = var_39.validate(var_45)
    var_47 = module_0.Integer()
    var_48 = module_0.Integer()
    var_49 = [var_47, var_48]
    var_50 = module_0.Array(var_49, var_45)
    var_51 = [var_42, var_46]
    var_52 = var_50.validate(var_51)
    var_53 = 1
    var_54 = 2
    var_55 = 3
    var_56 = [var_53, var_54, var_55]
    var_57 = var_50.validate(var_56)
    var_58 = module_0.Array(unique_items=var_53)
    var_59 = [var_53, var_57, var_25]
    var_60 = var_58.validate(var_59)
    var_61 = 1
    var_62 = 2
    var_63 = [var_61, var_62, var_62]
    var_64 = var_58.validate(var_63)
    var_65 = module_0.Integer()
    var_66 = module_0.Array(var_65)
    var_67 = [var_61, var_57, var_25]
    var_68 = var_66.serialize(var_67)
    var_69 = var_66.serialize(var_62)
    assert var_69 is None



# Parsed testcases at query #58
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Object(properties=var_4)
    var_6 = 'John'
    var_7 = 30
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = True
    var_11 = module_0.Object()
    var_12 = None
    var_13 = var_11.validate(var_12)
    assert var_13 is None
    var_14 = module_0.Object()
    var_15 = 'not a dict'
    var_16 = var_14.validate(var_15)
    var_17 = module_0.String()
    var_18 = {var_15: var_17}
    var_19 = [var_15]
    var_20 = module_0.Object(properties=var_18, required=var_19)
    var_21 = 'age'
    var_22 = 30
    var_23 = {var_21: var_22}
    var_24 = var_20.validate(var_23)
    var_25 = module_0.Object(min_properties=var_10)
    var_26 = {}
    var_27 = var_25.validate(var_26)
    var_28 = 2
    var_29 = module_0.Object(max_properties=var_28)
    var_30 = 'a'
    var_31 = 'b'
    var_32 = 'c'
    var_33 = 1
    var_34 = 2
    var_35 = 3
    var_36 = {var_30: var_33, var_31: var_34, var_32: var_35}
    var_37 = var_29.validate(var_36)
    var_38 = '^[a-z]+$'
    var_39 = module_0.String(pattern=var_38)
    var_40 = module_0.Object(property_names=var_39)
    var_41 = '123'
    var_42 = 'value'
    var_43 = {var_41: var_42}
    var_44 = var_40.validate(var_43)
    var_45 = module_0.String()
    var_46 = {var_41: var_45}
    var_47 = False
    var_48 = module_0.Object(properties=var_46, additional_properties=var_47)
    var_49 = 'name'
    var_50 = 'age'
    var_51 = 'John'
    var_52 = 30
    var_53 = {var_49: var_51, var_50: var_52}
    var_54 = var_48.validate(var_53)
    var_55 = module_0.String()
    var_56 = {var_49: var_55}
    var_57 = module_0.Integer()
    var_58 = module_0.Object(properties=var_56, additional_properties=var_57)
    var_59 = {var_49: var_54, var_50: var_36}
    var_60 = var_58.validate(var_59)
    var_61 = '^S_'
    var_62 = '^I_'
    var_63 = module_0.String()
    var_64 = module_0.Integer()
    var_65 = {var_61: var_63, var_62: var_64}
    var_66 = module_0.Object(pattern_properties=var_65)
    var_67 = 'S_name'
    var_68 = 'I_age'
    var_69 = {var_67: var_54, var_68: var_36}
    var_70 = var_66.validate(var_69)
    var_71 = 'address'
    var_72 = 'city'
    var_73 = module_0.String()
    var_74 = {var_72: var_73}
    var_75 = module_0.Object(properties=var_74)
    var_76 = {var_71: var_75}
    var_77 = module_0.Object(properties=var_76)
    var_78 = 'NYC'
    var_79 = {var_72: var_78}
    var_80 = {var_71: var_79}
    var_81 = var_77.validate(var_80)



# Parsed testcases at query #59
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = True
    var_5 = module_0.String()
    var_6 = [var_5, var_1]
    var_7 = module_0.Union(var_6)
    var_8 = [var_0, var_1]
    var_9 = module_0.Union(var_8)



# Parsed testcases at query #60
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
    var_14 = 3
    var_15 = module_0.Array(min_items=var_14)
    var_16 = 1
    var_17 = 2
    var_18 = [var_16, var_17]
    var_19 = var_15.validate(var_18)
    var_20 = 2
    var_21 = module_0.Array(max_items=var_20)
    var_22 = 1
    var_23 = 2
    var_24 = 3
    var_25 = [var_22, var_23, var_24]
    var_26 = var_21.validate(var_25)
    var_27 = module_0.Array(exact_items=var_20)
    var_28 = 1
    var_29 = [var_28]
    var_30 = var_27.validate(var_29)
    var_31 = module_0.Array(exact_items=var_20)
    var_32 = [var_28, var_20]
    var_33 = var_31.validate(var_32)
    var_34 = module_0.Integer()
    var_35 = module_0.Array(var_34)
    var_36 = 1
    var_37 = 'not an integer'
    var_38 = 3
    var_39 = [var_36, var_37, var_38]
    var_40 = var_35.validate(var_39)
    var_41 = module_0.Integer()
    var_42 = module_0.String()
    var_43 = [var_41, var_42]
    var_44 = module_0.Array(var_43)
    var_45 = 1
    var_46 = 2
    var_47 = [var_45, var_46]
    var_48 = var_44.validate(var_47)
    var_49 = module_0.Integer()
    var_50 = module_0.String()
    var_51 = [var_49, var_50]
    var_52 = module_0.Array(var_51)
    var_53 = 'two'
    var_54 = [var_45, var_53]
    var_55 = var_52.validate(var_54)
    var_56 = module_0.Integer()
    var_57 = module_0.String()
    var_58 = [var_56, var_57]
    var_59 = module_0.Array(var_58, var_48)
    var_60 = 1
    var_61 = 'two'
    var_62 = 3
    var_63 = [var_60, var_61, var_62]
    var_64 = var_59.validate(var_63)
    var_65 = module_0.Integer()
    var_66 = module_0.String()
    var_67 = [var_65, var_66]
    var_68 = module_0.Integer()
    var_69 = module_0.Array(var_67, var_68)
    var_70 = [var_60, var_53, var_64]
    var_71 = var_69.validate(var_70)
    var_72 = module_0.Array(unique_items=var_60)
    var_73 = 1
    var_74 = 2
    var_75 = [var_73, var_74, var_74]
    var_76 = var_72.validate(var_75)
    var_77 = module_0.Array(unique_items=var_73)
    var_78 = [var_73, var_20, var_64]
    var_79 = var_77.validate(var_78)
    var_80 = module_0.Array()
    var_81 = [var_73, var_53, var_64]
    var_82 = var_80.validate(var_81)
    var_83 = module_0.Integer()
    var_84 = module_0.Array(var_83)
    var_85 = module_0.Array(var_84)
    var_86 = [var_73, var_20]
    var_87 = 4
    var_88 = [var_64, var_87]
    var_89 = [var_86, var_88]
    var_90 = var_85.validate(var_89)
    var_91 = module_0.Integer()
    var_92 = module_0.Array(var_91)
    var_93 = module_0.Array(var_92)
    var_94 = 1
    var_95 = 2
    var_96 = [var_94, var_95]
    var_97 = 'invalid'
    var_98 = [var_97]
    var_99 = [var_96, var_98]
    var_100 = var_93.validate(var_99)



# Parsed testcases at query #61
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
    var_5 = module_0.Boolean()
    var_6 = None
    var_7 = var_5.validate(var_6)
    assert var_7 is None
    var_8 = module_0.Boolean()
    var_9 = None
    var_10 = var_8.validate(var_9)
    var_11 = module_0.Boolean(coerce_types=var_9)
    var_12 = 'true'
    var_13 = var_11.validate(var_12)
    assert var_13 is True
    var_14 = 'false'
    var_15 = var_11.validate(var_14)
    assert var_15 is False
    var_16 = 'on'
    var_17 = var_11.validate(var_16)
    assert var_17 is True
    var_18 = 'off'
    var_19 = var_11.validate(var_18)
    assert var_19 is False
    var_20 = '1'
    var_21 = var_11.validate(var_20)
    assert var_21 is True
    var_22 = '0'
    var_23 = var_11.validate(var_22)
    assert var_23 is False
    var_24 = ''
    var_25 = var_11.validate(var_24)
    assert var_25 is False
    var_26 = var_11.validate(var_9)
    assert var_26 is True
    var_27 = var_11.validate(var_3)
    assert var_27 is False
    var_28 = module_0.Boolean(coerce_types=var_9)
    var_29 = var_28.validate(var_24)
    assert var_29 is None
    var_30 = 'null'
    var_31 = var_28.validate(var_30)
    assert var_31 is None
    var_32 = 'none'
    var_33 = var_28.validate(var_32)
    assert var_33 is None
    var_34 = module_0.Boolean(coerce_types=var_3)
    var_35 = 'true'
    var_36 = var_34.validate(var_35)
    var_37 = 1
    var_38 = var_34.validate(var_37)
    var_39 = 'invalid'
    var_40 = var_34.validate(var_39)



# Parsed testcases at query #62
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
    var_17 = module_0.Array(min_items=var_16)
    var_18 = [var_6, var_16]
    var_19 = var_17.validate(var_18)
    var_20 = 1
    var_21 = [var_20]
    var_22 = var_17.validate(var_21)
    var_23 = module_0.Array(max_items=var_21)
    var_24 = [var_6, var_21]
    var_25 = var_23.validate(var_24)
    var_26 = 1
    var_27 = 2
    var_28 = 3
    var_29 = [var_26, var_27, var_28]
    var_30 = var_23.validate(var_29)
    var_31 = module_0.Array(exact_items=var_27)
    var_32 = [var_6, var_27]
    var_33 = var_31.validate(var_32)
    var_34 = 1
    var_35 = [var_34]
    var_36 = var_31.validate(var_35)
    var_37 = 1
    var_38 = 2
    var_39 = 3
    var_40 = [var_37, var_38, var_39]
    var_41 = var_31.validate(var_40)
    var_42 = module_0.Array(min_items=var_6)
    var_43 = []
    var_44 = var_42.validate(var_43)
    var_45 = module_0.Integer()
    var_46 = module_0.Array(var_45)
    var_47 = '1'
    var_48 = '2'
    var_49 = '3'
    var_50 = [var_47, var_48, var_49]
    var_51 = var_46.validate(var_50)
    var_52 = '1'
    var_53 = 'two'
    var_54 = '3'
    var_55 = [var_52, var_53, var_54]
    var_56 = var_46.validate(var_55)
    var_57 = module_0.Integer()
    var_58 = module_0.Integer()
    var_59 = [var_57, var_58]
    var_60 = module_0.Array(var_59, var_10)
    var_61 = [var_6, var_53]
    var_62 = var_60.validate(var_61)
    var_63 = 1
    var_64 = 2
    var_65 = 3
    var_66 = [var_63, var_64, var_65]
    var_67 = var_60.validate(var_66)
    var_68 = module_0.Integer()
    var_69 = module_0.Integer()
    var_70 = [var_68, var_69]
    var_71 = module_0.String()
    var_72 = module_0.Array(var_70, var_71)
    var_73 = 'three'
    var_74 = [var_6, var_64, var_73]
    var_75 = var_72.validate(var_74)
    var_76 = 1
    var_77 = 2
    var_78 = 3
    var_79 = [var_76, var_77, var_78]
    var_80 = var_72.validate(var_79)
    var_81 = True
    var_82 = module_0.Array(unique_items=var_81)
    var_83 = [var_81, var_77, var_78]
    var_84 = var_82.validate(var_83)
    var_85 = 1
    var_86 = 2
    var_87 = [var_85, var_86, var_86]
    var_88 = var_82.validate(var_87)
    var_89 = module_0.Integer()
    var_90 = module_0.Array(var_89)
    var_91 = [var_81, var_86, var_87]
    var_92 = var_90.serialize(var_91)
    var_93 = var_90.serialize(var_8)
    assert var_93 is None
    var_94 = module_0.Integer()
    var_95 = module_0.String()
    var_96 = [var_94, var_95]
    var_97 = module_0.Array(var_96)
    var_98 = 'two'
    var_99 = [var_81, var_98]
    var_100 = var_97.serialize(var_99)



# Parsed testcases at query #63
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
    var_21 = module_0.Boolean(coerce_types=var_3)
    var_22 = 'true'
    var_23 = var_21.validate(var_22)
    var_24 = 1
    var_25 = var_21.validate(var_24)
    var_26 = module_0.Boolean()
    var_27 = None
    var_28 = var_26.validate(var_27)
    assert var_28 is None
    var_29 = 'null'
    var_30 = var_26.validate(var_29)
    assert var_30 is None
    var_31 = 'none'
    var_32 = var_26.validate(var_31)
    assert var_32 is None
    var_33 = None
    var_34 = var_0.validate(var_33)
    var_35 = 'invalid'
    var_36 = var_0.validate(var_35)
    var_37 = 2
    var_38 = var_0.validate(var_37)



# Parsed testcases at query #64
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
    var_11 = module_0.Number()
    var_12 = True
    var_13 = var_11.validate(var_12)
    var_14 = 5
    var_15 = var_11.validate(var_14)
    assert var_15 == 5
    var_16 = 5.5
    var_17 = var_11.validate(var_16)
    var_18 = module_0.Number(coerce_types=var_16)
    var_19 = '5'
    var_20 = var_18.validate(var_19)
    assert var_20 == 5
    var_21 = module_0.Number(coerce_types=var_4)
    var_22 = '5'
    var_23 = var_21.validate(var_22)
    var_24 = module_0.Number()
    var_25 = 'inf'
    var_26 = float(var_25)
    var_27 = var_24.validate(var_26)
    var_28 = '0.01'
    var_29 = module_0.Number(precision=var_28)
    var_30 = 3.14159
    var_31 = var_29.validate(var_30)
    var_32 = module_0.Number(minimum=var_14)
    var_33 = var_32.validate(var_14)
    assert var_33 == 5
    var_34 = 4
    var_35 = var_32.validate(var_34)
    var_36 = module_0.Number(exclusive_minimum=var_14)
    var_37 = 6
    var_38 = var_36.validate(var_37)
    assert var_38 == 6
    var_39 = 5
    var_40 = var_36.validate(var_39)
    var_41 = 10
    var_42 = module_0.Number(maximum=var_41)
    var_43 = var_42.validate(var_41)
    assert var_43 == 10
    var_44 = 11
    var_45 = var_42.validate(var_44)
    var_46 = module_0.Number(exclusive_maximum=var_41)
    var_47 = 9
    var_48 = var_46.validate(var_47)
    assert var_48 == 9
    var_49 = 10
    var_50 = var_46.validate(var_49)
    var_51 = 2
    var_52 = module_0.Number(multiple_of=var_51)
    var_53 = 4
    var_54 = var_52.validate(var_53)
    assert var_54 == 4
    var_55 = 5
    var_56 = var_52.validate(var_55)
    var_57 = 0.5
    var_58 = module_0.Number(multiple_of=var_57)
    var_59 = var_58.validate(var_51)
    var_60 = 2.1
    var_61 = var_58.validate(var_60)



# Parsed testcases at query #65
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'Option A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'Option B'
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
    var_13 = [var_12]
    var_14 = True
    var_15 = module_0.Choice(choices=var_13)
    var_16 = None
    var_17 = var_15.validate(var_16)
    assert var_17 is None
    var_18 = None
    var_19 = var_7.validate(var_18)
    var_20 = ''
    var_21 = var_7.validate(var_20)
    var_22 = (var_20, var_21)
    var_23 = [var_22]
    var_24 = module_0.Choice(choices=var_23, coerce_types=var_14)
    var_25 = ''
    var_26 = var_24.validate(var_25)
    assert var_26 is None
    var_27 = (var_20, var_21)
    var_28 = (var_3, var_4)
    var_29 = [var_27, var_28]
    var_30 = module_0.Choice(choices=var_29)
    var_31 = var_30.validate(var_20)
    assert var_31 == 'a'
    var_32 = var_30.validate(var_3)
    assert var_32 == 'b'
    var_33 = [var_20, var_21]
    var_34 = [var_3, var_4]
    var_35 = [var_33, var_34]
    var_36 = module_0.Choice(choices=var_35)
    var_37 = var_36.validate(var_20)
    assert var_37 == 'a'
    var_38 = var_36.validate(var_3)
    assert var_38 == 'b'



# Parsed testcases at query #66
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = True
    var_5 = module_0.String()
    var_6 = [var_5, var_1]
    var_7 = module_0.Union(var_6)
    var_8 = []
    var_9 = module_0.Union(var_8)
    var_10 = 'not a field'
    var_11 = 123
    var_12 = [var_10, var_11]
    var_13 = module_0.Union(var_12)
    var_14 = [var_0]
    var_15 = module_0.Union(var_14)



# Parsed testcases at query #67
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
    var_8 = 12.34
    var_9 = var_3.validate(var_8)
    var_10 = True
    var_11 = module_0.String()
    var_12 = module_0.Integer()
    var_13 = [var_11, var_12]
    var_14 = module_0.Union(var_13)
    var_15 = None
    var_16 = var_14.validate(var_15)
    assert var_16 is None
    var_17 = False
    var_18 = module_0.String()
    var_19 = module_0.Integer()
    var_20 = [var_18, var_19]
    var_21 = module_0.Union(var_20)
    var_22 = None
    var_23 = var_21.validate(var_22)
    var_24 = 5
    var_25 = module_0.String(min_length=var_24)
    var_26 = 10
    var_27 = module_0.Integer(minimum=var_26)
    var_28 = [var_25, var_27]
    var_29 = module_0.Union(var_28)
    var_30 = 'valid_string'
    var_31 = var_29.validate(var_30)
    assert var_31 == 'valid_string'
    var_32 = 15
    var_33 = var_29.validate(var_32)
    assert var_33 == 15
    var_34 = 'short'
    var_35 = var_29.validate(var_34)
    var_36 = 5
    var_37 = var_29.validate(var_36)
    var_38 = module_0.Boolean(coerce_types=var_10)
    var_39 = module_0.Integer()
    var_40 = [var_38, var_39]
    var_41 = module_0.Union(var_40)
    var_42 = 'true'
    var_43 = var_41.validate(var_42)
    assert var_43 is True
    var_44 = var_41.validate(var_6)
    assert var_44 == 123
    var_45 = 'invalid'
    var_46 = var_41.validate(var_45)
    var_47 = 'type'
    var_48 = 'Custom type error'
    var_49 = {var_47: var_48}
    var_50 = module_0.String()
    var_51 = {var_47: var_48}
    var_52 = module_0.Integer()
    var_53 = [var_50, var_52]
    var_54 = module_0.Union(var_53)
    var_55 = 12.34
    var_56 = var_54.validate(var_55)
    var_57 = 'name'
    var_58 = module_0.String()
    var_59 = {var_57: var_58}
    var_60 = module_0.Object(properties=var_59)
    var_61 = module_0.Integer()
    var_62 = [var_60, var_61]
    var_63 = module_0.Union(var_62)
    var_64 = {var_57: var_56}
    var_65 = var_63.validate(var_64)
    var_66 = var_63.validate(var_6)
    assert var_66 == 123
    var_67 = 'invalid'
    var_68 = 'object'
    var_69 = {var_67: var_68}
    var_70 = var_63.validate(var_69)
    var_71 = module_0.String()
    var_72 = module_0.Array(var_71)
    var_73 = module_0.Integer()
    var_74 = [var_72, var_73]
    var_75 = module_0.Union(var_74)
    var_76 = 'array'
    var_77 = [var_68, var_76]
    var_78 = var_75.validate(var_77)
    var_79 = var_75.validate(var_70)
    assert var_79 == 123
    var_80 = 123
    var_81 = 456
    var_82 = [var_80, var_81]
    var_83 = var_75.validate(var_82)



# Parsed testcases at query #68
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
    var_10 = module_0.Array()
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = module_0.Array()
    var_14 = 'not a list'
    var_15 = var_13.validate(var_14)
    var_16 = module_0.Array(min_items=var_15)
    var_17 = [var_6, var_15]
    var_18 = var_16.validate(var_17)
    var_19 = 1
    var_20 = [var_19]
    var_21 = var_16.validate(var_20)
    var_22 = module_0.Array(max_items=var_20)
    var_23 = [var_6, var_20]
    var_24 = var_22.validate(var_23)
    var_25 = 1
    var_26 = 2
    var_27 = 3
    var_28 = [var_25, var_26, var_27]
    var_29 = var_22.validate(var_28)
    var_30 = module_0.Array(exact_items=var_26)
    var_31 = [var_6, var_26]
    var_32 = var_30.validate(var_31)
    var_33 = 1
    var_34 = [var_33]
    var_35 = var_30.validate(var_34)
    var_36 = 1
    var_37 = 2
    var_38 = 3
    var_39 = [var_36, var_37, var_38]
    var_40 = var_30.validate(var_39)
    var_41 = True
    var_42 = module_0.Array(unique_items=var_41)
    var_43 = [var_41, var_37, var_38]
    var_44 = var_42.validate(var_43)
    var_45 = 1
    var_46 = 2
    var_47 = [var_45, var_46, var_46]
    var_48 = var_42.validate(var_47)
    var_49 = module_0.Integer()
    var_50 = module_0.Array(var_49)
    var_51 = '1'
    var_52 = '2'
    var_53 = '3'
    var_54 = [var_51, var_52, var_53]
    var_55 = var_50.validate(var_54)
    var_56 = '1'
    var_57 = 'two'
    var_58 = '3'
    var_59 = [var_56, var_57, var_58]
    var_60 = var_50.validate(var_59)
    var_61 = module_0.Integer()
    var_62 = module_0.String()
    var_63 = module_0.Boolean()
    var_64 = [var_61, var_62, var_63]
    var_65 = module_0.Array(var_64)
    var_66 = 'two'
    var_67 = 'true'
    var_68 = [var_51, var_66, var_67]
    var_69 = var_65.validate(var_68)
    var_70 = '1'
    var_71 = 'two'
    var_72 = 'invalid'
    var_73 = [var_70, var_71, var_72]
    var_74 = var_65.validate(var_73)
    var_75 = module_0.Integer()
    var_76 = module_0.String()
    var_77 = [var_75, var_76]
    var_78 = False
    var_79 = module_0.Array(var_77, var_78)
    var_80 = [var_51, var_66]
    var_81 = var_79.validate(var_80)
    var_82 = '1'
    var_83 = 'two'
    var_84 = 'extra'
    var_85 = [var_82, var_83, var_84]
    var_86 = var_79.validate(var_85)
    var_87 = module_0.Integer()
    var_88 = module_0.String()
    var_89 = [var_87, var_88]
    var_90 = module_0.Boolean()
    var_91 = module_0.Array(var_89, var_90)
    var_92 = [var_51, var_66, var_67]
    var_93 = var_91.validate(var_92)
    var_94 = '1'
    var_95 = 'two'
    var_96 = 'invalid'
    var_97 = [var_94, var_95, var_96]
    var_98 = var_91.validate(var_97)
    var_99 = module_0.Array(min_items=var_41)
    var_100 = []
    var_101 = var_99.validate(var_100)



# Parsed testcases at query #69
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'Option A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'Option B'
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
    var_13 = [var_12]
    var_14 = True
    var_15 = module_0.Choice(choices=var_13)
    var_16 = None
    var_17 = var_15.validate(var_16)
    assert var_17 is None
    var_18 = None
    var_19 = var_7.validate(var_18)
    var_20 = (var_18, var_19)
    var_21 = [var_20]
    var_22 = module_0.Choice(choices=var_21, coerce_types=var_14)
    var_23 = ''
    var_24 = var_22.validate(var_23)
    assert var_24 is None
    var_25 = ''
    var_26 = var_7.validate(var_25)
    var_27 = (var_25, var_26)
    var_28 = (var_3, var_4)
    var_29 = [var_27, var_28]
    var_30 = module_0.Choice(choices=var_29)
    var_31 = var_30.validate(var_25)
    assert var_31 == 'a'
    var_32 = var_30.validate(var_3)
    assert var_32 == 'b'
    var_33 = [var_25, var_26]
    var_34 = [var_3, var_4]
    var_35 = [var_33, var_34]
    var_36 = module_0.Choice(choices=var_35)
    var_37 = var_36.validate(var_25)
    assert var_37 == 'a'
    var_38 = var_36.validate(var_3)
    assert var_38 == 'b'



# Parsed testcases at query #70
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = True
    var_5 = module_0.String()
    var_6 = [var_0, var_5]
    var_7 = module_0.Union(var_6)
    var_8 = []
    var_9 = module_0.Union(var_8)
    var_10 = 'invalid_field'
    var_11 = [var_0, var_10]
    var_12 = module_0.Union(var_11)
    var_13 = [var_0, var_1]
    var_14 = module_0.Union(var_13)



# Parsed testcases at query #71
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'Option A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'Option B'
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
    var_13 = [var_12]
    var_14 = True
    var_15 = module_0.Choice(choices=var_13)
    var_16 = None
    var_17 = var_15.validate(var_16)
    assert var_17 is None
    var_18 = None
    var_19 = var_7.validate(var_18)
    var_20 = (var_18, var_19)
    var_21 = [var_20]
    var_22 = module_0.Choice(choices=var_21, coerce_types=var_14)
    var_23 = ''
    var_24 = var_22.validate(var_23)
    assert var_24 is None
    var_25 = ''
    var_26 = var_7.validate(var_25)
    var_27 = (var_25, var_26)
    var_28 = (var_3, var_4)
    var_29 = [var_27, var_28]
    var_30 = module_0.Choice(choices=var_29)
    var_31 = var_30.validate(var_25)
    assert var_31 == 'a'
    var_32 = var_30.validate(var_3)
    assert var_32 == 'b'
    var_33 = [var_25, var_26]
    var_34 = [var_3, var_4]
    var_35 = [var_33, var_34]
    var_36 = module_0.Choice(choices=var_35)
    var_37 = var_36.validate(var_25)
    assert var_37 == 'a'
    var_38 = var_36.validate(var_3)
    assert var_38 == 'b'
    var_39 = 'Option 1'
    var_40 = (var_14, var_39)
    var_41 = 2
    var_42 = 'Option 2'
    var_43 = (var_41, var_42)
    var_44 = [var_40, var_43]
    var_45 = module_0.Choice(choices=var_44)
    var_46 = var_45.validate(var_14)
    assert var_46 == 1
    var_47 = var_45.validate(var_41)
    assert var_47 == 2
    var_48 = 3
    var_49 = var_45.validate(var_48)
    var_50 = (var_48, var_49)
    var_51 = [var_50]
    var_52 = False
    var_53 = module_0.Choice(choices=var_51, coerce_types=var_52)
    var_54 = 1
    var_55 = var_53.validate(var_54)



# Parsed testcases at query #72
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
    var_17 = module_0.Array(min_items=var_16)
    var_18 = [var_6, var_16]
    var_19 = var_17.validate(var_18)
    var_20 = 1
    var_21 = [var_20]
    var_22 = var_17.validate(var_21)
    var_23 = module_0.Array(max_items=var_21)
    var_24 = [var_6, var_21]
    var_25 = var_23.validate(var_24)
    var_26 = 1
    var_27 = 2
    var_28 = 3
    var_29 = [var_26, var_27, var_28]
    var_30 = var_23.validate(var_29)
    var_31 = module_0.Array(exact_items=var_27)
    var_32 = [var_6, var_27]
    var_33 = var_31.validate(var_32)
    var_34 = 1
    var_35 = [var_34]
    var_36 = var_31.validate(var_35)
    var_37 = 1
    var_38 = 2
    var_39 = 3
    var_40 = [var_37, var_38, var_39]
    var_41 = var_31.validate(var_40)
    var_42 = module_0.Array(min_items=var_6)
    var_43 = []
    var_44 = var_42.validate(var_43)
    var_45 = module_0.Integer()
    var_46 = module_0.Array(var_45)
    var_47 = [var_6, var_44, var_39]
    var_48 = var_46.validate(var_47)
    var_49 = 1
    var_50 = 'not an integer'
    var_51 = 3
    var_52 = [var_49, var_50, var_51]
    var_53 = var_46.validate(var_52)
    var_54 = module_0.Integer()
    var_55 = module_0.Integer()
    var_56 = [var_54, var_55]
    var_57 = module_0.Array(var_56, var_10)
    var_58 = [var_6, var_50]
    var_59 = var_57.validate(var_58)
    var_60 = 1
    var_61 = 2
    var_62 = 3
    var_63 = [var_60, var_61, var_62]
    var_64 = var_57.validate(var_63)
    var_65 = module_0.Integer()
    var_66 = module_0.Integer()
    var_67 = [var_65, var_66]
    var_68 = module_0.Integer()
    var_69 = module_0.Array(var_67, var_68)
    var_70 = [var_6, var_61, var_62]
    var_71 = var_69.validate(var_70)
    var_72 = 1
    var_73 = 2
    var_74 = 'not an integer'
    var_75 = [var_72, var_73, var_74]
    var_76 = var_69.validate(var_75)
    var_77 = True
    var_78 = module_0.Array(unique_items=var_77)
    var_79 = [var_77, var_73, var_74]
    var_80 = var_78.validate(var_79)
    var_81 = 1
    var_82 = 2
    var_83 = [var_81, var_82, var_82]
    var_84 = var_78.validate(var_83)
    var_85 = module_0.Integer()
    var_86 = module_0.Array(var_85)
    var_87 = [var_77, var_82, var_83]
    var_88 = var_86.serialize(var_87)
    var_89 = var_86.serialize(var_8)
    assert var_89 is None



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Array()
    var_1 = module_0.Field()
    var_2 = module_0.Array(var_1)
    var_3 = module_0.Field()
    var_4 = module_0.Field()
    var_5 = [var_3, var_4]
    var_6 = module_0.Array(var_5)
    var_7 = module_0.Field()
    var_8 = module_0.Array(additional_items=var_7)
    var_9 = 1
    var_10 = 10
    var_11 = module_0.Array(min_items=var_9, max_items=var_10)
    var_12 = 5
    var_13 = module_0.Array(exact_items=var_12)
    var_14 = True
    var_15 = module_0.Array(unique_items=var_14)
    var_16 = True
    var_17 = module_0.Array()
    var_18 = []
    var_19 = module_0.Array()



# Parsed testcases at query #2
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None
    var_4 = module_0.String(allow_blank=var_0)
    var_5 = ''
    var_6 = var_4.validate(var_5)
    assert var_6 == ''
    var_7 = var_4.validate(var_2)
    assert var_7 == ''
    var_8 = module_0.String()
    var_9 = 123
    var_10 = var_8.validate(var_9)
    var_11 = module_0.String()
    var_12 = ''
    var_13 = var_11.validate(var_12)
    var_14 = module_0.String(trim_whitespace=var_12)
    var_15 = '  hello  '
    var_16 = var_14.validate(var_15)
    assert var_16 == 'hello'
    var_17 = 3
    var_18 = module_0.String(min_length=var_17)
    var_19 = 'abc'
    var_20 = var_18.validate(var_19)
    assert var_20 == 'abc'
    var_21 = 'ab'
    var_22 = var_18.validate(var_21)
    var_23 = module_0.String(max_length=var_17)
    var_24 = var_23.validate(var_19)
    assert var_24 == 'abc'
    var_25 = 'abcd'
    var_26 = var_23.validate(var_25)
    var_27 = '^[a-z]+$'
    var_28 = module_0.String(pattern=var_27)
    var_29 = var_28.validate(var_19)
    assert var_29 == 'abc'
    var_30 = 'abc123'
    var_31 = var_28.validate(var_30)
    var_32 = 'email'
    var_33 = module_0.String(format=var_32)
    var_34 = 'test@example.com'
    var_35 = var_33.validate(var_34)
    assert var_35 == 'test@example.com'
    var_36 = 'invalid-email'
    var_37 = var_33.validate(var_36)
    var_38 = module_0.String()
    var_39 = 'a\x00b'
    var_40 = var_38.validate(var_39)
    assert var_40 == 'ab'
    var_41 = module_0.String(coerce_types=var_36)
    var_42 = var_41.validate(var_5)
    assert var_42 is None
    var_43 = module_0.String(format=var_32)
    var_44 = var_43.serialize(var_34)
    assert var_44 == 'test@example.com'



# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Object(properties=var_4)
    var_6 = 'John'
    var_7 = 30
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = True
    var_11 = module_0.Object()
    var_12 = None
    var_13 = var_11.validate(var_12)
    assert var_13 is None
    var_14 = module_0.Object()
    var_15 = None
    var_16 = var_14.validate(var_15)
    var_17 = module_0.Object()
    var_18 = 'not a dict'
    var_19 = var_17.validate(var_18)
    var_20 = module_0.Object()
    var_21 = 123
    var_22 = 'value'
    var_23 = {var_21: var_22}
    var_24 = var_20.validate(var_23)
    var_25 = module_0.String()
    var_26 = {var_21: var_25}
    var_27 = [var_21]
    var_28 = module_0.Object(properties=var_26, required=var_27)
    var_29 = {}
    var_30 = var_28.validate(var_29)
    var_31 = 2
    var_32 = module_0.Object(min_properties=var_31)
    var_33 = 'a'
    var_34 = 1
    var_35 = {var_33: var_34}
    var_36 = var_32.validate(var_35)
    var_37 = module_0.Object(max_properties=var_31)
    var_38 = 'a'
    var_39 = 'b'
    var_40 = 'c'
    var_41 = 1
    var_42 = 2
    var_43 = 3
    var_44 = {var_38: var_41, var_39: var_42, var_40: var_43}
    var_45 = var_37.validate(var_44)
    var_46 = 0
    var_47 = 120
    var_48 = module_0.Integer(minimum=var_46, maximum=var_47)
    var_49 = {var_39: var_48}
    var_50 = module_0.Object(properties=var_49)
    var_51 = 'age'
    var_52 = -5
    var_53 = {var_51: var_52}
    var_54 = var_50.validate(var_53)
    var_55 = '^S_'
    var_56 = '^I_'
    var_57 = module_0.String()
    var_58 = module_0.Integer()
    var_59 = {var_55: var_57, var_56: var_58}
    var_60 = module_0.Object(pattern_properties=var_59)
    var_61 = 'S_name'
    var_62 = 'I_age'
    var_63 = {var_61: var_43, var_62: var_44}
    var_64 = var_60.validate(var_63)
    var_65 = module_0.String()
    var_66 = {var_51: var_65}
    var_67 = module_0.Object(properties=var_66, additional_properties=var_10)
    var_68 = 'extra'
    var_69 = 'data'
    var_70 = {var_51: var_43, var_68: var_69}
    var_71 = var_67.validate(var_70)
    var_72 = module_0.String()
    var_73 = {var_51: var_72}
    var_74 = False
    var_75 = module_0.Object(properties=var_73, additional_properties=var_74)
    var_76 = 'name'
    var_77 = 'extra'
    var_78 = 'John'
    var_79 = 'data'
    var_80 = {var_76: var_78, var_77: var_79}
    var_81 = var_75.validate(var_80)
    var_82 = module_0.String()
    var_83 = {var_76: var_82}
    var_84 = module_0.Integer()
    var_85 = module_0.Object(properties=var_83, additional_properties=var_84)
    var_86 = {var_76: var_81, var_77: var_44}
    var_87 = var_85.validate(var_86)
    var_88 = 'name'
    var_89 = 'age'
    var_90 = 'John'
    var_91 = 'not an integer'
    var_92 = {var_88: var_90, var_89: var_91}
    var_93 = var_85.validate(var_92)
    var_94 = '^[a-z]+$'
    var_95 = module_0.String(pattern=var_94)
    var_96 = module_0.Object(property_names=var_95)
    var_97 = {var_88: var_93}
    var_98 = var_96.validate(var_97)
    var_99 = 'Name'
    var_100 = 'John'
    var_101 = {var_99: var_100}
    var_102 = var_96.validate(var_101)
    var_103 = 'Unknown'
    var_104 = module_0.String()
    var_105 = {var_99: var_104}
    var_106 = module_0.Object(properties=var_105)
    var_107 = {}
    var_108 = var_106.validate(var_107)



# Parsed testcases at query #4
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Field(default=var_0)
    var_2 = var_1.get_default_value()
    assert var_2 == 42
    var_3 = 'hello'
    var_4 = lambda : var_3
    var_5 = module_0.Field(default=var_4)
    var_6 = var_5.get_default_value()
    assert var_6 == 'hello'
    var_7 = module_0.Field()
    var_8 = var_7.get_default_value()
    assert var_8 is None
    var_9 = True
    var_10 = module_0.Field(allow_null=var_9)
    var_11 = var_10.get_default_value()
    assert var_11 is None
    var_12 = None
    var_13 = module_0.Field(default=var_12, allow_null=var_9)
    var_14 = var_13.get_default_value()
    assert var_14 is None



# Parsed testcases at query #5
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = 123
    var_2 = var_0.validate(var_1)
    assert var_2 == 123
    var_3 = 12.3
    var_4 = var_0.validate(var_3)
    var_5 = True
    var_6 = module_0.Number()
    var_7 = None
    var_8 = var_6.validate(var_7)
    assert var_8 is None
    var_9 = module_0.Number()
    var_10 = None
    var_11 = var_9.validate(var_10)
    var_12 = module_0.Number()
    var_13 = True
    var_14 = var_12.validate(var_13)
    var_15 = module_0.Number(coerce_types=var_5)
    var_16 = '123'
    var_17 = var_15.validate(var_16)
    assert var_17 == 123
    var_18 = '12.3'
    var_19 = var_15.validate(var_18)
    var_20 = module_0.Number(coerce_types=var_5)
    var_21 = 'abc'
    var_22 = var_20.validate(var_21)
    var_23 = module_0.Number()
    var_24 = 'inf'
    var_25 = float(var_24)
    var_26 = var_23.validate(var_25)
    var_27 = '-inf'
    var_28 = float(var_27)
    var_29 = var_23.validate(var_28)
    var_30 = 'nan'
    var_31 = float(var_30)
    var_32 = var_23.validate(var_31)
    var_33 = 10
    var_34 = module_0.Number(minimum=var_33)
    var_35 = var_34.validate(var_33)
    assert var_35 == 10
    var_36 = 9
    var_37 = var_34.validate(var_36)
    var_38 = module_0.Number(exclusive_minimum=var_33)
    var_39 = 11
    var_40 = var_38.validate(var_39)
    assert var_40 == 11
    var_41 = 10
    var_42 = var_38.validate(var_41)
    var_43 = module_0.Number(maximum=var_33)
    var_44 = var_43.validate(var_33)
    assert var_44 == 10
    var_45 = 11
    var_46 = var_43.validate(var_45)
    var_47 = module_0.Number(exclusive_maximum=var_33)
    var_48 = 9
    var_49 = var_47.validate(var_48)
    assert var_49 == 9
    var_50 = 10
    var_51 = var_47.validate(var_50)
    var_52 = 5
    var_53 = module_0.Number(multiple_of=var_52)
    var_54 = var_53.validate(var_33)
    assert var_54 == 10
    var_55 = 11
    var_56 = var_53.validate(var_55)
    var_57 = '0.01'
    var_58 = module_0.Number(precision=var_57)
    var_59 = 10.123
    var_60 = var_58.validate(var_59)
    var_61 = 10.125
    var_62 = var_58.validate(var_61)
    var_63 = var_58.validate(var_33)
    assert var_63 == 10
    var_64 = 10.5
    var_65 = var_58.validate(var_64)
    var_66 = '10.5'
    var_67 = var_58.validate(var_66)
    var_68 = 10.5
    var_69 = var_58.validate(var_68)
    var_70 = var_58.validate(var_33)
    var_71 = module_0.Number(coerce_types=var_5)
    var_72 = ''
    var_73 = var_71.validate(var_72)
    assert var_73 is None



# Parsed testcases at query #6
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'Option A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'Option B'
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
    var_13 = [var_12]
    var_14 = True
    var_15 = module_0.Choice(choices=var_13)
    var_16 = None
    var_17 = var_15.validate(var_16)
    assert var_17 is None
    var_18 = None
    var_19 = var_7.validate(var_18)
    var_20 = (var_18, var_19)
    var_21 = [var_20]
    var_22 = module_0.Choice(choices=var_21, coerce_types=var_14)
    var_23 = ''
    var_24 = var_22.validate(var_23)
    assert var_24 is None
    var_25 = 'a'
    var_26 = 'Option A'
    var_27 = (var_25, var_26)
    var_28 = [var_27]
    var_29 = True
    var_30 = False
    var_31 = module_0.Choice(choices=var_28, coerce_types=var_29)
    var_32 = ''
    var_33 = var_31.validate(var_32)
    var_34 = (var_25, var_26)
    var_35 = (var_28, var_29)
    var_36 = [var_34, var_35]
    var_37 = module_0.Choice(choices=var_36)
    var_38 = var_37.validate(var_25)
    assert var_38 == 'a'
    var_39 = var_37.validate(var_28)
    assert var_39 == 'b'
    var_40 = [var_25, var_26]
    var_41 = [var_28, var_29]
    var_42 = [var_40, var_41]
    var_43 = module_0.Choice(choices=var_42)
    var_44 = var_43.validate(var_25)
    assert var_44 == 'a'
    var_45 = var_43.validate(var_28)
    assert var_45 == 'b'



# Parsed testcases at query #7
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Array()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    var_3 = module_0.Array(var_1)
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = var_3.serialize(var_7)
    var_9 = module_0.Integer()
    var_10 = module_0.Array(var_9)
    var_11 = [var_4, var_5, var_6]
    var_12 = var_10.serialize(var_11)
    var_13 = '1'
    var_14 = '2'
    var_15 = '3'
    var_16 = [var_13, var_14, var_15]
    var_17 = var_10.serialize(var_16)
    var_18 = module_0.Integer()
    var_19 = module_0.Integer()
    var_20 = [var_18, var_19]
    var_21 = module_0.Array(var_20)
    var_22 = [var_4, var_5]
    var_23 = var_21.serialize(var_22)
    var_24 = [var_13, var_14]
    var_25 = var_21.serialize(var_24)
    var_26 = [var_18]
    var_27 = module_0.Array(var_26, var_19)
    var_28 = [var_4, var_5, var_6]
    var_29 = var_27.serialize(var_28)
    var_30 = [var_13, var_14, var_15]
    var_31 = var_27.serialize(var_30)
    var_32 = module_0.Decimal()
    var_33 = module_0.Array(var_32)
    var_34 = '1.5'
    var_35 = '2.5'



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
    var_7 = module_0.String(allow_blank=var_3)
    var_8 = ''
    var_9 = var_7.validate(var_8)
    assert var_9 == ''
    var_10 = module_0.String(trim_whitespace=var_3)
    var_11 = '  hello  '
    var_12 = var_10.validate(var_11)
    assert var_12 == 'hello'
    var_13 = 3
    var_14 = module_0.String(min_length=var_13)
    var_15 = var_14.validate(var_1)
    assert var_15 == 'hello'
    var_16 = 'hi'
    var_17 = var_14.validate(var_16)
    var_18 = 5
    var_19 = module_0.String(max_length=var_18)
    var_20 = var_19.validate(var_16)
    assert var_20 == 'hello'
    var_21 = 'hello world'
    var_22 = var_19.validate(var_21)
    var_23 = '^[a-z]+$'
    var_24 = module_0.String(pattern=var_23)
    var_25 = var_24.validate(var_21)
    assert var_25 == 'hello'
    var_26 = 'Hello'
    var_27 = var_24.validate(var_26)
    var_28 = 'email'
    var_29 = module_0.String(format=var_28)
    var_30 = 'test@example.com'
    var_31 = var_29.validate(var_30)
    assert var_31 == 'test@example.com'
    var_32 = 'not-an-email'
    var_33 = var_29.validate(var_32)
    var_34 = module_0.String(allow_blank=var_3, coerce_types=var_3)
    var_35 = var_34.validate(var_5)
    assert var_35 == ''
    var_36 = module_0.String(coerce_types=var_3)
    var_37 = var_36.validate(var_5)
    assert var_37 is None
    var_38 = module_0.String()
    var_39 = 'hello\x00world'
    var_40 = var_38.validate(var_39)
    assert var_40 == 'helloworld'
    var_41 = module_0.String()
    var_42 = 123
    var_43 = var_41.validate(var_42)
    var_44 = False
    var_45 = module_0.String(allow_blank=var_44)
    var_46 = ''
    var_47 = var_45.validate(var_46)



# Parsed testcases at query #9
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Number()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None
    var_4 = module_0.Number()
    var_5 = None
    var_6 = var_4.validate(var_5)
    var_7 = module_0.Number(coerce_types=var_5)
    var_8 = ''
    var_9 = var_7.validate(var_8)
    assert var_9 is None
    var_10 = module_0.Number()
    var_11 = True
    var_12 = var_10.validate(var_11)
    var_13 = 1.5
    var_14 = var_10.validate(var_13)
    var_15 = False
    var_16 = module_0.Number(coerce_types=var_15)
    var_17 = '123'
    var_18 = var_16.validate(var_17)
    var_19 = module_0.Number()
    var_20 = '123'
    var_21 = var_19.validate(var_20)
    assert var_21 == 123
    var_22 = module_0.Number()
    var_23 = 'inf'
    var_24 = float(var_23)
    var_25 = var_22.validate(var_24)
    var_26 = '0.00'
    var_27 = module_0.Number(precision=var_26)
    var_28 = '123.456'
    var_29 = var_27.validate(var_28)
    var_30 = 10
    var_31 = module_0.Number(minimum=var_30)
    var_32 = 9
    var_33 = var_31.validate(var_32)
    var_34 = var_31.validate(var_30)
    assert var_34 == 10
    var_35 = module_0.Number(exclusive_minimum=var_30)
    var_36 = 10
    var_37 = var_35.validate(var_36)
    var_38 = 11
    var_39 = var_35.validate(var_38)
    assert var_39 == 11
    var_40 = module_0.Number(maximum=var_30)
    var_41 = 11
    var_42 = var_40.validate(var_41)
    var_43 = var_40.validate(var_30)
    assert var_43 == 10
    var_44 = module_0.Number(exclusive_maximum=var_30)
    var_45 = 10
    var_46 = var_44.validate(var_45)
    var_47 = 9
    var_48 = var_44.validate(var_47)
    assert var_48 == 9
    var_49 = 3
    var_50 = module_0.Number(multiple_of=var_49)
    var_51 = 10
    var_52 = var_50.validate(var_51)
    var_53 = var_50.validate(var_47)
    assert var_53 == 9
    var_54 = 0.5
    var_55 = module_0.Number(multiple_of=var_54)
    var_56 = 1.25
    var_57 = var_55.validate(var_56)
    var_58 = 1.5
    var_59 = var_55.validate(var_58)



# Parsed testcases at query #10
#--------------------------


import typesystem.fields as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'default'
    var_2 = hasattr(var_0, var_1)
    var_3 = 'Test Title'
    var_4 = 'Test Description'
    var_5 = 'default_value'
    var_6 = True
    var_7 = False
    var_8 = 10
    var_9 = 2
    var_10 = '^[a-z]+$'
    var_11 = 'email'
    var_12 = module_0.String(allow_blank=var_6, trim_whitespace=var_7, max_length=var_8, min_length=var_9, pattern=var_10, format=var_11, coerce_types=var_7)
    var_13 = '^[0-9]+$'
    var_14 = module_1.compile(var_13)
    var_15 = module_0.String(pattern=var_14)
    var_16 = module_0.String(allow_blank=var_6)
    var_17 = module_0.String()



# Parsed testcases at query #11
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Number()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None
    var_4 = module_0.Number()
    var_5 = None
    var_6 = var_4.validate(var_5)
    var_7 = module_0.Number(coerce_types=var_5)
    var_8 = ''
    var_9 = var_7.validate(var_8)
    assert var_9 is None
    var_10 = module_0.Number()
    var_11 = True
    var_12 = var_10.validate(var_11)
    var_13 = 1.5
    var_14 = var_10.validate(var_13)
    var_15 = False
    var_16 = module_0.Number(coerce_types=var_15)
    var_17 = '123'
    var_18 = var_16.validate(var_17)
    var_19 = module_0.Number()
    var_20 = 'inf'
    var_21 = float(var_20)
    var_22 = var_19.validate(var_21)
    var_23 = 5
    var_24 = module_0.Number(minimum=var_23)
    var_25 = 4
    var_26 = var_24.validate(var_25)
    var_27 = var_24.validate(var_23)
    assert var_27 == 5
    var_28 = module_0.Number(exclusive_minimum=var_23)
    var_29 = 5
    var_30 = var_28.validate(var_29)
    var_31 = 5.1
    var_32 = var_28.validate(var_31)
    var_33 = 10
    var_34 = module_0.Number(maximum=var_33)
    var_35 = 11
    var_36 = var_34.validate(var_35)
    var_37 = var_34.validate(var_33)
    assert var_37 == 10
    var_38 = module_0.Number(exclusive_maximum=var_33)
    var_39 = 10
    var_40 = var_38.validate(var_39)
    var_41 = 9.9
    var_42 = var_38.validate(var_41)
    var_43 = 3
    var_44 = module_0.Number(multiple_of=var_43)
    var_45 = 4
    var_46 = var_44.validate(var_45)
    var_47 = 6
    var_48 = var_44.validate(var_47)
    assert var_48 == 6
    var_49 = 0.5
    var_50 = module_0.Number(multiple_of=var_49)
    var_51 = 1.2
    var_52 = var_50.validate(var_51)
    var_53 = var_50.validate(var_51)
    var_54 = '0.01'
    var_55 = module_0.Number(precision=var_54)
    var_56 = 3.14159
    var_57 = var_55.validate(var_56)
    var_58 = module_0.Number()
    var_59 = 123
    var_60 = var_58.validate(var_59)
    assert var_60 == 123
    var_61 = 12.3
    var_62 = var_58.validate(var_61)
    var_63 = '123'
    var_64 = var_58.validate(var_63)
    assert var_64 == 123



# Parsed testcases at query #12
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
    var_10 = 'not an array'
    var_11 = var_0.validate(var_10)
    var_12 = module_0.Array(min_items=var_11)
    var_13 = [var_6, var_11]
    var_14 = var_12.validate(var_13)
    var_15 = 1
    var_16 = [var_15]
    var_17 = var_12.validate(var_16)
    var_18 = module_0.Array(max_items=var_16)
    var_19 = [var_6, var_16]
    var_20 = var_18.validate(var_19)
    var_21 = 1
    var_22 = 2
    var_23 = 3
    var_24 = [var_21, var_22, var_23]
    var_25 = var_18.validate(var_24)
    var_26 = module_0.Array(exact_items=var_22)
    var_27 = [var_6, var_22]
    var_28 = var_26.validate(var_27)
    var_29 = 1
    var_30 = [var_29]
    var_31 = var_26.validate(var_30)
    var_32 = 1
    var_33 = 2
    var_34 = 3
    var_35 = [var_32, var_33, var_34]
    var_36 = var_26.validate(var_35)
    var_37 = True
    var_38 = module_0.Array(unique_items=var_37)
    var_39 = [var_37, var_33, var_34]
    var_40 = var_38.validate(var_39)
    var_41 = 1
    var_42 = 2
    var_43 = [var_41, var_42, var_42]
    var_44 = var_38.validate(var_43)
    var_45 = module_0.Integer()
    var_46 = module_0.Array(var_45)
    var_47 = '1'
    var_48 = '2'
    var_49 = '3'
    var_50 = [var_47, var_48, var_49]
    var_51 = var_46.validate(var_50)
    var_52 = '1'
    var_53 = 'two'
    var_54 = '3'
    var_55 = [var_52, var_53, var_54]
    var_56 = var_46.validate(var_55)
    var_57 = module_0.Integer()
    var_58 = module_0.Integer()
    var_59 = [var_57, var_58]
    var_60 = True
    var_61 = module_0.Array(var_59, var_60)
    var_62 = [var_60, var_53, var_54]
    var_63 = var_61.validate(var_62)
    var_64 = module_0.Integer()
    var_65 = module_0.Integer()
    var_66 = [var_64, var_65]
    var_67 = False
    var_68 = module_0.Array(var_66, var_67)
    var_69 = [var_60, var_53]
    var_70 = var_68.validate(var_69)
    var_71 = 1
    var_72 = 2
    var_73 = 3
    var_74 = [var_71, var_72, var_73]
    var_75 = var_68.validate(var_74)
    var_76 = [var_60, var_72, var_73]
    var_77 = var_0.serialize(var_76)
    var_78 = [var_60, var_72, var_73]
    var_79 = var_46.serialize(var_78)



# Parsed testcases at query #13
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
    var_5 = module_0.Boolean()
    var_6 = None
    var_7 = var_5.validate(var_6)
    assert var_7 is None
    var_8 = module_0.Boolean()
    var_9 = None
    var_10 = var_8.validate(var_9)
    var_11 = module_0.Boolean(coerce_types=var_9)
    var_12 = 'true'
    var_13 = var_11.validate(var_12)
    assert var_13 is True
    var_14 = 'false'
    var_15 = var_11.validate(var_14)
    assert var_15 is False
    var_16 = 'on'
    var_17 = var_11.validate(var_16)
    assert var_17 is True
    var_18 = 'off'
    var_19 = var_11.validate(var_18)
    assert var_19 is False
    var_20 = '1'
    var_21 = var_11.validate(var_20)
    assert var_21 is True
    var_22 = '0'
    var_23 = var_11.validate(var_22)
    assert var_23 is False
    var_24 = ''
    var_25 = var_11.validate(var_24)
    assert var_25 is False
    var_26 = var_11.validate(var_9)
    assert var_26 is True
    var_27 = var_11.validate(var_3)
    assert var_27 is False
    var_28 = module_0.Boolean(coerce_types=var_9)
    var_29 = var_28.validate(var_24)
    assert var_29 is None
    var_30 = 'null'
    var_31 = var_28.validate(var_30)
    assert var_31 is None
    var_32 = 'none'
    var_33 = var_28.validate(var_32)
    assert var_33 is None
    var_34 = module_0.Boolean(coerce_types=var_3)
    var_35 = 'true'
    var_36 = var_34.validate(var_35)
    var_37 = 1
    var_38 = var_34.validate(var_37)
    var_39 = 'invalid'
    var_40 = var_34.validate(var_39)



# Parsed testcases at query #14
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
    var_8 = module_0.String()
    var_9 = module_0.Integer()
    var_10 = [var_8, var_9]
    var_11 = module_0.Union(var_10)
    var_12 = 12.3
    var_13 = var_11.validate(var_12)
    var_14 = True
    var_15 = module_0.String()
    var_16 = module_0.Integer()
    var_17 = [var_15, var_16]
    var_18 = module_0.Union(var_17)
    var_19 = None
    var_20 = var_18.validate(var_19)
    assert var_20 is None
    var_21 = module_0.String()
    var_22 = module_0.Integer()
    var_23 = [var_21, var_22]
    var_24 = module_0.Union(var_23)
    var_25 = None
    var_26 = var_24.validate(var_25)
    var_27 = 5
    var_28 = module_0.String(min_length=var_27)
    var_29 = 10
    var_30 = module_0.Integer(minimum=var_29)
    var_31 = [var_28, var_30]
    var_32 = module_0.Union(var_31)
    var_33 = 'test'
    var_34 = var_32.validate(var_33)
    var_35 = 0
    var_36 = exc_info.value.messages()[var_35]
    var_37 = var_36.code
    assert var_37 == 'min_length'
    var_38 = module_0.String(min_length=var_27)
    var_39 = module_0.Integer(minimum=var_29)
    var_40 = [var_38, var_39]
    var_41 = module_0.Union(var_40)
    var_42 = 5
    var_43 = var_41.validate(var_42)
    var_44 = exc_info.value.messages()[var_35]
    var_45 = var_44.code
    assert var_45 == 'union'



# Parsed testcases at query #15
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
    var_11 = module_0.Array()
    var_12 = []
    var_13 = var_11.validate(var_12)
    var_14 = 2
    var_15 = module_0.Array(min_items=var_14)
    var_16 = 1
    var_17 = [var_16]
    var_18 = var_15.validate(var_17)
    var_19 = [var_16, var_14]
    var_20 = var_15.validate(var_19)
    var_21 = module_0.Array(max_items=var_14)
    var_22 = 1
    var_23 = 2
    var_24 = 3
    var_25 = [var_22, var_23, var_24]
    var_26 = var_21.validate(var_25)
    var_27 = [var_22, var_14]
    var_28 = var_21.validate(var_27)
    var_29 = module_0.Array(exact_items=var_14)
    var_30 = 1
    var_31 = [var_30]
    var_32 = var_29.validate(var_31)
    var_33 = 1
    var_34 = 2
    var_35 = 3
    var_36 = [var_33, var_34, var_35]
    var_37 = var_29.validate(var_36)
    var_38 = [var_33, var_14]
    var_39 = var_29.validate(var_38)
    var_40 = module_0.Array(unique_items=var_33)
    var_41 = 1
    var_42 = [var_41, var_41]
    var_43 = var_40.validate(var_42)
    var_44 = [var_41, var_14]
    var_45 = var_40.validate(var_44)
    var_46 = module_0.Integer()
    var_47 = module_0.Array(var_46)
    var_48 = 1
    var_49 = 'not an integer'
    var_50 = [var_48, var_49]
    var_51 = var_47.validate(var_50)
    var_52 = [var_48, var_14]
    var_53 = var_47.validate(var_52)
    var_54 = module_0.Integer()
    var_55 = module_0.Integer()
    var_56 = [var_54, var_55]
    var_57 = module_0.Array(var_56, var_51)
    var_58 = 1
    var_59 = 2
    var_60 = 3
    var_61 = [var_58, var_59, var_60]
    var_62 = var_57.validate(var_61)
    var_63 = [var_58, var_14]
    var_64 = var_57.validate(var_63)
    var_65 = module_0.Integer()
    var_66 = module_0.Integer()
    var_67 = [var_65, var_66]
    var_68 = module_0.Integer()
    var_69 = module_0.Array(var_67, var_68)
    var_70 = 3
    var_71 = [var_58, var_14, var_70]
    var_72 = var_69.validate(var_71)
    var_73 = 'name'
    var_74 = module_0.String()
    var_75 = {var_73: var_74}
    var_76 = module_0.Object(properties=var_75)
    var_77 = module_0.Array(var_76)
    var_78 = 'name'
    var_79 = 'test'
    var_80 = {var_78: var_79}
    var_81 = 'invalid'
    var_82 = 'object'
    var_83 = {var_81: var_82}
    var_84 = [var_80, var_83]
    var_85 = var_77.validate(var_84)
    var_86 = 'test'
    var_87 = {var_73: var_86}
    var_88 = 'test2'
    var_89 = {var_73: var_88}
    var_90 = [var_87, var_89]
    var_91 = var_77.validate(var_90)



# Parsed testcases at query #16
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
    var_8 = (var_0, var_1)
    var_9 = (var_3, var_4)
    var_10 = [var_8, var_9]
    var_11 = False
    var_12 = module_0.Choice(choices=var_10, coerce_types=var_11)
    var_13 = (var_0, var_1)
    var_14 = (var_3, var_4)
    var_15 = [var_13, var_14]
    var_16 = True
    var_17 = module_0.Choice(choices=var_15)
    var_18 = (var_0, var_1)
    var_19 = (var_3, var_4)
    var_20 = [var_18, var_19]
    var_21 = 'Test Choice'
    var_22 = 'A test choice field'
    var_23 = module_0.Choice(choices=var_20)
    var_24 = (var_0, var_1)
    var_25 = (var_3, var_4)
    var_26 = [var_24, var_25]
    var_27 = module_0.Choice(choices=var_26)
    var_28 = (var_0, var_1)
    var_29 = (var_3, var_4)
    var_30 = [var_28, var_29]
    var_31 = lambda : var_0
    var_32 = module_0.Choice(choices=var_30)
    var_33 = []
    var_34 = module_0.Choice(choices=var_33)
    var_35 = (var_0, var_1)
    var_36 = [var_35]
    var_37 = module_0.Choice(choices=var_36)
    var_38 = (var_0, var_1)
    var_39 = (var_3, var_4)
    var_40 = 'c'
    var_41 = 'C'
    var_42 = (var_40, var_41)
    var_43 = [var_38, var_39, var_42]
    var_44 = module_0.Choice(choices=var_43)
    var_45 = (var_3, var_4)
    var_46 = [var_0, var_45]
    var_47 = module_0.Choice(choices=var_46)
    var_48 = (var_0, var_1)
    var_49 = (var_3, var_4)
    var_50 = [var_48, var_49]
    var_51 = module_0.Choice(choices=var_50)



# Parsed testcases at query #17
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Number()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None
    var_4 = module_0.Number()
    var_5 = None
    var_6 = var_4.validate(var_5)
    var_7 = module_0.Number(coerce_types=var_5)
    var_8 = ''
    var_9 = var_7.validate(var_8)
    assert var_9 is None
    var_10 = module_0.Number()
    var_11 = True
    var_12 = var_10.validate(var_11)
    var_13 = False
    var_14 = module_0.Number(coerce_types=var_13)
    var_15 = 'abc'
    var_16 = var_14.validate(var_15)
    var_17 = module_0.Number()
    var_18 = 'inf'
    var_19 = float(var_18)
    var_20 = var_17.validate(var_19)
    var_21 = 5
    var_22 = module_0.Number(minimum=var_21)
    var_23 = 3
    var_24 = var_22.validate(var_23)
    var_25 = module_0.Number(exclusive_minimum=var_21)
    var_26 = 5
    var_27 = var_25.validate(var_26)
    var_28 = 10
    var_29 = module_0.Number(maximum=var_28)
    var_30 = 15
    var_31 = var_29.validate(var_30)
    var_32 = module_0.Number(exclusive_maximum=var_28)
    var_33 = 10
    var_34 = var_32.validate(var_33)
    var_35 = 3
    var_36 = module_0.Number(multiple_of=var_35)
    var_37 = 5
    var_38 = var_36.validate(var_37)
    var_39 = 0.5
    var_40 = module_0.Number(multiple_of=var_39)
    var_41 = 1.2
    var_42 = var_40.validate(var_41)
    var_43 = '0.01'
    var_44 = module_0.Number(precision=var_43)
    var_45 = 3.14159
    var_46 = var_44.validate(var_45)
    var_47 = var_44.validate(var_21)
    assert var_47 == 5
    var_48 = 5.5
    var_49 = var_44.validate(var_48)
    var_50 = module_0.Number(coerce_types=var_41)
    var_51 = '5'
    var_52 = var_50.validate(var_51)
    assert var_52 == 5
    var_53 = 5.5
    var_54 = var_50.validate(var_53)



# Parsed testcases at query #18
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
    var_5 = var_0.validate(var_1)
    assert var_5 is True
    var_6 = var_0.validate(var_3)
    assert var_6 is False
    var_7 = 'true'
    var_8 = var_0.validate(var_7)
    assert var_8 is True
    var_9 = 'false'
    var_10 = var_0.validate(var_9)
    assert var_10 is False
    var_11 = 'on'
    var_12 = var_0.validate(var_11)
    assert var_12 is True
    var_13 = 'off'
    var_14 = var_0.validate(var_13)
    assert var_14 is False
    var_15 = '1'
    var_16 = var_0.validate(var_15)
    assert var_16 is True
    var_17 = '0'
    var_18 = var_0.validate(var_17)
    assert var_18 is False
    var_19 = ''
    var_20 = var_0.validate(var_19)
    assert var_20 is False
    var_21 = module_0.Boolean(coerce_types=var_3)
    var_22 = 1
    var_23 = var_21.validate(var_22)
    var_24 = 'true'
    var_25 = var_21.validate(var_24)
    var_26 = module_0.Boolean()
    var_27 = None
    var_28 = var_26.validate(var_27)
    assert var_28 is None
    var_29 = 'null'
    var_30 = var_26.validate(var_29)
    assert var_30 is None
    var_31 = 'none'
    var_32 = var_26.validate(var_31)
    assert var_32 is None
    var_33 = None
    var_34 = var_0.validate(var_33)
    var_35 = 'invalid'
    var_36 = var_0.validate(var_35)
    var_37 = 2
    var_38 = var_0.validate(var_37)
    var_39 = 'yes'
    var_40 = var_0.validate(var_39)



# Parsed testcases at query #19
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Object(properties=var_4)
    var_6 = 'John'
    var_7 = 30
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = module_0.String()
    var_11 = {var_0: var_10}
    var_12 = True
    var_13 = module_0.Object(properties=var_11)
    var_14 = None
    var_15 = var_13.validate(var_14)
    assert var_15 is None
    var_16 = module_0.String()
    var_17 = {var_0: var_16}
    var_18 = [var_0]
    var_19 = module_0.Object(properties=var_17, required=var_18)
    var_20 = {}
    var_21 = var_19.validate(var_20)
    var_22 = module_0.String()
    var_23 = {var_20: var_22}
    var_24 = module_0.Object(properties=var_23)
    var_25 = 123
    var_26 = 'John'
    var_27 = {var_25: var_26}
    var_28 = var_24.validate(var_27)
    var_29 = module_0.Object(min_properties=var_12)
    var_30 = {}
    var_31 = var_29.validate(var_30)
    var_32 = module_0.Object(max_properties=var_12)
    var_33 = 'name'
    var_34 = 'age'
    var_35 = 'John'
    var_36 = 30
    var_37 = {var_33: var_35, var_34: var_36}
    var_38 = var_32.validate(var_37)
    var_39 = module_0.String()
    var_40 = {var_33: var_39}
    var_41 = False
    var_42 = module_0.Object(properties=var_40, additional_properties=var_41)
    var_43 = 'name'
    var_44 = 'age'
    var_45 = 'John'
    var_46 = 30
    var_47 = {var_43: var_45, var_44: var_46}
    var_48 = var_42.validate(var_47)
    var_49 = module_0.String()
    var_50 = {var_43: var_49}
    var_51 = module_0.Integer()
    var_52 = module_0.Object(properties=var_50, additional_properties=var_51)
    var_53 = {var_43: var_48, var_44: var_7}
    var_54 = var_52.validate(var_53)
    var_55 = '^S_'
    var_56 = '^I_'
    var_57 = module_0.String()
    var_58 = module_0.Integer()
    var_59 = {var_55: var_57, var_56: var_58}
    var_60 = module_0.Object(pattern_properties=var_59)
    var_61 = 'S_name'
    var_62 = 'I_age'
    var_63 = {var_61: var_48, var_62: var_7}
    var_64 = var_60.validate(var_63)
    var_65 = 2
    var_66 = module_0.String(min_length=var_65)
    var_67 = module_0.Object(property_names=var_66)
    var_68 = 'a'
    var_69 = 'John'
    var_70 = {var_68: var_69}
    var_71 = var_67.validate(var_70)
    var_72 = 'address'
    var_73 = 'city'
    var_74 = module_0.String()
    var_75 = {var_73: var_74}
    var_76 = module_0.Object(properties=var_75)
    var_77 = {var_72: var_76}
    var_78 = module_0.Object(properties=var_77)
    var_79 = 'New York'
    var_80 = {var_73: var_79}
    var_81 = {var_72: var_80}
    var_82 = var_78.validate(var_81)



# Parsed testcases at query #20
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = 123
    var_2 = var_0.validate(var_1)
    assert var_2 == 123
    var_3 = '123'
    var_4 = var_0.validate(var_3)
    assert var_4 == 123
    var_5 = var_0.validate(var_1)
    var_6 = module_0.Number()
    var_7 = 123.45
    var_8 = var_6.validate(var_7)
    var_9 = '123.45'
    var_10 = var_6.validate(var_9)
    var_11 = True
    var_12 = module_0.Number()
    var_13 = None
    var_14 = var_12.validate(var_13)
    assert var_14 is None
    var_15 = module_0.Number()
    var_16 = None
    var_17 = var_15.validate(var_16)
    var_18 = module_0.Number(coerce_types=var_11)
    var_19 = ''
    var_20 = var_18.validate(var_19)
    assert var_20 is None
    var_21 = 10
    var_22 = module_0.Number(minimum=var_21)
    var_23 = var_22.validate(var_21)
    assert var_23 == 10
    var_24 = 11
    var_25 = var_22.validate(var_24)
    assert var_25 == 11
    var_26 = 9
    var_27 = var_22.validate(var_26)
    var_28 = module_0.Number(exclusive_minimum=var_21)
    var_29 = var_28.validate(var_24)
    assert var_29 == 11
    var_30 = 10
    var_31 = var_28.validate(var_30)
    var_32 = module_0.Number(maximum=var_21)
    var_33 = var_32.validate(var_21)
    assert var_33 == 10
    var_34 = 9
    var_35 = var_32.validate(var_34)
    assert var_35 == 9
    var_36 = 11
    var_37 = var_32.validate(var_36)
    var_38 = module_0.Number(exclusive_maximum=var_21)
    var_39 = var_38.validate(var_34)
    assert var_39 == 9
    var_40 = 10
    var_41 = var_38.validate(var_40)
    var_42 = 5
    var_43 = module_0.Number(multiple_of=var_42)
    var_44 = var_43.validate(var_21)
    assert var_44 == 10
    var_45 = 11
    var_46 = var_43.validate(var_45)
    var_47 = '0.01'
    var_48 = module_0.Number(precision=var_47)
    var_49 = 123.455
    var_50 = var_48.validate(var_49)
    var_51 = module_0.Number()
    var_52 = 'inf'
    var_53 = float(var_52)
    var_54 = var_51.validate(var_53)
    var_55 = '-inf'
    var_56 = float(var_55)
    var_57 = var_51.validate(var_56)
    var_58 = 'nan'
    var_59 = float(var_58)
    var_60 = var_51.validate(var_59)
    var_61 = module_0.Number()
    var_62 = True
    var_63 = var_61.validate(var_62)
    var_64 = False
    var_65 = var_61.validate(var_64)
    var_66 = False
    var_67 = module_0.Number(coerce_types=var_66)
    var_68 = '123'
    var_69 = var_67.validate(var_68)



# Parsed testcases at query #21
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Number()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None
    var_4 = module_0.Number()
    var_5 = None
    var_6 = var_4.validate(var_5)
    var_7 = module_0.Number(coerce_types=var_5)
    var_8 = ''
    var_9 = var_7.validate(var_8)
    assert var_9 is None
    var_10 = module_0.Number()
    var_11 = True
    var_12 = var_10.validate(var_11)
    var_13 = 3.14
    var_14 = var_10.validate(var_13)
    var_15 = False
    var_16 = module_0.Number(coerce_types=var_15)
    var_17 = '123'
    var_18 = var_16.validate(var_17)
    var_19 = module_0.Number()
    var_20 = '123'
    var_21 = var_19.validate(var_20)
    assert var_21 == 123
    var_22 = module_0.Number()
    var_23 = 'inf'
    var_24 = float(var_23)
    var_25 = var_22.validate(var_24)
    var_26 = '0.01'
    var_27 = module_0.Number(precision=var_26)
    var_28 = '3.14159'
    var_29 = var_27.validate(var_28)
    var_30 = 5
    var_31 = module_0.Number(minimum=var_30)
    var_32 = 3
    var_33 = var_31.validate(var_32)
    var_34 = var_31.validate(var_30)
    assert var_34 == 5
    var_35 = module_0.Number(exclusive_minimum=var_30)
    var_36 = 5
    var_37 = var_35.validate(var_36)
    var_38 = 5.0001
    var_39 = var_35.validate(var_38)
    var_40 = 10
    var_41 = module_0.Number(maximum=var_40)
    var_42 = 15
    var_43 = var_41.validate(var_42)
    var_44 = var_41.validate(var_40)
    assert var_44 == 10
    var_45 = module_0.Number(exclusive_maximum=var_40)
    var_46 = 10
    var_47 = var_45.validate(var_46)
    var_48 = 9.999
    var_49 = var_45.validate(var_48)
    var_50 = 3
    var_51 = module_0.Number(multiple_of=var_50)
    var_52 = 10
    var_53 = var_51.validate(var_52)
    var_54 = 9
    var_55 = var_51.validate(var_54)
    assert var_55 == 9
    var_56 = 0.5
    var_57 = module_0.Number(multiple_of=var_56)
    var_58 = 1.2
    var_59 = var_57.validate(var_58)
    var_60 = var_57.validate(var_58)
    var_61 = 100
    var_62 = module_0.Number(minimum=var_15, maximum=var_61)
    var_63 = 50
    var_64 = var_62.validate(var_63)
    assert var_64 == 50



# Parsed testcases at query #22
#--------------------------


import typesystem.fields as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'Test Title'
    var_2 = 'Test Description'
    var_3 = 'default_value'
    var_4 = True
    var_5 = False
    var_6 = 100
    var_7 = 10
    var_8 = '^[a-z]+$'
    var_9 = 'email'
    var_10 = module_0.String(allow_blank=var_4, trim_whitespace=var_5, max_length=var_6, min_length=var_7, pattern=var_8, format=var_9, coerce_types=var_5)
    var_11 = module_0.String(allow_blank=var_4)
    var_12 = module_0.String()
    var_13 = '^[0-9]+$'
    var_14 = module_1.compile(var_13)
    var_15 = module_0.String(pattern=var_14)
    var_16 = 'invalid'
    var_17 = module_0.String(max_length=var_16)
    var_18 = 'invalid'
    var_19 = module_0.String(min_length=var_18)
    var_20 = 123
    var_21 = module_0.String(pattern=var_20)
    var_22 = 123
    var_23 = module_0.String(format=var_22)



# Parsed testcases at query #23
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Object(properties=var_4)
    var_6 = 'John'
    var_7 = 30
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = True
    var_11 = module_0.Object()
    var_12 = None
    var_13 = var_11.validate(var_12)
    assert var_13 is None
    var_14 = 'not an object'
    var_15 = var_5.validate(var_14)
    var_16 = 123
    var_17 = 'invalid key'
    var_18 = {var_16: var_17}
    var_19 = var_5.validate(var_18)
    var_20 = module_0.String()
    var_21 = {var_16: var_20}
    var_22 = [var_16]
    var_23 = module_0.Object(properties=var_21, required=var_22)
    var_24 = 'age'
    var_25 = 30
    var_26 = {var_24: var_25}
    var_27 = var_23.validate(var_26)
    var_28 = 2
    var_29 = module_0.Object(min_properties=var_28)
    var_30 = 'name'
    var_31 = 'John'
    var_32 = {var_30: var_31}
    var_33 = var_29.validate(var_32)
    var_34 = module_0.Object(max_properties=var_28)
    var_35 = 'name'
    var_36 = 'age'
    var_37 = 'city'
    var_38 = 'John'
    var_39 = 30
    var_40 = 'NYC'
    var_41 = {var_35: var_38, var_36: var_39, var_37: var_40}
    var_42 = var_34.validate(var_41)
    var_43 = '^[a-z]+$'
    var_44 = module_0.String(pattern=var_43)
    var_45 = module_0.Object(property_names=var_44)
    var_46 = 'Name'
    var_47 = 'John'
    var_48 = {var_46: var_47}
    var_49 = var_45.validate(var_48)
    var_50 = module_0.String()
    var_51 = {var_46: var_50}
    var_52 = False
    var_53 = module_0.Object(properties=var_51, additional_properties=var_52)
    var_54 = 'name'
    var_55 = 'age'
    var_56 = 'John'
    var_57 = 30
    var_58 = {var_54: var_56, var_55: var_57}
    var_59 = var_53.validate(var_58)
    var_60 = module_0.String()
    var_61 = {var_54: var_60}
    var_62 = module_0.Integer()
    var_63 = module_0.Object(properties=var_61, additional_properties=var_62)
    var_64 = {var_54: var_59, var_55: var_41}
    var_65 = var_63.validate(var_64)
    var_66 = 'name'
    var_67 = 'age'
    var_68 = 'John'
    var_69 = 'not an integer'
    var_70 = {var_66: var_68, var_67: var_69}
    var_71 = var_63.validate(var_70)
    var_72 = '^S_'
    var_73 = '^I_'
    var_74 = module_0.String()
    var_75 = module_0.Integer()
    var_76 = {var_72: var_74, var_73: var_75}
    var_77 = module_0.Object(pattern_properties=var_76)
    var_78 = 'S_name'
    var_79 = 'I_age'
    var_80 = {var_78: var_71, var_79: var_41}
    var_81 = var_77.validate(var_80)
    var_82 = 'Unknown'
    var_83 = module_0.String()
    var_84 = {var_66: var_83}
    var_85 = module_0.Object(properties=var_84)
    var_86 = {}
    var_87 = var_85.validate(var_86)
    var_88 = 'address'
    var_89 = 'street'
    var_90 = 'city'
    var_91 = module_0.String()
    var_92 = module_0.String()
    var_93 = {var_89: var_91, var_90: var_92}
    var_94 = module_0.Object(properties=var_93)
    var_95 = {var_88: var_94}
    var_96 = module_0.Object(properties=var_95)
    var_97 = '123 Main St'
    var_98 = 'NYC'
    var_99 = {var_89: var_97, var_90: var_98}
    var_100 = {var_88: var_99}
    var_101 = var_96.validate(var_100)



# Parsed testcases at query #24
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Const(var_0)
    var_2 = None
    var_3 = module_0.Const(var_2)
    var_4 = 'test'
    var_5 = module_0.Const(var_4)
    var_6 = 42
    var_7 = True
    var_8 = module_0.Const(var_6)



# Parsed testcases at query #25
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Number()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None
    var_4 = module_0.Number()
    var_5 = None
    var_6 = var_4.validate(var_5)
    var_7 = module_0.Number(coerce_types=var_5)
    var_8 = ''
    var_9 = var_7.validate(var_8)
    assert var_9 is None
    var_10 = module_0.Number()
    var_11 = True
    var_12 = var_10.validate(var_11)
    var_13 = 1.5
    var_14 = var_10.validate(var_13)
    var_15 = module_0.Number()
    var_16 = 'invalid'
    var_17 = var_15.validate(var_16)
    var_18 = module_0.Number()
    var_19 = 'inf'
    var_20 = float(var_19)
    var_21 = var_18.validate(var_20)
    var_22 = '-inf'
    var_23 = float(var_22)
    var_24 = var_18.validate(var_23)
    var_25 = 'nan'
    var_26 = float(var_25)
    var_27 = var_18.validate(var_26)
    var_28 = '0.01'
    var_29 = module_0.Number(precision=var_28)
    var_30 = '1.234'
    var_31 = var_29.validate(var_30)
    var_32 = 5
    var_33 = module_0.Number(minimum=var_32)
    var_34 = var_33.validate(var_32)
    assert var_34 == 5
    var_35 = 4
    var_36 = var_33.validate(var_35)
    var_37 = module_0.Number(exclusive_minimum=var_32)
    var_38 = 6
    var_39 = var_37.validate(var_38)
    assert var_39 == 6
    var_40 = 5
    var_41 = var_37.validate(var_40)
    var_42 = 10
    var_43 = module_0.Number(maximum=var_42)
    var_44 = var_43.validate(var_42)
    assert var_44 == 10
    var_45 = 11
    var_46 = var_43.validate(var_45)
    var_47 = module_0.Number(exclusive_maximum=var_42)
    var_48 = 9
    var_49 = var_47.validate(var_48)
    assert var_49 == 9
    var_50 = 10
    var_51 = var_47.validate(var_50)
    var_52 = 3
    var_53 = module_0.Number(multiple_of=var_52)
    var_54 = var_53.validate(var_48)
    assert var_54 == 9
    var_55 = 10
    var_56 = var_53.validate(var_55)
    var_57 = 0.5
    var_58 = module_0.Number(multiple_of=var_57)
    var_59 = 2.5
    var_60 = var_58.validate(var_59)
    var_61 = 2.6
    var_62 = var_58.validate(var_61)
    var_63 = module_0.Number()
    var_64 = 123
    var_65 = var_63.validate(var_64)
    assert var_65 == 123
    var_66 = 12.3
    var_67 = var_63.validate(var_66)
    var_68 = '123'
    var_69 = var_63.validate(var_68)
    assert var_69 == 123
    var_70 = '12.3'
    var_71 = var_63.validate(var_70)
    var_72 = var_63.validate(var_68)
    var_73 = var_63.validate(var_64)



# Parsed testcases at query #26
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Const(var_0)
    var_2 = None
    var_3 = module_0.Const(var_2)
    var_4 = 'hello'
    var_5 = module_0.Const(var_4)
    var_6 = 42
    var_7 = True
    var_8 = module_0.Const(var_6)



# Parsed testcases at query #27
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
    var_10 = module_0.Array()
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = module_0.Array()
    var_14 = 'not a list'
    var_15 = var_13.validate(var_14)
    var_16 = module_0.Array(min_items=var_15)
    var_17 = [var_6, var_15]
    var_18 = var_16.validate(var_17)
    var_19 = 1
    var_20 = [var_19]
    var_21 = var_16.validate(var_20)
    var_22 = module_0.Array(max_items=var_20)
    var_23 = [var_6, var_20]
    var_24 = var_22.validate(var_23)
    var_25 = 1
    var_26 = 2
    var_27 = 3
    var_28 = [var_25, var_26, var_27]
    var_29 = var_22.validate(var_28)
    var_30 = module_0.Array(exact_items=var_26)
    var_31 = [var_6, var_26]
    var_32 = var_30.validate(var_31)
    var_33 = 1
    var_34 = [var_33]
    var_35 = var_30.validate(var_34)
    var_36 = 1
    var_37 = 2
    var_38 = 3
    var_39 = [var_36, var_37, var_38]
    var_40 = var_30.validate(var_39)
    var_41 = module_0.Array(min_items=var_6)
    var_42 = []
    var_43 = var_41.validate(var_42)
    var_44 = module_0.Integer()
    var_45 = module_0.Array(var_44)
    var_46 = [var_6, var_43, var_38]
    var_47 = var_45.validate(var_46)
    var_48 = 1
    var_49 = 'two'
    var_50 = 3
    var_51 = [var_48, var_49, var_50]
    var_52 = var_45.validate(var_51)
    var_53 = module_0.Integer()
    var_54 = module_0.Integer()
    var_55 = [var_53, var_54]
    var_56 = False
    var_57 = module_0.Array(var_55, var_56)
    var_58 = [var_6, var_49]
    var_59 = var_57.validate(var_58)
    var_60 = 1
    var_61 = 2
    var_62 = 3
    var_63 = [var_60, var_61, var_62]
    var_64 = var_57.validate(var_63)
    var_65 = module_0.Integer()
    var_66 = module_0.Integer()
    var_67 = [var_65, var_66]
    var_68 = module_0.Integer()
    var_69 = module_0.Array(var_67, var_68)
    var_70 = [var_6, var_61, var_62]
    var_71 = var_69.validate(var_70)
    var_72 = True
    var_73 = module_0.Array(unique_items=var_72)
    var_74 = [var_72, var_61, var_62]
    var_75 = var_73.validate(var_74)
    var_76 = 1
    var_77 = 2
    var_78 = [var_76, var_77, var_77]
    var_79 = var_73.validate(var_78)
    var_80 = 'id'
    var_81 = module_0.Integer()
    var_82 = {var_80: var_81}
    var_83 = module_0.Object(properties=var_82)
    var_84 = module_0.Array(var_83)
    var_85 = {var_80: var_72}
    var_86 = {var_80: var_77}
    var_87 = [var_85, var_86]
    var_88 = var_84.validate(var_87)
    var_89 = 'id'
    var_90 = 1
    var_91 = {var_89: var_90}
    var_92 = 'two'
    var_93 = {var_89: var_92}
    var_94 = [var_91, var_93]
    var_95 = var_84.validate(var_94)
    var_96 = module_0.Integer()
    var_97 = module_0.Array(var_96)
    var_98 = [var_72, var_90, var_91]
    var_99 = var_97.serialize(var_98)
    var_100 = var_97.serialize(var_95)
    assert var_100 is None
    var_101 = module_0.Integer()
    var_102 = module_0.Float()
    var_103 = [var_101, var_102]
    var_104 = module_0.Array(var_103)
    var_105 = 2.5
    var_106 = [var_72, var_105]
    var_107 = var_104.serialize(var_106)



# Parsed testcases at query #28
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
    var_10 = module_0.Array()
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = module_0.Array()
    var_14 = 'not a list'
    var_15 = var_13.validate(var_14)
    var_16 = module_0.Array(min_items=var_15)
    var_17 = [var_6, var_15]
    var_18 = var_16.validate(var_17)
    var_19 = 1
    var_20 = [var_19]
    var_21 = var_16.validate(var_20)
    var_22 = module_0.Array(max_items=var_20)
    var_23 = [var_6, var_20]
    var_24 = var_22.validate(var_23)
    var_25 = 1
    var_26 = 2
    var_27 = 3
    var_28 = [var_25, var_26, var_27]
    var_29 = var_22.validate(var_28)
    var_30 = module_0.Array(exact_items=var_26)
    var_31 = [var_6, var_26]
    var_32 = var_30.validate(var_31)
    var_33 = 1
    var_34 = [var_33]
    var_35 = var_30.validate(var_34)
    var_36 = 1
    var_37 = 2
    var_38 = 3
    var_39 = [var_36, var_37, var_38]
    var_40 = var_30.validate(var_39)
    var_41 = True
    var_42 = module_0.Array(unique_items=var_41)
    var_43 = [var_41, var_37, var_38]
    var_44 = var_42.validate(var_43)
    var_45 = 1
    var_46 = 2
    var_47 = [var_45, var_46, var_46]
    var_48 = var_42.validate(var_47)
    var_49 = module_0.Integer()
    var_50 = module_0.Array(var_49)
    var_51 = '1'
    var_52 = '2'
    var_53 = '3'
    var_54 = [var_51, var_52, var_53]
    var_55 = var_50.validate(var_54)
    var_56 = '1'
    var_57 = 'two'
    var_58 = '3'
    var_59 = [var_56, var_57, var_58]
    var_60 = var_50.validate(var_59)
    var_61 = module_0.Integer()
    var_62 = module_0.Integer()
    var_63 = [var_61, var_62]
    var_64 = False
    var_65 = module_0.Array(var_63, var_64)
    var_66 = [var_41, var_57]
    var_67 = var_65.validate(var_66)
    var_68 = 1
    var_69 = 2
    var_70 = 3
    var_71 = [var_68, var_69, var_70]
    var_72 = var_65.validate(var_71)
    var_73 = module_0.Integer()
    var_74 = module_0.Integer()
    var_75 = [var_73, var_74]
    var_76 = module_0.Integer()
    var_77 = module_0.Array(var_75, var_76)
    var_78 = [var_41, var_69, var_70]
    var_79 = var_77.validate(var_78)
    var_80 = 1
    var_81 = 2
    var_82 = 'three'
    var_83 = [var_80, var_81, var_82]
    var_84 = var_77.validate(var_83)
    var_85 = module_0.Array(min_items=var_41)
    var_86 = []
    var_87 = var_85.validate(var_86)
    var_88 = module_0.Array(min_items=var_64)
    var_89 = []
    var_90 = var_88.validate(var_89)
    var_91 = 'id'
    var_92 = module_0.Integer()
    var_93 = {var_91: var_92}
    var_94 = module_0.Object(properties=var_93)
    var_95 = module_0.Array(var_94)
    var_96 = {var_91: var_51}
    var_97 = {var_91: var_52}
    var_98 = [var_96, var_97]
    var_99 = var_95.validate(var_98)
    var_100 = 'id'
    var_101 = '1'
    var_102 = {var_100: var_101}
    var_103 = 'invalid'
    var_104 = {var_100: var_103}
    var_105 = [var_102, var_104]
    var_106 = var_95.validate(var_105)



# Parsed testcases at query #29
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Number()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None
    var_4 = module_0.Number()
    var_5 = None
    var_6 = var_4.validate(var_5)
    var_7 = module_0.Number(coerce_types=var_5)
    var_8 = ''
    var_9 = var_7.validate(var_8)
    assert var_9 is None
    var_10 = module_0.Number()
    var_11 = True
    var_12 = var_10.validate(var_11)
    var_13 = 1.5
    var_14 = var_10.validate(var_13)
    var_15 = False
    var_16 = module_0.Number(coerce_types=var_15)
    var_17 = '123'
    var_18 = var_16.validate(var_17)
    var_19 = module_0.Number()
    var_20 = '123'
    var_21 = var_19.validate(var_20)
    assert var_21 == 123
    var_22 = module_0.Number()
    var_23 = 'inf'
    var_24 = float(var_23)
    var_25 = var_22.validate(var_24)
    var_26 = '0.01'
    var_27 = module_0.Number(precision=var_26)
    var_28 = '1.234'
    var_29 = var_27.validate(var_28)
    var_30 = 5
    var_31 = module_0.Number(minimum=var_30)
    var_32 = var_31.validate(var_30)
    assert var_32 == 5
    var_33 = 4
    var_34 = var_31.validate(var_33)
    var_35 = module_0.Number(exclusive_minimum=var_30)
    var_36 = 6
    var_37 = var_35.validate(var_36)
    assert var_37 == 6
    var_38 = 5
    var_39 = var_35.validate(var_38)
    var_40 = 10
    var_41 = module_0.Number(maximum=var_40)
    var_42 = var_41.validate(var_40)
    assert var_42 == 10
    var_43 = 11
    var_44 = var_41.validate(var_43)
    var_45 = module_0.Number(exclusive_maximum=var_40)
    var_46 = 9
    var_47 = var_45.validate(var_46)
    assert var_47 == 9
    var_48 = 10
    var_49 = var_45.validate(var_48)
    var_50 = 3
    var_51 = module_0.Number(multiple_of=var_50)
    var_52 = var_51.validate(var_36)
    assert var_52 == 6
    var_53 = 7
    var_54 = var_51.validate(var_53)
    var_55 = 0.5
    var_56 = module_0.Number(multiple_of=var_55)
    var_57 = var_56.validate(var_53)
    var_58 = 1.1
    var_59 = var_56.validate(var_58)



# Parsed testcases at query #30
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
    var_17 = module_0.Array(min_items=var_16)
    var_18 = [var_6, var_16]
    var_19 = var_17.validate(var_18)
    var_20 = 1
    var_21 = [var_20]
    var_22 = var_17.validate(var_21)
    var_23 = module_0.Array(max_items=var_21)
    var_24 = [var_6, var_21]
    var_25 = var_23.validate(var_24)
    var_26 = 1
    var_27 = 2
    var_28 = 3
    var_29 = [var_26, var_27, var_28]
    var_30 = var_23.validate(var_29)
    var_31 = module_0.Array(exact_items=var_27)
    var_32 = [var_6, var_27]
    var_33 = var_31.validate(var_32)
    var_34 = 1
    var_35 = [var_34]
    var_36 = var_31.validate(var_35)
    var_37 = 1
    var_38 = 2
    var_39 = 3
    var_40 = [var_37, var_38, var_39]
    var_41 = var_31.validate(var_40)
    var_42 = True
    var_43 = module_0.Array(unique_items=var_42)
    var_44 = [var_42, var_38, var_39]
    var_45 = var_43.validate(var_44)
    var_46 = 1
    var_47 = 2
    var_48 = [var_46, var_47, var_47]
    var_49 = var_43.validate(var_48)
    var_50 = module_0.Integer()
    var_51 = module_0.Array(var_50)
    var_52 = '1'
    var_53 = '2'
    var_54 = '3'
    var_55 = [var_52, var_53, var_54]
    var_56 = var_51.validate(var_55)
    var_57 = '1'
    var_58 = 'two'
    var_59 = '3'
    var_60 = [var_57, var_58, var_59]
    var_61 = var_51.validate(var_60)
    var_62 = module_0.Integer()
    var_63 = module_0.Float()
    var_64 = '1.0'
    var_65 = '2.5'
    var_66 = '3.0'
    var_67 = [var_52, var_65, var_66]
    var_68 = var_51.validate(var_67)
    var_69 = 2.5
    var_70 = '1'
    var_71 = 'two'
    var_72 = '3.0'
    var_73 = [var_70, var_71, var_72]
    var_74 = var_51.validate(var_73)
    var_75 = module_0.Integer()
    var_76 = [var_75]
    var_77 = module_0.Float()
    var_78 = module_0.Array(var_76, var_77)
    var_79 = [var_52, var_65, var_66]
    var_80 = var_78.validate(var_79)
    var_81 = '1'
    var_82 = 'two'
    var_83 = '3.0'
    var_84 = [var_81, var_82, var_83]
    var_85 = var_78.validate(var_84)
    var_86 = module_0.Integer()
    var_87 = [var_86]
    var_88 = module_0.Array(var_87, var_10)
    var_89 = [var_52]
    var_90 = var_88.validate(var_89)
    var_91 = '1'
    var_92 = '2'
    var_93 = [var_91, var_92]
    var_94 = var_88.validate(var_93)
    var_95 = module_0.Array(min_items=var_42)
    var_96 = []
    var_97 = var_95.validate(var_96)
    var_98 = module_0.Array(min_items=var_10)
    var_99 = []
    var_100 = var_98.validate(var_99)
    var_101 = module_0.Integer()
    var_102 = module_0.Array(var_101)
    var_103 = module_0.Array(var_102)
    var_104 = [var_52, var_53]
    var_105 = '4'
    var_106 = [var_54, var_105]
    var_107 = [var_104, var_106]
    var_108 = var_103.validate(var_107)
    var_109 = '1'
    var_110 = 'two'
    var_111 = [var_109, var_110]
    var_112 = '3'
    var_113 = '4'
    var_114 = [var_112, var_113]
    var_115 = [var_111, var_114]
    var_116 = var_103.validate(var_115)



# Parsed testcases at query #31
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'Option A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'Option B'
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
    var_13 = [var_12]
    var_14 = True
    var_15 = module_0.Choice(choices=var_13)
    var_16 = None
    var_17 = var_15.validate(var_16)
    assert var_17 is None
    var_18 = 'a'
    var_19 = 'Option A'
    var_20 = (var_18, var_19)
    var_21 = [var_20]
    var_22 = module_0.Choice(choices=var_21)
    var_23 = None
    var_24 = var_22.validate(var_23)
    var_25 = ''
    var_26 = var_7.validate(var_25)
    var_27 = (var_25, var_26)
    var_28 = [var_27]
    var_29 = module_0.Choice(choices=var_28, coerce_types=var_14)
    var_30 = ''
    var_31 = var_29.validate(var_30)
    assert var_31 is None
    var_32 = (var_25, var_26)
    var_33 = (var_21, var_22)
    var_34 = [var_32, var_33]
    var_35 = module_0.Choice(choices=var_34)
    var_36 = var_35.validate(var_25)
    assert var_36 == 'a'
    var_37 = [var_25, var_26]
    var_38 = [var_21, var_22]
    var_39 = [var_37, var_38]
    var_40 = module_0.Choice(choices=var_39)
    var_41 = var_40.validate(var_25)
    assert var_41 == 'a'



# Parsed testcases at query #32
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
    var_6 = 1.5
    var_7 = var_3.validate(var_6)
    var_8 = True
    var_9 = module_0.Integer()
    var_10 = module_0.Float()
    var_11 = [var_9, var_10]
    var_12 = module_0.Union(var_11)
    var_13 = None
    var_14 = var_12.validate(var_13)
    assert var_14 is None
    var_15 = 'invalid'
    var_16 = var_3.validate(var_15)
    var_17 = module_0.Integer()
    var_18 = module_0.Float()
    var_19 = module_0.Boolean()
    var_20 = [var_17, var_18, var_19]
    var_21 = module_0.Union(var_20)
    var_22 = True
    var_23 = var_21.validate(var_22)
    assert var_23 is True
    var_24 = 'a'
    var_25 = module_0.Integer()
    var_26 = {var_24: var_25}
    var_27 = module_0.Object(properties=var_26)
    var_28 = module_0.Integer()
    var_29 = module_0.Array(var_28)
    var_30 = [var_27, var_29]
    var_31 = module_0.Union(var_30)
    var_32 = {var_24: var_22}
    var_33 = var_31.validate(var_32)
    var_34 = 2
    var_35 = 3
    var_36 = [var_22, var_34, var_35]
    var_37 = var_31.validate(var_36)
    var_38 = module_0.Integer()
    var_39 = 0
    var_40 = module_0.Float(minimum=var_39)
    var_41 = [var_38, var_40]
    var_42 = module_0.Union(var_41)
    var_43 = -1
    var_44 = var_42.validate(var_43)
    var_45 = exc_info.value.messages()[var_39]
    var_46 = var_45.code
    assert var_46 == 'minimum'
    var_47 = module_0.Integer()
    var_48 = module_0.Float()
    var_49 = [var_47, var_48]
    var_50 = module_0.Union(var_49)
    var_51 = 'not a number'
    var_52 = var_50.validate(var_51)
    var_53 = exc_info.value.messages()[var_39]
    var_54 = var_53.code
    assert var_54 == 'union'



# Parsed testcases at query #33
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
    var_17 = module_0.Array(min_items=var_16)
    var_18 = [var_6, var_16]
    var_19 = var_17.validate(var_18)
    var_20 = 1
    var_21 = [var_20]
    var_22 = var_17.validate(var_21)
    var_23 = module_0.Array(max_items=var_21)
    var_24 = [var_6, var_21]
    var_25 = var_23.validate(var_24)
    var_26 = 1
    var_27 = 2
    var_28 = 3
    var_29 = [var_26, var_27, var_28]
    var_30 = var_23.validate(var_29)
    var_31 = module_0.Array(exact_items=var_27)
    var_32 = [var_6, var_27]
    var_33 = var_31.validate(var_32)
    var_34 = 1
    var_35 = [var_34]
    var_36 = var_31.validate(var_35)
    var_37 = 1
    var_38 = 2
    var_39 = 3
    var_40 = [var_37, var_38, var_39]
    var_41 = var_31.validate(var_40)
    var_42 = True
    var_43 = module_0.Array(unique_items=var_42)
    var_44 = [var_42, var_38, var_39]
    var_45 = var_43.validate(var_44)
    var_46 = 1
    var_47 = 2
    var_48 = [var_46, var_47, var_47]
    var_49 = var_43.validate(var_48)
    var_50 = module_0.Integer()
    var_51 = module_0.Array(var_50)
    var_52 = '1'
    var_53 = '2'
    var_54 = '3'
    var_55 = [var_52, var_53, var_54]
    var_56 = var_51.validate(var_55)
    var_57 = '1'
    var_58 = 'two'
    var_59 = '3'
    var_60 = [var_57, var_58, var_59]
    var_61 = var_51.validate(var_60)
    var_62 = module_0.Integer()
    var_63 = module_0.String()
    var_64 = module_0.Boolean()
    var_65 = [var_62, var_63, var_64]
    var_66 = module_0.Array(var_65)
    var_67 = 'two'
    var_68 = 'true'
    var_69 = [var_52, var_67, var_68]
    var_70 = var_66.validate(var_69)
    var_71 = '1'
    var_72 = 'two'
    var_73 = [var_71, var_72]
    var_74 = var_66.validate(var_73)
    var_75 = module_0.Integer()
    var_76 = [var_75]
    var_77 = module_0.Array(var_76, var_10)
    var_78 = [var_52]
    var_79 = var_77.validate(var_78)
    var_80 = '1'
    var_81 = 'extra'
    var_82 = [var_80, var_81]
    var_83 = var_77.validate(var_82)
    var_84 = module_0.Integer()
    var_85 = [var_84]
    var_86 = module_0.String()
    var_87 = module_0.Array(var_85, var_86)
    var_88 = 'extra'
    var_89 = [var_52, var_88]
    var_90 = var_87.validate(var_89)
    var_91 = '1'
    var_92 = 123
    var_93 = [var_91, var_92]
    var_94 = var_87.validate(var_93)
    var_95 = module_0.Array(min_items=var_42)
    var_96 = []
    var_97 = var_95.validate(var_96)
    var_98 = module_0.Array()
    var_99 = []
    var_100 = var_98.validate(var_99)
    var_101 = module_0.Integer()
    var_102 = module_0.Array(var_101)
    var_103 = [var_42, var_97, var_93]
    var_104 = var_102.serialize(var_103)
    var_105 = module_0.Integer()
    var_106 = module_0.String()
    var_107 = [var_105, var_106]
    var_108 = module_0.Array(var_107)
    var_109 = [var_42, var_67]
    var_110 = var_108.serialize(var_109)



# Parsed testcases at query #34
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
    var_5 = module_0.Boolean(coerce_types=var_1)
    var_6 = 'true'
    var_7 = var_5.validate(var_6)
    assert var_7 is True
    var_8 = 'false'
    var_9 = var_5.validate(var_8)
    assert var_9 is False
    var_10 = 'on'
    var_11 = var_5.validate(var_10)
    assert var_11 is True
    var_12 = 'off'
    var_13 = var_5.validate(var_12)
    assert var_13 is False
    var_14 = '1'
    var_15 = var_5.validate(var_14)
    assert var_15 is True
    var_16 = '0'
    var_17 = var_5.validate(var_16)
    assert var_17 is False
    var_18 = ''
    var_19 = var_5.validate(var_18)
    assert var_19 is False
    var_20 = var_5.validate(var_1)
    assert var_20 is True
    var_21 = var_5.validate(var_3)
    assert var_21 is False
    var_22 = module_0.Boolean(coerce_types=var_3)
    var_23 = 'true'
    var_24 = var_22.validate(var_23)
    var_25 = 'false'
    var_26 = var_22.validate(var_25)
    var_27 = 1
    var_28 = var_22.validate(var_27)
    var_29 = 0
    var_30 = var_22.validate(var_29)
    var_31 = module_0.Boolean()
    var_32 = None
    var_33 = var_31.validate(var_32)
    assert var_33 is None
    var_34 = 'null'
    var_35 = var_31.validate(var_34)
    assert var_35 is None
    var_36 = 'none'
    var_37 = var_31.validate(var_36)
    assert var_37 is None
    var_38 = module_0.Boolean()
    var_39 = None
    var_40 = var_38.validate(var_39)
    var_41 = module_0.Boolean()
    var_42 = 'invalid'
    var_43 = var_41.validate(var_42)
    var_44 = 2
    var_45 = var_41.validate(var_44)
    var_46 = []
    var_47 = var_41.validate(var_46)
    var_48 = {}
    var_49 = var_41.validate(var_48)



# Parsed testcases at query #35
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = 42
    var_5 = var_3.validate(var_4)
    assert var_5 == 42
    var_6 = 'hello'
    var_7 = var_3.validate(var_6)
    assert var_7 == 'hello'
    var_8 = True
    var_9 = module_0.Integer()
    var_10 = module_0.String()
    var_11 = [var_9, var_10]
    var_12 = module_0.Union(var_11)
    var_13 = None
    var_14 = var_12.validate(var_13)
    assert var_14 is None
    var_15 = module_0.Integer()
    var_16 = module_0.String()
    var_17 = [var_15, var_16]
    var_18 = module_0.Union(var_17)
    var_19 = 3.14
    var_20 = var_18.validate(var_19)
    var_21 = module_0.Integer()
    var_22 = 5
    var_23 = module_0.String(min_length=var_22)
    var_24 = [var_21, var_23]
    var_25 = module_0.Union(var_24)
    var_26 = 'hi'
    var_27 = var_25.validate(var_26)
    var_28 = 0
    var_29 = exc_info.value.messages()[var_28]
    var_30 = var_29.code
    assert var_30 == 'min_length'
    var_31 = module_0.Integer(minimum=var_28)
    var_32 = module_0.String(min_length=var_22)
    var_33 = [var_31, var_32]
    var_34 = module_0.Union(var_33)
    var_35 = -1
    var_36 = var_34.validate(var_35)
    var_37 = exc_info.value.messages()[var_28]
    var_38 = var_37.code
    assert var_38 == 'minimum'
    var_39 = module_0.Integer()
    var_40 = module_0.String()
    var_41 = [var_39, var_40]
    var_42 = module_0.Union(var_41)
    var_43 = None
    var_44 = var_42.validate(var_43)
    var_45 = exc_info.value.messages()[var_28]
    var_46 = var_45.code
    assert var_46 == 'null'



# Parsed testcases at query #36
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = 123
    var_2 = var_0.validate(var_1)
    assert var_2 == 123
    var_3 = 12.3
    var_4 = var_0.validate(var_3)
    var_5 = '123'
    var_6 = var_0.validate(var_5)
    assert var_6 == 123
    var_7 = True
    var_8 = module_0.Number()
    var_9 = None
    var_10 = var_8.validate(var_9)
    assert var_10 is None
    var_11 = module_0.Number()
    var_12 = None
    var_13 = var_11.validate(var_12)
    var_14 = 10
    var_15 = module_0.Number(minimum=var_14)
    var_16 = var_15.validate(var_14)
    assert var_16 == 10
    var_17 = 11
    var_18 = var_15.validate(var_17)
    assert var_18 == 11
    var_19 = 9
    var_20 = var_15.validate(var_19)
    var_21 = module_0.Number(exclusive_minimum=var_14)
    var_22 = var_21.validate(var_17)
    assert var_22 == 11
    var_23 = 10
    var_24 = var_21.validate(var_23)
    var_25 = module_0.Number(maximum=var_14)
    var_26 = var_25.validate(var_14)
    assert var_26 == 10
    var_27 = 9
    var_28 = var_25.validate(var_27)
    assert var_28 == 9
    var_29 = 11
    var_30 = var_25.validate(var_29)
    var_31 = module_0.Number(exclusive_maximum=var_14)
    var_32 = var_31.validate(var_27)
    assert var_32 == 9
    var_33 = 10
    var_34 = var_31.validate(var_33)
    var_35 = 5
    var_36 = module_0.Number(multiple_of=var_35)
    var_37 = var_36.validate(var_14)
    assert var_37 == 10
    var_38 = 11
    var_39 = var_36.validate(var_38)
    var_40 = '0.01'
    var_41 = module_0.Number(precision=var_40)
    var_42 = 1.234
    var_43 = var_41.validate(var_42)
    var_44 = module_0.Number()
    var_45 = 'abc'
    var_46 = var_44.validate(var_45)
    var_47 = module_0.Number()
    var_48 = True
    var_49 = var_47.validate(var_48)
    var_50 = module_0.Number()
    var_51 = 'inf'
    var_52 = float(var_51)
    var_53 = var_50.validate(var_52)
    var_54 = var_50.validate(var_51)
    assert var_54 == 123
    var_55 = 123.5
    var_56 = var_50.validate(var_55)
    var_57 = False
    var_58 = module_0.Number(coerce_types=var_57)
    var_59 = '123'
    var_60 = var_58.validate(var_59)



# Parsed testcases at query #37
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
    var_17 = module_0.Array(min_items=var_16)
    var_18 = [var_6, var_16]
    var_19 = var_17.validate(var_18)
    var_20 = 1
    var_21 = [var_20]
    var_22 = var_17.validate(var_21)
    var_23 = []
    var_24 = var_17.validate(var_23)
    var_25 = module_0.Array(max_items=var_24)
    var_26 = [var_6, var_24]
    var_27 = var_25.validate(var_26)
    var_28 = 1
    var_29 = 2
    var_30 = 3
    var_31 = [var_28, var_29, var_30]
    var_32 = var_25.validate(var_31)
    var_33 = module_0.Array(exact_items=var_29)
    var_34 = [var_6, var_29]
    var_35 = var_33.validate(var_34)
    var_36 = 1
    var_37 = [var_36]
    var_38 = var_33.validate(var_37)
    var_39 = 1
    var_40 = 2
    var_41 = 3
    var_42 = [var_39, var_40, var_41]
    var_43 = var_33.validate(var_42)
    var_44 = module_0.Integer()
    var_45 = module_0.Array(var_44)
    var_46 = '1'
    var_47 = '2'
    var_48 = '3'
    var_49 = [var_46, var_47, var_48]
    var_50 = var_45.validate(var_49)
    var_51 = '1'
    var_52 = 'two'
    var_53 = '3'
    var_54 = [var_51, var_52, var_53]
    var_55 = var_45.validate(var_54)
    var_56 = module_0.Integer()
    var_57 = module_0.String()
    var_58 = module_0.Boolean()
    var_59 = [var_56, var_57, var_58]
    var_60 = module_0.Array(var_59)
    var_61 = 'two'
    var_62 = 'true'
    var_63 = [var_46, var_61, var_62]
    var_64 = var_60.validate(var_63)
    var_65 = '1'
    var_66 = 'two'
    var_67 = [var_65, var_66]
    var_68 = var_60.validate(var_67)
    var_69 = '1'
    var_70 = 'two'
    var_71 = 'three'
    var_72 = 'four'
    var_73 = [var_69, var_70, var_71, var_72]
    var_74 = var_60.validate(var_73)
    var_75 = module_0.Integer()
    var_76 = module_0.String()
    var_77 = [var_75, var_76]
    var_78 = module_0.Array(var_77, var_10)
    var_79 = [var_46, var_61]
    var_80 = var_78.validate(var_79)
    var_81 = '1'
    var_82 = 'two'
    var_83 = 'three'
    var_84 = [var_81, var_82, var_83]
    var_85 = var_78.validate(var_84)
    var_86 = module_0.Integer()
    var_87 = module_0.String()
    var_88 = [var_86, var_87]
    var_89 = module_0.Boolean()
    var_90 = module_0.Array(var_88, var_89)
    var_91 = [var_46, var_61, var_62]
    var_92 = var_90.validate(var_91)
    var_93 = True
    var_94 = module_0.Array(unique_items=var_93)
    var_95 = [var_93, var_82, var_83]
    var_96 = var_94.validate(var_95)
    var_97 = 1
    var_98 = 2
    var_99 = [var_97, var_98, var_98]
    var_100 = var_94.validate(var_99)
    var_101 = module_0.Array(min_items=var_93)
    var_102 = []
    var_103 = var_101.validate(var_102)
    var_104 = module_0.Integer()
    var_105 = module_0.Array(var_104)
    var_106 = [var_93, var_103, var_99]
    var_107 = var_105.serialize(var_106)
    var_108 = var_105.serialize(var_8)
    assert var_108 is None



# Parsed testcases at query #38
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'Option A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'Option B'
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
    var_13 = [var_12]
    var_14 = True
    var_15 = module_0.Choice(choices=var_13)
    var_16 = None
    var_17 = var_15.validate(var_16)
    assert var_17 is None
    var_18 = None
    var_19 = var_7.validate(var_18)
    var_20 = (var_18, var_19)
    var_21 = [var_20]
    var_22 = module_0.Choice(choices=var_21, coerce_types=var_14)
    var_23 = ''
    var_24 = var_22.validate(var_23)
    assert var_24 is None
    var_25 = ''
    var_26 = var_7.validate(var_25)
    var_27 = (var_25, var_26)
    var_28 = (var_3, var_4)
    var_29 = [var_27, var_28]
    var_30 = module_0.Choice(choices=var_29)
    var_31 = var_30.validate(var_25)
    assert var_31 == 'a'
    var_32 = var_30.validate(var_3)
    assert var_32 == 'b'
    var_33 = [var_25, var_26]
    var_34 = [var_3, var_4]
    var_35 = [var_33, var_34]
    var_36 = module_0.Choice(choices=var_35)
    var_37 = var_36.validate(var_25)
    assert var_37 == 'a'
    var_38 = var_36.validate(var_3)
    assert var_38 == 'b'



# Parsed testcases at query #39
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
    var_6 = 'hello'
    var_7 = var_3.validate(var_6)
    assert var_7 == 'hello'
    var_8 = None
    var_9 = var_3.validate(var_8)
    var_10 = True
    var_11 = module_0.Integer()
    var_12 = module_0.String()
    var_13 = [var_11, var_12]
    var_14 = module_0.Union(var_13)
    var_15 = None
    var_16 = var_14.validate(var_15)
    assert var_16 is None
    var_17 = module_0.Integer()
    var_18 = module_0.Float()
    var_19 = module_0.String()
    var_20 = [var_17, var_18, var_19]
    var_21 = module_0.Union(var_20)
    var_22 = var_21.validate(var_4)
    assert var_22 == 123
    var_23 = 123.45
    var_24 = var_21.validate(var_23)
    var_25 = var_21.validate(var_6)
    assert var_25 == 'hello'
    var_26 = []
    var_27 = var_21.validate(var_26)
    var_28 = module_0.Integer()
    var_29 = module_0.Array(var_28)
    var_30 = 'key'
    var_31 = module_0.Integer()
    var_32 = {var_30: var_31}
    var_33 = module_0.Object(properties=var_32)
    var_34 = [var_29, var_33]
    var_35 = module_0.Union(var_34)
    var_36 = 2
    var_37 = 3
    var_38 = [var_10, var_36, var_37]
    var_39 = var_35.validate(var_38)
    var_40 = {var_30: var_4}
    var_41 = var_35.validate(var_40)
    var_42 = 0
    var_43 = module_0.Integer(minimum=var_42)
    var_44 = module_0.Integer(maximum=var_42)
    var_45 = [var_43, var_44]
    var_46 = module_0.Union(var_45)
    var_47 = -1
    var_48 = var_46.validate(var_47)
    var_49 = exc_info.value.messages()[var_42]
    var_50 = var_49.code
    assert var_50 == 'minimum'
    var_51 = module_0.Integer(minimum=var_42)
    var_52 = module_0.Integer(maximum=var_42)
    var_53 = [var_51, var_52]
    var_54 = module_0.Union(var_53)
    var_55 = 1
    var_56 = var_54.validate(var_55)
    var_57 = exc_info.value.messages()[var_42]
    var_58 = var_57.code
    assert var_58 == 'maximum'



# Parsed testcases at query #40
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Const(var_0)
    var_2 = None
    var_3 = module_0.Const(var_2)
    var_4 = 42
    var_5 = True
    var_6 = module_0.Const(var_4)
    var_7 = 'hello'
    var_8 = module_0.Const(var_7)
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = [var_9, var_10, var_11]
    var_13 = module_0.Const(var_12)
    var_14 = 'key'
    var_15 = 'value'
    var_16 = {var_14: var_15}
    var_17 = module_0.Const(var_16)



# Parsed testcases at query #41
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'Option A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'Option B'
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
    var_13 = [var_12]
    var_14 = True
    var_15 = module_0.Choice(choices=var_13)
    var_16 = None
    var_17 = var_15.validate(var_16)
    assert var_17 is None
    var_18 = None
    var_19 = var_7.validate(var_18)
    var_20 = (var_18, var_19)
    var_21 = [var_20]
    var_22 = module_0.Choice(choices=var_21, coerce_types=var_14)
    var_23 = ''
    var_24 = var_22.validate(var_23)
    assert var_24 is None
    var_25 = ''
    var_26 = var_7.validate(var_25)
    var_27 = (var_25, var_26)
    var_28 = (var_3, var_4)
    var_29 = [var_27, var_28]
    var_30 = module_0.Choice(choices=var_29)
    var_31 = var_30.validate(var_25)
    assert var_31 == 'a'
    var_32 = var_30.validate(var_3)
    assert var_32 == 'b'
    var_33 = [var_25, var_26]
    var_34 = [var_3, var_4]
    var_35 = [var_33, var_34]
    var_36 = module_0.Choice(choices=var_35)
    var_37 = var_36.validate(var_25)
    assert var_37 == 'a'
    var_38 = var_36.validate(var_3)
    assert var_38 == 'b'



# Parsed testcases at query #42
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Number()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None
    var_4 = module_0.Number()
    var_5 = None
    var_6 = var_4.validate(var_5)
    var_7 = module_0.Number(coerce_types=var_5)
    var_8 = ''
    var_9 = var_7.validate(var_8)
    assert var_9 is None
    var_10 = module_0.Number()
    var_11 = True
    var_12 = var_10.validate(var_11)
    var_13 = 3.14
    var_14 = var_10.validate(var_13)
    var_15 = module_0.Number()
    var_16 = 'not_a_number'
    var_17 = var_15.validate(var_16)
    var_18 = module_0.Number()
    var_19 = 'inf'
    var_20 = float(var_19)
    var_21 = var_18.validate(var_20)
    var_22 = '-inf'
    var_23 = float(var_22)
    var_24 = var_18.validate(var_23)
    var_25 = 'nan'
    var_26 = float(var_25)
    var_27 = var_18.validate(var_26)
    var_28 = '0.01'
    var_29 = module_0.Number(precision=var_28)
    var_30 = '3.14159'
    var_31 = var_29.validate(var_30)
    var_32 = 5
    var_33 = module_0.Number(minimum=var_32)
    var_34 = var_33.validate(var_32)
    assert var_34 == 5
    var_35 = 4
    var_36 = var_33.validate(var_35)
    var_37 = module_0.Number(exclusive_minimum=var_32)
    var_38 = 6
    var_39 = var_37.validate(var_38)
    assert var_39 == 6
    var_40 = 5
    var_41 = var_37.validate(var_40)
    var_42 = 10
    var_43 = module_0.Number(maximum=var_42)
    var_44 = var_43.validate(var_42)
    assert var_44 == 10
    var_45 = 11
    var_46 = var_43.validate(var_45)
    var_47 = module_0.Number(exclusive_maximum=var_42)
    var_48 = 9
    var_49 = var_47.validate(var_48)
    assert var_49 == 9
    var_50 = 10
    var_51 = var_47.validate(var_50)
    var_52 = 3
    var_53 = module_0.Number(multiple_of=var_52)
    var_54 = var_53.validate(var_48)
    assert var_54 == 9
    var_55 = 10
    var_56 = var_53.validate(var_55)
    var_57 = 0.5
    var_58 = module_0.Number(multiple_of=var_57)
    var_59 = 2.0
    var_60 = var_58.validate(var_59)
    var_61 = 2.1
    var_62 = var_58.validate(var_61)
    var_63 = module_0.Number()
    var_64 = 42
    var_65 = var_63.validate(var_64)
    assert var_65 == 42
    var_66 = module_0.Number()
    var_67 = 3.14
    var_68 = var_66.validate(var_67)
    var_69 = module_0.Number()
    var_70 = '42'
    var_71 = var_69.validate(var_70)
    assert var_71 == 42



# Parsed testcases at query #43
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
    var_6 = 'hello'
    var_7 = var_3.validate(var_6)
    assert var_7 == 'hello'
    var_8 = 12.3
    var_9 = var_3.validate(var_8)
    var_10 = True
    var_11 = module_0.Integer()
    var_12 = [var_11, var_1]
    var_13 = module_0.Union(var_12)
    var_14 = None
    var_15 = var_13.validate(var_14)
    assert var_15 is None
    var_16 = None
    var_17 = var_3.validate(var_16)
    var_18 = 'name'
    var_19 = module_0.String()
    var_20 = {var_18: var_19}
    var_21 = module_0.Object(properties=var_20)
    var_22 = [var_21, var_1]
    var_23 = module_0.Union(var_22)
    var_24 = 'test'
    var_25 = {var_18: var_24}
    var_26 = var_23.validate(var_25)
    var_27 = 0
    var_28 = module_0.Integer(minimum=var_27)
    var_29 = [var_28, var_1]
    var_30 = module_0.Union(var_29)
    var_31 = -1
    var_32 = var_30.validate(var_31)
    var_33 = module_0.Float()
    var_34 = [var_0, var_1, var_33]
    var_35 = module_0.Union(var_34)
    var_36 = 12.5
    var_37 = var_35.validate(var_36)
    var_38 = module_0.String(coerce_types=var_10)
    var_39 = [var_0, var_38]
    var_40 = module_0.Union(var_39)
    var_41 = ''
    var_42 = var_40.validate(var_41)
    assert var_42 == ''
    var_43 = True
    var_44 = var_3.validate(var_43)



# Parsed testcases at query #44
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'Option A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'Option B'
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
    var_13 = [var_12]
    var_14 = True
    var_15 = module_0.Choice(choices=var_13)
    var_16 = None
    var_17 = var_15.validate(var_16)
    assert var_17 is None
    var_18 = None
    var_19 = var_7.validate(var_18)
    var_20 = (var_18, var_19)
    var_21 = [var_20]
    var_22 = module_0.Choice(choices=var_21, coerce_types=var_14)
    var_23 = ''
    var_24 = var_22.validate(var_23)
    assert var_24 is None
    var_25 = ''
    var_26 = var_7.validate(var_25)
    var_27 = (var_25, var_26)
    var_28 = (var_3, var_4)
    var_29 = [var_27, var_28]
    var_30 = module_0.Choice(choices=var_29)
    var_31 = var_30.validate(var_25)
    assert var_31 == 'a'
    var_32 = [var_25, var_26]
    var_33 = [var_3, var_4]
    var_34 = [var_32, var_33]
    var_35 = module_0.Choice(choices=var_34)
    var_36 = var_35.validate(var_25)
    assert var_36 == 'a'



# Parsed testcases at query #45
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'Option A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'Option B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = module_0.Choice(choices=var_6)
    var_8 = var_7.validate(var_0)
    assert var_8 == 'a'
    var_9 = var_7.validate(var_3)
    assert var_9 == 'b'
    var_10 = (var_0, var_1)
    var_11 = (var_3, var_4)
    var_12 = [var_10, var_11]
    var_13 = module_0.Choice(choices=var_12)
    var_14 = var_13.validate(var_0)
    assert var_14 == 'a'
    var_15 = (var_0, var_1)
    var_16 = (var_3, var_4)
    var_17 = [var_15, var_16]
    var_18 = module_0.Choice(choices=var_17)
    var_19 = 'c'
    var_20 = var_18.validate(var_19)
    var_21 = (var_19, var_20)
    var_22 = (var_3, var_4)
    var_23 = [var_21, var_22]
    var_24 = True
    var_25 = module_0.Choice(choices=var_23)
    var_26 = None
    var_27 = var_25.validate(var_26)
    assert var_27 is None
    var_28 = (var_19, var_20)
    var_29 = (var_3, var_4)
    var_30 = [var_28, var_29]
    var_31 = False
    var_32 = module_0.Choice(choices=var_30)
    var_33 = None
    var_34 = var_32.validate(var_33)
    var_35 = (var_33, var_34)
    var_36 = (var_3, var_4)
    var_37 = [var_35, var_36]
    var_38 = module_0.Choice(choices=var_37, coerce_types=var_24)
    var_39 = ''
    var_40 = var_38.validate(var_39)
    assert var_40 is None
    var_41 = (var_33, var_34)
    var_42 = (var_3, var_4)
    var_43 = [var_41, var_42]
    var_44 = module_0.Choice(choices=var_43, coerce_types=var_24)
    var_45 = ''
    var_46 = var_44.validate(var_45)
    var_47 = (var_45, var_46)
    var_48 = (var_3, var_4)
    var_49 = [var_47, var_48]
    var_50 = module_0.Choice(choices=var_49, coerce_types=var_31)
    var_51 = ''
    var_52 = var_50.validate(var_51)



# Parsed testcases at query #46
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
    var_21 = module_0.Boolean(coerce_types=var_3)
    var_22 = 'true'
    var_23 = var_21.validate(var_22)
    var_24 = 1
    var_25 = var_21.validate(var_24)
    var_26 = module_0.Boolean()
    var_27 = None
    var_28 = var_26.validate(var_27)
    assert var_28 is None
    var_29 = 'null'
    var_30 = var_26.validate(var_29)
    assert var_30 is None
    var_31 = 'none'
    var_32 = var_26.validate(var_31)
    assert var_32 is None
    var_33 = var_26.validate(var_17)
    assert var_33 is None
    var_34 = None
    var_35 = var_0.validate(var_34)
    var_36 = 'invalid'
    var_37 = var_0.validate(var_36)
    var_38 = 2
    var_39 = var_0.validate(var_38)
    var_40 = []
    var_41 = var_0.validate(var_40)



# Parsed testcases at query #47
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
    var_14 = 'not an array'
    var_15 = var_11.validate(var_14)
    var_16 = module_0.Array(min_items=var_15)
    var_17 = [var_6, var_15]
    var_18 = var_16.validate(var_17)
    var_19 = 1
    var_20 = [var_19]
    var_21 = var_16.validate(var_20)
    var_22 = module_0.Array(max_items=var_20)
    var_23 = [var_6, var_20]
    var_24 = var_22.validate(var_23)
    var_25 = 1
    var_26 = 2
    var_27 = 3
    var_28 = [var_25, var_26, var_27]
    var_29 = var_22.validate(var_28)
    var_30 = module_0.Array(exact_items=var_26)
    var_31 = [var_6, var_26]
    var_32 = var_30.validate(var_31)
    var_33 = 1
    var_34 = [var_33]
    var_35 = var_30.validate(var_34)
    var_36 = 1
    var_37 = 2
    var_38 = 3
    var_39 = [var_36, var_37, var_38]
    var_40 = var_30.validate(var_39)
    var_41 = True
    var_42 = module_0.Array(unique_items=var_41)
    var_43 = [var_41, var_37, var_38]
    var_44 = var_42.validate(var_43)
    var_45 = 1
    var_46 = 2
    var_47 = [var_45, var_46, var_46]
    var_48 = var_42.validate(var_47)
    var_49 = module_0.Integer()
    var_50 = module_0.Array(var_49)
    var_51 = '1'
    var_52 = '2'
    var_53 = '3'
    var_54 = [var_51, var_52, var_53]
    var_55 = var_50.validate(var_54)
    var_56 = '1'
    var_57 = 'two'
    var_58 = '3'
    var_59 = [var_56, var_57, var_58]
    var_60 = var_50.validate(var_59)
    var_61 = module_0.Integer()
    var_62 = module_0.String()
    var_63 = module_0.Boolean()
    var_64 = [var_61, var_62, var_63]
    var_65 = module_0.Array(var_64)
    var_66 = 'two'
    var_67 = 'true'
    var_68 = [var_51, var_66, var_67]
    var_69 = var_65.validate(var_68)
    var_70 = '1'
    var_71 = 'two'
    var_72 = [var_70, var_71]
    var_73 = var_65.validate(var_72)
    var_74 = module_0.Integer()
    var_75 = module_0.String()
    var_76 = [var_74, var_75]
    var_77 = module_0.Array(var_76, var_10)
    var_78 = [var_51, var_66]
    var_79 = var_77.validate(var_78)
    var_80 = '1'
    var_81 = 'two'
    var_82 = 'extra'
    var_83 = [var_80, var_81, var_82]
    var_84 = var_77.validate(var_83)
    var_85 = module_0.Integer()
    var_86 = module_0.String()
    var_87 = [var_85, var_86]
    var_88 = module_0.Boolean()
    var_89 = module_0.Array(var_87, var_88)
    var_90 = [var_51, var_66, var_67]
    var_91 = var_89.validate(var_90)
    var_92 = '1'
    var_93 = 'two'
    var_94 = 'not a bool'
    var_95 = [var_92, var_93, var_94]
    var_96 = var_89.validate(var_95)
    var_97 = module_0.Array(min_items=var_41)
    var_98 = []
    var_99 = var_97.validate(var_98)
    var_100 = module_0.Array(min_items=var_10)
    var_101 = []
    var_102 = var_100.validate(var_101)



# Parsed testcases at query #48
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
    var_10 = module_0.Array()
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = module_0.Array()
    var_14 = 'not a list'
    var_15 = var_13.validate(var_14)
    var_16 = module_0.Array(min_items=var_15)
    var_17 = [var_6, var_15]
    var_18 = var_16.validate(var_17)
    var_19 = 1
    var_20 = [var_19]
    var_21 = var_16.validate(var_20)
    var_22 = module_0.Array(max_items=var_20)
    var_23 = [var_6, var_20]
    var_24 = var_22.validate(var_23)
    var_25 = 1
    var_26 = 2
    var_27 = 3
    var_28 = [var_25, var_26, var_27]
    var_29 = var_22.validate(var_28)
    var_30 = module_0.Array(exact_items=var_26)
    var_31 = [var_6, var_26]
    var_32 = var_30.validate(var_31)
    var_33 = 1
    var_34 = [var_33]
    var_35 = var_30.validate(var_34)
    var_36 = 1
    var_37 = 2
    var_38 = 3
    var_39 = [var_36, var_37, var_38]
    var_40 = var_30.validate(var_39)
    var_41 = True
    var_42 = module_0.Array(unique_items=var_41)
    var_43 = [var_41, var_37, var_38]
    var_44 = var_42.validate(var_43)
    var_45 = 1
    var_46 = 2
    var_47 = [var_45, var_46, var_46]
    var_48 = var_42.validate(var_47)
    var_49 = module_0.Integer()
    var_50 = module_0.Array(var_49)
    var_51 = '1'
    var_52 = '2'
    var_53 = '3'
    var_54 = [var_51, var_52, var_53]
    var_55 = var_50.validate(var_54)
    var_56 = '1'
    var_57 = 'two'
    var_58 = '3'
    var_59 = [var_56, var_57, var_58]
    var_60 = var_50.validate(var_59)
    var_61 = module_0.Integer()
    var_62 = module_0.String()
    var_63 = module_0.Boolean()
    var_64 = [var_61, var_62, var_63]
    var_65 = module_0.Array(var_64)
    var_66 = 'two'
    var_67 = 'true'
    var_68 = [var_51, var_66, var_67]
    var_69 = var_65.validate(var_68)
    var_70 = '1'
    var_71 = 'two'
    var_72 = [var_70, var_71]
    var_73 = var_65.validate(var_72)
    var_74 = module_0.Integer()
    var_75 = module_0.String()
    var_76 = [var_74, var_75]
    var_77 = False
    var_78 = module_0.Array(var_76, var_77)
    var_79 = [var_51, var_66]
    var_80 = var_78.validate(var_79)
    var_81 = '1'
    var_82 = 'two'
    var_83 = 'extra'
    var_84 = [var_81, var_82, var_83]
    var_85 = var_78.validate(var_84)
    var_86 = module_0.Integer()
    var_87 = module_0.String()
    var_88 = [var_86, var_87]
    var_89 = module_0.Boolean()
    var_90 = module_0.Array(var_88, var_89)
    var_91 = [var_51, var_66, var_67]
    var_92 = var_90.validate(var_91)
    var_93 = '1'
    var_94 = 'two'
    var_95 = 'not a bool'
    var_96 = [var_93, var_94, var_95]
    var_97 = var_90.validate(var_96)
    var_98 = module_0.Array(min_items=var_41)
    var_99 = []
    var_100 = var_98.validate(var_99)
    var_101 = module_0.Integer()
    var_102 = module_0.String()
    var_103 = [var_101, var_102]
    var_104 = module_0.Array(var_103)
    var_105 = [var_41, var_66]
    var_106 = var_104.serialize(var_105)
    var_107 = module_0.Array()
    var_108 = var_107.serialize(var_8)
    assert var_108 is None



# Parsed testcases at query #49
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
    var_10 = (var_0, var_1)
    var_11 = (var_3, var_4)
    var_12 = [var_10, var_11]
    var_13 = module_0.Choice(choices=var_12)
    var_14 = var_13.validate(var_0)
    assert var_14 == 'a'
    var_15 = (var_0, var_1)
    var_16 = (var_3, var_4)
    var_17 = [var_15, var_16]
    var_18 = module_0.Choice(choices=var_17)
    var_19 = 'c'
    var_20 = var_18.validate(var_19)
    var_21 = (var_19, var_20)
    var_22 = (var_3, var_4)
    var_23 = [var_21, var_22]
    var_24 = True
    var_25 = module_0.Choice(choices=var_23)
    var_26 = None
    var_27 = var_25.validate(var_26)
    assert var_27 is None
    var_28 = (var_19, var_20)
    var_29 = (var_3, var_4)
    var_30 = [var_28, var_29]
    var_31 = False
    var_32 = module_0.Choice(choices=var_30)
    var_33 = None
    var_34 = var_32.validate(var_33)
    var_35 = (var_33, var_34)
    var_36 = (var_3, var_4)
    var_37 = [var_35, var_36]
    var_38 = module_0.Choice(choices=var_37, coerce_types=var_24)
    var_39 = ''
    var_40 = var_38.validate(var_39)
    assert var_40 is None
    var_41 = (var_33, var_34)
    var_42 = (var_3, var_4)
    var_43 = [var_41, var_42]
    var_44 = module_0.Choice(choices=var_43, coerce_types=var_24)
    var_45 = ''
    var_46 = var_44.validate(var_45)



# Parsed testcases at query #50
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
    var_5 = module_0.Boolean()
    var_6 = None
    var_7 = var_5.validate(var_6)
    assert var_7 is None
    var_8 = module_0.Boolean(coerce_types=var_1)
    var_9 = 'true'
    var_10 = var_8.validate(var_9)
    assert var_10 is True
    var_11 = 'false'
    var_12 = var_8.validate(var_11)
    assert var_12 is False
    var_13 = 'on'
    var_14 = var_8.validate(var_13)
    assert var_14 is True
    var_15 = 'off'
    var_16 = var_8.validate(var_15)
    assert var_16 is False
    var_17 = '1'
    var_18 = var_8.validate(var_17)
    assert var_18 is True
    var_19 = '0'
    var_20 = var_8.validate(var_19)
    assert var_20 is False
    var_21 = var_8.validate(var_1)
    assert var_21 is True
    var_22 = var_8.validate(var_3)
    assert var_22 is False
    var_23 = module_0.Boolean(coerce_types=var_1)
    var_24 = ''
    var_25 = var_23.validate(var_24)
    assert var_25 is None
    var_26 = 'null'
    var_27 = var_23.validate(var_26)
    assert var_27 is None
    var_28 = 'none'
    var_29 = var_23.validate(var_28)
    assert var_29 is None
    var_30 = module_0.Boolean(coerce_types=var_3)
    var_31 = 'true'
    var_32 = var_30.validate(var_31)
    var_33 = 1
    var_34 = var_30.validate(var_33)
    var_35 = 'invalid'
    var_36 = var_30.validate(var_35)
    var_37 = module_0.Boolean()
    var_38 = None
    var_39 = var_37.validate(var_38)



# Parsed testcases at query #51
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'Option A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'Option B'
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
    var_13 = [var_12]
    var_14 = True
    var_15 = module_0.Choice(choices=var_13)
    var_16 = None
    var_17 = var_15.validate(var_16)
    assert var_17 is None
    var_18 = None
    var_19 = var_7.validate(var_18)
    var_20 = (var_18, var_19)
    var_21 = [var_20]
    var_22 = module_0.Choice(choices=var_21, coerce_types=var_14)
    var_23 = ''
    var_24 = var_22.validate(var_23)
    assert var_24 is None
    var_25 = ''
    var_26 = var_7.validate(var_25)
    var_27 = (var_25, var_26)
    var_28 = (var_3, var_4)
    var_29 = [var_27, var_28]
    var_30 = module_0.Choice(choices=var_29)
    var_31 = var_30.validate(var_25)
    assert var_31 == 'a'
    var_32 = [var_25, var_26]
    var_33 = [var_3, var_4]
    var_34 = [var_32, var_33]
    var_35 = module_0.Choice(choices=var_34)
    var_36 = var_35.validate(var_25)
    assert var_36 == 'a'



# Parsed testcases at query #52
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None
    var_4 = module_0.String(allow_blank=var_0)
    var_5 = var_4.validate(var_2)
    assert var_5 == ''
    var_6 = ''
    var_7 = var_4.validate(var_6)
    assert var_7 == ''
    var_8 = module_0.String()
    var_9 = 123
    var_10 = var_8.validate(var_9)
    var_11 = module_0.String()
    var_12 = 'a\x00b'
    var_13 = var_11.validate(var_12)
    assert var_13 == 'ab'
    var_14 = module_0.String(trim_whitespace=var_9)
    var_15 = '  hello  '
    var_16 = var_14.validate(var_15)
    assert var_16 == 'hello'
    var_17 = module_0.String(allow_blank=var_9, coerce_types=var_9)
    var_18 = var_17.validate(var_6)
    assert var_18 is None
    var_19 = 3
    var_20 = module_0.String(min_length=var_19)
    var_21 = 'abc'
    var_22 = var_20.validate(var_21)
    assert var_22 == 'abc'
    var_23 = 'ab'
    var_24 = var_20.validate(var_23)
    var_25 = module_0.String(max_length=var_19)
    var_26 = var_25.validate(var_21)
    assert var_26 == 'abc'
    var_27 = 'abcd'
    var_28 = var_25.validate(var_27)
    var_29 = '^[a-z]+$'
    var_30 = module_0.String(pattern=var_29)
    var_31 = var_30.validate(var_21)
    assert var_31 == 'abc'
    var_32 = 'abc1'
    var_33 = var_30.validate(var_32)
    var_34 = 'email'
    var_35 = module_0.String(format=var_34)
    var_36 = 'test@example.com'
    var_37 = var_35.validate(var_36)
    assert var_37 == 'test@example.com'
    var_38 = 'invalid-email'
    var_39 = var_35.validate(var_38)
    var_40 = module_0.String(format=var_34)
    var_41 = var_40.validate(var_36)
    assert var_41 == 'test@example.com'



# Parsed testcases at query #53
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
    var_17 = module_0.String(trim_whitespace=var_3)
    var_18 = '  hello  '
    var_19 = var_17.validate(var_18)
    assert var_19 == 'hello'
    var_20 = module_0.String(trim_whitespace=var_7)
    var_21 = var_20.validate(var_18)
    assert var_21 == '  hello  '
    var_22 = 5
    var_23 = module_0.String(max_length=var_22)
    var_24 = var_23.validate(var_15)
    assert var_24 == 'hello'
    var_25 = 'hello world'
    var_26 = var_23.validate(var_25)
    var_27 = module_0.String(min_length=var_22)
    var_28 = 'hello world'
    var_29 = var_27.validate(var_28)
    assert var_29 == 'hello world'
    var_30 = 'hi'
    var_31 = var_27.validate(var_30)
    var_32 = '^[a-z]+$'
    var_33 = module_0.String(pattern=var_32)
    var_34 = var_33.validate(var_30)
    assert var_34 == 'hello'
    var_35 = 'hello123'
    var_36 = var_33.validate(var_35)
    var_37 = 'email'
    var_38 = module_0.String(format=var_37)
    var_39 = 'test@example.com'
    var_40 = var_38.validate(var_39)
    assert var_40 == 'test@example.com'
    var_41 = 'notanemail'
    var_42 = var_38.validate(var_41)
    var_43 = module_0.String(allow_blank=var_3, coerce_types=var_3)
    var_44 = var_43.validate(var_5)
    assert var_44 == ''
    var_45 = module_0.String(coerce_types=var_3)
    var_46 = var_45.validate(var_12)
    assert var_46 is None
    var_47 = module_0.String()
    var_48 = 'hello\x00world'
    var_49 = var_47.validate(var_48)
    assert var_49 == 'helloworld'
    var_50 = 'default'
    var_51 = module_0.String()
    var_52 = lambda : var_50
    var_53 = module_0.String()



# Parsed testcases at query #54
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = module_0.Const(var_0)
    var_2 = None
    var_3 = module_0.Const(var_2)
    var_4 = var_1.validate(var_0)
    assert var_4 == 'test_value'
    var_5 = var_3.validate(var_2)
    assert var_5 is None
    var_6 = 'wrong_value'
    var_7 = var_1.validate(var_6)
    var_8 = 'not_null'
    var_9 = var_3.validate(var_8)
    var_10 = 'test_value'
    var_11 = True
    var_12 = module_0.Const(var_10)



# Parsed testcases at query #55
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Number()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None
    var_4 = module_0.Number()
    var_5 = None
    var_6 = var_4.validate(var_5)
    var_7 = module_0.Number(coerce_types=var_5)
    var_8 = ''
    var_9 = var_7.validate(var_8)
    assert var_9 is None
    var_10 = module_0.Number()
    var_11 = True
    var_12 = var_10.validate(var_11)
    var_13 = False
    var_14 = module_0.Number(coerce_types=var_13)
    var_15 = 'abc'
    var_16 = var_14.validate(var_15)
    var_17 = module_0.Number()
    var_18 = 'inf'
    var_19 = float(var_18)
    var_20 = var_17.validate(var_19)
    var_21 = 10
    var_22 = var_17.validate(var_21)
    assert var_22 == 10
    var_23 = 10.5
    var_24 = var_17.validate(var_23)
    var_25 = 10.5
    var_26 = var_17.validate(var_25)
    var_27 = module_0.Number(minimum=var_21)
    var_28 = var_27.validate(var_21)
    assert var_28 == 10
    var_29 = 9
    var_30 = var_27.validate(var_29)
    var_31 = module_0.Number(exclusive_minimum=var_21)
    var_32 = 11
    var_33 = var_31.validate(var_32)
    assert var_33 == 11
    var_34 = 10
    var_35 = var_31.validate(var_34)
    var_36 = module_0.Number(maximum=var_21)
    var_37 = var_36.validate(var_21)
    assert var_37 == 10
    var_38 = 11
    var_39 = var_36.validate(var_38)
    var_40 = module_0.Number(exclusive_maximum=var_21)
    var_41 = 9
    var_42 = var_40.validate(var_41)
    assert var_42 == 9
    var_43 = 10
    var_44 = var_40.validate(var_43)
    var_45 = 5
    var_46 = module_0.Number(multiple_of=var_45)
    var_47 = var_46.validate(var_21)
    assert var_47 == 10
    var_48 = 11
    var_49 = var_46.validate(var_48)
    var_50 = 0.5
    var_51 = module_0.Number(multiple_of=var_50)
    var_52 = var_51.validate(var_48)
    var_53 = 1.1
    var_54 = var_51.validate(var_53)
    var_55 = '0.01'
    var_56 = module_0.Number(precision=var_55)
    var_57 = 10.123
    var_58 = var_56.validate(var_57)
    var_59 = module_0.Number(coerce_types=var_53)
    var_60 = '10'
    var_61 = var_59.validate(var_60)
    assert var_61 == 10



# Parsed testcases at query #56
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
    var_11 = 123
    var_12 = var_8.validate(var_11)
    var_13 = 'key'
    var_14 = 'value'
    var_15 = {var_13: var_14}
    var_16 = var_8.validate(var_15)
    var_17 = 2
    var_18 = 4
    var_19 = module_0.Array(min_items=var_17, max_items=var_18)
    var_20 = [var_13, var_17]
    var_21 = var_19.validate(var_20)
    var_22 = 3
    var_23 = [var_13, var_17, var_22, var_18]
    var_24 = var_19.validate(var_23)
    var_25 = 1
    var_26 = [var_25]
    var_27 = var_19.validate(var_26)
    var_28 = 1
    var_29 = 2
    var_30 = 3
    var_31 = 4
    var_32 = 5
    var_33 = [var_28, var_29, var_30, var_31, var_32]
    var_34 = var_19.validate(var_33)
    var_35 = module_0.Array(exact_items=var_22)
    var_36 = [var_28, var_32, var_22]
    var_37 = var_35.validate(var_36)
    var_38 = 1
    var_39 = 2
    var_40 = [var_38, var_39]
    var_41 = var_35.validate(var_40)
    var_42 = 1
    var_43 = 2
    var_44 = 3
    var_45 = 4
    var_46 = [var_42, var_43, var_44, var_45]
    var_47 = var_35.validate(var_46)
    var_48 = module_0.Integer()
    var_49 = module_0.Array(var_48)
    var_50 = [var_42, var_46, var_22]
    var_51 = var_49.validate(var_50)
    var_52 = 1
    var_53 = 'two'
    var_54 = 3
    var_55 = [var_52, var_53, var_54]
    var_56 = var_49.validate(var_55)
    var_57 = module_0.Integer()
    var_58 = module_0.Integer()
    var_59 = [var_57, var_58]
    var_60 = module_0.Array(var_59, var_55)
    var_61 = [var_52, var_56]
    var_62 = var_60.validate(var_61)
    var_63 = 1
    var_64 = 2
    var_65 = 3
    var_66 = [var_63, var_64, var_65]
    var_67 = var_60.validate(var_66)
    var_68 = module_0.Integer()
    var_69 = module_0.Integer()
    var_70 = [var_68, var_69]
    var_71 = module_0.Integer()
    var_72 = module_0.Array(var_70, var_71)
    var_73 = [var_63, var_67, var_22]
    var_74 = var_72.validate(var_73)
    var_75 = 1
    var_76 = 2
    var_77 = 'three'
    var_78 = [var_75, var_76, var_77]
    var_79 = var_72.validate(var_78)
    var_80 = module_0.Array(unique_items=var_75)
    var_81 = [var_75, var_79, var_22]
    var_82 = var_80.validate(var_81)
    var_83 = 1
    var_84 = 2
    var_85 = [var_83, var_84, var_84]
    var_86 = var_80.validate(var_85)
    var_87 = module_0.Array(min_items=var_83)
    var_88 = []
    var_89 = var_87.validate(var_88)
    var_90 = module_0.Array()
    var_91 = []
    var_92 = var_90.validate(var_91)
    var_93 = 'name'
    var_94 = module_0.String()
    var_95 = {var_93: var_94}
    var_96 = module_0.Object(properties=var_95)
    var_97 = module_0.Array(var_96)
    var_98 = 'test'
    var_99 = {var_93: var_98}
    var_100 = [var_99]
    var_101 = var_97.validate(var_100)
    var_102 = 'name'
    var_103 = 'test'
    var_104 = {var_102: var_103}
    var_105 = 'invalid'
    var_106 = 'object'
    var_107 = {var_105: var_106}
    var_108 = [var_104, var_107]
    var_109 = var_97.validate(var_108)



# Parsed testcases at query #57
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'Option A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'Option B'
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
    var_13 = [var_12]
    var_14 = True
    var_15 = module_0.Choice(choices=var_13)
    var_16 = None
    var_17 = var_15.validate(var_16)
    assert var_17 is None
    var_18 = None
    var_19 = var_7.validate(var_18)
    var_20 = (var_18, var_19)
    var_21 = [var_20]
    var_22 = module_0.Choice(choices=var_21, coerce_types=var_14)
    var_23 = ''
    var_24 = var_22.validate(var_23)
    assert var_24 is None
    var_25 = ''
    var_26 = var_7.validate(var_25)
    var_27 = (var_25, var_26)
    var_28 = (var_3, var_4)
    var_29 = [var_27, var_28]
    var_30 = module_0.Choice(choices=var_29)
    var_31 = var_30.validate(var_25)
    assert var_31 == 'a'
    var_32 = var_30.validate(var_3)
    assert var_32 == 'b'
    var_33 = [var_25, var_26]
    var_34 = [var_3, var_4]
    var_35 = [var_33, var_34]
    var_36 = module_0.Choice(choices=var_35)
    var_37 = var_36.validate(var_25)
    assert var_37 == 'a'
    var_38 = var_36.validate(var_3)
    assert var_38 == 'b'



# Parsed testcases at query #58
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
    var_10 = module_0.Array()
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = module_0.Array()
    var_14 = 'not an array'
    var_15 = var_13.validate(var_14)
    var_16 = module_0.Array(min_items=var_15)
    var_17 = [var_6, var_15]
    var_18 = var_16.validate(var_17)
    var_19 = 1
    var_20 = [var_19]
    var_21 = var_16.validate(var_20)
    var_22 = module_0.Array(max_items=var_20)
    var_23 = [var_6, var_20]
    var_24 = var_22.validate(var_23)
    var_25 = 1
    var_26 = 2
    var_27 = 3
    var_28 = [var_25, var_26, var_27]
    var_29 = var_22.validate(var_28)
    var_30 = module_0.Array(exact_items=var_26)
    var_31 = [var_6, var_26]
    var_32 = var_30.validate(var_31)
    var_33 = 1
    var_34 = [var_33]
    var_35 = var_30.validate(var_34)
    var_36 = 1
    var_37 = 2
    var_38 = 3
    var_39 = [var_36, var_37, var_38]
    var_40 = var_30.validate(var_39)
    var_41 = True
    var_42 = module_0.Array(unique_items=var_41)
    var_43 = [var_41, var_37, var_38]
    var_44 = var_42.validate(var_43)
    var_45 = 1
    var_46 = 2
    var_47 = [var_45, var_46, var_46]
    var_48 = var_42.validate(var_47)
    var_49 = module_0.Integer()
    var_50 = module_0.Array(var_49)
    var_51 = '1'
    var_52 = '2'
    var_53 = '3'
    var_54 = [var_51, var_52, var_53]
    var_55 = var_50.validate(var_54)
    var_56 = '1'
    var_57 = 'two'
    var_58 = '3'
    var_59 = [var_56, var_57, var_58]
    var_60 = var_50.validate(var_59)
    var_61 = module_0.Integer()
    var_62 = module_0.Float()
    var_63 = [var_61, var_62]
    var_64 = module_0.Array(var_63)
    var_65 = '2.5'
    var_66 = [var_51, var_65]
    var_67 = var_64.validate(var_66)
    var_68 = '1'
    var_69 = 'two'
    var_70 = [var_68, var_69]
    var_71 = var_64.validate(var_70)
    var_72 = module_0.Integer()
    var_73 = [var_72]
    var_74 = False
    var_75 = module_0.Array(var_73, var_74)
    var_76 = [var_41]
    var_77 = var_75.validate(var_76)
    var_78 = 1
    var_79 = 2
    var_80 = [var_78, var_79]
    var_81 = var_75.validate(var_80)
    var_82 = module_0.Integer()
    var_83 = [var_82]
    var_84 = module_0.Float()
    var_85 = module_0.Array(var_83, var_84)
    var_86 = [var_41, var_65]
    var_87 = var_85.validate(var_86)
    var_88 = 1
    var_89 = 'two'
    var_90 = [var_88, var_89]
    var_91 = var_85.validate(var_90)
    var_92 = module_0.Array(min_items=var_41)
    var_93 = []
    var_94 = var_92.validate(var_93)
    var_95 = module_0.Array(min_items=var_74)
    var_96 = []
    var_97 = var_95.validate(var_96)
    var_98 = module_0.Integer()
    var_99 = module_0.Array(var_98)
    var_100 = [var_41, var_94, var_90]
    var_101 = var_99.serialize(var_100)
    var_102 = var_99.serialize(var_8)
    assert var_102 is None



# Parsed testcases at query #59
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
    var_11 = module_0.Number()
    var_12 = True
    var_13 = var_11.validate(var_12)
    var_14 = 1.5
    var_15 = var_11.validate(var_14)
    var_16 = module_0.Number(coerce_types=var_4)
    var_17 = 'abc'
    var_18 = var_16.validate(var_17)
    var_19 = module_0.Number(coerce_types=var_17)
    var_20 = '123'
    var_21 = var_19.validate(var_20)
    assert var_21 == 123
    var_22 = module_0.Number()
    var_23 = 'inf'
    var_24 = float(var_23)
    var_25 = var_22.validate(var_24)
    var_26 = '0.01'
    var_27 = module_0.Number(precision=var_26)
    var_28 = '1.234'
    var_29 = var_27.validate(var_28)
    var_30 = 5
    var_31 = module_0.Number(minimum=var_30)
    var_32 = var_31.validate(var_30)
    assert var_32 == 5
    var_33 = 4
    var_34 = var_31.validate(var_33)
    var_35 = module_0.Number(exclusive_minimum=var_30)
    var_36 = 6
    var_37 = var_35.validate(var_36)
    assert var_37 == 6
    var_38 = 5
    var_39 = var_35.validate(var_38)
    var_40 = 10
    var_41 = module_0.Number(maximum=var_40)
    var_42 = var_41.validate(var_40)
    assert var_42 == 10
    var_43 = 11
    var_44 = var_41.validate(var_43)
    var_45 = module_0.Number(exclusive_maximum=var_40)
    var_46 = 9
    var_47 = var_45.validate(var_46)
    assert var_47 == 9
    var_48 = 10
    var_49 = var_45.validate(var_48)
    var_50 = 3
    var_51 = module_0.Number(multiple_of=var_50)
    var_52 = var_51.validate(var_36)
    assert var_52 == 6
    var_53 = 7
    var_54 = var_51.validate(var_53)
    var_55 = 0.5
    var_56 = module_0.Number(multiple_of=var_55)
    var_57 = var_56.validate(var_53)
    var_58 = 1.1
    var_59 = var_56.validate(var_58)



# Parsed testcases at query #60
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
    var_5 = module_0.Boolean(coerce_types=var_1)
    var_6 = 'true'
    var_7 = var_5.validate(var_6)
    assert var_7 is True
    var_8 = 'false'
    var_9 = var_5.validate(var_8)
    assert var_9 is False
    var_10 = 'on'
    var_11 = var_5.validate(var_10)
    assert var_11 is True
    var_12 = 'off'
    var_13 = var_5.validate(var_12)
    assert var_13 is False
    var_14 = '1'
    var_15 = var_5.validate(var_14)
    assert var_15 is True
    var_16 = '0'
    var_17 = var_5.validate(var_16)
    assert var_17 is False
    var_18 = ''
    var_19 = var_5.validate(var_18)
    assert var_19 is False
    var_20 = var_5.validate(var_1)
    assert var_20 is True
    var_21 = var_5.validate(var_3)
    assert var_21 is False
    var_22 = module_0.Boolean()
    var_23 = None
    var_24 = var_22.validate(var_23)
    assert var_24 is None
    var_25 = module_0.Boolean(coerce_types=var_1)
    var_26 = 'null'
    var_27 = var_25.validate(var_26)
    assert var_27 is None
    var_28 = 'none'
    var_29 = var_25.validate(var_28)
    assert var_29 is None
    var_30 = module_0.Boolean()
    var_31 = None
    var_32 = var_30.validate(var_31)
    var_33 = 'invalid'
    var_34 = var_30.validate(var_33)
    var_35 = 2
    var_36 = var_30.validate(var_35)
    var_37 = 'yes'
    var_38 = var_30.validate(var_37)
    var_39 = module_0.Boolean(coerce_types=var_3)
    var_40 = 'true'
    var_41 = var_39.validate(var_40)
    var_42 = 1
    var_43 = var_39.validate(var_42)



# Parsed testcases at query #61
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Number()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None
    var_4 = module_0.Number()
    var_5 = None
    var_6 = var_4.validate(var_5)
    var_7 = module_0.Number()
    var_8 = True
    var_9 = var_7.validate(var_8)
    var_10 = 1.5
    var_11 = var_7.validate(var_10)
    var_12 = False
    var_13 = module_0.Number(coerce_types=var_12)
    var_14 = 'abc'
    var_15 = var_13.validate(var_14)
    var_16 = module_0.Number()
    var_17 = 'inf'
    var_18 = float(var_17)
    var_19 = var_16.validate(var_18)
    var_20 = 5
    var_21 = module_0.Number(minimum=var_20)
    var_22 = 3
    var_23 = var_21.validate(var_22)
    var_24 = module_0.Number(exclusive_minimum=var_20)
    var_25 = 5
    var_26 = var_24.validate(var_25)
    var_27 = 10
    var_28 = module_0.Number(maximum=var_27)
    var_29 = 12
    var_30 = var_28.validate(var_29)
    var_31 = module_0.Number(exclusive_maximum=var_27)
    var_32 = 10
    var_33 = var_31.validate(var_32)
    var_34 = 3
    var_35 = module_0.Number(multiple_of=var_34)
    var_36 = 5
    var_37 = var_35.validate(var_36)
    var_38 = 0.5
    var_39 = module_0.Number(multiple_of=var_38)
    var_40 = 1.2
    var_41 = var_39.validate(var_40)
    var_42 = module_0.Number()
    var_43 = var_42.validate(var_20)
    assert var_43 == 5
    var_44 = module_0.Number()
    var_45 = 5.5
    var_46 = var_44.validate(var_45)
    var_47 = module_0.Number()
    var_48 = '5.5'
    var_49 = var_47.validate(var_48)
    var_50 = '0.01'
    var_51 = module_0.Number(precision=var_50)
    var_52 = '5.555'
    var_53 = var_51.validate(var_52)
    var_54 = module_0.Number(coerce_types=var_40)
    var_55 = ''
    var_56 = var_54.validate(var_55)
    assert var_56 is None



# Parsed testcases at query #62
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
    var_21 = module_0.Boolean()
    var_22 = None
    var_23 = var_21.validate(var_22)
    assert var_23 is None
    var_24 = var_21.validate(var_17)
    assert var_24 is None
    var_25 = 'null'
    var_26 = var_21.validate(var_25)
    assert var_26 is None
    var_27 = 'none'
    var_28 = var_21.validate(var_27)
    assert var_28 is None
    var_29 = 'invalid'
    var_30 = var_0.validate(var_29)
    var_31 = 2
    var_32 = var_0.validate(var_31)
    var_33 = None
    var_34 = var_0.validate(var_33)
    var_35 = module_0.Boolean(coerce_types=var_3)
    var_36 = 'true'
    var_37 = var_35.validate(var_36)
    var_38 = 1
    var_39 = var_35.validate(var_38)



# Parsed testcases at query #63
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.Array(var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.validate(var_5)
    var_7 = True
    var_8 = module_0.Array()
    var_9 = None
    var_10 = var_8.validate(var_9)
    assert var_10 is None
    var_11 = False
    var_12 = module_0.Array()
    var_13 = None
    var_14 = var_12.validate(var_13)
    var_15 = module_0.Array()
    var_16 = 'not a list'
    var_17 = var_15.validate(var_16)
    var_18 = module_0.Array(min_items=var_3)
    var_19 = [var_7, var_3]
    var_20 = var_18.validate(var_19)
    var_21 = 1
    var_22 = [var_21]
    var_23 = var_18.validate(var_22)
    var_24 = module_0.Array(max_items=var_23)
    var_25 = [var_7, var_23]
    var_26 = var_24.validate(var_25)
    var_27 = 1
    var_28 = 2
    var_29 = 3
    var_30 = [var_27, var_28, var_29]
    var_31 = var_24.validate(var_30)
    var_32 = module_0.Array(exact_items=var_29)
    var_33 = [var_7, var_29]
    var_34 = var_32.validate(var_33)
    var_35 = 1
    var_36 = [var_35]
    var_37 = var_32.validate(var_36)
    var_38 = 1
    var_39 = 2
    var_40 = 3
    var_41 = [var_38, var_39, var_40]
    var_42 = var_32.validate(var_41)
    var_43 = True
    var_44 = module_0.Array(unique_items=var_43)
    var_45 = [var_43, var_40, var_41]
    var_46 = var_44.validate(var_45)
    var_47 = 1
    var_48 = 2
    var_49 = [var_47, var_48, var_48]
    var_50 = var_44.validate(var_49)
    var_51 = module_0.Integer()
    var_52 = module_0.Array(var_51)
    var_53 = '1'
    var_54 = '2'
    var_55 = '3'
    var_56 = [var_53, var_54, var_55]
    var_57 = var_52.validate(var_56)
    var_58 = '1'
    var_59 = 'two'
    var_60 = '3'
    var_61 = [var_58, var_59, var_60]
    var_62 = var_52.validate(var_61)
    var_63 = module_0.Integer()
    var_64 = module_0.String()
    var_65 = [var_63, var_64]
    var_66 = module_0.Array(var_65)
    var_67 = 'two'
    var_68 = [var_43, var_67]
    var_69 = var_66.validate(var_68)
    var_70 = 1
    var_71 = 2
    var_72 = [var_70, var_71]
    var_73 = var_66.validate(var_72)
    var_74 = module_0.Integer()
    var_75 = [var_74]
    var_76 = module_0.Array(var_75, var_11)
    var_77 = [var_43]
    var_78 = var_76.validate(var_77)
    var_79 = 1
    var_80 = 2
    var_81 = [var_79, var_80]
    var_82 = var_76.validate(var_81)
    var_83 = module_0.Integer()
    var_84 = [var_83]
    var_85 = module_0.String()
    var_86 = module_0.Array(var_84, var_85)
    var_87 = [var_43, var_67]
    var_88 = var_86.validate(var_87)
    var_89 = 1
    var_90 = 2
    var_91 = [var_89, var_90]
    var_92 = var_86.validate(var_91)
    var_93 = module_0.Array(min_items=var_43)
    var_94 = []
    var_95 = var_93.validate(var_94)
    var_96 = module_0.Array(min_items=var_11)
    var_97 = []
    var_98 = var_96.validate(var_97)
    var_99 = module_0.Integer()
    var_100 = module_0.Array(var_99)
    var_101 = module_0.Array(var_100)
    var_102 = [var_43, var_91]
    var_103 = 4
    var_104 = [var_92, var_103]
    var_105 = [var_102, var_104]
    var_106 = var_101.validate(var_105)
    var_107 = 1
    var_108 = 2
    var_109 = [var_107, var_108]
    var_110 = 'three'
    var_111 = 4
    var_112 = [var_110, var_111]
    var_113 = [var_109, var_112]
    var_114 = var_101.validate(var_113)
    var_115 = module_0.Integer()
    var_116 = module_0.Array(var_115)
    var_117 = [var_43, var_109, var_110]
    var_118 = var_116.serialize(var_117)
    var_119 = var_116.serialize(var_114)
    assert var_119 is None
    var_120 = module_0.Integer()
    var_121 = module_0.String()
    var_122 = [var_120, var_121]
    var_123 = module_0.Array(var_122)
    var_124 = [var_43, var_67]
    var_125 = var_123.serialize(var_124)



# Parsed testcases at query #64
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Const(var_0)
    var_2 = None
    var_3 = module_0.Const(var_2)
    var_4 = 'test'
    var_5 = module_0.Const(var_4)
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]
    var_10 = module_0.Const(var_9)
    var_11 = 42
    var_12 = True
    var_13 = module_0.Const(var_11)



# Parsed testcases at query #65
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
    var_21 = module_0.Boolean()
    var_22 = 'null'
    var_23 = var_21.validate(var_22)
    assert var_23 is None
    var_24 = 'none'
    var_25 = var_21.validate(var_24)
    assert var_25 is None
    var_26 = var_21.validate(var_17)
    assert var_26 is None
    var_27 = module_0.Boolean(coerce_types=var_3)
    var_28 = 'true'
    var_29 = var_27.validate(var_28)
    var_30 = 1
    var_31 = var_27.validate(var_30)
    var_32 = 'invalid'
    var_33 = var_0.validate(var_32)
    var_34 = 2
    var_35 = var_0.validate(var_34)
    var_36 = None
    var_37 = var_0.validate(var_36)
    var_38 = None
    var_39 = var_21.validate(var_38)
    assert var_39 is None



# Parsed testcases at query #66
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'Option A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'Option B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = module_0.Choice(choices=var_6)
    var_8 = var_7.validate(var_0)
    assert var_8 == 'a'
    var_9 = var_7.validate(var_3)
    assert var_9 == 'b'
    var_10 = (var_0, var_1)
    var_11 = (var_3, var_4)
    var_12 = [var_10, var_11]
    var_13 = module_0.Choice(choices=var_12)
    var_14 = var_13.validate(var_0)
    assert var_14 == 'a'
    var_15 = (var_0, var_1)
    var_16 = (var_3, var_4)
    var_17 = [var_15, var_16]
    var_18 = module_0.Choice(choices=var_17)
    var_19 = 'c'
    var_20 = var_18.validate(var_19)
    var_21 = (var_19, var_20)
    var_22 = (var_3, var_4)
    var_23 = [var_21, var_22]
    var_24 = True
    var_25 = module_0.Choice(choices=var_23)
    var_26 = None
    var_27 = var_25.validate(var_26)
    assert var_27 is None
    var_28 = (var_19, var_20)
    var_29 = (var_3, var_4)
    var_30 = [var_28, var_29]
    var_31 = False
    var_32 = module_0.Choice(choices=var_30)
    var_33 = None
    var_34 = var_32.validate(var_33)
    var_35 = (var_33, var_34)
    var_36 = (var_3, var_4)
    var_37 = [var_35, var_36]
    var_38 = module_0.Choice(choices=var_37, coerce_types=var_24)
    var_39 = ''
    var_40 = var_38.validate(var_39)
    assert var_40 is None
    var_41 = (var_33, var_34)
    var_42 = (var_3, var_4)
    var_43 = [var_41, var_42]
    var_44 = module_0.Choice(choices=var_43, coerce_types=var_24)
    var_45 = ''
    var_46 = var_44.validate(var_45)
    var_47 = (var_45, var_46)
    var_48 = (var_3, var_4)
    var_49 = [var_47, var_48]
    var_50 = module_0.Choice(choices=var_49, coerce_types=var_31)
    var_51 = ''
    var_52 = var_50.validate(var_51)



# Parsed testcases at query #67
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Object(properties=var_4)
    var_6 = 'John'
    var_7 = 30
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = True
    var_11 = module_0.Object()
    var_12 = None
    var_13 = var_11.validate(var_12)
    assert var_13 is None
    var_14 = module_0.Object()
    var_15 = None
    var_16 = var_14.validate(var_15)
    var_17 = 'not an object'
    var_18 = var_14.validate(var_17)
    var_19 = 123
    var_20 = 'value'
    var_21 = {var_19: var_20}
    var_22 = var_14.validate(var_21)
    var_23 = module_0.String()
    var_24 = {var_19: var_23}
    var_25 = [var_19]
    var_26 = module_0.Object(properties=var_24, required=var_25)
    var_27 = {}
    var_28 = var_26.validate(var_27)
    var_29 = module_0.Integer()
    var_30 = {var_28: var_29}
    var_31 = module_0.Object(properties=var_30)
    var_32 = 'age'
    var_33 = 'not a number'
    var_34 = {var_32: var_33}
    var_35 = var_31.validate(var_34)
    var_36 = 2
    var_37 = module_0.Object(min_properties=var_36)
    var_38 = 'a'
    var_39 = 1
    var_40 = {var_38: var_39}
    var_41 = var_37.validate(var_40)
    var_42 = module_0.Object(max_properties=var_36)
    var_43 = 'a'
    var_44 = 'b'
    var_45 = 'c'
    var_46 = 1
    var_47 = 2
    var_48 = 3
    var_49 = {var_43: var_46, var_44: var_47, var_45: var_48}
    var_50 = var_42.validate(var_49)
    var_51 = module_0.String()
    var_52 = {var_43: var_51}
    var_53 = False
    var_54 = module_0.Object(properties=var_52, additional_properties=var_53)
    var_55 = 'name'
    var_56 = 'extra'
    var_57 = 'John'
    var_58 = 'field'
    var_59 = {var_55: var_57, var_56: var_58}
    var_60 = var_54.validate(var_59)
    var_61 = module_0.String()
    var_62 = {var_55: var_61}
    var_63 = module_0.Integer()
    var_64 = module_0.Object(properties=var_62, additional_properties=var_63)
    var_65 = {var_55: var_60, var_56: var_49}
    var_66 = var_64.validate(var_65)
    var_67 = 'name'
    var_68 = 'age'
    var_69 = 'John'
    var_70 = 'not a number'
    var_71 = {var_67: var_69, var_68: var_70}
    var_72 = var_64.validate(var_71)
    var_73 = '^[a-z]+$'
    var_74 = module_0.String(pattern=var_73)
    var_75 = module_0.Object(property_names=var_74)
    var_76 = 'Name'
    var_77 = 'value'
    var_78 = {var_76: var_77}
    var_79 = var_75.validate(var_78)
    var_80 = module_0.String()
    var_81 = {var_76: var_80}
    var_82 = '^num_'
    var_83 = module_0.Integer()
    var_84 = {var_82: var_83}
    var_85 = module_0.Object(properties=var_81, pattern_properties=var_84)
    var_86 = 'num_age'
    var_87 = {var_76: var_72, var_86: var_49}
    var_88 = var_85.validate(var_87)
    var_89 = 'name'
    var_90 = 'num_age'
    var_91 = 'John'
    var_92 = 'not a number'
    var_93 = {var_89: var_91, var_90: var_92}
    var_94 = var_85.validate(var_93)
    var_95 = 'default'
    var_96 = module_0.String()
    var_97 = {var_89: var_96}
    var_98 = module_0.Object(properties=var_97)
    var_99 = {}
    var_100 = var_98.validate(var_99)



# Parsed testcases at query #68
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
    var_8 = module_0.Boolean()
    var_9 = [var_0, var_1, var_8]
    var_10 = module_0.Union(var_9)
    var_11 = True
    var_12 = var_10.validate(var_11)
    assert var_12 is True
    var_13 = module_0.String()
    var_14 = module_0.Integer()
    var_15 = [var_13, var_14]
    var_16 = module_0.Union(var_15)
    var_17 = None
    var_18 = var_16.validate(var_17)
    assert var_18 is None
    var_19 = module_0.String()
    var_20 = module_0.Integer()
    var_21 = [var_19, var_20]
    var_22 = module_0.Union(var_21)
    var_23 = None
    var_24 = var_22.validate(var_23)
    var_25 = []
    var_26 = var_22.validate(var_25)
    var_27 = 5
    var_28 = module_0.String(min_length=var_27)
    var_29 = module_0.Integer()
    var_30 = [var_28, var_29]
    var_31 = module_0.Union(var_30)
    var_32 = 'short'
    var_33 = var_31.validate(var_32)
    var_34 = module_0.String(min_length=var_27)
    var_35 = 10
    var_36 = module_0.Integer(minimum=var_35)
    var_37 = [var_34, var_36]
    var_38 = module_0.Union(var_37)
    var_39 = 'short'
    var_40 = var_38.validate(var_39)
    var_41 = module_0.String()
    var_42 = module_0.Integer()
    var_43 = [var_41, var_42]
    var_44 = module_0.Union(var_43)
    var_45 = []
    var_46 = var_44.validate(var_45)



# Parsed testcases at query #69
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'Option A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'Option B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = module_0.Choice(choices=var_6)
    var_8 = var_7.validate(var_0)
    assert var_8 == 'a'
    var_9 = var_7.validate(var_3)
    assert var_9 == 'b'
    var_10 = (var_0, var_1)
    var_11 = (var_3, var_4)
    var_12 = [var_10, var_11]
    var_13 = module_0.Choice(choices=var_12)
    var_14 = var_13.validate(var_0)
    assert var_14 == 'a'
    var_15 = (var_0, var_1)
    var_16 = (var_3, var_4)
    var_17 = [var_15, var_16]
    var_18 = module_0.Choice(choices=var_17)
    var_19 = 'c'
    var_20 = var_18.validate(var_19)
    var_21 = (var_19, var_20)
    var_22 = (var_3, var_4)
    var_23 = [var_21, var_22]
    var_24 = True
    var_25 = module_0.Choice(choices=var_23)
    var_26 = None
    var_27 = var_25.validate(var_26)
    assert var_27 is None
    var_28 = (var_19, var_20)
    var_29 = (var_3, var_4)
    var_30 = [var_28, var_29]
    var_31 = False
    var_32 = module_0.Choice(choices=var_30)
    var_33 = None
    var_34 = var_32.validate(var_33)
    var_35 = (var_33, var_34)
    var_36 = (var_3, var_4)
    var_37 = [var_35, var_36]
    var_38 = module_0.Choice(choices=var_37, coerce_types=var_24)
    var_39 = ''
    var_40 = var_38.validate(var_39)
    assert var_40 is None
    var_41 = (var_33, var_34)
    var_42 = (var_3, var_4)
    var_43 = [var_41, var_42]
    var_44 = module_0.Choice(choices=var_43, coerce_types=var_24)
    var_45 = ''
    var_46 = var_44.validate(var_45)
    var_47 = (var_45, var_46)
    var_48 = (var_3, var_4)
    var_49 = [var_47, var_48]
    var_50 = module_0.Choice(choices=var_49, coerce_types=var_31)
    var_51 = ''
    var_52 = var_50.validate(var_51)
    var_53 = (var_51, var_52)
    var_54 = (var_3, var_4)
    var_55 = [var_53, var_54]
    var_56 = module_0.Choice(choices=var_55, coerce_types=var_24)
    var_57 = var_56.validate(var_51)
    assert var_57 == 'a'
    var_58 = (var_51, var_52)
    var_59 = (var_3, var_4)
    var_60 = [var_58, var_59]
    var_61 = module_0.Choice(choices=var_60, coerce_types=var_24)
    var_62 = 'c'
    var_63 = var_61.validate(var_62)
    var_64 = (var_62, var_63)
    var_65 = (var_3, var_4)
    var_66 = [var_64, var_65]
    var_67 = module_0.Choice(choices=var_66, coerce_types=var_31)
    var_68 = var_67.validate(var_62)
    assert var_68 == 'a'
    var_69 = (var_62, var_63)
    var_70 = (var_3, var_4)
    var_71 = [var_69, var_70]
    var_72 = module_0.Choice(choices=var_71, coerce_types=var_31)
    var_73 = 'c'
    var_74 = var_72.validate(var_73)



# Parsed testcases at query #70
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
    var_14 = 'not an array'
    var_15 = var_11.validate(var_14)
    var_16 = module_0.Array(min_items=var_15)
    var_17 = [var_6, var_15]
    var_18 = var_16.validate(var_17)
    var_19 = 1
    var_20 = [var_19]
    var_21 = var_16.validate(var_20)
    var_22 = module_0.Array(max_items=var_20)
    var_23 = [var_6, var_20]
    var_24 = var_22.validate(var_23)
    var_25 = 1
    var_26 = 2
    var_27 = 3
    var_28 = [var_25, var_26, var_27]
    var_29 = var_22.validate(var_28)
    var_30 = module_0.Array(exact_items=var_26)
    var_31 = [var_6, var_26]
    var_32 = var_30.validate(var_31)
    var_33 = 1
    var_34 = [var_33]
    var_35 = var_30.validate(var_34)
    var_36 = 1
    var_37 = 2
    var_38 = 3
    var_39 = [var_36, var_37, var_38]
    var_40 = var_30.validate(var_39)
    var_41 = True
    var_42 = module_0.Array(unique_items=var_41)
    var_43 = [var_41, var_37, var_38]
    var_44 = var_42.validate(var_43)
    var_45 = 1
    var_46 = 2
    var_47 = [var_45, var_46, var_46]
    var_48 = var_42.validate(var_47)
    var_49 = module_0.Integer()
    var_50 = module_0.Array(var_49)
    var_51 = [var_41, var_46, var_47]
    var_52 = var_50.validate(var_51)
    var_53 = 1
    var_54 = 'not an int'
    var_55 = 3
    var_56 = [var_53, var_54, var_55]
    var_57 = var_50.validate(var_56)
    var_58 = module_0.Integer()
    var_59 = module_0.Integer()
    var_60 = [var_58, var_59]
    var_61 = module_0.Array(var_60, var_10)
    var_62 = [var_41, var_54]
    var_63 = var_61.validate(var_62)
    var_64 = 1
    var_65 = 2
    var_66 = 3
    var_67 = [var_64, var_65, var_66]
    var_68 = var_61.validate(var_67)
    var_69 = module_0.String()
    var_70 = module_0.Integer()
    var_71 = module_0.Integer()
    var_72 = [var_70, var_71]
    var_73 = module_0.Array(var_72, var_69)
    var_74 = 'three'
    var_75 = [var_41, var_65, var_74]
    var_76 = var_73.validate(var_75)
    var_77 = 1
    var_78 = 2
    var_79 = 3
    var_80 = [var_77, var_78, var_79]
    var_81 = var_73.validate(var_80)
    var_82 = module_0.Array(min_items=var_41)
    var_83 = []
    var_84 = var_82.validate(var_83)
    var_85 = module_0.Array(min_items=var_10)
    var_86 = []
    var_87 = var_85.validate(var_86)
    var_88 = module_0.Integer()
    var_89 = module_0.Array(var_88)
    var_90 = module_0.Array(var_89)
    var_91 = [var_41, var_84]
    var_92 = 4
    var_93 = [var_79, var_92]
    var_94 = [var_91, var_93]
    var_95 = var_90.validate(var_94)
    var_96 = 1
    var_97 = 'not an int'
    var_98 = [var_96, var_97]
    var_99 = 3
    var_100 = 4
    var_101 = [var_99, var_100]
    var_102 = [var_98, var_101]
    var_103 = var_90.validate(var_102)
    var_104 = module_0.Integer()
    var_105 = module_0.Array(var_104)
    var_106 = [var_41, var_97, var_98]
    var_107 = var_105.serialize(var_106)
    var_108 = var_105.serialize(var_102)
    assert var_108 is None



