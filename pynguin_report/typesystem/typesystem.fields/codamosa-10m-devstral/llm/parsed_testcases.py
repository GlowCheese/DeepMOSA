####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.Float()
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = module_0.Union(var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None
    var_7 = module_0.Integer()
    var_8 = module_0.Float()
    var_9 = [var_7, var_8]
    var_10 = False
    var_11 = module_0.Union(var_9)
    var_12 = None
    var_13 = var_11.validate(var_12)
    var_14 = exc_info.value.messages()[var_10]
    var_15 = var_14.code
    assert var_15 == 'null'
    var_16 = module_0.Integer()
    var_17 = module_0.Float()
    var_18 = [var_16, var_17]
    var_19 = module_0.Union(var_18)
    var_20 = 10
    var_21 = var_19.validate(var_20)
    assert var_21 == 10
    var_22 = module_0.Integer()
    var_23 = module_0.Float()
    var_24 = [var_22, var_23]
    var_25 = module_0.Union(var_24)
    var_26 = 10.5
    var_27 = var_25.validate(var_26)
    var_28 = module_0.Integer()
    var_29 = module_0.Float()
    var_30 = [var_28, var_29]
    var_31 = module_0.Union(var_30)
    var_32 = 'invalid'
    var_33 = var_31.validate(var_32)
    var_34 = exc_info.value.messages()[var_10]
    var_35 = var_34.code
    assert var_35 == 'union'
    var_36 = module_0.Integer(minimum=var_10)
    var_37 = module_0.Float()
    var_38 = [var_36, var_37]
    var_39 = module_0.Union(var_38)
    var_40 = -1
    var_41 = var_39.validate(var_40)
    var_42 = exc_info.value.messages()[var_10]
    var_43 = var_42.code
    assert var_43 == 'minimum'
    var_44 = module_0.Integer(minimum=var_10)
    var_45 = module_0.Float(minimum=var_10)
    var_46 = [var_44, var_45]
    var_47 = module_0.Union(var_46)
    var_48 = -1
    var_49 = var_47.validate(var_48)
    var_50 = exc_info.value.messages()[var_10]
    var_51 = var_50.code
    assert var_51 == 'union'



# Parsed testcases at query #2
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
    var_64 = [var_62, var_63]
    var_65 = module_0.Array(var_64)
    var_66 = 'two'
    var_67 = [var_52, var_66]
    var_68 = var_65.validate(var_67)
    var_69 = '1'
    var_70 = [var_69]
    var_71 = var_65.validate(var_70)
    var_72 = '1'
    var_73 = 'two'
    var_74 = 'three'
    var_75 = [var_72, var_73, var_74]
    var_76 = var_65.validate(var_75)
    var_77 = module_0.Integer()
    var_78 = [var_77]
    var_79 = module_0.String()
    var_80 = module_0.Array(var_78, var_79)
    var_81 = 'three'
    var_82 = [var_52, var_66, var_81]
    var_83 = var_80.validate(var_82)
    var_84 = '1'
    var_85 = 2
    var_86 = [var_84, var_85]
    var_87 = var_80.validate(var_86)
    var_88 = module_0.Integer()
    var_89 = module_0.String()
    var_90 = [var_88, var_89]
    var_91 = module_0.Array(var_90, var_10)
    var_92 = [var_52, var_66]
    var_93 = var_91.validate(var_92)
    var_94 = '1'
    var_95 = 'two'
    var_96 = 'three'
    var_97 = [var_94, var_95, var_96]
    var_98 = var_91.validate(var_97)
    var_99 = module_0.Array(min_items=var_42)
    var_100 = []
    var_101 = var_99.validate(var_100)
    var_102 = module_0.Array()
    var_103 = []
    var_104 = var_102.validate(var_103)



# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Choice()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = module_0.Choice(choices=var_4)
    var_6 = 'Option A'
    var_7 = (var_1, var_6)
    var_8 = 'Option B'
    var_9 = (var_2, var_8)
    var_10 = [var_7, var_9]
    var_11 = module_0.Choice(choices=var_10)
    var_12 = False
    var_13 = module_0.Choice(coerce_types=var_12)
    var_14 = 'Test Choice'
    var_15 = 'A test choice field'
    var_16 = True
    var_17 = module_0.Choice()
    var_18 = 'invalid_tuple'
    var_19 = [var_18]
    var_20 = module_0.Choice(choices=var_19)



# Parsed testcases at query #4
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None
    var_4 = module_0.String(allow_blank=var_0, coerce_types=var_0)
    var_5 = var_4.validate(var_2)
    assert var_5 == ''
    var_6 = False
    var_7 = module_0.String(allow_blank=var_0, coerce_types=var_6)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = module_0.String()
    var_11 = 'a\x00b'
    var_12 = var_10.validate(var_11)
    assert var_12 == 'ab'
    var_13 = module_0.String()
    var_14 = '  abc  '
    var_15 = var_13.validate(var_14)
    assert var_15 == 'abc'
    var_16 = module_0.String(allow_blank=var_8)
    var_17 = ''
    var_18 = var_16.validate(var_17)
    assert var_18 == ''
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
    var_40 = module_0.String()
    var_41 = 123
    var_42 = var_40.validate(var_41)
    var_43 = module_0.String()
    var_44 = None
    var_45 = var_43.validate(var_44)
    var_46 = module_0.String()
    var_47 = ''
    var_48 = var_46.validate(var_47)
    var_49 = module_0.String(allow_blank=var_47, coerce_types=var_47)
    var_50 = var_49.validate(var_17)
    assert var_50 is None



# Parsed testcases at query #5
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
    var_16 = module_0.Field()
    var_17 = module_0.Field()
    var_18 = module_0.Field()
    var_19 = [var_17, var_18]
    var_20 = True
    var_21 = module_0.Array(var_19, var_16, var_14, var_10, unique_items=var_20)



# Parsed testcases at query #6
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Array()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    var_3 = module_0.Array()
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = var_3.serialize(var_7)
    var_9 = module_0.Integer()
    var_10 = module_0.Array(var_9)
    var_11 = '1'
    var_12 = '2'
    var_13 = '3'
    var_14 = [var_11, var_12, var_13]
    var_15 = var_10.serialize(var_14)
    var_16 = module_0.Integer()
    var_17 = module_0.Float()
    var_18 = module_0.Decimal()
    var_19 = [var_16, var_17, var_18]
    var_20 = module_0.Array(var_19)
    var_21 = '2.5'
    var_22 = '3.7'
    var_23 = [var_11, var_21, var_22]
    var_24 = var_20.serialize(var_23)
    var_25 = module_0.Integer()
    var_26 = module_0.Float()
    var_27 = [var_25, var_26]
    var_28 = True
    var_29 = module_0.Array(var_27, var_28)
    var_30 = 'extra'
    var_31 = [var_11, var_21, var_30]
    var_32 = var_29.serialize(var_31)
    var_33 = module_0.Integer()
    var_34 = module_0.Float()
    var_35 = [var_33, var_34]
    var_36 = module_0.Decimal()
    var_37 = module_0.Array(var_35, var_36)
    var_38 = [var_11, var_21, var_22]
    var_39 = var_37.serialize(var_38)



# Parsed testcases at query #7
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



# Parsed testcases at query #8
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
    var_11 = []
    var_12 = module_0.Choice(choices=var_11)
    var_13 = [var_0, var_1]
    var_14 = False
    var_15 = module_0.Choice(choices=var_13, coerce_types=var_14)
    var_16 = [var_0, var_1]
    var_17 = True
    var_18 = module_0.Choice(choices=var_16)
    var_19 = [var_0, var_1]
    var_20 = module_0.Choice(choices=var_19)
    var_21 = [var_0, var_1]
    var_22 = 'Test Choice'
    var_23 = 'A test choice field'
    var_24 = module_0.Choice(choices=var_21)
    var_25 = [var_0, var_1]
    var_26 = module_0.Choice(choices=var_25)
    var_27 = (var_1, var_7)
    var_28 = [var_0, var_27, var_2]
    var_29 = module_0.Choice(choices=var_28)



# Parsed testcases at query #9
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



# Parsed testcases at query #10
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
    var_14 = False
    var_15 = module_0.Object()
    var_16 = None
    var_17 = var_15.validate(var_16)
    var_18 = module_0.Object()
    var_19 = 'not a dict'
    var_20 = var_18.validate(var_19)
    var_21 = module_0.Object()
    var_22 = 123
    var_23 = 'value'
    var_24 = {var_22: var_23}
    var_25 = var_21.validate(var_24)
    var_26 = '^[a-z]+$'
    var_27 = module_0.String(pattern=var_26)
    var_28 = module_0.Object(property_names=var_27)
    var_29 = '123'
    var_30 = 'value'
    var_31 = {var_29: var_30}
    var_32 = var_28.validate(var_31)
    var_33 = module_0.Object(min_properties=var_10)
    var_34 = {}
    var_35 = var_33.validate(var_34)
    var_36 = 2
    var_37 = module_0.Object(max_properties=var_36)
    var_38 = 'a'
    var_39 = 'b'
    var_40 = 'c'
    var_41 = 1
    var_42 = 2
    var_43 = 3
    var_44 = {var_38: var_41, var_39: var_42, var_40: var_43}
    var_45 = var_37.validate(var_44)
    var_46 = module_0.String()
    var_47 = {var_38: var_46}
    var_48 = [var_38]
    var_49 = module_0.Object(properties=var_47, required=var_48)
    var_50 = 'age'
    var_51 = 30
    var_52 = {var_50: var_51}
    var_53 = var_49.validate(var_52)
    var_54 = 'default'
    var_55 = module_0.String()
    var_56 = {var_50: var_55}
    var_57 = module_0.Object(properties=var_56)
    var_58 = {}
    var_59 = var_57.validate(var_58)
    var_60 = '^test_'
    var_61 = module_0.String()
    var_62 = {var_60: var_61}
    var_63 = module_0.Object(pattern_properties=var_62)
    var_64 = 'test_name'
    var_65 = 'value'
    var_66 = {var_64: var_65}
    var_67 = var_63.validate(var_66)
    var_68 = module_0.String()
    var_69 = {var_50: var_68}
    var_70 = module_0.Object(properties=var_69, additional_properties=var_14)
    var_71 = 'name'
    var_72 = 'age'
    var_73 = 'John'
    var_74 = 30
    var_75 = {var_71: var_73, var_72: var_74}
    var_76 = var_70.validate(var_75)
    var_77 = module_0.String()
    var_78 = {var_71: var_77}
    var_79 = module_0.Integer()
    var_80 = module_0.Object(properties=var_78, additional_properties=var_79)
    var_81 = {var_71: var_76, var_72: var_44}
    var_82 = var_80.validate(var_81)
    var_83 = 'address'
    var_84 = 'city'
    var_85 = module_0.String()
    var_86 = {var_84: var_85}
    var_87 = module_0.Object(properties=var_86)
    var_88 = {var_83: var_87}
    var_89 = module_0.Object(properties=var_88)
    var_90 = 'NYC'
    var_91 = {var_84: var_90}
    var_92 = {var_83: var_91}
    var_93 = var_89.validate(var_92)



# Parsed testcases at query #11
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
    var_19 = None
    var_20 = var_18.validate(var_19)
    var_21 = module_0.Integer()
    var_22 = module_0.String()
    var_23 = [var_21, var_22]
    var_24 = module_0.Union(var_23)
    var_25 = 1
    var_26 = 2
    var_27 = 3
    var_28 = [var_25, var_26, var_27]
    var_29 = var_24.validate(var_28)
    var_30 = 0
    var_31 = module_0.Integer(minimum=var_30)
    var_32 = 3
    var_33 = module_0.String(min_length=var_32)
    var_34 = [var_31, var_33]
    var_35 = module_0.Union(var_34)
    var_36 = -1
    var_37 = var_35.validate(var_36)
    var_38 = module_0.Integer(minimum=var_30)
    var_39 = module_0.String(min_length=var_32)
    var_40 = [var_38, var_39]
    var_41 = module_0.Union(var_40)
    var_42 = 'ab'
    var_43 = var_41.validate(var_42)



# Parsed testcases at query #12
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
    var_12 = [var_0]
    var_13 = module_0.Object(properties=var_11, required=var_12)
    var_14 = 'age'
    var_15 = 30
    var_16 = {var_14: var_15}
    var_17 = var_13.validate(var_16)
    var_18 = module_0.String()
    var_19 = {var_14: var_18}
    var_20 = True
    var_21 = module_0.Object(properties=var_19)
    var_22 = None
    var_23 = var_21.validate(var_22)
    assert var_23 is None
    var_24 = module_0.String()
    var_25 = {var_14: var_24}
    var_26 = module_0.Object(properties=var_25)
    var_27 = 'not a dict'
    var_28 = var_26.validate(var_27)
    var_29 = module_0.String()
    var_30 = {var_27: var_29}
    var_31 = module_0.Object(properties=var_30)
    var_32 = 123
    var_33 = 'value'
    var_34 = {var_32: var_33}
    var_35 = var_31.validate(var_34)
    var_36 = module_0.String()
    var_37 = module_0.Integer()
    var_38 = {var_32: var_36, var_33: var_37}
    var_39 = 2
    var_40 = module_0.Object(properties=var_38, min_properties=var_39)
    var_41 = 'name'
    var_42 = 'John'
    var_43 = {var_41: var_42}
    var_44 = var_40.validate(var_43)
    var_45 = module_0.String()
    var_46 = module_0.Integer()
    var_47 = {var_41: var_45, var_42: var_46}
    var_48 = module_0.Object(properties=var_47, max_properties=var_20)
    var_49 = 'name'
    var_50 = 'age'
    var_51 = 'John'
    var_52 = 30
    var_53 = {var_49: var_51, var_50: var_52}
    var_54 = var_48.validate(var_53)
    var_55 = module_0.String()
    var_56 = {var_49: var_55}
    var_57 = 3
    var_58 = module_0.String(min_length=var_57)
    var_59 = module_0.Object(properties=var_56, property_names=var_58)
    var_60 = 'na'
    var_61 = 'value'
    var_62 = {var_60: var_61}
    var_63 = var_59.validate(var_62)
    var_64 = module_0.String()
    var_65 = {var_60: var_64}
    var_66 = False
    var_67 = module_0.Object(properties=var_65, additional_properties=var_66)
    var_68 = 'name'
    var_69 = 'age'
    var_70 = 'John'
    var_71 = 30
    var_72 = {var_68: var_70, var_69: var_71}
    var_73 = var_67.validate(var_72)
    var_74 = module_0.String()
    var_75 = {var_68: var_74}
    var_76 = module_0.Integer()
    var_77 = module_0.Object(properties=var_75, additional_properties=var_76)
    var_78 = {var_68: var_73, var_69: var_7}
    var_79 = var_77.validate(var_78)
    var_80 = '^pref_'
    var_81 = module_0.String()
    var_82 = {var_80: var_81}
    var_83 = module_0.Object(pattern_properties=var_82)
    var_84 = 'pref_name'
    var_85 = {var_84: var_73}
    var_86 = var_83.validate(var_85)
    var_87 = 'address'
    var_88 = 'city'
    var_89 = module_0.String()
    var_90 = {var_88: var_89}
    var_91 = module_0.Object(properties=var_90)
    var_92 = {var_87: var_91}
    var_93 = module_0.Object(properties=var_92)
    var_94 = 'NYC'
    var_95 = {var_88: var_94}
    var_96 = {var_87: var_95}
    var_97 = var_93.validate(var_96)



# Parsed testcases at query #13
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



# Parsed testcases at query #14
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Field(default=var_0)
    var_2 = var_1.get_default_value()
    assert var_2 == 42
    var_3 = 'test'
    var_4 = lambda : var_3
    var_5 = module_0.Field(default=var_4)
    var_6 = var_5.get_default_value()
    assert var_6 == 'test'
    var_7 = module_0.Field()
    var_8 = var_7.get_default_value()
    assert var_8 is None
    var_9 = True
    var_10 = module_0.Field(allow_null=var_9)
    var_11 = var_10.get_default_value()
    assert var_11 is None
    var_12 = 'nullable'
    var_13 = module_0.Field(default=var_12, allow_null=var_9)
    var_14 = var_13.get_default_value()
    assert var_14 == 'nullable'



# Parsed testcases at query #15
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



# Parsed testcases at query #16
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
    var_12 = '^[0-9]+$'
    var_13 = module_1.compile(var_12)
    var_14 = module_0.String(pattern=var_13)
    var_15 = module_0.String(allow_blank=var_4)
    var_16 = module_0.String()
    var_17 = 'invalid'
    var_18 = module_0.String(max_length=var_17)
    var_19 = 'invalid'
    var_20 = module_0.String(min_length=var_19)
    var_21 = 123
    var_22 = module_0.String(pattern=var_21)
    var_23 = 123
    var_24 = module_0.String(format=var_23)



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
    var_42 = module_0.Integer()
    var_43 = module_0.Array(var_42)
    var_44 = '1'
    var_45 = '2'
    var_46 = '3'
    var_47 = [var_44, var_45, var_46]
    var_48 = var_43.validate(var_47)
    var_49 = '1'
    var_50 = 'two'
    var_51 = '3'
    var_52 = [var_49, var_50, var_51]
    var_53 = var_43.validate(var_52)
    var_54 = module_0.Integer()
    var_55 = module_0.String()
    var_56 = [var_54, var_55]
    var_57 = module_0.Array(var_56)
    var_58 = 'two'
    var_59 = [var_44, var_58]
    var_60 = var_57.validate(var_59)
    var_61 = '1'
    var_62 = 2
    var_63 = [var_61, var_62]
    var_64 = var_57.validate(var_63)
    var_65 = module_0.Integer()
    var_66 = [var_65]
    var_67 = module_0.Array(var_66, var_10)
    var_68 = [var_6]
    var_69 = var_67.validate(var_68)
    var_70 = 1
    var_71 = 2
    var_72 = [var_70, var_71]
    var_73 = var_67.validate(var_72)
    var_74 = module_0.Integer()
    var_75 = [var_74]
    var_76 = module_0.String()
    var_77 = module_0.Array(var_75, var_76)
    var_78 = [var_6, var_58]
    var_79 = var_77.validate(var_78)
    var_80 = 1
    var_81 = 2
    var_82 = [var_80, var_81]
    var_83 = var_77.validate(var_82)
    var_84 = True
    var_85 = module_0.Array(unique_items=var_84)
    var_86 = [var_84, var_81, var_82]
    var_87 = var_85.validate(var_86)
    var_88 = 1
    var_89 = 2
    var_90 = [var_88, var_89, var_88]
    var_91 = var_85.validate(var_90)
    var_92 = module_0.Array(min_items=var_84)
    var_93 = []
    var_94 = var_92.validate(var_93)
    var_95 = module_0.Array(min_items=var_10)
    var_96 = []
    var_97 = var_95.validate(var_96)



# Parsed testcases at query #18
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
    var_10 = 'not_a_field'
    var_11 = [var_0, var_10]
    var_12 = module_0.Union(var_11)
    var_13 = [var_0, var_1]
    var_14 = module_0.Union(var_13)



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



# Parsed testcases at query #20
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



# Parsed testcases at query #21
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
    var_14 = 'not a list'
    var_15 = var_0.validate(var_14)
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
    var_49 = 'not an int'
    var_50 = 3
    var_51 = [var_48, var_49, var_50]
    var_52 = var_45.validate(var_51)
    var_53 = module_0.Integer()
    var_54 = module_0.String()
    var_55 = [var_53, var_54]
    var_56 = module_0.Array(var_55)
    var_57 = 'test'
    var_58 = [var_6, var_57]
    var_59 = var_56.validate(var_58)
    var_60 = 1
    var_61 = 2
    var_62 = [var_60, var_61]
    var_63 = var_56.validate(var_62)
    var_64 = module_0.Integer()
    var_65 = [var_64]
    var_66 = module_0.Array(var_65, var_10)
    var_67 = [var_6]
    var_68 = var_66.validate(var_67)
    var_69 = 1
    var_70 = 2
    var_71 = [var_69, var_70]
    var_72 = var_66.validate(var_71)
    var_73 = module_0.Integer()
    var_74 = [var_73]
    var_75 = module_0.String()
    var_76 = module_0.Array(var_74, var_75)
    var_77 = [var_6, var_57]
    var_78 = var_76.validate(var_77)
    var_79 = 1
    var_80 = 2
    var_81 = [var_79, var_80]
    var_82 = var_76.validate(var_81)
    var_83 = True
    var_84 = module_0.Array(unique_items=var_83)
    var_85 = [var_83, var_80, var_81]
    var_86 = var_84.validate(var_85)
    var_87 = 1
    var_88 = 2
    var_89 = [var_87, var_88, var_88]
    var_90 = var_84.validate(var_89)
    var_91 = [var_83, var_88, var_89]
    var_92 = var_0.serialize(var_91)
    var_93 = [var_83, var_88, var_89]
    var_94 = var_45.serialize(var_93)
    var_95 = [var_83, var_57]
    var_96 = var_56.serialize(var_95)



# Parsed testcases at query #22
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
    var_13 = []
    var_14 = module_0.Union(var_13)
    var_15 = [var_0, var_1]
    var_16 = module_0.Union(var_15)



# Parsed testcases at query #23
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
    var_22 = module_0.Boolean(coerce_types=var_1)
    var_23 = 'null'
    var_24 = var_22.validate(var_23)
    assert var_24 is None
    var_25 = 'none'
    var_26 = var_22.validate(var_25)
    assert var_26 is None
    var_27 = var_22.validate(var_18)
    assert var_27 is None
    var_28 = module_0.Boolean(coerce_types=var_3)
    var_29 = 'true'
    var_30 = var_28.validate(var_29)
    var_31 = 1
    var_32 = var_28.validate(var_31)
    var_33 = '1'
    var_34 = var_28.validate(var_33)
    var_35 = module_0.Boolean()
    var_36 = None
    var_37 = var_35.validate(var_36)
    var_38 = module_0.Boolean()
    var_39 = None
    var_40 = var_38.validate(var_39)
    assert var_40 is None
    var_41 = module_0.Boolean(coerce_types=var_36)
    var_42 = 'invalid'
    var_43 = var_41.validate(var_42)



# Parsed testcases at query #24
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Decimal()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    var_3 = '123.456'
    var_4 = '-123.456'
    var_5 = '0'
    var_6 = '0.0001'
    var_7 = '999999999.999999999'



# Parsed testcases at query #25
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = module_0.Const(var_0)
    var_2 = None
    var_3 = module_0.Const(var_2)
    var_4 = 'test_value'
    var_5 = True
    var_6 = module_0.Const(var_4)



# Parsed testcases at query #26
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
    var_42 = module_0.Integer()
    var_43 = module_0.Array(var_42)
    var_44 = '1'
    var_45 = '2'
    var_46 = '3'
    var_47 = [var_44, var_45, var_46]
    var_48 = var_43.validate(var_47)
    var_49 = '1'
    var_50 = 'two'
    var_51 = '3'
    var_52 = [var_49, var_50, var_51]
    var_53 = var_43.validate(var_52)
    var_54 = module_0.Integer()
    var_55 = module_0.String()
    var_56 = module_0.Boolean()
    var_57 = [var_54, var_55, var_56]
    var_58 = module_0.Array(var_57)
    var_59 = 'two'
    var_60 = 'true'
    var_61 = [var_44, var_59, var_60]
    var_62 = var_58.validate(var_61)
    var_63 = '1'
    var_64 = 'two'
    var_65 = [var_63, var_64]
    var_66 = var_58.validate(var_65)
    var_67 = module_0.Integer()
    var_68 = module_0.String()
    var_69 = [var_67, var_68]
    var_70 = module_0.Array(var_69, var_10)
    var_71 = [var_44, var_59]
    var_72 = var_70.validate(var_71)
    var_73 = '1'
    var_74 = 'two'
    var_75 = 'extra'
    var_76 = [var_73, var_74, var_75]
    var_77 = var_70.validate(var_76)
    var_78 = module_0.Integer()
    var_79 = module_0.String()
    var_80 = [var_78, var_79]
    var_81 = module_0.Boolean()
    var_82 = module_0.Array(var_80, var_81)
    var_83 = [var_44, var_59, var_60]
    var_84 = var_82.validate(var_83)
    var_85 = True
    var_86 = module_0.Array(unique_items=var_85)
    var_87 = [var_85, var_74, var_75]
    var_88 = var_86.validate(var_87)
    var_89 = 1
    var_90 = 2
    var_91 = [var_89, var_90, var_90]
    var_92 = var_86.validate(var_91)
    var_93 = module_0.Array(min_items=var_85)
    var_94 = []
    var_95 = var_93.validate(var_94)
    var_96 = module_0.Integer()
    var_97 = module_0.Array(var_96)
    var_98 = [var_85, var_95, var_91]
    var_99 = var_97.serialize(var_98)
    var_100 = var_97.serialize(var_8)
    assert var_100 is None



# Parsed testcases at query #27
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Object()
    var_1 = 'name'
    var_2 = 'age'
    var_3 = module_0.String()
    var_4 = module_0.Integer()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.Object(properties=var_5)
    var_7 = '^S_'
    var_8 = '^I_'
    var_9 = module_0.String()
    var_10 = module_0.Integer()
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = module_0.Object(pattern_properties=var_11)
    var_13 = False
    var_14 = module_0.Object(additional_properties=var_13)
    var_15 = module_0.String()
    var_16 = module_0.Object(property_names=var_15)
    var_17 = 1
    var_18 = module_0.Object(min_properties=var_17)
    var_19 = 10
    var_20 = module_0.Object(max_properties=var_19)
    var_21 = [var_1, var_2]
    var_22 = module_0.Object(required=var_21)
    var_23 = module_0.String()
    var_24 = module_0.Integer()
    var_25 = {var_1: var_23, var_2: var_24}
    var_26 = module_0.String()
    var_27 = module_0.Integer()
    var_28 = {var_7: var_26, var_8: var_27}
    var_29 = module_0.String()
    var_30 = [var_1, var_2]
    var_31 = module_0.Object(properties=var_25, pattern_properties=var_28, additional_properties=var_13, property_names=var_29, min_properties=var_17, max_properties=var_19, required=var_30)
    var_32 = module_0.String()
    var_33 = module_0.Object(additional_properties=var_32)
    var_34 = module_0.String()
    var_35 = module_0.Object(properties=var_34)



# Parsed testcases at query #28
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
    var_25 = 1
    var_26 = var_22.validate(var_25)
    var_27 = module_0.Boolean()
    var_28 = None
    var_29 = var_27.validate(var_28)
    assert var_29 is None
    var_30 = 'null'
    var_31 = var_27.validate(var_30)
    assert var_31 is None
    var_32 = 'none'
    var_33 = var_27.validate(var_32)
    assert var_33 is None
    var_34 = module_0.Boolean()
    var_35 = None
    var_36 = var_34.validate(var_35)
    var_37 = module_0.Boolean()
    var_38 = 'invalid'
    var_39 = var_37.validate(var_38)
    var_40 = 2
    var_41 = var_37.validate(var_40)



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
    var_48 = [var_46, var_47, var_46]
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
    var_61 = [var_59]
    var_62 = module_0.Array(var_61, var_60)
    var_63 = 'two'
    var_64 = 'three'
    var_65 = [var_42, var_63, var_64]
    var_66 = var_62.validate(var_65)
    var_67 = 1
    var_68 = 2
    var_69 = 3
    var_70 = [var_67, var_68, var_69]
    var_71 = var_62.validate(var_70)
    var_72 = [var_59]
    var_73 = module_0.Array(var_72, var_10)
    var_74 = [var_42]
    var_75 = var_73.validate(var_74)
    var_76 = 1
    var_77 = 2
    var_78 = [var_76, var_77]
    var_79 = var_73.validate(var_78)
    var_80 = module_0.Array()
    var_81 = []
    var_82 = var_80.validate(var_81)
    var_83 = module_0.Array(min_items=var_42)
    var_84 = []
    var_85 = var_83.validate(var_84)
    var_86 = module_0.Array(var_59)
    var_87 = module_0.Array(var_86)
    var_88 = [var_42, var_85]
    var_89 = 4
    var_90 = [var_78, var_89]
    var_91 = [var_88, var_90]
    var_92 = var_87.validate(var_91)
    var_93 = 1
    var_94 = 2
    var_95 = [var_93, var_94]
    var_96 = 'three'
    var_97 = 4
    var_98 = [var_96, var_97]
    var_99 = [var_95, var_98]
    var_100 = var_87.validate(var_99)



# Parsed testcases at query #30
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



# Parsed testcases at query #31
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
    var_6 = 'test'
    var_7 = var_3.validate(var_6)
    assert var_7 == 'test'
    var_8 = module_0.Integer()
    var_9 = module_0.String()
    var_10 = [var_8, var_9]
    var_11 = module_0.Union(var_10)
    var_12 = 12.3
    var_13 = var_11.validate(var_12)
    var_14 = True
    var_15 = module_0.Integer()
    var_16 = module_0.String()
    var_17 = [var_15, var_16]
    var_18 = module_0.Union(var_17)
    var_19 = None
    var_20 = var_18.validate(var_19)
    assert var_20 is None
    var_21 = module_0.Integer()
    var_22 = module_0.String()
    var_23 = [var_21, var_22]
    var_24 = module_0.Union(var_23)
    var_25 = None
    var_26 = var_24.validate(var_25)
    var_27 = module_0.Integer()
    var_28 = module_0.Float()
    var_29 = [var_27, var_28]
    var_30 = module_0.Union(var_29)
    var_31 = var_30.validate(var_4)
    assert var_31 == 123
    var_32 = 0
    var_33 = module_0.Integer(minimum=var_32)
    var_34 = 10
    var_35 = module_0.Integer(minimum=var_34)
    var_36 = [var_33, var_35]
    var_37 = module_0.Union(var_36)
    var_38 = 5
    var_39 = var_37.validate(var_38)
    var_40 = exc_info.value.messages()[var_32]
    var_41 = var_40.code
    assert var_41 == 'minimum'
    var_42 = module_0.Integer()
    var_43 = module_0.String()
    var_44 = [var_42, var_43]
    var_45 = module_0.Union(var_44)
    var_46 = []
    var_47 = var_45.validate(var_46)
    var_48 = exc_info.value.messages()[var_32]
    var_49 = var_48.code
    assert var_49 == 'union'



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
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
    var_20 = module_0.Boolean()
    var_21 = [var_18, var_19, var_20]
    var_22 = module_0.Array(var_21)
    var_23 = 'test'
    var_24 = True
    var_25 = [var_4, var_23, var_24]
    var_26 = var_22.serialize(var_25)
    var_27 = 'true'
    var_28 = [var_13, var_23, var_27]
    var_29 = var_22.serialize(var_28)
    var_30 = module_0.Integer()
    var_31 = module_0.String()
    var_32 = [var_30, var_31]
    var_33 = module_0.Boolean()
    var_34 = module_0.Array(var_32, var_33)
    var_35 = True
    var_36 = False
    var_37 = [var_24, var_23, var_35, var_36]
    var_38 = var_34.serialize(var_37)



# Parsed testcases at query #2
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
    var_63 = module_0.Integer()
    var_64 = [var_62, var_63]
    var_65 = module_0.Array(var_64, var_10)
    var_66 = [var_42, var_58]
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
    var_78 = [var_42, var_69, var_70]
    var_79 = var_77.validate(var_78)
    var_80 = 1
    var_81 = 2
    var_82 = 'three'
    var_83 = [var_80, var_81, var_82]
    var_84 = var_77.validate(var_83)
    var_85 = module_0.Array(min_items=var_42)
    var_86 = []
    var_87 = var_85.validate(var_86)
    var_88 = module_0.Array(min_items=var_10)
    var_89 = []
    var_90 = var_88.validate(var_89)



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
    var_10 = module_0.String()
    var_11 = {var_0: var_10}
    var_12 = True
    var_13 = module_0.Object(properties=var_11)
    var_14 = None
    var_15 = var_13.validate(var_14)
    assert var_15 is None
    var_16 = module_0.String()
    var_17 = {var_0: var_16}
    var_18 = False
    var_19 = module_0.Object(properties=var_17)
    var_20 = None
    var_21 = var_19.validate(var_20)
    var_22 = 'not a dict'
    var_23 = var_5.validate(var_22)
    var_24 = module_0.String()
    var_25 = {var_22: var_24}
    var_26 = [var_22]
    var_27 = module_0.Object(properties=var_25, required=var_26)
    var_28 = {}
    var_29 = var_27.validate(var_28)
    var_30 = module_0.String()
    var_31 = module_0.Integer()
    var_32 = {var_28: var_30, var_29: var_31}
    var_33 = module_0.Object(properties=var_32, min_properties=var_12)
    var_34 = {var_28: var_6}
    var_35 = var_33.validate(var_34)
    var_36 = {}
    var_37 = var_33.validate(var_36)
    var_38 = module_0.String()
    var_39 = module_0.Integer()
    var_40 = {var_36: var_38, var_37: var_39}
    var_41 = module_0.Object(properties=var_40, max_properties=var_12)
    var_42 = {var_36: var_6}
    var_43 = var_41.validate(var_42)
    var_44 = 'name'
    var_45 = 'age'
    var_46 = 'John'
    var_47 = 30
    var_48 = {var_44: var_46, var_45: var_47}
    var_49 = var_41.validate(var_48)
    var_50 = 123
    var_51 = 'invalid key'
    var_52 = {var_50: var_51}
    var_53 = var_5.validate(var_52)
    var_54 = '^[a-z]+$'
    var_55 = module_0.String(pattern=var_54)
    var_56 = module_0.String()
    var_57 = {var_50: var_56}
    var_58 = module_0.Object(properties=var_57, property_names=var_55)
    var_59 = {var_50: var_49}
    var_60 = var_58.validate(var_59)
    var_61 = 'Name'
    var_62 = 'John'
    var_63 = {var_61: var_62}
    var_64 = var_58.validate(var_63)
    var_65 = module_0.String()
    var_66 = {var_61: var_65}
    var_67 = module_0.Object(properties=var_66, additional_properties=var_18)
    var_68 = {var_61: var_49}
    var_69 = var_67.validate(var_68)
    var_70 = 'name'
    var_71 = 'age'
    var_72 = 'John'
    var_73 = 30
    var_74 = {var_70: var_72, var_71: var_73}
    var_75 = var_67.validate(var_74)
    var_76 = module_0.String()
    var_77 = {var_70: var_76}
    var_78 = module_0.Integer()
    var_79 = module_0.Object(properties=var_77, additional_properties=var_78)
    var_80 = {var_70: var_75, var_71: var_7}
    var_81 = var_79.validate(var_80)
    var_82 = 'name'
    var_83 = 'age'
    var_84 = 'John'
    var_85 = 'not an integer'
    var_86 = {var_82: var_84, var_83: var_85}
    var_87 = var_79.validate(var_86)
    var_88 = module_0.String()
    var_89 = {var_82: var_88}
    var_90 = '^age_'
    var_91 = module_0.Integer()
    var_92 = {var_90: var_91}
    var_93 = module_0.Object(properties=var_89, pattern_properties=var_92)
    var_94 = 'age_1'
    var_95 = {var_82: var_87, var_94: var_7}
    var_96 = var_93.validate(var_95)
    var_97 = 'name'
    var_98 = 'age_1'
    var_99 = 'John'
    var_100 = 'not an integer'
    var_101 = {var_97: var_99, var_98: var_100}
    var_102 = var_93.validate(var_101)
    var_103 = 'Default'
    var_104 = module_0.String()
    var_105 = {var_97: var_104}
    var_106 = module_0.Object(properties=var_105)
    var_107 = {}
    var_108 = var_106.validate(var_107)



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
    var_7 = False
    var_8 = module_0.String()
    var_9 = None
    var_10 = var_8.validate(var_9)
    var_11 = module_0.String()
    var_12 = 123
    var_13 = var_11.validate(var_12)
    var_14 = module_0.String(allow_blank=var_3)
    var_15 = ''
    var_16 = var_14.validate(var_15)
    assert var_16 == ''
    var_17 = module_0.String(allow_blank=var_7)
    var_18 = ''
    var_19 = var_17.validate(var_18)
    var_20 = module_0.String(trim_whitespace=var_3)
    var_21 = '  hello  '
    var_22 = var_20.validate(var_21)
    assert var_22 == 'hello'
    var_23 = 5
    var_24 = module_0.String(min_length=var_23)
    var_25 = var_24.validate(var_18)
    assert var_25 == 'hello'
    var_26 = 'hi'
    var_27 = var_24.validate(var_26)
    var_28 = module_0.String(max_length=var_23)
    var_29 = var_28.validate(var_26)
    assert var_29 == 'hello'
    var_30 = 'hello world'
    var_31 = var_28.validate(var_30)
    var_32 = '^[a-z]+$'
    var_33 = module_0.String(pattern=var_32)
    var_34 = var_33.validate(var_30)
    assert var_34 == 'hello'
    var_35 = 'Hello'
    var_36 = var_33.validate(var_35)
    var_37 = 'email'
    var_38 = module_0.String(format=var_37)
    var_39 = 'test@example.com'
    var_40 = var_38.validate(var_39)
    assert var_40 == 'test@example.com'
    var_41 = 'invalid-email'
    var_42 = var_38.validate(var_41)
    var_43 = module_0.String()
    var_44 = 'he\x00llo'
    var_45 = var_43.validate(var_44)
    assert var_45 == 'hello'
    var_46 = module_0.String(coerce_types=var_3)
    var_47 = var_46.validate(var_15)
    assert var_47 is None



# Parsed testcases at query #5
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
    var_58 = module_0.String()
    var_59 = [var_57, var_58]
    var_60 = module_0.Array(var_59)
    var_61 = 'two'
    var_62 = [var_47, var_61]
    var_63 = var_60.validate(var_62)
    var_64 = 'one'
    var_65 = 'two'
    var_66 = [var_64, var_65]
    var_67 = var_60.validate(var_66)
    var_68 = module_0.Integer()
    var_69 = [var_68]
    var_70 = module_0.Array(var_69, var_10)
    var_71 = [var_6]
    var_72 = var_70.validate(var_71)
    var_73 = 1
    var_74 = 2
    var_75 = [var_73, var_74]
    var_76 = var_70.validate(var_75)
    var_77 = module_0.Integer()
    var_78 = [var_77]
    var_79 = module_0.String()
    var_80 = module_0.Array(var_78, var_79)
    var_81 = [var_6, var_61]
    var_82 = var_80.validate(var_81)
    var_83 = 1
    var_84 = 2
    var_85 = [var_83, var_84]
    var_86 = var_80.validate(var_85)
    var_87 = True
    var_88 = module_0.Array(unique_items=var_87)
    var_89 = [var_87, var_84, var_85]
    var_90 = var_88.validate(var_89)
    var_91 = 1
    var_92 = 2
    var_93 = [var_91, var_92, var_92]
    var_94 = var_88.validate(var_93)
    var_95 = module_0.Integer()
    var_96 = module_0.Array(var_95)
    var_97 = [var_87, var_92, var_93]
    var_98 = var_96.serialize(var_97)
    var_99 = var_96.serialize(var_8)
    assert var_99 is None
    var_100 = module_0.Integer()
    var_101 = module_0.String()
    var_102 = [var_100, var_101]
    var_103 = module_0.Array(var_102)
    var_104 = [var_87, var_61]
    var_105 = var_103.serialize(var_104)



# Parsed testcases at query #6
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
    var_7 = '12.3'
    var_8 = var_0.validate(var_7)
    var_9 = True
    var_10 = module_0.Number()
    var_11 = None
    var_12 = var_10.validate(var_11)
    assert var_12 is None
    var_13 = module_0.Number()
    var_14 = None
    var_15 = var_13.validate(var_14)
    var_16 = module_0.Number()
    var_17 = True
    var_18 = var_16.validate(var_17)
    var_19 = False
    var_20 = var_16.validate(var_19)
    var_21 = 10
    var_22 = module_0.Number(minimum=var_21)
    var_23 = var_22.validate(var_21)
    assert var_23 == 10
    var_24 = 9
    var_25 = var_22.validate(var_24)
    var_26 = module_0.Number(exclusive_minimum=var_21)
    var_27 = 11
    var_28 = var_26.validate(var_27)
    assert var_28 == 11
    var_29 = 10
    var_30 = var_26.validate(var_29)
    var_31 = module_0.Number(maximum=var_21)
    var_32 = var_31.validate(var_21)
    assert var_32 == 10
    var_33 = 11
    var_34 = var_31.validate(var_33)
    var_35 = module_0.Number(exclusive_maximum=var_21)
    var_36 = 9
    var_37 = var_35.validate(var_36)
    assert var_37 == 9
    var_38 = 10
    var_39 = var_35.validate(var_38)
    var_40 = 5
    var_41 = module_0.Number(multiple_of=var_40)
    var_42 = var_41.validate(var_21)
    assert var_42 == 10
    var_43 = 11
    var_44 = var_41.validate(var_43)
    var_45 = 0.5
    var_46 = module_0.Number(multiple_of=var_45)
    var_47 = 1.5
    var_48 = var_46.validate(var_47)
    var_49 = 1.6
    var_50 = var_46.validate(var_49)
    var_51 = '0.01'
    var_52 = module_0.Number(precision=var_51)
    var_53 = 1.234
    var_54 = var_52.validate(var_53)
    var_55 = module_0.Number()
    var_56 = 'inf'
    var_57 = float(var_56)
    var_58 = var_55.validate(var_57)
    var_59 = '-inf'
    var_60 = float(var_59)
    var_61 = var_55.validate(var_60)
    var_62 = 'nan'
    var_63 = float(var_62)
    var_64 = var_55.validate(var_63)
    var_65 = False
    var_66 = module_0.Number(coerce_types=var_65)
    var_67 = '123'
    var_68 = var_66.validate(var_67)
    var_69 = var_66.validate(var_67)
    assert var_69 == 123
    var_70 = 12.3
    var_71 = var_66.validate(var_70)



# Parsed testcases at query #7
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
    var_17 = 'abc'
    var_18 = var_16.validate(var_17)
    var_19 = module_0.Number()
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
    var_42 = 5.1
    var_43 = var_41.validate(var_42)
    var_44 = 5
    var_45 = var_41.validate(var_44)
    var_46 = 10
    var_47 = module_0.Number(maximum=var_46)
    var_48 = var_47.validate(var_46)
    assert var_48 == 10
    var_49 = 11
    var_50 = var_47.validate(var_49)
    var_51 = module_0.Number(exclusive_maximum=var_46)
    var_52 = 9.9
    var_53 = var_51.validate(var_52)
    var_54 = 10
    var_55 = var_51.validate(var_54)
    var_56 = 3
    var_57 = module_0.Number(multiple_of=var_56)
    var_58 = 9
    var_59 = var_57.validate(var_58)
    assert var_59 == 9
    var_60 = 10
    var_61 = var_57.validate(var_60)
    var_62 = 0.5
    var_63 = module_0.Number(multiple_of=var_62)
    var_64 = 2.5
    var_65 = var_63.validate(var_64)
    var_66 = 2.6
    var_67 = var_63.validate(var_66)



