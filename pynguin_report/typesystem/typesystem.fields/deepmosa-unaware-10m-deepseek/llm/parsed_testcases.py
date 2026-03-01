####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = True
    var_2 = module_0.Array(var_0)
    var_3 = None
    var_4 = var_2.serialize(var_3)
    assert var_4 is None
    var_5 = module_0.Integer()
    var_6 = False
    var_7 = module_0.Array(var_5)
    var_8 = var_7.serialize(var_3)
    assert var_8 is None
    var_9 = module_0.Integer()
    var_10 = module_0.Array(var_9)
    var_11 = 2
    var_12 = 3
    var_13 = [var_1, var_11, var_12]
    var_14 = var_10.serialize(var_13)
    var_15 = module_0.Integer()
    var_16 = module_0.String()
    var_17 = module_0.Boolean()
    var_18 = [var_15, var_16, var_17]
    var_19 = module_0.Array(var_18)
    var_20 = 42
    var_21 = 'hello'
    var_22 = [var_20, var_21, var_1]
    var_23 = var_19.serialize(var_22)
    var_24 = module_0.Decimal()
    var_25 = module_0.Array(var_24)
    var_26 = '1.5'
    var_27 = '2.75'
    var_28 = var_25.serialize(var_22)
    var_29 = module_0.Array(var_3)
    var_30 = 'two'
    var_31 = 'three'
    var_32 = {var_31: var_12}
    var_33 = [var_1, var_30, var_32]
    var_34 = var_29.serialize(var_33)
    var_35 = module_0.String()
    var_36 = module_0.Array(var_35)
    var_37 = []
    var_38 = var_36.serialize(var_37)
    var_39 = module_0.String()
    var_40 = module_0.Array(var_39)
    var_41 = 'a'
    var_42 = 'b'
    var_43 = 'c'
    var_44 = [var_41, var_42, var_43]
    var_45 = var_40.serialize(var_44)
    var_46 = module_0.Integer()
    var_47 = module_0.String()
    var_48 = [var_46, var_47]
    var_49 = module_0.Array(var_48)
    var_50 = [var_1]
    var_51 = var_49.serialize(var_50)
    var_52 = module_0.Integer()
    var_53 = [var_52]
    var_54 = module_0.Array(var_53, var_1)
    var_55 = 'extra'
    var_56 = 3.5
    var_57 = [var_1, var_55, var_56]
    var_58 = var_54.serialize(var_57)



# Parsed testcases at query #2
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
    var_28 = [var_22, var_20]
    var_29 = var_27.validate(var_28)
    var_30 = 1
    var_31 = [var_30]
    var_32 = var_27.validate(var_31)
    var_33 = 1
    var_34 = 2
    var_35 = 3
    var_36 = [var_33, var_34, var_35]
    var_37 = var_27.validate(var_36)
    var_38 = module_0.Integer()
    var_39 = module_0.Array(var_38)
    var_40 = [var_33, var_20, var_37]
    var_41 = var_39.validate(var_40)
    var_42 = 1
    var_43 = 'invalid'
    var_44 = 3
    var_45 = [var_42, var_43, var_44]
    var_46 = var_39.validate(var_45)
    var_47 = module_0.Integer()
    var_48 = module_0.String()
    var_49 = [var_47, var_48]
    var_50 = module_0.Array(var_49)
    var_51 = 'hello'
    var_52 = [var_42, var_51]
    var_53 = var_50.validate(var_52)
    var_54 = 'invalid'
    var_55 = 'hello'
    var_56 = [var_54, var_55]
    var_57 = var_50.validate(var_56)
    var_58 = module_0.Integer()
    var_59 = module_0.String()
    var_60 = [var_58, var_59]
    var_61 = module_0.Array(var_60, var_57)
    var_62 = [var_54, var_51]
    var_63 = var_61.validate(var_62)
    var_64 = 1
    var_65 = 'hello'
    var_66 = 'extra'
    var_67 = [var_64, var_65, var_66]
    var_68 = var_61.validate(var_67)
    var_69 = module_0.Integer()
    var_70 = [var_69]
    var_71 = module_0.String()
    var_72 = module_0.Array(var_70, var_71)
    var_73 = 'extra1'
    var_74 = 'extra2'
    var_75 = [var_64, var_73, var_74]
    var_76 = var_72.validate(var_75)
    var_77 = 1
    var_78 = 2
    var_79 = 3
    var_80 = [var_77, var_78, var_79]
    var_81 = var_72.validate(var_80)
    var_82 = module_0.Array(unique_items=var_77)
    var_83 = [var_77, var_20, var_81]
    var_84 = var_82.validate(var_83)
    var_85 = 1
    var_86 = 2
    var_87 = [var_85, var_86, var_85]
    var_88 = var_82.validate(var_87)
    var_89 = 'id'
    var_90 = 'name'
    var_91 = module_0.Integer()
    var_92 = module_0.String()
    var_93 = {var_89: var_91, var_90: var_92}
    var_94 = module_0.Object(properties=var_93)
    var_95 = module_0.Array(var_94)
    var_96 = 'Alice'
    var_97 = {var_89: var_85, var_90: var_96}
    var_98 = 'Bob'
    var_99 = {var_89: var_20, var_90: var_98}
    var_100 = [var_97, var_99]
    var_101 = var_95.validate(var_100)
    var_102 = module_0.Array()
    var_103 = 'key'
    var_104 = 'value'
    var_105 = {var_103: var_104}
    var_106 = [var_85, var_51, var_105]
    var_107 = var_102.validate(var_106)
    var_108 = module_0.Integer()
    var_109 = module_0.Array(var_108)
    var_110 = 'a'
    var_111 = 'b'
    var_112 = 'c'
    var_113 = [var_110, var_111, var_112]
    var_114 = var_109.validate(var_113)



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
    var_35 = var_34.validate(var_9)
    assert var_35 is True
    var_36 = var_34.validate(var_3)
    assert var_36 is False
    var_37 = 'true'
    var_38 = var_34.validate(var_37)
    var_39 = module_0.Boolean(coerce_types=var_37)
    var_40 = 'invalid'
    var_41 = var_39.validate(var_40)
    var_42 = 2
    var_43 = var_39.validate(var_42)
    var_44 = 'TRUE'
    var_45 = var_39.validate(var_44)
    assert var_45 is True
    var_46 = 'FALSE'
    var_47 = var_39.validate(var_46)
    assert var_47 is False
    var_48 = 'ON'
    var_49 = var_39.validate(var_48)
    assert var_49 is True
    var_50 = 'OFF'
    var_51 = var_39.validate(var_50)
    assert var_51 is False



# Parsed testcases at query #4
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
    var_11 = module_0.Boolean(coerce_types=var_3)
    var_12 = 'true'
    var_13 = var_11.validate(var_12)
    var_14 = module_0.Boolean(coerce_types=var_12)
    var_15 = 'true'
    var_16 = var_14.validate(var_15)
    assert var_16 is True
    var_17 = 'false'
    var_18 = var_14.validate(var_17)
    assert var_18 is False
    var_19 = 'on'
    var_20 = var_14.validate(var_19)
    assert var_20 is True
    var_21 = 'off'
    var_22 = var_14.validate(var_21)
    assert var_22 is False
    var_23 = '1'
    var_24 = var_14.validate(var_23)
    assert var_24 is True
    var_25 = '0'
    var_26 = var_14.validate(var_25)
    assert var_26 is False
    var_27 = ''
    var_28 = var_14.validate(var_27)
    assert var_28 is False
    var_29 = 'TRUE'
    var_30 = var_14.validate(var_29)
    assert var_30 is True
    var_31 = 'FALSE'
    var_32 = var_14.validate(var_31)
    assert var_32 is False
    var_33 = 'On'
    var_34 = var_14.validate(var_33)
    assert var_34 is True
    var_35 = 'Off'
    var_36 = var_14.validate(var_35)
    assert var_36 is False
    var_37 = var_14.validate(var_12)
    assert var_37 is True
    var_38 = var_14.validate(var_3)
    assert var_38 is False
    var_39 = module_0.Boolean(coerce_types=var_12)
    var_40 = 'null'
    var_41 = var_39.validate(var_40)
    assert var_41 is None
    var_42 = 'none'
    var_43 = var_39.validate(var_42)
    assert var_43 is None
    var_44 = var_39.validate(var_27)
    assert var_44 is None
    var_45 = module_0.Boolean(coerce_types=var_12)
    var_46 = 'invalid'
    var_47 = var_45.validate(var_46)
    var_48 = []
    var_49 = var_45.validate(var_48)
    var_50 = module_0.Boolean()
    var_51 = 'invalid'



# Parsed testcases at query #5
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = True
    var_2 = module_0.Array(var_0)
    var_3 = None
    var_4 = var_2.serialize(var_3)
    assert var_4 is None
    var_5 = module_0.String()
    var_6 = False
    var_7 = module_0.Array(var_5)
    var_8 = var_7.serialize(var_3)
    assert var_8 is None
    var_9 = module_0.Integer()
    var_10 = module_0.Array(var_9)
    var_11 = 2
    var_12 = 3
    var_13 = [var_1, var_11, var_12]
    var_14 = var_10.serialize(var_13)
    var_15 = module_0.Integer()
    var_16 = module_0.String()
    var_17 = module_0.Boolean()
    var_18 = [var_15, var_16, var_17]
    var_19 = module_0.Array(var_18)
    var_20 = 42
    var_21 = 'hello'
    var_22 = [var_20, var_21, var_1]
    var_23 = var_19.serialize(var_22)
    var_24 = module_0.Integer()
    var_25 = module_0.Array(var_24)
    var_26 = module_0.Array(var_25)
    var_27 = [var_1, var_11]
    var_28 = 4
    var_29 = [var_12, var_28]
    var_30 = [var_27, var_29]
    var_31 = var_26.serialize(var_30)
    var_32 = module_0.Array(var_3)
    var_33 = 'two'
    var_34 = 'three'
    var_35 = {var_34: var_12}
    var_36 = [var_1, var_33, var_35]
    var_37 = var_32.serialize(var_36)
    var_38 = 'a'
    var_39 = 'b'
    var_40 = 'c'
    var_41 = [var_38, var_39, var_40]
    var_42 = var_32.serialize(var_41)
    var_43 = module_0.String()
    var_44 = module_0.Array(var_43)
    var_45 = []
    var_46 = var_44.serialize(var_45)
    var_47 = module_0.String()
    var_48 = module_0.Array(var_47)
    var_49 = [var_38, var_39, var_40]
    var_50 = var_48.serialize(var_49)
    var_51 = module_0.Integer()
    var_52 = module_0.String()
    var_53 = [var_51, var_52]
    var_54 = module_0.Array(var_53)
    var_55 = 'extra'
    var_56 = 'items'
    var_57 = [var_1, var_33, var_55, var_56]
    var_58 = var_54.serialize(var_57)



# Parsed testcases at query #6
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
    var_35 = var_34.validate(var_9)
    assert var_35 is True
    var_36 = var_34.validate(var_3)
    assert var_36 is False
    var_37 = 'true'
    var_38 = var_34.validate(var_37)
    var_39 = 1
    var_40 = var_34.validate(var_39)
    var_41 = module_0.Boolean(coerce_types=var_39)
    var_42 = 'invalid'
    var_43 = var_41.validate(var_42)
    var_44 = 2
    var_45 = var_41.validate(var_44)
    var_46 = module_0.Boolean(coerce_types=var_44)
    var_47 = 'TRUE'
    var_48 = var_46.validate(var_47)
    assert var_48 is True
    var_49 = 'FALSE'
    var_50 = var_46.validate(var_49)
    assert var_50 is False
    var_51 = 'On'
    var_52 = var_46.validate(var_51)
    assert var_52 is True
    var_53 = 'Off'
    var_54 = var_46.validate(var_53)
    assert var_54 is False
    var_55 = module_0.Boolean()
    var_56 = 'type'



# Parsed testcases at query #7
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
    var_11 = module_0.Boolean(coerce_types=var_3)
    var_12 = 'true'
    var_13 = var_11.validate(var_12)
    var_14 = module_0.Boolean(coerce_types=var_12)
    var_15 = 'true'
    var_16 = var_14.validate(var_15)
    assert var_16 is True
    var_17 = 'false'
    var_18 = var_14.validate(var_17)
    assert var_18 is False
    var_19 = 'on'
    var_20 = var_14.validate(var_19)
    assert var_20 is True
    var_21 = 'off'
    var_22 = var_14.validate(var_21)
    assert var_22 is False
    var_23 = '1'
    var_24 = var_14.validate(var_23)
    assert var_24 is True
    var_25 = '0'
    var_26 = var_14.validate(var_25)
    assert var_26 is False
    var_27 = ''
    var_28 = var_14.validate(var_27)
    assert var_28 is False
    var_29 = var_14.validate(var_12)
    assert var_29 is True
    var_30 = var_14.validate(var_3)
    assert var_30 is False
    var_31 = 'TRUE'
    var_32 = var_14.validate(var_31)
    assert var_32 is True
    var_33 = 'FALSE'
    var_34 = var_14.validate(var_33)
    assert var_34 is False
    var_35 = 'On'
    var_36 = var_14.validate(var_35)
    assert var_36 is True
    var_37 = 'Off'
    var_38 = var_14.validate(var_37)
    assert var_38 is False
    var_39 = module_0.Boolean(coerce_types=var_12)
    var_40 = 'null'
    var_41 = var_39.validate(var_40)
    assert var_41 is None
    var_42 = 'none'
    var_43 = var_39.validate(var_42)
    assert var_43 is None
    var_44 = var_39.validate(var_27)
    assert var_44 is None
    var_45 = module_0.Boolean(coerce_types=var_12)
    var_46 = 'invalid'
    var_47 = var_45.validate(var_46)
    var_48 = []
    var_49 = var_45.validate(var_48)
    var_50 = module_0.Boolean()
    var_51 = var_50.validate(var_48)
    assert var_51 is True
    var_52 = var_50.validate(var_3)
    assert var_52 is False
    var_53 = None
    var_54 = var_50.validate(var_53)



# Parsed testcases at query #8
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Array()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = var_0.validate(var_4)
    var_6 = []
    var_7 = var_0.validate(var_6)
    var_8 = True
    var_9 = module_0.Array()
    var_10 = None
    var_11 = var_9.validate(var_10)
    assert var_11 is None
    var_12 = False
    var_13 = module_0.Array()
    var_14 = None
    var_15 = var_13.validate(var_14)
    var_16 = module_0.Array()
    var_17 = 'not a list'
    var_18 = var_16.validate(var_17)
    var_19 = module_0.Array(min_items=var_18)
    var_20 = [var_8, var_18]
    var_21 = var_19.validate(var_20)
    var_22 = [var_8, var_18, var_3]
    var_23 = var_19.validate(var_22)
    var_24 = 1
    var_25 = [var_24]
    var_26 = var_19.validate(var_25)
    var_27 = module_0.Array(min_items=var_8)
    var_28 = []
    var_29 = var_27.validate(var_28)
    var_30 = module_0.Array(max_items=var_29)
    var_31 = [var_8]
    var_32 = var_30.validate(var_31)
    var_33 = [var_8, var_29]
    var_34 = var_30.validate(var_33)
    var_35 = 1
    var_36 = 2
    var_37 = 3
    var_38 = [var_35, var_36, var_37]
    var_39 = var_30.validate(var_38)
    var_40 = module_0.Array(exact_items=var_36)
    var_41 = [var_8, var_36]
    var_42 = var_40.validate(var_41)
    var_43 = 1
    var_44 = [var_43]
    var_45 = var_40.validate(var_44)
    var_46 = 1
    var_47 = 2
    var_48 = 3
    var_49 = [var_46, var_47, var_48]
    var_50 = var_40.validate(var_49)
    var_51 = module_0.Integer()
    var_52 = module_0.Array(var_51)
    var_53 = [var_8, var_47, var_48]
    var_54 = var_52.validate(var_53)
    var_55 = 1
    var_56 = 'invalid'
    var_57 = 3
    var_58 = [var_55, var_56, var_57]
    var_59 = var_52.validate(var_58)
    var_60 = module_0.Integer()
    var_61 = module_0.String()
    var_62 = [var_60, var_61]
    var_63 = module_0.Array(var_62)
    var_64 = 'text'
    var_65 = [var_8, var_64]
    var_66 = var_63.validate(var_65)
    var_67 = 'invalid'
    var_68 = 'text'
    var_69 = [var_67, var_68]
    var_70 = var_63.validate(var_69)
    var_71 = module_0.Integer()
    var_72 = module_0.String()
    var_73 = [var_71, var_72]
    var_74 = module_0.Array(var_73, var_12)
    var_75 = [var_8, var_64]
    var_76 = var_74.validate(var_75)
    var_77 = 'extra'
    var_78 = [var_8, var_64, var_77]
    var_79 = var_74.validate(var_78)
    var_80 = module_0.Array(var_73, var_12, max_items=var_68)
    var_81 = 1
    var_82 = 'text'
    var_83 = 'extra'
    var_84 = [var_81, var_82, var_83]
    var_85 = var_80.validate(var_84)
    var_86 = module_0.String()
    var_87 = module_0.Integer()
    var_88 = [var_87]
    var_89 = module_0.Array(var_88, var_86)
    var_90 = 'valid'
    var_91 = [var_8, var_90, var_64]
    var_92 = var_89.validate(var_91)
    var_93 = 1
    var_94 = 2
    var_95 = 3
    var_96 = [var_93, var_94, var_95]
    var_97 = var_89.validate(var_96)
    var_98 = True
    var_99 = module_0.Array(unique_items=var_98)
    var_100 = [var_98, var_94, var_95]
    var_101 = var_99.validate(var_100)
    var_102 = 1
    var_103 = 2
    var_104 = [var_102, var_103, var_102]
    var_105 = var_99.validate(var_104)
    var_106 = 10
    var_107 = module_0.Integer(minimum=var_12, maximum=var_106)
    var_108 = True
    var_109 = module_0.Array(var_107, min_items=var_98, max_items=var_104, unique_items=var_108)
    var_110 = [var_108, var_103, var_104]
    var_111 = var_109.validate(var_110)
    var_112 = 1
    var_113 = 2
    var_114 = [var_112, var_113, var_112]
    var_115 = var_109.validate(var_114)
    var_116 = 1
    var_117 = 2
    var_118 = 11
    var_119 = [var_116, var_117, var_118]
    var_120 = var_109.validate(var_119)
    var_121 = module_0.Integer(minimum=var_12)
    var_122 = module_0.Array(var_121)
    var_123 = -1
    var_124 = 'invalid'
    var_125 = -2
    var_126 = [var_123, var_124, var_125]
    var_127 = var_122.validate(var_126)



# Parsed testcases at query #9
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = var_0.has_default()
    var_2 = 'test_default'
    var_3 = module_0.Field(default=var_2)
    var_4 = var_3.has_default()
    var_5 = var_3.get_default_value()
    assert var_5 == 'test_default'
    var_6 = None
    var_7 = module_0.Field(default=var_6)
    var_8 = var_7.has_default()
    var_9 = var_7.get_default_value()
    assert var_9 is None
    var_10 = 42
    var_11 = module_0.Field(default=var_10)
    var_12 = var_11.has_default()
    var_13 = var_11.get_default_value()
    assert var_13 == 42
    var_14 = 1
    var_15 = 2
    var_16 = 3
    var_17 = [var_14, var_15, var_16]
    var_18 = module_0.Field(default=var_17)
    var_19 = var_18.has_default()
    var_20 = var_18.get_default_value()
    var_21 = True
    var_22 = module_0.Field(allow_null=var_21)
    var_23 = var_22.has_default()
    var_24 = var_22.get_default_value()
    assert var_24 is None
    var_25 = True
    var_26 = 'explicit'
    var_27 = module_0.Field(default=var_26, allow_null=var_25)
    var_28 = var_27.has_default()
    var_29 = var_27.get_default_value()
    assert var_29 == 'explicit'



# Parsed testcases at query #10
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
    var_20 = [var_6, var_16, var_3]
    var_21 = var_17.validate(var_20)
    var_22 = 1
    var_23 = [var_22]
    var_24 = var_17.validate(var_23)
    var_25 = module_0.Array(min_items=var_6)
    var_26 = []
    var_27 = var_25.validate(var_26)
    var_28 = module_0.Array(max_items=var_24)
    var_29 = [var_6, var_27, var_24]
    var_30 = var_28.validate(var_29)
    var_31 = [var_6, var_27]
    var_32 = var_28.validate(var_31)
    var_33 = 1
    var_34 = 2
    var_35 = 3
    var_36 = 4
    var_37 = [var_33, var_34, var_35, var_36]
    var_38 = var_28.validate(var_37)
    var_39 = module_0.Array(exact_items=var_34)
    var_40 = [var_38, var_34]
    var_41 = var_39.validate(var_40)
    var_42 = 1
    var_43 = [var_42]
    var_44 = var_39.validate(var_43)
    var_45 = 1
    var_46 = 2
    var_47 = 3
    var_48 = [var_45, var_46, var_47]
    var_49 = var_39.validate(var_48)
    var_50 = module_0.Integer()
    var_51 = module_0.Array(var_50)
    var_52 = [var_38, var_46, var_47]
    var_53 = var_51.validate(var_52)
    var_54 = 1
    var_55 = 'invalid'
    var_56 = 3
    var_57 = [var_54, var_55, var_56]
    var_58 = var_51.validate(var_57)
    var_59 = module_0.Integer()
    var_60 = module_0.String()
    var_61 = module_0.Boolean()
    var_62 = [var_59, var_60, var_61]
    var_63 = module_0.Array(var_62)
    var_64 = 'test'
    var_65 = True
    var_66 = [var_38, var_64, var_65]
    var_67 = var_63.validate(var_66)
    var_68 = 1
    var_69 = 'test'
    var_70 = 'not boolean'
    var_71 = [var_68, var_69, var_70]
    var_72 = var_63.validate(var_71)
    var_73 = module_0.Integer()
    var_74 = module_0.String()
    var_75 = [var_73, var_74]
    var_76 = module_0.Array(var_75, var_10)
    var_77 = [var_65, var_64]
    var_78 = var_76.validate(var_77)
    var_79 = 1
    var_80 = 'test'
    var_81 = 'extra'
    var_82 = [var_79, var_80, var_81]
    var_83 = var_76.validate(var_82)
    var_84 = module_0.Integer()
    var_85 = module_0.String()
    var_86 = [var_84, var_85]
    var_87 = module_0.Boolean()
    var_88 = module_0.Array(var_86, var_87)
    var_89 = True
    var_90 = [var_65, var_64, var_89, var_10]
    var_91 = var_88.validate(var_90)
    var_92 = 1
    var_93 = 'test'
    var_94 = 'not boolean'
    var_95 = [var_92, var_93, var_94]
    var_96 = var_88.validate(var_95)
    var_97 = True
    var_98 = module_0.Array(unique_items=var_97)
    var_99 = [var_97, var_93, var_94]
    var_100 = var_98.validate(var_99)
    var_101 = 1
    var_102 = 2
    var_103 = [var_101, var_102, var_101]
    var_104 = var_98.validate(var_103)
    var_105 = module_0.Integer()
    var_106 = module_0.Array(var_105)
    var_107 = module_0.Array(var_106)
    var_108 = [var_97, var_102]
    var_109 = 4
    var_110 = [var_103, var_109]
    var_111 = [var_108, var_110]
    var_112 = var_107.validate(var_111)
    var_113 = 1
    var_114 = 2
    var_115 = [var_113, var_114]
    var_116 = 'invalid'
    var_117 = 4
    var_118 = [var_116, var_117]
    var_119 = [var_115, var_118]
    var_120 = var_107.validate(var_119)
    var_121 = module_0.Array()
    var_122 = True
    var_123 = [var_97, var_64, var_122, var_119]
    var_124 = var_121.validate(var_123)
    var_125 = module_0.Integer(minimum=var_10)
    var_126 = module_0.Array(var_125)
    var_127 = -1
    var_128 = -2
    var_129 = -3
    var_130 = [var_127, var_128, var_129]
    var_131 = var_126.validate(var_130)



# Parsed testcases at query #11
#--------------------------


import typesystem.fields as module_0
import typesystem.formats as module_1

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
    var_17 = '  hello  '
    var_18 = var_16.validate(var_17)
    assert var_18 == 'hello'
    var_19 = False
    var_20 = module_0.String(trim_whitespace=var_19)
    var_21 = var_20.validate(var_17)
    assert var_21 == '  hello  '
    var_22 = 3
    var_23 = module_0.String(min_length=var_22)
    var_24 = 'abc'
    var_25 = var_23.validate(var_24)
    assert var_25 == 'abc'
    var_26 = 'ab'
    var_27 = var_23.validate(var_26)
    var_28 = 5
    var_29 = module_0.String(max_length=var_28)
    var_30 = 'abcde'
    var_31 = var_29.validate(var_30)
    assert var_31 == 'abcde'
    var_32 = 'abcdef'
    var_33 = var_29.validate(var_32)
    var_34 = '^\\d+$'
    var_35 = module_0.String(pattern=var_34)
    var_36 = '123'
    var_37 = var_35.validate(var_36)
    assert var_37 == '123'
    var_38 = 'abc'
    var_39 = var_35.validate(var_38)
    var_40 = 'email'
    var_41 = module_0.String(format=var_40)
    var_42 = 'test@example.com'
    var_43 = var_41.validate(var_42)
    assert var_43 == 'test@example.com'
    var_44 = 'not-an-email'
    var_45 = var_41.validate(var_44)
    var_46 = module_0.String()
    var_47 = 'hello\x00world'
    var_48 = var_46.validate(var_47)
    assert var_48 == 'helloworld'
    var_49 = module_0.String()
    var_50 = 123
    var_51 = var_49.validate(var_50)
    var_52 = module_0.String(allow_blank=var_3, coerce_types=var_3)
    var_53 = var_52.validate(var_5)
    assert var_53 == ''
    var_54 = module_0.String(coerce_types=var_3)
    var_55 = var_54.validate(var_11)
    assert var_55 is None
    var_56 = 'uuid'
    var_57 = module_0.String(format=var_56)
    var_58 = module_1.UUIDFormat()
    var_59 = '12345678-1234-5678-1234-567812345678'
    var_60 = var_58.validate(var_59)
    var_61 = var_57.validate(var_60)
    var_62 = 2
    var_63 = 4
    var_64 = '^[a-z]+$'
    var_65 = module_0.String(max_length=var_63, min_length=var_62, pattern=var_64)
    var_66 = var_65.validate(var_24)
    assert var_66 == 'abc'
    var_67 = 'a'
    var_68 = var_65.validate(var_67)
    var_69 = 'abcde'
    var_70 = var_65.validate(var_69)
    var_71 = '123'
    var_72 = var_65.validate(var_71)



# Parsed testcases at query #12
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = var_3.any_of
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = 0
    var_7 = var_3.any_of[var_6]
    var_8 = 1
    var_9 = var_3.any_of[var_8]
    var_10 = True
    var_11 = module_0.String()
    var_12 = module_0.Integer()
    var_13 = [var_11, var_12]
    var_14 = module_0.Union(var_13)
    var_15 = 'hello'
    var_16 = var_3.validate(var_15)
    assert var_16 == 'hello'
    var_17 = 123
    var_18 = var_3.validate(var_17)
    assert var_18 == 123
    var_19 = None
    var_20 = var_14.validate(var_19)
    assert var_20 is None
    var_21 = None
    var_22 = var_3.validate(var_21)
    var_23 = True
    var_24 = var_3.validate(var_23)
    var_25 = 5
    var_26 = module_0.String(min_length=var_25)
    var_27 = 10
    var_28 = module_0.Integer(minimum=var_27)
    var_29 = [var_26, var_28]
    var_30 = module_0.Union(var_29)
    var_31 = 'hi'
    var_32 = var_30.validate(var_31)
    var_33 = 5
    var_34 = var_30.validate(var_33)
    var_35 = True
    var_36 = var_30.validate(var_35)
    var_37 = module_0.String()
    var_38 = module_0.Boolean()
    var_39 = module_0.Integer()
    var_40 = [var_37, var_38, var_39]
    var_41 = module_0.Union(var_40)
    var_42 = 'test'
    var_43 = var_41.validate(var_42)
    assert var_43 == 'test'
    var_44 = True
    var_45 = var_41.validate(var_44)
    assert var_45 is True
    var_46 = 42
    var_47 = var_41.validate(var_46)
    assert var_47 == 42
    var_48 = []
    var_49 = module_0.Union(var_48)
    var_50 = 'anything'
    var_51 = var_49.validate(var_50)



# Parsed testcases at query #13
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
    var_22 = module_0.String()
    var_23 = {var_20: var_22}
    var_24 = module_0.Object(properties=var_23)
    var_25 = 'not an object'
    var_26 = var_24.validate(var_25)
    var_27 = module_0.String()
    var_28 = {var_25: var_27}
    var_29 = [var_25]
    var_30 = module_0.Object(properties=var_28, required=var_29)
    var_31 = 'age'
    var_32 = 30
    var_33 = {var_31: var_32}
    var_34 = var_30.validate(var_33)
    var_35 = 'Unknown'
    var_36 = module_0.String()
    var_37 = {var_31: var_36}
    var_38 = module_0.Object(properties=var_37)
    var_39 = {}
    var_40 = var_38.validate(var_39)
    var_41 = 'a'
    var_42 = 'b'
    var_43 = module_0.String()
    var_44 = module_0.String()
    var_45 = {var_41: var_43, var_42: var_44}
    var_46 = module_0.Object(properties=var_45, min_properties=var_12)
    var_47 = {}
    var_48 = var_46.validate(var_47)
    var_49 = module_0.String()
    var_50 = {var_41: var_49}
    var_51 = module_0.Object(properties=var_50, max_properties=var_12)
    var_52 = 'a'
    var_53 = 'b'
    var_54 = 'test'
    var_55 = 'test2'
    var_56 = {var_52: var_54, var_53: var_55}
    var_57 = var_51.validate(var_56)
    var_58 = module_0.String()
    var_59 = {var_52: var_58}
    var_60 = module_0.Object(properties=var_59, additional_properties=var_12)
    var_61 = 'extra'
    var_62 = 'value'
    var_63 = {var_52: var_57, var_61: var_62}
    var_64 = var_60.validate(var_63)
    var_65 = module_0.String()
    var_66 = {var_52: var_65}
    var_67 = module_0.Object(properties=var_66, additional_properties=var_18)
    var_68 = 'name'
    var_69 = 'extra'
    var_70 = 'John'
    var_71 = 'value'
    var_72 = {var_68: var_70, var_69: var_71}
    var_73 = var_67.validate(var_72)
    var_74 = module_0.String()
    var_75 = {var_68: var_74}
    var_76 = module_0.Integer()
    var_77 = module_0.Object(properties=var_75, additional_properties=var_76)
    var_78 = 42
    var_79 = {var_68: var_73, var_61: var_78}
    var_80 = var_77.validate(var_79)
    var_81 = '^test_'
    var_82 = module_0.String()
    var_83 = {var_81: var_82}
    var_84 = module_0.Object(pattern_properties=var_83)
    var_85 = 'test_field'
    var_86 = {var_85: var_62}
    var_87 = var_84.validate(var_86)
    var_88 = '^[a-z]+$'
    var_89 = module_0.String(pattern=var_88)
    var_90 = module_0.Object(additional_properties=var_12, property_names=var_89)
    var_91 = 'UPPERCASE'
    var_92 = 'value'
    var_93 = {var_91: var_92}
    var_94 = var_90.validate(var_93)
    var_95 = module_0.Object(additional_properties=var_12)
    var_96 = 123
    var_97 = 'value'
    var_98 = {var_96: var_97}
    var_99 = var_95.validate(var_98)
    var_100 = 'nested'
    var_101 = 'inner'
    var_102 = module_0.Integer()
    var_103 = {var_101: var_102}
    var_104 = module_0.Object(properties=var_103)
    var_105 = {var_100: var_104}
    var_106 = module_0.Object(properties=var_105)
    var_107 = 'nested'
    var_108 = 'inner'
    var_109 = 'not an int'
    var_110 = {var_108: var_109}
    var_111 = {var_107: var_110}
    var_112 = var_106.validate(var_111)
    var_113 = 2
    var_114 = module_0.String(min_length=var_113)
    var_115 = module_0.Integer(minimum=var_18)
    var_116 = {var_107: var_114, var_108: var_115}
    var_117 = [var_107]
    var_118 = 3
    var_119 = module_0.Object(properties=var_116, min_properties=var_12, max_properties=var_118, required=var_117)
    var_120 = {var_107: var_112, var_108: var_7}
    var_121 = var_119.validate(var_120)



# Parsed testcases at query #14
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
    var_22 = module_0.String()
    var_23 = {var_20: var_22}
    var_24 = module_0.Object(properties=var_23)
    var_25 = 'not an object'
    var_26 = var_24.validate(var_25)
    var_27 = module_0.String()
    var_28 = {var_25: var_27}
    var_29 = [var_25]
    var_30 = module_0.Object(properties=var_28, required=var_29)
    var_31 = {}
    var_32 = var_30.validate(var_31)
    var_33 = 'Unknown'
    var_34 = module_0.String()
    var_35 = {var_31: var_34}
    var_36 = module_0.Object(properties=var_35)
    var_37 = {}
    var_38 = var_36.validate(var_37)
    var_39 = 'a'
    var_40 = 'b'
    var_41 = module_0.String()
    var_42 = module_0.String()
    var_43 = {var_39: var_41, var_40: var_42}
    var_44 = module_0.Object(properties=var_43, min_properties=var_12)
    var_45 = {}
    var_46 = var_44.validate(var_45)
    var_47 = module_0.String()
    var_48 = {var_39: var_47}
    var_49 = module_0.Object(properties=var_48, max_properties=var_12)
    var_50 = 'a'
    var_51 = 'b'
    var_52 = 'test'
    var_53 = 'extra'
    var_54 = {var_50: var_52, var_51: var_53}
    var_55 = var_49.validate(var_54)
    var_56 = '^test_'
    var_57 = module_0.String()
    var_58 = {var_56: var_57}
    var_59 = module_0.Object(pattern_properties=var_58)
    var_60 = 'test_key'
    var_61 = 'other'
    var_62 = 'value'
    var_63 = 'ignored'
    var_64 = {var_60: var_62, var_61: var_63}
    var_65 = var_59.validate(var_64)
    var_66 = module_0.String()
    var_67 = {var_50: var_66}
    var_68 = module_0.Object(properties=var_67, additional_properties=var_18)
    var_69 = 'name'
    var_70 = 'extra'
    var_71 = 'John'
    var_72 = 'field'
    var_73 = {var_69: var_71, var_70: var_72}
    var_74 = var_68.validate(var_73)
    var_75 = module_0.String()
    var_76 = {var_69: var_75}
    var_77 = module_0.Integer()
    var_78 = module_0.Object(properties=var_76, additional_properties=var_77)
    var_79 = 'extra'
    var_80 = 42
    var_81 = {var_69: var_74, var_79: var_80}
    var_82 = var_78.validate(var_81)
    var_83 = {}
    var_84 = '^[a-z]+$'
    var_85 = module_0.String(pattern=var_84)
    var_86 = module_0.Object(properties=var_83, property_names=var_85)
    var_87 = 'INVALID'
    var_88 = 'value'
    var_89 = {var_87: var_88}
    var_90 = var_86.validate(var_89)
    var_91 = module_0.String()
    var_92 = {var_87: var_91}
    var_93 = module_0.Object(properties=var_92)
    var_94 = 123
    var_95 = 'value'
    var_96 = {var_94: var_95}
    var_97 = var_93.validate(var_96)
    var_98 = 'person'
    var_99 = module_0.Integer(minimum=var_18)
    var_100 = {var_95: var_99}
    var_101 = module_0.Object(properties=var_100)
    var_102 = {var_98: var_101}
    var_103 = module_0.Object(properties=var_102)
    var_104 = 'person'
    var_105 = 'age'
    var_106 = -5
    var_107 = {var_105: var_106}
    var_108 = {var_104: var_107}
    var_109 = var_103.validate(var_108)
    var_110 = 'id'
    var_111 = 'email'
    var_112 = module_0.Integer()
    var_113 = module_0.String()
    var_114 = {var_110: var_112, var_111: var_113}
    var_115 = [var_110]
    var_116 = 3
    var_117 = module_0.Object(properties=var_114, min_properties=var_12, max_properties=var_116, required=var_115)
    var_118 = 'test@example.com'
    var_119 = {var_110: var_12, var_111: var_118}
    var_120 = var_117.validate(var_119)



# Parsed testcases at query #15
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
    var_20 = [var_6, var_16, var_3]
    var_21 = var_17.validate(var_20)
    var_22 = 1
    var_23 = [var_22]
    var_24 = var_17.validate(var_23)
    var_25 = module_0.Array(min_items=var_6)
    var_26 = []
    var_27 = var_25.validate(var_26)
    var_28 = module_0.Array(max_items=var_24)
    var_29 = [var_6, var_27, var_24]
    var_30 = var_28.validate(var_29)
    var_31 = [var_6, var_27]
    var_32 = var_28.validate(var_31)
    var_33 = 1
    var_34 = 2
    var_35 = 3
    var_36 = 4
    var_37 = [var_33, var_34, var_35, var_36]
    var_38 = var_28.validate(var_37)
    var_39 = module_0.Array(exact_items=var_35)
    var_40 = [var_38, var_34, var_35]
    var_41 = var_39.validate(var_40)
    var_42 = 1
    var_43 = 2
    var_44 = [var_42, var_43]
    var_45 = var_39.validate(var_44)
    var_46 = 1
    var_47 = 2
    var_48 = 3
    var_49 = 4
    var_50 = [var_46, var_47, var_48, var_49]
    var_51 = var_39.validate(var_50)
    var_52 = module_0.Integer()
    var_53 = module_0.Array(var_52)
    var_54 = [var_51, var_47, var_48]
    var_55 = var_53.validate(var_54)
    var_56 = 1
    var_57 = 'invalid'
    var_58 = 3
    var_59 = [var_56, var_57, var_58]
    var_60 = var_53.validate(var_59)
    var_61 = module_0.Integer()
    var_62 = module_0.String()
    var_63 = [var_61, var_62]
    var_64 = module_0.Array(var_63)
    var_65 = 'hello'
    var_66 = [var_51, var_65]
    var_67 = var_64.validate(var_66)
    var_68 = 'invalid'
    var_69 = 'hello'
    var_70 = [var_68, var_69]
    var_71 = var_64.validate(var_70)
    var_72 = module_0.Integer()
    var_73 = module_0.String()
    var_74 = [var_72, var_73]
    var_75 = module_0.Array(var_74, var_10)
    var_76 = [var_51, var_65]
    var_77 = var_75.validate(var_76)
    var_78 = 1
    var_79 = 'hello'
    var_80 = 'extra'
    var_81 = [var_78, var_79, var_80]
    var_82 = var_75.validate(var_81)
    var_83 = module_0.Integer()
    var_84 = module_0.String()
    var_85 = [var_83, var_84]
    var_86 = True
    var_87 = module_0.Array(var_85, var_86)
    var_88 = 'extra'
    var_89 = 4
    var_90 = [var_86, var_65, var_88, var_89]
    var_91 = var_87.validate(var_90)
    var_92 = module_0.Integer()
    var_93 = [var_92]
    var_94 = module_0.String()
    var_95 = module_0.Array(var_93, var_94)
    var_96 = 'world'
    var_97 = [var_86, var_65, var_96]
    var_98 = var_95.validate(var_97)
    var_99 = 1
    var_100 = 2
    var_101 = 3
    var_102 = [var_99, var_100, var_101]
    var_103 = var_95.validate(var_102)
    var_104 = True
    var_105 = module_0.Array(unique_items=var_104)
    var_106 = [var_104, var_100, var_101]
    var_107 = var_105.validate(var_106)
    var_108 = 1
    var_109 = 2
    var_110 = [var_108, var_109, var_108]
    var_111 = var_105.validate(var_110)
    var_112 = 'id'
    var_113 = 'name'
    var_114 = module_0.Integer()
    var_115 = module_0.String()
    var_116 = {var_112: var_114, var_113: var_115}
    var_117 = module_0.Object(properties=var_116)
    var_118 = module_0.Array(var_117)
    var_119 = 'Alice'
    var_120 = {var_112: var_104, var_113: var_119}
    var_121 = 'Bob'
    var_122 = {var_112: var_109, var_113: var_121}
    var_123 = [var_120, var_122]
    var_124 = var_118.validate(var_123)
    var_125 = 'id'
    var_126 = 'name'
    var_127 = 'invalid'
    var_128 = 'Alice'
    var_129 = {var_125: var_127, var_126: var_128}
    var_130 = [var_129]
    var_131 = var_118.validate(var_130)
    var_132 = module_0.Integer()
    var_133 = module_0.Array(var_132)
    var_134 = 'a'
    var_135 = 'b'
    var_136 = 'c'
    var_137 = [var_134, var_135, var_136]
    var_138 = var_133.validate(var_137)
    var_139 = module_0.Array()
    var_140 = 'string'
    var_141 = 'key'
    var_142 = 'value'
    var_143 = {var_141: var_142}
    var_144 = [var_104, var_135, var_136]
    var_145 = [var_104, var_140, var_143, var_144]
    var_146 = var_139.validate(var_145)



# Parsed testcases at query #16
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Integer()
    var_2 = module_0.String()
    var_3 = [var_1, var_2]
    var_4 = module_0.Union(var_3)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None
    var_7 = module_0.Integer()
    var_8 = module_0.String()
    var_9 = [var_7, var_8]
    var_10 = module_0.Union(var_9)
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = module_0.Integer()
    var_14 = module_0.String()
    var_15 = [var_13, var_14]
    var_16 = module_0.Union(var_15)
    var_17 = 42
    var_18 = var_16.validate(var_17)
    assert var_18 == 42
    var_19 = module_0.Integer()
    var_20 = module_0.String()
    var_21 = [var_19, var_20]
    var_22 = module_0.Union(var_21)
    var_23 = 'test'
    var_24 = var_22.validate(var_23)
    assert var_24 == 'test'
    var_25 = 10
    var_26 = module_0.Integer(minimum=var_25)
    var_27 = 5
    var_28 = module_0.String(min_length=var_27)
    var_29 = [var_26, var_28]
    var_30 = module_0.Union(var_29)
    var_31 = 5
    var_32 = var_30.validate(var_31)
    var_33 = module_0.Integer()
    var_34 = module_0.String()
    var_35 = [var_33, var_34]
    var_36 = module_0.Union(var_35)
    var_37 = True
    var_38 = var_36.validate(var_37)
    var_39 = module_0.Integer(minimum=var_25)
    var_40 = module_0.Integer(maximum=var_27)
    var_41 = [var_39, var_40]
    var_42 = module_0.Union(var_41)
    var_43 = 7
    var_44 = var_42.validate(var_43)
    var_45 = module_0.Integer()
    var_46 = module_0.Array(var_45)
    var_47 = 'value'
    var_48 = module_0.Integer()
    var_49 = {var_47: var_48}
    var_50 = module_0.Object(properties=var_49)
    var_51 = [var_46, var_50]
    var_52 = module_0.Union(var_51)
    var_53 = 2
    var_54 = 3
    var_55 = [var_43, var_53, var_54]
    var_56 = var_52.validate(var_55)
    var_57 = {var_47: var_17}
    var_58 = var_52.validate(var_57)
    var_59 = module_0.Integer()
    var_60 = module_0.String()
    var_61 = [var_59, var_60]
    var_62 = module_0.Union(var_61)
    var_63 = var_62.validate(var_5)
    assert var_63 is None
    var_64 = module_0.Decimal()
    var_65 = module_0.String()
    var_66 = [var_64, var_65]
    var_67 = module_0.Union(var_66)
    var_68 = '123.45'
    var_69 = var_67.validate(var_68)



# Parsed testcases at query #17
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
    var_21 = None
    var_22 = var_20.validate(var_21)
    var_23 = (var_21, var_22)
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
    assert var_40 is None
    var_41 = (var_33, var_34)
    var_42 = (var_3, var_4)
    var_43 = [var_41, var_42]
    var_44 = False
    var_45 = module_0.Choice(choices=var_43, coerce_types=var_44)
    var_46 = ''
    var_47 = var_45.validate(var_46)
    var_48 = 'One'
    var_49 = (var_13, var_48)
    var_50 = 2
    var_51 = 'Two'
    var_52 = (var_50, var_51)
    var_53 = [var_49, var_52]
    var_54 = module_0.Choice(choices=var_53)
    var_55 = var_54.validate(var_13)
    assert var_55 == 1
    var_56 = var_54.validate(var_50)
    assert var_56 == 2
    var_57 = '1'
    var_58 = 'String One'
    var_59 = (var_57, var_58)
    var_60 = 'Number Two'
    var_61 = (var_50, var_60)
    var_62 = [var_59, var_61]
    var_63 = module_0.Choice(choices=var_62)
    var_64 = var_63.validate(var_57)
    assert var_64 == '1'
    var_65 = var_63.validate(var_50)
    assert var_65 == 2
    var_66 = []
    var_67 = module_0.Choice(choices=var_66)
    var_68 = 'anything'
    var_69 = var_67.validate(var_68)
    var_70 = module_0.Choice()
    var_71 = 'anything'
    var_72 = var_70.validate(var_71)



# Parsed testcases at query #18
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
    var_10 = 'A'
    var_11 = (var_0, var_10)
    var_12 = 'B'
    var_13 = (var_3, var_12)
    var_14 = [var_11, var_13]
    var_15 = True
    var_16 = module_0.Choice(choices=var_14)
    var_17 = None
    var_18 = var_16.validate(var_17)
    assert var_18 is None
    var_19 = (var_0, var_10)
    var_20 = (var_3, var_12)
    var_21 = [var_19, var_20]
    var_22 = module_0.Choice(choices=var_21)
    var_23 = None
    var_24 = var_22.validate(var_23)
    var_25 = (var_23, var_10)
    var_26 = (var_3, var_12)
    var_27 = [var_25, var_26]
    var_28 = module_0.Choice(choices=var_27)
    var_29 = 'c'
    var_30 = var_28.validate(var_29)
    var_31 = (var_29, var_10)
    var_32 = (var_3, var_12)
    var_33 = [var_31, var_32]
    var_34 = module_0.Choice(choices=var_33)
    var_35 = ''
    var_36 = var_34.validate(var_35)
    var_37 = (var_35, var_10)
    var_38 = (var_3, var_12)
    var_39 = [var_37, var_38]
    var_40 = module_0.Choice(choices=var_39)
    var_41 = ''
    var_42 = var_40.validate(var_41)
    assert var_42 is None
    var_43 = (var_35, var_10)
    var_44 = (var_3, var_12)
    var_45 = [var_43, var_44]
    var_46 = False
    var_47 = module_0.Choice(choices=var_45, coerce_types=var_46)
    var_48 = ''
    var_49 = var_47.validate(var_48)
    var_50 = [var_48, var_3]
    var_51 = module_0.Choice(choices=var_50)
    var_52 = var_51.validate(var_48)
    assert var_52 == 'a'
    var_53 = var_51.validate(var_3)
    assert var_53 == 'b'
    var_54 = (var_3, var_4)
    var_55 = [var_48, var_54]
    var_56 = module_0.Choice(choices=var_55)
    var_57 = var_56.validate(var_48)
    assert var_57 == 'a'
    var_58 = var_56.validate(var_3)
    assert var_58 == 'b'
    var_59 = (var_48, var_10)
    var_60 = (var_3, var_12)
    var_61 = 'c'
    var_62 = 'C'
    var_63 = (var_61, var_62)
    var_64 = [var_59, var_60, var_63]
    var_65 = module_0.Choice(choices=var_64)
    var_66 = var_65.validate(var_61)
    assert var_66 == 'c'
    var_67 = var_65.choices
    var_68 = 2



# Parsed testcases at query #19
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'default'
    var_2 = hasattr(var_0, var_1)
    var_3 = var_0.get_default_value()
    assert var_3 is None
    var_4 = 'test_value'
    var_5 = module_0.Field(default=var_4)
    var_6 = var_5.get_default_value()
    assert var_6 == 'test_value'
    var_7 = 42
    var_8 = module_0.Field(default=var_7)
    var_9 = var_8.get_default_value()
    assert var_9 == 42
    var_10 = 1
    var_11 = 2
    var_12 = 3
    var_13 = [var_10, var_11, var_12]
    var_14 = module_0.Field(default=var_13)
    var_15 = var_14.get_default_value()
    var_16 = 100
    var_17 = lambda : var_16
    var_18 = module_0.Field(default=var_17)
    var_19 = var_18.get_default_value()
    assert var_19 == 100
    var_20 = None
    var_21 = lambda : var_20
    var_22 = module_0.Field(default=var_21)
    var_23 = var_22.get_default_value()
    assert var_23 is None
    var_24 = 'key'
    var_25 = 'value'
    var_26 = {var_24: var_25}
    var_27 = lambda : var_26
    var_28 = module_0.Field(default=var_27)
    var_29 = var_28.get_default_value()
    var_30 = 'called'
    var_31 = lambda : var_30
    var_32 = module_0.Field(default=var_31)
    var_33 = var_32.get_default_value()
    assert var_33 == 'called'
    var_34 = callable(var_33)
    var_35 = True
    var_36 = module_0.Field(allow_null=var_35)
    var_37 = var_36.get_default_value()
    assert var_37 is None
    var_38 = True
    var_39 = 'explicit'
    var_40 = module_0.Field(default=var_39, allow_null=var_38)
    var_41 = var_40.get_default_value()
    assert var_41 == 'explicit'



# Parsed testcases at query #20
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = var_0.has_default()
    var_2 = 'test_value'
    var_3 = module_0.Field(default=var_2)
    var_4 = var_3.has_default()
    var_5 = var_3.get_default_value()
    assert var_5 == 'test_value'
    var_6 = 0
    var_7 = None
    var_8 = module_0.Field(default=var_7)
    var_9 = var_8.has_default()
    var_10 = var_8.get_default_value()
    assert var_10 is None
    var_11 = True
    var_12 = module_0.Field(allow_null=var_11)
    var_13 = var_12.has_default()
    var_14 = var_12.get_default_value()
    assert var_14 is None
    var_15 = 'explicit'
    var_16 = module_0.Field(default=var_15, allow_null=var_11)
    var_17 = var_16.has_default()
    var_18 = var_16.get_default_value()
    assert var_18 == 'explicit'



# Parsed testcases at query #21
#--------------------------


import typesystem.fields as module_0
import typesystem.formats as module_1

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
    var_11 = 123
    var_12 = var_8.validate(var_11)
    var_13 = module_0.String(allow_blank=var_3)
    var_14 = ''
    var_15 = var_13.validate(var_14)
    assert var_15 == ''
    var_16 = module_0.String(allow_blank=var_7)
    var_17 = ''
    var_18 = var_16.validate(var_17)
    var_19 = module_0.String(trim_whitespace=var_3)
    var_20 = '  hello  '
    var_21 = var_19.validate(var_20)
    assert var_21 == 'hello'
    var_22 = module_0.String(trim_whitespace=var_7)
    var_23 = var_22.validate(var_20)
    assert var_23 == '  hello  '
    var_24 = 3
    var_25 = module_0.String(min_length=var_24)
    var_26 = 'abc'
    var_27 = var_25.validate(var_26)
    assert var_27 == 'abc'
    var_28 = 'ab'
    var_29 = var_25.validate(var_28)
    var_30 = 5
    var_31 = module_0.String(max_length=var_30)
    var_32 = 'abcde'
    var_33 = var_31.validate(var_32)
    assert var_33 == 'abcde'
    var_34 = 'abcdef'
    var_35 = var_31.validate(var_34)
    var_36 = '^\\d+$'
    var_37 = module_0.String(pattern=var_36)
    var_38 = '123'
    var_39 = var_37.validate(var_38)
    assert var_39 == '123'
    var_40 = 'abc'
    var_41 = var_37.validate(var_40)
    var_42 = 'email'
    var_43 = module_0.String(format=var_42)
    var_44 = 'test@example.com'
    var_45 = var_43.validate(var_44)
    assert var_45 == 'test@example.com'
    var_46 = 'not-an-email'
    var_47 = var_43.validate(var_46)
    var_48 = module_0.String()
    var_49 = 'hello\x00world'
    var_50 = var_48.validate(var_49)
    assert var_50 == 'helloworld'
    var_51 = module_0.String(allow_blank=var_3, coerce_types=var_3)
    var_52 = var_51.validate(var_5)
    assert var_52 == ''
    var_53 = module_0.String(trim_whitespace=var_3, coerce_types=var_3)
    var_54 = var_53.validate(var_14)
    assert var_54 is None
    var_55 = 'uuid'
    var_56 = module_0.String(format=var_55)
    var_57 = module_1.UUIDFormat()
    var_58 = '12345678-1234-5678-1234-567812345678'
    var_59 = var_57.validate(var_58)
    var_60 = var_56.validate(var_59)
    var_61 = 2
    var_62 = 4
    var_63 = '^[a-z]+$'
    var_64 = module_0.String(max_length=var_62, min_length=var_61, pattern=var_63)
    var_65 = var_64.validate(var_26)
    assert var_65 == 'abc'
    var_66 = 'a'
    var_67 = var_64.validate(var_66)
    var_68 = 'abcde'
    var_69 = var_64.validate(var_68)
    var_70 = '123'
    var_71 = var_64.validate(var_70)



# Parsed testcases at query #22
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
    var_18 = False
    var_19 = module_0.Choice(choices=var_17)
    var_20 = None
    var_21 = var_19.validate(var_20)
    var_22 = (var_20, var_21)
    var_23 = [var_22]
    var_24 = module_0.Choice(choices=var_23, coerce_types=var_12)
    var_25 = ''
    var_26 = var_24.validate(var_25)
    assert var_26 is None
    var_27 = (var_20, var_21)
    var_28 = [var_27]
    var_29 = module_0.Choice(choices=var_28)
    var_30 = ''
    var_31 = var_29.validate(var_30)
    var_32 = (var_30, var_31)
    var_33 = (var_3, var_4)
    var_34 = [var_32, var_33]
    var_35 = module_0.Choice(choices=var_34)
    var_36 = 'c'
    var_37 = var_35.validate(var_36)
    var_38 = 'key1'
    var_39 = 'Value 1'
    var_40 = (var_38, var_39)
    var_41 = 'key2'
    var_42 = 'Value 2'
    var_43 = (var_41, var_42)
    var_44 = [var_40, var_43]
    var_45 = module_0.Choice(choices=var_44)
    var_46 = var_45.validate(var_38)
    assert var_46 == 'key1'
    var_47 = var_45.validate(var_41)
    assert var_47 == 'key2'
    var_48 = [var_38, var_39]
    var_49 = [var_41, var_42]
    var_50 = [var_48, var_49]
    var_51 = module_0.Choice(choices=var_50)
    var_52 = var_51.validate(var_38)
    assert var_52 == 'key1'
    var_53 = 'option1'
    var_54 = 'option2'
    var_55 = [var_53, var_54]
    var_56 = module_0.Choice(choices=var_55)
    var_57 = var_56.validate(var_53)
    assert var_57 == 'option1'
    var_58 = var_56.validate(var_54)
    assert var_58 == 'option2'
    var_59 = 'One'
    var_60 = (var_12, var_59)
    var_61 = 2
    var_62 = 'Two'
    var_63 = (var_61, var_62)
    var_64 = [var_60, var_63]
    var_65 = module_0.Choice(choices=var_64)
    var_66 = var_65.validate(var_12)
    assert var_66 == 1
    var_67 = var_65.validate(var_61)
    assert var_67 == 2
    var_68 = 'Yes'
    var_69 = (var_12, var_68)
    var_70 = 'No'
    var_71 = (var_18, var_70)
    var_72 = [var_69, var_71]
    var_73 = module_0.Choice(choices=var_72)
    var_74 = var_73.validate(var_12)
    assert var_74 is True
    var_75 = var_73.validate(var_18)
    assert var_75 is False



# Parsed testcases at query #23
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = module_0.Const(var_0)
    var_2 = var_1.validate(var_0)
    assert var_2 == 'test_value'
    var_3 = None
    var_4 = module_0.Const(var_3)
    var_5 = var_4.validate(var_3)
    assert var_5 is None
    var_6 = 42
    var_7 = module_0.Const(var_6)
    var_8 = var_7.validate(var_6)
    assert var_8 == 42
    var_9 = 3.14
    var_10 = module_0.Const(var_9)
    var_11 = var_10.validate(var_9)
    var_12 = True
    var_13 = module_0.Const(var_12)
    var_14 = var_13.validate(var_12)
    assert var_14 is True
    var_15 = 2
    var_16 = 3
    var_17 = [var_12, var_15, var_16]
    var_18 = module_0.Const(var_17)
    var_19 = [var_12, var_15, var_16]
    var_20 = var_1.validate(var_19)
    var_21 = 'key'
    var_22 = 'value'
    var_23 = {var_21: var_22}
    var_24 = module_0.Const(var_23)
    var_25 = {var_21: var_22}
    var_26 = var_1.validate(var_25)
    var_27 = 'expected'
    var_28 = module_0.Const(var_27)
    var_29 = 'wrong'
    var_30 = var_28.validate(var_29)
    var_31 = module_0.Const(var_3)
    var_32 = 'not_none'
    var_33 = var_31.validate(var_32)
    var_34 = module_0.Const(var_27)
    var_35 = None
    var_36 = var_34.validate(var_35)



# Parsed testcases at query #24
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Array()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = var_0.validate(var_4)
    var_6 = []
    var_7 = var_0.validate(var_6)
    var_8 = True
    var_9 = module_0.Array()
    var_10 = None
    var_11 = var_9.validate(var_10)
    assert var_11 is None
    var_12 = False
    var_13 = module_0.Array()
    var_14 = None
    var_15 = var_13.validate(var_14)
    var_16 = module_0.Array()
    var_17 = 'not a list'
    var_18 = var_16.validate(var_17)
    var_19 = module_0.Array(min_items=var_18)
    var_20 = [var_8, var_18]
    var_21 = var_19.validate(var_20)
    var_22 = [var_8, var_18, var_3]
    var_23 = var_19.validate(var_22)
    var_24 = 1
    var_25 = [var_24]
    var_26 = var_19.validate(var_25)
    var_27 = module_0.Array(min_items=var_8)
    var_28 = []
    var_29 = var_27.validate(var_28)
    var_30 = module_0.Array(max_items=var_26)
    var_31 = [var_8, var_29, var_26]
    var_32 = var_30.validate(var_31)
    var_33 = [var_8, var_29]
    var_34 = var_30.validate(var_33)
    var_35 = 1
    var_36 = 2
    var_37 = 3
    var_38 = 4
    var_39 = [var_35, var_36, var_37, var_38]
    var_40 = var_30.validate(var_39)
    var_41 = module_0.Array(exact_items=var_37)
    var_42 = [var_8, var_36, var_37]
    var_43 = var_41.validate(var_42)
    var_44 = 1
    var_45 = 2
    var_46 = [var_44, var_45]
    var_47 = var_41.validate(var_46)
    var_48 = 1
    var_49 = 2
    var_50 = 3
    var_51 = 4
    var_52 = [var_48, var_49, var_50, var_51]
    var_53 = var_41.validate(var_52)
    var_54 = module_0.Integer()
    var_55 = module_0.Array(var_54)
    var_56 = [var_8, var_49, var_50]
    var_57 = var_55.validate(var_56)
    var_58 = 1
    var_59 = 'invalid'
    var_60 = 3
    var_61 = [var_58, var_59, var_60]
    var_62 = var_55.validate(var_61)
    var_63 = module_0.Integer()
    var_64 = module_0.String()
    var_65 = module_0.Boolean()
    var_66 = [var_63, var_64, var_65]
    var_67 = module_0.Array(var_66)
    var_68 = 'test'
    var_69 = True
    var_70 = [var_8, var_68, var_69]
    var_71 = var_67.validate(var_70)
    var_72 = 1
    var_73 = 'test'
    var_74 = 'not boolean'
    var_75 = [var_72, var_73, var_74]
    var_76 = var_67.validate(var_75)
    var_77 = module_0.Integer()
    var_78 = module_0.String()
    var_79 = [var_77, var_78]
    var_80 = module_0.Array(var_79, var_12)
    var_81 = [var_69, var_68]
    var_82 = var_80.validate(var_81)
    var_83 = 1
    var_84 = 'test'
    var_85 = 'extra'
    var_86 = [var_83, var_84, var_85]
    var_87 = var_80.validate(var_86)
    var_88 = module_0.Integer()
    var_89 = module_0.String()
    var_90 = [var_88, var_89]
    var_91 = module_0.Boolean()
    var_92 = module_0.Array(var_90, var_91)
    var_93 = True
    var_94 = [var_69, var_68, var_93, var_12]
    var_95 = var_92.validate(var_94)
    var_96 = 1
    var_97 = 'test'
    var_98 = 'not boolean'
    var_99 = [var_96, var_97, var_98]
    var_100 = var_92.validate(var_99)
    var_101 = True
    var_102 = module_0.Array(unique_items=var_101)
    var_103 = [var_101, var_97, var_98]
    var_104 = var_102.validate(var_103)
    var_105 = 1
    var_106 = 2
    var_107 = [var_105, var_106, var_105]
    var_108 = var_102.validate(var_107)
    var_109 = module_0.Integer()
    var_110 = module_0.Array(var_109)
    var_111 = module_0.Array(var_110)
    var_112 = [var_101, var_106]
    var_113 = 4
    var_114 = [var_107, var_113]
    var_115 = [var_112, var_114]
    var_116 = var_111.validate(var_115)
    var_117 = 1
    var_118 = 2
    var_119 = [var_117, var_118]
    var_120 = 'invalid'
    var_121 = 4
    var_122 = [var_120, var_121]
    var_123 = [var_119, var_122]
    var_124 = var_111.validate(var_123)
    var_125 = module_0.Integer()
    var_126 = module_0.Integer()
    var_127 = [var_125, var_126]
    var_128 = module_0.Array(var_127)
    var_129 = 'invalid1'
    var_130 = 'invalid2'
    var_131 = [var_129, var_130]
    var_132 = var_128.validate(var_131)



# Parsed testcases at query #25
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
    var_29 = 'null'
    var_30 = var_28.validate(var_29)
    assert var_30 is None
    var_31 = 'none'
    var_32 = var_28.validate(var_31)
    assert var_32 is None
    var_33 = var_28.validate(var_24)
    assert var_33 is None
    var_34 = module_0.Boolean(coerce_types=var_3)
    var_35 = var_34.validate(var_9)
    assert var_35 is True
    var_36 = var_34.validate(var_3)
    assert var_36 is False
    var_37 = 'true'
    var_38 = var_34.validate(var_37)
    var_39 = 1
    var_40 = var_34.validate(var_39)
    var_41 = module_0.Boolean(coerce_types=var_39)
    var_42 = 'invalid'
    var_43 = var_41.validate(var_42)
    var_44 = 2
    var_45 = var_41.validate(var_44)
    var_46 = module_0.Boolean(coerce_types=var_44)
    var_47 = 'TRUE'
    var_48 = var_46.validate(var_47)
    assert var_48 is True
    var_49 = 'FALSE'
    var_50 = var_46.validate(var_49)
    assert var_50 is False
    var_51 = 'On'
    var_52 = var_46.validate(var_51)
    assert var_52 is True
    var_53 = 'Off'
    var_54 = var_46.validate(var_53)
    assert var_54 is False
    var_55 = module_0.Boolean()
    var_56 = 'invalid'



# Parsed testcases at query #26
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Integer()
    var_2 = module_0.String()
    var_3 = [var_1, var_2]
    var_4 = module_0.Union(var_3)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None
    var_7 = module_0.Integer()
    var_8 = module_0.String()
    var_9 = [var_7, var_8]
    var_10 = module_0.Union(var_9)
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = module_0.Integer()
    var_14 = module_0.String()
    var_15 = [var_13, var_14]
    var_16 = module_0.Union(var_15)
    var_17 = 42
    var_18 = var_16.validate(var_17)
    assert var_18 == 42
    var_19 = 'hello'
    var_20 = var_16.validate(var_19)
    assert var_20 == 'hello'
    var_21 = module_0.Integer()
    var_22 = module_0.String()
    var_23 = [var_21, var_22]
    var_24 = module_0.Union(var_23)
    var_25 = 3.14
    var_26 = var_24.validate(var_25)
    var_27 = 10
    var_28 = module_0.Integer(minimum=var_27)
    var_29 = 5
    var_30 = module_0.String(min_length=var_29)
    var_31 = [var_28, var_30]
    var_32 = module_0.Union(var_31)
    var_33 = 5
    var_34 = var_32.validate(var_33)
    var_35 = module_0.Integer(minimum=var_27)
    var_36 = module_0.String(min_length=var_29)
    var_37 = [var_35, var_36]
    var_38 = module_0.Union(var_37)
    var_39 = 7
    var_40 = var_38.validate(var_39)
    var_41 = module_0.Integer()
    var_42 = module_0.String()
    var_43 = [var_41, var_42]
    var_44 = module_0.Union(var_43)
    var_45 = var_44.validate(var_5)
    assert var_45 is None
    var_46 = module_0.Integer()
    var_47 = module_0.Array(var_46)
    var_48 = 'value'
    var_49 = module_0.Integer()
    var_50 = {var_48: var_49}
    var_51 = module_0.Object(properties=var_50)
    var_52 = [var_47, var_51]
    var_53 = module_0.Union(var_52)
    var_54 = 2
    var_55 = 3
    var_56 = [var_39, var_54, var_55]
    var_57 = var_53.validate(var_56)
    var_58 = {var_48: var_17}
    var_59 = var_53.validate(var_58)
    var_60 = module_0.Integer(coerce_types=var_39)
    var_61 = module_0.Boolean(coerce_types=var_39)
    var_62 = [var_60, var_61]
    var_63 = module_0.Union(var_62)
    var_64 = '42'
    var_65 = var_63.validate(var_64)
    assert var_65 == 42
    var_66 = 'true'
    var_67 = var_63.validate(var_66)
    assert var_67 is True
    var_68 = module_0.Decimal()
    var_69 = module_0.Float()
    var_70 = [var_68, var_69]
    var_71 = module_0.Union(var_70)
    var_72 = '3.14'
    var_73 = 3.14
    var_74 = var_71.validate(var_73)



# Parsed testcases at query #27
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Const(var_0)
    var_2 = 'test'
    var_3 = module_0.Const(var_2)
    var_4 = None
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
    var_16 = 'default'
    var_17 = module_0.Const(var_16)



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = None
    var_3 = None
    var_4 = 'valid'
    var_5 = 'test'
    var_6 = 'type'
    var_7 = 'test'
    var_8 = 'minimum'
    var_9 = 'test'
    var_10 = 'nested'
    var_11 = [var_10]
    var_12 = 'test'
    var_13 = 'maximum'
    var_14 = 'test'



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
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
    var_11 = module_0.Boolean(coerce_types=var_3)
    var_12 = 'true'
    var_13 = var_11.validate(var_12)
    var_14 = module_0.Boolean(coerce_types=var_12)
    var_15 = 'true'
    var_16 = var_14.validate(var_15)
    assert var_16 is True
    var_17 = 'false'
    var_18 = var_14.validate(var_17)
    assert var_18 is False
    var_19 = 'on'
    var_20 = var_14.validate(var_19)
    assert var_20 is True
    var_21 = 'off'
    var_22 = var_14.validate(var_21)
    assert var_22 is False
    var_23 = '1'
    var_24 = var_14.validate(var_23)
    assert var_24 is True
    var_25 = '0'
    var_26 = var_14.validate(var_25)
    assert var_26 is False
    var_27 = ''
    var_28 = var_14.validate(var_27)
    assert var_28 is False
    var_29 = var_14.validate(var_12)
    assert var_29 is True
    var_30 = var_14.validate(var_3)
    assert var_30 is False
    var_31 = 'TRUE'
    var_32 = var_14.validate(var_31)
    assert var_32 is True
    var_33 = 'FALSE'
    var_34 = var_14.validate(var_33)
    assert var_34 is False
    var_35 = 'ON'
    var_36 = var_14.validate(var_35)
    assert var_36 is True
    var_37 = 'OFF'
    var_38 = var_14.validate(var_37)
    assert var_38 is False
    var_39 = module_0.Boolean(coerce_types=var_12)
    var_40 = 'null'
    var_41 = var_39.validate(var_40)
    assert var_41 is None
    var_42 = 'none'
    var_43 = var_39.validate(var_42)
    assert var_43 is None
    var_44 = var_39.validate(var_27)
    assert var_44 is None
    var_45 = 'invalid'
    var_46 = var_14.validate(var_45)
    var_47 = []
    var_48 = var_14.validate(var_47)



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
    var_21 = None
    var_22 = var_20.validate(var_21)
    var_23 = (var_21, var_22)
    var_24 = (var_3, var_4)
    var_25 = [var_23, var_24]
    var_26 = module_0.Choice(choices=var_25)
    var_27 = 'c'
    var_28 = var_26.validate(var_27)
    var_29 = (var_27, var_28)
    var_30 = (var_3, var_4)
    var_31 = [var_29, var_30]
    var_32 = module_0.Choice(choices=var_31, coerce_types=var_13)
    var_33 = ''
    var_34 = var_32.validate(var_33)
    var_35 = (var_33, var_34)
    var_36 = (var_3, var_4)
    var_37 = [var_35, var_36]
    var_38 = module_0.Choice(choices=var_37, coerce_types=var_13)
    var_39 = ''
    var_40 = var_38.validate(var_39)
    assert var_40 is None
    var_41 = (var_33, var_34)
    var_42 = (var_3, var_4)
    var_43 = [var_41, var_42]
    var_44 = False
    var_45 = module_0.Choice(choices=var_43, coerce_types=var_13)
    var_46 = ''
    var_47 = var_45.validate(var_46)
    var_48 = 'One'
    var_49 = (var_13, var_48)
    var_50 = 2
    var_51 = 'Two'
    var_52 = (var_50, var_51)
    var_53 = [var_49, var_52]
    var_54 = module_0.Choice(choices=var_53)
    var_55 = var_54.validate(var_13)
    assert var_55 == 1
    var_56 = var_54.validate(var_50)
    assert var_56 == 2
    var_57 = '1'
    var_58 = 'String One'
    var_59 = (var_57, var_58)
    var_60 = 'Number Two'
    var_61 = (var_50, var_60)
    var_62 = [var_59, var_61]
    var_63 = module_0.Choice(choices=var_62)
    var_64 = var_63.validate(var_57)
    assert var_64 == '1'
    var_65 = var_63.validate(var_50)
    assert var_65 == 2
    var_66 = 'Yes'
    var_67 = (var_13, var_66)
    var_68 = 'No'
    var_69 = (var_44, var_68)
    var_70 = [var_67, var_69]
    var_71 = module_0.Choice(choices=var_70)
    var_72 = var_71.validate(var_13)
    assert var_72 is True
    var_73 = var_71.validate(var_44)
    assert var_73 is False
    var_74 = 'key1'
    var_75 = 'Display 1'
    var_76 = (var_74, var_75)
    var_77 = 'key2'
    var_78 = 'Display 2'
    var_79 = (var_77, var_78)
    var_80 = [var_76, var_79]
    var_81 = module_0.Choice(choices=var_80)
    var_82 = var_81.validate(var_74)
    assert var_82 == 'key1'
    var_83 = var_81.validate(var_77)
    assert var_83 == 'key2'
    var_84 = [var_74, var_75]
    var_85 = [var_77, var_78]
    var_86 = [var_84, var_85]
    var_87 = module_0.Choice(choices=var_86)
    var_88 = var_87.validate(var_74)
    assert var_88 == 'key1'
    var_89 = var_87.validate(var_77)
    assert var_89 == 'key2'



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
    var_22 = module_0.String()
    var_23 = {var_20: var_22}
    var_24 = module_0.Object(properties=var_23)
    var_25 = 'not an object'
    var_26 = var_24.validate(var_25)
    var_27 = module_0.String()
    var_28 = {var_25: var_27}
    var_29 = [var_25]
    var_30 = module_0.Object(properties=var_28, required=var_29)
    var_31 = {var_25: var_6}
    var_32 = var_30.validate(var_31)
    var_33 = 'age'
    var_34 = 30
    var_35 = {var_33: var_34}
    var_36 = var_30.validate(var_35)
    var_37 = 'Unknown'
    var_38 = module_0.String()
    var_39 = {var_33: var_38}
    var_40 = module_0.Object(properties=var_39)
    var_41 = {}
    var_42 = var_40.validate(var_41)
    var_43 = 'a'
    var_44 = 'b'
    var_45 = module_0.String()
    var_46 = module_0.String()
    var_47 = {var_43: var_45, var_44: var_46}
    var_48 = module_0.Object(properties=var_47, min_properties=var_12)
    var_49 = 'test'
    var_50 = {var_43: var_49}
    var_51 = var_48.validate(var_50)
    var_52 = {}
    var_53 = var_48.validate(var_52)
    var_54 = 'c'
    var_55 = module_0.String()
    var_56 = module_0.String()
    var_57 = module_0.String()
    var_58 = {var_43: var_55, var_44: var_56, var_54: var_57}
    var_59 = 2
    var_60 = module_0.Object(properties=var_58, min_properties=var_59)
    var_61 = 'a'
    var_62 = 'test'
    var_63 = {var_61: var_62}
    var_64 = var_60.validate(var_63)
    var_65 = module_0.String()
    var_66 = module_0.String()
    var_67 = {var_43: var_65, var_44: var_66}
    var_68 = module_0.Object(properties=var_67, max_properties=var_59)
    var_69 = 'test2'
    var_70 = {var_43: var_49, var_44: var_69}
    var_71 = var_68.validate(var_70)
    var_72 = 'a'
    var_73 = 'b'
    var_74 = 'c'
    var_75 = 'test'
    var_76 = 'test2'
    var_77 = 'test3'
    var_78 = {var_72: var_75, var_73: var_76, var_74: var_77}
    var_79 = var_68.validate(var_78)
    var_80 = '^test_'
    var_81 = module_0.String()
    var_82 = {var_80: var_81}
    var_83 = module_0.Object(pattern_properties=var_82)
    var_84 = 'test_key'
    var_85 = 'value'
    var_86 = {var_84: var_85}
    var_87 = var_83.validate(var_86)
    var_88 = module_0.String()
    var_89 = {var_72: var_88}
    var_90 = module_0.Object(properties=var_89, additional_properties=var_12)
    var_91 = 'extra'
    var_92 = {var_72: var_77, var_91: var_85}
    var_93 = var_90.validate(var_92)
    var_94 = module_0.String()
    var_95 = {var_72: var_94}
    var_96 = module_0.Object(properties=var_95, additional_properties=var_18)
    var_97 = {var_72: var_77}
    var_98 = var_96.validate(var_97)
    var_99 = 'name'
    var_100 = 'extra'
    var_101 = 'John'
    var_102 = 'value'
    var_103 = {var_99: var_101, var_100: var_102}
    var_104 = var_96.validate(var_103)
    var_105 = module_0.String()
    var_106 = {var_99: var_105}
    var_107 = module_0.Integer()
    var_108 = module_0.Object(properties=var_106, additional_properties=var_107)
    var_109 = 42
    var_110 = {var_99: var_104, var_91: var_109}
    var_111 = var_108.validate(var_110)
    var_112 = 'name'
    var_113 = 'extra'
    var_114 = 'John'
    var_115 = 'not a number'
    var_116 = {var_112: var_114, var_113: var_115}
    var_117 = var_108.validate(var_116)
    var_118 = module_0.String()
    var_119 = {var_112: var_118}
    var_120 = '^[a-z]+$'
    var_121 = module_0.String(pattern=var_120)
    var_122 = module_0.Object(properties=var_119, property_names=var_121)
    var_123 = {var_112: var_117, var_113: var_85}
    var_124 = var_122.validate(var_123)
    var_125 = 'Name'
    var_126 = 'John'
    var_127 = {var_125: var_126}
    var_128 = var_122.validate(var_127)
    var_129 = module_0.String()
    var_130 = {var_125: var_129}
    var_131 = module_0.Object(properties=var_130)
    var_132 = 123
    var_133 = 'value'
    var_134 = {var_132: var_133}
    var_135 = var_131.validate(var_134)
    var_136 = 'person'
    var_137 = module_0.String()
    var_138 = module_0.Integer(minimum=var_18)
    var_139 = {var_132: var_137, var_133: var_138}
    var_140 = module_0.Object(properties=var_139)
    var_141 = {var_136: var_140}
    var_142 = module_0.Object(properties=var_141)
    var_143 = 'person'
    var_144 = 'age'
    var_145 = -5
    var_146 = {var_144: var_145}
    var_147 = {var_143: var_146}
    var_148 = var_142.validate(var_147)
    var_149 = 'username'
    var_150 = 'email'
    var_151 = 3
    var_152 = 20
    var_153 = module_0.String(max_length=var_152, min_length=var_151)
    var_154 = module_0.String()
    var_155 = {var_149: var_153, var_150: var_154}
    var_156 = '^meta_'
    var_157 = module_0.String()
    var_158 = {var_156: var_157}
    var_159 = [var_149]
    var_160 = module_0.Object(properties=var_155, pattern_properties=var_158, additional_properties=var_18, min_properties=var_12, max_properties=var_151, required=var_159)
    var_161 = 'meta_info'
    var_162 = 'johndoe'
    var_163 = 'some info'
    var_164 = {var_149: var_162, var_161: var_163}
    var_165 = var_160.validate(var_164)
    var_166 = 'username'
    var_167 = 'email'
    var_168 = 'meta_1'
    var_169 = 'meta_2'
    var_170 = 'jd'
    var_171 = 'test@test.com'
    var_172 = 'a'
    var_173 = 'b'
    var_174 = {var_166: var_170, var_167: var_171, var_168: var_172, var_169: var_173}
    var_175 = var_160.validate(var_174)



# Parsed testcases at query #4
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = True
    var_2 = module_0.Array(var_0)
    var_3 = None
    var_4 = var_2.serialize(var_3)
    assert var_4 is None
    var_5 = module_0.String()
    var_6 = False
    var_7 = module_0.Array(var_5)
    var_8 = var_7.serialize(var_3)
    assert var_8 is None
    var_9 = module_0.Integer()
    var_10 = module_0.Array(var_9)
    var_11 = 2
    var_12 = 3
    var_13 = [var_1, var_11, var_12]
    var_14 = var_10.serialize(var_13)
    var_15 = module_0.Integer()
    var_16 = module_0.String()
    var_17 = module_0.Boolean()
    var_18 = [var_15, var_16, var_17]
    var_19 = module_0.Array(var_18)
    var_20 = 42
    var_21 = 'hello'
    var_22 = [var_20, var_21, var_1]
    var_23 = var_19.serialize(var_22)
    var_24 = module_0.Array(var_3)
    var_25 = 'test'
    var_26 = 'key'
    var_27 = 'value'
    var_28 = {var_26: var_27}
    var_29 = [var_1, var_25, var_28]
    var_30 = var_24.serialize(var_29)
    var_31 = 'id'
    var_32 = 'name'
    var_33 = module_0.Integer()
    var_34 = module_0.String()
    var_35 = {var_31: var_33, var_32: var_34}
    var_36 = module_0.Object(properties=var_35)
    var_37 = module_0.Array(var_36)
    var_38 = 'Alice'
    var_39 = {var_31: var_1, var_32: var_38}
    var_40 = 'Bob'
    var_41 = {var_31: var_11, var_32: var_40}
    var_42 = [var_39, var_41]
    var_43 = var_37.serialize(var_42)
    var_44 = module_0.String()
    var_45 = module_0.Array(var_44)
    var_46 = []
    var_47 = var_45.serialize(var_46)
    var_48 = module_0.Decimal()
    var_49 = module_0.Array(var_48)
    var_50 = '1.5'
    var_51 = '2.75'
    var_52 = var_49.serialize(var_46)
    var_53 = module_0.Array(var_3)
    var_54 = 'string'
    var_55 = {var_26: var_27}
    var_56 = [var_1, var_54, var_1, var_3, var_55]
    var_57 = var_53.serialize(var_56)
    var_58 = module_0.Integer()
    var_59 = module_0.String()
    var_60 = [var_58, var_59]
    var_61 = module_0.Array(var_60)
    var_62 = 'extra'
    var_63 = 'items'
    var_64 = [var_1, var_21, var_62, var_63]
    var_65 = var_61.serialize(var_64)



# Parsed testcases at query #5
#--------------------------


import typesystem.fields as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = True
    var_2 = False
    var_3 = 100
    var_4 = 5
    var_5 = '^\\d+$'
    var_6 = 'email'
    var_7 = 'Test Title'
    var_8 = 'Test Description'
    var_9 = module_0.String(allow_blank=var_1, trim_whitespace=var_2, max_length=var_3, min_length=var_4, pattern=var_5, format=var_6, coerce_types=var_2)
    var_10 = '^[A-Z]+$'
    var_11 = module_1.compile(var_10)
    var_12 = module_0.String(pattern=var_11)
    var_13 = module_0.String(allow_blank=var_1)
    var_14 = module_0.String(allow_blank=var_2)
    var_15 = 'default'
    var_16 = hasattr(var_14, var_15)
    var_17 = module_0.String()
    var_18 = 'test'
    var_19 = module_0.String()
    var_20 = 'custom'
    var_21 = module_0.String(allow_blank=var_1)
    var_22 = 'invalid'
    var_23 = module_0.String(max_length=var_22)
    var_24 = 'invalid'
    var_25 = module_0.String(min_length=var_24)
    var_26 = 123
    var_27 = module_0.String(pattern=var_26)
    var_28 = 123
    var_29 = module_0.String(format=var_28)



# Parsed testcases at query #6
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = None
    var_3 = None
    var_4 = 'valid'
    var_5 = 'test'
    var_6 = 'type error'
    var_7 = 'type'
    var_8 = module_0.Message(text=var_6, code=var_7, index=var_2)
    var_9 = module_0.Message(text=var_6, code=var_7, index=var_2)
    var_10 = 'test'
    var_11 = 'min error'
    var_12 = 'minimum'
    var_13 = 'field'
    var_14 = [var_13]
    var_15 = module_0.Message(text=var_11, code=var_12, index=var_14)
    var_16 = module_0.Message(text=var_6, code=var_7, index=var_2)
    var_17 = 'test'
    var_18 = module_0.Message(text=var_11, code=var_12, index=var_2)
    var_19 = 'max error'
    var_20 = 'maximum'
    var_21 = module_0.Message(text=var_19, code=var_20, index=var_2)
    var_22 = 'test'
    var_23 = module_0.Message(text=var_6, code=var_7, index=var_2)
    var_24 = module_0.Message(text=var_11, code=var_12, index=var_2)
    var_25 = module_0.Message(text=var_6, code=var_7, index=var_2)
    var_26 = 'test'
    var_27 = [var_13]
    var_28 = module_0.Message(text=var_6, code=var_7, index=var_27)
    var_29 = module_0.Message(text=var_6, code=var_7, index=var_2)
    var_30 = 'test'



# Parsed testcases at query #7
#--------------------------


import typesystem.fields as module_0
import typesystem.formats as module_1

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
    var_11 = 123
    var_12 = var_8.validate(var_11)
    var_13 = module_0.String(allow_blank=var_7)
    var_14 = ''
    var_15 = var_13.validate(var_14)
    var_16 = module_0.String(allow_blank=var_3)
    var_17 = ''
    var_18 = var_16.validate(var_17)
    assert var_18 == ''
    var_19 = module_0.String(trim_whitespace=var_3)
    var_20 = '  hello  '
    var_21 = var_19.validate(var_20)
    assert var_21 == 'hello'
    var_22 = module_0.String(trim_whitespace=var_7)
    var_23 = var_22.validate(var_20)
    assert var_23 == '  hello  '
    var_24 = 3
    var_25 = module_0.String(min_length=var_24)
    var_26 = 'abc'
    var_27 = var_25.validate(var_26)
    assert var_27 == 'abc'
    var_28 = 'ab'
    var_29 = var_25.validate(var_28)
    var_30 = 5
    var_31 = module_0.String(max_length=var_30)
    var_32 = 'abcde'
    var_33 = var_31.validate(var_32)
    assert var_33 == 'abcde'
    var_34 = 'abcdef'
    var_35 = var_31.validate(var_34)
    var_36 = '^\\d+$'
    var_37 = module_0.String(pattern=var_36)
    var_38 = '123'
    var_39 = var_37.validate(var_38)
    assert var_39 == '123'
    var_40 = 'abc'
    var_41 = var_37.validate(var_40)
    var_42 = 'email'
    var_43 = module_0.String(format=var_42)
    var_44 = 'test@example.com'
    var_45 = var_43.validate(var_44)
    assert var_45 == 'test@example.com'
    var_46 = 'not-an-email'
    var_47 = var_43.validate(var_46)
    var_48 = module_0.String()
    var_49 = 'hello\x00world'
    var_50 = var_48.validate(var_49)
    assert var_50 == 'helloworld'
    var_51 = module_0.String(allow_blank=var_3, coerce_types=var_3)
    var_52 = var_51.validate(var_5)
    assert var_52 == ''
    var_53 = module_0.String(coerce_types=var_3)
    var_54 = var_53.validate(var_17)
    assert var_54 is None
    var_55 = 'uuid'
    var_56 = module_0.String(format=var_55)
    var_57 = module_1.UUIDFormat()
    var_58 = '12345678-1234-5678-1234-567812345678'
    var_59 = var_57.validate(var_58)
    var_60 = var_56.validate(var_59)



# Parsed testcases at query #8
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
    var_10 = 'A'
    var_11 = (var_0, var_10)
    var_12 = 'B'
    var_13 = (var_3, var_12)
    var_14 = [var_11, var_13]
    var_15 = True
    var_16 = module_0.Choice(choices=var_14)
    var_17 = None
    var_18 = var_16.validate(var_17)
    assert var_18 is None
    var_19 = (var_0, var_10)
    var_20 = (var_3, var_12)
    var_21 = [var_19, var_20]
    var_22 = module_0.Choice(choices=var_21)
    var_23 = None
    var_24 = var_22.validate(var_23)
    var_25 = (var_23, var_10)
    var_26 = (var_3, var_12)
    var_27 = [var_25, var_26]
    var_28 = module_0.Choice(choices=var_27)
    var_29 = 'c'
    var_30 = var_28.validate(var_29)
    var_31 = (var_29, var_10)
    var_32 = (var_3, var_12)
    var_33 = [var_31, var_32]
    var_34 = module_0.Choice(choices=var_33)
    var_35 = ''
    var_36 = var_34.validate(var_35)
    var_37 = (var_35, var_10)
    var_38 = (var_3, var_12)
    var_39 = [var_37, var_38]
    var_40 = module_0.Choice(choices=var_39)
    var_41 = ''
    var_42 = var_40.validate(var_41)
    assert var_42 is None
    var_43 = (var_35, var_10)
    var_44 = (var_3, var_12)
    var_45 = [var_43, var_44]
    var_46 = False
    var_47 = module_0.Choice(choices=var_45, coerce_types=var_46)
    var_48 = ''
    var_49 = var_47.validate(var_48)
    var_50 = 'One'
    var_51 = (var_15, var_50)
    var_52 = 2
    var_53 = 'Two'
    var_54 = (var_52, var_53)
    var_55 = [var_51, var_54]
    var_56 = module_0.Choice(choices=var_55)
    var_57 = var_56.validate(var_15)
    assert var_57 == 1
    var_58 = var_56.validate(var_52)
    assert var_58 == 2
    var_59 = '1'
    var_60 = 'String One'
    var_61 = (var_59, var_60)
    var_62 = 'Number Two'
    var_63 = (var_52, var_62)
    var_64 = [var_61, var_63]
    var_65 = module_0.Choice(choices=var_64)
    var_66 = var_65.validate(var_59)
    assert var_66 == '1'
    var_67 = var_65.validate(var_52)
    assert var_67 == 2
    var_68 = (var_48, var_10)
    var_69 = (var_3, var_12)
    var_70 = [var_68, var_69]
    var_71 = module_0.Choice(choices=var_70)
    var_72 = var_71.validate(var_48)
    assert var_72 == 'a'
    var_73 = (var_48, var_10)
    var_74 = (var_3, var_12)
    var_75 = [var_73, var_74]
    var_76 = module_0.Choice(choices=var_75)
    var_77 = 'c'



# Parsed testcases at query #9
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
    var_11 = module_0.Boolean(coerce_types=var_3)
    var_12 = 'true'
    var_13 = var_11.validate(var_12)
    var_14 = module_0.Boolean(coerce_types=var_12)
    var_15 = 'true'
    var_16 = var_14.validate(var_15)
    assert var_16 is True
    var_17 = 'false'
    var_18 = var_14.validate(var_17)
    assert var_18 is False
    var_19 = 'on'
    var_20 = var_14.validate(var_19)
    assert var_20 is True
    var_21 = 'off'
    var_22 = var_14.validate(var_21)
    assert var_22 is False
    var_23 = '1'
    var_24 = var_14.validate(var_23)
    assert var_24 is True
    var_25 = '0'
    var_26 = var_14.validate(var_25)
    assert var_26 is False
    var_27 = ''
    var_28 = var_14.validate(var_27)
    assert var_28 is False
    var_29 = var_14.validate(var_12)
    assert var_29 is True
    var_30 = var_14.validate(var_3)
    assert var_30 is False
    var_31 = 'TRUE'
    var_32 = var_14.validate(var_31)
    assert var_32 is True
    var_33 = 'FALSE'
    var_34 = var_14.validate(var_33)
    assert var_34 is False
    var_35 = 'On'
    var_36 = var_14.validate(var_35)
    assert var_36 is True
    var_37 = 'Off'
    var_38 = var_14.validate(var_37)
    assert var_38 is False
    var_39 = module_0.Boolean(coerce_types=var_12)
    var_40 = 'null'
    var_41 = var_39.validate(var_40)
    assert var_41 is None
    var_42 = 'none'
    var_43 = var_39.validate(var_42)
    assert var_43 is None
    var_44 = var_39.validate(var_27)
    assert var_44 is None
    var_45 = module_0.Boolean(coerce_types=var_12)
    var_46 = 'invalid'
    var_47 = var_45.validate(var_46)
    var_48 = []
    var_49 = var_45.validate(var_48)



# Parsed testcases at query #10
#--------------------------


import typesystem.fields as module_0
import typesystem.formats as module_1

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
    var_22 = 3
    var_23 = module_0.String(min_length=var_22)
    var_24 = 'abc'
    var_25 = var_23.validate(var_24)
    assert var_25 == 'abc'
    var_26 = 'ab'
    var_27 = var_23.validate(var_26)
    var_28 = 5
    var_29 = module_0.String(max_length=var_28)
    var_30 = 'abcde'
    var_31 = var_29.validate(var_30)
    assert var_31 == 'abcde'
    var_32 = 'abcdef'
    var_33 = var_29.validate(var_32)
    var_34 = '^\\d+$'
    var_35 = module_0.String(pattern=var_34)
    var_36 = '123'
    var_37 = var_35.validate(var_36)
    assert var_37 == '123'
    var_38 = 'abc'
    var_39 = var_35.validate(var_38)
    var_40 = 'email'
    var_41 = module_0.String(format=var_40)
    var_42 = 'test@example.com'
    var_43 = var_41.validate(var_42)
    assert var_43 == 'test@example.com'
    var_44 = 'not-an-email'
    var_45 = var_41.validate(var_44)
    var_46 = module_0.String()
    var_47 = 'hello\x00world'
    var_48 = var_46.validate(var_47)
    assert var_48 == 'helloworld'
    var_49 = module_0.String()
    var_50 = 123
    var_51 = var_49.validate(var_50)
    var_52 = module_0.String(allow_blank=var_3, coerce_types=var_3)
    var_53 = var_52.validate(var_5)
    assert var_53 == ''
    var_54 = module_0.String(coerce_types=var_3)
    var_55 = module_0.String(allow_blank=var_7, coerce_types=var_3)
    var_56 = var_55.validate(var_12)
    assert var_56 is None
    var_57 = 'uuid'
    var_58 = module_0.String(format=var_57)
    var_59 = module_1.UUIDFormat()
    var_60 = '12345678-1234-5678-1234-567812345678'
    var_61 = var_59.validate(var_60)
    var_62 = var_58.validate(var_61)



# Parsed testcases at query #11
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Const(var_0)
    var_2 = None
    var_3 = module_0.Const(var_2)
    var_4 = 'test'
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
    var_16 = 100
    var_17 = module_0.Const(var_16)



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
    var_22 = module_0.String()
    var_23 = {var_20: var_22}
    var_24 = module_0.Object(properties=var_23)
    var_25 = 'not an object'
    var_26 = var_24.validate(var_25)
    var_27 = module_0.String()
    var_28 = {var_25: var_27}
    var_29 = [var_25]
    var_30 = module_0.Object(properties=var_28, required=var_29)
    var_31 = 'age'
    var_32 = 30
    var_33 = {var_31: var_32}
    var_34 = var_30.validate(var_33)
    var_35 = 'Unknown'
    var_36 = module_0.String()
    var_37 = {var_31: var_36}
    var_38 = module_0.Object(properties=var_37)
    var_39 = {}
    var_40 = var_38.validate(var_39)
    var_41 = 'a'
    var_42 = 'b'
    var_43 = module_0.String()
    var_44 = module_0.String()
    var_45 = {var_41: var_43, var_42: var_44}
    var_46 = module_0.Object(properties=var_45, min_properties=var_12)
    var_47 = {}
    var_48 = var_46.validate(var_47)
    var_49 = module_0.String()
    var_50 = {var_41: var_49}
    var_51 = module_0.Object(properties=var_50, max_properties=var_12)
    var_52 = 'a'
    var_53 = 'b'
    var_54 = 'test'
    var_55 = 'test2'
    var_56 = {var_52: var_54, var_53: var_55}
    var_57 = var_51.validate(var_56)
    var_58 = '^test_'
    var_59 = module_0.String()
    var_60 = {var_58: var_59}
    var_61 = module_0.Object(pattern_properties=var_60)
    var_62 = 'test_1'
    var_63 = 'test_2'
    var_64 = 'value1'
    var_65 = 'value2'
    var_66 = {var_62: var_64, var_63: var_65}
    var_67 = var_61.validate(var_66)
    var_68 = module_0.String()
    var_69 = {var_52: var_68}
    var_70 = module_0.Object(properties=var_69, additional_properties=var_12)
    var_71 = 'extra'
    var_72 = 'field'
    var_73 = {var_52: var_57, var_71: var_72}
    var_74 = var_70.validate(var_73)
    var_75 = module_0.String()
    var_76 = {var_52: var_75}
    var_77 = module_0.Object(properties=var_76, additional_properties=var_18)
    var_78 = 'name'
    var_79 = 'extra'
    var_80 = 'John'
    var_81 = 'field'
    var_82 = {var_78: var_80, var_79: var_81}
    var_83 = var_77.validate(var_82)
    var_84 = module_0.String()
    var_85 = {var_78: var_84}
    var_86 = module_0.Integer()
    var_87 = module_0.Object(properties=var_85, additional_properties=var_86)
    var_88 = 'count'
    var_89 = 5
    var_90 = {var_78: var_83, var_88: var_89}
    var_91 = var_87.validate(var_90)
    var_92 = '^[a-z]+$'
    var_93 = module_0.String(pattern=var_92)
    var_94 = module_0.Object(additional_properties=var_12, property_names=var_93)
    var_95 = 'valid'
    var_96 = 'INVALID'
    var_97 = 'test'
    var_98 = {var_95: var_97, var_96: var_97}
    var_99 = var_94.validate(var_98)
    var_100 = module_0.Object(additional_properties=var_12)
    var_101 = 123
    var_102 = 'value'
    var_103 = {var_101: var_102}
    var_104 = var_100.validate(var_103)
    var_105 = 'person'
    var_106 = module_0.String()
    var_107 = module_0.Integer(minimum=var_18)
    var_108 = {var_101: var_106, var_102: var_107}
    var_109 = module_0.Object(properties=var_108)
    var_110 = {var_105: var_109}
    var_111 = module_0.Object(properties=var_110)
    var_112 = 'person'
    var_113 = 'age'
    var_114 = -5
    var_115 = {var_113: var_114}
    var_116 = {var_112: var_115}
    var_117 = var_111.validate(var_116)
    var_118 = 'id'
    var_119 = module_0.Integer()
    var_120 = {var_118: var_119}
    var_121 = '^attr_'
    var_122 = module_0.String()
    var_123 = {var_121: var_122}
    var_124 = module_0.Boolean()
    var_125 = 2
    var_126 = 4
    var_127 = module_0.Object(properties=var_120, pattern_properties=var_123, additional_properties=var_124, min_properties=var_125, max_properties=var_126)
    var_128 = 'attr_name'
    var_129 = 'enabled'
    var_130 = 'test'
    var_131 = {var_118: var_12, var_128: var_130, var_129: var_12}
    var_132 = var_127.validate(var_131)



# Parsed testcases at query #13
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Array()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = var_0.validate(var_4)
    var_6 = []
    var_7 = var_0.validate(var_6)
    var_8 = True
    var_9 = module_0.Array()
    var_10 = None
    var_11 = var_9.validate(var_10)
    assert var_11 is None
    var_12 = False
    var_13 = module_0.Array()
    var_14 = None
    var_15 = var_13.validate(var_14)
    var_16 = module_0.Array()
    var_17 = 'not a list'
    var_18 = var_16.validate(var_17)
    var_19 = module_0.Array(min_items=var_18)
    var_20 = [var_8, var_18]
    var_21 = var_19.validate(var_20)
    var_22 = [var_8, var_18, var_3]
    var_23 = var_19.validate(var_22)
    var_24 = 1
    var_25 = [var_24]
    var_26 = var_19.validate(var_25)
    var_27 = module_0.Array(min_items=var_8)
    var_28 = []
    var_29 = var_27.validate(var_28)
    var_30 = module_0.Array(max_items=var_29)
    var_31 = [var_8]
    var_32 = var_30.validate(var_31)
    var_33 = [var_8, var_29]
    var_34 = var_30.validate(var_33)
    var_35 = 1
    var_36 = 2
    var_37 = 3
    var_38 = [var_35, var_36, var_37]
    var_39 = var_30.validate(var_38)
    var_40 = module_0.Array(exact_items=var_37)
    var_41 = [var_8, var_36, var_37]
    var_42 = var_40.validate(var_41)
    var_43 = 1
    var_44 = 2
    var_45 = [var_43, var_44]
    var_46 = var_40.validate(var_45)
    var_47 = 1
    var_48 = 2
    var_49 = 3
    var_50 = 4
    var_51 = [var_47, var_48, var_49, var_50]
    var_52 = var_40.validate(var_51)
    var_53 = module_0.Integer()
    var_54 = module_0.Array(var_53)
    var_55 = [var_8, var_48, var_49]
    var_56 = var_54.validate(var_55)
    var_57 = 1
    var_58 = 'invalid'
    var_59 = 3
    var_60 = [var_57, var_58, var_59]
    var_61 = var_54.validate(var_60)
    var_62 = module_0.Integer()
    var_63 = module_0.String()
    var_64 = [var_62, var_63]
    var_65 = module_0.Array(var_64)
    var_66 = 'test'
    var_67 = [var_8, var_66]
    var_68 = var_65.validate(var_67)
    var_69 = 'invalid'
    var_70 = 'test'
    var_71 = [var_69, var_70]
    var_72 = var_65.validate(var_71)
    var_73 = module_0.Integer()
    var_74 = module_0.String()
    var_75 = [var_73, var_74]
    var_76 = module_0.Array(var_75, var_12)
    var_77 = [var_8, var_66]
    var_78 = var_76.validate(var_77)
    var_79 = 1
    var_80 = 'test'
    var_81 = 'extra'
    var_82 = [var_79, var_80, var_81]
    var_83 = var_76.validate(var_82)
    var_84 = module_0.Integer()
    var_85 = module_0.String()
    var_86 = [var_84, var_85]
    var_87 = module_0.Boolean()
    var_88 = module_0.Array(var_86, var_87)
    var_89 = True
    var_90 = [var_8, var_66, var_89, var_12]
    var_91 = var_88.validate(var_90)
    var_92 = 1
    var_93 = 'test'
    var_94 = 'not boolean'
    var_95 = [var_92, var_93, var_94]
    var_96 = var_88.validate(var_95)
    var_97 = True
    var_98 = module_0.Array(unique_items=var_97)
    var_99 = [var_97, var_93, var_94]
    var_100 = var_98.validate(var_99)
    var_101 = 1
    var_102 = 2
    var_103 = [var_101, var_102, var_101]
    var_104 = var_98.validate(var_103)
    var_105 = module_0.Integer()
    var_106 = module_0.Array(var_105)
    var_107 = module_0.Array(var_106)
    var_108 = [var_97, var_102]
    var_109 = 4
    var_110 = [var_103, var_109]
    var_111 = [var_108, var_110]
    var_112 = var_107.validate(var_111)
    var_113 = 1
    var_114 = 2
    var_115 = [var_113, var_114]
    var_116 = 3
    var_117 = 'invalid'
    var_118 = [var_116, var_117]
    var_119 = [var_115, var_118]
    var_120 = var_107.validate(var_119)
    var_121 = module_0.Integer()
    var_122 = module_0.Array(var_121, min_items=var_114, max_items=var_115)
    var_123 = 1
    var_124 = 'invalid'
    var_125 = 3
    var_126 = 4
    var_127 = 5
    var_128 = [var_123, var_124, var_125, var_126, var_127]
    var_129 = var_122.validate(var_128)
    var_130 = True
    var_131 = module_0.String()
    var_132 = module_0.Array(var_131)
    var_133 = 'a'
    var_134 = 'c'
    var_135 = [var_133, var_10, var_134]
    var_136 = var_132.validate(var_135)



# Parsed testcases at query #14
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
    var_22 = module_0.String()
    var_23 = {var_20: var_22}
    var_24 = module_0.Object(properties=var_23)
    var_25 = 'not an object'
    var_26 = var_24.validate(var_25)
    var_27 = module_0.String()
    var_28 = {var_25: var_27}
    var_29 = [var_25]
    var_30 = module_0.Object(properties=var_28, required=var_29)
    var_31 = 'age'
    var_32 = 30
    var_33 = {var_31: var_32}
    var_34 = var_30.validate(var_33)
    var_35 = 'Unknown'
    var_36 = module_0.String()
    var_37 = {var_31: var_36}
    var_38 = module_0.Object(properties=var_37)
    var_39 = {}
    var_40 = var_38.validate(var_39)
    var_41 = module_0.String()
    var_42 = {var_31: var_41}
    var_43 = module_0.Object(properties=var_42)
    var_44 = 123
    var_45 = 'value'
    var_46 = {var_44: var_45}
    var_47 = var_43.validate(var_46)
    var_48 = '^[a-z_]+$'
    var_49 = module_0.String(pattern=var_48)
    var_50 = {}
    var_51 = module_0.Object(properties=var_50, property_names=var_49)
    var_52 = 'InvalidKey'
    var_53 = 'value'
    var_54 = {var_52: var_53}
    var_55 = var_51.validate(var_54)
    var_56 = {}
    var_57 = 2
    var_58 = module_0.Object(properties=var_56, min_properties=var_57)
    var_59 = 'key1'
    var_60 = 'value1'
    var_61 = {var_59: var_60}
    var_62 = var_58.validate(var_61)
    var_63 = {}
    var_64 = module_0.Object(properties=var_63, min_properties=var_12)
    var_65 = {}
    var_66 = var_64.validate(var_65)
    var_67 = {}
    var_68 = module_0.Object(properties=var_67, max_properties=var_57)
    var_69 = 'k1'
    var_70 = 'k2'
    var_71 = 'k3'
    var_72 = 'v1'
    var_73 = 'v2'
    var_74 = 'v3'
    var_75 = {var_69: var_72, var_70: var_73, var_71: var_74}
    var_76 = var_68.validate(var_75)
    var_77 = '^test_'
    var_78 = module_0.String()
    var_79 = {var_77: var_78}
    var_80 = module_0.Object(pattern_properties=var_79)
    var_81 = 'test_key'
    var_82 = 'other'
    var_83 = 'value'
    var_84 = 'ignored'
    var_85 = {var_81: var_83, var_82: var_84}
    var_86 = var_80.validate(var_85)
    var_87 = module_0.String()
    var_88 = {var_69: var_87}
    var_89 = module_0.Object(properties=var_88, additional_properties=var_12)
    var_90 = 'extra'
    var_91 = {var_69: var_74, var_90: var_83}
    var_92 = var_89.validate(var_91)
    var_93 = module_0.String()
    var_94 = {var_69: var_93}
    var_95 = module_0.Object(properties=var_94, additional_properties=var_18)
    var_96 = 'name'
    var_97 = 'extra'
    var_98 = 'John'
    var_99 = 'value'
    var_100 = {var_96: var_98, var_97: var_99}
    var_101 = var_95.validate(var_100)
    var_102 = module_0.String()
    var_103 = {var_96: var_102}
    var_104 = module_0.Integer()
    var_105 = module_0.Object(properties=var_103, additional_properties=var_104)
    var_106 = 42
    var_107 = {var_96: var_101, var_90: var_106}
    var_108 = var_105.validate(var_107)
    var_109 = 'name'
    var_110 = 'extra'
    var_111 = 'John'
    var_112 = 'not a number'
    var_113 = {var_109: var_111, var_110: var_112}
    var_114 = var_105.validate(var_113)
    var_115 = module_0.String(min_length=var_57)
    var_116 = module_0.Integer(minimum=var_18)
    var_117 = {var_109: var_115, var_110: var_116}
    var_118 = module_0.Object(properties=var_117)
    var_119 = 'name'
    var_120 = 'age'
    var_121 = 'J'
    var_122 = -5
    var_123 = {var_119: var_121, var_120: var_122}
    var_124 = var_118.validate(var_123)
    var_125 = 'street'
    var_126 = 'city'
    var_127 = module_0.String()
    var_128 = module_0.String()
    var_129 = {var_125: var_127, var_126: var_128}
    var_130 = module_0.Object(properties=var_129)
    var_131 = 'address'
    var_132 = module_0.String()
    var_133 = {var_119: var_132, var_131: var_130}
    var_134 = module_0.Object(properties=var_133)
    var_135 = '123 Main'
    var_136 = 'Anytown'
    var_137 = {var_125: var_135, var_126: var_136}
    var_138 = {var_119: var_124, var_131: var_137}
    var_139 = var_134.validate(var_138)
    var_140 = module_0.String()
    var_141 = {var_119: var_140}
    var_142 = module_0.Object(properties=var_141, additional_properties=var_14)
    var_143 = {var_119: var_124, var_90: var_83}
    var_144 = var_142.validate(var_143)



# Parsed testcases at query #15
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = 42
    var_2 = var_0.validate(var_1)
    assert var_2 == 42
    var_3 = 0
    var_4 = var_0.validate(var_3)
    assert var_4 == 0
    var_5 = -10
    var_6 = var_0.validate(var_5)
    assert var_6 == -10
    var_7 = 3.14
    var_8 = var_0.validate(var_7)
    var_9 = -2.5
    var_10 = var_0.validate(var_9)
    var_11 = True
    var_12 = module_0.Number()
    var_13 = None
    var_14 = var_12.validate(var_13)
    assert var_14 is None
    var_15 = False
    var_16 = module_0.Number()
    var_17 = None
    var_18 = var_16.validate(var_17)
    var_19 = module_0.Number(coerce_types=var_11)
    var_20 = '42'
    var_21 = var_19.validate(var_20)
    assert var_21 == 42
    var_22 = '3.14'
    var_23 = var_19.validate(var_22)
    var_24 = False
    var_25 = module_0.Number(coerce_types=var_24)
    var_26 = '42'
    var_27 = var_25.validate(var_26)
    var_28 = True
    var_29 = var_0.validate(var_28)
    var_30 = False
    var_31 = var_0.validate(var_30)
    var_32 = 10
    var_33 = module_0.Number(minimum=var_32)
    var_34 = var_33.validate(var_32)
    assert var_34 == 10
    var_35 = 15
    var_36 = var_33.validate(var_35)
    assert var_36 == 15
    var_37 = 5
    var_38 = var_33.validate(var_37)
    var_39 = 100
    var_40 = module_0.Number(maximum=var_39)
    var_41 = var_40.validate(var_39)
    assert var_41 == 100
    var_42 = 50
    var_43 = var_40.validate(var_42)
    assert var_43 == 50
    var_44 = 150
    var_45 = var_40.validate(var_44)
    var_46 = module_0.Number(exclusive_minimum=var_32)
    var_47 = 11
    var_48 = var_46.validate(var_47)
    assert var_48 == 11
    var_49 = 10
    var_50 = var_46.validate(var_49)
    var_51 = module_0.Number(exclusive_maximum=var_39)
    var_52 = 99
    var_53 = var_51.validate(var_52)
    assert var_53 == 99
    var_54 = 100
    var_55 = var_51.validate(var_54)
    var_56 = 5
    var_57 = module_0.Number(multiple_of=var_56)
    var_58 = var_57.validate(var_32)
    assert var_58 == 10
    var_59 = var_57.validate(var_35)
    assert var_59 == 15
    var_60 = -20
    var_61 = var_57.validate(var_60)
    assert var_61 == -20
    var_62 = 12
    var_63 = var_57.validate(var_62)
    var_64 = 0.5
    var_65 = module_0.Number(multiple_of=var_64)
    var_66 = var_65.validate(var_11)
    var_67 = 1.5
    var_68 = var_65.validate(var_67)
    var_69 = 1.2
    var_70 = var_65.validate(var_69)
    var_71 = '0.01'
    var_72 = module_0.Number(precision=var_71)
    var_73 = 1.234
    var_74 = var_72.validate(var_73)
    var_75 = 1.235
    var_76 = var_72.validate(var_75)
    var_77 = module_0.Number()
    var_78 = 'inf'
    var_79 = float(var_78)
    var_80 = var_77.validate(var_79)
    var_81 = '-inf'
    var_82 = float(var_81)
    var_83 = var_77.validate(var_82)
    var_84 = 'nan'
    var_85 = float(var_84)
    var_86 = var_77.validate(var_85)
    var_87 = module_0.Number()
    var_88 = var_87.validate(var_84)
    assert var_88 == 42
    var_89 = var_87.validate(var_24)
    assert var_89 == 0
    var_90 = 3.14
    var_91 = var_87.validate(var_90)
    var_92 = module_0.Number()
    var_93 = '10.5'
    var_94 = module_0.Number(coerce_types=var_11)
    var_95 = ''
    var_96 = var_94.validate(var_95)
    assert var_96 is None
    var_97 = module_0.Number(minimum=var_24, maximum=var_39, multiple_of=var_32)
    var_98 = var_97.validate(var_24)
    assert var_98 == 0
    var_99 = var_97.validate(var_42)
    assert var_99 == 50
    var_100 = var_97.validate(var_39)
    assert var_100 == 100
    var_101 = 55
    var_102 = var_97.validate(var_101)



# Parsed testcases at query #16
#--------------------------


import typesystem.fields as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = True
    var_2 = False
    var_3 = 100
    var_4 = 5
    var_5 = '^\\d+$'
    var_6 = 'email'
    var_7 = 'Test Title'
    var_8 = 'Test Description'
    var_9 = module_0.String(allow_blank=var_1, trim_whitespace=var_2, max_length=var_3, min_length=var_4, pattern=var_5, format=var_6, coerce_types=var_2)
    var_10 = '^[A-Z]+$'
    var_11 = module_1.compile(var_10)
    var_12 = module_0.String(pattern=var_11)
    var_13 = module_0.String(allow_blank=var_1)
    var_14 = module_0.String(allow_blank=var_2)
    var_15 = 'default'
    var_16 = hasattr(var_14, var_15)
    var_17 = module_0.String()
    var_18 = 'custom'
    var_19 = module_0.String()
    var_20 = 'explicit'
    var_21 = module_0.String(allow_blank=var_1)
    var_22 = 'invalid'
    var_23 = module_0.String(max_length=var_22)
    var_24 = 'invalid'
    var_25 = module_0.String(min_length=var_24)
    var_26 = 123
    var_27 = module_0.String(pattern=var_26)
    var_28 = 123
    var_29 = module_0.String(format=var_28)



# Parsed testcases at query #17
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = 42
    var_2 = var_0.validate(var_1)
    assert var_2 == 42
    var_3 = 0
    var_4 = var_0.validate(var_3)
    assert var_4 == 0
    var_5 = -10
    var_6 = var_0.validate(var_5)
    assert var_6 == -10
    var_7 = 3.14
    var_8 = var_0.validate(var_7)
    var_9 = -2.5
    var_10 = var_0.validate(var_9)
    var_11 = True
    var_12 = module_0.Number()
    var_13 = None
    var_14 = var_12.validate(var_13)
    assert var_14 is None
    var_15 = module_0.Number()
    var_16 = None
    var_17 = var_15.validate(var_16)
    var_18 = '42'
    var_19 = var_0.validate(var_18)
    assert var_19 == 42
    var_20 = '3.14'
    var_21 = var_0.validate(var_20)
    var_22 = False
    var_23 = module_0.Number(coerce_types=var_22)
    var_24 = '42'
    var_25 = var_23.validate(var_24)
    var_26 = True
    var_27 = var_0.validate(var_26)
    var_28 = False
    var_29 = var_0.validate(var_28)
    var_30 = 10
    var_31 = module_0.Number(minimum=var_30)
    var_32 = var_31.validate(var_30)
    assert var_32 == 10
    var_33 = 15
    var_34 = var_31.validate(var_33)
    assert var_34 == 15
    var_35 = 5
    var_36 = var_31.validate(var_35)
    var_37 = 100
    var_38 = module_0.Number(maximum=var_37)
    var_39 = var_38.validate(var_37)
    assert var_39 == 100
    var_40 = 50
    var_41 = var_38.validate(var_40)
    assert var_41 == 50
    var_42 = 150
    var_43 = var_38.validate(var_42)
    var_44 = module_0.Number(exclusive_minimum=var_30)
    var_45 = 11
    var_46 = var_44.validate(var_45)
    assert var_46 == 11
    var_47 = 10.1
    var_48 = var_44.validate(var_47)
    var_49 = 10
    var_50 = var_44.validate(var_49)
    var_51 = module_0.Number(exclusive_maximum=var_37)
    var_52 = 99
    var_53 = var_51.validate(var_52)
    assert var_53 == 99
    var_54 = 99.9
    var_55 = var_51.validate(var_54)
    var_56 = 100
    var_57 = var_51.validate(var_56)
    var_58 = 5
    var_59 = module_0.Number(multiple_of=var_58)
    var_60 = var_59.validate(var_30)
    assert var_60 == 10
    var_61 = var_59.validate(var_22)
    assert var_61 == 0
    var_62 = -15
    var_63 = var_59.validate(var_62)
    assert var_63 == -15
    var_64 = 7
    var_65 = var_59.validate(var_64)
    var_66 = 0.5
    var_67 = module_0.Number(multiple_of=var_66)
    var_68 = var_67.validate(var_11)
    var_69 = 2.5
    var_70 = var_67.validate(var_69)
    var_71 = 1.2
    var_72 = var_67.validate(var_71)
    var_73 = '0.01'
    var_74 = module_0.Number(precision=var_73)
    var_75 = 1.23
    var_76 = var_74.validate(var_75)
    var_77 = 1.234
    var_78 = var_74.validate(var_77)
    var_79 = 'inf'
    var_80 = float(var_79)
    var_81 = var_0.validate(var_80)
    var_82 = '-inf'
    var_83 = float(var_82)
    var_84 = var_0.validate(var_83)
    var_85 = 'nan'
    var_86 = float(var_85)
    var_87 = var_0.validate(var_86)
    var_88 = module_0.Number()
    var_89 = var_88.validate(var_85)
    assert var_89 == 42
    var_90 = var_88.validate(var_22)
    assert var_90 == 0
    var_91 = 3.14
    var_92 = var_88.validate(var_91)
    var_93 = module_0.Number(coerce_types=var_11)
    var_94 = ''
    var_95 = var_93.validate(var_94)
    assert var_95 is None
    var_96 = 'not a number'
    var_97 = var_0.validate(var_96)
    var_98 = '10.5'
    var_99 = module_0.Number(minimum=var_22, maximum=var_37, multiple_of=var_30)
    var_100 = var_99.validate(var_22)
    assert var_100 == 0
    var_101 = var_99.validate(var_40)
    assert var_101 == 50
    var_102 = var_99.validate(var_37)
    assert var_102 == 100
    var_103 = 45
    var_104 = var_99.validate(var_103)



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
    var_5 = module_0.Boolean()
    var_6 = None
    var_7 = var_5.validate(var_6)
    assert var_7 is None
    var_8 = module_0.Boolean()
    var_9 = None
    var_10 = var_8.validate(var_9)
    var_11 = module_0.Boolean()
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
    var_28 = module_0.Boolean(coerce_types=var_3)
    var_29 = var_28.validate(var_9)
    assert var_29 is True
    var_30 = var_28.validate(var_3)
    assert var_30 is False
    var_31 = 'true'
    var_32 = var_28.validate(var_31)
    var_33 = module_0.Boolean()
    var_34 = 'null'
    var_35 = var_33.validate(var_34)
    assert var_35 is None
    var_36 = 'none'
    var_37 = var_33.validate(var_36)
    assert var_37 is None
    var_38 = var_33.validate(var_24)
    assert var_38 is None
    var_39 = module_0.Boolean()
    var_40 = 'invalid'
    var_41 = var_39.validate(var_40)
    var_42 = 2
    var_43 = var_39.validate(var_42)
    var_44 = module_0.Boolean()
    var_45 = 'TRUE'
    var_46 = var_44.validate(var_45)
    assert var_46 is True
    var_47 = 'FALSE'
    var_48 = var_44.validate(var_47)
    assert var_48 is False
    var_49 = 'On'
    var_50 = var_44.validate(var_49)
    assert var_50 is True
    var_51 = 'Off'
    var_52 = var_44.validate(var_51)
    assert var_52 is False
    var_53 = module_0.Boolean()
    var_54 = []
    var_55 = var_53.validate(var_54)
    var_56 = {}
    var_57 = var_53.validate(var_56)



# Parsed testcases at query #19
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.Integer()
    var_2 = module_0.String()
    var_3 = [var_1, var_2]
    var_4 = module_0.Union(var_3)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None
    var_7 = False
    var_8 = module_0.Integer()
    var_9 = module_0.String()
    var_10 = [var_8, var_9]
    var_11 = module_0.Union(var_10)
    var_12 = None
    var_13 = var_11.validate(var_12)
    var_14 = exc_info.value.messages()[var_7]
    var_15 = var_14.code
    assert var_15 == 'null'
    var_16 = module_0.Integer()
    var_17 = module_0.String()
    var_18 = [var_16, var_17]
    var_19 = module_0.Union(var_18)
    var_20 = 42
    var_21 = var_19.validate(var_20)
    assert var_21 == 42
    var_22 = module_0.Integer()
    var_23 = module_0.String()
    var_24 = [var_22, var_23]
    var_25 = module_0.Union(var_24)
    var_26 = 'hello'
    var_27 = var_25.validate(var_26)
    assert var_27 == 'hello'
    var_28 = module_0.Integer()
    var_29 = module_0.String()
    var_30 = [var_28, var_29]
    var_31 = module_0.Union(var_30)
    var_32 = True
    var_33 = var_31.validate(var_32)
    var_34 = exc_info.value.messages()[var_7]
    var_35 = var_34.code
    assert var_35 == 'union'
    var_36 = 10
    var_37 = module_0.Integer(minimum=var_36)
    var_38 = module_0.String()
    var_39 = [var_37, var_38]
    var_40 = module_0.Union(var_39)
    var_41 = 5
    var_42 = var_40.validate(var_41)
    var_43 = exc_info.value.messages()[var_7]
    var_44 = var_43.code
    assert var_44 == 'minimum'
    var_45 = module_0.Integer(minimum=var_36)
    var_46 = 5
    var_47 = module_0.String(min_length=var_46)
    var_48 = [var_45, var_47]
    var_49 = module_0.Union(var_48)
    var_50 = 3
    var_51 = var_49.validate(var_50)
    var_52 = exc_info.value.messages()[var_7]
    var_53 = var_52.code
    assert var_53 == 'union'
    var_54 = module_0.Integer()
    var_55 = module_0.Array(var_54)
    var_56 = 'value'
    var_57 = module_0.Integer()
    var_58 = {var_56: var_57}
    var_59 = module_0.Object(properties=var_58)
    var_60 = [var_55, var_59]
    var_61 = module_0.Union(var_60)
    var_62 = 2
    var_63 = 3
    var_64 = [var_50, var_62, var_63]
    var_65 = var_61.validate(var_64)
    var_66 = {var_56: var_20}
    var_67 = var_61.validate(var_66)
    var_68 = module_0.Decimal()
    var_69 = module_0.Float()
    var_70 = [var_68, var_69]
    var_71 = module_0.Union(var_70)
    var_72 = '3.14'
    var_73 = 3.14
    var_74 = var_71.validate(var_73)
    var_75 = module_0.Boolean()
    var_76 = module_0.Integer()
    var_77 = [var_75, var_76]
    var_78 = module_0.Union(var_77)
    var_79 = var_78.validate(var_50)
    assert var_79 is True
    var_80 = var_78.validate(var_7)
    assert var_80 is False
    var_81 = var_78.validate(var_20)
    assert var_81 == 42
    var_82 = module_0.Integer()
    var_83 = module_0.String()
    var_84 = [var_82, var_83]
    var_85 = module_0.Union(var_84)
    var_86 = var_85.validate(var_5)
    assert var_86 is None
    var_87 = module_0.Integer(coerce_types=var_50)
    var_88 = module_0.String(coerce_types=var_50)
    var_89 = [var_87, var_88]
    var_90 = module_0.Union(var_89)
    var_91 = '42'
    var_92 = var_90.validate(var_91)
    assert var_92 == 42
    var_93 = var_90.validate(var_20)
    assert var_93 == '42'
    var_94 = 'a'
    var_95 = 'A'
    var_96 = (var_94, var_95)
    var_97 = 'b'
    var_98 = 'B'
    var_99 = (var_97, var_98)
    var_100 = [var_96, var_99]
    var_101 = module_0.Choice(choices=var_100)
    var_102 = module_0.Integer()
    var_103 = [var_101, var_102]
    var_104 = module_0.Union(var_103)
    var_105 = var_104.validate(var_94)
    assert var_105 == 'a'
    var_106 = 100
    var_107 = var_104.validate(var_106)
    assert var_107 == 100
    var_108 = module_0.Date()
    var_109 = module_0.DateTime()
    var_110 = [var_108, var_109]
    var_111 = module_0.Union(var_110)
    var_112 = '2023-01-01'
    var_113 = var_111.validate(var_112)
    var_114 = 'type'
    var_115 = module_0.String()
    var_116 = module_0.Integer()
    var_117 = {var_114: var_115, var_56: var_116}
    var_118 = module_0.Object(properties=var_117)
    var_119 = module_0.String()
    var_120 = module_0.Array(var_119)
    var_121 = module_0.Boolean()
    var_122 = [var_118, var_120, var_121]
    var_123 = module_0.Union(var_122)
    var_124 = 'test'
    var_125 = {var_114: var_124, var_56: var_20}
    var_126 = var_123.validate(var_125)
    var_127 = 'c'
    var_128 = [var_94, var_97, var_127]
    var_129 = var_123.validate(var_128)
    var_130 = var_123.validate(var_50)
    assert var_130 is True
    var_131 = 'type'
    var_132 = 'test'
    var_133 = {var_131: var_132}
    var_134 = var_123.validate(var_133)
    var_135 = exc_info.value.messages()[var_7]
    var_136 = var_135.code
    assert var_136 == 'required'



# Parsed testcases at query #20
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Const(var_0)
    var_2 = 'hello'
    var_3 = module_0.Const(var_2)
    var_4 = None
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
    var_16 = 42
    var_17 = True
    var_18 = module_0.Const(var_16)



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
    var_6 = []
    var_7 = var_0.validate(var_6)
    var_8 = True
    var_9 = module_0.Array()
    var_10 = None
    var_11 = var_9.validate(var_10)
    assert var_11 is None
    var_12 = False
    var_13 = module_0.Array()
    var_14 = None
    var_15 = var_13.validate(var_14)
    var_16 = module_0.Array()
    var_17 = 'not an array'
    var_18 = var_16.validate(var_17)
    var_19 = module_0.Array(min_items=var_18)
    var_20 = [var_8, var_18]
    var_21 = var_19.validate(var_20)
    var_22 = [var_8, var_18, var_3]
    var_23 = var_19.validate(var_22)
    var_24 = 1
    var_25 = [var_24]
    var_26 = var_19.validate(var_25)
    var_27 = module_0.Array(min_items=var_8)
    var_28 = []
    var_29 = var_27.validate(var_28)
    var_30 = module_0.Array(max_items=var_26)
    var_31 = [var_8, var_29, var_26]
    var_32 = var_30.validate(var_31)
    var_33 = [var_8, var_29]
    var_34 = var_30.validate(var_33)
    var_35 = 1
    var_36 = 2
    var_37 = 3
    var_38 = 4
    var_39 = [var_35, var_36, var_37, var_38]
    var_40 = var_30.validate(var_39)
    var_41 = module_0.Array(exact_items=var_37)
    var_42 = [var_8, var_36, var_37]
    var_43 = var_41.validate(var_42)
    var_44 = 1
    var_45 = 2
    var_46 = [var_44, var_45]
    var_47 = var_41.validate(var_46)
    var_48 = 1
    var_49 = 2
    var_50 = 3
    var_51 = 4
    var_52 = [var_48, var_49, var_50, var_51]
    var_53 = var_41.validate(var_52)
    var_54 = True
    var_55 = module_0.Array(unique_items=var_54)
    var_56 = [var_54, var_49, var_50]
    var_57 = var_55.validate(var_56)
    var_58 = 1
    var_59 = 2
    var_60 = [var_58, var_59, var_58]
    var_61 = var_55.validate(var_60)
    var_62 = module_0.Integer()
    var_63 = module_0.Array(var_62)
    var_64 = [var_54, var_59, var_60]
    var_65 = var_63.validate(var_64)
    var_66 = 1
    var_67 = 'invalid'
    var_68 = 3
    var_69 = [var_66, var_67, var_68]
    var_70 = var_63.validate(var_69)
    var_71 = module_0.Integer()
    var_72 = module_0.String()
    var_73 = [var_71, var_72]
    var_74 = module_0.Array(var_73)
    var_75 = 'hello'
    var_76 = [var_54, var_75]
    var_77 = var_74.validate(var_76)
    var_78 = 'invalid'
    var_79 = 'hello'
    var_80 = [var_78, var_79]
    var_81 = var_74.validate(var_80)
    var_82 = module_0.Integer()
    var_83 = module_0.String()
    var_84 = [var_82, var_83]
    var_85 = module_0.Array(var_84, var_12)
    var_86 = [var_54, var_75]
    var_87 = var_85.validate(var_86)
    var_88 = 1
    var_89 = 'hello'
    var_90 = 'extra'
    var_91 = [var_88, var_89, var_90]
    var_92 = var_85.validate(var_91)
    var_93 = module_0.Integer()
    var_94 = module_0.String()
    var_95 = [var_93, var_94]
    var_96 = True
    var_97 = module_0.Array(var_95, var_96)
    var_98 = 'extra'
    var_99 = 4
    var_100 = [var_96, var_75, var_98, var_99]
    var_101 = var_97.validate(var_100)
    var_102 = module_0.Integer()
    var_103 = module_0.String()
    var_104 = [var_102, var_103]
    var_105 = module_0.Boolean()
    var_106 = module_0.Array(var_104, var_105)
    var_107 = True
    var_108 = [var_96, var_75, var_107, var_12]
    var_109 = var_106.validate(var_108)
    var_110 = 1
    var_111 = 'hello'
    var_112 = 'not boolean'
    var_113 = [var_110, var_111, var_112]
    var_114 = var_106.validate(var_113)
    var_115 = module_0.Integer()
    var_116 = True
    var_117 = module_0.Array(var_115, unique_items=var_116)
    var_118 = 1
    var_119 = 'invalid'
    var_120 = [var_118, var_119, var_118]
    var_121 = var_117.validate(var_120)
    var_122 = module_0.Integer()
    var_123 = module_0.Array(var_122)
    var_124 = module_0.Array(var_123)
    var_125 = [var_116, var_119]
    var_126 = [var_120, var_99]
    var_127 = [var_125, var_126]
    var_128 = var_124.validate(var_127)
    var_129 = 1
    var_130 = 2
    var_131 = [var_129, var_130]
    var_132 = 'invalid'
    var_133 = 4
    var_134 = [var_132, var_133]
    var_135 = [var_131, var_134]
    var_136 = var_124.validate(var_135)



# Parsed testcases at query #22
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Array()
    var_1 = module_0.String()
    var_2 = module_0.Array(var_1)
    var_3 = module_0.String()
    var_4 = module_0.Integer()
    var_5 = [var_3, var_4]
    var_6 = module_0.Array(var_5)
    var_7 = module_0.Boolean()
    var_8 = module_0.Array(var_5, var_7)
    var_9 = True
    var_10 = module_0.Array(var_5, var_9)
    var_11 = 10
    var_12 = module_0.Array(min_items=var_9, max_items=var_11)
    var_13 = 5
    var_14 = module_0.Array(exact_items=var_13)
    var_15 = module_0.Array(unique_items=var_9)
    var_16 = module_0.Array()
    var_17 = 3
    var_18 = module_0.Array(min_items=var_9, max_items=var_11, exact_items=var_17)
    var_19 = module_0.String()
    var_20 = module_0.Integer()
    var_21 = module_0.Boolean()
    var_22 = [var_19, var_20, var_21]
    var_23 = module_0.Array(var_22)
    var_24 = module_0.String()
    var_25 = module_0.Array(var_22, var_24)
    var_26 = module_0.Array(var_22, min_items=var_9)
    var_27 = module_0.Array(var_22, max_items=var_13)
    var_28 = module_0.String()
    var_29 = module_0.Array(var_28)



# Parsed testcases at query #23
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
    var_11 = 2
    var_12 = module_0.Array(min_items=var_11)
    var_13 = 1
    var_14 = [var_13]
    var_15 = var_12.validate(var_14)
    var_16 = [var_13, var_11]
    var_17 = var_12.validate(var_16)
    var_18 = 3
    var_19 = [var_13, var_11, var_18]
    var_20 = var_12.validate(var_19)
    var_21 = module_0.Array(max_items=var_11)
    var_22 = 1
    var_23 = 2
    var_24 = 3
    var_25 = [var_22, var_23, var_24]
    var_26 = var_21.validate(var_25)
    var_27 = [var_22]
    var_28 = var_21.validate(var_27)
    var_29 = [var_22, var_26]
    var_30 = var_21.validate(var_29)
    var_31 = module_0.Array(exact_items=var_26)
    var_32 = 1
    var_33 = [var_32]
    var_34 = var_31.validate(var_33)
    var_35 = 1
    var_36 = 2
    var_37 = 3
    var_38 = [var_35, var_36, var_37]
    var_39 = var_31.validate(var_38)
    var_40 = [var_35, var_39]
    var_41 = var_31.validate(var_40)
    var_42 = module_0.Array(min_items=var_35)
    var_43 = []
    var_44 = var_42.validate(var_43)
    var_45 = module_0.Integer()
    var_46 = module_0.Array(var_45)
    var_47 = [var_43, var_39, var_18]
    var_48 = var_46.validate(var_47)
    var_49 = 1
    var_50 = 'invalid'
    var_51 = 3
    var_52 = [var_49, var_50, var_51]
    var_53 = var_46.validate(var_52)
    var_54 = module_0.Integer()
    var_55 = module_0.String()
    var_56 = [var_54, var_55]
    var_57 = module_0.Array(var_56)
    var_58 = 'test'
    var_59 = [var_49, var_58]
    var_60 = var_57.validate(var_59)
    var_61 = 'invalid'
    var_62 = 'test'
    var_63 = [var_61, var_62]
    var_64 = var_57.validate(var_63)
    var_65 = module_0.Integer()
    var_66 = module_0.String()
    var_67 = [var_65, var_66]
    var_68 = module_0.Array(var_67, var_64)
    var_69 = 1
    var_70 = 'test'
    var_71 = 'extra'
    var_72 = [var_69, var_70, var_71]
    var_73 = var_68.validate(var_72)
    var_74 = module_0.Integer()
    var_75 = module_0.String()
    var_76 = [var_74, var_75]
    var_77 = module_0.Array(var_76, var_69)
    var_78 = 'extra'
    var_79 = 4
    var_80 = [var_69, var_58, var_78, var_79]
    var_81 = var_77.validate(var_80)
    var_82 = module_0.Integer()
    var_83 = [var_82]
    var_84 = module_0.String()
    var_85 = module_0.Array(var_83, var_84)
    var_86 = 'another'
    var_87 = [var_69, var_58, var_86]
    var_88 = var_85.validate(var_87)
    var_89 = 1
    var_90 = 2
    var_91 = 3
    var_92 = [var_89, var_90, var_91]
    var_93 = var_85.validate(var_92)
    var_94 = module_0.Array(unique_items=var_89)
    var_95 = [var_89, var_93, var_18]
    var_96 = var_94.validate(var_95)
    var_97 = 1
    var_98 = 2
    var_99 = [var_97, var_98, var_97]
    var_100 = var_94.validate(var_99)
    var_101 = module_0.Integer()
    var_102 = module_0.Array(var_101)
    var_103 = module_0.Array(var_102)
    var_104 = [var_97, var_93]
    var_105 = [var_18, var_79]
    var_106 = [var_104, var_105]
    var_107 = var_103.validate(var_106)
    var_108 = 1
    var_109 = 2
    var_110 = [var_108, var_109]
    var_111 = 'invalid'
    var_112 = 4
    var_113 = [var_111, var_112]
    var_114 = [var_110, var_113]
    var_115 = var_103.validate(var_114)
    var_116 = module_0.Array()
    var_117 = [var_108, var_58, var_108, var_109]
    var_118 = var_116.validate(var_117)
    var_119 = module_0.Integer()
    var_120 = module_0.Array(var_119, min_items=var_112)
    var_121 = 'invalid'
    var_122 = [var_121]
    var_123 = var_120.validate(var_122)



# Parsed testcases at query #24
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Choice(choices=var_3)
    var_5 = var_4.validate(var_0)
    assert var_5 == 'a'
    var_6 = var_4.validate(var_1)
    assert var_6 == 'b'
    var_7 = var_4.validate(var_2)
    assert var_7 == 'c'
    var_8 = 'Option A'
    var_9 = (var_0, var_8)
    var_10 = 'Option B'
    var_11 = (var_1, var_10)
    var_12 = [var_9, var_11]
    var_13 = module_0.Choice(choices=var_12)
    var_14 = var_13.validate(var_0)
    assert var_14 == 'a'
    var_15 = var_13.validate(var_1)
    assert var_15 == 'b'
    var_16 = [var_0, var_1]
    var_17 = module_0.Choice(choices=var_16)
    var_18 = None
    var_19 = var_17.validate(var_18)
    var_20 = [var_18, var_19]
    var_21 = True
    var_22 = module_0.Choice(choices=var_20)
    var_23 = None
    var_24 = var_22.validate(var_23)
    assert var_24 is None
    var_25 = [var_18, var_19]
    var_26 = module_0.Choice(choices=var_25)
    var_27 = 'c'
    var_28 = var_26.validate(var_27)
    var_29 = [var_27, var_28]
    var_30 = module_0.Choice(choices=var_29)
    var_31 = ''
    var_32 = var_30.validate(var_31)
    var_33 = [var_31, var_32]
    var_34 = module_0.Choice(choices=var_33)
    var_35 = ''
    var_36 = var_34.validate(var_35)
    assert var_36 is None
    var_37 = [var_31, var_32]
    var_38 = False
    var_39 = module_0.Choice(choices=var_37, coerce_types=var_38)
    var_40 = ''
    var_41 = var_39.validate(var_40)
    var_42 = 'A1'
    var_43 = (var_40, var_42)
    var_44 = 'A2'
    var_45 = (var_40, var_44)
    var_46 = 'B'
    var_47 = (var_41, var_46)
    var_48 = [var_43, var_45, var_47]
    var_49 = module_0.Choice(choices=var_48)
    var_50 = var_49.validate(var_40)
    assert var_50 == 'a'
    var_51 = var_49.validate(var_41)
    assert var_51 == 'b'
    var_52 = 'c'
    var_53 = var_49.validate(var_52)



# Parsed testcases at query #25
#--------------------------


import typesystem.base as module_0

def test_case_0():
    var_0 = None
    var_1 = (var_0, var_0)
    var_2 = None
    var_3 = 'valid'
    var_4 = (var_3, var_2)
    var_5 = 'test'
    var_6 = 'type'
    var_7 = module_0.Message(text=var_6, code=var_6)
    var_8 = [var_7]
    var_9 = module_0.ValidationError(messages=var_8)
    var_10 = (var_2, var_9)
    var_11 = (var_3, var_2)
    var_12 = module_0.Message(text=var_6, code=var_6)
    var_13 = [var_12]
    var_14 = module_0.ValidationError(messages=var_13)
    var_15 = (var_2, var_14)
    var_16 = module_0.Message(text=var_6, code=var_6)
    var_17 = [var_16]
    var_18 = module_0.ValidationError(messages=var_17)
    var_19 = (var_2, var_18)
    var_20 = 'test'
    var_21 = 'minimum'
    var_22 = module_0.Message(text=var_21, code=var_21)
    var_23 = [var_22]
    var_24 = module_0.ValidationError(messages=var_23)
    var_25 = (var_20, var_24)
    var_26 = module_0.Message(text=var_6, code=var_6)
    var_27 = [var_26]
    var_28 = module_0.ValidationError(messages=var_27)
    var_29 = (var_20, var_28)
    var_30 = 5
    var_31 = module_0.Message(text=var_21, code=var_21)
    var_32 = [var_31]
    var_33 = module_0.ValidationError(messages=var_32)
    var_34 = (var_30, var_33)
    var_35 = 'maximum'
    var_36 = module_0.Message(text=var_35, code=var_35)
    var_37 = [var_36]
    var_38 = module_0.ValidationError(messages=var_37)
    var_39 = (var_30, var_38)
    var_40 = 5
    var_41 = 'key'
    var_42 = [var_41]
    var_43 = module_0.Message(text=var_6, code=var_6, index=var_42)
    var_44 = [var_43]
    var_45 = module_0.ValidationError(messages=var_44)
    var_46 = (var_40, var_45)
    var_47 = 'test'
    var_48 = module_0.Message(text=var_6, code=var_6)
    var_49 = module_0.Message(text=var_21, code=var_21)
    var_50 = [var_48, var_49]
    var_51 = module_0.ValidationError(messages=var_50)
    var_52 = (var_47, var_51)
    var_53 = 5
    var_54 = (var_53, var_53)



# Parsed testcases at query #26
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Integer()
    var_1 = module_0.String()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = True
    var_5 = module_0.Integer()
    var_6 = [var_5, var_1]
    var_7 = module_0.Union(var_6)
    var_8 = 123
    var_9 = var_3.validate(var_8)
    assert var_9 == 123
    var_10 = 'test'
    var_11 = var_3.validate(var_10)
    assert var_11 == 'test'
    var_12 = None
    var_13 = var_3.validate(var_12)
    var_14 = None
    var_15 = var_7.validate(var_14)
    assert var_15 is None
    var_16 = True
    var_17 = var_3.validate(var_16)
    var_18 = 10
    var_19 = module_0.Integer(minimum=var_18)
    var_20 = 3
    var_21 = module_0.String(max_length=var_20)
    var_22 = [var_19, var_21]
    var_23 = module_0.Union(var_22)
    var_24 = 5
    var_25 = var_23.validate(var_24)
    var_26 = 'toolong'
    var_27 = var_23.validate(var_26)
    var_28 = []
    var_29 = module_0.Union(var_28)
    var_30 = 'anything'
    var_31 = var_29.validate(var_30)



