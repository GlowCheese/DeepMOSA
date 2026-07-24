####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 'value'
    var_9 = 'items'
    var_10 = 1
    var_11 = 2
    var_12 = 3
    var_13 = [var_10, var_11, var_12]
    var_14 = {var_9: var_13}
    var_15 = 0
    var_16 = [var_9, var_15]
    var_17 = module_0.get_in(var_16, var_14)
    assert var_17 == 1
    var_18 = [var_9, var_11]
    var_19 = module_0.get_in(var_18, var_14)
    assert var_19 == 3
    var_20 = 'purchase'
    var_21 = 'costs'
    var_22 = 'Apple'
    var_23 = 'Orange'
    var_24 = [var_22, var_23]
    var_25 = 0.5
    var_26 = 1.25
    var_27 = [var_25, var_26]
    var_28 = {var_9: var_24, var_21: var_27}
    var_29 = {var_20: var_28}
    var_30 = [var_20, var_9, var_15]
    var_31 = module_0.get_in(var_30, var_29)
    assert var_31 == 'Apple'
    var_32 = [var_20, var_9, var_10]
    var_33 = module_0.get_in(var_32, var_29)
    assert var_33 == 'Orange'
    var_34 = [var_20, var_21, var_15]
    var_35 = module_0.get_in(var_34, var_29)
    var_36 = 'name'
    var_37 = 'Alice'
    var_38 = {var_36: var_37}
    var_39 = [var_36]
    var_40 = module_0.get_in(var_39, var_38)
    assert var_40 == 'Alice'
    var_41 = {var_0: var_10}
    var_42 = []
    var_43 = module_0.get_in(var_42, var_41)
    var_44 = {var_1: var_10}
    var_45 = {var_0: var_44}
    var_46 = [var_0, var_2]
    var_47 = module_0.get_in(var_46, var_45)
    assert var_47 is None
    var_48 = 'x'
    var_49 = [var_48]
    var_50 = module_0.get_in(var_49, var_45)
    assert var_50 is None
    var_51 = 'y'
    var_52 = [var_48, var_51]
    var_53 = module_0.get_in(var_52, var_45)
    assert var_53 is None
    var_54 = {var_0: var_10}
    var_55 = [var_1]
    var_56 = module_0.get_in(var_55, var_54, var_15)
    assert var_56 == 0
    var_57 = [var_1]
    var_58 = 'default'
    var_59 = module_0.get_in(var_57, var_54, var_58)
    assert var_59 == 'default'
    var_60 = {var_0: var_10}
    var_61 = 'b'
    var_62 = [var_61]
    var_63 = True
    var_64 = module_0.get_in(var_62, var_60, no_default=var_63)
    var_65 = [var_10, var_11, var_12]
    var_66 = 10
    var_67 = [var_66]
    var_68 = True
    var_69 = module_0.get_in(var_67, var_65, no_default=var_68)
    var_70 = None
    var_71 = {var_66: var_70}
    var_72 = [var_66]
    var_73 = module_0.get_in(var_72, var_71)
    assert var_73 is None
    var_74 = {var_67: var_10}
    var_75 = {var_66: var_74}
    var_76 = [var_66, var_15]
    var_77 = module_0.get_in(var_76, var_75)
    assert var_77 is None
    var_78 = 'a'
    var_79 = 0
    var_80 = [var_78, var_79]
    var_81 = True
    var_82 = module_0.get_in(var_80, var_75, no_default=var_81)
    var_83 = 'users'
    var_84 = 'age'
    var_85 = 30
    var_86 = {var_36: var_37, var_84: var_85}
    var_87 = 'Bob'
    var_88 = 25
    var_89 = {var_36: var_87, var_84: var_88}
    var_90 = [var_86, var_89]
    var_91 = {var_83: var_90}
    var_92 = [var_83, var_15, var_36]
    var_93 = module_0.get_in(var_92, var_91)
    assert var_93 == 'Alice'
    var_94 = [var_83, var_10, var_84]
    var_95 = module_0.get_in(var_94, var_91)
    assert var_95 == 25



# Parsed testcases at query #2
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 'value'
    var_9 = 'items'
    var_10 = 'Apple'
    var_11 = 'Orange'
    var_12 = 'Banana'
    var_13 = [var_10, var_11, var_12]
    var_14 = {var_9: var_13}
    var_15 = 0
    var_16 = [var_9, var_15]
    var_17 = module_0.get_in(var_16, var_14)
    assert var_17 == 'Apple'
    var_18 = 2
    var_19 = [var_9, var_18]
    var_20 = module_0.get_in(var_19, var_14)
    assert var_20 == 'Banana'
    var_21 = 'purchase'
    var_22 = 'costs'
    var_23 = [var_10, var_11]
    var_24 = 0.5
    var_25 = 1.25
    var_26 = [var_24, var_25]
    var_27 = {var_9: var_23, var_22: var_26}
    var_28 = {var_21: var_27}
    var_29 = [var_21, var_9, var_15]
    var_30 = module_0.get_in(var_29, var_28)
    assert var_30 == 'Apple'
    var_31 = 1
    var_32 = [var_21, var_22, var_31]
    var_33 = module_0.get_in(var_32, var_28)
    var_34 = 'name'
    var_35 = 'Alice'
    var_36 = {var_34: var_35}
    var_37 = [var_34]
    var_38 = module_0.get_in(var_37, var_36)
    assert var_38 == 'Alice'
    var_39 = {var_0: var_3}
    var_40 = []
    var_41 = module_0.get_in(var_40, var_39)
    var_42 = {var_0: var_3}
    var_43 = [var_1]
    var_44 = module_0.get_in(var_43, var_42)
    assert var_44 is None
    var_45 = [var_0, var_1]
    var_46 = module_0.get_in(var_45, var_42)
    assert var_46 is None
    var_47 = 'total'
    var_48 = None
    var_49 = {var_47: var_48}
    var_50 = {var_21: var_49}
    var_51 = [var_21, var_47]
    var_52 = module_0.get_in(var_51, var_50, var_15)
    assert var_52 is None
    var_53 = [var_21, var_9]
    var_54 = module_0.get_in(var_53, var_50, var_15)
    assert var_54 == 0
    var_55 = [var_10]
    var_56 = {var_9: var_55}
    var_57 = 10
    var_58 = [var_9, var_57]
    var_59 = module_0.get_in(var_58, var_56)
    assert var_59 is None
    var_60 = [var_9, var_57]
    var_61 = 'default'
    var_62 = module_0.get_in(var_60, var_56, var_61)
    assert var_62 == 'default'
    var_63 = {}
    var_64 = 'y'
    var_65 = [var_64]
    var_66 = True
    var_67 = module_0.get_in(var_65, var_63, no_default=var_66)
    var_68 = 3
    var_69 = [var_31, var_18, var_68]
    var_70 = {var_9: var_69}
    var_71 = 'items'
    var_72 = 10
    var_73 = [var_71, var_72]
    var_74 = True
    var_75 = module_0.get_in(var_73, var_70, no_default=var_74)
    var_76 = {var_72: var_74}
    var_77 = {var_71: var_76}
    var_78 = [var_71, var_72]
    var_79 = True
    var_80 = module_0.get_in(var_78, var_77, no_default=var_79)
    assert var_80 == 'value'
    var_81 = {var_71: var_48}
    var_82 = [var_71, var_72]
    var_83 = module_0.get_in(var_82, var_81)
    assert var_83 is None
    var_84 = [var_71, var_72]
    var_85 = module_0.get_in(var_84, var_81, var_61)
    assert var_85 == 'default'
    var_86 = {var_79: var_74}
    var_87 = {var_15: var_86}
    var_88 = [var_15, var_79]
    var_89 = module_0.get_in(var_88, var_87)
    assert var_89 == 'value'
    var_90 = {var_71: var_74}
    var_91 = 'value2'
    var_92 = {var_72: var_91}
    var_93 = (var_90, var_92)
    var_94 = [var_15, var_71]
    var_95 = module_0.get_in(var_94, var_93)
    assert var_95 == 'value'
    var_96 = [var_79, var_72]
    var_97 = module_0.get_in(var_96, var_93)
    assert var_97 == 'value2'



# Parsed testcases at query #3
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 'value'
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = 4
    var_13 = [var_11, var_12]
    var_14 = [var_10, var_13]
    var_15 = [var_9, var_14]
    var_16 = [var_9, var_9, var_9]
    var_17 = module_0.get_in(var_16, var_15)
    assert var_17 == 4
    var_18 = 'items'
    var_19 = 'costs'
    var_20 = 'Apple'
    var_21 = 'Orange'
    var_22 = [var_20, var_21]
    var_23 = 0.5
    var_24 = 1.25
    var_25 = [var_23, var_24]
    var_26 = {var_18: var_22, var_19: var_25}
    var_27 = 0
    var_28 = [var_18, var_27]
    var_29 = module_0.get_in(var_28, var_26)
    assert var_29 == 'Apple'
    var_30 = [var_19, var_9]
    var_31 = module_0.get_in(var_30, var_26)
    var_32 = 'name'
    var_33 = 'Alice'
    var_34 = {var_32: var_33}
    var_35 = [var_32]
    var_36 = module_0.get_in(var_35, var_34)
    assert var_36 == 'Alice'
    var_37 = {var_0: var_9}
    var_38 = []
    var_39 = module_0.get_in(var_38, var_37)
    var_40 = {var_0: var_9}
    var_41 = [var_1]
    var_42 = module_0.get_in(var_41, var_40)
    assert var_42 is None
    var_43 = {var_0: var_9}
    var_44 = [var_1]
    var_45 = module_0.get_in(var_44, var_43, var_27)
    assert var_45 == 0
    var_46 = [var_1]
    var_47 = 'missing'
    var_48 = module_0.get_in(var_46, var_43, var_47)
    assert var_48 == 'missing'
    var_49 = {var_1: var_9}
    var_50 = {var_0: var_49}
    var_51 = [var_0, var_2]
    var_52 = module_0.get_in(var_51, var_50)
    assert var_52 is None
    var_53 = [var_0, var_2]
    var_54 = 42
    var_55 = module_0.get_in(var_53, var_50, var_54)
    assert var_55 == 42
    var_56 = [var_9, var_10, var_11]
    var_57 = 10
    var_58 = [var_57]
    var_59 = module_0.get_in(var_58, var_56)
    assert var_59 is None
    var_60 = [var_57]
    var_61 = 'out of range'
    var_62 = module_0.get_in(var_60, var_56, var_61)
    assert var_62 == 'out of range'
    var_63 = {var_0: var_9}
    var_64 = 'b'
    var_65 = [var_64]
    var_66 = True
    var_67 = module_0.get_in(var_65, var_63, no_default=var_66)
    var_68 = [var_9, var_10, var_11]
    var_69 = 10
    var_70 = [var_69]
    var_71 = True
    var_72 = module_0.get_in(var_70, var_68, no_default=var_71)
    var_73 = {var_70: var_9}
    var_74 = {var_69: var_73}
    var_75 = 'a'
    var_76 = 'c'
    var_77 = [var_75, var_76]
    var_78 = True
    var_79 = module_0.get_in(var_77, var_74, no_default=var_78)
    var_80 = None
    var_81 = {var_75: var_80}
    var_82 = [var_75]
    var_83 = module_0.get_in(var_82, var_81)
    assert var_83 is None
    var_84 = {var_75: var_9}
    var_85 = [var_75, var_27]
    var_86 = module_0.get_in(var_85, var_84)
    assert var_86 is None
    var_87 = [var_75, var_27]
    var_88 = 'error'
    var_89 = module_0.get_in(var_87, var_84, var_88)
    assert var_89 == 'error'
    var_90 = [var_9, var_10, var_11]
    var_91 = 'key'
    var_92 = [var_27, var_91]
    var_93 = module_0.get_in(var_92, var_90)
    assert var_93 is None
    var_94 = [var_27, var_91]
    var_95 = module_0.get_in(var_94, var_90, var_88)
    assert var_95 == 'error'
    var_96 = 'purchase'
    var_97 = 'credit card'
    var_98 = [var_20, var_21]
    var_99 = [var_23, var_24]
    var_100 = {var_18: var_98, var_19: var_99}
    var_101 = '5555-1234-1234-1234'
    var_102 = {var_32: var_33, var_96: var_100, var_97: var_101}
    var_103 = [var_96, var_18, var_27]
    var_104 = module_0.get_in(var_103, var_102)
    assert var_104 == 'Apple'
    var_105 = [var_32]
    var_106 = module_0.get_in(var_105, var_102)
    assert var_106 == 'Alice'
    var_107 = 'total'
    var_108 = [var_96, var_107]
    var_109 = module_0.get_in(var_108, var_102)
    assert var_109 is None
    var_110 = 'apple'
    var_111 = [var_96, var_18, var_110]
    var_112 = module_0.get_in(var_111, var_102)
    assert var_112 is None
    var_113 = [var_96, var_18, var_57]
    var_114 = module_0.get_in(var_113, var_102)
    assert var_114 is None
    var_115 = [var_96, var_107]
    var_116 = module_0.get_in(var_115, var_102, var_27)
    assert var_116 == 0



# Parsed testcases at query #4
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 'value'
    var_9 = 1
    var_10 = 2
    var_11 = [var_9, var_10]
    var_12 = 3
    var_13 = 4
    var_14 = [var_12, var_13]
    var_15 = [var_11, var_14]
    var_16 = 0
    var_17 = [var_16, var_9]
    var_18 = module_0.get_in(var_17, var_15)
    assert var_18 == 2
    var_19 = [var_9, var_16]
    var_20 = module_0.get_in(var_19, var_15)
    assert var_20 == 3
    var_21 = 'items'
    var_22 = 'name'
    var_23 = 'Apple'
    var_24 = {var_22: var_23}
    var_25 = 'Orange'
    var_26 = {var_22: var_25}
    var_27 = [var_24, var_26]
    var_28 = {var_21: var_27}
    var_29 = [var_21, var_16, var_22]
    var_30 = module_0.get_in(var_29, var_28)
    assert var_30 == 'Apple'
    var_31 = [var_21, var_9, var_22]
    var_32 = module_0.get_in(var_31, var_28)
    assert var_32 == 'Orange'
    var_33 = 'Alice'
    var_34 = {var_22: var_33}
    var_35 = [var_22]
    var_36 = module_0.get_in(var_35, var_34)
    assert var_36 == 'Alice'
    var_37 = {var_0: var_9}
    var_38 = []
    var_39 = module_0.get_in(var_38, var_37)
    var_40 = {var_1: var_9}
    var_41 = {var_0: var_40}
    var_42 = [var_0, var_2]
    var_43 = module_0.get_in(var_42, var_41)
    assert var_43 is None
    var_44 = 'x'
    var_45 = 'y'
    var_46 = 'z'
    var_47 = [var_44, var_45, var_46]
    var_48 = module_0.get_in(var_47, var_41)
    assert var_48 is None
    var_49 = {var_0: var_9}
    var_50 = [var_1]
    var_51 = module_0.get_in(var_50, var_49, var_16)
    assert var_51 == 0
    var_52 = [var_44, var_45]
    var_53 = 'not found'
    var_54 = module_0.get_in(var_52, var_49, var_53)
    assert var_54 == 'not found'
    var_55 = [var_9, var_10, var_12]
    var_56 = 10
    var_57 = [var_56]
    var_58 = module_0.get_in(var_57, var_55)
    assert var_58 is None
    var_59 = [var_56]
    var_60 = 'default'
    var_61 = module_0.get_in(var_59, var_55, var_60)
    assert var_61 == 'default'
    var_62 = {var_0: var_9}
    var_63 = 'b'
    var_64 = [var_63]
    var_65 = True
    var_66 = module_0.get_in(var_64, var_62, no_default=var_65)
    var_67 = [var_9, var_10, var_12]
    var_68 = 10
    var_69 = [var_68]
    var_70 = True
    var_71 = module_0.get_in(var_69, var_67, no_default=var_70)
    var_72 = 'string_value'
    var_73 = {var_68: var_72}
    var_74 = [var_68, var_69]
    var_75 = module_0.get_in(var_74, var_73)
    assert var_75 is None
    var_76 = [var_68, var_16]
    var_77 = module_0.get_in(var_76, var_73)
    assert var_77 is None
    var_78 = {var_68: var_72}
    var_79 = 'a'
    var_80 = 'b'
    var_81 = [var_79, var_80]
    var_82 = True
    var_83 = module_0.get_in(var_81, var_78, no_default=var_82)
    var_84 = 'users'
    var_85 = 'scores'
    var_86 = 20
    var_87 = 30
    var_88 = [var_56, var_86, var_87]
    var_89 = {var_22: var_33, var_85: var_88}
    var_90 = 'Bob'
    var_91 = 15
    var_92 = 25
    var_93 = 35
    var_94 = [var_91, var_92, var_93]
    var_95 = {var_22: var_90, var_85: var_94}
    var_96 = [var_89, var_95]
    var_97 = {var_84: var_96}
    var_98 = [var_84, var_16, var_22]
    var_99 = module_0.get_in(var_98, var_97)
    assert var_99 == 'Alice'
    var_100 = [var_84, var_9, var_85, var_10]
    var_101 = module_0.get_in(var_100, var_97)
    assert var_101 == 35
    var_102 = [var_84, var_16, var_85, var_16]
    var_103 = module_0.get_in(var_102, var_97)
    assert var_103 == 10
    var_104 = None
    var_105 = {var_80: var_104}
    var_106 = {var_79: var_105}
    var_107 = [var_79, var_80]
    var_108 = module_0.get_in(var_107, var_106)
    assert var_108 is None
    var_109 = [var_79]
    var_110 = {}
    var_111 = module_0.get_in(var_109, var_110)
    assert var_111 is None
    var_112 = [var_16]
    var_113 = []
    var_114 = module_0.get_in(var_112, var_113)
    assert var_114 is None
    var_115 = [var_79]
    var_116 = {}
    var_117 = 'empty'
    var_118 = module_0.get_in(var_115, var_116, var_117)
    assert var_118 == 'empty'



# Parsed testcases at query #5
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 'value'
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = 4
    var_13 = [var_11, var_12]
    var_14 = [var_10, var_13]
    var_15 = [var_9, var_14]
    var_16 = [var_9, var_9, var_9]
    var_17 = module_0.get_in(var_16, var_15)
    assert var_17 == 4
    var_18 = 'items'
    var_19 = 'name'
    var_20 = 'Apple'
    var_21 = {var_19: var_20}
    var_22 = 'Orange'
    var_23 = {var_19: var_22}
    var_24 = [var_21, var_23]
    var_25 = {var_18: var_24}
    var_26 = 0
    var_27 = [var_18, var_26, var_19]
    var_28 = module_0.get_in(var_27, var_25)
    assert var_28 == 'Apple'
    var_29 = [var_18, var_9, var_19]
    var_30 = module_0.get_in(var_29, var_25)
    assert var_30 == 'Orange'
    var_31 = 'Alice'
    var_32 = {var_19: var_31}
    var_33 = [var_19]
    var_34 = module_0.get_in(var_33, var_32)
    assert var_34 == 'Alice'
    var_35 = {var_0: var_3}
    var_36 = []
    var_37 = module_0.get_in(var_36, var_35)
    var_38 = {var_0: var_3}
    var_39 = [var_1]
    var_40 = module_0.get_in(var_39, var_38)
    assert var_40 is None
    var_41 = [var_1]
    var_42 = 'default'
    var_43 = module_0.get_in(var_41, var_38, var_42)
    assert var_43 == 'default'
    var_44 = [var_9, var_10, var_11]
    var_45 = 10
    var_46 = [var_45]
    var_47 = module_0.get_in(var_46, var_44)
    assert var_47 is None
    var_48 = [var_45]
    var_49 = module_0.get_in(var_48, var_44, var_42)
    assert var_49 == 'default'
    var_50 = {var_0: var_3}
    var_51 = 'b'
    var_52 = [var_51]
    var_53 = True
    var_54 = module_0.get_in(var_52, var_50, no_default=var_53)
    var_55 = [var_9, var_10, var_11]
    var_56 = 10
    var_57 = [var_56]
    var_58 = True
    var_59 = module_0.get_in(var_57, var_55, no_default=var_58)
    var_60 = 'string_value'
    var_61 = {var_56: var_60}
    var_62 = 'a'
    var_63 = 'b'
    var_64 = [var_62, var_63]
    var_65 = True
    var_66 = module_0.get_in(var_64, var_61, no_default=var_65)
    var_67 = None
    var_68 = {var_62: var_67}
    var_69 = [var_62]
    var_70 = module_0.get_in(var_69, var_68)
    assert var_70 is None
    var_71 = {var_62: var_67}
    var_72 = [var_62, var_63]
    var_73 = module_0.get_in(var_72, var_71)
    assert var_73 is None
    var_74 = {}
    var_75 = 'missing'
    var_76 = [var_75]
    var_77 = 42
    var_78 = module_0.get_in(var_76, var_74, var_77)
    assert var_78 == 42
    var_79 = {var_63: var_65}
    var_80 = {var_62: var_79}
    var_81 = [var_62, var_64]
    var_82 = 'not_found'
    var_83 = module_0.get_in(var_81, var_80, var_82)
    assert var_83 == 'not_found'
    var_84 = {var_9: var_65}
    var_85 = {var_26: var_84}
    var_86 = [var_26, var_9]
    var_87 = module_0.get_in(var_86, var_85)
    assert var_87 == 'value'



# Parsed testcases at query #6
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 'value'
    var_9 = 'items'
    var_10 = 'Apple'
    var_11 = 'Orange'
    var_12 = 'Banana'
    var_13 = [var_10, var_11, var_12]
    var_14 = {var_9: var_13}
    var_15 = 0
    var_16 = [var_9, var_15]
    var_17 = module_0.get_in(var_16, var_14)
    assert var_17 == 'Apple'
    var_18 = 1
    var_19 = [var_9, var_18]
    var_20 = module_0.get_in(var_19, var_14)
    assert var_20 == 'Orange'
    var_21 = 2
    var_22 = [var_9, var_21]
    var_23 = module_0.get_in(var_22, var_14)
    assert var_23 == 'Banana'
    var_24 = 'purchase'
    var_25 = 'costs'
    var_26 = [var_10, var_11]
    var_27 = 0.5
    var_28 = 1.25
    var_29 = [var_27, var_28]
    var_30 = {var_9: var_26, var_25: var_29}
    var_31 = {var_24: var_30}
    var_32 = [var_24, var_9, var_15]
    var_33 = module_0.get_in(var_32, var_31)
    assert var_33 == 'Apple'
    var_34 = [var_24, var_25, var_18]
    var_35 = module_0.get_in(var_34, var_31)
    var_36 = 'name'
    var_37 = 'Alice'
    var_38 = {var_36: var_37}
    var_39 = [var_36]
    var_40 = module_0.get_in(var_39, var_38)
    assert var_40 == 'Alice'
    var_41 = {var_0: var_3}
    var_42 = []
    var_43 = module_0.get_in(var_42, var_41)
    var_44 = {var_1: var_3}
    var_45 = {var_0: var_44}
    var_46 = [var_0, var_2]
    var_47 = module_0.get_in(var_46, var_45)
    assert var_47 is None
    var_48 = 'x'
    var_49 = 'y'
    var_50 = [var_48, var_49]
    var_51 = module_0.get_in(var_50, var_45)
    assert var_51 is None
    var_52 = {var_1: var_3}
    var_53 = {var_0: var_52}
    var_54 = [var_0, var_2]
    var_55 = module_0.get_in(var_54, var_53, var_15)
    assert var_55 == 0
    var_56 = [var_48]
    var_57 = 'default'
    var_58 = module_0.get_in(var_56, var_53, var_57)
    assert var_58 == 'default'
    var_59 = [var_10, var_11]
    var_60 = {var_9: var_59}
    var_61 = 10
    var_62 = [var_9, var_61]
    var_63 = module_0.get_in(var_62, var_60)
    assert var_63 is None
    var_64 = [var_9, var_61]
    var_65 = 'missing'
    var_66 = module_0.get_in(var_64, var_60, var_65)
    assert var_66 == 'missing'
    var_67 = {var_0: var_3}
    var_68 = 'b'
    var_69 = [var_68]
    var_70 = True
    var_71 = module_0.get_in(var_69, var_67, no_default=var_70)
    var_72 = [var_10]
    var_73 = {var_9: var_72}
    var_74 = 'items'
    var_75 = 10
    var_76 = [var_74, var_75]
    var_77 = True
    var_78 = module_0.get_in(var_76, var_73, no_default=var_77)
    var_79 = {var_75: var_77}
    var_80 = {var_74: var_79}
    var_81 = 'a'
    var_82 = 'c'
    var_83 = [var_81, var_82]
    var_84 = True
    var_85 = module_0.get_in(var_83, var_80, no_default=var_84)
    var_86 = 'string'
    var_87 = {var_81: var_86}
    var_88 = [var_81, var_82]
    var_89 = module_0.get_in(var_88, var_87)
    assert var_89 is None
    var_90 = {var_81: var_86}
    var_91 = 'a'
    var_92 = 'b'
    var_93 = [var_91, var_92]
    var_94 = True
    var_95 = module_0.get_in(var_93, var_90, no_default=var_94)
    var_96 = [var_18, var_21]
    var_97 = 3
    var_98 = 4
    var_99 = [var_97, var_98]
    var_100 = 5
    var_101 = 6
    var_102 = [var_100, var_101]
    var_103 = [var_96, var_99, var_102]
    var_104 = [var_15, var_18]
    var_105 = module_0.get_in(var_104, var_103)
    assert var_105 == 2
    var_106 = [var_21, var_15]
    var_107 = module_0.get_in(var_106, var_103)
    assert var_107 == 5
    var_108 = 'level1'
    var_109 = 'level2'
    var_110 = 'level3'
    var_111 = 'level4'
    var_112 = 'deep'
    var_113 = {var_111: var_112}
    var_114 = {var_110: var_113}
    var_115 = {var_109: var_114}
    var_116 = {var_108: var_115}
    var_117 = [var_108, var_109, var_110, var_111]
    var_118 = module_0.get_in(var_117, var_116)
    assert var_118 == 'deep'



# Parsed testcases at query #7
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 'value'
    var_9 = 'items'
    var_10 = 1
    var_11 = 2
    var_12 = 3
    var_13 = [var_10, var_11, var_12]
    var_14 = {var_9: var_13}
    var_15 = 0
    var_16 = [var_9, var_15]
    var_17 = module_0.get_in(var_16, var_14)
    assert var_17 == 1
    var_18 = [var_9, var_11]
    var_19 = module_0.get_in(var_18, var_14)
    assert var_19 == 3
    var_20 = 'purchase'
    var_21 = 'costs'
    var_22 = 'Apple'
    var_23 = 'Orange'
    var_24 = [var_22, var_23]
    var_25 = 0.5
    var_26 = 1.25
    var_27 = [var_25, var_26]
    var_28 = {var_9: var_24, var_21: var_27}
    var_29 = {var_20: var_28}
    var_30 = [var_20, var_9, var_15]
    var_31 = module_0.get_in(var_30, var_29)
    assert var_31 == 'Apple'
    var_32 = [var_20, var_9, var_10]
    var_33 = module_0.get_in(var_32, var_29)
    assert var_33 == 'Orange'
    var_34 = [var_20, var_21, var_15]
    var_35 = module_0.get_in(var_34, var_29)
    var_36 = 'name'
    var_37 = 'Alice'
    var_38 = {var_36: var_37}
    var_39 = [var_36]
    var_40 = module_0.get_in(var_39, var_38)
    assert var_40 == 'Alice'
    var_41 = {var_0: var_10}
    var_42 = []
    var_43 = module_0.get_in(var_42, var_41)
    var_44 = {var_1: var_10}
    var_45 = {var_0: var_44}
    var_46 = [var_0, var_2]
    var_47 = module_0.get_in(var_46, var_45)
    assert var_47 is None
    var_48 = {var_1: var_10}
    var_49 = {var_0: var_48}
    var_50 = [var_0, var_2]
    var_51 = module_0.get_in(var_50, var_49, var_15)
    assert var_51 == 0
    var_52 = 'x'
    var_53 = 'y'
    var_54 = [var_52, var_53]
    var_55 = 'missing'
    var_56 = module_0.get_in(var_54, var_49, var_55)
    assert var_56 == 'missing'
    var_57 = [var_10, var_11, var_12]
    var_58 = {var_9: var_57}
    var_59 = 10
    var_60 = [var_9, var_59]
    var_61 = module_0.get_in(var_60, var_58)
    assert var_61 is None
    var_62 = [var_10, var_11, var_12]
    var_63 = {var_9: var_62}
    var_64 = [var_9, var_59]
    var_65 = -1
    var_66 = module_0.get_in(var_64, var_63, var_65)
    assert var_66 == -1
    var_67 = {}
    var_68 = {var_0: var_67}
    var_69 = 'a'
    var_70 = 'missing'
    var_71 = [var_69, var_70]
    var_72 = True
    var_73 = module_0.get_in(var_71, var_68, no_default=var_72)
    var_74 = [var_10, var_11]
    var_75 = {var_9: var_74}
    var_76 = 'items'
    var_77 = 10
    var_78 = [var_76, var_77]
    var_79 = True
    var_80 = module_0.get_in(var_78, var_75, no_default=var_79)
    var_81 = None
    var_82 = {var_76: var_81}
    var_83 = [var_76]
    var_84 = module_0.get_in(var_83, var_82)
    assert var_84 is None
    var_85 = 'string'
    var_86 = {var_76: var_85}
    var_87 = [var_76, var_15]
    var_88 = module_0.get_in(var_87, var_86)
    assert var_88 is None
    var_89 = {var_76: var_85}
    var_90 = 'a'
    var_91 = 0
    var_92 = [var_90, var_91]
    var_93 = True
    var_94 = module_0.get_in(var_92, var_89, no_default=var_93)
    var_95 = 'level1'
    var_96 = 'level2'
    var_97 = 'level3'
    var_98 = 'level4'
    var_99 = 'deep'
    var_100 = {var_98: var_99}
    var_101 = {var_97: var_100}
    var_102 = {var_96: var_101}
    var_103 = {var_95: var_102}
    var_104 = [var_95, var_96, var_97, var_98]
    var_105 = module_0.get_in(var_104, var_103)
    assert var_105 == 'deep'
    var_106 = {var_36: var_37}
    var_107 = 'Bob'
    var_108 = {var_36: var_107}
    var_109 = [var_106, var_108]
    var_110 = [var_15, var_36]
    var_111 = module_0.get_in(var_110, var_109)
    assert var_111 == 'Alice'
    var_112 = [var_10, var_36]
    var_113 = module_0.get_in(var_112, var_109)
    assert var_113 == 'Bob'
    var_114 = {}
    var_115 = {var_90: var_114}
    var_116 = [var_90, var_91]
    var_117 = module_0.get_in(var_116, var_115, var_15)
    assert var_117 == 0
    var_118 = {}
    var_119 = {var_90: var_118}
    var_120 = [var_90, var_91]
    var_121 = False
    var_122 = module_0.get_in(var_120, var_119, var_121)
    assert var_122 is False



# Parsed testcases at query #8
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 'value'
    var_9 = 'items'
    var_10 = 1
    var_11 = 2
    var_12 = 3
    var_13 = [var_10, var_11, var_12]
    var_14 = {var_9: var_13}
    var_15 = 0
    var_16 = [var_9, var_15]
    var_17 = module_0.get_in(var_16, var_14)
    assert var_17 == 1
    var_18 = [var_9, var_11]
    var_19 = module_0.get_in(var_18, var_14)
    assert var_19 == 3
    var_20 = 'purchase'
    var_21 = 'costs'
    var_22 = 'Apple'
    var_23 = 'Orange'
    var_24 = [var_22, var_23]
    var_25 = 0.5
    var_26 = 1.25
    var_27 = [var_25, var_26]
    var_28 = {var_9: var_24, var_21: var_27}
    var_29 = {var_20: var_28}
    var_30 = [var_20, var_9, var_15]
    var_31 = module_0.get_in(var_30, var_29)
    assert var_31 == 'Apple'
    var_32 = [var_20, var_9, var_10]
    var_33 = module_0.get_in(var_32, var_29)
    assert var_33 == 'Orange'
    var_34 = [var_20, var_21, var_10]
    var_35 = module_0.get_in(var_34, var_29)
    var_36 = 'name'
    var_37 = 'Alice'
    var_38 = {var_36: var_37}
    var_39 = [var_36]
    var_40 = module_0.get_in(var_39, var_38)
    assert var_40 == 'Alice'
    var_41 = {var_0: var_10}
    var_42 = []
    var_43 = module_0.get_in(var_42, var_41)
    var_44 = {var_1: var_10}
    var_45 = {var_0: var_44}
    var_46 = 'missing'
    var_47 = [var_0, var_46]
    var_48 = module_0.get_in(var_47, var_45)
    assert var_48 is None
    var_49 = {var_1: var_10}
    var_50 = {var_0: var_49}
    var_51 = [var_0, var_46]
    var_52 = module_0.get_in(var_51, var_50, var_15)
    assert var_52 == 0
    var_53 = [var_46]
    var_54 = 'not found'
    var_55 = module_0.get_in(var_53, var_50, var_54)
    assert var_55 == 'not found'
    var_56 = [var_10, var_11, var_12]
    var_57 = {var_9: var_56}
    var_58 = 10
    var_59 = [var_9, var_58]
    var_60 = module_0.get_in(var_59, var_57)
    assert var_60 is None
    var_61 = [var_9, var_58]
    var_62 = -1
    var_63 = module_0.get_in(var_61, var_57, var_62)
    assert var_63 == -1
    var_64 = {var_0: var_10}
    var_65 = 'missing'
    var_66 = [var_65]
    var_67 = True
    var_68 = module_0.get_in(var_66, var_64, no_default=var_67)
    var_69 = [var_10, var_11, var_12]
    var_70 = {var_9: var_69}
    var_71 = 'items'
    var_72 = 10
    var_73 = [var_71, var_72]
    var_74 = True
    var_75 = module_0.get_in(var_73, var_70, no_default=var_74)
    var_76 = 42
    var_77 = {var_71: var_76}
    var_78 = [var_71, var_72]
    var_79 = module_0.get_in(var_78, var_77)
    assert var_79 is None
    var_80 = {var_71: var_76}
    var_81 = 'a'
    var_82 = 'b'
    var_83 = [var_81, var_82]
    var_84 = True
    var_85 = module_0.get_in(var_83, var_80, no_default=var_84)
    var_86 = [var_10, var_11]
    var_87 = 4
    var_88 = [var_12, var_87]
    var_89 = [var_86, var_88]
    var_90 = [var_15, var_10]
    var_91 = module_0.get_in(var_90, var_89)
    assert var_91 == 2
    var_92 = [var_10, var_15]
    var_93 = module_0.get_in(var_92, var_89)
    assert var_93 == 3
    var_94 = 'users'
    var_95 = 'age'
    var_96 = 30
    var_97 = {var_36: var_37, var_95: var_96}
    var_98 = 'Bob'
    var_99 = 25
    var_100 = {var_36: var_98, var_95: var_99}
    var_101 = [var_97, var_100]
    var_102 = {var_94: var_101}
    var_103 = [var_94, var_15, var_36]
    var_104 = module_0.get_in(var_103, var_102)
    assert var_104 == 'Alice'
    var_105 = [var_94, var_10, var_95]
    var_106 = module_0.get_in(var_105, var_102)
    assert var_106 == 25
    var_107 = [var_94, var_11, var_36]
    var_108 = module_0.get_in(var_107, var_102)
    assert var_108 is None



