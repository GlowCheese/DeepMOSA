####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 42
    var_9 = 'items'
    var_10 = 'name'
    var_11 = 'apple'
    var_12 = {var_10: var_11}
    var_13 = 'orange'
    var_14 = {var_10: var_13}
    var_15 = [var_12, var_14]
    var_16 = {var_9: var_15}
    var_17 = 0
    var_18 = [var_9, var_17, var_10]
    var_19 = module_0.get_in(var_18, var_16)
    assert var_19 == 'apple'
    var_20 = 1
    var_21 = {var_0: var_20}
    var_22 = [var_1]
    var_23 = module_0.get_in(var_22, var_21)
    assert var_23 is None
    var_24 = [var_1]
    var_25 = 'missing'
    var_26 = module_0.get_in(var_24, var_21, var_25)
    assert var_26 == 'missing'
    var_27 = {var_0: var_20}
    var_28 = 'b'
    var_29 = [var_28]
    var_30 = True
    var_31 = module_0.get_in(var_29, var_27, no_default=var_30)
    var_32 = 2
    var_33 = 3
    var_34 = [var_20, var_32, var_33]
    var_35 = 5
    var_36 = [var_35]
    var_37 = True
    var_38 = module_0.get_in(var_36, var_34, no_default=var_37)
    var_39 = 'users'
    var_40 = 'scores'
    var_41 = 'Alice'
    var_42 = 85
    var_43 = 90
    var_44 = [var_42, var_43]
    var_45 = {var_10: var_41, var_40: var_44}
    var_46 = [var_45]
    var_47 = {var_39: var_46}
    var_48 = [var_39, var_17, var_40, var_20]
    var_49 = module_0.get_in(var_48, var_47)
    assert var_49 == 90
    var_50 = {var_35: var_20}
    var_51 = []
    var_52 = module_0.get_in(var_51, var_50)
    var_53 = [var_35]
    var_54 = {}
    var_55 = module_0.get_in(var_53, var_54)
    assert var_55 is None
    var_56 = [var_35]
    var_57 = {}
    var_58 = module_0.get_in(var_56, var_57, var_17)
    assert var_58 == 0
    var_59 = None
    var_60 = {var_35: var_59}
    var_61 = [var_35]
    var_62 = module_0.get_in(var_61, var_60)
    assert var_62 is None
    var_63 = [var_35, var_36]
    var_64 = module_0.get_in(var_63, var_60)
    assert var_64 is None
    var_65 = 'deep'
    var_66 = {var_33: var_65}
    var_67 = {var_32: var_66}
    var_68 = {var_20: var_67}
    var_69 = [var_20, var_32, var_33]
    var_70 = module_0.get_in(var_69, var_68)
    assert var_70 == 'deep'



# Parsed testcases at query #2
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 42
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = [var_0]
    var_12 = module_0.get_in(var_11, var_6)
    var_13 = 1
    var_14 = 2
    var_15 = 3
    var_16 = [var_13, var_14, var_15]
    var_17 = 4
    var_18 = 5
    var_19 = 6
    var_20 = [var_17, var_18, var_19]
    var_21 = [var_16, var_20]
    var_22 = 0
    var_23 = [var_22, var_13]
    var_24 = module_0.get_in(var_23, var_21)
    assert var_24 == 2
    var_25 = [var_13, var_14]
    var_26 = module_0.get_in(var_25, var_21)
    assert var_26 == 6
    var_27 = {var_1: var_13}
    var_28 = {var_2: var_14}
    var_29 = [var_27, var_28]
    var_30 = {var_0: var_29}
    var_31 = [var_0, var_22, var_1]
    var_32 = module_0.get_in(var_31, var_30)
    assert var_32 == 1
    var_33 = [var_0, var_13, var_2]
    var_34 = module_0.get_in(var_33, var_30)
    assert var_34 == 2
    var_35 = {var_0: var_13}
    var_36 = [var_1]
    var_37 = module_0.get_in(var_36, var_35)
    assert var_37 is None
    var_38 = [var_1]
    var_39 = 'missing'
    var_40 = module_0.get_in(var_38, var_35, var_39)
    assert var_40 == 'missing'
    var_41 = [var_1]
    var_42 = module_0.get_in(var_41, var_35, var_22)
    assert var_42 == 0
    var_43 = {var_0: var_13}
    var_44 = 'b'
    var_45 = [var_44]
    var_46 = True
    var_47 = module_0.get_in(var_45, var_43, no_default=var_46)
    var_48 = [var_13, var_14, var_15]
    var_49 = 5
    var_50 = [var_49]
    var_51 = True
    var_52 = module_0.get_in(var_50, var_48, no_default=var_51)
    var_53 = {var_49: var_13}
    var_54 = []
    var_55 = module_0.get_in(var_54, var_53)
    var_56 = []
    var_57 = 'default'
    var_58 = module_0.get_in(var_56, var_53, var_57)
    var_59 = {}
    var_60 = {var_50: var_59}
    var_61 = {var_49: var_60}
    var_62 = [var_49, var_50, var_51]
    var_63 = module_0.get_in(var_62, var_61)
    assert var_63 is None
    var_64 = [var_49, var_50, var_51]
    var_65 = module_0.get_in(var_64, var_61, var_22)
    assert var_65 == 0
    var_66 = None
    var_67 = {var_49: var_66}
    var_68 = [var_49]
    var_69 = module_0.get_in(var_68, var_67)
    assert var_69 is None
    var_70 = [var_49, var_50]
    var_71 = module_0.get_in(var_70, var_67)
    assert var_71 is None
    var_72 = 'key'
    var_73 = [var_72]
    var_74 = {}
    var_75 = module_0.get_in(var_73, var_74)
    assert var_75 is None
    var_76 = [var_22]
    var_77 = []
    var_78 = module_0.get_in(var_76, var_77)
    assert var_78 is None
    var_79 = 'users'
    var_80 = 'name'
    var_81 = 'scores'
    var_82 = 'Alice'
    var_83 = 85
    var_84 = 92
    var_85 = 78
    var_86 = [var_83, var_84, var_85]
    var_87 = {var_80: var_82, var_81: var_86}
    var_88 = 'Bob'
    var_89 = 88
    var_90 = 95
    var_91 = 82
    var_92 = [var_89, var_90, var_91]
    var_93 = {var_80: var_88, var_81: var_92}
    var_94 = [var_87, var_93]
    var_95 = {var_79: var_94}
    var_96 = [var_79, var_22, var_80]
    var_97 = module_0.get_in(var_96, var_95)
    assert var_97 == 'Alice'
    var_98 = [var_79, var_13, var_81, var_14]
    var_99 = module_0.get_in(var_98, var_95)
    assert var_99 == 82
    var_100 = [var_79, var_14, var_80]
    var_101 = module_0.get_in(var_100, var_95)
    assert var_101 is None
    var_102 = [var_79, var_22, var_81, var_18]
    var_103 = module_0.get_in(var_102, var_95)
    assert var_103 is None



# Parsed testcases at query #3
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = 1
    var_3 = {var_0: var_2}
    var_4 = module_0.get_in(var_1, var_3)
    assert var_4 == 1
    var_5 = 'b'
    var_6 = [var_0, var_5]
    var_7 = 2
    var_8 = {var_5: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.get_in(var_6, var_9)
    assert var_10 == 2
    var_11 = 'c'
    var_12 = [var_0, var_5, var_11]
    var_13 = 3
    var_14 = {var_11: var_13}
    var_15 = {var_5: var_14}
    var_16 = {var_0: var_15}
    var_17 = module_0.get_in(var_12, var_16)
    assert var_17 == 3
    var_18 = 0
    var_19 = [var_18]
    var_20 = 10
    var_21 = [var_20]
    var_22 = module_0.get_in(var_19, var_21)
    assert var_22 == 10
    var_23 = [var_18, var_2]
    var_24 = [var_2, var_7]
    var_25 = [var_24]
    var_26 = module_0.get_in(var_23, var_25)
    assert var_26 == 2
    var_27 = [var_18, var_2, var_7]
    var_28 = [var_2, var_7, var_13]
    var_29 = [var_28]
    var_30 = [var_29]
    var_31 = module_0.get_in(var_27, var_30)
    assert var_31 == 3
    var_32 = [var_0, var_18]
    var_33 = [var_2, var_7, var_13]
    var_34 = {var_0: var_33}
    var_35 = module_0.get_in(var_32, var_34)
    assert var_35 == 1
    var_36 = [var_18, var_5]
    var_37 = 5
    var_38 = {var_5: var_37}
    var_39 = [var_38]
    var_40 = module_0.get_in(var_36, var_39)
    assert var_40 == 5
    var_41 = 'x'
    var_42 = [var_41]
    var_43 = {var_0: var_2}
    var_44 = module_0.get_in(var_42, var_43)
    assert var_44 is None
    var_45 = [var_41]
    var_46 = {var_0: var_2}
    var_47 = 'missing'
    var_48 = module_0.get_in(var_45, var_46, var_47)
    assert var_48 == 'missing'
    var_49 = [var_0, var_41]
    var_50 = {var_5: var_7}
    var_51 = {var_0: var_50}
    var_52 = module_0.get_in(var_49, var_51, var_18)
    assert var_52 == 0
    var_53 = [var_37]
    var_54 = [var_2, var_7, var_13]
    var_55 = 'out of range'
    var_56 = module_0.get_in(var_53, var_54, var_55)
    assert var_56 == 'out of range'
    var_57 = 'x'
    var_58 = [var_57]
    var_59 = 'a'
    var_60 = 1
    var_61 = {var_59: var_60}
    var_62 = True
    var_63 = module_0.get_in(var_58, var_61, no_default=var_62)
    var_64 = 5
    var_65 = [var_64]
    var_66 = 1
    var_67 = 2
    var_68 = 3
    var_69 = [var_66, var_67, var_68]
    var_70 = True
    var_71 = module_0.get_in(var_65, var_69, no_default=var_70)
    var_72 = 'a'
    var_73 = 'b'
    var_74 = [var_72, var_73]
    var_75 = 1
    var_76 = {var_72: var_75}
    var_77 = True
    var_78 = module_0.get_in(var_74, var_76, no_default=var_77)
    var_79 = []
    var_80 = {var_72: var_74}
    var_81 = module_0.get_in(var_79, var_80)
    var_82 = []
    var_83 = [var_74, var_71, var_13]
    var_84 = module_0.get_in(var_82, var_83)
    var_85 = []
    var_86 = 'hello'
    var_87 = module_0.get_in(var_85, var_86)
    assert var_87 == 'hello'
    var_88 = []
    var_89 = {var_72: var_74}
    var_90 = 'default'
    var_91 = module_0.get_in(var_88, var_89, var_90)
    var_92 = 'users'
    var_93 = 'name'
    var_94 = 'scores'
    var_95 = 'Alice'
    var_96 = 85
    var_97 = 92
    var_98 = 78
    var_99 = [var_96, var_97, var_98]
    var_100 = {var_93: var_95, var_94: var_99}
    var_101 = 'Bob'
    var_102 = 76
    var_103 = 88
    var_104 = 95
    var_105 = [var_102, var_103, var_104]
    var_106 = {var_93: var_101, var_94: var_105}
    var_107 = [var_100, var_106]
    var_108 = {var_92: var_107}
    var_109 = [var_92, var_18, var_93]
    var_110 = module_0.get_in(var_109, var_108)
    assert var_110 == 'Alice'
    var_111 = [var_92, var_74, var_94, var_71]
    var_112 = module_0.get_in(var_111, var_108)
    assert var_112 == 95
    var_113 = [var_92, var_18, var_94, var_37]
    var_114 = module_0.get_in(var_113, var_108, var_18)
    assert var_114 == 0
    var_115 = 'users'
    var_116 = 5
    var_117 = 'name'
    var_118 = [var_115, var_116, var_117]
    var_119 = True
    var_120 = module_0.get_in(var_118, var_108, no_default=var_119)



# Parsed testcases at query #4
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = 1
    var_3 = {var_0: var_2}
    var_4 = module_0.get_in(var_1, var_3)
    assert var_4 == 1
    var_5 = 'b'
    var_6 = [var_0, var_5]
    var_7 = 2
    var_8 = {var_5: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.get_in(var_6, var_9)
    assert var_10 == 2
    var_11 = 'c'
    var_12 = [var_0, var_5, var_11]
    var_13 = 3
    var_14 = {var_11: var_13}
    var_15 = {var_5: var_14}
    var_16 = {var_0: var_15}
    var_17 = module_0.get_in(var_12, var_16)
    assert var_17 == 3
    var_18 = 0
    var_19 = [var_18]
    var_20 = [var_2, var_7, var_13]
    var_21 = module_0.get_in(var_19, var_20)
    assert var_21 == 1
    var_22 = [var_2, var_18]
    var_23 = [var_2, var_7]
    var_24 = 4
    var_25 = [var_13, var_24]
    var_26 = [var_23, var_25]
    var_27 = module_0.get_in(var_22, var_26)
    assert var_27 == 3
    var_28 = [var_18, var_2]
    var_29 = [var_2, var_7]
    var_30 = [var_13, var_24]
    var_31 = [var_29, var_30]
    var_32 = module_0.get_in(var_28, var_31)
    assert var_32 == 2
    var_33 = {var_5: var_2}
    var_34 = {var_11: var_7}
    var_35 = [var_33, var_34]
    var_36 = {var_0: var_35}
    var_37 = [var_0, var_18, var_5]
    var_38 = module_0.get_in(var_37, var_36)
    assert var_38 == 1
    var_39 = [var_0, var_2, var_11]
    var_40 = module_0.get_in(var_39, var_36)
    assert var_40 == 2
    var_41 = 'x'
    var_42 = [var_41]
    var_43 = {var_0: var_2}
    var_44 = module_0.get_in(var_42, var_43)
    assert var_44 is None
    var_45 = [var_41]
    var_46 = {var_0: var_2}
    var_47 = module_0.get_in(var_45, var_46, var_18)
    assert var_47 == 0
    var_48 = [var_0, var_41]
    var_49 = {var_0: var_2}
    var_50 = 'missing'
    var_51 = module_0.get_in(var_48, var_49, var_50)
    assert var_51 == 'missing'
    var_52 = 'x'
    var_53 = [var_52]
    var_54 = 'a'
    var_55 = 1
    var_56 = {var_54: var_55}
    var_57 = True
    var_58 = module_0.get_in(var_53, var_56, no_default=var_57)
    var_59 = 5
    var_60 = [var_59]
    var_61 = 1
    var_62 = 2
    var_63 = 3
    var_64 = [var_61, var_62, var_63]
    var_65 = True
    var_66 = module_0.get_in(var_60, var_64, no_default=var_65)
    var_67 = 'a'
    var_68 = 'b'
    var_69 = [var_67, var_68]
    var_70 = 1
    var_71 = {var_67: var_70}
    var_72 = True
    var_73 = module_0.get_in(var_69, var_71, no_default=var_72)
    var_74 = []
    var_75 = {var_67: var_69}
    var_76 = module_0.get_in(var_74, var_75)
    var_77 = []
    var_78 = [var_69, var_66, var_13]
    var_79 = module_0.get_in(var_77, var_78)
    var_80 = [var_67]
    var_81 = None
    var_82 = 'not found'
    var_83 = module_0.get_in(var_80, var_81, var_82)
    assert var_83 == 'not found'
    var_84 = [var_50]
    var_85 = {}
    var_86 = 'custom'
    var_87 = module_0.get_in(var_84, var_85, var_86)
    assert var_87 == 'custom'
    var_88 = 'key'
    var_89 = [var_18, var_88]
    var_90 = [var_69]
    var_91 = [var_90]
    var_92 = module_0.get_in(var_89, var_91, var_86)
    assert var_92 == 'custom'
    var_93 = 'name'
    var_94 = 'purchase'
    var_95 = 'credit card'
    var_96 = 'Alice'
    var_97 = 'items'
    var_98 = 'costs'
    var_99 = 'Apple'
    var_100 = 'Orange'
    var_101 = [var_99, var_100]
    var_102 = 0.5
    var_103 = 1.25
    var_104 = [var_102, var_103]
    var_105 = {var_97: var_101, var_98: var_104}
    var_106 = '5555-1234-1234-1234'
    var_107 = {var_93: var_96, var_94: var_105, var_95: var_106}
    var_108 = [var_94, var_97, var_18]
    var_109 = module_0.get_in(var_108, var_107)
    assert var_109 == 'Apple'
    var_110 = [var_93]
    var_111 = module_0.get_in(var_110, var_107)
    assert var_111 == 'Alice'
    var_112 = 'total'
    var_113 = [var_94, var_112]
    var_114 = module_0.get_in(var_113, var_107)
    assert var_114 is None
    var_115 = [var_94, var_112]
    var_116 = module_0.get_in(var_115, var_107, var_18)
    assert var_116 == 0



# Parsed testcases at query #5
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = [var_0]
    var_12 = module_0.get_in(var_11, var_6)
    var_13 = {var_1: var_3}
    var_14 = 2
    var_15 = {var_2: var_14}
    var_16 = [var_13, var_15]
    var_17 = {var_0: var_16}
    var_18 = 0
    var_19 = [var_0, var_18, var_1]
    var_20 = module_0.get_in(var_19, var_17)
    assert var_20 == 1
    var_21 = [var_0, var_3, var_2]
    var_22 = module_0.get_in(var_21, var_17)
    assert var_22 == 2
    var_23 = 3
    var_24 = {var_2: var_23}
    var_25 = [var_3, var_14, var_24]
    var_26 = {var_1: var_25}
    var_27 = {var_0: var_26}
    var_28 = [var_0, var_1, var_18]
    var_29 = module_0.get_in(var_28, var_27)
    assert var_29 == 1
    var_30 = [var_0, var_1, var_14, var_2]
    var_31 = module_0.get_in(var_30, var_27)
    assert var_31 == 3
    var_32 = {var_0: var_3}
    var_33 = [var_1]
    var_34 = module_0.get_in(var_33, var_32)
    assert var_34 is None
    var_35 = [var_1]
    var_36 = 'missing'
    var_37 = module_0.get_in(var_35, var_32, var_36)
    assert var_37 == 'missing'
    var_38 = [var_1]
    var_39 = module_0.get_in(var_38, var_32, var_18)
    assert var_39 == 0
    var_40 = {var_0: var_3}
    var_41 = 'b'
    var_42 = [var_41]
    var_43 = True
    var_44 = module_0.get_in(var_42, var_40, no_default=var_43)
    var_45 = [var_44, var_14, var_23]
    var_46 = 5
    var_47 = [var_46]
    var_48 = True
    var_49 = module_0.get_in(var_47, var_45, no_default=var_48)
    var_50 = {var_46: var_49}
    var_51 = []
    var_52 = module_0.get_in(var_51, var_50)
    var_53 = None
    var_54 = {var_46: var_53}
    var_55 = [var_46]
    var_56 = module_0.get_in(var_55, var_54)
    assert var_56 is None
    var_57 = [var_46, var_47]
    var_58 = module_0.get_in(var_57, var_54)
    assert var_58 is None
    var_59 = {}
    var_60 = [var_46]
    var_61 = module_0.get_in(var_60, var_59)
    assert var_61 is None
    var_62 = [var_46, var_47]
    var_63 = module_0.get_in(var_62, var_59, var_36)
    assert var_63 == 'missing'
    var_64 = [var_49, var_14, var_23]
    var_65 = 5
    var_66 = [var_65]
    var_67 = module_0.get_in(var_66, var_64)
    assert var_67 is None
    var_68 = [var_65]
    var_69 = 'out of range'
    var_70 = module_0.get_in(var_68, var_64, var_69)
    assert var_70 == 'out of range'
    var_71 = 123
    var_72 = {var_46: var_71}
    var_73 = [var_46, var_47]
    var_74 = module_0.get_in(var_73, var_72)
    assert var_74 is None
    var_75 = {var_47: var_49}
    var_76 = {var_46: var_75}
    var_77 = 'd'
    var_78 = [var_46, var_48, var_77]
    var_79 = module_0.get_in(var_78, var_76)
    assert var_79 is None
    var_80 = [var_46, var_48, var_77]
    var_81 = 'nested missing'
    var_82 = module_0.get_in(var_80, var_76, var_81)
    assert var_82 == 'nested missing'
    var_83 = 'name'
    var_84 = 'purchase'
    var_85 = 'credit card'
    var_86 = 'Alice'
    var_87 = 'items'
    var_88 = 'costs'
    var_89 = 'Apple'
    var_90 = 'Orange'
    var_91 = [var_89, var_90]
    var_92 = 0.5
    var_93 = 1.25
    var_94 = [var_92, var_93]
    var_95 = {var_87: var_91, var_88: var_94}
    var_96 = '5555-1234-1234-1234'
    var_97 = {var_83: var_86, var_84: var_95, var_85: var_96}
    var_98 = [var_84, var_87, var_18]
    var_99 = module_0.get_in(var_98, var_97)
    assert var_99 == 'Apple'
    var_100 = [var_83]
    var_101 = module_0.get_in(var_100, var_97)
    assert var_101 == 'Alice'
    var_102 = 'total'
    var_103 = [var_84, var_102]
    var_104 = module_0.get_in(var_103, var_97)
    assert var_104 is None
    var_105 = [var_84, var_102]
    var_106 = module_0.get_in(var_105, var_97, var_18)
    assert var_106 == 0



# Parsed testcases at query #6
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 42
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = [var_0]
    var_12 = module_0.get_in(var_11, var_6)
    var_13 = 'd'
    var_14 = [var_0, var_1, var_13]
    var_15 = module_0.get_in(var_14, var_6)
    assert var_15 is None
    var_16 = 'x'
    var_17 = [var_0, var_16]
    var_18 = 'not found'
    var_19 = module_0.get_in(var_17, var_6, var_18)
    assert var_19 == 'not found'
    var_20 = [var_16]
    var_21 = 0
    var_22 = module_0.get_in(var_20, var_6, var_21)
    assert var_22 == 0
    var_23 = 'a'
    var_24 = 'b'
    var_25 = 'd'
    var_26 = [var_23, var_24, var_25]
    var_27 = True
    var_28 = module_0.get_in(var_26, var_6, no_default=var_27)
    var_29 = 1
    var_30 = 2
    var_31 = 3
    var_32 = [var_29, var_30, var_31]
    var_33 = 4
    var_34 = 5
    var_35 = 6
    var_36 = [var_33, var_34, var_35]
    var_37 = [var_32, var_36]
    var_38 = [var_21, var_29]
    var_39 = module_0.get_in(var_38, var_37)
    assert var_39 == 2
    var_40 = [var_29, var_30]
    var_41 = module_0.get_in(var_40, var_37)
    assert var_41 == 6
    var_42 = {var_24: var_29}
    var_43 = {var_25: var_30}
    var_44 = [var_42, var_43]
    var_45 = {var_23: var_44}
    var_46 = [var_23, var_21, var_24]
    var_47 = module_0.get_in(var_46, var_45)
    assert var_47 == 1
    var_48 = [var_23, var_29, var_25]
    var_49 = module_0.get_in(var_48, var_45)
    assert var_49 == 2
    var_50 = []
    var_51 = {var_23: var_29}
    var_52 = module_0.get_in(var_50, var_51)
    var_53 = []
    var_54 = [var_29, var_30, var_31]
    var_55 = module_0.get_in(var_53, var_54)
    var_56 = 10
    var_57 = [var_21, var_56]
    var_58 = [var_29, var_30]
    var_59 = [var_58]
    var_60 = module_0.get_in(var_57, var_59)
    assert var_60 is None
    var_61 = 0
    var_62 = 10
    var_63 = [var_61, var_62]
    var_64 = 1
    var_65 = 2
    var_66 = [var_64, var_65]
    var_67 = [var_66]
    var_68 = True
    var_69 = module_0.get_in(var_63, var_67, no_default=var_68)
    var_70 = [var_61, var_62]
    var_71 = 123
    var_72 = {var_61: var_71}
    var_73 = module_0.get_in(var_70, var_72)
    assert var_73 is None
    var_74 = [var_16]
    var_75 = {}
    var_76 = 'default'
    var_77 = module_0.get_in(var_74, var_75, var_76)
    assert var_77 == 'default'
    var_78 = [var_21]
    var_79 = []
    var_80 = 'empty'
    var_81 = module_0.get_in(var_78, var_79, var_80)
    assert var_81 == 'empty'
    var_82 = 'name'
    var_83 = 'purchase'
    var_84 = 'credit card'
    var_85 = 'Alice'
    var_86 = 'items'
    var_87 = 'costs'
    var_88 = 'Apple'
    var_89 = 'Orange'
    var_90 = [var_88, var_89]
    var_91 = 0.5
    var_92 = 1.25
    var_93 = [var_91, var_92]
    var_94 = {var_86: var_90, var_87: var_93}
    var_95 = '5555-1234-1234-1234'
    var_96 = {var_82: var_85, var_83: var_94, var_84: var_95}
    var_97 = [var_83, var_86, var_21]
    var_98 = module_0.get_in(var_97, var_96)
    assert var_98 == 'Apple'
    var_99 = [var_82]
    var_100 = module_0.get_in(var_99, var_96)
    assert var_100 == 'Alice'
    var_101 = 'total'
    var_102 = [var_83, var_101]
    var_103 = module_0.get_in(var_102, var_96)
    assert var_103 is None
    var_104 = 'apple'
    var_105 = [var_83, var_86, var_104]
    var_106 = module_0.get_in(var_105, var_96)
    assert var_106 is None
    var_107 = [var_83, var_86, var_56]
    var_108 = module_0.get_in(var_107, var_96)
    assert var_108 is None
    var_109 = [var_83, var_101]
    var_110 = module_0.get_in(var_109, var_96, var_21)
    assert var_110 == 0



# Parsed testcases at query #7
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 42
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = {var_1: var_11}
    var_13 = [var_9, var_10, var_12]
    var_14 = {var_0: var_13}
    var_15 = [var_0, var_10, var_1]
    var_16 = module_0.get_in(var_15, var_14)
    assert var_16 == 3
    var_17 = {var_0: var_9}
    var_18 = [var_1]
    var_19 = module_0.get_in(var_18, var_17)
    assert var_19 is None
    var_20 = [var_1]
    var_21 = 'not found'
    var_22 = module_0.get_in(var_20, var_17, var_21)
    assert var_22 == 'not found'
    var_23 = {var_0: var_9}
    var_24 = 'b'
    var_25 = [var_24]
    var_26 = True
    var_27 = module_0.get_in(var_25, var_23, no_default=var_26)
    var_28 = [var_9, var_10, var_11]
    var_29 = 5
    var_30 = [var_29]
    var_31 = True
    var_32 = module_0.get_in(var_30, var_28, no_default=var_31)
    var_33 = {var_30: var_9}
    var_34 = {var_29: var_33}
    var_35 = 'd'
    var_36 = [var_29, var_31, var_35]
    var_37 = module_0.get_in(var_36, var_34)
    assert var_37 is None
    var_38 = [var_29, var_31, var_35]
    var_39 = 0
    var_40 = module_0.get_in(var_38, var_34, var_39)
    assert var_40 == 0
    var_41 = {var_29: var_9}
    var_42 = []
    var_43 = module_0.get_in(var_42, var_41)
    var_44 = {var_30: var_9}
    var_45 = {var_31: var_10}
    var_46 = [var_44, var_45]
    var_47 = {var_29: var_46}
    var_48 = [var_29, var_39, var_30]
    var_49 = module_0.get_in(var_48, var_47)
    assert var_49 == 1
    var_50 = [var_29, var_9, var_31]
    var_51 = module_0.get_in(var_50, var_47)
    assert var_51 == 2
    var_52 = None
    var_53 = {var_29: var_52}
    var_54 = [var_29]
    var_55 = module_0.get_in(var_54, var_53)
    assert var_55 is None
    var_56 = [var_29, var_30]
    var_57 = module_0.get_in(var_56, var_53)
    assert var_57 is None
    var_58 = [var_29]
    var_59 = {}
    var_60 = module_0.get_in(var_58, var_59)
    assert var_60 is None
    var_61 = [var_39]
    var_62 = []
    var_63 = module_0.get_in(var_61, var_62)
    assert var_63 is None



# Parsed testcases at query #8
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = [var_0]
    var_12 = module_0.get_in(var_11, var_6)
    var_13 = {var_1: var_3}
    var_14 = 2
    var_15 = {var_2: var_14}
    var_16 = [var_13, var_15]
    var_17 = {var_0: var_16}
    var_18 = 0
    var_19 = [var_0, var_18, var_1]
    var_20 = module_0.get_in(var_19, var_17)
    assert var_20 == 1
    var_21 = [var_0, var_3, var_2]
    var_22 = module_0.get_in(var_21, var_17)
    assert var_22 == 2
    var_23 = 3
    var_24 = [var_3, var_14, var_23]
    var_25 = {var_2: var_24}
    var_26 = {var_1: var_25}
    var_27 = [var_26]
    var_28 = {var_0: var_27}
    var_29 = [var_0, var_18, var_1, var_2, var_3]
    var_30 = module_0.get_in(var_29, var_28)
    assert var_30 == 2
    var_31 = {var_0: var_3}
    var_32 = [var_1]
    var_33 = module_0.get_in(var_32, var_31)
    assert var_33 is None
    var_34 = [var_1]
    var_35 = 'missing'
    var_36 = module_0.get_in(var_34, var_31, var_35)
    assert var_36 == 'missing'
    var_37 = [var_1]
    var_38 = module_0.get_in(var_37, var_31, var_18)
    assert var_38 == 0
    var_39 = {var_0: var_3}
    var_40 = 'b'
    var_41 = [var_40]
    var_42 = True
    var_43 = module_0.get_in(var_41, var_39, no_default=var_42)
    var_44 = [var_43, var_14, var_23]
    var_45 = {var_40: var_44}
    var_46 = 'a'
    var_47 = 10
    var_48 = [var_46, var_47]
    var_49 = True
    var_50 = module_0.get_in(var_48, var_45, no_default=var_49)
    var_51 = {var_46: var_49}
    var_52 = []
    var_53 = module_0.get_in(var_52, var_51)
    var_54 = {var_47: var_49}
    var_55 = {var_46: var_54}
    var_56 = [var_46, var_48]
    var_57 = module_0.get_in(var_56, var_55)
    assert var_57 is None
    var_58 = [var_46, var_48]
    var_59 = module_0.get_in(var_58, var_55, var_35)
    assert var_59 == 'missing'
    var_60 = [var_46]
    var_61 = {}
    var_62 = module_0.get_in(var_60, var_61)
    assert var_62 is None
    var_63 = [var_46]
    var_64 = {}
    var_65 = module_0.get_in(var_63, var_64, var_18)
    assert var_65 == 0
    var_66 = None
    var_67 = {var_46: var_66}
    var_68 = [var_46]
    var_69 = module_0.get_in(var_68, var_67)
    assert var_69 is None
    var_70 = [var_46, var_47]
    var_71 = module_0.get_in(var_70, var_67)
    assert var_71 is None
    var_72 = 'value'
    var_73 = {var_23: var_72}
    var_74 = {var_14: var_73}
    var_75 = {var_49: var_74}
    var_76 = [var_49, var_14, var_23]
    var_77 = module_0.get_in(var_76, var_75)
    assert var_77 == 'value'



# Parsed testcases at query #9
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 42
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = [var_0]
    var_12 = module_0.get_in(var_11, var_6)
    var_13 = 'items'
    var_14 = 'counts'
    var_15 = 'Apple'
    var_16 = 'Orange'
    var_17 = [var_15, var_16]
    var_18 = 5
    var_19 = 3
    var_20 = [var_18, var_19]
    var_21 = {var_13: var_17, var_14: var_20}
    var_22 = 0
    var_23 = [var_13, var_22]
    var_24 = module_0.get_in(var_23, var_21)
    assert var_24 == 'Apple'
    var_25 = 1
    var_26 = [var_13, var_25]
    var_27 = module_0.get_in(var_26, var_21)
    assert var_27 == 'Orange'
    var_28 = [var_14, var_22]
    var_29 = module_0.get_in(var_28, var_21)
    assert var_29 == 5
    var_30 = 'x'
    var_31 = 'y'
    var_32 = [var_30, var_31]
    var_33 = {var_0: var_25}
    var_34 = module_0.get_in(var_32, var_33)
    assert var_34 is None
    var_35 = [var_30, var_31]
    var_36 = {var_0: var_25}
    var_37 = 'not found'
    var_38 = module_0.get_in(var_35, var_36, var_37)
    assert var_38 == 'not found'
    var_39 = [var_0, var_1, var_2]
    var_40 = {}
    var_41 = {var_1: var_40}
    var_42 = {var_0: var_41}
    var_43 = module_0.get_in(var_39, var_42)
    assert var_43 is None
    var_44 = 'x'
    var_45 = [var_44]
    var_46 = {}
    var_47 = True
    var_48 = module_0.get_in(var_45, var_46, no_default=var_47)
    var_49 = 'items'
    var_50 = 10
    var_51 = [var_49, var_50]
    var_52 = []
    var_53 = {var_49: var_52}
    var_54 = True
    var_55 = module_0.get_in(var_51, var_53, no_default=var_54)
    var_56 = 2
    var_57 = {var_49: var_25, var_50: var_56}
    var_58 = []
    var_59 = module_0.get_in(var_58, var_57)
    var_60 = None
    var_61 = {var_51: var_60}
    var_62 = {var_49: var_60, var_50: var_61}
    var_63 = [var_49]
    var_64 = module_0.get_in(var_63, var_62)
    assert var_64 is None
    var_65 = [var_50, var_51]
    var_66 = module_0.get_in(var_65, var_62)
    assert var_66 is None
    var_67 = 'deep'
    var_68 = {var_19: var_67}
    var_69 = {var_56: var_68}
    var_70 = {var_25: var_69}
    var_71 = [var_25, var_56, var_19]
    var_72 = module_0.get_in(var_71, var_70)
    assert var_72 == 'deep'
    var_73 = [var_49, var_50]
    var_74 = 'value'
    var_75 = {var_50: var_74}
    var_76 = {var_49: var_75}
    var_77 = 'default'
    var_78 = module_0.get_in(var_73, var_76, var_77)
    assert var_78 == 'value'
    var_79 = [var_49, var_50]
    var_80 = {var_49: var_52}
    var_81 = module_0.get_in(var_79, var_80)
    assert var_81 is None



# Parsed testcases at query #10
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 42
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = [var_0]
    var_12 = module_0.get_in(var_11, var_6)
    var_13 = 'd'
    var_14 = [var_0, var_1, var_13]
    var_15 = module_0.get_in(var_14, var_6)
    assert var_15 is None
    var_16 = [var_0, var_1, var_13]
    var_17 = 'not found'
    var_18 = module_0.get_in(var_16, var_6, var_17)
    assert var_18 == 'not found'
    var_19 = 'x'
    var_20 = 'y'
    var_21 = 'z'
    var_22 = [var_19, var_20, var_21]
    var_23 = 0
    var_24 = module_0.get_in(var_22, var_6, var_23)
    assert var_24 == 0
    var_25 = 'a'
    var_26 = 'b'
    var_27 = 'd'
    var_28 = [var_25, var_26, var_27]
    var_29 = True
    var_30 = module_0.get_in(var_28, var_6, no_default=var_29)
    var_31 = 'x'
    var_32 = [var_31]
    var_33 = True
    var_34 = module_0.get_in(var_32, var_6, no_default=var_33)
    var_35 = 'items'
    var_36 = 'nested'
    var_37 = 'Apple'
    var_38 = 'Orange'
    var_39 = [var_37, var_38]
    var_40 = 'list'
    var_41 = 1
    var_42 = 2
    var_43 = 'deep'
    var_44 = 'value'
    var_45 = {var_43: var_44}
    var_46 = [var_41, var_42, var_45]
    var_47 = {var_40: var_46}
    var_48 = {var_35: var_39, var_36: var_47}
    var_49 = [var_35, var_23]
    var_50 = module_0.get_in(var_49, var_48)
    assert var_50 == 'Apple'
    var_51 = [var_35, var_41]
    var_52 = module_0.get_in(var_51, var_48)
    assert var_52 == 'Orange'
    var_53 = [var_36, var_40, var_42, var_43]
    var_54 = module_0.get_in(var_53, var_48)
    assert var_54 == 'value'
    var_55 = 5
    var_56 = [var_35, var_55]
    var_57 = module_0.get_in(var_56, var_48)
    assert var_57 is None
    var_58 = [var_35, var_55]
    var_59 = 'missing'
    var_60 = module_0.get_in(var_58, var_48, var_59)
    assert var_60 == 'missing'
    var_61 = 'items'
    var_62 = 5
    var_63 = [var_61, var_62]
    var_64 = True
    var_65 = module_0.get_in(var_63, var_48, no_default=var_64)
    var_66 = 'any'
    var_67 = 'key'
    var_68 = [var_66, var_67]
    var_69 = {}
    var_70 = module_0.get_in(var_68, var_69)
    assert var_70 is None
    var_71 = [var_23]
    var_72 = []
    var_73 = module_0.get_in(var_71, var_72)
    assert var_73 is None
    var_74 = {var_67: var_44}
    var_75 = [var_67]
    var_76 = module_0.get_in(var_75, var_74)
    assert var_76 == 'value'
    var_77 = [var_59]
    var_78 = module_0.get_in(var_77, var_74)
    assert var_78 is None
    var_79 = 3
    var_80 = {var_79: var_43}
    var_81 = {var_42: var_80}
    var_82 = {var_41: var_81}
    var_83 = [var_41, var_42, var_79]
    var_84 = module_0.get_in(var_83, var_82)
    assert var_84 == 'deep'
    var_85 = []
    var_86 = module_0.get_in(var_85, var_6)
    var_87 = []
    var_88 = [var_41, var_42, var_79]
    var_89 = module_0.get_in(var_87, var_88)
    var_90 = []
    var_91 = 'string'
    var_92 = module_0.get_in(var_90, var_91)
    assert var_92 == 'string'
    var_93 = [var_23, var_41]
    var_94 = 'abc'
    var_95 = module_0.get_in(var_93, var_94)
    assert var_95 is None
    var_96 = [var_23, var_41]
    var_97 = 'type error'
    var_98 = module_0.get_in(var_96, var_94, var_97)
    assert var_98 == 'type error'
    var_99 = 'users'
    var_100 = 'name'
    var_101 = 'scores'
    var_102 = 'Alice'
    var_103 = 85
    var_104 = 92
    var_105 = 78
    var_106 = [var_103, var_104, var_105]
    var_107 = {var_100: var_102, var_101: var_106}
    var_108 = 'Bob'
    var_109 = 88
    var_110 = 95
    var_111 = 81
    var_112 = [var_109, var_110, var_111]
    var_113 = {var_100: var_108, var_101: var_112}
    var_114 = [var_107, var_113]
    var_115 = {var_99: var_114}
    var_116 = [var_99, var_23, var_100]
    var_117 = module_0.get_in(var_116, var_115)
    assert var_117 == 'Alice'
    var_118 = [var_99, var_41, var_101, var_42]
    var_119 = module_0.get_in(var_118, var_115)
    assert var_119 == 81
    var_120 = [var_99, var_42, var_100]
    var_121 = module_0.get_in(var_120, var_115)
    assert var_121 is None
    var_122 = [var_99, var_23, var_101, var_55]
    var_123 = module_0.get_in(var_122, var_115)
    assert var_123 is None



# Parsed testcases at query #11
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = 2
    var_10 = {var_1: var_9}
    var_11 = 3
    var_12 = {var_2: var_11}
    var_13 = [var_10, var_12]
    var_14 = {var_0: var_13}
    var_15 = 0
    var_16 = [var_0, var_15, var_1]
    var_17 = module_0.get_in(var_16, var_14)
    assert var_17 == 2
    var_18 = [var_0, var_3, var_2]
    var_19 = module_0.get_in(var_18, var_14)
    assert var_19 == 3
    var_20 = {var_2: var_11}
    var_21 = [var_3, var_9, var_20]
    var_22 = {var_1: var_21}
    var_23 = {var_0: var_22}
    var_24 = [var_0, var_1, var_9, var_2]
    var_25 = module_0.get_in(var_24, var_23)
    assert var_25 == 3
    var_26 = [var_0, var_1, var_3]
    var_27 = module_0.get_in(var_26, var_23)
    assert var_27 == 2
    var_28 = {var_0: var_3}
    var_29 = [var_1]
    var_30 = module_0.get_in(var_29, var_28)
    assert var_30 is None
    var_31 = [var_1]
    var_32 = 'missing'
    var_33 = module_0.get_in(var_31, var_28, var_32)
    assert var_33 == 'missing'
    var_34 = {var_0: var_3}
    var_35 = 'b'
    var_36 = [var_35]
    var_37 = True
    var_38 = module_0.get_in(var_36, var_34, no_default=var_37)
    var_39 = [var_38, var_9, var_11]
    var_40 = 5
    var_41 = [var_40]
    var_42 = True
    var_43 = module_0.get_in(var_41, var_39, no_default=var_42)
    var_44 = {var_40: var_43}
    var_45 = []
    var_46 = module_0.get_in(var_45, var_44)
    var_47 = {var_41: var_43}
    var_48 = {var_40: var_47}
    var_49 = 'd'
    var_50 = [var_40, var_42, var_49]
    var_51 = 'not found'
    var_52 = module_0.get_in(var_50, var_48, var_51)
    assert var_52 == 'not found'
    var_53 = None
    var_54 = {var_40: var_53}
    var_55 = [var_40]
    var_56 = module_0.get_in(var_55, var_54)
    assert var_56 is None
    var_57 = [var_40, var_41]
    var_58 = 'default'
    var_59 = module_0.get_in(var_57, var_54, var_58)
    assert var_59 == 'default'
    var_60 = 'key'
    var_61 = [var_60]
    var_62 = {}
    var_63 = 'empty'
    var_64 = module_0.get_in(var_61, var_62, var_63)
    assert var_64 == 'empty'
    var_65 = 'value'
    var_66 = {var_9: var_65}
    var_67 = {var_43: var_66}
    var_68 = [var_43, var_9]
    var_69 = module_0.get_in(var_68, var_67)
    assert var_69 == 'value'



# Parsed testcases at query #12
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = 1
    var_3 = {var_0: var_2}
    var_4 = module_0.get_in(var_1, var_3)
    assert var_4 == 1
    var_5 = 'b'
    var_6 = [var_0, var_5]
    var_7 = 2
    var_8 = {var_5: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.get_in(var_6, var_9)
    assert var_10 == 2
    var_11 = 'c'
    var_12 = [var_0, var_5, var_11]
    var_13 = 3
    var_14 = {var_11: var_13}
    var_15 = {var_5: var_14}
    var_16 = {var_0: var_15}
    var_17 = module_0.get_in(var_12, var_16)
    assert var_17 == 3
    var_18 = 0
    var_19 = [var_18]
    var_20 = 10
    var_21 = [var_20]
    var_22 = module_0.get_in(var_19, var_21)
    assert var_22 == 10
    var_23 = [var_2, var_18]
    var_24 = []
    var_25 = 20
    var_26 = [var_25]
    var_27 = [var_24, var_26]
    var_28 = module_0.get_in(var_23, var_27)
    assert var_28 == 20
    var_29 = [var_18, var_2]
    var_30 = 30
    var_31 = 40
    var_32 = [var_30, var_31]
    var_33 = [var_32]
    var_34 = module_0.get_in(var_29, var_33)
    assert var_34 == 40
    var_35 = 5
    var_36 = {var_5: var_35}
    var_37 = 6
    var_38 = {var_11: var_37}
    var_39 = [var_36, var_38]
    var_40 = {var_0: var_39}
    var_41 = [var_0, var_18, var_5]
    var_42 = module_0.get_in(var_41, var_40)
    assert var_42 == 5
    var_43 = [var_0, var_2, var_11]
    var_44 = module_0.get_in(var_43, var_40)
    assert var_44 == 6
    var_45 = 'x'
    var_46 = [var_45]
    var_47 = {}
    var_48 = module_0.get_in(var_46, var_47)
    assert var_48 is None
    var_49 = [var_45]
    var_50 = {}
    var_51 = 'missing'
    var_52 = module_0.get_in(var_49, var_50, var_51)
    assert var_52 == 'missing'
    var_53 = [var_0, var_45]
    var_54 = {}
    var_55 = {var_0: var_54}
    var_56 = module_0.get_in(var_53, var_55, var_18)
    assert var_56 == 0
    var_57 = 'x'
    var_58 = [var_57]
    var_59 = {}
    var_60 = True
    var_61 = module_0.get_in(var_58, var_59, no_default=var_60)
    var_62 = 5
    var_63 = [var_62]
    var_64 = []
    var_65 = True
    var_66 = module_0.get_in(var_63, var_64, no_default=var_65)
    var_67 = {var_62: var_64}
    var_68 = []
    var_69 = module_0.get_in(var_68, var_67)
    var_70 = None
    var_71 = {var_62: var_70}
    var_72 = [var_62]
    var_73 = module_0.get_in(var_72, var_71)
    assert var_73 is None
    var_74 = False
    var_75 = ''
    var_76 = {var_62: var_18, var_5: var_74, var_11: var_75}
    var_77 = [var_62]
    var_78 = module_0.get_in(var_77, var_76)
    assert var_78 == 0
    var_79 = [var_5]
    var_80 = module_0.get_in(var_79, var_76)
    assert var_80 is False
    var_81 = [var_11]
    var_82 = module_0.get_in(var_81, var_76)
    assert var_82 == ''
    var_83 = [var_62, var_5, var_11]
    var_84 = {}
    var_85 = {var_62: var_84}
    var_86 = module_0.get_in(var_83, var_85)
    assert var_86 is None
    var_87 = [var_62, var_5, var_11]
    var_88 = {}
    var_89 = {var_62: var_88}
    var_90 = 'not found'
    var_91 = module_0.get_in(var_87, var_89, var_90)
    assert var_91 == 'not found'
    var_92 = 'deep'
    var_93 = {var_13: var_92}
    var_94 = {var_7: var_93}
    var_95 = {var_64: var_94}
    var_96 = [var_64, var_7, var_13]
    var_97 = module_0.get_in(var_96, var_95)
    assert var_97 == 'deep'
    var_98 = (var_62, var_5)
    var_99 = 'value'
    var_100 = {var_98: var_99}
    var_101 = (var_62, var_5)
    var_102 = [var_101]
    var_103 = module_0.get_in(var_102, var_100)
    assert var_103 == 'value'



# Parsed testcases at query #13
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = 2
    var_10 = {var_1: var_9}
    var_11 = 3
    var_12 = {var_2: var_11}
    var_13 = [var_10, var_12]
    var_14 = {var_0: var_13}
    var_15 = 0
    var_16 = [var_0, var_15, var_1]
    var_17 = module_0.get_in(var_16, var_14)
    assert var_17 == 2
    var_18 = {var_0: var_3}
    var_19 = [var_1]
    var_20 = module_0.get_in(var_19, var_18)
    assert var_20 is None
    var_21 = [var_1]
    var_22 = 'missing'
    var_23 = module_0.get_in(var_21, var_18, var_22)
    assert var_23 == 'missing'
    var_24 = {var_0: var_3}
    var_25 = 'b'
    var_26 = [var_25]
    var_27 = True
    var_28 = module_0.get_in(var_26, var_24, no_default=var_27)
    var_29 = [var_28, var_9, var_11]
    var_30 = 5
    var_31 = [var_30]
    var_32 = True
    var_33 = module_0.get_in(var_31, var_29, no_default=var_32)
    var_34 = {var_31: var_33}
    var_35 = {var_30: var_34}
    var_36 = [var_30, var_32]
    var_37 = module_0.get_in(var_36, var_35)
    assert var_37 is None
    var_38 = [var_30, var_32]
    var_39 = module_0.get_in(var_38, var_35, var_15)
    assert var_39 == 0
    var_40 = {var_30: var_33}
    var_41 = []
    var_42 = module_0.get_in(var_41, var_40)
    var_43 = {var_31: var_9}
    var_44 = {var_32: var_11}
    var_45 = [var_43, var_44]
    var_46 = {var_30: var_45}
    var_47 = [var_30, var_33, var_32]
    var_48 = module_0.get_in(var_47, var_46)
    assert var_48 == 3
    var_49 = {var_31: var_33}
    var_50 = {var_30: var_49}
    var_51 = [var_30, var_31, var_32]
    var_52 = module_0.get_in(var_51, var_50)
    assert var_52 is None
    var_53 = [var_33, var_9, var_11]
    var_54 = 5
    var_55 = [var_54]
    var_56 = module_0.get_in(var_55, var_53)
    assert var_56 is None
    var_57 = {var_30: var_33}
    var_58 = [var_30, var_31]
    var_59 = module_0.get_in(var_58, var_57)
    assert var_59 is None
    var_60 = [var_30]
    var_61 = None
    var_62 = module_0.get_in(var_60, var_61)
    assert var_62 is None
    var_63 = [var_30]
    var_64 = 'default'
    var_65 = module_0.get_in(var_63, var_61, var_64)
    assert var_65 == 'default'



# Parsed testcases at query #14
#--------------------------


import pyrsistent._toolz as module_0
import builtins as module_1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 42
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = [var_0]
    var_12 = module_0.get_in(var_11, var_6)
    var_13 = 'items'
    var_14 = 'name'
    var_15 = 'price'
    var_16 = 'apple'
    var_17 = 1.0
    var_18 = {var_14: var_16, var_15: var_17}
    var_19 = 'orange'
    var_20 = 1.5
    var_21 = {var_14: var_19, var_15: var_20}
    var_22 = [var_18, var_21]
    var_23 = {var_13: var_22}
    var_24 = 0
    var_25 = [var_13, var_24, var_14]
    var_26 = module_0.get_in(var_25, var_23)
    assert var_26 == 'apple'
    var_27 = [var_13, var_17, var_15]
    var_28 = module_0.get_in(var_27, var_23)
    var_29 = 'x'
    var_30 = 'y'
    var_31 = [var_29, var_30]
    var_32 = {var_0: var_17}
    var_33 = module_0.get_in(var_31, var_32)
    assert var_33 is None
    var_34 = [var_29, var_30]
    var_35 = {var_0: var_17}
    var_36 = 'not found'
    var_37 = module_0.get_in(var_34, var_35, var_36)
    assert var_37 == 'not found'
    var_38 = [var_0, var_1]
    var_39 = {var_0: var_17}
    var_40 = module_0.get_in(var_38, var_39, var_24)
    assert var_40 == 0
    var_41 = 'x'
    var_42 = [var_41]
    var_43 = {}
    var_44 = True
    var_45 = module_0.get_in(var_42, var_43, no_default=var_44)
    var_46 = 'items'
    var_47 = 5
    var_48 = [var_46, var_47]
    var_49 = []
    var_50 = {var_46: var_49}
    var_51 = True
    var_52 = module_0.get_in(var_48, var_50, no_default=var_51)
    var_53 = 2
    var_54 = {var_46: var_17, var_47: var_53}
    var_55 = []
    var_56 = module_0.get_in(var_55, var_54)
    var_57 = None
    var_58 = {var_46: var_57}
    var_59 = [var_46]
    var_60 = module_0.get_in(var_59, var_58)
    assert var_60 is None
    var_61 = [var_46]
    var_62 = 'default'
    var_63 = module_0.get_in(var_61, var_58, var_62)
    assert var_63 is None
    var_64 = 10
    var_65 = 20
    var_66 = 30
    var_67 = [var_64, var_65, var_66]
    var_68 = [var_17]
    var_69 = module_0.get_in(var_68, var_67)
    assert var_69 == 20
    var_70 = [var_24]
    var_71 = module_0.get_in(var_70, var_67)
    assert var_71 == 10
    var_72 = [var_17, var_53]
    var_73 = 3
    var_74 = 4
    var_75 = [var_73, var_74]
    var_76 = [var_72, var_75]
    var_77 = [var_24, var_17]
    var_78 = module_0.get_in(var_77, var_76)
    assert var_78 == 2
    var_79 = [var_17, var_24]
    var_80 = module_0.get_in(var_79, var_76)
    assert var_80 == 3
    var_81 = [var_24]
    var_82 = 'string'
    var_83 = 'type error'
    var_84 = module_0.get_in(var_81, var_82, var_83)
    assert var_84 == 'type error'
    var_85 = module_1.object()
    var_86 = 'non'
    var_87 = 'existent'
    var_88 = [var_86, var_87]
    var_89 = {}
    var_90 = module_0.get_in(var_88, var_89, var_85)



# Parsed testcases at query #15
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = 1
    var_3 = {var_0: var_2}
    var_4 = module_0.get_in(var_1, var_3)
    assert var_4 == 1
    var_5 = 'b'
    var_6 = [var_0, var_5]
    var_7 = 2
    var_8 = {var_5: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.get_in(var_6, var_9)
    assert var_10 == 2
    var_11 = 'c'
    var_12 = [var_0, var_5, var_11]
    var_13 = 3
    var_14 = {var_11: var_13}
    var_15 = {var_5: var_14}
    var_16 = {var_0: var_15}
    var_17 = module_0.get_in(var_12, var_16)
    assert var_17 == 3
    var_18 = 0
    var_19 = [var_18]
    var_20 = [var_2, var_7, var_13]
    var_21 = module_0.get_in(var_19, var_20)
    assert var_21 == 1
    var_22 = [var_2, var_18]
    var_23 = [var_2, var_7]
    var_24 = 4
    var_25 = [var_13, var_24]
    var_26 = [var_23, var_25]
    var_27 = module_0.get_in(var_22, var_26)
    assert var_27 == 3
    var_28 = [var_18, var_2]
    var_29 = [var_2, var_7]
    var_30 = [var_13, var_24]
    var_31 = [var_29, var_30]
    var_32 = module_0.get_in(var_28, var_31)
    assert var_32 == 2
    var_33 = {var_5: var_2}
    var_34 = {var_11: var_7}
    var_35 = [var_33, var_34]
    var_36 = {var_0: var_35}
    var_37 = [var_0, var_18, var_5]
    var_38 = module_0.get_in(var_37, var_36)
    assert var_38 == 1
    var_39 = [var_0, var_2, var_11]
    var_40 = module_0.get_in(var_39, var_36)
    assert var_40 == 2
    var_41 = 'x'
    var_42 = [var_41]
    var_43 = {var_0: var_2}
    var_44 = module_0.get_in(var_42, var_43)
    assert var_44 is None
    var_45 = [var_0, var_41]
    var_46 = {var_5: var_2}
    var_47 = {var_0: var_46}
    var_48 = module_0.get_in(var_45, var_47)
    assert var_48 is None
    var_49 = [var_0, var_5, var_11]
    var_50 = {var_5: var_2}
    var_51 = {var_0: var_50}
    var_52 = module_0.get_in(var_49, var_51)
    assert var_52 is None
    var_53 = 5
    var_54 = [var_53]
    var_55 = [var_2, var_7, var_13]
    var_56 = module_0.get_in(var_54, var_55)
    assert var_56 is None
    var_57 = [var_18, var_53]
    var_58 = [var_2, var_7]
    var_59 = [var_13, var_24]
    var_60 = [var_58, var_59]
    var_61 = module_0.get_in(var_57, var_60)
    assert var_61 is None
    var_62 = [var_41]
    var_63 = {var_0: var_2}
    var_64 = 'not found'
    var_65 = module_0.get_in(var_62, var_63, var_64)
    assert var_65 == 'not found'
    var_66 = [var_0, var_41]
    var_67 = {var_5: var_2}
    var_68 = {var_0: var_67}
    var_69 = module_0.get_in(var_66, var_68, var_18)
    assert var_69 == 0
    var_70 = [var_53]
    var_71 = [var_2, var_7, var_13]
    var_72 = 'missing'
    var_73 = module_0.get_in(var_70, var_71, var_72)
    assert var_73 == 'missing'
    var_74 = 'x'
    var_75 = [var_74]
    var_76 = 'a'
    var_77 = 1
    var_78 = {var_76: var_77}
    var_79 = True
    var_80 = module_0.get_in(var_75, var_78, no_default=var_79)
    var_81 = 'a'
    var_82 = 'x'
    var_83 = [var_81, var_82]
    var_84 = 'b'
    var_85 = 1
    var_86 = {var_84: var_85}
    var_87 = {var_81: var_86}
    var_88 = True
    var_89 = module_0.get_in(var_83, var_87, no_default=var_88)
    var_90 = 5
    var_91 = [var_90]
    var_92 = 1
    var_93 = 2
    var_94 = 3
    var_95 = [var_92, var_93, var_94]
    var_96 = True
    var_97 = module_0.get_in(var_91, var_95, no_default=var_96)
    var_98 = 0
    var_99 = 'x'
    var_100 = [var_98, var_99]
    var_101 = 1
    var_102 = 2
    var_103 = 3
    var_104 = [var_101, var_102, var_103]
    var_105 = True
    var_106 = module_0.get_in(var_100, var_104, no_default=var_105)
    var_107 = []
    var_108 = {var_98: var_100}
    var_109 = module_0.get_in(var_107, var_108)
    var_110 = []
    var_111 = [var_100, var_105, var_13]
    var_112 = module_0.get_in(var_110, var_111)
    var_113 = []
    var_114 = 'hello'
    var_115 = module_0.get_in(var_113, var_114)
    assert var_115 == 'hello'
    var_116 = [var_98]
    var_117 = None
    var_118 = {var_98: var_117}
    var_119 = module_0.get_in(var_116, var_118)
    assert var_119 is None
    var_120 = [var_98, var_103]
    var_121 = {var_103: var_117}
    var_122 = {var_98: var_121}
    var_123 = module_0.get_in(var_120, var_122)
    assert var_123 is None
    var_124 = [var_98]
    var_125 = False
    var_126 = {var_98: var_125}
    var_127 = module_0.get_in(var_124, var_126)
    assert var_127 is False
    var_128 = [var_98, var_103]
    var_129 = False
    var_130 = {var_103: var_129}
    var_131 = {var_98: var_130}
    var_132 = module_0.get_in(var_128, var_131)
    assert var_132 is False
    var_133 = [var_98]
    var_134 = {var_98: var_129}
    var_135 = module_0.get_in(var_133, var_134)
    assert var_135 == 0
    var_136 = [var_98, var_103]
    var_137 = {var_103: var_129}
    var_138 = {var_98: var_137}
    var_139 = module_0.get_in(var_136, var_138)
    assert var_139 == 0
    var_140 = 'users'
    var_141 = 'id'
    var_142 = 'name'
    var_143 = 'orders'
    var_144 = 'Alice'
    var_145 = 'total'
    var_146 = 'A1'
    var_147 = 100
    var_148 = {var_141: var_146, var_145: var_147}
    var_149 = 'A2'
    var_150 = 200
    var_151 = {var_141: var_149, var_145: var_150}
    var_152 = [var_148, var_151]
    var_153 = {var_141: var_100, var_142: var_144, var_143: var_152}
    var_154 = 'Bob'
    var_155 = 'B1'
    var_156 = 150
    var_157 = {var_141: var_155, var_145: var_156}
    var_158 = [var_157]
    var_159 = {var_141: var_105, var_142: var_154, var_143: var_158}
    var_160 = [var_153, var_159]
    var_161 = {var_140: var_160}
    var_162 = [var_140, var_129, var_142]
    var_163 = module_0.get_in(var_162, var_161)
    assert var_163 == 'Alice'
    var_164 = [var_140, var_100, var_143, var_129, var_145]
    var_165 = module_0.get_in(var_164, var_161)
    assert var_165 == 150
    var_166 = [var_140, var_129, var_143, var_100, var_141]
    var_167 = module_0.get_in(var_166, var_161)
    assert var_167 == 'A2'
    var_168 = [var_140, var_105, var_142]
    var_169 = module_0.get_in(var_168, var_161)
    assert var_169 is None
    var_170 = [var_140, var_129, var_143, var_105]
    var_171 = module_0.get_in(var_170, var_161)
    assert var_171 is None
    var_172 = 'address'
    var_173 = [var_140, var_129, var_172]
    var_174 = module_0.get_in(var_173, var_161)
    assert var_174 is None



# Parsed testcases at query #16
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 42
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = [var_9, var_10, var_11]
    var_13 = 4
    var_14 = 5
    var_15 = 6
    var_16 = [var_13, var_14, var_15]
    var_17 = [var_12, var_16]
    var_18 = [var_9, var_10]
    var_19 = module_0.get_in(var_18, var_17)
    assert var_19 == 6
    var_20 = {var_1: var_9}
    var_21 = {var_2: var_10}
    var_22 = [var_20, var_21]
    var_23 = {var_0: var_22}
    var_24 = [var_0, var_9, var_2]
    var_25 = module_0.get_in(var_24, var_23)
    assert var_25 == 2
    var_26 = {var_0: var_9}
    var_27 = [var_1]
    var_28 = module_0.get_in(var_27, var_26)
    assert var_28 is None
    var_29 = [var_1]
    var_30 = 'missing'
    var_31 = module_0.get_in(var_29, var_26, var_30)
    assert var_31 == 'missing'
    var_32 = {var_0: var_9}
    var_33 = 'b'
    var_34 = [var_33]
    var_35 = True
    var_36 = module_0.get_in(var_34, var_32, no_default=var_35)
    var_37 = [var_9, var_10, var_11]
    var_38 = 5
    var_39 = [var_38]
    var_40 = True
    var_41 = module_0.get_in(var_39, var_37, no_default=var_40)
    var_42 = {var_38: var_9}
    var_43 = []
    var_44 = module_0.get_in(var_43, var_42)
    var_45 = {var_39: var_9}
    var_46 = {var_38: var_45}
    var_47 = [var_38, var_40]
    var_48 = module_0.get_in(var_47, var_46)
    assert var_48 is None
    var_49 = [var_38, var_40]
    var_50 = 0
    var_51 = module_0.get_in(var_49, var_46, var_50)
    assert var_51 == 0
    var_52 = None
    var_53 = {var_38: var_52}
    var_54 = [var_38]
    var_55 = module_0.get_in(var_54, var_53)
    assert var_55 is None
    var_56 = [var_38, var_39]
    var_57 = module_0.get_in(var_56, var_53)
    assert var_57 is None
    var_58 = 'key'
    var_59 = [var_58]
    var_60 = {}
    var_61 = module_0.get_in(var_59, var_60)
    assert var_61 is None
    var_62 = [var_50]
    var_63 = []
    var_64 = module_0.get_in(var_62, var_63)
    assert var_64 is None
    var_65 = 'users'
    var_66 = 'name'
    var_67 = 'scores'
    var_68 = 'Alice'
    var_69 = 85
    var_70 = 92
    var_71 = 78
    var_72 = [var_69, var_70, var_71]
    var_73 = {var_66: var_68, var_67: var_72}
    var_74 = 'Bob'
    var_75 = 88
    var_76 = 95
    var_77 = 81
    var_78 = [var_75, var_76, var_77]
    var_79 = {var_66: var_74, var_67: var_78}
    var_80 = [var_73, var_79]
    var_81 = {var_65: var_80}
    var_82 = [var_65, var_9, var_67, var_10]
    var_83 = module_0.get_in(var_82, var_81)
    assert var_83 == 81
    var_84 = [var_65, var_50, var_66]
    var_85 = module_0.get_in(var_84, var_81)
    assert var_85 == 'Alice'
    var_86 = {var_38: var_9}
    var_87 = [var_39]
    var_88 = 'default'
    var_89 = module_0.get_in(var_87, var_86, var_88)
    assert var_89 == 'default'
    var_90 = {var_38: var_9}
    var_91 = [var_38, var_39]
    var_92 = module_0.get_in(var_91, var_90)
    assert var_92 is None



# Parsed testcases at query #17
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 42
    var_9 = 10
    var_10 = {var_1: var_9}
    var_11 = 20
    var_12 = {var_2: var_11}
    var_13 = [var_10, var_12]
    var_14 = {var_0: var_13}
    var_15 = 0
    var_16 = [var_0, var_15, var_1]
    var_17 = module_0.get_in(var_16, var_14)
    assert var_17 == 10
    var_18 = 1
    var_19 = [var_0, var_18, var_2]
    var_20 = module_0.get_in(var_19, var_14)
    assert var_20 == 20
    var_21 = {var_1: var_18}
    var_22 = {var_0: var_21}
    var_23 = [var_0, var_2]
    var_24 = module_0.get_in(var_23, var_22)
    assert var_24 is None
    var_25 = [var_0, var_2]
    var_26 = 'missing'
    var_27 = module_0.get_in(var_25, var_22, var_26)
    assert var_27 == 'missing'
    var_28 = {var_1: var_18}
    var_29 = {var_0: var_28}
    var_30 = 'a'
    var_31 = 'c'
    var_32 = [var_30, var_31]
    var_33 = True
    var_34 = module_0.get_in(var_32, var_29, no_default=var_33)
    var_35 = 2
    var_36 = 3
    var_37 = [var_18, var_35, var_36]
    var_38 = {var_30: var_37}
    var_39 = 'a'
    var_40 = 5
    var_41 = [var_39, var_40]
    var_42 = True
    var_43 = module_0.get_in(var_41, var_38, no_default=var_42)
    var_44 = [var_18, var_35, var_36]
    var_45 = 4
    var_46 = 5
    var_47 = 6
    var_48 = [var_45, var_46, var_47]
    var_49 = [var_44, var_48]
    var_50 = [var_15, var_18]
    var_51 = module_0.get_in(var_50, var_49)
    assert var_51 == 2
    var_52 = [var_18, var_35]
    var_53 = module_0.get_in(var_52, var_49)
    assert var_53 == 6
    var_54 = {var_40: var_18}
    var_55 = [var_35, var_36, var_45]
    var_56 = {var_41: var_55}
    var_57 = [var_54, var_56]
    var_58 = {var_39: var_57}
    var_59 = [var_39, var_18, var_41, var_35]
    var_60 = module_0.get_in(var_59, var_58)
    assert var_60 == 4
    var_61 = {var_39: var_18}
    var_62 = []
    var_63 = module_0.get_in(var_62, var_61)
    var_64 = None
    var_65 = {var_40: var_64}
    var_66 = {var_39: var_65}
    var_67 = [var_39, var_40]
    var_68 = module_0.get_in(var_67, var_66)
    assert var_68 is None
    var_69 = [var_39]
    var_70 = {}
    var_71 = 'default'
    var_72 = module_0.get_in(var_69, var_70, var_71)
    assert var_72 == 'default'
    var_73 = {var_39: var_42}
    var_74 = [var_39, var_40]
    var_75 = module_0.get_in(var_74, var_73)
    assert var_75 is None



# Parsed testcases at query #18
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 42
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = {var_1: var_11}
    var_13 = [var_9, var_10, var_12]
    var_14 = {var_0: var_13}
    var_15 = [var_0, var_10, var_1]
    var_16 = module_0.get_in(var_15, var_14)
    assert var_16 == 3
    var_17 = {var_0: var_9}
    var_18 = [var_1]
    var_19 = module_0.get_in(var_18, var_17)
    assert var_19 is None
    var_20 = [var_1]
    var_21 = 'default'
    var_22 = module_0.get_in(var_20, var_17, var_21)
    assert var_22 == 'default'
    var_23 = {var_0: var_9}
    var_24 = 'b'
    var_25 = [var_24]
    var_26 = True
    var_27 = module_0.get_in(var_25, var_23, no_default=var_26)
    var_28 = [var_9, var_10, var_11]
    var_29 = 5
    var_30 = [var_29]
    var_31 = True
    var_32 = module_0.get_in(var_30, var_28, no_default=var_31)
    var_33 = {}
    var_34 = {var_30: var_33}
    var_35 = {var_29: var_34}
    var_36 = [var_29, var_30, var_31]
    var_37 = module_0.get_in(var_36, var_35)
    assert var_37 is None
    var_38 = [var_29, var_30, var_31]
    var_39 = 0
    var_40 = module_0.get_in(var_38, var_35, var_39)
    assert var_40 == 0
    var_41 = {var_29: var_9}
    var_42 = []
    var_43 = module_0.get_in(var_42, var_41)
    var_44 = {var_30: var_9}
    var_45 = {var_31: var_10}
    var_46 = [var_44, var_45]
    var_47 = {var_29: var_46}
    var_48 = [var_29, var_39, var_30]
    var_49 = module_0.get_in(var_48, var_47)
    assert var_49 == 1
    var_50 = [var_29, var_9, var_31]
    var_51 = module_0.get_in(var_50, var_47)
    assert var_51 == 2
    var_52 = {var_29: var_9}
    var_53 = [var_29, var_30]
    var_54 = module_0.get_in(var_53, var_52)
    assert var_54 is None
    var_55 = [var_29, var_30]
    var_56 = 'error'
    var_57 = module_0.get_in(var_55, var_52, var_56)
    assert var_57 == 'error'



# Parsed testcases at query #19
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 42
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = [var_0]
    var_12 = module_0.get_in(var_11, var_6)
    var_13 = 1
    var_14 = 2
    var_15 = 3
    var_16 = [var_13, var_14, var_15]
    var_17 = 4
    var_18 = 5
    var_19 = 6
    var_20 = [var_17, var_18, var_19]
    var_21 = [var_16, var_20]
    var_22 = 0
    var_23 = [var_22, var_13]
    var_24 = module_0.get_in(var_23, var_21)
    assert var_24 == 2
    var_25 = [var_13, var_14]
    var_26 = module_0.get_in(var_25, var_21)
    assert var_26 == 6
    var_27 = {var_1: var_13}
    var_28 = {var_2: var_14}
    var_29 = [var_27, var_28]
    var_30 = {var_0: var_29}
    var_31 = [var_0, var_22, var_1]
    var_32 = module_0.get_in(var_31, var_30)
    assert var_32 == 1
    var_33 = [var_0, var_13, var_2]
    var_34 = module_0.get_in(var_33, var_30)
    assert var_34 == 2
    var_35 = {var_0: var_13}
    var_36 = [var_1]
    var_37 = module_0.get_in(var_36, var_35)
    assert var_37 is None
    var_38 = [var_1]
    var_39 = 'missing'
    var_40 = module_0.get_in(var_38, var_35, var_39)
    assert var_40 == 'missing'
    var_41 = [var_1]
    var_42 = module_0.get_in(var_41, var_35, var_22)
    assert var_42 == 0
    var_43 = {var_0: var_13}
    var_44 = 'b'
    var_45 = [var_44]
    var_46 = True
    var_47 = module_0.get_in(var_45, var_43, no_default=var_46)
    var_48 = [var_13, var_14, var_15]
    var_49 = 5
    var_50 = [var_49]
    var_51 = True
    var_52 = module_0.get_in(var_50, var_48, no_default=var_51)
    var_53 = {var_49: var_13}
    var_54 = []
    var_55 = module_0.get_in(var_54, var_53)
    var_56 = {var_50: var_13}
    var_57 = {var_49: var_56}
    var_58 = 'd'
    var_59 = [var_49, var_51, var_58]
    var_60 = module_0.get_in(var_59, var_57)
    assert var_60 is None
    var_61 = [var_49, var_51, var_58]
    var_62 = 'default'
    var_63 = module_0.get_in(var_61, var_57, var_62)
    assert var_63 == 'default'
    var_64 = None
    var_65 = {var_49: var_64}
    var_66 = [var_49]
    var_67 = module_0.get_in(var_66, var_65)
    assert var_67 is None
    var_68 = [var_49, var_50]
    var_69 = module_0.get_in(var_68, var_65)
    assert var_69 is None
    var_70 = 'key'
    var_71 = [var_70]
    var_72 = {}
    var_73 = module_0.get_in(var_71, var_72)
    assert var_73 is None
    var_74 = [var_22]
    var_75 = []
    var_76 = module_0.get_in(var_74, var_75)
    assert var_76 is None
    var_77 = 'users'
    var_78 = 'name'
    var_79 = 'scores'
    var_80 = 'Alice'
    var_81 = 85
    var_82 = 92
    var_83 = 78
    var_84 = [var_81, var_82, var_83]
    var_85 = {var_78: var_80, var_79: var_84}
    var_86 = 'Bob'
    var_87 = 88
    var_88 = 95
    var_89 = 82
    var_90 = [var_87, var_88, var_89]
    var_91 = {var_78: var_86, var_79: var_90}
    var_92 = [var_85, var_91]
    var_93 = {var_77: var_92}
    var_94 = [var_77, var_22, var_78]
    var_95 = module_0.get_in(var_94, var_93)
    assert var_95 == 'Alice'
    var_96 = [var_77, var_13, var_79, var_14]
    var_97 = module_0.get_in(var_96, var_93)
    assert var_97 == 82
    var_98 = [var_77, var_22, var_79, var_18]
    var_99 = module_0.get_in(var_98, var_93)
    assert var_99 is None
    var_100 = [var_77, var_14, var_78]
    var_101 = module_0.get_in(var_100, var_93)
    assert var_101 is None



# Parsed testcases at query #20
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 42
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = [var_0]
    var_12 = module_0.get_in(var_11, var_6)
    var_13 = 1
    var_14 = 2
    var_15 = 3
    var_16 = [var_13, var_14, var_15]
    var_17 = 4
    var_18 = 5
    var_19 = 6
    var_20 = [var_17, var_18, var_19]
    var_21 = [var_16, var_20]
    var_22 = 0
    var_23 = [var_22, var_13]
    var_24 = module_0.get_in(var_23, var_21)
    assert var_24 == 2
    var_25 = [var_13, var_14]
    var_26 = module_0.get_in(var_25, var_21)
    assert var_26 == 6
    var_27 = 10
    var_28 = {var_1: var_27}
    var_29 = 20
    var_30 = {var_2: var_29}
    var_31 = [var_28, var_30]
    var_32 = {var_0: var_31}
    var_33 = [var_0, var_22, var_1]
    var_34 = module_0.get_in(var_33, var_32)
    assert var_34 == 10
    var_35 = [var_0, var_13, var_2]
    var_36 = module_0.get_in(var_35, var_32)
    assert var_36 == 20
    var_37 = 'x'
    var_38 = {var_37: var_13}
    var_39 = 'y'
    var_40 = [var_39]
    var_41 = module_0.get_in(var_40, var_38)
    assert var_41 is None
    var_42 = [var_39]
    var_43 = 'missing'
    var_44 = module_0.get_in(var_42, var_38, var_43)
    assert var_44 == 'missing'
    var_45 = [var_37, var_39]
    var_46 = module_0.get_in(var_45, var_38, var_22)
    assert var_46 == 0
    var_47 = {var_37: var_13}
    var_48 = 'y'
    var_49 = [var_48]
    var_50 = True
    var_51 = module_0.get_in(var_49, var_47, no_default=var_50)
    var_52 = [var_13, var_14, var_15]
    var_53 = 5
    var_54 = [var_53]
    var_55 = True
    var_56 = module_0.get_in(var_54, var_52, no_default=var_55)
    var_57 = {var_53: var_13}
    var_58 = []
    var_59 = module_0.get_in(var_58, var_57)
    var_60 = None
    var_61 = {var_53: var_60}
    var_62 = [var_53]
    var_63 = module_0.get_in(var_62, var_61)
    assert var_63 is None
    var_64 = [var_53, var_54]
    var_65 = 'default'
    var_66 = module_0.get_in(var_64, var_61, var_65)
    assert var_66 == 'default'
    var_67 = 'deep'
    var_68 = {var_15: var_67}
    var_69 = {var_14: var_68}
    var_70 = {var_13: var_69}
    var_71 = [var_13, var_14, var_15]
    var_72 = module_0.get_in(var_71, var_70)
    assert var_72 == 'deep'
    var_73 = {}
    var_74 = 'nested'
    var_75 = 'path'
    var_76 = [var_67, var_74, var_75]
    var_77 = []
    var_78 = module_0.get_in(var_76, var_73, var_77)
    var_79 = [var_67, var_74, var_75]
    var_80 = {}
    var_81 = module_0.get_in(var_79, var_73, var_80)
    var_82 = 'not a collection'
    var_83 = [var_22]
    var_84 = 'type error'
    var_85 = module_0.get_in(var_83, var_82, var_84)
    assert var_85 == 'type error'



# Parsed testcases at query #21
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 42
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = [var_0]
    var_12 = module_0.get_in(var_11, var_6)
    var_13 = 1
    var_14 = 2
    var_15 = 3
    var_16 = [var_13, var_14, var_15]
    var_17 = 4
    var_18 = 5
    var_19 = 6
    var_20 = [var_17, var_18, var_19]
    var_21 = [var_16, var_20]
    var_22 = 0
    var_23 = [var_22, var_13]
    var_24 = module_0.get_in(var_23, var_21)
    assert var_24 == 2
    var_25 = [var_13, var_14]
    var_26 = module_0.get_in(var_25, var_21)
    assert var_26 == 6
    var_27 = {var_1: var_13}
    var_28 = {var_2: var_14}
    var_29 = [var_27, var_28]
    var_30 = {var_0: var_29}
    var_31 = [var_0, var_22, var_1]
    var_32 = module_0.get_in(var_31, var_30)
    assert var_32 == 1
    var_33 = [var_0, var_13, var_2]
    var_34 = module_0.get_in(var_33, var_30)
    assert var_34 == 2
    var_35 = {var_0: var_13}
    var_36 = [var_1]
    var_37 = module_0.get_in(var_36, var_35)
    assert var_37 is None
    var_38 = [var_1]
    var_39 = 'missing'
    var_40 = module_0.get_in(var_38, var_35, var_39)
    assert var_40 == 'missing'
    var_41 = [var_1]
    var_42 = module_0.get_in(var_41, var_35, var_22)
    assert var_42 == 0
    var_43 = {var_0: var_13}
    var_44 = 'b'
    var_45 = [var_44]
    var_46 = True
    var_47 = module_0.get_in(var_45, var_43, no_default=var_46)
    var_48 = [var_13, var_14, var_15]
    var_49 = 5
    var_50 = [var_49]
    var_51 = True
    var_52 = module_0.get_in(var_50, var_48, no_default=var_51)
    var_53 = {var_49: var_13}
    var_54 = []
    var_55 = module_0.get_in(var_54, var_53)
    var_56 = {}
    var_57 = {var_50: var_56}
    var_58 = {var_49: var_57}
    var_59 = [var_49, var_50, var_51]
    var_60 = module_0.get_in(var_59, var_58)
    assert var_60 is None
    var_61 = [var_49, var_50, var_51]
    var_62 = 'default'
    var_63 = module_0.get_in(var_61, var_58, var_62)
    assert var_63 == 'default'
    var_64 = None
    var_65 = {var_49: var_64}
    var_66 = [var_49, var_50]
    var_67 = module_0.get_in(var_66, var_65)
    assert var_67 is None
    var_68 = 'key'
    var_69 = [var_68]
    var_70 = {}
    var_71 = module_0.get_in(var_69, var_70)
    assert var_71 is None
    var_72 = [var_22]
    var_73 = []
    var_74 = module_0.get_in(var_72, var_73)
    assert var_74 is None
    var_75 = 'users'
    var_76 = 'name'
    var_77 = 'scores'
    var_78 = 'Alice'
    var_79 = 85
    var_80 = 92
    var_81 = 78
    var_82 = [var_79, var_80, var_81]
    var_83 = {var_76: var_78, var_77: var_82}
    var_84 = 'Bob'
    var_85 = 76
    var_86 = 88
    var_87 = 95
    var_88 = [var_85, var_86, var_87]
    var_89 = {var_76: var_84, var_77: var_88}
    var_90 = [var_83, var_89]
    var_91 = {var_75: var_90}
    var_92 = [var_75, var_22, var_76]
    var_93 = module_0.get_in(var_92, var_91)
    assert var_93 == 'Alice'
    var_94 = [var_75, var_13, var_77, var_14]
    var_95 = module_0.get_in(var_94, var_91)
    assert var_95 == 95
    var_96 = [var_75, var_22, var_77, var_13]
    var_97 = module_0.get_in(var_96, var_91)
    assert var_97 == 92
    var_98 = [var_75, var_14, var_76]
    var_99 = module_0.get_in(var_98, var_91)
    assert var_99 is None
    var_100 = 'age'
    var_101 = [var_75, var_22, var_100]
    var_102 = module_0.get_in(var_101, var_91)
    assert var_102 is None



# Parsed testcases at query #22
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 42
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = [var_0]
    var_12 = module_0.get_in(var_11, var_6)
    var_13 = 'd'
    var_14 = [var_0, var_1, var_13]
    var_15 = module_0.get_in(var_14, var_6)
    assert var_15 is None
    var_16 = [var_0, var_1, var_13]
    var_17 = 'not found'
    var_18 = module_0.get_in(var_16, var_6, var_17)
    assert var_18 == 'not found'
    var_19 = 'x'
    var_20 = 'y'
    var_21 = 'z'
    var_22 = [var_19, var_20, var_21]
    var_23 = 0
    var_24 = module_0.get_in(var_22, var_6, var_23)
    assert var_24 == 0
    var_25 = 'a'
    var_26 = 'b'
    var_27 = 'd'
    var_28 = [var_25, var_26, var_27]
    var_29 = True
    var_30 = module_0.get_in(var_28, var_6, no_default=var_29)
    var_31 = 'items'
    var_32 = 'name'
    var_33 = 'price'
    var_34 = 'apple'
    var_35 = 1.0
    var_36 = {var_32: var_34, var_33: var_35}
    var_37 = 'orange'
    var_38 = 1.5
    var_39 = {var_32: var_37, var_33: var_38}
    var_40 = [var_36, var_39]
    var_41 = {var_31: var_40}
    var_42 = [var_31, var_23, var_32]
    var_43 = module_0.get_in(var_42, var_41)
    assert var_43 == 'apple'
    var_44 = [var_31, var_35, var_33]
    var_45 = module_0.get_in(var_44, var_41)
    var_46 = 5
    var_47 = [var_31, var_46, var_32]
    var_48 = module_0.get_in(var_47, var_41)
    assert var_48 is None
    var_49 = 'items'
    var_50 = 5
    var_51 = 'name'
    var_52 = [var_49, var_50, var_51]
    var_53 = True
    var_54 = module_0.get_in(var_52, var_41, no_default=var_53)
    var_55 = []
    var_56 = module_0.get_in(var_55, var_6)
    var_57 = [var_19]
    var_58 = module_0.get_in(var_57, var_6)
    assert var_58 is None
    var_59 = 'x'
    var_60 = [var_59]
    var_61 = True
    var_62 = module_0.get_in(var_60, var_6, no_default=var_61)
    var_63 = [var_59, var_60, var_61, var_13]
    var_64 = module_0.get_in(var_63, var_6)
    assert var_64 is None
    var_65 = 'purchase'
    var_66 = 'credit card'
    var_67 = 'Alice'
    var_68 = 'costs'
    var_69 = 'Apple'
    var_70 = 'Orange'
    var_71 = [var_69, var_70]
    var_72 = 0.5
    var_73 = 1.25
    var_74 = [var_72, var_73]
    var_75 = {var_31: var_71, var_68: var_74}
    var_76 = '5555-1234-1234-1234'
    var_77 = {var_32: var_67, var_65: var_75, var_66: var_76}
    var_78 = [var_65, var_31, var_23]
    var_79 = module_0.get_in(var_78, var_77)
    assert var_79 == 'Apple'
    var_80 = [var_32]
    var_81 = module_0.get_in(var_80, var_77)
    assert var_81 == 'Alice'
    var_82 = 'total'
    var_83 = [var_65, var_82]
    var_84 = module_0.get_in(var_83, var_77)
    assert var_84 is None
    var_85 = [var_65, var_82]
    var_86 = module_0.get_in(var_85, var_77, var_23)
    assert var_86 == 0



# Parsed testcases at query #23
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = [var_0]
    var_12 = module_0.get_in(var_11, var_6)
    var_13 = 2
    var_14 = 3
    var_15 = [var_3, var_13, var_14]
    var_16 = 4
    var_17 = 5
    var_18 = 6
    var_19 = [var_16, var_17, var_18]
    var_20 = [var_15, var_19]
    var_21 = 0
    var_22 = [var_21, var_3]
    var_23 = module_0.get_in(var_22, var_20)
    assert var_23 == 2
    var_24 = [var_3, var_13]
    var_25 = module_0.get_in(var_24, var_20)
    assert var_25 == 6
    var_26 = {var_1: var_3}
    var_27 = {var_2: var_13}
    var_28 = [var_26, var_27]
    var_29 = {var_0: var_28}
    var_30 = [var_0, var_21, var_1]
    var_31 = module_0.get_in(var_30, var_29)
    assert var_31 == 1
    var_32 = [var_0, var_3, var_2]
    var_33 = module_0.get_in(var_32, var_29)
    assert var_33 == 2
    var_34 = {var_0: var_3}
    var_35 = [var_1]
    var_36 = module_0.get_in(var_35, var_34)
    assert var_36 is None
    var_37 = [var_1]
    var_38 = 'default'
    var_39 = module_0.get_in(var_37, var_34, var_38)
    assert var_39 == 'default'
    var_40 = [var_1]
    var_41 = module_0.get_in(var_40, var_34, var_21)
    assert var_41 == 0
    var_42 = {var_0: var_3}
    var_43 = 'b'
    var_44 = [var_43]
    var_45 = True
    var_46 = module_0.get_in(var_44, var_42, no_default=var_45)
    var_47 = [var_46, var_13, var_14]
    var_48 = 5
    var_49 = [var_48]
    var_50 = True
    var_51 = module_0.get_in(var_49, var_47, no_default=var_50)
    var_52 = {var_48: var_51}
    var_53 = []
    var_54 = module_0.get_in(var_53, var_52)
    var_55 = {}
    var_56 = {var_49: var_55}
    var_57 = {var_48: var_56}
    var_58 = [var_48, var_49, var_50]
    var_59 = module_0.get_in(var_58, var_57)
    assert var_59 is None
    var_60 = [var_48, var_49, var_50]
    var_61 = 'missing'
    var_62 = module_0.get_in(var_60, var_57, var_61)
    assert var_62 == 'missing'
    var_63 = None
    var_64 = {var_48: var_63}
    var_65 = [var_48]
    var_66 = module_0.get_in(var_65, var_64)
    assert var_66 is None
    var_67 = [var_48, var_49]
    var_68 = module_0.get_in(var_67, var_64)
    assert var_68 is None
    var_69 = 'key'
    var_70 = [var_69]
    var_71 = {}
    var_72 = module_0.get_in(var_70, var_71)
    assert var_72 is None
    var_73 = [var_21]
    var_74 = []
    var_75 = module_0.get_in(var_73, var_74)
    assert var_75 is None
    var_76 = 'users'
    var_77 = 'name'
    var_78 = 'scores'
    var_79 = 'Alice'
    var_80 = 85
    var_81 = 92
    var_82 = 78
    var_83 = [var_80, var_81, var_82]
    var_84 = {var_77: var_79, var_78: var_83}
    var_85 = 'Bob'
    var_86 = 88
    var_87 = 95
    var_88 = 81
    var_89 = [var_86, var_87, var_88]
    var_90 = {var_77: var_85, var_78: var_89}
    var_91 = [var_84, var_90]
    var_92 = {var_76: var_91}
    var_93 = [var_76, var_21, var_77]
    var_94 = module_0.get_in(var_93, var_92)
    assert var_94 == 'Alice'
    var_95 = [var_76, var_51, var_78, var_13]
    var_96 = module_0.get_in(var_95, var_92)
    assert var_96 == 81
    var_97 = [var_76, var_21, var_78, var_51]
    var_98 = module_0.get_in(var_97, var_92)
    assert var_98 == 92



# Parsed testcases at query #24
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 42
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = [var_0]
    var_12 = module_0.get_in(var_11, var_6)
    var_13 = 1
    var_14 = 2
    var_15 = 3
    var_16 = [var_13, var_14, var_15]
    var_17 = 4
    var_18 = 5
    var_19 = 6
    var_20 = [var_17, var_18, var_19]
    var_21 = [var_16, var_20]
    var_22 = 0
    var_23 = [var_22, var_13]
    var_24 = module_0.get_in(var_23, var_21)
    assert var_24 == 2
    var_25 = [var_13, var_14]
    var_26 = module_0.get_in(var_25, var_21)
    assert var_26 == 6
    var_27 = 10
    var_28 = {var_1: var_27}
    var_29 = 20
    var_30 = {var_2: var_29}
    var_31 = [var_28, var_30]
    var_32 = {var_0: var_31}
    var_33 = [var_0, var_22, var_1]
    var_34 = module_0.get_in(var_33, var_32)
    assert var_34 == 10
    var_35 = [var_0, var_13, var_2]
    var_36 = module_0.get_in(var_35, var_32)
    assert var_36 == 20
    var_37 = 'x'
    var_38 = {var_37: var_13}
    var_39 = 'y'
    var_40 = [var_39]
    var_41 = module_0.get_in(var_40, var_38)
    assert var_41 is None
    var_42 = [var_39]
    var_43 = 'missing'
    var_44 = module_0.get_in(var_42, var_38, var_43)
    assert var_44 == 'missing'
    var_45 = [var_37, var_39]
    var_46 = module_0.get_in(var_45, var_38, var_22)
    assert var_46 == 0
    var_47 = {var_37: var_13}
    var_48 = 'y'
    var_49 = [var_48]
    var_50 = True
    var_51 = module_0.get_in(var_49, var_47, no_default=var_50)
    var_52 = [var_13, var_14, var_15]
    var_53 = 5
    var_54 = [var_53]
    var_55 = True
    var_56 = module_0.get_in(var_54, var_52, no_default=var_55)
    var_57 = {var_53: var_13}
    var_58 = []
    var_59 = module_0.get_in(var_58, var_57)
    var_60 = None
    var_61 = {var_53: var_60}
    var_62 = [var_53]
    var_63 = module_0.get_in(var_62, var_61)
    assert var_63 is None
    var_64 = [var_54]
    var_65 = 'default'
    var_66 = module_0.get_in(var_64, var_61, var_65)
    assert var_66 == 'default'
    var_67 = 'level1'
    var_68 = 'level2'
    var_69 = 'level3'
    var_70 = 'level4'
    var_71 = 'deep_value'
    var_72 = {var_70: var_71}
    var_73 = {var_69: var_72}
    var_74 = {var_68: var_73}
    var_75 = {var_67: var_74}
    var_76 = [var_67, var_68, var_69, var_70]
    var_77 = module_0.get_in(var_76, var_75)
    assert var_77 == 'deep_value'
    var_78 = [var_67, var_68, var_69, var_43]
    var_79 = module_0.get_in(var_78, var_75, var_60)
    assert var_79 is None
    var_80 = 'number_keys'
    var_81 = {var_15: var_80}
    var_82 = {var_14: var_81}
    var_83 = {var_13: var_82}
    var_84 = [var_13, var_14, var_15]
    var_85 = module_0.get_in(var_84, var_83)
    assert var_85 == 'number_keys'
    var_86 = 123
    var_87 = {var_53: var_86}
    var_88 = [var_53, var_54]
    var_89 = 'type_error'
    var_90 = module_0.get_in(var_88, var_87, var_89)
    assert var_90 == 'type_error'



# Parsed testcases at query #25
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = [var_0]
    var_12 = module_0.get_in(var_11, var_6)
    var_13 = 2
    var_14 = 3
    var_15 = [var_3, var_13, var_14]
    var_16 = 4
    var_17 = 5
    var_18 = 6
    var_19 = [var_16, var_17, var_18]
    var_20 = [var_15, var_19]
    var_21 = 0
    var_22 = [var_21, var_3]
    var_23 = module_0.get_in(var_22, var_20)
    assert var_23 == 2
    var_24 = [var_3, var_13]
    var_25 = module_0.get_in(var_24, var_20)
    assert var_25 == 6
    var_26 = {var_1: var_3}
    var_27 = {var_2: var_13}
    var_28 = [var_26, var_27]
    var_29 = {var_0: var_28}
    var_30 = [var_0, var_21, var_1]
    var_31 = module_0.get_in(var_30, var_29)
    assert var_31 == 1
    var_32 = [var_0, var_3, var_2]
    var_33 = module_0.get_in(var_32, var_29)
    assert var_33 == 2
    var_34 = {var_0: var_3}
    var_35 = [var_1]
    var_36 = module_0.get_in(var_35, var_34)
    assert var_36 is None
    var_37 = [var_1]
    var_38 = 'not found'
    var_39 = module_0.get_in(var_37, var_34, var_38)
    assert var_39 == 'not found'
    var_40 = [var_1]
    var_41 = module_0.get_in(var_40, var_34, var_21)
    assert var_41 == 0
    var_42 = {var_0: var_3}
    var_43 = 'b'
    var_44 = [var_43]
    var_45 = True
    var_46 = module_0.get_in(var_44, var_42, no_default=var_45)
    var_47 = [var_46, var_13, var_14]
    var_48 = 5
    var_49 = [var_48]
    var_50 = True
    var_51 = module_0.get_in(var_49, var_47, no_default=var_50)
    var_52 = {var_48: var_51}
    var_53 = []
    var_54 = module_0.get_in(var_53, var_52)
    var_55 = []
    var_56 = 'default'
    var_57 = module_0.get_in(var_55, var_52, var_56)
    var_58 = {}
    var_59 = {var_49: var_58}
    var_60 = {var_48: var_59}
    var_61 = [var_48, var_49, var_50]
    var_62 = module_0.get_in(var_61, var_60)
    assert var_62 is None
    var_63 = [var_48, var_49, var_50]
    var_64 = module_0.get_in(var_63, var_60, var_21)
    assert var_64 == 0
    var_65 = None
    var_66 = {var_48: var_65}
    var_67 = [var_48]
    var_68 = module_0.get_in(var_67, var_66)
    assert var_68 is None
    var_69 = [var_48, var_49]
    var_70 = module_0.get_in(var_69, var_66)
    assert var_70 is None
    var_71 = 'key'
    var_72 = [var_71]
    var_73 = {}
    var_74 = module_0.get_in(var_72, var_73)
    assert var_74 is None
    var_75 = [var_21]
    var_76 = []
    var_77 = module_0.get_in(var_75, var_76)
    assert var_77 is None
    var_78 = 'users'
    var_79 = 'name'
    var_80 = 'scores'
    var_81 = 'Alice'
    var_82 = 85
    var_83 = 92
    var_84 = 78
    var_85 = [var_82, var_83, var_84]
    var_86 = {var_79: var_81, var_80: var_85}
    var_87 = 'Bob'
    var_88 = 88
    var_89 = 79
    var_90 = 91
    var_91 = [var_88, var_89, var_90]
    var_92 = {var_79: var_87, var_80: var_91}
    var_93 = [var_86, var_92]
    var_94 = {var_78: var_93}
    var_95 = [var_78, var_21, var_79]
    var_96 = module_0.get_in(var_95, var_94)
    assert var_96 == 'Alice'
    var_97 = [var_78, var_51, var_80, var_13]
    var_98 = module_0.get_in(var_97, var_94)
    assert var_98 == 91
    var_99 = [var_78, var_21, var_80, var_51]
    var_100 = module_0.get_in(var_99, var_94)
    assert var_100 == 92
    var_101 = [var_78, var_13, var_79]
    var_102 = module_0.get_in(var_101, var_94)
    assert var_102 is None



# Parsed testcases at query #26
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 42
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = [var_0]
    var_12 = module_0.get_in(var_11, var_6)
    var_13 = 'items'
    var_14 = 'id'
    var_15 = 1
    var_16 = {var_14: var_15}
    var_17 = 2
    var_18 = {var_14: var_17}
    var_19 = [var_16, var_18]
    var_20 = {var_13: var_19}
    var_21 = 0
    var_22 = [var_13, var_21, var_14]
    var_23 = module_0.get_in(var_22, var_20)
    assert var_23 == 1
    var_24 = [var_13, var_15, var_14]
    var_25 = module_0.get_in(var_24, var_20)
    assert var_25 == 2
    var_26 = 10
    var_27 = {var_2: var_26}
    var_28 = {var_1: var_27}
    var_29 = 20
    var_30 = {var_2: var_29}
    var_31 = {var_1: var_30}
    var_32 = [var_28, var_31]
    var_33 = {var_0: var_32}
    var_34 = [var_0, var_21, var_1, var_2]
    var_35 = module_0.get_in(var_34, var_33)
    assert var_35 == 10
    var_36 = [var_0, var_15, var_1, var_2]
    var_37 = module_0.get_in(var_36, var_33)
    assert var_37 == 20
    var_38 = 'x'
    var_39 = {var_38: var_15}
    var_40 = 'y'
    var_41 = [var_40]
    var_42 = module_0.get_in(var_41, var_39)
    assert var_42 is None
    var_43 = [var_40]
    var_44 = 'missing'
    var_45 = module_0.get_in(var_43, var_39, var_44)
    assert var_45 == 'missing'
    var_46 = [var_38, var_40]
    var_47 = module_0.get_in(var_46, var_39, var_21)
    assert var_47 == 0
    var_48 = {var_38: var_15}
    var_49 = 'y'
    var_50 = [var_49]
    var_51 = True
    var_52 = module_0.get_in(var_50, var_48, no_default=var_51)
    var_53 = []
    var_54 = {var_13: var_53}
    var_55 = 'items'
    var_56 = 0
    var_57 = [var_55, var_56]
    var_58 = True
    var_59 = module_0.get_in(var_57, var_54, no_default=var_58)
    var_60 = 'key'
    var_61 = [var_60]
    var_62 = {}
    var_63 = module_0.get_in(var_61, var_62)
    assert var_63 is None
    var_64 = [var_21]
    var_65 = []
    var_66 = module_0.get_in(var_64, var_65)
    assert var_66 is None
    var_67 = {}
    var_68 = {var_55: var_67}
    var_69 = [var_55, var_56]
    var_70 = module_0.get_in(var_69, var_68)
    assert var_70 is None
    var_71 = [var_55, var_56]
    var_72 = []
    var_73 = module_0.get_in(var_71, var_68, var_72)
    var_74 = 3
    var_75 = 'deep'
    var_76 = {var_74: var_75}
    var_77 = {var_17: var_76}
    var_78 = {var_15: var_77}
    var_79 = [var_15, var_17, var_74]
    var_80 = module_0.get_in(var_79, var_78)
    assert var_80 == 'deep'
    var_81 = None
    var_82 = {var_55: var_81}
    var_83 = 'a'
    var_84 = 'b'
    var_85 = [var_83, var_84]
    var_86 = True
    var_87 = module_0.get_in(var_85, var_82, no_default=var_86)
    var_88 = [var_83, var_84]
    var_89 = module_0.get_in(var_88, var_82)
    assert var_89 is None
    var_90 = 'level1'
    var_91 = 'level2'
    var_92 = 'level3'
    var_93 = 'value'
    var_94 = {var_92: var_93}
    var_95 = {var_91: var_94}
    var_96 = {var_90: var_95}
    var_97 = [var_90, var_91, var_44]
    var_98 = 'default'
    var_99 = module_0.get_in(var_97, var_96, var_98)
    assert var_99 == 'default'
    var_100 = 'simple'
    var_101 = {var_100: var_93}
    var_102 = [var_100]
    var_103 = module_0.get_in(var_102, var_101)
    assert var_103 == 'value'
    var_104 = [var_100]
    var_105 = True
    var_106 = module_0.get_in(var_104, var_101, no_default=var_105)
    assert var_106 == 'value'



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = 1
    var_3 = {var_0: var_2}
    var_4 = module_0.get_in(var_1, var_3)
    assert var_4 == 1
    var_5 = 'b'
    var_6 = [var_0, var_5]
    var_7 = 2
    var_8 = {var_5: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.get_in(var_6, var_9)
    assert var_10 == 2
    var_11 = 'c'
    var_12 = [var_0, var_5, var_11]
    var_13 = 3
    var_14 = {var_11: var_13}
    var_15 = {var_5: var_14}
    var_16 = {var_0: var_15}
    var_17 = module_0.get_in(var_12, var_16)
    assert var_17 == 3
    var_18 = 0
    var_19 = [var_18]
    var_20 = 10
    var_21 = [var_20]
    var_22 = module_0.get_in(var_19, var_21)
    assert var_22 == 10
    var_23 = [var_18, var_2]
    var_24 = 20
    var_25 = [var_20, var_24]
    var_26 = [var_25]
    var_27 = module_0.get_in(var_23, var_26)
    assert var_27 == 20
    var_28 = [var_18, var_2, var_7]
    var_29 = 30
    var_30 = [var_20, var_24, var_29]
    var_31 = [var_30]
    var_32 = [var_31]
    var_33 = module_0.get_in(var_28, var_32)
    assert var_33 == 30
    var_34 = {var_5: var_2}
    var_35 = {var_11: var_7}
    var_36 = [var_34, var_35]
    var_37 = {var_0: var_36}
    var_38 = [var_0, var_18, var_5]
    var_39 = module_0.get_in(var_38, var_37)
    assert var_39 == 1
    var_40 = [var_0, var_2, var_11]
    var_41 = module_0.get_in(var_40, var_37)
    assert var_41 == 2
    var_42 = 'x'
    var_43 = [var_42]
    var_44 = {var_0: var_2}
    var_45 = module_0.get_in(var_43, var_44)
    assert var_45 is None
    var_46 = [var_42]
    var_47 = {var_0: var_2}
    var_48 = 'missing'
    var_49 = module_0.get_in(var_46, var_47, var_48)
    assert var_49 == 'missing'
    var_50 = [var_0, var_42]
    var_51 = {var_0: var_2}
    var_52 = module_0.get_in(var_50, var_51, var_18)
    assert var_52 == 0
    var_53 = 'x'
    var_54 = [var_53]
    var_55 = 'a'
    var_56 = 1
    var_57 = {var_55: var_56}
    var_58 = True
    var_59 = module_0.get_in(var_54, var_57, no_default=var_58)
    var_60 = 5
    var_61 = [var_60]
    var_62 = 1
    var_63 = 2
    var_64 = 3
    var_65 = [var_62, var_63, var_64]
    var_66 = True
    var_67 = module_0.get_in(var_61, var_65, no_default=var_66)
    var_68 = []
    var_69 = {var_60: var_62}
    var_70 = module_0.get_in(var_68, var_69)
    var_71 = []
    var_72 = [var_62, var_67, var_13]
    var_73 = module_0.get_in(var_71, var_72)
    var_74 = [var_60]
    var_75 = None
    var_76 = module_0.get_in(var_74, var_75)
    assert var_76 is None
    var_77 = [var_60]
    var_78 = 'default'
    var_79 = module_0.get_in(var_77, var_75, var_78)
    assert var_79 == 'default'
    var_80 = [var_60]
    var_81 = {}
    var_82 = module_0.get_in(var_80, var_81)
    assert var_82 is None
    var_83 = [var_18]
    var_84 = []
    var_85 = module_0.get_in(var_83, var_84)
    assert var_85 is None
    var_86 = 'users'
    var_87 = 'name'
    var_88 = 'orders'
    var_89 = 'Alice'
    var_90 = 'id'
    var_91 = 'items'
    var_92 = 'apple'
    var_93 = 'banana'
    var_94 = [var_92, var_93]
    var_95 = {var_90: var_62, var_91: var_94}
    var_96 = 'orange'
    var_97 = [var_96]
    var_98 = {var_90: var_67, var_91: var_97}
    var_99 = [var_95, var_98]
    var_100 = {var_87: var_89, var_88: var_99}
    var_101 = [var_100]
    var_102 = {var_86: var_101}
    var_103 = [var_86, var_18, var_88, var_62, var_91, var_18]
    var_104 = module_0.get_in(var_103, var_102)
    assert var_104 == 'orange'
    var_105 = [var_86, var_18, var_88, var_67]
    var_106 = module_0.get_in(var_105, var_102)
    assert var_106 is None
    var_107 = [var_86, var_18, var_88, var_67]
    var_108 = []
    var_109 = module_0.get_in(var_107, var_102, var_108)



# Parsed testcases at query #2
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = [var_0]
    var_12 = module_0.get_in(var_11, var_6)
    var_13 = 2
    var_14 = 3
    var_15 = [var_3, var_13, var_14]
    var_16 = 4
    var_17 = 5
    var_18 = 6
    var_19 = [var_16, var_17, var_18]
    var_20 = [var_15, var_19]
    var_21 = 0
    var_22 = [var_21, var_3]
    var_23 = module_0.get_in(var_22, var_20)
    assert var_23 == 2
    var_24 = [var_3, var_13]
    var_25 = module_0.get_in(var_24, var_20)
    assert var_25 == 6
    var_26 = {var_1: var_3}
    var_27 = {var_2: var_13}
    var_28 = [var_26, var_27]
    var_29 = {var_0: var_28}
    var_30 = [var_0, var_21, var_1]
    var_31 = module_0.get_in(var_30, var_29)
    assert var_31 == 1
    var_32 = [var_0, var_3, var_2]
    var_33 = module_0.get_in(var_32, var_29)
    assert var_33 == 2
    var_34 = {var_0: var_3}
    var_35 = [var_1]
    var_36 = module_0.get_in(var_35, var_34)
    assert var_36 is None
    var_37 = [var_1]
    var_38 = 'not found'
    var_39 = module_0.get_in(var_37, var_34, var_38)
    assert var_39 == 'not found'
    var_40 = [var_1]
    var_41 = module_0.get_in(var_40, var_34, var_21)
    assert var_41 == 0
    var_42 = {var_0: var_3}
    var_43 = 'b'
    var_44 = [var_43]
    var_45 = True
    var_46 = module_0.get_in(var_44, var_42, no_default=var_45)
    var_47 = [var_46, var_13, var_14]
    var_48 = 5
    var_49 = [var_48]
    var_50 = True
    var_51 = module_0.get_in(var_49, var_47, no_default=var_50)
    var_52 = {var_48: var_51}
    var_53 = []
    var_54 = module_0.get_in(var_53, var_52)
    var_55 = {}
    var_56 = {var_49: var_55}
    var_57 = {var_48: var_56}
    var_58 = [var_48, var_49, var_50]
    var_59 = module_0.get_in(var_58, var_57)
    assert var_59 is None
    var_60 = [var_48, var_49, var_50]
    var_61 = module_0.get_in(var_60, var_57, var_21)
    assert var_61 == 0
    var_62 = None
    var_63 = {var_48: var_62}
    var_64 = [var_48]
    var_65 = module_0.get_in(var_64, var_63)
    assert var_65 is None
    var_66 = [var_48, var_49]
    var_67 = module_0.get_in(var_66, var_63)
    assert var_67 is None
    var_68 = [var_48]
    var_69 = {}
    var_70 = module_0.get_in(var_68, var_69)
    assert var_70 is None
    var_71 = [var_21]
    var_72 = []
    var_73 = module_0.get_in(var_71, var_72)
    assert var_73 is None
    var_74 = 'users'
    var_75 = 'name'
    var_76 = 'scores'
    var_77 = 'Alice'
    var_78 = 85
    var_79 = 92
    var_80 = 78
    var_81 = [var_78, var_79, var_80]
    var_82 = {var_75: var_77, var_76: var_81}
    var_83 = 'Bob'
    var_84 = 88
    var_85 = 95
    var_86 = 82
    var_87 = [var_84, var_85, var_86]
    var_88 = {var_75: var_83, var_76: var_87}
    var_89 = [var_82, var_88]
    var_90 = {var_74: var_89}
    var_91 = [var_74, var_21, var_75]
    var_92 = module_0.get_in(var_91, var_90)
    assert var_92 == 'Alice'
    var_93 = [var_74, var_51, var_76, var_13]
    var_94 = module_0.get_in(var_93, var_90)
    assert var_94 == 82
    var_95 = [var_74, var_21, var_76, var_17]
    var_96 = module_0.get_in(var_95, var_90)
    assert var_96 is None
    var_97 = [var_74, var_13, var_75]
    var_98 = module_0.get_in(var_97, var_90)
    assert var_98 is None



# Parsed testcases at query #3
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = 2
    var_10 = {var_1: var_9}
    var_11 = 3
    var_12 = {var_2: var_11}
    var_13 = [var_10, var_12]
    var_14 = {var_0: var_13}
    var_15 = 0
    var_16 = [var_0, var_15, var_1]
    var_17 = module_0.get_in(var_16, var_14)
    assert var_17 == 2
    var_18 = [var_0, var_3, var_2]
    var_19 = module_0.get_in(var_18, var_14)
    assert var_19 == 3
    var_20 = {var_2: var_11}
    var_21 = [var_3, var_9, var_20]
    var_22 = {var_1: var_21}
    var_23 = {var_0: var_22}
    var_24 = [var_0, var_1, var_9, var_2]
    var_25 = module_0.get_in(var_24, var_23)
    assert var_25 == 3
    var_26 = {var_0: var_3}
    var_27 = [var_1]
    var_28 = module_0.get_in(var_27, var_26)
    assert var_28 is None
    var_29 = [var_1]
    var_30 = 'missing'
    var_31 = module_0.get_in(var_29, var_26, var_30)
    assert var_31 == 'missing'
    var_32 = {var_0: var_3}
    var_33 = 'b'
    var_34 = [var_33]
    var_35 = True
    var_36 = module_0.get_in(var_34, var_32, no_default=var_35)
    var_37 = [var_36, var_9, var_11]
    var_38 = 5
    var_39 = [var_38]
    var_40 = True
    var_41 = module_0.get_in(var_39, var_37, no_default=var_40)
    var_42 = {var_38: var_41}
    var_43 = []
    var_44 = module_0.get_in(var_43, var_42)
    var_45 = {var_39: var_41}
    var_46 = {var_38: var_45}
    var_47 = [var_38, var_40]
    var_48 = module_0.get_in(var_47, var_46)
    assert var_48 is None
    var_49 = [var_38, var_40]
    var_50 = module_0.get_in(var_49, var_46, var_15)
    assert var_50 == 0
    var_51 = None
    var_52 = {var_38: var_51}
    var_53 = [var_38, var_39]
    var_54 = module_0.get_in(var_53, var_52)
    assert var_54 is None
    var_55 = 'key'
    var_56 = [var_55]
    var_57 = {}
    var_58 = module_0.get_in(var_56, var_57)
    assert var_58 is None
    var_59 = [var_15]
    var_60 = []
    var_61 = module_0.get_in(var_59, var_60)
    assert var_61 is None
    var_62 = 'value'
    var_63 = {var_9: var_62}
    var_64 = {var_41: var_63}
    var_65 = [var_41, var_9]
    var_66 = module_0.get_in(var_65, var_64)
    assert var_66 == 'value'
    var_67 = [var_41, var_9, var_11]
    var_68 = '0'
    var_69 = [var_68]
    var_70 = module_0.get_in(var_69, var_67)
    assert var_70 is None



# Parsed testcases at query #4
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 42
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = {var_1: var_11}
    var_13 = [var_9, var_10, var_12]
    var_14 = {var_0: var_13}
    var_15 = [var_0, var_10, var_1]
    var_16 = module_0.get_in(var_15, var_14)
    assert var_16 == 3
    var_17 = {var_0: var_9}
    var_18 = [var_1]
    var_19 = module_0.get_in(var_18, var_17)
    assert var_19 is None
    var_20 = [var_1]
    var_21 = 'default'
    var_22 = module_0.get_in(var_20, var_17, var_21)
    assert var_22 == 'default'
    var_23 = {var_0: var_9}
    var_24 = 'b'
    var_25 = [var_24]
    var_26 = True
    var_27 = module_0.get_in(var_25, var_23, no_default=var_26)
    var_28 = [var_9, var_10, var_11]
    var_29 = 5
    var_30 = [var_29]
    var_31 = True
    var_32 = module_0.get_in(var_30, var_28, no_default=var_31)
    var_33 = {var_30: var_9}
    var_34 = {var_29: var_33}
    var_35 = 'd'
    var_36 = [var_29, var_31, var_35]
    var_37 = 'missing'
    var_38 = module_0.get_in(var_36, var_34, var_37)
    assert var_38 == 'missing'
    var_39 = {var_29: var_9}
    var_40 = []
    var_41 = module_0.get_in(var_40, var_39)
    var_42 = {var_30: var_9}
    var_43 = {var_31: var_10}
    var_44 = [var_42, var_43]
    var_45 = {var_29: var_44}
    var_46 = 0
    var_47 = [var_29, var_46, var_30]
    var_48 = module_0.get_in(var_47, var_45)
    assert var_48 == 1
    var_49 = [var_29, var_9, var_31]
    var_50 = module_0.get_in(var_49, var_45)
    assert var_50 == 2
    var_51 = None
    var_52 = {var_29: var_51}
    var_53 = [var_29]
    var_54 = module_0.get_in(var_53, var_52)
    assert var_54 is None
    var_55 = {var_29: var_51}
    var_56 = [var_29, var_30]
    var_57 = module_0.get_in(var_56, var_55)
    assert var_57 is None
    var_58 = {}
    var_59 = [var_29]
    var_60 = module_0.get_in(var_59, var_58)
    assert var_60 is None
    var_61 = [var_29]
    var_62 = []
    var_63 = module_0.get_in(var_61, var_58, var_62)



# Parsed testcases at query #5
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = 1
    var_3 = {var_0: var_2}
    var_4 = module_0.get_in(var_1, var_3)
    assert var_4 == 1
    var_5 = 'b'
    var_6 = [var_0, var_5]
    var_7 = 2
    var_8 = {var_5: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.get_in(var_6, var_9)
    assert var_10 == 2
    var_11 = 'c'
    var_12 = [var_0, var_5, var_11]
    var_13 = 3
    var_14 = {var_11: var_13}
    var_15 = {var_5: var_14}
    var_16 = {var_0: var_15}
    var_17 = module_0.get_in(var_12, var_16)
    assert var_17 == 3
    var_18 = 0
    var_19 = [var_18]
    var_20 = 10
    var_21 = [var_20]
    var_22 = module_0.get_in(var_19, var_21)
    assert var_22 == 10
    var_23 = [var_18, var_2]
    var_24 = 20
    var_25 = [var_20, var_24]
    var_26 = [var_25]
    var_27 = module_0.get_in(var_23, var_26)
    assert var_27 == 20
    var_28 = [var_18, var_2, var_7]
    var_29 = 30
    var_30 = [var_20, var_24, var_29]
    var_31 = [var_30]
    var_32 = [var_31]
    var_33 = module_0.get_in(var_28, var_32)
    assert var_33 == 30
    var_34 = {var_5: var_2}
    var_35 = {var_11: var_7}
    var_36 = [var_34, var_35]
    var_37 = {var_0: var_36}
    var_38 = [var_0, var_18, var_5]
    var_39 = module_0.get_in(var_38, var_37)
    assert var_39 == 1
    var_40 = [var_0, var_2, var_11]
    var_41 = module_0.get_in(var_40, var_37)
    assert var_41 == 2
    var_42 = 'x'
    var_43 = [var_42]
    var_44 = {}
    var_45 = module_0.get_in(var_43, var_44)
    assert var_45 is None
    var_46 = [var_42]
    var_47 = {}
    var_48 = module_0.get_in(var_46, var_47, var_18)
    assert var_48 == 0
    var_49 = [var_0, var_42]
    var_50 = {}
    var_51 = {var_0: var_50}
    var_52 = 'missing'
    var_53 = module_0.get_in(var_49, var_51, var_52)
    assert var_53 == 'missing'
    var_54 = 'x'
    var_55 = [var_54]
    var_56 = {}
    var_57 = True
    var_58 = module_0.get_in(var_55, var_56, no_default=var_57)
    var_59 = 0
    var_60 = [var_59]
    var_61 = []
    var_62 = True
    var_63 = module_0.get_in(var_60, var_61, no_default=var_62)
    var_64 = 'name'
    var_65 = 'purchase'
    var_66 = 'credit card'
    var_67 = 'Alice'
    var_68 = 'items'
    var_69 = 'costs'
    var_70 = 'Apple'
    var_71 = 'Orange'
    var_72 = [var_70, var_71]
    var_73 = 0.5
    var_74 = 1.25
    var_75 = [var_73, var_74]
    var_76 = {var_68: var_72, var_69: var_75}
    var_77 = '5555-1234-1234-1234'
    var_78 = {var_64: var_67, var_65: var_76, var_66: var_77}
    var_79 = [var_65, var_68, var_18]
    var_80 = module_0.get_in(var_79, var_78)
    assert var_80 == 'Apple'
    var_81 = [var_64]
    var_82 = module_0.get_in(var_81, var_78)
    assert var_82 == 'Alice'
    var_83 = 'total'
    var_84 = [var_65, var_83]
    var_85 = module_0.get_in(var_84, var_78)
    assert var_85 is None
    var_86 = [var_65, var_68, var_20]
    var_87 = module_0.get_in(var_86, var_78)
    assert var_87 is None
    var_88 = [var_65, var_83]
    var_89 = module_0.get_in(var_88, var_78, var_18)
    assert var_89 == 0
    var_90 = {var_59: var_61}
    var_91 = []
    var_92 = module_0.get_in(var_91, var_90)
    var_93 = None
    var_94 = {var_59: var_93}
    var_95 = [var_59]
    var_96 = module_0.get_in(var_95, var_94)
    assert var_96 is None
    var_97 = 'one'
    var_98 = 'two'
    var_99 = {var_61: var_97, var_7: var_98}
    var_100 = [var_61]
    var_101 = module_0.get_in(var_100, var_99)
    assert var_101 == 'one'
    var_102 = (var_59, var_5)
    var_103 = 'tuple_key'
    var_104 = {var_102: var_103}
    var_105 = (var_59, var_5)
    var_106 = [var_105]
    var_107 = module_0.get_in(var_106, var_104)
    assert var_107 == 'tuple_key'



# Parsed testcases at query #6
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = 1
    var_3 = {var_0: var_2}
    var_4 = module_0.get_in(var_1, var_3)
    assert var_4 == 1
    var_5 = 'b'
    var_6 = [var_0, var_5]
    var_7 = 2
    var_8 = {var_5: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.get_in(var_6, var_9)
    assert var_10 == 2
    var_11 = 'c'
    var_12 = [var_0, var_5, var_11]
    var_13 = 3
    var_14 = {var_11: var_13}
    var_15 = {var_5: var_14}
    var_16 = {var_0: var_15}
    var_17 = module_0.get_in(var_12, var_16)
    assert var_17 == 3
    var_18 = 0
    var_19 = [var_18]
    var_20 = 10
    var_21 = [var_20]
    var_22 = module_0.get_in(var_19, var_21)
    assert var_22 == 10
    var_23 = [var_18, var_2]
    var_24 = [var_2, var_7]
    var_25 = [var_24]
    var_26 = module_0.get_in(var_23, var_25)
    assert var_26 == 2
    var_27 = [var_18, var_2, var_7]
    var_28 = [var_2, var_7, var_13]
    var_29 = [var_28]
    var_30 = [var_29]
    var_31 = module_0.get_in(var_27, var_30)
    assert var_31 == 3
    var_32 = [var_0, var_18]
    var_33 = [var_2, var_7, var_13]
    var_34 = {var_0: var_33}
    var_35 = module_0.get_in(var_32, var_34)
    assert var_35 == 1
    var_36 = [var_18, var_5]
    var_37 = 5
    var_38 = {var_5: var_37}
    var_39 = [var_38]
    var_40 = module_0.get_in(var_36, var_39)
    assert var_40 == 5
    var_41 = 'x'
    var_42 = [var_41]
    var_43 = {var_0: var_2}
    var_44 = module_0.get_in(var_42, var_43)
    assert var_44 is None
    var_45 = [var_41]
    var_46 = {var_0: var_2}
    var_47 = 'not found'
    var_48 = module_0.get_in(var_45, var_46, var_47)
    assert var_48 == 'not found'
    var_49 = [var_0, var_41]
    var_50 = {var_5: var_2}
    var_51 = {var_0: var_50}
    var_52 = module_0.get_in(var_49, var_51, var_18)
    assert var_52 == 0
    var_53 = 'x'
    var_54 = [var_53]
    var_55 = 'a'
    var_56 = 1
    var_57 = {var_55: var_56}
    var_58 = True
    var_59 = module_0.get_in(var_54, var_57, no_default=var_58)
    var_60 = 5
    var_61 = [var_60]
    var_62 = 1
    var_63 = 2
    var_64 = 3
    var_65 = [var_62, var_63, var_64]
    var_66 = True
    var_67 = module_0.get_in(var_61, var_65, no_default=var_66)
    var_68 = 'a'
    var_69 = 'b'
    var_70 = [var_68, var_69]
    var_71 = 1
    var_72 = {var_68: var_71}
    var_73 = True
    var_74 = module_0.get_in(var_70, var_72, no_default=var_73)
    var_75 = []
    var_76 = {var_68: var_70}
    var_77 = module_0.get_in(var_75, var_76)
    var_78 = []
    var_79 = [var_70, var_67, var_13]
    var_80 = module_0.get_in(var_78, var_79)
    var_81 = [var_68]
    var_82 = None
    var_83 = 'default'
    var_84 = module_0.get_in(var_81, var_82, var_83)
    assert var_84 == 'default'
    var_85 = [var_18]
    var_86 = '0'
    var_87 = 'value'
    var_88 = {var_86: var_87}
    var_89 = module_0.get_in(var_85, var_88, var_47)
    assert var_89 == 'not found'
    var_90 = [var_86]
    var_91 = [var_70, var_67, var_13]
    var_92 = module_0.get_in(var_90, var_91, var_47)
    assert var_92 == 'not found'



# Parsed testcases at query #7
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 42
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = {var_1: var_11}
    var_13 = [var_9, var_10, var_12]
    var_14 = {var_0: var_13}
    var_15 = [var_0, var_10, var_1]
    var_16 = module_0.get_in(var_15, var_14)
    assert var_16 == 3
    var_17 = {var_0: var_9}
    var_18 = [var_1]
    var_19 = module_0.get_in(var_18, var_17)
    assert var_19 is None
    var_20 = [var_1]
    var_21 = 'default'
    var_22 = module_0.get_in(var_20, var_17, var_21)
    assert var_22 == 'default'
    var_23 = {var_0: var_9}
    var_24 = 'b'
    var_25 = [var_24]
    var_26 = True
    var_27 = module_0.get_in(var_25, var_23, no_default=var_26)
    var_28 = [var_9, var_10, var_11]
    var_29 = 5
    var_30 = [var_29]
    var_31 = True
    var_32 = module_0.get_in(var_30, var_28, no_default=var_31)
    var_33 = {var_30: var_9}
    var_34 = {var_29: var_33}
    var_35 = [var_29, var_31]
    var_36 = 'missing'
    var_37 = module_0.get_in(var_35, var_34, var_36)
    assert var_37 == 'missing'
    var_38 = {var_29: var_9}
    var_39 = []
    var_40 = module_0.get_in(var_39, var_38)
    var_41 = {var_30: var_9}
    var_42 = {var_31: var_10}
    var_43 = [var_41, var_42]
    var_44 = {var_29: var_43}
    var_45 = 0
    var_46 = [var_29, var_45, var_30]
    var_47 = module_0.get_in(var_46, var_44)
    assert var_47 == 1
    var_48 = [var_29, var_9, var_31]
    var_49 = module_0.get_in(var_48, var_44)
    assert var_49 == 2
    var_50 = None
    var_51 = {var_29: var_50}
    var_52 = [var_29, var_30]
    var_53 = module_0.get_in(var_52, var_51)
    assert var_53 is None
    var_54 = [var_29]
    var_55 = {}
    var_56 = module_0.get_in(var_54, var_55, var_21)
    assert var_56 == 'default'
    var_57 = 'value'
    var_58 = {var_10: var_57}
    var_59 = {var_9: var_58}
    var_60 = [var_9, var_10]
    var_61 = module_0.get_in(var_60, var_59)
    assert var_61 == 'value'



# Parsed testcases at query #8
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 42
    var_9 = 'items'
    var_10 = 'name'
    var_11 = 'apple'
    var_12 = {var_10: var_11}
    var_13 = 'orange'
    var_14 = {var_10: var_13}
    var_15 = [var_12, var_14]
    var_16 = {var_9: var_15}
    var_17 = 0
    var_18 = [var_9, var_17, var_10]
    var_19 = module_0.get_in(var_18, var_16)
    assert var_19 == 'apple'
    var_20 = 1
    var_21 = {var_1: var_20}
    var_22 = {var_0: var_21}
    var_23 = [var_0, var_2]
    var_24 = module_0.get_in(var_23, var_22)
    assert var_24 is None
    var_25 = [var_0, var_2]
    var_26 = 'not found'
    var_27 = module_0.get_in(var_25, var_22, var_26)
    assert var_27 == 'not found'
    var_28 = {var_1: var_20}
    var_29 = {var_0: var_28}
    var_30 = 'a'
    var_31 = 'c'
    var_32 = [var_30, var_31]
    var_33 = True
    var_34 = module_0.get_in(var_32, var_29, no_default=var_33)
    var_35 = [var_30, var_31, var_32]
    var_36 = {var_9: var_35}
    var_37 = 5
    var_38 = [var_9, var_37]
    var_39 = module_0.get_in(var_38, var_36)
    assert var_39 is None
    var_40 = [var_30, var_31, var_32]
    var_41 = {var_9: var_40}
    var_42 = 'items'
    var_43 = 5
    var_44 = [var_42, var_43]
    var_45 = True
    var_46 = module_0.get_in(var_44, var_41, no_default=var_45)
    var_47 = {var_42: var_20}
    var_48 = []
    var_49 = module_0.get_in(var_48, var_47)
    var_50 = 2
    var_51 = 'value'
    var_52 = {var_43: var_51}
    var_53 = [var_20, var_50, var_52]
    var_54 = {var_42: var_53}
    var_55 = [var_54]
    var_56 = [var_17, var_42, var_50, var_43]
    var_57 = module_0.get_in(var_56, var_55)
    assert var_57 == 'value'
    var_58 = 'x'
    var_59 = 'y'
    var_60 = 10
    var_61 = {var_59: var_60}
    var_62 = {var_58: var_61}
    var_63 = 'z'
    var_64 = [var_58, var_63]
    var_65 = 99
    var_66 = module_0.get_in(var_64, var_62, var_65)
    assert var_66 == 99
    var_67 = [var_42, var_43]
    var_68 = module_0.get_in(var_67, var_62, var_65)
    assert var_68 == 99
    var_69 = {var_42: var_45}
    var_70 = [var_42, var_43]
    var_71 = module_0.get_in(var_70, var_69)
    assert var_71 is None
    var_72 = {var_42: var_45}
    var_73 = 'a'
    var_74 = 'b'
    var_75 = [var_73, var_74]
    var_76 = True
    var_77 = module_0.get_in(var_75, var_72, no_default=var_76)



# Parsed testcases at query #9
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 42
    var_9 = 'items'
    var_10 = 'id'
    var_11 = 1
    var_12 = {var_10: var_11}
    var_13 = 2
    var_14 = {var_10: var_13}
    var_15 = [var_12, var_14]
    var_16 = {var_9: var_15}
    var_17 = 0
    var_18 = [var_9, var_17, var_10]
    var_19 = module_0.get_in(var_18, var_16)
    assert var_19 == 1
    var_20 = [var_9, var_11, var_10]
    var_21 = module_0.get_in(var_20, var_16)
    assert var_21 == 2
    var_22 = {var_0: var_11}
    var_23 = [var_1]
    var_24 = module_0.get_in(var_23, var_22)
    assert var_24 is None
    var_25 = [var_1]
    var_26 = 'not found'
    var_27 = module_0.get_in(var_25, var_22, var_26)
    assert var_27 == 'not found'
    var_28 = {var_0: var_11}
    var_29 = 'b'
    var_30 = [var_29]
    var_31 = True
    var_32 = module_0.get_in(var_30, var_28, no_default=var_31)
    var_33 = 3
    var_34 = [var_11, var_13, var_33]
    var_35 = 5
    var_36 = [var_35]
    var_37 = True
    var_38 = module_0.get_in(var_36, var_34, no_default=var_37)
    var_39 = {var_36: var_11}
    var_40 = {var_35: var_39}
    var_41 = [var_35, var_37]
    var_42 = module_0.get_in(var_41, var_40)
    assert var_42 is None
    var_43 = [var_35, var_37]
    var_44 = module_0.get_in(var_43, var_40, var_17)
    assert var_44 == 0
    var_45 = {var_35: var_11}
    var_46 = []
    var_47 = module_0.get_in(var_46, var_45)
    var_48 = 'users'
    var_49 = 'name'
    var_50 = 'age'
    var_51 = 'Alice'
    var_52 = 30
    var_53 = {var_49: var_51, var_50: var_52}
    var_54 = 'Bob'
    var_55 = 25
    var_56 = {var_49: var_54, var_50: var_55}
    var_57 = [var_53, var_56]
    var_58 = {var_48: var_57}
    var_59 = [var_48, var_17, var_49]
    var_60 = module_0.get_in(var_59, var_58)
    assert var_60 == 'Alice'
    var_61 = [var_48, var_11, var_50]
    var_62 = module_0.get_in(var_61, var_58)
    assert var_62 == 25
    var_63 = None
    var_64 = {var_35: var_63}
    var_65 = [var_35, var_36]
    var_66 = module_0.get_in(var_65, var_64)
    assert var_66 is None
    var_67 = 'key'
    var_68 = [var_67]
    var_69 = {}
    var_70 = module_0.get_in(var_68, var_69)
    assert var_70 is None
    var_71 = [var_17]
    var_72 = []
    var_73 = module_0.get_in(var_71, var_72)
    assert var_73 is None
    var_74 = {var_35: var_11}
    var_75 = [var_36]
    var_76 = 'DEFAULT'
    var_77 = module_0.get_in(var_75, var_74, var_76)
    assert var_77 == 'DEFAULT'
    var_78 = 'deep'
    var_79 = {var_33: var_78}
    var_80 = {var_13: var_79}
    var_81 = {var_11: var_80}
    var_82 = [var_11, var_13, var_33]
    var_83 = module_0.get_in(var_82, var_81)
    assert var_83 == 'deep'



# Parsed testcases at query #10
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = [var_0]
    var_12 = module_0.get_in(var_11, var_6)
    var_13 = {var_1: var_3}
    var_14 = 2
    var_15 = {var_2: var_14}
    var_16 = [var_13, var_15]
    var_17 = {var_0: var_16}
    var_18 = 0
    var_19 = [var_0, var_18, var_1]
    var_20 = module_0.get_in(var_19, var_17)
    assert var_20 == 1
    var_21 = [var_0, var_3, var_2]
    var_22 = module_0.get_in(var_21, var_17)
    assert var_22 == 2
    var_23 = 3
    var_24 = [var_3, var_14, var_23]
    var_25 = {var_2: var_24}
    var_26 = {var_1: var_25}
    var_27 = [var_26]
    var_28 = {var_0: var_27}
    var_29 = [var_0, var_18, var_1, var_2, var_14]
    var_30 = module_0.get_in(var_29, var_28)
    assert var_30 == 3
    var_31 = {var_0: var_3}
    var_32 = [var_1]
    var_33 = module_0.get_in(var_32, var_31)
    assert var_33 is None
    var_34 = [var_1]
    var_35 = 'not found'
    var_36 = module_0.get_in(var_34, var_31, var_35)
    assert var_36 == 'not found'
    var_37 = [var_1]
    var_38 = module_0.get_in(var_37, var_31, var_18)
    assert var_38 == 0
    var_39 = {var_0: var_3}
    var_40 = 'b'
    var_41 = [var_40]
    var_42 = True
    var_43 = module_0.get_in(var_41, var_39, no_default=var_42)
    var_44 = [var_43, var_14, var_23]
    var_45 = 5
    var_46 = [var_45]
    var_47 = True
    var_48 = module_0.get_in(var_46, var_44, no_default=var_47)
    var_49 = {var_45: var_48}
    var_50 = []
    var_51 = module_0.get_in(var_50, var_49)
    var_52 = []
    var_53 = 'default'
    var_54 = module_0.get_in(var_52, var_49, var_53)
    var_55 = None
    var_56 = {var_45: var_55}
    var_57 = [var_45]
    var_58 = module_0.get_in(var_57, var_56)
    assert var_58 is None
    var_59 = [var_45, var_46]
    var_60 = module_0.get_in(var_59, var_56)
    assert var_60 is None
    var_61 = {}
    var_62 = [var_45]
    var_63 = module_0.get_in(var_62, var_61)
    assert var_63 is None
    var_64 = [var_45]
    var_65 = {}
    var_66 = module_0.get_in(var_64, var_61, var_65)
    var_67 = 'value'
    var_68 = {var_23: var_67}
    var_69 = {var_14: var_68}
    var_70 = {var_48: var_69}
    var_71 = [var_48, var_14, var_23]
    var_72 = module_0.get_in(var_71, var_70)
    assert var_72 == 'value'
    var_73 = [var_48, var_14]
    var_74 = 4
    var_75 = [var_23, var_74]
    var_76 = [var_73, var_75]
    var_77 = 5
    var_78 = 6
    var_79 = [var_77, var_78]
    var_80 = 7
    var_81 = 8
    var_82 = [var_80, var_81]
    var_83 = [var_79, var_82]
    var_84 = [var_76, var_83]
    var_85 = [var_18, var_48, var_18]
    var_86 = module_0.get_in(var_85, var_84)
    assert var_86 == 3
    var_87 = [var_48, var_18, var_48]
    var_88 = module_0.get_in(var_87, var_84)
    assert var_88 == 6
    var_89 = {var_46: var_48}
    var_90 = {var_45: var_89}
    var_91 = 'd'
    var_92 = [var_45, var_47, var_91]
    var_93 = 'missing'
    var_94 = module_0.get_in(var_92, var_90, var_93)
    assert var_94 == 'missing'
    var_95 = {var_46: var_48}
    var_96 = {var_45: var_95}
    var_97 = 'a'
    var_98 = 'c'
    var_99 = 'd'
    var_100 = [var_97, var_98, var_99]
    var_101 = True
    var_102 = module_0.get_in(var_100, var_96, no_default=var_101)



# Parsed testcases at query #11
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = 1
    var_3 = {var_0: var_2}
    var_4 = module_0.get_in(var_1, var_3)
    assert var_4 == 1
    var_5 = 'b'
    var_6 = [var_0, var_5]
    var_7 = 2
    var_8 = {var_5: var_7}
    var_9 = {var_0: var_8}
    var_10 = module_0.get_in(var_6, var_9)
    assert var_10 == 2
    var_11 = 'c'
    var_12 = [var_0, var_5, var_11]
    var_13 = 3
    var_14 = {var_11: var_13}
    var_15 = {var_5: var_14}
    var_16 = {var_0: var_15}
    var_17 = module_0.get_in(var_12, var_16)
    assert var_17 == 3
    var_18 = 0
    var_19 = [var_18]
    var_20 = [var_2, var_7, var_13]
    var_21 = module_0.get_in(var_19, var_20)
    assert var_21 == 1
    var_22 = [var_2, var_18]
    var_23 = [var_2, var_7]
    var_24 = 4
    var_25 = [var_13, var_24]
    var_26 = [var_23, var_25]
    var_27 = module_0.get_in(var_22, var_26)
    assert var_27 == 3
    var_28 = [var_18, var_2]
    var_29 = [var_2, var_7]
    var_30 = [var_13, var_24]
    var_31 = [var_29, var_30]
    var_32 = module_0.get_in(var_28, var_31)
    assert var_32 == 2
    var_33 = {var_5: var_2}
    var_34 = {var_11: var_7}
    var_35 = [var_33, var_34]
    var_36 = {var_0: var_35}
    var_37 = [var_0, var_18, var_5]
    var_38 = module_0.get_in(var_37, var_36)
    assert var_38 == 1
    var_39 = [var_0, var_2, var_11]
    var_40 = module_0.get_in(var_39, var_36)
    assert var_40 == 2
    var_41 = 'x'
    var_42 = [var_41]
    var_43 = {var_0: var_2}
    var_44 = module_0.get_in(var_42, var_43)
    assert var_44 is None
    var_45 = [var_0, var_41]
    var_46 = {var_5: var_2}
    var_47 = {var_0: var_46}
    var_48 = module_0.get_in(var_45, var_47)
    assert var_48 is None
    var_49 = 5
    var_50 = [var_49]
    var_51 = [var_2, var_7, var_13]
    var_52 = module_0.get_in(var_50, var_51)
    assert var_52 is None
    var_53 = [var_0, var_5, var_11]
    var_54 = {}
    var_55 = {var_5: var_54}
    var_56 = {var_0: var_55}
    var_57 = module_0.get_in(var_53, var_56)
    assert var_57 is None
    var_58 = [var_41]
    var_59 = {var_0: var_2}
    var_60 = 'not found'
    var_61 = module_0.get_in(var_58, var_59, var_60)
    assert var_61 == 'not found'
    var_62 = [var_0, var_41]
    var_63 = {var_5: var_2}
    var_64 = {var_0: var_63}
    var_65 = module_0.get_in(var_62, var_64, var_18)
    assert var_65 == 0
    var_66 = [var_49]
    var_67 = [var_2, var_7, var_13]
    var_68 = 'missing'
    var_69 = module_0.get_in(var_66, var_67, var_68)
    assert var_69 == 'missing'
    var_70 = 'x'
    var_71 = [var_70]
    var_72 = 'a'
    var_73 = 1
    var_74 = {var_72: var_73}
    var_75 = True
    var_76 = module_0.get_in(var_71, var_74, no_default=var_75)
    var_77 = 'a'
    var_78 = 'x'
    var_79 = [var_77, var_78]
    var_80 = 'b'
    var_81 = 1
    var_82 = {var_80: var_81}
    var_83 = {var_77: var_82}
    var_84 = True
    var_85 = module_0.get_in(var_79, var_83, no_default=var_84)
    var_86 = 5
    var_87 = [var_86]
    var_88 = 1
    var_89 = 2
    var_90 = 3
    var_91 = [var_88, var_89, var_90]
    var_92 = True
    var_93 = module_0.get_in(var_87, var_91, no_default=var_92)
    var_94 = []
    var_95 = {var_86: var_88}
    var_96 = module_0.get_in(var_94, var_95)
    var_97 = []
    var_98 = [var_88, var_93, var_13]
    var_99 = module_0.get_in(var_97, var_98)
    var_100 = []
    var_101 = 'hello'
    var_102 = module_0.get_in(var_100, var_101)
    assert var_102 == 'hello'
    var_103 = [var_86]
    var_104 = None
    var_105 = module_0.get_in(var_103, var_104)
    assert var_105 is None
    var_106 = [var_86]
    var_107 = 'default'
    var_108 = module_0.get_in(var_106, var_104, var_107)
    assert var_108 == 'default'
    var_109 = 'name'
    var_110 = 'purchase'
    var_111 = 'credit card'
    var_112 = 'Alice'
    var_113 = 'items'
    var_114 = 'costs'
    var_115 = 'Apple'
    var_116 = 'Orange'
    var_117 = [var_115, var_116]
    var_118 = 0.5
    var_119 = 1.25
    var_120 = [var_118, var_119]
    var_121 = {var_113: var_117, var_114: var_120}
    var_122 = '5555-1234-1234-1234'
    var_123 = {var_109: var_112, var_110: var_121, var_111: var_122}
    var_124 = [var_110, var_113, var_18]
    var_125 = module_0.get_in(var_124, var_123)
    assert var_125 == 'Apple'
    var_126 = [var_109]
    var_127 = module_0.get_in(var_126, var_123)
    assert var_127 == 'Alice'
    var_128 = 'total'
    var_129 = [var_110, var_128]
    var_130 = module_0.get_in(var_129, var_123)
    assert var_130 is None
    var_131 = 'apple'
    var_132 = [var_110, var_113, var_131]
    var_133 = module_0.get_in(var_132, var_123)
    assert var_133 is None
    var_134 = 10
    var_135 = [var_110, var_113, var_134]
    var_136 = module_0.get_in(var_135, var_123)
    assert var_136 is None
    var_137 = [var_110, var_128]
    var_138 = module_0.get_in(var_137, var_123, var_18)
    assert var_138 == 0



# Parsed testcases at query #12
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 42
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = {var_1: var_11}
    var_13 = [var_9, var_10, var_12]
    var_14 = {var_0: var_13}
    var_15 = [var_0, var_10, var_1]
    var_16 = module_0.get_in(var_15, var_14)
    assert var_16 == 3
    var_17 = {var_0: var_9}
    var_18 = [var_1]
    var_19 = module_0.get_in(var_18, var_17)
    assert var_19 is None
    var_20 = [var_1]
    var_21 = 'default'
    var_22 = module_0.get_in(var_20, var_17, var_21)
    assert var_22 == 'default'
    var_23 = {var_0: var_9}
    var_24 = 'b'
    var_25 = [var_24]
    var_26 = True
    var_27 = module_0.get_in(var_25, var_23, no_default=var_26)
    var_28 = [var_9, var_10, var_11]
    var_29 = 5
    var_30 = [var_29]
    var_31 = True
    var_32 = module_0.get_in(var_30, var_28, no_default=var_31)
    var_33 = {var_30: var_9}
    var_34 = {var_29: var_33}
    var_35 = [var_29, var_31]
    var_36 = 'missing'
    var_37 = module_0.get_in(var_35, var_34, var_36)
    assert var_37 == 'missing'
    var_38 = {var_29: var_9}
    var_39 = []
    var_40 = module_0.get_in(var_39, var_38)
    var_41 = {var_30: var_9}
    var_42 = {var_31: var_10}
    var_43 = [var_41, var_42]
    var_44 = {var_29: var_43}
    var_45 = 0
    var_46 = [var_29, var_45, var_30]
    var_47 = module_0.get_in(var_46, var_44)
    assert var_47 == 1
    var_48 = [var_29, var_9, var_31]
    var_49 = module_0.get_in(var_48, var_44)
    assert var_49 == 2
    var_50 = None
    var_51 = {var_29: var_50}
    var_52 = [var_29, var_30]
    var_53 = module_0.get_in(var_52, var_51)
    assert var_53 is None
    var_54 = [var_29]
    var_55 = {}
    var_56 = 'not found'
    var_57 = module_0.get_in(var_54, var_55, var_56)
    assert var_57 == 'not found'
    var_58 = {var_10: var_11}
    var_59 = {var_9: var_58}
    var_60 = [var_9, var_10]
    var_61 = module_0.get_in(var_60, var_59)
    assert var_61 == 3



# Parsed testcases at query #13
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = [var_0]
    var_12 = module_0.get_in(var_11, var_6)
    var_13 = {var_1: var_3}
    var_14 = 2
    var_15 = {var_2: var_14}
    var_16 = [var_13, var_15]
    var_17 = {var_0: var_16}
    var_18 = 0
    var_19 = [var_0, var_18, var_1]
    var_20 = module_0.get_in(var_19, var_17)
    assert var_20 == 1
    var_21 = [var_0, var_3, var_2]
    var_22 = module_0.get_in(var_21, var_17)
    assert var_22 == 2
    var_23 = 3
    var_24 = {var_2: var_23}
    var_25 = [var_3, var_14, var_24]
    var_26 = {var_1: var_25}
    var_27 = {var_0: var_26}
    var_28 = [var_0, var_1, var_18]
    var_29 = module_0.get_in(var_28, var_27)
    assert var_29 == 1
    var_30 = [var_0, var_1, var_14, var_2]
    var_31 = module_0.get_in(var_30, var_27)
    assert var_31 == 3
    var_32 = {var_0: var_3}
    var_33 = [var_1]
    var_34 = module_0.get_in(var_33, var_32)
    assert var_34 is None
    var_35 = [var_1]
    var_36 = 'not found'
    var_37 = module_0.get_in(var_35, var_32, var_36)
    assert var_37 == 'not found'
    var_38 = [var_0, var_1]
    var_39 = module_0.get_in(var_38, var_32, var_18)
    assert var_39 == 0
    var_40 = {var_0: var_3}
    var_41 = 'b'
    var_42 = [var_41]
    var_43 = True
    var_44 = module_0.get_in(var_42, var_40, no_default=var_43)
    var_45 = 'a'
    var_46 = 'b'
    var_47 = [var_45, var_46]
    var_48 = True
    var_49 = module_0.get_in(var_47, var_40, no_default=var_48)
    var_50 = [var_48, var_14, var_23]
    var_51 = 5
    var_52 = [var_51]
    var_53 = True
    var_54 = module_0.get_in(var_52, var_50, no_default=var_53)
    var_55 = {var_51: var_54}
    var_56 = 'a'
    var_57 = 'b'
    var_58 = [var_56, var_57]
    var_59 = True
    var_60 = module_0.get_in(var_58, var_55, no_default=var_59)
    var_61 = {var_56: var_59}
    var_62 = []
    var_63 = module_0.get_in(var_62, var_61)
    var_64 = None
    var_65 = {var_56: var_64}
    var_66 = [var_56]
    var_67 = module_0.get_in(var_66, var_65)
    assert var_67 is None
    var_68 = [var_56, var_57]
    var_69 = 'default'
    var_70 = module_0.get_in(var_68, var_65, var_69)
    assert var_70 == 'default'
    var_71 = {}
    var_72 = {var_57: var_71}
    var_73 = {var_56: var_72}
    var_74 = [var_56, var_57, var_58]
    var_75 = module_0.get_in(var_74, var_73)
    assert var_75 is None
    var_76 = [var_56, var_57, var_58]
    var_77 = []
    var_78 = module_0.get_in(var_76, var_73, var_77)
    var_79 = 'deep'
    var_80 = {var_23: var_79}
    var_81 = {var_14: var_80}
    var_82 = {var_59: var_81}
    var_83 = [var_59, var_14, var_23]
    var_84 = module_0.get_in(var_83, var_82)
    assert var_84 == 'deep'
    var_85 = [var_59, var_14, var_23]
    var_86 = {var_56: var_85}
    var_87 = 5
    var_88 = [var_56, var_87]
    var_89 = module_0.get_in(var_88, var_86)
    assert var_89 is None
    var_90 = [var_56, var_87]
    var_91 = 'out of bounds'
    var_92 = module_0.get_in(var_90, var_86, var_91)
    assert var_92 == 'out of bounds'



# Parsed testcases at query #14
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 42
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = {var_1: var_11}
    var_13 = [var_9, var_10, var_12]
    var_14 = {var_0: var_13}
    var_15 = [var_0, var_10, var_1]
    var_16 = module_0.get_in(var_15, var_14)
    assert var_16 == 3
    var_17 = {var_0: var_9}
    var_18 = [var_1]
    var_19 = module_0.get_in(var_18, var_17)
    assert var_19 is None
    var_20 = [var_1]
    var_21 = 'default'
    var_22 = module_0.get_in(var_20, var_17, var_21)
    assert var_22 == 'default'
    var_23 = {var_0: var_9}
    var_24 = 'b'
    var_25 = [var_24]
    var_26 = True
    var_27 = module_0.get_in(var_25, var_23, no_default=var_26)
    var_28 = [var_9, var_10, var_11]
    var_29 = 5
    var_30 = [var_29]
    var_31 = True
    var_32 = module_0.get_in(var_30, var_28, no_default=var_31)
    var_33 = {var_30: var_9}
    var_34 = {var_29: var_33}
    var_35 = [var_29, var_31]
    var_36 = 'missing'
    var_37 = module_0.get_in(var_35, var_34, var_36)
    assert var_37 == 'missing'
    var_38 = {var_29: var_9}
    var_39 = []
    var_40 = module_0.get_in(var_39, var_38)
    var_41 = {var_30: var_9}
    var_42 = {var_31: var_10}
    var_43 = [var_41, var_42]
    var_44 = {var_29: var_43}
    var_45 = 0
    var_46 = [var_29, var_45, var_30]
    var_47 = module_0.get_in(var_46, var_44)
    assert var_47 == 1
    var_48 = [var_29, var_9, var_31]
    var_49 = module_0.get_in(var_48, var_44)
    assert var_49 == 2
    var_50 = None
    var_51 = {var_29: var_50}
    var_52 = [var_29, var_30]
    var_53 = module_0.get_in(var_52, var_51)
    assert var_53 is None
    var_54 = 'key'
    var_55 = [var_54]
    var_56 = {}
    var_57 = module_0.get_in(var_55, var_56)
    assert var_57 is None
    var_58 = [var_45]
    var_59 = []
    var_60 = module_0.get_in(var_58, var_59)
    assert var_60 is None



# Parsed testcases at query #15
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = [var_0]
    var_12 = module_0.get_in(var_11, var_6)
    var_13 = {var_1: var_3}
    var_14 = 2
    var_15 = {var_2: var_14}
    var_16 = [var_13, var_15]
    var_17 = {var_0: var_16}
    var_18 = 0
    var_19 = [var_0, var_18, var_1]
    var_20 = module_0.get_in(var_19, var_17)
    assert var_20 == 1
    var_21 = [var_0, var_3, var_2]
    var_22 = module_0.get_in(var_21, var_17)
    assert var_22 == 2
    var_23 = 3
    var_24 = {var_2: var_23}
    var_25 = [var_3, var_14, var_24]
    var_26 = {var_1: var_25}
    var_27 = {var_0: var_26}
    var_28 = [var_0, var_1, var_18]
    var_29 = module_0.get_in(var_28, var_27)
    assert var_29 == 1
    var_30 = [var_0, var_1, var_14, var_2]
    var_31 = module_0.get_in(var_30, var_27)
    assert var_31 == 3
    var_32 = {var_0: var_3}
    var_33 = [var_1]
    var_34 = module_0.get_in(var_33, var_32)
    assert var_34 is None
    var_35 = [var_1]
    var_36 = 'default'
    var_37 = module_0.get_in(var_35, var_32, var_36)
    assert var_37 == 'default'
    var_38 = [var_1]
    var_39 = []
    var_40 = module_0.get_in(var_38, var_32, var_39)
    var_41 = {var_0: var_3}
    var_42 = 'b'
    var_43 = [var_42]
    var_44 = True
    var_45 = module_0.get_in(var_43, var_41, no_default=var_44)
    var_46 = [var_45, var_14, var_23]
    var_47 = 5
    var_48 = [var_47]
    var_49 = True
    var_50 = module_0.get_in(var_48, var_46, no_default=var_49)
    var_51 = {var_47: var_50}
    var_52 = 'a'
    var_53 = 'b'
    var_54 = [var_52, var_53]
    var_55 = True
    var_56 = module_0.get_in(var_54, var_51, no_default=var_55)
    var_57 = {var_52: var_55}
    var_58 = []
    var_59 = module_0.get_in(var_58, var_57)
    var_60 = {}
    var_61 = {var_53: var_60}
    var_62 = {var_52: var_61}
    var_63 = [var_52, var_53, var_54]
    var_64 = module_0.get_in(var_63, var_62)
    assert var_64 is None
    var_65 = [var_52, var_53, var_54]
    var_66 = module_0.get_in(var_65, var_62, var_18)
    assert var_66 == 0
    var_67 = None
    var_68 = {var_52: var_67}
    var_69 = [var_52]
    var_70 = module_0.get_in(var_69, var_68)
    assert var_70 is None
    var_71 = [var_52, var_53]
    var_72 = module_0.get_in(var_71, var_68, var_36)
    assert var_72 == 'default'
    var_73 = 'value'
    var_74 = {var_23: var_73}
    var_75 = {var_14: var_74}
    var_76 = {var_55: var_75}
    var_77 = [var_55, var_14, var_23]
    var_78 = module_0.get_in(var_77, var_76)
    assert var_78 == 'value'



# Parsed testcases at query #16
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 42
    var_9 = 10
    var_10 = {var_1: var_9}
    var_11 = 20
    var_12 = {var_2: var_11}
    var_13 = [var_10, var_12]
    var_14 = {var_0: var_13}
    var_15 = 0
    var_16 = [var_0, var_15, var_1]
    var_17 = module_0.get_in(var_16, var_14)
    assert var_17 == 10
    var_18 = 1
    var_19 = [var_0, var_18, var_2]
    var_20 = module_0.get_in(var_19, var_14)
    assert var_20 == 20
    var_21 = {var_1: var_18}
    var_22 = {var_0: var_21}
    var_23 = [var_0, var_2]
    var_24 = module_0.get_in(var_23, var_22)
    assert var_24 is None
    var_25 = [var_0, var_2]
    var_26 = 'not found'
    var_27 = module_0.get_in(var_25, var_22, var_26)
    assert var_27 == 'not found'
    var_28 = {var_1: var_18}
    var_29 = {var_0: var_28}
    var_30 = 'a'
    var_31 = 'c'
    var_32 = [var_30, var_31]
    var_33 = True
    var_34 = module_0.get_in(var_32, var_29, no_default=var_33)
    var_35 = 2
    var_36 = 3
    var_37 = [var_18, var_35, var_36]
    var_38 = {var_30: var_37}
    var_39 = 'a'
    var_40 = 5
    var_41 = [var_39, var_40]
    var_42 = True
    var_43 = module_0.get_in(var_41, var_38, no_default=var_42)
    var_44 = [var_18, var_35, var_36]
    var_45 = 4
    var_46 = 5
    var_47 = 6
    var_48 = [var_45, var_46, var_47]
    var_49 = [var_44, var_48]
    var_50 = [var_15, var_18]
    var_51 = module_0.get_in(var_50, var_49)
    assert var_51 == 2
    var_52 = [var_18, var_35]
    var_53 = module_0.get_in(var_52, var_49)
    assert var_53 == 6
    var_54 = 'd'
    var_55 = {var_40: var_18}
    var_56 = {var_41: var_35}
    var_57 = [var_55, var_56]
    var_58 = 'e'
    var_59 = [var_36, var_45, var_46]
    var_60 = {var_58: var_59}
    var_61 = {var_39: var_57, var_54: var_60}
    var_62 = [var_39, var_15, var_40]
    var_63 = module_0.get_in(var_62, var_61)
    assert var_63 == 1
    var_64 = [var_54, var_58, var_35]
    var_65 = module_0.get_in(var_64, var_61)
    assert var_65 == 5
    var_66 = {var_39: var_18}
    var_67 = []
    var_68 = module_0.get_in(var_67, var_66)
    var_69 = None
    var_70 = {var_40: var_69}
    var_71 = {var_39: var_70}
    var_72 = [var_39, var_40]
    var_73 = module_0.get_in(var_72, var_71)
    assert var_73 is None
    var_74 = {var_40: var_69}
    var_75 = {var_39: var_74}
    var_76 = [var_39, var_40]
    var_77 = 'default'
    var_78 = module_0.get_in(var_76, var_75, var_77)
    assert var_78 is None
    var_79 = {var_39: var_42}
    var_80 = [var_39, var_40]
    var_81 = module_0.get_in(var_80, var_79)
    assert var_81 is None
    var_82 = [var_39, var_40]
    var_83 = 'error'
    var_84 = module_0.get_in(var_82, var_79, var_83)
    assert var_84 == 'error'



# Parsed testcases at query #17
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 42
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = {var_1: var_11}
    var_13 = [var_9, var_10, var_12]
    var_14 = {var_0: var_13}
    var_15 = [var_0, var_10, var_1]
    var_16 = module_0.get_in(var_15, var_14)
    assert var_16 == 3
    var_17 = {var_0: var_9}
    var_18 = [var_1]
    var_19 = module_0.get_in(var_18, var_17)
    assert var_19 is None
    var_20 = [var_1]
    var_21 = 'default'
    var_22 = module_0.get_in(var_20, var_17, var_21)
    assert var_22 == 'default'
    var_23 = {var_0: var_9}
    var_24 = 'b'
    var_25 = [var_24]
    var_26 = True
    var_27 = module_0.get_in(var_25, var_23, no_default=var_26)
    var_28 = [var_9, var_10, var_11]
    var_29 = 5
    var_30 = [var_29]
    var_31 = True
    var_32 = module_0.get_in(var_30, var_28, no_default=var_31)
    var_33 = {var_30: var_9}
    var_34 = {var_29: var_33}
    var_35 = [var_29, var_31]
    var_36 = 'missing'
    var_37 = module_0.get_in(var_35, var_34, var_36)
    assert var_37 == 'missing'
    var_38 = {var_29: var_9}
    var_39 = []
    var_40 = module_0.get_in(var_39, var_38)
    var_41 = {var_30: var_9}
    var_42 = {var_31: var_10}
    var_43 = [var_41, var_42]
    var_44 = {var_29: var_43}
    var_45 = 0
    var_46 = [var_29, var_45, var_30]
    var_47 = module_0.get_in(var_46, var_44)
    assert var_47 == 1
    var_48 = [var_29, var_9, var_31]
    var_49 = module_0.get_in(var_48, var_44)
    assert var_49 == 2
    var_50 = None
    var_51 = {var_29: var_50}
    var_52 = [var_29, var_30]
    var_53 = module_0.get_in(var_52, var_51)
    assert var_53 is None
    var_54 = [var_29]
    var_55 = {}
    var_56 = module_0.get_in(var_54, var_55, var_21)
    assert var_56 == 'default'
    var_57 = 'value'
    var_58 = {var_11: var_57}
    var_59 = {var_10: var_58}
    var_60 = {var_9: var_59}
    var_61 = [var_9, var_10, var_11]
    var_62 = module_0.get_in(var_61, var_60)
    assert var_62 == 'value'



# Parsed testcases at query #18
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 42
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = {var_1: var_11}
    var_13 = [var_9, var_10, var_12]
    var_14 = {var_0: var_13}
    var_15 = [var_0, var_10, var_1]
    var_16 = module_0.get_in(var_15, var_14)
    assert var_16 == 3
    var_17 = {var_0: var_9}
    var_18 = [var_1]
    var_19 = module_0.get_in(var_18, var_17)
    assert var_19 is None
    var_20 = [var_1]
    var_21 = 'default'
    var_22 = module_0.get_in(var_20, var_17, var_21)
    assert var_22 == 'default'
    var_23 = {var_0: var_9}
    var_24 = 'b'
    var_25 = [var_24]
    var_26 = True
    var_27 = module_0.get_in(var_25, var_23, no_default=var_26)
    var_28 = [var_9, var_10, var_11]
    var_29 = 5
    var_30 = [var_29]
    var_31 = True
    var_32 = module_0.get_in(var_30, var_28, no_default=var_31)
    var_33 = {var_30: var_9}
    var_34 = {var_29: var_33}
    var_35 = [var_29, var_31]
    var_36 = 'missing'
    var_37 = module_0.get_in(var_35, var_34, var_36)
    assert var_37 == 'missing'
    var_38 = {var_29: var_9}
    var_39 = []
    var_40 = module_0.get_in(var_39, var_38)
    var_41 = {var_30: var_9}
    var_42 = {var_31: var_10}
    var_43 = [var_41, var_42]
    var_44 = {var_29: var_43}
    var_45 = 0
    var_46 = [var_29, var_45, var_30]
    var_47 = module_0.get_in(var_46, var_44)
    assert var_47 == 1
    var_48 = [var_29, var_9, var_31]
    var_49 = module_0.get_in(var_48, var_44)
    assert var_49 == 2
    var_50 = None
    var_51 = {var_29: var_50}
    var_52 = [var_29, var_30]
    var_53 = module_0.get_in(var_52, var_51)
    assert var_53 is None
    var_54 = 'key'
    var_55 = [var_54]
    var_56 = {}
    var_57 = module_0.get_in(var_55, var_56, var_21)
    assert var_57 == 'default'
    var_58 = 'value'
    var_59 = {var_11: var_58}
    var_60 = {var_10: var_59}
    var_61 = {var_9: var_60}
    var_62 = [var_9, var_10, var_11]
    var_63 = module_0.get_in(var_62, var_61)
    assert var_63 == 'value'



# Parsed testcases at query #19
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 42
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = [var_0]
    var_12 = module_0.get_in(var_11, var_6)
    var_13 = 1
    var_14 = 2
    var_15 = 3
    var_16 = [var_13, var_14, var_15]
    var_17 = 4
    var_18 = 5
    var_19 = 6
    var_20 = [var_17, var_18, var_19]
    var_21 = [var_16, var_20]
    var_22 = 0
    var_23 = [var_22, var_13]
    var_24 = module_0.get_in(var_23, var_21)
    assert var_24 == 2
    var_25 = [var_13, var_14]
    var_26 = module_0.get_in(var_25, var_21)
    assert var_26 == 6
    var_27 = 10
    var_28 = {var_1: var_27}
    var_29 = 20
    var_30 = {var_2: var_29}
    var_31 = [var_28, var_30]
    var_32 = {var_0: var_31}
    var_33 = [var_0, var_22, var_1]
    var_34 = module_0.get_in(var_33, var_32)
    assert var_34 == 10
    var_35 = [var_0, var_13, var_2]
    var_36 = module_0.get_in(var_35, var_32)
    assert var_36 == 20
    var_37 = 'x'
    var_38 = {var_37: var_13}
    var_39 = 'y'
    var_40 = [var_39]
    var_41 = module_0.get_in(var_40, var_38)
    assert var_41 is None
    var_42 = [var_39]
    var_43 = 'missing'
    var_44 = module_0.get_in(var_42, var_38, var_43)
    assert var_44 == 'missing'
    var_45 = [var_37, var_39]
    var_46 = []
    var_47 = module_0.get_in(var_45, var_38, var_46)
    var_48 = {var_37: var_13}
    var_49 = 'y'
    var_50 = [var_49]
    var_51 = True
    var_52 = module_0.get_in(var_50, var_48, no_default=var_51)
    var_53 = [var_13, var_14, var_15]
    var_54 = 5
    var_55 = [var_54]
    var_56 = True
    var_57 = module_0.get_in(var_55, var_53, no_default=var_56)
    var_58 = {var_54: var_13}
    var_59 = []
    var_60 = module_0.get_in(var_59, var_58)
    var_61 = []
    var_62 = [var_13, var_14, var_15]
    var_63 = module_0.get_in(var_61, var_62)
    var_64 = None
    var_65 = {var_54: var_64}
    var_66 = [var_54]
    var_67 = module_0.get_in(var_66, var_65)
    assert var_67 is None
    var_68 = [var_54, var_55]
    var_69 = 'default'
    var_70 = module_0.get_in(var_68, var_65, var_69)
    assert var_70 == 'default'
    var_71 = 'key'
    var_72 = [var_71]
    var_73 = {}
    var_74 = 'empty'
    var_75 = module_0.get_in(var_72, var_73, var_74)
    assert var_75 == 'empty'
    var_76 = [var_22]
    var_77 = []
    var_78 = module_0.get_in(var_76, var_77, var_74)
    assert var_78 == 'empty'
    var_79 = {}
    var_80 = {var_54: var_79}
    var_81 = [var_54, var_55]
    var_82 = module_0.get_in(var_81, var_80)
    assert var_82 is None
    var_83 = [var_54, var_55]
    var_84 = module_0.get_in(var_83, var_80, var_22)
    assert var_84 == 0



# Parsed testcases at query #20
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 42
    var_9 = 'items'
    var_10 = 'apple'
    var_11 = 'banana'
    var_12 = 'cherry'
    var_13 = [var_10, var_11, var_12]
    var_14 = {var_9: var_13}
    var_15 = 1
    var_16 = [var_9, var_15]
    var_17 = module_0.get_in(var_16, var_14)
    assert var_17 == 'banana'
    var_18 = 10
    var_19 = {var_1: var_18}
    var_20 = 20
    var_21 = {var_1: var_20}
    var_22 = [var_19, var_21]
    var_23 = {var_0: var_22}
    var_24 = [var_0, var_15, var_1]
    var_25 = module_0.get_in(var_24, var_23)
    assert var_25 == 20
    var_26 = 'x'
    var_27 = {var_26: var_15}
    var_28 = 'y'
    var_29 = [var_28]
    var_30 = module_0.get_in(var_29, var_27)
    assert var_30 is None
    var_31 = [var_28]
    var_32 = 'missing'
    var_33 = module_0.get_in(var_31, var_27, var_32)
    assert var_33 == 'missing'
    var_34 = {var_26: var_15}
    var_35 = 'y'
    var_36 = [var_35]
    var_37 = True
    var_38 = module_0.get_in(var_36, var_34, no_default=var_37)
    var_39 = 2
    var_40 = 3
    var_41 = [var_15, var_39, var_40]
    var_42 = 5
    var_43 = [var_42]
    var_44 = True
    var_45 = module_0.get_in(var_43, var_41, no_default=var_44)
    var_46 = {var_42: var_15}
    var_47 = []
    var_48 = module_0.get_in(var_47, var_46)
    var_49 = {var_43: var_15}
    var_50 = {var_42: var_49}
    var_51 = 'd'
    var_52 = [var_42, var_44, var_51]
    var_53 = 'not found'
    var_54 = module_0.get_in(var_52, var_50, var_53)
    assert var_54 == 'not found'
    var_55 = None
    var_56 = {var_42: var_55}
    var_57 = [var_42]
    var_58 = module_0.get_in(var_57, var_56)
    assert var_58 is None
    var_59 = {var_42: var_45}
    var_60 = 'a'
    var_61 = 'b'
    var_62 = [var_60, var_61]
    var_63 = True
    var_64 = module_0.get_in(var_62, var_59, no_default=var_63)
    var_65 = 'key'
    var_66 = [var_65]
    var_67 = {}
    var_68 = 'empty'
    var_69 = module_0.get_in(var_66, var_67, var_68)
    assert var_69 == 'empty'
    var_70 = 'users'
    var_71 = 'name'
    var_72 = 'scores'
    var_73 = 'Alice'
    var_74 = 85
    var_75 = 92
    var_76 = 78
    var_77 = [var_74, var_75, var_76]
    var_78 = {var_71: var_73, var_72: var_77}
    var_79 = 'Bob'
    var_80 = 88
    var_81 = 95
    var_82 = 82
    var_83 = [var_80, var_81, var_82]
    var_84 = {var_71: var_79, var_72: var_83}
    var_85 = [var_78, var_84]
    var_86 = {var_70: var_85}
    var_87 = [var_70, var_15, var_72, var_39]
    var_88 = module_0.get_in(var_87, var_86)
    assert var_88 == 82



# Parsed testcases at query #21
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 42
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = [var_0]
    var_12 = module_0.get_in(var_11, var_6)
    var_13 = 'items'
    var_14 = 'name'
    var_15 = 'price'
    var_16 = 'apple'
    var_17 = 1.0
    var_18 = {var_14: var_16, var_15: var_17}
    var_19 = 'orange'
    var_20 = 1.5
    var_21 = {var_14: var_19, var_15: var_20}
    var_22 = [var_18, var_21]
    var_23 = {var_13: var_22}
    var_24 = 0
    var_25 = [var_13, var_24, var_14]
    var_26 = module_0.get_in(var_25, var_23)
    assert var_26 == 'apple'
    var_27 = [var_13, var_17, var_15]
    var_28 = module_0.get_in(var_27, var_23)
    var_29 = 'x'
    var_30 = 'y'
    var_31 = 'z'
    var_32 = [var_29, var_30, var_31]
    var_33 = {}
    var_34 = module_0.get_in(var_32, var_33)
    assert var_34 is None
    var_35 = 'd'
    var_36 = [var_0, var_1, var_35]
    var_37 = module_0.get_in(var_36, var_6)
    assert var_37 is None
    var_38 = 2
    var_39 = [var_13, var_38]
    var_40 = module_0.get_in(var_39, var_23)
    assert var_40 is None
    var_41 = [var_29, var_30]
    var_42 = {}
    var_43 = 'not found'
    var_44 = module_0.get_in(var_41, var_42, var_43)
    assert var_44 == 'not found'
    var_45 = [var_0, var_1, var_35]
    var_46 = module_0.get_in(var_45, var_6, var_24)
    assert var_46 == 0
    var_47 = 'x'
    var_48 = [var_47]
    var_49 = {}
    var_50 = True
    var_51 = module_0.get_in(var_48, var_49, no_default=var_50)
    var_52 = 'a'
    var_53 = 'b'
    var_54 = 'd'
    var_55 = [var_52, var_53, var_54]
    var_56 = True
    var_57 = module_0.get_in(var_55, var_6, no_default=var_56)
    var_58 = 'items'
    var_59 = 5
    var_60 = [var_58, var_59]
    var_61 = True
    var_62 = module_0.get_in(var_60, var_23, no_default=var_61)
    var_63 = []
    var_64 = module_0.get_in(var_63, var_6)
    var_65 = []
    var_66 = module_0.get_in(var_65, var_23)
    var_67 = [var_58]
    var_68 = None
    var_69 = module_0.get_in(var_67, var_68)
    assert var_69 is None
    var_70 = [var_58]
    var_71 = 'default'
    var_72 = module_0.get_in(var_70, var_68, var_71)
    assert var_72 == 'default'
    var_73 = [var_58, var_59]
    var_74 = {var_58: var_61}
    var_75 = module_0.get_in(var_73, var_74)
    assert var_75 is None
    var_76 = [var_58, var_59]
    var_77 = {var_58: var_61}
    var_78 = True
    var_79 = module_0.get_in(var_76, var_77, no_default=var_78)
    assert var_79 is None
    var_80 = 'purchase'
    var_81 = 'credit card'
    var_82 = 'Alice'
    var_83 = 'costs'
    var_84 = 'Apple'
    var_85 = 'Orange'
    var_86 = [var_84, var_85]
    var_87 = 0.5
    var_88 = 1.25
    var_89 = [var_87, var_88]
    var_90 = {var_13: var_86, var_83: var_89}
    var_91 = '5555-1234-1234-1234'
    var_92 = {var_14: var_82, var_80: var_90, var_81: var_91}
    var_93 = [var_80, var_13, var_24]
    var_94 = module_0.get_in(var_93, var_92)
    assert var_94 == 'Apple'
    var_95 = [var_14]
    var_96 = module_0.get_in(var_95, var_92)
    assert var_96 == 'Alice'
    var_97 = 'total'
    var_98 = [var_80, var_97]
    var_99 = module_0.get_in(var_98, var_92)
    assert var_99 is None
    var_100 = [var_80, var_97]
    var_101 = module_0.get_in(var_100, var_92, var_24)
    assert var_101 == 0



# Parsed testcases at query #22
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 42
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = [var_0]
    var_12 = module_0.get_in(var_11, var_6)
    var_13 = 1
    var_14 = 2
    var_15 = 3
    var_16 = [var_13, var_14, var_15]
    var_17 = 4
    var_18 = 5
    var_19 = 6
    var_20 = [var_17, var_18, var_19]
    var_21 = [var_16, var_20]
    var_22 = 0
    var_23 = [var_22, var_13]
    var_24 = module_0.get_in(var_23, var_21)
    assert var_24 == 2
    var_25 = [var_13, var_14]
    var_26 = module_0.get_in(var_25, var_21)
    assert var_26 == 6
    var_27 = 10
    var_28 = {var_1: var_27}
    var_29 = 20
    var_30 = {var_2: var_29}
    var_31 = [var_28, var_30]
    var_32 = {var_0: var_31}
    var_33 = [var_0, var_22, var_1]
    var_34 = module_0.get_in(var_33, var_32)
    assert var_34 == 10
    var_35 = [var_0, var_13, var_2]
    var_36 = module_0.get_in(var_35, var_32)
    assert var_36 == 20
    var_37 = 'x'
    var_38 = {var_37: var_13}
    var_39 = 'y'
    var_40 = [var_39]
    var_41 = module_0.get_in(var_40, var_38)
    assert var_41 is None
    var_42 = [var_39]
    var_43 = 'not found'
    var_44 = module_0.get_in(var_42, var_38, var_43)
    assert var_44 == 'not found'
    var_45 = [var_39]
    var_46 = module_0.get_in(var_45, var_38, var_22)
    assert var_46 == 0
    var_47 = {var_37: var_13}
    var_48 = 'y'
    var_49 = [var_48]
    var_50 = True
    var_51 = module_0.get_in(var_49, var_47, no_default=var_50)
    var_52 = [var_13, var_14, var_15]
    var_53 = 5
    var_54 = [var_53]
    var_55 = True
    var_56 = module_0.get_in(var_54, var_52, no_default=var_55)
    var_57 = {var_53: var_13}
    var_58 = []
    var_59 = module_0.get_in(var_58, var_57)
    var_60 = []
    var_61 = 'default'
    var_62 = module_0.get_in(var_60, var_57, var_61)
    var_63 = {}
    var_64 = {var_53: var_63}
    var_65 = [var_53, var_54]
    var_66 = module_0.get_in(var_65, var_64)
    assert var_66 is None
    var_67 = [var_53, var_54]
    var_68 = 'missing'
    var_69 = module_0.get_in(var_67, var_64, var_68)
    assert var_69 == 'missing'
    var_70 = None
    var_71 = {var_53: var_70}
    var_72 = [var_53]
    var_73 = module_0.get_in(var_72, var_71)
    assert var_73 is None
    var_74 = 'a'
    var_75 = 'b'
    var_76 = [var_74, var_75]
    var_77 = True
    var_78 = module_0.get_in(var_76, var_71, no_default=var_77)
    var_79 = 'value'
    var_80 = {var_15: var_79}
    var_81 = {var_14: var_80}
    var_82 = {var_13: var_81}
    var_83 = [var_13, var_14, var_15]
    var_84 = module_0.get_in(var_83, var_82)
    assert var_84 == 'value'
    var_85 = [var_13, var_14, var_15]
    var_86 = [var_85]
    var_87 = [var_22, var_74]
    var_88 = module_0.get_in(var_87, var_86)
    assert var_88 is None
    var_89 = 0
    var_90 = 'a'
    var_91 = [var_89, var_90]
    var_92 = True
    var_93 = module_0.get_in(var_91, var_86, no_default=var_92)



# Parsed testcases at query #23
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 42
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = [var_0]
    var_12 = module_0.get_in(var_11, var_6)
    var_13 = 1
    var_14 = 2
    var_15 = 3
    var_16 = [var_13, var_14, var_15]
    var_17 = 4
    var_18 = 5
    var_19 = 6
    var_20 = [var_17, var_18, var_19]
    var_21 = [var_16, var_20]
    var_22 = 0
    var_23 = [var_22, var_13]
    var_24 = module_0.get_in(var_23, var_21)
    assert var_24 == 2
    var_25 = [var_13, var_14]
    var_26 = module_0.get_in(var_25, var_21)
    assert var_26 == 6
    var_27 = {var_1: var_13}
    var_28 = {var_2: var_14}
    var_29 = [var_27, var_28]
    var_30 = {var_0: var_29}
    var_31 = [var_0, var_22, var_1]
    var_32 = module_0.get_in(var_31, var_30)
    assert var_32 == 1
    var_33 = [var_0, var_13, var_2]
    var_34 = module_0.get_in(var_33, var_30)
    assert var_34 == 2
    var_35 = {var_0: var_13}
    var_36 = [var_1]
    var_37 = module_0.get_in(var_36, var_35)
    assert var_37 is None
    var_38 = [var_1]
    var_39 = 'missing'
    var_40 = module_0.get_in(var_38, var_35, var_39)
    assert var_40 == 'missing'
    var_41 = [var_0, var_1]
    var_42 = module_0.get_in(var_41, var_35, var_22)
    assert var_42 == 0
    var_43 = {var_0: var_13}
    var_44 = 'b'
    var_45 = [var_44]
    var_46 = True
    var_47 = module_0.get_in(var_45, var_43, no_default=var_46)
    var_48 = [var_13, var_14, var_15]
    var_49 = 5
    var_50 = [var_49]
    var_51 = True
    var_52 = module_0.get_in(var_50, var_48, no_default=var_51)
    var_53 = {var_49: var_13}
    var_54 = []
    var_55 = module_0.get_in(var_54, var_53)
    var_56 = None
    var_57 = {var_49: var_56}
    var_58 = [var_49]
    var_59 = module_0.get_in(var_58, var_57)
    assert var_59 is None
    var_60 = [var_49, var_50]
    var_61 = 'default'
    var_62 = module_0.get_in(var_60, var_57, var_61)
    assert var_62 == 'default'
    var_63 = {}
    var_64 = {var_50: var_63}
    var_65 = {var_49: var_64}
    var_66 = [var_49, var_50, var_51]
    var_67 = module_0.get_in(var_66, var_65)
    assert var_67 is None
    var_68 = [var_49, var_50, var_51]
    var_69 = module_0.get_in(var_68, var_65, var_22)
    assert var_69 == 0
    var_70 = 'deep'
    var_71 = {var_15: var_70}
    var_72 = {var_14: var_71}
    var_73 = {var_13: var_72}
    var_74 = [var_13, var_14, var_15]
    var_75 = module_0.get_in(var_74, var_73)
    assert var_75 == 'deep'
    var_76 = (var_49, var_50)
    var_77 = 'value'
    var_78 = {var_76: var_77}
    var_79 = (var_49, var_50)
    var_80 = [var_79]
    var_81 = module_0.get_in(var_80, var_78)
    assert var_81 == 'value'



# Parsed testcases at query #24
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 1
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = [var_0]
    var_12 = module_0.get_in(var_11, var_6)
    var_13 = {var_1: var_3}
    var_14 = 2
    var_15 = {var_2: var_14}
    var_16 = [var_13, var_15]
    var_17 = {var_0: var_16}
    var_18 = 0
    var_19 = [var_0, var_18, var_1]
    var_20 = module_0.get_in(var_19, var_17)
    assert var_20 == 1
    var_21 = [var_0, var_3, var_2]
    var_22 = module_0.get_in(var_21, var_17)
    assert var_22 == 2
    var_23 = 3
    var_24 = {var_2: var_23}
    var_25 = [var_3, var_14, var_24]
    var_26 = {var_1: var_25}
    var_27 = {var_0: var_26}
    var_28 = [var_0, var_1, var_18]
    var_29 = module_0.get_in(var_28, var_27)
    assert var_29 == 1
    var_30 = [var_0, var_1, var_14, var_2]
    var_31 = module_0.get_in(var_30, var_27)
    assert var_31 == 3
    var_32 = {var_0: var_3}
    var_33 = [var_1]
    var_34 = module_0.get_in(var_33, var_32)
    assert var_34 is None
    var_35 = [var_1]
    var_36 = 'missing'
    var_37 = module_0.get_in(var_35, var_32, var_36)
    assert var_37 == 'missing'
    var_38 = [var_1]
    var_39 = module_0.get_in(var_38, var_32, var_18)
    assert var_39 == 0
    var_40 = {var_0: var_3}
    var_41 = 'b'
    var_42 = [var_41]
    var_43 = True
    var_44 = module_0.get_in(var_42, var_40, no_default=var_43)
    var_45 = [var_44, var_14, var_23]
    var_46 = 5
    var_47 = [var_46]
    var_48 = True
    var_49 = module_0.get_in(var_47, var_45, no_default=var_48)
    var_50 = {var_46: var_49}
    var_51 = []
    var_52 = module_0.get_in(var_51, var_50)
    var_53 = {var_47: var_49}
    var_54 = {var_46: var_53}
    var_55 = 'd'
    var_56 = [var_46, var_48, var_55]
    var_57 = module_0.get_in(var_56, var_54)
    assert var_57 is None
    var_58 = [var_46, var_48, var_55]
    var_59 = module_0.get_in(var_58, var_54, var_36)
    assert var_59 == 'missing'
    var_60 = None
    var_61 = {var_46: var_60}
    var_62 = [var_46]
    var_63 = module_0.get_in(var_62, var_61)
    assert var_63 is None
    var_64 = [var_46, var_47]
    var_65 = module_0.get_in(var_64, var_61)
    assert var_65 is None
    var_66 = [var_46]
    var_67 = {}
    var_68 = module_0.get_in(var_66, var_67)
    assert var_68 is None
    var_69 = [var_18]
    var_70 = []
    var_71 = module_0.get_in(var_69, var_70)
    assert var_71 is None
    var_72 = [var_18]
    var_73 = []
    var_74 = 'empty'
    var_75 = module_0.get_in(var_72, var_73, var_74)
    assert var_75 == 'empty'
    var_76 = 'users'
    var_77 = 'name'
    var_78 = 'scores'
    var_79 = 'Alice'
    var_80 = 85
    var_81 = 92
    var_82 = 78
    var_83 = [var_80, var_81, var_82]
    var_84 = {var_77: var_79, var_78: var_83}
    var_85 = 'Bob'
    var_86 = 88
    var_87 = 95
    var_88 = 82
    var_89 = [var_86, var_87, var_88]
    var_90 = {var_77: var_85, var_78: var_89}
    var_91 = [var_84, var_90]
    var_92 = {var_76: var_91}
    var_93 = [var_76, var_18, var_77]
    var_94 = module_0.get_in(var_93, var_92)
    assert var_94 == 'Alice'
    var_95 = [var_76, var_49, var_78, var_49]
    var_96 = module_0.get_in(var_95, var_92)
    assert var_96 == 95
    var_97 = 5
    var_98 = [var_76, var_18, var_78, var_97]
    var_99 = module_0.get_in(var_98, var_92)
    assert var_99 is None
    var_100 = [var_76, var_14, var_77]
    var_101 = module_0.get_in(var_100, var_92)
    assert var_101 is None



# Parsed testcases at query #25
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 42
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = {var_1: var_11}
    var_13 = [var_9, var_10, var_12]
    var_14 = {var_0: var_13}
    var_15 = [var_0, var_10, var_1]
    var_16 = module_0.get_in(var_15, var_14)
    assert var_16 == 3
    var_17 = {var_0: var_9}
    var_18 = [var_1]
    var_19 = module_0.get_in(var_18, var_17)
    assert var_19 is None
    var_20 = [var_1]
    var_21 = 'default'
    var_22 = module_0.get_in(var_20, var_17, var_21)
    assert var_22 == 'default'
    var_23 = {var_0: var_9}
    var_24 = 'b'
    var_25 = [var_24]
    var_26 = True
    var_27 = module_0.get_in(var_25, var_23, no_default=var_26)
    var_28 = [var_9, var_10, var_11]
    var_29 = 5
    var_30 = [var_29]
    var_31 = True
    var_32 = module_0.get_in(var_30, var_28, no_default=var_31)
    var_33 = {var_30: var_9}
    var_34 = {var_29: var_33}
    var_35 = 'd'
    var_36 = [var_29, var_31, var_35]
    var_37 = 'missing'
    var_38 = module_0.get_in(var_36, var_34, var_37)
    assert var_38 == 'missing'
    var_39 = {var_29: var_9}
    var_40 = []
    var_41 = module_0.get_in(var_40, var_39)
    var_42 = {var_30: var_9}
    var_43 = {var_31: var_10}
    var_44 = [var_42, var_43]
    var_45 = {var_29: var_44}
    var_46 = 0
    var_47 = [var_29, var_46, var_30]
    var_48 = module_0.get_in(var_47, var_45)
    assert var_48 == 1
    var_49 = [var_29, var_9, var_31]
    var_50 = module_0.get_in(var_49, var_45)
    assert var_50 == 2
    var_51 = None
    var_52 = {var_29: var_51}
    var_53 = [var_29]
    var_54 = module_0.get_in(var_53, var_52)
    assert var_54 is None
    var_55 = [var_9, var_10, var_11]
    var_56 = 4
    var_57 = 5
    var_58 = 6
    var_59 = [var_56, var_57, var_58]
    var_60 = [var_55, var_59]
    var_61 = [var_46, var_9]
    var_62 = module_0.get_in(var_61, var_60)
    assert var_62 == 2
    var_63 = [var_9, var_10]
    var_64 = module_0.get_in(var_63, var_60)
    assert var_64 == 6
    var_65 = {var_29: var_32}
    var_66 = [var_29, var_30]
    var_67 = 'error'
    var_68 = module_0.get_in(var_66, var_65, var_67)
    assert var_68 == 'error'



# Parsed testcases at query #26
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 42
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = [var_0]
    var_12 = module_0.get_in(var_11, var_6)
    var_13 = 1
    var_14 = 2
    var_15 = 3
    var_16 = [var_13, var_14, var_15]
    var_17 = 4
    var_18 = 5
    var_19 = 6
    var_20 = [var_17, var_18, var_19]
    var_21 = [var_16, var_20]
    var_22 = 0
    var_23 = [var_22, var_13]
    var_24 = module_0.get_in(var_23, var_21)
    assert var_24 == 2
    var_25 = [var_13, var_14]
    var_26 = module_0.get_in(var_25, var_21)
    assert var_26 == 6
    var_27 = {var_1: var_13}
    var_28 = {var_2: var_14}
    var_29 = [var_27, var_28]
    var_30 = {var_0: var_29}
    var_31 = [var_0, var_22, var_1]
    var_32 = module_0.get_in(var_31, var_30)
    assert var_32 == 1
    var_33 = [var_0, var_13, var_2]
    var_34 = module_0.get_in(var_33, var_30)
    assert var_34 == 2
    var_35 = {var_0: var_13}
    var_36 = [var_1]
    var_37 = module_0.get_in(var_36, var_35)
    assert var_37 is None
    var_38 = [var_1]
    var_39 = 'not found'
    var_40 = module_0.get_in(var_38, var_35, var_39)
    assert var_40 == 'not found'
    var_41 = [var_1]
    var_42 = module_0.get_in(var_41, var_35, var_22)
    assert var_42 == 0
    var_43 = {var_0: var_13}
    var_44 = 'b'
    var_45 = [var_44]
    var_46 = True
    var_47 = module_0.get_in(var_45, var_43, no_default=var_46)
    var_48 = [var_13, var_14, var_15]
    var_49 = 5
    var_50 = [var_49]
    var_51 = True
    var_52 = module_0.get_in(var_50, var_48, no_default=var_51)
    var_53 = {var_49: var_13}
    var_54 = []
    var_55 = module_0.get_in(var_54, var_53)
    var_56 = {}
    var_57 = {var_50: var_56}
    var_58 = {var_49: var_57}
    var_59 = [var_49, var_50, var_51]
    var_60 = module_0.get_in(var_59, var_58)
    assert var_60 is None
    var_61 = [var_49, var_50, var_51]
    var_62 = 'missing'
    var_63 = module_0.get_in(var_61, var_58, var_62)
    assert var_63 == 'missing'
    var_64 = None
    var_65 = {var_49: var_64}
    var_66 = [var_49]
    var_67 = module_0.get_in(var_66, var_65)
    assert var_67 is None
    var_68 = [var_49, var_50]
    var_69 = module_0.get_in(var_68, var_65)
    assert var_69 is None
    var_70 = 'key'
    var_71 = [var_70]
    var_72 = {}
    var_73 = module_0.get_in(var_71, var_72)
    assert var_73 is None
    var_74 = [var_22]
    var_75 = []
    var_76 = module_0.get_in(var_74, var_75)
    assert var_76 is None
    var_77 = 'users'
    var_78 = 'name'
    var_79 = 'scores'
    var_80 = 'Alice'
    var_81 = 85
    var_82 = 92
    var_83 = 78
    var_84 = [var_81, var_82, var_83]
    var_85 = {var_78: var_80, var_79: var_84}
    var_86 = 'Bob'
    var_87 = 88
    var_88 = 95
    var_89 = 82
    var_90 = [var_87, var_88, var_89]
    var_91 = {var_78: var_86, var_79: var_90}
    var_92 = [var_85, var_91]
    var_93 = {var_77: var_92}
    var_94 = [var_77, var_22, var_78]
    var_95 = module_0.get_in(var_94, var_93)
    assert var_95 == 'Alice'
    var_96 = [var_77, var_13, var_79, var_14]
    var_97 = module_0.get_in(var_96, var_93)
    assert var_97 == 82
    var_98 = [var_77, var_22, var_79, var_18]
    var_99 = module_0.get_in(var_98, var_93)
    assert var_99 is None
    var_100 = [var_77, var_14, var_78]
    var_101 = module_0.get_in(var_100, var_93)
    assert var_101 is None



# Parsed testcases at query #27
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 42
    var_9 = [var_0, var_1]
    var_10 = module_0.get_in(var_9, var_6)
    var_11 = [var_0]
    var_12 = module_0.get_in(var_11, var_6)
    var_13 = 1
    var_14 = 2
    var_15 = 3
    var_16 = [var_13, var_14, var_15]
    var_17 = 4
    var_18 = 5
    var_19 = 6
    var_20 = [var_17, var_18, var_19]
    var_21 = [var_16, var_20]
    var_22 = 0
    var_23 = [var_22, var_13]
    var_24 = module_0.get_in(var_23, var_21)
    assert var_24 == 2
    var_25 = [var_13, var_14]
    var_26 = module_0.get_in(var_25, var_21)
    assert var_26 == 6
    var_27 = {var_1: var_13}
    var_28 = {var_2: var_14}
    var_29 = [var_27, var_28]
    var_30 = {var_0: var_29}
    var_31 = [var_0, var_22, var_1]
    var_32 = module_0.get_in(var_31, var_30)
    assert var_32 == 1
    var_33 = [var_0, var_13, var_2]
    var_34 = module_0.get_in(var_33, var_30)
    assert var_34 == 2
    var_35 = {var_0: var_13}
    var_36 = [var_1]
    var_37 = module_0.get_in(var_36, var_35)
    assert var_37 is None
    var_38 = [var_1]
    var_39 = 'not found'
    var_40 = module_0.get_in(var_38, var_35, var_39)
    assert var_40 == 'not found'
    var_41 = [var_0, var_1]
    var_42 = module_0.get_in(var_41, var_35, var_22)
    assert var_42 == 0
    var_43 = {var_0: var_13}
    var_44 = 'b'
    var_45 = [var_44]
    var_46 = True
    var_47 = module_0.get_in(var_45, var_43, no_default=var_46)
    var_48 = [var_13, var_14, var_15]
    var_49 = 5
    var_50 = [var_49]
    var_51 = True
    var_52 = module_0.get_in(var_50, var_48, no_default=var_51)
    var_53 = {var_49: var_13}
    var_54 = []
    var_55 = module_0.get_in(var_54, var_53)
    var_56 = []
    var_57 = 'default'
    var_58 = module_0.get_in(var_56, var_53, var_57)
    var_59 = None
    var_60 = {var_49: var_59}
    var_61 = [var_49]
    var_62 = module_0.get_in(var_61, var_60)
    assert var_62 is None
    var_63 = [var_49, var_50]
    var_64 = module_0.get_in(var_63, var_60, var_57)
    assert var_64 == 'default'
    var_65 = 'key'
    var_66 = [var_65]
    var_67 = {}
    var_68 = module_0.get_in(var_66, var_67, var_57)
    assert var_68 == 'default'
    var_69 = [var_22]
    var_70 = []
    var_71 = module_0.get_in(var_69, var_70, var_57)
    assert var_71 == 'default'
    var_72 = {}
    var_73 = {var_49: var_72}
    var_74 = [var_49, var_50]
    var_75 = 'missing'
    var_76 = module_0.get_in(var_74, var_73, var_75)
    assert var_76 == 'missing'
    var_77 = 'deep'
    var_78 = {var_15: var_77}
    var_79 = {var_14: var_78}
    var_80 = {var_13: var_79}
    var_81 = [var_13, var_14, var_15]
    var_82 = module_0.get_in(var_81, var_80)
    assert var_82 == 'deep'
    var_83 = [var_13, var_14, var_15]
    var_84 = '0'
    var_85 = [var_84]
    var_86 = module_0.get_in(var_85, var_83, var_39)
    assert var_86 == 'not found'



# Parsed testcases at query #28
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 42
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = {var_1: var_11}
    var_13 = [var_9, var_10, var_12]
    var_14 = {var_0: var_13}
    var_15 = [var_0, var_10, var_1]
    var_16 = module_0.get_in(var_15, var_14)
    assert var_16 == 3
    var_17 = {var_0: var_9}
    var_18 = [var_1]
    var_19 = module_0.get_in(var_18, var_17)
    assert var_19 is None
    var_20 = [var_1]
    var_21 = 'default'
    var_22 = module_0.get_in(var_20, var_17, var_21)
    assert var_22 == 'default'
    var_23 = {var_0: var_9}
    var_24 = 'b'
    var_25 = [var_24]
    var_26 = True
    var_27 = module_0.get_in(var_25, var_23, no_default=var_26)
    var_28 = [var_9, var_10, var_11]
    var_29 = 5
    var_30 = [var_29]
    var_31 = True
    var_32 = module_0.get_in(var_30, var_28, no_default=var_31)
    var_33 = {var_30: var_9}
    var_34 = {var_29: var_33}
    var_35 = 'd'
    var_36 = [var_29, var_31, var_35]
    var_37 = 'missing'
    var_38 = module_0.get_in(var_36, var_34, var_37)
    assert var_38 == 'missing'
    var_39 = {var_29: var_9}
    var_40 = []
    var_41 = module_0.get_in(var_40, var_39)
    var_42 = {var_30: var_9}
    var_43 = {var_31: var_10}
    var_44 = [var_42, var_43]
    var_45 = {var_29: var_44}
    var_46 = 0
    var_47 = [var_29, var_46, var_30]
    var_48 = module_0.get_in(var_47, var_45)
    assert var_48 == 1
    var_49 = [var_29, var_9, var_31]
    var_50 = module_0.get_in(var_49, var_45)
    assert var_50 == 2
    var_51 = None
    var_52 = {var_29: var_51}
    var_53 = [var_29]
    var_54 = module_0.get_in(var_53, var_52)
    assert var_54 is None
    var_55 = {var_29: var_9}
    var_56 = [var_29, var_30]
    var_57 = module_0.get_in(var_56, var_55)
    assert var_57 is None
    var_58 = {var_30: var_9}
    var_59 = {var_29: var_58}
    var_60 = [var_29, var_31]
    var_61 = module_0.get_in(var_60, var_59, var_46)
    assert var_61 == 0



# Parsed testcases at query #29
#--------------------------


import pyrsistent._toolz as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.get_in(var_7, var_6)
    assert var_8 == 42
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = {var_1: var_11}
    var_13 = [var_9, var_10, var_12]
    var_14 = {var_0: var_13}
    var_15 = [var_0, var_10, var_1]
    var_16 = module_0.get_in(var_15, var_14)
    assert var_16 == 3
    var_17 = {var_0: var_9}
    var_18 = [var_1]
    var_19 = module_0.get_in(var_18, var_17)
    assert var_19 is None
    var_20 = [var_1]
    var_21 = 'default'
    var_22 = module_0.get_in(var_20, var_17, var_21)
    assert var_22 == 'default'
    var_23 = {var_0: var_9}
    var_24 = 'b'
    var_25 = [var_24]
    var_26 = True
    var_27 = module_0.get_in(var_25, var_23, no_default=var_26)
    var_28 = [var_9, var_10, var_11]
    var_29 = 5
    var_30 = [var_29]
    var_31 = True
    var_32 = module_0.get_in(var_30, var_28, no_default=var_31)
    var_33 = {var_30: var_9}
    var_34 = {var_29: var_33}
    var_35 = [var_29, var_31]
    var_36 = 'missing'
    var_37 = module_0.get_in(var_35, var_34, var_36)
    assert var_37 == 'missing'
    var_38 = {var_29: var_9}
    var_39 = []
    var_40 = module_0.get_in(var_39, var_38)
    var_41 = {var_30: var_9}
    var_42 = {var_31: var_10}
    var_43 = [var_41, var_42]
    var_44 = {var_29: var_43}
    var_45 = 0
    var_46 = [var_29, var_45, var_30]
    var_47 = module_0.get_in(var_46, var_44)
    assert var_47 == 1
    var_48 = [var_29, var_9, var_31]
    var_49 = module_0.get_in(var_48, var_44)
    assert var_49 == 2
    var_50 = None
    var_51 = {var_29: var_50}
    var_52 = [var_29, var_30]
    var_53 = module_0.get_in(var_52, var_51)
    assert var_53 is None
    var_54 = [var_29]
    var_55 = {}
    var_56 = module_0.get_in(var_54, var_55)
    assert var_56 is None
    var_57 = [var_45]
    var_58 = []
    var_59 = module_0.get_in(var_57, var_58)
    assert var_59 is None



