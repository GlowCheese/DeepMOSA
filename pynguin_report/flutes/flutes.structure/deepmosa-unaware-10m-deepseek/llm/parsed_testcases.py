####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x * y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1)
    var_23 = (var_2, var_4)
    var_24 = (var_22, var_23)
    var_25 = (var_5, var_6)
    var_26 = (var_15, var_16)
    var_27 = (var_25, var_26)
    var_28 = [var_24, var_27]
    var_29 = lambda x, y: x + y
    var_30 = module_0.map_structure_zip(var_29, var_28)
    var_31 = 'a'
    var_32 = 'b'
    var_33 = {var_31: var_0, var_32: var_1}
    var_34 = {var_31: var_2, var_32: var_4}
    var_35 = [var_33, var_34]
    var_36 = lambda x, y: x - y
    var_37 = module_0.map_structure_zip(var_36, var_35)
    var_38 = [var_0, var_1]
    var_39 = (var_2, var_4)
    var_40 = {var_31: var_38, var_32: var_39}
    var_41 = [var_5, var_6]
    var_42 = (var_15, var_16)
    var_43 = {var_31: var_41, var_32: var_42}
    var_44 = [var_40, var_43]
    var_45 = lambda x, y: x + y
    var_46 = module_0.map_structure_zip(var_45, var_44)
    var_47 = [var_0, var_1, var_2]
    var_48 = [var_4, var_5, var_6]
    var_49 = 9
    var_50 = [var_15, var_16, var_49]
    var_51 = [var_47, var_48, var_50]
    var_52 = lambda x, y, z: x + y + z
    var_53 = module_0.map_structure_zip(var_52, var_51)
    var_54 = [var_0, var_1, var_2]
    var_55 = module_0.no_map_instance(var_54)
    var_56 = [var_55, var_55]
    var_57 = lambda x, y: x + y
    var_58 = module_0.map_structure_zip(var_57, var_56)
    var_59 = [var_0, var_1]
    var_60 = lambda x, y: x + y
    var_61 = module_0.map_structure_zip(var_60, var_56)
    var_62 = 'Point'
    var_63 = 'x'
    var_64 = 'y'
    var_65 = [var_63, var_64]
    var_66 = lambda x, y: x + y
    var_67 = module_0.map_structure_zip(var_66, var_56)
    var_68 = {var_0, var_1}
    var_69 = {var_2, var_4}
    var_70 = [var_68, var_69]
    var_71 = lambda x, y: x + y
    var_72 = module_0.map_structure_zip(var_71, var_70)
    var_73 = []
    var_74 = []
    var_75 = [var_73, var_74]
    var_76 = lambda x, y: x + y
    var_77 = module_0.map_structure_zip(var_76, var_75)
    var_78 = [var_71, var_72, var_2]
    var_79 = [var_78]
    var_80 = lambda x: x * var_72
    var_81 = module_0.map_structure_zip(var_80, var_79)



# Parsed testcases at query #2
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = lambda x, y: x * y
    var_12 = [var_1, var_2]
    var_13 = [var_3, var_5]
    var_14 = [var_12, var_13]
    var_15 = [var_6, var_7]
    var_16 = 7
    var_17 = 8
    var_18 = [var_16, var_17]
    var_19 = [var_15, var_18]
    var_20 = [var_14, var_19]
    var_21 = module_0.map_structure_zip(var_11, var_20)
    var_22 = lambda x, y, z: x + y + z
    var_23 = (var_1, var_2)
    var_24 = (var_3, var_5)
    var_25 = (var_6, var_7)
    var_26 = [var_23, var_24, var_25]
    var_27 = module_0.map_structure_zip(var_22, var_26)
    var_28 = lambda x, y: f'{x}{y}'
    var_29 = 'a'
    var_30 = 'b'
    var_31 = {var_29: var_1, var_30: var_2}
    var_32 = {var_29: var_3, var_30: var_5}
    var_33 = [var_31, var_32]
    var_34 = module_0.map_structure_zip(var_28, var_33)
    var_35 = lambda x, y: x + y
    var_36 = [var_1, var_2]
    var_37 = (var_3, var_5)
    var_38 = {var_29: var_36, var_30: var_37}
    var_39 = [var_6, var_7]
    var_40 = (var_16, var_17)
    var_41 = {var_29: var_39, var_30: var_40}
    var_42 = [var_38, var_41]
    var_43 = module_0.map_structure_zip(var_35, var_42)
    var_44 = lambda x: x * var_2
    var_45 = [var_1, var_2, var_3]
    var_46 = [var_45]
    var_47 = module_0.map_structure_zip(var_44, var_46)
    var_48 = 'Point'
    var_49 = 'x'
    var_50 = 'y'
    var_51 = [var_49, var_50]
    var_52 = lambda x, y: Point(x.x + y.x, x.y + y.y)
    var_53 = [var_1, var_2, var_3]
    var_54 = [var_5, var_6, var_7]
    var_55 = lambda x, y: str(x) + str(y)
    var_56 = [var_1, var_2, var_3]
    var_57 = [var_1, var_2, var_3]
    var_58 = module_0.no_map_instance(var_57)
    var_59 = lambda x, y: x + y
    var_60 = [var_56, var_58]
    var_61 = module_0.map_structure_zip(var_59, var_60)
    var_62 = lambda x, y: x + y
    var_63 = 1
    var_64 = 2
    var_65 = {var_63, var_64}
    var_66 = 3
    var_67 = 4
    var_68 = {var_66, var_67}
    var_69 = [var_65, var_68]
    var_70 = module_0.map_structure_zip(var_62, var_69)
    var_71 = lambda x, y: x + y
    var_72 = []
    var_73 = []
    var_74 = [var_72, var_73]
    var_75 = module_0.map_structure_zip(var_71, var_74)
    var_76 = lambda x, y: x + y
    var_77 = {}
    var_78 = {}
    var_79 = [var_77, var_78]
    var_80 = module_0.map_structure_zip(var_76, var_79)
    var_81 = lambda x, y, z: x + y + z
    var_82 = [var_63, var_64]
    var_83 = [var_65, var_67]
    var_84 = [var_68, var_69]
    var_85 = [var_82, var_83, var_84]
    var_86 = module_0.map_structure_zip(var_81, var_85)



# Parsed testcases at query #3
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = lambda x, y: x * y
    var_12 = [var_1, var_2]
    var_13 = [var_3, var_5]
    var_14 = [var_12, var_13]
    var_15 = [var_6, var_7]
    var_16 = 7
    var_17 = 8
    var_18 = [var_16, var_17]
    var_19 = [var_15, var_18]
    var_20 = [var_14, var_19]
    var_21 = module_0.map_structure_zip(var_11, var_20)
    var_22 = lambda x, y, z: x + y + z
    var_23 = (var_1, var_2)
    var_24 = (var_3, var_5)
    var_25 = (var_6, var_7)
    var_26 = [var_23, var_24, var_25]
    var_27 = module_0.map_structure_zip(var_22, var_26)
    var_28 = 'Point'
    var_29 = 'x'
    var_30 = 'y'
    var_31 = [var_29, var_30]
    var_32 = lambda a, b: a + b
    var_33 = lambda x, y: x - y
    var_34 = 'a'
    var_35 = 'b'
    var_36 = 10
    var_37 = 20
    var_38 = {var_34: var_36, var_35: var_37}
    var_39 = {var_34: var_3, var_35: var_6}
    var_40 = [var_38, var_39]
    var_41 = module_0.map_structure_zip(var_33, var_40)
    var_42 = 'list'
    var_43 = 'tuple'
    var_44 = 'nested'
    var_45 = [var_1, var_2]
    var_46 = (var_3, var_5)
    var_47 = {var_34: var_6}
    var_48 = {var_42: var_45, var_43: var_46, var_44: var_47}
    var_49 = [var_7, var_16]
    var_50 = 9
    var_51 = (var_17, var_50)
    var_52 = {var_34: var_36}
    var_53 = {var_42: var_49, var_43: var_51, var_44: var_52}
    var_54 = lambda x, y: x * y
    var_55 = [var_48, var_53]
    var_56 = module_0.map_structure_zip(var_54, var_55)
    var_57 = 14
    var_58 = [var_7, var_57]
    var_59 = 24
    var_60 = 36
    var_61 = (var_59, var_60)
    var_62 = 50
    var_63 = {var_34: var_62}
    var_64 = {var_42: var_58, var_43: var_61, var_44: var_63}
    var_65 = [var_1, var_2, var_3]
    var_66 = module_0.no_map_instance(var_65)
    var_67 = lambda x, y: str(x) + str(y)
    var_68 = [var_66, var_66]
    var_69 = module_0.map_structure_zip(var_67, var_68)
    assert var_69 == '[1, 2, 3][1, 2, 3]'
    var_70 = [var_1, var_2, var_3]
    var_71 = lambda x, y: x + y
    var_72 = lambda x: x * var_2
    var_73 = [var_1, var_2, var_3]
    var_74 = [var_73]
    var_75 = module_0.map_structure_zip(var_72, var_74)
    var_76 = lambda x, y, z: x + y + z
    var_77 = [var_1, var_2]
    var_78 = [var_3, var_5]
    var_79 = [var_6, var_7]
    var_80 = [var_77, var_78, var_79]
    var_81 = module_0.map_structure_zip(var_76, var_80)
    var_82 = lambda x, y: x + y
    var_83 = []
    var_84 = []
    var_85 = [var_83, var_84]
    var_86 = module_0.map_structure_zip(var_82, var_85)
    var_87 = lambda x, y: x + y
    var_88 = 1
    var_89 = 2
    var_90 = {var_88, var_89}
    var_91 = 3
    var_92 = 4
    var_93 = {var_91, var_92}
    var_94 = [var_90, var_93]
    var_95 = module_0.map_structure_zip(var_87, var_94)
    var_96 = 'c'
    var_97 = {var_35: var_90}
    var_98 = (var_89, var_97)
    var_99 = [var_88, var_98]
    var_100 = 'd'
    var_101 = [var_92, var_93]
    var_102 = {var_100: var_101}
    var_103 = {var_34: var_99, var_96: var_102}
    var_104 = {var_35: var_17}
    var_105 = (var_16, var_104)
    var_106 = [var_94, var_105]
    var_107 = [var_50, var_36]
    var_108 = {var_100: var_107}
    var_109 = {var_34: var_106, var_96: var_108}
    var_110 = lambda x, y: x - y
    var_111 = [var_103, var_109]
    var_112 = module_0.map_structure_zip(var_110, var_111)
    var_113 = -5
    var_114 = -5
    var_115 = -5
    var_116 = {var_35: var_115}
    var_117 = (var_114, var_116)
    var_118 = [var_113, var_117]
    var_119 = -5
    var_120 = -5
    var_121 = [var_119, var_120]
    var_122 = {var_100: var_121}
    var_123 = {var_34: var_118, var_96: var_122}



# Parsed testcases at query #4
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = 'a'
    var_12 = 'b'
    var_13 = {var_11: var_2, var_12: var_3}
    var_14 = (var_5, var_6)
    var_15 = [var_1, var_13, var_14]
    var_16 = 10
    var_17 = 20
    var_18 = 30
    var_19 = {var_11: var_17, var_12: var_18}
    var_20 = 40
    var_21 = 50
    var_22 = (var_20, var_21)
    var_23 = [var_16, var_19, var_22]
    var_24 = lambda x, y: x + y
    var_25 = [var_15, var_23]
    var_26 = module_0.map_structure_zip(var_24, var_25)
    var_27 = lambda x, y, z: x + y + z
    var_28 = [var_1, var_2]
    var_29 = [var_3, var_5]
    var_30 = [var_6, var_7]
    var_31 = [var_28, var_29, var_30]
    var_32 = module_0.map_structure_zip(var_27, var_31)
    var_33 = 'Point'
    var_34 = 'x'
    var_35 = 'y'
    var_36 = [var_34, var_35]
    var_37 = lambda x, y: x + y
    var_38 = [var_1, var_2, var_3]
    var_39 = module_0.no_map_instance(var_38)
    var_40 = lambda x, y: x + y
    var_41 = [var_39, var_39]
    var_42 = module_0.map_structure_zip(var_40, var_41)
    assert var_42 == '[1, 2][3, 4]'
    var_43 = [var_1, var_2]
    var_44 = [var_3, var_5]
    var_45 = lambda x, y: str(x) + str(y)
    var_46 = (var_11, var_1)
    var_47 = (var_12, var_2)
    var_48 = [var_46, var_47]
    var_49 = (var_11, var_3)
    var_50 = (var_12, var_5)
    var_51 = [var_49, var_50]
    var_52 = lambda x, y: x * y
    var_53 = lambda x, y: x + y
    var_54 = 1
    var_55 = 2
    var_56 = {var_54, var_55}
    var_57 = 3
    var_58 = 4
    var_59 = {var_57, var_58}
    var_60 = [var_56, var_59]
    var_61 = module_0.map_structure_zip(var_53, var_60)
    var_62 = [var_54, var_55]
    var_63 = 'c'
    var_64 = 'd'
    var_65 = {var_63: var_56, var_64: var_58}
    var_66 = {var_11: var_62, var_12: var_65}
    var_67 = [var_59, var_60]
    var_68 = 7
    var_69 = 8
    var_70 = {var_63: var_68, var_64: var_69}
    var_71 = {var_11: var_67, var_12: var_70}
    var_72 = lambda x, y: x - y
    var_73 = [var_66, var_71]
    var_74 = module_0.map_structure_zip(var_72, var_73)
    var_75 = lambda x, y: x + y
    var_76 = []
    var_77 = []
    var_78 = [var_76, var_77]
    var_79 = module_0.map_structure_zip(var_75, var_78)
    var_80 = lambda x: x * var_55
    var_81 = [var_54, var_55, var_56]
    var_82 = [var_81]
    var_83 = module_0.map_structure_zip(var_80, var_82)



# Parsed testcases at query #5
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = lambda x: x + var_2
    var_7 = [var_2, var_0]
    var_8 = 4
    var_9 = [var_3, var_8]
    var_10 = [var_7, var_9]
    var_11 = module_0.map_structure(var_6, var_10)
    var_12 = lambda x: x.upper()
    var_13 = 'a'
    var_14 = 'b'
    var_15 = 'c'
    var_16 = (var_13, var_14, var_15)
    var_17 = module_0.map_structure(var_12, var_16)
    var_18 = lambda x: x * var_3
    var_19 = (var_2, var_0)
    var_20 = (var_3, var_8)
    var_21 = (var_19, var_20)
    var_22 = module_0.map_structure(var_18, var_21)
    var_23 = lambda x: x * var_0
    var_24 = {var_13: var_2, var_14: var_0}
    var_25 = module_0.map_structure(var_23, var_24)
    var_26 = 10
    var_27 = lambda x: x + var_26
    var_28 = 'x'
    var_29 = {var_28: var_2}
    var_30 = 'y'
    var_31 = {var_30: var_0}
    var_32 = {var_13: var_29, var_14: var_31}
    var_33 = module_0.map_structure(var_27, var_32)
    var_34 = lambda x: x ** var_0
    var_35 = {var_2, var_0, var_3}
    var_36 = module_0.map_structure(var_34, var_35)
    var_37 = {var_28: var_3}
    var_38 = [var_2, var_0, var_37]
    var_39 = 5
    var_40 = (var_8, var_39)
    var_41 = {var_13: var_38, var_14: var_40}
    var_42 = lambda x: x * var_0
    var_43 = module_0.map_structure(var_42, var_41)
    var_44 = lambda x: x + var_2
    var_45 = module_0.map_structure(var_44, var_39)
    assert var_45 == 6
    var_46 = [var_2, var_0, var_3]
    var_47 = module_0.no_map_instance(var_46)
    var_48 = lambda x: x * var_0
    var_49 = module_0.map_structure(var_48, var_47)
    var_50 = [var_2, var_0, var_3]
    var_51 = lambda x: x * var_0
    var_52 = [var_2, var_0, var_3]
    var_53 = 'Point'
    var_54 = [var_28, var_30]
    var_55 = lambda x: x + var_26
    var_56 = 11
    var_57 = 12
    var_58 = lambda x: x
    var_59 = []
    var_60 = module_0.map_structure(var_58, var_59)
    var_61 = lambda x: x
    var_62 = {}
    var_63 = module_0.map_structure(var_61, var_62)
    var_64 = lambda x: x
    var_65 = ()
    var_66 = module_0.map_structure(var_64, var_65)
    var_67 = [var_2, var_0, var_3]
    var_68 = 'list'
    var_69 = 'tuple'
    var_70 = 'set'
    var_71 = 'nested'
    var_72 = {var_13: var_3}
    var_73 = [var_2, var_0, var_72]
    var_74 = 6
    var_75 = [var_39, var_74]
    var_76 = (var_8, var_75)
    var_77 = 7
    var_78 = 8
    var_79 = {var_77, var_78}
    var_80 = 9
    var_81 = [var_80, var_26]
    var_82 = {var_30: var_81}
    var_83 = {var_28: var_82}
    var_84 = {var_68: var_73, var_69: var_76, var_70: var_79, var_71: var_83}
    var_85 = {var_13: var_74}
    var_86 = [var_0, var_8, var_85]
    var_87 = [var_26, var_57]
    var_88 = (var_78, var_87)
    var_89 = 14
    var_90 = 16
    var_91 = {var_89, var_90}
    var_92 = 18
    var_93 = 20
    var_94 = [var_92, var_93]
    var_95 = {var_30: var_94}
    var_96 = {var_28: var_95}
    var_97 = {var_68: var_86, var_69: var_88, var_70: var_91, var_71: var_96}
    var_98 = lambda x: x * var_0
    var_99 = module_0.map_structure(var_98, var_84)



# Parsed testcases at query #6
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = lambda x, y: x * y
    var_12 = [var_1, var_2]
    var_13 = [var_3, var_5]
    var_14 = [var_12, var_13]
    var_15 = [var_6, var_7]
    var_16 = 7
    var_17 = 8
    var_18 = [var_16, var_17]
    var_19 = [var_15, var_18]
    var_20 = [var_14, var_19]
    var_21 = module_0.map_structure_zip(var_11, var_20)
    var_22 = lambda x, y, z: x + y + z
    var_23 = (var_1, var_2)
    var_24 = (var_3, var_5)
    var_25 = (var_6, var_7)
    var_26 = [var_23, var_24, var_25]
    var_27 = module_0.map_structure_zip(var_22, var_26)
    var_28 = lambda x, y: f'{x}{y}'
    var_29 = 'a'
    var_30 = 'b'
    var_31 = {var_29: var_1, var_30: var_2}
    var_32 = {var_29: var_3, var_30: var_5}
    var_33 = [var_31, var_32]
    var_34 = module_0.map_structure_zip(var_28, var_33)
    var_35 = lambda x, y: x + y
    var_36 = [var_1, var_2]
    var_37 = (var_3, var_5)
    var_38 = {var_29: var_36, var_30: var_37}
    var_39 = [var_6, var_7]
    var_40 = (var_16, var_17)
    var_41 = {var_29: var_39, var_30: var_40}
    var_42 = [var_38, var_41]
    var_43 = module_0.map_structure_zip(var_35, var_42)
    var_44 = [var_1, var_2, var_3]
    var_45 = module_0.no_map_instance(var_44)
    var_46 = lambda x, y: x + y
    var_47 = [var_45, var_45]
    var_48 = module_0.map_structure_zip(var_46, var_47)
    var_49 = [var_1, var_2, var_3]
    var_50 = lambda x, y: x + y
    var_51 = lambda x: x * var_2
    var_52 = [var_1, var_2, var_3]
    var_53 = [var_52]
    var_54 = module_0.map_structure_zip(var_51, var_53)
    var_55 = lambda x, y, z: x + y + z
    var_56 = [var_1, var_2]
    var_57 = [var_3, var_5]
    var_58 = [var_6, var_7]
    var_59 = [var_56, var_57, var_58]
    var_60 = module_0.map_structure_zip(var_55, var_59)
    var_61 = lambda x, y: x + y
    var_62 = []
    var_63 = []
    var_64 = [var_62, var_63]
    var_65 = module_0.map_structure_zip(var_61, var_64)
    var_66 = lambda x, y: x + y
    var_67 = 1
    var_68 = 2
    var_69 = {var_67, var_68}
    var_70 = 3
    var_71 = 4
    var_72 = {var_70, var_71}
    var_73 = [var_69, var_72]
    var_74 = module_0.map_structure_zip(var_66, var_73)
    var_75 = 'Point'
    var_76 = 'x'
    var_77 = 'y'
    var_78 = [var_76, var_77]
    var_79 = lambda p1, p2: Point(p1.x + p2.x, p1.y + p2.y)



# Parsed testcases at query #7
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = lambda x: x + var_2
    var_7 = [var_2, var_0]
    var_8 = 4
    var_9 = [var_3, var_8]
    var_10 = [var_7, var_9]
    var_11 = module_0.map_structure(var_6, var_10)
    var_12 = lambda x: x.upper()
    var_13 = 'a'
    var_14 = 'b'
    var_15 = 'c'
    var_16 = (var_13, var_14, var_15)
    var_17 = module_0.map_structure(var_12, var_16)
    var_18 = lambda x: x * var_3
    var_19 = (var_2, var_0)
    var_20 = (var_3, var_8)
    var_21 = (var_19, var_20)
    var_22 = module_0.map_structure(var_18, var_21)
    var_23 = lambda x: x * var_0
    var_24 = {var_13: var_2, var_14: var_0}
    var_25 = module_0.map_structure(var_23, var_24)
    var_26 = 10
    var_27 = lambda x: x + var_26
    var_28 = 'x'
    var_29 = {var_28: var_2}
    var_30 = 'y'
    var_31 = {var_30: var_0}
    var_32 = {var_13: var_29, var_14: var_31}
    var_33 = module_0.map_structure(var_27, var_32)
    var_34 = lambda x: x * var_0
    var_35 = {var_2, var_0, var_3}
    var_36 = module_0.map_structure(var_34, var_35)
    var_37 = {var_28: var_3}
    var_38 = [var_2, var_0, var_37]
    var_39 = 5
    var_40 = (var_8, var_39)
    var_41 = {var_13: var_38, var_14: var_40}
    var_42 = lambda x: x + var_2
    var_43 = module_0.map_structure(var_42, var_41)
    var_44 = lambda x: x * var_0
    var_45 = module_0.map_structure(var_44, var_39)
    assert var_45 == 10
    var_46 = [var_2, var_0, var_3]
    var_47 = lambda x: x * var_0
    var_48 = [var_2, var_0, var_3]
    var_49 = module_0.no_map_instance(var_48)
    var_50 = lambda x: x * var_0
    var_51 = module_0.map_structure(var_50, var_49)
    var_52 = 'Point'
    var_53 = [var_28, var_30]
    var_54 = lambda x: x * var_0
    var_55 = lambda x: x * var_0
    var_56 = []
    var_57 = module_0.map_structure(var_55, var_56)
    var_58 = lambda x: x * var_0
    var_59 = {}
    var_60 = module_0.map_structure(var_58, var_59)
    var_61 = lambda x: x * var_0
    var_62 = set()
    var_63 = module_0.map_structure(var_61, var_62)
    var_64 = set()
    var_65 = [var_2, var_0, var_3]
    var_66 = [var_2, var_0]
    var_67 = [var_3, var_8]
    var_68 = [var_66, var_67]
    var_69 = 6
    var_70 = [var_39, var_69]
    var_71 = 7
    var_72 = 8
    var_73 = [var_71, var_72]
    var_74 = [var_70, var_73]
    var_75 = [var_68, var_74]
    var_76 = lambda x: x - var_2
    var_77 = module_0.map_structure(var_76, var_75)



# Parsed testcases at query #8
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = lambda x: x + var_2
    var_7 = [var_2, var_0]
    var_8 = 4
    var_9 = [var_3, var_8]
    var_10 = [var_7, var_9]
    var_11 = module_0.map_structure(var_6, var_10)
    var_12 = lambda x: x.upper()
    var_13 = 'a'
    var_14 = 'b'
    var_15 = 'c'
    var_16 = (var_13, var_14, var_15)
    var_17 = module_0.map_structure(var_12, var_16)
    var_18 = lambda x: x * var_3
    var_19 = (var_2, var_0)
    var_20 = (var_3, var_8)
    var_21 = (var_19, var_20)
    var_22 = module_0.map_structure(var_18, var_21)
    var_23 = 10
    var_24 = lambda x: x + var_23
    var_25 = {var_13: var_2, var_14: var_0}
    var_26 = module_0.map_structure(var_24, var_25)
    var_27 = lambda x: len(x)
    var_28 = 'hi'
    var_29 = 'hello'
    var_30 = {var_13: var_28, var_14: var_29}
    var_31 = module_0.map_structure(var_27, var_30)
    var_32 = lambda x: x ** var_0
    var_33 = {var_2, var_0, var_3}
    var_34 = module_0.map_structure(var_32, var_33)
    var_35 = [var_2, var_0, var_3]
    var_36 = 5
    var_37 = 6
    var_38 = (var_8, var_36, var_37)
    var_39 = {var_13: var_35, var_14: var_38}
    var_40 = lambda x: x * var_0
    var_41 = module_0.map_structure(var_40, var_39)
    var_42 = [var_2, var_0, var_3]
    var_43 = module_0.no_map_instance(var_42)
    var_44 = lambda x: x * var_0
    var_45 = module_0.map_structure(var_44, var_43)
    var_46 = [var_2, var_0, var_3]
    var_47 = lambda x: x * var_0
    var_48 = 'Point'
    var_49 = 'x'
    var_50 = 'y'
    var_51 = [var_49, var_50]
    var_52 = lambda x: x + var_23
    var_53 = 11
    var_54 = 12
    var_55 = lambda x: x + var_36
    var_56 = module_0.map_structure(var_55, var_23)
    assert var_56 == 15
    var_57 = lambda x: x
    var_58 = []
    var_59 = module_0.map_structure(var_57, var_58)
    var_60 = lambda x: x
    var_61 = {}
    var_62 = module_0.map_structure(var_60, var_61)
    var_63 = [var_2, var_0, var_3]
    var_64 = (var_2, var_0)
    var_65 = {var_14: var_64}
    var_66 = [var_3, var_8]
    var_67 = {var_15: var_66}
    var_68 = [var_65, var_67]
    var_69 = {var_13: var_68}
    var_70 = lambda x: x * var_0
    var_71 = module_0.map_structure(var_70, var_69)
    var_72 = (var_0, var_8)
    var_73 = {var_14: var_72}
    var_74 = 8
    var_75 = [var_37, var_74]
    var_76 = {var_15: var_75}
    var_77 = [var_73, var_76]
    var_78 = {var_13: var_77}



# Parsed testcases at query #9
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = lambda x: x + var_2
    var_7 = [var_2, var_0]
    var_8 = 4
    var_9 = [var_3, var_8]
    var_10 = [var_7, var_9]
    var_11 = module_0.map_structure(var_6, var_10)
    var_12 = lambda x: x.upper()
    var_13 = 'a'
    var_14 = 'b'
    var_15 = 'c'
    var_16 = (var_13, var_14, var_15)
    var_17 = module_0.map_structure(var_12, var_16)
    var_18 = lambda x: x * var_3
    var_19 = (var_2, var_0)
    var_20 = (var_3, var_8)
    var_21 = (var_19, var_20)
    var_22 = module_0.map_structure(var_18, var_21)
    var_23 = lambda x: x * var_0
    var_24 = {var_13: var_2, var_14: var_0}
    var_25 = module_0.map_structure(var_23, var_24)
    var_26 = 10
    var_27 = lambda x: x + var_26
    var_28 = 'x'
    var_29 = {var_28: var_2}
    var_30 = 'y'
    var_31 = {var_30: var_0}
    var_32 = {var_13: var_29, var_14: var_31}
    var_33 = module_0.map_structure(var_27, var_32)
    var_34 = lambda x: x ** var_0
    var_35 = {var_2, var_0, var_3}
    var_36 = module_0.map_structure(var_34, var_35)
    var_37 = [var_2, var_0, var_3]
    var_38 = 5
    var_39 = 6
    var_40 = (var_8, var_38, var_39)
    var_41 = 7
    var_42 = 8
    var_43 = {var_41, var_42}
    var_44 = {var_13: var_37, var_14: var_40, var_15: var_43}
    var_45 = lambda x: x * var_0
    var_46 = module_0.map_structure(var_45, var_44)
    var_47 = lambda x: x + var_2
    var_48 = module_0.map_structure(var_47, var_38)
    assert var_48 == 6
    var_49 = [var_2, var_0, var_3]
    var_50 = module_0.no_map_instance(var_49)
    var_51 = lambda x: x * var_0
    var_52 = module_0.map_structure(var_51, var_50)
    var_53 = [var_2, var_0, var_3]
    var_54 = lambda x: x * var_0
    var_55 = 'Point'
    var_56 = [var_28, var_30]
    var_57 = lambda x: x * var_0
    var_58 = lambda x: x
    var_59 = []
    var_60 = module_0.map_structure(var_58, var_59)
    var_61 = lambda x: x
    var_62 = {}
    var_63 = module_0.map_structure(var_61, var_62)
    var_64 = [var_2, var_0, var_3]



# Parsed testcases at query #10
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = lambda x, y: x * y
    var_12 = [var_1, var_2]
    var_13 = [var_3, var_5]
    var_14 = [var_12, var_13]
    var_15 = [var_6, var_7]
    var_16 = 7
    var_17 = 8
    var_18 = [var_16, var_17]
    var_19 = [var_15, var_18]
    var_20 = [var_14, var_19]
    var_21 = module_0.map_structure_zip(var_11, var_20)
    var_22 = lambda x, y, z: x + y + z
    var_23 = (var_1, var_2)
    var_24 = (var_3, var_5)
    var_25 = (var_6, var_7)
    var_26 = [var_23, var_24, var_25]
    var_27 = module_0.map_structure_zip(var_22, var_26)
    var_28 = lambda x, y: x - y
    var_29 = 'a'
    var_30 = 'b'
    var_31 = 10
    var_32 = 20
    var_33 = {var_29: var_31, var_30: var_32}
    var_34 = {var_29: var_3, var_30: var_6}
    var_35 = [var_33, var_34]
    var_36 = module_0.map_structure_zip(var_28, var_35)
    var_37 = lambda x, y: x + y
    var_38 = [var_1, var_2]
    var_39 = (var_3, var_5)
    var_40 = {var_29: var_38, var_30: var_39}
    var_41 = [var_6, var_7]
    var_42 = (var_16, var_17)
    var_43 = {var_29: var_41, var_30: var_42}
    var_44 = [var_40, var_43]
    var_45 = module_0.map_structure_zip(var_37, var_44)
    var_46 = lambda x: x * var_2
    var_47 = [var_1, var_2, var_3]
    var_48 = [var_47]
    var_49 = module_0.map_structure_zip(var_46, var_48)
    var_50 = lambda x, y, z: f'{x}{y}{z}'
    var_51 = [var_29, var_30]
    var_52 = 'c'
    var_53 = 'd'
    var_54 = [var_52, var_53]
    var_55 = 'e'
    var_56 = 'f'
    var_57 = [var_55, var_56]
    var_58 = [var_51, var_54, var_57]
    var_59 = module_0.map_structure_zip(var_50, var_58)
    var_60 = [var_1, var_2, var_3]
    var_61 = module_0.no_map_instance(var_60)
    var_62 = lambda x, y: x + y
    var_63 = [var_61, var_61]
    var_64 = module_0.map_structure_zip(var_62, var_63)
    var_65 = [var_1, var_2, var_3]
    var_66 = lambda x, y: x + y
    var_67 = 'Point'
    var_68 = 'x'
    var_69 = 'y'
    var_70 = [var_68, var_69]
    var_71 = lambda a, b: Point(a.x + b.x, a.y + b.y)
    var_72 = lambda x, y: x + y
    var_73 = []
    var_74 = []
    var_75 = [var_73, var_74]
    var_76 = module_0.map_structure_zip(var_72, var_75)
    var_77 = lambda x, y: x + y
    var_78 = {}
    var_79 = {}
    var_80 = [var_78, var_79]
    var_81 = module_0.map_structure_zip(var_77, var_80)
    var_82 = lambda x, y: x + y
    var_83 = 1
    var_84 = 2
    var_85 = {var_83, var_84}
    var_86 = 3
    var_87 = 4
    var_88 = {var_86, var_87}
    var_89 = [var_85, var_88]
    var_90 = module_0.map_structure_zip(var_82, var_89)
    var_91 = (var_29, var_83)
    var_92 = (var_30, var_84)
    var_93 = [var_91, var_92]
    var_94 = (var_29, var_85)
    var_95 = (var_30, var_87)
    var_96 = [var_94, var_95]
    var_97 = lambda x, y: x + y



# Parsed testcases at query #11
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = lambda x: x + var_2
    var_7 = [var_2, var_0]
    var_8 = 4
    var_9 = [var_3, var_8]
    var_10 = [var_7, var_9]
    var_11 = module_0.map_structure(var_6, var_10)
    var_12 = lambda x: x.upper()
    var_13 = 'a'
    var_14 = 'b'
    var_15 = 'c'
    var_16 = (var_13, var_14, var_15)
    var_17 = module_0.map_structure(var_12, var_16)
    var_18 = lambda x: x * var_3
    var_19 = (var_2, var_0)
    var_20 = (var_3, var_8)
    var_21 = (var_19, var_20)
    var_22 = module_0.map_structure(var_18, var_21)
    var_23 = lambda x: x * var_0
    var_24 = {var_13: var_2, var_14: var_0}
    var_25 = module_0.map_structure(var_23, var_24)
    var_26 = 10
    var_27 = lambda x: x + var_26
    var_28 = 'x'
    var_29 = {var_28: var_2}
    var_30 = 'y'
    var_31 = {var_30: var_0}
    var_32 = {var_13: var_29, var_14: var_31}
    var_33 = module_0.map_structure(var_27, var_32)
    var_34 = lambda x: x ** var_0
    var_35 = {var_2, var_0, var_3}
    var_36 = module_0.map_structure(var_34, var_35)
    var_37 = (var_2, var_0)
    var_38 = [var_3, var_8]
    var_39 = {var_13: var_37, var_14: var_38}
    var_40 = 5
    var_41 = 6
    var_42 = {var_40, var_41}
    var_43 = {var_15: var_42}
    var_44 = [var_39, var_43]
    var_45 = lambda x: x + var_2
    var_46 = module_0.map_structure(var_45, var_44)
    var_47 = (var_0, var_3)
    var_48 = [var_8, var_40]
    var_49 = {var_13: var_47, var_14: var_48}
    var_50 = 7
    var_51 = {var_41, var_50}
    var_52 = {var_15: var_51}
    var_53 = [var_49, var_52]
    var_54 = [var_2, var_0, var_3]
    var_55 = lambda x: x * var_0
    var_56 = [var_2, var_0, var_3]
    var_57 = module_0.no_map_instance(var_56)
    var_58 = lambda x: x * var_0
    var_59 = module_0.map_structure(var_58, var_57)
    var_60 = 'Point'
    var_61 = [var_28, var_30]
    var_62 = lambda x: x * var_0
    var_63 = lambda x: x
    var_64 = []
    var_65 = module_0.map_structure(var_63, var_64)
    var_66 = lambda x: x
    var_67 = {}
    var_68 = module_0.map_structure(var_66, var_67)
    var_69 = lambda x: x
    var_70 = set()
    var_71 = module_0.map_structure(var_69, var_70)
    var_72 = set()
    var_73 = lambda x: x * var_0
    var_74 = module_0.map_structure(var_73, var_40)
    assert var_74 == 10
    var_75 = '!'
    var_76 = lambda x: x + var_75
    var_77 = 'hello'
    var_78 = module_0.map_structure(var_76, var_77)
    assert var_78 == 'hello!'
    var_79 = [var_2, var_0]
    var_80 = module_0.no_map_instance(var_79)
    var_81 = [var_3, var_8]
    var_82 = [var_80, var_81]
    var_83 = lambda x: str(x)
    var_84 = module_0.map_structure(var_83, var_82)



# Parsed testcases at query #12
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = lambda x, y: x * y
    var_12 = [var_1, var_2]
    var_13 = [var_3, var_5]
    var_14 = [var_12, var_13]
    var_15 = [var_6, var_7]
    var_16 = 7
    var_17 = 8
    var_18 = [var_16, var_17]
    var_19 = [var_15, var_18]
    var_20 = [var_14, var_19]
    var_21 = module_0.map_structure_zip(var_11, var_20)
    var_22 = lambda x, y, z: x + y + z
    var_23 = (var_1, var_2)
    var_24 = (var_3, var_5)
    var_25 = (var_6, var_7)
    var_26 = [var_23, var_24, var_25]
    var_27 = module_0.map_structure_zip(var_22, var_26)
    var_28 = lambda x, y: f'{x}{y}'
    var_29 = 'a'
    var_30 = 'b'
    var_31 = {var_29: var_1, var_30: var_2}
    var_32 = {var_29: var_3, var_30: var_5}
    var_33 = [var_31, var_32]
    var_34 = module_0.map_structure_zip(var_28, var_33)
    var_35 = lambda x, y: x + y
    var_36 = [var_1, var_2]
    var_37 = (var_3, var_5)
    var_38 = {var_29: var_36, var_30: var_37}
    var_39 = [var_6, var_7]
    var_40 = (var_16, var_17)
    var_41 = {var_29: var_39, var_30: var_40}
    var_42 = [var_38, var_41]
    var_43 = module_0.map_structure_zip(var_35, var_42)
    var_44 = [var_1, var_2, var_3]
    var_45 = lambda x, y: x + y
    var_46 = [var_1, var_2, var_3]
    var_47 = [var_5, var_6, var_7]
    var_48 = module_0.no_map_instance(var_47)
    var_49 = lambda x, y: x + y
    var_50 = [var_46, var_48]
    var_51 = module_0.map_structure_zip(var_49, var_50)
    var_52 = lambda x: x * var_2
    var_53 = [var_1, var_2, var_3]
    var_54 = [var_53]
    var_55 = module_0.map_structure_zip(var_52, var_54)
    var_56 = lambda x, y, z: x + y + z
    var_57 = [var_1, var_2]
    var_58 = [var_3, var_5]
    var_59 = [var_6, var_7]
    var_60 = [var_57, var_58, var_59]
    var_61 = module_0.map_structure_zip(var_56, var_60)
    var_62 = lambda x, y: x + y
    var_63 = []
    var_64 = []
    var_65 = [var_63, var_64]
    var_66 = module_0.map_structure_zip(var_62, var_65)
    var_67 = lambda x, y: x + y
    var_68 = 1
    var_69 = 2
    var_70 = {var_68, var_69}
    var_71 = 3
    var_72 = 4
    var_73 = {var_71, var_72}
    var_74 = [var_70, var_73]
    var_75 = module_0.map_structure_zip(var_67, var_74)
    var_76 = 'Point'
    var_77 = 'x'
    var_78 = 'y'
    var_79 = [var_77, var_78]
    var_80 = lambda a, b: Point(a.x + b.x, a.y + b.y)



# Parsed testcases at query #13
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = lambda x: x + var_2
    var_7 = [var_2, var_0]
    var_8 = 4
    var_9 = [var_3, var_8]
    var_10 = [var_7, var_9]
    var_11 = module_0.map_structure(var_6, var_10)
    var_12 = lambda x: x.upper()
    var_13 = 'a'
    var_14 = 'b'
    var_15 = 'c'
    var_16 = (var_13, var_14, var_15)
    var_17 = module_0.map_structure(var_12, var_16)
    var_18 = lambda x: x * var_3
    var_19 = (var_2, var_0)
    var_20 = (var_3, var_8)
    var_21 = (var_19, var_20)
    var_22 = module_0.map_structure(var_18, var_21)
    var_23 = lambda x: x * var_0
    var_24 = {var_13: var_2, var_14: var_0}
    var_25 = module_0.map_structure(var_23, var_24)
    var_26 = 10
    var_27 = lambda x: x + var_26
    var_28 = 'x'
    var_29 = 'y'
    var_30 = {var_28: var_2, var_29: var_0}
    var_31 = {var_13: var_30, var_14: var_3}
    var_32 = module_0.map_structure(var_27, var_31)
    var_33 = lambda x: x ** var_0
    var_34 = {var_2, var_0, var_3}
    var_35 = module_0.map_structure(var_33, var_34)
    var_36 = {var_28: var_3}
    var_37 = [var_2, var_0, var_36]
    var_38 = 5
    var_39 = (var_8, var_38)
    var_40 = {var_13: var_37, var_14: var_39}
    var_41 = lambda x: x * var_0
    var_42 = module_0.map_structure(var_41, var_40)
    var_43 = lambda x: x + var_2
    var_44 = module_0.map_structure(var_43, var_38)
    assert var_44 == 6
    var_45 = [var_2, var_0, var_3]
    var_46 = lambda x: x * var_0
    var_47 = [var_2, var_0, var_3]
    var_48 = module_0.no_map_instance(var_47)
    var_49 = lambda x: x * var_0
    var_50 = module_0.map_structure(var_49, var_48)
    var_51 = 'Point'
    var_52 = [var_28, var_29]
    var_53 = lambda x: x * var_0
    var_54 = lambda x: x * var_0
    var_55 = []
    var_56 = module_0.map_structure(var_54, var_55)
    var_57 = lambda x: x * var_0
    var_58 = {}
    var_59 = module_0.map_structure(var_57, var_58)
    var_60 = [var_2, var_0, var_3]
    var_61 = (var_2, var_0)
    var_62 = {var_14: var_61}
    var_63 = [var_3, var_8]
    var_64 = [var_62, var_63]
    var_65 = {var_13: var_64, var_15: var_38}
    var_66 = lambda x: x + var_26
    var_67 = module_0.map_structure(var_66, var_65)



# Parsed testcases at query #14
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = lambda x, y: x * y
    var_12 = [var_1, var_2]
    var_13 = [var_3, var_5]
    var_14 = [var_12, var_13]
    var_15 = [var_6, var_7]
    var_16 = 7
    var_17 = 8
    var_18 = [var_16, var_17]
    var_19 = [var_15, var_18]
    var_20 = [var_14, var_19]
    var_21 = module_0.map_structure_zip(var_11, var_20)
    var_22 = lambda x, y, z: x + y + z
    var_23 = (var_1, var_2)
    var_24 = (var_3, var_5)
    var_25 = (var_6, var_7)
    var_26 = [var_23, var_24, var_25]
    var_27 = module_0.map_structure_zip(var_22, var_26)
    var_28 = lambda x, y: f'{x}{y}'
    var_29 = 'a'
    var_30 = 'b'
    var_31 = {var_29: var_1, var_30: var_2}
    var_32 = {var_29: var_3, var_30: var_5}
    var_33 = [var_31, var_32]
    var_34 = module_0.map_structure_zip(var_28, var_33)
    var_35 = lambda x, y: x + y
    var_36 = [var_1, var_2]
    var_37 = (var_3, var_5)
    var_38 = {var_29: var_36, var_30: var_37}
    var_39 = [var_6, var_7]
    var_40 = (var_16, var_17)
    var_41 = {var_29: var_39, var_30: var_40}
    var_42 = [var_38, var_41]
    var_43 = module_0.map_structure_zip(var_35, var_42)
    var_44 = 'Point'
    var_45 = 'x'
    var_46 = 'y'
    var_47 = [var_45, var_46]
    var_48 = lambda p1, p2: Point(p1.x + p2.x, p1.y + p2.y)
    var_49 = [var_1, var_2, var_3]
    var_50 = module_0.no_map_instance(var_49)
    var_51 = lambda x, y: x + y
    var_52 = [var_50, var_50]
    var_53 = module_0.map_structure_zip(var_51, var_52)
    var_54 = [var_1, var_2, var_3]
    var_55 = lambda x, y: x + y
    var_56 = lambda x: x * var_2
    var_57 = [var_1, var_2, var_3]
    var_58 = [var_57]
    var_59 = module_0.map_structure_zip(var_56, var_58)
    var_60 = lambda x, y, z: x + y + z
    var_61 = [var_1, var_2]
    var_62 = [var_3, var_5]
    var_63 = [var_6, var_7]
    var_64 = [var_61, var_62, var_63]
    var_65 = module_0.map_structure_zip(var_60, var_64)
    var_66 = lambda x, y: x + y
    var_67 = []
    var_68 = []
    var_69 = [var_67, var_68]
    var_70 = module_0.map_structure_zip(var_66, var_69)
    var_71 = lambda x, y: x + y
    var_72 = 1
    var_73 = 2
    var_74 = {var_72, var_73}
    var_75 = 3
    var_76 = 4
    var_77 = {var_75, var_76}
    var_78 = [var_74, var_77]
    var_79 = module_0.map_structure_zip(var_71, var_78)



# Parsed testcases at query #15
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = lambda x, y: x * y
    var_12 = [var_1, var_2]
    var_13 = [var_3, var_5]
    var_14 = [var_12, var_13]
    var_15 = [var_6, var_7]
    var_16 = 7
    var_17 = 8
    var_18 = [var_16, var_17]
    var_19 = [var_15, var_18]
    var_20 = [var_14, var_19]
    var_21 = module_0.map_structure_zip(var_11, var_20)
    var_22 = lambda x, y, z: x + y + z
    var_23 = (var_1, var_2)
    var_24 = (var_3, var_5)
    var_25 = (var_6, var_7)
    var_26 = [var_23, var_24, var_25]
    var_27 = module_0.map_structure_zip(var_22, var_26)
    var_28 = 'Point'
    var_29 = 'x'
    var_30 = 'y'
    var_31 = [var_29, var_30]
    var_32 = lambda a, b: Point(a.x + b.x, a.y + b.y)
    var_33 = lambda x, y: x + y
    var_34 = 'a'
    var_35 = 'b'
    var_36 = {var_34: var_1, var_35: var_2}
    var_37 = {var_34: var_3, var_35: var_5}
    var_38 = [var_36, var_37]
    var_39 = module_0.map_structure_zip(var_33, var_38)
    var_40 = lambda x, y: x + y
    var_41 = {var_34: var_1}
    var_42 = {var_35: var_2}
    var_43 = [var_41, var_42]
    var_44 = {var_34: var_3}
    var_45 = {var_35: var_5}
    var_46 = [var_44, var_45]
    var_47 = [var_43, var_46]
    var_48 = module_0.map_structure_zip(var_40, var_47)
    var_49 = [var_1, var_2, var_3]
    var_50 = module_0.no_map_instance(var_49)
    var_51 = lambda x, y: x + y
    var_52 = [var_50, var_50]
    var_53 = module_0.map_structure_zip(var_51, var_52)
    var_54 = [var_1, var_2, var_3]
    var_55 = lambda x, y: x + y
    var_56 = lambda x: x * var_2
    var_57 = [var_1, var_2, var_3]
    var_58 = [var_57]
    var_59 = module_0.map_structure_zip(var_56, var_58)
    var_60 = lambda x, y, z: x + y + z
    var_61 = [var_1, var_2]
    var_62 = [var_3, var_5]
    var_63 = [var_6, var_7]
    var_64 = [var_61, var_62, var_63]
    var_65 = module_0.map_structure_zip(var_60, var_64)
    var_66 = lambda x, y: x + y
    var_67 = []
    var_68 = []
    var_69 = [var_67, var_68]
    var_70 = module_0.map_structure_zip(var_66, var_69)
    var_71 = lambda x, y: x + y
    var_72 = 1
    var_73 = 2
    var_74 = {var_72, var_73}
    var_75 = 3
    var_76 = 4
    var_77 = {var_75, var_76}
    var_78 = [var_74, var_77]
    var_79 = module_0.map_structure_zip(var_71, var_78)
    var_80 = [var_72, var_73]
    var_81 = [var_74, var_76]
    var_82 = [var_80, var_81]
    var_83 = [var_77, var_78]
    var_84 = [var_16, var_17]
    var_85 = [var_83, var_84]
    var_86 = [var_82, var_85]
    var_87 = lambda x, y: x - y
    var_88 = [var_86, var_86]
    var_89 = module_0.map_structure_zip(var_87, var_88)
    var_90 = 0
    var_91 = [var_90, var_90]
    var_92 = [var_90, var_90]
    var_93 = [var_91, var_92]
    var_94 = [var_90, var_90]
    var_95 = [var_90, var_90]
    var_96 = [var_94, var_95]
    var_97 = [var_93, var_96]



# Parsed testcases at query #16
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = lambda x: x + var_2
    var_7 = [var_2, var_0]
    var_8 = 4
    var_9 = [var_3, var_8]
    var_10 = [var_7, var_9]
    var_11 = module_0.map_structure(var_6, var_10)
    var_12 = lambda x: x.upper()
    var_13 = 'a'
    var_14 = 'b'
    var_15 = 'c'
    var_16 = (var_13, var_14, var_15)
    var_17 = module_0.map_structure(var_12, var_16)
    var_18 = lambda x: x * var_3
    var_19 = (var_2, var_0)
    var_20 = (var_3, var_8)
    var_21 = (var_19, var_20)
    var_22 = module_0.map_structure(var_18, var_21)
    var_23 = lambda x: x * var_0
    var_24 = {var_13: var_2, var_14: var_0}
    var_25 = module_0.map_structure(var_23, var_24)
    var_26 = 10
    var_27 = lambda x: x + var_26
    var_28 = 'x'
    var_29 = {var_28: var_2}
    var_30 = 'y'
    var_31 = {var_30: var_0}
    var_32 = {var_13: var_29, var_14: var_31}
    var_33 = module_0.map_structure(var_27, var_32)
    var_34 = lambda x: x ** var_0
    var_35 = {var_2, var_0, var_3}
    var_36 = module_0.map_structure(var_34, var_35)
    var_37 = (var_2, var_0)
    var_38 = [var_3, var_8]
    var_39 = {var_13: var_37, var_14: var_38}
    var_40 = 5
    var_41 = [var_39, var_40]
    var_42 = lambda x: x * var_0
    var_43 = module_0.map_structure(var_42, var_41)
    var_44 = [var_2, var_0, var_3]
    var_45 = module_0.no_map_instance(var_44)
    var_46 = lambda x: x * var_0
    var_47 = module_0.map_structure(var_46, var_45)
    var_48 = [var_2, var_0, var_3]
    var_49 = lambda x: x * var_0
    var_50 = 'Point'
    var_51 = [var_28, var_30]
    var_52 = lambda x: x + var_26
    var_53 = 11
    var_54 = 12
    var_55 = lambda x: x * var_0
    var_56 = module_0.map_structure(var_55, var_40)
    assert var_56 == 10
    var_57 = lambda x: x.upper()
    var_58 = 'hello'
    var_59 = module_0.map_structure(var_57, var_58)
    assert var_59 == 'HELLO'
    var_60 = lambda x: x * var_0
    var_61 = []
    var_62 = module_0.map_structure(var_60, var_61)
    var_63 = lambda x: x * var_0
    var_64 = {}
    var_65 = module_0.map_structure(var_63, var_64)
    var_66 = [var_2, var_0, var_3]



# Parsed testcases at query #17
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 5
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = (var_1, var_2, var_3)
    var_6 = 4
    var_7 = {var_1: var_2, var_3: var_6}
    var_8 = {var_1, var_2, var_3}
    var_9 = [var_1, var_2]
    var_10 = [var_3, var_6]
    var_11 = [var_9, var_10]
    var_12 = 'a'
    var_13 = 'b'
    var_14 = 'x'
    var_15 = 'y'
    var_16 = {var_14: var_1, var_15: var_2}
    var_17 = 'z'
    var_18 = {var_17: var_3}
    var_19 = {var_12: var_16, var_13: var_18}
    var_20 = 'c'
    var_21 = [var_1, var_2, var_3]
    var_22 = (var_6, var_0)
    var_23 = 6
    var_24 = 7
    var_25 = {var_23, var_24}
    var_26 = {var_12: var_21, var_13: var_22, var_20: var_25}
    var_27 = [var_1, var_2, var_3]
    var_28 = module_0.no_map_instance(var_27)
    var_29 = [var_1, var_2, var_3]
    var_30 = 'Point'
    var_31 = [var_14, var_15]
    var_32 = True
    var_33 = [var_1, var_12, var_32]
    var_34 = 'key'
    var_35 = 'value'
    var_36 = {var_34: var_35}
    var_37 = []
    var_38 = {}
    var_39 = set()
    var_40 = set()
    var_41 = ()
    var_42 = 'list'
    var_43 = 'tuple'
    var_44 = 'set'
    var_45 = 'inner'
    var_46 = {var_45: var_2}
    var_47 = (var_3, var_6)
    var_48 = [var_32, var_46, var_47]
    var_49 = [var_23, var_24]
    var_50 = (var_0, var_49)
    var_51 = 8
    var_52 = 9
    var_53 = {var_51, var_52}
    var_54 = {var_42: var_48, var_43: var_50, var_44: var_53}
    var_55 = [var_32, var_2, var_3]
    var_56 = 'd'
    var_57 = {var_20: var_6, var_56: var_0}
    var_58 = {var_12: var_55, var_13: var_57}



# Parsed testcases at query #18
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = lambda x: x + var_2
    var_7 = [var_2, var_0]
    var_8 = 4
    var_9 = [var_3, var_8]
    var_10 = [var_7, var_9]
    var_11 = module_0.map_structure(var_6, var_10)
    var_12 = lambda x: x.upper()
    var_13 = 'a'
    var_14 = 'b'
    var_15 = 'c'
    var_16 = (var_13, var_14, var_15)
    var_17 = module_0.map_structure(var_12, var_16)
    var_18 = lambda x: x * var_3
    var_19 = (var_2, var_0)
    var_20 = (var_3, var_8)
    var_21 = (var_19, var_20)
    var_22 = module_0.map_structure(var_18, var_21)
    var_23 = lambda x: x * var_0
    var_24 = {var_13: var_2, var_14: var_0}
    var_25 = module_0.map_structure(var_23, var_24)
    var_26 = 10
    var_27 = lambda x: x + var_26
    var_28 = 'x'
    var_29 = {var_28: var_2}
    var_30 = 'y'
    var_31 = {var_30: var_0}
    var_32 = {var_13: var_29, var_14: var_31}
    var_33 = module_0.map_structure(var_27, var_32)
    var_34 = lambda x: x * var_0
    var_35 = {var_2, var_0, var_3}
    var_36 = module_0.map_structure(var_34, var_35)
    var_37 = {var_28: var_3}
    var_38 = [var_2, var_0, var_37]
    var_39 = 5
    var_40 = (var_8, var_39)
    var_41 = {var_13: var_38, var_14: var_40}
    var_42 = lambda x: x * var_0
    var_43 = module_0.map_structure(var_42, var_41)
    var_44 = 6
    var_45 = {var_28: var_44}
    var_46 = [var_0, var_8, var_45]
    var_47 = 8
    var_48 = (var_47, var_26)
    var_49 = {var_13: var_46, var_14: var_48}
    var_50 = [var_2, var_0, var_3]
    var_51 = lambda x: x * var_0
    var_52 = [var_2, var_0, var_3]
    var_53 = module_0.no_map_instance(var_52)
    var_54 = lambda x: x * var_0
    var_55 = module_0.map_structure(var_54, var_53)
    var_56 = 'Point'
    var_57 = [var_28, var_30]
    var_58 = lambda x: x + var_26
    var_59 = 11
    var_60 = 12
    var_61 = lambda x: x * var_3
    var_62 = module_0.map_structure(var_61, var_39)
    assert var_62 == 15
    var_63 = lambda x: x
    var_64 = []
    var_65 = module_0.map_structure(var_63, var_64)
    var_66 = lambda x: x
    var_67 = {}
    var_68 = module_0.map_structure(var_66, var_67)
    var_69 = lambda x: x
    var_70 = ()
    var_71 = module_0.map_structure(var_69, var_70)
    var_72 = [var_2, var_0, var_3]
    var_73 = [var_2, var_0]
    var_74 = [var_3, var_8]
    var_75 = [var_73, var_74]
    var_76 = [var_39, var_44]
    var_77 = 7
    var_78 = [var_77, var_47]
    var_79 = [var_76, var_78]
    var_80 = [var_75, var_79]
    var_81 = lambda x: x - var_2
    var_82 = module_0.map_structure(var_81, var_80)
    var_83 = 0
    var_84 = [var_83, var_2]
    var_85 = [var_0, var_3]
    var_86 = [var_84, var_85]
    var_87 = [var_8, var_39]
    var_88 = [var_44, var_77]
    var_89 = [var_87, var_88]
    var_90 = [var_86, var_89]



# Parsed testcases at query #19
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = lambda x, y: x * y
    var_12 = [var_1, var_2]
    var_13 = [var_3, var_5]
    var_14 = [var_12, var_13]
    var_15 = [var_6, var_7]
    var_16 = 7
    var_17 = 8
    var_18 = [var_16, var_17]
    var_19 = [var_15, var_18]
    var_20 = [var_14, var_19]
    var_21 = module_0.map_structure_zip(var_11, var_20)
    var_22 = lambda x, y, z: x + y + z
    var_23 = (var_1, var_2)
    var_24 = (var_3, var_5)
    var_25 = (var_6, var_7)
    var_26 = [var_23, var_24, var_25]
    var_27 = module_0.map_structure_zip(var_22, var_26)
    var_28 = lambda x, y: x + y
    var_29 = 'a'
    var_30 = 'b'
    var_31 = {var_29: var_1, var_30: var_2}
    var_32 = {var_29: var_3, var_30: var_5}
    var_33 = [var_31, var_32]
    var_34 = module_0.map_structure_zip(var_28, var_33)
    var_35 = lambda x, y: f'{x}{y}'
    var_36 = [var_1, var_2]
    var_37 = (var_3, var_5)
    var_38 = {var_29: var_36, var_30: var_37}
    var_39 = [var_6, var_7]
    var_40 = (var_16, var_17)
    var_41 = {var_29: var_39, var_30: var_40}
    var_42 = [var_38, var_41]
    var_43 = module_0.map_structure_zip(var_35, var_42)
    var_44 = [var_1, var_2, var_3]
    var_45 = module_0.no_map_instance(var_44)
    var_46 = lambda x, y: x + y
    var_47 = [var_45, var_45]
    var_48 = module_0.map_structure_zip(var_46, var_47)
    var_49 = [var_1, var_2, var_3]
    var_50 = lambda x, y: x + y
    var_51 = lambda x: x * var_2
    var_52 = [var_1, var_2, var_3]
    var_53 = [var_52]
    var_54 = module_0.map_structure_zip(var_51, var_53)
    var_55 = lambda x, y: x + y
    var_56 = []
    var_57 = []
    var_58 = [var_56, var_57]
    var_59 = module_0.map_structure_zip(var_55, var_58)
    var_60 = 'Point'
    var_61 = 'x'
    var_62 = 'y'
    var_63 = [var_61, var_62]
    var_64 = lambda a, b: Point(a.x + b.x, a.y + b.y)
    var_65 = lambda x, y: x + y
    var_66 = 1
    var_67 = 2
    var_68 = {var_66, var_67}
    var_69 = 3
    var_70 = 4
    var_71 = {var_69, var_70}
    var_72 = [var_68, var_71]
    var_73 = module_0.map_structure_zip(var_65, var_72)



# Parsed testcases at query #20
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = lambda x, y: x * y
    var_12 = [var_1, var_2]
    var_13 = [var_3, var_5]
    var_14 = [var_12, var_13]
    var_15 = [var_6, var_7]
    var_16 = 7
    var_17 = 8
    var_18 = [var_16, var_17]
    var_19 = [var_15, var_18]
    var_20 = [var_14, var_19]
    var_21 = module_0.map_structure_zip(var_11, var_20)
    var_22 = lambda x, y, z: x + y + z
    var_23 = (var_1, var_2)
    var_24 = (var_3, var_5)
    var_25 = (var_6, var_7)
    var_26 = [var_23, var_24, var_25]
    var_27 = module_0.map_structure_zip(var_22, var_26)
    var_28 = lambda x, y: f'{x}{y}'
    var_29 = (var_1, var_2)
    var_30 = (var_3, var_5)
    var_31 = (var_29, var_30)
    var_32 = 'a'
    var_33 = 'b'
    var_34 = (var_32, var_33)
    var_35 = 'c'
    var_36 = 'd'
    var_37 = (var_35, var_36)
    var_38 = (var_34, var_37)
    var_39 = [var_31, var_38]
    var_40 = module_0.map_structure_zip(var_28, var_39)
    var_41 = lambda x, y: x - y
    var_42 = 10
    var_43 = 20
    var_44 = {var_32: var_42, var_33: var_43}
    var_45 = {var_32: var_3, var_33: var_6}
    var_46 = [var_44, var_45]
    var_47 = module_0.map_structure_zip(var_41, var_46)
    var_48 = lambda x, y: x.upper() + y
    var_49 = 'hello'
    var_50 = {var_33: var_49}
    var_51 = {var_32: var_50}
    var_52 = 'world'
    var_53 = {var_33: var_52}
    var_54 = {var_32: var_53}
    var_55 = [var_51, var_54]
    var_56 = module_0.map_structure_zip(var_48, var_55)
    var_57 = lambda x, y: x + str(y)
    var_58 = [var_1, var_2]
    var_59 = (var_3, var_5)
    var_60 = {var_32: var_58, var_33: var_59}
    var_61 = [var_6, var_7]
    var_62 = (var_16, var_17)
    var_63 = {var_32: var_61, var_33: var_62}
    var_64 = [var_60, var_63]
    var_65 = module_0.map_structure_zip(var_57, var_64)
    var_66 = 'Point'
    var_67 = 'x'
    var_68 = 'y'
    var_69 = [var_67, var_68]
    var_70 = 0
    var_71 = lambda x, y: Point(x[var_70] + y[var_70], x[var_1] + y[var_1])
    var_72 = [var_1, var_2, var_3]
    var_73 = [var_5, var_6, var_7]
    var_74 = lambda x, y: x + y
    var_75 = lambda x: x * var_2
    var_76 = [var_1, var_2, var_3]
    var_77 = [var_76]
    var_78 = module_0.map_structure_zip(var_75, var_77)
    var_79 = lambda x, y, z: x + y + z
    var_80 = [var_1, var_2]
    var_81 = [var_3, var_5]
    var_82 = [var_6, var_7]
    var_83 = [var_80, var_81, var_82]
    var_84 = module_0.map_structure_zip(var_79, var_83)
    var_85 = lambda x, y: x + y
    var_86 = 1
    var_87 = 2
    var_88 = {var_86, var_87}
    var_89 = 3
    var_90 = 4
    var_91 = {var_89, var_90}
    var_92 = [var_88, var_91]
    var_93 = module_0.map_structure_zip(var_85, var_92)
    var_94 = lambda x, y: x + y
    var_95 = []
    var_96 = []
    var_97 = [var_95, var_96]
    var_98 = module_0.map_structure_zip(var_94, var_97)
    var_99 = None
    var_100 = lambda x, y: var_99
    var_101 = {}
    var_102 = {}
    var_103 = [var_101, var_102]
    var_104 = module_0.map_structure_zip(var_100, var_103)



# Parsed testcases at query #21
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = lambda x: x + var_2
    var_7 = [var_2, var_0]
    var_8 = 4
    var_9 = [var_3, var_8]
    var_10 = [var_7, var_9]
    var_11 = module_0.map_structure(var_6, var_10)
    var_12 = lambda x: x.upper()
    var_13 = 'a'
    var_14 = 'b'
    var_15 = 'c'
    var_16 = (var_13, var_14, var_15)
    var_17 = module_0.map_structure(var_12, var_16)
    var_18 = lambda x: x * var_3
    var_19 = (var_2, var_0)
    var_20 = (var_3, var_8)
    var_21 = (var_19, var_20)
    var_22 = module_0.map_structure(var_18, var_21)
    var_23 = lambda x: x * var_0
    var_24 = {var_13: var_2, var_14: var_0}
    var_25 = module_0.map_structure(var_23, var_24)
    var_26 = 10
    var_27 = lambda x: x + var_26
    var_28 = 'x'
    var_29 = 'y'
    var_30 = {var_28: var_2, var_29: var_0}
    var_31 = {var_13: var_30, var_14: var_3}
    var_32 = module_0.map_structure(var_27, var_31)
    var_33 = lambda x: x ** var_0
    var_34 = {var_2, var_0, var_3}
    var_35 = module_0.map_structure(var_33, var_34)
    var_36 = {var_28: var_3}
    var_37 = [var_2, var_0, var_36]
    var_38 = 5
    var_39 = (var_8, var_38)
    var_40 = {var_13: var_37, var_14: var_39}
    var_41 = lambda x: x * var_0
    var_42 = module_0.map_structure(var_41, var_40)
    var_43 = lambda x: x + var_2
    var_44 = module_0.map_structure(var_43, var_38)
    assert var_44 == 6
    var_45 = [var_2, var_0, var_3]
    var_46 = lambda x: x * var_0
    var_47 = [var_2, var_0, var_3]
    var_48 = module_0.no_map_instance(var_47)
    var_49 = lambda x: x * var_0
    var_50 = module_0.map_structure(var_49, var_48)
    var_51 = 'Point'
    var_52 = [var_28, var_29]
    var_53 = lambda x: x * var_0
    var_54 = lambda x: x
    var_55 = []
    var_56 = module_0.map_structure(var_54, var_55)
    var_57 = lambda x: x
    var_58 = {}
    var_59 = module_0.map_structure(var_57, var_58)
    var_60 = lambda x: x
    var_61 = set()
    var_62 = module_0.map_structure(var_60, var_61)
    var_63 = set()
    var_64 = [var_0, var_3]
    var_65 = {var_13: var_2, var_14: var_64}
    var_66 = [var_65, var_8]
    var_67 = [var_3, var_8]
    var_68 = {var_13: var_0, var_14: var_67}
    var_69 = [var_68, var_38]



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = lambda x: x + var_2
    var_7 = [var_2, var_0]
    var_8 = 4
    var_9 = [var_3, var_8]
    var_10 = [var_7, var_9]
    var_11 = module_0.map_structure(var_6, var_10)
    var_12 = lambda x: x.upper()
    var_13 = 'a'
    var_14 = 'b'
    var_15 = 'c'
    var_16 = (var_13, var_14, var_15)
    var_17 = module_0.map_structure(var_12, var_16)
    var_18 = lambda x: x * var_3
    var_19 = (var_2, var_0)
    var_20 = (var_3, var_8)
    var_21 = (var_19, var_20)
    var_22 = module_0.map_structure(var_18, var_21)
    var_23 = 10
    var_24 = lambda x: x + var_23
    var_25 = {var_13: var_2, var_14: var_0}
    var_26 = module_0.map_structure(var_24, var_25)
    var_27 = lambda x: len(x)
    var_28 = 'hi'
    var_29 = 'hello'
    var_30 = {var_13: var_28, var_14: var_29}
    var_31 = module_0.map_structure(var_27, var_30)
    var_32 = lambda x: x ** var_0
    var_33 = {var_2, var_0, var_3}
    var_34 = module_0.map_structure(var_32, var_33)
    var_35 = [var_2, var_0, var_3]
    var_36 = 5
    var_37 = 6
    var_38 = (var_8, var_36, var_37)
    var_39 = 'd'
    var_40 = 'e'
    var_41 = 7
    var_42 = 8
    var_43 = {var_39: var_41, var_40: var_42}
    var_44 = {var_13: var_35, var_14: var_38, var_15: var_43}
    var_45 = lambda x: x * var_0
    var_46 = module_0.map_structure(var_45, var_44)
    var_47 = [var_0, var_8, var_37]
    var_48 = 12
    var_49 = (var_42, var_23, var_48)
    var_50 = 14
    var_51 = 16
    var_52 = {var_39: var_50, var_40: var_51}
    var_53 = {var_13: var_47, var_14: var_49, var_15: var_52}
    var_54 = lambda x: x + var_36
    var_55 = module_0.map_structure(var_54, var_23)
    assert var_55 == 15
    var_56 = [var_2, var_0, var_3]
    var_57 = lambda x: x * var_0
    var_58 = [var_2, var_0, var_3]
    var_59 = module_0.no_map_instance(var_58)
    var_60 = lambda x: x * var_0
    var_61 = module_0.map_structure(var_60, var_59)
    var_62 = 'Point'
    var_63 = 'x'
    var_64 = 'y'
    var_65 = [var_63, var_64]
    var_66 = lambda x: x * var_0
    var_67 = lambda x: x
    var_68 = []
    var_69 = module_0.map_structure(var_67, var_68)
    var_70 = lambda x: x
    var_71 = {}
    var_72 = module_0.map_structure(var_70, var_71)
    var_73 = lambda x: x
    var_74 = ()
    var_75 = module_0.map_structure(var_73, var_74)
    var_76 = [var_2, var_0, var_3]
    var_77 = [var_2, var_0]
    var_78 = [var_3, var_8]
    var_79 = [var_77, var_78]
    var_80 = [var_36, var_37]
    var_81 = [var_41, var_42]
    var_82 = [var_80, var_81]
    var_83 = [var_79, var_82]
    var_84 = lambda x: x - var_2
    var_85 = module_0.map_structure(var_84, var_83)
    var_86 = 0
    var_87 = [var_86, var_2]
    var_88 = [var_0, var_3]
    var_89 = [var_87, var_88]
    var_90 = [var_8, var_36]
    var_91 = [var_37, var_41]
    var_92 = [var_90, var_91]
    var_93 = [var_89, var_92]



# Parsed testcases at query #2
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = lambda x: x + var_2
    var_7 = [var_2, var_0]
    var_8 = 4
    var_9 = [var_3, var_8]
    var_10 = [var_7, var_9]
    var_11 = module_0.map_structure(var_6, var_10)
    var_12 = lambda x: x.upper()
    var_13 = 'a'
    var_14 = 'b'
    var_15 = 'c'
    var_16 = (var_13, var_14, var_15)
    var_17 = module_0.map_structure(var_12, var_16)
    var_18 = lambda x: x * var_3
    var_19 = (var_2, var_0)
    var_20 = (var_3, var_8)
    var_21 = (var_19, var_20)
    var_22 = module_0.map_structure(var_18, var_21)
    var_23 = lambda x: x * var_0
    var_24 = {var_13: var_2, var_14: var_0}
    var_25 = module_0.map_structure(var_23, var_24)
    var_26 = 10
    var_27 = lambda x: x + var_26
    var_28 = 'x'
    var_29 = {var_28: var_2}
    var_30 = 'y'
    var_31 = {var_30: var_0}
    var_32 = {var_13: var_29, var_14: var_31}
    var_33 = module_0.map_structure(var_27, var_32)
    var_34 = lambda x: x * var_0
    var_35 = {var_2, var_0, var_3}
    var_36 = module_0.map_structure(var_34, var_35)
    var_37 = [var_2, var_0, var_3]
    var_38 = 5
    var_39 = 6
    var_40 = (var_8, var_38, var_39)
    var_41 = 7
    var_42 = 8
    var_43 = 9
    var_44 = {var_41, var_42, var_43}
    var_45 = {var_13: var_37, var_14: var_40, var_15: var_44}
    var_46 = [var_0, var_8, var_39]
    var_47 = 12
    var_48 = (var_42, var_26, var_47)
    var_49 = 14
    var_50 = 16
    var_51 = 18
    var_52 = {var_49, var_50, var_51}
    var_53 = {var_13: var_46, var_14: var_48, var_15: var_52}
    var_54 = lambda x: x * var_0
    var_55 = module_0.map_structure(var_54, var_45)
    var_56 = lambda x: x * var_0
    var_57 = module_0.map_structure(var_56, var_38)
    assert var_57 == 10
    var_58 = 'Point'
    var_59 = [var_28, var_30]
    var_60 = lambda x: x * var_0
    var_61 = [var_2, var_0, var_3]
    var_62 = lambda x: x * var_0
    var_63 = [var_2, var_0, var_3]
    var_64 = [var_2, var_0, var_3]
    var_65 = module_0.no_map_instance(var_64)
    var_66 = lambda x: x * var_0
    var_67 = module_0.map_structure(var_66, var_65)
    var_68 = lambda x: x * var_0
    var_69 = []
    var_70 = module_0.map_structure(var_68, var_69)
    var_71 = lambda x: x * var_0
    var_72 = {}
    var_73 = module_0.map_structure(var_71, var_72)
    var_74 = lambda x: x * var_0
    var_75 = ()
    var_76 = module_0.map_structure(var_74, var_75)
    var_77 = [var_2, var_0, var_3]
    var_78 = (var_2, var_0)
    var_79 = {var_28: var_78}
    var_80 = [var_3, var_8]
    var_81 = {var_30: var_80}
    var_82 = [var_79, var_81]
    var_83 = {var_13: var_82}
    var_84 = (var_0, var_8)
    var_85 = {var_28: var_84}
    var_86 = [var_39, var_42]
    var_87 = {var_30: var_86}
    var_88 = [var_85, var_87]
    var_89 = {var_13: var_88}
    var_90 = lambda x: x * var_0
    var_91 = module_0.map_structure(var_90, var_83)



# Parsed testcases at query #3
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = lambda x, y: x * y
    var_12 = [var_1, var_2]
    var_13 = [var_3, var_5]
    var_14 = [var_12, var_13]
    var_15 = [var_6, var_7]
    var_16 = 7
    var_17 = 8
    var_18 = [var_16, var_17]
    var_19 = [var_15, var_18]
    var_20 = [var_14, var_19]
    var_21 = module_0.map_structure_zip(var_11, var_20)
    var_22 = lambda x, y, z: x + y + z
    var_23 = (var_1, var_2)
    var_24 = (var_3, var_5)
    var_25 = (var_6, var_7)
    var_26 = [var_23, var_24, var_25]
    var_27 = module_0.map_structure_zip(var_22, var_26)
    var_28 = lambda x, y: f'{x}{y}'
    var_29 = 'a'
    var_30 = 'b'
    var_31 = {var_29: var_1, var_30: var_2}
    var_32 = {var_29: var_3, var_30: var_5}
    var_33 = [var_31, var_32]
    var_34 = module_0.map_structure_zip(var_28, var_33)
    var_35 = lambda x, y: x + y
    var_36 = [var_1, var_2]
    var_37 = (var_3, var_5)
    var_38 = {var_29: var_36, var_30: var_37}
    var_39 = [var_6, var_7]
    var_40 = (var_16, var_17)
    var_41 = {var_29: var_39, var_30: var_40}
    var_42 = [var_38, var_41]
    var_43 = module_0.map_structure_zip(var_35, var_42)
    var_44 = 'Point'
    var_45 = 'x'
    var_46 = 'y'
    var_47 = [var_45, var_46]
    var_48 = lambda p1, p2: Point(p1.x + p2.x, p1.y + p2.y)
    var_49 = [var_1, var_2, var_3]
    var_50 = module_0.no_map_instance(var_49)
    var_51 = lambda x, y: x + y
    var_52 = [var_50, var_50]
    var_53 = module_0.map_structure_zip(var_51, var_52)
    var_54 = [var_1, var_2, var_3]
    var_55 = lambda x, y: x + y
    var_56 = lambda x: x * var_2
    var_57 = [var_1, var_2, var_3]
    var_58 = [var_57]
    var_59 = module_0.map_structure_zip(var_56, var_58)
    var_60 = lambda x, y, z: x + y + z
    var_61 = [var_1, var_2]
    var_62 = [var_3, var_5]
    var_63 = [var_6, var_7]
    var_64 = [var_61, var_62, var_63]
    var_65 = module_0.map_structure_zip(var_60, var_64)
    var_66 = lambda x, y: x + y
    var_67 = 1
    var_68 = 2
    var_69 = {var_67, var_68}
    var_70 = 3
    var_71 = 4
    var_72 = {var_70, var_71}
    var_73 = [var_69, var_72]
    var_74 = module_0.map_structure_zip(var_66, var_73)
    var_75 = lambda x, y: x + y
    var_76 = []
    var_77 = []
    var_78 = [var_76, var_77]
    var_79 = module_0.map_structure_zip(var_75, var_78)
    var_80 = lambda x, y: x + y
    var_81 = {}
    var_82 = {}
    var_83 = [var_81, var_82]
    var_84 = module_0.map_structure_zip(var_80, var_83)



# Parsed testcases at query #4
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = lambda x, y: x * y
    var_12 = [var_1, var_2]
    var_13 = [var_3, var_5]
    var_14 = [var_12, var_13]
    var_15 = [var_6, var_7]
    var_16 = 7
    var_17 = 8
    var_18 = [var_16, var_17]
    var_19 = [var_15, var_18]
    var_20 = [var_14, var_19]
    var_21 = module_0.map_structure_zip(var_11, var_20)
    var_22 = lambda x, y, z: x + y + z
    var_23 = (var_1, var_2)
    var_24 = (var_3, var_5)
    var_25 = (var_6, var_7)
    var_26 = [var_23, var_24, var_25]
    var_27 = module_0.map_structure_zip(var_22, var_26)
    var_28 = lambda x, y: f'{x}{y}'
    var_29 = (var_1, var_2)
    var_30 = (var_3, var_5)
    var_31 = (var_29, var_30)
    var_32 = 'a'
    var_33 = 'b'
    var_34 = (var_32, var_33)
    var_35 = 'c'
    var_36 = 'd'
    var_37 = (var_35, var_36)
    var_38 = (var_34, var_37)
    var_39 = [var_31, var_38]
    var_40 = module_0.map_structure_zip(var_28, var_39)
    var_41 = lambda x, y: x - y
    var_42 = 10
    var_43 = 20
    var_44 = {var_32: var_42, var_33: var_43}
    var_45 = {var_32: var_3, var_33: var_6}
    var_46 = [var_44, var_45]
    var_47 = module_0.map_structure_zip(var_41, var_46)
    var_48 = lambda x, y: x.upper() + y
    var_49 = 'hello'
    var_50 = {var_33: var_49}
    var_51 = {var_32: var_50}
    var_52 = 'world'
    var_53 = {var_33: var_52}
    var_54 = {var_32: var_53}
    var_55 = [var_51, var_54]
    var_56 = module_0.map_structure_zip(var_48, var_55)
    var_57 = lambda x, y: x + y
    var_58 = [var_1, var_2]
    var_59 = (var_3, var_5)
    var_60 = {var_32: var_58, var_33: var_59}
    var_61 = [var_6, var_7]
    var_62 = (var_16, var_17)
    var_63 = {var_32: var_61, var_33: var_62}
    var_64 = [var_60, var_63]
    var_65 = module_0.map_structure_zip(var_57, var_64)
    var_66 = [var_1, var_2, var_3]
    var_67 = module_0.no_map_instance(var_66)
    var_68 = lambda x, y: (x, y)
    var_69 = [var_5, var_6, var_7]
    var_70 = [var_67, var_69]
    var_71 = module_0.map_structure_zip(var_68, var_70)
    var_72 = [var_1, var_2, var_3]
    var_73 = lambda x, y: x + y
    var_74 = [var_5, var_6, var_7]
    var_75 = lambda x: x * var_2
    var_76 = [var_1, var_2, var_3]
    var_77 = [var_76]
    var_78 = module_0.map_structure_zip(var_75, var_77)
    var_79 = lambda x, y, z: x + y + z
    var_80 = [var_1, var_2]
    var_81 = [var_3, var_5]
    var_82 = [var_6, var_7]
    var_83 = [var_80, var_81, var_82]
    var_84 = module_0.map_structure_zip(var_79, var_83)
    var_85 = lambda x, y: x + y
    var_86 = []
    var_87 = []
    var_88 = [var_86, var_87]
    var_89 = module_0.map_structure_zip(var_85, var_88)
    var_90 = lambda x, y: x + y
    var_91 = 1
    var_92 = 2
    var_93 = {var_91, var_92}
    var_94 = 3
    var_95 = 4
    var_96 = {var_94, var_95}
    var_97 = [var_93, var_96]
    var_98 = module_0.map_structure_zip(var_90, var_97)
    var_99 = 'Point'
    var_100 = 'x'
    var_101 = 'y'
    var_102 = [var_100, var_101]
    var_103 = lambda a, b: a + b
    var_104 = [var_92, var_93]
    var_105 = (var_91, var_104)
    var_106 = {var_35: var_95}
    var_107 = {var_32: var_105, var_33: var_106}
    var_108 = [var_97, var_16]
    var_109 = (var_96, var_108)
    var_110 = {var_35: var_17}
    var_111 = {var_32: var_109, var_33: var_110}
    var_112 = [var_107, var_111]
    var_113 = lambda x, y: x * y
    var_114 = module_0.map_structure_zip(var_113, var_112)
    var_115 = 12
    var_116 = 21
    var_117 = [var_115, var_116]
    var_118 = (var_96, var_117)
    var_119 = 32
    var_120 = {var_35: var_119}
    var_121 = {var_32: var_118, var_33: var_120}



# Parsed testcases at query #5
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = lambda x: x + var_2
    var_7 = [var_2, var_0]
    var_8 = 4
    var_9 = [var_3, var_8]
    var_10 = [var_7, var_9]
    var_11 = module_0.map_structure(var_6, var_10)
    var_12 = lambda x: x.upper()
    var_13 = 'a'
    var_14 = 'b'
    var_15 = 'c'
    var_16 = (var_13, var_14, var_15)
    var_17 = module_0.map_structure(var_12, var_16)
    var_18 = lambda x: x * var_3
    var_19 = (var_2, var_0)
    var_20 = (var_3, var_8)
    var_21 = (var_19, var_20)
    var_22 = module_0.map_structure(var_18, var_21)
    var_23 = lambda x: x * var_0
    var_24 = {var_13: var_2, var_14: var_0}
    var_25 = module_0.map_structure(var_23, var_24)
    var_26 = 10
    var_27 = lambda x: x + var_26
    var_28 = 'x'
    var_29 = {var_28: var_2}
    var_30 = 'y'
    var_31 = {var_30: var_0}
    var_32 = {var_13: var_29, var_14: var_31}
    var_33 = module_0.map_structure(var_27, var_32)
    var_34 = lambda x: x ** var_0
    var_35 = {var_2, var_0, var_3}
    var_36 = module_0.map_structure(var_34, var_35)
    var_37 = (var_2, var_0)
    var_38 = [var_3, var_8]
    var_39 = {var_13: var_37, var_14: var_38}
    var_40 = 5
    var_41 = [var_39, var_40]
    var_42 = lambda x: x * var_0
    var_43 = module_0.map_structure(var_42, var_41)
    var_44 = [var_2, var_0, var_3]
    var_45 = module_0.no_map_instance(var_44)
    var_46 = lambda x: x * var_0
    var_47 = module_0.map_structure(var_46, var_45)
    var_48 = [var_2, var_0, var_3]
    var_49 = lambda x: x * var_0
    var_50 = 'Point'
    var_51 = [var_28, var_30]
    var_52 = lambda x: x * var_0
    var_53 = lambda x: x + var_40
    var_54 = module_0.map_structure(var_53, var_26)
    assert var_54 == 15
    var_55 = lambda x: x
    var_56 = []
    var_57 = module_0.map_structure(var_55, var_56)
    var_58 = lambda x: x
    var_59 = {}
    var_60 = module_0.map_structure(var_58, var_59)
    var_61 = lambda x: x
    var_62 = ()
    var_63 = module_0.map_structure(var_61, var_62)
    var_64 = [var_2, var_0, var_3]
    var_65 = {var_14: var_3}
    var_66 = (var_0, var_65)
    var_67 = [var_2, var_66]
    var_68 = {var_13: var_67, var_15: var_8}
    var_69 = lambda x: x * var_0
    var_70 = module_0.map_structure(var_69, var_68)



# Parsed testcases at query #6
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 10
    var_4 = lambda x: x.upper()
    var_5 = 'hello'
    var_6 = module_0.map_structure(var_4, var_5)
    assert var_6 == 'HELLO'
    var_7 = 1
    var_8 = lambda x: x + var_7
    var_9 = 3
    var_10 = [var_7, var_0, var_9]
    var_11 = module_0.map_structure(var_8, var_10)
    var_12 = lambda x: x * var_0
    var_13 = [var_7, var_0]
    var_14 = 4
    var_15 = [var_9, var_14]
    var_16 = [var_13, var_15]
    var_17 = module_0.map_structure(var_12, var_16)
    var_18 = lambda x: x.upper()
    var_19 = 'a'
    var_20 = 'b'
    var_21 = 'c'
    var_22 = (var_19, var_20, var_21)
    var_23 = module_0.map_structure(var_18, var_22)
    var_24 = 10
    var_25 = lambda x: x + var_24
    var_26 = (var_7, var_0)
    var_27 = (var_9, var_14)
    var_28 = (var_26, var_27)
    var_29 = module_0.map_structure(var_25, var_28)
    var_30 = lambda x: x * var_0
    var_31 = {var_19: var_7, var_20: var_0}
    var_32 = module_0.map_structure(var_30, var_31)
    var_33 = lambda x: x.upper()
    var_34 = {var_20: var_5}
    var_35 = 'world'
    var_36 = {var_19: var_34, var_21: var_35}
    var_37 = module_0.map_structure(var_33, var_36)
    var_38 = lambda x: x * var_0
    var_39 = {var_7, var_0, var_9}
    var_40 = module_0.map_structure(var_38, var_39)
    var_41 = {var_20: var_9}
    var_42 = [var_7, var_0, var_41]
    var_43 = (var_14, var_2)
    var_44 = {var_19: var_42, var_21: var_43}
    var_45 = 6
    var_46 = {var_20: var_45}
    var_47 = [var_0, var_14, var_46]
    var_48 = 8
    var_49 = (var_48, var_24)
    var_50 = {var_19: var_47, var_21: var_49}
    var_51 = lambda x: x * var_0
    var_52 = module_0.map_structure(var_51, var_44)
    var_53 = [var_7, var_0, var_9]
    var_54 = module_0.no_map_instance(var_53)
    var_55 = lambda x: x * var_0
    var_56 = module_0.map_structure(var_55, var_54)
    var_57 = [var_7, var_0, var_9]
    var_58 = lambda x: x * var_0
    var_59 = [var_7, var_0, var_9]
    var_60 = 'Point'
    var_61 = 'x'
    var_62 = 'y'
    var_63 = [var_61, var_62]
    var_64 = lambda x: x * var_0
    var_65 = lambda x: x * var_0
    var_66 = []
    var_67 = module_0.map_structure(var_65, var_66)
    var_68 = lambda x: x * var_0
    var_69 = {}
    var_70 = module_0.map_structure(var_68, var_69)
    var_71 = lambda x: x * var_0
    var_72 = set()
    var_73 = module_0.map_structure(var_71, var_72)
    var_74 = set()
    var_75 = [var_7, var_0, var_9]
    var_76 = 'list'
    var_77 = 'tuple'
    var_78 = 'set'
    var_79 = {var_19: var_0, var_20: var_9}
    var_80 = (var_14, var_2)
    var_81 = [var_7, var_79, var_80]
    var_82 = 7
    var_83 = [var_82, var_48]
    var_84 = (var_45, var_83)
    var_85 = 9
    var_86 = {var_85, var_24}
    var_87 = {var_76: var_81, var_77: var_84, var_78: var_86}
    var_88 = 100
    var_89 = lambda x: x + var_88
    var_90 = module_0.map_structure(var_89, var_87)



# Parsed testcases at query #7
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = [var_0, var_1]
    var_5 = 4
    var_6 = [var_2, var_5]
    var_7 = [var_4, var_6]
    var_8 = 'a'
    var_9 = 'b'
    var_10 = {var_8: var_1, var_9: var_2}
    var_11 = 'x'
    var_12 = 'y'
    var_13 = {var_11: var_0, var_12: var_1}
    var_14 = {var_8: var_13, var_9: var_2}
    var_15 = (var_0, var_1, var_2)
    var_16 = 'Point'
    var_17 = [var_11, var_12]
    var_18 = {var_0, var_1, var_2}
    var_19 = 'c'
    var_20 = [var_0, var_1, var_2]
    var_21 = 5
    var_22 = (var_5, var_21)
    var_23 = 6
    var_24 = 7
    var_25 = {var_11: var_23, var_12: var_24}
    var_26 = {var_8: var_20, var_9: var_22, var_19: var_25}
    var_27 = lambda x: x * var_1
    var_28 = module_0.map_structure(var_27, var_26)
    assert var_28 == 'HELLO'
    var_29 = [var_0, var_1, var_2]
    var_30 = module_0.no_map_instance(var_29)
    var_31 = [var_0, var_1, var_2]
    var_32 = 'hello'
    var_33 = 10
    var_34 = lambda x: x * var_33
    var_35 = module_0.map_structure(var_34, var_21)
    assert var_35 == 50
    var_36 = []
    var_37 = {}
    var_38 = [var_0, var_1, var_2]



# Parsed testcases at query #8
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = lambda x: x + var_2
    var_7 = [var_2, var_0]
    var_8 = 4
    var_9 = [var_3, var_8]
    var_10 = [var_7, var_9]
    var_11 = module_0.map_structure(var_6, var_10)
    var_12 = (var_2, var_0, var_3)
    var_13 = lambda x: x * var_3
    var_14 = (var_2, var_0)
    var_15 = (var_3, var_8)
    var_16 = (var_14, var_15)
    var_17 = module_0.map_structure(var_13, var_16)
    var_18 = lambda x: x.upper()
    var_19 = 'a'
    var_20 = 'b'
    var_21 = 'hello'
    var_22 = 'world'
    var_23 = {var_19: var_21, var_20: var_22}
    var_24 = module_0.map_structure(var_18, var_23)
    var_25 = [var_21, var_22]
    var_26 = 'test'
    var_27 = [var_26]
    var_28 = {var_19: var_25, var_20: var_27}
    var_29 = lambda x: x ** var_0
    var_30 = {var_2, var_0, var_3}
    var_31 = module_0.map_structure(var_29, var_30)
    var_32 = 'c'
    var_33 = [var_2, var_0, var_3]
    var_34 = 5
    var_35 = (var_8, var_34)
    var_36 = 6
    var_37 = 7
    var_38 = {var_36, var_37}
    var_39 = {var_19: var_33, var_20: var_35, var_32: var_38}
    var_40 = lambda x: x * var_0
    var_41 = module_0.map_structure(var_40, var_39)
    var_42 = 'Point'
    var_43 = 'x'
    var_44 = 'y'
    var_45 = [var_43, var_44]
    var_46 = 10
    var_47 = lambda x: x + var_46
    var_48 = 11
    var_49 = 12
    var_50 = [var_2, var_0, var_3]
    var_51 = module_0.no_map_instance(var_50)
    var_52 = lambda x: x * var_0
    var_53 = module_0.map_structure(var_52, var_51)
    var_54 = [var_2, var_0, var_3]
    var_55 = lambda x: x * var_0
    var_56 = lambda x: x * var_0
    var_57 = module_0.map_structure(var_56, var_34)
    assert var_57 == 10
    var_58 = '!'
    var_59 = lambda x: x + var_58
    var_60 = module_0.map_structure(var_59, var_21)
    assert var_60 == 'hello!'
    var_61 = lambda x: x
    var_62 = []
    var_63 = module_0.map_structure(var_61, var_62)
    var_64 = lambda x: x
    var_65 = {}
    var_66 = module_0.map_structure(var_64, var_65)
    var_67 = lambda x: x
    var_68 = ()
    var_69 = module_0.map_structure(var_67, var_68)
    var_70 = lambda x: x
    var_71 = set()
    var_72 = module_0.map_structure(var_70, var_71)
    var_73 = set()
    var_74 = (var_2, var_0)
    var_75 = {var_20: var_74}
    var_76 = {var_3, var_8}
    var_77 = {var_32: var_76}
    var_78 = [var_75, var_77]
    var_79 = {var_19: var_78}
    var_80 = lambda x: x + var_2
    var_81 = module_0.map_structure(var_80, var_79)



# Parsed testcases at query #9
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = lambda x, y: x * y
    var_12 = [var_1, var_2]
    var_13 = [var_3, var_5]
    var_14 = [var_12, var_13]
    var_15 = [var_6, var_7]
    var_16 = 7
    var_17 = 8
    var_18 = [var_16, var_17]
    var_19 = [var_15, var_18]
    var_20 = [var_14, var_19]
    var_21 = module_0.map_structure_zip(var_11, var_20)
    var_22 = lambda x, y, z: x + y + z
    var_23 = (var_1, var_2)
    var_24 = (var_3, var_5)
    var_25 = (var_6, var_7)
    var_26 = [var_23, var_24, var_25]
    var_27 = module_0.map_structure_zip(var_22, var_26)
    var_28 = lambda x, y: f'{x}{y}'
    var_29 = (var_1, var_2)
    var_30 = (var_3, var_5)
    var_31 = (var_29, var_30)
    var_32 = 'a'
    var_33 = 'b'
    var_34 = (var_32, var_33)
    var_35 = 'c'
    var_36 = 'd'
    var_37 = (var_35, var_36)
    var_38 = (var_34, var_37)
    var_39 = [var_31, var_38]
    var_40 = module_0.map_structure_zip(var_28, var_39)
    var_41 = lambda x, y: x - y
    var_42 = 10
    var_43 = 20
    var_44 = {var_32: var_42, var_33: var_43}
    var_45 = {var_32: var_3, var_33: var_6}
    var_46 = [var_44, var_45]
    var_47 = module_0.map_structure_zip(var_41, var_46)
    var_48 = lambda x, y: x.upper() + y
    var_49 = 'hello'
    var_50 = {var_33: var_49}
    var_51 = {var_32: var_50}
    var_52 = 'world'
    var_53 = {var_33: var_52}
    var_54 = {var_32: var_53}
    var_55 = [var_51, var_54]
    var_56 = module_0.map_structure_zip(var_48, var_55)
    var_57 = lambda x, y: x + y
    var_58 = [var_1, var_2]
    var_59 = (var_3, var_5)
    var_60 = {var_32: var_58, var_33: var_59}
    var_61 = [var_6, var_7]
    var_62 = (var_16, var_17)
    var_63 = {var_32: var_61, var_33: var_62}
    var_64 = [var_60, var_63]
    var_65 = module_0.map_structure_zip(var_57, var_64)
    var_66 = lambda x, y, z: x * y * z
    var_67 = [var_1, var_2]
    var_68 = [var_3, var_5]
    var_69 = [var_6, var_7]
    var_70 = [var_67, var_68, var_69]
    var_71 = module_0.map_structure_zip(var_66, var_70)
    var_72 = 'Point'
    var_73 = 'x'
    var_74 = 'y'
    var_75 = [var_73, var_74]
    var_76 = lambda x, y: (x, y)
    var_77 = [var_1, var_2, var_3]
    var_78 = [var_5, var_6, var_7]
    var_79 = lambda x, y: x + y
    var_80 = len(var_71)
    assert var_80 == 2
    var_81 = lambda x, y: x + y
    var_82 = 1
    var_83 = 2
    var_84 = {var_82, var_83}
    var_85 = 3
    var_86 = 4
    var_87 = {var_85, var_86}
    var_88 = [var_84, var_87]
    var_89 = module_0.map_structure_zip(var_81, var_88)
    var_90 = lambda x: x * var_83
    var_91 = [var_82, var_83, var_84]
    var_92 = [var_91]
    var_93 = module_0.map_structure_zip(var_90, var_92)
    var_94 = lambda x, y: x + y
    var_95 = []
    var_96 = []
    var_97 = [var_95, var_96]
    var_98 = module_0.map_structure_zip(var_94, var_97)
    var_99 = lambda x, y: x + y
    var_100 = {}
    var_101 = {}
    var_102 = [var_100, var_101]
    var_103 = module_0.map_structure_zip(var_99, var_102)



# Parsed testcases at query #10
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = lambda x: x + var_2
    var_7 = [var_2, var_0]
    var_8 = 4
    var_9 = [var_3, var_8]
    var_10 = [var_7, var_9]
    var_11 = module_0.map_structure(var_6, var_10)
    var_12 = lambda x: x.upper()
    var_13 = 'a'
    var_14 = 'b'
    var_15 = 'c'
    var_16 = (var_13, var_14, var_15)
    var_17 = module_0.map_structure(var_12, var_16)
    var_18 = lambda x: x * var_3
    var_19 = (var_2, var_0)
    var_20 = (var_3, var_8)
    var_21 = (var_19, var_20)
    var_22 = module_0.map_structure(var_18, var_21)
    var_23 = lambda x: x * var_0
    var_24 = {var_13: var_2, var_14: var_0}
    var_25 = module_0.map_structure(var_23, var_24)
    var_26 = 10
    var_27 = lambda x: x + var_26
    var_28 = 'x'
    var_29 = {var_28: var_2}
    var_30 = 'y'
    var_31 = {var_30: var_0}
    var_32 = {var_13: var_29, var_14: var_31}
    var_33 = module_0.map_structure(var_27, var_32)
    var_34 = lambda x: x ** var_0
    var_35 = {var_2, var_0, var_3}
    var_36 = module_0.map_structure(var_34, var_35)
    var_37 = [var_2, var_0, var_3]
    var_38 = 5
    var_39 = (var_8, var_38)
    var_40 = 6
    var_41 = 7
    var_42 = {var_40, var_41}
    var_43 = {var_13: var_37, var_14: var_39, var_15: var_42}
    var_44 = lambda x: x - var_2
    var_45 = module_0.map_structure(var_44, var_43)
    var_46 = lambda x: x.upper()
    var_47 = 'hello'
    var_48 = module_0.map_structure(var_46, var_47)
    assert var_48 == 'HELLO'
    var_49 = [var_2, var_0, var_3]
    var_50 = lambda x: x * var_0
    var_51 = [var_2, var_0, var_3]
    var_52 = module_0.no_map_instance(var_51)
    var_53 = lambda x: x * var_0
    var_54 = module_0.map_structure(var_53, var_52)
    var_55 = 'Point'
    var_56 = [var_28, var_30]
    var_57 = lambda x: x * var_26
    var_58 = 20
    var_59 = lambda x: x
    var_60 = []
    var_61 = module_0.map_structure(var_59, var_60)
    var_62 = lambda x: x
    var_63 = {}
    var_64 = module_0.map_structure(var_62, var_63)
    var_65 = lambda x: x
    var_66 = ()
    var_67 = module_0.map_structure(var_65, var_66)
    var_68 = [var_2, var_0]
    var_69 = (var_3, var_8)
    var_70 = {var_13: var_68, var_14: var_69}
    var_71 = lambda x: x
    var_72 = module_0.map_structure(var_71, var_70)
    var_73 = [var_2, var_0, var_3]



# Parsed testcases at query #11
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = lambda x: x + var_2
    var_7 = [var_2, var_0]
    var_8 = 4
    var_9 = [var_3, var_8]
    var_10 = [var_7, var_9]
    var_11 = module_0.map_structure(var_6, var_10)
    var_12 = lambda x: x.upper()
    var_13 = 'a'
    var_14 = 'b'
    var_15 = 'c'
    var_16 = (var_13, var_14, var_15)
    var_17 = module_0.map_structure(var_12, var_16)
    var_18 = lambda x: x * var_3
    var_19 = (var_0, var_3)
    var_20 = (var_2, var_19, var_8)
    var_21 = module_0.map_structure(var_18, var_20)
    var_22 = 10
    var_23 = lambda x: x + var_22
    var_24 = {var_13: var_2, var_14: var_0}
    var_25 = module_0.map_structure(var_23, var_24)
    var_26 = lambda x: len(x)
    var_27 = 'hi'
    var_28 = 'hello'
    var_29 = {var_15: var_28}
    var_30 = {var_13: var_27, var_14: var_29}
    var_31 = module_0.map_structure(var_26, var_30)
    var_32 = lambda x: x ** var_0
    var_33 = {var_2, var_0, var_3}
    var_34 = module_0.map_structure(var_32, var_33)
    var_35 = [var_2, var_0, var_3]
    var_36 = 5
    var_37 = (var_8, var_36)
    var_38 = 'd'
    var_39 = 6
    var_40 = {var_38: var_39}
    var_41 = {var_13: var_35, var_14: var_37, var_15: var_40}
    var_42 = lambda x: x - var_2
    var_43 = module_0.map_structure(var_42, var_41)
    var_44 = [var_2, var_0, var_3]
    var_45 = module_0.no_map_instance(var_44)
    var_46 = lambda x: x * var_0
    var_47 = module_0.map_structure(var_46, var_45)
    var_48 = [var_2, var_0, var_3]
    var_49 = lambda x: x * var_0
    var_50 = 'Point'
    var_51 = 'x'
    var_52 = 'y'
    var_53 = [var_51, var_52]
    var_54 = lambda x: x * var_0
    var_55 = lambda x: x.upper()
    var_56 = module_0.map_structure(var_55, var_28)
    assert var_56 == 'HELLO'
    var_57 = lambda x: x
    var_58 = []
    var_59 = module_0.map_structure(var_57, var_58)
    var_60 = lambda x: x
    var_61 = {}
    var_62 = module_0.map_structure(var_60, var_61)
    var_63 = lambda x: x
    var_64 = ()
    var_65 = module_0.map_structure(var_63, var_64)
    var_66 = {var_13: var_0}
    var_67 = (var_3, var_8)
    var_68 = [var_2, var_66, var_67]
    var_69 = lambda x: x
    var_70 = module_0.map_structure(var_69, var_68)



# Parsed testcases at query #12
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = lambda x, y: x * y
    var_12 = [var_1, var_2]
    var_13 = [var_3, var_5]
    var_14 = [var_12, var_13]
    var_15 = [var_6, var_7]
    var_16 = 7
    var_17 = 8
    var_18 = [var_16, var_17]
    var_19 = [var_15, var_18]
    var_20 = [var_14, var_19]
    var_21 = module_0.map_structure_zip(var_11, var_20)
    var_22 = lambda x, y, z: x + y + z
    var_23 = (var_1, var_2)
    var_24 = (var_3, var_5)
    var_25 = (var_6, var_7)
    var_26 = [var_23, var_24, var_25]
    var_27 = module_0.map_structure_zip(var_22, var_26)
    var_28 = lambda x, y: f'{x}{y}'
    var_29 = (var_1, var_2)
    var_30 = (var_3, var_5)
    var_31 = (var_29, var_30)
    var_32 = 'a'
    var_33 = 'b'
    var_34 = (var_32, var_33)
    var_35 = 'c'
    var_36 = 'd'
    var_37 = (var_35, var_36)
    var_38 = (var_34, var_37)
    var_39 = [var_31, var_38]
    var_40 = module_0.map_structure_zip(var_28, var_39)
    var_41 = lambda x, y: x - y
    var_42 = 10
    var_43 = 20
    var_44 = {var_32: var_42, var_33: var_43}
    var_45 = {var_32: var_3, var_33: var_6}
    var_46 = [var_44, var_45]
    var_47 = module_0.map_structure_zip(var_41, var_46)
    var_48 = lambda x, y: x.upper() + y
    var_49 = 'k1'
    var_50 = 'k2'
    var_51 = 'hello'
    var_52 = {var_50: var_51}
    var_53 = {var_49: var_52}
    var_54 = 'world'
    var_55 = {var_50: var_54}
    var_56 = {var_49: var_55}
    var_57 = [var_53, var_56]
    var_58 = module_0.map_structure_zip(var_48, var_57)
    var_59 = lambda x, y: x + y
    var_60 = 'list'
    var_61 = 'tuple'
    var_62 = [var_1, var_2]
    var_63 = (var_3, var_5)
    var_64 = {var_60: var_62, var_61: var_63}
    var_65 = [var_6, var_7]
    var_66 = (var_16, var_17)
    var_67 = {var_60: var_65, var_61: var_66}
    var_68 = [var_64, var_67]
    var_69 = module_0.map_structure_zip(var_59, var_68)
    var_70 = lambda x: x * var_2
    var_71 = [var_1, var_2, var_3]
    var_72 = [var_71]
    var_73 = module_0.map_structure_zip(var_70, var_72)
    var_74 = lambda x, y, z: x + y + z
    var_75 = [var_1, var_2]
    var_76 = [var_3, var_5]
    var_77 = [var_6, var_7]
    var_78 = [var_75, var_76, var_77]
    var_79 = module_0.map_structure_zip(var_74, var_78)
    var_80 = [var_1, var_2, var_3]
    var_81 = module_0.no_map_instance(var_80)
    var_82 = lambda x, y: x + y
    var_83 = [var_81, var_81]
    var_84 = module_0.map_structure_zip(var_82, var_83)
    var_85 = [var_1, var_2, var_3]
    var_86 = lambda x, y: x + y
    var_87 = lambda x, y: x + y
    var_88 = 1
    var_89 = 2
    var_90 = {var_88, var_89}
    var_91 = 3
    var_92 = 4
    var_93 = {var_91, var_92}
    var_94 = [var_90, var_93]
    var_95 = module_0.map_structure_zip(var_87, var_94)
    var_96 = 'Point'
    var_97 = 'x'
    var_98 = 'y'
    var_99 = [var_97, var_98]
    var_100 = lambda a, b: a + b
    var_101 = lambda x, y: x + y
    var_102 = []
    var_103 = []
    var_104 = [var_102, var_103]
    var_105 = module_0.map_structure_zip(var_101, var_104)
    var_106 = lambda x, y: x + y
    var_107 = {}
    var_108 = {}
    var_109 = [var_107, var_108]
    var_110 = module_0.map_structure_zip(var_106, var_109)
    var_111 = lambda x, y: x + y
    var_112 = ()
    var_113 = ()
    var_114 = [var_112, var_113]
    var_115 = module_0.map_structure_zip(var_111, var_114)



# Parsed testcases at query #13
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 4
    var_5 = 5
    var_6 = 6
    var_7 = [var_4, var_5, var_6]
    var_8 = [var_3, var_7]
    var_9 = lambda x, y: x + y
    var_10 = module_0.map_structure_zip(var_9, var_8)
    var_11 = [var_0, var_1]
    var_12 = [var_2, var_4]
    var_13 = [var_11, var_12]
    var_14 = [var_5, var_6]
    var_15 = 7
    var_16 = 8
    var_17 = [var_15, var_16]
    var_18 = [var_14, var_17]
    var_19 = [var_13, var_18]
    var_20 = lambda x, y: x + y
    var_21 = module_0.map_structure_zip(var_20, var_19)
    var_22 = (var_0, var_1, var_2)
    var_23 = (var_4, var_5, var_6)
    var_24 = [var_22, var_23]
    var_25 = lambda x, y: x * y
    var_26 = module_0.map_structure_zip(var_25, var_24)
    var_27 = (var_0, var_1)
    var_28 = (var_2, var_4)
    var_29 = (var_27, var_28)
    var_30 = (var_5, var_6)
    var_31 = (var_15, var_16)
    var_32 = (var_30, var_31)
    var_33 = [var_29, var_32]
    var_34 = lambda x, y: x - y
    var_35 = module_0.map_structure_zip(var_34, var_33)
    var_36 = 'Point'
    var_37 = 'x'
    var_38 = 'y'
    var_39 = [var_37, var_38]
    var_40 = lambda x, y: x + y
    var_41 = module_0.map_structure_zip(var_40, var_33)
    var_42 = 'a'
    var_43 = 'b'
    var_44 = {var_42: var_0, var_43: var_1}
    var_45 = {var_42: var_2, var_43: var_4}
    var_46 = [var_44, var_45]
    var_47 = lambda x, y: x * y
    var_48 = module_0.map_structure_zip(var_47, var_46)
    var_49 = {var_37: var_0}
    var_50 = {var_42: var_49, var_43: var_1}
    var_51 = {var_37: var_2}
    var_52 = {var_42: var_51, var_43: var_4}
    var_53 = [var_50, var_52]
    var_54 = lambda x, y: x + y
    var_55 = module_0.map_structure_zip(var_54, var_53)
    var_56 = [var_0, var_1]
    var_57 = (var_2, var_4)
    var_58 = {var_42: var_56, var_43: var_57}
    var_59 = [var_5, var_6]
    var_60 = (var_15, var_16)
    var_61 = {var_42: var_59, var_43: var_60}
    var_62 = [var_58, var_61]
    var_63 = lambda x, y: x + y
    var_64 = module_0.map_structure_zip(var_63, var_62)
    var_65 = [var_0, var_1, var_2]
    var_66 = module_0.no_map_instance(var_65)
    var_67 = [var_66, var_66]
    var_68 = lambda x, y: x + y
    var_69 = module_0.map_structure_zip(var_68, var_67)
    var_70 = [var_0, var_1, var_2]
    var_71 = lambda x, y: x + y
    var_72 = module_0.map_structure_zip(var_71, var_67)
    var_73 = [var_0, var_1, var_2]
    var_74 = [var_73]
    var_75 = lambda x: x * var_1
    var_76 = module_0.map_structure_zip(var_75, var_74)
    var_77 = [var_0, var_1]
    var_78 = [var_2, var_4]
    var_79 = [var_5, var_6]
    var_80 = [var_77, var_78, var_79]
    var_81 = lambda x, y, z: x + y + z
    var_82 = module_0.map_structure_zip(var_81, var_80)
    var_83 = []
    var_84 = []
    var_85 = [var_83, var_84]
    var_86 = lambda x, y: x + y
    var_87 = module_0.map_structure_zip(var_86, var_85)
    var_88 = {var_0, var_1}
    var_89 = {var_2, var_4}
    var_90 = [var_88, var_89]
    var_91 = lambda x, y: x + y
    var_92 = module_0.map_structure_zip(var_91, var_90)
    var_93 = (var_42, var_91)
    var_94 = (var_43, var_92)
    var_95 = [var_93, var_94]
    var_96 = (var_42, var_2)
    var_97 = (var_43, var_4)
    var_98 = [var_96, var_97]
    var_99 = lambda x, y: x * y
    var_100 = module_0.map_structure_zip(var_99, var_90)



# Parsed testcases at query #14
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = lambda x: x + var_2
    var_7 = [var_2, var_0]
    var_8 = 4
    var_9 = [var_3, var_8]
    var_10 = [var_7, var_9]
    var_11 = module_0.map_structure(var_6, var_10)
    var_12 = lambda x: x.upper()
    var_13 = 'a'
    var_14 = 'b'
    var_15 = 'c'
    var_16 = (var_13, var_14, var_15)
    var_17 = module_0.map_structure(var_12, var_16)
    var_18 = lambda x: x * var_3
    var_19 = (var_2, var_0)
    var_20 = (var_3, var_8)
    var_21 = (var_19, var_20)
    var_22 = module_0.map_structure(var_18, var_21)
    var_23 = lambda x: x * var_0
    var_24 = {var_13: var_2, var_14: var_0}
    var_25 = module_0.map_structure(var_23, var_24)
    var_26 = 10
    var_27 = lambda x: x + var_26
    var_28 = 'x'
    var_29 = {var_28: var_2}
    var_30 = 'y'
    var_31 = {var_30: var_0}
    var_32 = {var_13: var_29, var_14: var_31}
    var_33 = module_0.map_structure(var_27, var_32)
    var_34 = lambda x: x * var_0
    var_35 = {var_2, var_0, var_3}
    var_36 = module_0.map_structure(var_34, var_35)
    var_37 = [var_2, var_0, var_3]
    var_38 = 5
    var_39 = (var_8, var_38)
    var_40 = 'd'
    var_41 = 6
    var_42 = {var_40: var_41}
    var_43 = {var_13: var_37, var_14: var_39, var_15: var_42}
    var_44 = lambda x: x + var_2
    var_45 = module_0.map_structure(var_44, var_43)
    var_46 = [var_0, var_3, var_8]
    var_47 = (var_38, var_41)
    var_48 = 7
    var_49 = {var_40: var_48}
    var_50 = {var_13: var_46, var_14: var_47, var_15: var_49}
    var_51 = [var_2, var_0, var_3]
    var_52 = lambda x: x * var_0
    var_53 = [var_2, var_0, var_3]
    var_54 = module_0.no_map_instance(var_53)
    var_55 = lambda x: x * var_0
    var_56 = module_0.map_structure(var_55, var_54)
    var_57 = 'Point'
    var_58 = [var_28, var_30]
    var_59 = lambda x: x * var_0
    var_60 = lambda x: x * var_0
    var_61 = module_0.map_structure(var_60, var_38)
    assert var_61 == 10
    var_62 = lambda x: x * var_0
    var_63 = []
    var_64 = module_0.map_structure(var_62, var_63)
    var_65 = lambda x: x * var_0
    var_66 = {}
    var_67 = module_0.map_structure(var_65, var_66)
    var_68 = [var_2, var_0, var_3]
    var_69 = [var_2, var_0]
    var_70 = [var_3, var_8]
    var_71 = [var_69, var_70]
    var_72 = [var_38, var_41]
    var_73 = 8
    var_74 = [var_48, var_73]
    var_75 = [var_72, var_74]
    var_76 = [var_71, var_75]
    var_77 = lambda x: x - var_2
    var_78 = module_0.map_structure(var_77, var_76)
    var_79 = 0
    var_80 = [var_79, var_2]
    var_81 = [var_0, var_3]
    var_82 = [var_80, var_81]
    var_83 = [var_8, var_38]
    var_84 = [var_41, var_48]
    var_85 = [var_83, var_84]
    var_86 = [var_82, var_85]



# Parsed testcases at query #15
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = lambda x, y: x * y
    var_12 = [var_1, var_2]
    var_13 = [var_3, var_5]
    var_14 = [var_12, var_13]
    var_15 = [var_6, var_7]
    var_16 = 7
    var_17 = 8
    var_18 = [var_16, var_17]
    var_19 = [var_15, var_18]
    var_20 = [var_14, var_19]
    var_21 = module_0.map_structure_zip(var_11, var_20)
    var_22 = lambda x, y, z: x + y + z
    var_23 = (var_1, var_2)
    var_24 = (var_3, var_5)
    var_25 = (var_6, var_7)
    var_26 = [var_23, var_24, var_25]
    var_27 = module_0.map_structure_zip(var_22, var_26)
    var_28 = 'Point'
    var_29 = 'x'
    var_30 = 'y'
    var_31 = [var_29, var_30]
    var_32 = lambda a, b: a.x + b.x
    var_33 = 10
    var_34 = lambda x, y: x - y
    var_35 = 'a'
    var_36 = 'b'
    var_37 = 20
    var_38 = {var_35: var_33, var_36: var_37}
    var_39 = {var_35: var_6, var_36: var_17}
    var_40 = [var_38, var_39]
    var_41 = module_0.map_structure_zip(var_34, var_40)
    var_42 = [var_1, var_2]
    var_43 = (var_3, var_5)
    var_44 = {var_35: var_42, var_36: var_43}
    var_45 = [var_6, var_7]
    var_46 = (var_16, var_17)
    var_47 = {var_35: var_45, var_36: var_46}
    var_48 = lambda x, y: x * y
    var_49 = [var_44, var_47]
    var_50 = module_0.map_structure_zip(var_48, var_49)
    var_51 = [var_1, var_2, var_3]
    var_52 = module_0.no_map_instance(var_51)
    var_53 = lambda x, y: str(x) + str(y)
    var_54 = [var_52, var_52]
    var_55 = module_0.map_structure_zip(var_53, var_54)
    assert var_55 == '[1, 2, 3][1, 2, 3]'
    var_56 = [var_1, var_2]
    var_57 = lambda x, y: x + y
    var_58 = lambda x: x * var_2
    var_59 = [var_1, var_2, var_3]
    var_60 = [var_59]
    var_61 = module_0.map_structure_zip(var_58, var_60)
    var_62 = lambda x, y, z: x + y + z
    var_63 = [var_1, var_2]
    var_64 = [var_3, var_5]
    var_65 = [var_6, var_7]
    var_66 = [var_63, var_64, var_65]
    var_67 = module_0.map_structure_zip(var_62, var_66)
    var_68 = lambda x, y: x + y
    var_69 = []
    var_70 = []
    var_71 = [var_69, var_70]
    var_72 = module_0.map_structure_zip(var_68, var_71)
    var_73 = lambda x, y: x + y
    var_74 = [var_6, var_33]
    var_75 = module_0.map_structure_zip(var_73, var_74)
    assert var_75 == 15
    var_76 = lambda x, y: x + y
    var_77 = 1
    var_78 = 2
    var_79 = {var_77, var_78}
    var_80 = 3
    var_81 = 4
    var_82 = {var_80, var_81}
    var_83 = [var_79, var_82]
    var_84 = module_0.map_structure_zip(var_76, var_83)



# Parsed testcases at query #16
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = lambda x, y: x * y
    var_12 = [var_1, var_2]
    var_13 = [var_3, var_5]
    var_14 = [var_12, var_13]
    var_15 = [var_6, var_7]
    var_16 = 7
    var_17 = 8
    var_18 = [var_16, var_17]
    var_19 = [var_15, var_18]
    var_20 = [var_14, var_19]
    var_21 = module_0.map_structure_zip(var_11, var_20)
    var_22 = lambda x, y, z: x + y + z
    var_23 = (var_1, var_2)
    var_24 = (var_3, var_5)
    var_25 = (var_6, var_7)
    var_26 = [var_23, var_24, var_25]
    var_27 = module_0.map_structure_zip(var_22, var_26)
    var_28 = lambda x, y: x - y
    var_29 = 'a'
    var_30 = 'b'
    var_31 = 10
    var_32 = 20
    var_33 = {var_29: var_31, var_30: var_32}
    var_34 = {var_29: var_3, var_30: var_6}
    var_35 = [var_33, var_34]
    var_36 = module_0.map_structure_zip(var_28, var_35)
    var_37 = lambda x, y: f'{x}{y}'
    var_38 = [var_1, var_2]
    var_39 = (var_3, var_5)
    var_40 = {var_29: var_38, var_30: var_39}
    var_41 = [var_6, var_7]
    var_42 = (var_16, var_17)
    var_43 = {var_29: var_41, var_30: var_42}
    var_44 = [var_40, var_43]
    var_45 = module_0.map_structure_zip(var_37, var_44)
    var_46 = [var_1, var_2, var_3]
    var_47 = module_0.no_map_instance(var_46)
    var_48 = lambda x, y: x + y
    var_49 = [var_47, var_47]
    var_50 = module_0.map_structure_zip(var_48, var_49)
    var_51 = [var_1, var_2, var_3]
    var_52 = lambda x, y: x + y
    var_53 = lambda x: x * var_2
    var_54 = [var_1, var_2, var_3]
    var_55 = [var_54]
    var_56 = module_0.map_structure_zip(var_53, var_55)
    var_57 = lambda x, y, z: x + y + z
    var_58 = [var_1, var_2, var_3]
    var_59 = [var_5, var_6, var_7]
    var_60 = 9
    var_61 = [var_16, var_17, var_60]
    var_62 = [var_58, var_59, var_61]
    var_63 = module_0.map_structure_zip(var_57, var_62)
    var_64 = lambda x, y: x + y
    var_65 = []
    var_66 = []
    var_67 = [var_65, var_66]
    var_68 = module_0.map_structure_zip(var_64, var_67)
    var_69 = 'Point'
    var_70 = 'x'
    var_71 = 'y'
    var_72 = [var_70, var_71]
    var_73 = lambda a, b: Point(a.x + b.x, a.y + b.y)
    var_74 = lambda x, y: x + y
    var_75 = 1
    var_76 = 2
    var_77 = {var_75, var_76}
    var_78 = 3
    var_79 = 4
    var_80 = {var_78, var_79}
    var_81 = [var_77, var_80]
    var_82 = module_0.map_structure_zip(var_74, var_81)
    var_83 = [var_31, var_32]
    var_84 = module_0.no_map_instance(var_83)
    var_85 = lambda x, y: x + y
    var_86 = [var_75, var_76]
    var_87 = [var_84, var_86]
    var_88 = 30
    var_89 = 40
    var_90 = [var_88, var_89]
    var_91 = [var_77, var_79]
    var_92 = [var_90, var_91]
    var_93 = [var_87, var_92]
    var_94 = module_0.map_structure_zip(var_85, var_93)



# Parsed testcases at query #17
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = lambda x, y: x * y
    var_12 = [var_1, var_2]
    var_13 = [var_3, var_5]
    var_14 = [var_12, var_13]
    var_15 = [var_6, var_7]
    var_16 = 7
    var_17 = 8
    var_18 = [var_16, var_17]
    var_19 = [var_15, var_18]
    var_20 = [var_14, var_19]
    var_21 = module_0.map_structure_zip(var_11, var_20)
    var_22 = lambda x, y, z: x + y + z
    var_23 = (var_1, var_2)
    var_24 = (var_3, var_5)
    var_25 = (var_6, var_7)
    var_26 = [var_23, var_24, var_25]
    var_27 = module_0.map_structure_zip(var_22, var_26)
    var_28 = lambda x, y: x - y
    var_29 = 'a'
    var_30 = 'b'
    var_31 = 10
    var_32 = 20
    var_33 = {var_29: var_31, var_30: var_32}
    var_34 = {var_29: var_3, var_30: var_17}
    var_35 = [var_33, var_34]
    var_36 = module_0.map_structure_zip(var_28, var_35)
    var_37 = lambda x, y: f'{x}{y}'
    var_38 = [var_1, var_2]
    var_39 = (var_3, var_5)
    var_40 = {var_29: var_38, var_30: var_39}
    var_41 = [var_6, var_7]
    var_42 = (var_16, var_17)
    var_43 = {var_29: var_41, var_30: var_42}
    var_44 = [var_40, var_43]
    var_45 = module_0.map_structure_zip(var_37, var_44)
    var_46 = [var_1, var_2, var_3]
    var_47 = module_0.no_map_instance(var_46)
    var_48 = lambda x, y: x + y
    var_49 = [var_47, var_47]
    var_50 = module_0.map_structure_zip(var_48, var_49)
    var_51 = [var_1, var_2, var_3]
    var_52 = lambda x, y: x + y
    var_53 = lambda x: x * var_2
    var_54 = [var_1, var_2, var_3]
    var_55 = [var_54]
    var_56 = module_0.map_structure_zip(var_53, var_55)
    var_57 = lambda x, y, z: x + y + z
    var_58 = [var_1, var_2]
    var_59 = [var_3, var_5]
    var_60 = [var_6, var_7]
    var_61 = [var_58, var_59, var_60]
    var_62 = module_0.map_structure_zip(var_57, var_61)
    var_63 = lambda x, y: x + y
    var_64 = []
    var_65 = []
    var_66 = [var_64, var_65]
    var_67 = module_0.map_structure_zip(var_63, var_66)
    var_68 = lambda x, y: x + y
    var_69 = 1
    var_70 = 2
    var_71 = {var_69, var_70}
    var_72 = 3
    var_73 = 4
    var_74 = {var_72, var_73}
    var_75 = [var_71, var_74]
    var_76 = module_0.map_structure_zip(var_68, var_75)
    var_77 = 'Point'
    var_78 = 'x'
    var_79 = 'y'
    var_80 = [var_78, var_79]
    var_81 = lambda a, b: Point(a.x + b.x, a.y + b.y)



# Parsed testcases at query #18
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = lambda x, y: x * y
    var_12 = [var_1, var_2]
    var_13 = [var_3, var_5]
    var_14 = [var_12, var_13]
    var_15 = [var_6, var_7]
    var_16 = 7
    var_17 = 8
    var_18 = [var_16, var_17]
    var_19 = [var_15, var_18]
    var_20 = [var_14, var_19]
    var_21 = module_0.map_structure_zip(var_11, var_20)
    var_22 = lambda x, y, z: x + y + z
    var_23 = (var_1, var_2)
    var_24 = (var_3, var_5)
    var_25 = (var_6, var_7)
    var_26 = [var_23, var_24, var_25]
    var_27 = module_0.map_structure_zip(var_22, var_26)
    var_28 = lambda x, y: x + y
    var_29 = 'a'
    var_30 = 'b'
    var_31 = {var_29: var_1, var_30: var_2}
    var_32 = {var_29: var_3, var_30: var_5}
    var_33 = [var_31, var_32]
    var_34 = module_0.map_structure_zip(var_28, var_33)
    var_35 = lambda x, y: f'{x}{y}'
    var_36 = [var_1, var_2]
    var_37 = (var_3, var_5)
    var_38 = {var_29: var_36, var_30: var_37}
    var_39 = [var_6, var_7]
    var_40 = (var_16, var_17)
    var_41 = {var_29: var_39, var_30: var_40}
    var_42 = [var_38, var_41]
    var_43 = module_0.map_structure_zip(var_35, var_42)
    var_44 = [var_1, var_2, var_3]
    var_45 = module_0.no_map_instance(var_44)
    var_46 = lambda x, y: x + y
    var_47 = [var_45, var_45]
    var_48 = module_0.map_structure_zip(var_46, var_47)
    var_49 = [var_1, var_2, var_3]
    var_50 = lambda x, y: x + y
    var_51 = lambda x: x * var_2
    var_52 = [var_1, var_2, var_3]
    var_53 = [var_52]
    var_54 = module_0.map_structure_zip(var_51, var_53)
    var_55 = lambda x, y, z: x + y + z
    var_56 = [var_1, var_2]
    var_57 = [var_3, var_5]
    var_58 = [var_6, var_7]
    var_59 = [var_56, var_57, var_58]
    var_60 = module_0.map_structure_zip(var_55, var_59)
    var_61 = lambda x, y: x + y
    var_62 = []
    var_63 = []
    var_64 = [var_62, var_63]
    var_65 = module_0.map_structure_zip(var_61, var_64)
    var_66 = lambda x, y: x + y
    var_67 = 1
    var_68 = 2
    var_69 = {var_67, var_68}
    var_70 = 3
    var_71 = 4
    var_72 = {var_70, var_71}
    var_73 = [var_69, var_72]
    var_74 = module_0.map_structure_zip(var_66, var_73)
    var_75 = 'Point'
    var_76 = 'x'
    var_77 = 'y'
    var_78 = [var_76, var_77]
    var_79 = lambda p1, p2: Point(p1.x + p2.x, p1.y + p2.y)



# Parsed testcases at query #19
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 1
    var_3 = 3
    var_4 = [var_2, var_0, var_3]
    var_5 = module_0.map_structure(var_1, var_4)
    var_6 = lambda x: x + var_2
    var_7 = [var_2, var_0]
    var_8 = 4
    var_9 = [var_3, var_8]
    var_10 = [var_7, var_9]
    var_11 = module_0.map_structure(var_6, var_10)
    var_12 = (var_2, var_0, var_3)
    var_13 = lambda x: x * var_0
    var_14 = (var_2, var_0)
    var_15 = (var_3, var_8)
    var_16 = (var_14, var_15)
    var_17 = module_0.map_structure(var_13, var_16)
    var_18 = lambda x: x.upper()
    var_19 = 'a'
    var_20 = 'b'
    var_21 = 'hello'
    var_22 = 'world'
    var_23 = {var_19: var_21, var_20: var_22}
    var_24 = module_0.map_structure(var_18, var_23)
    var_25 = lambda x: x * var_0
    var_26 = [var_2, var_0]
    var_27 = 'c'
    var_28 = {var_27: var_3}
    var_29 = {var_19: var_26, var_20: var_28}
    var_30 = module_0.map_structure(var_25, var_29)
    var_31 = lambda x: x ** var_0
    var_32 = {var_2, var_0, var_3}
    var_33 = module_0.map_structure(var_31, var_32)
    var_34 = [var_2, var_0, var_3]
    var_35 = 5
    var_36 = (var_8, var_35)
    var_37 = 'd'
    var_38 = 6
    var_39 = {var_37: var_38}
    var_40 = {var_19: var_34, var_20: var_36, var_27: var_39}
    var_41 = 10
    var_42 = lambda x: x + var_41
    var_43 = module_0.map_structure(var_42, var_40)
    var_44 = [var_2, var_0, var_3]
    var_45 = lambda x: x * var_0
    var_46 = [var_2, var_0, var_3]
    var_47 = lambda x: x * var_0
    var_48 = 'Point'
    var_49 = 'x'
    var_50 = 'y'
    var_51 = [var_49, var_50]
    var_52 = lambda x: x * var_3
    var_53 = lambda x: x + var_35
    var_54 = module_0.map_structure(var_53, var_41)
    assert var_54 == 15
    var_55 = '!'
    var_56 = lambda x: x + var_55
    var_57 = module_0.map_structure(var_56, var_21)
    assert var_57 == 'hello!'
    var_58 = lambda x: x
    var_59 = []
    var_60 = module_0.map_structure(var_58, var_59)
    var_61 = lambda x: x
    var_62 = {}
    var_63 = module_0.map_structure(var_61, var_62)
    var_64 = lambda x: x
    var_65 = ()
    var_66 = module_0.map_structure(var_64, var_65)
    var_67 = [var_2, var_0, var_3]



# Parsed testcases at query #20
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = 5
    var_3 = module_0.map_structure(var_1, var_2)
    assert var_3 == 10
    var_4 = lambda x: x.upper()
    var_5 = 'hello'
    var_6 = module_0.map_structure(var_4, var_5)
    assert var_6 == 'HELLO'
    var_7 = 1
    var_8 = lambda x: x + var_7
    var_9 = 3
    var_10 = [var_7, var_0, var_9]
    var_11 = module_0.map_structure(var_8, var_10)
    var_12 = lambda x: x * var_0
    var_13 = [var_7, var_0]
    var_14 = 4
    var_15 = [var_9, var_14]
    var_16 = [var_13, var_15]
    var_17 = module_0.map_structure(var_12, var_16)
    var_18 = lambda x: x.upper()
    var_19 = 'a'
    var_20 = 'b'
    var_21 = 'c'
    var_22 = (var_19, var_20, var_21)
    var_23 = module_0.map_structure(var_18, var_22)
    var_24 = 10
    var_25 = lambda x: x + var_24
    var_26 = (var_7, var_0)
    var_27 = (var_9, var_14)
    var_28 = (var_26, var_27)
    var_29 = module_0.map_structure(var_25, var_28)
    var_30 = lambda x: x * var_0
    var_31 = {var_19: var_7, var_20: var_0}
    var_32 = module_0.map_structure(var_30, var_31)
    var_33 = '!'
    var_34 = lambda x: x + var_33
    var_35 = {var_20: var_5}
    var_36 = 'world'
    var_37 = {var_19: var_35, var_21: var_36}
    var_38 = module_0.map_structure(var_34, var_37)
    var_39 = lambda x: x * var_0
    var_40 = {var_7, var_0, var_9}
    var_41 = module_0.map_structure(var_39, var_40)
    var_42 = {var_20: var_9}
    var_43 = [var_7, var_0, var_42]
    var_44 = (var_14, var_2)
    var_45 = {var_19: var_43, var_21: var_44}
    var_46 = lambda x: x * var_0
    var_47 = module_0.map_structure(var_46, var_45)
    var_48 = 6
    var_49 = {var_20: var_48}
    var_50 = [var_0, var_14, var_49]
    var_51 = 8
    var_52 = (var_51, var_24)
    var_53 = {var_19: var_50, var_21: var_52}
    var_54 = 'Point'
    var_55 = 'x'
    var_56 = 'y'
    var_57 = [var_55, var_56]
    var_58 = lambda x: x * var_0
    var_59 = [var_7, var_0, var_9]
    var_60 = module_0.no_map_instance(var_59)
    var_61 = lambda x: x * var_0
    var_62 = module_0.map_structure(var_61, var_60)
    var_63 = [var_7, var_0, var_9]
    var_64 = lambda x: x * var_0
    var_65 = lambda x: type(x).__name__
    var_66 = 42
    var_67 = module_0.map_structure(var_65, var_66)
    assert var_67 == 'int'
    var_68 = lambda x: type(x).__name__
    var_69 = 'test'
    var_70 = module_0.map_structure(var_68, var_69)
    assert var_70 == 'str'
    var_71 = lambda x: x
    var_72 = []
    var_73 = module_0.map_structure(var_71, var_72)
    var_74 = lambda x: x
    var_75 = {}
    var_76 = module_0.map_structure(var_74, var_75)
    var_77 = lambda x: x
    var_78 = set()
    var_79 = module_0.map_structure(var_77, var_78)
    var_80 = set()
    var_81 = lambda x: x
    var_82 = ()
    var_83 = module_0.map_structure(var_81, var_82)
    var_84 = 'list'
    var_85 = 'tuple'
    var_86 = 'set'
    var_87 = {var_19: var_0, var_20: var_9}
    var_88 = (var_14, var_2)
    var_89 = [var_7, var_87, var_88]
    var_90 = 7
    var_91 = [var_90, var_51]
    var_92 = (var_48, var_91)
    var_93 = 9
    var_94 = {var_93, var_24}
    var_95 = {var_84: var_89, var_85: var_92, var_86: var_94}
    var_96 = 100
    var_97 = lambda x: x + var_96
    var_98 = module_0.map_structure(var_97, var_95)
    var_99 = 101
    var_100 = 102
    var_101 = 103
    var_102 = {var_19: var_100, var_20: var_101}
    var_103 = 104
    var_104 = 105
    var_105 = (var_103, var_104)
    var_106 = [var_99, var_102, var_105]
    var_107 = 106
    var_108 = 107
    var_109 = 108
    var_110 = [var_108, var_109]
    var_111 = (var_107, var_110)
    var_112 = 109
    var_113 = 110
    var_114 = {var_112, var_113}
    var_115 = {var_84: var_106, var_85: var_111, var_86: var_114}



# Parsed testcases at query #21
#--------------------------


import flutes.structure as module_0

def test_case_0():
    var_0 = lambda x, y: x + y
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = [var_4, var_8]
    var_10 = module_0.map_structure_zip(var_0, var_9)
    var_11 = lambda x, y: x * y
    var_12 = [var_1, var_2]
    var_13 = [var_3, var_5]
    var_14 = [var_12, var_13]
    var_15 = [var_6, var_7]
    var_16 = 7
    var_17 = 8
    var_18 = [var_16, var_17]
    var_19 = [var_15, var_18]
    var_20 = [var_14, var_19]
    var_21 = module_0.map_structure_zip(var_11, var_20)
    var_22 = lambda x, y, z: x + y + z
    var_23 = (var_1, var_2)
    var_24 = (var_3, var_5)
    var_25 = (var_6, var_7)
    var_26 = [var_23, var_24, var_25]
    var_27 = module_0.map_structure_zip(var_22, var_26)
    var_28 = lambda x, y: f'{x}{y}'
    var_29 = 'a'
    var_30 = 'b'
    var_31 = {var_29: var_1, var_30: var_2}
    var_32 = {var_29: var_3, var_30: var_5}
    var_33 = [var_31, var_32]
    var_34 = module_0.map_structure_zip(var_28, var_33)
    var_35 = lambda x, y: x + y
    var_36 = [var_1, var_2]
    var_37 = (var_3, var_5)
    var_38 = {var_29: var_36, var_30: var_37}
    var_39 = [var_6, var_7]
    var_40 = (var_16, var_17)
    var_41 = {var_29: var_39, var_30: var_40}
    var_42 = [var_38, var_41]
    var_43 = module_0.map_structure_zip(var_35, var_42)
    var_44 = [var_1, var_2, var_3]
    var_45 = module_0.no_map_instance(var_44)
    var_46 = lambda x, y: x + y
    var_47 = [var_45, var_45]
    var_48 = module_0.map_structure_zip(var_46, var_47)
    var_49 = [var_1, var_2, var_3]
    var_50 = lambda x, y: x + y
    var_51 = 'Point'
    var_52 = 'x'
    var_53 = 'y'
    var_54 = [var_52, var_53]
    var_55 = lambda a, b: Point(a.x + b.x, a.y + b.y)
    var_56 = lambda x: x * var_2
    var_57 = [var_1, var_2, var_3]
    var_58 = [var_57]
    var_59 = module_0.map_structure_zip(var_56, var_58)
    var_60 = lambda x, y, z: x + y + z
    var_61 = [var_1, var_2]
    var_62 = [var_3, var_5]
    var_63 = [var_6, var_7]
    var_64 = [var_61, var_62, var_63]
    var_65 = module_0.map_structure_zip(var_60, var_64)
    var_66 = lambda x, y: x + y
    var_67 = 1
    var_68 = 2
    var_69 = {var_67, var_68}
    var_70 = 3
    var_71 = 4
    var_72 = {var_70, var_71}
    var_73 = [var_69, var_72]
    var_74 = module_0.map_structure_zip(var_66, var_73)
    var_75 = lambda x, y: x + y
    var_76 = []
    var_77 = []
    var_78 = [var_76, var_77]
    var_79 = module_0.map_structure_zip(var_75, var_78)
    var_80 = lambda x, y: x + y
    var_81 = {}
    var_82 = {}
    var_83 = [var_81, var_82]
    var_84 = module_0.map_structure_zip(var_80, var_83)



