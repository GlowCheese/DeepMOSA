####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import pyrsistent._plist as module_0


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = module_0.plist(var_2)
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]
    var_7 = module_0.plist(var_6)
    var_8 = 4
    var_9 = 5
    var_10 = None
    var_11 = -1
    var_12 = 'invalid'
    var_13 = 0
    var_14 = var_8 + var_12
    var_15 = var_8 - var_12
    var_16 = module_0.plist()
    var_17 = module_0.plist()
    var_18 = 3.14
    var_19 = True
    var_20 = '5'
    var_21 = 5
    var_22 = [var_21]
    var_23 = 5
    var_24 = (var_23,)
    var_25 = 'maxlen'
    var_26 = 5
    var_27 = {var_25: var_26}
    var_28 = 5
    var_29 = {var_28}
    var_30 = 5
    var_31 = 0
    var_32 = complex(var_30, var_31)
    var_33 = 0
    var_34 = 10
    var_35 = 6
    var_36 = var_34 ** var_35
    var_37 = 0
    var_38 = -1
    var_39 = 0
    var_40 = 'invalid'
    var_41 = -1
    var_42 = 'invalid'
    var_43 = 0
    var_44 = -1
    var_45 = 0
    var_46 = 'invalid'
    var_47 = -1
    var_48 = 'invalid'
    var_49 = -1
    var_50 = 'invalid'
    var_51 = -1



# Parsed testcases at query #2
#--------------------------


import pyrsistent._pdeque as module_0


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = repr(var_4)
    assert var_5 == 'pdeque([1, 2, 3])'
    var_6 = [var_0, var_1, var_2]
    var_7 = module_0.pdeque(var_6, var_2)
    var_8 = repr(var_7)
    assert var_8 == 'pdeque([1, 2, 3], maxlen=3)'
    var_9 = []
    var_10 = 0
    var_11 = module_0.pdeque(var_9, var_10)
    var_12 = repr(var_11)
    assert var_12 == 'pdeque([], maxlen=0)'
    var_13 = []
    var_14 = module_0.pdeque(var_13, var_0)
    var_15 = repr(var_14)
    assert var_15 == 'pdeque([], maxlen=1)'
    var_16 = [var_0]
    var_17 = module_0.pdeque(var_16, var_0)
    var_18 = repr(var_17)
    assert var_18 == 'pdeque([1], maxlen=1)'
    var_19 = [var_0, var_1]
    var_20 = module_0.pdeque(var_19, var_0)
    var_21 = repr(var_20)
    assert var_21 == 'pdeque([2], maxlen=1)'
    var_22 = [var_0, var_1]
    var_23 = module_0.pdeque(var_22, var_1)
    var_24 = repr(var_23)
    assert var_24 == 'pdeque([1, 2], maxlen=2)'
    var_25 = [var_0, var_1, var_2]
    var_26 = module_0.pdeque(var_25, var_1)
    var_27 = repr(var_26)
    assert var_27 == 'pdeque([2, 3], maxlen=2)'
    var_28 = [var_0, var_1, var_2]
    var_29 = module_0.pdeque(var_28, var_2)
    var_30 = repr(var_29)
    assert var_30 == 'pdeque([1, 2, 3], maxlen=3)'
    var_31 = [var_0, var_1, var_2]
    var_32 = 4
    var_33 = module_0.pdeque(var_31, var_32)
    var_34 = repr(var_33)
    assert var_34 == 'pdeque([1, 2, 3], maxlen=4)'
    var_35 = [var_0, var_1, var_2]
    var_36 = 5
    var_37 = module_0.pdeque(var_35, var_36)
    var_38 = repr(var_37)
    assert var_38 == 'pdeque([1, 2, 3], maxlen=5)'
    var_39 = [var_0, var_1, var_2]
    var_40 = 6
    var_41 = module_0.pdeque(var_39, var_40)
    var_42 = repr(var_41)
    assert var_42 == 'pdeque([1, 2, 3], maxlen=6)'
    var_43 = [var_0, var_1, var_2]
    var_44 = 7
    var_45 = module_0.pdeque(var_43, var_44)
    var_46 = repr(var_45)
    assert var_46 == 'pdeque([1, 2, 3], maxlen=7)'
    var_47 = [var_0, var_1, var_2]
    var_48 = 8
    var_49 = module_0.pdeque(var_47, var_48)
    var_50 = repr(var_49)
    assert var_50 == 'pdeque([1, 2, 3], maxlen=8)'
    var_51 = [var_0, var_1, var_2]
    var_52 = 9
    var_53 = module_0.pdeque(var_51, var_52)
    var_54 = repr(var_53)
    assert var_54 == 'pdeque([1, 2, 3], maxlen=9)'
    var_55 = [var_0, var_1, var_2]
    var_56 = 10
    var_57 = module_0.pdeque(var_55, var_56)
    var_58 = repr(var_57)
    assert var_58 == 'pdeque([1, 2, 3], maxlen=10)'
    var_59 = [var_0, var_1, var_2]
    var_60 = 11
    var_61 = module_0.pdeque(var_59, var_60)
    var_62 = repr(var_61)
    assert var_62 == 'pdeque([1, 2, 3], maxlen=11)'
    var_63 = [var_0, var_1, var_2]
    var_64 = 12
    var_65 = module_0.pdeque(var_63, var_64)
    var_66 = repr(var_65)
    assert var_66 == 'pdeque([1, 2, 3], maxlen=12)'
    var_67 = [var_0, var_1, var_2]
    var_68 = 13
    var_69 = module_0.pdeque(var_67, var_68)
    var_70 = repr(var_69)
    assert var_70 == 'pdeque([1, 2, 3], maxlen=13)'
    var_71 = [var_0, var_1, var_2]
    var_72 = 14
    var_73 = module_0.pdeque(var_71, var_72)
    var_74 = repr(var_73)
    assert var_74 == 'pdeque([1, 2, 3], maxlen=14)'
    var_75 = [var_0, var_1, var_2]
    var_76 = 15
    var_77 = module_0.pdeque(var_75, var_76)
    var_78 = repr(var_77)
    assert var_78 == 'pdeque([1, 2, 3], maxlen=15)'
    var_79 = [var_0, var_1, var_2]
    var_80 = 16
    var_81 = module_0.pdeque(var_79, var_80)
    var_82 = repr(var_81)
    assert var_82 == 'pdeque([1, 2, 3], maxlen=16)'
    var_83 = [var_0, var_1, var_2]
    var_84 = 17
    var_85 = module_0.pdeque(var_83, var_84)
    var_86 = repr(var_85)
    assert var_86 == 'pdeque([1, 2, 3], maxlen=17)'
    var_87 = [var_0, var_1, var_2]
    var_88 = 18
    var_89 = module_0.pdeque(var_87, var_88)
    var_90 = repr(var_89)
    assert var_90 == 'pdeque([1, 2, 3], maxlen=18)'
    var_91 = [var_0, var_1, var_2]
    var_92 = 19
    var_93 = module_0.pdeque(var_91, var_92)
    var_94 = repr(var_93)
    assert var_94 == 'pdeque([1, 2, 3], maxlen=19)'
    var_95 = [var_0, var_1, var_2]
    var_96 = 20
    var_97 = module_0.pdeque(var_95, var_96)
    var_98 = repr(var_97)
    assert var_98 == 'pdeque([1, 2, 3], maxlen=20)'
    var_99 = [var_0, var_1, var_2]
    var_100 = 21
    var_101 = module_0.pdeque(var_99, var_100)
    var_102 = repr(var_101)
    assert var_102 == 'pdeque([1, 2, 3], maxlen=21)'
    var_103 = [var_0, var_1, var_2]
    var_104 = 22
    var_105 = module_0.pdeque(var_103, var_104)
    var_106 = repr(var_105)
    assert var_106 == 'pdeque([1, 2, 3], maxlen=22)'
    var_107 = [var_0, var_1, var_2]
    var_108 = 23
    var_109 = module_0.pdeque(var_107, var_108)
    var_110 = repr(var_109)
    assert var_110 == 'pdeque([1, 2, 3], maxlen=23)'
    var_111 = [var_0, var_1, var_2]
    var_112 = 24
    var_113 = module_0.pdeque(var_111, var_112)
    var_114 = repr(var_113)
    assert var_114 == 'pdeque([1, 2, 3], maxlen=24)'
    var_115 = [var_0, var_1, var_2]
    var_116 = 25
    var_117 = module_0.pdeque(var_115, var_116)
    var_118 = repr(var_117)
    assert var_118 == 'pdeque([1, 2, 3], maxlen=25)'
    var_119 = [var_0, var_1, var_2]
    var_120 = 26
    var_121 = module_0.pdeque(var_119, var_120)
    var_122 = repr(var_121)
    assert var_122 == 'pdeque([1, 2, 3], maxlen=26)'
    var_123 = [var_0, var_1, var_2]
    var_124 = 27
    var_125 = module_0.pdeque(var_123, var_124)
    var_126 = repr(var_125)
    assert var_126 == 'pdeque([1, 2, 3], maxlen=27)'
    var_127 = [var_0, var_1, var_2]
    var_128 = 28
    var_129 = module_0.pdeque(var_127, var_128)
    var_130 = repr(var_129)
    assert var_130 == 'pdeque([1, 2, 3], maxlen=28)'
    var_131 = [var_0, var_1, var_2]
    var_132 = 29
    var_133 = module_0.pdeque(var_131, var_132)
    var_134 = repr(var_133)
    assert var_134 == 'pdeque([1, 2, 3], maxlen=29)'
    var_135 = [var_0, var_1, var_2]
    var_136 = 30
    var_137 = module_0.pdeque(var_135, var_136)
    var_138 = repr(var_137)
    assert var_138 == 'pdeque([1, 2, 3], maxlen=30)'
    var_139 = [var_0, var_1, var_2]
    var_140 = 31
    var_141 = module_0.pdeque(var_139, var_140)
    var_142 = repr(var_141)
    assert var_142 == 'pdeque([1, 2, 3], maxlen=31)'
    var_143 = [var_0, var_1, var_2]
    var_144 = 32
    var_145 = module_0.pdeque(var_143, var_144)
    var_146 = repr(var_145)
    assert var_146 == 'pdeque([1, 2, 3], maxlen=32)'
    var_147 = [var_0, var_1, var_2]
    var_148 = 33
    var_149 = module_0.pdeque(var_147, var_148)
    var_150 = repr(var_149)
    assert var_150 == 'pdeque([1, 2, 3], maxlen=33)'
    var_151 = [var_0, var_1, var_2]
    var_152 = 34
    var_153 = module_0.pdeque(var_151, var_152)
    var_154 = repr(var_153)
    assert var_154 == 'pdeque([1, 2, 3], maxlen=34)'
    var_155 = [var_0, var_1, var_2]
    var_156 = 35
    var_157 = module_0.pdeque(var_155, var_156)
    var_158 = repr(var_157)
    assert var_158 == 'pdeque([1, 2, 3], maxlen=35)'
    var_159 = [var_0, var_1, var_2]
    var_160 = 36
    var_161 = module_0.pdeque(var_159, var_160)
    var_162 = repr(var_161)
    assert var_162 == 'pdeque([1, 2, 3], maxlen=36)'
    var_163 = [var_0, var_1, var_2]
    var_164 = 37
    var_165 = module_0.pdeque(var_163, var_164)
    var_166 = repr(var_165)
    assert var_166 == 'pdeque([1, 2, 3], maxlen=37)'
    var_167 = [var_0, var_1, var_2]
    var_168 = 38
    var_169 = module_0.pdeque(var_167, var_168)
    var_170 = repr(var_169)
    assert var_170 == 'pdeque([1, 2, 3], maxlen=38)'
    var_171 = [var_0, var_1, var_2]
    var_172 = 39
    var_173 = module_0.pdeque(var_171, var_172)
    var_174 = repr(var_173)
    assert var_174 == 'pdeque([1, 2, 3], maxlen=39)'
    var_175 = [var_0, var_1, var_2]
    var_176 = 40
    var_177 = module_0.pdeque(var_175, var_176)
    var_178 = repr(var_177)
    assert var_178 == 'pdeque([1, 2, 3], maxlen=40)'
    var_179 = [var_0, var_1, var_2]
    var_180 = 41
    var_181 = module_0.pdeque(var_179, var_180)
    var_182 = repr(var_181)
    assert var_182 == 'pdeque([1, 2, 3], maxlen=41)'
    var_183 = [var_0, var_1, var_2]
    var_184 = 42
    var_185 = module_0.pdeque(var_183, var_184)
    var_186 = repr(var_185)
    assert var_186 == 'pdeque([1, 2, 3], maxlen=42)'
    var_187 = var_185



# Parsed testcases at query #3
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.pdeque(var_5)
    var_7 = [var_0, var_1, var_3, var_4]
    var_8 = module_0.pdeque(var_7)
    var_9 = [var_0, var_1, var_2, var_3, var_4]
    var_10 = module_0.pdeque(var_9)
    var_11 = [var_0, var_1, var_2, var_3]
    var_12 = module_0.pdeque(var_11)
    var_13 = [var_0, var_1, var_2, var_1, var_0]
    var_14 = module_0.pdeque(var_13)
    var_15 = [var_0, var_2, var_1, var_0]
    var_16 = module_0.pdeque(var_15)
    var_17 = [var_0, var_1, var_2]
    var_18 = module_0.pdeque(var_17)
    var_19 = 4
    var_20 = []
    var_21 = module_0.pdeque(var_20)
    var_22 = 1
    var_23 = [var_22, var_1, var_2, var_3, var_4]
    var_24 = module_0.pdeque(var_23, var_2)
    var_25 = [var_22, var_1, var_3, var_4]
    var_26 = module_0.pdeque(var_25, var_2)
    var_27 = [var_22, var_1, var_2, var_3, var_4]
    var_28 = module_0.pdeque(var_27, var_2)
    var_29 = [var_22, var_2, var_3, var_4]
    var_30 = module_0.pdeque(var_29, var_2)
    var_31 = [var_22, var_1, var_2, var_3, var_4]
    var_32 = module_0.pdeque(var_31, var_2)
    var_33 = [var_1, var_2, var_3, var_4]
    var_34 = module_0.pdeque(var_33, var_2)
    var_35 = [var_22, var_1, var_2, var_3, var_4]
    var_36 = module_0.pdeque(var_35, var_2)
    var_37 = [var_22, var_1, var_2, var_3]
    var_38 = module_0.pdeque(var_37, var_2)
    var_39 = [var_22, var_1, var_2, var_3, var_4]
    var_40 = module_0.pdeque(var_39, var_1)
    var_41 = [var_1, var_3, var_4]
    var_42 = module_0.pdeque(var_41, var_1)
    var_43 = 'All test cases passed'
    var_44 = print(var_43)



# Parsed testcases at query #4
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.pdeque(var_5)
    var_7 = [var_1, var_2]
    var_8 = module_0.pdeque(var_7)
    var_9 = [var_0, var_1, var_2]
    var_10 = module_0.pdeque(var_9)
    var_11 = [var_2, var_3, var_4]
    var_12 = module_0.pdeque(var_11)
    var_13 = [var_0, var_2, var_4]
    var_14 = module_0.pdeque(var_13)
    var_15 = [var_2, var_3]
    var_16 = module_0.pdeque(var_15)
    var_17 = [var_2, var_3, var_4]
    var_18 = module_0.pdeque(var_17)
    var_19 = [var_0, var_1, var_2]
    var_20 = module_0.pdeque(var_19)
    var_21 = [var_1, var_3]
    var_22 = module_0.pdeque(var_21)
    var_23 = [var_0, var_3]
    var_24 = module_0.pdeque(var_23)
    var_25 = []
    var_26 = module_0.pdeque(var_25)
    var_27 = []
    var_28 = module_0.pdeque(var_27)
    var_29 = 10
    var_30 = var_6[var_29]
    var_31 = 'invalid'
    var_32 = var_6[var_31]
    var_33 = [var_31, var_32, var_2, var_3, var_4]
    var_34 = module_0.pdeque(var_33, var_2)
    var_35 = [var_2, var_3]
    var_36 = module_0.pdeque(var_35, var_2)
    var_37 = []
    var_38 = module_0.pdeque(var_37)
    var_39 = 0
    var_40 = var_38[var_39]
    var_41 = 42
    var_42 = [var_41]
    var_43 = module_0.pdeque(var_42)
    var_44 = [var_41]
    var_45 = module_0.pdeque(var_44)
    var_46 = 1000
    var_47 = range(var_46)
    var_48 = module_0.pdeque(var_47)
    var_49 = 200
    var_50 = 300
    var_51 = range(var_49, var_50)
    var_52 = module_0.pdeque(var_51)
    var_53 = [var_4, var_3, var_2, var_40, var_39]
    var_54 = module_0.pdeque(var_53)
    var_55 = [var_3, var_2, var_40]
    var_56 = module_0.pdeque(var_55)
    var_57 = [var_4, var_3, var_2]
    var_58 = module_0.pdeque(var_57)
    var_59 = [var_4, var_2, var_39]
    var_60 = module_0.pdeque(var_59)
    var_61 = [var_3, var_40]
    var_62 = module_0.pdeque(var_61)
    var_63 = [var_39, var_40, var_2, var_3, var_4]
    var_64 = module_0.pdeque(var_63)
    var_65 = [var_2, var_3, var_4]
    var_66 = module_0.pdeque(var_65)
    var_67 = [var_39, var_40, var_2]
    var_68 = module_0.pdeque(var_67)
    var_69 = [var_39, var_40, var_2, var_3, var_4]
    var_70 = module_0.pdeque(var_69, var_3)
    var_71 = [var_40, var_3]
    var_72 = module_0.pdeque(var_71, var_3)
    var_73 = []
    var_74 = module_0.pdeque(var_73)
    var_75 = []
    var_76 = module_0.pdeque(var_75)
    var_77 = []
    var_78 = module_0.pdeque(var_77)
    var_79 = [var_2]
    var_80 = module_0.pdeque(var_79)
    var_81 = [var_3]
    var_82 = module_0.pdeque(var_81)
    var_83 = [var_39, var_40, var_2, var_3, var_4]
    var_84 = module_0.pdeque(var_83)
    var_85 = [var_39, var_40, var_2, var_3, var_4]
    var_86 = module_0.pdeque(var_85)
    var_87 = [var_39, var_40, var_2, var_3, var_4]
    var_88 = module_0.pdeque(var_87)
    var_89 = [var_2, var_3]
    var_90 = module_0.pdeque(var_89)
    var_91 = [var_39, var_40, var_2]
    var_92 = module_0.pdeque(var_91)
    var_93 = [var_39, var_40, var_2]
    var_94 = module_0.pdeque(var_93)
    var_95 = [var_2, var_3, var_4]
    var_96 = module_0.pdeque(var_95)
    var_97 = [var_39, var_40, var_2, var_3, var_4]
    var_98 = module_0.pdeque(var_97)
    var_99 = []
    var_100 = module_0.pdeque(var_99)
    var_101 = []
    var_102 = module_0.pdeque(var_101)
    var_103 = [var_4, var_2, var_39]
    var_104 = module_0.pdeque(var_103)
    var_105 = [var_3, var_40]
    var_106 = module_0.pdeque(var_105)
    var_107 = [var_4, var_2]
    var_108 = module_0.pdeque(var_107)
    var_109 = [var_3, var_40]
    var_110 = module_0.pdeque(var_109)
    var_111 = [var_4, var_3, var_2, var_40, var_39]
    var_112 = module_0.pdeque(var_111)
    var_113 = [var_4, var_3, var_2, var_40, var_39]
    var_114 = module_0.pdeque(var_113)
    var_115 = [var_3, var_2, var_40]
    var_116 = module_0.pdeque(var_115)
    var_117 = [var_3, var_2, var_40]
    var_118 = module_0.pdeque(var_117)
    var_119 = [var_4, var_2]
    var_120 = module_0.pdeque(var_119)
    var_121 = [var_4, var_2, var_39]
    var_122 = module_0.pdeque(var_121)
    var_123 = [var_4, var_40]
    var_124 = module_0.pdeque(var_123)
    var_125 = [var_4, var_40]
    var_126 = module_0.pdeque(var_125)
    var_127 = []
    var_128 = module_0.pdeque(var_127)
    var_129 = []
    var_130 = module_0.pdeque(var_129)
    var_131 = []
    var_132 = module_0.pdeque(var_131)
    var_133 = []
    var_134 = module_0.pdeque(var_133)
    var_135 = [var_41]
    var_136 = module_0.pdeque(var_135)
    var_137 = [var_41]
    var_138 = module_0.pdeque(var_137)
    var_139 = [var_41]
    var_140 = module_0.pdeque(var_139)
    var_141 = [var_41]
    var_142 = module_0.pdeque(var_141)
    var_143 = []
    var_144 = module_0.pdeque(var_143)
    var_145 = []
    var_146 = module_0.pdeque(var_145)
    var_147 = 10
    var_148 = 20
    var_149 = [var_147, var_148]
    var_150 = module_0.pdeque(var_149)
    var_151 = [var_147, var_148]
    var_152 = module_0.pdeque(var_151)
    var_153 = [var_147]
    var_154 = module_0.pdeque(var_153)
    var_155 = [var_148]
    var_156 = module_0.pdeque(var_155)
    var_157 = [var_148]
    var_158 = module_0.pdeque(var_157)
    var_159 = [var_147]
    var_160 = module_0.pdeque(var_159)
    var_161 = [var_148, var_147]
    var_162 = module_0.pdeque(var_161)
    var_163 = 0
    var_164 = var_6[::var_163]
    var_165 = [var_163]
    var_166 = module_0.pdeque(var_165)
    var_167 = [var_164]
    var_168 = module_0.pdeque(var_167)



# Parsed testcases at query #5
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.pdeque(var_5)
    var_7 = [var_1, var_2, var_3]
    var_8 = module_0.pdeque(var_7)
    var_9 = [var_0, var_1, var_2]
    var_10 = module_0.pdeque(var_9)
    var_11 = [var_2, var_3, var_4]
    var_12 = module_0.pdeque(var_11)
    var_13 = [var_0, var_2, var_4]
    var_14 = module_0.pdeque(var_13)
    var_15 = [var_1, var_2, var_3]
    var_16 = module_0.pdeque(var_15)
    var_17 = [var_2, var_3, var_4]
    var_18 = module_0.pdeque(var_17)
    var_19 = [var_0, var_1, var_2]
    var_20 = module_0.pdeque(var_19)
    var_21 = [var_1, var_3]
    var_22 = module_0.pdeque(var_21)
    var_23 = [var_0, var_3]
    var_24 = module_0.pdeque(var_23)
    var_25 = [var_2, var_4]
    var_26 = module_0.pdeque(var_25)
    var_27 = 10
    var_28 = var_6[var_27]
    var_29 = module_0.pdeque()
    var_30 = 0
    var_31 = var_29[var_30]
    var_32 = [var_30, var_31, var_2, var_3, var_4]
    var_33 = module_0.pdeque(var_32, var_2)
    var_34 = [var_2, var_3]
    var_35 = module_0.pdeque(var_34, var_2)
    var_36 = 'All tests passed!'
    var_37 = print(var_36)



# Parsed testcases at query #6
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.pdeque(var_5)
    var_7 = [var_1, var_2, var_3]
    var_8 = module_0.pdeque(var_7)
    var_9 = [var_0, var_1, var_2]
    var_10 = module_0.pdeque(var_9)
    var_11 = [var_2, var_3, var_4]
    var_12 = module_0.pdeque(var_11)
    var_13 = [var_0, var_2, var_4]
    var_14 = module_0.pdeque(var_13)
    var_15 = [var_4, var_3, var_2, var_1, var_0]
    var_16 = module_0.pdeque(var_15)
    var_17 = [var_0, var_1, var_2, var_3, var_4]
    var_18 = module_0.pdeque(var_17, var_2)
    var_19 = [var_2, var_3]
    var_20 = module_0.pdeque(var_19, var_2)
    var_21 = module_0.pdeque()
    var_22 = 0
    var_23 = var_21[var_22]
    var_24 = 10
    var_25 = var_6[var_24]
    var_26 = -10
    var_27 = var_6[var_26]
    var_28 = 'invalid'
    var_29 = var_6[var_28]
    var_30 = 'All tests passed!'
    var_31 = print(var_30)



# Parsed testcases at query #7
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.pdeque(var_5)
    var_7 = [var_1, var_2, var_3]
    var_8 = module_0.pdeque(var_7)
    var_9 = [var_0, var_1, var_2]
    var_10 = module_0.pdeque(var_9)
    var_11 = [var_2, var_3, var_4]
    var_12 = module_0.pdeque(var_11)
    var_13 = [var_0, var_2, var_4]
    var_14 = module_0.pdeque(var_13)
    var_15 = [var_1, var_2, var_3]
    var_16 = module_0.pdeque(var_15)
    var_17 = [var_2, var_3, var_4]
    var_18 = module_0.pdeque(var_17)
    var_19 = [var_0, var_1, var_2]
    var_20 = module_0.pdeque(var_19)
    var_21 = [var_1, var_3]
    var_22 = module_0.pdeque(var_21)
    var_23 = [var_0, var_3]
    var_24 = module_0.pdeque(var_23)
    var_25 = [var_2, var_4]
    var_26 = module_0.pdeque(var_25)
    var_27 = []
    var_28 = module_0.pdeque(var_27)
    var_29 = []
    var_30 = module_0.pdeque(var_29)
    var_31 = 10
    var_32 = var_6[var_31]
    var_33 = -10
    var_34 = var_6[var_33]
    var_35 = 'invalid'
    var_36 = var_6[var_35]
    var_37 = [var_35, var_36, var_2, var_3, var_4]
    var_38 = module_0.pdeque(var_37, var_2)
    var_39 = [var_2, var_3]
    var_40 = module_0.pdeque(var_39, var_2)
    var_41 = 42
    var_42 = [var_41]
    var_43 = module_0.pdeque(var_42)
    var_44 = []
    var_45 = module_0.pdeque(var_44)
    var_46 = 0
    var_47 = var_45[var_46]
    var_48 = 'All tests passed!'
    var_49 = print(var_48)



# Parsed testcases at query #8
#--------------------------



def test_case_0():
    var_0 = 2
    var_1 = 1
    var_2 = [var_0, var_1, var_0]
    var_3 = module_0.pdeque(var_2)
    var_4 = [var_1, var_0]
    var_5 = module_0.pdeque(var_4)
    var_6 = 3
    var_7 = [var_1, var_0, var_6]
    var_8 = module_0.pdeque(var_7)
    var_9 = [var_1, var_0]
    var_10 = module_0.pdeque(var_9)
    var_11 = [var_1, var_0, var_6]
    var_12 = module_0.pdeque(var_11)
    var_13 = 4
    var_14 = []
    var_15 = module_0.pdeque(var_14)
    var_16 = 1
    var_17 = [var_1]
    var_18 = module_0.pdeque(var_17)
    var_19 = []
    var_20 = module_0.pdeque(var_19)
    var_21 = [var_1, var_16, var_1, var_6, var_1]
    var_22 = module_0.pdeque(var_21)
    var_23 = [var_16, var_1, var_6, var_1]
    var_24 = module_0.pdeque(var_23)
    var_25 = [var_1, var_16, var_6]
    var_26 = module_0.pdeque(var_25, var_16)
    var_27 = [var_6]
    var_28 = module_0.pdeque(var_27, var_16)
    var_29 = [var_1, var_16, var_6]
    var_30 = module_0.pdeque(var_29, var_16)
    var_31 = [var_6]
    var_32 = module_0.pdeque(var_31, var_16)
    var_33 = [var_1, var_16, var_6]
    var_34 = module_0.pdeque(var_33, var_16)
    var_35 = [var_16]
    var_36 = module_0.pdeque(var_35, var_16)
    var_37 = 4
    var_38 = [var_1, var_16, var_6, var_37]
    var_39 = module_0.pdeque(var_38, var_6)
    var_40 = [var_1, var_6, var_37]
    var_41 = module_0.pdeque(var_40, var_6)
    var_42 = 'All test cases passed'
    var_43 = print(var_42)



# Parsed testcases at query #9
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.pdeque(var_5)
    var_7 = [var_1, var_2]
    var_8 = module_0.pdeque(var_7)
    var_9 = [var_0, var_1, var_2]
    var_10 = module_0.pdeque(var_9)
    var_11 = [var_2, var_3, var_4]
    var_12 = module_0.pdeque(var_11)
    var_13 = [var_0, var_2, var_4]
    var_14 = module_0.pdeque(var_13)
    var_15 = [var_4, var_3, var_2, var_1, var_0]
    var_16 = module_0.pdeque(var_15)
    var_17 = [var_0, var_1, var_2, var_3, var_4]
    var_18 = module_0.pdeque(var_17, var_2)
    var_19 = module_0.pdeque()
    var_20 = 0
    var_21 = var_19[var_20]
    var_22 = [var_20, var_21, var_2]
    var_23 = module_0.pdeque(var_22)
    var_24 = 5
    var_25 = var_23[var_24]
    var_26 = -5
    var_27 = var_23[var_26]
    var_28 = 'a'
    var_29 = var_23[var_28]
    var_30 = [var_28, var_29, var_2, var_3, var_4]
    var_31 = module_0.pdeque(var_30)
    var_32 = [var_29, var_3]
    var_33 = module_0.pdeque(var_32)
    var_34 = [var_28, var_3]
    var_35 = module_0.pdeque(var_34)
    var_36 = [var_4, var_2, var_28]
    var_37 = module_0.pdeque(var_36)
    var_38 = [var_28, var_29, var_2, var_3, var_4]
    var_39 = module_0.pdeque(var_38, var_3)
    var_40 = [var_29, var_2]
    var_41 = module_0.pdeque(var_40)
    var_42 = [var_29, var_2]
    var_43 = module_0.pdeque(var_42)
    var_44 = [var_3, var_4]
    var_45 = module_0.pdeque(var_44)
    var_46 = [var_29, var_3]
    var_47 = module_0.pdeque(var_46)
    var_48 = [var_4, var_3, var_2, var_29]
    var_49 = module_0.pdeque(var_48)
    var_50 = [var_28, var_29, var_2, var_3, var_4]
    var_51 = module_0.pdeque(var_50, var_2)
    var_52 = [var_3, var_4]
    var_53 = module_0.pdeque(var_52)
    var_54 = [var_2, var_3]
    var_55 = module_0.pdeque(var_54)
    var_56 = [var_2, var_3]
    var_57 = module_0.pdeque(var_56)
    var_58 = [var_28, var_29, var_2, var_3, var_4]
    var_59 = module_0.pdeque(var_58, var_3)
    var_60 = [var_29, var_3]
    var_61 = module_0.pdeque(var_60)
    var_62 = [var_4, var_2]
    var_63 = module_0.pdeque(var_62)
    var_64 = [var_28, var_29, var_2, var_3, var_4]
    var_65 = module_0.pdeque(var_64, var_2)
    var_66 = [var_4, var_3, var_2]
    var_67 = module_0.pdeque(var_66)
    var_68 = [var_4, var_2]
    var_69 = module_0.pdeque(var_68)
    var_70 = [var_28, var_29, var_2, var_3, var_4]
    var_71 = module_0.pdeque(var_70, var_29)
    var_72 = [var_3]
    var_73 = module_0.pdeque(var_72)
    var_74 = [var_4]
    var_75 = module_0.pdeque(var_74)
    var_76 = 0
    var_77 = var_71[::var_76]
    var_78 = 0
    var_79 = var_71[::var_78]
    var_80 = [var_78, var_79, var_2, var_3, var_4]
    var_81 = module_0.pdeque(var_80, var_2)
    var_82 = [var_4, var_3, var_2]
    var_83 = module_0.pdeque(var_82)
    var_84 = [var_4, var_2]
    var_85 = module_0.pdeque(var_84)
    var_86 = [var_4]
    var_87 = module_0.pdeque(var_86)
    var_88 = [var_4]
    var_89 = module_0.pdeque(var_88)
    var_90 = [var_4]
    var_91 = module_0.pdeque(var_90)
    var_92 = [var_4]
    var_93 = module_0.pdeque(var_92)
    var_94 = [var_4]
    var_95 = module_0.pdeque(var_94)
    var_96 = [var_4]
    var_97 = module_0.pdeque(var_96)
    var_98 = [var_4]
    var_99 = module_0.pdeque(var_98)
    var_100 = [var_4]
    var_101 = module_0.pdeque(var_100)
    var_102 = [var_4]
    var_103 = module_0.pdeque(var_102)
    var_104 = [var_4]
    var_105 = module_0.pdeque(var_104)
    var_106 = [var_4]
    var_107 = module_0.pdeque(var_106)
    var_108 = [var_4]
    var_109 = module_0.pdeque(var_108)
    var_110 = [var_4]
    var_111 = module_0.pdeque(var_110)
    var_112 = [var_4]
    var_113 = module_0.pdeque(var_112)
    var_114 = [var_4]
    var_115 = module_0.pdeque(var_114)
    var_116 = [var_4]
    var_117 = module_0.pdeque(var_116)
    var_118 = [var_4]
    var_119 = module_0.pdeque(var_118)
    var_120 = [var_4]
    var_121 = module_0.pdeque(var_120)
    var_122 = [var_4]
    var_123 = module_0.pdeque(var_122)
    var_124 = [var_4]
    var_125 = module_0.pdeque(var_124)
    var_126 = [var_4]
    var_127 = module_0.pdeque(var_126)
    var_128 = [var_4]
    var_129 = module_0.pdeque(var_128)
    var_130 = [var_4]
    var_131 = module_0.pdeque(var_130)
    var_132 = [var_4]
    var_133 = module_0.pdeque(var_132)
    var_134 = [var_4]
    var_135 = module_0.pdeque(var_134)
    var_136 = [var_4]
    var_137 = module_0.pdeque(var_136)
    var_138 = [var_4]
    var_139 = module_0.pdeque(var_138)
    var_140 = [var_4]
    var_141 = module_0.pdeque(var_140)
    var_142 = [var_4]
    var_143 = module_0.pdeque(var_142)
    var_144 = [var_4]
    var_145 = module_0.pdeque(var_144)
    var_146 = [var_4]
    var_147 = module_0.pdeque(var_146)
    var_148 = [var_4]
    var_149 = module_0.pdeque(var_148)
    var_150 = [var_4]
    var_151 = module_0.pdeque(var_150)
    var_152 = [var_4]
    var_153 = module_0.pdeque(var_152)
    var_154 = [var_4]
    var_155 = module_0.pdeque(var_154)
    var_156 = [var_4]
    var_157 = module_0.pdeque(var_156)
    var_158 = [var_4]
    var_159 = module_0.pdeque(var_158)



# Parsed testcases at query #10
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.pdeque(var_5)
    var_7 = [var_0, var_1, var_3, var_4]
    var_8 = module_0.pdeque(var_7)
    var_9 = [var_0, var_1, var_2, var_3, var_4]
    var_10 = module_0.pdeque(var_9)
    var_11 = [var_0, var_1, var_2, var_3]
    var_12 = module_0.pdeque(var_11)
    var_13 = [var_0, var_1, var_2, var_3, var_4]
    var_14 = module_0.pdeque(var_13)
    var_15 = 6
    var_16 = []
    var_17 = module_0.pdeque(var_16)
    var_18 = 1
    var_19 = [var_18, var_1, var_2, var_1, var_18]
    var_20 = module_0.pdeque(var_19)
    var_21 = [var_18, var_2, var_1, var_18]
    var_22 = module_0.pdeque(var_21)
    var_23 = [var_18, var_1, var_2, var_3, var_4]
    var_24 = module_0.pdeque(var_23, var_3)
    var_25 = [var_18, var_1, var_3, var_4]
    var_26 = module_0.pdeque(var_25, var_3)
    var_27 = [var_18, var_1, var_2, var_3, var_4]
    var_28 = module_0.pdeque(var_27, var_2)
    var_29 = [var_18, var_1, var_3, var_4]
    var_30 = module_0.pdeque(var_29, var_2)
    var_31 = [var_18, var_1, var_2, var_3, var_4]
    var_32 = module_0.pdeque(var_31, var_2)
    var_33 = [var_18, var_1, var_2, var_3]
    var_34 = module_0.pdeque(var_33, var_2)
    var_35 = [var_18, var_1, var_2, var_3, var_4]
    var_36 = module_0.pdeque(var_35, var_3)
    var_37 = [var_18, var_2, var_3, var_4]
    var_38 = module_0.pdeque(var_37, var_3)
    var_39 = [var_18, var_1, var_2, var_3, var_4]
    var_40 = module_0.pdeque(var_39, var_3)
    var_41 = [var_18, var_1, var_2, var_4]
    var_42 = module_0.pdeque(var_41, var_3)
    var_43 = 'All test cases passed'
    var_44 = print(var_43)



# Parsed testcases at query #11
#--------------------------



def test_case_0():
    var_0 = 2
    var_1 = 1
    var_2 = [var_0, var_1, var_0]
    var_3 = module_0.pdeque(var_2)
    var_4 = [var_1, var_0]
    var_5 = module_0.pdeque(var_4)
    var_6 = 3
    var_7 = [var_1, var_0, var_6]
    var_8 = module_0.pdeque(var_7)
    var_9 = [var_1, var_0]
    var_10 = module_0.pdeque(var_9)
    var_11 = [var_1, var_0, var_6]
    var_12 = module_0.pdeque(var_11)
    var_13 = 4
    var_14 = []
    var_15 = module_0.pdeque(var_14)
    var_16 = 1
    var_17 = [var_1]
    var_18 = module_0.pdeque(var_17)
    var_19 = []
    var_20 = module_0.pdeque(var_19)
    var_21 = [var_1, var_16, var_1, var_6, var_1]
    var_22 = module_0.pdeque(var_21)
    var_23 = [var_16, var_1, var_6, var_1]
    var_24 = module_0.pdeque(var_23)
    var_25 = [var_1, var_16, var_6]
    var_26 = module_0.pdeque(var_25, var_6)
    var_27 = [var_1, var_6]
    var_28 = module_0.pdeque(var_27, var_6)
    var_29 = [var_1, var_16, var_6]
    var_30 = module_0.pdeque(var_29, var_6)
    var_31 = [var_16, var_6]
    var_32 = module_0.pdeque(var_31, var_6)
    var_33 = [var_1, var_16, var_6]
    var_34 = module_0.pdeque(var_33, var_6)
    var_35 = [var_1, var_16]
    var_36 = module_0.pdeque(var_35, var_6)
    var_37 = 4
    var_38 = [var_1, var_16, var_6, var_37]
    var_39 = module_0.pdeque(var_38, var_37)
    var_40 = [var_1, var_6, var_37]
    var_41 = module_0.pdeque(var_40, var_37)
    var_42 = 'All test cases passed'
    var_43 = print(var_42)



# Parsed testcases at query #12
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.pdeque(var_5)
    var_7 = [var_1, var_2, var_3]
    var_8 = module_0.pdeque(var_7)
    var_9 = [var_0, var_1, var_2]
    var_10 = module_0.pdeque(var_9)
    var_11 = [var_2, var_3, var_4]
    var_12 = module_0.pdeque(var_11)
    var_13 = [var_0, var_2, var_4]
    var_14 = module_0.pdeque(var_13)
    var_15 = [var_4, var_3, var_2, var_1, var_0]
    var_16 = module_0.pdeque(var_15)
    var_17 = [var_1, var_2, var_3]
    var_18 = module_0.pdeque(var_17)
    var_19 = [var_2, var_3, var_4]
    var_20 = module_0.pdeque(var_19)
    var_21 = [var_0, var_1, var_2]
    var_22 = module_0.pdeque(var_21)
    var_23 = [var_1, var_3]
    var_24 = module_0.pdeque(var_23)
    var_25 = [var_0, var_3]
    var_26 = module_0.pdeque(var_25)
    var_27 = [var_2, var_4]
    var_28 = module_0.pdeque(var_27)
    var_29 = []
    var_30 = module_0.pdeque(var_29)
    var_31 = []
    var_32 = module_0.pdeque(var_31)
    var_33 = 10
    var_34 = var_6[var_33]
    var_35 = -10
    var_36 = var_6[var_35]
    var_37 = 'invalid'
    var_38 = var_6[var_37]
    var_39 = [var_37, var_38, var_2, var_3, var_4]
    var_40 = module_0.pdeque(var_39, var_2)
    var_41 = [var_2, var_3]
    var_42 = module_0.pdeque(var_41, var_2)
    var_43 = 42
    var_44 = [var_43]
    var_45 = module_0.pdeque(var_44)
    var_46 = [var_43]
    var_47 = module_0.pdeque(var_46)
    var_48 = []
    var_49 = module_0.pdeque(var_48)
    var_50 = 0
    var_51 = var_49[var_50]
    var_52 = []
    var_53 = module_0.pdeque(var_52)
    var_54 = 'All tests passed!'
    var_55 = print(var_54)



# Parsed testcases at query #13
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.pdeque(var_5)
    var_7 = [var_1, var_2]
    var_8 = module_0.pdeque(var_7)
    var_9 = [var_0, var_1, var_2]
    var_10 = module_0.pdeque(var_9)
    var_11 = [var_2, var_3, var_4]
    var_12 = module_0.pdeque(var_11)
    var_13 = [var_0, var_2, var_4]
    var_14 = module_0.pdeque(var_13)
    var_15 = [var_4, var_3, var_2, var_1, var_0]
    var_16 = module_0.pdeque(var_15)
    var_17 = [var_0, var_1, var_2, var_3, var_4]
    var_18 = module_0.pdeque(var_17, var_2)
    var_19 = [var_2, var_3]
    var_20 = module_0.pdeque(var_19, var_2)
    var_21 = module_0.pdeque()
    var_22 = 0
    var_23 = var_21[var_22]
    var_24 = 10
    var_25 = var_6[var_24]
    var_26 = -10
    var_27 = var_6[var_26]
    var_28 = 'invalid'
    var_29 = var_6[var_28]
    var_30 = 'All tests passed for __getitem__'
    var_31 = print(var_30)



# Parsed testcases at query #14
#--------------------------



def test_case_0():
    var_0 = module_0.pdeque()
    var_1 = module_0.pdeque()
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.pdeque(var_5)
    var_7 = [var_2, var_3, var_4]
    var_8 = module_0.pdeque(var_7)
    var_9 = [var_2, var_3, var_4]
    var_10 = module_0.pdeque(var_9)
    var_11 = 4
    var_12 = 5
    var_13 = 6
    var_14 = [var_11, var_12, var_13]
    var_15 = module_0.pdeque(var_14)
    var_16 = [var_2, var_3, var_4]
    var_17 = module_0.pdeque(var_16)
    var_18 = [var_2, var_3, var_4]
    var_19 = module_0.pdeque(var_18)
    var_20 = [var_2, var_3]
    var_21 = module_0.pdeque(var_20)
    var_22 = [var_2, var_3, var_4]
    var_23 = module_0.pdeque(var_22)
    var_24 = [var_4, var_3, var_2]
    var_25 = module_0.pdeque(var_24)
    var_26 = [var_2, var_3, var_4]
    var_27 = module_0.pdeque(var_26, var_12)
    var_28 = [var_2, var_3, var_4]
    var_29 = 10
    var_30 = module_0.pdeque(var_28, var_29)
    var_31 = [var_2, var_3, var_4]
    var_32 = module_0.pdeque(var_31, var_12)
    var_33 = [var_2, var_3, var_4]
    var_34 = module_0.pdeque(var_33)
    var_35 = [var_2, var_3, var_4]
    var_36 = module_0.pdeque(var_35, var_12)
    var_37 = [var_2, var_3, var_4]
    var_38 = module_0.pdeque(var_37, var_4)
    var_39 = [var_2, var_3, var_4]
    var_40 = module_0.pdeque(var_39, var_4)
    var_41 = [var_2, var_3, var_4]
    var_42 = module_0.pdeque(var_41, var_12)
    var_43 = [var_2, var_3, var_4]
    var_44 = module_0.pdeque(var_43, var_4)
    var_45 = [var_2, var_3, var_4]
    var_46 = module_0.pdeque(var_45, var_4)
    var_47 = []
    var_48 = module_0.pdeque(var_47, var_4)
    var_49 = []
    var_50 = module_0.pdeque(var_49, var_12)
    var_51 = []
    var_52 = module_0.pdeque(var_51, var_4)
    var_53 = []
    var_54 = module_0.pdeque(var_53, var_4)
    var_55 = []
    var_56 = module_0.pdeque(var_55, var_4)
    var_57 = []
    var_58 = module_0.pdeque(var_57)
    var_59 = [var_2, var_3, var_4]
    var_60 = module_0.pdeque(var_59, var_12)
    var_61 = [var_2, var_3, var_4]
    var_62 = module_0.pdeque(var_61, var_4)
    var_63 = [var_2, var_3, var_4]
    var_64 = module_0.pdeque(var_63, var_12)
    var_65 = [var_2, var_3, var_4]
    var_66 = module_0.pdeque(var_65, var_4)
    var_67 = [var_2, var_3, var_4]
    var_68 = module_0.pdeque(var_67, var_12)
    var_69 = [var_2, var_3, var_4]
    var_70 = module_0.pdeque(var_69, var_29)
    var_71 = []
    var_72 = module_0.pdeque(var_71, var_12)
    var_73 = []
    var_74 = module_0.pdeque(var_73, var_29)
    var_75 = []
    var_76 = module_0.pdeque(var_75, var_12)
    var_77 = []
    var_78 = module_0.pdeque(var_77, var_12)
    var_79 = [var_2, var_3, var_4]
    var_80 = module_0.pdeque(var_79, var_12)
    var_81 = [var_2, var_3, var_4]
    var_82 = module_0.pdeque(var_81)
    var_83 = []
    var_84 = module_0.pdeque(var_83, var_12)
    var_85 = []
    var_86 = module_0.pdeque(var_85)
    var_87 = []
    var_88 = module_0.pdeque(var_87, var_12)
    var_89 = []
    var_90 = module_0.pdeque(var_89, var_12)
    var_91 = [var_2, var_3, var_4]
    var_92 = module_0.pdeque(var_91, var_12)
    var_93 = [var_2, var_3, var_4]
    var_94 = module_0.pdeque(var_93, var_4)
    var_95 = [var_2, var_3, var_4]
    var_96 = module_0.pdeque(var_95, var_12)
    var_97 = [var_2, var_3, var_4]
    var_98 = module_0.pdeque(var_97, var_4)
    var_99 = []
    var_100 = module_0.pdeque(var_99, var_12)
    var_101 = [var_2, var_3, var_4]
    var_102 = module_0.pdeque(var_101, var_4)
    var_103 = []
    var_104 = module_0.pdeque(var_103, var_12)
    var_105 = [var_2, var_3, var_4]
    var_106 = module_0.pdeque(var_105, var_4)
    var_107 = []
    var_108 = module_0.pdeque(var_107, var_12)
    var_109 = []
    var_110 = module_0.pdeque(var_109, var_4)
    var_111 = []
    var_112 = module_0.pdeque(var_111, var_12)
    var_113 = []
    var_114 = module_0.pdeque(var_113, var_29)
    var_115 = []
    var_116 = module_0.pdeque(var_115, var_12)
    var_117 = []
    var_118 = module_0.pdeque(var_117, var_12)
    var_119 = []
    var_120 = module_0.pdeque(var_119, var_12)
    var_121 = []
    var_122 = module_0.pdeque(var_121, var_29)



####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.pdeque(var_5)
    var_7 = [var_3, var_4, var_0, var_1, var_2]
    var_8 = module_0.pdeque(var_7)
    var_9 = [var_0, var_1, var_2, var_3, var_4]
    var_10 = module_0.pdeque(var_9)
    var_11 = -2
    var_12 = [var_2, var_3, var_4, var_0, var_1]
    var_13 = module_0.pdeque(var_12)
    var_14 = [var_0, var_1, var_2, var_3, var_4]
    var_15 = module_0.pdeque(var_14)
    var_16 = 0
    var_17 = [var_0, var_1, var_2, var_3, var_4]
    var_18 = module_0.pdeque(var_17)
    var_19 = [var_0, var_1, var_2, var_3, var_4]
    var_20 = module_0.pdeque(var_19)
    var_21 = 7
    var_22 = [var_3, var_4, var_0, var_1, var_2]
    var_23 = module_0.pdeque(var_22)
    var_24 = [var_0, var_1, var_2, var_3, var_4]
    var_25 = module_0.pdeque(var_24)
    var_26 = -7
    var_27 = [var_2, var_3, var_4, var_0, var_1]
    var_28 = module_0.pdeque(var_27)
    var_29 = []
    var_30 = module_0.pdeque(var_29)
    var_31 = []
    var_32 = module_0.pdeque(var_31)
    var_33 = [var_0]
    var_34 = module_0.pdeque(var_33)
    var_35 = [var_0]
    var_36 = module_0.pdeque(var_35)
    var_37 = [var_0, var_1]
    var_38 = module_0.pdeque(var_37)
    var_39 = [var_1, var_0]
    var_40 = module_0.pdeque(var_39)
    var_41 = [var_0, var_1]
    var_42 = module_0.pdeque(var_41)
    var_43 = -1
    var_44 = [var_1, var_0]
    var_45 = module_0.pdeque(var_44)
    var_46 = [var_0, var_1, var_2, var_3, var_4]
    var_47 = module_0.pdeque(var_46, var_3)
    var_48 = [var_3, var_4, var_0, var_1]
    var_49 = module_0.pdeque(var_48, var_3)
    var_50 = [var_0, var_1, var_2, var_3, var_4]
    var_51 = module_0.pdeque(var_50, var_3)
    var_52 = -2
    var_53 = [var_2, var_3, var_4, var_0]
    var_54 = module_0.pdeque(var_53, var_3)
    var_55 = [var_0, var_1, var_2, var_3, var_4]
    var_56 = module_0.pdeque(var_55, var_3)
    var_57 = [var_3, var_4, var_0, var_1]
    var_58 = module_0.pdeque(var_57, var_3)
    var_59 = [var_0, var_1, var_2, var_3, var_4]
    var_60 = module_0.pdeque(var_59, var_3)
    var_61 = -7
    var_62 = [var_2, var_3, var_4, var_0]
    var_63 = module_0.pdeque(var_62, var_3)
    var_64 = [var_0, var_1, var_2, var_3, var_4]
    var_65 = module_0.pdeque(var_64, var_3)
    var_66 = [var_1, var_2, var_3, var_4]
    var_67 = module_0.pdeque(var_66, var_3)
    var_68 = [var_0]
    var_69 = module_0.pdeque(var_68, var_0)
    var_70 = [var_0]
    var_71 = module_0.pdeque(var_70, var_0)
    var_72 = [var_0, var_1]
    var_73 = module_0.pdeque(var_72, var_1)
    var_74 = [var_1, var_0]
    var_75 = module_0.pdeque(var_74, var_1)
    var_76 = [var_0, var_1]
    var_77 = module_0.pdeque(var_76, var_1)
    var_78 = -1
    var_79 = [var_1, var_0]
    var_80 = module_0.pdeque(var_79, var_1)
    var_81 = []
    var_82 = module_0.pdeque(var_81, var_2)
    var_83 = []
    var_84 = module_0.pdeque(var_83, var_2)
    var_85 = [var_0, var_1, var_2, var_3, var_4]
    var_86 = module_0.pdeque(var_85, var_4)
    var_87 = [var_0, var_1, var_2, var_3, var_4]
    var_88 = module_0.pdeque(var_87, var_4)
    var_89 = [var_0, var_1, var_2, var_3, var_4]
    var_90 = module_0.pdeque(var_89, var_4)
    var_91 = -5
    var_92 = [var_0, var_1, var_2, var_3, var_4]
    var_93 = module_0.pdeque(var_92, var_4)
    var_94 = [var_0, var_1, var_2, var_3, var_4]
    var_95 = module_0.pdeque(var_94, var_2)
    var_96 = [var_2, var_3, var_4]
    var_97 = module_0.pdeque(var_96, var_2)
    var_98 = [var_0, var_1, var_2, var_3, var_4]
    var_99 = module_0.pdeque(var_98, var_2)
    var_100 = -3
    var_101 = [var_2, var_3, var_4]
    var_102 = module_0.pdeque(var_101, var_2)
    var_103 = [var_0, var_1, var_2, var_3, var_4]
    var_104 = module_0.pdeque(var_103, var_2)
    var_105 = [var_2, var_3, var_4]
    var_106 = module_0.pdeque(var_105, var_2)
    var_107 = [var_0, var_1, var_2, var_3, var_4]
    var_108 = module_0.pdeque(var_107, var_2)
    var_109 = -7
    var_110 = [var_2, var_3, var_4]
    var_111 = module_0.pdeque(var_110, var_2)
    var_112 = [var_0, var_1, var_2, var_3, var_4]
    var_113 = module_0.pdeque(var_112, var_2)
    var_114 = [var_2, var_3, var_4]
    var_115 = module_0.pdeque(var_114, var_2)
    var_116 = [var_0, var_1, var_2, var_3, var_4]
    var_117 = module_0.pdeque(var_116, var_2)
    var_118 = 0
    var_119 = [var_2, var_3, var_4]
    var_120 = module_0.pdeque(var_119, var_2)
    var_121 = [var_0, var_1, var_2, var_3, var_4]
    var_122 = module_0.pdeque(var_121, var_2)
    var_123 = [var_4, var_2, var_3]
    var_124 = module_0.pdeque(var_123, var_2)
    var_125 = [var_0, var_1, var_2, var_3, var_4]
    var_126 = module_0.pdeque(var_125, var_2)
    var_127 = -1
    var_128 = [var_3, var_4, var_2]
    var_129 = module_0.pdeque(var_128, var_2)
    var_130 = [var_0, var_1, var_2, var_3, var_4]
    var_131 = module_0.pdeque(var_130, var_2)
    var_132 = [var_3, var_4, var_2]
    var_133 = module_0.pdeque(var_132, var_2)
    var_134 = [var_0, var_1, var_2, var_3, var_4]
    var_135 = module_0.pdeque(var_134, var_2)
    var_136 = -2
    var_137 = [var_4, var_2, var_3]
    var_138 = module_0.pdeque(var_137, var_2)



# Parsed testcases at query #2
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.pdeque(var_5)
    var_7 = [var_1, var_2]
    var_8 = module_0.pdeque(var_7)
    var_9 = [var_0, var_1, var_2]
    var_10 = module_0.pdeque(var_9)
    var_11 = [var_2, var_3, var_4]
    var_12 = module_0.pdeque(var_11)
    var_13 = [var_0, var_2, var_4]
    var_14 = module_0.pdeque(var_13)
    var_15 = [var_4, var_3, var_2, var_1, var_0]
    var_16 = module_0.pdeque(var_15)
    var_17 = [var_0, var_1, var_2, var_3, var_4]
    var_18 = module_0.pdeque(var_17, var_2)
    var_19 = module_0.pdeque()
    var_20 = 0
    var_21 = var_19[var_20]
    var_22 = -1
    var_23 = var_19[var_22]
    var_24 = 42
    var_25 = [var_24]
    var_26 = module_0.pdeque(var_25)
    var_27 = 1000
    var_28 = range(var_27)
    var_29 = module_0.pdeque(var_28)
    var_30 = [var_22, var_23, var_2, var_3, var_4]
    var_31 = module_0.pdeque(var_30, var_3)
    var_32 = [var_23, var_2]
    var_33 = module_0.pdeque(var_32, var_3)
    var_34 = [var_22, var_23, var_2]
    var_35 = module_0.pdeque(var_34, var_3)
    var_36 = [var_2, var_3, var_4]
    var_37 = module_0.pdeque(var_36, var_3)
    var_38 = [var_22, var_23, var_2, var_3, var_4]
    var_39 = module_0.pdeque(var_38)
    var_40 = [var_23, var_3]
    var_41 = module_0.pdeque(var_40)
    var_42 = [var_22, var_3]
    var_43 = module_0.pdeque(var_42)
    var_44 = [var_4, var_2]
    var_45 = module_0.pdeque(var_44)
    var_46 = [var_22, var_23, var_2]
    var_47 = module_0.pdeque(var_46)
    var_48 = 'invalid'
    var_49 = var_47[var_48]
    var_50 = 2.5
    var_51 = var_47[var_50]
    var_52 = [var_50, var_51, var_2]
    var_53 = module_0.pdeque(var_52)
    var_54 = 10
    var_55 = var_53[var_54]
    var_56 = -10
    var_57 = var_53[var_56]
    var_58 = [var_56, var_57, var_2, var_3, var_4]
    var_59 = module_0.pdeque(var_58, var_2)
    var_60 = var_59[var_56:var_2]
    var_61 = [var_57, var_2]
    var_62 = module_0.pdeque(var_61, var_2)
    var_63 = [var_56, var_57, var_2, var_3, var_4]
    var_64 = module_0.pdeque(var_63)
    var_65 = [var_2, var_3]
    var_66 = module_0.pdeque(var_65)
    var_67 = [var_57, var_2, var_3, var_4]
    var_68 = module_0.pdeque(var_67)
    var_69 = [var_56, var_57, var_2]
    var_70 = module_0.pdeque(var_69)
    var_71 = [var_56, var_57, var_2, var_3, var_4]
    var_72 = module_0.pdeque(var_71, var_3)
    var_73 = [var_56, var_2, var_4]
    var_74 = module_0.pdeque(var_73, var_3)
    var_75 = [var_56, var_57, var_2, var_3, var_4]
    var_76 = module_0.pdeque(var_75)
    var_77 = var_76[var_56:var_3]
    var_78 = [var_56, var_57, var_2, var_3, var_4]
    var_79 = module_0.pdeque(var_78)
    var_80 = [var_56, var_57, var_2, var_3, var_4]
    var_81 = module_0.pdeque(var_80)
    var_82 = [var_2]
    var_83 = module_0.pdeque(var_82)
    var_84 = [var_3]
    var_85 = module_0.pdeque(var_84)
    var_86 = [var_56, var_57, var_2, var_3, var_4]
    var_87 = module_0.pdeque(var_86)
    var_88 = []
    var_89 = module_0.pdeque(var_88)
    var_90 = []
    var_91 = module_0.pdeque(var_90)
    var_92 = []
    var_93 = module_0.pdeque(var_92)
    var_94 = [var_56, var_57, var_2, var_3, var_4]
    var_95 = module_0.pdeque(var_94)
    var_96 = [var_56, var_57, var_2]
    var_97 = module_0.pdeque(var_96)
    var_98 = [var_2, var_3, var_4]
    var_99 = module_0.pdeque(var_98)
    var_100 = [var_56, var_57, var_2, var_3, var_4]
    var_101 = module_0.pdeque(var_100)
    var_102 = [var_56, var_57, var_2, var_3, var_4]
    var_103 = module_0.pdeque(var_102)
    var_104 = var_103[var_56:var_3]
    var_105 = [var_57, var_2, var_3]
    var_106 = module_0.pdeque(var_105)
    var_107 = [var_56, var_57, var_2, var_3, var_4]
    var_108 = module_0.pdeque(var_107)
    var_109 = var_108[::var_57]
    var_110 = [var_56, var_2, var_4]
    var_111 = module_0.pdeque(var_110)
    var_112 = 100
    var_113 = range(var_112)
    var_114 = module_0.pdeque(var_113)
    var_115 = 0
    var_116 = 10
    var_117 = range(var_115, var_112, var_116)
    var_118 = module_0.pdeque(var_117)
    var_119 = [var_56, var_57, var_2, var_3, var_4]
    var_120 = module_0.pdeque(var_119)
    var_121 = [var_4, var_3, var_2, var_57, var_56]
    var_122 = module_0.pdeque(var_121)
    var_123 = [var_4, var_2, var_56]
    var_124 = module_0.pdeque(var_123)
    var_125 = [var_4, var_3, var_2]
    var_126 = module_0.pdeque(var_125)
    var_127 = [var_4, var_2]
    var_128 = module_0.pdeque(var_127)
    var_129 = 6
    var_130 = 7
    var_131 = 8
    var_132 = 9
    var_133 = [var_56, var_57, var_2, var_3, var_4, var_129, var_130, var_131, var_132, var_116]
    var_134 = module_0.pdeque(var_133)
    var_135 = [var_131, var_130, var_129, var_4, var_3]
    var_136 = module_0.pdeque(var_135)
    var_137 = [var_2, var_3, var_4, var_129, var_130]
    var_138 = module_0.pdeque(var_137)
    var_139 = [var_132, var_130, var_4]
    var_140 = module_0.pdeque(var_139)
    var_141 = [var_56, var_57, var_2, var_3, var_4]
    var_142 = module_0.pdeque(var_141, var_2)
    var_143 = [var_56, var_57, var_2, var_3, var_4]
    var_144 = module_0.pdeque(var_143, var_2)
    var_145 = [var_2, var_3, var_4]
    var_146 = module_0.pdeque(var_145, var_2)
    var_147 = [var_3, var_4]
    var_148 = module_0.pdeque(var_147, var_2)
    var_149 = [var_2, var_3]
    var_150 = module_0.pdeque(var_149, var_2)
    var_151 = [var_56, var_57, var_2, var_3, var_4]
    var_152 = module_0.pdeque(var_151, var_2)
    var_153 = 3
    var_154 = var_152[var_153]
    var_155 = -4
    var_156 = var_152[var_155]



# Parsed testcases at query #3
#--------------------------



def test_case_0():
    var_0 = 2
    var_1 = 1
    var_2 = [var_0, var_1, var_0]
    var_3 = module_0.pdeque(var_2)
    var_4 = [var_1, var_0]
    var_5 = module_0.pdeque(var_4)
    var_6 = 3
    var_7 = [var_1, var_0, var_6]
    var_8 = module_0.pdeque(var_7)
    var_9 = [var_1, var_0]
    var_10 = module_0.pdeque(var_9)
    var_11 = [var_1, var_0, var_6]
    var_12 = module_0.pdeque(var_11)
    var_13 = 4
    var_14 = []
    var_15 = module_0.pdeque(var_14)
    var_16 = 1
    var_17 = [var_1]
    var_18 = module_0.pdeque(var_17)
    var_19 = []
    var_20 = module_0.pdeque(var_19)
    var_21 = [var_1, var_16, var_1, var_6, var_1]
    var_22 = module_0.pdeque(var_21)
    var_23 = [var_16, var_1, var_6, var_1]
    var_24 = module_0.pdeque(var_23)
    var_25 = [var_1, var_16, var_6]
    var_26 = module_0.pdeque(var_25, var_6)
    var_27 = [var_1, var_6]
    var_28 = module_0.pdeque(var_27, var_6)
    var_29 = [var_1, var_16, var_6]
    var_30 = module_0.pdeque(var_29, var_6)
    var_31 = 4
    var_32 = [var_1, var_31, var_6]
    var_33 = module_0.pdeque(var_32, var_6)
    var_34 = [var_31, var_6]
    var_35 = module_0.pdeque(var_34, var_6)
    var_36 = [var_1, var_31, var_6]
    var_37 = module_0.pdeque(var_36, var_6)
    var_38 = [var_1, var_31]
    var_39 = module_0.pdeque(var_38, var_6)
    var_40 = 'All test cases passed'
    var_41 = print(var_40)



# Parsed testcases at query #4
#--------------------------



def test_case_0():
    var_0 = []
    var_1 = module_0.pdeque(var_0)
    var_2 = []
    var_3 = module_0.pdeque(var_2)
    var_4 = 1
    var_5 = [var_4]
    var_6 = module_0.pdeque(var_5)
    var_7 = []
    var_8 = module_0.pdeque(var_7)
    var_9 = 2
    var_10 = 3
    var_11 = [var_4, var_9, var_10]
    var_12 = module_0.pdeque(var_11)
    var_13 = [var_4, var_9]
    var_14 = module_0.pdeque(var_13)
    var_15 = [var_4]
    var_16 = module_0.pdeque(var_15)
    var_17 = [var_4, var_9, var_10]
    var_18 = module_0.pdeque(var_17)
    var_19 = -1
    var_20 = [var_9, var_10]
    var_21 = module_0.pdeque(var_20)
    var_22 = [var_4, var_9]
    var_23 = module_0.pdeque(var_22)
    var_24 = 5
    var_25 = []
    var_26 = module_0.pdeque(var_25)
    var_27 = [var_4, var_9, var_10]
    var_28 = module_0.pdeque(var_27, var_10)
    var_29 = [var_4, var_9]
    var_30 = module_0.pdeque(var_29, var_10)
    var_31 = []
    var_32 = 0
    var_33 = module_0.pdeque(var_31, var_32)
    var_34 = []
    var_35 = module_0.pdeque(var_34, var_32)



# Parsed testcases at query #5
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.pdeque(var_5)
    var_7 = [var_2, var_3, var_4]
    var_8 = module_0.pdeque(var_7)
    var_9 = [var_0, var_1, var_2, var_3, var_4]
    var_10 = module_0.pdeque(var_9)
    var_11 = -2
    var_12 = [var_0, var_1, var_2]
    var_13 = module_0.pdeque(var_12)
    var_14 = [var_0, var_1, var_2]
    var_15 = module_0.pdeque(var_14)
    var_16 = []
    var_17 = module_0.pdeque(var_16)
    var_18 = []
    var_19 = module_0.pdeque(var_18)
    var_20 = []
    var_21 = module_0.pdeque(var_20)
    var_22 = [var_0, var_1, var_2]
    var_23 = module_0.pdeque(var_22, var_2)
    var_24 = [var_1, var_2]
    var_25 = module_0.pdeque(var_24, var_2)
    var_26 = [var_0, var_1]
    var_27 = module_0.pdeque(var_26, var_1)
    var_28 = []
    var_29 = module_0.pdeque(var_28, var_1)
    var_30 = 'All tests passed!'
    var_31 = print(var_30)



####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = [var_2, var_0, var_1]
    var_6 = module_0.pdeque(var_5)
    var_7 = [var_0, var_1, var_2]
    var_8 = module_0.pdeque(var_7)
    var_9 = -2
    var_10 = [var_2, var_0, var_1]
    var_11 = module_0.pdeque(var_10)
    var_12 = [var_0, var_1, var_2]
    var_13 = module_0.pdeque(var_12)
    var_14 = [var_0, var_1, var_2]
    var_15 = module_0.pdeque(var_14)
    var_16 = [var_0, var_1, var_2]
    var_17 = module_0.pdeque(var_16)
    var_18 = 4
    var_19 = [var_2, var_0, var_1]
    var_20 = module_0.pdeque(var_19)
    var_21 = [var_0, var_1, var_2]
    var_22 = module_0.pdeque(var_21)
    var_23 = -4
    var_24 = [var_1, var_2, var_0]
    var_25 = module_0.pdeque(var_24)
    var_26 = []
    var_27 = module_0.pdeque(var_26)
    var_28 = []
    var_29 = module_0.pdeque(var_28)
    var_30 = [var_0]
    var_31 = module_0.pdeque(var_30)
    var_32 = [var_0]
    var_33 = module_0.pdeque(var_32)
    var_34 = [var_0, var_1]
    var_35 = module_0.pdeque(var_34)
    var_36 = [var_1, var_0]
    var_37 = module_0.pdeque(var_36)
    var_38 = [var_0, var_1]
    var_39 = module_0.pdeque(var_38)
    var_40 = -1
    var_41 = [var_1, var_0]
    var_42 = module_0.pdeque(var_41)
    var_43 = [var_0, var_1, var_2]
    var_44 = module_0.pdeque(var_43)
    var_45 = [var_1, var_2, var_0]
    var_46 = module_0.pdeque(var_45)



# Parsed testcases at query #2
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.pdeque(var_5)
    var_7 = [var_1, var_2, var_3]
    var_8 = module_0.pdeque(var_7)
    var_9 = [var_0, var_1, var_2]
    var_10 = module_0.pdeque(var_9)
    var_11 = [var_2, var_3, var_4]
    var_12 = module_0.pdeque(var_11)
    var_13 = [var_0, var_2, var_4]
    var_14 = module_0.pdeque(var_13)
    var_15 = [var_1, var_2, var_3]
    var_16 = module_0.pdeque(var_15)
    var_17 = [var_2, var_3, var_4]
    var_18 = module_0.pdeque(var_17)
    var_19 = [var_0, var_1, var_2]
    var_20 = module_0.pdeque(var_19)
    var_21 = [var_1, var_3]
    var_22 = module_0.pdeque(var_21)
    var_23 = [var_0, var_3]
    var_24 = module_0.pdeque(var_23)
    var_25 = [var_2, var_4]
    var_26 = module_0.pdeque(var_25)
    var_27 = 10
    var_28 = var_6[var_27]
    var_29 = module_0.pdeque()
    var_30 = 0
    var_31 = var_29[var_30]
    var_32 = [var_30, var_31, var_2, var_3, var_4]
    var_33 = module_0.pdeque(var_32, var_2)
    var_34 = [var_3, var_4]
    var_35 = module_0.pdeque(var_34, var_2)
    var_36 = [var_2, var_3]
    var_37 = module_0.pdeque(var_36, var_2)
    var_38 = [var_3, var_4]
    var_39 = module_0.pdeque(var_38, var_2)
    var_40 = [var_2, var_4]
    var_41 = module_0.pdeque(var_40, var_2)
    var_42 = [var_2, var_3]
    var_43 = module_0.pdeque(var_42, var_2)
    var_44 = [var_3, var_4]
    var_45 = module_0.pdeque(var_44, var_2)
    var_46 = [var_2, var_3]
    var_47 = module_0.pdeque(var_46, var_2)
    var_48 = 'All tests passed!'
    var_49 = print(var_48)



# Parsed testcases at query #3
#--------------------------



def test_case_0():
    var_0 = 2
    var_1 = 1
    var_2 = [var_0, var_1, var_0]
    var_3 = module_0.pdeque(var_2)
    var_4 = [var_1, var_0]
    var_5 = module_0.pdeque(var_4)
    var_6 = 3
    var_7 = [var_1, var_0, var_6]
    var_8 = module_0.pdeque(var_7)
    var_9 = [var_1, var_0]
    var_10 = module_0.pdeque(var_9)
    var_11 = [var_1, var_0, var_6]
    var_12 = module_0.pdeque(var_11)
    var_13 = 4
    var_14 = []
    var_15 = module_0.pdeque(var_14)
    var_16 = 1
    var_17 = [var_1]
    var_18 = module_0.pdeque(var_17)
    var_19 = []
    var_20 = module_0.pdeque(var_19)
    var_21 = [var_1, var_16, var_1, var_6, var_1]
    var_22 = module_0.pdeque(var_21)
    var_23 = [var_16, var_1, var_6, var_1]
    var_24 = module_0.pdeque(var_23)
    var_25 = [var_1, var_16, var_6]
    var_26 = module_0.pdeque(var_25, var_6)
    var_27 = [var_1, var_6]
    var_28 = module_0.pdeque(var_27, var_6)
    var_29 = [var_1, var_16, var_6]
    var_30 = module_0.pdeque(var_29, var_6)
    var_31 = 4
    var_32 = [var_1, var_31, var_6]
    var_33 = module_0.pdeque(var_32, var_6)
    var_34 = [var_1, var_31]
    var_35 = module_0.pdeque(var_34, var_6)
    var_36 = [var_1, var_31, var_6]
    var_37 = module_0.pdeque(var_36, var_6)
    var_38 = [var_31, var_6]
    var_39 = module_0.pdeque(var_38, var_6)
    var_40 = [var_1, var_31, var_6]
    var_41 = module_0.pdeque(var_40, var_6)
    var_42 = [var_1, var_6]
    var_43 = module_0.pdeque(var_42, var_6)
    var_44 = [var_1, var_31, var_6]
    var_45 = module_0.pdeque(var_44, var_6)
    var_46 = 4
    var_47 = [var_1, var_46, var_6]
    var_48 = module_0.pdeque(var_47, var_6)
    var_49 = [var_1, var_6]
    var_50 = module_0.pdeque(var_49, var_6)
    var_51 = [var_1, var_46]
    var_52 = module_0.pdeque(var_51, var_6)
    var_53 = [var_1]
    var_54 = module_0.pdeque(var_53, var_6)
    var_55 = [var_1, var_46]
    var_56 = module_0.pdeque(var_55, var_6)
    var_57 = 3
    var_58 = []
    var_59 = module_0.pdeque(var_58, var_6)
    var_60 = 1
    var_61 = [var_1]
    var_62 = module_0.pdeque(var_61, var_6)
    var_63 = []
    var_64 = module_0.pdeque(var_63, var_6)
    var_65 = [var_1, var_60]
    var_66 = module_0.pdeque(var_65, var_6)
    var_67 = [var_1]
    var_68 = module_0.pdeque(var_67, var_6)
    var_69 = [var_1, var_60, var_6]
    var_70 = module_0.pdeque(var_69, var_6)
    var_71 = [var_1, var_6]
    var_72 = module_0.pdeque(var_71, var_6)
    var_73 = 4
    var_74 = [var_1, var_60, var_6, var_73]
    var_75 = module_0.pdeque(var_74, var_6)
    var_76 = [var_1, var_6, var_73]
    var_77 = module_0.pdeque(var_76, var_6)
    var_78 = 5
    var_79 = [var_1, var_60, var_6, var_73, var_78]
    var_80 = module_0.pdeque(var_79, var_6)
    var_81 = [var_1, var_6, var_73, var_78]
    var_82 = module_0.pdeque(var_81, var_6)
    var_83 = 6
    var_84 = [var_1, var_60, var_6, var_73, var_78, var_83]
    var_85 = module_0.pdeque(var_84, var_6)
    var_86 = [var_1, var_6, var_73, var_78, var_83]
    var_87 = module_0.pdeque(var_86, var_6)
    var_88 = 7
    var_89 = [var_1, var_60, var_6, var_73, var_78, var_83, var_88]
    var_90 = module_0.pdeque(var_89, var_6)
    var_91 = [var_1, var_6, var_73, var_78, var_83, var_88]
    var_92 = module_0.pdeque(var_91, var_6)
    var_93 = 8
    var_94 = [var_1, var_60, var_6, var_73, var_78, var_83, var_88, var_93]
    var_95 = module_0.pdeque(var_94, var_6)
    var_96 = [var_1, var_6, var_73, var_78, var_83, var_88, var_93]
    var_97 = module_0.pdeque(var_96, var_6)
    var_98 = 9
    var_99 = [var_1, var_60, var_6, var_73, var_78, var_83, var_88, var_93, var_98]
    var_100 = module_0.pdeque(var_99, var_6)
    var_101 = [var_1, var_6, var_73, var_78, var_83, var_88, var_93, var_98]
    var_102 = module_0.pdeque(var_101, var_6)
    var_103 = 10
    var_104 = [var_1, var_60, var_6, var_73, var_78, var_83, var_88, var_93, var_98, var_103]
    var_105 = module_0.pdeque(var_104, var_6)
    var_106 = [var_1, var_6, var_73, var_78, var_83, var_88, var_93, var_98, var_103]
    var_107 = module_0.pdeque(var_106, var_6)



# Parsed testcases at query #4
#--------------------------



def test_case_0():
    var_0 = module_0.pdeque()
    var_1 = module_0.pdeque()
    var_2 = 2
    var_3 = module_0.pdeque()
    var_4 = -1
    var_5 = module_0.pdeque()
    var_6 = 1
    var_7 = [var_6]
    var_8 = module_0.pdeque(var_7)
    var_9 = module_0.pdeque()
    var_10 = module_0.pdeque()
    var_11 = -1
    var_12 = [var_6]
    var_13 = module_0.pdeque(var_12)
    var_14 = 3
    var_15 = [var_6, var_2, var_14]
    var_16 = module_0.pdeque(var_15)
    var_17 = [var_6, var_2]
    var_18 = module_0.pdeque(var_17)
    var_19 = [var_6]
    var_20 = module_0.pdeque(var_19)
    var_21 = -1
    var_22 = [var_2, var_14]
    var_23 = module_0.pdeque(var_22)
    var_24 = [var_6, var_2, var_14]
    var_25 = module_0.pdeque(var_24)
    var_26 = -2
    var_27 = [var_14]
    var_28 = module_0.pdeque(var_27)
    var_29 = [var_6, var_2, var_14]
    var_30 = module_0.pdeque(var_29)
    var_31 = 5
    var_32 = module_0.pdeque()
    var_33 = [var_6, var_2, var_14]
    var_34 = module_0.pdeque(var_33, var_14)
    var_35 = [var_6, var_2]
    var_36 = module_0.pdeque(var_35, var_14)
    var_37 = [var_6]
    var_38 = module_0.pdeque(var_37, var_14)
    var_39 = -1
    var_40 = [var_2, var_14]
    var_41 = module_0.pdeque(var_40, var_14)
    var_42 = []
    var_43 = 0
    var_44 = module_0.pdeque(var_42, var_43)
    var_45 = []
    var_46 = module_0.pdeque(var_45, var_43)
    var_47 = []
    var_48 = module_0.pdeque(var_47, var_43)
    var_49 = -1
    var_50 = []
    var_51 = module_0.pdeque(var_50, var_43)
    var_52 = [var_6]
    var_53 = module_0.pdeque(var_52, var_6)
    var_54 = []
    var_55 = module_0.pdeque(var_54, var_6)
    var_56 = []
    var_57 = module_0.pdeque(var_56, var_6)
    var_58 = -1
    var_59 = [var_6]
    var_60 = module_0.pdeque(var_59, var_6)
    var_61 = [var_6, var_2, var_14]
    var_62 = module_0.pdeque(var_61, var_31)
    var_63 = [var_6, var_2]
    var_64 = module_0.pdeque(var_63, var_31)
    var_65 = [var_6]
    var_66 = module_0.pdeque(var_65, var_31)
    var_67 = -1
    var_68 = [var_2, var_14]
    var_69 = module_0.pdeque(var_68, var_31)
    var_70 = [var_6, var_2, var_14]
    var_71 = module_0.pdeque(var_70, var_14)
    var_72 = [var_6, var_2]
    var_73 = module_0.pdeque(var_72, var_14)
    var_74 = [var_6]
    var_75 = module_0.pdeque(var_74, var_14)
    var_76 = -1
    var_77 = [var_2, var_14]
    var_78 = module_0.pdeque(var_77, var_14)
    var_79 = 4
    var_80 = [var_6, var_2, var_14, var_79, var_31]
    var_81 = module_0.pdeque(var_80, var_14)
    var_82 = [var_2, var_14, var_79]
    var_83 = module_0.pdeque(var_82, var_14)
    var_84 = [var_2, var_14]
    var_85 = module_0.pdeque(var_84, var_14)
    var_86 = -1
    var_87 = [var_14, var_79, var_31]
    var_88 = module_0.pdeque(var_87, var_14)
    var_89 = []
    var_90 = module_0.pdeque(var_89, var_43)
    var_91 = -1
    var_92 = []
    var_93 = module_0.pdeque(var_92, var_43)
    var_94 = [var_6]
    var_95 = module_0.pdeque(var_94, var_6)
    var_96 = -1
    var_97 = [var_6]
    var_98 = module_0.pdeque(var_97, var_6)
    var_99 = [var_6, var_2, var_14]
    var_100 = module_0.pdeque(var_99, var_31)
    var_101 = -1
    var_102 = [var_2, var_14]
    var_103 = module_0.pdeque(var_102, var_31)
    var_104 = [var_6, var_2, var_14]
    var_105 = module_0.pdeque(var_104, var_14)
    var_106 = -1
    var_107 = [var_2, var_14]
    var_108 = module_0.pdeque(var_107, var_14)
    var_109 = [var_6, var_2, var_14, var_79, var_31]
    var_110 = module_0.pdeque(var_109, var_14)
    var_111 = -1
    var_112 = [var_14, var_79, var_31]
    var_113 = module_0.pdeque(var_112, var_14)
    var_114 = 1000
    var_115 = range(var_114)
    var_116 = module_0.pdeque(var_115)
    var_117 = 999
    var_118 = range(var_117)
    var_119 = module_0.pdeque(var_118)
    var_120 = 500
    var_121 = range(var_120)
    var_122 = module_0.pdeque(var_121)
    var_123 = -500
    var_124 = range(var_120, var_114)
    var_125 = module_0.pdeque(var_124)
    var_126 = range(var_114)
    var_127 = module_0.pdeque(var_126, var_120)
    var_128 = range(var_120, var_117)
    var_129 = module_0.pdeque(var_128, var_120)
    var_130 = 250
    var_131 = 749
    var_132 = range(var_120, var_131)
    var_133 = module_0.pdeque(var_132, var_120)
    var_134 = -250
    var_135 = 750
    var_136 = range(var_135, var_114)
    var_137 = module_0.pdeque(var_136, var_120)
    var_138 = range(var_114)
    var_139 = module_0.pdeque(var_138, var_43)
    var_140 = []
    var_141 = module_0.pdeque(var_140, var_43)
    var_142 = []
    var_143 = module_0.pdeque(var_142, var_43)
    var_144 = -500
    var_145 = []
    var_146 = module_0.pdeque(var_145, var_43)
    var_147 = range(var_114)
    var_148 = module_0.pdeque(var_147, var_6)
    var_149 = []
    var_150 = module_0.pdeque(var_149, var_6)
    var_151 = []
    var_152 = module_0.pdeque(var_151, var_6)
    var_153 = -500
    var_154 = [var_117]
    var_155 = module_0.pdeque(var_154, var_6)
    var_156 = range(var_114)
    var_157 = 1500
    var_158 = module_0.pdeque(var_156, var_157)
    var_159 = range(var_117)
    var_160 = module_0.pdeque(var_159, var_157)
    var_161 = range(var_120)
    var_162 = module_0.pdeque(var_161, var_157)
    var_163 = -500
    var_164 = range(var_120, var_114)
    var_165 = module_0.pdeque(var_164, var_157)
    var_166 = range(var_114)
    var_167 = module_0.pdeque(var_166, var_114)
    var_168 = range(var_117)
    var_169 = module_0.pdeque(var_168, var_114)
    var_170 = range(var_120)
    var_171 = module_0.pdeque(var_170, var_114)
    var_172 = -500
    var_173 = range(var_120, var_114)
    var_174 = module_0.pdeque(var_173, var_114)
    var_175 = range(var_114)
    var_176 = module_0.pdeque(var_175, var_120)
    var_177 = range(var_120, var_117)
    var_178 = module_0.pdeque(var_177, var_120)
    var_179 = range(var_120, var_131)
    var_180 = module_0.pdeque(var_179, var_120)
    var_181 = -250
    var_182 = range(var_135, var_114)
    var_183 = module_0.pdeque(var_182, var_120)
    var_184 = range(var_114)
    var_185 = module_0.pdeque(var_184, var_43)
    var_186 = -1
    var_187 = []
    var_188 = module_0.pdeque(var_187, var_43)
    var_189 = range(var_114)
    var_190 = module_0.pdeque(var_189, var_6)
    var_191 = -1
    var_192 = [var_117]
    var_193 = module_0.pdeque(var_192, var_6)
    var_194 = range(var_114)
    var_195 = module_0.pdeque(var_194, var_157)



# Parsed testcases at query #5
#--------------------------



def test_case_0():
    var_0 = module_0.pdeque()
    var_1 = module_0.pdeque()
    var_2 = 1
    var_3 = [var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = module_0.pdeque()
    var_6 = 2
    var_7 = 3
    var_8 = [var_2, var_6, var_7]
    var_9 = module_0.pdeque(var_8)
    var_10 = [var_6, var_7]
    var_11 = module_0.pdeque(var_10)
    var_12 = [var_2, var_6, var_7]
    var_13 = module_0.pdeque(var_12)
    var_14 = 5
    var_15 = module_0.pdeque()
    var_16 = [var_2, var_6, var_7]
    var_17 = module_0.pdeque(var_16)
    var_18 = -2
    var_19 = [var_2]
    var_20 = module_0.pdeque(var_19)
    var_21 = [var_2, var_6, var_7]
    var_22 = module_0.pdeque(var_21)
    var_23 = 0
    var_24 = [var_2, var_6, var_7]
    var_25 = module_0.pdeque(var_24, var_7)
    var_26 = [var_6, var_7]
    var_27 = module_0.pdeque(var_26, var_7)
    var_28 = [var_2, var_6, var_7]
    var_29 = module_0.pdeque(var_28, var_7)
    var_30 = []
    var_31 = module_0.pdeque(var_30, var_7)
    var_32 = [var_2, var_6, var_7]
    var_33 = module_0.pdeque(var_32, var_7)
    var_34 = -2
    var_35 = [var_2]
    var_36 = module_0.pdeque(var_35, var_7)
    var_37 = [var_2, var_6, var_7]
    var_38 = module_0.pdeque(var_37, var_7)
    var_39 = 'All test cases passed!'
    var_40 = print(var_39)



# Parsed testcases at query #6
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.pdeque(var_5)
    var_7 = [var_1, var_2]
    var_8 = module_0.pdeque(var_7)
    var_9 = [var_1, var_2, var_3]
    var_10 = module_0.pdeque(var_9)
    var_11 = [var_0, var_2, var_4]
    var_12 = module_0.pdeque(var_11)
    var_13 = [var_4, var_3, var_2, var_1, var_0]
    var_14 = module_0.pdeque(var_13)
    var_15 = [var_2, var_3, var_4]
    var_16 = module_0.pdeque(var_15)
    var_17 = [var_0, var_1]
    var_18 = module_0.pdeque(var_17)
    var_19 = []
    var_20 = module_0.pdeque(var_19)
    var_21 = []
    var_22 = module_0.pdeque(var_21)
    var_23 = []
    var_24 = module_0.pdeque(var_23)
    var_25 = []
    var_26 = module_0.pdeque(var_25)
    var_27 = []
    var_28 = module_0.pdeque(var_27)
    var_29 = [var_0, var_2, var_4]
    var_30 = module_0.pdeque(var_29)
    var_31 = [var_4, var_2]
    var_32 = module_0.pdeque(var_31)
    var_33 = [var_4, var_3, var_2, var_1]
    var_34 = module_0.pdeque(var_33)
    var_35 = []
    var_36 = module_0.pdeque(var_35)
    var_37 = []
    var_38 = module_0.pdeque(var_37)
    var_39 = [var_0, var_1, var_2, var_3, var_4]
    var_40 = module_0.pdeque(var_39)
    var_41 = [var_4, var_3, var_2, var_1]
    var_42 = module_0.pdeque(var_41)
    var_43 = [var_0, var_3]
    var_44 = module_0.pdeque(var_43)
    var_45 = [var_4, var_1]
    var_46 = module_0.pdeque(var_45)
    var_47 = [var_0, var_4]
    var_48 = module_0.pdeque(var_47)
    var_49 = [var_4, var_0]
    var_50 = module_0.pdeque(var_49)
    var_51 = [var_0]
    var_52 = module_0.pdeque(var_51)
    var_53 = [var_4]
    var_54 = module_0.pdeque(var_53)
    var_55 = [var_0]
    var_56 = module_0.pdeque(var_55)
    var_57 = [var_4]
    var_58 = module_0.pdeque(var_57)
    var_59 = [var_0]
    var_60 = module_0.pdeque(var_59)
    var_61 = [var_4]
    var_62 = module_0.pdeque(var_61)
    var_63 = [var_0]
    var_64 = module_0.pdeque(var_63)
    var_65 = [var_4]
    var_66 = module_0.pdeque(var_65)
    var_67 = [var_0]
    var_68 = module_0.pdeque(var_67)
    var_69 = [var_4]
    var_70 = module_0.pdeque(var_69)
    var_71 = [var_0]
    var_72 = module_0.pdeque(var_71)
    var_73 = [var_4]
    var_74 = module_0.pdeque(var_73)
    var_75 = [var_0]
    var_76 = module_0.pdeque(var_75)
    var_77 = [var_4]
    var_78 = module_0.pdeque(var_77)
    var_79 = [var_0]
    var_80 = module_0.pdeque(var_79)
    var_81 = [var_4]
    var_82 = module_0.pdeque(var_81)
    var_83 = [var_0]
    var_84 = module_0.pdeque(var_83)
    var_85 = [var_4]
    var_86 = module_0.pdeque(var_85)
    var_87 = [var_0]
    var_88 = module_0.pdeque(var_87)
    var_89 = [var_4]
    var_90 = module_0.pdeque(var_89)
    var_91 = [var_0]
    var_92 = module_0.pdeque(var_91)
    var_93 = [var_4]
    var_94 = module_0.pdeque(var_93)
    var_95 = [var_0]
    var_96 = module_0.pdeque(var_95)
    var_97 = [var_4]
    var_98 = module_0.pdeque(var_97)
    var_99 = [var_0]
    var_100 = module_0.pdeque(var_99)
    var_101 = [var_4]
    var_102 = module_0.pdeque(var_101)
    var_103 = [var_0]
    var_104 = module_0.pdeque(var_103)
    var_105 = [var_4]
    var_106 = module_0.pdeque(var_105)
    var_107 = [var_0]
    var_108 = module_0.pdeque(var_107)
    var_109 = [var_4]
    var_110 = module_0.pdeque(var_109)
    var_111 = [var_0]
    var_112 = module_0.pdeque(var_111)
    var_113 = [var_4]
    var_114 = module_0.pdeque(var_113)
    var_115 = [var_0]
    var_116 = module_0.pdeque(var_115)
    var_117 = [var_4]
    var_118 = module_0.pdeque(var_117)
    var_119 = [var_0]
    var_120 = module_0.pdeque(var_119)
    var_121 = [var_4]
    var_122 = module_0.pdeque(var_121)
    var_123 = [var_0]
    var_124 = module_0.pdeque(var_123)
    var_125 = [var_4]
    var_126 = module_0.pdeque(var_125)
    var_127 = [var_0]
    var_128 = module_0.pdeque(var_127)
    var_129 = [var_4]
    var_130 = module_0.pdeque(var_129)
    var_131 = [var_0]
    var_132 = module_0.pdeque(var_131)
    var_133 = [var_4]
    var_134 = module_0.pdeque(var_133)
    var_135 = [var_0]
    var_136 = module_0.pdeque(var_135)
    var_137 = [var_4]
    var_138 = module_0.pdeque(var_137)
    var_139 = [var_0]
    var_140 = module_0.pdeque(var_139)
    var_141 = [var_4]
    var_142 = module_0.pdeque(var_141)
    var_143 = [var_0]
    var_144 = module_0.pdeque(var_143)
    var_145 = [var_4]
    var_146 = module_0.pdeque(var_145)
    var_147 = [var_0]
    var_148 = module_0.pdeque(var_147)
    var_149 = [var_4]
    var_150 = module_0.pdeque(var_149)
    var_151 = [var_0]
    var_152 = module_0.pdeque(var_151)
    var_153 = [var_4]
    var_154 = module_0.pdeque(var_153)
    var_155 = [var_0]
    var_156 = module_0.pdeque(var_155)
    var_157 = [var_4]
    var_158 = module_0.pdeque(var_157)
    var_159 = [var_0]
    var_160 = module_0.pdeque(var_159)
    var_161 = [var_4]
    var_162 = module_0.pdeque(var_161)
    var_163 = [var_0]
    var_164 = module_0.pdeque(var_163)
    var_165 = [var_4]
    var_166 = module_0.pdeque(var_165)
    var_167 = [var_0]
    var_168 = module_0.pdeque(var_167)
    var_169 = [var_4]
    var_170 = module_0.pdeque(var_169)
    var_171 = [var_0]
    var_172 = module_0.pdeque(var_171)
    var_173 = [var_4]
    var_174 = module_0.pdeque(var_173)
    var_175 = [var_0]
    var_176 = module_0.pdeque(var_175)
    var_177 = [var_4]
    var_178 = module_0.pdeque(var_177)
    var_179 = [var_0]
    var_180 = module_0.pdeque(var_179)
    var_181 = [var_4]
    var_182 = module_0.pdeque(var_181)
    var_183 = [var_0]
    var_184 = module_0.pdeque(var_183)
    var_185 = [var_4]
    var_186 = module_0.pdeque(var_185)
    var_187 = [var_0]
    var_188 = module_0.pdeque(var_187)
    var_189 = [var_4]
    var_190 = module_0.pdeque(var_189)
    var_191 = [var_0]
    var_192 = module_0.pdeque(var_191)
    var_193 = [var_4]
    var_194 = module_0.pdeque(var_193)
    var_195 = [var_0]
    var_196 = module_0.pdeque(var_195)
    var_197 = [var_4]
    var_198 = module_0.pdeque(var_197)
    var_199 = [var_0]
    var_200 = module_0.pdeque(var_199)
    var_201 = [var_4]
    var_202 = module_0.pdeque(var_201)
    var_203 = [var_0]
    var_204 = module_0.pdeque(var_203)
    var_205 = [var_4]
    var_206 = module_0.pdeque(var_205)
    var_207 = [var_0]
    var_208 = module_0.pdeque(var_207)
    var_209 = [var_4]
    var_210 = module_0.pdeque(var_209)
    var_211 = [var_0]
    var_212 = module_0.pdeque(var_211)
    var_213 = [var_4]
    var_214 = module_0.pdeque(var_213)
    var_215 = [var_0]
    var_216 = module_0.pdeque(var_215)
    var_217 = [var_4]
    var_218 = module_0.pdeque(var_217)
    var_219 = [var_0]
    var_220 = module_0.pdeque(var_219)
    var_221 = [var_4]
    var_222 = module_0.pdeque(var_221)
    var_223 = [var_0]
    var_224 = module_0.pdeque(var_223)
    var_225 = [var_4]
    var_226 = module_0.pdeque(var_225)
    var_227 = [var_0]
    var_228 = module_0.pdeque(var_227)



# Parsed testcases at query #7
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.pdeque(var_3)
    var_5 = [var_1, var_2]
    var_6 = module_0.pdeque(var_5)
    var_7 = 4
    var_8 = 5
    var_9 = [var_0, var_1, var_2, var_7, var_8]
    var_10 = module_0.pdeque(var_9)
    var_11 = [var_7, var_8]
    var_12 = module_0.pdeque(var_11)
    var_13 = [var_0, var_1]
    var_14 = module_0.pdeque(var_13)
    var_15 = []
    var_16 = module_0.pdeque(var_15)
    var_17 = [var_0, var_1, var_2]
    var_18 = module_0.pdeque(var_17)
    var_19 = -2
    var_20 = [var_0]
    var_21 = module_0.pdeque(var_20)
    var_22 = []
    var_23 = module_0.pdeque(var_22)
    var_24 = []
    var_25 = module_0.pdeque(var_24)
    var_26 = [var_0, var_1, var_2]
    var_27 = module_0.pdeque(var_26, var_2)
    var_28 = [var_1, var_2]
    var_29 = module_0.pdeque(var_28, var_2)
    var_30 = [var_0, var_1, var_2, var_7]
    var_31 = module_0.pdeque(var_30, var_2)
    var_32 = [var_2, var_7]
    var_33 = module_0.pdeque(var_32, var_2)
    var_34 = [var_0, var_1, var_2]
    var_35 = module_0.pdeque(var_34)
    var_36 = 0
    var_37 = 1000
    var_38 = range(var_37)
    var_39 = module_0.pdeque(var_38)
    var_40 = 500
    var_41 = [var_0, var_1, var_2]
    var_42 = module_0.pdeque(var_41)
    var_43 = 'a'
    var_44 = 3.14
    var_45 = None
    var_46 = [var_43, var_0, var_44, var_45]
    var_47 = module_0.pdeque(var_46)
    var_48 = [var_44, var_45]
    var_49 = module_0.pdeque(var_48)
    var_50 = [var_0, var_1, var_2]
    var_51 = module_0.pdeque(var_50)
    var_52 = [var_1, var_2, var_7]
    var_53 = module_0.pdeque(var_52)
    var_54 = [var_0, var_1, var_2]
    var_55 = module_0.pdeque(var_54)
    var_56 = [var_36, var_1, var_2]
    var_57 = module_0.pdeque(var_56)
    var_58 = 42
    var_59 = [var_58]
    var_60 = module_0.pdeque(var_59)
    var_61 = []
    var_62 = module_0.pdeque(var_61)
    var_63 = [var_0, var_1, var_2, var_7]
    var_64 = module_0.pdeque(var_63, var_2)
    var_65 = [var_0, var_1, var_2]
    var_66 = module_0.pdeque(var_65)
    var_67 = []
    var_68 = module_0.pdeque(var_67)
    var_69 = [var_0, var_1]
    var_70 = module_0.pdeque(var_69)
    var_71 = 10
    var_72 = []
    var_73 = module_0.pdeque(var_72)
    var_74 = range(var_71)
    var_75 = module_0.pdeque(var_74)
    var_76 = range(var_8, var_71)
    var_77 = list(var_76)
    var_78 = []
    var_79 = module_0.pdeque(var_78)
    var_80 = -1
    var_81 = []
    var_82 = module_0.pdeque(var_81)
    var_83 = 20
    var_84 = 30
    var_85 = 40
    var_86 = [var_71, var_83, var_84, var_85]
    var_87 = module_0.pdeque(var_86)
    var_88 = []
    var_89 = module_0.pdeque(var_88, var_36)
    var_90 = []
    var_91 = module_0.pdeque(var_90, var_36)
    var_92 = [var_0, var_1, var_2]
    var_93 = module_0.pdeque(var_92, var_8)
    var_94 = []
    var_95 = module_0.pdeque(var_94, var_8)
    var_96 = [var_0, var_1, var_2, var_7, var_8]
    var_97 = module_0.pdeque(var_96)
    var_98 = 10000
    var_99 = range(var_98)
    var_100 = module_0.pdeque(var_99)
    var_101 = 5000
    var_102 = [var_0, var_1, var_2]
    var_103 = module_0.pdeque(var_102)
    var_104 = [var_7, var_8]
    var_105 = [var_1, var_2, var_7, var_8]
    var_106 = module_0.pdeque(var_105)
    var_107 = [var_0, var_1, var_2]
    var_108 = module_0.pdeque(var_107)
    var_109 = -1
    var_110 = [var_36, var_109]
    var_111 = -1
    var_112 = [var_111, var_36, var_1, var_2]
    var_113 = module_0.pdeque(var_112)
    var_114 = [var_0, var_1, var_2]
    var_115 = module_0.pdeque(var_114)
    var_116 = []
    var_117 = module_0.pdeque(var_116)
    var_118 = []
    var_119 = module_0.pdeque(var_118)
    var_120 = [var_0, var_1, var_2, var_7, var_8]
    var_121 = module_0.pdeque(var_120)
    var_122 = 6
    var_123 = [var_36, var_7, var_8, var_122]
    var_124 = module_0.pdeque(var_123)
    var_125 = 100
    var_126 = range(var_125)
    var_127 = module_0.pdeque(var_126)
    var_128 = 60
    var_129 = range(var_128, var_125)
    var_130 = list(var_129)
    var_131 = [var_0, var_1, var_2, var_7]
    var_132 = module_0.pdeque(var_131)
    var_133 = [var_7, var_2]
    var_134 = module_0.pdeque(var_133)
    var_135 = [var_0, var_1, var_1, var_2, var_1, var_7]
    var_136 = module_0.pdeque(var_135)
    var_137 = [var_0, var_1, var_2, var_1, var_7]
    var_138 = module_0.pdeque(var_137)



# Parsed testcases at query #8
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.pdeque(var_5)
    var_7 = [var_0, var_1, var_3, var_4]
    var_8 = module_0.pdeque(var_7)
    var_9 = [var_0, var_1, var_2, var_3, var_4]
    var_10 = module_0.pdeque(var_9)
    var_11 = [var_0, var_1, var_2, var_3]
    var_12 = module_0.pdeque(var_11)
    var_13 = [var_0, var_1, var_2, var_3, var_4]
    var_14 = module_0.pdeque(var_13)
    var_15 = 6
    var_16 = []
    var_17 = module_0.pdeque(var_16)
    var_18 = 1
    var_19 = [var_18, var_1, var_2, var_1, var_18]
    var_20 = module_0.pdeque(var_19)
    var_21 = [var_18, var_2, var_1, var_18]
    var_22 = module_0.pdeque(var_21)
    var_23 = [var_18, var_1, var_2, var_3, var_4]
    var_24 = module_0.pdeque(var_23, var_3)
    var_25 = [var_18, var_1, var_3, var_4]
    var_26 = module_0.pdeque(var_25, var_3)
    var_27 = [var_18, var_1, var_2, var_3, var_4]
    var_28 = module_0.pdeque(var_27, var_2)
    var_29 = [var_1, var_3, var_4]
    var_30 = module_0.pdeque(var_29, var_2)
    var_31 = [var_18, var_1, var_2, var_3, var_4]
    var_32 = module_0.pdeque(var_31, var_2)
    var_33 = [var_18, var_1, var_3]
    var_34 = module_0.pdeque(var_33, var_2)
    var_35 = [var_18, var_1, var_2, var_3, var_4]
    var_36 = module_0.pdeque(var_35, var_3)
    var_37 = [var_18, var_2, var_3, var_4]
    var_38 = module_0.pdeque(var_37, var_3)
    var_39 = [var_18, var_1, var_2, var_3, var_4]
    var_40 = module_0.pdeque(var_39, var_3)
    var_41 = [var_18, var_1, var_2, var_4]
    var_42 = module_0.pdeque(var_41, var_3)
    var_43 = [var_18, var_1, var_2, var_3, var_4]
    var_44 = module_0.pdeque(var_43, var_3)
    var_45 = [var_1, var_2, var_3, var_4]
    var_46 = module_0.pdeque(var_45, var_3)
    var_47 = [var_18, var_1, var_2, var_1, var_18]
    var_48 = module_0.pdeque(var_47, var_3)
    var_49 = [var_18, var_2, var_1, var_18]
    var_50 = module_0.pdeque(var_49, var_3)
    var_51 = [var_18, var_1, var_2, var_3, var_4]
    var_52 = module_0.pdeque(var_51, var_3)
    var_53 = 6
    var_54 = [var_53, var_1, var_2, var_3, var_4]
    var_55 = module_0.pdeque(var_54, var_2)
    var_56 = [var_1, var_2, var_4]
    var_57 = module_0.pdeque(var_56, var_2)
    var_58 = [var_53, var_1, var_2, var_3, var_4]
    var_59 = module_0.pdeque(var_58, var_2)
    var_60 = [var_53, var_1, var_3]
    var_61 = module_0.pdeque(var_60, var_2)
    var_62 = [var_53, var_1, var_2]
    var_63 = module_0.pdeque(var_62, var_1)
    var_64 = [var_1, var_2]
    var_65 = module_0.pdeque(var_64, var_1)
    var_66 = [var_53, var_1, var_2]
    var_67 = module_0.pdeque(var_66, var_1)
    var_68 = [var_53, var_1]
    var_69 = module_0.pdeque(var_68, var_1)
    var_70 = []
    var_71 = 0
    var_72 = module_0.pdeque(var_70, var_71)
    var_73 = 1
    var_74 = []
    var_75 = module_0.pdeque(var_74, var_71)
    var_76 = 1
    var_77 = [var_76, var_1, var_2]
    var_78 = -1
    var_79 = module_0.pdeque(var_77, var_78)
    var_80 = 1
    var_81 = [var_80, var_1, var_2]
    var_82 = -1
    var_83 = module_0.pdeque(var_81, var_82)
    var_84 = 3
    var_85 = [var_84, var_1, var_2]
    var_86 = 2.5
    var_87 = module_0.pdeque(var_85, var_86)
    var_88 = 1
    var_89 = [var_88, var_1, var_2]
    var_90 = module_0.pdeque(var_89, var_86)
    var_91 = 3
    var_92 = [var_91, var_1, var_2]
    var_93 = '2'
    var_94 = module_0.pdeque(var_92, var_93)
    var_95 = 1
    var_96 = [var_95, var_1, var_2]
    var_97 = module_0.pdeque(var_96, var_93)
    var_98 = 3



# Parsed testcases at query #9
#--------------------------



def test_case_0():
    var_0 = module_0.pdeque()
    var_1 = module_0.pdeque()
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.pdeque(var_5)
    var_7 = [var_2, var_3, var_4]
    var_8 = module_0.pdeque(var_7)
    var_9 = [var_2, var_3, var_4]
    var_10 = module_0.pdeque(var_9)
    var_11 = 4
    var_12 = 5
    var_13 = 6
    var_14 = [var_11, var_12, var_13]
    var_15 = module_0.pdeque(var_14)
    var_16 = [var_2, var_3, var_4]
    var_17 = module_0.pdeque(var_16)
    var_18 = [var_2, var_3, var_4]
    var_19 = [var_2, var_3, var_4]
    var_20 = module_0.pdeque(var_19, var_12)
    var_21 = [var_2, var_3, var_4]
    var_22 = 10
    var_23 = module_0.pdeque(var_21, var_22)
    var_24 = [var_2, var_3, var_4]
    var_25 = module_0.pdeque(var_24)
    var_26 = [var_4, var_3, var_2]
    var_27 = module_0.pdeque(var_26)
    var_28 = [var_2, var_3, var_4]
    var_29 = module_0.pdeque(var_28)
    var_30 = [var_2, var_3, var_4, var_11]
    var_31 = module_0.pdeque(var_30)
    var_32 = [var_2, var_3, var_4]
    var_33 = module_0.pdeque(var_32, var_12)
    var_34 = [var_2, var_3, var_4]
    var_35 = module_0.pdeque(var_34, var_4)
    var_36 = [var_2, var_3, var_4]
    var_37 = module_0.pdeque(var_36)
    var_38 = [var_2, var_3, var_4]
    var_39 = module_0.pdeque(var_38)
    var_40 = [var_2, var_3]
    var_41 = [var_4]
    var_42 = True
    var_43 = [var_42, var_3, var_4]
    var_44 = module_0.pdeque(var_43, var_12)
    var_45 = [var_42, var_3, var_4]
    var_46 = module_0.pdeque(var_45, var_12)
    var_47 = [var_42, var_3]
    var_48 = [var_4]
    var_49 = True



# Parsed testcases at query #10
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.pdeque(var_5)
    var_7 = [var_0, var_1, var_2, var_3, var_4]
    var_8 = module_0.pdeque(var_7)
    var_9 = [var_0, var_1, var_2, var_3, var_4]
    var_10 = module_0.pdeque(var_9)
    var_11 = [var_1, var_2, var_3]
    var_12 = module_0.pdeque(var_11)
    var_13 = [var_0, var_1, var_2]
    var_14 = module_0.pdeque(var_13)
    var_15 = [var_2, var_3, var_4]
    var_16 = module_0.pdeque(var_15)
    var_17 = [var_0, var_2, var_4]
    var_18 = module_0.pdeque(var_17)
    var_19 = [var_0, var_1, var_2, var_3, var_4]
    var_20 = module_0.pdeque(var_19)
    var_21 = [var_4, var_3, var_2, var_1, var_0]
    var_22 = module_0.pdeque(var_21)
    var_23 = [var_4, var_3, var_2]
    var_24 = module_0.pdeque(var_23)
    var_25 = [var_2, var_1, var_0]
    var_26 = module_0.pdeque(var_25)
    var_27 = [var_0, var_1, var_2, var_3, var_4]
    var_28 = module_0.pdeque(var_27)
    var_29 = [var_1, var_3]
    var_30 = module_0.pdeque(var_29)
    var_31 = [var_0, var_3]
    var_32 = module_0.pdeque(var_31)
    var_33 = [var_2, var_4]
    var_34 = module_0.pdeque(var_33)
    var_35 = [var_0, var_1, var_2, var_3, var_4]
    var_36 = module_0.pdeque(var_35)
    var_37 = [var_2, var_3, var_4]
    var_38 = module_0.pdeque(var_37)
    var_39 = [var_0, var_1, var_2]
    var_40 = module_0.pdeque(var_39)
    var_41 = []
    var_42 = module_0.pdeque(var_41)
    var_43 = [var_0, var_1, var_2, var_3, var_4]
    var_44 = module_0.pdeque(var_43)
    var_45 = [var_1, var_2, var_3]
    var_46 = module_0.pdeque(var_45)
    var_47 = [var_3, var_4]
    var_48 = module_0.pdeque(var_47)
    var_49 = [var_0, var_1]
    var_50 = module_0.pdeque(var_49)
    var_51 = [var_0, var_1, var_2, var_3, var_4]
    var_52 = module_0.pdeque(var_51)
    var_53 = [var_4, var_2]
    var_54 = module_0.pdeque(var_53)
    var_55 = [var_4, var_2, var_0]
    var_56 = module_0.pdeque(var_55)
    var_57 = [var_3, var_1]
    var_58 = module_0.pdeque(var_57)
    var_59 = [var_0, var_1, var_2, var_3, var_4]
    var_60 = module_0.pdeque(var_59)
    var_61 = 0
    var_62 = var_60[::var_61]
    var_63 = [var_61, var_62, var_2, var_3, var_4]
    var_64 = module_0.pdeque(var_63)
    var_65 = 'invalid'
    var_66 = var_64[var_65]
    var_67 = [var_65, var_66, var_2, var_3, var_4]
    var_68 = module_0.pdeque(var_67)
    var_69 = 10
    var_70 = var_68[var_69]
    var_71 = module_0.pdeque()
    var_72 = 0
    var_73 = var_71[var_72]
    var_74 = [var_72, var_73, var_2, var_3, var_4]
    var_75 = module_0.pdeque(var_74, var_2)
    var_76 = [var_2, var_3, var_4]
    var_77 = module_0.pdeque(var_76, var_2)
    var_78 = [var_3, var_4]
    var_79 = module_0.pdeque(var_78, var_2)
    var_80 = [var_2, var_3]
    var_81 = module_0.pdeque(var_80, var_2)
    var_82 = [var_72, var_73, var_2, var_3, var_4]
    var_83 = module_0.pdeque(var_82, var_2)
    var_84 = [var_2, var_4]
    var_85 = module_0.pdeque(var_84, var_2)
    var_86 = [var_3]
    var_87 = module_0.pdeque(var_86, var_2)
    var_88 = [var_72, var_73, var_2, var_3, var_4]
    var_89 = module_0.pdeque(var_88, var_2)
    var_90 = [var_4, var_3, var_2]
    var_91 = module_0.pdeque(var_90, var_2)
    var_92 = [var_4, var_2]
    var_93 = module_0.pdeque(var_92, var_2)
    var_94 = [var_72, var_73, var_2, var_3, var_4]
    var_95 = module_0.pdeque(var_94, var_2)
    var_96 = [var_4]
    var_97 = module_0.pdeque(var_96, var_2)
    var_98 = [var_2, var_3]
    var_99 = module_0.pdeque(var_98, var_2)
    var_100 = []
    var_101 = module_0.pdeque(var_100, var_2)
    var_102 = [var_72, var_73, var_2, var_3, var_4]
    var_103 = module_0.pdeque(var_102, var_2)
    var_104 = [var_2, var_3]
    var_105 = module_0.pdeque(var_104, var_2)
    var_106 = [var_3, var_4]
    var_107 = module_0.pdeque(var_106, var_2)
    var_108 = [var_2]
    var_109 = module_0.pdeque(var_108, var_2)
    var_110 = [var_72, var_73, var_2, var_3, var_4]
    var_111 = module_0.pdeque(var_110, var_2)
    var_112 = [var_4, var_3]
    var_113 = module_0.pdeque(var_112, var_2)
    var_114 = [var_4, var_2]
    var_115 = module_0.pdeque(var_114, var_2)
    var_116 = [var_3, var_2]
    var_117 = module_0.pdeque(var_116, var_2)
    var_118 = [var_72, var_73, var_2, var_3, var_4]
    var_119 = module_0.pdeque(var_118, var_2)
    var_120 = 0
    var_121 = var_119[::var_120]
    var_122 = [var_120, var_121, var_2, var_3, var_4]
    var_123 = module_0.pdeque(var_122, var_2)
    var_124 = 'invalid'
    var_125 = var_123[var_124]
    var_126 = [var_124, var_125, var_2, var_3, var_4]
    var_127 = module_0.pdeque(var_126, var_2)
    var_128 = 10
    var_129 = var_127[var_128]
    var_130 = module_0.pdeque(maxlen=var_2)
    var_131 = 0
    var_132 = var_130[var_131]
    var_133 = 'All test cases passed!'
    var_134 = print(var_133)



# Parsed testcases at query #11
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.pdeque(var_5)
    var_7 = [var_0, var_1, var_3, var_4]
    var_8 = module_0.pdeque(var_7)
    var_9 = [var_0, var_1, var_2, var_3, var_4]
    var_10 = module_0.pdeque(var_9)
    var_11 = [var_0, var_1, var_2, var_3]
    var_12 = module_0.pdeque(var_11)
    var_13 = [var_0, var_1, var_2, var_1, var_3]
    var_14 = module_0.pdeque(var_13)
    var_15 = [var_0, var_2, var_1, var_3]
    var_16 = module_0.pdeque(var_15)
    var_17 = [var_0, var_1, var_2, var_3, var_4]
    var_18 = module_0.pdeque(var_17)
    var_19 = 6
    var_20 = []
    var_21 = module_0.pdeque(var_20)
    var_22 = 1
    var_23 = [var_22]
    var_24 = module_0.pdeque(var_23)
    var_25 = []
    var_26 = module_0.pdeque(var_25)
    var_27 = [var_22, var_22, var_22]
    var_28 = module_0.pdeque(var_27)
    var_29 = [var_22, var_22]
    var_30 = module_0.pdeque(var_29)
    var_31 = [var_22, var_1, var_2, var_3, var_4]
    var_32 = module_0.pdeque(var_31, var_3)
    var_33 = [var_22, var_1, var_3, var_4]
    var_34 = module_0.pdeque(var_33, var_3)
    var_35 = [var_22, var_1, var_2, var_3, var_4]
    var_36 = module_0.pdeque(var_35, var_3)
    var_37 = 6
    var_38 = [var_37, var_1, var_2, var_1, var_3]
    var_39 = module_0.pdeque(var_38, var_3)
    var_40 = [var_37, var_2, var_1, var_3]
    var_41 = module_0.pdeque(var_40, var_3)
    var_42 = 'All test cases passed'
    var_43 = print(var_42)



# Parsed testcases at query #12
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.pdeque(var_5)
    var_7 = [var_3, var_4, var_0, var_1, var_2]
    var_8 = module_0.pdeque(var_7)
    var_9 = [var_0, var_1, var_2, var_3, var_4]
    var_10 = module_0.pdeque(var_9)
    var_11 = -2
    var_12 = [var_2, var_3, var_4, var_0, var_1]
    var_13 = module_0.pdeque(var_12)
    var_14 = [var_0, var_1, var_2, var_3, var_4]
    var_15 = module_0.pdeque(var_14)
    var_16 = 0
    var_17 = [var_0, var_1, var_2, var_3, var_4]
    var_18 = module_0.pdeque(var_17)
    var_19 = 7
    var_20 = [var_3, var_4, var_0, var_1, var_2]
    var_21 = module_0.pdeque(var_20)
    var_22 = [var_0, var_1, var_2, var_3, var_4]
    var_23 = module_0.pdeque(var_22)
    var_24 = -7
    var_25 = [var_2, var_3, var_4, var_0, var_1]
    var_26 = module_0.pdeque(var_25)
    var_27 = []
    var_28 = module_0.pdeque(var_27)
    var_29 = []
    var_30 = module_0.pdeque(var_29)
    var_31 = [var_0]
    var_32 = module_0.pdeque(var_31)
    var_33 = [var_0]
    var_34 = module_0.pdeque(var_33)
    var_35 = [var_0, var_1, var_2, var_3, var_4]
    var_36 = module_0.pdeque(var_35, var_3)
    var_37 = [var_3, var_4, var_0, var_1]
    var_38 = module_0.pdeque(var_37, var_3)
    var_39 = [var_0, var_1, var_2, var_3, var_4]
    var_40 = module_0.pdeque(var_39, var_3)
    var_41 = -2
    var_42 = [var_2, var_3, var_4, var_0]
    var_43 = module_0.pdeque(var_42, var_3)
    var_44 = 'All test cases passed!'
    var_45 = print(var_44)



# Parsed testcases at query #13
#--------------------------


import pyrsistent._plist as module_0


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.plist(var_3)
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = module_0.plist(var_8)
    var_10 = 6
    var_11 = 10
    var_12 = None
    var_13 = -1
    var_14 = 'invalid'
    var_15 = 0
    var_16 = module_0.plist()
    var_17 = module_0.plist()
    var_18 = var_10 + var_6
    var_19 = var_10 - var_1
    var_20 = 3.14
    var_21 = True
    var_22 = 10
    var_23 = var_22 ** var_7
    var_24 = 0
    var_25 = '10'
    var_26 = -5
    var_27 = 0
    var_28 = -1
    var_29 = 0
    var_30 = -10
    var_31 = -3
    var_32 = 1000
    var_33 = -100
    var_34 = 0
    var_35 = 0
    var_36 = 0
    var_37 = 1000
    var_38 = 0
    var_39 = 0
    var_40 = 0
    var_41 = 0
    var_42 = 0
    var_43 = 0
    var_44 = 0
    var_45 = 0
    var_46 = 0
    var_47 = 0
    var_48 = 0
    var_49 = 0
    var_50 = 0



# Parsed testcases at query #14
#--------------------------


import pyrsistent._pdeque as module_0


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = 4
    var_4 = 5
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = module_0.pdeque(var_5)
    var_7 = [var_1, var_2]
    var_8 = module_0.pdeque(var_7)
    var_9 = [var_0, var_1, var_2]
    var_10 = module_0.pdeque(var_9)
    var_11 = [var_2, var_3, var_4]
    var_12 = module_0.pdeque(var_11)
    var_13 = [var_0, var_2, var_4]
    var_14 = module_0.pdeque(var_13)
    var_15 = [var_2, var_3]
    var_16 = module_0.pdeque(var_15)
    var_17 = [var_1, var_2, var_3, var_4]
    var_18 = module_0.pdeque(var_17)
    var_19 = [var_1, var_3]
    var_20 = module_0.pdeque(var_19)
    var_21 = [var_0, var_3]
    var_22 = module_0.pdeque(var_21)
    var_23 = []
    var_24 = module_0.pdeque(var_23)
    var_25 = 10
    var_26 = var_6[var_25]
    var_27 = 'invalid'
    var_28 = var_6[var_27]
    var_29 = [var_27, var_28, var_2, var_3, var_4]
    var_30 = module_0.pdeque(var_29, var_2)
    var_31 = [var_2, var_3]
    var_32 = module_0.pdeque(var_31, var_2)
    var_33 = 42
    var_34 = [var_33]
    var_35 = module_0.pdeque(var_34)
    var_36 = []
    var_37 = module_0.pdeque(var_36)
    var_38 = 0
    var_39 = var_37[var_38]
    var_40 = 'All tests passed!'
    var_41 = print(var_40)



