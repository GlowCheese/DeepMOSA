####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import mimesis.providers.person as module_0


def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()
    var_2 = 'surnames'
    var_3 = [var_2]
    var_4 = [var_2]
    var_5 = [var_2]
    var_6 = 'ru'
    var_7 = module_0.Person()
    var_8 = 'male'
    var_9 = [var_2, var_8]
    var_10 = 'female'
    var_11 = [var_2, var_10]
    var_12 = 'en'
    var_13 = module_0.Person()
    var_14 = [var_2]
    var_15 = var_13.surname()
    var_16 = 'INVALID_GENDER'
    var_17 = var_0.surname(var_16)
    var_18 = 100
    var_19 = range(var_18)
    var_20 = 42
    var_21 = module_0.Person()
    var_22 = var_21.surname()
    var_23 = module_0.Person()
    var_24 = var_23.surname()
    var_25 = 'All tests passed for Person.surname()'
    var_26 = print(var_25)



####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------



def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.email()
    var_2 = 1
    var_3 = '@'
    var_4 = email.split(var_3)[var_2]
    var_5 = 'example.com'
    var_6 = 'test.org'
    var_7 = [var_5, var_6]
    var_8 = var_0.email(var_7)
    var_9 = email.split(var_3)[var_2]
    var_10 = True
    var_11 = var_0.email(unique=var_10)
    var_12 = True
    var_13 = var_0.email(unique=var_12)
    var_14 = set()
    var_15 = True
    var_16 = var_0.email(unique=var_15)
    var_17 = len(var_14)
    assert var_17 == 100
    var_18 = 42
    var_19 = module_0.Person()
    var_20 = True
    var_21 = var_19.email(unique=var_20)
    var_22 = var_0.email()
    var_23 = var_0.email()
    var_24 = 0
    var_25 = email.split(var_21)[var_24]
    var_26 = 'ru'
    var_27 = 10
    var_28 = range(var_27)
    var_29 = [person.email() for _ in var_28]
    var_30 = set(var_29)
    var_31 = len(var_30)
    var_32 = 'All tests passed!'
    var_33 = print(var_32)



# Parsed testcases at query #2
#--------------------------



def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()
    var_2 = 'nationality'
    var_3 = [var_2]
    var_4 = [var_2]
    var_5 = [var_2]
    var_6 = 'INVALID_GENDER'
    var_7 = var_0.nationality(var_6)



####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------



def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.patronymic()
    var_2 = 'invalid_gender'
    var_3 = var_0.patronymic(var_2)



# Parsed testcases at query #2
#--------------------------



def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.patronymic()
    var_2 = 'ru'
    var_3 = module_0.Person()
    var_4 = len(var_1)
    var_5 = 0
    var_6 = var_4 > var_5
    var_7 = 'uk'
    var_8 = module_0.Person()
    var_9 = len(var_1)
    var_10 = var_9 > var_5



# Parsed testcases at query #3
#--------------------------



def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.patronymic()
    assert var_1 is None
    var_2 = module_0.Person()
    var_3 = module_0.Person()
    var_4 = 'invalid'
    var_5 = var_3.patronymic(var_4)



# Parsed testcases at query #4
#--------------------------



def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()
    var_2 = 'surnames'
    var_3 = [var_2]
    var_4 = [var_2]
    var_5 = [var_2]
    var_6 = 'INVALID'
    var_7 = var_0.surname(var_6)
    var_8 = len(var_1)
    var_9 = [var_7]
    var_10 = 128



# Parsed testcases at query #5
#--------------------------


import uuid as module_1


def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.email()
    var_2 = 1
    var_3 = '@'
    var_4 = email.split(var_3)[var_2]
    var_5 = 'example.com'
    var_6 = 'test.org'
    var_7 = [var_5, var_6]
    var_8 = var_0.email(var_7)
    var_9 = email.split(var_3)[var_2]
    var_10 = True
    var_11 = var_0.email(unique=var_10)
    var_12 = True
    var_13 = var_0.email(unique=var_12)
    var_14 = 0
    var_15 = '@'
    var_16 = email1.split(var_15)[var_14]
    var_17 = 4
    var_18 = module_1.UUID(var_16, version=var_17)
    var_19 = True
    var_20 = False
    var_21 = 42
    var_22 = module_0.Person()
    var_23 = True
    var_24 = var_22.email(unique=var_23)



# Parsed testcases at query #6
#--------------------------



def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()
    var_2 = 'nationality'
    var_3 = [var_2]
    var_4 = [var_2]
    var_5 = [var_2]
    var_6 = 'INVALID_GENDER'
    var_7 = var_0.nationality(var_6)



# Parsed testcases at query #7
#--------------------------



def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()
    var_2 = 'invalid_gender'
    var_3 = var_0.nationality(var_2)



# Parsed testcases at query #8
#--------------------------



def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.email()
    var_2 = 0
    var_3 = '@'
    var_4 = email.split(var_3)[var_2]
    var_5 = 1
    var_6 = email.split(var_3)[var_5]
    var_7 = 'example.com'
    var_8 = 'test.org'
    var_9 = [var_7, var_8]
    var_10 = var_0.email(var_9)
    var_11 = email.split(var_3)[var_5]
    var_12 = True
    var_13 = var_0.email(unique=var_12)
    var_14 = True
    var_15 = var_0.email(unique=var_14)
    var_16 = 42
    var_17 = module_0.Person()
    var_18 = True
    var_19 = var_17.email(unique=var_18)
    var_20 = module_0.Person()
    var_21 = False
    var_22 = var_20.email(unique=var_21)
    var_23 = False
    var_24 = var_20.email(unique=var_23)
    var_25 = module_0.Person()
    var_26 = False
    var_27 = var_25.email(unique=var_26)
    var_28 = False
    var_29 = var_25.email(unique=var_28)
    var_30 = module_0.Person()
    var_31 = True
    var_32 = var_30.email(unique=var_31)
    var_33 = True
    var_34 = var_30.email(unique=var_33)
    var_35 = [var_7, var_8]
    var_36 = True
    var_37 = var_30.email(var_35, var_36)
    var_38 = email.split(var_19)[var_36]
    var_39 = email.split(var_19)[var_28]
    var_40 = email.split(var_19)[var_28]
    var_41 = 4
    var_42 = module_1.UUID(var_40, version=var_41)
    var_43 = var_42.hex
    var_44 = [var_7, var_8]
    var_45 = False
    var_46 = var_30.email(var_44, var_45)
    var_47 = email.split(var_19)[var_36]
    var_48 = email.split(var_19)[var_45]
    var_49 = True
    var_50 = var_30.email(unique=var_49)
    var_51 = email.split(var_19)[var_49]
    var_52 = email.split(var_19)[var_45]
    var_53 = email.split(var_19)[var_45]
    var_54 = module_1.UUID(var_53, version=var_41)
    var_55 = var_54.hex
    var_56 = False
    var_57 = var_30.email(unique=var_56)
    var_58 = email.split(var_19)[var_49]
    var_59 = email.split(var_19)[var_56]
    var_60 = []
    var_61 = True
    var_62 = var_30.email(var_60, var_61)
    var_63 = []
    var_64 = False
    var_65 = var_30.email(var_63, var_64)
    var_66 = [var_7]
    var_67 = True
    var_68 = var_30.email(var_66, var_67)
    var_69 = email.split(var_64)[var_67]
    assert var_69 == 'example.com'
    var_70 = email.split(var_64)[var_56]
    var_71 = email.split(var_64)[var_56]
    var_72 = module_1.UUID(var_71, version=var_41)
    var_73 = var_72.hex
    var_74 = [var_7]
    var_75 = False
    var_76 = var_30.email(var_74, var_75)
    var_77 = email.split(var_64)[var_67]
    assert var_77 == 'example.com'
    var_78 = email.split(var_64)[var_75]
    var_79 = [var_7]
    var_80 = True
    var_81 = var_30.email(var_79, var_80)
    var_82 = email.split(var_64)[var_80]
    assert var_82 == 'example.com'
    var_83 = email.split(var_64)[var_75]
    var_84 = email.split(var_64)[var_75]
    var_85 = module_1.UUID(var_84, version=var_41)
    var_86 = var_85.hex
    var_87 = [var_7]
    var_88 = False
    var_89 = var_30.email(var_87, var_88)
    var_90 = email.split(var_64)[var_80]
    assert var_90 == 'example.com'
    var_91 = email.split(var_64)[var_88]
    var_92 = '@example.com'
    var_93 = [var_92]
    var_94 = True
    var_95 = var_30.email(var_93, var_94)
    var_96 = email.split(var_64)[var_94]
    assert var_96 == 'example.com'
    var_97 = email.split(var_64)[var_88]
    var_98 = email.split(var_64)[var_88]
    var_99 = module_1.UUID(var_98, version=var_41)
    var_100 = var_99.hex
    var_101 = [var_92]
    var_102 = False
    var_103 = var_30.email(var_101, var_102)
    var_104 = email.split(var_64)[var_94]
    assert var_104 == 'example.com'
    var_105 = email.split(var_64)[var_102]
    var_106 = '@@example.com'
    var_107 = [var_106]
    var_108 = True
    var_109 = var_30.email(var_107, var_108)
    var_110 = email.split(var_64)[var_108]
    assert var_110 == '@example.com'
    var_111 = email.split(var_64)[var_102]
    var_112 = email.split(var_64)[var_102]
    var_113 = module_1.UUID(var_112, version=var_41)
    var_114 = var_113.hex
    var_115 = [var_106]
    var_116 = False
    var_117 = var_30.email(var_115, var_116)
    var_118 = email.split(var_64)[var_108]
    assert var_118 == '@example.com'
    var_119 = email.split(var_64)[var_116]
    var_120 = 'example.co.uk'
    var_121 = [var_120]
    var_122 = True
    var_123 = var_30.email(var_121, var_122)
    var_124 = email.split(var_64)[var_122]
    assert var_124 == 'example.co.uk'
    var_125 = email.split(var_64)[var_116]
    var_126 = email.split(var_64)[var_116]
    var_127 = module_1.UUID(var_126, version=var_41)
    var_128 = var_127.hex
    var_129 = [var_120]
    var_130 = False
    var_131 = var_30.email(var_129, var_130)
    var_132 = email.split(var_64)[var_122]
    assert var_132 == 'example.co.uk'
    var_133 = email.split(var_64)[var_130]
    var_134 = 'example123.com'
    var_135 = [var_134]
    var_136 = True
    var_137 = var_30.email(var_135, var_136)
    var_138 = email.split(var_64)[var_136]
    assert var_138 == 'example123.com'
    var_139 = email.split(var_64)[var_130]
    var_140 = email.split(var_64)[var_130]
    var_141 = module_1.UUID(var_140, version=var_41)
    var_142 = var_141.hex
    var_143 = [var_134]
    var_144 = False
    var_145 = var_30.email(var_143, var_144)
    var_146 = email.split(var_64)[var_136]
    assert var_146 == 'example123.com'
    var_147 = email.split(var_64)[var_144]
    var_148 = 'example-test.com'
    var_149 = [var_148]
    var_150 = True
    var_151 = var_30.email(var_149, var_150)
    var_152 = email.split(var_64)[var_150]
    assert var_152 == 'example-test.com'
    var_153 = email.split(var_64)[var_144]
    var_154 = email.split(var_64)[var_144]
    var_155 = module_1.UUID(var_154, version=var_41)
    var_156 = var_155.hex
    var_157 = [var_148]
    var_158 = False
    var_159 = var_30.email(var_157, var_158)
    var_160 = email.split(var_64)[var_150]
    assert var_160 == 'example-test.com'
    var_161 = email.split(var_64)[var_158]
    var_162 = 'example_test.com'
    var_163 = [var_162]
    var_164 = True
    var_165 = var_30.email(var_163, var_164)
    var_166 = email.split(var_64)[var_164]
    assert var_166 == 'example_test.com'
    var_167 = email.split(var_64)[var_158]
    var_168 = email.split(var_64)[var_158]
    var_169 = module_1.UUID(var_168, version=var_41)
    var_170 = var_169.hex
    var_171 = [var_162]
    var_172 = False
    var_173 = var_30.email(var_171, var_172)
    var_174 = email.split(var_64)[var_164]
    assert var_174 == 'example_test.com'
    var_175 = email.split(var_64)[var_172]
    var_176 = 'example test.com'
    var_177 = [var_176]
    var_178 = True
    var_179 = var_30.email(var_177, var_178)
    var_180 = email.split(var_64)[var_178]
    assert var_180 == 'example test.com'
    var_181 = email.split(var_64)[var_172]
    var_182 = email.split(var_64)[var_172]
    var_183 = module_1.UUID(var_182, version=var_41)
    var_184 = var_183.hex
    var_185 = [var_176]
    var_186 = False
    var_187 = var_30.email(var_185, var_186)
    var_188 = email.split(var_64)[var_178]
    assert var_188 == 'example test.com'
    var_189 = email.split(var_64)[var_186]
    var_190 = 'example test.co.uk'
    var_191 = [var_190]
    var_192 = True
    var_193 = var_30.email(var_191, var_192)



# Parsed testcases at query #9
#--------------------------



def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.patronymic()
    var_2 = 'en'
    var_3 = module_0.Person()
    var_4 = var_3.patronymic()
    assert var_4 is None



# Parsed testcases at query #10
#--------------------------



def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()
    var_2 = 'surnames'
    var_3 = [var_2]
    var_4 = [var_2]
    var_5 = [var_2]
    var_6 = 'invalid'
    var_7 = var_0.surname(var_6)
    var_8 = 'ru'
    var_9 = module_0.Person()
    var_10 = var_9.surname()
    var_11 = [var_7]
    var_12 = [var_7]
    var_13 = [var_7]
    var_14 = 'invalid'
    var_15 = var_9.surname(var_14)
    var_16 = var_9.surname()
    var_17 = [var_15]
    var_18 = [var_15]
    var_19 = [var_15]
    var_20 = var_9.surname()
    var_21 = [var_15]
    var_22 = [var_15]
    var_23 = [var_15]
    var_24 = var_9.surname()
    var_25 = [var_15]
    var_26 = [var_15]
    var_27 = [var_15]
    var_28 = var_9.surname()
    var_29 = [var_15]
    var_30 = [var_15]
    var_31 = [var_15]
    var_32 = var_9.surname()
    var_33 = [var_15]
    var_34 = [var_15]
    var_35 = [var_15]
    var_36 = var_9.surname()
    var_37 = [var_15]
    var_38 = [var_15]
    var_39 = [var_15]
    var_40 = var_9.surname()
    var_41 = [var_15]
    var_42 = [var_15]
    var_43 = [var_15]
    var_44 = var_9.surname()
    var_45 = [var_15]
    var_46 = [var_15]



# Parsed testcases at query #11
#--------------------------



def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.patronymic()
    assert var_1 is None
    var_2 = 'ru'
    var_3 = module_0.Person()
    var_4 = 'uk'
    var_5 = module_0.Person()
    var_6 = 'en'
    var_7 = module_0.Person()
    var_8 = 'invalid'
    var_9 = var_0.patronymic(var_8)



# Parsed testcases at query #12
#--------------------------



def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()
    var_2 = 'surnames'
    var_3 = [var_2]
    var_4 = [var_2]
    var_5 = [var_2]
    var_6 = 'invalid_gender'
    var_7 = var_0.surname(var_6)
    var_8 = 'ru'
    var_9 = module_0.Person()
    var_10 = var_9.surname()
    var_11 = [var_7]
    var_12 = [var_7]
    var_13 = [var_7]
    var_14 = None
    var_15 = var_9.surname(var_14)
    var_16 = [var_7]
    var_17 = 'invalid_gender'
    var_18 = var_9.surname(var_17)
    var_19 = 123
    var_20 = var_9.surname(var_19)
    var_21 = 1.23
    var_22 = var_9.surname(var_21)
    var_23 = 1
    var_24 = 2
    var_25 = 3
    var_26 = [var_23, var_24, var_25]
    var_27 = var_9.surname(var_26)
    var_28 = 'key'
    var_29 = 'value'
    var_30 = {var_28: var_29}
    var_31 = var_9.surname(var_30)
    var_32 = 1
    var_33 = 2
    var_34 = 3
    var_35 = (var_32, var_33, var_34)
    var_36 = var_9.surname(var_35)
    var_37 = 1
    var_38 = 2
    var_39 = 3
    var_40 = {var_37, var_38, var_39}
    var_41 = var_9.surname(var_40)
    var_42 = True
    var_43 = var_9.surname(var_42)
    var_44 = None
    var_45 = var_9.surname(var_44)
    var_46 = ''
    var_47 = var_9.surname(var_46)
    var_48 = ' '
    var_49 = var_9.surname(var_48)
    var_50 = '\n'
    var_51 = var_9.surname(var_50)
    var_52 = '\t'
    var_53 = var_9.surname(var_52)
    var_54 = '\r'
    var_55 = var_9.surname(var_54)
    var_56 = '\x0c'
    var_57 = var_9.surname(var_56)
    var_58 = '\x0b'
    var_59 = var_9.surname(var_58)
    var_60 = '\x08'
    var_61 = var_9.surname(var_60)
    var_62 = '\x00'
    var_63 = var_9.surname(var_62)
    var_64 = '\x1b'
    var_65 = var_9.surname(var_64)
    var_66 = '\x7f'
    var_67 = var_9.surname(var_66)
    var_68 = '\x80'
    var_69 = var_9.surname(var_68)
    var_70 = '©'
    var_71 = var_9.surname(var_70)
    var_72 = '😀'
    var_73 = var_9.surname(var_72)
    var_74 = '\ud83d\ude00'
    var_75 = var_9.surname(var_74)
    var_76 = b'\xff'
    var_77 = var_9.surname(var_76)
    var_78 = '\ud800'
    var_79 = var_9.surname(var_78)



# Parsed testcases at query #13
#--------------------------



def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.patronymic()
    var_2 = 'en'
    var_3 = module_0.Person()
    var_4 = var_3.patronymic()
    assert var_4 is None



# Parsed testcases at query #14
#--------------------------



def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.patronymic()
    var_2 = 'en'
    var_3 = module_0.Person()
    var_4 = var_3.patronymic()
    assert var_4 is None



# Parsed testcases at query #15
#--------------------------



def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()
    var_2 = 'nationality'
    var_3 = [var_2]
    var_4 = module_0.Person()
    var_5 = [var_2]
    var_6 = module_0.Person()
    var_7 = [var_2]
    var_8 = module_0.Person()
    var_9 = [var_2]
    var_10 = module_0.Person()
    var_11 = [var_2]
    var_12 = module_0.Person()
    var_13 = [var_2]
    var_14 = module_0.Person()
    var_15 = [var_2]
    var_16 = module_0.Person()
    var_17 = [var_2]
    var_18 = module_0.Person()
    var_19 = [var_2]
    var_20 = module_0.Person()
    var_21 = [var_2]
    var_22 = module_0.Person()
    var_23 = [var_2]
    var_24 = module_0.Person()
    var_25 = [var_2]
    var_26 = module_0.Person()
    var_27 = [var_2]
    var_28 = module_0.Person()
    var_29 = [var_2]
    var_30 = module_0.Person()
    var_31 = [var_2]
    var_32 = module_0.Person()
    var_33 = [var_2]
    var_34 = module_0.Person()
    var_35 = [var_2]
    var_36 = module_0.Person()
    var_37 = [var_2]
    var_38 = module_0.Person()
    var_39 = [var_2]
    var_40 = module_0.Person()
    var_41 = [var_2]
    var_42 = module_0.Person()
    var_43 = [var_2]
    var_44 = module_0.Person()
    var_45 = [var_2]
    var_46 = module_0.Person()
    var_47 = [var_2]
    var_48 = module_0.Person()
    var_49 = [var_2]
    var_50 = module_0.Person()
    var_51 = [var_2]
    var_52 = module_0.Person()
    var_53 = [var_2]
    var_54 = module_0.Person()
    var_55 = [var_2]
    var_56 = module_0.Person()
    var_57 = [var_2]
    var_58 = module_0.Person()
    var_59 = [var_2]
    var_60 = module_0.Person()
    var_61 = [var_2]
    var_62 = module_0.Person()
    var_63 = [var_2]
    var_64 = module_0.Person()
    var_65 = var_1



# Parsed testcases at query #16
#--------------------------



def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()
    var_2 = 'surnames'
    var_3 = [var_2]
    var_4 = 'male'
    var_5 = [var_2, var_4]
    var_6 = 'female'
    var_7 = [var_2, var_6]
    var_8 = 'invalid_gender'
    var_9 = var_0.surname(var_8)
    var_10 = 'ru'
    var_11 = module_0.Person()
    var_12 = [var_9, var_4]
    var_13 = 'en'
    var_14 = module_0.Person()
    var_15 = var_14.surname()
    var_16 = [var_9]
    var_17 = 42
    var_18 = module_0.Person()
    var_19 = var_18.surname()
    var_20 = var_18.surname()
    var_21 = 'de'
    var_22 = 'fr'
    var_23 = 'es'
    var_24 = [var_13, var_10, var_21, var_22, var_23]
    var_25 = var_0.surname()
    var_26 = 'surnames'
    var_27 = [var_26]
    var_28 = 'surnames'
    var_29 = [var_28]
    var_30 = module_0.Person()
    var_31 = None
    var_32 = var_30.surname(var_31)
    var_33 = [var_28]
    var_34 = []
    var_35 = module_0.Person()
    var_36 = var_35._extract
    var_37 = []
    var_38 = var_35.surname()
    var_39 = module_0.Person()
    var_40 = var_39._extract
    var_41 = 'Smith'
    var_42 = [var_41]
    var_43 = var_39.surname()
    assert var_43 == 'Smith'
    var_44 = module_0.Person()
    var_45 = var_44._extract
    var_46 = 'Johnson'
    var_47 = [var_41, var_46]
    var_48 = []
    var_49 = {var_4: var_47, var_6: var_48}
    var_50 = var_44.surname(var_8)
    var_51 = 'invalid'
    var_52 = var_30.surname(var_28)
    var_53 = 'male'
    var_54 = var_30.surname(var_53)
    var_55 = 1
    var_56 = var_30.surname(var_55)
    var_57 = var_30.surname(var_31)
    var_58 = module_0.Person()
    var_59 = var_58.surname()
    var_60 = var_58.surname()
    var_61 = len(var_60)
    var_62 = var_58.surname()
    var_63 = [var_56]
    var_64 = []
    var_65 = var_58.surname()
    var_66 = var_58.surname()
    var_67 = 123
    var_68 = module_0.Person()
    var_69 = module_0.Person()
    var_70 = var_68.surname()
    var_71 = var_69.surname()
    var_72 = module_0.Person()
    var_73 = 456
    var_74 = module_0.Person()
    var_75 = var_58.surname()
    var_76 = var_58.surname()
    var_77 = len(var_62)
    var_78 = 'surnames'
    var_79 = [var_78]
    var_80 = 'invalid'
    var_81 = var_58.surname(var_80)
    var_82 = module_0.Person()
    var_83 = 100
    var_84 = range(var_83)
    var_85 = 999
    var_86 = module_0.Person()
    var_87 = 10
    var_88 = range(var_87)



# Parsed testcases at query #17
#--------------------------



def test_case_0():
    var_0 = module_0.Person()
    var_1 = None
    var_2 = var_0.patronymic(var_1)
    assert var_2 is None
    var_3 = 'ru'
    var_4 = module_0.Person()
    var_5 = 'uk'
    var_6 = module_0.Person()
    var_7 = 'en'
    var_8 = module_0.Person()
    var_9 = 'invalid'
    var_10 = var_0.patronymic(var_9)
    var_11 = 42
    var_12 = module_0.Person()
    var_13 = module_0.Person()
    var_14 = 'patronymic'
    var_15 = str(var_10)
    var_16 = [var_14, var_15]
    var_17 = []
    var_18 = module_0.Person()
    var_19 = module_0.Person()
    var_20 = 'patronymic'
    var_21 = []
    var_22 = 123
    var_23 = module_0.Person()
    var_24 = module_0.Person()
    var_25 = module_0.Person()
    var_26 = 456
    var_27 = module_0.Person()
    var_28 = module_0.Person()
    var_29 = var_28.patronymic()
    assert var_29 is None
    var_30 = module_0.Person()
    var_31 = var_30.patronymic()
    assert var_31 is None
    var_32 = []
    var_33 = var_30.patronymic()
    var_34 = var_6.patronymic()
    var_35 = module_0.Person()
    var_36 = var_35.patronymic()
    var_37 = module_0.Person()
    var_38 = var_37.patronymic()
    var_39 = module_0.Person()
    var_40 = var_39.patronymic()
    var_41 = module_0.Person()
    var_42 = var_41.patronymic()
    var_43 = module_0.Person()
    var_44 = module_0.Person()
    var_45 = []
    var_46 = module_0.Person()
    var_47 = module_0.Person()
    var_48 = module_0.Person()
    var_49 = module_0.Person()
    var_50 = module_0.Person()
    var_51 = var_50.patronymic()
    assert var_51 is None
    var_52 = module_0.Person()
    var_53 = var_52.patronymic()
    var_54 = []
    var_55 = var_52.patronymic()
    var_56 = var_6.patronymic()
    var_57 = module_0.Person()



# Parsed testcases at query #18
#--------------------------



def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()
    var_2 = 'surnames'
    var_3 = [var_2]
    var_4 = [var_2]
    var_5 = [var_2]
    var_6 = 'INVALID_GENDER'
    var_7 = var_0.surname(var_6)
    var_8 = 'ru'
    var_9 = module_0.Person()
    var_10 = var_9.surname()
    var_11 = [var_7]
    var_12 = [var_7]
    var_13 = [var_7]
    var_14 = 'INVALID_GENDER'
    var_15 = var_9.surname(var_14)
    var_16 = var_9.surname()
    var_17 = [var_15]
    var_18 = [var_15]
    var_19 = [var_15]
    var_20 = var_9.surname()
    var_21 = [var_15]
    var_22 = [var_15]
    var_23 = [var_15]
    var_24 = var_9.surname()
    var_25 = [var_15]
    var_26 = [var_15]
    var_27 = [var_15]
    var_28 = var_9.surname()
    var_29 = [var_15]
    var_30 = [var_15]
    var_31 = [var_15]
    var_32 = var_9.surname()
    var_33 = [var_15]
    var_34 = [var_15]
    var_35 = [var_15]
    var_36 = var_9.surname()
    var_37 = [var_15]
    var_38 = [var_15]
    var_39 = [var_15]
    var_40 = var_9.surname()
    var_41 = [var_15]
    var_42 = [var_15]
    var_43 = [var_15]
    var_44 = var_9.surname()
    var_45 = [var_15]



####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------



def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()
    var_2 = 'invalid_gender'
    var_3 = var_0.nationality(var_2)



# Parsed testcases at query #2
#--------------------------



def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()
    var_2 = 'surnames'
    var_3 = [var_2]
    var_4 = [var_2]
    var_5 = [var_2]
    var_6 = 'INVALID'
    var_7 = var_0.surname(var_6)
    var_8 = 'ru'
    var_9 = module_0.Person()
    var_10 = var_9.surname()
    var_11 = [var_7]
    var_12 = [var_7]
    var_13 = [var_7]
    var_14 = None
    var_15 = var_9.surname(var_14)
    var_16 = [var_7]
    var_17 = 'INVALID'
    var_18 = var_9.surname(var_17)
    var_19 = var_9.surname(var_17)
    var_20 = var_9.surname(var_17)
    var_21 = None
    var_22 = var_9.surname(var_21)
    var_23 = 'INVALID'
    var_24 = var_9.surname(var_23)
    var_25 = var_9.surname(var_23)
    var_26 = var_9.surname(var_23)
    var_27 = None
    var_28 = var_9.surname(var_27)
    var_29 = 'INVALID'
    var_30 = var_9.surname(var_29)
    var_31 = var_9.surname(var_29)
    var_32 = var_9.surname(var_29)
    var_33 = None
    var_34 = var_9.surname(var_33)
    var_35 = 'INVALID'
    var_36 = var_9.surname(var_35)
    var_37 = var_9.surname(var_35)
    var_38 = var_9.surname(var_35)
    var_39 = None
    var_40 = var_9.surname(var_39)
    var_41 = 'INVALID'
    var_42 = var_9.surname(var_41)
    var_43 = var_9.surname(var_41)
    var_44 = var_9.surname(var_41)
    var_45 = None
    var_46 = var_9.surname(var_45)
    var_47 = 'INVALID'
    var_48 = var_9.surname(var_47)
    var_49 = var_9.surname(var_47)
    var_50 = var_9.surname(var_47)
    var_51 = None
    var_52 = var_9.surname(var_51)
    var_53 = 'INVALID'
    var_54 = var_9.surname(var_53)
    var_55 = var_9.surname(var_53)



# Parsed testcases at query #3
#--------------------------



def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()
    var_2 = 'nationality'
    var_3 = [var_2]
    var_4 = [var_2]
    var_5 = [var_2]
    var_6 = 'INVALID_GENDER'
    var_7 = var_0.nationality(var_6)
    var_8 = 'ru'
    var_9 = module_0.Person()
    var_10 = var_9.nationality()
    var_11 = [var_7]
    var_12 = [var_7]
    var_13 = [var_7]
    var_14 = 'INVALID_GENDER'
    var_15 = var_9.nationality(var_14)
    var_16 = var_9.nationality()
    var_17 = [var_15]
    var_18 = None
    var_19 = var_9.nationality(var_18)



# Parsed testcases at query #4
#--------------------------



def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()
    var_2 = 'surnames'
    var_3 = [var_2]
    var_4 = [var_2]
    var_5 = [var_2]
    var_6 = 'ru'
    var_7 = module_0.Person()
    var_8 = var_7.surname()
    var_9 = [var_2]
    var_10 = 'male'
    var_11 = []
    var_12 = var_8 in var_3
    var_13 = 'female'
    var_14 = []
    var_15 = var_8 in var_4
    var_16 = 10
    var_17 = range(var_16)
    var_18 = {person.surname() for _ in var_17}
    var_19 = len(var_18)
    var_20 = 42
    var_21 = module_0.Person()
    var_22 = var_21.surname()
    var_23 = module_0.Person()
    var_24 = var_23.surname()
    var_25 = 'All tests passed for Person.surname()'
    var_26 = print(var_25)



# Parsed testcases at query #5
#--------------------------



def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()
    var_2 = 'invalid_gender'
    var_3 = var_0.nationality(var_2)



# Parsed testcases at query #6
#--------------------------



def test_case_0():
    var_0 = module_0.Person()
    var_1 = 10
    var_2 = var_0.password(var_1)
    var_3 = len(var_2)
    assert var_3 == 10
    var_4 = 8
    var_5 = True
    var_6 = var_0.password(var_4, var_5)
    var_7 = len(var_6)
    assert var_7 == 64
    var_8 = 12
    var_9 = var_0.password(var_8)
    var_10 = var_0.password(var_8)
    var_11 = var_0.password(var_1, var_5)
    var_12 = var_0.password(var_1, var_5)
    var_13 = var_0.password(var_5)
    var_14 = len(var_13)
    assert var_14 == 1
    var_15 = 100
    var_16 = var_0.password(var_15)
    var_17 = len(var_16)
    assert var_17 == 100
    var_18 = var_0.password(var_15)
    var_19 = 'test_password'
    var_20 = 123
    var_21 = module_0.Person()
    var_22 = len(var_19)
    var_23 = var_21.password(var_22, var_5)
    var_24 = module_0.Person()
    var_25 = 456
    var_26 = module_0.Person()
    var_27 = var_24.password(var_1)
    var_28 = var_26.password(var_1)
    var_29 = 'All tests passed!'
    var_30 = print(var_29)



# Parsed testcases at query #7
#--------------------------



def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()
    var_2 = 'nationality'
    var_3 = [var_2]
    var_4 = [var_2]
    var_5 = [var_2]
    var_6 = 'invalid_gender'
    var_7 = var_0.nationality(var_6)



# Parsed testcases at query #8
#--------------------------



def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.password()
    var_2 = len(var_1)
    assert var_2 == 8
    var_3 = 12
    var_4 = var_0.password(var_3)
    var_5 = len(var_4)
    assert var_5 == 12
    var_6 = True
    var_7 = var_0.password(hashed=var_6)
    var_8 = len(var_7)
    assert var_8 == 64
    var_9 = 10
    var_10 = var_0.password(var_9, var_6)
    var_11 = len(var_10)
    assert var_11 == 64
    var_12 = 100
    var_13 = var_0.password(var_12)
    var_14 = set()
    var_15 = var_0.password()
    var_16 = len(var_14)
    assert var_16 == 100
    var_17 = set()
    var_18 = True
    var_19 = var_0.password(hashed=var_18)
    var_20 = len(var_17)
    assert var_20 == 100
    var_21 = var_0.password(var_6)
    var_22 = len(var_21)
    assert var_22 == 1
    var_23 = 1000
    var_24 = var_0.password(var_23)
    var_25 = len(var_24)
    assert var_25 == 1000
    var_26 = -1
    var_27 = var_0.password(var_26)
    var_28 = 0
    var_29 = var_0.password(var_28)
    var_30 = 'invalid'
    var_31 = var_0.password(var_30)
    var_32 = 'invalid'
    var_33 = var_0.password(hashed=var_32)
    var_34 = 16
    var_35 = var_0.password(var_34, var_6)
    var_36 = len(var_35)
    assert var_36 == 64
    var_37 = var_0.password(hashed=var_6)
    var_38 = len(var_37)
    assert var_38 == 64
    var_39 = 20
    var_40 = var_0.password(var_39)
    var_41 = len(var_40)
    assert var_41 == 20
    var_42 = 32
    var_43 = var_0.password(var_42, var_6)
    var_44 = len(var_43)
    assert var_44 == 64
    var_45 = 24
    var_46 = False
    var_47 = var_0.password(var_45, var_46)
    var_48 = len(var_47)
    assert var_48 == 24
    var_49 = var_0.password(var_45, var_46)
    var_50 = len(var_49)
    assert var_50 == 24
    var_51 = var_0.password(var_45, var_46)
    var_52 = len(var_51)
    assert var_52 == 24
    var_53 = var_0.password(var_45, var_46)
    var_54 = len(var_53)
    assert var_54 == 24



# Parsed testcases at query #9
#--------------------------



def test_case_0():
    var_0 = module_0.Person()
    var_1 = 10
    var_2 = var_0.password(var_1)
    var_3 = len(var_2)
    assert var_3 == 10
    var_4 = 8
    var_5 = True
    var_6 = var_0.password(var_4, var_5)
    var_7 = len(var_6)
    assert var_7 == 64
    var_8 = 20
    var_9 = var_0.password(var_8)
    var_10 = var_0.password()
    var_11 = var_0.password()
    var_12 = 15
    var_13 = var_0.password(var_12)
    var_14 = len(var_13)
    assert var_14 == 15
    var_15 = var_0.password(hashed=var_5)
    var_16 = len(var_15)
    assert var_16 == 64
    var_17 = False
    var_18 = var_0.password(hashed=var_17)
    var_19 = len(var_18)
    assert var_19 == 8
    var_20 = var_0.password(var_17)
    var_21 = len(var_20)
    assert var_21 == 0
    var_22 = -5
    var_23 = var_0.password(var_22)
    var_24 = 1000
    var_25 = var_0.password(var_24)
    var_26 = len(var_25)
    assert var_26 == 1000



# Parsed testcases at query #10
#--------------------------



def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.password()
    var_2 = len(var_1)
    assert var_2 == 8
    var_3 = 12
    var_4 = var_0.password(var_3)
    var_5 = len(var_4)
    assert var_5 == 12
    var_6 = True
    var_7 = var_0.password(hashed=var_6)
    var_8 = len(var_7)
    assert var_8 == 64
    var_9 = '0123456789abcdef'
    var_10 = 10
    var_11 = var_0.password(var_10, var_6)
    var_12 = len(var_11)
    assert var_12 == 64
    var_13 = 100
    var_14 = var_0.password(var_13)
    var_15 = range(var_13)
    var_16 = 42
    var_17 = module_0.Person()
    var_18 = module_0.Person()
    var_19 = var_17.password()
    var_20 = var_18.password()
    var_21 = var_17.password(hashed=var_6)
    var_22 = var_18.password(hashed=var_6)
    var_23 = var_0.password(var_6)
    var_24 = len(var_23)
    assert var_24 == 1
    var_25 = 123
    var_26 = module_0.Person()
    var_27 = var_26.password(hashed=var_6)
    var_28 = module_0.Person()
    var_29 = var_28.password(hashed=var_6)
    var_30 = module_0.Person()
    var_31 = var_30.password()
    var_32 = var_30.password(hashed=var_6)
    var_33 = len(var_31)
    assert var_33 == 8
    var_34 = len(var_32)
    assert var_34 == 64



# Parsed testcases at query #11
#--------------------------



def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()
    var_2 = 'surnames'
    var_3 = [var_2]
    var_4 = [var_2]
    var_5 = [var_2]
    var_6 = 'INVALID'
    var_7 = var_0.surname(var_6)
    var_8 = 'ru'
    var_9 = module_0.Person()
    var_10 = 'male'
    var_11 = [var_7, var_10]
    var_12 = 'female'
    var_13 = [var_7, var_12]
    var_14 = 'en'
    var_15 = module_0.Person()
    var_16 = var_15.surname()
    var_17 = [var_7]
    var_18 = 42
    var_19 = module_0.Person()
    var_20 = var_19.surname()
    var_21 = module_0.Person()
    var_22 = var_21.surname()



# Parsed testcases at query #12
#--------------------------



def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.patronymic()
    var_2 = 'en'
    var_3 = module_0.Person()
    var_4 = var_3.patronymic()
    assert var_4 is None



# Parsed testcases at query #13
#--------------------------



def test_case_0():
    var_0 = module_0.Person()
    var_1 = None
    var_2 = var_0.patronymic(var_1)
    assert var_2 is None
    var_3 = 'ru'
    var_4 = module_0.Person()
    var_5 = 'uk'
    var_6 = module_0.Person()
    var_7 = 'en'
    var_8 = module_0.Person()
    var_9 = 'fr'
    var_10 = module_0.Person()
    var_11 = 'de'
    var_12 = module_0.Person()
    var_13 = 'it'
    var_14 = module_0.Person()
    var_15 = 'es'
    var_16 = module_0.Person()
    var_17 = 'pt'
    var_18 = module_0.Person()
    var_19 = 'pl'
    var_20 = module_0.Person()
    var_21 = 'nl'
    var_22 = module_0.Person()
    var_23 = 'sv'
    var_24 = module_0.Person()
    var_25 = 'da'
    var_26 = module_0.Person()
    var_27 = 'no'
    var_28 = module_0.Person()
    var_29 = 'fi'
    var_30 = module_0.Person()
    var_31 = 'cs'
    var_32 = module_0.Person()
    var_33 = 'hu'
    var_34 = module_0.Person()
    var_35 = 'ro'
    var_36 = module_0.Person()
    var_37 = 'bg'
    var_38 = module_0.Person()
    var_39 = 'el'
    var_40 = module_0.Person()
    var_41 = 'tr'
    var_42 = module_0.Person()
    var_43 = 'he'
    var_44 = module_0.Person()
    var_45 = 'ar'
    var_46 = module_0.Person()
    var_47 = 'fa'
    var_48 = module_0.Person()
    var_49 = 'hi'
    var_50 = module_0.Person()
    var_51 = 'th'
    var_52 = module_0.Person()
    var_53 = 'ko'
    var_54 = module_0.Person()
    var_55 = 'ja'
    var_56 = module_0.Person()
    var_57 = 'zh'
    var_58 = module_0.Person()
    var_59 = 'vi'
    var_60 = module_0.Person()
    var_61 = 'id'
    var_62 = module_0.Person()
    var_63 = 'ms'
    var_64 = module_0.Person()
    var_65 = 'fil'
    var_66 = module_0.Person()
    var_67 = 'sw'
    var_68 = module_0.Person()
    var_69 = 'af'
    var_70 = module_0.Person()
    var_71 = 'zu'
    var_72 = module_0.Person()
    var_73 = 'xh'
    var_74 = module_0.Person()
    var_75 = 'nso'
    var_76 = module_0.Person()
    var_77 = 'tn'
    var_78 = module_0.Person()
    var_79 = 'st'
    var_80 = module_0.Person()
    var_81 = 'ts'
    var_82 = module_0.Person()
    var_83 = 'ss'
    var_84 = module_0.Person()
    var_85 = 've'
    var_86 = module_0.Person()
    var_87 = 'nr'
    var_88 = module_0.Person()



# Parsed testcases at query #14
#--------------------------



def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()
    var_2 = 'invalid_gender'
    var_3 = var_0.nationality(var_2)



# Parsed testcases at query #15
#--------------------------



def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()
    var_2 = 'surnames'
    var_3 = [var_2]
    var_4 = 'male'
    var_5 = [var_2, var_4]
    var_6 = 'female'
    var_7 = [var_2, var_6]
    var_8 = 'invalid'
    var_9 = var_0.surname(var_8)



# Parsed testcases at query #16
#--------------------------



def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()
    var_2 = 'nationality'
    var_3 = [var_2]
    var_4 = [var_2]
    var_5 = [var_2]
    var_6 = 'invalid_gender'
    var_7 = var_0.nationality(var_6)



# Parsed testcases at query #17
#--------------------------



def test_case_0():
    var_0 = module_0.Person()
    var_1 = None
    var_2 = var_0.patronymic(var_1)
    assert var_2 is None
    var_3 = 'ru'
    var_4 = module_0.Person()
    var_5 = 'uk'
    var_6 = module_0.Person()
    var_7 = 'en'
    var_8 = module_0.Person()



# Parsed testcases at query #18
#--------------------------


import builtins as module_1


def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()
    var_2 = 'nationality'
    var_3 = [var_2]
    var_4 = 'male'
    var_5 = [var_2, var_4]
    var_6 = 'female'
    var_7 = [var_2, var_6]
    var_8 = 'invalid_gender'
    var_9 = var_0.nationality(var_8)
    var_10 = 'ru'
    var_11 = module_0.Person()
    var_12 = var_11.nationality()
    var_13 = [var_9]
    var_14 = [var_9, var_4]
    var_15 = [var_9, var_6]
    var_16 = var_11.nationality()
    var_17 = [var_9]
    var_18 = 'invalid_gender'
    var_19 = var_11.nationality(var_18)
    var_20 = 123
    var_21 = var_11.nationality(var_20)
    var_22 = []
    var_23 = var_11.nationality(var_22)
    var_24 = {}
    var_25 = var_11.nationality(var_24)
    var_26 = ()
    var_27 = var_11.nationality(var_26)
    var_28 = set()
    var_29 = var_11.nationality(var_28)
    var_30 = frozenset()
    var_31 = var_11.nationality(var_30)
    var_32 = module_1.object()
    var_33 = var_11.nationality(var_32)



# Parsed testcases at query #19
#--------------------------


import re as module_1


def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.email()
    var_2 = 1
    var_3 = '@'
    var_4 = email.split(var_3)[var_2]
    var_5 = 'example.com'
    var_6 = 'test.org'
    var_7 = [var_5, var_6]
    var_8 = var_0.email(var_7)
    var_9 = email.split(var_3)[var_2]
    var_10 = True
    var_11 = var_0.email(unique=var_10)
    var_12 = True
    var_13 = var_0.email(unique=var_12)
    var_14 = 42
    var_15 = module_0.Person()
    var_16 = True
    var_17 = var_15.email(unique=var_16)
    var_18 = var_0.email()
    var_19 = '^[a-zA-Z0-9]+@[a-zA-Z0-9]+\\.[a-zA-Z]{2,}$'
    var_20 = module_1.match(var_19, var_18)
    var_21 = []
    var_22 = var_0.email(var_21)
    var_23 = 'single.com'
    var_24 = [var_23]
    var_25 = var_0.email(var_24)
    var_26 = '@single.com'
    var_27 = var_0.email()
    var_28 = 0
    var_29 = email.split(var_17)[var_28]
    var_30 = len(var_29)



# Parsed testcases at query #20
#--------------------------



def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()
    var_2 = 'invalid_gender'
    var_3 = var_0.nationality(var_2)
    var_4 = 'All tests passed for method nationality of class Person'
    var_5 = print(var_4)



# Parsed testcases at query #21
#--------------------------



def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()
    var_2 = 'nationality'
    var_3 = [var_2]
    var_4 = 'male'
    var_5 = [var_2, var_4]
    var_6 = 'female'
    var_7 = [var_2, var_6]
    var_8 = 'invalid_gender'
    var_9 = var_0.nationality(var_8)



# Parsed testcases at query #22
#--------------------------


import builtins as module_1


def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()
    var_2 = 'surnames'
    var_3 = [var_2]
    var_4 = [var_2]
    var_5 = [var_2]
    var_6 = None
    var_7 = var_0.surname(var_6)
    var_8 = [var_2]
    var_9 = 'INVALID'
    var_10 = var_0.surname(var_9)
    var_11 = 'ru'
    var_12 = module_0.Person()
    var_13 = var_12.surname()
    var_14 = [var_10]
    var_15 = [var_10]
    var_16 = [var_10]
    var_17 = var_12.surname(var_6)
    var_18 = [var_10]
    var_19 = 'INVALID'
    var_20 = var_12.surname(var_19)
    var_21 = 'MALE'
    var_22 = var_12.surname(var_21)
    var_23 = 1
    var_24 = var_12.surname(var_23)
    var_25 = 1.0
    var_26 = var_12.surname(var_25)
    var_27 = True
    var_28 = var_12.surname(var_27)
    var_29 = []
    var_30 = var_12.surname(var_29)
    var_31 = {}
    var_32 = var_12.surname(var_31)
    var_33 = ()
    var_34 = var_12.surname(var_33)
    var_35 = set()
    var_36 = var_12.surname(var_35)
    var_37 = frozenset()
    var_38 = var_12.surname(var_37)
    var_39 = b''
    var_40 = var_12.surname(var_39)
    var_41 = bytearray()
    var_42 = var_12.surname(var_41)
    var_43 = b''
    var_44 = memoryview(var_43)
    var_45 = var_12.surname(var_44)
    var_46 = complex()
    var_47 = var_12.surname(var_46)
    var_48 = 0
    var_49 = range(var_48)
    var_50 = var_12.surname(var_49)
    var_51 = 0
    var_52 = slice(var_51)
    var_53 = var_12.surname(var_52)
    var_54 = module_1.object()
    var_55 = var_12.surname(var_54)
    var_56 = None
    var_57 = lambda : var_56
    var_58 = var_12.surname(var_57)
    var_59 = module_0.Person()
    var_60 = var_12.surname(var_59)
    var_61 = 'sys'
    var_62 = __import__(var_61)
    var_63 = var_12.surname(var_62)
    var_64 = 0
    var_65 = range(var_64)
    var_66 = var_12.surname(var_63)
    var_67 = var_12.surname(var_64)
    var_68 = var_12.surname(var_64)
    var_69 = var_12.surname(var_64)
    var_70 = var_12.surname(var_64)
    var_71 = var_12.surname(var_64)
    var_72 = var_12.surname(var_64)
    var_73 = var_12.surname(var_64)
    var_74 = var_12.surname(var_64)



