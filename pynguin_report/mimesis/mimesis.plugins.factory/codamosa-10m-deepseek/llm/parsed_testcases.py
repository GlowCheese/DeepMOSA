####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import mimesis.plugins.factory as module_0


def test_case_0():
    var_0 = 'name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 10
    var_3 = module_0.FactoryField(var_0)
    var_4 = True
    var_5 = 'ru'
    var_6 = module_0.FactoryField(var_0, var_5)
    var_7 = 'name'
    var_8 = 'invalid'
    var_9 = module_0.FactoryField(var_7, var_8)
    var_10 = ''
    var_11 = module_0.FactoryField(var_10)
    var_12 = 'name_with_underscore'
    var_13 = module_0.FactoryField(var_12)
    var_14 = '123'
    var_15 = module_0.FactoryField(var_14)
    var_16 = 'name with spaces'
    var_17 = module_0.FactoryField(var_16)
    var_18 = 'name-with-hyphen'
    var_19 = module_0.FactoryField(var_18)
    var_20 = 'name_with_émojis'
    var_21 = module_0.FactoryField(var_20)
    var_22 = 'a'
    var_23 = module_0.FactoryField(var_22)
    var_24 = 1000
    var_25 = var_22 * var_24
    var_26 = module_0.FactoryField(var_25)
    var_27 = 'True'
    var_28 = module_0.FactoryField(var_27)
    var_29 = '42'
    var_30 = module_0.FactoryField(var_29)
    var_31 = '3.14'
    var_32 = module_0.FactoryField(var_31)
    var_33 = '[1, 2, 3]'
    var_34 = module_0.FactoryField(var_33)
    var_35 = "{'key': 'value'}"
    var_36 = module_0.FactoryField(var_35)
    var_37 = '(1, 2, 3)'
    var_38 = module_0.FactoryField(var_37)
    var_39 = '{1, 2, 3}'
    var_40 = module_0.FactoryField(var_39)
    var_41 = 'frozenset([1, 2, 3])'
    var_42 = module_0.FactoryField(var_41)
    var_43 = b'bytes'
    var_44 = module_0.FactoryField(var_43)
    var_45 = bytearray(var_43)
    var_46 = module_0.FactoryField(var_45)
    var_47 = bytearray(var_43)
    var_48 = memoryview(var_43)
    var_49 = module_0.FactoryField(var_48)
    var_50 = memoryview(var_43)
    var_51 = 2
    var_52 = complex(var_4, var_51)
    var_53 = module_0.FactoryField(var_52)
    var_54 = complex(var_4, var_51)
    var_55 = range(var_2)
    var_56 = module_0.FactoryField(var_55)
    var_57 = range(var_2)
    var_58 = slice(var_4, var_2, var_51)
    var_59 = module_0.FactoryField(var_58)
    var_60 = slice(var_4, var_2, var_51)
    var_61 = lambda x: x ** var_51
    var_62 = module_0.FactoryField(var_61)



# Parsed testcases at query #2
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = 10
    var_2 = module_0.FactoryField(var_0)
    var_3 = module_0.FactoryField(var_0)
    var_4 = 'Mr.'
    var_5 = None
    var_6 = module_0.FactoryField(var_0, var_5)
    var_7 = 'All tests passed!'
    var_8 = print(var_7)



# Parsed testcases at query #3
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 10
    var_3 = module_0.FactoryField(var_0)



# Parsed testcases at query #4
#--------------------------



def test_case_0():
    var_0 = 'field_handlers'
    var_1 = []
    var_2 = {var_0: var_1}
    var_3 = 'name'
    var_4 = module_0.FactoryField(var_3)
    var_5 = '\ud800'
    var_6 = '\udfff'
    var_7 = 128
    var_8 = 'R'
    var_9 = 'AL'
    var_10 = 'RLE'
    var_11 = 'RLO'
    var_12 = (var_8, var_9, var_10, var_11)
    var_13 = 'L'
    var_14 = 'NFD'



# Parsed testcases at query #5
#--------------------------


import factory.builder as module_0
import mimesis.plugins.factory as module_1


def test_case_0():
    var_0 = module_0.Resolver()
    var_1 = module_0.BuildStep()
    var_2 = 'name'
    var_3 = module_1.FactoryField(var_2)
    var_4 = var_3.evaluate(var_0, var_1)



# Parsed testcases at query #6
#--------------------------


import mimesis.plugins.factory as module_0


def test_case_0():
    var_0 = 'person.full_name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 'male'
    var_3 = module_0.FactoryField(var_0)
    var_4 = 'female'



# Parsed testcases at query #7
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 10
    var_3 = module_0.FactoryField(var_0)



# Parsed testcases at query #8
#--------------------------


import factory.builder as module_0


def test_case_0():
    var_0 = module_0.Resolver()
    var_1 = module_0.BuildStep()
    var_2 = 'extra_param'
    var_3 = 'extra_value'
    var_4 = {var_2: var_3}
    var_5 = 'test_field'
    var_6 = 'value1'



# Parsed testcases at query #9
#--------------------------



def test_case_0():
    var_0 = module_0.Resolver()
    var_1 = module_0.BuildStep()
    var_2 = 'field_handlers'
    var_3 = []
    var_4 = 'person.full_name'
    var_5 = 'unique'
    var_6 = True
    var_7 = {var_5: var_6}
    var_8 = 'All tests passed!'
    var_9 = print(var_8)



# Parsed testcases at query #10
#--------------------------



def test_case_0():
    var_0 = module_0.Resolver()
    var_1 = module_0.BuildStep()
    var_2 = 'name'



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = {}
    var_1 = 'person.full_name'
    var_2 = 128
    var_3 = '<[^>]+>'
    var_4 = '&[^;]+;'
    var_5 = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\\\(\\\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    var_6 = '\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b'
    var_7 = '(?i)(\\b(select|insert|update|delete|drop|alter|create|truncate)\\b)'
    var_8 = '(?i)(\\b(onload|onerror|onclick|onmouseover|onmouseout|onkeydown|onkeyup|onkeypress)\\b)'
    var_9 = '(\\.\\./|\\.\\.\\\\)'
    var_10 = '(?i)(\\b(cmd|powershell|bash|sh|python|perl|ruby|php|java|javascript)\\b)'
    var_11 = '(?i)(\\b(password|secret|key|token|auth|credential)\\b)'
    var_12 = '(?i)(\\b(ssn|social security|credit card|bank account|phone number|address|email)\\b)'
    var_13 = 'badword1'
    var_14 = 'badword2'
    var_15 = 'badword3'
    var_16 = [var_13, var_14, var_15]
    var_17 = 'free money'
    var_18 = 'get rich quick'
    var_19 = 'click here'
    var_20 = [var_17, var_18, var_19]
    var_21 = 'buy now'
    var_22 = 'limited offer'
    var_23 = 'discount'
    var_24 = 'sale'
    var_25 = [var_21, var_22, var_23, var_24]
    var_26 = '<script>'
    var_27 = 'eval('
    var_28 = 'document.cookie'
    var_29 = 'window.location'
    var_30 = [var_26, var_27, var_28, var_29]
    var_31 = '(?i)(\\b(sql injection|xss|csrf|ssrf|xxe|rce|lfi|rfi)\\b)'
    var_32 = '(?i)(\\b(gdpr|ccpa|hipaa|ferpa|pii|phi)\\b)'
    var_33 = '(?i)(\\b(copyright|trademark|patent|license|agreement|contract)\\b)'
    var_34 = '(?i)(\\b(bias|discrimination|fairness|transparency|accountability)\\b)'
    var_35 = '(?i)(\\b(racism|sexism|homophobia|transphobia|xenophobia|ableism)\\b)'
    var_36 = '(?i)(\\b(climate change|pollution|deforestation|biodiversity|sustainability)\\b)'
    var_37 = '(?i)(\\b(toxic|hazardous|dangerous|unsafe|harmful)\\b)'
    var_38 = '(?i)(\\b(fraud|scam|ponzi|pyramid|insider trading)\\b)'



# Parsed testcases at query #12
#--------------------------



def test_case_0():
    var_0 = module_0.Resolver()
    var_1 = module_0.BuildStep()
    var_2 = 'name'
    var_3 = 0
    var_4 = 128



# Parsed testcases at query #13
#--------------------------


import mimesis.plugins.factory as module_0


def test_case_0():
    var_0 = 'person.full_name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 'male'
    var_3 = module_0.FactoryField(var_0)



# Parsed testcases at query #14
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 'gender'
    var_3 = 'male'
    var_4 = {var_2: var_3}
    var_5 = 'custom_handler'
    var_6 = 'custom_value'
    var_7 = lambda : var_6
    var_8 = (var_5, var_7)
    var_9 = [var_8]
    var_10 = module_0.FactoryField(var_0)
    var_11 = 'invalid_field'
    var_12 = module_0.FactoryField(var_11)
    var_13 = 'invalid_locale'
    var_14 = module_0.FactoryField(var_0, var_13)
    var_15 = module_0.FactoryField(var_0)
    var_16 = module_0.FactoryField(var_0, var_13)
    var_17 = 'female'
    var_18 = {var_2: var_17}
    var_19 = 'new_param'
    var_20 = 'value'
    var_21 = {var_19: var_20}
    var_22 = None
    var_23 = {}
    var_24 = 'age'
    var_25 = 30
    var_26 = {var_2: var_3, var_24: var_25}
    var_27 = 'address'
    var_28 = 'city'
    var_29 = 'country'
    var_30 = 'New York'
    var_31 = 'USA'
    var_32 = {var_28: var_30, var_29: var_31}
    var_33 = {var_2: var_3, var_27: var_32}
    var_34 = 'hobbies'
    var_35 = 'reading'
    var_36 = 'swimming'
    var_37 = [var_35, var_36]
    var_38 = {var_2: var_3, var_34: var_37}



# Parsed testcases at query #15
#--------------------------



def test_case_0():
    var_0 = 'person.full_name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = None
    var_3 = var_1.evaluate(var_2, var_2)
    var_4 = module_0.FactoryField(var_0)
    var_5 = 'gender'
    var_6 = 'female'
    var_7 = {var_5: var_6}
    var_8 = var_4.evaluate(var_2, var_2, var_7)
    var_9 = var_4.evaluate(var_2, var_2)
    assert var_9 == 'custom'
    var_10 = 'custom'
    var_11 = module_0.FactoryField(var_10)
    var_12 = 'field_handlers'
    var_13 = 'invalid.field'
    var_14 = module_0.FactoryField(var_13)
    var_15 = None
    var_16 = var_14.evaluate(var_15, var_15)
    var_17 = ''
    var_18 = module_0.FactoryField(var_17)
    var_19 = None
    var_20 = var_18.evaluate(var_19, var_19)
    var_21 = module_0.FactoryField(var_20)
    var_22 = None
    var_23 = var_21.evaluate(var_22, var_22)
    var_24 = 'invalid'
    var_25 = module_0.FactoryField(var_22, var_24)
    var_26 = None
    var_27 = var_25.evaluate(var_26, var_26)
    var_28 = module_0.FactoryField(var_10)
    var_29 = None
    var_30 = 'field_handlers'
    var_31 = 'invalid'
    var_32 = {var_30: var_31}
    var_33 = var_28.evaluate(var_29, var_29, var_32)
    var_34 = module_0.FactoryField(var_29)
    var_35 = None
    var_36 = 'invalid'
    var_37 = var_34.evaluate(var_35, var_35, var_36)
    var_38 = module_0.FactoryField(var_35)
    var_39 = var_38.evaluate(var_24, var_24)
    var_40 = module_0.FactoryField(var_35)
    var_41 = var_40.evaluate(var_36, var_36)
    var_42 = {}
    var_43 = module_0.FactoryField(var_35, **var_42)
    var_44 = var_43.evaluate(var_36, var_36)
    var_45 = 25
    var_46 = module_0.FactoryField(var_35)
    var_47 = var_46.evaluate(var_36, var_36)
    var_48 = module_0.FactoryField(var_35)
    var_49 = 'male'
    var_50 = {var_32: var_49}
    var_51 = var_48.evaluate(var_36, var_36, var_50)
    assert var_51 == 'custom'
    var_52 = 123
    var_53 = module_0.FactoryField(var_52)
    var_54 = None
    var_55 = var_53.evaluate(var_54, var_54)
    var_56 = module_0.FactoryField(var_54, **var_24)
    var_57 = None
    var_58 = var_56.evaluate(var_57, var_57)
    var_59 = module_0.FactoryField(var_57)
    var_60 = None
    var_61 = 'invalid'
    var_62 = var_59.evaluate(var_60, var_60, var_61)
    var_63 = module_0.FactoryField(var_60, var_24)
    var_64 = None
    var_65 = var_63.evaluate(var_64, var_64)
    var_66 = module_0.FactoryField(var_10)
    var_67 = None
    var_68 = 'field_handlers'
    var_69 = 'invalid'
    var_70 = {var_68: var_69}
    var_71 = var_66.evaluate(var_67, var_67, var_70)
    var_72 = module_0.FactoryField(var_10)
    var_73 = None
    var_74 = 'field_handlers'
    var_75 = 'invalid'
    var_76 = (var_75,)
    var_77 = [var_76]
    var_78 = {var_74: var_77}
    var_79 = var_72.evaluate(var_73, var_73, var_78)
    var_80 = module_0.FactoryField(var_10)
    var_81 = None
    var_82 = 'field_handlers'
    var_83 = 'custom'
    var_84 = 'invalid'
    var_85 = (var_83, var_84)
    var_86 = [var_85]
    var_87 = {var_82: var_86}
    var_88 = var_80.evaluate(var_81, var_81, var_87)
    var_89 = module_0.FactoryField(var_10)
    var_90 = module_0.FactoryField(var_10)
    var_91 = []
    var_92 = {var_12: var_91}
    var_93 = var_90.evaluate(var_82, var_82, var_92)
    var_94 = module_0.FactoryField(var_10)
    var_95 = {var_12: var_82}
    var_96 = var_94.evaluate(var_82, var_82, var_95)
    var_97 = module_0.FactoryField(var_10)
    var_98 = None
    var_99 = 'field_handlers'
    var_100 = 123
    var_101 = (var_100, var_84)
    var_102 = [var_101]
    var_103 = {var_99: var_102}
    var_104 = var_97.evaluate(var_98, var_98, var_103)
    var_105 = module_0.FactoryField(var_10)
    var_106 = None
    var_107 = 'field_handlers'
    var_108 = 'custom'
    var_109 = 123
    var_110 = (var_108, var_109)
    var_111 = [var_110]
    var_112 = {var_107: var_111}
    var_113 = var_105.evaluate(var_106, var_106, var_112)
    var_114 = module_0.FactoryField(var_10)
    var_115 = None
    var_116 = 'field_handlers'
    var_117 = 'custom'
    var_118 = 'extra'
    var_119 = (var_117, var_109, var_118)
    var_120 = [var_119]
    var_121 = {var_116: var_120}
    var_122 = var_114.evaluate(var_115, var_115, var_121)
    var_123 = module_0.FactoryField(var_10)
    var_124 = None
    var_125 = 'field_handlers'
    var_126 = 'custom'
    var_127 = (var_126,)
    var_128 = [var_127]
    var_129 = {var_125: var_128}
    var_130 = var_123.evaluate(var_124, var_124, var_129)
    var_131 = module_0.FactoryField(var_10)
    var_132 = None
    var_133 = 'field_handlers'
    var_134 = 'invalid'
    var_135 = {var_133: var_134}
    var_136 = var_131.evaluate(var_132, var_132, var_135)
    var_137 = module_0.FactoryField(var_10)
    var_138 = None
    var_139 = 'field_handlers'
    var_140 = 'custom'
    var_141 = {var_140: var_135}
    var_142 = {var_139: var_141}
    var_143 = var_137.evaluate(var_138, var_138, var_142)
    var_144 = module_0.FactoryField(var_10)
    var_145 = None
    var_146 = 'field_handlers'
    var_147 = 'custom'
    var_148 = (var_147, var_135)
    var_149 = {var_148}
    var_150 = {var_146: var_149}
    var_151 = var_144.evaluate(var_145, var_145, var_150)
    var_152 = module_0.FactoryField(var_10)
    var_153 = None
    var_154 = 'field_handlers'
    var_155 = 'custom'
    var_156 = (var_155, var_135)
    var_157 = (var_156,)
    var_158 = {var_154: var_157}
    var_159 = var_152.evaluate(var_153, var_153, var_158)
    var_160 = module_0.FactoryField(var_10)
    var_161 = None
    var_162 = 'field_handlers'
    var_163 = 'custom'
    var_164 = (var_163, var_135)
    var_165 = [var_164]
    var_166 = {var_162: var_158}
    var_167 = var_160.evaluate(var_161, var_161, var_166)
    var_168 = module_0.FactoryField(var_10)
    var_169 = None
    var_170 = 'field_handlers'
    var_171 = 'custom'
    var_172 = {var_170: var_171}
    var_173 = var_168.evaluate(var_169, var_169, var_172)
    var_174 = module_0.FactoryField(var_10)
    var_175 = None
    var_176 = 'field_handlers'
    var_177 = 123
    var_178 = {var_176: var_177}
    var_179 = var_174.evaluate(var_175, var_175, var_178)
    var_180 = module_0.FactoryField(var_10)



# Parsed testcases at query #16
#--------------------------


import factory.builder as module_0


def test_case_0():
    var_0 = module_0.Resolver()
    var_1 = module_0.BuildStep()
    var_2 = 'field_handlers'
    var_3 = []
    var_4 = 'name'



# Parsed testcases at query #17
#--------------------------



def test_case_0():
    var_0 = module_0.Resolver()
    var_1 = module_0.BuildStep()
    var_2 = 'test_field'
    var_3 = None



# Parsed testcases at query #18
#--------------------------



def test_case_0():
    var_0 = module_0.Resolver()
    var_1 = module_0.BuildStep()
    var_2 = 'name'



# Parsed testcases at query #19
#--------------------------


import mimesis.plugins.factory as module_0


def test_case_0():
    var_0 = {}
    var_1 = 'name'
    var_2 = module_0.FactoryField(var_1)
    var_3 = 0



# Parsed testcases at query #20
#--------------------------



def test_case_0():
    var_0 = {}
    var_1 = 'name'
    var_2 = module_0.FactoryField(var_1)



# Parsed testcases at query #21
#--------------------------



def test_case_0():
    var_0 = 'person.full_name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = None
    var_3 = var_1.evaluate(var_2, var_2)
    var_4 = module_0.FactoryField(var_0)
    var_5 = 'gender'
    var_6 = 'female'
    var_7 = {var_5: var_6}
    var_8 = var_4.evaluate(var_2, var_2, var_7)
    var_9 = var_4.evaluate(var_2, var_2)
    assert var_9 == 'custom'
    var_10 = 'custom'
    var_11 = module_0.FactoryField(var_10)
    var_12 = 'field_handlers'
    var_13 = module_0.FactoryField(var_10)
    var_14 = 'param'
    var_15 = 'value'
    var_16 = 'invalid_field'
    var_17 = module_0.FactoryField(var_16)
    var_18 = None
    var_19 = var_17.evaluate(var_18, var_18)
    var_20 = ''
    var_21 = module_0.FactoryField(var_20)
    var_22 = None
    var_23 = var_21.evaluate(var_22, var_22)
    var_24 = module_0.FactoryField(var_2)
    var_25 = None
    var_26 = var_24.evaluate(var_25, var_25)
    var_27 = 123
    var_28 = module_0.FactoryField(var_27)
    var_29 = None
    var_30 = var_28.evaluate(var_29, var_29)
    var_31 = 'person.full_name@'
    var_32 = module_0.FactoryField(var_31)
    var_33 = None
    var_34 = var_32.evaluate(var_33, var_33)
    var_35 = 'a'
    var_36 = 1000
    var_37 = var_35 * var_36
    var_38 = module_0.FactoryField(var_37)
    var_39 = None
    var_40 = var_38.evaluate(var_39, var_39)
    var_41 = 'person full name'
    var_42 = module_0.FactoryField(var_41)
    var_43 = None
    var_44 = var_42.evaluate(var_43, var_43)
    var_45 = 'person.fulñame'
    var_46 = module_0.FactoryField(var_45)
    var_47 = None
    var_48 = var_46.evaluate(var_47, var_47)
    var_49 = 'person.full_name❤'
    var_50 = module_0.FactoryField(var_49)
    var_51 = None
    var_52 = var_50.evaluate(var_51, var_51)
    var_53 = "<script>alert('xss')</script>"
    var_54 = module_0.FactoryField(var_53)
    var_55 = None
    var_56 = var_54.evaluate(var_55, var_55)
    var_57 = 'person.full_name; DROP TABLE users;'
    var_58 = module_0.FactoryField(var_57)
    var_59 = None
    var_60 = var_58.evaluate(var_59, var_59)
    var_61 = '../../etc/passwd'
    var_62 = module_0.FactoryField(var_61)
    var_63 = None
    var_64 = var_62.evaluate(var_63, var_63)
    var_65 = 'person.full_name\x00'
    var_66 = module_0.FactoryField(var_65)
    var_67 = None
    var_68 = var_66.evaluate(var_67, var_67)
    var_69 = 'person.full_name\n'
    var_70 = module_0.FactoryField(var_69)
    var_71 = None
    var_72 = var_70.evaluate(var_71, var_71)
    var_73 = 'person.full_name\r'
    var_74 = module_0.FactoryField(var_73)
    var_75 = None
    var_76 = var_74.evaluate(var_75, var_75)
    var_77 = 'person.full_name\t'
    var_78 = module_0.FactoryField(var_77)
    var_79 = None
    var_80 = var_78.evaluate(var_79, var_79)
    var_81 = 'person.full_name\x08'
    var_82 = module_0.FactoryField(var_81)
    var_83 = None
    var_84 = var_82.evaluate(var_83, var_83)
    var_85 = 'person.full_name\x0c'
    var_86 = module_0.FactoryField(var_85)
    var_87 = None
    var_88 = var_86.evaluate(var_87, var_87)
    var_89 = 'person.full_name\x0b'
    var_90 = module_0.FactoryField(var_89)
    var_91 = None
    var_92 = var_90.evaluate(var_91, var_91)
    var_93 = 'person.full_name\\e'
    var_94 = module_0.FactoryField(var_93)
    var_95 = None
    var_96 = var_94.evaluate(var_95, var_95)
    var_97 = 'person.full_name\x07'
    var_98 = module_0.FactoryField(var_97)
    var_99 = None
    var_100 = var_98.evaluate(var_99, var_99)
    var_101 = 'person.full_name\x7f'
    var_102 = module_0.FactoryField(var_101)
    var_103 = None
    var_104 = var_102.evaluate(var_103, var_103)
    var_105 = 'person.full_name\xa0'
    var_106 = module_0.FactoryField(var_105)
    var_107 = None
    var_108 = var_106.evaluate(var_107, var_107)
    var_109 = 'person.full_name\u200b'
    var_110 = module_0.FactoryField(var_109)
    var_111 = None
    var_112 = var_110.evaluate(var_111, var_111)
    var_113 = 'person.full_name\u200e'
    var_114 = module_0.FactoryField(var_113)
    var_115 = None
    var_116 = var_114.evaluate(var_115, var_115)
    var_117 = 'person.full_name\u200f'
    var_118 = module_0.FactoryField(var_117)
    var_119 = None
    var_120 = var_118.evaluate(var_119, var_119)
    var_121 = 'person.full_name\u202c'
    var_122 = module_0.FactoryField(var_121)
    var_123 = None
    var_124 = var_122.evaluate(var_123, var_123)
    var_125 = 'person.full_name\u202a'
    var_126 = module_0.FactoryField(var_125)
    var_127 = None
    var_128 = var_126.evaluate(var_127, var_127)
    var_129 = 'person.full_name\u202b'
    var_130 = module_0.FactoryField(var_129)
    var_131 = None
    var_132 = var_130.evaluate(var_131, var_131)
    var_133 = 'person.full_name\u202d'
    var_134 = module_0.FactoryField(var_133)
    var_135 = None
    var_136 = var_134.evaluate(var_135, var_135)
    var_137 = 'person.full_name\u202e'
    var_138 = module_0.FactoryField(var_137)
    var_139 = None
    var_140 = var_138.evaluate(var_139, var_139)
    var_141 = 'person.full_name￼'
    var_142 = module_0.FactoryField(var_141)
    var_143 = None
    var_144 = var_142.evaluate(var_143, var_143)
    var_145 = 'person.full_name�'
    var_146 = module_0.FactoryField(var_145)
    var_147 = None
    var_148 = var_146.evaluate(var_147, var_147)
    var_149 = 'person.full_name\ue000'
    var_150 = module_0.FactoryField(var_149)
    var_151 = None
    var_152 = var_150.evaluate(var_151, var_151)
    var_153 = 'person.full_name\uffff'
    var_154 = module_0.FactoryField(var_153)
    var_155 = None
    var_156 = var_154.evaluate(var_155, var_155)
    var_157 = 'person.full_name\ud800\udc00'
    var_158 = module_0.FactoryField(var_157)



####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import factory.builder as module_0


def test_case_0():
    var_0 = module_0.Resolver()
    var_1 = module_0.BuildStep()
    var_2 = 'test_field'
    var_3 = module_1.FactoryField(var_2)
    var_4 = var_3.evaluate(var_0, var_1)



# Parsed testcases at query #2
#--------------------------


import mimesis.plugins.factory as module_0


def test_case_0():
    var_0 = 'person.full_name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 'male'
    var_3 = module_0.FactoryField(var_0)



# Parsed testcases at query #3
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 10
    var_3 = module_0.FactoryField(var_0)



# Parsed testcases at query #4
#--------------------------



def test_case_0():
    var_0 = 'person.full_name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 'male'
    var_3 = module_0.FactoryField(var_0)
    var_4 = 'All tests passed!'
    var_5 = print(var_4)



# Parsed testcases at query #5
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 10
    var_3 = module_0.FactoryField(var_0)



# Parsed testcases at query #6
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 10
    var_3 = module_0.FactoryField(var_0)
    var_4 = 'All tests passed!'
    var_5 = print(var_4)



# Parsed testcases at query #7
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 10
    var_3 = module_0.FactoryField(var_0)



# Parsed testcases at query #8
#--------------------------


import builtins as module_1

import factory.builder as module_0


def test_case_0():
    var_0 = module_0.Resolver()
    var_1 = module_0.BuildStep()
    var_2 = None
    var_3 = 'person.full_name'
    var_4 = globals()
    var_5 = locals()
    var_6 = vars()
    var_7 = dir()
    var_8 = 'utf-8'
    var_9 = 0
    var_10 = super()
    var_11 = module_1.object()
    var_12 = lambda x: x
    var_13 = lambda x: x



# Parsed testcases at query #9
#--------------------------


import mimesis.plugins.factory as module_1


def test_case_0():
    var_0 = module_0.Resolver()
    var_1 = module_0.BuildStep()
    var_2 = 'field_handlers'
    var_3 = []
    var_4 = 'word'
    var_5 = module_1.FactoryField(var_4)
    var_6 = var_5.evaluate(var_0, var_1)



# Parsed testcases at query #10
#--------------------------


import mimesis.plugins.factory as module_0


def test_case_0():
    var_0 = 'person.full_name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 'male'
    var_3 = module_0.FactoryField(var_0)



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'field_handlers'
    var_1 = []
    var_2 = {var_0: var_1}
    var_3 = 'name'
    var_4 = 0
    var_5 = 128
    var_6 = 'utf-8'
    var_7 = 'ascii'
    var_8 = 'iso-8859-1'
    var_9 = 'windows-1252'
    var_10 = 'utf-16'
    var_11 = 'utf-32'
    var_12 = 'utf-7'
    var_13 = 'utf-8-sig'
    var_14 = 'utf-16-le'
    var_15 = 'utf-32-le'



# Parsed testcases at query #12
#--------------------------


import factory.builder as module_0


def test_case_0():
    var_0 = module_0.Resolver()
    var_1 = module_0.BuildStep()
    var_2 = 'name'
    var_3 = module_1.FactoryField(var_2)
    var_4 = var_3.evaluate(var_0, var_1)
    var_5 = 0
    var_6 = var_4[var_5]
    var_7 = len(var_4)
    var_8 = len(var_4)
    var_9 = 128



# Parsed testcases at query #13
#--------------------------



def test_case_0():
    var_0 = module_0.Resolver()
    var_1 = module_0.BuildStep()
    var_2 = 'field_handlers'
    var_3 = []
    var_4 = 'name'
    var_5 = 'gender'
    var_6 = 'female'
    var_7 = {var_5: var_6}
    var_8 = 'custom_field'
    var_9 = 'All tests passed!'
    var_10 = print(var_9)



# Parsed testcases at query #14
#--------------------------


import mimesis.plugins.factory as module_0


def test_case_0():
    var_0 = 'person.full_name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 'male'
    var_3 = module_0.FactoryField(var_0)
    var_4 = 'female'



# Parsed testcases at query #15
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 10
    var_3 = module_0.FactoryField(var_0)



# Parsed testcases at query #16
#--------------------------


import factory.builder as module_0


def test_case_0():
    var_0 = module_0.Resolver()
    var_1 = module_0.BuildStep()
    var_2 = 'name'
    var_3 = module_1.FactoryField(var_2)
    var_4 = var_3.evaluate(var_0, var_1)



# Parsed testcases at query #17
#--------------------------


import mimesis.plugins.factory as module_0


def test_case_0():
    var_0 = 'name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 10
    var_3 = module_0.FactoryField(var_0)



# Parsed testcases at query #18
#--------------------------



def test_case_0():
    var_0 = 'person.full_name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = None
    var_3 = var_1.evaluate(var_2, var_2)
    var_4 = 'gender'
    var_5 = 'female'
    var_6 = {var_4: var_5}
    var_7 = var_1.evaluate(var_2, var_2, var_6)
    var_8 = 'custom_field'
    var_9 = var_1.evaluate(var_2, var_2)
    assert var_9 == 'custom_value'
    var_10 = module_0.FactoryField(var_0)
    var_11 = module_0.FactoryField(var_0)
    var_12 = 'person.full_name'
    var_13 = module_0.FactoryField(var_12)
    var_14 = None
    var_15 = var_13.evaluate(var_14, var_14)
    var_16 = 'invalid_field'
    var_17 = module_0.FactoryField(var_16)
    var_18 = None
    var_19 = var_17.evaluate(var_18, var_18)
    var_20 = {}
    var_21 = module_0.FactoryField(var_18, **var_20)
    var_22 = var_21.evaluate(var_19, var_19)
    var_23 = module_0.FactoryField(var_18, var_19)
    var_24 = var_23.evaluate(var_19, var_19)
    var_25 = 'male'
    var_26 = module_0.FactoryField(var_18)
    var_27 = {var_4: var_5}
    var_28 = var_26.evaluate(var_19, var_19, var_27)
    var_29 = {}
    var_30 = module_0.FactoryField(var_18)



# Parsed testcases at query #19
#--------------------------



def test_case_0():
    var_0 = 'person.full_name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = None
    var_3 = var_1.evaluate(var_2, var_2)
    var_4 = len(var_3)
    var_5 = 'male'
    var_6 = module_0.FactoryField(var_0)
    var_7 = 'gender'
    var_8 = {var_7: var_5}
    var_9 = var_6.evaluate(var_2, var_2, var_8)
    var_10 = len(var_9)
    var_11 = var_6.evaluate(var_2, var_2)
    var_12 = len(var_11)
    var_13 = 'custom_field'
    var_14 = var_6.evaluate(var_2, var_2)
    assert var_14 == 'custom_value'
    var_15 = module_0.FactoryField(var_0)
    var_16 = module_0.FactoryField(var_0)
    var_17 = 'custom_field1'
    var_18 = 'custom_field2'



# Parsed testcases at query #20
#--------------------------



def test_case_0():
    var_0 = 'person.full_name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 'male'
    var_3 = module_0.FactoryField(var_0)
    var_4 = 'All tests passed!'
    var_5 = print(var_4)



# Parsed testcases at query #21
#--------------------------



def test_case_0():
    var_0 = 'person.full_name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 'male'
    var_3 = module_0.FactoryField(var_0)



# Parsed testcases at query #22
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 10
    var_3 = module_0.FactoryField(var_0)
    var_4 = 'All tests passed!'
    var_5 = print(var_4)



# Parsed testcases at query #23
#--------------------------



def test_case_0():
    var_0 = 'person.full_name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 'male'
    var_3 = module_0.FactoryField(var_0)
    var_4 = 'female'



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'person.full_name'
    var_1 = None
    var_2 = []



# Parsed testcases at query #25
#--------------------------



def test_case_0():
    var_0 = 'name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 10
    var_3 = module_0.FactoryField(var_0)



# Parsed testcases at query #26
#--------------------------



def test_case_0():
    var_0 = 'person.full_name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 'male'
    var_3 = module_0.FactoryField(var_0)
    var_4 = ''
    var_5 = module_0.FactoryField(var_4)
    var_6 = None
    var_7 = module_0.FactoryField(var_0, var_6)
    var_8 = {}
    var_9 = module_0.FactoryField(var_0, **var_8)
    var_10 = 30
    var_11 = module_0.FactoryField(var_0)
    var_12 = 'person.full_name@example.com'
    var_13 = module_0.FactoryField(var_12)
    var_14 = '123'
    var_15 = module_0.FactoryField(var_14)
    var_16 = 'All test cases passed!'
    var_17 = print(var_16)



# Parsed testcases at query #27
#--------------------------


import factory.builder as module_0


def test_case_0():
    var_0 = module_0.Resolver()
    var_1 = module_0.BuildStep()
    var_2 = 'name'
    var_3 = 0



# Parsed testcases at query #28
#--------------------------


import mimesis.plugins.factory as module_0


def test_case_0():
    var_0 = 'name'
    var_1 = module_0.FactoryField(var_0)
    var_2 = 10
    var_3 = module_0.FactoryField(var_0)



# Parsed testcases at query #29
#--------------------------


import factory.builder as module_0


def test_case_0():
    var_0 = module_0.Resolver()
    var_1 = module_0.BuildStep()
    var_2 = 'field_handlers'
    var_3 = []
    var_4 = 'name'
    var_5 = module_1.FactoryField(var_4)
    var_6 = var_5.evaluate(var_0, var_1)



