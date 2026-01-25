####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import mimesis.providers.internet as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.url()
    var_2 = 'https://'
    var_3 = '/'
    var_4 = module_1.split(var_3)
    var_5 = len(var_4)
    assert var_5 == 4
    var_6 = '.'
    var_7 = module_1.split(var_6)
    var_8 = len(var_7)



# Parsed testcases at query #2
#--------------------------


import mimesis.providers.internet as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.url()
    var_2 = 'https://'
    var_3 = '/'
    var_4 = module_1.split(var_3)
    var_5 = len(var_4)
    assert var_5 == 4
    var_6 = ':'
    var_7 = module_1.split(var_6)
    var_8 = len(var_7)
    assert var_8 == 2
    var_9 = '.'
    var_10 = module_1.split(var_9)
    var_11 = len(var_10)
    assert var_11 == 3
    var_12 = '-'
    var_13 = '_'
    var_14 = ' '
    var_15 = '?'
    var_16 = '&'
    var_17 = '='
    var_18 = '#'
    var_19 = '%'
    var_20 = '+'
    var_21 = '@'
    var_22 = '!'
    var_23 = '$'
    var_24 = '^'
    var_25 = '*'
    var_26 = '('
    var_27 = ')'
    var_28 = '['
    var_29 = ']'
    var_30 = '{'
    var_31 = '}'
    var_32 = '|'
    var_33 = '\\'
    var_34 = ';'
    var_35 = '"'
    var_36 = "'"
    var_37 = '<'
    var_38 = '>'
    var_39 = ','
    var_40 = '`'
    var_41 = '~'
    var_42 = '\t'
    var_43 = '\n'
    var_44 = '\r'
    var_45 = '\x0b'
    var_46 = '\x0c'
    var_47 = '\x1f'
    var_48 = '\x7f'
    var_49 = '\x80'
    var_50 = 'ÿ'
    var_51 = 'Ā'
    var_52 = '\uffff'
    var_53 = '𐀀'
    var_54 = '\U0010ffff'
    var_55 = '\x00'
    var_56 = '\x01'
    var_57 = '\x02'
    var_58 = '\x03'
    var_59 = '\x04'
    var_60 = '\x05'
    var_61 = '\x06'
    var_62 = '\x07'
    var_63 = '\x08'
    var_64 = '\x0e'
    var_65 = '\x0f'
    var_66 = '\x10'
    var_67 = '\x11'
    var_68 = '\x12'
    var_69 = '\x13'
    var_70 = '\x14'
    var_71 = '\x15'
    var_72 = '\x16'
    var_73 = '\x17'
    var_74 = '\x18'
    var_75 = '\x19'
    var_76 = '\x1a'
    var_77 = '\x1b'
    var_78 = '\x1c'
    var_79 = '\x1d'
    var_80 = '\x1e'
    var_81 = '\x9f'
    var_82 = '\xa0'
    var_83 = '\xad'
    var_84 = '\u0600'
    var_85 = '\u0601'
    var_86 = '\u0602'
    var_87 = '\u0603'
    var_88 = '\u0604'
    var_89 = '\u0605'
    var_90 = '؆'
    var_91 = '؇'
    var_92 = '؈'
    var_93 = '؉'
    var_94 = '؊'
    var_95 = '؋'
    var_96 = '،'
    var_97 = '؍'
    var_98 = '؎'
    var_99 = '؏'
    var_100 = 'ؐ'
    var_101 = 'ؑ'
    var_102 = 'ؒ'
    var_103 = 'ؓ'
    var_104 = 'ؔ'
    var_105 = 'ؕ'
    var_106 = 'ؖ'
    var_107 = 'ؗ'
    var_108 = 'ؘ'
    var_109 = 'ؙ'
    var_110 = 'ؚ'
    var_111 = '؛'
    var_112 = '\u061c'
    var_113 = '\u061d'
    var_114 = '؞'
    var_115 = '؟'
    var_116 = 'ؠ'
    var_117 = 'ء'
    var_118 = 'آ'
    var_119 = 'أ'
    var_120 = 'ؤ'
    var_121 = 'إ'
    var_122 = 'ئ'
    var_123 = 'ا'
    var_124 = 'ب'
    var_125 = 'ة'
    var_126 = 'ت'
    var_127 = 'ث'
    var_128 = 'ج'
    var_129 = 'ح'
    var_130 = 'خ'
    var_131 = 'د'
    var_132 = 'ذ'
    var_133 = 'ر'
    var_134 = 'ز'
    var_135 = 'س'
    var_136 = 'ش'
    var_137 = 'ص'
    var_138 = 'ض'
    var_139 = 'ط'
    var_140 = 'ظ'
    var_141 = 'ع'
    var_142 = 'غ'
    var_143 = 'ػ'
    var_144 = 'ؼ'
    var_145 = 'ؽ'
    var_146 = 'ؾ'
    var_147 = 'ؿ'
    var_148 = 'ـ'
    var_149 = 'ف'
    var_150 = 'ق'
    var_151 = 'ك'
    var_152 = 'ل'
    var_153 = 'م'
    var_154 = 'ن'
    var_155 = 'ه'
    var_156 = 'و'
    var_157 = 'ى'



# Parsed testcases at query #3
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.query_parameters()
    var_2 = len(var_1)
    var_3 = 5
    var_4 = var_0.query_parameters(var_3)
    var_5 = len(var_4)
    assert var_5 == 5
    var_6 = 32
    var_7 = var_0.query_parameters(var_6)
    var_8 = len(var_7)
    assert var_8 == 32
    var_9 = 33
    var_10 = var_0.query_parameters(var_9)
    var_11 = 0
    var_12 = var_0.query_parameters(var_11)



# Parsed testcases at query #4
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 'www'
    var_2 = 'api'
    var_3 = 'app'
    var_4 = [var_1, var_2, var_3]
    var_5 = 'https://'



# Parsed testcases at query #5
#--------------------------


import mimesis.providers.internet as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.url()
    var_2 = 'https://'
    var_3 = '/'
    var_4 = module_1.split(var_3)
    var_5 = len(var_4)



# Parsed testcases at query #6
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.query_parameters()
    var_2 = len(var_1)
    var_3 = 5
    var_4 = var_0.query_parameters(var_3)
    var_5 = len(var_4)
    assert var_5 == 5
    var_6 = 32
    var_7 = var_0.query_parameters(var_6)
    var_8 = len(var_7)
    assert var_8 == 32
    var_9 = 33
    var_10 = var_0.query_parameters(var_9)



# Parsed testcases at query #7
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.url()
    var_2 = 'https://'



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.query_parameters()
    var_2 = len(var_1)
    var_3 = 5
    var_4 = var_0.query_parameters(var_3)
    var_5 = len(var_4)
    assert var_5 == 5
    var_6 = 33
    var_7 = var_0.query_parameters(var_6)



# Parsed testcases at query #2
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = 'Unit test for method query_parameters of class Internet.'
    var_1 = module_0.Internet()
    var_2 = 5
    var_3 = var_1.query_parameters(var_2)
    var_4 = len(var_3)
    assert var_4 == 5



# Parsed testcases at query #3
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 5
    var_2 = var_0.query_parameters(var_1)
    var_3 = len(var_2)
    assert var_3 == 5



# Parsed testcases at query #4
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.query_parameters()
    var_2 = len(var_1)
    var_3 = 5
    var_4 = var_0.query_parameters(var_3)
    var_5 = len(var_4)
    assert var_5 == 5
    var_6 = 32
    var_7 = var_0.query_parameters(var_6)
    var_8 = len(var_7)
    assert var_8 == 32
    var_9 = 33
    var_10 = var_0.query_parameters(var_9)
    var_11 = 10
    var_12 = var_0.query_parameters(var_11)



# Parsed testcases at query #5
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.query_parameters()
    var_2 = len(var_1)
    var_3 = 15
    var_4 = var_0.query_parameters(var_3)
    var_5 = len(var_4)
    assert var_5 == 15



# Parsed testcases at query #6
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.query_parameters()
    var_2 = len(var_1)
    var_3 = 5
    var_4 = var_0.query_parameters(var_3)
    var_5 = len(var_4)
    assert var_5 == 5
    var_6 = 33
    var_7 = var_0.query_parameters(var_6)



# Parsed testcases at query #7
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.query_parameters()
    var_2 = len(var_1)



# Parsed testcases at query #8
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 5
    var_2 = var_0.query_parameters(var_1)
    var_3 = len(var_2)
    assert var_3 == 5



# Parsed testcases at query #9
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.query_parameters()
    var_2 = len(var_1)
    var_3 = 5
    var_4 = var_0.query_parameters(var_3)
    var_5 = len(var_4)
    assert var_5 == 5
    var_6 = 33
    var_7 = var_0.query_parameters(var_6)



# Parsed testcases at query #10
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.query_parameters()
    var_2 = len(var_1)
    var_3 = 5
    var_4 = var_0.query_parameters(var_3)
    var_5 = len(var_4)
    assert var_5 == 5
    var_6 = 33
    var_7 = var_0.query_parameters(var_6)



# Parsed testcases at query #11
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 5
    var_2 = var_0.query_parameters(var_1)
    var_3 = len(var_2)
    assert var_3 == 5



# Parsed testcases at query #12
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.query_parameters()
    var_2 = len(var_1)
    var_3 = 5
    var_4 = var_0.query_parameters(var_3)
    var_5 = len(var_4)
    assert var_5 == 5
    var_6 = 32
    var_7 = var_0.query_parameters(var_6)
    var_8 = len(var_7)
    assert var_8 == 32
    var_9 = 33
    var_10 = var_0.query_parameters(var_9)



# Parsed testcases at query #13
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 5
    var_2 = var_0.query_parameters(var_1)
    var_3 = len(var_2)
    assert var_3 == 5



# Parsed testcases at query #14
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.query_parameters()
    var_2 = len(var_1)
    var_3 = 5
    var_4 = var_0.query_parameters(var_3)
    var_5 = len(var_4)
    assert var_5 == 5
    var_6 = 33
    var_7 = var_0.query_parameters(var_6)



# Parsed testcases at query #15
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.query_parameters()
    var_2 = len(var_1)
    var_3 = len(var_1)
    var_4 = 5
    var_5 = var_0.query_parameters(var_4)
    var_6 = len(var_5)
    assert var_6 == 5
    var_7 = 33
    var_8 = var_0.query_parameters(var_7)



# Parsed testcases at query #16
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.query_parameters()
    var_2 = len(var_1)
    var_3 = 5
    var_4 = var_0.query_parameters(var_3)
    var_5 = len(var_4)
    assert var_5 == 5
    var_6 = 33
    var_7 = var_0.query_parameters(var_6)



# Parsed testcases at query #17
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.query_parameters()
    var_2 = len(var_1)
    var_3 = 5
    var_4 = var_0.query_parameters(var_3)
    var_5 = len(var_4)
    assert var_5 == 5
    var_6 = 33
    var_7 = var_0.query_parameters(var_6)



# Parsed testcases at query #18
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.query_parameters()
    var_2 = len(var_1)
    var_3 = 5
    var_4 = var_0.query_parameters(var_3)
    var_5 = len(var_4)
    assert var_5 == 5
    var_6 = 32
    var_7 = var_0.query_parameters(var_6)
    var_8 = len(var_7)
    assert var_8 == 32
    var_9 = 33
    var_10 = var_0.query_parameters(var_9)



# Parsed testcases at query #19
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.query_parameters()
    var_2 = len(var_1)
    var_3 = 10
    var_4 = var_0.query_parameters(var_3)
    var_5 = len(var_4)
    assert var_5 == 10
    var_6 = 33
    var_7 = var_0.query_parameters(var_6)



# Parsed testcases at query #20
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 5
    var_2 = var_0.query_parameters(var_1)
    var_3 = len(var_2)
    assert var_3 == 5



# Parsed testcases at query #21
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.query_parameters()
    var_2 = len(var_1)
    var_3 = 5
    var_4 = var_0.query_parameters(var_3)
    var_5 = len(var_4)
    assert var_5 == 5
    var_6 = 33
    var_7 = var_0.query_parameters(var_6)



# Parsed testcases at query #22
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 5
    var_2 = var_0.query_parameters(var_1)
    var_3 = len(var_2)
    assert var_3 == 5



# Parsed testcases at query #23
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.query_parameters()
    var_2 = len(var_1)



# Parsed testcases at query #24
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.query_parameters()
    var_2 = len(var_1)



# Parsed testcases at query #25
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = 5
    var_2 = var_0.query_parameters(var_1)
    var_3 = len(var_2)
    assert var_3 == 5



