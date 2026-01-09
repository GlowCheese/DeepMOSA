####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.fields as module_0


def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = (var_0, var_1)
    var_3 = 'key2'
    var_4 = 'value2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = True
    var_8 = module_0.Choice(choices=var_6)
    var_9 = None
    var_10 = var_8.validate(var_9)
    assert var_10 is None
    var_11 = (var_0, var_1)
    var_12 = (var_3, var_4)
    var_13 = [var_11, var_12]
    var_14 = False
    var_15 = module_0.Choice(choices=var_13)
    var_16 = None
    var_17 = var_15.validate(var_16)
    var_18 = (var_16, var_17)
    var_19 = (var_3, var_4)
    var_20 = [var_18, var_19]
    var_21 = module_0.Choice(choices=var_20)
    var_22 = 'key3'
    var_23 = var_21.validate(var_22)
    var_24 = (var_22, var_23)
    var_25 = (var_3, var_4)
    var_26 = [var_24, var_25]
    var_27 = module_0.Choice(choices=var_26)
    var_28 = var_27.validate(var_22)
    assert var_28 == 'key1'
    var_29 = (var_22, var_23)
    var_30 = (var_3, var_4)
    var_31 = [var_29, var_30]
    var_32 = module_0.Choice(choices=var_31, coerce_types=var_7)
    var_33 = ''
    var_34 = var_32.validate(var_33)
    assert var_34 is None
    var_35 = (var_22, var_23)
    var_36 = (var_3, var_4)
    var_37 = [var_35, var_36]
    var_38 = module_0.Choice(choices=var_37)
    var_39 = ''
    var_40 = var_38.validate(var_39)
    var_41 = (var_39, var_40)
    var_42 = (var_3, var_4)
    var_43 = [var_41, var_42]
    var_44 = module_0.Choice(choices=var_43, coerce_types=var_14)
    var_45 = ''
    var_46 = var_44.validate(var_45)
    var_47 = (var_45, var_46)
    var_48 = (var_3, var_4)
    var_49 = [var_47, var_48]
    var_50 = module_0.Choice(choices=var_49)
    var_51 = 123
    var_52 = var_50.validate(var_51)
    var_53 = (var_51, var_52)
    var_54 = (var_3, var_4)
    var_55 = [var_53, var_54]
    var_56 = module_0.Choice(choices=var_55)
    var_57 = 'key3'
    var_58 = var_56.validate(var_57)
    var_59 = (var_57, var_58)
    var_60 = (var_3, var_4)
    var_61 = [var_59, var_60]
    var_62 = module_0.Choice(choices=var_61)
    var_63 = var_62.validate(var_3)
    assert var_63 == 'key2'
    var_64 = (var_57, var_58)
    var_65 = (var_3, var_4)
    var_66 = [var_64, var_65]
    var_67 = module_0.Choice(choices=var_66)
    var_68 = 'KEY1'
    var_69 = var_67.validate(var_68)
    assert var_69 == 'KEY1'
    var_70 = (var_57, var_58)
    var_71 = (var_3, var_4)
    var_72 = [var_70, var_71]
    var_73 = module_0.Choice(choices=var_72)
    var_74 = ' key1 '
    var_75 = var_73.validate(var_74)
    assert var_75 == ' key1 '
    var_76 = (var_57, var_58)
    var_77 = (var_3, var_4)
    var_78 = [var_76, var_77]
    var_79 = module_0.Choice(choices=var_78)
    var_80 = '  key1  '
    var_81 = var_79.validate(var_80)
    assert var_81 == '  key1  '
    var_82 = (var_57, var_58)
    var_83 = (var_3, var_4)
    var_84 = [var_82, var_83]
    var_85 = module_0.Choice(choices=var_84)
    var_86 = 'key1!'
    var_87 = var_85.validate(var_86)
    assert var_87 == 'key1!'
    var_88 = (var_57, var_58)
    var_89 = (var_3, var_4)
    var_90 = [var_88, var_89]
    var_91 = module_0.Choice(choices=var_90)
    var_92 = 'key1é'
    var_93 = var_91.validate(var_92)
    assert var_93 == 'key1é'
    var_94 = (var_57, var_58)
    var_95 = (var_3, var_4)
    var_96 = [var_94, var_95]
    var_97 = module_0.Choice(choices=var_96)
    var_98 = 'key1❤'
    var_99 = var_97.validate(var_98)
    assert var_99 == 'key1❤'
    var_100 = (var_57, var_58)
    var_101 = (var_3, var_4)
    var_102 = [var_100, var_101]
    var_103 = module_0.Choice(choices=var_102)
    var_104 = 'key1\n'
    var_105 = var_103.validate(var_104)
    assert var_105 == 'key1\n'
    var_106 = (var_57, var_58)
    var_107 = (var_3, var_4)
    var_108 = [var_106, var_107]
    var_109 = module_0.Choice(choices=var_108)
    var_110 = 'key1\t'
    var_111 = var_109.validate(var_110)
    assert var_111 == 'key1\t'
    var_112 = (var_57, var_58)
    var_113 = (var_3, var_4)
    var_114 = [var_112, var_113]
    var_115 = module_0.Choice(choices=var_114)
    var_116 = 'key1\r'
    var_117 = var_115.validate(var_116)
    assert var_117 == 'key1\r'
    var_118 = (var_57, var_58)
    var_119 = (var_3, var_4)
    var_120 = [var_118, var_119]
    var_121 = module_0.Choice(choices=var_120)
    var_122 = 'key1\x08'
    var_123 = var_121.validate(var_122)
    assert var_123 == 'key1\x08'
    var_124 = (var_57, var_58)
    var_125 = (var_3, var_4)
    var_126 = [var_124, var_125]
    var_127 = module_0.Choice(choices=var_126)
    var_128 = 'key1\x0c'
    var_129 = var_127.validate(var_128)
    assert var_129 == 'key1\x0c'
    var_130 = (var_57, var_58)
    var_131 = (var_3, var_4)
    var_132 = [var_130, var_131]
    var_133 = module_0.Choice(choices=var_132)
    var_134 = 'key1\x0b'
    var_135 = var_133.validate(var_134)
    assert var_135 == 'key1\x0b'
    var_136 = (var_57, var_58)
    var_137 = (var_3, var_4)
    var_138 = [var_136, var_137]
    var_139 = module_0.Choice(choices=var_138)
    var_140 = 'key1\x00'
    var_141 = var_139.validate(var_140)
    assert var_141 == 'key1\x00'
    var_142 = (var_57, var_58)
    var_143 = (var_3, var_4)
    var_144 = [var_142, var_143]
    var_145 = module_0.Choice(choices=var_144)
    var_146 = 'key1\\'
    var_147 = var_145.validate(var_146)
    assert var_147 == 'key1\\'
    var_148 = (var_57, var_58)
    var_149 = (var_3, var_4)
    var_150 = [var_148, var_149]
    var_151 = module_0.Choice(choices=var_150)
    var_152 = 'key1"'
    var_153 = var_151.validate(var_152)
    assert var_153 == 'key1"'
    var_154 = (var_57, var_58)
    var_155 = (var_3, var_4)
    var_156 = [var_154, var_155]
    var_157 = module_0.Choice(choices=var_156)
    var_158 = "key1'"
    var_159 = var_157.validate(var_158)
    assert var_159 == "key1'"
    var_160 = (var_57, var_58)
    var_161 = (var_3, var_4)
    var_162 = [var_160, var_161]
    var_163 = module_0.Choice(choices=var_162)
    var_164 = 'key1`'
    var_165 = var_163.validate(var_164)
    assert var_165 == 'key1`'
    var_166 = (var_57, var_58)
    var_167 = (var_3, var_4)
    var_168 = [var_166, var_167]
    var_169 = module_0.Choice(choices=var_168)
    var_170 = 'key1~'
    var_171 = var_169.validate(var_170)
    assert var_171 == 'key1~'
    var_172 = (var_57, var_58)
    var_173 = (var_3, var_4)
    var_174 = [var_172, var_173]
    var_175 = module_0.Choice(choices=var_174)
    var_176 = var_175.validate(var_86)
    assert var_176 == 'key1!'
    var_177 = (var_57, var_58)
    var_178 = (var_3, var_4)
    var_179 = [var_177, var_178]
    var_180 = module_0.Choice(choices=var_179)
    var_181 = 'key1@'
    var_182 = var_180.validate(var_181)
    assert var_182 == 'key1@'
    var_183 = (var_57, var_58)
    var_184 = (var_3, var_4)
    var_185 = [var_183, var_184]
    var_186 = module_0.Choice(choices=var_185)
    var_187 = 'key1#'
    var_188 = var_186.validate(var_187)
    assert var_188 == 'key1#'
    var_189 = (var_57, var_58)
    var_190 = (var_3, var_4)
    var_191 = [var_189, var_190]
    var_192 = module_0.Choice(choices=var_191)
    var_193 = 'key1$'
    var_194 = var_192.validate(var_193)
    assert var_194 == 'key1$'



# Parsed testcases at query #2
#--------------------------



def test_case_0():
    var_0 = None
    var_1 = module_0.Const(var_0)
    var_2 = var_1.validate(var_0)
    assert var_2 is None
    var_3 = 1
    var_4 = var_1.validate(var_3)
    var_5 = 1
    var_6 = module_0.Const(var_5)
    var_7 = var_6.validate(var_5)
    assert var_7 == 1
    var_8 = 2
    var_9 = var_6.validate(var_8)
    var_10 = 'hello'
    var_11 = module_0.Const(var_10)
    var_12 = var_11.validate(var_10)
    assert var_12 == 'hello'
    var_13 = 'world'
    var_14 = var_11.validate(var_13)
    var_15 = True
    var_16 = module_0.Const(var_15)
    var_17 = True
    var_18 = var_16.validate(var_17)
    assert var_18 is True
    var_19 = False
    var_20 = var_16.validate(var_19)
    var_21 = False
    var_22 = module_0.Const(var_21)
    var_23 = var_22.validate(var_21)
    assert var_23 is False
    var_24 = True
    var_25 = var_22.validate(var_24)
    var_26 = []
    var_27 = module_0.Const(var_26)
    var_28 = []
    var_29 = var_27.validate(var_28)
    var_30 = 1
    var_31 = [var_30]
    var_32 = var_27.validate(var_31)
    var_33 = {}
    var_34 = module_0.Const(var_33)
    var_35 = {}
    var_36 = var_34.validate(var_35)
    var_37 = 'a'
    var_38 = 1
    var_39 = {var_37: var_38}
    var_40 = var_34.validate(var_39)
    var_41 = module_0.Const(var_21)
    var_42 = var_41.validate(var_21)
    assert var_42 == 0
    var_43 = 1
    var_44 = var_41.validate(var_43)
    var_45 = ''
    var_46 = module_0.Const(var_45)
    var_47 = var_46.validate(var_45)
    assert var_47 == ''
    var_48 = 'a'
    var_49 = var_46.validate(var_48)
    var_50 = 3.14
    var_51 = module_0.Const(var_50)
    var_52 = var_51.validate(var_50)
    var_53 = 3.141
    var_54 = var_51.validate(var_53)
    var_55 = -1
    var_56 = module_0.Const(var_55)
    var_57 = -1
    var_58 = var_56.validate(var_57)
    assert var_58 == -1
    var_59 = 1
    var_60 = var_56.validate(var_59)
    var_61 = 'null'
    var_62 = module_0.Const(var_61)
    var_63 = var_62.validate(var_61)
    assert var_63 == 'null'
    var_64 = None
    var_65 = var_62.validate(var_64)
    var_66 = 'true'
    var_67 = module_0.Const(var_66)
    var_68 = var_67.validate(var_66)
    assert var_68 == 'true'
    var_69 = True
    var_70 = var_67.validate(var_69)
    var_71 = 'false'
    var_72 = module_0.Const(var_71)
    var_73 = var_72.validate(var_71)
    assert var_73 == 'false'
    var_74 = False
    var_75 = var_72.validate(var_74)
    var_76 = '0'
    var_77 = module_0.Const(var_76)
    var_78 = var_77.validate(var_76)
    assert var_78 == '0'
    var_79 = 0
    var_80 = var_77.validate(var_79)
    var_81 = '1'
    var_82 = module_0.Const(var_81)
    var_83 = var_82.validate(var_81)
    assert var_83 == '1'
    var_84 = 1
    var_85 = var_82.validate(var_84)
    var_86 = '[]'
    var_87 = module_0.Const(var_86)
    var_88 = var_87.validate(var_86)
    assert var_88 == '[]'
    var_89 = []
    var_90 = var_87.validate(var_89)
    var_91 = '{}'
    var_92 = module_0.Const(var_91)
    var_93 = var_92.validate(var_91)
    assert var_93 == '{}'
    var_94 = {}
    var_95 = var_92.validate(var_94)
    var_96 = '3.14'
    var_97 = module_0.Const(var_96)
    var_98 = var_97.validate(var_96)
    assert var_98 == '3.14'
    var_99 = 3.14
    var_100 = var_97.validate(var_99)
    var_101 = '-1'
    var_102 = module_0.Const(var_101)
    var_103 = var_102.validate(var_101)
    assert var_103 == '-1'
    var_104 = -1
    var_105 = var_102.validate(var_104)
    var_106 = 'hello world'
    var_107 = module_0.Const(var_106)
    var_108 = var_107.validate(var_106)
    assert var_108 == 'hello world'
    var_109 = 'hello'
    var_110 = var_107.validate(var_109)
    var_111 = ' '
    var_112 = module_0.Const(var_111)
    var_113 = var_112.validate(var_111)
    assert var_113 == ' '
    var_114 = ''
    var_115 = var_112.validate(var_114)
    var_116 = '\n'
    var_117 = module_0.Const(var_116)
    var_118 = var_117.validate(var_116)
    assert var_118 == '\n'
    var_119 = ''
    var_120 = var_117.validate(var_119)
    var_121 = '\t'
    var_122 = module_0.Const(var_121)
    var_123 = var_122.validate(var_121)
    assert var_123 == '\t'
    var_124 = ''
    var_125 = var_122.validate(var_124)
    var_126 = '\\'
    var_127 = module_0.Const(var_126)
    var_128 = var_127.validate(var_126)
    assert var_128 == '\\'
    var_129 = '/'
    var_130 = var_127.validate(var_129)
    var_131 = '"'
    var_132 = module_0.Const(var_131)
    var_133 = var_132.validate(var_131)
    assert var_133 == '"'
    var_134 = "'"
    var_135 = var_132.validate(var_134)
    var_136 = "'"
    var_137 = module_0.Const(var_136)
    var_138 = var_137.validate(var_136)
    assert var_138 == "'"
    var_139 = '"'
    var_140 = var_137.validate(var_139)
    var_141 = '`'
    var_142 = module_0.Const(var_141)
    var_143 = var_142.validate(var_141)
    assert var_143 == '`'
    var_144 = '~'
    var_145 = var_142.validate(var_144)



# Parsed testcases at query #3
#--------------------------



def test_case_0():
    var_0 = True
    var_1 = module_0.Object()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None
    var_4 = False
    var_5 = module_0.Object()
    var_6 = None
    var_7 = var_5.validate(var_6)
    var_8 = module_0.Object()
    var_9 = 'not a dict'
    var_10 = var_8.validate(var_9)
    var_11 = module_0.Object()
    var_12 = 1
    var_13 = 'value'
    var_14 = {var_12: var_13}
    var_15 = var_11.validate(var_14)
    var_16 = '^[a-z]+$'
    var_17 = module_0.String(pattern=var_16)
    var_18 = module_0.Object(property_names=var_17)
    var_19 = 'UPPERCASE'
    var_20 = 'value'
    var_21 = {var_19: var_20}
    var_22 = var_18.validate(var_21)
    var_23 = 2
    var_24 = module_0.Object(min_properties=var_23)
    var_25 = 'key'
    var_26 = 'value'
    var_27 = {var_25: var_26}
    var_28 = var_24.validate(var_27)
    var_29 = module_0.Object(max_properties=var_25)
    var_30 = 'key1'
    var_31 = 'key2'
    var_32 = 'value1'
    var_33 = 'value2'
    var_34 = {var_30: var_32, var_31: var_33}
    var_35 = var_29.validate(var_34)
    var_36 = 'required_key'
    var_37 = [var_36]
    var_38 = module_0.Object(required=var_37)
    var_39 = 'other_key'
    var_40 = 'value'
    var_41 = {var_39: var_40}
    var_42 = var_38.validate(var_41)
    var_43 = 'key'
    var_44 = 'default_value'
    var_45 = module_0.String()
    var_46 = {var_43: var_45}
    var_47 = module_0.Object(properties=var_46)
    var_48 = {}
    var_49 = var_47.validate(var_48)
    var_50 = module_0.Integer()
    var_51 = {var_43: var_50}
    var_52 = module_0.Object(properties=var_51)
    var_53 = 'key'
    var_54 = 'not an integer'
    var_55 = {var_53: var_54}
    var_56 = var_52.validate(var_55)
    var_57 = module_0.Integer()
    var_58 = {var_34: var_57}
    var_59 = module_0.Object(pattern_properties=var_58)
    var_60 = 123
    var_61 = {var_43: var_60}
    var_62 = var_59.validate(var_61)
    var_63 = module_0.Integer()
    var_64 = {var_34: var_63}
    var_65 = module_0.Object(pattern_properties=var_64, additional_properties=var_56)
    var_66 = 'UPPERCASE'
    var_67 = 123
    var_68 = {var_66: var_67}
    var_69 = var_65.validate(var_68)
    var_70 = module_0.Integer()
    var_71 = {var_34: var_70}
    var_72 = module_0.String()
    var_73 = module_0.Object(pattern_properties=var_71, additional_properties=var_72)
    var_74 = 'UPPERCASE'
    var_75 = 'string'
    var_76 = {var_74: var_75}
    var_77 = var_73.validate(var_76)
    var_78 = module_0.Integer()
    var_79 = {var_34: var_78}
    var_80 = module_0.Object(pattern_properties=var_79, additional_properties=var_66)
    var_81 = 'anything'
    var_82 = {var_74: var_81}
    var_83 = var_80.validate(var_82)
    var_84 = module_0.Integer()
    var_85 = {var_34: var_84}
    var_86 = module_0.Object(pattern_properties=var_85, additional_properties=var_67)
    var_87 = {var_74: var_81}
    var_88 = var_86.validate(var_87)
    var_89 = module_0.Integer()
    var_90 = {var_34: var_89}
    var_91 = module_0.Object(pattern_properties=var_90)
    var_92 = 'key'
    var_93 = 'not an integer'
    var_94 = {var_92: var_93}
    var_95 = var_91.validate(var_94)
    var_96 = module_0.Integer()
    var_97 = {var_34: var_96}
    var_98 = module_0.String()
    var_99 = module_0.Object(pattern_properties=var_97, additional_properties=var_98)
    var_100 = 'UPPERCASE'
    var_101 = 123
    var_102 = {var_100: var_101}
    var_103 = var_99.validate(var_102)
    var_104 = module_0.Integer()
    var_105 = {var_34: var_104}
    var_106 = module_0.Object(pattern_properties=var_105, additional_properties=var_100)
    var_107 = {var_43: var_60, var_74: var_81}
    var_108 = var_106.validate(var_107)
    var_109 = module_0.Integer()
    var_110 = {var_34: var_109}
    var_111 = module_0.Object(pattern_properties=var_110, additional_properties=var_103)
    var_112 = 'key'
    var_113 = 'UPPERCASE'
    var_114 = 123
    var_115 = 'anything'
    var_116 = {var_112: var_114, var_113: var_115}
    var_117 = var_111.validate(var_116)
    var_118 = module_0.Integer()
    var_119 = {var_116: var_118}
    var_120 = module_0.Object(pattern_properties=var_119, additional_properties=var_113)
    var_121 = {var_43: var_60, var_74: var_81}
    var_122 = var_120.validate(var_121)
    var_123 = module_0.Integer()
    var_124 = {var_116: var_123}
    var_125 = module_0.String()
    var_126 = module_0.Object(pattern_properties=var_124, additional_properties=var_125)
    var_127 = {var_43: var_60, var_74: var_75}
    var_128 = var_126.validate(var_127)
    var_129 = module_0.Integer()
    var_130 = {var_116: var_129}
    var_131 = module_0.String()
    var_132 = module_0.Object(pattern_properties=var_130, additional_properties=var_131)
    var_133 = 'key'
    var_134 = 'UPPERCASE'
    var_135 = 123
    var_136 = {var_133: var_135, var_134: var_135}
    var_137 = var_132.validate(var_136)
    var_138 = module_0.Integer()
    var_139 = {var_137: var_138}
    var_140 = module_0.String()
    var_141 = module_0.Object(pattern_properties=var_139, additional_properties=var_140)
    var_142 = 'key'
    var_143 = 'UPPERCASE'
    var_144 = 'not an integer'
    var_145 = 'string'
    var_146 = {var_142: var_144, var_143: var_145}
    var_147 = var_141.validate(var_146)
    var_148 = module_0.Integer()
    var_149 = {var_146: var_148}
    var_150 = module_0.String()
    var_151 = module_0.Object(pattern_properties=var_149, additional_properties=var_150)
    var_152 = 'key'
    var_153 = 'UPPERCASE'
    var_154 = 'not an integer'
    var_155 = 123
    var_156 = {var_152: var_154, var_153: var_155}
    var_157 = var_151.validate(var_156)
    var_158 = module_0.Integer()
    var_159 = {var_156: var_158}
    var_160 = module_0.String()
    var_161 = module_0.Object(pattern_properties=var_159, additional_properties=var_160)
    var_162 = {var_43: var_60, var_74: var_75}
    var_163 = var_161.validate(var_162)
    var_164 = module_0.Integer()
    var_165 = {var_156: var_164}
    var_166 = module_0.Object(pattern_properties=var_165, additional_properties=var_155)
    var_167 = 'key'
    var_168 = 'UPPERCASE'
    var_169 = 123
    var_170 = 'string'
    var_171 = {var_167: var_169, var_168: var_170}
    var_172 = var_166.validate(var_171)
    var_173 = module_0.Integer()
    var_174 = {var_171: var_173}
    var_175 = module_0.Object(pattern_properties=var_174, additional_properties=var_167)
    var_176 = {var_43: var_60, var_74: var_75}
    var_177 = var_175.validate(var_176)
    var_178 = module_0.Integer()
    var_179 = {var_171: var_178}
    var_180 = module_0.Object(pattern_properties=var_179, additional_properties=var_168)



# Parsed testcases at query #4
#--------------------------


import datetime as module_1
import ipaddress as module_3
import uuid as module_2


def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = True
    var_8 = module_0.Choice(choices=var_6)
    var_9 = None
    var_10 = var_8.validate(var_9)
    assert var_10 is None
    var_11 = (var_0, var_1)
    var_12 = (var_3, var_4)
    var_13 = [var_11, var_12]
    var_14 = False
    var_15 = module_0.Choice(choices=var_13)
    var_16 = None
    var_17 = var_15.validate(var_16)
    var_18 = (var_16, var_17)
    var_19 = (var_3, var_4)
    var_20 = [var_18, var_19]
    var_21 = module_0.Choice(choices=var_20)
    var_22 = var_21.validate(var_16)
    assert var_22 == 'a'
    var_23 = (var_16, var_17)
    var_24 = (var_3, var_4)
    var_25 = [var_23, var_24]
    var_26 = module_0.Choice(choices=var_25)
    var_27 = 'c'
    var_28 = var_26.validate(var_27)
    var_29 = (var_27, var_28)
    var_30 = (var_3, var_4)
    var_31 = [var_29, var_30]
    var_32 = module_0.Choice(choices=var_31, coerce_types=var_7)
    var_33 = ''
    var_34 = var_32.validate(var_33)
    assert var_34 is None
    var_35 = (var_27, var_28)
    var_36 = (var_3, var_4)
    var_37 = [var_35, var_36]
    var_38 = module_0.Choice(choices=var_37)
    var_39 = ''
    var_40 = var_38.validate(var_39)
    var_41 = (var_39, var_40)
    var_42 = (var_3, var_4)
    var_43 = [var_41, var_42]
    var_44 = module_0.Choice(choices=var_43, coerce_types=var_14)
    var_45 = ''
    var_46 = var_44.validate(var_45)
    var_47 = (var_45, var_46)
    var_48 = (var_3, var_4)
    var_49 = [var_47, var_48]
    var_50 = module_0.Choice(choices=var_49)
    var_51 = 123
    var_52 = var_50.validate(var_51)
    var_53 = (var_51, var_52)
    var_54 = (var_3, var_4)
    var_55 = [var_53, var_54]
    var_56 = module_0.Choice(choices=var_55)
    var_57 = 'a'
    var_58 = 'A'
    var_59 = (var_57, var_58)
    var_60 = var_56.validate(var_59)
    var_61 = (var_57, var_58)
    var_62 = (var_60, var_4)
    var_63 = [var_61, var_62]
    var_64 = module_0.Choice(choices=var_63)
    var_65 = 'a'
    var_66 = [var_65]
    var_67 = var_64.validate(var_66)
    var_68 = (var_65, var_66)
    var_69 = (var_60, var_4)
    var_70 = [var_68, var_69]
    var_71 = module_0.Choice(choices=var_70)
    var_72 = 'key'
    var_73 = 'value'
    var_74 = {var_72: var_73}
    var_75 = var_71.validate(var_74)
    var_76 = (var_72, var_73)
    var_77 = (var_75, var_4)
    var_78 = [var_76, var_77]
    var_79 = module_0.Choice(choices=var_78)
    var_80 = 'a'
    var_81 = {var_80}
    var_82 = var_79.validate(var_81)
    var_83 = (var_80, var_81)
    var_84 = (var_75, var_4)
    var_85 = [var_83, var_84]
    var_86 = module_0.Choice(choices=var_85)
    var_87 = 'a'
    var_88 = {var_87}
    var_89 = frozenset(var_88)
    var_90 = var_86.validate(var_89)
    var_91 = (var_87, var_88)
    var_92 = (var_90, var_4)
    var_93 = [var_91, var_92]
    var_94 = module_0.Choice(choices=var_93)
    var_95 = 1
    var_96 = range(var_95)
    var_97 = var_94.validate(var_96)
    var_98 = (var_95, var_96)
    var_99 = (var_90, var_4)
    var_100 = [var_98, var_99]
    var_101 = module_0.Choice(choices=var_100)
    var_102 = b'a'
    var_103 = var_101.validate(var_102)
    var_104 = (var_102, var_103)
    var_105 = (var_90, var_4)
    var_106 = [var_104, var_105]
    var_107 = module_0.Choice(choices=var_106)
    var_108 = b'a'
    var_109 = bytearray(var_108)
    var_110 = var_107.validate(var_109)
    var_111 = (var_108, var_109)
    var_112 = (var_90, var_4)
    var_113 = [var_111, var_112]
    var_114 = module_0.Choice(choices=var_113)
    var_115 = b'a'
    var_116 = memoryview(var_115)
    var_117 = var_114.validate(var_116)
    var_118 = (var_115, var_116)
    var_119 = (var_90, var_4)
    var_120 = [var_118, var_119]
    var_121 = module_0.Choice(choices=var_120)
    var_122 = 1
    var_123 = 2
    var_124 = complex(var_122, var_123)
    var_125 = var_121.validate(var_124)
    var_126 = (var_122, var_123)
    var_127 = (var_125, var_4)
    var_128 = [var_126, var_127]
    var_129 = module_0.Choice(choices=var_128)
    var_130 = '1.23'
    var_131 = var_129.validate(var_123)
    var_132 = (var_130, var_123)
    var_133 = (var_125, var_4)
    var_134 = [var_132, var_133]
    var_135 = module_0.Choice(choices=var_134)
    var_136 = 1
    var_137 = 2
    var_138 = var_135.validate(var_131)
    var_139 = (var_136, var_137)
    var_140 = (var_138, var_4)
    var_141 = [var_139, var_140]
    var_142 = module_0.Choice(choices=var_141)
    var_143 = var_142.validate(var_136)
    var_144 = (var_136, var_143)
    var_145 = (var_138, var_4)
    var_146 = [var_144, var_145]
    var_147 = module_0.Choice(choices=var_146)
    var_148 = var_147.validate(var_136)
    var_149 = (var_136, var_148)
    var_150 = (var_138, var_4)
    var_151 = [var_149, var_150]
    var_152 = module_0.Choice(choices=var_151)
    var_153 = module_1.time()
    var_154 = var_152.validate(var_153)
    var_155 = (var_153, var_154)
    var_156 = (var_138, var_4)
    var_157 = [var_155, var_156]
    var_158 = module_0.Choice(choices=var_157)
    var_159 = 1
    var_160 = module_1.timedelta()
    var_161 = var_158.validate(var_160)
    var_162 = (var_159, var_160)
    var_163 = (var_138, var_4)
    var_164 = [var_162, var_163]
    var_165 = module_0.Choice(choices=var_164)
    var_166 = var_165.validate(var_159)
    var_167 = (var_159, var_166)
    var_168 = (var_138, var_4)
    var_169 = [var_167, var_168]
    var_170 = module_0.Choice(choices=var_169)
    var_171 = module_2.uuid4()
    var_172 = var_170.validate(var_171)
    var_173 = (var_171, var_172)
    var_174 = (var_138, var_4)
    var_175 = [var_173, var_174]
    var_176 = module_0.Choice(choices=var_175)
    var_177 = '192.168.0.1'
    var_178 = module_3.IPv4Address(var_177)
    var_179 = var_176.validate(var_178)
    var_180 = (var_177, var_178)
    var_181 = (var_138, var_4)
    var_182 = [var_180, var_181]
    var_183 = module_0.Choice(choices=var_182)
    var_184 = '::1'
    var_185 = module_3.IPv6Address(var_184)
    var_186 = var_183.validate(var_185)
    var_187 = (var_184, var_185)
    var_188 = (var_138, var_4)
    var_189 = [var_187, var_188]
    var_190 = module_0.Choice(choices=var_189)
    var_191 = '192.168.0.0/24'
    var_192 = module_3.IPv4Network(var_191)
    var_193 = var_190.validate(var_192)
    var_194 = (var_191, var_192)
    var_195 = (var_138, var_4)
    var_196 = [var_194, var_195]
    var_197 = module_0.Choice(choices=var_196)
    var_198 = '::/0'
    var_199 = module_3.IPv6Network(var_198)
    var_200 = var_197.validate(var_199)
    var_201 = (var_198, var_199)
    var_202 = (var_138, var_4)
    var_203 = [var_201, var_202]
    var_204 = module_0.Choice(choices=var_203)
    var_205 = '192.168.0.1/24'
    var_206 = module_3.IPv4Interface(var_205)
    var_207 = var_204.validate(var_206)
    var_208 = (var_205, var_206)
    var_209 = (var_138, var_4)
    var_210 = [var_208, var_209]
    var_211 = module_0.Choice(choices=var_210)
    var_212 = '::1/128'
    var_213 = module_3.IPv6Interface(var_212)
    var_214 = var_211.validate(var_213)
    var_215 = (var_212, var_213)
    var_216 = (var_138, var_4)
    var_217 = [var_215, var_216]
    var_218 = module_0.Choice(choices=var_217)
    var_219 = '/tmp'
    var_220 = var_218.validate(var_213)



# Parsed testcases at query #5
#--------------------------




# Parsed testcases at query #6
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = True
    var_8 = module_0.Choice(choices=var_6)
    var_9 = None
    var_10 = var_8.validate(var_9)
    assert var_10 is None
    var_11 = (var_0, var_1)
    var_12 = (var_3, var_4)
    var_13 = [var_11, var_12]
    var_14 = False
    var_15 = module_0.Choice(choices=var_13)
    var_16 = None
    var_17 = var_15.validate(var_16)
    var_18 = (var_16, var_17)
    var_19 = (var_3, var_4)
    var_20 = [var_18, var_19]
    var_21 = module_0.Choice(choices=var_20)
    var_22 = 'c'
    var_23 = var_21.validate(var_22)
    var_24 = (var_22, var_23)
    var_25 = (var_3, var_4)
    var_26 = [var_24, var_25]
    var_27 = module_0.Choice(choices=var_26)
    var_28 = var_27.validate(var_22)
    assert var_28 == 'a'
    var_29 = (var_22, var_23)
    var_30 = (var_3, var_4)
    var_31 = [var_29, var_30]
    var_32 = module_0.Choice(choices=var_31, coerce_types=var_7)
    var_33 = ''
    var_34 = var_32.validate(var_33)
    assert var_34 is None
    var_35 = (var_22, var_23)
    var_36 = (var_3, var_4)
    var_37 = [var_35, var_36]
    var_38 = module_0.Choice(choices=var_37, coerce_types=var_7)
    var_39 = ''
    var_40 = var_38.validate(var_39)
    var_41 = (var_39, var_40)
    var_42 = (var_3, var_4)
    var_43 = [var_41, var_42]
    var_44 = module_0.Choice(choices=var_43, coerce_types=var_14)
    var_45 = ''
    var_46 = var_44.validate(var_45)
    var_47 = (var_45, var_46)
    var_48 = (var_3, var_4)
    var_49 = [var_47, var_48]
    var_50 = module_0.Choice(choices=var_49, coerce_types=var_14)
    var_51 = ''
    var_52 = var_50.validate(var_51)



# Parsed testcases at query #7
#--------------------------



def test_case_0():
    var_0 = module_0.Decimal()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    var_3 = '1.5'
    var_4 = '2.0'



# Parsed testcases at query #8
#--------------------------



def test_case_0():
    var_0 = module_0.Decimal()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    var_3 = module_0.Decimal()
    var_4 = '10.5'
    var_5 = module_0.Decimal()
    var_6 = '10'



# Parsed testcases at query #9
#--------------------------



def test_case_0():
    var_0 = 42
    var_1 = module_0.Const(var_0)
    var_2 = None
    var_3 = module_0.Const(var_2)
    var_4 = 'test'
    var_5 = 'const'
    var_6 = 'Custom error'
    var_7 = {var_5: var_6}
    var_8 = module_0.Const(var_4)
    var_9 = 1
    var_10 = True
    var_11 = module_0.Const(var_9)



# Parsed testcases at query #10
#--------------------------



def test_case_0():
    var_0 = module_0.Array()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    var_3 = module_0.String()
    var_4 = module_0.Integer()
    var_5 = [var_3, var_4]
    var_6 = module_0.Array(var_5)
    var_7 = 'hello'
    var_8 = 123
    var_9 = [var_7, var_8]
    var_10 = var_6.serialize(var_9)
    var_11 = module_0.String()
    var_12 = module_0.Array(var_11)
    var_13 = 'world'
    var_14 = [var_7, var_13]
    var_15 = var_12.serialize(var_14)
    var_16 = module_0.Array()
    var_17 = [var_7, var_8]
    var_18 = var_16.serialize(var_17)
    var_19 = module_0.String()
    var_20 = module_0.Integer()
    var_21 = [var_19, var_20]
    var_22 = module_0.Array(var_21)
    var_23 = 'extra'
    var_24 = [var_7, var_8, var_23]
    var_25 = var_22.serialize(var_24)
    var_26 = module_0.String()
    var_27 = module_0.Integer()
    var_28 = [var_26, var_27]
    var_29 = module_0.Array(var_28)
    var_30 = [var_7]
    var_31 = var_29.serialize(var_30)
    var_32 = module_0.String()
    var_33 = module_0.Integer()
    var_34 = [var_32, var_33]
    var_35 = module_0.Array(var_34)
    var_36 = []
    var_37 = var_35.serialize(var_36)
    var_38 = module_0.String()
    var_39 = module_0.Array(var_38)
    var_40 = []
    var_41 = var_39.serialize(var_40)
    var_42 = module_0.Array()
    var_43 = []
    var_44 = var_42.serialize(var_43)
    var_45 = module_0.String()
    var_46 = module_0.Integer()
    var_47 = [var_45, var_46]
    var_48 = module_0.Array(var_47)
    var_49 = [var_1, var_8]
    var_50 = var_48.serialize(var_49)
    var_51 = module_0.String()
    var_52 = module_0.Array(var_51)
    var_53 = [var_1, var_13]
    var_54 = var_52.serialize(var_53)
    var_55 = module_0.Array()
    var_56 = [var_1, var_8]
    var_57 = var_55.serialize(var_56)
    var_58 = module_0.String()
    var_59 = module_0.Array(var_58)
    var_60 = [var_59]
    var_61 = module_0.Array(var_60)
    var_62 = [var_7, var_13]
    var_63 = [var_62]
    var_64 = var_61.serialize(var_63)
    var_65 = module_0.String()
    var_66 = module_0.Array(var_65)
    var_67 = module_0.Array(var_66)
    var_68 = [var_7, var_13]
    var_69 = [var_68]
    var_70 = var_67.serialize(var_69)
    var_71 = module_0.Array()
    var_72 = [var_7, var_13]
    var_73 = [var_72]
    var_74 = var_71.serialize(var_73)
    var_75 = 'key'
    var_76 = module_0.String()
    var_77 = {var_75: var_76}
    var_78 = module_0.Object(properties=var_77)
    var_79 = [var_78]
    var_80 = module_0.Array(var_79)
    var_81 = 'value'
    var_82 = {var_75: var_81}
    var_83 = [var_82]
    var_84 = var_80.serialize(var_83)
    var_85 = module_0.String()
    var_86 = {var_75: var_85}
    var_87 = module_0.Object(properties=var_86)
    var_88 = module_0.Array(var_87)
    var_89 = {var_75: var_81}
    var_90 = [var_89]
    var_91 = var_88.serialize(var_90)
    var_92 = module_0.Array()
    var_93 = {var_75: var_81}
    var_94 = [var_93]
    var_95 = var_92.serialize(var_94)
    var_96 = module_0.Boolean()
    var_97 = [var_96]
    var_98 = module_0.Array(var_97)
    var_99 = True
    var_100 = [var_99]
    var_101 = var_98.serialize(var_100)
    var_102 = module_0.Boolean()
    var_103 = module_0.Array(var_102)
    var_104 = [var_99]
    var_105 = var_103.serialize(var_104)
    var_106 = module_0.Array()
    var_107 = [var_99]
    var_108 = var_106.serialize(var_107)
    var_109 = module_0.Integer()
    var_110 = [var_109]
    var_111 = module_0.Array(var_110)
    var_112 = [var_8]
    var_113 = var_111.serialize(var_112)
    var_114 = module_0.Integer()
    var_115 = module_0.Array(var_114)
    var_116 = [var_8]
    var_117 = var_115.serialize(var_116)
    var_118 = module_0.Array()
    var_119 = [var_8]
    var_120 = var_118.serialize(var_119)
    var_121 = module_0.String()
    var_122 = [var_121]
    var_123 = module_0.Array(var_122)
    var_124 = [var_7]
    var_125 = var_123.serialize(var_124)
    var_126 = module_0.String()
    var_127 = module_0.Array(var_126)
    var_128 = [var_7]
    var_129 = var_127.serialize(var_128)
    var_130 = module_0.Array()
    var_131 = [var_7]
    var_132 = var_130.serialize(var_131)
    var_133 = module_0.String()
    var_134 = module_0.Integer()
    var_135 = [var_133, var_134]
    var_136 = module_0.Array(var_135)
    var_137 = [var_7, var_8]
    var_138 = var_136.serialize(var_137)
    var_139 = module_0.Field()
    var_140 = module_0.Array(var_139)
    var_141 = [var_7, var_8]
    var_142 = var_140.serialize(var_141)
    var_143 = module_0.Array()
    var_144 = [var_7, var_8]
    var_145 = var_143.serialize(var_144)
    var_146 = module_0.String()
    var_147 = module_0.Array(var_146)
    var_148 = [var_147]
    var_149 = module_0.Array(var_148)
    var_150 = [var_7, var_13]
    var_151 = [var_150]
    var_152 = var_149.serialize(var_151)
    var_153 = module_0.String()
    var_154 = module_0.Array(var_153)
    var_155 = module_0.Array(var_154)
    var_156 = [var_7, var_13]
    var_157 = [var_156]
    var_158 = var_155.serialize(var_157)
    var_159 = module_0.Array()
    var_160 = [var_7, var_13]
    var_161 = [var_160]
    var_162 = var_159.serialize(var_161)
    var_163 = module_0.String()
    var_164 = {var_75: var_163}
    var_165 = module_0.Object(properties=var_164)
    var_166 = [var_165]
    var_167 = module_0.Array(var_166)
    var_168 = {var_75: var_81}
    var_169 = [var_168]
    var_170 = var_167.serialize(var_169)
    var_171 = module_0.String()
    var_172 = {var_75: var_171}
    var_173 = module_0.Object(properties=var_172)
    var_174 = module_0.Array(var_173)
    var_175 = {var_75: var_81}
    var_176 = [var_175]
    var_177 = var_174.serialize(var_176)
    var_178 = module_0.Array()
    var_179 = {var_75: var_81}
    var_180 = [var_179]
    var_181 = var_178.serialize(var_180)
    var_182 = module_0.Boolean()
    var_183 = [var_182]
    var_184 = module_0.Array(var_183)
    var_185 = [var_99]
    var_186 = var_184.serialize(var_185)
    var_187 = module_0.Boolean()
    var_188 = module_0.Array(var_187)
    var_189 = [var_99]
    var_190 = var_188.serialize(var_189)
    var_191 = module_0.Array()
    var_192 = [var_99]
    var_193 = var_191.serialize(var_192)
    var_194 = module_0.Integer()
    var_195 = [var_194]
    var_196 = module_0.Array(var_195)
    var_197 = [var_8]
    var_198 = var_196.serialize(var_197)
    var_199 = module_0.Integer()
    var_200 = module_0.Array(var_199)
    var_201 = [var_8]
    var_202 = var_200.serialize(var_201)
    var_203 = module_0.Array()
    var_204 = [var_8]
    var_205 = var_203.serialize(var_204)
    var_206 = module_0.String()
    var_207 = [var_206]
    var_208 = module_0.Array(var_207)
    var_209 = [var_7]
    var_210 = var_208.serialize(var_209)



# Parsed testcases at query #11
#--------------------------



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
    var_11 = module_0.Number(coerce_types=var_6)
    var_12 = ''
    var_13 = var_11.validate(var_12)
    var_14 = module_0.Number()
    var_15 = True
    var_16 = var_14.validate(var_15)
    var_17 = 3.14
    var_18 = var_14.validate(var_17)
    var_19 = module_0.Number(coerce_types=var_4)
    var_20 = 'abc'
    var_21 = var_19.validate(var_20)
    var_22 = module_0.Number()
    var_23 = '123'
    var_24 = var_22.validate(var_23)
    assert var_24 == 123
    var_25 = module_0.Number()
    var_26 = 'abc'
    var_27 = var_25.validate(var_26)
    var_28 = module_0.Number()
    var_29 = 'inf'
    var_30 = float(var_29)
    var_31 = var_28.validate(var_30)
    var_32 = 10
    var_33 = module_0.Number(minimum=var_32)
    var_34 = 5
    var_35 = var_33.validate(var_34)
    var_36 = module_0.Number(exclusive_minimum=var_32)
    var_37 = 10
    var_38 = var_36.validate(var_37)
    var_39 = module_0.Number(maximum=var_32)
    var_40 = 15
    var_41 = var_39.validate(var_40)
    var_42 = module_0.Number(exclusive_maximum=var_32)
    var_43 = 10
    var_44 = var_42.validate(var_43)
    var_45 = 5
    var_46 = module_0.Number(multiple_of=var_45)
    var_47 = 7
    var_48 = var_46.validate(var_47)
    var_49 = module_0.Number(multiple_of=var_45)
    var_50 = var_49.validate(var_32)
    assert var_50 == 10
    var_51 = 0.5
    var_52 = module_0.Number(multiple_of=var_51)
    var_53 = 1.5
    var_54 = var_52.validate(var_53)
    var_55 = module_0.Number(multiple_of=var_51)
    var_56 = 1.2
    var_57 = var_55.validate(var_56)
    var_58 = '0.01'
    var_59 = module_0.Number(precision=var_58)
    var_60 = '1.234'
    var_61 = '1.23'
    var_62 = module_0.Number(precision=var_58)
    var_63 = 1.234
    var_64 = var_62.validate(var_63)
    var_65 = module_0.Number(precision=var_58)
    var_66 = var_65.validate(var_60)
    var_67 = module_0.Number(precision=var_58)
    var_68 = 'abc'
    var_69 = var_67.validate(var_68)
    var_70 = module_0.Number(precision=var_58)
    var_71 = module_0.Number(precision=var_58)
    var_72 = '-1.234'
    var_73 = '-1.23'
    var_74 = module_0.Number(precision=var_58)
    var_75 = '0'
    var_76 = '0.00'
    var_77 = module_0.Number(precision=var_58)
    var_78 = '1234567890.123456789'
    var_79 = '1234567890.12'
    var_80 = module_0.Number(precision=var_58)
    var_81 = '0.0000000001'
    var_82 = module_0.Number(precision=var_58)
    var_83 = '-0.0000000001'
    var_84 = '-0.00'
    var_85 = module_0.Number(precision=var_58)
    var_86 = '-1234567890.123456789'
    var_87 = '-1234567890.12'
    var_88 = module_0.Number(precision=var_58)
    var_89 = '  1.234  '
    var_90 = var_88.validate(var_89)
    var_91 = module_0.Number(precision=var_58)
    var_92 = '+1.234'
    var_93 = var_91.validate(var_92)
    var_94 = module_0.Number(precision=var_58)
    var_95 = var_94.validate(var_72)
    var_96 = module_0.Number(precision=var_58)
    var_97 = '1.234e-2'
    var_98 = var_96.validate(var_97)
    var_99 = module_0.Number(precision=var_58)
    var_100 = '+1.234e-2'
    var_101 = var_99.validate(var_100)
    var_102 = module_0.Number(precision=var_58)
    var_103 = '-1.234e-2'
    var_104 = var_102.validate(var_103)
    var_105 = '-0.01'
    var_106 = module_0.Number(precision=var_58)
    var_107 = '1.234e+10'
    var_108 = var_106.validate(var_107)
    var_109 = '12340000000.00'
    var_110 = module_0.Number(precision=var_58)
    var_111 = '1.234e-10'
    var_112 = var_110.validate(var_111)



# Parsed testcases at query #12
#--------------------------



def test_case_0():
    var_0 = True
    var_1 = module_0.Boolean()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None
    var_4 = False
    var_5 = module_0.Boolean()
    var_6 = None
    var_7 = var_5.validate(var_6)
    var_8 = module_0.Boolean()
    var_9 = var_8.validate(var_6)
    assert var_9 is True
    var_10 = module_0.Boolean(coerce_types=var_6)
    var_11 = 'true'
    var_12 = var_10.validate(var_11)
    assert var_12 is True
    var_13 = module_0.Boolean(coerce_types=var_6)
    var_14 = 'invalid'
    var_15 = var_13.validate(var_14)
    var_16 = module_0.Boolean(coerce_types=var_14)
    var_17 = 'null'
    var_18 = var_16.validate(var_17)
    assert var_18 is None
    var_19 = module_0.Boolean(coerce_types=var_14)
    var_20 = 'off'
    var_21 = var_19.validate(var_20)
    assert var_21 is False
    var_22 = module_0.Boolean(coerce_types=var_14)
    var_23 = var_22.validate(var_14)
    assert var_23 is True
    var_24 = module_0.Boolean(coerce_types=var_14)
    var_25 = 2
    var_26 = var_24.validate(var_25)
    var_27 = module_0.Boolean(coerce_types=var_25)
    var_28 = 3.14
    var_29 = var_27.validate(var_28)
    var_30 = module_0.Boolean(coerce_types=var_28)
    var_31 = 1
    var_32 = 2
    var_33 = 3
    var_34 = [var_31, var_32, var_33]
    var_35 = var_30.validate(var_34)
    var_36 = module_0.Boolean(coerce_types=var_31)
    var_37 = 'key'
    var_38 = 'value'
    var_39 = {var_37: var_38}
    var_40 = var_36.validate(var_39)
    var_41 = module_0.Boolean(coerce_types=var_37)
    var_42 = ''
    var_43 = var_41.validate(var_42)
    assert var_43 is None
    var_44 = module_0.Boolean(coerce_types=var_37)
    var_45 = var_44.validate(var_42)
    assert var_45 is False
    var_46 = module_0.Boolean(coerce_types=var_37)
    var_47 = 'on'
    var_48 = var_46.validate(var_47)
    assert var_48 is True
    var_49 = module_0.Boolean(coerce_types=var_37)
    var_50 = var_49.validate(var_20)
    assert var_50 is False
    var_51 = module_0.Boolean(coerce_types=var_37)
    var_52 = '1'
    var_53 = var_51.validate(var_52)
    assert var_53 is True
    var_54 = module_0.Boolean(coerce_types=var_37)
    var_55 = '0'
    var_56 = var_54.validate(var_55)
    assert var_56 is False
    var_57 = module_0.Boolean(coerce_types=var_37)
    var_58 = var_57.validate(var_40)
    assert var_58 is True
    var_59 = module_0.Boolean(coerce_types=var_37)
    var_60 = 'false'
    var_61 = var_59.validate(var_60)
    assert var_61 is False
    var_62 = module_0.Boolean(coerce_types=var_37)
    var_63 = var_62.validate(var_35)
    assert var_63 is None
    var_64 = module_0.Boolean(coerce_types=var_37)
    var_65 = 'none'
    var_66 = var_64.validate(var_65)
    assert var_66 is None
    var_67 = module_0.Boolean(coerce_types=var_37)
    var_68 = 'null'
    var_69 = var_67.validate(var_68)
    var_70 = module_0.Boolean(coerce_types=var_68)
    var_71 = 'none'
    var_72 = var_70.validate(var_71)
    var_73 = module_0.Boolean(coerce_types=var_71)
    var_74 = 'True'
    var_75 = var_73.validate(var_74)
    assert var_75 is True
    var_76 = module_0.Boolean(coerce_types=var_71)
    var_77 = 'False'
    var_78 = var_76.validate(var_77)
    assert var_78 is False
    var_79 = module_0.Boolean(coerce_types=var_71)
    var_80 = 'On'
    var_81 = var_79.validate(var_80)
    assert var_81 is True
    var_82 = module_0.Boolean(coerce_types=var_71)
    var_83 = 'Off'
    var_84 = var_82.validate(var_83)
    assert var_84 is False
    var_85 = module_0.Boolean(coerce_types=var_71)
    var_86 = '1.0'
    var_87 = var_85.validate(var_86)
    var_88 = module_0.Boolean(coerce_types=var_86)
    var_89 = '0.0'
    var_90 = var_88.validate(var_89)
    var_91 = module_0.Boolean(coerce_types=var_89)
    var_92 = 'yes'
    var_93 = var_91.validate(var_92)
    var_94 = module_0.Boolean(coerce_types=var_92)
    var_95 = 'no'
    var_96 = var_94.validate(var_95)
    var_97 = module_0.Boolean(coerce_types=var_95)
    var_98 = 'y'
    var_99 = var_97.validate(var_98)
    var_100 = module_0.Boolean(coerce_types=var_98)
    var_101 = 'n'
    var_102 = var_100.validate(var_101)
    var_103 = module_0.Boolean(coerce_types=var_101)
    var_104 = 't'
    var_105 = var_103.validate(var_104)
    var_106 = module_0.Boolean(coerce_types=var_104)
    var_107 = 'f'
    var_108 = var_106.validate(var_107)
    var_109 = module_0.Boolean(coerce_types=var_107)
    var_110 = 'true '
    var_111 = var_109.validate(var_110)
    assert var_111 is True
    var_112 = module_0.Boolean(coerce_types=var_107)
    var_113 = ' false'
    var_114 = var_112.validate(var_113)
    assert var_114 is False
    var_115 = module_0.Boolean(coerce_types=var_107)
    var_116 = ' true '
    var_117 = var_115.validate(var_116)
    assert var_117 is True
    var_118 = module_0.Boolean(coerce_types=var_107)
    var_119 = ' false '
    var_120 = var_118.validate(var_119)



# Parsed testcases at query #13
#--------------------------



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
    var_11 = module_0.Number(coerce_types=var_6)
    var_12 = ''
    var_13 = var_11.validate(var_12)
    var_14 = module_0.Number()
    var_15 = True
    var_16 = var_14.validate(var_15)
    var_17 = module_0.Number()
    var_18 = False
    var_19 = var_17.validate(var_18)
    var_20 = 3.14
    var_21 = var_17.validate(var_20)
    var_22 = module_0.Number(coerce_types=var_4)
    var_23 = '123'
    var_24 = var_22.validate(var_23)
    var_25 = module_0.Number()
    var_26 = 'abc'
    var_27 = var_25.validate(var_26)
    var_28 = module_0.Number()
    var_29 = 'inf'
    var_30 = float(var_29)
    var_31 = var_28.validate(var_30)
    var_32 = module_0.Number()
    var_33 = 'nan'
    var_34 = float(var_33)
    var_35 = var_32.validate(var_34)
    var_36 = 10
    var_37 = module_0.Number(minimum=var_36)
    var_38 = 5
    var_39 = var_37.validate(var_38)
    var_40 = module_0.Number(exclusive_minimum=var_36)
    var_41 = 10
    var_42 = var_40.validate(var_41)
    var_43 = 100
    var_44 = module_0.Number(maximum=var_43)
    var_45 = 150
    var_46 = var_44.validate(var_45)
    var_47 = module_0.Number(exclusive_maximum=var_43)
    var_48 = 100
    var_49 = var_47.validate(var_48)
    var_50 = 5
    var_51 = module_0.Number(multiple_of=var_50)
    var_52 = 7
    var_53 = var_51.validate(var_52)
    var_54 = module_0.Number(multiple_of=var_50)
    var_55 = var_54.validate(var_36)
    assert var_55 == 10
    var_56 = module_0.Number()
    var_57 = 42
    var_58 = var_56.validate(var_57)
    assert var_58 == 42



####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------



def test_case_0():
    var_0 = module_0.Decimal()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    var_3 = module_0.Decimal()
    var_4 = '10.5'
    var_5 = module_0.Decimal()
    var_6 = '10'



# Parsed testcases at query #2
#--------------------------



def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = (var_0, var_1)
    var_3 = 'key2'
    var_4 = 'value2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = True
    var_8 = module_0.Choice(choices=var_6)
    var_9 = None
    var_10 = var_8.validate(var_9)
    assert var_10 is None
    var_11 = (var_0, var_1)
    var_12 = (var_3, var_4)
    var_13 = [var_11, var_12]
    var_14 = False
    var_15 = module_0.Choice(choices=var_13)
    var_16 = None
    var_17 = var_15.validate(var_16)
    var_18 = (var_16, var_17)
    var_19 = (var_3, var_4)
    var_20 = [var_18, var_19]
    var_21 = module_0.Choice(choices=var_20)
    var_22 = var_21.validate(var_16)
    assert var_22 == 'key1'
    var_23 = (var_16, var_17)
    var_24 = (var_3, var_4)
    var_25 = [var_23, var_24]
    var_26 = module_0.Choice(choices=var_25)
    var_27 = 'key3'
    var_28 = var_26.validate(var_27)
    var_29 = (var_27, var_28)
    var_30 = (var_3, var_4)
    var_31 = [var_29, var_30]
    var_32 = module_0.Choice(choices=var_31, coerce_types=var_7)
    var_33 = ''
    var_34 = var_32.validate(var_33)
    assert var_34 is None
    var_35 = (var_27, var_28)
    var_36 = (var_3, var_4)
    var_37 = [var_35, var_36]
    var_38 = module_0.Choice(choices=var_37, coerce_types=var_7)
    var_39 = ''
    var_40 = var_38.validate(var_39)
    var_41 = (var_39, var_40)
    var_42 = (var_3, var_4)
    var_43 = [var_41, var_42]
    var_44 = module_0.Choice(choices=var_43, coerce_types=var_14)
    var_45 = ''
    var_46 = var_44.validate(var_45)
    var_47 = (var_45, var_46)
    var_48 = (var_3, var_4)
    var_49 = [var_47, var_48]
    var_50 = module_0.Choice(choices=var_49, coerce_types=var_14)
    var_51 = ''
    var_52 = var_50.validate(var_51)
    var_53 = (var_51, var_52)
    var_54 = (var_3, var_4)
    var_55 = [var_53, var_54]
    var_56 = module_0.Choice(choices=var_55)
    var_57 = 123
    var_58 = var_56.validate(var_57)
    var_59 = (var_57, var_58)
    var_60 = (var_3, var_4)
    var_61 = [var_59, var_60]
    var_62 = module_0.Choice(choices=var_61)
    var_63 = 'key3'
    var_64 = var_62.validate(var_63)
    var_65 = (var_63, var_64)
    var_66 = (var_3, var_4)
    var_67 = [var_65, var_66]
    var_68 = module_0.Choice(choices=var_67)
    var_69 = var_68.validate(var_3)
    assert var_69 == 'key2'
    var_70 = (var_63, var_64)
    var_71 = (var_3, var_4)
    var_72 = [var_70, var_71]
    var_73 = module_0.Choice(choices=var_72)
    var_74 = 'KEY1'
    var_75 = var_73.validate(var_74)
    var_76 = (var_74, var_75)
    var_77 = (var_3, var_4)
    var_78 = [var_76, var_77]
    var_79 = module_0.Choice(choices=var_78)
    var_80 = ' key1 '
    var_81 = var_79.validate(var_80)
    var_82 = (var_80, var_81)
    var_83 = (var_3, var_4)
    var_84 = [var_82, var_83]
    var_85 = module_0.Choice(choices=var_84)
    var_86 = '  key1  '
    var_87 = var_85.validate(var_86)
    var_88 = (var_86, var_87)
    var_89 = (var_3, var_4)
    var_90 = [var_88, var_89]
    var_91 = module_0.Choice(choices=var_90)
    var_92 = 'key1\n'
    var_93 = var_91.validate(var_92)
    var_94 = (var_92, var_93)
    var_95 = (var_3, var_4)
    var_96 = [var_94, var_95]
    var_97 = module_0.Choice(choices=var_96)
    var_98 = 'key1\x00'
    var_99 = var_97.validate(var_98)
    var_100 = (var_98, var_99)
    var_101 = (var_3, var_4)
    var_102 = [var_100, var_101]
    var_103 = module_0.Choice(choices=var_102)
    var_104 = 'key1\x08'
    var_105 = var_103.validate(var_104)
    var_106 = (var_104, var_105)
    var_107 = (var_3, var_4)
    var_108 = [var_106, var_107]
    var_109 = module_0.Choice(choices=var_108)
    var_110 = 'key1\t'
    var_111 = var_109.validate(var_110)
    var_112 = (var_110, var_111)
    var_113 = (var_3, var_4)
    var_114 = [var_112, var_113]
    var_115 = module_0.Choice(choices=var_114)
    var_116 = 'key1\r'
    var_117 = var_115.validate(var_116)
    var_118 = (var_116, var_117)
    var_119 = (var_3, var_4)
    var_120 = [var_118, var_119]
    var_121 = module_0.Choice(choices=var_120)
    var_122 = 'key1\x0c'
    var_123 = var_121.validate(var_122)
    var_124 = (var_122, var_123)
    var_125 = (var_3, var_4)
    var_126 = [var_124, var_125]
    var_127 = module_0.Choice(choices=var_126)
    var_128 = 'key1\x0b'
    var_129 = var_127.validate(var_128)
    var_130 = (var_128, var_129)
    var_131 = (var_3, var_4)
    var_132 = [var_130, var_131]
    var_133 = module_0.Choice(choices=var_132)
    var_134 = 'key1\\'
    var_135 = var_133.validate(var_134)
    var_136 = (var_134, var_135)
    var_137 = (var_3, var_4)
    var_138 = [var_136, var_137]
    var_139 = module_0.Choice(choices=var_138)
    var_140 = 'key1"'
    var_141 = var_139.validate(var_140)
    var_142 = (var_140, var_141)
    var_143 = (var_3, var_4)
    var_144 = [var_142, var_143]
    var_145 = module_0.Choice(choices=var_144)
    var_146 = "key1'"
    var_147 = var_145.validate(var_146)
    var_148 = (var_146, var_147)
    var_149 = (var_3, var_4)
    var_150 = [var_148, var_149]
    var_151 = module_0.Choice(choices=var_150)
    var_152 = 'key1`'
    var_153 = var_151.validate(var_152)
    var_154 = (var_152, var_153)
    var_155 = (var_3, var_4)
    var_156 = [var_154, var_155]
    var_157 = module_0.Choice(choices=var_156)
    var_158 = 'key1~'
    var_159 = var_157.validate(var_158)
    var_160 = (var_158, var_159)
    var_161 = (var_3, var_4)
    var_162 = [var_160, var_161]
    var_163 = module_0.Choice(choices=var_162)
    var_164 = 'key1!'
    var_165 = var_163.validate(var_164)
    var_166 = (var_164, var_165)
    var_167 = (var_3, var_4)
    var_168 = [var_166, var_167]
    var_169 = module_0.Choice(choices=var_168)
    var_170 = 'key1@'
    var_171 = var_169.validate(var_170)
    var_172 = (var_170, var_171)
    var_173 = (var_3, var_4)
    var_174 = [var_172, var_173]
    var_175 = module_0.Choice(choices=var_174)
    var_176 = 'key1#'
    var_177 = var_175.validate(var_176)
    var_178 = (var_176, var_177)
    var_179 = (var_3, var_4)
    var_180 = [var_178, var_179]
    var_181 = module_0.Choice(choices=var_180)



# Parsed testcases at query #3
#--------------------------



def test_case_0():
    var_0 = module_0.Decimal()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    var_3 = module_0.Decimal()
    var_4 = '10.5'
    var_5 = module_0.Decimal()
    var_6 = 10.5
    var_7 = var_5.serialize(var_6)
    var_8 = module_0.Decimal()
    var_9 = 10
    var_10 = var_8.serialize(var_9)
    var_11 = module_0.Decimal()
    var_12 = '10.5'
    var_13 = var_11.serialize(var_12)
    var_14 = module_0.Decimal()
    var_15 = 'not a number'
    var_16 = var_14.serialize(var_15)
    var_17 = module_0.Decimal()
    var_18 = 10.5
    var_19 = 20.5
    var_20 = [var_18, var_19]
    var_21 = var_17.serialize(var_20)
    var_22 = module_0.Decimal()
    var_23 = 'value'
    var_24 = {var_23: var_18}
    var_25 = var_22.serialize(var_24)
    var_26 = module_0.Decimal()
    var_27 = True
    var_28 = var_26.serialize(var_27)
    var_29 = module_0.Decimal()
    var_30 = complex(var_18, var_19)
    var_31 = var_29.serialize(var_30)
    var_32 = 'All test cases passed!'
    var_33 = print(var_32)



# Parsed testcases at query #4
#--------------------------



def test_case_0():
    var_0 = True
    var_1 = module_0.Object()
    var_2 = None
    var_3 = var_1.validate(var_2)
    assert var_3 is None
    var_4 = False
    var_5 = module_0.Object()
    var_6 = None
    var_7 = var_5.validate(var_6)
    var_8 = module_0.Object()
    var_9 = 'not a dict'
    var_10 = var_8.validate(var_9)
    var_11 = module_0.Object()
    var_12 = 1
    var_13 = 'value'
    var_14 = {var_12: var_13}
    var_15 = var_11.validate(var_14)
    var_16 = '^[a-z]+$'
    var_17 = module_0.String(pattern=var_16)
    var_18 = module_0.Object(property_names=var_17)
    var_19 = 'UPPERCASE'
    var_20 = 'value'
    var_21 = {var_19: var_20}
    var_22 = var_18.validate(var_21)
    var_23 = 2
    var_24 = module_0.Object(min_properties=var_23)
    var_25 = 'key1'
    var_26 = 'value1'
    var_27 = {var_25: var_26}
    var_28 = var_24.validate(var_27)
    var_29 = module_0.Object(max_properties=var_23)
    var_30 = 'key1'
    var_31 = 'key2'
    var_32 = 'key3'
    var_33 = 'value1'
    var_34 = 'value2'
    var_35 = 'value3'
    var_36 = {var_30: var_33, var_31: var_34, var_32: var_35}
    var_37 = var_29.validate(var_36)
    var_38 = 'key1'
    var_39 = [var_38]
    var_40 = module_0.Object(required=var_39)
    var_41 = 'key2'
    var_42 = 'value2'
    var_43 = {var_41: var_42}
    var_44 = var_40.validate(var_43)
    var_45 = '^key.*$'
    var_46 = module_0.String()
    var_47 = {var_45: var_46}
    var_48 = module_0.Object(pattern_properties=var_47)
    var_49 = 'value1'
    var_50 = {var_38: var_49}
    var_51 = var_48.validate(var_50)
    var_52 = module_0.String()
    var_53 = {var_45: var_52}
    var_54 = module_0.Object(pattern_properties=var_53, additional_properties=var_43)
    var_55 = 'other'
    var_56 = 'value'
    var_57 = {var_55: var_56}
    var_58 = var_54.validate(var_57)
    var_59 = module_0.String()
    var_60 = {var_45: var_59}
    var_61 = module_0.Integer()
    var_62 = module_0.Object(pattern_properties=var_60, additional_properties=var_61)
    var_63 = 'other'
    var_64 = 123
    var_65 = {var_63: var_64}
    var_66 = var_62.validate(var_65)
    var_67 = module_0.String()
    var_68 = {var_45: var_67}
    var_69 = module_0.Object(pattern_properties=var_68, additional_properties=var_55)
    var_70 = 'value'
    var_71 = {var_63: var_70}
    var_72 = var_69.validate(var_71)
    var_73 = module_0.Integer()
    var_74 = {var_45: var_73}
    var_75 = module_0.Object(pattern_properties=var_74)
    var_76 = 'key1'
    var_77 = 'not an integer'
    var_78 = {var_76: var_77}
    var_79 = var_75.validate(var_78)
    var_80 = module_0.Integer()
    var_81 = {var_45: var_80}
    var_82 = module_0.Object(pattern_properties=var_81)
    var_83 = {var_38: var_64}
    var_84 = var_82.validate(var_83)
    var_85 = module_0.Integer()
    var_86 = {var_45: var_85}
    var_87 = module_0.Object(pattern_properties=var_86, additional_properties=var_76)
    var_88 = {var_38: var_64, var_63: var_70}
    var_89 = var_87.validate(var_88)
    var_90 = module_0.Integer()
    var_91 = {var_45: var_90}
    var_92 = module_0.Object(pattern_properties=var_91, additional_properties=var_78)
    var_93 = 'key1'
    var_94 = 'other'
    var_95 = 123
    var_96 = 'value'
    var_97 = {var_93: var_95, var_94: var_96}
    var_98 = var_92.validate(var_97)
    var_99 = module_0.Integer()
    var_100 = {var_45: var_99}
    var_101 = module_0.String()
    var_102 = module_0.Object(pattern_properties=var_100, additional_properties=var_101)
    var_103 = {var_38: var_64, var_63: var_70}
    var_104 = var_102.validate(var_103)
    var_105 = module_0.Integer()
    var_106 = {var_45: var_105}
    var_107 = module_0.String()
    var_108 = module_0.Object(pattern_properties=var_106, additional_properties=var_107)
    var_109 = 'key1'
    var_110 = 'other'
    var_111 = 123
    var_112 = 456
    var_113 = {var_109: var_111, var_110: var_112}
    var_114 = var_108.validate(var_113)
    var_115 = module_0.Integer()
    var_116 = {var_45: var_115}
    var_117 = module_0.String()
    var_118 = module_0.Object(pattern_properties=var_116, additional_properties=var_117)
    var_119 = 'key1'
    var_120 = 'other'
    var_121 = 'not an integer'
    var_122 = 456
    var_123 = {var_119: var_121, var_120: var_122}
    var_124 = var_118.validate(var_123)
    var_125 = module_0.Integer()
    var_126 = {var_45: var_125}
    var_127 = module_0.String()
    var_128 = module_0.Object(pattern_properties=var_126, additional_properties=var_127)
    var_129 = 'key1'
    var_130 = 'other'
    var_131 = 'not an integer'
    var_132 = 456
    var_133 = {var_129: var_131, var_130: var_132}
    var_134 = var_128.validate(var_133)
    var_135 = module_0.Integer()
    var_136 = {var_45: var_135}
    var_137 = module_0.String()
    var_138 = module_0.Object(pattern_properties=var_136, additional_properties=var_137)
    var_139 = 'key1'
    var_140 = 'other'
    var_141 = 'not an integer'
    var_142 = 456
    var_143 = {var_139: var_141, var_140: var_142}
    var_144 = var_138.validate(var_143)
    var_145 = module_0.Integer()
    var_146 = {var_45: var_145}
    var_147 = module_0.String()
    var_148 = module_0.Object(pattern_properties=var_146, additional_properties=var_147)
    var_149 = 'key1'
    var_150 = 'key2'
    var_151 = 'other'
    var_152 = 'not an integer'
    var_153 = 456
    var_154 = {var_149: var_152, var_150: var_152, var_151: var_153}
    var_155 = var_148.validate(var_154)
    var_156 = module_0.Integer()
    var_157 = {var_45: var_156}
    var_158 = module_0.String()
    var_159 = 'required_key'
    var_160 = [var_159]
    var_161 = module_0.Object(pattern_properties=var_157, additional_properties=var_158, required=var_160)
    var_162 = 'key1'
    var_163 = 'key2'
    var_164 = 'other'
    var_165 = 'not an integer'
    var_166 = 456
    var_167 = {var_162: var_165, var_163: var_165, var_164: var_166}
    var_168 = var_161.validate(var_167)



# Parsed testcases at query #5
#--------------------------




# Parsed testcases at query #6
#--------------------------


import builtins as module_1


def test_case_0():
    var_0 = 5
    var_1 = module_0.Field(default=var_0)
    var_2 = var_1.get_default_value()
    assert var_2 == 5
    var_3 = 10
    var_4 = lambda : var_3
    var_5 = module_0.Field(default=var_4)
    var_6 = var_5.get_default_value()
    assert var_6 == 10
    var_7 = module_0.Field()
    var_8 = var_7.get_default_value()
    var_9 = None
    var_10 = module_0.Field(default=var_9)
    var_11 = var_10.get_default_value()
    assert var_11 is None
    var_12 = 'hello'
    var_13 = module_0.Field(default=var_12)
    var_14 = var_13.get_default_value()
    assert var_14 == 'hello'
    var_15 = 1
    var_16 = 2
    var_17 = 3
    var_18 = [var_15, var_16, var_17]
    var_19 = module_0.Field(default=var_18)
    var_20 = var_19.get_default_value()
    var_21 = 'a'
    var_22 = 'b'
    var_23 = {var_21: var_15, var_22: var_16}
    var_24 = module_0.Field(default=var_23)
    var_25 = var_24.get_default_value()
    var_26 = '1.5'
    var_27 = var_24.get_default_value()
    var_28 = True
    var_29 = module_0.Field(default=var_28)
    var_30 = var_29.get_default_value()
    assert var_30 is True
    var_31 = 3.14
    var_32 = module_0.Field(default=var_31)
    var_33 = var_32.get_default_value()
    var_34 = complex(var_28, var_16)
    var_35 = module_0.Field(default=var_34)
    var_36 = var_35.get_default_value()
    var_37 = complex(var_28, var_16)
    var_38 = {var_28, var_16, var_17}
    var_39 = module_0.Field(default=var_38)
    var_40 = var_39.get_default_value()
    var_41 = (var_28, var_16, var_17)
    var_42 = module_0.Field(default=var_41)
    var_43 = var_42.get_default_value()
    var_44 = range(var_8)
    var_45 = module_0.Field(default=var_44)
    var_46 = var_45.get_default_value()
    var_47 = range(var_8)
    var_48 = b'hello'
    var_49 = module_0.Field(default=var_48)
    var_50 = var_49.get_default_value()
    assert var_50 == b'hello'
    var_51 = bytearray(var_48)
    var_52 = module_0.Field(default=var_51)
    var_53 = var_52.get_default_value()
    var_54 = bytearray(var_48)
    var_55 = memoryview(var_48)
    var_56 = module_0.Field(default=var_55)
    var_57 = var_56.get_default_value()
    var_58 = [var_28, var_16, var_17]
    var_59 = frozenset(var_58)
    var_60 = module_0.Field(default=var_59)
    var_61 = var_60.get_default_value()
    var_62 = [var_28, var_16, var_17]
    var_63 = frozenset(var_62)
    var_64 = slice(var_28, var_3, var_16)
    var_65 = module_0.Field(default=var_64)
    var_66 = var_65.get_default_value()
    var_67 = slice(var_28, var_3, var_16)
    var_68 = var_65.get_default_value()
    var_69 = var_65.get_default_value()
    assert var_69 == 'hello'
    var_70 = 'world'
    var_71 = lambda : var_70
    var_72 = module_0.Field(default=var_71)
    var_73 = var_72.get_default_value()
    assert var_73 == 'world'
    var_74 = var_72.get_default_value()
    var_75 = var_74.value
    assert var_75 == 42
    var_76 = var_72.get_default_value()
    var_77 = var_72.get_default_value()
    var_78 = var_72.get_default_value()
    assert var_78 == 'async'
    var_79 = list(var_78)
    var_80 = var_72.get_default_value()
    var_81 = var_72.get_default_value()
    assert var_81 == 'async context'
    var_82 = list(var_81)
    var_83 = var_72.get_default_value()
    assert var_83 == 'property'
    var_84 = var_72.get_default_value()
    assert var_84 == 'static'
    var_85 = var_72.get_default_value()
    assert var_85 == 'class'
    var_86 = var_72.get_default_value()
    assert var_86 == 'descriptor'
    var_87 = var_72.get_default_value()
    assert var_87 == 'descriptor'
    var_88 = module_1.object()
    var_89 = var_72.get_default_value()
    var_90 = var_72.get_default_value()
    var_91 = var_72.get_default_value()
    var_92 = [var_88]
    var_93 = var_72.get_default_value()
    var_94 = var_72.get_default_value()
    var_95 = var_72.get_default_value()
    var_96 = var_72.get_default_value()
    var_97 = []
    var_98 = module_0.Field(default=var_97)
    var_99 = var_98.get_default_value()
    var_100 = {}
    var_101 = module_0.Field(default=var_100)



# Parsed testcases at query #7
#--------------------------



def test_case_0():
    var_0 = module_0.Array()
    var_1 = None
    var_2 = var_0.serialize(var_1)
    assert var_2 is None
    var_3 = module_0.String()
    var_4 = module_0.Integer()
    var_5 = [var_3, var_4]
    var_6 = module_0.Array(var_5)
    var_7 = 'hello'
    var_8 = 123
    var_9 = [var_7, var_8]
    var_10 = var_6.serialize(var_9)
    var_11 = module_0.String()
    var_12 = module_0.Array(var_11)
    var_13 = 'world'
    var_14 = [var_7, var_13]
    var_15 = var_12.serialize(var_14)
    var_16 = module_0.Array()
    var_17 = 1
    var_18 = 2
    var_19 = 3
    var_20 = [var_17, var_18, var_19]
    var_21 = var_16.serialize(var_20)



# Parsed testcases at query #8
#--------------------------



def test_case_0():
    var_0 = 42
    var_1 = module_0.Const(var_0)
    var_2 = None
    var_3 = module_0.Const(var_2)
    var_4 = 'test'
    var_5 = 'const'
    var_6 = 'Custom error'
    var_7 = {var_5: var_6}
    var_8 = module_0.Const(var_4)
    var_9 = 1
    var_10 = True
    var_11 = module_0.Const(var_9)



# Parsed testcases at query #9
#--------------------------



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
    var_16 = module_0.Array(max_items=var_11)
    var_17 = 1
    var_18 = 2
    var_19 = 3
    var_20 = [var_17, var_18, var_19]
    var_21 = var_16.validate(var_20)
    var_22 = module_0.Array(min_items=var_21, max_items=var_21)
    var_23 = 1
    var_24 = [var_23]
    var_25 = var_22.validate(var_24)
    var_26 = module_0.Array(unique_items=var_23)
    var_27 = 1
    var_28 = [var_27, var_27]
    var_29 = var_26.validate(var_28)
    var_30 = module_0.Integer()
    var_31 = module_0.Array(var_30)
    var_32 = 'not an integer'
    var_33 = [var_32]
    var_34 = var_31.validate(var_33)
    var_35 = module_0.Integer()
    var_36 = module_0.Array(var_35)
    var_37 = 3
    var_38 = [var_32, var_21, var_37]
    var_39 = var_36.validate(var_38)
    var_40 = module_0.Integer()
    var_41 = [var_40]
    var_42 = module_0.Array(var_41, var_20)
    var_43 = 1
    var_44 = 2
    var_45 = [var_43, var_44]
    var_46 = var_42.validate(var_45)
    var_47 = module_0.Integer()
    var_48 = [var_47]
    var_49 = module_0.Array(var_48, var_43)
    var_50 = [var_43, var_21]
    var_51 = var_49.validate(var_50)
    var_52 = module_0.Integer()
    var_53 = [var_52]
    var_54 = module_0.String()
    var_55 = module_0.Array(var_53, var_54)
    var_56 = 1
    var_57 = 'not a string'
    var_58 = [var_56, var_57]
    var_59 = var_55.validate(var_58)
    var_60 = module_0.Integer()
    var_61 = [var_60]
    var_62 = module_0.String()
    var_63 = module_0.Array(var_61, var_62)
    var_64 = 'string'
    var_65 = [var_56, var_64]
    var_66 = var_63.validate(var_65)
    var_67 = module_0.Integer()
    var_68 = module_0.String()
    var_69 = module_0.Array(var_67, var_68)
    var_70 = [var_56, var_21, var_64]
    var_71 = var_69.validate(var_70)
    var_72 = module_0.Integer()
    var_73 = module_0.Array(var_72, var_59)
    var_74 = 1
    var_75 = 2
    var_76 = [var_74, var_75]
    var_77 = var_73.validate(var_76)
    var_78 = module_0.Integer()
    var_79 = module_0.Array(var_78, var_74)
    var_80 = [var_74, var_21]
    var_81 = var_79.validate(var_80)
    var_82 = module_0.Integer()
    var_83 = module_0.String()
    var_84 = module_0.Array(var_82, var_83)
    var_85 = [var_74, var_21, var_64]
    var_86 = var_84.validate(var_85)
    var_87 = module_0.Integer()
    var_88 = module_0.String()
    var_89 = module_0.Array(var_87, var_88)
    var_90 = 1
    var_91 = 2
    var_92 = 3
    var_93 = [var_90, var_91, var_92]
    var_94 = var_89.validate(var_93)
    var_95 = module_0.Integer()
    var_96 = module_0.String()
    var_97 = module_0.Array(var_95, var_96)
    var_98 = [var_90, var_94, var_64]
    var_99 = var_97.validate(var_98)
    var_100 = module_0.Integer()
    var_101 = module_0.String()
    var_102 = module_0.Array(var_100, var_101, unique_items=var_90)
    var_103 = 1
    var_104 = 2
    var_105 = 'string'
    var_106 = [var_103, var_104, var_105, var_105]
    var_107 = var_102.validate(var_106)
    var_108 = module_0.Integer()
    var_109 = module_0.String()
    var_110 = module_0.Array(var_108, var_109, unique_items=var_103)
    var_111 = 'another string'
    var_112 = [var_103, var_107, var_64, var_111]
    var_113 = var_110.validate(var_112)
    var_114 = module_0.Integer()
    var_115 = module_0.String()
    var_116 = module_0.Array(var_114, var_115, unique_items=var_103)
    var_117 = 1
    var_118 = 2
    var_119 = 'string'
    var_120 = [var_117, var_118, var_119, var_117]
    var_121 = var_116.validate(var_120)
    var_122 = module_0.Integer()
    var_123 = module_0.String()
    var_124 = module_0.Array(var_122, var_123, unique_items=var_117)
    var_125 = [var_117, var_121, var_64, var_37]
    var_126 = var_124.validate(var_125)
    var_127 = module_0.Integer()
    var_128 = module_0.String()
    var_129 = module_0.Array(var_127, var_128, unique_items=var_117)
    var_130 = 1
    var_131 = 2
    var_132 = 'string'
    var_133 = [var_130, var_131, var_132, var_132, var_130]
    var_134 = var_129.validate(var_133)
    var_135 = module_0.Integer()
    var_136 = module_0.String()
    var_137 = module_0.Array(var_135, var_136, unique_items=var_130)
    var_138 = [var_130, var_134, var_64, var_111, var_37]
    var_139 = var_137.validate(var_138)
    var_140 = module_0.Integer()
    var_141 = module_0.String()
    var_142 = module_0.Array(var_140, var_141, unique_items=var_130)
    var_143 = 1
    var_144 = 2
    var_145 = 'string'
    var_146 = 'another string'
    var_147 = [var_143, var_144, var_145, var_146, var_145]
    var_148 = var_142.validate(var_147)
    var_149 = module_0.Integer()
    var_150 = module_0.String()
    var_151 = module_0.Array(var_149, var_150, unique_items=var_143)
    var_152 = 'yet another string'
    var_153 = [var_143, var_147, var_64, var_111, var_152]
    var_154 = var_151.validate(var_153)
    var_155 = module_0.Integer()
    var_156 = module_0.String()
    var_157 = module_0.Array(var_155, var_156, unique_items=var_143)
    var_158 = 1
    var_159 = 2
    var_160 = 'string'
    var_161 = 'another string'
    var_162 = 'yet another string'
    var_163 = [var_158, var_159, var_160, var_161, var_162, var_158]
    var_164 = var_157.validate(var_163)
    var_165 = module_0.Integer()
    var_166 = module_0.String()
    var_167 = module_0.Array(var_165, var_166, unique_items=var_158)
    var_168 = [var_158, var_162, var_64, var_111, var_152, var_37]
    var_169 = var_167.validate(var_168)



# Parsed testcases at query #10
#--------------------------



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
    var_16 = module_0.Array(max_items=var_11)
    var_17 = 1
    var_18 = 2
    var_19 = 3
    var_20 = [var_17, var_18, var_19]
    var_21 = var_16.validate(var_20)
    var_22 = module_0.Array(unique_items=var_17)
    var_23 = 1
    var_24 = [var_23, var_23]
    var_25 = var_22.validate(var_24)
    var_26 = module_0.Integer()
    var_27 = module_0.Array(var_26)
    var_28 = 1
    var_29 = 'not an integer'
    var_30 = [var_28, var_29]
    var_31 = var_27.validate(var_30)
    var_32 = module_0.Integer()
    var_33 = module_0.Array(var_32)
    var_34 = [var_28, var_21]
    var_35 = var_33.validate(var_34)
    var_36 = module_0.Integer()
    var_37 = [var_36]
    var_38 = module_0.Array(var_37, var_31)
    var_39 = 1
    var_40 = 2
    var_41 = [var_39, var_40]
    var_42 = var_38.validate(var_41)
    var_43 = module_0.Integer()
    var_44 = [var_43]
    var_45 = module_0.String()
    var_46 = module_0.Array(var_44, var_45)
    var_47 = 'two'
    var_48 = [var_39, var_47]
    var_49 = var_46.validate(var_48)
    var_50 = module_0.Array(exact_items=var_21)
    var_51 = 1
    var_52 = [var_51]
    var_53 = var_50.validate(var_52)
    var_54 = module_0.Array(exact_items=var_21)
    var_55 = [var_51, var_21]
    var_56 = var_54.validate(var_55)
    var_57 = module_0.Array(exact_items=var_21)
    var_58 = 1
    var_59 = 2
    var_60 = 3
    var_61 = [var_58, var_59, var_60]
    var_62 = var_57.validate(var_61)
    var_63 = 3
    var_64 = module_0.Array(min_items=var_58, max_items=var_63)
    var_65 = [var_58, var_62]
    var_66 = var_64.validate(var_65)
    var_67 = module_0.Array(min_items=var_58, max_items=var_63)
    var_68 = []
    var_69 = var_67.validate(var_68)
    var_70 = module_0.Array(min_items=var_68, max_items=var_63)
    var_71 = 1
    var_72 = 2
    var_73 = 3
    var_74 = 4
    var_75 = [var_71, var_72, var_73, var_74]
    var_76 = var_70.validate(var_75)
    var_77 = module_0.Array(unique_items=var_71)
    var_78 = 'a'
    var_79 = 1
    var_80 = {var_78: var_79}
    var_81 = {var_78: var_79}
    var_82 = [var_80, var_81]
    var_83 = var_77.validate(var_82)
    var_84 = module_0.Array(unique_items=var_78)
    var_85 = [var_78, var_82]
    var_86 = var_84.validate(var_85)
    var_87 = module_0.Array(unique_items=var_78)
    var_88 = 1
    var_89 = [var_88, var_88]
    var_90 = var_87.validate(var_89)
    var_91 = module_0.Array(unique_items=var_88)
    var_92 = [var_88, var_82, var_63]
    var_93 = var_91.validate(var_92)
    var_94 = module_0.Array(unique_items=var_88)
    var_95 = 1
    var_96 = 2
    var_97 = [var_95, var_96, var_95]
    var_98 = var_94.validate(var_97)
    var_99 = module_0.Array(unique_items=var_95)
    var_100 = 4
    var_101 = [var_95, var_82, var_63, var_100]
    var_102 = var_99.validate(var_101)
    var_103 = module_0.Array(unique_items=var_95)
    var_104 = 1
    var_105 = 2
    var_106 = 3
    var_107 = [var_104, var_105, var_106, var_104]
    var_108 = var_103.validate(var_107)
    var_109 = module_0.Array(unique_items=var_104)
    var_110 = 5
    var_111 = [var_104, var_108, var_63, var_100, var_110]
    var_112 = var_109.validate(var_111)
    var_113 = module_0.Array(unique_items=var_104)
    var_114 = 1
    var_115 = 2
    var_116 = 3
    var_117 = 4
    var_118 = [var_114, var_115, var_116, var_117, var_114]
    var_119 = var_113.validate(var_118)
    var_120 = module_0.Array(unique_items=var_114)
    var_121 = 6
    var_122 = [var_114, var_118, var_63, var_100, var_110, var_121]
    var_123 = var_120.validate(var_122)
    var_124 = module_0.Array(unique_items=var_114)
    var_125 = 1
    var_126 = 2
    var_127 = 3
    var_128 = 4
    var_129 = 5
    var_130 = [var_125, var_126, var_127, var_128, var_129, var_125]
    var_131 = var_124.validate(var_130)
    var_132 = module_0.Array(unique_items=var_125)
    var_133 = 7
    var_134 = [var_125, var_129, var_63, var_100, var_110, var_121, var_133]
    var_135 = var_132.validate(var_134)
    var_136 = module_0.Array(unique_items=var_125)
    var_137 = 1
    var_138 = 2
    var_139 = 3
    var_140 = 4
    var_141 = 5
    var_142 = 6
    var_143 = [var_137, var_138, var_139, var_140, var_141, var_142, var_137]
    var_144 = var_136.validate(var_143)
    var_145 = module_0.Array(unique_items=var_137)
    var_146 = 8
    var_147 = [var_137, var_141, var_63, var_100, var_110, var_121, var_133, var_146]
    var_148 = var_145.validate(var_147)
    var_149 = module_0.Array(unique_items=var_137)



# Parsed testcases at query #11
#--------------------------



def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = True
    var_8 = module_0.Choice(choices=var_6)
    var_9 = None
    var_10 = var_8.validate(var_9)
    assert var_10 is None
    var_11 = (var_0, var_1)
    var_12 = (var_3, var_4)
    var_13 = [var_11, var_12]
    var_14 = False
    var_15 = module_0.Choice(choices=var_13)
    var_16 = None
    var_17 = var_15.validate(var_16)
    var_18 = (var_16, var_17)
    var_19 = (var_3, var_4)
    var_20 = [var_18, var_19]
    var_21 = module_0.Choice(choices=var_20)
    var_22 = var_21.validate(var_16)
    assert var_22 == 'a'
    var_23 = (var_16, var_17)
    var_24 = (var_3, var_4)
    var_25 = [var_23, var_24]
    var_26 = module_0.Choice(choices=var_25)
    var_27 = 'c'
    var_28 = var_26.validate(var_27)
    var_29 = (var_27, var_28)
    var_30 = (var_3, var_4)
    var_31 = [var_29, var_30]
    var_32 = module_0.Choice(choices=var_31, coerce_types=var_7)
    var_33 = ''
    var_34 = var_32.validate(var_33)
    assert var_34 is None
    var_35 = (var_27, var_28)
    var_36 = (var_3, var_4)
    var_37 = [var_35, var_36]
    var_38 = module_0.Choice(choices=var_37)
    var_39 = ''
    var_40 = var_38.validate(var_39)
    var_41 = (var_39, var_40)
    var_42 = (var_3, var_4)
    var_43 = [var_41, var_42]
    var_44 = module_0.Choice(choices=var_43, coerce_types=var_14)
    var_45 = ''
    var_46 = var_44.validate(var_45)
    var_47 = (var_45, var_46)
    var_48 = (var_3, var_4)
    var_49 = [var_47, var_48]
    var_50 = module_0.Choice(choices=var_49)
    var_51 = 123
    var_52 = var_50.validate(var_51)
    var_53 = (var_51, var_52)
    var_54 = (var_3, var_4)
    var_55 = [var_53, var_54]
    var_56 = module_0.Choice(choices=var_55)
    var_57 = 'a'
    var_58 = 'A'
    var_59 = (var_57, var_58)
    var_60 = var_56.validate(var_59)
    var_61 = (var_57, var_58)
    var_62 = (var_60, var_4)
    var_63 = [var_61, var_62]
    var_64 = module_0.Choice(choices=var_63)
    var_65 = 'a'
    var_66 = [var_65]
    var_67 = var_64.validate(var_66)
    var_68 = (var_65, var_66)
    var_69 = (var_60, var_4)
    var_70 = [var_68, var_69]
    var_71 = module_0.Choice(choices=var_70)
    var_72 = 'key'
    var_73 = 'value'
    var_74 = {var_72: var_73}
    var_75 = var_71.validate(var_74)
    var_76 = (var_72, var_73)
    var_77 = (var_75, var_4)
    var_78 = [var_76, var_77]
    var_79 = module_0.Choice(choices=var_78)
    var_80 = 'a'
    var_81 = {var_80}
    var_82 = var_79.validate(var_81)
    var_83 = (var_80, var_81)
    var_84 = (var_75, var_4)
    var_85 = [var_83, var_84]
    var_86 = module_0.Choice(choices=var_85)
    var_87 = True
    var_88 = var_86.validate(var_87)
    var_89 = (var_87, var_88)
    var_90 = (var_75, var_4)
    var_91 = [var_89, var_90]
    var_92 = module_0.Choice(choices=var_91)
    var_93 = 1.23
    var_94 = var_92.validate(var_93)
    var_95 = (var_93, var_94)
    var_96 = (var_75, var_4)
    var_97 = [var_95, var_96]
    var_98 = module_0.Choice(choices=var_97)
    var_99 = 1
    var_100 = var_99 + var_94
    var_101 = var_98.validate(var_100)
    var_102 = (var_99, var_94)
    var_103 = (var_101, var_4)
    var_104 = [var_102, var_103]
    var_105 = module_0.Choice(choices=var_104)
    var_106 = b'a'
    var_107 = var_105.validate(var_106)
    var_108 = (var_106, var_107)
    var_109 = (var_101, var_4)
    var_110 = [var_108, var_109]
    var_111 = module_0.Choice(choices=var_110)
    var_112 = b'a'
    var_113 = bytearray(var_112)
    var_114 = var_111.validate(var_113)
    var_115 = (var_112, var_113)
    var_116 = (var_101, var_4)
    var_117 = [var_115, var_116]
    var_118 = module_0.Choice(choices=var_117)
    var_119 = b'a'
    var_120 = memoryview(var_119)
    var_121 = var_118.validate(var_120)
    var_122 = (var_119, var_120)
    var_123 = (var_101, var_4)
    var_124 = [var_122, var_123]
    var_125 = module_0.Choice(choices=var_124)
    var_126 = 5
    var_127 = range(var_126)
    var_128 = var_125.validate(var_127)
    var_129 = (var_126, var_127)
    var_130 = (var_101, var_4)
    var_131 = [var_129, var_130]
    var_132 = module_0.Choice(choices=var_131)
    var_133 = 1
    var_134 = 5
    var_135 = 2
    var_136 = slice(var_133, var_134, var_135)
    var_137 = var_132.validate(var_136)
    var_138 = (var_133, var_134)
    var_139 = (var_136, var_137)
    var_140 = [var_138, var_139]
    var_141 = module_0.Choice(choices=var_140)
    var_142 = lambda x: x
    var_143 = var_141.validate(var_142)
    var_144 = (var_142, var_143)
    var_145 = (var_136, var_137)
    var_146 = [var_144, var_145]
    var_147 = module_0.Choice(choices=var_146)
    var_148 = (var_142, var_143)
    var_149 = (var_136, var_137)
    var_150 = [var_148, var_149]
    var_151 = module_0.Choice(choices=var_150)
    var_152 = 'a'
    var_153 = 'A'
    var_154 = (var_152, var_153)
    var_155 = 'b'
    var_156 = 'B'
    var_157 = (var_155, var_156)
    var_158 = [var_154, var_157]
    var_159 = module_0.Choice(choices=var_158)
    var_160 = var_151.validate(var_159)
    var_161 = (var_152, var_153)
    var_162 = (var_155, var_156)
    var_163 = [var_161, var_162]
    var_164 = module_0.Choice(choices=var_163)
    var_165 = (var_152, var_153)
    var_166 = (var_155, var_156)
    var_167 = [var_165, var_166]
    var_168 = module_0.Choice(choices=var_167)
    var_169 = 5
    var_170 = range(var_169)
    var_171 = var_168.validate(var_154)
    var_172 = (var_169, var_170)
    var_173 = (var_171, var_156)
    var_174 = [var_172, var_173]
    var_175 = module_0.Choice(choices=var_174)
    var_176 = var_175.validate(var_169)
    var_177 = (var_169, var_176)
    var_178 = (var_171, var_156)
    var_179 = [var_177, var_178]
    var_180 = module_0.Choice(choices=var_179)
    var_181 = var_180.validate(var_169)
    var_182 = (var_169, var_181)
    var_183 = (var_171, var_156)
    var_184 = [var_182, var_183]
    var_185 = module_0.Choice(choices=var_184)
    var_186 = var_185.validate(var_169)
    var_187 = (var_169, var_186)
    var_188 = (var_171, var_156)
    var_189 = [var_187, var_188]
    var_190 = module_0.Choice(choices=var_189)
    var_191 = var_190.validate(var_169)



# Parsed testcases at query #12
#--------------------------




# Parsed testcases at query #13
#--------------------------



def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = module_0.Union(var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None
    var_7 = module_0.String()
    var_8 = module_0.Integer()
    var_9 = [var_7, var_8]
    var_10 = False
    var_11 = module_0.Union(var_9)
    var_12 = None
    var_13 = var_11.validate(var_12)
    var_14 = module_0.String()
    var_15 = module_0.Integer()
    var_16 = [var_14, var_15]
    var_17 = module_0.Union(var_16)
    var_18 = 'hello'
    var_19 = var_17.validate(var_18)
    assert var_19 == 'hello'
    var_20 = 123
    var_21 = var_17.validate(var_20)
    assert var_21 == 123
    var_22 = module_0.String()
    var_23 = module_0.Integer()
    var_24 = [var_22, var_23]
    var_25 = module_0.Union(var_24)
    var_26 = True
    var_27 = var_25.validate(var_26)
    var_28 = 5
    var_29 = module_0.String(min_length=var_28)
    var_30 = 10
    var_31 = module_0.Integer(minimum=var_30)
    var_32 = [var_29, var_31]
    var_33 = module_0.Union(var_32)
    var_34 = 'hi'
    var_35 = var_33.validate(var_34)
    var_36 = module_0.String(min_length=var_28)
    var_37 = module_0.Integer(minimum=var_30)
    var_38 = [var_36, var_37]
    var_39 = module_0.Union(var_38)
    var_40 = 5
    var_41 = var_39.validate(var_40)
    var_42 = 'name'
    var_43 = module_0.String()
    var_44 = {var_42: var_43}
    var_45 = module_0.Object(properties=var_44)
    var_46 = module_0.Integer()
    var_47 = [var_45, var_46]
    var_48 = module_0.Union(var_47)
    var_49 = 'name'
    var_50 = 123
    var_51 = {var_49: var_50}
    var_52 = var_48.validate(var_51)
    var_53 = module_0.String()
    var_54 = module_0.Array(var_53)
    var_55 = module_0.Integer()
    var_56 = [var_54, var_55]
    var_57 = module_0.Union(var_56)
    var_58 = 123
    var_59 = [var_58]
    var_60 = var_57.validate(var_59)
    var_61 = module_0.String()
    var_62 = {var_42: var_61}
    var_63 = module_0.Object(properties=var_62)
    var_64 = module_0.Integer()
    var_65 = [var_63, var_64]
    var_66 = module_0.Union(var_65)
    var_67 = 'name'
    var_68 = 123
    var_69 = {var_67: var_68}
    var_70 = var_66.validate(var_69)
    var_71 = 'age'
    var_72 = module_0.String()
    var_73 = module_0.Integer()
    var_74 = {var_42: var_72, var_71: var_73}
    var_75 = module_0.Object(properties=var_74)
    var_76 = module_0.Integer()
    var_77 = [var_75, var_76]
    var_78 = module_0.Union(var_77)
    var_79 = 'name'
    var_80 = 'age'
    var_81 = 123
    var_82 = 'invalid'
    var_83 = {var_79: var_81, var_80: var_82}
    var_84 = var_78.validate(var_83)
    var_85 = 'data'
    var_86 = module_0.String()
    var_87 = module_0.Array(var_86)
    var_88 = {var_85: var_87}
    var_89 = module_0.Object(properties=var_88)
    var_90 = module_0.Integer()
    var_91 = [var_89, var_90]
    var_92 = module_0.Union(var_91)
    var_93 = 'data'
    var_94 = 123
    var_95 = [var_94]
    var_96 = {var_93: var_95}
    var_97 = var_92.validate(var_96)
    var_98 = module_0.String()
    var_99 = module_0.Array(var_98)
    var_100 = {var_85: var_99}
    var_101 = module_0.Object(properties=var_100)
    var_102 = module_0.Integer()
    var_103 = [var_101, var_102]
    var_104 = module_0.Union(var_103)
    var_105 = 'data'
    var_106 = 123
    var_107 = 456
    var_108 = [var_106, var_107]
    var_109 = {var_105: var_108}
    var_110 = var_104.validate(var_109)
    var_111 = module_0.String(min_length=var_28)
    var_112 = module_0.Array(var_111)
    var_113 = {var_85: var_112}
    var_114 = module_0.Object(properties=var_113)
    var_115 = module_0.Integer()
    var_116 = [var_114, var_115]
    var_117 = module_0.Union(var_116)
    var_118 = 'data'
    var_119 = 'hi'
    var_120 = 'hello'
    var_121 = [var_119, var_120]
    var_122 = {var_118: var_121}
    var_123 = var_117.validate(var_122)
    var_124 = module_0.String(min_length=var_28)
    var_125 = module_0.Array(var_124)
    var_126 = {var_85: var_125}
    var_127 = module_0.Object(properties=var_126)
    var_128 = module_0.Integer()
    var_129 = [var_127, var_128]
    var_130 = module_0.Union(var_129)
    var_131 = 'data'
    var_132 = 'hi'
    var_133 = 'hello'
    var_134 = [var_132, var_133]
    var_135 = {var_131: var_134}
    var_136 = var_130.validate(var_135)
    var_137 = module_0.String(min_length=var_28)
    var_138 = module_0.Array(var_137)
    var_139 = {var_85: var_138}
    var_140 = module_0.Object(properties=var_139)
    var_141 = module_0.Integer()
    var_142 = [var_140, var_141]
    var_143 = module_0.Union(var_142)
    var_144 = 'data'
    var_145 = 'hi'
    var_146 = 'hello'
    var_147 = [var_145, var_146]
    var_148 = {var_144: var_147}
    var_149 = var_143.validate(var_148)
    var_150 = module_0.String(min_length=var_28)
    var_151 = module_0.Array(var_150)
    var_152 = {var_85: var_151}
    var_153 = module_0.Object(properties=var_152)
    var_154 = module_0.Integer()
    var_155 = [var_153, var_154]
    var_156 = module_0.Union(var_155)
    var_157 = 'data'
    var_158 = 'hi'
    var_159 = 'hello'
    var_160 = [var_158, var_159]
    var_161 = {var_157: var_160}
    var_162 = var_156.validate(var_161)
    var_163 = module_0.String(min_length=var_28)
    var_164 = module_0.Array(var_163)
    var_165 = {var_85: var_164}
    var_166 = module_0.Object(properties=var_165)
    var_167 = module_0.Integer()
    var_168 = [var_166, var_167]
    var_169 = module_0.Union(var_168)
    var_170 = 'data'
    var_171 = 'hi'
    var_172 = 'hello'
    var_173 = [var_171, var_172]
    var_174 = {var_170: var_173}
    var_175 = var_169.validate(var_174)
    var_176 = module_0.String(min_length=var_28)
    var_177 = module_0.Array(var_176)
    var_178 = {var_85: var_177}
    var_179 = module_0.Object(properties=var_178)
    var_180 = module_0.Integer()
    var_181 = [var_179, var_180]
    var_182 = module_0.Union(var_181)
    var_183 = 'data'
    var_184 = 'hi'
    var_185 = 'hello'
    var_186 = [var_184, var_185]
    var_187 = {var_183: var_186}
    var_188 = var_182.validate(var_187)
    var_189 = module_0.String(min_length=var_28)
    var_190 = module_0.Array(var_189)
    var_191 = {var_85: var_190}
    var_192 = module_0.Object(properties=var_191)
    var_193 = module_0.Integer()
    var_194 = [var_192, var_193]
    var_195 = module_0.Union(var_194)
    var_196 = 'data'
    var_197 = 'hi'
    var_198 = 'hello'
    var_199 = [var_197, var_198]
    var_200 = {var_196: var_199}
    var_201 = var_195.validate(var_200)
    var_202 = module_0.String(min_length=var_28)
    var_203 = module_0.Array(var_202)
    var_204 = {var_85: var_203}
    var_205 = module_0.Object(properties=var_204)
    var_206 = module_0.Integer()
    var_207 = [var_205, var_206]
    var_208 = module_0.Union(var_207)
    var_209 = 'data'
    var_210 = 'hi'
    var_211 = 'hello'
    var_212 = [var_210, var_211]
    var_213 = {var_209: var_212}
    var_214 = var_208.validate(var_213)
    var_215 = module_0.String(min_length=var_28)
    var_216 = module_0.Array(var_215)
    var_217 = {var_85: var_216}
    var_218 = module_0.Object(properties=var_217)
    var_219 = module_0.Integer()
    var_220 = [var_218, var_219]
    var_221 = module_0.Union(var_220)



