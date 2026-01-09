####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.fields as module_0
import typesystem.forms as module_1


def test_case_0():
    var_0 = module_0.Field()
    var_1 = None
    var_2 = module_1.Form(env=var_1, schema=var_1)
    var_3 = var_2.input_type_for_field(var_0)
    assert var_3 == 'text'
    var_4 = 'unknown'
    var_5 = module_0.Field()
    var_6 = var_2.input_type_for_field(var_5)
    assert var_6 == 'text'
    var_7 = 'email'
    var_8 = module_0.Field()
    var_9 = var_2.input_type_for_field(var_8)
    assert var_9 == 'email'
    var_10 = 'color'
    var_11 = module_0.Field()
    var_12 = var_2.input_type_for_field(var_11)
    assert var_12 == 'color'
    var_13 = 'datetime'
    var_14 = module_0.Field()
    var_15 = var_2.input_type_for_field(var_14)
    assert var_15 == 'datetime-local'
    var_16 = 'date'
    var_17 = module_0.Field()
    var_18 = var_2.input_type_for_field(var_17)
    assert var_18 == 'date'
    var_19 = 'month'
    var_20 = module_0.Field()
    var_21 = var_2.input_type_for_field(var_20)
    assert var_21 == 'month'
    var_22 = 'number'
    var_23 = module_0.Field()
    var_24 = var_2.input_type_for_field(var_23)
    assert var_24 == 'number'
    var_25 = 'password'
    var_26 = module_0.Field()
    var_27 = var_2.input_type_for_field(var_26)
    assert var_27 == 'password'
    var_28 = 'range'
    var_29 = module_0.Field()
    var_30 = var_2.input_type_for_field(var_29)
    assert var_30 == 'range'
    var_31 = 'search'
    var_32 = module_0.Field()
    var_33 = var_2.input_type_for_field(var_32)
    assert var_33 == 'search'
    var_34 = 'tel'
    var_35 = module_0.Field()
    var_36 = var_2.input_type_for_field(var_35)
    assert var_36 == 'tel'
    var_37 = 'time'
    var_38 = module_0.Field()
    var_39 = var_2.input_type_for_field(var_38)
    assert var_39 == 'time'
    var_40 = 'url'
    var_41 = module_0.Field()
    var_42 = var_2.input_type_for_field(var_41)
    assert var_42 == 'url'
    var_43 = 'week'
    var_44 = module_0.Field()
    var_45 = var_2.input_type_for_field(var_44)
    assert var_45 == 'week'
    var_46 = 'hidden'
    var_47 = module_0.Field()
    var_48 = var_2.input_type_for_field(var_47)
    assert var_48 == 'hidden'
    var_49 = 'text'
    var_50 = module_0.Field()
    var_51 = var_2.input_type_for_field(var_50)
    assert var_51 == 'text'
    var_52 = module_0.Field()
    var_53 = var_2.input_type_for_field(var_52)
    assert var_53 == 'color'
    var_54 = module_0.Field()
    var_55 = var_2.input_type_for_field(var_54)
    assert var_55 == 'datetime-local'
    var_56 = module_0.Field()
    var_57 = var_2.input_type_for_field(var_56)
    assert var_57 == 'date'
    var_58 = module_0.Field()
    var_59 = var_2.input_type_for_field(var_58)
    assert var_59 == 'month'
    var_60 = module_0.Field()
    var_61 = var_2.input_type_for_field(var_60)
    assert var_61 == 'number'
    var_62 = module_0.Field()
    var_63 = var_2.input_type_for_field(var_62)
    assert var_63 == 'password'
    var_64 = module_0.Field()
    var_65 = var_2.input_type_for_field(var_64)
    assert var_65 == 'range'
    var_66 = module_0.Field()
    var_67 = var_2.input_type_for_field(var_66)
    assert var_67 == 'search'
    var_68 = module_0.Field()
    var_69 = var_2.input_type_for_field(var_68)
    assert var_69 == 'tel'
    var_70 = module_0.Field()
    var_71 = var_2.input_type_for_field(var_70)
    assert var_71 == 'time'
    var_72 = module_0.Field()
    var_73 = var_2.input_type_for_field(var_72)
    assert var_73 == 'url'
    var_74 = module_0.Field()
    var_75 = var_2.input_type_for_field(var_74)
    assert var_75 == 'week'
    var_76 = module_0.Field()
    var_77 = var_2.input_type_for_field(var_76)
    assert var_77 == 'hidden'
    var_78 = module_0.Field()
    var_79 = var_2.input_type_for_field(var_78)
    assert var_79 == 'text'
    var_80 = module_0.Field()
    var_81 = var_2.input_type_for_field(var_80)
    assert var_81 == 'color'
    var_82 = module_0.Field()
    var_83 = var_2.input_type_for_field(var_82)
    assert var_83 == 'datetime-local'
    var_84 = module_0.Field()
    var_85 = var_2.input_type_for_field(var_84)
    assert var_85 == 'date'
    var_86 = module_0.Field()
    var_87 = var_2.input_type_for_field(var_86)
    assert var_87 == 'month'
    var_88 = module_0.Field()
    var_89 = var_2.input_type_for_field(var_88)
    assert var_89 == 'number'
    var_90 = module_0.Field()
    var_91 = var_2.input_type_for_field(var_90)
    assert var_91 == 'password'
    var_92 = module_0.Field()
    var_93 = var_2.input_type_for_field(var_92)
    assert var_93 == 'range'
    var_94 = module_0.Field()
    var_95 = var_2.input_type_for_field(var_94)
    assert var_95 == 'search'
    var_96 = module_0.Field()
    var_97 = var_2.input_type_for_field(var_96)
    assert var_97 == 'tel'
    var_98 = module_0.Field()
    var_99 = var_2.input_type_for_field(var_98)
    assert var_99 == 'time'
    var_100 = module_0.Field()
    var_101 = var_2.input_type_for_field(var_100)
    assert var_101 == 'url'
    var_102 = module_0.Field()
    var_103 = var_2.input_type_for_field(var_102)
    assert var_103 == 'week'
    var_104 = module_0.Field()
    var_105 = var_2.input_type_for_field(var_104)
    assert var_105 == 'hidden'
    var_106 = module_0.Field()
    var_107 = var_2.input_type_for_field(var_106)
    assert var_107 == 'text'
    var_108 = module_0.Field()
    var_109 = var_2.input_type_for_field(var_108)
    assert var_109 == 'color'
    var_110 = module_0.Field()



# Parsed testcases at query #2
#--------------------------


import typesystem.forms as module_2
import typesystem.schemas as module_1


def test_case_0():
    var_0 = 'name'
    var_1 = 10
    var_2 = module_0.String(max_length=var_1)
    var_3 = {var_0: var_2}
    var_4 = module_1.Schema(var_3)
    var_5 = None
    var_6 = module_2.Form(env=var_5, schema=var_4)
    var_7 = 'John'
    var_8 = {var_0: var_7}
    var_9 = var_6.validate(var_8)
    var_10 = module_0.String(max_length=var_1)
    var_11 = {var_0: var_10}
    var_12 = module_1.Schema(var_11)
    var_13 = module_2.Form(env=var_5, schema=var_12)
    var_14 = 'John Doe'
    var_15 = {var_0: var_14}
    var_16 = var_13.validate(var_15)
    var_17 = module_0.String(max_length=var_1)
    var_18 = {var_0: var_17}
    var_19 = module_1.Schema(var_18)
    var_20 = module_2.Form(env=var_5, schema=var_19)
    var_21 = {}
    var_22 = var_20.validate(var_21)
    var_23 = module_0.String(max_length=var_1)
    var_24 = {var_0: var_23}
    var_25 = module_1.Schema(var_24)
    var_26 = module_2.Form(env=var_5, schema=var_25)
    var_27 = 'age'
    var_28 = 25
    var_29 = {var_0: var_7, var_27: var_28}
    var_30 = var_26.validate(var_29)
    var_31 = 'person'
    var_32 = module_0.String(max_length=var_1)
    var_33 = {var_0: var_32}
    var_34 = module_0.Object()
    var_35 = {var_31: var_34}
    var_36 = module_1.Schema(var_35)
    var_37 = module_2.Form(env=var_5, schema=var_36)
    var_38 = {var_0: var_7}
    var_39 = {var_31: var_38}
    var_40 = var_37.validate(var_39)
    var_41 = module_0.String(max_length=var_1)
    var_42 = {var_0: var_41}
    var_43 = module_0.Object()
    var_44 = {var_31: var_43}
    var_45 = module_1.Schema(var_44)
    var_46 = module_2.Form(env=var_5, schema=var_45)
    var_47 = {var_0: var_14}
    var_48 = {var_31: var_47}
    var_49 = var_46.validate(var_48)
    var_50 = True
    var_51 = module_0.String(max_length=var_1)
    var_52 = {var_0: var_51}
    var_53 = module_1.Schema(var_52)
    var_54 = module_2.Form(env=var_5, schema=var_53)
    var_55 = {var_0: var_7}
    var_56 = var_54.validate(var_55)
    var_57 = module_0.String(max_length=var_1)
    var_58 = {var_0: var_57}
    var_59 = module_1.Schema(var_58)
    var_60 = module_2.Form(env=var_5, schema=var_59)
    var_61 = {}
    var_62 = var_60.validate(var_61)
    var_63 = module_0.String(max_length=var_1)
    var_64 = {var_0: var_63}
    var_65 = module_1.Schema(var_64)
    var_66 = module_2.Form(env=var_5, schema=var_65)
    var_67 = {var_0: var_5}
    var_68 = var_66.validate(var_67)
    var_69 = module_0.String(allow_blank=var_50, max_length=var_1)
    var_70 = {var_0: var_69}
    var_71 = module_1.Schema(var_70)
    var_72 = module_2.Form(env=var_5, schema=var_71)
    var_73 = ''
    var_74 = {var_0: var_73}
    var_75 = var_72.validate(var_74)
    var_76 = 'color'
    var_77 = 'red'
    var_78 = 'green'
    var_79 = 'blue'
    var_80 = [var_77, var_78, var_79]
    var_81 = module_0.Choice(choices=var_80)
    var_82 = {var_76: var_81}
    var_83 = module_1.Schema(var_82)
    var_84 = module_2.Form(env=var_5, schema=var_83)
    var_85 = {var_76: var_77}
    var_86 = var_84.validate(var_85)
    var_87 = [var_77, var_78, var_79]
    var_88 = module_0.Choice(choices=var_87)
    var_89 = {var_76: var_88}
    var_90 = module_1.Schema(var_89)
    var_91 = module_2.Form(env=var_5, schema=var_90)
    var_92 = 'yellow'
    var_93 = {var_76: var_92}
    var_94 = var_91.validate(var_93)
    var_95 = 'active'
    var_96 = module_0.Boolean()
    var_97 = {var_95: var_96}
    var_98 = module_1.Schema(var_97)
    var_99 = module_2.Form(env=var_5, schema=var_98)
    var_100 = {var_95: var_50}
    var_101 = var_99.validate(var_100)
    var_102 = module_0.Boolean()
    var_103 = {var_95: var_102}
    var_104 = module_1.Schema(var_103)
    var_105 = module_2.Form(env=var_5, schema=var_104)
    var_106 = 'yes'
    var_107 = {var_95: var_106}
    var_108 = var_105.validate(var_107)
    var_109 = module_0.String(max_length=var_1)
    var_110 = 2
    var_111 = module_0.String(max_length=var_110)
    var_112 = {var_0: var_109, var_27: var_111}
    var_113 = module_1.Schema(var_112)
    var_114 = module_2.Form(env=var_5, schema=var_113)
    var_115 = '25'
    var_116 = {var_0: var_14, var_27: var_115}
    var_117 = var_114.validate(var_116)
    var_118 = module_0.String(max_length=var_1)
    var_119 = module_0.String(max_length=var_110)
    var_120 = {var_0: var_118, var_27: var_119}
    var_121 = module_1.Schema(var_120)
    var_122 = module_2.Form(env=var_5, schema=var_121)
    var_123 = {var_0: var_7, var_27: var_115}
    var_124 = var_122.validate(var_123)
    var_125 = module_0.String(max_length=var_1)
    var_126 = module_0.String(max_length=var_110)
    var_127 = {var_0: var_125, var_27: var_126}
    var_128 = module_0.Object()
    var_129 = {var_31: var_128}
    var_130 = module_1.Schema(var_129)
    var_131 = module_2.Form(env=var_5, schema=var_130)
    var_132 = {var_0: var_14, var_27: var_115}
    var_133 = {var_31: var_132}
    var_134 = var_131.validate(var_133)
    var_135 = module_0.String(max_length=var_1)
    var_136 = module_0.String(max_length=var_110)
    var_137 = {var_0: var_135, var_27: var_136}
    var_138 = module_0.Object()
    var_139 = {var_31: var_138}
    var_140 = module_1.Schema(var_139)
    var_141 = module_2.Form(env=var_5, schema=var_140)
    var_142 = {var_0: var_7, var_27: var_115}
    var_143 = {var_31: var_142}
    var_144 = var_141.validate(var_143)
    var_145 = module_0.String(max_length=var_1)
    var_146 = module_0.String(max_length=var_110)
    var_147 = {var_0: var_145, var_27: var_146}
    var_148 = module_0.Object()
    var_149 = {var_31: var_148}
    var_150 = module_1.Schema(var_149)
    var_151 = module_2.Form(env=var_5, schema=var_150)
    var_152 = '250'
    var_153 = {var_0: var_7, var_27: var_152}
    var_154 = {var_31: var_153}
    var_155 = var_151.validate(var_154)
    var_156 = module_0.String(max_length=var_1)
    var_157 = module_0.String(max_length=var_110)
    var_158 = {var_0: var_156, var_27: var_157}
    var_159 = module_0.Object()
    var_160 = {var_31: var_159}
    var_161 = module_1.Schema(var_160)
    var_162 = module_2.Form(env=var_5, schema=var_161)
    var_163 = {var_0: var_7}
    var_164 = {var_31: var_163}
    var_165 = var_162.validate(var_164)
    var_166 = module_0.String(max_length=var_1)
    var_167 = module_0.String(max_length=var_110)
    var_168 = {var_0: var_166, var_27: var_167}
    var_169 = module_0.Object()
    var_170 = {var_31: var_169}
    var_171 = module_1.Schema(var_170)
    var_172 = module_2.Form(env=var_5, schema=var_171)
    var_173 = 'city'
    var_174 = 'New York'
    var_175 = {var_0: var_7, var_27: var_115, var_173: var_174}
    var_176 = {var_31: var_175}
    var_177 = var_172.validate(var_176)
    var_178 = module_0.String(max_length=var_1)
    var_179 = module_0.String(max_length=var_110)
    var_180 = {var_0: var_178, var_27: var_179}
    var_181 = module_0.Object()
    var_182 = {var_31: var_181}
    var_183 = module_1.Schema(var_182)
    var_184 = module_2.Form(env=var_5, schema=var_183)
    var_185 = {var_0: var_5, var_27: var_115}
    var_186 = {var_31: var_185}
    var_187 = var_184.validate(var_186)
    var_188 = module_0.String(allow_blank=var_50, max_length=var_1)
    var_189 = module_0.String(max_length=var_110)
    var_190 = {var_0: var_188, var_27: var_189}
    var_191 = module_0.Object()
    var_192 = {var_31: var_191}
    var_193 = module_1.Schema(var_192)
    var_194 = module_2.Form(env=var_5, schema=var_193)
    var_195 = {var_0: var_73, var_27: var_115}
    var_196 = {var_31: var_195}
    var_197 = var_194.validate(var_196)



# Parsed testcases at query #3
#--------------------------


import jinja2.environment as module_0
import typesystem.fields as module_3


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = module_1.Schema()
    var_2 = module_2.Form(env=var_0, schema=var_1)
    var_3 = module_3.Field()
    var_4 = var_2.template_for_field(var_3)
    assert var_4 == 'forms/input.html'
    var_5 = module_3.Choice()
    var_6 = var_2.template_for_field(var_5)
    assert var_6 == 'forms/select.html'
    var_7 = module_3.Boolean()
    var_8 = var_2.template_for_field(var_7)
    assert var_8 == 'forms/checkbox.html'
    var_9 = 'text'
    var_10 = module_3.String(format=var_9)
    var_11 = var_2.template_for_field(var_10)
    assert var_11 == 'forms/textarea.html'
    var_12 = 'email'
    var_13 = module_3.String(format=var_12)
    var_14 = var_2.template_for_field(var_13)
    assert var_14 == 'forms/input.html'
    var_15 = module_3.Object()
    var_16 = var_2.template_for_field(var_15)



# Parsed testcases at query #4
#--------------------------



def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'name'
    var_2 = 'John'
    var_3 = {var_1: var_2}
    var_4 = 'John Doe John Doe'
    var_5 = {var_1: var_4}



# Parsed testcases at query #5
#--------------------------


import jinja2.environment as module_1
import jinja2.loaders as module_0


def test_case_0():
    var_0 = {}
    var_1 = module_0.DictLoader(var_0)
    var_2 = module_1.Environment(loader=var_1)



# Parsed testcases at query #6
#--------------------------


import typesystem.fields as module_0


def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = module_1.Environment()
    var_3 = module_0.String()
    var_4 = module_1.Environment()
    var_5 = module_0.Integer()
    var_6 = module_1.Environment()
    var_7 = module_0.Integer()
    var_8 = module_1.Environment()
    var_9 = module_0.Integer()
    var_10 = module_1.Environment()
    var_11 = module_0.Integer()
    var_12 = module_1.Environment()
    var_13 = module_0.Integer()
    var_14 = module_1.Environment()
    var_15 = module_0.Integer()
    var_16 = module_1.Environment()
    var_17 = module_0.Boolean()
    var_18 = module_0.Integer()
    var_19 = module_1.Environment()
    var_20 = module_0.Integer()
    var_21 = module_1.Environment()



# Parsed testcases at query #7
#--------------------------


import typesystem.schemas as module_1


def test_case_0():
    var_0 = 'name'
    var_1 = module_0.String()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    var_4 = None
    var_5 = module_2.Form(env=var_4, schema=var_3, values=var_4)
    var_6 = var_5.validate()
    var_7 = var_5.render_fields()
    var_8 = module_0.String()
    var_9 = {var_0: var_8}
    var_10 = module_1.Schema(var_9)
    var_11 = module_2.Form(env=var_4, schema=var_10, values=var_4)
    var_12 = ''
    var_13 = {var_0: var_12}
    var_14 = var_11.validate(var_13)
    var_15 = var_11.render_fields()
    var_16 = module_0.String()
    var_17 = {var_0: var_16}
    var_18 = module_1.Schema(var_17)
    var_19 = 'John'
    var_20 = {var_0: var_19}
    var_21 = module_2.Form(env=var_4, schema=var_18, values=var_20)
    var_22 = var_21.validate()
    var_23 = var_21.render_fields()
    var_24 = module_0.String()
    var_25 = {var_0: var_24}
    var_26 = module_1.Schema(var_25)
    var_27 = {var_0: var_19}
    var_28 = module_2.Form(env=var_4, schema=var_26, values=var_27)
    var_29 = {var_0: var_12}
    var_30 = var_28.validate(var_29)
    var_31 = var_28.render_fields()
    var_32 = True
    var_33 = module_0.String()
    var_34 = {var_0: var_33}
    var_35 = module_1.Schema(var_34)
    var_36 = module_2.Form(env=var_4, schema=var_35, values=var_4)
    var_37 = var_36.validate()
    var_38 = var_36.render_fields()
    var_39 = module_0.String()
    var_40 = {var_0: var_39}
    var_41 = module_1.Schema(var_40)
    var_42 = module_2.Form(env=var_4, schema=var_41, values=var_4)
    var_43 = var_42.validate()
    var_44 = var_42.render_fields()
    var_45 = module_0.String()
    var_46 = {var_0: var_45}
    var_47 = module_1.Schema(var_46)
    var_48 = module_2.Form(env=var_4, schema=var_47, values=var_4)
    var_49 = var_48.validate()
    var_50 = var_48.render_fields()
    var_51 = module_0.String(allow_blank=var_32)
    var_52 = {var_0: var_51}
    var_53 = module_1.Schema(var_52)
    var_54 = module_2.Form(env=var_4, schema=var_53, values=var_4)
    var_55 = var_54.validate()
    var_56 = var_54.render_fields()
    var_57 = module_0.String()
    var_58 = {var_0: var_57}
    var_59 = module_1.Schema(var_58)
    var_60 = module_2.Form(env=var_4, schema=var_59, values=var_4)
    var_61 = var_60.validate()
    var_62 = var_60.render_fields()
    var_63 = module_0.String()
    var_64 = {var_0: var_63}
    var_65 = module_1.Schema(var_64)
    var_66 = module_2.Form(env=var_4, schema=var_65, values=var_4)
    var_67 = var_66.validate()
    var_68 = var_66.render_fields()
    var_69 = 'Full Name'
    var_70 = module_0.String()
    var_71 = {var_0: var_70}
    var_72 = module_1.Schema(var_71)
    var_73 = module_2.Form(env=var_4, schema=var_72, values=var_4)
    var_74 = var_73.validate()
    var_75 = var_73.render_fields()
    var_76 = module_0.String()
    var_77 = {var_0: var_76}
    var_78 = module_1.Schema(var_77)
    var_79 = module_2.Form(env=var_4, schema=var_78, values=var_4)
    var_80 = var_79.validate()
    var_81 = var_79.render_fields()
    var_82 = 'email'
    var_83 = module_0.String(format=var_82)
    var_84 = {var_82: var_83}
    var_85 = module_1.Schema(var_84)
    var_86 = module_2.Form(env=var_4, schema=var_85, values=var_4)
    var_87 = var_86.validate()
    var_88 = var_86.render_fields()
    var_89 = module_0.String()
    var_90 = {var_0: var_89}
    var_91 = module_1.Schema(var_90)
    var_92 = module_2.Form(env=var_4, schema=var_91, values=var_4)
    var_93 = var_92.validate()
    var_94 = var_92.render_fields()
    var_95 = 'unknown'
    var_96 = module_0.String(format=var_95)
    var_97 = {var_0: var_96}
    var_98 = module_1.Schema(var_97)
    var_99 = module_2.Form(env=var_4, schema=var_98, values=var_4)
    var_100 = var_99.validate()
    var_101 = var_99.render_fields()
    var_102 = 'color'
    var_103 = module_0.String(format=var_102)
    var_104 = {var_102: var_103}
    var_105 = module_1.Schema(var_104)
    var_106 = module_2.Form(env=var_4, schema=var_105, values=var_4)
    var_107 = var_106.validate()
    var_108 = var_106.render_fields()
    var_109 = 'datetime'
    var_110 = module_0.String(format=var_109)
    var_111 = {var_109: var_110}
    var_112 = module_1.Schema(var_111)
    var_113 = module_2.Form(env=var_4, schema=var_112, values=var_4)
    var_114 = var_113.validate()
    var_115 = var_113.render_fields()
    var_116 = 'date'
    var_117 = module_0.String(format=var_116)
    var_118 = {var_116: var_117}
    var_119 = module_1.Schema(var_118)
    var_120 = module_2.Form(env=var_4, schema=var_119, values=var_4)
    var_121 = var_120.validate()
    var_122 = var_120.render_fields()
    var_123 = module_0.String(format=var_82)
    var_124 = {var_82: var_123}
    var_125 = module_1.Schema(var_124)
    var_126 = module_2.Form(env=var_4, schema=var_125, values=var_4)
    var_127 = var_126.validate()
    var_128 = var_126.render_fields()
    var_129 = 'hidden'
    var_130 = module_0.String(format=var_129)
    var_131 = {var_129: var_130}
    var_132 = module_1.Schema(var_131)
    var_133 = module_2.Form(env=var_4, schema=var_132, values=var_4)
    var_134 = var_133.validate()
    var_135 = var_133.render_fields()
    var_136 = 'month'
    var_137 = module_0.String(format=var_136)
    var_138 = {var_136: var_137}
    var_139 = module_1.Schema(var_138)
    var_140 = module_2.Form(env=var_4, schema=var_139, values=var_4)
    var_141 = var_140.validate()
    var_142 = var_140.render_fields()
    var_143 = 'number'
    var_144 = module_0.String(format=var_143)
    var_145 = {var_143: var_144}
    var_146 = module_1.Schema(var_145)
    var_147 = module_2.Form(env=var_4, schema=var_146, values=var_4)
    var_148 = var_147.validate()
    var_149 = var_147.render_fields()
    var_150 = 'password'
    var_151 = module_0.String(format=var_150)
    var_152 = {var_150: var_151}
    var_153 = module_1.Schema(var_152)
    var_154 = module_2.Form(env=var_4, schema=var_153, values=var_4)
    var_155 = var_154.validate()
    var_156 = var_154.render_fields()
    var_157 = 'range'
    var_158 = module_0.String(format=var_157)
    var_159 = {var_157: var_158}
    var_160 = module_1.Schema(var_159)
    var_161 = module_2.Form(env=var_4, schema=var_160, values=var_4)
    var_162 = var_161.validate()
    var_163 = var_161.render_fields()
    var_164 = 'search'
    var_165 = module_0.String(format=var_164)
    var_166 = {var_164: var_165}
    var_167 = module_1.Schema(var_166)
    var_168 = module_2.Form(env=var_4, schema=var_167, values=var_4)
    var_169 = var_168.validate()
    var_170 = var_168.render_fields()
    var_171 = 'tel'
    var_172 = module_0.String(format=var_171)
    var_173 = {var_171: var_172}
    var_174 = module_1.Schema(var_173)
    var_175 = module_2.Form(env=var_4, schema=var_174, values=var_4)
    var_176 = var_175.validate()
    var_177 = var_175.render_fields()
    var_178 = 'text'
    var_179 = module_0.String(format=var_178)
    var_180 = {var_178: var_179}
    var_181 = module_1.Schema(var_180)
    var_182 = module_2.Form(env=var_4, schema=var_181, values=var_4)
    var_183 = var_182.validate()
    var_184 = var_182.render_fields()



# Parsed testcases at query #8
#--------------------------


import jinja2.environment as module_1
import jinja2.loaders as module_0


def test_case_0():
    var_0 = {}
    var_1 = module_0.DictLoader(var_0)
    var_2 = module_1.Environment(loader=var_1)
    var_3 = 'name'
    var_4 = 'email'
    var_5 = 'age'
    var_6 = 'password'
    var_7 = 'bio'
    var_8 = 'agree'
    var_9 = 'color'



# Parsed testcases at query #9
#--------------------------


import typesystem.fields as module_2
import typesystem.forms as module_4
import typesystem.schemas as module_3


def test_case_0():
    var_0 = {}
    var_1 = module_0.DictLoader(var_0)
    var_2 = module_1.Environment(loader=var_1)
    var_3 = 'name'
    var_4 = 'Name'
    var_5 = module_2.String()
    var_6 = {var_3: var_5}
    var_7 = module_3.Schema(var_6)
    var_8 = module_4.Form(env=var_2, schema=var_7)
    var_9 = module_2.String()
    var_10 = var_8.render_field(field_name=var_3, field=var_9)
    assert var_10 == ''
    var_11 = {}
    var_12 = module_0.DictLoader(var_11)
    var_13 = module_1.Environment(loader=var_12)
    var_14 = module_2.String()
    var_15 = {var_3: var_14}
    var_16 = module_3.Schema(var_15)
    var_17 = module_4.Form(env=var_13, schema=var_16)
    var_18 = ''
    var_19 = {var_3: var_18}
    var_20 = var_17.validate(var_19)
    var_21 = module_2.String()
    var_22 = 'This field is required.'
    var_23 = var_17.render_field(field_name=var_3, field=var_21, value=var_18, error=var_22)
    assert var_23 == ''
    var_24 = {}
    var_25 = module_0.DictLoader(var_24)
    var_26 = module_1.Environment(loader=var_25)
    var_27 = 'password'
    var_28 = 'Password'
    var_29 = module_2.String(format=var_27)
    var_30 = {var_27: var_29}
    var_31 = module_3.Schema(var_30)
    var_32 = module_4.Form(env=var_26, schema=var_31)
    var_33 = module_2.String(format=var_27)
    var_34 = 'secret'
    var_35 = var_32.render_field(field_name=var_27, field=var_33, value=var_34)
    assert var_35 == ''
    var_36 = {}
    var_37 = module_0.DictLoader(var_36)
    var_38 = module_1.Environment(loader=var_37)
    var_39 = 'description'
    var_40 = 'Description'
    var_41 = 'text'
    var_42 = module_2.String(format=var_41)
    var_43 = {var_39: var_42}
    var_44 = module_3.Schema(var_43)
    var_45 = module_4.Form(env=var_38, schema=var_44)
    var_46 = module_2.String(format=var_41)
    var_47 = 'Some description'
    var_48 = var_45.render_field(field_name=var_39, field=var_46, value=var_47)
    assert var_48 == ''
    var_49 = {}
    var_50 = module_0.DictLoader(var_49)
    var_51 = module_1.Environment(loader=var_50)
    var_52 = 'choice'
    var_53 = 'Choice'
    var_54 = 'a'
    var_55 = 'A'
    var_56 = (var_54, var_55)
    var_57 = 'b'
    var_58 = 'B'
    var_59 = (var_57, var_58)
    var_60 = [var_56, var_59]
    var_61 = module_2.Choice(choices=var_60)
    var_62 = {var_52: var_61}
    var_63 = module_3.Schema(var_62)
    var_64 = module_4.Form(env=var_51, schema=var_63)
    var_65 = (var_54, var_55)
    var_66 = (var_57, var_58)
    var_67 = [var_65, var_66]
    var_68 = module_2.Choice(choices=var_67)
    var_69 = var_64.render_field(field_name=var_52, field=var_68, value=var_54)
    assert var_69 == ''
    var_70 = {}
    var_71 = module_0.DictLoader(var_70)
    var_72 = module_1.Environment(loader=var_71)
    var_73 = 'agree'
    var_74 = 'Agree'
    var_75 = module_2.Boolean()
    var_76 = {var_73: var_75}
    var_77 = module_3.Schema(var_76)
    var_78 = module_4.Form(env=var_72, schema=var_77)
    var_79 = module_2.Boolean()
    var_80 = True
    var_81 = var_78.render_field(field_name=var_73, field=var_79, value=var_80)
    assert var_81 == ''
    var_82 = {}
    var_83 = module_0.DictLoader(var_82)
    var_84 = module_1.Environment(loader=var_83)
    var_85 = False
    var_86 = module_2.String()
    var_87 = {var_3: var_86}
    var_88 = module_3.Schema(var_87)
    var_89 = module_4.Form(env=var_84, schema=var_88)
    var_90 = module_2.String()
    var_91 = var_89.render_field(field_name=var_3, field=var_90)
    assert var_91 == ''
    var_92 = {}
    var_93 = module_0.DictLoader(var_92)
    var_94 = module_1.Environment(loader=var_93)
    var_95 = module_2.String()
    var_96 = {var_3: var_95}
    var_97 = module_3.Schema(var_96)
    var_98 = module_4.Form(env=var_94, schema=var_97)
    var_99 = module_2.String()
    var_100 = var_98.render_field(field_name=var_3, field=var_99)
    assert var_100 == ''
    var_101 = {}
    var_102 = module_0.DictLoader(var_101)
    var_103 = module_1.Environment(loader=var_102)
    var_104 = module_2.String(allow_blank=var_80)
    var_105 = {var_3: var_104}
    var_106 = module_3.Schema(var_105)
    var_107 = module_4.Form(env=var_103, schema=var_106)
    var_108 = module_2.String(allow_blank=var_80)
    var_109 = var_107.render_field(field_name=var_3, field=var_108)
    assert var_109 == ''
    var_110 = {}
    var_111 = module_0.DictLoader(var_110)
    var_112 = module_1.Environment(loader=var_111)
    var_113 = 'John'
    var_114 = module_2.String()
    var_115 = {var_3: var_114}
    var_116 = module_3.Schema(var_115)
    var_117 = module_4.Form(env=var_112, schema=var_116)
    var_118 = module_2.String()
    var_119 = var_117.render_field(field_name=var_3, field=var_118)
    assert var_119 == ''



# Parsed testcases at query #10
#--------------------------


import jinja2.environment as module_0
import typesystem.fields as module_3
import typesystem.forms as module_2
import typesystem.schemas as module_1


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = module_1.Schema()
    var_2 = module_2.Form(env=var_0, schema=var_1)
    var_3 = module_3.Field()
    var_4 = var_2.template_for_field(var_3)
    assert var_4 == 'forms/input.html'
    var_5 = module_3.Choice()
    var_6 = var_2.template_for_field(var_5)
    assert var_6 == 'forms/select.html'
    var_7 = module_3.Boolean()
    var_8 = var_2.template_for_field(var_7)
    assert var_8 == 'forms/checkbox.html'
    var_9 = 'text'
    var_10 = module_3.String(format=var_9)
    var_11 = var_2.template_for_field(var_10)
    assert var_11 == 'forms/textarea.html'
    var_12 = 'email'
    var_13 = module_3.String(format=var_12)
    var_14 = var_2.template_for_field(var_13)
    assert var_14 == 'forms/input.html'
    var_15 = module_3.Object()
    var_16 = var_2.template_for_field(var_15)



# Parsed testcases at query #11
#--------------------------


import jinja2.environment as module_1
import jinja2.loaders as module_0
import typesystem.fields as module_2
import typesystem.schemas as module_3


def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = 'forms/textarea.html'
    var_2 = 'forms/checkbox.html'
    var_3 = 'forms/select.html'
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}" />'
    var_5 = '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    var_6 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %} />'
    var_7 = '<select name="{{ field_name }}"><option value="{{ value }}">{{ value }}</option></select>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'name'
    var_12 = 'description'
    var_13 = 'active'
    var_14 = 'choice'
    var_15 = 'Name'
    var_16 = module_2.String()
    var_17 = 'Description'
    var_18 = 'text'
    var_19 = module_2.String(format=var_18)
    var_20 = 'Active'
    var_21 = module_2.Boolean()
    var_22 = 'Choice'
    var_23 = 'option1'
    var_24 = 'Option 1'
    var_25 = (var_23, var_24)
    var_26 = 'option2'
    var_27 = 'Option 2'
    var_28 = (var_26, var_27)
    var_29 = [var_25, var_28]
    var_30 = module_2.Choice(choices=var_29)
    var_31 = {var_11: var_16, var_12: var_19, var_13: var_21, var_14: var_30}
    var_32 = module_3.Schema(var_31)
    var_33 = module_4.Form(env=var_10, schema=var_32)
    var_34 = var_32.fields[var_11]
    var_35 = 'John'
    var_36 = var_33.render_field(field_name=var_11, field=var_34, value=var_35)
    assert var_36 == '<input type="text" name="name" value="John" />'
    var_37 = var_32.fields[var_12]
    var_38 = 'Some description'
    var_39 = var_33.render_field(field_name=var_12, field=var_37, value=var_38)
    assert var_39 == '<textarea name="description">Some description</textarea>'
    var_40 = var_32.fields[var_13]
    var_41 = True
    var_42 = var_33.render_field(field_name=var_13, field=var_40, value=var_41)
    assert var_42 == '<input type="checkbox" name="active" checked />'
    var_43 = var_32.fields[var_14]
    var_44 = var_33.render_field(field_name=var_14, field=var_43, value=var_23)
    assert var_44 == '<select name="choice"><option value="option1">option1</option></select>'
    var_45 = var_32.fields[var_11]
    var_46 = 'Invalid name'
    var_47 = var_33.render_field(field_name=var_11, field=var_45, value=var_35, error=var_46)
    assert var_47 == '<input type="text" name="name" value="John" />'
    var_48 = var_32.fields[var_11]
    var_49 = var_33.render_field(field_name=var_11, field=var_48)
    assert var_49 == '<input type="text" name="name" value="" />'
    var_50 = 'Password'
    var_51 = 'password'
    var_52 = module_2.String(format=var_51)
    var_53 = 'secret'
    var_54 = var_33.render_field(field_name=var_51, field=var_52, value=var_53)
    assert var_54 == '<input type="password" name="password" value="" />'
    var_55 = 'Email'
    var_56 = 'email'
    var_57 = module_2.String(format=var_56)
    var_58 = 'test@example.com'
    var_59 = var_33.render_field(field_name=var_56, field=var_57, value=var_58)
    assert var_59 == '<input type="email" name="email" value="test@example.com" />'
    var_60 = 'Required'
    var_61 = False
    var_62 = module_2.String()
    var_63 = 'required'
    var_64 = var_33.render_field(field_name=var_63, field=var_62)
    var_65 = 'Optional'
    var_66 = module_2.String()
    var_67 = 'optional'
    var_68 = var_33.render_field(field_name=var_67, field=var_66)
    var_69 = 'Default'
    var_70 = 'default_value'
    var_71 = module_2.String()
    var_72 = 'default'
    var_73 = var_33.render_field(field_name=var_72, field=var_71)
    var_74 = 'Blank'
    var_75 = module_2.String(allow_blank=var_41)
    var_76 = 'blank'
    var_77 = var_33.render_field(field_name=var_76, field=var_75)
    var_78 = 'NullableBlank'
    var_79 = module_2.String(allow_blank=var_41)
    var_80 = 'nullable_blank'
    var_81 = var_33.render_field(field_name=var_80, field=var_79)
    var_82 = 'Strict'
    var_83 = module_2.String(allow_blank=var_61)
    var_84 = 'strict'
    var_85 = var_33.render_field(field_name=var_84, field=var_83)
    var_86 = 'Custom'
    var_87 = 'custom'
    var_88 = module_2.String(format=var_87)
    var_89 = var_33.render_field(field_name=var_87, field=var_88)
    assert var_89 == '<input type="text" name="custom" value="" />'
    var_90 = 'Date'
    var_91 = 'date'
    var_92 = module_2.String(format=var_91)
    var_93 = '2023-01-01'
    var_94 = var_33.render_field(field_name=var_91, field=var_92, value=var_93)
    assert var_94 == '<input type="date" name="date" value="2023-01-01" />'
    var_95 = 'DateTime'
    var_96 = 'datetime'
    var_97 = module_2.String(format=var_96)
    var_98 = '2023-01-01T12:00'
    var_99 = var_33.render_field(field_name=var_96, field=var_97, value=var_98)
    assert var_99 == '<input type="datetime-local" name="datetime" value="2023-01-01T12:00" />'
    var_100 = 'NonString'
    var_101 = module_2.Boolean()
    var_102 = 'non_string'
    var_103 = var_33.render_field(field_name=var_102, field=var_101)
    assert var_103 == '<input type="checkbox" name="non_string"  />'
    var_104 = 'NonStringWithFormat'
    var_105 = module_2.Boolean()
    var_106 = 'non_string_with_format'
    var_107 = var_33.render_field(field_name=var_106, field=var_105)
    assert var_107 == '<input type="checkbox" name="non_string_with_format"  />'
    var_108 = 'NoFormat'
    var_109 = None
    var_110 = module_2.String(format=var_109)
    var_111 = 'no_format'
    var_112 = var_33.render_field(field_name=var_111, field=var_110)
    assert var_112 == '<input type="text" name="no_format" value="" />'
    var_113 = 'EmptyFormat'
    var_114 = ''
    var_115 = module_2.String(format=var_114)
    var_116 = 'empty_format'
    var_117 = var_33.render_field(field_name=var_116, field=var_115)
    assert var_117 == '<input type="text" name="empty_format" value="" />'
    var_118 = 'WhitespaceFormat'
    var_119 = ' '
    var_120 = module_2.String(format=var_119)
    var_121 = 'whitespace_format'
    var_122 = var_33.render_field(field_name=var_121, field=var_120)
    assert var_122 == '<input type="text" name="whitespace_format" value="" />'
    var_123 = 'NumberFormat'
    var_124 = 123
    var_125 = module_2.String(format=var_124)
    var_126 = 'number_format'
    var_127 = var_33.render_field(field_name=var_126, field=var_125)
    assert var_127 == '<input type="text" name="number_format" value="" />'
    var_128 = 'BoolFormat'
    var_129 = module_2.String(format=var_41)
    var_130 = 'bool_format'
    var_131 = var_33.render_field(field_name=var_130, field=var_129)
    assert var_131 == '<input type="text" name="bool_format" value="" />'
    var_132 = 'ListFormat'
    var_133 = [var_18, var_56]
    var_134 = module_2.String(format=var_133)
    var_135 = 'list_format'
    var_136 = var_33.render_field(field_name=var_135, field=var_134)
    assert var_136 == '<input type="text" name="list_format" value="" />'
    var_137 = 'DictFormat'
    var_138 = 'type'
    var_139 = {var_138: var_18}
    var_140 = module_2.String(format=var_139)
    var_141 = 'dict_format'
    var_142 = var_33.render_field(field_name=var_141, field=var_140)
    assert var_142 == '<input type="text" name="dict_format" value="" />'
    assert var_142 == '<input type="text" name="func_format" value="" />'
    assert var_142 == '<input type="text" name="class_format" value="" />'
    var_143 = 'FuncFormat'
    var_144 = 'func_format'
    var_145 = 'ClassFormat'
    var_146 = 'class_format'
    var_147 = 'InstanceFormat'



# Parsed testcases at query #12
#--------------------------


import jinja2.environment as module_0
import typesystem.fields as module_3
import typesystem.forms as module_2
import typesystem.schemas as module_1


def test_case_0():
    var_0 = module_0.Environment()
    var_1 = module_1.Schema()
    var_2 = module_2.Form(env=var_0, schema=var_1)
    var_3 = module_3.Field()
    var_4 = var_2.template_for_field(var_3)
    assert var_4 == 'forms/input.html'
    var_5 = module_3.Choice()
    var_6 = var_2.template_for_field(var_5)
    assert var_6 == 'forms/select.html'
    var_7 = module_3.Boolean()
    var_8 = var_2.template_for_field(var_7)
    assert var_8 == 'forms/checkbox.html'
    var_9 = 'text'
    var_10 = module_3.String(format=var_9)
    var_11 = var_2.template_for_field(var_10)
    assert var_11 == 'forms/textarea.html'
    var_12 = 'email'
    var_13 = module_3.String(format=var_12)
    var_14 = var_2.template_for_field(var_13)
    assert var_14 == 'forms/input.html'
    var_15 = module_3.Object()
    var_16 = var_2.template_for_field(var_15)



# Parsed testcases at query #13
#--------------------------


import jinja2.environment as module_1
import jinja2.loaders as module_0
import typesystem.fields as module_4
import typesystem.forms as module_3
import typesystem.schemas as module_2


def test_case_0():
    var_0 = {}
    var_1 = module_0.DictLoader(var_0)
    var_2 = module_1.Environment(loader=var_1)
    var_3 = {}
    var_4 = module_2.Schema(var_3)
    var_5 = module_3.Form(env=var_2, schema=var_4)
    var_6 = 'test_field'
    var_7 = module_4.String()
    var_8 = var_5.render_field(field_name=var_6, field=var_7)



# Parsed testcases at query #14
#--------------------------


import typesystem.fields as module_0
import typesystem.forms as module_1


def test_case_0():
    var_0 = '1'
    var_1 = 'One'
    var_2 = (var_0, var_1)
    var_3 = '2'
    var_4 = 'Two'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = module_0.Choice(choices=var_6)
    var_8 = None
    var_9 = module_1.Form(env=var_8, schema=var_8)
    var_10 = var_9.template_for_field(var_7)
    assert var_10 == 'forms/select.html'
    var_11 = module_0.Boolean()
    var_12 = var_9.template_for_field(var_11)
    assert var_12 == 'forms/checkbox.html'
    var_13 = 'text'
    var_14 = module_0.String(format=var_13)
    var_15 = var_9.template_for_field(var_14)
    assert var_15 == 'forms/textarea.html'
    var_16 = module_0.String()
    var_17 = var_9.template_for_field(var_16)
    assert var_17 == 'forms/input.html'
    var_18 = {}
    var_19 = module_0.Object(properties=var_18)
    var_20 = var_9.template_for_field(var_19)



####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import jinja2.environment as module_1
import jinja2.loaders as module_0
import typesystem.fields as module_2
import typesystem.forms as module_4
import typesystem.schemas as module_3


def test_case_0():
    var_0 = '.'
    var_1 = module_0.FileSystemLoader(var_0)
    var_2 = module_1.Environment(loader=var_1)
    var_3 = 'name'
    var_4 = module_2.String()
    var_5 = {var_3: var_4}
    var_6 = module_3.Schema(var_5)
    var_7 = module_4.Form(env=var_2, schema=var_6)
    var_8 = module_2.String()
    var_9 = var_7.render_field(field_name=var_3, field=var_8)
    var_10 = 'color'
    var_11 = 'red'
    var_12 = 'green'
    var_13 = 'blue'
    var_14 = [var_11, var_12, var_13]
    var_15 = module_2.Choice(choices=var_14)
    var_16 = {var_10: var_15}
    var_17 = module_3.Schema(var_16)
    var_18 = module_4.Form(env=var_2, schema=var_17)
    var_19 = [var_11, var_12, var_13]
    var_20 = module_2.Choice(choices=var_19)
    var_21 = var_18.render_field(field_name=var_10, field=var_20)
    var_22 = 'active'
    var_23 = module_2.Boolean()
    var_24 = {var_22: var_23}
    var_25 = module_3.Schema(var_24)
    var_26 = module_4.Form(env=var_2, schema=var_25)
    var_27 = module_2.Boolean()
    var_28 = var_26.render_field(field_name=var_22, field=var_27)
    var_29 = 'description'
    var_30 = 'text'
    var_31 = module_2.String(format=var_30)
    var_32 = {var_29: var_31}
    var_33 = module_3.Schema(var_32)
    var_34 = module_4.Form(env=var_2, schema=var_33)
    var_35 = module_2.String(format=var_30)
    var_36 = var_34.render_field(field_name=var_29, field=var_35)
    var_37 = 'email'
    var_38 = module_2.String(format=var_37)
    var_39 = {var_37: var_38}
    var_40 = module_3.Schema(var_39)
    var_41 = module_4.Form(env=var_2, schema=var_40)
    var_42 = module_2.String(format=var_37)
    var_43 = var_41.render_field(field_name=var_37, field=var_42)
    var_44 = 'password'
    var_45 = module_2.String(format=var_44)
    var_46 = {var_44: var_45}
    var_47 = module_3.Schema(var_46)
    var_48 = module_4.Form(env=var_2, schema=var_47)
    var_49 = module_2.String(format=var_44)
    var_50 = var_48.render_field(field_name=var_44, field=var_49)
    var_51 = 'birthday'
    var_52 = 'date'
    var_53 = module_2.String(format=var_52)
    var_54 = {var_51: var_53}
    var_55 = module_3.Schema(var_54)
    var_56 = module_4.Form(env=var_2, schema=var_55)
    var_57 = module_2.String(format=var_52)
    var_58 = var_56.render_field(field_name=var_51, field=var_57)
    var_59 = 'appointment'
    var_60 = 'datetime'
    var_61 = module_2.String(format=var_60)
    var_62 = {var_59: var_61}
    var_63 = module_3.Schema(var_62)
    var_64 = module_4.Form(env=var_2, schema=var_63)
    var_65 = module_2.String(format=var_60)
    var_66 = var_64.render_field(field_name=var_59, field=var_65)
    var_67 = 'alarm'
    var_68 = 'time'
    var_69 = module_2.String(format=var_68)
    var_70 = {var_67: var_69}
    var_71 = module_3.Schema(var_70)
    var_72 = module_4.Form(env=var_2, schema=var_71)
    var_73 = module_2.String(format=var_68)
    var_74 = var_72.render_field(field_name=var_67, field=var_73)
    var_75 = 'website'
    var_76 = 'url'
    var_77 = module_2.String(format=var_76)
    var_78 = {var_75: var_77}
    var_79 = module_3.Schema(var_78)
    var_80 = module_4.Form(env=var_2, schema=var_79)
    var_81 = module_2.String(format=var_76)
    var_82 = var_80.render_field(field_name=var_75, field=var_81)
    var_83 = 'phone'
    var_84 = 'tel'
    var_85 = module_2.String(format=var_84)
    var_86 = {var_83: var_85}
    var_87 = module_3.Schema(var_86)
    var_88 = module_4.Form(env=var_2, schema=var_87)
    var_89 = module_2.String(format=var_84)
    var_90 = var_88.render_field(field_name=var_83, field=var_89)
    var_91 = 'age'
    var_92 = 'number'
    var_93 = module_2.String(format=var_92)
    var_94 = {var_91: var_93}
    var_95 = module_3.Schema(var_94)
    var_96 = module_4.Form(env=var_2, schema=var_95)
    var_97 = module_2.String(format=var_92)
    var_98 = var_96.render_field(field_name=var_91, field=var_97)
    var_99 = 'volume'
    var_100 = 'range'
    var_101 = module_2.String(format=var_100)
    var_102 = {var_99: var_101}
    var_103 = module_3.Schema(var_102)
    var_104 = module_4.Form(env=var_2, schema=var_103)
    var_105 = module_2.String(format=var_100)
    var_106 = var_104.render_field(field_name=var_99, field=var_105)
    var_107 = module_2.String(format=var_10)
    var_108 = {var_10: var_107}
    var_109 = module_3.Schema(var_108)
    var_110 = module_4.Form(env=var_2, schema=var_109)
    var_111 = module_2.String(format=var_10)
    var_112 = var_110.render_field(field_name=var_10, field=var_111)
    var_113 = 'query'
    var_114 = 'search'
    var_115 = module_2.String(format=var_114)
    var_116 = {var_113: var_115}
    var_117 = module_3.Schema(var_116)
    var_118 = module_4.Form(env=var_2, schema=var_117)
    var_119 = module_2.String(format=var_114)
    var_120 = var_118.render_field(field_name=var_113, field=var_119)
    var_121 = 'month'
    var_122 = module_2.String(format=var_121)
    var_123 = {var_121: var_122}
    var_124 = module_3.Schema(var_123)
    var_125 = module_4.Form(env=var_2, schema=var_124)
    var_126 = module_2.String(format=var_121)
    var_127 = var_125.render_field(field_name=var_121, field=var_126)
    var_128 = 'week'
    var_129 = module_2.String(format=var_128)
    var_130 = {var_128: var_129}
    var_131 = module_3.Schema(var_130)
    var_132 = module_4.Form(env=var_2, schema=var_131)
    var_133 = module_2.String(format=var_128)
    var_134 = var_132.render_field(field_name=var_128, field=var_133)
    var_135 = 'secret'
    var_136 = 'hidden'
    var_137 = module_2.String(format=var_136)
    var_138 = {var_135: var_137}
    var_139 = module_3.Schema(var_138)
    var_140 = module_4.Form(env=var_2, schema=var_139)
    var_141 = module_2.String(format=var_136)
    var_142 = var_140.render_field(field_name=var_135, field=var_141)
    var_143 = module_2.String(format=var_30)
    var_144 = {var_29: var_143}
    var_145 = module_3.Schema(var_144)
    var_146 = module_4.Form(env=var_2, schema=var_145)
    var_147 = module_2.String(format=var_30)
    var_148 = 'This field is required.'
    var_149 = var_146.render_field(field_name=var_29, field=var_147, error=var_148)
    var_150 = module_2.String(format=var_30)
    var_151 = {var_29: var_150}
    var_152 = module_3.Schema(var_151)
    var_153 = module_4.Form(env=var_2, schema=var_152)
    var_154 = module_2.String(format=var_30)
    var_155 = 'Hello, world!'
    var_156 = var_153.render_field(field_name=var_29, field=var_154, value=var_155)
    var_157 = True
    var_158 = module_2.String(format=var_30)
    var_159 = {var_29: var_158}
    var_160 = module_3.Schema(var_159)
    var_161 = module_4.Form(env=var_2, schema=var_160)
    var_162 = module_2.String(format=var_30)
    var_163 = var_161.render_field(field_name=var_29, field=var_162)
    var_164 = False
    var_165 = module_2.String(format=var_30)
    var_166 = {var_29: var_165}
    var_167 = module_3.Schema(var_166)
    var_168 = module_4.Form(env=var_2, schema=var_167)
    var_169 = module_2.String(format=var_30)
    var_170 = var_168.render_field(field_name=var_29, field=var_169)
    var_171 = 'Description'
    var_172 = module_2.String(format=var_30)
    var_173 = {var_29: var_172}
    var_174 = module_3.Schema(var_173)
    var_175 = module_4.Form(env=var_2, schema=var_174)
    var_176 = module_2.String(format=var_30)
    var_177 = var_175.render_field(field_name=var_29, field=var_176)



# Parsed testcases at query #2
#--------------------------


import typesystem.fields as module_0


def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = None
    var_3 = module_0.Boolean()



# Parsed testcases at query #3
#--------------------------


import typesystem.forms as module_0


def test_case_0():
    var_0 = None
    var_1 = module_0.Form(env=var_0, schema=var_0)
    var_2 = 'email'
    var_3 = 'unknown'



# Parsed testcases at query #4
#--------------------------


import typesystem.fields as module_0
import typesystem.forms as module_1


def test_case_0():
    var_0 = '1'
    var_1 = 'One'
    var_2 = (var_0, var_1)
    var_3 = '2'
    var_4 = 'Two'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = module_0.Choice(choices=var_6)
    var_8 = None
    var_9 = module_1.Form(env=var_8, schema=var_8)
    var_10 = var_9.template_for_field(var_7)
    assert var_10 == 'forms/select.html'
    var_11 = module_0.Boolean()
    var_12 = var_9.template_for_field(var_11)
    assert var_12 == 'forms/checkbox.html'
    var_13 = 'text'
    var_14 = module_0.String(format=var_13)
    var_15 = var_9.template_for_field(var_14)
    assert var_15 == 'forms/textarea.html'
    var_16 = module_0.String()
    var_17 = var_9.template_for_field(var_16)
    assert var_17 == 'forms/input.html'
    var_18 = {}
    var_19 = module_0.Object(properties=var_18)
    var_20 = var_9.template_for_field(var_19)



# Parsed testcases at query #5
#--------------------------



def test_case_0():
    var_0 = '1'
    var_1 = 'One'
    var_2 = (var_0, var_1)
    var_3 = '2'
    var_4 = 'Two'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = module_0.Choice(choices=var_6)
    var_8 = None
    var_9 = module_1.Form(env=var_8, schema=var_8)
    var_10 = var_9.template_for_field(var_7)
    assert var_10 == 'forms/select.html'
    var_11 = module_0.Boolean()
    var_12 = var_9.template_for_field(var_11)
    assert var_12 == 'forms/checkbox.html'
    var_13 = 'text'
    var_14 = module_0.String(format=var_13)
    var_15 = var_9.template_for_field(var_14)
    assert var_15 == 'forms/textarea.html'
    var_16 = module_0.String()
    var_17 = var_9.template_for_field(var_16)
    assert var_17 == 'forms/input.html'
    var_18 = {}
    var_19 = module_0.Object(properties=var_18)
    var_20 = var_9.template_for_field(var_19)



# Parsed testcases at query #6
#--------------------------



def test_case_0():
    var_0 = module_0.Field()
    var_1 = None
    var_2 = module_1.Form(env=var_1, schema=var_1)
    var_3 = var_2.input_type_for_field(var_0)
    assert var_3 == 'text'
    var_4 = var_2.input_type_for_field(var_0)
    assert var_4 == 'text'
    var_5 = var_2.input_type_for_field(var_0)
    assert var_5 == 'email'
    var_6 = var_2.input_type_for_field(var_0)
    assert var_6 == 'color'
    var_7 = var_2.input_type_for_field(var_0)
    assert var_7 == 'datetime-local'
    var_8 = var_2.input_type_for_field(var_0)
    assert var_8 == 'date'
    var_9 = var_2.input_type_for_field(var_0)
    assert var_9 == 'month'
    var_10 = var_2.input_type_for_field(var_0)
    assert var_10 == 'number'
    var_11 = var_2.input_type_for_field(var_0)
    assert var_11 == 'password'
    var_12 = var_2.input_type_for_field(var_0)
    assert var_12 == 'range'
    var_13 = var_2.input_type_for_field(var_0)
    assert var_13 == 'search'
    var_14 = var_2.input_type_for_field(var_0)
    assert var_14 == 'tel'
    var_15 = var_2.input_type_for_field(var_0)
    assert var_15 == 'time'
    var_16 = var_2.input_type_for_field(var_0)
    assert var_16 == 'url'
    var_17 = var_2.input_type_for_field(var_0)
    assert var_17 == 'week'
    var_18 = var_2.input_type_for_field(var_0)
    assert var_18 == 'text'
    var_19 = var_2.input_type_for_field(var_0)
    assert var_19 == 'hidden'
    var_20 = var_2.input_type_for_field(var_0)
    assert var_20 == 'color'
    var_21 = var_2.input_type_for_field(var_0)
    assert var_21 == 'datetime-local'
    var_22 = var_2.input_type_for_field(var_0)
    assert var_22 == 'date'
    var_23 = var_2.input_type_for_field(var_0)
    assert var_23 == 'month'
    var_24 = var_2.input_type_for_field(var_0)
    assert var_24 == 'number'
    var_25 = var_2.input_type_for_field(var_0)
    assert var_25 == 'password'
    var_26 = var_2.input_type_for_field(var_0)
    assert var_26 == 'range'
    var_27 = var_2.input_type_for_field(var_0)
    assert var_27 == 'search'
    var_28 = var_2.input_type_for_field(var_0)
    assert var_28 == 'tel'
    var_29 = var_2.input_type_for_field(var_0)
    assert var_29 == 'time'
    var_30 = var_2.input_type_for_field(var_0)
    assert var_30 == 'url'
    var_31 = var_2.input_type_for_field(var_0)
    assert var_31 == 'week'
    var_32 = var_2.input_type_for_field(var_0)
    assert var_32 == 'text'
    var_33 = var_2.input_type_for_field(var_0)
    assert var_33 == 'hidden'
    var_34 = var_2.input_type_for_field(var_0)
    assert var_34 == 'color'
    var_35 = var_2.input_type_for_field(var_0)
    assert var_35 == 'datetime-local'
    var_36 = var_2.input_type_for_field(var_0)
    assert var_36 == 'date'
    var_37 = var_2.input_type_for_field(var_0)
    assert var_37 == 'month'
    var_38 = var_2.input_type_for_field(var_0)
    assert var_38 == 'number'
    var_39 = var_2.input_type_for_field(var_0)
    assert var_39 == 'password'
    var_40 = var_2.input_type_for_field(var_0)
    assert var_40 == 'range'
    var_41 = var_2.input_type_for_field(var_0)
    assert var_41 == 'search'
    var_42 = var_2.input_type_for_field(var_0)
    assert var_42 == 'tel'
    var_43 = var_2.input_type_for_field(var_0)
    assert var_43 == 'time'
    var_44 = var_2.input_type_for_field(var_0)
    assert var_44 == 'url'
    var_45 = var_2.input_type_for_field(var_0)
    assert var_45 == 'week'
    var_46 = var_2.input_type_for_field(var_0)
    assert var_46 == 'text'
    var_47 = var_2.input_type_for_field(var_0)
    assert var_47 == 'hidden'
    var_48 = var_2.input_type_for_field(var_0)
    assert var_48 == 'color'
    var_49 = var_2.input_type_for_field(var_0)
    assert var_49 == 'datetime-local'



# Parsed testcases at query #7
#--------------------------



def test_case_0():
    var_0 = '1'
    var_1 = 'One'
    var_2 = (var_0, var_1)
    var_3 = '2'
    var_4 = 'Two'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = module_0.Choice(choices=var_6)
    var_8 = None
    var_9 = module_1.Form(env=var_8, schema=var_8)
    var_10 = var_9.template_for_field(var_7)
    assert var_10 == 'forms/select.html'
    var_11 = module_0.Boolean()
    var_12 = var_9.template_for_field(var_11)
    assert var_12 == 'forms/checkbox.html'
    var_13 = 'text'
    var_14 = module_0.String(format=var_13)
    var_15 = var_9.template_for_field(var_14)
    assert var_15 == 'forms/textarea.html'
    var_16 = module_0.String()
    var_17 = var_9.template_for_field(var_16)
    assert var_17 == 'forms/input.html'
    var_18 = {}
    var_19 = module_0.Object(properties=var_18)
    var_20 = var_9.template_for_field(var_19)



# Parsed testcases at query #8
#--------------------------



def test_case_0():
    var_0 = 'email'
    var_1 = module_0.String(format=var_0)
    var_2 = None
    var_3 = module_1.Form(env=var_2, schema=var_2)
    var_4 = var_3.input_type_for_field(var_1)
    assert var_4 == 'email'
    var_5 = 'unknown'
    var_6 = module_0.String(format=var_5)
    var_7 = var_3.input_type_for_field(var_6)
    assert var_7 == 'text'
    var_8 = module_0.String()
    var_9 = var_3.input_type_for_field(var_8)
    assert var_9 == 'text'
    var_10 = module_0.Boolean()
    var_11 = var_3.input_type_for_field(var_10)
    assert var_11 == 'text'
    var_12 = '1'
    var_13 = 'One'
    var_14 = (var_12, var_13)
    var_15 = '2'
    var_16 = 'Two'
    var_17 = (var_15, var_16)
    var_18 = [var_14, var_17]
    var_19 = module_0.Choice(choices=var_18)
    var_20 = var_3.input_type_for_field(var_19)
    assert var_20 == 'text'
    var_21 = {}
    var_22 = module_0.Object(properties=var_21)
    var_23 = var_3.input_type_for_field(var_22)
    var_24 = 'color'
    var_25 = module_0.String(format=var_24)
    var_26 = var_3.input_type_for_field(var_25)
    assert var_26 == 'color'
    var_27 = 'datetime'
    var_28 = module_0.String(format=var_27)
    var_29 = var_3.input_type_for_field(var_28)
    assert var_29 == 'datetime-local'
    var_30 = 'date'
    var_31 = module_0.String(format=var_30)
    var_32 = var_3.input_type_for_field(var_31)
    assert var_32 == 'date'
    var_33 = 'month'
    var_34 = module_0.String(format=var_33)
    var_35 = var_3.input_type_for_field(var_34)
    assert var_35 == 'month'
    var_36 = 'number'
    var_37 = module_0.String(format=var_36)
    var_38 = var_3.input_type_for_field(var_37)
    assert var_38 == 'number'
    var_39 = 'password'
    var_40 = module_0.String(format=var_39)
    var_41 = var_3.input_type_for_field(var_40)
    assert var_41 == 'password'
    var_42 = 'range'
    var_43 = module_0.String(format=var_42)
    var_44 = var_3.input_type_for_field(var_43)
    assert var_44 == 'range'
    var_45 = 'search'
    var_46 = module_0.String(format=var_45)
    var_47 = var_3.input_type_for_field(var_46)
    assert var_47 == 'search'
    var_48 = 'tel'
    var_49 = module_0.String(format=var_48)
    var_50 = var_3.input_type_for_field(var_49)
    assert var_50 == 'tel'
    var_51 = 'time'
    var_52 = module_0.String(format=var_51)
    var_53 = var_3.input_type_for_field(var_52)
    assert var_53 == 'time'
    var_54 = 'url'
    var_55 = module_0.String(format=var_54)
    var_56 = var_3.input_type_for_field(var_55)
    assert var_56 == 'url'
    var_57 = 'week'
    var_58 = module_0.String(format=var_57)
    var_59 = var_3.input_type_for_field(var_58)
    assert var_59 == 'week'
    var_60 = 'hidden'
    var_61 = module_0.String(format=var_60)
    var_62 = var_3.input_type_for_field(var_61)
    assert var_62 == 'hidden'
    var_63 = 'text'
    var_64 = module_0.String(format=var_63)
    var_65 = var_3.input_type_for_field(var_64)
    assert var_65 == 'text'
    var_66 = module_0.String(format=var_23)
    var_67 = var_3.input_type_for_field(var_66)
    assert var_67 == 'email'
    var_68 = module_0.String(format=var_27)
    var_69 = var_3.input_type_for_field(var_68)
    assert var_69 == 'datetime-local'
    var_70 = module_0.String(format=var_30)
    var_71 = var_3.input_type_for_field(var_70)
    assert var_71 == 'date'
    var_72 = module_0.String(format=var_33)
    var_73 = var_3.input_type_for_field(var_72)
    assert var_73 == 'month'
    var_74 = module_0.String(format=var_36)
    var_75 = var_3.input_type_for_field(var_74)
    assert var_75 == 'number'
    var_76 = module_0.String(format=var_39)
    var_77 = var_3.input_type_for_field(var_76)
    assert var_77 == 'password'
    var_78 = module_0.String(format=var_42)
    var_79 = var_3.input_type_for_field(var_78)
    assert var_79 == 'range'
    var_80 = module_0.String(format=var_45)
    var_81 = var_3.input_type_for_field(var_80)
    assert var_81 == 'search'
    var_82 = module_0.String(format=var_48)
    var_83 = var_3.input_type_for_field(var_82)
    assert var_83 == 'tel'
    var_84 = module_0.String(format=var_51)
    var_85 = var_3.input_type_for_field(var_84)
    assert var_85 == 'time'
    var_86 = module_0.String(format=var_54)
    var_87 = var_3.input_type_for_field(var_86)
    assert var_87 == 'url'
    var_88 = module_0.String(format=var_57)
    var_89 = var_3.input_type_for_field(var_88)
    assert var_89 == 'week'
    var_90 = module_0.String(format=var_60)
    var_91 = var_3.input_type_for_field(var_90)
    assert var_91 == 'hidden'
    var_92 = module_0.String(format=var_63)
    var_93 = var_3.input_type_for_field(var_92)
    assert var_93 == 'text'
    var_94 = module_0.String(format=var_23)
    var_95 = var_3.input_type_for_field(var_94)
    assert var_95 == 'email'
    var_96 = module_0.String(format=var_27)
    var_97 = var_3.input_type_for_field(var_96)
    assert var_97 == 'datetime-local'
    var_98 = module_0.String(format=var_30)
    var_99 = var_3.input_type_for_field(var_98)
    assert var_99 == 'date'
    var_100 = module_0.String(format=var_33)
    var_101 = var_3.input_type_for_field(var_100)
    assert var_101 == 'month'



# Parsed testcases at query #9
#--------------------------



def test_case_0():
    var_0 = 'color'
    var_1 = module_0.Field()
    var_2 = None
    var_3 = module_1.Form(env=var_2, schema=var_2)
    var_4 = var_3.input_type_for_field(var_1)
    assert var_4 == 'color'
    var_5 = 'datetime'
    var_6 = module_0.Field()
    var_7 = module_1.Form(env=var_2, schema=var_2)
    var_8 = var_7.input_type_for_field(var_6)
    assert var_8 == 'datetime-local'
    var_9 = 'date'
    var_10 = module_0.Field()
    var_11 = module_1.Form(env=var_2, schema=var_2)
    var_12 = var_11.input_type_for_field(var_10)
    assert var_12 == 'date'
    var_13 = 'email'
    var_14 = module_0.Field()
    var_15 = module_1.Form(env=var_2, schema=var_2)
    var_16 = var_15.input_type_for_field(var_14)
    assert var_16 == 'email'
    var_17 = 'hidden'
    var_18 = module_0.Field()
    var_19 = module_1.Form(env=var_2, schema=var_2)
    var_20 = var_19.input_type_for_field(var_18)
    assert var_20 == 'hidden'
    var_21 = 'month'
    var_22 = module_0.Field()
    var_23 = module_1.Form(env=var_2, schema=var_2)
    var_24 = var_23.input_type_for_field(var_22)
    assert var_24 == 'month'
    var_25 = 'number'
    var_26 = module_0.Field()
    var_27 = module_1.Form(env=var_2, schema=var_2)
    var_28 = var_27.input_type_for_field(var_26)
    assert var_28 == 'number'
    var_29 = 'password'
    var_30 = module_0.Field()
    var_31 = module_1.Form(env=var_2, schema=var_2)
    var_32 = var_31.input_type_for_field(var_30)
    assert var_32 == 'password'
    var_33 = 'range'
    var_34 = module_0.Field()
    var_35 = module_1.Form(env=var_2, schema=var_2)
    var_36 = var_35.input_type_for_field(var_34)
    assert var_36 == 'range'
    var_37 = 'search'
    var_38 = module_0.Field()
    var_39 = module_1.Form(env=var_2, schema=var_2)
    var_40 = var_39.input_type_for_field(var_38)
    assert var_40 == 'search'
    var_41 = 'tel'
    var_42 = module_0.Field()
    var_43 = module_1.Form(env=var_2, schema=var_2)
    var_44 = var_43.input_type_for_field(var_42)
    assert var_44 == 'tel'
    var_45 = 'text'
    var_46 = module_0.Field()
    var_47 = module_1.Form(env=var_2, schema=var_2)
    var_48 = var_47.input_type_for_field(var_46)
    assert var_48 == 'text'
    var_49 = 'time'
    var_50 = module_0.Field()
    var_51 = module_1.Form(env=var_2, schema=var_2)
    var_52 = var_51.input_type_for_field(var_50)
    assert var_52 == 'time'
    var_53 = 'url'
    var_54 = module_0.Field()
    var_55 = module_1.Form(env=var_2, schema=var_2)
    var_56 = var_55.input_type_for_field(var_54)
    assert var_56 == 'url'
    var_57 = 'week'
    var_58 = module_0.Field()
    var_59 = module_1.Form(env=var_2, schema=var_2)
    var_60 = var_59.input_type_for_field(var_58)
    assert var_60 == 'week'
    var_61 = 'unknown'
    var_62 = module_0.Field()
    var_63 = module_1.Form(env=var_2, schema=var_2)
    var_64 = var_63.input_type_for_field(var_62)
    assert var_64 == 'text'
    var_65 = module_0.Field()
    var_66 = module_1.Form(env=var_2, schema=var_2)
    var_67 = var_66.input_type_for_field(var_65)
    assert var_67 == 'text'
    var_68 = True
    var_69 = module_0.Field()
    var_70 = module_1.Form(env=var_2, schema=var_2)
    var_71 = var_70.input_type_for_field(var_69)
    assert var_71 == 'color'
    var_72 = module_0.Field()
    var_73 = module_1.Form(env=var_2, schema=var_2)
    var_74 = var_73.input_type_for_field(var_72)
    assert var_74 == 'datetime-local'
    var_75 = module_0.Field()
    var_76 = module_1.Form(env=var_2, schema=var_2)
    var_77 = var_76.input_type_for_field(var_75)
    assert var_77 == 'date'
    var_78 = module_0.Field()
    var_79 = module_1.Form(env=var_2, schema=var_2)
    var_80 = var_79.input_type_for_field(var_78)
    assert var_80 == 'email'
    var_81 = module_0.Field()
    var_82 = module_1.Form(env=var_2, schema=var_2)
    var_83 = var_82.input_type_for_field(var_81)
    assert var_83 == 'hidden'
    var_84 = module_0.Field()
    var_85 = module_1.Form(env=var_2, schema=var_2)
    var_86 = var_85.input_type_for_field(var_84)
    assert var_86 == 'month'
    var_87 = module_0.Field()
    var_88 = module_1.Form(env=var_2, schema=var_2)
    var_89 = var_88.input_type_for_field(var_87)
    assert var_89 == 'number'
    var_90 = module_0.Field()
    var_91 = module_1.Form(env=var_2, schema=var_2)
    var_92 = var_91.input_type_for_field(var_90)
    assert var_92 == 'password'
    var_93 = module_0.Field()
    var_94 = module_1.Form(env=var_2, schema=var_2)
    var_95 = var_94.input_type_for_field(var_93)
    assert var_95 == 'range'
    var_96 = module_0.Field()
    var_97 = module_1.Form(env=var_2, schema=var_2)
    var_98 = var_97.input_type_for_field(var_96)
    assert var_98 == 'search'
    var_99 = module_0.Field()
    var_100 = module_1.Form(env=var_2, schema=var_2)
    var_101 = var_100.input_type_for_field(var_99)
    assert var_101 == 'tel'
    var_102 = module_0.Field()
    var_103 = module_1.Form(env=var_2, schema=var_2)
    var_104 = var_103.input_type_for_field(var_102)
    assert var_104 == 'text'
    var_105 = module_0.Field()
    var_106 = module_1.Form(env=var_2, schema=var_2)
    var_107 = var_106.input_type_for_field(var_105)
    assert var_107 == 'time'
    var_108 = module_0.Field()
    var_109 = module_1.Form(env=var_2, schema=var_2)
    var_110 = var_109.input_type_for_field(var_108)
    assert var_110 == 'url'
    var_111 = module_0.Field()
    var_112 = module_1.Form(env=var_2, schema=var_2)
    var_113 = var_112.input_type_for_field(var_111)
    assert var_113 == 'week'
    var_114 = module_0.Field()
    var_115 = module_1.Form(env=var_2, schema=var_2)
    var_116 = var_115.input_type_for_field(var_114)
    assert var_116 == 'text'
    var_117 = module_0.Field()
    var_118 = module_1.Form(env=var_2, schema=var_2)
    var_119 = var_118.input_type_for_field(var_117)
    assert var_119 == 'text'
    var_120 = False
    var_121 = module_0.Field()
    var_122 = module_1.Form(env=var_2, schema=var_2)
    var_123 = var_122.input_type_for_field(var_121)
    assert var_123 == 'color'
    var_124 = module_0.Field()
    var_125 = module_1.Form(env=var_2, schema=var_2)
    var_126 = var_125.input_type_for_field(var_124)
    assert var_126 == 'datetime-local'
    var_127 = module_0.Field()
    var_128 = module_1.Form(env=var_2, schema=var_2)
    var_129 = var_128.input_type_for_field(var_127)
    assert var_129 == 'date'
    var_130 = module_0.Field()
    var_131 = module_1.Form(env=var_2, schema=var_2)
    var_132 = var_131.input_type_for_field(var_130)
    assert var_132 == 'email'
    var_133 = module_0.Field()
    var_134 = module_1.Form(env=var_2, schema=var_2)
    var_135 = var_134.input_type_for_field(var_133)
    assert var_135 == 'hidden'
    var_136 = module_0.Field()
    var_137 = module_1.Form(env=var_2, schema=var_2)
    var_138 = var_137.input_type_for_field(var_136)
    assert var_138 == 'month'
    var_139 = module_0.Field()
    var_140 = module_1.Form(env=var_2, schema=var_2)



# Parsed testcases at query #10
#--------------------------


import typesystem.forms as module_0


def test_case_0():
    var_0 = 'templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = 'tests'
    var_3 = module_0.Jinja2Forms(package=var_2)
    var_4 = module_0.Jinja2Forms(directory=var_0, package=var_2)
    var_5 = module_0.Jinja2Forms()
    var_6 = 'templates'
    var_7 = module_0.Jinja2Forms(directory=var_6)
    var_8 = module_0.Jinja2Forms(directory=var_6)
    var_9 = module_0.Jinja2Forms(package=var_2)
    var_10 = module_0.Jinja2Forms(directory=var_6, package=var_2)
    var_11 = module_0.Jinja2Forms()
    var_12 = 'does_not_exist'
    var_13 = module_0.Jinja2Forms(directory=var_12)
    var_14 = module_0.Jinja2Forms(package=var_12)
    var_15 = module_0.Jinja2Forms(directory=var_12, package=var_12)
    var_16 = module_0.Jinja2Forms(directory=var_6, package=var_12)
    var_17 = module_0.Jinja2Forms(directory=var_12, package=var_2)
    var_18 = module_0.Jinja2Forms(directory=var_6, package=var_2)
    var_19 = module_0.Jinja2Forms(directory=var_6, package=var_2)
    var_20 = module_0.Jinja2Forms(directory=var_6, package=var_2)
    var_21 = module_0.Jinja2Forms(directory=var_6, package=var_2)
    var_22 = module_0.Jinja2Forms(directory=var_6, package=var_2)
    var_23 = module_0.Jinja2Forms(directory=var_6, package=var_2)
    var_24 = module_0.Jinja2Forms(directory=var_6, package=var_2)
    var_25 = module_0.Jinja2Forms(directory=var_6, package=var_2)
    var_26 = module_0.Jinja2Forms(directory=var_6, package=var_2)
    var_27 = module_0.Jinja2Forms(directory=var_6, package=var_2)
    var_28 = module_0.Jinja2Forms(directory=var_6, package=var_2)
    var_29 = module_0.Jinja2Forms(directory=var_6, package=var_2)
    var_30 = module_0.Jinja2Forms(directory=var_6, package=var_2)
    var_31 = module_0.Jinja2Forms(directory=var_6, package=var_2)
    var_32 = module_0.Jinja2Forms(directory=var_6, package=var_2)
    var_33 = module_0.Jinja2Forms(directory=var_6, package=var_2)
    var_34 = module_0.Jinja2Forms(directory=var_6, package=var_2)
    var_35 = module_0.Jinja2Forms(directory=var_6, package=var_2)
    var_36 = module_0.Jinja2Forms(directory=var_6, package=var_2)
    var_37 = module_0.Jinja2Forms(directory=var_6, package=var_2)



# Parsed testcases at query #11
#--------------------------


import jinja2.environment as module_1
import jinja2.loaders as module_0
import typesystem.fields as module_4
import typesystem.forms as module_3
import typesystem.schemas as module_2


def test_case_0():
    var_0 = {}
    var_1 = module_0.DictLoader(var_0)
    var_2 = module_1.Environment(loader=var_1)
    var_3 = {}
    var_4 = module_2.Schema(var_3)
    var_5 = module_3.Form(env=var_2, schema=var_4)
    var_6 = 'Test Field'
    var_7 = True
    var_8 = module_4.String(allow_blank=var_7)
    var_9 = 'test_field'
    var_10 = 'test value'
    var_11 = 'test error'
    var_12 = var_5.render_field(field_name=var_9, field=var_8, value=var_10, error=var_11)
    assert var_12 == ''



# Parsed testcases at query #12
#--------------------------



def test_case_0():
    var_0 = 'forms/input.html'
    var_1 = 'forms/textarea.html'
    var_2 = 'forms/checkbox.html'
    var_3 = 'forms/select.html'
    var_4 = '<input type="{{ input_type }}" name="{{ field_name }}" value="{{ value }}" />'
    var_5 = '<textarea name="{{ field_name }}">{{ value }}</textarea>'
    var_6 = '<input type="checkbox" name="{{ field_name }}" {% if value %}checked{% endif %} />'
    var_7 = '<select name="{{ field_name }}">{% for key, val in field.choices %}<option value="{{ key }}" {% if value == key %}selected{% endif %}>{{ val }}</option>{% endfor %}</select>'
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.DictLoader(var_8)
    var_10 = module_1.Environment(loader=var_9)
    var_11 = 'name'
    var_12 = 'email'
    var_13 = 'age'
    var_14 = 'bio'
    var_15 = 'active'
    var_16 = 'role'
    var_17 = 'John'
    var_18 = 'john@example.com'
    var_19 = 30
    var_20 = 'Developer'
    var_21 = True
    var_22 = 'admin'
    var_23 = {var_11: var_17, var_12: var_18, var_13: var_19, var_14: var_20, var_15: var_21, var_16: var_22}
    var_24 = ''
    var_25 = 'invalid'
    var_26 = 'not a number'
    var_27 = False
    var_28 = {var_11: var_24, var_12: var_25, var_13: var_26, var_14: var_24, var_15: var_27, var_16: var_25}
    var_29 = 'All tests passed!'
    var_30 = print(var_29)



# Parsed testcases at query #13
#--------------------------


import typesystem.fields as module_0
import typesystem.forms as module_1


def test_case_0():
    var_0 = module_0.Field()
    var_1 = None
    var_2 = module_1.Form(env=var_1, schema=var_1)
    var_3 = var_2.input_type_for_field(var_0)
    assert var_3 == 'text'
    var_4 = 'unknown'
    var_5 = module_0.Field()
    var_6 = var_2.input_type_for_field(var_5)
    assert var_6 == 'text'
    var_7 = 'email'
    var_8 = module_0.Field()
    var_9 = var_2.input_type_for_field(var_8)
    assert var_9 == 'email'
    var_10 = 'datetime'
    var_11 = module_0.Field()
    var_12 = var_2.input_type_for_field(var_11)
    assert var_12 == 'datetime-local'
    var_13 = 'date'
    var_14 = module_0.Field()
    var_15 = var_2.input_type_for_field(var_14)
    assert var_15 == 'date'
    var_16 = 'time'
    var_17 = module_0.Field()
    var_18 = var_2.input_type_for_field(var_17)
    assert var_18 == 'time'
    var_19 = 'url'
    var_20 = module_0.Field()
    var_21 = var_2.input_type_for_field(var_20)
    assert var_21 == 'url'
    var_22 = 'password'
    var_23 = module_0.Field()
    var_24 = var_2.input_type_for_field(var_23)
    assert var_24 == 'password'
    var_25 = 'search'
    var_26 = module_0.Field()
    var_27 = var_2.input_type_for_field(var_26)
    assert var_27 == 'search'
    var_28 = 'tel'
    var_29 = module_0.Field()
    var_30 = var_2.input_type_for_field(var_29)
    assert var_30 == 'tel'
    var_31 = 'color'
    var_32 = module_0.Field()
    var_33 = var_2.input_type_for_field(var_32)
    assert var_33 == 'color'
    var_34 = 'range'
    var_35 = module_0.Field()
    var_36 = var_2.input_type_for_field(var_35)
    assert var_36 == 'range'
    var_37 = 'month'
    var_38 = module_0.Field()
    var_39 = var_2.input_type_for_field(var_38)
    assert var_39 == 'month'
    var_40 = 'week'
    var_41 = module_0.Field()
    var_42 = var_2.input_type_for_field(var_41)
    assert var_42 == 'week'
    var_43 = 'number'
    var_44 = module_0.Field()
    var_45 = var_2.input_type_for_field(var_44)
    assert var_45 == 'number'
    var_46 = 'hidden'
    var_47 = module_0.Field()
    var_48 = var_2.input_type_for_field(var_47)
    assert var_48 == 'hidden'
    var_49 = 'text'
    var_50 = module_0.Field()
    var_51 = var_2.input_type_for_field(var_50)
    assert var_51 == 'text'
    var_52 = 'datetime-local'
    var_53 = module_0.Field()
    var_54 = var_2.input_type_for_field(var_53)
    assert var_54 == 'datetime-local'
    var_55 = module_0.Field()
    var_56 = var_2.input_type_for_field(var_55)
    assert var_56 == 'date'
    var_57 = module_0.Field()
    var_58 = var_2.input_type_for_field(var_57)
    assert var_58 == 'time'
    var_59 = module_0.Field()
    var_60 = var_2.input_type_for_field(var_59)
    assert var_60 == 'url'
    var_61 = module_0.Field()
    var_62 = var_2.input_type_for_field(var_61)
    assert var_62 == 'password'
    var_63 = module_0.Field()
    var_64 = var_2.input_type_for_field(var_63)
    assert var_64 == 'search'
    var_65 = module_0.Field()
    var_66 = var_2.input_type_for_field(var_65)
    assert var_66 == 'tel'
    var_67 = module_0.Field()
    var_68 = var_2.input_type_for_field(var_67)
    assert var_68 == 'color'
    var_69 = module_0.Field()
    var_70 = var_2.input_type_for_field(var_69)
    assert var_70 == 'range'
    var_71 = module_0.Field()
    var_72 = var_2.input_type_for_field(var_71)
    assert var_72 == 'month'
    var_73 = module_0.Field()
    var_74 = var_2.input_type_for_field(var_73)
    assert var_74 == 'week'
    var_75 = module_0.Field()
    var_76 = var_2.input_type_for_field(var_75)
    assert var_76 == 'number'
    var_77 = module_0.Field()
    var_78 = var_2.input_type_for_field(var_77)
    assert var_78 == 'hidden'
    var_79 = module_0.Field()
    var_80 = var_2.input_type_for_field(var_79)
    assert var_80 == 'text'
    var_81 = module_0.Field()
    var_82 = var_2.input_type_for_field(var_81)
    assert var_82 == 'datetime-local'
    var_83 = module_0.Field()
    var_84 = var_2.input_type_for_field(var_83)
    assert var_84 == 'date'
    var_85 = module_0.Field()
    var_86 = var_2.input_type_for_field(var_85)
    assert var_86 == 'time'
    var_87 = module_0.Field()
    var_88 = var_2.input_type_for_field(var_87)
    assert var_88 == 'url'
    var_89 = module_0.Field()
    var_90 = var_2.input_type_for_field(var_89)
    assert var_90 == 'password'
    var_91 = module_0.Field()
    var_92 = var_2.input_type_for_field(var_91)
    assert var_92 == 'search'
    var_93 = module_0.Field()
    var_94 = var_2.input_type_for_field(var_93)
    assert var_94 == 'tel'
    var_95 = module_0.Field()
    var_96 = var_2.input_type_for_field(var_95)
    assert var_96 == 'color'
    var_97 = module_0.Field()
    var_98 = var_2.input_type_for_field(var_97)
    assert var_98 == 'range'



# Parsed testcases at query #14
#--------------------------


import typesystem.fields as module_1
import typesystem.forms as module_0


def test_case_0():
    var_0 = '/path/to/templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = 'myapp'
    var_3 = module_0.Jinja2Forms(package=var_2)
    var_4 = var_3.env.loader
    var_5 = module_0.Jinja2Forms(directory=var_0, package=var_2)
    var_6 = var_5.env.loader
    var_7 = var_5.env.loader.loaders
    var_8 = len(var_7)
    assert var_8 == 2
    var_9 = 1
    var_10 = var_5.env.loader.loaders[var_9]
    var_11 = '/path/to/templates'
    var_12 = module_0.Jinja2Forms(directory=var_11)
    var_13 = module_0.Jinja2Forms()
    var_14 = module_0.Jinja2Forms(directory=var_13)
    var_15 = module_0.Jinja2Forms(directory=var_13)
    var_16 = var_15.env.loader
    var_17 = module_0.Jinja2Forms(package=var_12)
    var_18 = var_17.env.loader
    var_19 = module_0.Jinja2Forms(directory=var_13, package=var_12)
    var_20 = var_19.env.loader
    var_21 = var_19.env.loader.loaders
    var_22 = len(var_21)
    assert var_22 == 2
    var_23 = var_19.env.loader.loaders[var_9]
    var_24 = module_0.Jinja2Forms(directory=var_13)
    var_25 = var_24.env.loader
    var_26 = module_0.Jinja2Forms(package=var_12)
    var_27 = var_26.env.loader
    var_28 = module_0.Jinja2Forms(directory=var_13, package=var_12)
    var_29 = var_28.env.loader
    var_30 = var_28.env.loader.loaders
    var_31 = len(var_30)
    assert var_31 == 2
    var_32 = var_28.env.loader.loaders[var_9]
    var_33 = module_0.Jinja2Forms(directory=var_13)
    var_34 = 'forms/input.html'
    var_35 = module_0.Jinja2Forms(package=var_12)
    var_36 = module_0.Jinja2Forms(directory=var_13, package=var_12)
    var_37 = module_0.Jinja2Forms(directory=var_13)
    var_38 = 'test'
    var_39 = module_1.String()
    var_40 = 'Test'
    var_41 = True
    var_42 = 'text'
    var_43 = ''
    var_44 = None
    var_45 = module_0.Jinja2Forms(package=var_12)
    var_46 = module_1.String()
    var_47 = True
    var_48 = module_0.Jinja2Forms(directory=var_13, package=var_12)
    var_49 = module_1.String()
    var_50 = True
    var_51 = module_0.Jinja2Forms(directory=var_13)
    var_52 = module_1.String()
    var_53 = True
    var_54 = "<script>alert('xss')</script>"
    var_55 = module_0.Jinja2Forms(package=var_12)
    var_56 = module_1.String()
    var_57 = True
    var_58 = module_0.Jinja2Forms(directory=var_13, package=var_12)
    var_59 = module_1.String()
    var_60 = True
    var_61 = module_0.Jinja2Forms(directory=var_13)
    var_62 = module_1.String()
    var_63 = True
    var_64 = module_0.Jinja2Forms(package=var_12)
    var_65 = module_1.String()
    var_66 = True
    var_67 = module_0.Jinja2Forms(directory=var_13, package=var_12)
    var_68 = module_1.String()
    var_69 = True
    var_70 = module_0.Jinja2Forms(directory=var_13)



