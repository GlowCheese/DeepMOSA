####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.schemas as module_0


def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'test'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = 1
    var_6 = var_2.validate(var_5)
    assert var_6 == 1
    var_7 = None
    var_8 = var_2.validate(var_7)



# Parsed testcases at query #2
#--------------------------


import typesystem.fields as module_1


def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = 'default_value'
    var_3 = module_1.Field(default=var_2)
    var_4 = 'field1'
    var_5 = {var_4: var_3}
    var_6 = module_0.Schema(var_5)
    var_7 = module_1.Field()
    var_8 = 'field2'
    var_9 = {var_8: var_7}
    var_10 = module_0.Schema(var_9)
    var_11 = True
    var_12 = module_1.Field(read_only=var_11)
    var_13 = 'field3'
    var_14 = {var_13: var_12}
    var_15 = module_0.Schema(var_14)
    var_16 = {}
    var_17 = module_0.Schema(var_16)
    var_18 = {}
    var_19 = module_0.Schema(var_18)
    var_20 = 'type'
    var_21 = 'Custom type error'
    var_22 = {var_20: var_21}
    var_23 = {}
    var_24 = module_0.Schema(var_23)
    var_25 = {}
    var_26 = module_0.Schema(var_25)
    var_27 = {}
    var_28 = 'Test Schema'
    var_29 = 'A test schema'
    var_30 = module_0.Schema(var_27)
    var_31 = 'default1'
    var_32 = module_1.Field(default=var_31)
    var_33 = module_1.Field()
    var_34 = module_1.Field(read_only=var_11)
    var_35 = {var_4: var_32, var_8: var_33, var_13: var_34}
    var_36 = module_0.Schema(var_35)
    var_37 = 'All tests passed for Schema constructor.'
    var_38 = print(var_37)



# Parsed testcases at query #3
#--------------------------



def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = False
    var_2 = 'test_field'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = 'some_value'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'some_value'
    var_6 = None
    var_7 = var_3.validate(var_6)
    var_8 = True
    var_9 = module_0.Reference(var_7, var_0)
    var_10 = None
    var_11 = var_9.validate(var_10)
    assert var_11 is None



# Parsed testcases at query #4
#--------------------------



def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'test'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = var_2.validate(var_1)
    assert var_3 == 'test'
    var_4 = None
    var_5 = var_2.validate(var_4)
    assert var_5 is None
    var_6 = None
    var_7 = var_2.validate(var_6)



# Parsed testcases at query #5
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1


def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_1.Schema(var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None
    var_7 = False
    var_8 = module_1.Schema(var_2)
    var_9 = None
    var_10 = var_8.validate(var_9)
    var_11 = 'not a dict'
    var_12 = var_8.validate(var_11)
    var_13 = 'value'
    var_14 = {var_3: var_13}
    var_15 = var_8.validate(var_14)
    var_16 = module_0.Field()
    var_17 = {var_15: var_16}
    var_18 = module_1.Schema(var_17)
    var_19 = {}
    var_20 = var_18.validate(var_19)
    var_21 = module_0.Field(allow_null=var_7)
    var_22 = {var_20: var_21}
    var_23 = module_1.Schema(var_22)
    var_24 = {var_20: var_5}
    var_25 = var_23.validate(var_24)
    var_26 = module_0.Field()
    var_27 = {var_25: var_26}
    var_28 = module_1.Schema(var_27)
    var_29 = 'John'
    var_30 = {var_25: var_29}
    var_31 = var_28.validate(var_30)
    var_32 = module_0.Field(default=var_29)
    var_33 = {var_25: var_32}
    var_34 = module_1.Schema(var_33)
    var_35 = {}
    var_36 = var_34.validate(var_35)
    var_37 = module_0.Field(read_only=var_3)
    var_38 = {var_25: var_37}
    var_39 = module_1.Schema(var_38)
    var_40 = {var_25: var_29}
    var_41 = var_39.validate(var_40)
    var_42 = 'age'
    var_43 = module_0.Field()
    var_44 = {var_42: var_43}
    var_45 = module_1.Schema(var_44)
    var_46 = 'person'
    var_47 = {var_46: var_45}
    var_48 = module_1.Schema(var_47)
    var_49 = 25
    var_50 = {var_42: var_49}
    var_51 = {var_46: var_50}
    var_52 = var_48.validate(var_51)
    var_53 = {var_42: var_5}
    var_54 = {var_46: var_53}
    var_55 = var_48.validate(var_54)
    var_56 = module_0.Field(allow_null=var_7)
    var_57 = module_0.Field(allow_null=var_7)
    var_58 = {var_55: var_56, var_42: var_57}
    var_59 = module_1.Schema(var_58)
    var_60 = {var_55: var_5, var_42: var_5}
    var_61 = var_59.validate(var_60)
    var_62 = {var_61: var_29}
    var_63 = var_59.validate(var_60)
    var_64 = module_0.Field()
    var_65 = {var_61: var_64}
    var_66 = module_1.Schema(var_65)
    var_67 = {var_61: var_29, var_42: var_49}
    var_68 = var_66.validate(var_67)
    var_69 = module_1.Schema(var_65)
    var_70 = 'valid'
    var_71 = {var_61: var_70}
    var_72 = var_69.validate(var_71)
    var_73 = 'invalid'
    var_74 = {var_61: var_73}
    var_75 = var_69.validate(var_74)
    var_76 = 5
    var_77 = module_0.Field()
    var_78 = {var_75: var_77}
    var_79 = module_1.Schema(var_78)
    var_80 = {var_75: var_29}
    var_81 = var_79.validate(var_80)
    var_82 = 'Jonathan'
    var_83 = {var_75: var_82}
    var_84 = var_79.validate(var_83)
    var_85 = module_0.Field()
    var_86 = {var_84: var_85}
    var_87 = module_1.Schema(var_86)
    var_88 = ''
    var_89 = {var_84: var_88}
    var_90 = var_87.validate(var_89)
    var_91 = module_0.Field()
    var_92 = {var_90: var_91}
    var_93 = module_1.Schema(var_92)
    var_94 = {var_90: var_88}
    var_95 = var_93.validate(var_94)
    var_96 = '^[A-Za-z]+$'
    var_97 = module_0.Field()
    var_98 = {var_90: var_97}
    var_99 = module_1.Schema(var_98)
    var_100 = {var_90: var_29}
    var_101 = var_99.validate(var_100)
    var_102 = 'John123'
    var_103 = {var_90: var_102}
    var_104 = var_99.validate(var_103)
    var_105 = 'null'
    var_106 = 'Name cannot be null'
    var_107 = {var_105: var_106}
    var_108 = module_0.Field(allow_null=var_7)
    var_109 = {var_104: var_108}
    var_110 = module_1.Schema(var_109)
    var_111 = {var_104: var_5}
    var_112 = var_110.validate(var_111)
    var_113 = 'email'
    var_114 = module_0.Field(allow_null=var_7)
    var_115 = 150
    var_116 = module_0.Field()
    var_117 = module_0.Field()
    var_118 = {var_112: var_114, var_42: var_116, var_113: var_117}
    var_119 = module_1.Schema(var_118)
    var_120 = 'john@example.com'
    var_121 = {var_112: var_29, var_42: var_49, var_113: var_120}
    var_122 = var_119.validate(var_121)
    var_123 = 200
    var_124 = {var_112: var_5, var_42: var_123, var_113: var_73}
    var_125 = var_119.validate(var_124)
    var_126 = 'tags'
    var_127 = 'array'
    var_128 = 'string'
    var_129 = module_0.Field()
    var_130 = module_0.Field()
    var_131 = {var_126: var_130}
    var_132 = module_1.Schema(var_131)
    var_133 = 'data'
    var_134 = {var_133: var_132}
    var_135 = module_1.Schema(var_134)
    var_136 = 'tag1'
    var_137 = 'tag2'
    var_138 = [var_136, var_137]
    var_139 = {var_126: var_138}
    var_140 = {var_133: var_139}
    var_141 = var_135.validate(var_140)
    var_142 = 123
    var_143 = [var_136, var_142]
    var_144 = {var_126: var_143}
    var_145 = {var_133: var_144}
    var_146 = var_135.validate(var_145)
    var_147 = module_1.Schema(var_134)
    var_148 = {}
    var_149 = var_147.validate(var_148)
    var_150 = 'default'
    var_151 = lambda : var_150
    var_152 = module_0.Field(default=var_151)
    var_153 = {var_146: var_152}
    var_154 = module_1.Schema(var_153)
    var_155 = {}
    var_156 = var_154.validate(var_155)
    var_157 = module_0.Field(default=var_150)
    var_158 = {var_146: var_157}
    var_159 = module_1.Schema(var_158)
    var_160 = 'provided'
    var_161 = {var_146: var_160}
    var_162 = var_159.validate(var_161)
    var_163 = module_0.Field(default=var_150, allow_null=var_3)
    var_164 = {var_146: var_163}
    var_165 = module_1.Schema(var_164)
    var_166 = {var_146: var_5}
    var_167 = var_165.validate(var_166)
    var_168 = module_0.Field(default=var_150)
    var_169 = {var_146: var_168}
    var_170 = module_1.Schema(var_169)
    var_171 = {}



# Parsed testcases at query #6
#--------------------------


import typesystem.fields as module_1
import typesystem.schemas as module_0


def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = {}
    var_6 = False
    var_7 = module_0.Schema(var_5)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = {}
    var_11 = module_0.Schema(var_10)
    var_12 = 'not a dict'
    var_13 = var_11.validate(var_12)
    var_14 = {}
    var_15 = module_0.Schema(var_14)
    var_16 = 'value'
    var_17 = {var_12: var_16}
    var_18 = var_15.validate(var_17)
    var_19 = 'required_field'
    var_20 = module_1.Field()
    var_21 = {var_19: var_20}
    var_22 = module_0.Schema(var_21)
    var_23 = {}
    var_24 = var_22.validate(var_23)
    var_25 = 'field_with_default'
    var_26 = 'default_value'
    var_27 = module_1.Field(default=var_26)
    var_28 = {var_25: var_27}
    var_29 = module_0.Schema(var_28)
    var_30 = {}
    var_31 = var_29.validate(var_30)
    var_32 = 'read_only_field'
    var_33 = module_1.Field(read_only=var_24)
    var_34 = {var_32: var_33}
    var_35 = module_0.Schema(var_34)
    var_36 = {var_32: var_16}
    var_37 = var_35.validate(var_36)
    var_38 = 'field'
    var_39 = module_1.Field()
    var_40 = {var_38: var_39}
    var_41 = module_0.Schema(var_40)
    var_42 = 'invalid_value'
    var_43 = {var_38: var_42}
    var_44 = var_41.validate(var_43)
    var_45 = 'field1'
    var_46 = 'field2'
    var_47 = module_1.Field()
    var_48 = module_1.Field()
    var_49 = {var_45: var_47, var_46: var_48}
    var_50 = module_0.Schema(var_49)
    var_51 = 'value1'
    var_52 = 'value2'
    var_53 = {var_45: var_51, var_46: var_52}
    var_54 = var_50.validate(var_53)



# Parsed testcases at query #7
#--------------------------



def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = False
    var_2 = 'test'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = None
    var_5 = var_3.validate(var_4)
    var_6 = True
    var_7 = module_0.Reference(var_5, var_0)
    var_8 = None
    var_9 = var_7.validate(var_8)
    assert var_9 is None



# Parsed testcases at query #8
#--------------------------



def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = {}
    var_6 = False
    var_7 = module_0.Schema(var_5)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = {}
    var_11 = module_0.Schema(var_10)
    var_12 = 'not a dict'
    var_13 = var_11.validate(var_12)
    var_14 = {}
    var_15 = module_0.Schema(var_14)
    var_16 = 1
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = var_15.validate(var_18)
    var_20 = 'required_field'
    var_21 = module_1.Field()
    var_22 = {var_20: var_21}
    var_23 = module_0.Schema(var_22)
    var_24 = {}
    var_25 = var_23.validate(var_24)
    var_26 = 'field_with_default'
    var_27 = 'default_value'
    var_28 = module_1.Field(default=var_27)
    var_29 = {var_26: var_28}
    var_30 = module_0.Schema(var_29)
    var_31 = {}
    var_32 = var_30.validate(var_31)
    var_33 = 'field'
    var_34 = module_1.Field(allow_null=var_6)
    var_35 = {var_33: var_34}
    var_36 = module_0.Schema(var_35)
    var_37 = 'field'
    var_38 = None
    var_39 = {var_37: var_38}
    var_40 = var_36.validate(var_39)
    var_41 = module_1.Field()
    var_42 = {var_33: var_41}
    var_43 = module_0.Schema(var_42)
    var_44 = 'value'
    var_45 = {var_33: var_44}
    var_46 = var_43.validate(var_45)



# Parsed testcases at query #9
#--------------------------



def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = {}
    var_6 = False
    var_7 = module_0.Schema(var_5)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = {}
    var_11 = module_0.Schema(var_10)
    var_12 = 'not a dict'
    var_13 = var_11.validate(var_12)
    var_14 = {}
    var_15 = module_0.Schema(var_14)
    var_16 = 1
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = var_15.validate(var_18)
    var_20 = 'required_field'
    var_21 = module_1.Field()
    var_22 = {var_20: var_21}
    var_23 = module_0.Schema(var_22)
    var_24 = {}
    var_25 = var_23.validate(var_24)
    var_26 = 'field_with_default'
    var_27 = 'default_value'
    var_28 = module_1.Field(default=var_27)
    var_29 = {var_26: var_28}
    var_30 = module_0.Schema(var_29)
    var_31 = {}
    var_32 = var_30.validate(var_31)
    var_33 = 'field'
    var_34 = module_1.Field(allow_null=var_6)
    var_35 = {var_33: var_34}
    var_36 = module_0.Schema(var_35)
    var_37 = 'field'
    var_38 = None
    var_39 = {var_37: var_38}
    var_40 = var_36.validate(var_39)
    var_41 = module_1.Field()
    var_42 = {var_33: var_41}
    var_43 = module_0.Schema(var_42)
    var_44 = 'value'
    var_45 = {var_33: var_44}
    var_46 = var_43.validate(var_45)



# Parsed testcases at query #10
#--------------------------



def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = 'test'
    var_5 = module_0.Reference(var_4, var_0)
    var_6 = {var_1: var_4}
    var_7 = var_5.validate(var_6)
    var_8 = None
    var_9 = var_5.validate(var_8)



# Parsed testcases at query #11
#--------------------------



def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = {}
    var_6 = False
    var_7 = module_0.Schema(var_5)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = {}
    var_11 = module_0.Schema(var_10)
    var_12 = 'not a dict'
    var_13 = var_11.validate(var_12)
    var_14 = {}
    var_15 = module_0.Schema(var_14)
    var_16 = 1
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = var_15.validate(var_18)
    var_20 = 'required_field'
    var_21 = module_1.Field()
    var_22 = {var_20: var_21}
    var_23 = module_0.Schema(var_22)
    var_24 = {}
    var_25 = var_23.validate(var_24)
    var_26 = 'field_with_default'
    var_27 = 'default'
    var_28 = module_1.Field(default=var_27)
    var_29 = {var_26: var_28}
    var_30 = module_0.Schema(var_29)
    var_31 = {}
    var_32 = var_30.validate(var_31)
    var_33 = 'field'
    var_34 = module_1.Field(allow_null=var_6)
    var_35 = {var_33: var_34}
    var_36 = module_0.Schema(var_35)
    var_37 = 'field'
    var_38 = None
    var_39 = {var_37: var_38}
    var_40 = var_36.validate(var_39)
    var_41 = module_1.Field()
    var_42 = {var_33: var_41}
    var_43 = module_0.Schema(var_42)
    var_44 = 'value'
    var_45 = {var_33: var_44}
    var_46 = var_43.validate(var_45)



# Parsed testcases at query #12
#--------------------------



def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'test'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = var_2.validate(var_1)
    assert var_5 == 'test'



# Parsed testcases at query #13
#--------------------------



def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = 'test'
    var_5 = module_0.Reference(var_4, var_0)
    var_6 = {var_1: var_4}
    var_7 = var_5.validate(var_6)
    var_8 = None
    var_9 = var_5.validate(var_8)
    assert var_9 is None
    var_10 = None
    var_11 = var_5.validate(var_10)
    var_12 = 'invalid'
    var_13 = var_5.validate(var_12)



# Parsed testcases at query #14
#--------------------------



def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'test'
    var_2 = True
    var_3 = module_0.Reference(var_1, var_0)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None
    var_6 = False
    var_7 = module_0.Reference(var_1, var_0)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = module_0.Reference(var_8, var_0)
    var_11 = 'value'
    var_12 = var_10.validate(var_11)
    assert var_12 == 'value'
    var_13 = module_0.Reference(var_8, var_0)
    var_14 = None
    var_15 = var_13.validate(var_14)
    var_16 = module_0.Reference(var_14, var_0)
    var_17 = 'invalid'
    var_18 = var_16.validate(var_17)
    var_19 = module_0.Reference(var_17, var_0)
    var_20 = 123
    var_21 = var_19.validate(var_20)
    var_22 = module_0.Reference(var_20, var_0)
    var_23 = var_22.validate(var_4)
    assert var_23 is None
    var_24 = module_0.Reference(var_20, var_0)
    var_25 = var_24.validate(var_11)
    assert var_25 == 'value'
    var_26 = module_0.Reference(var_20, var_0)
    var_27 = 123
    var_28 = var_26.validate(var_27)
    assert var_28 == 123
    var_29 = module_0.Reference(var_20, var_0)
    var_30 = 2
    var_31 = 3
    var_32 = [var_21, var_30, var_31]
    var_33 = var_29.validate(var_32)
    var_34 = module_0.Reference(var_20, var_0)
    var_35 = 'key'
    var_36 = {var_35: var_11}
    var_37 = var_34.validate(var_36)
    var_38 = module_0.Reference(var_20, var_0)
    var_39 = module_0.Reference(var_20, var_0)
    var_40 = var_39.validate(var_21)
    assert var_40 is True
    var_41 = module_0.Reference(var_20, var_0)
    var_42 = 3.14
    var_43 = var_41.validate(var_42)
    var_44 = module_0.Reference(var_20, var_0)
    var_45 = 42
    var_46 = var_44.validate(var_45)
    assert var_46 == 42
    var_47 = module_0.Reference(var_20, var_0)
    var_48 = 'hello'
    var_49 = var_47.validate(var_48)
    assert var_49 == 'hello'
    var_50 = module_0.Reference(var_20, var_0)
    var_51 = (var_21, var_30, var_31)
    var_52 = var_50.validate(var_51)
    var_53 = module_0.Reference(var_20, var_0)
    var_54 = {var_21, var_30, var_31}
    var_55 = var_53.validate(var_54)
    var_56 = module_0.Reference(var_20, var_0)
    var_57 = [var_21, var_30, var_31]
    var_58 = frozenset(var_57)
    var_59 = var_56.validate(var_58)
    var_60 = [var_21, var_30, var_31]
    var_61 = frozenset(var_60)
    var_62 = module_0.Reference(var_20, var_0)
    var_63 = b'hello'
    var_64 = var_62.validate(var_63)
    assert var_64 == b'hello'
    var_65 = module_0.Reference(var_20, var_0)
    var_66 = bytearray(var_63)
    var_67 = var_65.validate(var_66)
    var_68 = bytearray(var_63)
    var_69 = module_0.Reference(var_20, var_0)
    var_70 = memoryview(var_63)
    var_71 = var_69.validate(var_70)
    var_72 = module_0.Reference(var_20, var_0)
    var_73 = module_0.Reference(var_20, var_0)
    var_74 = 5
    var_75 = range(var_74)
    var_76 = var_73.validate(var_75)
    var_77 = range(var_74)
    var_78 = module_0.Reference(var_20, var_0)
    var_79 = 10
    var_80 = slice(var_21, var_79, var_30)
    var_81 = var_78.validate(var_80)
    var_82 = slice(var_21, var_79, var_30)
    var_83 = module_0.Reference(var_20, var_0)
    var_84 = module_0.Reference(var_20, var_0)
    var_85 = lambda x: x
    var_86 = var_84.validate(var_85)
    var_87 = callable(var_86)
    var_88 = module_0.Reference(var_20, var_0)
    var_89 = module_0.Reference(var_20, var_0)
    var_90 = module_0.Reference(var_20, var_0)
    var_91 = range(var_74)
    var_92 = module_0.Reference(var_20, var_0)
    var_93 = module_0.Reference(var_20, var_0)
    var_94 = module_0.Reference(var_20, var_0)



# Parsed testcases at query #15
#--------------------------



def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = {}
    var_6 = False
    var_7 = module_0.Schema(var_5)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = {}
    var_11 = module_0.Schema(var_10)
    var_12 = 'not a dict'
    var_13 = var_11.validate(var_12)
    var_14 = {}
    var_15 = module_0.Schema(var_14)
    var_16 = 1
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = var_15.validate(var_18)
    var_20 = 'field1'
    var_21 = module_1.Field()
    var_22 = {var_20: var_21}
    var_23 = module_0.Schema(var_22)
    var_24 = {}
    var_25 = var_23.validate(var_24)
    var_26 = module_1.Field()
    var_27 = {var_20: var_26}
    var_28 = module_0.Schema(var_27)
    var_29 = 'value'
    var_30 = {var_20: var_29}
    var_31 = var_28.validate(var_30)
    var_32 = module_1.Field(read_only=var_24)
    var_33 = {var_20: var_32}
    var_34 = module_0.Schema(var_33)
    var_35 = {var_20: var_29}
    var_36 = var_34.validate(var_35)
    var_37 = 'default'
    var_38 = module_1.Field(default=var_37)
    var_39 = {var_20: var_38}
    var_40 = module_0.Schema(var_39)
    var_41 = {}
    var_42 = var_40.validate(var_41)
    var_43 = 'valid'
    var_44 = lambda x: x == var_43
    var_45 = [var_44]
    var_46 = module_1.Field()
    var_47 = {var_20: var_46}
    var_48 = module_0.Schema(var_47)
    var_49 = 'field1'
    var_50 = 'invalid'
    var_51 = {var_49: var_50}
    var_52 = var_48.validate(var_51)
    var_53 = 'field2'
    var_54 = module_1.Field()
    var_55 = lambda x: x == var_43
    var_56 = [var_55]
    var_57 = module_1.Field()
    var_58 = {var_20: var_54, var_53: var_57}
    var_59 = module_0.Schema(var_58)
    var_60 = 'field2'
    var_61 = 'invalid'
    var_62 = {var_60: var_61}
    var_63 = var_59.validate(var_62)



# Parsed testcases at query #16
#--------------------------



def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = False
    var_2 = 'test_field'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = 'some_value'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'some_value'
    var_6 = None
    var_7 = var_3.validate(var_6)
    var_8 = None
    var_9 = var_3.validate(var_8)
    assert var_9 is None



# Parsed testcases at query #17
#--------------------------



def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = {}
    var_6 = False
    var_7 = module_0.Schema(var_5)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = {}
    var_11 = module_0.Schema(var_10)
    var_12 = 'not a dict'
    var_13 = var_11.validate(var_12)
    var_14 = {}
    var_15 = module_0.Schema(var_14)
    var_16 = 1
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = var_15.validate(var_18)
    var_20 = 'required_field'
    var_21 = module_1.Field()
    var_22 = {var_20: var_21}
    var_23 = module_0.Schema(var_22)
    var_24 = {}
    var_25 = var_23.validate(var_24)
    var_26 = 'field_with_default'
    var_27 = 'default_value'
    var_28 = module_1.Field(default=var_27)
    var_29 = {var_26: var_28}
    var_30 = module_0.Schema(var_29)
    var_31 = {}
    var_32 = var_30.validate(var_31)
    var_33 = 'field'
    var_34 = module_1.Field(allow_null=var_6)
    var_35 = {var_33: var_34}
    var_36 = module_0.Schema(var_35)
    var_37 = 'field'
    var_38 = None
    var_39 = {var_37: var_38}
    var_40 = var_36.validate(var_39)
    var_41 = module_1.Field()
    var_42 = {var_33: var_41}
    var_43 = module_0.Schema(var_42)
    var_44 = 'value'
    var_45 = {var_33: var_44}
    var_46 = var_43.validate(var_45)



# Parsed testcases at query #18
#--------------------------



def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'test'
    var_2 = True
    var_3 = module_0.Reference(var_1, var_0)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None
    var_6 = False
    var_7 = module_0.Reference(var_1, var_0)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = module_0.Reference(var_8, var_0)
    var_11 = 'value'
    var_12 = var_10.validate(var_11)
    assert var_12 == 'value'
    var_13 = module_0.Reference(var_8, var_0)
    var_14 = None
    var_15 = var_13.validate(var_14)



# Parsed testcases at query #19
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1


def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_1.Schema(var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None
    var_7 = False
    var_8 = module_1.Schema(var_2)
    var_9 = None
    var_10 = var_8.validate(var_9)
    var_11 = 'not a dict'
    var_12 = var_8.validate(var_11)
    var_13 = 'one'
    var_14 = {var_3: var_13}
    var_15 = var_8.validate(var_14)
    var_16 = module_0.Field()
    var_17 = {var_15: var_16}
    var_18 = module_1.Schema(var_17)
    var_19 = {}
    var_20 = var_18.validate(var_19)
    var_21 = 'default_name'
    var_22 = module_0.Field(default=var_21)
    var_23 = {var_19: var_22}
    var_24 = module_1.Schema(var_23)
    var_25 = {}
    var_26 = var_24.validate(var_25)
    var_27 = 5
    var_28 = module_0.Field()
    var_29 = {var_19: var_28}
    var_30 = module_1.Schema(var_29)
    var_31 = 'name'
    var_32 = 'longname'
    var_33 = {var_31: var_32}
    var_34 = var_30.validate(var_33)
    var_35 = 10
    var_36 = module_0.Field()
    var_37 = {var_31: var_36}
    var_38 = module_1.Schema(var_37)
    var_39 = 'short'
    var_40 = {var_31: var_39}
    var_41 = var_38.validate(var_40)
    var_42 = 'inner'
    var_43 = module_0.Field()
    var_44 = {var_42: var_43}
    var_45 = module_1.Schema(var_44)
    var_46 = 'nested'
    var_47 = {var_46: var_45}
    var_48 = module_1.Schema(var_47)
    var_49 = {var_42: var_39}
    var_50 = {var_46: var_49}
    var_51 = var_48.validate(var_50)
    var_52 = 'nested'
    var_53 = 'inner'
    var_54 = 'toolong'
    var_55 = {var_53: var_54}
    var_56 = {var_52: var_55}
    var_57 = var_48.validate(var_56)
    var_58 = module_0.Field(read_only=var_54)
    var_59 = {var_52: var_58}
    var_60 = module_1.Schema(var_59)
    var_61 = 'ignored'
    var_62 = {var_52: var_61}
    var_63 = var_60.validate(var_62)
    var_64 = 'age'
    var_65 = module_0.Field()
    var_66 = module_0.Field()
    var_67 = {var_52: var_65, var_64: var_66}
    var_68 = module_1.Schema(var_67)
    var_69 = 'age'
    var_70 = -1
    var_71 = {var_69: var_70}
    var_72 = var_68.validate(var_71)
    var_73 = 'test'
    var_74 = {var_69: var_73}
    var_75 = var_68.validate(var_14)
    var_76 = 'custom'
    var_77 = module_1.Schema(var_67)
    var_78 = 'valid'
    var_79 = {var_76: var_78}
    var_80 = var_77.validate(var_79)
    var_81 = 'custom'
    var_82 = 'invalid'
    var_83 = {var_81: var_82}
    var_84 = var_77.validate(var_83)
    var_85 = module_0.Field(allow_null=var_83)
    var_86 = {var_81: var_85}
    var_87 = module_1.Schema(var_86)
    var_88 = {var_81: var_84}
    var_89 = var_87.validate(var_88)
    var_90 = module_0.Field(allow_null=var_57)
    var_91 = {var_81: var_90}
    var_92 = module_1.Schema(var_91)
    var_93 = 'name'
    var_94 = None
    var_95 = {var_93: var_94}
    var_96 = var_92.validate(var_95)
    var_97 = 'inner_name'
    var_98 = module_0.Field()
    var_99 = {var_97: var_98}
    var_100 = module_1.Schema(var_99)
    var_101 = 'outer'
    var_102 = {var_101: var_100}
    var_103 = module_1.Schema(var_102)
    var_104 = {var_97: var_73}
    var_105 = {var_101: var_104}
    var_106 = var_103.validate(var_105)
    var_107 = 'outer'
    var_108 = {}
    var_109 = {var_107: var_108}
    var_110 = var_103.validate(var_109)
    var_111 = module_0.Field()
    var_112 = {var_107: var_111}
    var_113 = module_1.Schema(var_112)
    var_114 = 'extra'
    var_115 = {var_107: var_73, var_114: var_61}
    var_116 = var_113.validate(var_115)
    var_117 = module_1.Schema(var_112)
    var_118 = {}
    var_119 = var_117.validate(var_118)
    var_120 = module_1.Schema(var_112)
    var_121 = {}
    var_122 = var_120.validate(var_121)
    var_123 = 'All tests passed!'
    var_124 = print(var_123)



# Parsed testcases at query #20
#--------------------------


import typesystem.schemas as module_0


def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'test'
    var_2 = True
    var_3 = module_0.Reference(var_1, var_0)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None
    var_6 = False
    var_7 = module_0.Reference(var_1, var_0)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = module_0.Reference(var_8, var_0)
    var_11 = var_10.validate(var_8)
    assert var_11 == 'test'



# Parsed testcases at query #21
#--------------------------



def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = False
    var_2 = 'test_field'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = 'some_value'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'some_value'
    var_6 = None
    var_7 = var_3.validate(var_6)
    var_8 = None
    var_9 = var_3.validate(var_8)
    assert var_9 is None



# Parsed testcases at query #22
#--------------------------


import typesystem.fields as module_1


def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = {}
    var_6 = False
    var_7 = module_0.Schema(var_5)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = {}
    var_11 = module_0.Schema(var_10)
    var_12 = 'not a dict'
    var_13 = var_11.validate(var_12)
    var_14 = {}
    var_15 = module_0.Schema(var_14)
    var_16 = 1
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = var_15.validate(var_18)
    var_20 = 'required_field'
    var_21 = module_1.Field()
    var_22 = {var_20: var_21}
    var_23 = module_0.Schema(var_22)
    var_24 = {}
    var_25 = var_23.validate(var_24)
    var_26 = 'field_with_default'
    var_27 = 'default_value'
    var_28 = module_1.Field(default=var_27)
    var_29 = {var_26: var_28}
    var_30 = module_0.Schema(var_29)
    var_31 = {}
    var_32 = var_30.validate(var_31)
    var_33 = 'field'
    var_34 = module_1.Field(allow_null=var_6)
    var_35 = {var_33: var_34}
    var_36 = module_0.Schema(var_35)
    var_37 = 'field'
    var_38 = None
    var_39 = {var_37: var_38}
    var_40 = var_36.validate(var_39)
    var_41 = module_1.Field()
    var_42 = {var_33: var_41}
    var_43 = module_0.Schema(var_42)
    var_44 = 'value'
    var_45 = {var_33: var_44}
    var_46 = var_43.validate(var_45)



# Parsed testcases at query #23
#--------------------------



def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = 'test'
    var_5 = module_0.Reference(var_4, var_0)
    var_6 = {var_1: var_4}
    var_7 = var_5.validate(var_6)
    var_8 = None
    var_9 = var_5.validate(var_8)
    assert var_9 is None
    var_10 = 'invalid'
    var_11 = var_5.validate(var_10)
    var_12 = 'name'
    var_13 = 123
    var_14 = {var_12: var_13}
    var_15 = var_5.validate(var_14)



# Parsed testcases at query #24
#--------------------------



def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'test'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = 1
    var_6 = var_2.validate(var_5)
    assert var_6 == 1



# Parsed testcases at query #25
#--------------------------



def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = {}
    var_6 = False
    var_7 = module_0.Schema(var_5)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = {}
    var_11 = module_0.Schema(var_10)
    var_12 = 'not a dict'
    var_13 = var_11.validate(var_12)
    var_14 = {}
    var_15 = module_0.Schema(var_14)
    var_16 = 1
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = var_15.validate(var_18)
    var_20 = 'required_field'
    var_21 = module_1.Field()
    var_22 = {var_20: var_21}
    var_23 = module_0.Schema(var_22)
    var_24 = {}
    var_25 = var_23.validate(var_24)
    var_26 = 'field_with_default'
    var_27 = 'default_value'
    var_28 = module_1.Field(default=var_27)
    var_29 = {var_26: var_28}
    var_30 = module_0.Schema(var_29)
    var_31 = {}
    var_32 = var_30.validate(var_31)
    var_33 = 'field'
    var_34 = module_1.Field(allow_null=var_6)
    var_35 = {var_33: var_34}
    var_36 = module_0.Schema(var_35)
    var_37 = 'field'
    var_38 = None
    var_39 = {var_37: var_38}
    var_40 = var_36.validate(var_39)
    var_41 = module_1.Field()
    var_42 = {var_33: var_41}
    var_43 = module_0.Schema(var_42)
    var_44 = 'value'
    var_45 = {var_33: var_44}
    var_46 = var_43.validate(var_45)



# Parsed testcases at query #26
#--------------------------



def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = {}
    var_6 = False
    var_7 = module_0.Schema(var_5)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = {}
    var_11 = module_0.Schema(var_10)
    var_12 = 'not a dict'
    var_13 = var_11.validate(var_12)
    var_14 = {}
    var_15 = module_0.Schema(var_14)
    var_16 = 1
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = var_15.validate(var_18)
    var_20 = 'required_field'
    var_21 = module_1.Field()
    var_22 = {var_20: var_21}
    var_23 = module_0.Schema(var_22)
    var_24 = {}
    var_25 = var_23.validate(var_24)
    var_26 = 'field_with_default'
    var_27 = 'default_value'
    var_28 = module_1.Field(default=var_27)
    var_29 = {var_26: var_28}
    var_30 = module_0.Schema(var_29)
    var_31 = {}
    var_32 = var_30.validate(var_31)
    var_33 = 'field'
    var_34 = module_1.Field(allow_null=var_19)
    var_35 = {var_33: var_34}
    var_36 = module_0.Schema(var_35)
    var_37 = 'field'
    var_38 = None
    var_39 = {var_37: var_38}
    var_40 = var_36.validate(var_39)
    var_41 = 'field1'
    var_42 = 'field2'
    var_43 = module_1.Field()
    var_44 = module_1.Field(allow_null=var_37)
    var_45 = {var_41: var_43, var_42: var_44}
    var_46 = module_0.Schema(var_45)
    var_47 = 'value1'
    var_48 = {var_41: var_47, var_42: var_38}
    var_49 = var_46.validate(var_48)
    var_50 = 'read_only_field'
    var_51 = module_1.Field(read_only=var_37)
    var_52 = {var_50: var_51}
    var_53 = module_0.Schema(var_52)
    var_54 = 'value'
    var_55 = {var_50: var_54}
    var_56 = var_53.validate(var_55)
    var_57 = 'nested_field'
    var_58 = module_1.Field()
    var_59 = {var_57: var_58}
    var_60 = module_0.Schema(var_59)
    var_61 = 'nested'
    var_62 = {var_61: var_60}
    var_63 = module_0.Schema(var_62)
    var_64 = {var_57: var_54}
    var_65 = {var_61: var_64}
    var_66 = var_63.validate(var_65)
    var_67 = module_1.Field(allow_null=var_40)
    var_68 = {var_57: var_67}
    var_69 = module_0.Schema(var_68)
    var_70 = {var_61: var_69}
    var_71 = module_0.Schema(var_70)
    var_72 = 'nested'
    var_73 = 'nested_field'
    var_74 = None
    var_75 = {var_73: var_74}
    var_76 = {var_72: var_75}
    var_77 = var_71.validate(var_76)
    var_78 = module_1.Field(allow_null=var_75)
    var_79 = module_1.Field(allow_null=var_75)
    var_80 = {var_41: var_78, var_42: var_79}
    var_81 = module_0.Schema(var_80)
    var_82 = 'field1'
    var_83 = 'field2'
    var_84 = None
    var_85 = {var_82: var_84, var_83: var_84}
    var_86 = var_81.validate(var_85)
    var_87 = module_1.Field()
    var_88 = {var_33: var_87}
    var_89 = module_0.Schema(var_88)
    var_90 = {var_33: var_54}
    var_91 = 'custom_field'
    var_92 = module_0.Schema(var_88)
    var_93 = 'custom_field'
    var_94 = 'invalid'
    var_95 = {var_93: var_94}
    var_96 = var_92.validate(var_95)
    var_97 = 'multi_field'
    var_98 = module_0.Schema(var_88)
    var_99 = 'multi_field'
    var_100 = 123
    var_101 = {var_99: var_100}
    var_102 = var_98.validate(var_101)
    var_103 = 'required_nested'
    var_104 = module_1.Field()
    var_105 = {var_103: var_104}
    var_106 = module_0.Schema(var_105)
    var_107 = {var_61: var_106}
    var_108 = module_0.Schema(var_107)
    var_109 = 'nested'
    var_110 = {}
    var_111 = {var_109: var_110}
    var_112 = var_108.validate(var_111)
    var_113 = 'nested_with_default'
    var_114 = 'nested_default'
    var_115 = module_1.Field(default=var_114)
    var_116 = {var_113: var_115}
    var_117 = module_0.Schema(var_116)
    var_118 = {var_61: var_117}
    var_119 = module_0.Schema(var_118)
    var_120 = {}
    var_121 = {var_61: var_120}
    var_122 = var_119.validate(var_121)
    var_123 = 'inner_field'
    var_124 = module_1.Field()
    var_125 = {var_123: var_124}
    var_126 = module_0.Schema(var_125)
    var_127 = 'middle_field'
    var_128 = {var_127: var_126}
    var_129 = module_0.Schema(var_128)
    var_130 = 'outer_field'
    var_131 = {var_130: var_129}
    var_132 = module_0.Schema(var_131)
    var_133 = {var_123: var_54}
    var_134 = {var_127: var_133}
    var_135 = {var_130: var_134}
    var_136 = var_132.validate(var_135)
    var_137 = 'default_field'
    var_138 = module_0.Schema(var_118)
    var_139 = {}
    var_140 = var_138.validate(var_139)
    var_141 = 'no_default_field'
    var_142 = module_0.Schema(var_118)
    var_143 = {}
    var_144 = var_142.validate(var_143)
    var_145 = 'has_default_field'
    var_146 = module_0.Schema(var_118)
    var_147 = {}
    var_148 = var_146.validate(var_147)
    var_149 = 'read_only_default'
    var_150 = module_1.Field(default=var_149, read_only=var_143)
    var_151 = {var_149: var_150}
    var_152 = module_0.Schema(var_151)
    var_153 = {}
    var_154 = var_152.validate(var_153)
    var_155 = module_1.Field(read_only=var_143)
    var_156 = {var_50: var_155}
    var_157 = module_0.Schema(var_156)
    var_158 = {var_50: var_54}
    var_159 = var_157.validate(var_158)
    var_160 = 'nullable_field'
    var_161 = module_1.Field(allow_null=var_143)
    var_162 = {var_160: var_161}
    var_163 = module_0.Schema(var_162)
    var_164 = {var_160: var_144}
    var_165 = var_163.validate(var_164)
    var_166 = module_1.Field(allow_null=var_143)
    var_167 = {var_160: var_166}
    var_168 = module_0.Schema(var_167)
    var_169 = {var_160: var_54}
    var_170 = var_168.validate(var_169)
    var_171 = 'non_nullable_field'
    var_172 = module_1.Field(allow_null=var_112)
    var_173 = {var_171: var_172}
    var_174 = module_0.Schema(var_173)
    var_175 = 'non_nullable_field'
    var_176 = None
    var_177 = {var_175: var_176}
    var_178 = var_174.validate(var_177)
    var_179 = module_1.Field(allow_null=var_178)
    var_180 = {var_171: var_179}
    var_181 = module_0.Schema(var_180)
    var_182 = {var_171: var_54}
    var_183 = var_181.validate(var_182)
    var_184 = 'custom_error'
    var_185 = 'Custom error message'
    var_186 = {var_184: var_185}
    var_187 = var_180



# Parsed testcases at query #27
#--------------------------



def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'test'
    var_2 = True
    var_3 = module_0.Reference(var_1, var_0)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None
    var_6 = False
    var_7 = module_0.Reference(var_1, var_0)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = module_0.Reference(var_8, var_0)
    var_11 = var_10.validate(var_8)
    assert var_11 == 'test'



# Parsed testcases at query #28
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1


def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_1.Schema(var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None
    var_7 = module_0.Field()
    var_8 = {var_0: var_7}
    var_9 = False
    var_10 = module_1.Schema(var_8)
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = module_0.Field()
    var_14 = {var_11: var_13}
    var_15 = module_1.Schema(var_14)
    var_16 = 'not a dict'
    var_17 = var_15.validate(var_16)
    var_18 = module_0.Field()
    var_19 = {var_16: var_18}
    var_20 = module_1.Schema(var_19)
    var_21 = 1
    var_22 = 'value'
    var_23 = {var_21: var_22}
    var_24 = var_20.validate(var_23)
    var_25 = module_0.Field()
    var_26 = {var_21: var_25}
    var_27 = module_1.Schema(var_26)
    var_28 = {}
    var_29 = var_27.validate(var_28)
    var_30 = 'default'
    var_31 = module_0.Field(default=var_30)
    var_32 = {var_28: var_31}
    var_33 = module_1.Schema(var_32)
    var_34 = {}
    var_35 = var_33.validate(var_34)
    var_36 = module_0.Field(read_only=var_23)
    var_37 = {var_28: var_36}
    var_38 = module_1.Schema(var_37)
    var_39 = 'value'
    var_40 = {var_28: var_39}
    var_41 = var_38.validate(var_40)
    var_42 = module_0.Field()
    var_43 = {var_28: var_42}
    var_44 = module_1.Schema(var_43)
    var_45 = 'name'
    var_46 = None
    var_47 = {var_45: var_46}
    var_48 = var_44.validate(var_47)
    var_49 = module_0.Field()
    var_50 = {var_45: var_49}
    var_51 = module_1.Schema(var_50)
    var_52 = 'John'
    var_53 = {var_45: var_52}
    var_54 = var_51.validate(var_53)



####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.schemas as module_0


def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = False
    var_2 = 'test'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = None
    var_5 = var_3.validate(var_4)



# Parsed testcases at query #2
#--------------------------


import typesystem.fields as module_1


def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = {}
    var_6 = False
    var_7 = module_0.Schema(var_5)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = {}
    var_11 = module_0.Schema(var_10)
    var_12 = 'not a dict'
    var_13 = var_11.validate(var_12)
    var_14 = {}
    var_15 = module_0.Schema(var_14)
    var_16 = 1
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = var_15.validate(var_18)
    var_20 = 'required_field'
    var_21 = module_1.Field()
    var_22 = {var_20: var_21}
    var_23 = module_0.Schema(var_22)
    var_24 = {}
    var_25 = var_23.validate(var_24)
    var_26 = 'field_with_default'
    var_27 = 'default_value'
    var_28 = module_1.Field(default=var_27)
    var_29 = {var_26: var_28}
    var_30 = module_0.Schema(var_29)
    var_31 = {}
    var_32 = var_30.validate(var_31)
    var_33 = 'field'
    var_34 = module_1.Field(allow_null=var_18)
    var_35 = {var_33: var_34}
    var_36 = module_0.Schema(var_35)
    var_37 = 'field'
    var_38 = None
    var_39 = {var_37: var_38}
    var_40 = var_36.validate(var_39)
    var_41 = 'field1'
    var_42 = 'field2'
    var_43 = module_1.Field()
    var_44 = 'default'
    var_45 = module_1.Field(default=var_44)
    var_46 = {var_41: var_43, var_42: var_45}
    var_47 = module_0.Schema(var_46)
    var_48 = 'value1'
    var_49 = {var_41: var_48}
    var_50 = var_47.validate(var_49)



# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1


def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)



# Parsed testcases at query #4
#--------------------------


import typesystem.fields as module_1
import typesystem.schemas as module_0


def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = {}
    var_6 = False
    var_7 = module_0.Schema(var_5)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = {}
    var_11 = module_0.Schema(var_10)
    var_12 = 'not a dict'
    var_13 = var_11.validate(var_12)
    var_14 = {}
    var_15 = module_0.Schema(var_14)
    var_16 = 1
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = var_15.validate(var_18)
    var_20 = 'required_field'
    var_21 = module_1.Field()
    var_22 = {var_20: var_21}
    var_23 = module_0.Schema(var_22)
    var_24 = {}
    var_25 = var_23.validate(var_24)
    var_26 = 'field_with_default'
    var_27 = 'default'
    var_28 = module_1.Field(default=var_27)
    var_29 = {var_26: var_28}
    var_30 = module_0.Schema(var_29)
    var_31 = {}
    var_32 = var_30.validate(var_31)
    var_33 = 'field'
    var_34 = module_1.Field(allow_null=var_6)
    var_35 = {var_33: var_34}
    var_36 = module_0.Schema(var_35)
    var_37 = 'field'
    var_38 = None
    var_39 = {var_37: var_38}
    var_40 = var_36.validate(var_39)
    var_41 = module_1.Field()
    var_42 = {var_33: var_41}
    var_43 = module_0.Schema(var_42)
    var_44 = 'value'
    var_45 = {var_33: var_44}
    var_46 = var_43.validate(var_45)



# Parsed testcases at query #5
#--------------------------



def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = {}
    var_6 = False
    var_7 = module_0.Schema(var_5)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = {}
    var_11 = module_0.Schema(var_10)
    var_12 = 'not a dict'
    var_13 = var_11.validate(var_12)
    var_14 = {}
    var_15 = module_0.Schema(var_14)
    var_16 = 1
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = var_15.validate(var_18)
    var_20 = 'required_field'
    var_21 = module_1.Field()
    var_22 = {var_20: var_21}
    var_23 = module_0.Schema(var_22)
    var_24 = {}
    var_25 = var_23.validate(var_24)
    var_26 = 'field_with_default'
    var_27 = 'default_value'
    var_28 = module_1.Field(default=var_27)
    var_29 = {var_26: var_28}
    var_30 = module_0.Schema(var_29)
    var_31 = {}
    var_32 = var_30.validate(var_31)
    var_33 = 'field'
    var_34 = module_1.Field(allow_null=var_19)
    var_35 = {var_33: var_34}
    var_36 = module_0.Schema(var_35)
    var_37 = 'field'
    var_38 = None
    var_39 = {var_37: var_38}
    var_40 = var_36.validate(var_39)
    var_41 = 'field1'
    var_42 = 'field2'
    var_43 = module_1.Field()
    var_44 = 'default'
    var_45 = module_1.Field(default=var_44)
    var_46 = {var_41: var_43, var_42: var_45}
    var_47 = module_0.Schema(var_46)
    var_48 = 'value1'
    var_49 = {var_41: var_48}
    var_50 = var_47.validate(var_49)
    var_51 = 'read_only_field'
    var_52 = module_1.Field(read_only=var_37)
    var_53 = {var_51: var_52}
    var_54 = module_0.Schema(var_53)
    var_55 = 'value'
    var_56 = {var_51: var_55}
    var_57 = var_54.validate(var_56)
    var_58 = 'nested_field'
    var_59 = module_1.Field(allow_null=var_40)
    var_60 = {var_58: var_59}
    var_61 = module_0.Schema(var_60)
    var_62 = 'nested'
    var_63 = {var_62: var_61}
    var_64 = module_0.Schema(var_63)
    var_65 = 'nested'
    var_66 = 'nested_field'
    var_67 = None
    var_68 = {var_66: var_67}
    var_69 = {var_65: var_68}
    var_70 = var_64.validate(var_69)



# Parsed testcases at query #6
#--------------------------



def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'test'
    var_2 = True
    var_3 = module_0.Reference(var_1, var_0)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None
    var_6 = False
    var_7 = module_0.Reference(var_1, var_0)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = module_0.Reference(var_8, var_0)
    var_11 = 'value'
    var_12 = var_10.validate(var_11)
    assert var_12 == 'value'



# Parsed testcases at query #7
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1


def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)



# Parsed testcases at query #8
#--------------------------


import typesystem.fields as module_1
import typesystem.schemas as module_0


def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = {}
    var_6 = False
    var_7 = module_0.Schema(var_5)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = {}
    var_11 = module_0.Schema(var_10)
    var_12 = 'not a dict'
    var_13 = var_11.validate(var_12)
    var_14 = {}
    var_15 = module_0.Schema(var_14)
    var_16 = 1
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = var_15.validate(var_18)
    var_20 = 'required_field'
    var_21 = module_1.Field()
    var_22 = {var_20: var_21}
    var_23 = module_0.Schema(var_22)
    var_24 = {}
    var_25 = var_23.validate(var_24)
    var_26 = 'field_with_default'
    var_27 = 'default_value'
    var_28 = module_1.Field(default=var_27)
    var_29 = {var_26: var_28}
    var_30 = module_0.Schema(var_29)
    var_31 = {}
    var_32 = var_30.validate(var_31)
    var_33 = 'field'
    var_34 = module_1.Field(allow_null=var_6)
    var_35 = {var_33: var_34}
    var_36 = module_0.Schema(var_35)
    var_37 = 'field'
    var_38 = None
    var_39 = {var_37: var_38}
    var_40 = var_36.validate(var_39)
    var_41 = module_1.Field()
    var_42 = {var_33: var_41}
    var_43 = module_0.Schema(var_42)
    var_44 = 'value'
    var_45 = {var_33: var_44}
    var_46 = var_43.validate(var_45)



# Parsed testcases at query #9
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1


def test_case_0():
    var_0 = 'name'
    var_1 = module_0.Field()
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_1.Schema(var_2)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None
    var_7 = False
    var_8 = module_1.Schema(var_2)
    var_9 = None
    var_10 = var_8.validate(var_9)
    var_11 = 'not a dict'
    var_12 = var_8.validate(var_11)
    var_13 = 'one'
    var_14 = {var_3: var_13}
    var_15 = var_8.validate(var_14)
    var_16 = module_0.Field()
    var_17 = {var_15: var_16}
    var_18 = module_1.Schema(var_17)
    var_19 = {}
    var_20 = var_18.validate(var_19)
    var_21 = 'default_name'
    var_22 = module_0.Field(default=var_21)
    var_23 = {var_20: var_22}
    var_24 = module_1.Schema(var_23)
    var_25 = {}
    var_26 = var_24.validate(var_25)
    var_27 = 'age'
    var_28 = lambda x: x > var_7
    var_29 = [var_28]
    var_30 = module_0.Field()
    var_31 = {var_27: var_30}
    var_32 = module_1.Schema(var_31)
    var_33 = -1
    var_34 = {var_27: var_33}
    var_35 = var_32.validate(var_34)
    var_36 = module_0.Field()
    var_37 = lambda x: x > var_7
    var_38 = [var_37]
    var_39 = module_0.Field()
    var_40 = {var_35: var_36, var_27: var_39}
    var_41 = module_1.Schema(var_40)
    var_42 = 'John'
    var_43 = 25
    var_44 = {var_35: var_42, var_27: var_43}
    var_45 = var_41.validate(var_44)



# Parsed testcases at query #10
#--------------------------


import typesystem.schemas as module_0


def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'test'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = 'value'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'value'



# Parsed testcases at query #11
#--------------------------


import builtins as module_1


def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = False
    var_2 = 'test'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = None
    var_5 = var_3.validate(var_4)
    var_6 = 'test'
    var_7 = var_3.validate(var_6)
    var_8 = 1
    var_9 = var_3.validate(var_8)
    var_10 = 1.0
    var_11 = var_3.validate(var_10)
    var_12 = True
    var_13 = var_3.validate(var_12)
    var_14 = False
    var_15 = var_3.validate(var_14)
    var_16 = []
    var_17 = var_3.validate(var_16)
    var_18 = {}
    var_19 = var_3.validate(var_18)
    var_20 = set()
    var_21 = var_3.validate(var_20)
    var_22 = tuple()
    var_23 = var_3.validate(var_22)
    var_24 = frozenset()
    var_25 = var_3.validate(var_24)
    var_26 = module_1.object()
    var_27 = var_3.validate(var_26)
    var_28 = lambda x: x
    var_29 = var_3.validate(var_28)
    var_30 = var_3.validate(var_28)
    var_31 = var_3.validate(var_28)
    var_32 = var_3.validate(var_28)
    var_33 = var_3.validate(var_28)
    var_34 = var_3.validate(var_28)
    var_35 = var_3.validate(var_28)
    var_36 = var_3.validate(var_28)
    var_37 = var_3.validate(var_28)
    var_38 = var_3.validate(var_28)
    var_39 = var_3.validate(var_28)
    var_40 = var_3.validate(var_28)
    var_41 = var_3.validate(var_28)
    var_42 = var_3.validate(var_28)
    var_43 = var_3.validate(var_28)
    var_44 = var_3.validate(var_28)
    var_45 = var_3.validate(var_28)
    var_46 = var_3.validate(var_28)
    var_47 = var_3.validate(var_28)
    var_48 = var_3.validate(var_28)
    var_49 = var_3.validate(var_28)
    var_50 = var_3.validate(var_28)
    var_51 = var_3.validate(var_28)
    var_52 = var_3.validate(var_28)
    var_53 = var_3.validate(var_28)
    var_54 = var_3.validate(var_28)
    var_55 = var_3.validate(var_28)
    var_56 = var_3.validate(var_28)
    var_57 = var_3.validate(var_28)
    var_58 = var_3.validate(var_28)



# Parsed testcases at query #12
#--------------------------


import typesystem.fields as module_1


def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = module_1.Field()
    var_3 = True
    var_4 = module_1.Field(read_only=var_3)
    var_5 = 'default'
    var_6 = module_1.Field(default=var_5)
    var_7 = 'field1'
    var_8 = 'field2'
    var_9 = 'field3'
    var_10 = {var_7: var_2, var_8: var_4, var_9: var_6}
    var_11 = module_0.Schema(var_10)
    var_12 = {}
    var_13 = module_0.Schema(var_12)
    var_14 = {}
    var_15 = module_0.Schema(var_14)
    var_16 = {}
    var_17 = module_0.Schema(var_16)
    var_18 = {}
    var_19 = module_0.Schema(var_18)



# Parsed testcases at query #13
#--------------------------



def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'test'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = 'value'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'value'



# Parsed testcases at query #14
#--------------------------



def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = module_1.Field()
    var_3 = True
    var_4 = module_1.Field(read_only=var_3)
    var_5 = 'default'
    var_6 = module_1.Field(default=var_5)
    var_7 = 'field1'
    var_8 = 'field2'
    var_9 = 'field3'
    var_10 = {var_7: var_2, var_8: var_4, var_9: var_6}
    var_11 = module_0.Schema(var_10)
    var_12 = {}
    var_13 = module_0.Schema(var_12)
    var_14 = 'type'
    var_15 = 'Custom type error.'
    var_16 = {var_14: var_15}
    var_17 = {}
    var_18 = module_0.Schema(var_17)



# Parsed testcases at query #15
#--------------------------



def test_case_0():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    var_2 = 'default_value'
    var_3 = module_1.Field(default=var_2)
    var_4 = 'field1'
    var_5 = {var_4: var_3}
    var_6 = module_0.Schema(var_5)
    var_7 = module_1.Field()
    var_8 = 'field2'
    var_9 = {var_8: var_7}
    var_10 = module_0.Schema(var_9)
    var_11 = True
    var_12 = module_1.Field(read_only=var_11)
    var_13 = 'field3'
    var_14 = {var_13: var_12}
    var_15 = module_0.Schema(var_14)
    var_16 = 'default'
    var_17 = module_1.Field(default=var_16)
    var_18 = module_1.Field()
    var_19 = module_1.Field(read_only=var_11)
    var_20 = {var_4: var_17, var_8: var_18, var_13: var_19}
    var_21 = module_0.Schema(var_20)
    var_22 = {}
    var_23 = module_0.Schema(var_22)
    var_24 = 'type'
    var_25 = 'Custom type error'
    var_26 = {var_24: var_25}
    var_27 = {}
    var_28 = module_0.Schema(var_27)
    var_29 = 'All tests passed for Schema constructor.'
    var_30 = print(var_29)



# Parsed testcases at query #16
#--------------------------



def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = {}
    var_6 = False
    var_7 = module_0.Schema(var_5)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = {}
    var_11 = module_0.Schema(var_10)
    var_12 = 'not a dict'
    var_13 = var_11.validate(var_12)
    var_14 = {}
    var_15 = module_0.Schema(var_14)
    var_16 = 1
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = var_15.validate(var_18)
    var_20 = 'field1'
    var_21 = 'field2'
    var_22 = module_1.Field()
    var_23 = module_1.Field()
    var_24 = {var_20: var_22, var_21: var_23}
    var_25 = module_0.Schema(var_24)
    var_26 = {}
    var_27 = var_25.validate(var_26)
    var_28 = module_1.Field()
    var_29 = module_1.Field()
    var_30 = {var_20: var_28, var_21: var_29}
    var_31 = module_0.Schema(var_30)
    var_32 = 'value1'
    var_33 = 'value2'
    var_34 = {var_20: var_32, var_21: var_33}
    var_35 = var_31.validate(var_34)
    var_36 = module_1.Field(read_only=var_26)
    var_37 = module_1.Field()
    var_38 = {var_20: var_36, var_21: var_37}
    var_39 = module_0.Schema(var_38)
    var_40 = {var_20: var_32, var_21: var_33}
    var_41 = var_39.validate(var_40)
    var_42 = 'default1'
    var_43 = module_1.Field(default=var_42)
    var_44 = module_1.Field()
    var_45 = {var_20: var_43, var_21: var_44}
    var_46 = module_0.Schema(var_45)
    var_47 = {var_21: var_33}
    var_48 = var_46.validate(var_47)
    var_49 = module_1.Field()
    var_50 = module_1.Field()
    var_51 = {var_20: var_49, var_21: var_50}
    var_52 = module_0.Schema(var_51)
    var_53 = {var_20: var_32, var_21: var_27}
    var_54 = var_52.validate(var_53)
    var_55 = 'nested_field'
    var_56 = module_1.Field()
    var_57 = {var_55: var_56}
    var_58 = module_0.Schema(var_57)
    var_59 = {var_20: var_58}
    var_60 = module_0.Schema(var_59)
    var_61 = 'value'
    var_62 = {var_55: var_61}
    var_63 = {var_20: var_62}
    var_64 = var_60.validate(var_63)
    var_65 = module_1.Field()
    var_66 = {var_55: var_65}
    var_67 = module_0.Schema(var_66)
    var_68 = {var_20: var_67}
    var_69 = module_0.Schema(var_68)
    var_70 = {var_55: var_27}
    var_71 = {var_20: var_70}
    var_72 = var_69.validate(var_71)
    var_73 = module_1.Field(read_only=var_72)
    var_74 = {var_55: var_73}
    var_75 = module_0.Schema(var_74)
    var_76 = {var_20: var_75}
    var_77 = module_0.Schema(var_76)
    var_78 = {var_55: var_61}
    var_79 = {var_20: var_78}
    var_80 = var_77.validate(var_79)
    var_81 = 'default'
    var_82 = module_1.Field(default=var_81)
    var_83 = {var_55: var_82}
    var_84 = module_0.Schema(var_83)
    var_85 = {var_20: var_84}
    var_86 = module_0.Schema(var_85)
    var_87 = {}
    var_88 = {var_20: var_87}
    var_89 = var_86.validate(var_88)
    var_90 = module_1.Field(read_only=var_72)
    var_91 = {var_55: var_90}
    var_92 = module_0.Schema(var_91)
    var_93 = {var_20: var_92}
    var_94 = module_0.Schema(var_93)
    var_95 = {var_55: var_27}
    var_96 = {var_20: var_95}
    var_97 = var_94.validate(var_96)
    var_98 = module_1.Field(default=var_81)
    var_99 = {var_55: var_98}
    var_100 = module_0.Schema(var_99)
    var_101 = {var_20: var_100}
    var_102 = module_0.Schema(var_101)
    var_103 = {var_55: var_27}
    var_104 = {var_20: var_103}
    var_105 = var_102.validate(var_104)
    var_106 = 'nested_field1'
    var_107 = 'nested_field2'
    var_108 = module_1.Field(read_only=var_105)
    var_109 = module_1.Field(default=var_81)
    var_110 = {var_106: var_108, var_107: var_109}
    var_111 = module_0.Schema(var_110)
    var_112 = {var_20: var_111}
    var_113 = module_0.Schema(var_112)
    var_114 = {var_106: var_27, var_107: var_27}
    var_115 = {var_20: var_114}
    var_116 = var_113.validate(var_115)
    var_117 = module_1.Field(read_only=var_116)
    var_118 = module_1.Field(default=var_81)
    var_119 = {var_106: var_117, var_107: var_118}
    var_120 = module_0.Schema(var_119)
    var_121 = {var_20: var_120}
    var_122 = module_0.Schema(var_121)
    var_123 = {var_107: var_27}
    var_124 = {var_20: var_123}
    var_125 = var_122.validate(var_124)
    var_126 = module_1.Field(read_only=var_125)
    var_127 = module_1.Field(default=var_81)
    var_128 = {var_106: var_126, var_107: var_127}
    var_129 = module_0.Schema(var_128)
    var_130 = {var_20: var_129}
    var_131 = module_0.Schema(var_130)
    var_132 = {var_106: var_27}
    var_133 = {var_20: var_132}
    var_134 = var_131.validate(var_133)
    var_135 = module_1.Field(read_only=var_125)
    var_136 = module_1.Field(default=var_81)
    var_137 = {var_106: var_135, var_107: var_136}
    var_138 = module_0.Schema(var_137)
    var_139 = {var_20: var_138}
    var_140 = module_0.Schema(var_139)
    var_141 = {}
    var_142 = {var_20: var_141}
    var_143 = var_140.validate(var_142)
    var_144 = module_1.Field(read_only=var_125)
    var_145 = module_1.Field(default=var_81)
    var_146 = {var_106: var_144, var_107: var_145}
    var_147 = module_0.Schema(var_146)
    var_148 = {var_20: var_147}
    var_149 = module_0.Schema(var_148)
    var_150 = {var_106: var_32, var_107: var_33}
    var_151 = {var_20: var_150}
    var_152 = var_149.validate(var_151)
    var_153 = module_1.Field(read_only=var_125)
    var_154 = module_1.Field(default=var_81)
    var_155 = {var_106: var_153, var_107: var_154}
    var_156 = module_0.Schema(var_155)
    var_157 = {var_20: var_156}
    var_158 = module_0.Schema(var_157)
    var_159 = {var_106: var_32}
    var_160 = {var_20: var_159}
    var_161 = var_158.validate(var_160)



# Parsed testcases at query #17
#--------------------------



def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = {}
    var_6 = False
    var_7 = module_0.Schema(var_5)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = {}
    var_11 = module_0.Schema(var_10)
    var_12 = 'not a dict'
    var_13 = var_11.validate(var_12)
    var_14 = {}
    var_15 = module_0.Schema(var_14)
    var_16 = 1
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = var_15.validate(var_18)
    var_20 = 'required_field'
    var_21 = module_1.Field()
    var_22 = {var_20: var_21}
    var_23 = module_0.Schema(var_22)
    var_24 = {}
    var_25 = var_23.validate(var_24)
    var_26 = module_1.Field()
    var_27 = {var_20: var_26}
    var_28 = module_0.Schema(var_27)
    var_29 = 'value'
    var_30 = {var_20: var_29}
    var_31 = var_28.validate(var_30)
    var_32 = 'field_with_default'
    var_33 = 'default_value'
    var_34 = module_1.Field(default=var_33)
    var_35 = {var_32: var_34}
    var_36 = module_0.Schema(var_35)
    var_37 = {}
    var_38 = var_36.validate(var_37)
    var_39 = 'read_only_field'
    var_40 = module_1.Field(read_only=var_24)
    var_41 = {var_39: var_40}
    var_42 = module_0.Schema(var_41)
    var_43 = {var_39: var_29}
    var_44 = var_42.validate(var_43)
    var_45 = 'field_with_validation'
    var_46 = module_1.Field(allow_null=var_19)
    var_47 = {var_45: var_46}
    var_48 = module_0.Schema(var_47)
    var_49 = 'field_with_validation'
    var_50 = None
    var_51 = {var_49: var_50}
    var_52 = var_48.validate(var_51)
    var_53 = module_1.Field()
    var_54 = module_1.Field(allow_null=var_52)
    var_55 = {var_20: var_53, var_45: var_54}
    var_56 = module_0.Schema(var_55)
    var_57 = 'field_with_validation'
    var_58 = None
    var_59 = {var_57: var_58}
    var_60 = var_56.validate(var_59)
    var_61 = 'nested_field'
    var_62 = module_1.Field()
    var_63 = {var_61: var_62}
    var_64 = module_0.Schema(var_63)
    var_65 = 'nested'
    var_66 = {var_65: var_64}
    var_67 = module_0.Schema(var_66)
    var_68 = {var_61: var_29}
    var_69 = {var_65: var_68}
    var_70 = var_67.validate(var_69)
    var_71 = module_1.Field(allow_null=var_60)
    var_72 = {var_61: var_71}
    var_73 = module_0.Schema(var_72)
    var_74 = {var_65: var_73}
    var_75 = module_0.Schema(var_74)
    var_76 = 'nested'
    var_77 = 'nested_field'
    var_78 = None
    var_79 = {var_77: var_78}
    var_80 = {var_76: var_79}
    var_81 = var_75.validate(var_80)
    var_82 = module_1.Field()
    var_83 = {var_80: var_82}
    var_84 = module_0.Schema(var_83)
    var_85 = {var_65: var_84}
    var_86 = module_0.Schema(var_85)
    var_87 = 'nested'
    var_88 = {}
    var_89 = {var_87: var_88}
    var_90 = var_86.validate(var_89)
    var_91 = module_1.Field(default=var_33)
    var_92 = {var_32: var_91}
    var_93 = module_0.Schema(var_92)
    var_94 = {var_65: var_93}
    var_95 = module_0.Schema(var_94)
    var_96 = {}
    var_97 = {var_65: var_96}
    var_98 = var_95.validate(var_97)
    var_99 = module_1.Field(read_only=var_87)
    var_100 = {var_39: var_99}
    var_101 = module_0.Schema(var_100)
    var_102 = {var_65: var_101}
    var_103 = module_0.Schema(var_102)
    var_104 = {var_39: var_29}
    var_105 = {var_65: var_104}
    var_106 = var_103.validate(var_105)
    var_107 = module_1.Field()
    var_108 = module_1.Field(allow_null=var_90)
    var_109 = {var_80: var_107, var_45: var_108}
    var_110 = module_0.Schema(var_109)
    var_111 = {var_65: var_110}
    var_112 = module_0.Schema(var_111)
    var_113 = 'nested'
    var_114 = 'field_with_validation'
    var_115 = None
    var_116 = {var_114: var_115}
    var_117 = {var_113: var_116}
    var_118 = var_112.validate(var_117)
    var_119 = 'nested_nested_field'
    var_120 = module_1.Field()
    var_121 = {var_119: var_120}
    var_122 = module_0.Schema(var_121)
    var_123 = 'nested_nested'
    var_124 = {var_123: var_122}
    var_125 = module_0.Schema(var_124)
    var_126 = {var_65: var_125}
    var_127 = module_0.Schema(var_126)
    var_128 = {var_119: var_29}
    var_129 = {var_123: var_128}
    var_130 = {var_65: var_129}
    var_131 = var_127.validate(var_130)
    var_132 = module_1.Field(allow_null=var_116)
    var_133 = {var_119: var_132}
    var_134 = module_0.Schema(var_133)
    var_135 = {var_123: var_134}
    var_136 = module_0.Schema(var_135)
    var_137 = {var_65: var_136}
    var_138 = module_0.Schema(var_137)
    var_139 = 'nested'
    var_140 = 'nested_nested'
    var_141 = 'nested_nested_field'
    var_142 = None
    var_143 = {var_141: var_142}
    var_144 = {var_140: var_143}
    var_145 = {var_139: var_144}
    var_146 = var_138.validate(var_145)
    var_147 = module_1.Field()
    var_148 = {var_143: var_147}
    var_149 = module_0.Schema(var_148)
    var_150 = {var_123: var_149}
    var_151 = module_0.Schema(var_150)
    var_152 = {var_65: var_151}
    var_153 = module_0.Schema(var_152)
    var_154 = 'nested'
    var_155 = 'nested_nested'
    var_156 = {}
    var_157 = {var_155: var_156}
    var_158 = {var_154: var_157}
    var_159 = var_153.validate(var_158)
    var_160 = module_1.Field(default=var_33)
    var_161 = {var_32: var_160}
    var_162 = module_0.Schema(var_161)
    var_163 = {var_123: var_162}
    var_164 = module_0.Schema(var_163)
    var_165 = {var_65: var_164}
    var_166 = module_0.Schema(var_165)
    var_167 = {}
    var_168 = {var_123: var_167}
    var_169 = {var_65: var_168}
    var_170 = var_166.validate(var_169)



# Parsed testcases at query #18
#--------------------------



def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = 'test'
    var_5 = module_0.Reference(var_4, var_0)
    var_6 = {var_1: var_4}
    var_7 = var_5.validate(var_6)
    var_8 = None
    var_9 = var_5.validate(var_8)
    assert var_9 is None
    var_10 = None
    var_11 = var_5.validate(var_10)
    var_12 = 'invalid'
    var_13 = var_5.validate(var_12)



# Parsed testcases at query #19
#--------------------------



def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'test'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = 'value'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'value'



# Parsed testcases at query #20
#--------------------------



def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = {}
    var_6 = False
    var_7 = module_0.Schema(var_5)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = {}
    var_11 = module_0.Schema(var_10)
    var_12 = 'not a dict'
    var_13 = var_11.validate(var_12)
    var_14 = {}
    var_15 = module_0.Schema(var_14)
    var_16 = 1
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = var_15.validate(var_18)
    var_20 = 'required_field'
    var_21 = module_1.Field()
    var_22 = {var_20: var_21}
    var_23 = module_0.Schema(var_22)
    var_24 = {}
    var_25 = var_23.validate(var_24)
    var_26 = 'field_with_default'
    var_27 = 'default_value'
    var_28 = module_1.Field(default=var_27)
    var_29 = {var_26: var_28}
    var_30 = module_0.Schema(var_29)
    var_31 = {}
    var_32 = var_30.validate(var_31)
    var_33 = 'field'
    var_34 = module_1.Field()
    var_35 = {var_33: var_34}
    var_36 = module_0.Schema(var_35)
    var_37 = 'field'
    var_38 = 'invalid_value'
    var_39 = {var_37: var_38}
    var_40 = var_36.validate(var_39)
    var_41 = module_1.Field()
    var_42 = {var_33: var_41}
    var_43 = module_0.Schema(var_42)
    var_44 = 'valid_value'
    var_45 = {var_33: var_44}
    var_46 = var_43.validate(var_45)



# Parsed testcases at query #21
#--------------------------



def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'test'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = 1
    var_6 = var_2.validate(var_5)
    assert var_6 == 1
    var_7 = None
    var_8 = var_2.validate(var_7)



# Parsed testcases at query #22
#--------------------------



def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'test'
    var_2 = True
    var_3 = module_0.Reference(var_1, var_0)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None
    var_6 = False
    var_7 = module_0.Reference(var_1, var_0)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = module_0.Reference(var_8, var_0)
    var_11 = var_10.validate(var_8)
    assert var_11 == 'test'



# Parsed testcases at query #23
#--------------------------



def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'test'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = 'not None'
    var_6 = var_2.validate(var_5)



# Parsed testcases at query #24
#--------------------------



def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Definitions()
    var_2 = True
    var_3 = module_0.Reference(var_0, var_1)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None
    var_6 = module_0.Definitions()
    var_7 = False
    var_8 = module_0.Reference(var_0, var_6)
    var_9 = None
    var_10 = var_8.validate(var_9)
    var_11 = module_0.Definitions()
    var_12 = module_0.Reference(var_9, var_11)
    var_13 = var_12.validate(var_9)
    assert var_13 == 'test'



# Parsed testcases at query #25
#--------------------------



def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'name'
    var_2 = module_1.Field()
    var_3 = {var_1: var_2}
    var_4 = 'test'
    var_5 = module_0.Reference(var_4, var_0)
    var_6 = {var_1: var_4}
    var_7 = var_5.validate(var_6)
    var_8 = None
    var_9 = var_5.validate(var_8)
    assert var_9 is None
    var_10 = None
    var_11 = False
    var_12 = var_5.validate(var_10)



# Parsed testcases at query #26
#--------------------------



def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = {}
    var_6 = False
    var_7 = module_0.Schema(var_5)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = {}
    var_11 = module_0.Schema(var_10)
    var_12 = 'not a dict'
    var_13 = var_11.validate(var_12)
    var_14 = {}
    var_15 = module_0.Schema(var_14)
    var_16 = 'value'
    var_17 = {var_12: var_16}
    var_18 = var_15.validate(var_17)
    var_19 = 'required_field'
    var_20 = module_1.Field()
    var_21 = {var_19: var_20}
    var_22 = module_0.Schema(var_21)
    var_23 = {}
    var_24 = var_22.validate(var_23)
    var_25 = 'field_with_default'
    var_26 = 'default_value'
    var_27 = module_1.Field(default=var_26)
    var_28 = {var_25: var_27}
    var_29 = module_0.Schema(var_28)
    var_30 = {}
    var_31 = var_29.validate(var_30)
    var_32 = 'read_only_field'
    var_33 = module_1.Field(read_only=var_24)
    var_34 = {var_32: var_33}
    var_35 = module_0.Schema(var_34)
    var_36 = {var_32: var_16}
    var_37 = var_35.validate(var_36)
    var_38 = 'field'
    var_39 = module_1.Field(allow_null=var_6)
    var_40 = {var_38: var_39}
    var_41 = module_0.Schema(var_40)
    var_42 = {var_38: var_13}
    var_43 = var_41.validate(var_42)
    var_44 = 'field1'
    var_45 = 'field2'
    var_46 = module_1.Field()
    var_47 = 'default'
    var_48 = module_1.Field(default=var_47)
    var_49 = {var_44: var_46, var_45: var_48}
    var_50 = module_0.Schema(var_49)
    var_51 = 'value1'
    var_52 = {var_44: var_51}
    var_53 = var_50.validate(var_52)
    var_54 = 'nested_field'
    var_55 = module_1.Field()
    var_56 = {var_54: var_55}
    var_57 = module_0.Schema(var_56)
    var_58 = 'nested'
    var_59 = {var_58: var_57}
    var_60 = module_0.Schema(var_59)
    var_61 = {var_54: var_16}
    var_62 = {var_58: var_61}
    var_63 = var_60.validate(var_62)
    var_64 = module_1.Field(allow_null=var_6)
    var_65 = {var_54: var_64}
    var_66 = module_0.Schema(var_65)
    var_67 = {var_58: var_66}
    var_68 = module_0.Schema(var_67)
    var_69 = {var_54: var_13}
    var_70 = {var_58: var_69}
    var_71 = var_68.validate(var_70)
    var_72 = {}
    var_73 = module_0.Schema(var_72)
    var_74 = 'key'
    var_75 = {var_74: var_16}
    var_76 = var_73.validate(var_70)
    var_77 = 'custom_field'
    var_78 = module_0.Schema(var_72)
    var_79 = 'valid'
    var_80 = {var_77: var_79}
    var_81 = var_78.validate(var_80)
    var_82 = module_0.Schema(var_72)
    var_83 = 'invalid'
    var_84 = {var_77: var_83}
    var_85 = var_82.validate(var_84)
    var_86 = module_1.Field()
    var_87 = module_1.Field(allow_null=var_6)
    var_88 = {var_19: var_86, var_38: var_87}
    var_89 = module_0.Schema(var_88)
    var_90 = {var_38: var_13}
    var_91 = var_89.validate(var_90)
    var_92 = {}
    var_93 = module_0.Schema(var_92)
    var_94 = {}
    var_95 = var_93.validate(var_94)
    var_96 = module_1.Field()
    var_97 = {var_38: var_96}
    var_98 = module_0.Schema(var_97)
    var_99 = 'extra'
    var_100 = {var_38: var_16, var_99: var_99}
    var_101 = var_98.validate(var_100)
    var_102 = module_1.Field(default=var_47)
    var_103 = {var_38: var_102}
    var_104 = module_0.Schema(var_103)
    var_105 = 'provided'
    var_106 = {var_38: var_105}
    var_107 = var_104.validate(var_106)
    var_108 = module_1.Field(default=var_47, allow_null=var_91)
    var_109 = {var_38: var_108}
    var_110 = module_0.Schema(var_109)
    var_111 = {var_38: var_13}
    var_112 = var_110.validate(var_111)
    var_113 = module_1.Field(default=var_47, allow_null=var_6)
    var_114 = {var_38: var_113}
    var_115 = module_0.Schema(var_114)
    var_116 = {var_38: var_13}
    var_117 = var_115.validate(var_116)



# Parsed testcases at query #27
#--------------------------



def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = {}
    var_6 = False
    var_7 = module_0.Schema(var_5)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = {}
    var_11 = module_0.Schema(var_10)
    var_12 = 'not a dict'
    var_13 = var_11.validate(var_12)
    var_14 = {}
    var_15 = module_0.Schema(var_14)
    var_16 = 1
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = var_15.validate(var_18)
    var_20 = 'required_field'
    var_21 = module_1.Field()
    var_22 = {var_20: var_21}
    var_23 = module_0.Schema(var_22)
    var_24 = {}
    var_25 = var_23.validate(var_24)
    var_26 = 'field_with_default'
    var_27 = 'default'
    var_28 = module_1.Field(default=var_27)
    var_29 = {var_26: var_28}
    var_30 = module_0.Schema(var_29)
    var_31 = {}
    var_32 = var_30.validate(var_31)
    var_33 = 'field'
    var_34 = module_1.Field(allow_null=var_6)
    var_35 = {var_33: var_34}
    var_36 = module_0.Schema(var_35)
    var_37 = 'field'
    var_38 = None
    var_39 = {var_37: var_38}
    var_40 = var_36.validate(var_39)
    var_41 = module_1.Field()
    var_42 = {var_33: var_41}
    var_43 = module_0.Schema(var_42)
    var_44 = 'value'
    var_45 = {var_33: var_44}
    var_46 = var_43.validate(var_45)



# Parsed testcases at query #28
#--------------------------



def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = 'test'
    var_2 = module_0.Reference(var_1, var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = 1
    var_6 = var_2.validate(var_5)



# Parsed testcases at query #29
#--------------------------



def test_case_0():
    var_0 = module_0.Definitions()
    var_1 = False
    var_2 = 'test'
    var_3 = module_0.Reference(var_2, var_0)
    var_4 = None
    var_5 = var_3.validate(var_4)



# Parsed testcases at query #30
#--------------------------



def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.Schema(var_0)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None
    var_5 = {}
    var_6 = False
    var_7 = module_0.Schema(var_5)
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = {}
    var_11 = module_0.Schema(var_10)
    var_12 = 'not a dict'
    var_13 = var_11.validate(var_12)
    var_14 = {}
    var_15 = module_0.Schema(var_14)
    var_16 = 1
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = var_15.validate(var_18)
    var_20 = 'field'
    var_21 = module_1.Field()
    var_22 = {var_20: var_21}
    var_23 = module_0.Schema(var_22)
    var_24 = {}
    var_25 = var_23.validate(var_24)
    var_26 = 'default'
    var_27 = module_1.Field(default=var_26)
    var_28 = {var_20: var_27}
    var_29 = module_0.Schema(var_28)
    var_30 = {}
    var_31 = var_29.validate(var_30)
    var_32 = module_1.Field(allow_null=var_6)
    var_33 = {var_20: var_32}
    var_34 = module_0.Schema(var_33)
    var_35 = 'field'
    var_36 = None
    var_37 = {var_35: var_36}
    var_38 = var_34.validate(var_37)
    var_39 = module_1.Field()
    var_40 = {var_20: var_39}
    var_41 = module_0.Schema(var_40)
    var_42 = 'value'
    var_43 = {var_20: var_42}
    var_44 = var_41.validate(var_43)



