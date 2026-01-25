####################################################################
#    TEST GENERATION BEGINS (CODAMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.fields as module_0

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
    var_8 = exc_info.value.messages()[var_4]
    var_9 = var_8.code
    assert var_9 == 'null'
    var_10 = module_0.Array()
    var_11 = 'not a list'
    var_12 = var_10.validate(var_11)
    var_13 = exc_info.value.messages()[var_4]
    var_14 = var_13.code
    assert var_14 == 'type'
    var_15 = module_0.Array()
    var_16 = 'key'
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = var_15.validate(var_18)
    var_20 = exc_info.value.messages()[var_19]
    var_21 = var_20.code
    assert var_21 == 'type'
    var_22 = module_0.Array(min_items=var_16)
    var_23 = []
    var_24 = var_22.validate(var_23)
    var_25 = exc_info.value.messages()[var_19]
    var_26 = var_25.code
    assert var_26 == 'empty'
    var_27 = 3
    var_28 = module_0.Array(min_items=var_27)
    var_29 = 1
    var_30 = 2
    var_31 = [var_29, var_30]
    var_32 = var_28.validate(var_31)
    var_33 = exc_info.value.messages()[var_32]
    var_34 = var_33.code
    assert var_34 == 'min_items'
    var_35 = 2
    var_36 = module_0.Array(max_items=var_35)
    var_37 = 1
    var_38 = 2
    var_39 = 3
    var_40 = [var_37, var_38, var_39]
    var_41 = var_36.validate(var_40)
    var_42 = exc_info.value.messages()[var_40]
    var_43 = var_42.code
    assert var_43 == 'max_items'
    var_44 = module_0.Array(exact_items=var_35)
    var_45 = 1
    var_46 = 2
    var_47 = 3
    var_48 = [var_45, var_46, var_47]
    var_49 = var_44.validate(var_48)
    var_50 = exc_info.value.messages()[var_48]
    var_51 = var_50.code
    assert var_51 == 'exact_items'
    var_52 = module_0.Array(exact_items=var_35)
    var_53 = [var_45, var_35]
    var_54 = var_52.validate(var_53)
    var_55 = module_0.Integer()
    var_56 = module_0.Array(var_55)
    var_57 = [var_45, var_35, var_27]
    var_58 = var_56.validate(var_57)
    var_59 = module_0.Integer()
    var_60 = module_0.Array(var_59)
    var_61 = 1
    var_62 = 'invalid'
    var_63 = 3
    var_64 = [var_61, var_62, var_63]
    var_65 = var_60.validate(var_64)
    var_66 = exc_info.value.messages()[var_64]
    var_67 = var_66.code
    assert var_67 == 'type'
    var_68 = module_0.Integer()
    var_69 = module_0.String()
    var_70 = [var_68, var_69]
    var_71 = module_0.Array(var_70)
    var_72 = 'test'
    var_73 = [var_61, var_72]
    var_74 = var_71.validate(var_73)
    var_75 = module_0.Integer()
    var_76 = module_0.String()
    var_77 = [var_75, var_76]
    var_78 = module_0.Array(var_77, var_64)
    var_79 = 1
    var_80 = 'test'
    var_81 = 'extra'
    var_82 = [var_79, var_80, var_81]
    var_83 = var_78.validate(var_82)
    var_84 = exc_info.value.messages()[var_82]
    var_85 = var_84.code
    assert var_85 == 'additional_items'
    var_86 = module_0.Integer()
    var_87 = module_0.String()
    var_88 = [var_86, var_87]
    var_89 = module_0.Integer()
    var_90 = module_0.Array(var_88, var_89)
    var_91 = [var_79, var_72, var_27]
    var_92 = var_90.validate(var_91)
    var_93 = module_0.Array(unique_items=var_79)
    var_94 = 1
    var_95 = 2
    var_96 = [var_94, var_95, var_94]
    var_97 = var_93.validate(var_96)
    var_98 = exc_info.value.messages()[var_97]
    var_99 = var_98.code
    assert var_99 == 'unique_items'
    var_100 = module_0.Array(unique_items=var_94)
    var_101 = [var_94, var_35, var_27]
    var_102 = var_100.validate(var_101)
    var_103 = module_0.Array(unique_items=var_94)
    var_104 = 'a'
    var_105 = 'b'
    var_106 = [var_104, var_105, var_104]
    var_107 = var_103.validate(var_106)
    var_108 = exc_info.value.messages()[var_107]
    var_109 = var_108.code
    assert var_109 == 'unique_items'
    var_110 = module_0.Array()
    var_111 = 'key'
    var_112 = 'value'
    var_113 = {var_111: var_112}
    var_114 = [var_104, var_72, var_105, var_113]
    var_115 = var_110.validate(var_114)
    var_116 = module_0.Integer(minimum=var_107)
    var_117 = module_0.Array(var_116)
    var_118 = 1
    var_119 = -5
    var_120 = 3
    var_121 = [var_118, var_119, var_120]
    var_122 = var_117.validate(var_121)
    var_123 = exc_info.value.messages()[var_121]
    var_124 = var_123.code
    assert var_124 == 'minimum'
    var_125 = module_0.Integer()
    var_126 = module_0.Array(var_125)
    var_127 = module_0.Array(var_126)
    var_128 = [var_118, var_35]
    var_129 = 4
    var_130 = [var_27, var_129]
    var_131 = [var_128, var_130]
    var_132 = var_127.validate(var_131)
    var_133 = module_0.Array(min_items=var_121)
    var_134 = []
    var_135 = var_133.validate(var_134)
    var_136 = module_0.Array(max_items=var_121)
    var_137 = []
    var_138 = var_136.validate(var_137)
    var_139 = module_0.Array(max_items=var_121)
    var_140 = 1
    var_141 = [var_140]
    var_142 = var_139.validate(var_141)
    var_143 = exc_info.value.messages()[var_121]
    var_144 = var_143.code
    assert var_144 == 'max_items'
    var_145 = module_0.Integer()
    var_146 = module_0.String()
    var_147 = [var_145, var_146]
    var_148 = module_0.Array(var_147, var_140)
    var_149 = 'extra'
    var_150 = 100
    var_151 = [var_140, var_72, var_149, var_150]
    var_152 = var_148.validate(var_151)
    var_153 = module_0.Integer()
    var_154 = module_0.Array(var_153)
    var_155 = 1
    var_156 = 'invalid'
    var_157 = 3
    var_158 = [var_155, var_156, var_157]
    var_159 = var_154.validate(var_158)
    var_160 = exc_info.value.messages()[var_158]



# Parsed testcases at query #2
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Array()
    var_1 = module_0.String()
    var_2 = module_0.Array(var_1)
    var_3 = module_0.String()
    var_4 = module_0.Integer()
    var_5 = [var_3, var_4]
    var_6 = module_0.Array(var_5)
    var_7 = module_0.String()
    var_8 = module_0.Integer()
    var_9 = [var_7, var_8]
    var_10 = True
    var_11 = module_0.Array(var_9, var_10)
    var_12 = module_0.Boolean()
    var_13 = module_0.String()
    var_14 = [var_13]
    var_15 = module_0.Array(var_14, var_12)
    var_16 = module_0.String()
    var_17 = 10
    var_18 = module_0.Array(var_16, min_items=var_10, max_items=var_17)
    var_19 = module_0.String()
    var_20 = 5
    var_21 = module_0.Array(var_19, exact_items=var_20)
    var_22 = module_0.String()
    var_23 = module_0.Array(var_22, unique_items=var_10)
    var_24 = module_0.String()
    var_25 = module_0.Integer()
    var_26 = (var_24, var_25)
    var_27 = module_0.Array(var_26)
    var_28 = var_27.items
    var_29 = module_0.String()
    var_30 = False
    var_31 = 2
    var_32 = 20
    var_33 = module_0.Array(var_29, var_30, var_31, var_32, unique_items=var_10)
    var_34 = module_0.String()
    var_35 = module_0.Array(var_34, min_items=var_10, max_items=var_17, exact_items=var_20)
    var_36 = module_0.String()
    var_37 = module_0.Integer()
    var_38 = module_0.Boolean()
    var_39 = [var_36, var_37, var_38]
    var_40 = module_0.Array(var_39, var_30)



# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = 'hello'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'hello'
    var_6 = 42
    var_7 = var_3.validate(var_6)
    assert var_7 == 42
    var_8 = module_0.String()
    var_9 = module_0.Integer()
    var_10 = [var_8, var_9]
    var_11 = True
    var_12 = module_0.Union(var_10)
    var_13 = None
    var_14 = var_12.validate(var_13)
    assert var_14 is None
    var_15 = module_0.String()
    var_16 = module_0.Integer()
    var_17 = [var_15, var_16]
    var_18 = module_0.Union(var_17)
    var_19 = None
    var_20 = var_18.validate(var_19)
    var_21 = 0
    var_22 = exc_info.value.messages()[var_21]
    var_23 = var_22.code
    assert var_23 == 'null'
    var_24 = 1
    var_25 = 2
    var_26 = 3
    var_27 = [var_24, var_25, var_26]
    var_28 = var_18.validate(var_27)
    var_29 = exc_info.value.messages()[var_21]
    var_30 = var_29.code
    assert var_30 == 'union'
    var_31 = module_0.String()
    var_32 = module_0.Integer()
    var_33 = [var_31, var_32]
    var_34 = module_0.Union(var_33)
    var_35 = 3
    var_36 = module_0.String(max_length=var_35)
    var_37 = module_0.Integer()
    var_38 = [var_36, var_37]
    var_39 = module_0.Union(var_38)
    var_40 = 'toolong'
    var_41 = var_39.validate(var_40)
    var_42 = 2
    var_43 = module_0.String(max_length=var_42)
    var_44 = 100
    var_45 = module_0.Integer(minimum=var_44)
    var_46 = [var_43, var_45]
    var_47 = module_0.Union(var_46)
    var_48 = 'toolong'
    var_49 = var_47.validate(var_48)
    var_50 = exc_info.value.messages()[var_21]
    var_51 = var_50.code
    assert var_51 == 'union'
    var_52 = module_0.String()
    var_53 = module_0.Integer()
    var_54 = [var_52, var_53]
    var_55 = module_0.Union(var_54)
    var_56 = var_55.validate(var_35)
    assert var_56 == 3
    var_57 = module_0.String()
    var_58 = module_0.Integer()
    var_59 = [var_57, var_58]
    var_60 = module_0.Union(var_59)
    var_61 = ''
    var_62 = var_60.validate(var_61)
    assert var_62 == ''
    var_63 = module_0.String()
    var_64 = module_0.Integer()
    var_65 = [var_63, var_64]
    var_66 = module_0.Union(var_65)
    var_67 = True
    var_68 = var_66.validate(var_67)
    var_69 = exc_info.value.messages()[var_21]
    var_70 = var_69.code
    assert var_70 == 'union'
    var_71 = module_0.String()
    var_72 = module_0.Float()
    var_73 = [var_71, var_72]
    var_74 = module_0.Union(var_73)
    var_75 = '123'
    var_76 = var_74.validate(var_75)
    assert var_76 == '123'
    var_77 = 'name'
    var_78 = module_0.String()
    var_79 = {var_77: var_78}
    var_80 = module_0.Object(properties=var_79)
    var_81 = module_0.String()
    var_82 = [var_80, var_81]
    var_83 = module_0.Union(var_82)
    var_84 = 'test'
    var_85 = {var_77: var_84}
    var_86 = var_83.validate(var_85)
    var_87 = module_0.Integer(minimum=var_21, maximum=var_44)
    var_88 = module_0.String()
    var_89 = [var_88, var_87]
    var_90 = module_0.Union(var_89)
    var_91 = 150
    var_92 = var_90.validate(var_91)
    var_93 = exc_info.value.messages()[var_21]
    var_94 = var_93.code
    assert var_94 == 'maximum'
    var_95 = module_0.String()
    var_96 = module_0.Array(var_95)
    var_97 = module_0.Integer()
    var_98 = [var_96, var_97]
    var_99 = module_0.Union(var_98)
    var_100 = 'key'
    var_101 = 'value'
    var_102 = {var_100: var_101}
    var_103 = var_99.validate(var_102)
    var_104 = exc_info.value.messages()[var_21]
    var_105 = var_104.code
    assert var_105 == 'union'



# Parsed testcases at query #4
#--------------------------


import typesystem.fields as module_0

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
    var_8 = exc_info.value.messages()[var_4]
    var_9 = var_8.code
    assert var_9 == 'null'
    var_10 = module_0.Array()
    var_11 = 'not a list'
    var_12 = var_10.validate(var_11)
    var_13 = exc_info.value.messages()[var_4]
    var_14 = var_13.code
    assert var_14 == 'type'
    var_15 = module_0.Array(min_items=var_11)
    var_16 = []
    var_17 = var_15.validate(var_16)
    var_18 = exc_info.value.messages()[var_4]
    var_19 = var_18.code
    assert var_19 == 'empty'
    var_20 = 3
    var_21 = module_0.Array(min_items=var_20)
    var_22 = 1
    var_23 = 2
    var_24 = [var_22, var_23]
    var_25 = var_21.validate(var_24)
    var_26 = exc_info.value.messages()[var_25]
    var_27 = var_26.code
    assert var_27 == 'min_items'
    var_28 = 2
    var_29 = module_0.Array(max_items=var_28)
    var_30 = 1
    var_31 = 2
    var_32 = 3
    var_33 = [var_30, var_31, var_32]
    var_34 = var_29.validate(var_33)
    var_35 = exc_info.value.messages()[var_33]
    var_36 = var_35.code
    assert var_36 == 'max_items'
    var_37 = module_0.Array(exact_items=var_28)
    var_38 = 1
    var_39 = 2
    var_40 = 3
    var_41 = [var_38, var_39, var_40]
    var_42 = var_37.validate(var_41)
    var_43 = exc_info.value.messages()[var_41]
    var_44 = var_43.code
    assert var_44 == 'exact_items'
    var_45 = module_0.Array(exact_items=var_28)
    var_46 = [var_38, var_28]
    var_47 = var_45.validate(var_46)
    var_48 = module_0.Integer()
    var_49 = module_0.Array(var_48)
    var_50 = [var_38, var_28, var_20]
    var_51 = var_49.validate(var_50)
    var_52 = module_0.Integer()
    var_53 = module_0.Array(var_52)
    var_54 = 1
    var_55 = 'invalid'
    var_56 = 3
    var_57 = [var_54, var_55, var_56]
    var_58 = var_53.validate(var_57)
    var_59 = module_0.Integer()
    var_60 = module_0.String()
    var_61 = [var_59, var_60]
    var_62 = module_0.Array(var_61)
    var_63 = 'test'
    var_64 = [var_54, var_63]
    var_65 = var_62.validate(var_64)
    var_66 = module_0.Integer()
    var_67 = module_0.String()
    var_68 = [var_66, var_67]
    var_69 = module_0.Array(var_68, var_57)
    var_70 = 1
    var_71 = 'test'
    var_72 = 3
    var_73 = [var_70, var_71, var_72]
    var_74 = var_69.validate(var_73)
    var_75 = module_0.Integer()
    var_76 = [var_75]
    var_77 = module_0.String()
    var_78 = module_0.Array(var_76, var_77)
    var_79 = 'extra'
    var_80 = [var_70, var_79]
    var_81 = var_78.validate(var_80)
    var_82 = module_0.Array(unique_items=var_70)
    var_83 = 1
    var_84 = 2
    var_85 = [var_83, var_84, var_83]
    var_86 = var_82.validate(var_85)
    var_87 = 'unique_items'
    var_88 = module_0.Array(unique_items=var_83)
    var_89 = [var_83, var_28, var_20]
    var_90 = var_88.validate(var_89)
    var_91 = module_0.Array()
    var_92 = []
    var_93 = var_91.validate(var_92)
    var_94 = [var_83, var_63, var_83]
    var_95 = var_91.validate(var_94)
    var_96 = module_0.Integer()
    var_97 = module_0.Array(var_96)
    var_98 = 1
    var_99 = 2
    var_100 = 'not_int'
    var_101 = [var_98, var_99, var_100]
    var_102 = var_97.validate(var_101)
    var_103 = module_0.Integer()
    var_104 = module_0.String()
    var_105 = module_0.Boolean()
    var_106 = [var_103, var_104, var_105]
    var_107 = module_0.Array(var_106)
    var_108 = 42
    var_109 = 'hello'
    var_110 = [var_108, var_109, var_98]
    var_111 = var_107.validate(var_110)
    var_112 = 4
    var_113 = module_0.Array(min_items=var_28, max_items=var_112)
    var_114 = [var_98, var_28]
    var_115 = var_113.validate(var_114)
    var_116 = [var_98, var_28, var_20, var_112]
    var_117 = var_113.validate(var_116)
    var_118 = 1
    var_119 = [var_118]
    var_120 = var_113.validate(var_119)
    var_121 = exc_info.value.messages()[var_101]
    var_122 = var_121.code
    assert var_122 == 'min_items'
    var_123 = 1
    var_124 = 2
    var_125 = 3
    var_126 = 4
    var_127 = 5
    var_128 = [var_123, var_124, var_125, var_126, var_127]
    var_129 = var_113.validate(var_128)
    var_130 = exc_info.value.messages()[var_126]
    var_131 = var_130.code
    assert var_131 == 'max_items'



# Parsed testcases at query #5
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Const(var_0)
    var_2 = None
    var_3 = module_0.Const(var_2)
    var_4 = 'hello'
    var_5 = module_0.Const(var_4)
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]
    var_10 = module_0.Const(var_9)
    var_11 = 'key'
    var_12 = 'value'
    var_13 = {var_11: var_12}
    var_14 = module_0.Const(var_13)
    var_15 = 42
    var_16 = True
    var_17 = module_0.Const(var_15)
    var_18 = 100
    var_19 = 'A constant field'
    var_20 = module_0.Const(var_18)
    var_21 = True
    var_22 = module_0.Const(var_21)
    var_23 = 3.14
    var_24 = module_0.Const(var_23)



# Parsed testcases at query #6
#--------------------------


import typesystem.fields as module_0

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
    var_8 = exc_info.value.messages()[var_4]
    var_9 = var_8.code
    assert var_9 == 'null'
    var_10 = module_0.Array()
    var_11 = 'not a list'
    var_12 = var_10.validate(var_11)
    var_13 = exc_info.value.messages()[var_4]
    var_14 = var_13.code
    assert var_14 == 'type'
    var_15 = module_0.Array(min_items=var_11)
    var_16 = []
    var_17 = var_15.validate(var_16)
    var_18 = exc_info.value.messages()[var_4]
    var_19 = var_18.code
    assert var_19 == 'empty'
    var_20 = 2
    var_21 = module_0.Array(exact_items=var_20)
    var_22 = 1
    var_23 = [var_22]
    var_24 = var_21.validate(var_23)
    var_25 = exc_info.value.messages()[var_4]
    var_26 = var_25.code
    assert var_26 == 'exact_items'
    var_27 = 3
    var_28 = module_0.Array(min_items=var_27)
    var_29 = 1
    var_30 = 2
    var_31 = [var_29, var_30]
    var_32 = var_28.validate(var_31)
    var_33 = exc_info.value.messages()[var_32]
    var_34 = var_33.code
    assert var_34 == 'min_items'
    var_35 = module_0.Array(max_items=var_20)
    var_36 = 1
    var_37 = 2
    var_38 = 3
    var_39 = [var_36, var_37, var_38]
    var_40 = var_35.validate(var_39)
    var_41 = exc_info.value.messages()[var_39]
    var_42 = var_41.code
    assert var_42 == 'max_items'
    var_43 = module_0.Array(unique_items=var_36)
    var_44 = 1
    var_45 = 2
    var_46 = [var_44, var_45, var_44]
    var_47 = var_43.validate(var_46)
    var_48 = exc_info.value.messages()[var_47]
    var_49 = var_48.code
    assert var_49 == 'unique_items'
    var_50 = module_0.Integer()
    var_51 = module_0.Array(var_50)
    var_52 = [var_44, var_20, var_27]
    var_53 = var_51.validate(var_52)
    var_54 = module_0.Integer()
    var_55 = module_0.Array(var_54)
    var_56 = 1
    var_57 = 'invalid'
    var_58 = 3
    var_59 = [var_56, var_57, var_58]
    var_60 = var_55.validate(var_59)
    var_61 = module_0.Integer()
    var_62 = module_0.String()
    var_63 = [var_61, var_62]
    var_64 = module_0.Array(var_63)
    var_65 = 'hello'
    var_66 = [var_56, var_65]
    var_67 = var_64.validate(var_66)
    var_68 = module_0.Integer()
    var_69 = module_0.String()
    var_70 = [var_68, var_69]
    var_71 = module_0.Array(var_70, var_56)
    var_72 = 'extra'
    var_73 = [var_56, var_65, var_72]
    var_74 = var_71.validate(var_73)
    var_75 = module_0.Integer()
    var_76 = [var_75]
    var_77 = module_0.String()
    var_78 = module_0.Array(var_76, var_77)
    var_79 = [var_56, var_72]
    var_80 = var_78.validate(var_79)
    var_81 = module_0.Integer()
    var_82 = [var_81]
    var_83 = module_0.Array(var_82, var_59)
    var_84 = 1
    var_85 = 2
    var_86 = [var_84, var_85]
    var_87 = var_83.validate(var_86)
    var_88 = module_0.Array()
    var_89 = []
    var_90 = var_88.validate(var_89)
    var_91 = 5
    var_92 = module_0.Array(min_items=var_84, max_items=var_91)
    var_93 = [var_84, var_20, var_27]
    var_94 = var_92.validate(var_93)
    var_95 = module_0.Array(unique_items=var_84)
    var_96 = [var_84, var_20, var_27]
    var_97 = var_95.validate(var_96)
    var_98 = module_0.String(min_length=var_20)
    var_99 = module_0.Array(var_98)
    var_100 = 'ab'
    var_101 = 'cd'
    var_102 = [var_100, var_101]
    var_103 = var_99.validate(var_102)
    var_104 = module_0.String(min_length=var_20)
    var_105 = module_0.Array(var_104)
    var_106 = 'a'
    var_107 = 'cd'
    var_108 = [var_106, var_107]
    var_109 = var_105.validate(var_108)
    var_110 = module_0.Array()
    var_111 = 'mixed'
    var_112 = 'key'
    var_113 = 'value'
    var_114 = {var_112: var_113}
    var_115 = [var_106, var_111, var_107, var_114]
    var_116 = var_110.validate(var_115)



# Parsed testcases at query #7
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'test_value'
    var_1 = module_0.Const(var_0)
    var_2 = var_1.validate(var_0)
    assert var_2 == 'test_value'
    var_3 = 42
    var_4 = module_0.Const(var_3)
    var_5 = var_4.validate(var_3)
    assert var_5 == 42
    var_6 = None
    var_7 = module_0.Const(var_6)
    var_8 = var_7.validate(var_6)
    assert var_8 is None
    var_9 = True
    var_10 = module_0.Const(var_9)
    var_11 = var_10.validate(var_9)
    assert var_11 is True
    var_12 = 'key'
    var_13 = 'value'
    var_14 = {var_12: var_13}
    var_15 = module_0.Const(var_14)
    var_16 = {var_12: var_13}
    var_17 = var_15.validate(var_16)
    var_18 = 'expected'
    var_19 = module_0.Const(var_18)
    var_20 = 'unexpected'
    var_21 = var_19.validate(var_20)
    var_22 = 0
    var_23 = exc_info.value.messages()[var_22]
    var_24 = var_23.code
    assert var_24 == 'const'
    var_25 = 10
    var_26 = module_0.Const(var_25)
    var_27 = 20
    var_28 = var_26.validate(var_27)
    var_29 = exc_info.value.messages()[var_22]
    var_30 = var_29.code
    assert var_30 == 'const'
    var_31 = module_0.Const(var_6)
    var_32 = 'not_none'
    var_33 = var_31.validate(var_32)
    var_34 = exc_info.value.messages()[var_22]
    var_35 = var_34.code
    assert var_35 == 'only_null'
    var_36 = module_0.Const(var_6)
    var_37 = var_36.validate(var_6)
    assert var_37 is None
    var_38 = module_0.Const(var_13)
    var_39 = None
    var_40 = var_38.validate(var_39)
    var_41 = exc_info.value.messages()[var_22]
    var_42 = var_41.code
    assert var_42 == 'only_null'
    var_43 = ''
    var_44 = module_0.Const(var_43)
    var_45 = var_44.validate(var_43)
    assert var_45 == ''
    var_46 = module_0.Const(var_22)
    var_47 = var_46.validate(var_22)
    assert var_47 == 0
    var_48 = 2
    var_49 = 3
    var_50 = [var_9, var_48, var_49]
    var_51 = module_0.Const(var_50)
    var_52 = [var_9, var_48, var_49]
    var_53 = var_51.validate(var_52)
    var_54 = [var_9, var_48, var_49]
    var_55 = module_0.Const(var_54)
    var_56 = 1
    var_57 = 2
    var_58 = [var_56, var_57]
    var_59 = var_55.validate(var_58)
    var_60 = exc_info.value.messages()[var_22]
    var_61 = var_60.code
    assert var_61 == 'const'



# Parsed testcases at query #8
#--------------------------


import typesystem.fields as module_0

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
    var_8 = exc_info.value.messages()[var_4]
    var_9 = var_8.code
    assert var_9 == 'null'
    var_10 = module_0.Array()
    var_11 = 'not a list'
    var_12 = var_10.validate(var_11)
    var_13 = exc_info.value.messages()[var_4]
    var_14 = var_13.code
    assert var_14 == 'type'
    var_15 = module_0.Array()
    var_16 = 'key'
    var_17 = 'value'
    var_18 = {var_16: var_17}
    var_19 = var_15.validate(var_18)
    var_20 = exc_info.value.messages()[var_19]
    var_21 = var_20.code
    assert var_21 == 'type'
    var_22 = 2
    var_23 = module_0.Array(exact_items=var_22)
    var_24 = 1
    var_25 = 2
    var_26 = 3
    var_27 = [var_24, var_25, var_26]
    var_28 = var_23.validate(var_27)
    var_29 = exc_info.value.messages()[var_27]
    var_30 = var_29.code
    assert var_30 == 'exact_items'
    var_31 = module_0.Array(exact_items=var_22)
    var_32 = [var_24, var_22]
    var_33 = var_31.validate(var_32)
    var_34 = module_0.Array(min_items=var_22)
    var_35 = 1
    var_36 = [var_35]
    var_37 = var_34.validate(var_36)
    var_38 = exc_info.value.messages()[var_27]
    var_39 = var_38.code
    assert var_39 == 'min_items'
    var_40 = module_0.Array(min_items=var_35)
    var_41 = []
    var_42 = var_40.validate(var_41)
    var_43 = exc_info.value.messages()[var_27]
    var_44 = var_43.code
    assert var_44 == 'empty'
    var_45 = module_0.Array(max_items=var_22)
    var_46 = 1
    var_47 = 2
    var_48 = 3
    var_49 = [var_46, var_47, var_48]
    var_50 = var_45.validate(var_49)
    var_51 = exc_info.value.messages()[var_49]
    var_52 = var_51.code
    assert var_52 == 'max_items'
    var_53 = module_0.Array(min_items=var_46)
    var_54 = 3
    var_55 = [var_46, var_22, var_54]
    var_56 = var_53.validate(var_55)
    var_57 = module_0.Array(max_items=var_54)
    var_58 = [var_46, var_22]
    var_59 = var_57.validate(var_58)
    var_60 = module_0.Integer()
    var_61 = module_0.Array(var_60)
    var_62 = [var_46, var_22, var_54]
    var_63 = var_61.validate(var_62)
    var_64 = module_0.Integer()
    var_65 = module_0.Array(var_64)
    var_66 = 1
    var_67 = 'invalid'
    var_68 = 3
    var_69 = [var_66, var_67, var_68]
    var_70 = var_65.validate(var_69)
    var_71 = exc_info.value.messages()[var_69]
    var_72 = var_71.index
    var_73 = module_0.Integer()
    var_74 = module_0.String()
    var_75 = [var_73, var_74]
    var_76 = module_0.Array(var_75)
    var_77 = 'hello'
    var_78 = [var_66, var_77]
    var_79 = var_76.validate(var_78)
    var_80 = module_0.Integer()
    var_81 = module_0.String()
    var_82 = [var_80, var_81]
    var_83 = module_0.Array(var_82)
    var_84 = 1
    var_85 = 123
    var_86 = [var_84, var_85]
    var_87 = var_83.validate(var_86)
    var_88 = exc_info.value.messages()[var_87]
    var_89 = var_88.index
    var_90 = module_0.Integer()
    var_91 = [var_90]
    var_92 = module_0.Array(var_91, var_84)
    var_93 = 'extra'
    var_94 = 'more'
    var_95 = [var_84, var_93, var_94]
    var_96 = var_92.validate(var_95)
    var_97 = module_0.Integer()
    var_98 = [var_97]
    var_99 = module_0.Array(var_98, var_87)
    var_100 = 1
    var_101 = 'extra'
    var_102 = [var_100, var_101]
    var_103 = var_99.validate(var_102)
    var_104 = exc_info.value.messages()[var_103]
    var_105 = var_104.index
    var_106 = module_0.Integer()
    var_107 = [var_106]
    var_108 = module_0.String()
    var_109 = module_0.Array(var_107, var_108)
    var_110 = [var_100, var_93, var_94]
    var_111 = var_109.validate(var_110)
    var_112 = module_0.Integer()
    var_113 = [var_112]
    var_114 = module_0.String()
    var_115 = module_0.Array(var_113, var_114)
    var_116 = 1
    var_117 = 123
    var_118 = [var_116, var_117]
    var_119 = var_115.validate(var_118)
    var_120 = exc_info.value.messages()[var_119]
    var_121 = var_120.index
    var_122 = module_0.Integer()
    var_123 = module_0.Array(var_122, unique_items=var_116)
    var_124 = [var_116, var_22, var_54]
    var_125 = var_123.validate(var_124)
    var_126 = module_0.Integer()
    var_127 = module_0.Array(var_126, unique_items=var_116)
    var_128 = 1
    var_129 = 2
    var_130 = [var_128, var_129, var_128]
    var_131 = var_127.validate(var_130)
    var_132 = exc_info.value.messages()[var_131]
    var_133 = var_132.code
    assert var_133 == 'unique_items'
    var_134 = module_0.Array()
    var_135 = []
    var_136 = var_134.validate(var_135)
    var_137 = 'name'
    var_138 = module_0.String()
    var_139 = {var_137: var_138}
    var_140 = module_0.Object(properties=var_139)
    var_141 = module_0.Array(var_140)
    var_142 = 'John'
    var_143 = {var_137: var_142}
    var_144 = 'Jane'
    var_145 = {var_137: var_144}
    var_146 = [var_143, var_145]
    var_147 = var_141.validate(var_146)
    var_148 = module_0.Array(var_129)
    var_149 = 'string'
    var_150 = 'key'
    var_151 = 'value'
    var_152 = {var_150: var_151}
    var_153 = [var_128, var_149, var_152]
    var_154 = var_148.validate(var_153)
    var_155 = module_0.String()
    var_156 = module_0.Array(var_155, unique_items=var_128)
    var_157 = 'a'
    var_158 = 'b'
    var_159 = 'c'
    var_160 = [var_157, var_158, var_159]
    var_161 = var_156.validate(var_160)
    var_162 = module_0.String()
    var_163 = module_0.Array(var_162, unique_items=var_128)
    var_164 = 'a'
    var_165 = 'b'
    var_166 = [var_164, var_165, var_164]
    var_167 = var_163.validate(var_166)
    var_168 = exc_info.value.messages()[var_167]
    var_169 = var_168.code
    assert var_169 == 'unique_items'



# Parsed testcases at query #9
#--------------------------


import typesystem.fields as module_0

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
    var_16 = 'invalid_key'
    var_17 = 'name'
    var_18 = 'age'
    var_19 = module_0.String()
    var_20 = module_0.Integer()
    var_21 = {var_17: var_19, var_18: var_20}
    var_22 = module_0.Object(properties=var_21)
    var_23 = 'John'
    var_24 = 30
    var_25 = {var_17: var_23, var_18: var_24}
    var_26 = var_22.validate(var_25)
    var_27 = module_0.String()
    var_28 = {var_17: var_27}
    var_29 = [var_17]
    var_30 = module_0.Object(properties=var_28, required=var_29)
    var_31 = {}
    var_32 = var_30.validate(var_31)
    var_33 = 'required'
    var_34 = 'Unknown'
    var_35 = module_0.String()
    var_36 = {var_17: var_35}
    var_37 = module_0.Object(properties=var_36)
    var_38 = {}
    var_39 = var_37.validate(var_38)
    var_40 = 2
    var_41 = module_0.Object(min_properties=var_40)
    var_42 = 'key'
    var_43 = 'value'
    var_44 = {var_42: var_43}
    var_45 = var_41.validate(var_44)
    var_46 = module_0.Object(min_properties=var_42)
    var_47 = {}
    var_48 = var_46.validate(var_47)
    var_49 = module_0.Object(max_properties=var_47)
    var_50 = 'key1'
    var_51 = 'key2'
    var_52 = 'value1'
    var_53 = 'value2'
    var_54 = {var_50: var_52, var_51: var_53}
    var_55 = var_49.validate(var_54)
    var_56 = module_0.Object(additional_properties=var_50)
    var_57 = 'extra'
    var_58 = 'field'
    var_59 = {var_57: var_58}
    var_60 = var_56.validate(var_59)
    var_61 = module_0.Object(additional_properties=var_53)
    var_62 = 'extra'
    var_63 = 'field'
    var_64 = {var_62: var_63}
    var_65 = var_61.validate(var_64)
    var_66 = 'invalid_property'
    var_67 = module_0.String()
    var_68 = module_0.Object(additional_properties=var_67)
    var_69 = {var_57: var_58}
    var_70 = var_68.validate(var_69)
    var_71 = '^S_'
    var_72 = module_0.String()
    var_73 = {var_71: var_72}
    var_74 = module_0.Object(pattern_properties=var_73)
    var_75 = 'S_name'
    var_76 = 'value'
    var_77 = {var_75: var_76}
    var_78 = var_74.validate(var_77)
    var_79 = '^[a-z]+$'
    var_80 = module_0.String(pattern=var_79)
    var_81 = module_0.Object(property_names=var_80)
    var_82 = 'Invalid'
    var_83 = 'value'
    var_84 = {var_82: var_83}
    var_85 = var_81.validate(var_84)
    var_86 = module_0.Integer()
    var_87 = {var_18: var_86}
    var_88 = module_0.Object(properties=var_87)
    var_89 = 'age'
    var_90 = 'not an integer'
    var_91 = {var_89: var_90}
    var_92 = var_88.validate(var_91)
    var_93 = module_0.Object()
    var_94 = 'key'
    var_95 = (var_94, var_76)
    var_96 = [var_95]



# Parsed testcases at query #10
#--------------------------


import typesystem.fields as module_0

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
    var_12 = module_0.Array(exact_items=var_11)
    var_13 = [var_9, var_11]
    var_14 = var_12.validate(var_13)
    var_15 = module_0.Array(exact_items=var_11)
    var_16 = 1
    var_17 = 2
    var_18 = 3
    var_19 = [var_16, var_17, var_18]
    var_20 = var_15.validate(var_19)
    var_21 = module_0.Array(min_items=var_11)
    var_22 = 1
    var_23 = [var_22]
    var_24 = var_21.validate(var_23)
    var_25 = module_0.Array(min_items=var_22)
    var_26 = []
    var_27 = var_25.validate(var_26)
    var_28 = module_0.Array(max_items=var_11)
    var_29 = 1
    var_30 = 2
    var_31 = 3
    var_32 = [var_29, var_30, var_31]
    var_33 = var_28.validate(var_32)
    var_34 = module_0.Integer()
    var_35 = module_0.Array(var_34)
    var_36 = 3
    var_37 = [var_29, var_11, var_36]
    var_38 = var_35.validate(var_37)
    var_39 = module_0.Integer()
    var_40 = module_0.Array(var_39)
    var_41 = 1
    var_42 = 'invalid'
    var_43 = 3
    var_44 = [var_41, var_42, var_43]
    var_45 = var_40.validate(var_44)
    var_46 = module_0.Integer()
    var_47 = module_0.String()
    var_48 = [var_46, var_47]
    var_49 = module_0.Array(var_48)
    var_50 = 'test'
    var_51 = [var_41, var_50]
    var_52 = var_49.validate(var_51)
    var_53 = module_0.Integer()
    var_54 = module_0.String()
    var_55 = [var_53, var_54]
    var_56 = module_0.Array(var_55, var_44)
    var_57 = 1
    var_58 = 'test'
    var_59 = 3
    var_60 = [var_57, var_58, var_59]
    var_61 = var_56.validate(var_60)
    var_62 = module_0.Integer()
    var_63 = module_0.String()
    var_64 = [var_62, var_63]
    var_65 = module_0.Integer()
    var_66 = module_0.Array(var_64, var_65)
    var_67 = [var_57, var_50, var_36]
    var_68 = var_66.validate(var_67)
    var_69 = module_0.Array(unique_items=var_57)
    var_70 = 1
    var_71 = 2
    var_72 = [var_70, var_71, var_70]
    var_73 = var_69.validate(var_72)
    var_74 = module_0.Array(unique_items=var_70)
    var_75 = [var_70, var_11, var_36]
    var_76 = var_74.validate(var_75)
    var_77 = module_0.Array()
    var_78 = []
    var_79 = var_77.validate(var_78)
    var_80 = module_0.Integer()
    var_81 = module_0.Array(var_80)
    var_82 = module_0.Array(var_81)
    var_83 = [var_70, var_11]
    var_84 = 4
    var_85 = [var_36, var_84]
    var_86 = [var_83, var_85]
    var_87 = var_82.validate(var_86)
    var_88 = module_0.Integer()
    var_89 = module_0.Array(var_88)
    var_90 = module_0.Array(var_89)
    var_91 = 1
    var_92 = 2
    var_93 = [var_91, var_92]
    var_94 = 3
    var_95 = 'invalid'
    var_96 = [var_94, var_95]
    var_97 = [var_93, var_96]
    var_98 = var_90.validate(var_97)
    var_99 = module_0.Integer()
    var_100 = [var_99]
    var_101 = module_0.Array(var_100, var_91)
    var_102 = 'anything'
    var_103 = [var_91, var_102, var_36]
    var_104 = var_101.validate(var_103)
    var_105 = module_0.Integer()
    var_106 = [var_105]
    var_107 = module_0.Array(var_106, var_94)
    var_108 = 1
    var_109 = 2
    var_110 = [var_108, var_109]
    var_111 = var_107.validate(var_110)
    var_112 = module_0.String()
    var_113 = module_0.Array(var_112)
    var_114 = 'a'
    var_115 = 'b'
    var_116 = 'c'
    var_117 = [var_114, var_115, var_116]
    var_118 = var_113.validate(var_117)
    var_119 = module_0.Array(min_items=var_97, max_items=var_84)
    var_120 = [var_108, var_97, var_36]
    var_121 = var_119.validate(var_120)
    var_122 = module_0.Array()
    var_123 = 'mixed'
    var_124 = [var_108, var_123, var_109]
    var_125 = var_122.validate(var_124)



# Parsed testcases at query #11
#--------------------------


import typesystem.fields as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'hello'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'hello'
    var_3 = True
    var_4 = module_0.String()
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None
    var_7 = module_0.String()
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = module_0.String(allow_blank=var_3)
    var_11 = ''
    var_12 = var_10.validate(var_11)
    assert var_12 == ''
    var_13 = False
    var_14 = module_0.String(allow_blank=var_13)
    var_15 = ''
    var_16 = var_14.validate(var_15)
    var_17 = module_0.String(trim_whitespace=var_3)
    var_18 = '  hello  '
    var_19 = var_17.validate(var_18)
    assert var_19 == 'hello'
    var_20 = module_0.String(trim_whitespace=var_13)
    var_21 = var_20.validate(var_18)
    assert var_21 == '  hello  '
    var_22 = 5
    var_23 = module_0.String(max_length=var_22)
    var_24 = var_23.validate(var_15)
    assert var_24 == 'hello'
    var_25 = 'toolong'
    var_26 = var_23.validate(var_25)
    var_27 = 3
    var_28 = module_0.String(min_length=var_27)
    var_29 = var_28.validate(var_25)
    assert var_29 == 'hello'
    var_30 = 'hi'
    var_31 = var_28.validate(var_30)
    var_32 = '^\\d+$'
    var_33 = module_0.String(pattern=var_32)
    var_34 = '12345'
    var_35 = var_33.validate(var_34)
    assert var_35 == '12345'
    var_36 = 'abc'
    var_37 = var_33.validate(var_36)
    var_38 = '^[a-z]+$'
    var_39 = module_1.compile(var_38)
    var_40 = module_0.String(pattern=var_39)
    var_41 = var_40.validate(var_36)
    assert var_41 == 'hello'
    var_42 = 'Hello123'
    var_43 = var_40.validate(var_42)
    var_44 = module_0.String()
    var_45 = 123
    var_46 = var_44.validate(var_45)
    var_47 = module_0.String()
    var_48 = 'hello\x00world'
    var_49 = var_47.validate(var_48)
    assert var_49 == 'helloworld'
    var_50 = module_0.String(coerce_types=var_3)
    var_51 = var_50.validate(var_11)
    assert var_51 is None
    var_52 = module_0.String(allow_blank=var_3, coerce_types=var_3)
    var_53 = var_52.validate(var_5)
    assert var_53 == ''
    var_54 = module_0.String(coerce_types=var_13)
    var_55 = ''
    var_56 = var_54.validate(var_55)
    var_57 = 'email'
    var_58 = module_0.String(format=var_57)
    var_59 = 'test@example.com'
    var_60 = var_58.validate(var_59)
    var_61 = 'uuid'
    var_62 = module_0.String(format=var_61)
    var_63 = '550e8400-e29b-41d4-a716-446655440000'
    var_64 = var_62.validate(var_63)
    var_65 = 'date'
    var_66 = module_0.String(format=var_65)
    var_67 = 2023



# Parsed testcases at query #12
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = var_0.get_default_value()
    assert var_1 is None
    var_2 = 'test_value'
    var_3 = module_0.Field(default=var_2)
    var_4 = var_3.get_default_value()
    assert var_4 == 'test_value'
    var_5 = 42
    var_6 = module_0.Field(default=var_5)
    var_7 = var_6.get_default_value()
    assert var_7 == 42
    var_8 = var_6.get_default_value()
    assert var_8 == 'callable_value'
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = [var_9, var_10, var_11]
    var_13 = lambda : var_12
    var_14 = module_0.Field(default=var_13)
    var_15 = var_14.get_default_value()
    var_16 = True
    var_17 = module_0.Field(allow_null=var_16)
    var_18 = var_17.get_default_value()
    assert var_18 is None
    var_19 = True
    var_20 = 'explicit'
    var_21 = module_0.Field(default=var_20, allow_null=var_19)
    var_22 = var_21.get_default_value()
    assert var_22 == 'explicit'
    var_23 = False
    var_24 = module_0.Field(default=var_23)
    var_25 = var_24.get_default_value()
    assert var_25 is False
    var_26 = []
    var_27 = module_0.Field(default=var_26)
    var_28 = var_27.get_default_value()
    var_29 = {}
    var_30 = module_0.Field(default=var_29)
    var_31 = var_30.get_default_value()
    var_32 = None
    var_33 = lambda : var_32
    var_34 = module_0.Field(default=var_33)
    var_35 = var_34.get_default_value()
    assert var_35 is None



# Parsed testcases at query #13
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Array(var_0)
    var_2 = None
    var_3 = var_1.serialize(var_2)
    assert var_3 is None
    var_4 = module_0.String()
    var_5 = module_0.Array(var_4)
    var_6 = 'hello'
    var_7 = 'world'
    var_8 = [var_6, var_7]
    var_9 = var_5.serialize(var_8)
    var_10 = module_0.String()
    var_11 = module_0.Integer()
    var_12 = [var_10, var_11]
    var_13 = module_0.Array(var_12)
    var_14 = 42
    var_15 = [var_6, var_14]
    var_16 = var_13.serialize(var_15)
    var_17 = module_0.Array(var_2)
    var_18 = 1
    var_19 = 'test'
    var_20 = 3.14
    var_21 = [var_18, var_19, var_20]
    var_22 = var_17.serialize(var_21)
    var_23 = module_0.Integer()
    var_24 = module_0.Array(var_23)
    var_25 = 2
    var_26 = 3
    var_27 = [var_18, var_25, var_26]
    var_28 = var_24.serialize(var_27)
    var_29 = module_0.Decimal()
    var_30 = module_0.Array(var_29)
    var_31 = '1.5'
    var_32 = '2.5'
    var_33 = module_0.Integer()
    var_34 = module_0.Array(var_33)
    var_35 = module_0.Array(var_34)
    var_36 = [var_18, var_25]
    var_37 = 4
    var_38 = [var_26, var_37]
    var_39 = [var_36, var_38]
    var_40 = var_35.serialize(var_39)
    var_41 = 'name'
    var_42 = 'age'
    var_43 = module_0.String()
    var_44 = module_0.Integer()
    var_45 = {var_41: var_43, var_42: var_44}
    var_46 = module_0.Object(properties=var_45)
    var_47 = module_0.Array(var_46)
    var_48 = 'Alice'
    var_49 = 30
    var_50 = {var_41: var_48, var_42: var_49}
    var_51 = [var_50]
    var_52 = var_47.serialize(var_51)
    var_53 = module_0.Boolean()
    var_54 = module_0.Array(var_53)
    var_55 = True
    var_56 = False
    var_57 = True
    var_58 = [var_55, var_56, var_57]
    var_59 = var_54.serialize(var_58)
    var_60 = module_0.Array(var_2)
    var_61 = 'string'
    var_62 = True
    var_63 = [var_57, var_61, var_20, var_62, var_2]
    var_64 = var_60.serialize(var_63)
    var_65 = module_0.String()
    var_66 = module_0.Array(var_65)
    var_67 = []
    var_68 = var_66.serialize(var_67)
    var_69 = module_0.String()
    var_70 = module_0.Integer()
    var_71 = module_0.Boolean()
    var_72 = [var_69, var_70, var_71]
    var_73 = module_0.Array(var_72)
    var_74 = True
    var_75 = [var_19, var_14, var_74]
    var_76 = var_73.serialize(var_75)
    var_77 = module_0.Decimal()
    var_78 = module_0.Integer()
    var_79 = [var_77, var_78]
    var_80 = module_0.Array(var_79)
    var_81 = 10



# Parsed testcases at query #14
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = True
    var_5 = module_0.String()
    var_6 = module_0.Integer()
    var_7 = [var_5, var_6]
    var_8 = module_0.Union(var_7)
    var_9 = module_0.String()
    var_10 = module_0.Integer()
    var_11 = [var_9, var_10]
    var_12 = module_0.Union(var_11)
    var_13 = False
    var_14 = module_0.String()
    var_15 = module_0.Integer()
    var_16 = [var_14, var_15]
    var_17 = module_0.Union(var_16)
    var_18 = module_0.String()
    var_19 = module_0.Integer()
    var_20 = [var_18, var_19]
    var_21 = module_0.Union(var_20)
    var_22 = module_0.String()
    var_23 = [var_22]
    var_24 = module_0.Union(var_23)
    var_25 = module_0.String()
    var_26 = module_0.Integer()
    var_27 = module_0.Float()
    var_28 = module_0.Boolean()
    var_29 = [var_25, var_26, var_27, var_28]
    var_30 = module_0.Union(var_29)
    var_31 = var_30.any_of
    var_32 = len(var_31)
    assert var_32 == 4
    var_33 = module_0.String()
    var_34 = module_0.Integer()
    var_35 = module_0.Float()
    var_36 = [var_33, var_34, var_35]
    var_37 = module_0.Union(var_36)



# Parsed testcases at query #15
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = var_0.get_default_value()
    assert var_1 is None
    var_2 = 'test_value'
    var_3 = module_0.Field(default=var_2)
    var_4 = var_3.get_default_value()
    assert var_4 == 'test_value'
    var_5 = 42
    var_6 = module_0.Field(default=var_5)
    var_7 = var_6.get_default_value()
    assert var_7 == 42
    var_8 = var_6.get_default_value()
    assert var_8 == 'generated_value'
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = [var_9, var_10, var_11]
    var_13 = lambda : var_12
    var_14 = module_0.Field(default=var_13)
    var_15 = var_14.get_default_value()
    var_16 = None
    var_17 = module_0.Field(default=var_16)
    var_18 = var_17.get_default_value()
    assert var_18 is None
    var_19 = True
    var_20 = module_0.Field(allow_null=var_19)
    var_21 = var_20.get_default_value()
    assert var_21 is None
    var_22 = 0
    var_23 = module_0.Field(default=var_22)
    var_24 = var_23.get_default_value()
    assert var_24 == 0
    var_25 = False
    var_26 = module_0.Field(default=var_25)
    var_27 = var_26.get_default_value()
    assert var_27 is False
    var_28 = ''
    var_29 = module_0.Field(default=var_28)
    var_30 = var_29.get_default_value()
    assert var_30 == ''
    var_31 = lambda : var_16
    var_32 = module_0.Field(default=var_31)
    var_33 = var_32.get_default_value()
    assert var_33 is None



# Parsed testcases at query #16
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Object()
    var_1 = 'name'
    var_2 = 'age'
    var_3 = module_0.String()
    var_4 = module_0.Integer()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.Object(properties=var_5)
    var_7 = var_6.properties
    var_8 = len(var_7)
    assert var_8 == 2
    var_9 = '^S_'
    var_10 = '^I_'
    var_11 = module_0.String()
    var_12 = module_0.Integer()
    var_13 = {var_9: var_11, var_10: var_12}
    var_14 = module_0.Object(pattern_properties=var_13)
    var_15 = var_14.pattern_properties
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = False
    var_18 = module_0.Object(additional_properties=var_17)
    var_19 = True
    var_20 = module_0.Object(additional_properties=var_19)
    var_21 = module_0.String()
    var_22 = module_0.Object(additional_properties=var_21)
    var_23 = '^[a-z]+$'
    var_24 = module_0.String(pattern=var_23)
    var_25 = module_0.Object(property_names=var_24)
    var_26 = 10
    var_27 = module_0.Object(min_properties=var_19, max_properties=var_26)
    var_28 = 'email'
    var_29 = [var_1, var_28]
    var_30 = module_0.Object(required=var_29)
    var_31 = (var_1, var_28)
    var_32 = module_0.Object(required=var_31)
    var_33 = module_0.String()
    var_34 = module_0.Integer()
    var_35 = {var_1: var_33, var_2: var_34}
    var_36 = '^meta_'
    var_37 = module_0.String()
    var_38 = {var_36: var_37}
    var_39 = [var_1]
    var_40 = module_0.String()
    var_41 = 20
    var_42 = 'User'
    var_43 = 'A user object'
    var_44 = module_0.Object(properties=var_35, pattern_properties=var_38, additional_properties=var_17, property_names=var_40, min_properties=var_19, max_properties=var_41, required=var_39)
    var_45 = module_0.String()
    var_46 = {var_1: var_45}
    var_47 = module_0.Object(properties=var_46)
    var_48 = var_47.properties
    var_49 = len(var_48)
    assert var_49 == 1
    var_50 = module_0.String()
    var_51 = None
    var_52 = module_0.Object(required=var_51)
    var_53 = 123
    var_54 = module_0.String()
    var_55 = {var_53: var_54}
    var_56 = module_0.Object(properties=var_55)
    var_57 = 'name'
    var_58 = 'not a field'
    var_59 = {var_57: var_58}
    var_60 = module_0.Object(properties=var_59)
    var_61 = 123
    var_62 = module_0.String()
    var_63 = {var_61: var_62}
    var_64 = module_0.Object(pattern_properties=var_63)
    var_65 = '^pattern'
    var_66 = 'not a field'
    var_67 = {var_65: var_66}
    var_68 = module_0.Object(pattern_properties=var_67)
    var_69 = 'invalid'
    var_70 = module_0.Object(additional_properties=var_69)
    var_71 = 'not an int'
    var_72 = module_0.Object(min_properties=var_71)
    var_73 = 'not an int'
    var_74 = module_0.Object(max_properties=var_73)
    var_75 = 123
    var_76 = 'valid'
    var_77 = [var_75, var_76]
    var_78 = module_0.Object(required=var_77)



# Parsed testcases at query #17
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Array()
    var_1 = module_0.String()
    var_2 = module_0.Array(var_1)
    var_3 = module_0.String()
    var_4 = module_0.Integer()
    var_5 = [var_3, var_4]
    var_6 = module_0.Array(var_5)
    var_7 = module_0.String()
    var_8 = module_0.Integer()
    var_9 = [var_7, var_8]
    var_10 = True
    var_11 = module_0.Array(var_9, var_10)
    var_12 = module_0.Boolean()
    var_13 = module_0.Array(var_9, var_12)
    var_14 = module_0.String()
    var_15 = module_0.Integer()
    var_16 = (var_14, var_15)
    var_17 = module_0.Array(var_16)
    var_18 = list(var_16)
    var_19 = module_0.String()
    var_20 = 5
    var_21 = module_0.Array(var_19, min_items=var_20)
    var_22 = module_0.String()
    var_23 = 10
    var_24 = module_0.Array(var_22, max_items=var_23)
    var_25 = module_0.String()
    var_26 = 7
    var_27 = module_0.Array(var_25, exact_items=var_26)
    var_28 = module_0.String()
    var_29 = module_0.Array(var_28, unique_items=var_10)
    var_30 = module_0.Integer()
    var_31 = False
    var_32 = module_0.Array(var_30, var_31, var_10, var_20, unique_items=var_10)
    var_33 = module_0.String()
    var_34 = 2
    var_35 = module_0.Array(var_33, min_items=var_34, max_items=var_23, exact_items=var_20)
    var_36 = module_0.String()
    var_37 = module_0.Integer()
    var_38 = module_0.Boolean()
    var_39 = [var_36, var_37, var_38]
    var_40 = module_0.Array(var_39, var_31)
    var_41 = module_0.Array(var_39, var_10)
    var_42 = module_0.Array(var_39, var_31, var_10, var_23)



# Parsed testcases at query #18
#--------------------------


import typesystem.fields as module_0

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
    var_16 = module_0.Object()
    var_17 = {}
    var_18 = var_16.validate(var_17)
    var_19 = 'name'
    var_20 = 'age'
    var_21 = module_0.String()
    var_22 = module_0.Integer()
    var_23 = {var_19: var_21, var_20: var_22}
    var_24 = module_0.Object(properties=var_23)
    var_25 = 'John'
    var_26 = 30
    var_27 = {var_19: var_25, var_20: var_26}
    var_28 = var_24.validate(var_27)
    var_29 = module_0.String()
    var_30 = {var_19: var_29}
    var_31 = [var_19]
    var_32 = module_0.Object(properties=var_30, required=var_31)
    var_33 = {}
    var_34 = var_32.validate(var_33)
    var_35 = module_0.String()
    var_36 = {var_19: var_35}
    var_37 = [var_19]
    var_38 = module_0.Object(properties=var_36, required=var_37)
    var_39 = {var_19: var_25}
    var_40 = var_38.validate(var_39)
    var_41 = 'Unknown'
    var_42 = module_0.String()
    var_43 = {var_19: var_42}
    var_44 = module_0.Object(properties=var_43)
    var_45 = {}
    var_46 = var_44.validate(var_45)
    var_47 = '^S_'
    var_48 = module_0.String()
    var_49 = {var_47: var_48}
    var_50 = module_0.Object(pattern_properties=var_49)
    var_51 = 'S_name'
    var_52 = {var_51: var_25}
    var_53 = var_50.validate(var_52)
    var_54 = module_0.Object(additional_properties=var_33)
    var_55 = 'extra'
    var_56 = 'value'
    var_57 = {var_55: var_56}
    var_58 = var_54.validate(var_57)
    var_59 = module_0.Object(additional_properties=var_15)
    var_60 = 'extra'
    var_61 = 'value'
    var_62 = {var_60: var_61}
    var_63 = var_59.validate(var_62)
    var_64 = module_0.String()
    var_65 = module_0.Object(additional_properties=var_64)
    var_66 = {var_55: var_56}
    var_67 = var_65.validate(var_66)
    var_68 = 2
    var_69 = module_0.Object(min_properties=var_68)
    var_70 = 'a'
    var_71 = 1
    var_72 = {var_70: var_71}
    var_73 = var_69.validate(var_72)
    var_74 = module_0.Object(min_properties=var_70)
    var_75 = {}
    var_76 = var_74.validate(var_75)
    var_77 = module_0.Object(max_properties=var_75)
    var_78 = 'a'
    var_79 = 'b'
    var_80 = 1
    var_81 = 2
    var_82 = {var_78: var_80, var_79: var_81}
    var_83 = var_77.validate(var_82)
    var_84 = '^[a-z]+$'
    var_85 = module_0.String(pattern=var_84)
    var_86 = module_0.Object(property_names=var_85)
    var_87 = 'Invalid'
    var_88 = 'value'
    var_89 = {var_87: var_88}
    var_90 = var_86.validate(var_89)
    var_91 = module_0.Integer(minimum=var_90)
    var_92 = {var_20: var_91}
    var_93 = module_0.Object(properties=var_92)
    var_94 = 'age'
    var_95 = -5
    var_96 = {var_94: var_95}
    var_97 = var_93.validate(var_96)
    var_98 = module_0.Integer(minimum=var_97)
    var_99 = {var_20: var_98}
    var_100 = module_0.Object(properties=var_99)
    var_101 = 25
    var_102 = {var_20: var_101}
    var_103 = var_100.validate(var_102)
    var_104 = module_0.String()
    var_105 = module_0.Integer(minimum=var_97)
    var_106 = {var_83: var_104, var_20: var_105}
    var_107 = [var_83]
    var_108 = 3
    var_109 = module_0.Object(properties=var_106, additional_properties=var_97, min_properties=var_94, max_properties=var_108, required=var_107)
    var_110 = {var_83: var_25, var_20: var_26}
    var_111 = var_109.validate(var_110)
    var_112 = module_0.Object()
    var_113 = 'key'
    var_114 = (var_113, var_56)
    var_115 = [var_114]



# Parsed testcases at query #19
#--------------------------


import typesystem.fields as module_0

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
    var_8 = exc_info.value.messages()[var_4]
    var_9 = var_8.code
    assert var_9 == 'null'
    var_10 = module_0.Array()
    var_11 = 'not a list'
    var_12 = var_10.validate(var_11)
    var_13 = exc_info.value.messages()[var_4]
    var_14 = var_13.code
    assert var_14 == 'type'
    var_15 = 3
    var_16 = module_0.Array(exact_items=var_15)
    var_17 = 1
    var_18 = 2
    var_19 = [var_17, var_18]
    var_20 = var_16.validate(var_19)
    var_21 = exc_info.value.messages()[var_20]
    var_22 = var_21.code
    assert var_22 == 'exact_items'
    var_23 = 2
    var_24 = module_0.Array(min_items=var_23)
    var_25 = 1
    var_26 = [var_25]
    var_27 = var_24.validate(var_26)
    var_28 = exc_info.value.messages()[var_20]
    var_29 = var_28.code
    assert var_29 == 'min_items'
    var_30 = module_0.Array(min_items=var_25)
    var_31 = []
    var_32 = var_30.validate(var_31)
    var_33 = exc_info.value.messages()[var_20]
    var_34 = var_33.code
    assert var_34 == 'empty'
    var_35 = module_0.Array(max_items=var_23)
    var_36 = 1
    var_37 = 2
    var_38 = 3
    var_39 = [var_36, var_37, var_38]
    var_40 = var_35.validate(var_39)
    var_41 = exc_info.value.messages()[var_39]
    var_42 = var_41.code
    assert var_42 == 'max_items'
    var_43 = module_0.Integer()
    var_44 = module_0.Array(var_43)
    var_45 = [var_36, var_23, var_15]
    var_46 = var_44.validate(var_45)
    var_47 = module_0.Integer()
    var_48 = module_0.Array(var_47)
    var_49 = 1
    var_50 = 'not an int'
    var_51 = 3
    var_52 = [var_49, var_50, var_51]
    var_53 = var_48.validate(var_52)
    var_54 = module_0.Integer()
    var_55 = module_0.String()
    var_56 = [var_54, var_55]
    var_57 = module_0.Array(var_56)
    var_58 = 'hello'
    var_59 = [var_49, var_58]
    var_60 = var_57.validate(var_59)
    var_61 = module_0.Integer()
    var_62 = module_0.String()
    var_63 = [var_61, var_62]
    var_64 = module_0.Array(var_63, var_49)
    var_65 = 'extra'
    var_66 = [var_49, var_58, var_65]
    var_67 = var_64.validate(var_66)
    var_68 = module_0.Integer()
    var_69 = module_0.String()
    var_70 = [var_68, var_69]
    var_71 = module_0.Array(var_70, var_52)
    var_72 = 1
    var_73 = 'hello'
    var_74 = 'extra'
    var_75 = [var_72, var_73, var_74]
    var_76 = var_71.validate(var_75)
    var_77 = exc_info.value.messages()[var_75]
    var_78 = var_77.code
    assert var_78 == 'additional_items'
    var_79 = module_0.Integer()
    var_80 = module_0.String()
    var_81 = [var_79, var_80]
    var_82 = module_0.Integer()
    var_83 = module_0.Array(var_81, var_82)
    var_84 = 42
    var_85 = [var_72, var_58, var_84]
    var_86 = var_83.validate(var_85)
    var_87 = module_0.Integer()
    var_88 = module_0.Array(var_87, unique_items=var_72)
    var_89 = 1
    var_90 = 2
    var_91 = [var_89, var_90, var_89]
    var_92 = var_88.validate(var_91)
    var_93 = 'unique_items'
    var_94 = module_0.Integer()
    var_95 = module_0.Array(var_94, unique_items=var_89)
    var_96 = [var_89, var_23, var_15]
    var_97 = var_95.validate(var_96)
    var_98 = module_0.Array()
    var_99 = []
    var_100 = var_98.validate(var_99)
    var_101 = module_0.Array()
    var_102 = 'key'
    var_103 = 'value'
    var_104 = {var_102: var_103}
    var_105 = [var_89, var_58, var_104]
    var_106 = var_101.validate(var_105)
    var_107 = module_0.Integer()
    var_108 = module_0.Array(var_107, min_items=var_23)
    var_109 = 1
    var_110 = 'invalid'
    var_111 = [var_109, var_110]
    var_112 = var_108.validate(var_111)
    var_113 = module_0.Integer()
    var_114 = 5
    var_115 = module_0.Array(var_113, min_items=var_109, max_items=var_114, unique_items=var_109)
    var_116 = [var_109, var_23, var_15]
    var_117 = var_115.validate(var_116)



# Parsed testcases at query #20
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'red'
    var_1 = 'green'
    var_2 = 'blue'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Choice(choices=var_3)
    var_5 = var_4.choices
    var_6 = len(var_5)
    assert var_6 == 3
    var_7 = 'r'
    var_8 = 'Red'
    var_9 = (var_7, var_8)
    var_10 = 'g'
    var_11 = 'Green'
    var_12 = (var_10, var_11)
    var_13 = 'b'
    var_14 = 'Blue'
    var_15 = (var_13, var_14)
    var_16 = [var_9, var_12, var_15]
    var_17 = module_0.Choice(choices=var_16)
    var_18 = var_17.choices
    var_19 = len(var_18)
    assert var_19 == 3
    var_20 = (var_10, var_11)
    var_21 = [var_0, var_20, var_2]
    var_22 = module_0.Choice(choices=var_21)
    var_23 = var_22.choices
    var_24 = len(var_23)
    assert var_24 == 3
    var_25 = []
    var_26 = module_0.Choice(choices=var_25)
    var_27 = var_26.choices
    var_28 = len(var_27)
    assert var_28 == 0
    var_29 = None
    var_30 = module_0.Choice(choices=var_29)
    var_31 = var_30.choices
    var_32 = len(var_31)
    assert var_32 == 0
    var_33 = 'a'
    var_34 = [var_33, var_13]
    var_35 = True
    var_36 = module_0.Choice(choices=var_34)
    var_37 = [var_33, var_13]
    var_38 = False
    var_39 = module_0.Choice(choices=var_37, coerce_types=var_38)
    var_40 = [var_33, var_13]
    var_41 = 'Select Option'
    var_42 = 'Choose one option'
    var_43 = module_0.Choice(choices=var_40)
    var_44 = [var_33, var_13]
    var_45 = module_0.Choice(choices=var_44)
    var_46 = [var_33, var_13]
    var_47 = module_0.Choice(choices=var_46)
    var_48 = 'x'
    var_49 = 'y'
    var_50 = 'z'
    var_51 = [var_48, var_49, var_50]
    var_52 = module_0.Choice(choices=var_51)
    var_53 = var_52.choices
    var_54 = len(var_53)
    assert var_54 == 3
    var_55 = '1'
    var_56 = 'Option 1'
    var_57 = (var_55, var_56)
    var_58 = '2'
    var_59 = 'Option 2'
    var_60 = (var_58, var_59)
    var_61 = [var_57, var_60]
    var_62 = 'Test Field'
    var_63 = module_0.Choice(choices=var_61, coerce_types=var_38)



# Parsed testcases at query #21
#--------------------------


import typesystem.fields as module_0

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
    var_10 = module_0.Boolean()
    var_11 = var_10.validate(var_4)
    assert var_11 is False
    var_12 = module_0.Boolean(coerce_types=var_6)
    var_13 = 'true'
    var_14 = var_12.validate(var_13)
    assert var_14 is True
    var_15 = module_0.Boolean(coerce_types=var_6)
    var_16 = 'false'
    var_17 = var_15.validate(var_16)
    assert var_17 is False
    var_18 = module_0.Boolean(coerce_types=var_6)
    var_19 = 'True'
    var_20 = var_18.validate(var_19)
    assert var_20 is True
    var_21 = module_0.Boolean(coerce_types=var_6)
    var_22 = 'on'
    var_23 = var_21.validate(var_22)
    assert var_23 is True
    var_24 = module_0.Boolean(coerce_types=var_6)
    var_25 = 'off'
    var_26 = var_24.validate(var_25)
    assert var_26 is False
    var_27 = module_0.Boolean(coerce_types=var_6)
    var_28 = var_27.validate(var_6)
    assert var_28 is True
    var_29 = module_0.Boolean(coerce_types=var_6)
    var_30 = var_29.validate(var_4)
    assert var_30 is False
    var_31 = module_0.Boolean(coerce_types=var_6)
    var_32 = ''
    var_33 = var_31.validate(var_32)
    assert var_33 is False
    var_34 = module_0.Boolean(coerce_types=var_6)
    var_35 = var_34.validate(var_32)
    assert var_35 is None
    var_36 = module_0.Boolean(coerce_types=var_6)
    var_37 = 'null'
    var_38 = var_36.validate(var_37)
    assert var_38 is None
    var_39 = module_0.Boolean(coerce_types=var_6)
    var_40 = 'none'
    var_41 = var_39.validate(var_40)
    assert var_41 is None
    var_42 = module_0.Boolean(coerce_types=var_6)
    var_43 = 'invalid'
    var_44 = var_42.validate(var_43)
    var_45 = module_0.Boolean(coerce_types=var_43)
    var_46 = []
    var_47 = var_45.validate(var_46)
    var_48 = module_0.Boolean(coerce_types=var_4)
    var_49 = 'true'
    var_50 = var_48.validate(var_49)
    var_51 = module_0.Boolean(coerce_types=var_49)
    var_52 = 1.5
    var_53 = var_51.validate(var_52)



# Parsed testcases at query #22
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Const(var_0)
    var_2 = None
    var_3 = module_0.Const(var_2)
    var_4 = 'hello'
    var_5 = module_0.Const(var_4)
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = module_0.Const(var_8)
    var_10 = 1
    var_11 = 2
    var_12 = 3
    var_13 = [var_10, var_11, var_12]
    var_14 = module_0.Const(var_13)
    var_15 = True
    var_16 = module_0.Const(var_15)
    var_17 = 42
    var_18 = True
    var_19 = module_0.Const(var_17)
    var_20 = 'test'
    var_21 = True
    var_22 = module_0.Const(var_20)
    var_23 = 3.14
    var_24 = module_0.Const(var_23)
    var_25 = ''
    var_26 = module_0.Const(var_25)



# Parsed testcases at query #23
#--------------------------


import typesystem.fields as module_0
import re as module_1
import uuid as module_2

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'hello'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'hello'
    var_3 = True
    var_4 = module_0.String()
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None
    var_7 = module_0.String()
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = module_0.String(allow_blank=var_3)
    var_11 = ''
    var_12 = var_10.validate(var_11)
    assert var_12 == ''
    var_13 = False
    var_14 = module_0.String(allow_blank=var_13)
    var_15 = ''
    var_16 = var_14.validate(var_15)
    var_17 = module_0.String()
    var_18 = 123
    var_19 = var_17.validate(var_18)
    var_20 = module_0.String(trim_whitespace=var_3)
    var_21 = '  hello  '
    var_22 = var_20.validate(var_21)
    assert var_22 == 'hello'
    var_23 = module_0.String(trim_whitespace=var_13)
    var_24 = var_23.validate(var_21)
    assert var_24 == '  hello  '
    var_25 = 5
    var_26 = module_0.String(max_length=var_25)
    var_27 = var_26.validate(var_18)
    assert var_27 == 'hello'
    var_28 = 'hello world'
    var_29 = var_26.validate(var_28)
    var_30 = 3
    var_31 = module_0.String(min_length=var_30)
    var_32 = var_31.validate(var_28)
    assert var_32 == 'hello'
    var_33 = 'hi'
    var_34 = var_31.validate(var_33)
    var_35 = '^\\d+$'
    var_36 = module_0.String(pattern=var_35)
    var_37 = '12345'
    var_38 = var_36.validate(var_37)
    assert var_38 == '12345'
    var_39 = 'abc'
    var_40 = var_36.validate(var_39)
    var_41 = '^[a-z]+$'
    var_42 = module_1.compile(var_41)
    var_43 = module_0.String(pattern=var_42)
    var_44 = 'abc'
    var_45 = var_43.validate(var_44)
    assert var_45 == 'abc'
    var_46 = 'ABC'
    var_47 = var_43.validate(var_46)
    var_48 = module_0.String()
    var_49 = 'hello\x00world'
    var_50 = var_48.validate(var_49)
    assert var_50 == 'helloworld'
    var_51 = module_0.String(coerce_types=var_3)
    var_52 = var_51.validate(var_5)
    assert var_52 is None
    var_53 = module_0.String(allow_blank=var_3, coerce_types=var_3)
    var_54 = var_53.validate(var_5)
    assert var_54 == ''
    var_55 = module_0.String(coerce_types=var_3)
    var_56 = var_55.validate(var_11)
    assert var_56 is None
    var_57 = 'email'
    var_58 = module_0.String(format=var_57)
    var_59 = 'test@example.com'
    var_60 = var_58.validate(var_59)
    var_61 = 'uuid'
    var_62 = module_0.String(format=var_61)
    var_63 = module_2.uuid4()
    var_64 = var_62.validate(var_63)
    var_65 = 2
    var_66 = 10
    var_67 = module_0.String(allow_blank=var_13, max_length=var_66, min_length=var_65)
    var_68 = var_67.validate(var_46)
    assert var_68 == 'hello'
    var_69 = 'a'
    var_70 = var_67.validate(var_69)
    var_71 = 'a'
    var_72 = 11
    var_73 = var_71 * var_72
    var_74 = var_67.validate(var_73)



# Parsed testcases at query #24
#--------------------------


import typesystem.fields as module_0

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
    var_10 = module_0.Union(var_9)
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = 0
    var_14 = exc_info.value.messages()[var_13]
    var_15 = var_14.code
    assert var_15 == 'null'
    var_16 = module_0.String()
    var_17 = module_0.Integer()
    var_18 = [var_16, var_17]
    var_19 = module_0.Union(var_18)
    var_20 = 'hello'
    var_21 = var_19.validate(var_20)
    assert var_21 == 'hello'
    var_22 = module_0.String()
    var_23 = module_0.Integer()
    var_24 = [var_22, var_23]
    var_25 = module_0.Union(var_24)
    var_26 = 42
    var_27 = var_25.validate(var_26)
    assert var_27 == 42
    var_28 = module_0.String()
    var_29 = module_0.Integer()
    var_30 = [var_28, var_29]
    var_31 = module_0.Union(var_30)
    var_32 = var_31.validate(var_5)
    assert var_32 is None
    var_33 = module_0.String()
    var_34 = module_0.Integer()
    var_35 = [var_33, var_34]
    var_36 = module_0.Union(var_35)
    var_37 = var_36.validate(var_5)
    assert var_37 is None
    var_38 = module_0.String()
    var_39 = module_0.Integer()
    var_40 = [var_38, var_39]
    var_41 = module_0.Union(var_40)
    var_42 = 1
    var_43 = 2
    var_44 = 3
    var_45 = [var_42, var_43, var_44]
    var_46 = var_41.validate(var_45)
    var_47 = exc_info.value.messages()[var_13]
    var_48 = var_47.code
    assert var_48 == 'union'
    var_49 = 10
    var_50 = module_0.String(min_length=var_49)
    var_51 = 100
    var_52 = module_0.Integer(minimum=var_51)
    var_53 = [var_50, var_52]
    var_54 = module_0.Union(var_53)
    var_55 = 'short'
    var_56 = var_54.validate(var_55)
    var_57 = exc_info.value.messages()[var_13]
    var_58 = var_57.code
    assert var_58 == 'min_length'
    var_59 = module_0.String(min_length=var_49)
    var_60 = module_0.Integer()
    var_61 = [var_59, var_60]
    var_62 = module_0.Union(var_61)
    var_63 = 'short'
    var_64 = var_62.validate(var_63)
    var_65 = exc_info.value.messages()[var_13]
    var_66 = var_65.code
    assert var_66 == 'min_length'
    var_67 = module_0.String()
    var_68 = module_0.Integer()
    var_69 = [var_67, var_68]
    var_70 = module_0.Union(var_69)
    var_71 = 3.14
    var_72 = var_70.validate(var_71)
    var_73 = 3.14
    var_74 = var_70.validate(var_73)
    var_75 = module_0.String()
    var_76 = module_0.Integer()
    var_77 = [var_75, var_76]
    var_78 = module_0.Union(var_77)
    var_79 = True
    var_80 = var_78.validate(var_79)
    var_81 = exc_info.value.messages()[var_13]
    var_82 = var_81.code
    assert var_82 == 'union'
    var_83 = []
    var_84 = module_0.Union(var_83)
    var_85 = 'anything'
    var_86 = var_84.validate(var_85)
    var_87 = exc_info.value.messages()[var_13]
    var_88 = var_87.code
    assert var_88 == 'union'
    var_89 = 'name'
    var_90 = module_0.String()
    var_91 = {var_89: var_90}
    var_92 = module_0.Object(properties=var_91)
    var_93 = module_0.Integer()
    var_94 = module_0.Array(var_93)
    var_95 = [var_92, var_94]
    var_96 = module_0.Union(var_95)
    var_97 = 'test'
    var_98 = {var_89: var_97}
    var_99 = var_96.validate(var_98)
    var_100 = 2
    var_101 = 3
    var_102 = [var_45, var_100, var_101]
    var_103 = var_96.validate(var_102)
    var_104 = module_0.Integer()
    var_105 = module_0.String()
    var_106 = [var_104, var_105]
    var_107 = module_0.Union(var_106)
    var_108 = var_107.validate(var_26)
    assert var_108 == 42
    var_109 = module_0.Integer(coerce_types=var_45)
    var_110 = module_0.String()
    var_111 = [var_109, var_110]
    var_112 = module_0.Union(var_111)
    var_113 = '123'
    var_114 = var_112.validate(var_113)
    assert var_114 == '123'



# Parsed testcases at query #25
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Const(var_0)
    var_2 = None
    var_3 = module_0.Const(var_2)
    var_4 = 'test_value'
    var_5 = module_0.Const(var_4)
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]
    var_10 = module_0.Const(var_9)
    var_11 = 'key'
    var_12 = 'value'
    var_13 = {var_11: var_12}
    var_14 = module_0.Const(var_13)
    var_15 = 42
    var_16 = True
    var_17 = module_0.Const(var_15)
    var_18 = 10
    var_19 = 'A constant field'
    var_20 = module_0.Const(var_18)
    var_21 = True
    var_22 = module_0.Const(var_21)
    var_23 = 3.14
    var_24 = module_0.Const(var_23)



# Parsed testcases at query #26
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Const(var_0)
    var_2 = None
    var_3 = module_0.Const(var_2)
    var_4 = 'test'
    var_5 = module_0.Const(var_4)
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]
    var_10 = module_0.Const(var_9)
    var_11 = 'key'
    var_12 = 'value'
    var_13 = {var_11: var_12}
    var_14 = module_0.Const(var_13)
    var_15 = 42
    var_16 = True
    var_17 = module_0.Const(var_15)
    var_18 = 'A constant field'
    var_19 = module_0.Const(var_17)
    var_20 = 100
    var_21 = 'Custom error'
    var_22 = module_0.Const(var_20)
    var_23 = True
    var_24 = module_0.Const(var_23)
    var_25 = 3.14
    var_26 = module_0.Const(var_25)
    var_27 = ''
    var_28 = module_0.Const(var_27)
    var_29 = 0
    var_30 = module_0.Const(var_29)



# Parsed testcases at query #27
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'red'
    var_1 = 'green'
    var_2 = 'blue'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Choice(choices=var_3)
    var_5 = var_4.validate(var_0)
    assert var_5 == 'red'
    var_6 = var_4.validate(var_1)
    assert var_6 == 'green'
    var_7 = var_4.validate(var_2)
    assert var_7 == 'blue'
    var_8 = '1'
    var_9 = 'Option 1'
    var_10 = (var_8, var_9)
    var_11 = '2'
    var_12 = 'Option 2'
    var_13 = (var_11, var_12)
    var_14 = [var_10, var_13]
    var_15 = module_0.Choice(choices=var_14)
    var_16 = var_15.validate(var_8)
    assert var_16 == '1'
    var_17 = var_15.validate(var_11)
    assert var_17 == '2'
    var_18 = [var_0, var_1, var_2]
    var_19 = module_0.Choice(choices=var_18)
    var_20 = 'yellow'
    var_21 = var_19.validate(var_20)
    var_22 = [var_20, var_21, var_2]
    var_23 = module_0.Choice(choices=var_22)
    var_24 = None
    var_25 = var_23.validate(var_24)
    var_26 = [var_24, var_25, var_2]
    var_27 = True
    var_28 = module_0.Choice(choices=var_26)
    var_29 = None
    var_30 = var_28.validate(var_29)
    assert var_30 is None
    var_31 = [var_24, var_25, var_2]
    var_32 = module_0.Choice(choices=var_31)
    var_33 = ''
    var_34 = var_32.validate(var_33)
    var_35 = [var_33, var_34, var_2]
    var_36 = module_0.Choice(choices=var_35, coerce_types=var_27)
    var_37 = ''
    var_38 = var_36.validate(var_37)
    assert var_38 is None
    var_39 = [var_33, var_34, var_2]
    var_40 = False
    var_41 = module_0.Choice(choices=var_39, coerce_types=var_40)
    var_42 = ''
    var_43 = var_41.validate(var_42)
    var_44 = []
    var_45 = module_0.Choice(choices=var_44)
    var_46 = 'anything'
    var_47 = var_45.validate(var_46)
    var_48 = 'Green'
    var_49 = (var_11, var_48)
    var_50 = [var_46, var_49, var_2]
    var_51 = module_0.Choice(choices=var_50)
    var_52 = var_51.validate(var_46)
    assert var_52 == 'red'
    var_53 = var_51.validate(var_11)
    assert var_53 == '2'
    var_54 = var_51.validate(var_2)
    assert var_54 == 'blue'
    var_55 = '3'
    var_56 = [var_8, var_11, var_55]
    var_57 = module_0.Choice(choices=var_56)
    var_58 = var_57.validate(var_8)
    assert var_58 == '1'
    var_59 = '4'
    var_60 = var_57.validate(var_59)



# Parsed testcases at query #28
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'red'
    var_1 = 'green'
    var_2 = 'blue'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Choice(choices=var_3)
    var_5 = var_4.validate(var_0)
    assert var_5 == 'red'
    var_6 = var_4.validate(var_1)
    assert var_6 == 'green'
    var_7 = var_4.validate(var_2)
    assert var_7 == 'blue'
    var_8 = [var_0, var_1, var_2]
    var_9 = module_0.Choice(choices=var_8)
    var_10 = 'yellow'
    var_11 = var_9.validate(var_10)
    var_12 = 'r'
    var_13 = 'Red'
    var_14 = (var_12, var_13)
    var_15 = 'g'
    var_16 = 'Green'
    var_17 = (var_15, var_16)
    var_18 = 'b'
    var_19 = 'Blue'
    var_20 = (var_18, var_19)
    var_21 = [var_14, var_17, var_20]
    var_22 = module_0.Choice(choices=var_21)
    var_23 = var_22.validate(var_12)
    assert var_23 == 'r'
    var_24 = var_22.validate(var_15)
    assert var_24 == 'g'
    var_25 = 'Red'
    var_26 = var_22.validate(var_25)
    var_27 = [var_25, var_26, var_2]
    var_28 = module_0.Choice(choices=var_27)
    var_29 = None
    var_30 = var_28.validate(var_29)
    var_31 = [var_29, var_30, var_2]
    var_32 = True
    var_33 = module_0.Choice(choices=var_31)
    var_34 = None
    var_35 = var_33.validate(var_34)
    assert var_35 is None
    var_36 = [var_29, var_30, var_2]
    var_37 = False
    var_38 = module_0.Choice(choices=var_36, coerce_types=var_32)
    var_39 = ''
    var_40 = var_38.validate(var_39)
    var_41 = [var_39, var_40, var_2]
    var_42 = module_0.Choice(choices=var_41, coerce_types=var_32)
    var_43 = ''
    var_44 = var_42.validate(var_43)
    assert var_44 is None
    var_45 = [var_39, var_40, var_2]
    var_46 = module_0.Choice(choices=var_45, coerce_types=var_37)
    var_47 = ''
    var_48 = var_46.validate(var_47)
    var_49 = [var_47, var_48, var_2]
    var_50 = module_0.Choice(choices=var_49, coerce_types=var_37)
    var_51 = var_50.validate(var_43)
    assert var_51 is None
    var_52 = []
    var_53 = module_0.Choice(choices=var_52)
    var_54 = 'any'
    var_55 = var_53.validate(var_54)
    var_56 = '1'
    var_57 = '2'
    var_58 = '3'
    var_59 = [var_56, var_57, var_58]
    var_60 = module_0.Choice(choices=var_59)
    var_61 = var_60.validate(var_56)
    assert var_61 == '1'
    var_62 = '4'
    var_63 = var_60.validate(var_62)



# Parsed testcases at query #29
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Object(properties=var_4)
    var_6 = 'John'
    var_7 = 30
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = True
    var_11 = module_0.Object()
    var_12 = None
    var_13 = var_11.validate(var_12)
    assert var_13 is None
    var_14 = False
    var_15 = module_0.Object()
    var_16 = None
    var_17 = var_15.validate(var_16)
    var_18 = module_0.Object()
    var_19 = 'not a dict'
    var_20 = var_18.validate(var_19)
    var_21 = module_0.Object()
    var_22 = 1
    var_23 = 'value'
    var_24 = {var_22: var_23}
    var_25 = var_21.validate(var_24)
    var_26 = 'invalid_key'
    var_27 = module_0.String()
    var_28 = {var_22: var_27}
    var_29 = [var_22]
    var_30 = module_0.Object(properties=var_28, required=var_29)
    var_31 = {}
    var_32 = var_30.validate(var_31)
    var_33 = 'required'
    var_34 = 2
    var_35 = module_0.Object(min_properties=var_34)
    var_36 = 'key'
    var_37 = 'value'
    var_38 = {var_36: var_37}
    var_39 = var_35.validate(var_38)
    var_40 = module_0.Object(min_properties=var_10)
    var_41 = {}
    var_42 = var_40.validate(var_41)
    var_43 = module_0.Object(max_properties=var_10)
    var_44 = 'key1'
    var_45 = 'key2'
    var_46 = 'value1'
    var_47 = 'value2'
    var_48 = {var_44: var_46, var_45: var_47}
    var_49 = var_43.validate(var_48)
    var_50 = 'Unknown'
    var_51 = module_0.String()
    var_52 = {var_44: var_51}
    var_53 = module_0.Object(properties=var_52)
    var_54 = {}
    var_55 = var_53.validate(var_54)
    var_56 = module_0.String()
    var_57 = {var_44: var_56}
    var_58 = module_0.Object(properties=var_57, additional_properties=var_10)
    var_59 = 'extra'
    var_60 = 'data'
    var_61 = {var_44: var_49, var_59: var_60}
    var_62 = var_58.validate(var_61)
    var_63 = module_0.String()
    var_64 = {var_44: var_63}
    var_65 = module_0.Object(properties=var_64, additional_properties=var_14)
    var_66 = 'name'
    var_67 = 'extra'
    var_68 = 'John'
    var_69 = 'data'
    var_70 = {var_66: var_68, var_67: var_69}
    var_71 = var_65.validate(var_70)
    var_72 = 'invalid_property'
    var_73 = module_0.String()
    var_74 = {var_66: var_73}
    var_75 = module_0.Integer()
    var_76 = module_0.Object(properties=var_74, additional_properties=var_75)
    var_77 = {var_66: var_71, var_67: var_7}
    var_78 = var_76.validate(var_77)
    var_79 = '^num_'
    var_80 = module_0.Integer()
    var_81 = {var_79: var_80}
    var_82 = module_0.Object(pattern_properties=var_81)
    var_83 = 'num_1'
    var_84 = 'num_2'
    var_85 = 10
    var_86 = 20
    var_87 = {var_83: var_85, var_84: var_86}
    var_88 = var_82.validate(var_87)
    var_89 = '^[a-z]+$'
    var_90 = module_0.String(pattern=var_89)
    var_91 = module_0.Object(property_names=var_90)
    var_92 = 'Invalid'
    var_93 = 'value'
    var_94 = {var_92: var_93}
    var_95 = var_91.validate(var_94)
    var_96 = 'user'
    var_97 = module_0.String()
    var_98 = module_0.Integer()
    var_99 = {var_92: var_97, var_93: var_98}
    var_100 = module_0.Object(properties=var_99)
    var_101 = {var_96: var_100}
    var_102 = module_0.Object(properties=var_101)
    var_103 = {var_92: var_71, var_93: var_7}
    var_104 = {var_96: var_103}
    var_105 = var_102.validate(var_104)
    var_106 = module_0.Integer()
    var_107 = {var_93: var_106}
    var_108 = module_0.Object(properties=var_107)
    var_109 = {var_96: var_108}
    var_110 = module_0.Object(properties=var_109)
    var_111 = 'user'
    var_112 = 'age'
    var_113 = 'not an int'
    var_114 = {var_112: var_113}
    var_115 = {var_111: var_114}
    var_116 = var_110.validate(var_115)
    var_117 = module_0.String()
    var_118 = {var_111: var_117}
    var_119 = module_0.Object(properties=var_118)
    var_120 = (var_111, var_116)
    var_121 = [var_120]
    var_122 = 5
    var_123 = module_0.String(max_length=var_122)
    var_124 = {var_111: var_123}
    var_125 = module_0.Object(properties=var_124)
    var_126 = 'name'
    var_127 = 'VeryLongName'
    var_128 = {var_126: var_127}
    var_129 = var_125.validate(var_128)
    var_130 = 'max_length'



# Parsed testcases at query #30
#--------------------------


import typesystem.fields as module_0

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
    var_8 = exc_info.value.messages()[var_4]
    var_9 = var_8.code
    assert var_9 == 'null'
    var_10 = module_0.Array()
    var_11 = 'not a list'
    var_12 = var_10.validate(var_11)
    var_13 = exc_info.value.messages()[var_4]
    var_14 = var_13.code
    assert var_14 == 'type'
    var_15 = 2
    var_16 = module_0.Array(exact_items=var_15)
    var_17 = 1
    var_18 = [var_17]
    var_19 = var_16.validate(var_18)
    var_20 = exc_info.value.messages()[var_4]
    var_21 = var_20.code
    assert var_21 == 'exact_items'
    var_22 = module_0.Array(min_items=var_15)
    var_23 = 1
    var_24 = [var_23]
    var_25 = var_22.validate(var_24)
    var_26 = exc_info.value.messages()[var_4]
    var_27 = var_26.code
    assert var_27 == 'min_items'
    var_28 = module_0.Array(min_items=var_23)
    var_29 = []
    var_30 = var_28.validate(var_29)
    var_31 = exc_info.value.messages()[var_4]
    var_32 = var_31.code
    assert var_32 == 'empty'
    var_33 = module_0.Array(max_items=var_15)
    var_34 = 1
    var_35 = 2
    var_36 = 3
    var_37 = [var_34, var_35, var_36]
    var_38 = var_33.validate(var_37)
    var_39 = exc_info.value.messages()[var_37]
    var_40 = var_39.code
    assert var_40 == 'max_items'
    var_41 = module_0.Integer()
    var_42 = module_0.Array(var_41)
    var_43 = 3
    var_44 = [var_34, var_15, var_43]
    var_45 = var_42.validate(var_44)
    var_46 = module_0.Integer()
    var_47 = module_0.Array(var_46)
    var_48 = 1
    var_49 = 'not an int'
    var_50 = 3
    var_51 = [var_48, var_49, var_50]
    var_52 = var_47.validate(var_51)
    var_53 = module_0.Integer()
    var_54 = module_0.String()
    var_55 = [var_53, var_54]
    var_56 = module_0.Array(var_55)
    var_57 = 'test'
    var_58 = [var_48, var_57]
    var_59 = var_56.validate(var_58)
    var_60 = module_0.Integer()
    var_61 = module_0.String()
    var_62 = [var_60, var_61]
    var_63 = module_0.Array(var_62, var_51)
    var_64 = 1
    var_65 = 'test'
    var_66 = 3
    var_67 = [var_64, var_65, var_66]
    var_68 = var_63.validate(var_67)
    var_69 = exc_info.value.messages()[var_67]
    var_70 = var_69.code
    assert var_70 == 'max_items'
    var_71 = module_0.Integer()
    var_72 = module_0.String()
    var_73 = [var_71, var_72]
    var_74 = module_0.Float()
    var_75 = module_0.Array(var_73, var_74)
    var_76 = 3.5
    var_77 = [var_64, var_57, var_76]
    var_78 = var_75.validate(var_77)
    var_79 = module_0.Array(unique_items=var_64)
    var_80 = 1
    var_81 = 2
    var_82 = [var_80, var_81, var_80]
    var_83 = var_79.validate(var_82)
    var_84 = exc_info.value.messages()[var_83]
    var_85 = var_84.code
    assert var_85 == 'unique_items'
    var_86 = module_0.Array(unique_items=var_80)
    var_87 = [var_80, var_15, var_43]
    var_88 = var_86.validate(var_87)
    var_89 = module_0.Array()
    var_90 = []
    var_91 = var_89.validate(var_90)
    var_92 = 'key'
    var_93 = module_0.String()
    var_94 = {var_92: var_93}
    var_95 = module_0.Object(properties=var_94)
    var_96 = module_0.Array(var_95)
    var_97 = 'value'
    var_98 = {var_92: var_97}
    var_99 = [var_98]
    var_100 = var_96.validate(var_99)
    var_101 = module_0.Integer()
    var_102 = {var_92: var_101}
    var_103 = module_0.Object(properties=var_102)
    var_104 = module_0.Array(var_103)
    var_105 = 'key'
    var_106 = 'not an int'
    var_107 = {var_105: var_106}
    var_108 = [var_107]
    var_109 = var_104.validate(var_108)
    var_110 = module_0.Array(var_106)
    var_111 = [var_105, var_57, var_76, var_106]
    var_112 = var_110.validate(var_111)
    var_113 = module_0.Array(exact_items=var_43)
    var_114 = [var_105, var_15, var_43]
    var_115 = var_113.validate(var_114)
    var_116 = module_0.Integer()
    var_117 = module_0.Array(var_116)
    var_118 = 1
    var_119 = 'not int'
    var_120 = 'also not int'
    var_121 = [var_118, var_119, var_120]
    var_122 = var_117.validate(var_121)



# Parsed testcases at query #31
#--------------------------


import typesystem.fields as module_0

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
    var_10 = module_0.Boolean()
    var_11 = var_10.validate(var_4)
    assert var_11 is False
    var_12 = module_0.Boolean(coerce_types=var_6)
    var_13 = 'true'
    var_14 = var_12.validate(var_13)
    assert var_14 is True
    var_15 = module_0.Boolean(coerce_types=var_6)
    var_16 = 'false'
    var_17 = var_15.validate(var_16)
    assert var_17 is False
    var_18 = module_0.Boolean(coerce_types=var_6)
    var_19 = 'True'
    var_20 = var_18.validate(var_19)
    assert var_20 is True
    var_21 = module_0.Boolean(coerce_types=var_6)
    var_22 = 'on'
    var_23 = var_21.validate(var_22)
    assert var_23 is True
    var_24 = module_0.Boolean(coerce_types=var_6)
    var_25 = 'off'
    var_26 = var_24.validate(var_25)
    assert var_26 is False
    var_27 = module_0.Boolean(coerce_types=var_6)
    var_28 = '1'
    var_29 = var_27.validate(var_28)
    assert var_29 is True
    var_30 = module_0.Boolean(coerce_types=var_6)
    var_31 = '0'
    var_32 = var_30.validate(var_31)
    assert var_32 is False
    var_33 = module_0.Boolean(coerce_types=var_6)
    var_34 = ''
    var_35 = var_33.validate(var_34)
    assert var_35 is False
    var_36 = module_0.Boolean(coerce_types=var_6)
    var_37 = var_36.validate(var_6)
    assert var_37 is True
    var_38 = module_0.Boolean(coerce_types=var_6)
    var_39 = var_38.validate(var_4)
    assert var_39 is False
    var_40 = module_0.Boolean(coerce_types=var_6)
    var_41 = 'null'
    var_42 = var_40.validate(var_41)
    assert var_42 is None
    var_43 = module_0.Boolean(coerce_types=var_6)
    var_44 = 'none'
    var_45 = var_43.validate(var_44)
    assert var_45 is None
    var_46 = module_0.Boolean(coerce_types=var_6)
    var_47 = var_46.validate(var_34)
    assert var_47 is None
    var_48 = module_0.Boolean(coerce_types=var_6)
    var_49 = 'invalid'
    var_50 = var_48.validate(var_49)
    var_51 = module_0.Boolean(coerce_types=var_4)
    var_52 = 'true'
    var_53 = var_51.validate(var_52)
    var_54 = module_0.Boolean(coerce_types=var_52)
    var_55 = []
    var_56 = var_54.validate(var_55)
    var_57 = module_0.Boolean(coerce_types=var_55)
    var_58 = {}
    var_59 = var_57.validate(var_58)



# Parsed testcases at query #32
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Const(var_0)
    var_2 = None
    var_3 = module_0.Const(var_2)
    var_4 = 'test_value'
    var_5 = module_0.Const(var_4)
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]
    var_10 = module_0.Const(var_9)
    var_11 = 'key'
    var_12 = 'value'
    var_13 = {var_11: var_12}
    var_14 = module_0.Const(var_13)
    var_15 = 42
    var_16 = True
    var_17 = module_0.Const(var_15)
    var_18 = 100
    var_19 = True
    var_20 = module_0.Const(var_18)
    var_21 = 50
    var_22 = module_0.Const(var_21)
    var_23 = True
    var_24 = module_0.Const(var_23)
    var_25 = 3.14
    var_26 = module_0.Const(var_25)



# Parsed testcases at query #33
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'red'
    var_1 = 'green'
    var_2 = 'blue'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Choice(choices=var_3)
    var_5 = var_4.validate(var_0)
    assert var_5 == 'red'
    var_6 = var_4.validate(var_1)
    assert var_6 == 'green'
    var_7 = var_4.validate(var_2)
    assert var_7 == 'blue'
    var_8 = [var_0, var_1, var_2]
    var_9 = module_0.Choice(choices=var_8)
    var_10 = 'yellow'
    var_11 = var_9.validate(var_10)
    var_12 = '1'
    var_13 = 'Option 1'
    var_14 = (var_12, var_13)
    var_15 = '2'
    var_16 = 'Option 2'
    var_17 = (var_15, var_16)
    var_18 = [var_14, var_17]
    var_19 = module_0.Choice(choices=var_18)
    var_20 = var_19.validate(var_12)
    assert var_20 == '1'
    var_21 = var_19.validate(var_15)
    assert var_21 == '2'
    var_22 = (var_12, var_13)
    var_23 = (var_15, var_16)
    var_24 = [var_22, var_23]
    var_25 = module_0.Choice(choices=var_24)
    var_26 = '3'
    var_27 = var_25.validate(var_26)
    var_28 = [var_26, var_27]
    var_29 = True
    var_30 = module_0.Choice(choices=var_28)
    var_31 = None
    var_32 = var_30.validate(var_31)
    assert var_32 is None
    var_33 = [var_26, var_27]
    var_34 = False
    var_35 = module_0.Choice(choices=var_33)
    var_36 = None
    var_37 = var_35.validate(var_36)
    var_38 = [var_36, var_37]
    var_39 = module_0.Choice(choices=var_38, coerce_types=var_29)
    var_40 = ''
    var_41 = var_39.validate(var_40)
    assert var_41 is None
    var_42 = [var_36, var_37]
    var_43 = module_0.Choice(choices=var_42, coerce_types=var_29)
    var_44 = ''
    var_45 = var_43.validate(var_44)
    var_46 = [var_44, var_45]
    var_47 = module_0.Choice(choices=var_46, coerce_types=var_34)
    var_48 = ''
    var_49 = var_47.validate(var_48)
    var_50 = 2
    var_51 = 3
    var_52 = [var_29, var_50, var_51]
    var_53 = module_0.Choice(choices=var_52)
    var_54 = var_53.validate(var_29)
    assert var_54 == 1
    var_55 = var_53.validate(var_50)
    assert var_55 == 2
    var_56 = [var_29, var_50, var_51]
    var_57 = module_0.Choice(choices=var_56)
    var_58 = 4
    var_59 = var_57.validate(var_58)
    var_60 = []
    var_61 = module_0.Choice(choices=var_60)
    var_62 = var_61.validate(var_31)
    assert var_62 is None
    var_63 = [var_40, var_58, var_59]
    var_64 = module_0.Choice(choices=var_63)
    var_65 = var_64.validate(var_40)
    assert var_65 == ''
    var_66 = 'Green Color'
    var_67 = (var_59, var_66)
    var_68 = [var_58, var_67, var_2]
    var_69 = module_0.Choice(choices=var_68)
    var_70 = var_69.validate(var_58)
    assert var_70 == 'red'
    var_71 = var_69.validate(var_59)
    assert var_71 == 'green'
    var_72 = var_69.validate(var_2)
    assert var_72 == 'blue'



# Parsed testcases at query #34
#--------------------------


import typesystem.fields as module_0

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
    var_10 = module_0.Boolean()
    var_11 = var_10.validate(var_4)
    assert var_11 is False
    var_12 = module_0.Boolean(coerce_types=var_6)
    var_13 = 'true'
    var_14 = var_12.validate(var_13)
    assert var_14 is True
    var_15 = 'True'
    var_16 = var_12.validate(var_15)
    assert var_16 is True
    var_17 = 'false'
    var_18 = var_12.validate(var_17)
    assert var_18 is False
    var_19 = 'False'
    var_20 = var_12.validate(var_19)
    assert var_20 is False
    var_21 = 'on'
    var_22 = var_12.validate(var_21)
    assert var_22 is True
    var_23 = 'off'
    var_24 = var_12.validate(var_23)
    assert var_24 is False
    var_25 = '1'
    var_26 = var_12.validate(var_25)
    assert var_26 is True
    var_27 = '0'
    var_28 = var_12.validate(var_27)
    assert var_28 is False
    var_29 = module_0.Boolean(coerce_types=var_6)
    var_30 = ''
    var_31 = var_29.validate(var_30)
    assert var_31 is False
    var_32 = module_0.Boolean(coerce_types=var_6)
    var_33 = 'null'
    var_34 = var_32.validate(var_33)
    assert var_34 is None
    var_35 = 'none'
    var_36 = var_32.validate(var_35)
    assert var_36 is None
    var_37 = module_0.Boolean(coerce_types=var_6)
    var_38 = var_37.validate(var_6)
    assert var_38 is True
    var_39 = var_37.validate(var_4)
    assert var_39 is False
    var_40 = module_0.Boolean(coerce_types=var_4)
    var_41 = 'true'
    var_42 = var_40.validate(var_41)
    var_43 = module_0.Boolean(coerce_types=var_41)
    var_44 = 'invalid'
    var_45 = var_43.validate(var_44)
    var_46 = module_0.Boolean(coerce_types=var_44)
    var_47 = []
    var_48 = var_46.validate(var_47)
    var_49 = module_0.Boolean(coerce_types=var_47)
    var_50 = 'TRUE'
    var_51 = var_49.validate(var_50)
    assert var_51 is True
    var_52 = 'FALSE'
    var_53 = var_49.validate(var_52)
    assert var_53 is False
    var_54 = 'ON'
    var_55 = var_49.validate(var_54)
    assert var_55 is True
    var_56 = 'OFF'
    var_57 = var_49.validate(var_56)
    assert var_57 is False



# Parsed testcases at query #35
#--------------------------


import typesystem.fields as module_0

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
    var_12 = {}
    var_13 = var_11.validate(var_12)
    var_14 = 'name'
    var_15 = 'age'
    var_16 = module_0.String()
    var_17 = module_0.Integer()
    var_18 = {var_14: var_16, var_15: var_17}
    var_19 = module_0.Object(properties=var_18)
    var_20 = 'John'
    var_21 = 30
    var_22 = {var_14: var_20, var_15: var_21}
    var_23 = var_19.validate(var_22)
    var_24 = [var_14]
    var_25 = module_0.Object(required=var_24)
    var_26 = {}
    var_27 = var_25.validate(var_26)
    var_28 = 'required'
    var_29 = module_0.Object()
    var_30 = 1
    var_31 = 'value'
    var_32 = {var_30: var_31}
    var_33 = var_29.validate(var_32)
    var_34 = 'invalid_key'
    var_35 = 2
    var_36 = module_0.Object(min_properties=var_35)
    var_37 = 'a'
    var_38 = 1
    var_39 = {var_37: var_38}
    var_40 = var_36.validate(var_39)
    var_41 = module_0.Object(min_properties=var_37)
    var_42 = {}
    var_43 = var_41.validate(var_42)
    var_44 = module_0.Object(max_properties=var_42)
    var_45 = 'a'
    var_46 = 'b'
    var_47 = 1
    var_48 = 2
    var_49 = {var_45: var_47, var_46: var_48}
    var_50 = var_44.validate(var_49)
    var_51 = '^[a-z]+$'
    var_52 = module_0.String(pattern=var_51)
    var_53 = module_0.Object(property_names=var_52)
    var_54 = '123'
    var_55 = 'value'
    var_56 = {var_54: var_55}
    var_57 = var_53.validate(var_56)
    var_58 = 'invalid_property'
    var_59 = module_0.String()
    var_60 = {var_14: var_59}
    var_61 = module_0.Object(properties=var_60, additional_properties=var_54)
    var_62 = 'extra'
    var_63 = 'value'
    var_64 = {var_14: var_20, var_62: var_63}
    var_65 = var_61.validate(var_64)
    var_66 = module_0.String()
    var_67 = {var_14: var_66}
    var_68 = module_0.Object(properties=var_67, additional_properties=var_57)
    var_69 = 'name'
    var_70 = 'extra'
    var_71 = 'John'
    var_72 = 'value'
    var_73 = {var_69: var_71, var_70: var_72}
    var_74 = var_68.validate(var_73)
    var_75 = module_0.String()
    var_76 = {var_14: var_75}
    var_77 = module_0.Integer()
    var_78 = module_0.Object(properties=var_76, additional_properties=var_77)
    var_79 = {var_14: var_20, var_15: var_21}
    var_80 = var_78.validate(var_79)
    var_81 = '^num_'
    var_82 = module_0.Integer()
    var_83 = {var_81: var_82}
    var_84 = module_0.Object(pattern_properties=var_83)
    var_85 = 'num_1'
    var_86 = 'num_2'
    var_87 = 10
    var_88 = 20
    var_89 = {var_85: var_87, var_86: var_88}
    var_90 = var_84.validate(var_89)
    var_91 = 'Unknown'
    var_92 = module_0.String()
    var_93 = {var_14: var_92}
    var_94 = module_0.Object(properties=var_93)
    var_95 = {}
    var_96 = var_94.validate(var_95)
    var_97 = module_0.Integer(minimum=var_72)
    var_98 = {var_15: var_97}
    var_99 = module_0.Object(properties=var_98)
    var_100 = 'age'
    var_101 = -5
    var_102 = {var_100: var_101}
    var_103 = var_99.validate(var_102)
    var_104 = 'minimum'
    var_105 = module_0.Object()
    var_106 = 'key'
    var_107 = (var_106, var_63)
    var_108 = [var_107]
    var_109 = 'user'
    var_110 = module_0.String()
    var_111 = module_0.Integer()
    var_112 = {var_14: var_110, var_15: var_111}
    var_113 = [var_14]
    var_114 = module_0.Object(properties=var_112, required=var_113)
    var_115 = {var_109: var_114}
    var_116 = module_0.Object(properties=var_115)
    var_117 = {var_14: var_20, var_15: var_21}
    var_118 = {var_109: var_117}
    var_119 = var_116.validate(var_118)



# Parsed testcases at query #36
#--------------------------


import typesystem.fields as module_0

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
    var_14 = exc_info.value.messages()[var_10]
    var_15 = var_14.code
    assert var_15 == 'null'
    var_16 = module_0.String()
    var_17 = module_0.Integer()
    var_18 = [var_16, var_17]
    var_19 = module_0.Union(var_18)
    var_20 = 'hello'
    var_21 = var_19.validate(var_20)
    assert var_21 == 'hello'
    var_22 = module_0.String()
    var_23 = module_0.Integer()
    var_24 = [var_22, var_23]
    var_25 = module_0.Union(var_24)
    var_26 = 42
    var_27 = var_25.validate(var_26)
    assert var_27 == 42
    var_28 = module_0.String()
    var_29 = module_0.Integer()
    var_30 = [var_28, var_29]
    var_31 = module_0.Union(var_30)
    var_32 = []
    var_33 = var_31.validate(var_32)
    var_34 = exc_info.value.messages()[var_10]
    var_35 = var_34.code
    assert var_35 == 'union'
    var_36 = module_0.String()
    var_37 = module_0.Integer()
    var_38 = [var_36, var_37]
    var_39 = module_0.Union(var_38)
    var_40 = var_39.validate(var_5)
    assert var_40 is None
    var_41 = module_0.String()
    var_42 = module_0.Integer()
    var_43 = [var_41, var_42]
    var_44 = module_0.Union(var_43)
    var_45 = var_44.validate(var_5)
    assert var_45 is None
    var_46 = 10
    var_47 = module_0.Integer(minimum=var_46)
    var_48 = module_0.String()
    var_49 = [var_47, var_48]
    var_50 = module_0.Union(var_49)
    var_51 = 5
    var_52 = var_50.validate(var_51)
    var_53 = exc_info.value.messages()[var_10]
    var_54 = var_53.code
    assert var_54 == 'minimum'
    var_55 = module_0.Integer()
    var_56 = module_0.String()
    var_57 = [var_55, var_56]
    var_58 = module_0.Union(var_57)
    var_59 = 'test'
    var_60 = var_58.validate(var_59)
    assert var_60 == 'test'
    var_61 = module_0.Integer()
    var_62 = module_0.String()
    var_63 = [var_61, var_62]
    var_64 = module_0.Union(var_63)
    var_65 = 3.0
    var_66 = var_64.validate(var_65)
    assert var_66 == 3
    var_67 = module_0.String()
    var_68 = module_0.Integer()
    var_69 = [var_67, var_68]
    var_70 = module_0.Union(var_69)
    var_71 = ''
    var_72 = var_70.validate(var_71)
    assert var_72 == ''
    var_73 = module_0.String()
    var_74 = module_0.Integer()
    var_75 = [var_73, var_74]
    var_76 = module_0.Union(var_75)
    var_77 = True
    var_78 = var_76.validate(var_77)
    var_79 = exc_info.value.messages()[var_10]
    var_80 = var_79.code
    assert var_80 == 'union'
    var_81 = 100
    var_82 = module_0.Integer(minimum=var_81)
    var_83 = 50
    var_84 = module_0.Integer(minimum=var_83)
    var_85 = [var_82, var_84]
    var_86 = module_0.Union(var_85)
    var_87 = 30
    var_88 = var_86.validate(var_87)
    var_89 = exc_info.value.messages()[var_10]
    var_90 = var_89.code
    assert var_90 == 'union'
    var_91 = 5
    var_92 = module_0.String(max_length=var_91)
    var_93 = module_0.String()
    var_94 = [var_92, var_93]
    var_95 = module_0.Union(var_94)
    var_96 = 'hi'
    var_97 = var_95.validate(var_96)
    assert var_97 == 'hi'
    var_98 = 2
    var_99 = module_0.String(max_length=var_98)
    var_100 = module_0.String()
    var_101 = [var_99, var_100]
    var_102 = module_0.Union(var_101)
    var_103 = var_102.validate(var_20)
    assert var_103 == 'hello'



# Parsed testcases at query #37
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = 'hello'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'hello'
    var_6 = 42
    var_7 = var_3.validate(var_6)
    assert var_7 == 42
    var_8 = module_0.String()
    var_9 = module_0.Integer()
    var_10 = [var_8, var_9]
    var_11 = True
    var_12 = module_0.Union(var_10)
    var_13 = None
    var_14 = var_12.validate(var_13)
    assert var_14 is None
    var_15 = module_0.String()
    var_16 = module_0.Integer()
    var_17 = [var_15, var_16]
    var_18 = False
    var_19 = module_0.Union(var_17)
    var_20 = None
    var_21 = var_19.validate(var_20)
    var_22 = exc_info.value.messages()[var_18]
    var_23 = var_22.code
    assert var_23 == 'null'
    var_24 = module_0.String()
    var_25 = module_0.Integer()
    var_26 = [var_24, var_25]
    var_27 = module_0.Union(var_26)
    var_28 = 1
    var_29 = 2
    var_30 = 3
    var_31 = [var_28, var_29, var_30]
    var_32 = var_27.validate(var_31)
    var_33 = exc_info.value.messages()[var_18]
    var_34 = var_33.code
    assert var_34 == 'union'
    var_35 = module_0.String()
    var_36 = module_0.Integer()
    var_37 = [var_35, var_36]
    var_38 = module_0.Union(var_37)
    var_39 = var_38.validate(var_13)
    assert var_39 is None
    var_40 = module_0.String()
    var_41 = module_0.Integer()
    var_42 = [var_40, var_41]
    var_43 = module_0.Union(var_42)
    var_44 = var_43.validate(var_13)
    assert var_44 is None
    var_45 = module_0.Integer()
    var_46 = module_0.Float()
    var_47 = [var_45, var_46]
    var_48 = module_0.Union(var_47)
    var_49 = 3.14
    var_50 = var_48.validate(var_49)
    var_51 = 5
    var_52 = module_0.String(min_length=var_51)
    var_53 = module_0.Integer()
    var_54 = [var_52, var_53]
    var_55 = module_0.Union(var_54)
    var_56 = 'ab'
    var_57 = var_55.validate(var_56)
    var_58 = exc_info.value.messages()[var_18]
    var_59 = var_58.code
    assert var_59 == 'min_length'
    var_60 = 10
    var_61 = module_0.Integer(minimum=var_60)
    var_62 = module_0.String(min_length=var_51)
    var_63 = [var_61, var_62]
    var_64 = module_0.Union(var_63)
    var_65 = 5
    var_66 = var_64.validate(var_65)
    var_67 = module_0.String()
    var_68 = module_0.Integer()
    var_69 = [var_67, var_68]
    var_70 = module_0.Union(var_69)
    var_71 = True
    var_72 = var_70.validate(var_71)
    var_73 = exc_info.value.messages()[var_18]
    var_74 = var_73.code
    assert var_74 == 'union'
    var_75 = module_0.String()
    var_76 = module_0.Integer()
    var_77 = [var_75, var_76]
    var_78 = module_0.Union(var_77)
    var_79 = ''
    var_80 = var_78.validate(var_79)
    assert var_80 == ''
    var_81 = module_0.String()
    var_82 = module_0.Integer()
    var_83 = [var_81, var_82]
    var_84 = module_0.Union(var_83)
    var_85 = var_84.validate(var_18)
    assert var_85 == 0
    var_86 = module_0.String()
    var_87 = module_0.Float()
    var_88 = [var_86, var_87]
    var_89 = module_0.Union(var_88)
    var_90 = 2.718
    var_91 = var_89.validate(var_90)
    var_92 = module_0.String()
    var_93 = module_0.Integer()
    var_94 = [var_92, var_93]
    var_95 = module_0.Union(var_94)
    var_96 = module_0.Boolean()
    var_97 = [var_95, var_96]
    var_98 = module_0.Union(var_97)
    var_99 = 'test'
    var_100 = var_98.validate(var_99)
    assert var_100 == 'test'
    var_101 = var_98.validate(var_32)
    assert var_101 == 42
    var_102 = var_98.validate(var_11)
    assert var_102 is True



# Parsed testcases at query #38
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'red'
    var_1 = 'green'
    var_2 = 'blue'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Choice(choices=var_3)
    var_5 = var_4.validate(var_0)
    assert var_5 == 'red'
    var_6 = var_4.validate(var_1)
    assert var_6 == 'green'
    var_7 = var_4.validate(var_2)
    assert var_7 == 'blue'
    var_8 = '1'
    var_9 = 'Option 1'
    var_10 = (var_8, var_9)
    var_11 = '2'
    var_12 = 'Option 2'
    var_13 = (var_11, var_12)
    var_14 = [var_10, var_13]
    var_15 = module_0.Choice(choices=var_14)
    var_16 = var_15.validate(var_8)
    assert var_16 == '1'
    var_17 = var_15.validate(var_11)
    assert var_17 == '2'
    var_18 = [var_0, var_1, var_2]
    var_19 = module_0.Choice(choices=var_18)
    var_20 = 'yellow'
    var_21 = var_19.validate(var_20)
    var_22 = [var_20, var_21, var_2]
    var_23 = module_0.Choice(choices=var_22)
    var_24 = None
    var_25 = var_23.validate(var_24)
    var_26 = [var_24, var_25, var_2]
    var_27 = True
    var_28 = module_0.Choice(choices=var_26)
    var_29 = None
    var_30 = var_28.validate(var_29)
    assert var_30 is None
    var_31 = [var_24, var_25, var_2]
    var_32 = False
    var_33 = module_0.Choice(choices=var_31, coerce_types=var_32)
    var_34 = ''
    var_35 = var_33.validate(var_34)
    var_36 = [var_34, var_35, var_2]
    var_37 = module_0.Choice(choices=var_36, coerce_types=var_27)
    var_38 = ''
    var_39 = var_37.validate(var_38)
    assert var_39 is None
    var_40 = [var_34, var_35, var_2]
    var_41 = module_0.Choice(choices=var_40, coerce_types=var_27)
    var_42 = ''
    var_43 = var_41.validate(var_42)
    var_44 = 'Green'
    var_45 = (var_11, var_44)
    var_46 = [var_42, var_45, var_2]
    var_47 = module_0.Choice(choices=var_46)
    var_48 = var_47.validate(var_42)
    assert var_48 == 'red'
    var_49 = var_47.validate(var_11)
    assert var_49 == '2'
    var_50 = var_47.validate(var_2)
    assert var_50 == 'blue'
    var_51 = []
    var_52 = module_0.Choice(choices=var_51)
    var_53 = 'red'
    var_54 = var_52.validate(var_53)
    var_55 = 'One'
    var_56 = (var_8, var_55)
    var_57 = 'Two'
    var_58 = (var_11, var_57)
    var_59 = [var_56, var_58]
    var_60 = module_0.Choice(choices=var_59)
    var_61 = var_60.validate(var_8)
    assert var_61 == '1'
    var_62 = '3'
    var_63 = var_60.validate(var_62)



# Parsed testcases at query #39
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Const(var_0)
    var_2 = None
    var_3 = module_0.Const(var_2)
    var_4 = 'hello'
    var_5 = module_0.Const(var_4)
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]
    var_10 = module_0.Const(var_9)
    var_11 = 'key'
    var_12 = 'value'
    var_13 = {var_11: var_12}
    var_14 = module_0.Const(var_13)
    var_15 = 42
    var_16 = True
    var_17 = module_0.Const(var_15)
    var_18 = 42
    var_19 = False
    var_20 = module_0.Const(var_18)
    var_21 = 99
    var_22 = 'A constant field'
    var_23 = module_0.Const(var_21)
    var_24 = 5
    var_25 = module_0.Const(var_24)
    var_26 = 'const'
    var_27 = hasattr(var_25, var_26)



# Parsed testcases at query #40
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'red'
    var_1 = 'green'
    var_2 = 'blue'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Choice(choices=var_3)
    var_5 = var_4.validate(var_0)
    assert var_5 == 'red'
    var_6 = var_4.validate(var_1)
    assert var_6 == 'green'
    var_7 = var_4.validate(var_2)
    assert var_7 == 'blue'
    var_8 = 'r'
    var_9 = 'Red'
    var_10 = (var_8, var_9)
    var_11 = 'g'
    var_12 = 'Green'
    var_13 = (var_11, var_12)
    var_14 = 'b'
    var_15 = 'Blue'
    var_16 = (var_14, var_15)
    var_17 = [var_10, var_13, var_16]
    var_18 = module_0.Choice(choices=var_17)
    var_19 = var_18.validate(var_8)
    assert var_19 == 'r'
    var_20 = var_18.validate(var_11)
    assert var_20 == 'g'
    var_21 = var_18.validate(var_14)
    assert var_21 == 'b'
    var_22 = [var_0, var_1, var_2]
    var_23 = module_0.Choice(choices=var_22)
    var_24 = 'yellow'
    var_25 = var_23.validate(var_24)
    var_26 = [var_24, var_25, var_2]
    var_27 = True
    var_28 = module_0.Choice(choices=var_26)
    var_29 = None
    var_30 = var_28.validate(var_29)
    assert var_30 is None
    var_31 = [var_24, var_25, var_2]
    var_32 = False
    var_33 = module_0.Choice(choices=var_31)
    var_34 = None
    var_35 = var_33.validate(var_34)
    var_36 = [var_34, var_35, var_2]
    var_37 = module_0.Choice(choices=var_36, coerce_types=var_27)
    var_38 = ''
    var_39 = var_37.validate(var_38)
    assert var_39 is None
    var_40 = [var_34, var_35, var_2]
    var_41 = module_0.Choice(choices=var_40, coerce_types=var_27)
    var_42 = ''
    var_43 = var_41.validate(var_42)
    var_44 = [var_42, var_43, var_2]
    var_45 = module_0.Choice(choices=var_44, coerce_types=var_32)
    var_46 = ''
    var_47 = var_45.validate(var_46)
    var_48 = (var_11, var_12)
    var_49 = [var_46, var_48, var_2]
    var_50 = module_0.Choice(choices=var_49)
    var_51 = var_50.validate(var_46)
    assert var_51 == 'red'
    var_52 = var_50.validate(var_11)
    assert var_52 == 'g'
    var_53 = var_50.validate(var_2)
    assert var_53 == 'blue'
    var_54 = 2
    var_55 = 3
    var_56 = [var_27, var_54, var_55]
    var_57 = module_0.Choice(choices=var_56)
    var_58 = var_57.validate(var_27)
    assert var_58 == 1
    var_59 = var_57.validate(var_54)
    assert var_59 == 2
    var_60 = var_57.validate(var_55)
    assert var_60 == 3
    var_61 = [var_27, var_54, var_55]
    var_62 = module_0.Choice(choices=var_61)
    var_63 = 4
    var_64 = var_62.validate(var_63)
    var_65 = []
    var_66 = module_0.Choice(choices=var_65)
    var_67 = 'anything'
    var_68 = var_66.validate(var_67)
    var_69 = module_0.Choice(choices=var_29)
    var_70 = 'anything'
    var_71 = var_69.validate(var_70)



# Parsed testcases at query #41
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'hello'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'hello'
    var_3 = True
    var_4 = module_0.String()
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None
    var_7 = module_0.String()
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = module_0.String(allow_blank=var_3)
    var_11 = ''
    var_12 = var_10.validate(var_11)
    assert var_12 == ''
    var_13 = False
    var_14 = module_0.String(allow_blank=var_13)
    var_15 = ''
    var_16 = var_14.validate(var_15)
    var_17 = module_0.String()
    var_18 = 123
    var_19 = var_17.validate(var_18)
    var_20 = module_0.String(trim_whitespace=var_3)
    var_21 = '  hello  '
    var_22 = var_20.validate(var_21)
    assert var_22 == 'hello'
    var_23 = module_0.String(trim_whitespace=var_13)
    var_24 = var_23.validate(var_21)
    assert var_24 == '  hello  '
    var_25 = 3
    var_26 = module_0.String(min_length=var_25)
    var_27 = var_26.validate(var_18)
    assert var_27 == 'hello'
    var_28 = 'hi'
    var_29 = var_26.validate(var_28)
    var_30 = 5
    var_31 = module_0.String(max_length=var_30)
    var_32 = var_31.validate(var_28)
    assert var_32 == 'hello'
    var_33 = 'hello world'
    var_34 = var_31.validate(var_33)
    var_35 = '^\\d+$'
    var_36 = module_0.String(pattern=var_35)
    var_37 = '12345'
    var_38 = var_36.validate(var_37)
    assert var_38 == '12345'
    var_39 = 'abc'
    var_40 = var_36.validate(var_39)
    var_41 = module_0.String()
    var_42 = 'hel\x00lo'
    var_43 = var_41.validate(var_42)
    assert var_43 == 'hello'
    var_44 = module_0.String(allow_blank=var_3, coerce_types=var_3)
    var_45 = var_44.validate(var_5)
    assert var_45 == ''
    var_46 = module_0.String(trim_whitespace=var_3, coerce_types=var_3)
    var_47 = var_46.validate(var_11)
    assert var_47 is None
    var_48 = 'email'
    var_49 = module_0.String(format=var_48)
    var_50 = 'test@example.com'
    var_51 = var_49.validate(var_50)
    var_52 = 'date'
    var_53 = module_0.String(format=var_52)
    var_54 = 2023



# Parsed testcases at query #42
#--------------------------


import typesystem.fields as module_0

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
    var_10 = module_0.Union(var_9)
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = 0
    var_14 = exc_info.value.messages()[var_13]
    var_15 = var_14.code
    assert var_15 == 'null'
    var_16 = module_0.String()
    var_17 = module_0.Integer()
    var_18 = [var_16, var_17]
    var_19 = module_0.Union(var_18)
    var_20 = 'hello'
    var_21 = var_19.validate(var_20)
    assert var_21 == 'hello'
    var_22 = module_0.String()
    var_23 = module_0.Integer()
    var_24 = [var_22, var_23]
    var_25 = module_0.Union(var_24)
    var_26 = 42
    var_27 = var_25.validate(var_26)
    assert var_27 == 42
    var_28 = module_0.String()
    var_29 = module_0.Integer()
    var_30 = [var_28, var_29]
    var_31 = module_0.Union(var_30)
    var_32 = []
    var_33 = var_31.validate(var_32)
    var_34 = exc_info.value.messages()[var_13]
    var_35 = var_34.code
    assert var_35 == 'union'
    var_36 = module_0.String()
    var_37 = module_0.Integer()
    var_38 = [var_36, var_37]
    var_39 = module_0.Union(var_38)
    var_40 = var_39.validate(var_5)
    assert var_40 is None
    var_41 = module_0.String()
    var_42 = module_0.Integer()
    var_43 = [var_41, var_42]
    var_44 = module_0.Union(var_43)
    var_45 = var_44.validate(var_5)
    assert var_45 is None
    var_46 = 2
    var_47 = module_0.String(max_length=var_46)
    var_48 = module_0.Integer()
    var_49 = [var_47, var_48]
    var_50 = module_0.Union(var_49)
    var_51 = 'toolong'
    var_52 = var_50.validate(var_51)
    var_53 = exc_info.value.messages()[var_13]
    var_54 = var_53.code
    assert var_54 == 'max_length'
    var_55 = module_0.Float()
    var_56 = module_0.String()
    var_57 = [var_55, var_56]
    var_58 = module_0.Union(var_57)
    var_59 = 3.14
    var_60 = var_58.validate(var_59)
    var_61 = module_0.String()
    var_62 = module_0.Integer()
    var_63 = [var_61, var_62]
    var_64 = module_0.Union(var_63)
    var_65 = True
    var_66 = var_64.validate(var_65)
    var_67 = exc_info.value.messages()[var_13]
    var_68 = var_67.code
    assert var_68 == 'union'
    var_69 = module_0.String()
    var_70 = module_0.Integer()
    var_71 = [var_69, var_70]
    var_72 = module_0.Union(var_71)
    var_73 = 'key'
    var_74 = 'value'
    var_75 = {var_73: var_74}
    var_76 = var_72.validate(var_75)
    var_77 = exc_info.value.messages()[var_13]
    var_78 = var_77.code
    assert var_78 == 'union'
    var_79 = 10
    var_80 = module_0.Integer(minimum=var_79)
    var_81 = 5
    var_82 = module_0.Integer(maximum=var_81)
    var_83 = [var_80, var_82]
    var_84 = module_0.Union(var_83)
    var_85 = 7
    var_86 = var_84.validate(var_85)
    var_87 = exc_info.value.messages()[var_13]
    var_88 = var_87.code
    assert var_88 == 'union'
    var_89 = module_0.String()
    var_90 = module_0.Integer()
    var_91 = [var_89, var_90]
    var_92 = module_0.Union(var_91)
    var_93 = '123'
    var_94 = var_92.validate(var_93)
    assert var_94 == '123'
    var_95 = []
    var_96 = module_0.Union(var_95)
    var_97 = 'any value'
    var_98 = var_96.validate(var_97)
    var_99 = exc_info.value.messages()[var_13]
    var_100 = var_99.code
    assert var_100 == 'union'
    var_101 = 'name'
    var_102 = module_0.String()
    var_103 = {var_101: var_102}
    var_104 = module_0.Object(properties=var_103)
    var_105 = module_0.String()
    var_106 = module_0.Array(var_105)
    var_107 = [var_104, var_106]
    var_108 = module_0.Union(var_107)
    var_109 = 'John'
    var_110 = {var_101: var_109}
    var_111 = var_108.validate(var_110)
    var_112 = 'a'
    var_113 = 'b'
    var_114 = [var_112, var_113]
    var_115 = var_108.validate(var_114)
    var_116 = module_0.String()
    var_117 = module_0.Array(var_116)
    var_118 = module_0.String()
    var_119 = [var_117, var_118]
    var_120 = module_0.Union(var_119)
    var_121 = 123
    var_122 = [var_121]
    var_123 = var_120.validate(var_122)
    var_124 = exc_info.value.messages()[var_13]
    var_125 = var_124.code
    assert var_125 == 'union'



# Parsed testcases at query #43
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'red'
    var_1 = 'green'
    var_2 = 'blue'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Choice(choices=var_3)
    var_5 = var_4.validate(var_0)
    assert var_5 == 'red'
    var_6 = var_4.validate(var_1)
    assert var_6 == 'green'
    var_7 = var_4.validate(var_2)
    assert var_7 == 'blue'
    var_8 = 'r'
    var_9 = 'Red'
    var_10 = (var_8, var_9)
    var_11 = 'g'
    var_12 = 'Green'
    var_13 = (var_11, var_12)
    var_14 = 'b'
    var_15 = 'Blue'
    var_16 = (var_14, var_15)
    var_17 = [var_10, var_13, var_16]
    var_18 = module_0.Choice(choices=var_17)
    var_19 = var_18.validate(var_8)
    assert var_19 == 'r'
    var_20 = var_18.validate(var_11)
    assert var_20 == 'g'
    var_21 = var_18.validate(var_14)
    assert var_21 == 'b'
    var_22 = [var_0, var_1, var_2]
    var_23 = False
    var_24 = module_0.Choice(choices=var_22)
    var_25 = None
    var_26 = var_24.validate(var_25)
    var_27 = [var_25, var_26, var_2]
    var_28 = True
    var_29 = module_0.Choice(choices=var_27)
    var_30 = None
    var_31 = var_29.validate(var_30)
    assert var_31 is None
    var_32 = [var_25, var_26, var_2]
    var_33 = module_0.Choice(choices=var_32)
    var_34 = 'yellow'
    var_35 = var_33.validate(var_34)
    var_36 = [var_34, var_35, var_2]
    var_37 = module_0.Choice(choices=var_36, coerce_types=var_28)
    var_38 = ''
    var_39 = var_37.validate(var_38)
    assert var_39 is None
    var_40 = [var_34, var_35, var_2]
    var_41 = module_0.Choice(choices=var_40, coerce_types=var_28)
    var_42 = ''
    var_43 = var_41.validate(var_42)
    var_44 = [var_42, var_43, var_2]
    var_45 = module_0.Choice(choices=var_44, coerce_types=var_23)
    var_46 = ''
    var_47 = var_45.validate(var_46)
    var_48 = []
    var_49 = module_0.Choice(choices=var_48)
    var_50 = 'red'
    var_51 = var_49.validate(var_50)
    var_52 = 2
    var_53 = 3
    var_54 = [var_28, var_52, var_53]
    var_55 = module_0.Choice(choices=var_54)
    var_56 = var_55.validate(var_28)
    assert var_56 == 1
    var_57 = var_55.validate(var_52)
    assert var_57 == 2
    var_58 = [var_28, var_52, var_53]
    var_59 = module_0.Choice(choices=var_58)
    var_60 = 4
    var_61 = var_59.validate(var_60)



# Parsed testcases at query #44
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'red'
    var_1 = 'green'
    var_2 = 'blue'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Choice(choices=var_3)
    var_5 = var_4.validate(var_0)
    assert var_5 == 'red'
    var_6 = var_4.validate(var_1)
    assert var_6 == 'green'
    var_7 = var_4.validate(var_2)
    assert var_7 == 'blue'
    var_8 = 'r'
    var_9 = 'Red'
    var_10 = (var_8, var_9)
    var_11 = 'g'
    var_12 = 'Green'
    var_13 = (var_11, var_12)
    var_14 = 'b'
    var_15 = 'Blue'
    var_16 = (var_14, var_15)
    var_17 = [var_10, var_13, var_16]
    var_18 = module_0.Choice(choices=var_17)
    var_19 = var_18.validate(var_8)
    assert var_19 == 'r'
    var_20 = var_18.validate(var_11)
    assert var_20 == 'g'
    var_21 = var_18.validate(var_14)
    assert var_21 == 'b'
    var_22 = [var_0, var_1, var_2]
    var_23 = module_0.Choice(choices=var_22)
    var_24 = 'yellow'
    var_25 = var_23.validate(var_24)
    var_26 = [var_24, var_25]
    var_27 = False
    var_28 = module_0.Choice(choices=var_26)
    var_29 = None
    var_30 = var_28.validate(var_29)
    var_31 = [var_29, var_30]
    var_32 = True
    var_33 = module_0.Choice(choices=var_31)
    var_34 = None
    var_35 = var_33.validate(var_34)
    assert var_35 is None
    var_36 = [var_29, var_30]
    var_37 = module_0.Choice(choices=var_36, coerce_types=var_32)
    var_38 = ''
    var_39 = var_37.validate(var_38)
    assert var_39 is None
    var_40 = [var_29, var_30]
    var_41 = module_0.Choice(choices=var_40, coerce_types=var_32)
    var_42 = ''
    var_43 = var_41.validate(var_42)
    var_44 = [var_42, var_43]
    var_45 = module_0.Choice(choices=var_44, coerce_types=var_27)
    var_46 = ''
    var_47 = var_45.validate(var_46)
    var_48 = []
    var_49 = module_0.Choice(choices=var_48)
    var_50 = 'anything'
    var_51 = var_49.validate(var_50)
    var_52 = module_0.Choice(choices=var_34)
    var_53 = 'anything'
    var_54 = var_52.validate(var_53)
    var_55 = 'only'
    var_56 = [var_55]
    var_57 = module_0.Choice(choices=var_56)
    var_58 = var_57.validate(var_55)
    assert var_58 == 'only'
    var_59 = 'other'
    var_60 = var_57.validate(var_59)
    var_61 = 2
    var_62 = 3
    var_63 = [var_32, var_61, var_62]
    var_64 = module_0.Choice(choices=var_63)
    var_65 = var_64.validate(var_32)
    assert var_65 == 1
    var_66 = var_64.validate(var_61)
    assert var_66 == 2
    var_67 = 4
    var_68 = var_64.validate(var_67)



# Parsed testcases at query #45
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = module_0.Boolean()
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Union(var_3)
    var_5 = True
    var_6 = module_0.String()
    var_7 = module_0.Integer()
    var_8 = [var_6, var_7]
    var_9 = module_0.Union(var_8)
    var_10 = module_0.String()
    var_11 = module_0.Integer()
    var_12 = [var_10, var_11]
    var_13 = module_0.Union(var_12)
    var_14 = module_0.String()
    var_15 = [var_14]
    var_16 = module_0.Union(var_15)
    var_17 = []
    var_18 = module_0.Union(var_17)
    var_19 = module_0.String()
    var_20 = module_0.Integer()
    var_21 = [var_19, var_20]
    var_22 = False
    var_23 = 'Test union field'
    var_24 = module_0.Union(var_21)
    var_25 = module_0.String()
    var_26 = module_0.Integer()
    var_27 = module_0.Boolean()
    var_28 = [var_25, var_26, var_27]
    var_29 = module_0.Union(var_28)



####################################################################
#    TEST GENERATION BEGINS (CODAMOSA + claude-haiku-4-5 t=0.8)    #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.fields as module_0

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
    var_8 = exc_info.value.messages()[var_4]
    var_9 = var_8.code
    assert var_9 == 'null'
    var_10 = module_0.Array()
    var_11 = 'not a list'
    var_12 = var_10.validate(var_11)
    var_13 = exc_info.value.messages()[var_4]
    var_14 = var_13.code
    assert var_14 == 'type'
    var_15 = module_0.Array()
    var_16 = {}
    var_17 = var_15.validate(var_16)
    var_18 = exc_info.value.messages()[var_4]
    var_19 = var_18.code
    assert var_19 == 'type'
    var_20 = module_0.Array(min_items=var_16)
    var_21 = []
    var_22 = var_20.validate(var_21)
    var_23 = exc_info.value.messages()[var_4]
    var_24 = var_23.code
    assert var_24 == 'empty'
    var_25 = 3
    var_26 = module_0.Array(min_items=var_25)
    var_27 = 1
    var_28 = 2
    var_29 = [var_27, var_28]
    var_30 = var_26.validate(var_29)
    var_31 = exc_info.value.messages()[var_30]
    var_32 = var_31.code
    assert var_32 == 'min_items'
    var_33 = 2
    var_34 = module_0.Array(max_items=var_33)
    var_35 = 1
    var_36 = 2
    var_37 = 3
    var_38 = [var_35, var_36, var_37]
    var_39 = var_34.validate(var_38)
    var_40 = exc_info.value.messages()[var_38]
    var_41 = var_40.code
    assert var_41 == 'max_items'
    var_42 = module_0.Array(exact_items=var_33)
    var_43 = 1
    var_44 = 2
    var_45 = 3
    var_46 = [var_43, var_44, var_45]
    var_47 = var_42.validate(var_46)
    var_48 = exc_info.value.messages()[var_46]
    var_49 = var_48.code
    assert var_49 == 'max_items'
    var_50 = module_0.Array(exact_items=var_33)
    var_51 = [var_43, var_33]
    var_52 = var_50.validate(var_51)
    var_53 = module_0.Integer()
    var_54 = module_0.Array(var_53)
    var_55 = [var_43, var_33, var_25]
    var_56 = var_54.validate(var_55)
    var_57 = module_0.Integer()
    var_58 = module_0.Array(var_57)
    var_59 = 1
    var_60 = 'invalid'
    var_61 = 3
    var_62 = [var_59, var_60, var_61]
    var_63 = var_58.validate(var_62)
    var_64 = module_0.Integer()
    var_65 = module_0.String()
    var_66 = [var_64, var_65]
    var_67 = module_0.Array(var_66)
    var_68 = 'hello'
    var_69 = [var_59, var_68]
    var_70 = var_67.validate(var_69)
    var_71 = module_0.Integer()
    var_72 = module_0.String()
    var_73 = [var_71, var_72]
    var_74 = module_0.Array(var_73, var_62)
    var_75 = 1
    var_76 = 'hello'
    var_77 = 3
    var_78 = [var_75, var_76, var_77]
    var_79 = var_74.validate(var_78)
    var_80 = exc_info.value.messages()[var_78]
    var_81 = var_80.code
    assert var_81 == 'max_items'
    var_82 = module_0.Integer()
    var_83 = module_0.String()
    var_84 = [var_82, var_83]
    var_85 = module_0.Array(var_84, var_75)
    var_86 = [var_75, var_68, var_25]
    var_87 = var_85.validate(var_86)
    var_88 = module_0.Integer()
    var_89 = [var_88]
    var_90 = module_0.String()
    var_91 = module_0.Array(var_89, var_90)
    var_92 = [var_75, var_68]
    var_93 = var_91.validate(var_92)
    var_94 = module_0.Array(unique_items=var_75)
    var_95 = [var_75, var_33, var_25]
    var_96 = var_94.validate(var_95)
    var_97 = module_0.Array(unique_items=var_75)
    var_98 = 1
    var_99 = 2
    var_100 = [var_98, var_99, var_98]
    var_101 = var_97.validate(var_100)
    var_102 = 'unique_items'
    var_103 = module_0.Array(unique_items=var_98)
    var_104 = 'a'
    var_105 = 'b'
    var_106 = [var_104, var_105, var_104]
    var_107 = var_103.validate(var_106)
    var_108 = module_0.Array()
    var_109 = 'mixed'
    var_110 = 'key'
    var_111 = 'value'
    var_112 = {var_110: var_111}
    var_113 = [var_104, var_109, var_112]
    var_114 = var_108.validate(var_113)
    var_115 = module_0.Integer()
    var_116 = module_0.Array(var_115)
    var_117 = '1'
    var_118 = '2'
    var_119 = '3'
    var_120 = [var_117, var_118, var_119]
    var_121 = var_116.validate(var_120)
    var_122 = 5
    var_123 = module_0.Integer(minimum=var_122)
    var_124 = module_0.Array(var_123)
    var_125 = 1
    var_126 = 10
    var_127 = 3
    var_128 = [var_125, var_126, var_127]
    var_129 = var_124.validate(var_128)
    var_130 = 'minimum'
    var_131 = 'id'
    var_132 = 'name'
    var_133 = module_0.Integer()
    var_134 = module_0.String()
    var_135 = {var_131: var_133, var_132: var_134}
    var_136 = module_0.Object(properties=var_135)
    var_137 = module_0.Array(var_136)
    var_138 = 'test'
    var_139 = {var_131: var_125, var_132: var_138}
    var_140 = [var_139]
    var_141 = var_137.validate(var_140)
    var_142 = module_0.Array()
    var_143 = []
    var_144 = var_142.validate(var_143)
    var_145 = module_0.Integer()
    var_146 = module_0.Array(var_145)
    var_147 = [var_125, var_126, var_25]
    var_148 = var_146.validate(var_147)
    var_149 = module_0.Array(min_items=var_33)
    var_150 = [var_125, var_33]
    var_151 = var_149.validate(var_150)
    var_152 = module_0.Array(max_items=var_33)
    var_153 = [var_125, var_33]
    var_154 = var_152.validate(var_153)



# Parsed testcases at query #2
#--------------------------


import typesystem.fields as module_0

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
    var_8 = exc_info.value.messages()[var_4]
    var_9 = var_8.code
    assert var_9 == 'null'
    var_10 = module_0.Array()
    var_11 = 'not a list'
    var_12 = var_10.validate(var_11)
    var_13 = exc_info.value.messages()[var_4]
    var_14 = var_13.code
    assert var_14 == 'type'
    var_15 = 'key'
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = var_10.validate(var_17)
    var_19 = exc_info.value.messages()[var_18]
    var_20 = var_19.code
    assert var_20 == 'type'
    var_21 = 3
    var_22 = module_0.Array(exact_items=var_21)
    var_23 = 2
    var_24 = [var_15, var_23, var_21]
    var_25 = var_22.validate(var_24)
    var_26 = module_0.Array(exact_items=var_21)
    var_27 = 1
    var_28 = 2
    var_29 = [var_27, var_28]
    var_30 = var_26.validate(var_29)
    var_31 = exc_info.value.messages()[var_30]
    var_32 = var_31.code
    assert var_32 == 'exact_items'
    var_33 = module_0.Array(min_items=var_23)
    var_34 = 1
    var_35 = [var_34]
    var_36 = var_33.validate(var_35)
    var_37 = exc_info.value.messages()[var_30]
    var_38 = var_37.code
    assert var_38 == 'min_items'
    var_39 = module_0.Array(min_items=var_34)
    var_40 = []
    var_41 = var_39.validate(var_40)
    var_42 = exc_info.value.messages()[var_30]
    var_43 = var_42.code
    assert var_43 == 'empty'
    var_44 = module_0.Array(max_items=var_23)
    var_45 = 1
    var_46 = 2
    var_47 = 3
    var_48 = [var_45, var_46, var_47]
    var_49 = var_44.validate(var_48)
    var_50 = exc_info.value.messages()[var_48]
    var_51 = var_50.code
    assert var_51 == 'max_items'
    var_52 = module_0.Integer()
    var_53 = module_0.Array(var_52)
    var_54 = [var_45, var_23, var_21]
    var_55 = var_53.validate(var_54)
    var_56 = module_0.Integer()
    var_57 = module_0.Array(var_56)
    var_58 = 1
    var_59 = 'not an integer'
    var_60 = 3
    var_61 = [var_58, var_59, var_60]
    var_62 = var_57.validate(var_61)
    var_63 = module_0.Integer()
    var_64 = module_0.String()
    var_65 = [var_63, var_64]
    var_66 = module_0.Array(var_65)
    var_67 = 'hello'
    var_68 = [var_58, var_67]
    var_69 = var_66.validate(var_68)
    var_70 = module_0.Integer()
    var_71 = module_0.String()
    var_72 = [var_70, var_71]
    var_73 = module_0.Array(var_72, var_61)
    var_74 = [var_58, var_67]
    var_75 = var_73.validate(var_74)
    var_76 = module_0.Integer()
    var_77 = module_0.String()
    var_78 = [var_76, var_77]
    var_79 = module_0.Integer()
    var_80 = module_0.Array(var_78, var_79)
    var_81 = 42
    var_82 = [var_58, var_67, var_81]
    var_83 = var_80.validate(var_82)
    var_84 = module_0.Array(unique_items=var_58)
    var_85 = [var_58, var_23, var_21]
    var_86 = var_84.validate(var_85)
    var_87 = module_0.Array(unique_items=var_58)
    var_88 = 1
    var_89 = 2
    var_90 = [var_88, var_89, var_88]
    var_91 = var_87.validate(var_90)
    var_92 = exc_info.value.messages()[var_91]
    var_93 = var_92.code
    assert var_93 == 'unique_items'
    var_94 = module_0.Array(unique_items=var_88)
    var_95 = 'a'
    var_96 = 'b'
    var_97 = [var_95, var_96, var_95]
    var_98 = var_94.validate(var_97)
    var_99 = exc_info.value.messages()[var_98]
    var_100 = var_99.code
    assert var_100 == 'unique_items'
    var_101 = module_0.Array()
    var_102 = []
    var_103 = var_101.validate(var_102)
    var_104 = 5
    var_105 = module_0.Integer(minimum=var_104)
    var_106 = module_0.Array(var_105)
    var_107 = 10
    var_108 = 3
    var_109 = 8
    var_110 = [var_107, var_108, var_109]
    var_111 = var_106.validate(var_110)
    var_112 = module_0.Integer()
    var_113 = module_0.String()
    var_114 = [var_112, var_113]
    var_115 = module_0.Array(var_114)
    var_116 = 'not an int'
    var_117 = 'hello'
    var_118 = [var_116, var_117]
    var_119 = var_115.validate(var_118)
    var_120 = 4
    var_121 = module_0.Array(min_items=var_23, max_items=var_120)
    var_122 = [var_116, var_23, var_21]
    var_123 = var_121.validate(var_122)
    var_124 = module_0.Array(min_items=var_23, max_items=var_120)
    var_125 = 1
    var_126 = [var_125]
    var_127 = var_124.validate(var_126)
    var_128 = exc_info.value.messages()[var_119]
    var_129 = var_128.code
    assert var_129 == 'min_items'
    var_130 = module_0.Array(min_items=var_23, max_items=var_120)
    var_131 = 1
    var_132 = 2
    var_133 = 3
    var_134 = 4
    var_135 = 5
    var_136 = [var_131, var_132, var_133, var_134, var_135]
    var_137 = var_130.validate(var_136)
    var_138 = exc_info.value.messages()[var_134]
    var_139 = var_138.code
    assert var_139 == 'max_items'
    var_140 = module_0.Array(var_132)
    var_141 = 'string'
    var_142 = 'key'
    var_143 = 'value'
    var_144 = {var_142: var_143}
    var_145 = [var_131, var_141, var_144]
    var_146 = var_140.validate(var_145)
    var_147 = module_0.Integer()
    var_148 = [var_147]
    var_149 = module_0.Array(var_148, var_134)
    var_150 = [var_131]
    var_151 = var_149.validate(var_150)
    var_152 = module_0.Integer()
    var_153 = module_0.Array(var_152)
    var_154 = [var_131, var_23, var_21]
    var_155 = var_153.validate(var_154)



# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Const(var_0)
    var_2 = 'hello'
    var_3 = module_0.Const(var_2)
    var_4 = None
    var_5 = module_0.Const(var_4)
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]
    var_10 = module_0.Const(var_9)
    var_11 = 'key'
    var_12 = 'value'
    var_13 = {var_11: var_12}
    var_14 = module_0.Const(var_13)
    var_15 = True
    var_16 = module_0.Const(var_15)
    var_17 = 42
    var_18 = True
    var_19 = module_0.Const(var_17)
    var_20 = 100
    var_21 = 'A constant field'
    var_22 = module_0.Const(var_20)
    var_23 = 50
    var_24 = module_0.Const(var_23)
    var_25 = 3.14
    var_26 = module_0.Const(var_25)
    var_27 = ''
    var_28 = module_0.Const(var_27)
    var_29 = 0
    var_30 = module_0.Const(var_29)
    var_31 = False
    var_32 = module_0.Const(var_31)



# Parsed testcases at query #4
#--------------------------


import typesystem.fields as module_0

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
    var_8 = exc_info.value.messages()[var_4]
    var_9 = var_8.code
    assert var_9 == 'null'
    var_10 = module_0.Object()
    var_11 = 'not a dict'
    var_12 = var_10.validate(var_11)
    var_13 = exc_info.value.messages()[var_4]
    var_14 = var_13.code
    assert var_14 == 'type'
    var_15 = module_0.Object()
    var_16 = {}
    var_17 = var_15.validate(var_16)
    var_18 = 'name'
    var_19 = module_0.String()
    var_20 = {var_18: var_19}
    var_21 = module_0.Object(properties=var_20)
    var_22 = 'John'
    var_23 = {var_18: var_22}
    var_24 = var_21.validate(var_23)
    var_25 = module_0.String()
    var_26 = {var_18: var_25}
    var_27 = [var_18]
    var_28 = module_0.Object(properties=var_26, required=var_27)
    var_29 = {}
    var_30 = var_28.validate(var_29)
    var_31 = module_0.Object()
    var_32 = 123
    var_33 = 'value'
    var_34 = {var_32: var_33}
    var_35 = var_31.validate(var_34)
    var_36 = exc_info.value.messages()[var_35]
    var_37 = var_36.code
    assert var_37 == 'invalid_key'
    var_38 = 'Unknown'
    var_39 = module_0.String()
    var_40 = {var_18: var_39}
    var_41 = module_0.Object(properties=var_40)
    var_42 = {}
    var_43 = var_41.validate(var_42)
    var_44 = '^[a-z]+$'
    var_45 = module_0.String(pattern=var_44)
    var_46 = module_0.Object(property_names=var_45)
    var_47 = 'Name'
    var_48 = 'value'
    var_49 = {var_47: var_48}
    var_50 = var_46.validate(var_49)
    var_51 = exc_info.value.messages()[var_50]
    var_52 = var_51.code
    assert var_52 == 'invalid_property'
    var_53 = 2
    var_54 = module_0.Object(min_properties=var_53)
    var_55 = 'a'
    var_56 = 1
    var_57 = {var_55: var_56}
    var_58 = var_54.validate(var_57)
    var_59 = exc_info.value.messages()[var_58]
    var_60 = var_59.code
    assert var_60 == 'min_properties'
    var_61 = module_0.Object(min_properties=var_55)
    var_62 = {}
    var_63 = var_61.validate(var_62)
    var_64 = exc_info.value.messages()[var_58]
    var_65 = var_64.code
    assert var_65 == 'empty'
    var_66 = module_0.Object(max_properties=var_62)
    var_67 = 'a'
    var_68 = 'b'
    var_69 = 1
    var_70 = 2
    var_71 = {var_67: var_69, var_68: var_70}
    var_72 = var_66.validate(var_71)
    var_73 = exc_info.value.messages()[var_70]
    var_74 = var_73.code
    assert var_74 == 'max_properties'
    var_75 = module_0.Object(additional_properties=var_67)
    var_76 = 'extra'
    var_77 = 'value'
    var_78 = {var_76: var_77}
    var_79 = var_75.validate(var_78)
    var_80 = module_0.Object(additional_properties=var_70)
    var_81 = 'extra'
    var_82 = 'value'
    var_83 = {var_81: var_82}
    var_84 = var_80.validate(var_83)
    var_85 = exc_info.value.messages()[var_84]
    var_86 = var_85.code
    assert var_86 == 'invalid_property'
    var_87 = module_0.Integer()
    var_88 = module_0.Object(additional_properties=var_87)
    var_89 = 'count'
    var_90 = 42
    var_91 = {var_89: var_90}
    var_92 = var_88.validate(var_91)
    var_93 = '^num'
    var_94 = module_0.Integer()
    var_95 = {var_93: var_94}
    var_96 = module_0.Object(pattern_properties=var_95)
    var_97 = 'number'
    var_98 = 123
    var_99 = {var_97: var_98}
    var_100 = var_96.validate(var_99)
    var_101 = 5
    var_102 = module_0.String(min_length=var_101)
    var_103 = {var_18: var_102}
    var_104 = module_0.Object(properties=var_103)
    var_105 = 'name'
    var_106 = 'Jo'
    var_107 = {var_105: var_106}
    var_108 = var_104.validate(var_107)
    var_109 = exc_info.value.messages()[var_108]
    var_110 = var_109.code
    assert var_110 == 'min_length'
    var_111 = 'id'
    var_112 = module_0.Integer()
    var_113 = module_0.String()
    var_114 = {var_111: var_112, var_18: var_113}
    var_115 = [var_111]
    var_116 = module_0.Object(properties=var_114, additional_properties=var_108, required=var_115)
    var_117 = 'Test'
    var_118 = {var_111: var_105, var_18: var_117}
    var_119 = var_116.validate(var_118)
    var_120 = module_0.Object()
    var_121 = 'key'
    var_122 = (var_121, var_77)
    var_123 = [var_122]



# Parsed testcases at query #5
#--------------------------


import typesystem.fields as module_0

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
    var_10 = module_0.Boolean()
    var_11 = var_10.validate(var_4)
    assert var_11 is False
    var_12 = module_0.Boolean(coerce_types=var_6)
    var_13 = 'true'
    var_14 = var_12.validate(var_13)
    assert var_14 is True
    var_15 = module_0.Boolean(coerce_types=var_6)
    var_16 = 'false'
    var_17 = var_15.validate(var_16)
    assert var_17 is False
    var_18 = module_0.Boolean(coerce_types=var_6)
    var_19 = 'on'
    var_20 = var_18.validate(var_19)
    assert var_20 is True
    var_21 = module_0.Boolean(coerce_types=var_6)
    var_22 = 'off'
    var_23 = var_21.validate(var_22)
    assert var_23 is False
    var_24 = module_0.Boolean(coerce_types=var_6)
    var_25 = '1'
    var_26 = var_24.validate(var_25)
    assert var_26 is True
    var_27 = module_0.Boolean(coerce_types=var_6)
    var_28 = '0'
    var_29 = var_27.validate(var_28)
    assert var_29 is False
    var_30 = module_0.Boolean(coerce_types=var_6)
    var_31 = ''
    var_32 = var_30.validate(var_31)
    assert var_32 is False
    var_33 = module_0.Boolean(coerce_types=var_6)
    var_34 = var_33.validate(var_6)
    assert var_34 is True
    var_35 = module_0.Boolean(coerce_types=var_6)
    var_36 = var_35.validate(var_4)
    assert var_36 is False
    var_37 = module_0.Boolean(coerce_types=var_6)
    var_38 = 'TRUE'
    var_39 = var_37.validate(var_38)
    assert var_39 is True
    var_40 = module_0.Boolean(coerce_types=var_6)
    var_41 = 'null'
    var_42 = var_40.validate(var_41)
    assert var_42 is None
    var_43 = 'none'
    var_44 = var_40.validate(var_43)
    assert var_44 is None
    var_45 = var_40.validate(var_31)
    assert var_45 is None
    var_46 = module_0.Boolean(coerce_types=var_6)
    var_47 = var_46.validate(var_31)
    assert var_47 is False
    var_48 = module_0.Boolean(coerce_types=var_6)
    var_49 = 'invalid'
    var_50 = var_48.validate(var_49)
    var_51 = module_0.Boolean(coerce_types=var_4)
    var_52 = 'true'
    var_53 = var_51.validate(var_52)
    var_54 = module_0.Boolean(coerce_types=var_52)
    var_55 = []
    var_56 = var_54.validate(var_55)
    var_57 = module_0.Boolean(coerce_types=var_55)
    var_58 = {}
    var_59 = var_57.validate(var_58)



# Parsed testcases at query #6
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'red'
    var_1 = 'green'
    var_2 = 'blue'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Choice(choices=var_3)
    var_5 = var_4.choices
    var_6 = len(var_5)
    assert var_6 == 3
    var_7 = 'r'
    var_8 = 'Red'
    var_9 = (var_7, var_8)
    var_10 = 'g'
    var_11 = 'Green'
    var_12 = (var_10, var_11)
    var_13 = 'b'
    var_14 = 'Blue'
    var_15 = (var_13, var_14)
    var_16 = [var_9, var_12, var_15]
    var_17 = module_0.Choice(choices=var_16)
    var_18 = var_17.choices
    var_19 = len(var_18)
    assert var_19 == 3
    var_20 = (var_10, var_11)
    var_21 = [var_0, var_20]
    var_22 = module_0.Choice(choices=var_21)
    var_23 = var_22.choices
    var_24 = len(var_23)
    assert var_24 == 2
    var_25 = []
    var_26 = module_0.Choice(choices=var_25)
    var_27 = None
    var_28 = module_0.Choice(choices=var_27)
    var_29 = 'a'
    var_30 = [var_29, var_13]
    var_31 = 'Select Option'
    var_32 = 'Choose one option'
    var_33 = module_0.Choice(choices=var_30)
    var_34 = [var_29, var_13]
    var_35 = module_0.Choice(choices=var_34)
    var_36 = [var_29, var_13]
    var_37 = True
    var_38 = module_0.Choice(choices=var_36)
    var_39 = [var_29, var_13]
    var_40 = module_0.Choice(choices=var_39)
    var_41 = [var_29, var_13]
    var_42 = False
    var_43 = module_0.Choice(choices=var_41, coerce_types=var_42)
    var_44 = [var_29, var_13]
    var_45 = module_0.Choice(choices=var_44)
    var_46 = 'x'
    var_47 = 'X'
    var_48 = [var_46, var_47]
    var_49 = 'y'
    var_50 = 'Y'
    var_51 = [var_49, var_50]
    var_52 = [var_48, var_51]
    var_53 = module_0.Choice(choices=var_52)
    var_54 = [var_29, var_13]
    var_55 = module_0.Choice(choices=var_54)



# Parsed testcases at query #7
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'red'
    var_1 = 'green'
    var_2 = 'blue'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Choice(choices=var_3)
    var_5 = var_4.validate(var_0)
    assert var_5 == 'red'
    var_6 = var_4.validate(var_1)
    assert var_6 == 'green'
    var_7 = var_4.validate(var_2)
    assert var_7 == 'blue'
    var_8 = [var_0, var_1, var_2]
    var_9 = module_0.Choice(choices=var_8)
    var_10 = 'yellow'
    var_11 = var_9.validate(var_10)
    var_12 = 'r'
    var_13 = 'Red'
    var_14 = (var_12, var_13)
    var_15 = 'g'
    var_16 = 'Green'
    var_17 = (var_15, var_16)
    var_18 = 'b'
    var_19 = 'Blue'
    var_20 = (var_18, var_19)
    var_21 = [var_14, var_17, var_20]
    var_22 = module_0.Choice(choices=var_21)
    var_23 = var_22.validate(var_12)
    assert var_23 == 'r'
    var_24 = var_22.validate(var_15)
    assert var_24 == 'g'
    var_25 = var_22.validate(var_18)
    assert var_25 == 'b'
    var_26 = [var_10, var_11, var_2]
    var_27 = module_0.Choice(choices=var_26)
    var_28 = None
    var_29 = var_27.validate(var_28)
    var_30 = [var_28, var_29, var_2]
    var_31 = True
    var_32 = module_0.Choice(choices=var_30)
    var_33 = None
    var_34 = var_32.validate(var_33)
    assert var_34 is None
    var_35 = [var_28, var_29, var_2]
    var_36 = False
    var_37 = module_0.Choice(choices=var_35, coerce_types=var_31)
    var_38 = ''
    var_39 = var_37.validate(var_38)
    var_40 = [var_38, var_39, var_2]
    var_41 = module_0.Choice(choices=var_40, coerce_types=var_31)
    var_42 = ''
    var_43 = var_41.validate(var_42)
    assert var_43 is None
    var_44 = [var_38, var_39, var_2]
    var_45 = module_0.Choice(choices=var_44, coerce_types=var_36)
    var_46 = ''
    var_47 = var_45.validate(var_46)
    var_48 = [var_46, var_47, var_2]
    var_49 = module_0.Choice(choices=var_48, coerce_types=var_36)
    var_50 = var_49.validate(var_42)
    assert var_50 is None
    var_51 = '1'
    var_52 = 'Option 1'
    var_53 = (var_51, var_52)
    var_54 = 'two'
    var_55 = '3'
    var_56 = 'Option 3'
    var_57 = (var_55, var_56)
    var_58 = [var_53, var_54, var_57]
    var_59 = module_0.Choice(choices=var_58)
    var_60 = var_59.validate(var_51)
    assert var_60 == '1'
    var_61 = var_59.validate(var_54)
    assert var_61 == 'two'
    var_62 = var_59.validate(var_55)
    assert var_62 == '3'
    var_63 = []
    var_64 = module_0.Choice(choices=var_63)
    var_65 = 'anything'
    var_66 = var_64.validate(var_65)
    var_67 = '2'
    var_68 = [var_51, var_67, var_55]
    var_69 = module_0.Choice(choices=var_68)
    var_70 = var_69.validate(var_51)
    assert var_70 == '1'
    var_71 = '4'
    var_72 = var_69.validate(var_71)



# Parsed testcases at query #8
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'red'
    var_1 = 'green'
    var_2 = 'blue'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Choice(choices=var_3)
    var_5 = var_4.validate(var_0)
    assert var_5 == 'red'
    var_6 = var_4.validate(var_1)
    assert var_6 == 'green'
    var_7 = var_4.validate(var_2)
    assert var_7 == 'blue'
    var_8 = '1'
    var_9 = 'Option 1'
    var_10 = (var_8, var_9)
    var_11 = '2'
    var_12 = 'Option 2'
    var_13 = (var_11, var_12)
    var_14 = [var_10, var_13]
    var_15 = module_0.Choice(choices=var_14)
    var_16 = var_15.validate(var_8)
    assert var_16 == '1'
    var_17 = var_15.validate(var_11)
    assert var_17 == '2'
    var_18 = [var_0, var_1, var_2]
    var_19 = module_0.Choice(choices=var_18)
    var_20 = 'yellow'
    var_21 = var_19.validate(var_20)
    var_22 = [var_20, var_21, var_2]
    var_23 = module_0.Choice(choices=var_22)
    var_24 = None
    var_25 = var_23.validate(var_24)
    var_26 = [var_24, var_25, var_2]
    var_27 = True
    var_28 = module_0.Choice(choices=var_26)
    var_29 = None
    var_30 = var_28.validate(var_29)
    assert var_30 is None
    var_31 = [var_24, var_25, var_2]
    var_32 = module_0.Choice(choices=var_31, coerce_types=var_27)
    var_33 = ''
    var_34 = var_32.validate(var_33)
    assert var_34 is None
    var_35 = [var_24, var_25, var_2]
    var_36 = False
    var_37 = module_0.Choice(choices=var_35, coerce_types=var_36)
    var_38 = ''
    var_39 = var_37.validate(var_38)
    var_40 = [var_38, var_39, var_2]
    var_41 = module_0.Choice(choices=var_40, coerce_types=var_27)
    var_42 = ''
    var_43 = var_41.validate(var_42)
    var_44 = []
    var_45 = module_0.Choice(choices=var_44)
    var_46 = 'anything'
    var_47 = var_45.validate(var_46)
    var_48 = 'One'
    var_49 = (var_27, var_48)
    var_50 = 2
    var_51 = 'Two'
    var_52 = (var_50, var_51)
    var_53 = 3
    var_54 = 'Three'
    var_55 = (var_53, var_54)
    var_56 = [var_49, var_52, var_55]
    var_57 = module_0.Choice(choices=var_56)
    var_58 = var_57.validate(var_27)
    assert var_58 == 1
    var_59 = var_57.validate(var_50)
    assert var_59 == 2
    var_60 = 'simple'
    var_61 = 'complex'
    var_62 = 'Complex Option'
    var_63 = (var_61, var_62)
    var_64 = [var_60, var_63]
    var_65 = module_0.Choice(choices=var_64)
    var_66 = var_65.validate(var_60)
    assert var_66 == 'simple'
    var_67 = var_65.validate(var_61)
    assert var_67 == 'complex'



# Parsed testcases at query #9
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = 42
    var_2 = var_0.validate(var_1)
    assert var_2 == 42
    var_3 = 3.14
    var_4 = var_0.validate(var_3)
    var_5 = '42'
    var_6 = var_0.validate(var_5)
    assert var_6 == 42
    var_7 = '3.14'
    var_8 = var_0.validate(var_7)
    var_9 = True
    var_10 = module_0.Number()
    var_11 = None
    var_12 = var_10.validate(var_11)
    assert var_12 is None
    var_13 = False
    var_14 = module_0.Number()
    var_15 = None
    var_16 = var_14.validate(var_15)
    var_17 = module_0.Number(coerce_types=var_9)
    var_18 = ''
    var_19 = var_17.validate(var_18)
    assert var_19 is None
    var_20 = module_0.Number()
    var_21 = True
    var_22 = var_20.validate(var_21)
    var_23 = 3.14
    var_24 = module_0.Number(coerce_types=var_13)
    var_25 = '42'
    var_26 = var_24.validate(var_25)
    var_27 = 10
    var_28 = module_0.Number(minimum=var_27)
    var_29 = var_28.validate(var_27)
    assert var_29 == 10
    var_30 = 20
    var_31 = var_28.validate(var_30)
    assert var_31 == 20
    var_32 = 5
    var_33 = var_28.validate(var_32)
    var_34 = 100
    var_35 = module_0.Number(maximum=var_34)
    var_36 = var_35.validate(var_34)
    assert var_36 == 100
    var_37 = 50
    var_38 = var_35.validate(var_37)
    assert var_38 == 50
    var_39 = 150
    var_40 = var_35.validate(var_39)
    var_41 = module_0.Number(exclusive_minimum=var_27)
    var_42 = 10.1
    var_43 = var_41.validate(var_42)
    var_44 = 10
    var_45 = var_41.validate(var_44)
    var_46 = 5
    var_47 = var_41.validate(var_46)
    var_48 = module_0.Number(exclusive_maximum=var_34)
    var_49 = 99.9
    var_50 = var_48.validate(var_49)
    var_51 = 100
    var_52 = var_48.validate(var_51)
    var_53 = 150
    var_54 = var_48.validate(var_53)
    var_55 = module_0.Number()
    var_56 = 'inf'
    var_57 = float(var_56)
    var_58 = var_55.validate(var_57)
    var_59 = '-inf'
    var_60 = float(var_59)
    var_61 = var_55.validate(var_60)
    var_62 = 'nan'
    var_63 = float(var_62)
    var_64 = var_55.validate(var_63)
    var_65 = 5
    var_66 = module_0.Number(multiple_of=var_65)
    var_67 = var_66.validate(var_27)
    assert var_67 == 10
    var_68 = 15
    var_69 = var_66.validate(var_68)
    assert var_69 == 15
    var_70 = 7
    var_71 = var_66.validate(var_70)
    var_72 = 0.5
    var_73 = module_0.Number(multiple_of=var_72)
    var_74 = var_73.validate(var_9)
    var_75 = 1.5
    var_76 = var_73.validate(var_75)
    var_77 = 1.3
    var_78 = var_73.validate(var_77)
    var_79 = '0.01'
    var_80 = 3.14159
    var_81 = module_0.Number()
    var_82 = 'not_a_number'
    var_83 = var_81.validate(var_82)
    var_84 = module_0.Number()
    var_85 = '42.50'



# Parsed testcases at query #10
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = var_0.get_default_value()
    assert var_1 is None
    var_2 = 'test_value'
    var_3 = module_0.Field(default=var_2)
    var_4 = var_3.get_default_value()
    assert var_4 == 'test_value'
    var_5 = 42
    var_6 = module_0.Field(default=var_5)
    var_7 = var_6.get_default_value()
    assert var_7 == 42
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = [var_8, var_9, var_10]
    var_12 = lambda : var_11
    var_13 = module_0.Field(default=var_12)
    var_14 = var_13.get_default_value()
    var_15 = None
    var_16 = module_0.Field(default=var_15)
    var_17 = var_16.get_default_value()
    assert var_17 is None
    var_18 = True
    var_19 = module_0.Field(allow_null=var_18)
    var_20 = var_19.get_default_value()
    assert var_20 is None
    var_21 = ''
    var_22 = module_0.Field(default=var_21)
    var_23 = var_22.get_default_value()
    assert var_23 == ''
    var_24 = 0
    var_25 = module_0.Field(default=var_24)
    var_26 = var_25.get_default_value()
    assert var_26 == 0
    var_27 = False
    var_28 = module_0.Field(default=var_27)
    var_29 = var_28.get_default_value()
    assert var_29 is False
    var_30 = [var_18, var_9, var_10]
    var_31 = module_0.Field(default=var_30)
    var_32 = var_31.get_default_value()
    var_33 = 'key'
    var_34 = 'value'
    var_35 = {var_33: var_34}
    var_36 = module_0.Field(default=var_35)
    var_37 = var_36.get_default_value()



# Parsed testcases at query #11
#--------------------------


import typesystem.fields as module_0

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
    var_12 = 'key'
    var_13 = 'value'
    var_14 = {var_12: var_13}
    var_15 = var_11.validate(var_14)
    var_16 = module_0.Object()
    var_17 = 1
    var_18 = 'value'
    var_19 = {var_17: var_18}
    var_20 = var_16.validate(var_19)
    var_21 = 'name'
    var_22 = [var_21]
    var_23 = module_0.Object(required=var_22)
    var_24 = {}
    var_25 = var_23.validate(var_24)
    var_26 = 'John'
    var_27 = {var_21: var_26}
    var_28 = var_23.validate(var_27)
    var_29 = 'age'
    var_30 = module_0.String()
    var_31 = module_0.Integer()
    var_32 = {var_21: var_30, var_29: var_31}
    var_33 = module_0.Object(properties=var_32)
    var_34 = {var_21: var_26}
    var_35 = var_33.validate(var_34)
    var_36 = 2
    var_37 = module_0.Object(min_properties=var_36)
    var_38 = 'key'
    var_39 = 'value'
    var_40 = {var_38: var_39}
    var_41 = var_37.validate(var_40)
    var_42 = 3
    var_43 = module_0.Object(min_properties=var_42)
    var_44 = 'key1'
    var_45 = 'key2'
    var_46 = 'value1'
    var_47 = 'value2'
    var_48 = {var_44: var_46, var_45: var_47}
    var_49 = var_43.validate(var_48)
    var_50 = module_0.Object(max_properties=var_44)
    var_51 = 'key1'
    var_52 = 'key2'
    var_53 = 'value1'
    var_54 = 'value2'
    var_55 = {var_51: var_53, var_52: var_54}
    var_56 = var_50.validate(var_55)
    var_57 = module_0.String()
    var_58 = {var_21: var_57}
    var_59 = module_0.Object(properties=var_58, additional_properties=var_54)
    var_60 = 'name'
    var_61 = 'extra'
    var_62 = 'John'
    var_63 = 'field'
    var_64 = {var_60: var_62, var_61: var_63}
    var_65 = var_59.validate(var_64)
    var_66 = {var_21: var_26}
    var_67 = var_59.validate(var_66)
    var_68 = module_0.String()
    var_69 = {var_21: var_68}
    var_70 = module_0.Object(properties=var_69, additional_properties=var_60)
    var_71 = 'extra'
    var_72 = 'field'
    var_73 = {var_21: var_26, var_71: var_72}
    var_74 = var_70.validate(var_73)
    var_75 = module_0.String()
    var_76 = {var_21: var_75}
    var_77 = module_0.Integer()
    var_78 = module_0.Object(properties=var_76, additional_properties=var_77)
    var_79 = 'count'
    var_80 = 42
    var_81 = {var_21: var_26, var_79: var_80}
    var_82 = var_78.validate(var_81)
    var_83 = '^num_'
    var_84 = module_0.Integer()
    var_85 = {var_83: var_84}
    var_86 = module_0.Object(pattern_properties=var_85)
    var_87 = 'num_one'
    var_88 = 'num_two'
    var_89 = {var_87: var_60, var_88: var_36}
    var_90 = var_86.validate(var_89)
    var_91 = '^[a-z]+$'
    var_92 = module_0.String(pattern=var_91)
    var_93 = module_0.Object(property_names=var_92)
    var_94 = 'Invalid'
    var_95 = 'value'
    var_96 = {var_94: var_95}
    var_97 = var_93.validate(var_96)
    var_98 = 'valid'
    var_99 = {var_98: var_65}
    var_100 = var_93.validate(var_99)
    var_101 = module_0.Integer(minimum=var_97)
    var_102 = {var_29: var_101}
    var_103 = module_0.Object(properties=var_102)
    var_104 = 'age'
    var_105 = -5
    var_106 = {var_104: var_105}
    var_107 = var_103.validate(var_106)
    var_108 = 'user'
    var_109 = module_0.String()
    var_110 = module_0.Integer()
    var_111 = {var_21: var_109, var_29: var_110}
    var_112 = [var_21]
    var_113 = module_0.Object(properties=var_111, required=var_112)
    var_114 = {var_108: var_113}
    var_115 = module_0.Object(properties=var_114)
    var_116 = 30
    var_117 = {var_21: var_26, var_29: var_116}
    var_118 = {var_108: var_117}
    var_119 = var_115.validate(var_118)
    var_120 = module_0.Object()
    var_121 = (var_64, var_65)
    var_122 = [var_121]



# Parsed testcases at query #12
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Array(var_0)
    var_2 = None
    var_3 = var_1.serialize(var_2)
    assert var_3 is None
    var_4 = module_0.String()
    var_5 = module_0.Integer()
    var_6 = [var_4, var_5]
    var_7 = module_0.Array(var_6)
    var_8 = 'hello'
    var_9 = 42
    var_10 = [var_8, var_9]
    var_11 = var_7.serialize(var_10)
    var_12 = module_0.String()
    var_13 = module_0.Array(var_12)
    var_14 = 'a'
    var_15 = 'b'
    var_16 = 'c'
    var_17 = [var_14, var_15, var_16]
    var_18 = var_13.serialize(var_17)
    var_19 = module_0.Array(var_2)
    var_20 = 1
    var_21 = 'two'
    var_22 = 3.0
    var_23 = [var_20, var_21, var_22]
    var_24 = var_19.serialize(var_23)
    var_25 = module_0.Decimal()
    var_26 = module_0.Array(var_25)
    var_27 = '1.5'
    var_28 = '2.5'
    var_29 = module_0.Integer()
    var_30 = module_0.Array(var_29)
    var_31 = module_0.Array(var_30)
    var_32 = 2
    var_33 = [var_20, var_32]
    var_34 = 4
    var_35 = [var_22, var_34]
    var_36 = [var_33, var_35]
    var_37 = var_31.serialize(var_36)
    var_38 = 'name'
    var_39 = 'age'
    var_40 = module_0.String()
    var_41 = module_0.Integer()
    var_42 = {var_38: var_40, var_39: var_41}
    var_43 = module_0.Object(properties=var_42)
    var_44 = module_0.Array(var_43)
    var_45 = 'Alice'
    var_46 = 30
    var_47 = {var_38: var_45, var_39: var_46}
    var_48 = 'Bob'
    var_49 = 25
    var_50 = {var_38: var_48, var_39: var_49}
    var_51 = [var_47, var_50]
    var_52 = var_44.serialize(var_51)
    var_53 = module_0.Array(var_2)
    var_54 = [var_20, var_21, var_22, var_2]
    var_55 = var_53.serialize(var_54)
    var_56 = module_0.Boolean()
    var_57 = module_0.Array(var_56)
    var_58 = True
    var_59 = False
    var_60 = True
    var_61 = [var_58, var_59, var_60]
    var_62 = var_57.serialize(var_61)
    var_63 = 'A'
    var_64 = (var_14, var_63)
    var_65 = 'B'
    var_66 = (var_15, var_65)
    var_67 = [var_64, var_66]
    var_68 = module_0.Choice(choices=var_67)
    var_69 = module_0.Array(var_68)
    var_70 = [var_14, var_15, var_14]
    var_71 = var_69.serialize(var_70)
    var_72 = module_0.String()
    var_73 = module_0.Array(var_72)
    var_74 = []
    var_75 = var_73.serialize(var_74)



# Parsed testcases at query #13
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'hello'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'hello'
    var_3 = True
    var_4 = module_0.String()
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None
    var_7 = False
    var_8 = module_0.String()
    var_9 = None
    var_10 = var_8.validate(var_9)
    var_11 = module_0.String(allow_blank=var_3)
    var_12 = ''
    var_13 = var_11.validate(var_12)
    assert var_13 == ''
    var_14 = module_0.String(allow_blank=var_7)
    var_15 = ''
    var_16 = var_14.validate(var_15)
    var_17 = module_0.String()
    var_18 = 123
    var_19 = var_17.validate(var_18)
    var_20 = module_0.String(trim_whitespace=var_3)
    var_21 = '  hello  '
    var_22 = var_20.validate(var_21)
    assert var_22 == 'hello'
    var_23 = module_0.String(trim_whitespace=var_7)
    var_24 = var_23.validate(var_21)
    assert var_24 == '  hello  '
    var_25 = 3
    var_26 = module_0.String(min_length=var_25)
    var_27 = var_26.validate(var_18)
    assert var_27 == 'hello'
    var_28 = 'hi'
    var_29 = var_26.validate(var_28)
    var_30 = 5
    var_31 = module_0.String(max_length=var_30)
    var_32 = var_31.validate(var_28)
    assert var_32 == 'hello'
    var_33 = 'hello world'
    var_34 = var_31.validate(var_33)
    var_35 = '^\\d+$'
    var_36 = module_0.String(pattern=var_35)
    var_37 = '12345'
    var_38 = var_36.validate(var_37)
    assert var_38 == '12345'
    var_39 = 'hello'
    var_40 = var_36.validate(var_39)
    var_41 = module_0.String()
    var_42 = 'hel\x00lo'
    var_43 = var_41.validate(var_42)
    assert var_43 == 'hello'
    var_44 = module_0.String(allow_blank=var_3, coerce_types=var_3)
    var_45 = var_44.validate(var_5)
    assert var_45 == ''
    var_46 = module_0.String(allow_blank=var_3, coerce_types=var_7)
    var_47 = None
    var_48 = var_46.validate(var_47)
    var_49 = module_0.String(coerce_types=var_3)
    var_50 = var_49.validate(var_12)
    assert var_50 is None
    var_51 = module_0.String(coerce_types=var_7)
    var_52 = var_51.validate(var_12)
    assert var_52 == ''
    var_53 = 'email'
    var_54 = module_0.String(format=var_53)
    var_55 = 'test@example.com'
    var_56 = var_54.validate(var_55)
    assert var_56 == 'test@example.com'
    var_57 = 2
    var_58 = 10
    var_59 = module_0.String(allow_blank=var_7, trim_whitespace=var_3, max_length=var_58, min_length=var_57)
    var_60 = var_59.validate(var_21)
    assert var_60 == 'hello'
    var_61 = 'a'
    var_62 = var_59.validate(var_61)
    var_63 = 'a'
    var_64 = 20
    var_65 = var_63 * var_64
    var_66 = var_59.validate(var_65)



# Parsed testcases at query #14
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'hello'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'hello'
    var_3 = True
    var_4 = module_0.String()
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None
    var_7 = module_0.String()
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = module_0.String(allow_blank=var_3)
    var_11 = ''
    var_12 = var_10.validate(var_11)
    assert var_12 == ''
    var_13 = False
    var_14 = module_0.String(allow_blank=var_13)
    var_15 = ''
    var_16 = var_14.validate(var_15)
    var_17 = module_0.String(trim_whitespace=var_3)
    var_18 = '  hello  '
    var_19 = var_17.validate(var_18)
    assert var_19 == 'hello'
    var_20 = module_0.String(trim_whitespace=var_13)
    var_21 = var_20.validate(var_18)
    assert var_21 == '  hello  '
    var_22 = module_0.String()
    var_23 = 123
    var_24 = var_22.validate(var_23)
    var_25 = 5
    var_26 = module_0.String(max_length=var_25)
    var_27 = var_26.validate(var_23)
    assert var_27 == 'hello'
    var_28 = 'toolong'
    var_29 = var_26.validate(var_28)
    var_30 = 3
    var_31 = module_0.String(min_length=var_30)
    var_32 = var_31.validate(var_28)
    assert var_32 == 'hello'
    var_33 = 'hi'
    var_34 = var_31.validate(var_33)
    var_35 = '^\\d+$'
    var_36 = module_0.String(pattern=var_35)
    var_37 = '123'
    var_38 = var_36.validate(var_37)
    assert var_38 == '123'
    var_39 = 'abc'
    var_40 = var_36.validate(var_39)
    var_41 = module_0.String()
    var_42 = 'hel\x00lo'
    var_43 = var_41.validate(var_42)
    assert var_43 == 'hello'
    var_44 = module_0.String(allow_blank=var_3, coerce_types=var_3)
    var_45 = var_44.validate(var_5)
    assert var_45 == ''
    var_46 = module_0.String(allow_blank=var_3, coerce_types=var_13)
    var_47 = None
    var_48 = var_46.validate(var_47)
    var_49 = module_0.String(coerce_types=var_3)
    var_50 = var_49.validate(var_11)
    assert var_50 is None
    var_51 = module_0.String(coerce_types=var_13)
    var_52 = var_51.validate(var_11)
    assert var_52 == ''
    var_53 = 'email'
    var_54 = module_0.String(format=var_53)
    var_55 = 'test@example.com'
    var_56 = var_54.validate(var_55)
    assert var_56 == 'test@example.com'
    var_57 = 2
    var_58 = 10
    var_59 = module_0.String(allow_blank=var_13, max_length=var_58, min_length=var_57)
    var_60 = var_59.validate(var_47)
    assert var_60 == 'hello'
    var_61 = 'a'
    var_62 = var_59.validate(var_61)



# Parsed testcases at query #15
#--------------------------


import typesystem.fields as module_0

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
    var_8 = exc_info.value.messages()[var_4]
    var_9 = var_8.code
    assert var_9 == 'null'
    var_10 = module_0.Array()
    var_11 = 'not a list'
    var_12 = var_10.validate(var_11)
    var_13 = exc_info.value.messages()[var_4]
    var_14 = var_13.code
    assert var_14 == 'type'
    var_15 = 2
    var_16 = module_0.Array(exact_items=var_15)
    var_17 = 1
    var_18 = [var_17]
    var_19 = var_16.validate(var_18)
    var_20 = exc_info.value.messages()[var_4]
    var_21 = var_20.code
    assert var_21 == 'exact_items'
    var_22 = 1
    var_23 = 2
    var_24 = 3
    var_25 = [var_22, var_23, var_24]
    var_26 = var_16.validate(var_25)
    var_27 = exc_info.value.messages()[var_25]
    var_28 = var_27.code
    assert var_28 == 'exact_items'
    var_29 = module_0.Array(min_items=var_15)
    var_30 = 1
    var_31 = [var_30]
    var_32 = var_29.validate(var_31)
    var_33 = exc_info.value.messages()[var_25]
    var_34 = var_33.code
    assert var_34 == 'min_items'
    var_35 = module_0.Array(min_items=var_30)
    var_36 = []
    var_37 = var_35.validate(var_36)
    var_38 = exc_info.value.messages()[var_25]
    var_39 = var_38.code
    assert var_39 == 'empty'
    var_40 = module_0.Array(max_items=var_15)
    var_41 = 1
    var_42 = 2
    var_43 = 3
    var_44 = [var_41, var_42, var_43]
    var_45 = var_40.validate(var_44)
    var_46 = exc_info.value.messages()[var_44]
    var_47 = var_46.code
    assert var_47 == 'max_items'
    var_48 = module_0.Array(min_items=var_15)
    var_49 = [var_41, var_15]
    var_50 = var_48.validate(var_49)
    var_51 = module_0.Array(max_items=var_15)
    var_52 = [var_41, var_15]
    var_53 = var_51.validate(var_52)
    var_54 = module_0.Integer()
    var_55 = module_0.Array(var_54)
    var_56 = 3
    var_57 = [var_41, var_15, var_56]
    var_58 = var_55.validate(var_57)
    var_59 = module_0.Integer()
    var_60 = module_0.Array(var_59)
    var_61 = 1
    var_62 = 'not an int'
    var_63 = 3
    var_64 = [var_61, var_62, var_63]
    var_65 = var_60.validate(var_64)
    var_66 = module_0.Integer()
    var_67 = module_0.String()
    var_68 = [var_66, var_67]
    var_69 = module_0.Array(var_68)
    var_70 = 'hello'
    var_71 = [var_61, var_70]
    var_72 = var_69.validate(var_71)
    var_73 = module_0.Integer()
    var_74 = module_0.String()
    var_75 = [var_73, var_74]
    var_76 = module_0.Array(var_75)
    var_77 = [var_61, var_70]
    var_78 = var_76.validate(var_77)
    var_79 = module_0.Integer()
    var_80 = module_0.String()
    var_81 = [var_79, var_80]
    var_82 = module_0.Array(var_81, var_64)
    var_83 = 1
    var_84 = 'hello'
    var_85 = 3
    var_86 = [var_83, var_84, var_85]
    var_87 = var_82.validate(var_86)
    var_88 = exc_info.value.messages()[var_86]
    var_89 = var_88.code
    assert var_89 == 'max_items'
    var_90 = module_0.Integer()
    var_91 = module_0.String()
    var_92 = [var_90, var_91]
    var_93 = module_0.Array(var_92, var_83)
    var_94 = [var_83, var_70, var_56]
    var_95 = var_93.validate(var_94)
    var_96 = module_0.Integer()
    var_97 = [var_96]
    var_98 = module_0.String()
    var_99 = module_0.Array(var_97, var_98)
    var_100 = [var_83, var_70]
    var_101 = var_99.validate(var_100)
    var_102 = module_0.Array(unique_items=var_83)
    var_103 = [var_83, var_15, var_56]
    var_104 = var_102.validate(var_103)
    var_105 = module_0.Array(unique_items=var_83)
    var_106 = 1
    var_107 = 2
    var_108 = [var_106, var_107, var_106]
    var_109 = var_105.validate(var_108)
    var_110 = exc_info.value.messages()[var_109]
    var_111 = var_110.code
    assert var_111 == 'unique_items'
    var_112 = module_0.Array()
    var_113 = []
    var_114 = var_112.validate(var_113)
    var_115 = module_0.Integer()
    var_116 = module_0.Array(var_115)
    var_117 = 1
    var_118 = 'invalid'
    var_119 = [var_117, var_118]
    var_120 = var_116.validate(var_119)
    var_121 = module_0.Array(var_118)
    var_122 = [var_117, var_70, var_118]
    var_123 = var_121.validate(var_122)
    var_124 = module_0.Array(unique_items=var_117)
    var_125 = 'a'
    var_126 = 'b'
    var_127 = 'c'
    var_128 = [var_125, var_126, var_127]
    var_129 = var_124.validate(var_128)
    var_130 = module_0.Array(unique_items=var_117)
    var_131 = 'a'
    var_132 = 'b'
    var_133 = [var_131, var_132, var_131]
    var_134 = var_130.validate(var_133)
    var_135 = exc_info.value.messages()[var_134]
    var_136 = var_135.code
    assert var_136 == 'unique_items'



# Parsed testcases at query #16
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Array()
    var_1 = module_0.String()
    var_2 = module_0.Array(var_1)
    var_3 = module_0.Integer()
    var_4 = module_0.Boolean()
    var_5 = [var_3, var_4]
    var_6 = module_0.Array(var_5)
    var_7 = [var_3, var_4]
    var_8 = True
    var_9 = module_0.Array(var_7, var_8)
    var_10 = module_0.String()
    var_11 = [var_3]
    var_12 = module_0.Array(var_11, var_10)
    var_13 = 10
    var_14 = module_0.Array(var_10, min_items=var_8, max_items=var_13)
    var_15 = 5
    var_16 = module_0.Array(var_10, exact_items=var_15)
    var_17 = 2
    var_18 = 8
    var_19 = module_0.Array(var_10, min_items=var_17, max_items=var_18, exact_items=var_15)
    var_20 = module_0.Array(var_10, unique_items=var_8)
    var_21 = (var_3, var_4)
    var_22 = module_0.Array(var_21)
    var_23 = var_22.items
    var_24 = module_0.Array(var_10)
    var_25 = []
    var_26 = module_0.Array(var_10)
    var_27 = [var_3, var_4]
    var_28 = False
    var_29 = module_0.Array(var_27, var_10, var_8, var_13, unique_items=var_8)
    var_30 = [var_3, var_4, var_10]
    var_31 = module_0.Array(var_30)
    var_32 = [var_3, var_4]
    var_33 = module_0.Array(var_32, var_8)



# Parsed testcases at query #17
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = True
    var_5 = module_0.String()
    var_6 = module_0.Integer()
    var_7 = [var_5, var_6]
    var_8 = module_0.Union(var_7)
    var_9 = module_0.String()
    var_10 = module_0.Integer()
    var_11 = [var_9, var_10]
    var_12 = module_0.Union(var_11)
    var_13 = module_0.Float()
    var_14 = [var_13]
    var_15 = module_0.Union(var_14)
    var_16 = var_15.any_of
    var_17 = len(var_16)
    assert var_17 == 1
    var_18 = []
    var_19 = module_0.Union(var_18)
    var_20 = module_0.String()
    var_21 = module_0.Integer()
    var_22 = [var_20, var_21]
    var_23 = False
    var_24 = module_0.Union(var_22)
    var_25 = module_0.String()
    var_26 = module_0.Integer()
    var_27 = [var_25, var_26]
    var_28 = module_0.Union(var_27)
    var_29 = 'key'
    var_30 = module_0.String()
    var_31 = {var_29: var_30}
    var_32 = module_0.Object(properties=var_31)
    var_33 = module_0.Integer()
    var_34 = module_0.Array(var_33)
    var_35 = [var_32, var_34]
    var_36 = module_0.Union(var_35)
    var_37 = module_0.String()
    var_38 = module_0.Integer()
    var_39 = [var_37, var_38]
    var_40 = module_0.Union(var_39)
    var_41 = module_0.Float()
    var_42 = [var_40, var_41]
    var_43 = module_0.Union(var_42)
    var_44 = var_43.any_of
    var_45 = len(var_44)
    assert var_45 == 2



# Parsed testcases at query #18
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Const(var_0)
    var_2 = None
    var_3 = module_0.Const(var_2)
    var_4 = 'test'
    var_5 = module_0.Const(var_4)
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]
    var_10 = module_0.Const(var_9)
    var_11 = 'key'
    var_12 = 'value'
    var_13 = {var_11: var_12}
    var_14 = module_0.Const(var_13)
    var_15 = True
    var_16 = module_0.Const(var_15)
    var_17 = 42
    var_18 = True
    var_19 = module_0.Const(var_17)
    var_20 = None
    var_21 = False
    var_22 = module_0.Const(var_20)
    var_23 = 100
    var_24 = 50
    var_25 = module_0.Const(var_23)
    var_26 = 5
    var_27 = 'const'
    var_28 = 'Custom error'
    var_29 = {var_27: var_28}
    var_30 = module_0.Const(var_26)



# Parsed testcases at query #19
#--------------------------


import typesystem.fields as module_0

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
    var_10 = module_0.Boolean()
    var_11 = var_10.validate(var_4)
    assert var_11 is False
    var_12 = module_0.Boolean(coerce_types=var_6)
    var_13 = 'true'
    var_14 = var_12.validate(var_13)
    assert var_14 is True
    var_15 = module_0.Boolean(coerce_types=var_6)
    var_16 = 'false'
    var_17 = var_15.validate(var_16)
    assert var_17 is False
    var_18 = module_0.Boolean(coerce_types=var_6)
    var_19 = 'on'
    var_20 = var_18.validate(var_19)
    assert var_20 is True
    var_21 = module_0.Boolean(coerce_types=var_6)
    var_22 = 'off'
    var_23 = var_21.validate(var_22)
    assert var_23 is False
    var_24 = module_0.Boolean(coerce_types=var_6)
    var_25 = '1'
    var_26 = var_24.validate(var_25)
    assert var_26 is True
    var_27 = module_0.Boolean(coerce_types=var_6)
    var_28 = '0'
    var_29 = var_27.validate(var_28)
    assert var_29 is False
    var_30 = module_0.Boolean(coerce_types=var_6)
    var_31 = ''
    var_32 = var_30.validate(var_31)
    assert var_32 is False
    var_33 = module_0.Boolean(coerce_types=var_6)
    var_34 = var_33.validate(var_6)
    assert var_34 is True
    var_35 = module_0.Boolean(coerce_types=var_6)
    var_36 = var_35.validate(var_4)
    assert var_36 is False
    var_37 = module_0.Boolean(coerce_types=var_6)
    var_38 = 'TRUE'
    var_39 = var_37.validate(var_38)
    assert var_39 is True
    var_40 = module_0.Boolean(coerce_types=var_6)
    var_41 = 'FALSE'
    var_42 = var_40.validate(var_41)
    assert var_42 is False
    var_43 = module_0.Boolean(coerce_types=var_6)
    var_44 = var_43.validate(var_31)
    assert var_44 is False
    var_45 = module_0.Boolean(coerce_types=var_6)
    var_46 = 'null'
    var_47 = var_45.validate(var_46)
    assert var_47 is None
    var_48 = module_0.Boolean(coerce_types=var_6)
    var_49 = 'none'
    var_50 = var_48.validate(var_49)
    assert var_50 is None
    var_51 = module_0.Boolean(coerce_types=var_4)
    var_52 = 'true'
    var_53 = var_51.validate(var_52)
    var_54 = module_0.Boolean(coerce_types=var_52)
    var_55 = 'invalid'
    var_56 = var_54.validate(var_55)
    var_57 = module_0.Boolean(coerce_types=var_55)
    var_58 = []
    var_59 = var_57.validate(var_58)



# Parsed testcases at query #20
#--------------------------


import typesystem.fields as module_0

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
    var_8 = exc_info.value.messages()[var_4]
    var_9 = var_8.code
    assert var_9 == 'null'
    var_10 = module_0.Array()
    var_11 = 'not a list'
    var_12 = var_10.validate(var_11)
    var_13 = exc_info.value.messages()[var_4]
    var_14 = var_13.code
    assert var_14 == 'type'
    var_15 = {}
    var_16 = var_10.validate(var_15)
    var_17 = exc_info.value.messages()[var_4]
    var_18 = var_17.code
    assert var_18 == 'type'
    var_19 = module_0.Array(min_items=var_15)
    var_20 = []
    var_21 = var_19.validate(var_20)
    var_22 = exc_info.value.messages()[var_4]
    var_23 = var_22.code
    assert var_23 == 'empty'
    var_24 = 3
    var_25 = module_0.Array(min_items=var_24)
    var_26 = 1
    var_27 = 2
    var_28 = [var_26, var_27]
    var_29 = var_25.validate(var_28)
    var_30 = exc_info.value.messages()[var_29]
    var_31 = var_30.code
    assert var_31 == 'min_items'
    var_32 = 2
    var_33 = module_0.Array(max_items=var_32)
    var_34 = 1
    var_35 = 2
    var_36 = 3
    var_37 = [var_34, var_35, var_36]
    var_38 = var_33.validate(var_37)
    var_39 = exc_info.value.messages()[var_37]
    var_40 = var_39.code
    assert var_40 == 'max_items'
    var_41 = module_0.Array(exact_items=var_32)
    var_42 = 1
    var_43 = 2
    var_44 = 3
    var_45 = [var_42, var_43, var_44]
    var_46 = var_41.validate(var_45)
    var_47 = exc_info.value.messages()[var_45]
    var_48 = var_47.code
    assert var_48 == 'exact_items'
    var_49 = module_0.Array(exact_items=var_32)
    var_50 = [var_42, var_32]
    var_51 = var_49.validate(var_50)
    var_52 = module_0.Array()
    var_53 = [var_42, var_32, var_24]
    var_54 = var_52.validate(var_53)
    var_55 = module_0.Integer()
    var_56 = module_0.Array(var_55)
    var_57 = [var_42, var_32, var_24]
    var_58 = var_56.validate(var_57)
    var_59 = module_0.Integer()
    var_60 = module_0.Array(var_59)
    var_61 = 1
    var_62 = 'not an integer'
    var_63 = 3
    var_64 = [var_61, var_62, var_63]
    var_65 = var_60.validate(var_64)
    var_66 = module_0.Integer()
    var_67 = module_0.String()
    var_68 = [var_66, var_67]
    var_69 = module_0.Array(var_68)
    var_70 = 'hello'
    var_71 = [var_61, var_70]
    var_72 = var_69.validate(var_71)
    var_73 = module_0.Integer()
    var_74 = module_0.String()
    var_75 = [var_73, var_74]
    var_76 = module_0.Array(var_75, var_64)
    var_77 = 1
    var_78 = 'hello'
    var_79 = 3
    var_80 = [var_77, var_78, var_79]
    var_81 = var_76.validate(var_80)
    var_82 = module_0.Integer()
    var_83 = module_0.String()
    var_84 = [var_82, var_83]
    var_85 = module_0.Boolean()
    var_86 = module_0.Array(var_84, var_85)
    var_87 = [var_77, var_70, var_77]
    var_88 = var_86.validate(var_87)
    var_89 = module_0.Array(unique_items=var_77)
    var_90 = 1
    var_91 = 2
    var_92 = [var_90, var_91, var_90]
    var_93 = var_89.validate(var_92)
    var_94 = 'unique_items'
    var_95 = module_0.Array(unique_items=var_90)
    var_96 = [var_90, var_32, var_24]
    var_97 = var_95.validate(var_96)
    var_98 = module_0.Array(unique_items=var_90)
    var_99 = 'a'
    var_100 = 'b'
    var_101 = [var_99, var_100, var_99]
    var_102 = var_98.validate(var_101)
    var_103 = 4
    var_104 = module_0.Array(min_items=var_32, max_items=var_103)
    var_105 = [var_99, var_32, var_24]
    var_106 = var_104.validate(var_105)
    var_107 = module_0.Integer()
    var_108 = module_0.Array(var_107)
    var_109 = module_0.Array(var_108)
    var_110 = [var_99, var_32]
    var_111 = [var_24, var_103]
    var_112 = [var_110, var_111]
    var_113 = var_109.validate(var_112)
    var_114 = module_0.Integer()
    var_115 = module_0.Array(var_114)
    var_116 = module_0.Array(var_115)
    var_117 = 1
    var_118 = 'invalid'
    var_119 = [var_117, var_118]
    var_120 = 3
    var_121 = 4
    var_122 = [var_120, var_121]
    var_123 = [var_119, var_122]
    var_124 = var_116.validate(var_123)
    var_125 = 'name'
    var_126 = module_0.String()
    var_127 = {var_125: var_126}
    var_128 = module_0.Object(properties=var_127)
    var_129 = module_0.Array(var_128)
    var_130 = 'test'
    var_131 = {var_125: var_130}
    var_132 = [var_131]
    var_133 = var_129.validate(var_132)
    var_134 = module_0.Array()
    var_135 = []
    var_136 = var_134.validate(var_135)
    var_137 = module_0.Integer()
    var_138 = [var_137]
    var_139 = module_0.Array(var_138, var_117)
    var_140 = [var_117, var_32, var_24]
    var_141 = var_139.validate(var_140)



# Parsed testcases at query #21
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'hello'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'hello'
    var_3 = True
    var_4 = module_0.String()
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None
    var_7 = module_0.String()
    var_8 = None
    var_9 = var_7.validate(var_8)
    var_10 = module_0.String(allow_blank=var_3)
    var_11 = ''
    var_12 = var_10.validate(var_11)
    assert var_12 == ''
    var_13 = False
    var_14 = module_0.String(allow_blank=var_13)
    var_15 = ''
    var_16 = var_14.validate(var_15)
    var_17 = module_0.String()
    var_18 = 123
    var_19 = var_17.validate(var_18)
    var_20 = module_0.String(trim_whitespace=var_3)
    var_21 = '  hello  '
    var_22 = var_20.validate(var_21)
    assert var_22 == 'hello'
    var_23 = module_0.String(trim_whitespace=var_13)
    var_24 = var_23.validate(var_21)
    assert var_24 == '  hello  '
    var_25 = 5
    var_26 = module_0.String(max_length=var_25)
    var_27 = var_26.validate(var_18)
    assert var_27 == 'hello'
    var_28 = 'toolong'
    var_29 = var_26.validate(var_28)
    var_30 = 3
    var_31 = module_0.String(min_length=var_30)
    var_32 = var_31.validate(var_28)
    assert var_32 == 'hello'
    var_33 = 'hi'
    var_34 = var_31.validate(var_33)
    var_35 = '^\\d+$'
    var_36 = module_0.String(pattern=var_35)
    var_37 = '123'
    var_38 = var_36.validate(var_37)
    assert var_38 == '123'
    var_39 = 'abc'
    var_40 = var_36.validate(var_39)
    var_41 = module_0.String()
    var_42 = 'hello\x00world'
    var_43 = var_41.validate(var_42)
    assert var_43 == 'helloworld'
    var_44 = module_0.String(allow_blank=var_3, coerce_types=var_3)
    var_45 = var_44.validate(var_5)
    assert var_45 == ''
    var_46 = module_0.String(coerce_types=var_3)
    var_47 = var_46.validate(var_11)
    assert var_47 is None
    var_48 = 'email'
    var_49 = module_0.String(format=var_48)
    var_50 = 'test@example.com'
    var_51 = var_49.validate(var_50)



# Parsed testcases at query #22
#--------------------------


import typesystem.fields as module_0
import collections as module_1

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
    var_12 = {}
    var_13 = var_11.validate(var_12)
    var_14 = module_0.Object()
    var_15 = 1
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = var_14.validate(var_17)
    var_19 = 'invalid_key'
    var_20 = 'name'
    var_21 = module_0.String()
    var_22 = {var_20: var_21}
    var_23 = module_0.Object(properties=var_22)
    var_24 = 'John'
    var_25 = {var_20: var_24}
    var_26 = var_23.validate(var_25)
    var_27 = 'Unknown'
    var_28 = module_0.String()
    var_29 = {var_20: var_28}
    var_30 = module_0.Object(properties=var_29)
    var_31 = {}
    var_32 = var_30.validate(var_31)
    var_33 = [var_20]
    var_34 = module_0.Object(required=var_33)
    var_35 = {}
    var_36 = var_34.validate(var_35)
    var_37 = 'required'
    var_38 = [var_20]
    var_39 = module_0.String()
    var_40 = {var_20: var_39}
    var_41 = module_0.Object(properties=var_40, required=var_38)
    var_42 = {var_20: var_24}
    var_43 = var_41.validate(var_42)
    var_44 = 2
    var_45 = module_0.Object(min_properties=var_44)
    var_46 = 'key'
    var_47 = 'value'
    var_48 = {var_46: var_47}
    var_49 = var_45.validate(var_48)
    var_50 = module_0.Object(min_properties=var_46)
    var_51 = {}
    var_52 = var_50.validate(var_51)
    var_53 = module_0.Object(max_properties=var_51)
    var_54 = 'key1'
    var_55 = 'key2'
    var_56 = 'value1'
    var_57 = 'value2'
    var_58 = {var_54: var_56, var_55: var_57}
    var_59 = var_53.validate(var_58)
    var_60 = module_0.Object(additional_properties=var_54)
    var_61 = 'extra'
    var_62 = 'value'
    var_63 = {var_61: var_62}
    var_64 = var_60.validate(var_63)
    var_65 = module_0.Object(additional_properties=var_57)
    var_66 = 'extra'
    var_67 = 'value'
    var_68 = {var_66: var_67}
    var_69 = var_65.validate(var_68)
    var_70 = 'invalid_property'
    var_71 = module_0.Integer()
    var_72 = module_0.Object(additional_properties=var_71)
    var_73 = 42
    var_74 = {var_61: var_73}
    var_75 = var_72.validate(var_74)
    var_76 = '^num'
    var_77 = module_0.Integer()
    var_78 = {var_76: var_77}
    var_79 = module_0.Object(pattern_properties=var_78)
    var_80 = 'num_field'
    var_81 = 123
    var_82 = {var_80: var_81}
    var_83 = var_79.validate(var_82)
    var_84 = module_0.Integer()
    var_85 = {var_76: var_84}
    var_86 = module_0.Object(pattern_properties=var_85)
    var_87 = 'num_field'
    var_88 = 'not_an_int'
    var_89 = {var_87: var_88}
    var_90 = var_86.validate(var_89)
    var_91 = 'type'
    var_92 = 3
    var_93 = module_0.String(min_length=var_92)
    var_94 = module_0.Object(property_names=var_93)
    var_95 = 'ab'
    var_96 = 'value'
    var_97 = {var_95: var_96}
    var_98 = var_94.validate(var_97)
    var_99 = 'nested'
    var_100 = 'id'
    var_101 = module_0.Integer()
    var_102 = {var_100: var_101}
    var_103 = module_0.Object(properties=var_102)
    var_104 = {var_99: var_103}
    var_105 = module_0.Object(properties=var_104)
    var_106 = {var_100: var_73}
    var_107 = {var_99: var_106}
    var_108 = var_105.validate(var_107)
    var_109 = module_0.Integer()
    var_110 = {var_100: var_109}
    var_111 = module_0.Object(properties=var_110)
    var_112 = {var_99: var_111}
    var_113 = module_0.Object(properties=var_112)
    var_114 = 'nested'
    var_115 = 'id'
    var_116 = 'invalid'
    var_117 = {var_115: var_116}
    var_118 = {var_114: var_117}
    var_119 = var_113.validate(var_118)
    var_120 = module_0.String()
    var_121 = {var_20: var_120}
    var_122 = module_0.Integer()
    var_123 = module_0.Object(properties=var_121, additional_properties=var_122)
    var_124 = 'age'
    var_125 = 30
    var_126 = {var_20: var_24, var_124: var_125}
    var_127 = var_123.validate(var_126)
    var_128 = module_0.Object()
    var_129 = 'key'
    var_130 = {var_129: var_62}
    var_131 = module_1.UserDict(var_130)
    var_132 = var_128.validate(var_131)



# Parsed testcases at query #23
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String()
    var_2 = module_0.Integer()
    var_3 = [var_1, var_2]
    var_4 = module_0.Union(var_3)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None
    var_7 = module_0.String()
    var_8 = module_0.Integer()
    var_9 = [var_7, var_8]
    var_10 = module_0.Union(var_9)
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = 0
    var_14 = exc_info.value.messages()[var_13]
    var_15 = var_14.code
    assert var_15 == 'null'
    var_16 = module_0.String()
    var_17 = module_0.Integer()
    var_18 = [var_16, var_17]
    var_19 = module_0.Union(var_18)
    var_20 = 'hello'
    var_21 = var_19.validate(var_20)
    assert var_21 == 'hello'
    var_22 = module_0.String()
    var_23 = module_0.Integer()
    var_24 = [var_22, var_23]
    var_25 = module_0.Union(var_24)
    var_26 = 42
    var_27 = var_25.validate(var_26)
    assert var_27 == 42
    var_28 = module_0.String()
    var_29 = module_0.Integer()
    var_30 = [var_28, var_29]
    var_31 = module_0.Union(var_30)
    var_32 = 3.14
    var_33 = var_31.validate(var_32)
    assert var_33 == 3
    var_34 = module_0.String()
    var_35 = module_0.Integer()
    var_36 = [var_34, var_35]
    var_37 = module_0.Union(var_36)
    var_38 = 1
    var_39 = 2
    var_40 = 3
    var_41 = [var_38, var_39, var_40]
    var_42 = var_37.validate(var_41)
    var_43 = exc_info.value.messages()[var_13]
    var_44 = var_43.code
    assert var_44 == 'union'
    var_45 = 2
    var_46 = module_0.String(max_length=var_45)
    var_47 = module_0.Integer()
    var_48 = [var_46, var_47]
    var_49 = module_0.Union(var_48)
    var_50 = 'toolong'
    var_51 = var_49.validate(var_50)
    var_52 = exc_info.value.messages()[var_13]
    var_53 = var_52.code
    assert var_53 == 'max_length'
    var_54 = module_0.String()
    var_55 = False
    var_56 = module_0.Integer()
    var_57 = [var_54, var_56]
    var_58 = module_0.Union(var_57)
    var_59 = var_58.validate(var_42)
    assert var_59 is None
    var_60 = module_0.String()
    var_61 = module_0.Integer()
    var_62 = [var_60, var_61]
    var_63 = module_0.Union(var_62)
    var_64 = var_63.validate(var_42)
    assert var_64 is None
    var_65 = module_0.Boolean()
    var_66 = module_0.Integer()
    var_67 = [var_65, var_66]
    var_68 = module_0.Union(var_67)
    var_69 = var_68.validate(var_50)
    assert var_69 is True
    var_70 = False
    var_71 = var_68.validate(var_70)
    assert var_71 is False
    var_72 = module_0.String()
    var_73 = module_0.Array(var_72)
    var_74 = 'name'
    var_75 = module_0.String()
    var_76 = {var_74: var_75}
    var_77 = module_0.Object(properties=var_76)
    var_78 = [var_73, var_77]
    var_79 = module_0.Union(var_78)
    var_80 = 'a'
    var_81 = 'b'
    var_82 = [var_80, var_81]
    var_83 = var_79.validate(var_82)
    var_84 = 'John'
    var_85 = {var_74: var_84}
    var_86 = var_79.validate(var_85)
    var_87 = 5
    var_88 = module_0.String(min_length=var_87)
    var_89 = 10
    var_90 = module_0.Integer(minimum=var_89)
    var_91 = [var_88, var_90]
    var_92 = module_0.Union(var_91)
    var_93 = 2
    var_94 = var_92.validate(var_93)
    var_95 = exc_info.value.messages()[var_70]
    var_96 = var_95.code
    assert var_96 == 'union'
    var_97 = module_0.String(max_length=var_45)
    var_98 = module_0.Integer(maximum=var_87)
    var_99 = [var_97, var_98]
    var_100 = module_0.Union(var_99)
    var_101 = 'toolong'
    var_102 = var_100.validate(var_101)
    var_103 = exc_info.value.messages()[var_70]
    var_104 = var_103.code
    assert var_104 == 'max_length'
    var_105 = module_0.String()
    var_106 = module_0.Integer(coerce_types=var_101)
    var_107 = [var_105, var_106]
    var_108 = module_0.Union(var_107)
    var_109 = '123'
    var_110 = var_108.validate(var_109)
    assert var_110 == '123'
    var_111 = module_0.Boolean(coerce_types=var_101)
    var_112 = module_0.String()
    var_113 = [var_111, var_112]
    var_114 = module_0.Union(var_113)
    var_115 = 'true'
    var_116 = var_114.validate(var_115)
    assert var_116 is True
    var_117 = var_114.validate(var_20)
    assert var_117 == 'hello'
    var_118 = []
    var_119 = module_0.Union(var_118)
    var_120 = 'value'
    var_121 = var_119.validate(var_120)
    var_122 = exc_info.value.messages()[var_70]
    var_123 = var_122.code
    assert var_123 == 'union'
    var_124 = module_0.Float()
    var_125 = module_0.Integer()
    var_126 = [var_124, var_125]
    var_127 = module_0.Union(var_126)
    var_128 = var_127.validate(var_32)
    var_129 = var_127.validate(var_26)
    assert var_129 == 42



# Parsed testcases at query #24
#--------------------------


import typesystem.fields as module_0

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
    var_12 = {}
    var_13 = var_11.validate(var_12)
    var_14 = module_0.Object()
    var_15 = 1
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = var_14.validate(var_17)
    var_19 = 'invalid_key'
    var_20 = 'name'
    var_21 = [var_20]
    var_22 = module_0.Object(required=var_21)
    var_23 = {}
    var_24 = var_22.validate(var_23)
    var_25 = 'required'
    var_26 = [var_20]
    var_27 = module_0.Object(required=var_26)
    var_28 = 'John'
    var_29 = {var_20: var_28}
    var_30 = var_27.validate(var_29)
    var_31 = 'age'
    var_32 = module_0.String()
    var_33 = module_0.Integer()
    var_34 = {var_20: var_32, var_31: var_33}
    var_35 = module_0.Object(properties=var_34)
    var_36 = 30
    var_37 = {var_20: var_28, var_31: var_36}
    var_38 = var_35.validate(var_37)
    var_39 = module_0.Integer()
    var_40 = {var_31: var_39}
    var_41 = module_0.Object(properties=var_40)
    var_42 = 'age'
    var_43 = 'not an integer'
    var_44 = {var_42: var_43}
    var_45 = var_41.validate(var_44)
    var_46 = 'type'
    var_47 = 'Unknown'
    var_48 = module_0.String()
    var_49 = {var_20: var_48}
    var_50 = module_0.Object(properties=var_49)
    var_51 = {}
    var_52 = var_50.validate(var_51)
    var_53 = module_0.Object(additional_properties=var_42)
    var_54 = 'extra'
    var_55 = 'value'
    var_56 = {var_54: var_55}
    var_57 = var_53.validate(var_56)
    var_58 = module_0.Object(additional_properties=var_45)
    var_59 = 'extra'
    var_60 = 'value'
    var_61 = {var_59: var_60}
    var_62 = var_58.validate(var_61)
    var_63 = 'invalid_property'
    var_64 = module_0.Integer()
    var_65 = module_0.Object(additional_properties=var_64)
    var_66 = 42
    var_67 = {var_54: var_66}
    var_68 = var_65.validate(var_67)
    var_69 = '^S_'
    var_70 = '^I_'
    var_71 = module_0.String()
    var_72 = module_0.Integer()
    var_73 = {var_69: var_71, var_70: var_72}
    var_74 = module_0.Object(pattern_properties=var_73)
    var_75 = 'S_name'
    var_76 = 'I_age'
    var_77 = {var_75: var_28, var_76: var_36}
    var_78 = var_74.validate(var_77)
    var_79 = module_0.Object(min_properties=var_59)
    var_80 = {}
    var_81 = var_79.validate(var_80)
    var_82 = 2
    var_83 = module_0.Object(min_properties=var_82)
    var_84 = 'a'
    var_85 = 1
    var_86 = {var_84: var_85}
    var_87 = var_83.validate(var_86)
    var_88 = module_0.Object(max_properties=var_84)
    var_89 = 'a'
    var_90 = 'b'
    var_91 = 1
    var_92 = 2
    var_93 = {var_89: var_91, var_90: var_92}
    var_94 = var_88.validate(var_93)
    var_95 = '^[a-z]+$'
    var_96 = module_0.String(pattern=var_95)
    var_97 = module_0.Object(property_names=var_96)
    var_98 = '123'
    var_99 = 'value'
    var_100 = {var_98: var_99}
    var_101 = var_97.validate(var_100)
    var_102 = 'user'
    var_103 = module_0.String()
    var_104 = module_0.Integer()
    var_105 = {var_20: var_103, var_31: var_104}
    var_106 = [var_20]
    var_107 = module_0.Object(properties=var_105, required=var_106)
    var_108 = {var_102: var_107}
    var_109 = [var_102]
    var_110 = module_0.Object(properties=var_108, required=var_109)
    var_111 = {var_20: var_28, var_31: var_36}
    var_112 = {var_102: var_111}
    var_113 = var_110.validate(var_112)
    var_114 = module_0.String()
    var_115 = module_0.Integer()
    var_116 = {var_20: var_114, var_31: var_115}
    var_117 = [var_20]
    var_118 = module_0.Object(properties=var_116, required=var_117)
    var_119 = {var_102: var_118}
    var_120 = [var_102]
    var_121 = module_0.Object(properties=var_119, required=var_120)
    var_122 = 'user'
    var_123 = 'age'
    var_124 = 30
    var_125 = {var_123: var_124}
    var_126 = {var_122: var_125}
    var_127 = var_121.validate(var_126)



# Parsed testcases at query #25
#--------------------------


import typesystem.fields as module_0

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
    var_11 = module_0.Number()
    var_12 = True
    var_13 = var_11.validate(var_12)
    var_14 = 3.14
    var_15 = var_11.validate(var_14)
    var_16 = 3.0
    var_17 = var_11.validate(var_16)
    assert var_17 == 3
    var_18 = module_0.Number(coerce_types=var_4)
    var_19 = '123'
    var_20 = var_18.validate(var_19)
    var_21 = module_0.Number(coerce_types=var_19)
    var_22 = '123'
    var_23 = var_21.validate(var_22)
    assert var_23 == 123
    var_24 = module_0.Number()
    var_25 = 'not_a_number'
    var_26 = var_24.validate(var_25)
    var_27 = module_0.Number()
    var_28 = 'inf'
    var_29 = float(var_28)
    var_30 = var_27.validate(var_29)
    var_31 = module_0.Number()
    var_32 = 'nan'
    var_33 = float(var_32)
    var_34 = var_31.validate(var_33)
    var_35 = 10
    var_36 = module_0.Number(minimum=var_35)
    var_37 = 5
    var_38 = var_36.validate(var_37)
    var_39 = var_36.validate(var_35)
    assert var_39 == 10
    var_40 = 15
    var_41 = var_36.validate(var_40)
    assert var_41 == 15
    var_42 = module_0.Number(exclusive_minimum=var_35)
    var_43 = 10
    var_44 = var_42.validate(var_43)
    var_45 = 11
    var_46 = var_42.validate(var_45)
    assert var_46 == 11
    var_47 = 100
    var_48 = module_0.Number(maximum=var_47)
    var_49 = 150
    var_50 = var_48.validate(var_49)
    var_51 = var_48.validate(var_47)
    assert var_51 == 100
    var_52 = 50
    var_53 = var_48.validate(var_52)
    assert var_53 == 50
    var_54 = module_0.Number(exclusive_maximum=var_47)
    var_55 = 100
    var_56 = var_54.validate(var_55)
    var_57 = 99
    var_58 = var_54.validate(var_57)
    assert var_58 == 99
    var_59 = '0.01'
    var_60 = 3.14159
    var_61 = var_54.validate(var_60)
    var_62 = 3.14
    var_63 = var_61 - var_62
    var_64 = abs(var_63)
    var_65 = 5
    var_66 = module_0.Number(multiple_of=var_65)
    var_67 = var_66.validate(var_35)
    assert var_67 == 10
    var_68 = 12
    var_69 = var_66.validate(var_68)
    var_70 = 0.5
    var_71 = module_0.Number(multiple_of=var_70)
    var_72 = 2.5
    var_73 = var_71.validate(var_72)
    var_74 = 2.3
    var_75 = var_71.validate(var_74)
    var_76 = 42
    var_77 = var_71.validate(var_76)
    assert var_77 == 42
    var_78 = '42'
    var_79 = var_71.validate(var_78)
    assert var_79 == 42
    var_80 = var_71.validate(var_78)
    var_81 = var_71.validate(var_62)
    var_82 = '3.14'
    var_83 = var_71.validate(var_82)
    var_84 = var_71.validate(var_82)
    var_85 = module_0.Number(minimum=var_4, maximum=var_47, multiple_of=var_35)
    var_86 = var_85.validate(var_52)
    assert var_86 == 50
    var_87 = var_85.validate(var_4)
    assert var_87 == 0
    var_88 = var_85.validate(var_47)
    assert var_88 == 100
    var_89 = 55
    var_90 = var_85.validate(var_89)



# Parsed testcases at query #26
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Object(properties=var_4)
    var_6 = 'John'
    var_7 = 30
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = module_0.Object()
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = True
    var_14 = module_0.Object()
    var_15 = None
    var_16 = var_14.validate(var_15)
    assert var_16 is None
    var_17 = module_0.Object()
    var_18 = 'not a dict'
    var_19 = var_17.validate(var_18)
    var_20 = module_0.Object()
    var_21 = 1
    var_22 = 'value'
    var_23 = {var_21: var_22}
    var_24 = var_20.validate(var_23)
    var_25 = 'invalid_key'
    var_26 = module_0.String()
    var_27 = {var_21: var_26}
    var_28 = [var_21]
    var_29 = module_0.Object(properties=var_27, required=var_28)
    var_30 = {}
    var_31 = var_29.validate(var_30)
    var_32 = 'required'
    var_33 = 2
    var_34 = module_0.Object(min_properties=var_33)
    var_35 = 'a'
    var_36 = 1
    var_37 = {var_35: var_36}
    var_38 = var_34.validate(var_37)
    var_39 = module_0.Object(min_properties=var_13)
    var_40 = {}
    var_41 = var_39.validate(var_40)
    var_42 = module_0.Object(max_properties=var_13)
    var_43 = 'a'
    var_44 = 'b'
    var_45 = 1
    var_46 = 2
    var_47 = {var_43: var_45, var_44: var_46}
    var_48 = var_42.validate(var_47)
    var_49 = 'Unknown'
    var_50 = module_0.String()
    var_51 = {var_43: var_50}
    var_52 = module_0.Object(properties=var_51)
    var_53 = {}
    var_54 = var_52.validate(var_53)
    var_55 = module_0.String()
    var_56 = {var_43: var_55}
    var_57 = module_0.Object(properties=var_56, additional_properties=var_13)
    var_58 = 'extra'
    var_59 = 'value'
    var_60 = {var_43: var_48, var_58: var_59}
    var_61 = var_57.validate(var_60)
    var_62 = module_0.String()
    var_63 = {var_43: var_62}
    var_64 = False
    var_65 = module_0.Object(properties=var_63, additional_properties=var_64)
    var_66 = 'name'
    var_67 = 'extra'
    var_68 = 'John'
    var_69 = 'value'
    var_70 = {var_66: var_68, var_67: var_69}
    var_71 = var_65.validate(var_70)
    var_72 = 'invalid_property'
    var_73 = module_0.String()
    var_74 = {var_66: var_73}
    var_75 = module_0.Integer()
    var_76 = module_0.Object(properties=var_74, additional_properties=var_75)
    var_77 = {var_66: var_71, var_67: var_7}
    var_78 = var_76.validate(var_77)
    var_79 = '^S_'
    var_80 = '^I_'
    var_81 = module_0.String()
    var_82 = module_0.Integer()
    var_83 = {var_79: var_81, var_80: var_82}
    var_84 = module_0.Object(pattern_properties=var_83)
    var_85 = 'S_name'
    var_86 = 'I_age'
    var_87 = {var_85: var_71, var_86: var_7}
    var_88 = var_84.validate(var_87)
    var_89 = '^[a-z]+$'
    var_90 = module_0.String(pattern=var_89)
    var_91 = module_0.Object(property_names=var_90)
    var_92 = 'Invalid'
    var_93 = 'value'
    var_94 = {var_92: var_93}
    var_95 = var_91.validate(var_94)
    var_96 = 5
    var_97 = module_0.String(min_length=var_96)
    var_98 = {var_92: var_97}
    var_99 = module_0.Object(properties=var_98)
    var_100 = 'user'
    var_101 = {var_100: var_99}
    var_102 = module_0.Object(properties=var_101)
    var_103 = 'user'
    var_104 = 'name'
    var_105 = 'ab'
    var_106 = {var_104: var_105}
    var_107 = {var_103: var_106}
    var_108 = var_102.validate(var_107)
    var_109 = module_0.Object()
    var_110 = {}
    var_111 = var_109.validate(var_110)
    var_112 = module_0.Object()
    var_113 = 'a'
    var_114 = (var_113, var_13)
    var_115 = 'b'
    var_116 = (var_115, var_33)
    var_117 = [var_114, var_116]



# Parsed testcases at query #27
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = True
    var_5 = module_0.String()
    var_6 = module_0.Integer()
    var_7 = [var_5, var_6]
    var_8 = module_0.Union(var_7)
    var_9 = var_8.any_of
    var_10 = len(var_9)
    assert var_10 == 2
    var_11 = module_0.String()
    var_12 = module_0.Integer()
    var_13 = [var_11, var_12]
    var_14 = module_0.Union(var_13)
    var_15 = []
    var_16 = module_0.Union(var_15)
    var_17 = module_0.Boolean()
    var_18 = [var_17]
    var_19 = module_0.Union(var_18)
    var_20 = var_19.any_of
    var_21 = len(var_20)
    assert var_21 == 1
    var_22 = module_0.String()
    var_23 = [var_22]
    var_24 = False
    var_25 = module_0.Union(var_23)
    var_26 = module_0.String()
    var_27 = module_0.Integer()
    var_28 = [var_27, var_26]
    var_29 = module_0.Union(var_28)
    var_30 = module_0.String()
    var_31 = module_0.Integer()
    var_32 = module_0.Boolean()
    var_33 = module_0.Float()
    var_34 = [var_30, var_31, var_32, var_33]
    var_35 = module_0.Union(var_34)
    var_36 = var_35.any_of
    var_37 = len(var_36)
    assert var_37 == 4



# Parsed testcases at query #28
#--------------------------


import typesystem.fields as module_0

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
    var_10 = module_0.Boolean()
    var_11 = var_10.validate(var_4)
    assert var_11 is False
    var_12 = module_0.Boolean(coerce_types=var_6)
    var_13 = 'true'
    var_14 = var_12.validate(var_13)
    assert var_14 is True
    var_15 = 'True'
    var_16 = var_12.validate(var_15)
    assert var_16 is True
    var_17 = 'false'
    var_18 = var_12.validate(var_17)
    assert var_18 is False
    var_19 = 'False'
    var_20 = var_12.validate(var_19)
    assert var_20 is False
    var_21 = 'on'
    var_22 = var_12.validate(var_21)
    assert var_22 is True
    var_23 = 'off'
    var_24 = var_12.validate(var_23)
    assert var_24 is False
    var_25 = '1'
    var_26 = var_12.validate(var_25)
    assert var_26 is True
    var_27 = '0'
    var_28 = var_12.validate(var_27)
    assert var_28 is False
    var_29 = module_0.Boolean(coerce_types=var_6)
    var_30 = ''
    var_31 = var_29.validate(var_30)
    assert var_31 is False
    var_32 = module_0.Boolean(coerce_types=var_6)
    var_33 = var_32.validate(var_6)
    assert var_33 is True
    var_34 = var_32.validate(var_4)
    assert var_34 is False
    var_35 = module_0.Boolean(coerce_types=var_6)
    var_36 = var_35.validate(var_30)
    assert var_36 is False
    var_37 = 'null'
    var_38 = var_35.validate(var_37)
    assert var_38 is None
    var_39 = 'none'
    var_40 = var_35.validate(var_39)
    assert var_40 is None
    var_41 = module_0.Boolean(coerce_types=var_4)
    var_42 = 'true'
    var_43 = var_41.validate(var_42)
    var_44 = module_0.Boolean(coerce_types=var_4)
    var_45 = var_44.validate(var_42)
    assert var_45 is True
    var_46 = module_0.Boolean(coerce_types=var_42)
    var_47 = 'invalid'
    var_48 = var_46.validate(var_47)
    var_49 = module_0.Boolean(coerce_types=var_47)
    var_50 = []
    var_51 = var_49.validate(var_50)



# Parsed testcases at query #29
#--------------------------


import typesystem.fields as module_0

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
    var_11 = 'name'
    var_12 = 'age'
    var_13 = module_0.String()
    var_14 = module_0.Integer()
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = module_0.Object(properties=var_15)
    var_17 = 'John'
    var_18 = 30
    var_19 = {var_11: var_17, var_12: var_18}
    var_20 = var_16.validate(var_19)
    var_21 = module_0.Object()
    var_22 = 123
    var_23 = 'value'
    var_24 = {var_22: var_23}
    var_25 = var_21.validate(var_24)
    var_26 = 'invalid_key'
    var_27 = [var_11]
    var_28 = module_0.Object(required=var_27)
    var_29 = {}
    var_30 = var_28.validate(var_29)
    var_31 = 'required'
    var_32 = 'Unknown'
    var_33 = module_0.String()
    var_34 = {var_11: var_33}
    var_35 = module_0.Object(properties=var_34)
    var_36 = {}
    var_37 = var_35.validate(var_36)
    var_38 = 2
    var_39 = module_0.Object(min_properties=var_38)
    var_40 = 'name'
    var_41 = 'John'
    var_42 = {var_40: var_41}
    var_43 = var_39.validate(var_42)
    var_44 = module_0.Object(max_properties=var_40)
    var_45 = 'name'
    var_46 = 'age'
    var_47 = 'John'
    var_48 = 30
    var_49 = {var_45: var_47, var_46: var_48}
    var_50 = var_44.validate(var_49)
    var_51 = module_0.String()
    var_52 = {var_49: var_51}
    var_53 = module_0.Object(properties=var_52, additional_properties=var_45)
    var_54 = 'extra'
    var_55 = 'value'
    var_56 = {var_49: var_17, var_54: var_55}
    var_57 = var_53.validate(var_56)
    var_58 = module_0.String()
    var_59 = {var_49: var_58}
    var_60 = module_0.Object(properties=var_59, additional_properties=var_48)
    var_61 = 'name'
    var_62 = 'extra'
    var_63 = 'John'
    var_64 = 'value'
    var_65 = {var_61: var_63, var_62: var_64}
    var_66 = var_60.validate(var_65)
    var_67 = 'invalid_property'
    var_68 = module_0.String()
    var_69 = {var_65: var_68}
    var_70 = module_0.Integer()
    var_71 = module_0.Object(properties=var_69, additional_properties=var_70)
    var_72 = 'count'
    var_73 = 42
    var_74 = {var_65: var_17, var_72: var_73}
    var_75 = var_71.validate(var_74)
    var_76 = '^num_'
    var_77 = module_0.Integer()
    var_78 = {var_76: var_77}
    var_79 = module_0.Object(pattern_properties=var_78)
    var_80 = 'num_1'
    var_81 = 'num_2'
    var_82 = 10
    var_83 = 20
    var_84 = {var_80: var_82, var_81: var_83}
    var_85 = var_79.validate(var_84)
    var_86 = '^[a-z]+$'
    var_87 = module_0.String(pattern=var_86)
    var_88 = module_0.Object(property_names=var_87)
    var_89 = 'Invalid'
    var_90 = 'value'
    var_91 = {var_89: var_90}
    var_92 = var_88.validate(var_91)
    var_93 = 'user'
    var_94 = module_0.String()
    var_95 = {var_65: var_94}
    var_96 = module_0.Object(properties=var_95)
    var_97 = {var_93: var_96}
    var_98 = module_0.Object(properties=var_97)
    var_99 = {var_65: var_17}
    var_100 = {var_93: var_99}
    var_101 = var_98.validate(var_100)
    var_102 = module_0.Integer()
    var_103 = {var_66: var_102}
    var_104 = module_0.Object(properties=var_103)
    var_105 = {var_93: var_104}
    var_106 = module_0.Object(properties=var_105)
    var_107 = 'user'
    var_108 = 'age'
    var_109 = 'not_an_int'
    var_110 = {var_108: var_109}
    var_111 = {var_107: var_110}
    var_112 = var_106.validate(var_111)
    var_113 = module_0.Object()
    var_114 = {}
    var_115 = var_113.validate(var_114)
    var_116 = module_0.Object()
    var_117 = 'key'
    var_118 = (var_117, var_55)
    var_119 = [var_118]
    var_120 = [var_111, var_112]
    var_121 = module_0.String()
    var_122 = module_0.Integer()
    var_123 = {var_111: var_121, var_112: var_122}
    var_124 = module_0.Object(properties=var_123, required=var_120)
    var_125 = 'name'
    var_126 = 'age'
    var_127 = 'John'
    var_128 = 'invalid'
    var_129 = {var_125: var_127, var_126: var_128}
    var_130 = var_124.validate(var_129)
    var_131 = 5
    var_132 = module_0.String(max_length=var_131)
    var_133 = {var_129: var_132}
    var_134 = module_0.Object(properties=var_133)
    var_135 = 'name'
    var_136 = 'VeryLongName'
    var_137 = {var_135: var_136}
    var_138 = var_134.validate(var_137)



# Parsed testcases at query #30
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Const(var_0)
    var_2 = None
    var_3 = module_0.Const(var_2)
    var_4 = 'test_value'
    var_5 = module_0.Const(var_4)
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]
    var_10 = module_0.Const(var_9)
    var_11 = 'key'
    var_12 = 'value'
    var_13 = {var_11: var_12}
    var_14 = module_0.Const(var_13)
    var_15 = 42
    var_16 = True
    var_17 = module_0.Const(var_15)
    var_18 = None
    var_19 = False
    var_20 = module_0.Const(var_18)
    var_21 = 'default_value'
    var_22 = module_0.Const(var_12)
    var_23 = 100
    var_24 = 'Custom error'
    var_25 = module_0.Const(var_23)



# Parsed testcases at query #31
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = [var_0, var_1]
    var_5 = True
    var_6 = module_0.Union(var_4)
    var_7 = module_0.String()
    var_8 = module_0.Integer()
    var_9 = [var_7, var_8]
    var_10 = module_0.Union(var_9)
    var_11 = module_0.String()
    var_12 = module_0.Integer()
    var_13 = [var_11, var_12]
    var_14 = module_0.Union(var_13)
    var_15 = []
    var_16 = module_0.Union(var_15)
    var_17 = [var_0]
    var_18 = module_0.Union(var_17)
    var_19 = var_18.any_of
    var_20 = len(var_19)
    assert var_20 == 1
    var_21 = module_0.String()
    var_22 = module_0.Integer()
    var_23 = module_0.Float()
    var_24 = module_0.Boolean()
    var_25 = module_0.String()
    var_26 = module_0.Array(var_25)
    var_27 = [var_21, var_22, var_23, var_24, var_26]
    var_28 = module_0.Union(var_27)
    var_29 = var_28.any_of
    var_30 = len(var_29)
    assert var_30 == 5
    var_31 = [var_0]
    var_32 = False
    var_33 = module_0.Union(var_31)



# Parsed testcases at query #32
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 42
    var_1 = '42'
    var_2 = 3.14
    var_3 = '3.14'
    var_4 = True
    var_5 = module_0.Number()
    var_6 = None
    var_7 = var_5.validate(var_6)
    assert var_7 is None
    var_8 = False
    var_9 = module_0.Number()
    var_10 = None
    var_11 = var_9.validate(var_10)
    var_12 = module_0.Number(coerce_types=var_4)
    var_13 = ''
    var_14 = var_12.validate(var_13)
    assert var_14 is None
    var_15 = module_0.Number()
    var_16 = True
    var_17 = var_15.validate(var_16)
    var_18 = 3.14
    var_19 = var_15.validate(var_18)
    var_20 = module_0.Number(coerce_types=var_8)
    var_21 = '42'
    var_22 = var_20.validate(var_21)
    var_23 = module_0.Number()
    var_24 = 'inf'
    var_25 = float(var_24)
    var_26 = var_23.validate(var_25)
    var_27 = '-inf'
    var_28 = float(var_27)
    var_29 = var_23.validate(var_28)
    var_30 = 'nan'
    var_31 = float(var_30)
    var_32 = var_23.validate(var_31)
    var_33 = 10
    var_34 = module_0.Number(minimum=var_33)
    var_35 = var_34.validate(var_33)
    assert var_35 == 10
    var_36 = 20
    var_37 = var_34.validate(var_36)
    assert var_37 == 20
    var_38 = 5
    var_39 = var_34.validate(var_38)
    var_40 = module_0.Number(exclusive_minimum=var_33)
    var_41 = 11
    var_42 = var_40.validate(var_41)
    assert var_42 == 11
    var_43 = 10
    var_44 = var_40.validate(var_43)
    var_45 = 100
    var_46 = module_0.Number(maximum=var_45)
    var_47 = var_46.validate(var_45)
    assert var_47 == 100
    var_48 = 50
    var_49 = var_46.validate(var_48)
    assert var_49 == 50
    var_50 = 150
    var_51 = var_46.validate(var_50)
    var_52 = module_0.Number(exclusive_maximum=var_45)
    var_53 = 99
    var_54 = var_52.validate(var_53)
    assert var_54 == 99
    var_55 = 100
    var_56 = var_52.validate(var_55)
    var_57 = 5
    var_58 = module_0.Number(multiple_of=var_57)
    var_59 = var_58.validate(var_33)
    assert var_59 == 10
    var_60 = var_58.validate(var_8)
    assert var_60 == 0
    var_61 = 7
    var_62 = var_58.validate(var_61)
    var_63 = 0.5
    var_64 = module_0.Number(multiple_of=var_63)
    var_65 = var_64.validate(var_4)
    var_66 = 1.5
    var_67 = var_64.validate(var_66)
    var_68 = 1.3
    var_69 = var_64.validate(var_68)
    var_70 = '0.01'
    var_71 = 3.145
    var_72 = var_64.validate(var_71)
    var_73 = '1'
    var_74 = var_64.validate(var_68)
    assert var_74 == 42
    var_75 = module_0.Number()
    var_76 = '42.5'
    var_77 = module_0.Number()
    var_78 = 'not_a_number'
    var_79 = var_77.validate(var_78)
    var_80 = module_0.Number(minimum=var_8, maximum=var_45, multiple_of=var_33)
    var_81 = var_80.validate(var_48)
    assert var_81 == 50
    var_82 = 150
    var_83 = var_80.validate(var_82)
    var_84 = 55
    var_85 = var_80.validate(var_84)



# Parsed testcases at query #33
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Number()
    var_1 = 42
    var_2 = var_0.validate(var_1)
    assert var_2 == 42
    var_3 = 0
    var_4 = var_0.validate(var_3)
    assert var_4 == 0
    var_5 = -10
    var_6 = var_0.validate(var_5)
    assert var_6 == -10
    var_7 = 3.14
    var_8 = var_0.validate(var_7)
    var_9 = -2.5
    var_10 = var_0.validate(var_9)
    var_11 = '100'
    var_12 = var_0.validate(var_11)
    assert var_12 == 100
    var_13 = '3.14'
    var_14 = var_0.validate(var_13)
    var_15 = True
    var_16 = module_0.Number()
    var_17 = None
    var_18 = var_16.validate(var_17)
    assert var_18 is None
    var_19 = False
    var_20 = module_0.Number()
    var_21 = None
    var_22 = var_20.validate(var_21)
    var_23 = ''
    var_24 = var_16.validate(var_23)
    assert var_24 is None
    var_25 = True
    var_26 = var_0.validate(var_25)
    var_27 = 10
    var_28 = module_0.Number(minimum=var_27)
    var_29 = var_28.validate(var_27)
    assert var_29 == 10
    var_30 = 100
    var_31 = var_28.validate(var_30)
    assert var_31 == 100
    var_32 = 5
    var_33 = var_28.validate(var_32)
    var_34 = 50
    var_35 = module_0.Number(maximum=var_34)
    var_36 = var_35.validate(var_34)
    assert var_36 == 50
    var_37 = var_35.validate(var_27)
    assert var_37 == 10
    var_38 = 100
    var_39 = var_35.validate(var_38)
    var_40 = module_0.Number(exclusive_minimum=var_27)
    var_41 = 11
    var_42 = var_40.validate(var_41)
    assert var_42 == 11
    var_43 = 10
    var_44 = var_40.validate(var_43)
    var_45 = module_0.Number(exclusive_maximum=var_34)
    var_46 = 49
    var_47 = var_45.validate(var_46)
    assert var_47 == 49
    var_48 = 50
    var_49 = var_45.validate(var_48)
    var_50 = 'inf'
    var_51 = float(var_50)
    var_52 = var_0.validate(var_51)
    var_53 = '-inf'
    var_54 = float(var_53)
    var_55 = var_0.validate(var_54)
    var_56 = 'nan'
    var_57 = float(var_56)
    var_58 = var_0.validate(var_57)
    var_59 = 5
    var_60 = module_0.Number(multiple_of=var_59)
    var_61 = var_60.validate(var_27)
    assert var_61 == 10
    var_62 = var_60.validate(var_19)
    assert var_62 == 0
    var_63 = 7
    var_64 = var_60.validate(var_63)
    var_65 = 0.5
    var_66 = module_0.Number(multiple_of=var_65)
    var_67 = var_66.validate(var_15)
    var_68 = 2.5
    var_69 = var_66.validate(var_68)
    var_70 = 1.3
    var_71 = var_66.validate(var_70)
    var_72 = '0.01'
    var_73 = module_0.Number(precision=var_72)
    var_74 = 3.14159
    var_75 = var_73.validate(var_74)
    var_76 = False
    var_77 = module_0.Number(coerce_types=var_76)
    var_78 = var_77.validate(var_70)
    assert var_78 == 42
    var_79 = '100'
    var_80 = var_77.validate(var_79)
    var_81 = module_0.Integer()
    var_82 = 3.14
    var_83 = 3.0
    var_84 = module_0.Number()
    var_85 = '10.5'



# Parsed testcases at query #34
#--------------------------


import typesystem.fields as module_0

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
    var_10 = module_0.Boolean()
    var_11 = var_10.validate(var_4)
    assert var_11 is False
    var_12 = module_0.Boolean(coerce_types=var_6)
    var_13 = 'true'
    var_14 = var_12.validate(var_13)
    assert var_14 is True
    var_15 = module_0.Boolean(coerce_types=var_6)
    var_16 = 'false'
    var_17 = var_15.validate(var_16)
    assert var_17 is False
    var_18 = module_0.Boolean(coerce_types=var_6)
    var_19 = 'on'
    var_20 = var_18.validate(var_19)
    assert var_20 is True
    var_21 = module_0.Boolean(coerce_types=var_6)
    var_22 = 'off'
    var_23 = var_21.validate(var_22)
    assert var_23 is False
    var_24 = module_0.Boolean(coerce_types=var_6)
    var_25 = '1'
    var_26 = var_24.validate(var_25)
    assert var_26 is True
    var_27 = module_0.Boolean(coerce_types=var_6)
    var_28 = '0'
    var_29 = var_27.validate(var_28)
    assert var_29 is False
    var_30 = module_0.Boolean(coerce_types=var_6)
    var_31 = ''
    var_32 = var_30.validate(var_31)
    assert var_32 is False
    var_33 = module_0.Boolean(coerce_types=var_6)
    var_34 = var_33.validate(var_6)
    assert var_34 is True
    var_35 = module_0.Boolean(coerce_types=var_6)
    var_36 = var_35.validate(var_4)
    assert var_36 is False
    var_37 = module_0.Boolean(coerce_types=var_6)
    var_38 = 'TRUE'
    var_39 = var_37.validate(var_38)
    assert var_39 is True
    var_40 = module_0.Boolean(coerce_types=var_6)
    var_41 = 'null'
    var_42 = var_40.validate(var_41)
    assert var_42 is None
    var_43 = module_0.Boolean(coerce_types=var_6)
    var_44 = 'none'
    var_45 = var_43.validate(var_44)
    assert var_45 is None
    var_46 = module_0.Boolean(coerce_types=var_6)
    var_47 = 'invalid'
    var_48 = var_46.validate(var_47)
    var_49 = module_0.Boolean(coerce_types=var_4)
    var_50 = 'true'
    var_51 = var_49.validate(var_50)
    var_52 = module_0.Boolean(coerce_types=var_4)
    var_53 = 1
    var_54 = var_52.validate(var_53)
    var_55 = module_0.Boolean(coerce_types=var_53)
    var_56 = var_55.validate(var_31)
    assert var_56 is None



# Parsed testcases at query #35
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = True
    var_2 = var_0.validate(var_1)
    assert var_2 is True
    var_3 = False
    var_4 = var_0.validate(var_3)
    assert var_4 is False
    var_5 = 'true'
    var_6 = var_0.validate(var_5)
    assert var_6 is True
    var_7 = 'false'
    var_8 = var_0.validate(var_7)
    assert var_8 is False
    var_9 = 'on'
    var_10 = var_0.validate(var_9)
    assert var_10 is True
    var_11 = 'off'
    var_12 = var_0.validate(var_11)
    assert var_12 is False
    var_13 = '1'
    var_14 = var_0.validate(var_13)
    assert var_14 is True
    var_15 = '0'
    var_16 = var_0.validate(var_15)
    assert var_16 is False
    var_17 = ''
    var_18 = var_0.validate(var_17)
    assert var_18 is False
    var_19 = var_0.validate(var_1)
    assert var_19 is True
    var_20 = var_0.validate(var_3)
    assert var_20 is False
    var_21 = 'TRUE'
    var_22 = var_0.validate(var_21)
    assert var_22 is True
    var_23 = 'FALSE'
    var_24 = var_0.validate(var_23)
    assert var_24 is False
    var_25 = 'On'
    var_26 = var_0.validate(var_25)
    assert var_26 is True
    var_27 = 'OFF'
    var_28 = var_0.validate(var_27)
    assert var_28 is False
    var_29 = module_0.Boolean()
    var_30 = None
    var_31 = var_29.validate(var_30)
    var_32 = module_0.Boolean()
    var_33 = None
    var_34 = var_32.validate(var_33)
    assert var_34 is None
    var_35 = module_0.Boolean(coerce_types=var_30)
    var_36 = 'null'
    var_37 = var_35.validate(var_36)
    assert var_37 is None
    var_38 = 'none'
    var_39 = var_35.validate(var_38)
    assert var_39 is None
    var_40 = module_0.Boolean(coerce_types=var_3)
    var_41 = 'invalid'
    var_42 = var_40.validate(var_41)
    var_43 = module_0.Boolean(coerce_types=var_41)
    var_44 = 'invalid'
    var_45 = var_43.validate(var_44)
    var_46 = []
    var_47 = var_0.validate(var_46)
    var_48 = var_32.validate(var_17)
    assert var_48 is False
    var_49 = var_35.validate(var_17)
    assert var_49 is False



# Parsed testcases at query #36
#--------------------------


import typesystem.fields as module_0

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
    var_16 = 'invalid_key'
    var_17 = module_0.Object()
    var_18 = 'key'
    var_19 = 'value'
    var_20 = {var_18: var_19}
    var_21 = var_17.validate(var_20)
    var_22 = 'name'
    var_23 = 'age'
    var_24 = module_0.String()
    var_25 = module_0.Integer()
    var_26 = {var_22: var_24, var_23: var_25}
    var_27 = module_0.Object(properties=var_26)
    var_28 = 'John'
    var_29 = 30
    var_30 = {var_22: var_28, var_23: var_29}
    var_31 = var_27.validate(var_30)
    var_32 = [var_22]
    var_33 = module_0.Object(required=var_32)
    var_34 = {}
    var_35 = var_33.validate(var_34)
    var_36 = 'required'
    var_37 = 'Unknown'
    var_38 = module_0.String()
    var_39 = {var_22: var_38}
    var_40 = module_0.Object(properties=var_39)
    var_41 = {}
    var_42 = var_40.validate(var_41)
    var_43 = 2
    var_44 = module_0.Object(min_properties=var_43)
    var_45 = 'key'
    var_46 = 'value'
    var_47 = {var_45: var_46}
    var_48 = var_44.validate(var_47)
    var_49 = module_0.Object(min_properties=var_45)
    var_50 = {}
    var_51 = var_49.validate(var_50)
    var_52 = module_0.Object(max_properties=var_50)
    var_53 = 'key1'
    var_54 = 'key2'
    var_55 = 'value1'
    var_56 = 'value2'
    var_57 = {var_53: var_55, var_54: var_56}
    var_58 = var_52.validate(var_57)
    var_59 = module_0.Object(additional_properties=var_53)
    var_60 = 'key1'
    var_61 = 'key2'
    var_62 = 'value1'
    var_63 = 'value2'
    var_64 = {var_60: var_62, var_61: var_63}
    var_65 = var_59.validate(var_64)
    var_66 = module_0.String()
    var_67 = {var_22: var_66}
    var_68 = module_0.Object(properties=var_67, additional_properties=var_56)
    var_69 = 'name'
    var_70 = 'extra'
    var_71 = 'John'
    var_72 = 'field'
    var_73 = {var_69: var_71, var_70: var_72}
    var_74 = var_68.validate(var_73)
    var_75 = 'invalid_property'
    var_76 = module_0.String()
    var_77 = {var_22: var_76}
    var_78 = module_0.Integer()
    var_79 = module_0.Object(properties=var_77, additional_properties=var_78)
    var_80 = 'count'
    var_81 = 42
    var_82 = {var_22: var_28, var_80: var_81}
    var_83 = var_79.validate(var_82)
    var_84 = '^num_'
    var_85 = module_0.Integer()
    var_86 = {var_84: var_85}
    var_87 = module_0.Object(pattern_properties=var_86)
    var_88 = 'num_1'
    var_89 = 'num_2'
    var_90 = 10
    var_91 = 20
    var_92 = {var_88: var_90, var_89: var_91}
    var_93 = var_87.validate(var_92)
    var_94 = '^[a-z]+$'
    var_95 = module_0.String(pattern=var_94)
    var_96 = module_0.Object(property_names=var_95)
    var_97 = 'Invalid'
    var_98 = 'value'
    var_99 = {var_97: var_98}
    var_100 = var_96.validate(var_99)
    var_101 = 'id'
    var_102 = module_0.Integer()
    var_103 = {var_101: var_102}
    var_104 = module_0.Object(properties=var_103)
    var_105 = 'nested'
    var_106 = {var_105: var_104}
    var_107 = module_0.Object(properties=var_106)
    var_108 = 123
    var_109 = {var_101: var_108}
    var_110 = {var_105: var_109}
    var_111 = var_107.validate(var_110)
    var_112 = module_0.Integer()
    var_113 = {var_101: var_112}
    var_114 = module_0.Object(properties=var_113)
    var_115 = {var_105: var_114}
    var_116 = module_0.Object(properties=var_115)
    var_117 = 'nested'
    var_118 = 'id'
    var_119 = 'not_an_int'
    var_120 = {var_118: var_119}
    var_121 = {var_117: var_120}
    var_122 = var_116.validate(var_121)
    var_123 = module_0.Object()
    var_124 = {}
    var_125 = var_123.validate(var_124)
    var_126 = module_0.String()
    var_127 = {var_18: var_19}
    var_128 = var_123.validate(var_127)
    var_129 = module_0.String()
    var_130 = module_0.Integer()
    var_131 = {var_22: var_129, var_23: var_130}
    var_132 = [var_22, var_23]
    var_133 = module_0.Object(properties=var_131, required=var_132)
    var_134 = {}
    var_135 = var_133.validate(var_134)



# Parsed testcases at query #37
#--------------------------


import typesystem.fields as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'hello'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'hello'
    var_3 = True
    var_4 = module_0.String()
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None
    var_7 = False
    var_8 = module_0.String()
    var_9 = None
    var_10 = var_8.validate(var_9)
    var_11 = module_0.String(allow_blank=var_7)
    var_12 = ''
    var_13 = var_11.validate(var_12)
    var_14 = module_0.String(allow_blank=var_3)
    var_15 = ''
    var_16 = var_14.validate(var_15)
    assert var_16 == ''
    var_17 = module_0.String(trim_whitespace=var_3)
    var_18 = '  hello  '
    var_19 = var_17.validate(var_18)
    assert var_19 == 'hello'
    var_20 = module_0.String(trim_whitespace=var_7)
    var_21 = var_20.validate(var_18)
    assert var_21 == '  hello  '
    var_22 = 5
    var_23 = module_0.String(max_length=var_22)
    var_24 = var_23.validate(var_12)
    assert var_24 == 'hello'
    var_25 = 'toolong'
    var_26 = var_23.validate(var_25)
    var_27 = 3
    var_28 = module_0.String(min_length=var_27)
    var_29 = var_28.validate(var_25)
    assert var_29 == 'hello'
    var_30 = 'hi'
    var_31 = var_28.validate(var_30)
    var_32 = '^\\d+$'
    var_33 = module_0.String(pattern=var_32)
    var_34 = '123'
    var_35 = var_33.validate(var_34)
    assert var_35 == '123'
    var_36 = 'abc'
    var_37 = var_33.validate(var_36)
    var_38 = '^[a-z]+$'
    var_39 = module_1.compile(var_38)
    var_40 = module_0.String(pattern=var_39)
    var_41 = 'abc'
    var_42 = var_40.validate(var_41)
    assert var_42 == 'abc'
    var_43 = 'ABC'
    var_44 = var_40.validate(var_43)
    var_45 = module_0.String()
    var_46 = 123
    var_47 = var_45.validate(var_46)
    var_48 = module_0.String()
    var_49 = 'hel\x00lo'
    var_50 = var_48.validate(var_49)
    assert var_50 == 'hello'
    var_51 = module_0.String(allow_blank=var_3, coerce_types=var_3)
    var_52 = var_51.validate(var_5)
    assert var_52 == ''
    var_53 = module_0.String(trim_whitespace=var_3, coerce_types=var_3)
    var_54 = var_53.validate(var_15)
    assert var_54 is None
    var_55 = module_0.String(allow_blank=var_3, coerce_types=var_7)
    var_56 = None
    var_57 = var_55.validate(var_56)
    var_58 = 'email'
    var_59 = module_0.String(format=var_58)
    var_60 = 'test@example.com'
    var_61 = var_59.validate(var_60)
    var_62 = 'uuid'
    var_63 = module_0.String(format=var_62)
    var_64 = '550e8400-e29b-41d4-a716-446655440000'
    var_65 = var_63.validate(var_64)
    var_66 = 2
    var_67 = 10
    var_68 = module_0.String(max_length=var_67, min_length=var_66, pattern=var_38)
    var_69 = var_68.validate(var_56)
    assert var_69 == 'hello'
    var_70 = 'a'
    var_71 = var_68.validate(var_70)
    var_72 = 'verylongstring'
    var_73 = var_68.validate(var_72)
    var_74 = 'HELLO'
    var_75 = var_68.validate(var_74)



# Parsed testcases at query #38
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = True
    var_2 = var_0.validate(var_1)
    assert var_2 is True
    var_3 = False
    var_4 = var_0.validate(var_3)
    assert var_4 is False
    var_5 = module_0.Boolean()
    var_6 = None
    var_7 = var_5.validate(var_6)
    var_8 = module_0.Boolean()
    var_9 = None
    var_10 = var_8.validate(var_9)
    assert var_10 is None
    var_11 = module_0.Boolean()
    var_12 = 'true'
    var_13 = var_11.validate(var_12)
    assert var_13 is True
    var_14 = 'True'
    var_15 = var_11.validate(var_14)
    assert var_15 is True
    var_16 = 'TRUE'
    var_17 = var_11.validate(var_16)
    assert var_17 is True
    var_18 = 'false'
    var_19 = var_11.validate(var_18)
    assert var_19 is False
    var_20 = 'False'
    var_21 = var_11.validate(var_20)
    assert var_21 is False
    var_22 = 'FALSE'
    var_23 = var_11.validate(var_22)
    assert var_23 is False
    var_24 = 'on'
    var_25 = var_11.validate(var_24)
    assert var_25 is True
    var_26 = 'off'
    var_27 = var_11.validate(var_26)
    assert var_27 is False
    var_28 = '1'
    var_29 = var_11.validate(var_28)
    assert var_29 is True
    var_30 = '0'
    var_31 = var_11.validate(var_30)
    assert var_31 is False
    var_32 = module_0.Boolean()
    var_33 = var_32.validate(var_6)
    assert var_33 is True
    var_34 = var_32.validate(var_3)
    assert var_34 is False
    var_35 = module_0.Boolean()
    var_36 = ''
    var_37 = var_35.validate(var_36)
    assert var_37 is False
    var_38 = module_0.Boolean()
    var_39 = var_38.validate(var_36)
    assert var_39 is None
    var_40 = module_0.Boolean()
    var_41 = 'null'
    var_42 = var_40.validate(var_41)
    assert var_42 is None
    var_43 = 'none'
    var_44 = var_40.validate(var_43)
    assert var_44 is None
    var_45 = module_0.Boolean()
    var_46 = 'invalid'
    var_47 = var_45.validate(var_46)
    var_48 = module_0.Boolean()
    var_49 = []
    var_50 = var_48.validate(var_49)
    var_51 = module_0.Boolean(coerce_types=var_3)
    var_52 = 'true'
    var_53 = var_51.validate(var_52)
    var_54 = module_0.Boolean(coerce_types=var_3)
    var_55 = 1
    var_56 = var_54.validate(var_55)
    var_57 = module_0.Boolean(coerce_types=var_3)
    var_58 = None
    var_59 = var_57.validate(var_58)



# Parsed testcases at query #39
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = True
    var_5 = module_0.String()
    var_6 = module_0.Integer()
    var_7 = [var_5, var_6]
    var_8 = module_0.Union(var_7)
    var_9 = module_0.String()
    var_10 = module_0.Integer()
    var_11 = [var_9, var_10]
    var_12 = module_0.Union(var_11)
    var_13 = module_0.String()
    var_14 = module_0.Integer()
    var_15 = module_0.Float()
    var_16 = [var_13, var_14, var_15]
    var_17 = module_0.Union(var_16)
    var_18 = module_0.String()
    var_19 = [var_18]
    var_20 = module_0.Union(var_19)
    var_21 = var_20.any_of
    var_22 = len(var_21)
    assert var_22 == 1
    var_23 = module_0.String()
    var_24 = module_0.Integer()
    var_25 = module_0.Float()
    var_26 = module_0.Boolean()
    var_27 = module_0.Array()
    var_28 = [var_23, var_24, var_25, var_26, var_27]
    var_29 = module_0.Union(var_28)
    var_30 = var_29.any_of
    var_31 = len(var_30)
    assert var_31 == 5
    var_32 = module_0.String()
    var_33 = module_0.Integer()
    var_34 = [var_32, var_33]
    var_35 = False
    var_36 = module_0.Union(var_34)
    var_37 = module_0.String()
    var_38 = module_0.Integer()
    var_39 = [var_37, var_38]
    var_40 = module_0.Union(var_39)
    var_41 = 'name'
    var_42 = module_0.String()
    var_43 = {var_41: var_42}
    var_44 = module_0.Object(properties=var_43)
    var_45 = module_0.Integer()
    var_46 = module_0.Array(var_45)
    var_47 = [var_44, var_46]
    var_48 = module_0.Union(var_47)



# Parsed testcases at query #40
#--------------------------


import typesystem.fields as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'hello'
    var_2 = var_0.validate(var_1)
    assert var_2 == 'hello'
    var_3 = '  hello  '
    var_4 = var_0.validate(var_3)
    assert var_4 == 'hello'
    var_5 = False
    var_6 = module_0.String(trim_whitespace=var_5)
    var_7 = var_6.validate(var_3)
    assert var_7 == '  hello  '
    var_8 = True
    var_9 = module_0.String()
    var_10 = None
    var_11 = var_9.validate(var_10)
    assert var_11 is None
    var_12 = module_0.String()
    var_13 = module_0.String(allow_blank=var_8)
    var_14 = ''
    var_15 = var_13.validate(var_14)
    assert var_15 == ''
    var_16 = module_0.String(allow_blank=var_5)
    var_17 = module_0.String()
    var_18 = 123
    var_19 = 5
    var_20 = module_0.String(max_length=var_19)
    var_21 = var_20.validate(var_1)
    assert var_21 == 'hello'
    var_22 = 'toolong'
    var_23 = 3
    var_24 = module_0.String(min_length=var_23)
    var_25 = var_24.validate(var_1)
    assert var_25 == 'hello'
    var_26 = 'hi'
    var_27 = '^\\d+$'
    var_28 = module_0.String(pattern=var_27)
    var_29 = '123'
    var_30 = var_28.validate(var_29)
    assert var_30 == '123'
    var_31 = 'abc'
    var_32 = '^[a-z]+$'
    var_33 = module_1.compile(var_32)
    var_34 = module_0.String(pattern=var_33)
    var_35 = var_34.validate(var_31)
    assert var_35 == 'abc'
    var_36 = module_0.String()
    var_37 = 'hel\x00lo'
    var_38 = var_36.validate(var_37)
    assert var_38 == 'hello'
    var_39 = module_0.String(allow_blank=var_8, coerce_types=var_8)
    var_40 = var_39.validate(var_10)
    assert var_40 == ''
    var_41 = module_0.String(coerce_types=var_8)
    var_42 = var_41.validate(var_14)
    assert var_42 is None
    var_43 = 'email'
    var_44 = module_0.String(format=var_43)
    var_45 = 'test@example.com'
    var_46 = var_44.validate(var_45)
    assert var_46 == 'test@example.com'
    var_47 = 'invalid-email'
    var_48 = 2
    var_49 = 10
    var_50 = module_0.String(allow_blank=var_5, max_length=var_49, min_length=var_48)
    var_51 = var_50.validate(var_1)
    assert var_51 == 'hello'
    var_52 = 'a'



# Parsed testcases at query #41
#--------------------------


import typesystem.fields as module_0

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
    var_10 = module_0.Boolean()
    var_11 = var_10.validate(var_4)
    assert var_11 is False
    var_12 = module_0.Boolean(coerce_types=var_6)
    var_13 = 'true'
    var_14 = var_12.validate(var_13)
    assert var_14 is True
    var_15 = module_0.Boolean(coerce_types=var_6)
    var_16 = 'false'
    var_17 = var_15.validate(var_16)
    assert var_17 is False
    var_18 = module_0.Boolean(coerce_types=var_6)
    var_19 = 'on'
    var_20 = var_18.validate(var_19)
    assert var_20 is True
    var_21 = module_0.Boolean(coerce_types=var_6)
    var_22 = 'off'
    var_23 = var_21.validate(var_22)
    assert var_23 is False
    var_24 = module_0.Boolean(coerce_types=var_6)
    var_25 = '1'
    var_26 = var_24.validate(var_25)
    assert var_26 is True
    var_27 = module_0.Boolean(coerce_types=var_6)
    var_28 = '0'
    var_29 = var_27.validate(var_28)
    assert var_29 is False
    var_30 = module_0.Boolean(coerce_types=var_6)
    var_31 = ''
    var_32 = var_30.validate(var_31)
    assert var_32 is False
    var_33 = module_0.Boolean(coerce_types=var_6)
    var_34 = var_33.validate(var_6)
    assert var_34 is True
    var_35 = module_0.Boolean(coerce_types=var_6)
    var_36 = var_35.validate(var_4)
    assert var_36 is False
    var_37 = module_0.Boolean(coerce_types=var_6)
    var_38 = 'TRUE'
    var_39 = var_37.validate(var_38)
    assert var_39 is True
    var_40 = 'FALSE'
    var_41 = var_37.validate(var_40)
    assert var_41 is False
    var_42 = module_0.Boolean(coerce_types=var_6)
    var_43 = 'null'
    var_44 = var_42.validate(var_43)
    assert var_44 is None
    var_45 = 'none'
    var_46 = var_42.validate(var_45)
    assert var_46 is None
    var_47 = var_42.validate(var_31)
    assert var_47 is None
    var_48 = module_0.Boolean(coerce_types=var_6)
    var_49 = 'invalid'
    var_50 = var_48.validate(var_49)
    var_51 = module_0.Boolean(coerce_types=var_4)
    var_52 = 'true'
    var_53 = var_51.validate(var_52)
    var_54 = module_0.Boolean(coerce_types=var_4)
    var_55 = 1
    var_56 = var_54.validate(var_55)
    var_57 = module_0.Boolean(coerce_types=var_4)
    var_58 = None
    var_59 = var_57.validate(var_58)
    var_60 = module_0.Boolean(coerce_types=var_58)
    var_61 = 'TrUe'
    var_62 = var_60.validate(var_61)
    assert var_62 is True
    var_63 = 'FaLsE'
    var_64 = var_60.validate(var_63)
    assert var_64 is False
    var_65 = module_0.Boolean(coerce_types=var_58)
    var_66 = 2
    var_67 = var_65.validate(var_66)
    var_68 = module_0.Boolean(coerce_types=var_66)
    var_69 = []
    var_70 = var_68.validate(var_69)



# Parsed testcases at query #42
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Object(properties=var_4)
    var_6 = 'John'
    var_7 = 30
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = True
    var_11 = module_0.Object()
    var_12 = None
    var_13 = var_11.validate(var_12)
    assert var_13 is None
    var_14 = False
    var_15 = module_0.Object()
    var_16 = None
    var_17 = var_15.validate(var_16)
    var_18 = module_0.Object()
    var_19 = 'not a dict'
    var_20 = var_18.validate(var_19)
    var_21 = module_0.String()
    var_22 = {var_19: var_21}
    var_23 = [var_19]
    var_24 = module_0.Object(properties=var_22, required=var_23)
    var_25 = {}
    var_26 = var_24.validate(var_25)
    var_27 = 'required'
    var_28 = module_0.Object()
    var_29 = 123
    var_30 = 'value'
    var_31 = {var_29: var_30}
    var_32 = var_28.validate(var_31)
    var_33 = 'invalid_key'
    var_34 = 2
    var_35 = module_0.Object(min_properties=var_34)
    var_36 = 'a'
    var_37 = 1
    var_38 = {var_36: var_37}
    var_39 = var_35.validate(var_38)
    var_40 = module_0.Object(max_properties=var_10)
    var_41 = 'a'
    var_42 = 'b'
    var_43 = 1
    var_44 = 2
    var_45 = {var_41: var_43, var_42: var_44}
    var_46 = var_40.validate(var_45)
    var_47 = 'Unknown'
    var_48 = module_0.String()
    var_49 = {var_41: var_48}
    var_50 = module_0.Object(properties=var_49)
    var_51 = {}
    var_52 = var_50.validate(var_51)
    var_53 = module_0.String()
    var_54 = {var_41: var_53}
    var_55 = module_0.Object(properties=var_54, additional_properties=var_10)
    var_56 = 'extra'
    var_57 = 'value'
    var_58 = {var_41: var_46, var_56: var_57}
    var_59 = var_55.validate(var_58)
    var_60 = module_0.String()
    var_61 = {var_41: var_60}
    var_62 = module_0.Object(properties=var_61, additional_properties=var_14)
    var_63 = 'name'
    var_64 = 'extra'
    var_65 = 'John'
    var_66 = 'value'
    var_67 = {var_63: var_65, var_64: var_66}
    var_68 = var_62.validate(var_67)
    var_69 = 'invalid_property'
    var_70 = module_0.String()
    var_71 = {var_63: var_70}
    var_72 = module_0.Integer()
    var_73 = module_0.Object(properties=var_71, additional_properties=var_72)
    var_74 = {var_63: var_68, var_64: var_7}
    var_75 = var_73.validate(var_74)
    var_76 = '^num_'
    var_77 = module_0.Integer()
    var_78 = {var_76: var_77}
    var_79 = module_0.Object(pattern_properties=var_78)
    var_80 = 'num_1'
    var_81 = 'num_2'
    var_82 = 10
    var_83 = 20
    var_84 = {var_80: var_82, var_81: var_83}
    var_85 = var_79.validate(var_84)
    var_86 = '^[a-z]+$'
    var_87 = module_0.String(pattern=var_86)
    var_88 = module_0.Object(property_names=var_87)
    var_89 = 'Name'
    var_90 = 'value'
    var_91 = {var_89: var_90}
    var_92 = var_88.validate(var_91)
    var_93 = 'user'
    var_94 = module_0.String()
    var_95 = {var_89: var_94}
    var_96 = module_0.Object(properties=var_95)
    var_97 = {var_93: var_96}
    var_98 = module_0.Object(properties=var_97)
    var_99 = {var_89: var_68}
    var_100 = {var_93: var_99}
    var_101 = var_98.validate(var_100)
    var_102 = module_0.Integer()
    var_103 = {var_90: var_102}
    var_104 = module_0.Object(properties=var_103)
    var_105 = {var_93: var_104}
    var_106 = module_0.Object(properties=var_105)
    var_107 = 'user'
    var_108 = 'age'
    var_109 = 'not_an_int'
    var_110 = {var_108: var_109}
    var_111 = {var_107: var_110}
    var_112 = var_106.validate(var_111)
    var_113 = module_0.Object()
    var_114 = {}
    var_115 = var_113.validate(var_114)
    var_116 = module_0.String()
    var_117 = {var_107: var_116}
    var_118 = module_0.Object(properties=var_117)
    var_119 = (var_107, var_112)
    var_120 = [var_119]



# Parsed testcases at query #43
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = [var_0]
    var_2 = module_0.Union(var_1)
    var_3 = module_0.Integer()
    var_4 = module_0.Float()
    var_5 = [var_3, var_4]
    var_6 = module_0.Union(var_5)
    var_7 = True
    var_8 = module_0.String()
    var_9 = [var_8]
    var_10 = module_0.Union(var_9)
    var_11 = module_0.String()
    var_12 = module_0.Integer()
    var_13 = [var_11, var_12]
    var_14 = module_0.Union(var_13)
    var_15 = module_0.Float()
    var_16 = module_0.Boolean()
    var_17 = [var_15, var_16]
    var_18 = module_0.Union(var_17)
    var_19 = module_0.String()
    var_20 = [var_19]
    var_21 = module_0.Union(var_20)
    var_22 = []
    var_23 = module_0.Union(var_22)
    var_24 = 'name'
    var_25 = module_0.String()
    var_26 = {var_24: var_25}
    var_27 = module_0.Object(properties=var_26)
    var_28 = module_0.Integer()
    var_29 = module_0.Array(var_28)
    var_30 = [var_27, var_29]
    var_31 = module_0.Union(var_30)
    var_32 = module_0.String()
    var_33 = False
    var_34 = module_0.Integer()
    var_35 = [var_34, var_32]
    var_36 = module_0.Union(var_35)
    var_37 = module_0.String()
    var_38 = module_0.Integer()
    var_39 = [var_37, var_38]
    var_40 = module_0.Union(var_39)



# Parsed testcases at query #44
#--------------------------


import typesystem.fields as module_0

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
    var_10 = module_0.Boolean()
    var_11 = var_10.validate(var_4)
    assert var_11 is False
    var_12 = module_0.Boolean(coerce_types=var_6)
    var_13 = 'true'
    var_14 = var_12.validate(var_13)
    assert var_14 is True
    var_15 = module_0.Boolean(coerce_types=var_6)
    var_16 = 'false'
    var_17 = var_15.validate(var_16)
    assert var_17 is False
    var_18 = module_0.Boolean(coerce_types=var_6)
    var_19 = 'on'
    var_20 = var_18.validate(var_19)
    assert var_20 is True
    var_21 = module_0.Boolean(coerce_types=var_6)
    var_22 = 'off'
    var_23 = var_21.validate(var_22)
    assert var_23 is False
    var_24 = module_0.Boolean(coerce_types=var_6)
    var_25 = '1'
    var_26 = var_24.validate(var_25)
    assert var_26 is True
    var_27 = module_0.Boolean(coerce_types=var_6)
    var_28 = '0'
    var_29 = var_27.validate(var_28)
    assert var_29 is False
    var_30 = module_0.Boolean(coerce_types=var_6)
    var_31 = ''
    var_32 = var_30.validate(var_31)
    assert var_32 is False
    var_33 = module_0.Boolean(coerce_types=var_6)
    var_34 = var_33.validate(var_6)
    assert var_34 is True
    var_35 = module_0.Boolean(coerce_types=var_6)
    var_36 = var_35.validate(var_4)
    assert var_36 is False
    var_37 = module_0.Boolean(coerce_types=var_6)
    var_38 = 'TRUE'
    var_39 = var_37.validate(var_38)
    assert var_39 is True
    var_40 = module_0.Boolean(coerce_types=var_6)
    var_41 = 'FALSE'
    var_42 = var_40.validate(var_41)
    assert var_42 is False
    var_43 = module_0.Boolean(coerce_types=var_6)
    var_44 = 'invalid'
    var_45 = var_43.validate(var_44)
    var_46 = module_0.Boolean(coerce_types=var_4)
    var_47 = 'true'
    var_48 = var_46.validate(var_47)
    var_49 = module_0.Boolean(coerce_types=var_47)
    var_50 = 'null'
    var_51 = var_49.validate(var_50)
    assert var_51 is None
    var_52 = module_0.Boolean(coerce_types=var_47)
    var_53 = 'none'
    var_54 = var_52.validate(var_53)
    assert var_54 is None
    var_55 = module_0.Boolean(coerce_types=var_47)
    var_56 = var_55.validate(var_31)
    assert var_56 is None
    var_57 = module_0.Boolean(coerce_types=var_47)
    var_58 = []
    var_59 = var_57.validate(var_58)
    var_60 = module_0.Boolean(coerce_types=var_58)
    var_61 = {}
    var_62 = var_60.validate(var_61)



# Parsed testcases at query #45
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.String()
    var_2 = module_0.Integer()
    var_3 = [var_1, var_2]
    var_4 = module_0.Union(var_3)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None
    var_7 = module_0.String()
    var_8 = module_0.Integer()
    var_9 = [var_7, var_8]
    var_10 = module_0.Union(var_9)
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = 0
    var_14 = exc_info.value.messages()[var_13]
    var_15 = var_14.code
    assert var_15 == 'null'
    var_16 = module_0.String()
    var_17 = module_0.Integer()
    var_18 = [var_16, var_17]
    var_19 = module_0.Union(var_18)
    var_20 = 'hello'
    var_21 = var_19.validate(var_20)
    assert var_21 == 'hello'
    var_22 = module_0.String()
    var_23 = module_0.Integer()
    var_24 = [var_22, var_23]
    var_25 = module_0.Union(var_24)
    var_26 = 42
    var_27 = var_25.validate(var_26)
    assert var_27 == 42
    var_28 = module_0.Boolean()
    var_29 = module_0.String()
    var_30 = module_0.Integer()
    var_31 = [var_28, var_29, var_30]
    var_32 = module_0.Union(var_31)
    var_33 = var_32.validate(var_11)
    assert var_33 is True
    var_34 = 100
    var_35 = module_0.Integer(minimum=var_13, maximum=var_34)
    var_36 = module_0.String()
    var_37 = [var_35, var_36]
    var_38 = module_0.Union(var_37)
    var_39 = 50
    var_40 = var_38.validate(var_39)
    assert var_40 == 50
    var_41 = module_0.Integer(minimum=var_13, maximum=var_34)
    var_42 = module_0.String()
    var_43 = [var_41, var_42]
    var_44 = module_0.Union(var_43)
    var_45 = 'valid_string'
    var_46 = var_44.validate(var_45)
    assert var_46 == 'valid_string'
    var_47 = module_0.Integer()
    var_48 = module_0.Float()
    var_49 = [var_47, var_48]
    var_50 = module_0.Union(var_49)
    var_51 = 'not_a_number'
    var_52 = var_50.validate(var_51)
    var_53 = exc_info.value.messages()[var_13]
    var_54 = var_53.code
    assert var_54 == 'union'
    var_55 = 10
    var_56 = module_0.Integer(minimum=var_55)
    var_57 = module_0.String()
    var_58 = [var_56, var_57]
    var_59 = module_0.Union(var_58)
    var_60 = 5
    var_61 = var_59.validate(var_60)
    var_62 = error.messages()[var_13]
    var_63 = var_62.code
    assert var_63 == 'minimum'
    var_64 = module_0.Integer(minimum=var_55)
    var_65 = 20
    var_66 = module_0.Integer(minimum=var_65)
    var_67 = [var_64, var_66]
    var_68 = module_0.Union(var_67)
    var_69 = 5
    var_70 = var_68.validate(var_69)
    var_71 = exc_info.value.messages()[var_13]
    var_72 = var_71.code
    assert var_72 == 'union'
    var_73 = module_0.String()
    var_74 = module_0.Integer()
    var_75 = [var_73, var_74]
    var_76 = module_0.Union(var_75)
    var_77 = var_76.validate(var_5)
    assert var_77 is None
    var_78 = module_0.String(coerce_types=var_69)
    var_79 = module_0.Integer()
    var_80 = [var_78, var_79]
    var_81 = module_0.Union(var_80)
    var_82 = ''
    var_83 = var_81.validate(var_82)
    assert var_83 == ''
    var_84 = module_0.Integer()
    var_85 = module_0.Float()
    var_86 = [var_84, var_85]
    var_87 = module_0.Union(var_86)
    var_88 = 3.14
    var_89 = var_87.validate(var_88)
    var_90 = module_0.Boolean(coerce_types=var_69)
    var_91 = module_0.String()
    var_92 = [var_90, var_91]
    var_93 = module_0.Union(var_92)
    var_94 = 'true'
    var_95 = var_93.validate(var_94)
    assert var_95 is True
    var_96 = module_0.Integer()
    var_97 = module_0.Array(var_96)
    var_98 = module_0.Object()
    var_99 = [var_97, var_98]
    var_100 = module_0.Union(var_99)
    var_101 = 2
    var_102 = 3
    var_103 = [var_69, var_101, var_102]
    var_104 = var_100.validate(var_103)
    var_105 = module_0.Integer()
    var_106 = module_0.Array(var_105)
    var_107 = module_0.Object()
    var_108 = [var_106, var_107]
    var_109 = module_0.Union(var_108)
    var_110 = 'key'
    var_111 = 'value'
    var_112 = {var_110: var_111}
    var_113 = var_109.validate(var_112)



# Parsed testcases at query #46
#--------------------------


import typesystem.fields as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.String()
    var_1 = True
    var_2 = module_0.String(allow_blank=var_1)
    var_3 = 'custom'
    var_4 = module_0.String(allow_blank=var_1)
    var_5 = 10
    var_6 = module_0.String(max_length=var_5)
    var_7 = 5
    var_8 = module_0.String(min_length=var_7)
    var_9 = '^\\d+$'
    var_10 = module_0.String(pattern=var_9)
    var_11 = '^[a-z]+$'
    var_12 = module_1.compile(var_11)
    var_13 = module_0.String(pattern=var_12)
    var_14 = 'email'
    var_15 = module_0.String(format=var_14)
    var_16 = False
    var_17 = module_0.String(coerce_types=var_16)
    var_18 = 'Username'
    var_19 = "User's login name"
    var_20 = 'guest'
    var_21 = 50
    var_22 = 3
    var_23 = '^[a-zA-Z0-9_]+$'
    var_24 = module_0.String(allow_blank=var_16, trim_whitespace=var_1, max_length=var_21, min_length=var_22, pattern=var_23, format=var_14, coerce_types=var_1)
    var_25 = module_0.String()
    var_26 = module_0.String(trim_whitespace=var_16)



# Parsed testcases at query #47
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Const(var_0)
    var_2 = None
    var_3 = module_0.Const(var_2)
    var_4 = 'test_value'
    var_5 = module_0.Const(var_4)
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]
    var_10 = module_0.Const(var_9)
    var_11 = 'key'
    var_12 = 'value'
    var_13 = {var_11: var_12}
    var_14 = module_0.Const(var_13)
    var_15 = 42
    var_16 = True
    var_17 = module_0.Const(var_15)
    var_18 = 100
    var_19 = 'A constant field'
    var_20 = module_0.Const(var_18)
    var_21 = True
    var_22 = module_0.Const(var_21)
    var_23 = 3.14
    var_24 = module_0.Const(var_23)



# Parsed testcases at query #48
#--------------------------


import typesystem.fields as module_0

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
    var_11 = module_0.Array()
    var_12 = 'key'
    var_13 = 'value'
    var_14 = {var_12: var_13}
    var_15 = var_11.validate(var_14)
    var_16 = 3
    var_17 = module_0.Array(exact_items=var_16)
    var_18 = 1
    var_19 = 2
    var_20 = [var_18, var_19]
    var_21 = var_17.validate(var_20)
    var_22 = module_0.Array(exact_items=var_16)
    var_23 = 1
    var_24 = 2
    var_25 = 3
    var_26 = 4
    var_27 = [var_23, var_24, var_25, var_26]
    var_28 = var_22.validate(var_27)
    var_29 = module_0.Array(exact_items=var_27)
    var_30 = 2
    var_31 = [var_23, var_30, var_27]
    var_32 = var_29.validate(var_31)
    var_33 = module_0.Array(min_items=var_23)
    var_34 = []
    var_35 = var_33.validate(var_34)
    var_36 = module_0.Array(min_items=var_27)
    var_37 = 1
    var_38 = 2
    var_39 = [var_37, var_38]
    var_40 = var_36.validate(var_39)
    var_41 = module_0.Array(max_items=var_30)
    var_42 = 1
    var_43 = 2
    var_44 = 3
    var_45 = [var_42, var_43, var_44]
    var_46 = var_41.validate(var_45)
    var_47 = module_0.Integer()
    var_48 = module_0.Array(var_47)
    var_49 = [var_42, var_30, var_46]
    var_50 = var_48.validate(var_49)
    var_51 = module_0.Integer()
    var_52 = module_0.Array(var_51)
    var_53 = 1
    var_54 = 'invalid'
    var_55 = 3
    var_56 = [var_53, var_54, var_55]
    var_57 = var_52.validate(var_56)
    var_58 = module_0.Integer()
    var_59 = module_0.String()
    var_60 = [var_58, var_59]
    var_61 = module_0.Array(var_60)
    var_62 = 'hello'
    var_63 = [var_53, var_62]
    var_64 = var_61.validate(var_63)
    var_65 = module_0.Integer()
    var_66 = module_0.String()
    var_67 = [var_65, var_66]
    var_68 = module_0.Array(var_67, var_56)
    var_69 = 1
    var_70 = 'hello'
    var_71 = 'extra'
    var_72 = [var_69, var_70, var_71]
    var_73 = var_68.validate(var_72)
    var_74 = module_0.Integer()
    var_75 = module_0.String()
    var_76 = [var_74, var_75]
    var_77 = module_0.Array(var_76, var_69)
    var_78 = 'extra'
    var_79 = [var_69, var_62, var_78]
    var_80 = var_77.validate(var_79)
    var_81 = module_0.Integer()
    var_82 = [var_81]
    var_83 = module_0.String()
    var_84 = module_0.Array(var_82, var_83)
    var_85 = 'world'
    var_86 = [var_69, var_62, var_85]
    var_87 = var_84.validate(var_86)
    var_88 = module_0.Array(unique_items=var_69)
    var_89 = [var_69, var_30, var_73]
    var_90 = var_88.validate(var_89)
    var_91 = module_0.Array(unique_items=var_69)
    var_92 = 1
    var_93 = 2
    var_94 = 3
    var_95 = [var_92, var_93, var_93, var_94]
    var_96 = var_91.validate(var_95)
    var_97 = module_0.Array()
    var_98 = []
    var_99 = var_97.validate(var_98)
    var_100 = 'name'
    var_101 = 'age'
    var_102 = module_0.String()
    var_103 = module_0.Integer()
    var_104 = {var_100: var_102, var_101: var_103}
    var_105 = module_0.Object(properties=var_104)
    var_106 = module_0.Array(var_105)
    var_107 = 'Alice'
    var_108 = 30
    var_109 = {var_100: var_107, var_101: var_108}
    var_110 = 'Bob'
    var_111 = 25
    var_112 = {var_100: var_110, var_101: var_111}
    var_113 = [var_109, var_112]
    var_114 = var_106.validate(var_113)
    var_115 = module_0.Integer()
    var_116 = {var_101: var_115}
    var_117 = module_0.Object(properties=var_116)
    var_118 = module_0.Array(var_117)
    var_119 = 'age'
    var_120 = 'not an integer'
    var_121 = {var_119: var_120}
    var_122 = [var_121]
    var_123 = var_118.validate(var_122)
    var_124 = module_0.Array()
    var_125 = 'string'
    var_126 = 3.14
    var_127 = [var_119, var_125, var_126, var_119]
    var_128 = var_124.validate(var_127)
    var_129 = module_0.Array(min_items=var_30)
    var_130 = [var_119, var_30]
    var_131 = var_129.validate(var_130)
    var_132 = module_0.Array(max_items=var_30)
    var_133 = [var_119, var_30]
    var_134 = var_132.validate(var_133)



# Parsed testcases at query #49
#--------------------------


import typesystem.fields as module_0

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
    var_10 = module_0.Union(var_9)
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = 0
    var_14 = exc_info.value.messages()[var_13]
    var_15 = var_14.code
    assert var_15 == 'null'
    var_16 = module_0.String()
    var_17 = module_0.Integer()
    var_18 = [var_16, var_17]
    var_19 = module_0.Union(var_18)
    var_20 = 'hello'
    var_21 = var_19.validate(var_20)
    assert var_21 == 'hello'
    var_22 = module_0.String()
    var_23 = module_0.Integer()
    var_24 = [var_22, var_23]
    var_25 = module_0.Union(var_24)
    var_26 = 42
    var_27 = var_25.validate(var_26)
    assert var_27 == 42
    var_28 = module_0.String()
    var_29 = module_0.Integer()
    var_30 = [var_28, var_29]
    var_31 = module_0.Union(var_30)
    var_32 = 1
    var_33 = 2
    var_34 = 3
    var_35 = [var_32, var_33, var_34]
    var_36 = var_31.validate(var_35)
    var_37 = exc_info.value.messages()[var_13]
    var_38 = var_37.code
    assert var_38 == 'union'
    var_39 = module_0.String()
    var_40 = module_0.Integer()
    var_41 = [var_39, var_40]
    var_42 = module_0.Union(var_41)
    var_43 = var_42.validate(var_36)
    assert var_43 is None
    var_44 = 10
    var_45 = module_0.Integer(minimum=var_44)
    var_46 = module_0.String()
    var_47 = [var_45, var_46]
    var_48 = module_0.Union(var_47)
    var_49 = 5
    var_50 = var_48.validate(var_49)
    var_51 = exc_info.value.messages()[var_13]
    var_52 = var_51.code
    assert var_52 == 'minimum'
    var_53 = 100
    var_54 = module_0.Integer(minimum=var_53)
    var_55 = 50
    var_56 = module_0.Integer(maximum=var_55)
    var_57 = [var_54, var_56]
    var_58 = module_0.Union(var_57)
    var_59 = 25
    var_60 = var_58.validate(var_59)
    assert var_60 == 25
    var_61 = 5
    var_62 = module_0.String(max_length=var_61)
    var_63 = module_0.String()
    var_64 = [var_62, var_63]
    var_65 = module_0.Union(var_64)
    var_66 = 'hi'
    var_67 = var_65.validate(var_66)
    assert var_67 == 'hi'
    var_68 = module_0.String()
    var_69 = module_0.Integer()
    var_70 = [var_68, var_69]
    var_71 = module_0.Union(var_70)
    var_72 = True
    var_73 = var_71.validate(var_72)
    var_74 = exc_info.value.messages()[var_13]
    var_75 = var_74.code
    assert var_75 == 'union'
    var_76 = module_0.Integer()
    var_77 = module_0.String()
    var_78 = [var_76, var_77]
    var_79 = module_0.Union(var_78)
    var_80 = 3.0
    var_81 = var_79.validate(var_80)
    assert var_81 == 3
    var_82 = module_0.Integer(minimum=var_55)
    var_83 = module_0.Integer(maximum=var_44)
    var_84 = [var_82, var_83]
    var_85 = module_0.Union(var_84)
    var_86 = 25
    var_87 = var_85.validate(var_86)
    var_88 = exc_info.value.messages()[var_13]
    var_89 = var_88.code
    assert var_89 == 'minimum'



# Parsed testcases at query #50
#--------------------------


import typesystem.fields as module_0

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
    var_8 = exc_info.value.messages()[var_4]
    var_9 = var_8.code
    assert var_9 == 'null'
    var_10 = module_0.Array()
    var_11 = 'not a list'
    var_12 = var_10.validate(var_11)
    var_13 = exc_info.value.messages()[var_4]
    var_14 = var_13.code
    assert var_14 == 'type'
    var_15 = module_0.Array(min_items=var_11)
    var_16 = []
    var_17 = var_15.validate(var_16)
    var_18 = exc_info.value.messages()[var_4]
    var_19 = var_18.code
    assert var_19 == 'empty'
    var_20 = 3
    var_21 = module_0.Array(min_items=var_20)
    var_22 = 1
    var_23 = 2
    var_24 = [var_22, var_23]
    var_25 = var_21.validate(var_24)
    var_26 = exc_info.value.messages()[var_25]
    var_27 = var_26.code
    assert var_27 == 'min_items'
    var_28 = 2
    var_29 = module_0.Array(max_items=var_28)
    var_30 = 1
    var_31 = 2
    var_32 = 3
    var_33 = [var_30, var_31, var_32]
    var_34 = var_29.validate(var_33)
    var_35 = exc_info.value.messages()[var_33]
    var_36 = var_35.code
    assert var_36 == 'max_items'
    var_37 = module_0.Array(exact_items=var_28)
    var_38 = 1
    var_39 = [var_38]
    var_40 = var_37.validate(var_39)
    var_41 = exc_info.value.messages()[var_33]
    var_42 = var_41.code
    assert var_42 == 'exact_items'
    var_43 = module_0.Array(exact_items=var_28)
    var_44 = [var_38, var_28]
    var_45 = var_43.validate(var_44)
    var_46 = module_0.Integer()
    var_47 = module_0.Array(var_46)
    var_48 = [var_38, var_28, var_20]
    var_49 = var_47.validate(var_48)
    var_50 = module_0.Integer()
    var_51 = module_0.Array(var_50)
    var_52 = 1
    var_53 = 'invalid'
    var_54 = 3
    var_55 = [var_52, var_53, var_54]
    var_56 = var_51.validate(var_55)
    var_57 = module_0.Integer()
    var_58 = module_0.String()
    var_59 = [var_57, var_58]
    var_60 = module_0.Array(var_59)
    var_61 = 'hello'
    var_62 = [var_52, var_61]
    var_63 = var_60.validate(var_62)
    var_64 = module_0.Integer()
    var_65 = [var_64]
    var_66 = module_0.String()
    var_67 = module_0.Array(var_65, var_66)
    var_68 = 'world'
    var_69 = [var_52, var_61, var_68]
    var_70 = var_67.validate(var_69)
    var_71 = module_0.Integer()
    var_72 = [var_71]
    var_73 = module_0.Array(var_72, var_55)
    var_74 = [var_52]
    var_75 = var_73.validate(var_74)
    var_76 = module_0.Array(unique_items=var_52)
    var_77 = 1
    var_78 = 2
    var_79 = [var_77, var_78, var_77]
    var_80 = var_76.validate(var_79)
    var_81 = 'unique_items'
    var_82 = module_0.Array(unique_items=var_77)
    var_83 = [var_77, var_28, var_20]
    var_84 = var_82.validate(var_83)
    var_85 = module_0.Integer()
    var_86 = module_0.Array(var_85)
    var_87 = 1
    var_88 = 'not_an_int'
    var_89 = 3
    var_90 = [var_87, var_88, var_89]
    var_91 = var_86.validate(var_90)
    var_92 = module_0.Integer()
    var_93 = module_0.String()
    var_94 = [var_92, var_93]
    var_95 = module_0.Array(var_94)
    var_96 = 'invalid_int'
    var_97 = 123
    var_98 = [var_96, var_97]
    var_99 = var_95.validate(var_98)
    var_100 = module_0.Array()
    var_101 = []
    var_102 = var_100.validate(var_101)
    var_103 = module_0.Array()
    var_104 = 'string'
    var_105 = 3.14
    var_106 = [var_96, var_104, var_105, var_97]
    var_107 = var_103.validate(var_106)
    var_108 = 4
    var_109 = module_0.Array(min_items=var_28, max_items=var_108)
    var_110 = [var_96, var_28]
    var_111 = var_109.validate(var_110)
    var_112 = [var_96, var_28, var_20, var_108]
    var_113 = var_109.validate(var_112)
    var_114 = 1
    var_115 = [var_114]
    var_116 = var_109.validate(var_115)
    var_117 = exc_info.value.messages()[var_99]
    var_118 = var_117.code
    assert var_118 == 'min_items'
    var_119 = 1
    var_120 = 2
    var_121 = 3
    var_122 = 4
    var_123 = 5
    var_124 = [var_119, var_120, var_121, var_122, var_123]
    var_125 = var_109.validate(var_124)
    var_126 = exc_info.value.messages()[var_122]
    var_127 = var_126.code
    assert var_127 == 'max_items'
    var_128 = module_0.Integer()
    var_129 = module_0.Array(var_128)
    var_130 = '1'
    var_131 = '2'
    var_132 = '3'
    var_133 = [var_130, var_131, var_132]
    var_134 = var_129.validate(var_133)
    var_135 = module_0.Array(unique_items=var_119)
    var_136 = 'a'
    var_137 = 'b'
    var_138 = [var_136, var_137, var_136]
    var_139 = var_135.validate(var_138)



# Parsed testcases at query #51
#--------------------------


import typesystem.fields as module_0

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
    var_11 = module_0.Number()
    var_12 = True
    var_13 = var_11.validate(var_12)
    var_14 = module_0.Number()
    var_15 = 3.14
    var_16 = var_14.validate(var_15)
    var_17 = module_0.Number(coerce_types=var_4)
    var_18 = 'not_a_number'
    var_19 = var_17.validate(var_18)
    var_20 = module_0.Number()
    var_21 = 42
    var_22 = var_20.validate(var_21)
    assert var_22 == 42
    var_23 = module_0.Number()
    var_24 = 3.14
    var_25 = var_23.validate(var_24)
    var_26 = module_0.Number()
    var_27 = '123'
    var_28 = var_26.validate(var_27)
    assert var_28 == 123
    var_29 = module_0.Number()
    var_30 = 'inf'
    var_31 = float(var_30)
    var_32 = var_29.validate(var_31)
    var_33 = module_0.Number()
    var_34 = 'nan'
    var_35 = float(var_34)
    var_36 = var_33.validate(var_35)
    var_37 = 10
    var_38 = module_0.Number(minimum=var_37)
    var_39 = 5
    var_40 = var_38.validate(var_39)
    var_41 = module_0.Number(minimum=var_37)
    var_42 = var_41.validate(var_37)
    assert var_42 == 10
    var_43 = module_0.Number(exclusive_minimum=var_37)
    var_44 = 10
    var_45 = var_43.validate(var_44)
    var_46 = module_0.Number(exclusive_minimum=var_37)
    var_47 = 11
    var_48 = var_46.validate(var_47)
    assert var_48 == 11
    var_49 = 100
    var_50 = module_0.Number(maximum=var_49)
    var_51 = 150
    var_52 = var_50.validate(var_51)
    var_53 = module_0.Number(maximum=var_49)
    var_54 = var_53.validate(var_49)
    assert var_54 == 100
    var_55 = module_0.Number(exclusive_maximum=var_49)
    var_56 = 100
    var_57 = var_55.validate(var_56)
    var_58 = module_0.Number(exclusive_maximum=var_49)
    var_59 = 99
    var_60 = var_58.validate(var_59)
    assert var_60 == 99
    var_61 = 5
    var_62 = module_0.Number(multiple_of=var_61)
    var_63 = 7
    var_64 = var_62.validate(var_63)
    var_65 = module_0.Number(multiple_of=var_61)
    var_66 = 15
    var_67 = var_65.validate(var_66)
    assert var_67 == 15
    var_68 = 0.5
    var_69 = module_0.Number(multiple_of=var_68)
    var_70 = 1.3
    var_71 = var_69.validate(var_70)
    var_72 = module_0.Number(multiple_of=var_68)
    var_73 = 1.5
    var_74 = var_72.validate(var_73)
    var_75 = '0.01'
    var_76 = module_0.Number(precision=var_75)
    var_77 = 3.146
    var_78 = var_76.validate(var_77)
    var_79 = module_0.Number()
    var_80 = 42.7
    var_81 = var_79.validate(var_80)
    assert var_81 == 42
    var_82 = module_0.Number()
    var_83 = var_82.validate(var_21)
    var_84 = module_0.Number()
    var_85 = '123.45'
    var_86 = module_0.Number(coerce_types=var_4)
    var_87 = 1
    var_88 = 2
    var_89 = 3
    var_90 = [var_87, var_88, var_89]
    var_91 = var_86.validate(var_90)



# Parsed testcases at query #52
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Object(properties=var_4)
    var_6 = 'John'
    var_7 = 30
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = var_5.validate(var_8)
    var_10 = module_0.Object()
    var_11 = None
    var_12 = var_10.validate(var_11)
    var_13 = True
    var_14 = module_0.Object()
    var_15 = None
    var_16 = var_14.validate(var_15)
    assert var_16 is None
    var_17 = module_0.Object()
    var_18 = 'not a dict'
    var_19 = var_17.validate(var_18)
    var_20 = module_0.Object()
    var_21 = 1
    var_22 = 'value'
    var_23 = {var_21: var_22}
    var_24 = var_20.validate(var_23)
    var_25 = 'invalid_key'
    var_26 = module_0.String()
    var_27 = {var_21: var_26}
    var_28 = [var_21]
    var_29 = module_0.Object(properties=var_27, required=var_28)
    var_30 = {}
    var_31 = var_29.validate(var_30)
    var_32 = 'required'
    var_33 = module_0.String()
    var_34 = {var_30: var_33}
    var_35 = [var_30]
    var_36 = module_0.Object(properties=var_34, required=var_35)
    var_37 = {var_30: var_6}
    var_38 = var_36.validate(var_37)
    var_39 = 'Unknown'
    var_40 = module_0.String()
    var_41 = {var_30: var_40}
    var_42 = module_0.Object(properties=var_41)
    var_43 = {}
    var_44 = var_42.validate(var_43)
    var_45 = 2
    var_46 = module_0.Object(min_properties=var_45)
    var_47 = 'a'
    var_48 = 1
    var_49 = {var_47: var_48}
    var_50 = var_46.validate(var_49)
    var_51 = module_0.Object(min_properties=var_13)
    var_52 = {}
    var_53 = var_51.validate(var_52)
    var_54 = module_0.Object(max_properties=var_13)
    var_55 = 'a'
    var_56 = 'b'
    var_57 = 1
    var_58 = 2
    var_59 = {var_55: var_57, var_56: var_58}
    var_60 = var_54.validate(var_59)
    var_61 = module_0.String()
    var_62 = {var_55: var_61}
    var_63 = module_0.Object(properties=var_62, additional_properties=var_13)
    var_64 = 'extra'
    var_65 = 'value'
    var_66 = {var_55: var_60, var_64: var_65}
    var_67 = var_63.validate(var_66)
    var_68 = module_0.String()
    var_69 = {var_55: var_68}
    var_70 = False
    var_71 = module_0.Object(properties=var_69, additional_properties=var_70)
    var_72 = 'name'
    var_73 = 'extra'
    var_74 = 'John'
    var_75 = 'value'
    var_76 = {var_72: var_74, var_73: var_75}
    var_77 = var_71.validate(var_76)
    var_78 = 'invalid_property'
    var_79 = module_0.String()
    var_80 = {var_72: var_79}
    var_81 = module_0.Integer()
    var_82 = module_0.Object(properties=var_80, additional_properties=var_81)
    var_83 = 'count'
    var_84 = 5
    var_85 = {var_72: var_77, var_83: var_84}
    var_86 = var_82.validate(var_85)
    var_87 = '^S_'
    var_88 = module_0.String()
    var_89 = {var_87: var_88}
    var_90 = module_0.Object(pattern_properties=var_89)
    var_91 = 'S_name'
    var_92 = {var_91: var_77}
    var_93 = var_90.validate(var_92)
    var_94 = '^[a-z]+$'
    var_95 = module_0.String(pattern=var_94)
    var_96 = module_0.Object(property_names=var_95)
    var_97 = 'Invalid'
    var_98 = 'value'
    var_99 = {var_97: var_98}
    var_100 = var_96.validate(var_99)
    var_101 = module_0.Integer()
    var_102 = {var_98: var_101}
    var_103 = module_0.Object(properties=var_102)
    var_104 = 'age'
    var_105 = 'not_a_number'
    var_106 = {var_104: var_105}
    var_107 = var_103.validate(var_106)
    var_108 = module_0.String()
    var_109 = {var_104: var_108}
    var_110 = module_0.Object(properties=var_109)
    var_111 = (var_104, var_77)
    var_112 = [var_111]
    var_113 = 'status'
    var_114 = module_0.String()
    var_115 = 'active'
    var_116 = module_0.String()
    var_117 = {var_104: var_114, var_113: var_116}
    var_118 = [var_104]
    var_119 = module_0.Object(properties=var_117, required=var_118)
    var_120 = {var_104: var_77}
    var_121 = var_119.validate(var_120)
    var_122 = 'user'
    var_123 = module_0.Integer()
    var_124 = {var_105: var_123}
    var_125 = module_0.Object(properties=var_124)
    var_126 = {var_122: var_125}
    var_127 = module_0.Object(properties=var_126)
    var_128 = 'user'
    var_129 = 'age'
    var_130 = 'invalid'
    var_131 = {var_129: var_130}
    var_132 = {var_128: var_131}
    var_133 = var_127.validate(var_132)



# Parsed testcases at query #53
#--------------------------


import typesystem.fields as module_0

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
    var_8 = exc_info.value.messages()[var_4]
    var_9 = var_8.code
    assert var_9 == 'null'
    var_10 = module_0.Array()
    var_11 = 'not a list'
    var_12 = var_10.validate(var_11)
    var_13 = exc_info.value.messages()[var_4]
    var_14 = var_13.code
    assert var_14 == 'type'
    var_15 = 2
    var_16 = module_0.Array(exact_items=var_15)
    var_17 = 1
    var_18 = [var_17]
    var_19 = var_16.validate(var_18)
    var_20 = exc_info.value.messages()[var_4]
    var_21 = var_20.code
    assert var_21 == 'exact_items'
    var_22 = 1
    var_23 = 2
    var_24 = 3
    var_25 = [var_22, var_23, var_24]
    var_26 = var_16.validate(var_25)
    var_27 = exc_info.value.messages()[var_25]
    var_28 = var_27.code
    assert var_28 == 'exact_items'
    var_29 = module_0.Array(min_items=var_15)
    var_30 = 1
    var_31 = [var_30]
    var_32 = var_29.validate(var_31)
    var_33 = exc_info.value.messages()[var_25]
    var_34 = var_33.code
    assert var_34 == 'min_items'
    var_35 = module_0.Array(min_items=var_30)
    var_36 = []
    var_37 = var_35.validate(var_36)
    var_38 = exc_info.value.messages()[var_25]
    var_39 = var_38.code
    assert var_39 == 'empty'
    var_40 = module_0.Array(max_items=var_15)
    var_41 = 1
    var_42 = 2
    var_43 = 3
    var_44 = [var_41, var_42, var_43]
    var_45 = var_40.validate(var_44)
    var_46 = exc_info.value.messages()[var_44]
    var_47 = var_46.code
    assert var_47 == 'max_items'
    var_48 = module_0.Integer()
    var_49 = module_0.Array(var_48)
    var_50 = 3
    var_51 = [var_41, var_15, var_50]
    var_52 = var_49.validate(var_51)
    var_53 = module_0.Integer()
    var_54 = module_0.Array(var_53)
    var_55 = 1
    var_56 = 'not an int'
    var_57 = 3
    var_58 = [var_55, var_56, var_57]
    var_59 = var_54.validate(var_58)
    var_60 = exc_info.value.messages()[var_58]
    var_61 = var_60.index
    var_62 = module_0.Integer()
    var_63 = module_0.String()
    var_64 = [var_62, var_63]
    var_65 = module_0.Array(var_64)
    var_66 = 'hello'
    var_67 = [var_55, var_66]
    var_68 = var_65.validate(var_67)
    var_69 = module_0.Integer()
    var_70 = module_0.String()
    var_71 = [var_69, var_70]
    var_72 = module_0.Array(var_71, var_58)
    var_73 = 1
    var_74 = 'hello'
    var_75 = 'extra'
    var_76 = [var_73, var_74, var_75]
    var_77 = var_72.validate(var_76)
    var_78 = exc_info.value.messages()[var_76]
    var_79 = var_78.code
    assert var_79 == 'additional_items'
    var_80 = module_0.Integer()
    var_81 = [var_80]
    var_82 = module_0.String()
    var_83 = module_0.Array(var_81, var_82)
    var_84 = 'world'
    var_85 = [var_73, var_66, var_84]
    var_86 = var_83.validate(var_85)
    var_87 = module_0.Integer()
    var_88 = module_0.Array(var_87, unique_items=var_73)
    var_89 = 1
    var_90 = 2
    var_91 = [var_89, var_90, var_89]
    var_92 = var_88.validate(var_91)
    var_93 = exc_info.value.messages()[var_92]
    var_94 = var_93.code
    assert var_94 == 'unique_items'
    var_95 = module_0.Integer()
    var_96 = module_0.Array(var_95, unique_items=var_89)
    var_97 = [var_89, var_15, var_50]
    var_98 = var_96.validate(var_97)
    var_99 = module_0.Array()
    var_100 = []
    var_101 = var_99.validate(var_100)
    var_102 = module_0.Array()
    var_103 = 'string'
    var_104 = 'key'
    var_105 = 'value'
    var_106 = {var_104: var_105}
    var_107 = [var_89, var_103, var_106]
    var_108 = var_102.validate(var_107)
    var_109 = module_0.Integer()
    var_110 = module_0.Array(var_109)
    var_111 = 'not int'
    var_112 = 'also not int'
    var_113 = [var_111, var_112]
    var_114 = var_110.validate(var_113)
    var_115 = module_0.Array(min_items=var_15)
    var_116 = [var_111, var_15, var_50]
    var_117 = var_115.validate(var_116)
    var_118 = module_0.Array(max_items=var_50)
    var_119 = [var_111, var_15]
    var_120 = var_118.validate(var_119)
    var_121 = module_0.Integer()
    var_122 = module_0.Array(var_121)
    var_123 = module_0.Array(var_122)
    var_124 = [var_111, var_15]
    var_125 = 4
    var_126 = [var_50, var_125]
    var_127 = [var_124, var_126]
    var_128 = var_123.validate(var_127)
    var_129 = module_0.Array(exact_items=var_50)
    var_130 = [var_111, var_15, var_50]
    var_131 = var_129.validate(var_130)



# Parsed testcases at query #54
#--------------------------


import typesystem.fields as module_0
import collections as module_1

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
    var_12 = {}
    var_13 = var_11.validate(var_12)
    var_14 = module_0.Object()
    var_15 = 1
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = var_14.validate(var_17)
    var_19 = 'name'
    var_20 = [var_19]
    var_21 = module_0.Object(required=var_20)
    var_22 = {}
    var_23 = var_21.validate(var_22)
    var_24 = 'required'
    var_25 = module_0.String()
    var_26 = {var_19: var_25}
    var_27 = module_0.Object(properties=var_26)
    var_28 = 'John'
    var_29 = {var_19: var_28}
    var_30 = var_27.validate(var_29)
    var_31 = module_0.String()
    var_32 = {var_19: var_31}
    var_33 = module_0.Object(properties=var_32)
    var_34 = {}
    var_35 = var_33.validate(var_34)
    var_36 = 'age'
    var_37 = module_0.Integer()
    var_38 = {var_36: var_37}
    var_39 = module_0.Object(properties=var_38)
    var_40 = 'age'
    var_41 = 'not an int'
    var_42 = {var_40: var_41}
    var_43 = var_39.validate(var_42)
    var_44 = 2
    var_45 = module_0.Object(min_properties=var_44)
    var_46 = 'a'
    var_47 = 1
    var_48 = {var_46: var_47}
    var_49 = var_45.validate(var_48)
    var_50 = module_0.Object(min_properties=var_46)
    var_51 = {}
    var_52 = var_50.validate(var_51)
    var_53 = module_0.Object(max_properties=var_51)
    var_54 = 'a'
    var_55 = 'b'
    var_56 = 1
    var_57 = 2
    var_58 = {var_54: var_56, var_55: var_57}
    var_59 = var_53.validate(var_58)
    var_60 = module_0.String()
    var_61 = {var_19: var_60}
    var_62 = module_0.Object(properties=var_61)
    var_63 = 'extra'
    var_64 = 'field'
    var_65 = {var_19: var_28, var_63: var_64}
    var_66 = var_62.validate(var_65)
    var_67 = module_0.String()
    var_68 = {var_19: var_67}
    var_69 = module_0.Object(properties=var_68, additional_properties=var_57)
    var_70 = 'name'
    var_71 = 'extra'
    var_72 = 'John'
    var_73 = 'field'
    var_74 = {var_70: var_72, var_71: var_73}
    var_75 = var_69.validate(var_74)
    var_76 = 'invalid_property'
    var_77 = module_0.String()
    var_78 = {var_19: var_77}
    var_79 = module_0.Integer()
    var_80 = module_0.Object(properties=var_78, additional_properties=var_79)
    var_81 = 30
    var_82 = {var_19: var_28, var_36: var_81}
    var_83 = var_80.validate(var_82)
    var_84 = '^num_'
    var_85 = module_0.Integer()
    var_86 = {var_84: var_85}
    var_87 = module_0.Object(pattern_properties=var_86)
    var_88 = 'num_1'
    var_89 = 'num_2'
    var_90 = 10
    var_91 = 20
    var_92 = {var_88: var_90, var_89: var_91}
    var_93 = var_87.validate(var_92)
    var_94 = module_0.Integer()
    var_95 = {var_84: var_94}
    var_96 = module_0.Object(pattern_properties=var_95)
    var_97 = 'num_1'
    var_98 = 'not an int'
    var_99 = {var_97: var_98}
    var_100 = var_96.validate(var_99)
    var_101 = '^[a-z]+$'
    var_102 = module_0.String(pattern=var_101)
    var_103 = module_0.Object(property_names=var_102)
    var_104 = 'Name'
    var_105 = 'value'
    var_106 = {var_104: var_105}
    var_107 = var_103.validate(var_106)
    var_108 = module_0.String(pattern=var_101)
    var_109 = module_0.Object(property_names=var_108)
    var_110 = 'value'
    var_111 = {var_19: var_110}
    var_112 = var_109.validate(var_111)
    var_113 = 'user'
    var_114 = module_0.String()
    var_115 = module_0.Integer()
    var_116 = {var_19: var_114, var_36: var_115}
    var_117 = [var_19]
    var_118 = module_0.Object(properties=var_116, required=var_117)
    var_119 = {var_113: var_118}
    var_120 = module_0.Object(properties=var_119)
    var_121 = {var_19: var_28, var_36: var_81}
    var_122 = {var_113: var_121}
    var_123 = var_120.validate(var_122)
    var_124 = 'key'
    var_125 = {var_124: var_110}
    var_126 = module_1.UserDict(var_125)
    var_127 = module_0.Object()
    var_128 = var_127.validate(var_126)



# Parsed testcases at query #55
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = [var_0, var_1]
    var_3 = module_0.Union(var_2)
    var_4 = True
    var_5 = module_0.String()
    var_6 = module_0.Integer()
    var_7 = [var_5, var_6]
    var_8 = module_0.Union(var_7)
    var_9 = module_0.String()
    var_10 = module_0.Integer()
    var_11 = [var_9, var_10]
    var_12 = module_0.Union(var_11)
    var_13 = module_0.String()
    var_14 = module_0.Integer()
    var_15 = module_0.Boolean()
    var_16 = [var_13, var_14, var_15]
    var_17 = module_0.Union(var_16)
    var_18 = module_0.String()
    var_19 = module_0.Integer()
    var_20 = [var_18, var_19]
    var_21 = module_0.Union(var_20)
    var_22 = module_0.String()
    var_23 = [var_22]
    var_24 = module_0.Union(var_23)
    var_25 = var_24.any_of
    var_26 = len(var_25)
    assert var_26 == 1
    var_27 = module_0.String()
    var_28 = module_0.Integer()
    var_29 = module_0.Boolean()
    var_30 = module_0.Float()
    var_31 = module_0.Array()
    var_32 = [var_27, var_28, var_29, var_30, var_31]
    var_33 = module_0.Union(var_32)
    var_34 = var_33.any_of
    var_35 = len(var_34)
    assert var_35 == 5
    var_36 = []
    var_37 = module_0.Union(var_36)
    var_38 = False
    var_39 = module_0.String()
    var_40 = module_0.Integer()
    var_41 = module_0.Boolean()
    var_42 = [var_39, var_40, var_41]
    var_43 = module_0.Union(var_42)
    var_44 = module_0.String()
    var_45 = module_0.Integer()
    var_46 = module_0.Boolean()
    var_47 = [var_44, var_45, var_46]
    var_48 = module_0.Union(var_47)



