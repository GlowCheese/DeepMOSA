####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.base as module_0


def test_case_0():
    var_0 = 'test_value'
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = iter(var_1)
    var_3 = next(var_2)
    assert var_3 == 'test_value'
    var_4 = next(var_2)
    assert var_4 is None
    var_5 = 'test_error'
    var_6 = module_0.ValidationError(text=var_5)
    var_7 = module_0.ValidationResult(error=var_6)
    var_8 = iter(var_7)
    var_9 = next(var_8)
    assert var_9 is None
    var_10 = next(var_8)
    var_11 = 'All test cases passed'
    var_12 = print(var_11)



# Parsed testcases at query #2
#--------------------------



def test_case_0():
    var_0 = 'Invalid input'
    var_1 = 'invalid'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = repr(var_2)
    assert var_3 == "BaseError(text='Invalid input', code='invalid')"
    var_4 = 'username'
    var_5 = module_0.BaseError(text=var_0, code=var_1, key=var_4)
    var_6 = repr(var_5)
    assert var_6 == "BaseError(text='Invalid input', code='invalid')"
    var_7 = [var_4]
    var_8 = module_0.Message(text=var_0, code=var_1, index=var_7)
    var_9 = 'Missing field'
    var_10 = 'required'
    var_11 = 'email'
    var_12 = [var_11]
    var_13 = module_0.Message(text=var_9, code=var_10, index=var_12)
    var_14 = [var_8, var_13]
    var_15 = module_0.BaseError(messages=var_14)
    var_16 = repr(var_15)
    assert var_16 == "BaseError([Message(text='Invalid input', code='invalid', index=['username']), Message(text='Missing field', code='required', index=['email'])])"
    var_17 = 1
    var_18 = 5
    var_19 = 10
    var_20 = module_0.Position(var_17, var_18, var_19)
    var_21 = module_0.BaseError(text=var_0, code=var_1, position=var_20)
    var_22 = repr(var_21)
    assert var_22 == "BaseError(text='Invalid input', code='invalid')"
    var_23 = module_0.Position(var_17, var_18, var_19)
    var_24 = 15
    var_25 = module_0.Position(var_17, var_19, var_24)
    var_26 = module_0.BaseError(text=var_0, code=var_1)
    var_27 = repr(var_26)
    assert var_27 == "BaseError(text='Invalid input', code='invalid')"
    var_28 = ''
    var_29 = module_0.BaseError(text=var_28, code=var_28)
    var_30 = repr(var_29)
    assert var_30 == "BaseError(text='', code='')"
    var_31 = 'Invalid input: \n\t'
    var_32 = module_0.BaseError(text=var_31, code=var_1)
    var_33 = repr(var_32)
    assert var_33 == "BaseError(text='Invalid input: \\n\\t', code='invalid')"
    var_34 = 'Invalid input: 🚀'
    var_35 = module_0.BaseError(text=var_34, code=var_1)
    var_36 = repr(var_35)
    assert var_36 == "BaseError(text='Invalid input: 🚀', code='invalid')"
    var_37 = 'A'
    var_38 = 1000
    var_39 = var_37 * var_38
    var_40 = module_0.BaseError(text=var_39, code=var_1)
    var_41 = repr(var_40)
    var_42 = None
    var_43 = module_0.BaseError(text=var_0, code=var_42)
    var_44 = repr(var_43)
    assert var_44 == "BaseError(text='Invalid input', code='custom')"
    var_45 = module_0.BaseError(text=var_0, code=var_28)
    var_46 = repr(var_45)
    assert var_46 == "BaseError(text='Invalid input', code='')"
    var_47 = module_0.BaseError(text=var_42, code=var_1)
    var_48 = repr(var_47)
    assert var_48 == "BaseError(text=None, code='invalid')"
    var_49 = module_0.BaseError(text=var_28, code=var_1)
    var_50 = repr(var_49)
    assert var_50 == "BaseError(text='', code='invalid')"
    var_51 = module_0.BaseError(text=var_0, code=var_1, key=var_42)
    var_52 = repr(var_51)
    assert var_52 == "BaseError(text='Invalid input', code='invalid')"
    var_53 = module_0.BaseError(text=var_0, code=var_1, key=var_28)
    var_54 = repr(var_53)
    assert var_54 == "BaseError(text='Invalid input', code='invalid')"
    var_55 = 0
    var_56 = module_0.BaseError(text=var_0, code=var_1, key=var_55)
    var_57 = repr(var_56)
    assert var_57 == "BaseError(text='Invalid input', code='invalid')"
    var_58 = -1
    var_59 = module_0.BaseError(text=var_0, code=var_1, key=var_58)
    var_60 = repr(var_59)
    assert var_60 == "BaseError(text='Invalid input', code='invalid')"
    var_61 = 1000000
    var_62 = module_0.BaseError(text=var_0, code=var_1, key=var_61)
    var_63 = repr(var_62)
    assert var_63 == "BaseError(text='Invalid input', code='invalid')"
    var_64 = 3.14
    var_65 = module_0.BaseError(text=var_0, code=var_1, key=var_64)
    var_66 = repr(var_65)
    assert var_66 == "BaseError(text='Invalid input', code='invalid')"
    var_67 = True
    var_68 = module_0.BaseError(text=var_0, code=var_1, key=var_67)
    var_69 = repr(var_68)
    assert var_69 == "BaseError(text='Invalid input', code='invalid')"
    var_70 = 2
    var_71 = 3
    var_72 = [var_67, var_70, var_71]
    var_73 = module_0.BaseError(text=var_0, code=var_1, key=var_72)
    var_74 = repr(var_73)
    assert var_74 == "BaseError(text='Invalid input', code='invalid')"
    var_75 = 'a'
    var_76 = {var_75: var_67}
    var_77 = module_0.BaseError(text=var_0, code=var_1, key=var_76)
    var_78 = repr(var_77)
    assert var_78 == "BaseError(text='Invalid input', code='invalid')"
    var_79 = (var_67, var_70, var_71)
    var_80 = module_0.BaseError(text=var_0, code=var_1, key=var_79)
    var_81 = repr(var_80)
    assert var_81 == "BaseError(text='Invalid input', code='invalid')"
    var_82 = {var_67, var_70, var_71}
    var_83 = module_0.BaseError(text=var_0, code=var_1, key=var_82)
    var_84 = repr(var_83)
    assert var_84 == "BaseError(text='Invalid input', code='invalid')"
    var_85 = module_0.BaseError(text=var_0, code=var_1, position=var_42)
    var_86 = repr(var_85)
    assert var_86 == "BaseError(text='Invalid input', code='invalid')"
    var_87 = module_0.Position(var_55, var_55, var_55)
    var_88 = module_0.BaseError(text=var_0, code=var_1, position=var_87)
    var_89 = repr(var_88)
    assert var_89 == "BaseError(text='Invalid input', code='invalid')"
    var_90 = -1
    var_91 = -1
    var_92 = -1
    var_93 = module_0.Position(var_90, var_91, var_92)
    var_94 = module_0.BaseError(text=var_0, code=var_1, position=var_93)
    var_95 = repr(var_94)
    assert var_95 == "BaseError(text='Invalid input', code='invalid')"
    var_96 = module_0.Position(var_61, var_61, var_61)
    var_97 = module_0.BaseError(text=var_0, code=var_1, position=var_96)
    var_98 = repr(var_97)
    assert var_98 == "BaseError(text='Invalid input', code='invalid')"
    var_99 = 1.5
    var_100 = 2.5
    var_101 = 3.5
    var_102 = module_0.Position(var_99, var_100, var_101)
    var_103 = module_0.BaseError(text=var_0, code=var_1, position=var_102)
    var_104 = repr(var_103)
    assert var_104 == "BaseError(text='Invalid input', code='invalid')"
    var_105 = True
    var_106 = False
    var_107 = True
    var_108 = module_0.Position(var_105, var_106, var_107)
    var_109 = module_0.BaseError(text=var_0, code=var_1, position=var_108)
    var_110 = repr(var_109)
    assert var_110 == "BaseError(text='Invalid input', code='invalid')"
    var_111 = '1'
    var_112 = '2'
    var_113 = '3'
    var_114 = module_0.Position(var_111, var_112, var_113)
    var_115 = module_0.BaseError(text=var_0, code=var_1, position=var_114)
    var_116 = repr(var_115)
    assert var_116 == "BaseError(text='Invalid input', code='invalid')"
    var_117 = [var_107]
    var_118 = [var_70]
    var_119 = [var_71]
    var_120 = module_0.Position(var_117, var_118, var_119)
    var_121 = module_0.BaseError(text=var_0, code=var_1, position=var_120)
    var_122 = repr(var_121)
    assert var_122 == "BaseError(text='Invalid input', code='invalid')"
    var_123 = 'line'
    var_124 = {var_123: var_107}
    var_125 = 'column'
    var_126 = {var_125: var_70}
    var_127 = 'char'
    var_128 = {var_127: var_71}
    var_129 = module_0.Position(var_124, var_126, var_128)
    var_130 = module_0.BaseError(text=var_0, code=var_1, position=var_129)
    var_131 = repr(var_130)
    assert var_131 == "BaseError(text='Invalid input', code='invalid')"
    var_132 = (var_107,)
    var_133 = (var_70,)
    var_134 = (var_71,)
    var_135 = module_0.Position(var_132, var_133, var_134)
    var_136 = module_0.BaseError(text=var_0, code=var_1, position=var_135)
    var_137 = repr(var_136)
    assert var_137 == "BaseError(text='Invalid input', code='invalid')"
    var_138 = {var_107}
    var_139 = {var_70}
    var_140 = {var_71}
    var_141 = module_0.Position(var_138, var_139, var_140)
    var_142 = module_0.BaseError(text=var_0, code=var_1, position=var_141)
    var_143 = repr(var_142)
    assert var_143 == "BaseError(text='Invalid input', code='invalid')"



# Parsed testcases at query #3
#--------------------------



def test_case_0():
    var_0 = 'Error message'
    var_1 = module_0.Message(text=var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_0.Position(var_2, var_2, var_3)
    var_5 = 'max_length'
    var_6 = 'username'
    var_7 = module_0.Message(text=var_0, code=var_5, key=var_6, position=var_4)
    var_8 = 'users'
    var_9 = 3
    var_10 = [var_8, var_9, var_6]
    var_11 = module_0.Message(text=var_0, index=var_10)
    var_12 = module_0.Position(var_2, var_2, var_3)
    var_13 = 10
    var_14 = 9
    var_15 = module_0.Position(var_2, var_13, var_14)
    var_16 = module_0.Message(text=var_0, start_position=var_12, end_position=var_15)
    var_17 = 'Error message'
    var_18 = 'username'
    var_19 = 'users'
    var_20 = 3
    var_21 = [var_19, var_20, var_18]
    var_22 = module_0.Message(text=var_17, key=var_18, index=var_21)
    var_23 = 'Error message'
    var_24 = module_0.Message(text=var_23, position=var_4, start_position=var_12)
    var_25 = 'Error message'
    var_26 = module_0.Message(text=var_25, position=var_4, end_position=var_15)



# Parsed testcases at query #4
#--------------------------



def test_case_0():
    var_0 = 42
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = 'Invalid data'
    var_3 = module_0.ValidationError(text=var_2)
    var_4 = module_0.ValidationResult(error=var_3)



# Parsed testcases at query #5
#--------------------------



def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'key1'
    var_3 = 'index1'
    var_4 = [var_3]
    var_5 = 1
    var_6 = module_0.Position(var_5, var_5, var_5)
    var_7 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_4, position=var_6)
    var_8 = [var_3]
    var_9 = module_0.Position(var_5, var_5, var_5)
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_8, position=var_9)
    var_11 = 'Error1'
    var_12 = [var_3]
    var_13 = module_0.Position(var_5, var_5, var_5)
    var_14 = module_0.Message(text=var_11, code=var_1, key=var_2, index=var_12, position=var_13)
    var_15 = 'Error2'
    var_16 = [var_3]
    var_17 = module_0.Position(var_5, var_5, var_5)
    var_18 = module_0.Message(text=var_15, code=var_1, key=var_2, index=var_16, position=var_17)
    var_19 = 'code1'
    var_20 = [var_3]
    var_21 = module_0.Position(var_5, var_5, var_5)
    var_22 = module_0.Message(text=var_0, code=var_19, key=var_2, index=var_20, position=var_21)
    var_23 = 'code2'
    var_24 = [var_3]
    var_25 = module_0.Position(var_5, var_5, var_5)
    var_26 = module_0.Message(text=var_0, code=var_23, key=var_2, index=var_24, position=var_25)
    var_27 = [var_3]
    var_28 = module_0.Position(var_5, var_5, var_5)
    var_29 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_27, position=var_28)
    var_30 = 'index2'
    var_31 = [var_30]
    var_32 = module_0.Position(var_5, var_5, var_5)
    var_33 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_31, position=var_32)
    var_34 = [var_3]
    var_35 = module_0.Position(var_5, var_5, var_5)
    var_36 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_34, position=var_35)
    var_37 = [var_3]
    var_38 = 2
    var_39 = module_0.Position(var_38, var_38, var_38)
    var_40 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_37, position=var_39)
    var_41 = [var_3]
    var_42 = module_0.Position(var_5, var_5, var_5)
    var_43 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_41, position=var_42)
    var_44 = [var_3]
    var_45 = module_0.Position(var_5, var_5, var_5)
    var_46 = 10
    var_47 = module_0.Position(var_5, var_46, var_46)
    var_48 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_44, start_position=var_45, end_position=var_47)
    var_49 = [var_3]
    var_50 = module_0.Position(var_5, var_5, var_5)
    var_51 = module_0.Position(var_5, var_46, var_46)
    var_52 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_49, start_position=var_50, end_position=var_51)
    var_53 = [var_3]
    var_54 = module_0.Position(var_5, var_5, var_5)
    var_55 = module_0.Position(var_5, var_46, var_46)
    var_56 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_53, start_position=var_54, end_position=var_55)
    var_57 = [var_3]
    var_58 = module_0.Position(var_38, var_5, var_5)
    var_59 = module_0.Position(var_5, var_46, var_46)
    var_60 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_57, start_position=var_58, end_position=var_59)
    var_61 = [var_3]
    var_62 = module_0.Position(var_5, var_5, var_5)
    var_63 = module_0.Position(var_5, var_46, var_46)
    var_64 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_61, start_position=var_62, end_position=var_63)
    var_65 = [var_3]
    var_66 = module_0.Position(var_5, var_5, var_5)
    var_67 = module_0.Position(var_38, var_46, var_46)
    var_68 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_65, start_position=var_66, end_position=var_67)
    var_69 = [var_3]
    var_70 = module_0.Position(var_5, var_5, var_5)
    var_71 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_69, position=var_70)
    var_72 = [var_3]
    var_73 = module_0.Position(var_5, var_5, var_5)
    var_74 = module_0.Position(var_5, var_5, var_5)
    var_75 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_72, start_position=var_73, end_position=var_74)
    var_76 = [var_3]
    var_77 = module_0.Position(var_5, var_5, var_5)
    var_78 = module_0.Position(var_5, var_46, var_46)
    var_79 = module_0.Message(text=var_11, code=var_1, key=var_2, index=var_76, start_position=var_77, end_position=var_78)
    var_80 = [var_3]
    var_81 = module_0.Position(var_5, var_5, var_5)
    var_82 = module_0.Position(var_5, var_46, var_46)
    var_83 = module_0.Message(text=var_15, code=var_1, key=var_2, index=var_80, start_position=var_81, end_position=var_82)
    var_84 = [var_3]
    var_85 = module_0.Position(var_5, var_5, var_5)
    var_86 = module_0.Position(var_5, var_46, var_46)
    var_87 = module_0.Message(text=var_0, code=var_19, key=var_2, index=var_84, start_position=var_85, end_position=var_86)
    var_88 = [var_3]
    var_89 = module_0.Position(var_5, var_5, var_5)
    var_90 = module_0.Position(var_5, var_46, var_46)
    var_91 = module_0.Message(text=var_0, code=var_23, key=var_2, index=var_88, start_position=var_89, end_position=var_90)
    var_92 = [var_3]
    var_93 = module_0.Position(var_5, var_5, var_5)
    var_94 = module_0.Position(var_5, var_46, var_46)
    var_95 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_92, start_position=var_93, end_position=var_94)
    var_96 = [var_30]
    var_97 = module_0.Position(var_5, var_5, var_5)
    var_98 = module_0.Position(var_5, var_46, var_46)
    var_99 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_96, start_position=var_97, end_position=var_98)
    var_100 = None
    var_101 = module_0.Position(var_5, var_5, var_5)
    var_102 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_100, position=var_101)
    var_103 = [var_2]
    var_104 = module_0.Position(var_5, var_5, var_5)
    var_105 = module_0.Message(text=var_0, code=var_1, key=var_100, index=var_103, position=var_104)
    var_106 = module_0.Position(var_5, var_5, var_5)
    var_107 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_100, position=var_106)
    var_108 = 'key2'
    var_109 = [var_108]
    var_110 = module_0.Position(var_5, var_5, var_5)
    var_111 = module_0.Message(text=var_0, code=var_1, key=var_100, index=var_109, position=var_110)
    var_112 = [var_3]
    var_113 = module_0.Position(var_5, var_5, var_5)
    var_114 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_112, position=var_113)
    var_115 = [var_3]
    var_116 = module_0.Position(var_5, var_5, var_5)
    var_117 = module_0.Position(var_5, var_5, var_5)
    var_118 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_115, start_position=var_116, end_position=var_117)
    var_119 = []
    var_120 = module_0.Position(var_5, var_5, var_5)
    var_121 = module_0.Message(text=var_0, code=var_1, key=var_100, index=var_119, position=var_120)
    var_122 = []
    var_123 = module_0.Position(var_5, var_5, var_5)
    var_124 = module_0.Message(text=var_0, code=var_1, key=var_100, index=var_122, position=var_123)
    var_125 = []
    var_126 = module_0.Position(var_5, var_5, var_5)
    var_127 = module_0.Message(text=var_11, code=var_1, key=var_100, index=var_125, position=var_126)
    var_128 = []
    var_129 = module_0.Position(var_5, var_5, var_5)
    var_130 = module_0.Message(text=var_15, code=var_1, key=var_100, index=var_128, position=var_129)



# Parsed testcases at query #6
#--------------------------



def test_case_0():
    var_0 = 'Error message'
    var_1 = module_0.Message(text=var_0)
    var_2 = 1
    var_3 = 0
    var_4 = module_0.Position(var_2, var_2, var_3)
    var_5 = 'max_length'
    var_6 = 'username'
    var_7 = module_0.Message(text=var_0, code=var_5, key=var_6, position=var_4)
    var_8 = module_0.Position(var_2, var_2, var_3)
    var_9 = 10
    var_10 = 9
    var_11 = module_0.Position(var_2, var_9, var_10)
    var_12 = module_0.Message(text=var_0, start_position=var_8, end_position=var_11)
    var_13 = 'users'
    var_14 = 3
    var_15 = [var_13, var_14, var_6]
    var_16 = module_0.Message(text=var_0, index=var_15)
    var_17 = 'Error message'
    var_18 = 'username'
    var_19 = 'users'
    var_20 = 3
    var_21 = [var_19, var_20, var_18]
    var_22 = module_0.Message(text=var_17, key=var_18, index=var_21)
    var_23 = 'Error message'
    var_24 = module_0.Message(text=var_23, position=var_4, start_position=var_8)
    var_25 = 'Error message'
    var_26 = module_0.Message(text=var_25, position=var_4, end_position=var_11)
    var_27 = module_0.Message(text=var_25, code=var_20, key=var_21)
    var_28 = module_0.Message(text=var_25, code=var_20, key=var_21)
    var_29 = 'Different message'
    var_30 = module_0.Message(text=var_29, code=var_20, key=var_21)
    var_31 = hash(var_27)
    var_32 = hash(var_28)
    var_33 = repr(var_27)
    var_34 = module_0.Message(text=var_25, position=var_4)
    var_35 = repr(var_34)
    var_36 = module_0.Message(text=var_25, start_position=var_8, end_position=var_11)
    var_37 = repr(var_36)



# Parsed testcases at query #7
#--------------------------



def test_case_0():
    var_0 = 'Test message'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'test_code'
    var_3 = module_0.Message(text=var_0, code=var_2)
    var_4 = 'username'
    var_5 = module_0.Message(text=var_0, key=var_4)
    var_6 = 'users'
    var_7 = 3
    var_8 = [var_6, var_7, var_4]
    var_9 = module_0.Message(text=var_0, index=var_8)
    var_10 = 1
    var_11 = 0
    var_12 = module_0.Position(var_10, var_10, var_11)
    var_13 = module_0.Message(text=var_0, position=var_12)
    var_14 = module_0.Position(var_10, var_10, var_11)
    var_15 = 5
    var_16 = 4
    var_17 = module_0.Position(var_10, var_15, var_16)
    var_18 = module_0.Message(text=var_0, start_position=var_14, end_position=var_17)
    var_19 = 'Test message'
    var_20 = 'username'
    var_21 = 'users'
    var_22 = 3
    var_23 = [var_21, var_22, var_20]
    var_24 = module_0.Message(text=var_19, key=var_20, index=var_23)
    var_25 = module_0.Position(var_10, var_10, var_11)
    var_26 = 'Test message'
    var_27 = module_0.Message(text=var_26, position=var_25, start_position=var_25)
    var_28 = 'Test message'
    var_29 = module_0.Message(text=var_28, position=var_25, end_position=var_25)



# Parsed testcases at query #8
#--------------------------



def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Message(text=var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = module_0.Position(var_2, var_3, var_4)
    var_6 = 'max_length'
    var_7 = 'username'
    var_8 = module_0.Message(text=var_0, code=var_6, key=var_7, position=var_5)
    var_9 = 'users'
    var_10 = [var_9, var_4, var_7]
    var_11 = module_0.Message(text=var_0, code=var_6, index=var_10)
    var_12 = module_0.Position(var_2, var_3, var_4)
    var_13 = 5
    var_14 = 6
    var_15 = module_0.Position(var_2, var_13, var_14)
    var_16 = module_0.Message(text=var_0, start_position=var_12, end_position=var_15)
    var_17 = 'test'
    var_18 = 'username'
    var_19 = 'users'
    var_20 = 3
    var_21 = [var_19, var_20, var_18]
    var_22 = module_0.Message(text=var_17, key=var_18, index=var_21)
    var_23 = 'test'
    var_24 = module_0.Message(text=var_23, position=var_5, start_position=var_12)
    var_25 = 'test'
    var_26 = module_0.Message(text=var_25, position=var_5, end_position=var_15)



# Parsed testcases at query #9
#--------------------------



def test_case_0():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = 'key'
    var_3 = 'index'
    var_4 = [var_3]
    var_5 = 1
    var_6 = module_0.Position(var_5, var_5, var_5)
    var_7 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_4, position=var_6)
    var_8 = [var_3]
    var_9 = module_0.Position(var_5, var_5, var_5)
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_8, position=var_9)
    var_11 = [var_3]
    var_12 = module_0.Position(var_5, var_5, var_5)
    var_13 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_11, position=var_12)
    var_14 = 'Different Error'
    var_15 = [var_3]
    var_16 = module_0.Position(var_5, var_5, var_5)
    var_17 = module_0.Message(text=var_14, code=var_1, key=var_2, index=var_15, position=var_16)
    var_18 = [var_3]
    var_19 = module_0.Position(var_5, var_5, var_5)
    var_20 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_18, position=var_19)
    var_21 = 'different'
    var_22 = [var_3]
    var_23 = module_0.Position(var_5, var_5, var_5)
    var_24 = module_0.Message(text=var_0, code=var_21, key=var_2, index=var_22, position=var_23)
    var_25 = [var_3]
    var_26 = module_0.Position(var_5, var_5, var_5)
    var_27 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_25, position=var_26)
    var_28 = [var_21]
    var_29 = module_0.Position(var_5, var_5, var_5)
    var_30 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_28, position=var_29)
    var_31 = [var_3]
    var_32 = module_0.Position(var_5, var_5, var_5)
    var_33 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_31, position=var_32)
    var_34 = [var_3]
    var_35 = 2
    var_36 = module_0.Position(var_35, var_35, var_35)
    var_37 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_34, position=var_36)
    var_38 = [var_3]
    var_39 = module_0.Position(var_5, var_5, var_5)
    var_40 = module_0.Position(var_5, var_5, var_5)
    var_41 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_38, start_position=var_39, end_position=var_40)
    var_42 = [var_3]
    var_43 = module_0.Position(var_35, var_35, var_35)
    var_44 = module_0.Position(var_35, var_35, var_35)
    var_45 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_42, start_position=var_43, end_position=var_44)
    var_46 = [var_3]
    var_47 = module_0.Position(var_5, var_5, var_5)
    var_48 = module_0.Position(var_5, var_5, var_5)
    var_49 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_46, start_position=var_47, end_position=var_48)
    var_50 = [var_3]
    var_51 = module_0.Position(var_5, var_5, var_5)
    var_52 = module_0.Position(var_5, var_5, var_5)
    var_53 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_50, start_position=var_51, end_position=var_52)
    var_54 = [var_3]
    var_55 = module_0.Position(var_5, var_5, var_5)
    var_56 = module_0.Position(var_5, var_5, var_5)
    var_57 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_54, start_position=var_55, end_position=var_56)
    var_58 = [var_3]
    var_59 = module_0.Position(var_5, var_5, var_5)
    var_60 = module_0.Position(var_35, var_35, var_35)
    var_61 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_58, start_position=var_59, end_position=var_60)
    var_62 = [var_3]
    var_63 = module_0.Position(var_5, var_5, var_5)
    var_64 = module_0.Position(var_5, var_5, var_5)
    var_65 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_62, start_position=var_63, end_position=var_64)
    var_66 = [var_3]
    var_67 = module_0.Position(var_35, var_35, var_35)
    var_68 = module_0.Position(var_5, var_5, var_5)
    var_69 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_66, start_position=var_67, end_position=var_68)
    var_70 = [var_3]
    var_71 = module_0.Position(var_5, var_5, var_5)
    var_72 = module_0.Position(var_5, var_5, var_5)
    var_73 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_70, start_position=var_71, end_position=var_72)
    var_74 = [var_3]
    var_75 = module_0.Position(var_5, var_5, var_5)
    var_76 = module_0.Position(var_5, var_5, var_5)
    var_77 = module_0.Message(text=var_14, code=var_1, key=var_2, index=var_74, start_position=var_75, end_position=var_76)
    var_78 = [var_3]
    var_79 = module_0.Position(var_5, var_5, var_5)
    var_80 = module_0.Position(var_5, var_5, var_5)
    var_81 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_78, start_position=var_79, end_position=var_80)
    var_82 = [var_3]
    var_83 = module_0.Position(var_5, var_5, var_5)
    var_84 = module_0.Position(var_5, var_5, var_5)
    var_85 = module_0.Message(text=var_0, code=var_21, key=var_2, index=var_82, start_position=var_83, end_position=var_84)
    var_86 = [var_3]
    var_87 = module_0.Position(var_5, var_5, var_5)
    var_88 = module_0.Position(var_5, var_5, var_5)
    var_89 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_86, start_position=var_87, end_position=var_88)
    var_90 = [var_21]
    var_91 = module_0.Position(var_5, var_5, var_5)
    var_92 = module_0.Position(var_5, var_5, var_5)
    var_93 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_90, start_position=var_91, end_position=var_92)
    var_94 = [var_3]
    var_95 = module_0.Position(var_5, var_5, var_5)
    var_96 = module_0.Position(var_5, var_5, var_5)
    var_97 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_94, start_position=var_95, end_position=var_96)
    var_98 = [var_3]
    var_99 = module_0.Position(var_5, var_5, var_5)
    var_100 = module_0.Position(var_5, var_5, var_5)
    var_101 = module_0.Message(text=var_0, code=var_1, key=var_21, index=var_98, start_position=var_99, end_position=var_100)
    var_102 = [var_3]
    var_103 = module_0.Position(var_5, var_5, var_5)
    var_104 = module_0.Position(var_5, var_5, var_5)
    var_105 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_102, start_position=var_103, end_position=var_104)
    var_106 = [var_21]
    var_107 = module_0.Position(var_5, var_5, var_5)
    var_108 = module_0.Position(var_5, var_5, var_5)
    var_109 = module_0.Message(text=var_0, code=var_1, key=var_21, index=var_106, start_position=var_107, end_position=var_108)
    var_110 = [var_3]
    var_111 = module_0.Position(var_5, var_5, var_5)
    var_112 = module_0.Position(var_5, var_5, var_5)
    var_113 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_110, start_position=var_111, end_position=var_112)
    var_114 = [var_21]
    var_115 = module_0.Position(var_5, var_5, var_5)
    var_116 = module_0.Position(var_5, var_5, var_5)
    var_117 = module_0.Message(text=var_14, code=var_1, key=var_21, index=var_114, start_position=var_115, end_position=var_116)
    var_118 = [var_3]
    var_119 = module_0.Position(var_5, var_5, var_5)
    var_120 = module_0.Position(var_5, var_5, var_5)
    var_121 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_118, start_position=var_119, end_position=var_120)
    var_122 = [var_21]
    var_123 = module_0.Position(var_5, var_5, var_5)
    var_124 = module_0.Position(var_5, var_5, var_5)
    var_125 = module_0.Message(text=var_0, code=var_21, key=var_21, index=var_122, start_position=var_123, end_position=var_124)
    var_126 = [var_3]
    var_127 = module_0.Position(var_5, var_5, var_5)
    var_128 = module_0.Position(var_5, var_5, var_5)
    var_129 = module_0.Message(text=var_0, code=var_1, key=var_2, index=var_126, start_position=var_127, end_position=var_128)
    var_130 = [var_21]
    var_131 = module_0.Position(var_5, var_5, var_5)
    var_132 = module_0.Position(var_5, var_5, var_5)
    var_133 = module_0.Message(text=var_14, code=var_21, key=var_21, index=var_130, start_position=var_131, end_position=var_132)



# Parsed testcases at query #10
#--------------------------



def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'max_length'
    var_3 = module_0.Message(text=var_0, code=var_2)
    var_4 = 'username'
    var_5 = module_0.Message(text=var_0, key=var_4)
    var_6 = 'users'
    var_7 = 3
    var_8 = [var_6, var_7, var_4]
    var_9 = module_0.Message(text=var_0, index=var_8)
    var_10 = 1
    var_11 = 2
    var_12 = module_0.Position(var_10, var_11, var_7)
    var_13 = module_0.Message(text=var_0, position=var_12)
    var_14 = module_0.Position(var_10, var_11, var_7)
    var_15 = 5
    var_16 = 6
    var_17 = module_0.Position(var_10, var_15, var_16)
    var_18 = module_0.Message(text=var_0, start_position=var_14, end_position=var_17)
    var_19 = module_0.Message(text=var_0, code=var_2)
    var_20 = module_0.Message(text=var_0, code=var_2)
    var_21 = module_0.Message(text=var_0, code=var_2)
    var_22 = module_0.Message(text=var_0, code=var_2)
    var_23 = hash(var_21)
    var_24 = hash(var_22)
    var_25 = module_0.Message(text=var_0, code=var_2)
    var_26 = repr(var_25)
    assert var_26 == "Message(text='test', code='max_length')"
    var_27 = [var_6, var_7, var_4]
    var_28 = module_0.Message(text=var_0, index=var_27)
    var_29 = repr(var_28)
    assert var_29 == "Message(text='test', code='custom', index=['users', 3, 'username'])"
    var_30 = module_0.Position(var_10, var_11, var_7)
    var_31 = module_0.Message(text=var_0, position=var_30)
    var_32 = repr(var_31)
    assert var_32 == "Message(text='test', code='custom', position=Position(line_no=1, column_no=2, char_index=3))"
    var_33 = module_0.Position(var_10, var_11, var_7)
    var_34 = module_0.Position(var_10, var_15, var_16)
    var_35 = module_0.Message(text=var_0, start_position=var_33, end_position=var_34)
    var_36 = repr(var_35)
    assert var_36 == "Message(text='test', code='custom', start_position=Position(line_no=1, column_no=2, char_index=3), end_position=Position(line_no=1, column_no=5, char_index=6))"



# Parsed testcases at query #11
#--------------------------



def test_case_0():
    var_0 = 'Invalid value'
    var_1 = 'invalid'
    var_2 = 'username'
    var_3 = module_0.ValidationError(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_5 = [var_4]
    var_6 = str(var_3)
    assert var_6 == 'Invalid value'
    var_7 = [var_2]
    var_8 = module_0.Message(text=var_0, code=var_1, index=var_7)
    var_9 = [var_8]
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_11 = 'Too long'
    var_12 = 'max_length'
    var_13 = module_0.Message(text=var_11, code=var_12, key=var_2)
    var_14 = [var_10, var_13]
    var_15 = module_0.ValidationError(messages=var_14)
    var_16 = str(var_15)
    assert var_16 == "{'username': 'Too long'}"
    var_17 = 'users'
    var_18 = [var_17, var_2]
    var_19 = module_0.Message(text=var_0, code=var_1, index=var_18)
    var_20 = [var_17, var_2]
    var_21 = module_0.Message(text=var_11, code=var_12, index=var_20)
    var_22 = [var_19, var_21]
    var_23 = module_0.ValidationError(text=var_0, code=var_1, key=var_2)
    var_24 = module_0.ValidationError(text=var_0, code=var_1, key=var_2)
    var_25 = 'email'
    var_26 = module_0.ValidationError(text=var_0, code=var_1, key=var_25)
    var_27 = hash(var_23)
    var_28 = hash(var_24)
    var_29 = hash(var_23)
    var_30 = hash(var_26)
    var_31 = repr(var_23)
    assert var_31 == "ValidationError(text='Invalid value', code='invalid')"
    var_32 = repr(var_15)
    assert var_32 == "ValidationError([Message(text='Invalid value', code='invalid', index=['username']), Message(text='Too long', code='max_length', index=['username'])])"
    var_33 = bool(var_23)
    assert var_33 is True
    var_34 = module_0.ValidationError(text=var_0)
    var_35 = bool(var_34)
    assert var_35 is True
    var_36 = list(var_23)
    var_37 = list(var_15)
    var_38 = len(var_23)
    assert var_38 == 1
    var_39 = len(var_15)
    assert var_39 == 1
    var_40 = 1
    var_41 = 0
    var_42 = module_0.Position(var_40, var_40, var_41)
    var_43 = module_0.ValidationError(text=var_0, code=var_1, position=var_42)
    var_44 = module_0.Message(text=var_0, code=var_1, position=var_42)
    var_45 = [var_44]
    var_46 = str(var_43)
    assert var_46 == 'Invalid value'
    var_47 = []
    var_48 = module_0.Message(text=var_0, code=var_1, index=var_47, position=var_42)
    var_49 = [var_48]
    var_50 = module_0.Position(var_40, var_40, var_41)
    var_51 = 10
    var_52 = 9
    var_53 = module_0.Position(var_40, var_51, var_52)
    var_54 = module_0.ValidationError(text=var_0, code=var_1)
    var_55 = module_0.Message(text=var_0, code=var_1, start_position=var_50, end_position=var_53)
    var_56 = [var_55]
    var_57 = str(var_54)
    assert var_57 == 'Invalid value'
    var_58 = []
    var_59 = module_0.Message(text=var_0, code=var_1, index=var_58, start_position=var_50, end_position=var_53)
    var_60 = [var_59]
    var_61 = [var_17, var_41, var_2]
    var_62 = module_0.ValidationError(text=var_0, code=var_1)
    var_63 = [var_17, var_41, var_2]
    var_64 = module_0.Message(text=var_0, code=var_1, index=var_63)
    var_65 = [var_64]
    var_66 = str(var_62)
    assert var_66 == "{'users': {0: {'username': 'Invalid value'}}}"
    var_67 = [var_17, var_41, var_2]
    var_68 = module_0.Message(text=var_0, code=var_1, index=var_67)
    var_69 = [var_68]
    var_70 = [var_17, var_41, var_2]
    var_71 = module_0.Message(text=var_0, code=var_1, index=var_70)
    var_72 = [var_17, var_41, var_2]
    var_73 = module_0.Message(text=var_11, code=var_12, index=var_72)
    var_74 = 'Invalid email'
    var_75 = [var_17, var_41, var_25]
    var_76 = module_0.Message(text=var_74, code=var_1, index=var_75)
    var_77 = [var_71, var_73, var_76]
    var_78 = module_0.ValidationError(messages=var_77)
    var_79 = str(var_78)
    assert var_79 == "{'users': {0: {'username': 'Too long', 'email': 'Invalid email'}}}"
    var_80 = 'data'
    var_81 = [var_80, var_17, var_41, var_2]
    var_82 = module_0.Message(text=var_0, code=var_1, index=var_81)
    var_83 = [var_80, var_17, var_41, var_2]
    var_84 = module_0.Message(text=var_11, code=var_12, index=var_83)
    var_85 = [var_80, var_17, var_41, var_25]
    var_86 = module_0.Message(text=var_74, code=var_1, index=var_85)
    var_87 = [var_82, var_84, var_86]
    var_88 = [var_17, var_41, var_2]
    var_89 = module_0.ValidationError(text=var_0, code=var_1)
    var_90 = [var_17, var_41, var_2]
    var_91 = module_0.ValidationError(text=var_0, code=var_1)
    var_92 = [var_17, var_40, var_2]
    var_93 = module_0.ValidationError(text=var_0, code=var_1)
    var_94 = hash(var_89)
    var_95 = hash(var_91)
    var_96 = hash(var_89)
    var_97 = hash(var_93)
    var_98 = repr(var_89)
    assert var_98 == "ValidationError([Message(text='Invalid value', code='invalid', index=['users', 0, 'username'])])"
    var_99 = bool(var_89)
    assert var_99 is True
    var_100 = list(var_89)
    var_101 = list(var_78)
    var_102 = len(var_89)
    assert var_102 == 1
    var_103 = len(var_78)
    assert var_103 == 1
    var_104 = module_0.ValidationError(text=var_0, code=var_1)
    var_105 = module_0.Message(text=var_0, code=var_1)
    var_106 = [var_105]
    var_107 = str(var_104)
    assert var_107 == 'Invalid value'
    var_108 = []
    var_109 = module_0.Message(text=var_0, code=var_1, index=var_108)
    var_110 = [var_109]
    var_111 = [var_80]
    var_112 = module_0.Message(text=var_0, code=var_1, index=var_111)
    var_113 = [var_112]
    var_114 = module_0.ValidationError(text=var_0, code=var_1)
    var_115 = module_0.ValidationError(text=var_0, code=var_1)
    var_116 = module_0.ValidationError(text=var_0, code=var_12)
    var_117 = hash(var_114)
    var_118 = hash(var_115)
    var_119 = hash(var_114)
    var_120 = hash(var_116)
    var_121 = repr(var_114)
    assert var_121 == "ValidationError(text='Invalid value', code='invalid')"
    var_122 = bool(var_114)
    assert var_122 is True
    var_123 = list(var_114)
    var_124 = len(var_114)
    assert var_124 == 1
    var_125 = module_0.ValidationError(text=var_0)
    var_126 = 'custom'
    var_127 = module_0.Message(text=var_0, code=var_126)
    var_128 = [var_127]
    var_129 = str(var_125)
    assert var_129 == 'Invalid value'
    var_130 = []
    var_131 = module_0.Message(text=var_0, code=var_126, index=var_130)
    var_132 = [var_131]
    var_133 = [var_80]
    var_134 = module_0.Message(text=var_0, code=var_126, index=var_133)
    var_135 = [var_134]
    var_136 = module_0.ValidationError(text=var_0)
    var_137 = module_0.ValidationError(text=var_0)
    var_138 = module_0.ValidationError(text=var_0, code=var_1)
    var_139 = hash(var_136)
    var_140 = hash(var_137)
    var_141 = hash(var_136)
    var_142 = hash(var_138)



# Parsed testcases at query #12
#--------------------------



def test_case_0():
    var_0 = 'Invalid input'
    var_1 = 'invalid'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = str(var_2)
    assert var_3 == 'Invalid input'
    var_4 = 'username'
    var_5 = module_0.BaseError(text=var_0, code=var_1, key=var_4)
    var_6 = str(var_5)
    assert var_6 == "{'username': 'Invalid input'}"
    var_7 = module_0.Message(text=var_0, code=var_1, key=var_4)
    var_8 = 'Too short'
    var_9 = 'min_length'
    var_10 = 'password'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_12 = [var_7, var_11]
    var_13 = module_0.BaseError(messages=var_12)
    var_14 = str(var_13)
    assert var_14 == "{'username': 'Invalid input', 'password': 'Too short'}"
    var_15 = 'user'
    var_16 = [var_15, var_4]
    var_17 = module_0.Message(text=var_0, code=var_1, index=var_16)
    var_18 = [var_15, var_10]
    var_19 = module_0.Message(text=var_8, code=var_9, index=var_18)
    var_20 = [var_17, var_19]
    var_21 = module_0.BaseError(messages=var_20)
    var_22 = str(var_21)
    assert var_22 == "{'user': {'username': 'Invalid input', 'password': 'Too short'}}"
    var_23 = []
    var_24 = module_0.BaseError(messages=var_23)
    var_25 = str(var_24)
    assert var_25 == '{}'
    var_26 = 1
    var_27 = 5
    var_28 = 10
    var_29 = module_0.Position(var_26, var_27, var_28)
    var_30 = module_0.BaseError(text=var_0, code=var_1, position=var_29)
    var_31 = str(var_30)
    assert var_31 == 'Invalid input'
    var_32 = module_0.Position(var_26, var_27, var_28)
    var_33 = 15
    var_34 = module_0.Position(var_26, var_28, var_33)
    var_35 = module_0.BaseError(text=var_0, code=var_1)
    var_36 = str(var_35)
    assert var_36 == 'Invalid input'
    var_37 = module_0.Position(var_26, var_27, var_28)
    var_38 = module_0.BaseError(text=var_0, code=var_1, key=var_4, position=var_37)
    var_39 = str(var_38)
    assert var_39 == "{'username': 'Invalid input'}"
    var_40 = module_0.Position(var_26, var_27, var_28)
    var_41 = module_0.Position(var_26, var_28, var_33)
    var_42 = module_0.BaseError(text=var_0, code=var_1, key=var_4)
    var_43 = str(var_42)
    assert var_43 == "{'username': 'Invalid input'}"
    var_44 = []
    var_45 = module_0.BaseError(text=var_0, code=var_1)
    var_46 = str(var_45)
    assert var_46 == 'Invalid input'
    var_47 = [var_15, var_4]
    var_48 = module_0.BaseError(text=var_0, code=var_1)
    var_49 = str(var_48)
    assert var_49 == "{'user': {'username': 'Invalid input'}}"
    var_50 = module_0.Position(var_26, var_27, var_28)
    var_51 = [var_15, var_4]
    var_52 = module_0.BaseError(text=var_0, code=var_1, position=var_50)
    var_53 = str(var_52)
    assert var_53 == "{'user': {'username': 'Invalid input'}}"
    var_54 = module_0.Position(var_26, var_27, var_28)
    var_55 = module_0.Position(var_26, var_28, var_33)
    var_56 = [var_15, var_4]
    var_57 = module_0.BaseError(text=var_0, code=var_1)
    var_58 = str(var_57)
    assert var_58 == "{'user': {'username': 'Invalid input'}}"
    var_59 = []
    var_60 = module_0.BaseError(text=var_0, code=var_1)
    var_61 = str(var_60)
    assert var_61 == 'Invalid input'
    var_62 = module_0.Position(var_26, var_27, var_28)
    var_63 = []
    var_64 = module_0.BaseError(text=var_0, code=var_1, position=var_62)
    var_65 = str(var_64)
    assert var_65 == 'Invalid input'
    var_66 = module_0.Position(var_26, var_27, var_28)
    var_67 = module_0.Position(var_26, var_28, var_33)
    var_68 = []
    var_69 = module_0.BaseError(text=var_0, code=var_1)
    var_70 = str(var_69)
    assert var_70 == 'Invalid input'
    var_71 = module_0.Position(var_26, var_27, var_28)
    var_72 = module_0.Position(var_26, var_28, var_33)
    var_73 = []
    var_74 = module_0.BaseError(text=var_0, code=var_1)
    var_75 = str(var_74)
    assert var_75 == 'Invalid input'
    var_76 = module_0.Position(var_26, var_27, var_28)
    var_77 = module_0.Position(var_26, var_28, var_33)
    var_78 = []
    var_79 = module_0.BaseError(text=var_0, code=var_1)
    var_80 = str(var_79)
    assert var_80 == 'Invalid input'
    var_81 = module_0.Position(var_26, var_27, var_28)
    var_82 = module_0.Position(var_26, var_28, var_33)
    var_83 = []
    var_84 = module_0.BaseError(text=var_0, code=var_1)
    var_85 = str(var_84)
    assert var_85 == 'Invalid input'
    var_86 = module_0.Position(var_26, var_27, var_28)
    var_87 = module_0.Position(var_26, var_28, var_33)
    var_88 = []
    var_89 = module_0.BaseError(text=var_0, code=var_1)
    var_90 = str(var_89)
    assert var_90 == 'Invalid input'



# Parsed testcases at query #13
#--------------------------



def test_case_0():
    var_0 = 'Error message'
    var_1 = 'custom'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = str(var_2)
    assert var_3 == 'Error message'
    var_4 = 'key'
    var_5 = module_0.BaseError(text=var_0, code=var_1, key=var_4)
    var_6 = str(var_5)
    assert var_6 == "{'key': 'Error message'}"
    var_7 = 'Error 1'
    var_8 = 'key1'
    var_9 = module_0.Message(text=var_7, code=var_1, key=var_8)
    var_10 = 'Error 2'
    var_11 = 'key2'
    var_12 = module_0.Message(text=var_10, code=var_1, key=var_11)
    var_13 = [var_9, var_12]
    var_14 = module_0.BaseError(messages=var_13)
    var_15 = str(var_14)
    assert var_15 == "{'key1': 'Error 1', 'key2': 'Error 2'}"
    var_16 = 'subkey1'
    var_17 = [var_8, var_16]
    var_18 = module_0.Message(text=var_7, code=var_1, index=var_17)
    var_19 = 'subkey2'
    var_20 = [var_11, var_19]
    var_21 = module_0.Message(text=var_10, code=var_1, index=var_20)
    var_22 = [var_18, var_21]
    var_23 = module_0.BaseError(messages=var_22)
    var_24 = str(var_23)
    assert var_24 == "{'key1': {'subkey1': 'Error 1'}, 'key2': {'subkey2': 'Error 2'}}"
    var_25 = []
    var_26 = module_0.BaseError(messages=var_25)
    var_27 = str(var_26)
    assert var_27 == '{}'
    var_28 = 1
    var_29 = 0
    var_30 = module_0.Position(var_28, var_28, var_29)
    var_31 = module_0.BaseError(text=var_0, code=var_1, position=var_30)
    var_32 = str(var_31)
    assert var_32 == 'Error message'
    var_33 = module_0.Position(var_28, var_28, var_29)
    var_34 = 5
    var_35 = 4
    var_36 = module_0.Position(var_28, var_34, var_35)
    var_37 = module_0.BaseError(text=var_0, code=var_1)
    var_38 = str(var_37)
    assert var_38 == 'Error message'
    var_39 = module_0.BaseError(text=var_0, code=var_1, key=var_4, position=var_30)
    var_40 = str(var_39)
    assert var_40 == "{'key': 'Error message'}"
    var_41 = module_0.BaseError(text=var_0, code=var_1, key=var_4)
    var_42 = str(var_41)
    assert var_42 == "{'key': 'Error message'}"
    var_43 = []
    var_44 = module_0.BaseError(text=var_0, code=var_1)
    var_45 = str(var_44)
    assert var_45 == 'Error message'
    var_46 = [var_8, var_11]
    var_47 = module_0.BaseError(text=var_0, code=var_1)
    var_48 = str(var_47)
    assert var_48 == "{'key1': {'key2': 'Error message'}}"
    var_49 = [var_29, var_28]
    var_50 = module_0.BaseError(text=var_0, code=var_1)
    var_51 = str(var_50)
    assert var_51 == "{0: {1: 'Error message'}}"
    var_52 = [var_4, var_29]
    var_53 = module_0.BaseError(text=var_0, code=var_1)
    var_54 = str(var_53)
    assert var_54 == "{'key': {0: 'Error message'}}"
    var_55 = 'key with spaces'
    var_56 = 'key-with-dashes'
    var_57 = [var_55, var_56]
    var_58 = module_0.BaseError(text=var_0, code=var_1)
    var_59 = str(var_58)
    assert var_59 == "{'key with spaces': {'key-with-dashes': 'Error message'}}"
    var_60 = ''
    var_61 = [var_60]
    var_62 = module_0.BaseError(text=var_0, code=var_1)
    var_63 = str(var_62)
    assert var_63 == "{'': 'Error message'}"
    var_64 = None
    var_65 = [var_64]
    var_66 = module_0.BaseError(text=var_0, code=var_1)
    var_67 = str(var_66)
    assert var_67 == "{None: 'Error message'}"
    var_68 = True
    var_69 = [var_68]
    var_70 = module_0.BaseError(text=var_0, code=var_1)
    var_71 = str(var_70)
    assert var_71 == "{True: 'Error message'}"
    var_72 = 3.14
    var_73 = [var_72]
    var_74 = module_0.BaseError(text=var_0, code=var_1)
    var_75 = str(var_74)
    assert var_75 == "{3.14: 'Error message'}"
    var_76 = str(var_74)
    assert var_76 == "{(1+2j): 'Error message'}"
    var_77 = 2
    var_78 = (var_68, var_77)
    var_79 = [var_78]
    var_80 = module_0.BaseError(text=var_0, code=var_1)
    var_81 = str(var_80)
    assert var_81 == "{(1, 2): 'Error message'}"
    var_82 = [var_68, var_77]
    var_83 = [var_82]
    var_84 = module_0.BaseError(text=var_0, code=var_1)
    var_85 = str(var_84)
    assert var_85 == "{'[1, 2]': 'Error message'}"
    var_86 = {var_68, var_77}
    var_87 = [var_86]
    var_88 = module_0.BaseError(text=var_0, code=var_1)
    var_89 = str(var_88)
    assert var_89 == "{'{1, 2}': 'Error message'}"
    var_90 = 'value'
    var_91 = {var_4: var_90}
    var_92 = [var_91]
    var_93 = module_0.BaseError(text=var_0, code=var_1)
    var_94 = str(var_93)
    assert var_94 == "{'{'key': 'value'}': 'Error message'}"
    var_95 = str(var_93)
    assert var_95 == "{'CustomObject()': 'Error message'}"
    var_96 = 'key3'
    var_97 = [var_8, var_11, var_96]
    var_98 = module_0.BaseError(text=var_0, code=var_1)
    var_99 = str(var_98)
    assert var_99 == "{'key1': {'key2': {'key3': 'Error message'}}}"
    var_100 = [var_4, var_4]
    var_101 = module_0.BaseError(text=var_0, code=var_1)
    var_102 = str(var_101)
    assert var_102 == "{'key': {'key': 'Error message'}}"
    var_103 = [var_60, var_60]
    var_104 = module_0.BaseError(text=var_0, code=var_1)
    var_105 = str(var_104)
    assert var_105 == "{'': {'': 'Error message'}}"
    var_106 = [var_64, var_64]
    var_107 = module_0.BaseError(text=var_0, code=var_1)
    var_108 = str(var_107)
    assert var_108 == "{None: {None: 'Error message'}}"
    var_109 = True
    var_110 = [var_4, var_29, var_109]
    var_111 = module_0.BaseError(text=var_0, code=var_1)
    var_112 = str(var_111)
    assert var_112 == "{'key': {0: {True: 'Error message'}}}"
    var_113 = 'key_with_underscores'
    var_114 = [var_55, var_56, var_113]
    var_115 = module_0.BaseError(text=var_0, code=var_1)
    var_116 = str(var_115)
    assert var_116 == "{'key with spaces': {'key-with-dashes': {'key_with_underscores': 'Error message'}}}"
    var_117 = [var_60, var_64]
    var_118 = module_0.BaseError(text=var_0, code=var_1)
    var_119 = str(var_118)
    assert var_119 == "{'': {None: 'Error message'}}"
    var_120 = [var_64, var_60]
    var_121 = module_0.BaseError(text=var_0, code=var_1)
    var_122 = str(var_121)
    assert var_122 == "{None: {'': 'Error message'}}"
    var_123 = True
    var_124 = [var_123, var_123]
    var_125 = module_0.BaseError(text=var_0, code=var_1)
    var_126 = str(var_125)
    assert var_126 == "{True: {1: 'Error message'}}"
    var_127 = True
    var_128 = [var_123, var_127]
    var_129 = module_0.BaseError(text=var_0, code=var_1)
    var_130 = str(var_129)
    assert var_130 == "{1: {True: 'Error message'}}"
    var_131 = [var_72, var_4]
    var_132 = module_0.BaseError(text=var_0, code=var_1)
    var_133 = str(var_132)
    assert var_133 == "{3.14: {'key': 'Error message'}}"
    var_134 = [var_4, var_72]
    var_135 = module_0.BaseError(text=var_0, code=var_1)



# Parsed testcases at query #14
#--------------------------



def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Message(text=var_0)
    var_2 = 'max_length'
    var_3 = module_0.Message(text=var_0, code=var_2)
    var_4 = 'username'
    var_5 = module_0.Message(text=var_0, key=var_4)
    var_6 = 'users'
    var_7 = 3
    var_8 = [var_6, var_7, var_4]
    var_9 = module_0.Message(text=var_0, index=var_8)
    var_10 = 1
    var_11 = 2
    var_12 = module_0.Position(var_10, var_11, var_7)
    var_13 = module_0.Message(text=var_0, position=var_12)
    var_14 = module_0.Position(var_10, var_11, var_7)
    var_15 = 5
    var_16 = 6
    var_17 = module_0.Position(var_10, var_15, var_16)
    var_18 = module_0.Message(text=var_0, start_position=var_14, end_position=var_17)
    var_19 = 'test'
    var_20 = 'username'
    var_21 = 'users'
    var_22 = 3
    var_23 = [var_21, var_22, var_20]
    var_24 = module_0.Message(text=var_19, key=var_20, index=var_23)
    var_25 = 'test'
    var_26 = module_0.Message(text=var_25, position=var_12, start_position=var_14)
    var_27 = 'test'
    var_28 = module_0.Message(text=var_27, position=var_12, end_position=var_17)



# Parsed testcases at query #15
#--------------------------



def test_case_0():
    var_0 = 'Invalid input'
    var_1 = 'invalid'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = str(var_2)
    assert var_3 == 'Invalid input'
    var_4 = 'username'
    var_5 = module_0.BaseError(text=var_0, code=var_1, key=var_4)
    var_6 = str(var_5)
    assert var_6 == "{'username': 'Invalid input'}"
    var_7 = module_0.Message(text=var_0, code=var_1, key=var_4)
    var_8 = 'Missing field'
    var_9 = 'missing'
    var_10 = 'email'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10)
    var_12 = [var_7, var_11]
    var_13 = module_0.BaseError(messages=var_12)
    var_14 = str(var_13)
    assert var_14 == "{'username': 'Invalid input', 'email': 'Missing field'}"
    var_15 = 'user'
    var_16 = [var_15, var_4]
    var_17 = module_0.Message(text=var_0, code=var_1, index=var_16)
    var_18 = [var_15, var_10]
    var_19 = module_0.Message(text=var_8, code=var_9, index=var_18)
    var_20 = [var_17, var_19]
    var_21 = module_0.BaseError(messages=var_20)
    var_22 = str(var_21)
    assert var_22 == "{'user': {'username': 'Invalid input', 'email': 'Missing field'}}"
    var_23 = []
    var_24 = module_0.BaseError(messages=var_23)
    var_25 = str(var_24)
    assert var_25 == '{}'
    var_26 = 1
    var_27 = 5
    var_28 = 10
    var_29 = module_0.Position(var_26, var_27, var_28)
    var_30 = module_0.BaseError(text=var_0, code=var_1, position=var_29)
    var_31 = str(var_30)
    assert var_31 == 'Invalid input'
    var_32 = module_0.Position(var_26, var_27, var_28)
    var_33 = 15
    var_34 = module_0.Position(var_26, var_28, var_33)
    var_35 = module_0.BaseError(text=var_0, code=var_1)
    var_36 = str(var_35)
    assert var_36 == 'Invalid input'
    var_37 = module_0.BaseError(text=var_0, code=var_1, key=var_4, position=var_29)
    var_38 = str(var_37)
    assert var_38 == "{'username': 'Invalid input'}"
    var_39 = module_0.BaseError(text=var_0, code=var_1, key=var_4)
    var_40 = str(var_39)
    assert var_40 == "{'username': 'Invalid input'}"
    var_41 = []
    var_42 = module_0.BaseError(text=var_0, code=var_1)
    var_43 = str(var_42)
    assert var_43 == 'Invalid input'
    var_44 = [var_15, var_4]
    var_45 = module_0.BaseError(text=var_0, code=var_1)
    var_46 = str(var_45)
    assert var_46 == "{'user': {'username': 'Invalid input'}}"
    var_47 = 0
    var_48 = [var_47, var_26]
    var_49 = module_0.BaseError(text=var_0, code=var_1)
    var_50 = str(var_49)
    assert var_50 == "{0: {1: 'Invalid input'}}"
    var_51 = [var_15, var_47]
    var_52 = module_0.BaseError(text=var_0, code=var_1)
    var_53 = str(var_52)
    assert var_53 == "{'user': {0: 'Invalid input'}}"
    var_54 = [var_4]
    var_55 = module_0.BaseError(text=var_0, code=var_1)
    var_56 = str(var_55)
    assert var_56 == "{'username': 'Invalid input'}"
    var_57 = [var_15, var_4]
    var_58 = module_0.BaseError(text=var_0, code=var_1)
    var_59 = str(var_58)
    assert var_59 == "{'user': {'username': 'Invalid input'}}"
    var_60 = 'profile'
    var_61 = [var_15, var_60, var_4]
    var_62 = module_0.BaseError(text=var_0, code=var_1)
    var_63 = str(var_62)
    assert var_63 == "{'user': {'profile': {'username': 'Invalid input'}}}"
    var_64 = 'settings'
    var_65 = [var_15, var_60, var_64, var_4]
    var_66 = module_0.BaseError(text=var_0, code=var_1)
    var_67 = str(var_66)
    assert var_67 == "{'user': {'profile': {'settings': {'username': 'Invalid input'}}}}"
    var_68 = 'preferences'
    var_69 = [var_15, var_60, var_64, var_68, var_4]
    var_70 = module_0.BaseError(text=var_0, code=var_1)
    var_71 = str(var_70)
    assert var_71 == "{'user': {'profile': {'settings': {'preferences': {'username': 'Invalid input'}}}}}"
    var_72 = 'security'
    var_73 = [var_15, var_60, var_64, var_68, var_72, var_4]
    var_74 = module_0.BaseError(text=var_0, code=var_1)
    var_75 = str(var_74)
    assert var_75 == "{'user': {'profile': {'settings': {'preferences': {'security': {'username': 'Invalid input'}}}}}}"
    var_76 = 'authentication'
    var_77 = [var_15, var_60, var_64, var_68, var_72, var_76, var_4]
    var_78 = module_0.BaseError(text=var_0, code=var_1)
    var_79 = str(var_78)
    assert var_79 == "{'user': {'profile': {'settings': {'preferences': {'security': {'authentication': {'username': 'Invalid input'}}}}}}}"
    var_80 = 'two_factor'
    var_81 = [var_15, var_60, var_64, var_68, var_72, var_76, var_80, var_4]
    var_82 = module_0.BaseError(text=var_0, code=var_1)
    var_83 = str(var_82)
    assert var_83 == "{'user': {'profile': {'settings': {'preferences': {'security': {'authentication': {'two_factor': {'username': 'Invalid input'}}}}}}}}"
    var_84 = 'method'
    var_85 = [var_15, var_60, var_64, var_68, var_72, var_76, var_80, var_84, var_4]
    var_86 = module_0.BaseError(text=var_0, code=var_1)
    var_87 = str(var_86)
    assert var_87 == "{'user': {'profile': {'settings': {'preferences': {'security': {'authentication': {'two_factor': {'method': {'username': 'Invalid input'}}}}}}}}}"
    var_88 = 'type'
    var_89 = [var_15, var_60, var_64, var_68, var_72, var_76, var_80, var_84, var_88, var_4]
    var_90 = module_0.BaseError(text=var_0, code=var_1)
    var_91 = str(var_90)
    assert var_91 == "{'user': {'profile': {'settings': {'preferences': {'security': {'authentication': {'two_factor': {'method': {'type': {'username': 'Invalid input'}}}}}}}}}}"
    var_92 = 'provider'
    var_93 = [var_15, var_60, var_64, var_68, var_72, var_76, var_80, var_84, var_88, var_92, var_4]
    var_94 = module_0.BaseError(text=var_0, code=var_1)
    var_95 = str(var_94)
    assert var_95 == "{'user': {'profile': {'settings': {'preferences': {'security': {'authentication': {'two_factor': {'method': {'type': {'provider': {'username': 'Invalid input'}}}}}}}}}}}"
    var_96 = [var_15, var_60, var_64, var_68, var_72, var_76, var_80, var_84, var_88, var_92, var_64, var_4]
    var_97 = module_0.BaseError(text=var_0, code=var_1)
    var_98 = str(var_97)
    assert var_98 == "{'user': {'profile': {'settings': {'preferences': {'security': {'authentication': {'two_factor': {'method': {'type': {'provider': {'settings': {'username': 'Invalid input'}}}}}}}}}}}}"
    var_99 = 'configuration'
    var_100 = [var_15, var_60, var_64, var_68, var_72, var_76, var_80, var_84, var_88, var_92, var_64, var_99, var_4]
    var_101 = module_0.BaseError(text=var_0, code=var_1)
    var_102 = str(var_101)
    assert var_102 == "{'user': {'profile': {'settings': {'preferences': {'security': {'authentication': {'two_factor': {'method': {'type': {'provider': {'settings': {'configuration': {'username': 'Invalid input'}}}}}}}}}}}}}"



####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------



def test_case_0():
    var_0 = 'Error 1'
    var_1 = 'code1'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = module_0.BaseError(text=var_0, code=var_1)
    var_4 = module_0.BaseError(text=var_0, code=var_1)
    var_5 = 'Error 2'
    var_6 = 'code2'
    var_7 = module_0.BaseError(text=var_5, code=var_6)
    var_8 = module_0.Message(text=var_0, code=var_1)
    var_9 = module_0.Message(text=var_5, code=var_6)
    var_10 = [var_8, var_9]
    var_11 = module_0.BaseError(messages=var_10)
    var_12 = module_0.Message(text=var_5, code=var_6)
    var_13 = module_0.Message(text=var_0, code=var_1)
    var_14 = [var_12, var_13]
    var_15 = module_0.BaseError(messages=var_14)
    var_16 = 'key1'
    var_17 = [var_16]
    var_18 = module_0.Message(text=var_0, code=var_1, index=var_17)
    var_19 = [var_18]
    var_20 = module_0.BaseError(messages=var_19)
    var_21 = 'key2'
    var_22 = [var_21]
    var_23 = module_0.Message(text=var_0, code=var_1, index=var_22)
    var_24 = [var_23]
    var_25 = module_0.BaseError(messages=var_24)
    var_26 = 1
    var_27 = module_0.Position(var_26, var_26, var_26)
    var_28 = module_0.Message(text=var_0, code=var_1, position=var_27)
    var_29 = [var_28]
    var_30 = module_0.BaseError(messages=var_29)
    var_31 = 2
    var_32 = module_0.Position(var_31, var_31, var_31)
    var_33 = module_0.Message(text=var_0, code=var_1, position=var_32)
    var_34 = [var_33]
    var_35 = module_0.BaseError(messages=var_34)
    var_36 = module_0.Position(var_26, var_26, var_26)
    var_37 = module_0.Position(var_26, var_31, var_31)
    var_38 = module_0.Message(text=var_0, code=var_1, start_position=var_36, end_position=var_37)
    var_39 = [var_38]
    var_40 = module_0.BaseError(messages=var_39)
    var_41 = module_0.Position(var_26, var_26, var_26)
    var_42 = 3
    var_43 = module_0.Position(var_26, var_42, var_42)
    var_44 = module_0.Message(text=var_0, code=var_1, start_position=var_41, end_position=var_43)
    var_45 = [var_44]
    var_46 = module_0.BaseError(messages=var_45)
    var_47 = module_0.Message(text=var_0, code=var_1)
    var_48 = [var_47]
    var_49 = module_0.BaseError(messages=var_48)
    var_50 = module_0.Message(text=var_0, code=var_6)
    var_51 = [var_50]
    var_52 = module_0.BaseError(messages=var_51)
    var_53 = module_0.Message(text=var_0, code=var_1)
    var_54 = [var_53]
    var_55 = module_0.BaseError(messages=var_54)
    var_56 = module_0.Message(text=var_5, code=var_1)
    var_57 = [var_56]
    var_58 = module_0.BaseError(messages=var_57)
    var_59 = module_0.Message(text=var_0, code=var_1)
    var_60 = [var_59]
    var_61 = module_0.BaseError(messages=var_60)
    var_62 = module_0.Message(text=var_0, code=var_1)
    var_63 = module_0.Message(text=var_5, code=var_6)
    var_64 = [var_62, var_63]
    var_65 = module_0.BaseError(messages=var_64)
    var_66 = module_0.Message(text=var_0, code=var_1)
    var_67 = [var_66]
    var_68 = module_0.BaseError(messages=var_67)
    var_69 = module_0.Message(text=var_0, code=var_1, key=var_16)
    var_70 = [var_69]
    var_71 = module_0.BaseError(messages=var_70)
    var_72 = [var_16, var_21]
    var_73 = module_0.Message(text=var_0, code=var_1, index=var_72)
    var_74 = [var_73]
    var_75 = module_0.BaseError(messages=var_74)
    var_76 = [var_16]
    var_77 = module_0.Message(text=var_0, code=var_1, index=var_76)
    var_78 = [var_77]
    var_79 = module_0.BaseError(messages=var_78)
    var_80 = [var_16, var_21]
    var_81 = module_0.Message(text=var_0, code=var_1, index=var_80)
    var_82 = [var_81]
    var_83 = module_0.BaseError(messages=var_82)
    var_84 = 'key3'
    var_85 = [var_16, var_84]
    var_86 = module_0.Message(text=var_0, code=var_1, index=var_85)
    var_87 = [var_86]
    var_88 = module_0.BaseError(messages=var_87)
    var_89 = [var_16, var_31]
    var_90 = module_0.Message(text=var_0, code=var_1, index=var_89)
    var_91 = [var_90]
    var_92 = module_0.BaseError(messages=var_91)
    var_93 = '2'
    var_94 = [var_16, var_93]
    var_95 = module_0.Message(text=var_0, code=var_1, index=var_94)
    var_96 = [var_95]
    var_97 = module_0.BaseError(messages=var_96)
    var_98 = [var_16, var_21]
    var_99 = module_0.Message(text=var_0, code=var_1, index=var_98)
    var_100 = [var_99]
    var_101 = module_0.BaseError(messages=var_100)
    var_102 = [var_21, var_16]
    var_103 = module_0.Message(text=var_0, code=var_1, index=var_102)
    var_104 = [var_103]
    var_105 = module_0.BaseError(messages=var_104)
    var_106 = [var_16, var_16]
    var_107 = module_0.Message(text=var_0, code=var_1, index=var_106)
    var_108 = [var_107]
    var_109 = module_0.BaseError(messages=var_108)
    var_110 = [var_16, var_16]
    var_111 = module_0.Message(text=var_0, code=var_1, index=var_110)
    var_112 = [var_111]
    var_113 = module_0.BaseError(messages=var_112)
    var_114 = [var_16, var_16]
    var_115 = module_0.Message(text=var_0, code=var_1, index=var_114)
    var_116 = [var_115]
    var_117 = module_0.BaseError(messages=var_116)
    var_118 = [var_16, var_21]
    var_119 = module_0.Message(text=var_0, code=var_1, index=var_118)
    var_120 = [var_119]
    var_121 = module_0.BaseError(messages=var_120)
    var_122 = [var_16, var_16]
    var_123 = module_0.Message(text=var_0, code=var_1, index=var_122)
    var_124 = [var_123]
    var_125 = module_0.BaseError(messages=var_124)
    var_126 = [var_21, var_21]
    var_127 = module_0.Message(text=var_0, code=var_1, index=var_126)
    var_128 = [var_127]
    var_129 = module_0.BaseError(messages=var_128)
    var_130 = [var_16, var_16]
    var_131 = module_0.Message(text=var_0, code=var_1, index=var_130)
    var_132 = [var_131]
    var_133 = module_0.BaseError(messages=var_132)
    var_134 = [var_16, var_16, var_16]
    var_135 = module_0.Message(text=var_0, code=var_1, index=var_134)
    var_136 = [var_135]
    var_137 = module_0.BaseError(messages=var_136)
    var_138 = [var_16, var_16, var_16]
    var_139 = module_0.Message(text=var_0, code=var_1, index=var_138)
    var_140 = [var_139]
    var_141 = module_0.BaseError(messages=var_140)
    var_142 = [var_16, var_16]
    var_143 = module_0.Message(text=var_0, code=var_1, index=var_142)
    var_144 = [var_143]
    var_145 = module_0.BaseError(messages=var_144)
    var_146 = [var_16, var_16, var_16]
    var_147 = module_0.Message(text=var_0, code=var_1, index=var_146)
    var_148 = [var_147]
    var_149 = module_0.BaseError(messages=var_148)
    var_150 = [var_16, var_16, var_21]
    var_151 = module_0.Message(text=var_0, code=var_1, index=var_150)
    var_152 = [var_151]
    var_153 = module_0.BaseError(messages=var_152)
    var_154 = [var_16, var_16, var_16]
    var_155 = module_0.Message(text=var_0, code=var_1, index=var_154)
    var_156 = [var_155]
    var_157 = module_0.BaseError(messages=var_156)
    var_158 = [var_16, var_21, var_21]
    var_159 = module_0.Message(text=var_0, code=var_1, index=var_158)
    var_160 = [var_159]
    var_161 = module_0.BaseError(messages=var_160)
    var_162 = [var_16, var_16, var_16]
    var_163 = module_0.Message(text=var_0, code=var_1, index=var_162)
    var_164 = [var_163]
    var_165 = module_0.BaseError(messages=var_164)
    var_166 = [var_21, var_21, var_21]
    var_167 = module_0.Message(text=var_0, code=var_1, index=var_166)
    var_168 = [var_167]
    var_169 = module_0.BaseError(messages=var_168)
    var_170 = [var_16, var_16, var_16]
    var_171 = module_0.Message(text=var_0, code=var_1, index=var_170)
    var_172 = [var_171]
    var_173 = module_0.BaseError(messages=var_172)



# Parsed testcases at query #2
#--------------------------



def test_case_0():
    var_0 = 42
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = 'Invalid input'
    var_3 = module_0.ValidationError(text=var_2)
    var_4 = module_0.ValidationResult(error=var_3)
    var_5 = module_0.ValidationError(text=var_2)
    assert var_5 is None
    var_6 = module_0.ValidationResult(value=var_0, error=var_5)
    var_7 = module_0.ValidationResult()
    var_8 = 'All test cases passed!'
    var_9 = print(var_8)



# Parsed testcases at query #3
#--------------------------



def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'key'
    var_3 = module_0.BaseError(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.BaseError(text=var_0, code=var_1, key=var_2)
    var_5 = 'Different error message'
    var_6 = module_0.BaseError(text=var_5, code=var_1, key=var_2)
    var_7 = 'Error 1'
    var_8 = module_0.Message(text=var_7)
    var_9 = 'Error 2'
    var_10 = module_0.Message(text=var_9)
    var_11 = [var_8, var_10]
    var_12 = module_0.BaseError(messages=var_11)
    var_13 = module_0.Message(text=var_9)
    var_14 = module_0.Message(text=var_7)
    var_15 = [var_13, var_14]
    var_16 = module_0.BaseError(messages=var_15)
    var_17 = module_0.Message(text=var_7)
    var_18 = [var_17]
    var_19 = module_0.BaseError(messages=var_18)
    var_20 = 'different_code'
    var_21 = module_0.BaseError(text=var_0, code=var_20, key=var_2)
    var_22 = 'different_key'
    var_23 = module_0.BaseError(text=var_0, code=var_1, key=var_22)
    var_24 = 1
    var_25 = module_0.Position(var_24, var_24, var_24)
    var_26 = module_0.BaseError(text=var_0, code=var_1, key=var_2, position=var_25)
    var_27 = module_0.Position(var_24, var_24, var_24)
    var_28 = 2
    var_29 = module_0.Position(var_24, var_28, var_28)
    var_30 = module_0.BaseError(text=var_0, code=var_1, key=var_2)
    var_31 = [var_2]
    var_32 = module_0.Message(text=var_0, index=var_31)
    var_33 = [var_32]
    var_34 = module_0.BaseError(messages=var_33)
    var_35 = [var_22]
    var_36 = module_0.Message(text=var_0, index=var_35)
    var_37 = [var_36]
    var_38 = module_0.BaseError(messages=var_37)
    var_39 = module_0.BaseError(text=var_5, code=var_1, key=var_2)
    var_40 = module_0.BaseError(text=var_0, code=var_20, key=var_2)
    var_41 = module_0.BaseError(text=var_0, code=var_1, key=var_22)
    var_42 = module_0.Position(var_24, var_24, var_24)
    var_43 = module_0.BaseError(text=var_0, code=var_1, key=var_2, position=var_42)
    var_44 = module_0.Position(var_24, var_24, var_24)
    var_45 = module_0.Position(var_24, var_28, var_28)
    var_46 = module_0.BaseError(text=var_0, code=var_1, key=var_2)
    var_47 = [var_2]
    var_48 = module_0.Message(text=var_0, index=var_47)
    var_49 = [var_48]
    var_50 = module_0.BaseError(messages=var_49)
    var_51 = [var_22]
    var_52 = module_0.Message(text=var_0, index=var_51)
    var_53 = [var_52]
    var_54 = module_0.BaseError(messages=var_53)
    var_55 = module_0.BaseError(text=var_5, code=var_1, key=var_2)
    var_56 = module_0.BaseError(text=var_0, code=var_20, key=var_2)
    var_57 = module_0.BaseError(text=var_0, code=var_1, key=var_22)
    var_58 = module_0.Position(var_24, var_24, var_24)
    var_59 = module_0.BaseError(text=var_0, code=var_1, key=var_2, position=var_58)
    var_60 = module_0.Position(var_24, var_24, var_24)
    var_61 = module_0.Position(var_24, var_28, var_28)
    var_62 = module_0.BaseError(text=var_0, code=var_1, key=var_2)
    var_63 = [var_2]
    var_64 = module_0.Message(text=var_0, index=var_63)
    var_65 = [var_64]
    var_66 = module_0.BaseError(messages=var_65)
    var_67 = [var_22]
    var_68 = module_0.Message(text=var_0, index=var_67)
    var_69 = [var_68]
    var_70 = module_0.BaseError(messages=var_69)
    var_71 = module_0.BaseError(text=var_5, code=var_1, key=var_2)
    var_72 = module_0.BaseError(text=var_0, code=var_20, key=var_2)
    var_73 = module_0.BaseError(text=var_0, code=var_1, key=var_22)
    var_74 = module_0.Position(var_24, var_24, var_24)
    var_75 = module_0.BaseError(text=var_0, code=var_1, key=var_2, position=var_74)
    var_76 = module_0.Position(var_24, var_24, var_24)
    var_77 = module_0.Position(var_24, var_28, var_28)
    var_78 = module_0.BaseError(text=var_0, code=var_1, key=var_2)
    var_79 = [var_2]
    var_80 = module_0.Message(text=var_0, index=var_79)
    var_81 = [var_80]
    var_82 = module_0.BaseError(messages=var_81)
    var_83 = [var_22]
    var_84 = module_0.Message(text=var_0, index=var_83)
    var_85 = [var_84]
    var_86 = module_0.BaseError(messages=var_85)
    var_87 = module_0.BaseError(text=var_5, code=var_1, key=var_2)
    var_88 = module_0.BaseError(text=var_0, code=var_20, key=var_2)
    var_89 = module_0.BaseError(text=var_0, code=var_1, key=var_22)
    var_90 = module_0.Position(var_24, var_24, var_24)
    var_91 = module_0.BaseError(text=var_0, code=var_1, key=var_2, position=var_90)
    var_92 = module_0.Position(var_24, var_24, var_24)
    var_93 = module_0.Position(var_24, var_28, var_28)
    var_94 = module_0.BaseError(text=var_0, code=var_1, key=var_2)
    var_95 = [var_2]
    var_96 = module_0.Message(text=var_0, index=var_95)
    var_97 = [var_96]
    var_98 = module_0.BaseError(messages=var_97)
    var_99 = [var_22]
    var_100 = module_0.Message(text=var_0, index=var_99)
    var_101 = [var_100]
    var_102 = module_0.BaseError(messages=var_101)
    var_103 = module_0.BaseError(text=var_5, code=var_1, key=var_2)



# Parsed testcases at query #4
#--------------------------



def test_case_0():
    var_0 = 'Invalid JSON'
    var_1 = 'invalid_json'
    var_2 = module_0.ParseError(text=var_0, code=var_1)
    var_3 = module_0.Message(text=var_0, code=var_1)
    var_4 = [var_3]
    var_5 = 'field1'
    var_6 = module_0.Message(text=var_0, code=var_1, key=var_5)
    var_7 = 'Missing field'
    var_8 = 'missing_field'
    var_9 = 'field2'
    var_10 = module_0.Message(text=var_7, code=var_8, key=var_9)
    var_11 = [var_6, var_10]
    var_12 = module_0.ParseError(messages=var_11)
    var_13 = 'Invalid value'
    var_14 = 'invalid'
    var_15 = 'subfield'
    var_16 = [var_5, var_15]
    var_17 = module_0.Message(text=var_13, code=var_14, index=var_16)
    var_18 = 'missing'
    var_19 = [var_9]
    var_20 = module_0.Message(text=var_7, code=var_18, index=var_19)
    var_21 = [var_17, var_20]
    var_22 = module_0.ParseError(messages=var_21)
    var_23 = 'All tests passed!'
    var_24 = print(var_23)



# Parsed testcases at query #5
#--------------------------



def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = module_0.BaseError(text=var_0, code=var_1)
    var_4 = 'Different error message'
    var_5 = module_0.BaseError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.BaseError(text=var_0, code=var_6)
    var_8 = 'key'
    var_9 = module_0.BaseError(text=var_0, code=var_1, key=var_8)
    var_10 = 1
    var_11 = 0
    var_12 = module_0.Position(var_10, var_10, var_11)
    var_13 = module_0.BaseError(text=var_0, code=var_1, position=var_12)
    var_14 = 'Error 1'
    var_15 = 'code1'
    var_16 = module_0.Message(text=var_14, code=var_15)
    var_17 = 'Error 2'
    var_18 = 'code2'
    var_19 = module_0.Message(text=var_17, code=var_18)
    var_20 = [var_16, var_19]
    var_21 = module_0.Message(text=var_17, code=var_18)
    var_22 = module_0.Message(text=var_14, code=var_15)
    var_23 = [var_21, var_22]
    var_24 = module_0.BaseError(messages=var_20)
    var_25 = module_0.BaseError(messages=var_23)
    var_26 = module_0.Message(text=var_14, code=var_15)
    var_27 = [var_26]
    var_28 = module_0.BaseError(messages=var_27)
    var_29 = 'key1'
    var_30 = [var_29]
    var_31 = module_0.Message(text=var_14, code=var_15, index=var_30)
    var_32 = [var_31]
    var_33 = 'key2'
    var_34 = [var_33]
    var_35 = module_0.Message(text=var_14, code=var_15, index=var_34)
    var_36 = [var_35]
    var_37 = module_0.BaseError(messages=var_32)
    var_38 = module_0.BaseError(messages=var_36)
    var_39 = module_0.Position(var_10, var_10, var_11)
    var_40 = 5
    var_41 = 4
    var_42 = module_0.Position(var_10, var_40, var_41)
    var_43 = module_0.Message(text=var_14, code=var_15, start_position=var_39, end_position=var_42)
    var_44 = [var_43]
    var_45 = module_0.Message(text=var_14, code=var_15, start_position=var_39, end_position=var_42)
    var_46 = [var_45]
    var_47 = module_0.BaseError(messages=var_44)
    var_48 = module_0.BaseError(messages=var_46)
    var_49 = 2
    var_50 = 10
    var_51 = module_0.Position(var_49, var_10, var_50)
    var_52 = 14
    var_53 = module_0.Position(var_49, var_40, var_52)
    var_54 = module_0.Message(text=var_14, code=var_15, start_position=var_51, end_position=var_53)
    var_55 = [var_54]
    var_56 = module_0.BaseError(messages=var_55)
    var_57 = module_0.Message(text=var_14, code=var_15, start_position=var_39, end_position=var_53)
    var_58 = [var_57]
    var_59 = module_0.BaseError(messages=var_58)
    var_60 = module_0.Message(text=var_14, code=var_15, start_position=var_51, end_position=var_42)
    var_61 = [var_60]
    var_62 = module_0.BaseError(messages=var_61)
    var_63 = module_0.Message(text=var_14, code=var_15, start_position=var_39, end_position=var_42)
    var_64 = [var_63]
    var_65 = module_0.Message(text=var_14, code=var_15, start_position=var_39, end_position=var_42)
    var_66 = [var_65]
    var_67 = module_0.BaseError(messages=var_64)
    var_68 = module_0.BaseError(messages=var_66)
    var_69 = module_0.Message(text=var_14, code=var_15, start_position=var_39, end_position=var_42)
    var_70 = [var_69]
    var_71 = module_0.Message(text=var_14, code=var_15, start_position=var_39, end_position=var_42)
    var_72 = [var_71]
    var_73 = module_0.BaseError(messages=var_70)
    var_74 = module_0.BaseError(messages=var_72)
    var_75 = module_0.Message(text=var_14, code=var_15, start_position=var_39, end_position=var_42)
    var_76 = [var_75]
    var_77 = module_0.Message(text=var_14, code=var_15, start_position=var_39, end_position=var_42)
    var_78 = [var_77]
    var_79 = module_0.BaseError(messages=var_76)
    var_80 = module_0.BaseError(messages=var_78)
    var_81 = module_0.Message(text=var_14, code=var_15, start_position=var_39, end_position=var_42)
    var_82 = [var_81]
    var_83 = module_0.Message(text=var_14, code=var_15, start_position=var_39, end_position=var_42)
    var_84 = [var_83]
    var_85 = module_0.BaseError(messages=var_82)
    var_86 = module_0.BaseError(messages=var_84)
    var_87 = module_0.Message(text=var_14, code=var_15, start_position=var_39, end_position=var_42)
    var_88 = [var_87]
    var_89 = module_0.Message(text=var_14, code=var_15, start_position=var_39, end_position=var_42)
    var_90 = [var_89]
    var_91 = module_0.BaseError(messages=var_88)
    var_92 = module_0.BaseError(messages=var_90)
    var_93 = module_0.Message(text=var_14, code=var_15, start_position=var_39, end_position=var_42)
    var_94 = [var_93]
    var_95 = module_0.Message(text=var_14, code=var_15, start_position=var_39, end_position=var_42)
    var_96 = [var_95]
    var_97 = module_0.BaseError(messages=var_94)
    var_98 = module_0.BaseError(messages=var_96)
    var_99 = module_0.Message(text=var_14, code=var_15, start_position=var_39, end_position=var_42)
    var_100 = [var_99]
    var_101 = module_0.Message(text=var_14, code=var_15, start_position=var_39, end_position=var_42)
    var_102 = [var_101]
    var_103 = module_0.BaseError(messages=var_100)
    var_104 = module_0.BaseError(messages=var_102)
    var_105 = module_0.Message(text=var_14, code=var_15, start_position=var_39, end_position=var_42)
    var_106 = [var_105]
    var_107 = module_0.Message(text=var_14, code=var_15, start_position=var_39, end_position=var_42)
    var_108 = [var_107]
    var_109 = module_0.BaseError(messages=var_106)
    var_110 = module_0.BaseError(messages=var_108)
    var_111 = module_0.Message(text=var_14, code=var_15, start_position=var_39, end_position=var_42)
    var_112 = [var_111]
    var_113 = module_0.Message(text=var_14, code=var_15, start_position=var_39, end_position=var_42)
    var_114 = [var_113]
    var_115 = module_0.BaseError(messages=var_112)
    var_116 = module_0.BaseError(messages=var_114)
    var_117 = module_0.Message(text=var_14, code=var_15, start_position=var_39, end_position=var_42)
    var_118 = [var_117]



# Parsed testcases at query #6
#--------------------------



def test_case_0():
    var_0 = 'Invalid input'
    var_1 = 'invalid'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    var_3 = str(var_2)
    assert var_3 == 'Invalid input'
    var_4 = 'field1'
    var_5 = [var_4]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    var_7 = 'Missing field'
    var_8 = 'missing'
    var_9 = 'field2'
    var_10 = [var_9]
    var_11 = module_0.Message(text=var_7, code=var_8, index=var_10)
    var_12 = [var_6, var_11]
    var_13 = module_0.ValidationError(messages=var_12)
    var_14 = "{'field1': 'Invalid input', 'field2': 'Missing field'}"
    var_15 = str(var_13)
    var_16 = 'subfield'
    var_17 = [var_4, var_16]
    var_18 = module_0.Message(text=var_0, code=var_1, index=var_17)
    var_19 = [var_9]
    var_20 = module_0.Message(text=var_7, code=var_8, index=var_19)
    var_21 = [var_18, var_20]
    var_22 = module_0.ValidationError(messages=var_21)
    var_23 = "{'field1': {'subfield': 'Invalid input'}, 'field2': 'Missing field'}"
    var_24 = str(var_22)
    var_25 = []
    var_26 = module_0.ValidationError(messages=var_25)
    var_27 = str(var_26)
    assert var_27 == '{}'
    var_28 = module_0.ValidationError(text=var_0, code=var_1, key=var_4)
    var_29 = str(var_28)
    assert var_29 == "{'field1': 'Invalid input'}"
    var_30 = 1
    var_31 = 0
    var_32 = module_0.Position(var_30, var_30, var_31)
    var_33 = module_0.ValidationError(text=var_0, code=var_1, position=var_32)
    var_34 = str(var_33)
    assert var_34 == 'Invalid input'
    var_35 = module_0.Position(var_30, var_30, var_31)
    var_36 = 5
    var_37 = 4
    var_38 = module_0.Position(var_30, var_36, var_37)
    var_39 = module_0.ValidationError(text=var_0, code=var_1)
    var_40 = str(var_39)
    assert var_40 == 'Invalid input'
    var_41 = module_0.ValidationError(text=var_0, code=var_1)
    var_42 = str(var_41)
    assert var_42 == 'Invalid input'
    var_43 = module_0.ValidationError(text=var_0, key=var_4)
    var_44 = str(var_43)
    assert var_44 == "{'field1': 'Invalid input'}"
    var_45 = [var_4]
    var_46 = module_0.Message(text=var_0, index=var_45)
    var_47 = [var_46]
    var_48 = module_0.ValidationError(messages=var_47)
    var_49 = str(var_48)
    assert var_49 == "{'field1': 'Invalid input'}"
    var_50 = 'subfield1'
    var_51 = [var_4, var_50]
    var_52 = module_0.Message(text=var_0, index=var_51)
    var_53 = 'subfield2'
    var_54 = [var_4, var_53]
    var_55 = module_0.Message(text=var_7, index=var_54)
    var_56 = [var_52, var_55]
    var_57 = module_0.ValidationError(messages=var_56)
    var_58 = "{'field1': {'subfield1': 'Invalid input', 'subfield2': 'Missing field'}}"
    var_59 = str(var_57)
    var_60 = [var_31]
    var_61 = module_0.Message(text=var_0, index=var_60)
    var_62 = [var_61]
    var_63 = module_0.ValidationError(messages=var_62)
    var_64 = str(var_63)
    assert var_64 == "{0: 'Invalid input'}"
    var_65 = [var_4, var_31]
    var_66 = module_0.Message(text=var_0, index=var_65)
    var_67 = [var_66]
    var_68 = module_0.ValidationError(messages=var_67)
    var_69 = str(var_68)
    assert var_69 == "{'field1': {0: 'Invalid input'}}"
    var_70 = []
    var_71 = module_0.Message(text=var_0, index=var_70)
    var_72 = [var_71]
    var_73 = module_0.ValidationError(messages=var_72)
    var_74 = str(var_73)
    assert var_74 == 'Invalid input'
    var_75 = module_0.Message(text=var_0)
    var_76 = [var_75]
    var_77 = module_0.ValidationError(messages=var_76)
    var_78 = str(var_77)
    assert var_78 == 'Invalid input'
    var_79 = 'custom'
    var_80 = module_0.ValidationError(text=var_0, code=var_79)
    var_81 = str(var_80)
    assert var_81 == 'Invalid input'
    var_82 = 'max_length'
    var_83 = module_0.ValidationError(text=var_0, code=var_82)
    var_84 = str(var_83)
    assert var_84 == 'Invalid input'
    var_85 = [var_4, var_50]
    var_86 = module_0.Message(text=var_0, index=var_85)
    var_87 = [var_4, var_53]
    var_88 = module_0.Message(text=var_7, index=var_87)
    var_89 = 'Too long'
    var_90 = [var_9]
    var_91 = module_0.Message(text=var_89, index=var_90)
    var_92 = [var_86, var_88, var_91]
    var_93 = module_0.ValidationError(messages=var_92)
    var_94 = "{'field1': {'subfield1': 'Invalid input', 'subfield2': 'Missing field'}, 'field2': 'Too long'}"
    var_95 = str(var_93)
    var_96 = [var_4]
    var_97 = module_0.Message(text=var_0, index=var_96)
    var_98 = [var_4]
    var_99 = module_0.Message(text=var_7, index=var_98)
    var_100 = [var_97, var_99]
    var_101 = module_0.ValidationError(messages=var_100)
    var_102 = str(var_101)
    assert var_102 == "{'field1': 'Missing field'}"
    var_103 = 'a'
    var_104 = 'b'
    var_105 = 'c'
    var_106 = 'd'
    var_107 = [var_103, var_104, var_105, var_106]
    var_108 = module_0.Message(text=var_0, index=var_107)
    var_109 = [var_108]
    var_110 = module_0.ValidationError(messages=var_109)
    var_111 = str(var_110)
    assert var_111 == "{'a': {'b': {'c': {'d': 'Invalid input'}}}}"
    var_112 = 'All tests passed!'
    var_113 = print(var_112)



# Parsed testcases at query #7
#--------------------------



def test_case_0():
    var_0 = 42
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = 'Invalid input'
    var_3 = module_0.ValidationError(text=var_2)
    assert var_3 is None
    var_4 = module_0.ValidationResult(error=var_3)
    var_5 = 'name'
    var_6 = 'age'
    var_7 = 'John'
    var_8 = 30
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = module_0.ValidationResult(value=var_9)
    var_11 = 'Error 1'
    var_12 = module_0.Message(text=var_11)
    var_13 = 'Error 2'
    var_14 = module_0.Message(text=var_13)
    var_15 = [var_12, var_14]
    var_16 = module_0.ValidationError(messages=var_15)
    var_17 = module_0.ValidationResult(error=var_16)
    var_18 = 1
    var_19 = 2
    var_20 = 3
    var_21 = [var_18, var_19, var_20]
    var_22 = module_0.ValidationResult(value=var_21)
    var_23 = 'Error A'
    var_24 = module_0.Message(text=var_23)
    var_25 = 'Error B'
    var_26 = module_0.Message(text=var_25)
    var_27 = [var_24, var_26]
    var_28 = module_0.ValidationError(messages=var_27)
    assert var_28 is None
    var_29 = module_0.ValidationResult(error=var_28)
    var_30 = 'Hello, World!'
    var_31 = module_0.ValidationResult(value=var_30)
    var_32 = 'Single error'
    var_33 = module_0.ValidationError(text=var_32)
    var_34 = module_0.ValidationResult(error=var_33)
    var_35 = 'All test cases passed!'
    var_36 = print(var_35)



# Parsed testcases at query #8
#--------------------------



def test_case_0():
    var_0 = 'test'
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = 'error'
    var_3 = module_0.ValidationError(text=var_2)
    assert var_3 is None
    var_4 = module_0.ValidationResult(error=var_3)
    var_5 = module_0.ValidationResult(value=var_0, error=var_3)
    var_6 = module_0.ValidationResult()
    var_7 = [var_0]
    var_8 = module_0.ValidationResult(value=var_7)
    var_9 = 'key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = module_0.ValidationResult(value=var_11)
    var_13 = (var_0,)
    var_14 = module_0.ValidationResult(value=var_13)
    var_15 = {var_0}
    var_16 = module_0.ValidationResult(value=var_15)
    var_17 = {var_0}
    var_18 = frozenset(var_17)
    var_19 = module_0.ValidationResult(value=var_18)
    var_20 = {var_0}
    var_21 = frozenset(var_20)
    var_22 = 10
    var_23 = range(var_22)
    var_24 = module_0.ValidationResult(value=var_23)
    var_25 = range(var_22)
    var_26 = b'test'
    var_27 = module_0.ValidationResult(value=var_26)
    var_28 = bytearray(var_26)
    var_29 = module_0.ValidationResult(value=var_28)
    var_30 = bytearray(var_26)
    var_31 = memoryview(var_26)
    var_32 = module_0.ValidationResult(value=var_31)
    var_33 = memoryview(var_26)
    var_34 = 1
    var_35 = 2
    var_36 = complex(var_34, var_35)
    var_37 = module_0.ValidationResult(value=var_36)
    var_38 = complex(var_34, var_35)
    var_39 = True
    var_40 = module_0.ValidationResult(value=var_39)
    var_41 = None
    var_42 = module_0.ValidationResult(value=var_41)
    var_43 = module_0.ValidationResult(value=var_39)
    var_44 = module_0.ValidationResult(value=var_39)
    var_45 = module_0.ValidationResult(value=var_0)
    var_46 = [var_0]
    var_47 = [var_46]
    var_48 = module_0.ValidationResult(value=var_47)
    var_49 = 'nested'
    var_50 = {var_49: var_10}
    var_51 = {var_9: var_50}
    var_52 = module_0.ValidationResult(value=var_51)
    var_53 = (var_0,)
    var_54 = (var_53,)
    var_55 = module_0.ValidationResult(value=var_54)
    var_56 = {var_0}
    var_57 = frozenset(var_56)
    var_58 = {var_57}
    var_59 = module_0.ValidationResult(value=var_58)
    var_60 = {var_0}
    var_61 = frozenset(var_60)
    var_62 = {var_61}
    var_63 = range(var_22)
    var_64 = module_0.ValidationResult(value=var_63)
    var_65 = range(var_22)
    var_66 = module_0.ValidationResult(value=var_26)
    var_67 = bytearray(var_26)
    var_68 = module_0.ValidationResult(value=var_67)
    var_69 = bytearray(var_26)
    var_70 = memoryview(var_26)
    var_71 = module_0.ValidationResult(value=var_70)
    var_72 = memoryview(var_26)
    var_73 = complex(var_39, var_35)
    var_74 = module_0.ValidationResult(value=var_73)
    var_75 = complex(var_39, var_35)
    var_76 = True
    var_77 = module_0.ValidationResult(value=var_76)
    var_78 = module_0.ValidationResult(value=var_41)
    var_79 = module_0.ValidationResult(value=var_76)
    var_80 = module_0.ValidationResult(value=var_76)
    var_81 = module_0.ValidationResult(value=var_0)
    var_82 = {var_9: var_10}
    var_83 = [var_82]
    var_84 = module_0.ValidationResult(value=var_83)
    var_85 = [var_10]
    var_86 = {var_9: var_85}
    var_87 = module_0.ValidationResult(value=var_86)
    var_88 = {var_0}
    var_89 = (var_88,)
    var_90 = module_0.ValidationResult(value=var_89)
    var_91 = (var_0,)
    var_92 = {var_91}
    var_93 = module_0.ValidationResult(value=var_92)



# Parsed testcases at query #9
#--------------------------



def test_case_0():
    var_0 = 'Error 1'
    var_1 = 'code1'
    var_2 = 'key1'
    var_3 = module_0.BaseError(text=var_0, code=var_1, key=var_2)
    var_4 = module_0.BaseError(text=var_0, code=var_1, key=var_2)
    var_5 = 'Error 2'
    var_6 = 'code2'
    var_7 = 'key2'
    var_8 = module_0.BaseError(text=var_5, code=var_6, key=var_7)
    var_9 = module_0.Message(text=var_0, code=var_1)
    var_10 = module_0.Message(text=var_5, code=var_6)
    var_11 = [var_9, var_10]
    var_12 = module_0.BaseError(messages=var_11)
    var_13 = module_0.BaseError(messages=var_11)
    var_14 = module_0.Message(text=var_0, code=var_1)
    var_15 = [var_14]
    var_16 = module_0.BaseError(messages=var_15)
    var_17 = module_0.Message(text=var_5, code=var_6)
    var_18 = module_0.Message(text=var_0, code=var_1)
    var_19 = [var_17, var_18]
    var_20 = module_0.BaseError(messages=var_19)
    var_21 = [var_2]
    var_22 = module_0.Message(text=var_0, code=var_1, index=var_21)
    var_23 = [var_22]
    var_24 = module_0.BaseError(messages=var_23)
    var_25 = module_0.Message(text=var_0, code=var_1)
    var_26 = [var_25]
    var_27 = module_0.BaseError(messages=var_26)
    var_28 = 1
    var_29 = 0
    var_30 = module_0.Position(var_28, var_28, var_29)
    var_31 = 2
    var_32 = 10
    var_33 = module_0.Position(var_31, var_31, var_32)
    var_34 = module_0.Message(text=var_0, code=var_1, position=var_30)
    var_35 = [var_34]
    var_36 = module_0.BaseError(messages=var_35)
    var_37 = module_0.Message(text=var_0, code=var_1)
    var_38 = [var_37]
    var_39 = module_0.BaseError(messages=var_38)
    var_40 = module_0.Message(text=var_0, code=var_1, start_position=var_30, end_position=var_33)
    var_41 = [var_40]
    var_42 = module_0.BaseError(messages=var_41)
    var_43 = module_0.Message(text=var_0, code=var_1)
    var_44 = [var_43]
    var_45 = module_0.BaseError(messages=var_44)
    var_46 = module_0.BaseError(text=var_0, code=var_1)
    var_47 = module_0.BaseError(text=var_0, code=var_6)
    var_48 = module_0.BaseError(text=var_0, code=var_1, key=var_2)
    var_49 = module_0.BaseError(text=var_0, code=var_1, key=var_7)
    var_50 = module_0.BaseError(text=var_0, code=var_1)
    var_51 = module_0.BaseError(text=var_5, code=var_1)
    var_52 = [var_2, var_7]
    var_53 = module_0.Message(text=var_0, code=var_1, index=var_52)
    var_54 = [var_53]
    var_55 = [var_2]
    var_56 = module_0.Message(text=var_0, code=var_1, index=var_55)
    var_57 = [var_56]
    var_58 = module_0.BaseError(messages=var_54)
    var_59 = module_0.BaseError(messages=var_57)
    var_60 = [var_2]
    var_61 = module_0.Message(text=var_0, code=var_1, index=var_60)
    var_62 = [var_61]
    var_63 = [var_7]
    var_64 = module_0.Message(text=var_0, code=var_1, index=var_63)
    var_65 = [var_64]
    var_66 = module_0.BaseError(messages=var_62)
    var_67 = module_0.BaseError(messages=var_65)
    var_68 = 3
    var_69 = 20
    var_70 = module_0.Position(var_68, var_68, var_69)
    var_71 = module_0.Message(text=var_0, code=var_1, position=var_30)
    var_72 = [var_71]
    var_73 = module_0.Message(text=var_0, code=var_1, position=var_70)
    var_74 = [var_73]
    var_75 = module_0.BaseError(messages=var_72)
    var_76 = module_0.BaseError(messages=var_74)
    var_77 = 4
    var_78 = 30
    var_79 = module_0.Position(var_77, var_77, var_78)
    var_80 = module_0.Message(text=var_0, code=var_1, start_position=var_30, end_position=var_33)
    var_81 = [var_80]
    var_82 = module_0.Message(text=var_0, code=var_1, start_position=var_70, end_position=var_79)
    var_83 = [var_82]
    var_84 = module_0.BaseError(messages=var_81)
    var_85 = module_0.BaseError(messages=var_83)
    var_86 = module_0.Message(text=var_0, code=var_1)
    var_87 = [var_86]
    var_88 = module_0.Message(text=var_0, code=var_6)
    var_89 = [var_88]
    var_90 = module_0.BaseError(messages=var_87)
    var_91 = module_0.BaseError(messages=var_89)
    var_92 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_93 = [var_92]
    var_94 = module_0.Message(text=var_0, code=var_1, key=var_7)
    var_95 = [var_94]
    var_96 = module_0.BaseError(messages=var_93)
    var_97 = module_0.BaseError(messages=var_95)
    var_98 = module_0.Message(text=var_0, code=var_1)
    var_99 = [var_98]
    var_100 = module_0.Message(text=var_5, code=var_1)
    var_101 = [var_100]
    var_102 = module_0.BaseError(messages=var_99)
    var_103 = module_0.BaseError(messages=var_101)
    var_104 = [var_2, var_7]
    var_105 = module_0.Message(text=var_0, code=var_1, index=var_104)
    var_106 = [var_105]
    var_107 = 'key3'
    var_108 = [var_107]
    var_109 = module_0.Message(text=var_0, code=var_1, index=var_108)
    var_110 = [var_109]
    var_111 = module_0.BaseError(messages=var_106)
    var_112 = module_0.BaseError(messages=var_110)
    var_113 = module_0.Message(text=var_0, code=var_1, position=var_30)
    var_114 = [var_113]
    var_115 = module_0.Message(text=var_0, code=var_1, start_position=var_70, end_position=var_79)
    var_116 = [var_115]
    var_117 = module_0.BaseError(messages=var_114)
    var_118 = module_0.BaseError(messages=var_116)
    var_119 = module_0.Message(text=var_0, code=var_1, key=var_2)
    var_120 = [var_119]
    var_121 = module_0.Message(text=var_0, code=var_6, key=var_7)
    var_122 = [var_121]
    var_123 = module_0.BaseError(messages=var_120)
    var_124 = module_0.BaseError(messages=var_122)



# Parsed testcases at query #10
#--------------------------



def test_case_0():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    var_3 = module_0.BaseError(text=var_0, code=var_1)
    var_4 = 'Different error message'
    var_5 = module_0.BaseError(text=var_4, code=var_1)
    var_6 = 'different_code'
    var_7 = module_0.BaseError(text=var_0, code=var_6)
    var_8 = 'key'
    var_9 = module_0.BaseError(text=var_0, code=var_1, key=var_8)
    var_10 = 1
    var_11 = 0
    var_12 = module_0.Position(var_10, var_10, var_11)
    var_13 = 2
    var_14 = 10
    var_15 = module_0.Position(var_13, var_13, var_14)
    var_16 = module_0.BaseError(text=var_0, code=var_1, position=var_12)
    var_17 = module_0.BaseError(text=var_0, code=var_1, position=var_15)
    var_18 = 'Message 1'
    var_19 = 'code1'
    var_20 = module_0.Message(text=var_18, code=var_19)
    var_21 = 'Message 2'
    var_22 = 'code2'
    var_23 = module_0.Message(text=var_21, code=var_22)
    var_24 = [var_20, var_23]
    var_25 = module_0.Message(text=var_18, code=var_19)
    var_26 = [var_25]
    var_27 = module_0.BaseError(messages=var_24)
    var_28 = module_0.BaseError(messages=var_26)
    var_29 = module_0.Message(text=var_18, code=var_19)
    var_30 = module_0.Message(text=var_21, code=var_22)
    var_31 = [var_29, var_30]
    var_32 = module_0.Message(text=var_21, code=var_22)
    var_33 = module_0.Message(text=var_18, code=var_19)
    var_34 = [var_32, var_33]
    var_35 = module_0.BaseError(messages=var_31)
    var_36 = module_0.BaseError(messages=var_34)
    var_37 = module_0.Message(text=var_18, code=var_19)
    var_38 = module_0.Message(text=var_21, code=var_22)
    var_39 = [var_37, var_38]
    var_40 = module_0.Message(text=var_18, code=var_19)
    var_41 = 'Different message'
    var_42 = module_0.Message(text=var_41, code=var_22)
    var_43 = [var_40, var_42]
    var_44 = module_0.BaseError(messages=var_39)
    var_45 = module_0.BaseError(messages=var_43)
    var_46 = module_0.Message(text=var_18, code=var_19)
    var_47 = module_0.Message(text=var_21, code=var_22)
    var_48 = [var_46, var_47]
    var_49 = module_0.Message(text=var_18, code=var_19)
    var_50 = module_0.Message(text=var_21, code=var_6)
    var_51 = [var_49, var_50]
    var_52 = module_0.BaseError(messages=var_48)
    var_53 = module_0.BaseError(messages=var_51)
    var_54 = 'key1'
    var_55 = module_0.Message(text=var_18, code=var_19, key=var_54)
    var_56 = 'key2'
    var_57 = module_0.Message(text=var_21, code=var_22, key=var_56)
    var_58 = [var_55, var_57]
    var_59 = module_0.Message(text=var_18, code=var_19, key=var_54)
    var_60 = 'different_key'
    var_61 = module_0.Message(text=var_21, code=var_22, key=var_60)
    var_62 = [var_59, var_61]
    var_63 = module_0.BaseError(messages=var_58)
    var_64 = module_0.BaseError(messages=var_62)
    var_65 = module_0.Position(var_10, var_10, var_11)
    var_66 = module_0.Position(var_13, var_13, var_14)
    var_67 = module_0.Message(text=var_18, code=var_19, position=var_65)
    var_68 = module_0.Message(text=var_21, code=var_22, position=var_66)
    var_69 = [var_67, var_68]
    var_70 = module_0.Message(text=var_18, code=var_19, position=var_65)
    var_71 = module_0.Message(text=var_21, code=var_22, position=var_65)
    var_72 = [var_70, var_71]
    var_73 = module_0.BaseError(messages=var_69)
    var_74 = module_0.BaseError(messages=var_72)
    var_75 = module_0.Position(var_10, var_10, var_11)
    var_76 = 5
    var_77 = 4
    var_78 = module_0.Position(var_10, var_76, var_77)
    var_79 = module_0.Position(var_13, var_10, var_14)
    var_80 = 14
    var_81 = module_0.Position(var_13, var_76, var_80)
    var_82 = module_0.Message(text=var_18, code=var_19, start_position=var_75, end_position=var_78)
    var_83 = module_0.Message(text=var_21, code=var_22, start_position=var_79, end_position=var_81)
    var_84 = [var_82, var_83]
    var_85 = module_0.Message(text=var_18, code=var_19, start_position=var_75, end_position=var_78)
    var_86 = module_0.Message(text=var_21, code=var_22, start_position=var_75, end_position=var_78)
    var_87 = [var_85, var_86]
    var_88 = module_0.BaseError(messages=var_84)
    var_89 = module_0.BaseError(messages=var_87)
    var_90 = module_0.Message(text=var_18, code=var_19, start_position=var_75, end_position=var_78)
    var_91 = module_0.Message(text=var_21, code=var_22, start_position=var_79, end_position=var_81)
    var_92 = [var_90, var_91]
    var_93 = module_0.Message(text=var_18, code=var_19, start_position=var_79, end_position=var_81)
    var_94 = module_0.Message(text=var_21, code=var_22, start_position=var_75, end_position=var_78)
    var_95 = [var_93, var_94]
    var_96 = module_0.BaseError(messages=var_92)
    var_97 = module_0.BaseError(messages=var_95)
    var_98 = module_0.Message(text=var_18, code=var_19, start_position=var_75, end_position=var_78)
    var_99 = module_0.Message(text=var_21, code=var_22, start_position=var_79, end_position=var_81)
    var_100 = [var_98, var_99]
    var_101 = module_0.Message(text=var_18, code=var_19, start_position=var_78, end_position=var_75)
    var_102 = module_0.Message(text=var_21, code=var_22, start_position=var_81, end_position=var_79)
    var_103 = [var_101, var_102]
    var_104 = module_0.BaseError(messages=var_100)
    var_105 = module_0.BaseError(messages=var_103)
    var_106 = module_0.Message(text=var_18, code=var_19, start_position=var_75, end_position=var_78)
    var_107 = module_0.Message(text=var_21, code=var_22, start_position=var_79, end_position=var_81)
    var_108 = [var_106, var_107]
    var_109 = module_0.Message(text=var_18, code=var_19, start_position=var_81, end_position=var_79)
    var_110 = module_0.Message(text=var_21, code=var_22, start_position=var_78, end_position=var_75)
    var_111 = [var_109, var_110]
    var_112 = module_0.BaseError(messages=var_108)
    var_113 = module_0.BaseError(messages=var_111)
    var_114 = module_0.Message(text=var_18, code=var_19, start_position=var_75, end_position=var_78)
    var_115 = module_0.Message(text=var_21, code=var_22, start_position=var_79, end_position=var_81)
    var_116 = [var_114, var_115]
    var_117 = module_0.Message(text=var_21, code=var_22, start_position=var_81, end_position=var_79)
    var_118 = module_0.Message(text=var_18, code=var_19, start_position=var_78, end_position=var_75)
    var_119 = [var_117, var_118]
    var_120 = module_0.BaseError(messages=var_116)
    var_121 = module_0.BaseError(messages=var_119)



# Parsed testcases at query #11
#--------------------------



def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    var_4 = module_0.Position(var_0, var_1, var_2)
    var_5 = module_0.Position(var_0, var_1, var_2)
    var_6 = module_0.Position(var_1, var_1, var_2)
    var_7 = module_0.Position(var_0, var_1, var_2)
    var_8 = module_0.Position(var_0, var_2, var_2)
    var_9 = module_0.Position(var_0, var_1, var_2)
    var_10 = 4
    var_11 = module_0.Position(var_0, var_1, var_10)
    var_12 = module_0.Position(var_0, var_1, var_2)
    var_13 = module_0.Position(var_0, var_1, var_2)
    var_14 = module_0.Position(var_0, var_1, var_2)
    var_15 = module_0.Position(var_0, var_1, var_2)
    var_16 = module_0.Position(var_0, var_1, var_10)
    var_17 = module_0.Position(var_0, var_1, var_2)
    var_18 = module_0.Position(var_0, var_2, var_2)
    var_19 = module_0.Position(var_0, var_1, var_2)
    var_20 = module_0.Position(var_1, var_1, var_2)
    var_21 = module_0.Position(var_0, var_1, var_2)
    var_22 = 5
    var_23 = 6
    var_24 = module_0.Position(var_10, var_22, var_23)
    var_25 = module_0.Position(var_0, var_1, var_2)
    var_26 = module_0.Position(var_0, var_1, var_2)
    var_27 = module_0.Position(var_0, var_1, var_2)
    var_28 = module_0.Position(var_1, var_0, var_2)
    var_29 = module_0.Position(var_0, var_1, var_2)
    var_30 = -1
    var_31 = -2
    var_32 = -3
    var_33 = module_0.Position(var_30, var_31, var_32)
    var_34 = module_0.Position(var_0, var_1, var_2)
    var_35 = 10
    var_36 = 20
    var_37 = 30
    var_38 = module_0.Position(var_35, var_36, var_37)
    var_39 = module_0.Position(var_0, var_1, var_2)
    var_40 = module_0.Position(var_0, var_1, var_2)
    var_41 = module_0.Position(var_0, var_1, var_2)
    var_42 = module_0.Position(var_0, var_1, var_2)
    var_43 = module_0.Position(var_0, var_1, var_2)
    var_44 = module_0.Position(var_0, var_1, var_2)
    var_45 = module_0.Position(var_0, var_1, var_2)



# Parsed testcases at query #12
#--------------------------



def test_case_0():
    var_0 = 42
    var_1 = module_0.ValidationResult(value=var_0)
    var_2 = 'Invalid data'
    var_3 = module_0.ValidationError(text=var_2)
    assert var_3 is None
    var_4 = module_0.ValidationResult(error=var_3)
    var_5 = 'name'
    var_6 = 'age'
    var_7 = 'John'
    var_8 = 30
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = module_0.ValidationResult(value=var_9)
    var_11 = 'invalid'
    var_12 = module_0.ValidationError(text=var_2, code=var_11)
    assert var_12 is None
    var_13 = module_0.ValidationResult(error=var_12)
    var_14 = 1
    var_15 = 2
    var_16 = 3
    var_17 = [var_14, var_15, var_16]
    var_18 = module_0.ValidationResult(value=var_17)
    var_19 = module_0.ValidationError(text=var_2, code=var_11)
    assert var_19 is None
    var_20 = module_0.ValidationResult(error=var_19)
    var_21 = (var_14, var_15, var_16)
    var_22 = module_0.ValidationResult(value=var_21)
    var_23 = module_0.ValidationError(text=var_2, code=var_11)
    assert var_23 is None
    var_24 = module_0.ValidationResult(error=var_23)
    var_25 = {var_14, var_15, var_16}
    var_26 = module_0.ValidationResult(value=var_25)
    var_27 = module_0.ValidationError(text=var_2, code=var_11)
    assert var_27 is None
    var_28 = module_0.ValidationResult(error=var_27)
    var_29 = 'key'
    var_30 = 'value'
    var_31 = {var_29: var_30}
    var_32 = module_0.ValidationResult(value=var_31)
    var_33 = module_0.ValidationError(text=var_2, code=var_11)
    assert var_33 is None
    var_34 = module_0.ValidationResult(error=var_33)
    var_35 = 'Hello, world!'
    var_36 = module_0.ValidationResult(value=var_35)
    var_37 = module_0.ValidationError(text=var_2, code=var_11)
    assert var_37 is None
    var_38 = module_0.ValidationResult(error=var_37)
    var_39 = module_0.ValidationResult(value=var_0)
    var_40 = module_0.ValidationError(text=var_2, code=var_11)
    assert var_40 is None
    var_41 = module_0.ValidationResult(error=var_40)
    var_42 = 3.14
    var_43 = module_0.ValidationResult(value=var_42)
    var_44 = module_0.ValidationError(text=var_2, code=var_11)
    assert var_44 is None
    var_45 = module_0.ValidationResult(error=var_44)
    var_46 = True
    var_47 = module_0.ValidationResult(value=var_46)
    var_48 = module_0.ValidationError(text=var_2, code=var_11)
    assert var_48 is None
    var_49 = module_0.ValidationResult(error=var_48)
    var_50 = module_0.ValidationError(text=var_2, code=var_11)
    assert var_50 is None
    var_51 = module_0.ValidationResult(error=var_50)
    var_52 = module_0.ValidationError(text=var_2, code=var_11)
    assert var_52 is None
    var_53 = module_0.ValidationResult(error=var_52)
    var_54 = module_0.ValidationError(text=var_2, code=var_11)
    assert var_54 is None
    var_55 = module_0.ValidationResult(error=var_54)
    var_56 = module_0.ValidationError(text=var_2, code=var_11)
    assert var_56 is None
    var_57 = module_0.ValidationResult(error=var_56)
    var_58 = module_0.ValidationError(text=var_2, code=var_11)
    assert var_58 is None
    var_59 = module_0.ValidationResult(error=var_58)
    var_60 = module_0.ValidationError(text=var_2, code=var_11)
    assert var_60 is None
    var_61 = module_0.ValidationResult(error=var_60)



