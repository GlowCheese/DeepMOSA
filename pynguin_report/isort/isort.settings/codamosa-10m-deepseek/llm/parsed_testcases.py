####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is True
    var_3 = 'test.txt'
    var_4 = var_0.is_supported_filetype(var_3)
    assert var_4 is False
    var_5 = 'test.py~'
    var_6 = var_0.is_supported_filetype(var_5)
    assert var_6 is False
    var_7 = 'test.pyo'
    var_8 = var_0.is_supported_filetype(var_7)
    assert var_8 is False
    var_9 = 'test.pyc'
    var_10 = var_0.is_supported_filetype(var_9)
    assert var_10 is False
    var_11 = 'test.pyd'
    var_12 = var_0.is_supported_filetype(var_11)
    assert var_12 is False
    var_13 = 'test.pxe'
    var_14 = var_0.is_supported_filetype(var_13)
    assert var_14 is False
    var_15 = 'test.pxi'
    var_16 = var_0.is_supported_filetype(var_15)
    assert var_16 is False
    var_17 = 'test.pyi'
    var_18 = var_0.is_supported_filetype(var_17)
    assert var_18 is False
    var_19 = 'test.pyw'
    var_20 = var_0.is_supported_filetype(var_19)
    assert var_20 is False
    var_21 = 'test.pyx'
    var_22 = var_0.is_supported_filetype(var_21)
    assert var_22 is False
    var_23 = 'test.pyz'
    var_24 = var_0.is_supported_filetype(var_23)
    assert var_24 is False
    var_25 = 'test.pywz'
    var_26 = var_0.is_supported_filetype(var_25)
    assert var_26 is False
    var_27 = 'test.py3'
    var_28 = var_0.is_supported_filetype(var_27)
    assert var_28 is False
    var_29 = 'test.py2'
    var_30 = var_0.is_supported_filetype(var_29)
    assert var_30 is False
    var_31 = 'test.py1'
    var_32 = var_0.is_supported_filetype(var_31)
    assert var_32 is False
    var_33 = 'test.py0'
    var_34 = var_0.is_supported_filetype(var_33)
    assert var_34 is False
    var_35 = 'test.pyx~'
    var_36 = var_0.is_supported_filetype(var_35)
    assert var_36 is False
    var_37 = 'test.pyc~'
    var_38 = var_0.is_supported_filetype(var_37)
    assert var_38 is False
    var_39 = 'test.pyo~'
    var_40 = var_0.is_supported_filetype(var_39)
    assert var_40 is False
    var_41 = 'test.pyd~'
    var_42 = var_0.is_supported_filetype(var_41)
    assert var_42 is False
    var_43 = 'test.pxe~'
    var_44 = var_0.is_supported_filetype(var_43)
    assert var_44 is False
    var_45 = 'test.pxi~'
    var_46 = var_0.is_supported_filetype(var_45)
    assert var_46 is False
    var_47 = 'test.pyi~'
    var_48 = var_0.is_supported_filetype(var_47)
    assert var_48 is False
    var_49 = 'test.pyw~'
    var_50 = var_0.is_supported_filetype(var_49)
    assert var_50 is False
    var_51 = var_0.is_supported_filetype(var_35)
    assert var_51 is False
    var_52 = 'test.pyz~'
    var_53 = var_0.is_supported_filetype(var_52)
    assert var_53 is False
    var_54 = 'test.pywz~'
    var_55 = var_0.is_supported_filetype(var_54)
    assert var_55 is False
    var_56 = 'test.py3~'
    var_57 = var_0.is_supported_filetype(var_56)
    assert var_57 is False
    var_58 = 'test.py2~'
    var_59 = var_0.is_supported_filetype(var_58)
    assert var_59 is False
    var_60 = 'test.py1~'
    var_61 = var_0.is_supported_filetype(var_60)
    assert var_61 is False
    var_62 = 'test.py0~'
    var_63 = var_0.is_supported_filetype(var_62)
    assert var_63 is False
    var_64 = var_0.is_supported_filetype(var_5)
    assert var_64 is False
    var_65 = var_0.is_supported_filetype(var_37)
    assert var_65 is False
    var_66 = var_0.is_supported_filetype(var_39)
    assert var_66 is False
    var_67 = var_0.is_supported_filetype(var_41)
    assert var_67 is False
    var_68 = var_0.is_supported_filetype(var_43)
    assert var_68 is False
    var_69 = var_0.is_supported_filetype(var_45)
    assert var_69 is False
    var_70 = var_0.is_supported_filetype(var_47)
    assert var_70 is False
    var_71 = var_0.is_supported_filetype(var_49)
    assert var_71 is False
    var_72 = var_0.is_supported_filetype(var_35)
    assert var_72 is False
    var_73 = var_0.is_supported_filetype(var_52)
    assert var_73 is False
    var_74 = var_0.is_supported_filetype(var_54)
    assert var_74 is False
    var_75 = var_0.is_supported_filetype(var_56)
    assert var_75 is False
    var_76 = var_0.is_supported_filetype(var_58)
    assert var_76 is False
    var_77 = var_0.is_supported_filetype(var_60)
    assert var_77 is False
    var_78 = var_0.is_supported_filetype(var_62)
    assert var_78 is False
    var_79 = var_0.is_supported_filetype(var_5)
    assert var_79 is False
    var_80 = var_0.is_supported_filetype(var_37)
    assert var_80 is False
    var_81 = var_0.is_supported_filetype(var_39)
    assert var_81 is False
    var_82 = var_0.is_supported_filetype(var_41)
    assert var_82 is False
    var_83 = var_0.is_supported_filetype(var_43)
    assert var_83 is False
    var_84 = var_0.is_supported_filetype(var_45)
    assert var_84 is False
    var_85 = var_0.is_supported_filetype(var_47)
    assert var_85 is False
    var_86 = var_0.is_supported_filetype(var_49)
    assert var_86 is False
    var_87 = var_0.is_supported_filetype(var_35)
    assert var_87 is False
    var_88 = var_0.is_supported_filetype(var_52)
    assert var_88 is False
    var_89 = var_0.is_supported_filetype(var_54)
    assert var_89 is False
    var_90 = var_0.is_supported_filetype(var_56)
    assert var_90 is False
    var_91 = var_0.is_supported_filetype(var_58)
    assert var_91 is False
    var_92 = var_0.is_supported_filetype(var_60)
    assert var_92 is False
    var_93 = var_0.is_supported_filetype(var_62)
    assert var_93 is False
    var_94 = var_0.is_supported_filetype(var_5)
    assert var_94 is False
    var_95 = var_0.is_supported_filetype(var_37)
    assert var_95 is False
    var_96 = var_0.is_supported_filetype(var_39)
    assert var_96 is False
    var_97 = var_0.is_supported_filetype(var_41)
    assert var_97 is False
    var_98 = var_0.is_supported_filetype(var_43)
    assert var_98 is False
    var_99 = var_0.is_supported_filetype(var_45)
    assert var_99 is False
    var_100 = var_0.is_supported_filetype(var_47)
    assert var_100 is False
    var_101 = var_0.is_supported_filetype(var_49)
    assert var_101 is False
    var_102 = var_0.is_supported_filetype(var_35)
    assert var_102 is False
    var_103 = var_0.is_supported_filetype(var_52)
    assert var_103 is False
    var_104 = var_0.is_supported_filetype(var_54)
    assert var_104 is False
    var_105 = var_0.is_supported_filetype(var_56)
    assert var_105 is False
    var_106 = var_0.is_supported_filetype(var_58)
    assert var_106 is False
    var_107 = var_0.is_supported_filetype(var_60)
    assert var_107 is False
    var_108 = var_0.is_supported_filetype(var_62)
    assert var_108 is False
    var_109 = var_0.is_supported_filetype(var_5)
    assert var_109 is False
    var_110 = var_0.is_supported_filetype(var_37)
    assert var_110 is False
    var_111 = var_0.is_supported_filetype(var_39)
    assert var_111 is False
    var_112 = var_0.is_supported_filetype(var_41)
    assert var_112 is False
    var_113 = var_0.is_supported_filetype(var_43)
    assert var_113 is False
    var_114 = var_0.is_supported_filetype(var_45)
    assert var_114 is False
    var_115 = var_0.is_supported_filetype(var_47)
    assert var_115 is False
    var_116 = var_0.is_supported_filetype(var_49)
    assert var_116 is False
    var_117 = var_0.is_supported_filetype(var_35)
    assert var_117 is False
    var_118 = var_0.is_supported_filetype(var_52)
    assert var_118 is False
    var_119 = var_0.is_supported_filetype(var_54)
    assert var_119 is False
    var_120 = var_0.is_supported_filetype(var_56)
    assert var_120 is False
    var_121 = var_0.is_supported_filetype(var_58)
    assert var_121 is False
    var_122 = var_0.is_supported_filetype(var_60)
    assert var_122 is False
    var_123 = var_0.is_supported_filetype(var_62)
    assert var_123 is False
    var_124 = var_0.is_supported_filetype(var_5)
    assert var_124 is False
    var_125 = var_0.is_supported_filetype(var_37)
    assert var_125 is False
    var_126 = var_0.is_supported_filetype(var_39)
    assert var_126 is False
    var_127 = var_0.is_supported_filetype(var_41)
    assert var_127 is False
    var_128 = var_0.is_supported_filetype(var_43)
    assert var_128 is False
    var_129 = var_0.is_supported_filetype(var_45)
    assert var_129 is False
    var_130 = var_0.is_supported_filetype(var_47)
    assert var_130 is False
    var_131 = var_0.is_supported_filetype(var_49)
    assert var_131 is False
    var_132 = var_0.is_supported_filetype(var_35)
    assert var_132 is False
    var_133 = var_0.is_supported_filetype(var_52)
    assert var_133 is False
    var_134 = var_0.is_supported_filetype(var_54)
    assert var_134 is False
    var_135 = var_0.is_supported_filetype(var_56)
    assert var_135 is False
    var_136 = var_0.is_supported_filetype(var_58)
    assert var_136 is False
    var_137 = var_0.is_supported_filetype(var_60)
    assert var_137 is False
    var_138 = var_0.is_supported_filetype(var_62)
    assert var_138 is False
    var_139 = var_0.is_supported_filetype(var_5)
    assert var_139 is False
    var_140 = var_0.is_supported_filetype(var_37)
    assert var_140 is False
    var_141 = var_0.is_supported_filetype(var_39)
    assert var_141 is False
    var_142 = var_0.is_supported_filetype(var_41)
    assert var_142 is False
    var_143 = var_0.is_supported_filetype(var_43)
    assert var_143 is False
    var_144 = var_0.is_supported_filetype(var_45)
    assert var_144 is False
    var_145 = var_0.is_supported_filetype(var_47)
    assert var_145 is False
    var_146 = var_0.is_supported_filetype(var_49)
    assert var_146 is False
    var_147 = var_0.is_supported_filetype(var_35)
    assert var_147 is False
    var_148 = var_0.is_supported_filetype(var_52)
    assert var_148 is False
    var_149 = var_0.is_supported_filetype(var_54)
    assert var_149 is False
    var_150 = var_0.is_supported_filetype(var_56)
    assert var_150 is False
    var_151 = var_0.is_supported_filetype(var_58)
    assert var_151 is False
    var_152 = var_0.is_supported_filetype(var_60)
    assert var_152 is False



# Parsed testcases at query #2
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test the constructor of the Config class.'
    var_1 = module_0.Config()
    var_2 = 'tests/example_settings_file.cfg'
    var_3 = module_0.Config(var_2)
    var_4 = 'tests/example_settings_file.cfg'
    var_5 = module_0.Config(settings_path=var_4)
    var_6 = 100
    var_7 = module_0.Config()
    var_8 = module_0.Config(var_2)
    var_9 = module_0.Config(settings_path=var_4)
    var_10 = module_0.Config(var_2, var_4)
    var_11 = 'black'
    var_12 = module_0.Config(var_2, var_4)
    var_13 = module_0.Config()
    var_14 = module_0.Config(var_2, var_4, var_13)
    var_15 = module_0.Config()
    var_16 = 'tests'
    var_17 = [var_16]
    var_18 = module_0.Config(var_2, var_4, var_15)
    var_19 = module_0.Config()
    var_20 = [var_16]
    var_21 = '*.py'
    var_22 = [var_21]
    var_23 = module_0.Config(var_2, var_4, var_19)
    var_24 = module_0.Config()
    var_25 = [var_16]
    var_26 = [var_21]
    var_27 = module_0.Config()
    var_28 = [var_16]
    var_29 = [var_21]
    var_30 = '^test$'
    var_31 = 'test'
    var_32 = (var_30, var_31)
    var_33 = [var_32]
    var_34 = module_0.Config()
    var_35 = [var_16]
    var_36 = [var_21]
    var_37 = (var_30, var_31)
    var_38 = [var_37]
    var_39 = '# test'
    var_40 = (var_39,)
    var_41 = module_0.Config()
    var_42 = [var_16]
    var_43 = [var_21]
    var_44 = (var_30, var_31)
    var_45 = [var_44]
    var_46 = (var_39,)
    var_47 = (var_39,)
    var_48 = module_0.Config()
    var_49 = [var_16]
    var_50 = [var_21]
    var_51 = (var_30, var_31)
    var_52 = [var_51]
    var_53 = (var_39,)
    var_54 = (var_39,)
    var_55 = [var_31]
    var_56 = frozenset(var_55)
    var_57 = {var_31: var_56}
    var_58 = module_0.Config()
    var_59 = [var_16]
    var_60 = [var_21]
    var_61 = (var_30, var_31)
    var_62 = [var_61]
    var_63 = (var_39,)
    var_64 = (var_39,)
    var_65 = [var_31]
    var_66 = frozenset(var_65)
    var_67 = {var_31: var_66}
    var_68 = {var_31: var_39}
    var_69 = module_0.Config()
    var_70 = [var_16]
    var_71 = [var_21]
    var_72 = (var_30, var_31)
    var_73 = [var_72]
    var_74 = (var_39,)
    var_75 = (var_39,)
    var_76 = [var_31]
    var_77 = frozenset(var_76)
    var_78 = {var_31: var_77}
    var_79 = {var_31: var_39}
    var_80 = {var_31: var_39}
    var_81 = 'black'
    var_82 = module_0.Config()
    var_83 = 'tests'
    var_84 = [var_83]
    var_85 = '*.py'
    var_86 = [var_85]
    var_87 = '^test$'
    var_88 = 'test'
    var_89 = (var_87, var_88)
    var_90 = [var_89]
    var_91 = '# test'
    var_92 = (var_91,)
    var_93 = (var_91,)
    var_94 = [var_88]
    var_95 = frozenset(var_94)
    var_96 = {var_88: var_95}
    var_97 = {var_88: var_91}
    var_98 = {var_88: var_91}
    var_99 = 'value'
    var_100 = 'source'
    var_101 = {var_99: var_88, var_100: var_88}
    var_102 = {var_88: var_101}
    var_103 = 100
    var_104 = 'black'
    var_105 = module_0.Config()
    var_106 = 'tests'
    var_107 = [var_106]
    var_108 = '*.py'
    var_109 = [var_108]
    var_110 = '^test$'
    var_111 = 'test'
    var_112 = (var_110, var_111)
    var_113 = [var_112]
    var_114 = '# test'
    var_115 = (var_114,)
    var_116 = (var_114,)
    var_117 = [var_111]
    var_118 = frozenset(var_117)
    var_119 = {var_111: var_118}
    var_120 = {var_111: var_114}
    var_121 = {var_111: var_114}
    var_122 = 'value'
    var_123 = 'source'
    var_124 = {var_122: var_111, var_123: var_111}
    var_125 = {var_111: var_124}
    var_126 = {var_122: var_111, var_123: var_111}
    var_127 = {var_111: var_126}
    var_128 = 100



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'dir1'
    var_1 = 'dir2'
    var_2 = 'subdir'
    var_3 = '[settings]\nline_length=80\n'
    var_4 = '[settings]\nline_length=100\n'
    var_5 = '[settings]\nline_length=120\n'
    var_6 = '.isort.cfg'



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'nested'
    var_1 = '.isort.cfg'
    var_2 = '[settings]\nline_length=100\nprofile=black\n'
    var_3 = '[settings]\nline_length=80\nprofile=black\n'



# Parsed testcases at query #5
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'subdir'
    var_1 = '.isort.cfg'
    var_2 = '[settings]\nline_length=80\n'
    var_3 = '[settings]\nline_length=100\n'



# Parsed testcases at query #7
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = True
    var_2 = '.isort.cfg'
    var_3 = '[isort]\nline_length=80\n'
    var_4 = module_0.find_all_configs(var_0)



# Parsed testcases at query #8
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is True
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = var_0.is_supported_filetype(var_1)
    assert var_5 is False
    var_6 = 'test.txt'
    var_7 = var_0.is_supported_filetype(var_6)
    assert var_7 is False
    var_8 = 'test.py~'
    var_9 = var_0.is_supported_filetype(var_8)
    assert var_9 is False



# Parsed testcases at query #9
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is True
    var_3 = 'py'
    var_4 = var_0.is_supported_filetype(var_1)
    assert var_4 is False
    var_5 = 'test.txt'
    var_6 = var_0.is_supported_filetype(var_5)
    assert var_6 is False
    var_7 = 'test.py~'
    var_8 = var_0.is_supported_filetype(var_7)
    assert var_8 is False
    var_9 = 'test.fifo'
    var_10 = var_0.is_supported_filetype(var_9)
    assert var_10 is False
    var_11 = 'test.sh'
    var_12 = var_0.is_supported_filetype(var_11)
    assert var_12 is True



# Parsed testcases at query #10
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'tests'
    var_1 = 'temp'
    var_2 = {var_0, var_1}
    var_3 = '*.log'
    var_4 = '*.tmp'
    var_5 = {var_3, var_4}
    var_6 = module_0.Config()
    var_7 = 'tests/test_file.py'
    var_8 = 'temp/temp_file.log'
    var_9 = 'src/main.py'
    var_10 = 'src'



# Parsed testcases at query #11
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = frozenset()
    var_2 = frozenset()
    var_3 = frozenset()
    var_4 = frozenset()
    var_5 = frozenset()
    var_6 = frozenset()
    var_7 = frozenset()
    var_8 = frozenset()
    var_9 = frozenset()
    var_10 = frozenset()
    var_11 = frozenset()
    var_12 = frozenset()
    var_13 = frozenset()
    var_14 = frozenset()
    var_15 = frozenset()
    var_16 = 'py'
    var_17 = {var_16}
    var_18 = frozenset(var_17)
    var_19 = frozenset()
    var_20 = 'src'



# Parsed testcases at query #12
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is True
    var_3 = 'test.txt'
    var_4 = var_0.is_supported_filetype(var_3)
    assert var_4 is False
    var_5 = 'test~'
    var_6 = var_0.is_supported_filetype(var_5)
    assert var_6 is False
    var_7 = 'test.ipynb'
    var_8 = var_0.is_supported_filetype(var_7)
    assert var_8 is True
    var_9 = 'test'
    var_10 = var_0.is_supported_filetype(var_9)
    assert var_10 is False
    var_11 = 'test.PY'
    var_12 = var_0.is_supported_filetype(var_11)
    assert var_12 is True
    var_13 = 'test.TXT'
    var_14 = var_0.is_supported_filetype(var_13)
    assert var_14 is False
    var_15 = 'test.IPYNB'
    var_16 = var_0.is_supported_filetype(var_15)
    assert var_16 is True
    var_17 = 'test.~'
    var_18 = var_0.is_supported_filetype(var_17)
    assert var_18 is False



# Parsed testcases at query #13
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = module_0.Config()
    var_2 = '*.py'



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = '.isort.cfg'
    var_1 = '[settings]\nline_length=88\n'
    var_2 = 'subdir'
    var_3 = '[settings]\nline_length=100\n'
    var_4 = 'file.txt'
    var_5 = 'This is a test file.'
    var_6 = '.isort.cfg'
    var_7 = 'Invalid content'



# Parsed testcases at query #15
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'auto'
    var_1 = module_0._Config(var_0)
    var_2 = '3'
    var_3 = module_0._Config(var_2)
    var_4 = 'all'
    var_5 = module_0._Config(var_4)
    var_6 = 'invalid'
    var_7 = module_0._Config(var_6)
    var_8 = True
    var_9 = module_0._Config(force_alphabetical_sort=var_8)
    var_10 = 80
    var_11 = 79
    var_12 = module_0._Config(line_length=var_11, wrap_length=var_10)



# Parsed testcases at query #16
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'auto'
    var_1 = module_0._Config(var_0)
    var_2 = '3'
    var_3 = module_0._Config(var_2)
    var_4 = 'all'
    var_5 = module_0._Config(var_4)
    var_6 = 'invalid'
    var_7 = module_0._Config(var_6)
    var_8 = True
    var_9 = module_0._Config(force_alphabetical_sort=var_8)
    var_10 = 80
    var_11 = 79
    var_12 = module_0._Config(line_length=var_11, wrap_length=var_10)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_settings.ini'
    var_1 = '/path/to/settings'
    var_2 = module_0.Config(var_0, var_1)
    var_3 = 'test_file.py'
    var_4 = [var_3]
    var_5 = 'other_file.py'
    var_6 = [var_5]
    var_7 = 'test_*.py'
    var_8 = [var_7]
    var_9 = '/path/to/settings'
    var_10 = set()
    var_11 = '/path/to/settings/test_file.py'
    var_12 = {var_11}



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'subdir'
    var_1 = '.isort.cfg'
    var_2 = '[settings]\nline_length=80'
    var_3 = '[settings]\nline_length=100'



# Parsed testcases at query #3
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = 'test'
    var_3 = '*.py'
    var_4 = 'test.*'



# Parsed testcases at query #4
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    pass



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'subdir'
    var_1 = '.isort.cfg'
    var_2 = '[settings]\nline_length=100'
    var_3 = '[settings]\nline_length=80'
    var_4 = 'line_length'



# Parsed testcases at query #7
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = var_0.sources
    var_2 = var_0.sources
    var_3 = len(var_2)
    assert var_3 == 1
    var_4 = 'test_settings.ini'
    var_5 = '[isort]\nline_length = 88\n'
    var_6 = module_0.Config(var_4)
    var_7 = 'test_settings_path'
    var_8 = True
    var_9 = '[isort]\nline_length = 99\n'
    var_10 = module_0.Config(settings_path=var_7)
    var_11 = 100
    var_12 = module_0.Config()
    var_13 = module_0.Config(config=var_12)
    var_14 = 200
    var_15 = module_0.Config()
    var_16 = 300
    var_17 = 'os'
    var_18 = [var_17]
    var_19 = module_0.Config()
    var_20 = True
    var_21 = module_0.Config()
    var_22 = 'black'
    var_23 = module_0.Config()
    var_24 = 'invalid_profile'
    var_25 = module_0.Config()
    var_26 = 'terminal'
    var_27 = module_0.Config()
    var_28 = 'invalid_formatter'
    var_29 = module_0.Config()
    var_30 = 'native'
    var_31 = module_0.Config()
    var_32 = 'invalid_sort_order'
    var_33 = module_0.Config()



# Parsed testcases at query #8
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is True
    var_3 = 'test.txt'
    var_4 = var_0.is_supported_filetype(var_3)
    assert var_4 is False
    var_5 = 'test~'
    var_6 = var_0.is_supported_filetype(var_5)
    assert var_6 is False
    var_7 = 'test.bak'
    var_8 = var_0.is_supported_filetype(var_7)
    assert var_8 is False
    var_9 = 'test.PY'
    var_10 = var_0.is_supported_filetype(var_9)
    assert var_10 is True
    var_11 = 'test.TXT'
    var_12 = var_0.is_supported_filetype(var_11)
    assert var_12 is False
    var_13 = 'test.PYC'
    var_14 = var_0.is_supported_filetype(var_13)
    assert var_14 is False
    var_15 = 'test.PYD'
    var_16 = var_0.is_supported_filetype(var_15)
    assert var_16 is False
    var_17 = 'test.PYW'
    var_18 = var_0.is_supported_filetype(var_17)
    assert var_18 is False
    var_19 = 'test.PYO'
    var_20 = var_0.is_supported_filetype(var_19)
    assert var_20 is False
    var_21 = 'test.PYC~'
    var_22 = var_0.is_supported_filetype(var_21)
    assert var_22 is False
    var_23 = 'test.PYD~'
    var_24 = var_0.is_supported_filetype(var_23)
    assert var_24 is False
    var_25 = 'test.PYW~'
    var_26 = var_0.is_supported_filetype(var_25)
    assert var_26 is False
    var_27 = 'test.PYO~'
    var_28 = var_0.is_supported_filetype(var_27)
    assert var_28 is False
    var_29 = 'test.CFG'
    var_30 = var_0.is_supported_filetype(var_29)
    assert var_30 is False
    var_31 = 'test.CFG~'
    var_32 = var_0.is_supported_filetype(var_31)
    assert var_32 is False
    var_33 = 'test.INI'
    var_34 = var_0.is_supported_filetype(var_33)
    assert var_34 is False
    var_35 = 'test.INI~'
    var_36 = var_0.is_supported_filetype(var_35)
    assert var_36 is False
    var_37 = 'test.LOG'
    var_38 = var_0.is_supported_filetype(var_37)
    assert var_38 is False
    var_39 = 'test.LOG~'
    var_40 = var_0.is_supported_filetype(var_39)
    assert var_40 is False
    var_41 = 'test.TMP'
    var_42 = var_0.is_supported_filetype(var_41)
    assert var_42 is False
    var_43 = 'test.TMP~'
    var_44 = var_0.is_supported_filetype(var_43)
    assert var_44 is False
    var_45 = 'test.TEMP'
    var_46 = var_0.is_supported_filetype(var_45)
    assert var_46 is False
    var_47 = 'test.TEMP~'
    var_48 = var_0.is_supported_filetype(var_47)
    assert var_48 is False
    var_49 = 'test.BAK'
    var_50 = var_0.is_supported_filetype(var_49)
    assert var_50 is False
    var_51 = 'test.BAK~'
    var_52 = var_0.is_supported_filetype(var_51)
    assert var_52 is False
    var_53 = 'test.SWAP'
    var_54 = var_0.is_supported_filetype(var_53)
    assert var_54 is False
    var_55 = 'test.SWAP~'
    var_56 = var_0.is_supported_filetype(var_55)
    assert var_56 is False
    var_57 = 'test.SWP'
    var_58 = var_0.is_supported_filetype(var_57)
    assert var_58 is False
    var_59 = 'test.SWP~'
    var_60 = var_0.is_supported_filetype(var_59)
    assert var_60 is False
    var_61 = 'test.BAT'
    var_62 = var_0.is_supported_filetype(var_61)
    assert var_62 is False
    var_63 = 'test.BAT~'
    var_64 = var_0.is_supported_filetype(var_63)
    assert var_64 is False
    var_65 = 'test.CMD'
    var_66 = var_0.is_supported_filetype(var_65)
    assert var_66 is False
    var_67 = 'test.CMD~'
    var_68 = var_0.is_supported_filetype(var_67)
    assert var_68 is False
    var_69 = 'test.EXE'
    var_70 = var_0.is_supported_filetype(var_69)
    assert var_70 is False
    var_71 = 'test.EXE~'
    var_72 = var_0.is_supported_filetype(var_71)
    assert var_72 is False
    var_73 = 'test.DLL'
    var_74 = var_0.is_supported_filetype(var_73)
    assert var_74 is False
    var_75 = 'test.DLL~'
    var_76 = var_0.is_supported_filetype(var_75)
    assert var_76 is False
    var_77 = 'test.SO'
    var_78 = var_0.is_supported_filetype(var_77)
    assert var_78 is False
    var_79 = 'test.SO~'
    var_80 = var_0.is_supported_filetype(var_79)
    assert var_80 is False
    var_81 = var_0.is_supported_filetype(var_15)
    assert var_81 is False
    var_82 = var_0.is_supported_filetype(var_23)
    assert var_82 is False
    var_83 = var_0.is_supported_filetype(var_13)
    assert var_83 is False
    var_84 = var_0.is_supported_filetype(var_21)
    assert var_84 is False
    var_85 = var_0.is_supported_filetype(var_19)
    assert var_85 is False
    var_86 = var_0.is_supported_filetype(var_27)
    assert var_86 is False
    var_87 = var_0.is_supported_filetype(var_17)
    assert var_87 is False
    var_88 = var_0.is_supported_filetype(var_25)
    assert var_88 is False
    var_89 = var_0.is_supported_filetype(var_13)
    assert var_89 is False
    var_90 = var_0.is_supported_filetype(var_21)
    assert var_90 is False
    var_91 = var_0.is_supported_filetype(var_15)
    assert var_91 is False
    var_92 = var_0.is_supported_filetype(var_23)
    assert var_92 is False
    var_93 = var_0.is_supported_filetype(var_17)
    assert var_93 is False
    var_94 = var_0.is_supported_filetype(var_25)
    assert var_94 is False
    var_95 = var_0.is_supported_filetype(var_19)
    assert var_95 is False
    var_96 = var_0.is_supported_filetype(var_27)
    assert var_96 is False
    var_97 = var_0.is_supported_filetype(var_13)
    assert var_97 is False
    var_98 = var_0.is_supported_filetype(var_21)
    assert var_98 is False
    var_99 = var_0.is_supported_filetype(var_15)
    assert var_99 is False
    var_100 = var_0.is_supported_filetype(var_23)
    assert var_100 is False
    var_101 = var_0.is_supported_filetype(var_17)
    assert var_101 is False
    var_102 = var_0.is_supported_filetype(var_25)
    assert var_102 is False
    var_103 = var_0.is_supported_filetype(var_19)
    assert var_103 is False
    var_104 = var_0.is_supported_filetype(var_27)
    assert var_104 is False
    var_105 = var_0.is_supported_filetype(var_13)
    assert var_105 is False
    var_106 = var_0.is_supported_filetype(var_21)
    assert var_106 is False
    var_107 = var_0.is_supported_filetype(var_15)
    assert var_107 is False
    var_108 = var_0.is_supported_filetype(var_23)
    assert var_108 is False
    var_109 = var_0.is_supported_filetype(var_17)
    assert var_109 is False
    var_110 = var_0.is_supported_filetype(var_25)
    assert var_110 is False
    var_111 = var_0.is_supported_filetype(var_19)
    assert var_111 is False
    var_112 = var_0.is_supported_filetype(var_27)
    assert var_112 is False
    var_113 = var_0.is_supported_filetype(var_13)
    assert var_113 is False
    var_114 = var_0.is_supported_filetype(var_21)
    assert var_114 is False
    var_115 = var_0.is_supported_filetype(var_15)
    assert var_115 is False
    var_116 = var_0.is_supported_filetype(var_23)
    assert var_116 is False
    var_117 = var_0.is_supported_filetype(var_17)
    assert var_117 is False
    var_118 = var_0.is_supported_filetype(var_25)
    assert var_118 is False
    var_119 = var_0.is_supported_filetype(var_19)
    assert var_119 is False
    var_120 = var_0.is_supported_filetype(var_27)
    assert var_120 is False
    var_121 = var_0.is_supported_filetype(var_13)
    assert var_121 is False
    var_122 = var_0.is_supported_filetype(var_21)
    assert var_122 is False
    var_123 = var_0.is_supported_filetype(var_15)
    assert var_123 is False
    var_124 = var_0.is_supported_filetype(var_23)
    assert var_124 is False
    var_125 = var_0.is_supported_filetype(var_17)
    assert var_125 is False
    var_126 = var_0.is_supported_filetype(var_25)
    assert var_126 is False
    var_127 = var_0.is_supported_filetype(var_19)
    assert var_127 is False
    var_128 = var_0.is_supported_filetype(var_27)
    assert var_128 is False
    var_129 = var_0.is_supported_filetype(var_13)
    assert var_129 is False
    var_130 = var_0.is_supported_filetype(var_21)
    assert var_130 is False
    var_131 = var_0.is_supported_filetype(var_15)
    assert var_131 is False
    var_132 = var_0.is_supported_filetype(var_23)
    assert var_132 is False
    var_133 = var_0.is_supported_filetype(var_17)
    assert var_133 is False
    var_134 = var_0.is_supported_filetype(var_25)
    assert var_134 is False
    var_135 = var_0.is_supported_filetype(var_19)
    assert var_135 is False
    var_136 = var_0.is_supported_filetype(var_27)
    assert var_136 is False
    var_137 = var_0.is_supported_filetype(var_13)
    assert var_137 is False
    var_138 = var_0.is_supported_filetype(var_21)
    assert var_138 is False
    var_139 = var_0.is_supported_filetype(var_15)
    assert var_139 is False
    var_140 = var_0.is_supported_filetype(var_23)
    assert var_140 is False
    var_141 = var_0.is_supported_filetype(var_17)
    assert var_141 is False
    var_142 = var_0.is_supported_filetype(var_25)
    assert var_142 is False
    var_143 = var_0.is_supported_filetype(var_19)
    assert var_143 is False
    var_144 = var_0.is_supported_filetype(var_27)
    assert var_144 is False
    var_145 = var_0.is_supported_filetype(var_13)
    assert var_145 is False
    var_146 = var_0.is_supported_filetype(var_21)
    assert var_146 is False
    var_147 = var_0.is_supported_filetype(var_15)
    assert var_147 is False
    var_148 = var_0.is_supported_filetype(var_23)
    assert var_148 is False
    var_149 = var_0.is_supported_filetype(var_17)
    assert var_149 is False
    var_150 = var_0.is_supported_filetype(var_25)
    assert var_150 is False
    var_151 = var_0.is_supported_filetype(var_19)
    assert var_151 is False
    var_152 = var_0.is_supported_filetype(var_27)
    assert var_152 is False
    var_153 = var_0.is_supported_filetype(var_13)
    assert var_153 is False
    var_154 = var_0.is_supported_filetype(var_21)
    assert var_154 is False
    var_155 = var_0.is_supported_filetype(var_15)
    assert var_155 is False
    var_156 = var_0.is_supported_filetype(var_23)
    assert var_156 is False
    var_157 = var_0.is_supported_filetype(var_17)
    assert var_157 is False
    var_158 = var_0.is_supported_filetype(var_25)
    assert var_158 is False
    var_159 = var_0.is_supported_filetype(var_19)
    assert var_159 is False
    var_160 = var_0.is_supported_filetype(var_27)
    assert var_160 is False
    var_161 = var_0.is_supported_filetype(var_13)
    assert var_161 is False
    var_162 = var_0.is_supported_filetype(var_21)
    assert var_162 is False
    var_163 = var_0.is_supported_filetype(var_15)
    assert var_163 is False
    var_164 = var_0.is_supported_filetype(var_23)
    assert var_164 is False



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'sub1'
    var_1 = '[settings]\nknown_first_party=test1'
    var_2 = 'sub2'
    var_3 = '[tool.isort]\nknown_first_party=test2'
    var_4 = 0
    var_5 = 'known_first_party'
    var_6 = 'known_first_party'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '.isort.cfg'
    var_2 = 'pyproject.toml'
    var_3 = [var_0, var_1, var_2]
    var_4 = '[isort]\nprofile=black'
    var_5 = 'subdir1'
    var_6 = 'subdir2'
    var_7 = [var_5, var_6]
    var_8 = '[isort]\nprofile=black'
    var_9 = 'nonexistent.cfg'



# Parsed testcases at query #11
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'auto'
    var_1 = module_0._Config(var_0)
    var_2 = 'all'
    var_3 = module_0._Config(var_2)
    var_4 = '3'
    var_5 = module_0._Config(var_4)
    var_6 = 'invalid'
    var_7 = module_0._Config(var_6)
    var_8 = frozenset()
    var_9 = module_0._Config(known_standard_library=var_8)
    var_10 = var_9.py_version
    var_11 = True
    var_12 = module_0._Config(force_alphabetical_sort=var_11)
    var_13 = 80
    var_14 = 79
    var_15 = module_0._Config(line_length=var_14, wrap_length=var_13)



# Parsed testcases at query #12
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = 'test_dir'
    var_3 = 'test_dir/test.py'
    var_4 = '*.py'
    var_5 = 'test_dir/*'
    var_6 = '.git'



# Parsed testcases at query #13
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.ini'
    var_2 = module_0.Config(var_1)
    var_3 = module_0.Config(settings_path=var_1)
    var_4 = module_0.Config()
    var_5 = module_0.Config(config=var_4)
    var_6 = 100
    var_7 = module_0.Config()
    var_8 = 'black'
    var_9 = module_0.Config()
    var_10 = 'test'
    var_11 = [var_10]
    var_12 = frozenset(var_11)
    var_13 = {var_10: var_12}
    var_14 = module_0.Config()
    var_15 = {var_10: var_10}
    var_16 = module_0.Config()
    var_17 = {var_10: var_10}
    var_18 = module_0.Config()
    var_19 = 'src'
    var_20 = [var_19]
    var_21 = module_0.Config()
    var_22 = var_21.src_paths
    var_23 = [str(path) for path in var_22]
    var_24 = 'text'
    var_25 = module_0.Config()
    var_26 = 'natural'
    var_27 = module_0.Config()
    var_28 = [var_10]
    var_29 = module_0.Config()
    var_30 = [var_10]
    var_31 = module_0.Config()
    var_32 = True
    var_33 = module_0.Config()
    var_34 = [var_10]
    var_35 = module_0.Config()
    var_36 = [var_10]
    var_37 = module_0.Config()
    var_38 = module_0.Config()
    var_39 = module_0.Config()
    var_40 = '3.8'
    var_41 = module_0.Config()
    var_42 = [var_10]
    var_43 = module_0.Config()
    var_44 = [var_10]
    var_45 = module_0.Config()
    var_46 = module_0.Config()
    var_47 = [var_10]
    var_48 = module_0.Config()
    var_49 = module_0.Config()
    var_50 = module_0.Config()
    var_51 = [var_10]
    var_52 = module_0.Config()
    var_53 = [var_10]
    var_54 = module_0.Config()
    var_55 = 2
    var_56 = module_0.Config()
    var_57 = module_0.Config()
    var_58 = module_0.Config()
    var_59 = module_0.Config()
    var_60 = module_0.Config()
    var_61 = module_0.Config()
    var_62 = module_0.Config()
    var_63 = module_0.Config()
    var_64 = module_0.Config()
    var_65 = module_0.Config()
    var_66 = module_0.Config()
    var_67 = module_0.Config()
    var_68 = module_0.Config()
    var_69 = module_0.Config()
    var_70 = module_0.Config()
    var_71 = module_0.Config()
    var_72 = module_0.Config()
    var_73 = module_0.Config()
    var_74 = module_0.Config()
    var_75 = module_0.Config()
    var_76 = module_0.Config()
    var_77 = module_0.Config()
    var_78 = module_0.Config()
    var_79 = module_0.Config()
    var_80 = module_0.Config()
    var_81 = module_0.Config()
    var_82 = module_0.Config()
    var_83 = module_0.Config()
    var_84 = module_0.Config()



# Parsed testcases at query #14
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = '*.py'
    var_3 = 'test.*'
    var_4 = var_0.skip_glob.add



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'dir1'
    var_1 = 'dir2'
    var_2 = 'dir3'
    var_3 = '[isort]\nknown_third_party=requests\n'
    var_4 = '.isort.cfg'
    var_5 = 'known_third_party'
    var_6 = 'requests'
    var_7 = {var_6}
    var_8 = frozenset(var_7)



# Parsed testcases at query #16
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'tests'
    var_1 = {var_0}
    var_2 = module_0.Config()
    var_3 = 'tests/test_file.py'
    var_4 = 'src/test_file.py'
    var_5 = '*.py'
    var_6 = {var_5}
    var_7 = module_0.Config()
    var_8 = 'test_file.py'
    var_9 = 'test_file.txt'
    var_10 = True
    var_11 = module_0.Config()
    var_12 = '.git'
    var_13 = {var_0}
    var_14 = {var_5}
    var_15 = module_0.Config()



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'subdir1'
    var_1 = 'subdir2'
    var_2 = '[settings]\nline_length=80\n'
    var_3 = '[tool.isort]\nline_length=100\n'
    var_4 = '[isort]\nline_length=120\n'
    var_5 = set()
    var_6 = '.isort.cfg'
    var_7 = 'pyproject.toml'
    var_8 = 'setup.cfg'



# Parsed testcases at query #18
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = {var_1}
    var_3 = module_0.Config()



# Parsed testcases at query #19
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = '3'
    var_1 = module_0._Config(var_0)
    var_2 = 'invalid'
    var_3 = module_0._Config(var_2)
    var_4 = 3
    var_5 = 8
    var_6 = 0
    var_7 = 'final'
    var_8 = 'auto'
    var_9 = module_0._Config(var_8)
    var_10 = True
    var_11 = module_0._Config(force_alphabetical_sort=var_10)
    var_12 = 80
    var_13 = 79
    var_14 = module_0._Config(line_length=var_13, wrap_length=var_12)



# Parsed testcases at query #20
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_settings.cfg'
    var_2 = module_0.Config(var_1)
    var_3 = module_0.Config(settings_path=var_1)
    var_4 = module_0.Config()
    var_5 = module_0.Config(config=var_4)
    var_6 = 100
    var_7 = module_0.Config()
    var_8 = 'black'
    var_9 = module_0.Config()
    var_10 = module_0.Config(var_1)
    var_11 = module_0.Config(settings_path=var_1)
    var_12 = module_0.Config()
    var_13 = module_0.Config(config=var_12)
    var_14 = module_0.Config()
    var_15 = module_0.Config(var_1)
    var_16 = module_0.Config(settings_path=var_1)
    var_17 = module_0.Config()
    var_18 = module_0.Config(config=var_17)
    var_19 = module_0.Config()
    var_20 = module_0.Config(var_1, config=var_19)
    var_21 = module_0.Config()
    var_22 = module_0.Config(settings_path=var_1, config=var_21)



