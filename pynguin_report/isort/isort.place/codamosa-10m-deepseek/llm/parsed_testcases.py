####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
    var_2 = 'numpy'
    var_3 = module_0.module(var_2)
    assert var_3 == 'THIRDPARTY'
    var_4 = '.local_module'
    var_5 = module_0.module(var_4)
    assert var_5 == 'LOCALFOLDER'
    var_6 = 'unknown_module'
    var_7 = module_0.module(var_6)
    assert var_7 == 'THIRDPARTY'
    var_8 = 'custom_module'
    var_9 = 'CUSTOM'
    var_10 = module_1.Config()
    var_11 = module_0.module(var_8, var_10)
    assert var_11 == 'CUSTOM'



# Parsed testcases at query #2
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'numpy'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'THIRDPARTY'
    var_5 = 'local_module'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'FIRSTPARTY'
    var_7 = '.local_module'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'LOCALFOLDER'



# Parsed testcases at query #3
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
    var_2 = 'os.path'
    var_3 = module_0.module(var_2)
    assert var_3 == 'STDLIB'
    var_4 = 'numpy'
    var_5 = module_0.module(var_4)
    assert var_5 == 'THIRDPARTY'
    var_6 = 'pandas'
    var_7 = module_0.module(var_6)
    assert var_7 == 'THIRDPARTY'
    var_8 = 'local.module'
    var_9 = module_0.module(var_8)
    assert var_9 == 'LOCALFOLDER'
    var_10 = 'custom.module'
    var_11 = 'custom.*'
    var_12 = 'CUSTOM'
    var_13 = (var_11, var_12)
    var_14 = [var_13]
    var_15 = module_1.Config()
    var_16 = module_0.module(var_10, var_15)
    assert var_16 == 'CUSTOM'



# Parsed testcases at query #4
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'isort.settings'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'FIRSTPARTY'
    var_5 = 'local_module'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'LOCALFOLDER'
    var_7 = 'unknown_module'
    var_8 = module_1.module(var_7, var_0)



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'tests*'
    var_1 = [var_0]
    var_2 = '^django\\.contrib'
    var_3 = 'DJANGO'
    var_4 = 'FUTURE'
    var_5 = 'STDLIB'
    var_6 = 'THIRDPARTY'
    var_7 = 'FIRSTPARTY'
    var_8 = 'LOCALFOLDER'
    var_9 = [var_4, var_5, var_6, var_7, var_8]
    var_10 = '/src'
    var_11 = 'os'
    var_12 = 'django.contrib.admin'
    var_13 = 'tests.test_module'
    var_14 = '.local_module'
    var_15 = 'src_package'
    var_16 = 'unknown_package'



# Parsed testcases at query #6
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'sys'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'STDLIB'
    var_5 = 'django'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'THIRDPARTY'
    var_7 = 'numpy'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'THIRDPARTY'
    var_9 = '.local_module'
    var_10 = module_1.module(var_9, var_0)
    assert var_10 == 'LOCALFOLDER'
    var_11 = 'local_module'
    var_12 = module_1.module(var_11, var_0)
    assert var_12 == 'FIRSTPARTY'



# Parsed testcases at query #7
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '^django$'
    var_1 = 'THIRDPARTY'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = module_0.Config()
    var_5 = 'django'
    var_6 = module_1.module(var_5, var_4)
    assert var_6 == 'THIRDPARTY'
    var_7 = '.local_module'
    var_8 = module_1.module(var_7, var_4)
    assert var_8 == 'LOCALFOLDER'
    var_9 = 'tests'
    var_10 = [var_9]
    var_11 = module_0.Config()
    var_12 = module_1.module(var_9, var_11)
    assert var_12 == 'tests'
    var_13 = 'unknown_module'
    var_14 = module_1.module(var_13, var_11)
    var_15 = '/path/to/project'
    var_16 = 'project_module'
    var_17 = module_1.module(var_16, var_11)
    assert var_17 == 'FIRSTPARTY'



# Parsed testcases at query #8
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
    var_2 = 'django'
    var_3 = module_0.module(var_2)
    assert var_3 == 'THIRDPARTY'
    var_4 = '.local_module'
    var_5 = module_0.module(var_4)
    assert var_5 == 'LOCALFOLDER'
    var_6 = 'my_project'
    var_7 = module_0.module(var_6)
    assert var_7 == 'FIRSTPARTY'
    var_8 = 'unknown_module'
    var_9 = module_0.module(var_8)
    assert var_9 == 'THIRDPARTY'
    var_10 = '^my_project'
    var_11 = 'FIRSTPARTY'
    var_12 = (var_10, var_11)
    var_13 = [var_12]
    var_14 = module_1.Config()
    var_15 = 'my_project.module'
    var_16 = module_0.module(var_15, var_14)
    assert var_16 == 'FIRSTPARTY'
    var_17 = 'other_project.module'
    var_18 = module_0.module(var_17, var_14)
    assert var_18 == 'THIRDPARTY'
    var_19 = 'special.module'
    var_20 = [var_19]
    var_21 = module_1.Config()
    var_22 = module_0.module(var_19, var_21)
    assert var_22 == 'special.module'
    var_23 = 'special.module.sub'
    var_24 = module_0.module(var_23, var_21)
    assert var_24 == 'special.module'
    var_25 = 'All module function tests passed!'
    var_26 = print(var_25)



# Parsed testcases at query #9
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'isort'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'THIRDPARTY'
    var_5 = '.local_module'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'LOCALFOLDER'
    var_7 = 'tests.test_module'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'FIRSTPARTY'
    var_9 = 'unknown_module'
    var_10 = module_1.module(var_9, var_0)



# Parsed testcases at query #10
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'numpy'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'THIRDPARTY'
    var_5 = 'local_module'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'FIRSTPARTY'
    var_7 = '.local_module'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'LOCALFOLDER'
    var_9 = 'unknown_module'
    var_10 = module_1.module(var_9, var_0)



# Parsed testcases at query #11
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'django'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'THIRDPARTY'
    var_5 = '.local_module'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'LOCALFOLDER'
    var_7 = 'my_project'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'FIRSTPARTY'
    var_9 = 'unknown_module'
    var_10 = module_1.module(var_9, var_0)



# Parsed testcases at query #12
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'sys'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'STDLIB'
    var_5 = 'math'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'STDLIB'
    var_7 = 'numpy'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'THIRDPARTY'
    var_9 = 'pandas'
    var_10 = module_1.module(var_9, var_0)
    assert var_10 == 'THIRDPARTY'
    var_11 = 'my_local_module'
    var_12 = module_1.module(var_11, var_0)
    assert var_12 == 'FIRSTPARTY'
    var_13 = '.my_local_module'
    var_14 = module_1.module(var_13, var_0)
    assert var_14 == 'LOCALFOLDER'
    var_15 = [var_11]
    var_16 = module_0.Config()
    var_17 = module_1.module(var_11, var_16)
    assert var_17 == 'FIRSTPARTY'
    var_18 = [var_11]
    var_19 = module_0.Config()
    var_20 = module_1.module(var_11, var_19)
    assert var_20 == 'THIRDPARTY'
    var_21 = [var_11]
    var_22 = [var_11]
    var_23 = module_0.Config()
    var_24 = module_1.module(var_11, var_23)
    assert var_24 == 'FIRSTPARTY'
    var_25 = [var_11]
    var_26 = module_0.Config()
    var_27 = module_1.module(var_11, var_26)
    assert var_27 == 'my_local_module'
    var_28 = 'THIRDPARTY'
    var_29 = module_0.Config()
    var_30 = module_1.module(var_11, var_29)
    assert var_30 == 'THIRDPARTY'
    var_31 = 'FIRSTPARTY'
    var_32 = module_0.Config()
    var_33 = module_1.module(var_11, var_32)
    assert var_33 == 'FIRSTPARTY'
    var_34 = 'STDLIB'
    var_35 = module_0.Config()
    var_36 = module_1.module(var_11, var_35)
    assert var_36 == 'STDLIB'
    var_37 = 'LOCALFOLDER'
    var_38 = module_0.Config()
    var_39 = module_1.module(var_11, var_38)
    assert var_39 == 'LOCALFOLDER'
    var_40 = [var_11]
    var_41 = module_0.Config()
    var_42 = module_1.module(var_11, var_41)
    assert var_42 == 'FIRSTPARTY'
    var_43 = [var_11]
    var_44 = module_0.Config()
    var_45 = module_1.module(var_11, var_44)
    assert var_45 == 'THIRDPARTY'
    var_46 = [var_11]
    var_47 = module_0.Config()
    var_48 = module_1.module(var_11, var_47)
    assert var_48 == 'FIRSTPARTY'
    var_49 = [var_11]
    var_50 = module_0.Config()
    var_51 = module_1.module(var_11, var_50)
    assert var_51 == 'FIRSTPARTY'
    var_52 = [var_11]
    var_53 = module_0.Config()
    var_54 = module_1.module(var_11, var_53)
    assert var_54 == 'THIRDPARTY'
    var_55 = [var_11]
    var_56 = module_0.Config()
    var_57 = module_1.module(var_11, var_56)
    assert var_57 == 'THIRDPARTY'
    var_58 = [var_11]
    var_59 = module_0.Config()
    var_60 = module_1.module(var_11, var_59)
    assert var_60 == 'THIRDPARTY'
    var_61 = [var_11]
    var_62 = module_0.Config()
    var_63 = module_1.module(var_11, var_62)
    assert var_63 == 'THIRDPARTY'
    var_64 = [var_11]
    var_65 = [var_11]
    var_66 = module_0.Config()
    var_67 = module_1.module(var_11, var_66)
    assert var_67 == 'FIRSTPARTY'
    var_68 = [var_11]
    var_69 = [var_11]
    var_70 = module_0.Config()
    var_71 = module_1.module(var_11, var_70)
    assert var_71 == 'FIRSTPARTY'
    var_72 = [var_11]
    var_73 = [var_11]
    var_74 = module_0.Config()
    var_75 = module_1.module(var_11, var_74)
    assert var_75 == 'FIRSTPARTY'
    var_76 = [var_11]
    var_77 = [var_11]
    var_78 = module_0.Config()
    var_79 = module_1.module(var_11, var_78)
    assert var_79 == 'FIRSTPARTY'
    var_80 = [var_11]
    var_81 = module_0.Config()
    var_82 = module_1.module(var_11, var_81)
    assert var_82 == 'my_local_module'
    var_83 = [var_11]
    var_84 = module_0.Config()
    var_85 = module_1.module(var_11, var_84)
    assert var_85 == 'my_local_module'
    var_86 = [var_11]
    var_87 = module_0.Config()
    var_88 = module_1.module(var_11, var_87)
    assert var_88 == 'my_local_module'
    var_89 = [var_11]
    var_90 = module_0.Config()
    var_91 = module_1.module(var_11, var_90)
    assert var_91 == 'my_local_module'
    var_92 = [var_11]
    var_93 = [var_11]
    var_94 = module_0.Config()
    var_95 = module_1.module(var_11, var_94)
    assert var_95 == 'my_local_module'
    var_96 = [var_11]
    var_97 = [var_11]
    var_98 = module_0.Config()
    var_99 = module_1.module(var_11, var_98)
    assert var_99 == 'my_local_module'
    var_100 = [var_11]
    var_101 = [var_11]
    var_102 = module_0.Config()
    var_103 = module_1.module(var_11, var_102)
    assert var_103 == 'my_local_module'
    var_104 = [var_11]
    var_105 = [var_11]
    var_106 = module_0.Config()
    var_107 = module_1.module(var_11, var_106)
    assert var_107 == 'my_local_module'
    var_108 = [var_11]
    var_109 = [var_11]
    var_110 = module_0.Config()
    var_111 = module_1.module(var_11, var_110)
    assert var_111 == 'my_local_module'
    var_112 = [var_11]
    var_113 = [var_11]
    var_114 = module_0.Config()
    var_115 = module_1.module(var_11, var_114)
    assert var_115 == 'my_local_module'
    var_116 = [var_11]
    var_117 = [var_11]
    var_118 = module_0.Config()
    var_119 = module_1.module(var_11, var_118)
    assert var_119 == 'my_local_module'
    var_120 = [var_11]
    var_121 = [var_11]
    var_122 = module_0.Config()
    var_123 = module_1.module(var_11, var_122)
    assert var_123 == 'my_local_module'
    var_124 = [var_11]
    var_125 = [var_11]
    var_126 = [var_11]
    var_127 = module_0.Config()
    var_128 = module_1.module(var_11, var_127)
    assert var_128 == 'my_local_module'
    var_129 = [var_11]
    var_130 = [var_11]
    var_131 = [var_11]
    var_132 = module_0.Config()
    var_133 = module_1.module(var_11, var_132)
    assert var_133 == 'my_local_module'
    var_134 = [var_11]
    var_135 = [var_11]
    var_136 = [var_11]
    var_137 = module_0.Config()
    var_138 = module_1.module(var_11, var_137)
    assert var_138 == 'my_local_module'
    var_139 = [var_11]
    var_140 = [var_11]
    var_141 = [var_11]
    var_142 = module_0.Config()
    var_143 = module_1.module(var_11, var_142)
    assert var_143 == 'my_local_module'



# Parsed testcases at query #13
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'django'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'THIRDPARTY'
    var_5 = 'local_module'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'FIRSTPARTY'
    var_7 = '.local_module'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'LOCALFOLDER'
    var_9 = 'forced_separate_module'
    var_10 = module_1.module(var_9, var_0)
    assert var_10 == 'forced_separate_module'



# Parsed testcases at query #14
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'numpy'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'THIRDPARTY'
    var_5 = 'isort'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'FIRSTPARTY'
    var_7 = '.local_module'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'LOCALFOLDER'
    var_9 = 'unknown_module'
    var_10 = module_1.module(var_9, var_0)
    assert var_10 == 'THIRDPARTY'



# Parsed testcases at query #15
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'Test the module function.'
    var_1 = module_0.Config()
    var_2 = 'os'
    var_3 = module_1.module(var_2, var_1)
    assert var_3 == 'STDLIB'
    var_4 = 'django'
    var_5 = module_1.module(var_4, var_1)
    assert var_5 == 'THIRDPARTY'
    var_6 = 'my_local_module'
    var_7 = module_1.module(var_6, var_1)
    assert var_7 == 'FIRSTPARTY'
    var_8 = '.hidden_module'
    var_9 = module_1.module(var_8, var_1)
    assert var_9 == 'LOCALFOLDER'
    var_10 = '_private_module'
    var_11 = module_1.module(var_10, var_1)
    assert var_11 == 'FIRSTPARTY'
    var_12 = 'tests.test_module'
    var_13 = module_1.module(var_12, var_1)
    assert var_13 == 'FIRSTPARTY'
    var_14 = 'setup'
    var_15 = module_1.module(var_14, var_1)
    assert var_15 == 'FIRSTPARTY'
    var_16 = 'conftest'
    var_17 = module_1.module(var_16, var_1)
    assert var_17 == 'FIRSTPARTY'
    var_18 = 'my_namespace.module'
    var_19 = module_1.module(var_18, var_1)
    assert var_19 == 'FIRSTPARTY'
    var_20 = 'my_namespace'
    var_21 = module_1.module(var_20, var_1)
    assert var_21 == 'FIRSTPARTY'
    var_22 = 'my_forced_separate_module'
    var_23 = module_1.module(var_22, var_1)
    assert var_23 == 'my_forced_separate_module'
    var_24 = 'All module function tests passed!'
    var_25 = print(var_24)



# Parsed testcases at query #16
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'django'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'THIRDPARTY'
    var_5 = 'local_module'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'FIRSTPARTY'
    var_7 = '.local_module'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'LOCALFOLDER'
    var_9 = 'forced_separate_module'
    var_10 = module_1.module(var_9, var_0)
    assert var_10 == 'forced_separate_module'
    var_11 = 'unknown_module'
    var_12 = module_1.module(var_11, var_0)



# Parsed testcases at query #17
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'django'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'THIRDPARTY'
    var_5 = '.local_module'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'LOCALFOLDER'
    var_7 = 'my_project'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'FIRSTPARTY'
    var_9 = 'unknown_module'
    var_10 = module_1.module(var_9, var_0)
    assert var_10 == 'THIRDPARTY'
    var_11 = 'All tests passed!'
    var_12 = print(var_11)



# Parsed testcases at query #18
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_module'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'test_module'
    var_3 = '.local_module'
    var_4 = module_1.module(var_3, var_0)
    var_5 = '^test_pattern'
    var_6 = 'THIRDPARTY'
    var_7 = (var_5, var_6)
    var_8 = 'test_pattern.module'
    var_9 = module_1.module(var_8, var_0)
    assert var_9 == 'THIRDPARTY'
    var_10 = '/path/to/src'
    var_11 = 'src_module'
    var_12 = module_1.module(var_11, var_0)
    var_13 = 'unknown_module'
    var_14 = module_1.module(var_13, var_0)



# Parsed testcases at query #19
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'numpy'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'THIRDPARTY'
    var_5 = '.local_module'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'LOCALFOLDER'
    var_7 = 'src.local_module'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'FIRSTPARTY'



# Parsed testcases at query #20
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = '.local'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'LOCALFOLDER'
    var_5 = 'os.path'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'STDLIB'
    var_7 = 'numpy'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'THIRDPARTY'
    var_9 = 'src.module'
    var_10 = module_1.module(var_9, var_0)
    assert var_10 == 'FIRSTPARTY'
    var_11 = 'unknown.module'
    var_12 = module_1.module(var_11, var_0)
    assert var_12 == 'STDLIB'
    var_13 = 'setup'
    var_14 = module_1.module(var_13, var_0)
    assert var_14 == 'STDLIB'
    var_15 = 'pyproject'
    var_16 = module_1.module(var_15, var_0)
    assert var_16 == 'STDLIB'
    var_17 = 'pkg_resources'
    var_18 = module_1.module(var_17, var_0)
    assert var_18 == 'STDLIB'
    var_19 = 'pkgutil'
    var_20 = module_1.module(var_19, var_0)
    assert var_20 == 'STDLIB'
    var_21 = 'unknown'
    var_22 = module_1.module(var_21, var_0)
    assert var_22 == 'STDLIB'
    var_23 = module_1.module(var_11, var_0)
    assert var_23 == 'STDLIB'
    var_24 = 'unknown.module.submodule'
    var_25 = module_1.module(var_24, var_0)
    assert var_25 == 'STDLIB'
    var_26 = 'unknown.module.submodule.subsubmodule'
    var_27 = module_1.module(var_26, var_0)
    assert var_27 == 'STDLIB'
    var_28 = 'unknown.module.submodule.subsubmodule.subsubsubmodule'
    var_29 = module_1.module(var_28, var_0)
    assert var_29 == 'STDLIB'
    var_30 = 'unknown.module.submodule.subsubmodule.subsubsubmodule.subsubsubsubmodule'
    var_31 = module_1.module(var_30, var_0)
    assert var_31 == 'STDLIB'
    var_32 = 'unknown.module.submodule.subsubmodule.subsubsubmodule.subsubsubsubmodule.subsubsubsubsubmodule'
    var_33 = module_1.module(var_32, var_0)
    assert var_33 == 'STDLIB'
    var_34 = 'unknown.module.submodule.subsubmodule.subsubsubmodule.subsubsubsubmodule.subsubsubsubsubmodule.subsubsubsubsubsubmodule'
    var_35 = module_1.module(var_34, var_0)
    assert var_35 == 'STDLIB'
    var_36 = 'unknown.module.submodule.subsubmodule.subsubsubmodule.subsubsubsubmodule.subsubsubsubsubmodule.subsubsubsubsubsubmodule.subsubsubsubsubsubsubmodule'
    var_37 = module_1.module(var_36, var_0)
    assert var_37 == 'STDLIB'
    var_38 = 'unknown.module.submodule.subsubmodule.subsubsubmodule.subsubsubsubmodule.subsubsubsubsubmodule.subsubsubsubsubsubmodule.subsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubmodule'
    var_39 = module_1.module(var_38, var_0)
    assert var_39 == 'STDLIB'
    var_40 = 'unknown.module.submodule.subsubmodule.subsubsubmodule.subsubsubsubmodule.subsubsubsubsubmodule.subsubsubsubsubsubmodule.subsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubmodule'
    var_41 = module_1.module(var_40, var_0)
    assert var_41 == 'STDLIB'
    var_42 = 'unknown.module.submodule.subsubmodule.subsubsubmodule.subsubsubsubmodule.subsubsubsubsubmodule.subsubsubsubsubsubmodule.subsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubmodule'
    var_43 = module_1.module(var_42, var_0)
    assert var_43 == 'STDLIB'
    var_44 = 'unknown.module.submodule.subsubmodule.subsubsubmodule.subsubsubsubmodule.subsubsubsubsubmodule.subsubsubsubsubsubmodule.subsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubmodule'
    var_45 = module_1.module(var_44, var_0)
    assert var_45 == 'STDLIB'
    var_46 = 'unknown.module.submodule.subsubmodule.subsubsubmodule.subsubsubsubmodule.subsubsubsubsubmodule.subsubsubsubsubsubmodule.subsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubmodule'
    var_47 = module_1.module(var_46, var_0)
    assert var_47 == 'STDLIB'
    var_48 = 'unknown.module.submodule.subsubmodule.subsubsubmodule.subsubsubsubmodule.subsubsubsubsubmodule.subsubsubsubsubsubmodule.subsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubmodule'
    var_49 = module_1.module(var_48, var_0)
    assert var_49 == 'STDLIB'
    var_50 = 'unknown.module.submodule.subsubmodule.subsubsubmodule.subsubsubsubmodule.subsubsubsubsubmodule.subsubsubsubsubsubmodule.subsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubmodule'
    var_51 = module_1.module(var_50, var_0)
    assert var_51 == 'STDLIB'
    var_52 = 'unknown.module.submodule.subsubmodule.subsubsubmodule.subsubsubsubmodule.subsubsubsubsubmodule.subsubsubsubsubsubmodule.subsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubsubmodule'
    var_53 = module_1.module(var_52, var_0)
    assert var_53 == 'STDLIB'
    var_54 = 'unknown.module.submodule.subsubmodule.subsubsubmodule.subsubsubsubmodule.subsubsubsubsubmodule.subsubsubsubsubsubmodule.subsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubsubsubmodule'
    var_55 = module_1.module(var_54, var_0)
    assert var_55 == 'STDLIB'
    var_56 = 'unknown.module.submodule.subsubmodule.subsubsubmodule.subsubsubsubmodule.subsubsubsubsubmodule.subsubsubsubsubsubmodule.subsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubsubsubsubmodule'
    var_57 = module_1.module(var_56, var_0)
    assert var_57 == 'STDLIB'
    var_58 = 'unknown.module.submodule.subsubmodule.subsubsubmodule.subsubsubsubmodule.subsubsubsubsubmodule.subsubsubsubsubsubmodule.subsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubsubsubsubsubmodule'
    var_59 = module_1.module(var_58, var_0)
    assert var_59 == 'STDLIB'
    var_60 = 'unknown.module.submodule.subsubmodule.subsubsubmodule.subsubsubsubmodule.subsubsubsubsubmodule.subsubsubsubsubsubmodule.subsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubsubsubsubsubsubmodule'
    var_61 = module_1.module(var_60, var_0)
    assert var_61 == 'STDLIB'



# Parsed testcases at query #21
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'django'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'THIRDPARTY'
    var_5 = 'isort'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'FIRSTPARTY'
    var_7 = '.local'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'LOCALFOLDER'
    var_9 = 'unknown.module'
    var_10 = module_1.module(var_9, var_0)



# Parsed testcases at query #22
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'django'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'THIRDPARTY'
    var_5 = 'local_module'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'FIRSTPARTY'
    var_7 = '.local_module'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'LOCALFOLDER'
    var_9 = 'unknown_module'
    var_10 = module_1.module(var_9, var_0)
    assert var_10 == 'THIRDPARTY'



# Parsed testcases at query #23
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'sys'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'STDLIB'
    var_5 = 'isort'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'THIRDPARTY'
    var_7 = 'pytest'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'THIRDPARTY'
    var_9 = '.local_module'
    var_10 = module_1.module(var_9, var_0)
    assert var_10 == 'LOCALFOLDER'
    var_11 = 'local_module'
    var_12 = module_1.module(var_11, var_0)
    assert var_12 == 'FIRSTPARTY'



# Parsed testcases at query #24
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django*'
    var_1 = 'THIRDPARTY'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = module_0.Config()
    var_5 = 'django.contrib'
    var_6 = module_1.module(var_5, var_4)
    assert var_6 == 'THIRDPARTY'
    var_7 = '.local_module'
    var_8 = module_1.module(var_7, var_4)
    assert var_8 == 'LOCALFOLDER'
    var_9 = 'tests*'
    var_10 = [var_9]
    var_11 = module_0.Config()
    var_12 = 'tests.test_module'
    var_13 = module_1.module(var_12, var_11)
    assert var_13 == 'tests'
    var_14 = '/path/to/project'
    var_15 = 'unknown_module'
    var_16 = module_1.module(var_15, var_11)



# Parsed testcases at query #25
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'Test cases for the module function.'
    var_1 = module_0.Config()
    var_2 = 'test_mod'
    var_3 = 'test_mod.sub'
    var_4 = module_1.module(var_3, var_1)
    assert var_4 == 'test_mod'
    var_5 = '.local_mod'
    var_6 = module_1.module(var_5, var_1)
    var_7 = '^test_mod$'
    var_8 = 'TEST'
    var_9 = (var_7, var_8)
    var_10 = module_1.module(var_2, var_1)
    assert var_10 == 'TEST'
    var_11 = 'src'
    var_12 = 'src_module'
    var_13 = module_1.module(var_12, var_1)
    var_14 = 'unknown_module'
    var_15 = module_1.module(var_14, var_1)



# Parsed testcases at query #26
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'sys'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'STDLIB'
    var_5 = 'django'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'THIRDPARTY'
    var_7 = 'isort'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'FIRSTPARTY'
    var_9 = '.local_module'
    var_10 = module_1.module(var_9, var_0)
    assert var_10 == 'LOCALFOLDER'
    var_11 = 'local_module'
    var_12 = module_1.module(var_11, var_0)
    assert var_12 == 'FIRSTPARTY'



# Parsed testcases at query #27
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'sys'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'STDLIB'
    var_5 = 'django'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'THIRDPARTY'
    var_7 = 'numpy'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'THIRDPARTY'
    var_9 = 'isort'
    var_10 = module_1.module(var_9, var_0)
    assert var_10 == 'THIRDPARTY'
    var_11 = '.local_module'
    var_12 = module_1.module(var_11, var_0)
    assert var_12 == 'LOCALFOLDER'
    var_13 = 'local_module'
    var_14 = module_1.module(var_13, var_0)
    assert var_14 == 'FIRSTPARTY'
    var_15 = 'unknown_module'
    var_16 = module_1.module(var_15, var_0)
    assert var_16 == 'THIRDPARTY'
    var_17 = [var_5]
    var_18 = module_0.Config()
    var_19 = module_1.module(var_5, var_18)
    assert var_19 == 'django'
    var_20 = 'django.contrib'
    var_21 = module_1.module(var_20, var_18)
    assert var_21 == 'django'
    var_22 = '^django.*'
    var_23 = 'DJANGO'
    var_24 = (var_22, var_23)
    var_25 = [var_24]
    var_26 = module_0.Config()
    var_27 = module_1.module(var_5, var_26)
    assert var_27 == 'DJANGO'
    var_28 = module_1.module(var_20, var_26)
    assert var_28 == 'DJANGO'
    var_29 = '.'
    var_30 = [var_29]
    var_31 = module_0.Config()
    var_32 = 'test_module'
    var_33 = module_1.module(var_32, var_31)
    assert var_33 == 'FIRSTPARTY'



# Parsed testcases at query #28
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'sys'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'STDLIB'
    var_5 = 'math'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'STDLIB'
    var_7 = 'random'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'STDLIB'
    var_9 = 'collections'
    var_10 = module_1.module(var_9, var_0)
    assert var_10 == 'STDLIB'
    var_11 = 'datetime'
    var_12 = module_1.module(var_11, var_0)
    assert var_12 == 'STDLIB'
    var_13 = 'json'
    var_14 = module_1.module(var_13, var_0)
    assert var_14 == 'STDLIB'
    var_15 = 're'
    var_16 = module_1.module(var_15, var_0)
    assert var_16 == 'STDLIB'
    var_17 = 'fnmatch'
    var_18 = module_1.module(var_17, var_0)
    assert var_18 == 'STDLIB'
    var_19 = 'pathlib'
    var_20 = module_1.module(var_19, var_0)
    assert var_20 == 'STDLIB'
    var_21 = 'isort'
    var_22 = module_1.module(var_21, var_0)
    assert var_22 == 'THIRDPARTY'
    var_23 = 'pytest'
    var_24 = module_1.module(var_23, var_0)
    assert var_24 == 'THIRDPARTY'
    var_25 = '.local_module'
    var_26 = module_1.module(var_25, var_0)
    assert var_26 == 'LOCALFOLDER'
    var_27 = 'local_module'
    var_28 = module_1.module(var_27, var_0)
    assert var_28 == 'FIRSTPARTY'
    var_29 = 'numpy'
    var_30 = module_1.module(var_29, var_0)
    assert var_30 == 'THIRDPARTY'
    var_31 = 'pandas'
    var_32 = module_1.module(var_31, var_0)
    assert var_32 == 'THIRDPARTY'
    var_33 = 'requests'
    var_34 = module_1.module(var_33, var_0)
    assert var_34 == 'THIRDPARTY'
    var_35 = 'flask'
    var_36 = module_1.module(var_35, var_0)
    assert var_36 == 'THIRDPARTY'
    var_37 = 'django'
    var_38 = module_1.module(var_37, var_0)
    assert var_38 == 'THIRDPARTY'
    var_39 = 'tensorflow'
    var_40 = module_1.module(var_39, var_0)
    assert var_40 == 'THIRDPARTY'
    var_41 = 'torch'
    var_42 = module_1.module(var_41, var_0)
    assert var_42 == 'THIRDPARTY'
    var_43 = 'sklearn'
    var_44 = module_1.module(var_43, var_0)
    assert var_44 == 'THIRDPARTY'
    var_45 = 'matplotlib'
    var_46 = module_1.module(var_45, var_0)
    assert var_46 == 'THIRDPARTY'
    var_47 = 'seaborn'
    var_48 = module_1.module(var_47, var_0)
    assert var_48 == 'THIRDPARTY'
    var_49 = 'scipy'
    var_50 = module_1.module(var_49, var_0)
    assert var_50 == 'THIRDPARTY'
    var_51 = 'nltk'
    var_52 = module_1.module(var_51, var_0)
    assert var_52 == 'THIRDPARTY'
    var_53 = 'spacy'
    var_54 = module_1.module(var_53, var_0)
    assert var_54 == 'THIRDPARTY'
    var_55 = 'transformers'
    var_56 = module_1.module(var_55, var_0)
    assert var_56 == 'THIRDPARTY'
    var_57 = 'keras'
    var_58 = module_1.module(var_57, var_0)
    assert var_58 == 'THIRDPARTY'
    var_59 = 'opencv'
    var_60 = module_1.module(var_59, var_0)
    assert var_60 == 'THIRDPARTY'
    var_61 = 'cv2'
    var_62 = module_1.module(var_61, var_0)
    assert var_62 == 'THIRDPARTY'
    var_63 = 'pillow'
    var_64 = module_1.module(var_63, var_0)
    assert var_64 == 'THIRDPARTY'
    var_65 = 'PIL'
    var_66 = module_1.module(var_65, var_0)
    assert var_66 == 'THIRDPARTY'
    var_67 = 'pygame'
    var_68 = module_1.module(var_67, var_0)
    assert var_68 == 'THIRDPARTY'
    var_69 = 'tkinter'
    var_70 = module_1.module(var_69, var_0)
    assert var_70 == 'STDLIB'
    var_71 = 'sqlite3'
    var_72 = module_1.module(var_71, var_0)
    assert var_72 == 'STDLIB'
    var_73 = 'mysql.connector'
    var_74 = module_1.module(var_73, var_0)
    assert var_74 == 'THIRDPARTY'
    var_75 = 'psycopg2'
    var_76 = module_1.module(var_75, var_0)
    assert var_76 == 'THIRDPARTY'
    var_77 = 'sqlalchemy'
    var_78 = module_1.module(var_77, var_0)
    assert var_78 == 'THIRDPARTY'
    var_79 = 'bs4'
    var_80 = module_1.module(var_79, var_0)
    assert var_80 == 'THIRDPARTY'
    var_81 = 'beautifulsoup4'
    var_82 = module_1.module(var_81, var_0)
    assert var_82 == 'THIRDPARTY'
    var_83 = 'lxml'
    var_84 = module_1.module(var_83, var_0)
    assert var_84 == 'THIRDPARTY'
    var_85 = 'selenium'
    var_86 = module_1.module(var_85, var_0)
    assert var_86 == 'THIRDPARTY'
    var_87 = 'scrapy'
    var_88 = module_1.module(var_87, var_0)
    assert var_88 == 'THIRDPARTY'
    var_89 = 'urllib3'
    var_90 = module_1.module(var_89, var_0)
    assert var_90 == 'THIRDPARTY'
    var_91 = 'httpx'
    var_92 = module_1.module(var_91, var_0)
    assert var_92 == 'THIRDPARTY'
    var_93 = 'aiohttp'
    var_94 = module_1.module(var_93, var_0)
    assert var_94 == 'THIRDPARTY'
    var_95 = 'fastapi'
    var_96 = module_1.module(var_95, var_0)
    assert var_96 == 'THIRDPARTY'
    var_97 = 'uvicorn'
    var_98 = module_1.module(var_97, var_0)
    assert var_98 == 'THIRDPARTY'
    var_99 = 'starlette'
    var_100 = module_1.module(var_99, var_0)
    assert var_100 == 'THIRDPARTY'
    var_101 = 'pydantic'
    var_102 = module_1.module(var_101, var_0)
    assert var_102 == 'THIRDPARTY'
    var_103 = 'marshmallow'
    var_104 = module_1.module(var_103, var_0)
    assert var_104 == 'THIRDPARTY'
    var_105 = 'click'
    var_106 = module_1.module(var_105, var_0)
    assert var_106 == 'THIRDPARTY'
    var_107 = 'typer'
    var_108 = module_1.module(var_107, var_0)
    assert var_108 == 'THIRDPARTY'
    var_109 = 'rich'
    var_110 = module_1.module(var_109, var_0)
    assert var_110 == 'THIRDPARTY'
    var_111 = 'tqdm'
    var_112 = module_1.module(var_111, var_0)
    assert var_112 == 'THIRDPARTY'
    var_113 = 'loguru'
    var_114 = module_1.module(var_113, var_0)
    assert var_114 == 'THIRDPARTY'
    var_115 = 'structlog'
    var_116 = module_1.module(var_115, var_0)
    assert var_116 == 'THIRDPARTY'
    var_117 = 'colorama'
    var_118 = module_1.module(var_117, var_0)
    assert var_118 == 'THIRDPARTY'
    var_119 = 'pygments'
    var_120 = module_1.module(var_119, var_0)
    assert var_120 == 'THIRDPARTY'
    var_121 = 'black'
    var_122 = module_1.module(var_121, var_0)
    assert var_122 == 'THIRDPARTY'
    var_123 = 'flake8'
    var_124 = module_1.module(var_123, var_0)
    assert var_124 == 'THIRDPARTY'
    var_125 = 'mypy'
    var_126 = module_1.module(var_125, var_0)
    assert var_126 == 'THIRDPARTY'
    var_127 = 'pylint'
    var_128 = module_1.module(var_127, var_0)
    assert var_128 == 'THIRDPARTY'
    var_129 = module_1.module(var_21, var_0)
    assert var_129 == 'THIRDPARTY'
    var_130 = 'coverage'
    var_131 = module_1.module(var_130, var_0)
    assert var_131 == 'THIRDPARTY'
    var_132 = module_1.module(var_23, var_0)
    assert var_132 == 'THIRDPARTY'
    var_133 = 'hypothesis'
    var_134 = module_1.module(var_133, var_0)
    assert var_134 == 'THIRDPARTY'
    var_135 = 'tox'
    var_136 = module_1.module(var_135, var_0)
    assert var_136 == 'THIRDPARTY'
    var_137 = 'virtualenv'
    var_138 = module_1.module(var_137, var_0)
    assert var_138 == 'THIRDPARTY'
    var_139 = 'pip'
    var_140 = module_1.module(var_139, var_0)
    assert var_140 == 'STDLIB'
    var_141 = 'setuptools'
    var_142 = module_1.module(var_141, var_0)
    assert var_142 == 'THIRDPARTY'
    var_143 = 'wheel'
    var_144 = module_1.module(var_143, var_0)
    assert var_144 == 'THIRDPARTY'
    var_145 = 'twine'
    var_146 = module_1.module(var_145, var_0)
    assert var_146 == 'THIRDPARTY'
    var_147 = 'poetry'
    var_148 = module_1.module(var_147, var_0)
    assert var_148 == 'THIRDPARTY'
    var_149 = 'pipenv'
    var_150 = module_1.module(var_149, var_0)
    assert var_150 == 'THIRDPARTY'
    var_151 = 'conda'
    var_152 = module_1.module(var_151, var_0)
    assert var_152 == 'THIRDPARTY'
    var_153 = 'pyenv'
    var_154 = module_1.module(var_153, var_0)
    assert var_154 == 'THIRDPARTY'
    var_155 = 'virtualenvwrapper'
    var_156 = module_1.module(var_155, var_0)
    assert var_156 == 'THIRDPARTY'
    var_157 = 'fabric'
    var_158 = module_1.module(var_157, var_0)
    assert var_158 == 'THIRDPARTY'
    var_159 = 'invoke'
    var_160 = module_1.module(var_159, var_0)
    assert var_160 == 'THIRDPARTY'
    var_161 = 'paramiko'
    var_162 = module_1.module(var_161, var_0)
    assert var_162 == 'THIRDPARTY'
    var_163 = 'ssh'
    var_164 = module_1.module(var_163, var_0)
    assert var_164 == 'THIRDPARTY'
    var_165 = 'scp'
    var_166 = module_1.module(var_165, var_0)
    assert var_166 == 'THIRDPARTY'
    var_167 = 'sftp'
    var_168 = module_1.module(var_167, var_0)
    assert var_168 == 'THIRDPARTY'
    var_169 = 'pexpect'
    var_170 = module_1.module(var_169, var_0)
    assert var_170 == 'THIRDPARTY'
    var_171 = 'pty'
    var_172 = module_1.module(var_171, var_0)
    assert var_172 == 'STDLIB'
    var_173 = 'termcolor'
    var_174 = module_1.module(var_173, var_0)
    assert var_174 == 'THIRDPARTY'
    var_175 = 'blessed'
    var_176 = module_1.module(var_175, var_0)
    assert var_176 == 'THIRDPARTY'
    var_177 = 'prompt_toolkit'
    var_178 = module_1.module(var_177, var_0)
    assert var_178 == 'THIRDPARTY'
    var_179 = 'readline'
    var_180 = module_1.module(var_179, var_0)
    assert var_180 == 'STDLIB'
    var_181 = 'curses'
    var_182 = module_1.module(var_181, var_0)
    assert var_182 == 'STDLIB'
    var_183 = 'npyscreen'
    var_184 = module_1.module(var_183, var_0)
    assert var_184 == 'THIRDPARTY'
    var_185 = 'urwid'
    var_186 = module_1.module(var_185, var_0)
    assert var_186 == 'THIRDPARTY'
    var_187 = 'term'
    var_188 = module_1.module(var_187, var_0)
    assert var_188 == 'THIRDPARTY'
    var_189 = 'colorlog'
    var_190 = module_1.module(var_189, var_0)
    assert var_190 == 'THIRDPARTY'
    var_191 = 'pyfiglet'
    var_192 = module_1.module(var_191, var_0)
    assert var_192 == 'THIRDPARTY'
    var_193 = 'art'
    var_194 = module_1.module(var_193, var_0)
    assert var_194 == 'THIRDPARTY'
    var_195 = 'asciimatics'
    var_196 = module_1.module(var_195, var_0)
    assert var_196 == 'THIRDPARTY'
    var_197 = 'asciitree'
    var_198 = module_1.module(var_197, var_0)
    assert var_198 == 'THIRDPARTY'
    var_199 = 'asciichart'
    var_200 = module_1.module(var_199, var_0)
    assert var_200 == 'THIRDPARTY'
    var_201 = 'ascii_magic'
    var_202 = module_1.module(var_201, var_0)
    assert var_202 == 'THIRDPARTY'
    var_203 = 'ascii_graph'
    var_204 = module_1.module(var_203, var_0)
    assert var_204 == 'THIRDPARTY'
    var_205 = 'ascii_art'
    var_206 = module_1.module(var_205, var_0)
    assert var_206 == 'THIRDPARTY'
    var_207 = 'ascii_histogram'
    var_208 = module_1.module(var_207, var_0)
    assert var_208 == 'THIRDPARTY'
    var_209 = 'ascii_plot'
    var_210 = module_1.module(var_209, var_0)
    assert var_210 == 'THIRDPARTY'
    var_211 = 'ascii_sparkline'
    var_212 = module_1.module(var_211, var_0)
    assert var_212 == 'THIRDPARTY'
    var_213 = 'ascii_table'
    var_214 = module_1.module(var_213, var_0)
    assert var_214 == 'THIRDPARTY'
    var_215 = 'ascii_tree'
    var_216 = module_1.module(var_215, var_0)
    assert var_216 == 'THIRDPARTY'
    var_217 = 'ascii_utils'
    var_218 = module_1.module(var_217, var_0)
    assert var_218 == 'THIRDPARTY'
    var_219 = 'ascii_widget'
    var_220 = module_1.module(var_219, var_0)
    assert var_220 == 'THIRDPARTY'
    var_221 = 'ascii_widgets'
    var_222 = module_1.module(var_221, var_0)
    assert var_222 == 'THIRDPARTY'
    var_223 = 'ascii_widgets.widgets'
    var_224 = module_1.module(var_223, var_0)
    assert var_224 == 'THIRDPARTY'
    var_225 = 'ascii_widgets.widgets.base'
    var_226 = module_1.module(var_225, var_0)
    assert var_226 == 'THIRDPARTY'
    var_227 = 'ascii_widgets.widgets.text'
    var_228 = module_1.module(var_227, var_0)
    assert var_228 == 'THIRDPARTY'
    var_229 = 'ascii_widgets.widgets.button'
    var_230 = module_1.module(var_229, var_0)
    assert var_230 == 'THIRDPARTY'
    var_231 = 'ascii_widgets.widgets.label'
    var_232 = module_1.module(var_231, var_0)
    assert var_232 == 'THIRDPARTY'



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1)
    var_3 = 'os.path'
    var_4 = module_1.module(var_3)
    var_5 = '.local_module'
    var_6 = module_1.module(var_5, var_0)
    var_7 = 'isort'
    var_8 = module_1.module(var_7, var_0)
    var_9 = 'unknown_module'
    var_10 = module_1.module(var_9, var_0)



# Parsed testcases at query #2
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'django'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'THIRDPARTY'
    var_5 = 'my_local_module'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'FIRSTPARTY'
    var_7 = '.local_module'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'LOCALFOLDER'
    var_9 = '_private_module'
    var_10 = module_1.module(var_9, var_0)
    assert var_10 == 'FIRSTPARTY'
    var_11 = 'tests.test_module'
    var_12 = module_1.module(var_11, var_0)
    assert var_12 == 'FIRSTPARTY'
    var_13 = 'setup'
    var_14 = module_1.module(var_13, var_0)
    assert var_14 == 'FIRSTPARTY'
    var_15 = 'conftest'
    var_16 = module_1.module(var_15, var_0)
    assert var_16 == 'FIRSTPARTY'
    var_17 = 'my_package.setup'
    var_18 = module_1.module(var_17, var_0)
    assert var_18 == 'FIRSTPARTY'
    var_19 = 'my_package.tests.test_module'
    var_20 = module_1.module(var_19, var_0)
    assert var_20 == 'FIRSTPARTY'
    var_21 = 'my_package._private_module'
    var_22 = module_1.module(var_21, var_0)
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'my_package.local_module'
    var_24 = module_1.module(var_23, var_0)
    assert var_24 == 'FIRSTPARTY'
    var_25 = 'my_package..local_module'
    var_26 = module_1.module(var_25, var_0)
    assert var_26 == 'LOCALFOLDER'
    var_27 = 'my_package.._private_module'
    var_28 = module_1.module(var_27, var_0)
    assert var_28 == 'LOCALFOLDER'
    var_29 = 'my_package..tests.test_module'
    var_30 = module_1.module(var_29, var_0)
    assert var_30 == 'LOCALFOLDER'
    var_31 = 'my_package..setup'
    var_32 = module_1.module(var_31, var_0)
    assert var_32 == 'LOCALFOLDER'
    var_33 = 'my_package..conftest'
    var_34 = module_1.module(var_33, var_0)
    assert var_34 == 'LOCALFOLDER'
    var_35 = 'my_package..my_local_module'
    var_36 = module_1.module(var_35, var_0)
    assert var_36 == 'LOCALFOLDER'
    var_37 = 'my_package..django'
    var_38 = module_1.module(var_37, var_0)
    assert var_38 == 'LOCALFOLDER'
    var_39 = 'my_package..os'
    var_40 = module_1.module(var_39, var_0)
    assert var_40 == 'LOCALFOLDER'
    var_41 = 'my_package..my_package'
    var_42 = module_1.module(var_41, var_0)
    assert var_42 == 'LOCALFOLDER'
    var_43 = 'my_package..my_package.setup'
    var_44 = module_1.module(var_43, var_0)
    assert var_44 == 'LOCALFOLDER'
    var_45 = 'my_package..my_package.tests.test_module'
    var_46 = module_1.module(var_45, var_0)
    assert var_46 == 'LOCALFOLDER'
    var_47 = 'my_package..my_package._private_module'
    var_48 = module_1.module(var_47, var_0)
    assert var_48 == 'LOCALFOLDER'
    var_49 = 'my_package..my_package.local_module'
    var_50 = module_1.module(var_49, var_0)
    assert var_50 == 'LOCALFOLDER'
    var_51 = 'my_package..my_package..local_module'
    var_52 = module_1.module(var_51, var_0)
    assert var_52 == 'LOCALFOLDER'
    var_53 = 'my_package..my_package.._private_module'
    var_54 = module_1.module(var_53, var_0)
    assert var_54 == 'LOCALFOLDER'
    var_55 = 'my_package..my_package..tests.test_module'
    var_56 = module_1.module(var_55, var_0)
    assert var_56 == 'LOCALFOLDER'
    var_57 = 'my_package..my_package..setup'
    var_58 = module_1.module(var_57, var_0)
    assert var_58 == 'LOCALFOLDER'
    var_59 = 'my_package..my_package..conftest'
    var_60 = module_1.module(var_59, var_0)
    assert var_60 == 'LOCALFOLDER'
    var_61 = 'my_package..my_package..my_local_module'
    var_62 = module_1.module(var_61, var_0)
    assert var_62 == 'LOCALFOLDER'
    var_63 = 'my_package..my_package..django'
    var_64 = module_1.module(var_63, var_0)
    assert var_64 == 'LOCALFOLDER'
    var_65 = 'my_package..my_package..os'
    var_66 = module_1.module(var_65, var_0)
    assert var_66 == 'LOCALFOLDER'
    var_67 = 'my_package..my_package..my_package'
    var_68 = module_1.module(var_67, var_0)
    assert var_68 == 'LOCALFOLDER'
    var_69 = 'my_package..my_package..my_package.setup'
    var_70 = module_1.module(var_69, var_0)
    assert var_70 == 'LOCALFOLDER'
    var_71 = 'my_package..my_package..my_package.tests.test_module'
    var_72 = module_1.module(var_71, var_0)
    assert var_72 == 'LOCALFOLDER'
    var_73 = 'my_package..my_package..my_package._private_module'
    var_74 = module_1.module(var_73, var_0)
    assert var_74 == 'LOCALFOLDER'
    var_75 = 'my_package..my_package..my_package.local_module'
    var_76 = module_1.module(var_75, var_0)
    assert var_76 == 'LOCALFOLDER'
    var_77 = 'my_package..my_package..my_package..local_module'
    var_78 = module_1.module(var_77, var_0)
    assert var_78 == 'LOCALFOLDER'
    var_79 = 'my_package..my_package..my_package.._private_module'
    var_80 = module_1.module(var_79, var_0)
    assert var_80 == 'LOCALFOLDER'
    var_81 = 'my_package..my_package..my_package..tests.test_module'
    var_82 = module_1.module(var_81, var_0)
    assert var_82 == 'LOCALFOLDER'
    var_83 = 'my_package..my_package..my_package..setup'
    var_84 = module_1.module(var_83, var_0)
    assert var_84 == 'LOCALFOLDER'
    var_85 = 'my_package..my_package..my_package..conftest'
    var_86 = module_1.module(var_85, var_0)
    assert var_86 == 'LOCALFOLDER'
    var_87 = 'my_package..my_package..my_package..my_local_module'
    var_88 = module_1.module(var_87, var_0)
    assert var_88 == 'LOCALFOLDER'
    var_89 = 'my_package..my_package..my_package..django'
    var_90 = module_1.module(var_89, var_0)
    assert var_90 == 'LOCALFOLDER'
    var_91 = 'my_package..my_package..my_package..os'
    var_92 = module_1.module(var_91, var_0)
    assert var_92 == 'LOCALFOLDER'
    var_93 = 'my_package..my_package..my_package..my_package'
    var_94 = module_1.module(var_93, var_0)
    assert var_94 == 'LOCALFOLDER'
    var_95 = 'my_package..my_package..my_package..my_package.setup'
    var_96 = module_1.module(var_95, var_0)
    assert var_96 == 'LOCALFOLDER'
    var_97 = 'my_package..my_package..my_package..my_package.tests.test_module'
    var_98 = module_1.module(var_97, var_0)
    assert var_98 == 'LOCALFOLDER'
    var_99 = 'my_package..my_package..my_package..my_package._private_module'
    var_100 = module_1.module(var_99, var_0)
    assert var_100 == 'LOCALFOLDER'
    var_101 = 'my_package..my_package..my_package..my_package.local_module'
    var_102 = module_1.module(var_101, var_0)
    assert var_102 == 'LOCALFOLDER'
    var_103 = 'my_package..my_package..my_package..my_package..local_module'
    var_104 = module_1.module(var_103, var_0)
    assert var_104 == 'LOCALFOLDER'
    var_105 = 'my_package..my_package..my_package..my_package.._private_module'
    var_106 = module_1.module(var_105, var_0)
    assert var_106 == 'LOCALFOLDER'
    var_107 = 'my_package..my_package..my_package..my_package..tests.test_module'
    var_108 = module_1.module(var_107, var_0)
    assert var_108 == 'LOCALFOLDER'
    var_109 = 'my_package..my_package..my_package..my_package..setup'
    var_110 = module_1.module(var_109, var_0)
    assert var_110 == 'LOCALFOLDER'
    var_111 = 'my_package..my_package..my_package..my_package..conftest'
    var_112 = module_1.module(var_111, var_0)
    assert var_112 == 'LOCALFOLDER'
    var_113 = 'my_package..my_package..my_package..my_package..my_local_module'
    var_114 = module_1.module(var_113, var_0)
    assert var_114 == 'LOCALFOLDER'
    var_115 = 'my_package..my_package..my_package..my_package..django'
    var_116 = module_1.module(var_115, var_0)
    assert var_116 == 'LOCALFOLDER'
    var_117 = 'my_package..my_package..my_package..my_package..os'
    var_118 = module_1.module(var_117, var_0)
    assert var_118 == 'LOCALFOLDER'
    var_119 = 'my_package..my_package..my_package..my_package..my_package'
    var_120 = module_1.module(var_119, var_0)
    assert var_120 == 'LOCALFOLDER'
    var_121 = 'my_package..my_package..my_package..my_package..my_package.setup'
    var_122 = module_1.module(var_121, var_0)
    assert var_122 == 'LOCALFOLDER'
    var_123 = 'my_package..my_package..my_package..my_package..my_package.tests.test_module'
    var_124 = module_1.module(var_123, var_0)
    assert var_124 == 'LOCALFOLDER'
    var_125 = 'my_package..my_package..my_package..my_package..my_package._private_module'
    var_126 = module_1.module(var_125, var_0)
    assert var_126 == 'LOCALFOLDER'
    var_127 = 'my_package..my_package..my_package..my_package..my_package.local_module'
    var_128 = module_1.module(var_127, var_0)
    assert var_128 == 'LOCALFOLDER'
    var_129 = 'my_package..my_package..my_package..my_package..my_package..local_module'
    var_130 = module_1.module(var_129, var_0)
    assert var_130 == 'LOCALFOLDER'
    var_131 = 'my_package..my_package..my_package..my_package..my_package.._private_module'
    var_132 = module_1.module(var_131, var_0)
    assert var_132 == 'LOCALFOLDER'
    var_133 = 'my_package..my_package..my_package..my_package..my_package..tests.test_module'
    var_134 = module_1.module(var_133, var_0)
    assert var_134 == 'LOCALFOLDER'
    var_135 = 'my_package..my_package..my_package..my_package..my_package..setup'
    var_136 = module_1.module(var_135, var_0)
    assert var_136 == 'LOCALFOLDER'
    var_137 = 'my_package..my_package..my_package..my_package..my_package..conftest'
    var_138 = module_1.module(var_137, var_0)
    assert var_138 == 'LOCALFOLDER'
    var_139 = 'my_package..my_package..my_package..my_package..my_package..my_local_module'
    var_140 = module_1.module(var_139, var_0)
    assert var_140 == 'LOCALFOLDER'
    var_141 = 'my_package..my_package..my_package..my_package..my_package..django'
    var_142 = module_1.module(var_141, var_0)
    assert var_142 == 'LOCALFOLDER'
    var_143 = 'my_package..my_package..my_package..my_package..my_package..os'
    var_144 = module_1.module(var_143, var_0)
    assert var_144 == 'LOCALFOLDER'
    var_145 = 'my_package..my_package..my_package..my_package..my_package..my_package'
    var_146 = module_1.module(var_145, var_0)
    assert var_146 == 'LOCALFOLDER'
    var_147 = 'my_package..my_package..my_package..my_package..my_package..my_package.setup'
    var_148 = module_1.module(var_147, var_0)
    assert var_148 == 'LOCALFOLDER'
    var_149 = 'my_package..my_package..my_package..my_package..my_package..my_package.tests.test_module'
    var_150 = module_1.module(var_149, var_0)
    assert var_150 == 'LOCALFOLDER'



# Parsed testcases at query #3
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'sys'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'STDLIB'
    var_5 = 'django'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'THIRDPARTY'
    var_7 = 'numpy'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'THIRDPARTY'
    var_9 = 'local_module'
    var_10 = module_1.module(var_9, var_0)
    assert var_10 == 'FIRSTPARTY'
    var_11 = '.local_module'
    var_12 = module_1.module(var_11, var_0)
    assert var_12 == 'LOCALFOLDER'
    var_13 = '_private_module'
    var_14 = module_1.module(var_13, var_0)
    assert var_14 == 'FIRSTPARTY'
    var_15 = '__main__'
    var_16 = module_1.module(var_15, var_0)
    assert var_16 == 'STDLIB'
    var_17 = 'pytest'
    var_18 = module_1.module(var_17, var_0)
    assert var_18 == 'THIRDPARTY'
    var_19 = 'unittest'
    var_20 = module_1.module(var_19, var_0)
    assert var_20 == 'STDLIB'
    var_21 = 'conftest'
    var_22 = module_1.module(var_21, var_0)
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'setup'
    var_24 = module_1.module(var_23, var_0)
    assert var_24 == 'FIRSTPARTY'
    var_25 = 'tests'
    var_26 = module_1.module(var_25, var_0)
    assert var_26 == 'FIRSTPARTY'
    var_27 = 'src'
    var_28 = module_1.module(var_27, var_0)
    assert var_28 == 'FIRSTPARTY'
    var_29 = 'lib'
    var_30 = module_1.module(var_29, var_0)
    assert var_30 == 'FIRSTPARTY'
    var_31 = 'bin'
    var_32 = module_1.module(var_31, var_0)
    assert var_32 == 'FIRSTPARTY'
    var_33 = 'docs'
    var_34 = module_1.module(var_33, var_0)
    assert var_34 == 'FIRSTPARTY'
    var_35 = 'scripts'
    var_36 = module_1.module(var_35, var_0)
    assert var_36 == 'FIRSTPARTY'
    var_37 = 'config'
    var_38 = module_1.module(var_37, var_0)
    assert var_38 == 'FIRSTPARTY'
    var_39 = 'settings'
    var_40 = module_1.module(var_39, var_0)
    assert var_40 == 'FIRSTPARTY'
    var_41 = 'utils'
    var_42 = module_1.module(var_41, var_0)
    assert var_42 == 'FIRSTPARTY'
    var_43 = 'helpers'
    var_44 = module_1.module(var_43, var_0)
    assert var_44 == 'FIRSTPARTY'
    var_45 = 'models'
    var_46 = module_1.module(var_45, var_0)
    assert var_46 == 'FIRSTPARTY'
    var_47 = 'views'
    var_48 = module_1.module(var_47, var_0)
    assert var_48 == 'FIRSTPARTY'
    var_49 = 'controllers'
    var_50 = module_1.module(var_49, var_0)
    assert var_50 == 'FIRSTPARTY'
    var_51 = 'services'
    var_52 = module_1.module(var_51, var_0)
    assert var_52 == 'FIRSTPARTY'
    var_53 = 'api'
    var_54 = module_1.module(var_53, var_0)
    assert var_54 == 'FIRSTPARTY'
    var_55 = 'web'
    var_56 = module_1.module(var_55, var_0)
    assert var_56 == 'FIRSTPARTY'
    var_57 = 'cli'
    var_58 = module_1.module(var_57, var_0)
    assert var_58 == 'FIRSTPARTY'
    var_59 = 'commands'
    var_60 = module_1.module(var_59, var_0)
    assert var_60 == 'FIRSTPARTY'
    var_61 = 'tasks'
    var_62 = module_1.module(var_61, var_0)
    assert var_62 == 'FIRSTPARTY'
    var_63 = 'jobs'
    var_64 = module_1.module(var_63, var_0)
    assert var_64 == 'FIRSTPARTY'
    var_65 = 'workers'
    var_66 = module_1.module(var_65, var_0)
    assert var_66 == 'FIRSTPARTY'
    var_67 = 'queues'
    var_68 = module_1.module(var_67, var_0)
    assert var_68 == 'FIRSTPARTY'
    var_69 = 'events'
    var_70 = module_1.module(var_69, var_0)
    assert var_70 == 'FIRSTPARTY'
    var_71 = 'handlers'
    var_72 = module_1.module(var_71, var_0)
    assert var_72 == 'FIRSTPARTY'
    var_73 = 'middleware'
    var_74 = module_1.module(var_73, var_0)
    assert var_74 == 'FIRSTPARTY'
    var_75 = 'filters'
    var_76 = module_1.module(var_75, var_0)
    assert var_76 == 'FIRSTPARTY'
    var_77 = 'decorators'
    var_78 = module_1.module(var_77, var_0)
    assert var_78 == 'FIRSTPARTY'
    var_79 = 'mixins'
    var_80 = module_1.module(var_79, var_0)
    assert var_80 == 'FIRSTPARTY'
    var_81 = 'exceptions'
    var_82 = module_1.module(var_81, var_0)
    assert var_82 == 'FIRSTPARTY'
    var_83 = 'constants'
    var_84 = module_1.module(var_83, var_0)
    assert var_84 == 'FIRSTPARTY'
    var_85 = 'types'
    var_86 = module_1.module(var_85, var_0)
    assert var_86 == 'FIRSTPARTY'
    var_87 = 'interfaces'
    var_88 = module_1.module(var_87, var_0)
    assert var_88 == 'FIRSTPARTY'
    var_89 = 'abstracts'
    var_90 = module_1.module(var_89, var_0)
    assert var_90 == 'FIRSTPARTY'
    var_91 = 'base'
    var_92 = module_1.module(var_91, var_0)
    assert var_92 == 'FIRSTPARTY'
    var_93 = 'core'
    var_94 = module_1.module(var_93, var_0)
    assert var_94 == 'FIRSTPARTY'
    var_95 = 'common'
    var_96 = module_1.module(var_95, var_0)
    assert var_96 == 'FIRSTPARTY'
    var_97 = 'shared'
    var_98 = module_1.module(var_97, var_0)
    assert var_98 == 'FIRSTPARTY'
    var_99 = module_1.module(var_41, var_0)
    assert var_99 == 'FIRSTPARTY'
    var_100 = module_1.module(var_43, var_0)
    assert var_100 == 'FIRSTPARTY'
    var_101 = 'tools'
    var_102 = module_1.module(var_101, var_0)
    assert var_102 == 'FIRSTPARTY'
    var_103 = 'extensions'
    var_104 = module_1.module(var_103, var_0)
    assert var_104 == 'FIRSTPARTY'
    var_105 = 'plugins'
    var_106 = module_1.module(var_105, var_0)
    assert var_106 == 'FIRSTPARTY'
    var_107 = 'integrations'
    var_108 = module_1.module(var_107, var_0)
    assert var_108 == 'FIRSTPARTY'
    var_109 = 'adapters'
    var_110 = module_1.module(var_109, var_0)
    assert var_110 == 'FIRSTPARTY'
    var_111 = 'connectors'
    var_112 = module_1.module(var_111, var_0)
    assert var_112 == 'FIRSTPARTY'
    var_113 = 'drivers'
    var_114 = module_1.module(var_113, var_0)
    assert var_114 == 'FIRSTPARTY'
    var_115 = 'providers'
    var_116 = module_1.module(var_115, var_0)
    assert var_116 == 'FIRSTPARTY'
    var_117 = 'clients'
    var_118 = module_1.module(var_117, var_0)
    assert var_118 == 'FIRSTPARTY'
    var_119 = 'servers'
    var_120 = module_1.module(var_119, var_0)
    assert var_120 == 'FIRSTPARTY'
    var_121 = 'proxies'
    var_122 = module_1.module(var_121, var_0)
    assert var_122 == 'FIRSTPARTY'
    var_123 = 'gateways'
    var_124 = module_1.module(var_123, var_0)
    assert var_124 == 'FIRSTPARTY'
    var_125 = 'brokers'
    var_126 = module_1.module(var_125, var_0)
    assert var_126 == 'FIRSTPARTY'
    var_127 = module_1.module(var_67, var_0)
    assert var_127 == 'FIRSTPARTY'
    var_128 = 'streams'
    var_129 = module_1.module(var_128, var_0)
    assert var_129 == 'FIRSTPARTY'
    var_130 = 'pipelines'
    var_131 = module_1.module(var_130, var_0)
    assert var_131 == 'FIRSTPARTY'
    var_132 = 'processors'
    var_133 = module_1.module(var_132, var_0)
    assert var_133 == 'FIRSTPARTY'
    var_134 = 'transformers'
    var_135 = module_1.module(var_134, var_0)
    assert var_135 == 'FIRSTPARTY'
    var_136 = 'loaders'
    var_137 = module_1.module(var_136, var_0)
    assert var_137 == 'FIRSTPARTY'
    var_138 = 'parsers'
    var_139 = module_1.module(var_138, var_0)
    assert var_139 == 'FIRSTPARTY'
    var_140 = 'serializers'
    var_141 = module_1.module(var_140, var_0)
    assert var_141 == 'FIRSTPARTY'
    var_142 = 'validators'
    var_143 = module_1.module(var_142, var_0)
    assert var_143 == 'FIRSTPARTY'
    var_144 = 'normalizers'
    var_145 = module_1.module(var_144, var_0)
    assert var_145 == 'FIRSTPARTY'
    var_146 = 'formatters'
    var_147 = module_1.module(var_146, var_0)
    assert var_147 == 'FIRSTPARTY'
    var_148 = 'renderers'
    var_149 = module_1.module(var_148, var_0)
    assert var_149 == 'FIRSTPARTY'
    var_150 = 'generators'
    var_151 = module_1.module(var_150, var_0)
    assert var_151 == 'FIRSTPARTY'
    var_152 = 'factories'
    var_153 = module_1.module(var_152, var_0)
    assert var_153 == 'FIRSTPARTY'
    var_154 = 'builders'
    var_155 = module_1.module(var_154, var_0)
    assert var_155 == 'FIRSTPARTY'
    var_156 = 'assemblers'
    var_157 = module_1.module(var_156, var_0)
    assert var_157 == 'FIRSTPARTY'
    var_158 = 'composers'
    var_159 = module_1.module(var_158, var_0)
    assert var_159 == 'FIRSTPARTY'
    var_160 = 'orchestrators'
    var_161 = module_1.module(var_160, var_0)
    assert var_161 == 'FIRSTPARTY'
    var_162 = 'managers'
    var_163 = module_1.module(var_162, var_0)
    assert var_163 == 'FIRSTPARTY'
    var_164 = 'directors'
    var_165 = module_1.module(var_164, var_0)
    assert var_165 == 'FIRSTPARTY'
    var_166 = 'coordinators'
    var_167 = module_1.module(var_166, var_0)
    assert var_167 == 'FIRSTPARTY'
    var_168 = 'supervisors'
    var_169 = module_1.module(var_168, var_0)
    assert var_169 == 'FIRSTPARTY'
    var_170 = 'monitors'
    var_171 = module_1.module(var_170, var_0)
    assert var_171 == 'FIRSTPARTY'
    var_172 = 'observers'
    var_173 = module_1.module(var_172, var_0)
    assert var_173 == 'FIRSTPARTY'
    var_174 = 'listeners'
    var_175 = module_1.module(var_174, var_0)
    assert var_175 == 'FIRSTPARTY'
    var_176 = 'watchers'
    var_177 = module_1.module(var_176, var_0)
    assert var_177 == 'FIRSTPARTY'
    var_178 = 'trackers'
    var_179 = module_1.module(var_178, var_0)
    assert var_179 == 'FIRSTPARTY'
    var_180 = 'loggers'
    var_181 = module_1.module(var_180, var_0)
    assert var_181 == 'FIRSTPARTY'
    var_182 = 'reporters'
    var_183 = module_1.module(var_182, var_0)
    assert var_183 == 'FIRSTPARTY'
    var_184 = 'exporters'
    var_185 = module_1.module(var_184, var_0)
    assert var_185 == 'FIRSTPARTY'
    var_186 = 'importers'
    var_187 = module_1.module(var_186, var_0)
    assert var_187 == 'FIRSTPARTY'
    var_188 = 'migrators'
    var_189 = module_1.module(var_188, var_0)
    assert var_189 == 'FIRSTPARTY'
    var_190 = 'upgraders'
    var_191 = module_1.module(var_190, var_0)
    assert var_191 == 'FIRSTPARTY'
    var_192 = 'downgraders'
    var_193 = module_1.module(var_192, var_0)
    assert var_193 == 'FIRSTPARTY'
    var_194 = 'converters'
    var_195 = module_1.module(var_194, var_0)
    assert var_195 == 'FIRSTPARTY'
    var_196 = 'translators'
    var_197 = module_1.module(var_196, var_0)
    assert var_197 == 'FIRSTPARTY'
    var_198 = 'interpreters'
    var_199 = module_1.module(var_198, var_0)
    assert var_199 == 'FIRSTPARTY'
    var_200 = 'executors'
    var_201 = module_1.module(var_200, var_0)
    assert var_201 == 'FIRSTPARTY'
    var_202 = 'runners'
    var_203 = module_1.module(var_202, var_0)
    assert var_203 == 'FIRSTPARTY'
    var_204 = 'schedulers'
    var_205 = module_1.module(var_204, var_0)
    assert var_205 == 'FIRSTPARTY'
    var_206 = 'timers'
    var_207 = module_1.module(var_206, var_0)
    assert var_207 == 'FIRSTPARTY'
    var_208 = 'triggers'
    var_209 = module_1.module(var_208, var_0)
    assert var_209 == 'FIRSTPARTY'
    var_210 = 'activators'
    var_211 = module_1.module(var_210, var_0)
    assert var_211 == 'FIRSTPARTY'
    var_212 = 'deactivators'
    var_213 = module_1.module(var_212, var_0)
    assert var_213 == 'FIRSTPARTY'
    var_214 = 'enablers'
    var_215 = module_1.module(var_214, var_0)
    assert var_215 == 'FIRSTPARTY'
    var_216 = 'disablers'
    var_217 = module_1.module(var_216, var_0)
    assert var_217 == 'FIRSTPARTY'
    var_218 = 'togglers'
    var_219 = module_1.module(var_218, var_0)
    assert var_219 == 'FIRSTPARTY'
    var_220 = 'switchers'
    var_221 = module_1.module(var_220, var_0)
    assert var_221 == 'FIRSTPARTY'
    var_222 = 'selectors'
    var_223 = module_1.module(var_222, var_0)
    assert var_223 == 'FIRSTPARTY'
    var_224 = module_1.module(var_75, var_0)
    assert var_224 == 'FIRSTPARTY'
    var_225 = 'sorters'
    var_226 = module_1.module(var_225, var_0)
    assert var_226 == 'FIRSTPARTY'
    var_227 = 'groupers'
    var_228 = module_1.module(var_227, var_0)
    assert var_228 == 'FIRSTPARTY'
    var_229 = 'aggregators'
    var_230 = module_1.module(var_229, var_0)
    assert var_230 == 'FIRSTPARTY'
    var_231 = 'reducers'
    var_232 = module_1.module(var_231, var_0)
    assert var_232 == 'FIRSTPARTY'
    var_233 = 'mappers'
    var_234 = module_1.module(var_233, var_0)
    assert var_234 == 'FIRSTPARTY'
    var_235 = 'projectors'
    var_236 = module_1.module(var_235, var_0)
    assert var_236 == 'FIRSTPARTY'
    var_237 = 'extractors'
    var_238 = module_1.module(var_237, var_0)
    assert var_238 == 'FIRSTPARTY'
    var_239 = 'injectors'
    var_240 = module_1.module(var_239, var_0)
    assert var_240 == 'FIRSTPARTY'
    var_241 = 'binders'
    var_242 = module_1.module(var_241, var_0)
    assert var_242 == 'FIRSTPARTY'
    var_243 = 'linkers'
    var_244 = module_1.module(var_243, var_0)
    assert var_244 == 'FIRSTPARTY'
    var_245 = 'joiners'
    var_246 = module_1.module(var_245, var_0)
    assert var_246 == 'FIRSTPARTY'
    var_247 = 'mergers'
    var_248 = module_1.module(var_247, var_0)
    assert var_248 == 'FIRSTPARTY'
    var_249 = 'splitters'
    var_250 = module_1.module(var_249, var_0)
    assert var_250 == 'FIRSTPARTY'
    var_251 = 'dividers'
    var_252 = module_1.module(var_251, var_0)
    assert var_252 == 'FIRSTPARTY'



# Parsed testcases at query #4
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'sys'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'STDLIB'
    var_5 = 'django'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'THIRDPARTY'
    var_7 = 'numpy'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'THIRDPARTY'
    var_9 = '.local_module'
    var_10 = module_1.module(var_9, var_0)
    assert var_10 == 'LOCALFOLDER'
    var_11 = 'local_module'
    var_12 = module_1.module(var_11, var_0)
    assert var_12 == 'FIRSTPARTY'
    var_13 = 'unknown_module'
    var_14 = module_1.module(var_13, var_0)
    assert var_14 == 'THIRDPARTY'
    var_15 = 'tests'
    var_16 = [var_15]
    var_17 = module_0.Config()
    var_18 = module_1.module(var_15, var_17)
    assert var_18 == 'tests'
    var_19 = 'tests.module'
    var_20 = module_1.module(var_19, var_17)
    assert var_20 == 'tests'
    var_21 = '^test_.*'
    var_22 = 'TESTS'
    var_23 = (var_21, var_22)
    var_24 = [var_23]
    var_25 = module_0.Config()
    var_26 = 'test_module'
    var_27 = module_1.module(var_26, var_25)
    assert var_27 == 'TESTS'
    var_28 = 'test.utils'
    var_29 = module_1.module(var_28, var_25)
    assert var_29 == 'TESTS'
    var_30 = 'namespace'
    var_31 = [var_30]
    var_32 = module_0.Config()
    var_33 = 'namespace.module'
    var_34 = module_1.module(var_33, var_32)
    assert var_34 == 'FIRSTPARTY'
    var_35 = 'All tests passed!'
    var_36 = print(var_35)



# Parsed testcases at query #5
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'sys'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'STDLIB'
    var_5 = 'django'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'THIRDPARTY'
    var_7 = 'numpy'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'THIRDPARTY'
    var_9 = 'isort'
    var_10 = module_1.module(var_9, var_0)
    assert var_10 == 'FIRSTPARTY'
    var_11 = '.local_module'
    var_12 = module_1.module(var_11, var_0)
    assert var_12 == 'LOCALFOLDER'
    var_13 = 'local_module'
    var_14 = module_1.module(var_13, var_0)
    assert var_14 == 'FIRSTPARTY'
    var_15 = 'All tests passed!'
    var_16 = print(var_15)



# Parsed testcases at query #6
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = '.local_module'
    var_1 = module_0.module(var_0)
    var_2 = 'django.*'
    var_3 = 'THIRDPARTY'
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = module_1.Config()
    var_7 = 'django.contrib'
    var_8 = module_0.module(var_7)
    assert var_8 == 'THIRDPARTY'
    var_9 = 'tests'
    var_10 = [var_9]
    var_11 = module_1.Config()
    var_12 = 'tests.module'
    var_13 = module_0.module(var_12)
    assert var_13 == 'tests'
    var_14 = '/path/to/src'
    var_15 = 'src.module'
    var_16 = module_0.module(var_15)
    assert var_16 == 'FIRSTPARTY'
    var_17 = 'unknown_module'
    var_18 = module_0.module(var_17)
    assert var_18 == 'STDLIB'



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'isort'
    var_1 = 'THIRDPARTY'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = 'forced'
    var_5 = [var_4]
    var_6 = 'src'
    var_7 = 'STDLIB'
    var_8 = 'local.module'
    var_9 = '.local.module'



# Parsed testcases at query #8
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'numpy'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'THIRDPARTY'
    var_5 = 'local_module'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'FIRSTPARTY'
    var_7 = '.local_module'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'LOCALFOLDER'
    var_9 = 'unknown_module'
    var_10 = module_1.module(var_9, var_0)
    assert var_10 == 'THIRDPARTY'



# Parsed testcases at query #9
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'django'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'THIRDPARTY'
    var_5 = '.local_module'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'LOCALFOLDER'
    var_7 = 'my_project'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'FIRSTPARTY'
    var_9 = 'unknown_module'
    var_10 = module_1.module(var_9, var_0)
    assert var_10 == 'THIRDPARTY'
    var_11 = 'special_module'
    var_12 = module_1.module(var_11, var_0)
    assert var_12 == 'special_module'
    var_13 = 'special_module.submodule'
    var_14 = module_1.module(var_13, var_0)
    assert var_14 == 'special_module'
    var_15 = '^my_pattern.*'
    var_16 = 'CUSTOM'
    var_17 = (var_15, var_16)
    var_18 = 'my_pattern_module'
    var_19 = module_1.module(var_18, var_0)
    assert var_19 == 'CUSTOM'
    var_20 = 'my_namespace'
    var_21 = 'my_namespace.module'
    var_22 = module_1.module(var_21, var_0)
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'All tests passed!'
    var_24 = print(var_23)



# Parsed testcases at query #10
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'django'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'THIRDPARTY'
    var_5 = 'my_local_module'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'FIRSTPARTY'
    var_7 = '.local_module'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'LOCALFOLDER'
    var_9 = 'my_namespace.module'
    var_10 = module_1.module(var_9, var_0)
    assert var_10 == 'FIRSTPARTY'
    var_11 = 'All tests passed!'
    var_12 = print(var_11)



# Parsed testcases at query #11
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'numpy'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'THIRDPARTY'
    var_5 = 'my_project'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'FIRSTPARTY'
    var_7 = '.local_module'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'LOCALFOLDER'
    var_9 = 'unknown_module'
    var_10 = module_1.module(var_9, var_0)
    assert var_10 == 'THIRDPARTY'



# Parsed testcases at query #12
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'django'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'THIRDPARTY'
    var_5 = '.local_module'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'LOCALFOLDER'
    var_7 = 'my_project.my_module'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'FIRSTPARTY'
    var_9 = 'unknown_module'
    var_10 = module_1.module(var_9, var_0)
    assert var_10 == 'THIRDPARTY'



# Parsed testcases at query #13
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'isort'
    var_1 = 'THIRDPARTY'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = module_0.Config()
    var_5 = module_1.module(var_0, var_4)
    assert var_5 == 'THIRDPARTY'
    var_6 = '.local_module'
    var_7 = module_1.module(var_6)
    assert var_7 == 'LOCALFOLDER'
    var_8 = '/path/to/src'
    var_9 = 'src_module'
    var_10 = module_1.module(var_9, var_4)
    assert var_10 == 'FIRSTPARTY'
    var_11 = 'unknown_module'
    var_12 = module_1.module(var_11)
    assert var_12 == 'STDLIB'
    var_13 = 'separate_module'
    var_14 = [var_13]
    var_15 = module_0.Config()
    var_16 = module_1.module(var_13, var_15)
    assert var_16 == 'separate_module'
    var_17 = 'ns_pkg'
    var_18 = [var_17]
    var_19 = module_0.Config()
    var_20 = 'ns_pkg.sub_module'
    var_21 = module_1.module(var_20, var_19)
    assert var_21 == 'FIRSTPARTY'
    var_22 = 'All unit tests for module function passed.'
    var_23 = print(var_22)



# Parsed testcases at query #14
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'django'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'THIRDPARTY'
    var_5 = '.local_module'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'LOCALFOLDER'
    var_7 = 'my_project.module'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'FIRSTPARTY'
    var_9 = 'unknown_module'
    var_10 = module_1.module(var_9, var_0)
    var_11 = 'special_module'
    var_12 = module_1.module(var_11, var_0)
    assert var_12 == 'special_module'
    var_13 = 'special_module.submodule'
    var_14 = module_1.module(var_13, var_0)
    assert var_14 == 'special_module'
    var_15 = 'test_*'
    var_16 = 'TESTS'
    var_17 = (var_15, var_16)
    var_18 = 'test_module'
    var_19 = module_1.module(var_18, var_0)
    assert var_19 == 'TESTS'
    var_20 = 'All tests passed!'
    var_21 = print(var_20)



# Parsed testcases at query #15
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'THIRDPARTY'
    var_1 = module_0.Config()
    var_2 = 'django.core'
    var_3 = module_1.module(var_2, var_1)
    assert var_3 == 'THIRDPARTY'
    var_4 = 'django'
    var_5 = module_1.module(var_2, var_1)
    assert var_5 == 'django'
    var_6 = '.local_module'
    var_7 = module_1.module(var_6, var_1)
    assert var_7 == 'LOCALFOLDER'
    var_8 = '/path/to/src'
    var_9 = 'src_module'
    var_10 = module_1.module(var_9, var_1)
    assert var_10 == 'FIRSTPARTY'
    var_11 = 'unknown_module'
    var_12 = module_1.module(var_11, var_1)
    assert var_12 == 'THIRDPARTY'



# Parsed testcases at query #16
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'django'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'THIRDPARTY'
    var_5 = 'local_module'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'FIRSTPARTY'
    var_7 = '.local_module'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'LOCALFOLDER'
    var_9 = 'isort'
    var_10 = module_1.module(var_9, var_0)
    assert var_10 == 'FIRSTPARTY'



# Parsed testcases at query #17
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'numpy'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'THIRDPARTY'
    var_5 = '._internal'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'LOCALFOLDER'
    var_7 = 'isort'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'FIRSTPARTY'
    var_9 = 'unknown_module'
    var_10 = module_1.module(var_9, var_0)
    assert var_10 == 'THIRDPARTY'



# Parsed testcases at query #18
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '^django$'
    var_1 = 'THIRDPARTY'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = module_0.Config()
    var_5 = 'django'
    var_6 = module_1.module(var_5, var_4)
    assert var_6 == 'THIRDPARTY'
    var_7 = '.local_module'
    var_8 = module_1.module(var_7, var_4)
    assert var_8 == 'LOCALFOLDER'
    var_9 = 'tests'
    var_10 = [var_9]
    var_11 = module_0.Config()
    var_12 = module_1.module(var_9, var_11)
    assert var_12 == 'tests'
    var_13 = 'unknown_module'
    var_14 = module_1.module(var_13, var_11)
    var_15 = '/path/to/project'
    var_16 = True
    var_17 = lambda path: var_16
    var_18 = 'project_module'
    var_19 = module_1.module(var_18, var_11)
    assert var_19 == 'FIRSTPARTY'
    var_20 = 'All tests passed!'
    var_21 = print(var_20)



# Parsed testcases at query #19
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = module_1.module(var_0, var_2)
    assert var_3 == 'test_module'
    var_4 = 'test_module.submodule'
    var_5 = module_1.module(var_4, var_2)
    assert var_5 == 'test_module'
    var_6 = 'another_module'
    var_7 = module_1.module(var_6, var_2)
    var_8 = '.local_module'
    var_9 = module_1.module(var_8)
    assert var_9 == 'LOCALFOLDER'
    var_10 = 'local_module'
    var_11 = module_1.module(var_10)
    var_12 = '^test.*'
    var_13 = 'tests'
    var_14 = module_1.module(var_0, var_2)
    assert var_14 == 'tests'
    var_15 = module_1.module(var_4, var_2)
    assert var_15 == 'tests'
    var_16 = module_1.module(var_6, var_2)
    var_17 = '/src'
    var_18 = 'src_module'
    var_19 = module_1.module(var_18, var_2)
    assert var_19 == 'FIRSTPARTY'
    var_20 = module_1.module(var_6, var_2)
    var_21 = 'unknown_module'
    var_22 = module_1.module(var_21)
    assert var_22 == 'STDLIB'
    var_23 = 'THIRDPARTY'
    var_24 = module_0.Config()
    var_25 = module_1.module(var_21, var_24)
    assert var_25 == 'THIRDPARTY'



# Parsed testcases at query #20
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'sys'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'STDLIB'
    var_5 = 'django'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'THIRDPARTY'
    var_7 = 'numpy'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'THIRDPARTY'
    var_9 = '.local_module'
    var_10 = module_1.module(var_9, var_0)
    assert var_10 == 'LOCALFOLDER'
    var_11 = 'local_module'
    var_12 = module_1.module(var_11, var_0)
    assert var_12 == 'FIRSTPARTY'
    var_13 = 'unknown_module'
    var_14 = module_1.module(var_13, var_0)
    assert var_14 == 'THIRDPARTY'
    var_15 = 'forced_module'
    var_16 = module_1.module(var_15, var_0)
    assert var_16 == 'forced_module'
    var_17 = 'forced_module.submodule'
    var_18 = module_1.module(var_17, var_0)
    assert var_18 == 'forced_module'
    var_19 = '^test_.*'
    var_20 = 'TESTS'
    var_21 = (var_19, var_20)
    var_22 = 'test_module'
    var_23 = module_1.module(var_22, var_0)
    assert var_23 == 'TESTS'
    var_24 = 'test_module.submodule'
    var_25 = module_1.module(var_24, var_0)
    assert var_25 == 'TESTS'
    var_26 = 'namespace_pkg'
    var_27 = module_1.module(var_26, var_0)
    assert var_27 == 'FIRSTPARTY'
    var_28 = 'namespace_pkg.submodule'
    var_29 = module_1.module(var_28, var_0)
    assert var_29 == 'FIRSTPARTY'
    var_30 = 'All tests passed!'
    var_31 = print(var_30)



