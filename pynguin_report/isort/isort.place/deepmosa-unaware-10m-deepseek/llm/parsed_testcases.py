####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_module'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'test_module'
    var_3 = 'test_module.submodule'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'test_module'
    var_5 = '.local_module'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'LOCALFOLDER'
    var_7 = '^django\\.*'
    var_8 = 'THIRDPARTY'
    var_9 = 'FIRSTPARTY'
    var_10 = 'django.test'
    var_11 = module_1.module(var_10, var_0)
    assert var_11 == 'THIRDPARTY'
    var_12 = 'unknown_module'
    var_13 = module_1.module(var_12, var_0)
    assert var_13 == 'STDLIB'
    var_14 = '/test/src'
    var_15 = 'py'
    var_16 = [var_15]
    var_17 = 'mymodule'
    var_18 = module_1.module(var_17, var_0)
    assert var_18 == 'FIRSTPARTY'
    var_19 = 'mynamespace'
    var_20 = False
    var_21 = True
    var_22 = 'mynamespace.submodule'
    var_23 = module_1.module(var_22, var_0)
    assert var_23 == 'FIRSTPARTY'
    var_24 = 'cached'
    var_25 = 'cached.module'
    var_26 = module_1.module(var_25, var_0)
    assert var_26 == 'cached'
    var_27 = module_1.module(var_25, var_0)
    assert var_27 == 'cached'



# Parsed testcases at query #2
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = set()
    var_4 = False
    var_5 = 'THIRDPARTY'
    var_6 = 'FIRSTPARTY'
    var_7 = 'LOCALFOLDER'
    var_8 = [var_6, var_7, var_5]
    var_9 = 'py'
    var_10 = [var_9]
    var_11 = frozenset(var_10)
    var_12 = module_0.Config()
    var_13 = 'os'
    var_14 = module_1.module(var_13, var_12)
    assert var_14 == 'THIRDPARTY'
    var_15 = 'test*'
    var_16 = 'test_module'
    var_17 = module_1.module(var_16, var_12)
    assert var_17 == 'test'
    var_18 = '.local'
    var_19 = module_1.module(var_18, var_12)
    assert var_19 == 'LOCALFOLDER'
    var_20 = '^django'
    var_21 = 'django.test'
    var_22 = module_1.module(var_21, var_12)
    assert var_22 == 'FIRSTPARTY'
    var_23 = '/fake/path'
    var_24 = 'my_module'
    var_25 = module_1.module(var_24, var_12)
    assert var_25 == 'FIRSTPARTY'
    var_26 = 'CUSTOM'
    var_27 = 'unknown'
    var_28 = module_1.module(var_27, var_12)
    assert var_28 == 'CUSTOM'



# Parsed testcases at query #3
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'collections'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'STDLIB'
    var_5 = 'pytest'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'THIRDPARTY'
    var_7 = 'test_module'
    var_8 = [var_7]
    var_9 = module_0.Config()
    var_10 = module_1.module(var_7, var_9)
    assert var_10 == 'test_module'
    var_11 = '.local_module'
    var_12 = module_1.module(var_11, var_9)
    assert var_12 == 'LOCALFOLDER'
    var_13 = 'local_module'
    var_14 = module_1.module(var_13, var_9)
    var_15 = '^django\\.'
    var_16 = 'DJANGO'
    var_17 = (var_15, var_16)
    var_18 = [var_17]
    var_19 = 'STDLIB'
    var_20 = 'THIRDPARTY'
    var_21 = 'FIRSTPARTY'
    var_22 = 'LOCALFOLDER'
    var_23 = [var_16, var_19, var_20, var_21, var_22]
    var_24 = module_0.Config()
    var_25 = 'django.test'
    var_26 = module_1.module(var_25, var_24)
    assert var_26 == 'DJANGO'
    var_27 = 'django.contrib.auth'
    var_28 = module_1.module(var_27, var_24)
    assert var_28 == 'DJANGO'
    var_29 = 'flask'
    var_30 = module_1.module(var_29, var_24)
    assert var_30 == 'THIRDPARTY'
    var_31 = 'CUSTOM'
    var_32 = module_0.Config()
    var_33 = 'unknown_module'
    var_34 = module_1.module(var_33, var_32)
    assert var_34 == 'CUSTOM'
    var_35 = '/test/src'
    var_36 = 'my_module'
    var_37 = module_1.module(var_36, var_32)
    assert var_37 == 'FIRSTPARTY'
    var_38 = 'my_namespace'
    var_39 = [var_38]
    var_40 = True
    var_41 = 'FIRSTPARTY'
    var_42 = 'reason'
    var_43 = 'my_namespace.submodule'
    var_44 = module_1.module(var_43, var_32)
    assert var_44 == 'FIRSTPARTY'
    var_45 = module_0.Config()
    var_46 = module_1.module(var_41, var_45)
    var_47 = module_1.module(var_41, var_45)
    var_48 = ''
    var_49 = module_1.module(var_48, var_45)
    assert var_49 == 'STDLIB'
    var_50 = len(var_44)
    assert var_50 == 2
    var_51 = var_44[var_40]



# Parsed testcases at query #4
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
    var_2 = 'sys'
    var_3 = module_0.module(var_2)
    assert var_3 == 'STDLIB'
    var_4 = 'collections'
    var_5 = module_0.module(var_4)
    assert var_5 == 'STDLIB'
    var_6 = 'django'
    var_7 = module_0.module(var_6)
    assert var_7 == 'THIRDPARTY'
    var_8 = 'numpy'
    var_9 = module_0.module(var_8)
    assert var_9 == 'THIRDPARTY'
    var_10 = '.'
    var_11 = module_0.module(var_10)
    assert var_11 == 'LOCALFOLDER'
    var_12 = '.local_module'
    var_13 = module_0.module(var_12)
    assert var_13 == 'LOCALFOLDER'
    var_14 = 'custom_lib'
    var_15 = [var_14]
    var_16 = 'my_project'
    var_17 = [var_16]
    var_18 = 'special'
    var_19 = [var_18]
    var_20 = module_1.Config()
    var_21 = module_0.module(var_14, var_20)
    assert var_21 == 'THIRDPARTY'
    var_22 = module_0.module(var_16, var_20)
    assert var_22 == 'FIRSTPARTY'
    var_23 = module_0.module(var_18, var_20)
    assert var_23 == 'special'
    var_24 = 'os.path'
    var_25 = module_0.module(var_24)
    assert var_25 == 'STDLIB'
    var_26 = 'django.contrib'
    var_27 = module_0.module(var_26)
    assert var_27 == 'THIRDPARTY'
    var_28 = 'test*'
    var_29 = [var_28]
    var_30 = module_1.Config()
    var_31 = 'test_module'
    var_32 = module_0.module(var_31, var_30)
    assert var_32 == 'test'
    var_33 = 'testing'
    var_34 = module_0.module(var_33, var_30)
    assert var_34 == 'test'
    var_35 = 'CUSTOM'
    var_36 = module_1.Config()
    var_37 = 'unknown_module'
    var_38 = module_0.module(var_37, var_36)
    assert var_38 == 'CUSTOM'
    var_39 = '.hidden'
    var_40 = module_0.module(var_39)
    assert var_40 == 'LOCALFOLDER'
    var_41 = '..parent'
    var_42 = module_0.module(var_41)
    assert var_42 == 'LOCALFOLDER'
    var_43 = 'regular'
    var_44 = module_0.module(var_43)
    var_45 = '^google\\.'
    var_46 = 'GOOGLE'
    var_47 = (var_45, var_46)
    var_48 = '^aws\\.'
    var_49 = 'AWS'
    var_50 = (var_48, var_49)
    var_51 = [var_47, var_50]
    var_52 = module_1.Config()
    var_53 = 'google.cloud'
    var_54 = module_0.module(var_53, var_52)
    assert var_54 == 'GOOGLE'
    var_55 = 'aws.s3'
    var_56 = module_0.module(var_55, var_52)
    assert var_56 == 'AWS'



# Parsed testcases at query #5
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'testlib'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'testlib'
    var_3 = 'testlib.submodule'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'testlib'
    var_5 = '.local'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'LOCALFOLDER'
    var_7 = '^django\\.'
    var_8 = 'THIRDPARTY'
    var_9 = 'FIRSTPARTY'
    var_10 = 'django.test'
    var_11 = module_1.module(var_10, var_0)
    assert var_11 == 'THIRDPARTY'
    var_12 = '/test/src'
    var_13 = 'mymodule'
    var_14 = module_1.module(var_13, var_0)
    assert var_14 == 'FIRSTPARTY'
    var_15 = 'mypackage'
    var_16 = module_1.module(var_15, var_0)
    assert var_16 == 'FIRSTPARTY'
    var_17 = 'src'
    var_18 = module_1.module(var_17, var_0)
    assert var_18 == 'FIRSTPARTY'
    var_19 = 'unknown'
    var_20 = module_1.module(var_19, var_0)
    var_21 = 'mynamespace'
    var_22 = 'mynamespace.sub'
    var_23 = module_1.module(var_22, var_0)
    assert var_23 == 'FIRSTPARTY'



# Parsed testcases at query #6
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_separate'
    var_2 = 'test_separate.module'
    var_3 = module_1.module(var_2, var_0)
    assert var_3 == 'test_separate'
    var_4 = module_1.module(var_1, var_0)
    assert var_4 == 'test_separate'
    var_5 = '.local_module'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'LOCALFOLDER'
    var_7 = '^django\\.*'
    var_8 = 'DJANGO'
    var_9 = 'FIRSTPARTY'
    var_10 = 'THIRDPARTY'
    var_11 = 'STDLIB'
    var_12 = 'django.app'
    var_13 = module_1.module(var_12, var_0)
    assert var_13 == 'DJANGO'
    var_14 = 'unknown_module'
    var_15 = module_1.module(var_14, var_0)
    assert var_15 == 'THIRDPARTY'
    var_16 = '/test/src'
    var_17 = 'test.namespace'
    var_18 = 'test.namespace.sub'
    var_19 = module_1.module(var_18, var_0)
    assert var_19 == 'FIRSTPARTY'
    var_20 = 'cached_module'
    var_21 = module_1.module(var_20, var_0)
    var_22 = module_1.module(var_20, var_0)



# Parsed testcases at query #7
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'collections'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'STDLIB'
    var_5 = 'typing'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'STDLIB'
    var_7 = 'requests'
    var_8 = 'pytest'
    var_9 = [var_7, var_8]
    var_10 = module_0.Config()
    var_11 = module_1.module(var_7, var_10)
    assert var_11 == 'THIRDPARTY'
    var_12 = module_1.module(var_8, var_10)
    assert var_12 == 'THIRDPARTY'
    var_13 = 'myapp'
    var_14 = 'mylib'
    var_15 = [var_13, var_14]
    var_16 = module_0.Config()
    var_17 = module_1.module(var_13, var_16)
    assert var_17 == 'FIRSTPARTY'
    var_18 = 'mylib.utils'
    var_19 = module_1.module(var_18, var_16)
    assert var_19 == 'FIRSTPARTY'
    var_20 = 'local_module'
    var_21 = [var_20]
    var_22 = module_0.Config()
    var_23 = module_1.module(var_20, var_22)
    assert var_23 == 'LOCALFOLDER'
    var_24 = 'THIRDPARTY'
    var_25 = module_0.Config()
    var_26 = 'unknown_module'
    var_27 = module_1.module(var_26, var_25)
    assert var_27 == 'THIRDPARTY'
    var_28 = 'special'
    var_29 = [var_28]
    var_30 = module_0.Config()
    var_31 = 'special.module'
    var_32 = module_1.module(var_31, var_30)
    assert var_32 == 'special'
    var_33 = '.local'
    var_34 = module_1.module(var_33, var_30)
    assert var_34 == 'LOCALFOLDER'
    var_35 = '..parent'
    var_36 = module_1.module(var_35, var_30)
    assert var_36 == 'LOCALFOLDER'
    var_37 = '^google\\.cloud\\.'
    var_38 = 'GOOGLE'
    var_39 = (var_37, var_38)
    var_40 = '^aws\\.'
    var_41 = 'AWS'
    var_42 = (var_40, var_41)
    var_43 = [var_39, var_42]
    var_44 = module_0.Config()
    var_45 = 'google.cloud.storage'
    var_46 = module_1.module(var_45, var_44)
    assert var_46 == 'GOOGLE'
    var_47 = 'aws.s3'
    var_48 = module_1.module(var_47, var_44)
    assert var_48 == 'AWS'
    var_49 = '/src'
    var_50 = 'mypackage'
    var_51 = module_1.module(var_50, var_44)
    assert var_51 == 'FIRSTPARTY'
    var_52 = 'mynamespace'
    var_53 = [var_52]
    var_54 = module_0.Config()
    var_55 = 'mynamespace.subpackage'
    var_56 = module_1.module(var_55, var_54)
    assert var_56 == 'FIRSTPARTY'
    var_57 = True
    var_58 = module_0.Config()
    var_59 = 'auto_namespace.sub'
    var_60 = module_1.module(var_59, var_58)
    assert var_60 == 'FIRSTPARTY'
    var_61 = 'django'
    var_62 = [var_61]
    var_63 = 'myproject'
    var_64 = [var_63]
    var_65 = module_0.Config()
    var_66 = module_1.module(var_61, var_65)
    assert var_66 == 'THIRDPARTY'
    var_67 = module_1.module(var_63, var_65)
    assert var_67 == 'FIRSTPARTY'
    var_68 = 'unknown'
    var_69 = module_1.module(var_68, var_65)
    var_70 = 'tests'
    var_71 = 'docs'
    var_72 = [var_70, var_71]
    var_73 = module_0.Config()
    var_74 = 'tests.unit'
    var_75 = module_1.module(var_74, var_73)
    assert var_75 == 'tests'
    var_76 = 'docs.source'
    var_77 = module_1.module(var_76, var_73)
    assert var_77 == 'docs'
    var_78 = 'regular.module'
    var_79 = module_1.module(var_78, var_73)
    var_80 = '^test_'
    var_81 = 'TEST'
    var_82 = (var_80, var_81)
    var_83 = [var_82]
    var_84 = module_0.Config()
    var_85 = 'test_module'
    var_86 = module_1.module(var_85, var_84)
    assert var_86 == 'TEST'
    var_87 = 'test_utils.helpers'
    var_88 = module_1.module(var_87, var_84)
    assert var_88 == 'TEST'
    var_89 = '/project/src'
    var_90 = 'src'
    var_91 = module_1.module(var_90, var_84)
    assert var_91 == 'FIRSTPARTY'
    var_92 = module_0.Config()
    var_93 = 'unknown'
    var_94 = module_1.module(var_93, var_92)



# Parsed testcases at query #8
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'testlib'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'testlib'
    var_3 = 'testlib.submodule'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'testlib'
    var_5 = '.local_module'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'LOCALFOLDER'
    var_7 = '^django\\.*'
    var_8 = 'DJANGO'
    var_9 = 'FIRSTPARTY'
    var_10 = 'THIRDPARTY'
    var_11 = 'STDLIB'
    var_12 = 'django.test'
    var_13 = module_1.module(var_12, var_0)
    assert var_13 == 'DJANGO'
    var_14 = 'unknown_module'
    var_15 = module_1.module(var_14, var_0)
    assert var_15 == 'THIRDPARTY'
    var_16 = module_0.Config()
    var_17 = 'some_module'
    var_18 = module_1.module(var_17, var_16)
    assert var_18 == 'STDLIB'



# Parsed testcases at query #9
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
    var_5 = 'collections'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'STDLIB'
    var_7 = 'test_module'
    var_8 = [var_7]
    var_9 = module_0.Config()
    var_10 = module_1.module(var_7, var_9)
    assert var_10 == 'test_module'
    var_11 = '.local_module'
    var_12 = module_1.module(var_11, var_9)
    assert var_12 == 'LOCALFOLDER'
    var_13 = '^django.*'
    var_14 = 'DJANGO'
    var_15 = (var_13, var_14)
    var_16 = [var_15]
    var_17 = module_0.Config()
    var_18 = 'django.test'
    var_19 = module_1.module(var_18, var_17)
    assert var_19 == 'DJANGO'
    var_20 = 'django.contrib.auth'
    var_21 = module_1.module(var_20, var_17)
    assert var_21 == 'DJANGO'
    var_22 = 'THIRDPARTY'
    var_23 = module_0.Config()
    var_24 = 'unknown_module'
    var_25 = module_1.module(var_24, var_23)
    assert var_25 == 'THIRDPARTY'
    var_26 = 'my_namespace'
    var_27 = {var_26}
    var_28 = module_0.Config()
    var_29 = 'my_namespace.module'
    var_30 = module_1.module(var_29, var_28)
    assert var_30 == 'FIRSTPARTY'
    var_31 = module_0.Config()
    var_32 = 'mymodule'
    var_33 = '__init__.py'
    var_34 = module_1.module(var_32, var_31)
    assert var_34 == 'FIRSTPARTY'
    var_35 = 'test*'
    var_36 = [var_35]
    var_37 = module_0.Config()
    var_38 = module_1.module(var_34, var_37)
    assert var_38 == 'test'
    var_39 = 'testing'
    var_40 = module_1.module(var_39, var_37)
    assert var_40 == 'test'
    var_41 = 'specific'
    var_42 = [var_41]
    var_43 = module_0.Config()
    var_44 = 'specific.module'
    var_45 = module_1.module(var_44, var_43)
    assert var_45 == 'specific'
    var_46 = 'forced'
    var_47 = [var_46]
    var_48 = '^forced.*'
    var_49 = 'KNOWN'
    var_50 = (var_48, var_49)
    var_51 = [var_50]
    var_52 = module_0.Config()
    var_53 = 'forced.module'
    var_54 = module_1.module(var_53, var_52)
    assert var_54 == 'forced'



# Parsed testcases at query #10
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
    var_2 = 'sys'
    var_3 = module_0.module(var_2)
    assert var_3 == 'STDLIB'
    var_4 = 'django'
    var_5 = [var_4]
    var_6 = module_1.Config()
    var_7 = module_0.module(var_4, var_6)
    assert var_7 == 'django'
    var_8 = 'django.contrib'
    var_9 = module_0.module(var_8, var_6)
    assert var_9 == 'django'
    var_10 = module_1.Config()
    var_11 = '.local_module'
    var_12 = module_0.module(var_11, var_10)
    assert var_12 == 'LOCALFOLDER'
    var_13 = '..parent_module'
    var_14 = module_0.module(var_13, var_10)
    assert var_14 == 'LOCALFOLDER'
    var_15 = '^google\\.cloud.*'
    var_16 = 'THIRDPARTY'
    var_17 = '^boto3.*'
    var_18 = 'STDLIB'
    var_19 = 'FIRSTPARTY'
    var_20 = 'LOCALFOLDER'
    var_21 = [var_18, var_16, var_19, var_20]
    var_22 = 'google.cloud.storage'
    var_23 = module_0.module(var_22, var_10)
    assert var_23 == 'THIRDPARTY'
    var_24 = 'boto3.s3'
    var_25 = module_0.module(var_24, var_10)
    assert var_25 == 'THIRDPARTY'
    var_26 = '/test/src'
    var_27 = 'mymodule'
    var_28 = module_0.module(var_27, var_10)
    assert var_28 == 'FIRSTPARTY'
    var_29 = 'mynamespace'
    var_30 = [var_29]
    var_31 = True
    var_32 = 'mynamespace.submodule'
    var_33 = module_0.module(var_32, var_10)
    assert var_33 == 'FIRSTPARTY'
    var_34 = module_1.Config()
    var_35 = 'unknown_module'
    var_36 = module_0.module(var_35, var_34)
    assert var_36 == 'THIRDPARTY'
    var_37 = module_1.Config()
    var_38 = 'test*'
    var_39 = [var_38]
    var_40 = module_1.Config()
    var_41 = 'test_module'
    var_42 = module_0.module(var_41, var_40)
    assert var_42 == 'test*'
    var_43 = 'testing'
    var_44 = module_0.module(var_43, var_40)
    assert var_44 == 'test*'
    var_45 = 'exact'
    var_46 = [var_45]
    var_47 = module_1.Config()
    var_48 = module_0.module(var_45, var_47)
    assert var_48 == 'exact'
    var_49 = 'exact.sub'
    var_50 = module_0.module(var_49, var_47)
    assert var_50 == 'exact'
    var_51 = [var_4]
    var_52 = module_1.Config()
    var_53 = 'flask'
    var_54 = module_0.module(var_53, var_52)



# Parsed testcases at query #11
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
    var_5 = 'collections'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'STDLIB'
    var_7 = 'pytest'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'THIRDPARTY'
    var_9 = 'numpy'
    var_10 = module_1.module(var_9, var_0)
    assert var_10 == 'THIRDPARTY'
    var_11 = '.local_module'
    var_12 = module_1.module(var_11, var_0)
    assert var_12 == 'LOCALFOLDER'
    var_13 = '.subpackage.module'
    var_14 = module_1.module(var_13, var_0)
    assert var_14 == 'LOCALFOLDER'
    var_15 = 'special'
    var_16 = [var_15]
    var_17 = module_0.Config()
    var_18 = module_1.module(var_15, var_17)
    assert var_18 == 'special'
    var_19 = 'special.module'
    var_20 = module_1.module(var_19, var_17)
    assert var_20 == 'special'
    var_21 = 'special*'
    var_22 = [var_21]
    var_23 = module_0.Config()
    var_24 = 'special_module'
    var_25 = module_1.module(var_24, var_23)
    assert var_25 == 'special_module'
    var_26 = '^django\\.'
    var_27 = 'DJANGO'
    var_28 = (var_26, var_27)
    var_29 = [var_28]
    var_30 = 'STDLIB'
    var_31 = 'THIRDPARTY'
    var_32 = [var_27, var_30, var_31]
    var_33 = module_0.Config()
    var_34 = 'django.apps'
    var_35 = module_1.module(var_34, var_33)
    assert var_35 == 'DJANGO'
    var_36 = 'django.contrib.auth'
    var_37 = module_1.module(var_36, var_33)
    assert var_37 == 'DJANGO'
    var_38 = 'mylib'
    var_39 = '__init__.py'
    var_40 = var_4 / var_39
    var_41 = module_0.Config()
    var_42 = module_1.module(var_38, var_41)
    assert var_42 == 'FIRSTPARTY'
    var_43 = 'mylib.submodule'
    var_44 = module_1.module(var_43, var_41)
    assert var_44 == 'FIRSTPARTY'
    var_45 = 'myns'
    var_46 = True
    var_47 = [var_45]
    var_48 = module_0.Config()
    var_49 = module_1.module(var_45, var_48)
    assert var_49 == 'FIRSTPARTY'
    var_50 = 'myns.subpackage'
    var_51 = module_1.module(var_50, var_48)
    assert var_51 == 'FIRSTPARTY'
    var_52 = 'CUSTOM'
    var_53 = module_0.Config()
    var_54 = 'unknown_module'
    var_55 = module_1.module(var_54, var_53)
    assert var_55 == 'CUSTOM'



# Parsed testcases at query #12
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
    var_2 = 'collections'
    var_3 = module_0.module(var_2)
    assert var_3 == 'STDLIB'
    var_4 = 'pytest'
    var_5 = module_0.module(var_4)
    assert var_5 == 'THIRDPARTY'
    var_6 = '.local_module'
    var_7 = module_0.module(var_6)
    assert var_7 == 'LOCALFOLDER'
    var_8 = '.subpackage.module'
    var_9 = module_0.module(var_8)
    assert var_9 == 'LOCALFOLDER'
    var_10 = 'test'
    var_11 = [var_10]
    var_12 = module_1.Config()
    var_13 = module_0.module(var_10, var_12)
    assert var_13 == 'test'
    var_14 = 'test.module'
    var_15 = module_0.module(var_14, var_12)
    assert var_15 == 'test'
    var_16 = '^django.*'
    var_17 = 'DJANGO'
    var_18 = (var_16, var_17)
    var_19 = [var_18]
    var_20 = module_1.Config()
    var_21 = 'django'
    var_22 = module_0.module(var_21, var_20)
    assert var_22 == 'DJANGO'
    var_23 = 'django.contrib'
    var_24 = module_0.module(var_23, var_20)
    assert var_24 == 'DJANGO'
    var_25 = '/fake/src'
    var_26 = 'isort.utils'
    var_27 = True
    var_28 = 'fakemodule'
    var_29 = module_0.module(var_28, var_20)
    assert var_29 == 'FIRSTPARTY'
    var_30 = 'CUSTOM'
    var_31 = module_1.Config()
    var_32 = 'unknown_module'
    var_33 = module_0.module(var_32, var_31)
    assert var_33 == 'CUSTOM'
    var_34 = 'mynamespace'
    var_35 = [var_34]
    var_36 = True
    var_37 = module_0.module(var_27)
    var_38 = module_0.module(var_27)
    var_39 = 'os.path'
    var_40 = module_0.module(var_39)
    assert var_40 == 'STDLIB'
    var_41 = 'collections.abc'
    var_42 = module_0.module(var_41)
    assert var_42 == 'STDLIB'



# Parsed testcases at query #13
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'collections'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'STDLIB'
    var_5 = 'test_module'
    var_6 = [var_5]
    var_7 = module_0.Config()
    var_8 = module_1.module(var_5, var_7)
    assert var_8 == 'test_module'
    var_9 = 'test_module.submodule'
    var_10 = module_1.module(var_9, var_7)
    assert var_10 == 'test_module'
    var_11 = module_0.Config()
    var_12 = '.local_module'
    var_13 = module_1.module(var_12, var_11)
    assert var_13 == 'LOCALFOLDER'
    var_14 = '..parent_module'
    var_15 = module_1.module(var_14, var_11)
    assert var_15 == 'LOCALFOLDER'
    var_16 = '^django\\.'
    var_17 = 'DJANGO'
    var_18 = (var_16, var_17)
    var_19 = '^requests$'
    var_20 = 'THIRDPARTY'
    var_21 = (var_19, var_20)
    var_22 = [var_18, var_21]
    var_23 = 'STDLIB'
    var_24 = 'FIRSTPARTY'
    var_25 = [var_23, var_20, var_24, var_17]
    var_26 = module_0.Config()
    var_27 = 'django.apps'
    var_28 = module_1.module(var_27, var_26)
    assert var_28 == 'DJANGO'
    var_29 = 'django'
    var_30 = module_1.module(var_29, var_26)
    assert var_30 == 'DJANGO'
    var_31 = 'requests'
    var_32 = module_1.module(var_31, var_26)
    assert var_32 == 'THIRDPARTY'
    var_33 = module_0.Config()
    var_34 = 'mymodule'
    var_35 = '__init__.py'
    var_36 = module_1.module(var_34, var_33)
    assert var_36 == 'FIRSTPARTY'
    var_37 = 'mymodule.submodule'
    var_38 = module_1.module(var_37, var_33)
    assert var_38 == 'FIRSTPARTY'
    var_39 = True
    var_40 = module_0.Config()
    var_41 = 'mynamespace'
    var_42 = module_1.module(var_41, var_40)
    assert var_42 == 'FIRSTPARTY'
    var_43 = 'mynamespace.subpackage'
    var_44 = module_1.module(var_43, var_40)
    assert var_44 == 'FIRSTPARTY'
    var_45 = 'CUSTOM'
    var_46 = module_0.Config()
    var_47 = 'unknown_module'
    var_48 = module_1.module(var_47, var_46)
    assert var_48 == 'CUSTOM'
    var_49 = module_0.Config()
    var_50 = module_1.module(var_1, var_49)
    var_51 = module_1.module(var_1, var_49)



# Parsed testcases at query #14
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
    var_2 = 'collections'
    var_3 = module_0.module(var_2)
    assert var_3 == 'STDLIB'
    var_4 = 'pytest'
    var_5 = module_0.module(var_4)
    assert var_5 == 'THIRDPARTY'
    var_6 = 'django'
    var_7 = module_0.module(var_6)
    assert var_7 == 'THIRDPARTY'
    var_8 = '.local_module'
    var_9 = module_0.module(var_8)
    assert var_9 == 'LOCALFOLDER'
    var_10 = '.subpackage.module'
    var_11 = module_0.module(var_10)
    assert var_11 == 'LOCALFOLDER'
    var_12 = 'custom_lib'
    var_13 = [var_12]
    var_14 = 'my_project'
    var_15 = [var_14]
    var_16 = 'FUTURE'
    var_17 = 'STDLIB'
    var_18 = 'THIRDPARTY'
    var_19 = 'FIRSTPARTY'
    var_20 = 'LOCALFOLDER'
    var_21 = [var_16, var_17, var_18, var_19, var_20]
    var_22 = module_1.Config()
    var_23 = module_0.module(var_12, var_22)
    assert var_23 == 'THIRDPARTY'
    var_24 = module_0.module(var_14, var_22)
    assert var_24 == 'FIRSTPARTY'
    var_25 = 'special'
    var_26 = [var_25]
    var_27 = module_1.Config()
    var_28 = 'special.module'
    var_29 = module_0.module(var_28, var_27)
    assert var_29 == 'special'
    var_30 = 'CUSTOM'
    var_31 = module_1.Config()
    var_32 = 'unknown_module'
    var_33 = module_0.module(var_32, var_31)
    assert var_33 == 'CUSTOM'
    var_34 = '^google\\.cloud\\..*'
    var_35 = 'GOOGLE'
    var_36 = '^aws\\..*'
    var_37 = 'AWS'
    var_38 = [var_16, var_17, var_18, var_19, var_20, var_35, var_37]
    var_39 = 'google.cloud.storage'
    var_40 = 'aws.s3'
    var_41 = 'other.cloud.storage'



# Parsed testcases at query #15
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'test_module'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'test_module'
    var_5 = '.local_module'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'LOCALFOLDER'
    var_7 = '^django.*'
    var_8 = 'THIRDPARTY'
    var_9 = 'STDLIB'
    var_10 = 'FIRSTPARTY'
    var_11 = 'django.test'
    var_12 = module_1.module(var_11, var_0)
    assert var_12 == 'THIRDPARTY'
    var_13 = '/test/src'
    var_14 = 'py'
    var_15 = [var_14]
    var_16 = 'mynamespace'
    var_17 = 'some_module'
    var_18 = module_1.module(var_17, var_0)
    var_19 = module_1.module(var_17, var_0)
    var_20 = module_0.Config()
    var_21 = 'unknown_module'
    var_22 = module_1.module(var_21, var_20)



# Parsed testcases at query #16
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
    var_2 = 'sys'
    var_3 = module_0.module(var_2)
    assert var_3 == 'STDLIB'
    var_4 = 'collections'
    var_5 = module_0.module(var_4)
    assert var_5 == 'STDLIB'
    var_6 = 'pytest'
    var_7 = module_0.module(var_6)
    assert var_7 == 'THIRDPARTY'
    var_8 = 'numpy'
    var_9 = module_0.module(var_8)
    assert var_9 == 'THIRDPARTY'
    var_10 = '.local_module'
    var_11 = module_0.module(var_10)
    assert var_11 == 'LOCALFOLDER'
    var_12 = '.subpackage.module'
    var_13 = module_0.module(var_12)
    assert var_13 == 'LOCALFOLDER'
    var_14 = 'test_module'
    var_15 = [var_14]
    var_16 = module_1.Config()
    var_17 = module_0.module(var_14, var_16)
    assert var_17 == 'test_module'
    var_18 = 'test_module.sub'
    var_19 = module_0.module(var_18, var_16)
    assert var_19 == 'test_module'
    var_20 = '^myapp\\.*'
    var_21 = 'MYAPP'
    var_22 = 'STDLIB'
    var_23 = 'THIRDPARTY'
    var_24 = 'FIRSTPARTY'
    var_25 = [var_22, var_23, var_21, var_24]
    var_26 = 'myapp'
    var_27 = module_0.module(var_26, var_16)
    assert var_27 == 'MYAPP'
    var_28 = 'myapp.utils'
    var_29 = module_0.module(var_28, var_16)
    assert var_29 == 'MYAPP'
    var_30 = 'CUSTOM'
    var_31 = module_1.Config()
    var_32 = 'unknown_module'
    var_33 = module_0.module(var_32, var_31)
    assert var_33 == 'CUSTOM'
    var_34 = '/project/src'
    var_35 = 'FIRSTPARTY'
    var_36 = 'Found in one of the configured src_paths: /project/src'
    var_37 = 'mymodule'
    var_38 = module_0.module(var_37, var_31)
    assert var_38 == 'FIRSTPARTY'
    var_39 = 'mynamespace'
    var_40 = [var_39]
    var_41 = True
    var_42 = 'FIRSTPARTY'
    var_43 = 'Found in one of the configured src_paths: /project/src'
    var_44 = 'mynamespace.sub'
    var_45 = module_0.module(var_44, var_31)
    assert var_45 == 'FIRSTPARTY'
    var_46 = module_1.Config()
    var_47 = module_0.module(var_42, var_46)
    var_48 = module_0.module(var_42, var_46)
    var_49 = 'collections.abc'
    var_50 = module_0.module(var_49)
    assert var_50 == 'STDLIB'
    var_51 = 'django.contrib.auth'
    var_52 = module_0.module(var_51)
    assert var_52 == 'THIRDPARTY'
    var_53 = 'special'
    var_54 = [var_53]
    var_55 = '^special\\.*'
    var_56 = 'SPECIAL'
    var_57 = [var_22, var_23, var_56, var_24]
    var_58 = module_0.module(var_53, var_46)
    assert var_58 == 'special'
    var_59 = 'special.module'
    var_60 = module_0.module(var_59, var_46)
    assert var_60 == 'special'



# Parsed testcases at query #17
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_separate'
    var_2 = 'test_separate.module'
    var_3 = module_1.module(var_2, var_0)
    assert var_3 == 'test_separate'
    var_4 = '.test_separate.module'
    var_5 = module_1.module(var_4, var_0)
    assert var_5 == 'test_separate'
    var_6 = '.local_module'
    var_7 = module_1.module(var_6, var_0)
    assert var_7 == 'LOCALFOLDER'
    var_8 = '^django'
    var_9 = 'THIRDPARTY'
    var_10 = 'FIRSTPARTY'
    var_11 = 'django.apps'
    var_12 = module_1.module(var_11, var_0)
    assert var_12 == 'THIRDPARTY'
    var_13 = 'some_module'
    var_14 = module_1.module(var_13, var_0)
    var_15 = '/test/src'
    var_16 = 'test.namespace'
    var_17 = 'test.namespace.module'
    var_18 = module_1.module(var_17, var_0)
    assert var_18 == 'FIRSTPARTY'
    var_19 = 'test.auto_namespace.module'
    var_20 = module_1.module(var_19, var_0)
    assert var_20 == 'FIRSTPARTY'
    var_21 = 'cached_module'
    var_22 = module_1.module(var_21, var_0)
    var_23 = module_1.module(var_21, var_0)
    var_24 = 'exact_match'
    var_25 = module_1.module(var_24, var_0)
    assert var_25 == 'exact_match'
    var_26 = 'exact_match.submodule'
    var_27 = module_1.module(var_26, var_0)
    assert var_27 == 'exact_match'
    var_28 = '/src/project'
    var_29 = 'project'
    var_30 = module_1.module(var_29, var_0)
    assert var_30 == 'FIRSTPARTY'



# Parsed testcases at query #18
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_module'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'test_module'
    var_3 = 'test_module.submodule'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'test_module'
    var_5 = '.local_module'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'LOCALFOLDER'
    var_7 = '^django\\.*'
    var_8 = 'THIRDPARTY'
    var_9 = 'FIRSTPARTY'
    var_10 = 'django.test'
    var_11 = module_1.module(var_10, var_0)
    assert var_11 == 'THIRDPARTY'
    var_12 = 'unknown_module'
    var_13 = module_1.module(var_12, var_0)
    assert var_13 == 'STDLIB'
    var_14 = module_0.Config()
    var_15 = 'some_module'
    var_16 = module_1.module(var_15, var_14)
    assert var_16 == 'FIRSTPARTY'



# Parsed testcases at query #19
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'collections'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'STDLIB'
    var_5 = 'typing'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'STDLIB'
    var_7 = 'pytest'
    var_8 = 'requests'
    var_9 = [var_7, var_8]
    var_10 = module_0.Config()
    var_11 = module_1.module(var_7, var_10)
    assert var_11 == 'THIRDPARTY'
    var_12 = module_1.module(var_8, var_10)
    assert var_12 == 'THIRDPARTY'
    var_13 = 'myapp'
    var_14 = 'mylib'
    var_15 = [var_13, var_14]
    var_16 = module_0.Config()
    var_17 = module_1.module(var_13, var_16)
    assert var_17 == 'FIRSTPARTY'
    var_18 = 'mylib.utils'
    var_19 = module_1.module(var_18, var_16)
    assert var_19 == 'FIRSTPARTY'
    var_20 = 'special'
    var_21 = [var_20]
    var_22 = module_0.Config()
    var_23 = 'special.module'
    var_24 = module_1.module(var_23, var_22)
    assert var_24 == 'special'
    var_25 = '.local_module'
    var_26 = module_1.module(var_25, var_22)
    assert var_26 == 'LOCALFOLDER'
    var_27 = '.subpackage.module'
    var_28 = module_1.module(var_27, var_22)
    assert var_28 == 'LOCALFOLDER'
    var_29 = 'THIRDPARTY'
    var_30 = module_0.Config()
    var_31 = 'unknown_module'
    var_32 = module_1.module(var_31, var_30)
    assert var_32 == 'THIRDPARTY'
    var_33 = '^google\\.cloud\\..*'
    var_34 = 'GOOGLE'
    var_35 = (var_33, var_34)
    var_36 = '^aws\\.'
    var_37 = 'AWS'
    var_38 = (var_36, var_37)
    var_39 = [var_35, var_38]
    var_40 = 'FIRSTPARTY'
    var_41 = 'STDLIB'
    var_42 = [var_34, var_37, var_40, var_29, var_41]
    var_43 = module_0.Config()
    var_44 = 'google.cloud.storage'
    var_45 = module_1.module(var_44, var_43)
    assert var_45 == 'GOOGLE'
    var_46 = 'aws.s3'
    var_47 = module_1.module(var_46, var_43)
    assert var_47 == 'AWS'
    var_48 = '/project/src'
    var_49 = 'mymodule'
    var_50 = module_1.module(var_49, var_43)
    assert var_50 == 'FIRSTPARTY'
    var_51 = 'mynamespace'
    var_52 = [var_51]
    var_53 = module_0.Config()
    var_54 = 'mynamespace.subpackage'
    var_55 = module_1.module(var_54, var_53)
    assert var_55 == 'FIRSTPARTY'
    var_56 = True
    var_57 = module_0.Config()
    var_58 = 'auto_namespace.sub'
    var_59 = module_1.module(var_58, var_57)
    assert var_59 == 'FIRSTPARTY'
    var_60 = 'pandas'
    var_61 = [var_60]
    var_62 = 'numpy'
    var_63 = [var_62]
    var_64 = module_0.Config()
    var_65 = module_1.module(var_60, var_64)
    assert var_65 == 'THIRDPARTY'
    var_66 = module_1.module(var_62, var_64)
    assert var_66 == 'FIRSTPARTY'
    var_67 = 'tests'
    var_68 = 'docs'
    var_69 = [var_67, var_68]
    var_70 = module_0.Config()
    var_71 = 'tests.unit'
    var_72 = module_1.module(var_71, var_70)
    assert var_72 == 'tests'
    var_73 = 'docs.source'
    var_74 = module_1.module(var_73, var_70)
    assert var_74 == 'docs'
    var_75 = module_0.Config()
    var_76 = 'sys'
    var_77 = module_1.module(var_76, var_75)
    assert var_77 == 'STDLIB'
    var_78 = 'itertools'
    var_79 = module_1.module(var_78, var_75)
    assert var_79 == 'STDLIB'
    var_80 = []
    var_81 = module_0.Config()
    var_82 = 'external_lib'
    var_83 = module_1.module(var_82, var_81)
    assert var_83 == 'THIRDPARTY'
    var_84 = []
    var_85 = module_0.Config()
    var_86 = 'internal'
    var_87 = module_1.module(var_86, var_85)
    assert var_87 == 'FIRSTPARTY'
    var_88 = []
    var_89 = module_0.Config()
    var_90 = 'any.module'
    var_91 = module_1.module(var_90, var_89)
    assert var_91 == 'THIRDPARTY'
    var_92 = module_0.Config()
    var_93 = 'unknown'
    var_94 = module_1.module(var_93, var_92)
    assert var_94 == 'STDLIB'
    var_95 = []
    var_96 = module_0.Config()
    var_97 = 'pattern.match'
    var_98 = module_1.module(var_97, var_96)
    assert var_98 == 'THIRDPARTY'
    var_99 = []
    var_100 = module_0.Config()
    var_101 = 'local'
    var_102 = module_1.module(var_101, var_100)
    assert var_102 == 'THIRDPARTY'
    var_103 = []
    var_104 = module_0.Config()
    var_105 = 'ns.pkg'
    var_106 = module_1.module(var_105, var_104)
    assert var_106 == 'THIRDPARTY'
    var_107 = False
    var_108 = module_0.Config()
    var_109 = 'auto.ns'
    var_110 = module_1.module(var_109, var_108)
    assert var_110 == 'THIRDPARTY'



# Parsed testcases at query #20
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_separate'
    var_2 = 'test_separate.module'
    var_3 = module_1.module(var_2, var_0)
    assert var_3 == 'test_separate'
    var_4 = module_1.module(var_1, var_0)
    assert var_4 == 'test_separate'
    var_5 = '.local_module'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'LOCALFOLDER'
    var_7 = '^django\\.'
    var_8 = 'DJANGO'
    var_9 = 'FIRSTPARTY'
    var_10 = 'THIRDPARTY'
    var_11 = 'django.app'
    var_12 = module_1.module(var_11, var_0)
    assert var_12 == 'DJANGO'
    var_13 = '/test/src'
    var_14 = 'mymodule'
    var_15 = 'mymodule'
    var_16 = module_1.module(var_15, var_0)
    assert var_16 == 'FIRSTPARTY'
    var_17 = 'some_unknown_module'
    var_18 = module_1.module(var_17, var_0)
    var_19 = 'mynamespace'
    var_20 = False
    var_21 = 'mynamespace.submodule'
    var_22 = module_1.module(var_21, var_0)
    assert var_22 == 'FIRSTPARTY'
    var_23 = True
    var_24 = 'src'
    var_25 = module_1.module(var_24, var_0)
    assert var_25 == 'FIRSTPARTY'



# Parsed testcases at query #21
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
    var_2 = 'collections'
    var_3 = module_0.module(var_2)
    assert var_3 == 'STDLIB'
    var_4 = 'pytest'
    var_5 = module_0.module(var_4)
    assert var_5 == 'THIRDPARTY'
    var_6 = '.local_module'
    var_7 = module_0.module(var_6)
    assert var_7 == 'LOCALFOLDER'
    var_8 = '.subpackage.module'
    var_9 = module_0.module(var_8)
    assert var_9 == 'LOCALFOLDER'
    var_10 = 'test'
    var_11 = [var_10]
    var_12 = module_1.Config()
    var_13 = module_0.module(var_10, var_12)
    assert var_13 == 'test'
    var_14 = 'test.module'
    var_15 = module_0.module(var_14, var_12)
    assert var_15 == 'test'
    var_16 = '^django.*'
    var_17 = 'DJANGO'
    var_18 = (var_16, var_17)
    var_19 = [var_18]
    var_20 = [var_17]
    var_21 = module_1.Config()
    var_22 = 'django'
    var_23 = module_0.module(var_22, var_21)
    assert var_23 == 'DJANGO'
    var_24 = 'django.contrib'
    var_25 = module_0.module(var_24, var_21)
    assert var_25 == 'DJANGO'
    var_26 = '/test/src'
    var_27 = 'CUSTOM'
    var_28 = module_1.Config()
    var_29 = 'unknown_module'
    var_30 = module_0.module(var_29, var_28)
    assert var_30 == 'CUSTOM'
    var_31 = module_1.Config()
    var_32 = module_0.module(var_0, var_31)
    var_33 = module_0.module(var_0, var_31)



# Parsed testcases at query #22
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'collections'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'STDLIB'
    var_5 = 'typing'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'STDLIB'
    var_7 = 'pytest'
    var_8 = 'requests'
    var_9 = [var_7, var_8]
    var_10 = module_0.Config()
    var_11 = module_1.module(var_7, var_10)
    assert var_11 == 'THIRDPARTY'
    var_12 = module_1.module(var_8, var_10)
    assert var_12 == 'THIRDPARTY'
    var_13 = 'myapp'
    var_14 = 'mylib'
    var_15 = [var_13, var_14]
    var_16 = module_0.Config()
    var_17 = module_1.module(var_13, var_16)
    assert var_17 == 'FIRSTPARTY'
    var_18 = 'mylib.utils'
    var_19 = module_1.module(var_18, var_16)
    assert var_19 == 'FIRSTPARTY'
    var_20 = 'test'
    var_21 = [var_20]
    var_22 = module_0.Config()
    var_23 = 'test.module'
    var_24 = module_1.module(var_23, var_22)
    assert var_24 == 'test'
    var_25 = '.local_module'
    var_26 = module_1.module(var_25, var_22)
    assert var_26 == 'LOCALFOLDER'
    var_27 = '.subpackage.module'
    var_28 = module_1.module(var_27, var_22)
    assert var_28 == 'LOCALFOLDER'
    var_29 = 'THIRDPARTY'
    var_30 = module_0.Config()
    var_31 = 'unknown_lib'
    var_32 = module_1.module(var_31, var_30)
    assert var_32 == 'THIRDPARTY'
    var_33 = '^google\\.cloud\\..*'
    var_34 = 'GOOGLE'
    var_35 = (var_33, var_34)
    var_36 = '^aws\\.'
    var_37 = 'AWS'
    var_38 = (var_36, var_37)
    var_39 = [var_35, var_38]
    var_40 = module_0.Config()
    var_41 = 'google.cloud.storage'
    var_42 = module_1.module(var_41, var_40)
    assert var_42 == 'GOOGLE'
    var_43 = 'aws.s3'
    var_44 = module_1.module(var_43, var_40)
    assert var_44 == 'AWS'
    var_45 = '/project/src'
    var_46 = ''
    var_47 = 'mymodule'
    var_48 = module_1.module(var_47, var_40)
    assert var_48 == 'FIRSTPARTY'
    var_49 = '/project/src/mymodule.py'
    var_50 = 'mynamespace'
    var_51 = [var_50]
    var_52 = module_0.Config()
    var_53 = 'mynamespace.subpackage'
    var_54 = module_1.module(var_53, var_52)
    assert var_54 == 'FIRSTPARTY'
    var_55 = True
    var_56 = module_0.Config()
    var_57 = '/tmp/test_namespace'
    var_58 = 'subpkg'
    var_59 = 'test_namespace.subpkg'
    var_60 = module_1.module(var_59, var_56)
    assert var_60 == 'FIRSTPARTY'
    var_61 = 'sys'
    var_62 = module_0.Config()
    var_63 = module_1.module(var_61, var_62)
    assert var_63 == 'STDLIB'
    var_64 = 'os.path'
    var_65 = module_0.Config()
    var_66 = module_1.module(var_64, var_65)
    assert var_66 == 'STDLIB'



# Parsed testcases at query #23
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'some_module'
    var_1 = module_0.module(var_0)
    assert var_1 == 'FIRSTPARTY'
    var_2 = 'test'
    var_3 = [var_2]
    var_4 = module_1.Config()
    var_5 = 'test.module'
    var_6 = module_0.module(var_5, var_4)
    assert var_6 == 'test'
    var_7 = '.local_module'
    var_8 = module_0.module(var_7)
    assert var_8 == 'LOCALFOLDER'
    var_9 = '^django.*'
    var_10 = 'THIRDPARTY'
    var_11 = 'django.app'
    var_12 = module_0.module(var_11, var_4)
    assert var_12 == 'THIRDPARTY'
    var_13 = '^django\\.contrib'
    var_14 = 'DJANGO'
    var_15 = 'django.contrib.admin'
    var_16 = module_0.module(var_15, var_4)
    assert var_16 == 'DJANGO'
    var_17 = '/test/src'
    var_18 = 'mymodule'
    var_19 = module_0.module(var_18, var_4)
    assert var_19 == 'FIRSTPARTY'
    var_20 = 'mynamespace'
    var_21 = [var_20]
    var_22 = True
    var_23 = 'mynamespace.subpackage'
    var_24 = module_0.module(var_23, var_4)
    assert var_24 == 'FIRSTPARTY'
    var_25 = 'module'
    var_26 = module_0.module(var_25, var_4)
    assert var_26 == 'FIRSTPARTY'
    var_27 = 'CUSTOM'
    var_28 = module_1.Config()
    var_29 = 'unknown'
    var_30 = module_0.module(var_29, var_28)
    var_31 = module_0.module(var_29, var_28)
    var_32 = 'special'
    var_33 = [var_32]
    var_34 = module_1.Config()
    var_35 = '.special.module'
    var_36 = module_0.module(var_35, var_34)
    assert var_36 == 'special'
    var_37 = 'exactmatch'
    var_38 = [var_37]
    var_39 = module_1.Config()
    var_40 = module_0.module(var_37, var_39)
    assert var_40 == 'exactmatch'



# Parsed testcases at query #24
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'collections'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'STDLIB'
    var_5 = 'test_module'
    var_6 = [var_5]
    var_7 = module_0.Config()
    var_8 = module_1.module(var_5, var_7)
    assert var_8 == 'test_module'
    var_9 = module_0.Config()
    var_10 = '.local_module'
    var_11 = module_1.module(var_10, var_9)
    assert var_11 == 'LOCALFOLDER'
    var_12 = '^django\\.'
    var_13 = 'THIRDPARTY'
    var_14 = (var_12, var_13)
    var_15 = [var_14]
    var_16 = module_0.Config()
    var_17 = 'django.apps'
    var_18 = module_1.module(var_17, var_16)
    assert var_18 == 'THIRDPARTY'
    var_19 = module_0.Config()
    var_20 = 'mymodule'
    var_21 = '__init__.py'
    var_22 = module_1.module(var_20, var_19)
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'mynamespace'
    var_24 = [var_23]
    var_25 = True
    var_26 = module_0.Config()
    var_27 = 'submodule'
    var_28 = '__init__.py'
    var_29 = 'mynamespace.submodule'
    var_30 = module_1.module(var_29, var_26)
    assert var_30 == 'FIRSTPARTY'
    var_31 = 'CUSTOM'
    var_32 = module_0.Config()
    var_33 = 'unknown_module'
    var_34 = module_1.module(var_33, var_32)
    assert var_34 == 'CUSTOM'



# Parsed testcases at query #25
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'some_module'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'test_module'
    var_4 = [var_3]
    var_5 = module_0.Config()
    var_6 = module_1.module(var_3, var_5)
    assert var_6 == 'test_module'
    var_7 = 'test_*'
    var_8 = [var_7]
    var_9 = module_0.Config()
    var_10 = 'test_utils'
    var_11 = module_1.module(var_10, var_9)
    assert var_11 == 'test_*'
    var_12 = module_0.Config()
    var_13 = '.local_module'
    var_14 = module_1.module(var_13, var_12)
    assert var_14 == 'LOCALFOLDER'
    var_15 = '^django\\.'
    var_16 = 'THIRDPARTY'
    var_17 = (var_15, var_16)
    var_18 = [var_17]
    var_19 = 'STDLIB'
    var_20 = [var_19, var_16]
    var_21 = module_0.Config()
    var_22 = 'django.contrib'
    var_23 = module_1.module(var_22, var_21)
    assert var_23 == 'THIRDPARTY'
    var_24 = (var_15, var_16)
    var_25 = [var_24]
    var_26 = [var_19, var_16]
    var_27 = module_0.Config()
    var_28 = 'django.contrib.auth'
    var_29 = module_1.module(var_28, var_27)
    assert var_29 == 'THIRDPARTY'
    var_30 = 'CUSTOM'
    var_31 = module_0.Config()
    var_32 = 'unknown_module'
    var_33 = module_1.module(var_32, var_31)
    assert var_33 == 'CUSTOM'
    var_34 = 'special'
    var_35 = [var_34]
    var_36 = '^special\\.'
    var_37 = (var_36, var_16)
    var_38 = [var_37]
    var_39 = [var_19, var_16]
    var_40 = module_0.Config()
    var_41 = 'special.module'
    var_42 = module_1.module(var_41, var_40)
    assert var_42 == 'special'
    var_43 = 'test'
    var_44 = [var_43]
    var_45 = module_0.Config()
    var_46 = '.test'
    var_47 = module_1.module(var_46, var_45)
    assert var_47 == 'test'
    var_48 = module_0.Config()
    var_49 = module_1.module(var_1, var_48)
    var_50 = module_1.module(var_1, var_48)



# Parsed testcases at query #26
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'collections'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'STDLIB'
    var_5 = 'test_module'
    var_6 = [var_5]
    var_7 = module_0.Config()
    var_8 = module_1.module(var_5, var_7)
    assert var_8 == 'test_module'
    var_9 = module_0.Config()
    var_10 = '.local_module'
    var_11 = module_1.module(var_10, var_9)
    assert var_11 == 'LOCALFOLDER'
    var_12 = '^django.*'
    var_13 = 'THIRDPARTY'
    var_14 = (var_12, var_13)
    var_15 = [var_14]
    var_16 = module_0.Config()
    var_17 = 'django.contrib.auth'
    var_18 = module_1.module(var_17, var_16)
    assert var_18 == 'THIRDPARTY'
    var_19 = 'CUSTOM'
    var_20 = module_0.Config()
    var_21 = 'unknown_module'
    var_22 = module_1.module(var_21, var_20)
    assert var_22 == 'CUSTOM'
    var_23 = 'my_namespace'
    var_24 = [var_23]
    var_25 = '/fake/path'
    var_26 = 'my_namespace.subpackage'
    var_27 = module_1.module(var_26, var_20)
    assert var_27 == 'FIRSTPARTY'
    var_28 = True
    var_29 = '/fake/src'
    var_30 = module_0.Config()
    var_31 = module_1.module(var_1, var_30)
    var_32 = module_1.module(var_1, var_30)



# Parsed testcases at query #27
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'collections'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'STDLIB'
    var_5 = 'django'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'THIRDPARTY'
    var_7 = 'requests'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'THIRDPARTY'
    var_9 = '.local_module'
    var_10 = module_1.module(var_9, var_0)
    assert var_10 == 'LOCALFOLDER'
    var_11 = '..parent_module'
    var_12 = module_1.module(var_11, var_0)
    assert var_12 == 'LOCALFOLDER'
    var_13 = 'special'
    var_14 = 'special.module'
    var_15 = module_1.module(var_14, var_0)
    assert var_15 == 'special'
    var_16 = module_1.module(var_13, var_0)
    assert var_16 == 'special'
    var_17 = '^custom\\.'
    var_18 = 'CUSTOM'
    var_19 = (var_17, var_18)
    var_20 = 'custom.package'
    var_21 = module_1.module(var_20, var_0)
    assert var_21 == 'CUSTOM'
    var_22 = '/test/src'
    var_23 = 'FIRSTPARTY'
    var_24 = 'reason'
    var_25 = 'mymodule'
    var_26 = module_1.module(var_25, var_0)
    assert var_26 == 'FIRSTPARTY'
    var_27 = 'unknown'
    var_28 = module_1.module(var_27, var_0)
    assert var_28 == 'DEFAULT'



# Parsed testcases at query #28
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = set()
    var_4 = False
    var_5 = 'THIRDPARTY'
    var_6 = 'FIRSTPARTY'
    var_7 = 'LOCALFOLDER'
    var_8 = [var_6, var_7, var_5]
    var_9 = 'py'
    var_10 = [var_9]
    var_11 = frozenset(var_10)
    var_12 = module_0.Config()
    var_13 = 'os'
    var_14 = module_1.module(var_13, var_12)
    assert var_14 == 'THIRDPARTY'
    var_15 = 'test'
    var_16 = 'test.module'
    var_17 = module_1.module(var_16, var_12)
    assert var_17 == 'test'
    var_18 = '.local'
    var_19 = module_1.module(var_18, var_12)
    assert var_19 == 'LOCALFOLDER'
    var_20 = '^django'
    var_21 = 'DJANGO'
    var_22 = 'django.apps'
    var_23 = module_1.module(var_22, var_12)
    assert var_23 == 'DJANGO'
    var_24 = '/src'
    var_25 = 'myapp'
    var_26 = module_1.module(var_25, var_12)
    assert var_26 == 'FIRSTPARTY'
    var_27 = 'STDLIB'
    var_28 = 'sys'
    var_29 = module_1.module(var_28, var_12)
    assert var_29 == 'STDLIB'



# Parsed testcases at query #29
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'some_module'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'test_module'
    var_4 = [var_3]
    var_5 = module_0.Config()
    var_6 = module_1.module(var_3, var_5)
    assert var_6 == 'test_module'
    var_7 = module_0.Config()
    var_8 = '.local_module'
    var_9 = module_1.module(var_8, var_7)
    assert var_9 == 'LOCALFOLDER'
    var_10 = '^django\\.'
    var_11 = 'THIRDPARTY'
    var_12 = (var_10, var_11)
    var_13 = [var_12]
    var_14 = module_0.Config()
    var_15 = 'django.apps'
    var_16 = module_1.module(var_15, var_14)
    assert var_16 == 'THIRDPARTY'
    var_17 = 'src'
    var_18 = 'my_package'
    var_19 = module_1.module(var_18, var_14)
    assert var_19 == 'FIRSTPARTY'
    var_20 = 'my_namespace'
    var_21 = [var_20]
    var_22 = True
    var_23 = 'my_namespace.subpackage'
    var_24 = module_1.module(var_23, var_14)
    assert var_24 == 'FIRSTPARTY'
    var_25 = 'project'
    var_26 = 'project.module'
    var_27 = module_1.module(var_26, var_14)
    assert var_27 == 'FIRSTPARTY'
    var_28 = 'CUSTOM'
    var_29 = module_0.Config()
    var_30 = 'unknown_module'
    var_31 = module_1.module(var_30, var_29)
    assert var_31 == 'CUSTOM'
    var_32 = 'test_*'
    var_33 = [var_32]
    var_34 = module_0.Config()
    var_35 = 'test_something'
    var_36 = module_1.module(var_35, var_34)
    assert var_36 == 'test_*'
    var_37 = 'internal'
    var_38 = [var_37]
    var_39 = module_0.Config()
    var_40 = '.internal'
    var_41 = module_1.module(var_40, var_39)
    assert var_41 == 'internal'
    var_42 = '^requests\\.'
    var_43 = (var_42, var_11)
    var_44 = '^numpy\\.'
    var_45 = (var_44, var_11)
    var_46 = [var_43, var_45]
    var_47 = module_0.Config()
    var_48 = 'requests.models'
    var_49 = module_1.module(var_48, var_47)
    assert var_49 == 'THIRDPARTY'
    var_50 = 'numpy.array'
    var_51 = module_1.module(var_50, var_47)
    assert var_51 == 'THIRDPARTY'



# Parsed testcases at query #30
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'collections'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'STDLIB'
    var_5 = 'django'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'THIRDPARTY'
    var_7 = 'requests'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'THIRDPARTY'
    var_9 = '.local_module'
    var_10 = module_1.module(var_9, var_0)
    assert var_10 == 'LOCALFOLDER'
    var_11 = '..parent_module'
    var_12 = module_1.module(var_11, var_0)
    assert var_12 == 'LOCALFOLDER'
    var_13 = 'special'
    var_14 = 'special.module'
    var_15 = module_1.module(var_14, var_0)
    assert var_15 == 'special'
    var_16 = '^myapp\\..*'
    var_17 = 'FIRSTPARTY'
    var_18 = 'myapp.utils'
    var_19 = module_1.module(var_18, var_0)
    assert var_19 == 'FIRSTPARTY'
    var_20 = '/fake/src'
    var_21 = 'FIRSTPARTY'
    var_22 = 'reason'
    var_23 = 'mymodule'
    var_24 = module_1.module(var_23, var_0)
    assert var_24 == 'FIRSTPARTY'
    var_25 = 'unknown'
    var_26 = module_1.module(var_25, var_0)
    assert var_26 == 'CUSTOM'



# Parsed testcases at query #31
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
    var_5 = 'collections'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'STDLIB'
    var_7 = 'pytest'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'THIRDPARTY'
    var_9 = 'numpy'
    var_10 = module_1.module(var_9, var_0)
    assert var_10 == 'THIRDPARTY'
    var_11 = '.local_module'
    var_12 = module_1.module(var_11, var_0)
    assert var_12 == 'LOCALFOLDER'
    var_13 = '.subpackage.module'
    var_14 = module_1.module(var_13, var_0)
    assert var_14 == 'LOCALFOLDER'
    var_15 = 'test*'
    var_16 = 'test_module'
    var_17 = module_1.module(var_16, var_0)
    assert var_17 == 'test'
    var_18 = 'test.utils'
    var_19 = module_1.module(var_18, var_0)
    assert var_19 == 'test'
    var_20 = '^myapp\\.'
    var_21 = 'MYAPP'
    var_22 = (var_20, var_21)
    var_23 = 'STDLIB'
    var_24 = 'THIRDPARTY'
    var_25 = 'FIRSTPARTY'
    var_26 = 'myapp.utils'
    var_27 = module_1.module(var_26, var_0)
    assert var_27 == 'MYAPP'
    var_28 = 'myapp.models.user'
    var_29 = module_1.module(var_28, var_0)
    assert var_29 == 'MYAPP'
    var_30 = 'unknown_module'
    var_31 = module_1.module(var_30, var_0)
    assert var_31 == 'CUSTOM'
    var_32 = ''
    var_33 = module_1.module(var_32, var_0)
    assert var_33 == 'CUSTOM'
    var_34 = '...'
    var_35 = module_1.module(var_34, var_0)
    assert var_35 == 'LOCALFOLDER'
    var_36 = '.special*'
    var_37 = '.special_module'
    var_38 = module_1.module(var_37, var_0)
    assert var_38 == '.special'
    var_39 = 'django*'
    var_40 = 'django.contrib.auth'
    var_41 = module_1.module(var_40, var_0)
    assert var_41 == 'django'
    var_42 = 'exactmatch'
    var_43 = module_1.module(var_42, var_0)
    assert var_43 == 'exactmatch'
    var_44 = 'exactmatch.submodule'
    var_45 = module_1.module(var_44, var_0)
    assert var_45 == 'exactmatch'
    var_46 = module_1.module(var_1, var_0)
    var_47 = module_1.module(var_1, var_0)



# Parsed testcases at query #32
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
    var_5 = 'collections'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'STDLIB'
    var_7 = 'pytest'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'THIRDPARTY'
    var_9 = 'numpy'
    var_10 = module_1.module(var_9, var_0)
    assert var_10 == 'THIRDPARTY'
    var_11 = '.local_module'
    var_12 = module_1.module(var_11, var_0)
    assert var_12 == 'LOCALFOLDER'
    var_13 = '.subpackage.module'
    var_14 = module_1.module(var_13, var_0)
    assert var_14 == 'LOCALFOLDER'
    var_15 = 'special'
    var_16 = [var_15]
    var_17 = module_0.Config()
    var_18 = module_1.module(var_15, var_17)
    assert var_18 == 'special'
    var_19 = 'special.module'
    var_20 = module_1.module(var_19, var_17)
    assert var_20 == 'special'
    var_21 = 'special*'
    var_22 = [var_21]
    var_23 = module_0.Config()
    var_24 = 'special_lib'
    var_25 = module_1.module(var_24, var_23)
    assert var_25 == 'special_lib'
    var_26 = 'special_lib.utils'
    var_27 = module_1.module(var_26, var_23)
    assert var_27 == 'special_lib'
    var_28 = [var_7]
    var_29 = module_0.Config()
    var_30 = module_1.module(var_7, var_29)
    assert var_30 == 'THIRDPARTY'
    var_31 = 'myproject'
    var_32 = [var_31]
    var_33 = module_0.Config()
    var_34 = module_1.module(var_31, var_33)
    assert var_34 == 'FIRSTPARTY'
    var_35 = 'myproject.utils'
    var_36 = module_1.module(var_35, var_33)
    assert var_36 == 'FIRSTPARTY'
    var_37 = 'src'
    var_38 = var_1 / var_37
    var_39 = 'mypackage'
    var_40 = var_38 / var_39
    var_41 = var_38 / var_39
    var_42 = '__init__.py'
    var_43 = var_41 / var_42
    var_44 = [var_38]
    var_45 = module_0.Config()
    var_46 = module_1.module(var_39, var_45)
    assert var_46 == 'FIRSTPARTY'
    var_47 = 'mypackage.submodule'
    var_48 = module_1.module(var_47, var_45)
    assert var_48 == 'FIRSTPARTY'
    var_49 = 'src'
    var_50 = var_1 / var_49
    var_51 = 'namespace_pkg'
    var_52 = var_50 / var_51
    var_53 = 'subpkg'
    var_54 = var_52 / var_53
    var_55 = [var_50]
    var_56 = True
    var_57 = module_0.Config()
    var_58 = 'namespace_pkg.subpkg'
    var_59 = module_1.module(var_58, var_57)
    assert var_59 == 'FIRSTPARTY'
    var_60 = 'CUSTOM'
    var_61 = module_0.Config()
    var_62 = 'unknown_module'
    var_63 = module_1.module(var_62, var_61)
    assert var_63 == 'CUSTOM'



# Parsed testcases at query #33
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'testlib'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'testlib'
    var_3 = 'testlib.submodule'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'testlib'
    var_5 = '.local'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'LOCALFOLDER'
    var_7 = '.local.sub'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'LOCALFOLDER'
    var_9 = '^django\\.'
    var_10 = 'DJANGO'
    var_11 = 'FIRSTPARTY'
    var_12 = 'THIRDPARTY'
    var_13 = 'STDLIB'
    var_14 = 'django.app'
    var_15 = module_1.module(var_14, var_0)
    assert var_15 == 'DJANGO'
    var_16 = 'django.contrib.auth'
    var_17 = module_1.module(var_16, var_0)
    assert var_17 == 'DJANGO'
    var_18 = 'some_unknown_module'
    var_19 = module_1.module(var_18, var_0)
    assert var_19 == 'THIRDPARTY'
    var_20 = '/test/src'
    var_21 = 'mymodule'
    var_22 = module_1.module(var_21, var_0)
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'mynamespace'
    var_24 = 'mynamespace.sub'
    var_25 = module_1.module(var_24, var_0)
    assert var_25 == 'FIRSTPARTY'
    var_26 = 'cached_module'
    var_27 = module_1.module(var_26, var_0)
    var_28 = module_1.module(var_26, var_0)
    var_29 = module_0.Config()
    var_30 = 'any_module'
    var_31 = module_1.module(var_30, var_29)
    assert var_31 == 'STDLIB'



# Parsed testcases at query #34
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
    var_2 = 'collections'
    var_3 = module_0.module(var_2)
    assert var_3 == 'STDLIB'
    var_4 = 'django'
    var_5 = module_0.module(var_4)
    assert var_5 == 'THIRDPARTY'
    var_6 = '.local_module'
    var_7 = module_0.module(var_6)
    assert var_7 == 'LOCALFOLDER'
    var_8 = '.subpackage.module'
    var_9 = module_0.module(var_8)
    assert var_9 == 'LOCALFOLDER'
    var_10 = 'test'
    var_11 = [var_10]
    var_12 = module_1.Config()
    var_13 = module_0.module(var_10, var_12)
    assert var_13 == 'test'
    var_14 = 'test.module'
    var_15 = module_0.module(var_14, var_12)
    assert var_15 == 'test'
    var_16 = '^django.*'
    var_17 = 'DJANGO'
    var_18 = (var_16, var_17)
    var_19 = [var_18]
    var_20 = module_1.Config()
    var_21 = module_0.module(var_4, var_20)
    assert var_21 == 'DJANGO'
    var_22 = 'django.contrib'
    var_23 = module_0.module(var_22, var_20)
    assert var_23 == 'DJANGO'
    var_24 = '/test/path'
    var_25 = 'module'
    var_26 = module_0.module(var_25, var_20)
    assert var_26 == 'FIRSTPARTY'
    var_27 = 'test.namespace'
    var_28 = [var_27]
    var_29 = True
    var_30 = 'FIRSTPARTY'
    var_31 = 'reason'
    var_32 = 'test.namespace.module'
    var_33 = module_0.module(var_32, var_20)
    var_34 = 'CUSTOM'
    var_35 = module_1.Config()
    var_36 = 'unknown_module'
    var_37 = module_0.module(var_36, var_35)
    assert var_37 == 'CUSTOM'
    var_38 = module_1.Config()
    var_39 = module_0.module(var_30, var_38)
    var_40 = module_0.module(var_30, var_38)



# Parsed testcases at query #35
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'collections'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'STDLIB'
    var_5 = 'typing'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'STDLIB'
    var_7 = 'pytest'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'THIRDPARTY'
    var_9 = 'requests'
    var_10 = module_1.module(var_9, var_0)
    assert var_10 == 'THIRDPARTY'
    var_11 = 'numpy'
    var_12 = module_1.module(var_11, var_0)
    assert var_12 == 'THIRDPARTY'
    var_13 = '.local_module'
    var_14 = module_1.module(var_13, var_0)
    assert var_14 == 'LOCALFOLDER'
    var_15 = '..parent_module'
    var_16 = module_1.module(var_15, var_0)
    assert var_16 == 'LOCALFOLDER'
    var_17 = '...deep_module'
    var_18 = module_1.module(var_17, var_0)
    assert var_18 == 'LOCALFOLDER'
    var_19 = 'special'
    var_20 = 'special.module'
    var_21 = module_1.module(var_20, var_0)
    assert var_21 == 'special'
    var_22 = module_1.module(var_19, var_0)
    assert var_22 == 'special'
    var_23 = '^myapp\\..*'
    var_24 = 'MYAPP'
    var_25 = 'myapp.utils'
    var_26 = module_1.module(var_25, var_0)
    assert var_26 == 'MYAPP'
    var_27 = 'myapp.models.user'
    var_28 = module_1.module(var_27, var_0)
    assert var_28 == 'MYAPP'
    var_29 = '/project/src'
    var_30 = 'FIRSTPARTY'
    var_31 = 'reason'
    var_32 = 'myproject'
    var_33 = module_1.module(var_32, var_0)
    assert var_33 == 'FIRSTPARTY'
    var_34 = 'unknown_module'
    var_35 = module_1.module(var_34, var_0)
    assert var_35 == 'CUSTOM'
    var_36 = 'another.unknown'
    var_37 = module_1.module(var_36, var_0)
    assert var_37 == 'CUSTOM'
    var_38 = 'some.module'
    var_39 = module_1.module(var_38, var_0)
    assert var_39 == 'CUSTOM'



# Parsed testcases at query #36
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_separate'
    var_2 = 'test_separate.module'
    var_3 = module_1.module(var_2, var_0)
    assert var_3 == 'test_separate'
    var_4 = module_1.module(var_1, var_0)
    assert var_4 == 'test_separate'
    var_5 = '.local_module'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'LOCALFOLDER'
    var_7 = '^django\\.*'
    var_8 = 'DJANGO'
    var_9 = 'FIRSTPARTY'
    var_10 = 'THIRDPARTY'
    var_11 = 'STDLIB'
    var_12 = 'django.app'
    var_13 = module_1.module(var_12, var_0)
    assert var_13 == 'DJANGO'
    var_14 = 'django'
    var_15 = module_1.module(var_14, var_0)
    assert var_15 == 'DJANGO'
    var_16 = 'unknown_module'
    var_17 = module_1.module(var_16, var_0)
    assert var_17 == 'THIRDPARTY'
    var_18 = 'os'
    var_19 = module_1.module(var_18)
    assert var_19 == 'STDLIB'
    var_20 = 'collections'
    var_21 = module_1.module(var_20)
    assert var_21 == 'STDLIB'
    var_22 = 'tests*'
    var_23 = 'tests.unit.test_module'
    var_24 = module_1.module(var_23, var_0)
    assert var_24 == 'tests'
    var_25 = '.tests.integration'
    var_26 = module_1.module(var_25, var_0)
    assert var_26 == 'LOCALFOLDER'
    var_27 = 'exact_match'
    var_28 = 'pattern*'
    var_29 = module_1.module(var_27, var_0)
    assert var_29 == 'exact_match'
    var_30 = 'pattern.module'
    var_31 = module_1.module(var_30, var_0)
    assert var_31 == 'pattern'
    var_32 = 'pattern'
    var_33 = module_1.module(var_32, var_0)
    assert var_33 == 'pattern'
    var_34 = 'test.module'
    var_35 = module_0.Config()
    var_36 = 'any_module'
    var_37 = module_1.module(var_36, var_35)



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = set()
    var_4 = False
    var_5 = 'THIRDPARTY'
    var_6 = 'FIRSTPARTY'
    var_7 = 'LOCALFOLDER'
    var_8 = [var_6, var_7, var_5]
    var_9 = 'py'
    var_10 = [var_9]
    var_11 = frozenset(var_10)
    var_12 = module_0.Config()
    var_13 = 'os'
    var_14 = module_1.module(var_13, var_12)
    assert var_14 == 'THIRDPARTY'
    var_15 = 'unknown'
    var_16 = module_1.module(var_15, var_12)
    assert var_16 == 'FIRSTPARTY'
    var_17 = 'test*'
    var_18 = 'test_module'
    var_19 = module_1.module(var_18, var_12)
    assert var_19 == 'test'
    var_20 = '.local'
    var_21 = module_1.module(var_20, var_12)
    assert var_21 == 'LOCALFOLDER'
    var_22 = '^django'
    var_23 = 'DJANGO'
    var_24 = 'django.contrib'
    var_25 = module_1.module(var_24, var_12)
    assert var_25 == 'DJANGO'
    var_26 = '/fake/src'
    var_27 = 'myapp'
    var_28 = module_1.module(var_27, var_12)
    assert var_28 == 'FIRSTPARTY'
    var_29 = 'mynamespace'
    var_30 = 'mynamespace.sub'
    var_31 = module_1.module(var_30, var_12)
    assert var_31 == 'FIRSTPARTY'
    var_32 = 'auto.sub'
    var_33 = module_1.module(var_32, var_12)
    assert var_33 == 'FIRSTPARTY'



# Parsed testcases at query #2
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'collections'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'STDLIB'
    var_5 = 'django'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'THIRDPARTY'
    var_7 = 'requests'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'THIRDPARTY'
    var_9 = '.local_module'
    var_10 = module_1.module(var_9, var_0)
    assert var_10 == 'LOCALFOLDER'
    var_11 = 'my_project'
    var_12 = module_1.module(var_11, var_0)
    assert var_12 == 'FIRSTPARTY'
    var_13 = 'special'
    var_14 = 'special.module'
    var_15 = module_1.module(var_14, var_0)
    assert var_15 == 'special'
    var_16 = '^my_special.*'
    var_17 = 'CUSTOM'
    var_18 = 'my_special_package'
    var_19 = module_1.module(var_18, var_0)
    assert var_19 == 'CUSTOM'
    var_20 = 'unknown_module'
    var_21 = module_1.module(var_20, var_0)
    assert var_21 == 'DEFAULT'
    var_22 = '/fake/src'
    var_23 = 'fake_module'
    var_24 = module_1.module(var_23, var_0)
    assert var_24 == 'FIRSTPARTY'
    var_25 = 'numpy'
    var_26 = module_1.module(var_25, var_0)
    assert var_26 == 'THIRDPARTY'
    var_27 = 'typing'
    var_28 = module_1.module(var_27, var_0)
    assert var_28 == 'STDLIB'
    var_29 = '..relative'
    var_30 = module_1.module(var_29, var_0)
    assert var_30 == 'LOCALFOLDER'



# Parsed testcases at query #3
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = set()
    var_4 = False
    var_5 = 'THIRDPARTY'
    var_6 = 'FIRSTPARTY'
    var_7 = [var_6, var_5]
    var_8 = 'py'
    var_9 = [var_8]
    var_10 = frozenset(var_9)
    var_11 = module_0.Config()
    var_12 = 'os'
    var_13 = module_1.module(var_12, var_11)
    assert var_13 == 'THIRDPARTY'
    var_14 = 'unknown'
    var_15 = module_1.module(var_14, var_11)
    assert var_15 == 'FIRSTPARTY'
    var_16 = 'test*'
    var_17 = 'test_module'
    var_18 = module_1.module(var_17, var_11)
    assert var_18 == 'test*'
    var_19 = '.local'
    var_20 = module_1.module(var_19, var_11)
    assert var_20 == 'LOCALFOLDER'
    var_21 = '^django'
    var_22 = 'DJANGO'
    var_23 = 'django.contrib'
    var_24 = module_1.module(var_23, var_11)
    assert var_24 == 'DJANGO'
    var_25 = '/fake/src'
    var_26 = 'mymodule'
    var_27 = module_1.module(var_26, var_11)
    assert var_27 == 'FIRSTPARTY'
    var_28 = 'mynamespace'
    var_29 = 'mynamespace.sub'
    var_30 = module_1.module(var_29, var_11)
    assert var_30 == 'FIRSTPARTY'
    var_31 = 'namespace.sub'
    var_32 = module_1.module(var_31, var_11)
    assert var_32 == 'FIRSTPARTY'



# Parsed testcases at query #4
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
    var_2 = 'sys'
    var_3 = module_0.module(var_2)
    assert var_3 == 'STDLIB'
    var_4 = 'collections'
    var_5 = module_0.module(var_4)
    assert var_5 == 'STDLIB'
    var_6 = 'pytest'
    var_7 = module_0.module(var_6)
    assert var_7 == 'THIRDPARTY'
    var_8 = 'isort'
    var_9 = module_0.module(var_8)
    assert var_9 == 'THIRDPARTY'
    var_10 = '.local_module'
    var_11 = module_0.module(var_10)
    assert var_11 == 'LOCALFOLDER'
    var_12 = '.subpackage.module'
    var_13 = module_0.module(var_12)
    assert var_13 == 'LOCALFOLDER'
    var_14 = 'test_module'
    var_15 = [var_14]
    var_16 = module_1.Config()
    var_17 = module_0.module(var_14, var_16)
    assert var_17 == 'test_module'
    var_18 = 'test_module.sub'
    var_19 = module_0.module(var_18, var_16)
    assert var_19 == 'test_module'
    var_20 = '^test_.*'
    var_21 = 'TEST'
    var_22 = (var_20, var_21)
    var_23 = [var_22]
    var_24 = 'STDLIB'
    var_25 = 'THIRDPARTY'
    var_26 = 'FIRSTPARTY'
    var_27 = [var_21, var_24, var_25, var_26]
    var_28 = module_1.Config()
    var_29 = 'test_example'
    var_30 = module_0.module(var_29, var_28)
    assert var_30 == 'TEST'
    var_31 = module_0.module(var_18, var_28)
    assert var_31 == 'TEST'
    var_32 = '.'
    var_33 = 'CUSTOM'
    var_34 = module_1.Config()
    var_35 = 'unknown_module'
    var_36 = module_0.module(var_35, var_34)
    assert var_36 == 'CUSTOM'
    var_37 = ''
    var_38 = module_0.module(var_37)
    assert var_38 == 'STDLIB'
    var_39 = 'os.path'
    var_40 = module_0.module(var_39)
    assert var_40 == 'STDLIB'
    var_41 = 'collections.abc'
    var_42 = module_0.module(var_41)
    assert var_42 == 'STDLIB'
    var_43 = module_0.module(var_0)
    var_44 = module_0.module(var_0)



# Parsed testcases at query #5
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = set()
    var_4 = False
    var_5 = 'THIRDPARTY'
    var_6 = 'FIRSTPARTY'
    var_7 = 'LOCALFOLDER'
    var_8 = [var_6, var_5, var_7]
    var_9 = 'py'
    var_10 = [var_9]
    var_11 = frozenset(var_10)
    var_12 = module_0.Config()
    var_13 = 'os'
    var_14 = module_1.module(var_13, var_12)
    assert var_14 == 'THIRDPARTY'
    var_15 = '.local'
    var_16 = module_1.module(var_15, var_12)
    assert var_16 == 'LOCALFOLDER'
    var_17 = 'special'
    var_18 = 'special.module'
    var_19 = module_1.module(var_18, var_12)
    assert var_19 == 'special'
    var_20 = '^django'
    var_21 = 'DJANGO'
    var_22 = 'django.apps'
    var_23 = module_1.module(var_22, var_12)
    assert var_23 == 'DJANGO'
    var_24 = '/fake/src'
    var_25 = 'fake_module'
    var_26 = module_1.module(var_25, var_12)
    assert var_26 == 'FIRSTPARTY'
    var_27 = 'STDLIB'
    var_28 = 'unknown'
    var_29 = module_1.module(var_28, var_12)
    assert var_29 == 'STDLIB'



# Parsed testcases at query #6
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'collections'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'STDLIB'
    var_5 = 'typing'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'STDLIB'
    var_7 = 'requests'
    var_8 = 'pytest'
    var_9 = [var_7, var_8]
    var_10 = module_0.Config()
    var_11 = module_1.module(var_7, var_10)
    assert var_11 == 'THIRDPARTY'
    var_12 = module_1.module(var_8, var_10)
    assert var_12 == 'THIRDPARTY'
    var_13 = 'my_module'
    var_14 = 'my_package'
    var_15 = [var_13, var_14]
    var_16 = module_0.Config()
    var_17 = module_1.module(var_13, var_16)
    assert var_17 == 'FIRSTPARTY'
    var_18 = 'my_package.submodule'
    var_19 = module_1.module(var_18, var_16)
    assert var_19 == 'FIRSTPARTY'
    var_20 = 'special'
    var_21 = [var_20]
    var_22 = module_0.Config()
    var_23 = module_1.module(var_20, var_22)
    assert var_23 == 'special'
    var_24 = 'special.submodule'
    var_25 = module_1.module(var_24, var_22)
    assert var_25 == 'special'
    var_26 = '.local_module'
    var_27 = module_1.module(var_26, var_22)
    assert var_27 == 'LOCALFOLDER'
    var_28 = '.relative.import'
    var_29 = module_1.module(var_28, var_22)
    assert var_29 == 'LOCALFOLDER'
    var_30 = 'THIRDPARTY'
    var_31 = module_0.Config()
    var_32 = 'unknown_module'
    var_33 = module_1.module(var_32, var_31)
    assert var_33 == 'THIRDPARTY'
    var_34 = '^google\\.cloud\\..*'
    var_35 = 'GOOGLE'
    var_36 = (var_34, var_35)
    var_37 = '^aws\\.services\\..*'
    var_38 = 'AWS'
    var_39 = (var_37, var_38)
    var_40 = [var_36, var_39]
    var_41 = 'STDLIB'
    var_42 = 'FIRSTPARTY'
    var_43 = [var_41, var_35, var_38, var_30, var_42]
    var_44 = module_0.Config()
    var_45 = 'google.cloud.storage'
    var_46 = module_1.module(var_45, var_44)
    assert var_46 == 'GOOGLE'
    var_47 = 'aws.services.s3'
    var_48 = module_1.module(var_47, var_44)
    assert var_48 == 'AWS'
    var_49 = 'google.cloud'
    var_50 = module_1.module(var_49, var_44)
    assert var_50 == 'GOOGLE'
    var_51 = '/fake/src'
    var_52 = 'fake_module'
    var_53 = module_1.module(var_52, var_44)
    assert var_53 == 'FIRSTPARTY'
    var_54 = 'my_namespace'
    var_55 = [var_54]
    var_56 = module_0.Config()
    var_57 = 'my_namespace.subpackage'
    var_58 = module_1.module(var_57, var_56)
    assert var_58 == 'FIRSTPARTY'
    var_59 = True
    var_60 = module_0.Config()
    var_61 = 'auto_namespace.sub'
    var_62 = module_1.module(var_61, var_60)
    assert var_62 == 'FIRSTPARTY'
    var_63 = 'numpy'
    var_64 = [var_63]
    var_65 = 'myapp'
    var_66 = [var_65]
    var_67 = module_0.Config()
    var_68 = module_1.module(var_63, var_67)
    assert var_68 == 'THIRDPARTY'
    var_69 = 'myapp.utils'
    var_70 = module_1.module(var_69, var_67)
    assert var_70 == 'FIRSTPARTY'
    var_71 = 'unknown.lib'
    var_72 = module_1.module(var_71, var_67)
    var_73 = 'tests'
    var_74 = 'docs'
    var_75 = [var_73, var_74]
    var_76 = module_0.Config()
    var_77 = 'tests.unit'
    var_78 = module_1.module(var_77, var_76)
    assert var_78 == 'tests'
    var_79 = 'docs.source'
    var_80 = module_1.module(var_79, var_76)
    assert var_80 == 'docs'
    var_81 = 'regular.module'
    var_82 = module_1.module(var_81, var_76)
    var_83 = '^test_.*'
    var_84 = 'TEST'
    var_85 = (var_83, var_84)
    var_86 = [var_85]
    var_87 = [var_41, var_84, var_30]
    var_88 = module_0.Config()
    var_89 = 'test_module'
    var_90 = module_1.module(var_89, var_88)
    assert var_90 == 'TEST'
    var_91 = 'test_utils.helpers'
    var_92 = module_1.module(var_91, var_88)
    assert var_92 == 'TEST'
    var_93 = 'normal.module'
    var_94 = module_1.module(var_93, var_88)



# Parsed testcases at query #7
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'some_module'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'test'
    var_4 = [var_3]
    var_5 = module_0.Config()
    var_6 = 'test.module'
    var_7 = module_1.module(var_6, var_5)
    assert var_7 == 'test'
    var_8 = module_0.Config()
    var_9 = '.local_module'
    var_10 = module_1.module(var_9, var_8)
    assert var_10 == 'LOCALFOLDER'
    var_11 = '^django\\.'
    var_12 = 'THIRDPARTY'
    var_13 = (var_11, var_12)
    var_14 = [var_13]
    var_15 = module_0.Config()
    var_16 = 'django.app'
    var_17 = module_1.module(var_16, var_15)
    assert var_17 == 'THIRDPARTY'
    var_18 = 'mypackage'
    var_19 = ''
    var_20 = [var_3]
    var_21 = module_0.Config()
    var_22 = module_1.module(var_19, var_21)
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'parent'
    var_24 = 'child'
    var_25 = ''
    var_26 = [var_20]
    var_27 = module_0.Config()
    var_28 = 'parent.child'
    var_29 = module_1.module(var_28, var_27)
    assert var_29 == 'FIRSTPARTY'
    var_30 = 'namespace_pkg'
    var_31 = [var_3]
    var_32 = True
    var_33 = module_0.Config()
    var_34 = module_1.module(var_30, var_33)
    assert var_34 == 'FIRSTPARTY'
    var_35 = 'compiled.pyd'
    var_36 = 'dummy content'
    var_37 = [var_24]
    var_38 = module_0.Config()
    var_39 = 'compiled'
    var_40 = module_1.module(var_39, var_38)
    assert var_40 == 'FIRSTPARTY'
    var_41 = [var_36]
    var_42 = module_0.Config()
    var_43 = True
    var_44 = module_1.module(var_34, var_42)
    assert var_44 == 'FIRSTPARTY'



# Parsed testcases at query #8
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
    var_2 = 'sys'
    var_3 = module_0.module(var_2)
    assert var_3 == 'STDLIB'
    var_4 = 'django'
    var_5 = [var_4]
    var_6 = module_1.Config()
    var_7 = module_0.module(var_4, var_6)
    assert var_7 == 'django'
    var_8 = 'django.contrib'
    var_9 = module_0.module(var_8, var_6)
    assert var_9 == 'django'
    var_10 = '.local_module'
    var_11 = module_0.module(var_10)
    assert var_11 == 'LOCALFOLDER'
    var_12 = '..parent_module'
    var_13 = module_0.module(var_12)
    assert var_13 == 'LOCALFOLDER'
    var_14 = '^requests$'
    var_15 = 'THIRDPARTY'
    var_16 = (var_14, var_15)
    var_17 = [var_16]
    var_18 = module_1.Config()
    var_19 = 'requests'
    var_20 = module_0.module(var_19, var_18)
    assert var_20 == 'THIRDPARTY'
    var_21 = 'requests.models'
    var_22 = module_0.module(var_21, var_18)
    assert var_22 == 'THIRDPARTY'
    var_23 = '/test/src'
    var_24 = 'FIRSTPARTY'
    var_25 = 'reason'
    var_26 = 'mymodule'
    var_27 = module_0.module(var_26, var_18)
    assert var_27 == 'FIRSTPARTY'
    var_28 = 'mynamespace'
    var_29 = [var_28]
    var_30 = True
    var_31 = 'FIRSTPARTY'
    var_32 = 'reason'
    var_33 = 'mynamespace.submodule'
    var_34 = module_0.module(var_33, var_18)
    assert var_34 == 'FIRSTPARTY'
    var_35 = module_1.Config()
    var_36 = module_0.module(var_31, var_35)
    var_37 = module_0.module(var_31, var_35)
    var_38 = ''
    var_39 = module_0.module(var_38)
    assert var_39 == 'STDLIB'
    var_40 = module_1.Config()
    var_41 = 'nonexistent_module_xyz'
    var_42 = module_0.module(var_41, var_40)
    assert var_42 == 'STDLIB'



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
    var_4 = [var_3]
    var_5 = module_0.Config()
    var_6 = module_1.module(var_3, var_5)
    assert var_6 == 'django'
    var_7 = 'django.contrib'
    var_8 = module_1.module(var_7, var_5)
    assert var_8 == 'django'
    var_9 = '.local_module'
    var_10 = module_1.module(var_9, var_5)
    assert var_10 == 'LOCALFOLDER'
    var_11 = '..parent_module'
    var_12 = module_1.module(var_11, var_5)
    assert var_12 == 'LOCALFOLDER'
    var_13 = '^django'
    var_14 = 'THIRDPARTY'
    var_15 = (var_13, var_14)
    var_16 = [var_15]
    var_17 = 'FUTURE'
    var_18 = 'STDLIB'
    var_19 = 'FIRSTPARTY'
    var_20 = 'LOCALFOLDER'
    var_21 = [var_17, var_18, var_14, var_19, var_20]
    var_22 = module_0.Config()
    var_23 = 'django.test'
    var_24 = module_1.module(var_23, var_22)
    assert var_24 == 'THIRDPARTY'
    var_25 = '/test/src'
    var_26 = 'my_module'
    var_27 = module_1.module(var_26, var_22)
    assert var_27 == 'FIRSTPARTY'
    var_28 = 'my_namespace'
    var_29 = [var_28]
    var_30 = True
    var_31 = 'FIRSTPARTY'
    var_32 = 'Found in src_paths'
    var_33 = 'my_namespace.submodule'
    var_34 = module_1.module(var_33, var_22)
    assert var_34 == 'FIRSTPARTY'
    var_35 = module_0.Config()
    var_36 = 'unknown_module'
    var_37 = module_1.module(var_36, var_35)
    assert var_37 == 'THIRDPARTY'



# Parsed testcases at query #10
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
    var_2 = 'sys'
    var_3 = module_0.module(var_2)
    assert var_3 == 'STDLIB'
    var_4 = 'collections'
    var_5 = module_0.module(var_4)
    assert var_5 == 'STDLIB'
    var_6 = 'pytest'
    var_7 = module_0.module(var_6)
    assert var_7 == 'THIRDPARTY'
    var_8 = 'numpy'
    var_9 = module_0.module(var_8)
    assert var_9 == 'THIRDPARTY'
    var_10 = '.local_module'
    var_11 = module_0.module(var_10)
    assert var_11 == 'LOCALFOLDER'
    var_12 = '.subpackage.module'
    var_13 = module_0.module(var_12)
    assert var_13 == 'LOCALFOLDER'
    var_14 = 'test*'
    var_15 = [var_14]
    var_16 = module_1.Config()
    var_17 = 'test_module'
    var_18 = module_0.module(var_17, var_16)
    assert var_18 == 'test'
    var_19 = 'test.utils'
    var_20 = module_0.module(var_19, var_16)
    assert var_20 == 'test'
    var_21 = '^myapp\\.*'
    var_22 = 'MYAPP'
    var_23 = '^company\\.*'
    var_24 = 'COMPANY'
    var_25 = 'myapp.utils'
    var_26 = module_0.module(var_25, var_16)
    assert var_26 == 'MYAPP'
    var_27 = 'company.api'
    var_28 = module_0.module(var_27, var_16)
    assert var_28 == 'COMPANY'
    var_29 = 'src'
    var_30 = var_0 / var_29
    var_31 = '__init__.py'
    var_32 = var_30 / var_31
    var_33 = [var_30]
    var_34 = module_1.Config()
    var_35 = 'mymodule.py'
    var_36 = var_30 / var_35
    var_37 = 'MYDEFAULT'
    var_38 = module_1.Config()
    var_39 = 'unknown_module'
    var_40 = module_0.module(var_39, var_38)
    assert var_40 == 'MYDEFAULT'
    var_41 = 'os.path'
    var_42 = module_0.module(var_41)
    assert var_42 == 'STDLIB'
    var_43 = 'collections.abc'
    var_44 = module_0.module(var_43)
    assert var_44 == 'STDLIB'
    var_45 = module_1.Config()
    var_46 = module_0.module(var_0, var_45)
    var_47 = module_0.module(var_0, var_45)
    var_48 = ''
    var_49 = module_0.module(var_48)
    assert var_49 == 'STDLIB'
    var_50 = 'mynamespace'
    var_51 = [var_50]
    var_52 = True
    var_53 = module_1.Config()
    var_54 = 'very.long.nested.module.name'
    var_55 = module_0.module(var_54)
    assert var_55 == 'THIRDPARTY'
    var_56 = 'special*'
    var_57 = [var_56]
    var_58 = '^special\\.*'
    var_59 = 'KNOWNSPECIAL'
    var_60 = 'special.module'
    var_61 = module_0.module(var_60, var_53)
    assert var_61 == 'special'



# Parsed testcases at query #11
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
    var_2 = 'collections'
    var_3 = module_0.module(var_2)
    assert var_3 == 'STDLIB'
    var_4 = 'pytest'
    var_5 = module_0.module(var_4)
    assert var_5 == 'THIRDPARTY'
    var_6 = '.local_module'
    var_7 = module_0.module(var_6)
    assert var_7 == 'LOCALFOLDER'
    var_8 = '.subpackage.module'
    var_9 = module_0.module(var_8)
    assert var_9 == 'LOCALFOLDER'
    var_10 = 'test*'
    var_11 = [var_10]
    var_12 = module_1.Config()
    var_13 = 'test_module'
    var_14 = module_0.module(var_13, var_12)
    assert var_14 == 'test'
    var_15 = 'test.utils'
    var_16 = module_0.module(var_15, var_12)
    assert var_16 == 'test'
    var_17 = '^django.*'
    var_18 = 'DJANGO'
    var_19 = (var_17, var_18)
    var_20 = [var_19]
    var_21 = module_1.Config()
    var_22 = 'django.core'
    var_23 = module_0.module(var_22, var_21)
    assert var_23 == 'DJANGO'
    var_24 = 'django.contrib.auth'
    var_25 = module_0.module(var_24, var_21)
    assert var_25 == 'DJANGO'
    var_26 = '/project/src'
    var_27 = 'my_module'
    var_28 = module_0.module(var_27, var_21)
    assert var_28 == 'FIRSTPARTY'
    var_29 = 'my_namespace'
    var_30 = [var_29]
    var_31 = True
    var_32 = 'FIRSTPARTY'
    var_33 = 'Found in src_paths'
    var_34 = 'my_namespace.subpackage'
    var_35 = module_0.module(var_34, var_21)
    assert var_35 == 'FIRSTPARTY'
    var_36 = 'CUSTOM'
    var_37 = module_1.Config()
    var_38 = 'unknown_module'
    var_39 = module_0.module(var_38, var_37)
    assert var_39 == 'CUSTOM'
    var_40 = 'numpy'
    var_41 = module_0.module(var_40)
    assert var_41 == 'THIRDPARTY'
    var_42 = 'numpy.core'
    var_43 = module_0.module(var_42)
    assert var_43 == 'THIRDPARTY'
    var_44 = 'numpy.core.multiarray'
    var_45 = module_0.module(var_44)
    assert var_45 == 'THIRDPARTY'
    var_46 = module_1.Config()
    var_47 = module_0.module(var_32, var_46)
    assert var_47 == 'STDLIB'
    var_48 = module_1.Config()
    var_49 = module_0.module(var_32, var_48)
    var_50 = module_0.module(var_32, var_48)



# Parsed testcases at query #12
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_separate'
    var_2 = 'test_separate.module'
    var_3 = module_1.module(var_2, var_0)
    assert var_3 == 'test_separate'
    var_4 = module_1.module(var_1, var_0)
    assert var_4 == 'test_separate'
    var_5 = '.local_module'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'LOCALFOLDER'
    var_7 = '^django\\.'
    var_8 = 'DJANGO'
    var_9 = 'FIRSTPARTY'
    var_10 = 'THIRDPARTY'
    var_11 = 'STDLIB'
    var_12 = 'django.app'
    var_13 = module_1.module(var_12, var_0)
    assert var_13 == 'DJANGO'
    var_14 = '/test/src'
    var_15 = 'py'
    var_16 = [var_15]
    var_17 = 'myapp'
    var_18 = module_1.module(var_17, var_0)
    assert var_18 == 'FIRSTPARTY'
    var_19 = 'some_unknown_module'
    var_20 = module_1.module(var_19, var_0)
    var_21 = 'mynamespace'
    var_22 = 'mynamespace.submodule'
    var_23 = module_1.module(var_22, var_0)
    assert var_23 == 'FIRSTPARTY'
    var_24 = 'cached_test'
    var_25 = 'cached_test.module'
    var_26 = module_1.module(var_25, var_0)
    var_27 = module_1.module(var_25, var_0)



# Parsed testcases at query #13
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
    assert var_4 == 'LOCALFOLDER'
    var_5 = '^django\\.'
    var_6 = 'THIRDPARTY'
    var_7 = (var_5, var_6)
    var_8 = 'FIRSTPARTY'
    var_9 = 'django.test'
    var_10 = module_1.module(var_9, var_0)
    assert var_10 == 'THIRDPARTY'
    var_11 = 'unknown_module'
    var_12 = module_1.module(var_11, var_0)
    var_13 = '/test/path'
    var_14 = 'module'
    var_15 = module_1.module(var_14, var_0)
    assert var_15 == 'FIRSTPARTY'
    var_16 = 'namespace'
    var_17 = 'namespace.sub'
    var_18 = module_1.module(var_17, var_0)
    assert var_18 == 'FIRSTPARTY'
    var_19 = 'auto_ns.sub'
    var_20 = module_1.module(var_19, var_0)
    assert var_20 == 'FIRSTPARTY'
    var_21 = 'exact'
    var_22 = module_1.module(var_21, var_0)
    assert var_22 == 'exact'
    var_23 = 'exact.sub'
    var_24 = module_1.module(var_23, var_0)
    assert var_24 == 'exact'
    var_25 = '.exact'
    var_26 = module_1.module(var_25, var_0)
    assert var_26 == 'LOCALFOLDER'
    var_27 = 'cached_module'
    var_28 = 'forced'
    var_29 = '^known\\.'
    var_30 = 'KNOWN'
    var_31 = (var_29, var_30)
    var_32 = '/src'
    var_33 = module_1.module(var_28, var_0)
    assert var_33 == 'forced'
    var_34 = '.local'
    var_35 = module_1.module(var_34, var_0)
    assert var_35 == 'LOCALFOLDER'
    var_36 = 'known.pattern'
    var_37 = module_1.module(var_36, var_0)
    assert var_37 == 'KNOWN'
    var_38 = 'src_module'
    var_39 = module_1.module(var_38, var_0)
    assert var_39 == 'FIRSTPARTY'
    var_40 = 'default_module'
    var_41 = module_1.module(var_40, var_0)



# Parsed testcases at query #14
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'some_module'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'test_module'
    var_4 = [var_3]
    var_5 = module_0.Config()
    var_6 = module_1.module(var_3, var_5)
    assert var_6 == 'test_module'
    var_7 = 'test_*'
    var_8 = [var_7]
    var_9 = module_0.Config()
    var_10 = module_1.module(var_3, var_9)
    assert var_10 == 'test_*'
    var_11 = [var_3]
    var_12 = module_0.Config()
    var_13 = '.test_module'
    var_14 = module_1.module(var_13, var_12)
    assert var_14 == 'test_module'
    var_15 = module_0.Config()
    var_16 = '.local_module'
    var_17 = module_1.module(var_16, var_15)
    assert var_17 == 'LOCALFOLDER'
    var_18 = '^django\\.'
    var_19 = 'THIRDPARTY'
    var_20 = (var_18, var_19)
    var_21 = [var_20]
    var_22 = module_0.Config()
    var_23 = 'django.apps'
    var_24 = module_1.module(var_23, var_22)
    assert var_24 == 'THIRDPARTY'
    var_25 = '^django$'
    var_26 = (var_25, var_19)
    var_27 = [var_26]
    var_28 = module_0.Config()
    var_29 = 'django.apps.config'
    var_30 = module_1.module(var_29, var_28)
    assert var_30 == 'THIRDPARTY'
    var_31 = 'FIRSTPARTY'
    var_32 = module_0.Config()
    var_33 = 'unknown_module'
    var_34 = module_1.module(var_33, var_32)
    assert var_34 == 'FIRSTPARTY'
    var_35 = 'os'
    var_36 = module_1.module(var_35)
    assert var_36 == 'STDLIB'
    var_37 = 'exact'
    var_38 = [var_37]
    var_39 = module_0.Config()
    var_40 = module_1.module(var_37, var_39)
    assert var_40 == 'exact'
    var_41 = 'exact.submodule'
    var_42 = module_1.module(var_41, var_39)
    assert var_42 == 'exact'
    var_43 = 'partial*'
    var_44 = [var_43]
    var_45 = module_0.Config()
    var_46 = 'partial_match'
    var_47 = module_1.module(var_46, var_45)
    assert var_47 == 'partial*'
    var_48 = 'partial'
    var_49 = module_1.module(var_48, var_45)
    assert var_49 == 'partial*'
    var_50 = 'special'
    var_51 = [var_50]
    var_52 = module_0.Config()
    var_53 = '.special'
    var_54 = module_1.module(var_53, var_52)
    assert var_54 == 'special'
    var_55 = '^special\\.'
    var_56 = (var_55, var_19)
    var_57 = [var_56]
    var_58 = module_0.Config()
    var_59 = '.special.module'
    var_60 = module_1.module(var_59, var_58)
    assert var_60 == 'LOCALFOLDER'
    var_61 = '^myapp\\.'
    var_62 = (var_61, var_19)
    var_63 = [var_62]
    var_64 = '/fake/path'
    var_65 = 'myapp.utils'
    var_66 = module_1.module(var_65, var_58)
    assert var_66 == 'THIRDPARTY'
    var_67 = 'cached'
    var_68 = [var_67]
    var_69 = module_0.Config()
    var_70 = 'cached.module'
    var_71 = module_1.module(var_70, var_69)
    var_72 = module_1.module(var_70, var_69)
    var_73 = module_0.Config()
    var_74 = ''
    var_75 = module_1.module(var_74, var_73)
    assert var_75 == 'STDLIB'
    var_76 = 'deep.module'
    var_77 = [var_76]
    var_78 = module_0.Config()
    var_79 = 'deep.module.structure.here'
    var_80 = module_1.module(var_79, var_78)
    assert var_80 == 'deep.module'



# Parsed testcases at query #15
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_module'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'test_module'
    var_3 = 'test_module.submodule'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'test_module'
    var_5 = '.local_module'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'LOCALFOLDER'
    var_7 = '^django\\.*'
    var_8 = 'THIRDPARTY'
    var_9 = 'FIRSTPARTY'
    var_10 = 'django.test'
    var_11 = module_1.module(var_10, var_0)
    assert var_11 == 'THIRDPARTY'
    var_12 = 'unknown_module'
    var_13 = module_1.module(var_12, var_0)
    assert var_13 == 'STDLIB'
    var_14 = '/test/src'
    var_15 = 'mymodule'
    var_16 = module_1.module(var_15, var_0)
    assert var_16 == 'FIRSTPARTY'
    var_17 = 'mynamespace'
    var_18 = 'mynamespace.subpackage'
    var_19 = module_1.module(var_18, var_0)
    assert var_19 == 'FIRSTPARTY'
    var_20 = 'auto_ns.sub'
    var_21 = module_1.module(var_20, var_0)
    assert var_21 == 'FIRSTPARTY'



# Parsed testcases at query #16
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
    var_5 = 'collections'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'STDLIB'
    var_7 = 'test_module'
    var_8 = [var_7]
    var_9 = module_0.Config()
    var_10 = module_1.module(var_7, var_9)
    assert var_10 == 'test_module'
    var_11 = 'test_module.submodule'
    var_12 = module_1.module(var_11, var_9)
    assert var_12 == 'test_module'
    var_13 = module_0.Config()
    var_14 = '.local_module'
    var_15 = module_1.module(var_14, var_13)
    assert var_15 == 'LOCALFOLDER'
    var_16 = '.subpackage.module'
    var_17 = module_1.module(var_16, var_13)
    assert var_17 == 'LOCALFOLDER'
    var_18 = '^django\\.'
    var_19 = 'DJANGO'
    var_20 = (var_18, var_19)
    var_21 = '^requests'
    var_22 = 'THIRDPARTY'
    var_23 = (var_21, var_22)
    var_24 = [var_20, var_23]
    var_25 = module_0.Config()
    var_26 = 'django.apps'
    var_27 = module_1.module(var_26, var_25)
    assert var_27 == 'DJANGO'
    var_28 = 'django.contrib.auth'
    var_29 = module_1.module(var_28, var_25)
    assert var_29 == 'DJANGO'
    var_30 = 'requests'
    var_31 = module_1.module(var_30, var_25)
    assert var_31 == 'THIRDPARTY'
    var_32 = 'requests.models'
    var_33 = module_1.module(var_32, var_25)
    assert var_33 == 'THIRDPARTY'
    var_34 = module_0.Config()
    var_35 = 'unknown_module'
    var_36 = module_1.module(var_35, var_34)
    assert var_36 == 'THIRDPARTY'
    var_37 = 'special'
    var_38 = [var_37]
    var_39 = '^numpy'
    var_40 = 'NUMPY'
    var_41 = (var_39, var_40)
    var_42 = [var_41]
    var_43 = 'OTHER'
    var_44 = module_0.Config()
    var_45 = module_1.module(var_37, var_44)
    assert var_45 == 'special'
    var_46 = 'special.sub'
    var_47 = module_1.module(var_46, var_44)
    assert var_47 == 'special'
    var_48 = 'numpy.array'
    var_49 = module_1.module(var_48, var_44)
    assert var_49 == 'NUMPY'
    var_50 = 'unknown'
    var_51 = module_1.module(var_50, var_44)
    assert var_51 == 'OTHER'
    var_52 = module_0.Config()
    var_53 = 'forced'
    var_54 = [var_53]
    var_55 = '^known'
    var_56 = 'KNOWN'
    var_57 = (var_55, var_56)
    var_58 = [var_57]
    var_59 = 'DEFAULT'
    var_60 = module_0.Config()
    var_61 = 'forced.module'
    var_62 = module_1.module(var_61, var_60)
    assert var_62 == 'forced'
    var_63 = '.local'
    var_64 = module_1.module(var_63, var_60)
    assert var_64 == 'LOCALFOLDER'
    var_65 = 'known.pattern'
    var_66 = module_1.module(var_65, var_60)
    assert var_66 == 'KNOWN'
    var_67 = 'other'
    var_68 = module_1.module(var_67, var_60)
    assert var_68 == 'DEFAULT'



# Parsed testcases at query #17
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
    var_2 = 'collections'
    var_3 = module_0.module(var_2)
    assert var_3 == 'STDLIB'
    var_4 = 'django'
    var_5 = module_0.module(var_4)
    assert var_5 == 'THIRDPARTY'
    var_6 = 'requests'
    var_7 = module_0.module(var_6)
    assert var_7 == 'THIRDPARTY'
    var_8 = 'django.contrib'
    var_9 = [var_8]
    var_10 = module_1.Config()
    var_11 = 'django.contrib.admin'
    var_12 = module_0.module(var_11, var_10)
    assert var_12 == 'django.contrib'
    var_13 = '.local_module'
    var_14 = module_0.module(var_13)
    assert var_14 == 'LOCALFOLDER'
    var_15 = '.subpackage.module'
    var_16 = module_0.module(var_15)
    assert var_16 == 'LOCALFOLDER'
    var_17 = '^myproject\\.*'
    var_18 = 'FIRSTPARTY'
    var_19 = '^test_.*'
    var_20 = 'TESTS'
    var_21 = 'STDLIB'
    var_22 = 'THIRDPARTY'
    var_23 = [var_18, var_21, var_22, var_20]
    var_24 = 'myproject.utils'
    var_25 = module_0.module(var_24, var_10)
    assert var_25 == 'FIRSTPARTY'
    var_26 = 'test_module'
    var_27 = module_0.module(var_26, var_10)
    assert var_27 == 'TESTS'
    var_28 = 'src'
    var_29 = var_0 / var_28
    var_30 = 'mypackage'
    var_31 = var_29 / var_30
    var_32 = var_29 / var_30
    var_33 = '__init__.py'
    var_34 = var_32 / var_33
    var_35 = [var_29]
    var_36 = module_1.Config()
    var_37 = module_0.module(var_30, var_36)
    assert var_37 == 'FIRSTPARTY'
    var_38 = 'mypackage.submodule'
    var_39 = module_0.module(var_38, var_36)
    assert var_39 == 'FIRSTPARTY'
    var_40 = 'src'
    var_41 = var_0 / var_40
    var_42 = 'mynamespace'
    var_43 = var_41 / var_42
    var_44 = 'setup.cfg'
    var_45 = var_43 / var_44
    var_46 = [var_41]
    var_47 = True
    var_48 = module_1.Config()
    var_49 = module_0.module(var_42, var_48)
    assert var_49 == 'FIRSTPARTY'
    var_50 = 'CUSTOM'
    var_51 = module_1.Config()
    var_52 = 'unknown_module'
    var_53 = module_0.module(var_52, var_51)
    assert var_53 == 'CUSTOM'
    var_54 = module_1.Config()
    var_55 = module_0.module(var_0, var_54)
    var_56 = module_0.module(var_0, var_54)



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
    assert var_4 == 'LOCALFOLDER'
    var_5 = '^django\\.'
    var_6 = 'THIRDPARTY'
    var_7 = 'django.test'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'THIRDPARTY'
    var_9 = 'unknown_module'
    var_10 = module_1.module(var_9, var_0)
    var_11 = 'test.*'
    var_12 = 'test.sub.module'
    var_13 = module_1.module(var_12, var_0)
    assert var_13 == 'test.*'
    var_14 = 'exact_match'
    var_15 = module_1.module(var_14, var_0)
    assert var_15 == 'exact_match'
    var_16 = '.hidden*'
    var_17 = '.hidden_module'
    var_18 = module_1.module(var_17, var_0)
    assert var_18 == '.hidden*'
    var_19 = '^requests$'
    var_20 = 'requests'
    var_21 = module_1.module(var_20, var_0)
    assert var_21 == 'THIRDPARTY'
    var_22 = 'requests.models'
    var_23 = module_1.module(var_22, var_0)
    var_24 = '^numpy\\.'
    var_25 = 'SCIENTIFIC'
    var_26 = '^numpy$'
    var_27 = 'numpy'
    var_28 = module_1.module(var_27, var_0)
    assert var_28 == 'THIRDPARTY'
    var_29 = 'numpy.array'
    var_30 = module_1.module(var_29, var_0)
    assert var_30 == 'SCIENTIFIC'
    var_31 = module_0.Config()
    var_32 = 'any_module'
    var_33 = module_1.module(var_32, var_31)
    var_34 = 'test_caching'
    var_35 = module_1.module(var_34, var_0)
    var_36 = module_1.module(var_34, var_0)



# Parsed testcases at query #19
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_separate'
    var_2 = 'test_separate.module'
    var_3 = module_1.module(var_2, var_0)
    assert var_3 == 'test_separate'
    var_4 = module_1.module(var_1, var_0)
    assert var_4 == 'test_separate'
    var_5 = '.local_module'
    var_6 = module_1.module(var_5)
    assert var_6 == 'LOCALFOLDER'
    var_7 = '.local.module'
    var_8 = module_1.module(var_7)
    assert var_8 == 'LOCALFOLDER'
    var_9 = '^django\\.'
    var_10 = 'THIRDPARTY'
    var_11 = (var_9, var_10)
    var_12 = 'django.app'
    var_13 = module_1.module(var_12, var_0)
    assert var_13 == 'THIRDPARTY'
    var_14 = 'django'
    var_15 = module_1.module(var_14, var_0)
    assert var_15 == 'THIRDPARTY'
    var_16 = 'unknown_module'
    var_17 = module_1.module(var_16, var_0)
    assert var_17 == 'STDLIB'
    var_18 = module_0.Config()
    var_19 = 'some_random_module'
    var_20 = module_1.module(var_19, var_18)
    assert var_20 == 'CUSTOM'
    var_21 = 'special'
    var_22 = 'special.deeply.nested.module'
    var_23 = module_1.module(var_22, var_0)
    assert var_23 == 'special'
    var_24 = 'test*'
    var_25 = 'test_module'
    var_26 = module_1.module(var_25, var_0)
    assert var_26 == 'test*'
    var_27 = 'testing'
    var_28 = module_1.module(var_27, var_0)
    assert var_28 == 'test*'
    var_29 = 'local'
    var_30 = '.local'
    var_31 = module_1.module(var_30, var_0)
    assert var_31 == 'LOCALFOLDER'
    var_32 = 'cached_module'
    var_33 = module_1.module(var_32, var_0)
    var_34 = module_1.module(var_32, var_0)



# Parsed testcases at query #20
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = set()
    var_4 = False
    var_5 = 'THIRDPARTY'
    var_6 = 'FIRSTPARTY'
    var_7 = 'LOCALFOLDER'
    var_8 = [var_6, var_7, var_5]
    var_9 = 'py'
    var_10 = [var_9]
    var_11 = frozenset(var_10)
    var_12 = module_0.Config()
    var_13 = 'os'
    var_14 = module_1.module(var_13, var_12)
    assert var_14 == 'THIRDPARTY'
    var_15 = 'test'
    var_16 = 'test.module'
    var_17 = module_1.module(var_16, var_12)
    assert var_17 == 'test'
    var_18 = '.local'
    var_19 = module_1.module(var_18, var_12)
    assert var_19 == 'LOCALFOLDER'
    var_20 = '^django'
    var_21 = 'DJANGO'
    var_22 = 'django.contrib'
    var_23 = module_1.module(var_22, var_12)
    assert var_23 == 'DJANGO'
    var_24 = '/test/src'
    var_25 = 'mymodule'
    var_26 = module_1.module(var_25, var_12)
    assert var_26 == 'FIRSTPARTY'
    var_27 = 'STDLIB'
    var_28 = 'unknown'
    var_29 = module_1.module(var_28, var_12)
    assert var_29 == 'STDLIB'



# Parsed testcases at query #21
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
    var_2 = 'collections'
    var_3 = module_0.module(var_2)
    assert var_3 == 'STDLIB'
    var_4 = 'pytest'
    var_5 = module_0.module(var_4)
    assert var_5 == 'THIRDPARTY'
    var_6 = '.local_module'
    var_7 = module_0.module(var_6)
    assert var_7 == 'LOCALFOLDER'
    var_8 = '.subpackage.module'
    var_9 = module_0.module(var_8)
    assert var_9 == 'LOCALFOLDER'
    var_10 = 'test*'
    var_11 = [var_10]
    var_12 = module_1.Config()
    var_13 = 'test_module'
    var_14 = module_0.module(var_13, var_12)
    assert var_14 == 'test_module'
    var_15 = 'test.utils'
    var_16 = module_0.module(var_15, var_12)
    assert var_16 == 'test.utils'
    var_17 = '^django\\.'
    var_18 = 'DJANGO'
    var_19 = (var_17, var_18)
    var_20 = [var_19]
    var_21 = module_1.Config()
    var_22 = 'django.core'
    var_23 = module_0.module(var_22, var_21)
    assert var_23 == 'DJANGO'
    var_24 = 'django.utils'
    var_25 = module_0.module(var_24, var_21)
    assert var_25 == 'DJANGO'
    var_26 = '/test/src'
    var_27 = 'my_module'
    var_28 = module_0.module(var_27, var_21)
    assert var_28 == 'FIRSTPARTY'
    var_29 = 'CUSTOM'
    var_30 = module_1.Config()
    var_31 = 'unknown_module'
    var_32 = module_0.module(var_31, var_30)
    assert var_32 == 'CUSTOM'
    var_33 = 'my_namespace'
    var_34 = [var_33]
    var_35 = True
    var_36 = 'FIRSTPARTY'
    var_37 = 'reason'
    var_38 = 'my_namespace.subpackage'
    var_39 = module_0.module(var_38, var_30)
    assert var_39 == 'FIRSTPARTY'
    var_40 = ''
    var_41 = module_0.module(var_40)
    assert var_41 == 'STDLIB'
    var_42 = module_1.Config()
    var_43 = module_0.module(var_36, var_42)
    var_44 = module_0.module(var_36, var_42)



# Parsed testcases at query #22
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'test_module'
    var_4 = [var_3]
    var_5 = module_0.Config()
    var_6 = module_1.module(var_3, var_5)
    assert var_6 == 'test_module'
    var_7 = '.local_module'
    var_8 = module_1.module(var_7, var_5)
    assert var_8 == 'LOCALFOLDER'
    var_9 = '^django\\.'
    var_10 = 'THIRDPARTY'
    var_11 = (var_9, var_10)
    var_12 = [var_11]
    var_13 = module_0.Config()
    var_14 = 'django.test'
    var_15 = module_1.module(var_14, var_13)
    assert var_15 == 'THIRDPARTY'
    var_16 = 'CUSTOM'
    var_17 = module_0.Config()
    var_18 = 'unknown_module'
    var_19 = module_1.module(var_18, var_17)
    assert var_19 == 'CUSTOM'
    var_20 = module_0.Config()
    var_21 = module_1.module(var_18, var_20)
    assert var_21 == 'THIRDPARTY'
    var_22 = 'exact'
    var_23 = [var_22]
    var_24 = module_0.Config()
    var_25 = module_1.module(var_22, var_24)
    assert var_25 == 'exact'
    var_26 = 'test*'
    var_27 = [var_26]
    var_28 = module_0.Config()
    var_29 = module_1.module(var_3, var_28)
    assert var_29 == 'test*'
    var_30 = [var_26]
    var_31 = module_0.Config()
    var_32 = '.test_module'
    var_33 = module_1.module(var_32, var_31)
    assert var_33 == 'test*'
    var_34 = 'specific'
    var_35 = [var_34]
    var_36 = module_0.Config()
    var_37 = 'other'
    var_38 = module_1.module(var_37, var_36)
    assert var_38 == 'THIRDPARTY'



# Parsed testcases at query #23
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
    var_2 = 'sys'
    var_3 = module_0.module(var_2)
    assert var_3 == 'STDLIB'
    var_4 = 'collections'
    var_5 = module_0.module(var_4)
    assert var_5 == 'STDLIB'
    var_6 = 'pytest'
    var_7 = module_0.module(var_6)
    assert var_7 == 'THIRDPARTY'
    var_8 = 'numpy'
    var_9 = module_0.module(var_8)
    assert var_9 == 'THIRDPARTY'
    var_10 = '.local_module'
    var_11 = module_0.module(var_10)
    assert var_11 == 'LOCALFOLDER'
    var_12 = '.subpackage.module'
    var_13 = module_0.module(var_12)
    assert var_13 == 'LOCALFOLDER'
    var_14 = 'test*'
    var_15 = [var_14]
    var_16 = module_1.Config()
    var_17 = 'test_module'
    var_18 = module_0.module(var_17, var_16)
    assert var_18 == 'test'
    var_19 = 'test.utils'
    var_20 = module_0.module(var_19, var_16)
    assert var_20 == 'test'
    var_21 = '^myapp\\.*'
    var_22 = 'MYAPP'
    var_23 = 'STDLIB'
    var_24 = 'THIRDPARTY'
    var_25 = 'FIRSTPARTY'
    var_26 = 'LOCALFOLDER'
    var_27 = [var_23, var_24, var_25, var_22, var_26]
    var_28 = 'myapp.utils'
    var_29 = module_0.module(var_28, var_16)
    assert var_29 == 'MYAPP'
    var_30 = 'myapp.core.models'
    var_31 = module_0.module(var_30, var_16)
    assert var_31 == 'MYAPP'
    var_32 = 'CUSTOM'
    var_33 = module_1.Config()
    var_34 = 'unknown_module'
    var_35 = module_0.module(var_34, var_33)
    assert var_35 == 'CUSTOM'
    var_36 = '/test/src'
    var_37 = 'mynamespace'
    var_38 = [var_37]
    var_39 = True
    var_40 = ''
    var_41 = module_0.module(var_40)
    assert var_41 == 'STDLIB'
    var_42 = 'deeply.nested.module.name'
    var_43 = module_0.module(var_42)
    assert var_43 == 'THIRDPARTY'
    var_44 = 'special*'
    var_45 = [var_44]
    var_46 = '^special\\.*'
    var_47 = 'SPECIAL'
    var_48 = [var_23, var_24, var_25, var_47, var_26]
    var_49 = 'special_module'
    var_50 = module_0.module(var_49, var_33)
    assert var_50 == 'special'



# Parsed testcases at query #24
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
    var_2 = 'collections'
    var_3 = module_0.module(var_2)
    assert var_3 == 'STDLIB'
    var_4 = 'pytest'
    var_5 = module_0.module(var_4)
    assert var_5 == 'THIRDPARTY'
    var_6 = '.local_module'
    var_7 = module_0.module(var_6)
    assert var_7 == 'LOCALFOLDER'
    var_8 = '.subpackage.module'
    var_9 = module_0.module(var_8)
    assert var_9 == 'LOCALFOLDER'
    var_10 = 'test_module'
    var_11 = [var_10]
    var_12 = module_1.Config()
    var_13 = module_0.module(var_10, var_12)
    assert var_13 == 'test_module'
    var_14 = 'test_module.sub'
    var_15 = module_0.module(var_14, var_12)
    assert var_15 == 'test_module'
    var_16 = '^django\\.'
    var_17 = 'DJANGO'
    var_18 = (var_16, var_17)
    var_19 = [var_18]
    var_20 = module_1.Config()
    var_21 = 'django.apps'
    var_22 = module_0.module(var_21, var_20)
    assert var_22 == 'DJANGO'
    var_23 = 'django.contrib.auth'
    var_24 = module_0.module(var_23, var_20)
    assert var_24 == 'DJANGO'
    var_25 = '/test/src'
    var_26 = 'FIRSTPARTY'
    var_27 = 'reason'
    var_28 = 'mymodule'
    var_29 = module_0.module(var_28, var_20)
    assert var_29 == 'FIRSTPARTY'
    var_30 = 'CUSTOM'
    var_31 = module_1.Config()
    var_32 = 'unknown_module'
    var_33 = module_0.module(var_32, var_31)
    assert var_33 == 'CUSTOM'
    var_34 = module_1.Config()
    var_35 = module_0.module(var_26, var_34)
    var_36 = module_0.module(var_26, var_34)
    var_37 = ''
    var_38 = module_0.module(var_37)
    assert var_38 == 'STDLIB'
    var_39 = None
    var_40 = module_0.module(var_26, var_39)
    assert var_40 == 'STDLIB'



# Parsed testcases at query #25
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'collections'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'STDLIB'
    var_5 = 'django'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'THIRDPARTY'
    var_7 = 'requests'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'THIRDPARTY'
    var_9 = '.local_module'
    var_10 = module_1.module(var_9, var_0)
    assert var_10 == 'LOCALFOLDER'
    var_11 = 'myproject'
    var_12 = module_1.module(var_11, var_0)
    assert var_12 == 'FIRSTPARTY'
    var_13 = 'test'
    var_14 = 'test.module'
    var_15 = module_1.module(var_14, var_0)
    assert var_15 == 'test'
    var_16 = '^myapp\\.*'
    var_17 = 'MYAPP'
    var_18 = 'STDLIB'
    var_19 = 'THIRDPARTY'
    var_20 = 'FIRSTPARTY'
    var_21 = 'LOCALFOLDER'
    var_22 = 'myapp.utils'
    var_23 = module_1.module(var_22, var_0)
    assert var_23 == 'MYAPP'
    var_24 = 'CUSTOM'
    var_25 = 'unknown.module'
    var_26 = module_1.module(var_25, var_0)
    assert var_26 == 'CUSTOM'
    var_27 = '/fake/path'
    var_28 = 'fake_module'
    var_29 = module_1.module(var_28, var_0)
    assert var_29 == 'FIRSTPARTY'
    var_30 = 'mynamespace'
    var_31 = 'mynamespace.submodule'
    var_32 = module_1.module(var_31, var_0)
    assert var_32 == 'FIRSTPARTY'



# Parsed testcases at query #26
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
    var_2 = 'collections'
    var_3 = module_0.module(var_2)
    assert var_3 == 'STDLIB'
    var_4 = 'django'
    var_5 = module_0.module(var_4)
    assert var_5 == 'THIRDPARTY'
    var_6 = 'requests'
    var_7 = module_0.module(var_6)
    assert var_7 == 'THIRDPARTY'
    var_8 = 'test_module'
    var_9 = [var_8]
    var_10 = module_1.Config()
    var_11 = module_0.module(var_8, var_10)
    assert var_11 == 'test_module'
    var_12 = module_1.Config()
    var_13 = '.local_module'
    var_14 = module_0.module(var_13, var_12)
    assert var_14 == 'LOCALFOLDER'
    var_15 = '^myapp\\.'
    var_16 = 'FIRSTPARTY'
    var_17 = (var_15, var_16)
    var_18 = [var_17]
    var_19 = module_1.Config()
    var_20 = 'myapp.models'
    var_21 = module_0.module(var_20, var_19)
    assert var_21 == 'FIRSTPARTY'
    var_22 = 'CUSTOM'
    var_23 = module_1.Config()
    var_24 = 'unknown_module'
    var_25 = module_0.module(var_24, var_23)
    assert var_25 == 'CUSTOM'
    var_26 = '^myapp\\.submodule\\.'
    var_27 = (var_26, var_16)
    var_28 = [var_27]
    var_29 = module_1.Config()
    var_30 = 'myapp.submodule.utils'
    var_31 = module_0.module(var_30, var_29)
    assert var_31 == 'FIRSTPARTY'
    var_32 = 'myapp.other.utils'
    var_33 = module_0.module(var_32, var_29)
    assert var_33 == 'THIRDPARTY'
    var_34 = 'exact'
    var_35 = [var_34]
    var_36 = module_1.Config()
    var_37 = module_0.module(var_34, var_36)
    assert var_37 == 'exact'
    var_38 = 'test*'
    var_39 = [var_38]
    var_40 = module_1.Config()
    var_41 = module_0.module(var_8, var_40)
    assert var_41 == 'test*'
    var_42 = 'test'
    var_43 = module_0.module(var_42, var_40)
    assert var_43 == 'test*'
    var_44 = [var_4]
    var_45 = '^django\\.'
    var_46 = 'THIRDPARTY'
    var_47 = (var_45, var_46)
    var_48 = [var_47]
    var_49 = module_1.Config()
    var_50 = module_0.module(var_4, var_49)
    assert var_50 == 'django'
    var_51 = 'django.contrib'
    var_52 = module_0.module(var_51, var_49)
    assert var_52 == 'THIRDPARTY'
    var_53 = '^\\.'
    var_54 = (var_53, var_46)
    var_55 = [var_54]
    var_56 = module_1.Config()
    var_57 = '.local'
    var_58 = module_0.module(var_57, var_56)
    assert var_58 == 'LOCALFOLDER'
    var_59 = module_1.Config()
    var_60 = ''
    var_61 = module_0.module(var_60, var_59)
    assert var_61 == 'THIRDPARTY'
    var_62 = module_1.Config()
    var_63 = 'very.deeply.nested.module'
    var_64 = module_0.module(var_63, var_62)
    assert var_64 == 'THIRDPARTY'



# Parsed testcases at query #27
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'some_module'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
    var_2 = 'test_module'
    var_3 = [var_2]
    var_4 = module_1.Config()
    var_5 = module_0.module(var_2, var_4)
    assert var_5 == 'test_module'
    var_6 = module_1.Config()
    var_7 = '.local_module'
    var_8 = module_0.module(var_7, var_6)
    assert var_8 == 'LOCALFOLDER'
    var_9 = '^django\\.'
    var_10 = 'THIRDPARTY'
    var_11 = (var_9, var_10)
    var_12 = [var_11]
    var_13 = module_1.Config()
    var_14 = 'django.apps'
    var_15 = module_0.module(var_14, var_13)
    assert var_15 == 'THIRDPARTY'
    var_16 = '/test/path'
    var_17 = 'FIRSTPARTY'
    var_18 = 'reason'
    var_19 = 'my_module'
    var_20 = module_0.module(var_19, var_13)
    assert var_20 == 'FIRSTPARTY'
    var_21 = module_1.Config()
    var_22 = 'unknown_module'
    var_23 = module_0.module(var_22, var_21)
    assert var_23 == 'STDLIB'



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
    var_5 = 'collections'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'STDLIB'
    var_7 = 'django'
    var_8 = [var_7]
    var_9 = module_0.Config()
    var_10 = module_1.module(var_7, var_9)
    assert var_10 == 'django'
    var_11 = 'django.contrib'
    var_12 = module_1.module(var_11, var_9)
    assert var_12 == 'django'
    var_13 = module_0.Config()
    var_14 = '.local_module'
    var_15 = module_1.module(var_14, var_13)
    assert var_15 == 'LOCALFOLDER'
    var_16 = '.subpackage.module'
    var_17 = module_1.module(var_16, var_13)
    assert var_17 == 'LOCALFOLDER'
    var_18 = '^requests$'
    var_19 = 'THIRDPARTY'
    var_20 = (var_18, var_19)
    var_21 = [var_20]
    var_22 = module_0.Config()
    var_23 = 'requests'
    var_24 = module_1.module(var_23, var_22)
    assert var_24 == 'THIRDPARTY'
    var_25 = 'requests.models'
    var_26 = module_1.module(var_25, var_22)
    assert var_26 == 'THIRDPARTY'
    var_27 = module_0.Config()
    var_28 = 'mymodule.py'
    var_29 = ''
    var_30 = 'mymodule'
    var_31 = module_1.module(var_30, var_27)
    assert var_31 == 'FIRSTPARTY'
    var_32 = module_0.Config()
    var_33 = 'some_unknown_module'
    var_34 = module_1.module(var_33, var_32)
    var_35 = 'test*'
    var_36 = [var_35]
    var_37 = module_0.Config()
    var_38 = 'test_module'
    var_39 = module_1.module(var_38, var_37)
    assert var_39 == 'test*'
    var_40 = 'testing'
    var_41 = module_1.module(var_40, var_37)
    assert var_41 == 'test*'
    var_42 = 'exact'
    var_43 = [var_42]
    var_44 = module_0.Config()
    var_45 = module_1.module(var_42, var_44)
    assert var_45 == 'exact'
    var_46 = 'exact.sub'
    var_47 = module_1.module(var_46, var_44)
    assert var_47 == 'exact'
    var_48 = True
    var_49 = module_0.Config()
    var_50 = 'mynamespace'
    var_51 = 'subpackage'
    var_52 = '__init__.py'
    var_53 = ''
    var_54 = 'mynamespace.subpackage'
    var_55 = module_1.module(var_54, var_49)
    assert var_55 == 'FIRSTPARTY'
    var_56 = 'mypackage'
    var_57 = var_1 / var_56
    var_58 = [var_57]
    var_59 = module_0.Config()
    var_60 = module_1.module(var_56, var_59)
    assert var_60 == 'FIRSTPARTY'
    var_61 = module_0.Config()
    var_62 = module_1.module(var_1, var_61)
    var_63 = module_1.module(var_1, var_61)



# Parsed testcases at query #29
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
    var_2 = 'sys'
    var_3 = module_0.module(var_2)
    assert var_3 == 'STDLIB'
    var_4 = 'collections'
    var_5 = module_0.module(var_4)
    assert var_5 == 'STDLIB'
    var_6 = 'pytest'
    var_7 = module_0.module(var_6)
    assert var_7 == 'THIRDPARTY'
    var_8 = 'numpy'
    var_9 = module_0.module(var_8)
    assert var_9 == 'THIRDPARTY'
    var_10 = '.local_module'
    var_11 = module_0.module(var_10)
    assert var_11 == 'LOCALFOLDER'
    var_12 = '.subpackage.module'
    var_13 = module_0.module(var_12)
    assert var_13 == 'LOCALFOLDER'
    var_14 = 'test*'
    var_15 = [var_14]
    var_16 = module_1.Config()
    var_17 = 'test_module'
    var_18 = module_0.module(var_17, var_16)
    assert var_18 == 'test_module'
    var_19 = 'test.subpackage'
    var_20 = module_0.module(var_19, var_16)
    assert var_20 == 'test'
    var_21 = '^myapp\\.*'
    var_22 = 'MYAPP'
    var_23 = 'STDLIB'
    var_24 = 'THIRDPARTY'
    var_25 = 'FIRSTPARTY'
    var_26 = [var_23, var_24, var_25, var_22]
    var_27 = 'myapp.utils'
    var_28 = module_0.module(var_27, var_16)
    assert var_28 == 'MYAPP'
    var_29 = 'myapp.models.user'
    var_30 = module_0.module(var_29, var_16)
    assert var_30 == 'MYAPP'
    var_31 = 'CUSTOM'
    var_32 = module_1.Config()
    var_33 = 'unknown_module'
    var_34 = module_0.module(var_33, var_32)
    assert var_34 == 'CUSTOM'
    var_35 = '/test/src'
    var_36 = 'mynamespace'
    var_37 = [var_36]
    var_38 = True
    var_39 = 'mynamespace.subpackage'
    var_40 = module_0.module(var_39, var_32)
    assert var_40 == 'FIRSTPARTY'
    var_41 = 'mymodule'
    var_42 = module_0.module(var_41, var_32)
    assert var_42 == 'FIRSTPARTY'
    var_43 = 'deeply.nested.module.name'
    var_44 = module_0.module(var_43)
    assert var_44 == 'THIRDPARTY'
    var_45 = 'exact_match'
    var_46 = [var_45]
    var_47 = module_1.Config()
    var_48 = module_0.module(var_45, var_47)
    assert var_48 == 'exact_match'
    var_49 = 'exact_match_extra'
    var_50 = module_0.module(var_49, var_47)
    assert var_50 == 'exact_match'
    var_51 = [var_14]
    var_52 = module_1.Config()
    var_53 = 'testing'
    var_54 = module_0.module(var_53, var_52)
    assert var_54 == 'testing'
    var_55 = 'test.unit'
    var_56 = module_0.module(var_55, var_52)
    assert var_56 == 'test'
    var_57 = 'local*'
    var_58 = [var_57]
    var_59 = module_1.Config()
    var_60 = module_0.module(var_10, var_59)
    assert var_60 == 'LOCALFOLDER'
    var_61 = ''
    var_62 = module_0.module(var_61, var_59)



# Parsed testcases at query #30
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
    var_2 = 'collections'
    assert var_2 == 'FIRSTPARTY'
    var_3 = module_0.module(var_2)
    assert var_3 == 'STDLIB'
    var_4 = 'pytest'
    var_5 = module_0.module(var_4)
    assert var_5 == 'THIRDPARTY'
    var_6 = '.local_module'
    var_7 = module_0.module(var_6)
    assert var_7 == 'LOCALFOLDER'
    var_8 = '.subpackage.module'
    var_9 = module_0.module(var_8)
    assert var_9 == 'LOCALFOLDER'
    var_10 = 'requests'
    var_11 = [var_10]
    var_12 = 'myproject'
    var_13 = [var_12]
    var_14 = 'THIRDPARTY'
    var_15 = module_1.Config()
    var_16 = module_0.module(var_10, var_15)
    assert var_16 == 'THIRDPARTY'
    var_17 = module_0.module(var_12, var_15)
    assert var_17 == 'FIRSTPARTY'
    var_18 = 'unknown_lib'
    var_19 = module_0.module(var_18, var_15)
    assert var_19 == 'THIRDPARTY'
    var_20 = 'special'
    var_21 = [var_20]
    var_22 = module_1.Config()
    var_23 = 'special.module'
    var_24 = module_0.module(var_23, var_22)
    assert var_24 == 'special'
    var_25 = module_0.module(var_20, var_22)
    assert var_25 == 'special'
    var_26 = '^google\\.*'
    var_27 = 'GOOGLE'
    var_28 = 'STDLIB'
    var_29 = 'FIRSTPARTY'
    var_30 = 'LOCALFOLDER'
    var_31 = [var_28, var_27, var_14, var_29, var_30]
    var_32 = 'google.cloud.storage'
    var_33 = 'google.auth'
    var_34 = 'mynamespace'
    var_35 = [var_34]
    var_36 = '/fake/src'
    var_37 = 'isort.utils'
    var_38 = True
    var_39 = 'mynamespace.subpackage'
    var_40 = module_0.module(var_38)
    assert var_40 == 'STDLIB'
    var_41 = module_0.module(var_2)
    assert var_41 == 'STDLIB'
    var_42 = ''
    var_43 = 'CUSTOM'
    var_44 = module_1.Config()
    var_45 = module_0.module(var_42, var_44)
    assert var_45 == 'CUSTOM'
    var_46 = 'very.long.module.path.name'
    var_47 = module_1.Config()
    var_48 = module_0.module(var_46, var_47)
    assert var_48 == 'THIRDPARTY'



# Parsed testcases at query #31
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'some_module'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'test_module'
    var_4 = [var_3]
    var_5 = module_0.Config()
    var_6 = module_1.module(var_3, var_5)
    assert var_6 == 'test_module'
    var_7 = 'test_*'
    var_8 = [var_7]
    var_9 = module_0.Config()
    var_10 = module_1.module(var_3, var_9)
    assert var_10 == 'test_*'
    var_11 = [var_3]
    var_12 = module_0.Config()
    var_13 = '.test_module'
    var_14 = module_1.module(var_13, var_12)
    assert var_14 == 'test_module'
    var_15 = module_0.Config()
    var_16 = '.local_module'
    var_17 = module_1.module(var_16, var_15)
    assert var_17 == 'LOCALFOLDER'
    var_18 = '^django\\.'
    var_19 = 'THIRDPARTY'
    var_20 = (var_18, var_19)
    var_21 = [var_20]
    var_22 = module_0.Config()
    var_23 = 'django.apps'
    var_24 = module_1.module(var_23, var_22)
    assert var_24 == 'THIRDPARTY'
    var_25 = (var_18, var_19)
    var_26 = '^django\\.apps\\.'
    var_27 = 'FIRSTPARTY'
    var_28 = (var_26, var_27)
    var_29 = [var_25, var_28]
    var_30 = module_0.Config()
    var_31 = 'django.apps.config'
    var_32 = module_1.module(var_31, var_30)
    assert var_32 == 'FIRSTPARTY'
    var_33 = module_0.Config()
    var_34 = 'unknown_module'
    var_35 = module_1.module(var_34, var_33)
    assert var_35 == 'THIRDPARTY'
    var_36 = module_0.Config()
    var_37 = ''
    var_38 = module_1.module(var_37, var_36)
    assert var_38 == 'STDLIB'
    var_39 = module_0.Config()
    var_40 = 'very.deeply.nested.module'
    var_41 = module_1.module(var_40, var_39)
    assert var_41 == 'STDLIB'
    var_42 = 'exact_match'
    var_43 = [var_42]
    var_44 = module_0.Config()
    var_45 = module_1.module(var_42, var_44)
    assert var_45 == 'exact_match'
    var_46 = 'exact_match.submodule'
    var_47 = module_1.module(var_46, var_44)
    assert var_47 == 'exact_match'
    var_48 = 'no_wildcard'
    var_49 = [var_48]
    var_50 = module_0.Config()
    var_51 = module_1.module(var_48, var_50)
    assert var_51 == 'no_wildcard'
    var_52 = 'no_wildcard.extra'
    var_53 = module_1.module(var_52, var_50)
    assert var_53 == 'no_wildcard'



# Parsed testcases at query #32
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
    var_2 = 'sys'
    var_3 = module_0.module(var_2)
    assert var_3 == 'STDLIB'
    var_4 = 'django'
    var_5 = [var_4]
    var_6 = module_1.Config()
    var_7 = module_0.module(var_4, var_6)
    assert var_7 == 'django'
    var_8 = 'django.contrib'
    var_9 = module_0.module(var_8, var_6)
    assert var_9 == 'django'
    var_10 = '.local_module'
    var_11 = module_0.module(var_10)
    assert var_11 == 'LOCALFOLDER'
    var_12 = '..parent_module'
    var_13 = module_0.module(var_12)
    assert var_13 == 'LOCALFOLDER'
    var_14 = '^google\\.'
    var_15 = 'THIRDPARTY'
    var_16 = '^requests$'
    var_17 = 'STDLIB'
    var_18 = 'FIRSTPARTY'
    var_19 = 'LOCALFOLDER'
    var_20 = [var_17, var_15, var_18, var_19]
    var_21 = 'google.cloud.storage'
    var_22 = module_0.module(var_21, var_6)
    assert var_22 == 'THIRDPARTY'
    var_23 = 'requests'
    var_24 = module_0.module(var_23, var_6)
    assert var_24 == 'THIRDPARTY'
    var_25 = '/test/src'
    var_26 = 'mymodule'
    var_27 = lambda path: path.name == var_26
    var_28 = 'mypackage'
    var_29 = lambda path: path.name == var_28
    var_30 = False
    var_31 = lambda src_path, module_name: var_30
    var_32 = module_0.module(var_26, var_6)
    var_33 = 'mynamespace'
    var_34 = [var_33]
    var_35 = True
    var_36 = module_1.Config()
    var_37 = module_0.module(var_26, var_36)
    var_38 = module_0.module(var_26, var_36)
    var_39 = ''
    var_40 = module_0.module(var_39)
    assert var_40 == 'STDLIB'
    var_41 = 'a'
    var_42 = 1000
    var_43 = var_41 * var_42
    var_44 = module_0.module(var_43)
    assert var_44 == 'STDLIB'



# Parsed testcases at query #33
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = set()
    var_4 = False
    var_5 = 'THIRDPARTY'
    var_6 = 'FIRSTPARTY'
    var_7 = 'LOCALFOLDER'
    var_8 = [var_6, var_7, var_5]
    var_9 = 'py'
    var_10 = [var_9]
    var_11 = frozenset(var_10)
    var_12 = module_0.Config()
    var_13 = 'os'
    var_14 = module_1.module(var_13, var_12)
    assert var_14 == 'THIRDPARTY'
    var_15 = 'unknown'
    var_16 = module_1.module(var_15, var_12)
    assert var_16 == 'FIRSTPARTY'
    var_17 = 'test*'
    var_18 = 'test_module'
    var_19 = module_1.module(var_18, var_12)
    assert var_19 == 'test'
    var_20 = '.local'
    var_21 = module_1.module(var_20, var_12)
    assert var_21 == 'LOCALFOLDER'
    var_22 = '^django'
    var_23 = 'DJANGO'
    var_24 = 'django.contrib'
    var_25 = module_1.module(var_24, var_12)
    assert var_25 == 'DJANGO'
    var_26 = '/fake/src'
    var_27 = 'mymodule'
    var_28 = module_1.module(var_27, var_12)
    assert var_28 == 'FIRSTPARTY'
    var_29 = 'mynamespace'
    var_30 = 'mynamespace.sub'
    var_31 = module_1.module(var_30, var_12)
    assert var_31 == 'FIRSTPARTY'
    var_32 = 'auto.sub'
    var_33 = module_1.module(var_32, var_12)
    assert var_33 == 'FIRSTPARTY'
    var_34 = 'src'
    var_35 = module_1.module(var_34, var_12)
    assert var_35 == 'FIRSTPARTY'



# Parsed testcases at query #34
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'collections'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'STDLIB'
    var_5 = 'django'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'THIRDPARTY'
    var_7 = 'requests'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'THIRDPARTY'
    var_9 = '.local_module'
    var_10 = module_1.module(var_9, var_0)
    assert var_10 == 'LOCALFOLDER'
    var_11 = '..parent_module'
    var_12 = module_1.module(var_11, var_0)
    assert var_12 == 'LOCALFOLDER'
    var_13 = 'my_project'
    var_14 = module_1.module(var_13, var_0)
    assert var_14 == 'FIRSTPARTY'
    var_15 = 'test'
    var_16 = module_1.module(var_15, var_0)
    assert var_16 == 'test'
    var_17 = 'test.module'
    var_18 = module_1.module(var_17, var_0)
    assert var_18 == 'test'
    var_19 = '^custom\\.'
    var_20 = 'CUSTOM'
    var_21 = (var_19, var_20)
    var_22 = 'custom.module'
    var_23 = module_1.module(var_22, var_0)
    assert var_23 == 'CUSTOM'
    var_24 = 'custom.sub.module'
    var_25 = module_1.module(var_24, var_0)
    assert var_25 == 'CUSTOM'
    var_26 = 'unknown'
    var_27 = module_1.module(var_26, var_0)
    assert var_27 == 'DEFAULT'
    var_28 = '/fake/path'
    var_29 = 'fake_module'
    var_30 = module_1.module(var_29, var_0)
    assert var_30 == 'FIRSTPARTY'
    var_31 = 'arbitrary'
    var_32 = module_1.module(var_31, var_0)
    assert var_32 == 'THIRDPARTY'
    var_33 = 'namespace'
    var_34 = 'namespace.sub'
    var_35 = module_1.module(var_34, var_0)
    assert var_35 == 'FIRSTPARTY'
    var_36 = 'auto_namespace.sub'
    var_37 = module_1.module(var_36, var_0)
    assert var_37 == 'FIRSTPARTY'



# Parsed testcases at query #35
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'collections'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'STDLIB'
    var_5 = 'django'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'THIRDPARTY'
    var_7 = 'requests'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'THIRDPARTY'
    var_9 = '.local_module'
    var_10 = module_1.module(var_9, var_0)
    assert var_10 == 'LOCALFOLDER'
    var_11 = '..parent_module'
    var_12 = module_1.module(var_11, var_0)
    assert var_12 == 'LOCALFOLDER'
    var_13 = 'my_project'
    var_14 = module_1.module(var_13, var_0)
    assert var_14 == 'FIRSTPARTY'
    var_15 = 'test'
    var_16 = 'test.module'
    var_17 = module_1.module(var_16, var_0)
    assert var_17 == 'test'
    var_18 = '^custom\\.'
    var_19 = 'CUSTOM'
    var_20 = (var_18, var_19)
    var_21 = 'STDLIB'
    var_22 = 'THIRDPARTY'
    var_23 = 'FIRSTPARTY'
    var_24 = 'LOCALFOLDER'
    var_25 = 'custom.package'
    var_26 = module_1.module(var_25, var_0)
    assert var_26 == 'CUSTOM'
    var_27 = 'DEFAULT'
    var_28 = 'unknown'
    var_29 = module_1.module(var_28, var_0)
    assert var_29 == 'DEFAULT'



# Parsed testcases at query #36
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'some_module'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
    var_2 = 'test_module'
    var_3 = [var_2]
    var_4 = module_1.Config()
    var_5 = module_0.module(var_2, var_4)
    assert var_5 == 'test_module'
    var_6 = '.local_module'
    var_7 = module_0.module(var_6)
    assert var_7 == 'LOCALFOLDER'
    var_8 = '^django\\.'
    var_9 = 'THIRDPARTY'
    var_10 = (var_8, var_9)
    var_11 = [var_10]
    var_12 = module_1.Config()
    var_13 = 'django.apps'
    var_14 = module_0.module(var_13, var_12)
    assert var_14 == 'THIRDPARTY'
    var_15 = '/test/path'
    var_16 = 'my_module'
    var_17 = module_0.module(var_16, var_12)
    assert var_17 == 'FIRSTPARTY'
    var_18 = 'my_namespace'
    var_19 = [var_18]
    var_20 = True
    var_21 = 'my_namespace.submodule'
    var_22 = module_0.module(var_21, var_12)
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'parent.child'
    var_24 = module_0.module(var_23, var_12)
    assert var_24 == 'FIRSTPARTY'
    var_25 = 'CUSTOM'
    var_26 = module_1.Config()
    var_27 = 'unknown_module'
    var_28 = module_0.module(var_27, var_26)
    assert var_28 == 'CUSTOM'



# Parsed testcases at query #37
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
    var_5 = 'collections'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'STDLIB'
    var_7 = 'pytest'
    var_8 = 'requests'
    var_9 = [var_7, var_8]
    var_10 = module_0.Config()
    var_11 = module_1.module(var_7, var_10)
    assert var_11 == 'THIRDPARTY'
    var_12 = module_1.module(var_8, var_10)
    assert var_12 == 'THIRDPARTY'
    var_13 = 'myapp'
    var_14 = [var_13]
    var_15 = module_0.Config()
    var_16 = module_1.module(var_13, var_15)
    assert var_16 == 'FIRSTPARTY'
    var_17 = 'myapp.utils'
    var_18 = module_1.module(var_17, var_15)
    assert var_18 == 'FIRSTPARTY'
    var_19 = '.local_module'
    var_20 = module_1.module(var_19, var_15)
    assert var_20 == 'LOCALFOLDER'
    var_21 = '.subpackage.module'
    var_22 = module_1.module(var_21, var_15)
    assert var_22 == 'LOCALFOLDER'
    var_23 = 'special'
    var_24 = [var_23]
    var_25 = module_0.Config()
    var_26 = module_1.module(var_23, var_25)
    assert var_26 == 'special'
    var_27 = 'special.utils'
    var_28 = module_1.module(var_27, var_25)
    assert var_28 == 'special'
    var_29 = 'THIRDPARTY'
    var_30 = module_0.Config()
    var_31 = 'unknown_module'
    var_32 = module_1.module(var_31, var_30)
    assert var_32 == 'THIRDPARTY'
    var_33 = 'external'
    var_34 = [var_33]
    var_35 = 'internal'
    var_36 = [var_35]
    var_37 = 'separated'
    var_38 = [var_37]
    var_39 = module_0.Config()
    var_40 = module_1.module(var_33, var_39)
    assert var_40 == 'THIRDPARTY'
    var_41 = module_1.module(var_35, var_39)
    assert var_41 == 'FIRSTPARTY'
    var_42 = module_1.module(var_37, var_39)
    assert var_42 == 'separated'
    var_43 = 'other'
    var_44 = module_1.module(var_43, var_39)



