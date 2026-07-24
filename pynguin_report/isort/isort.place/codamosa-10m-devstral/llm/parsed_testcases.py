####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
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
    var_4 = 'test*'
    var_5 = [var_4]
    var_6 = module_1.Config()
    var_7 = 'test_module'
    var_8 = module_0.module(var_7, var_6)
    assert var_8 == 'test*'
    var_9 = '.local_module'
    var_10 = module_0.module(var_9)
    assert var_10 == 'LOCALFOLDER'
    var_11 = 'THIRDPARTY'
    var_12 = (var_4, var_11)
    var_13 = [var_12]
    var_14 = module_1.Config()
    var_15 = 'test_library'
    var_16 = module_0.module(var_15, var_14)
    assert var_16 == 'THIRDPARTY'
    var_17 = '/project/src'
    var_18 = 'project'
    var_19 = module_0.module(var_18, var_14)
    assert var_19 == 'FIRSTPARTY'
    var_20 = [var_18]
    var_21 = 'project.submodule'
    var_22 = module_0.module(var_21, var_14)
    assert var_22 == 'FIRSTPARTY'
    var_23 = True
    var_24 = module_0.module(var_21, var_14)
    assert var_24 == 'FIRSTPARTY'
    var_25 = module_1.Config()
    var_26 = 'unknown_module'
    var_27 = module_0.module(var_26, var_25)
    assert var_27 == 'THIRDPARTY'



# Parsed testcases at query #2
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
    var_10 = '.local'
    var_11 = module_0.module(var_10)
    assert var_11 == 'LOCALFOLDER'
    var_12 = '.local.module'
    var_13 = module_0.module(var_12)
    assert var_13 == 'LOCALFOLDER'
    var_14 = '^test_'
    var_15 = 'TESTS'
    var_16 = 'test_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_module.submodule'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/path/to/src'
    var_21 = 'src_module'
    var_22 = module_0.module(var_21, var_6)
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'src_module.submodule'
    var_24 = module_0.module(var_23, var_6)
    assert var_24 == 'FIRSTPARTY'
    var_25 = 'namespace'
    var_26 = [var_25]
    var_27 = 'namespace.module'
    var_28 = module_0.module(var_27, var_6)
    assert var_28 == 'FIRSTPARTY'



# Parsed testcases at query #3
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
    var_2 = 'test*'
    var_3 = [var_2]
    var_4 = module_1.Config()
    var_5 = 'test_module'
    var_6 = module_0.module(var_5, var_4)
    assert var_6 == 'test*'
    var_7 = '.local_module'
    var_8 = module_0.module(var_7)
    assert var_8 == 'LOCALFOLDER'
    var_9 = '^django'
    var_10 = 'DJANGO'
    var_11 = 'django.contrib'
    var_12 = module_0.module(var_11, var_4)
    assert var_12 == 'DJANGO'
    var_13 = '/path/to/src'
    var_14 = 'my_module'
    var_15 = module_0.module(var_14, var_4)
    assert var_15 == 'FIRSTPARTY'
    var_16 = 'my_namespace'
    var_17 = [var_16]
    var_18 = 'my_namespace.submodule'
    var_19 = module_0.module(var_18, var_4)
    assert var_19 == 'FIRSTPARTY'



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
    var_4 = '.local_module'
    var_5 = module_0.module(var_4)
    var_6 = '.another_local'
    var_7 = module_0.module(var_6)
    var_8 = 'django'
    var_9 = 'flask'
    var_10 = [var_8, var_9]
    var_11 = module_1.Config()
    var_12 = module_0.module(var_8, var_11)
    assert var_12 == 'django'
    var_13 = module_0.module(var_9, var_11)
    assert var_13 == 'flask'
    var_14 = 'django_ext'
    var_15 = module_0.module(var_14, var_11)
    assert var_15 == 'django'
    var_16 = 'flask_app'
    var_17 = module_0.module(var_16, var_11)
    assert var_17 == 'flask'
    var_18 = '^django.*'
    var_19 = 'DJANGO'
    var_20 = module_0.module(var_8, var_11)
    assert var_20 == 'DJANGO'
    var_21 = 'django.contrib'
    var_22 = module_0.module(var_21, var_11)
    assert var_22 == 'DJANGO'
    var_23 = 'src'
    var_24 = var_0 / var_23
    var_25 = 'my_package'
    var_26 = var_24 / var_25
    var_27 = var_24 / var_25
    var_28 = '__init__.py'
    var_29 = var_27 / var_28
    var_30 = '#'
    var_31 = [var_24]
    var_32 = module_1.Config()
    var_33 = module_0.module(var_25, var_32)
    var_34 = 'my_package.submodule'
    var_35 = module_0.module(var_34, var_32)
    var_36 = 'src'
    var_37 = var_0 / var_36
    var_38 = 'namespace'
    var_39 = var_37 / var_38
    var_40 = var_37 / var_38
    var_41 = 'module.py'
    var_42 = var_40 / var_41
    var_43 = '#'
    var_44 = [var_37]
    var_45 = [var_38]
    var_46 = module_1.Config()
    var_47 = 'namespace.module'
    var_48 = module_0.module(var_47, var_46)



# Parsed testcases at query #5
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    var_2 = '.local_module'
    var_3 = module_0.module(var_2)
    var_4 = 'test*'
    var_5 = [var_4]
    var_6 = module_1.Config()
    var_7 = 'test_module'
    var_8 = module_0.module(var_7, var_6)
    assert var_8 == 'test*'
    var_9 = '^django'
    var_10 = 'DJANGO'
    var_11 = 'django.contrib'
    var_12 = module_0.module(var_11, var_6)
    assert var_12 == 'DJANGO'
    var_13 = 'src'
    var_14 = var_0 / var_13
    var_15 = 'mymodule'
    var_16 = var_14 / var_15
    var_17 = [var_14]
    var_18 = module_1.Config()
    var_19 = module_0.module(var_15, var_18)
    var_20 = 'src'
    var_21 = var_0 / var_20
    var_22 = 'namespace'
    var_23 = var_21 / var_22
    var_24 = [var_21]
    var_25 = [var_22]
    var_26 = module_1.Config()
    var_27 = 'namespace.submodule'
    var_28 = module_0.module(var_27, var_26)
    var_29 = 'src'
    var_30 = var_0 / var_29
    var_31 = 'auto_namespace'
    var_32 = var_30 / var_31
    var_33 = [var_30]
    var_34 = True
    var_35 = module_1.Config()
    var_36 = 'auto_namespace.submodule'
    var_37 = module_0.module(var_36, var_35)



# Parsed testcases at query #6
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
    var_8 = '.local_module'
    var_9 = module_0.module(var_8)
    assert var_9 == 'LOCALFOLDER'
    var_10 = '^test_.*'
    var_11 = 'TESTS'
    var_12 = 'test_example'
    var_13 = module_0.module(var_12, var_6)
    assert var_13 == 'TESTS'
    var_14 = '/path/to/src'
    var_15 = 'my_module'
    var_16 = module_0.module(var_15, var_6)
    assert var_16 == 'FIRSTPARTY'
    var_17 = 'my_namespace'
    var_18 = [var_17]
    var_19 = 'my_namespace.submodule'
    var_20 = module_0.module(var_19, var_6)
    assert var_20 == 'FIRSTPARTY'
    var_21 = 'unknown_module'
    var_22 = module_0.module(var_21)
    assert var_22 == 'THIRDPARTY'



# Parsed testcases at query #7
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
    var_12 = '..parent_module'
    var_13 = module_0.module(var_12)
    var_14 = '^test_'
    var_15 = 'TESTS'
    var_16 = 'test_example'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_example.module'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/project/src'
    var_21 = 'project'
    var_22 = module_0.module(var_21, var_6)
    var_23 = 'project.module'
    var_24 = module_0.module(var_23, var_6)
    var_25 = [var_21]
    var_26 = 'project.submodule'
    var_27 = module_0.module(var_26, var_6)
    var_28 = True
    var_29 = module_0.module(var_26, var_6)
    var_30 = 'THIRDPARTY'
    var_31 = module_1.Config()
    var_32 = 'unknown_module'
    var_33 = module_0.module(var_32, var_31)
    assert var_33 == 'THIRDPARTY'



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
    var_8 = '.local_module'
    var_9 = module_0.module(var_8)
    assert var_9 == 'LOCALFOLDER'
    var_10 = '^test_.*'
    var_11 = 'TESTS'
    var_12 = 'test_example'
    var_13 = module_0.module(var_12, var_6)
    assert var_13 == 'TESTS'
    var_14 = '/path/to/src'
    var_15 = 'my_module'
    var_16 = module_0.module(var_15, var_6)
    assert var_16 == 'FIRSTPARTY'
    var_17 = 'my_namespace'
    var_18 = [var_17]
    var_19 = 'my_namespace.submodule'
    var_20 = module_0.module(var_19, var_6)
    assert var_20 == 'FIRSTPARTY'



# Parsed testcases at query #9
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
    var_2 = 'test*'
    var_3 = [var_2]
    var_4 = module_1.Config()
    var_5 = 'test_module'
    var_6 = module_0.module(var_5, var_4)
    assert var_6 == 'test*'
    var_7 = '.local_module'
    var_8 = module_0.module(var_7)
    assert var_8 == 'LOCALFOLDER'
    var_9 = '^django'
    var_10 = 'DJANGO'
    var_11 = 'django.contrib'
    var_12 = module_0.module(var_11, var_4)
    assert var_12 == 'DJANGO'
    var_13 = 'my_project'
    var_14 = var_0 / var_13
    var_15 = 'module.py'
    var_16 = var_14 / var_15
    var_17 = [var_14]
    var_18 = module_1.Config()
    var_19 = 'my_project.module'
    var_20 = module_0.module(var_19, var_18)
    assert var_20 == 'FIRSTPARTY'



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
    var_8 = 'django.apps'
    var_9 = module_0.module(var_8, var_6)
    assert var_9 == 'django'
    var_10 = '.local_module'
    var_11 = module_0.module(var_10)
    assert var_11 == 'LOCALFOLDER'
    var_12 = '.sub.local_module'
    var_13 = module_0.module(var_12)
    assert var_13 == 'LOCALFOLDER'
    var_14 = '^test_.*'
    var_15 = 'TESTS'
    var_16 = 'test_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_package.submodule'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/path/to/project'
    var_21 = 'project'
    var_22 = module_0.module(var_21, var_6)
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'project.submodule'
    var_24 = module_0.module(var_23, var_6)
    assert var_24 == 'FIRSTPARTY'
    var_25 = 'project.namespace'
    var_26 = [var_25]
    var_27 = 'project.namespace.submodule'
    var_28 = module_0.module(var_27, var_6)
    assert var_28 == 'FIRSTPARTY'
    var_29 = 'requests'
    var_30 = module_0.module(var_29)
    assert var_30 == 'THIRDPARTY'
    var_31 = 'flask'
    var_32 = module_0.module(var_31)
    assert var_32 == 'THIRDPARTY'



# Parsed testcases at query #11
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
    var_2 = 'test*'
    var_3 = [var_2]
    var_4 = module_1.Config()
    var_5 = 'test_module'
    var_6 = module_0.module(var_5, var_4)
    assert var_6 == 'test*'
    var_7 = '.local_module'
    var_8 = module_0.module(var_7)
    assert var_8 == 'LOCALFOLDER'
    var_9 = '^django'
    var_10 = 'DJANGO'
    var_11 = 'django.contrib'
    var_12 = module_0.module(var_11, var_4)
    assert var_12 == 'DJANGO'
    var_13 = '/path/to/src'
    var_14 = 'my_module'
    var_15 = module_0.module(var_14, var_4)
    assert var_15 == 'FIRSTPARTY'
    var_16 = 'my_namespace'
    var_17 = [var_16]
    var_18 = 'my_namespace.submodule'
    var_19 = module_0.module(var_18, var_4)
    assert var_19 == 'FIRSTPARTY'



# Parsed testcases at query #12
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
    var_12 = '.sub.local_module'
    var_13 = module_0.module(var_12)
    assert var_13 == 'LOCALFOLDER'
    var_14 = '^test_.*'
    var_15 = 'TESTS'
    var_16 = 'test_example'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_example.sub'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/path/to/project'
    var_21 = 'project'
    var_22 = module_0.module(var_21, var_6)
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'project.sub'
    var_24 = module_0.module(var_23, var_6)
    assert var_24 == 'FIRSTPARTY'
    var_25 = 'project.namespace'
    var_26 = [var_25]
    var_27 = 'project.namespace.sub'
    var_28 = module_0.module(var_27, var_6)
    assert var_28 == 'FIRSTPARTY'
    var_29 = 'unknown_module'
    var_30 = module_0.module(var_29)
    assert var_30 == 'THIRDPARTY'



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
    var_5 = '.local_module'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'LOCALFOLDER'
    var_7 = 'my_project'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'FIRSTPARTY'
    var_9 = 'unknown_module'
    var_10 = module_1.module(var_9, var_0)
    assert var_10 == 'THIRDPARTY'



# Parsed testcases at query #14
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
    var_8 = 'django.apps'
    var_9 = module_0.module(var_8, var_6)
    assert var_9 == 'django'
    var_10 = '.local_module'
    var_11 = module_0.module(var_10)
    var_12 = '.local.submodule'
    var_13 = module_0.module(var_12)
    var_14 = '^test.*'
    var_15 = 'TESTS'
    var_16 = 'test_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test.submodule'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/path/to/src'
    var_21 = 'my_module'
    var_22 = module_0.module(var_21, var_6)
    var_23 = 'my_namespace'
    var_24 = [var_23]
    var_25 = False
    var_26 = 'my_namespace.submodule'
    var_27 = module_0.module(var_26, var_6)
    var_28 = 'THIRDPARTY'
    var_29 = module_1.Config()
    var_30 = 'unknown_module'
    var_31 = module_0.module(var_30, var_29)
    assert var_31 == 'THIRDPARTY'



# Parsed testcases at query #15
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
    var_12 = '.another_local'
    var_13 = module_0.module(var_12)
    var_14 = '^test_.*'
    var_15 = 'TESTS'
    var_16 = 'test_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_another'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/path/to/src'
    var_21 = 'my_module'
    var_22 = module_0.module(var_21, var_6)
    var_23 = 'my_package'
    var_24 = module_0.module(var_23, var_6)
    var_25 = 'my_namespace'
    var_26 = [var_25]
    var_27 = 'my_namespace.submodule'
    var_28 = module_0.module(var_27, var_6)
    var_29 = 'THIRDPARTY'
    var_30 = module_1.Config()
    var_31 = 'unknown_module'
    var_32 = module_0.module(var_31, var_30)
    assert var_32 == 'THIRDPARTY'



# Parsed testcases at query #17
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
    var_8 = '.local_module'
    var_9 = module_0.module(var_8)
    assert var_9 == 'LOCALFOLDER'
    var_10 = '^test_.*'
    var_11 = 'TESTS'
    var_12 = 'test_example'
    var_13 = module_0.module(var_12, var_6)
    assert var_13 == 'TESTS'
    var_14 = '/path/to/src'
    var_15 = 'my_module'
    var_16 = module_0.module(var_15, var_6)
    assert var_16 == 'FIRSTPARTY'
    var_17 = 'my_namespace'
    var_18 = [var_17]
    var_19 = True
    var_20 = 'my_namespace.submodule'
    var_21 = module_0.module(var_20, var_6)
    assert var_21 == 'FIRSTPARTY'



# Parsed testcases at query #18
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
    var_12 = '.another_local'
    var_13 = module_0.module(var_12)
    var_14 = '^test_'
    var_15 = 'TESTS'
    var_16 = 'test_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_utils'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/project/src'
    var_21 = 'project_module'
    var_22 = module_0.module(var_21, var_6)
    var_23 = 'project.namespace'
    var_24 = [var_23]
    var_25 = True
    var_26 = 'project.namespace.submodule'
    var_27 = module_0.module(var_26, var_6)
    var_28 = 'THIRDPARTY'
    var_29 = module_1.Config()
    var_30 = 'some_third_party'
    var_31 = module_0.module(var_30, var_29)
    assert var_31 == 'THIRDPARTY'



# Parsed testcases at query #19
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    var_2 = 'test*'
    var_3 = [var_2]
    var_4 = module_1.Config()
    var_5 = 'test_module'
    var_6 = module_0.module(var_5, var_4)
    assert var_6 == 'test*'
    var_7 = '.local_module'
    var_8 = module_0.module(var_7)
    var_9 = '^django'
    var_10 = 'DJANGO'
    var_11 = 'django.contrib'
    var_12 = module_0.module(var_11, var_4)
    assert var_12 == 'DJANGO'
    var_13 = 'my_project'
    var_14 = var_0 / var_13
    var_15 = 'module.py'
    var_16 = var_14 / var_15
    var_17 = [var_14]
    var_18 = module_1.Config()
    var_19 = 'my_project.module'
    var_20 = module_0.module(var_19, var_18)
    var_21 = 'namespace'
    var_22 = var_0 / var_21
    var_23 = 'submodule.py'
    var_24 = var_22 / var_23
    var_25 = [var_22]
    var_26 = [var_21]
    var_27 = module_1.Config()
    var_28 = 'namespace.submodule'
    var_29 = module_0.module(var_28, var_27)
    var_30 = 'non_existent_module'
    var_31 = module_0.module(var_30)



# Parsed testcases at query #20
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    var_2 = 'test*'
    var_3 = [var_2]
    var_4 = module_1.Config()
    var_5 = 'test_module'
    var_6 = module_0.module(var_5, var_4)
    assert var_6 == 'test*'
    var_7 = '.local_module'
    var_8 = module_0.module(var_7)
    var_9 = '^test_.*'
    var_10 = 'TEST'
    var_11 = 'test_example'
    var_12 = module_0.module(var_11, var_4)
    assert var_12 == 'TEST'
    var_13 = '/path/to/src'
    var_14 = 'src_module'
    var_15 = module_0.module(var_14, var_4)
    var_16 = 'namespace'
    var_17 = [var_16]
    var_18 = 'namespace.submodule'
    var_19 = module_0.module(var_18, var_4)
    var_20 = 'nonexistent_module'
    var_21 = module_0.module(var_20)



# Parsed testcases at query #21
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
    var_4 = 'test*'
    var_5 = [var_4]
    var_6 = module_1.Config()
    var_7 = 'test_module'
    var_8 = module_0.module(var_7, var_6)
    assert var_8 == 'test*'
    var_9 = '.test_module'
    var_10 = module_0.module(var_9, var_6)
    assert var_10 == 'test*'
    var_11 = '.local_module'
    var_12 = module_0.module(var_11)
    var_13 = '..parent_module'
    var_14 = module_0.module(var_13)
    var_15 = '^django'
    var_16 = 'DJANGO'
    var_17 = 'django.contrib'
    var_18 = module_0.module(var_17, var_6)
    assert var_18 == 'DJANGO'
    var_19 = 'django.shortcuts'
    var_20 = module_0.module(var_19, var_6)
    assert var_20 == 'DJANGO'
    var_21 = '/path/to/src'
    var_22 = 'my_module'
    var_23 = module_0.module(var_22, var_6)
    var_24 = 'my_package'
    var_25 = module_0.module(var_24, var_6)
    var_26 = 'my_namespace'
    var_27 = [var_26]
    var_28 = True
    var_29 = 'my_namespace.submodule'
    var_30 = module_0.module(var_29, var_6)
    var_31 = 'THIRDPARTY'
    var_32 = module_1.Config()
    var_33 = 'unknown_module'
    var_34 = module_0.module(var_33, var_32)
    assert var_34 == 'THIRDPARTY'



# Parsed testcases at query #22
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
    var_12 = '.local.submodule'
    var_13 = module_0.module(var_12)
    assert var_13 == 'LOCALFOLDER'
    var_14 = '^test_.*'
    var_15 = 'TESTS'
    var_16 = 'test_example'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_example.submodule'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = 'my_project'
    var_21 = var_0 / var_20
    var_22 = 'module.py'
    var_23 = var_21 / var_22
    var_24 = [var_21]
    var_25 = module_1.Config()
    var_26 = module_0.module(var_20, var_25)
    assert var_26 == 'FIRSTPARTY'
    var_27 = 'my_project.submodule'
    var_28 = module_0.module(var_27, var_25)
    assert var_28 == 'FIRSTPARTY'
    var_29 = 'namespace_pkg'
    var_30 = var_0 / var_29
    var_31 = 'submodule.py'
    var_32 = var_30 / var_31
    var_33 = [var_30]
    var_34 = [var_29]
    var_35 = module_1.Config()
    var_36 = module_0.module(var_29, var_35)
    assert var_36 == 'FIRSTPARTY'
    var_37 = 'namespace_pkg.submodule'
    var_38 = module_0.module(var_37, var_35)
    assert var_38 == 'FIRSTPARTY'
    var_39 = 'unknown_module'
    var_40 = module_0.module(var_39)
    assert var_40 == 'THIRDPARTY'



# Parsed testcases at query #23
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    var_2 = 'test*'
    var_3 = [var_2]
    var_4 = module_1.Config()
    var_5 = 'test_module'
    var_6 = module_0.module(var_5, var_4)
    assert var_6 == 'test*'
    var_7 = '.local_module'
    var_8 = module_0.module(var_7)
    var_9 = '^django'
    var_10 = 'DJANGO'
    var_11 = 'django.contrib'
    var_12 = module_0.module(var_11, var_4)
    assert var_12 == 'DJANGO'
    var_13 = 'my_project'
    var_14 = var_0 / var_13
    var_15 = 'module.py'
    var_16 = var_14 / var_15
    var_17 = [var_14]
    var_18 = module_1.Config()
    var_19 = 'my_project.module'
    var_20 = module_0.module(var_19, var_18)
    var_21 = 'namespace'
    var_22 = var_0 / var_21
    var_23 = 'submodule.py'
    var_24 = var_22 / var_23
    var_25 = [var_22]
    var_26 = [var_21]
    var_27 = module_1.Config()
    var_28 = 'namespace.submodule'
    var_29 = module_0.module(var_28, var_27)
    var_30 = 'non_existent_module'
    var_31 = module_0.module(var_30)



# Parsed testcases at query #24
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
    var_4 = 'test*'
    var_5 = [var_4]
    var_6 = module_1.Config()
    var_7 = 'test_module'
    var_8 = module_0.module(var_7, var_6)
    assert var_8 == 'test*'
    var_9 = '.local_module'
    var_10 = module_0.module(var_9)
    assert var_10 == 'LOCALFOLDER'
    var_11 = '^django'
    var_12 = 'DJANGO'
    var_13 = 'django.contrib'
    var_14 = module_0.module(var_13, var_6)
    assert var_14 == 'DJANGO'
    var_15 = '/path/to/src'
    var_16 = 'my_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'FIRSTPARTY'
    var_18 = 'my_namespace'
    var_19 = [var_18]
    var_20 = 'my_namespace.submodule'
    var_21 = module_0.module(var_20, var_6)
    assert var_21 == 'FIRSTPARTY'
    var_22 = 'unknown_module'
    var_23 = module_0.module(var_22)
    assert var_23 == 'THIRDPARTY'



# Parsed testcases at query #25
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
    var_2 = 'test*'
    var_3 = [var_2]
    var_4 = module_1.Config()
    var_5 = 'test_module'
    var_6 = module_0.module(var_5, var_4)
    assert var_6 == 'test*'
    var_7 = '.local_module'
    var_8 = module_0.module(var_7)
    assert var_8 == 'LOCALFOLDER'
    var_9 = '^django'
    var_10 = 'DJANGO'
    var_11 = 'django.contrib'
    var_12 = module_0.module(var_11, var_4)
    assert var_12 == 'DJANGO'
    var_13 = '/src'
    var_14 = 'src_module'
    var_15 = module_0.module(var_14, var_4)
    assert var_15 == 'FIRSTPARTY'
    var_16 = 'parent'
    var_17 = [var_16]
    var_18 = True
    var_19 = 'parent.child'
    var_20 = module_0.module(var_19, var_4)
    assert var_20 == 'FIRSTPARTY'



# Parsed testcases at query #26
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
    var_2 = 'test*'
    var_3 = [var_2]
    var_4 = module_1.Config()
    var_5 = 'test_module'
    var_6 = module_0.module(var_5, var_4)
    assert var_6 == 'test*'
    var_7 = '.local_module'
    var_8 = module_0.module(var_7)
    var_9 = '^django'
    var_10 = 'DJANGO'
    var_11 = 'django.contrib'
    var_12 = module_0.module(var_11, var_4)
    assert var_12 == 'DJANGO'
    var_13 = '/path/to/src'
    var_14 = 'my_module'
    var_15 = module_0.module(var_14, var_4)
    var_16 = 'my_namespace'
    var_17 = [var_16]
    var_18 = 'my_namespace.sub_module'
    var_19 = module_0.module(var_18, var_4)



# Parsed testcases at query #27
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
    var_4 = 'numpy'
    var_5 = [var_4]
    var_6 = module_1.Config()
    var_7 = module_0.module(var_4, var_6)
    assert var_7 == 'numpy'
    var_8 = 'numpy.core'
    var_9 = module_0.module(var_8, var_6)
    assert var_9 == 'numpy'
    var_10 = '.local_module'
    var_11 = module_0.module(var_10)
    var_12 = '..parent_module'
    var_13 = module_0.module(var_12)
    var_14 = '^django'
    var_15 = 'DJANGO'
    var_16 = 'django'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'DJANGO'
    var_18 = 'django.contrib'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'DJANGO'
    var_20 = 'my_project'
    var_21 = var_0 / var_20
    var_22 = 'module.py'
    var_23 = var_21 / var_22
    var_24 = [var_21]
    var_25 = module_1.Config()
    var_26 = module_0.module(var_20, var_25)
    assert var_26 == 'FIRSTPARTY'
    var_27 = 'my_project.submodule'
    var_28 = module_0.module(var_27, var_25)
    assert var_28 == 'FIRSTPARTY'
    var_29 = 'namespace'
    var_30 = var_0 / var_29
    var_31 = 'submodule.py'
    var_32 = var_30 / var_31
    var_33 = [var_30]
    var_34 = [var_29]
    var_35 = module_1.Config()
    var_36 = 'namespace.submodule'
    var_37 = module_0.module(var_36, var_35)
    assert var_37 == 'FIRSTPARTY'
    var_38 = 'non_existent_module'
    var_39 = module_0.module(var_38)
    assert var_39 == 'THIRDPARTY'



# Parsed testcases at query #28
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
    var_12 = '.local.submodule'
    var_13 = module_0.module(var_12)
    var_14 = '^test_.*'
    var_15 = 'TESTS'
    var_16 = 'test_utils'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_utils.helper'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/project/src'
    var_21 = 'project'
    var_22 = module_0.module(var_21, var_6)
    var_23 = 'project.module'
    var_24 = module_0.module(var_23, var_6)
    var_25 = 'project.namespace'
    var_26 = [var_25]
    var_27 = 'project.namespace.submodule'
    var_28 = module_0.module(var_27, var_6)
    var_29 = True
    var_30 = module_0.module(var_27, var_6)
    var_31 = 'THIRDPARTY'
    var_32 = module_1.Config()
    var_33 = 'unknown_module'
    var_34 = module_0.module(var_33, var_32)
    assert var_34 == 'THIRDPARTY'



# Parsed testcases at query #29
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
    var_2 = 'test*'
    var_3 = [var_2]
    var_4 = module_1.Config()
    var_5 = 'test_module'
    var_6 = module_0.module(var_5, var_4)
    assert var_6 == 'test*'
    var_7 = '.local_module'
    var_8 = module_0.module(var_7)
    assert var_8 == 'LOCALFOLDER'
    var_9 = '^test_.*'
    var_10 = 'TEST'
    var_11 = 'test_example'
    var_12 = module_0.module(var_11, var_4)
    assert var_12 == 'TEST'
    var_13 = '/path/to/src'
    var_14 = 'src_module'
    var_15 = module_0.module(var_14, var_4)
    assert var_15 == 'FIRSTPARTY'
    var_16 = 'namespace'
    var_17 = [var_16]
    var_18 = True
    var_19 = 'namespace.submodule'
    var_20 = module_0.module(var_19, var_4)
    assert var_20 == 'FIRSTPARTY'



# Parsed testcases at query #30
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    var_2 = 'sys'
    var_3 = module_0.module(var_2)
    var_4 = 'django'
    var_5 = [var_4]
    var_6 = module_1.Config()
    var_7 = module_0.module(var_4, var_6)
    assert var_7 == 'django'
    var_8 = '.local_module'
    var_9 = module_0.module(var_8)
    var_10 = '^test_'
    var_11 = 'TESTS'
    var_12 = 'test_example'
    var_13 = module_0.module(var_12, var_6)
    assert var_13 == 'TESTS'
    var_14 = '/project/src'
    var_15 = 'project'
    var_16 = module_0.module(var_15, var_6)
    var_17 = [var_15]
    var_18 = 'project.submodule'
    var_19 = module_0.module(var_18, var_6)
    var_20 = 'nonexistent_module'
    var_21 = module_0.module(var_20)



# Parsed testcases at query #31
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
    var_12 = '^test_'
    var_13 = 'TESTS'
    var_14 = 'test_module'
    var_15 = module_0.module(var_14, var_6)
    assert var_15 == 'TESTS'
    var_16 = 'test_package.submodule'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'src'
    var_19 = var_0 / var_18
    var_20 = 'my_package'
    var_21 = var_19 / var_20
    var_22 = var_19 / var_20
    var_23 = '__init__.py'
    var_24 = var_22 / var_23
    var_25 = ''
    var_26 = [var_19]
    var_27 = module_1.Config()
    var_28 = module_0.module(var_20, var_27)
    assert var_28 == 'FIRSTPARTY'
    var_29 = 'my_package.submodule'
    var_30 = module_0.module(var_29, var_27)
    assert var_30 == 'FIRSTPARTY'
    var_31 = 'src'
    var_32 = var_0 / var_31
    var_33 = 'namespace'
    var_34 = var_32 / var_33
    var_35 = var_32 / var_33
    var_36 = 'module.py'
    var_37 = var_35 / var_36
    var_38 = ''
    var_39 = [var_32]
    var_40 = [var_33]
    var_41 = module_1.Config()
    var_42 = 'namespace.module'
    var_43 = module_0.module(var_42, var_41)
    assert var_43 == 'FIRSTPARTY'
    var_44 = 'unknown_module'
    var_45 = module_0.module(var_44)
    assert var_45 == 'THIRDPARTY'



# Parsed testcases at query #32
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    var_2 = 'test*'
    var_3 = [var_2]
    var_4 = module_1.Config()
    var_5 = 'test_module'
    var_6 = module_0.module(var_5, var_4)
    assert var_6 == 'test*'
    var_7 = '.local_module'
    var_8 = module_0.module(var_7)
    var_9 = '^django'
    var_10 = 'DJANGO'
    var_11 = 'django.contrib'
    var_12 = module_0.module(var_11, var_4)
    assert var_12 == 'DJANGO'
    var_13 = 'my_project'
    var_14 = var_0 / var_13
    var_15 = 'module.py'
    var_16 = var_14 / var_15
    var_17 = [var_14]
    var_18 = module_1.Config()
    var_19 = module_0.module(var_13, var_18)



# Parsed testcases at query #33
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
    var_12 = '.local.submodule'
    var_13 = module_0.module(var_12)
    assert var_13 == 'LOCALFOLDER'
    var_14 = '^test_.*'
    var_15 = 'TESTS'
    var_16 = 'test_example'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_example.submodule'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/path/to/project'
    var_21 = 'project'
    var_22 = module_0.module(var_21, var_6)
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'project.submodule'
    var_24 = module_0.module(var_23, var_6)
    assert var_24 == 'FIRSTPARTY'
    var_25 = 'project.namespace'
    var_26 = [var_25]
    var_27 = 'project.namespace.submodule'
    var_28 = module_0.module(var_27, var_6)
    assert var_28 == 'FIRSTPARTY'
    var_29 = True
    var_30 = module_0.module(var_27, var_6)
    assert var_30 == 'FIRSTPARTY'
    var_31 = 'unknown_module'
    var_32 = module_0.module(var_31)
    assert var_32 == 'THIRDPARTY'



# Parsed testcases at query #34
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
    var_2 = 'test*'
    var_3 = [var_2]
    var_4 = module_1.Config()
    var_5 = 'test_module'
    var_6 = module_0.module(var_5, var_4)
    assert var_6 == 'test*'
    var_7 = '.local_module'
    var_8 = module_0.module(var_7)
    var_9 = '^django'
    var_10 = 'DJANGO'
    var_11 = 'django.contrib'
    var_12 = module_0.module(var_11, var_4)
    assert var_12 == 'DJANGO'
    var_13 = '/path/to/src'
    var_14 = 'my_module'
    var_15 = module_0.module(var_14, var_4)
    var_16 = 'my_namespace'
    var_17 = [var_16]
    var_18 = 'my_namespace.submodule'
    var_19 = module_0.module(var_18, var_4)
    var_20 = 'THIRDPARTY'
    var_21 = module_1.Config()
    var_22 = 'unknown_module'
    var_23 = module_0.module(var_22, var_21)
    assert var_23 == 'THIRDPARTY'



# Parsed testcases at query #35
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
    var_2 = 'test*'
    var_3 = [var_2]
    var_4 = module_1.Config()
    var_5 = 'test_module'
    var_6 = module_0.module(var_5, var_4)
    assert var_6 == 'test*'
    var_7 = '.local_module'
    var_8 = module_0.module(var_7)
    var_9 = '^django'
    var_10 = 'DJANGO'
    var_11 = 'django.contrib'
    var_12 = module_0.module(var_11, var_4)
    assert var_12 == 'DJANGO'
    var_13 = '/path/to/src'
    var_14 = 'my_module'
    var_15 = module_0.module(var_14, var_4)
    var_16 = 'my_namespace'
    var_17 = [var_16]
    var_18 = True
    var_19 = 'my_namespace.sub_module'
    var_20 = module_0.module(var_19, var_4)



# Parsed testcases at query #36
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
    var_12 = '.sub.local_module'
    var_13 = module_0.module(var_12)
    assert var_13 == 'LOCALFOLDER'
    var_14 = '^test_.*'
    var_15 = 'TESTS'
    var_16 = 'test_example'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_example.module'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/path/to/src'
    var_21 = 'my_module'
    var_22 = module_0.module(var_21, var_6)
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'my_module.submodule'
    var_24 = module_0.module(var_23, var_6)
    assert var_24 == 'FIRSTPARTY'
    var_25 = 'namespace_pkg'
    var_26 = [var_25]
    var_27 = module_0.module(var_25, var_6)
    assert var_27 == 'FIRSTPARTY'
    var_28 = 'namespace_pkg.submodule'
    var_29 = module_0.module(var_28, var_6)
    assert var_29 == 'FIRSTPARTY'
    var_30 = 'THIRDPARTY'
    var_31 = module_1.Config()
    var_32 = 'unknown_module'
    var_33 = module_0.module(var_32, var_31)
    assert var_33 == 'THIRDPARTY'



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
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
    var_12 = '.local.module'
    var_13 = module_0.module(var_12)
    var_14 = '^test_.*'
    var_15 = 'TESTS'
    var_16 = 'test_example'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_example.module'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/path/to/src'
    var_21 = 'src_module'
    var_22 = module_0.module(var_21, var_6)
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'namespace'
    var_24 = [var_23]
    var_25 = 'namespace.module'
    var_26 = module_0.module(var_25, var_6)
    assert var_26 == 'FIRSTPARTY'
    var_27 = True
    var_28 = '.py'
    var_29 = [var_28]
    var_30 = frozenset(var_29)
    var_31 = 'auto_namespace.module'
    var_32 = module_0.module(var_31, var_6)
    assert var_32 == 'FIRSTPARTY'
    var_33 = 'THIRDPARTY'
    var_34 = module_1.Config()
    var_35 = 'some_third_party'
    var_36 = module_0.module(var_35, var_34)
    assert var_36 == 'THIRDPARTY'



# Parsed testcases at query #2
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
    var_2 = 'numpy'
    var_3 = [var_2]
    var_4 = module_1.Config()
    var_5 = module_0.module(var_2, var_4)
    assert var_5 == 'numpy'
    var_6 = '.local'
    var_7 = module_0.module(var_6)
    assert var_7 == 'LOCALFOLDER'
    var_8 = 'django.*'
    var_9 = 'DJANGO'
    var_10 = 'django.contrib'
    var_11 = module_0.module(var_10, var_4)
    assert var_11 == 'DJANGO'
    var_12 = '/path/to/src'
    var_13 = 'mymodule'
    var_14 = module_0.module(var_13, var_4)
    assert var_14 == 'FIRSTPARTY'



# Parsed testcases at query #3
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
    var_12 = '.local.module'
    var_13 = module_0.module(var_12)
    var_14 = '^test_.*'
    var_15 = 'TESTS'
    var_16 = 'test_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_package.module'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/path/to/project'
    var_21 = 'project'
    var_22 = module_0.module(var_21, var_6)
    var_23 = 'project.module'
    var_24 = module_0.module(var_23, var_6)
    var_25 = 'project.namespace'
    var_26 = [var_25]
    var_27 = 'project.namespace.module'
    var_28 = module_0.module(var_27, var_6)



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
    var_12 = '..parent_module'
    var_13 = module_0.module(var_12)
    var_14 = '^test_.*'
    var_15 = 'TESTS'
    var_16 = 'test_example'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_example.module'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/project/src'
    var_21 = 'project'
    var_22 = module_0.module(var_21, var_6)
    var_23 = 'project.module'
    var_24 = module_0.module(var_23, var_6)
    var_25 = 'project.namespace'
    var_26 = [var_25]
    var_27 = 'project.namespace.submodule'
    var_28 = module_0.module(var_27, var_6)
    var_29 = True
    var_30 = module_0.module(var_27, var_6)
    var_31 = 'THIRDPARTY'
    var_32 = module_1.Config()
    var_33 = 'unknown_module'
    var_34 = module_0.module(var_33, var_32)
    assert var_34 == 'THIRDPARTY'



# Parsed testcases at query #5
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    var_2 = 'sys'
    var_3 = module_0.module(var_2)
    var_4 = 'django'
    var_5 = [var_4]
    var_6 = module_1.Config()
    var_7 = module_0.module(var_4, var_6)
    assert var_7 == 'django'
    var_8 = 'django.apps'
    var_9 = module_0.module(var_8, var_6)
    assert var_9 == 'django'
    var_10 = '.local_module'
    var_11 = module_0.module(var_10)
    var_12 = '.local.submodule'
    var_13 = module_0.module(var_12)
    var_14 = '^test_.*'
    var_15 = 'TESTS'
    var_16 = 'test_example'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_example.submodule'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/path/to/project'
    var_21 = 'project_module'
    var_22 = module_0.module(var_21, var_6)
    var_23 = 'project.nested'
    var_24 = [var_23]
    var_25 = True
    var_26 = 'project.nested.deep'
    var_27 = module_0.module(var_26, var_6)
    var_28 = 'THIRDPARTY'
    var_29 = module_1.Config()
    var_30 = 'unknown_module'
    var_31 = module_0.module(var_30, var_29)
    assert var_31 == 'THIRDPARTY'



# Parsed testcases at query #6
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
    var_12 = '.local.submodule'
    var_13 = module_0.module(var_12)
    var_14 = '^test_.*'
    var_15 = 'TESTS'
    var_16 = 'test_example'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_example.submodule'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = 'my_project'
    var_21 = var_0 / var_20
    var_22 = 'module.py'
    var_23 = var_21 / var_22
    var_24 = [var_21]
    var_25 = module_1.Config()
    var_26 = module_0.module(var_20, var_25)
    var_27 = 'my_project.submodule'
    var_28 = module_0.module(var_27, var_25)
    var_29 = 'namespace_pkg'
    var_30 = var_0 / var_29
    var_31 = 'subpkg'
    var_32 = var_30 / var_31
    var_33 = [var_30]
    var_34 = [var_29]
    var_35 = module_1.Config()
    var_36 = module_0.module(var_29, var_35)
    var_37 = 'namespace_pkg.subpkg'
    var_38 = module_0.module(var_37, var_35)
    var_39 = 'unknown_module'
    var_40 = module_0.module(var_39)



# Parsed testcases at query #7
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
    var_12 = '.sub.local_module'
    var_13 = module_0.module(var_12)
    assert var_13 == 'LOCALFOLDER'
    var_14 = '^test_.*'
    var_15 = 'TESTS'
    var_16 = 'test_example'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_another'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/path/to/src'
    var_21 = 'my_module'
    var_22 = module_0.module(var_21, var_6)
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'my_package.submodule'
    var_24 = module_0.module(var_23, var_6)
    assert var_24 == 'FIRSTPARTY'
    var_25 = 'my_namespace'
    var_26 = [var_25]
    var_27 = 'my_namespace.submodule'
    var_28 = module_0.module(var_27, var_6)
    assert var_28 == 'FIRSTPARTY'
    var_29 = 'THIRDPARTY'
    var_30 = module_1.Config()
    var_31 = 'unknown_module'
    var_32 = module_0.module(var_31, var_30)
    assert var_32 == 'THIRDPARTY'



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
    var_5 = module_0.module(var_4)
    assert var_5 == 'THIRDPARTY'
    var_6 = 'requests'
    var_7 = module_0.module(var_6)
    assert var_7 == 'THIRDPARTY'
    var_8 = 'custom'
    var_9 = [var_8]
    var_10 = module_1.Config()
    var_11 = 'custom_module'
    var_12 = module_0.module(var_11, var_10)
    assert var_12 == 'custom'
    var_13 = 'custom_sub.module'
    var_14 = module_0.module(var_13, var_10)
    assert var_14 == 'custom'
    var_15 = '.local_module'
    var_16 = module_0.module(var_15)
    assert var_16 == 'LOCALFOLDER'
    var_17 = '.local.submodule'
    var_18 = module_0.module(var_17)
    assert var_18 == 'LOCALFOLDER'
    var_19 = '^test_.*'
    var_20 = 'TESTS'
    var_21 = 'test_module'
    var_22 = module_0.module(var_21, var_10)
    assert var_22 == 'TESTS'
    var_23 = 'test_sub.module'
    var_24 = module_0.module(var_23, var_10)
    assert var_24 == 'TESTS'
    var_25 = '/path/to/src'
    var_26 = 'src_module'
    var_27 = module_0.module(var_26, var_10)
    assert var_27 == 'FIRSTPARTY'
    var_28 = 'src.submodule'
    var_29 = module_0.module(var_28, var_10)
    assert var_29 == 'FIRSTPARTY'
    var_30 = 'namespace'
    var_31 = [var_30]
    var_32 = 'namespace.submodule'
    var_33 = module_0.module(var_32, var_10)
    assert var_33 == 'FIRSTPARTY'
    var_34 = True
    var_35 = 'auto_namespace.submodule'
    var_36 = module_0.module(var_35, var_10)
    assert var_36 == 'FIRSTPARTY'



# Parsed testcases at query #9
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
    var_2 = 'django'
    var_3 = [var_2]
    var_4 = module_1.Config()
    var_5 = module_0.module(var_2, var_4)
    assert var_5 == 'django'
    var_6 = '.local_module'
    var_7 = module_0.module(var_6)
    assert var_7 == 'LOCALFOLDER'
    var_8 = '^test_'
    var_9 = 'TESTS'
    var_10 = 'test_example'
    var_11 = module_0.module(var_10, var_4)
    assert var_11 == 'TESTS'
    var_12 = '/project/src'
    var_13 = 'my_module'
    var_14 = module_0.module(var_13, var_4)
    assert var_14 == 'FIRSTPARTY'
    var_15 = 'my_namespace'
    var_16 = [var_15]
    var_17 = True
    var_18 = 'my_namespace.submodule'
    var_19 = module_0.module(var_18, var_4)
    assert var_19 == 'FIRSTPARTY'



# Parsed testcases at query #10
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
    var_2 = 'test*'
    var_3 = [var_2]
    var_4 = module_1.Config()
    var_5 = 'test_module'
    var_6 = module_0.module(var_5, var_4)
    assert var_6 == 'test*'
    var_7 = '.local_module'
    var_8 = module_0.module(var_7)
    assert var_8 == 'LOCALFOLDER'
    var_9 = '^django'
    var_10 = 'DJANGO'
    var_11 = 'django.contrib'
    var_12 = module_0.module(var_11, var_4)
    assert var_12 == 'DJANGO'
    var_13 = '/path/to/src'
    var_14 = 'my_module'
    var_15 = module_0.module(var_14, var_4)
    assert var_15 == 'FIRSTPARTY'
    var_16 = 'my_namespace'
    var_17 = [var_16]
    var_18 = 'my_namespace.sub_module'
    var_19 = module_0.module(var_18, var_4)
    assert var_19 == 'FIRSTPARTY'
    var_20 = 'THIRDPARTY'
    var_21 = module_1.Config()
    var_22 = 'some_third_party_module'
    var_23 = module_0.module(var_22, var_21)
    assert var_23 == 'THIRDPARTY'



# Parsed testcases at query #11
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
    var_2 = 'test*'
    var_3 = [var_2]
    var_4 = module_1.Config()
    var_5 = 'test_module'
    var_6 = module_0.module(var_5, var_4)
    assert var_6 == 'test*'
    var_7 = '.local_module'
    var_8 = module_0.module(var_7)
    var_9 = '^django'
    var_10 = 'DJANGO'
    var_11 = 'django.contrib'
    var_12 = module_0.module(var_11, var_4)
    assert var_12 == 'DJANGO'
    var_13 = '/path/to/src'
    var_14 = 'my_module'
    var_15 = module_0.module(var_14, var_4)
    var_16 = 'my_namespace'
    var_17 = [var_16]
    var_18 = 'my_namespace.submodule'
    var_19 = module_0.module(var_18, var_4)
    var_20 = 'THIRDPARTY'
    var_21 = module_1.Config()
    var_22 = 'unknown_module'
    var_23 = module_0.module(var_22, var_21)
    assert var_23 == 'THIRDPARTY'



# Parsed testcases at query #12
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
    var_5 = 'flask'
    var_6 = [var_4, var_5]
    var_7 = module_1.Config()
    var_8 = module_0.module(var_4, var_7)
    assert var_8 == 'django'
    var_9 = module_0.module(var_5, var_7)
    assert var_9 == 'flask'
    var_10 = 'django.contrib'
    var_11 = module_0.module(var_10, var_7)
    assert var_11 == 'django'
    var_12 = 'flask.ext'
    var_13 = module_0.module(var_12, var_7)
    assert var_13 == 'flask'
    var_14 = '.local_module'
    var_15 = module_0.module(var_14)
    var_16 = '.sub.local_module'
    var_17 = module_0.module(var_16)
    var_18 = '^test_.*'
    var_19 = 'TESTS'
    var_20 = 'test_module'
    var_21 = module_0.module(var_20, var_7)
    assert var_21 == 'TESTS'
    var_22 = 'test_package.submodule'
    var_23 = module_0.module(var_22, var_7)
    assert var_23 == 'TESTS'
    var_24 = '/path/to/src'
    var_25 = 'src_module'
    var_26 = module_0.module(var_25, var_7)
    var_27 = 'src_package.submodule'
    var_28 = module_0.module(var_27, var_7)
    var_29 = 'namespace_package'
    var_30 = [var_29]
    var_31 = module_0.module(var_29, var_7)
    var_32 = 'namespace_package.submodule'
    var_33 = module_0.module(var_32, var_7)



# Parsed testcases at query #13
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
    var_8 = '.local_module'
    var_9 = module_0.module(var_8)
    var_10 = '^django'
    var_11 = 'DJANGO'
    var_12 = 'django.contrib'
    var_13 = module_0.module(var_12, var_6)
    assert var_13 == 'DJANGO'
    var_14 = 'myproject'
    var_15 = var_0 / var_14
    var_16 = 'module.py'
    var_17 = var_15 / var_16
    var_18 = '# test'
    var_19 = [var_15]
    var_20 = module_1.Config()
    var_21 = module_0.module(var_14, var_20)



# Parsed testcases at query #14
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
    var_8 = '.local_module'
    var_9 = module_0.module(var_8)
    assert var_9 == 'LOCALFOLDER'
    var_10 = '^test_.*'
    var_11 = 'TESTS'
    var_12 = 'test_example'
    var_13 = module_0.module(var_12, var_6)
    assert var_13 == 'TESTS'
    var_14 = '/project/src'
    var_15 = 'my_module'
    var_16 = module_0.module(var_15, var_6)
    assert var_16 == 'FIRSTPARTY'
    var_17 = 'my_namespace'
    var_18 = [var_17]
    var_19 = 'my_namespace.submodule'
    var_20 = module_0.module(var_19, var_6)
    assert var_20 == 'FIRSTPARTY'
    var_21 = 'non_existent_module'
    var_22 = module_0.module(var_21)
    assert var_22 == 'THIRDPARTY'



# Parsed testcases at query #15
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
    var_12 = '.local.submodule'
    var_13 = module_0.module(var_12)
    assert var_13 == 'LOCALFOLDER'
    var_14 = '^test_.*'
    var_15 = 'TESTS'
    var_16 = 'test_example'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_example.submodule'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = 'my_project'
    var_21 = var_0 / var_20
    var_22 = 'module.py'
    var_23 = var_21 / var_22
    var_24 = [var_21]
    var_25 = module_1.Config()
    var_26 = module_0.module(var_20, var_25)
    assert var_26 == 'FIRSTPARTY'
    var_27 = 'my_project.submodule'
    var_28 = module_0.module(var_27, var_25)
    assert var_28 == 'FIRSTPARTY'
    var_29 = 'namespace_pkg'
    var_30 = var_0 / var_29
    var_31 = 'submodule.py'
    var_32 = var_30 / var_31
    var_33 = [var_30]
    var_34 = [var_29]
    var_35 = module_1.Config()
    var_36 = 'namespace_pkg.submodule'
    var_37 = module_0.module(var_36, var_35)
    assert var_37 == 'FIRSTPARTY'
    var_38 = 'THIRDPARTY'
    var_39 = module_1.Config()
    var_40 = 'unknown_module'
    var_41 = module_0.module(var_40, var_39)
    assert var_41 == 'THIRDPARTY'



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
    var_4 = 'django'
    var_5 = [var_4]
    var_6 = module_1.Config()
    var_7 = module_0.module(var_4, var_6)
    assert var_7 == 'django'
    var_8 = '.local_module'
    var_9 = module_0.module(var_8)
    var_10 = '^test_'
    var_11 = 'TESTS'
    var_12 = 'test_module'
    var_13 = module_0.module(var_12, var_6)
    assert var_13 == 'TESTS'
    var_14 = '/project/src'
    var_15 = 'project'
    var_16 = module_0.module(var_15, var_6)
    var_17 = 'project.nested'
    var_18 = [var_17]
    var_19 = True
    var_20 = module_0.module(var_17, var_6)
    var_21 = 'nonexistent_module'
    var_22 = module_0.module(var_21)
    assert var_22 == 'THIRDPARTY'



# Parsed testcases at query #17
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
    var_4 = 'test*'
    var_5 = [var_4]
    var_6 = module_1.Config()
    var_7 = 'test_module'
    var_8 = module_0.module(var_7, var_6)
    assert var_8 == 'test*'
    var_9 = '.local_module'
    var_10 = module_0.module(var_9)
    assert var_10 == 'LOCALFOLDER'
    var_11 = '^django'
    var_12 = 'DJANGO'
    var_13 = 'django.contrib'
    var_14 = module_0.module(var_13, var_6)
    assert var_14 == 'DJANGO'
    var_15 = '/path/to/src'
    var_16 = 'my_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'FIRSTPARTY'
    var_18 = 'my_namespace'
    var_19 = [var_18]
    var_20 = True
    var_21 = 'my_namespace.submodule'
    var_22 = module_0.module(var_21, var_6)
    assert var_22 == 'FIRSTPARTY'



# Parsed testcases at query #18
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
    var_12 = '.local.submodule'
    var_13 = module_0.module(var_12)
    assert var_13 == 'LOCALFOLDER'
    var_14 = '^test.*'
    var_15 = 'TESTS'
    var_16 = 'test_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_utils'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/path/to/src'
    var_21 = 'src_module'
    var_22 = module_0.module(var_21, var_6)
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'namespace'
    var_24 = [var_23]
    var_25 = 'namespace.submodule'
    var_26 = module_0.module(var_25, var_6)
    assert var_26 == 'FIRSTPARTY'



# Parsed testcases at query #19
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
    var_2 = 'test*'
    var_3 = [var_2]
    var_4 = module_1.Config()
    var_5 = 'test_module'
    var_6 = module_0.module(var_5, var_4)
    assert var_6 == 'test*'
    var_7 = '.local_module'
    var_8 = module_0.module(var_7)
    assert var_8 == 'LOCALFOLDER'
    var_9 = '^test_.*'
    var_10 = 'TEST'
    var_11 = 'test_example'
    var_12 = module_0.module(var_11, var_4)
    assert var_12 == 'TEST'
    var_13 = '/path/to/src'
    var_14 = 'src_module'
    var_15 = module_0.module(var_14, var_4)
    assert var_15 == 'FIRSTPARTY'
    var_16 = 'namespace'
    var_17 = [var_16]
    var_18 = True
    var_19 = 'namespace.submodule'
    var_20 = module_0.module(var_19, var_4)
    assert var_20 == 'FIRSTPARTY'
    var_21 = 'THIRDPARTY'
    var_22 = module_1.Config()
    var_23 = 'unknown_module'
    var_24 = module_0.module(var_23, var_22)
    assert var_24 == 'THIRDPARTY'



# Parsed testcases at query #20
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
    var_12 = '.sub.local_module'
    var_13 = module_0.module(var_12)
    assert var_13 == 'LOCALFOLDER'
    var_14 = '^test_.*'
    var_15 = 'TESTS'
    var_16 = 'test_example'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_example.sub'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/path/to/src'
    var_21 = 'my_module'
    var_22 = module_0.module(var_21, var_6)
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'my_namespace'
    var_24 = [var_23]
    var_25 = 'my_namespace.sub_module'
    var_26 = module_0.module(var_25, var_6)
    assert var_26 == 'FIRSTPARTY'
    var_27 = True
    var_28 = 'auto_namespace.sub_module'
    var_29 = module_0.module(var_28, var_6)
    assert var_29 == 'FIRSTPARTY'
    var_30 = 'THIRDPARTY'
    var_31 = module_1.Config()
    var_32 = 'unknown_module'
    var_33 = module_0.module(var_32, var_31)
    assert var_33 == 'THIRDPARTY'



# Parsed testcases at query #21
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
    var_12 = '^test_'
    var_13 = 'TESTS'
    var_14 = 'test_module'
    var_15 = module_0.module(var_14, var_6)
    assert var_15 == 'TESTS'
    var_16 = 'my_project'
    var_17 = var_0 / var_16
    var_18 = 'module.py'
    var_19 = var_17 / var_18
    var_20 = '# test'
    var_21 = [var_17]
    var_22 = module_1.Config()
    var_23 = module_0.module(var_16, var_22)
    assert var_23 == 'FIRSTPARTY'



# Parsed testcases at query #22
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
    var_8 = '.local_module'
    var_9 = module_0.module(var_8)
    var_10 = '^django'
    var_11 = 'DJANGO'
    var_12 = 'django.contrib'
    var_13 = module_0.module(var_12, var_6)
    assert var_13 == 'DJANGO'
    var_14 = '/path/to/src'
    var_15 = 'my_module'
    var_16 = module_0.module(var_15, var_6)
    var_17 = 'my_namespace'
    var_18 = [var_17]
    var_19 = 'my_namespace.submodule'
    var_20 = module_0.module(var_19, var_6)
    var_21 = True
    var_22 = 'my_namespace.submodule'
    var_23 = module_0.module(var_22, var_6)
    var_24 = 'THIRDPARTY'
    var_25 = module_1.Config()
    var_26 = 'unknown_module'
    var_27 = module_0.module(var_26, var_25)
    assert var_27 == 'THIRDPARTY'



# Parsed testcases at query #23
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    var_2 = 'test*'
    var_3 = [var_2]
    var_4 = module_1.Config()
    var_5 = 'test_module'
    var_6 = module_0.module(var_5, var_4)
    assert var_6 == 'test*'
    var_7 = '.local'
    var_8 = module_0.module(var_7)
    var_9 = '^django'
    var_10 = 'DJANGO'
    var_11 = 'django.contrib'
    var_12 = module_0.module(var_11, var_4)
    assert var_12 == 'DJANGO'
    var_13 = '/path/to/src'
    var_14 = 'mymodule'
    var_15 = module_0.module(var_14, var_4)



# Parsed testcases at query #24
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    var_2 = 'test*'
    var_3 = [var_2]
    var_4 = module_1.Config()
    var_5 = 'test_module'
    var_6 = module_0.module(var_5, var_4)
    assert var_6 == 'test*'
    var_7 = '.local_module'
    var_8 = module_0.module(var_7)
    var_9 = '^django'
    var_10 = 'DJANGO'
    var_11 = 'django.contrib'
    var_12 = module_0.module(var_11, var_4)
    assert var_12 == 'DJANGO'
    var_13 = '/path/to/src'
    var_14 = 'src_module'
    var_15 = module_0.module(var_14, var_4)
    var_16 = 'namespace'
    var_17 = [var_16]
    var_18 = 'namespace.module'
    var_19 = module_0.module(var_18, var_4)



# Parsed testcases at query #25
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    var_2 = 'sys'
    var_3 = module_0.module(var_2)
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
    var_12 = '..parent_module'
    var_13 = module_0.module(var_12)
    var_14 = '^test_.*'
    var_15 = 'TESTS'
    var_16 = 'test_example'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_example.module'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/path/to/src'
    var_21 = 'my_module'
    var_22 = module_0.module(var_21, var_6)
    var_23 = 'my_package'
    var_24 = module_0.module(var_23, var_6)
    var_25 = 'my_namespace'
    var_26 = [var_25]
    var_27 = True
    var_28 = 'my_namespace.submodule'
    var_29 = module_0.module(var_28, var_6)
    var_30 = 'THIRDPARTY'
    var_31 = module_1.Config()
    var_32 = 'unknown_module'
    var_33 = module_0.module(var_32, var_31)
    assert var_33 == 'THIRDPARTY'



# Parsed testcases at query #26
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1)
    var_3 = 'django'
    var_4 = module_1.module(var_3)
    var_5 = '.'
    var_6 = module_1.module(var_5)
    var_7 = 'local_module'
    var_8 = module_1.module(var_7)
    var_9 = 'custom_module'
    var_10 = module_1.module(var_9)



# Parsed testcases at query #27
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
    var_8 = 'django.apps'
    var_9 = module_0.module(var_8, var_6)
    assert var_9 == 'django'
    var_10 = '.local_module'
    var_11 = module_0.module(var_10)
    assert var_11 == 'LOCALFOLDER'
    var_12 = '.sub.local_module'
    var_13 = module_0.module(var_12)
    assert var_13 == 'LOCALFOLDER'
    var_14 = 'mycompany'
    var_15 = [var_14]
    var_16 = module_1.Config()
    var_17 = 'mycompany.utils'
    var_18 = module_0.module(var_17)
    assert var_18 == 'FIRSTPARTY'
    var_19 = 'mycompany.core'
    var_20 = module_0.module(var_19)
    assert var_20 == 'FIRSTPARTY'
    var_21 = '/project/src'
    var_22 = 'project'
    var_23 = module_0.module(var_22, var_16)
    assert var_23 == 'FIRSTPARTY'
    var_24 = 'project.submodule'
    var_25 = module_0.module(var_24, var_16)
    assert var_25 == 'FIRSTPARTY'
    var_26 = [var_22]
    var_27 = 'project.subpackage'
    var_28 = module_0.module(var_27, var_16)
    assert var_28 == 'FIRSTPARTY'
    var_29 = 'unknown_module'
    var_30 = module_0.module(var_29)
    assert var_30 == 'THIRDPARTY'



# Parsed testcases at query #28
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
    var_12 = '.local.package'
    var_13 = module_0.module(var_12)
    assert var_13 == 'LOCALFOLDER'
    var_14 = '^test.*'
    var_15 = 'TESTS'
    var_16 = 'test_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_package.submodule'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/project/src'
    var_21 = 'project'
    var_22 = module_0.module(var_21, var_6)
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'project.submodule'
    var_24 = module_0.module(var_23, var_6)
    assert var_24 == 'FIRSTPARTY'
    var_25 = 'project.namespace'
    var_26 = [var_25]
    var_27 = 'project.namespace.submodule'
    var_28 = module_0.module(var_27, var_6)
    assert var_28 == 'FIRSTPARTY'
    var_29 = 'THIRDPARTY'
    var_30 = module_1.Config()
    var_31 = 'unknown_module'
    var_32 = module_0.module(var_31, var_30)
    assert var_32 == 'THIRDPARTY'



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
    var_12 = '.sub.local_module'
    var_13 = module_0.module(var_12)
    assert var_13 == 'LOCALFOLDER'
    var_14 = '^test_.*'
    var_15 = 'TESTS'
    var_16 = 'test_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_package.submodule'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/path/to/project'
    var_21 = 'project_module'
    var_22 = module_0.module(var_21, var_6)
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'project_module.submodule'
    var_24 = module_0.module(var_23, var_6)
    assert var_24 == 'FIRSTPARTY'
    var_25 = 'project_module'
    var_26 = [var_25]
    var_27 = 'project_module.submodule'
    var_28 = module_0.module(var_27, var_6)
    assert var_28 == 'FIRSTPARTY'



# Parsed testcases at query #30
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
    var_4 = 'test*'
    var_5 = [var_4]
    var_6 = module_1.Config()
    var_7 = 'test_module'
    var_8 = module_0.module(var_7, var_6)
    assert var_8 == 'test*'
    var_9 = '.local_module'
    var_10 = module_0.module(var_9)
    assert var_10 == 'LOCALFOLDER'
    var_11 = 'django.*'
    var_12 = 'DJANGO'
    var_13 = 'django.contrib'
    var_14 = module_0.module(var_13, var_6)
    assert var_14 == 'DJANGO'
    var_15 = '/path/to/src'
    var_16 = 'my_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'FIRSTPARTY'
    var_18 = 'my_namespace'
    var_19 = [var_18]
    var_20 = 'my_namespace.sub_module'
    var_21 = module_0.module(var_20, var_6)
    assert var_21 == 'FIRSTPARTY'



# Parsed testcases at query #31
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
    var_8 = 'django.apps'
    var_9 = module_0.module(var_8, var_6)
    assert var_9 == 'django'
    var_10 = '.local_module'
    var_11 = module_0.module(var_10)
    assert var_11 == 'LOCALFOLDER'
    var_12 = '.local.submodule'
    var_13 = module_0.module(var_12)
    assert var_13 == 'LOCALFOLDER'
    var_14 = '^test_.*'
    var_15 = 'TESTS'
    var_16 = 'test_example'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_example.submodule'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/project/src'
    var_21 = 'my_module'
    var_22 = module_0.module(var_21, var_6)
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'my_namespace'
    var_24 = [var_23]
    var_25 = 'my_namespace.submodule'
    var_26 = module_0.module(var_25, var_6)
    assert var_26 == 'FIRSTPARTY'
    var_27 = 'THIRDPARTY'
    var_28 = module_1.Config()
    var_29 = 'external_library'
    var_30 = module_0.module(var_29, var_28)
    assert var_30 == 'THIRDPARTY'



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
    var_4 = 'custom'
    var_5 = [var_4]
    var_6 = module_1.Config()
    var_7 = 'custom_module'
    var_8 = module_0.module(var_7, var_6)
    assert var_8 == 'custom'
    var_9 = '.local_module'
    var_10 = module_0.module(var_9)
    var_11 = '^test_.*'
    var_12 = 'TESTS'
    var_13 = 'test_example'
    var_14 = module_0.module(var_13, var_6)
    assert var_14 == 'TESTS'
    var_15 = '/path/to/src'
    var_16 = 'my_module'
    var_17 = module_0.module(var_16, var_6)
    var_18 = 'my_namespace'
    var_19 = [var_18]
    var_20 = 'my_namespace.submodule'
    var_21 = module_0.module(var_20, var_6)



# Parsed testcases at query #33
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
    var_2 = '.local_module'
    var_3 = module_0.module(var_2)
    assert var_3 == 'LOCALFOLDER'
    var_4 = 'django'
    var_5 = [var_4]
    var_6 = module_1.Config()
    var_7 = module_0.module(var_4, var_6)
    assert var_7 == 'django'
    var_8 = '^test_'
    var_9 = 'TESTS'
    var_10 = 'test_module'
    var_11 = module_0.module(var_10, var_6)
    assert var_11 == 'TESTS'
    var_12 = '/path/to/src'
    var_13 = 'my_module'
    var_14 = module_0.module(var_13, var_6)
    assert var_14 == 'FIRSTPARTY'
    var_15 = 'my_namespace'
    var_16 = [var_15]
    var_17 = 'my_namespace.sub_module'
    var_18 = module_0.module(var_17, var_6)
    assert var_18 == 'FIRSTPARTY'
    var_19 = 'non_existent_module'
    var_20 = module_0.module(var_19)
    assert var_20 == 'THIRDPARTY'



# Parsed testcases at query #34
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    var_2 = 'test*'
    var_3 = [var_2]
    var_4 = module_1.Config()
    var_5 = 'test_module'
    var_6 = module_0.module(var_5, var_4)
    assert var_6 == 'test*'
    var_7 = '.local_module'
    var_8 = module_0.module(var_7)
    var_9 = 'django.*'
    var_10 = 'DJANGO'
    var_11 = 'django.contrib'
    var_12 = module_0.module(var_11, var_4)
    assert var_12 == 'DJANGO'
    var_13 = 'mypackage'
    var_14 = var_0 / var_13
    var_15 = '__init__.py'
    var_16 = var_14 / var_15
    var_17 = [var_7]
    var_18 = module_1.Config()
    var_19 = module_0.module(var_13, var_18)



# Parsed testcases at query #35
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    var_2 = 'test*'
    var_3 = [var_2]
    var_4 = module_1.Config()
    var_5 = 'test_module'
    var_6 = module_0.module(var_5, var_4)
    assert var_6 == 'test*'
    var_7 = '.local_module'
    var_8 = module_0.module(var_7)
    var_9 = '^django'
    var_10 = 'DJANGO'
    var_11 = 'django.contrib'
    var_12 = module_0.module(var_11, var_4)
    assert var_12 == 'DJANGO'
    var_13 = 'myproject'
    var_14 = var_0 / var_13
    var_15 = 'module.py'
    var_16 = var_14 / var_15
    var_17 = [var_14]
    var_18 = module_1.Config()
    var_19 = module_0.module(var_13, var_18)



