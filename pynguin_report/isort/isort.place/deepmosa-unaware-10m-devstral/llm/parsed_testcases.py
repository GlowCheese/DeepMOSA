####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
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
    var_12 = '.sub.local_module'
    var_13 = module_0.module(var_12)
    var_14 = 'mycompany'
    var_15 = [var_14]
    var_16 = module_1.Config()
    var_17 = 'mycompany.utils'
    var_18 = module_0.module(var_17, var_16)
    assert var_18 == 'FIRSTPARTY'
    var_19 = '/path/to/src'
    var_20 = 'src_module'
    var_21 = module_0.module(var_20, var_16)
    assert var_21 == 'FIRSTPARTY'
    var_22 = 'THIRDPARTY'
    var_23 = module_1.Config()
    var_24 = 'unknown_module'
    var_25 = module_0.module(var_24, var_23)
    assert var_25 == 'THIRDPARTY'



# Parsed testcases at query #2
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
    var_9 = '^django.*'
    var_10 = 'DJANGO'
    var_11 = 'django.contrib'
    var_12 = module_0.module(var_11, var_4)
    assert var_12 == 'DJANGO'
    var_13 = '/src'
    var_14 = 'my_module'
    var_15 = module_0.module(var_14, var_4)



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
    var_8 = '.local_module'
    var_9 = module_0.module(var_8)
    assert var_9 == 'LOCALFOLDER'
    var_10 = '^test_.*'
    var_11 = 'TESTS'
    var_12 = 'test_example'
    var_13 = module_0.module(var_12, var_6)
    assert var_13 == 'TESTS'
    var_14 = '/project/src'
    var_15 = 'project'
    var_16 = module_0.module(var_15, var_6)
    assert var_16 == 'FIRSTPARTY'
    var_17 = [var_15]
    var_18 = 'project.submodule'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'FIRSTPARTY'
    var_20 = True
    var_21 = '.py'
    var_22 = [var_21]
    var_23 = frozenset(var_22)
    var_24 = module_0.module(var_18, var_6)
    assert var_24 == 'FIRSTPARTY'



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
    var_8 = 'django.apps'
    var_9 = module_0.module(var_8, var_6)
    assert var_9 == 'django'
    var_10 = '.local_module'
    var_11 = module_0.module(var_10)
    assert var_11 == 'LOCALFOLDER'
    var_12 = '..parent_module'
    var_13 = module_0.module(var_12)
    assert var_13 == 'LOCALFOLDER'
    var_14 = '^test_'
    var_15 = 'TESTS'
    var_16 = 'test_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_sub.module'
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
    var_29 = True
    var_30 = 'auto_namespace.submodule'
    var_31 = module_0.module(var_30, var_6)
    assert var_31 == 'FIRSTPARTY'



# Parsed testcases at query #5
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
    var_12 = '.local.package'
    var_13 = module_0.module(var_12)
    var_14 = '^test_.*'
    var_15 = 'TESTS'
    var_16 = 'test_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_package.submodule'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/project/src'
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
    var_29 = 'unknown_module'
    var_30 = module_0.module(var_29)
    assert var_30 == 'THIRDPARTY'



# Parsed testcases at query #6
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
    var_18 = False
    var_19 = 'namespace.submodule'
    var_20 = module_0.module(var_19, var_4)
    var_21 = 'nonexistent'
    var_22 = module_0.module(var_21, var_4)



# Parsed testcases at query #7
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
    var_14 = 'my_module'
    var_15 = module_0.module(var_14, var_4)
    var_16 = 'my_namespace'
    var_17 = [var_16]
    var_18 = 'my_namespace.submodule'
    var_19 = module_0.module(var_18, var_4)



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
    var_4 = 'test*'
    var_5 = [var_4]
    var_6 = module_1.Config()
    var_7 = 'test_module'
    var_8 = module_0.module(var_7, var_6)
    assert var_8 == 'test*'
    var_9 = '.local_module'
    var_10 = module_0.module(var_9)
    assert var_10 == 'LOCALFOLDER'
    var_11 = '^test_.*'
    var_12 = 'TESTS'
    var_13 = 'test_example'
    var_14 = module_0.module(var_13, var_6)
    assert var_14 == 'TESTS'
    var_15 = '/path/to/src'
    var_16 = 'src_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'FIRSTPARTY'
    var_18 = 'namespace'
    var_19 = [var_18]
    var_20 = True
    var_21 = 'namespace.submodule'
    var_22 = module_0.module(var_21, var_6)
    assert var_22 == 'FIRSTPARTY'



# Parsed testcases at query #9
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1)
    var_3 = 'sys'
    var_4 = module_1.module(var_3)
    var_5 = 'django'
    var_6 = module_1.module(var_5)
    var_7 = '.'
    var_8 = module_1.module(var_7)
    var_9 = '.local'
    var_10 = module_1.module(var_9)
    var_11 = 'myproject'
    var_12 = module_1.module(var_11)
    var_13 = 'myproject.utils'
    var_14 = module_1.module(var_13)
    var_15 = 'thirdparty'
    var_16 = module_1.module(var_15)
    var_17 = 'thirdparty.utils'
    var_18 = module_1.module(var_17)



# Parsed testcases at query #10
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
    var_14 = 'mymodule'
    var_15 = module_0.module(var_14, var_4)



# Parsed testcases at query #11
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
    var_10 = 'my_project'
    var_11 = [var_10]
    var_12 = module_1.Config()
    var_13 = 'my_project.utils'
    var_14 = module_0.module(var_13, var_12)
    assert var_14 == 'FIRSTPARTY'
    var_15 = '/path/to/project'
    var_16 = 'project'
    var_17 = module_0.module(var_16, var_12)
    assert var_17 == 'FIRSTPARTY'
    var_18 = 'project.subpackage'
    var_19 = [var_18]
    var_20 = 'project.subpackage.module'
    var_21 = module_0.module(var_20, var_12)
    assert var_21 == 'FIRSTPARTY'
    var_22 = 'non_existent_module'
    var_23 = module_0.module(var_22)
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
    var_12 = '^test_.*'
    var_13 = 'TESTS'
    var_14 = 'test_example'
    var_15 = module_0.module(var_14, var_6)
    assert var_15 == 'TESTS'
    var_16 = '/path/to/src'
    var_17 = 'my_module'
    var_18 = module_0.module(var_17, var_6)
    assert var_18 == 'FIRSTPARTY'
    var_19 = 'my_namespace'
    var_20 = [var_19]
    var_21 = 'my_namespace.submodule'
    var_22 = module_0.module(var_21, var_6)
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'unknown_module'
    var_24 = module_0.module(var_23)
    assert var_24 == 'THIRDPARTY'



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
    var_8 = 'django.apps'
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
    var_20 = 'src'
    var_21 = var_0 / var_20
    var_22 = 'my_module'
    var_23 = var_21 / var_22
    var_24 = var_21 / var_22
    var_25 = '__init__.py'
    var_26 = var_24 / var_25
    var_27 = [var_21]
    var_28 = module_1.Config()
    var_29 = module_0.module(var_22, var_28)
    assert var_29 == 'FIRSTPARTY'
    var_30 = 'my_module.submodule'
    var_31 = module_0.module(var_30, var_28)
    assert var_31 == 'FIRSTPARTY'
    var_32 = 'src'
    var_33 = var_0 / var_32
    var_34 = 'namespace'
    var_35 = var_33 / var_34
    var_36 = var_33 / var_34
    var_37 = 'module.py'
    var_38 = var_36 / var_37
    var_39 = [var_33]
    var_40 = [var_34]
    var_41 = module_1.Config()
    var_42 = 'namespace.module'
    var_43 = module_0.module(var_42, var_41)
    assert var_43 == 'FIRSTPARTY'
    var_44 = 'unknown_module'
    var_45 = module_0.module(var_44)
    assert var_45 == 'THIRDPARTY'



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
    var_4 = 'test*'
    var_5 = [var_4]
    var_6 = module_1.Config()
    var_7 = 'test_module'
    var_8 = module_0.module(var_7, var_6)
    assert var_8 == 'test*'
    var_9 = '.local_module'
    var_10 = module_0.module(var_9)
    var_11 = '^django'
    var_12 = 'DJANGO'
    var_13 = 'django.contrib'
    var_14 = module_0.module(var_13, var_6)
    assert var_14 == 'DJANGO'
    var_15 = '/path/to/src'
    var_16 = 'my_module'
    var_17 = module_0.module(var_16, var_6)
    var_18 = 'my_namespace'
    var_19 = [var_18]
    var_20 = 'my_namespace.sub_module'
    var_21 = module_0.module(var_20, var_6)
    var_22 = 'non_existent_module'
    var_23 = module_0.module(var_22)
    assert var_23 == 'THIRDPARTY'



# Parsed testcases at query #15
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1)
    var_3 = 'sys'
    var_4 = module_1.module(var_3)
    var_5 = 'django'
    var_6 = module_1.module(var_5)
    var_7 = 'numpy'
    var_8 = module_1.module(var_7)
    var_9 = 'my_project'
    var_10 = module_1.module(var_9)
    var_11 = '.local_module'
    var_12 = module_1.module(var_11)
    var_13 = 'unknown_module'
    var_14 = module_1.module(var_13)



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
    assert var_11 == 'LOCALFOLDER'
    var_12 = '.sub.local_module'
    var_13 = module_0.module(var_12)
    assert var_13 == 'LOCALFOLDER'
    var_14 = 'company'
    var_15 = [var_14]
    var_16 = module_1.Config()
    var_17 = 'company.module'
    var_18 = module_0.module(var_17)
    assert var_18 == 'FIRSTPARTY'
    var_19 = 'company.sub.module'
    var_20 = module_0.module(var_19)
    assert var_20 == 'FIRSTPARTY'
    var_21 = '/path/to/src'
    var_22 = 'src_module'
    var_23 = module_0.module(var_22, var_16)
    assert var_23 == 'FIRSTPARTY'
    var_24 = 'namespace'
    var_25 = [var_24]
    var_26 = 'namespace.submodule'
    var_27 = module_0.module(var_26, var_16)
    assert var_27 == 'FIRSTPARTY'
    var_28 = 'nonexistent_module'
    var_29 = module_0.module(var_28)
    assert var_29 == 'THIRDPARTY'



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
    var_5 = 'my_project'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'FIRSTPARTY'
    var_7 = '.local_module'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'LOCALFOLDER'
    var_9 = 'custom_pattern'
    var_10 = module_1.module(var_9, var_0)
    assert var_10 == 'DEFAULT'



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
    var_12 = '.sub.local_module'
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



# Parsed testcases at query #19
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
    var_11 = '^django'
    var_12 = 'DJANGO'
    var_13 = 'django.contrib'
    var_14 = module_0.module(var_13, var_6)
    assert var_14 == 'DJANGO'
    var_15 = '/path/to/src'
    var_16 = 'my_module'
    var_17 = module_0.module(var_16, var_6)
    var_18 = 'my_namespace'
    var_19 = [var_18]
    var_20 = 'my_namespace.submodule'
    var_21 = module_0.module(var_20, var_6)



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
    var_9 = '^django'
    var_10 = 'DJANGO'
    var_11 = 'django.contrib'
    var_12 = module_0.module(var_11, var_4)
    assert var_12 == 'DJANGO'
    var_13 = '/path/to/src'
    var_14 = 'mymodule'
    var_15 = module_0.module(var_14, var_4)
    var_16 = 'mynamespace'
    var_17 = [var_16]
    var_18 = 'mynamespace.submodule'
    var_19 = module_0.module(var_18, var_4)



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
    var_12 = '.sub.local_module'
    var_13 = module_0.module(var_12)
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
    var_23 = 'my_package'
    var_24 = module_0.module(var_23, var_6)
    assert var_24 == 'FIRSTPARTY'
    var_25 = 'namespace_pkg'
    var_26 = [var_25]
    var_27 = True
    var_28 = 'namespace_pkg.sub'
    var_29 = module_0.module(var_28, var_6)
    assert var_29 == 'FIRSTPARTY'
    var_30 = 'THIRDPARTY'
    var_31 = module_1.Config()
    var_32 = 'unknown_module'
    var_33 = module_0.module(var_32, var_31)
    assert var_33 == 'THIRDPARTY'



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
    var_4 = 'test*'
    var_5 = [var_4]
    var_6 = module_1.Config()
    var_7 = 'test_module'
    var_8 = module_0.module(var_7, var_6)
    assert var_8 == 'test*'
    var_9 = '.local_module'
    var_10 = module_0.module(var_9)
    var_11 = '^django.*'
    var_12 = 'DJANGO'
    var_13 = 'django.contrib'
    var_14 = module_0.module(var_13, var_6)
    assert var_14 == 'DJANGO'
    var_15 = '/project/src'
    var_16 = 'project'
    var_17 = module_0.module(var_16, var_6)
    var_18 = 'project.sub'
    var_19 = [var_18]
    var_20 = 'project.sub.module'
    var_21 = module_0.module(var_20, var_6)
    var_22 = True
    var_23 = module_0.module(var_20, var_6)



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
    var_4 = 'django'
    var_5 = [var_4]
    var_6 = module_1.Config()
    var_7 = module_0.module(var_4, var_6)
    assert var_7 == 'django'
    var_8 = '.local_module'
    var_9 = module_0.module(var_8)
    var_10 = '^test_.*'
    var_11 = 'TESTS'
    var_12 = 'test_example'
    var_13 = module_0.module(var_12, var_6)
    assert var_13 == 'TESTS'
    var_14 = '/project/src'
    var_15 = 'project'
    var_16 = module_0.module(var_15, var_6)
    var_17 = 'project.sub'
    var_18 = [var_17]
    var_19 = 'project.sub.module'
    var_20 = module_0.module(var_19, var_6)
    var_21 = 'non_existent_module'
    var_22 = module_0.module(var_21)
    assert var_22 == 'THIRDPARTY'



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
    var_4 = '.local_module'
    var_5 = module_0.module(var_4)
    assert var_5 == 'LOCALFOLDER'
    var_6 = '.another_local'
    var_7 = module_0.module(var_6)
    assert var_7 == 'LOCALFOLDER'
    var_8 = 'django'
    var_9 = 'pytest'
    var_10 = [var_8, var_9]
    var_11 = module_1.Config()
    var_12 = module_0.module(var_8, var_11)
    assert var_12 == 'django'
    var_13 = module_0.module(var_9, var_11)
    assert var_13 == 'pytest'
    var_14 = 'django.contrib'
    var_15 = module_0.module(var_14, var_11)
    assert var_15 == 'django'
    var_16 = 'pytest.cov'
    var_17 = module_0.module(var_16, var_11)
    assert var_17 == 'pytest'
    var_18 = '^django'
    var_19 = 'DJANGO'
    var_20 = module_0.module(var_8, var_11)
    assert var_20 == 'DJANGO'
    var_21 = module_0.module(var_14, var_11)
    assert var_21 == 'DJANGO'
    var_22 = 'my_project'
    var_23 = var_0 / var_22
    var_24 = 'module.py'
    var_25 = var_23 / var_24
    var_26 = [var_23]
    var_27 = module_1.Config()
    var_28 = module_0.module(var_22)
    assert var_28 == 'FIRSTPARTY'
    var_29 = 'my_project.module'
    var_30 = module_0.module(var_29)
    assert var_30 == 'FIRSTPARTY'
    var_31 = 'namespace'
    var_32 = var_0 / var_31
    var_33 = 'submodule.py'
    var_34 = var_32 / var_33
    var_35 = [var_32]
    var_36 = [var_31]
    var_37 = module_1.Config()
    var_38 = module_0.module(var_31)
    assert var_38 == 'FIRSTPARTY'
    var_39 = 'namespace.submodule'
    var_40 = module_0.module(var_39)
    assert var_40 == 'FIRSTPARTY'
    var_41 = 'auto_namespace'
    var_42 = var_0 / var_41
    var_43 = 'submodule.py'
    var_44 = var_42 / var_43
    var_45 = [var_42]
    var_46 = True
    var_47 = module_1.Config()
    var_48 = module_0.module(var_41)
    assert var_48 == 'FIRSTPARTY'
    var_49 = 'auto_namespace.submodule'
    var_50 = module_0.module(var_49)
    assert var_50 == 'FIRSTPARTY'



# Parsed testcases at query #25
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
    var_14 = 'mycompany'
    var_15 = [var_14]
    var_16 = module_1.Config()
    var_17 = 'mycompany.utils'
    var_18 = module_0.module(var_17, var_16)
    assert var_18 == 'FIRSTPARTY'
    var_19 = 'mycompany.core.models'
    var_20 = module_0.module(var_19, var_16)
    assert var_20 == 'FIRSTPARTY'
    var_21 = '/home/user/projects/myproject'
    var_22 = 'myproject'
    var_23 = module_0.module(var_22, var_16)
    assert var_23 == 'FIRSTPARTY'
    var_24 = 'myproject.utils'
    var_25 = module_0.module(var_24, var_16)
    assert var_25 == 'FIRSTPARTY'
    var_26 = '/home/user/projects/namespace_pkg'
    var_27 = 'namespace_pkg'
    var_28 = [var_27]
    var_29 = 'namespace_pkg.submodule'
    var_30 = module_0.module(var_29, var_16)
    assert var_30 == 'FIRSTPARTY'
    var_31 = 'THIRDPARTY'
    var_32 = module_1.Config()
    var_33 = 'some_unknown_module'
    var_34 = module_0.module(var_33, var_32)
    assert var_34 == 'THIRDPARTY'



# Parsed testcases at query #26
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
    var_11 = 'mycompany'
    var_12 = [var_11]
    var_13 = module_1.Config()
    var_14 = 'mycompany.module'
    var_15 = module_0.module(var_14)
    assert var_15 == 'FIRSTPARTY'
    var_16 = '/path/to/src'
    var_17 = 'src_module'
    var_18 = module_0.module(var_17, var_13)
    assert var_18 == 'FIRSTPARTY'
    var_19 = 'namespace_pkg'
    var_20 = [var_19]
    var_21 = True
    var_22 = 'namespace_pkg.module'
    var_23 = module_0.module(var_22, var_13)
    assert var_23 == 'FIRSTPARTY'



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
    var_18 = 'test_example.submodule'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = 'src'
    var_21 = var_0 / var_20
    var_22 = 'mymodule'
    var_23 = var_21 / var_22
    var_24 = '__init__.py'
    var_25 = var_23 / var_24
    var_26 = [var_21]
    var_27 = module_1.Config()
    var_28 = module_0.module(var_22, var_27)
    assert var_28 == 'FIRSTPARTY'
    var_29 = 'mymodule.submodule'
    var_30 = module_0.module(var_29, var_27)
    assert var_30 == 'FIRSTPARTY'
    var_31 = 'src'
    var_32 = var_0 / var_31
    var_33 = 'namespace'
    var_34 = var_32 / var_33
    var_35 = 'module.py'
    var_36 = var_34 / var_35
    var_37 = [var_32]
    var_38 = [var_33]
    var_39 = module_1.Config()
    var_40 = 'namespace.module'
    var_41 = module_0.module(var_40, var_39)
    assert var_41 == 'FIRSTPARTY'
    var_42 = 'unknown_module'
    var_43 = module_0.module(var_42)
    assert var_43 == 'THIRDPARTY'



# Parsed testcases at query #28
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
    var_11 = 'django.core'
    var_12 = module_0.module(var_11, var_4)
    assert var_12 == 'DJANGO'
    var_13 = '/path/to/src'
    var_14 = 'my_module'
    var_15 = module_0.module(var_14, var_4)
    var_16 = 'my_namespace'
    var_17 = [var_16]
    var_18 = 'my_namespace.sub_module'
    var_19 = module_0.module(var_18, var_4)



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
    var_12 = '.local.submodule'
    var_13 = module_0.module(var_12)
    assert var_13 == 'LOCALFOLDER'
    var_14 = '^test_.*'
    var_15 = 'TESTS'
    var_16 = 'test_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_utils.helper'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/project/src'
    var_21 = 'project'
    var_22 = module_0.module(var_21, var_6)
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'project.utils'
    var_24 = module_0.module(var_23, var_6)
    assert var_24 == 'FIRSTPARTY'
    var_25 = [var_21]
    var_26 = 'project.subpackage'
    var_27 = module_0.module(var_26, var_6)
    assert var_27 == 'FIRSTPARTY'
    var_28 = 'unknown_module'
    var_29 = module_0.module(var_28)
    assert var_29 == 'THIRDPARTY'



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
    var_11 = '^django.*'
    var_12 = 'DJANGO'
    var_13 = 'django.contrib.auth'
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
    var_22 = 'non_existent_module'
    var_23 = module_0.module(var_22)
    assert var_23 == 'THIRDPARTY'



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
    assert var_11 == 'LOCALFOLDER'
    var_12 = '.sub.local_module'
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
    var_8 = '.local_module'
    var_9 = module_0.module(var_8)
    var_10 = 'mycompany'
    var_11 = [var_10]
    var_12 = module_1.Config()
    var_13 = 'mycompany.utils'
    var_14 = module_0.module(var_13)
    assert var_14 == 'FIRSTPARTY'
    var_15 = '/path/to/src'
    var_16 = 'src_module'
    var_17 = module_0.module(var_16)
    assert var_17 == 'FIRSTPARTY'
    var_18 = 'namespace_pkg'
    var_19 = [var_18]
    var_20 = 'namespace_pkg.submodule'
    var_21 = module_0.module(var_20)
    assert var_21 == 'FIRSTPARTY'
    var_22 = 'nonexistent_module'
    var_23 = module_0.module(var_22)
    assert var_23 == 'THIRDPARTY'



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
    var_23 = 'project.namespace'
    var_24 = [var_23]
    var_25 = True
    var_26 = 'project.namespace.submodule'
    var_27 = module_0.module(var_26, var_6)



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
    var_9 = '^django'
    var_10 = 'DJANGO'
    var_11 = 'django.contrib'
    var_12 = module_0.module(var_11, var_4)
    assert var_12 == 'DJANGO'
    var_13 = 'src'
    var_14 = var_0 / var_13
    var_15 = 'my_module.py'
    var_16 = var_14 / var_15
    var_17 = '# test'
    var_18 = [var_14]
    var_19 = module_1.Config()
    var_20 = 'my_module'
    var_21 = module_0.module(var_20, var_19)
    var_22 = 'src'
    var_23 = var_0 / var_22
    var_24 = 'namespace'
    var_25 = var_23 / var_24
    var_26 = var_23 / var_24
    var_27 = 'module.py'
    var_28 = var_26 / var_27
    var_29 = '# test'
    var_30 = [var_23]
    var_31 = [var_24]
    var_32 = module_1.Config()
    var_33 = 'namespace.module'
    var_34 = module_0.module(var_33, var_32)



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
    var_13 = '/path/to/src'
    var_14 = 'my_module'
    var_15 = module_0.module(var_14, var_4)
    var_16 = 'my_namespace'
    var_17 = [var_16]
    var_18 = 'my_namespace.submodule'
    var_19 = module_0.module(var_18, var_4)



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
    var_12 = '.another_local'
    var_13 = module_0.module(var_12)
    var_14 = '^test.*'
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
    var_23 = 'my_namespace'
    var_24 = [var_23]
    var_25 = True
    var_26 = 'my_namespace.submodule'
    var_27 = module_0.module(var_26, var_6)
    var_28 = 'THIRDPARTY'
    var_29 = module_1.Config()
    var_30 = 'unknown_module'
    var_31 = module_0.module(var_30, var_29)
    assert var_31 == 'THIRDPARTY'



# Parsed testcases at query #37
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
    var_12 = 'test_example'
    var_13 = module_0.module(var_12, var_6)
    assert var_13 == 'TESTS'
    var_14 = '/path/to/src'
    var_15 = 'module'
    var_16 = module_0.module(var_15, var_6)
    var_17 = 'namespace'
    var_18 = [var_17]
    var_19 = True
    var_20 = 'namespace.submodule'
    var_21 = module_0.module(var_20, var_6)
    var_22 = 'nonexistent_module'
    var_23 = module_0.module(var_22)
    assert var_23 == 'THIRDPARTY'



# Parsed testcases at query #38
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
    var_10 = '^test_.*'
    var_11 = 'TESTS'
    var_12 = 'test_example'
    var_13 = module_0.module(var_12, var_6)
    assert var_13 == 'TESTS'
    var_14 = '/path/to/project'
    var_15 = 'project_module'
    var_16 = module_0.module(var_15, var_6)
    var_17 = 'project.nested'
    var_18 = [var_17]
    var_19 = 'project.nested.module'
    var_20 = module_0.module(var_19, var_6)
    var_21 = 'unknown_module'
    var_22 = module_0.module(var_21)



# Parsed testcases at query #39
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
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'project.module'
    var_24 = module_0.module(var_23, var_6)
    assert var_24 == 'FIRSTPARTY'
    var_25 = [var_21]
    var_26 = 'project.submodule'
    var_27 = module_0.module(var_26, var_6)
    assert var_27 == 'FIRSTPARTY'
    var_28 = 'THIRDPARTY'
    var_29 = module_1.Config()
    var_30 = 'unknown_module'
    var_31 = module_0.module(var_30, var_29)
    assert var_31 == 'THIRDPARTY'



# Parsed testcases at query #40
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
    var_13 = 'src'
    var_14 = var_0 / var_13
    var_15 = 'mymodule'
    var_16 = var_14 / var_15
    var_17 = '__init__.py'
    var_18 = var_16 / var_17
    var_19 = [var_14]
    var_20 = module_1.Config()
    var_21 = module_0.module(var_15, var_20)



# Parsed testcases at query #41
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
    var_12 = '^test_.*'
    var_13 = 'TESTS'
    var_14 = 'test_example'
    var_15 = module_0.module(var_14, var_6)
    assert var_15 == 'TESTS'
    var_16 = '/project/src'
    var_17 = 'project'
    var_18 = module_0.module(var_17, var_6)
    var_19 = 'project.sub'
    var_20 = [var_19]
    var_21 = True
    var_22 = 'project.sub.module'
    var_23 = module_0.module(var_22, var_6)



# Parsed testcases at query #42
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
    var_15 = 'project'
    var_16 = module_0.module(var_15, var_6)
    assert var_16 == 'FIRSTPARTY'
    var_17 = 'project.sub'
    var_18 = [var_17]
    var_19 = True
    var_20 = 'project.sub.module'
    var_21 = module_0.module(var_20, var_6)
    assert var_21 == 'FIRSTPARTY'
    var_22 = 'nonexistent_module'
    var_23 = module_0.module(var_22)
    assert var_23 == 'THIRDPARTY'



# Parsed testcases at query #43
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
    var_14 = 'mycompany'
    var_15 = [var_14]
    var_16 = module_1.Config()
    var_17 = 'mycompany.utils'
    var_18 = module_0.module(var_17)
    assert var_18 == 'FIRSTPARTY'
    var_19 = 'mycompany.core'
    var_20 = module_0.module(var_19)
    assert var_20 == 'FIRSTPARTY'
    var_21 = '/path/to/project'
    var_22 = 'project'
    var_23 = module_0.module(var_22, var_16)
    assert var_23 == 'FIRSTPARTY'
    var_24 = 'project.submodule'
    var_25 = module_0.module(var_24, var_16)
    assert var_25 == 'FIRSTPARTY'
    var_26 = '/path/to/namespace'
    var_27 = 'namespace'
    var_28 = [var_27]
    var_29 = 'namespace.sub'
    var_30 = module_0.module(var_29, var_16)
    assert var_30 == 'FIRSTPARTY'
    var_31 = 'unknown_module'
    var_32 = module_0.module(var_31)
    assert var_32 == 'THIRDPARTY'



# Parsed testcases at query #44
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
    var_16 = 'test_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_sub.submodule'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/path/to/src'
    var_21 = 'src_module'
    var_22 = module_0.module(var_21, var_6)
    var_23 = 'namespace'
    var_24 = [var_23]
    var_25 = 'namespace.submodule'
    var_26 = module_0.module(var_25, var_6)
    var_27 = True
    var_28 = 'auto_namespace.submodule'
    var_29 = module_0.module(var_28, var_6)



# Parsed testcases at query #45
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
    var_14 = 'my_module'
    var_15 = module_0.module(var_14, var_4)
    var_16 = 'my_namespace'
    var_17 = [var_16]
    var_18 = False
    var_19 = 'my_namespace.sub_module'
    var_20 = module_0.module(var_19, var_4)



# Parsed testcases at query #46
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
    var_14 = '^test.*'
    var_15 = 'TESTS'
    var_16 = 'test_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_package.submodule'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = 'src'
    var_21 = var_0 / var_20
    var_22 = 'my_package'
    var_23 = var_21 / var_22
    var_24 = var_21 / var_22
    var_25 = '__init__.py'
    var_26 = var_24 / var_25
    var_27 = [var_21]
    var_28 = module_1.Config()
    var_29 = module_0.module(var_22, var_28)
    var_30 = 'my_package.submodule'
    var_31 = module_0.module(var_30, var_28)
    var_32 = 'src'
    var_33 = var_0 / var_32
    var_34 = 'namespace'
    var_35 = var_33 / var_34
    var_36 = [var_33]
    var_37 = [var_34]
    var_38 = module_1.Config()
    var_39 = module_0.module(var_34, var_38)
    var_40 = 'namespace.submodule'
    var_41 = module_0.module(var_40, var_38)
    var_42 = 'unknown_module'
    var_43 = module_0.module(var_42)



# Parsed testcases at query #47
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
    var_20 = 'src'
    var_21 = var_0 / var_20
    var_22 = 'my_package'
    var_23 = var_21 / var_22
    var_24 = var_21 / var_22
    var_25 = '__init__.py'
    var_26 = var_24 / var_25
    var_27 = [var_21]
    var_28 = module_1.Config()
    var_29 = module_0.module(var_22, var_28)
    var_30 = 'my_package.module'
    var_31 = module_0.module(var_30, var_28)
    var_32 = 'src'
    var_33 = var_0 / var_32
    var_34 = 'namespace'
    var_35 = var_33 / var_34
    var_36 = var_33 / var_34
    var_37 = 'module.py'
    var_38 = var_36 / var_37
    var_39 = [var_33]
    var_40 = True
    var_41 = module_1.Config()
    var_42 = 'namespace.module'
    var_43 = module_0.module(var_42, var_41)
    var_44 = 'THIRDPARTY'
    var_45 = module_1.Config()
    var_46 = 'unknown_module'
    var_47 = module_0.module(var_46, var_45)
    assert var_47 == 'THIRDPARTY'



# Parsed testcases at query #48
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



# Parsed testcases at query #49
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



# Parsed testcases at query #50
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
    var_18 = 'namespace.submodule'
    var_19 = module_0.module(var_18, var_4)
    assert var_19 == 'FIRSTPARTY'



# Parsed testcases at query #51
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
    var_20 = 'unknown_module'
    var_21 = module_0.module(var_20)



# Parsed testcases at query #52
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
    var_10 = '.local_module'
    var_11 = module_0.module(var_10)
    var_12 = '.another_local'
    var_13 = module_0.module(var_12)
    var_14 = '^django.*'
    var_15 = 'DJANGO'
    var_16 = 'django.contrib'
    var_17 = module_0.module(var_16, var_7)
    assert var_17 == 'DJANGO'
    var_18 = 'django.core'
    var_19 = module_0.module(var_18, var_7)
    assert var_19 == 'DJANGO'
    var_20 = '/path/to/src'
    var_21 = 'mymodule'
    var_22 = module_0.module(var_21, var_7)
    var_23 = 'my_namespace'
    var_24 = [var_23]
    var_25 = 'my_namespace.submodule'
    var_26 = module_0.module(var_25, var_7)



# Parsed testcases at query #53
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
    var_14 = 'my_module'
    var_15 = module_0.module(var_14, var_4)
    var_16 = 'my_namespace'
    var_17 = [var_16]
    var_18 = 'my_namespace.submodule'
    var_19 = module_0.module(var_18, var_4)



# Parsed testcases at query #54
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
    var_10 = '^test_.*'
    var_11 = 'TESTS'
    var_12 = 'test_example'
    var_13 = module_0.module(var_12, var_6)
    assert var_13 == 'TESTS'
    var_14 = '/path/to/src'
    var_15 = 'src_module'
    var_16 = module_0.module(var_15, var_6)
    var_17 = 'namespace_pkg'
    var_18 = [var_17]
    var_19 = 'namespace_pkg.submodule'
    var_20 = module_0.module(var_19, var_6)



# Parsed testcases at query #55
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



# Parsed testcases at query #56
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
    var_19 = 'myproject.module'
    var_20 = module_0.module(var_19, var_18)
    var_21 = 'namespace'
    var_22 = var_0 / var_21
    var_23 = [var_22]
    var_24 = [var_21]
    var_25 = module_1.Config()
    var_26 = 'namespace.submodule'
    var_27 = module_0.module(var_26, var_25)



# Parsed testcases at query #57
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
    var_18 = True
    var_19 = 'namespace.submodule'
    var_20 = module_0.module(var_19, var_4)



# Parsed testcases at query #58
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
    var_14 = '^django'
    var_15 = 'DJANGO'
    var_16 = module_0.module(var_4, var_6)
    assert var_16 == 'DJANGO'
    var_17 = module_0.module(var_8, var_6)
    assert var_17 == 'DJANGO'
    var_18 = 'my_project'
    var_19 = var_0 / var_18
    var_20 = 'module.py'
    var_21 = var_19 / var_20
    var_22 = '# test'
    var_23 = [var_19]
    var_24 = module_1.Config()
    var_25 = 'module'
    var_26 = module_0.module(var_25, var_24)
    assert var_26 == 'FIRSTPARTY'
    var_27 = 'module.submodule'
    var_28 = module_0.module(var_27, var_24)
    assert var_28 == 'FIRSTPARTY'
    var_29 = 'namespace_package'
    var_30 = var_0 / var_29
    var_31 = [var_30]
    var_32 = [var_29]
    var_33 = module_1.Config()
    var_34 = module_0.module(var_29, var_33)
    assert var_34 == 'FIRSTPARTY'
    var_35 = 'namespace_package.submodule'
    var_36 = module_0.module(var_35, var_33)
    assert var_36 == 'FIRSTPARTY'



# Parsed testcases at query #59
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
    var_18 = False
    var_19 = 'namespace.submodule'
    var_20 = module_0.module(var_19, var_4)



# Parsed testcases at query #60
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
    var_12 = 'company'
    var_13 = [var_12]
    var_14 = module_1.Config()
    var_15 = 'company.module'
    var_16 = module_0.module(var_15, var_14)
    assert var_16 == 'FIRSTPARTY'
    var_17 = '/path/to/project'
    var_18 = 'project'
    var_19 = module_0.module(var_18, var_14)
    assert var_19 == 'FIRSTPARTY'
    var_20 = 'project.sub'
    var_21 = [var_20]
    var_22 = 'project.sub.module'
    var_23 = module_0.module(var_22, var_14)
    assert var_23 == 'FIRSTPARTY'
    var_24 = True
    var_25 = module_0.module(var_22, var_14)
    assert var_25 == 'FIRSTPARTY'



# Parsed testcases at query #61
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
    var_19 = 'mycompany.core.models'
    var_20 = module_0.module(var_19)
    assert var_20 == 'FIRSTPARTY'
    var_21 = '/project/src'
    var_22 = 'project'
    var_23 = module_0.module(var_22, var_16)
    assert var_23 == 'FIRSTPARTY'
    var_24 = 'project.utils'
    var_25 = module_0.module(var_24, var_16)
    assert var_25 == 'FIRSTPARTY'
    var_26 = 'project.namespace'
    var_27 = [var_26]
    var_28 = 'project.namespace.sub'
    var_29 = module_0.module(var_28, var_16)
    assert var_29 == 'FIRSTPARTY'
    var_30 = 'unknown_module'
    var_31 = module_0.module(var_30)
    assert var_31 == 'THIRDPARTY'



# Parsed testcases at query #62
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



# Parsed testcases at query #63
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
    assert var_5 == 'LOCALFOLDER'
    var_6 = '.another_local'
    var_7 = module_0.module(var_6)
    assert var_7 == 'LOCALFOLDER'
    var_8 = 'django'
    var_9 = [var_8]
    var_10 = module_1.Config()
    var_11 = module_0.module(var_8, var_10)
    assert var_11 == 'django'
    var_12 = 'django.apps'
    var_13 = module_0.module(var_12, var_10)
    assert var_13 == 'django'
    var_14 = '^test_'
    var_15 = 'TESTS'
    var_16 = 'test_example'
    var_17 = module_0.module(var_16, var_10)
    assert var_17 == 'TESTS'
    var_18 = 'test_another'
    var_19 = module_0.module(var_18, var_10)
    assert var_19 == 'TESTS'
    var_20 = '/project/src'
    var_21 = 'project'
    var_22 = module_0.module(var_21, var_10)
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'project.module'
    var_24 = module_0.module(var_23, var_10)
    assert var_24 == 'FIRSTPARTY'
    var_25 = 'project.namespace'
    var_26 = [var_25]
    var_27 = 'project.namespace.module'
    var_28 = module_0.module(var_27, var_10)
    assert var_28 == 'FIRSTPARTY'
    var_29 = 'THIRDPARTY'
    var_30 = module_1.Config()
    var_31 = 'unknown_module'
    var_32 = module_0.module(var_31, var_30)
    assert var_32 == 'THIRDPARTY'



# Parsed testcases at query #64
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
    var_13 = '/path/to/src'
    var_14 = 'mymodule'
    var_15 = module_0.module(var_14, var_4)



# Parsed testcases at query #65
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
    var_23 = [var_22]
    var_24 = [var_21]
    var_25 = module_1.Config()
    var_26 = 'namespace.submodule'
    var_27 = module_0.module(var_26, var_25)
    var_28 = 'nonexistent_module'
    var_29 = module_0.module(var_28)



# Parsed testcases at query #66
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



# Parsed testcases at query #67
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
    var_13 = 'src'
    var_14 = var_0 / var_13
    var_15 = 'my_module'
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



# Parsed testcases at query #68
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
    var_16 = 'test_utils'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_another'
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
    var_29 = 'namespace'
    var_30 = var_0 / var_29
    var_31 = 'submodule.py'
    var_32 = var_30 / var_31
    var_33 = [var_30]
    var_34 = [var_29]
    var_35 = module_1.Config()
    var_36 = 'namespace.submodule'
    var_37 = module_0.module(var_36, var_35)



# Parsed testcases at query #69
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
    var_9 = '^test_'
    var_10 = 'TEST'
    var_11 = module_0.module(var_5, var_4)
    assert var_11 == 'TEST'
    var_12 = '/path/to/src'
    var_13 = 'src_module'
    var_14 = module_0.module(var_13, var_4)
    assert var_14 == 'FIRSTPARTY'
    var_15 = 'namespace'
    var_16 = [var_15]
    var_17 = True
    var_18 = 'namespace.submodule'
    var_19 = module_0.module(var_18, var_4)
    assert var_19 == 'FIRSTPARTY'



# Parsed testcases at query #70
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
    var_20 = 'src'
    var_21 = var_0 / var_20
    var_22 = 'my_package'
    var_23 = var_21 / var_22
    var_24 = var_21 / var_22
    var_25 = '__init__.py'
    var_26 = var_24 / var_25
    var_27 = ''
    var_28 = [var_21]
    var_29 = module_1.Config()
    var_30 = module_0.module(var_22, var_29)
    assert var_30 == 'FIRSTPARTY'
    var_31 = 'my_package.submodule'
    var_32 = module_0.module(var_31, var_29)
    assert var_32 == 'FIRSTPARTY'
    var_33 = 'src'
    var_34 = var_0 / var_33
    var_35 = 'namespace'
    var_36 = var_34 / var_35
    var_37 = var_34 / var_35
    var_38 = 'module.py'
    var_39 = var_37 / var_38
    var_40 = ''
    var_41 = [var_34]
    var_42 = [var_35]
    var_43 = module_1.Config()
    var_44 = 'namespace.module'
    var_45 = module_0.module(var_44, var_43)
    assert var_45 == 'FIRSTPARTY'



# Parsed testcases at query #71
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    var_2 = 'numpy'
    var_3 = 'pandas'
    var_4 = [var_2, var_3]
    var_5 = module_1.Config()
    var_6 = module_0.module(var_2, var_5)
    assert var_6 == 'numpy'
    var_7 = 'pandas.core'
    var_8 = module_0.module(var_7, var_5)
    assert var_8 == 'pandas'
    var_9 = '.local_module'
    var_10 = module_0.module(var_9)
    var_11 = '^django'
    var_12 = 'DJANGO'
    var_13 = 'django'
    var_14 = module_0.module(var_13, var_5)
    assert var_14 == 'DJANGO'
    var_15 = 'django.contrib'
    var_16 = module_0.module(var_15, var_5)
    assert var_16 == 'DJANGO'
    var_17 = 'my_project'
    var_18 = var_0 / var_17
    var_19 = 'module.py'
    var_20 = var_18 / var_19
    var_21 = '# test'
    var_22 = [var_18]
    var_23 = module_1.Config()
    var_24 = module_0.module(var_17, var_23)



# Parsed testcases at query #72
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
    var_14 = '^test_.*'
    var_15 = 'TESTS'
    var_16 = 'test_example'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_example.submodule'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/path/to/src'
    var_21 = 'module'
    var_22 = module_0.module(var_21, var_6)
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'namespace'
    var_24 = [var_23]
    var_25 = True
    var_26 = 'namespace.submodule'
    var_27 = module_0.module(var_26, var_6)
    assert var_27 == 'FIRSTPARTY'
    var_28 = 'THIRDPARTY'
    var_29 = module_1.Config()
    var_30 = 'unknown_module'
    var_31 = module_0.module(var_30, var_29)
    assert var_31 == 'THIRDPARTY'



# Parsed testcases at query #73
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
    var_9 = '^django.*'
    var_10 = 'DJANGO'
    var_11 = 'django.contrib'
    var_12 = module_0.module(var_11, var_4)
    assert var_12 == 'DJANGO'
    var_13 = '/path/to/src'
    var_14 = 'src_module'
    var_15 = module_0.module(var_14, var_4)
    var_16 = 'namespace'
    var_17 = [var_16]
    var_18 = True
    var_19 = 'namespace.submodule'
    var_20 = module_0.module(var_19, var_4)



# Parsed testcases at query #74
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
    var_21 = 'non_existent_module'
    var_22 = module_0.module(var_21)
    assert var_22 == 'THIRDPARTY'



# Parsed testcases at query #75
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    var_2 = 'sys'
    var_3 = module_0.module(var_2)
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
    var_14 = 'django.contrib'
    var_15 = module_0.module(var_14, var_11)
    assert var_15 == 'django'
    var_16 = 'flask.ext'
    var_17 = module_0.module(var_16, var_11)
    assert var_17 == 'flask'
    var_18 = '^test_.*'
    var_19 = 'TESTS'
    var_20 = 'test_module'
    var_21 = module_0.module(var_20, var_11)
    assert var_21 == 'TESTS'
    var_22 = 'test_another'
    var_23 = module_0.module(var_22, var_11)
    assert var_23 == 'TESTS'
    var_24 = '/path/to/src'
    var_25 = 'my_module'
    var_26 = module_0.module(var_25, var_11)
    var_27 = 'my_package.submodule'
    var_28 = module_0.module(var_27, var_11)
    var_29 = 'my_namespace'
    var_30 = [var_29]
    var_31 = 'my_namespace.submodule'
    var_32 = module_0.module(var_31, var_11)
    var_33 = True
    var_34 = 'auto_namespace.submodule'
    var_35 = module_0.module(var_34, var_11)



# Parsed testcases at query #76
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
    var_13 = 'src'
    var_14 = var_0 / var_13
    var_15 = 'my_package'
    var_16 = var_14 / var_15
    var_17 = var_14 / var_15
    var_18 = '__init__.py'
    var_19 = var_17 / var_18
    var_20 = ''
    var_21 = [var_14]
    var_22 = module_1.Config()
    var_23 = module_0.module(var_15, var_22)
    assert var_23 == 'FIRSTPARTY'
    var_24 = 'src'
    var_25 = var_0 / var_24
    var_26 = 'namespace'
    var_27 = var_25 / var_26
    var_28 = var_25 / var_26
    var_29 = 'module.py'
    var_30 = var_28 / var_29
    var_31 = ''
    var_32 = [var_25]
    var_33 = [var_26]
    var_34 = module_1.Config()
    var_35 = 'namespace.module'
    var_36 = module_0.module(var_35, var_34)
    assert var_36 == 'FIRSTPARTY'



# Parsed testcases at query #77
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
    var_14 = '^test_'
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
    var_29 = True
    var_30 = module_0.module(var_27, var_6)
    assert var_30 == 'FIRSTPARTY'
    var_31 = 'unknown_module'
    var_32 = module_0.module(var_31)
    assert var_32 == 'THIRDPARTY'
    var_33 = 'unknown.package'
    var_34 = module_0.module(var_33)
    assert var_34 == 'THIRDPARTY'



# Parsed testcases at query #78
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
    var_4 = '.local'
    var_5 = module_0.module(var_4)
    assert var_5 == 'LOCALFOLDER'
    var_6 = '.module'
    var_7 = module_0.module(var_6)
    assert var_7 == 'LOCALFOLDER'
    var_8 = 'django'
    var_9 = [var_8]
    var_10 = module_1.Config()
    var_11 = module_0.module(var_8, var_10)
    assert var_11 == 'django'
    var_12 = 'django.contrib'
    var_13 = module_0.module(var_12, var_10)
    assert var_13 == 'django'
    var_14 = '^test.*'
    var_15 = 'TESTS'
    var_16 = 'test_module'
    var_17 = module_0.module(var_16, var_10)
    assert var_17 == 'TESTS'
    var_18 = 'test_package.submodule'
    var_19 = module_0.module(var_18, var_10)
    assert var_19 == 'TESTS'
    var_20 = 'src'
    var_21 = var_0 / var_20
    var_22 = 'my_package'
    var_23 = var_21 / var_22
    var_24 = var_21 / var_22
    var_25 = '__init__.py'
    var_26 = var_24 / var_25
    var_27 = [var_21]
    var_28 = module_1.Config()
    var_29 = module_0.module(var_22, var_28)
    assert var_29 == 'FIRSTPARTY'
    var_30 = 'my_package.submodule'
    var_31 = module_0.module(var_30, var_28)
    assert var_31 == 'FIRSTPARTY'
    var_32 = 'src'
    var_33 = var_0 / var_32
    var_34 = 'namespace'
    var_35 = var_33 / var_34
    var_36 = var_33 / var_34
    var_37 = 'module.py'
    var_38 = var_36 / var_37
    var_39 = [var_33]
    var_40 = [var_34]
    var_41 = module_1.Config()
    var_42 = 'namespace.module'
    var_43 = module_0.module(var_42, var_41)
    assert var_43 == 'FIRSTPARTY'
    var_44 = 'src'
    var_45 = var_0 / var_44
    var_46 = 'auto_namespace'
    var_47 = var_45 / var_46
    var_48 = var_45 / var_46
    var_49 = 'module.py'
    var_50 = var_48 / var_49
    var_51 = [var_45]
    var_52 = True
    var_53 = module_1.Config()
    var_54 = 'auto_namespace.module'
    var_55 = module_0.module(var_54, var_53)
    assert var_55 == 'FIRSTPARTY'
    var_56 = 'unknown_module'
    var_57 = module_0.module(var_56)
    assert var_57 == 'THIRDPARTY'



# Parsed testcases at query #79
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
    var_20 = 'src'
    var_21 = var_0 / var_20
    var_22 = 'my_module'
    var_23 = var_21 / var_22
    var_24 = '__init__.py'
    var_25 = var_23 / var_24
    var_26 = ''
    var_27 = [var_21]
    var_28 = module_1.Config()
    var_29 = module_0.module(var_22, var_28)
    var_30 = 'my_module.submodule'
    var_31 = module_0.module(var_30, var_28)
    var_32 = 'src'
    var_33 = var_0 / var_32
    var_34 = 'namespace'
    var_35 = var_33 / var_34
    var_36 = 'module.py'
    var_37 = var_35 / var_36
    var_38 = ''
    var_39 = [var_33]
    var_40 = [var_34]
    var_41 = module_1.Config()
    var_42 = module_0.module(var_34, var_41)
    var_43 = 'namespace.module'
    var_44 = module_0.module(var_43, var_41)
    var_45 = 'src'
    var_46 = var_0 / var_45
    var_47 = 'auto_namespace'
    var_48 = var_46 / var_47
    var_49 = 'module.py'
    var_50 = var_48 / var_49
    var_51 = ''
    var_52 = [var_46]
    var_53 = True
    var_54 = module_1.Config()
    var_55 = module_0.module(var_47, var_54)
    var_56 = 'auto_namespace.module'
    var_57 = module_0.module(var_56, var_54)
    var_58 = 'THIRDPARTY'
    var_59 = module_1.Config()
    var_60 = 'unknown_module'
    var_61 = module_0.module(var_60, var_59)
    assert var_61 == 'THIRDPARTY'



# Parsed testcases at query #80
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
    var_5 = 'pandas'
    var_6 = [var_4, var_5]
    var_7 = module_1.Config()
    var_8 = module_0.module(var_4, var_7)
    assert var_8 == 'numpy'
    var_9 = module_0.module(var_5, var_7)
    assert var_9 == 'pandas'
    var_10 = 'numpy.core'
    var_11 = module_0.module(var_10, var_7)
    assert var_11 == 'numpy'
    var_12 = 'pandas.io'
    var_13 = module_0.module(var_12, var_7)
    assert var_13 == 'pandas'
    var_14 = '.local_module'
    var_15 = module_0.module(var_14)
    var_16 = '.local.submodule'
    var_17 = module_0.module(var_16)
    var_18 = '^django'
    var_19 = 'DJANGO'
    var_20 = 'django'
    var_21 = module_0.module(var_20, var_7)
    assert var_21 == 'DJANGO'
    var_22 = 'django.contrib'
    var_23 = module_0.module(var_22, var_7)
    assert var_23 == 'DJANGO'
    var_24 = 'my_project'
    var_25 = var_0 / var_24
    var_26 = 'module.py'
    var_27 = var_25 / var_26
    var_28 = [var_25]
    var_29 = module_1.Config()
    var_30 = module_0.module(var_24, var_29)
    var_31 = 'my_project.module'
    var_32 = module_0.module(var_31, var_29)
    var_33 = 'namespace_pkg'
    var_34 = var_0 / var_33
    var_35 = 'submodule.py'
    var_36 = var_34 / var_35
    var_37 = [var_34]
    var_38 = [var_33]
    var_39 = module_1.Config()
    var_40 = module_0.module(var_33, var_39)
    var_41 = 'namespace_pkg.submodule'
    var_42 = module_0.module(var_41, var_39)
    var_43 = 'auto_ns_pkg'
    var_44 = var_0 / var_43
    var_45 = 'submodule.py'
    var_46 = var_44 / var_45
    var_47 = [var_44]
    var_48 = True
    var_49 = module_1.Config()
    var_50 = module_0.module(var_43, var_49)
    var_51 = 'auto_ns_pkg.submodule'
    var_52 = module_0.module(var_51, var_49)



# Parsed testcases at query #81
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
    var_14 = 'my_module'
    var_15 = module_0.module(var_14, var_4)
    var_16 = 'my_namespace'
    var_17 = [var_16]
    var_18 = 'my_namespace.submodule'
    var_19 = module_0.module(var_18, var_4)



# Parsed testcases at query #82
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
    var_10 = '^test_.*'
    var_11 = 'TESTS'
    var_12 = 'test_example'
    var_13 = module_0.module(var_12, var_6)
    assert var_13 == 'TESTS'
    var_14 = '/path/to/src'
    var_15 = 'my_module'
    var_16 = module_0.module(var_15, var_6)
    var_17 = 'my_namespace'
    var_18 = [var_17]
    var_19 = 'my_namespace.submodule'
    var_20 = module_0.module(var_19, var_6)
    var_21 = 'THIRDPARTY'
    var_22 = module_1.Config()
    var_23 = 'unknown_module'
    var_24 = module_0.module(var_23, var_22)
    assert var_24 == 'THIRDPARTY'



# Parsed testcases at query #83
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
    var_11 = '^django'
    var_12 = 'DJANGO'
    var_13 = 'django.contrib'
    var_14 = module_0.module(var_13, var_6)
    assert var_14 == 'DJANGO'
    var_15 = '/path/to/src'
    var_16 = 'my_module'
    var_17 = module_0.module(var_16, var_6)
    var_18 = 'my_namespace'
    var_19 = [var_18]
    var_20 = 'my_namespace.submodule'
    var_21 = module_0.module(var_20, var_6)



# Parsed testcases at query #84
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
    var_18 = 'test_example.submodule'
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



# Parsed testcases at query #85
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
    var_13 = 'src'
    var_14 = var_0 / var_13
    var_15 = 'my_package'
    var_16 = var_14 / var_15
    var_17 = var_14 / var_15
    var_18 = '__init__.py'
    var_19 = var_17 / var_18
    var_20 = [var_14]
    var_21 = module_1.Config()
    var_22 = module_0.module(var_15, var_21)
    var_23 = 'src'
    var_24 = var_0 / var_23
    var_25 = 'namespace'
    var_26 = var_24 / var_25
    var_27 = [var_24]
    var_28 = [var_25]
    var_29 = module_1.Config()
    var_30 = 'namespace.submodule'
    var_31 = module_0.module(var_30, var_29)
    var_32 = 'os.path'
    var_33 = module_0.module(var_32)
    var_34 = 'nonexistent_module'
    var_35 = module_0.module(var_34)



# Parsed testcases at query #86
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
    var_12 = '.sub.local_module'
    var_13 = module_0.module(var_12)
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
    var_23 = 'my_package'
    var_24 = module_0.module(var_23, var_6)
    var_25 = 'my_namespace'
    var_26 = [var_25]
    var_27 = True
    var_28 = 'my_namespace.submodule'
    var_29 = module_0.module(var_28, var_6)



# Parsed testcases at query #87
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    var_2 = 'sys'
    var_3 = module_0.module(var_2)
    var_4 = '.local_module'
    var_5 = module_0.module(var_4)
    var_6 = 'test*'
    var_7 = [var_6]
    var_8 = module_1.Config()
    var_9 = 'test_module'
    var_10 = module_0.module(var_9, var_8)
    assert var_10 == 'test*'
    var_11 = '^django'
    var_12 = 'django.contrib'
    var_13 = module_0.module(var_12, var_8)
    var_14 = '/path/to/src'
    var_15 = 'my_module'
    var_16 = module_0.module(var_15, var_8)
    var_17 = 'my_namespace'
    var_18 = [var_17]
    var_19 = 'my_namespace.sub_module'
    var_20 = module_0.module(var_19, var_8)
    var_21 = True
    var_22 = 'auto_namespace.sub_module'
    var_23 = module_0.module(var_22, var_8)
    var_24 = 'unknown_module'
    var_25 = module_0.module(var_24, var_8)



# Parsed testcases at query #88
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
    var_18 = 'test_package.submodule'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/path/to/src'
    var_21 = 'src_module'
    var_22 = module_0.module(var_21, var_6)
    var_23 = 'src_package.submodule'
    var_24 = module_0.module(var_23, var_6)
    var_25 = 'namespace_package'
    var_26 = [var_25]
    var_27 = 'namespace_package.submodule'
    var_28 = module_0.module(var_27, var_6)
    var_29 = True
    var_30 = 'auto_namespace.submodule'
    var_31 = module_0.module(var_30, var_6)
    var_32 = 'THIRDPARTY'
    var_33 = module_1.Config()
    var_34 = 'unknown_module'
    var_35 = module_0.module(var_34, var_33)
    assert var_35 == 'THIRDPARTY'



# Parsed testcases at query #89
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
    var_20 = '/path/to/src'
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
    var_29 = 'unknown_module'
    var_30 = module_0.module(var_29, var_28)
    assert var_30 == 'THIRDPARTY'



# Parsed testcases at query #90
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
    var_10 = '^test_.*'
    var_11 = 'TESTS'
    var_12 = 'test_example'
    var_13 = module_0.module(var_12, var_6)
    assert var_13 == 'TESTS'
    var_14 = '/project/src'
    var_15 = 'project'
    var_16 = module_0.module(var_15, var_6)
    var_17 = 'project.subpackage'
    var_18 = [var_17]
    var_19 = 'project.subpackage.module'
    var_20 = module_0.module(var_19, var_6)
    var_21 = 'non_existent_module'
    var_22 = module_0.module(var_21)



# Parsed testcases at query #91
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
    var_9 = '^django.*'
    var_10 = 'DJANGO'
    var_11 = 'django.contrib'
    var_12 = module_0.module(var_11, var_4)
    assert var_12 == 'DJANGO'
    var_13 = '/path/to/src'
    var_14 = 'src_module'
    var_15 = module_0.module(var_14, var_4)



# Parsed testcases at query #92
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



# Parsed testcases at query #93
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
    var_20 = 'src'
    var_21 = var_0 / var_20
    var_22 = 'my_module'
    var_23 = var_21 / var_22
    var_24 = '__init__.py'
    var_25 = var_23 / var_24
    var_26 = [var_21]
    var_27 = module_1.Config()
    var_28 = module_0.module(var_22, var_27)
    var_29 = 'my_module.submodule'
    var_30 = module_0.module(var_29, var_27)
    var_31 = 'src'
    var_32 = var_0 / var_31
    var_33 = 'namespace'
    var_34 = var_32 / var_33
    var_35 = 'module.py'
    var_36 = var_34 / var_35
    var_37 = "print('hello')"
    var_38 = [var_32]
    var_39 = [var_33]
    var_40 = module_1.Config()
    var_41 = 'namespace.module'
    var_42 = module_0.module(var_41, var_40)
    var_43 = 'THIRDPARTY'
    var_44 = module_1.Config()
    var_45 = 'unknown_module'
    var_46 = module_0.module(var_45, var_44)
    assert var_46 == 'THIRDPARTY'



# Parsed testcases at query #94
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
    var_14 = 'company'
    var_15 = [var_14]
    var_16 = module_1.Config()
    var_17 = 'company.module'
    var_18 = module_0.module(var_17, var_16)
    assert var_18 == 'FIRSTPARTY'
    var_19 = '/project/src'
    var_20 = 'project'
    var_21 = module_0.module(var_20, var_16)
    assert var_21 == 'FIRSTPARTY'
    var_22 = [var_20]
    var_23 = 'project.submodule'
    var_24 = module_0.module(var_23, var_16)
    assert var_24 == 'FIRSTPARTY'



# Parsed testcases at query #95
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
    var_14 = 'my_module'
    var_15 = module_0.module(var_14, var_4)



# Parsed testcases at query #96
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
    var_14 = 'mymodule'
    var_15 = module_0.module(var_14, var_4)
    var_16 = 'namespace'
    var_17 = [var_16]
    var_18 = 'namespace.submodule'
    var_19 = module_0.module(var_18, var_4)
    var_20 = True
    var_21 = 'namespace.submodule'
    var_22 = module_0.module(var_21, var_4)



# Parsed testcases at query #97
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
    var_38 = 'nonexistent_module'
    var_39 = module_0.module(var_38)
    assert var_39 == 'THIRDPARTY'



# Parsed testcases at query #98
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
    var_18 = '^test_'
    var_19 = 'TESTS'
    var_20 = 'test_module'
    var_21 = module_0.module(var_20, var_11)
    assert var_21 == 'TESTS'
    var_22 = 'test_another'
    var_23 = module_0.module(var_22, var_11)
    assert var_23 == 'TESTS'
    var_24 = '/project/src'
    var_25 = 'project'
    var_26 = module_0.module(var_25, var_11)
    var_27 = 'project.submodule'
    var_28 = module_0.module(var_27, var_11)
    var_29 = [var_25]
    var_30 = module_0.module(var_25, var_11)
    var_31 = module_0.module(var_27, var_11)
    var_32 = 'THIRDPARTY'
    var_33 = module_1.Config()
    var_34 = 'unknown_module'
    var_35 = module_0.module(var_34, var_33)
    assert var_35 == 'THIRDPARTY'



# Parsed testcases at query #99
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
    var_10 = 'mycompany'
    var_11 = [var_10]
    var_12 = module_1.Config()
    var_13 = 'mycompany.utils'
    var_14 = module_0.module(var_13)
    assert var_14 == 'FIRSTPARTY'
    var_15 = '/project/src'
    var_16 = 'project'
    var_17 = module_0.module(var_16, var_12)
    assert var_17 == 'FIRSTPARTY'
    var_18 = 'project.sub'
    var_19 = [var_18]
    var_20 = 'project.sub.module'
    var_21 = module_0.module(var_20, var_12)
    assert var_21 == 'FIRSTPARTY'
    var_22 = 'nonexistent_module'
    var_23 = module_0.module(var_22)
    assert var_23 == 'THIRDPARTY'
    var_24 = 'CUSTOM'
    var_25 = module_1.Config()
    var_26 = 'unknown'
    var_27 = module_0.module(var_26, var_25)
    assert var_27 == 'CUSTOM'



# Parsed testcases at query #100
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
    var_13 = '/project/src'
    var_14 = 'my_module'
    var_15 = module_0.module(var_14, var_4)
    assert var_15 == 'FIRSTPARTY'
    var_16 = 'my_namespace'
    var_17 = [var_16]
    var_18 = False
    var_19 = 'my_namespace.nested'
    var_20 = module_0.module(var_19, var_4)
    assert var_20 == 'FIRSTPARTY'
    var_21 = module_1.Config()
    var_22 = 'CaseSensitive'
    var_23 = module_0.module(var_22, var_21)
    assert var_23 == 'THIRDPARTY'



# Parsed testcases at query #101
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
    var_14 = 'my_module'
    var_15 = module_0.module(var_14, var_4)
    var_16 = 'my_namespace'
    var_17 = [var_16]
    var_18 = 'my_namespace.submodule'
    var_19 = module_0.module(var_18, var_4)



# Parsed testcases at query #102
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
    var_19 = 'myproject.module'
    var_20 = module_0.module(var_19, var_18)



# Parsed testcases at query #103
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
    var_14 = '^test_.*'
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
    var_25 = 'unknown_module'
    var_26 = module_0.module(var_25)
    assert var_26 == 'THIRDPARTY'



# Parsed testcases at query #104
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
    var_14 = 'mymodule'
    var_15 = module_0.module(var_14, var_4)
    var_16 = 'mynamespace'
    var_17 = [var_16]
    var_18 = 'mynamespace.submodule'
    var_19 = module_0.module(var_18, var_4)



# Parsed testcases at query #105
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
    var_12 = '.another.local'
    var_13 = module_0.module(var_12)
    var_14 = '^test.*'
    var_15 = 'TESTS'
    var_16 = 'test_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_utils'
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
    var_29 = 'namespace'
    var_30 = var_0 / var_29
    var_31 = [var_30]
    var_32 = [var_29]
    var_33 = module_1.Config()
    var_34 = 'namespace.submodule'
    var_35 = module_0.module(var_34, var_33)
    var_36 = 'THIRDPARTY'
    var_37 = module_1.Config()
    var_38 = 'unknown_module'
    var_39 = module_0.module(var_38, var_37)
    assert var_39 == 'THIRDPARTY'



# Parsed testcases at query #106
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
    var_4 = 'pytest'
    var_5 = [var_4]
    var_6 = module_1.Config()
    var_7 = module_0.module(var_4, var_6)
    assert var_7 == 'pytest'
    var_8 = '.local_module'
    var_9 = module_0.module(var_8)
    var_10 = '^django'
    var_11 = 'DJANGO'
    var_12 = 'django.contrib'
    var_13 = module_0.module(var_12, var_6)
    assert var_13 == 'DJANGO'
    var_14 = '/project/src'
    var_15 = 'project'
    var_16 = module_0.module(var_15, var_6)
    var_17 = 'project.sub'
    var_18 = [var_17]
    var_19 = True
    var_20 = 'project.sub.module'
    var_21 = module_0.module(var_20, var_6)



# Parsed testcases at query #107
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
    var_11 = 'django.models'
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



# Parsed testcases at query #108
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
    var_18 = 'test_sub.module'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/project/src'
    var_21 = 'my_module'
    var_22 = module_0.module(var_21, var_6)
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'my_package'
    var_24 = module_0.module(var_23, var_6)
    assert var_24 == 'FIRSTPARTY'
    var_25 = 'my_namespace'
    var_26 = [var_25]
    var_27 = True
    var_28 = 'my_namespace.submodule'
    var_29 = module_0.module(var_28, var_6)
    assert var_29 == 'FIRSTPARTY'



# Parsed testcases at query #109
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
    var_29 = 'namespace'
    var_30 = var_0 / var_29
    var_31 = [var_30]
    var_32 = [var_29]
    var_33 = module_1.Config()
    var_34 = 'namespace.submodule'
    var_35 = module_0.module(var_34, var_33)
    assert var_35 == 'FIRSTPARTY'
    var_36 = 'THIRDPARTY'
    var_37 = module_1.Config()
    var_38 = 'unknown_module'
    var_39 = module_0.module(var_38, var_37)
    assert var_39 == 'THIRDPARTY'



# Parsed testcases at query #110
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
    var_6 = 'django.contrib'
    var_7 = module_0.module(var_6, var_4)
    assert var_7 == 'django'
    var_8 = '.local_module'
    var_9 = module_0.module(var_8)
    var_10 = '^test_'
    var_11 = 'TESTS'
    var_12 = 'test_module'
    var_13 = module_0.module(var_12, var_4)
    assert var_13 == 'TESTS'
    var_14 = '/path/to/src'
    var_15 = 'my_module'
    var_16 = module_0.module(var_15, var_4)
    var_17 = 'my_namespace'
    var_18 = [var_17]
    var_19 = False
    var_20 = 'my_namespace.submodule'
    var_21 = module_0.module(var_20, var_4)
    var_22 = 'THIRDPARTY'
    var_23 = module_1.Config()
    var_24 = 'unknown_module'
    var_25 = module_0.module(var_24, var_23)
    assert var_25 == 'THIRDPARTY'



# Parsed testcases at query #111
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
    var_9 = 'test_another'
    var_10 = module_0.module(var_9, var_6)
    assert var_10 == 'test*'
    var_11 = '.local_module'
    var_12 = module_0.module(var_11)
    assert var_12 == 'LOCALFOLDER'
    var_13 = '.another.local'
    var_14 = module_0.module(var_13)
    assert var_14 == 'LOCALFOLDER'
    var_15 = '^django'
    var_16 = 'DJANGO'
    var_17 = 'django.contrib'
    var_18 = module_0.module(var_17, var_6)
    assert var_18 == 'DJANGO'
    var_19 = 'django.core'
    var_20 = module_0.module(var_19, var_6)
    assert var_20 == 'DJANGO'
    var_21 = 'myproject'
    var_22 = var_0 / var_21
    var_23 = 'module.py'
    var_24 = var_22 / var_23
    var_25 = [var_22]
    var_26 = module_1.Config()
    var_27 = 'myproject.module'
    var_28 = module_0.module(var_27, var_26)
    assert var_28 == 'FIRSTPARTY'
    var_29 = module_0.module(var_21, var_26)
    assert var_29 == 'FIRSTPARTY'
    var_30 = 'namespace'
    var_31 = var_0 / var_30
    var_32 = 'submodule.py'
    var_33 = var_31 / var_32
    var_34 = [var_31]
    var_35 = [var_30]
    var_36 = module_1.Config()
    var_37 = 'namespace.submodule'
    var_38 = module_0.module(var_37, var_36)
    assert var_38 == 'FIRSTPARTY'
    var_39 = 'THIRDPARTY'
    var_40 = module_1.Config()
    var_41 = 'unknown_module'
    var_42 = module_0.module(var_41, var_40)
    assert var_42 == 'THIRDPARTY'



# Parsed testcases at query #112
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
    var_23 = 'src.module'
    var_24 = module_0.module(var_23, var_6)
    var_25 = 'namespace'
    var_26 = [var_25]
    var_27 = True
    var_28 = 'namespace.module'
    var_29 = module_0.module(var_28, var_6)
    var_30 = 'THIRDPARTY'
    var_31 = module_1.Config()
    var_32 = 'unknown_module'
    var_33 = module_0.module(var_32, var_31)
    assert var_33 == 'THIRDPARTY'



# Parsed testcases at query #113
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
    var_14 = 'my_module'
    var_15 = module_0.module(var_14, var_4)



# Parsed testcases at query #114
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
    var_20 = 'src'
    var_21 = var_0 / var_20
    var_22 = 'my_package'
    var_23 = var_21 / var_22
    var_24 = var_21 / var_22
    var_25 = '__init__.py'
    var_26 = var_24 / var_25
    var_27 = [var_21]
    var_28 = module_1.Config()
    var_29 = module_0.module(var_22, var_28)
    assert var_29 == 'FIRSTPARTY'
    var_30 = 'my_package.submodule'
    var_31 = module_0.module(var_30, var_28)
    assert var_31 == 'FIRSTPARTY'
    var_32 = 'src'
    var_33 = var_0 / var_32
    var_34 = 'namespace'
    var_35 = var_33 / var_34
    var_36 = var_33 / var_34
    var_37 = 'module.py'
    var_38 = var_36 / var_37
    var_39 = [var_33]
    var_40 = [var_34]
    var_41 = module_1.Config()
    var_42 = 'namespace.module'
    var_43 = module_0.module(var_42, var_41)
    assert var_43 == 'FIRSTPARTY'
    var_44 = 'unknown_module'
    var_45 = module_0.module(var_44)
    assert var_45 == 'THIRDPARTY'



# Parsed testcases at query #115
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
    var_12 = '.local_module.submodule'
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
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'project.namespace'
    var_24 = [var_23]
    var_25 = 'project.namespace.submodule'
    var_26 = module_0.module(var_25, var_6)
    assert var_26 == 'FIRSTPARTY'



# Parsed testcases at query #116
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
    var_4 = 'test'
    var_5 = [var_4]
    var_6 = module_1.Config()
    var_7 = 'test_module'
    var_8 = module_0.module(var_7, var_6)
    assert var_8 == 'test'
    var_9 = '.local_module'
    var_10 = module_0.module(var_9)
    assert var_10 == 'LOCALFOLDER'
    var_11 = '^django'
    var_12 = 'DJANGO'
    var_13 = 'django.contrib'
    var_14 = module_0.module(var_13, var_6)
    assert var_14 == 'DJANGO'
    var_15 = '/path/to/src'
    var_16 = 'src_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'FIRSTPARTY'
    var_18 = 'namespace'
    var_19 = [var_18]
    var_20 = 'namespace.module'
    var_21 = module_0.module(var_20, var_6)
    assert var_21 == 'FIRSTPARTY'
    var_22 = True
    var_23 = 'auto_namespace.module'
    var_24 = module_0.module(var_23, var_6)
    assert var_24 == 'FIRSTPARTY'
    var_25 = 'unknown_module'
    var_26 = module_0.module(var_25)
    assert var_26 == 'THIRDPARTY'



# Parsed testcases at query #117
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
    var_12 = '.sub.local_module'
    var_13 = module_0.module(var_12)
    var_14 = '^test_.*'
    var_15 = 'TESTS'
    var_16 = 'test_example'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_example.submodule'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/path/to/src'
    var_21 = 'module'
    var_22 = module_0.module(var_21, var_6)
    var_23 = 'module.submodule'
    var_24 = module_0.module(var_23, var_6)
    var_25 = 'namespace'
    var_26 = [var_25]
    var_27 = True
    var_28 = 'namespace.submodule'
    var_29 = module_0.module(var_28, var_6)
    var_30 = 'THIRDPARTY'
    var_31 = module_1.Config()
    var_32 = 'unknown_module'
    var_33 = module_0.module(var_32, var_31)
    assert var_33 == 'THIRDPARTY'



# Parsed testcases at query #118
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
    var_6 = '.local'
    var_7 = module_0.module(var_6)
    var_8 = '^django'
    var_9 = 'THIRDPARTY'
    var_10 = 'django.contrib'
    var_11 = module_0.module(var_10, var_4)
    assert var_11 == 'THIRDPARTY'
    var_12 = '/path/to/src'
    var_13 = 'mymodule'
    var_14 = module_0.module(var_13, var_4)
    var_15 = 'namespace'
    var_16 = [var_15]
    var_17 = 'namespace.submodule'
    var_18 = module_0.module(var_17, var_4)
    var_19 = 'nonexistent'
    var_20 = module_0.module(var_19)
    assert var_20 == 'STDLIB'



# Parsed testcases at query #119
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
    var_18 = 'test.utils'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = 'myproject'
    var_21 = var_0 / var_20
    var_22 = 'module.py'
    var_23 = var_21 / var_22
    var_24 = [var_21]
    var_25 = module_1.Config()
    var_26 = module_0.module(var_20, var_25)
    assert var_26 == 'FIRSTPARTY'
    var_27 = 'myproject.submodule'
    var_28 = module_0.module(var_27, var_25)
    assert var_28 == 'FIRSTPARTY'
    var_29 = 'namespace'
    var_30 = var_0 / var_29
    var_31 = 'subnamespace'
    var_32 = var_30 / var_31
    var_33 = [var_30]
    var_34 = [var_29]
    var_35 = module_1.Config()
    var_36 = module_0.module(var_29, var_35)
    assert var_36 == 'FIRSTPARTY'
    var_37 = 'namespace.subnamespace'
    var_38 = module_0.module(var_37, var_35)
    assert var_38 == 'FIRSTPARTY'
    var_39 = 'CUSTOM'
    var_40 = module_1.Config()
    var_41 = 'unknown_module'
    var_42 = module_0.module(var_41, var_40)
    assert var_42 == 'CUSTOM'



# Parsed testcases at query #120
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
    var_14 = '/path/to/src'
    var_15 = 'my_module'
    var_16 = module_0.module(var_15, var_6)
    var_17 = 'my_namespace'
    var_18 = [var_17]
    var_19 = False
    var_20 = 'my_namespace.sub_module'
    var_21 = module_0.module(var_20, var_6)
    var_22 = 'THIRDPARTY'
    var_23 = module_1.Config()
    var_24 = 'unknown_module'
    var_25 = module_0.module(var_24, var_23)
    assert var_25 == 'THIRDPARTY'



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
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
    var_20 = True
    var_21 = '.py'
    var_22 = [var_21]
    var_23 = frozenset(var_22)
    var_24 = 'auto_namespace.submodule'
    var_25 = module_0.module(var_24, var_4)
    assert var_25 == 'FIRSTPARTY'
    var_26 = 'unknown_module'
    var_27 = module_0.module(var_26)
    assert var_27 == 'THIRDPARTY'



# Parsed testcases at query #2
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
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
    var_13 = '/path/to/src'
    var_14 = 'my_module'
    var_15 = module_0.module(var_14, var_6)
    assert var_15 == 'FIRSTPARTY'
    var_16 = 'my_namespace'
    var_17 = [var_16]
    var_18 = 'my_namespace.submodule'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'FIRSTPARTY'
    var_20 = True
    var_21 = '.py'
    var_22 = [var_21]
    var_23 = frozenset(var_22)
    var_24 = 'my_namespace.submodule'
    var_25 = module_0.module(var_24, var_6)
    assert var_25 == 'FIRSTPARTY'
    var_26 = 'THIRDPARTY'
    var_27 = module_1.Config()
    var_28 = 'some_module'
    var_29 = module_0.module(var_28, var_27)
    assert var_29 == 'THIRDPARTY'



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
    assert var_11 == 'LOCALFOLDER'
    var_12 = '^test_'
    var_13 = 'TESTS'
    var_14 = 'test_module'
    var_15 = module_0.module(var_14, var_6)
    assert var_15 == 'TESTS'
    var_16 = 'test_sub.module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = '/project/src'
    var_19 = 'project'
    var_20 = module_0.module(var_19, var_6)
    assert var_20 == 'FIRSTPARTY'
    var_21 = 'project.submodule'
    var_22 = module_0.module(var_21, var_6)
    assert var_22 == 'FIRSTPARTY'
    var_23 = [var_19]
    var_24 = module_0.module(var_21, var_6)
    assert var_24 == 'FIRSTPARTY'
    var_25 = 'THIRDPARTY'
    var_26 = module_1.Config()
    var_27 = 'unknown_module'
    var_28 = module_0.module(var_27, var_26)
    assert var_28 == 'THIRDPARTY'



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
    var_10 = '.local'
    var_11 = module_0.module(var_10)
    var_12 = '.local.module'
    var_13 = module_0.module(var_12)
    var_14 = '^test.*'
    var_15 = 'TESTS'
    var_16 = 'test_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_utils'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/path/to/src'
    var_21 = 'my_module'
    var_22 = module_0.module(var_21, var_6)
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'my_namespace'
    var_24 = [var_23]
    var_25 = True
    var_26 = 'my_namespace.submodule'
    var_27 = module_0.module(var_26, var_6)
    assert var_27 == 'FIRSTPARTY'
    var_28 = 'THIRDPARTY'
    var_29 = module_1.Config()
    var_30 = 'some_external_library'
    var_31 = module_0.module(var_30, var_29)
    assert var_31 == 'THIRDPARTY'



# Parsed testcases at query #5
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
    var_14 = '^test.*'
    var_15 = 'TESTS'
    var_16 = 'test_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_package.submodule'
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



# Parsed testcases at query #6
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
    var_4 = '.local_module'
    var_5 = module_0.module(var_4)
    assert var_5 == 'LOCALFOLDER'
    var_6 = '.another_local'
    var_7 = module_0.module(var_6)
    assert var_7 == 'LOCALFOLDER'
    var_8 = 'test*'
    var_9 = [var_8]
    var_10 = module_1.Config()
    var_11 = 'test_module'
    var_12 = module_0.module(var_11, var_10)
    assert var_12 == 'test*'
    var_13 = 'test_another'
    var_14 = module_0.module(var_13, var_10)
    assert var_14 == 'test*'
    var_15 = '^django'
    var_16 = 'DJANGO'
    var_17 = 'django.core'
    var_18 = module_0.module(var_17, var_10)
    assert var_18 == 'DJANGO'
    var_19 = 'django.contrib'
    var_20 = module_0.module(var_19, var_10)
    assert var_20 == 'DJANGO'
    var_21 = '/path/to/src'
    var_22 = 'my_module'
    var_23 = module_0.module(var_22, var_10)
    assert var_23 == 'FIRSTPARTY'
    var_24 = 'my_namespace'
    var_25 = [var_24]
    var_26 = False
    var_27 = 'my_namespace.submodule'
    var_28 = module_0.module(var_27, var_10)
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
    var_20 = 'namespace'
    var_21 = var_0 / var_20
    var_22 = 'submodule.py'
    var_23 = var_21 / var_22
    var_24 = [var_21]
    var_25 = [var_20]
    var_26 = module_1.Config()
    var_27 = 'namespace.submodule'
    var_28 = module_0.module(var_27, var_26)



# Parsed testcases at query #9
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1)
    assert var_2 == 'STDLIB'
    var_3 = 'django'
    var_4 = module_1.module(var_3)
    assert var_4 == 'THIRDPARTY'
    var_5 = 'my_project'
    var_6 = module_1.module(var_5)
    assert var_6 == 'FIRSTPARTY'
    var_7 = '.local_module'
    var_8 = module_1.module(var_7)
    assert var_8 == 'LOCALFOLDER'
    var_9 = 'unknown_module'
    var_10 = module_1.module(var_9)



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
    var_18 = 'my_namespace.submodule'
    var_19 = module_0.module(var_18, var_4)
    assert var_19 == 'FIRSTPARTY'



# Parsed testcases at query #11
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
    var_12 = '.another.local'
    var_13 = module_0.module(var_12)
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
    var_23 = 'project.submodule'
    var_24 = module_0.module(var_23, var_6)
    var_25 = 'project.namespace'
    var_26 = [var_25]
    var_27 = True
    var_28 = 'project.namespace.submodule'
    var_29 = module_0.module(var_28, var_6)



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
    var_21 = 'project'
    var_22 = module_0.module(var_21, var_6)
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'project.submodule'
    var_24 = module_0.module(var_23, var_6)
    assert var_24 == 'FIRSTPARTY'
    var_25 = 'project'
    var_26 = [var_25]
    var_27 = True
    var_28 = 'project'
    var_29 = module_0.module(var_28, var_6)
    assert var_29 == 'FIRSTPARTY'
    var_30 = 'project.submodule'
    var_31 = module_0.module(var_30, var_6)
    assert var_31 == 'FIRSTPARTY'
    var_32 = 'requests'
    var_33 = module_0.module(var_32)
    assert var_33 == 'THIRDPARTY'
    var_34 = 'flask'
    var_35 = module_0.module(var_34)
    assert var_35 == 'THIRDPARTY'



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
    var_4 = 'test*'
    var_5 = [var_4]
    var_6 = module_1.Config()
    var_7 = 'test_module'
    var_8 = module_0.module(var_7, var_6)
    assert var_8 == 'test*'
    var_9 = '.local_module'
    var_10 = module_0.module(var_9)
    var_11 = '^django'
    var_12 = 'DJANGO'
    var_13 = 'django.contrib'
    var_14 = module_0.module(var_13, var_6)
    assert var_14 == 'DJANGO'
    var_15 = '/path/to/src'
    var_16 = 'my_module'
    var_17 = module_0.module(var_16, var_6)
    var_18 = 'my_namespace'
    var_19 = [var_18]
    var_20 = 'my_namespace.submodule'
    var_21 = module_0.module(var_20, var_6)



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
    var_14 = '/path/to/src'
    var_15 = 'my_module'
    var_16 = module_0.module(var_15, var_6)
    assert var_16 == 'FIRSTPARTY'
    var_17 = 'my_namespace'
    var_18 = [var_17]
    var_19 = 'my_namespace.submodule'
    var_20 = module_0.module(var_19, var_6)
    assert var_20 == 'FIRSTPARTY'



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
    var_12 = '^test.*'
    var_13 = 'TESTS'
    var_14 = 'test_module'
    var_15 = module_0.module(var_14, var_6)
    assert var_15 == 'TESTS'
    var_16 = 'test.utils'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'my_project'
    var_19 = var_0 / var_18
    var_20 = 'module.py'
    var_21 = var_19 / var_20
    var_22 = [var_19]
    var_23 = module_1.Config()
    var_24 = module_0.module(var_18, var_23)
    assert var_24 == 'FIRSTPARTY'
    var_25 = 'my_project.module'
    var_26 = module_0.module(var_25, var_23)
    assert var_26 == 'FIRSTPARTY'
    var_27 = 'namespace_pkg'
    var_28 = var_0 / var_27
    var_29 = 'submodule.py'
    var_30 = var_28 / var_29
    var_31 = [var_28]
    var_32 = [var_27]
    var_33 = module_1.Config()
    var_34 = 'namespace_pkg.submodule'
    var_35 = module_0.module(var_34, var_33)
    assert var_35 == 'FIRSTPARTY'
    var_36 = 'auto_ns_pkg'
    var_37 = var_0 / var_36
    var_38 = 'submodule.py'
    var_39 = var_37 / var_38
    var_40 = [var_37]
    var_41 = True
    var_42 = module_1.Config()
    var_43 = 'auto_ns_pkg.submodule'
    var_44 = module_0.module(var_43, var_42)
    assert var_44 == 'FIRSTPARTY'
    var_45 = 'unknown_module'
    var_46 = module_0.module(var_45)
    assert var_46 == 'THIRDPARTY'



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
    var_20 = '/path/to/src'
    var_21 = 'src_module'
    var_22 = module_0.module(var_21, var_6)
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'src_module.submodule'
    var_24 = module_0.module(var_23, var_6)
    assert var_24 == 'FIRSTPARTY'
    var_25 = 'THIRDPARTY'
    var_26 = module_1.Config()
    var_27 = 'unknown_module'
    var_28 = module_0.module(var_27, var_26)
    assert var_28 == 'THIRDPARTY'



# Parsed testcases at query #17
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1)
    assert var_2 == 'STDLIB'
    var_3 = 'django'
    var_4 = module_1.module(var_3)
    assert var_4 == 'THIRDPARTY'
    var_5 = 'my_project'
    var_6 = module_1.module(var_5)
    assert var_6 == 'FIRSTPARTY'
    var_7 = '.local_module'
    var_8 = module_1.module(var_7)
    assert var_8 == 'LOCALFOLDER'
    var_9 = 'unknown_module'
    var_10 = module_1.module(var_9)
    assert var_10 == 'THIRDPARTY'



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
    assert var_11 == 'LOCALFOLDER'
    var_12 = '.sub.local_module'
    var_13 = module_0.module(var_12)
    assert var_13 == 'LOCALFOLDER'
    var_14 = '^django'
    var_15 = 'DJANGO'
    var_16 = 'django'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'DJANGO'
    var_18 = 'django.contrib'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'DJANGO'
    var_20 = '/project/src'
    var_21 = 'project'
    var_22 = module_0.module(var_21, var_6)
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'project.module'
    var_24 = module_0.module(var_23, var_6)
    assert var_24 == 'FIRSTPARTY'
    var_25 = [var_21]
    var_26 = 'project.sub'
    var_27 = module_0.module(var_26, var_6)
    assert var_27 == 'FIRSTPARTY'
    var_28 = True
    var_29 = module_0.module(var_26, var_6)
    assert var_29 == 'FIRSTPARTY'
    var_30 = 'unknown_module'
    var_31 = module_0.module(var_30)
    assert var_31 == 'THIRDPARTY'



# Parsed testcases at query #19
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
    var_12 = '^test.*'
    var_13 = 'TESTS'
    var_14 = 'test_module'
    var_15 = module_0.module(var_14, var_6)
    assert var_15 == 'TESTS'
    var_16 = 'test_package.submodule'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = '/project/src'
    var_19 = 'project'
    var_20 = module_0.module(var_19, var_6)
    var_21 = 'project.submodule'
    var_22 = module_0.module(var_21, var_6)
    var_23 = 'project.namespace'
    var_24 = [var_23]
    var_25 = True
    var_26 = 'project.namespace.submodule'
    var_27 = module_0.module(var_26, var_6)
    var_28 = 'unknown_module'
    var_29 = module_0.module(var_28)



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
    var_4 = '.local_module'
    var_5 = module_0.module(var_4)
    var_6 = 'test*'
    var_7 = [var_6]
    var_8 = module_1.Config()
    var_9 = 'test_module'
    var_10 = module_0.module(var_9, var_8)
    assert var_10 == 'test*'
    var_11 = '^django'
    var_12 = 'DJANGO'
    var_13 = 'django.core'
    var_14 = module_0.module(var_13, var_8)
    assert var_14 == 'DJANGO'
    var_15 = '/path/to/src'
    var_16 = 'src_module'
    var_17 = module_0.module(var_16, var_8)
    var_18 = 'namespace_pkg'
    var_19 = [var_18]
    var_20 = 'namespace_pkg.submodule'
    var_21 = module_0.module(var_20, var_8)



# Parsed testcases at query #22
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
    var_14 = 'my_module'
    var_15 = module_0.module(var_14, var_4)
    var_16 = 'my_namespace'
    var_17 = [var_16]
    var_18 = False
    var_19 = 'my_namespace.submodule'
    var_20 = module_0.module(var_19, var_4)
    var_21 = True
    var_22 = 'auto_namespace.submodule'
    var_23 = module_0.module(var_22, var_4)



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
    var_13 = '/path/to/src'
    var_14 = 'my_module'
    var_15 = module_0.module(var_14, var_4)
    var_16 = 'my_namespace'
    var_17 = [var_16]
    var_18 = 'my_namespace.submodule'
    var_19 = module_0.module(var_18, var_4)
    var_20 = True
    var_21 = 'auto_namespace.submodule'
    var_22 = module_0.module(var_21, var_4)



# Parsed testcases at query #24
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    var_3 = 'django'
    var_4 = module_1.module(var_3, var_0)
    var_5 = '.local_module'
    var_6 = module_1.module(var_5, var_0)
    var_7 = 'my_project'
    var_8 = module_1.module(var_7, var_0)
    var_9 = 'numpy'
    var_10 = module_1.module(var_9, var_0)
    var_11 = 'pytest'
    var_12 = module_1.module(var_11, var_0)



# Parsed testcases at query #25
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
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
    var_13 = '/path/to/src'
    var_14 = 'my_module'
    var_15 = module_0.module(var_14, var_6)
    var_16 = 'my_namespace'
    var_17 = [var_16]
    var_18 = 'my_namespace.submodule'
    var_19 = module_0.module(var_18, var_6)



# Parsed testcases at query #26
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
    var_14 = '/path/to/src'
    var_15 = 'my_module'
    var_16 = module_0.module(var_15, var_6)
    var_17 = 'my_namespace'
    var_18 = [var_17]
    var_19 = 'my_namespace.submodule'
    var_20 = module_0.module(var_19, var_6)
    var_21 = 'unknown_module'
    var_22 = module_0.module(var_21)
    assert var_22 == 'THIRDPARTY'



# Parsed testcases at query #27
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
    var_17 = '# test'
    var_18 = [var_14]
    var_19 = module_1.Config()
    var_20 = module_0.module(var_13, var_19)
    var_21 = 'namespace'
    var_22 = var_0 / var_21
    var_23 = [var_22]
    var_24 = [var_21]
    var_25 = module_1.Config()
    var_26 = 'namespace.submodule'
    var_27 = module_0.module(var_26, var_25)



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
    var_12 = 'pytest'
    var_13 = 'THIRDPARTY'
    var_14 = (var_12, var_13)
    var_15 = [var_14]
    var_16 = module_1.Config()
    var_17 = module_0.module(var_12, var_16)
    assert var_17 == 'THIRDPARTY'
    var_18 = '/path/to/project'
    var_19 = 'project'
    var_20 = module_0.module(var_19, var_16)
    assert var_20 == 'FIRSTPARTY'
    var_21 = 'project.sub'
    var_22 = [var_21]
    var_23 = 'project.sub.module'
    var_24 = module_0.module(var_23, var_16)
    assert var_24 == 'FIRSTPARTY'
    var_25 = module_1.Config()
    var_26 = 'unknown_module'
    var_27 = module_0.module(var_26, var_25)
    assert var_27 == 'THIRDPARTY'



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
    var_12 = '.sub.local_module'
    var_13 = module_0.module(var_12)
    var_14 = '^test_.*'
    var_15 = 'TESTS'
    var_16 = 'test_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_sub.module'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/path/to/src'
    var_21 = 'src_module'
    var_22 = module_0.module(var_21, var_6)
    var_23 = 'src.sub_module'
    var_24 = module_0.module(var_23, var_6)
    var_25 = 'namespace'
    var_26 = [var_25]
    var_27 = True
    var_28 = 'namespace.sub_module'
    var_29 = module_0.module(var_28, var_6)
    var_30 = 'unknown_module'
    var_31 = module_0.module(var_30)
    assert var_31 == 'THIRDPARTY'



# Parsed testcases at query #30
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
    var_13 = 'src'
    var_14 = var_0 / var_13
    var_15 = 'my_module.py'
    var_16 = var_14 / var_15
    var_17 = [var_14]
    var_18 = module_1.Config()
    var_19 = 'my_module'
    var_20 = module_0.module(var_19, var_18)
    var_21 = 'src'
    var_22 = var_0 / var_21
    var_23 = 'namespace'
    var_24 = var_22 / var_23
    var_25 = var_22 / var_23
    var_26 = 'module.py'
    var_27 = var_25 / var_26
    var_28 = [var_22]
    var_29 = [var_23]
    var_30 = module_1.Config()
    var_31 = 'namespace.module'
    var_32 = module_0.module(var_31, var_30)
    var_33 = 'nonexistent_module'
    var_34 = module_0.module(var_33)



# Parsed testcases at query #31
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
    var_13 = '/project/src'
    var_14 = 'my_module'
    var_15 = module_0.module(var_14, var_4)
    assert var_15 == 'FIRSTPARTY'
    var_16 = 'my_namespace'
    var_17 = [var_16]
    var_18 = 'my_namespace.submodule'
    var_19 = module_0.module(var_18, var_4)
    assert var_19 == 'FIRSTPARTY'



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
    var_7 = '.local'
    var_8 = module_0.module(var_7)
    var_9 = '^test_.*'
    var_10 = 'TEST'
    var_11 = 'test_example'
    var_12 = module_0.module(var_11, var_4)
    assert var_12 == 'TEST'
    var_13 = '/src'
    var_14 = 'src_module'
    var_15 = module_0.module(var_14, var_4)
    var_16 = 'namespace'
    var_17 = [var_16]
    var_18 = 'namespace.module'
    var_19 = module_0.module(var_18, var_4)
    var_20 = 'nonexistent_module'
    var_21 = module_0.module(var_20)



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
    var_4 = 'requests'
    var_5 = module_0.module(var_4)
    assert var_5 == 'THIRDPARTY'
    var_6 = 'django'
    var_7 = [var_6]
    var_8 = module_1.Config()
    var_9 = module_0.module(var_6, var_8)
    assert var_9 == 'django'
    var_10 = 'django.contrib'
    var_11 = module_0.module(var_10, var_8)
    assert var_11 == 'django'
    var_12 = '.local_module'
    var_13 = module_0.module(var_12)
    assert var_13 == 'LOCALFOLDER'
    var_14 = '.sub.local_module'
    var_15 = module_0.module(var_14)
    assert var_15 == 'LOCALFOLDER'
    var_16 = '^test_.*'
    var_17 = 'TESTS'
    var_18 = 'test_example'
    var_19 = module_0.module(var_18, var_8)
    assert var_19 == 'TESTS'
    var_20 = 'test_example.submodule'
    var_21 = module_0.module(var_20, var_8)
    assert var_21 == 'TESTS'
    var_22 = '/path/to/project'
    var_23 = 'project'
    var_24 = module_0.module(var_23, var_8)
    assert var_24 == 'FIRSTPARTY'
    var_25 = 'project.submodule'
    var_26 = module_0.module(var_25, var_8)
    assert var_26 == 'FIRSTPARTY'
    var_27 = 'project'
    var_28 = [var_27]
    var_29 = 'project.submodule'
    var_30 = module_0.module(var_29, var_8)
    assert var_30 == 'FIRSTPARTY'
    var_31 = True
    var_32 = 'project.submodule'
    var_33 = module_0.module(var_32, var_8)
    assert var_33 == 'FIRSTPARTY'



# Parsed testcases at query #34
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
    var_18 = 'test_example.submodule'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/project/src'
    var_21 = 'project'
    var_22 = module_0.module(var_21, var_6)
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'project.submodule'
    var_24 = module_0.module(var_23, var_6)
    assert var_24 == 'FIRSTPARTY'
    var_25 = [var_21]
    var_26 = module_0.module(var_21, var_6)
    assert var_26 == 'FIRSTPARTY'
    var_27 = module_0.module(var_23, var_6)
    assert var_27 == 'FIRSTPARTY'
    var_28 = 'unknown_module'
    var_29 = module_0.module(var_28)
    assert var_29 == 'THIRDPARTY'



# Parsed testcases at query #35
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
    var_20 = 'my_project'
    var_21 = var_0 / var_20
    var_22 = 'module.py'
    var_23 = var_21 / var_22
    var_24 = [var_21]
    var_25 = module_1.Config()
    var_26 = module_0.module(var_20, var_25)
    var_27 = 'my_project.module'
    var_28 = module_0.module(var_27, var_25)
    var_29 = 'namespace'
    var_30 = var_0 / var_29
    var_31 = 'submodule.py'
    var_32 = var_30 / var_31
    var_33 = [var_30]
    var_34 = [var_29]
    var_35 = module_1.Config()
    var_36 = 'namespace.submodule'
    var_37 = module_0.module(var_36, var_35)
    var_38 = 'auto_namespace'
    var_39 = var_0 / var_38
    var_40 = 'submodule.py'
    var_41 = var_39 / var_40
    var_42 = [var_39]
    var_43 = True
    var_44 = module_1.Config()
    var_45 = 'auto_namespace.submodule'
    var_46 = module_0.module(var_45, var_44)
    var_47 = 'unknown_module'
    var_48 = module_0.module(var_47)



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
    var_19 = False
    var_20 = 'my_namespace.submodule'
    var_21 = module_0.module(var_20, var_6)
    assert var_21 == 'FIRSTPARTY'



# Parsed testcases at query #37
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
    var_22 = 'non_existent_module'
    var_23 = module_0.module(var_22)
    assert var_23 == 'THIRDPARTY'



# Parsed testcases at query #38
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
    var_19 = 'my_namespace.submodule'
    var_20 = module_0.module(var_19, var_4)



# Parsed testcases at query #39
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    var_2 = 'sys'
    var_3 = module_0.module(var_2)
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
    var_13 = '.another.local'
    var_14 = module_0.module(var_13)
    var_15 = '^test_.*'
    var_16 = 'TEST'
    var_17 = 'test_example'
    var_18 = module_0.module(var_17, var_6)
    assert var_18 == 'TEST'
    var_19 = 'test_example.module'
    var_20 = module_0.module(var_19, var_6)
    assert var_20 == 'TEST'
    var_21 = 'src'
    var_22 = var_0 / var_21
    var_23 = 'mymodule'
    var_24 = var_22 / var_23
    var_25 = var_22 / var_23
    var_26 = '__init__.py'
    var_27 = var_25 / var_26
    var_28 = ''
    var_29 = [var_22]
    var_30 = module_1.Config()
    var_31 = module_0.module(var_23, var_30)
    var_32 = 'mymodule.submodule'
    var_33 = module_0.module(var_32, var_30)
    var_34 = 'src'
    var_35 = var_0 / var_34
    var_36 = 'namespace'
    var_37 = var_35 / var_36
    var_38 = var_35 / var_36
    var_39 = 'module.py'
    var_40 = var_38 / var_39
    var_41 = ''
    var_42 = [var_35]
    var_43 = [var_36]
    var_44 = module_1.Config()
    var_45 = 'namespace.module'
    var_46 = module_0.module(var_45, var_44)
    var_47 = 'src'
    var_48 = var_0 / var_47
    var_49 = 'auto_ns'
    var_50 = var_48 / var_49
    var_51 = var_48 / var_49
    var_52 = 'module.py'
    var_53 = var_51 / var_52
    var_54 = ''
    var_55 = [var_48]
    var_56 = True
    var_57 = module_1.Config()
    var_58 = 'auto_ns.module'
    var_59 = module_0.module(var_58, var_57)



# Parsed testcases at query #40
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
    assert var_5 == 'LOCALFOLDER'
    var_6 = '.another_local'
    var_7 = module_0.module(var_6)
    assert var_7 == 'LOCALFOLDER'
    var_8 = 'django'
    var_9 = 'flask'
    var_10 = [var_8, var_9]
    var_11 = module_1.Config()
    var_12 = module_0.module(var_8, var_11)
    assert var_12 == 'django'
    var_13 = module_0.module(var_9, var_11)
    assert var_13 == 'flask'
    var_14 = 'django.contrib'
    var_15 = module_0.module(var_14, var_11)
    assert var_15 == 'django'
    var_16 = 'flask.ext'
    var_17 = module_0.module(var_16, var_11)
    assert var_17 == 'flask'
    var_18 = '^test.*'
    var_19 = 'TESTS'
    var_20 = 'test_module'
    var_21 = module_0.module(var_20, var_11)
    assert var_21 == 'TESTS'
    var_22 = 'test_another'
    var_23 = module_0.module(var_22, var_11)
    assert var_23 == 'TESTS'
    var_24 = 'tests.utils'
    var_25 = module_0.module(var_24, var_11)
    assert var_25 == 'TESTS'
    var_26 = '/project/src'
    var_27 = 'project'
    var_28 = module_0.module(var_27, var_11)
    assert var_28 == 'FIRSTPARTY'
    var_29 = 'project.utils'
    var_30 = module_0.module(var_29, var_11)
    assert var_30 == 'FIRSTPARTY'
    var_31 = [var_27]
    var_32 = 'project.subpackage'
    var_33 = module_0.module(var_32, var_11)
    assert var_33 == 'FIRSTPARTY'
    var_34 = True
    var_35 = 'py'
    var_36 = [var_35]
    var_37 = frozenset(var_36)
    var_38 = module_0.module(var_32, var_11)
    assert var_38 == 'FIRSTPARTY'
    var_39 = 'THIRDPARTY'
    var_40 = module_1.Config()
    var_41 = 'requests'
    var_42 = module_0.module(var_41, var_40)
    assert var_42 == 'THIRDPARTY'



# Parsed testcases at query #41
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
    var_16 = 'test_sub.module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'my_project'
    var_19 = var_0 / var_18
    var_20 = 'module.py'
    var_21 = var_19 / var_20
    var_22 = [var_19]
    var_23 = module_1.Config()
    var_24 = module_0.module(var_18, var_23)
    var_25 = 'my_project.submodule'
    var_26 = module_0.module(var_25, var_23)
    var_27 = 'namespace'
    var_28 = var_0 / var_27
    var_29 = 'submodule.py'
    var_30 = var_28 / var_29
    var_31 = [var_28]
    var_32 = [var_27]
    var_33 = module_1.Config()
    var_34 = 'namespace.submodule'
    var_35 = module_0.module(var_34, var_33)
    var_36 = 'nonexistent_module_xyz'
    var_37 = module_0.module(var_36)



# Parsed testcases at query #42
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
    var_14 = '^test_.*'
    var_15 = 'TESTS'
    var_16 = 'test_example'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_example.submodule'
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



# Parsed testcases at query #43
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
    var_18 = 'test_sub.module'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/project/src'
    var_21 = 'project'
    var_22 = module_0.module(var_21, var_6)
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'project.submodule'
    var_24 = module_0.module(var_23, var_6)
    assert var_24 == 'FIRSTPARTY'
    var_25 = [var_21]
    var_26 = module_0.module(var_21, var_6)
    assert var_26 == 'FIRSTPARTY'
    var_27 = 'project.sub'
    var_28 = module_0.module(var_27, var_6)
    assert var_28 == 'FIRSTPARTY'



# Parsed testcases at query #44
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
    var_4 = 'pytest'
    var_5 = [var_4]
    var_6 = module_1.Config()
    var_7 = module_0.module(var_4, var_6)
    assert var_7 == 'pytest'
    var_8 = '.local_module'
    var_9 = module_0.module(var_8)
    assert var_9 == 'LOCALFOLDER'
    var_10 = '^django'
    var_11 = 'DJANGO'
    var_12 = 'django.contrib'
    var_13 = module_0.module(var_12, var_6)
    assert var_13 == 'DJANGO'
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



# Parsed testcases at query #45
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
    var_14 = 'my_module'
    var_15 = module_0.module(var_14, var_4)



# Parsed testcases at query #46
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
    var_14 = '^test.*'
    var_15 = 'TESTS'
    var_16 = 'test_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_package.submodule'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/path/to/src'
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
    var_29 = 'unknown_module'
    var_30 = module_0.module(var_29, var_28)
    assert var_30 == 'THIRDPARTY'



# Parsed testcases at query #47
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
    var_14 = '^test_'
    var_15 = 'TESTS'
    var_16 = 'test_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_package.submodule'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = 'myproject'
    var_21 = var_0 / var_20
    var_22 = 'module.py'
    var_23 = var_21 / var_22
    var_24 = [var_21]
    var_25 = module_1.Config()
    var_26 = module_0.module(var_20, var_25)
    assert var_26 == 'FIRSTPARTY'
    var_27 = 'myproject.module'
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
    var_38 = 'THIRDPARTY'
    var_39 = module_1.Config()
    var_40 = 'unknown_module'
    var_41 = module_0.module(var_40, var_39)
    assert var_41 == 'THIRDPARTY'



# Parsed testcases at query #48
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
    var_11 = '^django'
    var_12 = 'DJANGO'
    var_13 = 'django.contrib'
    var_14 = module_0.module(var_13, var_6)
    assert var_14 == 'DJANGO'
    var_15 = '/path/to/src'
    var_16 = 'my_module'
    var_17 = module_0.module(var_16, var_6)
    var_18 = 'my_namespace'
    var_19 = [var_18]
    var_20 = True
    var_21 = 'my_namespace.submodule'
    var_22 = module_0.module(var_21, var_6)



# Parsed testcases at query #49
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
    var_14 = '^test_.*'
    var_15 = 'TESTS'
    var_16 = 'test_example'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_example.module'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/path/to/project'
    var_21 = 'project'
    var_22 = module_0.module(var_21, var_6)
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'project.module'
    var_24 = module_0.module(var_23, var_6)
    assert var_24 == 'FIRSTPARTY'
    var_25 = [var_21]
    var_26 = 'project.subpackage'
    var_27 = module_0.module(var_26, var_6)
    assert var_27 == 'FIRSTPARTY'
    var_28 = 'requests'
    var_29 = module_0.module(var_28)
    assert var_29 == 'THIRDPARTY'
    var_30 = 'flask'
    var_31 = module_0.module(var_30)
    assert var_31 == 'THIRDPARTY'



# Parsed testcases at query #50
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
    var_6 = '.local'
    var_7 = module_0.module(var_6)
    assert var_7 == 'LOCALFOLDER'
    var_8 = '^django.*'
    var_9 = 'THIRDPARTY'
    var_10 = 'django.contrib'
    var_11 = module_0.module(var_10, var_4)
    assert var_11 == 'THIRDPARTY'
    var_12 = '/path/to/project'
    var_13 = 'project'
    var_14 = module_0.module(var_13, var_4)
    assert var_14 == 'FIRSTPARTY'
    var_15 = 'project.sub'
    var_16 = [var_15]
    var_17 = 'project.sub.module'
    var_18 = module_0.module(var_17, var_4)
    assert var_18 == 'FIRSTPARTY'
    var_19 = True
    var_20 = module_0.module(var_17, var_4)
    assert var_20 == 'FIRSTPARTY'
    var_21 = '/path/to/module'
    var_22 = 'module'
    var_23 = module_0.module(var_22, var_4)
    assert var_23 == 'FIRSTPARTY'



# Parsed testcases at query #51
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
    var_14 = '^test.*'
    var_15 = 'TESTS'
    var_16 = 'test_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test.utils'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/project/src'
    var_21 = 'project'
    var_22 = module_0.module(var_21, var_6)
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'project.module'
    var_24 = module_0.module(var_23, var_6)
    assert var_24 == 'FIRSTPARTY'
    var_25 = 'THIRDPARTY'
    var_26 = module_1.Config()
    var_27 = 'external_library'
    var_28 = module_0.module(var_27, var_26)
    assert var_28 == 'THIRDPARTY'



# Parsed testcases at query #52
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
    var_17 = '# test'
    var_18 = [var_14]
    var_19 = module_1.Config()
    var_20 = module_0.module(var_13, var_19)



# Parsed testcases at query #53
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
    var_15 = 'project'
    var_16 = module_0.module(var_15, var_6)
    assert var_16 == 'FIRSTPARTY'
    var_17 = [var_15]
    var_18 = 'project.submodule'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'FIRSTPARTY'



# Parsed testcases at query #54
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
    var_14 = 'my_module'
    var_15 = module_0.module(var_14, var_4)



# Parsed testcases at query #55
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
    var_11 = '^django'
    var_12 = 'DJANGO'
    var_13 = 'django.contrib'
    var_14 = module_0.module(var_13, var_6)
    assert var_14 == 'DJANGO'
    var_15 = '/path/to/src'
    var_16 = 'my_module'
    var_17 = module_0.module(var_16, var_6)
    var_18 = 'my_namespace'
    var_19 = [var_18]
    var_20 = 'my_namespace.sub_module'
    var_21 = module_0.module(var_20, var_6)
    var_22 = 'THIRDPARTY'
    var_23 = module_1.Config()
    var_24 = 'unknown_module'
    var_25 = module_0.module(var_24, var_23)
    assert var_25 == 'THIRDPARTY'



# Parsed testcases at query #56
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
    var_14 = 'mymodule'
    var_15 = module_0.module(var_14, var_4)
    assert var_15 == 'FIRSTPARTY'
    var_16 = 'mynamespace'
    var_17 = [var_16]
    var_18 = False
    var_19 = 'mynamespace.submodule'
    var_20 = module_0.module(var_19, var_4)
    assert var_20 == 'FIRSTPARTY'



# Parsed testcases at query #57
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
    var_11 = '^django'
    var_12 = 'DJANGO'
    var_13 = 'django.contrib'
    var_14 = module_0.module(var_13, var_6)
    assert var_14 == 'DJANGO'
    var_15 = '/path/to/src'
    var_16 = 'my_module'
    var_17 = module_0.module(var_16, var_6)
    var_18 = 'my_namespace'
    var_19 = [var_18]
    var_20 = True
    var_21 = 'my_namespace.submodule'
    var_22 = module_0.module(var_21, var_6)



# Parsed testcases at query #58
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
    var_23 = 'unknown_module'
    var_24 = module_0.module(var_23)
    assert var_24 == 'THIRDPARTY'



# Parsed testcases at query #59
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
    var_12 = '.another.local'
    var_13 = module_0.module(var_12)
    var_14 = '^test.*'
    var_15 = 'TESTS'
    var_16 = 'test_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_another'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = 'src'
    var_21 = var_0 / var_20
    var_22 = 'my_module.py'
    var_23 = var_21 / var_22
    var_24 = [var_21]
    var_25 = module_1.Config()
    var_26 = 'my_module'
    var_27 = module_0.module(var_26, var_25)
    assert var_27 == 'FIRSTPARTY'
    var_28 = 'src'
    var_29 = var_0 / var_28
    var_30 = 'namespace'
    var_31 = var_29 / var_30
    var_32 = var_29 / var_30
    var_33 = 'module.py'
    var_34 = var_32 / var_33
    var_35 = [var_29]
    var_36 = [var_30]
    var_37 = module_1.Config()
    var_38 = 'namespace.module'
    var_39 = module_0.module(var_38, var_37)
    assert var_39 == 'FIRSTPARTY'
    var_40 = 'THIRDPARTY'
    var_41 = module_1.Config()
    var_42 = 'unknown_module'
    var_43 = module_0.module(var_42, var_41)
    assert var_43 == 'THIRDPARTY'



# Parsed testcases at query #60
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
    var_13 = 'src'
    var_14 = var_0 / var_13
    var_15 = 'my_module'
    var_16 = var_14 / var_15
    var_17 = '__init__.py'
    var_18 = var_16 / var_17
    var_19 = [var_14]
    var_20 = module_1.Config()
    var_21 = module_0.module(var_15, var_20)
    var_22 = 'src'
    var_23 = var_0 / var_22
    var_24 = 'namespace'
    var_25 = var_23 / var_24
    var_26 = 'module.py'
    var_27 = var_25 / var_26
    var_28 = [var_23]
    var_29 = [var_24]
    var_30 = module_1.Config()
    var_31 = 'namespace.module'
    var_32 = module_0.module(var_31, var_30)
    var_33 = 'nonexistent_module_12345'
    var_34 = module_0.module(var_33)
    assert var_34 == 'THIRDPARTY'



# Parsed testcases at query #61
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
    var_14 = 'my_module'
    var_15 = module_0.module(var_14, var_4)
    var_16 = 'my_namespace'
    var_17 = [var_16]
    var_18 = 'my_namespace.submodule'
    var_19 = module_0.module(var_18, var_4)



# Parsed testcases at query #62
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
    var_15 = 'project_module'
    var_16 = module_0.module(var_15, var_6)
    assert var_16 == 'FIRSTPARTY'



# Parsed testcases at query #63
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
    var_18 = 'test_example.submodule'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = 'src'
    var_21 = var_0 / var_20
    var_22 = 'mymodule'
    var_23 = var_21 / var_22
    var_24 = '__init__.py'
    var_25 = var_23 / var_24
    var_26 = [var_21]
    var_27 = module_1.Config()
    var_28 = module_0.module(var_22, var_27)
    assert var_28 == 'FIRSTPARTY'
    var_29 = 'mymodule.sub'
    var_30 = module_0.module(var_29, var_27)
    assert var_30 == 'FIRSTPARTY'
    var_31 = 'src'
    var_32 = var_0 / var_31
    var_33 = 'namespace'
    var_34 = var_32 / var_33
    var_35 = 'module.py'
    var_36 = var_34 / var_35
    var_37 = [var_32]
    var_38 = [var_33]
    var_39 = module_1.Config()
    var_40 = 'namespace.module'
    var_41 = module_0.module(var_40, var_39)
    assert var_41 == 'FIRSTPARTY'
    var_42 = 'nonexistent_module'
    var_43 = module_0.module(var_42)
    assert var_43 == 'THIRDPARTY'



# Parsed testcases at query #64
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
    var_12 = '^test.*'
    var_13 = 'TESTS'
    var_14 = 'test_module'
    var_15 = module_0.module(var_14, var_6)
    assert var_15 == 'TESTS'
    var_16 = 'test_package.submodule'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = '/path/to/src'
    var_19 = 'src_module'
    var_20 = module_0.module(var_19, var_6)
    assert var_20 == 'FIRSTPARTY'
    var_21 = 'namespace_package'
    var_22 = [var_21]
    var_23 = 'namespace_package.submodule'
    var_24 = module_0.module(var_23, var_6)
    assert var_24 == 'FIRSTPARTY'



# Parsed testcases at query #65
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    var_2 = 'numpy'
    var_3 = [var_2]
    var_4 = module_1.Config()
    var_5 = module_0.module(var_2, var_4)
    assert var_5 == 'numpy'
    var_6 = '.local'
    var_7 = module_0.module(var_6)
    var_8 = '^django'
    var_9 = 'DJANGO'
    var_10 = 'django.contrib'
    var_11 = module_0.module(var_10, var_4)
    assert var_11 == 'DJANGO'
    var_12 = '/src'
    var_13 = 'mymodule'
    var_14 = module_0.module(var_13, var_4)



# Parsed testcases at query #66
#--------------------------


import isort.place as module_0
import isort.settings as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
    var_2 = '.local'
    var_3 = module_0.module(var_2)
    assert var_3 == 'LOCALFOLDER'
    var_4 = 'test*'
    var_5 = [var_4]
    var_6 = module_1.Config()
    var_7 = 'test_module'
    var_8 = module_0.module(var_7, var_6)
    assert var_8 == 'test*'
    var_9 = 'django.*'
    var_10 = 'DJANGO'
    var_11 = 'django.contrib'
    var_12 = module_0.module(var_11, var_6)
    assert var_12 == 'DJANGO'
    var_13 = '/path/to/src'
    var_14 = 'my_module'
    var_15 = module_0.module(var_14, var_6)
    assert var_15 == 'FIRSTPARTY'



# Parsed testcases at query #67
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
    var_12 = '.local_module.submodule'
    var_13 = module_0.module(var_12)
    var_14 = '^test_'
    var_15 = 'TESTS'
    var_16 = 'test_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_module.submodule'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = 'src'
    var_21 = var_0 / var_20
    var_22 = 'my_package'
    var_23 = var_21 / var_22
    var_24 = var_21 / var_22
    var_25 = '__init__.py'
    var_26 = var_24 / var_25
    var_27 = ''
    var_28 = [var_21]
    var_29 = module_1.Config()
    var_30 = module_0.module(var_22, var_29)
    var_31 = 'my_package.submodule'
    var_32 = module_0.module(var_31, var_29)
    var_33 = 'src'
    var_34 = var_0 / var_33
    var_35 = 'namespace_package'
    var_36 = var_34 / var_35
    var_37 = var_34 / var_35
    var_38 = 'module.py'
    var_39 = var_37 / var_38
    var_40 = ''
    var_41 = [var_34]
    var_42 = [var_35]
    var_43 = module_1.Config()
    var_44 = 'namespace_package.module'
    var_45 = module_0.module(var_44, var_43)
    var_46 = 'unknown_module'
    var_47 = module_0.module(var_46)



# Parsed testcases at query #68
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
    var_12 = '.sub.local_module'
    var_13 = module_0.module(var_12)
    var_14 = '^test.*'
    var_15 = 'TESTS'
    var_16 = 'test_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_utils'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/project/src'
    var_21 = 'project'
    var_22 = module_0.module(var_21, var_6)
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'project.utils'
    var_24 = module_0.module(var_23, var_6)
    assert var_24 == 'FIRSTPARTY'
    var_25 = 'project.namespace'
    var_26 = [var_25]
    var_27 = 'project.namespace.sub'
    var_28 = module_0.module(var_27, var_6)
    assert var_28 == 'FIRSTPARTY'



# Parsed testcases at query #69
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
    var_14 = '^test_'
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
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'project.module'
    var_24 = module_0.module(var_23, var_6)
    assert var_24 == 'FIRSTPARTY'
    var_25 = 'project.namespace'
    var_26 = [var_25]
    var_27 = 'project.namespace.submodule'
    var_28 = module_0.module(var_27, var_6)
    assert var_28 == 'FIRSTPARTY'
    var_29 = 'unknown_module'
    var_30 = module_0.module(var_29)
    assert var_30 == 'THIRDPARTY'



# Parsed testcases at query #70
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
    var_20 = False
    var_21 = 'my_namespace.submodule'
    var_22 = module_0.module(var_21, var_6)
    assert var_22 == 'FIRSTPARTY'



# Parsed testcases at query #71
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
    var_5 = 'pandas'
    var_6 = [var_4, var_5]
    var_7 = module_1.Config()
    var_8 = module_0.module(var_4, var_7)
    assert var_8 == 'numpy'
    var_9 = 'pandas.core'
    var_10 = module_0.module(var_9, var_7)
    assert var_10 == 'pandas'
    var_11 = '.local_module'
    var_12 = module_0.module(var_11)
    assert var_12 == 'LOCALFOLDER'
    var_13 = '.sub.local_module'
    var_14 = module_0.module(var_13)
    assert var_14 == 'LOCALFOLDER'
    var_15 = '^django'
    var_16 = 'DJANGO'
    var_17 = 'django'
    var_18 = module_0.module(var_17, var_7)
    assert var_18 == 'DJANGO'
    var_19 = 'django.contrib'
    var_20 = module_0.module(var_19, var_7)
    assert var_20 == 'DJANGO'
    var_21 = 'my_project'
    var_22 = var_0 / var_21
    var_23 = 'module.py'
    var_24 = var_22 / var_23
    var_25 = [var_22]
    var_26 = module_1.Config()
    var_27 = 'my_project.module'
    var_28 = module_0.module(var_27, var_26)
    assert var_28 == 'FIRSTPARTY'
    var_29 = 'namespace'
    var_30 = var_0 / var_29
    var_31 = 'submodule.py'
    var_32 = var_30 / var_31
    var_33 = [var_25]
    var_34 = [var_29]
    var_35 = module_1.Config()
    var_36 = 'namespace.submodule'
    var_37 = module_0.module(var_36, var_35)
    assert var_37 == 'FIRSTPARTY'



# Parsed testcases at query #72
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
    assert var_5 == 'LOCALFOLDER'
    var_6 = 'django'
    var_7 = [var_6]
    var_8 = module_1.Config()
    var_9 = module_0.module(var_6, var_8)
    assert var_9 == 'django'
    var_10 = '^test_'
    var_11 = 'TESTS'
    var_12 = 'test_example'
    var_13 = module_0.module(var_12, var_8)
    assert var_13 == 'TESTS'
    var_14 = '/path/to/src'
    var_15 = 'module'
    var_16 = module_0.module(var_15, var_8)
    assert var_16 == 'FIRSTPARTY'
    var_17 = 'namespace'
    var_18 = [var_17]
    var_19 = 'namespace.module'
    var_20 = module_0.module(var_19, var_8)
    assert var_20 == 'FIRSTPARTY'
    var_21 = True
    var_22 = 'py'
    var_23 = [var_22]
    var_24 = frozenset(var_23)
    var_25 = 'namespace.module'
    var_26 = module_0.module(var_25, var_8)
    assert var_26 == 'FIRSTPARTY'
    var_27 = 'THIRDPARTY'
    var_28 = module_1.Config()
    var_29 = 'unknown_module'
    var_30 = module_0.module(var_29, var_28)
    assert var_30 == 'THIRDPARTY'



# Parsed testcases at query #73
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
    var_14 = '^test_.*'
    var_15 = 'TESTS'
    var_16 = 'test_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_package.submodule'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = 'my_project'
    var_21 = var_0 / var_20
    var_22 = 'module.py'
    var_23 = var_21 / var_22
    var_24 = [var_21]
    var_25 = module_1.Config()
    var_26 = 'my_project.module'
    var_27 = module_0.module(var_26, var_25)
    assert var_27 == 'FIRSTPARTY'
    var_28 = 'namespace'
    var_29 = var_0 / var_28
    var_30 = 'submodule.py'
    var_31 = var_29 / var_30
    var_32 = [var_24]
    var_33 = [var_28]
    var_34 = module_1.Config()
    var_35 = 'namespace.submodule'
    var_36 = module_0.module(var_35, var_34)
    assert var_36 == 'FIRSTPARTY'
    var_37 = 'THIRDPARTY'
    var_38 = module_1.Config()
    var_39 = 'unknown_module'
    var_40 = module_0.module(var_39, var_38)
    assert var_40 == 'THIRDPARTY'



# Parsed testcases at query #74
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
    assert var_15 == 'LOCALFOLDER'
    var_16 = '.sub.local_module'
    var_17 = module_0.module(var_16)
    assert var_17 == 'LOCALFOLDER'
    var_18 = '^test_.*'
    var_19 = 'TESTS'
    var_20 = 'test_example'
    var_21 = module_0.module(var_20, var_7)
    assert var_21 == 'TESTS'
    var_22 = 'test_example.submodule'
    var_23 = module_0.module(var_22, var_7)
    assert var_23 == 'TESTS'
    var_24 = 'src'
    var_25 = var_0 / var_24
    var_26 = 'my_package'
    var_27 = var_25 / var_26
    var_28 = var_25 / var_26
    var_29 = '__init__.py'
    var_30 = var_28 / var_29
    var_31 = ''
    var_32 = [var_25]
    var_33 = module_1.Config()
    var_34 = module_0.module(var_26, var_33)
    assert var_34 == 'FIRSTPARTY'
    var_35 = 'my_package.submodule'
    var_36 = module_0.module(var_35, var_33)
    assert var_36 == 'FIRSTPARTY'
    var_37 = 'src'
    var_38 = var_0 / var_37
    var_39 = 'namespace_package'
    var_40 = var_38 / var_39
    var_41 = var_38 / var_39
    var_42 = 'module.py'
    var_43 = var_41 / var_42
    var_44 = ''
    var_45 = [var_38]
    var_46 = [var_39]
    var_47 = module_1.Config()
    var_48 = module_0.module(var_39, var_47)
    assert var_48 == 'FIRSTPARTY'
    var_49 = 'namespace_package.module'
    var_50 = module_0.module(var_49, var_47)
    assert var_50 == 'FIRSTPARTY'
    var_51 = 'unknown_module'
    var_52 = module_0.module(var_51)
    assert var_52 == 'THIRDPARTY'



# Parsed testcases at query #75
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1)
    var_3 = 'django'
    var_4 = module_1.module(var_3)
    var_5 = '.local_module'
    var_6 = module_1.module(var_5)
    var_7 = 'my_project'
    var_8 = module_1.module(var_7)
    var_9 = 'unknown_module'
    var_10 = module_1.module(var_9)



# Parsed testcases at query #76
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
    var_12 = '.local_module.submodule'
    var_13 = module_0.module(var_12)
    var_14 = '^test.*'
    var_15 = 'TESTS'
    var_16 = 'test_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_module.submodule'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = 'src'
    var_21 = var_0 / var_20
    var_22 = 'my_package'
    var_23 = var_21 / var_22
    var_24 = var_21 / var_22
    var_25 = '__init__.py'
    var_26 = var_24 / var_25
    var_27 = [var_21]
    var_28 = module_1.Config()
    var_29 = module_0.module(var_22, var_28)
    var_30 = 'my_package.submodule'
    var_31 = module_0.module(var_30, var_28)
    var_32 = 'src'
    var_33 = var_0 / var_32
    var_34 = 'namespace_package'
    var_35 = var_33 / var_34
    var_36 = [var_33]
    var_37 = True
    var_38 = module_1.Config()
    var_39 = module_0.module(var_34, var_38)
    var_40 = 'namespace_package.submodule'
    var_41 = module_0.module(var_40, var_38)
    var_42 = 'unknown_module'
    var_43 = module_0.module(var_42)



# Parsed testcases at query #77
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
    var_11 = '^django'
    var_12 = 'DJANGO'
    var_13 = 'django.core'
    var_14 = module_0.module(var_13, var_6)
    assert var_14 == 'DJANGO'
    var_15 = '/path/to/src'
    var_16 = 'my_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'FIRSTPARTY'
    var_18 = 'my_namespace'
    var_19 = [var_18]
    var_20 = True
    var_21 = 'my_namespace.sub_module'
    var_22 = module_0.module(var_21, var_6)
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'unknown_module'
    var_24 = module_0.module(var_23)
    assert var_24 == 'THIRDPARTY'



# Parsed testcases at query #78
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



# Parsed testcases at query #79
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
    var_14 = '/path/to/src'
    var_15 = 'my_module'
    var_16 = module_0.module(var_15, var_6)
    var_17 = 'my_namespace'
    var_18 = [var_17]
    var_19 = True
    var_20 = 'my_namespace.sub_module'
    var_21 = module_0.module(var_20, var_6)



# Parsed testcases at query #80
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1)
    var_3 = 'test*'
    var_4 = [var_3]
    var_5 = module_0.Config()
    var_6 = 'test_module'
    var_7 = module_1.module(var_6)
    assert var_7 == 'test*'
    var_8 = '.local_module'
    var_9 = module_1.module(var_8)
    var_10 = 'django.*'
    var_11 = 'DJANGO'
    var_12 = 'django.contrib'
    var_13 = module_1.module(var_12)
    assert var_13 == 'DJANGO'
    var_14 = '/path/to/src'
    var_15 = 'my_module'
    var_16 = module_1.module(var_15)



# Parsed testcases at query #81
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
    assert var_5 == 'LOCALFOLDER'
    var_6 = 'django'
    var_7 = [var_6]
    var_8 = module_1.Config()
    var_9 = module_0.module(var_6, var_8)
    assert var_9 == 'django'
    var_10 = '^django'
    var_11 = 'DJANGO'
    var_12 = module_0.module(var_6, var_8)
    assert var_12 == 'DJANGO'
    var_13 = 'my_project'
    var_14 = var_0 / var_13
    var_15 = 'module.py'
    var_16 = var_14 / var_15
    var_17 = [var_14]
    var_18 = module_1.Config()
    var_19 = module_0.module(var_13, var_18)
    assert var_19 == 'FIRSTPARTY'



# Parsed testcases at query #82
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
    var_12 = '^test_.*'
    var_13 = 'TESTS'
    var_14 = 'test_example'
    var_15 = module_0.module(var_14, var_6)
    assert var_15 == 'TESTS'
    var_16 = '/path/to/src'
    var_17 = 'my_module'
    var_18 = module_0.module(var_17, var_6)
    var_19 = 'my_namespace'
    var_20 = [var_19]
    var_21 = 'my_namespace.submodule'
    var_22 = module_0.module(var_21, var_6)



# Parsed testcases at query #83
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
    var_11 = '^django'
    var_12 = 'DJANGO'
    var_13 = 'django.contrib'
    var_14 = module_0.module(var_13, var_6)
    assert var_14 == 'DJANGO'
    var_15 = '/project/src'
    var_16 = 'project'
    var_17 = module_0.module(var_16, var_6)
    var_18 = 'project.sub'
    var_19 = [var_18]
    var_20 = True
    var_21 = 'project.sub.module'
    var_22 = module_0.module(var_21, var_6)



# Parsed testcases at query #84
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
    var_16 = 'test_namespace'
    var_17 = [var_16]
    var_18 = True
    var_19 = 'test_namespace.submodule'
    var_20 = module_0.module(var_19, var_4)
    assert var_20 == 'FIRSTPARTY'



# Parsed testcases at query #85
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
    var_12 = '.local.module'
    var_13 = module_0.module(var_12)
    var_14 = '^test_'
    var_15 = 'TESTS'
    var_16 = 'test_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_package.submodule'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = 'src'
    var_21 = var_0 / var_20
    var_22 = 'my_package'
    var_23 = var_21 / var_22
    var_24 = [var_21]
    var_25 = module_1.Config()
    var_26 = module_0.module(var_22, var_25)
    assert var_26 == 'FIRSTPARTY'
    var_27 = 'my_package.submodule'
    var_28 = module_0.module(var_27, var_25)
    assert var_28 == 'FIRSTPARTY'
    var_29 = 'src'
    var_30 = var_0 / var_29
    var_31 = 'namespace'
    var_32 = var_30 / var_31
    var_33 = [var_30]
    var_34 = [var_31]
    var_35 = module_1.Config()
    var_36 = 'namespace.package'
    var_37 = module_0.module(var_36, var_35)
    assert var_37 == 'FIRSTPARTY'



# Parsed testcases at query #86
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
    var_4 = 'tests'
    var_5 = [var_4]
    var_6 = module_1.Config()
    var_7 = 'tests.test_module'
    var_8 = module_0.module(var_7, var_6)
    assert var_8 == 'tests'
    var_9 = 'test_module'
    var_10 = module_0.module(var_9, var_6)
    assert var_10 == 'tests'
    var_11 = '.local_module'
    var_12 = module_0.module(var_11)
    var_13 = '.another_local'
    var_14 = module_0.module(var_13)
    var_15 = '^django'
    var_16 = 'DJANGO'
    var_17 = 'django.conf'
    var_18 = module_0.module(var_17, var_6)
    assert var_18 == 'DJANGO'
    var_19 = 'django.apps'
    var_20 = module_0.module(var_19, var_6)
    assert var_20 == 'DJANGO'
    var_21 = 'my_project'
    var_22 = var_0 / var_21
    var_23 = 'module.py'
    var_24 = var_22 / var_23
    var_25 = [var_22]
    var_26 = module_1.Config()
    var_27 = 'my_project.module'
    var_28 = module_0.module(var_27, var_26)
    var_29 = module_0.module(var_21, var_26)
    var_30 = 'namespace_pkg'
    var_31 = var_0 / var_30
    var_32 = 'submodule.py'
    var_33 = var_31 / var_32
    var_34 = [var_31]
    var_35 = [var_30]
    var_36 = module_1.Config()
    var_37 = 'namespace_pkg.submodule'
    var_38 = module_0.module(var_37, var_36)
    var_39 = 'THIRDPARTY'
    var_40 = module_1.Config()
    var_41 = 'unknown_module'
    var_42 = module_0.module(var_41, var_40)
    assert var_42 == 'THIRDPARTY'



# Parsed testcases at query #87
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
    var_14 = '^test_'
    var_15 = 'TESTS'
    var_16 = 'test_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_sub.submodule'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/project/src'
    var_21 = 'project'
    var_22 = module_0.module(var_21, var_6)
    var_23 = 'project.submodule'
    var_24 = module_0.module(var_23, var_6)
    var_25 = 'project.namespace'
    var_26 = [var_25]
    var_27 = True
    var_28 = 'project.namespace.submodule'
    var_29 = module_0.module(var_28, var_6)
    var_30 = 'unknown_module'
    var_31 = module_0.module(var_30)
    assert var_31 == 'THIRDPARTY'



# Parsed testcases at query #88
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
    var_9 = '^django.*'
    var_10 = 'DJANGO'
    var_11 = 'django.contrib'
    var_12 = module_0.module(var_11, var_4)
    assert var_12 == 'DJANGO'
    var_13 = 'src'
    var_14 = var_0 / var_13
    var_15 = 'mymodule'
    var_16 = var_14 / var_15
    var_17 = '__init__.py'
    var_18 = var_16 / var_17
    var_19 = [var_14]
    var_20 = module_1.Config()
    var_21 = module_0.module(var_15, var_20)



# Parsed testcases at query #89
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
    var_14 = '^test.*'
    var_15 = 'TESTS'
    var_16 = 'test_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_package.submodule'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/path/to/src'
    var_21 = 'src_module'
    var_22 = module_0.module(var_21, var_6)
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'src_package.submodule'
    var_24 = module_0.module(var_23, var_6)
    assert var_24 == 'FIRSTPARTY'
    var_25 = 'namespace_package'
    var_26 = [var_25]
    var_27 = 'namespace_package.submodule'
    var_28 = module_0.module(var_27, var_6)
    assert var_28 == 'FIRSTPARTY'
    var_29 = True
    var_30 = 'auto_namespace.submodule'
    var_31 = module_0.module(var_30, var_6)
    assert var_31 == 'FIRSTPARTY'
    var_32 = 'THIRDPARTY'
    var_33 = module_1.Config()
    var_34 = 'unknown_module'
    var_35 = module_0.module(var_34, var_33)
    assert var_35 == 'THIRDPARTY'



# Parsed testcases at query #90
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



# Parsed testcases at query #91
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
    var_20 = '/path/to/project'
    var_21 = 'project'
    var_22 = module_0.module(var_21, var_6)
    var_23 = 'project.module'
    var_24 = module_0.module(var_23, var_6)
    var_25 = 'project.namespace'
    var_26 = [var_25]
    var_27 = module_0.module(var_25, var_6)
    var_28 = 'project.namespace.module'
    var_29 = module_0.module(var_28, var_6)
    var_30 = True
    var_31 = module_0.module(var_25, var_6)
    var_32 = module_0.module(var_28, var_6)
    var_33 = 'THIRDPARTY'
    var_34 = module_1.Config()
    var_35 = 'unknown_module'
    var_36 = module_0.module(var_35, var_34)
    assert var_36 == 'THIRDPARTY'



# Parsed testcases at query #92
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
    var_14 = '^test_'
    var_15 = 'TESTS'
    var_16 = 'test_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_sub.submodule'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = 'src'
    var_21 = var_0 / var_20
    var_22 = 'my_package'
    var_23 = var_21 / var_22
    var_24 = var_21 / var_22
    var_25 = '__init__.py'
    var_26 = var_24 / var_25
    var_27 = ''
    var_28 = [var_21]
    var_29 = module_1.Config()
    var_30 = module_0.module(var_22, var_29)
    var_31 = 'my_package.submodule'
    var_32 = module_0.module(var_31, var_29)
    var_33 = 'src'
    var_34 = var_0 / var_33
    var_35 = 'namespace'
    var_36 = var_34 / var_35
    var_37 = var_34 / var_35
    var_38 = 'submodule.py'
    var_39 = var_37 / var_38
    var_40 = ''
    var_41 = [var_34]
    var_42 = True
    var_43 = module_1.Config()
    var_44 = 'namespace.submodule'
    var_45 = module_0.module(var_44, var_43)
    var_46 = 'unknown_module'
    var_47 = module_0.module(var_46)



# Parsed testcases at query #93
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
    var_14 = '^test.*'
    var_15 = 'TESTS'
    var_16 = 'test_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_package.module'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/path/to/src'
    var_21 = 'my_module'
    var_22 = module_0.module(var_21, var_6)
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'my_package'
    var_24 = module_0.module(var_23, var_6)
    assert var_24 == 'FIRSTPARTY'
    var_25 = 'my_namespace'
    var_26 = [var_25]
    var_27 = True
    var_28 = 'my_namespace.submodule'
    var_29 = module_0.module(var_28, var_6)
    assert var_29 == 'FIRSTPARTY'
    var_30 = 'THIRDPARTY'
    var_31 = module_1.Config()
    var_32 = 'unknown_module'
    var_33 = module_0.module(var_32, var_31)
    assert var_33 == 'THIRDPARTY'



# Parsed testcases at query #94
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
    var_13 = 'src'
    var_14 = var_0 / var_13
    var_15 = 'mypackage'
    var_16 = var_14 / var_15
    var_17 = var_14 / var_15
    var_18 = '__init__.py'
    var_19 = var_17 / var_18
    var_20 = [var_14]
    var_21 = module_1.Config()
    var_22 = module_0.module(var_15, var_21)



# Parsed testcases at query #95
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
    var_11 = '^django'
    var_12 = 'DJANGO'
    var_13 = 'django.contrib'
    var_14 = module_0.module(var_13, var_6)
    assert var_14 == 'DJANGO'
    var_15 = '/path/to/project'
    var_16 = 'project_module'
    var_17 = module_0.module(var_16, var_6)
    var_18 = 'project'
    var_19 = [var_18]
    var_20 = True
    var_21 = 'project.submodule'
    var_22 = module_0.module(var_21, var_6)
    var_23 = 'THIRDPARTY'
    var_24 = module_1.Config()
    var_25 = 'unknown_module'
    var_26 = module_0.module(var_25, var_24)
    assert var_26 == 'THIRDPARTY'



# Parsed testcases at query #96
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
    var_14 = '^test_'
    var_15 = 'TESTS'
    var_16 = 'test_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_package.module'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/path/to/src'
    var_21 = 'src_module'
    var_22 = module_0.module(var_21, var_6)
    var_23 = 'namespace_package'
    var_24 = [var_23]
    var_25 = 'namespace_package.submodule'
    var_26 = module_0.module(var_25, var_6)



# Parsed testcases at query #97
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
    var_12 = '.sub.local_module'
    var_13 = module_0.module(var_12)
    var_14 = '^test_'
    var_15 = 'TESTS'
    var_16 = 'test_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_sub.module'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = 'src'
    var_21 = var_0 / var_20
    var_22 = 'my_module'
    var_23 = var_21 / var_22
    var_24 = '__init__.py'
    var_25 = var_23 / var_24
    var_26 = [var_21]
    var_27 = module_1.Config()
    var_28 = module_0.module(var_22, var_27)
    var_29 = 'my_module.sub'
    var_30 = module_0.module(var_29, var_27)
    var_31 = 'src'
    var_32 = var_0 / var_31
    var_33 = 'namespace'
    var_34 = var_32 / var_33
    var_35 = 'module.py'
    var_36 = var_34 / var_35
    var_37 = 'test'
    var_38 = [var_32]
    var_39 = [var_33]
    var_40 = module_1.Config()
    var_41 = 'namespace.module'
    var_42 = module_0.module(var_41, var_40)
    var_43 = 'THIRDPARTY'
    var_44 = module_1.Config()
    var_45 = 'unknown_module'
    var_46 = module_0.module(var_45, var_44)
    assert var_46 == 'THIRDPARTY'



# Parsed testcases at query #98
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
    var_20 = '/path/to/src'
    var_21 = 'src_module'
    var_22 = module_0.module(var_21, var_6)
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'namespace'
    var_24 = [var_23]
    var_25 = 'namespace.submodule'
    var_26 = module_0.module(var_25, var_6)
    assert var_26 == 'FIRSTPARTY'



# Parsed testcases at query #99
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1)
    assert var_2 == 'STDLIB'
    var_3 = 'sys'
    var_4 = module_1.module(var_3)
    assert var_4 == 'STDLIB'
    var_5 = 'django'
    var_6 = module_1.module(var_5)
    assert var_6 == 'THIRDPARTY'
    var_7 = 'my_project'
    var_8 = module_1.module(var_7)
    assert var_8 == 'FIRSTPARTY'
    var_9 = '.local_module'
    var_10 = module_1.module(var_9)
    assert var_10 == 'LOCALFOLDER'
    var_11 = 'pytest'
    var_12 = module_1.module(var_11)
    assert var_12 == 'THIRDPARTY'
    var_13 = '__future__'
    var_14 = module_1.module(var_13)
    assert var_14 == 'FUTURE'
    var_15 = 'typing'
    var_16 = module_1.module(var_15)
    assert var_16 == 'TYPING'



# Parsed testcases at query #100
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
    var_12 = '.local.module'
    var_13 = module_0.module(var_12)
    var_14 = '^test.*'
    var_15 = 'TESTS'
    var_16 = 'test_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_package.module'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/path/to/project'
    var_21 = 'project_module'
    var_22 = module_0.module(var_21, var_6)
    var_23 = 'project'
    var_24 = [var_23]
    var_25 = 'project.submodule'
    var_26 = module_0.module(var_25, var_6)
    var_27 = True
    var_28 = 'project.submodule'
    var_29 = module_0.module(var_28, var_6)
    var_30 = 'THIRDPARTY'
    var_31 = module_1.Config()
    var_32 = 'unknown_module'
    var_33 = module_0.module(var_32, var_31)
    assert var_33 == 'THIRDPARTY'



# Parsed testcases at query #101
#--------------------------


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1)
    var_3 = 'django'
    var_4 = module_1.module(var_3)
    var_5 = 'my_project'
    var_6 = module_1.module(var_5)
    var_7 = '.local_module'
    var_8 = module_1.module(var_7)
    var_9 = '^test_.*'
    var_10 = 'TESTS'
    var_11 = 'test_example'
    var_12 = module_1.module(var_11, var_0)
    assert var_12 == 'TESTS'
    var_13 = 'custom_module'
    var_14 = module_1.module(var_13, var_0)
    assert var_14 == 'custom_module'



# Parsed testcases at query #102
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
    var_14 = 'my_module'
    var_15 = module_0.module(var_14, var_4)



# Parsed testcases at query #103
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
    var_22 = 'THIRDPARTY'
    var_23 = module_1.Config()
    var_24 = 'unknown_module'
    var_25 = module_0.module(var_24, var_23)
    assert var_25 == 'THIRDPARTY'



# Parsed testcases at query #104
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



# Parsed testcases at query #105
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
    var_18 = 'test_example.submodule'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = '/path/to/src'
    var_21 = 'my_module'
    var_22 = module_0.module(var_21, var_6)
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'my_module.submodule'
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



# Parsed testcases at query #106
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
    var_12 = '.subpackage.module'
    var_13 = module_0.module(var_12)
    assert var_13 == 'LOCALFOLDER'
    var_14 = 'mycompany'
    var_15 = [var_14]
    var_16 = module_1.Config()
    var_17 = 'mycompany.utils'
    var_18 = module_0.module(var_17, var_16)
    assert var_18 == 'FIRSTPARTY'
    var_19 = '/project/src'
    var_20 = 'project'
    var_21 = module_0.module(var_20, var_16)
    assert var_21 == 'FIRSTPARTY'
    var_22 = 'project.subpackage'
    var_23 = [var_22]
    var_24 = 'project.subpackage.module'
    var_25 = module_0.module(var_24, var_16)
    assert var_25 == 'FIRSTPARTY'
    var_26 = 'unknown_module'
    var_27 = module_0.module(var_26)
    assert var_27 == 'THIRDPARTY'



# Parsed testcases at query #107
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
    var_12 = 'test_example'
    var_13 = module_0.module(var_12, var_6)
    assert var_13 == 'TESTS'
    var_14 = '/project/src'
    var_15 = 'project'
    var_16 = module_0.module(var_15, var_6)
    var_17 = [var_15]
    var_18 = 'project.submodule'
    var_19 = module_0.module(var_18, var_6)
    var_20 = True
    var_21 = '.py'
    var_22 = [var_21]
    var_23 = frozenset(var_22)
    var_24 = module_0.module(var_18, var_6)



# Parsed testcases at query #108
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
    var_12 = '.another.local'
    var_13 = module_0.module(var_12)
    var_14 = '^test.*'
    var_15 = 'TESTS'
    var_16 = 'test_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_another.module'
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



# Parsed testcases at query #109
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



# Parsed testcases at query #110
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
    var_16 = 'parent'
    var_17 = [var_16]
    var_18 = True
    var_19 = 'parent.child'
    var_20 = module_0.module(var_19, var_4)



# Parsed testcases at query #111
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
    var_18 = 'namespace.submodule'
    var_19 = module_0.module(var_18, var_4)
    assert var_19 == 'FIRSTPARTY'
    var_20 = 'nonexistent_module'
    var_21 = module_0.module(var_20)
    assert var_21 == 'THIRDPARTY'



# Parsed testcases at query #112
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
    var_12 = '^django'
    var_13 = 'DJANGO'
    var_14 = 'django'
    var_15 = module_0.module(var_14, var_6)
    assert var_15 == 'DJANGO'
    var_16 = 'django.contrib'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'DJANGO'
    var_18 = '/path/to/src'
    var_19 = 'my_module'
    var_20 = module_0.module(var_19, var_6)
    var_21 = 'my_namespace'
    var_22 = [var_21]
    var_23 = False
    var_24 = 'my_namespace.submodule'
    var_25 = module_0.module(var_24, var_6)
    var_26 = 'THIRDPARTY'
    var_27 = module_1.Config()
    var_28 = 'unknown_module'
    var_29 = module_0.module(var_28, var_27)
    assert var_29 == 'THIRDPARTY'



# Parsed testcases at query #113
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
    var_14 = '^test_'
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
    var_25 = [var_21]
    var_26 = module_0.module(var_21, var_6)
    assert var_26 == 'FIRSTPARTY'
    var_27 = module_0.module(var_23, var_6)
    assert var_27 == 'FIRSTPARTY'
    var_28 = 'THIRDPARTY'
    var_29 = module_1.Config()
    var_30 = 'unknown_module'
    var_31 = module_0.module(var_30, var_29)
    assert var_31 == 'THIRDPARTY'



# Parsed testcases at query #114
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
    var_14 = '^test_'
    var_15 = 'TESTS'
    var_16 = 'test_module'
    var_17 = module_0.module(var_16, var_6)
    assert var_17 == 'TESTS'
    var_18 = 'test_sub.submodule'
    var_19 = module_0.module(var_18, var_6)
    assert var_19 == 'TESTS'
    var_20 = 'src'
    var_21 = var_0 / var_20
    var_22 = 'my_package'
    var_23 = var_21 / var_22
    var_24 = var_21 / var_22
    var_25 = '__init__.py'
    var_26 = var_24 / var_25
    var_27 = ''
    var_28 = [var_21]
    var_29 = module_1.Config()
    var_30 = module_0.module(var_22, var_29)
    var_31 = 'my_package.submodule'
    var_32 = module_0.module(var_31, var_29)
    var_33 = 'src'
    var_34 = var_0 / var_33
    var_35 = 'namespace'
    var_36 = var_34 / var_35
    var_37 = var_34 / var_35
    var_38 = 'submodule.py'
    var_39 = var_37 / var_38
    var_40 = ''
    var_41 = [var_34]
    var_42 = True
    var_43 = module_1.Config()
    var_44 = 'namespace.submodule'
    var_45 = module_0.module(var_44, var_43)
    var_46 = 'unknown_module'
    var_47 = module_0.module(var_46)



# Parsed testcases at query #115
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
    var_14 = '/path/to/src'
    var_15 = 'src_module'
    var_16 = module_0.module(var_15, var_6)
    var_17 = 'namespace'
    var_18 = [var_17]
    var_19 = 'namespace.submodule'
    var_20 = module_0.module(var_19, var_6)
    var_21 = True
    var_22 = '.py'
    var_23 = [var_22]
    var_24 = frozenset(var_23)
    var_25 = 'auto_namespace.submodule'
    var_26 = module_0.module(var_25, var_6)



