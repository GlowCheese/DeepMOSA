####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.place as module_1
import isort.settings as module_0


def test_case_0():
    var_0 = 'test_pattern'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'test_pattern.module'
    var_4 = module_1.module(var_3, var_2)
    assert var_4 == 'test_pattern'
    var_5 = module_1.module(var_0, var_2)
    assert var_5 == 'test_pattern'
    var_6 = 'other.module'
    var_7 = module_1.module(var_6, var_2)
    var_8 = module_0.Config()
    var_9 = '.local_module'
    var_10 = module_1.module(var_9, var_8)
    var_11 = 'not_local'
    var_12 = module_1.module(var_11, var_8)
    var_13 = '^test\\.*'
    var_14 = 'TEST_SECTION'
    var_15 = 'test.module'
    var_16 = module_1.module(var_15, var_8)
    assert var_16 == 'TEST_SECTION'
    var_17 = module_1.module(var_6, var_8)
    var_18 = '/fake/path'
    var_19 = 'CUSTOM_DEFAULT'
    var_20 = module_0.Config()
    var_21 = 'unknown.module'
    var_22 = module_1.module(var_21, var_20)
    assert var_22 == 'CUSTOM_DEFAULT'
    var_23 = 'All tests passed!'
    var_24 = print(var_23)



# Parsed testcases at query #2
#--------------------------



def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = module_1.module(var_0, var_2)
    assert var_3 == 'test_module'
    var_4 = module_0.Config()
    var_5 = '.local_module'
    var_6 = module_1.module(var_5, var_4)
    var_7 = 'test.*'
    var_8 = 'test_section'
    var_9 = (var_7, var_8)
    var_10 = [var_9]
    var_11 = module_0.Config()
    var_12 = 'test.module'
    var_13 = module_1.module(var_12, var_11)
    assert var_13 == 'test_section'
    var_14 = module_0.Config()
    var_15 = 'unknown_module'
    var_16 = module_1.module(var_15, var_14)
    var_17 = 'All tests passed!'
    var_18 = print(var_17)



# Parsed testcases at query #3
#--------------------------



def test_case_0():
    var_0 = 'test*'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'test_module'
    var_4 = module_1.module(var_3, var_2)
    assert var_4 == 'test*'
    var_5 = '.local_module'
    var_6 = module_1.module(var_5, var_2)
    var_7 = '^django.*'
    var_8 = 'THIRDPARTY'
    var_9 = 'django.app'
    var_10 = module_1.module(var_9, var_2)
    assert var_10 == 'THIRDPARTY'
    var_11 = '/src'
    var_12 = module_1.module(var_3, var_2)
    assert var_12 == 'FIRSTPARTY'
    var_13 = 'unknown_module'
    var_14 = module_1.module(var_13, var_2)
    var_15 = 'All tests passed!'
    var_16 = print(var_15)



# Parsed testcases at query #4
#--------------------------




# Parsed testcases at query #5
#--------------------------



def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = module_1.module(var_0, var_2)
    assert var_3 == 'test_module'
    var_4 = module_0.Config()
    var_5 = '.local_module'
    var_6 = module_1.module(var_5, var_4)
    var_7 = 'test_pattern'
    var_8 = 'test_section'
    var_9 = (var_7, var_8)
    var_10 = [var_9]
    var_11 = module_0.Config()
    var_12 = 'test_pattern.module'
    var_13 = module_1.module(var_12, var_11)
    assert var_13 == 'test_section'
    var_14 = '/path/to/src'
    var_15 = 'src_module'
    var_16 = module_1.module(var_15, var_11)
    var_17 = module_0.Config()
    var_18 = 'unknown_module'
    var_19 = module_1.module(var_18, var_17)
    var_20 = 'All tests passed!'
    var_21 = print(var_20)



# Parsed testcases at query #6
#--------------------------



def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = module_1.module(var_0, var_2)
    assert var_3 == 'test_module'
    var_4 = 'test_module.submodule'
    var_5 = module_1.module(var_4, var_2)
    assert var_5 == 'test_module'
    var_6 = module_0.Config()
    var_7 = '.local_module'
    var_8 = module_1.module(var_7, var_6)
    assert var_8 == 'LOCALFOLDER'
    var_9 = '^django'
    var_10 = 'THIRDPARTY'
    var_11 = (var_9, var_10)
    var_12 = [var_11]
    var_13 = module_0.Config()
    var_14 = 'django.contrib'
    var_15 = module_1.module(var_14, var_13)
    assert var_15 == 'THIRDPARTY'
    var_16 = '/path/to/src'
    var_17 = module_0.Config()
    var_18 = 'unknown_module'
    var_19 = module_1.module(var_18, var_17)
    var_20 = 'All tests passed!'
    var_21 = print(var_20)



# Parsed testcases at query #7
#--------------------------



def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = module_1.module(var_0, var_2)
    assert var_3 == 'test_module'
    var_4 = module_0.Config()
    var_5 = '.local_module'
    var_6 = module_1.module(var_5, var_4)
    var_7 = 'test.*'
    var_8 = 'test_section'
    var_9 = (var_7, var_8)
    var_10 = [var_9]
    var_11 = module_0.Config()
    var_12 = 'test.module'
    var_13 = module_1.module(var_12, var_11)
    assert var_13 == 'test_section'
    var_14 = module_0.Config()
    var_15 = 'unknown_module'
    var_16 = module_1.module(var_15, var_14)
    var_17 = 'All tests passed!'
    var_18 = print(var_17)



# Parsed testcases at query #8
#--------------------------



def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = module_1.module(var_0, var_2)
    assert var_3 == 'test_module'
    var_4 = module_0.Config()
    var_5 = '.local_module'
    var_6 = module_1.module(var_5, var_4)
    assert var_6 == 'LOCALFOLDER'
    var_7 = 'test_pattern'
    var_8 = 'KNOWN'
    var_9 = (var_7, var_8)
    var_10 = [var_9]
    var_11 = module_0.Config()
    var_12 = 'test_pattern.module'
    var_13 = module_1.module(var_12, var_11)
    assert var_13 == 'KNOWN'
    var_14 = module_0.Config()
    var_15 = 'unknown_module'
    var_16 = module_1.module(var_15, var_14)
    var_17 = 'All tests passed!'
    var_18 = print(var_17)



# Parsed testcases at query #9
#--------------------------




# Parsed testcases at query #10
#--------------------------



def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = module_1.module(var_0, var_2)
    assert var_3 == 'test_module'
    var_4 = module_0.Config()
    var_5 = '.local_module'
    var_6 = module_1.module(var_5, var_4)
    var_7 = '^django'
    var_8 = 'THIRDPARTY'
    var_9 = 'django.test'
    var_10 = module_1.module(var_9, var_4)
    assert var_10 == 'THIRDPARTY'
    var_11 = '/src'
    var_12 = 'my_module'
    var_13 = module_1.module(var_12, var_4)
    var_14 = module_0.Config()
    var_15 = 'unknown_module'
    var_16 = module_1.module(var_15, var_14)
    var_17 = 'All tests passed!'
    var_18 = print(var_17)



# Parsed testcases at query #11
#--------------------------



def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = module_1.module(var_0, var_2)
    assert var_3 == 'test_module'
    var_4 = module_0.Config()
    var_5 = '.local_module'
    var_6 = module_1.module(var_5, var_4)
    var_7 = 'test.*'
    var_8 = 'test_section'
    var_9 = (var_7, var_8)
    var_10 = [var_9]
    var_11 = module_0.Config()
    var_12 = module_1.module(var_0, var_11)
    assert var_12 == 'test_section'
    var_13 = '/path/to/src'
    var_14 = 'src_module'
    var_15 = module_1.module(var_14, var_11)
    var_16 = module_0.Config()
    var_17 = 'unknown_module'
    var_18 = module_1.module(var_17, var_16)



# Parsed testcases at query #12
#--------------------------



def test_case_0():
    var_0 = '.example'
    var_1 = module_0.Config()
    var_2 = module_1.module(var_0, var_1)
    assert var_2 == 'LOCALFOLDER'
    var_3 = 'example'
    var_4 = [var_3]
    var_5 = module_0.Config()
    var_6 = module_1.module(var_3, var_5)
    assert var_6 == 'example'
    var_7 = 'THIRDPARTY'
    var_8 = (var_3, var_7)
    var_9 = [var_8]
    var_10 = module_0.Config()
    var_11 = module_1.module(var_3, var_10)
    assert var_11 == 'THIRDPARTY'
    var_12 = '/path/to/src'
    var_13 = module_1.module(var_3, var_10)
    assert var_13 == 'FIRSTPARTY'
    var_14 = 'unknown'
    var_15 = module_0.Config()
    var_16 = module_1.module(var_14, var_15)
    assert var_16 == 'FIRSTPARTY'



# Parsed testcases at query #13
#--------------------------



def test_case_0():
    var_0 = 'test_pattern'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'test_pattern.module'
    var_4 = module_1.module(var_3, var_2)
    assert var_4 == 'test_pattern'
    var_5 = module_1.module(var_0, var_2)
    assert var_5 == 'test_pattern'
    var_6 = 'other.module'
    var_7 = module_1.module(var_6, var_2)
    var_8 = module_0.Config()
    var_9 = '.local_module'
    var_10 = module_1.module(var_9, var_8)
    assert var_10 == 'LOCALFOLDER'
    var_11 = 'local_module'
    var_12 = module_1.module(var_11, var_8)
    var_13 = '^django\\.*'
    var_14 = 'DJANGO'
    var_15 = 'django.app'
    var_16 = module_1.module(var_15, var_8)
    assert var_16 == 'DJANGO'
    var_17 = 'django'
    var_18 = module_1.module(var_17, var_8)
    assert var_18 == 'DJANGO'
    var_19 = 'flask.app'
    var_20 = module_1.module(var_19, var_8)
    var_21 = '/src'
    var_22 = 'THIRDPARTY'
    var_23 = module_0.Config()
    var_24 = 'unknown.module'
    var_25 = module_1.module(var_24, var_23)
    assert var_25 == 'THIRDPARTY'
    var_26 = 'All tests passed!'
    var_27 = print(var_26)



# Parsed testcases at query #14
#--------------------------



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
    var_9 = 'All tests passed!'
    var_10 = print(var_9)



# Parsed testcases at query #15
#--------------------------



def test_case_0():
    var_0 = 'test_pattern'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'test_pattern.module'
    var_4 = module_1.module(var_3, var_2)
    assert var_4 == 'test_pattern'
    var_5 = 'Test case 1 passed'
    var_6 = print(var_5)
    var_7 = module_0.Config()
    var_8 = '.local_module'
    var_9 = module_1.module(var_8, var_7)
    var_10 = 'Test case 2 passed'
    var_11 = print(var_10)
    var_12 = '^django\\.*'
    var_13 = 'THIRDPARTY'
    var_14 = 'django.app'
    var_15 = module_1.module(var_14, var_7)
    assert var_15 == 'THIRDPARTY'
    var_16 = 'Test case 3 passed'
    var_17 = print(var_16)
    var_18 = 'src'
    var_19 = var_0 / var_18
    var_20 = 'mymodule.py'
    var_21 = var_19 / var_20
    var_22 = [var_19]
    var_23 = module_0.Config()
    var_24 = 'mymodule'
    var_25 = module_1.module(var_24, var_23)
    var_26 = 'Test case 4 passed'
    var_27 = print(var_26)
    var_28 = module_0.Config()
    var_29 = 'unknown_module'
    var_30 = module_1.module(var_29, var_28)
    var_31 = 'Test case 5 passed'
    var_32 = print(var_31)
    var_33 = 'All tests passed!'
    var_34 = print(var_33)



# Parsed testcases at query #16
#--------------------------



def test_case_0():
    var_0 = 'test_pattern'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'test_pattern.module'
    var_4 = module_1.module(var_3, var_2)
    assert var_4 == 'test_pattern'
    var_5 = module_0.Config()
    var_6 = '.local_module'
    var_7 = module_1.module(var_6, var_5)
    var_8 = '^known_module'
    var_9 = 'known_section'
    var_10 = 'known_module.submodule'
    var_11 = module_1.module(var_10, var_5)
    assert var_11 == 'known_section'
    var_12 = 'src'
    var_13 = var_0 / var_12
    var_14 = 'mymodule.py'
    var_15 = var_13 / var_14
    var_16 = [var_13]
    var_17 = module_0.Config()
    var_18 = 'mymodule'
    var_19 = module_1.module(var_18, var_17)
    var_20 = module_0.Config()
    var_21 = 'unknown_module'
    var_22 = module_1.module(var_21, var_20)
    var_23 = 'All tests passed!'
    var_24 = print(var_23)



# Parsed testcases at query #17
#--------------------------



def test_case_0():
    var_0 = 'test_pattern'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'test_pattern.module'
    var_4 = module_1.module(var_3, var_2)
    assert var_4 == 'test_pattern'
    var_5 = '.local_module'
    var_6 = module_1.module(var_5, var_2)
    var_7 = '^known_.*'
    var_8 = 'KNOWN'
    var_9 = 'known_module'
    var_10 = module_1.module(var_9, var_2)
    assert var_10 == 'KNOWN'
    var_11 = '/fake/path'
    var_12 = 'unknown_module'
    var_13 = module_1.module(var_12, var_2)



# Parsed testcases at query #18
#--------------------------



def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = module_1.module(var_0, var_2)
    assert var_3 == 'test_module'
    var_4 = module_0.Config()
    var_5 = '.local_module'
    var_6 = module_1.module(var_5, var_4)
    var_7 = 'test_pattern'
    var_8 = 'known_section'
    var_9 = (var_7, var_8)
    var_10 = [var_9]
    var_11 = module_0.Config()
    var_12 = 'test_pattern.module'
    var_13 = module_1.module(var_12, var_11)
    assert var_13 == 'known_section'
    var_14 = '/path/to/src'
    var_15 = 'src_module'
    var_16 = module_1.module(var_15, var_11)
    var_17 = module_0.Config()
    var_18 = 'unknown_module'
    var_19 = module_1.module(var_18, var_17)
    var_20 = 'All tests passed!'
    var_21 = print(var_20)



# Parsed testcases at query #19
#--------------------------



def test_case_0():
    var_0 = 'test_pattern'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'test_pattern.module'
    var_4 = module_1.module(var_3, var_2)
    assert var_4 == 'test_pattern'
    var_5 = module_0.Config()
    var_6 = '.local_module'
    var_7 = module_1.module(var_6, var_5)
    assert var_7 == 'LOCALFOLDER'
    var_8 = '^django'
    var_9 = 'THIRDPARTY'
    var_10 = (var_8, var_9)
    var_11 = [var_10]
    var_12 = module_0.Config()
    var_13 = 'django.app'
    var_14 = module_1.module(var_13, var_12)
    assert var_14 == 'THIRDPARTY'
    var_15 = '/src'
    var_16 = 'my_module'
    var_17 = module_1.module(var_16, var_12)
    assert var_17 == 'FIRSTPARTY'
    var_18 = module_0.Config()
    var_19 = 'unknown_module'
    var_20 = module_1.module(var_19, var_18)
    var_21 = 'All tests passed!'
    var_22 = print(var_21)



# Parsed testcases at query #20
#--------------------------




# Parsed testcases at query #21
#--------------------------



def test_case_0():
    var_0 = 'test_pattern'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'test_pattern.module'
    var_4 = module_1.module(var_3, var_2)
    assert var_4 == 'test_pattern'
    var_5 = 'Test case 1 passed'
    var_6 = print(var_5)
    var_7 = module_0.Config()
    var_8 = '.local_module'
    var_9 = module_1.module(var_8, var_7)
    var_10 = 'Test case 2 passed'
    var_11 = print(var_10)
    var_12 = '^django'
    var_13 = 'THIRDPARTY'
    var_14 = (var_12, var_13)
    var_15 = [var_14]
    var_16 = module_0.Config()
    var_17 = 'django.contrib.auth'
    var_18 = module_1.module(var_17, var_16)
    assert var_18 == 'THIRDPARTY'
    var_19 = 'Test case 3 passed'
    var_20 = print(var_19)
    var_21 = '/path/to/src'
    var_22 = 'my_module'
    var_23 = module_1.module(var_22, var_16)
    var_24 = 'Test case 4 passed'
    var_25 = print(var_24)
    var_26 = module_0.Config()
    var_27 = 'unknown_module'
    var_28 = module_1.module(var_27, var_26)
    var_29 = 'Test case 5 passed'
    var_30 = print(var_29)
    var_31 = 'All test cases passed'
    var_32 = print(var_31)



# Parsed testcases at query #22
#--------------------------



def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = module_1.module(var_0, var_2)
    assert var_3 == 'test_module'
    var_4 = 'Test case 1 passed'
    var_5 = print(var_4)
    var_6 = module_0.Config()
    var_7 = '.local_module'
    var_8 = module_1.module(var_7, var_6)
    var_9 = 'Test case 2 passed'
    var_10 = print(var_9)
    var_11 = 'test_pattern'
    var_12 = 'test_section'
    var_13 = (var_11, var_12)
    var_14 = [var_13]
    var_15 = module_0.Config()
    var_16 = 'test_pattern.module'
    var_17 = module_1.module(var_16, var_15)
    assert var_17 == 'test_section'
    var_18 = 'Test case 3 passed'
    var_19 = print(var_18)
    var_20 = module_0.Config()
    var_21 = 'unknown_module'
    var_22 = module_1.module(var_21, var_20)
    var_23 = 'Test case 4 passed'
    var_24 = print(var_23)
    var_25 = '/src'
    var_26 = 'src_module'
    var_27 = module_1.module(var_26, var_20)
    var_28 = 'Test case 5 passed'
    var_29 = print(var_28)
    var_30 = 'All test cases passed'
    var_31 = print(var_30)



# Parsed testcases at query #23
#--------------------------



def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = module_1.module(var_0, var_2)
    assert var_3 == 'test_module'
    var_4 = module_0.Config()
    var_5 = '.local_module'
    var_6 = module_1.module(var_5, var_4)
    var_7 = 'test_pattern'
    var_8 = 'test_section'
    var_9 = (var_7, var_8)
    var_10 = [var_9]
    var_11 = module_0.Config()
    var_12 = 'test_pattern.module'
    var_13 = module_1.module(var_12, var_11)
    assert var_13 == 'test_section'
    var_14 = '/path/to/src'
    var_15 = 'src_module'
    var_16 = module_1.module(var_15, var_11)
    var_17 = module_0.Config()
    var_18 = 'unknown_module'
    var_19 = module_1.module(var_18, var_17)
    var_20 = 'All tests passed!'
    var_21 = print(var_20)



# Parsed testcases at query #24
#--------------------------



def test_case_0():
    var_0 = 'test_pattern'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'test_pattern.module'
    var_4 = module_1.module(var_3, var_2)
    assert var_4 == 'test_pattern'
    var_5 = module_1.module(var_0, var_2)
    assert var_5 == 'test_pattern'
    var_6 = 'other.module'
    var_7 = module_1.module(var_6, var_2)
    var_8 = module_0.Config()
    var_9 = '.local_module'
    var_10 = module_1.module(var_9, var_8)
    var_11 = 'local_module'
    var_12 = module_1.module(var_11, var_8)
    var_13 = '^django'
    var_14 = 'THIRDPARTY'
    var_15 = (var_13, var_14)
    var_16 = [var_15]
    var_17 = module_0.Config()
    var_18 = 'django.module'
    var_19 = module_1.module(var_18, var_17)
    assert var_19 == 'THIRDPARTY'
    var_20 = module_1.module(var_6, var_17)
    var_21 = '/src'
    var_22 = module_0.Config()
    var_23 = 'unknown.module'
    var_24 = module_1.module(var_23, var_22)



# Parsed testcases at query #25
#--------------------------



def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'requests'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'THIRDPARTY'
    var_5 = 'my_local_module'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'FIRSTPARTY'
    var_7 = '.local_module'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'LOCALFOLDER'
    var_9 = 'my_forced_separate_module'
    var_10 = module_1.module(var_9, var_0)
    assert var_10 == 'my_forced_separate_module'
    var_11 = 'my_known_pattern_module'
    var_12 = module_1.module(var_11, var_0)
    assert var_12 == 'my_known_pattern_module'
    var_13 = 'my_src_path_module'
    var_14 = module_1.module(var_13, var_0)
    assert var_14 == 'FIRSTPARTY'



# Parsed testcases at query #26
#--------------------------




# Parsed testcases at query #27
#--------------------------



def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = module_1.module(var_0, var_2)
    assert var_3 == 'test_module'
    var_4 = module_0.Config()
    var_5 = '.local_module'
    var_6 = module_1.module(var_5, var_4)
    var_7 = 'test_pattern'
    var_8 = 'test_section'
    var_9 = (var_7, var_8)
    var_10 = [var_9]
    var_11 = module_0.Config()
    var_12 = 'test_pattern.module'
    var_13 = module_1.module(var_12, var_11)
    assert var_13 == 'test_section'
    var_14 = module_0.Config()
    var_15 = 'unknown_module'
    var_16 = module_1.module(var_15, var_14)
    var_17 = 'All tests passed!'
    var_18 = print(var_17)



# Parsed testcases at query #28
#--------------------------



def test_case_0():
    var_0 = 'test*'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'test_module'
    var_4 = module_1.module(var_3, var_2)
    assert var_4 == 'test'
    var_5 = 'test'
    var_6 = module_1.module(var_5, var_2)
    assert var_6 == 'test'
    var_7 = 'test.module'
    var_8 = module_1.module(var_7, var_2)
    assert var_8 == 'test'
    var_9 = 'other'
    var_10 = module_1.module(var_9, var_2)
    var_11 = module_0.Config()
    var_12 = '.local_module'
    var_13 = module_1.module(var_12, var_11)
    var_14 = 'local_module'
    var_15 = module_1.module(var_14, var_11)
    var_16 = '^django.*'
    var_17 = 'DJANGO'
    var_18 = 'django.test'
    var_19 = module_1.module(var_18, var_11)
    assert var_19 == 'DJANGO'
    var_20 = 'django'
    var_21 = module_1.module(var_20, var_11)
    assert var_21 == 'DJANGO'
    var_22 = module_1.module(var_9, var_11)
    var_23 = '/path/to/src'
    var_24 = 'my_module'
    var_25 = module_1.module(var_24, var_11)
    var_26 = 'my_module'
    var_27 = module_1.module(var_26, var_11)
    var_28 = 'THIRDPARTY'
    var_29 = module_0.Config()
    var_30 = 'unknown_module'
    var_31 = module_1.module(var_30, var_29)
    assert var_31 == 'THIRDPARTY'
    var_32 = 'All tests passed!'
    var_33 = print(var_32)



# Parsed testcases at query #29
#--------------------------



def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = module_1.module(var_0, var_2)
    assert var_3 == 'test_module'
    var_4 = 'Test case 1 passed'
    var_5 = print(var_4)
    var_6 = module_0.Config()
    var_7 = '.local_module'
    var_8 = module_1.module(var_7, var_6)
    var_9 = 'Test case 2 passed'
    var_10 = print(var_9)
    var_11 = 'test_pattern'
    var_12 = 'test_section'
    var_13 = (var_11, var_12)
    var_14 = [var_13]
    var_15 = module_0.Config()
    var_16 = 'test_pattern.module'
    var_17 = module_1.module(var_16, var_15)
    assert var_17 == 'test_section'
    var_18 = 'Test case 3 passed'
    var_19 = print(var_18)
    var_20 = '/fake/path'
    var_21 = 'fake_module'
    var_22 = module_1.module(var_21, var_15)
    var_23 = 'Test case 4 passed'
    var_24 = print(var_23)
    var_25 = module_0.Config()
    var_26 = 'unknown_module'
    var_27 = module_1.module(var_26, var_25)
    var_28 = 'Test case 5 passed'
    var_29 = print(var_28)
    var_30 = 'All tests passed!'
    var_31 = print(var_30)



# Parsed testcases at query #30
#--------------------------



def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = module_1.module(var_0, var_2)
    assert var_3 == 'test_module'
    var_4 = '.local_module'
    var_5 = module_1.module(var_4, var_2)
    var_6 = '^django'
    var_7 = 'THIRDPARTY'
    var_8 = (var_6, var_7)
    var_9 = [var_8]
    var_10 = module_0.Config()
    var_11 = 'django.contrib'
    var_12 = module_1.module(var_11, var_10)
    assert var_12 == 'THIRDPARTY'
    var_13 = 'unknown_module'
    var_14 = module_1.module(var_13, var_10)
    var_15 = 'All tests passed!'
    var_16 = print(var_15)



# Parsed testcases at query #31
#--------------------------



def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = module_1.module(var_0, var_2)
    assert var_3 == 'test_module'
    var_4 = module_0.Config()
    var_5 = '.local_module'
    var_6 = module_1.module(var_5, var_4)
    assert var_6 == 'LOCALFOLDER'
    var_7 = 'test.*'
    var_8 = 'THIRDPARTY'
    var_9 = (var_7, var_8)
    var_10 = [var_9]
    var_11 = module_0.Config()
    var_12 = 'test.module'
    var_13 = module_1.module(var_12, var_11)
    assert var_13 == 'THIRDPARTY'
    var_14 = '/path/to/src'
    var_15 = 'src_module'
    var_16 = module_1.module(var_15, var_11)
    assert var_16 == 'FIRSTPARTY'
    var_17 = module_0.Config()
    var_18 = 'unknown_module'
    var_19 = module_1.module(var_18, var_17)
    var_20 = 'All tests passed!'
    var_21 = print(var_20)



# Parsed testcases at query #32
#--------------------------



def test_case_0():
    var_0 = 'test_pattern'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'test_pattern.module'
    var_4 = module_1.module(var_3, var_2)
    assert var_4 == 'test_pattern'
    var_5 = module_1.module(var_0, var_2)
    assert var_5 == 'test_pattern'
    var_6 = 'other.module'
    var_7 = module_1.module(var_6, var_2)
    var_8 = module_0.Config()
    var_9 = '.local_module'
    var_10 = module_1.module(var_9, var_8)
    assert var_10 == 'LOCALFOLDER'
    var_11 = 'not_local'
    var_12 = module_1.module(var_11, var_8)
    var_13 = 'test.*'
    var_14 = 'TEST_SECTION'
    var_15 = (var_13, var_14)
    var_16 = [var_15]
    var_17 = module_0.Config()
    var_18 = 'test.module'
    var_19 = module_1.module(var_18, var_17)
    assert var_19 == 'TEST_SECTION'
    var_20 = 'test'
    var_21 = module_1.module(var_20, var_17)
    assert var_21 == 'TEST_SECTION'
    var_22 = module_1.module(var_6, var_17)
    var_23 = 'DEFAULT'
    var_24 = module_0.Config()
    var_25 = 'unknown.module'
    var_26 = module_1.module(var_25, var_24)
    assert var_26 == 'DEFAULT'
    var_27 = 'All tests passed!'
    var_28 = print(var_27)



# Parsed testcases at query #33
#--------------------------



def test_case_0():
    var_0 = 'test_pattern'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'test_pattern.module'
    var_4 = module_1.module(var_3, var_2)
    assert var_4 == 'test_pattern'
    var_5 = module_0.Config()
    var_6 = '.local_module'
    var_7 = module_1.module(var_6, var_5)
    assert var_7 == 'LOCALFOLDER'
    var_8 = "re.compile(r'^django')"
    var_9 = 'THIRDPARTY'
    var_10 = (var_8, var_9)
    var_11 = [var_10]
    var_12 = module_0.Config()
    var_13 = 'django.app'
    var_14 = module_1.module(var_13, var_12)
    assert var_14 == 'THIRDPARTY'
    var_15 = '/path/to/src'
    var_16 = 'src_module'
    var_17 = module_1.module(var_16, var_12)
    assert var_17 == 'FIRSTPARTY'
    var_18 = module_0.Config()
    var_19 = 'unknown_module'
    var_20 = module_1.module(var_19, var_18)
    var_21 = 'All tests passed!'
    var_22 = print(var_21)



####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------



def test_case_0():
    var_0 = 'test_pattern'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'test_pattern.module'
    var_4 = module_1.module(var_3, var_2)
    assert var_4 == 'test_pattern'
    var_5 = module_0.Config()
    var_6 = '.local_module'
    var_7 = module_1.module(var_6, var_5)
    assert var_7 == 'LOCALFOLDER'
    var_8 = '^django'
    var_9 = 'THIRDPARTY'
    var_10 = 'django.contrib'
    var_11 = module_1.module(var_10, var_5)
    assert var_11 == 'THIRDPARTY'
    var_12 = '/path/to/src'
    var_13 = True
    var_14 = lambda path: var_13
    var_15 = 'my_module'
    var_16 = module_1.module(var_15, var_5)
    assert var_16 == 'FIRSTPARTY'
    var_17 = module_0.Config()
    var_18 = 'unknown_module'
    var_19 = module_1.module(var_18, var_17)
    var_20 = 'All tests passed!'
    var_21 = print(var_20)



# Parsed testcases at query #2
#--------------------------



def test_case_0():
    var_0 = 'test_pattern'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'test_pattern.module'
    var_4 = module_1.module(var_3, var_2)
    assert var_4 == 'test_pattern'
    var_5 = 'Test case 1 passed'
    var_6 = print(var_5)
    var_7 = module_0.Config()
    var_8 = '.local_module'
    var_9 = module_1.module(var_8, var_7)
    var_10 = 'Test case 2 passed'
    var_11 = print(var_10)
    var_12 = '^django'
    var_13 = 'THIRDPARTY'
    var_14 = 'django.app'
    var_15 = module_1.module(var_14, var_7)
    assert var_15 == 'THIRDPARTY'
    var_16 = 'Test case 3 passed'
    var_17 = print(var_16)
    var_18 = module_0.Config()
    var_19 = 'unknown_module'
    var_20 = module_1.module(var_19, var_18)
    var_21 = 'Test case 4 passed'
    var_22 = print(var_21)
    var_23 = 'Test case 5 skipped (requires filesystem mocking)'
    var_24 = print(var_23)
    var_25 = 'All tests passed!'
    var_26 = print(var_25)



# Parsed testcases at query #3
#--------------------------



def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = module_1.module(var_0, var_2)
    assert var_3 == 'test_module'
    var_4 = module_0.Config()
    var_5 = '.local_module'
    var_6 = module_1.module(var_5, var_4)
    var_7 = 'test.*'
    var_8 = 'TEST'
    var_9 = (var_7, var_8)
    var_10 = [var_9]
    var_11 = module_0.Config()
    var_12 = module_1.module(var_0, var_11)
    assert var_12 == 'TEST'
    var_13 = '/path/to/src'
    var_14 = 'src_module'
    var_15 = module_1.module(var_14, var_11)
    var_16 = module_0.Config()
    var_17 = 'unknown_module'
    var_18 = module_1.module(var_17, var_16)



# Parsed testcases at query #4
#--------------------------



def test_case_0():
    var_0 = 'test_pattern'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'test_pattern.module'
    var_4 = module_1.module(var_3, var_2)
    assert var_4 == 'test_pattern'
    var_5 = module_0.Config()
    var_6 = '.local_module'
    var_7 = module_1.module(var_6, var_5)
    assert var_7 == 'LOCALFOLDER'
    var_8 = 'test.*'
    var_9 = 'THIRDPARTY'
    var_10 = (var_8, var_9)
    var_11 = [var_10]
    var_12 = module_0.Config()
    var_13 = 'test.module'
    var_14 = module_1.module(var_13, var_12)
    assert var_14 == 'THIRDPARTY'
    var_15 = '/path/to/src'
    var_16 = 'src_module'
    var_17 = module_1.module(var_16, var_12)
    assert var_17 == 'FIRSTPARTY'
    var_18 = module_0.Config()
    var_19 = 'unknown_module'
    var_20 = module_1.module(var_19, var_18)
    var_21 = 'All tests passed!'
    var_22 = print(var_21)



# Parsed testcases at query #5
#--------------------------



def test_case_0():
    var_0 = 'test*'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'test_module'
    var_4 = module_1.module(var_3, var_2)
    assert var_4 == 'test*'
    var_5 = module_0.Config()
    var_6 = '.local_module'
    var_7 = module_1.module(var_6, var_5)
    assert var_7 == 'LOCALFOLDER'
    var_8 = '^django'
    var_9 = 'THIRDPARTY'
    var_10 = (var_8, var_9)
    var_11 = [var_10]
    var_12 = module_0.Config()
    var_13 = 'django.contrib'
    var_14 = module_1.module(var_13, var_12)
    assert var_14 == 'THIRDPARTY'
    var_15 = module_0.Config()
    var_16 = 'unknown_module'
    var_17 = module_1.module(var_16, var_15)
    var_18 = '/path/to/src'



# Parsed testcases at query #6
#--------------------------



def test_case_0():
    var_0 = 'test_pattern'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'test_pattern.module'
    var_4 = module_1.module(var_3, var_2)
    assert var_4 == 'test_pattern'
    var_5 = module_0.Config()
    var_6 = '.local_module'
    var_7 = module_1.module(var_6, var_5)
    assert var_7 == 'LOCALFOLDER'
    var_8 = 'test.*'
    var_9 = 'THIRDPARTY'
    var_10 = (var_8, var_9)
    var_11 = [var_10]
    var_12 = module_0.Config()
    var_13 = 'test.module'
    var_14 = module_1.module(var_13, var_12)
    assert var_14 == 'THIRDPARTY'
    var_15 = '/path/to/src'
    var_16 = 'src_module'
    var_17 = module_1.module(var_16, var_12)
    assert var_17 == 'FIRSTPARTY'
    var_18 = module_0.Config()
    var_19 = 'unknown_module'
    var_20 = module_1.module(var_19, var_18)
    assert var_20 == 'STDLIB'



# Parsed testcases at query #7
#--------------------------



def test_case_0():
    var_0 = 'test_pattern'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'test_pattern.module'
    var_4 = module_1.module(var_3, var_2)
    assert var_4 == 'test_pattern'
    var_5 = module_0.Config()
    var_6 = '.local_module'
    var_7 = module_1.module(var_6, var_5)
    assert var_7 == 'LOCALFOLDER'
    var_8 = 'test.*'
    var_9 = 'test_section'
    var_10 = (var_8, var_9)
    var_11 = [var_10]
    var_12 = module_0.Config()
    var_13 = 'test.module'
    var_14 = module_1.module(var_13, var_12)
    assert var_14 == 'test_section'
    var_15 = module_0.Config()
    var_16 = 'unknown_module'
    var_17 = module_1.module(var_16, var_15)
    assert var_17 == 'FIRSTPARTY'
    var_18 = 'All tests passed!'
    var_19 = print(var_18)



# Parsed testcases at query #8
#--------------------------



def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = module_1.module(var_0, var_2)
    assert var_3 == 'test_module'
    var_4 = module_0.Config()
    var_5 = '.local_module'
    var_6 = module_1.module(var_5, var_4)
    assert var_6 == 'LOCALFOLDER'
    var_7 = '^django'
    var_8 = 'THIRDPARTY'
    var_9 = 'django.contrib'
    var_10 = module_1.module(var_9, var_4)
    assert var_10 == 'THIRDPARTY'
    var_11 = '/path/to/src'
    var_12 = module_0.Config()
    var_13 = 'unknown_module'
    var_14 = module_1.module(var_13, var_12)



# Parsed testcases at query #9
#--------------------------




# Parsed testcases at query #10
#--------------------------



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
    var_7 = '.relative_module'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'LOCALFOLDER'
    var_9 = 'my_forced_separate_module'
    var_10 = module_1.module(var_9, var_0)
    assert var_10 == 'my_forced_separate_module'



# Parsed testcases at query #11
#--------------------------



def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = module_1.module(var_0, var_2)
    assert var_3 == 'test_module'
    var_4 = 'test_module.submodule'
    var_5 = module_1.module(var_4, var_2)
    assert var_5 == 'test_module'
    var_6 = module_0.Config()
    var_7 = '.local_module'
    var_8 = module_1.module(var_7, var_6)
    assert var_8 == 'LOCALFOLDER'
    var_9 = '^django'
    var_10 = 'THIRDPARTY'
    var_11 = (var_9, var_10)
    var_12 = [var_11]
    var_13 = module_0.Config()
    var_14 = 'django.contrib'
    var_15 = module_1.module(var_14, var_13)
    assert var_15 == 'THIRDPARTY'
    var_16 = '/path/to/src'
    var_17 = module_0.Config()
    var_18 = 'unknown_module'
    var_19 = module_1.module(var_18, var_17)
    var_20 = 'All tests passed!'
    var_21 = print(var_20)



# Parsed testcases at query #12
#--------------------------




# Parsed testcases at query #13
#--------------------------




# Parsed testcases at query #14
#--------------------------




# Parsed testcases at query #15
#--------------------------




# Parsed testcases at query #16
#--------------------------



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
    var_7 = '.relative_module'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'LOCALFOLDER'
    var_9 = '_private_module'
    var_10 = module_1.module(var_9, var_0)
    assert var_10 == 'FIRSTPARTY'
    var_11 = '__main__'
    var_12 = module_1.module(var_11, var_0)
    assert var_12 == 'FIRSTPARTY'
    var_13 = 'my.namespace.module'
    var_14 = module_1.module(var_13, var_0)
    assert var_14 == 'FIRSTPARTY'
    var_15 = 'my.forced.separate.module'
    var_16 = module_1.module(var_15, var_0)
    assert var_16 == 'my.forced.separate'
    var_17 = 'my.known.pattern.module'
    var_18 = module_1.module(var_17, var_0)
    assert var_18 == 'my.known.pattern'
    var_19 = 'my.src.path.module'
    var_20 = module_1.module(var_19, var_0)
    assert var_20 == 'FIRSTPARTY'
    var_21 = 'my.namespace.package.module'
    var_22 = module_1.module(var_21, var_0)
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'my.namespace.package'
    var_24 = module_1.module(var_23, var_0)
    assert var_24 == 'FIRSTPARTY'
    var_25 = 'my.namespace'
    var_26 = module_1.module(var_25, var_0)
    assert var_26 == 'FIRSTPARTY'
    var_27 = 'my'
    var_28 = module_1.module(var_27, var_0)
    assert var_28 == 'FIRSTPARTY'
    var_29 = 'my.module'
    var_30 = module_1.module(var_29, var_0)
    assert var_30 == 'FIRSTPARTY'
    var_31 = 'my.module.submodule'
    var_32 = module_1.module(var_31, var_0)
    assert var_32 == 'FIRSTPARTY'
    var_33 = 'my.module.submodule.deep'
    var_34 = module_1.module(var_33, var_0)
    assert var_34 == 'FIRSTPARTY'
    var_35 = 'my.module.submodule.deep.deeper'
    var_36 = module_1.module(var_35, var_0)
    assert var_36 == 'FIRSTPARTY'
    var_37 = 'my.module.submodule.deep.deeper.deepest'
    var_38 = module_1.module(var_37, var_0)
    assert var_38 == 'FIRSTPARTY'
    var_39 = 'my.module.submodule.deep.deeper.deepest.and.beyond'
    var_40 = module_1.module(var_39, var_0)
    assert var_40 == 'FIRSTPARTY'
    var_41 = 'my.module.submodule.deep.deeper.deepest.and.beyond.infinity'
    var_42 = module_1.module(var_41, var_0)
    assert var_42 == 'FIRSTPARTY'
    var_43 = 'my.module.submodule.deep.deeper.deepest.and.beyond.infinity.and.beyond'
    var_44 = module_1.module(var_43, var_0)
    assert var_44 == 'FIRSTPARTY'
    var_45 = 'my.module.submodule.deep.deeper.deepest.and.beyond.infinity.and.beyond.infinity'
    var_46 = module_1.module(var_45, var_0)
    assert var_46 == 'FIRSTPARTY'
    var_47 = 'my.module.submodule.deep.deeper.deepest.and.beyond.infinity.and.beyond.infinity.and.beyond'
    var_48 = module_1.module(var_47, var_0)
    assert var_48 == 'FIRSTPARTY'
    var_49 = 'my.module.submodule.deep.deeper.deepest.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity'
    var_50 = module_1.module(var_49, var_0)
    assert var_50 == 'FIRSTPARTY'
    var_51 = 'my.module.submodule.deep.deeper.deepest.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond'
    var_52 = module_1.module(var_51, var_0)
    assert var_52 == 'FIRSTPARTY'
    var_53 = 'my.module.submodule.deep.deeper.deepest.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity'
    var_54 = module_1.module(var_53, var_0)
    assert var_54 == 'FIRSTPARTY'
    var_55 = 'my.module.submodule.deep.deeper.deepest.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond'
    var_56 = module_1.module(var_55, var_0)
    assert var_56 == 'FIRSTPARTY'
    var_57 = 'my.module.submodule.deep.deeper.deepest.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity'
    var_58 = module_1.module(var_57, var_0)
    assert var_58 == 'FIRSTPARTY'
    var_59 = 'my.module.submodule.deep.deeper.deepest.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond'
    var_60 = module_1.module(var_59, var_0)
    assert var_60 == 'FIRSTPARTY'
    var_61 = 'my.module.submodule.deep.deeper.deepest.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity'
    var_62 = module_1.module(var_61, var_0)
    assert var_62 == 'FIRSTPARTY'
    var_63 = 'my.module.submodule.deep.deeper.deepest.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond'
    var_64 = module_1.module(var_63, var_0)
    assert var_64 == 'FIRSTPARTY'
    var_65 = 'my.module.submodule.deep.deeper.deepest.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity'
    var_66 = module_1.module(var_65, var_0)
    assert var_66 == 'FIRSTPARTY'
    var_67 = 'my.module.submodule.deep.deeper.deepest.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond'
    var_68 = module_1.module(var_67, var_0)
    assert var_68 == 'FIRSTPARTY'
    var_69 = 'my.module.submodule.deep.deeper.deepest.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity'
    var_70 = module_1.module(var_69, var_0)
    assert var_70 == 'FIRSTPARTY'
    var_71 = 'my.module.submodule.deep.deeper.deepest.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond'
    var_72 = module_1.module(var_71, var_0)
    assert var_72 == 'FIRSTPARTY'
    var_73 = 'my.module.submodule.deep.deeper.deepest.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity'
    var_74 = module_1.module(var_73, var_0)
    assert var_74 == 'FIRSTPARTY'
    var_75 = 'my.module.submodule.deep.deeper.deepest.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond'
    var_76 = module_1.module(var_75, var_0)
    assert var_76 == 'FIRSTPARTY'
    var_77 = 'my.module.submodule.deep.deeper.deepest.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity'
    var_78 = module_1.module(var_77, var_0)
    assert var_78 == 'FIRSTPARTY'
    var_79 = 'my.module.submodule.deep.deeper.deepest.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond'
    var_80 = module_1.module(var_79, var_0)
    assert var_80 == 'FIRSTPARTY'
    var_81 = 'my.module.submodule.deep.deeper.deepest.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity'
    var_82 = module_1.module(var_81, var_0)
    assert var_82 == 'FIRSTPARTY'
    var_83 = 'my.module.submodule.deep.deeper.deepest.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond'
    var_84 = module_1.module(var_83, var_0)
    assert var_84 == 'FIRSTPARTY'
    var_85 = 'my.module.submodule.deep.deeper.deepest.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity'
    var_86 = module_1.module(var_85, var_0)
    assert var_86 == 'FIRSTPARTY'
    var_87 = 'my.module.submodule.deep.deeper.deepest.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond'
    var_88 = module_1.module(var_87, var_0)
    assert var_88 == 'FIRSTPARTY'
    var_89 = 'my.module.submodule.deep.deeper.deepest.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity.and.beyond.infinity'
    var_90 = module_1.module(var_89, var_0)
    assert var_90 == 'FIRSTPARTY'



# Parsed testcases at query #17
#--------------------------



def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'requests'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'THIRDPARTY'
    var_5 = 'my_local_module'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'FIRSTPARTY'
    var_7 = '.relative_module'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'LOCALFOLDER'
    var_9 = 'All tests passed!'
    var_10 = print(var_9)



# Parsed testcases at query #18
#--------------------------



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
    var_7 = '.relative_module'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'LOCALFOLDER'
    var_9 = 'my_forced_separate_module'
    var_10 = module_1.module(var_9, var_0)
    assert var_10 == 'my_forced_separate_module'
    var_11 = 'my_known_pattern_module'
    var_12 = module_1.module(var_11, var_0)
    assert var_12 == 'my_known_pattern_module'
    var_13 = 'my_src_path_module'
    var_14 = module_1.module(var_13, var_0)
    assert var_14 == 'FIRSTPARTY'
    var_15 = 'my_namespace_package'
    var_16 = module_1.module(var_15, var_0)
    assert var_16 == 'FIRSTPARTY'
    var_17 = 'my_namespace_package.submodule'
    var_18 = module_1.module(var_17, var_0)
    assert var_18 == 'FIRSTPARTY'
    var_19 = 'my_namespace_package.submodule.subsubmodule'
    var_20 = module_1.module(var_19, var_0)
    assert var_20 == 'FIRSTPARTY'
    var_21 = 'my_namespace_package.submodule.subsubmodule.subsubsubmodule'
    var_22 = module_1.module(var_21, var_0)
    assert var_22 == 'FIRSTPARTY'
    var_23 = 'my_namespace_package.submodule.subsubmodule.subsubsubmodule.subsubsubsubmodule'
    var_24 = module_1.module(var_23, var_0)
    assert var_24 == 'FIRSTPARTY'
    var_25 = 'my_namespace_package.submodule.subsubmodule.subsubsubmodule.subsubsubsubmodule.subsubsubsubsubmodule'
    var_26 = module_1.module(var_25, var_0)
    assert var_26 == 'FIRSTPARTY'
    var_27 = 'my_namespace_package.submodule.subsubmodule.subsubsubmodule.subsubsubsubmodule.subsubsubsubsubmodule.subsubsubsubsubsubmodule'
    var_28 = module_1.module(var_27, var_0)
    assert var_28 == 'FIRSTPARTY'
    var_29 = 'my_namespace_package.submodule.subsubmodule.subsubsubmodule.subsubsubsubmodule.subsubsubsubsubmodule.subsubsubsubsubsubmodule.subsubsubsubsubsubsubmodule'
    var_30 = module_1.module(var_29, var_0)
    assert var_30 == 'FIRSTPARTY'
    var_31 = 'my_namespace_package.submodule.subsubmodule.subsubsubmodule.subsubsubsubmodule.subsubsubsubsubmodule.subsubsubsubsubsubmodule.subsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubmodule'
    var_32 = module_1.module(var_31, var_0)
    assert var_32 == 'FIRSTPARTY'
    var_33 = 'my_namespace_package.submodule.subsubmodule.subsubsubmodule.subsubsubsubmodule.subsubsubsubsubmodule.subsubsubsubsubsubmodule.subsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubmodule'
    var_34 = module_1.module(var_33, var_0)
    assert var_34 == 'FIRSTPARTY'
    var_35 = 'my_namespace_package.submodule.subsubmodule.subsubsubmodule.subsubsubsubmodule.subsubsubsubsubmodule.subsubsubsubsubsubmodule.subsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubmodule'
    var_36 = module_1.module(var_35, var_0)
    assert var_36 == 'FIRSTPARTY'
    var_37 = 'my_namespace_package.submodule.subsubmodule.subsubsubmodule.subsubsubsubmodule.subsubsubsubsubmodule.subsubsubsubsubsubmodule.subsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubmodule'
    var_38 = module_1.module(var_37, var_0)
    assert var_38 == 'FIRSTPARTY'
    var_39 = 'my_namespace_package.submodule.subsubmodule.subsubsubmodule.subsubsubsubmodule.subsubsubsubsubmodule.subsubsubsubsubsubmodule.subsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubmodule'
    var_40 = module_1.module(var_39, var_0)
    assert var_40 == 'FIRSTPARTY'
    var_41 = 'my_namespace_package.submodule.subsubmodule.subsubsubmodule.subsubsubsubmodule.subsubsubsubsubmodule.subsubsubsubsubsubmodule.subsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubmodule'
    var_42 = module_1.module(var_41, var_0)
    assert var_42 == 'FIRSTPARTY'
    var_43 = 'my_namespace_package.submodule.subsubmodule.subsubsubmodule.subsubsubsubmodule.subsubsubsubsubmodule.subsubsubsubsubsubmodule.subsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubmodule'
    var_44 = module_1.module(var_43, var_0)
    assert var_44 == 'FIRSTPARTY'
    var_45 = 'my_namespace_package.submodule.subsubmodule.subsubsubmodule.subsubsubsubmodule.subsubsubsubsubmodule.subsubsubsubsubsubmodule.subsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubsubmodule'
    var_46 = module_1.module(var_45, var_0)
    assert var_46 == 'FIRSTPARTY'
    var_47 = 'my_namespace_package.submodule.subsubmodule.subsubsubmodule.subsubsubsubmodule.subsubsubsubsubmodule.subsubsubsubsubsubmodule.subsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubsubsubmodule'
    var_48 = module_1.module(var_47, var_0)
    assert var_48 == 'FIRSTPARTY'
    var_49 = 'my_namespace_package.submodule.subsubmodule.subsubsubmodule.subsubsubsubmodule.subsubsubsubsubmodule.subsubsubsubsubsubmodule.subsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubsubsubsubmodule'
    var_50 = module_1.module(var_49, var_0)
    assert var_50 == 'FIRSTPARTY'
    var_51 = 'my_namespace_package.submodule.subsubmodule.subsubsubmodule.subsubsubsubmodule.subsubsubsubsubmodule.subsubsubsubsubsubmodule.subsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubsubsubsubsubmodule'
    var_52 = module_1.module(var_51, var_0)
    assert var_52 == 'FIRSTPARTY'
    var_53 = 'my_namespace_package.submodule.subsubmodule.subsubsubmodule.subsubsubsubmodule.subsubsubsubsubmodule.subsubsubsubsubsubmodule.subsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubsubsubsubsubmodule.subsubsubsubsubsubsubsubsubsubsubsubsubsubsubsubsubsubsubmodule'
    var_54 = module_1.module(var_53, var_0)
    assert var_54 == 'FIRSTPARTY'



# Parsed testcases at query #19
#--------------------------



def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = module_1.module(var_0, var_2)
    assert var_3 == 'test_module'
    var_4 = 'Test case 1 passed'
    var_5 = print(var_4)
    var_6 = module_0.Config()
    var_7 = '.local_module'
    var_8 = module_1.module(var_7, var_6)
    var_9 = 'Test case 2 passed'
    var_10 = print(var_9)
    var_11 = 'test_pattern'
    var_12 = 'test_section'
    var_13 = (var_11, var_12)
    var_14 = [var_13]
    var_15 = module_0.Config()
    var_16 = 'test_pattern.module'
    var_17 = module_1.module(var_16, var_15)
    assert var_17 == 'test_section'
    var_18 = 'Test case 3 passed'
    var_19 = print(var_18)
    var_20 = '/src'
    var_21 = 'src_module'
    var_22 = module_1.module(var_21, var_15)
    var_23 = 'Test case 4 passed'
    var_24 = print(var_23)
    var_25 = module_0.Config()
    var_26 = 'unknown_module'
    var_27 = module_1.module(var_26, var_25)
    var_28 = 'Test case 5 passed'
    var_29 = print(var_28)



# Parsed testcases at query #20
#--------------------------



def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'os'
    var_2 = module_1.module(var_1, var_0)
    assert var_2 == 'STDLIB'
    var_3 = 'numpy'
    var_4 = module_1.module(var_3, var_0)
    assert var_4 == 'THIRDPARTY'
    var_5 = 'my_local_module'
    var_6 = module_1.module(var_5, var_0)
    assert var_6 == 'FIRSTPARTY'
    var_7 = '.local_module'
    var_8 = module_1.module(var_7, var_0)
    assert var_8 == 'LOCALFOLDER'



# Parsed testcases at query #21
#--------------------------



def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = module_1.module(var_0, var_2)
    assert var_3 == 'test_module'
    var_4 = 'test_module.submodule'
    var_5 = module_1.module(var_4, var_2)
    assert var_5 == 'test_module'
    var_6 = module_0.Config()
    var_7 = '.local_module'
    var_8 = module_1.module(var_7, var_6)
    assert var_8 == 'LOCALFOLDER'
    var_9 = '^django\\.*'
    var_10 = 'THIRDPARTY'
    var_11 = 'django.test'
    var_12 = module_1.module(var_11, var_6)
    assert var_12 == 'THIRDPARTY'
    var_13 = '/path/to/src'
    var_14 = module_0.Config()
    var_15 = 'unknown_module'
    var_16 = module_1.module(var_15, var_14)
    var_17 = 'All tests passed!'
    var_18 = print(var_17)



# Parsed testcases at query #22
#--------------------------



def test_case_0():
    var_0 = 'test_pattern'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'test_pattern.module'
    var_4 = module_1.module(var_3, var_2)
    assert var_4 == 'test_pattern'
    var_5 = '.local_module'
    var_6 = module_1.module(var_5, var_2)
    assert var_6 == 'LOCALFOLDER'
    var_7 = '^known\\.pattern'
    var_8 = 'KNOWN'
    var_9 = 'known.pattern.module'
    var_10 = module_1.module(var_9, var_2)
    assert var_10 == 'KNOWN'
    var_11 = '/fake/path'
    var_12 = True
    var_13 = lambda path: var_12
    var_14 = 'fake_module'
    var_15 = module_1.module(var_14, var_2)
    assert var_15 == 'FIRSTPARTY'
    var_16 = 'unknown_module'
    var_17 = module_1.module(var_16, var_2)
    var_18 = 'All tests passed!'
    var_19 = print(var_18)



# Parsed testcases at query #23
#--------------------------



def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = module_1.module(var_0, var_2)
    assert var_3 == 'test_module'
    var_4 = 'test_module.submodule'
    var_5 = module_1.module(var_4, var_2)
    assert var_5 == 'test_module'
    var_6 = module_0.Config()
    var_7 = '.local_module'
    var_8 = module_1.module(var_7, var_6)
    assert var_8 == 'LOCALFOLDER'
    var_9 = '^django'
    var_10 = 'THIRDPARTY'
    var_11 = (var_9, var_10)
    var_12 = [var_11]
    var_13 = module_0.Config()
    var_14 = 'django.contrib'
    var_15 = module_1.module(var_14, var_13)
    assert var_15 == 'THIRDPARTY'
    var_16 = '/path/to/src'
    var_17 = module_0.Config()
    var_18 = 'unknown_module'
    var_19 = module_1.module(var_18, var_17)
    var_20 = 'All tests passed!'
    var_21 = print(var_20)



# Parsed testcases at query #24
#--------------------------



def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = module_1.module(var_0, var_2)
    assert var_3 == 'test_module'
    var_4 = module_0.Config()
    var_5 = '.local_module'
    var_6 = module_1.module(var_5, var_4)
    assert var_6 == 'LOCALFOLDER'
    var_7 = 'test.*'
    var_8 = 'THIRDPARTY'
    var_9 = (var_7, var_8)
    var_10 = [var_9]
    var_11 = module_0.Config()
    var_12 = 'test.module'
    var_13 = module_1.module(var_12, var_11)
    assert var_13 == 'THIRDPARTY'
    var_14 = module_0.Config()
    var_15 = 'unknown_module'
    var_16 = module_1.module(var_15, var_14)
    assert var_16 == 'FIRSTPARTY'
    var_17 = 'All tests passed!'
    var_18 = print(var_17)



# Parsed testcases at query #25
#--------------------------



def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = module_1.module(var_0, var_2)
    assert var_3 == 'test_module'
    var_4 = 'test_module.submodule'
    var_5 = module_1.module(var_4, var_2)
    assert var_5 == 'test_module'
    var_6 = module_0.Config()
    var_7 = '.local_module'
    var_8 = module_1.module(var_7, var_6)
    assert var_8 == 'LOCALFOLDER'
    var_9 = '^django'
    var_10 = 'THIRDPARTY'
    var_11 = (var_9, var_10)
    var_12 = [var_11]
    var_13 = module_0.Config()
    var_14 = 'django.test'
    var_15 = module_1.module(var_14, var_13)
    assert var_15 == 'THIRDPARTY'
    var_16 = '/path/to/src'
    var_17 = module_0.Config()
    var_18 = 'unknown_module'
    var_19 = module_1.module(var_18, var_17)



# Parsed testcases at query #26
#--------------------------



def test_case_0():
    var_0 = 'test_pattern'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'test_pattern.module'
    var_4 = module_1.module(var_3, var_2)
    assert var_4 == 'test_pattern'
    var_5 = 'Test case 1 passed'
    var_6 = print(var_5)
    var_7 = module_0.Config()
    var_8 = '.local_module'
    var_9 = module_1.module(var_8, var_7)
    assert var_9 == 'LOCALFOLDER'
    var_10 = 'Test case 2 passed'
    var_11 = print(var_10)
    var_12 = '^django'
    var_13 = 'THIRDPARTY'
    var_14 = 'django.app'
    var_15 = module_1.module(var_14, var_7)
    assert var_15 == 'THIRDPARTY'
    var_16 = 'Test case 3 passed'
    var_17 = print(var_16)
    var_18 = '/nonexistent'
    var_19 = 'unknown_module'
    var_20 = module_1.module(var_19, var_7)
    var_21 = 'Test case 4 passed'
    var_22 = print(var_21)
    var_23 = module_0.Config()
    var_24 = 'some_random_module'
    var_25 = module_1.module(var_24, var_23)
    var_26 = 'Test case 5 passed'
    var_27 = print(var_26)
    var_28 = 'All tests passed!'
    var_29 = print(var_28)



# Parsed testcases at query #27
#--------------------------



def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = module_1.module(var_0, var_2)
    assert var_3 == 'test_module'
    var_4 = 'test_module.submodule'
    var_5 = module_1.module(var_4, var_2)
    assert var_5 == 'test_module'
    var_6 = module_0.Config()
    var_7 = '.local_module'
    var_8 = module_1.module(var_7, var_6)
    assert var_8 == 'LOCALFOLDER'
    var_9 = '^django'
    var_10 = 'THIRDPARTY'
    var_11 = (var_9, var_10)
    var_12 = [var_11]
    var_13 = module_0.Config()
    var_14 = 'django.contrib'
    var_15 = module_1.module(var_14, var_13)
    assert var_15 == 'THIRDPARTY'
    var_16 = '/path/to/src'
    var_17 = True
    var_18 = 'my_module'
    var_19 = module_1.module(var_18, var_13)
    assert var_19 == 'FIRSTPARTY'
    var_20 = module_0.Config()
    var_21 = 'unknown_module'
    var_22 = module_1.module(var_21, var_20)



# Parsed testcases at query #28
#--------------------------



def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = module_1.module(var_0, var_2)
    assert var_3 == 'test_module'
    var_4 = module_0.Config()
    var_5 = '.local_module'
    var_6 = module_1.module(var_5, var_4)
    assert var_6 == 'LOCALFOLDER'
    var_7 = '^django'
    var_8 = 'THIRDPARTY'
    var_9 = 'django.contrib'
    var_10 = module_1.module(var_9, var_4)
    assert var_10 == 'THIRDPARTY'
    var_11 = '/path/to/src'
    var_12 = 'my_module'
    var_13 = module_1.module(var_12, var_4)
    assert var_13 == 'FIRSTPARTY'
    var_14 = module_0.Config()
    var_15 = 'unknown_module'
    var_16 = module_1.module(var_15, var_14)
    var_17 = 'All tests passed!'
    var_18 = print(var_17)



# Parsed testcases at query #29
#--------------------------



def test_case_0():
    var_0 = 'test_pattern'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'test_pattern.module'
    var_4 = module_1.module(var_3, var_2)
    assert var_4 == 'test_pattern'
    var_5 = '.local_module'
    var_6 = module_1.module(var_5, var_2)
    assert var_6 == 'LOCALFOLDER'
    var_7 = '^known\\.pattern'
    var_8 = 'KNOWN'
    var_9 = 'known.pattern.module'
    var_10 = module_1.module(var_9, var_2)
    assert var_10 == 'KNOWN'
    var_11 = '/fake/path'
    var_12 = True
    var_13 = lambda x: var_12
    var_14 = 'fake_module'
    var_15 = module_1.module(var_14, var_2)
    assert var_15 == 'FIRSTPARTY'
    var_16 = 'unknown_module'
    var_17 = module_1.module(var_16, var_2)
    var_18 = 'All tests passed!'
    var_19 = print(var_18)



# Parsed testcases at query #30
#--------------------------



def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = module_1.module(var_0, var_2)
    assert var_3 == 'test_module'
    var_4 = 'test_module.submodule'
    var_5 = module_1.module(var_4, var_2)
    assert var_5 == 'test_module'
    var_6 = module_0.Config()
    var_7 = '.local_module'
    var_8 = module_1.module(var_7, var_6)
    assert var_8 == 'LOCALFOLDER'
    var_9 = '^django'
    var_10 = 'THIRDPARTY'
    var_11 = 'django.contrib'
    var_12 = module_1.module(var_11, var_6)
    assert var_12 == 'THIRDPARTY'
    var_13 = module_0.Config()
    var_14 = 'unknown_module'
    var_15 = module_1.module(var_14, var_13)
    var_16 = 'All tests passed!'
    var_17 = print(var_16)



