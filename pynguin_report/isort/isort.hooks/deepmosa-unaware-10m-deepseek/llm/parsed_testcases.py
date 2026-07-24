####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'echo'
    var_1 = 'test'
    var_2 = [var_0, var_1]
    var_3 = module_0.get_lines(var_2)
    var_4 = [var_0, var_1]
    var_5 = True
    var_6 = 'echo'
    var_7 = 'test'
    var_8 = [var_6, var_7]
    var_9 = module_0.get_lines(var_8)
    var_10 = 'echo'
    var_11 = 'test'
    var_12 = [var_10, var_11]
    var_13 = module_0.get_lines(var_12)
    var_14 = 'echo'
    var_15 = 'test'
    var_16 = [var_14, var_15]
    var_17 = module_0.get_lines(var_16)
    var_18 = 'echo'
    var_19 = 'test'
    var_20 = [var_18, var_19]
    var_21 = module_0.get_lines(var_20)



# Parsed testcases at query #2
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'git_hook_module'
    var_1 = __import__(var_0)
    var_2 = var_1.get_lines
    var_3 = __import__(var_0)
    var_4 = var_3.get_output
    var_5 = __import__(var_0)
    var_6 = var_5.api.check_code_string
    var_7 = __import__(var_0)
    var_8 = var_7.api.sort_file
    var_9 = 'file1.py'
    var_10 = 'file2.py'
    var_11 = 'file3.txt'
    var_12 = [var_9, var_10, var_11]
    var_13 = lambda cmd: var_12
    var_14 = 'import os\nimport sys'
    var_15 = lambda cmd: var_14
    var_16 = []
    var_17 = __import__(var_0)
    var_18 = True
    var_19 = False
    var_20 = module_0.git_hook(var_18, var_19)
    assert var_20 == 0
    var_21 = __import__(var_0)
    var_22 = module_0.git_hook(var_18, var_19)
    assert var_22 == 0
    var_23 = __import__(var_0)
    var_24 = module_0.git_hook(var_18, var_19)
    assert var_24 == 2
    var_25 = __import__(var_0)
    var_26 = module_0.git_hook(var_19, var_19)
    assert var_26 == 0
    var_27 = []
    var_28 = __import__(var_0)
    var_29 = module_0.git_hook(var_18, var_18)
    assert var_29 == 2
    var_30 = len(var_27)
    assert var_30 == 2
    var_31 = '--cached'
    var_32 = [var_9]
    var_33 = []
    var_34 = __import__(var_0)
    var_35 = module_0.git_hook(var_18, var_19, var_18)
    assert var_35 == 1
    var_36 = []
    var_37 = -1
    var_38 = 'dir'
    var_39 = -1
    var_40 = [var_9]
    var_41 = __import__(var_0)
    var_42 = 'dir1'
    var_43 = 'dir2'
    var_44 = [var_42, var_43]
    var_45 = module_0.git_hook(var_18, var_19, directories=var_44)
    var_46 = len(var_36)
    assert var_46 == 2
    var_47 = __import__(var_0)
    var_48 = module_0.git_hook(var_18, var_19)
    assert var_48 == 0



# Parsed testcases at query #3
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.git_hook(var_0, var_1)
    assert var_2 == 0
    var_3 = True
    var_4 = False
    var_5 = module_0.git_hook(var_3, var_4)
    assert var_5 == 0
    var_6 = True
    var_7 = False
    var_8 = module_0.git_hook(var_6, var_7)
    assert var_8 == 1
    var_9 = False
    var_10 = module_0.git_hook(var_9, var_9)
    assert var_10 == 0
    var_11 = True
    var_12 = module_0.git_hook(var_11, var_11)
    assert var_12 == 1
    var_13 = True
    var_14 = False
    var_15 = module_0.git_hook(var_13, var_14)
    assert var_15 == 0
    var_16 = 'File skipped'
    var_17 = True
    var_18 = False
    var_19 = module_0.git_hook(var_17, var_18)
    assert var_19 == 0
    var_20 = True
    var_21 = False
    var_22 = module_0.git_hook(var_20, var_21, var_20)
    assert var_22 == 0
    var_23 = True
    var_24 = False
    var_25 = 'src/'
    var_26 = [var_25]
    var_27 = module_0.git_hook(var_23, var_24, directories=var_26)
    assert var_27 == 0
    var_28 = True
    var_29 = False
    var_30 = module_0.git_hook(var_28, var_29)
    assert var_30 == 1



# Parsed testcases at query #4
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.git_hook(var_0, var_1)
    assert var_2 == 0
    var_3 = True
    var_4 = False
    var_5 = module_0.git_hook(var_3, var_4)
    assert var_5 == 0
    var_6 = True
    var_7 = False
    var_8 = module_0.git_hook(var_6, var_7)
    assert var_8 == 2
    var_9 = False
    var_10 = module_0.git_hook(var_9, var_9)
    assert var_10 == 0
    var_11 = True
    var_12 = module_0.git_hook(var_11, var_11)
    assert var_12 == 1
    var_13 = 'Skipped'
    var_14 = True
    var_15 = False
    var_16 = module_0.git_hook(var_14, var_15)
    assert var_16 == 0
    var_17 = True
    var_18 = False
    var_19 = module_0.git_hook(var_17, var_18, var_17)
    var_20 = True
    var_21 = False
    var_22 = 'src'
    var_23 = 'tests'
    var_24 = [var_22, var_23]
    var_25 = module_0.git_hook(var_20, var_21, directories=var_24)
    var_26 = False
    var_27 = True
    var_28 = module_0.git_hook(var_27, var_26)
    assert var_28 == 2



# Parsed testcases at query #5
#--------------------------


import isort.hooks as module_0
import posixpath as module_1

def test_case_0():
    var_0 = 'test1.py'
    var_1 = 'test2.py'
    var_2 = 'test3.txt'
    var_3 = 'test4.py'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = 'import os\nimport sys\n'
    var_6 = ''
    var_7 = True
    var_8 = False
    var_9 = module_0.git_hook(var_7, var_8)
    assert var_9 == 0
    var_10 = [var_0, var_1]
    var_11 = '\n'
    var_12 = module_1.join(var_10)
    var_13 = module_0.git_hook(var_7, var_8)
    assert var_13 == 2
    var_14 = module_0.git_hook(var_8, var_8)
    assert var_14 == 0
    var_15 = module_0.git_hook(var_7, var_8)
    assert var_15 == 0
    var_16 = []
    var_17 = module_0.git_hook(var_7, var_7)
    assert var_17 == 2
    var_18 = len(var_16)
    assert var_18 == 2
    var_19 = [var_0, var_1]
    var_20 = []
    var_21 = lambda cmd: diff_cmd_called.append(cmd) or var_19
    var_22 = module_0.git_hook(var_7, var_8, var_7)
    assert var_22 == 2
    var_23 = '--cached'
    var_24 = 'dir1/test1.py'
    var_25 = 'dir2/test2.py'
    var_26 = [var_24, var_25]
    var_27 = []
    var_28 = lambda cmd: diff_cmd_called.append(cmd) or var_26
    var_29 = 'src/'
    var_30 = [var_29]
    var_31 = module_0.git_hook(var_7, var_8, directories=var_30)
    assert var_31 == 2
    var_32 = module_0.git_hook(var_7, var_8)
    assert var_32 == 0
    var_33 = 'test1.txt'
    var_34 = 'test2.md'
    var_35 = 'test3.yaml'
    var_36 = [var_33, var_34, var_35]
    var_37 = module_1.join(var_36)
    var_38 = module_0.git_hook(var_7, var_8)
    assert var_38 == 0
    var_39 = 'test2.txt'
    var_40 = 'test3.py'
    var_41 = 'test4.md'
    var_42 = [var_0, var_39, var_40, var_41]
    var_43 = module_0.git_hook(var_7, var_8)
    assert var_43 == 2



# Parsed testcases at query #6
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'git_hook_module'
    var_1 = __import__(var_0)
    var_2 = var_1.get_lines
    var_3 = __import__(var_0)
    var_4 = var_3.get_output
    var_5 = __import__(var_0)
    var_6 = var_5.api.check_code_string
    var_7 = __import__(var_0)
    var_8 = var_7.api.sort_file
    var_9 = __import__(var_0)
    var_10 = var_9.Config
    var_11 = True
    var_12 = False
    var_13 = module_0.git_hook(var_11, var_12)
    assert var_13 == 0
    var_14 = 'file1.py'
    var_15 = 'file2.txt'
    var_16 = module_0.git_hook(var_11, var_12)
    assert var_16 == 1
    var_17 = module_0.git_hook(var_11, var_12)
    assert var_17 == 0
    var_18 = module_0.git_hook(var_12, var_12)
    assert var_18 == 0
    var_19 = module_0.git_hook(var_12, var_11)
    assert var_19 == 0
    var_20 = module_0.git_hook(var_11, var_12, var_11)
    assert var_20 == 1
    var_21 = '.isort.cfg'
    var_22 = module_0.git_hook(var_11, var_12, settings_file=var_21)
    assert var_22 == 0
    var_23 = 'src'
    var_24 = 'tests'
    var_25 = [var_23, var_24]
    var_26 = module_0.git_hook(var_11, var_12, directories=var_25)
    assert var_26 == 0
    var_27 = 'file2.py'
    var_28 = 'file3.txt'
    var_29 = module_0.git_hook(var_11, var_12)
    assert var_29 == 2
    var_30 = module_0.git_hook(var_11, var_12)
    assert var_30 == 0



# Parsed testcases at query #7
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.git_hook(var_0, var_1)
    assert var_2 == 0
    var_3 = True
    var_4 = False
    var_5 = module_0.git_hook(var_3, var_4)
    assert var_5 == 2
    var_6 = True
    var_7 = False
    var_8 = module_0.git_hook(var_6, var_7)
    assert var_8 == 0
    var_9 = False
    var_10 = module_0.git_hook(var_9, var_9)
    assert var_10 == 0
    var_11 = True
    var_12 = module_0.git_hook(var_11, var_11)
    assert var_12 == 1
    var_13 = True
    var_14 = False
    var_15 = module_0.git_hook(var_13, var_14)
    assert var_15 == 0
    var_16 = True
    var_17 = False
    var_18 = module_0.git_hook(var_16, var_17)
    assert var_18 == 0
    var_19 = True
    var_20 = module_0.git_hook(lazy=var_19)
    var_21 = 0
    var_22 = 'src'
    var_23 = 'tests'
    var_24 = [var_22, var_23]
    var_25 = module_0.git_hook(directories=var_24)
    var_26 = 0



# Parsed testcases at query #8
#--------------------------




# Parsed testcases at query #9
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.git_hook(var_0, var_1)
    assert var_2 == 0
    var_3 = 'git'
    var_4 = 'diff-index'
    var_5 = '--cached'
    var_6 = '--name-only'
    var_7 = '--diff-filter=ACMRTUXB'
    var_8 = 'HEAD'
    var_9 = [var_3, var_4, var_5, var_6, var_7, var_8]
    var_10 = False
    var_11 = True
    var_12 = module_0.git_hook(var_11, var_10)
    assert var_12 == 1
    var_13 = 'content1'
    var_14 = 'file1.py'
    var_15 = 'content2'
    var_16 = 'file2.py'
    var_17 = False
    var_18 = module_0.git_hook(var_17, var_17)
    assert var_18 == 0
    var_19 = True
    var_20 = module_0.git_hook(var_19, var_19)
    assert var_20 == 1
    var_21 = 'file1.py'
    var_22 = True
    var_23 = False
    var_24 = module_0.git_hook(var_22, var_23, var_22)
    var_25 = 'git'
    var_26 = 'diff-index'
    var_27 = '--name-only'
    var_28 = '--diff-filter=ACMRTUXB'
    var_29 = 'HEAD'
    var_30 = [var_25, var_26, var_27, var_28, var_29]
    var_31 = True
    var_32 = False
    var_33 = 'src'
    var_34 = 'tests'
    var_35 = [var_33, var_34]
    var_36 = module_0.git_hook(var_31, var_32, directories=var_35)
    var_37 = 'git'
    var_38 = 'diff-index'
    var_39 = '--cached'
    var_40 = '--name-only'
    var_41 = '--diff-filter=ACMRTUXB'
    var_42 = 'HEAD'
    var_43 = [var_37, var_38, var_39, var_40, var_41, var_42, var_33, var_34]
    var_44 = 'message'
    var_45 = True
    var_46 = False
    var_47 = module_0.git_hook(var_45, var_46)
    assert var_47 == 0
    var_48 = True
    var_49 = False
    var_50 = module_0.git_hook(var_48, var_49)
    assert var_50 == 0



# Parsed testcases at query #10
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.git_hook(var_0, var_1)
    assert var_2 == 0
    var_3 = True
    var_4 = False
    var_5 = module_0.git_hook(var_3, var_4)
    assert var_5 == 2
    var_6 = False
    var_7 = module_0.git_hook(var_6, var_6)
    assert var_7 == 0
    var_8 = True
    var_9 = module_0.git_hook(var_8, var_8)
    assert var_9 == 1
    var_10 = True
    var_11 = False
    var_12 = module_0.git_hook(var_10, var_11, var_10)
    assert var_12 == 0
    var_13 = True
    var_14 = False
    var_15 = 'src'
    var_16 = [var_15]
    var_17 = module_0.git_hook(var_13, var_14, directories=var_16)
    assert var_17 == 1
    var_18 = True
    var_19 = False
    var_20 = module_0.git_hook(var_18, var_19)
    assert var_20 == 0
    var_21 = 'test'
    var_22 = True
    var_23 = False
    var_24 = module_0.git_hook(var_22, var_23)
    assert var_24 == 0



# Parsed testcases at query #11
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'git_hook_module'
    var_1 = __import__(var_0)
    var_2 = var_1.get_lines
    var_3 = __import__(var_0)
    var_4 = var_3.get_output
    var_5 = __import__(var_0)
    var_6 = var_5.api.check_code_string
    var_7 = __import__(var_0)
    var_8 = var_7.api.sort_file
    var_9 = True
    var_10 = False
    var_11 = module_0.git_hook(var_9, var_10)
    assert var_11 == 0
    var_12 = 'file.txt'
    var_13 = 'README.md'
    var_14 = module_0.git_hook(var_9, var_10)
    assert var_14 == 0
    var_15 = 'test.py'
    var_16 = module_0.git_hook(var_9, var_10)
    assert var_16 == 0
    var_17 = module_0.git_hook(var_9, var_10)
    assert var_17 == 1
    var_18 = module_0.git_hook(var_10, var_10)
    assert var_18 == 0
    var_19 = module_0.git_hook(var_9, var_9)
    assert var_19 == 1
    var_20 = 'test1.py'
    var_21 = 'test2.py'
    var_22 = 'test3.py'
    var_23 = 'import os\nimport sys'
    var_24 = 'import sys\nimport os'
    var_25 = 'import json\nimport os'
    var_26 = module_0.git_hook(var_9, var_10)
    assert var_26 == 2
    var_27 = 'skipped'
    var_28 = module_0.git_hook(var_9, var_10)
    assert var_28 == 0
    var_29 = module_0.git_hook(var_9, var_10, var_9)
    assert var_29 == 1
    var_30 = 'dir/test.py'
    var_31 = 'dir/'
    var_32 = [var_31]
    var_33 = module_0.git_hook(var_9, var_10, directories=var_32)
    assert var_33 == 1
    var_34 = '.isort.cfg'
    var_35 = module_0.git_hook(var_9, var_10, settings_file=var_34)
    assert var_35 == 1



# Parsed testcases at query #12
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'git_hook_module'
    var_1 = __import__(var_0)
    var_2 = var_1.get_lines
    var_3 = __import__(var_0)
    var_4 = var_3.get_output
    var_5 = __import__(var_0)
    var_6 = var_5.api.check_code_string
    var_7 = __import__(var_0)
    var_8 = var_7.api.sort_file
    var_9 = __import__(var_0)
    var_10 = var_9.Config
    var_11 = 'file1.py'
    var_12 = 'file2.py'
    var_13 = 'file3.txt'
    var_14 = [var_11, var_12, var_13]
    var_15 = 'import os\nimport sys'
    var_16 = 'git_hook_module'
    var_17 = __import__(var_16)
    var_18 = False
    var_19 = module_0.git_hook(var_18, var_18)
    assert var_19 == 0
    var_20 = __import__(var_16)
    var_21 = True
    var_22 = module_0.git_hook(var_21, var_18)
    assert var_22 == 1
    var_23 = 'dir1/file1.py'
    var_24 = 'dir2/file2.py'
    var_25 = [var_23, var_24]
    var_26 = __import__(var_16)
    var_27 = 'dir1'
    var_28 = [var_27]
    var_29 = module_0.git_hook(var_21, var_18, var_21, directories=var_28)
    var_30 = []
    var_31 = __import__(var_16)
    var_32 = module_0.git_hook(var_21, var_18)
    assert var_32 == 0
    var_33 = 'test.py'
    var_34 = [var_33]
    var_35 = __import__(var_16)
    var_36 = module_0.git_hook(var_21, var_21)
    assert var_36 == 0



# Parsed testcases at query #13
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.git_hook(var_0, var_1)
    assert var_2 == 0
    var_3 = True
    var_4 = False
    var_5 = module_0.git_hook(var_3, var_4)
    assert var_5 == 2
    var_6 = False
    var_7 = module_0.git_hook(var_6, var_6)
    assert var_7 == 0
    var_8 = True
    var_9 = False
    var_10 = module_0.git_hook(var_8, var_9)
    assert var_10 == 0
    var_11 = True
    var_12 = module_0.git_hook(var_11, var_11)
    assert var_12 == 1
    var_13 = True
    var_14 = module_0.git_hook(lazy=var_13)
    assert var_14 == 0
    var_15 = True
    var_16 = False
    var_17 = module_0.git_hook(var_15, var_16)
    assert var_17 == 0
    var_18 = True
    var_19 = False
    var_20 = module_0.git_hook(var_18, var_19)
    assert var_20 == 0
    var_21 = 'src/'
    var_22 = 'tests/'
    var_23 = [var_21, var_22]
    var_24 = module_0.git_hook(directories=var_23)
    assert var_24 == 0
    var_25 = False
    var_26 = True
    var_27 = module_0.git_hook(var_26, var_25)
    assert var_27 == 2



# Parsed testcases at query #14
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.git_hook(var_0, var_1)
    assert var_2 == 0
    var_3 = True
    var_4 = False
    var_5 = module_0.git_hook(var_3, var_4)
    assert var_5 == 0
    var_6 = True
    var_7 = False
    var_8 = module_0.git_hook(var_6, var_7)
    assert var_8 == 1
    var_9 = False
    var_10 = module_0.git_hook(var_9, var_9)
    assert var_10 == 0
    var_11 = True
    var_12 = module_0.git_hook(var_11, var_11)
    assert var_12 == 1
    var_13 = False
    var_14 = True
    var_15 = module_0.git_hook(var_13, var_13, var_14)
    var_16 = False
    var_17 = 'src'
    var_18 = 'tests'
    var_19 = [var_17, var_18]
    var_20 = module_0.git_hook(var_16, var_16, directories=var_19)
    var_21 = True
    var_22 = False
    var_23 = module_0.git_hook(var_21, var_22)
    assert var_23 == 0
    var_24 = 'test.py'
    var_25 = True
    var_26 = False
    var_27 = module_0.git_hook(var_25, var_26)
    assert var_27 == 0
    var_28 = b'import os\nimport sys\n'
    var_29 = b'import sys\nimport os\n'
    var_30 = [var_25, var_19, var_20]
    var_31 = True
    var_32 = False
    var_33 = module_0.git_hook(var_31, var_32)
    assert var_33 == 1



# Parsed testcases at query #15
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.git_hook(var_0, var_1)
    assert var_2 == 0
    var_3 = True
    var_4 = False
    var_5 = module_0.git_hook(var_3, var_4)
    assert var_5 == 2
    var_6 = False
    var_7 = module_0.git_hook(var_6, var_6)
    assert var_7 == 0
    var_8 = True
    var_9 = False
    var_10 = module_0.git_hook(var_8, var_9)
    assert var_10 == 0
    var_11 = True
    var_12 = module_0.git_hook(var_11, var_11)
    assert var_12 == 2
    var_13 = True
    var_14 = module_0.git_hook(lazy=var_13)
    var_15 = 0
    var_16 = 'src'
    var_17 = 'tests'
    var_18 = [var_16, var_17]
    var_19 = module_0.git_hook(directories=var_18)
    var_20 = 0
    var_21 = True
    var_22 = False
    var_23 = module_0.git_hook(var_21, var_22)
    assert var_23 == 0
    var_24 = True
    var_25 = False
    var_26 = module_0.git_hook(var_24, var_25)
    var_27 = '.isort.cfg'
    var_28 = module_0.git_hook(settings_file=var_27)



# Parsed testcases at query #16
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.git_hook(var_0, var_1)
    assert var_2 == 0
    var_3 = True
    var_4 = False
    var_5 = module_0.git_hook(var_3, var_4)
    assert var_5 == 1
    var_6 = True
    var_7 = False
    var_8 = module_0.git_hook(var_6, var_7)
    assert var_8 == 0
    var_9 = False
    var_10 = module_0.git_hook(var_9, var_9)
    assert var_10 == 0
    var_11 = True
    var_12 = module_0.git_hook(var_11, var_11)
    assert var_12 == 1
    var_13 = True
    var_14 = False
    var_15 = module_0.git_hook(modify=var_14, lazy=var_13)
    var_16 = True
    var_17 = False
    var_18 = module_0.git_hook(var_16, var_17)
    assert var_18 == 0
    var_19 = 'File skipped'
    var_20 = True
    var_21 = False
    var_22 = module_0.git_hook(var_20, var_21)
    assert var_22 == 0
    var_23 = 'src'
    var_24 = 'tests'
    var_25 = [var_23, var_24]
    var_26 = module_0.git_hook(directories=var_25)
    var_27 = []
    var_28 = False
    var_29 = True
    var_30 = module_0.git_hook(var_29, var_28)
    assert var_30 == 2



# Parsed testcases at query #17
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.git_hook(var_0, var_1)
    assert var_2 == 0
    var_3 = True
    var_4 = False
    var_5 = module_0.git_hook(var_3, var_4)
    assert var_5 == 2
    var_6 = False
    var_7 = module_0.git_hook(var_6, var_6)
    assert var_7 == 0
    var_8 = True
    var_9 = module_0.git_hook(var_8, var_8)
    assert var_9 == 1
    var_10 = 'Skipped'
    var_11 = True
    var_12 = False
    var_13 = module_0.git_hook(var_11, var_12)
    assert var_13 == 0
    var_14 = True
    var_15 = False
    var_16 = module_0.git_hook(var_14, var_15)
    assert var_16 == 0
    var_17 = True
    var_18 = False
    var_19 = module_0.git_hook(modify=var_18, lazy=var_17)
    var_20 = 'src'
    var_21 = 'tests'
    var_22 = [var_20, var_21]
    var_23 = False
    var_24 = module_0.git_hook(modify=var_23, directories=var_22)
    var_25 = True
    var_26 = False
    var_27 = module_0.git_hook(var_25, var_26)
    assert var_27 == 0



# Parsed testcases at query #18
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'git_hook_module'
    var_1 = __import__(var_0)
    var_2 = var_1.get_lines
    var_3 = __import__(var_0)
    var_4 = var_3.get_output
    var_5 = __import__(var_0)
    var_6 = var_5.api.check_code_string
    var_7 = __import__(var_0)
    var_8 = var_7.api.sort_file
    var_9 = 'file1.py'
    var_10 = 'file2.py'
    var_11 = 'file3.txt'
    var_12 = [var_9, var_10, var_11]
    var_13 = lambda cmd: var_12
    var_14 = 'import os\nimport sys'
    var_15 = lambda cmd: var_14
    var_16 = []
    var_17 = __import__(var_0)
    var_18 = module_0.git_hook()
    assert var_18 == 0
    var_19 = False
    var_20 = None
    var_21 = __import__(var_0)
    var_22 = module_0.git_hook(var_19)
    assert var_22 == 0
    var_23 = __import__(var_0)
    var_24 = True
    var_25 = module_0.git_hook(var_24)
    assert var_25 == 2
    var_26 = __import__(var_0)
    var_27 = module_0.git_hook(var_24, var_24)
    assert var_27 == 2
    var_28 = __import__(var_0)
    var_29 = module_0.git_hook(var_24)
    assert var_29 == 0
    var_30 = '--cached'
    var_31 = [var_9]
    var_32 = []
    var_33 = __import__(var_0)
    var_34 = module_0.git_hook(var_24, lazy=var_24)
    assert var_34 == 1
    var_35 = 'dir1'
    var_36 = 'dir1/file1.py'
    var_37 = [var_36]
    var_38 = []
    var_39 = __import__(var_0)
    var_40 = [var_35]
    var_41 = module_0.git_hook(var_24, directories=var_40)
    assert var_41 == 1
    var_42 = __import__(var_0)
    var_43 = module_0.git_hook(var_24)
    assert var_43 == 0



# Parsed testcases at query #19
#--------------------------


import isort.hooks as module_0
import posixpath as module_1

def test_case_0():
    var_0 = 'file1.py'
    var_1 = 'file2.py'
    var_2 = 'file3.txt'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'import os\nimport sys'
    var_5 = ''
    var_6 = module_0.git_hook()
    assert var_6 == 0
    var_7 = [var_0, var_1]
    var_8 = '\n'
    var_9 = module_1.join(var_7)
    var_10 = False
    var_11 = '/test'
    var_12 = '/test/file1.py'
    var_13 = True
    var_14 = module_0.git_hook(var_13)
    assert var_14 == 2
    var_15 = module_0.git_hook(var_13)
    assert var_15 == 0
    var_16 = module_0.git_hook(var_10)
    assert var_16 == 0
    var_17 = []
    var_18 = module_0.git_hook(modify=var_13)
    assert var_18 == 0
    var_19 = len(var_17)
    assert var_19 == 2
    var_20 = [var_0]
    var_21 = module_1.join(var_20)
    var_22 = module_0.git_hook(var_13, lazy=var_13)
    assert var_22 == 1
    var_23 = 'dir1/file1.py'
    var_24 = 'dir2/file2.py'
    var_25 = [var_23, var_24]
    var_26 = module_1.join(var_25)
    var_27 = 'dir1'
    var_28 = [var_27]
    var_29 = module_0.git_hook(var_13, directories=var_28)
    assert var_29 == 2
    var_30 = module_0.git_hook(var_13)
    assert var_30 == 0
    var_31 = 'file1.txt'
    var_32 = 'file2.md'
    var_33 = [var_31, var_32]
    var_34 = module_1.join(var_33)
    var_35 = module_0.git_hook(var_13)
    assert var_35 == 0



# Parsed testcases at query #20
#--------------------------


import isort.hooks as module_0
import posixpath as module_1

def test_case_0():
    var_0 = 'git_hook_module'
    var_1 = __import__(var_0)
    var_2 = var_1.get_lines
    var_3 = __import__(var_0)
    var_4 = var_3.get_output
    var_5 = __import__(var_0)
    var_6 = var_5.api.check_code_string
    var_7 = __import__(var_0)
    var_8 = var_7.api.sort_file
    var_9 = __import__(var_0)
    var_10 = var_9.Config
    var_11 = module_0.git_hook()
    assert var_11 == 0
    var_12 = 'git'
    var_13 = 'diff-index'
    var_14 = '--cached'
    var_15 = '--name-only'
    var_16 = '--diff-filter=ACMRTUXB'
    var_17 = 'HEAD'
    var_18 = [var_12, var_13, var_14, var_15, var_16, var_17]
    var_19 = 'file1.py'
    var_20 = 'file2.py'
    var_21 = False
    var_22 = module_0.git_hook(var_21)
    assert var_22 == 0
    var_23 = True
    var_24 = module_0.git_hook(var_23, var_21)
    assert var_24 == 1
    var_25 = module_0.git_hook(var_23, var_23)
    assert var_25 == 2
    var_26 = 'file1.txt'
    var_27 = 'file2.md'
    var_28 = module_0.git_hook()
    assert var_28 == 0
    var_29 = module_0.git_hook(lazy=var_23)
    assert var_29 == 0
    var_30 = [var_12, var_13, var_15, var_16, var_17]
    var_31 = 'dir1'
    var_32 = 'dir2'
    var_33 = [var_31, var_32]
    var_34 = module_0.git_hook(directories=var_33)
    assert var_34 == 0
    var_35 = [var_12, var_13, var_14, var_15, var_16, var_17, var_31, var_32]
    var_36 = 'skipped'
    var_37 = module_0.git_hook()
    assert var_37 == 0
    var_38 = '.isort.cfg'
    var_39 = module_0.git_hook(settings_file=var_38)
    assert var_39 == 0
    var_40 = 'git_hook_module'
    var_41 = __import__(var_40)
    var_42 = module_1.abspath(var_19)
    var_43 = module_1.dirname(var_42)



# Parsed testcases at query #21
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'git_hook_module'
    var_1 = __import__(var_0)
    var_2 = var_1.get_lines
    var_3 = __import__(var_0)
    var_4 = var_3.get_output
    var_5 = __import__(var_0)
    var_6 = var_5.api.check_code_string
    var_7 = __import__(var_0)
    var_8 = var_7.api.sort_file
    var_9 = 'file1.py'
    var_10 = 'file2.py'
    var_11 = 'file3.txt'
    var_12 = [var_9, var_10, var_11]
    var_13 = lambda cmd: var_12
    var_14 = 'import os\nimport sys'
    var_15 = lambda cmd: var_14
    var_16 = True
    var_17 = lambda code, file_path, config: var_16
    var_18 = None
    var_19 = lambda filename, config: var_18
    var_20 = 'git_hook_module'
    var_21 = __import__(var_20)
    var_22 = False
    var_23 = module_0.git_hook(var_22, var_22)
    assert var_23 == 0
    var_24 = __import__(var_20)
    var_25 = True
    var_26 = module_0.git_hook(var_25, var_22)
    assert var_26 == 0
    var_27 = lambda code, file_path, config: var_22
    var_28 = __import__(var_20)
    var_29 = module_0.git_hook(var_22, var_22)
    assert var_29 == 0
    var_30 = __import__(var_20)
    var_31 = module_0.git_hook(var_25, var_22)
    assert var_31 == 2
    var_32 = []
    var_33 = lambda filename, config: sort_calls.append(filename)
    var_34 = __import__(var_20)
    var_35 = module_0.git_hook(var_25, var_25)
    assert var_35 == 2
    var_36 = []
    var_37 = lambda cmd: var_36
    var_38 = __import__(var_20)
    var_39 = module_0.git_hook(var_25, var_25)
    assert var_39 == 0
    var_40 = '--cached'
    var_41 = 'file1.py'
    var_42 = [var_41]
    var_43 = 'file4.py'
    var_44 = [var_41, var_43]
    var_45 = lambda cmd: var_42 if var_40 in cmd else var_44
    var_46 = __import__(var_20)
    var_47 = module_0.git_hook(var_25, var_22, var_25)
    assert var_47 == 2
    var_48 = 'dir1'
    var_49 = 'dir1/file1.py'
    var_50 = [var_49]
    var_51 = []
    var_52 = lambda cmd: var_50 if var_48 in cmd else var_51
    var_53 = __import__(var_20)
    var_54 = [var_48]
    var_55 = module_0.git_hook(var_25, var_22, directories=var_54)
    assert var_55 == 1
    var_56 = [var_41]
    var_57 = lambda cmd: var_56
    var_58 = __import__(var_20)
    var_59 = module_0.git_hook(var_25, var_22)
    assert var_59 == 0



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.git_hook(var_0, var_1)
    assert var_2 == 0
    var_3 = True
    var_4 = False
    var_5 = module_0.git_hook(var_3, var_4)
    assert var_5 == 0
    var_6 = True
    var_7 = False
    var_8 = module_0.git_hook(var_6, var_7)
    assert var_8 == 0
    var_9 = False
    var_10 = True
    var_11 = module_0.git_hook(var_10, var_9)
    assert var_11 == 2
    var_12 = False
    var_13 = module_0.git_hook(var_12, var_12)
    assert var_13 == 0
    var_14 = True
    var_15 = module_0.git_hook(var_14, var_14)
    assert var_15 == 1
    var_16 = 'test'
    var_17 = True
    var_18 = False
    var_19 = module_0.git_hook(var_17, var_18)
    assert var_19 == 0
    var_20 = True
    var_21 = False
    var_22 = module_0.git_hook(var_20, var_21, var_20)
    assert var_22 == 0
    var_23 = '--cached'
    var_24 = True
    var_25 = False
    var_26 = 'src'
    var_27 = 'tests'
    var_28 = [var_26, var_27]
    var_29 = module_0.git_hook(var_24, var_25, directories=var_28)
    assert var_29 == 0



# Parsed testcases at query #2
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.git_hook(var_0, var_1)
    assert var_2 == 0
    var_3 = True
    var_4 = False
    var_5 = module_0.git_hook(var_3, var_4)
    assert var_5 == 2
    var_6 = True
    var_7 = False
    var_8 = module_0.git_hook(var_6, var_7)
    assert var_8 == 0
    var_9 = False
    var_10 = module_0.git_hook(var_9, var_9)
    assert var_10 == 0
    var_11 = True
    var_12 = module_0.git_hook(var_11, var_11)
    assert var_12 == 1
    var_13 = True
    var_14 = module_0.git_hook(lazy=var_13)
    var_15 = 0
    var_16 = 'src'
    var_17 = 'tests'
    var_18 = [var_16, var_17]
    var_19 = module_0.git_hook(directories=var_18)
    var_20 = 0
    var_21 = True
    var_22 = False
    var_23 = module_0.git_hook(var_21, var_22)
    assert var_23 == 2
    var_24 = 'test'
    var_25 = True
    var_26 = False
    var_27 = module_0.git_hook(var_25, var_26)
    assert var_27 == 0
    var_28 = '.isort.cfg'
    var_29 = module_0.git_hook(settings_file=var_28)
    assert var_29 == 0



# Parsed testcases at query #3
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'echo'
    var_1 = '-e'
    var_2 = 'line1\nline2\nline3'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.get_lines(var_3)
    var_5 = '  line1  \n\tline2\t\n  line3  '
    var_6 = [var_0, var_1, var_5]
    var_7 = module_0.get_lines(var_6)
    var_8 = '-n'
    var_9 = [var_0, var_8]
    var_10 = module_0.get_lines(var_9)
    var_11 = 'single line'
    var_12 = [var_0, var_11]
    var_13 = module_0.get_lines(var_12)
    var_14 = '\n\n\n'
    var_15 = [var_0, var_1, var_14]
    var_16 = module_0.get_lines(var_15)



# Parsed testcases at query #4
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.git_hook(var_0, var_1)
    assert var_2 == 0
    var_3 = True
    var_4 = False
    var_5 = module_0.git_hook(var_3, var_4)
    assert var_5 == 2
    var_6 = False
    var_7 = module_0.git_hook(var_6, var_6)
    assert var_7 == 0
    var_8 = True
    var_9 = module_0.git_hook(var_8, var_8)
    assert var_9 == 1
    var_10 = 'message'
    var_11 = True
    var_12 = False
    var_13 = module_0.git_hook(var_11, var_12)
    assert var_13 == 0
    var_14 = True
    var_15 = False
    var_16 = module_0.git_hook(var_14, var_15)
    assert var_16 == 0
    var_17 = True
    var_18 = False
    var_19 = module_0.git_hook(var_17, var_18, var_17)
    assert var_19 == 0
    var_20 = True
    var_21 = False
    var_22 = 'src/'
    var_23 = [var_22]
    var_24 = module_0.git_hook(var_20, var_21, directories=var_23)
    assert var_24 == 1
    var_25 = True
    var_26 = False
    var_27 = module_0.git_hook(var_25, var_26)
    assert var_27 == 0
    var_28 = True
    var_29 = False
    var_30 = '.isort.cfg'
    var_31 = module_0.git_hook(var_28, var_29, settings_file=var_30)
    assert var_31 == 1



# Parsed testcases at query #5
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.git_hook(var_0, var_1)
    assert var_2 == 0
    var_3 = True
    var_4 = False
    var_5 = module_0.git_hook(var_3, var_4)
    assert var_5 == 0
    var_6 = True
    var_7 = False
    var_8 = module_0.git_hook(var_6, var_7)
    assert var_8 == 2
    var_9 = False
    var_10 = module_0.git_hook(var_9, var_9)
    assert var_10 == 0
    var_11 = True
    var_12 = module_0.git_hook(var_11, var_11)
    assert var_12 == 2
    var_13 = 'test'
    var_14 = True
    var_15 = False
    var_16 = module_0.git_hook(var_14, var_15)
    assert var_16 == 0
    var_17 = True
    var_18 = False
    var_19 = module_0.git_hook(var_17, var_18)
    assert var_19 == 0
    var_20 = True
    var_21 = False
    var_22 = module_0.git_hook(var_20, var_21, var_20)
    assert var_22 == 0
    var_23 = True
    var_24 = False
    var_25 = 'src/'
    var_26 = 'tests/'
    var_27 = [var_25, var_26]
    var_28 = module_0.git_hook(var_23, var_24, directories=var_27)
    assert var_28 == 0
    var_29 = True
    var_30 = False
    var_31 = module_0.git_hook(var_29, var_30)
    assert var_31 == 2
    var_32 = b'test1.py\n'
    var_33 = b'import os\nimport sys\n'
    var_34 = [var_30, var_26]
    var_35 = True
    var_36 = False
    var_37 = module_0.git_hook(var_35, var_36)
    assert var_37 == 0



# Parsed testcases at query #6
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.git_hook(var_0, var_1)
    assert var_2 == 0
    var_3 = True
    var_4 = False
    var_5 = module_0.git_hook(var_3, var_4)
    assert var_5 == 0
    var_6 = True
    var_7 = False
    var_8 = module_0.git_hook(var_6, var_7)
    assert var_8 == 0
    var_9 = True
    var_10 = False
    var_11 = module_0.git_hook(var_9, var_10)
    assert var_11 == 2
    var_12 = False
    var_13 = module_0.git_hook(var_12, var_12)
    assert var_13 == 0
    var_14 = True
    var_15 = module_0.git_hook(var_14, var_14)
    assert var_15 == 1
    var_16 = True
    var_17 = module_0.git_hook(lazy=var_16)
    var_18 = 0
    var_19 = 'src'
    var_20 = 'tests'
    var_21 = [var_19, var_20]
    var_22 = module_0.git_hook(directories=var_21)
    var_23 = 0
    var_24 = 'File skipped'
    var_25 = True
    var_26 = False
    var_27 = module_0.git_hook(var_25, var_26)
    assert var_27 == 0
    var_28 = '.isort.cfg'
    var_29 = module_0.git_hook(settings_file=var_28)
    assert var_29 == 0



# Parsed testcases at query #7
#--------------------------


import isort.hooks as module_0
import posixpath as module_1

def test_case_0():
    var_0 = 'test1.py'
    var_1 = 'test2.py'
    var_2 = 'test3.txt'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'import os\nimport sys'
    var_5 = ''
    var_6 = module_0.git_hook()
    assert var_6 == 0
    var_7 = '\n'
    var_8 = module_1.join(var_3)
    var_9 = False
    var_10 = module_0.git_hook(var_9)
    assert var_10 == 0
    var_11 = True
    var_12 = module_0.git_hook(var_11)
    assert var_12 == 2
    var_13 = module_0.git_hook(var_11, var_11)
    assert var_13 == 1
    var_14 = module_1.join(var_3)
    var_15 = module_0.git_hook(var_11, lazy=var_11)
    assert var_15 == 2
    var_16 = module_0.git_hook(var_11)
    assert var_16 == 0
    var_17 = module_0.git_hook()
    assert var_17 == 0
    var_18 = 'test.txt\ntest.md'
    var_19 = module_0.git_hook(var_11)
    assert var_19 == 0



# Parsed testcases at query #8
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'git_hook_module'
    var_1 = __import__(var_0)
    var_2 = var_1.get_lines
    var_3 = __import__(var_0)
    var_4 = var_3.get_output
    var_5 = __import__(var_0)
    var_6 = var_5.api.check_code_string
    var_7 = __import__(var_0)
    var_8 = var_7.api.sort_file
    var_9 = __import__(var_0)
    var_10 = var_9.Config
    var_11 = module_0.git_hook()
    assert var_11 == 0
    var_12 = 'git'
    var_13 = 'diff-index'
    var_14 = '--cached'
    var_15 = '--name-only'
    var_16 = '--diff-filter=ACMRTUXB'
    var_17 = 'HEAD'
    var_18 = [var_12, var_13, var_14, var_15, var_16, var_17]
    var_19 = 'file1.py'
    var_20 = 'file2.py'
    var_21 = False
    var_22 = module_0.git_hook(var_21, var_21)
    assert var_22 == 0
    var_23 = True
    var_24 = module_0.git_hook(var_23, var_21)
    assert var_24 == 1
    var_25 = module_0.git_hook(var_23, var_23)
    assert var_25 == 2
    var_26 = 'file2.txt'
    var_27 = 'file3.py'
    var_28 = module_0.git_hook(var_23, var_21)
    assert var_28 == 1
    var_29 = module_0.git_hook(lazy=var_23)
    var_30 = [var_12, var_13, var_15, var_16, var_17]
    var_31 = 'src'
    var_32 = 'tests'
    var_33 = [var_31, var_32]
    var_34 = module_0.git_hook(directories=var_33)
    var_35 = [var_12, var_13, var_14, var_15, var_16, var_17, var_31, var_32]
    var_36 = 'git_hook_module'
    var_37 = __import__(var_36)
    var_38 = module_0.git_hook(var_23, var_23)
    assert var_38 == 0
    var_39 = '.isort.cfg'
    var_40 = module_0.git_hook(settings_file=var_39)
    var_41 = __import__(var_36)



# Parsed testcases at query #9
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.git_hook(var_0, var_1)
    assert var_2 == 0
    var_3 = True
    var_4 = False
    var_5 = module_0.git_hook(var_3, var_4)
    assert var_5 == 0
    var_6 = False
    var_7 = True
    var_8 = module_0.git_hook(var_7, var_6)
    assert var_8 == 1
    var_9 = False
    var_10 = module_0.git_hook(var_9, var_9)
    assert var_10 == 0
    var_11 = False
    var_12 = True
    var_13 = module_0.git_hook(var_12, var_12)
    assert var_13 == 1
    var_14 = True
    var_15 = False
    var_16 = module_0.git_hook(modify=var_15, lazy=var_14)
    var_17 = 'src'
    var_18 = 'tests'
    var_19 = [var_17, var_18]
    var_20 = False
    var_21 = module_0.git_hook(modify=var_20, directories=var_19)
    var_22 = True
    var_23 = False
    var_24 = module_0.git_hook(var_22, var_23)
    var_25 = 'test'
    var_26 = True
    var_27 = False
    var_28 = module_0.git_hook(var_26, var_27)
    assert var_28 == 0
    var_29 = '.isort.cfg'
    var_30 = False
    var_31 = module_0.git_hook(modify=var_30, settings_file=var_29)
    var_32 = b'file1.py\n'
    var_33 = b''
    var_34 = [var_30, var_20]
    var_35 = True
    var_36 = False
    var_37 = module_0.git_hook(var_35, var_36)
    assert var_37 == 0



# Parsed testcases at query #10
#--------------------------


import isort.hooks as module_0
import posixpath as module_1

def test_case_0():
    var_0 = 'file1.py'
    var_1 = 'file2.py'
    var_2 = 'file3.txt'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'import os\nimport sys'
    var_5 = ''
    var_6 = module_0.git_hook()
    assert var_6 == 0
    var_7 = '\n'
    var_8 = module_1.join(var_3)
    var_9 = False
    var_10 = module_0.git_hook(var_9)
    assert var_10 == 0
    var_11 = True
    var_12 = module_0.git_hook(var_11)
    assert var_12 == 1
    var_13 = module_0.git_hook(var_9, var_11)
    assert var_13 == 0
    var_14 = module_0.git_hook(var_11, var_11)
    assert var_14 == 1
    var_15 = module_0.git_hook(var_11, lazy=var_11)
    assert var_15 == 1
    var_16 = '.isort.cfg'
    var_17 = module_0.git_hook(var_11, settings_file=var_16)
    assert var_17 == 1
    var_18 = 'dir1'
    var_19 = [var_18]
    var_20 = module_0.git_hook(var_11, directories=var_19)
    assert var_20 == 0
    var_21 = 'test.py'
    var_22 = [var_21]
    var_23 = module_0.git_hook(var_11)
    assert var_23 == 0
    var_24 = module_0.git_hook(var_11)
    assert var_24 == 0



# Parsed testcases at query #11
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.git_hook(var_0, var_1)
    assert var_2 == 0
    var_3 = True
    var_4 = False
    var_5 = module_0.git_hook(var_3, var_4)
    assert var_5 == 2
    var_6 = False
    var_7 = module_0.git_hook(var_6, var_6)
    assert var_7 == 0
    var_8 = True
    var_9 = False
    var_10 = module_0.git_hook(var_8, var_9)
    assert var_10 == 0
    var_11 = True
    var_12 = False
    var_13 = module_0.git_hook(var_11, var_12)
    assert var_13 == 0
    var_14 = True
    var_15 = module_0.git_hook(var_14, var_14)
    assert var_15 == 1
    var_16 = True
    var_17 = False
    var_18 = module_0.git_hook(var_16, var_17, var_16)
    assert var_18 == 0
    var_19 = True
    var_20 = False
    var_21 = module_0.git_hook(var_19, var_20)
    assert var_21 == 0
    var_22 = '.isort.cfg'
    var_23 = True
    var_24 = False
    var_25 = module_0.git_hook(var_23, var_24, settings_file=var_22)
    assert var_25 == 0
    var_26 = 'src/'
    var_27 = [var_26]
    var_28 = True
    var_29 = False
    var_30 = module_0.git_hook(var_28, var_29, directories=var_27)
    assert var_30 == 1



# Parsed testcases at query #12
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.git_hook(var_0, var_1)
    assert var_2 == 0
    var_3 = True
    var_4 = False
    var_5 = module_0.git_hook(var_3, var_4)
    assert var_5 == 0
    var_6 = True
    var_7 = False
    var_8 = module_0.git_hook(var_6, var_7)
    assert var_8 == 0
    var_9 = True
    var_10 = False
    var_11 = module_0.git_hook(var_9, var_10)
    assert var_11 == 1
    var_12 = False
    var_13 = module_0.git_hook(var_12, var_12)
    assert var_13 == 0
    var_14 = True
    var_15 = module_0.git_hook(var_14, var_14)
    assert var_15 == 1
    var_16 = 'Skipped'
    var_17 = True
    var_18 = False
    var_19 = module_0.git_hook(var_17, var_18)
    assert var_19 == 0
    var_20 = True
    var_21 = False
    var_22 = module_0.git_hook(var_20, var_21, var_20)
    assert var_22 == 0
    var_23 = '--cached'
    var_24 = True
    var_25 = False
    var_26 = 'src/'
    var_27 = 'tests/'
    var_28 = [var_26, var_27]
    var_29 = module_0.git_hook(var_24, var_25, directories=var_28)
    assert var_29 == 0
    var_30 = True
    var_31 = False
    var_32 = module_0.git_hook(var_30, var_31)
    assert var_32 == 1



# Parsed testcases at query #13
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'git_hook_module'
    var_1 = __import__(var_0)
    var_2 = var_1.get_lines
    var_3 = __import__(var_0)
    var_4 = var_3.get_output
    var_5 = __import__(var_0)
    var_6 = var_5.api.check_code_string
    var_7 = __import__(var_0)
    var_8 = var_7.api.sort_file
    var_9 = __import__(var_0)
    var_10 = var_9.Config
    var_11 = True
    var_12 = False
    var_13 = module_0.git_hook(var_11, var_12)
    assert var_13 == 0
    var_14 = 'file1.py'
    var_15 = module_0.git_hook(var_11, var_12)
    assert var_15 == 0
    var_16 = module_0.git_hook(var_11, var_12)
    assert var_16 == 1
    var_17 = module_0.git_hook(var_12, var_12)
    assert var_17 == 0
    var_18 = module_0.git_hook(var_11, var_11)
    assert var_18 == 1
    var_19 = 'file1.txt'
    var_20 = 'file2.py'
    var_21 = module_0.git_hook(var_11, var_12)
    var_22 = 'content'
    var_23 = 'pathlib'
    var_24 = __import__(var_23)
    var_25 = module_0.git_hook(lazy=var_11)
    var_26 = 'src'
    var_27 = 'tests'
    var_28 = [var_26, var_27]
    var_29 = module_0.git_hook(directories=var_28)
    var_30 = 'git_hook_module'
    var_31 = __import__(var_30)
    var_32 = module_0.git_hook(var_11, var_12)
    assert var_32 == 0
    var_33 = 'file3.py'
    var_34 = module_0.git_hook(var_11, var_12)
    assert var_34 == 2



# Parsed testcases at query #14
#--------------------------


import isort.hooks as module_0
import posixpath as module_1

def test_case_0():
    var_0 = 'get_lines'
    var_1 = 'test_module'
    var_2 = __import__(var_1)
    var_3 = var_2.git_hook.__globals__[var_0]
    var_4 = 'get_output'
    var_5 = __import__(var_1)
    var_6 = var_5.git_hook.__globals__[var_4]
    var_7 = 'api'
    var_8 = __import__(var_1)
    var_9 = var_8.git_hook.__globals__[var_7]
    var_10 = var_9.check_code_string
    var_11 = __import__(var_1)
    var_12 = var_11.git_hook.__globals__[var_7]
    var_13 = var_12.sort_file
    var_14 = 'Config'
    var_15 = __import__(var_1)
    var_16 = var_15.git_hook.__globals__[var_14]
    var_17 = True
    var_18 = False
    var_19 = module_0.git_hook(var_17, var_18)
    assert var_19 == 0
    var_20 = 'README.md'
    var_21 = 'requirements.txt'
    var_22 = module_0.git_hook(var_17, var_18)
    assert var_22 == 0
    var_23 = 'src/main.py'
    var_24 = module_0.git_hook(var_17, var_18)
    assert var_24 == 0
    var_25 = module_0.git_hook(var_17, var_18)
    assert var_25 == 1
    var_26 = module_0.git_hook(var_18, var_18)
    assert var_26 == 0
    var_27 = module_0.git_hook(var_17, var_17)
    assert var_27 == 1
    var_28 = 'src/file1.py'
    var_29 = 'src/file2.py'
    var_30 = 'docs/readme.md'
    var_31 = 'import a'
    var_32 = 'import b'
    var_33 = 'text content'
    var_34 = module_0.git_hook(var_17, var_18)
    assert var_34 == 1
    var_35 = module_0.git_hook(var_17, var_18, var_17)
    var_36 = 'src/'
    var_37 = [var_36]
    var_38 = module_0.git_hook(var_17, var_18, directories=var_37)
    var_39 = module_0.git_hook(var_17, var_18)
    assert var_39 == 0
    var_40 = 'pyproject.toml'
    var_41 = module_0.git_hook(var_17, var_18, settings_file=var_40)
    var_42 = module_1.abspath(var_23)
    var_43 = module_1.dirname(var_42)



# Parsed testcases at query #15
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.git_hook(var_0, var_1)
    assert var_2 == 0
    var_3 = True
    var_4 = False
    var_5 = module_0.git_hook(var_3, var_4)
    assert var_5 == 1
    var_6 = True
    var_7 = False
    var_8 = module_0.git_hook(var_6, var_7)
    assert var_8 == 0
    var_9 = False
    var_10 = module_0.git_hook(var_9, var_9)
    assert var_10 == 0
    var_11 = True
    var_12 = module_0.git_hook(var_11, var_11)
    assert var_12 == 1
    var_13 = True
    var_14 = False
    var_15 = module_0.git_hook(var_13, var_14)
    assert var_15 == 0
    var_16 = True
    var_17 = module_0.git_hook(lazy=var_16)
    var_18 = 0
    var_19 = 'src'
    var_20 = 'tests'
    var_21 = [var_19, var_20]
    var_22 = module_0.git_hook(directories=var_21)
    var_23 = 0
    var_24 = 'File skipped'
    var_25 = True
    var_26 = False
    var_27 = module_0.git_hook(var_25, var_26)
    assert var_27 == 0
    var_28 = False
    var_29 = True
    var_30 = module_0.git_hook(var_29, var_28)
    assert var_30 == 2



