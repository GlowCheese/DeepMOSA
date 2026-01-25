####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = True
    var_2 = module_0.git_hook(var_1)
    assert var_2 == 0
    var_3 = module_0.git_hook(modify=var_1)
    assert var_3 == 0
    var_4 = module_0.git_hook(lazy=var_1)
    assert var_4 == 0
    var_5 = ''
    var_6 = module_0.git_hook(settings_file=var_5)
    assert var_6 == 0
    var_7 = 'src/'
    var_8 = [var_7]
    var_9 = module_0.git_hook(directories=var_8)
    assert var_9 == 0
    var_10 = [var_7]
    var_11 = module_0.git_hook(var_1, var_1, var_1, var_5, var_10)
    assert var_11 == 0



# Parsed testcases at query #2
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'echo'
    var_1 = '-e'
    var_2 = 'line1\nline2\nline3'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'line1'
    var_5 = 'line2'
    var_6 = 'line3'
    var_7 = [var_4, var_5, var_6]
    var_8 = module_0.get_lines(var_3)
    var_9 = 'single_line'
    var_10 = [var_0, var_9]
    var_11 = [var_9]
    var_12 = module_0.get_lines(var_10)
    var_13 = ''
    var_14 = [var_0, var_13]
    var_15 = [var_13]
    var_16 = module_0.get_lines(var_14)
    var_17 = '  line1  \n  line2  \n  line3  '
    var_18 = [var_0, var_1, var_17]
    var_19 = [var_4, var_5, var_6]
    var_20 = module_0.get_lines(var_18)



# Parsed testcases at query #3
#--------------------------


import isort.hooks as module_0
import posixpath as module_1

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = False
    var_2 = module_0.git_hook(var_1)
    assert var_2 == 0
    var_3 = True
    var_4 = module_0.git_hook(var_3)
    assert var_4 == 2
    var_5 = True
    var_6 = module_0.git_hook(modify=var_5)
    assert var_6 == 0
    var_7 = True
    var_8 = module_0.git_hook(lazy=var_7)
    assert var_8 == 0
    var_9 = 'git'
    var_10 = 'diff-index'
    var_11 = '--name-only'
    var_12 = '--diff-filter=ACMRTUXB'
    var_13 = 'HEAD'
    var_14 = [var_9, var_10, var_11, var_12, var_13]
    var_15 = 'src/'
    var_16 = 'tests/'
    var_17 = [var_15, var_16]
    var_18 = module_0.git_hook(directories=var_17)
    assert var_18 == 0
    var_19 = 'git'
    var_20 = 'diff-index'
    var_21 = '--cached'
    var_22 = '--name-only'
    var_23 = '--diff-filter=ACMRTUXB'
    var_24 = 'HEAD'
    var_25 = [var_19, var_20, var_21, var_22, var_23, var_24, var_15, var_16]
    var_26 = 'pyproject.toml'
    var_27 = module_0.git_hook(settings_file=var_26)
    assert var_27 == 0
    var_28 = 'file1.py'
    var_29 = module_1.abspath(var_28)
    var_30 = module_1.dirname(var_29)
    var_31 = module_0.git_hook()
    assert var_31 == 0



# Parsed testcases at query #4
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'echo'
    var_1 = 'line1\nline2\nline3'
    var_2 = [var_0, var_1]
    var_3 = 'line1'
    var_4 = 'line2'
    var_5 = 'line3'
    var_6 = [var_3, var_4, var_5]
    var_7 = module_0.get_lines(var_2)
    var_8 = ''
    var_9 = [var_0, var_8]
    var_10 = []
    var_11 = module_0.get_lines(var_9)
    var_12 = 'single_line'
    var_13 = [var_0, var_12]
    var_14 = [var_12]
    var_15 = module_0.get_lines(var_13)



# Parsed testcases at query #5
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'echo'
    var_1 = '-e'
    var_2 = 'line1\nline2\nline3'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.get_lines(var_3)
    var_5 = 'single_line'
    var_6 = [var_0, var_5]
    var_7 = module_0.get_lines(var_6)
    var_8 = 'line1\n\nline2'
    var_9 = [var_0, var_1, var_8]
    var_10 = module_0.get_lines(var_9)



# Parsed testcases at query #6
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = module_0.git_hook()
    assert var_1 == 0
    var_2 = module_0.git_hook()
    assert var_2 == 0
    var_3 = module_0.git_hook()
    assert var_3 == 0
    var_4 = True
    var_5 = module_0.git_hook(var_4)
    assert var_5 == 2
    var_6 = True
    var_7 = module_0.git_hook(modify=var_6)
    assert var_7 == 0
    var_8 = True
    var_9 = module_0.git_hook(lazy=var_8)
    assert var_9 == 0
    var_10 = 'test.py'
    var_11 = 'src/'
    var_12 = [var_11]
    var_13 = module_0.git_hook(settings_file=var_10, directories=var_12)
    assert var_13 == 0



# Parsed testcases at query #7
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = module_0.git_hook()
    assert var_1 == 0
    var_2 = module_0.git_hook()
    assert var_2 == 0
    var_3 = True
    var_4 = module_0.git_hook(var_3)
    assert var_4 == 2
    var_5 = True
    var_6 = module_0.git_hook(modify=var_5)
    assert var_6 == 0
    var_7 = True
    var_8 = module_0.git_hook(var_7, var_7)
    assert var_8 == 2
    var_9 = True
    var_10 = module_0.git_hook(lazy=var_9)
    assert var_10 == 0
    var_11 = 'src/'
    var_12 = [var_11]
    var_13 = module_0.git_hook(directories=var_12)
    assert var_13 == 0
    var_14 = '.isort.cfg'
    var_15 = module_0.git_hook(settings_file=var_14)
    assert var_15 == 0
    var_16 = module_0.git_hook()
    assert var_16 == 0



# Parsed testcases at query #8
#--------------------------


import isort.hooks as module_0
import posixpath as module_1

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = []
    var_2 = 0
    var_3 = b'test.py\n'
    var_4 = []
    var_5 = b'import os\nimport sys\n'
    var_6 = True
    var_7 = module_0.git_hook(var_6)
    assert var_7 == 1
    var_8 = module_0.git_hook(var_6, var_6)
    var_9 = []
    var_10 = 0
    var_11 = b'test.py\n'
    var_12 = []
    var_13 = b'import os\nimport sys\n'
    var_14 = False
    var_15 = module_0.git_hook(var_14)
    assert var_15 == 0
    var_16 = []
    var_17 = 0
    var_18 = b'test.py\n'
    var_19 = []
    var_20 = b'import os\nimport sys\n'
    var_21 = True
    var_22 = module_0.git_hook(var_21, lazy=var_21)
    assert var_22 == 1
    var_23 = 'git'
    var_24 = 'diff-index'
    var_25 = '--name-only'
    var_26 = '--diff-filter=ACMRTUXB'
    var_27 = 'HEAD'
    var_28 = [var_23, var_24, var_25, var_26, var_27]
    var_29 = []
    var_30 = 0
    var_31 = b'src/test.py\n'
    var_32 = []
    var_33 = b'import os\nimport sys\n'
    var_34 = 'src/'
    var_35 = [var_34]
    var_36 = True
    var_37 = module_0.git_hook(var_36, directories=var_35)
    assert var_37 == 1
    var_38 = 'git'
    var_39 = 'diff-index'
    var_40 = '--cached'
    var_41 = '--name-only'
    var_42 = '--diff-filter=ACMRTUXB'
    var_43 = 'HEAD'
    var_44 = [var_38, var_39, var_40, var_41, var_42, var_43, var_34]
    var_45 = []
    var_46 = 0
    var_47 = b'test.py\n'
    var_48 = []
    var_49 = b'import os\nimport sys\n'
    var_50 = '.isort.cfg'
    var_51 = True
    var_52 = module_0.git_hook(var_51, settings_file=var_50)
    var_53 = 'test.py'
    var_54 = module_1.abspath(var_53)
    var_55 = module_1.dirname(var_54)
    var_56 = []
    var_57 = 0
    var_58 = b'test.py\n'
    var_59 = []
    var_60 = b'import os\nimport sys\n'
    var_61 = True
    var_62 = module_0.git_hook(var_61)
    assert var_62 == 0



# Parsed testcases at query #9
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = True
    var_2 = module_0.git_hook(var_1)
    assert var_2 == 0
    var_3 = module_0.git_hook(modify=var_1)
    assert var_3 == 0
    var_4 = module_0.git_hook(lazy=var_1)
    assert var_4 == 0
    var_5 = ''
    var_6 = module_0.git_hook(settings_file=var_5)
    assert var_6 == 0
    var_7 = 'src'
    var_8 = [var_7]
    var_9 = module_0.git_hook(directories=var_8)
    assert var_9 == 0
    var_10 = [var_7]
    var_11 = module_0.git_hook(var_1, var_1, var_1, var_5, var_10)
    assert var_11 == 0



# Parsed testcases at query #10
#--------------------------


import isort.hooks as module_0
import posixpath as module_1

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = b'file1.py\nfile2.py'
    var_2 = b'print("test")'
    var_3 = module_0.git_hook()
    assert var_3 == 0
    var_4 = b'file1.py\nfile2.py'
    var_5 = b'print("test")'
    var_6 = True
    var_7 = module_0.git_hook(var_6)
    assert var_7 == 1
    var_8 = b'file1.py\nfile2.py'
    var_9 = b'print("test")'
    var_10 = True
    var_11 = module_0.git_hook(modify=var_10)
    var_12 = b'file1.py\nfile2.py'
    var_13 = b'print("test")'
    var_14 = True
    var_15 = module_0.git_hook(lazy=var_14)
    var_16 = 'git'
    var_17 = 'diff-index'
    var_18 = '--name-only'
    var_19 = '--diff-filter=ACMRTUXB'
    var_20 = 'HEAD'
    var_21 = [var_16, var_17, var_18, var_19, var_20]
    var_22 = b'src/file1.py\nsrc/file2.py'
    var_23 = b'print("test")'
    var_24 = 'src/'
    var_25 = [var_24]
    var_26 = module_0.git_hook(directories=var_25)
    var_27 = 'git'
    var_28 = 'diff-index'
    var_29 = '--cached'
    var_30 = '--name-only'
    var_31 = '--diff-filter=ACMRTUXB'
    var_32 = 'HEAD'
    var_33 = [var_27, var_28, var_29, var_30, var_31, var_32, var_24]
    var_34 = True
    var_35 = b'file1.py\nfile2.py'
    var_36 = b'print("test")'
    var_37 = '.isort.cfg'
    var_38 = module_0.git_hook(settings_file=var_37)
    var_39 = 'file1.py'
    var_40 = module_1.abspath(var_39)
    var_41 = module_1.dirname(var_40)
    var_42 = b'file1.py\nfile2.py'
    var_43 = b'print("test")'
    var_44 = module_0.git_hook()
    assert var_44 == 0



# Parsed testcases at query #11
#--------------------------


import isort.hooks as module_0
import posixpath as module_1

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = module_0.git_hook()
    assert var_1 == 0
    var_2 = module_0.git_hook()
    assert var_2 == 0
    var_3 = True
    var_4 = module_0.git_hook(var_3)
    assert var_4 == 2
    var_5 = True
    var_6 = module_0.git_hook(modify=var_5)
    assert var_6 == 0
    var_7 = True
    var_8 = module_0.git_hook(lazy=var_7)
    var_9 = 'git'
    var_10 = 'diff-index'
    var_11 = '--name-only'
    var_12 = '--diff-filter=ACMRTUXB'
    var_13 = 'HEAD'
    var_14 = [var_9, var_10, var_11, var_12, var_13]
    var_15 = 'src/'
    var_16 = [var_15]
    var_17 = module_0.git_hook(directories=var_16)
    var_18 = 'git'
    var_19 = 'diff-index'
    var_20 = '--cached'
    var_21 = '--name-only'
    var_22 = '--diff-filter=ACMRTUXB'
    var_23 = 'HEAD'
    var_24 = [var_18, var_19, var_20, var_21, var_22, var_23, var_15]
    var_25 = True
    var_26 = '.isort.cfg'
    var_27 = module_0.git_hook(settings_file=var_26)
    var_28 = 'file1.py'
    var_29 = module_1.abspath(var_28)
    var_30 = module_1.dirname(var_29)
    var_31 = module_0.git_hook()
    assert var_31 == 0



# Parsed testcases at query #12
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = True
    var_2 = module_0.git_hook(var_1)
    assert var_2 == 0
    var_3 = module_0.git_hook(modify=var_1)
    assert var_3 == 0
    var_4 = module_0.git_hook(lazy=var_1)
    assert var_4 == 0
    var_5 = ''
    var_6 = module_0.git_hook(settings_file=var_5)
    assert var_6 == 0
    var_7 = '.'
    var_8 = [var_7]
    var_9 = module_0.git_hook(directories=var_8)
    assert var_9 == 0



# Parsed testcases at query #13
#--------------------------


import isort.hooks as module_0
import posixpath as module_1

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = False
    var_2 = module_0.git_hook(var_1)
    assert var_2 == 0
    var_3 = True
    var_4 = module_0.git_hook(var_3)
    assert var_4 == 2
    var_5 = True
    var_6 = module_0.git_hook(modify=var_5)
    assert var_6 == 0
    var_7 = True
    var_8 = module_0.git_hook(lazy=var_7)
    assert var_8 == 0
    var_9 = 'git'
    var_10 = 'diff-index'
    var_11 = '--name-only'
    var_12 = '--diff-filter=ACMRTUXB'
    var_13 = 'HEAD'
    var_14 = [var_9, var_10, var_11, var_12, var_13]
    var_15 = 'src/'
    var_16 = [var_15]
    var_17 = module_0.git_hook(directories=var_16)
    assert var_17 == 0
    var_18 = 'git'
    var_19 = 'diff-index'
    var_20 = '--cached'
    var_21 = '--name-only'
    var_22 = '--diff-filter=ACMRTUXB'
    var_23 = 'HEAD'
    var_24 = [var_18, var_19, var_20, var_21, var_22, var_23, var_15]
    var_25 = True
    var_26 = '.isort.cfg'
    var_27 = module_0.git_hook(settings_file=var_26)
    var_28 = 'file1.py'
    var_29 = module_1.abspath(var_28)
    var_30 = module_1.dirname(var_29)



# Parsed testcases at query #14
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = module_0.git_hook()
    assert var_1 == 0
    var_2 = True
    var_3 = module_0.git_hook(var_2)
    assert var_3 == 2
    var_4 = True
    var_5 = module_0.git_hook(modify=var_4)
    var_6 = True
    var_7 = module_0.git_hook(lazy=var_6)
    var_8 = 'git'
    var_9 = 'diff-index'
    var_10 = '--name-only'
    var_11 = '--diff-filter=ACMRTUXB'
    var_12 = 'HEAD'
    var_13 = [var_8, var_9, var_10, var_11, var_12]
    var_14 = 'src/'
    var_15 = 'tests/'
    var_16 = [var_14, var_15]
    var_17 = module_0.git_hook(directories=var_16)
    var_18 = 'git'
    var_19 = 'diff-index'
    var_20 = '--cached'
    var_21 = '--name-only'
    var_22 = '--diff-filter=ACMRTUXB'
    var_23 = 'HEAD'
    var_24 = [var_18, var_19, var_20, var_21, var_22, var_23, var_14, var_15]
    var_25 = True



# Parsed testcases at query #15
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = module_0.git_hook()
    assert var_1 == 0
    var_2 = module_0.git_hook()
    assert var_2 == 0
    var_3 = module_0.git_hook()
    assert var_3 == 0
    var_4 = True
    var_5 = module_0.git_hook(var_4)
    assert var_5 == 2
    var_6 = True
    var_7 = module_0.git_hook(modify=var_6)
    assert var_7 == 0
    var_8 = True
    var_9 = module_0.git_hook(var_8, var_8)
    assert var_9 == 2
    var_10 = True
    var_11 = module_0.git_hook(lazy=var_10)
    var_12 = 'git'
    var_13 = 'diff-index'
    var_14 = '--name-only'
    var_15 = '--diff-filter=ACMRTUXB'
    var_16 = 'HEAD'
    var_17 = [var_12, var_13, var_14, var_15, var_16]
    var_18 = 'src/'
    var_19 = [var_18]
    var_20 = module_0.git_hook(directories=var_19)
    var_21 = 'git'
    var_22 = 'diff-index'
    var_23 = '--cached'
    var_24 = '--name-only'
    var_25 = '--diff-filter=ACMRTUXB'
    var_26 = 'HEAD'
    var_27 = [var_21, var_22, var_23, var_24, var_25, var_26, var_18]
    var_28 = True
    var_29 = 'pyproject.toml'
    var_30 = module_0.git_hook(settings_file=var_29)



# Parsed testcases at query #16
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = False
    var_2 = module_0.git_hook(var_1)
    assert var_2 == 0
    var_3 = True
    var_4 = module_0.git_hook(var_3)
    assert var_4 == 0
    var_5 = module_0.git_hook(modify=var_3)
    assert var_5 == 0
    var_6 = module_0.git_hook(lazy=var_3)
    assert var_6 == 0
    var_7 = ''
    var_8 = module_0.git_hook(settings_file=var_7)
    assert var_8 == 0
    var_9 = '.'
    var_10 = [var_9]
    var_11 = module_0.git_hook(directories=var_10)
    assert var_11 == 0
    var_12 = [var_9]
    var_13 = module_0.git_hook(var_3, var_3, var_3, var_7, var_12)
    assert var_13 == 0



# Parsed testcases at query #17
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = False
    var_2 = module_0.git_hook(var_1)
    assert var_2 == 0
    var_3 = True
    var_4 = module_0.git_hook(var_3)
    assert var_4 == 1
    var_5 = True
    var_6 = module_0.git_hook(modify=var_5)
    var_7 = True
    var_8 = module_0.git_hook(lazy=var_7)
    var_9 = 'git'
    var_10 = 'diff-index'
    var_11 = '--name-only'
    var_12 = '--diff-filter=ACMRTUXB'
    var_13 = 'HEAD'
    var_14 = [var_9, var_10, var_11, var_12, var_13]
    var_15 = 'src'
    var_16 = [var_15]
    var_17 = module_0.git_hook(directories=var_16)
    var_18 = 'git'
    var_19 = 'diff-index'
    var_20 = '--cached'
    var_21 = '--name-only'
    var_22 = '--diff-filter=ACMRTUXB'
    var_23 = 'HEAD'
    var_24 = [var_18, var_19, var_20, var_21, var_22, var_23, var_15]
    var_25 = True
    var_26 = module_0.git_hook()
    assert var_26 == 0



# Parsed testcases at query #18
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = module_0.git_hook()
    assert var_1 == 0
    var_2 = True
    var_3 = module_0.git_hook(var_2)
    var_4 = False
    var_5 = module_0.git_hook(var_4)
    assert var_5 == 0
    var_6 = module_0.git_hook(modify=var_2)
    assert var_6 == 0
    var_7 = module_0.git_hook(var_2, var_2)
    var_8 = module_0.git_hook(lazy=var_2)
    assert var_8 == 0
    var_9 = 'pyproject.toml'
    var_10 = module_0.git_hook(settings_file=var_9)
    assert var_10 == 0
    var_11 = 'src/'
    var_12 = [var_11]
    var_13 = module_0.git_hook(directories=var_12)
    assert var_13 == 0



# Parsed testcases at query #19
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = module_0.git_hook()
    assert var_1 == 0
    var_2 = True
    var_3 = module_0.git_hook(var_2)
    assert var_3 == 1
    var_4 = True
    var_5 = module_0.git_hook(modify=var_4)
    var_6 = 'file.py'
    var_7 = True
    var_8 = module_0.git_hook(lazy=var_7)
    var_9 = 'git'
    var_10 = 'diff-index'
    var_11 = '--name-only'
    var_12 = '--diff-filter=ACMRTUXB'
    var_13 = 'HEAD'
    var_14 = [var_9, var_10, var_11, var_12, var_13]
    var_15 = 'src/'
    var_16 = [var_15]
    var_17 = module_0.git_hook(directories=var_16)
    var_18 = 'git'
    var_19 = 'diff-index'
    var_20 = '--cached'
    var_21 = '--name-only'
    var_22 = '--diff-filter=ACMRTUXB'
    var_23 = 'HEAD'
    var_24 = [var_18, var_19, var_20, var_21, var_22, var_23, var_15]
    var_25 = True
    var_26 = module_0.git_hook()
    assert var_26 == 0
    var_27 = module_0.git_hook()
    assert var_27 == 0



# Parsed testcases at query #20
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'subprocess.run'
    var_1 = []
    var_2 = b''
    var_3 = module_0.git_hook()
    assert var_3 == 0
    var_4 = []
    var_5 = b'file1.txt\nfile2.md'
    var_6 = module_0.git_hook()
    assert var_6 == 0
    var_7 = []
    var_8 = b'file1.py\nfile2.py'
    var_9 = []
    var_10 = b'print("hello")'
    var_11 = []
    var_12 = b'print("world")'
    var_13 = 'isort.api.check_code_string'
    var_14 = True
    var_15 = module_0.git_hook()
    assert var_15 == 0
    var_16 = []
    var_17 = []
    var_18 = []
    var_19 = False
    var_20 = module_0.git_hook(var_14)
    assert var_20 == 2
    var_21 = []
    var_22 = []
    var_23 = []
    var_24 = 'isort.api.sort_file'
    var_25 = module_0.git_hook(modify=var_14)
    var_26 = []
    var_27 = []
    var_28 = []
    var_29 = module_0.git_hook(lazy=var_14)
    assert var_29 == 0
    var_30 = []
    var_31 = b'src/file1.py\nsrc/file2.py'
    var_32 = []
    var_33 = []
    var_34 = 'src/'
    var_35 = [var_34]
    var_36 = module_0.git_hook(directories=var_35)
    assert var_36 == 0



# Parsed testcases at query #21
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = False
    var_2 = module_0.git_hook(var_1)
    assert var_2 == 0
    var_3 = True
    var_4 = module_0.git_hook(var_3)
    assert var_4 == 0
    var_5 = module_0.git_hook(modify=var_3)
    assert var_5 == 0
    var_6 = module_0.git_hook(lazy=var_3)
    assert var_6 == 0
    var_7 = ''
    var_8 = module_0.git_hook(settings_file=var_7)
    assert var_8 == 0
    var_9 = 'src/'
    var_10 = [var_9]
    var_11 = module_0.git_hook(directories=var_10)
    assert var_11 == 0



# Parsed testcases at query #22
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = True
    var_2 = module_0.git_hook(var_1)
    assert var_2 == 0
    var_3 = module_0.git_hook(modify=var_1)
    assert var_3 == 0
    var_4 = module_0.git_hook(lazy=var_1)
    assert var_4 == 0
    var_5 = 'src/'
    var_6 = [var_5]
    var_7 = module_0.git_hook(directories=var_6)
    assert var_7 == 0
    var_8 = '.isort.cfg'
    var_9 = module_0.git_hook(settings_file=var_8)
    assert var_9 == 0



# Parsed testcases at query #23
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'subprocess.run'
    var_1 = []
    var_2 = b''
    var_3 = module_0.git_hook()
    assert var_3 == 0
    var_4 = []
    var_5 = b'file1.py\nfile2.py'
    var_6 = []
    var_7 = b'print("test")'
    var_8 = 'isort.api.check_code_string'
    var_9 = False
    var_10 = module_0.git_hook(var_9)
    assert var_10 == 0
    var_11 = []
    var_12 = []
    var_13 = True
    var_14 = module_0.git_hook(var_13)
    assert var_14 == 2
    var_15 = []
    var_16 = []
    var_17 = 'isort.api.sort_file'
    var_18 = module_0.git_hook(modify=var_13)
    var_19 = []
    var_20 = []
    var_21 = module_0.git_hook(lazy=var_13)
    var_22 = 'git'
    var_23 = 'diff-index'
    var_24 = '--name-only'
    var_25 = '--diff-filter=ACMRTUXB'
    var_26 = 'HEAD'
    var_27 = [var_22, var_23, var_24, var_25, var_26]
    var_28 = []
    var_29 = []
    var_30 = 'src/'
    var_31 = [var_30]
    var_32 = module_0.git_hook(directories=var_31)
    var_33 = '--cached'
    var_34 = [var_22, var_23, var_33, var_24, var_25, var_26, var_30]
    var_35 = []
    var_36 = b'file1.txt\nfile2.md'
    var_37 = []
    var_38 = module_0.git_hook()
    assert var_38 == 0
    var_39 = []
    var_40 = b'file1.py'
    var_41 = []
    var_42 = module_0.git_hook()
    assert var_42 == 0



# Parsed testcases at query #24
#--------------------------


import isort.hooks as module_0
import posixpath as module_1

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = 'file.txt'
    var_2 = 'file.js'
    var_3 = module_0.git_hook()
    assert var_3 == 0
    var_4 = 'file.py'
    var_5 = module_0.git_hook()
    assert var_5 == 0
    var_6 = module_0.git_hook()
    assert var_6 == 0
    var_7 = True
    var_8 = module_0.git_hook(var_7)
    assert var_8 == 1
    var_9 = module_0.git_hook(modify=var_7)
    var_10 = module_0.git_hook(lazy=var_7)
    var_11 = 'git'
    var_12 = 'diff-index'
    var_13 = '--name-only'
    var_14 = '--diff-filter=ACMRTUXB'
    var_15 = 'HEAD'
    var_16 = [var_11, var_12, var_13, var_14, var_15]
    var_17 = 'src'
    var_18 = [var_17]
    var_19 = module_0.git_hook(directories=var_18)
    var_20 = '--cached'
    var_21 = [var_11, var_12, var_20, var_13, var_14, var_15, var_17]
    var_22 = 'setup.cfg'
    var_23 = module_0.git_hook(settings_file=var_22)
    var_24 = module_1.abspath(var_4)
    var_25 = module_1.dirname(var_24)



# Parsed testcases at query #25
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = True
    var_2 = module_0.git_hook(var_1)
    assert var_2 == 0
    var_3 = module_0.git_hook(modify=var_1)
    assert var_3 == 0
    var_4 = module_0.git_hook(lazy=var_1)
    assert var_4 == 0
    var_5 = ''
    var_6 = module_0.git_hook(settings_file=var_5)
    assert var_6 == 0
    var_7 = '.'
    var_8 = [var_7]
    var_9 = module_0.git_hook(directories=var_8)
    assert var_9 == 0
    var_10 = [var_7]
    var_11 = module_0.git_hook(var_1, var_1, var_1, var_5, var_10)
    assert var_11 == 0



# Parsed testcases at query #26
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = False
    var_2 = module_0.git_hook(var_1)
    assert var_2 == 0
    var_3 = True
    var_4 = module_0.git_hook(var_3)
    assert var_4 == 1
    var_5 = module_0.git_hook(modify=var_3)
    assert var_5 == 0
    var_6 = module_0.git_hook(lazy=var_3)
    assert var_6 == 0
    var_7 = 'pyproject.toml'
    var_8 = module_0.git_hook(settings_file=var_7)
    assert var_8 == 0
    var_9 = 'src/'
    var_10 = [var_9]
    var_11 = module_0.git_hook(directories=var_10)
    assert var_11 == 0



# Parsed testcases at query #27
#--------------------------


import isort.hooks as module_0
import posixpath as module_1

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = module_0.git_hook()
    assert var_1 == 0
    var_2 = True
    var_3 = module_0.git_hook(var_2)
    assert var_3 == 0
    var_4 = True
    var_5 = module_0.git_hook(var_4)
    assert var_5 == 2
    var_6 = True
    var_7 = module_0.git_hook(var_6, var_6)
    assert var_7 == 2
    var_8 = True
    var_9 = module_0.git_hook(lazy=var_8)
    assert var_9 == 0
    var_10 = 'git'
    var_11 = 'diff-index'
    var_12 = '--name-only'
    var_13 = '--diff-filter=ACMRTUXB'
    var_14 = 'HEAD'
    var_15 = [var_10, var_11, var_12, var_13, var_14]
    var_16 = 'src'
    var_17 = 'tests'
    var_18 = [var_16, var_17]
    var_19 = module_0.git_hook(directories=var_18)
    assert var_19 == 0
    var_20 = 'git'
    var_21 = 'diff-index'
    var_22 = '--cached'
    var_23 = '--name-only'
    var_24 = '--diff-filter=ACMRTUXB'
    var_25 = 'HEAD'
    var_26 = [var_20, var_21, var_22, var_23, var_24, var_25, var_16, var_17]
    var_27 = True
    var_28 = 'pyproject.toml'
    var_29 = module_0.git_hook(settings_file=var_28)
    assert var_29 == 0
    var_30 = 'file1.py'
    var_31 = module_1.abspath(var_30)
    var_32 = module_1.dirname(var_31)



# Parsed testcases at query #28
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = module_0.git_hook()
    assert var_1 == 0
    var_2 = module_0.git_hook()
    assert var_2 == 0
    var_3 = module_0.git_hook()
    assert var_3 == 0
    var_4 = True
    var_5 = module_0.git_hook(var_4)
    assert var_5 == 1
    var_6 = True
    var_7 = module_0.git_hook(modify=var_6)
    assert var_7 == 0
    var_8 = 'file.py'
    var_9 = True
    var_10 = module_0.git_hook(lazy=var_9)
    assert var_10 == 0
    var_11 = 'git'
    var_12 = 'diff-index'
    var_13 = '--name-only'
    var_14 = '--diff-filter=ACMRTUXB'
    var_15 = 'HEAD'
    var_16 = [var_11, var_12, var_13, var_14, var_15]
    var_17 = 'src'
    var_18 = [var_17]
    var_19 = module_0.git_hook(directories=var_18)
    assert var_19 == 0
    var_20 = 'git'
    var_21 = 'diff-index'
    var_22 = '--cached'
    var_23 = '--name-only'
    var_24 = '--diff-filter=ACMRTUXB'
    var_25 = 'HEAD'
    var_26 = [var_20, var_21, var_22, var_23, var_24, var_25, var_17]
    var_27 = True
    var_28 = 'pyproject.toml'
    var_29 = module_0.git_hook(settings_file=var_28)



# Parsed testcases at query #29
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = False
    var_2 = module_0.git_hook(var_1)
    assert var_2 == 0
    var_3 = True
    var_4 = module_0.git_hook(var_3)
    var_5 = module_0.git_hook(modify=var_3)
    assert var_5 == 0
    var_6 = module_0.git_hook(lazy=var_3)
    assert var_6 == 0
    var_7 = 'pyproject.toml'
    var_8 = module_0.git_hook(settings_file=var_7)
    assert var_8 == 0
    var_9 = 'src/'
    var_10 = [var_9]
    var_11 = module_0.git_hook(directories=var_10)
    assert var_11 == 0



# Parsed testcases at query #30
#--------------------------


import isort.hooks as module_0
import posixpath as module_1

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = module_0.git_hook()
    assert var_1 == 0
    var_2 = module_0.git_hook()
    assert var_2 == 0
    var_3 = False
    var_4 = module_0.git_hook(var_3)
    assert var_4 == 0
    var_5 = True
    var_6 = module_0.git_hook(var_5)
    assert var_6 == 2
    var_7 = True
    var_8 = module_0.git_hook(modify=var_7)
    var_9 = True
    var_10 = module_0.git_hook(lazy=var_9)
    var_11 = 'git'
    var_12 = 'diff-index'
    var_13 = '--name-only'
    var_14 = '--diff-filter=ACMRTUXB'
    var_15 = 'HEAD'
    var_16 = [var_11, var_12, var_13, var_14, var_15]
    var_17 = 'src/'
    var_18 = 'tests/'
    var_19 = [var_17, var_18]
    var_20 = module_0.git_hook(directories=var_19)
    var_21 = 'git'
    var_22 = 'diff-index'
    var_23 = '--cached'
    var_24 = '--name-only'
    var_25 = '--diff-filter=ACMRTUXB'
    var_26 = 'HEAD'
    var_27 = [var_21, var_22, var_23, var_24, var_25, var_26, var_17, var_18]
    var_28 = True
    var_29 = '.isort.cfg'
    var_30 = module_0.git_hook(settings_file=var_29)
    var_31 = 'file1.py'
    var_32 = module_1.abspath(var_31)
    var_33 = module_1.dirname(var_32)



# Parsed testcases at query #31
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = False
    var_2 = module_0.git_hook(var_1)
    assert var_2 == 0
    var_3 = True
    var_4 = module_0.git_hook(var_3)
    assert var_4 == 1
    var_5 = True
    var_6 = module_0.git_hook(modify=var_5)
    var_7 = True
    var_8 = module_0.git_hook(lazy=var_7)
    var_9 = 'git'
    var_10 = 'diff-index'
    var_11 = '--name-only'
    var_12 = '--diff-filter=ACMRTUXB'
    var_13 = 'HEAD'
    var_14 = [var_9, var_10, var_11, var_12, var_13]
    var_15 = 'src/'
    var_16 = [var_15]
    var_17 = module_0.git_hook(directories=var_16)
    var_18 = 'git'
    var_19 = 'diff-index'
    var_20 = '--cached'
    var_21 = '--name-only'
    var_22 = '--diff-filter=ACMRTUXB'
    var_23 = 'HEAD'
    var_24 = [var_18, var_19, var_20, var_21, var_22, var_23, var_15]
    var_25 = True



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = module_0.git_hook()
    assert var_1 == 0
    var_2 = module_0.git_hook()
    assert var_2 == 0
    var_3 = False
    var_4 = module_0.git_hook(var_3)
    assert var_4 == 0
    var_5 = True
    var_6 = module_0.git_hook(var_5)
    assert var_6 == 2
    var_7 = True
    var_8 = module_0.git_hook(modify=var_7)
    assert var_8 == 0
    var_9 = True
    var_10 = module_0.git_hook(lazy=var_9)
    assert var_10 == 0
    var_11 = 'src/'
    var_12 = [var_11]
    var_13 = module_0.git_hook(directories=var_12)
    assert var_13 == 0
    var_14 = module_0.git_hook()
    assert var_14 == 0



# Parsed testcases at query #2
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'echo'
    var_1 = '-e'
    var_2 = 'line1\nline2\nline3'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.get_lines(var_3)
    var_5 = 'single_line'
    var_6 = [var_0, var_5]
    var_7 = module_0.get_lines(var_6)
    var_8 = 'line1\n\nline2'
    var_9 = [var_0, var_1, var_8]
    var_10 = module_0.get_lines(var_9)



# Parsed testcases at query #3
#--------------------------


import isort.hooks as module_0
import posixpath as module_1

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = module_0.git_hook()
    assert var_1 == 0
    var_2 = True
    var_3 = module_0.git_hook(var_2)
    assert var_3 == 1
    var_4 = True
    var_5 = module_0.git_hook(modify=var_4)
    var_6 = True
    var_7 = module_0.git_hook(lazy=var_6)
    var_8 = 'git'
    var_9 = 'diff-index'
    var_10 = '--name-only'
    var_11 = '--diff-filter=ACMRTUXB'
    var_12 = 'HEAD'
    var_13 = [var_8, var_9, var_10, var_11, var_12]
    var_14 = 'src'
    var_15 = [var_14]
    var_16 = module_0.git_hook(directories=var_15)
    var_17 = 'git'
    var_18 = 'diff-index'
    var_19 = '--cached'
    var_20 = '--name-only'
    var_21 = '--diff-filter=ACMRTUXB'
    var_22 = 'HEAD'
    var_23 = [var_17, var_18, var_19, var_20, var_21, var_22, var_14]
    var_24 = True
    var_25 = 'config.cfg'
    var_26 = module_0.git_hook(settings_file=var_25)
    var_27 = 'file.py'
    var_28 = module_1.abspath(var_27)
    var_29 = module_1.dirname(var_28)
    var_30 = module_0.git_hook()
    assert var_30 == 0



# Parsed testcases at query #4
#--------------------------


import isort.hooks as module_0
import posixpath as module_1

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = module_0.git_hook()
    assert var_1 == 0
    var_2 = module_0.git_hook()
    assert var_2 == 0
    var_3 = False
    var_4 = module_0.git_hook(var_3)
    assert var_4 == 0
    var_5 = True
    var_6 = module_0.git_hook(var_5)
    assert var_6 == 2
    var_7 = True
    var_8 = module_0.git_hook(modify=var_7)
    assert var_8 == 0
    var_9 = True
    var_10 = module_0.git_hook(lazy=var_9)
    var_11 = 'git'
    var_12 = 'diff-index'
    var_13 = '--name-only'
    var_14 = '--diff-filter=ACMRTUXB'
    var_15 = 'HEAD'
    var_16 = [var_11, var_12, var_13, var_14, var_15]
    var_17 = 'src/'
    var_18 = 'tests/'
    var_19 = [var_17, var_18]
    var_20 = module_0.git_hook(directories=var_19)
    var_21 = 'git'
    var_22 = 'diff-index'
    var_23 = '--cached'
    var_24 = '--name-only'
    var_25 = '--diff-filter=ACMRTUXB'
    var_26 = 'HEAD'
    var_27 = [var_21, var_22, var_23, var_24, var_25, var_26, var_17, var_18]
    var_28 = True
    var_29 = 'pyproject.toml'
    var_30 = module_0.git_hook(settings_file=var_29)
    var_31 = 'file1.py'
    var_32 = module_1.abspath(var_31)
    var_33 = module_1.dirname(var_32)



# Parsed testcases at query #5
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'echo'
    var_1 = '-e'
    var_2 = 'line1\nline2\nline3'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'line1'
    var_5 = 'line2'
    var_6 = 'line3'
    var_7 = [var_4, var_5, var_6]
    var_8 = module_0.get_lines(var_3)
    var_9 = 'single_line'
    var_10 = [var_0, var_9]
    var_11 = [var_9]
    var_12 = module_0.get_lines(var_10)
    var_13 = 'line1\n\nline2'
    var_14 = [var_0, var_1, var_13]
    var_15 = ''
    var_16 = [var_4, var_15, var_5]
    var_17 = module_0.get_lines(var_14)
    var_18 = '  line1  \n  line2  '
    var_19 = [var_0, var_1, var_18]
    var_20 = [var_4, var_5]
    var_21 = module_0.get_lines(var_19)



# Parsed testcases at query #6
#--------------------------


import isort.hooks as module_0
import posixpath as module_1

def test_case_0():
    var_0 = 'subprocess.run'
    var_1 = 'Mock'
    var_2 = ()
    var_3 = 'stdout'
    var_4 = b''
    var_5 = {var_3: var_4}
    var_6 = module_0.git_hook()
    assert var_6 == 0
    var_7 = ()
    var_8 = b'file1.py\nfile2.py'
    var_9 = {var_3: var_8}
    var_10 = ()
    var_11 = b'print("test")'
    var_12 = {var_3: var_11}
    var_13 = ()
    var_14 = {var_3: var_11}
    var_15 = 'isort.api.check_code_string'
    var_16 = False
    var_17 = module_0.git_hook(var_16)
    assert var_17 == 0
    var_18 = ()
    var_19 = {var_3: var_8}
    var_20 = ()
    var_21 = {var_3: var_11}
    var_22 = ()
    var_23 = {var_3: var_11}
    var_24 = True
    var_25 = module_0.git_hook(var_24)
    assert var_25 == 2
    var_26 = ()
    var_27 = b'file1.py'
    var_28 = {var_3: var_27}
    var_29 = ()
    var_30 = {var_3: var_11}
    var_31 = 'isort.api.sort_file'
    var_32 = module_0.git_hook(modify=var_24)
    var_33 = ()
    var_34 = {var_3: var_27}
    var_35 = ()
    var_36 = {var_3: var_11}
    var_37 = module_0.git_hook(lazy=var_24)
    var_38 = 'git'
    var_39 = 'diff-index'
    var_40 = '--name-only'
    var_41 = '--diff-filter=ACMRTUXB'
    var_42 = 'HEAD'
    var_43 = [var_38, var_39, var_40, var_41, var_42]
    var_44 = ()
    var_45 = {var_3: var_27}
    var_46 = ()
    var_47 = {var_3: var_11}
    var_48 = 'src/'
    var_49 = [var_48]
    var_50 = module_0.git_hook(directories=var_49)
    var_51 = '--cached'
    var_52 = [var_38, var_39, var_51, var_40, var_41, var_42, var_48]
    var_53 = ()
    var_54 = {var_3: var_27}
    var_55 = ()
    var_56 = {var_3: var_11}
    var_57 = '.isort.cfg'
    var_58 = module_0.git_hook(settings_file=var_57)
    var_59 = 'file1.py'
    var_60 = module_1.abspath(var_59)
    var_61 = module_1.dirname(var_60)



# Parsed testcases at query #7
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = True
    var_2 = module_0.git_hook(var_1)
    assert var_2 == 0
    var_3 = module_0.git_hook(modify=var_1)
    assert var_3 == 0
    var_4 = module_0.git_hook(lazy=var_1)
    assert var_4 == 0
    var_5 = ''
    var_6 = module_0.git_hook(settings_file=var_5)
    assert var_6 == 0
    var_7 = '.'
    var_8 = [var_7]
    var_9 = module_0.git_hook(directories=var_8)
    assert var_9 == 0
    var_10 = [var_7]
    var_11 = module_0.git_hook(var_1, var_1, var_1, var_5, var_10)
    assert var_11 == 0



# Parsed testcases at query #8
#--------------------------


import isort.hooks as module_0
import posixpath as module_1

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = False
    var_2 = module_0.git_hook(var_1)
    assert var_2 == 0
    var_3 = True
    var_4 = module_0.git_hook(var_3)
    assert var_4 == 2
    var_5 = True
    var_6 = module_0.git_hook(modify=var_5)
    assert var_6 == 0
    var_7 = True
    var_8 = module_0.git_hook(lazy=var_7)
    assert var_8 == 0
    var_9 = 'git'
    var_10 = 'diff-index'
    var_11 = '--name-only'
    var_12 = '--diff-filter=ACMRTUXB'
    var_13 = 'HEAD'
    var_14 = [var_9, var_10, var_11, var_12, var_13]
    var_15 = 'src/'
    var_16 = [var_15]
    var_17 = module_0.git_hook(directories=var_16)
    assert var_17 == 0
    var_18 = 'git'
    var_19 = 'diff-index'
    var_20 = '--cached'
    var_21 = '--name-only'
    var_22 = '--diff-filter=ACMRTUXB'
    var_23 = 'HEAD'
    var_24 = [var_18, var_19, var_20, var_21, var_22, var_23, var_15]
    var_25 = True
    var_26 = 'pyproject.toml'
    var_27 = module_0.git_hook(settings_file=var_26)
    assert var_27 == 0
    var_28 = 'file1.py'
    var_29 = module_1.abspath(var_28)
    var_30 = module_1.dirname(var_29)



# Parsed testcases at query #9
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'subprocess.run'
    var_1 = []
    var_2 = b''
    var_3 = module_0.git_hook()
    assert var_3 == 0
    var_4 = []
    var_5 = b'file.txt\nfile.js'
    var_6 = module_0.git_hook()
    assert var_6 == 0
    var_7 = []
    var_8 = b'file.py\nfile2.py'
    var_9 = []
    var_10 = b'print("hello")'
    var_11 = []
    var_12 = b'print("world")'
    var_13 = 'isort.api.check_code_string'
    var_14 = True
    var_15 = module_0.git_hook()
    assert var_15 == 0
    var_16 = []
    var_17 = []
    var_18 = []
    var_19 = False
    var_20 = module_0.git_hook()
    assert var_20 == 0
    var_21 = []
    var_22 = []
    var_23 = []
    var_24 = module_0.git_hook(var_14)
    assert var_24 == 2
    var_25 = []
    var_26 = []
    var_27 = []
    var_28 = 'isort.api.sort_file'
    var_29 = module_0.git_hook(modify=var_14)
    var_30 = []
    var_31 = b'file.py'
    var_32 = module_0.git_hook(lazy=var_14)
    var_33 = 'git'
    var_34 = 'diff-index'
    var_35 = '--name-only'
    var_36 = '--diff-filter=ACMRTUXB'
    var_37 = 'HEAD'
    var_38 = [var_33, var_34, var_35, var_36, var_37]
    var_39 = []
    var_40 = 'src/'
    var_41 = [var_40]
    var_42 = module_0.git_hook(directories=var_41)
    var_43 = '--cached'
    var_44 = [var_33, var_34, var_43, var_35, var_36, var_37, var_40]



# Parsed testcases at query #10
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = False
    var_2 = module_0.git_hook(var_1)
    assert var_2 == 0
    var_3 = True
    var_4 = module_0.git_hook(var_3)
    assert var_4 == 0
    var_5 = module_0.git_hook(modify=var_3)
    assert var_5 == 0
    var_6 = module_0.git_hook(lazy=var_3)
    assert var_6 == 0
    var_7 = ''
    var_8 = module_0.git_hook(settings_file=var_7)
    assert var_8 == 0
    var_9 = 'src/'
    var_10 = [var_9]
    var_11 = module_0.git_hook(directories=var_10)
    assert var_11 == 0
    var_12 = [var_9]
    var_13 = module_0.git_hook(var_3, var_3, var_3, var_7, var_12)
    assert var_13 == 0



# Parsed testcases at query #11
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = module_0.git_hook()
    assert var_1 == 0
    var_2 = True
    var_3 = module_0.git_hook(var_2)
    assert var_3 == 1
    var_4 = True
    var_5 = module_0.git_hook(modify=var_4)
    var_6 = True
    var_7 = module_0.git_hook(lazy=var_6)
    var_8 = 'git'
    var_9 = 'diff-index'
    var_10 = '--name-only'
    var_11 = '--diff-filter=ACMRTUXB'
    var_12 = 'HEAD'
    var_13 = [var_8, var_9, var_10, var_11, var_12]
    var_14 = 'src/'
    var_15 = [var_14]
    var_16 = module_0.git_hook(directories=var_15)
    var_17 = 'git'
    var_18 = 'diff-index'
    var_19 = '--cached'
    var_20 = '--name-only'
    var_21 = '--diff-filter=ACMRTUXB'
    var_22 = 'HEAD'
    var_23 = [var_17, var_18, var_19, var_20, var_21, var_22, var_14]
    var_24 = True



# Parsed testcases at query #12
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = 'file.txt'
    var_2 = 'file.md'
    var_3 = module_0.git_hook()
    assert var_3 == 0
    var_4 = 'file.py'
    var_5 = module_0.git_hook()
    assert var_5 == 0
    var_6 = True
    var_7 = module_0.git_hook(var_6)
    assert var_7 == 1
    var_8 = module_0.git_hook(modify=var_6)
    assert var_8 == 0
    var_9 = module_0.git_hook(lazy=var_6)
    assert var_9 == 0
    var_10 = 'pyproject.toml'
    var_11 = module_0.git_hook(settings_file=var_10)
    assert var_11 == 0
    var_12 = 'dir1/file.py'
    var_13 = 'dir2/file.py'
    var_14 = 'dir1'
    var_15 = [var_14]
    var_16 = module_0.git_hook(directories=var_15)
    assert var_16 == 0



# Parsed testcases at query #13
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = b'file1.py\nfile2.py'
    var_2 = b"print('hello')"
    var_3 = module_0.git_hook()
    assert var_3 == 0
    var_4 = b'file1.py\nfile2.py'
    var_5 = b"print('hello')"
    var_6 = True
    var_7 = module_0.git_hook(var_6)
    assert var_7 == 1
    var_8 = b'file1.py\nfile2.py'
    var_9 = b"print('hello')"
    var_10 = True
    var_11 = module_0.git_hook(modify=var_10)
    assert var_11 == 0
    var_12 = b'file1.py\nfile2.py'
    var_13 = b"print('hello')"
    var_14 = True
    var_15 = module_0.git_hook(lazy=var_14)
    assert var_15 == 0
    var_16 = 'git'
    var_17 = 'diff-index'
    var_18 = '--name-only'
    var_19 = '--diff-filter=ACMRTUXB'
    var_20 = 'HEAD'
    var_21 = [var_16, var_17, var_18, var_19, var_20]
    var_22 = b'src/file1.py\nsrc/file2.py'
    var_23 = b"print('hello')"
    var_24 = 'src/'
    var_25 = [var_24]
    var_26 = module_0.git_hook(directories=var_25)
    assert var_26 == 0
    var_27 = 'git'
    var_28 = 'diff-index'
    var_29 = '--cached'
    var_30 = '--name-only'
    var_31 = '--diff-filter=ACMRTUXB'
    var_32 = 'HEAD'
    var_33 = [var_27, var_28, var_29, var_30, var_31, var_32, var_24]
    var_34 = True
    var_35 = b'file1.py\nfile2.py'
    var_36 = b"print('hello')"
    var_37 = module_0.git_hook()
    assert var_37 == 0



# Parsed testcases at query #14
#--------------------------


import isort.hooks as module_0
import posixpath as module_1

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = module_0.git_hook()
    assert var_1 == 0
    var_2 = True
    var_3 = module_0.git_hook(var_2)
    assert var_3 == 1
    var_4 = True
    var_5 = module_0.git_hook(modify=var_4)
    var_6 = True
    var_7 = module_0.git_hook(lazy=var_6)
    var_8 = 'git'
    var_9 = 'diff-index'
    var_10 = '--name-only'
    var_11 = '--diff-filter=ACMRTUXB'
    var_12 = 'HEAD'
    var_13 = [var_8, var_9, var_10, var_11, var_12]
    var_14 = 'src/'
    var_15 = [var_14]
    var_16 = module_0.git_hook(directories=var_15)
    var_17 = 'git'
    var_18 = 'diff-index'
    var_19 = '--cached'
    var_20 = '--name-only'
    var_21 = '--diff-filter=ACMRTUXB'
    var_22 = 'HEAD'
    var_23 = [var_17, var_18, var_19, var_20, var_21, var_22, var_14]
    var_24 = True
    var_25 = '.isort.cfg'
    var_26 = module_0.git_hook(settings_file=var_25)
    var_27 = 'file1.py'
    var_28 = module_1.abspath(var_27)
    var_29 = module_1.dirname(var_28)



# Parsed testcases at query #15
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = module_0.git_hook()
    assert var_1 == 0
    var_2 = module_0.git_hook()
    assert var_2 == 0
    var_3 = True
    var_4 = module_0.git_hook(var_3)
    assert var_4 == 2
    var_5 = True
    var_6 = module_0.git_hook(modify=var_5)
    assert var_6 == 0
    var_7 = True
    var_8 = module_0.git_hook(var_7, var_7)
    assert var_8 == 2
    var_9 = module_0.git_hook()
    assert var_9 == 0



# Parsed testcases at query #16
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0



# Parsed testcases at query #17
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = module_0.git_hook()
    assert var_1 == 0
    var_2 = module_0.git_hook()
    assert var_2 == 0
    var_3 = True
    var_4 = module_0.git_hook(var_3)
    assert var_4 == 2
    var_5 = True
    var_6 = module_0.git_hook(modify=var_5)
    assert var_6 == 0
    var_7 = True
    var_8 = module_0.git_hook(lazy=var_7)
    assert var_8 == 0
    var_9 = 'dir1'
    var_10 = [var_9]
    var_11 = module_0.git_hook(directories=var_10)
    assert var_11 == 0
    var_12 = 'pyproject.toml'
    var_13 = module_0.git_hook(settings_file=var_12)
    assert var_13 == 0



# Parsed testcases at query #18
#--------------------------


import isort.hooks as module_0
import posixpath as module_1

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = module_0.git_hook()
    assert var_1 == 0
    var_2 = True
    var_3 = module_0.git_hook(var_2)
    assert var_3 == 2
    var_4 = True
    var_5 = module_0.git_hook(modify=var_4)
    assert var_5 == 0
    var_6 = True
    var_7 = module_0.git_hook(lazy=var_6)
    var_8 = 'git'
    var_9 = 'diff-index'
    var_10 = '--name-only'
    var_11 = '--diff-filter=ACMRTUXB'
    var_12 = 'HEAD'
    var_13 = [var_8, var_9, var_10, var_11, var_12]
    var_14 = 'src/'
    var_15 = 'tests/'
    var_16 = [var_14, var_15]
    var_17 = module_0.git_hook(directories=var_16)
    var_18 = 'git'
    var_19 = 'diff-index'
    var_20 = '--cached'
    var_21 = '--name-only'
    var_22 = '--diff-filter=ACMRTUXB'
    var_23 = 'HEAD'
    var_24 = [var_18, var_19, var_20, var_21, var_22, var_23, var_14, var_15]
    var_25 = True
    var_26 = 'pyproject.toml'
    var_27 = module_0.git_hook(settings_file=var_26)
    var_28 = 'file1.py'
    var_29 = module_1.abspath(var_28)
    var_30 = module_1.dirname(var_29)
    var_31 = module_0.git_hook()
    assert var_31 == 0



# Parsed testcases at query #19
#--------------------------


import isort.hooks as module_0
import posixpath as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(var_0)
    assert var_1 == 0
    var_2 = False
    var_3 = module_0.git_hook(var_2)
    assert var_3 == 0
    var_4 = True
    var_5 = module_0.git_hook(var_4)
    assert var_5 == 0
    var_6 = False
    var_7 = module_0.git_hook(var_6)
    assert var_7 == 0
    var_8 = True
    var_9 = module_0.git_hook(var_8)
    assert var_9 == 0
    var_10 = False
    var_11 = module_0.git_hook(var_10)
    assert var_11 == 0
    var_12 = False
    var_13 = module_0.git_hook(var_12)
    assert var_13 == 0
    var_14 = True
    var_15 = module_0.git_hook(var_14)
    assert var_15 == 2
    var_16 = True
    var_17 = module_0.git_hook(var_16, var_16)
    assert var_17 == 2
    var_18 = True
    var_19 = module_0.git_hook(lazy=var_18)
    var_20 = 'git'
    var_21 = 'diff-index'
    var_22 = '--name-only'
    var_23 = '--diff-filter=ACMRTUXB'
    var_24 = 'HEAD'
    var_25 = [var_20, var_21, var_22, var_23, var_24]
    var_26 = 'src/'
    var_27 = 'tests/'
    var_28 = [var_26, var_27]
    var_29 = module_0.git_hook(directories=var_28)
    var_30 = 'git'
    var_31 = 'diff-index'
    var_32 = '--cached'
    var_33 = '--name-only'
    var_34 = '--diff-filter=ACMRTUXB'
    var_35 = 'HEAD'
    var_36 = [var_30, var_31, var_32, var_33, var_34, var_35, var_26, var_27]
    var_37 = True
    var_38 = '.isort.cfg'
    var_39 = module_0.git_hook(settings_file=var_38)
    var_40 = 'file1.py'
    var_41 = module_1.abspath(var_40)
    var_42 = module_1.dirname(var_41)
    var_43 = True
    var_44 = module_0.git_hook(var_43)
    assert var_44 == 0



# Parsed testcases at query #20
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = True
    var_2 = module_0.git_hook(var_1)
    assert var_2 == 0
    var_3 = module_0.git_hook(modify=var_1)
    assert var_3 == 0
    var_4 = module_0.git_hook(lazy=var_1)
    assert var_4 == 0
    var_5 = ''
    var_6 = module_0.git_hook(settings_file=var_5)
    assert var_6 == 0
    var_7 = 'src/'
    var_8 = [var_7]
    var_9 = module_0.git_hook(directories=var_8)
    assert var_9 == 0
    var_10 = [var_7]
    var_11 = module_0.git_hook(var_1, var_1, var_1, var_5, var_10)
    assert var_11 == 0



# Parsed testcases at query #21
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'subprocess.run'
    var_1 = module_0.git_hook()
    assert var_1 == 0
    var_2 = module_0.git_hook()
    assert var_2 == 0
    var_3 = 'isort.api.check_code_string'
    var_4 = True
    var_5 = module_0.git_hook()
    assert var_5 == 0
    var_6 = False
    var_7 = module_0.git_hook()
    assert var_7 == 0
    var_8 = module_0.git_hook(var_4)
    assert var_8 == 2
    var_9 = 'isort.api.sort_file'
    var_10 = module_0.git_hook(modify=var_4)
    var_11 = module_0.git_hook(lazy=var_4)
    var_12 = 'git'
    var_13 = 'diff-index'
    var_14 = '--name-only'
    var_15 = '--diff-filter=ACMRTUXB'
    var_16 = 'HEAD'
    var_17 = [var_12, var_13, var_14, var_15, var_16]
    var_18 = 'src'
    var_19 = [var_18]
    var_20 = module_0.git_hook(directories=var_19)
    var_21 = '--cached'
    var_22 = [var_12, var_13, var_21, var_14, var_15, var_16, var_18]
    var_23 = module_0.git_hook()
    assert var_23 == 0



# Parsed testcases at query #22
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'import os\nimport sys\n'
    var_2 = 'run'
    var_3 = 'check_code_string'
    var_4 = True
    var_5 = module_0.git_hook(var_4)
    assert var_5 == 1
    var_6 = False
    var_7 = module_0.git_hook(var_6)
    assert var_7 == 0
    var_8 = 'sort_file'
    var_9 = module_0.git_hook(modify=var_4)
    var_10 = module_0.git_hook()
    assert var_10 == 0
    var_11 = 'src/'
    var_12 = [var_11]
    var_13 = module_0.git_hook(directories=var_12)



# Parsed testcases at query #23
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0



# Parsed testcases at query #24
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = True
    var_2 = module_0.git_hook(var_1)
    assert var_2 == 0
    var_3 = module_0.git_hook(modify=var_1)
    assert var_3 == 0
    var_4 = module_0.git_hook(lazy=var_1)
    assert var_4 == 0
    var_5 = ''
    var_6 = module_0.git_hook(settings_file=var_5)
    assert var_6 == 0
    var_7 = 'src/'
    var_8 = [var_7]
    var_9 = module_0.git_hook(directories=var_8)
    assert var_9 == 0



# Parsed testcases at query #25
#--------------------------


import isort.hooks as module_0
import posixpath as module_1

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = module_0.git_hook()
    assert var_1 == 0
    var_2 = True
    var_3 = module_0.git_hook(var_2)
    assert var_3 == 2
    var_4 = True
    var_5 = module_0.git_hook(modify=var_4)
    assert var_5 == 0
    var_6 = True
    var_7 = module_0.git_hook(lazy=var_6)
    assert var_7 == 0
    var_8 = 'src/'
    var_9 = [var_8]
    var_10 = module_0.git_hook(directories=var_9)
    assert var_10 == 0
    var_11 = '.isort.cfg'
    var_12 = module_0.git_hook(settings_file=var_11)
    var_13 = 'file1.py'
    var_14 = module_1.abspath(var_13)
    var_15 = module_1.dirname(var_14)



# Parsed testcases at query #26
#--------------------------


import isort.hooks as module_0
import posixpath as module_1

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = module_0.git_hook()
    assert var_1 == 0
    var_2 = module_0.git_hook()
    assert var_2 == 0
    var_3 = module_0.git_hook()
    assert var_3 == 0
    var_4 = True
    var_5 = module_0.git_hook(var_4)
    assert var_5 == 1
    var_6 = True
    var_7 = module_0.git_hook(modify=var_6)
    var_8 = True
    var_9 = module_0.git_hook(lazy=var_8)
    var_10 = 'git'
    var_11 = 'diff-index'
    var_12 = '--name-only'
    var_13 = '--diff-filter=ACMRTUXB'
    var_14 = 'HEAD'
    var_15 = [var_10, var_11, var_12, var_13, var_14]
    var_16 = 'src'
    var_17 = [var_16]
    var_18 = module_0.git_hook(directories=var_17)
    var_19 = 'git'
    var_20 = 'diff-index'
    var_21 = '--cached'
    var_22 = '--name-only'
    var_23 = '--diff-filter=ACMRTUXB'
    var_24 = 'HEAD'
    var_25 = [var_19, var_20, var_21, var_22, var_23, var_24, var_16]
    var_26 = True
    var_27 = 'setup.cfg'
    var_28 = module_0.git_hook(settings_file=var_27)
    var_29 = 'file.py'
    var_30 = module_1.abspath(var_29)
    var_31 = module_1.dirname(var_30)



# Parsed testcases at query #27
#--------------------------


import isort.hooks as module_0
import posixpath as module_1

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(var_0)
    assert var_1 == 0
    var_2 = 'git'
    var_3 = 'diff-index'
    var_4 = '--cached'
    var_5 = '--name-only'
    var_6 = '--diff-filter=ACMRTUXB'
    var_7 = 'HEAD'
    var_8 = [var_2, var_3, var_4, var_5, var_6, var_7]
    var_9 = True
    var_10 = module_0.git_hook(var_9)
    assert var_10 == 0
    var_11 = 'git'
    var_12 = 'diff-index'
    var_13 = '--cached'
    var_14 = '--name-only'
    var_15 = '--diff-filter=ACMRTUXB'
    var_16 = 'HEAD'
    var_17 = [var_11, var_12, var_13, var_14, var_15, var_16]
    var_18 = True
    var_19 = module_0.git_hook(var_18)
    assert var_19 == 0
    var_20 = True
    var_21 = module_0.git_hook(var_20)
    assert var_21 == 2
    var_22 = False
    var_23 = module_0.git_hook(var_22)
    assert var_23 == 0
    var_24 = True
    var_25 = module_0.git_hook(var_24, var_24)
    var_26 = True
    var_27 = module_0.git_hook(lazy=var_26)
    var_28 = 'git'
    var_29 = 'diff-index'
    var_30 = '--name-only'
    var_31 = '--diff-filter=ACMRTUXB'
    var_32 = 'HEAD'
    var_33 = [var_28, var_29, var_30, var_31, var_32]
    var_34 = 'src/'
    var_35 = [var_34]
    var_36 = module_0.git_hook(directories=var_35)
    var_37 = 'git'
    var_38 = 'diff-index'
    var_39 = '--cached'
    var_40 = '--name-only'
    var_41 = '--diff-filter=ACMRTUXB'
    var_42 = 'HEAD'
    var_43 = [var_37, var_38, var_39, var_40, var_41, var_42, var_34]
    var_44 = True
    var_45 = '.isort.cfg'
    var_46 = module_0.git_hook(settings_file=var_45)
    var_47 = 'file1.py'
    var_48 = module_1.abspath(var_47)
    var_49 = module_1.dirname(var_48)



# Parsed testcases at query #28
#--------------------------


import isort.hooks as module_0
import posixpath as module_1

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = module_0.git_hook()
    assert var_1 == 0
    var_2 = module_0.git_hook()
    assert var_2 == 0
    var_3 = True
    var_4 = module_0.git_hook(var_3)
    assert var_4 == 2
    var_5 = True
    var_6 = module_0.git_hook(modify=var_5)
    var_7 = True
    var_8 = module_0.git_hook(lazy=var_7)
    var_9 = 'src/'
    var_10 = [var_9]
    var_11 = module_0.git_hook(directories=var_10)
    var_12 = 'setup.cfg'
    var_13 = module_0.git_hook(settings_file=var_12)
    var_14 = 'file1.py'
    var_15 = module_1.abspath(var_14)
    var_16 = module_1.dirname(var_15)
    var_17 = module_0.git_hook()
    assert var_17 == 0



# Parsed testcases at query #29
#--------------------------


import isort.hooks as module_0
import posixpath as module_1

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = True
    var_2 = module_0.git_hook(var_1)
    assert var_2 == 1
    var_3 = True
    var_4 = module_0.git_hook(var_3)
    assert var_4 == 0
    var_5 = True
    var_6 = module_0.git_hook(var_5, var_5)
    var_7 = True
    var_8 = module_0.git_hook(lazy=var_7)
    var_9 = 'git'
    var_10 = 'diff-index'
    var_11 = '--name-only'
    var_12 = '--diff-filter=ACMRTUXB'
    var_13 = 'HEAD'
    var_14 = [var_9, var_10, var_11, var_12, var_13]
    var_15 = 'src/'
    var_16 = [var_15]
    var_17 = module_0.git_hook(directories=var_16)
    var_18 = '--cached'
    var_19 = [var_9, var_10, var_18, var_11, var_12, var_13, var_15]
    var_20 = 'pyproject.toml'
    var_21 = module_0.git_hook(settings_file=var_20)
    var_22 = 'test.py'
    var_23 = module_1.abspath(var_22)
    var_24 = module_1.dirname(var_23)
    var_25 = True
    var_26 = module_0.git_hook(var_25)
    assert var_26 == 0



# Parsed testcases at query #30
#--------------------------


import isort.hooks as module_0
import posixpath as module_1

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = module_0.git_hook()
    assert var_1 == 0
    var_2 = module_0.git_hook()
    assert var_2 == 0
    var_3 = True
    var_4 = module_0.git_hook(var_3)
    assert var_4 == 2
    var_5 = True
    var_6 = module_0.git_hook(modify=var_5)
    var_7 = True
    var_8 = module_0.git_hook(lazy=var_7)
    var_9 = 'git'
    var_10 = 'diff-index'
    var_11 = '--name-only'
    var_12 = '--diff-filter=ACMRTUXB'
    var_13 = 'HEAD'
    var_14 = [var_9, var_10, var_11, var_12, var_13]
    var_15 = 'src'
    var_16 = [var_15]
    var_17 = module_0.git_hook(directories=var_16)
    var_18 = 'git'
    var_19 = 'diff-index'
    var_20 = '--cached'
    var_21 = '--name-only'
    var_22 = '--diff-filter=ACMRTUXB'
    var_23 = 'HEAD'
    var_24 = [var_18, var_19, var_20, var_21, var_22, var_23, var_15]
    var_25 = True
    var_26 = 'pyproject.toml'
    var_27 = module_0.git_hook(settings_file=var_26)
    var_28 = 'file1.py'
    var_29 = module_1.abspath(var_28)
    var_30 = module_1.dirname(var_29)



