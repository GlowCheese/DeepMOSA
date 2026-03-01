####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
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



# Parsed testcases at query #2
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
    var_7 = module_0.git_hook(var_6, var_6)
    assert var_7 == 2
    var_8 = True
    var_9 = module_0.git_hook(lazy=var_8)
    var_10 = 'git'
    var_11 = 'diff-index'
    var_12 = '--name-only'
    var_13 = '--diff-filter=ACMRTUXB'
    var_14 = 'HEAD'
    var_15 = [var_10, var_11, var_12, var_13, var_14]
    var_16 = 'src/'
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
    var_29 = 'file1.py'
    var_30 = module_1.abspath(var_29)
    var_31 = module_1.dirname(var_30)



# Parsed testcases at query #3
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
    var_12 = True
    var_13 = module_0.git_hook(lazy=var_12)
    var_14 = 'git'
    var_15 = 'diff-index'
    var_16 = '--name-only'
    var_17 = '--diff-filter=ACMRTUXB'
    var_18 = 'HEAD'
    var_19 = [var_14, var_15, var_16, var_17, var_18]
    var_20 = 'src'
    var_21 = [var_20]
    var_22 = module_0.git_hook(directories=var_21)
    var_23 = 'git'
    var_24 = 'diff-index'
    var_25 = '--cached'
    var_26 = '--name-only'
    var_27 = '--diff-filter=ACMRTUXB'
    var_28 = 'HEAD'
    var_29 = [var_23, var_24, var_25, var_26, var_27, var_28, var_20]
    var_30 = True
    var_31 = b'file1.txt\nfile2.py'
    var_32 = b"print('hello')"
    var_33 = module_0.git_hook()
    assert var_33 == 0
    var_34 = b'file1.py'
    var_35 = b"print('hello')"
    var_36 = module_0.git_hook()
    assert var_36 == 0



# Parsed testcases at query #4
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



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
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
    var_13 = '-n'
    var_14 = [var_0, var_13]
    var_15 = []
    var_16 = module_0.get_lines(var_14)
    var_17 = '  line1  \n  line2  \n  line3  '
    var_18 = [var_0, var_1, var_17]
    var_19 = [var_4, var_5, var_6]
    var_20 = module_0.get_lines(var_18)



# Parsed testcases at query #2
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
    var_8 = 'single_line'
    var_9 = [var_0, var_8]
    var_10 = [var_8]
    var_11 = module_0.get_lines(var_9)
    var_12 = 'line1\n\nline2'
    var_13 = [var_0, var_12]
    var_14 = ''
    var_15 = [var_3, var_14, var_4]
    var_16 = module_0.get_lines(var_13)
    var_17 = '  line1  \n  line2  '
    var_18 = [var_0, var_17]
    var_19 = [var_3, var_4]
    var_20 = module_0.get_lines(var_18)



# Parsed testcases at query #3
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'line1\nline2\nline3'
    var_1 = 'git_hook.get_output'
    var_2 = 'dummy'
    var_3 = 'command'
    var_4 = [var_2, var_3]
    var_5 = module_0.get_lines(var_4)
    var_6 = [var_2, var_3]



# Parsed testcases at query #4
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'subprocess.run'
    var_1 = []
    var_2 = b''
    var_3 = module_0.git_hook()
    assert var_3 == 0
    var_4 = []
    var_5 = b'file.txt\nfile2.md'
    var_6 = module_0.git_hook()
    assert var_6 == 0
    var_7 = []
    var_8 = b'file.py\nfile2.py'
    var_9 = []
    var_10 = b'import os\nimport sys\n'
    var_11 = []
    var_12 = 'isort.api.check_code_string'
    var_13 = True
    var_14 = module_0.git_hook()
    assert var_14 == 0
    var_15 = []
    var_16 = []
    var_17 = b'import sys\nimport os\n'
    var_18 = []
    var_19 = False
    var_20 = module_0.git_hook()
    assert var_20 == 0
    var_21 = []
    var_22 = []
    var_23 = []
    var_24 = module_0.git_hook(var_13)
    assert var_24 == 2
    var_25 = []
    var_26 = []
    var_27 = []
    var_28 = 'isort.api.sort_file'
    var_29 = module_0.git_hook(modify=var_13)
    assert var_29 == 0
    var_30 = []
    var_31 = []
    var_32 = []
    var_33 = module_0.git_hook(lazy=var_13)
    assert var_33 == 0
    var_34 = []
    var_35 = []
    var_36 = []
    var_37 = 'src/'
    var_38 = [var_37]
    var_39 = module_0.git_hook(directories=var_38)
    assert var_39 == 0
    var_40 = []
    var_41 = []
    var_42 = []
    var_43 = '.isort.cfg'
    var_44 = module_0.git_hook(settings_file=var_43)
    assert var_44 == 0



# Parsed testcases at query #5
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
    var_7 = 'pyproject.toml'
    var_8 = module_0.git_hook(settings_file=var_7)
    assert var_8 == 0
    var_9 = 'src/'
    var_10 = [var_9]
    var_11 = module_0.git_hook(directories=var_10)
    assert var_11 == 0
    var_12 = [var_9]
    var_13 = module_0.git_hook(var_3, var_3, var_3, var_7, var_12)
    assert var_13 == 0



# Parsed testcases at query #6
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
    assert var_5 == 2
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
    var_16 = 'src/'
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
    var_27 = '.isort.cfg'
    var_28 = module_0.git_hook(settings_file=var_27)
    var_29 = 'file.py'
    var_30 = module_1.abspath(var_29)
    var_31 = module_1.dirname(var_30)



# Parsed testcases at query #7
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



# Parsed testcases at query #8
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
    var_5 = '.isort.cfg'
    var_6 = module_0.git_hook(settings_file=var_5)
    assert var_6 == 0
    var_7 = 'src/'
    var_8 = [var_7]
    var_9 = module_0.git_hook(directories=var_8)
    assert var_9 == 0
    var_10 = [var_7]
    var_11 = module_0.git_hook(var_1, var_1, var_1, var_5, var_10)
    assert var_11 == 0



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
    var_18 = [var_17]
    var_19 = module_0.git_hook(directories=var_18)
    var_20 = 'git'
    var_21 = 'diff-index'
    var_22 = '--cached'
    var_23 = '--name-only'
    var_24 = '--diff-filter=ACMRTUXB'
    var_25 = 'HEAD'
    var_26 = [var_20, var_21, var_22, var_23, var_24, var_25, var_17]
    var_27 = True
    var_28 = '.isort.cfg'
    var_29 = module_0.git_hook(settings_file=var_28)
    var_30 = 'file1.py'
    var_31 = module_1.abspath(var_30)
    var_32 = module_1.dirname(var_31)



# Parsed testcases at query #12
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'subprocess.run'
    var_1 = b''
    var_2 = module_0.git_hook()
    assert var_2 == 0
    var_3 = b'file1.py\nfile2.py'
    var_4 = b'print("test")'
    var_5 = 'isort.api.check_code_string'
    var_6 = False
    var_7 = module_0.git_hook(var_6)
    assert var_7 == 0
    var_8 = True
    var_9 = module_0.git_hook(var_8)
    assert var_9 == 2
    var_10 = 'isort.api.sort_file'
    var_11 = module_0.git_hook(modify=var_8)
    var_12 = module_0.git_hook(lazy=var_8)
    var_13 = 'src'
    var_14 = 'tests'
    var_15 = [var_13, var_14]
    var_16 = module_0.git_hook(directories=var_15)
    var_17 = module_0.git_hook()
    assert var_17 == 0



# Parsed testcases at query #13
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
    var_5 = module_0.git_hook(var_4, var_4)
    assert var_5 == 2
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
    var_25 = module_0.git_hook()
    assert var_25 == 0



# Parsed testcases at query #14
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'subprocess.run'
    var_1 = b''
    var_2 = 0
    var_3 = module_0.git_hook()
    assert var_3 == 0
    var_4 = b'file.txt\nfile.js'
    var_5 = module_0.git_hook()
    assert var_5 == 0
    var_6 = b'file.py'
    var_7 = b'import os\nimport sys\n'
    var_8 = 'isort.api.check_code_string'
    var_9 = True
    var_10 = module_0.git_hook()
    assert var_10 == 0
    var_11 = b'import sys\nimport os\n'
    var_12 = False
    var_13 = module_0.git_hook()
    assert var_13 == 0
    var_14 = False
    var_15 = module_0.git_hook(var_9)
    assert var_15 == 1
    var_16 = False
    var_17 = 'isort.api.sort_file'
    var_18 = module_0.git_hook(modify=var_9)
    var_19 = False
    var_20 = module_0.git_hook(lazy=var_9)
    var_21 = 'git'
    var_22 = 'diff-index'
    var_23 = '--name-only'
    var_24 = '--diff-filter=ACMRTUXB'
    var_25 = 'HEAD'
    var_26 = [var_21, var_22, var_23, var_24, var_25]
    var_27 = False
    var_28 = 'src/'
    var_29 = [var_28]
    var_30 = module_0.git_hook(directories=var_29)
    var_31 = '--cached'
    var_32 = [var_21, var_22, var_31, var_23, var_24, var_25, var_28]
    var_33 = False
    var_34 = '.isort.cfg'
    var_35 = module_0.git_hook(settings_file=var_34)



# Parsed testcases at query #15
#--------------------------


import isort.hooks as module_0

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
    var_6 = module_0.git_hook(var_4, var_4)
    assert var_6 == 2
    var_7 = True
    var_8 = module_0.git_hook(lazy=var_7)
    assert var_8 == 0
    var_9 = 'src/'
    var_10 = [var_9]
    var_11 = module_0.git_hook(directories=var_10)
    assert var_11 == 0
    var_12 = 'setup.cfg'
    var_13 = module_0.git_hook(settings_file=var_12)
    assert var_13 == 0
    var_14 = module_0.git_hook()
    assert var_14 == 0



# Parsed testcases at query #16
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = []
    var_2 = 0
    var_3 = b'file1.py\nfile2.py'
    var_4 = []
    var_5 = b'print("hello")'
    var_6 = []
    var_7 = b'print("world")'
    var_8 = module_0.git_hook()
    assert var_8 == 0
    var_9 = []
    var_10 = 0
    var_11 = b'file1.py\nfile2.py'
    var_12 = []
    var_13 = b'print("hello")'
    var_14 = []
    var_15 = b'print("world")'
    var_16 = True
    var_17 = module_0.git_hook(var_16)
    assert var_17 == 2
    var_18 = []
    var_19 = 0
    var_20 = b'file1.py\nfile2.py'
    var_21 = []
    var_22 = b'print("hello")'
    var_23 = []
    var_24 = b'print("world")'
    var_25 = False
    var_26 = module_0.git_hook(var_25)
    assert var_26 == 0
    var_27 = []
    var_28 = 0
    var_29 = b'file1.py\nfile2.py'
    var_30 = []
    var_31 = b'print("hello")'
    var_32 = []
    var_33 = b'print("world")'
    var_34 = True
    var_35 = module_0.git_hook(modify=var_34)
    var_36 = []
    var_37 = 0
    var_38 = b'file1.py\nfile2.py'
    var_39 = []
    var_40 = b'print("hello")'
    var_41 = []
    var_42 = b'print("world")'
    var_43 = True
    var_44 = module_0.git_hook(lazy=var_43)
    assert var_44 == 0
    var_45 = 'git'
    var_46 = 'diff-index'
    var_47 = '--name-only'
    var_48 = '--diff-filter=ACMRTUXB'
    var_49 = 'HEAD'
    var_50 = [var_45, var_46, var_47, var_48, var_49]
    var_51 = []
    var_52 = 0
    var_53 = b'file1.py\nfile2.py'
    var_54 = []
    var_55 = b'print("hello")'
    var_56 = []
    var_57 = b'print("world")'
    var_58 = 'src/'
    var_59 = [var_58]
    var_60 = module_0.git_hook(directories=var_59)
    assert var_60 == 0
    var_61 = 'git'
    var_62 = 'diff-index'
    var_63 = '--cached'
    var_64 = '--name-only'
    var_65 = '--diff-filter=ACMRTUXB'
    var_66 = 'HEAD'
    var_67 = [var_61, var_62, var_63, var_64, var_65, var_66, var_58]
    var_68 = True
    var_69 = []
    var_70 = 0
    var_71 = b'file1.py\nfile2.py'
    var_72 = []
    var_73 = b'print("hello")'
    var_74 = []
    var_75 = b'print("world")'
    var_76 = '.isort.cfg'
    var_77 = module_0.git_hook(settings_file=var_76)
    assert var_77 == 0



# Parsed testcases at query #17
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
    var_25 = 'pyproject.toml'
    var_26 = module_0.git_hook(settings_file=var_25)
    var_27 = 'file1.py'
    var_28 = module_1.abspath(var_27)
    var_29 = module_1.dirname(var_28)



# Parsed testcases at query #18
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



# Parsed testcases at query #19
#--------------------------


import isort.hooks as module_0

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
    var_37 = module_0.git_hook()
    assert var_37 == 0



# Parsed testcases at query #20
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = True
    var_2 = module_0.git_hook(var_1)
    assert var_2 == 1
    var_3 = False
    var_4 = module_0.git_hook(var_3)
    assert var_4 == 0
    var_5 = module_0.git_hook()
    assert var_5 == 0
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
    var_16 = 'src/'
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



# Parsed testcases at query #21
#--------------------------


import isort.hooks as module_0
import posixpath as module_1

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = True
    var_2 = module_0.git_hook(var_1)
    assert var_2 == 1
    var_3 = False
    var_4 = module_0.git_hook(var_3)
    assert var_4 == 0
    var_5 = True
    var_6 = module_0.git_hook(var_5)
    assert var_6 == 0
    var_7 = False
    var_8 = module_0.git_hook(var_7)
    assert var_8 == 0
    var_9 = True
    var_10 = module_0.git_hook(modify=var_9)
    var_11 = True
    var_12 = module_0.git_hook(lazy=var_11)
    var_13 = 'git'
    var_14 = 'diff-index'
    var_15 = '--name-only'
    var_16 = '--diff-filter=ACMRTUXB'
    var_17 = 'HEAD'
    var_18 = [var_13, var_14, var_15, var_16, var_17]
    var_19 = 'src/'
    var_20 = [var_19]
    var_21 = module_0.git_hook(directories=var_20)
    var_22 = 'git'
    var_23 = 'diff-index'
    var_24 = '--cached'
    var_25 = '--name-only'
    var_26 = '--diff-filter=ACMRTUXB'
    var_27 = 'HEAD'
    var_28 = [var_22, var_23, var_24, var_25, var_26, var_27, var_19]
    var_29 = True
    var_30 = 'pyproject.toml'
    var_31 = module_0.git_hook(settings_file=var_30)
    var_32 = 'test.py'
    var_33 = module_1.abspath(var_32)
    var_34 = module_1.dirname(var_33)



# Parsed testcases at query #22
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
    var_25 = module_0.git_hook()



# Parsed testcases at query #23
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
    var_25 = 'pyproject.toml'
    var_26 = module_0.git_hook(settings_file=var_25)
    var_27 = 'file1.py'
    var_28 = module_1.abspath(var_27)
    var_29 = module_1.dirname(var_28)
    var_30 = module_0.git_hook()
    assert var_30 == 0



# Parsed testcases at query #24
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



# Parsed testcases at query #25
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = 'sorted_file.py'
    var_2 = [var_1]
    var_3 = lambda cmd: var_2
    var_4 = 'import os\nimport sys\n'
    var_5 = lambda cmd: var_4
    var_6 = True
    var_7 = module_0.git_hook()
    assert var_7 == 0
    var_8 = 'unsorted_file.py'
    var_9 = [var_8]
    var_10 = lambda cmd: var_9
    var_11 = 'import sys\nimport os\n'
    var_12 = lambda cmd: var_11
    var_13 = False
    var_14 = module_0.git_hook()
    assert var_14 == 0
    var_15 = module_0.git_hook(var_6)
    assert var_15 == 1
    var_16 = None
    var_17 = module_0.git_hook(modify=var_6)
    var_18 = 'lazy_file.py'
    var_19 = [var_18]
    var_20 = lambda cmd: var_19
    var_21 = lambda cmd: var_11
    var_22 = module_0.git_hook(lazy=var_6)
    assert var_22 == 0
    var_23 = 'dir_file.py'
    var_24 = [var_23]
    var_25 = lambda cmd: var_24
    var_26 = lambda cmd: var_11
    var_27 = 'src/'
    var_28 = [var_27]
    var_29 = module_0.git_hook(directories=var_28)
    assert var_29 == 0
    var_30 = 'settings_file.py'
    var_31 = [var_30]
    var_32 = lambda cmd: var_31
    var_33 = lambda cmd: var_11
    var_34 = 'pyproject.toml'
    var_35 = module_0.git_hook(settings_file=var_34)
    assert var_35 == 0



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
    assert var_5 == 2
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
    var_16 = 'src/'
    var_17 = 'tests/'
    var_18 = [var_16, var_17]
    var_19 = module_0.git_hook(directories=var_18)
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
    var_30 = 'file.py'
    var_31 = module_1.abspath(var_30)
    var_32 = module_1.dirname(var_31)



# Parsed testcases at query #27
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



# Parsed testcases at query #28
#--------------------------


import isort.hooks as module_0
import posixpath as module_1
import isort.settings as module_2

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = 'test.py'
    var_2 = module_0.git_hook()
    assert var_2 == 0
    var_3 = module_0.git_hook()
    assert var_3 == 0
    var_4 = True
    var_5 = module_0.git_hook(var_4)
    assert var_5 == 1
    var_6 = module_0.git_hook(modify=var_4)
    var_7 = ''
    var_8 = module_1.abspath(var_1)
    var_9 = module_1.dirname(var_8)
    var_10 = module_2.Config(var_7, var_9)
    var_11 = module_0.git_hook(lazy=var_4)
    assert var_11 == 0
    var_12 = 'src/'
    var_13 = [var_12]
    var_14 = module_0.git_hook(directories=var_13)
    assert var_14 == 0
    var_15 = '.isort.cfg'
    var_16 = module_0.git_hook(settings_file=var_15)
    assert var_16 == 0



# Parsed testcases at query #29
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
    var_25 = module_0.git_hook()
    assert var_25 == 0



# Parsed testcases at query #30
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
    var_14 = -1
    var_15 = 'dir1'
    var_16 = 'dir2'
    var_17 = [var_15, var_16]
    var_18 = module_0.git_hook(directories=var_17)
    var_19 = 'git'
    var_20 = 'diff-index'
    var_21 = '--cached'
    var_22 = '--name-only'
    var_23 = '--diff-filter=ACMRTUXB'
    var_24 = 'HEAD'
    var_25 = [var_19, var_20, var_21, var_22, var_23, var_24, var_15, var_16]
    var_26 = True
    var_27 = -1
    var_28 = module_0.git_hook()
    assert var_28 == 0



# Parsed testcases at query #31
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



# Parsed testcases at query #32
#--------------------------


import isort.hooks as module_0
import posixpath as module_1

def test_case_0():
    var_0 = 'subprocess.run'
    var_1 = b''
    var_2 = module_0.git_hook()
    assert var_2 == 0
    var_3 = b'file1.py\nfile2.py'
    var_4 = b'print("test")'
    var_5 = 'isort.api.check_code_string'
    var_6 = False
    var_7 = module_0.git_hook()
    assert var_7 == 0
    var_8 = True
    var_9 = module_0.git_hook(var_8)
    assert var_9 == 2
    var_10 = 'isort.api.sort_file'
    var_11 = module_0.git_hook(modify=var_8)
    var_12 = module_0.git_hook(lazy=var_8)
    var_13 = 'git'
    var_14 = 'diff-index'
    var_15 = '--name-only'
    var_16 = '--diff-filter=ACMRTUXB'
    var_17 = 'HEAD'
    var_18 = [var_13, var_14, var_15, var_16, var_17]
    var_19 = 'src/'
    var_20 = [var_19]
    var_21 = module_0.git_hook(directories=var_20)
    var_22 = '--cached'
    var_23 = [var_13, var_14, var_22, var_15, var_16, var_17, var_19]
    var_24 = '.isort.cfg'
    var_25 = module_0.git_hook(settings_file=var_24)
    var_26 = 'file1.py'
    var_27 = module_1.abspath(var_26)
    var_28 = module_1.dirname(var_27)



# Parsed testcases at query #33
#--------------------------


import isort.hooks as module_0

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
    var_12 = 'file1.py'
    var_13 = b'file1.py\nfile2.py'
    var_14 = b'print("test")'
    var_15 = True
    var_16 = module_0.git_hook(lazy=var_15)
    assert var_16 == 0
    var_17 = 'git'
    var_18 = 'diff-index'
    var_19 = '--name-only'
    var_20 = '--diff-filter=ACMRTUXB'
    var_21 = 'HEAD'
    var_22 = [var_17, var_18, var_19, var_20, var_21]
    var_23 = b'src/file1.py\nsrc/file2.py'
    var_24 = b'print("test")'
    var_25 = 'src/'
    var_26 = [var_25]
    var_27 = module_0.git_hook(directories=var_26)
    assert var_27 == 0
    var_28 = 'git'
    var_29 = 'diff-index'
    var_30 = '--cached'
    var_31 = '--name-only'
    var_32 = '--diff-filter=ACMRTUXB'
    var_33 = 'HEAD'
    var_34 = [var_28, var_29, var_30, var_31, var_32, var_33, var_25]
    var_35 = True
    var_36 = b'file1.py\nfile2.py'
    var_37 = b'print("test")'
    var_38 = module_0.git_hook()
    assert var_38 == 0



# Parsed testcases at query #34
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
    var_10 = True
    var_11 = module_0.git_hook(var_10)
    assert var_11 == 2
    var_12 = True
    var_13 = module_0.git_hook(modify=var_12)
    assert var_13 == 0
    var_14 = True
    var_15 = module_0.git_hook(lazy=var_14)
    assert var_15 == 0
    var_16 = 'pyproject.toml'
    var_17 = module_0.git_hook(settings_file=var_16)
    assert var_17 == 0
    var_18 = 'src'
    var_19 = [var_18]
    var_20 = module_0.git_hook(directories=var_19)
    assert var_20 == 0
    var_21 = module_0.git_hook()
    assert var_21 == 0



# Parsed testcases at query #35
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



# Parsed testcases at query #36
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
    var_7 = 'pyproject.toml'
    var_8 = module_0.git_hook(settings_file=var_7)
    assert var_8 == 0
    var_9 = 'src/'
    var_10 = [var_9]
    var_11 = module_0.git_hook(directories=var_10)
    assert var_11 == 0



# Parsed testcases at query #37
#--------------------------


import isort.hooks as module_0
import isort.settings as module_1

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
    var_6 = module_0.git_hook()
    assert var_6 == 0
    var_7 = True
    var_8 = module_0.git_hook(var_7)
    assert var_8 == 1
    var_9 = module_0.git_hook(modify=var_7)
    var_10 = 'git'
    var_11 = 'diff-index'
    var_12 = '--name-only'
    var_13 = '--diff-filter=ACMRTUXB'
    var_14 = 'HEAD'
    var_15 = [var_10, var_11, var_12, var_13, var_14]
    var_16 = True
    var_17 = module_0.git_hook(lazy=var_16)
    var_18 = 'test_settings'
    var_19 = '/test/path'
    var_20 = module_1.Config(var_18, var_19)
    var_21 = 'test_settings'
    var_22 = module_0.git_hook(settings_file=var_21)
    var_23 = '/test/path'
    var_24 = '--cached'
    var_25 = 'dir1'
    var_26 = 'dir2'
    var_27 = [var_10, var_11, var_24, var_12, var_13, var_14, var_25, var_26]
    var_28 = 'dir1'
    var_29 = 'dir2'
    var_30 = [var_28, var_29]
    var_31 = module_0.git_hook(directories=var_30)
    var_32 = True



# Parsed testcases at query #38
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
    var_16 = 'src/'
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
    var_27 = '.isort.cfg'
    var_28 = module_0.git_hook(settings_file=var_27)
    var_29 = 'file.py'
    var_30 = module_1.abspath(var_29)
    var_31 = module_1.dirname(var_30)



# Parsed testcases at query #39
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = True
    var_2 = module_0.git_hook(var_1)
    assert var_2 == 0
    var_3 = module_0.git_hook(var_1)
    assert var_3 == 1
    var_4 = module_0.git_hook(modify=var_1)
    assert var_4 == 0
    var_5 = module_0.git_hook(lazy=var_1)
    assert var_5 == 0
    var_6 = 'src/'
    var_7 = [var_6]
    var_8 = module_0.git_hook(directories=var_7)
    assert var_8 == 0



# Parsed testcases at query #40
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
    var_26 = 'pyproject.toml'
    var_27 = module_0.git_hook(settings_file=var_26)
    var_28 = 'file1.py'
    var_29 = module_1.abspath(var_28)
    var_30 = module_1.dirname(var_29)



# Parsed testcases at query #41
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
    var_9 = 'src'
    var_10 = [var_9]
    var_11 = module_0.git_hook(directories=var_10)
    assert var_11 == 0
    var_12 = [var_9]
    var_13 = module_0.git_hook(var_3, var_3, var_3, var_7, var_12)
    assert var_13 == 0



# Parsed testcases at query #42
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



# Parsed testcases at query #43
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
    var_10 = 'dir1'
    var_11 = [var_10]
    var_12 = module_0.git_hook(directories=var_11)
    assert var_12 == 0



# Parsed testcases at query #44
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



# Parsed testcases at query #45
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
    assert var_7 == 0
    var_8 = True
    var_9 = module_0.git_hook(lazy=var_8)
    var_10 = 'git'
    var_11 = 'diff-index'
    var_12 = '--name-only'
    var_13 = '--diff-filter=ACMRTUXB'
    var_14 = 'HEAD'
    var_15 = [var_10, var_11, var_12, var_13, var_14]
    var_16 = 'src/'
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
    var_27 = 'pyproject.toml'
    var_28 = module_0.git_hook(settings_file=var_27)
    var_29 = 'file.py'
    var_30 = module_1.abspath(var_29)
    var_31 = module_1.dirname(var_30)



# Parsed testcases at query #46
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



# Parsed testcases at query #47
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



# Parsed testcases at query #48
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
    var_26 = module_0.git_hook()
    assert var_26 == 0



# Parsed testcases at query #49
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
    var_7 = True
    var_8 = module_0.git_hook(lazy=var_7)
    var_9 = 'src/'
    var_10 = [var_9]
    var_11 = module_0.git_hook(directories=var_10)
    var_12 = '.isort.cfg'
    var_13 = module_0.git_hook(settings_file=var_12)



# Parsed testcases at query #50
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
    var_25 = 'pyproject.toml'
    var_26 = module_0.git_hook(settings_file=var_25)
    var_27 = 'file1.py'
    var_28 = module_1.abspath(var_27)
    var_29 = module_1.dirname(var_28)
    var_30 = module_0.git_hook()
    assert var_30 == 0



# Parsed testcases at query #51
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
    var_25 = module_0.git_hook()



# Parsed testcases at query #52
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
    var_8 = []
    var_9 = 'isort.api.check_code_string'
    var_10 = False
    var_11 = module_0.git_hook()
    assert var_11 == 0
    var_12 = []
    var_13 = []
    var_14 = []
    var_15 = True
    var_16 = module_0.git_hook(var_15)
    assert var_16 == 2
    var_17 = []
    var_18 = []
    var_19 = []
    var_20 = 'isort.api.sort_file'
    var_21 = module_0.git_hook(modify=var_15)
    var_22 = []
    var_23 = module_0.git_hook(lazy=var_15)
    var_24 = 'git'
    var_25 = 'diff-index'
    var_26 = '--name-only'
    var_27 = '--diff-filter=ACMRTUXB'
    var_28 = 'HEAD'
    var_29 = [var_24, var_25, var_26, var_27, var_28]
    var_30 = []
    var_31 = 'src/'
    var_32 = [var_31]
    var_33 = module_0.git_hook(directories=var_32)
    var_34 = '--cached'
    var_35 = [var_24, var_25, var_34, var_26, var_27, var_28, var_31]
    var_36 = []
    var_37 = []
    var_38 = '.isort.cfg'
    var_39 = module_0.git_hook(settings_file=var_38)
    var_40 = []
    var_41 = []
    var_42 = module_0.git_hook()
    assert var_42 == 0



# Parsed testcases at query #53
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
    var_31 = True
    var_32 = module_0.git_hook(var_31, settings_file=var_29)
    assert var_32 == 2



# Parsed testcases at query #54
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
    assert var_7 == 0
    var_8 = 'file.py'
    var_9 = True
    var_10 = module_0.git_hook(lazy=var_9)
    var_11 = 'git'
    var_12 = 'diff-index'
    var_13 = '--name-only'
    var_14 = '--diff-filter=ACMRTUXB'
    var_15 = 'HEAD'
    var_16 = [var_11, var_12, var_13, var_14, var_15]
    var_17 = 'src/'
    var_18 = [var_17]
    var_19 = module_0.git_hook(directories=var_18)
    var_20 = 'git'
    var_21 = 'diff-index'
    var_22 = '--cached'
    var_23 = '--name-only'
    var_24 = '--diff-filter=ACMRTUXB'
    var_25 = 'HEAD'
    var_26 = [var_20, var_21, var_22, var_23, var_24, var_25, var_17]
    var_27 = True
    var_28 = '.isort.cfg'
    var_29 = module_0.git_hook(settings_file=var_28)
    var_30 = 'file.py'
    var_31 = module_1.abspath(var_30)
    var_32 = module_1.dirname(var_31)



# Parsed testcases at query #55
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'subprocess.run'
    var_1 = b''
    var_2 = module_0.git_hook()
    assert var_2 == 0
    var_3 = b'file.txt\nfile2.md'
    var_4 = module_0.git_hook()
    assert var_4 == 0
    var_5 = b'file.py\nfile2.py'
    var_6 = b'print("hello")'
    var_7 = b'print("world")'
    var_8 = 'isort.api.check_code_string'
    var_9 = True
    var_10 = module_0.git_hook()
    assert var_10 == 0
    var_11 = False
    var_12 = module_0.git_hook()
    assert var_12 == 0
    var_13 = module_0.git_hook(var_9)
    assert var_13 == 2
    var_14 = 'isort.api.sort_file'
    var_15 = module_0.git_hook(modify=var_9)
    var_16 = b'file.py'
    var_17 = module_0.git_hook()
    assert var_17 == 0
    var_18 = module_0.git_hook(lazy=var_9)
    var_19 = 'git'
    var_20 = 'diff-index'
    var_21 = '--name-only'
    var_22 = '--diff-filter=ACMRTUXB'
    var_23 = 'HEAD'
    var_24 = [var_19, var_20, var_21, var_22, var_23]
    var_25 = 'src/'
    var_26 = [var_25]
    var_27 = module_0.git_hook(directories=var_26)
    var_28 = '--cached'
    var_29 = [var_19, var_20, var_28, var_21, var_22, var_23, var_25]



# Parsed testcases at query #56
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
    assert var_5 == 2
    var_6 = True
    var_7 = module_0.git_hook(modify=var_6)
    assert var_7 == 0
    var_8 = True
    var_9 = module_0.git_hook(var_8, var_8)
    assert var_9 == 2
    var_10 = True
    var_11 = module_0.git_hook(lazy=var_10)
    assert var_11 == 0
    var_12 = 'git'
    var_13 = 'diff-index'
    var_14 = '--name-only'
    var_15 = '--diff-filter=ACMRTUXB'
    var_16 = 'HEAD'
    var_17 = [var_12, var_13, var_14, var_15, var_16]
    var_18 = 'src/'
    var_19 = [var_18]
    var_20 = module_0.git_hook(directories=var_19)
    assert var_20 == 0
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
    var_31 = 'file1.py'
    var_32 = module_1.abspath(var_31)
    var_33 = module_1.dirname(var_32)



# Parsed testcases at query #57
#--------------------------


import isort.hooks as module_0

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
    assert var_11 == 0
    var_12 = b'file1.py\nfile2.py'
    var_13 = b'print("test")'
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
    var_23 = b'print("test")'
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
    var_34 = b'file1.py\nfile2.py'
    var_35 = b'print("test")'
    var_36 = module_0.git_hook()
    assert var_36 == 0



# Parsed testcases at query #58
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
    assert var_4 == 1
    var_5 = True
    var_6 = module_0.git_hook(modify=var_5)
    assert var_6 == 0
    var_7 = 'file1.py'
    var_8 = True
    var_9 = module_0.git_hook(var_8, var_8)
    assert var_9 == 1
    var_10 = 'file1.py'
    var_11 = True
    var_12 = module_0.git_hook(lazy=var_11)
    var_13 = 'git'
    var_14 = 'diff-index'
    var_15 = '--name-only'
    var_16 = '--diff-filter=ACMRTUXB'
    var_17 = 'HEAD'
    var_18 = [var_13, var_14, var_15, var_16, var_17]
    var_19 = 'src/'
    var_20 = 'tests/'
    var_21 = [var_19, var_20]
    var_22 = module_0.git_hook(directories=var_21)
    var_23 = 'git'
    var_24 = 'diff-index'
    var_25 = '--cached'
    var_26 = '--name-only'
    var_27 = '--diff-filter=ACMRTUXB'
    var_28 = 'HEAD'
    var_29 = [var_23, var_24, var_25, var_26, var_27, var_28, var_19, var_20]
    var_30 = True
    var_31 = module_0.git_hook()
    assert var_31 == 0



# Parsed testcases at query #59
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



# Parsed testcases at query #60
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(var_0)
    assert var_1 == 0
    var_2 = False
    var_3 = module_0.git_hook(var_2)
    assert var_3 == 0
    var_4 = 'file1.py'
    var_5 = module_0.git_hook(var_0)
    assert var_5 == 0
    var_6 = module_0.git_hook(var_2)
    assert var_6 == 0
    var_7 = module_0.git_hook(var_0)
    assert var_7 == 1
    var_8 = module_0.git_hook(var_2)
    assert var_8 == 0
    var_9 = module_0.git_hook(var_0, var_0)
    var_10 = module_0.git_hook(var_0, lazy=var_0)
    assert var_10 == 0
    var_11 = module_0.git_hook(var_2, lazy=var_0)
    assert var_11 == 0
    var_12 = 'dir1'
    var_13 = [var_12]
    var_14 = module_0.git_hook(var_0, directories=var_13)
    assert var_14 == 0
    var_15 = [var_12]
    var_16 = module_0.git_hook(var_2, directories=var_15)
    assert var_16 == 0
    var_17 = 'settings.cfg'
    var_18 = module_0.git_hook(var_0, settings_file=var_17)
    assert var_18 == 0
    var_19 = module_0.git_hook(var_2, settings_file=var_17)
    assert var_19 == 0
    var_20 = module_0.git_hook(var_0)
    assert var_20 == 0
    var_21 = module_0.git_hook(var_2)
    assert var_21 == 0



# Parsed testcases at query #61
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
    assert var_5 == 2
    var_6 = True
    var_7 = module_0.git_hook(modify=var_6)
    assert var_7 == 0
    var_8 = True
    var_9 = module_0.git_hook(lazy=var_8)
    var_10 = 'git'
    var_11 = 'diff-index'
    var_12 = '--name-only'
    var_13 = '--diff-filter=ACMRTUXB'
    var_14 = 'HEAD'
    var_15 = [var_10, var_11, var_12, var_13, var_14]
    var_16 = 'src/'
    var_17 = 'tests/'
    var_18 = [var_16, var_17]
    var_19 = module_0.git_hook(directories=var_18)
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
    var_30 = 'file1.py'
    var_31 = module_1.abspath(var_30)
    var_32 = module_1.dirname(var_31)
    var_33 = module_0.git_hook()
    assert var_33 == 0



# Parsed testcases at query #62
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
    var_7 = 'pyproject.toml'
    var_8 = module_0.git_hook(settings_file=var_7)
    assert var_8 == 0
    var_9 = 'src/'
    var_10 = [var_9]
    var_11 = module_0.git_hook(directories=var_10)
    assert var_11 == 0



# Parsed testcases at query #63
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
    var_4 = 'file1.txt'
    var_5 = 'file2.md'
    var_6 = module_0.git_hook(var_0)
    assert var_6 == 0
    var_7 = module_0.git_hook(var_2)
    assert var_7 == 0
    var_8 = 'file1.py'
    var_9 = 'file2.py'
    var_10 = module_0.git_hook(var_0)
    assert var_10 == 0
    var_11 = module_0.git_hook(var_2)
    assert var_11 == 0
    var_12 = module_0.git_hook(var_2)
    assert var_12 == 0
    var_13 = module_0.git_hook(var_0)
    assert var_13 == 2
    var_14 = module_0.git_hook(var_0, var_0)
    var_15 = module_0.git_hook(lazy=var_0)
    var_16 = 'git'
    var_17 = 'diff-index'
    var_18 = '--name-only'
    var_19 = '--diff-filter=ACMRTUXB'
    var_20 = 'HEAD'
    var_21 = [var_16, var_17, var_18, var_19, var_20]
    var_22 = 'dir1'
    var_23 = 'dir2'
    var_24 = [var_22, var_23]
    var_25 = module_0.git_hook(directories=var_24)
    var_26 = '--cached'
    var_27 = [var_16, var_17, var_26, var_18, var_19, var_20, var_22, var_23]
    var_28 = 'settings.cfg'
    var_29 = module_0.git_hook(settings_file=var_28)
    var_30 = module_1.abspath(var_8)
    var_31 = module_1.dirname(var_30)
    var_32 = module_0.git_hook(var_0)
    assert var_32 == 0



# Parsed testcases at query #64
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



# Parsed testcases at query #65
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
    var_29 = 'pyproject.toml'
    var_30 = module_0.git_hook(settings_file=var_29)
    var_31 = 'file1.py'
    var_32 = module_1.abspath(var_31)
    var_33 = module_1.dirname(var_32)



# Parsed testcases at query #66
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = b'file1.py\nfile2.py'
    var_2 = b'print("hello")'
    var_3 = module_0.git_hook()
    assert var_3 == 0
    var_4 = b'file1.py\nfile2.py'
    var_5 = b'print("hello")'
    var_6 = True
    var_7 = module_0.git_hook(var_6)
    assert var_7 == 1
    var_8 = b'file1.py\nfile2.py'
    var_9 = b'print("hello")'
    var_10 = True
    var_11 = module_0.git_hook(modify=var_10)
    assert var_11 == 0
    var_12 = 'file1.py'
    var_13 = b'file1.py\nfile2.py'
    var_14 = b'print("hello")'
    var_15 = True
    var_16 = module_0.git_hook(lazy=var_15)
    assert var_16 == 0
    var_17 = 'git'
    var_18 = 'diff-index'
    var_19 = '--name-only'
    var_20 = '--diff-filter=ACMRTUXB'
    var_21 = 'HEAD'
    var_22 = [var_17, var_18, var_19, var_20, var_21]
    var_23 = b'file1.py\nfile2.py'
    var_24 = b'print("hello")'
    var_25 = 'src/'
    var_26 = [var_25]
    var_27 = module_0.git_hook(directories=var_26)
    assert var_27 == 0
    var_28 = 'git'
    var_29 = 'diff-index'
    var_30 = '--cached'
    var_31 = '--name-only'
    var_32 = '--diff-filter=ACMRTUXB'
    var_33 = 'HEAD'
    var_34 = [var_28, var_29, var_30, var_31, var_32, var_33, var_25]
    var_35 = True
    var_36 = b'file1.py\nfile2.py'
    var_37 = b'print("hello")'
    var_38 = module_0.git_hook()
    assert var_38 == 0



# Parsed testcases at query #67
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'subprocess.run'
    var_1 = b''
    var_2 = module_0.git_hook()
    assert var_2 == 0
    var_3 = b'file1.py\nfile2.py'
    var_4 = b'print("hello")'
    var_5 = b'print("world")'
    var_6 = 'isort.api.check_code_string'
    var_7 = True
    var_8 = module_0.git_hook()
    assert var_8 == 0
    var_9 = False
    var_10 = module_0.git_hook(var_7)
    assert var_10 == 2
    var_11 = module_0.git_hook(var_9)
    assert var_11 == 0
    var_12 = 'isort.api.sort_file'
    var_13 = module_0.git_hook(modify=var_7)
    var_14 = module_0.git_hook(lazy=var_7)
    assert var_14 == 0
    var_15 = 'src/'
    var_16 = [var_15]
    var_17 = module_0.git_hook(directories=var_16)
    assert var_17 == 0
    var_18 = '.isort.cfg'
    var_19 = module_0.git_hook(settings_file=var_18)
    assert var_19 == 0



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
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
    assert var_7 == 0
    var_8 = 'file.py'
    var_9 = True
    var_10 = module_0.git_hook(lazy=var_9)
    var_11 = 'git'
    var_12 = 'diff-index'
    var_13 = '--name-only'
    var_14 = '--diff-filter=ACMRTUXB'
    var_15 = 'HEAD'
    var_16 = [var_11, var_12, var_13, var_14, var_15]
    var_17 = 'src/'
    var_18 = [var_17]
    var_19 = module_0.git_hook(directories=var_18)
    var_20 = 'git'
    var_21 = 'diff-index'
    var_22 = '--cached'
    var_23 = '--name-only'
    var_24 = '--diff-filter=ACMRTUXB'
    var_25 = 'HEAD'
    var_26 = [var_20, var_21, var_22, var_23, var_24, var_25, var_17]
    var_27 = True
    var_28 = '.isort.cfg'
    var_29 = module_0.git_hook(settings_file=var_28)
    var_30 = 'file.py'
    var_31 = module_1.abspath(var_30)
    var_32 = module_1.dirname(var_31)



# Parsed testcases at query #2
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



# Parsed testcases at query #3
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
    var_8 = 'single_line'
    var_9 = [var_0, var_8]
    var_10 = [var_8]
    var_11 = module_0.get_lines(var_9)
    var_12 = 'line1\n\nline2'
    var_13 = [var_0, var_12]
    var_14 = ''
    var_15 = [var_3, var_14, var_4]
    var_16 = module_0.get_lines(var_13)
    var_17 = '  line1  \n  line2  '
    var_18 = [var_0, var_17]
    var_19 = [var_3, var_4]
    var_20 = module_0.get_lines(var_18)



# Parsed testcases at query #4
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
    var_15 = 'src/'
    var_16 = 'tests/'
    var_17 = [var_15, var_16]
    var_18 = module_0.git_hook(directories=var_17)
    var_19 = 'git'
    var_20 = 'diff-index'
    var_21 = '--cached'
    var_22 = '--name-only'
    var_23 = '--diff-filter=ACMRTUXB'
    var_24 = 'HEAD'
    var_25 = [var_19, var_20, var_21, var_22, var_23, var_24, var_15, var_16]
    var_26 = True
    var_27 = module_0.git_hook()
    assert var_27 == 0



# Parsed testcases at query #5
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
    var_19 = 'git'
    var_20 = 'diff-index'
    var_21 = '--cached'
    var_22 = '--name-only'
    var_23 = '--diff-filter=ACMRTUXB'
    var_24 = 'HEAD'
    var_25 = [var_19, var_20, var_21, var_22, var_23, var_24, var_15, var_16]
    var_26 = True
    var_27 = module_0.git_hook()
    assert var_27 == 0



# Parsed testcases at query #6
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'subprocess.run'
    var_1 = b''
    var_2 = module_0.git_hook()
    assert var_2 == 0
    var_3 = b'file1.py\nfile2.py'
    var_4 = b'print("test")'
    var_5 = 'isort.api.check_code_string'
    var_6 = False
    var_7 = module_0.git_hook()
    assert var_7 == 0
    var_8 = True
    var_9 = module_0.git_hook(var_8)
    assert var_9 == 1
    var_10 = 'isort.api.sort_file'
    var_11 = module_0.git_hook(modify=var_8)
    var_12 = module_0.git_hook(lazy=var_8)
    var_13 = 'git'
    var_14 = 'diff-index'
    var_15 = '--name-only'
    var_16 = '--diff-filter=ACMRTUXB'
    var_17 = 'HEAD'
    var_18 = [var_13, var_14, var_15, var_16, var_17]
    var_19 = 'src/'
    var_20 = [var_19]
    var_21 = module_0.git_hook(directories=var_20)
    var_22 = '--cached'
    var_23 = [var_13, var_14, var_22, var_15, var_16, var_17, var_19]
    var_24 = module_0.git_hook()
    assert var_24 == 0



# Parsed testcases at query #7
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



# Parsed testcases at query #8
#--------------------------


import isort.hooks as module_0
import posixpath as module_1
import isort.settings as module_2
import zipfile as module_3

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
    var_20 = False
    var_21 = module_0.git_hook(var_20)
    assert var_21 == 0
    var_22 = True
    var_23 = module_0.git_hook(var_22)
    assert var_23 == 2
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
    var_35 = 'tests/'
    var_36 = [var_34, var_35]
    var_37 = module_0.git_hook(directories=var_36)
    var_38 = 'git'
    var_39 = 'diff-index'
    var_40 = '--cached'
    var_41 = '--name-only'
    var_42 = '--diff-filter=ACMRTUXB'
    var_43 = 'HEAD'
    var_44 = [var_38, var_39, var_40, var_41, var_42, var_43, var_34, var_35]
    var_45 = True
    var_46 = 'setup.cfg'
    var_47 = module_0.git_hook(settings_file=var_46)
    var_48 = 'file1.py'
    var_49 = module_1.abspath(var_48)
    var_50 = module_1.dirname(var_49)
    var_51 = module_2.Config(var_46, var_50)
    var_52 = module_3.Path(var_48)
    var_53 = True
    var_54 = module_0.git_hook(var_53)
    assert var_54 == 0



# Parsed testcases at query #9
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = 'test.py'
    var_2 = [var_1]
    var_3 = lambda cmd: var_2
    var_4 = 'import os\nimport sys\n'
    var_5 = lambda cmd: var_4
    var_6 = module_0.git_hook()
    assert var_6 == 0
    var_7 = 'import sys\nimport os\n'
    var_8 = lambda cmd: var_7
    var_9 = True
    var_10 = module_0.git_hook(var_9)
    assert var_10 == 1
    var_11 = lambda cmd: var_7
    var_12 = module_0.git_hook(modify=var_9)
    var_13 = '--cached'
    var_14 = [var_1]
    var_15 = []
    var_16 = lambda cmd: var_14 if var_13 not in cmd else var_15
    var_17 = module_0.git_hook(lazy=var_9)
    assert var_17 == 0
    var_18 = 'src'
    var_19 = [var_1]
    var_20 = []
    var_21 = lambda cmd: var_19 if var_18 in cmd else var_20
    var_22 = [var_18]
    var_23 = module_0.git_hook(directories=var_22)
    assert var_23 == 0



# Parsed testcases at query #10
#--------------------------


import isort.hooks as module_0

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
    assert var_9 == 2
    var_10 = False
    var_11 = module_0.git_hook(var_10)
    assert var_11 == 0
    var_12 = True
    var_13 = module_0.git_hook(var_12, var_12)
    assert var_13 == 2
    var_14 = True
    var_15 = module_0.git_hook(lazy=var_14)
    var_16 = 'git'
    var_17 = 'diff-index'
    var_18 = '--name-only'
    var_19 = '--diff-filter=ACMRTUXB'
    var_20 = 'HEAD'
    var_21 = [var_16, var_17, var_18, var_19, var_20]
    var_22 = 'src/'
    var_23 = [var_22]
    var_24 = module_0.git_hook(directories=var_23)
    var_25 = 'git'
    var_26 = 'diff-index'
    var_27 = '--cached'
    var_28 = '--name-only'
    var_29 = '--diff-filter=ACMRTUXB'
    var_30 = 'HEAD'
    var_31 = [var_25, var_26, var_27, var_28, var_29, var_30, var_22]
    var_32 = True
    var_33 = True
    var_34 = module_0.git_hook(var_33)
    assert var_34 == 0



# Parsed testcases at query #11
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'subprocess.run'
    var_1 = b'file1.py\nfile2.py'
    var_2 = 'git_hook.get_output'
    var_3 = 'import os\nimport sys'
    var_4 = 'git_hook.api.check_code_string'
    var_5 = False
    var_6 = True
    var_7 = module_0.git_hook(var_6)
    assert var_7 == 1
    var_8 = module_0.git_hook(var_5)
    assert var_8 == 0
    var_9 = 'git_hook.api.sort_file'
    var_10 = module_0.git_hook(modify=var_6)
    var_11 = 'git_hook.get_lines'
    var_12 = []
    var_13 = module_0.git_hook()
    assert var_13 == 0
    var_14 = 'file.txt'
    var_15 = [var_14]
    var_16 = module_0.git_hook()
    assert var_16 == 0



# Parsed testcases at query #12
#--------------------------


import isort.hooks as module_0

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
    assert var_9 == 2
    var_10 = False
    var_11 = True
    var_12 = module_0.git_hook(var_10, var_11)
    var_13 = True
    var_14 = module_0.git_hook(var_13, lazy=var_13)
    var_15 = True
    var_16 = 'src'
    var_17 = [var_16]
    var_18 = module_0.git_hook(var_15, directories=var_17)
    var_19 = True
    var_20 = 'pyproject.toml'
    var_21 = module_0.git_hook(var_19, settings_file=var_20)
    var_22 = True
    var_23 = module_0.git_hook(var_22)
    assert var_23 == 0



# Parsed testcases at query #13
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



# Parsed testcases at query #14
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
    assert var_7 == 0
    var_8 = 'file.py'
    var_9 = True
    var_10 = module_0.git_hook(lazy=var_9)
    var_11 = 'git'
    var_12 = 'diff-index'
    var_13 = '--name-only'
    var_14 = '--diff-filter=ACMRTUXB'
    var_15 = 'HEAD'
    var_16 = [var_11, var_12, var_13, var_14, var_15]
    var_17 = 'src/'
    var_18 = [var_17]
    var_19 = module_0.git_hook(directories=var_18)
    var_20 = 'git'
    var_21 = 'diff-index'
    var_22 = '--cached'
    var_23 = '--name-only'
    var_24 = '--diff-filter=ACMRTUXB'
    var_25 = 'HEAD'
    var_26 = [var_20, var_21, var_22, var_23, var_24, var_25, var_17]
    var_27 = True
    var_28 = '.isort.cfg'
    var_29 = module_0.git_hook(settings_file=var_28)
    var_30 = 'file.py'
    var_31 = module_1.abspath(var_30)
    var_32 = module_1.dirname(var_31)
    var_33 = module_0.git_hook()
    assert var_33 == 0



# Parsed testcases at query #15
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
    assert var_7 == 0
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
    var_27 = '.isort.cfg'
    var_28 = module_0.git_hook(settings_file=var_27)
    var_29 = 'file.py'
    var_30 = module_1.abspath(var_29)
    var_31 = module_1.dirname(var_30)



# Parsed testcases at query #16
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
    var_25 = 'config.cfg'
    var_26 = module_0.git_hook(settings_file=var_25)
    var_27 = 'file.py'
    var_28 = module_1.abspath(var_27)
    var_29 = module_1.dirname(var_28)
    var_30 = module_0.git_hook()
    assert var_30 == 0



# Parsed testcases at query #17
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



# Parsed testcases at query #18
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'subprocess.run'
    var_1 = b''
    var_2 = module_0.git_hook()
    assert var_2 == 0
    var_3 = b'file1.txt\nfile2.md'
    var_4 = module_0.git_hook()
    assert var_4 == 0
    var_5 = b'file1.py\nfile2.py'
    var_6 = b'print("hello")\n'
    var_7 = b'print("world")\n'
    var_8 = 'isort.api.check_code_string'
    var_9 = True
    var_10 = module_0.git_hook()
    assert var_10 == 0
    var_11 = False
    var_12 = module_0.git_hook()
    assert var_12 == 0
    var_13 = module_0.git_hook(var_9)
    assert var_13 == 2
    var_14 = 'isort.api.sort_file'
    var_15 = module_0.git_hook(modify=var_9)
    var_16 = module_0.git_hook(lazy=var_9)
    var_17 = 'src'
    var_18 = [var_17]
    var_19 = module_0.git_hook(directories=var_18)
    var_20 = '.isort.cfg'
    var_21 = module_0.git_hook(settings_file=var_20)



# Parsed testcases at query #19
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.git_hook()
    assert var_1 == 0
    var_2 = 'test.py'
    var_3 = [var_2]
    var_4 = 'print("hello")'
    var_5 = True
    var_6 = module_0.git_hook()
    assert var_6 == 0
    var_7 = False
    var_8 = module_0.git_hook()
    assert var_8 == 0
    var_9 = True
    var_10 = module_0.git_hook(var_9)
    assert var_10 == 1
    var_11 = True
    var_12 = module_0.git_hook(modify=var_11)
    var_13 = module_0.git_hook()
    assert var_13 == 0
    var_14 = [var_12]
    var_15 = True
    var_16 = module_0.git_hook(lazy=var_15)
    var_17 = 'git'
    var_18 = 'diff-index'
    var_19 = '--name-only'
    var_20 = '--diff-filter=ACMRTUXB'
    var_21 = 'HEAD'
    var_22 = [var_17, var_18, var_19, var_20, var_21]
    var_23 = [var_16]
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



# Parsed testcases at query #20
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
    assert var_7 == 0
    var_8 = True
    var_9 = module_0.git_hook(lazy=var_8)
    var_10 = 'git'
    var_11 = 'diff-index'
    var_12 = '--name-only'
    var_13 = '--diff-filter=ACMRTUXB'
    var_14 = 'HEAD'
    var_15 = [var_10, var_11, var_12, var_13, var_14]
    var_16 = 'src/'
    var_17 = 'tests/'
    var_18 = [var_16, var_17]
    var_19 = module_0.git_hook(directories=var_18)
    var_20 = 'git'
    var_21 = 'diff-index'
    var_22 = '--cached'
    var_23 = '--name-only'
    var_24 = '--diff-filter=ACMRTUXB'
    var_25 = 'HEAD'
    var_26 = [var_20, var_21, var_22, var_23, var_24, var_25, var_16, var_17]
    var_27 = True
    var_28 = '.isort.cfg'
    var_29 = module_0.git_hook(settings_file=var_28)
    var_30 = 'file.py'
    var_31 = module_1.abspath(var_30)
    var_32 = module_1.dirname(var_31)



# Parsed testcases at query #21
#--------------------------


import isort.hooks as module_0
import posixpath as module_1

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = False
    var_2 = True
    var_3 = module_0.git_hook(var_1, var_1)
    assert var_3 == 0
    var_4 = False
    var_5 = True
    var_6 = module_0.git_hook(var_5, var_4)
    assert var_6 == 1
    var_7 = False
    var_8 = True
    var_9 = module_0.git_hook(var_7, var_8)
    var_10 = 'file1.py'
    var_11 = False
    var_12 = True
    var_13 = module_0.git_hook(lazy=var_12)
    var_14 = 'git'
    var_15 = 'diff-index'
    var_16 = '--name-only'
    var_17 = '--diff-filter=ACMRTUXB'
    var_18 = 'HEAD'
    var_19 = [var_14, var_15, var_16, var_17, var_18]
    var_20 = False
    var_21 = True
    var_22 = 'src/'
    var_23 = [var_22]
    var_24 = module_0.git_hook(directories=var_23)
    var_25 = 'git'
    var_26 = 'diff-index'
    var_27 = '--cached'
    var_28 = '--name-only'
    var_29 = '--diff-filter=ACMRTUXB'
    var_30 = 'HEAD'
    var_31 = [var_25, var_26, var_27, var_28, var_29, var_30, var_22]
    var_32 = '.isort.cfg'
    var_33 = module_0.git_hook(settings_file=var_32)
    var_34 = 'file1.py'
    var_35 = module_1.abspath(var_34)
    var_36 = module_1.dirname(var_35)
    var_37 = module_0.git_hook()
    assert var_37 == 0



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
    var_5 = ''
    var_6 = module_0.git_hook(settings_file=var_5)
    assert var_6 == 0
    var_7 = '.'
    var_8 = [var_7]
    var_9 = module_0.git_hook(directories=var_8)
    assert var_9 == 0



# Parsed testcases at query #23
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



# Parsed testcases at query #24
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
    assert var_6 == 0
    var_7 = 'file.py'
    var_8 = True
    var_9 = module_0.git_hook(lazy=var_8)
    var_10 = 'git'
    var_11 = 'diff-index'
    var_12 = '--name-only'
    var_13 = '--diff-filter=ACMRTUXB'
    var_14 = 'HEAD'
    var_15 = [var_10, var_11, var_12, var_13, var_14]
    var_16 = 'src/'
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
    var_27 = module_0.git_hook()
    assert var_27 == 0



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
    var_29 = 'pyproject.toml'
    var_30 = module_0.git_hook(settings_file=var_29)
    var_31 = 'file1.py'
    var_32 = module_1.abspath(var_31)
    var_33 = module_1.dirname(var_32)



# Parsed testcases at query #27
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
    assert var_7 == 0
    var_8 = True
    var_9 = module_0.git_hook(lazy=var_8)
    var_10 = 'git'
    var_11 = 'diff-index'
    var_12 = '--name-only'
    var_13 = '--diff-filter=ACMRTUXB'
    var_14 = 'HEAD'
    var_15 = [var_10, var_11, var_12, var_13, var_14]
    var_16 = 'src/'
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
    var_27 = '.isort.cfg'
    var_28 = module_0.git_hook(settings_file=var_27)
    var_29 = 'file.py'
    var_30 = module_1.abspath(var_29)
    var_31 = module_1.dirname(var_30)



# Parsed testcases at query #28
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
    var_25 = 'pyproject.toml'
    var_26 = module_0.git_hook(settings_file=var_25)
    var_27 = 'file1.py'
    var_28 = module_1.abspath(var_27)
    var_29 = module_1.dirname(var_28)



# Parsed testcases at query #29
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = 'test.py'
    var_2 = module_0.git_hook()
    assert var_2 == 0
    var_3 = module_0.git_hook()
    assert var_3 == 0
    var_4 = True
    var_5 = module_0.git_hook(var_4)
    assert var_5 == 1
    var_6 = module_0.git_hook(modify=var_4)
    assert var_6 == 0
    var_7 = module_0.git_hook(lazy=var_4)
    assert var_7 == 0
    var_8 = 'pyproject.toml'
    var_9 = module_0.git_hook(settings_file=var_8)
    assert var_9 == 0
    var_10 = 'src/'
    var_11 = [var_10]
    var_12 = module_0.git_hook(directories=var_11)
    assert var_12 == 0
    var_13 = module_0.git_hook()
    assert var_13 == 0



# Parsed testcases at query #30
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



# Parsed testcases at query #31
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'subprocess.run'
    var_1 = b''
    var_2 = module_0.git_hook()
    assert var_2 == 0
    var_3 = b'file1.py\nfile2.py'
    var_4 = b'print("test")'
    var_5 = 'isort.api.check_code_string'
    var_6 = False
    var_7 = module_0.git_hook(var_6)
    assert var_7 == 0
    var_8 = True
    var_9 = module_0.git_hook(var_8)
    assert var_9 == 2
    var_10 = 'isort.api.sort_file'
    var_11 = module_0.git_hook(modify=var_8)
    var_12 = module_0.git_hook(lazy=var_8)
    var_13 = 'src/'
    var_14 = [var_13]
    var_15 = module_0.git_hook(directories=var_14)
    var_16 = '.isort.cfg'
    var_17 = module_0.git_hook(settings_file=var_16)



# Parsed testcases at query #32
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'subprocess.run'
    var_1 = 'obj'
    var_2 = 'stdout'
    var_3 = b''
    var_4 = {var_2: var_3}
    var_5 = module_0.git_hook()
    assert var_5 == 0
    var_6 = b'file1.py\nfile2.py'
    var_7 = {var_2: var_6}
    var_8 = b'print("test")'
    var_9 = {var_2: var_8}
    var_10 = 'isort.api.check_code_string'
    var_11 = False
    var_12 = module_0.git_hook()
    assert var_12 == 0
    var_13 = {var_2: var_6}
    var_14 = {var_2: var_8}
    var_15 = True
    var_16 = module_0.git_hook(var_15)
    assert var_16 == 1
    var_17 = {var_2: var_6}
    var_18 = {var_2: var_8}
    var_19 = 'isort.api.sort_file'
    var_20 = module_0.git_hook(modify=var_15)
    var_21 = {var_2: var_6}
    var_22 = {var_2: var_8}
    var_23 = module_0.git_hook(lazy=var_15)
    var_24 = 'git'
    var_25 = 'diff-index'
    var_26 = '--name-only'
    var_27 = '--diff-filter=ACMRTUXB'
    var_28 = 'HEAD'
    var_29 = [var_24, var_25, var_26, var_27, var_28]
    var_30 = {var_2: var_6}
    var_31 = {var_2: var_8}
    var_32 = 'src/'
    var_33 = [var_32]
    var_34 = module_0.git_hook(directories=var_33)
    var_35 = '--cached'
    var_36 = [var_24, var_25, var_35, var_26, var_27, var_28, var_32]
    var_37 = {var_2: var_6}
    var_38 = {var_2: var_8}
    var_39 = module_0.git_hook()
    assert var_39 == 0



# Parsed testcases at query #33
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
    assert var_5 == 2
    var_6 = True
    var_7 = module_0.git_hook(modify=var_6)
    assert var_7 == 0
    var_8 = True
    var_9 = module_0.git_hook(lazy=var_8)
    assert var_9 == 0
    var_10 = 'git'
    var_11 = 'diff-index'
    var_12 = '--name-only'
    var_13 = '--diff-filter=ACMRTUXB'
    var_14 = 'HEAD'
    var_15 = [var_10, var_11, var_12, var_13, var_14]
    var_16 = 'src/'
    var_17 = [var_16]
    var_18 = module_0.git_hook(directories=var_17)
    assert var_18 == 0
    var_19 = 'git'
    var_20 = 'diff-index'
    var_21 = '--cached'
    var_22 = '--name-only'
    var_23 = '--diff-filter=ACMRTUXB'
    var_24 = 'HEAD'
    var_25 = [var_19, var_20, var_21, var_22, var_23, var_24, var_16]
    var_26 = True
    var_27 = 'pyproject.toml'
    var_28 = module_0.git_hook(settings_file=var_27)
    assert var_28 == 0
    var_29 = 'file1.py'
    var_30 = module_1.abspath(var_29)
    var_31 = module_1.dirname(var_30)
    var_32 = module_0.git_hook()
    assert var_32 == 0



# Parsed testcases at query #34
#--------------------------


import isort.hooks as module_0
import posixpath as module_1

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = False
    var_2 = True
    var_3 = module_0.git_hook(var_1)
    assert var_3 == 0
    var_4 = False
    var_5 = True
    var_6 = module_0.git_hook(var_5)
    assert var_6 == 1
    var_7 = False
    var_8 = True
    var_9 = module_0.git_hook(var_7, var_8)
    assert var_9 == 0
    var_10 = 'file1.py'
    var_11 = False
    var_12 = True
    var_13 = module_0.git_hook(lazy=var_12)
    var_14 = 'git'
    var_15 = 'diff-index'
    var_16 = '--name-only'
    var_17 = '--diff-filter=ACMRTUXB'
    var_18 = 'HEAD'
    var_19 = [var_14, var_15, var_16, var_17, var_18]
    var_20 = False
    var_21 = True
    var_22 = 'src'
    var_23 = [var_22]
    var_24 = module_0.git_hook(directories=var_23)
    var_25 = 'git'
    var_26 = 'diff-index'
    var_27 = '--cached'
    var_28 = '--name-only'
    var_29 = '--diff-filter=ACMRTUXB'
    var_30 = 'HEAD'
    var_31 = [var_25, var_26, var_27, var_28, var_29, var_30, var_22]
    var_32 = 'pyproject.toml'
    var_33 = module_0.git_hook(settings_file=var_32)
    var_34 = 'file1.py'
    var_35 = module_1.abspath(var_34)
    var_36 = module_1.dirname(var_35)



# Parsed testcases at query #35
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
    var_5 = False
    var_6 = module_0.git_hook(var_5)
    assert var_6 == 0
    var_7 = True
    var_8 = module_0.git_hook(var_7, var_7)
    assert var_8 == 2
    var_9 = True
    var_10 = module_0.git_hook(var_9, lazy=var_9)
    assert var_10 == 2
    var_11 = 'git'
    var_12 = 'diff-index'
    var_13 = '--name-only'
    var_14 = '--diff-filter=ACMRTUXB'
    var_15 = 'HEAD'
    var_16 = [var_11, var_12, var_13, var_14, var_15]
    var_17 = True
    var_18 = 'src'
    var_19 = [var_18]
    var_20 = module_0.git_hook(var_17, directories=var_19)
    assert var_20 == 2
    var_21 = 'git'
    var_22 = 'diff-index'
    var_23 = '--cached'
    var_24 = '--name-only'
    var_25 = '--diff-filter=ACMRTUXB'
    var_26 = 'HEAD'
    var_27 = [var_21, var_22, var_23, var_24, var_25, var_26, var_18]



# Parsed testcases at query #36
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
    assert var_7 == 0
    var_8 = 'git'
    var_9 = 'diff-index'
    var_10 = '--name-only'
    var_11 = '--diff-filter=ACMRTUXB'
    var_12 = 'HEAD'
    var_13 = [var_8, var_9, var_10, var_11, var_12]
    var_14 = 'src/'
    var_15 = [var_14]
    var_16 = module_0.git_hook(directories=var_15)
    assert var_16 == 0
    var_17 = 'git'
    var_18 = 'diff-index'
    var_19 = '--cached'
    var_20 = '--name-only'
    var_21 = '--diff-filter=ACMRTUXB'
    var_22 = 'HEAD'
    var_23 = [var_17, var_18, var_19, var_20, var_21, var_22, var_14]
    var_24 = True
    var_25 = module_0.git_hook()
    assert var_25 == 0



# Parsed testcases at query #37
#--------------------------


import isort.hooks as module_0

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
    var_12 = True
    var_13 = module_0.git_hook(lazy=var_12)
    var_14 = 'git'
    var_15 = 'diff-index'
    var_16 = '--name-only'
    var_17 = '--diff-filter=ACMRTUXB'
    var_18 = 'HEAD'
    var_19 = [var_14, var_15, var_16, var_17, var_18]
    var_20 = 'src/'
    var_21 = [var_20]
    var_22 = module_0.git_hook(directories=var_21)
    var_23 = 'git'
    var_24 = 'diff-index'
    var_25 = '--cached'
    var_26 = '--name-only'
    var_27 = '--diff-filter=ACMRTUXB'
    var_28 = 'HEAD'
    var_29 = [var_23, var_24, var_25, var_26, var_27, var_28, var_20]
    var_30 = True
    var_31 = b'file1.txt\nfile2.py'
    var_32 = b'print("test")'
    var_33 = module_0.git_hook()
    assert var_33 == 0



# Parsed testcases at query #38
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



# Parsed testcases at query #39
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
    var_17 = 'path/to/settings'
    var_18 = module_0.git_hook(settings_file=var_17)
    var_19 = 'src/'
    var_20 = 'tests/'
    var_21 = [var_19, var_20]
    var_22 = module_0.git_hook(directories=var_21)
    var_23 = 'git'
    var_24 = 'diff-index'
    var_25 = '--cached'
    var_26 = '--name-only'
    var_27 = '--diff-filter=ACMRTUXB'
    var_28 = 'HEAD'
    var_29 = [var_23, var_24, var_25, var_26, var_27, var_28, var_19, var_20]
    var_30 = True



# Parsed testcases at query #40
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'subprocess.run'
    var_1 = []
    var_2 = b''
    var_3 = module_0.git_hook()
    assert var_3 == 0
    var_4 = []
    var_5 = b'file.txt'
    var_6 = module_0.git_hook()
    assert var_6 == 0
    var_7 = []
    var_8 = b'test.py'
    var_9 = []
    var_10 = b'print("hello")'
    var_11 = 'isort.api.check_code_string'
    var_12 = True
    var_13 = module_0.git_hook()
    assert var_13 == 0
    var_14 = []
    var_15 = []
    var_16 = False
    var_17 = module_0.git_hook(var_12)
    assert var_17 == 1
    var_18 = []
    var_19 = []
    var_20 = module_0.git_hook(var_16)
    assert var_20 == 0
    var_21 = []
    var_22 = []
    var_23 = 'isort.api.sort_file'
    var_24 = module_0.git_hook(modify=var_12)
    var_25 = []
    var_26 = module_0.git_hook(lazy=var_12)
    var_27 = 'git'
    var_28 = 'diff-index'
    var_29 = '--name-only'
    var_30 = '--diff-filter=ACMRTUXB'
    var_31 = 'HEAD'
    var_32 = [var_27, var_28, var_29, var_30, var_31]
    var_33 = []
    var_34 = 'src/'
    var_35 = [var_34]
    var_36 = module_0.git_hook(directories=var_35)
    var_37 = '--cached'
    var_38 = [var_27, var_28, var_37, var_29, var_30, var_31, var_34]
    var_39 = []
    var_40 = []
    var_41 = 'setup.cfg'
    var_42 = module_0.git_hook(settings_file=var_41)



# Parsed testcases at query #41
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



# Parsed testcases at query #42
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
    var_25 = 'pyproject.toml'
    var_26 = module_0.git_hook(settings_file=var_25)
    var_27 = 'file.py'
    var_28 = module_1.abspath(var_27)
    var_29 = module_1.dirname(var_28)



# Parsed testcases at query #43
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



# Parsed testcases at query #44
#--------------------------


import isort.hooks as module_0
import posixpath as module_1

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = 'file.txt'
    var_2 = 'file2.md'
    var_3 = module_0.git_hook()
    assert var_3 == 0
    var_4 = 'file1.py'
    var_5 = 'file2.py'
    var_6 = module_0.git_hook()
    assert var_6 == 0
    var_7 = module_0.git_hook()
    assert var_7 == 0
    var_8 = True
    var_9 = module_0.git_hook(var_8)
    assert var_9 == 1
    var_10 = module_0.git_hook(modify=var_8)
    var_11 = 'unstaged.py'
    var_12 = module_0.git_hook(lazy=var_8)
    var_13 = 'test_settings.cfg'
    var_14 = module_0.git_hook(settings_file=var_13)
    var_15 = module_1.abspath(var_4)
    var_16 = module_1.dirname(var_15)
    var_17 = 'dir1/file1.py'
    var_18 = 'dir2/file2.py'
    var_19 = 'dir1'
    var_20 = 'dir2'
    var_21 = [var_19, var_20]
    var_22 = module_0.git_hook(directories=var_21)



# Parsed testcases at query #45
#--------------------------


import isort.hooks as module_0
import posixpath as module_1

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = b'file1.py\nfile2.py'
    var_2 = b'print("hello")'
    var_3 = b'print("world")'
    var_4 = module_0.git_hook()
    assert var_4 == 0
    var_5 = b'file1.py\nfile2.py'
    var_6 = b'print("hello")'
    var_7 = b'print("world")'
    var_8 = True
    var_9 = module_0.git_hook(var_8)
    assert var_9 == 2
    var_10 = b'file1.py\nfile2.py'
    var_11 = b'print("hello")'
    var_12 = b'print("world")'
    var_13 = True
    var_14 = module_0.git_hook(modify=var_13)
    var_15 = True
    var_16 = module_0.git_hook(lazy=var_15)
    var_17 = 'git'
    var_18 = 'diff-index'
    var_19 = '--name-only'
    var_20 = '--diff-filter=ACMRTUXB'
    var_21 = 'HEAD'
    var_22 = [var_17, var_18, var_19, var_20, var_21]
    var_23 = 'src/'
    var_24 = [var_23]
    var_25 = module_0.git_hook(directories=var_24)
    var_26 = 'git'
    var_27 = 'diff-index'
    var_28 = '--cached'
    var_29 = '--name-only'
    var_30 = '--diff-filter=ACMRTUXB'
    var_31 = 'HEAD'
    var_32 = [var_26, var_27, var_28, var_29, var_30, var_31, var_23]
    var_33 = True
    var_34 = b'file1.py'
    var_35 = b'print("hello")'
    var_36 = '.isort.cfg'
    var_37 = module_0.git_hook(settings_file=var_36)
    var_38 = 'file1.py'
    var_39 = module_1.abspath(var_38)
    var_40 = module_1.dirname(var_39)



# Parsed testcases at query #46
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



# Parsed testcases at query #47
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



# Parsed testcases at query #48
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



# Parsed testcases at query #49
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



# Parsed testcases at query #50
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
    var_25 = 'config.cfg'
    var_26 = module_0.git_hook(settings_file=var_25)
    var_27 = 'file1.py'
    var_28 = module_1.abspath(var_27)
    var_29 = module_1.dirname(var_28)



# Parsed testcases at query #51
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



# Parsed testcases at query #52
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0



# Parsed testcases at query #53
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = 'file.txt'
    var_2 = 'file.md'
    var_3 = module_0.git_hook()
    assert var_3 == 0
    var_4 = 'file1.py'
    var_5 = 'file2.py'
    var_6 = module_0.git_hook()
    assert var_6 == 0
    var_7 = True
    var_8 = module_0.git_hook(var_7)
    assert var_8 == 2
    var_9 = module_0.git_hook(modify=var_7)
    assert var_9 == 0
    var_10 = module_0.git_hook(lazy=var_7)
    assert var_10 == 0
    var_11 = 'path/to/settings'
    var_12 = module_0.git_hook(settings_file=var_11)
    assert var_12 == 0
    var_13 = 'dir1/file1.py'
    var_14 = 'dir2/file2.py'
    var_15 = 'dir1'
    var_16 = 'dir2'
    var_17 = [var_15, var_16]
    var_18 = module_0.git_hook(directories=var_17)
    assert var_18 == 0



# Parsed testcases at query #54
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



# Parsed testcases at query #55
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



# Parsed testcases at query #56
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = b'file1.py\nfile2.py'
    var_2 = b'print("test")'
    var_3 = False
    var_4 = module_0.git_hook(var_3, var_3)
    assert var_4 == 0
    var_5 = b'file1.py\nfile2.py'
    var_6 = b'print("test")'
    var_7 = True
    var_8 = False
    var_9 = module_0.git_hook(var_7, var_8)
    assert var_9 == 1
    var_10 = b'file1.py\nfile2.py'
    var_11 = b'print("test")'
    var_12 = False
    var_13 = True
    var_14 = module_0.git_hook(var_12, var_13)
    assert var_14 == 0
    var_15 = b'file1.py\nfile2.py'
    var_16 = b'print("test")'
    var_17 = True
    var_18 = module_0.git_hook(lazy=var_17)
    assert var_18 == 0
    var_19 = b'src/file1.py\ntests/file2.py'
    var_20 = b'print("test")'
    var_21 = 'src/'
    var_22 = [var_21]
    var_23 = module_0.git_hook(directories=var_22)
    assert var_23 == 0



# Parsed testcases at query #57
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
    var_14 = 'git'
    var_15 = 'diff-index'
    var_16 = '--name-only'
    var_17 = '--diff-filter=ACMRTUXB'
    var_18 = 'HEAD'
    var_19 = [var_14, var_15, var_16, var_17, var_18, var_11]
    var_20 = True
    var_21 = 'setup.cfg'
    var_22 = module_0.git_hook(settings_file=var_21)
    var_23 = module_0.git_hook()
    assert var_23 == 0



# Parsed testcases at query #58
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
    assert var_5 == 0
    var_6 = True
    var_7 = module_0.git_hook(lazy=var_6)
    assert var_7 == 0
    var_8 = 'git'
    var_9 = 'diff-index'
    var_10 = '--name-only'
    var_11 = '--diff-filter=ACMRTUXB'
    var_12 = 'HEAD'
    var_13 = [var_8, var_9, var_10, var_11, var_12]
    var_14 = 'src/'
    var_15 = [var_14]
    var_16 = module_0.git_hook(directories=var_15)
    assert var_16 == 0
    var_17 = 'git'
    var_18 = 'diff-index'
    var_19 = '--cached'
    var_20 = '--name-only'
    var_21 = '--diff-filter=ACMRTUXB'
    var_22 = 'HEAD'
    var_23 = [var_17, var_18, var_19, var_20, var_21, var_22, var_14]
    var_24 = True
    var_25 = module_0.git_hook()
    assert var_25 == 0



# Parsed testcases at query #59
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
    assert var_7 == 0
    var_8 = True
    var_9 = module_0.git_hook(lazy=var_8)
    assert var_9 == 0
    var_10 = 'src/'
    var_11 = [var_10]
    var_12 = module_0.git_hook(directories=var_11)
    assert var_12 == 0
    var_13 = 'pyproject.toml'
    var_14 = module_0.git_hook(settings_file=var_13)
    var_15 = 'file1.py'
    var_16 = module_1.abspath(var_15)
    var_17 = module_1.dirname(var_16)



# Parsed testcases at query #60
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
    var_10 = 'src/'
    var_11 = [var_10]
    var_12 = module_0.git_hook(directories=var_11)
    assert var_12 == 0
    var_13 = '.isort.cfg'
    var_14 = module_0.git_hook(settings_file=var_13)
    assert var_14 == 0
    var_15 = module_0.git_hook()
    assert var_15 == 0



# Parsed testcases at query #61
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = 'sorted_file.py'
    var_2 = [var_1]
    var_3 = lambda cmd: var_2
    var_4 = 'import os\nimport sys\n'
    var_5 = lambda cmd: var_4
    var_6 = True
    var_7 = module_0.git_hook()
    assert var_7 == 0
    var_8 = 'unsorted_file.py'
    var_9 = [var_8]
    var_10 = lambda cmd: var_9
    var_11 = 'import sys\nimport os\n'
    var_12 = lambda cmd: var_11
    var_13 = False
    var_14 = module_0.git_hook(var_6)
    assert var_14 == 1
    var_15 = module_0.git_hook(var_13)
    assert var_15 == 0
    var_16 = None
    var_17 = module_0.git_hook(var_6, var_6)
    assert var_17 == 1
    var_18 = '--cached'
    var_19 = [var_8]
    var_20 = []
    var_21 = lambda cmd: var_19 if var_18 not in cmd else var_20
    var_22 = module_0.git_hook(lazy=var_6)
    assert var_22 == 1
    var_23 = 'src'
    var_24 = [var_8]
    var_25 = []
    var_26 = lambda cmd: var_24 if var_23 in cmd else var_25
    var_27 = [var_23]
    var_28 = module_0.git_hook(directories=var_27)
    assert var_28 == 1
    var_29 = [var_8]
    var_30 = lambda cmd: var_29
    var_31 = lambda settings_file, settings_path: var_16
    var_32 = 'pyproject.toml'
    var_33 = module_0.git_hook(settings_file=var_32)
    assert var_33 == 1
    var_34 = ()
    var_35 = module_0.git_hook()
    assert var_35 == 0



# Parsed testcases at query #62
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'subprocess.run'
    var_1 = b''
    var_2 = module_0.git_hook()
    assert var_2 == 0
    var_3 = b'file1.txt\nfile2.md'
    var_4 = module_0.git_hook()
    assert var_4 == 0
    var_5 = b'file1.py\nfile2.py'
    var_6 = b'print("hello")'
    var_7 = b'print("world")'
    var_8 = 'isort.api.check_code_string'
    var_9 = True
    var_10 = module_0.git_hook()
    assert var_10 == 0
    var_11 = False
    var_12 = module_0.git_hook(var_11)
    assert var_12 == 0
    var_13 = module_0.git_hook(var_9)
    assert var_13 == 2
    var_14 = 'isort.api.sort_file'
    var_15 = module_0.git_hook(modify=var_9)
    var_16 = module_0.git_hook(lazy=var_9)
    var_17 = 'git'
    var_18 = 'diff-index'
    var_19 = '--name-only'
    var_20 = '--diff-filter=ACMRTUXB'
    var_21 = 'HEAD'
    var_22 = [var_17, var_18, var_19, var_20, var_21]
    var_23 = 'src/'
    var_24 = [var_23]
    var_25 = module_0.git_hook(directories=var_24)
    var_26 = '--cached'
    var_27 = [var_17, var_18, var_26, var_19, var_20, var_21, var_23]



# Parsed testcases at query #63
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
    var_8 = 'file1.py'
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



# Parsed testcases at query #64
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
    var_10 = module_0.git_hook(var_1)
    assert var_10 == 1
    var_11 = module_0.git_hook(modify=var_1)
    assert var_11 == 0
    var_12 = module_0.git_hook(lazy=var_1)
    assert var_12 == 1
    var_13 = module_0.git_hook(settings_file=var_5)
    assert var_13 == 1
    var_14 = [var_7]
    var_15 = module_0.git_hook(directories=var_14)
    assert var_15 == 1



# Parsed testcases at query #65
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = 'file.txt'
    var_2 = 'file.md'
    var_3 = module_0.git_hook()
    assert var_3 == 0
    var_4 = 'file1.py'
    var_5 = 'file2.py'
    var_6 = module_0.git_hook()
    assert var_6 == 0
    var_7 = True
    var_8 = module_0.git_hook(var_7)
    assert var_8 == 2
    var_9 = False
    var_10 = module_0.git_hook(var_9)
    assert var_10 == 0
    var_11 = False
    var_12 = module_0.git_hook(modify=var_7)
    var_13 = module_0.git_hook(lazy=var_7)
    var_14 = 'src'
    var_15 = 'tests'
    var_16 = [var_14, var_15]
    var_17 = module_0.git_hook(directories=var_16)
    var_18 = {}
    var_19 = lambda **kwargs: config_kwargs.update(kwargs)
    var_20 = 'pyproject.toml'
    var_21 = module_0.git_hook(settings_file=var_20)



# Parsed testcases at query #66
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(var_0)
    assert var_1 == 0
    var_2 = True
    var_3 = module_0.git_hook(var_2)
    assert var_3 == 0
    var_4 = False
    var_5 = module_0.git_hook(var_4)
    assert var_5 == 0
    var_6 = True
    var_7 = module_0.git_hook(var_6)
    assert var_7 == 2
    var_8 = True
    var_9 = module_0.git_hook(var_8, var_8)
    assert var_9 == 2
    var_10 = True
    var_11 = module_0.git_hook(var_10, lazy=var_10)
    assert var_11 == 2
    var_12 = True
    var_13 = 'dir1'
    var_14 = 'dir2'
    var_15 = [var_13, var_14]
    var_16 = module_0.git_hook(var_12, directories=var_15)
    assert var_16 == 2
    var_17 = True
    var_18 = 'settings.cfg'
    var_19 = module_0.git_hook(var_17, settings_file=var_18)
    assert var_19 == 2



