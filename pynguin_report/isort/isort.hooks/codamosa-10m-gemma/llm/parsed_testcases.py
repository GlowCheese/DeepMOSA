####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'line1\n  line2  \nline3\t'
    var_1 = 'dummy'
    var_2 = 'command'
    var_3 = [var_1, var_2]
    var_4 = module_0.get_lines(var_3)
    var_5 = 'single_line'
    var_6 = 'dummy'
    var_7 = 'command'
    var_8 = [var_6, var_7]
    var_9 = module_0.get_lines(var_8)
    var_10 = ''
    var_11 = 'dummy'
    var_12 = 'command'
    var_13 = [var_11, var_12]
    var_14 = module_0.get_lines(var_13)
    var_15 = 'git'
    var_16 = 'diff-index'
    var_17 = '--cached'
    var_18 = [var_15, var_16, var_17]
    var_19 = module_0.get_lines(var_18)
    var_20 = True



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'config.ini'



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = '\n'
    var_1 = 'pyproject.toml'
    var_2 = 'diff-index'
    var_3 = 0



# Parsed testcases at query #4
#--------------------------


import email._encoded_words as module_0

def test_case_0():
    var_0 = '\n'
    var_1 = module_0.encode()
    var_2 = '.py'
    var_3 = b'content'
    var_4 = 'config.ini'
    var_5 = 0
    var_6 = 0

import isort.hooks as module_0

def test_case_0():
    var_0 = b'file1.py\n'
    var_1 = b'content'
    var_2 = True
    var_3 = module_0.git_hook(var_2)
    assert var_3 == 0



# Parsed testcases at query #5
#--------------------------


import email._encoded_words as module_0

def test_case_0():
    var_0 = '\n'
    var_1 = module_0.encode()
    var_2 = b'import os\nimport sys'
    var_3 = []
    var_4 = 'config.py'
    var_5 = 0



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = b'\n'
    var_1 = ''
    var_2 = 0



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = '\n'
    var_1 = 'import os\nimport sys'
    var_2 = ''
    var_3 = 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(var_0)
    assert var_1 == 0



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = '\n'
    var_1 = 'import os\nimport sys'



# Parsed testcases at query #9
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'git'
    var_1 = 'status'
    var_2 = '--short'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.get_lines(var_3)
    var_5 = True
    var_6 = 'git'
    var_7 = 'ls-files'
    var_8 = [var_6, var_7]
    var_9 = module_0.get_lines(var_8)
    var_10 = 'echo'
    var_11 = 'test'
    var_12 = [var_10, var_11]
    var_13 = module_0.get_lines(var_12)
    var_14 = 1
    var_15 = 'git'
    var_16 = 'error'
    var_17 = [var_15, var_16]
    var_18 = [var_15, var_16]
    var_19 = module_0.get_lines(var_18)



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
    assert var_4 == 1
    var_5 = True
    var_6 = module_0.git_hook(var_5, var_5)
    assert var_6 == 1
    var_7 = 'file1.py'
    var_8 = True
    var_9 = module_0.git_hook(lazy=var_8)
    var_10 = 'src/'
    var_11 = [var_10]
    var_12 = module_0.git_hook(directories=var_11)
    var_13 = True
    var_14 = module_0.git_hook(var_13)
    assert var_14 == 0
    var_15 = True
    var_16 = module_0.git_hook(var_15)
    assert var_16 == 0



# Parsed testcases at query #11
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'dummy'
    var_1 = 'command'
    var_2 = [var_0, var_1]
    var_3 = module_0.get_lines(var_2)
    var_4 = 'dummy'
    var_5 = 'command'
    var_6 = [var_4, var_5]
    var_7 = module_0.get_lines(var_6)
    var_8 = 'dummy'
    var_9 = 'command'
    var_10 = [var_8, var_9]
    var_11 = module_0.get_lines(var_10)
    var_12 = 'ls'
    var_13 = '-l'
    var_14 = [var_12, var_13]
    var_15 = module_0.get_lines(var_14)
    var_16 = True



# Parsed testcases at query #12
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'line1\n  line2  \nline3\n'
    var_1 = 'dummy_cmd'
    var_2 = [var_1]
    var_3 = module_0.get_lines(var_2)
    var_4 = 'dummy_cmd'
    var_5 = [var_4]
    var_6 = module_0.get_lines(var_5)
    var_7 = 1
    var_8 = 'dummy_cmd'
    var_9 = [var_8]
    var_10 = 'dummy_cmd'
    var_11 = [var_10]
    var_12 = module_0.get_lines(var_11)
    var_13 = 'dummy_cmd'
    var_14 = [var_13]
    var_15 = module_0.get_lines(var_14)



# Parsed testcases at query #13
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'some'
    var_1 = 'command'
    var_2 = [var_0, var_1]
    var_3 = module_0.get_lines(var_2)
    var_4 = [var_0, var_1]
    var_5 = True
    var_6 = 'some'
    var_7 = 'command'
    var_8 = [var_6, var_7]
    var_9 = module_0.get_lines(var_8)
    var_10 = 1
    var_11 = 'cmd'
    var_12 = [var_11]
    var_13 = 'cmd'
    var_14 = [var_13]
    var_15 = module_0.get_lines(var_14)
    var_16 = 'some'
    var_17 = 'command'
    var_18 = [var_16, var_17]
    var_19 = module_0.get_lines(var_18)
    var_20 = 0



# Parsed testcases at query #14
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'line1\n  line2  \nline3\t'
    var_1 = 'dummy'
    var_2 = 'command'
    var_3 = [var_1, var_2]
    var_4 = module_0.get_lines(var_3)
    var_5 = 'dummy'
    var_6 = 'command'
    var_7 = [var_5, var_6]
    var_8 = module_0.get_lines(var_7)
    var_9 = 'dummy'
    var_10 = 'command'
    var_11 = [var_9, var_10]
    var_12 = module_0.get_lines(var_11)
    var_13 = 1
    var_14 = 'dummy'
    var_15 = [var_14]
    var_16 = 'dummy'
    var_17 = [var_16]
    var_18 = module_0.get_lines(var_17)



# Parsed testcases at query #15
#--------------------------


import email._encoded_words as module_0

def test_case_0():
    var_0 = '\n'
    var_1 = module_0.encode()
    var_2 = module_0.encode()
    var_3 = 1
    var_4 = 'config.py'



# Parsed testcases at query #16
#--------------------------


import email._encoded_words as module_0

def test_case_0():
    var_0 = 'git'
    var_1 = 'diff-index'
    var_2 = '--cached'
    var_3 = '--name-only'
    var_4 = '--diff-filter=ACMRTUXB'
    var_5 = 'HEAD'
    var_6 = [var_0, var_1, var_2, var_3, var_4, var_5]
    var_7 = '--cached'
    var_8 = '\n'
    var_9 = module_0.encode()
    var_10 = module_0.encode()
    var_11 = 0



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = '\n'
    var_1 = b'import os\nimport sys\n'
    var_2 = []
    var_3 = 'config.py'
    var_4 = 0



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = '\n'
    var_1 = 'import os\nimport sys'
    var_2 = ''
    var_3 = '--diff-filter'
    var_4 = 0
    var_5 = [arg for arg in var_0 if var_3 in arg[var_4][var_4]]
    var_6 = len(var_5)



# Parsed testcases at query #3
#--------------------------


import email._encoded_words as module_0

def test_case_0():
    var_0 = '\n'
    var_1 = 'utf-8'
    var_2 = module_0.encode(var_1)
    var_3 = b'import os\nimport sys'
    var_4 = 'config.ini'
    var_5 = 0



# Parsed testcases at query #4
#--------------------------


import email._encoded_words as module_0

def test_case_0():
    var_0 = '\n'
    var_1 = module_0.encode()
    var_2 = b'content'
    var_3 = 'test_config.ini'
    var_4 = 0



# Parsed testcases at query #5
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'some'
    var_1 = 'command'
    var_2 = [var_0, var_1]
    var_3 = module_0.get_lines(var_2)
    var_4 = [var_0, var_1]
    var_5 = True
    var_6 = 'some'
    var_7 = 'command'
    var_8 = [var_6, var_7]
    var_9 = module_0.get_lines(var_8)
    var_10 = 'some'
    var_11 = 'command'
    var_12 = [var_10, var_11]
    var_13 = module_0.get_lines(var_12)
    var_14 = 1
    var_15 = 'cmd'
    var_16 = [var_15]
    var_17 = 'cmd'
    var_18 = [var_17]
    var_19 = module_0.get_lines(var_18)



# Parsed testcases at query #6
#--------------------------


import email._encoded_words as module_0

def test_case_0():
    var_0 = '\n'
    var_1 = module_0.encode()
    var_2 = b'\n'
    var_3 = var_1 + var_2
    var_4 = module_0.encode()
    var_5 = ''

import isort.hooks as module_0

def test_case_0():
    var_0 = 'Test that directories argument is appended to git command.'
    var_1 = 'src'
    var_2 = 'tests'
    var_3 = [var_1, var_2]
    var_4 = module_0.git_hook(directories=var_3)
    var_5 = 0

import isort.hooks as module_0

def test_case_0():
    var_0 = 'Test that isort.exceptions.FileSkipped is handled gracefully.'
    var_1 = True
    var_2 = module_0.git_hook(var_1)
    assert var_2 == 0



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = '\n'
    var_1 = 'config.ini'
    var_2 = 0

import isort.hooks as module_0

def test_case_0():
    var_0 = 'FileSkipped'
    var_1 = True
    var_2 = module_0.git_hook(var_1)
    assert var_2 == 0



# Parsed testcases at query #8
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'dummy'
    var_1 = 'command'
    var_2 = [var_0, var_1]
    var_3 = module_0.get_lines(var_2)
    var_4 = 'dummy'
    var_5 = 'command'
    var_6 = [var_4, var_5]
    var_7 = module_0.get_lines(var_6)
    var_8 = 'dummy'
    var_9 = 'command'
    var_10 = [var_8, var_9]
    var_11 = module_0.get_lines(var_10)
    var_12 = 1
    var_13 = 'dummy'
    var_14 = [var_13]
    var_15 = 'dummy'
    var_16 = 'command'
    var_17 = [var_15, var_16]
    var_18 = module_0.get_lines(var_17)



# Parsed testcases at query #9
#--------------------------


import email._encoded_words as module_0

def test_case_0():
    var_0 = '\n'
    var_1 = 'import os\nimport sys'
    var_2 = module_0.encode()
    var_3 = module_0.encode()
    var_4 = module_0.encode()
    var_5 = 'config.ini'
    var_6 = 0



# Parsed testcases at query #10
#--------------------------


import email._encoded_words as module_0
import email.base64mime as module_1

def test_case_0():
    var_0 = '\n'
    var_1 = module_0.encode()
    var_2 = b''
    var_3 = b'import os\nimport sys'
    var_4 = module_1.decode()
    var_5 = module_0.encode()
    var_6 = b''
    var_7 = module_1.decode()
    var_8 = module_0.encode()
    var_9 = [var_4]
    var_10 = ''



# Parsed testcases at query #11
#--------------------------


import isort.hooks as module_0

def test_case_0():
    var_0 = 'line1\n  line2  \nline3\n'
    var_1 = 'line1'
    var_2 = 'line_2'
    var_3 = 'line3'
    var_4 = [var_1, var_2, var_3]
    var_5 = 'line2'
    var_6 = [var_1, var_5, var_3]
    var_7 = 'git'
    var_8 = 'status'
    var_9 = [var_7, var_8]
    var_10 = module_0.get_lines(var_9)
    var_11 = [var_7, var_8]
    var_12 = True
    var_13 = 'git'
    var_14 = 'diff'
    var_15 = [var_13, var_14]
    var_16 = module_0.get_lines(var_15)
    var_17 = 'git'
    var_18 = 'ls-files'
    var_19 = [var_17, var_18]
    var_20 = module_0.get_lines(var_19)
    var_21 = 0
    var_22 = 1
    var_23 = 'cmd'
    var_24 = 'false'
    var_25 = [var_24]
    var_26 = module_0.get_lines(var_25)



# Parsed testcases at query #12
#--------------------------


import email._encoded_words as module_0

def test_case_0():
    var_0 = '\n'
    var_1 = module_0.encode()
    var_2 = 'config.ini'
    var_3 = '.py'
    var_4 = '--cached'
    var_5 = 0
    var_6 = 'diff-index'
    var_7 = [arg for (arg, kwargs) in var_0 if var_4 not in arg[var_5] and var_6 in arg[var_5]]



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = '\n'
    var_1 = 'import os\nimport sys\n'
    var_2 = []
    var_3 = 0



