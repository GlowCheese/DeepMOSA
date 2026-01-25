####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = 'echo'
    var_1 = 'line1\nline2\nline3'
    var_2 = [var_0, var_1]
    var_3 = module_0.get_lines(var_2)
    var_4 = 'line1'
    var_5 = 'line2'
    var_6 = 'line3'
    var_7 = [var_4, var_5, var_6]

import isort.hooks as module_0

def test_case_0():
    var_0 = 'echo'
    var_1 = ''
    var_2 = [var_0, var_1]
    var_3 = module_0.get_lines(var_2)
    var_4 = []

import isort.hooks as module_0

def test_case_0():
    var_0 = 'echo'
    var_1 = '  line1  \n  line2  \n  line3  '
    var_2 = [var_0, var_1]
    var_3 = module_0.get_lines(var_2)
    var_4 = 'line1'
    var_5 = 'line2'
    var_6 = 'line3'
    var_7 = [var_4, var_5, var_6]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_git_hook_strict_mode. Retrieved 2/3 statements.
# Partially parsed test_git_hook_modify_mode. Retrieved 2/3 statements.
# Partially parsed test_git_hook_lazy_mode. Retrieved 2/3 statements.
# Partially parsed test_git_hook_with_settings_file. Retrieved 2/3 statements.
# Partially parsed test_git_hook_with_directories. Retrieved 3/4 statements.


import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(var_0)

import isort.hooks as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.git_hook(var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(modify=var_0)

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(lazy=var_0)

import isort.hooks as module_0

def test_case_0():
    var_0 = '.isort.cfg'
    var_1 = module_0.git_hook(settings_file=var_0)

import isort.hooks as module_0

def test_case_0():
    var_0 = 'src'
    var_1 = [var_0]
    var_2 = module_0.git_hook(directories=var_1)

import isort.hooks as module_0

def test_case_0():
    var_0 = 'nonexistent_directory'
    var_1 = [var_0]
    var_2 = module_0.git_hook(directories=var_1)
    assert var_2 == 0



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_git_hook_strict_mode. Retrieved 2/3 statements.
# Partially parsed test_git_hook_strict_and_modify. Retrieved 2/3 statements.
# Partially parsed test_git_hook_lazy_and_strict. Retrieved 2/3 statements.
# Partially parsed test_git_hook_all_options. Retrieved 5/6 statements.


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(var_0)

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(modify=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(lazy=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = module_0.git_hook(settings_file=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = 'src'
    var_1 = [var_0]
    var_2 = module_0.git_hook(directories=var_1)
    assert var_2 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(var_0, var_0)

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(var_0, lazy=var_0)

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = 'setup.cfg'
    var_2 = 'src'
    var_3 = [var_2]
    var_4 = module_0.git_hook(var_0, var_0, var_0, var_1, var_3)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_git_hook_strict_mode. Retrieved 2/3 statements.
# Partially parsed test_git_hook_combination_of_modes. Retrieved 2/3 statements.


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(var_0)

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(modify=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(lazy=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = '.isort.cfg'
    var_1 = module_0.git_hook(settings_file=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = 'src'
    var_1 = [var_0]
    var_2 = module_0.git_hook(directories=var_1)
    assert var_2 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(var_0, var_0, var_0)



# Parsed testcases at query #5
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0



# Parsed testcases at query #6
#--------------------------




def test_case_0():
    var_0 = []



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_git_hook_strict_mode. Retrieved 2/3 statements.
# Partially parsed test_git_hook_lazy_mode. Retrieved 2/3 statements.
# Partially parsed test_git_hook_with_settings_file. Retrieved 2/3 statements.
# Partially parsed test_git_hook_with_directories. Retrieved 3/4 statements.


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(var_0)

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(modify=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(lazy=var_0)

import isort.hooks as module_0

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = module_0.git_hook(settings_file=var_0)

import isort.hooks as module_0

def test_case_0():
    var_0 = 'src'
    var_1 = [var_0]
    var_2 = module_0.git_hook(directories=var_1)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_predicate_evaluates_to_false_when_no_files_modified. Retrieved 1/2 statements.


def test_case_0():
    var_0 = []



# Parsed testcases at query #9
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0



# Parsed testcases at query #10
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(modify=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(lazy=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = 'settings.ini'
    var_1 = module_0.git_hook(settings_file=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = 'src'
    var_1 = [var_0]
    var_2 = module_0.git_hook(directories=var_1)
    assert var_2 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(var_0, var_0, var_0)
    assert var_1 == 0



# Parsed testcases at query #11
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = ''
    var_3 = None
    var_4 = module_0.git_hook(var_0, var_1, var_1, var_2, var_3)
    assert var_4 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = ''
    var_3 = None
    var_4 = module_0.git_hook(var_0, var_1, var_1, var_2, var_3)

import isort.hooks as module_0

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = None
    var_3 = module_0.git_hook(var_0, var_0, var_0, var_1, var_2)
    assert var_3 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = None
    var_3 = module_0.git_hook(var_0, var_0, var_0, var_1, var_2)
    assert var_3 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = ''
    var_3 = None
    var_4 = module_0.git_hook(var_0, var_1, var_0, var_2, var_3)
    assert var_4 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = ''
    var_3 = None
    var_4 = module_0.git_hook(var_0, var_0, var_1, var_2, var_3)
    assert var_4 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = False
    var_1 = '.isort.cfg'
    var_2 = None
    var_3 = module_0.git_hook(var_0, var_0, var_0, var_1, var_2)
    assert var_3 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = False
    var_1 = ''
    var_2 = 'src'
    var_3 = [var_2]
    var_4 = module_0.git_hook(var_0, var_0, var_0, var_1, var_3)
    assert var_4 == 0



# Parsed testcases at query #12
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = module_0.git_hook(var_0, var_1)

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(modify=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(lazy=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = 'src'
    var_1 = [var_0]
    var_2 = module_0.git_hook(directories=var_1)
    assert var_2 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = '.isort.cfg'
    var_1 = module_0.git_hook(settings_file=var_0)
    assert var_1 == 0



# Parsed testcases at query #13
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(modify=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(lazy=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = '.isort.cfg'
    var_1 = module_0.git_hook(settings_file=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = 'src'
    var_1 = 'tests'
    var_2 = [var_0, var_1]
    var_3 = module_0.git_hook(directories=var_2)
    assert var_3 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(var_0, var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(var_0, lazy=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(modify=var_0, lazy=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(var_0, var_0, var_0)
    assert var_1 == 0



# Parsed testcases at query #14
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = 'git'
    var_1 = 'diff-index'
    var_2 = '--cached'
    var_3 = '--name-only'
    var_4 = '--diff-filter=ACMRTUXB'
    var_5 = 'HEAD'
    var_6 = [var_0, var_1, var_2, var_3, var_4, var_5]
    var_7 = []
    var_8 = lambda cmd: var_7
    var_9 = module_0.git_hook()
    assert var_9 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = 'git'
    var_1 = 'diff-index'
    var_2 = '--cached'
    var_3 = '--name-only'
    var_4 = '--diff-filter=ACMRTUXB'
    var_5 = 'HEAD'
    var_6 = [var_0, var_1, var_2, var_3, var_4, var_5]
    var_7 = 'test.txt'
    var_8 = [var_7]
    var_9 = lambda cmd: var_8
    var_10 = ''
    var_11 = lambda cmd: var_10
    var_12 = module_0.git_hook()
    assert var_12 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = 'git'
    var_1 = 'diff-index'
    var_2 = '--cached'
    var_3 = '--name-only'
    var_4 = '--diff-filter=ACMRTUXB'
    var_5 = 'HEAD'
    var_6 = [var_0, var_1, var_2, var_3, var_4, var_5]
    var_7 = 'test.py'
    var_8 = [var_7]
    var_9 = lambda cmd: var_8
    var_10 = 'import os\nimport sys'
    var_11 = lambda cmd: var_10
    var_12 = True
    var_13 = lambda *args, **kwargs: var_12
    var_14 = module_0.git_hook()
    assert var_14 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = 'git'
    var_1 = 'diff-index'
    var_2 = '--cached'
    var_3 = '--name-only'
    var_4 = '--diff-filter=ACMRTUXB'
    var_5 = 'HEAD'
    var_6 = [var_0, var_1, var_2, var_3, var_4, var_5]
    var_7 = 'test.py'
    var_8 = [var_7]
    var_9 = lambda cmd: var_8
    var_10 = 'import sys\nimport os'
    var_11 = lambda cmd: var_10
    var_12 = False
    var_13 = lambda *args, **kwargs: var_12
    var_14 = True
    var_15 = module_0.git_hook(var_14)
    assert var_15 == 1

import isort.hooks as module_0

def test_case_0():
    var_0 = 'git'
    var_1 = 'diff-index'
    var_2 = '--cached'
    var_3 = '--name-only'
    var_4 = '--diff-filter=ACMRTUXB'
    var_5 = 'HEAD'
    var_6 = [var_0, var_1, var_2, var_3, var_4, var_5]
    var_7 = 'test.py'
    var_8 = [var_7]
    var_9 = lambda cmd: var_8
    var_10 = 'import sys\nimport os'
    var_11 = lambda cmd: var_10
    var_12 = False
    var_13 = lambda *args, **kwargs: var_12
    var_14 = module_0.git_hook(var_12)
    assert var_14 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = 'git'
    var_1 = 'diff-index'
    var_2 = '--cached'
    var_3 = '--name-only'
    var_4 = '--diff-filter=ACMRTUXB'
    var_5 = 'HEAD'
    var_6 = [var_0, var_1, var_2, var_3, var_4, var_5]
    var_7 = 'test.py'
    var_8 = [var_7]
    var_9 = lambda cmd: var_8
    var_10 = 'import sys\nimport os'
    var_11 = lambda cmd: var_10
    var_12 = False
    var_13 = lambda *args, **kwargs: var_12
    var_14 = None
    var_15 = lambda *args, **kwargs: var_14
    var_16 = True
    var_17 = module_0.git_hook(modify=var_16)
    assert var_17 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = 'git'
    var_1 = 'diff-index'
    var_2 = '--name-only'
    var_3 = '--diff-filter=ACMRTUXB'
    var_4 = 'HEAD'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = '--cached'
    var_7 = 'test.py'
    var_8 = [var_7]
    var_9 = []
    var_10 = lambda cmd: var_8 if var_6 not in cmd else var_9
    var_11 = 'import sys\nimport os'
    var_12 = lambda cmd: var_11
    var_13 = False
    var_14 = lambda *args, **kwargs: var_13
    var_15 = True
    var_16 = module_0.git_hook(var_15, lazy=var_15)
    assert var_16 == 1

import isort.hooks as module_0

def test_case_0():
    var_0 = 'git'
    var_1 = 'diff-index'
    var_2 = '--cached'
    var_3 = '--name-only'
    var_4 = '--diff-filter=ACMRTUXB'
    var_5 = 'HEAD'
    var_6 = 'src'
    var_7 = [var_0, var_1, var_2, var_3, var_4, var_5, var_6]
    var_8 = 'src/test.py'
    var_9 = [var_8]
    var_10 = lambda cmd: var_9
    var_11 = 'import sys\nimport os'
    var_12 = lambda cmd: var_11
    var_13 = False
    var_14 = lambda *args, **kwargs: var_13
    var_15 = [var_6]
    var_16 = True
    var_17 = module_0.git_hook(var_16, directories=var_15)
    assert var_17 == 1



# Parsed testcases at query #15
#--------------------------




def test_case_0():
    var_0 = []



# Parsed testcases at query #16
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.git_hook(var_0)
    assert var_1 == 0



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_git_hook_strict_mode. Retrieved 6/7 statements.


import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = False
    var_3 = ''
    var_4 = None
    var_5 = module_0.git_hook(var_0, var_1, var_2, var_3, var_4)

import isort.hooks as module_0

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = False
    var_3 = ''
    var_4 = None
    var_5 = module_0.git_hook(var_0, var_1, var_2, var_3, var_4)
    assert var_5 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = False
    var_1 = False
    var_2 = True
    var_3 = ''
    var_4 = None
    var_5 = module_0.git_hook(var_0, var_1, var_2, var_3, var_4)
    assert var_5 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = False
    var_1 = False
    var_2 = False
    var_3 = ''
    var_4 = 'src'
    var_5 = [var_4]
    var_6 = module_0.git_hook(var_0, var_1, var_2, var_3, var_5)
    assert var_6 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = False
    var_1 = False
    var_2 = False
    var_3 = 'settings.cfg'
    var_4 = None
    var_5 = module_0.git_hook(var_0, var_1, var_2, var_3, var_4)
    assert var_5 == 0



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = 'echo'
    var_1 = 'hello\nworld\n  python  '
    var_2 = [var_0, var_1]
    var_3 = module_0.get_lines(var_2)

import isort.hooks as module_0

def test_case_0():
    var_0 = 'echo'
    var_1 = ''
    var_2 = [var_0, var_1]
    var_3 = module_0.get_lines(var_2)

import isort.hooks as module_0

def test_case_0():
    var_0 = 'echo'
    var_1 = '  line1  \n  line2  \n  line3  '
    var_2 = [var_0, var_1]
    var_3 = module_0.get_lines(var_2)

import isort.hooks as module_0

def test_case_0():
    var_0 = 'echo'
    var_1 = 'single line'
    var_2 = [var_0, var_1]
    var_3 = module_0.get_lines(var_2)

import isort.hooks as module_0

def test_case_0():
    var_0 = 'echo'
    var_1 = 'line1 line2 line3'
    var_2 = [var_0, var_1]
    var_3 = module_0.get_lines(var_2)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_git_hook_strict_mode. Retrieved 2/3 statements.
# Partially parsed test_git_hook_strict_with_errors. Retrieved 5/14 statements.


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(var_0)

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(modify=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(lazy=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = '.isort.cfg'
    var_1 = module_0.git_hook(settings_file=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = 'src'
    var_1 = [var_0]
    var_2 = module_0.git_hook(directories=var_1)
    assert var_2 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = 'test_git_hook.get_lines'
    var_1 = 'test_git_hook.get_output'
    var_2 = 'test_git_hook.api.check_code_string'
    var_3 = True
    var_4 = module_0.git_hook(var_3)



# Parsed testcases at query #3
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0



# Parsed testcases at query #4
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = ''
    var_3 = None
    var_4 = module_0.git_hook(var_0, var_1, var_1, var_2, var_3)
    assert var_4 == 0



# Parsed testcases at query #5
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(var_0)
    assert var_1 == 0



# Parsed testcases at query #6
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(modify=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(lazy=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = '.isort.cfg'
    var_1 = module_0.git_hook(settings_file=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = 'src'
    var_1 = [var_0]
    var_2 = module_0.git_hook(directories=var_1)
    assert var_2 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(modify=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(lazy=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = '.isort.cfg'
    var_1 = 'src'
    var_2 = [var_1]
    var_3 = module_0.git_hook(settings_file=var_0, directories=var_2)
    assert var_3 == 0



# Parsed testcases at query #7
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_git_hook_strict_mode. Retrieved 2/3 statements.


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(var_0)

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(modify=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(lazy=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = module_0.git_hook(settings_file=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = 'src'
    var_1 = [var_0]
    var_2 = module_0.git_hook(directories=var_1)
    assert var_2 == 0



# Parsed testcases at query #9
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(var_0)
    assert var_1 == 0



# Parsed testcases at query #10
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0



# Parsed testcases at query #11
#--------------------------




def test_case_0():
    var_0 = []



# Parsed testcases at query #12
#--------------------------




def test_case_0():
    var_0 = []



# Parsed testcases at query #13
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = ''
    var_3 = None
    var_4 = module_0.git_hook(var_0, var_1, var_1, var_2, var_3)
    assert var_4 == 0



# Parsed testcases at query #14
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0



# Parsed testcases at query #15
#--------------------------




def test_case_0():
    var_0 = []



# Parsed testcases at query #16
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = ''
    var_3 = None
    var_4 = module_0.git_hook(var_0, var_0, var_1, var_2, var_3)
    assert var_4 == 0



# Parsed testcases at query #17
#--------------------------




def test_case_0():
    var_0 = []



