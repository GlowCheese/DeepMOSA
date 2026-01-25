####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
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
    var_1 = module_0.git_hook(lazy=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = 'src/'
    var_1 = 'tests/'
    var_2 = [var_0, var_1]
    var_3 = module_0.git_hook(directories=var_2)
    assert var_3 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = '.isort.cfg'
    var_1 = module_0.git_hook(settings_file=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(modify=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(var_0)
    assert var_1 == 1

import isort.hooks as module_0

def test_case_0():
    var_0 = 'src/'
    var_1 = 'tests/'
    var_2 = 'docs/'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.git_hook(directories=var_3)
    assert var_4 == 0



# Parsed testcases at query #2
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = 'echo'
    var_1 = 'line1\nline2\nline3'
    var_2 = [var_0, var_1]
    var_3 = module_0.get_lines(var_2)
    var_4 = bool(var_3 == ['line1', 'line2', 'line3'])
    assert var_4 is True

import isort.hooks as module_0

def test_case_0():
    var_0 = 'echo'
    var_1 = '  line1  \n  line2  '
    var_2 = [var_0, var_1]
    var_3 = module_0.get_lines(var_2)
    var_4 = bool(var_3 == ['line1', 'line2'])
    assert var_4 is True

import isort.hooks as module_0

def test_case_0():
    var_0 = 'echo'
    var_1 = ''
    var_2 = [var_0, var_1]
    var_3 = module_0.get_lines(var_2)
    var_4 = bool(var_3 == [''])
    assert var_4 is True

import isort.hooks as module_0

def test_case_0():
    var_0 = 'echo'
    var_1 = 'single_line'
    var_2 = [var_0, var_1]
    var_3 = module_0.get_lines(var_2)
    var_4 = bool(var_3 == ['single_line'])
    assert var_4 is True



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
    var_0 = 'setup.cfg'
    var_1 = module_0.git_hook(settings_file=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = 'src/'
    var_1 = [var_0]
    var_2 = module_0.git_hook(directories=var_1)
    assert var_2 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = 'setup.cfg'
    var_2 = 'src/'
    var_3 = [var_2]
    var_4 = module_0.git_hook(var_0, var_0, var_0, var_1, var_3)
    assert var_4 == 0



# Parsed testcases at query #5
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0



# Parsed testcases at query #6
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = ''
    var_3 = None
    var_4 = module_0.git_hook(var_0, var_1, var_1, var_2, var_3)
    assert var_4 == 0



# Parsed testcases at query #7
#--------------------------




def test_case_0():
    var_0 = bool(not [])
    assert var_0 is True



# Parsed testcases at query #8
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0



# Parsed testcases at query #9
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
    var_0 = False
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
    var_0 = 'pyproject.toml'
    var_1 = module_0.git_hook(settings_file=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = 'src/'
    var_1 = [var_0]
    var_2 = module_0.git_hook(directories=var_1)
    assert var_2 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = 'pyproject.toml'
    var_2 = 'src/'
    var_3 = [var_2]
    var_4 = module_0.git_hook(var_0, var_0, var_0, var_1, var_3)
    assert var_4 == 0



# Parsed testcases at query #10
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

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(lazy=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(modify=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = module_0.git_hook(settings_file=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = 'src/'
    var_1 = [var_0]
    var_2 = module_0.git_hook(directories=var_1)
    assert var_2 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = 'setup.cfg'
    var_2 = 'src/'
    var_3 = [var_2]
    var_4 = module_0.git_hook(var_0, var_0, var_0, var_1, var_3)
    assert var_4 == 0



# Parsed testcases at query #11
#--------------------------




def test_case_0():
    var_0 = bool(not [])
    assert var_0 is True



# Parsed testcases at query #12
#--------------------------




def test_case_0():
    var_0 = bool(not [])
    assert var_0 is True



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




import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = ''
    var_3 = None
    var_4 = module_0.git_hook(var_0, var_1, var_1, var_2, var_3)
    assert var_4 == 0



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



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = 'echo'
    var_1 = 'Hello, World!'
    var_2 = [var_0, var_1]
    var_3 = module_0.get_lines(var_2)
    var_4 = bool(var_3 == ['Hello, World!'])
    assert var_4 is True

import isort.hooks as module_0

def test_case_0():
    var_0 = 'printf'
    var_1 = 'Line 1\nLine 2\nLine 3'
    var_2 = [var_0, var_1]
    var_3 = module_0.get_lines(var_2)
    var_4 = bool(var_3 == ['Line 1', 'Line 2', 'Line 3'])
    assert var_4 is True

import isort.hooks as module_0

def test_case_0():
    var_0 = 'echo'
    var_1 = '-n'
    var_2 = [var_0, var_1]
    var_3 = module_0.get_lines(var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import isort.hooks as module_0

def test_case_0():
    var_0 = 'printf'
    var_1 = '  Line with spaces  \nAnother line  '
    var_2 = [var_0, var_1]
    var_3 = module_0.get_lines(var_2)
    var_4 = bool(var_3 == ['Line with spaces', 'Another line'])
    assert var_4 is True



# Parsed testcases at query #2
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
    var_0 = 'src/'
    var_1 = [var_0]
    var_2 = module_0.git_hook(directories=var_1)
    assert var_2 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(lazy=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(modify=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = module_0.git_hook(settings_file=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(var_0)
    var_2 = bool(var_1 > 0)
    assert var_2 is True

import isort.hooks as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.git_hook(var_0)
    assert var_1 == 0



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
    var_0 = module_0.git_hook()
    assert var_0 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.git_hook(var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(var_0)
    assert var_1 == 1

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
    var_0 = 'pyproject.toml'
    var_1 = module_0.git_hook(settings_file=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = 'src/'
    var_1 = [var_0]
    var_2 = module_0.git_hook(directories=var_1)
    assert var_2 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = 'pyproject.toml'
    var_2 = 'src/'
    var_3 = [var_2]
    var_4 = module_0.git_hook(var_0, var_0, var_0, var_1, var_3)
    assert var_4 == 0



# Parsed testcases at query #5
#--------------------------




def test_case_0():
    var_0 = bool(not [])
    assert var_0 is True



# Parsed testcases at query #6
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

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(var_0)
    assert var_1 == 1

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
    var_0 = 'src/'
    var_1 = [var_0]
    var_2 = module_0.git_hook(directories=var_1)
    assert var_2 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = 'setup.cfg'
    var_2 = 'src/'
    var_3 = [var_2]
    var_4 = module_0.git_hook(var_0, var_0, var_0, var_1, var_3)
    assert var_4 == 0



# Parsed testcases at query #7
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0



# Parsed testcases at query #8
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0



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



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_git_hook_with_staged_files_no_errors. Retrieved 1/4 statements.
# Partially parsed test_git_hook_with_staged_files_with_errors. Retrieved 1/4 statements.
# Partially parsed test_git_hook_strict_mode_with_errors. Retrieved 2/5 statements.
# Partially parsed test_git_hook_modify_mode. Retrieved 2/6 statements.
# Partially parsed test_git_hook_lazy_mode. Retrieved 2/5 statements.
# Partially parsed test_git_hook_with_directories. Retrieved 3/6 statements.
# Partially parsed test_git_hook_with_settings_file. Retrieved 2/6 statements.


import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(var_0)
    assert var_1 == 2

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
    var_0 = 'src'
    var_1 = [var_0]
    var_2 = module_0.git_hook(directories=var_1)

import isort.hooks as module_0

def test_case_0():
    var_0 = 'path/to/settings'
    var_1 = module_0.git_hook(settings_file=var_0)



# Parsed testcases at query #12
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0



# Parsed testcases at query #13
#--------------------------




def test_case_0():
    var_0 = []
    var_1 = bool(not var_0)
    assert var_1 is True



# Parsed testcases at query #14
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
    var_0 = 'setup.cfg'
    var_1 = module_0.git_hook(settings_file=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = 'src/'
    var_1 = [var_0]
    var_2 = module_0.git_hook(directories=var_1)
    assert var_2 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = 'setup.cfg'
    var_2 = 'src/'
    var_3 = [var_2]
    var_4 = module_0.git_hook(var_0, var_0, var_0, var_1, var_3)
    assert var_4 == 0



# Parsed testcases at query #15
#--------------------------




def test_case_0():
    var_0 = bool(not [])
    assert var_0 is True



# Parsed testcases at query #16
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0



# Parsed testcases at query #17
#--------------------------




def test_case_0():
    var_0 = bool(not [])
    assert var_0 is True



