####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = 'echo'
    var_1 = '-e'
    var_2 = '  line1  \n  line2  \n  line3  '
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.get_lines(var_3)
    var_5 = bool(var_4 == ['line1', 'line2', 'line3'])
    assert var_5 is True



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
    var_0 = True
    var_1 = module_0.git_hook(var_0)
    var_2 = bool(var_1 > 0)
    assert var_2 is True

import isort.hooks as module_0

def test_case_0():
    var_0 = False
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



# Parsed testcases at query #3
#--------------------------




def test_case_0():
    var_0 = bool(not [])
    assert var_0 is True



# Parsed testcases at query #4
#--------------------------




def test_case_0():
    var_0 = bool(not [])
    assert var_0 is True



# Parsed testcases at query #5
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
    var_0 = 'src/'
    var_1 = 'tests/'
    var_2 = [var_0, var_1]
    var_3 = module_0.git_hook(directories=var_2)
    assert var_3 == 0

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
    var_0 = True
    var_1 = module_0.git_hook(modify=var_0)
    assert var_1 == 0



# Parsed testcases at query #6
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0



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
    var_1 = [var_0]
    var_2 = module_0.git_hook(directories=var_1)
    assert var_2 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = 'setup.cfg'
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
    var_1 = 'setup.cfg'
    var_2 = 'src/'
    var_3 = [var_2]
    var_4 = module_0.git_hook(var_0, var_0, var_0, var_1, var_3)
    assert var_4 == 0



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
    var_0 = True
    var_1 = module_0.git_hook(var_0)
    var_2 = bool(var_1 > 0)
    assert var_2 is True

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
    var_1 = module_0.git_hook(var_0, lazy=var_0)
    var_2 = bool(var_1 > 0)
    assert var_2 is True

import isort.hooks as module_0

def test_case_0():
    var_0 = 'src/'
    var_1 = [var_0]
    var_2 = True
    var_3 = module_0.git_hook(var_2, directories=var_1)
    var_4 = bool(var_3 > 0)
    assert var_4 is True

import isort.hooks as module_0

def test_case_0():
    var_0 = '.custom_isort.cfg'
    var_1 = True
    var_2 = module_0.git_hook(var_1, settings_file=var_0)
    var_3 = bool(var_2 > 0)
    assert var_3 is True



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



# Parsed testcases at query #11
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0



# Parsed testcases at query #12
#--------------------------




def test_case_0():
    var_0 = bool(not [])
    assert var_0 is True



# Parsed testcases at query #13
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



# Parsed testcases at query #14
#--------------------------




def test_case_0():
    var_0 = bool(not [])
    assert var_0 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_git_hook_with_lazy_flag. Retrieved 2/3 statements.
# Partially parsed test_git_hook_with_directories. Retrieved 4/5 statements.
# Partially parsed test_git_hook_with_settings_file. Retrieved 2/3 statements.
# Partially parsed test_git_hook_modify_flag. Retrieved 2/3 statements.


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

import isort.hooks as module_0

def test_case_0():
    var_0 = 'src/'
    var_1 = [var_0]
    var_2 = True
    var_3 = module_0.git_hook(var_2, directories=var_1)

import isort.hooks as module_0

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = module_0.git_hook(settings_file=var_0)

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(modify=var_0)



# Parsed testcases at query #16
#--------------------------




def test_case_0():
    var_0 = bool(not [])
    assert var_0 is True



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
    var_1 = 'tests/'
    var_2 = [var_0, var_1]
    var_3 = module_0.git_hook(directories=var_2)
    assert var_3 == 0

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
    var_0 = True
    var_1 = module_0.git_hook(modify=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = '.isort.cfg'
    var_2 = 'src/'
    var_3 = [var_2]
    var_4 = module_0.git_hook(var_0, var_0, var_0, var_1, var_3)
    assert var_4 == 0



# Parsed testcases at query #3
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
    var_1 = 'tests/'
    var_2 = [var_0, var_1]
    var_3 = module_0.git_hook(directories=var_2)
    assert var_3 == 0

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




import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0



# Parsed testcases at query #6
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0



# Parsed testcases at query #7
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
    var_1 = 'tests/'
    var_2 = [var_0, var_1]
    var_3 = module_0.git_hook(directories=var_2)
    assert var_3 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = 'pyproject.toml'
    var_2 = 'src/'
    var_3 = 'tests/'
    var_4 = [var_2, var_3]
    var_5 = module_0.git_hook(var_0, var_0, var_0, var_1, var_4)
    assert var_5 == 0



# Parsed testcases at query #8
#--------------------------




def test_case_0():
    var_0 = []
    var_1 = bool(not var_0)
    assert var_1 is True



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
    var_2 = bool(var_1 >= 0)
    assert var_2 is True

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
    var_1 = module_0.git_hook(var_0, var_0)
    var_2 = bool(var_1 >= 0)
    assert var_2 is True

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = 'pyproject.toml'
    var_2 = 'src/'
    var_3 = [var_2]
    var_4 = module_0.git_hook(var_0, var_0, var_0, var_1, var_3)
    var_5 = bool(var_4 >= 0)
    assert var_5 is True



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



# Parsed testcases at query #11
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
    var_0 = 'pyproject.toml'
    var_1 = module_0.git_hook(settings_file=var_0)
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
    var_0 = True
    var_1 = module_0.git_hook(var_0)
    assert var_1 == 1

import isort.hooks as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.git_hook(var_0)
    assert var_1 == 0



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
    var_1 = module_0.git_hook(var_0)
    assert var_1 == 0
    var_2 = False
    var_3 = module_0.git_hook(var_2)
    assert var_3 == 0
    var_4 = module_0.git_hook(modify=var_0)
    assert var_4 == 0
    var_5 = module_0.git_hook(lazy=var_0)
    assert var_5 == 0
    var_6 = 'pyproject.toml'
    var_7 = module_0.git_hook(settings_file=var_6)
    assert var_7 == 0
    var_8 = 'src/'
    var_9 = [var_8]
    var_10 = module_0.git_hook(directories=var_9)
    assert var_10 == 0



# Parsed testcases at query #14
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(var_0)
    assert var_1 == 0



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




import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0



