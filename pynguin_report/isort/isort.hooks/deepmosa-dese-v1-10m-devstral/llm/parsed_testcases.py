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
    var_4 = module_0.get_lines(var_3)

import isort.hooks as module_0

def test_case_0():
    var_0 = 'echo'
    var_1 = '  line1  \n  line2  '
    var_2 = [var_0, var_1]
    var_3 = module_0.get_lines(var_2)

import isort.hooks as module_0

def test_case_0():
    var_0 = 'echo'
    var_1 = '-n'
    var_2 = ''
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.get_lines(var_3)



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
    var_0 = 'src/'
    var_1 = 'tests/'
    var_2 = [var_0, var_1]
    var_3 = module_0.git_hook(directories=var_2)



# Parsed testcases at query #3
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(var_0)
    assert var_1 == 0



# Parsed testcases at query #4
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0



# Parsed testcases at query #5
#--------------------------




def test_case_0():
    pass



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



# Parsed testcases at query #9
#--------------------------




def test_case_0():
    pass



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
    var_1 = module_0.git_hook(var_0)

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
    var_0 = 'src/'
    var_1 = [var_0]
    var_2 = module_0.git_hook(directories=var_1)
    assert var_2 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = module_0.git_hook(settings_file=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(modify=var_0)
    assert var_1 == 0



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
    var_0 = module_0.git_hook()
    assert var_0 == 0

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
    var_1 = module_0.git_hook(modify=var_0, lazy=var_0)
    assert var_1 == 0



# Parsed testcases at query #12
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = ''
    var_3 = None
    var_4 = module_0.git_hook(var_0, var_1, var_1, var_2, var_3)
    assert var_4 == 0



# Parsed testcases at query #13
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0



# Parsed testcases at query #14
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #15
#--------------------------




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
    var_0 = 'setup.cfg'
    var_1 = module_0.git_hook(settings_file=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = 'src/'
    var_1 = [var_0]
    var_2 = module_0.git_hook(directories=var_1)
    assert var_2 == 0



# Parsed testcases at query #16
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
    var_0 = 'pyproject.toml'
    var_1 = module_0.git_hook(settings_file=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = 'src/'
    var_1 = [var_0]
    var_2 = module_0.git_hook(directories=var_1)
    assert var_2 == 0



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

import isort.hooks as module_0

def test_case_0():
    var_0 = 'echo'
    var_1 = '  line1  \n  line2  '
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
    var_1 = '-n'
    var_2 = 'single_line'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.get_lines(var_3)



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
    var_0 = False
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

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = '.isort.cfg'
    var_2 = 'src/'
    var_3 = [var_2]
    var_4 = module_0.git_hook(var_0, var_0, var_0, var_1, var_3)



# Parsed testcases at query #5
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0



# Parsed testcases at query #6
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #7
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_git_hook_strict_mode. Retrieved 2/3 statements.
# Partially parsed test_git_hook_with_modify_flag. Retrieved 2/3 statements.
# Partially parsed test_git_hook_with_lazy_flag. Retrieved 2/3 statements.
# Partially parsed test_git_hook_with_settings_file. Retrieved 2/3 statements.
# Partially parsed test_git_hook_with_directories. Retrieved 4/5 statements.


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
    var_0 = 'setup.cfg'
    var_1 = module_0.git_hook(settings_file=var_0)

import isort.hooks as module_0

def test_case_0():
    var_0 = 'src/'
    var_1 = [var_0]
    var_2 = True
    var_3 = module_0.git_hook(var_2, directories=var_1)



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
    var_5 = 'setup.cfg'
    var_6 = module_0.git_hook(settings_file=var_5)
    assert var_6 == 0
    var_7 = 'src/'
    var_8 = [var_7]
    var_9 = module_0.git_hook(directories=var_8)
    assert var_9 == 0

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
    var_5 = 'setup.cfg'
    var_6 = module_0.git_hook(settings_file=var_5)
    assert var_6 == 0
    var_7 = 'src/'
    var_8 = [var_7]
    var_9 = module_0.git_hook(directories=var_8)
    assert var_9 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(var_0)
    assert var_1 == 1
    var_2 = module_0.git_hook(var_0, var_0)
    assert var_2 == 1
    var_3 = module_0.git_hook(var_0, lazy=var_0)
    assert var_3 == 1
    var_4 = 'setup.cfg'
    var_5 = module_0.git_hook(var_0, settings_file=var_4)
    assert var_5 == 1
    var_6 = 'src/'
    var_7 = [var_6]
    var_8 = module_0.git_hook(var_0, directories=var_7)
    assert var_8 == 1

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.git_hook(var_0)
    assert var_1 == 2
    var_2 = module_0.git_hook(var_0, var_0)
    assert var_2 == 2
    var_3 = module_0.git_hook(var_0, lazy=var_0)
    assert var_3 == 2
    var_4 = 'setup.cfg'
    var_5 = module_0.git_hook(var_0, settings_file=var_4)
    assert var_5 == 2
    var_6 = 'src/'
    var_7 = [var_6]
    var_8 = module_0.git_hook(var_0, directories=var_7)
    assert var_8 == 2

import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0
    var_1 = True
    var_2 = module_0.git_hook(modify=var_1)
    assert var_2 == 0
    var_3 = module_0.git_hook(lazy=var_1)
    assert var_3 == 0
    var_4 = 'setup.cfg'
    var_5 = module_0.git_hook(settings_file=var_4)
    assert var_5 == 0
    var_6 = 'src/'
    var_7 = [var_6]
    var_8 = module_0.git_hook(directories=var_7)
    assert var_8 == 0



# Parsed testcases at query #10
#--------------------------




def test_case_0():
    pass



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
    var_1 = [var_0]
    var_2 = module_0.git_hook(directories=var_1)
    assert var_2 == 0

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
    assert var_1 == 0



# Parsed testcases at query #12
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #13
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0



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
    var_1 = module_0.git_hook(modify=var_0)
    assert var_1 == 0

import isort.hooks as module_0

def test_case_0():
    var_0 = True
    var_1 = 'setup.cfg'
    var_2 = '.'
    var_3 = [var_2]
    var_4 = module_0.git_hook(var_0, var_0, var_0, var_1, var_3)
    assert var_4 == 0



# Parsed testcases at query #15
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0



# Parsed testcases at query #16
#--------------------------




import isort.hooks as module_0

def test_case_0():
    var_0 = module_0.git_hook()
    assert var_0 == 0



# Parsed testcases at query #17
#--------------------------




def test_case_0():
    pass



