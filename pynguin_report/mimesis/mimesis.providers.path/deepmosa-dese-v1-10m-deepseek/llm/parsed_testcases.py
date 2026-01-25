####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_path_initialization_with_default_platform. Retrieved 4/8 statements.
# Partially parsed test_path_initialization_with_linux_platform. Retrieved 3/4 statements.
# Partially parsed test_path_initialization_with_win32_platform. Retrieved 3/4 statements.
# Partially parsed test_path_initialization_with_win64_platform. Retrieved 3/4 statements.
# Partially parsed test_path_initialization_with_darwin_platform. Retrieved 3/4 statements.
# Partially parsed test_path_initialization_with_freebsd_platform. Retrieved 3/4 statements.


import mimesis.providers.path as module_0

def test_case_0():
    var_0 = module_0.Path()
    var_1 = 'win'
    var_2 = var_0._pathlib_home
    var_3 = var_0._pathlib_home

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'linux'
    var_1 = module_0.Path(var_0)
    var_2 = var_1._pathlib_home

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'win32'
    var_1 = module_0.Path(var_0)
    var_2 = var_1._pathlib_home

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'win64'
    var_1 = module_0.Path(var_0)
    var_2 = var_1._pathlib_home

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'darwin'
    var_1 = module_0.Path(var_0)
    var_2 = var_1._pathlib_home

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'freebsd10'
    var_1 = module_0.Path(var_0)
    var_2 = var_1._pathlib_home



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_path_constructor_with_default_platform. Retrieved 3/6 statements.
# Partially parsed test_path_constructor_with_custom_platform. Retrieved 3/4 statements.
# Partially parsed test_path_constructor_with_freebsd_platform. Retrieved 3/4 statements.
# Partially parsed test_path_constructor_with_win64_platform. Retrieved 3/4 statements.


import mimesis.providers.path as module_0

def test_case_0():
    var_0 = module_0.Path()
    var_1 = var_0._pathlib_home
    var_2 = 'win'

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'linux'
    var_1 = module_0.Path(var_0)
    var_2 = var_1._pathlib_home

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'freebsd10'
    var_1 = module_0.Path(var_0)
    var_2 = var_1._pathlib_home

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'win64'
    var_1 = module_0.Path(var_0)
    var_2 = var_1._pathlib_home

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Path()



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_user_method_returns_correct_path_for_linux. Retrieved 4/5 statements.
# Partially parsed test_user_method_returns_correct_path_for_windows. Retrieved 4/5 statements.
# Partially parsed test_user_method_returns_correct_path_for_darwin. Retrieved 4/5 statements.
# Partially parsed test_user_method_returns_correct_path_for_freebsd. Retrieved 4/5 statements.
# Partially parsed test_user_method_returns_lowercase_username_for_non_windows. Retrieved 6/7 statements.
# Partially parsed test_user_method_returns_capitalized_username_for_windows. Retrieved 7/8 statements.


import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'linux'
    var_1 = module_0.Path(var_0)
    var_2 = var_1.user()
    var_3 = '/home/'

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'win32'
    var_1 = module_0.Path(var_0)
    var_2 = var_1.user()
    var_3 = '\\Users\\'

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'darwin'
    var_1 = module_0.Path(var_0)
    var_2 = var_1.user()
    var_3 = '/Users/'

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'freebsd'
    var_1 = module_0.Path(var_0)
    var_2 = var_1.user()
    var_3 = '/home/'

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'linux'
    var_1 = module_0.Path(var_0)
    var_2 = var_1.user()
    var_3 = -1
    var_4 = '/'
    var_5 = result.split(var_4)[var_3]

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'win32'
    var_1 = module_0.Path(var_0)
    var_2 = var_1.user()
    var_3 = 0
    var_4 = -1
    var_5 = '\\'
    var_6 = result.split(var_5)[var_4][var_3]



