####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_path_initialization_with_default_platform. Retrieved 4/8 statements.
# Partially parsed test_path_initialization_with_specific_platform_linux. Retrieved 5/6 statements.
# Partially parsed test_path_initialization_with_specific_platform_darwin. Retrieved 5/6 statements.
# Partially parsed test_path_initialization_with_specific_platform_win32. Retrieved 5/6 statements.
# Partially parsed test_path_initialization_with_specific_platform_win64. Retrieved 5/6 statements.
# Partially parsed test_path_initialization_with_specific_platform_freebsd. Retrieved 5/6 statements.
# Failed to parse test_path_initialization_with_random_instance.


import mimesis.providers.path as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Path(**var_0)
    var_2 = var_1.platform
    var_3 = 'win'
    var_4 = var_1._pathlib_home
    var_5 = var_1._pathlib_home

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'linux'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = var_2.platform
    assert var_3 == 'linux'
    var_4 = var_2._pathlib_home
    var_5 = var_2._pathlib_home
    var_6 = str(var_5)

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'darwin'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = var_2.platform
    assert var_3 == 'darwin'
    var_4 = var_2._pathlib_home
    var_5 = var_2._pathlib_home
    var_6 = str(var_5)

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'win32'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = var_2.platform
    assert var_3 == 'win32'
    var_4 = var_2._pathlib_home
    var_5 = var_2._pathlib_home
    var_6 = str(var_5)

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'win64'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = var_2.platform
    assert var_3 == 'win64'
    var_4 = var_2._pathlib_home
    var_5 = var_2._pathlib_home
    var_6 = str(var_5)

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'freebsd10'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = var_2.platform
    assert var_3 == 'freebsd'
    var_4 = var_2._pathlib_home
    var_5 = var_2._pathlib_home
    var_6 = str(var_5)

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 12345
    var_1 = 'seed'
    var_2 = {var_1: var_0}
    var_3 = module_0.Path(**var_2)
    var_4 = var_3.seed
    var_5 = bool(var_3.seed == var_0)
    assert var_5 is True

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.Path(**var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 999
    var_1 = 'linux'
    var_2 = 'seed'
    var_3 = {var_2: var_0}
    var_4 = module_0.Path(var_1, **var_3)
    var_5 = var_4.seed
    assert var_5 == 999
    var_6 = var_4.platform
    assert var_6 == 'linux'



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_user_returns_path_with_username. Retrieved 5/8 statements.
# Partially parsed test_user_returns_capitalized_username_on_windows. Retrieved 9/12 statements.
# Partially parsed test_user_returns_lowercase_username_on_linux. Retrieved 6/8 statements.
# Partially parsed test_user_returns_valid_username_from_list. Retrieved 6/9 statements.


import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'linux'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = var_2.user()
    var_4 = '/home/'
    var_5 = '/'

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'win32'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = var_2.user()
    var_4 = 'C:\\Users\\'
    var_5 = 2
    var_6 = '\\'
    var_7 = var_3.split(var_6)[var_5]
    var_8 = 0
    var_9 = var_7[var_8]

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'linux'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = var_2.user()
    var_4 = 2
    var_5 = '/'
    var_6 = var_3.split(var_5)[var_4]

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'linux'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = var_2.user()
    var_4 = 2
    var_5 = '/'
    var_6 = var_3.split(var_5)[var_4]

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'linux'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = 10
    var_4 = range(var_3)
    var_5 = [var_2.user() for _ in var_4]
    var_6 = 2
    var_7 = '/'
    var_8 = [r.split(var_7)[var_6] for r in var_5]
    var_9 = set(var_8)
    var_10 = len(var_9)
    var_11 = bool(var_10 > 1)
    assert var_11 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_path_initialization_with_win32_platform. Retrieved 3/4 statements.
# Partially parsed test_path_initialization_with_darwin_platform. Retrieved 3/4 statements.
# Failed to parse test_path_initialization_with_random_instance.
# Partially parsed test_path_initialization_with_platform_and_random. Retrieved 1/3 statements.


import mimesis.providers.path as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Path(**var_0)
    var_2 = var_1.platform

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'linux'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = var_2.platform
    assert var_3 == 'linux'

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'win32'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = var_2.platform
    assert var_3 == 'win32'
    var_4 = var_2._pathlib_home

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'darwin'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = var_2.platform
    assert var_3 == 'darwin'
    var_4 = var_2._pathlib_home

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'freebsd10'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = var_2.platform
    assert var_3 == 'freebsd'

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 42
    var_1 = 'seed'
    var_2 = {var_1: var_0}
    var_3 = module_0.Path(**var_2)
    var_4 = var_3.seed
    assert var_4 == 42

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'linux'
    var_1 = 123
    var_2 = 'seed'
    var_3 = {var_2: var_1}
    var_4 = module_0.Path(var_0, **var_3)
    var_5 = var_4.platform
    assert var_5 == 'linux'
    var_6 = var_4.seed
    assert var_6 == 123

def test_case_0():
    var_0 = 'win64'

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.Path(**var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True



