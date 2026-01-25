####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_path_initialization_default_platform. Retrieved 4/6 statements.
# Partially parsed test_path_initialization_custom_platform_linux. Retrieved 5/6 statements.
# Partially parsed test_path_initialization_custom_platform_windows. Retrieved 5/6 statements.
# Partially parsed test_path_initialization_freebsd_platform. Retrieved 5/6 statements.
# Partially parsed test_path_initialization_with_seed. Retrieved 2/3 statements.
# Failed to parse test_path_initialization_with_custom_random.


import mimesis.providers.path as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Path(**var_0)
    var_2 = var_1.platform
    var_3 = var_1._pathlib_home
    var_4 = var_1._pathlib_home
    var_5 = str(var_4)

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
    var_0 = 'freebsd12'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = var_2.platform
    assert var_3 == 'freebsd'
    var_4 = var_2._pathlib_home
    var_5 = var_2._pathlib_home
    var_6 = str(var_5)

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
    var_0 = 'not_a_random_object'
    var_1 = 'random'
    var_2 = {var_1: var_0}
    var_3 = module_0.Path(**var_2)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_user_linux_platform. Retrieved 7/9 statements.
# Partially parsed test_user_windows_platform. Retrieved 7/9 statements.
# Partially parsed test_user_freebsd_platform. Retrieved 7/9 statements.


import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'linux'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = var_2.user()
    var_4 = '/home/'
    var_5 = -1
    var_6 = '/'
    var_7 = var_3.split(var_6)[var_5]

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'win32'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = var_2.user()
    var_4 = '\\Users\\'
    var_5 = -1
    var_6 = '\\'
    var_7 = var_3.split(var_6)[var_5]

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'freebsd'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = var_2.user()
    var_4 = '/home/'
    var_5 = -1
    var_6 = '/'
    var_7 = var_3.split(var_6)[var_5]



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_user_linux_platform. Retrieved 7/9 statements.
# Partially parsed test_user_windows_platform. Retrieved 7/9 statements.
# Partially parsed test_user_freebsd_platform. Retrieved 7/9 statements.


import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'linux'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = var_2.user()
    var_4 = '/home/'
    var_5 = -1
    var_6 = '/'
    var_7 = var_3.split(var_6)[var_5]

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'win32'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = var_2.user()
    var_4 = '\\Users\\'
    var_5 = -1
    var_6 = '\\'
    var_7 = var_3.split(var_6)[var_5]

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'freebsd'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = var_2.user()
    var_4 = '/home/'
    var_5 = -1
    var_6 = '/'
    var_7 = var_3.split(var_6)[var_5]



