####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_path_constructor_linux_platform. Retrieved 3/4 statements.
# Partially parsed test_path_constructor_darwin_platform. Retrieved 3/4 statements.
# Partially parsed test_path_constructor_win32_platform. Retrieved 3/4 statements.
# Partially parsed test_path_constructor_win64_platform. Retrieved 3/4 statements.
# Partially parsed test_path_constructor_freebsd_platform. Retrieved 3/4 statements.
# Partially parsed test_path_constructor_with_custom_random. Retrieved 1/3 statements.


import mimesis.providers.path as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Path(**var_0)
    var_2 = var_1.platform
    var_3 = var_1.random
    var_4 = bool(var_1.random is not None)
    assert var_4 is True

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'linux'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = var_2.platform
    assert var_3 == 'linux'
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
    var_0 = 'win32'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = var_2.platform
    assert var_3 == 'win32'
    var_4 = var_2._pathlib_home

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'win64'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = var_2.platform
    assert var_3 == 'win64'
    var_4 = var_2._pathlib_home

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'freebsd11'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = var_2.platform
    assert var_3 == 'freebsd'
    var_4 = var_2._pathlib_home

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'linux'
    var_1 = 42
    var_2 = 'seed'
    var_3 = {var_2: var_1}
    var_4 = module_0.Path(var_0, **var_3)
    var_5 = var_4.platform
    assert var_5 == 'linux'
    var_6 = var_4.seed
    assert var_6 == 42

def test_case_0():
    var_0 = 'linux'

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Path(**var_0)
    var_2 = var_1.Meta.name
    assert var_2 == 'path'

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'linux'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = var_2._pathlib_home
    var_4 = str(var_3)
    assert var_4 == '/home'

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'win32'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = var_2._pathlib_home
    var_4 = str(var_3)
    var_5 = 'Users'
    var_6 = bool('Users' in var_4)
    assert var_6 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_path_constructor_linux_platform. Retrieved 3/4 statements.
# Partially parsed test_path_constructor_darwin_platform. Retrieved 3/4 statements.
# Partially parsed test_path_constructor_win32_platform. Retrieved 3/4 statements.
# Partially parsed test_path_constructor_win64_platform. Retrieved 3/4 statements.
# Partially parsed test_path_constructor_freebsd_platform. Retrieved 3/4 statements.
# Partially parsed test_path_constructor_with_custom_random. Retrieved 1/3 statements.
# Partially parsed test_path_constructor_meta_name. Retrieved 3/4 statements.


import mimesis.providers.path as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Path(**var_0)
    var_2 = var_1.platform
    var_3 = var_1.seed
    var_4 = var_1.random
    var_5 = bool(var_1.random is not None)
    assert var_5 is True

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'linux'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = var_2.platform
    assert var_3 == 'linux'
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
    var_0 = 'win32'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = var_2.platform
    assert var_3 == 'win32'
    var_4 = var_2._pathlib_home

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'win64'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = var_2.platform
    assert var_3 == 'win64'
    var_4 = var_2._pathlib_home

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'freebsd11'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = var_2.platform
    assert var_3 == 'freebsd'
    var_4 = var_2._pathlib_home

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'linux'
    var_1 = 42
    var_2 = 'seed'
    var_3 = {var_2: var_1}
    var_4 = module_0.Path(var_0, **var_3)
    var_5 = var_4.platform
    assert var_5 == 'linux'
    var_6 = var_4.seed
    assert var_6 == 42

def test_case_0():
    var_0 = 'linux'

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'linux'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = var_2._pathlib_home
    var_4 = str(var_3)
    var_5 = 'home'
    var_6 = bool('home' in var_4)
    assert var_6 is True

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'win32'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = var_2._pathlib_home
    var_4 = str(var_3)
    var_5 = 'Users'
    var_6 = bool('Users' in var_4)
    assert var_6 is True

import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'linux'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = 'Meta'



