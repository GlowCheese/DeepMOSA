####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_path_initialization_with_win32_platform. Retrieved 3/4 statements.
# Partially parsed test_path_initialization_with_darwin_platform. Retrieved 3/4 statements.
# Partially parsed test_path_initialization_with_win64_platform. Retrieved 3/4 statements.
# Failed to parse test_path_initialization_with_random_instance.
# Partially parsed test_path_initialization_home_path_for_linux. Retrieved 5/9 statements.
# Partially parsed test_path_initialization_home_path_for_win32. Retrieved 5/9 statements.


import mimesis.providers.path as module_0


def test_case_0():
    var_0 = {}
    var_1 = module_0.Path(**var_0)
    var_2 = var_1.platform


def test_case_0():
    var_0 = 'linux'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = var_2.platform
    assert var_3 == 'linux'


def test_case_0():
    var_0 = 'win32'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = var_2.platform
    assert var_3 == 'win32'
    var_4 = var_2._pathlib_home


def test_case_0():
    var_0 = 'darwin'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = var_2.platform
    assert var_3 == 'darwin'
    var_4 = var_2._pathlib_home


def test_case_0():
    var_0 = 'freebsd10'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = var_2.platform
    assert var_3 == 'freebsd'


def test_case_0():
    var_0 = 'win64'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = var_2.platform
    assert var_3 == 'win64'
    var_4 = var_2._pathlib_home


def test_case_0():
    var_0 = 42
    var_1 = 'seed'
    var_2 = {var_1: var_0}
    var_3 = module_0.Path(**var_2)
    var_4 = var_3.seed
    assert var_4 == 42


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
    var_0 = 'linux'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = []
    var_4 = 'home'
    var_5 = var_2._pathlib_home
    var_6 = str(var_5)


def test_case_0():
    var_0 = 'win32'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = []
    var_4 = 'home'
    var_5 = var_2._pathlib_home
    var_6 = str(var_5)



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_path_initialization_with_windows_platform. Retrieved 3/4 statements.
# Partially parsed test_path_initialization_with_posix_platform. Retrieved 3/4 statements.
# Failed to parse test_path_initialization_with_random_instance.
# Partially parsed test_path_initialization_home_path_set_correctly. Retrieved 5/6 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Path(**var_0)
    var_2 = var_1.platform


def test_case_0():
    var_0 = 'linux'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = var_2.platform
    assert var_3 == 'linux'


def test_case_0():
    var_0 = 'freebsd10'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = var_2.platform
    assert var_3 == 'freebsd'


def test_case_0():
    var_0 = 'win32'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = var_2.platform
    assert var_3 == 'win32'
    var_4 = var_2._pathlib_home


def test_case_0():
    var_0 = 'darwin'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = var_2.platform
    assert var_3 == 'darwin'
    var_4 = var_2._pathlib_home


def test_case_0():
    var_0 = 42
    var_1 = 'seed'
    var_2 = {var_1: var_0}
    var_3 = module_0.Path(**var_2)
    var_4 = var_3.seed
    assert var_4 == 42


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
    var_0 = 'linux'
    var_1 = {}
    var_2 = module_0.Path(var_0, **var_1)
    var_3 = 'home'
    var_4 = var_2._pathlib_home
    var_5 = str(var_4)



