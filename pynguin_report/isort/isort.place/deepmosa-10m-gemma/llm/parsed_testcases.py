####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_src_path_returns_firstparty_when_module_exists_in_src_path. Retrieved 9/29 statements.
# Partially parsed test_src_path_handles_nested_modules_in_namespace_package. Retrieved 12/38 statements.


import pathlib as module_0
import isort.settings as module_1
import isort.place as module_2

def test_case_0():
    var_0 = '/tmp/nonexistent_module'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = [var_3]
    var_5 = set()
    var_6 = False
    var_7 = 'py'
    var_8 = [var_7]
    var_9 = frozenset(var_8)
    var_10 = 'src_paths'
    var_11 = 'namespace_packages'
    var_12 = 'auto_identify_namespace_packages'
    var_13 = 'supported_extensions'
    var_14 = {var_10: var_4, var_11: var_5, var_12: var_6, var_13: var_9}
    var_15 = module_1.Config(**var_14)
    var_16 = 'nonexistent'
    var_17 = module_2._src_path(var_16, var_15)
    assert var_17 is None

def test_case_0():
    var_0 = 'src'
    var_1 = 'my_module.py'
    var_2 = ''
    var_3 = set()
    var_4 = False
    var_5 = 'py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)
    var_8 = 'my_module'
    var_9 = 'Found in one of the configured src_paths'

def test_case_0():
    var_0 = 'src'
    var_1 = 'my_namespace'
    var_2 = '__init__.py'
    var_3 = "__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_4 = 'sub_module'
    var_5 = ''
    var_6 = {var_1}
    var_7 = True
    var_8 = 'py'
    var_9 = [var_8]
    var_10 = frozenset(var_9)
    var_11 = 'my_namespace.sub_module'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_src_path_returns_none_when_no_src_paths_match. Retrieved 13/25 statements.
# Partially parsed test_src_path_returns_firstparty_when_module_found_in_src_path. Retrieved 9/34 statements.


import pathlib as module_0
import isort.settings as module_1
import isort.place as module_2

def test_case_0():
    var_0 = 'firstparty'
    var_1 = 'firstparty'
    var_2 = '/tmp/non_existent_src'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Path(*var_3, **var_4)
    var_6 = [var_5]
    var_7 = set()
    var_8 = False
    var_9 = 'py'
    var_10 = [var_9]
    var_11 = frozenset(var_10)
    var_12 = 'src_paths'
    var_13 = 'namespace_packages'
    var_14 = 'auto_identify_namespace_packages'
    var_15 = 'supported_extensions'
    var_16 = {var_12: var_6, var_13: var_7, var_14: var_8, var_15: var_11}
    var_17 = module_1.Config(**var_16)
    var_18 = 'my_module'
    var_19 = module_2._src_path(var_18, var_17)
    assert var_19 is None

def test_case_0():
    var_0 = 'firstparty'
    var_1 = 'src'
    var_2 = 'my_module'
    var_3 = '__init__.py'
    var_4 = set()
    var_5 = False
    var_6 = 'py'
    var_7 = [var_6]
    var_8 = frozenset(var_7)
    var_9 = 'Found in one of the configured src_paths'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_is_namespace_package_with_init_file_and_valid_content. Retrieved 14/17 statements.
# Partially parsed test_is_namespace_package_with_init_file_and_invalid_content. Retrieved 12/14 statements.
# Partially parsed test_is_namespace_package_no_init_but_contains_py_files. Retrieved 12/14 statements.
# Partially parsed test_is_namespace_package_no_init_but_contains_config_files. Retrieved 12/14 statements.


import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = var_3.touch()
    var_5 = 'py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)
    var_8 = module_1._is_namespace_package(var_3, var_7)
    assert var_8 is False
    var_9 = var_3.unlink()

import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'namespace_pkg'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = '__init__.py'
    var_7 = var_3 / var_6
    var_8 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_9 = 'py'
    var_10 = [var_9]
    var_11 = frozenset(var_10)
    var_12 = module_1._is_namespace_package(var_3, var_11)
    assert var_12 is True
    var_13 = b'__import__("pkg_resources").declare_namespace(__name__)'
    var_14 = module_1._is_namespace_package(var_3, var_11)
    assert var_14 is True
    var_15 = var_3.rmdir()

import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'regular_pkg'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = '__init__.py'
    var_7 = var_3 / var_6
    var_8 = b"print('hello')"
    var_9 = 'py'
    var_10 = [var_9]
    var_11 = frozenset(var_10)
    var_12 = module_1._is_namespace_package(var_3, var_11)
    assert var_12 is False
    var_13 = var_3.rmdir()

import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'src_pkg'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = 'module.py'
    var_7 = var_3 / var_6
    var_8 = 'py'
    var_9 = [var_8]
    var_10 = frozenset(var_9)
    var_11 = module_1._is_namespace_package(var_3, var_10)
    assert var_11 is False
    var_12 = var_3 / var_6
    var_13 = var_3.rmdir()

import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'config_pkg'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = 'pyproject.toml'
    var_7 = var_3 / var_6
    var_8 = 'py'
    var_9 = [var_8]
    var_10 = frozenset(var_9)
    var_11 = module_1._is_namespace_package(var_3, var_10)
    assert var_11 is False
    var_12 = var_3 / var_6
    var_13 = var_3.rmdir()

import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'empty_pkg'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = 'py'
    var_7 = [var_6]
    var_8 = frozenset(var_7)
    var_9 = module_1._is_namespace_package(var_3, var_8)
    assert var_9 is True
    var_10 = var_3.rmdir()



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_src_path_returns_none_when_no_src_path_matches. Retrieved 10/25 statements.
# Partially parsed test_src_path_finds_module_in_src_path. Retrieved 23/38 statements.
# Partially parsed test_src_path_handles_namespace_packages_recursion. Retrieved 25/42 statements.


import pathlib as module_0

def test_case_0():
    var_0 = 'firstparty'
    var_1 = 'firstparty'
    var_2 = '/tmp/nonexistent_src'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Path(*var_3, **var_4)
    var_6 = [var_5]
    var_7 = set()
    var_8 = False
    var_9 = 'py'
    var_10 = [var_9]
    var_11 = 'nonexistent_module'

import pathlib as module_0
import isort.settings as module_1
import isort.place as module_2

def test_case_0():
    var_0 = 'firstparty'
    var_1 = 'Sections'
    var_2 = ()
    var_3 = 'FIRSTPARTY'
    var_4 = 'firstparty'
    var_5 = {var_3: var_4}
    var_6 = '/tmp/test_src_path_unit_test'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_0.Path(*var_7, **var_8)
    var_10 = True
    var_11 = var_9.mkdir(parents=var_10, exist_ok=var_10)
    var_12 = 'my_module'
    var_13 = var_9 / var_12
    var_14 = '__init__.py'
    var_15 = var_13 / var_14
    var_16 = [var_9]
    var_17 = set()
    var_18 = False
    var_19 = 'py'
    var_20 = [var_19]
    var_21 = frozenset(var_20)
    var_22 = 'src_paths'
    var_23 = 'namespace_packages'
    var_24 = 'auto_identify_namespace_packages'
    var_25 = 'supported_extensions'
    var_26 = {var_22: var_16, var_23: var_17, var_24: var_18, var_25: var_21}
    var_27 = module_1.Config(**var_26)
    var_28 = module_2._src_path(var_12, var_27)
    var_29 = bool(var_28 == ('firstparty', f'Found in one of the configured src_paths: {var_9}.'))
    assert var_29 is True
    var_30 = [var_6]
    var_31 = {}
    var_32 = module_0.Path(*var_30, **var_31)

import pathlib as module_0
import isort.settings as module_1
import isort.place as module_2

def test_case_0():
    var_0 = 'Sections'
    var_1 = ()
    var_2 = 'FIRSTPARTY'
    var_3 = 'firstparty'
    var_4 = {var_2: var_3}
    var_5 = '/tmp/test_namespace_src'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Path(*var_6, **var_7)
    var_9 = True
    var_10 = var_8.mkdir(parents=var_9, exist_ok=var_9)
    var_11 = 'pkg'
    var_12 = var_8 / var_11
    var_13 = 'sub'
    var_14 = var_12 / var_13
    var_15 = '__init__.py'
    var_16 = var_14 / var_15
    var_17 = [var_8]
    var_18 = {var_11}
    var_19 = False
    var_20 = 'py'
    var_21 = [var_20]
    var_22 = frozenset(var_21)
    var_23 = 'src_paths'
    var_24 = 'namespace_packages'
    var_25 = 'auto_identify_namespace_packages'
    var_26 = 'supported_extensions'
    var_27 = {var_23: var_17, var_24: var_18, var_25: var_19, var_26: var_22}
    var_28 = module_1.Config(**var_27)
    var_29 = 'pkg.sub'
    var_30 = module_2._src_path(var_29, var_28)
    var_31 = [var_5]
    var_32 = {}
    var_33 = module_0.Path(*var_31, **var_32)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_is_namespace_package_evaluates_true_at_line_2. Retrieved 5/12 statements.


def test_case_0():
    var_0 = '__init__.py'
    var_1 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_forced_separate_matches_pattern_without_star_suffix_by_adding_star. Retrieved 3/5 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/path/to/dir/*'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = 'other/file.txt'
    var_5 = module_1._forced_separate(var_4, var_3)
    assert var_5 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/data/*'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = '/data/file.txt'
    var_5 = module_1._forced_separate(var_4, var_3)
    var_6 = bool(var_5 == ('/data/*', 'Matched forced_separate (/data/*) config value.'))
    assert var_6 is True

import isort.settings as module_0

def test_case_0():
    var_0 = '/usr/bin'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = '/usr/bin/python'

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/tmp/*'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = '/tmp/log.txt'
    var_5 = module_1._forced_separate(var_4, var_3)
    var_6 = bool(var_5 == ('/tmp/*', 'Matched forced_separate (/tmp/*) config value.'))
    assert var_6 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/tmp/*'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = './tmp/log.txt'
    var_5 = module_1._forced_separate(var_4, var_3)
    var_6 = bool(var_5 == ('/tmp/*', 'Matched forced_separate (/tmp/*) config value.'))
    assert var_6 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/a/*'
    var_1 = '/b/*'
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = '/a/file.txt'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('/a/*', 'Matched forced_separate (/a/*) config value.'))
    assert var_7 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_src_path_predicate_false. Retrieved 11/19 statements.


import pathlib as module_0

def test_case_0():
    var_0 = '.py'
    var_1 = '/tmp/dummy_src'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Path(*var_2, **var_3)
    var_5 = True
    var_6 = var_4.mkdir(parents=var_5, exist_ok=var_5)
    var_7 = 'a.py'
    var_8 = var_4 / var_7
    var_9 = "print('hello')"
    var_10 = 'a.b'
    var_11 = [var_4]
    var_12 = ()



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_src_path_is_module_success. Retrieved 1/9 statements.
# Partially parsed test_src_path_is_module_wrong_name. Retrieved 2/10 statements.
# Partially parsed test_src_path_is_module_not_a_directory. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'my_module'

def test_case_0():
    var_0 = 'my_module'
    var_1 = 'wrong_name'

def test_case_0():
    var_0 = 'my_module'

import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/tmp/non_existent_module_12345'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'non_existent_module_12345'
    var_5 = module_1._src_path_is_module(var_3, var_4)
    assert var_5 is False



# Parsed testcases at query #9
#--------------------------




import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'my_module'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = module_1._is_module(var_3)
    assert var_4 is True

import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'my_extension'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = module_1._is_module(var_3)
    assert var_4 is True

import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'my_package'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = module_1._is_module(var_3)
    assert var_4 is True

import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'non_existent'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = module_1._is_module(var_3)
    assert var_4 is False



# Parsed testcases at query #10
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/logs/'
    var_1 = [var_0]
    var_2 = 'data/file.txt'
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = module_1._forced_separate(var_2, var_4)
    assert var_5 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/logs/'
    var_1 = [var_0]
    var_2 = '/logs/error.log'
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = module_1._forced_separate(var_2, var_4)
    var_6 = bool(var_5 == ('/logs/', 'Matched forced_separate (/logs/) config value.'))
    assert var_6 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/tmp/*.tmp'
    var_1 = [var_0]
    var_2 = '/tmp/test.tmp'
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = module_1._forced_separate(var_2, var_4)
    var_6 = bool(var_5 == ('/tmp/*.tmp', 'Matched forced_separate (/tmp/*.tmp) config value.'))
    assert var_6 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/configs/'
    var_1 = [var_0]
    var_2 = './configs/settings.json'
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = module_1._forced_separate(var_2, var_4)
    var_6 = bool(var_5 == ('/configs/', 'Matched forced_separate (/configs/) config value.'))
    assert var_6 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/other/'
    var_1 = '/logs/'
    var_2 = [var_0, var_1]
    var_3 = '/logs/app.log'
    var_4 = {}
    var_5 = module_0.Config(**var_4)
    var_6 = module_1._forced_separate(var_3, var_5)
    var_7 = bool(var_6 == ('/logs/', 'Matched forced_separate (/logs/) config value.'))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = '/logs/app.log'
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = module_1._forced_separate(var_1, var_3)
    assert var_4 is None



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_is_module_returns_true_when_py_file_exists. Retrieved 4/6 statements.


import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/fake/path/module.py'
    var_1 = '/fake/path/module'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Path(*var_2, **var_3)
    var_5 = module_1._is_module(var_4)
    assert var_5 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_is_namespace_package_regular_dir_no_init_with_py_files. Retrieved 4/11 statements.
# Partially parsed test_is_namespace_package_regular_dir_no_init_with_config_files. Retrieved 4/9 statements.
# Partially parsed test_is_namespace_package_empty_dir_no_init. Retrieved 3/6 statements.
# Partially parsed test_is_namespace_package_init_with_pkg_resources_double_quotes. Retrieved 5/10 statements.
# Partially parsed test_is_namespace_package_init_with_pkg_resources_single_quotes. Retrieved 5/10 statements.
# Partially parsed test_is_namespace_package_init_with_pkgutil_double_quotes. Retrieved 5/10 statements.
# Partially parsed test_is_namespace_package_init_with_pkgutil_single_quotes. Retrieved 5/10 statements.
# Partially parsed test_is_namespace_package_init_with_regular_content_returns_false. Retrieved 5/10 statements.


import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = module_1._is_namespace_package(var_3, var_6)
    assert var_7 is False

def test_case_0():
    var_0 = 'module.py'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)

def test_case_0():
    var_0 = '__init__.py'
    var_1 = b'__import__("pkg_resources").declare_namespace(__name__)'
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '__init__.py'
    var_1 = b"__import__('pkg_resources').declare_namespace(__name__)"
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '__init__.py'
    var_1 = b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '__init__.py'
    var_1 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '__init__.py'
    var_1 = b"print('hello world')"
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_src_path_returns_firstparty_when_module_exists_in_src_path. Retrieved 9/37 statements.
# Partially parsed test_src_path_handles_nested_modules_in_namespace_package. Retrieved 11/41 statements.


import pathlib as module_0
import isort.settings as module_1
import isort.place as module_2

def test_case_0():
    var_0 = 'non_existent_src'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = [var_3]
    var_5 = set()
    var_6 = False
    var_7 = 'py'
    var_8 = [var_7]
    var_9 = frozenset(var_8)
    var_10 = 'src_paths'
    var_11 = 'namespace_packages'
    var_12 = 'auto_identify_namespace_packages'
    var_13 = 'supported_extensions'
    var_14 = {var_10: var_4, var_11: var_5, var_12: var_6, var_13: var_9}
    var_15 = module_1.Config(**var_14)
    var_16 = 'non_existent_module'
    var_17 = module_2._src_path(var_16, var_15)
    assert var_17 is None

def test_case_0():
    var_0 = 'src'
    var_1 = 'my_module'
    var_2 = '__init__.py'
    var_3 = set()
    var_4 = False
    var_5 = 'py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)
    var_8 = 'sections'
    var_9 = 'Found in one of the configured src_paths'

def test_case_0():
    var_0 = 'src'
    var_1 = 'my_pkg'
    var_2 = 'sub_module'
    var_3 = '__init__.py'
    var_4 = {var_1}
    var_5 = False
    var_6 = 'py'
    var_7 = [var_6]
    var_8 = frozenset(var_7)
    var_9 = 'sections'
    var_10 = 'my_pkg.sub_module'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_src_path_predicate_false_by_namespace_not_in_config. Retrieved 9/19 statements.


import pathlib as module_0

def test_case_0():
    var_0 = '.py'
    var_1 = False
    var_2 = '/tmp/src'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Path(*var_3, **var_4)
    var_6 = [var_5]
    var_7 = 'root.submodule'
    var_8 = ()
    var_9 = True
    var_10 = var_5.mkdir(parents=var_9, exist_ok=var_9)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_src_path_predicate_true_via_namespace_packages. Retrieved 16/28 statements.


import pathlib as module_0

def test_case_0():
    var_0 = 'my_module'
    var_1 = '/tmp/src'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Path(*var_2, **var_3)
    var_5 = '/tmp'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Path(*var_6, **var_7)
    var_9 = True
    var_10 = var_8.mkdir(exist_ok=var_9)
    var_11 = 'my_module.sub'
    var_12 = [var_8]
    var_13 = ()
    var_14 = '/tmp/test_module_dir'
    var_15 = [var_14]
    var_16 = {}
    var_17 = module_0.Path(*var_15, **var_16)
    var_18 = var_17.mkdir(exist_ok=var_9)
    var_19 = [var_14]
    var_20 = {}
    var_21 = module_0.Path(*var_19, **var_20)
    var_22 = '/tmp/src_module'
    var_23 = [var_22]
    var_24 = {}
    var_25 = module_0.Path(*var_23, **var_24)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_src_path_returns_none_when_no_match_found. Retrieved 8/19 statements.
# Partially parsed test_src_path_returns_firstparty_when_module_exists_in_src_path. Retrieved 8/29 statements.


import pathlib as module_0

def test_case_0():
    var_0 = '/tmp/src'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = [var_3]
    var_5 = set()
    var_6 = False
    var_7 = 'py'
    var_8 = [var_7]
    var_9 = 'nonexistent.module'

def test_case_0():
    var_0 = 'src'
    var_1 = 'my_module'
    var_2 = '__init__.py'
    var_3 = set()
    var_4 = False
    var_5 = 'py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)
    var_8 = 'Found in one of the configured src_paths'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_is_module_true_when_py_file_exists. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'test_module.py'
    var_1 = ''
    var_2 = 'pathlib.Path.exists'
    var_3 = True
    var_4 = 'your_module.exists_case_sensitive'
    var_5 = 'test_module'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_is_module_logic_with_py_exists. Retrieved 4/10 statements.
# Partially parsed test_is_module_logic_with_extension_exists. Retrieved 5/12 statements.
# Partially parsed test_is_module_logic_with_init_exists. Retrieved 6/10 statements.
# Partially parsed test_is_module_returns_false_when_nothing_exists. Retrieved 3/7 statements.


import pathlib as module_0

def test_case_0():
    var_0 = '/fake/path/module'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)

import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/fake/path/module'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = '.py'
    var_5 = module_1._is_module(var_3)
    assert var_5 is True

import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/fake/path/module'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 0
    var_5 = str(var_0)
    var_6 = module_1._is_module(var_3)
    assert var_6 is True

import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/fake/path/package'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = '__init__.py'
    var_5 = var_3 / var_4
    var_6 = str(var_5)
    var_7 = module_1._is_module(var_3)
    assert var_7 is True

import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/fake/path/module'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = module_1._is_module(var_3)
    assert var_4 is False



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_src_path_returns_none_when_no_match_found. Retrieved 10/25 statements.
# Partially parsed test_src_path_returns_firstparty_when_module_found_in_src_path. Retrieved 8/30 statements.
# Partially parsed test_src_path_handles_nested_namespace_packages. Retrieved 11/35 statements.


import pathlib as module_0

def test_case_0():
    var_0 = 'firstparty'
    var_1 = 'firstparty'
    var_2 = '/tmp/src'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Path(*var_3, **var_4)
    var_6 = [var_5]
    var_7 = set()
    var_8 = False
    var_9 = 'py'
    var_10 = [var_9]
    var_11 = 'non_existent_module'

def test_case_0():
    var_0 = 'firstparty'
    var_1 = 'my_module'
    var_2 = '__init__.py'
    var_3 = set()
    var_4 = False
    var_5 = 'py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)
    var_8 = 'Found in one of the configured src_paths'

def test_case_0():
    var_0 = 'firstparty'
    var_1 = 'pkg'
    var_2 = 'subpkg'
    var_3 = '__init__.py'
    var_4 = b"__import__('pkg_resources').declare_namespace(__name__)"
    var_5 = {var_1}
    var_6 = True
    var_7 = 'py'
    var_8 = [var_7]
    var_9 = frozenset(var_8)
    var_10 = 'pkg.subpkg'
    var_11 = 'Found in one of the configured src_paths'



# Parsed testcases at query #3
#--------------------------




import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/fake/path/my_module'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'my_module'
    var_5 = module_1._src_path_is_module(var_3, var_4)
    assert var_5 is True

import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/fake/path/my_module'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'my_module'
    var_5 = module_1._src_path_is_module(var_3, var_4)
    assert var_5 is False

import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/fake/path/my_module.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'my_module'
    var_5 = module_1._src_path_is_module(var_3, var_4)
    assert var_5 is False

import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/fake/path/my_module'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'my_module'
    var_5 = module_1._src_path_is_module(var_3, var_4)
    assert var_5 is False



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_src_path_evaluates_true_at_line_26. Retrieved 9/19 statements.


import pathlib as module_0

def test_case_0():
    var_0 = '/tmp/src'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = [var_3]
    var_5 = 'my_module'
    var_6 = {var_5}
    var_7 = True
    var_8 = '.py'
    var_9 = {var_8}
    var_10 = 'my_module.sub_module'
    var_11 = '/tmp/src'
    var_12 = [var_11]
    var_13 = {}
    var_14 = module_0.Path(*var_12, **var_13)
    var_15 = [var_14]
    var_16 = ()
    var_17 = bool(True)
    assert var_17 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_src_path_returns_none_when_no_match_found. Retrieved 8/19 statements.
# Partially parsed test_src_path_returns_firstparty_when_module_found_in_src_path. Retrieved 5/18 statements.
# Partially parsed test_src_path_handles_nested_namespace_packages. Retrieved 16/27 statements.


import pathlib as module_0

def test_case_0():
    var_0 = '/tmp/non_existent_path_12345'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = [var_3]
    var_5 = set()
    var_6 = False
    var_7 = 'py'
    var_8 = [var_7]
    var_9 = 'non_existent_module'

def test_case_0():
    var_0 = set()
    var_1 = False
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)
    var_5 = 'Found in one of the configured src_paths'

import pathlib as module_0
import isort.settings as module_1
import isort.place as module_2

def test_case_0():
    var_0 = '/tmp/namespace_test'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(parents=var_4, exist_ok=var_4)
    var_6 = 'my_namespace'
    var_7 = var_3 / var_6
    var_8 = [var_3]
    var_9 = {var_6}
    var_10 = False
    var_11 = 'py'
    var_12 = [var_11]
    var_13 = frozenset(var_12)
    var_14 = 'src_paths'
    var_15 = 'namespace_packages'
    var_16 = 'auto_identify_namespace_packages'
    var_17 = 'supported_extensions'
    var_18 = {var_14: var_8, var_15: var_9, var_16: var_10, var_17: var_13}
    var_19 = module_1.Config(**var_18)
    var_20 = 'my_namespace.sub_module'
    var_21 = (var_6,)
    var_22 = module_2._src_path(var_20, var_19, prefix=var_21)
    assert var_22 is None



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_forced_separate_dot_prefix_match. Retrieved 6/8 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/a/b/'
    var_1 = [var_0]
    var_2 = 'test_file.txt'
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = module_1._forced_separate(var_2, var_4)
    assert var_5 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/a/b/*'
    var_1 = [var_0]
    var_2 = '/a/b/c.txt'
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = module_1._forced_separate(var_2, var_4)
    var_6 = bool(var_5 == ('/a/mock/b/*', 'Matched forced_separate (/a/b/*) config value.'))
    assert var_6 is True
    var_7 = {}
    var_8 = module_0.Config(**var_7)
    var_9 = module_1._forced_separate(var_2, var_8)
    var_10 = bool(var_9 == ('/a/b/*', 'Matched forced_separate (/a/b/*) config value.'))
    assert var_10 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/a/b'
    var_1 = [var_0]
    var_2 = '/a/b/c.txt'
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = module_1._forced_separate(var_2, var_4)
    var_6 = bool(var_5 == ('/a/b', 'Matched forced_separate (/a/b) config value.'))
    assert var_6 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/a/b'
    var_1 = [var_0]
    var_2 = '/a/b/c.txt'
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = './a/b/c.txt'
    var_6 = {}
    var_7 = module_0.Config(**var_6)
    var_8 = module_1._forced_separate(var_5, var_7)
    var_9 = bool(var_8 == ('/a/b', 'Matched forced_separate (/a/b) config value.'))
    assert var_9 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/other/'
    var_1 = '/a/b'
    var_2 = [var_0, var_1]
    var_3 = '/a/b/c.txt'
    var_4 = {}
    var_5 = module_0.Config(**var_4)
    var_6 = module_1._forced_separate(var_3, var_5)
    var_7 = bool(var_6 == ('/a/b', 'Matched forced_separate (/a/b) config value.'))
    assert var_7 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_known_pattern_returns_match_when_pattern_exists_in_config_sections. Retrieved 5/10 statements.
# Partially parsed test_known_pattern_returns_none_when_pattern_does_not_match_name. Retrieved 4/9 statements.
# Partially parsed test_known_pattern_returns_none_when_placement_not_in_sections. Retrieved 5/10 statements.
# Partially parsed test_known_pattern_checks_hierarchical_parts_from_longest_to_shortest. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'utils.*'
    var_1 = 'utility_section'
    var_2 = 'other_section'
    var_3 = [var_1, var_2]
    var_4 = 'utils.helpers.string'

def test_case_0():
    var_0 = 'utils.*'
    var_1 = 'utility_section'
    var_2 = [var_1]
    var_3 = 'core.logic'

def test_case_0():
    var_0 = 'utils.*'
    var_1 = 'utility_section'
    var_2 = 'other_section'
    var_3 = [var_2]
    var_4 = 'utils.helpers'

def test_case_0():
    var_0 = 'a.b'
    var_1 = 'section_b'
    var_2 = [var_1]

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'known_patterns'
    var_3 = 'sections'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = ''
    var_7 = module_1._known_pattern(var_6, var_5)
    assert var_7 is None



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_is_namespace_package_with_init_and_pkg_resources. Retrieved 12/14 statements.
# Partially parsed test_is_namespace_package_with_init_and_pkgutil. Retrieved 12/14 statements.
# Partially parsed test_is_namespace_package_with_init_but_no_namespace_declaration. Retrieved 12/14 statements.
# Partially parsed test_is_namespace_package_no_init_but_contains_src_files. Retrieved 13/15 statements.
# Partially parsed test_is_namespace_package_no_init_but_contains_pyproject. Retrieved 12/15 statements.


import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'content'
    var_5 = var_3.write_text(var_4)
    var_6 = 'py'
    var_7 = [var_6]
    var_8 = frozenset(var_7)
    var_9 = module_1._is_namespace_package(var_3, var_8)
    assert var_9 is False
    var_10 = var_3.unlink()

import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test_pkg_res'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = '__init__.py'
    var_7 = var_3 / var_6
    var_8 = b"__import__('pkg_resources').declare_namespace(__name__)"
    var_9 = 'py'
    var_10 = [var_9]
    var_11 = frozenset(var_10)
    var_12 = module_1._is_namespace_package(var_3, var_11)
    assert var_12 is True
    var_13 = var_3.rmdir()

import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test_pkg_util'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = '__init__.py'
    var_7 = var_3 / var_6
    var_8 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_9 = 'py'
    var_10 = [var_9]
    var_11 = frozenset(var_10)
    var_12 = module_1._is_namespace_package(var_3, var_11)
    assert var_12 is True
    var_13 = var_3.rmdir()

import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test_regular_pkg'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = '__init__.py'
    var_7 = var_3 / var_6
    var_8 = b"print('hello')"
    var_9 = 'py'
    var_10 = [var_9]
    var_11 = frozenset(var_10)
    var_12 = module_1._is_namespace_package(var_3, var_11)
    assert var_12 is False
    var_13 = var_3.rmdir()

import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test_namespace_with_files'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = 'module.py'
    var_7 = var_3 / var_6
    var_8 = 'content'
    var_9 = 'py'
    var_10 = [var_9]
    var_11 = frozenset(var_10)
    var_12 = module_1._is_namespace_package(var_3, var_11)
    assert var_12 is False
    var_13 = var_3 / var_6
    var_14 = var_3.rmdir()

import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test_empty_namespace'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = 'py'
    var_7 = [var_6]
    var_8 = frozenset(var_7)
    var_9 = module_1._is_namespace_package(var_3, var_8)
    assert var_9 is True
    var_10 = var_3.rmdir()

import pathlib as module_0

def test_case_0():
    var_0 = 'test_pyproject_pkg'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = 'pyproject.toml'
    var_7 = var_3 / var_6
    var_8 = ''
    var_9 = 'py'
    var_10 = [var_9]
    var_11 = frozenset(var_10)
    var_12 = var_3 / var_6
    var_13 = var_3.rmdir()



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_is_namespace_package_evaluates_true_at_line_6_when_init_exists. Retrieved 5/12 statements.


def test_case_0():
    var_0 = '__init__.py'
    var_1 = "__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)



# Parsed testcases at query #10
#--------------------------




import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'module.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = module_1._is_module(var_3)
    assert var_4 is True



# Parsed testcases at query #11
#--------------------------




import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/fake/path/my_module'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'my_module'
    var_5 = module_1._src_path_is_module(var_3, var_4)
    assert var_5 is True

import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/fake/path/wrong_name'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'my_module'
    var_5 = module_1._src_path_is_module(var_3, var_4)
    assert var_5 is False

import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/fake/path/my_module.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'my_module'
    var_5 = module_1._src_path_is_module(var_3, var_4)
    assert var_5 is False

import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/fake/path/my_module'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'my_module'
    var_5 = module_1._src_path_is_module(var_3, var_4)
    assert var_5 is False



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_is_namespace_package_evaluates_true_on_line_4. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = '__init__.py'
    var_4 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"



# Parsed testcases at query #13
#--------------------------




import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/fake/path/module'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = module_1._is_module(var_3)
    assert var_4 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_src_path_predicate_true. Retrieved 8/31 statements.


def test_case_0():
    var_0 = ()
    var_1 = 'my_module'
    var_2 = 'my_module.submodule'
    var_3 = 'my_module'
    var_4 = 'test_predicate_dir'
    var_5 = True
    var_6 = 'my_module.sub'
    var_7 = ()



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_is_namespace_package_is_dir_but_no_init_and_has_src_files. Retrieved 5/12 statements.
# Partially parsed test_is_namespace_package_is_dir_but_no_init_and_has_config_files. Retrieved 5/11 statements.
# Partially parsed test_is_namespace_package_with_valid_pkg_resources_init. Retrieved 5/11 statements.
# Partially parsed test_is_namespace_package_with_valid_pkgutil_init. Retrieved 5/11 statements.
# Partially parsed test_is_namespace_package_with_invalid_init_content. Retrieved 5/11 statements.
# Partially parsed test_is_namespace_package_empty_dir_with_init_is_true. Retrieved 5/11 statements.


import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = module_1._is_namespace_package(var_3, var_6)
    assert var_7 is False

def test_case_0():
    var_0 = 'module.py'
    var_1 = ''
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = ''
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '__init__.py'
    var_1 = "__import__('pkg_resources').declare_namespace(__name__)"
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '__init__.py'
    var_1 = "__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '__init__.py'
    var_1 = "print('hello')"
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '__init__.py'
    var_1 = '__import__("pkg_resources").declare_namespace(__name__)'
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)



# Parsed testcases at query #16
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test_pattern'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'test_pattern_suffix'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 is not None)
    assert var_7 is True
    var_8 = var_6[0]
    assert var_8 == 'test_pattern'



# Parsed testcases at query #17
#--------------------------




import pathlib as module_0
import isort.settings as module_1
import isort.place as module_2

def test_case_0():
    var_0 = '/tmp/src'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = [var_3]
    var_5 = set()
    var_6 = False
    var_7 = 'py'
    var_8 = [var_7]
    var_9 = 'src_paths'
    var_10 = 'namespace_packages'
    var_11 = 'auto_identify_namespace_packages'
    var_12 = 'supported_extensions'
    var_13 = {var_9: var_4, var_10: var_5, var_11: var_6, var_12: var_8}
    var_14 = module_1.Config(**var_13)
    var_15 = 'root_module'
    var_16 = [var_0]
    var_17 = {}
    var_18 = module_0.Path(*var_16, **var_17)
    var_19 = [var_18]
    var_20 = ()
    var_21 = module_2._src_path(var_15, var_14, var_19, var_20)
    var_22 = bool(var_21 is not None)
    assert var_22 is True



# Parsed testcases at query #18
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test_pattern'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'test_pattern_suffix'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 is not None)
    assert var_7 is True



# Parsed testcases at query #19
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test_pattern'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'test_pattern_suffix'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 is not None)
    assert var_7 is True
    var_8 = var_6[0]
    assert var_8 == 'test_pattern'
    var_9 = 'Matched'
    var_10 = bool('Matched' in var_6[1])
    assert var_10 is True



# Parsed testcases at query #20
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test_pattern'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'test_pattern_suffix'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 is not None)
    assert var_7 is True
    var_8 = var_6[0]
    assert var_8 == 'test_pattern'
    var_9 = 'Matched'
    var_10 = bool('Matched' in var_6[1])
    assert var_10 is True



