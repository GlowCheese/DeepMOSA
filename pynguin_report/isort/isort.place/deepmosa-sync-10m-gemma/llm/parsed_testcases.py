####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_src_path_returns_none_when_no_match. Retrieved 8/14 statements.
# Partially parsed test_src_path_returns_firstparty_when_module_exists_as_dir. Retrieved 14/31 statements.
# Partially parsed test_src_path_identifies_namespace_package. Retrieved 16/33 statements.
# Partially parsed test_src_path_handles_single_file_module. Retrieved 10/26 statements.


import pathlib as module_0

def test_case_0():
    var_0 = '/tmp/nonexistent_module'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = 'my_module'
    var_7 = '/tmp/fake'
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_0.Path(*var_8, **var_9)
    var_11 = [var_10]

import pathlib as module_0

def test_case_0():
    var_0 = '/tmp/src'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = 'my_module'
    var_7 = var_0 / var_6
    var_8 = 'submodule'
    var_9 = var_7 / var_8
    var_10 = '__init__.py'
    var_11 = var_9 / var_10
    var_12 = True
    var_13 = lambda x: var_12
    var_14 = 'my_module.submodule'
    var_15 = [var_7]
    var_16 = 'Found in one of the configured src_paths'

import pathlib as module_0

def test_case_0():
    var_0 = '/tmp/src'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'my_module'
    var_5 = 'py'
    var_6 = [var_5]
    var_7 = 'my_module'
    var_8 = var_0 / var_7
    var_9 = 'submodule'
    var_10 = var_8 / var_9
    var_11 = '__init__.py'
    var_12 = var_10 / var_11
    var_13 = "__import__('pkg_resources').declare_namespace(__name__)"
    var_14 = True
    var_15 = lambda x: var_14
    var_16 = 'my_module.submodule'
    var_17 = [var_8]
    var_18 = 'Found in one of the configured src_paths'

import pathlib as module_0

def test_case_0():
    var_0 = '/tmp/src'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = 'standalone.py'
    var_7 = var_0 / var_6
    var_8 = True
    var_9 = lambda x: var_8
    var_10 = 'standalone'
    var_11 = [var_6]



# Parsed testcases at query #2
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_is_module_evaluates_true_when_py_file_exists. Retrieved 4/9 statements.


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



# Parsed testcases at query #4
#--------------------------




import pathlib as module_0
import isort.settings as module_1
import isort.place as module_2

def test_case_0():
    var_0 = '/tmp/dummy_src'
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
    var_15 = 'dummy'
    var_16 = [var_0]
    var_17 = {}
    var_18 = module_0.Path(*var_16, **var_17)
    var_19 = [var_18]
    var_20 = module_2._src_path(var_15, var_14, var_19)
    var_21 = bool(var_20 is not None or True)
    assert var_21 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_src_path_is_module_true. Retrieved 1/7 statements.
# Partially parsed test_src_path_is_module_false_name_mismatch. Retrieved 1/7 statements.
# Partially parsed test_src_path_is_module_false_not_a_directory. Retrieved 1/7 statements.
# Partially parsed test_src_path_is_module_false_not_exists. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'my_module'

def test_case_0():
    var_0 = 'my_module'

def test_case_0():
    var_0 = 'my_module'

def test_case_0():
    var_0 = 'my_module'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_known_pattern_returns_none_when_no_match. Retrieved 4/9 statements.
# Partially parsed test_known_pattern_returns_none_when_placement_not_in_sections. Retrieved 4/9 statements.
# Partially parsed test_known_pattern_matches_exact_name. Retrieved 3/8 statements.
# Partially parsed test_known_pattern_matches_parent_module. Retrieved 4/10 statements.
# Partially parsed test_known_pattern_matches_longest_possible_prefix. Retrieved 6/13 statements.
# Partially parsed test_known_pattern_with_multiple_dots. Retrieved 4/9 statements.
# Partially parsed test_known_pattern_no_match_due_to_regex_mismatch. Retrieved 4/9 statements.
# Partially parsed test_known_pattern_matches_single_part_name. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'abc'
    var_1 = 'section1'
    var_2 = [var_1]
    var_3 = 'xyz.def'

def test_case_0():
    var_0 = 'abc'
    var_1 = 'section2'
    var_2 = 'section1'
    var_3 = [var_2]

def test_case_0():
    var_0 = 'abc'
    var_1 = 'section1'
    var_2 = [var_1]

def test_case_0():
    var_0 = 'a\\.b'
    var_1 = 'section1'
    var_2 = [var_1]
    var_3 = 'a.b.c'

def test_case_0():
    var_0 = 'a\\.b'
    var_1 = 'section_short'
    var_2 = 'a\\.b\\.c'
    var_3 = 'section_long'
    var_4 = [var_1, var_3]
    var_5 = 'a.b.c'

def test_case_0():
    var_0 = 'pkg\\.sub'
    var_1 = 'sec'
    var_2 = [var_1]
    var_3 = 'pkg.sub.module'

def test_case_0():
    var_0 = 'different\\.path'
    var_1 = 'sec'
    var_2 = [var_1]
    var_3 = 'pkg.sub'

def test_case_0():
    var_0 = 'root'
    var_1 = 'sec'
    var_2 = [var_1]



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_src_path_is_module_evaluates_to_true. Retrieved 2/15 statements.


def test_case_0():
    var_0 = True
    var_1 = 'my_module'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_src_path_predicate_true_namespace_in_config. Retrieved 16/25 statements.


import pathlib as module_0
import isort.settings as module_1
import isort.place as module_2

def test_case_0():
    var_0 = '/tmp/src'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = [var_3]
    var_5 = 'my_package'
    var_6 = {var_5}
    var_7 = False
    var_8 = '.py'
    var_9 = [var_8]
    var_10 = 'src_paths'
    var_11 = 'namespace_packages'
    var_12 = 'auto_identify_namespace_packages'
    var_13 = 'supported_extensions'
    var_14 = {var_10: var_4, var_11: var_6, var_12: var_7, var_13: var_9}
    var_15 = module_1.Config(**var_14)
    var_16 = '/tmp/src/my_package'
    var_17 = tuple()
    var_18 = 'my_package.submodule'
    var_19 = [var_0]
    var_20 = {}
    var_21 = module_0.Path(*var_19, **var_20)
    var_22 = [var_21]
    var_23 = ()
    var_24 = module_2._src_path(var_18, var_15, var_22, var_23)
    var_25 = bool(var_24 is not None)
    assert var_25 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_is_namespace_package_regular_package_with_init. Retrieved 11/14 statements.
# Partially parsed test_is_namespace_package_regular_package_with_pkgutil. Retrieved 11/14 statements.
# Partially parsed test_is_namespace_package_regular_package_with_invalid_init. Retrieved 11/14 statements.
# Partially parsed test_is_namespace_package_namespace_without_init_but_has_src_files. Retrieved 9/14 statements.
# Partially parsed test_is_namespace_package_namespace_without_init_and_no_src_files. Retrieved 8/10 statements.
# Partially parsed test_is_namespace_package_namespace_without_init_with_config_files. Retrieved 11/14 statements.
# Partially parsed test_is_namespace_package_namespace_without_init_with_setup_cfg. Retrieved 11/14 statements.


import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test_file.py'
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
    var_0 = 'test_pkg'
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

import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test_pkg_pkgutil'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = '__init__.py'
    var_7 = var_3 / var_6
    var_8 = b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    var_9 = 'py'
    var_10 = [var_9]
    var_11 = frozenset(var_10)
    var_12 = module_1._is_namespace_package(var_3, var_11)
    assert var_12 is True

import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test_pkg_invalid'
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

import pathlib as module_0

def test_case_0():
    var_0 = 'test_ns_with_files'
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

import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test_ns_empty'
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

import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test_ns_config'
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
    var_12 = module_1._is_namespace_package(var_3, var_11)
    assert var_12 is False

import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test_ns_setup_cfg'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = 'setup.cfg'
    var_7 = var_3 / var_6
    var_8 = ''
    var_9 = 'py'
    var_10 = [var_9]
    var_11 = frozenset(var_10)
    var_12 = module_1._is_namespace_package(var_3, var_11)
    assert var_12 is False



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_src_path_handles_nested_namespace_recursion. Retrieved 13/23 statements.


import pathlib as module_0
import isort.settings as module_1
import isort.place as module_2

def test_case_0():
    var_0 = 'non_existent_path'
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

import pathlib as module_0
import isort.settings as module_1
import isort.place as module_2

def test_case_0():
    var_0 = '.'
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
    var_16 = 'test_module'
    var_17 = module_2._src_path(var_16, var_15)
    var_18 = var_17[0]
    assert var_18 == 'FIRSTPARTY'
    var_19 = 'Found in one of the configured src_paths'
    var_20 = bool('Found in one of the configured src_paths' in var_17[1])
    assert var_20 is True

import pathlib as module_0
import isort.settings as module_1
import isort.place as module_2

def test_case_0():
    var_0 = '.'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = [var_3]
    var_5 = 'my_namespace'
    var_6 = {var_5}
    var_7 = False
    var_8 = 'py'
    var_9 = [var_8]
    var_10 = frozenset(var_9)
    var_11 = 'src_paths'
    var_12 = 'namespace_packages'
    var_13 = 'auto_identify_namespace_packages'
    var_14 = 'supported_extensions'
    var_15 = {var_11: var_4, var_12: var_6, var_13: var_7, var_14: var_10}
    var_16 = module_1.Config(**var_15)
    var_17 = 'my_namespace.submodule'
    var_18 = module_2._src_path(var_17, var_16)
    assert var_18 is None
    var_19 = bool(var_0)
    assert var_19 is True
    var_20 = len(var_18)
    assert var_20 == 2



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_forced_separate_handles_explicit_wildcard. Retrieved 3/6 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/a/b'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = '/a/b'
    var_5 = module_1._forced_separate(var_4, var_3)
    var_6 = bool(var_5 == ('/a/b', 'Matched forced_separate (/a/b) config value.'))
    assert var_6 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/a/'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = '/a/c'
    var_5 = module_1._forced_separate(var_4, var_3)
    var_6 = bool(var_5 == ('/a/', 'Matched forced_separate (/a/) config value.'))
    assert var_6 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'pattern'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = '.pattern'
    var_5 = module_1._forced_separate(var_4, var_3)
    var_6 = bool(var_5 == ('pattern', 'Matched forced_separate (pattern) config value.'))
    assert var_6 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/x/y'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = '/a/b'
    var_5 = module_1._forced_separate(var_4, var_3)
    assert var_5 is None

import isort.settings as module_0

def test_case_0():
    var_0 = '/a/*'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = '/a/b'

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/a/'
    var_1 = '/b/'
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = '/a/test'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('/a/', 'Matched forced_separate (/a/) config value.'))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = '/any/path'
    var_4 = module_1._forced_separate(var_3, var_2)
    assert var_4 is None



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_is_module_returns_true_when_py_file_exists. Retrieved 4/8 statements.


import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = 'test'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Path(*var_4, **var_5)
    var_7 = module_1._is_module(var_6)
    assert var_7 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_is_namespace_package_evaluates_true. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = '__init__.py'
    var_4 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_src_path_predicate_true. Retrieved 19/43 statements.


import pathlib as module_0

def test_case_0():
    var_0 = 'my_namespace'
    var_1 = '/tmp/src'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Path(*var_2, **var_3)
    var_5 = True
    var_6 = var_4.mkdir(parents=var_5, exist_ok=var_5)
    var_7 = 'my_namespace'
    var_8 = 'my_namespace.submodule'
    var_9 = 'builtins'
    var_10 = 'print'
    var_11 = False
    var_12 = 'firstparty'
    var_13 = '/tmp/test_src'
    var_14 = [var_13]
    var_15 = {}
    var_16 = module_0.Path(*var_14, **var_15)
    var_17 = var_16.resolve()
    var_18 = 'test_pkg'
    var_19 = var_17 / var_18
    var_20 = 'test_pkg.sub'
    var_21 = [var_17]
    var_22 = ()



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_is_module_returns_true_for_py_file. Retrieved 6/10 statements.
# Partially parsed test_is_module_returns_true_for_extension_suffix. Retrieved 6/11 statements.
# Partially parsed test_is_module_returns_true_for_init_py. Retrieved 6/10 statements.
# Partially parsed test_is_module_returns_false_when_no_files_exist. Retrieved 3/7 statements.


import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'my_package/module.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = str(var_3)
    var_5 = 'my_package/module'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Path(*var_6, **var_7)
    var_9 = module_1._is_module(var_8)
    assert var_9 is True

import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'my_package/module.cpython-39-x86_64-linux-gnu.so'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = str(var_3)
    var_5 = 'my_package/module'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Path(*var_6, **var_7)
    var_9 = module_1._is_module(var_8)
    assert var_9 is True

import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'my_package/__init__.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = str(var_3)
    var_5 = 'my_package'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Path(*var_6, **var_7)
    var_9 = module_1._is_module(var_8)
    assert var_9 is True

import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'non_existent_path'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = module_1._is_module(var_3)
    assert var_4 is False



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_src_path_predicate_false. Retrieved 5/11 statements.


import pathlib as module_0

def test_case_0():
    var_0 = '/tmp/dummy'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = [var_3]
    var_5 = 'a.b'
    var_6 = ()



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_known_pattern_predicate_evaluates_to_false. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'non_existent_pattern'
    var_1 = 'some_section'
    var_2 = 'other_section'
    var_3 = [var_2]
    var_4 = 'module.submodule'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_is_namespace_package_evaluates_true_when_init_exists_and_contains_namespace_declaration. Retrieved 12/14 statements.


import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test_namespace_pkg'
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



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_src_path_predicate_true. Retrieved 12/21 statements.


import pathlib as module_0

def test_case_0():
    var_0 = '.py'
    var_1 = '/tmp/src'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Path(*var_2, **var_3)
    var_5 = True
    var_6 = var_4.mkdir(parents=var_5, exist_ok=var_5)
    var_7 = 'my_module.submodule'
    var_8 = 'my_module'
    var_9 = var_4 / var_8
    var_10 = '__init__.py'
    var_11 = var_9 / var_10
    var_12 = [var_4]
    var_13 = ()



# Parsed testcases at query #20
#--------------------------




import pathlib as module_0
import isort.settings as module_1
import isort.place as module_2

def test_case_0():
    var_0 = '/tmp/src'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(parents=var_4, exist_ok=var_4)
    var_6 = [var_3]
    var_7 = 'different.namespace'
    var_8 = {var_7}
    var_9 = False
    var_10 = '.py'
    var_11 = [var_10]
    var_12 = 'src_paths'
    var_13 = 'namespace_packages'
    var_14 = 'auto_identify_namespace_packages'
    var_15 = 'supported_extensions'
    var_16 = {var_12: var_6, var_13: var_8, var_14: var_9, var_15: var_11}
    var_17 = module_1.Config(**var_16)
    var_18 = 'my_module.sub'
    var_19 = ()
    var_20 = module_2._src_path(var_18, var_17, prefix=var_19)
    var_21 = bool(var_20 is not None)
    assert var_21 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_is_namespace_package_evaluates_true_on_line_4. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = '__init__.py'
    var_4 = ''



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_src_path_namespace_package_true. Retrieved 12/27 statements.


import pathlib as module_0

def test_case_0():
    var_0 = '/tmp/src'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = [var_3]
    var_5 = 'root.submodule'
    var_6 = {var_5}
    var_7 = False
    var_8 = []
    var_9 = '/tmp/src/root'
    var_10 = True
    var_11 = 'root'
    var_12 = 'root.submodule'
    var_13 = '/tmp/src'
    var_14 = [var_13]
    var_15 = {}
    var_16 = module_0.Path(*var_14, **var_15)
    var_17 = [var_16]
    var_18 = ()



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_src_path_predicate_true. Retrieved 9/38 statements.


def test_case_0():
    var_0 = 'my_folder'
    var_1 = 'part1'
    var_2 = 'content'
    var_3 = []
    var_4 = False
    var_5 = 'py'
    var_6 = [var_5]
    var_7 = 'a.b'
    var_8 = ()



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_src_path_is_module_returns_true_when_conditions_met. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'my_module'
    var_1 = 'my_module'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_is_namespace_package_regular_package_with_init_no_namespace_marker. Retrieved 12/14 statements.
# Partially parsed test_is_namespace_package_regular_package_with_pkg_resources_marker. Retrieved 12/14 statements.
# Partially parsed test_is_namespace_package_regular_package_with_pkgutil_marker. Retrieved 12/14 statements.
# Partially parsed test_is_namespace_package_no_init_but_contains_py_files. Retrieved 13/15 statements.
# Partially parsed test_is_namespace_package_no_init_but_contains_config_files. Retrieved 13/15 statements.
# Partially parsed test_is_namespace_package_with_double_quotes_marker. Retrieved 12/14 statements.


import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'non_existent_path'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = module_1._is_namespace_package(var_3, var_6)
    assert var_7 is False

import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(parents=var_4, exist_ok=var_4)
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
    var_0 = 'test_pkg'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(parents=var_4, exist_ok=var_4)
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
    var_0 = 'test_pkg'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(parents=var_4, exist_ok=var_4)
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
    var_0 = 'test_pkg'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(parents=var_4, exist_ok=var_4)
    var_6 = 'module.py'
    var_7 = var_3 / var_6
    var_8 = ''
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
    var_0 = 'test_pkg'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(parents=var_4, exist_ok=var_4)
    var_6 = 'pyproject.toml'
    var_7 = var_3 / var_6
    var_8 = ''
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
    var_0 = 'test_pkg'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(parents=var_4, exist_ok=var_4)
    var_6 = 'py'
    var_7 = [var_6]
    var_8 = frozenset(var_7)
    var_9 = module_1._is_namespace_package(var_3, var_8)
    assert var_9 is True
    var_10 = var_3.rmdir()

import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(parents=var_4, exist_ok=var_4)
    var_6 = '__init__.py'
    var_7 = var_3 / var_6
    var_8 = b'__import__("pkg_resources").declare_namespace(__name__)'
    var_9 = 'py'
    var_10 = [var_9]
    var_11 = frozenset(var_10)
    var_12 = module_1._is_namespace_package(var_3, var_11)
    assert var_12 is True
    var_13 = var_3.rmdir()



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_src_path_returns_none_when_no_match_found. Retrieved 9/17 statements.
# Partially parsed test_src_path_returns_firstparty_when_module_found_in_src_path. Retrieved 6/14 statements.
# Partially parsed test_src_path_identifies_namespace_package_recursively. Retrieved 8/17 statements.


import pathlib as module_0

def test_case_0():
    var_0 = '/tmp/nonexistent_module'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = 'nonexistent'
    var_7 = '/tmp/fake'
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_0.Path(*var_8, **var_9)
    var_11 = [var_10]
    var_12 = ()

import pathlib as module_0

def test_case_0():
    var_0 = '/tmp/src'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'my_mod'
    var_5 = '/tmp/src'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Path(*var_6, **var_7)
    var_9 = [var_8]
    var_10 = 'Found in one of the configured src_paths'

import pathlib as module_0

def test_case_0():
    var_0 = '/tmp/src'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = 'pkg.submod'
    var_7 = '/tmp/src'
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_0.Path(*var_8, **var_9)
    var_11 = [var_10]



# Parsed testcases at query #2
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #3
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
    var_7 = '.py'
    var_8 = [var_7]
    var_9 = 'src_paths'
    var_10 = 'namespace_packages'
    var_11 = 'auto_identify_namespace_packages'
    var_12 = 'supported_extensions'
    var_13 = {var_9: var_4, var_10: var_5, var_11: var_6, var_12: var_8}
    var_14 = module_1.Config(**var_13)
    var_15 = 'pkg.mod'
    var_16 = ()
    var_17 = module_2._src_path(var_15, var_14, prefix=var_16)
    var_18 = bool(var_17 is not None or var_17 is None)
    assert var_18 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_is_module_returns_true_when_py_file_exists. Retrieved 4/10 statements.


import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/fake/directory/my_module'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = '.py'
    var_5 = module_1._is_module(var_3)
    assert var_5 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_src_path_predicate_true_via_namespace_packages. Retrieved 15/31 statements.


import pathlib as module_0

def test_case_0():
    var_0 = 'my_package'
    var_1 = '/tmp/test_dir'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Path(*var_2, **var_3)
    var_5 = '.py'
    var_6 = '/tmp/test_src'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_0.Path(*var_7, **var_8)
    var_10 = var_9.resolve()
    var_11 = var_10 / var_0
    var_12 = True
    var_13 = 'my_package.submodule'
    var_14 = [var_10]
    var_15 = ()
    var_16 = 'my_package.submodule'
    var_17 = [var_10]
    var_18 = ()



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_src_path_returns_none_when_no_match_found. Retrieved 6/12 statements.


import pathlib as module_0

def test_case_0():
    var_0 = '/tmp/src'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = [var_3]
    var_5 = []
    var_6 = False
    var_7 = frozenset(['py'])
    var_8 = 'firstparty'
    var_9 = 'sections'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_src_path_predicate_true. Retrieved 12/19 statements.


import pathlib as module_0

def test_case_0():
    var_0 = 'src/my_module'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'test_env'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Path(*var_5, **var_6)
    var_8 = var_7.resolve()
    var_9 = 'my_module'
    var_10 = var_8 / var_9
    var_11 = True
    var_12 = var_10 / var_9
    var_13 = 'dummy content'
    var_14 = [var_10]
    var_15 = ()



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_forced_separate_implicit_wildcard. Retrieved 5/7 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/src/'
    var_1 = '/tests/'
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = 'main.py'
    var_6 = module_1._forced_separate(var_5, var_4)
    assert var_6 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/src/*'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = '/src/utils.py'
    var_5 = module_1._forced_separate(var_4, var_3)
    var_6 = bool(var_5 == ('/src/*', 'Matched forced_separate (/src/*) config value.'))
    assert var_6 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/lib'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = '/lib/module.py'
    var_5 = '/lib/internal.py'
    var_6 = module_1._forced_separate(var_5, var_3)
    var_7 = bool(var_6 == ('/lib', 'Matched forced_separate (/lib) config value.'))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/data'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = './data/file.txt'
    var_5 = module_1._forced_separate(var_4, var_3)
    var_6 = bool(var_5 == ('/data', 'Matched forced_separate (/data) config value.'))
    assert var_6 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/ignore/'
    var_1 = '/target/'
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = '/target/file.py'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('/target/', 'Matched forced_separate (/target/) config value.'))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = '/any/path'
    var_4 = module_1._forced_separate(var_3, var_2)
    assert var_4 is None



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_src_path_returns_none_when_no_match_found. Retrieved 6/13 statements.
# Partially parsed test_src_path_returns_firstparty_when_module_exists_in_src_paths. Retrieved 9/17 statements.
# Partially parsed test_src_path_handles_nested_modules_with_namespace_packages. Retrieved 10/18 statements.
# Partially parsed test_src_path_handles_single_file_module_at_src_root. Retrieved 9/17 statements.


import pathlib as module_0

def test_case_0():
    var_0 = '/tmp/src'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = [var_3]
    var_5 = ()
    var_6 = False
    var_7 = frozenset(['py'])
    var_8 = 'firstparty'
    var_9 = 'nonexistent_module'

import pathlib as module_0

def test_case_0():
    var_0 = '/tmp/src'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = [var_3]
    var_5 = ()
    var_6 = False
    var_7 = frozenset(['py'])
    var_8 = 'firstparty'
    var_9 = 'my_module'
    var_10 = '/tmp/src'
    var_11 = [var_10]
    var_12 = {}
    var_13 = module_0.Path(*var_11, **var_12)
    var_14 = [var_13]

import pathlib as module_0

def test_case_0():
    var_0 = '/tmp/src'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = [var_3]
    var_5 = 'root.sub'
    var_6 = False
    var_7 = ()
    var_8 = 'firstparty'
    var_9 = 'root.sub'
    var_10 = '/tmp/src'
    var_11 = [var_10]
    var_12 = {}
    var_13 = module_0.Path(*var_11, **var_12)
    var_14 = [var_13]
    var_15 = ()

import pathlib as module_0

def test_case_0():
    var_0 = '/tmp/src'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = [var_3]
    var_5 = ()
    var_6 = False
    var_7 = frozenset(['py'])
    var_8 = 'firstparty'
    var_9 = 'my_mod'
    var_10 = '/tmp/src'
    var_11 = [var_10]
    var_12 = {}
    var_13 = module_0.Path(*var_11, **var_12)
    var_14 = [var_13]



# Parsed testcases at query #10
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



# Parsed testcases at query #11
#--------------------------




import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/fake/path/my_module'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = module_1._is_module(var_3)
    assert var_4 is True



# Parsed testcases at query #12
#--------------------------




import isort.settings as module_0
import pathlib as module_1
import isort.place as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/tmp'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_1.Path(*var_3, **var_4)
    var_6 = [var_5]
    var_7 = 'some_module'
    var_8 = 'some'
    var_9 = (var_8,)
    var_10 = module_2._src_path(var_7, var_1, var_6, var_9)
    assert var_10 is None



# Parsed testcases at query #13
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
    var_5 = 'my_package'
    var_6 = {var_5}
    var_7 = False
    var_8 = '.py'
    var_9 = [var_8]
    var_10 = 'src_paths'
    var_11 = 'namespace_packages'
    var_12 = 'auto_identify_namespace_packages'
    var_13 = 'supported_extensions'
    var_14 = {var_10: var_4, var_11: var_6, var_12: var_7, var_13: var_9}
    var_15 = module_1.Config(**var_14)
    var_16 = 'my_package.submodule'
    var_17 = ()
    var_18 = module_2._src_path(var_16, var_15, prefix=var_17)
    var_19 = bool(var_18 is not None)
    assert var_19 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_src_path_is_module_success. Retrieved 1/7 statements.
# Partially parsed test_src_path_is_module_fails_due_to_name_mismatch. Retrieved 1/7 statements.
# Partially parsed test_src_path_is_module_fails_due_to_not_a_directory. Retrieved 1/7 statements.
# Partially parsed test_src_path_is_module_fails_due_to_existence_check. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 'my_module'

def test_case_0():
    var_0 = 'my_module'

def test_case_0():
    var_0 = 'my_module'

def test_case_0():
    var_0 = 'my_module'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_is_namespace_package_not_a_directory. Retrieved 5/12 statements.
# Partially parsed test_is_namespace_package_no_init_and_has_py_file. Retrieved 5/12 statements.
# Partially parsed test_is_namespace_package_no_init_and_has_config_file. Retrieved 5/12 statements.
# Partially parsed test_is_namespace_package_no_init_and_empty_dir. Retrieved 3/8 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_init. Retrieved 5/12 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_init. Retrieved 5/12 statements.
# Partially parsed test_is_namespace_package_regular_init_fails. Retrieved 5/12 statements.


def test_case_0():
    var_0 = 'not_a_dir.txt'
    var_1 = 'content'
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = 'module.py'
    var_1 = 'content'
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
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)

def test_case_0():
    var_0 = '__init__.py'
    var_1 = b"__import__('pkg_resources').declare_namespace(__name__)"
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
    var_1 = b"print('hello')"
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_known_pattern_returns_none_when_no_match. Retrieved 4/14 statements.
# Partially parsed test_known_pattern_returns_match_for_exact_name. Retrieved 3/13 statements.
# Partially parsed test_known_pattern_matches_parent_module. Retrieved 4/14 statements.
# Partially parsed test_known_pattern_respects_sections_constraint. Retrieved 5/15 statements.
# Partially parsed test_known_pattern_prefers_most_specific_match. Retrieved 5/17 statements.


def test_case_0():
    var_0 = 'abc'
    var_1 = 'section1'
    var_2 = [var_1]
    var_3 = 'xyz.def'

def test_case_0():
    var_0 = 'a.b'
    var_1 = 'section1'
    var_2 = [var_1]

def test_case_0():
    var_0 = 'a'
    var_1 = 'section1'
    var_2 = [var_1]
    var_3 = 'a.b.c'

def test_case_0():
    var_0 = 'a'
    var_1 = 'section1'
    var_2 = 'section2'
    var_3 = [var_2]
    var_4 = 'a.b'

def test_case_0():
    var_0 = 'a'
    var_1 = 'general'
    var_2 = 'a.b'
    var_3 = 'specific'
    var_4 = [var_1, var_3]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_src_path_returns_none_when_no_match_found. Retrieved 8/11 statements.
# Partially parsed test_src_path_returns_firstparty_when_module_found_in_src_path. Retrieved 9/26 statements.
# Partially parsed test_src_path_handles_nested_modules_in_namespace_packages. Retrieved 8/26 statements.


import pathlib as module_0

def test_case_0():
    var_0 = '/tmp/nonexistent_module'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = [var_3]
    var_5 = []
    var_6 = False
    var_7 = frozenset(['py'])
    var_8 = 'my_module'
    var_9 = '/tmp/fake'
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_0.Path(*var_10, **var_11)
    var_13 = [var_12]

import pathlib as module_0

def test_case_0():
    var_0 = '/tmp/src'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = [var_3]
    var_5 = []
    var_6 = False
    var_7 = 'src'
    var_8 = 'my_module'
    var_9 = '__init__.py'
    var_10 = []
    var_11 = False
    var_12 = frozenset(['py'])

def test_case_0():
    var_0 = 'src'
    var_1 = 'parent'
    var_2 = '__init__.py'
    var_3 = 'child'
    var_4 = 'parent'
    var_5 = [var_4]
    var_6 = False
    var_7 = frozenset(['py'])
    var_8 = 'parent.child'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_known_pattern_predicate_evaluates_to_false. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'not_matching'
    var_1 = 'some_section'
    var_2 = 'other_section'
    var_3 = [var_2]
    var_4 = 'module.submodule'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_src_path_predicate_true. Retrieved 11/20 statements.


import pathlib as module_0
import isort.settings as module_1
import isort.place as module_2

def test_case_0():
    var_0 = '/tmp/test_src_path_dir'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = var_3.resolve()
    var_5 = True
    var_6 = 'my_module'
    var_7 = var_4 / var_6
    var_8 = var_4.name
    var_9 = [var_4]
    var_10 = 'src_paths'
    var_11 = {var_10: var_9}
    var_12 = module_1.Config(**var_11)
    var_13 = ()
    var_14 = module_2._src_path(var_8, var_12, prefix=var_13)
    var_15 = bool(var_14 is not None)
    assert var_15 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_src_path_is_module_returns_true_when_conditions_met. Retrieved 2/13 statements.


def test_case_0():
    var_0 = 'my_module'
    var_1 = 'my_module'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_is_module_returns_true_for_py_file. Retrieved 6/8 statements.
# Partially parsed test_is_module_returns_true_for_extension_suffix. Retrieved 6/8 statements.
# Partially parsed test_is_module_returns_true_for_init_py. Retrieved 6/8 statements.
# Partially parsed test_is_module_returns_false_when_no_files_exist. Retrieved 3/5 statements.
# Partially parsed test_is_module_checks_py_suffix_first. Retrieved 7/9 statements.
# Partially parsed test_is_module_checks_init_py_last. Retrieved 3/7 statements.


import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'my_module.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = str(var_3)
    var_5 = 'my_module'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Path(*var_6, **var_7)
    var_9 = module_1._is_module(var_8)
    assert var_9 is True

import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'my_module.cpython-39-x86_64-linux-gnu.so'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = str(var_3)
    var_5 = 'my_module'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Path(*var_6, **var_7)
    var_9 = module_1._is_module(var_8)
    assert var_9 is True

import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'my_module/__init__.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = str(var_3)
    var_5 = 'my_module'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.Path(*var_6, **var_7)
    var_9 = module_1._is_module(var_8)
    assert var_9 is True

import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'non_existent'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = module_1._is_module(var_3)
    assert var_4 is False

import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = True
    var_1 = 'my_module'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Path(*var_2, **var_3)
    var_5 = module_1._is_module(var_4)
    assert var_5 is True
    var_6 = 'my_module.py'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_0.Path(*var_7, **var_8)
    var_10 = str(var_9)

import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'my_package'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = module_1._is_module(var_3)
    assert var_4 is True



# Parsed testcases at query #22
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/path/to/dir/'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = 'other/file.txt'
    var_5 = module_1._forced_separate(var_4, var_3)
    assert var_5 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/path/to/dir/'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = '/path/to/dir/file.txt'
    var_5 = module_1._forced_separate(var_4, var_3)
    var_6 = bool(var_5 == ('/path/to/dir/', 'Matched forced_separate (/path/to/dir/) config value.'))
    assert var_6 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/path/to/*'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = '/path/to/sub/file.txt'
    var_5 = module_1._forced_separate(var_4, var_3)
    var_6 = bool(var_5 == ('/path/to/*', 'Matched forced_separate (/path/to/*) config value.'))
    assert var_6 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'data'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = '.data/file.txt'
    var_5 = module_1._forced_separate(var_4, var_3)
    var_6 = bool(var_5 == ('data', 'Matched forced_separate (data) config value.'))
    assert var_6 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = '/path/to/dir/'
    var_4 = module_1._forced_separate(var_3, var_2)
    assert var_4 is None



# Parsed testcases at query #23
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'pattern'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'pattern_match'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = var_6[0]
    assert var_7 == 'pattern'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_known_pattern_predicate_false. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'non_matching_pattern'
    var_1 = 'target_section'
    var_2 = 'other_section'
    var_3 = [var_2]
    var_4 = 'some.module'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_is_namespace_package_evaluates_true_at_line_two. Retrieved 9/11 statements.


import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = 'py'
    var_7 = [var_6]
    var_8 = frozenset(var_7)
    var_9 = module_1._is_package(var_3)
    assert var_9 is True
    var_10 = module_1._is_namespace_package(var_3, var_8)
    var_11 = bool(var_10 is not False)
    assert var_11 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_src_path_is_module_evaluates_true. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'my_module'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_is_namespace_package_with_init_and_pkg_resources. Retrieved 12/14 statements.
# Partially parsed test_is_namespace_package_with_init_and_pkgutil. Retrieved 12/14 statements.
# Partially parsed test_is_namespace_package_with_init_but_no_namespace_marker. Retrieved 12/14 statements.
# Partially parsed test_is_namespace_package_no_init_but_contains_src_extensions. Retrieved 12/14 statements.
# Partially parsed test_is_namespace_package_no_init_but_contains_config_files. Retrieved 12/14 statements.


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
    var_0 = 'test_pkgutil'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = '__init__.py'
    var_7 = var_3 / var_6
    var_8 = b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
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
    var_13 = var_3.rmdir()

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
import isort.place as module_1

def test_case_0():
    var_0 = 'test_namespace_with_config'
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
    var_12 = module_1._is_namespace_package(var_3, var_11)
    assert var_12 is False
    var_13 = var_3.rmdir()



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_known_pattern_predicate_true. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'root'
    var_1 = 'sub'
    var_2 = [var_0, var_1]
    var_3 = 'auth\\..*'
    var_4 = 'auth.user.login'



# Parsed testcases at query #29
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



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_is_namespace_package_evaluates_true_at_line_six. Retrieved 10/14 statements.


import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test_namespace_dir'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = '__init__.py'
    var_7 = var_3 / var_6
    var_8 = 'py'
    var_9 = [var_8]
    var_10 = frozenset(var_9)
    var_11 = module_1._is_namespace_package(var_3, var_10)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_known_pattern_returns_none_when_no_match. Retrieved 4/9 statements.
# Partially parsed test_known_pattern_returns_none_when_placement_not_in_sections. Retrieved 4/9 statements.
# Partially parsed test_known_pattern_returns_match_for_exact_name. Retrieved 3/8 statements.
# Partially parsed test_known_pattern_matches_parent_module_hierarchically. Retrieved 4/9 statements.
# Partially parsed test_known_pattern_prefers_longest_matching_module_name. Retrieved 6/13 statements.
# Partially parsed test_known_pattern_handles_empty_name. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'abc'
    var_1 = 'section1'
    var_2 = [var_1]
    var_3 = 'xyz.def'

def test_case_0():
    var_0 = 'abc'
    var_1 = 'section2'
    var_2 = 'section1'
    var_3 = [var_2]

def test_case_0():
    var_0 = 'abc'
    var_1 = 'section1'
    var_2 = [var_1]

def test_case_0():
    var_0 = 'a\\.b'
    var_1 = 'section1'
    var_2 = [var_1]
    var_3 = 'a.b.c'

def test_case_0():
    var_0 = 'a'
    var_1 = 'section_short'
    var_2 = 'a\\.b'
    var_3 = 'section_long'
    var_4 = [var_1, var_3]
    var_5 = 'a.b.c'

def test_case_0():
    var_0 = '.*'
    var_1 = 'section1'
    var_2 = [var_1]
    var_3 = ''



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_is_namespace_package_regular_package_with_init_no_namespace_marker. Retrieved 12/14 statements.
# Partially parsed test_is_namespace_package_regular_package_with_pkg_resources_marker. Retrieved 12/14 statements.
# Partially parsed test_is_namespace_package_regular_package_with_pkgutil_marker. Retrieved 12/14 statements.
# Partially parsed test_is_namespace_package_namespace_without_init_with_other_files. Retrieved 13/15 statements.
# Partially parsed test_is_namespace_package_namespace_without_init_with_config_file. Retrieved 13/15 statements.


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
    var_0 = 'test_pkg_reg'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = '__init__.py'
    var_7 = var_3 / var_6
    var_8 = "print('hello')"
    var_9 = 'py'
    var_10 = [var_9]
    var_11 = frozenset(var_10)
    var_12 = module_1._is_namespace_package(var_3, var_11)
    assert var_12 is False
    var_13 = var_3.rmdir()

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
    var_8 = "__import__('pkg_resources').declare_namespace(__name__)"
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
    var_8 = "__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_9 = 'py'
    var_10 = [var_9]
    var_11 = frozenset(var_10)
    var_12 = module_1._is_namespace_package(var_3, var_11)
    assert var_12 is True
    var_13 = var_3.rmdir()

import pathlib as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test_pkg_no_init'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = 'module.py'
    var_7 = var_3 / var_6
    var_8 = 'pass'
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
    var_0 = 'test_pkg_empty'
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
import isort.place as module_1

def test_case_0():
    var_0 = 'test_pkg_config'
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
    var_12 = module_1._is_namespace_package(var_3, var_11)
    assert var_12 is False
    var_13 = var_3 / var_6
    var_14 = var_3.rmdir()



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_is_namespace_package_true_when_no_init_and_no_src_files. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_src_path_returns_none_when_no_match_found. Retrieved 8/16 statements.
# Partially parsed test_src_path_returns_firstparty_when_module_found_in_src_path. Retrieved 4/24 statements.
# Partially parsed test_src_path_handles_nested_modules_as_namespace. Retrieved 12/23 statements.
# Partially parsed test_src_path_identifies_module_at_root_of_src_path. Retrieved 4/18 statements.


import pathlib as module_0

def test_case_0():
    var_0 = '/fake/src'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = 'nonexistent_module'
    var_7 = '/non/existent'
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_0.Path(*var_8, **var_9)
    var_11 = [var_10]

def test_case_0():
    var_0 = 'src'
    var_1 = 'my_module'
    var_2 = '__init__.py'
    var_3 = 'my_module'
    var_4 = 'Found in one of the configured src_paths'

import pathlib as module_0

def test_case_0():
    var_0 = '/fake/src'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'root'
    var_5 = 'py'
    var_6 = [var_5]
    var_7 = 'FIRSTPARTY'
    var_8 = 'Found in one of the configured src_paths: /fake/src.'
    var_9 = 'root.submodule'
    var_10 = '/fake/src'
    var_11 = [var_10]
    var_12 = {}
    var_13 = module_0.Path(*var_11, **var_12)
    var_14 = [var_13]
    var_15 = ()

def test_case_0():
    var_0 = 'standalone_mod.py'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = 'standalone_mod'
    var_4 = 'Found in one of the configured src_paths'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_src_path_is_module_evaluates_to_true. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'test_module'



