####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_src_path_returns_none_when_module_not_found. Retrieved 1/9 statements.
# Partially parsed test_src_path_returns_firstparty_when_module_found. Retrieved 2/14 statements.
# Partially parsed test_src_path_with_nested_module. Retrieved 4/20 statements.
# Partially parsed test_src_path_with_py_file. Retrieved 2/12 statements.
# Partially parsed test_src_path_with_custom_src_paths. Retrieved 3/18 statements.
# Partially parsed test_src_path_with_prefix. Retrieved 4/21 statements.


def test_case_0():
    var_0 = 'nonexistent_module'

def test_case_0():
    var_0 = 'test_module'
    var_1 = '__init__.py'

def test_case_0():
    var_0 = 'parent'
    var_1 = '__init__.py'
    var_2 = 'child'
    var_3 = 'parent.child'

def test_case_0():
    var_0 = 'module.py'
    var_1 = 'module'

def test_case_0():
    var_0 = 'custom_src'
    var_1 = 'my_module'
    var_2 = '__init__.py'

def test_case_0():
    var_0 = 'parent'
    var_1 = '__init__.py'
    var_2 = 'child'
    var_3 = (var_0,)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 5/29 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_0]
    var_4 = 'mymodule'
    var_5 = 'mypackage'
    var_6 = 'mymodule'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_known_pattern_matches_exact_module. Retrieved 20/24 statements.
# Partially parsed test_known_pattern_matches_submodule. Retrieved 19/22 statements.
# Partially parsed test_known_pattern_no_match. Retrieved 19/21 statements.
# Partially parsed test_known_pattern_section_not_in_config. Retrieved 21/23 statements.
# Partially parsed test_known_pattern_matches_longest_prefix. Retrieved 27/29 statements.
# Partially parsed test_known_pattern_empty_known_patterns. Retrieved 10/12 statements.
# Partially parsed test_known_pattern_single_part_name. Retrieved 18/20 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'Config'
    var_1 = 'known_patterns'
    var_2 = 'sections'
    var_3 = [var_1, var_2]
    var_4 = 'Pattern'
    var_5 = 'pattern'
    var_6 = [var_5]
    var_7 = ()
    var_8 = 'match'
    var_9 = 'django'
    var_10 = lambda self, x: x == var_9
    var_11 = {var_8: var_10}
    var_12 = type(var_4, var_7, var_11)
    var_13 = var_12()
    var_14 = 'third_party'
    var_15 = (var_13, var_14)
    var_16 = [var_15]
    var_17 = [var_14]
    var_18 = 'known_patterns'
    var_19 = 'sections'
    var_20 = {var_18: var_16, var_19: var_17}
    var_21 = module_0.Config(**var_20)
    var_22 = module_1._known_pattern(var_9, var_21)
    var_23 = bool(var_22 is not None)
    assert var_23 is True
    var_24 = var_22[0]
    assert var_24 == 'third_party'
    var_25 = 'Matched configured known pattern'
    var_26 = bool('Matched configured known pattern' in var_22[1])
    assert var_26 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'Config'
    var_1 = 'known_patterns'
    var_2 = 'sections'
    var_3 = [var_1, var_2]
    var_4 = 'Pattern'
    var_5 = ()
    var_6 = 'match'
    var_7 = 'django.conf'
    var_8 = lambda self, x: x == var_7
    var_9 = {var_6: var_8}
    var_10 = type(var_4, var_5, var_9)
    var_11 = var_10()
    var_12 = 'third_party'
    var_13 = (var_11, var_12)
    var_14 = [var_13]
    var_15 = [var_12]
    var_16 = 'known_patterns'
    var_17 = 'sections'
    var_18 = {var_16: var_14, var_17: var_15}
    var_19 = module_0.Config(**var_18)
    var_20 = 'django.conf.settings'
    var_21 = module_1._known_pattern(var_20, var_19)
    var_22 = bool(var_21 is not None)
    assert var_22 is True
    var_23 = var_21[0]
    assert var_23 == 'third_party'

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'Config'
    var_1 = 'known_patterns'
    var_2 = 'sections'
    var_3 = [var_1, var_2]
    var_4 = 'Pattern'
    var_5 = ()
    var_6 = 'match'
    var_7 = False
    var_8 = lambda self, x: var_7
    var_9 = {var_6: var_8}
    var_10 = type(var_4, var_5, var_9)
    var_11 = var_10()
    var_12 = 'third_party'
    var_13 = (var_11, var_12)
    var_14 = [var_13]
    var_15 = [var_12]
    var_16 = 'known_patterns'
    var_17 = 'sections'
    var_18 = {var_16: var_14, var_17: var_15}
    var_19 = module_0.Config(**var_18)
    var_20 = 'mymodule'
    var_21 = module_1._known_pattern(var_20, var_19)
    assert var_21 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'Config'
    var_1 = 'known_patterns'
    var_2 = 'sections'
    var_3 = [var_1, var_2]
    var_4 = 'Pattern'
    var_5 = ()
    var_6 = 'match'
    var_7 = True
    var_8 = lambda self, x: var_7
    var_9 = {var_6: var_8}
    var_10 = type(var_4, var_5, var_9)
    var_11 = var_10()
    var_12 = 'unknown_section'
    var_13 = (var_11, var_12)
    var_14 = [var_13]
    var_15 = 'third_party'
    var_16 = 'stdlib'
    var_17 = [var_15, var_16]
    var_18 = 'known_patterns'
    var_19 = 'sections'
    var_20 = {var_18: var_14, var_19: var_17}
    var_21 = module_0.Config(**var_20)
    var_22 = 'mymodule'
    var_23 = module_1._known_pattern(var_22, var_21)
    assert var_23 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'Config'
    var_1 = 'known_patterns'
    var_2 = 'sections'
    var_3 = [var_1, var_2]
    var_4 = 'Pattern'
    var_5 = ()
    var_6 = 'match'
    var_7 = 'django'
    var_8 = lambda self, x: x.startswith(var_7)
    var_9 = {var_6: var_8}
    var_10 = type(var_4, var_5, var_9)
    var_11 = var_10()
    var_12 = ()
    var_13 = 'django.conf'
    var_14 = lambda self, x: x == var_13
    var_15 = {var_6: var_14}
    var_16 = type(var_4, var_12, var_15)
    var_17 = var_16()
    var_18 = 'third_party'
    var_19 = (var_11, var_18)
    var_20 = 'local'
    var_21 = (var_17, var_20)
    var_22 = [var_19, var_21]
    var_23 = [var_18, var_20]
    var_24 = 'known_patterns'
    var_25 = 'sections'
    var_26 = {var_24: var_22, var_25: var_23}
    var_27 = module_0.Config(**var_26)
    var_28 = 'django.conf.settings'
    var_29 = module_1._known_pattern(var_28, var_27)
    var_30 = bool(var_29 is not None)
    assert var_30 is True
    var_31 = var_29[0]
    assert var_31 == 'local'

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'Config'
    var_1 = 'known_patterns'
    var_2 = 'sections'
    var_3 = [var_1, var_2]
    var_4 = []
    var_5 = 'third_party'
    var_6 = [var_5]
    var_7 = 'known_patterns'
    var_8 = 'sections'
    var_9 = {var_7: var_4, var_8: var_6}
    var_10 = module_0.Config(**var_9)
    var_11 = 'django'
    var_12 = module_1._known_pattern(var_11, var_10)
    assert var_12 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'Config'
    var_1 = 'known_patterns'
    var_2 = 'sections'
    var_3 = [var_1, var_2]
    var_4 = 'Pattern'
    var_5 = ()
    var_6 = 'match'
    var_7 = 'os'
    var_8 = lambda self, x: x == var_7
    var_9 = {var_6: var_8}
    var_10 = type(var_4, var_5, var_9)
    var_11 = var_10()
    var_12 = 'stdlib'
    var_13 = (var_11, var_12)
    var_14 = [var_13]
    var_15 = [var_12]
    var_16 = 'known_patterns'
    var_17 = 'sections'
    var_18 = {var_16: var_14, var_17: var_15}
    var_19 = module_0.Config(**var_18)
    var_20 = module_1._known_pattern(var_7, var_19)
    var_21 = bool(var_20 is not None)
    assert var_21 is True
    var_22 = var_20[0]
    assert var_22 == 'stdlib'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 3/18 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 2/17 statements.
# Partially parsed test_is_module_not_a_module. Retrieved 1/13 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 3/20 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = '.py'
    var_2 = 0

def test_case_0():
    var_0 = 'test_package'
    var_1 = '__init__.py'

def test_case_0():
    var_0 = 'not_a_module'

def test_case_0():
    var_0 = 'test_module'
    var_1 = 0
    var_2 = '.so'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_src_path_returns_none_when_module_not_found. Retrieved 2/11 statements.
# Partially parsed test_src_path_returns_firstparty_when_module_is_found. Retrieved 2/12 statements.
# Partially parsed test_src_path_with_nested_module_not_namespace_package. Retrieved 2/14 statements.
# Partially parsed test_src_path_with_empty_name. Retrieved 2/11 statements.
# Partially parsed test_src_path_with_custom_src_paths. Retrieved 2/9 statements.
# Partially parsed test_src_path_with_prefix. Retrieved 4/15 statements.


def test_case_0():
    var_0 = '/nonexistent/path'
    var_1 = [var_0]
    var_2 = 'nonexistent_module'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'test_module'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'parent.child'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = ''

def test_case_0():
    var_0 = '/custom/src'
    var_1 = [var_0]
    var_2 = 'module'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'child'
    var_3 = 'parent'
    var_4 = (var_3,)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 4/19 statements.


def test_case_0():
    var_0 = 'test_module.py'
    var_1 = '# test module'
    var_2 = 'pathlib_helper'
    var_3 = ''



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_src_path_returns_none_when_module_not_found. Retrieved 4/13 statements.
# Partially parsed test_src_path_returns_firstparty_when_module_found. Retrieved 5/20 statements.
# Partially parsed test_src_path_with_nested_module. Retrieved 7/25 statements.
# Partially parsed test_src_path_with_py_file_module. Retrieved 5/17 statements.
# Partially parsed test_src_path_uses_default_src_paths. Retrieved 6/18 statements.
# Partially parsed test_src_path_with_prefix. Retrieved 7/22 statements.


def test_case_0():
    var_0 = 'src'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = 'nonexistent_module'

def test_case_0():
    var_0 = 'src'
    var_1 = 'mymodule'
    var_2 = '__init__.py'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = 'Found in one of the configured src_paths'

def test_case_0():
    var_0 = 'src'
    var_1 = 'mypackage'
    var_2 = '__init__.py'
    var_3 = 'submodule'
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = 'mypackage.submodule'

def test_case_0():
    var_0 = 'src'
    var_1 = 'mymodule.py'
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = 'mymodule'

def test_case_0():
    var_0 = 'src'
    var_1 = 'mymodule.py'
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = 'mymodule'
    var_5 = None

def test_case_0():
    var_0 = 'src'
    var_1 = 'parent'
    var_2 = '__init__.py'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = 'child'
    var_6 = (var_1,)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_src_path_returns_none_when_module_not_found. Retrieved 12/17 statements.
# Partially parsed test_src_path_returns_firstparty_when_module_is_file. Retrieved 14/21 statements.
# Partially parsed test_src_path_returns_firstparty_when_module_is_package. Retrieved 14/23 statements.
# Partially parsed test_src_path_with_nested_module. Retrieved 16/27 statements.
# Partially parsed test_src_path_with_custom_src_paths. Retrieved 15/24 statements.
# Partially parsed test_src_path_with_src_path_is_module. Retrieved 12/19 statements.


def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'src_paths'
    var_3 = 'namespace_packages'
    var_4 = 'auto_identify_namespace_packages'
    var_5 = 'supported_extensions'
    var_6 = frozenset()
    var_7 = False
    var_8 = 'py'
    var_9 = [var_8]
    var_10 = frozenset(var_9)
    var_11 = 'nonexistent_module'

def test_case_0():
    var_0 = 'test_module.py'
    var_1 = '# test'
    var_2 = 'Config'
    var_3 = ()
    var_4 = 'src_paths'
    var_5 = 'namespace_packages'
    var_6 = 'auto_identify_namespace_packages'
    var_7 = 'supported_extensions'
    var_8 = frozenset()
    var_9 = False
    var_10 = 'py'
    var_11 = [var_10]
    var_12 = frozenset(var_11)
    var_13 = 'test_module'

def test_case_0():
    var_0 = 'test_package'
    var_1 = '__init__.py'
    var_2 = '# test'
    var_3 = 'Config'
    var_4 = ()
    var_5 = 'src_paths'
    var_6 = 'namespace_packages'
    var_7 = 'auto_identify_namespace_packages'
    var_8 = 'supported_extensions'
    var_9 = frozenset()
    var_10 = False
    var_11 = 'py'
    var_12 = [var_11]
    var_13 = frozenset(var_12)

def test_case_0():
    var_0 = 'parent_package'
    var_1 = '__init__.py'
    var_2 = '# test'
    var_3 = 'child_module.py'
    var_4 = 'Config'
    var_5 = ()
    var_6 = 'src_paths'
    var_7 = 'namespace_packages'
    var_8 = 'auto_identify_namespace_packages'
    var_9 = 'supported_extensions'
    var_10 = frozenset()
    var_11 = False
    var_12 = 'py'
    var_13 = [var_12]
    var_14 = frozenset(var_13)
    var_15 = 'parent_package.child_module'

def test_case_0():
    var_0 = 'custom_src'
    var_1 = 'my_module.py'
    var_2 = '# test'
    var_3 = 'Config'
    var_4 = ()
    var_5 = 'src_paths'
    var_6 = 'namespace_packages'
    var_7 = 'auto_identify_namespace_packages'
    var_8 = 'supported_extensions'
    var_9 = frozenset()
    var_10 = False
    var_11 = 'py'
    var_12 = [var_11]
    var_13 = frozenset(var_12)
    var_14 = 'my_module'

def test_case_0():
    var_0 = 'module_name'
    var_1 = 'Config'
    var_2 = ()
    var_3 = 'src_paths'
    var_4 = 'namespace_packages'
    var_5 = 'auto_identify_namespace_packages'
    var_6 = 'supported_extensions'
    var_7 = frozenset()
    var_8 = False
    var_9 = 'py'
    var_10 = [var_9]
    var_11 = frozenset(var_10)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_namespace_package_predicate_evaluates_to_false. Retrieved 3/20 statements.


def test_case_0():
    var_0 = 'mymodule'
    var_1 = '__init__.py'
    var_2 = 'mymodule'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_namespace_in_config_namespace_packages. Retrieved 9/30 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'myapp.submodule'
    var_3 = '.py'
    var_4 = [var_0]
    var_5 = True
    var_6 = '/src/myapp'
    var_7 = [var_6]
    var_8 = 'myapp.submodule'
    var_9 = [var_0]
    var_10 = ()
    var_11 = '.'
    var_12 = *var_10



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_forced_separate_matches_pattern_with_asterisk. Retrieved 2/6 statements.
# Partially parsed test_forced_separate_matches_pattern_without_asterisk. Retrieved 2/6 statements.
# Partially parsed test_forced_separate_matches_with_dot_prefix. Retrieved 2/6 statements.
# Partially parsed test_forced_separate_no_match. Retrieved 2/6 statements.
# Partially parsed test_forced_separate_empty_list. Retrieved 1/5 statements.
# Partially parsed test_forced_separate_exact_match. Retrieved 1/5 statements.
# Partially parsed test_forced_separate_pattern_with_explicit_asterisk. Retrieved 2/6 statements.
# Partially parsed test_forced_separate_multiple_patterns_first_match. Retrieved 3/7 statements.
# Partially parsed test_forced_separate_multiple_patterns_second_match. Retrieved 3/7 statements.
# Partially parsed test_forced_separate_wildcard_pattern. Retrieved 2/6 statements.
# Partially parsed test_forced_separate_case_sensitive. Retrieved 2/6 statements.
# Partially parsed test_forced_separate_dot_prefix_with_pattern. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'django.db'
    var_1 = 'django.db.models'

def test_case_0():
    var_0 = 'flask'
    var_1 = 'flask.app'

def test_case_0():
    var_0 = 'utils'
    var_1 = '.utils.helpers'

def test_case_0():
    var_0 = 'django.db'
    var_1 = 'requests.api'

def test_case_0():
    var_0 = 'any.module'

def test_case_0():
    var_0 = 'os'

def test_case_0():
    var_0 = 'pytest.*'
    var_1 = 'pytest.fixture'

def test_case_0():
    var_0 = 'django.db'
    var_1 = 'flask'
    var_2 = 'django.db.models'

def test_case_0():
    var_0 = 'django.db'
    var_1 = 'flask'
    var_2 = 'flask.app'

def test_case_0():
    var_0 = 'lib*'
    var_1 = 'library.core'

def test_case_0():
    var_0 = 'Django'
    var_1 = 'django.db'

def test_case_0():
    var_0 = 'requests'
    var_1 = '.requests.api'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_src_paths_is_not_none. Retrieved 3/14 statements.


def test_case_0():
    var_0 = '/some/path'
    var_1 = [var_0]
    var_2 = '/custom/path'
    var_3 = [var_2]
    var_4 = 'test_module'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_src_path_is_module. Retrieved 5/32 statements.


def test_case_0():
    var_0 = 'mymodule'
    var_1 = 'othermodule'
    var_2 = 'mymodule'
    var_3 = 'mymodule'
    var_4 = 'othermodule'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_is_namespace_package_not_a_package. Retrieved 4/7 statements.
# Partially parsed test_is_namespace_package_regular_package_with_init. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_double_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path_double_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_py_files. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_no_py_files. Retrieved 4/8 statements.
# Partially parsed test_is_namespace_package_no_init_with_setup_cfg. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_pyproject_toml. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'nonexistent'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = '__init__.py'
    var_5 = '# regular package'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = '__init__.py'
    var_5 = b"__import__('pkg_resources').declare_namespace(__name__)"

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = '__init__.py'
    var_5 = b'__import__("pkg_resources").declare_namespace(__name__)'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = '__init__.py'
    var_5 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = '__init__.py'
    var_5 = b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = 'module.py'
    var_5 = '# module'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'pkg'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = 'setup.cfg'
    var_5 = ''

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = 'pyproject.toml'
    var_5 = ''



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_src_path_is_module. Retrieved 1/18 statements.


def test_case_0():
    var_0 = 'test_module'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_src_path_returns_none_when_module_not_found. Retrieved 12/18 statements.
# Partially parsed test_src_path_finds_module_in_src_paths. Retrieved 13/23 statements.
# Partially parsed test_src_path_finds_py_file_module. Retrieved 13/21 statements.
# Partially parsed test_src_path_with_nested_module. Retrieved 15/29 statements.
# Partially parsed test_src_path_with_custom_src_paths_param. Retrieved 14/27 statements.
# Partially parsed test_src_path_respects_namespace_packages. Retrieved 16/30 statements.
# Partially parsed test_src_path_with_prefix. Retrieved 15/30 statements.


def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'src_paths'
    var_3 = 'namespace_packages'
    var_4 = 'auto_identify_namespace_packages'
    var_5 = 'supported_extensions'
    var_6 = frozenset()
    var_7 = False
    var_8 = 'py'
    var_9 = [var_8]
    var_10 = frozenset(var_9)
    var_11 = 'nonexistent_module'

def test_case_0():
    var_0 = 'mymodule'
    var_1 = '__init__.py'
    var_2 = 'Config'
    var_3 = ()
    var_4 = 'src_paths'
    var_5 = 'namespace_packages'
    var_6 = 'auto_identify_namespace_packages'
    var_7 = 'supported_extensions'
    var_8 = frozenset()
    var_9 = False
    var_10 = 'py'
    var_11 = [var_10]
    var_12 = frozenset(var_11)
    var_13 = 'src_paths'

def test_case_0():
    var_0 = 'mymodule.py'
    var_1 = 'Config'
    var_2 = ()
    var_3 = 'src_paths'
    var_4 = 'namespace_packages'
    var_5 = 'auto_identify_namespace_packages'
    var_6 = 'supported_extensions'
    var_7 = frozenset()
    var_8 = False
    var_9 = 'py'
    var_10 = [var_9]
    var_11 = frozenset(var_10)
    var_12 = 'mymodule'

def test_case_0():
    var_0 = 'mypkg'
    var_1 = '__init__.py'
    var_2 = 'submodule'
    var_3 = 'Config'
    var_4 = ()
    var_5 = 'src_paths'
    var_6 = 'namespace_packages'
    var_7 = 'auto_identify_namespace_packages'
    var_8 = 'supported_extensions'
    var_9 = frozenset()
    var_10 = False
    var_11 = 'py'
    var_12 = [var_11]
    var_13 = frozenset(var_12)
    var_14 = 'mypkg.submodule'

def test_case_0():
    var_0 = 'custom'
    var_1 = 'mymodule'
    var_2 = '__init__.py'
    var_3 = 'Config'
    var_4 = ()
    var_5 = 'src_paths'
    var_6 = 'namespace_packages'
    var_7 = 'auto_identify_namespace_packages'
    var_8 = 'supported_extensions'
    var_9 = frozenset()
    var_10 = False
    var_11 = 'py'
    var_12 = [var_11]
    var_13 = frozenset(var_12)

def test_case_0():
    var_0 = 'mypkg'
    var_1 = '__init__.py'
    var_2 = 'submodule'
    var_3 = 'Config'
    var_4 = ()
    var_5 = 'src_paths'
    var_6 = 'namespace_packages'
    var_7 = 'auto_identify_namespace_packages'
    var_8 = 'supported_extensions'
    var_9 = [var_0]
    var_10 = frozenset(var_9)
    var_11 = False
    var_12 = 'py'
    var_13 = [var_12]
    var_14 = frozenset(var_13)
    var_15 = 'mypkg.submodule'

def test_case_0():
    var_0 = 'mypkg'
    var_1 = '__init__.py'
    var_2 = 'submodule'
    var_3 = 'Config'
    var_4 = ()
    var_5 = 'src_paths'
    var_6 = 'namespace_packages'
    var_7 = 'auto_identify_namespace_packages'
    var_8 = 'supported_extensions'
    var_9 = frozenset()
    var_10 = False
    var_11 = 'py'
    var_12 = [var_11]
    var_13 = frozenset(var_12)
    var_14 = (var_0,)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_namespace_package_predicate_evaluates_to_false. Retrieved 4/18 statements.


def test_case_0():
    var_0 = 'mymodule'
    var_1 = '__init__.py'
    var_2 = ''
    var_3 = 'mymodule.submodule'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_is_namespace_package_not_a_package. Retrieved 4/7 statements.
# Partially parsed test_is_namespace_package_regular_package_with_init. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_single_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_double_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_single_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_double_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_python_files. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_setup_cfg. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_pyproject_toml. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_empty_directory. Retrieved 4/8 statements.
# Partially parsed test_is_namespace_package_no_init_with_other_extensions. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'nonexistent'

def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = '__init__.py'
    var_5 = '# regular package'

def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = '__init__.py'
    var_5 = b"__import__('pkg_resources').declare_namespace(__name__)"

def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = '__init__.py'
    var_5 = b'__import__("pkg_resources").declare_namespace(__name__)'

def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = '__init__.py'
    var_5 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"

def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = '__init__.py'
    var_5 = b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)'

def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = 'module.py'
    var_5 = '# some module'

def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = 'setup.cfg'
    var_5 = '[metadata]'

def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = 'pyproject.toml'
    var_5 = '[build-system]'

def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'pkg'

def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = 'data.txt'
    var_5 = 'some data'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 7/18 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 3/12 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 3/11 statements.
# Partially parsed test_is_module_not_found. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module.py'
    var_2 = ''
    var_3 = '__main__'
    var_4 = 'builtins.__import__'
    var_5 = None
    var_6 = lambda *args, **kwargs: var_5

def test_case_0():
    var_0 = 'test_module'
    var_1 = 0
    var_2 = ''

def test_case_0():
    var_0 = 'test_package'
    var_1 = '__init__.py'
    var_2 = ''

def test_case_0():
    var_0 = 'nonexistent_module'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_true. Retrieved 7/21 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'myapp.submodule'
    var_3 = '.py'
    var_4 = [var_0]
    var_5 = 'myapp.submodule.nested'
    var_6 = ()
    var_7 = 'myapp'
    var_8 = ()
    var_9 = 'myapp'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_namespace_package_predicate_evaluates_to_false. Retrieved 15/30 statements.


def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'src_paths'
    var_3 = 'namespace_packages'
    var_4 = 'auto_identify_namespace_packages'
    var_5 = 'supported_extensions'
    var_6 = []
    var_7 = False
    var_8 = '.py'
    var_9 = [var_8]
    var_10 = 'foo.bar'
    var_11 = ()
    var_12 = '.'
    var_13 = 1
    var_14 = *var_11



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_namespace_package_predicate_evaluates_to_true. Retrieved 7/19 statements.


def test_case_0():
    var_0 = 'myapp'
    var_1 = (var_0,)
    var_2 = 'myapp.submodule.nested'
    var_3 = '.'
    var_4 = 'submodule'
    var_5 = (var_4,)
    var_6 = var_1 + var_5



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 5/23 statements.


def test_case_0():
    var_0 = 'importlib.machinery.EXTENSION_SUFFIXES'
    var_1 = []
    var_2 = 'test_module'
    var_3 = '.py'
    var_4 = '__init__.py'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_predicate_at_line_16_evaluates_to_true. Retrieved 8/25 statements.


def test_case_0():
    var_0 = 'mymodule'
    var_1 = '.py'
    var_2 = 'mymodule'
    var_3 = 0
    var_4 = '.'
    var_5 = 1
    var_6 = var_2.split(var_4, var_5)[var_3]
    var_7 = ()
    var_8 = bool(not var_7)
    assert var_8 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_is_module_predicate_line_3_true. Retrieved 7/19 statements.


def test_case_0():
    var_0 = 'test_module.py'
    var_1 = '# test module'
    var_2 = 'test_module'
    var_3 = 'pathlib.Path.exists'
    var_4 = True
    var_5 = lambda self: var_4
    var_6 = '.py'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_is_namespace_package_predicate_line_2. Retrieved 5/27 statements.


def test_case_0():
    var_0 = '__init__.py'
    var_1 = ''
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_is_namespace_package_not_a_package. Retrieved 4/7 statements.
# Partially parsed test_is_namespace_package_regular_package_with_init. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_namespace_single_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_namespace_double_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path_single_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path_double_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_python_files. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_setup_cfg. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_pyproject_toml. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_no_python_files. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_empty_directory. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'nonexistent'

def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'mypkg'
    var_4 = '__init__.py'
    var_5 = '# regular package'

def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'mypkg'
    var_4 = '__init__.py'
    var_5 = "__import__('pkg_resources').declare_namespace(__name__)"

def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'mypkg'
    var_4 = '__init__.py'
    var_5 = '__import__("pkg_resources").declare_namespace(__name__)'

def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'mypkg'
    var_4 = '__init__.py'
    var_5 = "__path__ = __import__('pkgutil').extend_path(__path__, __name__)"

def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'mypkg'
    var_4 = '__init__.py'
    var_5 = '__path__ = __import__("pkgutil").extend_path(__path__, __name__)'

def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'mypkg'
    var_4 = 'module.py'
    var_5 = '# some module'

def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'mypkg'
    var_4 = 'setup.cfg'
    var_5 = '[metadata]'

def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'mypkg'
    var_4 = 'pyproject.toml'
    var_5 = '[build-system]'

def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'mypkg'
    var_4 = 'data.txt'
    var_5 = 'some data'

def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'mypkg'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_forced_separate_predicate_line_2. Retrieved 5/12 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_pattern'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(var_1, **var_2)
    var_4 = 'test_file'
    var_5 = None
    assert var_5 == 'test_pattern'
    var_6 = bool(var_5 is not None)
    assert var_6 is True



# Parsed testcases at query #29
#--------------------------




import isort.settings as module_0
import fnmatch as module_1

def test_case_0():
    var_0 = 'mymodule.py'
    var_1 = 'mymodule'
    var_2 = [var_1]
    var_3 = 'forced_separate'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = 'mymodule'
    var_7 = f'{var_6}*'
    var_8 = module_1.fnmatch(var_0, var_7)
    var_9 = '.'
    var_10 = var_9 + var_7
    var_11 = module_1.fnmatch(var_0, var_10)
    var_12 = var_8 or var_11
    assert var_12 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_src_path_predicate_evaluates_to_true. Retrieved 5/26 statements.


def test_case_0():
    var_0 = '/test/src'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_0]
    var_4 = '/test/src/mymodule'
    var_5 = [var_4]
    var_6 = [var_4]
    var_7 = '/test/src'
    var_8 = [var_7]
    var_9 = 'mymodule'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_namespace_package_predicate_evaluates_to_true. Retrieved 7/26 statements.


def test_case_0():
    var_0 = 'myapp'
    var_1 = 'nested'
    var_2 = '__init__.py'
    var_3 = 'myapp.nested'
    var_4 = ()
    var_5 = 'firstparty'
    var_6 = 'Found in one of the configured src_paths: '



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_src_path_is_module_with_matching_module_name_and_directory. Retrieved 1/8 statements.
# Partially parsed test_src_path_is_module_with_non_matching_module_name. Retrieved 1/8 statements.
# Partially parsed test_src_path_is_module_with_non_directory_path. Retrieved 1/8 statements.
# Partially parsed test_src_path_is_module_with_case_sensitive_not_exists. Retrieved 1/8 statements.
# Partially parsed test_src_path_is_module_all_conditions_false. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 'mymodule'

def test_case_0():
    var_0 = 'othermodule'

def test_case_0():
    var_0 = 'mymodule'

def test_case_0():
    var_0 = 'mymodule'

def test_case_0():
    var_0 = 'module2'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_src_path_predicate_evaluates_to_true. Retrieved 10/23 statements.


def test_case_0():
    var_0 = '.py'
    var_1 = 'test_module.py'
    var_2 = '# test module'
    var_3 = '_is_namespace_package'
    var_4 = False
    var_5 = '_is_module'
    var_6 = True
    var_7 = '_is_package'
    var_8 = '_src_path_is_module'
    var_9 = 'test_module'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_is_namespace_package_predicate_line_6_true. Retrieved 2/21 statements.


def test_case_0():
    var_0 = 'test_package'
    var_1 = frozenset()



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_src_path_predicate_line_26_evaluates_true. Retrieved 3/19 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_0]
    var_4 = 'mymodule'
    var_5 = 'Found in one of the configured src_paths'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 3/7 statements.
# Partially parsed test_is_module_with_init_file. Retrieved 3/8 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 3/11 statements.
# Partially parsed test_is_module_not_found. Retrieved 1/3 statements.
# Partially parsed test_is_module_with_non_module_file. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'module.py'
    var_1 = '# test module'
    var_2 = 'module'

def test_case_0():
    var_0 = 'package'
    var_1 = '__init__.py'
    var_2 = '# init'

def test_case_0():
    var_0 = 0
    var_1 = ''
    var_2 = 'module'

def test_case_0():
    var_0 = 'nonexistent'

def test_case_0():
    var_0 = 'notmodule.txt'
    var_1 = 'some text'
    var_2 = 'notmodule'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_is_namespace_package_not_a_package. Retrieved 4/7 statements.
# Partially parsed test_is_namespace_package_regular_package_with_init. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_double_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path_double_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_python_files. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_no_python_files. Retrieved 4/8 statements.
# Partially parsed test_is_namespace_package_no_init_with_setup_cfg. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_pyproject_toml. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_other_extension. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'not_a_package'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'regular_package'
    var_1 = '__init__.py'
    var_2 = b'# regular package'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = '__init__.py'
    var_2 = b"__import__('pkg_resources').declare_namespace(__name__)"
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = '__init__.py'
    var_2 = b'__import__("pkg_resources").declare_namespace(__name__)'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = '__init__.py'
    var_2 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = '__init__.py'
    var_2 = b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = 'module.py'
    var_2 = b'# some module'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = 'setup.cfg'
    var_2 = b'# config'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = 'pyproject.toml'
    var_2 = b'# config'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = 'module.pyx'
    var_2 = b'# cython module'
    var_3 = 'py'
    var_4 = 'pyx'
    var_5 = [var_3, var_4]
    var_6 = frozenset(var_5)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_is_namespace_package_predicate_line_6_true. Retrieved 6/28 statements.


def test_case_0():
    var_0 = 'test_package'
    var_1 = 'test_module'
    var_2 = 'py'
    var_3 = 'pyx'
    var_4 = [var_2, var_3]
    var_5 = frozenset(var_4)



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_is_namespace_package_returns_true_at_line_4. Retrieved 5/29 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_pkg'
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_src_path_returns_none_when_module_not_found. Retrieved 12/17 statements.
# Partially parsed test_src_path_returns_firstparty_when_module_found_as_file. Retrieved 14/21 statements.
# Partially parsed test_src_path_returns_firstparty_when_package_found. Retrieved 14/23 statements.
# Partially parsed test_src_path_with_nested_module_and_namespace_package. Retrieved 17/30 statements.
# Partially parsed test_src_path_with_custom_src_paths. Retrieved 15/24 statements.
# Partially parsed test_src_path_src_path_is_module_case. Retrieved 12/19 statements.
# Partially parsed test_src_path_with_prefix. Retrieved 17/28 statements.


def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'src_paths'
    var_3 = 'namespace_packages'
    var_4 = 'auto_identify_namespace_packages'
    var_5 = 'supported_extensions'
    var_6 = set()
    var_7 = False
    var_8 = 'py'
    var_9 = [var_8]
    var_10 = frozenset(var_9)
    var_11 = 'nonexistent_module'

def test_case_0():
    var_0 = 'test_module.py'
    var_1 = '# test'
    var_2 = 'Config'
    var_3 = ()
    var_4 = 'src_paths'
    var_5 = 'namespace_packages'
    var_6 = 'auto_identify_namespace_packages'
    var_7 = 'supported_extensions'
    var_8 = set()
    var_9 = False
    var_10 = 'py'
    var_11 = [var_10]
    var_12 = frozenset(var_11)
    var_13 = 'test_module'
    var_14 = 'Found in one of the configured src_paths'

def test_case_0():
    var_0 = 'test_package'
    var_1 = '__init__.py'
    var_2 = '# test'
    var_3 = 'Config'
    var_4 = ()
    var_5 = 'src_paths'
    var_6 = 'namespace_packages'
    var_7 = 'auto_identify_namespace_packages'
    var_8 = 'supported_extensions'
    var_9 = set()
    var_10 = False
    var_11 = 'py'
    var_12 = [var_11]
    var_13 = frozenset(var_12)

def test_case_0():
    var_0 = 'parent'
    var_1 = '__init__.py'
    var_2 = "__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_3 = 'child'
    var_4 = '# test'
    var_5 = 'Config'
    var_6 = ()
    var_7 = 'src_paths'
    var_8 = 'namespace_packages'
    var_9 = 'auto_identify_namespace_packages'
    var_10 = 'supported_extensions'
    var_11 = {var_0}
    var_12 = False
    var_13 = 'py'
    var_14 = [var_13]
    var_15 = frozenset(var_14)
    var_16 = 'parent.child'

def test_case_0():
    var_0 = 'custom_src'
    var_1 = 'custom_module.py'
    var_2 = '# test'
    var_3 = 'Config'
    var_4 = ()
    var_5 = 'src_paths'
    var_6 = 'namespace_packages'
    var_7 = 'auto_identify_namespace_packages'
    var_8 = 'supported_extensions'
    var_9 = set()
    var_10 = False
    var_11 = 'py'
    var_12 = [var_11]
    var_13 = frozenset(var_12)
    var_14 = 'custom_module'

def test_case_0():
    var_0 = 'mymodule'
    var_1 = 'Config'
    var_2 = ()
    var_3 = 'src_paths'
    var_4 = 'namespace_packages'
    var_5 = 'auto_identify_namespace_packages'
    var_6 = 'supported_extensions'
    var_7 = set()
    var_8 = False
    var_9 = 'py'
    var_10 = [var_9]
    var_11 = frozenset(var_10)

import isort.place as module_0

def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'src_paths'
    var_3 = 'namespace_packages'
    var_4 = 'auto_identify_namespace_packages'
    var_5 = 'supported_extensions'
    var_6 = []
    var_7 = set()
    var_8 = False
    var_9 = 'py'
    var_10 = [var_9]
    var_11 = frozenset(var_10)
    var_12 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_11}
    var_13 = type(var_0, var_1, var_12)
    var_14 = var_13()
    var_15 = 'any_module'
    var_16 = module_0._src_path(var_15, var_14)
    assert var_16 is None

def test_case_0():
    var_0 = 'parent'
    var_1 = '__init__.py'
    var_2 = "__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_3 = 'child.py'
    var_4 = '# test'
    var_5 = 'Config'
    var_6 = ()
    var_7 = 'src_paths'
    var_8 = 'namespace_packages'
    var_9 = 'auto_identify_namespace_packages'
    var_10 = 'supported_extensions'
    var_11 = {var_0}
    var_12 = False
    var_13 = 'py'
    var_14 = [var_13]
    var_15 = frozenset(var_14)
    var_16 = 'parent.child'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_is_namespace_package_predicate_line_6_evaluates_to_true. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'test_package'
    var_1 = 'setup.cfg'
    var_2 = '__init__.py'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_is_namespace_package_predicate_at_line_13_true. Retrieved 12/26 statements.


def test_case_0():
    var_0 = 'test_package'
    var_1 = 'module.py'
    var_2 = '# source code'
    var_3 = 0
    var_4 = 'py'
    var_5 = 'pyx'
    var_6 = [var_4, var_5]
    var_7 = frozenset(var_6)
    var_8 = '.'
    var_9 = 'setup.cfg'
    var_10 = 'pyproject.toml'
    var_11 = (var_9, var_10)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_src_path_predicate_at_line_26_evaluates_to_true. Retrieved 3/18 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_0]
    var_4 = 'mymodule'
    var_5 = 'Found in one of the configured src_paths'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_is_namespace_package_not_a_package. Retrieved 5/8 statements.
# Partially parsed test_is_namespace_package_with_init_file_no_namespace_declaration. Retrieved 7/13 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_namespace_single_quote. Retrieved 7/13 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_namespace_double_quote. Retrieved 7/13 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path_single_quote. Retrieved 7/13 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path_double_quote. Retrieved 7/13 statements.
# Partially parsed test_is_namespace_package_no_init_file_with_py_files. Retrieved 7/13 statements.
# Partially parsed test_is_namespace_package_no_init_file_with_pyx_files. Retrieved 7/13 statements.
# Partially parsed test_is_namespace_package_no_init_file_with_setup_cfg. Retrieved 7/13 statements.
# Partially parsed test_is_namespace_package_no_init_file_with_pyproject_toml. Retrieved 7/13 statements.
# Partially parsed test_is_namespace_package_no_init_file_empty_directory. Retrieved 5/9 statements.
# Partially parsed test_is_namespace_package_no_init_file_with_non_src_files. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'py'
    var_1 = 'pyx'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'nonexistent'

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyx'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'mypkg'
    var_5 = '__init__.py'
    var_6 = '# regular package'

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyx'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'mypkg'
    var_5 = '__init__.py'
    var_6 = "__import__('pkg_resources').declare_namespace(__name__)"

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyx'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'mypkg'
    var_5 = '__init__.py'
    var_6 = '__import__("pkg_resources").declare_namespace(__name__)'

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyx'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'mypkg'
    var_5 = '__init__.py'
    var_6 = "__path__ = __import__('pkgutil').extend_path(__path__, __name__)"

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyx'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'mypkg'
    var_5 = '__init__.py'
    var_6 = '__path__ = __import__("pkgutil").extend_path(__path__, __name__)'

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyx'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'mypkg'
    var_5 = 'module.py'
    var_6 = '# some code'

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyx'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'mypkg'
    var_5 = 'module.pyx'
    var_6 = '# cython code'

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyx'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'mypkg'
    var_5 = 'setup.cfg'
    var_6 = '[metadata]'

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyx'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'mypkg'
    var_5 = 'pyproject.toml'
    var_6 = '[project]'

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyx'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'mypkg'

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyx'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'mypkg'
    var_5 = 'readme.txt'
    var_6 = 'readme'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_is_namespace_package_not_a_package. Retrieved 5/8 statements.
# Partially parsed test_is_namespace_package_regular_package_with_init. Retrieved 7/13 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_namespace_single_quotes. Retrieved 7/13 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_namespace_double_quotes. Retrieved 7/13 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path_single_quotes. Retrieved 7/13 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path_double_quotes. Retrieved 7/13 statements.
# Partially parsed test_is_namespace_package_no_init_with_source_files. Retrieved 7/13 statements.
# Partially parsed test_is_namespace_package_no_init_with_setup_cfg. Retrieved 7/13 statements.
# Partially parsed test_is_namespace_package_no_init_with_pyproject_toml. Retrieved 7/13 statements.
# Partially parsed test_is_namespace_package_no_init_no_source_files. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'py'
    var_1 = 'pyi'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'nonexistent'

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyi'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'test_pkg'
    var_5 = '__init__.py'
    var_6 = '# regular package\n'

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyi'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'test_pkg'
    var_5 = '__init__.py'
    var_6 = "__import__('pkg_resources').declare_namespace(__name__)\n"

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyi'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'test_pkg'
    var_5 = '__init__.py'
    var_6 = '__import__("pkg_resources").declare_namespace(__name__)\n'

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyi'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'test_pkg'
    var_5 = '__init__.py'
    var_6 = "__path__ = __import__('pkgutil').extend_path(__path__, __name__)\n"

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyi'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'test_pkg'
    var_5 = '__init__.py'
    var_6 = '__path__ = __import__("pkgutil").extend_path(__path__, __name__)\n'

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyi'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'test_pkg'
    var_5 = 'module.py'
    var_6 = '# module\n'

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyi'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'test_pkg'
    var_5 = 'setup.cfg'
    var_6 = '[metadata]\n'

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyi'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'test_pkg'
    var_5 = 'pyproject.toml'
    var_6 = '[tool]\n'

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyi'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'test_pkg'
    var_5 = 'data.txt'
    var_6 = 'data\n'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_src_path_is_module_predicate_evaluates_to_true. Retrieved 2/14 statements.


def test_case_0():
    var_0 = 'my_module'
    var_1 = 'my_module'



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_src_path_returns_none_when_module_not_found. Retrieved 2/11 statements.
# Partially parsed test_src_path_returns_firstparty_when_module_found. Retrieved 2/12 statements.
# Partially parsed test_src_path_with_nested_module. Retrieved 2/14 statements.
# Partially parsed test_src_path_with_custom_src_paths. Retrieved 2/13 statements.
# Partially parsed test_src_path_with_prefix. Retrieved 4/14 statements.
# Partially parsed test_src_path_checks_namespace_packages. Retrieved 4/14 statements.


def test_case_0():
    var_0 = '/nonexistent/path'
    var_1 = [var_0]
    var_2 = 'nonexistent_module'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'mymodule'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'parent.child'

def test_case_0():
    var_0 = '/custom/src'
    var_1 = [var_0]
    var_2 = 'module'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'module'
    var_3 = 'parent'
    var_4 = (var_3,)

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'parent'
    var_3 = 'child'
    var_4 = (var_2,)



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 8/22 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 4/12 statements.
# Partially parsed test_is_module_with_init_file. Retrieved 2/9 statements.
# Partially parsed test_is_module_not_a_module. Retrieved 3/9 statements.
# Partially parsed test_is_module_with_multiple_extension_suffixes. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = '.py'
    var_2 = 'importlib.machinery.EXTENSION_SUFFIXES'
    var_3 = []
    var_4 = 'builtins.__import__'
    var_5 = lambda *args, **kwargs: __import__(*args, **kwargs)
    var_6 = 'your_module_name'
    var_7 = 'exists_case_sensitive'

def test_case_0():
    var_0 = 'test_module'
    var_1 = '.so'
    var_2 = 'importlib.machinery.EXTENSION_SUFFIXES'
    var_3 = [var_1]

def test_case_0():
    var_0 = 'test_module'
    var_1 = '__init__.py'

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'importlib.machinery.EXTENSION_SUFFIXES'
    var_2 = []

def test_case_0():
    var_0 = 'test_module'
    var_1 = '.pyd'
    var_2 = 'importlib.machinery.EXTENSION_SUFFIXES'
    var_3 = '.so'
    var_4 = '.dylib'
    var_5 = [var_3, var_1, var_4]



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_is_namespace_package_not_a_package. Retrieved 4/7 statements.
# Partially parsed test_is_namespace_package_regular_package_with_init. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_namespace_single_quote. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_namespace_double_quote. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path_single_quote. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path_double_quote. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_python_files. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_setup_cfg. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_pyproject_toml. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_empty_directory. Retrieved 4/8 statements.
# Partially parsed test_is_namespace_package_no_init_with_non_source_files. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'nonexistent'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = '__init__.py'
    var_5 = '# regular package'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = '__init__.py'
    var_5 = b"__import__('pkg_resources').declare_namespace(__name__)"

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = '__init__.py'
    var_5 = b'__import__("pkg_resources").declare_namespace(__name__)'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = '__init__.py'
    var_5 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = '__init__.py'
    var_5 = b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = 'module.py'
    var_5 = '# module'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = 'setup.cfg'
    var_5 = ''

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = 'pyproject.toml'
    var_5 = ''

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'pkg'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = 'readme.txt'
    var_5 = 'readme'



# Parsed testcases at query #50
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.py'
    var_3 = False
    var_4 = True
    assert var_4 is True
    var_5 = var_1.forced_separate
    var_6 = len(var_5)
    var_7 = bool(var_6 > 0)
    assert var_7 is True
    var_8 = var_1.forced_separate[0]
    assert var_8 == '*.py'



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 3/17 statements.


def test_case_0():
    var_0 = 'test_module.py'
    var_1 = '# test module'
    var_2 = 'test_module'



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_is_namespace_package_predicate_line_5. Retrieved 3/14 statements.


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = b"__import__('pkg_resources').declare_namespace(__name__)"



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_is_namespace_package_not_a_package. Retrieved 4/7 statements.
# Partially parsed test_is_namespace_package_regular_package_with_init. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_namespace_single_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_namespace_double_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path_single_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path_double_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_source_files. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_setup_cfg. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_pyproject_toml. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_no_source_files. Retrieved 4/8 statements.
# Partially parsed test_is_namespace_package_no_init_with_non_source_file. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'not_a_package'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'regular_package'
    var_1 = '__init__.py'
    var_2 = '# regular package'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_pkg1'
    var_1 = '__init__.py'
    var_2 = b"__import__('pkg_resources').declare_namespace(__name__)"
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_pkg2'
    var_1 = '__init__.py'
    var_2 = b'__import__("pkg_resources").declare_namespace(__name__)'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_pkg3'
    var_1 = '__init__.py'
    var_2 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_pkg4'
    var_1 = '__init__.py'
    var_2 = b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'has_source'
    var_1 = 'module.py'
    var_2 = '# module'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'has_setup_cfg'
    var_1 = 'setup.cfg'
    var_2 = ''
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'has_pyproject_toml'
    var_1 = 'pyproject.toml'
    var_2 = ''
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'empty_namespace'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'namespace_with_txt'
    var_1 = 'readme.txt'
    var_2 = 'readme'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_is_namespace_package_predicate_line_5. Retrieved 2/12 statements.


def test_case_0():
    var_0 = '__init__.py'
    var_1 = b"__import__('pkg_resources').declare_namespace(__name__)"



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_is_namespace_package_not_a_package. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_regular_package_with_init. Retrieved 6/11 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_namespace_single_quotes. Retrieved 6/11 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_namespace_double_quotes. Retrieved 6/11 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path_single_quotes. Retrieved 6/11 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path_double_quotes. Retrieved 6/11 statements.
# Partially parsed test_is_namespace_package_no_init_with_py_files. Retrieved 6/11 statements.
# Partially parsed test_is_namespace_package_no_init_with_setup_cfg. Retrieved 6/11 statements.
# Partially parsed test_is_namespace_package_no_init_with_pyproject_toml. Retrieved 6/11 statements.
# Partially parsed test_is_namespace_package_no_init_no_py_files. Retrieved 4/7 statements.
# Partially parsed test_is_namespace_package_no_init_with_non_matching_extension. Retrieved 6/11 statements.
# Partially parsed test_is_namespace_package_no_init_with_pyx_extension. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'nonexistent'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'test_pkg'
    var_4 = '__init__.py'
    var_5 = '# regular package'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'test_pkg'
    var_4 = '__init__.py'
    var_5 = b"__import__('pkg_resources').declare_namespace(__name__)"

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'test_pkg'
    var_4 = '__init__.py'
    var_5 = b'__import__("pkg_resources").declare_namespace(__name__)'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'test_pkg'
    var_4 = '__init__.py'
    var_5 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'test_pkg'
    var_4 = '__init__.py'
    var_5 = b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'test_pkg'
    var_4 = 'module.py'
    var_5 = '# module'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'test_pkg'
    var_4 = 'setup.cfg'
    var_5 = '# config'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'test_pkg'
    var_4 = 'pyproject.toml'
    var_5 = '# config'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'test_pkg'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'test_pkg'
    var_4 = 'data.txt'
    var_5 = '# data'

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyx'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'test_pkg'
    var_5 = 'module.pyx'
    var_6 = '# cython'



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_is_namespace_package_predicate_at_line_13_true. Retrieved 7/32 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_pkg'
    var_2 = 'module.py'
    var_3 = '# some code'
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_is_namespace_package_returns_true_for_namespace_package. Retrieved 4/29 statements.


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = b"__import__('pkg_resources').declare_namespace(__name__)"
    var_3 = frozenset()



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_is_namespace_package_returns_true_for_namespace_package. Retrieved 6/15 statements.


def test_case_0():
    var_0 = 'test_package'
    var_1 = '__init__.py'
    var_2 = b"__import__('pkg_resources').declare_namespace(__name__)"
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_src_path_is_module. Retrieved 5/32 statements.


def test_case_0():
    var_0 = 'mymodule'
    var_1 = 'othermodule'
    var_2 = 'mymodule'
    var_3 = 'mymodule'
    var_4 = 'othermodule'



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_is_namespace_package_predicate_line_6. Retrieved 8/26 statements.


def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'test_pkg'
    var_4 = True
    var_5 = '__init__.py'
    var_6 = 'namespace_pkg'
    var_7 = 'pyproject.toml'



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 2/9 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 4/12 statements.
# Partially parsed test_is_module_with_init_file. Retrieved 2/8 statements.
# Partially parsed test_is_module_not_a_module. Retrieved 2/8 statements.


def test_case_0():
    var_0 = '__main__.exists_case_sensitive'
    var_1 = 'test_module'

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = '__main__.exists_case_sensitive'
    var_3 = 'test_module'

def test_case_0():
    var_0 = '__main__.exists_case_sensitive'
    var_1 = 'test_package'

def test_case_0():
    var_0 = '__main__.exists_case_sensitive'
    var_1 = 'not_a_module'



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 2/9 statements.


def test_case_0():
    var_0 = '__main__.exists_case_sensitive'
    var_1 = 'test_module'



# Parsed testcases at query #63
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django.db'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(var_1, **var_2)
    var_4 = 'django.db.models'
    var_5 = module_1._forced_separate(var_4, var_3)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = var_5[0]
    assert var_7 == 'django.db'
    var_8 = 'Matched forced_separate'
    var_9 = bool('Matched forced_separate' in var_5[1])
    assert var_9 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django.*'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(var_1, **var_2)
    var_4 = 'django.db.models'
    var_5 = module_1._forced_separate(var_4, var_3)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = var_5[0]
    assert var_7 == 'django.*'

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django.db'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(var_1, **var_2)
    var_4 = '.django.db.models'
    var_5 = module_1._forced_separate(var_4, var_3)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = var_5[0]
    assert var_7 == 'django.db'

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django.db'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(var_1, **var_2)
    var_4 = 'flask.app'
    var_5 = module_1._forced_separate(var_4, var_3)
    assert var_5 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Config(var_0, **var_1)
    var_3 = 'django.db'
    var_4 = module_1._forced_separate(var_3, var_2)
    assert var_4 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django.db'
    var_1 = 'flask.app'
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Config(var_2, **var_3)
    var_5 = 'django.db.models'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 is not None)
    assert var_7 is True
    var_8 = var_6[0]
    assert var_8 == 'django.db'

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django.db'
    var_1 = 'flask.app'
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Config(var_2, **var_3)
    var_5 = 'flask.app.views'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 is not None)
    assert var_7 is True
    var_8 = var_6[0]
    assert var_8 == 'flask.app'

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'lib?.core'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(var_1, **var_2)
    var_4 = 'lib1.core.utils'
    var_5 = module_1._forced_separate(var_4, var_3)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = var_5[0]
    assert var_7 == 'lib?.core'

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'exact'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(var_1, **var_2)
    var_4 = module_1._forced_separate(var_0, var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True
    var_6 = var_4[0]
    assert var_6 == 'exact'



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_is_namespace_package_not_a_package. Retrieved 4/7 statements.
# Partially parsed test_is_namespace_package_regular_package_with_init. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_namespace_single_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_namespace_double_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path_single_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path_double_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_python_files. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_setup_cfg. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_pyproject_toml. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_no_source_files. Retrieved 4/8 statements.
# Partially parsed test_is_namespace_package_no_init_with_non_source_files. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'nonexistent'

def test_case_0():
    var_0 = 'regular_package'
    var_1 = '__init__.py'
    var_2 = '# regular package'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = '__init__.py'
    var_2 = b"__import__('pkg_resources').declare_namespace(__name__)"
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = '__init__.py'
    var_2 = b'__import__("pkg_resources").declare_namespace(__name__)'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = '__init__.py'
    var_2 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = '__init__.py'
    var_2 = b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = 'module.py'
    var_2 = '# module'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = 'setup.cfg'
    var_2 = ''
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = 'pyproject.toml'
    var_2 = ''
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = 'py'
    var_2 = {var_1}
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = 'readme.txt'
    var_2 = 'readme'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_namespace_packages_predicate_evaluates_to_false. Retrieved 6/23 statements.


def test_case_0():
    var_0 = 'mymodule'
    var_1 = 'submodule.py'
    var_2 = '# submodule'
    var_3 = 'mymodule.submodule'
    var_4 = ()
    var_5 = 'mymodule'



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_is_namespace_package_returns_true_for_namespace_package. Retrieved 6/15 statements.


def test_case_0():
    var_0 = 'test_package'
    var_1 = '__init__.py'
    var_2 = b"__import__('pkg_resources').declare_namespace(__name__)"
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_namespace_package_predicate_evaluates_true. Retrieved 11/24 statements.


def test_case_0():
    var_0 = 'myapp.submodule'
    var_1 = '/src'
    var_2 = [var_1]
    var_3 = '.py'
    var_4 = 'myapp.submodule.nested'
    var_5 = 'myapp'
    var_6 = 'submodule.nested'
    var_7 = [var_6]
    var_8 = var_5 in var_1
    var_9 = False
    var_10 = var_8 or var_9
    var_11 = var_7 and var_10
    assert var_11 is True



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_src_path_returns_none_when_module_not_found. Retrieved 3/8 statements.
# Partially parsed test_src_path_returns_firstparty_when_module_found_as_file. Retrieved 4/14 statements.
# Partially parsed test_src_path_returns_firstparty_when_package_found. Retrieved 4/15 statements.
# Partially parsed test_src_path_with_nested_module_non_namespace. Retrieved 7/20 statements.
# Partially parsed test_src_path_with_multiple_src_paths. Retrieved 5/17 statements.
# Partially parsed test_src_path_with_prefix. Retrieved 6/18 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'nonexistent_module'
    var_3 = '/tmp'
    var_4 = [var_3]

def test_case_0():
    var_0 = 'src'
    var_1 = 'mymodule.py'
    var_2 = '# test module'
    var_3 = 'mymodule'

def test_case_0():
    var_0 = 'src'
    var_1 = 'mypackage'
    var_2 = '__init__.py'
    var_3 = '# package init'

def test_case_0():
    var_0 = 'src'
    var_1 = 'mypackage'
    var_2 = '__init__.py'
    var_3 = '# package init'
    var_4 = 'nested.py'
    var_5 = '# nested module'
    var_6 = 'mypackage.nested'

def test_case_0():
    var_0 = 'src1'
    var_1 = 'src2'
    var_2 = 'mymodule.py'
    var_3 = '# test module'
    var_4 = 'mymodule'

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'nonexistent'
    var_3 = None
    var_4 = module_1._src_path(var_2, var_1, var_3)
    assert var_4 is None

def test_case_0():
    var_0 = 'src'
    var_1 = 'mypackage'
    var_2 = '__init__.py'
    var_3 = '# package init'
    var_4 = 'nested'
    var_5 = (var_1,)



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_src_path_returns_none_when_module_not_found. Retrieved 12/18 statements.
# Partially parsed test_src_path_finds_module_in_src_paths. Retrieved 14/24 statements.
# Partially parsed test_src_path_finds_nested_module. Retrieved 16/30 statements.
# Partially parsed test_src_path_with_multiple_src_paths. Retrieved 16/30 statements.
# Partially parsed test_src_path_with_explicit_src_paths_parameter. Retrieved 15/28 statements.
# Partially parsed test_src_path_with_prefix. Retrieved 16/27 statements.
# Partially parsed test_src_path_finds_python_file_module. Retrieved 14/22 statements.


def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'src_paths'
    var_3 = 'namespace_packages'
    var_4 = 'auto_identify_namespace_packages'
    var_5 = 'supported_extensions'
    var_6 = frozenset()
    var_7 = False
    var_8 = 'py'
    var_9 = [var_8]
    var_10 = frozenset(var_9)
    var_11 = 'nonexistent_module'

def test_case_0():
    var_0 = 'my_module'
    var_1 = '__init__.py'
    var_2 = ''
    var_3 = 'Config'
    var_4 = ()
    var_5 = 'src_paths'
    var_6 = 'namespace_packages'
    var_7 = 'auto_identify_namespace_packages'
    var_8 = 'supported_extensions'
    var_9 = frozenset()
    var_10 = False
    var_11 = 'py'
    var_12 = [var_11]
    var_13 = frozenset(var_12)

def test_case_0():
    var_0 = 'my_package'
    var_1 = '__init__.py'
    var_2 = ''
    var_3 = 'submodule'
    var_4 = 'Config'
    var_5 = ()
    var_6 = 'src_paths'
    var_7 = 'namespace_packages'
    var_8 = 'auto_identify_namespace_packages'
    var_9 = 'supported_extensions'
    var_10 = frozenset()
    var_11 = False
    var_12 = 'py'
    var_13 = [var_12]
    var_14 = frozenset(var_13)
    var_15 = 'my_package.submodule'

def test_case_0():
    var_0 = 'src1'
    var_1 = 'src2'
    var_2 = 'target_module'
    var_3 = '__init__.py'
    var_4 = ''
    var_5 = 'Config'
    var_6 = ()
    var_7 = 'src_paths'
    var_8 = 'namespace_packages'
    var_9 = 'auto_identify_namespace_packages'
    var_10 = 'supported_extensions'
    var_11 = frozenset()
    var_12 = False
    var_13 = 'py'
    var_14 = [var_13]
    var_15 = frozenset(var_14)

def test_case_0():
    var_0 = 'custom_src'
    var_1 = 'my_module'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'Config'
    var_5 = ()
    var_6 = 'src_paths'
    var_7 = 'namespace_packages'
    var_8 = 'auto_identify_namespace_packages'
    var_9 = 'supported_extensions'
    var_10 = frozenset()
    var_11 = False
    var_12 = 'py'
    var_13 = [var_12]
    var_14 = frozenset(var_13)

def test_case_0():
    var_0 = 'my_package'
    var_1 = '__init__.py'
    var_2 = ''
    var_3 = 'Config'
    var_4 = ()
    var_5 = 'src_paths'
    var_6 = 'namespace_packages'
    var_7 = 'auto_identify_namespace_packages'
    var_8 = 'supported_extensions'
    var_9 = frozenset()
    var_10 = False
    var_11 = 'py'
    var_12 = [var_11]
    var_13 = frozenset(var_12)
    var_14 = 'submodule'
    var_15 = (var_0,)

def test_case_0():
    var_0 = 'my_module.py'
    var_1 = ''
    var_2 = 'Config'
    var_3 = ()
    var_4 = 'src_paths'
    var_5 = 'namespace_packages'
    var_6 = 'auto_identify_namespace_packages'
    var_7 = 'supported_extensions'
    var_8 = frozenset()
    var_9 = False
    var_10 = 'py'
    var_11 = [var_10]
    var_12 = frozenset(var_11)
    var_13 = 'my_module'



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_is_namespace_package_not_a_package. Retrieved 4/7 statements.
# Partially parsed test_is_namespace_package_regular_package_with_init. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_double_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path_double_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_py_files. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_setup_cfg. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_no_src_files. Retrieved 4/8 statements.
# Partially parsed test_is_namespace_package_no_init_with_pyproject_toml. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'nonexistent'

def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'mypkg'
    var_4 = '__init__.py'
    var_5 = '# regular package'

def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'mypkg'
    var_4 = '__init__.py'
    var_5 = b"__import__('pkg_resources').declare_namespace(__name__)"

def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'mypkg'
    var_4 = '__init__.py'
    var_5 = b'__import__("pkg_resources").declare_namespace(__name__)'

def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'mypkg'
    var_4 = '__init__.py'
    var_5 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"

def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'mypkg'
    var_4 = '__init__.py'
    var_5 = b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)'

def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'mypkg'
    var_4 = 'module.py'
    var_5 = '# some module'

def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'mypkg'
    var_4 = 'setup.cfg'
    var_5 = '[metadata]'

def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'mypkg'

def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'mypkg'
    var_4 = 'pyproject.toml'
    var_5 = '[tool.poetry]'



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 4/24 statements.


def test_case_0():
    var_0 = 'builtins.__import__'
    var_1 = 'test_module'
    var_2 = '.py'
    var_3 = '__init__.py'



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_src_path_returns_none_when_module_not_found. Retrieved 4/13 statements.
# Partially parsed test_src_path_returns_firstparty_when_module_found. Retrieved 5/18 statements.
# Partially parsed test_src_path_with_nested_module_non_namespace. Retrieved 5/18 statements.
# Partially parsed test_src_path_with_custom_src_paths. Retrieved 6/20 statements.
# Partially parsed test_src_path_with_empty_prefix. Retrieved 6/19 statements.
# Partially parsed test_src_path_with_prefix. Retrieved 7/20 statements.


def test_case_0():
    var_0 = '/nonexistent/path'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = 'nonexistent_module'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = '/src/mymodule'
    var_5 = [var_4]
    var_6 = 'mymodule'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = '/src/package'
    var_5 = [var_4]
    var_6 = 'package.submodule'

def test_case_0():
    var_0 = '/src1'
    var_1 = [var_0]
    var_2 = '/src2'
    var_3 = [var_2]
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = '/src2/mymodule'
    var_7 = [var_6]
    var_8 = 'mymodule'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = '/src/mymodule'
    var_5 = [var_4]
    var_6 = 'mymodule'
    var_7 = ()

def test_case_0():
    var_0 = '/src/package'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = '/src/package/submodule'
    var_5 = [var_4]
    var_6 = 'submodule'
    var_7 = 'package'
    var_8 = (var_7,)



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_is_namespace_package_not_a_package. Retrieved 4/7 statements.
# Partially parsed test_is_namespace_package_regular_package_with_init. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_namespace_single_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_namespace_double_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path_single_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path_double_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_no_source_files. Retrieved 4/8 statements.
# Partially parsed test_is_namespace_package_no_init_with_py_file. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_setup_cfg. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_pyproject_toml. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_non_matching_extension. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'nonexistent'

def test_case_0():
    var_0 = 'regular_package'
    var_1 = '__init__.py'
    var_2 = '# regular package'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = '__init__.py'
    var_2 = b"__import__('pkg_resources').declare_namespace(__name__)"
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = '__init__.py'
    var_2 = b'__import__("pkg_resources").declare_namespace(__name__)'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = '__init__.py'
    var_2 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = '__init__.py'
    var_2 = b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = 'py'
    var_2 = {var_1}
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = 'module.py'
    var_2 = '# module'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = 'setup.cfg'
    var_2 = '[metadata]'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = 'pyproject.toml'
    var_2 = '[build-system]'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = 'file.txt'
    var_2 = 'content'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 2/16 statements.
# Partially parsed test_is_module_with_init_file. Retrieved 2/17 statements.
# Partially parsed test_is_module_not_a_module. Retrieved 1/13 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 2/19 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = '.py'

def test_case_0():
    var_0 = 'test_package'
    var_1 = '__init__.py'

def test_case_0():
    var_0 = 'not_a_module'

def test_case_0():
    var_0 = 'test_module'
    var_1 = 0



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_is_namespace_package_predicate_line_13_true. Retrieved 4/25 statements.


def test_case_0():
    var_0 = 'test_package'
    var_1 = 'setup.cfg'
    var_2 = ''
    var_3 = frozenset()



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_src_path_is_module. Retrieved 1/18 statements.


def test_case_0():
    var_0 = 'mymodule'



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_src_path_predicate_line_26_evaluates_to_true. Retrieved 3/18 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_0]
    var_4 = 'mymodule'
    var_5 = 'Found in one of the configured src_paths'



# Parsed testcases at query #78
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 'test_*'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(var_1, **var_2)
    var_4 = var_3.forced_separate
    var_5 = bool(var_4)
    assert var_5 is True



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_namespace_package_predicate_evaluates_to_true. Retrieved 12/19 statements.


def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'src_paths'
    var_3 = 'namespace_packages'
    var_4 = 'auto_identify_namespace_packages'
    var_5 = 'supported_extensions'
    var_6 = 'myapp.submodule'
    var_7 = [var_6]
    var_8 = False
    var_9 = '.py'
    var_10 = [var_9]
    var_11 = 'myapp.submodule.nested'



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_src_path_predicate_line_26_evaluates_to_true. Retrieved 8/17 statements.


def test_case_0():
    var_0 = 'src'
    var_1 = 'mymodule.py'
    var_2 = '# test module'
    var_3 = []
    var_4 = False
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = 'mymodule'
    var_8 = 'Found in one of the configured src_paths'



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_is_namespace_package_predicate_at_line_5. Retrieved 3/14 statements.


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = b"__import__('pkg_resources').declare_namespace(__name__)"



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_is_namespace_package_not_a_package. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_regular_package_with_init. Retrieved 6/11 statements.
# Partially parsed test_is_namespace_package_namespace_with_pkg_resources_single_quotes. Retrieved 6/11 statements.
# Partially parsed test_is_namespace_package_namespace_with_pkg_resources_double_quotes. Retrieved 6/11 statements.
# Partially parsed test_is_namespace_package_namespace_with_pkgutil_single_quotes. Retrieved 6/11 statements.
# Partially parsed test_is_namespace_package_namespace_with_pkgutil_double_quotes. Retrieved 6/11 statements.
# Partially parsed test_is_namespace_package_no_init_with_source_files. Retrieved 6/11 statements.
# Partially parsed test_is_namespace_package_no_init_with_setup_cfg. Retrieved 6/11 statements.
# Partially parsed test_is_namespace_package_no_init_with_pyproject_toml. Retrieved 6/11 statements.
# Partially parsed test_is_namespace_package_no_init_empty_directory. Retrieved 4/7 statements.
# Partially parsed test_is_namespace_package_no_init_with_other_extensions. Retrieved 6/11 statements.
# Partially parsed test_is_namespace_package_init_with_other_content. Retrieved 6/11 statements.
# Partially parsed test_is_namespace_package_large_init_file. Retrieved 10/15 statements.


def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'nonexistent'

def test_case_0():
    var_0 = 'regular_package'
    var_1 = '__init__.py'
    var_2 = '# regular package'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = '__init__.py'
    var_2 = b"__import__('pkg_resources').declare_namespace(__name__)"
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = '__init__.py'
    var_2 = b'__import__("pkg_resources").declare_namespace(__name__)'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = '__init__.py'
    var_2 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = '__init__.py'
    var_2 = b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = 'module.py'
    var_2 = '# some code'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = 'setup.cfg'
    var_2 = '[metadata]'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = 'pyproject.toml'
    var_2 = '[build-system]'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = 'py'
    var_2 = {var_1}
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = 'file.txt'
    var_2 = 'some text'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = '__init__.py'
    var_2 = b'some other content'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = '__init__.py'
    var_2 = b'x'
    var_3 = 5000
    var_4 = var_2 * var_3
    var_5 = b"__import__('pkg_resources').declare_namespace(__name__)"
    var_6 = var_4 + var_5
    var_7 = 'py'
    var_8 = {var_7}
    var_9 = frozenset(var_8)



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_is_namespace_package_predicate_at_line_5. Retrieved 3/13 statements.


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = b"__import__('pkg_resources').declare_namespace(__name__)"



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 2/8 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 2/9 statements.
# Partially parsed test_is_module_with_init_file. Retrieved 2/8 statements.
# Partially parsed test_is_module_not_found. Retrieved 2/8 statements.


def test_case_0():
    var_0 = '__main__.exists_case_sensitive'
    var_1 = 'test_module'

def test_case_0():
    var_0 = '__main__.exists_case_sensitive'
    var_1 = 'test_module'

def test_case_0():
    var_0 = '__main__.exists_case_sensitive'
    var_1 = 'test_package'

def test_case_0():
    var_0 = '__main__.exists_case_sensitive'
    var_1 = 'not_a_module'



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_forced_separate_predicate_evaluates_to_true. Retrieved 5/17 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(var_1, **var_2)
    var_4 = 'test_module'
    var_5 = '.'



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_src_path_returns_none_when_module_not_found. Retrieved 12/17 statements.
# Partially parsed test_src_path_returns_firstparty_when_module_found. Retrieved 13/22 statements.
# Partially parsed test_src_path_with_nested_module. Retrieved 15/28 statements.
# Partially parsed test_src_path_with_py_file_module. Retrieved 13/20 statements.
# Partially parsed test_src_path_with_custom_src_paths. Retrieved 14/26 statements.
# Partially parsed test_src_path_with_prefix. Retrieved 15/29 statements.
# Partially parsed test_src_path_with_src_path_is_module. Retrieved 12/20 statements.


def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'src_paths'
    var_3 = 'namespace_packages'
    var_4 = 'auto_identify_namespace_packages'
    var_5 = 'supported_extensions'
    var_6 = frozenset()
    var_7 = False
    var_8 = 'py'
    var_9 = [var_8]
    var_10 = frozenset(var_9)
    var_11 = 'nonexistent'

def test_case_0():
    var_0 = 'mymodule'
    var_1 = '__init__.py'
    var_2 = 'Config'
    var_3 = ()
    var_4 = 'src_paths'
    var_5 = 'namespace_packages'
    var_6 = 'auto_identify_namespace_packages'
    var_7 = 'supported_extensions'
    var_8 = frozenset()
    var_9 = False
    var_10 = 'py'
    var_11 = [var_10]
    var_12 = frozenset(var_11)

def test_case_0():
    var_0 = 'parent'
    var_1 = '__init__.py'
    var_2 = 'child'
    var_3 = 'Config'
    var_4 = ()
    var_5 = 'src_paths'
    var_6 = 'namespace_packages'
    var_7 = 'auto_identify_namespace_packages'
    var_8 = 'supported_extensions'
    var_9 = frozenset()
    var_10 = False
    var_11 = 'py'
    var_12 = [var_11]
    var_13 = frozenset(var_12)
    var_14 = 'parent.child'

def test_case_0():
    var_0 = 'mymodule.py'
    var_1 = 'Config'
    var_2 = ()
    var_3 = 'src_paths'
    var_4 = 'namespace_packages'
    var_5 = 'auto_identify_namespace_packages'
    var_6 = 'supported_extensions'
    var_7 = frozenset()
    var_8 = False
    var_9 = 'py'
    var_10 = [var_9]
    var_11 = frozenset(var_10)
    var_12 = 'mymodule'

def test_case_0():
    var_0 = 'custom_src'
    var_1 = 'mymodule'
    var_2 = '__init__.py'
    var_3 = 'Config'
    var_4 = ()
    var_5 = 'src_paths'
    var_6 = 'namespace_packages'
    var_7 = 'auto_identify_namespace_packages'
    var_8 = 'supported_extensions'
    var_9 = frozenset()
    var_10 = False
    var_11 = 'py'
    var_12 = [var_11]
    var_13 = frozenset(var_12)

def test_case_0():
    var_0 = 'parent'
    var_1 = '__init__.py'
    var_2 = 'child'
    var_3 = 'Config'
    var_4 = ()
    var_5 = 'src_paths'
    var_6 = 'namespace_packages'
    var_7 = 'auto_identify_namespace_packages'
    var_8 = 'supported_extensions'
    var_9 = frozenset()
    var_10 = False
    var_11 = 'py'
    var_12 = [var_11]
    var_13 = frozenset(var_12)
    var_14 = (var_0,)

def test_case_0():
    var_0 = 'mymodule'
    var_1 = 'Config'
    var_2 = ()
    var_3 = 'src_paths'
    var_4 = 'namespace_packages'
    var_5 = 'auto_identify_namespace_packages'
    var_6 = 'supported_extensions'
    var_7 = frozenset()
    var_8 = False
    var_9 = 'py'
    var_10 = [var_9]
    var_11 = frozenset(var_10)



# Parsed testcases at query #87
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 3/12 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 4/18 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 3/13 statements.
# Partially parsed test_is_module_not_a_module. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module.py'
    var_2 = '__main__.exists_case_sensitive'

def test_case_0():
    var_0 = 'test_module'
    var_1 = 0
    var_2 = '.so'
    var_3 = '__main__.exists_case_sensitive'

def test_case_0():
    var_0 = 'test_package'
    var_1 = '__init__.py'
    var_2 = '__main__.exists_case_sensitive'

def test_case_0():
    var_0 = 'not_a_module'
    var_1 = '__main__.exists_case_sensitive'



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_src_path_is_module_predicate_evaluates_to_true. Retrieved 3/16 statements.


def test_case_0():
    var_0 = 'mymodule'
    var_1 = '__main__.exists_case_sensitive'
    var_2 = 'mymodule'



# Parsed testcases at query #89
#--------------------------

# Partially parsed test_src_path_predicate_at_line_26_evaluates_to_true. Retrieved 8/38 statements.


def test_case_0():
    var_0 = 'test_module.py'
    var_1 = '# test module'
    var_2 = '_is_module'
    var_3 = None
    var_4 = '_is_package'
    var_5 = '_src_path_is_module'
    var_6 = '_is_namespace_package'
    var_7 = 'test_module'



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_namespace_package_predicate_line_6_evaluates_to_true. Retrieved 4/17 statements.


def test_case_0():
    var_0 = 'test_package'
    var_1 = 'module.py'
    var_2 = '# some module'
    var_3 = '__init__.py'



# Parsed testcases at query #91
#--------------------------

# Partially parsed test_src_path_predicate_evaluates_to_true. Retrieved 3/22 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_0]
    var_4 = 'mymodule'



# Parsed testcases at query #92
#--------------------------

# Partially parsed test_is_namespace_package_predicate_line_13_true. Retrieved 7/32 statements.


def test_case_0():
    var_0 = 'test_package'
    var_1 = '__main__._is_package'
    var_2 = 'module.py'
    var_3 = '# some code'
    var_4 = 'py'
    var_5 = {var_4}
    var_6 = frozenset(var_5)



# Parsed testcases at query #93
#--------------------------

# Partially parsed test_namespace_package_predicate_evaluates_to_true. Retrieved 3/16 statements.


def test_case_0():
    var_0 = 'mypackage'
    var_1 = 'submodule'
    var_2 = 'mypackage.submodule'



# Parsed testcases at query #94
#--------------------------

# Partially parsed test_is_namespace_package_not_a_package. Retrieved 4/7 statements.
# Partially parsed test_is_namespace_package_regular_package_with_init. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_namespace_single_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_namespace_double_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path_single_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path_double_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_py_files. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_setup_cfg. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_pyproject_toml. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_no_config_files. Retrieved 4/8 statements.
# Partially parsed test_is_namespace_package_no_init_with_other_extension. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'nonexistent'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = '__init__.py'
    var_5 = '# regular package'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = '__init__.py'
    var_5 = "__import__('pkg_resources').declare_namespace(__name__)"

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = '__init__.py'
    var_5 = '__import__("pkg_resources").declare_namespace(__name__)'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = '__init__.py'
    var_5 = "__path__ = __import__('pkgutil').extend_path(__path__, __name__)"

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = '__init__.py'
    var_5 = '__path__ = __import__("pkgutil").extend_path(__path__, __name__)'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = 'module.py'
    var_5 = '# module'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = 'setup.cfg'
    var_5 = ''

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = 'pyproject.toml'
    var_5 = ''

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'pkg'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = 'data.txt'
    var_5 = 'data'



# Parsed testcases at query #95
#--------------------------

# Partially parsed test_src_path_predicate_at_line_26_evaluates_to_true. Retrieved 3/20 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_0]
    var_4 = 'mymodule'
    var_5 = True



# Parsed testcases at query #96
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(var_1, **var_2)
    var_4 = module_1._forced_separate(var_0, var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True
    var_6 = var_4[0]
    assert var_6 == 'django'
    var_7 = 'Matched forced_separate'
    var_8 = bool('Matched forced_separate' in var_4[1])
    assert var_8 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django*'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(var_1, **var_2)
    var_4 = 'django.db'
    var_5 = module_1._forced_separate(var_4, var_3)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = var_5[0]
    assert var_7 == 'django*'

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(var_1, **var_2)
    var_4 = '.django'
    var_5 = module_1._forced_separate(var_4, var_3)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = var_5[0]
    assert var_7 == 'django'

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(var_1, **var_2)
    var_4 = 'flask'
    var_5 = module_1._forced_separate(var_4, var_3)
    assert var_5 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django'
    var_1 = 'flask'
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Config(var_2, **var_3)
    var_5 = 'django.models'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 is not None)
    assert var_7 is True
    var_8 = var_6[0]
    assert var_8 == 'django'

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django'
    var_1 = 'flask'
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Config(var_2, **var_3)
    var_5 = 'flask.app'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 is not None)
    assert var_7 is True
    var_8 = var_6[0]
    assert var_8 == 'flask'

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Config(var_0, **var_1)
    var_3 = 'django'
    var_4 = module_1._forced_separate(var_3, var_2)
    assert var_4 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'lib.*.utils'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(var_1, **var_2)
    var_4 = 'lib.core.utils'
    var_5 = module_1._forced_separate(var_4, var_3)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = var_5[0]
    assert var_7 == 'lib.*.utils'

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'myapp'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(var_1, **var_2)
    var_4 = 'myapp.views.forms'
    var_5 = module_1._forced_separate(var_4, var_3)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = var_5[0]
    assert var_7 == 'myapp'



# Parsed testcases at query #97
#--------------------------

# Partially parsed test_namespace_package_predicate_evaluates_to_false. Retrieved 8/18 statements.


def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = 'mymodule'
    var_5 = '__init__.py'
    var_6 = ''
    var_7 = 'mymodule.submodule'
    var_8 = ()



# Parsed testcases at query #98
#--------------------------

# Partially parsed test_is_namespace_package_predicate_line_6_true. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'test_package'
    var_1 = 'some_module.py'
    var_2 = '# module'
    var_3 = '__init__.py'



# Parsed testcases at query #99
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 3/12 statements.


def test_case_0():
    var_0 = 'importlib.machinery.EXTENSION_SUFFIXES'
    var_1 = []
    var_2 = 'test_module'



# Parsed testcases at query #100
#--------------------------

# Partially parsed test_src_path_returns_none_when_module_not_found. Retrieved 4/13 statements.
# Partially parsed test_src_path_returns_firstparty_when_module_found. Retrieved 6/23 statements.
# Partially parsed test_src_path_with_nested_module_and_namespace_package. Retrieved 6/21 statements.
# Partially parsed test_src_path_with_auto_identify_namespace_packages. Retrieved 5/21 statements.
# Partially parsed test_src_path_with_src_path_is_module. Retrieved 4/17 statements.
# Partially parsed test_src_path_with_custom_src_paths. Retrieved 4/17 statements.
# Partially parsed test_src_path_with_prefix. Retrieved 6/20 statements.


def test_case_0():
    var_0 = 'src'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = 'nonexistent_module'

def test_case_0():
    var_0 = 'src'
    var_1 = 'mymodule'
    var_2 = '__init__.py'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = 'mymodule'

def test_case_0():
    var_0 = 'src'
    var_1 = 'parent'
    var_2 = [var_1]
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = 'parent.child'

def test_case_0():
    var_0 = 'src'
    var_1 = 'parent'
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = 'parent.child'

def test_case_0():
    var_0 = 'src'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = 'src'

def test_case_0():
    var_0 = 'custom'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = 'mymodule'

def test_case_0():
    var_0 = 'src'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = 'child'
    var_4 = 'parent'
    var_5 = (var_4,)



# Parsed testcases at query #101
#--------------------------

# Partially parsed test_is_namespace_package_predicate_line_5. Retrieved 2/12 statements.


def test_case_0():
    var_0 = '__init__.py'
    var_1 = ''



# Parsed testcases at query #102
#--------------------------

# Partially parsed test_src_path_predicate_at_line_26_evaluates_to_true. Retrieved 3/17 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_0]
    var_4 = 'mymodule'



# Parsed testcases at query #103
#--------------------------

# Partially parsed test_src_path_is_module_with_matching_directory. Retrieved 2/11 statements.
# Partially parsed test_src_path_is_module_with_non_matching_name. Retrieved 2/11 statements.
# Partially parsed test_src_path_is_module_with_file_not_directory. Retrieved 3/12 statements.
# Partially parsed test_src_path_is_module_case_sensitive_check_fails. Retrieved 2/11 statements.
# Partially parsed test_src_path_is_module_nonexistent_path. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'mymodule'
    var_1 = 'mymodule'

def test_case_0():
    var_0 = 'mymodule'
    var_1 = 'differentname'

def test_case_0():
    var_0 = 'mymodule'
    var_1 = 'content'
    var_2 = 'mymodule'

def test_case_0():
    var_0 = 'mymodule'
    var_1 = 'mymodule'

def test_case_0():
    var_0 = '/nonexistent/path/mymodule'
    var_1 = [var_0]
    var_2 = 'mymodule'



# Parsed testcases at query #104
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 7/17 statements.
# Partially parsed test_is_module_with_init_file. Retrieved 3/12 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 2/8 statements.
# Partially parsed test_is_module_not_found. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = '.py'
    var_2 = 'builtins.__import__'
    var_3 = None
    var_4 = lambda *args, **kwargs: var_3
    var_5 = []
    var_6 = 'exists_case_sensitive'

def test_case_0():
    var_0 = 'test_package'
    var_1 = '__init__.py'
    var_2 = 'exists_case_sensitive'

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'exists_case_sensitive'

def test_case_0():
    var_0 = 'nonexistent_module'
    var_1 = 'exists_case_sensitive'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 6/18 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 7/17 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 6/18 statements.
# Partially parsed test_is_module_not_a_module. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module.py'
    var_2 = ''
    var_3 = 'importlib.machinery.EXTENSION_SUFFIXES'
    var_4 = []
    var_5 = '.exists_case_sensitive'

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module.so'
    var_2 = ''
    var_3 = 'importlib.machinery.EXTENSION_SUFFIXES'
    var_4 = '.so'
    var_5 = [var_4]
    var_6 = '.exists_case_sensitive'

def test_case_0():
    var_0 = 'test_package'
    var_1 = '__init__.py'
    var_2 = ''
    var_3 = 'importlib.machinery.EXTENSION_SUFFIXES'
    var_4 = []
    var_5 = '.exists_case_sensitive'

def test_case_0():
    var_0 = 'not_a_module'
    var_1 = 'importlib.machinery.EXTENSION_SUFFIXES'
    var_2 = []
    var_3 = '.exists_case_sensitive'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_is_module_predicate_evaluates_to_true. Retrieved 7/24 statements.


def test_case_0():
    var_0 = 'pathlib.Path.with_suffix'
    var_1 = lambda self, suffix: Path(str(self) + suffix)
    var_2 = 'test_module'
    var_3 = 'test_module.py'
    var_4 = ''
    var_5 = 'builtins.__import__'
    var_6 = lambda name, *args, **kwargs: __import__(name, *args, **kwargs)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_is_namespace_package_not_a_package. Retrieved 4/7 statements.
# Partially parsed test_is_namespace_package_regular_package_with_init. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_declare. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_single_quote. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_double_quote. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_double_quote. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_python_files. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_setup_cfg. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_pyproject_toml. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_empty_dir. Retrieved 4/8 statements.
# Partially parsed test_is_namespace_package_no_init_non_python_files. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'nonexistent'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'mypkg'
    var_4 = '__init__.py'
    var_5 = '# regular package'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'mypkg'
    var_4 = '__init__.py'
    var_5 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'mypkg'
    var_4 = '__init__.py'
    var_5 = b"__import__('pkg_resources').declare_namespace(__name__)"

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'mypkg'
    var_4 = '__init__.py'
    var_5 = b'__import__("pkg_resources").declare_namespace(__name__)'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'mypkg'
    var_4 = '__init__.py'
    var_5 = b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'mypkg'
    var_4 = 'module.py'
    var_5 = '# some code'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'mypkg'
    var_4 = 'setup.cfg'
    var_5 = '[metadata]'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'mypkg'
    var_4 = 'pyproject.toml'
    var_5 = '[tool.poetry]'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'mypkg'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'mypkg'
    var_4 = 'data.txt'
    var_5 = 'some data'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_is_namespace_package_not_a_package. Retrieved 5/8 statements.
# Partially parsed test_is_namespace_package_regular_package_with_init. Retrieved 7/13 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_namespace_single_quotes. Retrieved 7/13 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_namespace_double_quotes. Retrieved 7/13 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path_single_quotes. Retrieved 7/13 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path_double_quotes. Retrieved 7/13 statements.
# Partially parsed test_is_namespace_package_no_init_with_source_files. Retrieved 7/13 statements.
# Partially parsed test_is_namespace_package_no_init_with_setup_cfg. Retrieved 7/13 statements.
# Partially parsed test_is_namespace_package_no_init_with_pyproject_toml. Retrieved 7/13 statements.
# Partially parsed test_is_namespace_package_no_init_empty_directory. Retrieved 5/9 statements.
# Partially parsed test_is_namespace_package_no_init_with_non_source_files. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'py'
    var_1 = 'pyx'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'nonexistent'

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyx'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'pkg'
    var_5 = '__init__.py'
    var_6 = '# regular package'

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyx'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'pkg'
    var_5 = '__init__.py'
    var_6 = b"__import__('pkg_resources').declare_namespace(__name__)"

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyx'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'pkg'
    var_5 = '__init__.py'
    var_6 = b'__import__("pkg_resources").declare_namespace(__name__)'

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyx'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'pkg'
    var_5 = '__init__.py'
    var_6 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyx'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'pkg'
    var_5 = '__init__.py'
    var_6 = b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)'

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyx'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'pkg'
    var_5 = 'module.py'
    var_6 = '# some code'

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyx'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'pkg'
    var_5 = 'setup.cfg'
    var_6 = '# setup'

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyx'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'pkg'
    var_5 = 'pyproject.toml'
    var_6 = '# project'

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyx'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'pkg'

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyx'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'pkg'
    var_5 = 'readme.txt'
    var_6 = 'readme'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_known_pattern_matches_exact_module. Retrieved 21/25 statements.
# Partially parsed test_known_pattern_matches_submodule. Retrieved 19/22 statements.
# Partially parsed test_known_pattern_no_match. Retrieved 19/21 statements.
# Partially parsed test_known_pattern_section_not_in_config. Retrieved 21/23 statements.
# Partially parsed test_known_pattern_matches_longest_prefix_first. Retrieved 27/29 statements.
# Partially parsed test_known_pattern_multiple_patterns_first_match_wins. Retrieved 26/28 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'Config'
    var_1 = 'known_patterns'
    var_2 = 'sections'
    var_3 = [var_1, var_2]
    var_4 = 'Pattern'
    var_5 = 'pattern'
    var_6 = [var_5]
    var_7 = 'PatternObj'
    var_8 = ()
    var_9 = 'match'
    var_10 = 'django'
    var_11 = lambda self, x: x == var_10
    var_12 = {var_9: var_11}
    var_13 = type(var_7, var_8, var_12)
    var_14 = var_13()
    var_15 = 'third_party'
    var_16 = (var_14, var_15)
    var_17 = [var_16]
    var_18 = [var_15]
    var_19 = 'known_patterns'
    var_20 = 'sections'
    var_21 = {var_19: var_17, var_20: var_18}
    var_22 = module_0.Config(**var_21)
    var_23 = module_1._known_pattern(var_10, var_22)
    var_24 = bool(var_23 is not None)
    assert var_24 is True
    var_25 = var_23[0]
    assert var_25 == 'third_party'
    var_26 = 'Matched configured known pattern'
    var_27 = bool('Matched configured known pattern' in var_23[1])
    assert var_27 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'Config'
    var_1 = 'known_patterns'
    var_2 = 'sections'
    var_3 = [var_1, var_2]
    var_4 = 'PatternObj'
    var_5 = ()
    var_6 = 'match'
    var_7 = 'django'
    var_8 = lambda self, x: x.startswith(var_7)
    var_9 = {var_6: var_8}
    var_10 = type(var_4, var_5, var_9)
    var_11 = var_10()
    var_12 = 'third_party'
    var_13 = (var_11, var_12)
    var_14 = [var_13]
    var_15 = [var_12]
    var_16 = 'known_patterns'
    var_17 = 'sections'
    var_18 = {var_16: var_14, var_17: var_15}
    var_19 = module_0.Config(**var_18)
    var_20 = 'django.conf.settings'
    var_21 = module_1._known_pattern(var_20, var_19)
    var_22 = bool(var_21 is not None)
    assert var_22 is True
    var_23 = var_21[0]
    assert var_23 == 'third_party'
    var_24 = 'Matched configured known pattern'
    var_25 = bool('Matched configured known pattern' in var_21[1])
    assert var_25 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'Config'
    var_1 = 'known_patterns'
    var_2 = 'sections'
    var_3 = [var_1, var_2]
    var_4 = 'PatternObj'
    var_5 = ()
    var_6 = 'match'
    var_7 = False
    var_8 = lambda self, x: var_7
    var_9 = {var_6: var_8}
    var_10 = type(var_4, var_5, var_9)
    var_11 = var_10()
    var_12 = 'third_party'
    var_13 = (var_11, var_12)
    var_14 = [var_13]
    var_15 = [var_12]
    var_16 = 'known_patterns'
    var_17 = 'sections'
    var_18 = {var_16: var_14, var_17: var_15}
    var_19 = module_0.Config(**var_18)
    var_20 = 'mymodule'
    var_21 = module_1._known_pattern(var_20, var_19)
    assert var_21 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'Config'
    var_1 = 'known_patterns'
    var_2 = 'sections'
    var_3 = [var_1, var_2]
    var_4 = 'PatternObj'
    var_5 = ()
    var_6 = 'match'
    var_7 = True
    var_8 = lambda self, x: var_7
    var_9 = {var_6: var_8}
    var_10 = type(var_4, var_5, var_9)
    var_11 = var_10()
    var_12 = 'invalid_section'
    var_13 = (var_11, var_12)
    var_14 = [var_13]
    var_15 = 'third_party'
    var_16 = 'stdlib'
    var_17 = [var_15, var_16]
    var_18 = 'known_patterns'
    var_19 = 'sections'
    var_20 = {var_18: var_14, var_19: var_17}
    var_21 = module_0.Config(**var_20)
    var_22 = 'mymodule'
    var_23 = module_1._known_pattern(var_22, var_21)
    assert var_23 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'Config'
    var_1 = 'known_patterns'
    var_2 = 'sections'
    var_3 = [var_1, var_2]
    var_4 = 'PatternObj'
    var_5 = ()
    var_6 = 'match'
    var_7 = 'django.conf'
    var_8 = lambda self, x: x == var_7
    var_9 = {var_6: var_8}
    var_10 = type(var_4, var_5, var_9)
    var_11 = var_10()
    var_12 = ()
    var_13 = 'django'
    var_14 = lambda self, x: x == var_13
    var_15 = {var_6: var_14}
    var_16 = type(var_4, var_12, var_15)
    var_17 = var_16()
    var_18 = 'third_party'
    var_19 = (var_11, var_18)
    var_20 = 'stdlib'
    var_21 = (var_17, var_20)
    var_22 = [var_19, var_21]
    var_23 = [var_18, var_20]
    var_24 = 'known_patterns'
    var_25 = 'sections'
    var_26 = {var_24: var_22, var_25: var_23}
    var_27 = module_0.Config(**var_26)
    var_28 = 'django.conf.settings'
    var_29 = module_1._known_pattern(var_28, var_27)
    var_30 = bool(var_29 is not None)
    assert var_30 is True
    var_31 = var_29[0]
    assert var_31 == 'third_party'

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'Config'
    var_1 = 'known_patterns'
    var_2 = 'sections'
    var_3 = [var_1, var_2]
    var_4 = 'PatternObj'
    var_5 = ()
    var_6 = 'match'
    var_7 = 'test'
    var_8 = lambda self, x: x.startswith(var_7)
    var_9 = {var_6: var_8}
    var_10 = type(var_4, var_5, var_9)
    var_11 = var_10()
    var_12 = ()
    var_13 = lambda self, x: x.startswith(var_7)
    var_14 = {var_6: var_13}
    var_15 = type(var_4, var_12, var_14)
    var_16 = var_15()
    var_17 = 'firstparty'
    var_18 = (var_11, var_17)
    var_19 = 'thirdparty'
    var_20 = (var_16, var_19)
    var_21 = [var_18, var_20]
    var_22 = [var_17, var_19]
    var_23 = 'known_patterns'
    var_24 = 'sections'
    var_25 = {var_23: var_21, var_24: var_22}
    var_26 = module_0.Config(**var_25)
    var_27 = 'test.module'
    var_28 = module_1._known_pattern(var_27, var_26)
    var_29 = bool(var_28 is not None)
    assert var_29 is True
    var_30 = var_28[0]
    assert var_30 == 'firstparty'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_src_path_is_module_returns_true_when_module_exists. Retrieved 2/8 statements.
# Partially parsed test_src_path_is_module_returns_false_when_names_dont_match. Retrieved 2/8 statements.
# Partially parsed test_src_path_is_module_returns_false_when_not_directory. Retrieved 2/8 statements.
# Partially parsed test_src_path_is_module_returns_false_when_case_sensitive_check_fails. Retrieved 2/8 statements.


def test_case_0():
    var_0 = '/test/mymodule'
    var_1 = [var_0]
    var_2 = 'mymodule'

def test_case_0():
    var_0 = '/test/mymodule'
    var_1 = [var_0]
    var_2 = 'othermodule'

def test_case_0():
    var_0 = '/test/mymodule'
    var_1 = [var_0]
    var_2 = 'mymodule'

def test_case_0():
    var_0 = '/test/mymodule'
    var_1 = [var_0]
    var_2 = 'mymodule'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_src_path_finds_module_in_src_paths. Retrieved 5/19 statements.
# Partially parsed test_src_path_finds_package_in_src_paths. Retrieved 5/21 statements.
# Partially parsed test_src_path_returns_none_when_module_not_found. Retrieved 3/11 statements.
# Partially parsed test_src_path_with_nested_module_in_namespace_package. Retrieved 7/25 statements.
# Partially parsed test_src_path_uses_provided_src_paths. Retrieved 6/22 statements.
# Partially parsed test_src_path_with_empty_prefix. Retrieved 6/20 statements.


def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = 'test_module.py'
    var_3 = '# test module'
    var_4 = 'test_module'
    var_5 = 'Found in one of the configured src_paths'

def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = 'test_package'
    var_3 = '__init__.py'
    var_4 = ''

def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = 'nonexistent_module'

def test_case_0():
    var_0 = 'parent'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = 'child'
    var_4 = '__init__.py'
    var_5 = ''
    var_6 = 'parent.child'

def test_case_0():
    var_0 = 'custom_src'
    var_1 = 'my_module.py'
    var_2 = '# module'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = 'my_module'

def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = 'simple_module.py'
    var_3 = ''
    var_4 = 'simple_module'
    var_5 = ()



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 10/29 statements.


def test_case_0():
    var_0 = 'importlib.machinery.EXTENSION_SUFFIXES'
    var_1 = []
    var_2 = 'test_module'
    var_3 = 'builtins.__import__'
    var_4 = None
    var_5 = lambda *args, **kwargs: var_4
    var_6 = 'test_module.py'
    var_7 = ''
    var_8 = '.py'
    var_9 = '__init__.py'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_is_namespace_package_not_a_package. Retrieved 5/8 statements.
# Partially parsed test_is_namespace_package_regular_package_with_init. Retrieved 7/13 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_namespace_single_quotes. Retrieved 7/13 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_namespace_double_quotes. Retrieved 7/13 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path_single_quotes. Retrieved 7/13 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path_double_quotes. Retrieved 7/13 statements.
# Partially parsed test_is_namespace_package_no_init_file_with_python_source. Retrieved 7/13 statements.
# Partially parsed test_is_namespace_package_no_init_file_with_cython_source. Retrieved 7/13 statements.
# Partially parsed test_is_namespace_package_no_init_file_with_setup_cfg. Retrieved 7/13 statements.
# Partially parsed test_is_namespace_package_no_init_file_with_pyproject_toml. Retrieved 7/13 statements.
# Partially parsed test_is_namespace_package_no_init_file_empty_directory. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'py'
    var_1 = 'pyx'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'nonexistent'

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyx'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'my_package'
    var_5 = '__init__.py'
    var_6 = '# regular package'

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyx'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'my_package'
    var_5 = '__init__.py'
    var_6 = b"__import__('pkg_resources').declare_namespace(__name__)"

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyx'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'my_package'
    var_5 = '__init__.py'
    var_6 = b'__import__("pkg_resources").declare_namespace(__name__)'

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyx'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'my_package'
    var_5 = '__init__.py'
    var_6 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyx'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'my_package'
    var_5 = '__init__.py'
    var_6 = b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)'

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyx'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'my_package'
    var_5 = 'module.py'
    var_6 = '# some module'

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyx'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'my_package'
    var_5 = 'module.pyx'
    var_6 = '# some cython module'

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyx'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'my_package'
    var_5 = 'setup.cfg'
    var_6 = '[metadata]'

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyx'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'my_package'
    var_5 = 'pyproject.toml'
    var_6 = '[build-system]'

def test_case_0():
    var_0 = 'py'
    var_1 = 'pyx'
    var_2 = {var_0, var_1}
    var_3 = frozenset(var_2)
    var_4 = 'my_package'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_src_path_returns_none_when_module_not_found. Retrieved 12/18 statements.
# Partially parsed test_src_path_returns_firstparty_when_module_is_file. Retrieved 14/22 statements.
# Partially parsed test_src_path_returns_firstparty_when_package_exists. Retrieved 14/24 statements.
# Partially parsed test_src_path_with_nested_module_in_package. Retrieved 17/29 statements.
# Partially parsed test_src_path_with_custom_src_paths. Retrieved 15/26 statements.
# Partially parsed test_src_path_with_prefix_parameter. Retrieved 16/27 statements.
# Partially parsed test_src_path_src_path_is_module_case. Retrieved 12/20 statements.


def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'src_paths'
    var_3 = 'namespace_packages'
    var_4 = 'auto_identify_namespace_packages'
    var_5 = 'supported_extensions'
    var_6 = []
    var_7 = False
    var_8 = 'py'
    var_9 = [var_8]
    var_10 = frozenset(var_9)
    var_11 = 'nonexistent_module'

def test_case_0():
    var_0 = 'mymodule.py'
    var_1 = '# test module'
    var_2 = 'Config'
    var_3 = ()
    var_4 = 'src_paths'
    var_5 = 'namespace_packages'
    var_6 = 'auto_identify_namespace_packages'
    var_7 = 'supported_extensions'
    var_8 = []
    var_9 = False
    var_10 = 'py'
    var_11 = [var_10]
    var_12 = frozenset(var_11)
    var_13 = 'mymodule'

def test_case_0():
    var_0 = 'mypackage'
    var_1 = '__init__.py'
    var_2 = '# init'
    var_3 = 'Config'
    var_4 = ()
    var_5 = 'src_paths'
    var_6 = 'namespace_packages'
    var_7 = 'auto_identify_namespace_packages'
    var_8 = 'supported_extensions'
    var_9 = []
    var_10 = False
    var_11 = 'py'
    var_12 = [var_11]
    var_13 = frozenset(var_12)

def test_case_0():
    var_0 = 'mypackage'
    var_1 = '__init__.py'
    var_2 = '# init'
    var_3 = 'nested.py'
    var_4 = '# nested'
    var_5 = 'Config'
    var_6 = ()
    var_7 = 'src_paths'
    var_8 = 'namespace_packages'
    var_9 = 'auto_identify_namespace_packages'
    var_10 = 'supported_extensions'
    var_11 = []
    var_12 = False
    var_13 = 'py'
    var_14 = [var_13]
    var_15 = frozenset(var_14)
    var_16 = 'mypackage.nested'

def test_case_0():
    var_0 = 'custom_src'
    var_1 = 'testmod.py'
    var_2 = '# test'
    var_3 = 'Config'
    var_4 = ()
    var_5 = 'src_paths'
    var_6 = 'namespace_packages'
    var_7 = 'auto_identify_namespace_packages'
    var_8 = 'supported_extensions'
    var_9 = []
    var_10 = False
    var_11 = 'py'
    var_12 = [var_11]
    var_13 = frozenset(var_12)
    var_14 = 'testmod'

def test_case_0():
    var_0 = 'parent'
    var_1 = '__init__.py'
    var_2 = '# init'
    var_3 = 'Config'
    var_4 = ()
    var_5 = 'src_paths'
    var_6 = 'namespace_packages'
    var_7 = 'auto_identify_namespace_packages'
    var_8 = 'supported_extensions'
    var_9 = []
    var_10 = False
    var_11 = 'py'
    var_12 = [var_11]
    var_13 = frozenset(var_12)
    var_14 = 'child'
    var_15 = (var_0,)

def test_case_0():
    var_0 = 'mymodule'
    var_1 = 'Config'
    var_2 = ()
    var_3 = 'src_paths'
    var_4 = 'namespace_packages'
    var_5 = 'auto_identify_namespace_packages'
    var_6 = 'supported_extensions'
    var_7 = []
    var_8 = False
    var_9 = 'py'
    var_10 = [var_9]
    var_11 = frozenset(var_10)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_predicate_at_line_16_evaluates_to_true. Retrieved 11/31 statements.


def test_case_0():
    var_0 = '.py'
    var_1 = 'mymodule'
    var_2 = 'mymodule.py'
    var_3 = '# module'
    var_4 = '# module file'
    var_5 = 'mymodule'
    var_6 = ()
    var_7 = 0
    var_8 = '.'
    var_9 = 1
    var_10 = var_5.split(var_8, var_9)[var_7]
    var_11 = bool(not var_6)
    assert var_11 is True



# Parsed testcases at query #12
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django.db'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(var_1, **var_2)
    var_4 = 'django.db.models'
    var_5 = module_1._forced_separate(var_4, var_3)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = var_5[0]
    assert var_7 == 'django.db'
    var_8 = 'Matched forced_separate'
    var_9 = bool('Matched forced_separate' in var_5[1])
    assert var_9 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django.db*'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(var_1, **var_2)
    var_4 = 'django.db.models'
    var_5 = module_1._forced_separate(var_4, var_3)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = var_5[0]
    assert var_7 == 'django.db*'

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django.db'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(var_1, **var_2)
    var_4 = '.django.db.models'
    var_5 = module_1._forced_separate(var_4, var_3)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = var_5[0]
    assert var_7 == 'django.db'

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django.db'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(var_1, **var_2)
    var_4 = 'flask.app'
    var_5 = module_1._forced_separate(var_4, var_3)
    assert var_5 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Config(var_0, **var_1)
    var_3 = 'django.db.models'
    var_4 = module_1._forced_separate(var_3, var_2)
    assert var_4 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django.db'
    var_1 = 'flask.app'
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Config(var_2, **var_3)
    var_5 = 'django.db.models'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 is not None)
    assert var_7 is True
    var_8 = var_6[0]
    assert var_8 == 'django.db'

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django.db'
    var_1 = 'flask.app'
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Config(var_2, **var_3)
    var_5 = 'flask.app.views'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 is not None)
    assert var_7 is True
    var_8 = var_6[0]
    assert var_8 == 'flask.app'

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(var_1, **var_2)
    var_4 = module_1._forced_separate(var_0, var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True
    var_6 = var_4[0]
    assert var_6 == 'django'

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django.d?'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(var_1, **var_2)
    var_4 = 'django.db'
    var_5 = module_1._forced_separate(var_4, var_3)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = var_5[0]
    assert var_7 == 'django.d?'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_src_path_is_module_returns_true_when_conditions_met. Retrieved 2/12 statements.


def test_case_0():
    var_0 = 'mymodule'
    var_1 = True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_true. Retrieved 5/22 statements.


def test_case_0():
    var_0 = 'my.namespace'
    var_1 = '/src'
    var_2 = [var_1]
    var_3 = '.py'
    var_4 = 'my.namespace.module'
    var_5 = [var_1]
    var_6 = ()



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 3/14 statements.


def test_case_0():
    var_0 = 'importlib.machinery.EXTENSION_SUFFIXES'
    var_1 = []
    var_2 = 'test_module'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_src_path_predicate_line_7_false. Retrieved 3/14 statements.


def test_case_0():
    var_0 = '/some/path'
    var_1 = [var_0]
    var_2 = '/test/path'
    var_3 = [var_2]
    var_4 = 'test_module'



# Parsed testcases at query #17
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django.db'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(var_1, **var_2)
    var_4 = module_1._forced_separate(var_0, var_3)
    var_5 = bool(var_4 == ('django.db', 'Matched forced_separate (django.db) config value.'))
    assert var_5 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django.db'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(var_1, **var_2)
    var_4 = 'django.db.models'
    var_5 = module_1._forced_separate(var_4, var_3)
    var_6 = bool(var_5 == ('django.db', 'Matched forced_separate (django.db) config value.'))
    assert var_6 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django.db'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(var_1, **var_2)
    var_4 = '.django.db.models'
    var_5 = module_1._forced_separate(var_4, var_3)
    var_6 = bool(var_5 == ('django.db', 'Matched forced_separate (django.db) config value.'))
    assert var_6 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django.*'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(var_1, **var_2)
    var_4 = 'django.db'
    var_5 = module_1._forced_separate(var_4, var_3)
    var_6 = bool(var_5 == ('django.*', 'Matched forced_separate (django.*) config value.'))
    assert var_6 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django.db'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(var_1, **var_2)
    var_4 = 'flask.app'
    var_5 = module_1._forced_separate(var_4, var_3)
    assert var_5 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Config(var_0, **var_1)
    var_3 = 'django.db'
    var_4 = module_1._forced_separate(var_3, var_2)
    assert var_4 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django.db'
    var_1 = 'flask.app'
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Config(var_2, **var_3)
    var_5 = 'django.db.models'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('django.db', 'Matched forced_separate (django.db) config value.'))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django.db'
    var_1 = 'flask.app'
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Config(var_2, **var_3)
    var_5 = 'flask.app.routes'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('flask.app', 'Matched forced_separate (flask.app) config value.'))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test.?'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(var_1, **var_2)
    var_4 = 'test.a'
    var_5 = module_1._forced_separate(var_4, var_3)
    var_6 = bool(var_5 == ('test.?', 'Matched forced_separate (test.?) config value.'))
    assert var_6 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'Django.db'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(var_1, **var_2)
    var_4 = 'django.db'
    var_5 = module_1._forced_separate(var_4, var_3)
    assert var_5 is None



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_src_path_predicate_line_26_evaluates_to_true. Retrieved 4/23 statements.


def test_case_0():
    var_0 = '.py'
    var_1 = 'test_module'
    var_2 = '__init__.py'
    var_3 = 'test_module'
    var_4 = 'Found in one of the configured src_paths'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_predicate_at_line_16_evaluates_to_true. Retrieved 7/28 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = False
    var_3 = []
    var_4 = 'mymodule'
    var_5 = ()
    var_6 = 'mymodule'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_src_path_is_module_evaluates_to_true. Retrieved 1/12 statements.


def test_case_0():
    var_0 = 'my_module'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_src_paths_is_not_none. Retrieved 5/12 statements.


def test_case_0():
    var_0 = '/default/src'
    var_1 = [var_0]
    var_2 = []
    var_3 = False
    var_4 = []
    var_5 = '/custom/src'
    var_6 = [var_5]
    var_7 = 'test_module'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_namespace_packages_predicate_evaluates_to_false. Retrieved 8/21 statements.


def test_case_0():
    var_0 = 'mymodule'
    var_1 = '__init__.py'
    var_2 = ''
    var_3 = 'mymodule.submodule'
    var_4 = ()
    var_5 = 1
    var_6 = '.'
    var_7 = var_3.split(var_6, var_5)[var_5:]
    var_8 = bool(var_7 == ['submodule'])
    assert var_8 is True
    var_9 = 'mymodule'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_src_path_returns_none_when_module_not_found. Retrieved 12/18 statements.
# Partially parsed test_src_path_returns_firstparty_when_module_found. Retrieved 14/24 statements.
# Partially parsed test_src_path_with_nested_module. Retrieved 16/30 statements.
# Partially parsed test_src_path_with_py_file. Retrieved 14/22 statements.
# Partially parsed test_src_path_with_custom_src_paths. Retrieved 15/27 statements.
# Partially parsed test_src_path_with_empty_prefix. Retrieved 15/25 statements.
# Partially parsed test_src_path_src_path_is_module_case. Retrieved 12/20 statements.


def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'src_paths'
    var_3 = 'namespace_packages'
    var_4 = 'auto_identify_namespace_packages'
    var_5 = 'supported_extensions'
    var_6 = []
    var_7 = False
    var_8 = 'py'
    var_9 = [var_8]
    var_10 = frozenset(var_9)
    var_11 = 'nonexistent_module'

def test_case_0():
    var_0 = 'mymodule'
    var_1 = '__init__.py'
    var_2 = ''
    var_3 = 'Config'
    var_4 = ()
    var_5 = 'src_paths'
    var_6 = 'namespace_packages'
    var_7 = 'auto_identify_namespace_packages'
    var_8 = 'supported_extensions'
    var_9 = []
    var_10 = False
    var_11 = 'py'
    var_12 = [var_11]
    var_13 = frozenset(var_12)
    var_14 = 'src_paths'

def test_case_0():
    var_0 = 'parent'
    var_1 = '__init__.py'
    var_2 = ''
    var_3 = 'child'
    var_4 = 'Config'
    var_5 = ()
    var_6 = 'src_paths'
    var_7 = 'namespace_packages'
    var_8 = 'auto_identify_namespace_packages'
    var_9 = 'supported_extensions'
    var_10 = []
    var_11 = False
    var_12 = 'py'
    var_13 = [var_12]
    var_14 = frozenset(var_13)
    var_15 = 'parent.child'

def test_case_0():
    var_0 = 'mymodule.py'
    var_1 = ''
    var_2 = 'Config'
    var_3 = ()
    var_4 = 'src_paths'
    var_5 = 'namespace_packages'
    var_6 = 'auto_identify_namespace_packages'
    var_7 = 'supported_extensions'
    var_8 = []
    var_9 = False
    var_10 = 'py'
    var_11 = [var_10]
    var_12 = frozenset(var_11)
    var_13 = 'mymodule'

def test_case_0():
    var_0 = 'custom_src'
    var_1 = 'mymodule'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'Config'
    var_5 = ()
    var_6 = 'src_paths'
    var_7 = 'namespace_packages'
    var_8 = 'auto_identify_namespace_packages'
    var_9 = 'supported_extensions'
    var_10 = []
    var_11 = False
    var_12 = 'py'
    var_13 = [var_12]
    var_14 = frozenset(var_13)

def test_case_0():
    var_0 = 'testmodule'
    var_1 = '__init__.py'
    var_2 = ''
    var_3 = 'Config'
    var_4 = ()
    var_5 = 'src_paths'
    var_6 = 'namespace_packages'
    var_7 = 'auto_identify_namespace_packages'
    var_8 = 'supported_extensions'
    var_9 = []
    var_10 = False
    var_11 = 'py'
    var_12 = [var_11]
    var_13 = frozenset(var_12)
    var_14 = ()

def test_case_0():
    var_0 = 'mymodule'
    var_1 = 'Config'
    var_2 = ()
    var_3 = 'src_paths'
    var_4 = 'namespace_packages'
    var_5 = 'auto_identify_namespace_packages'
    var_6 = 'supported_extensions'
    var_7 = []
    var_8 = False
    var_9 = 'py'
    var_10 = [var_9]
    var_11 = frozenset(var_10)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_src_path_returns_none_when_module_not_found. Retrieved 2/11 statements.
# Partially parsed test_src_path_returns_firstparty_when_module_is_package. Retrieved 2/14 statements.
# Partially parsed test_src_path_returns_firstparty_when_module_exists. Retrieved 2/14 statements.
# Partially parsed test_src_path_handles_nested_module_names. Retrieved 3/16 statements.
# Partially parsed test_src_path_with_custom_src_paths. Retrieved 2/15 statements.
# Partially parsed test_src_path_with_src_path_is_module. Retrieved 2/14 statements.


def test_case_0():
    var_0 = '/nonexistent/path'
    var_1 = [var_0]
    var_2 = 'nonexistent_module'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'mymodule'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'mymodule'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'parent'
    var_3 = 'parent.child'

def test_case_0():
    var_0 = '/custom/src'
    var_1 = [var_0]
    var_2 = 'mymodule'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'mymodule'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_src_path_predicate_line_26_true. Retrieved 5/29 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_0]
    var_4 = 'mymodule'
    var_5 = 'mypackage'
    var_6 = 'mymodule'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_is_module_predicate_line_3_true. Retrieved 3/15 statements.


def test_case_0():
    var_0 = 'test_module.py'
    var_1 = '# test module'
    var_2 = 'test_module'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_is_namespace_package_predicate_line_2. Retrieved 4/15 statements.


def test_case_0():
    var_0 = 'py'
    var_1 = 'pyx'
    var_2 = [var_0, var_1]
    var_3 = frozenset(var_2)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_true. Retrieved 8/25 statements.


def test_case_0():
    var_0 = 'myapp.submodule'
    var_1 = '/src'
    var_2 = [var_1]
    var_3 = '.py'
    var_4 = 'myapp.submodule.nested'
    var_5 = [var_1]
    var_6 = ()
    var_7 = '.'
    var_8 = 1
    var_9 = *var_6



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_is_namespace_package_returns_true_for_namespace_package. Retrieved 7/30 statements.


def test_case_0():
    var_0 = 'test_package'
    var_1 = '__init__.py'
    var_2 = b"__import__('pkg_resources').declare_namespace(__name__)"
    var_3 = '__main__._is_package'
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_is_namespace_package_predicate_line_5. Retrieved 3/15 statements.


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = b"__import__('pkg_resources').declare_namespace(__name__)"



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_namespace_package_predicate_evaluates_to_false. Retrieved 4/16 statements.


def test_case_0():
    var_0 = '.py'
    var_1 = 'mypackage'
    var_2 = 'submodule.py'
    var_3 = 'mypackage.submodule'



# Parsed testcases at query #32
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test_file.py'
    var_3 = var_1.forced_separate
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = var_1.forced_separate
    var_6 = len(var_5)
    var_7 = bool(var_6 > 0)
    assert var_7 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_is_namespace_package_predicate_line_4. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = '__init__.py'
    var_4 = b"__import__('pkg_resources').declare_namespace(__name__)"



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_is_namespace_package_predicate_at_line_5. Retrieved 3/14 statements.


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = b"__import__('pkg_resources').declare_namespace(__name__)"



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 5/21 statements.


def test_case_0():
    var_0 = 'importlib.machinery.EXTENSION_SUFFIXES'
    var_1 = []
    var_2 = 'test_module'
    var_3 = '.py'
    var_4 = '__init__.py'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_is_namespace_package_predicate_line_6_true. Retrieved 8/33 statements.


def test_case_0():
    var_0 = 'namespace_pkg'
    var_1 = 'subdir'
    var_2 = 'module.py'
    var_3 = '# module'
    var_4 = 'empty_namespace'
    var_5 = 'py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_is_namespace_package_predicate_line_5. Retrieved 2/12 statements.


def test_case_0():
    var_0 = '__init__.py'
    var_1 = b"__import__('pkg_resources').declare_namespace(__name__)"



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_predicate_at_line_16_evaluates_to_true. Retrieved 4/25 statements.


def test_case_0():
    var_0 = 'mymodule'
    var_1 = 'mymodule.py'
    var_2 = ()
    var_3 = 'mymodule'
    var_4 = bool(not var_2)
    assert var_4 is True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_is_namespace_package_not_a_package. Retrieved 4/7 statements.
# Partially parsed test_is_namespace_package_regular_package_with_init. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_namespace_single_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_namespace_double_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path_single_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path_double_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_without_init_with_py_files. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_without_init_with_setup_cfg. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_without_init_with_pyproject_toml. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_without_init_empty_directory. Retrieved 4/8 statements.
# Partially parsed test_is_namespace_package_without_init_with_non_matching_extensions. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'nonexistent'

def test_case_0():
    var_0 = 'regular_package'
    var_1 = '__init__.py'
    var_2 = '# Regular package'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'ns_package'
    var_1 = '__init__.py'
    var_2 = b"__import__('pkg_resources').declare_namespace(__name__)"
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'ns_package'
    var_1 = '__init__.py'
    var_2 = b'__import__("pkg_resources").declare_namespace(__name__)'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'ns_package'
    var_1 = '__init__.py'
    var_2 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'ns_package'
    var_1 = '__init__.py'
    var_2 = b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'ns_package'
    var_1 = 'module.py'
    var_2 = '# Some module'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'ns_package'
    var_1 = 'setup.cfg'
    var_2 = '[metadata]'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'ns_package'
    var_1 = 'pyproject.toml'
    var_2 = '[project]'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'ns_package'
    var_1 = 'py'
    var_2 = {var_1}
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'ns_package'
    var_1 = 'file.txt'
    var_2 = 'Some text'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_src_path_is_module. Retrieved 2/12 statements.


def test_case_0():
    var_0 = 'mymodule'
    var_1 = True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_is_module_predicate_evaluates_to_true. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'exists_case_sensitive'
    var_1 = 'test_module'
    var_2 = '.py'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 7/24 statements.


def test_case_0():
    var_0 = 'importlib.machinery.EXTENSION_SUFFIXES'
    var_1 = []
    var_2 = 'test_module'
    var_3 = '__main__'
    var_4 = 'exists_case_sensitive'
    var_5 = '.py'
    var_6 = '__init__.py'



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_src_path_returns_none_when_module_not_found. Retrieved 12/17 statements.
# Partially parsed test_src_path_returns_firstparty_when_module_is_file. Retrieved 14/21 statements.
# Partially parsed test_src_path_returns_firstparty_when_module_is_package. Retrieved 14/23 statements.
# Partially parsed test_src_path_with_nested_module_non_namespace. Retrieved 16/29 statements.
# Partially parsed test_src_path_with_multiple_src_paths. Retrieved 16/27 statements.
# Partially parsed test_src_path_with_custom_src_paths_parameter. Retrieved 15/25 statements.
# Partially parsed test_src_path_with_prefix_parameter. Retrieved 16/30 statements.


def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'src_paths'
    var_3 = 'namespace_packages'
    var_4 = 'auto_identify_namespace_packages'
    var_5 = 'supported_extensions'
    var_6 = []
    var_7 = False
    var_8 = 'py'
    var_9 = [var_8]
    var_10 = frozenset(var_9)
    var_11 = 'nonexistent_module'

def test_case_0():
    var_0 = 'mymodule.py'
    var_1 = '# test module'
    var_2 = 'Config'
    var_3 = ()
    var_4 = 'src_paths'
    var_5 = 'namespace_packages'
    var_6 = 'auto_identify_namespace_packages'
    var_7 = 'supported_extensions'
    var_8 = []
    var_9 = False
    var_10 = 'py'
    var_11 = [var_10]
    var_12 = frozenset(var_11)
    var_13 = 'mymodule'
    var_14 = 'Found in one of the configured src_paths'

def test_case_0():
    var_0 = 'mypackage'
    var_1 = '__init__.py'
    var_2 = '# package init'
    var_3 = 'Config'
    var_4 = ()
    var_5 = 'src_paths'
    var_6 = 'namespace_packages'
    var_7 = 'auto_identify_namespace_packages'
    var_8 = 'supported_extensions'
    var_9 = []
    var_10 = False
    var_11 = 'py'
    var_12 = [var_11]
    var_13 = frozenset(var_12)

def test_case_0():
    var_0 = 'parent'
    var_1 = '__init__.py'
    var_2 = ''
    var_3 = 'child'
    var_4 = 'Config'
    var_5 = ()
    var_6 = 'src_paths'
    var_7 = 'namespace_packages'
    var_8 = 'auto_identify_namespace_packages'
    var_9 = 'supported_extensions'
    var_10 = []
    var_11 = False
    var_12 = 'py'
    var_13 = [var_12]
    var_14 = frozenset(var_13)
    var_15 = 'parent.child'

def test_case_0():
    var_0 = 'src1'
    var_1 = 'src2'
    var_2 = 'mymodule.py'
    var_3 = '# test'
    var_4 = 'Config'
    var_5 = ()
    var_6 = 'src_paths'
    var_7 = 'namespace_packages'
    var_8 = 'auto_identify_namespace_packages'
    var_9 = 'supported_extensions'
    var_10 = []
    var_11 = False
    var_12 = 'py'
    var_13 = [var_12]
    var_14 = frozenset(var_13)
    var_15 = 'mymodule'

def test_case_0():
    var_0 = 'custom_src'
    var_1 = 'testmod.py'
    var_2 = '# test'
    var_3 = 'Config'
    var_4 = ()
    var_5 = 'src_paths'
    var_6 = 'namespace_packages'
    var_7 = 'auto_identify_namespace_packages'
    var_8 = 'supported_extensions'
    var_9 = []
    var_10 = False
    var_11 = 'py'
    var_12 = [var_11]
    var_13 = frozenset(var_12)
    var_14 = 'testmod'

def test_case_0():
    var_0 = 'parent'
    var_1 = '__init__.py'
    var_2 = ''
    var_3 = 'child'
    var_4 = 'Config'
    var_5 = ()
    var_6 = 'src_paths'
    var_7 = 'namespace_packages'
    var_8 = 'auto_identify_namespace_packages'
    var_9 = 'supported_extensions'
    var_10 = []
    var_11 = False
    var_12 = 'py'
    var_13 = [var_12]
    var_14 = frozenset(var_13)
    var_15 = (var_0,)



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_src_path_returns_none_when_module_not_found. Retrieved 2/11 statements.
# Partially parsed test_src_path_returns_firstparty_when_module_found. Retrieved 3/17 statements.
# Partially parsed test_src_path_with_nested_module. Retrieved 5/23 statements.
# Partially parsed test_src_path_with_py_file. Retrieved 3/15 statements.
# Partially parsed test_src_path_uses_provided_src_paths. Retrieved 4/20 statements.
# Partially parsed test_src_path_with_prefix. Retrieved 5/20 statements.


def test_case_0():
    var_0 = 'src'
    var_1 = 'nonexistent_module'

def test_case_0():
    var_0 = 'src'
    var_1 = 'mymodule'
    var_2 = '__init__.py'

def test_case_0():
    var_0 = 'src'
    var_1 = 'mypackage'
    var_2 = '__init__.py'
    var_3 = 'nested'
    var_4 = 'mypackage.nested'

def test_case_0():
    var_0 = 'src'
    var_1 = 'mymodule.py'
    var_2 = 'mymodule'

def test_case_0():
    var_0 = 'custom_src'
    var_1 = 'mymodule'
    var_2 = '__init__.py'
    var_3 = 'default'

def test_case_0():
    var_0 = 'src'
    var_1 = 'mypackage'
    var_2 = '__init__.py'
    var_3 = 'submodule'
    var_4 = (var_1,)



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_is_namespace_package_not_a_package. Retrieved 4/8 statements.
# Partially parsed test_is_namespace_package_regular_package_with_init. Retrieved 6/13 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_namespace_single_quotes. Retrieved 6/13 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_namespace_double_quotes. Retrieved 6/13 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path_single_quotes. Retrieved 6/13 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path_double_quotes. Retrieved 6/13 statements.
# Partially parsed test_is_namespace_package_no_init_with_py_files. Retrieved 6/13 statements.
# Partially parsed test_is_namespace_package_no_init_no_py_files. Retrieved 4/9 statements.
# Partially parsed test_is_namespace_package_no_init_with_setup_cfg. Retrieved 6/13 statements.
# Partially parsed test_is_namespace_package_no_init_with_pyproject_toml. Retrieved 6/13 statements.
# Partially parsed test_is_namespace_package_no_init_with_other_extension. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'not_a_package'
    var_1 = 'py'
    var_2 = {var_1}
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'regular_package'
    var_1 = '__init__.py'
    var_2 = '# regular package'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = '__init__.py'
    var_2 = b"__import__('pkg_resources').declare_namespace(__name__)"
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = '__init__.py'
    var_2 = b'__import__("pkg_resources").declare_namespace(__name__)'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = '__init__.py'
    var_2 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = '__init__.py'
    var_2 = b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'package_with_py'
    var_1 = 'module.py'
    var_2 = '# module'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = 'py'
    var_2 = {var_1}
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'package_with_setup'
    var_1 = 'setup.cfg'
    var_2 = '# setup'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'package_with_pyproject'
    var_1 = 'pyproject.toml'
    var_2 = '# pyproject'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = 'file.txt'
    var_2 = '# text file'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_src_path_predicate_line_26_evaluates_to_true. Retrieved 11/35 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'mymodule.py'
    var_1 = '# test module'
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = '.py'
    var_5 = 'sections.FIRSTPARTY'
    var_6 = 'FIRSTPARTY'
    var_7 = '_is_module'
    var_8 = '_is_package'
    var_9 = '_src_path_is_module'
    var_10 = '_is_namespace_package'
    var_11 = 'mymodule'
    var_12 = 'Found in one of the configured src_paths'



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_src_path_predicate_line_26_evaluates_to_true. Retrieved 3/18 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_0]
    var_4 = 'mymodule'
    var_5 = 'Found in one of the configured src_paths'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_src_path_is_module. Retrieved 6/17 statements.


def test_case_0():
    var_0 = 'mymodule'
    var_1 = '/tmp/mymodule'
    var_2 = 'mymodule'
    var_3 = var_0 == var_2
    var_4 = True
    var_5 = var_3 and var_4 and var_4
    assert var_5 is True



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_predicate_at_line_13_evaluates_to_true. Retrieved 14/28 statements.


def test_case_0():
    var_0 = 'test_package'
    var_1 = 'module.py'
    var_2 = '# some code'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)
    var_6 = 'namespace_pkg'
    var_7 = 'module.py'
    var_8 = 'code'
    var_9 = '.'
    var_10 = 'setup.cfg'
    var_11 = 'pyproject.toml'
    var_12 = (var_10, var_11)
    var_13 = [filepath for filepath in var_3 if filepath.suffix.lstrip(var_9) in var_5 or filepath.name.lower() in var_12]
    var_14 = bool(var_13)
    assert var_14 is True



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_is_namespace_package_not_a_package. Retrieved 6/14 statements.
# Partially parsed test_is_namespace_package_regular_package_with_init. Retrieved 7/16 statements.
# Partially parsed test_is_namespace_package_namespace_with_pkg_resources. Retrieved 7/15 statements.
# Partially parsed test_is_namespace_package_namespace_with_pkgutil. Retrieved 7/15 statements.
# Partially parsed test_is_namespace_package_no_init_no_src_files. Retrieved 5/11 statements.
# Partially parsed test_is_namespace_package_no_init_with_src_files. Retrieved 7/15 statements.
# Partially parsed test_is_namespace_package_no_init_with_pyproject_toml. Retrieved 7/15 statements.
# Partially parsed test_is_namespace_package_namespace_with_pkgutil_double_quotes. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 'builtins.__import__'
    var_1 = 'not_package'
    var_2 = '__main__.exists_case_sensitive'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'regular_package'
    var_1 = '__init__.py'
    var_2 = b'# regular package'
    var_3 = '__main__.exists_case_sensitive'
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = '__init__.py'
    var_2 = b"__import__('pkg_resources').declare_namespace(__name__)"
    var_3 = '__main__.exists_case_sensitive'
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = '__init__.py'
    var_2 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_3 = '__main__.exists_case_sensitive'
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = '__main__.exists_case_sensitive'
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = 'module.py'
    var_2 = '# some code'
    var_3 = '__main__.exists_case_sensitive'
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = 'pyproject.toml'
    var_2 = '[build-system]'
    var_3 = '__main__.exists_case_sensitive'
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = '__init__.py'
    var_2 = b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    var_3 = '__main__.exists_case_sensitive'
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_is_namespace_package_predicate_at_line_6_true. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'test_package'
    var_1 = 'module.py'
    var_2 = '# module'
    var_3 = '__init__.py'



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_is_namespace_package_not_a_package. Retrieved 4/7 statements.
# Partially parsed test_is_namespace_package_regular_package_with_init. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_double_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path_double_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_python_files. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_setup_cfg. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_pyproject_toml. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_empty_directory. Retrieved 4/8 statements.
# Partially parsed test_is_namespace_package_no_init_with_non_python_files. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'nonexistent'

def test_case_0():
    var_0 = 'regular_package'
    var_1 = '__init__.py'
    var_2 = '# regular package'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_pkg'
    var_1 = '__init__.py'
    var_2 = b"__import__('pkg_resources').declare_namespace(__name__)"
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_pkg'
    var_1 = '__init__.py'
    var_2 = b'__import__("pkg_resources").declare_namespace(__name__)'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_pkg'
    var_1 = '__init__.py'
    var_2 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_pkg'
    var_1 = '__init__.py'
    var_2 = b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_pkg'
    var_1 = 'module.py'
    var_2 = '# some module'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_pkg'
    var_1 = 'setup.cfg'
    var_2 = '[metadata]'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_pkg'
    var_1 = 'pyproject.toml'
    var_2 = '[build-system]'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_pkg'
    var_1 = 'py'
    var_2 = {var_1}
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'namespace_pkg'
    var_1 = 'readme.txt'
    var_2 = 'readme'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)



# Parsed testcases at query #53
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 'test_pattern'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = False
    var_6 = True
    assert var_6 is True



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_is_namespace_package_not_a_package. Retrieved 4/7 statements.
# Partially parsed test_is_namespace_package_regular_package_with_init. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_namespace. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_namespace_double_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path_double_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_py_files. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_setup_cfg. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_pyproject_toml. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_empty_directory. Retrieved 4/8 statements.
# Partially parsed test_is_namespace_package_no_init_with_non_source_files. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'nonexistent'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = '__init__.py'
    var_5 = b'# regular package'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = '__init__.py'
    var_5 = b"__import__('pkg_resources').declare_namespace(__name__)"

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = '__init__.py'
    var_5 = b'__import__("pkg_resources").declare_namespace(__name__)'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = '__init__.py'
    var_5 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = '__init__.py'
    var_5 = b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = 'module.py'
    var_5 = '# module'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = 'setup.cfg'
    var_5 = ''

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = 'pyproject.toml'
    var_5 = ''

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'pkg'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = 'readme.txt'
    var_5 = 'readme'



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_src_path_is_module. Retrieved 2/12 statements.


def test_case_0():
    var_0 = 'my_module'
    var_1 = True



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_namespace_packages_predicate_evaluates_to_false. Retrieved 7/16 statements.


def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = 'mymodule'
    var_5 = '__init__.py'
    var_6 = ''
    var_7 = 'mymodule.submodule'



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_is_namespace_package_returns_true_at_line_4. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'test_package'
    var_1 = True
    assert var_1 is False



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_src_path_returns_none_when_module_not_found. Retrieved 2/14 statements.
# Partially parsed test_src_path_returns_firstparty_when_module_found. Retrieved 2/16 statements.
# Partially parsed test_src_path_returns_firstparty_when_package_found. Retrieved 2/16 statements.
# Partially parsed test_src_path_handles_nested_module_with_namespace_package. Retrieved 7/24 statements.
# Partially parsed test_src_path_with_src_path_is_module_match. Retrieved 2/16 statements.
# Partially parsed test_src_path_with_custom_src_paths. Retrieved 2/16 statements.


def test_case_0():
    var_0 = '/nonexistent/path'
    var_1 = [var_0]
    var_2 = 'nonexistent_module'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'mymodule'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'mypackage'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'parent'
    var_3 = [var_2]
    var_4 = None
    var_5 = 'Found in one of the configured src_paths: /src'
    var_6 = (var_2, var_5)
    var_7 = 'parent.child'

def test_case_0():
    var_0 = '/src/mymodule'
    var_1 = [var_0]
    var_2 = 'mymodule'

def test_case_0():
    var_0 = '/custom/src'
    var_1 = [var_0]
    var_2 = 'mymodule'



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_namespace_package_predicate_evaluates_to_true. Retrieved 5/17 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'myapp.submodule'
    var_3 = '.py'
    var_4 = 'myapp.submodule.nested'
    var_5 = [var_0]
    var_6 = ()



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_is_namespace_package_predicate_line_6_true. Retrieved 5/29 statements.


def test_case_0():
    var_0 = 'test_package'
    var_1 = 'py'
    var_2 = 'pyx'
    var_3 = {var_1, var_2}
    var_4 = frozenset(var_3)



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_src_path_is_module. Retrieved 2/12 statements.


def test_case_0():
    var_0 = 'mymodule'
    var_1 = True



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_src_path_is_module_with_matching_module_name_and_directory. Retrieved 1/8 statements.
# Partially parsed test_src_path_is_module_with_non_matching_module_name. Retrieved 1/8 statements.
# Partially parsed test_src_path_is_module_with_non_directory_path. Retrieved 1/8 statements.
# Partially parsed test_src_path_is_module_with_case_sensitive_check_failing. Retrieved 1/8 statements.
# Partially parsed test_src_path_is_module_all_conditions_false. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 'test_module'

def test_case_0():
    var_0 = 'other_module'

def test_case_0():
    var_0 = 'test_module'

def test_case_0():
    var_0 = 'test_module'

def test_case_0():
    var_0 = 'other_module'



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_src_path_finds_module_in_src_paths. Retrieved 15/27 statements.
# Partially parsed test_src_path_returns_none_when_module_not_found. Retrieved 13/21 statements.
# Partially parsed test_src_path_with_nested_module. Retrieved 17/33 statements.
# Partially parsed test_src_path_finds_py_file_module. Retrieved 15/25 statements.
# Partially parsed test_src_path_with_custom_src_paths_parameter. Retrieved 16/30 statements.


def test_case_0():
    var_0 = 'src'
    var_1 = 'mymodule'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'Config'
    var_5 = ()
    var_6 = 'src_paths'
    var_7 = 'namespace_packages'
    var_8 = 'auto_identify_namespace_packages'
    var_9 = 'supported_extensions'
    var_10 = frozenset()
    var_11 = False
    var_12 = 'py'
    var_13 = [var_12]
    var_14 = frozenset(var_13)

def test_case_0():
    var_0 = 'src'
    var_1 = 'Config'
    var_2 = ()
    var_3 = 'src_paths'
    var_4 = 'namespace_packages'
    var_5 = 'auto_identify_namespace_packages'
    var_6 = 'supported_extensions'
    var_7 = frozenset()
    var_8 = False
    var_9 = 'py'
    var_10 = [var_9]
    var_11 = frozenset(var_10)
    var_12 = 'nonexistent'

def test_case_0():
    var_0 = 'src'
    var_1 = 'mypackage'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'submodule'
    var_5 = 'Config'
    var_6 = ()
    var_7 = 'src_paths'
    var_8 = 'namespace_packages'
    var_9 = 'auto_identify_namespace_packages'
    var_10 = 'supported_extensions'
    var_11 = frozenset()
    var_12 = False
    var_13 = 'py'
    var_14 = [var_13]
    var_15 = frozenset(var_14)
    var_16 = 'mypackage.submodule'

def test_case_0():
    var_0 = 'src'
    var_1 = 'mymodule.py'
    var_2 = ''
    var_3 = 'Config'
    var_4 = ()
    var_5 = 'src_paths'
    var_6 = 'namespace_packages'
    var_7 = 'auto_identify_namespace_packages'
    var_8 = 'supported_extensions'
    var_9 = frozenset()
    var_10 = False
    var_11 = 'py'
    var_12 = [var_11]
    var_13 = frozenset(var_12)
    var_14 = 'mymodule'

def test_case_0():
    var_0 = 'custom_src'
    var_1 = 'mymodule'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'Config'
    var_5 = ()
    var_6 = 'src_paths'
    var_7 = 'namespace_packages'
    var_8 = 'auto_identify_namespace_packages'
    var_9 = 'supported_extensions'
    var_10 = 'other'
    var_11 = frozenset()
    var_12 = False
    var_13 = 'py'
    var_14 = [var_13]
    var_15 = frozenset(var_14)



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_is_namespace_package_not_a_package. Retrieved 4/10 statements.
# Partially parsed test_is_namespace_package_regular_package_with_init. Retrieved 6/13 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_namespace_single_quotes. Retrieved 6/13 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_namespace_double_quotes. Retrieved 6/13 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path_single_quotes. Retrieved 6/13 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path_double_quotes. Retrieved 6/13 statements.
# Partially parsed test_is_namespace_package_no_init_with_py_files. Retrieved 6/13 statements.
# Partially parsed test_is_namespace_package_no_init_with_setup_cfg. Retrieved 6/13 statements.
# Partially parsed test_is_namespace_package_no_init_with_pyproject_toml. Retrieved 6/13 statements.
# Partially parsed test_is_namespace_package_no_init_no_files. Retrieved 4/9 statements.
# Partially parsed test_is_namespace_package_empty_init_file. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'not_a_package'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'regular_package'
    var_1 = '__init__.py'
    var_2 = '# regular init file'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_pkg1'
    var_1 = '__init__.py'
    var_2 = b"__import__('pkg_resources').declare_namespace(__name__)"
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_pkg2'
    var_1 = '__init__.py'
    var_2 = b'__import__("pkg_resources").declare_namespace(__name__)'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_pkg3'
    var_1 = '__init__.py'
    var_2 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_pkg4'
    var_1 = '__init__.py'
    var_2 = b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_pkg5'
    var_1 = 'module.py'
    var_2 = '# some module'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_pkg6'
    var_1 = 'setup.cfg'
    var_2 = '[metadata]'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_pkg7'
    var_1 = 'pyproject.toml'
    var_2 = '[build-system]'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_pkg8'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'namespace_pkg9'
    var_1 = '__init__.py'
    var_2 = b''
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_src_path_predicate_line_26_evaluates_to_true. Retrieved 15/47 statements.


def test_case_0():
    var_0 = 'mymodule'
    var_1 = None
    var_2 = None
    var_3 = None
    var_4 = None
    var_5 = 'src_path'
    var_6 = '_is_module'
    var_7 = None
    var_8 = '_is_package'
    var_9 = '_src_path_is_module'
    var_10 = '_is_namespace_package'
    var_11 = '_is_module'
    var_12 = '_is_package'
    var_13 = '_src_path_is_module'
    var_14 = '_is_namespace_package'
    var_15 = 'Found in one of the configured src_paths'



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_src_path_predicate_line_26_evaluates_to_true. Retrieved 4/23 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_0]
    var_4 = 'mymodule'
    var_5 = ()



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 4/20 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 4/20 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 4/20 statements.


def test_case_0():
    var_0 = '__main__.exists_case_sensitive'
    var_1 = 'module'
    var_2 = '.py'
    var_3 = '__init__.py'

def test_case_0():
    var_0 = '__main__.exists_case_sensitive'
    var_1 = 'module'
    var_2 = '.py'
    var_3 = '__init__.py'

def test_case_0():
    var_0 = '__main__.exists_case_sensitive'
    var_1 = 'module'
    var_2 = '.py'
    var_3 = '__init__.py'



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_src_path_is_module_with_matching_directory. Retrieved 1/8 statements.
# Partially parsed test_src_path_is_module_with_non_matching_name. Retrieved 1/8 statements.
# Partially parsed test_src_path_is_module_with_non_directory. Retrieved 1/8 statements.
# Partially parsed test_src_path_is_module_with_case_sensitive_failure. Retrieved 1/8 statements.
# Partially parsed test_src_path_is_module_all_conditions_false. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 'mymodule'

def test_case_0():
    var_0 = 'othermodule'

def test_case_0():
    var_0 = 'mymodule'

def test_case_0():
    var_0 = 'mymodule'

def test_case_0():
    var_0 = 'module2'



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 8/31 statements.


def test_case_0():
    var_0 = 'builtins.__import__'
    var_1 = 'test_module'
    var_2 = 'test_module.py'
    var_3 = 'pathlib'
    var_4 = __import__(var_3)
    var_5 = 'exists_case_sensitive'
    var_6 = '.py'
    var_7 = '__init__.py'



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_forced_separate_matches_with_asterisk. Retrieved 2/6 statements.
# Partially parsed test_forced_separate_matches_without_asterisk. Retrieved 2/6 statements.
# Partially parsed test_forced_separate_matches_with_dot_prefix. Retrieved 2/6 statements.
# Partially parsed test_forced_separate_no_match. Retrieved 2/6 statements.
# Partially parsed test_forced_separate_empty_list. Retrieved 1/5 statements.
# Partially parsed test_forced_separate_pattern_with_asterisk. Retrieved 2/6 statements.
# Partially parsed test_forced_separate_exact_match. Retrieved 1/5 statements.
# Partially parsed test_forced_separate_multiple_patterns_first_match. Retrieved 3/7 statements.
# Partially parsed test_forced_separate_multiple_patterns_second_match. Retrieved 3/7 statements.
# Partially parsed test_forced_separate_case_sensitive. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'django.db'
    var_1 = 'django.db.models'

def test_case_0():
    var_0 = 'django'
    var_1 = 'django.core'

def test_case_0():
    var_0 = 'models'
    var_1 = '.models.base'

def test_case_0():
    var_0 = 'django.db'
    var_1 = 'flask.app'

def test_case_0():
    var_0 = 'any.module'

def test_case_0():
    var_0 = 'test.*.module'
    var_1 = 'test.sub.module.code'

def test_case_0():
    var_0 = 'mymodule'

def test_case_0():
    var_0 = 'django.db'
    var_1 = 'flask.app'
    var_2 = 'django.db.models'

def test_case_0():
    var_0 = 'django.db'
    var_1 = 'flask.app'
    var_2 = 'flask.app.routes'

def test_case_0():
    var_0 = 'Django'
    var_1 = 'django.core'



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_is_namespace_package_returns_true_for_namespace_package. Retrieved 8/37 statements.


def test_case_0():
    var_0 = 'test_namespace'
    var_1 = '__init__.py'
    var_2 = b"__import__('pkg_resources').declare_namespace(__name__)"
    var_3 = 0
    var_4 = 'test_module'
    var_5 = 'py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_namespace_package_predicate_evaluates_to_true. Retrieved 13/21 statements.


def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'src_paths'
    var_3 = 'namespace_packages'
    var_4 = 'auto_identify_namespace_packages'
    var_5 = 'supported_extensions'
    var_6 = 'mypackage'
    var_7 = [var_6]
    var_8 = False
    var_9 = '.py'
    var_10 = [var_9]
    var_11 = 'mypackage.submodule'
    var_12 = ()



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_src_path_predicate_line_26_evaluates_to_true. Retrieved 9/33 statements.


def test_case_0():
    var_0 = 'test_module.py'
    var_1 = '# test module'
    var_2 = '\nfrom pathlib import Path\nfrom typing import Iterable\n\nclass MockConfig:\n    def __init__(self):\n        self.src_paths = []\n        self.namespace_packages = []\n        self.auto_identify_namespace_packages = False\n        self.supported_extensions = [".py"]\n\nclass sections:\n    FIRSTPARTY = "FIRSTPARTY"\n\ndef _is_module(path):\n    return path.is_file() and path.suffix == ".py"\n\ndef _is_package(path):\n    return False\n\ndef _src_path_is_module(src_path, module_name):\n    return False\n\ndef _is_namespace_package(path, extensions):\n    return False\n\ndef _src_path(\n    name: str,\n    config: MockConfig,\n    src_paths: Iterable[Path] | None = None,\n    prefix: tuple[str, ...] = (),\n) -> tuple[str, str] | None:\n    if src_paths is None:\n        src_paths = config.src_paths\n\n    root_module_name, *nested_module = name.split(".", 1)\n    new_prefix = (*prefix, root_module_name)\n    namespace = ".".join(new_prefix)\n\n    for src_path in src_paths:\n        module_path = (src_path / root_module_name).resolve()\n        if not prefix and not module_path.is_dir() and src_path.name == root_module_name:\n            module_path = src_path.resolve()\n        if nested_module and (\n            namespace in config.namespace_packages\n            or (\n                config.auto_identify_namespace_packages\n                and _is_namespace_package(module_path, config.supported_extensions)\n            )\n        ):\n            return _src_path(nested_module[0], config, (module_path,), new_prefix)\n        if (\n            _is_module(module_path)\n            or _is_package(module_path)\n            or _src_path_is_module(src_path, root_module_name)\n        ):\n            return (sections.FIRSTPARTY, f"Found in one of the configured src_paths: {src_path}.")\n\n    return None\n'
    var_3 = '# test'
    var_4 = {}
    var_5 = exec(var_2, var_4)
    var_6 = '_src_path'
    var_7 = var_4[var_6]
    var_8 = 'test_module'
    var_9 = 'Found in one of the configured src_paths'



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_is_namespace_package_predicate_line_5. Retrieved 2/13 statements.


def test_case_0():
    var_0 = '__init__.py'
    var_1 = b"__import__('pkg_resources').declare_namespace(__name__)"



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_src_path_predicate_at_line_26_evaluates_to_true. Retrieved 5/29 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_0]
    var_4 = 'test_module'
    var_5 = 'test_module'
    var_6 = 'test_module'



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_is_namespace_package_predicate_line_6_true. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'test_package'
    var_1 = 'module.py'
    var_2 = '# test module'
    var_3 = '__init__.py'



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_namespace_package_predicate_evaluates_to_false. Retrieved 4/20 statements.


def test_case_0():
    var_0 = 'mymodule'
    var_1 = '__init__.py'
    var_2 = ''
    var_3 = 'mymodule'



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_is_namespace_package_predicate_at_line_13_evaluates_to_true. Retrieved 6/29 statements.


def test_case_0():
    var_0 = 'test_package'
    var_1 = 'module.py'
    var_2 = '# some code'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)



# Parsed testcases at query #79
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django.db'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(var_1, **var_2)
    var_4 = module_1._forced_separate(var_0, var_3)
    var_5 = bool(var_4 == ('django.db', 'Matched forced_separate (django.db) config value.'))
    assert var_5 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django.*'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(var_1, **var_2)
    var_4 = 'django.db.models'
    var_5 = module_1._forced_separate(var_4, var_3)
    var_6 = bool(var_5 == ('django.*', 'Matched forced_separate (django.*) config value.'))
    assert var_6 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(var_1, **var_2)
    var_4 = 'django.db'
    var_5 = module_1._forced_separate(var_4, var_3)
    var_6 = bool(var_5 == ('django', 'Matched forced_separate (django) config value.'))
    assert var_6 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django.db'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(var_1, **var_2)
    var_4 = '.django.db.models'
    var_5 = module_1._forced_separate(var_4, var_3)
    var_6 = bool(var_5 == ('django.db', 'Matched forced_separate (django.db) config value.'))
    assert var_6 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django.db'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(var_1, **var_2)
    var_4 = 'flask.app'
    var_5 = module_1._forced_separate(var_4, var_3)
    assert var_5 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Config(var_0, **var_1)
    var_3 = 'django.db'
    var_4 = module_1._forced_separate(var_3, var_2)
    assert var_4 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django.db'
    var_1 = 'flask'
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Config(var_2, **var_3)
    var_5 = 'django.db.models'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('django.db', 'Matched forced_separate (django.db) config value.'))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django.db'
    var_1 = 'flask'
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Config(var_2, **var_3)
    var_5 = 'flask.app'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('flask', 'Matched forced_separate (flask) config value.'))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'app?.models'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(var_1, **var_2)
    var_4 = 'app1.models.views'
    var_5 = module_1._forced_separate(var_4, var_3)
    var_6 = bool(var_5 == ('app?.models', 'Matched forced_separate (app?.models) config value.'))
    assert var_6 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(var_1, **var_2)
    var_4 = module_1._forced_separate(var_0, var_3)
    var_5 = bool(var_4 == ('django', 'Matched forced_separate (django) config value.'))
    assert var_5 is True



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_is_module_predicate_evaluates_to_true. Retrieved 5/20 statements.


def test_case_0():
    var_0 = '__main__.exists_case_sensitive'
    var_1 = 'test_module'
    var_2 = '.py'
    var_3 = []
    var_4 = '__init__.py'



