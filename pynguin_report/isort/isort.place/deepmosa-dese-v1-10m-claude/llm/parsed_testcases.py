####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_src_path_returns_none_when_module_not_found. Retrieved 12/17 statements.
# Partially parsed test_src_path_returns_firstparty_when_module_found. Retrieved 18/29 statements.
# Partially parsed test_src_path_with_nested_module. Retrieved 20/35 statements.
# Partially parsed test_src_path_with_module_file. Retrieved 18/27 statements.
# Partially parsed test_src_path_with_custom_src_paths. Retrieved 19/32 statements.
# Partially parsed test_src_path_with_prefix. Retrieved 20/33 statements.


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
    var_2 = 'sections'
    var_3 = ()
    var_4 = 'FIRSTPARTY'
    var_5 = {var_4: var_4}
    var_6 = type(var_2, var_3, var_5)
    var_7 = 'Config'
    var_8 = ()
    var_9 = 'src_paths'
    var_10 = 'namespace_packages'
    var_11 = 'auto_identify_namespace_packages'
    var_12 = 'supported_extensions'
    var_13 = []
    var_14 = False
    var_15 = 'py'
    var_16 = [var_15]
    var_17 = frozenset(var_16)

def test_case_0():
    var_0 = 'parent'
    var_1 = '__init__.py'
    var_2 = 'child'
    var_3 = 'sections'
    var_4 = ()
    var_5 = 'FIRSTPARTY'
    var_6 = {var_5: var_5}
    var_7 = type(var_3, var_4, var_6)
    var_8 = 'Config'
    var_9 = ()
    var_10 = 'src_paths'
    var_11 = 'namespace_packages'
    var_12 = 'auto_identify_namespace_packages'
    var_13 = 'supported_extensions'
    var_14 = []
    var_15 = False
    var_16 = 'py'
    var_17 = [var_16]
    var_18 = frozenset(var_17)
    var_19 = 'parent.child'

def test_case_0():
    var_0 = 'mymodule.py'
    var_1 = 'sections'
    var_2 = ()
    var_3 = 'FIRSTPARTY'
    var_4 = {var_3: var_3}
    var_5 = type(var_1, var_2, var_4)
    var_6 = 'Config'
    var_7 = ()
    var_8 = 'src_paths'
    var_9 = 'namespace_packages'
    var_10 = 'auto_identify_namespace_packages'
    var_11 = 'supported_extensions'
    var_12 = []
    var_13 = False
    var_14 = 'py'
    var_15 = [var_14]
    var_16 = frozenset(var_15)
    var_17 = 'mymodule'

def test_case_0():
    var_0 = 'custom_src'
    var_1 = 'mymodule'
    var_2 = '__init__.py'
    var_3 = 'sections'
    var_4 = ()
    var_5 = 'FIRSTPARTY'
    var_6 = {var_5: var_5}
    var_7 = type(var_3, var_4, var_6)
    var_8 = 'Config'
    var_9 = ()
    var_10 = 'src_paths'
    var_11 = 'namespace_packages'
    var_12 = 'auto_identify_namespace_packages'
    var_13 = 'supported_extensions'
    var_14 = []
    var_15 = False
    var_16 = 'py'
    var_17 = [var_16]
    var_18 = frozenset(var_17)

def test_case_0():
    var_0 = 'parent'
    var_1 = 'child'
    var_2 = '__init__.py'
    var_3 = 'sections'
    var_4 = ()
    var_5 = 'FIRSTPARTY'
    var_6 = {var_5: var_5}
    var_7 = type(var_3, var_4, var_6)
    var_8 = 'Config'
    var_9 = ()
    var_10 = 'src_paths'
    var_11 = 'namespace_packages'
    var_12 = 'auto_identify_namespace_packages'
    var_13 = 'supported_extensions'
    var_14 = []
    var_15 = False
    var_16 = 'py'
    var_17 = [var_16]
    var_18 = frozenset(var_17)
    var_19 = (var_0,)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 5/17 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 5/16 statements.
# Partially parsed test_is_module_with_init_file. Retrieved 5/18 statements.
# Partially parsed test_is_module_not_a_module. Retrieved 4/14 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = '.py'
    var_2 = 'importlib.machinery.EXTENSION_SUFFIXES'
    var_3 = []
    var_4 = 'exists_case_sensitive'

def test_case_0():
    var_0 = 'test_module'
    var_1 = '.so'
    var_2 = 'importlib.machinery.EXTENSION_SUFFIXES'
    var_3 = [var_1]
    var_4 = 'exists_case_sensitive'

def test_case_0():
    var_0 = 'test_module'
    var_1 = '__init__.py'
    var_2 = 'importlib.machinery.EXTENSION_SUFFIXES'
    var_3 = []
    var_4 = 'exists_case_sensitive'

def test_case_0():
    var_0 = 'not_a_module'
    var_1 = 'importlib.machinery.EXTENSION_SUFFIXES'
    var_2 = []
    var_3 = 'exists_case_sensitive'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_is_namespace_package_not_a_package. Retrieved 4/7 statements.
# Partially parsed test_is_namespace_package_regular_package_with_init. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_declare. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_py_files. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_no_py_files. Retrieved 4/8 statements.
# Partially parsed test_is_namespace_package_no_init_with_setup_cfg. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_pyproject_toml. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_double_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_double_quotes. Retrieved 6/12 statements.


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
    var_5 = "__path__ = __import__('pkgutil').extend_path(__path__, __name__)"

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'mypkg'
    var_4 = '__init__.py'
    var_5 = "__import__('pkg_resources').declare_namespace(__name__)"

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'mypkg'
    var_4 = 'module.py'
    var_5 = '# module'

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
    var_4 = 'setup.cfg'
    var_5 = '[metadata]'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'mypkg'
    var_4 = 'pyproject.toml'
    var_5 = '[build-system]'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'mypkg'
    var_4 = '__init__.py'
    var_5 = '__path__ = __import__("pkgutil").extend_path(__path__, __name__)'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'mypkg'
    var_4 = '__init__.py'
    var_5 = '__import__("pkg_resources").declare_namespace(__name__)'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_known_pattern_matches_exact_module. Retrieved 15/24 statements.
# Partially parsed test_known_pattern_matches_partial_module. Retrieved 14/21 statements.
# Partially parsed test_known_pattern_no_match. Retrieved 14/21 statements.
# Partially parsed test_known_pattern_section_not_in_config. Retrieved 16/23 statements.
# Partially parsed test_known_pattern_multiple_patterns_first_match_wins. Retrieved 18/27 statements.
# Partially parsed test_known_pattern_regex_match. Retrieved 14/21 statements.


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
    var_13 = 'third_party'
    var_14 = [var_13]

def test_case_0():
    var_0 = 'Config'
    var_1 = 'known_patterns'
    var_2 = 'sections'
    var_3 = [var_1, var_2]
    var_4 = 'Pattern'
    var_5 = ()
    var_6 = 'match'
    var_7 = 'django'
    var_8 = lambda self, x: x == var_7
    var_9 = {var_6: var_8}
    var_10 = type(var_4, var_5, var_9)
    var_11 = 'third_party'
    var_12 = [var_11]
    var_13 = 'django.conf.settings'

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
    var_11 = 'third_party'
    var_12 = [var_11]
    var_13 = 'mymodule'

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
    var_11 = 'nonexistent'
    var_12 = 'third_party'
    var_13 = 'stdlib'
    var_14 = [var_12, var_13]
    var_15 = 'mymodule'

def test_case_0():
    var_0 = 'Config'
    var_1 = 'known_patterns'
    var_2 = 'sections'
    var_3 = [var_1, var_2]
    var_4 = 'Pattern'
    var_5 = ()
    var_6 = 'match'
    var_7 = 'django'
    var_8 = lambda self, x: x == var_7
    var_9 = {var_6: var_8}
    var_10 = type(var_4, var_5, var_9)
    var_11 = ()
    var_12 = lambda self, x: x == var_7
    var_13 = {var_6: var_12}
    var_14 = type(var_4, var_11, var_13)
    var_15 = 'third_party'
    var_16 = 'other'
    var_17 = [var_15, var_16]

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
    var_11 = 'third_party'
    var_12 = [var_11]
    var_13 = 'django.utils.text'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_src_path_predicate_line_7_evaluates_to_false. Retrieved 3/14 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = '/another/path'
    var_2 = 'module_name'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_src_path_namespace_package_in_config. Retrieved 9/27 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = 'myapp.submodule'
    var_2 = '.py'
    var_3 = 'myapp.submodule.utils'
    var_4 = ()
    var_5 = '.'
    var_6 = 1
    var_7 = *var_4
    var_8 = 'myapp'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_namespace_in_config_namespace_packages. Retrieved 13/31 statements.


def test_case_0():
    var_0 = '/test/src'
    var_1 = 'my.namespace'
    var_2 = '.py'
    var_3 = 'src'
    var_4 = var_0 / var_3
    var_5 = 'my'
    var_6 = var_4 / var_5
    var_7 = 'my.namespace'
    var_8 = ()
    var_9 = 'my'
    var_10 = 'namespace'
    var_11 = [var_10]
    var_12 = 'my'



# Parsed testcases at query #8
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django.db'
    var_1 = [var_0]
    var_2 = module_0.Config(var_1)
    var_3 = 'django.db.models'
    var_4 = module_1._forced_separate(var_3, var_2)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'mypackage'
    var_1 = [var_0]
    var_2 = module_0.Config(var_1)
    var_3 = 'mypackage.submodule'
    var_4 = module_1._forced_separate(var_3, var_2)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = [var_0]
    var_2 = module_0.Config(var_1)
    var_3 = '.module.submodule'
    var_4 = module_1._forced_separate(var_3, var_2)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django.db'
    var_1 = [var_0]
    var_2 = module_0.Config(var_1)
    var_3 = 'requests.api'
    var_4 = module_1._forced_separate(var_3, var_2)
    assert var_4 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = module_0.Config(var_0)
    var_2 = 'any.module'
    var_3 = module_1._forced_separate(var_2, var_1)
    assert var_3 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'package*'
    var_1 = [var_0]
    var_2 = module_0.Config(var_1)
    var_3 = 'package.module'
    var_4 = module_1._forced_separate(var_3, var_2)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django.db'
    var_1 = 'requests'
    var_2 = 'flask'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.Config(var_3)
    var_5 = 'requests.api'
    var_6 = module_1._forced_separate(var_5, var_4)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test.module'
    var_1 = [var_0]
    var_2 = module_0.Config(var_1)
    var_3 = 'test.module.sub'
    var_4 = module_1._forced_separate(var_3, var_2)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 5/20 statements.


def test_case_0():
    var_0 = 'test_module.py'
    var_1 = '# test module'
    var_2 = 'builtins.__import__'
    var_3 = lambda name, *args: __import__(name, *args)
    var_4 = ''



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_known_pattern_matches_exact_module. Retrieved 11/20 statements.
# Partially parsed test_known_pattern_matches_submodule. Retrieved 8/16 statements.
# Partially parsed test_known_pattern_no_match. Retrieved 8/16 statements.
# Partially parsed test_known_pattern_placement_not_in_sections. Retrieved 9/17 statements.
# Partially parsed test_known_pattern_multiple_patterns_first_match. Retrieved 9/19 statements.
# Partially parsed test_known_pattern_checks_longest_module_first. Retrieved 8/16 statements.
# Partially parsed test_known_pattern_empty_patterns. Retrieved 10/12 statements.


def test_case_0():
    var_0 = 'Config'
    var_1 = 'known_patterns'
    var_2 = 'sections'
    var_3 = [var_1, var_2]
    var_4 = 'Pattern'
    var_5 = 'pattern'
    var_6 = [var_5]
    var_7 = '^django$'
    var_8 = 'third_party'
    var_9 = [var_8]
    var_10 = 'django'

def test_case_0():
    var_0 = 'Config'
    var_1 = 'known_patterns'
    var_2 = 'sections'
    var_3 = [var_1, var_2]
    var_4 = '^django'
    var_5 = 'third_party'
    var_6 = [var_5]
    var_7 = 'django.conf.settings'

def test_case_0():
    var_0 = 'Config'
    var_1 = 'known_patterns'
    var_2 = 'sections'
    var_3 = [var_1, var_2]
    var_4 = '^django$'
    var_5 = 'third_party'
    var_6 = [var_5]
    var_7 = 'myapp'

def test_case_0():
    var_0 = 'Config'
    var_1 = 'known_patterns'
    var_2 = 'sections'
    var_3 = [var_1, var_2]
    var_4 = '^django$'
    var_5 = 'third_party'
    var_6 = 'stdlib'
    var_7 = [var_6]
    var_8 = 'django'

def test_case_0():
    var_0 = 'Config'
    var_1 = 'known_patterns'
    var_2 = 'sections'
    var_3 = [var_1, var_2]
    var_4 = '^django'
    var_5 = '^flask'
    var_6 = 'third_party'
    var_7 = [var_6]
    var_8 = 'django.conf'

def test_case_0():
    var_0 = 'Config'
    var_1 = 'known_patterns'
    var_2 = 'sections'
    var_3 = [var_1, var_2]
    var_4 = '^myapp\\.utils$'
    var_5 = 'local'
    var_6 = [var_5]
    var_7 = 'myapp.utils.helpers'

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
    var_7 = module_0.Config()
    var_8 = 'django'
    var_9 = module_1._known_pattern(var_8, var_7)
    assert var_9 is None



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_predicate_at_line_16_evaluates_to_true. Retrieved 4/23 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'mymodule'
    var_2 = ()
    var_3 = 'mymodule'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_line_16_predicate_evaluates_to_true. Retrieved 7/18 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = False
    var_3 = []
    var_4 = 'mymodule'
    var_5 = ''
    var_6 = ()



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_is_namespace_package_returns_true_for_valid_namespace_package. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = b"__import__('pkg_resources').declare_namespace(__name__)"
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_src_paths_is_not_none. Retrieved 6/12 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = []
    var_3 = module_0.Config()
    var_4 = '/custom/path'
    var_5 = 'test_module'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 9/32 statements.


def test_case_0():
    var_0 = 'test_module.py'
    var_1 = '# test module'
    var_2 = 'importlib'
    var_3 = 'importlib.machinery.EXTENSION_SUFFIXES'
    var_4 = []
    var_5 = '.py'
    var_6 = ''
    var_7 = []
    var_8 = '__init__.py'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_src_path_returns_none_when_module_not_found. Retrieved 2/11 statements.
# Partially parsed test_src_path_returns_firstparty_when_module_found. Retrieved 2/12 statements.
# Partially parsed test_src_path_with_nested_module_not_namespace. Retrieved 2/12 statements.
# Partially parsed test_src_path_with_namespace_package. Retrieved 3/13 statements.
# Partially parsed test_src_path_auto_identify_namespace_packages. Retrieved 4/14 statements.
# Partially parsed test_src_path_src_path_is_module. Retrieved 2/12 statements.
# Partially parsed test_src_path_with_custom_src_paths. Retrieved 2/12 statements.
# Partially parsed test_src_path_with_prefix. Retrieved 4/14 statements.


def test_case_0():
    var_0 = '/nonexistent/path'
    var_1 = 'nonexistent_module'

def test_case_0():
    var_0 = '/src'
    var_1 = 'mymodule'

def test_case_0():
    var_0 = '/src'
    var_1 = 'parent.child'

def test_case_0():
    var_0 = '/src'
    var_1 = 'parent'
    var_2 = 'parent.child'

def test_case_0():
    var_0 = '/src'
    var_1 = 'py'
    var_2 = {var_1}
    var_3 = 'parent.child'

def test_case_0():
    var_0 = '/src'
    var_1 = 'mymodule'

def test_case_0():
    var_0 = '/custom/src'
    var_1 = 'mymodule'

def test_case_0():
    var_0 = '/src'
    var_1 = 'child'
    var_2 = 'parent'
    var_3 = (var_2,)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_namespace_package_predicate_evaluates_to_true. Retrieved 19/30 statements.


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
    var_12 = ()
    var_13 = 'myapp'
    var_14 = (var_13,)
    var_15 = '.'
    var_16 = 'submodule'
    var_17 = 'nested'
    var_18 = [var_16, var_17]



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 8/25 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 9/25 statements.
# Partially parsed test_is_module_with_init_file. Retrieved 8/25 statements.
# Partially parsed test_is_module_not_a_module. Retrieved 6/20 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module.py'
    var_2 = '# test'
    var_3 = 'importlib.machinery.EXTENSION_SUFFIXES'
    var_4 = []
    var_5 = '__main__'
    var_6 = '__main__'
    var_7 = 'exists_case_sensitive'

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module.so'
    var_2 = '# test'
    var_3 = 'importlib.machinery.EXTENSION_SUFFIXES'
    var_4 = '.so'
    var_5 = [var_4]
    var_6 = '__main__'
    var_7 = '__main__'
    var_8 = 'exists_case_sensitive'

def test_case_0():
    var_0 = 'test_package'
    var_1 = '__init__.py'
    var_2 = '# test'
    var_3 = 'importlib.machinery.EXTENSION_SUFFIXES'
    var_4 = []
    var_5 = '__main__'
    var_6 = '__main__'
    var_7 = 'exists_case_sensitive'

def test_case_0():
    var_0 = 'not_a_module'
    var_1 = 'importlib.machinery.EXTENSION_SUFFIXES'
    var_2 = []
    var_3 = '__main__'
    var_4 = '__main__'
    var_5 = 'exists_case_sensitive'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_src_path_predicate_at_line_26_evaluates_to_true. Retrieved 3/20 statements.


def test_case_0():
    var_0 = '/test/src'
    var_1 = '.py'
    var_2 = 'mymodule'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_src_path_predicate_line_26_evaluates_to_true. Retrieved 4/21 statements.


def test_case_0():
    var_0 = '.py'
    var_1 = 'test_module'
    var_2 = '__init__.py'
    var_3 = 'test_module'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_src_path_is_module_with_matching_dir. Retrieved 2/9 statements.
# Partially parsed test_src_path_is_module_with_non_matching_name. Retrieved 2/9 statements.
# Partially parsed test_src_path_is_module_with_non_dir. Retrieved 2/9 statements.
# Partially parsed test_src_path_is_module_with_case_sensitive_not_exists. Retrieved 2/9 statements.


def test_case_0():
    var_0 = '/some/path/mymodule'
    var_1 = 'mymodule'

def test_case_0():
    var_0 = '/some/path/mymodule'
    var_1 = 'othermodule'

def test_case_0():
    var_0 = '/some/path/mymodule'
    var_1 = 'mymodule'

def test_case_0():
    var_0 = '/some/path/mymodule'
    var_1 = 'mymodule'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_namespace_package_predicate_evaluates_to_true. Retrieved 7/20 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'myapp.submodule.nested'
    var_2 = ()
    var_3 = '.'
    var_4 = 1
    var_5 = *var_2
    var_6 = var_0.namespace_packages



# Parsed testcases at query #23
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
    var_3 = 'mypkg'
    var_4 = '__init__.py'
    var_5 = '# regular package'

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
    var_5 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"

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
    var_5 = '[build-system]'

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
    var_4 = 'readme.txt'
    var_5 = 'readme'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_src_path_returns_none_when_module_not_found. Retrieved 12/18 statements.
# Partially parsed test_src_path_returns_firstparty_when_module_found. Retrieved 13/23 statements.
# Partially parsed test_src_path_with_nested_module. Retrieved 15/29 statements.
# Partially parsed test_src_path_with_custom_src_paths. Retrieved 14/27 statements.
# Partially parsed test_src_path_with_py_file. Retrieved 13/21 statements.
# Partially parsed test_src_path_with_prefix. Retrieved 15/30 statements.
# Partially parsed test_src_path_with_namespace_package. Retrieved 15/25 statements.
# Partially parsed test_src_path_returns_none_for_empty_name. Retrieved 12/17 statements.


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
    var_0 = 'namespace_pkg'
    var_1 = '__init__.py'
    var_2 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
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
    var_11 = ''



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_namespace_package_predicate_evaluates_to_false. Retrieved 15/31 statements.


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
    var_10 = ()
    var_11 = 'package.module'
    var_12 = '.'
    var_13 = 1
    var_14 = *var_10



# Parsed testcases at query #26
#--------------------------

# Failed to parse test_is_namespace_package_predicate_at_line_2.




# Parsed testcases at query #27
#--------------------------

# Partially parsed test_src_path_predicate_at_line_26_evaluates_to_true. Retrieved 9/28 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = '.py'
    var_2 = 'src'
    var_3 = var_0 / var_2
    var_4 = 'mymodule.py'
    var_5 = var_3 / var_4
    var_6 = '# test module'
    var_7 = 'mymodule'
    var_8 = [var_3]



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_src_path_namespace_package_in_config. Retrieved 10/30 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = 'my.nested'
    var_2 = '.py'
    var_3 = 'my.nested.module'
    var_4 = 'my'
    var_5 = 'nested.module'
    var_6 = [var_5]
    var_7 = bool(var_6)
    var_8 = var_4 in var_1
    var_9 = var_7 and var_8
    assert var_9 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_src_path_predicate_line_26_evaluates_to_true. Retrieved 3/19 statements.


def test_case_0():
    var_0 = '.py'
    var_1 = '/src'
    var_2 = 'mymodule'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 8/26 statements.


def test_case_0():
    var_0 = 'importlib.machinery.EXTENSION_SUFFIXES'
    var_1 = []
    var_2 = 'test_module'
    var_3 = 'pathlib'
    var_4 = __import__(var_3)
    var_5 = 'Path'
    var_6 = '.py'
    var_7 = '__init__.py'



# Parsed testcases at query #31
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = [var_0]
    var_2 = module_0.Config(var_1)
    var_3 = 'test_file'
    var_4 = module_1._forced_separate(var_3, var_2)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_src_path_returns_none_when_module_not_found. Retrieved 12/17 statements.
# Partially parsed test_src_path_returns_firstparty_when_module_is_file. Retrieved 14/21 statements.
# Partially parsed test_src_path_returns_firstparty_when_module_is_package. Retrieved 14/23 statements.
# Partially parsed test_src_path_with_nested_module. Retrieved 17/28 statements.
# Partially parsed test_src_path_with_multiple_src_paths. Retrieved 16/27 statements.
# Partially parsed test_src_path_uses_config_src_paths_when_none_provided. Retrieved 15/22 statements.
# Partially parsed test_src_path_with_custom_src_paths_parameter. Retrieved 15/25 statements.


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
    var_0 = 'mymodule.py'
    var_1 = '# test module'
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
    var_13 = 'mymodule'

def test_case_0():
    var_0 = 'mypackage'
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
    var_0 = 'mypackage'
    var_1 = '__init__.py'
    var_2 = ''
    var_3 = 'submodule.py'
    var_4 = '# submodule'
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
    var_0 = 'src1'
    var_1 = 'src2'
    var_2 = 'mymodule.py'
    var_3 = '# test module'
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
    var_15 = 'mymodule'

def test_case_0():
    var_0 = 'mymodule.py'
    var_1 = '# test module'
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
    var_13 = 'mymodule'
    var_14 = None

def test_case_0():
    var_0 = 'custom'
    var_1 = 'mymodule.py'
    var_2 = '# test module'
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



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_known_pattern_predicate_false. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'section1'
    var_1 = 'section2'
    var_2 = 'pattern1'
    var_3 = 'section3'
    var_4 = (var_2, var_3)
    var_5 = 'test.module'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_src_path_finds_module_in_src_paths. Retrieved 11/34 statements.
# Partially parsed test_src_path_returns_none_for_missing_module. Retrieved 3/19 statements.
# Partially parsed test_src_path_with_nested_module. Retrieved 9/33 statements.


def test_case_0():
    var_0 = 'src'
    var_1 = 'mymodule.py'
    var_2 = '# test module'
    var_3 = 'pathlib.Path.exists'
    var_4 = True
    var_5 = lambda self: var_4
    var_6 = 'importlib.machinery.EXTENSION_SUFFIXES'
    var_7 = []
    var_8 = 'isort.parse.sections'
    var_9 = 'isort.parse.exists_case_sensitive'
    var_10 = 'mymodule'

def test_case_0():
    var_0 = 'src'
    var_1 = 'isort.parse.exists_case_sensitive'
    var_2 = 'nonexistent'

def test_case_0():
    var_0 = 'src'
    var_1 = 'mypackage'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'nested.py'
    var_5 = '# nested module'
    var_6 = 'isort.parse.exists_case_sensitive'
    var_7 = 'isort.parse.sections'
    var_8 = 'mypackage.nested'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_src_paths_is_not_none. Retrieved 6/15 statements.


def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = []
    var_3 = '/custom/src'
    var_4 = 'test_module'
    var_5 = ()



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 4/22 statements.


def test_case_0():
    var_0 = 'test_module.py'
    var_1 = '# test module'
    var_2 = 0
    var_3 = 'test_module'



# Parsed testcases at query #37
#--------------------------




import fnmatch as module_0

def test_case_0():
    var_0 = 'mymodule'
    var_1 = 'mymodule'
    var_2 = var_1
    var_3 = module_0.fnmatch(var_0, var_2)
    var_4 = '.'
    var_5 = var_4 + var_2
    var_6 = module_0.fnmatch(var_0, var_5)
    var_7 = 'mymodule.submodule'
    var_8 = 'mymodule'
    var_9 = f'{var_8}*'
    var_10 = module_0.fnmatch(var_7, var_9)
    var_11 = var_4 + var_9
    var_12 = module_0.fnmatch(var_7, var_11)
    var_13 = '.mymodule'
    var_14 = 'mymodule'
    var_15 = var_14
    var_16 = module_0.fnmatch(var_13, var_15)
    var_17 = var_4 + var_15
    var_18 = module_0.fnmatch(var_13, var_17)
    var_19 = 'test_utils'
    var_20 = 'test_*'
    var_21 = var_20
    var_22 = module_0.fnmatch(var_19, var_21)
    var_23 = var_4 + var_21
    var_24 = module_0.fnmatch(var_19, var_23)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_is_namespace_package_predicate_line_2. Retrieved 6/19 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = '\ndef _is_package(path):\n    return True\n\ndef _is_namespace_package(path, src_extensions):\n    if not _is_package(path):\n        return False\n    \n    init_file = path / "__init__.py"\n    if not init_file.exists():\n        filenames = [\n            filepath\n            for filepath in path.iterdir()\n            if filepath.suffix.lstrip(".") in src_extensions\n            or filepath.name.lower() in ("setup.cfg", "pyproject.toml")\n        ]\n        if filenames:\n            return False\n    else:\n        with init_file.open("rb") as open_init_file:\n            file_start = open_init_file.read(4096)\n            if (\n                b"__import__(\'pkg_resources\').declare_namespace(__name__)" not in file_start\n                and b\'__import__("pkg_resources").declare_namespace(__name__)\' not in file_start\n                and b"__path__ = __import__(\'pkgutil\').extend_path(__path__, __name__)"\n                not in file_start\n                and b\'__path__ = __import__("pkgutil").extend_path(__path__, __name__)\'\n                not in file_start\n            ):\n                return False\n    return True\n'
    var_2 = '_is_namespace_package'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_src_path_is_module. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'test_module'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 3/12 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 3/17 statements.
# Partially parsed test_is_module_with_package. Retrieved 3/12 statements.
# Partially parsed test_is_module_not_found. Retrieved 1/7 statements.
# Partially parsed test_is_module_with_directory_without_init. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 'test_module.py'
    var_1 = '# test module'
    var_2 = 'test_module'

def test_case_0():
    var_0 = 0
    var_1 = '.so'
    var_2 = 'test_module'

def test_case_0():
    var_0 = 'test_package'
    var_1 = '__init__.py'
    var_2 = '# package init'

def test_case_0():
    var_0 = 'non_existent_module'

def test_case_0():
    var_0 = 'plain_directory'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 9/28 statements.


def test_case_0():
    var_0 = 'pathlib.Path.exists'
    var_1 = False
    var_2 = lambda self: var_1
    var_3 = 'test_module'
    var_4 = '.py'
    var_5 = [var_1]
    var_6 = None
    var_7 = 'importlib.machinery.EXTENSION_SUFFIXES'
    var_8 = []



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_is_namespace_package_predicate_at_line_2_true. Retrieved 5/27 statements.


def test_case_0():
    var_0 = '__init__.py'
    var_1 = b"__import__('pkg_resources').declare_namespace(__name__)"
    var_2 = 'py'
    var_3 = {var_2}
    var_4 = frozenset(var_3)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_known_pattern_matches_exact_module. Retrieved 1/8 statements.
# Partially parsed test_known_pattern_matches_submodule. Retrieved 1/8 statements.
# Partially parsed test_known_pattern_no_match. Retrieved 1/8 statements.
# Partially parsed test_known_pattern_section_not_in_config. Retrieved 1/8 statements.
# Partially parsed test_known_pattern_multiple_patterns_first_match. Retrieved 1/8 statements.
# Partially parsed test_known_pattern_partial_module_match. Retrieved 1/8 statements.
# Partially parsed test_known_pattern_empty_patterns. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'django'

def test_case_0():
    var_0 = 'django.conf.settings'

def test_case_0():
    var_0 = 'mymodule'

def test_case_0():
    var_0 = 'django'

def test_case_0():
    var_0 = 'django.conf'

def test_case_0():
    var_0 = 'requests.api.get'

def test_case_0():
    var_0 = 'django'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_src_path_is_module. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 'test_module'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_src_path_is_module. Retrieved 10/37 statements.


def test_case_0():
    var_0 = '/some/path/mymodule'
    var_1 = 'mymodule'
    var_2 = '/some/path/mymodule'
    var_3 = 'different'
    var_4 = '/some/path/mymodule'
    var_5 = 'mymodule'
    var_6 = '/some/path/mymodule'
    var_7 = 'mymodule'
    var_8 = '/some/path/mymodule'
    var_9 = 'other'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_predicate_at_line_16_evaluates_to_true. Retrieved 3/20 statements.


def test_case_0():
    var_0 = 'mymodule'
    var_1 = 'mymodule'
    var_2 = ()



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_src_path_predicate_at_line_26_evaluates_to_true. Retrieved 3/18 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = '.py'
    var_2 = 'mymodule'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_src_path_returns_none_when_module_not_found. Retrieved 12/17 statements.
# Partially parsed test_src_path_returns_firstparty_when_module_is_file. Retrieved 14/21 statements.
# Partially parsed test_src_path_returns_firstparty_when_module_is_package. Retrieved 14/23 statements.
# Partially parsed test_src_path_with_nested_module_name. Retrieved 17/28 statements.
# Partially parsed test_src_path_with_custom_src_paths_parameter. Retrieved 15/25 statements.
# Partially parsed test_src_path_with_prefix_parameter. Retrieved 16/30 statements.
# Partially parsed test_src_path_returns_firstparty_for_src_path_module. Retrieved 12/19 statements.


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
    var_0 = 'mymodule.py'
    var_1 = '# test module'
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
    var_13 = 'mymodule'

def test_case_0():
    var_0 = 'mypackage'
    var_1 = '__init__.py'
    var_2 = '# package'
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
    var_0 = 'mypackage'
    var_1 = '__init__.py'
    var_2 = ''
    var_3 = 'submodule.py'
    var_4 = '# submodule'
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
    var_2 = '# module'
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
    var_0 = 'mypackage'
    var_1 = '__init__.py'
    var_2 = ''
    var_3 = 'subpackage'
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
    var_15 = (var_0,)

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



# Parsed testcases at query #49
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django'
    var_1 = [var_0]
    var_2 = module_0.Config(var_1)
    var_3 = module_1._forced_separate(var_0, var_2)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django.*'
    var_1 = [var_0]
    var_2 = module_0.Config(var_1)
    var_3 = 'django.core'
    var_4 = module_1._forced_separate(var_3, var_2)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = [var_0]
    var_2 = module_0.Config(var_1)
    var_3 = '.test'
    var_4 = module_1._forced_separate(var_3, var_2)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django'
    var_1 = [var_0]
    var_2 = module_0.Config(var_1)
    var_3 = 'flask'
    var_4 = module_1._forced_separate(var_3, var_2)
    assert var_4 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = module_0.Config(var_0)
    var_2 = 'django'
    var_3 = module_1._forced_separate(var_2, var_1)
    assert var_3 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django'
    var_1 = 'flask'
    var_2 = [var_0, var_1]
    var_3 = module_0.Config(var_2)
    var_4 = 'django.core'
    var_5 = module_1._forced_separate(var_4, var_3)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django'
    var_1 = 'flask'
    var_2 = [var_0, var_1]
    var_3 = module_0.Config(var_2)
    var_4 = 'flask.app'
    var_5 = module_1._forced_separate(var_4, var_3)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test*'
    var_1 = [var_0]
    var_2 = module_0.Config(var_1)
    var_3 = 'testing'
    var_4 = module_1._forced_separate(var_3, var_2)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'lib'
    var_1 = [var_0]
    var_2 = module_0.Config(var_1)
    var_3 = '.library'
    var_4 = module_1._forced_separate(var_3, var_2)
    assert var_4 is None



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_known_pattern_predicate_false. Retrieved 9/30 statements.


def test_case_0():
    var_0 = 'section1'
    var_1 = 'section2'
    var_2 = True
    var_3 = 'unknown_section'
    var_4 = False
    var_5 = 'module.submodule'
    var_6 = '.'
    var_7 = -1
    var_8 = 'unknown_section'



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_src_path_returns_none_when_module_not_found. Retrieved 2/10 statements.
# Partially parsed test_src_path_returns_firstparty_when_module_found. Retrieved 3/12 statements.
# Partially parsed test_src_path_with_nested_module_non_namespace. Retrieved 2/10 statements.
# Partially parsed test_src_path_with_custom_src_paths. Retrieved 3/13 statements.
# Partially parsed test_src_path_with_prefix. Retrieved 4/12 statements.
# Partially parsed test_src_path_src_path_is_module_match. Retrieved 2/10 statements.


def test_case_0():
    var_0 = '/fake/src'
    var_1 = 'nonexistent_module'

def test_case_0():
    var_0 = '/src'
    var_1 = '/src/mymodule.py'
    var_2 = 'mymodule'

def test_case_0():
    var_0 = '/src'
    var_1 = 'package.submodule'

def test_case_0():
    var_0 = '/custom/src'
    var_1 = '/default/src'
    var_2 = 'mymodule'

def test_case_0():
    var_0 = '/src'
    var_1 = 'module'
    var_2 = 'parent'
    var_3 = (var_2,)

def test_case_0():
    var_0 = '/src/mymodule'
    var_1 = 'mymodule'



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_namespace_package_predicate_evaluates_to_true. Retrieved 2/13 statements.


def test_case_0():
    var_0 = 'myapp'
    var_1 = 'myapp.sub.module'



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_predicate_line_16_evaluates_to_true. Retrieved 5/25 statements.


def test_case_0():
    var_0 = 'mymodule'
    var_1 = 'mymodule'
    var_2 = ()
    var_3 = '.'
    var_4 = 1



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_is_namespace_package_returns_false_when_not_package. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'non_package'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_src_path_predicate_line_26_evaluates_to_true. Retrieved 3/16 statements.


def test_case_0():
    var_0 = '/test/src'
    var_1 = '.py'
    var_2 = 'test_module'



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_src_path_predicate_line_26_true. Retrieved 3/19 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = '.py'
    var_2 = 'module_name'



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_src_path_returns_none_when_module_not_found. Retrieved 12/17 statements.
# Partially parsed test_src_path_finds_module_in_src_paths. Retrieved 18/29 statements.
# Partially parsed test_src_path_finds_py_module. Retrieved 18/27 statements.
# Partially parsed test_src_path_with_nested_module_name. Retrieved 19/30 statements.
# Partially parsed test_src_path_uses_provided_src_paths. Retrieved 19/33 statements.


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
    var_13 = 'sections'
    var_14 = ()
    var_15 = 'FIRSTPARTY'
    var_16 = {var_15: var_15}
    var_17 = type(var_13, var_14, var_16)

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
    var_12 = 'sections'
    var_13 = ()
    var_14 = 'FIRSTPARTY'
    var_15 = {var_14: var_14}
    var_16 = type(var_12, var_13, var_15)
    var_17 = 'mymodule'

def test_case_0():
    var_0 = 'mypackage'
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
    var_13 = 'sections'
    var_14 = ()
    var_15 = 'FIRSTPARTY'
    var_16 = {var_15: var_15}
    var_17 = type(var_13, var_14, var_16)
    var_18 = 'mypackage.submodule'

def test_case_0():
    var_0 = 'custom_src'
    var_1 = 'testmodule'
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
    var_14 = 'sections'
    var_15 = ()
    var_16 = 'FIRSTPARTY'
    var_17 = {var_16: var_16}
    var_18 = type(var_14, var_15, var_17)



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_is_namespace_package_predicate_evaluates_to_false. Retrieved 6/21 statements.


def test_case_0():
    var_0 = '__init__.py'
    var_1 = '# regular package'
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)
    var_5 = b'# This is a regular package init file'



# Parsed testcases at query #59
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
# Partially parsed test_is_namespace_package_no_init_no_py_files. Retrieved 4/8 statements.
# Partially parsed test_is_namespace_package_no_init_with_txt_file. Retrieved 6/12 statements.


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
    var_5 = '# some python code'

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

def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'mypkg'
    var_4 = 'readme.txt'
    var_5 = 'readme'



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_is_namespace_package_returns_true_for_namespace_package. Retrieved 7/31 statements.


def test_case_0():
    var_0 = 'namespace_pkg'
    var_1 = '__init__.py'
    var_2 = b"__import__('pkg_resources').declare_namespace(__name__)"
    var_3 = 'py'
    var_4 = 'pyx'
    var_5 = {var_3, var_4}
    var_6 = frozenset(var_5)



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_is_namespace_package_predicate_evaluates_to_false. Retrieved 6/29 statements.


def test_case_0():
    var_0 = '__init__.py'
    var_1 = '# Regular package'
    var_2 = 'py'
    var_3 = 'pyx'
    var_4 = [var_2, var_3]
    var_5 = frozenset(var_4)



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_is_namespace_package_predicate_evaluates_to_false. Retrieved 7/20 statements.


def test_case_0():
    var_0 = '__init__.py'
    var_1 = '# regular package'
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)
    var_5 = 'not_a_package'
    var_6 = False



# Parsed testcases at query #63
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
# Partially parsed test_is_namespace_package_no_init_empty_directory. Retrieved 4/7 statements.
# Partially parsed test_is_namespace_package_no_init_with_non_source_files. Retrieved 6/11 statements.
# Partially parsed test_is_namespace_package_multiple_extensions. Retrieved 7/12 statements.


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
    var_0 = 'namespace_pkg1'
    var_1 = '__init__.py'
    var_2 = b"__import__('pkg_resources').declare_namespace(__name__)"
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_pkg2'
    var_1 = '__init__.py'
    var_2 = b'__import__("pkg_resources").declare_namespace(__name__)'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_pkg3'
    var_1 = '__init__.py'
    var_2 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_pkg4'
    var_1 = '__init__.py'
    var_2 = b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_pkg5'
    var_1 = 'module.py'
    var_2 = '# module'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_pkg6'
    var_1 = 'setup.cfg'
    var_2 = ''
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_pkg7'
    var_1 = 'pyproject.toml'
    var_2 = ''
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_pkg8'
    var_1 = 'py'
    var_2 = {var_1}
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'namespace_pkg9'
    var_1 = 'data.txt'
    var_2 = 'data'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_pkg10'
    var_1 = 'module.pyx'
    var_2 = '# cython module'
    var_3 = 'py'
    var_4 = 'pyx'
    var_5 = {var_3, var_4}
    var_6 = frozenset(var_5)



# Parsed testcases at query #64
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
# Partially parsed test_is_namespace_package_no_init_empty_dir. Retrieved 4/7 statements.
# Partially parsed test_is_namespace_package_no_init_with_non_matching_extension. Retrieved 6/11 statements.
# Partially parsed test_is_namespace_package_large_init_file_with_namespace_marker. Retrieved 10/15 statements.


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
    var_5 = '# some code'

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
    var_4 = 'file.txt'
    var_5 = 'content'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = '__init__.py'
    var_5 = b'# '
    var_6 = 2100
    var_7 = var_5 * var_6
    var_8 = b"__import__('pkg_resources').declare_namespace(__name__)"
    var_9 = var_7 + var_8



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_is_namespace_package_predicate_line_5. Retrieved 2/11 statements.


def test_case_0():
    var_0 = '__init__.py'
    var_1 = b"__import__('pkg_resources').declare_namespace(__name__)"



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_is_namespace_package_returns_true_at_line_4. Retrieved 4/15 statements.


def test_case_0():
    var_0 = 'namespace_pkg'
    var_1 = 'module.py'
    var_2 = '# module'
    var_3 = True
    assert var_3 is True



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_is_namespace_package_predicate_line_6_true. Retrieved 5/27 statements.


def test_case_0():
    var_0 = 'test_package'
    var_1 = 'py'
    var_2 = 'pyx'
    var_3 = [var_1, var_2]
    var_4 = frozenset(var_3)



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_is_namespace_package_predicate_line_6_true. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'test_package'
    var_1 = '__init__.pyi'
    var_2 = ''
    var_3 = '__init__.py'



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_is_namespace_package_predicate_line_6_true. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'test_package'
    var_1 = 'module.py'
    var_2 = '# test module'
    var_3 = '__init__.py'



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_is_namespace_package_predicate_line_13_true. Retrieved 8/37 statements.


def test_case_0():
    var_0 = 'test_package'
    var_1 = '.gitkeep'
    var_2 = 'py'
    var_3 = 'pyx'
    var_4 = {var_2, var_3}
    var_5 = frozenset(var_4)
    var_6 = 'module.py'
    var_7 = '# test'



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_is_namespace_package_predicate_line_6_true. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'test_package'
    var_1 = '__init__.py'



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_is_namespace_package_not_a_package. Retrieved 4/7 statements.
# Partially parsed test_is_namespace_package_regular_package_with_init. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_namespace_single_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_namespace_double_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path_single_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path_double_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_without_init_with_python_files. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_without_init_with_setup_cfg. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_without_init_with_pyproject_toml. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_without_init_no_source_files. Retrieved 6/12 statements.


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
    var_2 = '# setup'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = 'pyproject.toml'
    var_2 = '# config'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = 'readme.txt'
    var_2 = '# readme'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_is_namespace_package_predicate_at_line_5. Retrieved 3/14 statements.


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = b"__import__('pkg_resources').declare_namespace(__name__)"



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_is_namespace_package_returns_true_for_namespace_package. Retrieved 7/30 statements.


def test_case_0():
    var_0 = 'mypkg'
    var_1 = '__init__.py'
    var_2 = b"__import__('pkg_resources').declare_namespace(__name__)"
    var_3 = '__main__._is_package'
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_is_namespace_package_returns_true_when_conditions_met. Retrieved 13/35 statements.


def test_case_0():
    var_0 = 'test_package'
    var_1 = '__init__.py'
    var_2 = b"__import__('pkg_resources').declare_namespace(__name__)"
    var_3 = 'py'
    var_4 = 'pyx'
    var_5 = [var_3, var_4]
    var_6 = frozenset(var_5)
    var_7 = '__init__.py'
    var_8 = 4096
    var_9 = b"__import__('pkg_resources').declare_namespace(__name__)"
    var_10 = b'__import__("pkg_resources").declare_namespace(__name__)'
    var_11 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_12 = b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)'



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_is_namespace_package_predicate_at_line_5. Retrieved 3/16 statements.


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = b"__import__('pkg_resources').declare_namespace(__name__)"



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_is_namespace_package_not_a_package. Retrieved 4/7 statements.
# Partially parsed test_is_namespace_package_regular_package_with_init. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_namespace_single_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_namespace_double_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path_single_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path_double_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_py_files. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_no_py_files. Retrieved 4/8 statements.
# Partially parsed test_is_namespace_package_no_init_with_setup_cfg. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_pyproject_toml. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_multiple_extensions. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'not_a_package'
    var_1 = 'py'
    var_2 = {var_1}
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'regular_package'
    var_1 = '__init__.py'
    var_2 = '# regular init file'
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
    var_1 = 'py'
    var_2 = {var_1}
    var_3 = frozenset(var_2)

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
    var_1 = 'module.pyx'
    var_2 = '# cython module'
    var_3 = 'py'
    var_4 = 'pyx'
    var_5 = {var_3, var_4}
    var_6 = frozenset(var_5)



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_is_namespace_package_predicate_at_line_5. Retrieved 3/14 statements.


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = b"__import__('pkg_resources').declare_namespace(__name__)"



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_is_namespace_package_predicate_at_line_5. Retrieved 3/14 statements.


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = b"__import__('pkg_resources').declare_namespace(__name__)"



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_is_namespace_package_predicate_line_13_true. Retrieved 7/30 statements.


def test_case_0():
    var_0 = '__main__._is_package'
    var_1 = 'test_namespace'
    var_2 = 'module.py'
    var_3 = '# test module'
    var_4 = 'txt'
    var_5 = [var_4]
    var_6 = frozenset(var_5)



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 7/29 statements.


def test_case_0():
    var_0 = 'pathlib.Path.exists'
    var_1 = False
    var_2 = lambda self: var_1
    var_3 = 'test_module'
    var_4 = '.py'
    var_5 = 'builtins.__import__'
    var_6 = '__init__.py'



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_src_path_predicate_line_26_evaluates_to_true. Retrieved 3/21 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = '.py'
    var_2 = 'mymodule'



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_is_namespace_package_returns_true_at_line_4. Retrieved 6/32 statements.


def test_case_0():
    var_0 = 'test_namespace_pkg'
    var_1 = '__init__.py'
    var_2 = b"__import__('pkg_resources').declare_namespace(__name__)"
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_src_path_is_module_predicate_evaluates_to_true. Retrieved 3/16 statements.


def test_case_0():
    var_0 = 'my_module'
    var_1 = '__main__.exists_case_sensitive'
    var_2 = 'my_module'



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_is_namespace_package_predicate_line_6_true. Retrieved 4/32 statements.


def test_case_0():
    var_0 = 'test_package'
    var_1 = 'submodule'
    var_2 = '__init__.py'
    var_3 = frozenset()



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_src_path_predicate_line_26_evaluates_to_true. Retrieved 4/18 statements.


def test_case_0():
    var_0 = '.py'
    var_1 = 'test_module.py'
    var_2 = '# test module'
    var_3 = 'test_module'



# Parsed testcases at query #87
#--------------------------

# Partially parsed test_src_path_returns_none_when_module_not_found. Retrieved 1/9 statements.
# Partially parsed test_src_path_returns_firstparty_when_module_exists. Retrieved 2/13 statements.
# Partially parsed test_src_path_with_nested_module_not_namespace. Retrieved 2/12 statements.
# Partially parsed test_src_path_with_custom_src_paths. Retrieved 3/17 statements.
# Partially parsed test_src_path_with_prefix. Retrieved 3/12 statements.
# Partially parsed test_src_path_src_path_is_module_match. Retrieved 1/10 statements.
# Partially parsed test_src_path_with_namespace_package_in_config. Retrieved 4/15 statements.


def test_case_0():
    var_0 = 'nonexistent_module'

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module'

def test_case_0():
    var_0 = 'parent'
    var_1 = 'parent.child'

def test_case_0():
    var_0 = 'custom_src'
    var_1 = 'my_module'
    var_2 = 'my_module'

def test_case_0():
    var_0 = 'module_name'
    var_1 = 'parent'
    var_2 = (var_1,)

def test_case_0():
    var_0 = 'test_module'

def test_case_0():
    var_0 = 'parent.child'
    var_1 = [var_0]
    var_2 = 'parent'
    var_3 = 'parent.child.module'



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'importlib.machinery.EXTENSION_SUFFIXES'
    var_1 = []
    var_2 = '__main__'
    var_3 = 'exists_case_sensitive'
    var_4 = 'test_module'



# Parsed testcases at query #89
#--------------------------

# Partially parsed test_known_pattern_matches_exact_module. Retrieved 8/22 statements.
# Partially parsed test_known_pattern_matches_submodule. Retrieved 8/22 statements.
# Partially parsed test_known_pattern_matches_longest_module_first. Retrieved 7/21 statements.
# Partially parsed test_known_pattern_no_match. Retrieved 5/15 statements.
# Partially parsed test_known_pattern_placement_not_in_sections. Retrieved 5/13 statements.
# Partially parsed test_known_pattern_empty_patterns. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'first_party'
    var_1 = 'third_party'
    var_2 = '^django$'
    var_3 = '^myapp$'
    var_4 = 'myapp'
    var_5 = 'Matched configured known pattern '
    var_6 = 0
    var_7 = 1

def test_case_0():
    var_0 = 'first_party'
    var_1 = 'third_party'
    var_2 = '^django'
    var_3 = '^myapp'
    var_4 = 'myapp.utils.helpers'
    var_5 = 'Matched configured known pattern '
    var_6 = 0
    var_7 = 1

def test_case_0():
    var_0 = 'first_party'
    var_1 = 'third_party'
    var_2 = '^myapp\\.utils$'
    var_3 = '^myapp$'
    var_4 = 'myapp.utils.helpers'
    var_5 = 'Matched configured known pattern '
    var_6 = 0

def test_case_0():
    var_0 = 'first_party'
    var_1 = 'third_party'
    var_2 = '^django$'
    var_3 = '^flask$'
    var_4 = 'myapp'

def test_case_0():
    var_0 = 'first_party'
    var_1 = 'third_party'
    var_2 = '^myapp$'
    var_3 = 'unknown_section'
    var_4 = 'myapp'

def test_case_0():
    var_0 = 'first_party'
    var_1 = 'third_party'
    var_2 = 'myapp'



# Parsed testcases at query #90
#--------------------------




import isort.settings as module_0

def test_case_0():
    var_0 = 'example_module'
    var_1 = 'example_module'
    var_2 = [var_1]
    var_3 = module_0.Config(var_2)
    var_4 = False
    var_5 = True
    assert var_5 is True



# Parsed testcases at query #91
#--------------------------

# Partially parsed test_namespace_package_predicate_evaluates_to_true. Retrieved 3/16 statements.


def test_case_0():
    var_0 = 'myapp'
    var_1 = 'submodule'
    var_2 = 'myapp.submodule'



# Parsed testcases at query #92
#--------------------------

# Partially parsed test_src_path_finds_module_in_src_paths. Retrieved 9/21 statements.
# Partially parsed test_src_path_returns_none_for_missing_module. Retrieved 5/12 statements.
# Partially parsed test_src_path_finds_package_in_src_paths. Retrieved 9/23 statements.
# Partially parsed test_src_path_with_nested_module. Retrieved 11/29 statements.
# Partially parsed test_src_path_with_custom_src_paths. Retrieved 9/22 statements.


def test_case_0():
    var_0 = 'src'
    var_1 = 'test_module.py'
    var_2 = '# test module'
    var_3 = frozenset()
    var_4 = False
    var_5 = frozenset(['py'])
    var_6 = 'FIRSTPARTY'
    var_7 = 'sections'
    var_8 = 'test_module'

def test_case_0():
    var_0 = 'src'
    var_1 = frozenset()
    var_2 = False
    var_3 = frozenset(['py'])
    var_4 = 'nonexistent_module'

def test_case_0():
    var_0 = 'src'
    var_1 = 'test_package'
    var_2 = '__init__.py'
    var_3 = '# test package'
    var_4 = frozenset()
    var_5 = False
    var_6 = frozenset(['py'])
    var_7 = 'FIRSTPARTY'
    var_8 = 'sections'

def test_case_0():
    var_0 = 'src'
    var_1 = 'parent'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'child'
    var_5 = frozenset()
    var_6 = False
    var_7 = frozenset(['py'])
    var_8 = 'FIRSTPARTY'
    var_9 = 'sections'
    var_10 = 'parent.child'

def test_case_0():
    var_0 = 'custom_src'
    var_1 = 'custom_module.py'
    var_2 = '# custom module'
    var_3 = frozenset()
    var_4 = False
    var_5 = frozenset(['py'])
    var_6 = 'FIRSTPARTY'
    var_7 = 'sections'
    var_8 = 'custom_module'



# Parsed testcases at query #93
#--------------------------

# Partially parsed test_is_namespace_package_predicate_line_4_true. Retrieved 7/20 statements.


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = b"__import__('pkg_resources').declare_namespace(__name__)"
    var_3 = 'py'
    var_4 = 'pyx'
    var_5 = {var_3, var_4}
    var_6 = frozenset(var_5)



# Parsed testcases at query #94
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
# Partially parsed test_is_namespace_package_no_init_no_source_files. Retrieved 4/7 statements.
# Partially parsed test_is_namespace_package_no_init_with_other_extensions. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'nonexistent'

def test_case_0():
    var_0 = 'regular_package'
    var_1 = '__init__.py'
    var_2 = '# regular package'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = '__init__.py'
    var_2 = "__import__('pkg_resources').declare_namespace(__name__)"
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = '__init__.py'
    var_2 = '__import__("pkg_resources").declare_namespace(__name__)'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = '__init__.py'
    var_2 = "__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = '__init__.py'
    var_2 = '__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = 'module.py'
    var_2 = '# some module'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = 'setup.cfg'
    var_2 = '[metadata]'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = 'pyproject.toml'
    var_2 = '[build-system]'
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
    var_1 = 'data.txt'
    var_2 = 'some data'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)



# Parsed testcases at query #95
#--------------------------

# Partially parsed test_is_namespace_package_not_a_package. Retrieved 4/7 statements.
# Partially parsed test_is_namespace_package_regular_package_with_init. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_namespace. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_double_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_double_quotes. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_python_files. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_setup_cfg. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_with_pyproject_toml. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_no_init_no_files. Retrieved 4/8 statements.
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
    var_2 = '# some module'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'ns_package'
    var_1 = 'setup.cfg'
    var_2 = ''
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'ns_package'
    var_1 = 'pyproject.toml'
    var_2 = ''
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
    var_1 = 'readme.txt'
    var_2 = 'readme'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)



# Parsed testcases at query #96
#--------------------------

# Partially parsed test_namespace_package_predicate_evaluates_to_true. Retrieved 6/20 statements.


def test_case_0():
    var_0 = 'myapp'
    var_1 = 'submodule'
    var_2 = True
    var_3 = 'nested.py'
    var_4 = '# nested module'
    var_5 = 'myapp.nested'



# Parsed testcases at query #97
#--------------------------

# Partially parsed test_known_pattern_predicate_false. Retrieved 9/31 statements.


def test_case_0():
    var_0 = 'section1'
    var_1 = 'section2'
    var_2 = True
    var_3 = 'unknown_section'
    var_4 = False
    var_5 = 'test.module'
    var_6 = '.'
    var_7 = -1
    var_8 = var_1 and var_2
    assert var_8 is False



# Parsed testcases at query #98
#--------------------------

# Partially parsed test_src_path_predicate_line_26_true. Retrieved 3/18 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = '.py'
    var_2 = 'mymodule'



# Parsed testcases at query #99
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 6/24 statements.


def test_case_0():
    var_0 = 'test_module.py'
    var_1 = '# test module'
    var_2 = '__main__.exists_case_sensitive'
    var_3 = 'test_module'
    var_4 = '.py'
    var_5 = '__init__.py'



# Parsed testcases at query #100
#--------------------------

# Partially parsed test_src_path_is_module_predicate_evaluates_to_true. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 'test_module'



# Parsed testcases at query #101
#--------------------------

# Partially parsed test_namespace_package_predicate_evaluates_to_false. Retrieved 4/12 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = '.py'
    var_2 = 'mymodule.submodule'
    var_3 = ()



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 3/17 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 3/17 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 3/20 statements.
# Partially parsed test_is_module_not_a_module. Retrieved 1/12 statements.
# Partially parsed test_is_module_regular_directory. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'test_module.py'
    var_1 = ''
    var_2 = 'test_module'

def test_case_0():
    var_0 = 'test_package'
    var_1 = '__init__.py'
    var_2 = ''

def test_case_0():
    var_0 = 0
    var_1 = ''
    var_2 = 'test_module'

def test_case_0():
    var_0 = 'nonexistent_module'

def test_case_0():
    var_0 = 'regular_directory'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 2/8 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 5/15 statements.
# Partially parsed test_is_module_with_init_file. Retrieved 2/8 statements.
# Partially parsed test_is_module_not_a_module. Retrieved 2/8 statements.


def test_case_0():
    var_0 = '__main__.exists_case_sensitive'
    var_1 = 'test_module'

def test_case_0():
    var_0 = 'count'
    var_1 = 0
    var_2 = {var_0: var_1}
    var_3 = '__main__.exists_case_sensitive'
    var_4 = 'test_module'

def test_case_0():
    var_0 = '__main__.exists_case_sensitive'
    var_1 = 'test_package'

def test_case_0():
    var_0 = '__main__.exists_case_sensitive'
    var_1 = 'not_a_module'



# Parsed testcases at query #3
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django/'
    var_1 = [var_0]
    var_2 = module_0.Config(var_1)
    var_3 = 'django/models.py'
    var_4 = module_1._forced_separate(var_3, var_2)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'mypackage'
    var_1 = [var_0]
    var_2 = module_0.Config(var_1)
    var_3 = 'mypackage/utils.py'
    var_4 = module_1._forced_separate(var_3, var_2)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django/'
    var_1 = [var_0]
    var_2 = module_0.Config(var_1)
    var_3 = '.django/models.py'
    var_4 = module_1._forced_separate(var_3, var_2)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django/'
    var_1 = [var_0]
    var_2 = module_0.Config(var_1)
    var_3 = 'flask/app.py'
    var_4 = module_1._forced_separate(var_3, var_2)
    assert var_4 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = module_0.Config(var_0)
    var_2 = 'any/module.py'
    var_3 = module_1._forced_separate(var_2, var_1)
    assert var_3 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django/'
    var_1 = 'flask/'
    var_2 = [var_0, var_1]
    var_3 = module_0.Config(var_2)
    var_4 = 'django/models.py'
    var_5 = module_1._forced_separate(var_4, var_3)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'django/'
    var_1 = 'flask/'
    var_2 = [var_0, var_1]
    var_3 = module_0.Config(var_2)
    var_4 = 'flask/app.py'
    var_5 = module_1._forced_separate(var_4, var_3)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = [var_0]
    var_2 = module_0.Config(var_1)
    var_3 = module_1._forced_separate(var_0, var_2)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'src/*.py'
    var_1 = [var_0]
    var_2 = module_0.Config(var_1)
    var_3 = 'src/main.py'
    var_4 = module_1._forced_separate(var_3, var_2)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_src_path_finds_module_in_src_paths. Retrieved 7/21 statements.
# Partially parsed test_src_path_returns_none_when_module_not_found. Retrieved 6/16 statements.
# Partially parsed test_src_path_with_nested_module. Retrieved 9/27 statements.
# Partially parsed test_src_path_with_namespace_package. Retrieved 8/20 statements.
# Partially parsed test_src_path_uses_default_src_paths. Retrieved 8/22 statements.


def test_case_0():
    var_0 = 'src'
    var_1 = 'mymodule'
    var_2 = '__init__.py'
    var_3 = 'importlib.machinery.EXTENSION_SUFFIXES'
    var_4 = []
    var_5 = 'exists_case_sensitive'
    var_6 = True

def test_case_0():
    var_0 = 'src'
    var_1 = 'importlib.machinery.EXTENSION_SUFFIXES'
    var_2 = []
    var_3 = 'exists_case_sensitive'
    var_4 = False
    var_5 = 'nonexistent'

def test_case_0():
    var_0 = 'src'
    var_1 = 'mypackage'
    var_2 = '__init__.py'
    var_3 = 'nested'
    var_4 = 'importlib.machinery.EXTENSION_SUFFIXES'
    var_5 = []
    var_6 = 'exists_case_sensitive'
    var_7 = True
    var_8 = 'mypackage.nested'

def test_case_0():
    var_0 = 'src'
    var_1 = 'namespace_pkg'
    var_2 = [var_1]
    var_3 = 'importlib.machinery.EXTENSION_SUFFIXES'
    var_4 = []
    var_5 = 'exists_case_sensitive'
    var_6 = True
    var_7 = 'namespace_pkg.submodule'

def test_case_0():
    var_0 = 'src'
    var_1 = 'mymodule'
    var_2 = '__init__.py'
    var_3 = 'importlib.machinery.EXTENSION_SUFFIXES'
    var_4 = []
    var_5 = 'exists_case_sensitive'
    var_6 = True
    var_7 = None



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_src_path_predicate_line_26_true. Retrieved 4/25 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = '.py'
    var_2 = '/src/mymodule'
    var_3 = 'mymodule'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_src_path_finds_module_in_src_paths. Retrieved 8/24 statements.
# Partially parsed test_src_path_returns_none_for_missing_module. Retrieved 7/19 statements.
# Partially parsed test_src_path_with_nested_module_namespace_package. Retrieved 11/34 statements.
# Partially parsed test_src_path_with_custom_src_paths. Retrieved 7/22 statements.


def test_case_0():
    var_0 = 'src'
    var_1 = 'mymodule'
    var_2 = '__init__.py'
    var_3 = 'importlib.machinery.EXTENSION_SUFFIXES'
    var_4 = []
    var_5 = 'pathlib.Path.exists'
    var_6 = True
    var_7 = 'pathlib.Path.is_dir'

def test_case_0():
    var_0 = 'src'
    var_1 = 'importlib.machinery.EXTENSION_SUFFIXES'
    var_2 = []
    var_3 = 'pathlib.Path.exists'
    var_4 = False
    var_5 = 'pathlib.Path.is_dir'
    var_6 = 'nonexistent'

def test_case_0():
    var_0 = 'src'
    var_1 = 'parent'
    var_2 = 'child'
    var_3 = '__init__.py'
    var_4 = 'importlib.machinery.EXTENSION_SUFFIXES'
    var_5 = []
    var_6 = 'pathlib.Path.exists'
    var_7 = True
    var_8 = 'pathlib.Path.is_dir'
    var_9 = 'pathlib.Path.resolve'
    var_10 = 'parent.child'

def test_case_0():
    var_0 = 'custom'
    var_1 = 'testmod'
    var_2 = 'importlib.machinery.EXTENSION_SUFFIXES'
    var_3 = []
    var_4 = 'pathlib.Path.exists'
    var_5 = True
    var_6 = 'pathlib.Path.is_dir'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_at_line_16_evaluates_to_true. Retrieved 9/29 statements.


def test_case_0():
    var_0 = 'mymodule'
    var_1 = 'test.py'
    var_2 = '# test'
    var_3 = 'mymodule'
    var_4 = ()
    var_5 = 0
    var_6 = '.'
    var_7 = 1
    var_8 = name.split(var_6, var_7)[var_5]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_is_module_predicate_evaluates_to_true. Retrieved 7/22 statements.


def test_case_0():
    var_0 = 'importlib.machinery.EXTENSION_SUFFIXES'
    var_1 = []
    var_2 = 'exists_case_sensitive'
    var_3 = 'test_module'
    var_4 = '.py'
    var_5 = []
    var_6 = '__init__.py'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_known_pattern_match_found. Retrieved 19/28 statements.
# Partially parsed test_known_pattern_no_match. Retrieved 19/28 statements.
# Partially parsed test_known_pattern_placement_not_in_sections. Retrieved 16/23 statements.
# Partially parsed test_known_pattern_multiple_patterns. Retrieved 22/31 statements.
# Partially parsed test_known_pattern_longest_match. Retrieved 10/21 statements.
# Partially parsed test_known_pattern_empty_patterns. Retrieved 11/13 statements.


def test_case_0():
    var_0 = 'Config'
    var_1 = 'known_patterns'
    var_2 = 'sections'
    var_3 = [var_1, var_2]
    var_4 = 'Pattern'
    var_5 = 'pattern'
    var_6 = 'placement'
    var_7 = [var_5, var_6]
    var_8 = 'PatternObj'
    var_9 = ()
    var_10 = 'match'
    var_11 = 'django'
    var_12 = lambda self, x: x == var_11
    var_13 = {var_10: var_12}
    var_14 = type(var_8, var_9, var_13)
    var_15 = 'third_party'
    var_16 = 'stdlib'
    var_17 = [var_15, var_16]
    var_18 = 'django.db'

def test_case_0():
    var_0 = 'Config'
    var_1 = 'known_patterns'
    var_2 = 'sections'
    var_3 = [var_1, var_2]
    var_4 = 'Pattern'
    var_5 = 'pattern'
    var_6 = 'placement'
    var_7 = [var_5, var_6]
    var_8 = 'PatternObj'
    var_9 = ()
    var_10 = 'match'
    var_11 = False
    var_12 = lambda self, x: var_11
    var_13 = {var_10: var_12}
    var_14 = type(var_8, var_9, var_13)
    var_15 = 'third_party'
    var_16 = 'stdlib'
    var_17 = [var_15, var_16]
    var_18 = 'mymodule'

def test_case_0():
    var_0 = 'Config'
    var_1 = 'known_patterns'
    var_2 = 'sections'
    var_3 = [var_1, var_2]
    var_4 = 'PatternObj'
    var_5 = ()
    var_6 = 'match'
    var_7 = 'django'
    var_8 = lambda self, x: x == var_7
    var_9 = {var_6: var_8}
    var_10 = type(var_4, var_5, var_9)
    var_11 = 'invalid_section'
    var_12 = 'third_party'
    var_13 = 'stdlib'
    var_14 = [var_12, var_13]
    var_15 = 'django.db'

def test_case_0():
    var_0 = 'Config'
    var_1 = 'known_patterns'
    var_2 = 'sections'
    var_3 = [var_1, var_2]
    var_4 = 'Pattern1'
    var_5 = ()
    var_6 = 'match'
    var_7 = False
    var_8 = lambda self, x: var_7
    var_9 = {var_6: var_8}
    var_10 = type(var_4, var_5, var_9)
    var_11 = 'Pattern2'
    var_12 = ()
    var_13 = 'requests'
    var_14 = lambda self, x: x == var_13
    var_15 = {var_6: var_14}
    var_16 = type(var_11, var_12, var_15)
    var_17 = 'first'
    var_18 = 'third_party'
    var_19 = 'stdlib'
    var_20 = [var_18, var_19, var_17]
    var_21 = 'requests.api'

def test_case_0():
    var_0 = 'Config'
    var_1 = 'known_patterns'
    var_2 = 'sections'
    var_3 = [var_1, var_2]
    var_4 = 'django'
    var_5 = 'django.db'
    var_6 = 'section1'
    var_7 = 'section2'
    var_8 = [var_6, var_7]
    var_9 = 'django.db.models'

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'Config'
    var_1 = 'known_patterns'
    var_2 = 'sections'
    var_3 = [var_1, var_2]
    var_4 = []
    var_5 = 'third_party'
    var_6 = 'stdlib'
    var_7 = [var_5, var_6]
    var_8 = module_0.Config()
    var_9 = 'django'
    var_10 = module_1._known_pattern(var_9, var_8)
    assert var_10 is None



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_src_path_predicate_line_26_evaluates_to_true. Retrieved 7/24 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = '.py'
    var_2 = '/src/mymodule'
    var_3 = True
    var_4 = False
    var_5 = False
    var_6 = var_3 or var_4 or var_5
    assert var_6 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_src_path_finds_module_in_src_paths. Retrieved 9/45 statements.
# Partially parsed test_src_path_returns_none_for_missing_module. Retrieved 2/13 statements.
# Partially parsed test_src_path_with_nested_module_names. Retrieved 9/39 statements.


def test_case_0():
    var_0 = 'src'
    var_1 = 'mymodule.py'
    var_2 = '# test module'
    var_3 = 'exists_case_sensitive'
    var_4 = '_is_module'
    var_5 = '_is_package'
    var_6 = '_is_namespace_package'
    var_7 = '_src_path_is_module'
    var_8 = 'mymodule'

def test_case_0():
    var_0 = 'src'
    var_1 = 'nonexistent_module'

def test_case_0():
    var_0 = 'src'
    var_1 = 'mypackage'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'exists_case_sensitive'
    var_5 = '_is_module'
    var_6 = '_is_package'
    var_7 = '_is_namespace_package'
    var_8 = '_src_path_is_module'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_src_path_simple_module. Retrieved 4/17 statements.
# Partially parsed test_src_path_nested_module. Retrieved 4/17 statements.
# Partially parsed test_src_path_module_not_found. Retrieved 4/19 statements.
# Partially parsed test_src_path_with_custom_src_paths. Retrieved 5/20 statements.
# Partially parsed test_src_path_with_prefix. Retrieved 6/19 statements.
# Partially parsed test_src_path_is_package. Retrieved 4/18 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = 'mymodule'

def test_case_0():
    var_0 = '/src'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = 'mypackage.submodule'

def test_case_0():
    var_0 = '/src'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = 'nonexistent'

def test_case_0():
    var_0 = '/default'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = '/custom'
    var_4 = 'mymodule'

def test_case_0():
    var_0 = '/src'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = 'submodule'
    var_4 = 'mypackage'
    var_5 = (var_4,)

def test_case_0():
    var_0 = '/src'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = 'mypackage'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_is_namespace_package_not_a_package. Retrieved 4/7 statements.
# Partially parsed test_is_namespace_package_regular_package_with_init. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_namespace_single_quote. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_declare_namespace_double_quote. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path_single_quote. Retrieved 6/12 statements.
# Partially parsed test_is_namespace_package_with_pkgutil_extend_path_double_quote. Retrieved 6/12 statements.
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
    var_5 = '[metadata]'

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)
    var_3 = 'pkg'
    var_4 = 'pyproject.toml'
    var_5 = '[project]'

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



