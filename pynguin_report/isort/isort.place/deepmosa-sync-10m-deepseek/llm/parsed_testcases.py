####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_known_pattern_matches. Retrieved 6/22 statements.
# Partially parsed test_known_pattern_matches_deeper. Retrieved 4/18 statements.
# Partially parsed test_known_pattern_no_match. Retrieved 3/15 statements.
# Partially parsed test_known_pattern_section_not_in_config. Retrieved 4/16 statements.
# Partially parsed test_known_pattern_empty_name. Retrieved 3/15 statements.
# Partially parsed test_known_pattern_multiple_patterns_first_wins. Retrieved 5/21 statements.
# Partially parsed test_known_pattern_check_order_longest_first. Retrieved 6/22 statements.


def test_case_0():
    var_0 = 'section1'
    var_1 = 'section2'
    var_2 = [var_0, var_1]
    var_3 = 'test'
    var_4 = 'foo.bar'
    var_5 = 'test.module'

def test_case_0():
    var_0 = 'section1'
    var_1 = [var_0]
    var_2 = 'a.b'
    var_3 = 'a.b.c'

def test_case_0():
    var_0 = 'section1'
    var_1 = [var_0]
    var_2 = 'unknown.module'

def test_case_0():
    var_0 = 'section1'
    var_1 = [var_0]
    var_2 = 'section2'
    var_3 = 'any.module'

def test_case_0():
    var_0 = 'section1'
    var_1 = [var_0]
    var_2 = ''

def test_case_0():
    var_0 = 'section1'
    var_1 = 'section2'
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = 'module'

def test_case_0():
    var_0 = 'section1'
    var_1 = 'section2'
    var_2 = [var_0, var_1]
    var_3 = 'a'
    var_4 = 'a.b'
    var_5 = 'a.b.c'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test__src_path_with_exact_module_match_in_src_paths. Retrieved 7/13 statements.
# Partially parsed test__src_path_with_package_match_in_src_paths. Retrieved 7/13 statements.
# Partially parsed test__src_path_with_nested_module_in_namespace_package. Retrieved 8/14 statements.
# Partially parsed test__src_path_with_nested_module_auto_identified_namespace_package. Retrieved 7/13 statements.
# Partially parsed test__src_path_with_no_match_in_src_paths. Retrieved 7/13 statements.
# Partially parsed test__src_path_with_src_path_is_module_match. Retrieved 7/13 statements.
# Partially parsed test__src_path_with_custom_src_paths_argument. Retrieved 8/14 statements.
# Partially parsed test__src_path_with_prefix_argument. Retrieved 9/15 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'mymodule'
    var_8 = [var_0]

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'mypackage'
    var_8 = [var_0]

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'mynamespace'
    var_3 = {var_2}
    var_4 = False
    var_5 = 'py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)
    var_8 = 'mynamespace.nested'
    var_9 = [var_0]

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = True
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'mynamespace.nested'
    var_8 = [var_0]

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'unknown'
    var_8 = [var_0]

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'src'
    var_8 = [var_0]

def test_case_0():
    var_0 = '/other'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'mymodule'
    var_8 = '/custom'
    var_9 = [var_8]

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'submodule'
    var_8 = [var_0]
    var_9 = 'mypackage'
    var_10 = (var_9,)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 2/12 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 1/13 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 2/11 statements.
# Partially parsed test_is_module_without_any_module_files. Retrieved 1/8 statements.
# Partially parsed test_is_module_case_sensitive_check. Retrieved 2/15 statements.


def test_case_0():
    var_0 = 'module'
    var_1 = '.py'

def test_case_0():
    var_0 = 'module'

def test_case_0():
    var_0 = 'module'
    var_1 = '__init__.py'

def test_case_0():
    var_0 = 'not_a_module'

def test_case_0():
    var_0 = 'Module'
    var_1 = '.py'



# Parsed testcases at query #4
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'src'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = module_1._forced_separate(var_0, var_4)
    var_6 = bool(var_5 == ('src', 'Matched forced_separate (src) config value.'))
    assert var_6 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'src'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'src/main.py'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('src', 'Matched forced_separate (src) config value.'))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'src'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '.src'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('src', 'Matched forced_separate (src) config value.'))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'src'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '.src/main.py'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('src', 'Matched forced_separate (src) config value.'))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'src/*'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'src/main.py'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('src/*', 'Matched forced_separate (src/*) config value.'))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'src/*'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '.src/main.py'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('src/*', 'Matched forced_separate (src/*) config value.'))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'src'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'tests'
    var_6 = module_1._forced_separate(var_5, var_4)
    assert var_6 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'src'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'source'
    var_6 = module_1._forced_separate(var_5, var_4)
    assert var_6 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = 'forced_separate'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'src'
    var_5 = module_1._forced_separate(var_4, var_3)
    assert var_5 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'docs'
    var_1 = 'src'
    var_2 = [var_0, var_1]
    var_3 = 'forced_separate'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = module_1._forced_separate(var_1, var_5)
    var_7 = bool(var_6 == ('docs', 'Matched forced_separate (docs) config value.'))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'docs'
    var_1 = 'src'
    var_2 = [var_0, var_1]
    var_3 = 'forced_separate'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = 'src/main.py'
    var_7 = module_1._forced_separate(var_6, var_5)
    var_8 = bool(var_7 == ('docs', 'Matched forced_separate (docs) config value.'))
    assert var_8 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_is_namespace_package_with_valid_namespace_package_without_init. Retrieved 7/19 statements.
# Partially parsed test_is_namespace_package_with_non_package_path. Retrieved 5/11 statements.
# Partially parsed test_is_namespace_package_with_init_containing_pkg_resources_declare_namespace_single_quotes. Retrieved 6/17 statements.
# Partially parsed test_is_namespace_package_with_init_containing_pkg_resources_declare_namespace_double_quotes. Retrieved 6/17 statements.
# Partially parsed test_is_namespace_package_with_init_containing_pkgutil_extend_path_single_quotes. Retrieved 6/17 statements.
# Partially parsed test_is_namespace_package_with_init_containing_pkgutil_extend_path_double_quotes. Retrieved 6/17 statements.
# Partially parsed test_is_namespace_package_with_init_missing_namespace_markers. Retrieved 6/17 statements.
# Partially parsed test_is_namespace_package_without_init_but_with_py_files. Retrieved 6/20 statements.
# Partially parsed test_is_namespace_package_without_init_but_with_setup_cfg. Retrieved 6/20 statements.
# Partially parsed test_is_namespace_package_without_init_but_with_pyproject_toml. Retrieved 6/20 statements.
# Partially parsed test_is_namespace_package_without_init_and_no_files. Retrieved 7/17 statements.


def test_case_0():
    var_0 = '/fake/path'
    var_1 = [var_0]
    var_2 = True
    var_3 = False
    var_4 = []
    var_5 = 'py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/fake/path'
    var_1 = False
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/fake/path'
    var_1 = True
    var_2 = b"__import__('pkg_resources').declare_namespace(__name__)"
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = '/fake/path'
    var_1 = True
    var_2 = b'__import__("pkg_resources").declare_namespace(__name__)'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = '/fake/path'
    var_1 = True
    var_2 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = '/fake/path'
    var_1 = True
    var_2 = b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = '/fake/path'
    var_1 = True
    var_2 = b"print('hello')"
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = '/fake/path'
    var_1 = True
    var_2 = False
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = '/fake/path'
    var_1 = True
    var_2 = False
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = '/fake/path'
    var_1 = True
    var_2 = False
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = '/fake/path'
    var_1 = True
    var_2 = False
    var_3 = []
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_src_path_finds_module_in_src_paths. Retrieved 8/14 statements.
# Partially parsed test_src_path_returns_none_for_missing_module. Retrieved 7/11 statements.
# Partially parsed test_src_path_handles_nested_module_in_namespace_package. Retrieved 9/15 statements.
# Partially parsed test_src_path_handles_nested_module_with_auto_identify. Retrieved 8/14 statements.
# Partially parsed test_src_path_uses_custom_src_paths. Retrieved 8/16 statements.
# Partially parsed test_src_path_handles_root_module_matching_src_path_name. Retrieved 8/14 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'mymodule'
    var_8 = 'Found in one of the configured src_paths: /src.'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'missing'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'parent'
    var_3 = {var_2}
    var_4 = False
    var_5 = 'py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)
    var_8 = 'parent.child'
    var_9 = 'Found in one of the configured src_paths: /src.'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = True
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'parent.child'
    var_8 = 'Found in one of the configured src_paths: /src.'

def test_case_0():
    var_0 = '/custom'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'mymodule'
    var_8 = [var_0]
    var_9 = 'Found in one of the configured src_paths: /custom.'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'src'
    var_8 = 'Found in one of the configured src_paths: /src.'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_at_line_16_evaluates_to_true. Retrieved 3/18 statements.


def test_case_0():
    var_0 = '/some/path'
    var_1 = [var_0]
    var_2 = '/some/path/root_module_name'
    var_3 = [var_2]
    var_4 = ()



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_namespace_in_config_namespace_packages. Retrieved 9/15 statements.
# Partially parsed test_auto_identify_namespace_packages_true_and_is_namespace_package. Retrieved 10/16 statements.
# Partially parsed test_nested_module_true_and_namespace_in_config_namespace_packages. Retrieved 9/15 statements.
# Partially parsed test_nested_module_true_and_auto_identify_true_and_is_namespace_package. Retrieved 10/16 statements.


def test_case_0():
    var_0 = 'some.namespace'
    var_1 = {var_0}
    var_2 = False
    var_3 = '/src'
    var_4 = [var_3]
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = 'some.namespace.module'
    var_8 = [var_3]
    var_9 = 'some'
    var_10 = (var_9,)

def test_case_0():
    var_0 = set()
    var_1 = True
    var_2 = '/src'
    var_3 = [var_2]
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = 'some.namespace.module'
    var_7 = '/src'
    var_8 = [var_7]
    var_9 = [var_2]
    var_10 = 'some'
    var_11 = (var_10,)

def test_case_0():
    var_0 = 'some.namespace'
    var_1 = {var_0}
    var_2 = False
    var_3 = '/src'
    var_4 = [var_3]
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = 'some.namespace.module'
    var_8 = [var_3]
    var_9 = 'some'
    var_10 = (var_9,)

def test_case_0():
    var_0 = set()
    var_1 = True
    var_2 = '/src'
    var_3 = [var_2]
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = 'some.namespace.module'
    var_7 = '/src'
    var_8 = [var_7]
    var_9 = [var_2]
    var_10 = 'some'
    var_11 = (var_10,)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_predicate_at_line_16_true. Retrieved 11/25 statements.


def test_case_0():
    var_0 = '_test_module'
    var_1 = None
    var_2 = []
    var_3 = set()
    var_4 = False
    var_5 = ()
    var_6 = '/tmp/test_root'
    var_7 = [var_6]
    var_8 = True
    var_9 = '__init__.py'
    var_10 = 'test_root'
    var_11 = ()



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 4/9 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 5/11 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'some_module'
    var_1 = [var_0]
    var_2 = 'some_module.py'
    var_3 = lambda p: p == var_2

def test_case_0():
    var_0 = 'some_extension'
    var_1 = [var_0]
    var_2 = 'some_extension.pyd'
    var_3 = lambda p: p == var_2
    var_4 = '.pyd'

def test_case_0():
    var_0 = 'some_package'
    var_1 = [var_0]
    var_2 = 'some_package/__init__.py'
    var_3 = lambda p: p == var_2



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_src_path_is_module_true_for_valid_module_dir. Retrieved 1/5 statements.
# Partially parsed test_src_path_is_module_false_for_wrong_name. Retrieved 2/6 statements.
# Partially parsed test_src_path_is_module_false_for_file. Retrieved 1/5 statements.
# Partially parsed test_src_path_is_module_false_for_nonexistent_path. Retrieved 1/3 statements.
# Partially parsed test_src_path_is_module_false_for_case_mismatch. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]
    var_2 = 'wrong_name'

def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'nonexistent_module'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'TestModule'
    var_1 = [var_0]
    var_2 = 'testmodule'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_known_pattern_predicate_false. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'section1'
    var_1 = 'section2'
    var_2 = [var_0, var_1]
    var_3 = 'module.submodule'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_is_module_with_py_extension. Retrieved 9/20 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 9/21 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 9/20 statements.


def test_case_0():
    var_0 = 'some_module'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = 'builtins'
    var_4 = __import__(var_3)
    var_5 = 'exists_case_sensitive'
    var_6 = 'exists_case_sensitive'
    var_7 = 'builtins'
    var_8 = __import__(var_7)
    var_9 = var_8.__dict__[var_6]

def test_case_0():
    var_0 = 'some_module'
    var_1 = [var_0]
    var_2 = 0
    var_3 = 'builtins'
    var_4 = __import__(var_3)
    var_5 = 'exists_case_sensitive'
    var_6 = 'exists_case_sensitive'
    var_7 = 'builtins'
    var_8 = __import__(var_7)
    var_9 = var_8.__dict__[var_6]

def test_case_0():
    var_0 = 'some_module'
    var_1 = [var_0]
    var_2 = '__init__.py'
    var_3 = 'builtins'
    var_4 = __import__(var_3)
    var_5 = 'exists_case_sensitive'
    var_6 = 'exists_case_sensitive'
    var_7 = 'builtins'
    var_8 = __import__(var_7)
    var_9 = var_8.__dict__[var_6]



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_namespace_package_without_init_and_no_files. Retrieved 4/11 statements.
# Partially parsed test_namespace_package_without_init_and_no_source_files. Retrieved 5/14 statements.
# Partially parsed test_namespace_package_without_init_and_no_source_files_but_setup_cfg. Retrieved 5/14 statements.
# Partially parsed test_namespace_package_without_init_and_no_source_files_but_pyproject_toml. Retrieved 5/14 statements.
# Partially parsed test_namespace_package_without_init_and_source_files. Retrieved 5/14 statements.
# Partially parsed test_namespace_package_with_init_and_namespace_declaration_pkg_resources_single_quotes. Retrieved 6/15 statements.
# Partially parsed test_namespace_package_with_init_and_namespace_declaration_pkg_resources_double_quotes. Retrieved 6/15 statements.
# Partially parsed test_namespace_package_with_init_and_namespace_declaration_pkgutil_single_quotes. Retrieved 6/15 statements.
# Partially parsed test_namespace_package_with_init_and_namespace_declaration_pkgutil_double_quotes. Retrieved 6/15 statements.
# Partially parsed test_namespace_package_with_init_and_no_namespace_declaration. Retrieved 6/15 statements.
# Partially parsed test_namespace_package_with_init_and_namespace_declaration_after_4096_bytes. Retrieved 10/19 statements.
# Partially parsed test_not_a_package. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'README.txt'
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'setup.cfg'
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'pyproject.toml'
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'module.py'
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = "__import__('pkg_resources').declare_namespace(__name__)"
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '__import__("pkg_resources").declare_namespace(__name__)'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = "__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = ''
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = ' '
    var_3 = 4096
    var_4 = var_2 * var_3
    var_5 = "__import__('pkg_resources').declare_namespace(__name__)"
    var_6 = var_4 + var_5
    var_7 = 'py'
    var_8 = [var_7]
    var_9 = frozenset(var_8)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test__src_path_with_exact_module_match. Retrieved 7/11 statements.
# Partially parsed test__src_path_with_nested_module_in_namespace_package. Retrieved 8/14 statements.
# Partially parsed test__src_path_with_auto_identified_namespace_package. Retrieved 7/13 statements.
# Partially parsed test__src_path_no_match. Retrieved 7/11 statements.
# Partially parsed test__src_path_with_custom_src_paths. Retrieved 9/12 statements.
# Partially parsed test__src_path_with_prefix. Retrieved 9/15 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'mymodule'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'mypackage'
    var_3 = {var_2}
    var_4 = False
    var_5 = 'py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)
    var_8 = 'mypackage.nested'
    var_9 = [var_0]

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = True
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'namespace.nested'
    var_8 = [var_0]

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'unknown'

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = set()
    var_2 = False
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)
    var_6 = 'src_paths'
    var_7 = 'namespace_packages'
    var_8 = 'auto_identify_namespace_packages'
    var_9 = 'supported_extensions'
    var_10 = {var_6: var_0, var_7: var_1, var_8: var_2, var_9: var_5}
    var_11 = module_0.Config(**var_10)
    var_12 = 'module'
    var_13 = '/custom'
    var_14 = [var_13]

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'sub.module'
    var_8 = [var_0]
    var_9 = 'base'
    var_10 = (var_9,)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_src_path_finds_module_in_src_paths. Retrieved 4/17 statements.
# Partially parsed test_src_path_returns_none_for_missing_module. Retrieved 4/17 statements.
# Partially parsed test_src_path_handles_nested_module_in_namespace_package. Retrieved 5/18 statements.
# Partially parsed test_src_path_handles_auto_identified_namespace_package. Retrieved 5/20 statements.
# Partially parsed test_src_path_with_custom_src_paths_argument. Retrieved 5/20 statements.
# Partially parsed test_src_path_src_path_is_module_condition. Retrieved 5/20 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = 'mymodule'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = 'missingmodule'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'mypackage'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = 'mypackage.nested'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = True
    var_5 = 'namespace.nested'

def test_case_0():
    var_0 = '/default'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = '/custom'
    var_5 = [var_4]
    var_6 = 'mymodule'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = True
    var_5 = 'src'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 10/25 statements.


def test_case_0():
    var_0 = '/test/path'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = 'module'
    var_7 = [var_0]
    var_8 = ()
    var_9 = '.'
    var_10 = 1
    var_11 = *var_8



# Parsed testcases at query #18
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = set()
    var_1 = False
    var_2 = []
    var_3 = []
    var_4 = 'namespace_packages'
    var_5 = 'auto_identify_namespace_packages'
    var_6 = 'src_paths'
    var_7 = 'supported_extensions'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_2, var_7: var_3}
    var_9 = module_0.Config(**var_8)
    var_10 = 'a.b'
    var_11 = []
    var_12 = ()
    var_13 = module_1._src_path(var_10, var_9, var_11, var_12)
    assert var_13 is None



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_is_namespace_package_returns_false_for_non_package. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_returns_false_for_directory_without_init_and_python_files. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_returns_false_for_directory_without_init_but_has_py_file. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_returns_false_for_directory_without_init_but_has_setup_cfg. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_returns_false_for_directory_without_init_but_has_pyproject_toml. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_returns_false_for_init_without_namespace_declaration. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_returns_true_for_init_with_pkg_resources_single_quote. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_returns_true_for_init_with_pkg_resources_double_quote. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_returns_true_for_init_with_pkgutil_single_quote. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_returns_true_for_init_with_pkgutil_double_quote. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_returns_false_for_directory_without_init_but_has_other_src_extension. Retrieved 5/7 statements.


def test_case_0():
    var_0 = '/non/existent'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/empty/dir'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/dir/with/py'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/dir/with/config'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/dir/with/pyproject'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/dir/with/init'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/dir/with/namespace1'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/dir/with/namespace2'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/dir/with/namespace3'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/dir/with/namespace4'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/dir/with/other_ext'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = 'pyx'
    var_4 = [var_2, var_3]
    var_5 = frozenset(var_4)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 7/13 statements.


def test_case_0():
    var_0 = '/fake/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = 'mymodule'
    var_7 = [var_0]
    var_8 = ()
    var_9 = 'Found in one of the configured src_paths:'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_src_path_found_module_in_src_paths. Retrieved 4/6 statements.
# Partially parsed test_src_path_found_package_in_src_paths. Retrieved 4/6 statements.
# Partially parsed test_src_path_found_src_path_is_module. Retrieved 4/6 statements.
# Partially parsed test_src_path_namespace_package_with_nested_module. Retrieved 5/8 statements.
# Partially parsed test_src_path_auto_identify_namespace_packages. Retrieved 4/7 statements.
# Partially parsed test_src_path_not_found. Retrieved 4/6 statements.
# Partially parsed test_src_path_with_custom_src_paths. Retrieved 3/6 statements.
# Partially parsed test_src_path_with_prefix. Retrieved 6/8 statements.
# Partially parsed test_src_path_namespace_package_not_in_config. Retrieved 4/7 statements.
# Partially parsed test_src_path_auto_identify_namespace_packages_disabled. Retrieved 4/7 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/src'
    var_3 = [var_2]
    var_4 = 'mymodule'
    var_5 = module_1._src_path(var_4, var_1)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/src'
    var_3 = [var_2]
    var_4 = 'mypackage'
    var_5 = module_1._src_path(var_4, var_1)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/src'
    var_3 = [var_2]
    var_4 = 'src'
    var_5 = module_1._src_path(var_4, var_1)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/src'
    var_3 = [var_2]
    var_4 = 'namespace'
    var_5 = 'namespace.submodule'
    var_6 = module_1._src_path(var_5, var_1)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/src'
    var_3 = [var_2]
    var_4 = 'namespace.submodule'
    var_5 = module_1._src_path(var_4, var_1)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/src'
    var_3 = [var_2]
    var_4 = 'unknown'
    var_5 = module_1._src_path(var_4, var_1)
    assert var_5 is None

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/custom'
    var_3 = [var_2]
    var_4 = 'mymodule'

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/src'
    var_3 = [var_2]
    var_4 = 'submodule'
    var_5 = 'parent'
    var_6 = (var_5,)
    var_7 = module_1._src_path(var_4, var_1, prefix=var_6)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/src'
    var_3 = [var_2]
    var_4 = 'namespace.submodule'
    var_5 = module_1._src_path(var_4, var_1)
    assert var_5 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/src'
    var_3 = [var_2]
    var_4 = 'namespace.submodule'
    var_5 = module_1._src_path(var_4, var_1)
    assert var_5 is None



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_src_path_finds_module_in_src_paths. Retrieved 4/13 statements.
# Partially parsed test_src_path_returns_firstparty_on_module_match. Retrieved 4/14 statements.
# Partially parsed test_src_path_handles_nested_module_with_namespace. Retrieved 5/16 statements.
# Partially parsed test_src_path_auto_identifies_namespace_package. Retrieved 4/15 statements.
# Partially parsed test_src_path_src_path_is_module_match. Retrieved 4/14 statements.
# Partially parsed test_src_path_with_custom_src_paths. Retrieved 4/16 statements.
# Partially parsed test_src_path_with_prefix. Retrieved 6/16 statements.
# Partially parsed test_src_path_no_match_returns_none. Retrieved 4/16 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = 'mymodule'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = 'mymodule'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'mypackage'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = 'mypackage.submodule'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = 'mypackage.submodule'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = 'src'

def test_case_0():
    var_0 = '/custom'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = [var_0]
    var_5 = 'mymodule'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = 'submodule'
    var_5 = 'mypackage'
    var_6 = (var_5,)

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = 'unknown'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 1/6 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 1/6 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 1/6 statements.
# Partially parsed test_is_module_no_match. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'some_module'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'some_module'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'some_module'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'some_module'
    var_1 = [var_0]



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_namespace_not_in_config_and_auto_identify_false. Retrieved 7/10 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = set()
    var_1 = False
    var_2 = 'namespace_packages'
    var_3 = 'auto_identify_namespace_packages'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'a.b'
    var_7 = '/src'
    var_8 = [var_7]
    var_9 = 'a'
    var_10 = (var_9,)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_known_pattern_matches_configured_pattern. Retrieved 6/12 statements.
# Partially parsed test_known_pattern_no_match. Retrieved 5/11 statements.
# Partially parsed test_known_pattern_placement_not_in_sections. Retrieved 6/12 statements.
# Partially parsed test_known_pattern_matches_longest_module_prefix. Retrieved 7/15 statements.
# Partially parsed test_known_pattern_empty_name. Retrieved 5/11 statements.
# Partially parsed test_known_pattern_no_known_patterns. Retrieved 4/7 statements.
# Partially parsed test_known_pattern_no_sections. Retrieved 5/11 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'section1'
    var_3 = 'section2'
    var_4 = '^myapp\\.utils$'
    var_5 = 'myapp.utils.helpers'
    var_6 = module_1._known_pattern(var_5, var_1)
    var_7 = bool(var_6 == ('section1', "Matched configured known pattern re.compile('^myapp\\\\.utils$')"))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'section1'
    var_3 = '^other\\.module$'
    var_4 = 'myapp.utils'
    var_5 = module_1._known_pattern(var_4, var_1)
    assert var_5 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'section1'
    var_3 = '^myapp$'
    var_4 = 'section2'
    var_5 = 'myapp.utils'
    var_6 = module_1._known_pattern(var_5, var_1)
    assert var_6 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'section1'
    var_3 = 'section2'
    var_4 = '^myapp$'
    var_5 = '^myapp\\.utils$'
    var_6 = 'myapp.utils.helpers'
    var_7 = module_1._known_pattern(var_6, var_1)
    var_8 = bool(var_7 == ('section2', "Matched configured known pattern re.compile('^myapp\\\\.utils$')"))
    assert var_8 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'section1'
    var_3 = '^$'
    var_4 = ''
    var_5 = module_1._known_pattern(var_4, var_1)
    var_6 = bool(var_5 == ('section1', "Matched configured known pattern re.compile('^$')"))
    assert var_6 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'section1'
    var_3 = 'myapp.utils'
    var_4 = module_1._known_pattern(var_3, var_1)
    assert var_4 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '^myapp$'
    var_3 = 'section1'
    var_4 = 'myapp'
    var_5 = module_1._known_pattern(var_4, var_1)
    assert var_5 is None



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_src_path_is_module_true. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_known_pattern_matches_configured_pattern. Retrieved 9/20 statements.
# Partially parsed test_known_pattern_no_match. Retrieved 8/16 statements.
# Partially parsed test_known_pattern_section_not_in_sections. Retrieved 9/17 statements.
# Partially parsed test_known_pattern_matches_longest_module. Retrieved 11/24 statements.
# Partially parsed test_known_pattern_empty_name. Retrieved 9/20 statements.
# Partially parsed test_known_pattern_multiple_patterns_first_matches. Retrieved 11/24 statements.


def test_case_0():
    var_0 = 'Config'
    var_1 = 'known_patterns'
    var_2 = 'sections'
    var_3 = [var_1, var_2]
    var_4 = '^myapp\\.utils$'
    var_5 = 'THIRD_PARTY'
    var_6 = [var_5]
    var_7 = 'myapp.utils.logging'
    var_8 = 'Matched configured known pattern '

def test_case_0():
    var_0 = 'Config'
    var_1 = 'known_patterns'
    var_2 = 'sections'
    var_3 = [var_1, var_2]
    var_4 = '^other\\.module$'
    var_5 = 'THIRD_PARTY'
    var_6 = [var_5]
    var_7 = 'myapp.utils'

def test_case_0():
    var_0 = 'Config'
    var_1 = 'known_patterns'
    var_2 = 'sections'
    var_3 = [var_1, var_2]
    var_4 = '^myapp$'
    var_5 = 'THIRD_PARTY'
    var_6 = 'FIRST_PARTY'
    var_7 = [var_6]
    var_8 = 'myapp.utils'

def test_case_0():
    var_0 = 'Config'
    var_1 = 'known_patterns'
    var_2 = 'sections'
    var_3 = [var_1, var_2]
    var_4 = '^myapp$'
    var_5 = '^myapp\\.utils$'
    var_6 = 'FIRST_PARTY'
    var_7 = 'THIRD_PARTY'
    var_8 = [var_6, var_7]
    var_9 = 'myapp.utils.logging'
    var_10 = 'Matched configured known pattern '

def test_case_0():
    var_0 = 'Config'
    var_1 = 'known_patterns'
    var_2 = 'sections'
    var_3 = [var_1, var_2]
    var_4 = '^$'
    var_5 = 'FIRST_PARTY'
    var_6 = [var_5]
    var_7 = ''
    var_8 = 'Matched configured known pattern '

def test_case_0():
    var_0 = 'Config'
    var_1 = 'known_patterns'
    var_2 = 'sections'
    var_3 = [var_1, var_2]
    var_4 = '^myapp\\.utils$'
    var_5 = '^myapp\\.utils\\.logging$'
    var_6 = 'FIRST_PARTY'
    var_7 = 'THIRD_PARTY'
    var_8 = [var_6, var_7]
    var_9 = 'myapp.utils.logging'
    var_10 = 'Matched configured known pattern '



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_namespace_in_config_namespace_packages. Retrieved 9/15 statements.
# Partially parsed test_auto_identify_namespace_packages_true_and_is_namespace_package. Retrieved 11/18 statements.
# Partially parsed test_nested_module_true_and_namespace_in_config_namespace_packages. Retrieved 9/15 statements.
# Partially parsed test_nested_module_true_and_auto_identify_true_and_is_namespace_package. Retrieved 10/16 statements.


def test_case_0():
    var_0 = 'test.namespace'
    var_1 = {var_0}
    var_2 = False
    var_3 = '/src'
    var_4 = [var_3]
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = 'test.namespace.module'
    var_8 = [var_3]
    var_9 = 'test'
    var_10 = (var_9,)

def test_case_0():
    var_0 = set()
    var_1 = True
    var_2 = '/src'
    var_3 = [var_2]
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = '/src/test/namespace'
    var_7 = [var_6]
    var_8 = 'test.namespace.module'
    var_9 = '/src'
    var_10 = [var_9]
    var_11 = [var_2]
    var_12 = 'test'
    var_13 = (var_12,)

def test_case_0():
    var_0 = 'a.b'
    var_1 = {var_0}
    var_2 = False
    var_3 = '/src'
    var_4 = [var_3]
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = 'a.b.c'
    var_8 = [var_3]
    var_9 = 'a'
    var_10 = (var_9,)

def test_case_0():
    var_0 = set()
    var_1 = True
    var_2 = '/src'
    var_3 = [var_2]
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = 'x.y.z'
    var_7 = '/src'
    var_8 = [var_7]
    var_9 = [var_2]
    var_10 = 'x'
    var_11 = (var_10,)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_is_module_with_py_extension. Retrieved 6/13 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 9/18 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'some_module'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = '__main__'
    var_4 = __import__(var_3)
    var_5 = var_4.exists_case_sensitive
    var_6 = __import__(var_3)

def test_case_0():
    var_0 = 'some_module'
    var_1 = [var_0]
    var_2 = '.so'
    var_3 = '__main__'
    var_4 = __import__(var_3)
    var_5 = var_4.exists_case_sensitive
    var_6 = 'importlib'
    var_7 = __import__(var_6)
    var_8 = var_7.machinery.EXTENSION_SUFFIXES
    var_9 = __import__(var_3)

def test_case_0():
    var_0 = 'some_module'
    var_1 = [var_0]
    var_2 = '__init__.py'
    var_3 = '__main__'
    var_4 = __import__(var_3)
    var_5 = var_4.exists_case_sensitive
    var_6 = __import__(var_3)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_namespace_package_without_init_and_no_files. Retrieved 4/11 statements.
# Partially parsed test_namespace_package_without_init_and_no_src_files. Retrieved 5/14 statements.
# Partially parsed test_namespace_package_with_init_and_namespace_declaration. Retrieved 6/15 statements.
# Partially parsed test_namespace_package_with_init_and_namespace_declaration_double_quotes. Retrieved 6/15 statements.
# Partially parsed test_namespace_package_with_init_and_extend_path. Retrieved 6/15 statements.
# Partially parsed test_namespace_package_with_init_and_extend_path_double_quotes. Retrieved 6/15 statements.


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'README.txt'
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = b"__import__('pkg_resources').declare_namespace(__name__)"
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = b'__import__("pkg_resources").declare_namespace(__name__)'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 1/7 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 2/12 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 2/10 statements.
# Partially parsed test_is_module_no_match. Retrieved 1/7 statements.
# Partially parsed test_is_module_case_sensitive_check. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'some_module'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'some_module'
    var_1 = [var_0]
    var_2 = 0

def test_case_0():
    var_0 = 'some_module'
    var_1 = [var_0]
    var_2 = '__init__.py'

def test_case_0():
    var_0 = 'some_module'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'some_module'
    var_1 = [var_0]



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_namespace_in_config_namespace_packages_false. Retrieved 8/14 statements.


def test_case_0():
    var_0 = set()
    var_1 = False
    var_2 = '/test'
    var_3 = [var_2]
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = 'module'
    var_7 = [var_2]
    var_8 = 'existing'
    var_9 = (var_8,)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_src_path_is_module_true. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_namespace_not_in_config_and_auto_identify_false. Retrieved 7/10 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = set()
    var_1 = False
    var_2 = 'namespace_packages'
    var_3 = 'auto_identify_namespace_packages'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'a.b'
    var_7 = '/src'
    var_8 = [var_7]
    var_9 = 'a'
    var_10 = (var_9,)



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_src_path_finds_module_in_src_paths. Retrieved 5/9 statements.
# Partially parsed test_src_path_returns_none_for_missing_module. Retrieved 5/9 statements.
# Partially parsed test_src_path_handles_nested_module_in_namespace_package. Retrieved 6/12 statements.
# Partially parsed test_src_path_handles_auto_identified_namespace_package. Retrieved 5/11 statements.
# Partially parsed test_src_path_handles_root_module_matching_src_path_name. Retrieved 5/9 statements.
# Partially parsed test_src_path_uses_provided_src_paths_parameter. Retrieved 7/10 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = frozenset()
    var_5 = 'mymodule'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = frozenset()
    var_5 = 'missing'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'mypackage'
    var_3 = {var_2}
    var_4 = False
    var_5 = frozenset()
    var_6 = 'mypackage.nested'
    var_7 = [var_0]

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = True
    var_4 = frozenset()
    var_5 = 'namespace.nested'
    var_6 = [var_0]

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = frozenset()
    var_5 = 'src'

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = set()
    var_2 = False
    var_3 = frozenset()
    var_4 = 'src_paths'
    var_5 = 'namespace_packages'
    var_6 = 'auto_identify_namespace_packages'
    var_7 = 'supported_extensions'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_2, var_7: var_3}
    var_9 = module_0.Config(**var_8)
    var_10 = 'mymodule'
    var_11 = '/custom'
    var_12 = [var_11]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test__src_path_finds_module_in_src_paths. Retrieved 4/13 statements.
# Partially parsed test__src_path_returns_firstparty_on_module_match. Retrieved 4/15 statements.
# Partially parsed test__src_path_handles_nested_module_with_namespace_package. Retrieved 5/17 statements.
# Partially parsed test__src_path_handles_src_path_is_module_case. Retrieved 4/15 statements.
# Partially parsed test__src_path_with_custom_src_paths_argument. Retrieved 5/18 statements.
# Partially parsed test__src_path_with_prefix_argument. Retrieved 6/17 statements.
# Partially parsed test__src_path_returns_none_when_no_match. Retrieved 4/14 statements.
# Partially parsed test__src_path_handles_auto_identify_namespace_packages. Retrieved 4/16 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = 'mymodule'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = 'mymodule'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'mypackage'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = 'mypackage.nested'

def test_case_0():
    var_0 = '/src/mymodule'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = 'mymodule'

def test_case_0():
    var_0 = '/default'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = '/custom'
    var_5 = [var_4]
    var_6 = 'mymodule'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = 'nested'
    var_5 = 'mypackage'
    var_6 = (var_5,)

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = 'unknown'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = 'mypackage.nested'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_known_pattern_matches_configured_pattern. Retrieved 6/12 statements.
# Partially parsed test_known_pattern_no_match. Retrieved 5/11 statements.
# Partially parsed test_known_pattern_placement_not_in_sections. Retrieved 6/12 statements.
# Partially parsed test_known_pattern_matches_longest_module_prefix. Retrieved 7/15 statements.
# Partially parsed test_known_pattern_empty_name. Retrieved 5/11 statements.
# Partially parsed test_known_pattern_no_known_patterns. Retrieved 4/7 statements.
# Partially parsed test_known_pattern_no_sections. Retrieved 5/11 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'section1'
    var_3 = 'section2'
    var_4 = '^myapp\\.utils$'
    var_5 = 'myapp.utils.helpers'
    var_6 = module_1._known_pattern(var_5, var_1)
    var_7 = bool(var_6 == ('section1', "Matched configured known pattern re.compile('^myapp\\\\.utils$')"))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'section1'
    var_3 = '^other\\.module$'
    var_4 = 'myapp.utils'
    var_5 = module_1._known_pattern(var_4, var_1)
    assert var_5 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'section1'
    var_3 = '^myapp$'
    var_4 = 'section2'
    var_5 = 'myapp.utils'
    var_6 = module_1._known_pattern(var_5, var_1)
    assert var_6 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'section1'
    var_3 = 'section2'
    var_4 = '^myapp$'
    var_5 = '^myapp\\.utils$'
    var_6 = 'myapp.utils.helpers'
    var_7 = module_1._known_pattern(var_6, var_1)
    var_8 = bool(var_7 == ('section2', "Matched configured known pattern re.compile('^myapp\\\\.utils$')"))
    assert var_8 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'section1'
    var_3 = '^$'
    var_4 = ''
    var_5 = module_1._known_pattern(var_4, var_1)
    var_6 = bool(var_5 == ('section1', "Matched configured known pattern re.compile('^$')"))
    assert var_6 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'section1'
    var_3 = 'myapp.utils'
    var_4 = module_1._known_pattern(var_3, var_1)
    assert var_4 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '^myapp$'
    var_3 = 'section1'
    var_4 = 'myapp'
    var_5 = module_1._known_pattern(var_4, var_1)
    assert var_5 is None



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 1/7 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 1/8 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 1/7 statements.
# Partially parsed test_is_module_no_match. Retrieved 2/10 statements.
# Partially parsed test_is_module_first_condition_true_short_circuit. Retrieved 1/7 statements.
# Partially parsed test_is_module_second_condition_true. Retrieved 3/12 statements.
# Partially parsed test_is_module_third_condition_true. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'test.py'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'test.so'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'test/__init__.py'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'test.txt'
    var_1 = [var_0]
    var_2 = 'test/__init__.txt'
    var_3 = [var_2]

def test_case_0():
    var_0 = 'test.py'
    var_1 = [var_0]

def test_case_0():
    var_0 = '.so'
    var_1 = 'test.so'
    var_2 = [var_1]
    var_3 = 'test.py'
    var_4 = [var_3]

def test_case_0():
    var_0 = 'test.py'
    var_1 = [var_0]
    var_2 = 'test/__init__.py'
    var_3 = [var_2]



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 2/11 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 1/13 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 1/8 statements.
# Partially parsed test_is_module_no_match. Retrieved 2/10 statements.
# Partially parsed test_is_module_case_sensitive. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'module.py'
    var_1 = 'module'

def test_case_0():
    var_0 = 'module'

def test_case_0():
    var_0 = '__init__.py'

def test_case_0():
    var_0 = 'module.txt'
    var_1 = 'module'

def test_case_0():
    var_0 = 'Module.py'
    var_1 = 'module'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_known_pattern_predicate_false. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'section1'
    var_1 = [var_0]
    var_2 = 'test.module.name'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_is_namespace_package_with_nonexistent_path. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_file_path. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_directory_no_init_and_no_files. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_directory_no_init_but_has_py_file. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_directory_no_init_but_has_setup_cfg. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_directory_no_init_but_has_pyproject_toml. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_init_containing_pkg_resources_single_quote. Retrieved 5/7 statements.
# Partially parsed test_is_namespace_package_with_init_containing_pkg_resources_double_quote. Retrieved 5/7 statements.
# Partially parsed test_is_namespace_package_with_init_containing_pkgutil_single_quote. Retrieved 5/7 statements.
# Partially parsed test_is_namespace_package_with_init_containing_pkgutil_double_quote. Retrieved 5/7 statements.
# Partially parsed test_is_namespace_package_with_init_not_containing_namespace_markers. Retrieved 5/7 statements.
# Partially parsed test_is_namespace_package_with_init_containing_marker_beyond_4096_bytes. Retrieved 9/11 statements.


def test_case_0():
    var_0 = '/nonexistent'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/some/file.txt'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/empty/dir'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/dir/with/py'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/dir/with/setup'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/dir/with/pyproject'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/dir/with/init'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)
    var_5 = b"__import__('pkg_resources').declare_namespace(__name__)"

def test_case_0():
    var_0 = '/dir/with/init'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)
    var_5 = b'__import__("pkg_resources").declare_namespace(__name__)'

def test_case_0():
    var_0 = '/dir/with/init'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)
    var_5 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"

def test_case_0():
    var_0 = '/dir/with/init'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)
    var_5 = b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)'

def test_case_0():
    var_0 = '/dir/with/init'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)
    var_5 = b"print('hello')"

def test_case_0():
    var_0 = '/dir/with/init'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)
    var_5 = b'a'
    var_6 = 4096
    var_7 = var_5 * var_6
    var_8 = b"__import__('pkg_resources').declare_namespace(__name__)"
    var_9 = var_7 + var_8



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_namespace_in_config_namespace_packages. Retrieved 9/15 statements.


def test_case_0():
    var_0 = 'test.namespace'
    var_1 = {var_0}
    var_2 = False
    var_3 = '/src'
    var_4 = [var_3]
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = 'test.namespace.module'
    var_8 = [var_3]
    var_9 = 'test'
    var_10 = (var_9,)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_src_path_is_module_true_for_matching_dir. Retrieved 1/5 statements.
# Partially parsed test_src_path_is_module_false_for_wrong_name. Retrieved 2/6 statements.
# Partially parsed test_src_path_is_module_false_for_file. Retrieved 1/5 statements.
# Partially parsed test_src_path_is_module_false_for_case_mismatch. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]
    var_2 = 'wrong_name'

def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'Test_Module'
    var_1 = [var_0]
    var_2 = 'test_module'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 1/6 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 2/8 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 1/6 statements.
# Partially parsed test_is_module_no_match. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'some_module'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'some_module'
    var_1 = [var_0]
    var_2 = '.so'

def test_case_0():
    var_0 = 'some_module'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'some_module'
    var_1 = [var_0]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_known_pattern_predicate_false. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'section1'
    var_1 = [var_0]
    var_2 = 'test.module'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_predicate_at_line_16_true. Retrieved 21/27 statements.


def test_case_0():
    var_0 = []
    var_1 = 'Path'
    var_2 = ()
    var_3 = 'name'
    var_4 = 'resolve'
    var_5 = 'root_module'
    var_6 = lambda self: self
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = type(var_1, var_2, var_7)
    var_9 = var_8()
    var_10 = ()
    var_11 = 'is_dir'
    var_12 = False
    var_13 = lambda : var_12
    var_14 = {var_11: var_13}
    var_15 = type(var_1, var_10, var_14)
    var_16 = var_15()
    var_17 = ()
    var_18 = 'root_module'
    var_19 = var_9.name
    var_20 = var_19 == var_18



# Parsed testcases at query #13
#--------------------------

# Failed to parse test_predicate_at_line_26_true.




# Parsed testcases at query #14
#--------------------------

# Partially parsed test_predicate_at_line_26_true. Retrieved 11/22 statements.


def test_case_0():
    var_0 = '/test/path'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = [var_0]
    var_7 = 'module'
    var_8 = ()
    var_9 = '.'
    var_10 = *var_8
    var_11 = (var_10, var_7)
    var_12 = []



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_src_path_finds_module_in_src_paths. Retrieved 4/6 statements.
# Partially parsed test_src_path_returns_none_for_missing_module. Retrieved 4/6 statements.
# Partially parsed test_src_path_handles_nested_module_in_namespace_package. Retrieved 6/9 statements.
# Partially parsed test_src_path_handles_auto_identify_namespace_packages. Retrieved 6/9 statements.
# Partially parsed test_src_path_uses_provided_src_paths. Retrieved 3/6 statements.
# Partially parsed test_src_path_handles_root_module_matching_src_path_name. Retrieved 4/6 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/tmp/test_src'
    var_3 = [var_2]
    var_4 = 'mymodule'
    var_5 = module_1._src_path(var_4, var_1)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/tmp/test_src'
    var_3 = [var_2]
    var_4 = 'missingmodule'
    var_5 = module_1._src_path(var_4, var_1)
    assert var_5 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/tmp/test_src'
    var_3 = [var_2]
    var_4 = 'namespace'
    var_5 = 'namespace.submodule'
    var_6 = (var_4,)
    var_7 = module_1._src_path(var_5, var_1, prefix=var_6)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/tmp/test_src'
    var_3 = [var_2]
    var_4 = 'namespace.submodule'
    var_5 = 'namespace'
    var_6 = (var_5,)
    var_7 = module_1._src_path(var_4, var_1, prefix=var_6)

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/custom/path'
    var_3 = [var_2]
    var_4 = 'mymodule'

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/tmp/test_src'
    var_3 = [var_2]
    var_4 = 'test_src'
    var_5 = module_1._src_path(var_4, var_1)



# Parsed testcases at query #16
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'some.namespace'
    var_1 = {var_0}
    var_2 = False
    var_3 = []
    var_4 = []
    var_5 = 'namespace_packages'
    var_6 = 'auto_identify_namespace_packages'
    var_7 = 'src_paths'
    var_8 = 'supported_extensions'
    var_9 = {var_5: var_1, var_6: var_2, var_7: var_3, var_8: var_4}
    var_10 = module_0.Config(**var_9)
    var_11 = 'some.namespace.module'
    var_12 = []
    var_13 = 'some'
    var_14 = (var_13,)
    var_15 = module_1._src_path(var_11, var_10, var_12, var_14)
    var_16 = bool(var_15 is not None)
    assert var_16 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_namespace_package_without_init_and_no_files. Retrieved 4/11 statements.
# Partially parsed test_namespace_package_without_init_and_no_source_files. Retrieved 5/14 statements.
# Partially parsed test_namespace_package_with_init_containing_pkg_resources_declare_namespace_single_quotes. Retrieved 6/15 statements.
# Partially parsed test_namespace_package_with_init_containing_pkg_resources_declare_namespace_double_quotes. Retrieved 6/15 statements.
# Partially parsed test_namespace_package_with_init_containing_pkgutil_extend_path_single_quotes. Retrieved 6/15 statements.
# Partially parsed test_namespace_package_with_init_containing_pkgutil_extend_path_double_quotes. Retrieved 6/15 statements.


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'README.txt'
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = "__import__('pkg_resources').declare_namespace(__name__)"
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '__import__("pkg_resources").declare_namespace(__name__)'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = "__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_namespace_not_in_config_and_auto_identify_false. Retrieved 7/10 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = set()
    var_1 = False
    var_2 = 'namespace_packages'
    var_3 = 'auto_identify_namespace_packages'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'a.b'
    var_7 = '/test'
    var_8 = [var_7]
    var_9 = 'a'
    var_10 = (var_9,)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 6/16 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = [var_0]
    var_7 = 'mymodule'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_src_path_predicate_at_line_26_true. Retrieved 7/13 statements.


def test_case_0():
    var_0 = '/test/path'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = 'module'
    var_7 = [var_0]
    var_8 = ()
    var_9 = 'Found in one of the configured src_paths:'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 2/11 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 1/13 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 2/12 statements.
# Partially parsed test_is_module_without_any. Retrieved 1/8 statements.
# Partially parsed test_is_module_case_sensitive_py. Retrieved 3/14 statements.
# Partially parsed test_is_module_case_sensitive_extension. Retrieved 3/16 statements.
# Partially parsed test_is_module_case_sensitive_init. Retrieved 3/15 statements.


def test_case_0():
    var_0 = 'module'
    var_1 = '.py'

def test_case_0():
    var_0 = 'module'

def test_case_0():
    var_0 = 'module'
    var_1 = '__init__.py'

def test_case_0():
    var_0 = 'module'

def test_case_0():
    var_0 = 'Module'
    var_1 = '.py'
    var_2 = 'module'

def test_case_0():
    var_0 = 'Module'
    var_1 = 0
    var_2 = 'module'

def test_case_0():
    var_0 = 'Module'
    var_1 = '__init__.py'
    var_2 = 'module'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 2/11 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 1/14 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 2/12 statements.
# Partially parsed test_is_module_no_files. Retrieved 1/7 statements.
# Partially parsed test_is_module_case_sensitive_mismatch. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'module'
    var_1 = '.py'

def test_case_0():
    var_0 = 'module'

def test_case_0():
    var_0 = 'package'
    var_1 = '__init__.py'

def test_case_0():
    var_0 = 'nonexistent'

def test_case_0():
    var_0 = 'Module'
    var_1 = 'module.py'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_predicate_at_line_26_true. Retrieved 18/20 statements.


import isort.place as module_0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = False
    var_3 = []
    var_4 = 'Path'
    var_5 = ()
    var_6 = 'resolve'
    var_7 = 'is_dir'
    var_8 = 'name'
    var_9 = lambda self: self
    var_10 = False
    var_11 = lambda : var_10
    var_12 = 'mymodule'
    var_13 = {var_6: var_9, var_7: var_11, var_8: var_12}
    var_14 = type(var_4, var_5, var_13)
    var_15 = var_14()
    var_16 = 'mymodule'
    var_17 = module_0._src_path_is_module(var_15, var_16)
    assert var_17 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_src_path_finds_module_in_src_paths. Retrieved 5/16 statements.
# Partially parsed test_src_path_finds_package_in_src_paths. Retrieved 6/19 statements.
# Partially parsed test_src_path_finds_root_module_as_src_path. Retrieved 5/16 statements.
# Partially parsed test_src_path_handles_nested_module_in_namespace_package. Retrieved 8/23 statements.
# Partially parsed test_src_path_returns_none_when_module_not_found. Retrieved 4/13 statements.
# Partially parsed test_src_path_uses_provided_src_paths. Retrieved 5/16 statements.
# Partially parsed test_src_path_handles_auto_identified_namespace_package. Retrieved 8/25 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = '/src/module.py'
    var_5 = [var_4]
    var_6 = 'module'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = '/src/package'
    var_5 = [var_4]
    var_6 = '__init__.py'
    var_7 = 'package'

def test_case_0():
    var_0 = '/src/module'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = [var_0]
    var_5 = True
    var_6 = 'module'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'namespace'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = '/src/namespace'
    var_6 = [var_5]
    var_7 = 'nested'
    var_8 = '__init__.py'
    var_9 = 'namespace.nested'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = 'nonexistent'

def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = '/custom'
    var_3 = [var_2]
    var_4 = '/custom/module.py'
    var_5 = [var_4]
    var_6 = 'module'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = '/src/namespace'
    var_5 = [var_4]
    var_6 = '__init__.py'
    var_7 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_8 = 'nested'
    var_9 = 'namespace.nested'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_namespace_in_config_namespace_packages. Retrieved 11/14 statements.
# Partially parsed test_auto_identify_namespace_packages_true_and_is_namespace_package. Retrieved 11/15 statements.
# Partially parsed test_nested_module_true_and_namespace_in_config_namespace_packages. Retrieved 11/14 statements.
# Partially parsed test_nested_module_true_and_auto_identify_and_is_namespace_package. Retrieved 11/15 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'some.namespace'
    var_1 = {var_0}
    var_2 = False
    var_3 = []
    var_4 = set()
    var_5 = 'namespace_packages'
    var_6 = 'auto_identify_namespace_packages'
    var_7 = 'src_paths'
    var_8 = 'supported_extensions'
    var_9 = {var_5: var_1, var_6: var_2, var_7: var_3, var_8: var_4}
    var_10 = module_0.Config(**var_9)
    var_11 = '/some/path'
    var_12 = [var_11]
    var_13 = 'some.namespace.module'
    var_14 = 'some'
    var_15 = 'namespace'
    var_16 = (var_14, var_15)

import isort.settings as module_0

def test_case_0():
    var_0 = set()
    var_1 = True
    var_2 = []
    var_3 = '.py'
    var_4 = {var_3}
    var_5 = 'namespace_packages'
    var_6 = 'auto_identify_namespace_packages'
    var_7 = 'src_paths'
    var_8 = 'supported_extensions'
    var_9 = {var_5: var_0, var_6: var_1, var_7: var_2, var_8: var_4}
    var_10 = module_0.Config(**var_9)
    var_11 = '/some/path'
    var_12 = [var_11]
    var_13 = 'some.namespace.module'
    var_14 = 'some'
    var_15 = 'namespace'
    var_16 = (var_14, var_15)

import isort.settings as module_0

def test_case_0():
    var_0 = 'some.namespace'
    var_1 = {var_0}
    var_2 = False
    var_3 = []
    var_4 = set()
    var_5 = 'namespace_packages'
    var_6 = 'auto_identify_namespace_packages'
    var_7 = 'src_paths'
    var_8 = 'supported_extensions'
    var_9 = {var_5: var_1, var_6: var_2, var_7: var_3, var_8: var_4}
    var_10 = module_0.Config(**var_9)
    var_11 = '/some/path'
    var_12 = [var_11]
    var_13 = 'some.namespace.module'
    var_14 = 'some'
    var_15 = 'namespace'
    var_16 = (var_14, var_15)

import isort.settings as module_0

def test_case_0():
    var_0 = set()
    var_1 = True
    var_2 = []
    var_3 = '.py'
    var_4 = {var_3}
    var_5 = 'namespace_packages'
    var_6 = 'auto_identify_namespace_packages'
    var_7 = 'src_paths'
    var_8 = 'supported_extensions'
    var_9 = {var_5: var_0, var_6: var_1, var_7: var_2, var_8: var_4}
    var_10 = module_0.Config(**var_9)
    var_11 = '/some/path'
    var_12 = [var_11]
    var_13 = 'some.namespace.module'
    var_14 = 'some'
    var_15 = 'namespace'
    var_16 = (var_14, var_15)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_src_path_finds_module_in_src_paths. Retrieved 7/11 statements.
# Partially parsed test_src_path_returns_none_for_missing_module. Retrieved 7/11 statements.
# Partially parsed test_src_path_handles_nested_module_in_namespace_package. Retrieved 8/12 statements.
# Partially parsed test_src_path_auto_identifies_namespace_package. Retrieved 7/11 statements.
# Partially parsed test_src_path_uses_provided_src_paths. Retrieved 9/12 statements.
# Partially parsed test_src_path_handles_root_module_matching_src_path_name. Retrieved 7/11 statements.
# Partially parsed test_src_path_with_prefix. Retrieved 10/16 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'mymodule'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'missing'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'mypackage'
    var_3 = {var_2}
    var_4 = False
    var_5 = 'py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)
    var_8 = 'mypackage.nested'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = True
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'namespace.nested'

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = set()
    var_2 = False
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)
    var_6 = 'src_paths'
    var_7 = 'namespace_packages'
    var_8 = 'auto_identify_namespace_packages'
    var_9 = 'supported_extensions'
    var_10 = {var_6: var_0, var_7: var_1, var_8: var_2, var_9: var_5}
    var_11 = module_0.Config(**var_10)
    var_12 = '/custom'
    var_13 = [var_12]
    var_14 = 'mymodule'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'src'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'nested'
    var_8 = '/src/mypackage'
    var_9 = [var_8]
    var_10 = 'mypackage'
    var_11 = (var_10,)



# Parsed testcases at query #27
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'foo'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = module_1._forced_separate(var_0, var_4)
    var_6 = bool(var_5 == ('foo', 'Matched forced_separate (foo) config value.'))
    assert var_6 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'foo*'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'foobar'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('foo*', 'Matched forced_separate (foo*) config value.'))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'foo'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '.foo'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('foo', 'Matched forced_separate (foo) config value.'))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'foo*'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '.foobar'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('foo*', 'Matched forced_separate (foo*) config value.'))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'foo'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'bar'
    var_6 = module_1._forced_separate(var_5, var_4)
    assert var_6 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'foo*'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'bar'
    var_6 = module_1._forced_separate(var_5, var_4)
    assert var_6 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = 'forced_separate'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'foo'
    var_5 = module_1._forced_separate(var_4, var_3)
    assert var_5 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'foo'
    var_1 = 'bar'
    var_2 = [var_0, var_1]
    var_3 = 'forced_separate'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = module_1._forced_separate(var_0, var_5)
    var_7 = bool(var_6 == ('foo', 'Matched forced_separate (foo) config value.'))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'foo'
    var_1 = 'bar'
    var_2 = [var_0, var_1]
    var_3 = 'forced_separate'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = module_1._forced_separate(var_1, var_5)
    var_7 = bool(var_6 == ('bar', 'Matched forced_separate (bar) config value.'))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'foo'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'foobar'
    var_6 = module_1._forced_separate(var_5, var_4)
    assert var_6 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'foo*'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'foo'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('foo*', 'Matched forced_separate (foo*) config value.'))
    assert var_7 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test__src_path_with_exact_module_match. Retrieved 4/6 statements.
# Partially parsed test__src_path_with_nested_module_in_namespace_package. Retrieved 5/8 statements.
# Partially parsed test__src_path_with_auto_identify_namespace_packages_enabled. Retrieved 6/10 statements.
# Partially parsed test__src_path_no_match. Retrieved 4/6 statements.
# Partially parsed test__src_path_with_custom_src_paths. Retrieved 3/6 statements.
# Partially parsed test__src_path_with_prefix. Retrieved 7/9 statements.
# Partially parsed test__src_path_with_src_path_is_module. Retrieved 4/6 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/test/path'
    var_3 = [var_2]
    var_4 = 'module'
    var_5 = module_1._src_path(var_4, var_1)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/test/path'
    var_3 = [var_2]
    var_4 = 'namespace'
    var_5 = 'namespace.nested'
    var_6 = module_1._src_path(var_5, var_1)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/test/path'
    var_3 = [var_2]
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = 'namespace.nested'
    var_7 = module_1._src_path(var_6, var_1)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/test/path'
    var_3 = [var_2]
    var_4 = 'unknown'
    var_5 = module_1._src_path(var_4, var_1)
    assert var_5 is None

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/custom/path'
    var_3 = [var_2]
    var_4 = 'module'

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/test/path'
    var_3 = [var_2]
    var_4 = 'pre'
    var_5 = 'fix'
    var_6 = (var_4, var_5)
    var_7 = 'module'
    var_8 = module_1._src_path(var_7, var_1, prefix=var_6)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/test/path'
    var_3 = [var_2]
    var_4 = 'path'
    var_5 = module_1._src_path(var_4, var_1)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_src_path_is_module_true_for_valid_dir. Retrieved 1/5 statements.
# Partially parsed test_src_path_is_module_false_for_wrong_name. Retrieved 2/6 statements.
# Partially parsed test_src_path_is_module_false_for_file. Retrieved 1/5 statements.
# Partially parsed test_src_path_is_module_false_for_case_mismatch. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]
    var_2 = 'wrong_name'

def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'Test_Module'
    var_1 = [var_0]
    var_2 = 'test_module'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_known_pattern_predicate_false. Retrieved 3/5 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.module'
    var_3 = module_1._known_pattern(var_2, var_1)
    assert var_3 is None



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_known_pattern_predicate_false. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'section1'
    var_1 = [var_0]
    var_2 = 'test.module.name'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_namespace_package_without_init_and_no_files. Retrieved 5/11 statements.
# Partially parsed test_namespace_package_without_init_and_no_source_files. Retrieved 7/17 statements.
# Partially parsed test_namespace_package_with_init_and_namespace_declaration. Retrieved 5/14 statements.
# Partially parsed test_namespace_package_with_init_and_namespace_declaration_double_quotes. Retrieved 5/14 statements.
# Partially parsed test_namespace_package_with_init_and_pkgutil_extend_path. Retrieved 5/14 statements.
# Partially parsed test_namespace_package_with_init_and_pkgutil_extend_path_double_quotes. Retrieved 5/14 statements.


def test_case_0():
    var_0 = False
    var_1 = lambda : var_0
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = False
    var_1 = lambda : var_0
    var_2 = '.txt'
    var_3 = 'file.txt'
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = True
    var_1 = lambda : var_0
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = True
    var_1 = lambda : var_0
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = True
    var_1 = lambda : var_0
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = True
    var_1 = lambda : var_0
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)



# Parsed testcases at query #33
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'foo'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = module_1._forced_separate(var_0, var_4)
    var_6 = bool(var_5 == ('foo', 'Matched forced_separate (foo) config value.'))
    assert var_6 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'foo*'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'foobar'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('foo*', 'Matched forced_separate (foo*) config value.'))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'foo'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'foobar'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('foo', 'Matched forced_separate (foo) config value.'))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'foo'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '.foo'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('foo', 'Matched forced_separate (foo) config value.'))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'foo*'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '.foobar'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('foo*', 'Matched forced_separate (foo*) config value.'))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'foo'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'bar'
    var_6 = module_1._forced_separate(var_5, var_4)
    assert var_6 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = 'forced_separate'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'foo'
    var_5 = module_1._forced_separate(var_4, var_3)
    assert var_5 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'bar'
    var_1 = 'foo'
    var_2 = 'baz'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'forced_separate'
    var_5 = {var_4: var_3}
    var_6 = module_0.Config(**var_5)
    var_7 = module_1._forced_separate(var_1, var_6)
    var_8 = bool(var_7 == ('foo', 'Matched forced_separate (foo) config value.'))
    assert var_8 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'bar'
    var_1 = 'foo*'
    var_2 = 'baz'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'forced_separate'
    var_5 = {var_4: var_3}
    var_6 = module_0.Config(**var_5)
    var_7 = 'foobar'
    var_8 = module_1._forced_separate(var_7, var_6)
    var_9 = bool(var_8 == ('foo*', 'Matched forced_separate (foo*) config value.'))
    assert var_9 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'foo/bar'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = module_1._forced_separate(var_0, var_4)
    var_6 = bool(var_5 == ('foo/bar', 'Matched forced_separate (foo/bar) config value.'))
    assert var_6 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'foo/*'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'foo/bar'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('foo/*', 'Matched forced_separate (foo/*) config value.'))
    assert var_7 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_namespace_package_without_init_and_no_files. Retrieved 4/11 statements.
# Partially parsed test_namespace_package_without_init_and_no_src_files. Retrieved 5/14 statements.
# Partially parsed test_namespace_package_without_init_and_no_matching_extensions. Retrieved 5/14 statements.
# Partially parsed test_namespace_package_without_init_and_setup_cfg. Retrieved 5/14 statements.
# Partially parsed test_namespace_package_without_init_and_pyproject_toml. Retrieved 5/14 statements.
# Partially parsed test_namespace_package_without_init_and_py_file. Retrieved 5/14 statements.
# Partially parsed test_namespace_package_with_init_and_namespace_declaration_single_quotes. Retrieved 6/15 statements.
# Partially parsed test_namespace_package_with_init_and_namespace_declaration_double_quotes. Retrieved 6/15 statements.
# Partially parsed test_namespace_package_with_init_and_extend_path_single_quotes. Retrieved 6/15 statements.
# Partially parsed test_namespace_package_with_init_and_extend_path_double_quotes. Retrieved 6/15 statements.
# Partially parsed test_namespace_package_with_init_and_no_namespace_declaration. Retrieved 6/15 statements.
# Partially parsed test_not_a_package. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'README.txt'
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'file.txt'
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'setup.cfg'
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'pyproject.toml'
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'module.py'
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = "__import__('pkg_resources').declare_namespace(__name__)"
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '__import__("pkg_resources").declare_namespace(__name__)'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = "__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = ''
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_namespace_in_config_namespace_packages. Retrieved 9/15 statements.
# Partially parsed test_auto_identify_namespace_packages_true_and_is_namespace_package. Retrieved 10/16 statements.
# Partially parsed test_nested_module_true_and_namespace_in_config_namespace_packages. Retrieved 9/15 statements.
# Partially parsed test_nested_module_true_and_auto_identify_true_and_is_namespace_package. Retrieved 10/16 statements.


def test_case_0():
    var_0 = 'some.namespace'
    var_1 = {var_0}
    var_2 = False
    var_3 = '/src'
    var_4 = [var_3]
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = 'some.namespace.module'
    var_8 = [var_3]
    var_9 = 'some'
    var_10 = (var_9,)

def test_case_0():
    var_0 = set()
    var_1 = True
    var_2 = '/src'
    var_3 = [var_2]
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = 'some.namespace.module'
    var_7 = '/src'
    var_8 = [var_7]
    var_9 = [var_2]
    var_10 = 'some'
    var_11 = (var_10,)

def test_case_0():
    var_0 = 'some.namespace'
    var_1 = {var_0}
    var_2 = False
    var_3 = '/src'
    var_4 = [var_3]
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = 'some.namespace.module'
    var_8 = [var_3]
    var_9 = 'some'
    var_10 = (var_9,)

def test_case_0():
    var_0 = set()
    var_1 = True
    var_2 = '/src'
    var_3 = [var_2]
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = 'some.namespace.module'
    var_7 = '/src'
    var_8 = [var_7]
    var_9 = [var_2]
    var_10 = 'some'
    var_11 = (var_10,)



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_namespace_not_in_config_and_auto_identify_false. Retrieved 10/13 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = set()
    var_1 = False
    var_2 = 'namespace_packages'
    var_3 = 'auto_identify_namespace_packages'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'nested'
    var_7 = [var_6]
    var_8 = 'prefix.root'
    var_9 = var_5.namespace_packages
    var_10 = var_8 in var_9
    var_11 = var_5.auto_identify_namespace_packages
    var_12 = var_5.supported_extensions



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_is_namespace_package_with_non_existent_path. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_file_path. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_directory_no_init_and_no_files. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_directory_no_init_but_py_files. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_directory_no_init_but_setup_cfg. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_directory_no_init_but_pyproject_toml. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_init_containing_pkg_resources_single_quotes. Retrieved 5/7 statements.
# Partially parsed test_is_namespace_package_with_init_containing_pkg_resources_double_quotes. Retrieved 5/7 statements.
# Partially parsed test_is_namespace_package_with_init_containing_pkgutil_single_quotes. Retrieved 5/7 statements.
# Partially parsed test_is_namespace_package_with_init_containing_pkgutil_double_quotes. Retrieved 5/7 statements.
# Partially parsed test_is_namespace_package_with_init_without_namespace_markers. Retrieved 5/7 statements.
# Partially parsed test_is_namespace_package_with_src_extensions_including_txt. Retrieved 5/7 statements.


def test_case_0():
    var_0 = '/non/existent'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/some/file.txt'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/empty/dir'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/dir/with/py'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/dir/with/cfg'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/dir/with/toml'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/dir/with/init1'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)
    var_5 = b"__import__('pkg_resources').declare_namespace(__name__)"

def test_case_0():
    var_0 = '/dir/with/init2'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)
    var_5 = b'__import__("pkg_resources").declare_namespace(__name__)'

def test_case_0():
    var_0 = '/dir/with/init3'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)
    var_5 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"

def test_case_0():
    var_0 = '/dir/with/init4'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)
    var_5 = b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)'

def test_case_0():
    var_0 = '/dir/with/init5'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)
    var_5 = b"print('hello')"

def test_case_0():
    var_0 = '/dir/with/txt'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = 'txt'
    var_4 = [var_2, var_3]
    var_5 = frozenset(var_4)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_is_module_with_py_extension. Retrieved 2/11 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 2/12 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'some_module'
    var_1 = [var_0]
    var_2 = '.py'

def test_case_0():
    var_0 = 'some_module'
    var_1 = [var_0]
    var_2 = 0

def test_case_0():
    var_0 = 'some_module'
    var_1 = [var_0]
    var_2 = '__init__.py'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_namespace_not_in_config_and_auto_identify_false. Retrieved 7/10 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = set()
    var_1 = False
    var_2 = 'namespace_packages'
    var_3 = 'auto_identify_namespace_packages'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'a.b'
    var_7 = '/tmp'
    var_8 = [var_7]
    var_9 = 'a'
    var_10 = (var_9,)



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_src_path_is_module_true_for_matching_dir. Retrieved 4/7 statements.
# Partially parsed test_src_path_is_module_false_for_wrong_name. Retrieved 4/7 statements.
# Partially parsed test_src_path_is_module_false_for_file. Retrieved 5/8 statements.
# Partially parsed test_src_path_is_module_false_for_case_mismatch. Retrieved 5/8 statements.
# Partially parsed test_src_path_is_module_false_for_all_conditions_failing. Retrieved 4/7 statements.


def test_case_0():
    var_0 = '/some/path/mymodule'
    var_1 = [var_0]
    var_2 = True
    var_3 = lambda x: var_2
    var_4 = 'mymodule'

def test_case_0():
    var_0 = '/some/path/mymodule'
    var_1 = [var_0]
    var_2 = True
    var_3 = lambda x: var_2
    var_4 = 'othermodule'

def test_case_0():
    var_0 = '/some/path/mymodule'
    var_1 = [var_0]
    var_2 = False
    var_3 = True
    var_4 = lambda x: var_3
    var_5 = 'mymodule'

def test_case_0():
    var_0 = '/some/path/mymodule'
    var_1 = [var_0]
    var_2 = True
    var_3 = False
    var_4 = lambda x: var_3
    var_5 = 'mymodule'

def test_case_0():
    var_0 = '/some/path/mymodule'
    var_1 = [var_0]
    var_2 = False
    var_3 = lambda x: var_2
    var_4 = 'othermodule'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_namespace_package_without_init_and_no_files. Retrieved 4/11 statements.
# Partially parsed test_namespace_package_without_init_and_no_source_files_but_has_setup_cfg. Retrieved 5/14 statements.
# Partially parsed test_namespace_package_without_init_and_no_source_files_but_has_pyproject_toml. Retrieved 5/14 statements.
# Partially parsed test_namespace_package_without_init_and_has_source_file. Retrieved 5/14 statements.
# Partially parsed test_namespace_package_with_init_containing_pkg_resources_declare_namespace_single_quotes. Retrieved 6/15 statements.
# Partially parsed test_namespace_package_with_init_containing_pkg_resources_declare_namespace_double_quotes. Retrieved 6/15 statements.
# Partially parsed test_namespace_package_with_init_containing_pkgutil_extend_path_single_quotes. Retrieved 6/15 statements.
# Partially parsed test_namespace_package_with_init_containing_pkgutil_extend_path_double_quotes. Retrieved 6/15 statements.
# Partially parsed test_namespace_package_with_init_without_namespace_declaration. Retrieved 6/15 statements.
# Partially parsed test_not_a_package. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'setup.cfg'
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'pyproject.toml'
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'module.py'
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = b"__import__('pkg_resources').declare_namespace(__name__)"
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = b'__import__("pkg_resources").declare_namespace(__name__)'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = b''
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)



