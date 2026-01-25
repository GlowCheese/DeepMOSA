####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test__src_path_with_existing_module. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_nested_module. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_namespace_package. Retrieved 4/8 statements.
# Partially parsed test__src_path_with_auto_identified_namespace_package. Retrieved 3/7 statements.
# Partially parsed test__src_path_with_non_existing_module. Retrieved 2/6 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = None
    var_1 = 'src_paths'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'module'
    var_5 = module_1._src_path(var_4, var_3)
    assert var_5 is None

def test_case_0():
    var_0 = 'src'
    var_1 = [var_0]
    var_2 = 'module'

def test_case_0():
    var_0 = 'src'
    var_1 = [var_0]
    var_2 = 'parent.child'

def test_case_0():
    var_0 = 'src'
    var_1 = [var_0]
    var_2 = 'parent'
    var_3 = [var_2]
    var_4 = 'parent.child'

def test_case_0():
    var_0 = 'src'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'parent.child'

def test_case_0():
    var_0 = 'src'
    var_1 = [var_0]
    var_2 = 'nonexistent'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_namespace_not_in_config_and_not_auto_identified. Retrieved 7/13 statements.


def test_case_0():
    var_0 = '/some/path'
    var_1 = [var_0]
    var_2 = []
    var_3 = False
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = 'module.submodule'
    var_7 = [var_0]
    var_8 = ()



# Parsed testcases at query #3
#--------------------------

# Partially parsed test__known_pattern_matches_first_pattern. Retrieved 5/10 statements.
# Partially parsed test__known_pattern_matches_second_pattern. Retrieved 6/13 statements.
# Partially parsed test__known_pattern_no_match. Retrieved 5/10 statements.
# Partially parsed test__known_pattern_placement_not_in_sections. Retrieved 5/10 statements.
# Partially parsed test__known_pattern_empty_name. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'section1'
    var_1 = 'section2'
    var_2 = {var_0, var_1}
    var_3 = '^test\\.module$'
    var_4 = 'test.module.submodule'

def test_case_0():
    var_0 = 'section1'
    var_1 = 'section2'
    var_2 = {var_0, var_1}
    var_3 = '^test\\.module$'
    var_4 = '^test\\.module\\.submodule$'
    var_5 = 'test.module.submodule'

def test_case_0():
    var_0 = 'section1'
    var_1 = 'section2'
    var_2 = {var_0, var_1}
    var_3 = '^other\\.module$'
    var_4 = 'test.module'

def test_case_0():
    var_0 = 'section1'
    var_1 = {var_0}
    var_2 = '^test\\.module$'
    var_3 = 'section2'
    var_4 = 'test.module'

def test_case_0():
    var_0 = 'section1'
    var_1 = {var_0}
    var_2 = '^test$'
    var_3 = ''



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_true. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 'test_namespace'
    var_1 = {var_0}
    var_2 = False
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = '/test/src'
    var_6 = [var_5]
    var_7 = 'test_namespace.submodule'
    var_8 = [var_5]
    var_9 = ()



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = '/path/to/src'
    var_2 = [var_1]
    var_3 = 'module'
    var_4 = [var_3]
    var_5 = False
    var_6 = [var_1]
    var_7 = ()



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_true. Retrieved 9/15 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = 'module'
    var_2 = (var_1,)
    var_3 = False
    var_4 = 'src'
    var_5 = [var_4]
    var_6 = '.py'
    var_7 = [var_6]
    var_8 = 'src/module'
    var_9 = [var_8]
    var_10 = ()



# Parsed testcases at query #7
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = module_1._forced_separate(var_0, var_4)
    var_6 = bool(var_5 == ('test', 'Matched forced_separate (test) config value.'))
    assert var_6 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test*'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'test123'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('test*', 'Matched forced_separate (test*) config value.'))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '.hidden'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = module_1._forced_separate(var_0, var_4)
    var_6 = bool(var_5 == ('.hidden', 'Matched forced_separate (.hidden) config value.'))
    assert var_6 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'hidden'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '.hidden123'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('hidden', 'Matched forced_separate (hidden) config value.'))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'other'
    var_6 = module_1._forced_separate(var_5, var_4)
    assert var_6 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = 'forced_separate'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'test'
    var_5 = module_1._forced_separate(var_4, var_3)
    assert var_5 is None



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_src_path_is_module_when_name_matches_and_is_dir_and_exists_case_sensitive. Retrieved 2/5 statements.
# Partially parsed test_src_path_is_not_module_when_name_does_not_match. Retrieved 3/6 statements.
# Partially parsed test_src_path_is_not_module_when_not_a_directory. Retrieved 2/5 statements.
# Partially parsed test_src_path_is_not_module_when_does_not_exist_case_sensitively. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'module_name'
    var_1 = [var_0]
    var_2 = True

def test_case_0():
    var_0 = 'different_name'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'module_name'

def test_case_0():
    var_0 = 'module_name'
    var_1 = [var_0]
    var_2 = False

def test_case_0():
    var_0 = 'module_name'
    var_1 = [var_0]
    var_2 = True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_predicate_at_line_16_evaluates_to_true. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'module'
    var_1 = 'src'
    var_2 = [var_1]
    var_3 = 'src/module'
    var_4 = [var_3]
    var_5 = ()



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_forced_separate_predicate_false. Retrieved 4/5 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test'
    var_3 = 'other'
    var_4 = module_1._forced_separate(var_3, var_1)
    assert var_4 is None



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_known_pattern_predicate_false. Retrieved 6/8 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'pattern1'
    var_3 = 'placement1'
    var_4 = (var_2, var_3)
    var_5 = 'test.module'
    var_6 = module_1._known_pattern(var_5, var_1)
    assert var_6 is None



# Parsed testcases at query #12
#--------------------------

# Partially parsed test__is_namespace_package_not_a_package. Retrieved 4/7 statements.
# Partially parsed test__is_namespace_package_with_init_file. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_with_init_file_double_quotes. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_with_init_file_pkgutil. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_with_init_file_pkgutil_double_quotes. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_with_init_file_no_namespace. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_without_init_file. Retrieved 5/9 statements.
# Partially parsed test__is_namespace_package_without_init_file_with_py_files. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_without_init_file_with_setup_cfg. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_without_init_file_with_pyproject_toml. Retrieved 7/13 statements.


def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/tmp/test_package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = "__import__('pkg_resources').declare_namespace(__name__)"
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/tmp/test_package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = '__import__("pkg_resources").declare_namespace(__name__)'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/tmp/test_package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = "__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/tmp/test_package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = '__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/tmp/test_package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = '# Regular package'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/tmp/test_package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = '/tmp/test_package'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'module.py'
    var_4 = '# Some code'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/tmp/test_package'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'setup.cfg'
    var_4 = '[metadata]\nname = test'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/tmp/test_package'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'pyproject.toml'
    var_4 = "[build-system]\nrequires = ['setuptools']"
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test__src_path_is_module_returns_true. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]



# Parsed testcases at query #14
#--------------------------

# Partially parsed test__src_path_with_module_in_src_paths. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_nested_module. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_namespace_package. Retrieved 4/8 statements.
# Partially parsed test__src_path_with_auto_identify_namespace_packages. Retrieved 6/10 statements.
# Partially parsed test__src_path_with_src_path_is_module. Retrieved 2/6 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = None
    var_1 = 'src_paths'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'module'
    var_5 = module_1._src_path(var_4, var_3)
    assert var_5 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = 'src_paths'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'module'
    var_5 = module_1._src_path(var_4, var_3)
    assert var_5 is None

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = 'module'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = 'parent.child'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = 'parent'
    var_3 = [var_2]
    var_4 = 'parent.child'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)
    var_6 = 'parent.child'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = 'src'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_forced_separate_not_ends_with_asterisk. Retrieved 4/5 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test'
    var_3 = 'test_file'
    var_4 = module_1._forced_separate(var_3, var_1)
    var_5 = bool(var_4 is not None)
    assert var_5 is True
    var_6 = var_4[0]
    assert var_6 == 'test'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 2/5 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 1/6 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 2/5 statements.
# Partially parsed test_is_module_non_existent. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'module'
    var_1 = [var_0]
    var_2 = 'module.py'

def test_case_0():
    var_0 = 'module'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'package'
    var_1 = [var_0]
    var_2 = 'package/__init__.py'

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = [var_0]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_predicate_at_line_6_evaluates_to_true. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'some_path'
    var_1 = [var_0]
    var_2 = '__init__.py'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 'test_namespace'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)
    var_5 = 'module.py'
    var_6 = ''



# Parsed testcases at query #19
#--------------------------

# Partially parsed test__is_namespace_package_returns_false_when_path_is_not_a_package. Retrieved 4/7 statements.
# Partially parsed test__is_namespace_package_returns_false_when_path_is_a_package_with_init_file. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_true_when_path_is_a_package_without_init_file_and_no_source_files. Retrieved 5/9 statements.
# Partially parsed test__is_namespace_package_returns_false_when_path_is_a_package_without_init_file_but_has_source_files. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_false_when_path_is_a_package_without_init_file_but_has_setup_cfg. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_false_when_path_is_a_package_without_init_file_but_has_pyproject_toml. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_true_when_path_is_a_package_with_init_file_containing_pkg_resources_declare_namespace. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_true_when_path_is_a_package_with_init_file_containing_pkgutil_extend_path. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_false_when_path_is_a_package_with_init_file_not_containing_namespace_declaration. Retrieved 7/13 statements.


def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/existing/package/with/__init__.py'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = "print('hello')"
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/existing/package/without/__init__.py'
    var_1 = [var_0]
    var_2 = True
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = '/existing/package/with/source/files'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'module.py'
    var_4 = "print('hello')"
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/existing/package/with/setup.cfg'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'setup.cfg'
    var_4 = '[metadata]\nname = test'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/existing/package/with/pyproject.toml'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'pyproject.toml'
    var_4 = "[build-system]\nrequires = ['setuptools']"
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/existing/package/with/namespace/__init__.py'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = "__import__('pkg_resources').declare_namespace(__name__)"
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/existing/package/with/namespace/__init__.py'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = "__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/existing/package/without/namespace/__init__.py'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = "print('hello')"
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'some_path'
    var_1 = [var_0]
    var_2 = '__init__.py'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_init_file_exists_with_namespace_declaration. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 'some_namespace_package'
    var_1 = [var_0]
    var_2 = '__init__.py'
    var_3 = b'__import__("pkg_resources").declare_namespace(__name__)'
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 1/3 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 1/3 statements.
# Partially parsed test_is_module_with_init_file. Retrieved 1/3 statements.
# Partially parsed test_is_module_without_any_file. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'module.py'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'module.so'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'package'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = [var_0]



# Parsed testcases at query #23
#--------------------------

# Partially parsed test__is_module_returns_true_for_py_file. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'module.py'
    var_1 = [var_0]



# Parsed testcases at query #24
#--------------------------

# Partially parsed test__is_namespace_package_with_non_package_path. Retrieved 4/7 statements.
# Partially parsed test__is_namespace_package_with_package_having_init_file. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_with_package_having_init_file_no_namespace_declaration. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_with_package_no_init_file_but_source_files. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_with_package_no_init_file_no_source_files. Retrieved 5/9 statements.
# Partially parsed test__is_namespace_package_with_package_having_setup_cfg. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_with_package_having_pyproject_toml. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_with_package_init_file_using_double_quotes. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_with_package_init_file_using_pkg_resources. Retrieved 7/13 statements.


def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/path/to/package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = "__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/path/to/package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = '# Regular package'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/path/to/package'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'module.py'
    var_4 = '# Module content'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/path/to/package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = '/path/to/package'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'setup.cfg'
    var_4 = '[metadata]\nname = package'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/path/to/package'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'pyproject.toml'
    var_4 = "[build-system]\nrequires = ['setuptools']"
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/path/to/package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = '__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/path/to/package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = '__import__("pkg_resources").declare_namespace(__name__)'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_known_pattern_returns_none_when_placement_not_in_sections. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 'placement'
    var_2 = []
    var_3 = 'test.module'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_init_file_exists. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'some_path'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)
    var_5 = '__init__.py'



# Parsed testcases at query #27
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test.module'
    var_1 = []
    var_2 = []
    var_3 = 'known_patterns'
    var_4 = 'sections'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.Config(**var_5)
    var_7 = module_1._known_pattern(var_0, var_6)
    assert var_7 is None



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_src_path_returns_none_for_non_existent_module. Retrieved 7/11 statements.
# Partially parsed test_src_path_returns_firstparty_for_existing_module. Retrieved 7/11 statements.
# Partially parsed test_src_path_returns_firstparty_for_nested_module. Retrieved 7/11 statements.
# Partially parsed test_src_path_returns_firstparty_for_namespace_package. Retrieved 8/12 statements.
# Partially parsed test_src_path_returns_firstparty_for_auto_identified_namespace_package. Retrieved 7/11 statements.


def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = '.py'
    var_5 = {var_4}
    var_6 = frozenset(var_5)
    var_7 = 'non_existent_module'

def test_case_0():
    var_0 = '.'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = '.py'
    var_5 = {var_4}
    var_6 = frozenset(var_5)
    var_7 = 'existing_module'

def test_case_0():
    var_0 = '.'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = '.py'
    var_5 = {var_4}
    var_6 = frozenset(var_5)
    var_7 = 'parent.child'

def test_case_0():
    var_0 = '.'
    var_1 = [var_0]
    var_2 = 'parent'
    var_3 = {var_2}
    var_4 = False
    var_5 = '.py'
    var_6 = {var_5}
    var_7 = frozenset(var_6)
    var_8 = 'parent.child'

def test_case_0():
    var_0 = '.'
    var_1 = [var_0]
    var_2 = set()
    var_3 = True
    var_4 = '.py'
    var_5 = {var_4}
    var_6 = frozenset(var_5)
    var_7 = 'parent.child'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_src_path_is_module_returns_true_for_valid_module. Retrieved 2/5 statements.
# Partially parsed test_src_path_is_module_returns_false_for_non_matching_name. Retrieved 3/6 statements.
# Partially parsed test_src_path_is_module_returns_false_for_non_directory. Retrieved 2/5 statements.
# Partially parsed test_src_path_is_module_returns_false_for_case_sensitive_mismatch. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'valid_module'
    var_1 = [var_0]
    var_2 = True

def test_case_0():
    var_0 = 'different_name'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'valid_module'

def test_case_0():
    var_0 = 'valid_module'
    var_1 = [var_0]
    var_2 = False

def test_case_0():
    var_0 = 'Valid_Module'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'valid_module'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test__src_path_with_non_existent_module. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_module_in_src_paths. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_nested_module. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_namespace_package. Retrieved 4/8 statements.
# Partially parsed test__src_path_with_auto_identified_namespace_package. Retrieved 6/10 statements.
# Partially parsed test__src_path_with_module_in_root_src_path. Retrieved 2/6 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = ''
    var_1 = []
    var_2 = 'src_paths'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = module_1._src_path(var_0, var_4)
    assert var_5 is None

def test_case_0():
    var_0 = 'non_existent_module'
    var_1 = '/tmp'
    var_2 = [var_1]

def test_case_0():
    var_0 = 'existing_module'
    var_1 = '/path/to/src'
    var_2 = [var_1]

def test_case_0():
    var_0 = 'parent.child'
    var_1 = '/path/to/src'
    var_2 = [var_1]

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = 'parent'
    var_3 = {var_2}
    var_4 = 'parent.child'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = True
    var_3 = '.py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)
    var_6 = 'parent.child'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = 'src'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 'test_namespace_package'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)
    var_5 = True
    var_6 = 'module.py'
    var_7 = '# dummy content'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_is_module_returns_true_for_py_file. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'module.py'
    var_1 = [var_0]



# Parsed testcases at query #33
#--------------------------

# Partially parsed test__src_path_is_module_returns_true_for_valid_module. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'valid_module'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'valid_module'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_init_file_does_not_exist. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 'nonexistent_path'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_is_module_returns_true_for_py_file. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'module.py'
    var_1 = [var_0]



# Parsed testcases at query #36
#--------------------------

# Partially parsed test__src_path_with_module_in_src_paths. Retrieved 5/11 statements.
# Partially parsed test__src_path_with_nested_module_in_src_paths. Retrieved 8/15 statements.
# Partially parsed test__src_path_with_namespace_package. Retrieved 6/12 statements.
# Partially parsed test__src_path_with_auto_identified_namespace_package. Retrieved 10/18 statements.
# Partially parsed test__src_path_with_src_path_as_module. Retrieved 5/10 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = None
    var_1 = set()
    var_2 = False
    var_3 = frozenset()
    var_4 = 'src_paths'
    var_5 = 'namespace_packages'
    var_6 = 'auto_identify_namespace_packages'
    var_7 = 'supported_extensions'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_2, var_7: var_3}
    var_9 = module_0.Config(**var_8)
    var_10 = 'module'
    var_11 = module_1._src_path(var_10, var_9)
    assert var_11 is None

import isort.settings as module_0
import isort.place as module_1

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
    var_10 = 'module'
    var_11 = module_1._src_path(var_10, var_9)
    assert var_11 is None

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = 'module'
    var_3 = set()
    var_4 = False
    var_5 = frozenset()

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = 'parent'
    var_3 = 'child'
    var_4 = True
    var_5 = set()
    var_6 = False
    var_7 = frozenset()
    var_8 = 'parent.child'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = 'namespace'
    var_3 = {var_2}
    var_4 = False
    var_5 = frozenset()
    var_6 = 'namespace.module'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = 'namespace'
    var_3 = 'module.py'
    var_4 = 'content'
    var_5 = set()
    var_6 = True
    var_7 = '.py'
    var_8 = {var_7}
    var_9 = frozenset(var_8)
    var_10 = 'namespace.module'

def test_case_0():
    var_0 = '/path/to/module'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = frozenset()
    var_5 = 'module'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test__is_namespace_package_not_a_package. Retrieved 4/7 statements.
# Partially parsed test__is_namespace_package_with_init_file. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_without_init_file. Retrieved 5/9 statements.
# Partially parsed test__is_namespace_package_with_other_files. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_with_setup_cfg. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_with_pyproject_toml. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_init_without_namespace_declaration. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_init_with_pkgutil. Retrieved 7/13 statements.


def test_case_0():
    var_0 = '/nonexistent'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/tmp/test_package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = "__import__('pkg_resources').declare_namespace(__name__)"
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/tmp/test_namespace'
    var_1 = [var_0]
    var_2 = True
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = '/tmp/test_with_files'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'module.py'
    var_4 = '# test'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/tmp/test_with_setup'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'setup.cfg'
    var_4 = '[metadata]'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/tmp/test_with_pyproject'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'pyproject.toml'
    var_4 = '[build-system]'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/tmp/test_regular_package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = '# regular package'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/tmp/test_pkgutil_namespace'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = '__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_false. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 'module'
    var_1 = []
    var_2 = False
    var_3 = 'src'
    var_4 = [var_3]
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = 'src/module'
    var_8 = [var_7]
    var_9 = ()



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'module'
    var_1 = [var_0]
    var_2 = 'module.py'
    var_3 = ''



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_predicate_at_line_13_evaluates_to_true. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 'test_package'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)
    var_5 = 'module.py'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_src_path_module_in_src_paths. Retrieved 7/11 statements.
# Partially parsed test_src_path_nested_module. Retrieved 7/11 statements.
# Partially parsed test_src_path_namespace_package. Retrieved 8/12 statements.
# Partially parsed test_src_path_auto_identify_namespace_package. Retrieved 7/11 statements.
# Partially parsed test_src_path_not_found. Retrieved 7/11 statements.


def test_case_0():
    var_0 = '/project/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = 'py'
    var_5 = {var_4}
    var_6 = frozenset(var_5)
    var_7 = 'module'

def test_case_0():
    var_0 = '/project/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = 'py'
    var_5 = {var_4}
    var_6 = frozenset(var_5)
    var_7 = 'parent.child'

def test_case_0():
    var_0 = '/project/src'
    var_1 = [var_0]
    var_2 = 'parent'
    var_3 = {var_2}
    var_4 = False
    var_5 = 'py'
    var_6 = {var_5}
    var_7 = frozenset(var_6)
    var_8 = 'parent.child'

def test_case_0():
    var_0 = '/project/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = True
    var_4 = 'py'
    var_5 = {var_4}
    var_6 = frozenset(var_5)
    var_7 = 'parent.child'

def test_case_0():
    var_0 = '/project/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = 'py'
    var_5 = {var_4}
    var_6 = frozenset(var_5)
    var_7 = 'nonexistent'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = '/path/to/src'
    var_2 = [var_1]
    var_3 = 'module'
    var_4 = [var_3]
    var_5 = False
    var_6 = [var_1]
    var_7 = ()



# Parsed testcases at query #43
#--------------------------

# Partially parsed test__src_path_with_empty_name. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_module_in_src_paths. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_nested_module_in_src_paths. Retrieved 4/8 statements.
# Partially parsed test__src_path_with_auto_identify_namespace_packages. Retrieved 6/10 statements.
# Partially parsed test__src_path_with_module_not_in_src_paths. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_prefix_and_module_in_src_paths. Retrieved 4/8 statements.
# Partially parsed test__src_path_with_custom_src_paths. Retrieved 3/9 statements.


def test_case_0():
    var_0 = ''
    var_1 = '/some/path'
    var_2 = [var_1]

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = None
    var_2 = 'src_paths'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = module_1._src_path(var_0, var_4)
    assert var_5 is None

def test_case_0():
    var_0 = '/some/path'
    var_1 = [var_0]
    var_2 = 'module'

def test_case_0():
    var_0 = '/some/path'
    var_1 = [var_0]
    var_2 = 'parent'
    var_3 = {var_2}
    var_4 = 'parent.child'

def test_case_0():
    var_0 = '/some/path'
    var_1 = [var_0]
    var_2 = True
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)
    var_6 = 'parent.child'

def test_case_0():
    var_0 = '/some/path'
    var_1 = [var_0]
    var_2 = 'nonexistent'

def test_case_0():
    var_0 = '/some/path'
    var_1 = [var_0]
    var_2 = 'module'
    var_3 = 'parent'
    var_4 = (var_3,)

def test_case_0():
    var_0 = '/custom/path'
    var_1 = [var_0]
    var_2 = 'module'
    var_3 = '/another/path'
    var_4 = [var_3]



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_predicate_at_line_13_evaluates_to_true. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 'test_package'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)
    var_5 = 'module.py'
    var_6 = '# test module'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_is_module_returns_true_for_py_file. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'module.py'
    var_1 = [var_0]



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = 'src'
    var_2 = [var_1]
    var_3 = 'module'
    var_4 = [var_3]
    var_5 = False
    var_6 = '.py'
    var_7 = [var_6]
    var_8 = [var_1]
    var_9 = ()



# Parsed testcases at query #47
#--------------------------

# Partially parsed test__src_path_with_simple_module_in_src_paths. Retrieved 7/11 statements.
# Partially parsed test__src_path_with_nested_module_in_src_paths. Retrieved 7/11 statements.
# Partially parsed test__src_path_with_namespace_package. Retrieved 8/12 statements.
# Partially parsed test__src_path_with_auto_identified_namespace_package. Retrieved 7/11 statements.
# Partially parsed test__src_path_with_module_not_found. Retrieved 7/11 statements.
# Partially parsed test__src_path_with_root_module_in_src_path. Retrieved 7/11 statements.


def test_case_0():
    var_0 = '/project/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'module'

def test_case_0():
    var_0 = '/project/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'parent.child'

def test_case_0():
    var_0 = '/project/src'
    var_1 = [var_0]
    var_2 = 'parent'
    var_3 = {var_2}
    var_4 = False
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)
    var_8 = 'parent.child'

def test_case_0():
    var_0 = '/project/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = True
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'parent.child'

def test_case_0():
    var_0 = '/project/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'nonexistent'

def test_case_0():
    var_0 = '/project/src/module'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'module'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test__is_namespace_package_returns_true_for_namespace_package. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 'namespace_package'
    var_1 = [var_0]
    var_2 = '__init__.py'
    var_3 = b'__import__("pkg_resources").declare_namespace(__name__)'
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)



# Parsed testcases at query #49
#--------------------------

# Partially parsed test__is_module_returns_true_for_py_file. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'test_module.py'
    var_1 = [var_0]



# Parsed testcases at query #50
#--------------------------

# Partially parsed test__is_namespace_package_returns_true_when_package_has_no_init_file_and_no_source_files. Retrieved 7/11 statements.


def test_case_0():
    var_0 = '/fake/namespace/package'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = '.pyx'
    var_4 = '.pxd'
    var_5 = '.pyi'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = frozenset(var_6)



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 'some_namespace_package'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_false. Retrieved 9/24 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = []
    var_2 = False
    var_3 = '/path/to/src'
    var_4 = [var_3]
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = [var_3]
    var_8 = ()
    var_9 = 'module'
    var_10 = [var_3]
    var_11 = 'module'



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = '/path/to/src'
    var_2 = [var_1]
    var_3 = 'module'
    var_4 = [var_3]
    var_5 = False
    var_6 = '.py'
    var_7 = [var_6]
    var_8 = [var_1]
    var_9 = ()



# Parsed testcases at query #54
#--------------------------

# Partially parsed test__src_path_module_in_src_paths. Retrieved 7/11 statements.
# Partially parsed test__src_path_nested_module_in_src_paths. Retrieved 7/11 statements.
# Partially parsed test__src_path_namespace_package_in_src_paths. Retrieved 8/12 statements.
# Partially parsed test__src_path_auto_identify_namespace_package. Retrieved 7/11 statements.
# Partially parsed test__src_path_not_found. Retrieved 7/11 statements.
# Partially parsed test__src_path_root_module_in_src_path. Retrieved 7/11 statements.


def test_case_0():
    var_0 = '/project/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'module'

def test_case_0():
    var_0 = '/project/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'parent.child'

def test_case_0():
    var_0 = '/project/src'
    var_1 = [var_0]
    var_2 = 'parent'
    var_3 = {var_2}
    var_4 = False
    var_5 = 'py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)
    var_8 = 'parent.child'

def test_case_0():
    var_0 = '/project/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = True
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'parent.child'

def test_case_0():
    var_0 = '/project/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'nonexistent'

def test_case_0():
    var_0 = '/project/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'src'



# Parsed testcases at query #55
#--------------------------

# Partially parsed test__is_namespace_package_returns_true_when_init_file_exists_and_contains_declare_namespace. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 'some_path'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)
    var_5 = '__init__.py'
    var_6 = '__import__("pkg_resources").declare_namespace(__name__)'



# Parsed testcases at query #56
#--------------------------

# Partially parsed test__src_path_is_module_returns_true. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'module_name'
    var_1 = [var_0]



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_filenames_list_not_empty. Retrieved 8/11 statements.


def test_case_0():
    var_0 = 'some_path'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)
    var_5 = '.'
    var_6 = 'setup.cfg'
    var_7 = 'pyproject.toml'
    var_8 = (var_6, var_7)



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 8/16 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = '/path/to/src'
    var_2 = [var_1]
    var_3 = []
    var_4 = True
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = [var_1]
    var_8 = ()
    var_9 = lambda *args: var_4



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_false. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = ()
    var_2 = False
    var_3 = '/path/to/src'
    var_4 = [var_3]
    var_5 = '.py'
    var_6 = (var_5,)
    var_7 = [var_3]
    var_8 = ()



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_is_module_returns_true_for_py_file. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'test_module.py'
    var_1 = [var_0]



# Parsed testcases at query #61
#--------------------------

# Partially parsed test__is_namespace_package_without_init_file. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 'test_package'
    var_1 = [var_0]
    var_2 = 'module.py'
    var_3 = ''
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)



# Parsed testcases at query #62
#--------------------------

# Partially parsed test__src_path_with_nested_module_in_namespace_package. Retrieved 4/8 statements.
# Partially parsed test__src_path_with_nested_module_in_auto_identified_namespace_package. Retrieved 5/9 statements.
# Partially parsed test__src_path_with_module_in_src_path. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_package_in_src_path. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_src_path_is_module. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_no_match. Retrieved 2/6 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'package'
    var_3 = {var_2}
    var_4 = 'package.submodule'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = 'package.submodule'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'module'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'package'

def test_case_0():
    var_0 = '/src/module'
    var_1 = [var_0]
    var_2 = 'module'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'nonexistent'



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 2/5 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 1/6 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 2/5 statements.
# Partially parsed test_is_module_nonexistent. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'module'
    var_1 = [var_0]
    var_2 = 'module.py'

def test_case_0():
    var_0 = 'module'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'package'
    var_1 = [var_0]
    var_2 = 'package/__init__.py'

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = [var_0]



# Parsed testcases at query #64
#--------------------------

# Partially parsed test__is_namespace_package_returns_false_when_path_is_not_a_package. Retrieved 4/7 statements.
# Partially parsed test__is_namespace_package_returns_false_when_init_file_exists_without_namespace_declaration. Retrieved 7/15 statements.
# Partially parsed test__is_namespace_package_returns_true_when_init_file_contains_pkg_resources_declaration. Retrieved 7/15 statements.
# Partially parsed test__is_namespace_package_returns_true_when_init_file_contains_pkgutil_declaration. Retrieved 7/15 statements.
# Partially parsed test__is_namespace_package_returns_false_when_package_contains_py_files. Retrieved 7/15 statements.
# Partially parsed test__is_namespace_package_returns_false_when_package_contains_setup_cfg. Retrieved 7/15 statements.
# Partially parsed test__is_namespace_package_returns_false_when_package_contains_pyproject_toml. Retrieved 7/15 statements.
# Partially parsed test__is_namespace_package_returns_true_when_package_is_empty. Retrieved 5/10 statements.


def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = {var_2}
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/tmp/test_package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = "print('hello')"
    var_5 = '.py'
    var_6 = {var_5}
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/tmp/test_package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = '__import__("pkg_resources").declare_namespace(__name__)'
    var_5 = '.py'
    var_6 = {var_5}
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/tmp/test_package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = '__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    var_5 = '.py'
    var_6 = {var_5}
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/tmp/test_package'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'module.py'
    var_4 = "print('hello')"
    var_5 = '.py'
    var_6 = {var_5}
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/tmp/test_package'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'setup.cfg'
    var_4 = '[metadata]\nname=test'
    var_5 = '.py'
    var_6 = {var_5}
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/tmp/test_package'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'pyproject.toml'
    var_4 = "[build-system]\nrequires = ['setuptools']"
    var_5 = '.py'
    var_6 = {var_5}
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/tmp/test_package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '.py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_init_file_exists. Retrieved 1/3 statements.


def test_case_0():
    var_0 = '__init__.py'



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_true. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = 'module'
    var_2 = {var_1}
    var_3 = False
    var_4 = '/path/to/src'
    var_5 = [var_4]
    var_6 = '.py'
    var_7 = [var_6]
    var_8 = [var_4]
    var_9 = ()



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'some_namespace_package'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)



# Parsed testcases at query #68
#--------------------------

# Partially parsed test__is_namespace_package_returns_false_when_not_a_package. Retrieved 4/7 statements.
# Partially parsed test__is_namespace_package_returns_false_when_init_exists_without_namespace_declaration. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_true_when_init_contains_pkg_resources_declaration. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_true_when_init_contains_pkgutil_declaration. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_false_when_package_contains_source_files. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_false_when_package_contains_setup_cfg. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_false_when_package_contains_pyproject_toml. Retrieved 7/13 statements.


def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/tmp/test_package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = '# regular package'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/tmp/test_namespace_pkg'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = '__import__("pkg_resources").declare_namespace(__name__)'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/tmp/test_namespace_pkgutil'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = '__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/tmp/test_package_with_files'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'module.py'
    var_4 = '# some code'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/tmp/test_package_with_setup'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'setup.cfg'
    var_4 = '[metadata]\nname=test'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/tmp/test_package_with_pyproject'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'pyproject.toml'
    var_4 = '[build-system]\nrequires = []'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_predicate_at_line_13_evaluates_to_true. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 'test_namespace_package'
    var_1 = [var_0]
    var_2 = 'module.py'
    var_3 = '# test module'
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_is_module_returns_true_for_py_file. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'module.py'
    var_1 = [var_0]



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'module'
    var_1 = [var_0]
    var_2 = 'module.py'



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 8/18 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = []
    var_2 = True
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = '/path/to/src'
    var_6 = [var_5]
    var_7 = [var_5]
    var_8 = ()
    var_9 = '/path/to/src/module'
    var_10 = [var_9]



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_predicate_at_line_26. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 'src'
    var_1 = [var_0]
    var_2 = []
    var_3 = False
    var_4 = 'module.submodule'
    var_5 = [var_0]
    var_6 = ()



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 1/3 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 1/3 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 1/3 statements.
# Partially parsed test_is_module_without_any_file. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'module'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'module'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'module'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'not_module'
    var_1 = [var_0]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_known_pattern_matches_first_pattern. Retrieved 4/9 statements.
# Partially parsed test_known_pattern_matches_second_pattern. Retrieved 6/13 statements.
# Partially parsed test_known_pattern_no_match. Retrieved 4/9 statements.
# Partially parsed test_known_pattern_placement_not_in_sections. Retrieved 5/10 statements.
# Partially parsed test_known_pattern_matches_submodule. Retrieved 4/9 statements.
# Partially parsed test_known_pattern_empty_name. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'test.*'
    var_1 = 'placement1'
    var_2 = [var_1]
    var_3 = 'test.module'

def test_case_0():
    var_0 = 'test.*'
    var_1 = 'placement1'
    var_2 = 'module.*'
    var_3 = 'placement2'
    var_4 = [var_1, var_3]
    var_5 = 'module.submodule'

def test_case_0():
    var_0 = 'test.*'
    var_1 = 'placement1'
    var_2 = [var_1]
    var_3 = 'other.module'

def test_case_0():
    var_0 = 'test.*'
    var_1 = 'placement1'
    var_2 = 'placement2'
    var_3 = [var_2]
    var_4 = 'test.module'

def test_case_0():
    var_0 = 'test.*'
    var_1 = 'placement1'
    var_2 = [var_1]
    var_3 = 'parent.test.module'

def test_case_0():
    var_0 = 'test.*'
    var_1 = 'placement1'
    var_2 = [var_1]
    var_3 = ''

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'known_patterns'
    var_3 = 'sections'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.Config(**var_4)
    var_6 = 'test.module'
    var_7 = module_1._known_pattern(var_6, var_5)
    assert var_7 is None



# Parsed testcases at query #3
#--------------------------

# Partially parsed test__src_path_with_module_in_src_paths. Retrieved 5/9 statements.
# Partially parsed test__src_path_with_nested_module_in_src_paths. Retrieved 5/9 statements.
# Partially parsed test__src_path_with_namespace_package. Retrieved 6/10 statements.
# Partially parsed test__src_path_with_auto_identified_namespace_package. Retrieved 5/9 statements.
# Partially parsed test__src_path_with_src_path_is_module. Retrieved 5/9 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = None
    var_1 = set()
    var_2 = False
    var_3 = frozenset()
    var_4 = 'src_paths'
    var_5 = 'namespace_packages'
    var_6 = 'auto_identify_namespace_packages'
    var_7 = 'supported_extensions'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_2, var_7: var_3}
    var_9 = module_0.Config(**var_8)
    var_10 = 'module'
    var_11 = module_1._src_path(var_10, var_9)
    assert var_11 is None

import isort.settings as module_0
import isort.place as module_1

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
    var_10 = 'module'
    var_11 = module_1._src_path(var_10, var_9)
    assert var_11 is None

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = frozenset()
    var_5 = 'module'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = frozenset()
    var_5 = 'parent.child'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = 'parent'
    var_3 = {var_2}
    var_4 = False
    var_5 = frozenset()
    var_6 = 'parent.child'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = True
    var_4 = frozenset()
    var_5 = 'parent.child'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = frozenset()
    var_5 = 'src'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_predicate_at_line_7. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_path'
    var_2 = [var_1]
    var_3 = None
    assert var_3 is None
    var_4 = ()



# Parsed testcases at query #5
#--------------------------

# Partially parsed test__src_path_with_module_in_src_path. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_nested_module_in_namespace_package. Retrieved 4/8 statements.
# Partially parsed test__src_path_with_auto_identify_namespace_packages. Retrieved 3/7 statements.
# Partially parsed test__src_path_with_module_not_in_src_path. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_prefix. Retrieved 4/8 statements.
# Partially parsed test__src_path_with_src_path_is_module. Retrieved 2/6 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = None
    var_1 = 'src_paths'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'module'
    var_5 = module_1._src_path(var_4, var_3)
    assert var_5 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = 'src_paths'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'module'
    var_5 = module_1._src_path(var_4, var_3)
    assert var_5 is None

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = 'module'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = 'parent'
    var_3 = [var_2]
    var_4 = 'parent.child'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'parent.child'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = 'nonexistent'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = 'module.submodule'
    var_3 = 'module'
    var_4 = (var_3,)

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = 'src'



# Parsed testcases at query #6
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test*'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'testfile'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('test*', 'Matched forced_separate (test*) config value.'))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'testfile'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('test*', 'Matched forced_separate (test*) config value.'))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '.hidden'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '.hiddenfile'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('.hidden*', 'Matched forced_separate (.hidden*) config value.'))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'other*'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'testfile'
    var_6 = module_1._forced_separate(var_5, var_4)
    assert var_6 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = 'forced_separate'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'testfile'
    var_5 = module_1._forced_separate(var_4, var_3)
    assert var_5 is None



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = '/path/to/src'
    var_2 = [var_1]
    var_3 = 'module'
    var_4 = [var_3]
    var_5 = False
    var_6 = [var_1]
    var_7 = ()



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = '/path/to/src'
    var_2 = [var_1]
    var_3 = 'module'
    var_4 = [var_3]
    var_5 = False
    var_6 = '.py'
    var_7 = [var_6]
    var_8 = [var_1]
    var_9 = ()



# Parsed testcases at query #9
#--------------------------

# Partially parsed test__src_path_found_in_src_paths. Retrieved 7/11 statements.
# Partially parsed test__src_path_nested_module. Retrieved 8/12 statements.
# Partially parsed test__src_path_not_found. Retrieved 7/11 statements.
# Partially parsed test__src_path_with_prefix. Retrieved 9/13 statements.
# Partially parsed test__src_path_namespace_package. Retrieved 8/12 statements.


def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = 'py'
    var_5 = {var_4}
    var_6 = frozenset(var_5)
    var_7 = 'module'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = 'parent'
    var_3 = {var_2}
    var_4 = False
    var_5 = 'py'
    var_6 = {var_5}
    var_7 = frozenset(var_6)
    var_8 = 'parent.child'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = 'py'
    var_5 = {var_4}
    var_6 = frozenset(var_5)
    var_7 = 'nonexistent'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = 'py'
    var_5 = {var_4}
    var_6 = frozenset(var_5)
    var_7 = 'module'
    var_8 = 'parent'
    var_9 = (var_8,)

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = 'namespace'
    var_3 = {var_2}
    var_4 = True
    var_5 = 'py'
    var_6 = {var_5}
    var_7 = frozenset(var_6)
    var_8 = 'namespace.module'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_src_paths_is_not_none. Retrieved 2/6 statements.


def test_case_0():
    var_0 = '/some/path'
    var_1 = [var_0]
    var_2 = 'module'



# Parsed testcases at query #11
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test_file'
    var_1 = 'test*'
    var_2 = [var_1]
    var_3 = 'forced_separate'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = module_1._forced_separate(var_0, var_5)
    var_7 = bool(var_6 == ('test', 'Matched forced_separate (test) config value.'))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '.test_file'
    var_1 = 'test*'
    var_2 = [var_1]
    var_3 = 'forced_separate'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = module_1._forced_separate(var_0, var_5)
    var_7 = bool(var_6 == ('test', 'Matched forced_separate (test) config value.'))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test_file'
    var_1 = 'test'
    var_2 = [var_1]
    var_3 = 'forced_separate'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = module_1._forced_separate(var_0, var_5)
    var_7 = bool(var_6 == ('test', 'Matched forced_separate (test) config value.'))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '.test_file'
    var_1 = 'test'
    var_2 = [var_1]
    var_3 = 'forced_separate'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = module_1._forced_separate(var_0, var_5)
    var_7 = bool(var_6 == ('test', 'Matched forced_separate (test) config value.'))
    assert var_7 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'module'
    var_1 = [var_0]
    var_2 = '.py'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_true. Retrieved 4/9 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test_namespace'
    var_3 = 'test_namespace.submodule'
    var_4 = '/path/to/src'
    var_5 = [var_4]



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_false. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = []
    var_2 = False
    var_3 = '/path/to/src'
    var_4 = [var_3]
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = [var_3]
    var_8 = ()



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'module'
    var_1 = [var_0]
    var_2 = '__init__.py'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_true. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = 'module'
    var_2 = {var_1}
    var_3 = 'src'
    var_4 = [var_3]
    var_5 = False
    var_6 = '.py'
    var_7 = [var_6]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test__src_path_with_no_nested_module_and_valid_module_path. Retrieved 7/11 statements.
# Partially parsed test__src_path_with_no_nested_module_and_invalid_module_path. Retrieved 7/11 statements.
# Partially parsed test__src_path_with_nested_module_and_namespace_package. Retrieved 8/12 statements.
# Partially parsed test__src_path_with_nested_module_and_auto_identified_namespace_package. Retrieved 7/11 statements.
# Partially parsed test__src_path_with_src_path_is_module. Retrieved 7/11 statements.


def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = 'py'
    var_5 = {var_4}
    var_6 = frozenset(var_5)
    var_7 = 'module'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = 'py'
    var_5 = {var_4}
    var_6 = frozenset(var_5)
    var_7 = 'nonexistent'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = 'parent'
    var_3 = {var_2}
    var_4 = False
    var_5 = 'py'
    var_6 = {var_5}
    var_7 = frozenset(var_6)
    var_8 = 'parent.child'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = True
    var_4 = 'py'
    var_5 = {var_4}
    var_6 = frozenset(var_5)
    var_7 = 'parent.child'

def test_case_0():
    var_0 = '/path/to/module'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = 'py'
    var_5 = {var_4}
    var_6 = frozenset(var_5)
    var_7 = 'module'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test__src_path_returns_none_for_non_existent_module. Retrieved 5/9 statements.
# Partially parsed test__src_path_returns_firstparty_for_module_in_src_path. Retrieved 5/9 statements.
# Partially parsed test__src_path_returns_firstparty_for_nested_module_in_namespace_package. Retrieved 6/10 statements.
# Partially parsed test__src_path_returns_firstparty_for_module_in_nested_src_path. Retrieved 5/9 statements.
# Partially parsed test__src_path_returns_firstparty_for_module_with_prefix. Retrieved 7/11 statements.
# Partially parsed test__src_path_returns_firstparty_for_namespace_package_with_auto_identify. Retrieved 5/9 statements.


def test_case_0():
    var_0 = '/fake/path'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = frozenset()
    var_5 = 'non_existent_module'

def test_case_0():
    var_0 = '/real/path'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = frozenset()
    var_5 = 'existing_module'

def test_case_0():
    var_0 = '/real/path'
    var_1 = [var_0]
    var_2 = 'parent'
    var_3 = {var_2}
    var_4 = False
    var_5 = frozenset()
    var_6 = 'parent.child'

def test_case_0():
    var_0 = '/real/path'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = frozenset()
    var_5 = 'parent.child'

def test_case_0():
    var_0 = '/real/path'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = frozenset()
    var_5 = 'module'
    var_6 = 'parent'
    var_7 = (var_6,)

def test_case_0():
    var_0 = '/real/path'
    var_1 = [var_0]
    var_2 = set()
    var_3 = True
    var_4 = frozenset()
    var_5 = 'namespace_pkg'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test__src_path_with_empty_name. Retrieved 5/9 statements.
# Partially parsed test__src_path_with_module_in_src_paths. Retrieved 6/13 statements.
# Partially parsed test__src_path_with_package_in_src_paths. Retrieved 7/17 statements.
# Partially parsed test__src_path_with_nested_module_in_namespace_package. Retrieved 8/18 statements.
# Partially parsed test__src_path_with_auto_identified_namespace_package. Retrieved 9/19 statements.
# Partially parsed test__src_path_with_non_existent_module. Retrieved 5/9 statements.
# Partially parsed test__src_path_with_src_path_as_module. Retrieved 6/16 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = None
    var_1 = ()
    var_2 = False
    var_3 = frozenset()
    var_4 = 'src_paths'
    var_5 = 'namespace_packages'
    var_6 = 'auto_identify_namespace_packages'
    var_7 = 'supported_extensions'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_2, var_7: var_3}
    var_9 = module_0.Config(**var_8)
    var_10 = 'module'
    var_11 = module_1._src_path(var_10, var_9)
    assert var_11 is None

def test_case_0():
    var_0 = '/some/path'
    var_1 = [var_0]
    var_2 = ()
    var_3 = False
    var_4 = frozenset()
    var_5 = ''

def test_case_0():
    var_0 = '/some/path'
    var_1 = [var_0]
    var_2 = ()
    var_3 = False
    var_4 = frozenset()
    var_5 = '/some/path/module.py'
    var_6 = [var_5]
    var_7 = 'module'

def test_case_0():
    var_0 = '/some/path'
    var_1 = [var_0]
    var_2 = ()
    var_3 = False
    var_4 = frozenset()
    var_5 = '/some/path/package'
    var_6 = [var_5]
    var_7 = '/some/path/package/__init__.py'
    var_8 = [var_7]
    var_9 = 'package'

def test_case_0():
    var_0 = '/some/path'
    var_1 = [var_0]
    var_2 = 'namespace'
    var_3 = (var_2,)
    var_4 = False
    var_5 = frozenset()
    var_6 = '/some/path/namespace'
    var_7 = [var_6]
    var_8 = '/some/path/namespace/nested.py'
    var_9 = [var_8]
    var_10 = 'namespace.nested'

def test_case_0():
    var_0 = '/some/path'
    var_1 = [var_0]
    var_2 = ()
    var_3 = True
    var_4 = 'py'
    var_5 = {var_4}
    var_6 = frozenset(var_5)
    var_7 = '/some/path/namespace'
    var_8 = [var_7]
    var_9 = '/some/path/namespace/nested.py'
    var_10 = [var_9]
    var_11 = 'namespace.nested'

def test_case_0():
    var_0 = '/some/path'
    var_1 = [var_0]
    var_2 = ()
    var_3 = False
    var_4 = frozenset()
    var_5 = 'nonexistent'

def test_case_0():
    var_0 = '/some/path/module'
    var_1 = [var_0]
    var_2 = ()
    var_3 = False
    var_4 = frozenset()
    var_5 = [var_0]
    var_6 = '/some/path/module/__init__.py'
    var_7 = [var_6]
    var_8 = 'module'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_src_path_is_module_when_name_matches_and_is_directory_and_exists_case_sensitive. Retrieved 2/5 statements.
# Partially parsed test_src_path_is_not_module_when_name_does_not_match. Retrieved 3/6 statements.
# Partially parsed test_src_path_is_not_module_when_not_directory. Retrieved 2/5 statements.
# Partially parsed test_src_path_is_not_module_when_does_not_exist_case_sensitive. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'module_name'
    var_1 = [var_0]
    var_2 = True

def test_case_0():
    var_0 = 'different_name'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'module_name'

def test_case_0():
    var_0 = 'module_name'
    var_1 = [var_0]
    var_2 = False

def test_case_0():
    var_0 = 'module_name'
    var_1 = [var_0]
    var_2 = True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test__is_namespace_package_returns_false_when_path_is_not_a_package. Retrieved 4/7 statements.
# Partially parsed test__is_namespace_package_returns_false_when_init_file_exists_without_namespace_declaration. Retrieved 6/15 statements.
# Partially parsed test__is_namespace_package_returns_true_when_init_file_exists_with_pkg_resources_declaration. Retrieved 6/15 statements.
# Partially parsed test__is_namespace_package_returns_true_when_init_file_exists_with_pkgutil_declaration. Retrieved 6/15 statements.
# Partially parsed test__is_namespace_package_returns_false_when_no_init_file_and_has_source_files. Retrieved 6/15 statements.
# Partially parsed test__is_namespace_package_returns_false_when_no_init_file_and_has_setup_cfg. Retrieved 6/15 statements.
# Partially parsed test__is_namespace_package_returns_false_when_no_init_file_and_has_pyproject_toml. Retrieved 6/15 statements.
# Partially parsed test__is_namespace_package_returns_true_when_no_init_file_and_no_source_files. Retrieved 4/11 statements.


def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = {var_2}
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = 'test_package'
    var_1 = '__init__.py'
    var_2 = "print('hello')"
    var_3 = '.py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'test_package'
    var_1 = '__init__.py'
    var_2 = '__import__("pkg_resources").declare_namespace(__name__)'
    var_3 = '.py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'test_package'
    var_1 = '__init__.py'
    var_2 = '__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    var_3 = '.py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'test_package'
    var_1 = 'module.py'
    var_2 = "print('hello')"
    var_3 = '.py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'test_package'
    var_1 = 'setup.cfg'
    var_2 = '[metadata]\nname = test'
    var_3 = '.py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'test_package'
    var_1 = 'pyproject.toml'
    var_2 = "[build-system]\nrequires = ['setuptools']"
    var_3 = '.py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'test_package'
    var_1 = '.py'
    var_2 = {var_1}
    var_3 = frozenset(var_2)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test__src_path_with_module_in_src_path. Retrieved 2/8 statements.
# Partially parsed test__src_path_with_nested_module_in_src_path. Retrieved 5/12 statements.
# Partially parsed test__src_path_with_namespace_package. Retrieved 4/10 statements.
# Partially parsed test__src_path_with_auto_identified_namespace_package. Retrieved 9/17 statements.
# Partially parsed test__src_path_with_module_not_found. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_src_path_as_module. Retrieved 2/7 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = None
    var_1 = 'src_paths'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'module'
    var_5 = module_1._src_path(var_4, var_3)
    assert var_5 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = 'src_paths'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'module'
    var_5 = module_1._src_path(var_4, var_3)
    assert var_5 is None

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = 'module'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = 'parent'
    var_3 = 'child'
    var_4 = True
    var_5 = 'parent.child'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = 'namespace'
    var_3 = [var_2]
    var_4 = 'namespace.module'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = 'namespace'
    var_3 = 'module.py'
    var_4 = 'content'
    var_5 = True
    var_6 = '.py'
    var_7 = [var_6]
    var_8 = frozenset(var_7)
    var_9 = 'namespace.module'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = 'nonexistent'

def test_case_0():
    var_0 = '/path/to/module'
    var_1 = [var_0]
    var_2 = 'module'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_true. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = 'module'
    var_2 = {var_1}
    var_3 = '/path/to/src'
    var_4 = [var_3]
    var_5 = False
    var_6 = '.py'
    var_7 = [var_6]
    var_8 = [var_3]
    var_9 = ()



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_predicate_at_line_16_evaluates_to_true. Retrieved 8/21 statements.


def test_case_0():
    var_0 = 'module'
    var_1 = 'src'
    var_2 = [var_1]
    var_3 = []
    var_4 = False
    var_5 = [var_1]
    var_6 = ()
    var_7 = '.'
    var_8 = 1
    var_9 = *var_6



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_src_path_returns_none_when_module_not_found. Retrieved 2/6 statements.
# Partially parsed test_src_path_returns_firstparty_when_module_found_in_src_paths. Retrieved 2/7 statements.
# Partially parsed test_src_path_handles_nested_modules_with_namespace_packages. Retrieved 7/12 statements.
# Partially parsed test_src_path_handles_auto_identified_namespace_packages. Retrieved 6/11 statements.
# Partially parsed test_src_path_returns_firstparty_when_src_path_is_module. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'nonexistent_module'
    var_1 = '/fake/path'
    var_2 = [var_1]

def test_case_0():
    var_0 = 'existing_module'
    var_1 = '/real/path'
    var_2 = [var_1]

def test_case_0():
    var_0 = '/real/path'
    var_1 = [var_0]
    var_2 = 'parent'
    var_3 = {var_2}
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'parent.child'

def test_case_0():
    var_0 = '/real/path'
    var_1 = [var_0]
    var_2 = True
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)
    var_6 = 'auto_ns.child'

def test_case_0():
    var_0 = '/real/path/module'
    var_1 = [var_0]
    var_2 = 'module'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test__src_path_with_empty_name. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_nonexistent_module. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_module_in_src_paths. Retrieved 2/7 statements.
# Partially parsed test__src_path_with_package_in_src_paths. Retrieved 2/7 statements.
# Partially parsed test__src_path_with_nested_module_in_namespace_package. Retrieved 4/9 statements.
# Partially parsed test__src_path_with_auto_identified_namespace_package. Retrieved 3/8 statements.
# Partially parsed test__src_path_with_src_path_is_module. Retrieved 2/7 statements.
# Partially parsed test__src_path_with_custom_src_paths. Retrieved 2/7 statements.


def test_case_0():
    var_0 = ''
    var_1 = '/some/path'
    var_2 = [var_1]

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = '/some/path'
    var_2 = [var_1]

def test_case_0():
    var_0 = 'module'
    var_1 = '/some/path'
    var_2 = [var_1]

def test_case_0():
    var_0 = 'package'
    var_1 = '/some/path'
    var_2 = [var_1]

def test_case_0():
    var_0 = '/some/path'
    var_1 = [var_0]
    var_2 = 'namespace'
    var_3 = [var_2]
    var_4 = 'namespace.nested'

def test_case_0():
    var_0 = '/some/path'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'namespace.nested'

def test_case_0():
    var_0 = 'module'
    var_1 = '/some/path'
    var_2 = [var_1]

def test_case_0():
    var_0 = 'module'
    var_1 = '/custom/path'
    var_2 = [var_1]



# Parsed testcases at query #27
#--------------------------

# Partially parsed test__is_namespace_package_returns_true_when_package_and_init_file_has_declare_namespace. Retrieved 9/25 statements.


def test_case_0():
    var_0 = 'some_path'
    assert var_0 is True
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)
    var_5 = True
    var_6 = lambda p: var_5
    var_7 = []
    var_8 = '__init__.py'
    var_9 = b'__import__("pkg_resources").declare_namespace(__name__)'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test__is_namespace_package_predicate_true. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 'some_path'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = {var_2}
    var_4 = frozenset(var_3)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = '/path/to/src'
    var_2 = [var_1]
    var_3 = 'module'
    var_4 = [var_3]
    var_5 = False
    var_6 = [var_1]
    var_7 = ()



# Parsed testcases at query #30
#--------------------------

# Partially parsed test__is_module_returns_true_for_py_file. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'module.py'
    var_1 = [var_0]



# Parsed testcases at query #31
#--------------------------

# Partially parsed test__src_path_with_non_existent_module. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_existent_module. Retrieved 3/9 statements.
# Partially parsed test__src_path_with_nested_module. Retrieved 5/15 statements.
# Partially parsed test__src_path_with_namespace_package. Retrieved 8/18 statements.
# Partially parsed test__src_path_with_auto_identify_namespace_package. Retrieved 9/19 statements.
# Partially parsed test__src_path_with_src_path_is_module. Retrieved 3/9 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = None
    var_1 = 'src_paths'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'module'
    var_5 = module_1._src_path(var_4, var_3)
    assert var_5 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = 'src_paths'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'module'
    var_5 = module_1._src_path(var_4, var_3)
    assert var_5 is None

def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = [var_0]
    var_2 = 'module'

def test_case_0():
    var_0 = '/existent/path'
    var_1 = [var_0]
    var_2 = '/existent/path/module.py'
    var_3 = [var_2]
    var_4 = 'module'

def test_case_0():
    var_0 = '/existent/path'
    var_1 = [var_0]
    var_2 = '/existent/path/parent'
    var_3 = [var_2]
    var_4 = '/existent/path/parent/__init__.py'
    var_5 = [var_4]
    var_6 = '/existent/path/parent/child.py'
    var_7 = [var_6]
    var_8 = 'parent.child'

def test_case_0():
    var_0 = '/existent/path'
    var_1 = [var_0]
    var_2 = 'parent'
    var_3 = {var_2}
    var_4 = '/existent/path/parent'
    var_5 = [var_4]
    var_6 = '/existent/path/parent/__init__.py'
    var_7 = [var_6]
    var_8 = '__import__("pkg_resources").declare_namespace(__name__)'
    var_9 = '/existent/path/parent/child.py'
    var_10 = [var_9]
    var_11 = 'parent.child'

def test_case_0():
    var_0 = '/existent/path'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = '/existent/path/parent'
    var_6 = [var_5]
    var_7 = '/existent/path/parent/__init__.py'
    var_8 = [var_7]
    var_9 = '__import__("pkg_resources").declare_namespace(__name__)'
    var_10 = '/existent/path/parent/child.py'
    var_11 = [var_10]
    var_12 = 'parent.child'

def test_case_0():
    var_0 = '/existent/path/module'
    var_1 = [var_0]
    var_2 = '/existent/path/module/__init__.py'
    var_3 = [var_2]
    var_4 = 'module'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_src_path_finds_module_in_src_paths. Retrieved 7/11 statements.
# Partially parsed test_src_path_finds_nested_module. Retrieved 7/11 statements.
# Partially parsed test_src_path_returns_none_when_module_not_found. Retrieved 7/11 statements.
# Partially parsed test_src_path_handles_namespace_packages. Retrieved 8/12 statements.
# Partially parsed test_src_path_handles_auto_identified_namespace_packages. Retrieved 7/11 statements.
# Partially parsed test_src_path_handles_src_path_is_module. Retrieved 7/11 statements.


def test_case_0():
    var_0 = '/project/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)
    var_6 = False
    var_7 = 'module'

def test_case_0():
    var_0 = '/project/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)
    var_6 = False
    var_7 = 'parent.child'

def test_case_0():
    var_0 = '/project/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)
    var_6 = False
    var_7 = 'nonexistent'

def test_case_0():
    var_0 = '/project/src'
    var_1 = [var_0]
    var_2 = 'parent'
    var_3 = {var_2}
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = False
    var_8 = 'parent.child'

def test_case_0():
    var_0 = '/project/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)
    var_6 = True
    var_7 = 'parent.child'

def test_case_0():
    var_0 = '/project/src/module'
    var_1 = [var_0]
    var_2 = set()
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)
    var_6 = False
    var_7 = 'module'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = '/path/to/src'
    var_2 = [var_1]
    var_3 = 'module'
    var_4 = [var_3]
    var_5 = False
    var_6 = '.py'
    var_7 = [var_6]
    var_8 = [var_1]
    var_9 = ()



# Parsed testcases at query #34
#--------------------------

# Partially parsed test__src_path_is_module_returns_true. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'valid_module'
    var_1 = [var_0]



# Parsed testcases at query #35
#--------------------------

# Partially parsed test__src_path_is_module_returns_true. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'module_name'
    var_1 = [var_0]



# Parsed testcases at query #36
#--------------------------

# Partially parsed test__is_module_returns_true_for_py_file. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'module.py'
    var_1 = [var_0]



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_src_paths_is_not_none. Retrieved 3/6 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'some_path'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = 'module'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_predicate_at_line_16_evaluates_to_true. Retrieved 8/21 statements.


def test_case_0():
    var_0 = 'module'
    var_1 = 'src'
    var_2 = [var_1]
    var_3 = []
    var_4 = False
    var_5 = [var_1]
    var_6 = ()
    var_7 = '.'
    var_8 = 1
    var_9 = *var_6
    var_10 = [var_1]



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_predicate_at_line_16. Retrieved 7/17 statements.


def test_case_0():
    var_0 = 'module'
    var_1 = 'src'
    var_2 = [var_1]
    var_3 = []
    var_4 = False
    var_5 = 'src/module'
    var_6 = [var_5]
    var_7 = ()
    var_8 = 'module'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test__src_path_returns_none_when_no_matching_path. Retrieved 5/9 statements.
# Partially parsed test__src_path_returns_firstparty_when_module_found_in_src_paths. Retrieved 5/10 statements.
# Partially parsed test__src_path_handles_nested_modules_with_namespace_packages. Retrieved 6/11 statements.
# Partially parsed test__src_path_handles_src_path_is_module_case. Retrieved 5/10 statements.


def test_case_0():
    var_0 = '/fake/path'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = frozenset()
    var_5 = 'nonexistent.module'

def test_case_0():
    var_0 = '/fake/path'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = frozenset()
    var_5 = 'existing.module'

def test_case_0():
    var_0 = '/fake/path'
    var_1 = [var_0]
    var_2 = 'existing'
    var_3 = {var_2}
    var_4 = False
    var_5 = frozenset()
    var_6 = 'existing.nested.module'

def test_case_0():
    var_0 = '/fake/path'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = frozenset()
    var_5 = 'existing.module'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test__src_path_with_nested_module_in_namespace_package. Retrieved 6/10 statements.
# Partially parsed test__src_path_with_module_in_src_paths. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_package_in_src_paths. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_src_path_is_module. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_no_match. Retrieved 2/6 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'test'
    var_3 = {var_2}
    var_4 = 'py'
    var_5 = {var_4}
    var_6 = 'test.module'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'module'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'package'

def test_case_0():
    var_0 = '/src/module'
    var_1 = [var_0]
    var_2 = 'module'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'nonexistent'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test__is_namespace_package_returns_true_when_package_without_init_file. Retrieved 7/9 statements.


def test_case_0():
    var_0 = 'some_package'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = '.pyx'
    var_4 = [var_2, var_3]
    var_5 = frozenset(var_4)
    var_6 = True
    var_7 = lambda p: var_6



# Parsed testcases at query #43
#--------------------------

# Partially parsed test__is_module_returns_true_for_py_file. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'module.py'
    var_1 = [var_0]



# Parsed testcases at query #44
#--------------------------

# Partially parsed test__is_module_returns_true_for_py_file. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'test_module.py'
    var_1 = [var_0]



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 9/15 statements.


def test_case_0():
    var_0 = 'test_module.submodule'
    var_1 = '/path/to/src'
    var_2 = [var_1]
    var_3 = 'test_module'
    var_4 = [var_3]
    var_5 = False
    var_6 = '.py'
    var_7 = [var_6]
    var_8 = '/path/to/src/test_module'
    var_9 = [var_8]
    var_10 = ()
    var_11 = 'Found in one of the configured src_paths:'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'module'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = ''



# Parsed testcases at query #47
#--------------------------

# Partially parsed test__is_namespace_package_returns_false_when_not_package. Retrieved 4/7 statements.


def test_case_0():
    var_0 = '/some/path'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_src_path_is_module_returns_true_when_conditions_met. Retrieved 1/5 statements.
# Partially parsed test_src_path_is_module_returns_false_when_name_mismatch. Retrieved 2/6 statements.
# Partially parsed test_src_path_is_module_returns_false_when_not_directory. Retrieved 1/5 statements.
# Partially parsed test_src_path_is_module_returns_false_when_case_sensitive_mismatch. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]
    var_2 = 'other_module'

def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'Test_Module'
    var_1 = [var_0]
    var_2 = 'test_module'



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_src_path_is_module_when_module_name_matches_path_name_and_is_directory_and_exists_case_sensitive. Retrieved 3/6 statements.
# Partially parsed test_src_path_is_not_module_when_module_name_does_not_match_path_name. Retrieved 3/6 statements.
# Partially parsed test_src_path_is_not_module_when_path_is_not_directory. Retrieved 3/6 statements.
# Partially parsed test_src_path_is_not_module_when_path_does_not_exist_case_sensitive. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'module_name'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'module_name'

def test_case_0():
    var_0 = 'different_name'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'module_name'

def test_case_0():
    var_0 = 'module_name'
    var_1 = [var_0]
    var_2 = False
    var_3 = 'module_name'

def test_case_0():
    var_0 = 'module_name'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'module_name'



