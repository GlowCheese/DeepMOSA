####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_known_pattern_matches_first_segment. Retrieved 4/9 statements.
# Partially parsed test_known_pattern_matches_middle_segment. Retrieved 4/9 statements.
# Partially parsed test_known_pattern_matches_last_segment. Retrieved 4/9 statements.
# Partially parsed test_known_pattern_no_match. Retrieved 4/9 statements.
# Partially parsed test_known_pattern_placement_not_in_sections. Retrieved 5/10 statements.
# Partially parsed test_known_pattern_empty_name. Retrieved 4/9 statements.
# Partially parsed test_known_pattern_single_segment_match. Retrieved 3/8 statements.
# Partially parsed test_known_pattern_single_segment_no_match. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'foo'
    var_1 = 'placement1'
    var_2 = [var_1]
    var_3 = 'foo.bar.baz'

def test_case_0():
    var_0 = 'bar'
    var_1 = 'placement2'
    var_2 = [var_1]
    var_3 = 'foo.bar.baz'

def test_case_0():
    var_0 = 'baz'
    var_1 = 'placement3'
    var_2 = [var_1]
    var_3 = 'foo.bar.baz'

def test_case_0():
    var_0 = 'qux'
    var_1 = 'placement4'
    var_2 = [var_1]
    var_3 = 'foo.bar.baz'

def test_case_0():
    var_0 = 'foo'
    var_1 = 'placement5'
    var_2 = 'placement1'
    var_3 = [var_2]
    var_4 = 'foo.bar.baz'

def test_case_0():
    var_0 = 'foo'
    var_1 = 'placement1'
    var_2 = [var_1]
    var_3 = ''

def test_case_0():
    var_0 = 'single'
    var_1 = 'placement1'
    var_2 = [var_1]

def test_case_0():
    var_0 = 'other'
    var_1 = 'placement1'
    var_2 = [var_1]
    var_3 = 'single'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_known_pattern_predicate_false. Retrieved 7/9 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.*'
    var_3 = 'section1'
    var_4 = (var_2, var_3)
    var_5 = 'section2'
    var_6 = 'test.module'
    var_7 = module_1._known_pattern(var_6, var_1)
    assert var_7 is None



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_src_path_with_non_existing_module. Retrieved 2/6 statements.
# Partially parsed test_src_path_with_existing_module_in_src_paths. Retrieved 3/10 statements.
# Partially parsed test_src_path_with_nested_module_in_namespace_package. Retrieved 5/16 statements.
# Partially parsed test_src_path_with_auto_identify_namespace_package. Retrieved 7/18 statements.
# Partially parsed test_src_path_with_src_path_is_module. Retrieved 3/14 statements.
# Partially parsed test_src_path_with_prefix. Retrieved 6/22 statements.
# Partially parsed test_src_path_with_non_dir_module_in_root. Retrieved 3/10 statements.


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
    var_0 = '/tmp'
    var_1 = [var_0]
    var_2 = 'non_existing_module'

def test_case_0():
    var_0 = '/tmp'
    var_1 = [var_0]
    var_2 = [var_0]
    var_3 = 'existing_module.py'
    var_4 = 'existing_module'

def test_case_0():
    var_0 = '/tmp'
    var_1 = [var_0]
    var_2 = 'parent'
    var_3 = {var_2}
    var_4 = [var_0]
    var_5 = [var_0]
    var_6 = 'child.py'
    var_7 = 'parent.child'

def test_case_0():
    var_0 = '/tmp'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = [var_0]
    var_6 = 'parent'
    var_7 = [var_0]
    var_8 = 'child.py'
    var_9 = 'parent.child'

def test_case_0():
    var_0 = '/tmp'
    var_1 = [var_0]
    var_2 = [var_0]
    var_3 = 'module'
    var_4 = [var_0]
    var_5 = '__init__.py'

def test_case_0():
    var_0 = '/tmp'
    var_1 = [var_0]
    var_2 = [var_0]
    var_3 = 'parent'
    var_4 = [var_0]
    var_5 = 'child'
    var_6 = [var_0]
    var_7 = '__init__.py'
    var_8 = 'parent.child'
    var_9 = (var_3,)

def test_case_0():
    var_0 = '/tmp'
    var_1 = [var_0]
    var_2 = [var_0]
    var_3 = 'module.py'
    var_4 = 'module'



# Parsed testcases at query #4
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test*'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'test_file'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('test*', 'Matched forced_separate (test*) config value.'))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test*'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '.test_file'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('test*', 'Matched forced_separate (test*) config value.'))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'other*'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'test_file'
    var_6 = module_1._forced_separate(var_5, var_4)
    assert var_6 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'test_file'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('test', 'Matched forced_separate (test) config value.'))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = 'forced_separate'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'test_file'
    var_5 = module_1._forced_separate(var_4, var_3)
    assert var_5 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test*'
    var_1 = 'other*'
    var_2 = [var_0, var_1]
    var_3 = 'forced_separate'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = 'other_file'
    var_7 = module_1._forced_separate(var_6, var_5)
    var_8 = bool(var_7 == ('other*', 'Matched forced_separate (other*) config value.'))
    assert var_8 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test__src_path_with_non_existent_module. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_module_in_src_paths. Retrieved 2/7 statements.
# Partially parsed test__src_path_with_nested_module_in_namespace_package. Retrieved 4/9 statements.
# Partially parsed test__src_path_with_module_in_root_src_path. Retrieved 2/7 statements.
# Partially parsed test__src_path_with_auto_identify_namespace_package. Retrieved 6/11 statements.


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
    var_0 = '/path/to/module'
    var_1 = [var_0]
    var_2 = 'module'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = True
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)
    var_6 = 'parent.child'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test__src_path_with_nested_module_and_namespace_package. Retrieved 7/11 statements.
# Partially parsed test__src_path_with_nested_module_and_auto_identify_namespace_package. Retrieved 6/10 statements.
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
    var_6 = frozenset(var_5)
    var_7 = 'test.nested'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)
    var_6 = 'test.nested'

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



# Parsed testcases at query #7
#--------------------------

# Partially parsed test__is_namespace_package_returns_false_when_not_a_package. Retrieved 4/6 statements.
# Partially parsed test__is_namespace_package_returns_false_when_init_file_exists_without_namespace_declaration. Retrieved 7/12 statements.
# Partially parsed test__is_namespace_package_returns_true_when_init_file_exists_with_pkg_resources_declaration_single_quotes. Retrieved 7/12 statements.
# Partially parsed test__is_namespace_package_returns_true_when_init_file_exists_with_pkg_resources_declaration_double_quotes. Retrieved 7/12 statements.
# Partially parsed test__is_namespace_package_returns_true_when_init_file_exists_with_pkgutil_declaration_single_quotes. Retrieved 7/12 statements.
# Partially parsed test__is_namespace_package_returns_true_when_init_file_exists_with_pkgutil_declaration_double_quotes. Retrieved 7/12 statements.
# Partially parsed test__is_namespace_package_returns_false_when_no_init_file_and_source_files_exist. Retrieved 7/12 statements.
# Partially parsed test__is_namespace_package_returns_false_when_no_init_file_and_setup_cfg_exists. Retrieved 7/12 statements.
# Partially parsed test__is_namespace_package_returns_false_when_no_init_file_and_pyproject_toml_exists. Retrieved 7/12 statements.
# Partially parsed test__is_namespace_package_returns_true_when_no_init_file_and_no_source_files. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 'not_a_package'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = 'package_with_init'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = "print('hello')"
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = "__import__('pkg_resources').declare_namespace(__name__)"
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = '__import__("pkg_resources").declare_namespace(__name__)'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = "__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = '__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = 'package_with_files'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'module.py'
    var_4 = "print('hello')"
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = 'package_with_setup'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'setup.cfg'
    var_4 = '[metadata]\nname = test'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = 'package_with_pyproject'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'pyproject.toml'
    var_4 = "[build-system]\nrequires = ['setuptools']"
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = 'empty_package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_predicate_at_line_16. Retrieved 7/19 statements.


def test_case_0():
    var_0 = 'module'
    var_1 = 'src'
    var_2 = [var_1]
    var_3 = []
    var_4 = False
    var_5 = 'src/module'
    var_6 = [var_5]
    var_7 = ()
    var_8 = [var_5]
    var_9 = [var_1]
    var_10 = 'module'



# Parsed testcases at query #9
#--------------------------




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
    var_7 = bool(var_6 == ('test', 'Matched forced_separate (test) config value.'))
    assert var_7 is True
    var_8 = '.testfile'
    var_9 = module_1._forced_separate(var_8, var_4)
    var_10 = bool(var_9 == ('test', 'Matched forced_separate (test) config value.'))
    assert var_10 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_src_path_is_module_returns_true_for_valid_module. Retrieved 2/5 statements.
# Partially parsed test_src_path_is_module_returns_false_for_non_directory. Retrieved 2/5 statements.
# Partially parsed test_src_path_is_module_returns_false_for_name_mismatch. Retrieved 3/6 statements.
# Partially parsed test_src_path_is_module_returns_false_for_case_sensitive_mismatch. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'valid_module'
    var_1 = [var_0]
    var_2 = True

def test_case_0():
    var_0 = 'not_a_directory'
    var_1 = [var_0]
    var_2 = False

def test_case_0():
    var_0 = 'different_name'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'module_name'

def test_case_0():
    var_0 = 'MODULE_NAME'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'module_name'



# Parsed testcases at query #11
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



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_known_pattern_predicate_false. Retrieved 6/8 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'pattern'
    var_3 = 'placement'
    var_4 = (var_2, var_3)
    var_5 = 'test.module'
    var_6 = module_1._known_pattern(var_5, var_1)
    assert var_6 is None



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_true. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 'test.namespace'
    var_1 = {var_0}
    var_2 = False
    var_3 = '/test/path'
    var_4 = [var_3]
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = 'test.namespace.module'
    var_8 = [var_3]
    var_9 = ()



# Parsed testcases at query #14
#--------------------------

# Partially parsed test__src_path_with_empty_name. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_module_in_src_paths. Retrieved 4/11 statements.
# Partially parsed test__src_path_with_package_in_src_paths. Retrieved 5/13 statements.
# Partially parsed test__src_path_with_nested_module_in_namespace_package. Retrieved 7/15 statements.
# Partially parsed test__src_path_with_auto_identified_namespace_package. Retrieved 7/15 statements.
# Partially parsed test__src_path_with_src_path_is_module. Retrieved 4/12 statements.


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
    var_0 = '/some/path'
    var_1 = [var_0]
    var_2 = ''

def test_case_0():
    var_0 = '/some/path'
    var_1 = [var_0]
    var_2 = '/some/path/module.py'
    var_3 = [var_2]
    var_4 = True
    var_5 = 'module'

def test_case_0():
    var_0 = '/some/path'
    var_1 = [var_0]
    var_2 = '/some/path/package'
    var_3 = [var_2]
    var_4 = True
    var_5 = '__init__.py'
    var_6 = 'package'

def test_case_0():
    var_0 = '/some/path'
    var_1 = [var_0]
    var_2 = 'parent'
    var_3 = {var_2}
    var_4 = '/some/path/parent'
    var_5 = [var_4]
    var_6 = True
    var_7 = 'child.py'
    var_8 = 'parent.child'

def test_case_0():
    var_0 = '/some/path'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = '/some/path/parent'
    var_6 = [var_5]
    var_7 = 'child.py'
    var_8 = 'parent.child'

def test_case_0():
    var_0 = '/some/path/module'
    var_1 = [var_0]
    var_2 = [var_0]
    var_3 = True
    var_4 = '__init__.py'
    var_5 = 'module'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test__src_path_with_nested_module. Retrieved 6/10 statements.


def test_case_0():
    var_0 = '/test'
    var_1 = [var_0]
    var_2 = 'test'
    var_3 = {var_2}
    var_4 = 'py'
    var_5 = {var_4}
    var_6 = 'test.module'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test__src_path_with_exact_match_in_src_paths. Retrieved 7/11 statements.
# Partially parsed test__src_path_with_nested_module. Retrieved 7/11 statements.
# Partially parsed test__src_path_with_namespace_package. Retrieved 8/12 statements.
# Partially parsed test__src_path_with_auto_identified_namespace_package. Retrieved 7/11 statements.
# Partially parsed test__src_path_with_no_match. Retrieved 7/11 statements.


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



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]
    var_2 = 'test_module.py'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_is_module_returns_true_for_py_file. Retrieved 1/3 statements.
# Partially parsed test_is_module_returns_true_for_extension_suffix. Retrieved 1/3 statements.
# Partially parsed test_is_module_returns_true_for_init_file. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'example.py'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'example.so'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'example/__init__.py'
    var_1 = [var_0]



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 2/5 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 1/6 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 2/5 statements.
# Partially parsed test_is_module_without_any_file. Retrieved 1/3 statements.


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



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_known_pattern_predicate_false. Retrieved 11/13 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'pattern1'
    var_3 = 'placement1'
    var_4 = (var_2, var_3)
    var_5 = 'pattern2'
    var_6 = 'placement2'
    var_7 = (var_5, var_6)
    var_8 = 'section1'
    var_9 = 'section2'
    var_10 = 'test.module.name'
    var_11 = module_1._known_pattern(var_10, var_1)
    assert var_11 is None



# Parsed testcases at query #21
#--------------------------

# Partially parsed test__is_module_returns_true_for_py_file. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'module.py'
    var_1 = [var_0]



# Parsed testcases at query #22
#--------------------------

# Partially parsed test__src_path_finds_module_in_src_paths. Retrieved 2/6 statements.
# Partially parsed test__src_path_finds_nested_module. Retrieved 2/6 statements.
# Partially parsed test__src_path_finds_namespace_package. Retrieved 4/8 statements.
# Partially parsed test__src_path_returns_none_when_not_found. Retrieved 2/6 statements.
# Partially parsed test__src_path_finds_module_in_custom_src_paths. Retrieved 2/8 statements.
# Partially parsed test__src_path_finds_module_with_prefix. Retrieved 4/8 statements.


def test_case_0():
    var_0 = '/project/src'
    var_1 = [var_0]
    var_2 = 'module'

def test_case_0():
    var_0 = '/project/src'
    var_1 = [var_0]
    var_2 = 'parent.child'

def test_case_0():
    var_0 = '/project/src'
    var_1 = [var_0]
    var_2 = 'parent'
    var_3 = {var_2}
    var_4 = 'parent.child'

def test_case_0():
    var_0 = '/project/src'
    var_1 = [var_0]
    var_2 = 'nonexistent'

def test_case_0():
    var_0 = '/custom/src'
    var_1 = [var_0]
    var_2 = 'module'
    var_3 = [var_0]

def test_case_0():
    var_0 = '/project/src'
    var_1 = [var_0]
    var_2 = 'module'
    var_3 = 'parent'
    var_4 = (var_3,)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test__src_path_with_module_in_src_paths. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_nested_module_in_src_paths. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_namespace_package_in_src_paths. Retrieved 4/8 statements.
# Partially parsed test__src_path_with_auto_identify_namespace_packages. Retrieved 3/7 statements.
# Partially parsed test__src_path_with_module_not_in_src_paths. Retrieved 2/6 statements.


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
    var_3 = 'parent.child'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = 'nonexistent'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test__is_namespace_package_returns_true_when_package_has_no_init_file_and_no_source_files. Retrieved 3/9 statements.


def test_case_0():
    var_0 = '.py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_false. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = '/path/to/src'
    var_2 = [var_1]
    var_3 = []
    var_4 = False
    var_5 = [var_1]
    var_6 = ()



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 2/5 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 1/6 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 2/5 statements.
# Partially parsed test_is_module_no_match. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'module'
    var_1 = [var_0]
    var_2 = 'module.py'

def test_case_0():
    var_0 = 'module'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'module'
    var_1 = [var_0]
    var_2 = 'module/__init__.py'

def test_case_0():
    var_0 = 'not_a_module'
    var_1 = [var_0]



# Parsed testcases at query #27
#--------------------------

# Partially parsed test__src_path_simple_module. Retrieved 7/11 statements.
# Partially parsed test__src_path_nested_module. Retrieved 7/11 statements.
# Partially parsed test__src_path_namespace_package. Retrieved 8/12 statements.
# Partially parsed test__src_path_auto_identify_namespace. Retrieved 7/11 statements.
# Partially parsed test__src_path_not_found. Retrieved 7/11 statements.


def test_case_0():
    var_0 = '/project'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'module'

def test_case_0():
    var_0 = '/project'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'parent.child'

def test_case_0():
    var_0 = '/project'
    var_1 = [var_0]
    var_2 = 'parent'
    var_3 = {var_2}
    var_4 = False
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)
    var_8 = 'parent.child'

def test_case_0():
    var_0 = '/project'
    var_1 = [var_0]
    var_2 = set()
    var_3 = True
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'parent.child'

def test_case_0():
    var_0 = '/project'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'nonexistent'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_true. Retrieved 10/22 statements.


def test_case_0():
    var_0 = 'test_namespace'
    var_1 = {var_0}
    var_2 = False
    var_3 = '/test/src'
    var_4 = [var_3]
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = 'test_namespace.submodule'
    var_8 = [var_3]
    var_9 = ()
    var_10 = (var_0,)
    var_11 = 'test_namespace'
    var_12 = [var_3]



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_true. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = 'module'
    var_2 = [var_1]
    var_3 = '/path/to/src'
    var_4 = [var_3]
    var_5 = False
    var_6 = '.py'
    var_7 = [var_6]
    var_8 = [var_3]
    var_9 = ()



# Parsed testcases at query #30
#--------------------------

# Partially parsed test__src_path_with_empty_name. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_non_existent_module. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_existing_module. Retrieved 3/9 statements.
# Partially parsed test__src_path_with_nested_module. Retrieved 4/11 statements.
# Partially parsed test__src_path_with_namespace_package. Retrieved 6/13 statements.
# Partially parsed test__src_path_with_auto_identified_namespace_package. Retrieved 8/16 statements.
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

def test_case_0():
    var_0 = '/tmp'
    var_1 = [var_0]
    var_2 = ''

def test_case_0():
    var_0 = '/tmp'
    var_1 = [var_0]
    var_2 = 'non_existent_module'

def test_case_0():
    var_0 = '/tmp'
    var_1 = [var_0]
    var_2 = '/tmp/existing_module.py'
    var_3 = [var_2]
    var_4 = 'existing_module'

def test_case_0():
    var_0 = '/tmp'
    var_1 = [var_0]
    var_2 = '/tmp/parent/child.py'
    var_3 = [var_2]
    var_4 = True
    var_5 = 'parent.child'

def test_case_0():
    var_0 = '/tmp'
    var_1 = [var_0]
    var_2 = 'parent'
    var_3 = [var_2]
    var_4 = '/tmp/parent/child.py'
    var_5 = [var_4]
    var_6 = True
    var_7 = 'parent.child'

def test_case_0():
    var_0 = '/tmp'
    var_1 = [var_0]
    var_2 = True
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)
    var_6 = '/tmp/parent'
    var_7 = [var_6]
    var_8 = 'child.py'
    var_9 = 'parent.child'

def test_case_0():
    var_0 = '/tmp/module'
    var_1 = [var_0]
    var_2 = [var_0]
    var_3 = True
    var_4 = 'module'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_predicate_at_line_26. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'src'
    var_1 = [var_0]
    var_2 = []
    var_3 = False
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = 'module.submodule'
    var_7 = [var_0]
    var_8 = ()



# Parsed testcases at query #32
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



# Parsed testcases at query #33
#--------------------------

# Partially parsed test__is_namespace_package_returns_true_when_package_has_no_init_file_and_no_source_files. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'test_package'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test__is_namespace_package_returns_false_when_not_a_package. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 'not_a_package'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test__src_path_is_module_returns_true. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'module'
    var_1 = [var_0]



# Parsed testcases at query #36
#--------------------------

# Partially parsed test__src_path_with_empty_name. Retrieved 5/9 statements.
# Partially parsed test__src_path_with_nested_module_and_namespace_package. Retrieved 6/14 statements.
# Partially parsed test__src_path_with_nested_module_and_auto_identify_namespace_package. Retrieved 7/16 statements.
# Partially parsed test__src_path_with_module_in_src_paths. Retrieved 5/13 statements.
# Partially parsed test__src_path_with_package_in_src_paths. Retrieved 5/13 statements.
# Partially parsed test__src_path_with_src_path_is_module. Retrieved 5/13 statements.


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

def test_case_0():
    var_0 = '/path'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = frozenset()
    var_5 = ''

def test_case_0():
    var_0 = '/path'
    var_1 = [var_0]
    var_2 = 'parent'
    var_3 = {var_2}
    var_4 = False
    var_5 = frozenset()
    var_6 = 'parent.child'

def test_case_0():
    var_0 = '/path'
    var_1 = [var_0]
    var_2 = set()
    var_3 = True
    var_4 = '.py'
    var_5 = {var_4}
    var_6 = frozenset(var_5)
    var_7 = 'parent.child'

def test_case_0():
    var_0 = '/path'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = frozenset()
    var_5 = 'module'

def test_case_0():
    var_0 = '/path'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = frozenset()
    var_5 = 'package'

def test_case_0():
    var_0 = '/path/module'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = frozenset()
    var_5 = 'module'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test__is_namespace_package_returns_false_when_not_a_package. Retrieved 4/6 statements.
# Partially parsed test__is_namespace_package_returns_false_when_init_file_exists_without_namespace_declaration. Retrieved 7/14 statements.
# Partially parsed test__is_namespace_package_returns_false_when_init_file_exists_with_pkg_resources_declaration. Retrieved 7/14 statements.
# Partially parsed test__is_namespace_package_returns_false_when_init_file_exists_with_pkgutil_declaration. Retrieved 7/14 statements.
# Partially parsed test__is_namespace_package_returns_false_when_no_init_file_but_has_source_files. Retrieved 7/14 statements.
# Partially parsed test__is_namespace_package_returns_false_when_no_init_file_but_has_setup_cfg. Retrieved 7/14 statements.
# Partially parsed test__is_namespace_package_returns_false_when_no_init_file_but_has_pyproject_toml. Retrieved 7/14 statements.
# Partially parsed test__is_namespace_package_returns_true_when_no_init_file_and_no_source_files. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'non_existent_path'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = 'test_package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = "print('hello')"
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = 'test_package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = '__import__("pkg_resources").declare_namespace(__name__)'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = 'test_package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = '__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = 'test_package'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'module.py'
    var_4 = "print('hello')"
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = 'test_package'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'setup.cfg'
    var_4 = '[metadata]\nname = test_package'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = 'test_package'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'pyproject.toml'
    var_4 = "[build-system]\nrequires = ['setuptools']"
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = 'test_package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = '/path/to/src'
    var_2 = [var_1]
    var_3 = 'module'
    var_4 = {var_3}
    var_5 = False
    var_6 = '.py'
    var_7 = [var_6]
    var_8 = [var_1]
    var_9 = ()



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_known_pattern_predicate_false. Retrieved 7/9 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test.module'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = 'test.*'
    var_4 = 'placement'
    var_5 = (var_3, var_4)
    var_6 = 'other_section'
    var_7 = module_1._known_pattern(var_0, var_2)
    assert var_7 is None



# Parsed testcases at query #40
#--------------------------

# Partially parsed test__src_path_with_module_in_src_paths. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_nested_module_in_namespace_package. Retrieved 4/8 statements.
# Partially parsed test__src_path_with_auto_identify_namespace_packages. Retrieved 6/10 statements.
# Partially parsed test__src_path_with_src_path_is_module. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_no_match. Retrieved 2/6 statements.


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
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)
    var_6 = 'parent.child'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = 'src'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = 'nonexistent'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_false. Retrieved 9/24 statements.


def test_case_0():
    var_0 = ()
    var_1 = False
    var_2 = 'src'
    var_3 = [var_2]
    var_4 = '.py'
    var_5 = (var_4,)
    var_6 = 'module.submodule'
    var_7 = [var_2]
    var_8 = ()
    var_9 = 'module'
    var_10 = 'module'



# Parsed testcases at query #42
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



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = '/path/to/src'
    var_2 = [var_1]
    var_3 = 'module'
    var_4 = [var_3]
    var_5 = False



# Parsed testcases at query #44
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



# Parsed testcases at query #45
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



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_true. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 'test.namespace'
    var_1 = {var_0}
    var_2 = False
    var_3 = '/test/path'
    var_4 = [var_3]
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = 'test.namespace.module'
    var_8 = [var_3]
    var_9 = ()



# Parsed testcases at query #47
#--------------------------

# Partially parsed test__is_module_returns_true_for_py_file. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'test_module.py'
    var_1 = [var_0]



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_true. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 'test.namespace'
    var_1 = {var_0}
    var_2 = False
    var_3 = '.py'
    var_4 = {var_3}
    var_5 = '/test/path'
    var_6 = [var_5]
    var_7 = 'test.namespace.module'
    var_8 = [var_5]
    var_9 = ()



# Parsed testcases at query #49
#--------------------------

# Partially parsed test__is_namespace_package_returns_true_for_valid_namespace_package. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 'valid_namespace_package'
    var_1 = [var_0]
    var_2 = '__init__.py'
    var_3 = b'__import__("pkg_resources").declare_namespace(__name__)'
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)



# Parsed testcases at query #50
#--------------------------

# Partially parsed test__is_namespace_package_with_namespace_declaration. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 'test_namespace'
    var_1 = [var_0]
    var_2 = '__init__.py'
    var_3 = "__import__('pkg_resources').declare_namespace(__name__)"
    var_4 = 'py'
    var_5 = {var_4}
    var_6 = frozenset(var_5)



# Parsed testcases at query #51
#--------------------------

# Partially parsed test__is_namespace_package_with_non_package_path. Retrieved 4/6 statements.
# Partially parsed test__is_namespace_package_with_package_having_init_file. Retrieved 7/12 statements.
# Partially parsed test__is_namespace_package_with_package_having_pkg_resources_declaration. Retrieved 7/12 statements.
# Partially parsed test__is_namespace_package_with_package_having_pkgutil_declaration. Retrieved 7/12 statements.
# Partially parsed test__is_namespace_package_with_package_having_py_files. Retrieved 7/12 statements.
# Partially parsed test__is_namespace_package_with_package_having_setup_cfg. Retrieved 7/12 statements.
# Partially parsed test__is_namespace_package_with_package_having_pyproject_toml. Retrieved 7/12 statements.
# Partially parsed test__is_namespace_package_with_empty_package. Retrieved 5/8 statements.


def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/existing/package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = '# Regular package'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/existing/package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = '__import__("pkg_resources").declare_namespace(__name__)'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/existing/package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = '__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/existing/package'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'module.py'
    var_4 = '# Some code'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/existing/package'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'setup.cfg'
    var_4 = '[metadata]\nname=package'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/existing/package'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'pyproject.toml'
    var_4 = '[build-system]\nrequires = []'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/existing/package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)



# Parsed testcases at query #52
#--------------------------

# Partially parsed test__is_namespace_package_with_empty_directory. Retrieved 3/8 statements.
# Partially parsed test__is_namespace_package_with_non_package. Retrieved 6/15 statements.
# Partially parsed test__is_namespace_package_with_init_file. Retrieved 6/15 statements.
# Partially parsed test__is_namespace_package_with_namespace_declaration. Retrieved 6/15 statements.
# Partially parsed test__is_namespace_package_with_pkgutil_declaration. Retrieved 6/15 statements.
# Partially parsed test__is_namespace_package_with_py_files. Retrieved 6/15 statements.
# Partially parsed test__is_namespace_package_with_setup_cfg. Retrieved 6/15 statements.
# Partially parsed test__is_namespace_package_with_pyproject_toml. Retrieved 6/15 statements.


def test_case_0():
    var_0 = '.py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)

def test_case_0():
    var_0 = 'non_package'
    var_1 = 'file.txt'
    var_2 = 'content'
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'package'
    var_1 = '__init__.py'
    var_2 = 'content'
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = '__init__.py'
    var_2 = "__import__('pkg_resources').declare_namespace(__name__)"
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = '__init__.py'
    var_2 = "__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'package'
    var_1 = 'module.py'
    var_2 = 'content'
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'package'
    var_1 = 'setup.cfg'
    var_2 = 'content'
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'package'
    var_1 = 'pyproject.toml'
    var_2 = 'content'
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)



# Parsed testcases at query #53
#--------------------------

# Partially parsed test__is_namespace_package_returns_false_when_not_package. Retrieved 4/7 statements.


def test_case_0():
    var_0 = '/non/package/path'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)



# Parsed testcases at query #54
#--------------------------

# Partially parsed test__is_namespace_package_returns_false_when_not_package. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 'not_a_package'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)



# Parsed testcases at query #55
#--------------------------

# Partially parsed test__is_namespace_package_returns_false_when_not_package. Retrieved 4/6 statements.


def test_case_0():
    var_0 = '/non/package'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_false. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = ()
    var_2 = False
    var_3 = '/some/path'
    var_4 = [var_3]
    var_5 = '.py'
    var_6 = (var_5,)
    var_7 = [var_3]
    var_8 = ()



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'module'
    var_1 = [var_0]
    var_2 = 'module.py'



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = 'src'
    var_2 = [var_1]
    var_3 = 'module'
    var_4 = [var_3]
    var_5 = False
    var_6 = 'src/module'
    var_7 = [var_6]
    var_8 = ()



# Parsed testcases at query #59
#--------------------------

# Partially parsed test__src_path_is_module_returns_true_when_conditions_are_met. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'example_module'
    assert var_0 is True
    var_1 = [var_0]
    var_2 = True
    var_3 = 'example_module'



# Parsed testcases at query #60
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



# Parsed testcases at query #61
#--------------------------

# Partially parsed test__is_namespace_package_returns_false_for_non_package. Retrieved 4/7 statements.
# Partially parsed test__is_namespace_package_returns_false_for_package_with_init. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_false_for_package_with_py_files. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_false_for_package_with_setup_cfg. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_false_for_package_with_pyproject_toml. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_true_for_namespace_package_with_pkg_resources. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_true_for_namespace_package_with_pkgutil. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_true_for_empty_package. Retrieved 5/9 statements.


def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/existing/package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = '# Regular __init__.py'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/existing/package'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'module.py'
    var_4 = '# Some module'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/existing/package'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'setup.cfg'
    var_4 = '[metadata]'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/existing/package'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'pyproject.toml'
    var_4 = '[build-system]'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/existing/package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = '__import__("pkg_resources").declare_namespace(__name__)'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/existing/package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = '__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/existing/package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)



# Parsed testcases at query #62
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



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_src_path_with_empty_name. Retrieved 2/6 statements.
# Partially parsed test_src_path_with_non_existent_module. Retrieved 2/6 statements.
# Partially parsed test_src_path_with_existing_module_file. Retrieved 3/10 statements.
# Partially parsed test_src_path_with_existing_package. Retrieved 4/15 statements.
# Partially parsed test_src_path_with_nested_module. Retrieved 5/20 statements.
# Partially parsed test_src_path_with_namespace_package. Retrieved 5/12 statements.
# Partially parsed test_src_path_with_auto_identify_namespace_package. Retrieved 7/14 statements.
# Partially parsed test_src_path_with_src_path_is_module. Retrieved 3/14 statements.


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
    var_0 = '/some/path'
    var_1 = [var_0]
    var_2 = ''

def test_case_0():
    var_0 = '/some/path'
    var_1 = [var_0]
    var_2 = 'non_existent_module'

def test_case_0():
    var_0 = '/some/path'
    var_1 = [var_0]
    var_2 = '/some/path/module.py'
    var_3 = [var_2]
    var_4 = 'module'

def test_case_0():
    var_0 = '/some/path'
    var_1 = [var_0]
    var_2 = '/some/path/package'
    var_3 = [var_2]
    var_4 = '__init__.py'
    var_5 = 'package'

def test_case_0():
    var_0 = '/some/path'
    var_1 = [var_0]
    var_2 = '/some/path/package'
    var_3 = [var_2]
    var_4 = '__init__.py'
    var_5 = 'nested.py'
    var_6 = 'package.nested'

def test_case_0():
    var_0 = '/some/path'
    var_1 = [var_0]
    var_2 = 'package'
    var_3 = {var_2}
    var_4 = '/some/path/package'
    var_5 = [var_4]
    var_6 = 'package.nested'

def test_case_0():
    var_0 = '/some/path'
    var_1 = [var_0]
    var_2 = True
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)
    var_6 = '/some/path/package'
    var_7 = [var_6]
    var_8 = 'package.nested'

def test_case_0():
    var_0 = '/some/path/module'
    var_1 = [var_0]
    var_2 = [var_0]
    var_3 = '__init__.py'
    var_4 = 'module'



# Parsed testcases at query #2
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
    var_0 = 'module'
    var_1 = [var_0]
    var_2 = 'module/__init__.py'

def test_case_0():
    var_0 = 'non_existent_module'
    var_1 = [var_0]



# Parsed testcases at query #3
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test*'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'testfile.txt'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('test*', 'Matched forced_separate (test*) config value.'))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test*'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '.testfile.txt'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('test*', 'Matched forced_separate (test*) config value.'))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'other*'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'testfile.txt'
    var_6 = module_1._forced_separate(var_5, var_4)
    assert var_6 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'testfile.txt'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('test', 'Matched forced_separate (test) config value.'))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = 'forced_separate'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'testfile.txt'
    var_5 = module_1._forced_separate(var_4, var_3)
    assert var_5 is None



# Parsed testcases at query #4
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test*'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'testfile.txt'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('test*', 'Matched forced_separate (test*) config value.'))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test*'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '.testfile.txt'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('test*', 'Matched forced_separate (test*) config value.'))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'other*'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'testfile.txt'
    var_6 = module_1._forced_separate(var_5, var_4)
    assert var_6 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'testfile.txt'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('test', 'Matched forced_separate (test) config value.'))
    assert var_7 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test*'
    var_1 = 'other*'
    var_2 = [var_0, var_1]
    var_3 = 'forced_separate'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = 'otherfile.txt'
    var_7 = module_1._forced_separate(var_6, var_5)
    var_8 = bool(var_7 == ('other*', 'Matched forced_separate (other*) config value.'))
    assert var_8 is True

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = 'forced_separate'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'testfile.txt'
    var_5 = module_1._forced_separate(var_4, var_3)
    assert var_5 is None



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_src_path_finds_module_in_src_paths. Retrieved 7/11 statements.
# Partially parsed test_src_path_finds_nested_module. Retrieved 7/11 statements.
# Partially parsed test_src_path_returns_none_for_non_existent_module. Retrieved 7/11 statements.
# Partially parsed test_src_path_handles_namespace_packages. Retrieved 8/12 statements.
# Partially parsed test_src_path_handles_auto_identified_namespace_packages. Retrieved 7/11 statements.
# Partially parsed test_src_path_with_custom_src_paths. Retrieved 7/11 statements.
# Partially parsed test_src_path_with_prefix. Retrieved 9/13 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)
    var_6 = False
    var_7 = 'module'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)
    var_6 = False
    var_7 = 'parent.child'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)
    var_6 = False
    var_7 = 'nonexistent'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'parent'
    var_3 = {var_2}
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = False
    var_8 = 'parent.child'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)
    var_6 = True
    var_7 = 'parent.child'

def test_case_0():
    var_0 = '/custom/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)
    var_6 = False
    var_7 = 'module'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)
    var_6 = False
    var_7 = 'module'
    var_8 = 'parent'
    var_9 = (var_8,)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = None
    var_1 = set()
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)
    var_5 = False
    var_6 = 'src_paths'
    var_7 = 'namespace_packages'
    var_8 = 'supported_extensions'
    var_9 = 'auto_identify_namespace_packages'
    var_10 = {var_6: var_0, var_7: var_1, var_8: var_4, var_9: var_5}
    var_11 = module_0.Config(**var_10)
    var_12 = 'module'
    var_13 = module_1._src_path(var_12, var_11)
    assert var_13 is None



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_predicate_at_line_16_evaluates_to_true. Retrieved 8/20 statements.


def test_case_0():
    var_0 = 'module'
    var_1 = 'src'
    var_2 = [var_1]
    var_3 = 'src/module'
    var_4 = [var_3]
    var_5 = ()
    var_6 = '.'
    var_7 = 1
    var_8 = *var_5
    var_9 = 0



# Parsed testcases at query #7
#--------------------------

# Partially parsed test__is_namespace_package_returns_false_when_not_a_package. Retrieved 4/7 statements.
# Partially parsed test__is_namespace_package_returns_false_when_init_exists_without_namespace_pattern. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_true_when_init_exists_with_pkg_resources_pattern. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_true_when_init_exists_with_pkgutil_pattern. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_false_when_no_init_but_has_py_files. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_false_when_no_init_but_has_setup_cfg. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_true_when_no_init_and_no_relevant_files. Retrieved 7/13 statements.


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
    var_4 = "print('hello')"
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
    var_0 = '/tmp/test_with_py_files'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'module.py'
    var_4 = 'x = 1'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/tmp/test_with_setup_cfg'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'setup.cfg'
    var_4 = '[metadata]\nname = test'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/tmp/test_empty_namespace'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'README.md'
    var_4 = '# Test'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test__src_path_is_module_returns_true_for_valid_module. Retrieved 2/5 statements.
# Partially parsed test__src_path_is_module_returns_false_for_non_matching_name. Retrieved 3/6 statements.
# Partially parsed test__src_path_is_module_returns_false_for_file. Retrieved 2/5 statements.
# Partially parsed test__src_path_is_module_returns_false_for_nonexistent_path. Retrieved 1/3 statements.
# Partially parsed test__src_path_is_module_returns_false_for_case_sensitive_mismatch. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'valid_module'
    var_1 = [var_0]
    var_2 = True

def test_case_0():
    var_0 = 'valid_module'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'invalid_module'

def test_case_0():
    var_0 = 'file_module.py'
    var_1 = [var_0]
    var_2 = 'file_module'

def test_case_0():
    var_0 = 'nonexistent_module'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'CaseSensitiveModule'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'casesensitivemodule'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test__src_path_is_module_returns_true_when_conditions_met. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'module_name'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'module_name'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test__is_module_returns_true_for_py_file. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'module.py'
    var_1 = [var_0]



# Parsed testcases at query #11
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
    var_0 = 'module'
    var_1 = [var_0]
    var_2 = 'module/__init__.py'

def test_case_0():
    var_0 = 'non_existent_module'
    var_1 = [var_0]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_predicate_at_line_16. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 'src'
    var_1 = [var_0]
    var_2 = []
    var_3 = False
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = [var_0]
    var_7 = 'module'
    var_8 = ()



# Parsed testcases at query #13
#--------------------------

# Partially parsed test__is_namespace_package_returns_false_when_path_is_not_a_package. Retrieved 4/7 statements.
# Partially parsed test__is_namespace_package_returns_false_when_init_file_exists_without_namespace_declaration. Retrieved 6/15 statements.
# Partially parsed test__is_namespace_package_returns_true_when_init_file_has_pkg_resources_declaration. Retrieved 6/15 statements.
# Partially parsed test__is_namespace_package_returns_true_when_init_file_has_pkgutil_declaration. Retrieved 6/15 statements.
# Partially parsed test__is_namespace_package_returns_false_when_package_contains_py_files. Retrieved 6/15 statements.
# Partially parsed test__is_namespace_package_returns_false_when_package_contains_setup_cfg. Retrieved 6/15 statements.
# Partially parsed test__is_namespace_package_returns_false_when_package_contains_pyproject_toml. Retrieved 6/15 statements.
# Partially parsed test__is_namespace_package_returns_true_when_package_is_empty. Retrieved 4/11 statements.


def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = 'package'
    var_1 = '__init__.py'
    var_2 = '# regular package'
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'package'
    var_1 = '__init__.py'
    var_2 = '__import__("pkg_resources").declare_namespace(__name__)'
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'package'
    var_1 = '__init__.py'
    var_2 = '__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'package'
    var_1 = 'module.py'
    var_2 = '# some code'
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'package'
    var_1 = 'setup.cfg'
    var_2 = '[metadata]\nname = test'
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'package'
    var_1 = 'pyproject.toml'
    var_2 = "[build-system]\nrequires = ['setuptools']"
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'package'
    var_1 = '.py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]
    var_2 = 'test_module.py'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test__src_path_with_module_in_src_paths. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_nested_module_in_namespace_package. Retrieved 4/8 statements.
# Partially parsed test__src_path_with_auto_identify_namespace_packages. Retrieved 6/10 statements.
# Partially parsed test__src_path_with_module_not_found. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_empty_name. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_root_module_name_matching_src_path. Retrieved 2/6 statements.


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
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)
    var_6 = 'parent.child'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = 'nonexistent'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = ''

def test_case_0():
    var_0 = '/path/to/module'
    var_1 = [var_0]
    var_2 = 'module'



# Parsed testcases at query #16
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
    var_0 = 'foo.*'
    var_1 = 'placement1'
    var_2 = 'bar.*'
    var_3 = 'placement2'
    var_4 = [var_1, var_3]
    var_5 = 'bar.module'

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
    var_3 = 'test.sub.module'

def test_case_0():
    var_0 = 'test.*'
    var_1 = 'placement1'
    var_2 = [var_1]
    var_3 = ''



# Parsed testcases at query #17
#--------------------------

# Partially parsed test__src_path_with_module_in_src_paths. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_nested_module_in_src_paths. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_namespace_package. Retrieved 4/8 statements.
# Partially parsed test__src_path_with_auto_identify_namespace_package. Retrieved 6/10 statements.
# Partially parsed test__src_path_with_module_not_in_src_paths. Retrieved 2/6 statements.


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
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)
    var_6 = 'parent.child'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = 'nonexistent'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test__src_path_with_module_not_found. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_module_found_in_src_path. Retrieved 2/8 statements.
# Partially parsed test__src_path_with_nested_module_in_namespace_package. Retrieved 9/16 statements.
# Partially parsed test__src_path_with_auto_identify_namespace_package. Retrieved 10/17 statements.
# Partially parsed test__src_path_with_module_in_root_src_path. Retrieved 2/7 statements.


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
    var_0 = '/fake/path'
    var_1 = [var_0]
    var_2 = 'module'

def test_case_0():
    var_0 = '/real/path'
    var_1 = [var_0]
    var_2 = 'module'

def test_case_0():
    var_0 = '/real/path'
    var_1 = [var_0]
    var_2 = 'parent'
    var_3 = {var_2}
    var_4 = 'parent.child'
    var_5 = 'child'
    var_6 = '/real/path/parent'
    var_7 = [var_6]
    var_8 = (var_2,)
    var_9 = 'parent'
    var_10 = (var_9,)

def test_case_0():
    var_0 = '/real/path'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = 'parent.child'
    var_6 = 'child'
    var_7 = '/real/path/parent'
    var_8 = [var_7]
    var_9 = (var_2,)
    var_10 = 'parent'
    var_11 = (var_10,)

def test_case_0():
    var_0 = '/root/module'
    var_1 = [var_0]
    var_2 = 'module'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test__is_module_returns_true_for_py_file. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'test_module.py'
    var_1 = [var_0]



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_true. Retrieved 9/12 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_namespace'
    var_1 = {var_0}
    var_2 = False
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = 'namespace_packages'
    var_6 = 'auto_identify_namespace_packages'
    var_7 = 'supported_extensions'
    var_8 = {var_5: var_1, var_6: var_2, var_7: var_4}
    var_9 = module_0.Config(**var_8)
    var_10 = '/path/to/src'
    var_11 = [var_10]
    var_12 = 'test_namespace.submodule'
    var_13 = ()



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 8/20 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = 'src'
    var_2 = [var_1]
    var_3 = []
    var_4 = True
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = [var_1]
    var_8 = ()
    var_9 = [var_1]
    var_10 = 'module'
    var_11 = [var_1]



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_predicate_at_line_16. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 'module'
    var_1 = 'src'
    var_2 = [var_1]
    var_3 = []
    var_4 = False
    var_5 = [var_1]
    var_6 = ()



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_is_module_returns_true_for_py_file. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'module.py'
    var_1 = [var_0]



# Parsed testcases at query #24
#--------------------------

# Partially parsed test__is_namespace_package_returns_true. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 'valid_namespace_package'
    assert var_0 is True
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = '.pyx'
    var_4 = '.pxd'
    var_5 = [var_2, var_3, var_4]
    var_6 = frozenset(var_5)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test__src_path_with_module_in_src_paths. Retrieved 4/12 statements.
# Partially parsed test__src_path_with_nested_module_in_src_paths. Retrieved 7/16 statements.
# Partially parsed test__src_path_with_namespace_package. Retrieved 4/10 statements.
# Partially parsed test__src_path_with_auto_identified_namespace_package. Retrieved 7/13 statements.
# Partially parsed test__src_path_with_src_path_is_module. Retrieved 4/11 statements.


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
    var_3 = '__init__.py'
    var_4 = ''

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = 'parent'
    var_3 = 'child'
    var_4 = True
    var_5 = '__init__.py'
    var_6 = ''
    var_7 = 'parent.child'

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
    var_3 = True
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'namespace.module'

def test_case_0():
    var_0 = '/path/to/module'
    var_1 = [var_0]
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'module'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = '/path/to/src'
    var_2 = [var_1]
    var_3 = 'module'
    var_4 = [var_3]
    var_5 = False
    var_6 = None
    var_7 = ()



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 8/18 statements.


def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = []
    var_3 = True
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = 'module.submodule'
    var_7 = [var_0]
    var_8 = ()
    var_9 = '/path/to/src/module'
    var_10 = [var_9]



# Parsed testcases at query #28
#--------------------------

# Partially parsed test__src_path_with_module_in_src_paths. Retrieved 5/10 statements.
# Partially parsed test__src_path_with_package_in_src_paths. Retrieved 5/10 statements.
# Partially parsed test__src_path_with_src_path_is_module. Retrieved 5/10 statements.
# Partially parsed test__src_path_with_nested_module_in_namespace_package. Retrieved 6/11 statements.
# Partially parsed test__src_path_with_nested_module_in_auto_identified_namespace_package. Retrieved 5/10 statements.
# Partially parsed test__src_path_with_nested_module_not_in_namespace_package. Retrieved 5/10 statements.


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
    var_5 = 'package'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = frozenset()
    var_5 = 'src'

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
    var_2 = set()
    var_3 = True
    var_4 = frozenset()
    var_5 = 'namespace.module'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = frozenset()
    var_5 = 'namespace.module'



# Parsed testcases at query #29
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



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = '/path/to/src'
    var_2 = [var_1]
    var_3 = 'module'
    var_4 = [var_3]
    var_5 = False
    var_6 = '/path/to/src/module'
    var_7 = [var_6]
    var_8 = ()



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_false. Retrieved 8/12 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = []
    var_2 = False
    var_3 = '/path/to/src'
    var_4 = [var_3]
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = None
    var_8 = ()



# Parsed testcases at query #32
#--------------------------

# Partially parsed test__src_path_with_module_in_src_paths. Retrieved 3/11 statements.
# Partially parsed test__src_path_with_nested_module_in_src_paths. Retrieved 6/15 statements.
# Partially parsed test__src_path_with_namespace_package. Retrieved 5/13 statements.
# Partially parsed test__src_path_with_auto_identify_namespace_package. Retrieved 5/13 statements.
# Partially parsed test__src_path_with_module_not_in_src_paths. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_root_module_in_src_paths. Retrieved 2/9 statements.


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
    var_3 = '__init__.py'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = 'parent'
    var_3 = 'child'
    var_4 = True
    var_5 = '__init__.py'
    var_6 = 'parent.child'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = 'namespace'
    var_3 = '__init__.py'
    var_4 = "__import__('pkg_resources').declare_namespace(__name__)"
    var_5 = [var_2]

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = 'namespace'
    var_3 = '__init__.py'
    var_4 = "__import__('pkg_resources').declare_namespace(__name__)"
    var_5 = True

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = 'nonexistent'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = [var_0]
    var_2 = '__init__.py'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test__is_namespace_package_with_non_package_path. Retrieved 4/7 statements.
# Partially parsed test__is_namespace_package_with_package_having_init_file. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_with_package_having_init_file_with_declare_namespace. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_with_package_having_init_file_with_extend_path. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_with_package_having_no_init_file_and_no_source_files. Retrieved 5/9 statements.
# Partially parsed test__is_namespace_package_with_package_having_no_init_file_but_with_source_files. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_with_package_having_no_init_file_but_with_setup_cfg. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_with_package_having_no_init_file_but_with_pyproject_toml. Retrieved 7/13 statements.


def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/package/with/__init__.py'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = ''
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/package/with/__init__.py'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = '__import__("pkg_resources").declare_namespace(__name__)'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/package/with/__init__.py'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = '__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/package/without/init'
    var_1 = [var_0]
    var_2 = True
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = '/package/with/source'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'module.py'
    var_4 = ''
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/package/with/setup'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'setup.cfg'
    var_4 = ''
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/package/with/pyproject'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'pyproject.toml'
    var_4 = ''
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_true. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 'test_namespace'
    var_1 = [var_0]
    var_2 = False
    var_3 = '/test/src'
    var_4 = [var_3]
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = 'test_namespace.submodule'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 2/5 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 1/6 statements.
# Partially parsed test_is_module_with_init_file. Retrieved 2/5 statements.
# Partially parsed test_is_module_no_match. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'module'
    var_1 = [var_0]
    var_2 = 'module.py'

def test_case_0():
    var_0 = 'module'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'module'
    var_1 = [var_0]
    var_2 = 'module/__init__.py'

def test_case_0():
    var_0 = 'not_a_module'
    var_1 = [var_0]



# Parsed testcases at query #36
#--------------------------

# Partially parsed test__is_namespace_package_returns_true_when_package_and_no_init_file_and_no_source_files. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 'some_namespace_package'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = '.pyx'
    var_4 = '.pxd'
    var_5 = '.pxi'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = frozenset(var_6)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test__is_module_returns_true_for_py_file. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'module.py'
    var_1 = [var_0]



# Parsed testcases at query #38
#--------------------------

# Partially parsed test__src_path_is_module_returns_true_when_conditions_are_met. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'example_module'
    var_1 = [var_0]
    var_2 = True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test__src_path__returns_firstparty_when_module_found_in_src_path. Retrieved 2/13 statements.
# Partially parsed test__src_path__returns_firstparty_when_package_found_in_src_path. Retrieved 2/15 statements.
# Partially parsed test__src_path__returns_firstparty_when_nested_module_in_namespace_package. Retrieved 6/21 statements.
# Partially parsed test__src_path__returns_firstparty_when_src_path_is_module. Retrieved 2/11 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = []
    var_2 = 'src_paths'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = module_1._src_path(var_0, var_4)
    assert var_5 is None

def test_case_0():
    var_0 = 'module.py'
    var_1 = 'module'

def test_case_0():
    var_0 = 'package'
    var_1 = '__init__.py'

def test_case_0():
    var_0 = 'namespace'
    var_1 = '__init__.py'
    var_2 = "__import__('pkgutil').extend_path(__path__, __name__)"
    var_3 = 'module.py'
    var_4 = {var_0}
    var_5 = 'namespace.module'

def test_case_0():
    var_0 = 'module'
    var_1 = '__init__.py'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_predicate_at_line_16_evaluates_to_true. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 'module'
    var_1 = 'module'
    var_2 = [var_1]
    var_3 = []
    var_4 = False
    var_5 = [var_1]
    var_6 = ()



# Parsed testcases at query #41
#--------------------------

# Partially parsed test__src_path_found_in_src_paths. Retrieved 7/11 statements.
# Partially parsed test__src_path_not_found. Retrieved 7/11 statements.
# Partially parsed test__src_path_nested_module. Retrieved 7/11 statements.
# Partially parsed test__src_path_namespace_package. Retrieved 8/12 statements.
# Partially parsed test__src_path_auto_identify_namespace_package. Retrieved 7/11 statements.


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
    var_2 = set()
    var_3 = False
    var_4 = 'py'
    var_5 = {var_4}
    var_6 = frozenset(var_5)
    var_7 = 'parent.child'

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



# Parsed testcases at query #42
#--------------------------

# Partially parsed test__is_namespace_package_returns_true_for_valid_namespace_package. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 'valid_namespace_pkg'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_true. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = 'module'
    var_2 = {var_1}
    var_3 = 'src'
    var_4 = [var_3]
    var_5 = False
    var_6 = '.py'
    var_7 = [var_6]
    var_8 = [var_3]
    var_9 = ()



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_known_pattern_matches_first_segment. Retrieved 4/9 statements.
# Partially parsed test_known_pattern_matches_middle_segment. Retrieved 4/9 statements.
# Partially parsed test_known_pattern_matches_last_segment. Retrieved 4/9 statements.
# Partially parsed test_known_pattern_no_match. Retrieved 4/9 statements.
# Partially parsed test_known_pattern_placement_not_in_sections. Retrieved 5/10 statements.
# Partially parsed test_known_pattern_multiple_patterns_first_match_wins. Retrieved 6/13 statements.
# Partially parsed test_known_pattern_empty_name. Retrieved 4/9 statements.
# Partially parsed test_known_pattern_single_segment_match. Retrieved 3/8 statements.
# Partially parsed test_known_pattern_single_segment_no_match. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'foo'
    var_1 = 'placement1'
    var_2 = [var_1]
    var_3 = 'foo.bar.baz'

def test_case_0():
    var_0 = 'bar'
    var_1 = 'placement2'
    var_2 = [var_1]
    var_3 = 'foo.bar.baz'

def test_case_0():
    var_0 = 'baz'
    var_1 = 'placement3'
    var_2 = [var_1]
    var_3 = 'foo.bar.baz'

def test_case_0():
    var_0 = 'qux'
    var_1 = 'placement4'
    var_2 = [var_1]
    var_3 = 'foo.bar.baz'

def test_case_0():
    var_0 = 'foo'
    var_1 = 'placement5'
    var_2 = 'placement6'
    var_3 = [var_2]
    var_4 = 'foo.bar.baz'

def test_case_0():
    var_0 = 'foo'
    var_1 = 'placement1'
    var_2 = 'bar'
    var_3 = 'placement2'
    var_4 = [var_1, var_3]
    var_5 = 'foo.bar.baz'

def test_case_0():
    var_0 = 'foo'
    var_1 = 'placement1'
    var_2 = [var_1]
    var_3 = ''

def test_case_0():
    var_0 = 'foo'
    var_1 = 'placement1'
    var_2 = [var_1]

def test_case_0():
    var_0 = 'bar'
    var_1 = 'placement1'
    var_2 = [var_1]
    var_3 = 'foo'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test__src_path_with_module_not_found. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_module_in_src_path. Retrieved 3/9 statements.
# Partially parsed test__src_path_with_package_in_src_path. Retrieved 4/12 statements.
# Partially parsed test__src_path_with_nested_module. Retrieved 5/15 statements.
# Partially parsed test__src_path_with_namespace_package. Retrieved 5/11 statements.
# Partially parsed test__src_path_with_auto_identify_namespace_package. Retrieved 6/12 statements.
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
    var_0 = '/tmp'
    var_1 = [var_0]
    var_2 = 'nonexistent_module'

def test_case_0():
    var_0 = '/tmp'
    var_1 = [var_0]
    var_2 = '/tmp/module.py'
    var_3 = [var_2]
    var_4 = 'module'

def test_case_0():
    var_0 = '/tmp'
    var_1 = [var_0]
    var_2 = '/tmp/package'
    var_3 = [var_2]
    var_4 = '__init__.py'
    var_5 = 'package'

def test_case_0():
    var_0 = '/tmp'
    var_1 = [var_0]
    var_2 = '/tmp/package'
    var_3 = [var_2]
    var_4 = '__init__.py'
    var_5 = 'nested.py'
    var_6 = 'package.nested'

def test_case_0():
    var_0 = '/tmp'
    var_1 = [var_0]
    var_2 = 'package'
    var_3 = {var_2}
    var_4 = '/tmp/package'
    var_5 = [var_4]
    var_6 = 'package.nested'

def test_case_0():
    var_0 = '/tmp'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = '/tmp/package'
    var_6 = [var_5]
    var_7 = 'package.nested'

def test_case_0():
    var_0 = '/tmp/module'
    var_1 = [var_0]
    var_2 = '/tmp/module.py'
    var_3 = [var_2]
    var_4 = 'module'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test__is_namespace_package_returns_true_for_valid_namespace. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 'valid_namespace'
    var_1 = [var_0]
    var_2 = '__init__.py'
    var_3 = "__import__('pkg_resources').declare_namespace(__name__)"
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_src_path_is_module_when_path_matches_module_name. Retrieved 2/6 statements.
# Partially parsed test_src_path_is_not_module_when_path_does_not_match_module_name. Retrieved 3/7 statements.
# Partially parsed test_src_path_is_not_module_when_path_is_not_directory. Retrieved 1/5 statements.
# Partially parsed test_src_path_is_not_module_when_path_does_not_exist_case_sensitively. Retrieved 3/7 statements.


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

def test_case_0():
    var_0 = 'module_name'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'MODULE_NAME'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 'some_path'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)
    var_5 = 'file.py'



# Parsed testcases at query #49
#--------------------------

# Partially parsed test__is_namespace_package_returns_true_for_namespace_package_without_init. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'namespace_pkg'
    var_1 = 'module.py'
    var_2 = '# empty'
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)



# Parsed testcases at query #50
#--------------------------

# Partially parsed test__is_namespace_package_returns_false_when_not_a_package. Retrieved 4/7 statements.
# Partially parsed test__is_namespace_package_returns_false_when_init_file_exists_without_namespace_declaration. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_false_when_init_file_exists_with_pkg_resources_declaration. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_false_when_init_file_exists_with_pkgutil_declaration. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_false_when_no_init_file_but_has_py_files. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_false_when_no_init_file_but_has_setup_cfg. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_true_when_no_init_file_and_no_py_files. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'non_existent_path'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = 'test_package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = '# Regular package'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = 'test_namespace_pkg_resources'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = "__import__('pkg_resources').declare_namespace(__name__)"
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = 'test_namespace_pkgutil'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = "__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = 'test_namespace_with_py_files'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'module.py'
    var_4 = '# Python module'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = 'test_namespace_with_setup_cfg'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'setup.cfg'
    var_4 = '[metadata]\nname = test'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = 'test_namespace_empty'
    var_1 = [var_0]
    var_2 = True
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)



# Parsed testcases at query #51
#--------------------------

# Partially parsed test__is_namespace_package_with_non_package_path. Retrieved 4/7 statements.
# Partially parsed test__is_namespace_package_with_package_but_no_init_file. Retrieved 5/9 statements.
# Partially parsed test__is_namespace_package_with_package_and_init_file_but_no_namespace_declaration. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_with_package_and_init_file_with_pkg_resources_declaration. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_with_package_and_init_file_with_pkgutil_declaration. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_with_package_and_init_file_with_mixed_quotes_declaration. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_with_package_and_init_file_with_single_quotes_declaration. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_with_package_and_init_file_with_pkgutil_single_quotes_declaration. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_with_package_and_init_file_but_has_source_files. Retrieved 9/17 statements.
# Partially parsed test__is_namespace_package_with_package_and_init_file_but_has_setup_cfg. Retrieved 9/17 statements.
# Partially parsed test__is_namespace_package_with_package_and_init_file_but_has_pyproject_toml. Retrieved 9/17 statements.


def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/existing/package/path'
    var_1 = [var_0]
    var_2 = True
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = '/existing/package/path'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = '# Regular package'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/existing/package/path'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = '__import__("pkg_resources").declare_namespace(__name__)'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/existing/package/path'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = '__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/existing/package/path'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = '__import__("pkg_resources").declare_namespace(__name__)'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/existing/package/path'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = "__import__('pkg_resources').declare_namespace(__name__)"
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/existing/package/path'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = "__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/existing/package/path'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = '# Regular package'
    var_5 = 'module.py'
    var_6 = '# Some code'
    var_7 = '.py'
    var_8 = [var_7]
    var_9 = frozenset(var_8)

def test_case_0():
    var_0 = '/existing/package/path'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = '# Regular package'
    var_5 = 'setup.cfg'
    var_6 = '[metadata]'
    var_7 = '.py'
    var_8 = [var_7]
    var_9 = frozenset(var_8)

def test_case_0():
    var_0 = '/existing/package/path'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = '# Regular package'
    var_5 = 'pyproject.toml'
    var_6 = '[build-system]'
    var_7 = '.py'
    var_8 = [var_7]
    var_9 = frozenset(var_8)



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_predicate_at_line_13_evaluates_to_false. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 'test_package'
    var_1 = [var_0]
    var_2 = 'module.py'
    var_3 = ''
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_predicate_false_when_filenames_exist. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'test_path'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)
    var_5 = 'test.py'



# Parsed testcases at query #54
#--------------------------

# Partially parsed test__is_namespace_package_returns_false_when_path_is_not_a_package. Retrieved 4/7 statements.
# Partially parsed test__is_namespace_package_returns_false_when_init_file_exists_without_namespace_declaration. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_true_when_init_file_contains_pkg_resources_declaration. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_true_when_init_file_contains_pkgutil_declaration. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_false_when_package_contains_py_files. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_false_when_package_contains_setup_cfg. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_false_when_package_contains_pyproject_toml. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_true_when_package_is_empty. Retrieved 5/9 statements.


def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = {var_2}
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/existing/package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = "print('hello')"
    var_5 = '.py'
    var_6 = {var_5}
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/existing/package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = "__import__('pkg_resources').declare_namespace(__name__)"
    var_5 = '.py'
    var_6 = {var_5}
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/existing/package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = "__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_5 = '.py'
    var_6 = {var_5}
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/existing/package'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'module.py'
    var_4 = "print('hello')"
    var_5 = '.py'
    var_6 = {var_5}
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/existing/package'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'setup.cfg'
    var_4 = '[metadata]\nname=test'
    var_5 = '.py'
    var_6 = {var_5}
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/existing/package'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'pyproject.toml'
    var_4 = "[build-system]\nrequires = ['setuptools']"
    var_5 = '.py'
    var_6 = {var_5}
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/existing/package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '.py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_is_namespace_package_without_init_file. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 'test_namespace'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'module.py'
    var_4 = '# test module'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 'test_package'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)
    var_5 = '__init__.py'
    var_6 = b"__import__('pkg_resources').declare_namespace(__name__)"



# Parsed testcases at query #57
#--------------------------

# Partially parsed test__is_namespace_package_returns_true_when_init_file_does_not_exist_and_no_source_files. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 'some_namespace_package'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_true. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 'test_namespace'
    var_1 = {var_0}
    var_2 = False
    var_3 = '/test/src'
    var_4 = [var_3]
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = 'test_namespace.submodule'
    var_8 = [var_3]
    var_9 = ()



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_predicate_at_line_16. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'module'
    var_1 = 'src'
    var_2 = [var_1]
    var_3 = []
    var_4 = False
    var_5 = 'src/module'
    var_6 = [var_5]
    var_7 = ()



# Parsed testcases at query #60
#--------------------------

# Partially parsed test__is_namespace_package_returns_false_when_path_is_not_a_package. Retrieved 4/7 statements.
# Partially parsed test__is_namespace_package_returns_false_when_init_file_exists_without_namespace_declaration. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_true_when_init_file_contains_pkg_resources_declare_namespace. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_true_when_init_file_contains_pkgutil_extend_path. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_false_when_package_contains_py_files. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_false_when_package_contains_setup_cfg. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_false_when_package_contains_pyproject_toml. Retrieved 7/13 statements.


def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = [var_0]
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/valid/package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = '# Regular package'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/valid/namespace/package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = '__import__("pkg_resources").declare_namespace(__name__)'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/valid/namespace/package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = '__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/valid/package/with/files'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'module.py'
    var_4 = '# Some code'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/valid/package/with/setup'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'setup.cfg'
    var_4 = '[metadata]\nname=test'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/valid/package/with/pyproject'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'pyproject.toml'
    var_4 = '[build-system]\nrequires = []'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_known_pattern_predicate_false. Retrieved 7/9 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.*'
    var_3 = 'section1'
    var_4 = (var_2, var_3)
    var_5 = 'section2'
    var_6 = 'test.module'
    var_7 = module_1._known_pattern(var_6, var_1)
    assert var_7 is None



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = 'src'
    var_2 = [var_1]
    var_3 = []
    var_4 = True
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = [var_1]
    var_8 = ()



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_known_pattern_predicate_false. Retrieved 7/9 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'section1'
    var_3 = 'pattern1'
    var_4 = 'section2'
    var_5 = (var_3, var_4)
    var_6 = 'test.module'
    var_7 = module_1._known_pattern(var_6, var_1)
    assert var_7 is None



# Parsed testcases at query #64
#--------------------------

# Partially parsed test__is_module_returns_true_for_py_file. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'test_module.py'
    var_1 = [var_0]
    var_2 = '.py'



# Parsed testcases at query #65
#--------------------------

# Partially parsed test__is_namespace_package_returns_false_for_non_package. Retrieved 4/7 statements.
# Partially parsed test__is_namespace_package_returns_false_for_package_with_init. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_false_for_package_with_non_namespace_init. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_true_for_package_with_pkg_resources_namespace. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_true_for_package_with_pkgutil_namespace. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_false_for_package_with_py_files. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_false_for_package_with_setup_cfg. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_false_for_package_with_pyproject_toml. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_true_for_empty_package. Retrieved 5/9 statements.


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
    var_4 = '__path__ = []'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/existing/package/with/non_namespace/__init__.py'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = '# regular init file'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/existing/package/with/pkg_resources/namespace/__init__.py'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = "__import__('pkg_resources').declare_namespace(__name__)"
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/existing/package/with/pkgutil/namespace/__init__.py'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = "__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/existing/package/with/py/files'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'module.py'
    var_4 = '# some code'
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
    var_0 = '/existing/empty/package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)



