####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test__src_path_with_empty_name. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_module_in_src_paths. Retrieved 3/9 statements.
# Partially parsed test__src_path_with_package_in_src_paths. Retrieved 4/12 statements.
# Partially parsed test__src_path_with_nested_module. Retrieved 5/15 statements.
# Partially parsed test__src_path_with_namespace_package. Retrieved 5/11 statements.
# Partially parsed test__src_path_with_auto_identify_namespace_package. Retrieved 8/16 statements.
# Partially parsed test__src_path_with_src_path_is_module. Retrieved 3/11 statements.
# Partially parsed test__src_path_with_no_match. Retrieved 2/6 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Config()
    var_2 = 'module'
    var_3 = module_1._src_path(var_2, var_1)
    assert var_3 is None

def test_case_0():
    var_0 = '/some/path'
    var_1 = ''

def test_case_0():
    var_0 = '/some/path'
    var_1 = '/some/path/module.py'
    var_2 = 'module'

def test_case_0():
    var_0 = '/some/path'
    var_1 = '/some/path/package'
    var_2 = '__init__.py'
    var_3 = 'package'

def test_case_0():
    var_0 = '/some/path'
    var_1 = '/some/path/parent'
    var_2 = '__init__.py'
    var_3 = 'child.py'
    var_4 = 'parent.child'

def test_case_0():
    var_0 = '/some/path'
    var_1 = 'parent'
    var_2 = {var_1}
    var_3 = '/some/path/parent'
    var_4 = 'parent.child'

def test_case_0():
    var_0 = '/some/path'
    var_1 = True
    var_2 = 'py'
    var_3 = {var_2}
    var_4 = frozenset(var_3)
    var_5 = '/some/path/parent'
    var_6 = 'child.py'
    var_7 = 'parent.child'

def test_case_0():
    var_0 = '/some/path/module'
    var_1 = '__init__.py'
    var_2 = 'module'

def test_case_0():
    var_0 = '/some/path'
    var_1 = 'nonexistent'



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
    var_0 = 'foo.*'
    var_1 = 'placement1'
    var_2 = 'bar.*'
    var_3 = 'placement2'
    var_4 = [var_1, var_3]
    var_5 = 'bar.baz'

def test_case_0():
    var_0 = 'test.*'
    var_1 = 'placement1'
    var_2 = [var_1]
    var_3 = 'other.module'

def test_case_0():
    var_0 = 'test.*'
    var_1 = 'invalid_placement'
    var_2 = 'placement1'
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



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 2/5 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 1/6 statements.
# Partially parsed test_is_module_with_init_file. Retrieved 2/5 statements.
# Partially parsed test_is_module_non_existent. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'module'
    var_1 = 'module.py'

def test_case_0():
    var_0 = 'module'

def test_case_0():
    var_0 = 'module'
    var_1 = 'module/__init__.py'

def test_case_0():
    var_0 = 'non_existent_module'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test__src_path_with_non_existent_module. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_existing_module_file. Retrieved 3/9 statements.
# Partially parsed test__src_path_with_existing_package. Retrieved 4/12 statements.
# Partially parsed test__src_path_with_nested_module. Retrieved 5/15 statements.
# Partially parsed test__src_path_with_namespace_package. Retrieved 5/11 statements.
# Partially parsed test__src_path_with_auto_identified_namespace_package. Retrieved 8/16 statements.
# Partially parsed test__src_path_with_src_path_is_module. Retrieved 3/11 statements.
# Partially parsed test__src_path_with_non_matching_module_name. Retrieved 3/9 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Config()
    var_2 = 'module'
    var_3 = module_1._src_path(var_2, var_1)
    assert var_3 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = module_0.Config()
    var_2 = 'module'
    var_3 = module_1._src_path(var_2, var_1)
    assert var_3 is None

def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = 'module'

def test_case_0():
    var_0 = '/existing/path'
    var_1 = '/existing/path/module.py'
    var_2 = 'module'

def test_case_0():
    var_0 = '/existing/path'
    var_1 = '/existing/path/package'
    var_2 = '__init__.py'
    var_3 = 'package'

def test_case_0():
    var_0 = '/existing/path'
    var_1 = '/existing/path/package'
    var_2 = '__init__.py'
    var_3 = 'nested.py'
    var_4 = 'package.nested'

def test_case_0():
    var_0 = '/existing/path'
    var_1 = 'package'
    var_2 = {var_1}
    var_3 = '/existing/path/package'
    var_4 = 'package.nested'

def test_case_0():
    var_0 = '/existing/path'
    var_1 = True
    var_2 = 'py'
    var_3 = {var_2}
    var_4 = frozenset(var_3)
    var_5 = '/existing/path/package'
    var_6 = 'module.py'
    var_7 = 'package.nested'

def test_case_0():
    var_0 = '/existing/path/module'
    var_1 = '__init__.py'
    var_2 = 'module'

def test_case_0():
    var_0 = '/existing/path'
    var_1 = '/existing/path/other_module.py'
    var_2 = 'module'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'test_module'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_is_module_returns_true_for_py_file. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'module.py'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = '/path/to/src'
    var_2 = 'module'
    var_3 = [var_2]
    var_4 = False
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = ()



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'unknown.module'
    var_1 = 'known.*'
    var_2 = 'placement'
    var_3 = []



# Parsed testcases at query #9
#--------------------------

# Partially parsed test__is_module_returns_true_for_py_file. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'module.py'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_src_path_is_module_returns_true_for_valid_module. Retrieved 2/5 statements.
# Partially parsed test_src_path_is_module_returns_false_for_non_matching_name. Retrieved 3/6 statements.
# Partially parsed test_src_path_is_module_returns_false_for_non_directory. Retrieved 2/5 statements.
# Partially parsed test_src_path_is_module_returns_false_for_case_sensitive_mismatch. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'valid_module'
    var_1 = True

def test_case_0():
    var_0 = 'different_name'
    var_1 = True
    var_2 = 'valid_module'

def test_case_0():
    var_0 = 'valid_module'
    var_1 = False

def test_case_0():
    var_0 = 'Valid_Module'
    var_1 = True
    var_2 = 'valid_module'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 2/5 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 1/6 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 2/5 statements.
# Partially parsed test_is_module_with_no_matching_files. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'module'
    var_1 = 'module.py'

def test_case_0():
    var_0 = 'module'

def test_case_0():
    var_0 = 'package'
    var_1 = 'package/__init__.py'

def test_case_0():
    var_0 = 'nonexistent'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test__src_path_with_module_in_src_paths. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_nested_module. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_namespace_package. Retrieved 4/8 statements.
# Partially parsed test__src_path_with_auto_identify_namespace_packages. Retrieved 3/7 statements.
# Partially parsed test__src_path_with_src_path_is_module. Retrieved 2/6 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Config()
    var_2 = 'module'
    var_3 = module_1._src_path(var_2, var_1)
    assert var_3 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = module_0.Config()
    var_2 = 'module'
    var_3 = module_1._src_path(var_2, var_1)
    assert var_3 is None

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = 'module'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = 'parent.child'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = 'parent'
    var_2 = {var_1}
    var_3 = 'parent.child'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = True
    var_2 = 'parent.child'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = 'src'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test__is_namespace_package_with_non_package_path. Retrieved 4/6 statements.
# Partially parsed test__is_namespace_package_with_package_containing_init. Retrieved 7/12 statements.
# Partially parsed test__is_namespace_package_with_package_missing_init. Retrieved 5/8 statements.
# Partially parsed test__is_namespace_package_with_package_containing_py_file. Retrieved 7/12 statements.
# Partially parsed test__is_namespace_package_with_package_containing_setup_cfg. Retrieved 7/12 statements.
# Partially parsed test__is_namespace_package_with_package_containing_pyproject_toml. Retrieved 7/12 statements.
# Partially parsed test__is_namespace_package_with_namespace_package_pkg_resources. Retrieved 7/12 statements.
# Partially parsed test__is_namespace_package_with_namespace_package_pkgutil. Retrieved 7/12 statements.


def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = '.py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = '/existing/package'
    var_1 = True
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = '/existing/package'
    var_1 = True
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/existing/package'
    var_1 = True
    var_2 = 'module.py'
    var_3 = ''
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = '/existing/package'
    var_1 = True
    var_2 = 'setup.cfg'
    var_3 = ''
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = '/existing/package'
    var_1 = True
    var_2 = 'pyproject.toml'
    var_3 = ''
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = '/existing/package'
    var_1 = True
    var_2 = '__init__.py'
    var_3 = "__import__('pkg_resources').declare_namespace(__name__)"
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = '/existing/package'
    var_1 = True
    var_2 = '__init__.py'
    var_3 = "__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_predicate_at_line_26. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = 'src'
    var_2 = 'module'
    var_3 = [var_2]
    var_4 = False



# Parsed testcases at query #15
#--------------------------

# Partially parsed test__src_path_is_module_returns_true_for_valid_module. Retrieved 2/5 statements.
# Partially parsed test__src_path_is_module_returns_false_for_non_matching_name. Retrieved 3/6 statements.
# Partially parsed test__src_path_is_module_returns_false_for_non_directory. Retrieved 2/5 statements.
# Partially parsed test__src_path_is_module_returns_false_for_case_insensitive_match. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'valid_module'
    var_1 = True

def test_case_0():
    var_0 = 'different_name'
    var_1 = True
    var_2 = 'valid_module'

def test_case_0():
    var_0 = 'valid_module'
    var_1 = False

def test_case_0():
    var_0 = 'Valid_Module'
    var_1 = True
    var_2 = 'valid_module'



# Parsed testcases at query #16
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test*'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'testfile'
    var_4 = module_1._forced_separate(var_3, var_2)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test*'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = '.testfile'
    var_4 = module_1._forced_separate(var_3, var_2)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'other*'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'testfile'
    var_4 = module_1._forced_separate(var_3, var_2)
    assert var_4 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'testfile'
    var_4 = module_1._forced_separate(var_3, var_2)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = module_0.Config()
    var_2 = 'testfile'
    var_3 = module_1._forced_separate(var_2, var_1)
    assert var_3 is None



# Parsed testcases at query #17
#--------------------------

# Partially parsed test__src_path_with_nonexistent_module. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_module_in_src_paths. Retrieved 4/11 statements.
# Partially parsed test__src_path_with_nested_module. Retrieved 5/15 statements.
# Partially parsed test__src_path_with_namespace_package. Retrieved 10/23 statements.
# Partially parsed test__src_path_with_explicit_namespace_package. Retrieved 6/16 statements.
# Partially parsed test__src_path_with_root_module_in_src_path. Retrieved 4/11 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = ''
    var_1 = []
    var_2 = module_0.Config()
    var_3 = module_1._src_path(var_0, var_2)
    assert var_3 is None

def test_case_0():
    var_0 = 'nonexistent_module'
    var_1 = '/tmp'

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = True
    var_2 = 'module.py'
    var_3 = 'module'

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = True
    var_2 = 'parent'
    var_3 = 'child.py'
    var_4 = 'parent.child'

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = True
    var_2 = 'namespace'
    var_3 = '__init__.py'
    var_4 = '__import__("pkgutil").extend_path(__path__, __name__)'
    var_5 = 'module.py'
    var_6 = 'py'
    var_7 = {var_6}
    var_8 = frozenset(var_7)
    var_9 = 'namespace.module'

def test_case_0():
    var_0 = '/tmp/project'
    var_1 = True
    var_2 = 'namespace'
    var_3 = 'module.py'
    var_4 = {var_2}
    var_5 = 'namespace.module'

def test_case_0():
    var_0 = '/tmp/module'
    var_1 = True
    var_2 = '__init__.py'
    var_3 = 'module'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test__src_path_with_module_in_src_paths. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_nested_module_in_src_paths. Retrieved 4/8 statements.
# Partially parsed test__src_path_with_auto_identify_namespace_packages. Retrieved 5/9 statements.
# Partially parsed test__src_path_with_src_path_is_module. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_module_not_found. Retrieved 2/6 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Config()
    var_2 = 'module'
    var_3 = module_1._src_path(var_2, var_1)
    assert var_3 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = module_0.Config()
    var_2 = 'module'
    var_3 = module_1._src_path(var_2, var_1)
    assert var_3 is None

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = 'module'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = 'parent'
    var_2 = {var_1}
    var_3 = 'parent.child'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = True
    var_2 = 'py'
    var_3 = {var_2}
    var_4 = 'parent.child'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = 'src'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = 'nonexistent'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_known_pattern_predicate_false. Retrieved 7/9 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.*'
    var_2 = 'section1'
    var_3 = (var_1, var_2)
    var_4 = 'section2'
    var_5 = 'test.module'
    var_6 = module_1._known_pattern(var_5, var_0)
    assert var_6 is None



# Parsed testcases at query #20
#--------------------------

# Partially parsed test__src_path_is_module_returns_true_when_conditions_met. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'module_name'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_true. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = 'module'
    var_2 = [var_1]
    var_3 = False
    var_4 = '/path/to/src'
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = ()



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 2/5 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 1/6 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 2/5 statements.
# Partially parsed test_is_module_no_match. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'module'
    var_1 = 'module.py'

def test_case_0():
    var_0 = 'module'

def test_case_0():
    var_0 = 'module'
    var_1 = 'module/__init__.py'

def test_case_0():
    var_0 = 'nonexistent_module'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_is_module_returns_true_for_py_file. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'module.py'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test__src_path_with_exact_match_in_src_paths. Retrieved 7/11 statements.
# Partially parsed test__src_path_with_nested_module_in_namespace_package. Retrieved 11/18 statements.
# Partially parsed test__src_path_with_auto_identified_namespace_package. Retrieved 11/18 statements.
# Partially parsed test__src_path_with_module_in_root_src_path. Retrieved 7/11 statements.
# Partially parsed test__src_path_with_no_match. Retrieved 7/11 statements.


def test_case_0():
    var_0 = '/project/src'
    var_1 = set()
    var_2 = False
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)
    var_6 = 'module'

def test_case_0():
    var_0 = '/project/src'
    var_1 = 'parent'
    var_2 = {var_1}
    var_3 = False
    var_4 = 'py'
    var_5 = {var_4}
    var_6 = frozenset(var_5)
    var_7 = 'parent.child'
    var_8 = 'child'
    var_9 = '/project/src/parent'
    var_10 = (var_1,)

def test_case_0():
    var_0 = '/project/src'
    var_1 = set()
    var_2 = True
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)
    var_6 = 'namespace.child'
    var_7 = 'child'
    var_8 = '/project/src/namespace'
    var_9 = 'namespace'
    var_10 = (var_9,)

def test_case_0():
    var_0 = '/project/src'
    var_1 = set()
    var_2 = False
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)
    var_6 = 'src'

def test_case_0():
    var_0 = '/project/src'
    var_1 = set()
    var_2 = False
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)
    var_6 = 'nonexistent'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test__is_namespace_package_returns_false_when_not_package. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 'not_a_package'
    var_1 = '.py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_is_module_returns_true_for_py_file. Retrieved 1/3 statements.
# Partially parsed test_is_module_returns_true_for_extension_suffix_file. Retrieved 1/3 statements.
# Partially parsed test_is_module_returns_true_for_init_file. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'module.py'

def test_case_0():
    var_0 = 'module.so'

def test_case_0():
    var_0 = 'package/__init__.py'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 8/19 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = '/path/to/src'
    var_2 = []
    var_3 = True
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = ()
    var_7 = 'module'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_src_path_is_module_returns_true. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'module_name'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_true. Retrieved 9/21 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = {var_0}
    var_2 = False
    var_3 = '/path/to/src'
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = 'test.submodule'
    var_7 = ()
    var_8 = 'test'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test__src_path_is_module_returns_true_when_conditions_met. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'test_module'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test__src_path_with_nested_module_in_namespace_package. Retrieved 8/12 statements.
# Partially parsed test__src_path_with_nested_module_in_auto_identified_namespace_package. Retrieved 7/11 statements.
# Partially parsed test__src_path_with_module_in_src_paths. Retrieved 7/11 statements.
# Partially parsed test__src_path_with_package_in_src_paths. Retrieved 7/11 statements.
# Partially parsed test__src_path_with_src_path_is_module. Retrieved 7/11 statements.
# Partially parsed test__src_path_with_no_match. Retrieved 7/11 statements.


def test_case_0():
    var_0 = '/project'
    var_1 = 'project'
    var_2 = {var_1}
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)
    var_6 = False
    var_7 = 'project.submodule'

def test_case_0():
    var_0 = '/project'
    var_1 = set()
    var_2 = 'py'
    var_3 = {var_2}
    var_4 = frozenset(var_3)
    var_5 = True
    var_6 = 'project.submodule'

def test_case_0():
    var_0 = '/project'
    var_1 = set()
    var_2 = 'py'
    var_3 = {var_2}
    var_4 = frozenset(var_3)
    var_5 = False
    var_6 = 'module'

def test_case_0():
    var_0 = '/project'
    var_1 = set()
    var_2 = 'py'
    var_3 = {var_2}
    var_4 = frozenset(var_3)
    var_5 = False
    var_6 = 'package'

def test_case_0():
    var_0 = '/project'
    var_1 = set()
    var_2 = 'py'
    var_3 = {var_2}
    var_4 = frozenset(var_3)
    var_5 = False
    var_6 = 'project'

def test_case_0():
    var_0 = '/project'
    var_1 = set()
    var_2 = 'py'
    var_3 = {var_2}
    var_4 = frozenset(var_3)
    var_5 = False
    var_6 = 'nonexistent'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test__is_namespace_package_with_non_package_path. Retrieved 4/7 statements.
# Partially parsed test__is_namespace_package_with_empty_directory. Retrieved 3/8 statements.
# Partially parsed test__is_namespace_package_with_init_file. Retrieved 5/12 statements.
# Partially parsed test__is_namespace_package_with_namespace_declaration. Retrieved 5/12 statements.
# Partially parsed test__is_namespace_package_with_pkgutil_declaration. Retrieved 5/12 statements.
# Partially parsed test__is_namespace_package_with_py_file. Retrieved 5/12 statements.
# Partially parsed test__is_namespace_package_with_setup_cfg. Retrieved 5/12 statements.
# Partially parsed test__is_namespace_package_with_pyproject_toml. Retrieved 5/12 statements.


def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = '.py'
    var_2 = {var_1}
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = '.py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)

def test_case_0():
    var_0 = '__init__.py'
    var_1 = ''
    var_2 = '.py'
    var_3 = {var_2}
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '__init__.py'
    var_1 = "__import__('pkg_resources').declare_namespace(__name__)"
    var_2 = '.py'
    var_3 = {var_2}
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '__init__.py'
    var_1 = "__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_2 = '.py'
    var_3 = {var_2}
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = 'module.py'
    var_1 = ''
    var_2 = '.py'
    var_3 = {var_2}
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = ''
    var_2 = '.py'
    var_3 = {var_2}
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = ''
    var_2 = '.py'
    var_3 = {var_2}
    var_4 = frozenset(var_3)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test__is_namespace_package_returns_true_when_predicate_at_line_4_is_true. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 'some_path'
    var_1 = '.py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = '/path/to/src'
    var_2 = 'module'
    var_3 = [var_2]
    var_4 = False
    var_5 = '/path/to/src/module'
    var_6 = ()



# Parsed testcases at query #35
#--------------------------

# Partially parsed test__src_path_with_non_existent_module. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_existing_module_file. Retrieved 3/9 statements.
# Partially parsed test__src_path_with_existing_package. Retrieved 4/12 statements.
# Partially parsed test__src_path_with_nested_module. Retrieved 5/17 statements.
# Partially parsed test__src_path_with_namespace_package. Retrieved 5/11 statements.
# Partially parsed test__src_path_with_auto_identified_namespace_package. Retrieved 8/16 statements.
# Partially parsed test__src_path_with_src_path_as_module. Retrieved 3/11 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Config()
    var_2 = 'module'
    var_3 = module_1._src_path(var_2, var_1)
    assert var_3 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = module_0.Config()
    var_2 = 'module'
    var_3 = module_1._src_path(var_2, var_1)
    assert var_3 is None

def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = 'module'

def test_case_0():
    var_0 = '/existing/path'
    var_1 = '/existing/path/module.py'
    var_2 = 'module'

def test_case_0():
    var_0 = '/existing/path'
    var_1 = '/existing/path/module'
    var_2 = '__init__.py'
    var_3 = 'module'

def test_case_0():
    var_0 = '/existing/path'
    var_1 = '/existing/path/module'
    var_2 = '__init__.py'
    var_3 = 'nested'
    var_4 = 'module.nested'

def test_case_0():
    var_0 = '/existing/path'
    var_1 = 'module'
    var_2 = {var_1}
    var_3 = '/existing/path/module'
    var_4 = 'module.nested'

def test_case_0():
    var_0 = '/existing/path'
    var_1 = True
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)
    var_5 = '/existing/path/module'
    var_6 = 'nested.py'
    var_7 = 'module.nested'

def test_case_0():
    var_0 = '/existing/path/module'
    var_1 = '__init__.py'
    var_2 = 'module'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test__src_path_with_empty_name. Retrieved 1/2 statements.
# Partially parsed test__src_path_with_non_existent_module. Retrieved 1/2 statements.
# Partially parsed test__src_path_with_module_in_src_paths. Retrieved 1/2 statements.
# Partially parsed test__src_path_with_nested_module_in_namespace_package. Retrieved 1/2 statements.
# Partially parsed test__src_path_with_module_in_prefix. Retrieved 3/4 statements.
# Partially parsed test__src_path_with_custom_src_paths. Retrieved 2/5 statements.
# Partially parsed test__src_path_with_namespace_package_configured. Retrieved 2/4 statements.
# Partially parsed test__src_path_with_auto_identified_namespace_package. Retrieved 1/3 statements.


def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = 'non_existent_module'

def test_case_0():
    var_0 = 'existing_module'

def test_case_0():
    var_0 = 'namespace_package.nested_module'

def test_case_0():
    var_0 = 'module'
    var_1 = 'prefix'
    var_2 = (var_1,)

def test_case_0():
    var_0 = '/custom/path'
    var_1 = 'module'

def test_case_0():
    var_0 = 'configured_namespace'
    var_1 = 'configured_namespace.module'

def test_case_0():
    var_0 = 'auto_identified_namespace.module'



# Parsed testcases at query #37
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test_*'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'test_file.py'
    var_4 = module_1._forced_separate(var_3, var_2)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'test_file.py'
    var_4 = module_1._forced_separate(var_3, var_2)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '.hidden'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = '.hidden_file'
    var_4 = module_1._forced_separate(var_3, var_2)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'other_*'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'test_file.py'
    var_4 = module_1._forced_separate(var_3, var_2)
    assert var_4 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = module_0.Config()
    var_2 = 'test_file.py'
    var_3 = module_1._forced_separate(var_2, var_1)
    assert var_3 is None



# Parsed testcases at query #38
#--------------------------

# Partially parsed test__is_namespace_package_with_valid_init_file. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 'valid_namespace_pkg'
    var_1 = '__init__.py'
    var_2 = '__import__("pkg_resources").declare_namespace(__name__)'
    var_3 = '.py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_predicate_at_line_6_evaluates_to_true. Retrieved 7/10 statements.


def test_case_0():
    var_0 = 'some_namespace_package'
    var_1 = '.py'
    var_2 = '.pyx'
    var_3 = '.pxd'
    var_4 = [var_1, var_2, var_3]
    var_5 = frozenset(var_4)
    var_6 = '__init__.py'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_true. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 'test_namespace'
    var_1 = {var_0}
    var_2 = False
    var_3 = '/test/src'
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = 'test_namespace.submodule'
    var_7 = ()



# Parsed testcases at query #41
#--------------------------

# Partially parsed test__src_path_with_non_existent_module. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_existing_module_in_src_paths. Retrieved 3/16 statements.
# Partially parsed test__src_path_with_nested_module. Retrieved 6/20 statements.
# Partially parsed test__src_path_with_namespace_package. Retrieved 8/21 statements.
# Partially parsed test__src_path_with_explicit_namespace_package. Retrieved 5/18 statements.
# Partially parsed test__src_path_with_src_path_as_module. Retrieved 3/12 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = ''
    var_1 = []
    var_2 = module_0.Config()
    var_3 = module_1._src_path(var_0, var_2)
    assert var_3 is None

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = '/tmp'

def test_case_0():
    var_0 = 'mymodule'
    var_1 = '__init__.py'
    var_2 = ''

def test_case_0():
    var_0 = 'parent'
    var_1 = 'child'
    var_2 = True
    var_3 = '__init__.py'
    var_4 = ''
    var_5 = 'parent.child'

def test_case_0():
    var_0 = 'namespace'
    var_1 = 'module.py'
    var_2 = ''
    var_3 = True
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'namespace.module'

def test_case_0():
    var_0 = 'namespace'
    var_1 = 'module.py'
    var_2 = ''
    var_3 = [var_0]
    var_4 = 'namespace.module'

def test_case_0():
    var_0 = 'mymodule'
    var_1 = '__init__.py'
    var_2 = ''



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_known_pattern_predicate_false. Retrieved 6/8 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test.module'
    var_1 = module_0.Config()
    var_2 = 'pattern'
    var_3 = 'placement'
    var_4 = (var_2, var_3)
    var_5 = module_1._known_pattern(var_0, var_1)
    assert var_5 is None



# Parsed testcases at query #43
#--------------------------

# Partially parsed test__is_namespace_package_with_non_package_path. Retrieved 4/7 statements.
# Partially parsed test__is_namespace_package_with_empty_directory. Retrieved 3/8 statements.
# Partially parsed test__is_namespace_package_with_init_file_but_no_namespace_declaration. Retrieved 5/12 statements.
# Partially parsed test__is_namespace_package_with_pkg_resources_declaration. Retrieved 5/12 statements.
# Partially parsed test__is_namespace_package_with_pkgutil_declaration. Retrieved 5/12 statements.
# Partially parsed test__is_namespace_package_with_py_file_present. Retrieved 5/12 statements.
# Partially parsed test__is_namespace_package_with_setup_cfg_present. Retrieved 5/12 statements.
# Partially parsed test__is_namespace_package_with_pyproject_toml_present. Retrieved 5/12 statements.
# Partially parsed test__is_namespace_package_with_no_files_and_no_init. Retrieved 3/8 statements.


def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = '.py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = '.py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)

def test_case_0():
    var_0 = '__init__.py'
    var_1 = '# Regular __init__.py'
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '__init__.py'
    var_1 = "__import__('pkg_resources').declare_namespace(__name__)"
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '__init__.py'
    var_1 = "__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = 'module.py'
    var_1 = '# Some Python code'
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = '[metadata]\nname = test'
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = 'pyproject.toml'
    var_1 = "[build-system]\nrequires = ['setuptools']"
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '.py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_predicate_at_line_6_evaluates_to_true. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 'some_namespace_package'
    var_1 = '.py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)
    var_4 = '__init__.py'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_src_path_is_module_returns_true_for_valid_module. Retrieved 2/5 statements.
# Partially parsed test_src_path_is_module_returns_false_for_non_matching_name. Retrieved 3/6 statements.
# Partially parsed test_src_path_is_module_returns_false_for_non_directory. Retrieved 2/5 statements.
# Partially parsed test_src_path_is_module_returns_false_for_case_sensitive_mismatch. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'valid_module'
    var_1 = True

def test_case_0():
    var_0 = 'different_name'
    var_1 = True
    var_2 = 'valid_module'

def test_case_0():
    var_0 = 'valid_module'
    var_1 = False

def test_case_0():
    var_0 = 'Valid_Module'
    var_1 = True
    var_2 = 'valid_module'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 2/7 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 1/6 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 2/7 statements.
# Partially parsed test_is_module_without_any_file. Retrieved 1/3 statements.


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



# Parsed testcases at query #2
#--------------------------

# Partially parsed test__src_path_with_empty_name. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_nonexistent_module. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_root_module_in_src_paths. Retrieved 1/5 statements.
# Partially parsed test__src_path_with_nested_module_in_src_paths. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_namespace_package. Retrieved 3/7 statements.
# Partially parsed test__src_path_with_auto_identified_namespace_package. Retrieved 6/10 statements.
# Partially parsed test__src_path_with_module_in_src_paths. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_package_in_src_paths. Retrieved 1/5 statements.
# Partially parsed test__src_path_with_src_path_is_module. Retrieved 1/5 statements.


def test_case_0():
    var_0 = ''
    var_1 = '.'

def test_case_0():
    var_0 = 'nonexistent_module'
    var_1 = '.'

def test_case_0():
    var_0 = 'root_module'

def test_case_0():
    var_0 = 'root_module.nested_module'
    var_1 = 'root_module'

def test_case_0():
    var_0 = 'namespace_package'
    var_1 = [var_0]
    var_2 = 'namespace_package.nested_module'

def test_case_0():
    var_0 = 'auto_namespace_package'
    var_1 = True
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)
    var_5 = 'auto_namespace_package.nested_module'

def test_case_0():
    var_0 = 'module_name'
    var_1 = 'module_name.py'

def test_case_0():
    var_0 = 'package_name'

def test_case_0():
    var_0 = 'module_name'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_is_module_returns_true_for_py_file. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'test_module.py'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_predicate_at_line_7_evaluates_to_false. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = '/some/path'
    var_2 = ''
    var_3 = (var_2,)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_known_pattern_matches_first_pattern. Retrieved 5/10 statements.
# Partially parsed test_known_pattern_matches_deepest_module. Retrieved 8/15 statements.
# Partially parsed test_known_pattern_no_match. Retrieved 5/10 statements.
# Partially parsed test_known_pattern_placement_not_in_sections. Retrieved 6/11 statements.
# Partially parsed test_known_pattern_empty_name. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'test.*'
    var_1 = 'placement1'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = 'test.module'

def test_case_0():
    var_0 = 'test.*'
    var_1 = 'placement1'
    var_2 = 'test.module.*'
    var_3 = 'placement2'
    var_4 = {}
    var_5 = {}
    var_6 = {var_1: var_4, var_3: var_5}
    var_7 = 'test.module.sub'

def test_case_0():
    var_0 = 'other.*'
    var_1 = 'placement1'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = 'test.module'

def test_case_0():
    var_0 = 'test.*'
    var_1 = 'placement1'
    var_2 = 'placement2'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'test.module'

def test_case_0():
    var_0 = 'test.*'
    var_1 = 'placement1'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = ''



# Parsed testcases at query #6
#--------------------------

# Partially parsed test__is_namespace_package_with_non_package_path. Retrieved 4/6 statements.
# Partially parsed test__is_namespace_package_with_empty_directory. Retrieved 5/8 statements.
# Partially parsed test__is_namespace_package_with_init_file. Retrieved 7/12 statements.
# Partially parsed test__is_namespace_package_with_non_namespace_init. Retrieved 7/12 statements.
# Partially parsed test__is_namespace_package_with_py_files. Retrieved 7/12 statements.
# Partially parsed test__is_namespace_package_with_setup_cfg. Retrieved 7/12 statements.
# Partially parsed test__is_namespace_package_with_pyproject_toml. Retrieved 7/12 statements.
# Partially parsed test__is_namespace_package_with_pkg_resources_declare. Retrieved 7/12 statements.
# Partially parsed test__is_namespace_package_with_pkgutil_extend_path. Retrieved 7/12 statements.


def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = '.py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = '/tmp/empty_dir'
    var_1 = True
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/tmp/pkg'
    var_1 = True
    var_2 = '__init__.py'
    var_3 = "__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = '/tmp/pkg'
    var_1 = True
    var_2 = '__init__.py'
    var_3 = "print('hello')"
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = '/tmp/pkg'
    var_1 = True
    var_2 = 'module.py'
    var_3 = "print('hello')"
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = '/tmp/pkg'
    var_1 = True
    var_2 = 'setup.cfg'
    var_3 = '[metadata]\nname=test'
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = '/tmp/pkg'
    var_1 = True
    var_2 = 'pyproject.toml'
    var_3 = "[build-system]\nrequires = ['setuptools']"
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = '/tmp/pkg'
    var_1 = True
    var_2 = '__init__.py'
    var_3 = "__import__('pkg_resources').declare_namespace(__name__)"
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = '/tmp/pkg'
    var_1 = True
    var_2 = '__init__.py'
    var_3 = "__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_is_namespace_package_with_valid_files. Retrieved 9/15 statements.


def test_case_0():
    var_0 = 'test_package'
    var_1 = '.py'
    var_2 = '.pyx'
    var_3 = '.pxd'
    var_4 = [var_1, var_2, var_3]
    var_5 = frozenset(var_4)
    var_6 = 'module.py'
    var_7 = ''
    var_8 = 'setup.cfg'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test__is_namespace_package_returns_false_when_path_is_not_a_package. Retrieved 4/7 statements.
# Partially parsed test__is_namespace_package_returns_false_when_init_file_exists_without_namespace_declaration. Retrieved 6/16 statements.
# Partially parsed test__is_namespace_package_returns_true_when_init_file_exists_with_pkg_resources_declaration. Retrieved 6/15 statements.
# Partially parsed test__is_namespace_package_returns_true_when_init_file_exists_with_pkgutil_declaration. Retrieved 6/15 statements.
# Partially parsed test__is_namespace_package_returns_false_when_no_init_file_and_has_source_files. Retrieved 6/15 statements.
# Partially parsed test__is_namespace_package_returns_false_when_no_init_file_and_has_setup_cfg. Retrieved 6/15 statements.
# Partially parsed test__is_namespace_package_returns_false_when_no_init_file_and_has_pyproject_toml. Retrieved 6/15 statements.
# Partially parsed test__is_namespace_package_returns_true_when_no_init_file_and_no_source_files. Retrieved 4/11 statements.


def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = '.py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'test_package'
    var_1 = '__init__.py'
    var_2 = "print('hello')"
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'test_package'
    var_1 = '__init__.py'
    var_2 = '__import__("pkg_resources").declare_namespace(__name__)'
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'test_package'
    var_1 = '__init__.py'
    var_2 = '__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'test_package'
    var_1 = 'module.py'
    var_2 = "print('hello')"
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'test_package'
    var_1 = 'setup.cfg'
    var_2 = '[metadata]\nname = test'
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'test_package'
    var_1 = 'pyproject.toml'
    var_2 = "[build-system]\nrequires = ['setuptools']"
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'test_package'
    var_1 = '.py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 2/5 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 1/6 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 2/5 statements.
# Partially parsed test_is_module_no_match. Retrieved 1/3 statements.


def test_case_0():
    var_0 = '/some/path/module'
    var_1 = '/some/path/module.py'

def test_case_0():
    var_0 = '/some/path/module'

def test_case_0():
    var_0 = '/some/path/module'
    var_1 = '/some/path/module/__init__.py'

def test_case_0():
    var_0 = '/some/path/not_a_module'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 3/7 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 3/7 statements.
# Partially parsed test_is_module_with_init_file. Retrieved 3/7 statements.
# Partially parsed test_is_not_module. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module.py'
    var_2 = ''

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module.so'
    var_2 = ''

def test_case_0():
    var_0 = 'test_module'
    var_1 = '__init__.py'
    var_2 = ''

def test_case_0():
    var_0 = 'test_module'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test__is_namespace_package_returns_true_when_predicate_at_line_4_is_true. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 'valid_namespace_package'
    var_1 = '.py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)



# Parsed testcases at query #12
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = module_0.Config()
    var_2 = 'test.txt'
    var_3 = module_1._forced_separate(var_2, var_1)
    assert var_3 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'test.txt'
    var_4 = module_1._forced_separate(var_3, var_2)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test*'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'test123.txt'
    var_4 = module_1._forced_separate(var_3, var_2)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '.hidden'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = '.hiddenfile'
    var_4 = module_1._forced_separate(var_3, var_2)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = 'data'
    var_2 = [var_0, var_1]
    var_3 = module_0.Config()
    var_4 = 'data.csv'
    var_5 = module_1._forced_separate(var_4, var_3)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test__is_namespace_package_predicate_true. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 'some_namespace_package'
    var_1 = '.py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test__is_namespace_package_without_init_file. Retrieved 4/6 statements.


def test_case_0():
    var_0 = '/some/path'
    var_1 = '.py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test__is_namespace_package_returns_false_when_not_a_package. Retrieved 4/7 statements.
# Partially parsed test__is_namespace_package_returns_false_when_has_init_without_namespace_declaration. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_true_when_has_init_with_pkg_resources_declaration. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_true_when_has_init_with_pkgutil_declaration. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_false_when_has_py_files. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_false_when_has_setup_cfg. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_false_when_has_pyproject_toml. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_true_when_no_init_and_no_files. Retrieved 5/9 statements.


def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = '.py'
    var_2 = {var_1}
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = '/tmp/test_package'
    var_1 = True
    var_2 = '__init__.py'
    var_3 = '# Regular package'
    var_4 = '.py'
    var_5 = {var_4}
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = '/tmp/test_namespace_pkg_resources'
    var_1 = True
    var_2 = '__init__.py'
    var_3 = '__import__("pkg_resources").declare_namespace(__name__)'
    var_4 = '.py'
    var_5 = {var_4}
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = '/tmp/test_namespace_pkgutil'
    var_1 = True
    var_2 = '__init__.py'
    var_3 = '__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    var_4 = '.py'
    var_5 = {var_4}
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = '/tmp/test_package_with_files'
    var_1 = True
    var_2 = 'module.py'
    var_3 = '# Some code'
    var_4 = '.py'
    var_5 = {var_4}
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = '/tmp/test_package_with_setup'
    var_1 = True
    var_2 = 'setup.cfg'
    var_3 = '[metadata]\nname = test'
    var_4 = '.py'
    var_5 = {var_4}
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = '/tmp/test_package_with_pyproject'
    var_1 = True
    var_2 = 'pyproject.toml'
    var_3 = '[build-system]\nrequires = []'
    var_4 = '.py'
    var_5 = {var_4}
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = '/tmp/test_namespace_empty'
    var_1 = True
    var_2 = '.py'
    var_3 = {var_2}
    var_4 = frozenset(var_3)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_false. Retrieved 8/12 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = []
    var_2 = False
    var_3 = '/path/to/src'
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = None
    var_7 = ()



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = '/path/to/src'
    var_2 = 'module'
    var_3 = [var_2]
    var_4 = False
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = ()



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_predicate_at_line_6_evaluates_to_true. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 'some_non_existent_path'
    var_1 = '.py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)
    var_4 = '__init__.py'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_predicate_at_line_6_evaluates_to_true. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'test_namespace_package'
    var_1 = '__init__.py'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = 'src'
    var_2 = 'module'
    var_3 = [var_2]
    var_4 = False
    var_5 = 'src/module'
    var_6 = ()



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_predicate_at_line_13_evaluates_to_true. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 'test_package'
    var_1 = 'module.py'
    var_2 = "print('test')"
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_false. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = []
    var_2 = False
    var_3 = '/some/path'
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = ()



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_true. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = 'module'
    var_2 = {var_1}
    var_3 = 'src'
    var_4 = False



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_is_namespace_package_without_init_file. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 'some_package'
    var_1 = 'module.py'
    var_2 = ''
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test__is_namespace_package_returns_true_for_valid_namespace_package. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 'valid_namespace_package'
    var_1 = 'module.py'
    var_2 = "print('hello')"
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 2/5 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 1/6 statements.
# Partially parsed test_is_module_with_init_file. Retrieved 2/5 statements.
# Partially parsed test_is_module_non_existent. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'module'
    var_1 = 'module.py'

def test_case_0():
    var_0 = 'module'

def test_case_0():
    var_0 = 'module'
    var_1 = 'module/__init__.py'

def test_case_0():
    var_0 = 'non_existent_module'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_is_namespace_package_with_files. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 'test_package'
    var_1 = 'module.py'
    var_2 = '# test'
    var_3 = '.py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 8/18 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = []
    var_2 = True
    var_3 = '/path/to/src'
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = ()
    var_7 = '/path/to/src/module'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 1/3 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 1/3 statements.
# Partially parsed test_is_module_with_init_file. Retrieved 1/3 statements.
# Partially parsed test_is_module_without_any_file. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'module.py'

def test_case_0():
    var_0 = 'module.so'

def test_case_0():
    var_0 = 'package'

def test_case_0():
    var_0 = 'nonexistent'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_forced_separate_predicate_evaluates_to_true. Retrieved 4/5 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test'
    var_2 = 'test_file'
    var_3 = module_1._forced_separate(var_2, var_0)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_true. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 'test_namespace'
    var_1 = {var_0}
    var_2 = False
    var_3 = '/test/src'
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = 'test_namespace.submodule'
    var_7 = ()



# Parsed testcases at query #32
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'unknown.module'
    var_1 = []
    var_2 = []
    var_3 = module_0.Config()
    var_4 = module_1._known_pattern(var_0, var_3)
    assert var_4 is None



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_src_path_is_module_with_valid_module. Retrieved 2/5 statements.
# Partially parsed test_src_path_is_module_with_invalid_module_name. Retrieved 3/6 statements.
# Partially parsed test_src_path_is_module_with_non_directory_path. Retrieved 2/5 statements.
# Partially parsed test_src_path_is_module_with_case_sensitive_mismatch. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'valid_module'
    var_1 = True

def test_case_0():
    var_0 = 'invalid_module'
    var_1 = True
    var_2 = 'different_name'

def test_case_0():
    var_0 = 'not_a_directory'
    var_1 = False

def test_case_0():
    var_0 = 'CaseSensitive'
    var_1 = True
    var_2 = 'casesensitive'



# Parsed testcases at query #34
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = module_0.Config()
    var_3 = 'test.module'
    var_4 = module_1._known_pattern(var_3, var_2)
    assert var_4 is None



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 2/5 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 1/6 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 2/5 statements.
# Partially parsed test_is_module_non_existent. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'module'
    var_1 = 'module.py'

def test_case_0():
    var_0 = 'module'

def test_case_0():
    var_0 = 'package'
    var_1 = 'package/__init__.py'

def test_case_0():
    var_0 = 'nonexistent'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_known_pattern_matches_first_pattern. Retrieved 5/10 statements.
# Partially parsed test_known_pattern_matches_second_pattern. Retrieved 8/15 statements.
# Partially parsed test_known_pattern_no_match. Retrieved 5/10 statements.
# Partially parsed test_known_pattern_placement_not_in_sections. Retrieved 6/11 statements.
# Partially parsed test_known_pattern_empty_name. Retrieved 5/10 statements.
# Partially parsed test_known_pattern_single_part_name. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'test.*'
    var_1 = 'placement1'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = 'test.module'

def test_case_0():
    var_0 = 'test.*'
    var_1 = 'placement1'
    var_2 = 'module.*'
    var_3 = 'placement2'
    var_4 = {}
    var_5 = {}
    var_6 = {var_1: var_4, var_3: var_5}
    var_7 = 'module.test'

def test_case_0():
    var_0 = 'test.*'
    var_1 = 'placement1'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = 'other.module'

def test_case_0():
    var_0 = 'test.*'
    var_1 = 'placement1'
    var_2 = 'placement2'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = 'test.module'

def test_case_0():
    var_0 = 'test.*'
    var_1 = 'placement1'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = ''

def test_case_0():
    var_0 = 'test'
    var_1 = 'placement1'
    var_2 = {}
    var_3 = {var_1: var_2}



# Parsed testcases at query #37
#--------------------------

# Partially parsed test__is_namespace_package_without_init_file. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 'test_namespace'
    var_1 = 'module1.py'
    var_2 = ''
    var_3 = '.py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_src_path_is_module_when_name_matches_and_is_directory_and_exists_case_sensitive. Retrieved 2/4 statements.
# Partially parsed test_src_path_is_not_module_when_name_does_not_match. Retrieved 2/4 statements.
# Partially parsed test_src_path_is_not_module_when_not_directory. Retrieved 2/4 statements.
# Partially parsed test_src_path_is_not_module_when_does_not_exist_case_sensitive. Retrieved 2/4 statements.


def test_case_0():
    var_0 = '/some/path/module_name'
    var_1 = 'module_name'

def test_case_0():
    var_0 = '/some/path/different_name'
    var_1 = 'module_name'

def test_case_0():
    var_0 = '/some/path/module_name.py'
    var_1 = 'module_name'

def test_case_0():
    var_0 = '/some/path/module_name'
    var_1 = 'module_name'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_predicate_at_line_16_evaluates_to_true. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'module'
    var_1 = 'src'
    var_2 = 'src/module'
    var_3 = ()



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_is_module_returns_true_for_py_file. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'module.py'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 9/13 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = '/path/to/src'
    var_2 = 'module'
    var_3 = [var_2]
    var_4 = False
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = None
    var_8 = ()



# Parsed testcases at query #42
#--------------------------

# Partially parsed test__is_namespace_package_returns_true_when_predicate_at_line_4_is_true. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 'some_namespace_package'
    var_1 = '.py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = '/path/to/src'
    var_2 = 'module'
    var_3 = [var_2]
    var_4 = False
    var_5 = ()



# Parsed testcases at query #44
#--------------------------

# Partially parsed test__src_path_with_module_in_src_paths. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_nested_module. Retrieved 4/8 statements.
# Partially parsed test__src_path_with_auto_identify_namespace_packages. Retrieved 3/7 statements.
# Partially parsed test__src_path_with_module_not_in_src_paths. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_src_path_is_module. Retrieved 2/6 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Config()
    var_2 = 'module'
    var_3 = module_1._src_path(var_2, var_1)
    assert var_3 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = module_0.Config()
    var_2 = 'module'
    var_3 = module_1._src_path(var_2, var_1)
    assert var_3 is None

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = 'module'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = 'parent'
    var_2 = {var_1}
    var_3 = 'parent.child'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = True
    var_2 = 'parent.child'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = 'nonexistent'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = 'src'



# Parsed testcases at query #45
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test*'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'test_file'
    var_4 = module_1._forced_separate(var_3, var_2)



# Parsed testcases at query #46
#--------------------------

# Partially parsed test__src_path_is_module_returns_true_when_conditions_are_met. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'module_name'



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = '/path/to/src'
    var_2 = 'module'
    var_3 = [var_2]
    var_4 = False
    var_5 = ()



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_src_path_is_module_when_name_matches_and_is_directory_and_exists_case_sensitive. Retrieved 3/6 statements.
# Partially parsed test_src_path_is_not_module_when_name_does_not_match. Retrieved 3/6 statements.
# Partially parsed test_src_path_is_not_module_when_not_directory. Retrieved 3/6 statements.
# Partially parsed test_src_path_is_not_module_when_case_sensitive_exists_fails. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'module_name'
    var_1 = True
    var_2 = 'module_name'

def test_case_0():
    var_0 = 'different_name'
    var_1 = True
    var_2 = 'module_name'

def test_case_0():
    var_0 = 'module_name'
    var_1 = False
    var_2 = 'module_name'

def test_case_0():
    var_0 = 'module_name'
    var_1 = True
    var_2 = 'module_name'



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_src_path_is_module_when_module_name_matches_and_path_is_directory_and_exists_case_sensitive. Retrieved 2/4 statements.
# Partially parsed test_src_path_is_not_module_when_module_name_does_not_match. Retrieved 2/4 statements.
# Partially parsed test_src_path_is_not_module_when_path_is_not_directory. Retrieved 2/4 statements.
# Partially parsed test_src_path_is_not_module_when_path_does_not_exist_case_sensitive. Retrieved 2/4 statements.


def test_case_0():
    var_0 = '/path/to/module'
    var_1 = 'module'

def test_case_0():
    var_0 = '/path/to/module'
    var_1 = 'other_module'

def test_case_0():
    var_0 = '/path/to/file.py'
    var_1 = 'file.py'

def test_case_0():
    var_0 = '/path/to/nonexistent'
    var_1 = 'nonexistent'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_known_pattern_predicate_false. Retrieved 11/13 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test.module'
    var_1 = module_0.Config()
    var_2 = 'pattern1'
    var_3 = 'placement1'
    var_4 = (var_2, var_3)
    var_5 = 'pattern2'
    var_6 = 'placement2'
    var_7 = (var_5, var_6)
    var_8 = 'section1'
    var_9 = 'section2'
    var_10 = module_1._known_pattern(var_0, var_1)
    assert var_10 is None



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 'example_namespace_package'
    var_1 = True
    var_2 = '.py'
    var_3 = {var_2}
    var_4 = frozenset(var_3)



# Parsed testcases at query #52
#--------------------------

# Partially parsed test__src_path_is_module_returns_true. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module'



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_false. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = 'src'
    var_2 = []
    var_3 = False
    var_4 = ()



# Parsed testcases at query #54
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'test'
    var_4 = module_1._forced_separate(var_3, var_2)



# Parsed testcases at query #55
#--------------------------

# Partially parsed test__src_path_with_empty_name. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_non_existent_module. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_existing_module_file. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_existing_package. Retrieved 2/6 statements.
# Partially parsed test__src_path_with_nested_module_in_namespace_package. Retrieved 4/8 statements.
# Partially parsed test__src_path_with_auto_identified_namespace_package. Retrieved 6/10 statements.
# Partially parsed test__src_path_with_src_path_is_module. Retrieved 2/6 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = None
    var_1 = module_0.Config()
    var_2 = 'module'
    var_3 = module_1._src_path(var_2, var_1)
    assert var_3 is None

def test_case_0():
    var_0 = '/some/path'
    var_1 = ''

def test_case_0():
    var_0 = '/some/path'
    var_1 = 'non_existent_module'

def test_case_0():
    var_0 = '/some/path'
    var_1 = 'existing_module'

def test_case_0():
    var_0 = '/some/path'
    var_1 = 'existing_package'

def test_case_0():
    var_0 = '/some/path'
    var_1 = 'namespace_package'
    var_2 = {var_1}
    var_3 = 'namespace_package.nested_module'

def test_case_0():
    var_0 = '/some/path'
    var_1 = True
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)
    var_5 = 'auto_namespace.nested_module'

def test_case_0():
    var_0 = '/some/path'
    var_1 = 'module_name'



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_true. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 'test_namespace'
    var_1 = [var_0]
    var_2 = False
    var_3 = '/test/src'
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = 'test_namespace.submodule'
    var_7 = ()



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 4/9 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'module.submodule'
    var_2 = '/path/to/src'
    var_3 = ()



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_false. Retrieved 10/28 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = []
    var_2 = False
    var_3 = '/path/to/src'
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = ()
    var_7 = '.'
    var_8 = 1
    var_9 = *var_6



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_known_pattern_exact_match. Retrieved 6/8 statements.
# Partially parsed test_known_pattern_partial_match. Retrieved 6/8 statements.
# Partially parsed test_known_pattern_no_match. Retrieved 6/8 statements.
# Partially parsed test_known_pattern_multiple_patterns. Retrieved 9/11 statements.
# Partially parsed test_known_pattern_placement_not_in_sections. Retrieved 7/9 statements.
# Partially parsed test_known_pattern_empty_name. Retrieved 6/8 statements.
# Partially parsed test_known_pattern_single_part_name. Retrieved 5/7 statements.
# Partially parsed test_known_pattern_no_patterns. Retrieved 4/6 statements.
# Partially parsed test_known_pattern_no_sections. Retrieved 6/8 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.*'
    var_2 = 'placement1'
    var_3 = (var_1, var_2)
    var_4 = 'test.module'
    var_5 = module_1._known_pattern(var_4, var_0)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.*'
    var_2 = 'placement1'
    var_3 = (var_1, var_2)
    var_4 = 'test.module.submodule'
    var_5 = module_1._known_pattern(var_4, var_0)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'other.*'
    var_2 = 'placement1'
    var_3 = (var_1, var_2)
    var_4 = 'test.module'
    var_5 = module_1._known_pattern(var_4, var_0)
    assert var_5 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.*'
    var_2 = 'placement1'
    var_3 = (var_1, var_2)
    var_4 = 'other.*'
    var_5 = 'placement2'
    var_6 = (var_4, var_5)
    var_7 = 'test.module'
    var_8 = module_1._known_pattern(var_7, var_0)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.*'
    var_2 = 'placement1'
    var_3 = (var_1, var_2)
    var_4 = 'placement2'
    var_5 = 'test.module'
    var_6 = module_1._known_pattern(var_5, var_0)
    assert var_6 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.*'
    var_2 = 'placement1'
    var_3 = (var_1, var_2)
    var_4 = ''
    var_5 = module_1._known_pattern(var_4, var_0)
    assert var_5 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test'
    var_2 = 'placement1'
    var_3 = (var_1, var_2)
    var_4 = module_1._known_pattern(var_1, var_0)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'placement1'
    var_2 = 'test.module'
    var_3 = module_1._known_pattern(var_2, var_0)
    assert var_3 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.*'
    var_2 = 'placement1'
    var_3 = (var_1, var_2)
    var_4 = 'test.module'
    var_5 = module_1._known_pattern(var_4, var_0)
    assert var_5 is None



# Parsed testcases at query #60
#--------------------------

# Partially parsed test__is_namespace_package_returns_true_when_init_file_exists_with_namespace_declaration. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 'some_namespace_package'
    var_1 = '.py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)
    var_4 = '__init__.py'
    var_5 = "__import__('pkg_resources').declare_namespace(__name__)"



# Parsed testcases at query #61
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test*'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'test_file'
    var_4 = module_1._forced_separate(var_3, var_2)



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_src_path_is_module_returns_true_for_valid_module. Retrieved 2/5 statements.
# Partially parsed test_src_path_is_module_returns_false_for_file. Retrieved 2/5 statements.
# Partially parsed test_src_path_is_module_returns_false_for_different_name. Retrieved 3/6 statements.
# Partially parsed test_src_path_is_module_returns_false_for_case_insensitive_match. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'valid_module'
    var_1 = True

def test_case_0():
    var_0 = 'file.py'
    var_1 = False

def test_case_0():
    var_0 = 'module'
    var_1 = True
    var_2 = 'different_name'

def test_case_0():
    var_0 = 'Module'
    var_1 = True
    var_2 = 'module'



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_is_module_returns_true_for_py_file. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'module.py'



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 9/13 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = '/path/to/src'
    var_2 = 'module'
    var_3 = [var_2]
    var_4 = False
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = None
    var_8 = ()



# Parsed testcases at query #65
#--------------------------

# Partially parsed test__is_namespace_package_returns_false_when_path_is_not_a_package. Retrieved 4/7 statements.
# Partially parsed test__is_namespace_package_returns_false_when_init_file_exists_without_namespace_declaration. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_true_when_init_file_has_pkg_resources_declaration. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_true_when_init_file_has_pkgutil_declaration. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_false_when_package_contains_source_files. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_false_when_package_contains_setup_cfg. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_false_when_package_contains_pyproject_toml. Retrieved 7/13 statements.
# Partially parsed test__is_namespace_package_returns_true_for_empty_directory. Retrieved 5/9 statements.


def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = '.py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = '/tmp/test_package'
    var_1 = True
    var_2 = '__init__.py'
    var_3 = '# Regular package'
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = '/tmp/test_namespace_package'
    var_1 = True
    var_2 = '__init__.py'
    var_3 = '__import__("pkg_resources").declare_namespace(__name__)'
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = '/tmp/test_namespace_package'
    var_1 = True
    var_2 = '__init__.py'
    var_3 = '__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = '/tmp/test_package_with_files'
    var_1 = True
    var_2 = 'module.py'
    var_3 = '# Some code'
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = '/tmp/test_package_with_setup'
    var_1 = True
    var_2 = 'setup.cfg'
    var_3 = '[metadata]\nname=test'
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = '/tmp/test_package_with_pyproject'
    var_1 = True
    var_2 = 'pyproject.toml'
    var_3 = '[build-system]\nrequires = []'
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = '/tmp/empty_namespace_package'
    var_1 = True
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_init_file_exists. Retrieved 1/3 statements.


def test_case_0():
    var_0 = '/some/path/__init__.py'



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_true. Retrieved 9/12 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'module.submodule'
    var_1 = 'module'
    var_2 = (var_1,)
    var_3 = False
    var_4 = '.py'
    var_5 = (var_4,)
    var_6 = ()
    var_7 = module_0.Config()
    var_8 = ()



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_is_module_returns_true_for_py_file. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'test_module.py'



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_predicate_at_line_6_evaluates_to_true. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'some_path'
    var_1 = '__init__.py'



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 9/19 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = '/path/to/src'
    var_2 = []
    var_3 = True
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = ()
    var_7 = 'module'
    var_8 = [var_4]



# Parsed testcases at query #71
#--------------------------

# Partially parsed test__is_module_returns_true_for_py_file. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'test_module.py'



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_known_pattern_matches_first_pattern. Retrieved 4/9 statements.
# Partially parsed test_known_pattern_matches_second_pattern. Retrieved 6/13 statements.
# Partially parsed test_known_pattern_no_match. Retrieved 4/9 statements.
# Partially parsed test_known_pattern_placement_not_in_sections. Retrieved 5/10 statements.
# Partially parsed test_known_pattern_empty_name. Retrieved 4/9 statements.
# Partially parsed test_known_pattern_single_part_name. Retrieved 3/8 statements.
# Partially parsed test_known_pattern_multi_part_name. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'test.*'
    var_1 = 'section1'
    var_2 = [var_1]
    var_3 = 'test.module'

def test_case_0():
    var_0 = 'test.*'
    var_1 = 'section1'
    var_2 = 'module.*'
    var_3 = 'section2'
    var_4 = [var_1, var_3]
    var_5 = 'test.module'

def test_case_0():
    var_0 = 'other.*'
    var_1 = 'section1'
    var_2 = [var_1]
    var_3 = 'test.module'

def test_case_0():
    var_0 = 'test.*'
    var_1 = 'section2'
    var_2 = 'section1'
    var_3 = [var_2]
    var_4 = 'test.module'

def test_case_0():
    var_0 = 'test.*'
    var_1 = 'section1'
    var_2 = [var_1]
    var_3 = ''

def test_case_0():
    var_0 = 'test'
    var_1 = 'section1'
    var_2 = [var_1]

def test_case_0():
    var_0 = 'test.*module'
    var_1 = 'section1'
    var_2 = [var_1]
    var_3 = 'test.middle.module'



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 9/20 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = '/path/to/src'
    var_2 = []
    var_3 = True
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = ()
    var_7 = '/path/to/src/module'
    var_8 = '__init__.py'



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 9/15 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = 'src'
    var_2 = 'module'
    var_3 = [var_2]
    var_4 = False
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = 'src/module'
    var_8 = ()



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_false. Retrieved 11/26 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = ()
    var_2 = False
    var_3 = '/path/to/src'
    var_4 = '.py'
    var_5 = (var_4,)
    var_6 = ()
    var_7 = 'module'
    var_8 = 'module'
    var_9 = 'submodule'
    var_10 = [var_9]



