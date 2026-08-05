####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_test_src_path_returns_none_when_no_match. Retrieved 3/11 statements.
# Partially parsed test_test_src_path_finds_module_in_src_path. Retrieved 3/21 statements.
# Partially parsed test_test_src_path_handles_nested_namespace_packages. Retrieved 6/25 statements.
# Partially parsed test_test_src_path_handles_single_file_module. Retrieved 2/18 statements.


def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = 'non_existent_module'
    var_2 = '/tmp/fake_dir'

def test_case_0():
    var_0 = 'src'
    var_1 = 'my_module'
    var_2 = '__init__.py'

def test_case_0():
    var_0 = 'src'
    var_1 = 'my_namespace'
    var_2 = 'sub_module.py'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = 'my_namespace.sub_module'

def test_case_0():
    var_0 = 'src'
    var_1 = 'standalone'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_is_module_returns_true_for_py_file. Retrieved 2/6 statements.
# Partially parsed test_is_module_returns_true_for_extension_suffix. Retrieved 2/9 statements.
# Partially parsed test_is_module_returns_true_for_init_py. Retrieved 2/6 statements.
# Partially parsed test_is_module_returns_false_when_no_files_exist. Retrieved 1/3 statements.


def test_case_0():
    var_0 = '/tmp/test_module'
    var_1 = '/tmp/test_module.py'

def test_case_0():
    var_0 = '/tmp/test_ext'
    var_1 = 0

def test_case_0():
    var_0 = '/tmp/package'
    var_1 = '/tmp/package/__init__.py'

def test_case_0():
    var_0 = '/tmp/non_existent_module'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_src_path_predicate_false_via_namespace_not_in_config. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 'other.namespace'
    var_1 = '/tmp/dummy'
    var_2 = True
    var_3 = 'my_module.sub_module'
    var_4 = ()



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_src_path_predicate_true_via_namespace_packages. Retrieved 5/16 statements.


def test_case_0():
    var_0 = 'my_package'
    var_1 = '/tmp/src'
    var_2 = True
    var_3 = 'my_package.submodule'
    var_4 = ()



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_src_path_predicate_false_via_namespace_not_in_config. Retrieved 5/18 statements.


def test_case_0():
    var_0 = '.py'
    var_1 = '/tmp/dummy_src'
    var_2 = True
    var_3 = 'root.submodule'
    var_4 = ()



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_src_path_predicate_true. Retrieved 5/58 statements.


def test_case_0():
    var_0 = 'src'
    var_1 = 'foo'
    var_2 = 'content'
    var_3 = 'test'
    var_4 = 'dummy'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_src_path_is_module_success. Retrieved 2/7 statements.
# Partially parsed test_src_path_is_module_name_mismatch. Retrieved 2/7 statements.
# Partially parsed test_src_path_is_module_not_a_directory. Retrieved 2/7 statements.
# Partially parsed test_src_path_is_module_not_exists. Retrieved 2/7 statements.


def test_case_0():
    var_0 = '/path/to/my_module'
    var_1 = 'my_module'

def test_case_0():
    var_0 = '/path/to/wrong_name'
    var_1 = 'my_module'

def test_case_0():
    var_0 = '/path/to/my_module.py'
    var_1 = 'my_module'

def test_case_0():
    var_0 = '/path/to/non_existent'
    var_1 = 'my_module'



# Parsed testcases at query #8
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_src_path_predicate_false_when_src_paths_provided. Retrieved 4/16 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = 'root'
    var_2 = (var_1,)
    var_3 = '/tmp/src'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_src_path_returns_none_if_no_match. Retrieved 7/11 statements.
# Partially parsed test_src_path_returns_firstparty_if_module_exists_in_src_path. Retrieved 9/20 statements.
# Partially parsed test_src_path_handles_nested_namespace_packages. Retrieved 11/26 statements.
# Partially parsed test_src_path_with_prefix_accumulation. Retrieved 11/34 statements.


def test_case_0():
    var_0 = '/tmp/nonexistent_root'
    var_1 = set()
    var_2 = False
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)
    var_6 = 'missing_module'

def test_case_0():
    var_0 = '/tmp/src_root'
    var_1 = True
    var_2 = 'my_module.py'
    var_3 = set()
    var_4 = False
    var_5 = 'py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)
    var_8 = 'my_module'

def test_case_0():
    var_0 = '/tmp/namespace_root'
    var_1 = 'parent'
    var_2 = 'child'
    var_3 = True
    var_4 = '__init__.py'
    var_5 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_6 = {var_1}
    var_7 = 'py'
    var_8 = [var_7]
    var_9 = frozenset(var_8)
    var_10 = 'parent.child'

def test_case_0():
    var_0 = '/tmp/prefix_test'
    var_1 = True
    var_2 = 'a'
    var_3 = '__init__.py'
    var_4 = 'b'
    var_5 = {var_2}
    var_6 = False
    var_7 = 'py'
    var_8 = [var_7]
    var_9 = frozenset(var_8)
    var_10 = 'a.b'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_src_path_predicate_true. Retrieved 7/19 statements.


def test_case_0():
    var_0 = 'my_package'
    var_1 = '/tmp/src'
    var_2 = True
    var_3 = 'my_package'
    var_4 = 'my_package.sub_module'
    var_5 = 'my_package.sub_module'
    var_6 = ()



# Parsed testcases at query #12
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'src/utils'
    var_1 = [var_0]
    var_2 = 'other/path'
    var_3 = module_0.Config()
    var_4 = module_1._forced_separate(var_2, var_3)
    assert var_4 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'src/utils'
    var_1 = [var_0]
    var_2 = 'src/utils/helper'
    var_3 = module_0.Config()
    var_4 = module_1._forced_separate(var_2, var_3)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'tests/*'
    var_1 = [var_0]
    var_2 = 'tests/integration/test_logic'
    var_3 = module_0.Config()
    var_4 = module_1._forced_separate(var_2, var_3)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'config'
    var_1 = [var_0]
    var_2 = '.config/settings'
    var_3 = module_0.Config()
    var_4 = module_1._forced_separate(var_2, var_3)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'abc'
    var_1 = 'xyz'
    var_2 = [var_0, var_1]
    var_3 = 'xyz/data'
    var_4 = module_0.Config()
    var_5 = module_1._forced_separate(var_3, var_4)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_src_path_returns_none_when_no_match_found. Retrieved 2/10 statements.
# Partially parsed test_src_path_returns_firstparty_when_module_exists_in_src_path. Retrieved 3/20 statements.
# Partially parsed test_src_path_handles_nested_namespace_packages. Retrieved 8/30 statements.
# Partially parsed test_src_path_handles_auto_identify_namespace_packages. Retrieved 8/30 statements.


def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = 'my_module'

def test_case_0():
    var_0 = 'my_module'
    var_1 = '__init__.py'
    var_2 = 'firstparty'

def test_case_0():
    var_0 = 'parent'
    var_1 = '__init__.py'
    var_2 = "__import__('pkg_resources').declare_namespace(__name__)"
    var_3 = 'child'
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = 'firstparty'
    var_7 = 'parent.child'

def test_case_0():
    var_0 = 'parent'
    var_1 = '__init__.py'
    var_2 = "__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_3 = 'child'
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = 'firstparty'
    var_7 = 'parent.child'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_forced_separate_does_not_match_unrelated_string. Retrieved 5/7 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/logs/*'
    var_1 = 'temp*'
    var_2 = [var_0, var_1]
    var_3 = module_0.Config()
    var_4 = 'data/file.txt'
    var_5 = module_1._forced_separate(var_4, var_3)
    assert var_5 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/logs/*'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = '/logs/error.log'
    var_4 = module_1._forced_separate(var_3, var_2)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'temp'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'template.txt'
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
    var_0 = '/a/*'
    var_1 = '/b/*'
    var_2 = [var_0, var_1]
    var_3 = module_0.Config()
    var_4 = '/b/file.txt'
    var_5 = module_1._forced_separate(var_4, var_3)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/important/*'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'random_name'
    var_4 = 'not_important/file.txt'
    var_5 = module_1._forced_separate(var_4, var_2)
    assert var_5 is None



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_src_path_returns_none_if_no_match. Retrieved 6/19 statements.
# Partially parsed test_src_path_returns_firstparty_for_direct_module_match. Retrieved 14/32 statements.
# Partially parsed test_src_path_handles_nested_namespace_packages. Retrieved 17/35 statements.


def test_case_0():
    var_0 = '/tmp/src'
    var_1 = set()
    var_2 = False
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = 'nonexistent_module'

def test_case_0():
    var_0 = 'pathlib.Path.exists'
    var_1 = True
    var_2 = 'pathlib.Path.is_dir'
    var_3 = 'builtins.exists_case_sensitive'
    var_4 = '__main__._is_module'
    var_5 = '/tmp/src'
    var_6 = set()
    var_7 = False
    var_8 = 'py'
    var_9 = [var_8]
    var_10 = frozenset(var_9)
    var_11 = 'pathlib.Path.resolve'
    var_12 = lambda self: self
    var_13 = 'my_mod'

def test_case_0():
    var_0 = 'pathlib.Path.exists'
    var_1 = True
    var_2 = 'pathlib.Path.is_dir'
    var_3 = 'builtins.exists_case_sensitive'
    var_4 = 'pathlib.Path.resolve'
    var_5 = lambda self: self
    var_6 = '__main__._is_namespace_package'
    var_7 = '__main__._src_path'
    var_8 = 'nested'
    var_9 = 'found'
    var_10 = (var_8, var_9)
    var_11 = '/tmp/src'
    var_12 = set()
    var_13 = 'py'
    var_14 = [var_13]
    var_15 = frozenset(var_14)
    var_16 = 'pkg.submod'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_src_path_returns_none_when_no_match_found. Retrieved 2/10 statements.
# Partially parsed test_src_path_returns_firstparty_when_module_is_found. Retrieved 4/27 statements.
# Partially parsed test_src_path_handles_nested_namespace_packages. Retrieved 6/24 statements.
# Partially parsed test_src_path_identifies_module_at_root_of_src_path. Retrieved 3/23 statements.


def test_case_0():
    var_0 = '/tmp/nonexistent_src'
    var_1 = 'nonexistent_module'

def test_case_0():
    var_0 = 'src'
    var_1 = 'my_module'
    var_2 = '__init__.py'
    var_3 = 'sections'

def test_case_0():
    var_0 = 'src'
    var_1 = 'pkg'
    var_2 = 'submodule.py'
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = 'pkg.submodule'

def test_case_0():
    var_0 = 'standalone_module.py'
    var_1 = 'sections'
    var_2 = 'standalone_module'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_src_path_predicate_false. Retrieved 2/14 statements.


def test_case_0():
    var_0 = '/tmp/dummy'
    var_1 = 'some.module'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_src_path_predicate_is_false_when_src_paths_provided. Retrieved 4/18 statements.


def test_case_0():
    var_0 = '/tmp'
    var_1 = 'some_module'
    var_2 = 'some'
    var_3 = (var_2,)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_src_path_is_module_true. Retrieved 2/9 statements.
# Partially parsed test_src_path_is_module_false_name_mismatch. Retrieved 2/9 statements.
# Partially parsed test_src_path_is_module_false_not_a_directory. Retrieved 2/9 statements.
# Partially parsed test_src_path_is_module_false_not_exists. Retrieved 2/9 statements.


def test_case_0():
    var_0 = '/path/to/my_module'
    var_1 = 'my_module'

def test_case_0():
    var_0 = '/path/to/my_module'
    var_1 = 'my_module'

def test_case_0():
    var_0 = '/path/to/my_module'
    var_1 = 'my_module'

def test_case_0():
    var_0 = '/path/to/my_module'
    var_1 = 'my_module'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_src_path_evaluates_true_via_namespace_packages. Retrieved 9/24 statements.


def test_case_0():
    var_0 = '/tmp/test_pkg'
    var_1 = True
    var_2 = 'my_package.sub_module'
    var_3 = {var_2}
    var_4 = False
    var_5 = 'py'
    var_6 = [var_5]
    var_7 = 'my_package'
    var_8 = ()



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_src_path_is_module_returns_true_when_all_conditions_met. Retrieved 2/7 statements.
# Partially parsed test_src_path_is_module_success. Retrieved 1/9 statements.
# Failed to parse test_src_path_is_module_evaluates_true.


def test_case_0():
    var_0 = 'my_module'
    var_1 = 'my_module'

def test_case_0():
    var_0 = 'my_module'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_src_path_predicate_false. Retrieved 7/21 statements.


def test_case_0():
    var_0 = '/tmp/src'
    var_1 = set()
    var_2 = True
    var_3 = '.py'
    var_4 = {var_3}
    var_5 = 'a.b'
    var_6 = ()



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_src_path_is_module_evaluates_to_true. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'my_module'
    var_1 = True
    var_2 = 'my_module'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_forced_separate_matches_dot_prefix_pattern. Retrieved 3/6 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/logs/*.log'
    var_1 = '/tmp/*'
    var_2 = [var_0, var_1]
    var_3 = module_0.Config()
    var_4 = 'data.txt'
    var_5 = module_1._forced_separate(var_4, var_3)
    assert var_5 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/logs/*.log'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = '/logs/error.log'
    var_4 = module_1._forced_separate(var_3, var_2)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/usr/bin'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = '/usr/bin/python'
    var_4 = module_1._forced_separate(var_3, var_2)

import isort.settings as module_0

def test_case_0():
    var_0 = '/var/log'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = '/var/log/syslog'

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/etc'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = '/etc/passwd'
    var_4 = module_1._forced_separate(var_3, var_2)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '.hidden_pattern*'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = '/.hidden_pattern_file'
    var_4 = module_1._forced_separate(var_3, var_2)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = '/a/*'
    var_1 = '/a/b/*'
    var_2 = [var_0, var_1]
    var_3 = module_0.Config()
    var_4 = '/a/b/c.txt'
    var_5 = module_1._forced_separate(var_4, var_3)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_known_pattern_returns_none_when_no_match. Retrieved 5/15 statements.
# Partially parsed test_known_pattern_returns_none_when_placement_not_in_sections. Retrieved 5/15 statements.
# Partially parsed test_known_pattern_returns_match_on_exact_module_name. Retrieved 4/14 statements.
# Partially parsed test_known_pattern_returns_match_on_parent_module_name. Retrieved 4/14 statements.
# Partially parsed test_known_pattern_prioritizes_longer_matches. Retrieved 6/18 statements.


def test_case_0():
    var_0 = 'abc'
    var_1 = 'section1'
    var_2 = 'section2'
    var_3 = [var_2]
    var_4 = 'xyz.def'

def test_case_0():
    var_0 = 'abc'
    var_1 = 'section1'
    var_2 = 'section2'
    var_3 = [var_2]
    var_4 = 'abc.def'

def test_case_0():
    var_0 = 'a\\.b'
    var_1 = 'section1'
    var_2 = [var_1]
    var_3 = 'a.b'

def test_case_0():
    var_0 = 'a'
    var_1 = 'section1'
    var_2 = [var_1]
    var_3 = 'a.b.c'

def test_case_0():
    var_0 = 'a'
    var_1 = 'short_section'
    var_2 = 'a\\.b'
    var_3 = 'long_section'
    var_4 = [var_1, var_3]
    var_5 = 'a.b'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_src_path_predicate_true. Retrieved 8/42 statements.


def test_case_0():
    var_0 = 'fake_dir'
    var_1 = 'my_module'
    var_2 = 'content'
    var_3 = 'my_module'
    var_4 = 'content'
    var_5 = 'extra'
    var_6 = 'dummy'
    var_7 = ()



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_src_path_predicate_false_when_src_paths_provided. Retrieved 4/14 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = '/tmp/test_path'
    var_2 = 'test_module'
    var_3 = ()



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_forced_separate_predicate_evaluates_to_true. Retrieved 10/20 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_pattern'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'test_pattern_suffix'
    var_4 = 0
    var_5 = var_2.forced_separate[var_4]
    var_6 = '*'
    var_7 = f'{var_5}*'
    var_8 = '.'
    var_9 = True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_is_module_logic_with_py_file_exists. Retrieved 2/12 statements.
# Partially parsed test_is_module_logic_with_extension_exists. Retrieved 3/13 statements.
# Partially parsed test_is_module_logic_with_init_exists. Retrieved 2/12 statements.
# Partially parsed test_is_module_returns_false_when_nothing_exists. Retrieved 1/9 statements.


def test_case_0():
    pass

def test_case_0():
    var_0 = '/fake/path/mymodule'
    var_1 = '.py'

def test_case_0():
    var_0 = '/fake/path/mymodule'
    var_1 = 0
    var_2 = str(var_0)

def test_case_0():
    var_0 = '/fake/path/mypackage'
    var_1 = '__init__.py'

def test_case_0():
    var_0 = '/fake/path/nonexistent'
    assert var_0 is False



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_src_path_returns_none_when_no_match. Retrieved 4/13 statements.
# Partially parsed test_src_path_returns_firstparty_for_existing_module. Retrieved 4/21 statements.
# Partially parsed test_src_path_handles_nested_modules_in_namespace. Retrieved 9/32 statements.
# Partially parsed test_src_path_with_direct_module_file_as_src_root. Retrieved 3/17 statements.


def test_case_0():
    var_0 = '/tmp/nonexistent_module'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = 'my_module'

def test_case_0():
    var_0 = 'src'
    var_1 = 'my_module.py'
    var_2 = 'FIRSTPARTY'
    var_3 = 'my_module'

def test_case_0():
    var_0 = 'src'
    var_1 = 'my_package'
    var_2 = '__init__.py'
    var_3 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_4 = 'sub_module'
    var_5 = 'py'
    var_6 = [var_5]
    var_7 = 'FIRSTPARTY'
    var_8 = 'my_package.sub_module'

def test_case_0():
    var_0 = 'standalone.py'
    var_1 = 'FIRSTPARTY'
    var_2 = 'standalone'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_src_path_namespace_package_true. Retrieved 9/31 statements.


def test_case_0():
    var_0 = 'src'
    var_1 = 'my_package'
    var_2 = '__init__.py'
    var_3 = {var_1}
    var_4 = False
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = 'my_package.submodule'
    var_8 = ()



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_src_path_predicate_true. Retrieved 6/25 statements.


def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'my_module'
    var_2 = var_0 / var_1
    var_3 = 'abc'
    var_4 = 'abc'
    var_5 = ()



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_is_namespace_package_not_a_directory. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_regular_package_with_init_no_namespace_marker. Retrieved 7/15 statements.
# Partially parsed test_is_namespace_package_regular_package_with_pkg_resources_marker. Retrieved 7/15 statements.
# Partially parsed test_is_namespace_package_regular_package_with_pkgutil_marker. Retrieved 7/15 statements.
# Partially parsed test_is_namespace_package_namespace_without_init_but_has_py_files. Retrieved 6/15 statements.
# Partially parsed test_is_namespace_package_namespace_without_init_and_no_other_files. Retrieved 5/10 statements.
# Partially parsed test_is_namespace_package_namespace_without_init_with_config_files. Retrieved 6/15 statements.


def test_case_0():
    var_0 = 'non_existent_directory'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'test_pkg_regular'
    var_1 = True
    var_2 = '__init__.py'
    var_3 = b"print('hello')"
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = 'test_pkg_pkg_resources'
    var_1 = True
    var_2 = '__init__.py'
    var_3 = b"__import__('pkg_resources').declare_namespace(__name__)"
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = 'test_pkg_pkgutil'
    var_1 = True
    var_2 = '__init__.py'
    var_3 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = 'test_ns_with_py'
    var_1 = True
    var_2 = 'module.py'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'test_ns_empty'
    var_1 = True
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = 'test_ns_with_config'
    var_1 = True
    var_2 = 'pyproject.toml'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_is_namespace_package_returns_false_when_filenames_exists. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)
    var_3 = 'module.py'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_src_path_returns_none_when_no_match_found. Retrieved 4/13 statements.
# Partially parsed test_src_path_returns_firstparty_when_module_exists. Retrieved 4/26 statements.
# Partially parsed test_src_path_handles_nested_modules. Retrieved 7/33 statements.
# Partially parsed test_src_path_with_prefix_logic. Retrieved 7/33 statements.


def test_case_0():
    var_0 = '/tmp/nonexistent_module'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = 'my_module'

def test_case_0():
    var_0 = 'my_module'
    var_1 = '__init__.py'
    var_2 = 'sections'
    var_3 = 'my_module'

def test_case_0():
    var_0 = 'parent'
    var_1 = 'child'
    var_2 = '__init__.py'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = 'sections'
    var_6 = 'parent.child'

def test_case_0():
    var_0 = 'src'
    var_1 = 'my_pkg'
    var_2 = '__init__.py'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = 'sections'
    var_6 = 'my_pkg'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_is_namespace_package_returns_false_when_filenames_found. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 'test_namespace_pkg'
    var_1 = True
    var_2 = 'module.py'
    var_3 = ''
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)



