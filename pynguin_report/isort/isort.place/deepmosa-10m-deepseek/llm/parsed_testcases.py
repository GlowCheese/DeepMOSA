####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_src_path_finds_module_in_src_paths. Retrieved 6/11 statements.
# Partially parsed test_src_path_handles_nested_module_with_namespace_package. Retrieved 7/12 statements.
# Partially parsed test_src_path_returns_none_for_unknown_module. Retrieved 6/11 statements.
# Partially parsed test_src_path_uses_provided_src_paths. Retrieved 5/12 statements.
# Partially parsed test_src_path_handles_src_path_is_module. Retrieved 6/11 statements.
# Partially parsed test_src_path_auto_identifies_namespace_package. Retrieved 6/11 statements.


import isort.place as module_1
import isort.settings as module_0


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/test/path'
    var_3 = [var_2]
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = 'mymodule'
    var_7 = module_1._src_path(var_6, var_1)


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/test/path'
    var_3 = [var_2]
    var_4 = 'mypackage'
    var_5 = 'py'
    var_6 = [var_5]
    var_7 = 'mypackage.nested'
    var_8 = module_1._src_path(var_7, var_1)


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/test/path'
    var_3 = [var_2]
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = 'unknown'
    var_7 = module_1._src_path(var_6, var_1)
    assert var_7 is None


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = '/custom/path'
    var_5 = [var_4]
    var_6 = 'mymodule'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/test/path'
    var_3 = [var_2]
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = 'path'
    var_7 = module_1._src_path(var_6, var_1)


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/test/path'
    var_3 = [var_2]
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = 'namespace.nested'
    var_7 = module_1._src_path(var_6, var_1)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_src_path_is_module_true_for_valid_directory. Retrieved 4/7 statements.
# Partially parsed test_src_path_is_module_false_for_wrong_name. Retrieved 4/7 statements.
# Partially parsed test_src_path_is_module_false_for_file_not_dir. Retrieved 5/8 statements.
# Partially parsed test_src_path_is_module_false_for_case_mismatch. Retrieved 5/8 statements.
# Partially parsed test_src_path_is_module_false_for_all_conditions_failing. Retrieved 4/7 statements.


def test_case_0():
    var_0 = '/some/path/module_name'
    var_1 = [var_0]
    var_2 = True
    var_3 = lambda x: var_2
    var_4 = 'module_name'

def test_case_0():
    var_0 = '/some/path/module_name'
    var_1 = [var_0]
    var_2 = True
    var_3 = lambda x: var_2
    var_4 = 'different_name'

def test_case_0():
    var_0 = '/some/path/module_name'
    var_1 = [var_0]
    var_2 = False
    var_3 = True
    var_4 = lambda x: var_3
    var_5 = 'module_name'

def test_case_0():
    var_0 = '/some/path/module_name'
    var_1 = [var_0]
    var_2 = True
    var_3 = False
    var_4 = lambda x: var_3
    var_5 = 'module_name'

def test_case_0():
    var_0 = '/some/path/module_name'
    var_1 = [var_0]
    var_2 = False
    var_3 = lambda x: var_2
    var_4 = 'different_name'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_known_pattern_matches_configured_pattern. Retrieved 3/14 statements.
# Partially parsed test_known_pattern_no_match. Retrieved 2/10 statements.
# Partially parsed test_known_pattern_placement_not_in_sections. Retrieved 3/11 statements.
# Partially parsed test_known_pattern_matches_longest_module_first. Retrieved 4/15 statements.
# Partially parsed test_known_pattern_empty_name. Retrieved 2/10 statements.
# Partially parsed test_known_pattern_multiple_patterns_first_matches. Retrieved 4/18 statements.


def test_case_0():
    var_0 = 'placement_name'
    var_1 = 'module.submodule.name'
    var_2 = 'Matched configured known pattern '

def test_case_0():
    var_0 = 'placement_name'
    var_1 = 'module.submodule.name'

def test_case_0():
    var_0 = 'placement_name'
    var_1 = 'other_section'
    var_2 = 'module.submodule.name'

def test_case_0():
    var_0 = 'module.submodule'
    var_1 = 'placement_name'
    var_2 = 'module.submodule.name'
    var_3 = 'Matched configured known pattern '

def test_case_0():
    var_0 = 'placement_name'
    var_1 = ''

def test_case_0():
    var_0 = 'placement1'
    var_1 = 'placement2'
    var_2 = 'module.name'
    var_3 = 'Matched configured known pattern '



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 4/9 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/fake/path'
    var_3 = [var_2]
    var_4 = 'mymodule'
    var_5 = [var_2]
    var_6 = ()
    var_7 = 'Found in one of the configured src_paths:'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_known_pattern_predicate_false. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'section1'
    var_1 = [var_0]
    var_2 = 'module.name'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_namespace_in_config_namespace_packages. Retrieved 9/14 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.namespace'
    var_3 = '.py'
    var_4 = '/src'
    var_5 = [var_4]
    var_6 = 'test.namespace.module'
    var_7 = None
    var_8 = 'test'
    var_9 = (var_8,)
    var_10 = module_1._src_path(var_6, var_1, var_7, var_9)
    var_11 = bool(var_10 is not None)
    assert var_11 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_is_namespace_package_with_valid_namespace_package. Retrieved 4/17 statements.
# Partially parsed test_is_namespace_package_with_valid_namespace_package_double_quotes. Retrieved 4/17 statements.
# Partially parsed test_is_namespace_package_with_valid_namespace_package_pkgutil. Retrieved 4/17 statements.
# Partially parsed test_is_namespace_package_with_valid_namespace_package_pkgutil_double_quotes. Retrieved 4/17 statements.
# Partially parsed test_is_namespace_package_without_init_and_no_files. Retrieved 4/16 statements.
# Partially parsed test_is_namespace_package_without_init_but_has_source_files. Retrieved 4/19 statements.
# Partially parsed test_is_namespace_package_without_init_but_has_setup_cfg. Retrieved 4/19 statements.
# Partially parsed test_is_namespace_package_without_init_but_has_pyproject_toml. Retrieved 4/19 statements.
# Partially parsed test_is_namespace_package_with_init_but_no_namespace_marker. Retrieved 4/17 statements.
# Partially parsed test_is_namespace_package_path_not_a_package. Retrieved 4/11 statements.
# Partially parsed test_is_namespace_package_path_does_not_exist_case_sensitive. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'some_path'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'some_path'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'some_path'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'some_path'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'some_path'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'some_path'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'some_path'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'some_path'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'some_path'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'some_path'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'some_path'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_is_namespace_package_with_valid_namespace_package. Retrieved 6/15 statements.
# Partially parsed test_is_namespace_package_with_non_package_path. Retrieved 4/10 statements.
# Partially parsed test_is_namespace_package_with_regular_package_has_init. Retrieved 6/15 statements.
# Partially parsed test_is_namespace_package_with_namespace_package_no_init_but_has_files. Retrieved 6/15 statements.
# Partially parsed test_is_namespace_package_with_namespace_package_no_init_and_no_files. Retrieved 4/11 statements.
# Partially parsed test_is_namespace_package_with_namespace_declare_pkg_resources_single_quotes. Retrieved 6/15 statements.
# Partially parsed test_is_namespace_package_with_namespace_declare_pkg_resources_double_quotes. Retrieved 6/15 statements.
# Partially parsed test_is_namespace_package_with_namespace_extend_path_pkgutil_single_quotes. Retrieved 6/15 statements.
# Partially parsed test_is_namespace_package_with_namespace_extend_path_pkgutil_double_quotes. Retrieved 6/15 statements.
# Partially parsed test_is_namespace_package_with_non_py_extension_file_in_dir. Retrieved 6/15 statements.
# Partially parsed test_is_namespace_package_with_setup_cfg_file_in_dir. Retrieved 6/15 statements.
# Partially parsed test_is_namespace_package_with_pyproject_toml_file_in_dir. Retrieved 6/15 statements.
# Partially parsed test_is_namespace_package_with_py_file_in_dir. Retrieved 6/15 statements.
# Partially parsed test_is_namespace_package_with_init_but_no_namespace_marker. Retrieved 6/15 statements.


def test_case_0():
    var_0 = 'mypkg'
    var_1 = '__init__.py'
    var_2 = b'__import__("pkgutil").extend_path(__path__, __name__)'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'not_a_pkg'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'regularpkg'
    var_1 = '__init__.py'
    var_2 = b'print("hello")'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'namespacepkg'
    var_1 = 'module.py'
    var_2 = b''
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'emptynamespace'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'pkgresourcesns'
    var_1 = '__init__.py'
    var_2 = b"__import__('pkg_resources').declare_namespace(__name__)"
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'pkgresourcesns2'
    var_1 = '__init__.py'
    var_2 = b'__import__("pkg_resources").declare_namespace(__name__)'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'pkgutilns'
    var_1 = '__init__.py'
    var_2 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'pkgutilns2'
    var_1 = '__init__.py'
    var_2 = b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'mixedext'
    var_1 = 'data.txt'
    var_2 = b''
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'setupcfgdir'
    var_1 = 'setup.cfg'
    var_2 = b''
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'pyprojectdir'
    var_1 = 'pyproject.toml'
    var_2 = b''
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'haspyfile'
    var_1 = 'code.py'
    var_2 = b''
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'regularinit'
    var_1 = '__init__.py'
    var_2 = b'# just a comment'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)



# Parsed testcases at query #9
#--------------------------





def test_case_0():
    var_0 = 'foo'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = module_1._forced_separate(var_0, var_4)
    var_6 = bool(var_5 == ('foo', 'Matched forced_separate (foo) config value.'))
    assert var_6 is True


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


def test_case_0():
    var_0 = 'foo'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'bar'
    var_6 = module_1._forced_separate(var_5, var_4)
    assert var_6 is None


def test_case_0():
    var_0 = 'foo'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'foobar'
    var_6 = module_1._forced_separate(var_5, var_4)
    assert var_6 is None


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


def test_case_0():
    var_0 = 'f*o'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'foo'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('f*o', 'Matched forced_separate (f*o) config value.'))
    assert var_7 is True


def test_case_0():
    var_0 = 'f?o'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'foo'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('f?o', 'Matched forced_separate (f?o) config value.'))
    assert var_7 is True


def test_case_0():
    var_0 = []
    var_1 = 'forced_separate'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'foo'
    var_5 = module_1._forced_separate(var_4, var_3)
    assert var_5 is None


def test_case_0():
    var_0 = 'test[abc]'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'testa'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('test[abc]', 'Matched forced_separate (test[abc]) config value.'))
    assert var_7 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_src_path_is_module_true. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_is_namespace_package_false_when_filenames_exist. Retrieved 4/11 statements.


def test_case_0():
    var_0 = '__init__.py'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 2/12 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 1/14 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 2/12 statements.
# Partially parsed test_is_module_without_any_files. Retrieved 1/9 statements.
# Partially parsed test_is_module_case_sensitive_check. Retrieved 2/13 statements.


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
    var_1 = 'module.py'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_namespace_package_with_files_in_directory_but_no_init. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'some.py'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_is_module_with_py_extension. Retrieved 1/5 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 3/8 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 2/6 statements.
# Partially parsed test_is_module_false. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'some_module'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'some_extension'
    var_1 = [var_0]
    var_2 = '.so'
    var_3 = '.pyd'

def test_case_0():
    var_0 = 'some_package'
    var_1 = [var_0]
    var_2 = '__init__.py'

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = [var_0]



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_namespace_in_config_namespace_packages. Retrieved 10/15 statements.
# Partially parsed test_auto_identify_namespace_packages_true_and_is_namespace_package. Retrieved 9/15 statements.
# Partially parsed test_nested_module_and_namespace_in_config_namespace_packages. Retrieved 10/15 statements.
# Partially parsed test_nested_module_and_auto_identify_namespace_packages_true. Retrieved 9/15 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'some.namespace'
    var_3 = '.py'
    var_4 = '/some/path'
    var_5 = [var_4]
    var_6 = 'some.namespace.module'
    var_7 = None
    var_8 = 'some'
    var_9 = 'namespace'
    var_10 = (var_8, var_9)
    var_11 = module_1._src_path(var_6, var_1, var_7, var_10)
    var_12 = bool(var_11 is not None)
    assert var_12 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '.py'
    var_3 = '/some/path'
    var_4 = [var_3]
    var_5 = 'some.namespace.module'
    var_6 = None
    var_7 = 'some'
    var_8 = 'namespace'
    var_9 = (var_7, var_8)
    var_10 = module_1._src_path(var_5, var_1, var_6, var_9)
    var_11 = bool(var_10 is not None)
    assert var_11 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'some.namespace'
    var_3 = '.py'
    var_4 = '/some/path'
    var_5 = [var_4]
    var_6 = 'some.namespace.module'
    var_7 = None
    var_8 = 'some'
    var_9 = 'namespace'
    var_10 = (var_8, var_9)
    var_11 = module_1._src_path(var_6, var_1, var_7, var_10)
    var_12 = bool(var_11 is not None)
    assert var_12 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '.py'
    var_3 = '/some/path'
    var_4 = [var_3]
    var_5 = 'some.namespace.module'
    var_6 = None
    var_7 = 'some'
    var_8 = 'namespace'
    var_9 = (var_7, var_8)
    var_10 = module_1._src_path(var_5, var_1, var_6, var_9)
    var_11 = bool(var_10 is not None)
    assert var_11 is True



# Parsed testcases at query #16
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



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_namespace_package_without_init_and_no_files. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)



# Parsed testcases at query #18
#--------------------------





def test_case_0():
    var_0 = set()
    var_1 = False
    var_2 = []
    var_3 = set()
    var_4 = 'namespace_packages'
    var_5 = 'auto_identify_namespace_packages'
    var_6 = 'src_paths'
    var_7 = 'supported_extensions'
    var_8 = {var_4: var_0, var_5: var_1, var_6: var_2, var_7: var_3}
    var_9 = module_0.Config(**var_8)
    var_10 = 'nested'
    var_11 = [var_10]
    var_12 = 'root'
    var_13 = var_9.namespace_packages
    var_14 = var_12 in var_13
    var_15 = var_9.auto_identify_namespace_packages
    var_16 = var_15 and var_1
    var_17 = var_14 or var_16
    assert var_17 is False



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_namespace_not_in_config_and_auto_identify_false. Retrieved 7/10 statements.



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



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_namespace_package_without_init_and_no_files. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_predicate_at_line_26_true_for_module. Retrieved 9/19 statements.
# Partially parsed test_predicate_at_line_26_true_for_package. Retrieved 9/19 statements.
# Partially parsed test_predicate_at_line_26_true_for_src_path_is_module. Retrieved 9/19 statements.
# Partially parsed test_predicate_at_line_26_true_for_combination. Retrieved 10/20 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = '/src/root_module.py'
    var_7 = [var_6]
    var_8 = [var_0]
    var_9 = 'root_module'
    var_10 = lambda p: var_3
    var_11 = lambda s, r: var_3

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = '/src/root_module'
    var_7 = [var_6]
    var_8 = [var_0]
    var_9 = 'root_module'
    var_10 = lambda p: var_3
    var_11 = lambda s, r: var_3

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = '/src/root_module'
    var_7 = [var_6]
    var_8 = [var_0]
    var_9 = 'root_module'
    var_10 = lambda p: var_3
    var_11 = lambda p: var_3

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = '/src/root_module.py'
    var_7 = [var_6]
    var_8 = [var_0]
    var_9 = 'root_module'
    var_10 = True
    var_11 = lambda p: var_10
    var_12 = lambda s, r: var_10



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_namespace_package_without_init_and_no_files. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_namespace_in_config_namespace_packages. Retrieved 9/15 statements.
# Partially parsed test_auto_identify_namespace_packages_true_and_is_namespace_package. Retrieved 11/18 statements.
# Partially parsed test_nested_module_and_namespace_in_config_namespace_packages. Retrieved 9/15 statements.
# Partially parsed test_nested_module_and_auto_identify_and_is_namespace_package. Retrieved 11/18 statements.


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
    var_6 = '/src/some/namespace'
    var_7 = [var_6]
    var_8 = 'some.namespace.module'
    var_9 = '/src'
    var_10 = [var_9]
    var_11 = [var_2]
    var_12 = 'some'
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
    var_6 = '/src/a/b'
    var_7 = [var_6]
    var_8 = 'a.b.c'
    var_9 = '/src'
    var_10 = [var_9]
    var_11 = [var_2]
    var_12 = 'a'
    var_13 = (var_12,)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test__src_path_finds_module_in_src_paths. Retrieved 4/13 statements.
# Partially parsed test__src_path_returns_firstparty_for_module. Retrieved 4/15 statements.
# Partially parsed test__src_path_handles_nested_module_with_namespace. Retrieved 5/17 statements.
# Partially parsed test__src_path_identifies_src_path_as_module. Retrieved 4/15 statements.
# Partially parsed test__src_path_returns_none_for_no_match. Retrieved 4/14 statements.
# Partially parsed test__src_path_uses_custom_src_paths. Retrieved 5/18 statements.
# Partially parsed test__src_path_handles_auto_identify_namespace_packages. Retrieved 4/17 statements.


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
    var_2 = 'parent'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = 'parent.child'

def test_case_0():
    var_0 = '/src/mymodule'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = 'mymodule'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = 'unknown'

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
    var_4 = 'parent.child'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_is_module_with_py_extension. Retrieved 2/7 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 1/5 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'some_module'
    var_1 = [var_0]
    var_2 = '.py'

def test_case_0():
    var_0 = 'some_extension'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'some_package'
    var_1 = [var_0]
    var_2 = '__init__.py'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_known_pattern_predicate_false. Retrieved 5/21 statements.


def test_case_0():
    var_0 = 'section1'
    var_1 = 'section2'
    var_2 = [var_0, var_1]
    var_3 = 'some.module.name'
    var_4 = '.'
    var_5 = 0
    var_6 = -1



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 1/5 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 2/9 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 2/8 statements.
# Partially parsed test_is_module_none_exist. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'some_module'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'some_extension'
    var_1 = [var_0]
    var_2 = '.so'

def test_case_0():
    var_0 = 'some_package'
    var_1 = [var_0]
    var_2 = '__init__.py'

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = [var_0]



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_known_pattern_matches_configured_pattern. Retrieved 8/28 statements.
# Partially parsed test_known_pattern_no_match. Retrieved 3/15 statements.
# Partially parsed test_known_pattern_placement_not_in_sections. Retrieved 5/17 statements.
# Partially parsed test_known_pattern_matches_longest_module_prefix. Retrieved 8/28 statements.
# Partially parsed test_known_pattern_matches_first_pattern_in_order. Retrieved 7/27 statements.
# Partially parsed test_known_pattern_empty_name. Retrieved 3/15 statements.


def test_case_0():
    var_0 = 'foo.bar'
    var_1 = 'SECTION_A'
    var_2 = 'baz'
    var_3 = 'SECTION_B'
    var_4 = {var_1, var_3}
    var_5 = 'foo.bar.module'
    var_6 = 'Matched configured known pattern '
    var_7 = 0

def test_case_0():
    var_0 = 'SECTION_A'
    var_1 = {var_0}
    var_2 = 'unknown.module'

def test_case_0():
    var_0 = 'SECTION_C'
    var_1 = 'SECTION_A'
    var_2 = 'SECTION_B'
    var_3 = {var_1, var_2}
    var_4 = 'any.module'

def test_case_0():
    var_0 = 'foo'
    var_1 = 'SECTION_A'
    var_2 = 'foo.bar'
    var_3 = 'SECTION_B'
    var_4 = {var_1, var_3}
    var_5 = 'foo.bar.baz'
    var_6 = 'Matched configured known pattern '
    var_7 = 0

def test_case_0():
    var_0 = 'foo.bar'
    var_1 = 'SECTION_A'
    var_2 = 'SECTION_B'
    var_3 = {var_1, var_2}
    var_4 = 'foo.bar.module'
    var_5 = 'Matched configured known pattern '
    var_6 = 0

def test_case_0():
    var_0 = 'SECTION_A'
    var_1 = {var_0}
    var_2 = ''



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_src_path_is_module_true_for_matching_dir. Retrieved 1/5 statements.
# Partially parsed test_src_path_is_module_false_for_non_matching_name. Retrieved 2/6 statements.
# Partially parsed test_src_path_is_module_false_for_file. Retrieved 1/5 statements.
# Partially parsed test_src_path_is_module_false_for_case_mismatch. Retrieved 2/6 statements.
# Partially parsed test_src_path_is_module_false_for_nonexistent_path. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]
    var_2 = 'different_module'

def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'Test_Module'
    var_1 = [var_0]
    var_2 = 'test_module'

def test_case_0():
    var_0 = 'nonexistent_module'
    var_1 = [var_0]



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_namespace_in_config_namespace_packages. Retrieved 10/16 statements.


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
    var_10 = 'namespace'
    var_11 = (var_9, var_10)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_namespace_package_without_init_and_no_files. Retrieved 4/11 statements.
# Partially parsed test_namespace_package_without_init_and_only_dot_files. Retrieved 5/14 statements.
# Partially parsed test_namespace_package_without_init_and_only_non_source_files. Retrieved 5/14 statements.
# Partially parsed test_namespace_package_without_init_and_only_setup_cfg. Retrieved 5/14 statements.
# Partially parsed test_namespace_package_without_init_and_only_pyproject_toml. Retrieved 5/14 statements.
# Partially parsed test_namespace_package_without_init_and_source_file. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '.gitignore'
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'README.md'
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



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_src_path_finds_module_in_src_paths. Retrieved 7/11 statements.
# Partially parsed test_src_path_returns_none_for_missing_module. Retrieved 7/11 statements.
# Partially parsed test_src_path_handles_nested_module_in_namespace_package. Retrieved 8/12 statements.
# Partially parsed test_src_path_handles_auto_identified_namespace_package. Retrieved 7/11 statements.
# Partially parsed test_src_path_uses_provided_src_paths_parameter. Retrieved 9/12 statements.
# Partially parsed test_src_path_handles_src_path_is_module_case. Retrieved 7/11 statements.


def test_case_0():
    var_0 = '/test/path'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'mymodule'

def test_case_0():
    var_0 = '/test/path'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'missingmodule'

def test_case_0():
    var_0 = '/test/path'
    var_1 = [var_0]
    var_2 = 'namespace'
    var_3 = {var_2}
    var_4 = False
    var_5 = 'py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)
    var_8 = 'namespace.nested'

def test_case_0():
    var_0 = '/test/path'
    var_1 = [var_0]
    var_2 = set()
    var_3 = True
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'auto_ns.nested'


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
    var_12 = '/custom/path'
    var_13 = [var_12]
    var_14 = 'mymodule'

def test_case_0():
    var_0 = '/test/path'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'path'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_is_namespace_package_with_valid_namespace_package. Retrieved 6/16 statements.
# Partially parsed test_is_namespace_package_with_non_package_path. Retrieved 4/10 statements.
# Partially parsed test_is_namespace_package_with_regular_package. Retrieved 6/15 statements.
# Partially parsed test_is_namespace_package_with_missing_init_and_py_files. Retrieved 4/11 statements.
# Partially parsed test_is_namespace_package_with_missing_init_but_has_py_file. Retrieved 6/15 statements.
# Partially parsed test_is_namespace_package_with_missing_init_but_has_setup_cfg. Retrieved 6/15 statements.
# Partially parsed test_is_namespace_package_with_missing_init_but_has_pyproject_toml. Retrieved 6/15 statements.
# Partially parsed test_is_namespace_package_with_pkg_resources_namespace. Retrieved 6/15 statements.
# Partially parsed test_is_namespace_package_with_double_quotes_pkg_resources. Retrieved 6/15 statements.
# Partially parsed test_is_namespace_package_with_single_quotes_pkgutil. Retrieved 6/15 statements.
# Partially parsed test_is_namespace_package_with_double_quotes_pkgutil. Retrieved 6/15 statements.
# Partially parsed test_is_namespace_package_with_init_but_no_namespace_marker. Retrieved 6/15 statements.
# Partially parsed test_is_namespace_package_with_src_extensions_other_than_py. Retrieved 4/11 statements.
# Partially parsed test_is_namespace_package_with_missing_init_and_other_extension_file. Retrieved 6/15 statements.


def test_case_0():
    var_0 = 'mypkg'
    var_1 = '__init__.py'
    var_2 = b'__import__("pkgutil").extend_path(__path__, __name__)'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'mypkg'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'mypkg'
    var_1 = '__init__.py'
    var_2 = b'print("hello")'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'mypkg'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'mypkg'
    var_1 = 'module.py'
    var_2 = b''
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'mypkg'
    var_1 = 'setup.cfg'
    var_2 = b''
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'mypkg'
    var_1 = 'pyproject.toml'
    var_2 = b''
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'mypkg'
    var_1 = '__init__.py'
    var_2 = b"__import__('pkg_resources').declare_namespace(__name__)"
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'mypkg'
    var_1 = '__init__.py'
    var_2 = b'__import__("pkg_resources").declare_namespace(__name__)'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'mypkg'
    var_1 = '__init__.py'
    var_2 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'mypkg'
    var_1 = '__init__.py'
    var_2 = b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'mypkg'
    var_1 = '__init__.py'
    var_2 = b''
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'mypkg'
    var_1 = 'txt'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'mypkg'
    var_1 = 'data.txt'
    var_2 = b''
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_is_module_with_py_suffix. Retrieved 1/4 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 1/5 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 1/4 statements.
# Partially parsed test_is_module_with_all_false. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'some_module'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'some_extension'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'some_package'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = [var_0]



# Parsed testcases at query #35
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



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 2/12 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 1/15 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 2/12 statements.
# Partially parsed test_is_module_no_files. Retrieved 1/8 statements.
# Partially parsed test_is_module_case_sensitive_py. Retrieved 2/12 statements.
# Partially parsed test_is_module_case_sensitive_init. Retrieved 2/12 statements.


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
    var_1 = '.PY'

def test_case_0():
    var_0 = 'module'
    var_1 = '__INIT__.py'



# Parsed testcases at query #2
#--------------------------

# Failed to parse test_is_module_with_py_file.
# Failed to parse test_is_module_with_extension_suffix.
# Failed to parse test_is_module_with_init_py.
# Failed to parse test_is_module_no_match.
# Partially parsed test_is_module_py_false_extension_true. Retrieved 3/10 statements.
# Partially parsed test_is_module_py_true_extension_false. Retrieved 3/10 statements.
# Failed to parse test_is_module_init_py_only.


def test_case_0():
    var_0 = '.py'
    var_1 = 'test.py'
    var_2 = 'test.so'

def test_case_0():
    var_0 = '.py'
    var_1 = 'test.py'
    var_2 = 'test.so'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 3/13 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 2/10 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 3/12 statements.
# Partially parsed test_is_module_no_match. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'some_module'
    var_1 = [var_0]
    var_2 = 'some_module.py'
    var_3 = [var_2]
    var_4 = '.py'

def test_case_0():
    var_0 = 'some_module'
    var_1 = [var_0]
    var_2 = 'some_module.so'
    var_3 = [var_2]

def test_case_0():
    var_0 = 'some_module'
    var_1 = [var_0]
    var_2 = 'some_module/__init__.py'
    var_3 = [var_2]
    var_4 = '__init__.py'

def test_case_0():
    var_0 = 'some_module'
    var_1 = [var_0]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test__src_path_with_exact_module_match. Retrieved 6/11 statements.
# Partially parsed test__src_path_with_nested_module_in_namespace. Retrieved 7/12 statements.
# Partially parsed test__src_path_with_auto_identified_namespace. Retrieved 6/11 statements.
# Partially parsed test__src_path_with_no_match. Retrieved 6/11 statements.
# Partially parsed test__src_path_with_custom_src_paths. Retrieved 5/13 statements.
# Partially parsed test__src_path_with_prefix. Retrieved 8/13 statements.
# Partially parsed test__src_path_with_src_path_is_module. Retrieved 6/11 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/test/path'
    var_3 = [var_2]
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = 'mymodule'
    var_7 = module_1._src_path(var_6, var_1)


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/test/path'
    var_3 = [var_2]
    var_4 = 'mypackage'
    var_5 = 'py'
    var_6 = [var_5]
    var_7 = 'mypackage.submodule'
    var_8 = module_1._src_path(var_7, var_1)


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/test/path'
    var_3 = [var_2]
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = 'namespace.sub'
    var_7 = module_1._src_path(var_6, var_1)


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/test/path'
    var_3 = [var_2]
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = 'unknown'
    var_7 = module_1._src_path(var_6, var_1)
    assert var_7 is None


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/custom/path'
    var_3 = [var_2]
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = 'module'
    var_7 = [var_2]


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/test/path'
    var_3 = [var_2]
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = 'sub'
    var_7 = 'base'
    var_8 = (var_7,)
    var_9 = module_1._src_path(var_6, var_1, prefix=var_8)


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/test/path'
    var_3 = [var_2]
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = 'path'
    var_7 = module_1._src_path(var_6, var_1)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 4/12 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/fake/path'
    var_3 = [var_2]
    var_4 = '.py'
    var_5 = 'mymodule'
    var_6 = [var_2]
    var_7 = 'Found in one of the configured src_paths:'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_is_namespace_package_with_non_existent_path. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_file_path. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_directory_no_init_and_no_files. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_directory_no_init_but_py_file. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_directory_no_init_but_setup_cfg. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_directory_no_init_but_pyproject_toml. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_init_but_no_namespace_markers. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_init_and_pkg_resources_single_quote. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_init_and_pkg_resources_double_quote. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_init_and_pkgutil_single_quote. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_init_and_pkgutil_double_quote. Retrieved 4/6 statements.


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

def test_case_0():
    var_0 = '/dir/with/pkg_resources_single'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/dir/with/pkg_resources_double'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/dir/with/pkgutil_single'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/dir/with/pkgutil_double'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_namespace_in_config_namespace_packages. Retrieved 9/15 statements.
# Partially parsed test_auto_identify_namespace_packages_true_and_is_namespace_package. Retrieved 10/21 statements.
# Partially parsed test_nested_module_true_and_namespace_in_config_namespace_packages. Retrieved 9/15 statements.
# Partially parsed test_nested_module_true_and_auto_identify_true_and_is_namespace_package. Retrieved 10/21 statements.


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
    var_6 = 'test.namespace.module'
    var_7 = '/src'
    var_8 = [var_7]
    var_9 = [var_2]
    var_10 = 'test'
    var_11 = (var_10,)

def test_case_0():
    var_0 = 'existing.namespace'
    var_1 = {var_0}
    var_2 = False
    var_3 = '/src'
    var_4 = [var_3]
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = 'existing.namespace.submodule'
    var_8 = [var_3]
    var_9 = 'existing'
    var_10 = (var_9,)

def test_case_0():
    var_0 = set()
    var_1 = True
    var_2 = '/src'
    var_3 = [var_2]
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = 'auto.namespace.inner'
    var_7 = '/src'
    var_8 = [var_7]
    var_9 = [var_2]
    var_10 = 'auto'
    var_11 = (var_10,)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_known_pattern_matches_configured_pattern. Retrieved 6/12 statements.
# Partially parsed test_known_pattern_no_match. Retrieved 5/11 statements.
# Partially parsed test_known_pattern_placement_not_in_sections. Retrieved 6/12 statements.
# Partially parsed test_known_pattern_matches_longest_module_prefix. Retrieved 7/15 statements.
# Partially parsed test_known_pattern_matches_single_part_name. Retrieved 5/11 statements.
# Partially parsed test_known_pattern_empty_name. Retrieved 5/11 statements.
# Partially parsed test_known_pattern_no_known_patterns. Retrieved 4/7 statements.
# Partially parsed test_known_pattern_no_sections. Retrieved 5/11 statements.



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


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'section1'
    var_3 = '^other\\.module$'
    var_4 = 'myapp.utils'
    var_5 = module_1._known_pattern(var_4, var_1)
    assert var_5 is None


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'section1'
    var_3 = '^myapp$'
    var_4 = 'section2'
    var_5 = 'myapp.utils'
    var_6 = module_1._known_pattern(var_5, var_1)
    assert var_6 is None


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


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'section1'
    var_3 = '^myapp$'
    var_4 = 'myapp'
    var_5 = module_1._known_pattern(var_4, var_1)
    var_6 = bool(var_5 == ('section1', "Matched configured known pattern re.compile('^myapp$')"))
    assert var_6 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'section1'
    var_3 = '^$'
    var_4 = ''
    var_5 = module_1._known_pattern(var_4, var_1)
    var_6 = bool(var_5 == ('section1', "Matched configured known pattern re.compile('^$')"))
    assert var_6 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'section1'
    var_3 = 'myapp.utils'
    var_4 = module_1._known_pattern(var_3, var_1)
    assert var_4 is None


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '^myapp$'
    var_3 = 'section1'
    var_4 = 'myapp'
    var_5 = module_1._known_pattern(var_4, var_1)
    assert var_5 is None



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_namespace_package_without_init_and_no_files. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'mypkg'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test__src_path_finds_module_in_src_paths. Retrieved 4/15 statements.
# Partially parsed test__src_path_returns_none_for_missing_module. Retrieved 4/15 statements.
# Partially parsed test__src_path_handles_nested_module_in_namespace_package. Retrieved 5/17 statements.
# Partially parsed test__src_path_handles_src_path_is_module. Retrieved 4/15 statements.
# Partially parsed test__src_path_with_auto_identify_namespace_packages. Retrieved 4/16 statements.


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
    var_4 = 'missing'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'namespace'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = 'namespace.nested'

def test_case_0():
    var_0 = '/src/mymodule'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = 'mymodule'

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = 'namespace.nested'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_known_pattern_matches. Retrieved 3/11 statements.
# Partially parsed test_known_pattern_no_match. Retrieved 3/11 statements.
# Partially parsed test_known_pattern_section_not_in_sections. Retrieved 4/12 statements.
# Partially parsed test_known_pattern_matches_longest_first. Retrieved 5/15 statements.
# Partially parsed test_known_pattern_matches_first_pattern. Retrieved 4/14 statements.
# Partially parsed test_known_pattern_no_parts. Retrieved 3/11 statements.
# Partially parsed test_known_pattern_exact_match. Retrieved 3/11 statements.


def test_case_0():
    var_0 = '^test\\.module'
    var_1 = 'SECTION_A'
    var_2 = 'test.module.sub'

def test_case_0():
    var_0 = '^other\\.module'
    var_1 = 'SECTION_A'
    var_2 = 'test.module.sub'

def test_case_0():
    var_0 = '^test\\.module'
    var_1 = 'SECTION_A'
    var_2 = 'SECTION_B'
    var_3 = 'test.module.sub'

def test_case_0():
    var_0 = '^test'
    var_1 = 'SECTION_A'
    var_2 = '^test\\.module'
    var_3 = 'SECTION_B'
    var_4 = 'test.module.sub'

def test_case_0():
    var_0 = '^test\\.module'
    var_1 = 'SECTION_A'
    var_2 = 'SECTION_B'
    var_3 = 'test.module.sub'

def test_case_0():
    var_0 = '^test'
    var_1 = 'SECTION_A'
    var_2 = ''

def test_case_0():
    var_0 = '^test\\.module\\.sub$'
    var_1 = 'SECTION_A'
    var_2 = 'test.module.sub'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_namespace_package_without_init_and_no_files. Retrieved 4/11 statements.
# Partially parsed test_namespace_package_without_init_and_only_non_source_files. Retrieved 6/17 statements.
# Partially parsed test_namespace_package_without_init_and_setup_cfg. Retrieved 5/14 statements.
# Partially parsed test_namespace_package_without_init_and_pyproject_toml. Retrieved 5/14 statements.
# Partially parsed test_namespace_package_without_init_and_source_file. Retrieved 5/14 statements.
# Partially parsed test_namespace_package_with_init_and_namespace_declaration_single_quotes. Retrieved 6/15 statements.
# Partially parsed test_namespace_package_with_init_and_namespace_declaration_double_quotes. Retrieved 6/15 statements.
# Partially parsed test_namespace_package_with_init_and_extend_path_single_quotes. Retrieved 6/15 statements.
# Partially parsed test_namespace_package_with_init_and_extend_path_double_quotes. Retrieved 6/15 statements.
# Partially parsed test_namespace_package_with_init_and_no_namespace_declaration. Retrieved 6/15 statements.
# Partially parsed test_non_package_directory. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'README.txt'
    var_2 = 'data.json'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

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
    var_2 = "print('hello')"
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'some_file.txt'
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test__src_path_finds_module_in_src_paths. Retrieved 7/11 statements.
# Partially parsed test__src_path_handles_nested_module_in_namespace_package. Retrieved 8/12 statements.
# Partially parsed test__src_path_returns_none_when_not_found. Retrieved 7/11 statements.
# Partially parsed test__src_path_uses_provided_src_paths. Retrieved 9/12 statements.
# Partially parsed test__src_path_handles_root_module_matching_src_path_name. Retrieved 7/11 statements.
# Partially parsed test__src_path_auto_identifies_namespace_packages. Retrieved 7/11 statements.


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

def test_case_0():
    var_0 = '/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'unknown'


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
    var_3 = True
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'namespace.sub'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 2/11 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 1/13 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 2/12 statements.
# Partially parsed test_is_module_no_match. Retrieved 1/8 statements.
# Partially parsed test_is_module_case_sensitive_py. Retrieved 2/11 statements.
# Partially parsed test_is_module_case_sensitive_init. Retrieved 2/12 statements.


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
    var_0 = 'module'
    var_1 = '.PY'

def test_case_0():
    var_0 = 'module'
    var_1 = '__INIT__.py'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_namespace_package_without_init_and_no_files. Retrieved 5/11 statements.
# Partially parsed test_namespace_package_without_init_and_no_src_files. Retrieved 7/15 statements.
# Partially parsed test_namespace_package_without_init_and_no_src_files_but_setup_cfg. Retrieved 7/15 statements.
# Partially parsed test_namespace_package_without_init_and_no_src_files_but_pyproject_toml. Retrieved 7/15 statements.
# Partially parsed test_namespace_package_without_init_and_src_file_present. Retrieved 7/15 statements.


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
    var_3 = 'README.txt'
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = False
    var_1 = lambda : var_0
    var_2 = '.cfg'
    var_3 = 'setup.cfg'
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = False
    var_1 = lambda : var_0
    var_2 = '.toml'
    var_3 = 'pyproject.toml'
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = False
    var_1 = lambda : var_0
    var_2 = '.py'
    var_3 = 'module.py'
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 4/6 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/fake/path'
    var_3 = [var_2]
    var_4 = 'mymodule'
    var_5 = module_1._src_path(var_4, var_1)
    var_6 = bool(var_5 is not None)
    assert var_6 is True
    var_7 = var_5[0]
    var_8 = 'Found in one of the configured src_paths:'
    var_9 = bool('Found in one of the configured src_paths:' in var_5[1])
    assert var_9 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_is_namespace_package_with_valid_namespace_package. Retrieved 7/14 statements.
# Partially parsed test_is_namespace_package_with_non_package_path. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_regular_package. Retrieved 7/14 statements.
# Partially parsed test_is_namespace_package_with_namespace_package_using_pkgutil. Retrieved 7/14 statements.
# Partially parsed test_is_namespace_package_without_init_but_with_py_files. Retrieved 7/14 statements.
# Partially parsed test_is_namespace_package_without_init_but_with_setup_cfg. Retrieved 7/14 statements.
# Partially parsed test_is_namespace_package_without_init_and_no_files. Retrieved 5/9 statements.
# Partially parsed test_is_namespace_package_with_double_quotes_in_declare_namespace. Retrieved 7/14 statements.
# Partially parsed test_is_namespace_package_with_double_quotes_in_pkgutil. Retrieved 7/14 statements.
# Partially parsed test_is_namespace_package_with_other_src_extensions. Retrieved 7/14 statements.


def test_case_0():
    var_0 = '/tmp/test_package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = b"__import__('pkg_resources').declare_namespace(__name__)"
    var_5 = 'py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/tmp/nonexistent'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/tmp/regular_package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = b''
    var_5 = 'py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/tmp/namespace_pkgutil'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_5 = 'py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/tmp/package_with_py'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'module.py'
    var_4 = b''
    var_5 = 'py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/tmp/package_with_setup_cfg'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'setup.cfg'
    var_4 = b''
    var_5 = 'py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/tmp/empty_package'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = '/tmp/double_quote_package'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = b'__import__("pkg_resources").declare_namespace(__name__)'
    var_5 = 'py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/tmp/double_quote_pkgutil'
    var_1 = [var_0]
    var_2 = True
    var_3 = '__init__.py'
    var_4 = b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    var_5 = 'py'
    var_6 = [var_5]
    var_7 = frozenset(var_6)

def test_case_0():
    var_0 = '/tmp/package_with_other_ext'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'module.txt'
    var_4 = b''
    var_5 = 'txt'
    var_6 = [var_5]
    var_7 = frozenset(var_6)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_is_namespace_package_with_non_existent_path. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_file_path. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_directory_no_init_and_no_files. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_directory_no_init_but_has_py_file. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_directory_no_init_but_has_setup_cfg. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_directory_no_init_but_has_pyproject_toml. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_init_but_no_namespace_declaration. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_init_and_pkg_resources_single_quote. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_init_and_pkg_resources_double_quote. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_init_and_pkgutil_single_quote. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_init_and_pkgutil_double_quote. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_init_and_namespace_declaration_at_end_of_file. Retrieved 4/6 statements.


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

def test_case_0():
    var_0 = '/dir/with/pkg_resources_single'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/dir/with/pkg_resources_double'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/dir/with/pkgutil_single'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/dir/with/pkgutil_double'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/dir/with/long_init'
    var_1 = [var_0]
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_predicate_at_line_26_evaluates_to_true. Retrieved 15/26 statements.


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
    var_9 = []
    var_10 = '.'
    var_11 = (var_7,)
    var_12 = var_8 + var_11
    var_13 = True
    var_14 = lambda p: var_13
    var_15 = lambda p: var_3
    var_16 = lambda s, r: var_3



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_known_pattern_returns_tuple_when_placement_in_sections_and_pattern_matches. Retrieved 4/16 statements.


def test_case_0():
    var_0 = 'test_section'
    var_1 = True
    var_2 = 'some.module.name'
    var_3 = 'Matched configured known pattern '



# Parsed testcases at query #21
#--------------------------





def test_case_0():
    var_0 = 'exact'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = module_1._forced_separate(var_0, var_4)
    var_6 = bool(var_5 == ('exact', 'Matched forced_separate (exact) config value.'))
    assert var_6 is True


def test_case_0():
    var_0 = 'dir/*'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'dir/file.txt'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('dir/*', 'Matched forced_separate (dir/*) config value.'))
    assert var_7 is True


def test_case_0():
    var_0 = 'prefix'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'prefix_suffix'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('prefix', 'Matched forced_separate (prefix) config value.'))
    assert var_7 is True


def test_case_0():
    var_0 = '.hidden'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = module_1._forced_separate(var_0, var_4)
    var_6 = bool(var_5 == ('.hidden', 'Matched forced_separate (.hidden) config value.'))
    assert var_6 is True


def test_case_0():
    var_0 = '.dir/*'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '.dir/file.txt'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('.dir/*', 'Matched forced_separate (.dir/*) config value.'))
    assert var_7 is True


def test_case_0():
    var_0 = 'other'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'nomatch'
    var_6 = module_1._forced_separate(var_5, var_4)
    assert var_6 is None


def test_case_0():
    var_0 = []
    var_1 = 'forced_separate'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'any'
    var_5 = module_1._forced_separate(var_4, var_3)
    assert var_5 is None


def test_case_0():
    var_0 = 'first'
    var_1 = 'second'
    var_2 = [var_0, var_1]
    var_3 = 'forced_separate'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = 'first_match'
    var_7 = module_1._forced_separate(var_6, var_5)
    var_8 = bool(var_7 == ('first', 'Matched forced_separate (first) config value.'))
    assert var_8 is True


def test_case_0():
    var_0 = 'first'
    var_1 = 'second'
    var_2 = [var_0, var_1]
    var_3 = 'forced_separate'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = 'second_match'
    var_7 = module_1._forced_separate(var_6, var_5)
    var_8 = bool(var_7 == ('second', 'Matched forced_separate (second) config value.'))
    assert var_8 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_namespace_package_with_pkgutil_extend_path_single_quotes. Retrieved 5/13 statements.
# Partially parsed test_namespace_package_with_pkgutil_extend_path_double_quotes. Retrieved 5/13 statements.
# Partially parsed test_namespace_package_with_pkg_resources_declare_namespace_single_quotes. Retrieved 5/13 statements.
# Partially parsed test_namespace_package_with_pkg_resources_declare_namespace_double_quotes. Retrieved 5/13 statements.


def test_case_0():
    var_0 = '__init__.py'
    var_1 = "__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '__init__.py'
    var_1 = '__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
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
    var_1 = '__import__("pkg_resources").declare_namespace(__name__)'
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_namespace_package_with_no_init_and_no_files. Retrieved 4/11 statements.
# Partially parsed test_namespace_package_with_no_init_and_no_source_files_but_setup_cfg. Retrieved 5/14 statements.
# Partially parsed test_namespace_package_with_no_init_and_no_source_files_but_pyproject_toml. Retrieved 5/14 statements.
# Partially parsed test_namespace_package_with_no_init_and_source_file. Retrieved 5/14 statements.


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



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_src_path_is_module_true_for_valid_dir. Retrieved 1/5 statements.
# Partially parsed test_src_path_is_module_false_for_wrong_name. Retrieved 2/6 statements.
# Partially parsed test_src_path_is_module_false_for_file. Retrieved 1/5 statements.
# Partially parsed test_src_path_is_module_false_for_case_mismatch_on_case_sensitive_fs. Retrieved 2/6 statements.
# Partially parsed test_src_path_is_module_true_for_case_match_on_case_sensitive_fs. Retrieved 1/5 statements.


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

def test_case_0():
    var_0 = 'test_module'
    var_1 = [var_0]



# Parsed testcases at query #25
#--------------------------

# Partially parsed test__src_path_with_exact_module_match. Retrieved 5/9 statements.
# Partially parsed test__src_path_with_nested_module_in_namespace_package. Retrieved 6/12 statements.
# Partially parsed test__src_path_with_auto_identified_namespace_package. Retrieved 7/13 statements.
# Partially parsed test__src_path_with_no_match. Retrieved 5/9 statements.
# Partially parsed test__src_path_with_src_path_is_module_match. Retrieved 5/9 statements.
# Partially parsed test__src_path_with_custom_src_paths. Retrieved 7/10 statements.
# Partially parsed test__src_path_with_prefix. Retrieved 7/13 statements.


def test_case_0():
    var_0 = '/test/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = frozenset()
    var_5 = 'mymodule'

def test_case_0():
    var_0 = '/test/src'
    var_1 = [var_0]
    var_2 = 'mypackage'
    var_3 = {var_2}
    var_4 = False
    var_5 = frozenset()
    var_6 = 'mypackage.nested'
    var_7 = [var_0]

def test_case_0():
    var_0 = '/test/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = True
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'namespace.nested'
    var_8 = [var_0]

def test_case_0():
    var_0 = '/test/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = frozenset()
    var_5 = 'unknown'

def test_case_0():
    var_0 = '/test/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = frozenset()
    var_5 = 'src'


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
    var_11 = '/custom/path'
    var_12 = [var_11]

def test_case_0():
    var_0 = '/test/src'
    var_1 = [var_0]
    var_2 = set()
    var_3 = False
    var_4 = frozenset()
    var_5 = 'mymodule'
    var_6 = [var_0]
    var_7 = 'pre'
    var_8 = (var_7,)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_src_path_is_module_true_for_valid_dir. Retrieved 1/5 statements.
# Partially parsed test_src_path_is_module_false_for_wrong_name. Retrieved 2/6 statements.
# Partially parsed test_src_path_is_module_false_for_file. Retrieved 1/5 statements.
# Partially parsed test_src_path_is_module_false_for_nonexistent. Retrieved 1/3 statements.
# Partially parsed test_src_path_is_module_false_for_case_mismatch. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'valid_module'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'some_dir'
    var_1 = [var_0]
    var_2 = 'different_name'

def test_case_0():
    var_0 = 'file.txt'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'Module'
    var_1 = [var_0]
    var_2 = 'module'



# Parsed testcases at query #27
#--------------------------





def test_case_0():
    var_0 = 'exact'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = module_1._forced_separate(var_0, var_4)
    var_6 = bool(var_5 == ('exact', 'Matched forced_separate (exact) config value.'))
    assert var_6 is True


def test_case_0():
    var_0 = 'prefix*'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'prefix_suffix'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('prefix*', 'Matched forced_separate (prefix*) config value.'))
    assert var_7 is True


def test_case_0():
    var_0 = 'hidden'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '.hidden'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('hidden', 'Matched forced_separate (hidden) config value.'))
    assert var_7 is True


def test_case_0():
    var_0 = 'pre*'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = '.pre_suffix'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('pre*', 'Matched forced_separate (pre*) config value.'))
    assert var_7 is True


def test_case_0():
    var_0 = 'other'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'nomatch'
    var_6 = module_1._forced_separate(var_5, var_4)
    assert var_6 is None


def test_case_0():
    var_0 = []
    var_1 = 'forced_separate'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
    var_4 = 'any'
    var_5 = module_1._forced_separate(var_4, var_3)
    assert var_5 is None


def test_case_0():
    var_0 = 'first'
    var_1 = 'second'
    var_2 = [var_0, var_1]
    var_3 = 'forced_separate'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = module_1._forced_separate(var_0, var_5)
    var_7 = bool(var_6 == ('first', 'Matched forced_separate (first) config value.'))
    assert var_7 is True


def test_case_0():
    var_0 = 'first'
    var_1 = 'second'
    var_2 = [var_0, var_1]
    var_3 = 'forced_separate'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = module_1._forced_separate(var_1, var_5)
    var_7 = bool(var_6 == ('second', 'Matched forced_separate (second) config value.'))
    assert var_7 is True


def test_case_0():
    var_0 = 'partial'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'partial_extra'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('partial', 'Matched forced_separate (partial) config value.'))
    assert var_7 is True


def test_case_0():
    var_0 = 'exact*'
    var_1 = [var_0]
    var_2 = 'forced_separate'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = 'exact'
    var_6 = module_1._forced_separate(var_5, var_4)
    var_7 = bool(var_6 == ('exact*', 'Matched forced_separate (exact*) config value.'))
    assert var_7 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 1/6 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 5/12 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 1/6 statements.
# Partially parsed test_is_module_no_match. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'some_module'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'some_module'
    var_1 = [var_0]
    var_2 = '.so'
    var_3 = '.pyd'
    var_4 = [var_2, var_3]
    var_5 = '.so'

def test_case_0():
    var_0 = 'some_module'
    var_1 = [var_0]

def test_case_0():
    var_0 = 'some_module'
    var_1 = [var_0]
    var_2 = '.so'
    var_3 = '.pyd'
    var_4 = [var_2, var_3]



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_namespace_not_in_config_and_auto_identify_false. Retrieved 7/10 statements.



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



