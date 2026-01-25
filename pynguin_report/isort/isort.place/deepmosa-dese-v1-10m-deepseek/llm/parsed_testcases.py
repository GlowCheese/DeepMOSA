####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_src_path_with_valid_module. Retrieved 4/8 statements.
# Partially parsed test_src_path_with_nested_module. Retrieved 5/9 statements.
# Partially parsed test_src_path_with_namespace_package. Retrieved 5/9 statements.
# Partially parsed test_src_path_with_invalid_module. Retrieved 4/8 statements.
# Partially parsed test_src_path_with_src_path_is_module. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'src'
    var_1 = set()
    var_2 = False
    var_3 = 'module_name'

def test_case_0():
    var_0 = 'src'
    var_1 = 'namespace'
    var_2 = {var_1}
    var_3 = False
    var_4 = 'namespace.module_name'

def test_case_0():
    var_0 = 'src'
    var_1 = 'namespace'
    var_2 = {var_1}
    var_3 = True
    var_4 = 'namespace.module_name'

def test_case_0():
    var_0 = 'src'
    var_1 = set()
    var_2 = False
    var_3 = 'invalid_module'

def test_case_0():
    var_0 = 'src'
    var_1 = set()
    var_2 = False



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_auto_identify_namespace_packages. Retrieved 7/21 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test.namespace'
    var_1 = {var_0}
    var_2 = False
    var_3 = []
    var_4 = []
    var_5 = 'test.namespace.module'
    var_6 = module_0.Config()
    var_7 = module_1._src_path(var_5, var_6)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = set()
    var_1 = True
    var_2 = []
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = 'test.namespace.module'
    var_6 = module_0.Config()
    var_7 = module_1._src_path(var_5, var_6)



# Parsed testcases at query #3
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test.namespace'
    var_1 = {var_0}
    var_2 = False
    var_3 = module_0.Config()
    var_4 = 'test.namespace.module'
    var_5 = module_1._src_path(var_4, var_3)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = set()
    var_1 = True
    var_2 = module_0.Config()
    var_3 = 'test.namespace.module'
    var_4 = module_1._src_path(var_3, var_2)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_is_namespace_package_with_valid_namespace. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_invalid_namespace. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_non_empty_directory. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_init_file_containing_namespace_declaration. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_init_file_missing_namespace_declaration. Retrieved 4/6 statements.


def test_case_0():
    var_0 = '/path/to/namespace'
    var_1 = 'py'
    var_2 = {var_1}
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = '/path/to/not_namespace'
    var_1 = 'py'
    var_2 = {var_1}
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = '/path/to/non_empty_directory'
    var_1 = 'py'
    var_2 = {var_1}
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = '/path/to/namespace_with_init'
    var_1 = 'py'
    var_2 = {var_1}
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = '/path/to/namespace_with_init_missing_declaration'
    var_1 = 'py'
    var_2 = {var_1}
    var_3 = frozenset(var_2)



# Parsed testcases at query #5
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test*'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'test_file.txt'
    var_4 = module_1._forced_separate(var_3, var_2)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'test_file.txt'
    var_4 = module_1._forced_separate(var_3, var_2)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test*'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = '.test_file.txt'
    var_4 = module_1._forced_separate(var_3, var_2)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test*'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'example_file.txt'
    var_4 = module_1._forced_separate(var_3, var_2)
    assert var_4 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test*'
    var_1 = 'example*'
    var_2 = [var_0, var_1]
    var_3 = module_0.Config()
    var_4 = 'test_file.txt'
    var_5 = module_1._forced_separate(var_4, var_3)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test*'
    var_1 = 'example*'
    var_2 = [var_0, var_1]
    var_3 = module_0.Config()
    var_4 = 'example_file.txt'
    var_5 = module_1._forced_separate(var_4, var_3)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test*'
    var_1 = 'example*'
    var_2 = [var_0, var_1]
    var_3 = module_0.Config()
    var_4 = 'other_file.txt'
    var_5 = module_1._forced_separate(var_4, var_3)
    assert var_5 is None



# Parsed testcases at query #6
#--------------------------

# Partially parsed test__is_module_with_py_file. Retrieved 2/7 statements.
# Partially parsed test__is_module_with_extension_suffix. Retrieved 1/6 statements.
# Partially parsed test__is_module_with_init_py. Retrieved 2/7 statements.
# Partially parsed test__is_module_with_no_valid_files. Retrieved 1/3 statements.


def test_case_0():
    var_0 = '/path/to/module'
    var_1 = '.py'

def test_case_0():
    var_0 = '/path/to/module'

def test_case_0():
    var_0 = '/path/to/module'
    var_1 = '__init__.py'

def test_case_0():
    var_0 = '/path/to/nonexistent'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_src_path_finds_module_in_src_paths. Retrieved 7/11 statements.
# Partially parsed test_src_path_finds_package_in_src_paths. Retrieved 7/11 statements.
# Partially parsed test_src_path_finds_namespace_package. Retrieved 8/12 statements.
# Partially parsed test_src_path_with_auto_identify_namespace_packages. Retrieved 7/11 statements.
# Partially parsed test_src_path_returns_none_for_unfound_module. Retrieved 7/11 statements.
# Partially parsed test_src_path_with_custom_src_paths. Retrieved 7/13 statements.
# Partially parsed test_src_path_with_prefix. Retrieved 9/13 statements.


def test_case_0():
    var_0 = 'src'
    var_1 = set()
    var_2 = False
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)
    var_6 = 'module_name'

def test_case_0():
    var_0 = 'src'
    var_1 = set()
    var_2 = False
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)
    var_6 = 'package_name'

def test_case_0():
    var_0 = 'src'
    var_1 = 'namespace'
    var_2 = {var_1}
    var_3 = False
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'namespace.module_name'

def test_case_0():
    var_0 = 'src'
    var_1 = set()
    var_2 = True
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)
    var_6 = 'namespace.module_name'

def test_case_0():
    var_0 = 'src'
    var_1 = set()
    var_2 = False
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)
    var_6 = 'nonexistent_module'

def test_case_0():
    var_0 = 'custom_src'
    var_1 = set()
    var_2 = False
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)
    var_6 = 'module_name'

def test_case_0():
    var_0 = 'src'
    var_1 = set()
    var_2 = False
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)
    var_6 = 'module_name'
    var_7 = 'prefix'
    var_8 = (var_7,)



# Parsed testcases at query #8
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test.namespace'
    var_1 = {var_0}
    var_2 = False
    var_3 = []
    var_4 = []
    var_5 = module_0.Config()
    var_6 = 'test.namespace.module'
    var_7 = []
    var_8 = 'test'
    var_9 = (var_8,)
    var_10 = module_1._src_path(var_6, var_5, var_7, var_9)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_namespace_not_in_config_and_not_auto_identified. Retrieved 13/25 statements.


def test_case_0():
    var_0 = set()
    var_1 = False
    var_2 = []
    var_3 = []
    var_4 = 'some_path'
    var_5 = 'nested_module'
    var_6 = [var_5]
    var_7 = 'pre'
    var_8 = 'fix'
    var_9 = (var_7, var_8)
    var_10 = 'pre.fix.root_module_name'
    var_11 = 'root_module_name'
    var_12 = 'module_path'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_is_module_with_py_extension. Retrieved 2/9 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 2/10 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = '.py'

def test_case_0():
    var_0 = 'test_module'
    var_1 = 0

def test_case_0():
    var_0 = 'test_module'
    var_1 = '__init__.py'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 2/7 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 2/9 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'test_file'
    var_1 = '.py'

def test_case_0():
    var_0 = 'test_file'
    var_1 = 0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = '__init__.py'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 2/6 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 2/7 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = '.py'

def test_case_0():
    var_0 = 'test_module'
    var_1 = 0

def test_case_0():
    var_0 = 'test_module'
    var_1 = '__init__.py'



# Parsed testcases at query #13
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test.namespace'
    var_1 = {var_0}
    var_2 = False
    var_3 = []
    var_4 = []
    var_5 = module_0.Config()
    var_6 = 'test.namespace.module'
    var_7 = []
    var_8 = ()
    var_9 = module_1._src_path(var_6, var_5, var_7, var_8)



# Parsed testcases at query #14
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test.namespace'
    var_1 = {var_0}
    var_2 = False
    var_3 = module_0.Config()
    var_4 = 'test.namespace.module'
    var_5 = None
    var_6 = 'test'
    var_7 = 'namespace'
    var_8 = (var_6, var_7)
    var_9 = module_1._src_path(var_4, var_3, var_5, var_8)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = set()
    var_1 = True
    var_2 = '.py'
    var_3 = {var_2}
    var_4 = module_0.Config()
    var_5 = 'test.namespace.module'
    var_6 = None
    var_7 = 'test'
    var_8 = 'namespace'
    var_9 = (var_7, var_8)
    var_10 = module_1._src_path(var_5, var_4, var_6, var_9)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test__known_pattern_with_match. Retrieved 1/10 statements.
# Partially parsed test__known_pattern_without_match. Retrieved 1/10 statements.
# Partially parsed test__known_pattern_with_invalid_section. Retrieved 1/10 statements.
# Partially parsed test__known_pattern_with_empty_name. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 'example.module'

def test_case_0():
    var_0 = 'example.module'

def test_case_0():
    var_0 = 'example.module'

def test_case_0():
    var_0 = ''



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 2/7 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 1/6 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 2/7 statements.
# Partially parsed test_is_module_with_nonexistent_path. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'example'
    var_1 = '.py'

def test_case_0():
    var_0 = 'example'

def test_case_0():
    var_0 = 'example'
    var_1 = '__init__.py'

def test_case_0():
    var_0 = 'nonexistent'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test__is_module_with_py_file. Retrieved 2/7 statements.
# Partially parsed test__is_module_with_extension_suffix. Retrieved 1/6 statements.
# Partially parsed test__is_module_with_init_py. Retrieved 2/7 statements.
# Partially parsed test__is_module_with_multiple_conditions. Retrieved 3/15 statements.


def test_case_0():
    var_0 = 'example'
    var_1 = '.py'

def test_case_0():
    var_0 = 'example'

def test_case_0():
    var_0 = 'example'
    var_1 = '__init__.py'

def test_case_0():
    var_0 = 'example'
    var_1 = '.py'
    var_2 = '__init__.py'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_is_namespace_package_with_namespace_declaration. Retrieved 6/10 statements.
# Partially parsed test_is_namespace_package_with_extend_path_declaration. Retrieved 6/10 statements.
# Partially parsed test_is_namespace_package_with_double_quotes_declaration. Retrieved 6/10 statements.
# Partially parsed test_is_namespace_package_with_double_quotes_extend_path. Retrieved 6/10 statements.


def test_case_0():
    var_0 = '/some/namespace/package'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)
    var_4 = '__init__.py'
    var_5 = b"__import__('pkg_resources').declare_namespace(__name__)"

def test_case_0():
    var_0 = '/some/namespace/package'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)
    var_4 = '__init__.py'
    var_5 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"

def test_case_0():
    var_0 = '/some/namespace/package'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)
    var_4 = '__init__.py'
    var_5 = b'__import__("pkg_resources").declare_namespace(__name__)'

def test_case_0():
    var_0 = '/some/namespace/package'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)
    var_4 = '__init__.py'
    var_5 = b'__path__ = __import__("pkgutil").extend_path(__path__, __name__)'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_is_namespace_package_true_when_init_contains_namespace_declaration. Retrieved 2/18 statements.
# Partially parsed test_is_namespace_package_true_when_init_contains_extend_path_declaration. Retrieved 2/18 statements.
# Partially parsed test_is_namespace_package_true_when_no_init_and_no_other_files. Retrieved 2/18 statements.


def test_case_0():
    var_0 = b"__import__('pkg_resources').declare_namespace(__name__)"
    var_1 = frozenset()

def test_case_0():
    var_0 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_1 = frozenset()

def test_case_0():
    var_0 = False
    var_1 = frozenset()



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_src_path_is_module_true. Retrieved 1/5 statements.
# Partially parsed test_src_path_is_module_false_name_mismatch. Retrieved 2/6 statements.
# Partially parsed test_src_path_is_module_false_not_dir. Retrieved 1/5 statements.
# Partially parsed test_src_path_is_module_false_case_sensitive. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'test_module'

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'wrong_name'

def test_case_0():
    var_0 = 'test_file.txt'

def test_case_0():
    var_0 = 'Test_Module'
    var_1 = 'test_module'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_src_path_predicate_evaluates_to_true. Retrieved 2/18 statements.


def test_case_0():
    var_0 = '/some/path'
    var_1 = 'some_module'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 1/4 statements.
# Partially parsed test_is_module_with_extension. Retrieved 1/6 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 2/5 statements.
# Partially parsed test_is_module_with_nonexistent_path. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'test_module.py'

def test_case_0():
    var_0 = 'test_module'

def test_case_0():
    var_0 = 'test_package'
    var_1 = 'test_package/__init__.py'

def test_case_0():
    var_0 = 'nonexistent'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_src_path_is_module. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'test_module'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_is_namespace_package_with_namespace_declaration. Retrieved 5/10 statements.
# Partially parsed test_is_namespace_package_with_extend_path_declaration. Retrieved 5/10 statements.
# Partially parsed test_is_namespace_package_with_empty_directory. Retrieved 5/10 statements.


def test_case_0():
    var_0 = '/some/namespace/package'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)
    var_4 = '__init__.py'

def test_case_0():
    var_0 = '/some/namespace/package'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)
    var_4 = '__init__.py'

def test_case_0():
    var_0 = '/some/namespace/package'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)
    var_4 = '__init__.py'



# Parsed testcases at query #25
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test*'
    var_1 = [var_0]
    var_2 = 'test123'
    var_3 = module_0.Config()
    var_4 = module_1._forced_separate(var_2, var_3)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = [var_0]
    var_2 = 'test123'
    var_3 = module_0.Config()
    var_4 = module_1._forced_separate(var_2, var_3)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test*'
    var_1 = [var_0]
    var_2 = '.test123'
    var_3 = module_0.Config()
    var_4 = module_1._forced_separate(var_2, var_3)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'abc*'
    var_1 = [var_0]
    var_2 = 'test123'
    var_3 = module_0.Config()
    var_4 = module_1._forced_separate(var_2, var_3)
    assert var_4 is None



# Parsed testcases at query #26
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'foo'
    var_1 = [var_0]
    var_2 = 'foo'
    var_3 = module_0.Config()
    var_4 = module_1._forced_separate(var_2, var_3)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'foo*'
    var_1 = [var_0]
    var_2 = 'foobar'
    var_3 = module_0.Config()
    var_4 = module_1._forced_separate(var_2, var_3)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'foo'
    var_1 = [var_0]
    var_2 = '.foo'
    var_3 = module_0.Config()
    var_4 = module_1._forced_separate(var_2, var_3)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'bar'
    var_1 = [var_0]
    var_2 = 'foo'
    var_3 = module_0.Config()
    var_4 = module_1._forced_separate(var_2, var_3)
    assert var_4 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'bar'
    var_1 = 'foo'
    var_2 = [var_0, var_1]
    var_3 = 'foo'
    var_4 = module_0.Config()
    var_5 = module_1._forced_separate(var_3, var_4)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'bar'
    var_1 = 'baz'
    var_2 = [var_0, var_1]
    var_3 = 'foo'
    var_4 = module_0.Config()
    var_5 = module_1._forced_separate(var_3, var_4)
    assert var_5 is None



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_src_path_finds_module_in_src_paths. Retrieved 5/9 statements.
# Partially parsed test_src_path_returns_none_when_module_not_found. Retrieved 5/9 statements.
# Partially parsed test_src_path_handles_namespace_packages. Retrieved 6/10 statements.
# Partially parsed test_src_path_handles_nested_module_in_namespace_package. Retrieved 6/10 statements.
# Partially parsed test_src_path_handles_module_in_root_src_path. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'src'
    var_1 = set()
    var_2 = False
    var_3 = frozenset()
    var_4 = 'module_name'

def test_case_0():
    var_0 = 'src'
    var_1 = set()
    var_2 = False
    var_3 = frozenset()
    var_4 = 'non_existent_module'

def test_case_0():
    var_0 = 'src'
    var_1 = 'namespace.package'
    var_2 = {var_1}
    var_3 = True
    var_4 = frozenset()
    var_5 = 'namespace.package.module'

def test_case_0():
    var_0 = 'src'
    var_1 = 'namespace.package'
    var_2 = {var_1}
    var_3 = True
    var_4 = frozenset()
    var_5 = 'namespace.package.nested.module'

def test_case_0():
    var_0 = 'src'
    var_1 = set()
    var_2 = False
    var_3 = frozenset()



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_known_pattern_predicate_false. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'test.module'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_src_path_with_valid_module. Retrieved 4/6 statements.
# Partially parsed test_src_path_with_nonexistent_module. Retrieved 4/6 statements.
# Partially parsed test_src_path_with_namespace_package. Retrieved 5/8 statements.
# Partially parsed test_src_path_with_auto_identified_namespace_package. Retrieved 4/7 statements.
# Partially parsed test_src_path_with_module_in_src_path_root. Retrieved 4/6 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'tests/test_data'
    var_2 = 'valid_module'
    var_3 = module_1._src_path(var_2, var_0)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'tests/test_data'
    var_2 = 'nonexistent_module'
    var_3 = module_1._src_path(var_2, var_0)
    assert var_3 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'tests/test_data'
    var_2 = 'namespace.package'
    var_3 = 'namespace.package.submodule'
    var_4 = module_1._src_path(var_3, var_0)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'tests/test_data'
    var_2 = 'auto_namespace.submodule'
    var_3 = module_1._src_path(var_2, var_0)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'tests/test_data/module_at_root'
    var_2 = 'module_at_root'
    var_3 = module_1._src_path(var_2, var_0)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_namespace_in_config_namespace_packages. Retrieved 6/8 statements.
# Partially parsed test_auto_identify_namespace_packages. Retrieved 6/8 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test.namespace'
    var_1 = {var_0}
    var_2 = False
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = module_0.Config()
    var_6 = 'test.namespace.module'
    var_7 = module_1._src_path(var_6, var_5)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = set()
    var_1 = True
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = module_0.Config()
    var_5 = 'test.namespace.module'
    var_6 = module_1._src_path(var_5, var_4)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_src_path_is_module_true. Retrieved 1/5 statements.
# Partially parsed test_src_path_is_module_false_wrong_name. Retrieved 2/6 statements.
# Partially parsed test_src_path_is_module_false_not_dir. Retrieved 1/5 statements.
# Partially parsed test_src_path_is_module_false_case_sensitive. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'valid_module'

def test_case_0():
    var_0 = 'invalid_module'
    var_1 = 'wrong_name'

def test_case_0():
    var_0 = 'not_a_directory'

def test_case_0():
    var_0 = 'case_sensitive_module'
    var_1 = 'Case_Sensitive_Module'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_src_path_predicate_evaluates_to_true. Retrieved 7/23 statements.


def test_case_0():
    var_0 = 'src'
    var_1 = 'namespace'
    var_2 = {var_1}
    var_3 = True
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = 'module'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_src_path_is_module_true. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 'module_name'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_src_path_predicate_evaluates_to_true. Retrieved 4/31 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = '.py'
    var_2 = [var_1]
    var_3 = 'module'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 1/3 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 1/3 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 1/3 statements.
# Partially parsed test_is_module_with_nonexistent_path. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'test_module.py'

def test_case_0():
    var_0 = 'test_package'

def test_case_0():
    var_0 = 'test_extension'

def test_case_0():
    var_0 = 'nonexistent'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test__is_namespace_package_with_valid_namespace_package. Retrieved 4/6 statements.
# Partially parsed test__is_namespace_package_with_non_package_directory. Retrieved 4/6 statements.
# Partially parsed test__is_namespace_package_with_init_file_and_namespace_declaration. Retrieved 4/6 statements.
# Partially parsed test__is_namespace_package_with_init_file_but_no_namespace_declaration. Retrieved 4/6 statements.
# Partially parsed test__is_namespace_package_with_no_init_file_but_other_files. Retrieved 4/6 statements.
# Partially parsed test__is_namespace_package_with_no_init_file_and_no_other_files. Retrieved 4/6 statements.


def test_case_0():
    var_0 = '/valid/namespace/package'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = '/non/package/directory'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = '/valid/init/namespace/package'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = '/invalid/init/namespace/package'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = '/invalid/no/init/namespace/package'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = '/valid/no/init/namespace/package'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_namespace_not_in_config_and_auto_identify_disabled. Retrieved 8/11 statements.


def test_case_0():
    var_0 = set()
    var_1 = False
    var_2 = []
    var_3 = []
    var_4 = 'test'
    var_5 = []
    var_6 = 'existing'
    var_7 = (var_6,)



# Parsed testcases at query #38
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'example*'
    var_1 = [var_0]
    var_2 = module_0.Config(var_1)
    var_3 = 'example'
    var_4 = module_1._forced_separate(var_3, var_2)



# Parsed testcases at query #39
#--------------------------

# Partially parsed test__src_path_with_none_src_paths. Retrieved 6/10 statements.
# Partially parsed test__src_path_with_existing_module. Retrieved 5/11 statements.
# Partially parsed test__src_path_with_namespace_package. Retrieved 6/12 statements.
# Partially parsed test__src_path_with_auto_identify_namespace_package. Retrieved 5/11 statements.
# Partially parsed test__src_path_with_src_path_is_module. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'src'
    var_1 = set()
    var_2 = False
    var_3 = frozenset()
    var_4 = 'module'
    var_5 = None

def test_case_0():
    var_0 = 'tests'
    var_1 = set()
    var_2 = False
    var_3 = frozenset()
    var_4 = 'test_module'

def test_case_0():
    var_0 = 'src'
    var_1 = 'namespace'
    var_2 = {var_1}
    var_3 = False
    var_4 = frozenset()
    var_5 = 'namespace.module'

def test_case_0():
    var_0 = 'src'
    var_1 = set()
    var_2 = True
    var_3 = frozenset()
    var_4 = 'namespace.module'

def test_case_0():
    var_0 = 'src'
    var_1 = set()
    var_2 = False
    var_3 = frozenset()



# Parsed testcases at query #40
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = set()
    var_1 = False
    var_2 = []
    var_3 = []
    var_4 = module_0.Config()
    var_5 = 'example'
    var_6 = []
    var_7 = ()
    var_8 = 'module'
    var_9 = [var_8]
    var_10 = 'example'
    var_11 = 'path/to/module'
    var_12 = var_4.namespace_packages
    var_13 = var_10 in var_12
    var_14 = var_4.auto_identify_namespace_packages
    var_15 = var_4.supported_extensions
    var_16 = module_1._is_namespace_package(var_11, var_15)
    var_17 = var_14 and var_16
    var_18 = var_13 or var_17



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_namespace_in_config_namespace_packages. Retrieved 17/19 statements.
# Partially parsed test_auto_identify_namespace_packages_with_namespace_package. Retrieved 29/32 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'Config'
    var_1 = 'src_paths'
    var_2 = 'namespace_packages'
    var_3 = 'auto_identify_namespace_packages'
    var_4 = 'supported_extensions'
    var_5 = [var_1, var_2, var_3, var_4]
    var_6 = []
    var_7 = 'test.namespace'
    var_8 = {var_7}
    var_9 = False
    var_10 = []
    var_11 = module_0.Config()
    var_12 = 'test.namespace.module'
    var_13 = 'test'
    var_14 = 'namespace'
    var_15 = (var_13, var_14)
    var_16 = module_1._src_path(var_12, var_11, prefix=var_15)

import isort.settings as module_0
import pathlib as module_1
import isort.place as module_2

def test_case_0():
    var_0 = 'Config'
    var_1 = 'src_paths'
    var_2 = 'namespace_packages'
    var_3 = 'auto_identify_namespace_packages'
    var_4 = 'supported_extensions'
    var_5 = [var_1, var_2, var_3, var_4]
    var_6 = []
    var_7 = set()
    var_8 = True
    var_9 = '.py'
    var_10 = [var_9]
    var_11 = module_0.Config()
    var_12 = 'Path'
    var_13 = 'is_dir'
    var_14 = 'resolve'
    var_15 = 'name'
    var_16 = [var_13, var_14, var_15]
    var_17 = False
    var_18 = lambda : var_17
    var_19 = 'module_path'
    var_20 = lambda : var_19
    var_21 = 'module'
    var_22 = module_1.Path()
    var_23 = 'test.namespace.module'
    var_24 = (var_22,)
    var_25 = 'test'
    var_26 = 'namespace'
    var_27 = (var_25, var_26)
    var_28 = module_2._src_path(var_23, var_11, var_24, var_27)



# Parsed testcases at query #42
#--------------------------

# Partially parsed test__known_pattern_with_matching_pattern. Retrieved 4/18 statements.
# Partially parsed test__known_pattern_with_no_matching_pattern. Retrieved 4/18 statements.
# Partially parsed test__known_pattern_with_matching_pattern_but_section_not_in_config. Retrieved 5/19 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = 'section_name'
    var_2 = {var_1}
    var_3 = 'module.submodule.component'

def test_case_0():
    var_0 = 'module.submodule'
    var_1 = 'section_name'
    var_2 = {var_1}
    var_3 = 'another.module.component'

def test_case_0():
    var_0 = 'module.submodule'
    var_1 = 'section_name'
    var_2 = 'another_section'
    var_3 = {var_2}
    var_4 = 'module.submodule.component'



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_src_path_predicate_evaluates_to_true. Retrieved 7/36 statements.


def test_case_0():
    var_0 = '/path/to/src'
    var_1 = set()
    var_2 = True
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = 'module_name'
    var_6 = ()



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_is_namespace_package_with_init_file. Retrieved 6/13 statements.
# Partially parsed test_is_namespace_package_without_init_file_and_files. Retrieved 4/8 statements.
# Partially parsed test_is_namespace_package_without_init_file_and_with_files. Retrieved 5/13 statements.
# Partially parsed test_is_namespace_package_with_init_file_and_no_namespace_declaration. Retrieved 6/13 statements.
# Partially parsed test_is_namespace_package_with_init_file_and_namespace_declaration_different_syntax. Retrieved 6/13 statements.
# Partially parsed test_is_namespace_package_with_init_file_and_namespace_declaration_double_quotes. Retrieved 6/13 statements.
# Partially parsed test_is_namespace_package_with_init_file_and_namespace_declaration_double_quotes_pkgutil. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'test_package'
    var_1 = '__init__.py'
    var_2 = "__import__('pkg_resources').declare_namespace(__name__)"
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'test_package'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'test_package'
    var_1 = 'test_file.py'
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = 'test_package'
    var_1 = '__init__.py'
    var_2 = "print('Hello World')"
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'test_package'
    var_1 = '__init__.py'
    var_2 = "__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'test_package'
    var_1 = '__init__.py'
    var_2 = '__import__("pkg_resources").declare_namespace(__name__)'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'test_package'
    var_1 = '__init__.py'
    var_2 = '__path__ = __import__("pkgutil").extend_path(__path__, __name__)'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_is_namespace_package_without_init_file_and_no_files. Retrieved 6/11 statements.
# Partially parsed test_is_namespace_package_with_init_file_and_namespace_declaration. Retrieved 8/15 statements.
# Partially parsed test_is_namespace_package_with_init_file_and_extend_path_declaration. Retrieved 8/15 statements.


def test_case_0():
    var_0 = '/some/path'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)
    var_4 = True
    var_5 = []

def test_case_0():
    var_0 = '/some/path'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)
    var_4 = True
    var_5 = '__init__.py'
    var_6 = b"__import__('pkg_resources').declare_namespace(__name__)"
    var_7 = [var_6]

def test_case_0():
    var_0 = '/some/path'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)
    var_4 = True
    var_5 = '__init__.py'
    var_6 = b"__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_7 = [var_6]



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_src_path_predicate_evaluates_true. Retrieved 11/25 statements.


import isort.place as module_0

def test_case_0():
    var_0 = '/mock/path'
    var_1 = 'mock_module'
    var_2 = '/'
    var_3 = var_0 + var_2
    var_4 = var_3 + var_1
    var_5 = module_0._is_module(var_4)
    var_6 = var_0 + var_2
    var_7 = var_6 + var_1
    var_8 = module_0._is_package(var_7)
    var_9 = module_0._src_path_is_module(var_0, var_1)
    var_10 = var_5 or var_8 or var_9
    assert var_10 is True



# Parsed testcases at query #47
#--------------------------

# Partially parsed test__is_namespace_package_with_non_package_path. Retrieved 4/6 statements.
# Partially parsed test__is_namespace_package_with_empty_directory. Retrieved 5/10 statements.
# Partially parsed test__is_namespace_package_with_non_empty_directory_and_no_init. Retrieved 7/16 statements.
# Partially parsed test__is_namespace_package_with_init_but_no_namespace_markers. Retrieved 7/15 statements.
# Partially parsed test__is_namespace_package_with_pkg_resources_marker. Retrieved 7/15 statements.
# Partially parsed test__is_namespace_package_with_pkgutil_marker. Retrieved 7/15 statements.


def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = '/empty/directory'
    var_1 = True
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '/directory/with/files'
    var_1 = True
    var_2 = 'file.py'
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)
    var_6 = 'file.py'

def test_case_0():
    var_0 = '/package/without/namespace'
    var_1 = True
    var_2 = '__init__.py'
    var_3 = "print('hello')"
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = '/package/with/pkg_resources'
    var_1 = True
    var_2 = '__init__.py'
    var_3 = "__import__('pkg_resources').declare_namespace(__name__)"
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)

def test_case_0():
    var_0 = '/package/with/pkgutil'
    var_1 = True
    var_2 = '__init__.py'
    var_3 = "__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_is_namespace_package_with_valid_namespace_package. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_invalid_package. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_non_namespace_package. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_non_existent_package. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_valid_namespace_package_and_additional_files. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 'valid_namespace_package'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'invalid_package'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'non_namespace_package'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'non_existent_package'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'valid_namespace_package_with_files'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_known_pattern_with_no_matching_pattern. Retrieved 1/7 statements.
# Partially parsed test_known_pattern_with_non_matching_pattern. Retrieved 1/10 statements.
# Partially parsed test_known_pattern_with_non_matching_section. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 'test.module'

def test_case_0():
    var_0 = 'test.module'

def test_case_0():
    var_0 = 'test.module'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_is_namespace_package_true. Retrieved 2/31 statements.


def test_case_0():
    var_0 = b"__import__('pkg_resources').declare_namespace(__name__)"
    var_1 = frozenset()



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_is_namespace_package_with_valid_package. Retrieved 6/11 statements.
# Partially parsed test_is_namespace_package_with_non_package. Retrieved 4/7 statements.
# Partially parsed test_is_namespace_package_with_empty_package. Retrieved 4/7 statements.
# Partially parsed test_is_namespace_package_with_non_namespace_package. Retrieved 6/11 statements.
# Partially parsed test_is_namespace_package_with_other_files. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'valid_package'
    var_1 = '__init__.py'
    var_2 = "__import__('pkg_resources').declare_namespace(__name__)"
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'non_package'
    var_1 = '.py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'empty_package'
    var_1 = '.py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'non_namespace_package'
    var_1 = '__init__.py'
    var_2 = "print('hello')"
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'package_with_files'
    var_1 = 'setup.cfg'
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_is_namespace_package_returns_false_when_not_a_package. Retrieved 4/7 statements.
# Partially parsed test_is_namespace_package_returns_false_when_init_file_missing_and_other_files_present. Retrieved 5/10 statements.
# Partially parsed test_is_namespace_package_returns_false_when_init_file_exists_but_no_namespace_declaration. Retrieved 6/13 statements.


def test_case_0():
    var_0 = '/some/path'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = '/some/path'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)
    var_4 = 'setup.cfg'

def test_case_0():
    var_0 = '/some/path'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)
    var_4 = '__init__.py'
    var_5 = b"print('Hello, World!')"



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_known_pattern_predicate_evaluates_false. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'some.module.name'



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_src_path_namespace_in_config_namespace_packages. Retrieved 6/8 statements.
# Partially parsed test_src_path_auto_identify_namespace_packages. Retrieved 6/10 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'test.namespace'
    var_1 = {var_0}
    var_2 = False
    var_3 = '.py'
    var_4 = [var_3]
    var_5 = 'test.namespace.module'
    var_6 = module_0.Config()
    var_7 = module_1._src_path(var_5, var_6)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = set()
    var_1 = True
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = 'test.namespace.module'
    var_5 = module_0.Config()
    var_6 = module_1._src_path(var_4, var_5)



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_predicate_at_line_19_evaluates_to_false. Retrieved 7/10 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = set()
    var_1 = False
    var_2 = module_0.Config()
    var_3 = 'module'
    var_4 = 'path'
    var_5 = 'prefix'
    var_6 = (var_5,)



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_src_path_returns_tuple_when_predicate_true. Retrieved 2/17 statements.


def test_case_0():
    var_0 = '/path/to/module'
    var_1 = 'module'



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 1/5 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 1/5 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'example'

def test_case_0():
    var_0 = 'example'

def test_case_0():
    var_0 = 'example'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test__is_module_with_py_file. Retrieved 2/7 statements.
# Partially parsed test__is_module_with_extension_suffix. Retrieved 1/6 statements.
# Partially parsed test__is_module_with_init_py. Retrieved 2/7 statements.
# Partially parsed test__is_module_with_nonexistent_path. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = '.py'

def test_case_0():
    var_0 = 'test_module'

def test_case_0():
    var_0 = 'test_module'
    var_1 = '__init__.py'

def test_case_0():
    var_0 = 'nonexistent_module'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test__known_pattern_with_matching_pattern. Retrieved 3/17 statements.
# Partially parsed test__known_pattern_with_no_matching_pattern. Retrieved 4/18 statements.
# Partially parsed test__known_pattern_with_matching_pattern_but_not_in_sections. Retrieved 4/18 statements.
# Partially parsed test__known_pattern_with_multiple_patterns. Retrieved 5/21 statements.


def test_case_0():
    var_0 = 'module.submodule'
    var_1 = 'section1'
    var_2 = {var_1}

def test_case_0():
    var_0 = 'module.submodule'
    var_1 = 'section1'
    var_2 = {var_1}
    var_3 = 'module.other'

def test_case_0():
    var_0 = 'module.submodule'
    var_1 = 'section1'
    var_2 = 'section2'
    var_3 = {var_2}

def test_case_0():
    var_0 = 'module.submodule'
    var_1 = 'section1'
    var_2 = 'module'
    var_3 = 'section2'
    var_4 = {var_1, var_3}



# Parsed testcases at query #3
#--------------------------

# Partially parsed test__src_path_finds_module_in_src_paths. Retrieved 5/9 statements.
# Partially parsed test__src_path_handles_namespace_packages. Retrieved 7/11 statements.
# Partially parsed test__src_path_auto_identifies_namespace_packages. Retrieved 6/10 statements.
# Partially parsed test__src_path_returns_none_for_non_existent_module. Retrieved 5/9 statements.
# Partially parsed test__src_path_handles_root_module_in_src_path. Retrieved 5/9 statements.


def test_case_0():
    var_0 = '/path/to/src'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)
    var_4 = 'module'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = 'namespace'
    var_2 = [var_1]
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)
    var_6 = 'namespace.module'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = True
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)
    var_5 = 'namespace.module'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)
    var_4 = 'nonexistent'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)
    var_4 = 'src'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_src_path_predicate_evaluates_to_true. Retrieved 4/18 statements.


def test_case_0():
    var_0 = set()
    var_1 = False
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = 'test_module'



# Parsed testcases at query #5
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'foo*'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'foobar'
    var_4 = module_1._forced_separate(var_3, var_2)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'foo'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'foobar'
    var_4 = module_1._forced_separate(var_3, var_2)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'foo*'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = '.foobar'
    var_4 = module_1._forced_separate(var_3, var_2)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'foo*'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = 'bar'
    var_4 = module_1._forced_separate(var_3, var_2)
    assert var_4 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'foo*'
    var_1 = 'bar*'
    var_2 = [var_0, var_1]
    var_3 = module_0.Config()
    var_4 = 'foobar'
    var_5 = module_1._forced_separate(var_4, var_3)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'foo*'
    var_1 = 'bar*'
    var_2 = [var_0, var_1]
    var_3 = module_0.Config()
    var_4 = 'barbaz'
    var_5 = module_1._forced_separate(var_4, var_3)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = []
    var_1 = module_0.Config()
    var_2 = 'foobar'
    var_3 = module_1._forced_separate(var_2, var_1)
    assert var_3 is None



# Parsed testcases at query #6
#--------------------------

# Partially parsed test__src_path_finds_module. Retrieved 6/8 statements.
# Partially parsed test__src_path_finds_namespace_package. Retrieved 6/8 statements.
# Partially parsed test__src_path_returns_none_when_not_found. Retrieved 6/8 statements.
# Partially parsed test__src_path_finds_module_in_nested_path. Retrieved 8/13 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = set()
    var_1 = False
    var_2 = frozenset(['py'])
    var_3 = module_0.Config()
    var_4 = 'module_name'
    var_5 = module_1._src_path(var_4, var_3)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = set(['namespace'])
    var_1 = True
    var_2 = frozenset(['py'])
    var_3 = module_0.Config()
    var_4 = 'namespace.module_name'
    var_5 = module_1._src_path(var_4, var_3)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = set()
    var_1 = False
    var_2 = frozenset(['py'])
    var_3 = module_0.Config()
    var_4 = 'unknown_module'
    var_5 = module_1._src_path(var_4, var_3)
    assert var_5 is None

import isort.settings as module_0

def test_case_0():
    var_0 = set()
    var_1 = False
    var_2 = frozenset(['py'])
    var_3 = module_0.Config()
    var_4 = 'nested.module_name'
    var_5 = '/path/to/src'
    var_6 = 'nested'
    var_7 = (var_6,)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_is_namespace_package_with_valid_namespace_package. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_non_package. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_init_file_not_namespace. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_no_init_file_but_other_files. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_no_init_file_and_no_other_files. Retrieved 4/6 statements.


def test_case_0():
    var_0 = '/valid/namespace/package'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = '/invalid/not/a/package'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = '/invalid/not/a/namespace/package'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = '/invalid/no/init/but/other/files'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = '/valid/namespace/package/no/init/no/files'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_is_namespace_package_with_valid_namespace_package. Retrieved 6/13 statements.
# Partially parsed test_is_namespace_package_with_invalid_namespace_package. Retrieved 6/13 statements.
# Partially parsed test_is_namespace_package_with_no_init_file_but_other_files. Retrieved 6/14 statements.
# Partially parsed test_is_namespace_package_with_no_init_file_and_no_other_files. Retrieved 4/8 statements.
# Partially parsed test_is_namespace_package_with_extend_path_in_init_file. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'valid_namespace_package'
    var_1 = '__init__.py'
    var_2 = "__import__('pkg_resources').declare_namespace(__name__)"
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'invalid_namespace_package'
    var_1 = '__init__.py'
    var_2 = "print('Hello, World!')"
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'no_init_file_but_other_files'
    var_1 = 'setup.cfg'
    var_2 = '[metadata]'
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'no_init_file_and_no_other_files'
    var_1 = 'py'
    var_2 = {var_1}
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'extend_path_in_init_file'
    var_1 = '__init__.py'
    var_2 = "__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_3 = 'py'
    var_4 = {var_3}
    var_5 = frozenset(var_4)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_src_path_with_valid_module. Retrieved 5/9 statements.
# Partially parsed test_src_path_with_namespace_package. Retrieved 6/10 statements.
# Partially parsed test_src_path_with_auto_identify_namespace_package. Retrieved 5/9 statements.
# Partially parsed test_src_path_with_invalid_module. Retrieved 5/9 statements.
# Partially parsed test_src_path_with_src_path_is_module. Retrieved 5/9 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = set()
    var_2 = False
    var_3 = frozenset()
    var_4 = 'valid_module'

def test_case_0():
    var_0 = '/src'
    var_1 = 'namespace'
    var_2 = {var_1}
    var_3 = False
    var_4 = frozenset()
    var_5 = 'namespace.module'

def test_case_0():
    var_0 = '/src'
    var_1 = set()
    var_2 = True
    var_3 = frozenset()
    var_4 = 'auto_namespace.module'

def test_case_0():
    var_0 = '/src'
    var_1 = set()
    var_2 = False
    var_3 = frozenset()
    var_4 = 'invalid_module'

def test_case_0():
    var_0 = '/src'
    var_1 = set()
    var_2 = False
    var_3 = frozenset()
    var_4 = 'src'



# Parsed testcases at query #10
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'foo*'
    var_1 = [var_0]
    var_2 = 'foo.bar'
    var_3 = module_0.Config()
    var_4 = module_1._forced_separate(var_2, var_3)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'foo*'
    var_1 = [var_0]
    var_2 = 'foobar'
    var_3 = module_0.Config()
    var_4 = module_1._forced_separate(var_2, var_3)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'foo*'
    var_1 = [var_0]
    var_2 = '.foo.bar'
    var_3 = module_0.Config()
    var_4 = module_1._forced_separate(var_2, var_3)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'foo*'
    var_1 = [var_0]
    var_2 = 'bar.foo'
    var_3 = module_0.Config()
    var_4 = module_1._forced_separate(var_2, var_3)
    assert var_4 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'foo*'
    var_1 = 'bar*'
    var_2 = [var_0, var_1]
    var_3 = 'foo.bar'
    var_4 = module_0.Config()
    var_5 = module_1._forced_separate(var_3, var_4)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'foo*'
    var_1 = 'bar*'
    var_2 = [var_0, var_1]
    var_3 = 'bar.foo'
    var_4 = module_0.Config()
    var_5 = module_1._forced_separate(var_3, var_4)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'foo'
    var_1 = [var_0]
    var_2 = 'foo.bar'
    var_3 = module_0.Config()
    var_4 = module_1._forced_separate(var_2, var_3)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'foo'
    var_1 = [var_0]
    var_2 = 'foobar'
    var_3 = module_0.Config()
    var_4 = module_1._forced_separate(var_2, var_3)
    assert var_4 is None



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_src_path_is_module_valid_directory. Retrieved 1/5 statements.
# Partially parsed test_src_path_is_module_invalid_directory. Retrieved 1/3 statements.
# Partially parsed test_src_path_is_module_case_sensitive. Retrieved 2/6 statements.
# Partially parsed test_src_path_is_module_not_a_directory. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'valid_module'

def test_case_0():
    var_0 = 'invalid_module'

def test_case_0():
    var_0 = 'Module'
    var_1 = 'module'

def test_case_0():
    var_0 = 'file.txt'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_is_namespace_package_with_nonexistent_init_file_and_no_other_files. Retrieved 4/6 statements.
# Partially parsed test_is_namespace_package_with_existing_init_file_and_namespace_declaration. Retrieved 6/10 statements.
# Partially parsed test_is_namespace_package_with_existing_init_file_and_no_namespace_declaration. Retrieved 6/10 statements.
# Partially parsed test_is_namespace_package_with_nonexistent_init_file_and_other_files. Retrieved 5/9 statements.


def test_case_0():
    var_0 = '/nonexistent/path'
    var_1 = '.py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = '/valid/namespace/path'
    var_1 = '.py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)
    var_4 = '__init__.py'
    var_5 = "__import__('pkg_resources').declare_namespace(__name__)"

def test_case_0():
    var_0 = '/invalid/namespace/path'
    var_1 = '.py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)
    var_4 = '__init__.py'
    var_5 = 'some other content'

def test_case_0():
    var_0 = '/path/with/other/files'
    var_1 = '.py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)
    var_4 = 'some_file.py'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_is_module_with_py_extension. Retrieved 1/12 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 1/12 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 1/12 statements.


def test_case_0():
    var_0 = 'module'

def test_case_0():
    var_0 = 'module'

def test_case_0():
    var_0 = 'module'



# Parsed testcases at query #14
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'example.namespace'
    var_1 = {var_0}
    var_2 = False
    var_3 = module_0.Config()
    var_4 = 'example.namespace.module'
    var_5 = module_1._src_path(var_4, var_3)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = set()
    var_1 = True
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = module_0.Config()
    var_5 = 'example.namespace.module'
    var_6 = module_1._src_path(var_5, var_4)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 2/7 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 2/9 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 2/9 statements.
# Partially parsed test_is_module_without_any_files. Retrieved 1/3 statements.
# Partially parsed test_is_module_with_case_sensitive_check. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'example_module.py'
    var_1 = 'example_module'

def test_case_0():
    var_0 = 'example_module'
    var_1 = 0

def test_case_0():
    var_0 = 'example_package'
    var_1 = '__init__.py'

def test_case_0():
    var_0 = 'non_existent_module'

def test_case_0():
    var_0 = 'Example_Module.py'
    var_1 = 'example_module'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test__is_namespace_package_with_non_existent_path. Retrieved 4/6 statements.
# Partially parsed test__is_namespace_package_with_file_path. Retrieved 4/6 statements.
# Partially parsed test__is_namespace_package_with_empty_directory. Retrieved 4/6 statements.
# Partially parsed test__is_namespace_package_with_non_empty_directory_no_init. Retrieved 4/6 statements.
# Partially parsed test__is_namespace_package_with_init_but_no_namespace_declaration. Retrieved 4/6 statements.
# Partially parsed test__is_namespace_package_with_pkg_resources_namespace. Retrieved 4/6 statements.
# Partially parsed test__is_namespace_package_with_pkgutil_extend_path. Retrieved 4/6 statements.


def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = '/some/file.txt'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = '/empty/directory'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = '/directory/with/files'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = '/package/with/init'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = '/namespace/package'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = '/namespace/package'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test__known_pattern_matches_configured_pattern. Retrieved 3/17 statements.
# Partially parsed test__known_pattern_no_match. Retrieved 4/18 statements.
# Partially parsed test__known_pattern_section_not_in_config. Retrieved 4/18 statements.
# Partially parsed test__known_pattern_matches_longest_possible_module. Retrieved 5/21 statements.


def test_case_0():
    var_0 = 'foo.bar'
    var_1 = 'test_section'
    var_2 = {var_1}

def test_case_0():
    var_0 = 'foo.bar'
    var_1 = 'test_section'
    var_2 = {var_1}
    var_3 = 'not.matching'

def test_case_0():
    var_0 = 'foo.bar'
    var_1 = 'test_section'
    var_2 = 'other_section'
    var_3 = {var_2}

def test_case_0():
    var_0 = 'foo'
    var_1 = 'test_section1'
    var_2 = 'foo.bar'
    var_3 = 'test_section2'
    var_4 = {var_1, var_3}



# Parsed testcases at query #18
#--------------------------

# Partially parsed test__src_path_is_module_returns_true_for_valid_module. Retrieved 3/8 statements.
# Partially parsed test__src_path_is_module_returns_false_for_incorrect_name. Retrieved 3/8 statements.
# Partially parsed test__src_path_is_module_returns_false_for_file_instead_of_dir. Retrieved 2/7 statements.
# Partially parsed test__src_path_is_module_returns_false_for_case_mismatch. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'valid_module'
    var_1 = True
    assert var_1 is True
    var_2 = 'valid_module'

def test_case_0():
    var_0 = 'valid_module'
    var_1 = True
    assert var_1 is False
    var_2 = 'different_name'

def test_case_0():
    var_0 = 'file_module'
    var_1 = 'file_module'

def test_case_0():
    var_0 = 'ValidModule'
    var_1 = True
    assert var_1 is False
    var_2 = 'validmodule'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_src_path_finds_module_in_src_paths. Retrieved 5/9 statements.
# Partially parsed test_src_path_finds_package_in_src_paths. Retrieved 5/9 statements.
# Partially parsed test_src_path_finds_namespace_package. Retrieved 5/9 statements.
# Partially parsed test_src_path_returns_none_for_unfound_module. Retrieved 5/9 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = set()
    var_2 = False
    var_3 = frozenset()
    var_4 = 'module'

def test_case_0():
    var_0 = '/src'
    var_1 = set()
    var_2 = False
    var_3 = frozenset()
    var_4 = 'package'

def test_case_0():
    var_0 = '/src'
    var_1 = 'namespace.package'
    var_2 = {var_1}
    var_3 = True
    var_4 = frozenset()

def test_case_0():
    var_0 = '/src'
    var_1 = set()
    var_2 = False
    var_3 = frozenset()
    var_4 = 'unknown'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_known_pattern_matches. Retrieved 5/9 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = '^module\\.submodule$'
    var_2 = 'section'
    var_3 = 'module.submodule'
    var_4 = module_1._known_pattern(var_3, var_0)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 9/12 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = set()
    var_1 = False
    var_2 = {}
    var_3 = module_0.Config()
    var_4 = 'example'
    var_5 = 'path/to/src'
    var_6 = 'existing'
    var_7 = 'prefix'
    var_8 = (var_6, var_7)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_src_path_is_module_returns_true_when_all_conditions_are_met. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'test_module'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_src_path_is_module_valid. Retrieved 1/5 statements.
# Partially parsed test_src_path_is_module_invalid_name. Retrieved 2/6 statements.
# Partially parsed test_src_path_is_module_not_directory. Retrieved 1/5 statements.
# Partially parsed test_src_path_is_module_case_sensitive_mismatch. Retrieved 2/6 statements.
# Partially parsed test_src_path_is_module_nonexistent_path. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'valid_module'

def test_case_0():
    var_0 = 'invalid_module'
    var_1 = 'different_name'

def test_case_0():
    var_0 = 'not_a_directory'

def test_case_0():
    var_0 = 'CaseSensitive'
    var_1 = 'casesensitive'

def test_case_0():
    var_0 = 'nonexistent'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 1/3 statements.
# Partially parsed test_is_module_with_py_extension. Retrieved 1/3 statements.
# Partially parsed test_is_module_with_so_extension. Retrieved 1/3 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 1/3 statements.
# Partially parsed test_is_module_with_nonexistent_path. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'test_module.py'

def test_case_0():
    var_0 = 'test_module'

def test_case_0():
    var_0 = 'test_module'

def test_case_0():
    var_0 = 'test_package'

def test_case_0():
    var_0 = 'nonexistent'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_is_module_with_py_suffix. Retrieved 2/7 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 1/6 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'example'
    var_1 = '.py'

def test_case_0():
    var_0 = 'example'

def test_case_0():
    var_0 = 'example'
    var_1 = '__init__.py'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_src_path_predicate_evaluates_to_true. Retrieved 11/63 statements.


def test_case_0():
    var_0 = '/src'
    var_1 = '.py'
    var_2 = [var_1]
    var_3 = '/src/module.py'
    var_4 = True
    var_5 = 'module'
    var_6 = [var_1]
    var_7 = '/src/package'
    var_8 = 'package'
    var_9 = 'module.py'
    var_10 = [var_1]



# Parsed testcases at query #27
#--------------------------

# Partially parsed test__known_pattern_matches_configured_pattern. Retrieved 3/17 statements.
# Partially parsed test__known_pattern_no_match_when_section_not_in_config. Retrieved 4/18 statements.
# Partially parsed test__known_pattern_no_match_when_pattern_does_not_match. Retrieved 4/18 statements.
# Partially parsed test__known_pattern_matches_longest_possible_module_name. Retrieved 5/21 statements.
# Partially parsed test__known_pattern_returns_none_when_no_patterns_configured. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'a.b'
    var_1 = 'section1'
    var_2 = {var_1}

def test_case_0():
    var_0 = 'a.b'
    var_1 = 'section1'
    var_2 = 'section2'
    var_3 = {var_2}

def test_case_0():
    var_0 = 'a.b'
    var_1 = 'section1'
    var_2 = {var_1}
    var_3 = 'x.y'

def test_case_0():
    var_0 = 'a.b.c'
    var_1 = 'section1'
    var_2 = 'a.b'
    var_3 = 'section2'
    var_4 = {var_1, var_3}

def test_case_0():
    var_0 = []
    var_1 = 'section1'
    var_2 = {var_1}
    var_3 = 'a.b'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_namespace_not_in_config_and_not_auto_identify. Retrieved 7/10 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = set()
    var_1 = False
    var_2 = module_0.Config()
    var_3 = 'module.name'
    var_4 = '/path/to/src'
    var_5 = 'module'
    var_6 = (var_5,)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_is_namespace_package_true_when_init_file_contains_namespace_declaration. Retrieved 5/12 statements.
# Partially parsed test_is_namespace_package_true_when_init_file_contains_extend_path_declaration. Retrieved 5/12 statements.
# Partially parsed test_is_namespace_package_true_when_no_init_file_and_no_matching_files. Retrieved 3/8 statements.


def test_case_0():
    var_0 = '__init__.py'
    var_1 = "__import__('pkg_resources').declare_namespace(__name__)"
    var_2 = 'py'
    var_3 = {var_2}
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = '__init__.py'
    var_1 = "__path__ = __import__('pkgutil').extend_path(__path__, __name__)"
    var_2 = 'py'
    var_3 = {var_2}
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = 'py'
    var_1 = {var_0}
    var_2 = frozenset(var_1)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_is_module_with_py_extension. Retrieved 4/9 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 1/10 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module.py'
    var_2 = lambda x: x == var_1
    var_3 = var_2

def test_case_0():
    var_0 = 'test_module'

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module/__init__.py'
    var_2 = lambda x: x == var_1
    var_3 = var_2



# Parsed testcases at query #31
#--------------------------

# Partially parsed test__src_path_finds_module_in_src_paths. Retrieved 7/11 statements.
# Partially parsed test__src_path_handles_nested_module_in_namespace_package. Retrieved 8/12 statements.
# Partially parsed test__src_path_returns_none_for_non_existent_module. Retrieved 7/11 statements.
# Partially parsed test__src_path_handles_src_path_is_module_case. Retrieved 7/11 statements.
# Partially parsed test__src_path_auto_identifies_namespace_packages. Retrieved 7/11 statements.


def test_case_0():
    var_0 = '/path/to/src'
    var_1 = set()
    var_2 = False
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)
    var_6 = 'module'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = 'namespace'
    var_2 = {var_1}
    var_3 = False
    var_4 = 'py'
    var_5 = [var_4]
    var_6 = frozenset(var_5)
    var_7 = 'namespace.module'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = set()
    var_2 = False
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)
    var_6 = 'nonexistent'

def test_case_0():
    var_0 = '/path/to/module'
    var_1 = set()
    var_2 = False
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)
    var_6 = 'module'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = set()
    var_2 = True
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)
    var_6 = 'namespace.module'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_known_pattern_found. Retrieved 5/9 statements.
# Partially parsed test_known_pattern_not_found. Retrieved 5/9 statements.
# Partially parsed test_known_pattern_section_not_in_config. Retrieved 6/10 statements.
# Partially parsed test_known_pattern_partial_match. Retrieved 5/9 statements.
# Partially parsed test_known_pattern_multiple_patterns. Retrieved 7/13 statements.


import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'module\\.submodule'
    var_2 = 'section1'
    var_3 = 'module.submodule.function'
    var_4 = module_1._known_pattern(var_3, var_0)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'module\\.submodule'
    var_2 = 'section1'
    var_3 = 'othermodule.function'
    var_4 = module_1._known_pattern(var_3, var_0)
    assert var_4 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'module\\.submodule'
    var_2 = 'section1'
    var_3 = 'section2'
    var_4 = 'module.submodule.function'
    var_5 = module_1._known_pattern(var_4, var_0)
    assert var_5 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'module\\.submodule'
    var_2 = 'section1'
    var_3 = 'module.submodule.partial.function'
    var_4 = module_1._known_pattern(var_3, var_0)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'module\\.submodule'
    var_2 = 'section1'
    var_3 = 'othermodule'
    var_4 = 'section2'
    var_5 = 'module.submodule.function'
    var_6 = module_1._known_pattern(var_5, var_0)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test__src_path_returns_none_for_non_existent_module. Retrieved 2/6 statements.
# Partially parsed test__src_path_returns_firstparty_for_existing_module. Retrieved 2/9 statements.
# Partially parsed test__src_path_returns_firstparty_for_existing_package. Retrieved 2/11 statements.
# Partially parsed test__src_path_handles_namespace_packages. Retrieved 5/14 statements.
# Partially parsed test__src_path_handles_src_path_as_module. Retrieved 1/9 statements.
# Partially parsed test__src_path_handles_nested_modules_in_namespace_packages. Retrieved 6/17 statements.


def test_case_0():
    var_0 = '/nonexistent'
    var_1 = 'nonexistent_module'

def test_case_0():
    var_0 = 'existing_module.py'
    var_1 = 'existing_module'

def test_case_0():
    var_0 = 'existing_package'
    var_1 = '__init__.py'

def test_case_0():
    var_0 = 'namespace_pkg'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'nested_module.py'
    var_4 = 'namespace_pkg.nested_module'

def test_case_0():
    var_0 = 'module_name'

def test_case_0():
    var_0 = 'namespace_pkg'
    var_1 = 'nested_pkg'
    var_2 = '__init__.py'
    var_3 = [var_0]
    var_4 = True
    var_5 = 'namespace_pkg.nested_pkg'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test__src_path_with_none_src_paths. Retrieved 6/10 statements.
# Partially parsed test__src_path_with_namespace_package. Retrieved 6/10 statements.
# Partially parsed test__src_path_with_auto_identify_namespace_package. Retrieved 5/10 statements.
# Partially parsed test__src_path_with_module. Retrieved 5/10 statements.
# Partially parsed test__src_path_with_package. Retrieved 5/10 statements.
# Partially parsed test__src_path_with_src_path_is_module. Retrieved 5/10 statements.
# Partially parsed test__src_path_not_found. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'src'
    var_1 = set()
    var_2 = False
    var_3 = frozenset()
    var_4 = 'module'
    var_5 = None

def test_case_0():
    var_0 = 'src'
    var_1 = 'namespace'
    var_2 = {var_1}
    var_3 = False
    var_4 = frozenset()
    var_5 = 'namespace.module'

def test_case_0():
    var_0 = 'src'
    var_1 = set()
    var_2 = True
    var_3 = frozenset()
    var_4 = 'namespace.module'

def test_case_0():
    var_0 = 'src'
    var_1 = set()
    var_2 = False
    var_3 = frozenset()
    var_4 = 'module'

def test_case_0():
    var_0 = 'src'
    var_1 = set()
    var_2 = False
    var_3 = frozenset()
    var_4 = 'module'

def test_case_0():
    var_0 = 'src'
    var_1 = set()
    var_2 = False
    var_3 = frozenset()
    var_4 = 'src'

def test_case_0():
    var_0 = 'src'
    var_1 = set()
    var_2 = False
    var_3 = frozenset()
    var_4 = 'module'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_is_module_with_py_file. Retrieved 1/3 statements.
# Partially parsed test_is_module_with_init_py. Retrieved 1/3 statements.
# Partially parsed test_is_module_with_extension_suffix. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'test_module.py'

def test_case_0():
    var_0 = 'test_package'

def test_case_0():
    var_0 = 'test_extension'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_is_namespace_package_with_no_init_file_and_no_src_files. Retrieved 4/7 statements.
# Partially parsed test_is_namespace_package_with_init_file_containing_namespace_declaration. Retrieved 5/10 statements.


def test_case_0():
    var_0 = '/fake/path'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = '/fake/path'
    assert var_0 is True
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)
    var_4 = b"__import__('pkg_resources').declare_namespace(__name__)"



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_is_namespace_package_with_empty_directory. Retrieved 3/8 statements.
# Partially parsed test_is_namespace_package_with_non_python_files. Retrieved 4/11 statements.
# Partially parsed test_is_namespace_package_with_non_source_files. Retrieved 4/11 statements.
# Partially parsed test_is_namespace_package_with_source_files. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)

def test_case_0():
    var_0 = 'README.md'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'setup.cfg'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'module.py'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test__is_namespace_package_with_valid_namespace_package. Retrieved 4/21 statements.


def test_case_0():
    var_0 = "__import__('pkg_resources').declare_namespace(__name__)"
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)



# Parsed testcases at query #39
#--------------------------

# Partially parsed test__src_path_finds_module_in_src_paths. Retrieved 2/6 statements.
# Partially parsed test__src_path_returns_none_when_module_not_found. Retrieved 2/6 statements.
# Partially parsed test__src_path_handles_namespace_package. Retrieved 4/8 statements.
# Partially parsed test__src_path_handles_auto_identify_namespace_packages. Retrieved 3/7 statements.
# Partially parsed test__src_path_handles_module_in_root_src_path. Retrieved 2/6 statements.


def test_case_0():
    var_0 = '/path/to/src'
    var_1 = 'module_name'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = 'nonexistent_module'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = 'namespace'
    var_2 = {var_1}
    var_3 = 'namespace.module'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = True
    var_2 = 'namespace.module'

def test_case_0():
    var_0 = '/path/to/src'
    var_1 = 'src'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_namespace_in_config_namespace_packages. Retrieved 9/12 statements.
# Partially parsed test_auto_identify_namespace_packages_with_valid_namespace. Retrieved 12/14 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'test.namespace'
    var_1 = {var_0}
    var_2 = False
    var_3 = module_0.Config()
    var_4 = 'test.namespace.module'
    var_5 = '/path'
    var_6 = 'test'
    var_7 = 'namespace'
    var_8 = (var_6, var_7)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = set()
    var_1 = True
    var_2 = '.py'
    var_3 = {var_2}
    var_4 = module_0.Config()
    var_5 = 'test.namespace.module'
    var_6 = '/path'
    var_7 = [var_2]
    var_8 = 'test'
    var_9 = 'namespace'
    var_10 = (var_8, var_9)
    var_11 = module_1._src_path(var_5, var_4, var_7, var_10)



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_is_namespace_package_with_no_init_file_and_no_other_files. Retrieved 3/11 statements.
# Partially parsed test_is_namespace_package_with_namespace_declaration_in_init_file. Retrieved 3/13 statements.


def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)

def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)



# Parsed testcases at query #42
#--------------------------




import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'section1'
    var_1 = 'section2'
    var_2 = [var_0, var_1]
    var_3 = 'pattern1'
    var_4 = 'section1'
    var_5 = (var_3, var_4)
    var_6 = 'pattern2'
    var_7 = 'section2'
    var_8 = (var_6, var_7)
    var_9 = [var_5, var_8]
    var_10 = module_0.Config()
    var_11 = 'unknown.module'
    var_12 = module_1._known_pattern(var_11, var_10)
    assert var_12 is None

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'section1'
    var_1 = 'section2'
    var_2 = [var_0, var_1]
    var_3 = 'pattern1'
    var_4 = 'section1'
    var_5 = (var_3, var_4)
    var_6 = 'pattern2'
    var_7 = 'section2'
    var_8 = (var_6, var_7)
    var_9 = [var_5, var_8]
    var_10 = module_0.Config()
    var_11 = 'pattern1.module'
    var_12 = module_1._known_pattern(var_11, var_10)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'section1'
    var_1 = 'section2'
    var_2 = [var_0, var_1]
    var_3 = 'pattern1'
    var_4 = 'section1'
    var_5 = (var_3, var_4)
    var_6 = 'pattern2'
    var_7 = 'section2'
    var_8 = (var_6, var_7)
    var_9 = [var_5, var_8]
    var_10 = module_0.Config()
    var_11 = 'module.pattern1'
    var_12 = module_1._known_pattern(var_11, var_10)

import isort.settings as module_0
import isort.place as module_1

def test_case_0():
    var_0 = 'section1'
    var_1 = [var_0]
    var_2 = 'pattern1'
    var_3 = 'section1'
    var_4 = (var_2, var_3)
    var_5 = 'pattern2'
    var_6 = 'section2'
    var_7 = (var_5, var_6)
    var_8 = [var_4, var_7]
    var_9 = module_0.Config()
    var_10 = 'pattern2.module'
    var_11 = module_1._known_pattern(var_10, var_9)
    assert var_11 is None



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_src_path_is_module_true_when_valid. Retrieved 1/5 statements.
# Partially parsed test_src_path_is_module_false_when_name_mismatch. Retrieved 2/6 statements.
# Partially parsed test_src_path_is_module_false_when_not_dir. Retrieved 1/5 statements.
# Partially parsed test_src_path_is_module_false_when_case_mismatch. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'valid_module'

def test_case_0():
    var_0 = 'valid_module'
    var_1 = 'different_name'

def test_case_0():
    var_0 = 'file.txt'

def test_case_0():
    var_0 = 'Module'
    var_1 = 'module'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_is_namespace_package_with_no_init_file_and_no_other_files. Retrieved 3/11 statements.
# Partially parsed test_is_namespace_package_with_namespace_declaration_in_init_file. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)

def test_case_0():
    var_0 = 'py'
    var_1 = [var_0]
    var_2 = frozenset(var_1)



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_is_namespace_package_with_init_file_and_namespace_declaration. Retrieved 6/13 statements.
# Partially parsed test_is_namespace_package_with_init_file_and_no_namespace_declaration. Retrieved 6/13 statements.
# Partially parsed test_is_namespace_package_without_init_file_and_no_files. Retrieved 4/8 statements.
# Partially parsed test_is_namespace_package_without_init_file_and_with_files. Retrieved 5/13 statements.
# Partially parsed test_is_namespace_package_without_init_file_and_with_setup_cfg. Retrieved 5/13 statements.
# Partially parsed test_is_namespace_package_without_init_file_and_with_pyproject_toml. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 'test_package'
    var_1 = '__init__.py'
    var_2 = "__import__('pkg_resources').declare_namespace(__name__)"
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'test_package'
    var_1 = '__init__.py'
    var_2 = "print('hello world')"
    var_3 = 'py'
    var_4 = [var_3]
    var_5 = frozenset(var_4)

def test_case_0():
    var_0 = 'test_package'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)

def test_case_0():
    var_0 = 'test_package'
    var_1 = 'test.py'
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = 'test_package'
    var_1 = 'setup.cfg'
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)

def test_case_0():
    var_0 = 'test_package'
    var_1 = 'pyproject.toml'
    var_2 = 'py'
    var_3 = [var_2]
    var_4 = frozenset(var_3)



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_is_namespace_package_with_valid_namespace_package. Retrieved 11/22 statements.
# Partially parsed test_is_namespace_package_with_invalid_namespace_package. Retrieved 11/22 statements.
# Partially parsed test_is_namespace_package_with_non_package_directory. Retrieved 8/11 statements.
# Partially parsed test_is_namespace_package_with_init_file_but_not_namespace. Retrieved 9/16 statements.
# Partially parsed test_is_namespace_package_with_filenames_and_no_init_file. Retrieved 10/18 statements.


def test_case_0():
    var_0 = '/valid/namespace/package'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)
    var_4 = True
    var_5 = lambda p: var_4
    var_6 = lambda : var_4
    var_7 = 'file1.py'
    var_8 = False
    var_9 = lambda : var_8
    var_10 = b"__import__('pkg_resources').declare_namespace(__name__)"

def test_case_0():
    var_0 = '/invalid/namespace/package'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)
    var_4 = True
    var_5 = lambda p: var_4
    var_6 = lambda : var_4
    var_7 = 'file1.py'
    var_8 = False
    var_9 = lambda : var_8
    var_10 = b'invalid_content'

def test_case_0():
    var_0 = '/non/package/directory'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)
    var_4 = True
    var_5 = lambda p: var_4
    var_6 = False
    var_7 = lambda : var_6

def test_case_0():
    var_0 = '/package/with/init/file'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)
    var_4 = True
    var_5 = lambda p: var_4
    var_6 = lambda : var_4
    var_7 = lambda : var_4
    var_8 = b'invalid_content'

def test_case_0():
    var_0 = '/package/with/filenames'
    var_1 = 'py'
    var_2 = [var_1]
    var_3 = frozenset(var_2)
    var_4 = True
    var_5 = lambda p: var_4
    var_6 = lambda : var_4
    var_7 = 'file1.py'
    var_8 = False
    var_9 = lambda : var_8



# Parsed testcases at query #47
#--------------------------

# Partially parsed test__forced_separate_matches_exact_pattern. Retrieved 7/9 statements.
# Partially parsed test__forced_separate_matches_pattern_with_wildcard. Retrieved 8/10 statements.
# Partially parsed test__forced_separate_matches_dot_prefix. Retrieved 8/10 statements.
# Partially parsed test__forced_separate_no_match. Retrieved 8/10 statements.
# Partially parsed test__forced_separate_empty_config. Retrieved 7/9 statements.
# Partially parsed test__forced_separate_matches_first_pattern_in_list. Retrieved 8/10 statements.


def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'forced_separate'
    var_3 = 'exact'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = type(var_0, var_1, var_5)

def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'forced_separate'
    var_3 = 'prefix*'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = type(var_0, var_1, var_5)
    var_7 = 'prefix123'

def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'forced_separate'
    var_3 = 'hidden*'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = type(var_0, var_1, var_5)
    var_7 = '.hiddenfile'

def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'forced_separate'
    var_3 = 'nomatch'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = type(var_0, var_1, var_5)
    var_7 = 'other'

def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'forced_separate'
    var_3 = []
    var_4 = {var_2: var_3}
    var_5 = type(var_0, var_1, var_4)
    var_6 = 'any'

def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'forced_separate'
    var_3 = 'first'
    var_4 = 'second'
    var_5 = [var_3, var_4]
    var_6 = {var_2: var_5}
    var_7 = type(var_0, var_1, var_6)



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_src_path_predicate_evaluates_to_true. Retrieved 5/17 statements.


def test_case_0():
    var_0 = set()
    var_1 = False
    var_2 = '.py'
    var_3 = [var_2]
    var_4 = '/mock/path/module.py'
    var_5 = 'module'



