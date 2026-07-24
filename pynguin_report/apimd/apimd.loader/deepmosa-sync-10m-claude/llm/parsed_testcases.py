####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_gen_api_basic. Retrieved 20/32 statements.
# Partially parsed test_gen_api_dry_mode. Retrieved 18/28 statements.
# Partially parsed test_gen_api_multiple_modules. Retrieved 22/32 statements.
# Partially parsed test_gen_api_empty_content. Retrieved 18/29 statements.
# Partially parsed test_gen_api_with_pwd. Retrieved 23/37 statements.
# Partially parsed test_gen_api_custom_level. Retrieved 19/30 statements.
# Partially parsed test_gen_api_with_link_option. Retrieved 18/28 statements.
# Partially parsed test_gen_api_with_toc_option. Retrieved 18/28 statements.


def test_case_0():
    var_0 = 'Test gen_api with basic parameters.'
    var_1 = 'docs'
    var_2 = 'apimd.loader.isdir'
    var_3 = False
    var_4 = lambda x: var_3
    var_5 = 'apimd.loader.mkdir'
    var_6 = None
    var_7 = lambda x: var_6
    var_8 = 'apimd.loader.loader'
    var_9 = '# Test API\n'
    var_10 = lambda *args, **kwargs: var_9
    var_11 = 'apimd.loader._site_path'
    var_12 = '/fake/path'
    var_13 = lambda x: var_12
    var_14 = 'apimd.loader._write'
    var_15 = lambda path, doc: var_6
    var_16 = 'Test'
    var_17 = 'test_module'
    var_18 = {var_16: var_17}
    var_19 = True

def test_case_0():
    var_0 = 'Test gen_api with dry mode enabled.'
    var_1 = 'apimd.loader.isdir'
    var_2 = False
    var_3 = lambda x: var_2
    var_4 = 'apimd.loader.mkdir'
    var_5 = None
    var_6 = lambda x: var_5
    var_7 = 'apimd.loader.loader'
    var_8 = '# Module API\n'
    var_9 = lambda *args, **kwargs: var_8
    var_10 = 'apimd.loader._site_path'
    var_11 = '/fake/path'
    var_12 = lambda x: var_11
    var_13 = 'MyModule'
    var_14 = 'mymodule'
    var_15 = {var_13: var_14}
    var_16 = 'docs'
    var_17 = True

def test_case_0():
    var_0 = 'Test gen_api with multiple root modules.'
    var_1 = 'apimd.loader.isdir'
    var_2 = False
    var_3 = lambda x: var_2
    var_4 = 'apimd.loader.mkdir'
    var_5 = None
    var_6 = lambda x: var_5
    var_7 = 'apimd.loader.loader'
    var_8 = '# API\n'
    var_9 = lambda *args, **kwargs: var_8
    var_10 = 'apimd.loader._site_path'
    var_11 = '/fake/path'
    var_12 = lambda x: var_11
    var_13 = 'apimd.loader._write'
    var_14 = lambda path, doc: var_5
    var_15 = 'Module1'
    var_16 = 'Module2'
    var_17 = 'mod1'
    var_18 = 'mod2'
    var_19 = {var_15: var_17, var_16: var_18}
    var_20 = 'docs'
    var_21 = True

def test_case_0():
    var_0 = 'Test gen_api when loader returns empty content.'
    var_1 = 'apimd.loader.isdir'
    var_2 = False
    var_3 = lambda x: var_2
    var_4 = 'apimd.loader.mkdir'
    var_5 = None
    var_6 = lambda x: var_5
    var_7 = 'apimd.loader.loader'
    var_8 = '   \n\n   '
    var_9 = lambda *args, **kwargs: var_8
    var_10 = 'apimd.loader._site_path'
    var_11 = '/fake/path'
    var_12 = lambda x: var_11
    var_13 = 'Empty'
    var_14 = 'empty_mod'
    var_15 = {var_13: var_14}
    var_16 = 'docs'
    var_17 = True

def test_case_0():
    var_0 = 'Test gen_api with custom pwd parameter.'
    var_1 = []
    var_2 = 'apimd.loader.sys_path'
    var_3 = 'apimd.loader.isdir'
    var_4 = False
    var_5 = lambda x: var_4
    var_6 = 'apimd.loader.mkdir'
    var_7 = None
    var_8 = lambda x: var_7
    var_9 = 'apimd.loader.loader'
    var_10 = '# API\n'
    var_11 = lambda *args, **kwargs: var_10
    var_12 = 'apimd.loader._site_path'
    var_13 = '/fake/path'
    var_14 = lambda x: var_13
    var_15 = 'apimd.loader._write'
    var_16 = lambda path, doc: var_7
    var_17 = 'site-packages'
    var_18 = 'Test'
    var_19 = 'test'
    var_20 = {var_18: var_19}
    var_21 = 'docs'
    var_22 = True

def test_case_0():
    var_0 = 'Test gen_api with custom heading level.'
    var_1 = 'apimd.loader.isdir'
    var_2 = False
    var_3 = lambda x: var_2
    var_4 = 'apimd.loader.mkdir'
    var_5 = None
    var_6 = lambda x: var_5
    var_7 = 'apimd.loader.loader'
    var_8 = 'Content\n'
    var_9 = lambda *args, **kwargs: var_8
    var_10 = 'apimd.loader._site_path'
    var_11 = '/fake/path'
    var_12 = lambda x: var_11
    var_13 = 'Test'
    var_14 = 'test'
    var_15 = {var_13: var_14}
    var_16 = 'docs'
    var_17 = 2
    var_18 = True
    var_19 = '## Test API'

def test_case_0():
    var_0 = 'Test gen_api with link parameter.'
    var_1 = 'apimd.loader.isdir'
    var_2 = False
    var_3 = lambda x: var_2
    var_4 = 'apimd.loader.mkdir'
    var_5 = None
    var_6 = lambda x: var_5
    var_7 = 'apimd.loader.loader'
    var_8 = '# API\n'
    var_9 = lambda *args, **kwargs: var_8
    var_10 = 'apimd.loader._site_path'
    var_11 = '/fake/path'
    var_12 = lambda x: var_11
    var_13 = 'Test'
    var_14 = 'test'
    var_15 = {var_13: var_14}
    var_16 = 'docs'
    var_17 = True

def test_case_0():
    var_0 = 'Test gen_api with toc parameter.'
    var_1 = 'apimd.loader.isdir'
    var_2 = False
    var_3 = lambda x: var_2
    var_4 = 'apimd.loader.mkdir'
    var_5 = None
    var_6 = lambda x: var_5
    var_7 = 'apimd.loader.loader'
    var_8 = '# API\n'
    var_9 = lambda *args, **kwargs: var_8
    var_10 = 'apimd.loader._site_path'
    var_11 = '/fake/path'
    var_12 = lambda x: var_11
    var_13 = 'Test'
    var_14 = 'test'
    var_15 = {var_13: var_14}
    var_16 = 'docs'
    var_17 = True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_walk_packages. Retrieved 8/31 statements.
# Partially parsed test_walk_packages_empty_directory. Retrieved 2/8 statements.
# Partially parsed test_walk_packages_ignores_non_python_files. Retrieved 7/23 statements.
# Partially parsed test_walk_packages_pep561_stub_files. Retrieved 5/17 statements.


def test_case_0():
    var_0 = 'Test walk_packages function.'
    var_1 = 'testpkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'module1.py'
    var_5 = 'module2.pyi'
    var_6 = 'subpkg'
    var_7 = 'module3.py'
    var_8 = 'testpkg'
    var_9 = 'testpkg.module1'
    var_10 = 'testpkg.module2'
    var_11 = 'testpkg.subpkg'
    var_12 = 'testpkg.subpkg.module3'
    var_13 = bool(var_1)
    assert var_13 is True

def test_case_0():
    var_0 = 'Test walk_packages with empty directory.'
    var_1 = 'emptypkg'

def test_case_0():
    var_0 = 'Test walk_packages ignores non-Python files.'
    var_1 = 'testpkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'module.py'
    var_5 = 'readme.txt'
    var_6 = 'data.json'
    var_7 = 'testpkg'
    var_8 = 'testpkg.module'

def test_case_0():
    var_0 = 'Test walk_packages with PEP 561 stub files.'
    var_1 = 'testpkg'
    var_2 = '__init__.pyi'
    var_3 = ''
    var_4 = 'module.pyi'
    var_5 = 'testpkg'
    var_6 = 'testpkg.module'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_read_returns_file_content. Retrieved 2/6 statements.
# Partially parsed test_read_empty_file. Retrieved 2/6 statements.
# Partially parsed test_read_multiline_file. Retrieved 2/6 statements.
# Partially parsed test_read_file_with_special_characters. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'test_script.txt'
    var_1 = 'Hello, World!\nThis is a test file.'

def test_case_0():
    var_0 = 'empty.txt'
    var_1 = ''

def test_case_0():
    var_0 = 'multiline.txt'
    var_1 = 'line1\nline2\nline3\n'

def test_case_0():
    var_0 = 'special.txt'
    var_1 = "Special chars: !@#$%^&*()_+-=[]{}|;:',.<>?/\n"



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_gen_api_creates_directory. Retrieved 5/13 statements.
# Partially parsed test_gen_api_dry_mode_no_file_created. Retrieved 5/13 statements.
# Partially parsed test_gen_api_write_mode_creates_file. Retrieved 5/13 statements.
# Partially parsed test_gen_api_with_multiple_packages. Retrieved 7/16 statements.
# Partially parsed test_gen_api_returns_sequence. Retrieved 5/13 statements.
# Partially parsed test_gen_api_with_custom_level. Retrieved 6/14 statements.
# Partially parsed test_gen_api_with_custom_link_setting. Retrieved 6/14 statements.
# Partially parsed test_gen_api_with_toc_enabled. Retrieved 5/13 statements.
# Partially parsed test_gen_api_empty_root_names. Retrieved 3/9 statements.
# Partially parsed test_gen_api_file_content_contains_title. Retrieved 7/15 statements.


def test_case_0():
    var_0 = 'new_docs'
    var_1 = 'Test'
    var_2 = 'os'
    var_3 = {var_1: var_2}
    var_4 = True

def test_case_0():
    var_0 = 'docs'
    var_1 = 'Test'
    var_2 = 'os'
    var_3 = {var_1: var_2}
    var_4 = True

def test_case_0():
    var_0 = 'docs'
    var_1 = 'Test'
    var_2 = 'os'
    var_3 = {var_1: var_2}
    var_4 = False
    var_5 = 'os-api.md'

def test_case_0():
    var_0 = 'docs'
    var_1 = 'OS'
    var_2 = 'Sys'
    var_3 = 'os'
    var_4 = 'sys'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = True

def test_case_0():
    var_0 = 'docs'
    var_1 = 'Test'
    var_2 = 'os'
    var_3 = {var_1: var_2}
    var_4 = True

def test_case_0():
    var_0 = 'docs'
    var_1 = 'Test'
    var_2 = 'os'
    var_3 = {var_1: var_2}
    var_4 = 2
    var_5 = True

def test_case_0():
    var_0 = 'docs'
    var_1 = 'Test'
    var_2 = 'os'
    var_3 = {var_1: var_2}
    var_4 = False
    var_5 = True

def test_case_0():
    var_0 = 'docs'
    var_1 = 'Test'
    var_2 = 'os'
    var_3 = {var_1: var_2}
    var_4 = True

def test_case_0():
    var_0 = 'docs'
    var_1 = {}
    var_2 = True

def test_case_0():
    var_0 = 'docs'
    var_1 = 'MyAPI'
    var_2 = 'os'
    var_3 = {var_1: var_2}
    var_4 = 1
    var_5 = True
    var_6 = 'MyAPI API'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 2/7 statements.
# Partially parsed test_write_overwrites_existing_file. Retrieved 3/10 statements.
# Partially parsed test_write_handles_empty_string. Retrieved 2/7 statements.
# Partially parsed test_write_handles_multiline_content. Retrieved 2/7 statements.
# Partially parsed test_write_handles_special_characters. Retrieved 2/7 statements.
# Partially parsed test_write_handles_unicode_content. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Original content'
    var_2 = 'New content'

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = ''

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Line 1\nLine 2\nLine 3'

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = "Special chars: !@#$%^&*()_+-=[]{}|;:',.<>?/~`"

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Unicode: 你好世界 🌍 Привет'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_gen_api_creates_directory_when_prefix_not_exists. Retrieved 6/14 statements.


def test_case_0():
    var_0 = "Test that gen_api creates the prefix directory when it doesn't exist."
    var_1 = 'new_docs'
    var_2 = 'PYTHONPATH'
    var_3 = ''
    var_4 = {}
    var_5 = True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_loader_basic. Retrieved 5/17 statements.
# Partially parsed test_loader_with_toc. Retrieved 4/15 statements.
# Partially parsed test_loader_multiple_files. Retrieved 7/21 statements.
# Partially parsed test_loader_no_link. Retrieved 5/16 statements.
# Partially parsed test_loader_different_level. Retrieved 6/17 statements.
# Partially parsed test_loader_with_class. Retrieved 5/16 statements.
# Partially parsed test_loader_stub_file. Retrieved 5/16 statements.


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\ndef func(): pass'
    var_3 = True
    var_4 = False
    var_5 = 'test_pkg'

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\ndef func(): pass'
    var_3 = True
    var_4 = '**Table of contents:**'

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""'
    var_3 = 'module.py'
    var_4 = '"""Test module."""\ndef test_func(): pass'
    var_5 = True
    var_6 = False

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""'
    var_3 = False
    var_4 = 1

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""'
    var_3 = True
    var_4 = 2
    var_5 = False

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\nclass TestClass:\n    """Test class."""\n    pass'
    var_3 = True
    var_4 = False

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.pyi'
    var_2 = '"""Test package stub."""\ndef stub_func() -> int: ...'
    var_3 = True
    var_4 = False



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_site_path_with_valid_package. Retrieved 2/5 statements.
# Partially parsed test_site_path_with_standard_library_package. Retrieved 2/3 statements.
# Partially parsed test_site_path_returns_string. Retrieved 2/3 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = module_0._site_path(var_0)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_package_xyz_12345'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'sys'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = module_0._site_path(var_0)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'json'
    var_1 = module_0._site_path(var_0)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_write_predicate_false. Retrieved 2/12 statements.


def test_case_0():
    var_0 = 'test_write_predicate.txt'
    var_1 = 'test content'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_loader_basic. Retrieved 6/15 statements.
# Partially parsed test_loader_with_submodule. Retrieved 8/19 statements.
# Partially parsed test_loader_with_toc. Retrieved 5/12 statements.
# Partially parsed test_loader_with_level. Retrieved 6/13 statements.
# Partially parsed test_loader_without_link. Retrieved 6/13 statements.
# Partially parsed test_loader_with_class. Retrieved 6/13 statements.
# Partially parsed test_loader_empty_package. Retrieved 6/14 statements.
# Partially parsed test_loader_pyi_stub. Retrieved 6/13 statements.
# Partially parsed test_loader_multiple_files. Retrieved 8/17 statements.
# Partially parsed test_loader_with_constants. Retrieved 6/13 statements.
# Partially parsed test_loader_nested_packages. Retrieved 9/24 statements.


def test_case_0():
    var_0 = 'Test loader with a basic package structure.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\ndef foo(): pass'
    var_4 = True
    var_5 = False
    var_6 = 'test_pkg'
    var_7 = 'foo'

def test_case_0():
    var_0 = 'Test loader with nested modules.'
    var_1 = 'mylib'
    var_2 = '__init__.py'
    var_3 = '"""Main package."""\nVERSION = \'1.0\''
    var_4 = 'sub'
    var_5 = '"""Submodule."""\ndef bar(): pass'
    var_6 = True
    var_7 = False
    var_8 = 'mylib'
    var_9 = 'mylib.sub'

def test_case_0():
    var_0 = 'Test loader with table of contents enabled.'
    var_1 = 'docs_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Package with docs."""\ndef func1(): pass\ndef func2(): pass'
    var_4 = True
    var_5 = '**Table of contents:**'
    var_6 = 'func1'
    var_7 = 'func2'

def test_case_0():
    var_0 = 'Test loader with different heading levels.'
    var_1 = 'level_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Package."""\ndef test(): pass'
    var_4 = False
    var_5 = 2
    var_6 = '###'

def test_case_0():
    var_0 = 'Test loader with link disabled.'
    var_1 = 'nolink_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Package."""\ndef method(): pass'
    var_4 = False
    var_5 = 1
    var_6 = '<a id='

def test_case_0():
    var_0 = 'Test loader with class definitions.'
    var_1 = 'class_pkg'
    var_2 = '"""Package with class."""\nclass MyClass:\n    """A test class."""\n    def method(self): pass\n'
    var_3 = '__init__.py'
    var_4 = True
    var_5 = False
    var_6 = 'MyClass'
    var_7 = 'method'

def test_case_0():
    var_0 = 'Test loader with empty package.'
    var_1 = 'empty_pkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader with .pyi stub files.'
    var_1 = 'stub_pkg'
    var_2 = '__init__.pyi'
    var_3 = '"""Stub file."""\ndef stub_func(): ...'
    var_4 = True
    var_5 = False
    var_6 = 'stub_pkg'

def test_case_0():
    var_0 = 'Test loader with multiple Python files.'
    var_1 = 'multi_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Main module."""\ndef main(): pass'
    var_4 = 'utils.py'
    var_5 = '"""Utils module."""\ndef helper(): pass'
    var_6 = True
    var_7 = False
    var_8 = 'multi_pkg'
    var_9 = 'multi_pkg.utils'

def test_case_0():
    var_0 = 'Test loader with module constants.'
    var_1 = 'const_pkg'
    var_2 = '"""Package with constants."""\nMAX_VALUE = 100\nMIN_VALUE = 0\ndef process(): pass\n'
    var_3 = '__init__.py'
    var_4 = True
    var_5 = False
    var_6 = 'const_pkg'

def test_case_0():
    var_0 = 'Test loader with deeply nested packages.'
    var_1 = 'root_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Root."""\ndef root_func(): pass'
    var_4 = 'level1'
    var_5 = '"""Level 1."""\ndef func_l1(): pass'
    var_6 = 'level2'
    var_7 = '"""Level 2."""\ndef func_l2(): pass'
    var_8 = True
    var_9 = 'root_pkg'
    var_10 = 'root_pkg.level1'
    var_11 = 'root_pkg.level1.level2'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_loader_predicate_line_13_false. Retrieved 9/22 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = "Test that the predicate at line 13 evaluates to False when ext is not '.py'."
    var_1 = 'test_module'
    var_2 = '/path/test_module'
    var_3 = (var_1, var_2)
    var_4 = '/root'
    var_5 = '/pwd'
    var_6 = False
    var_7 = 1
    var_8 = module_0.loader(var_4, var_5, var_6, var_7, var_6)



# Parsed testcases at query #12
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 22 (for title, name in root_names.items()) evaluates to False when root_names is empty.'
    var_1 = {}
    var_2 = None
    var_3 = 'docs'
    var_4 = True
    var_5 = False
    var_6 = module_0.gen_api(var_1, var_2, prefix=var_3, link=var_4, level=var_4, toc=var_5, dry=var_4)
    var_7 = bool(var_6 == [])
    assert var_7 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_load_module_success. Retrieved 4/14 statements.
# Partially parsed test_load_module_builtin. Retrieved 4/14 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module.py'
    var_2 = 'def test_func():\n    """Test function."""\n    pass\n'
    var_3 = 'test_module'

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'nonexistent.module.path'
    var_2 = '/fake/path.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'
    var_2 = '/nonexistent/path/to/os.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'json_test.py'
    var_2 = '"""Test module."""\ndef func():\n    """Function."""\n    pass\n'
    var_3 = 'json_test'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_read_returns_file_contents. Retrieved 2/6 statements.
# Partially parsed test_read_returns_empty_string_for_empty_file. Retrieved 2/6 statements.
# Partially parsed test_read_returns_multiline_content. Retrieved 2/6 statements.
# Partially parsed test_read_preserves_whitespace. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'

def test_case_0():
    var_0 = 'empty.txt'
    var_1 = ''

def test_case_0():
    var_0 = 'multiline.txt'
    var_1 = 'Line 1\nLine 2\nLine 3'

def test_case_0():
    var_0 = 'whitespace.txt'
    var_1 = '  indented\n\ttabbed\n'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_read_returns_file_contents. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_gen_api_iterates_over_root_names. Retrieved 8/17 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 22 (for title, name in root_names.items()) evaluates to True.'
    var_1 = 'MyLib'
    var_2 = 'OtherLib'
    var_3 = 'mylib'
    var_4 = 'otherlib'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = True
    var_7 = module_0.gen_api(var_5, dry=var_6)



# Parsed testcases at query #17
#--------------------------




import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '/fake/path.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False



# Parsed testcases at query #18
#--------------------------




def test_case_0():
    var_0 = 'Test that the predicate at line 13 (ext == ".py") evaluates to False.'
    var_1 = '.pyi'
    var_2 = bool(not var_1 == '.py')
    assert var_2 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_loader_predicate_line_15_false. Retrieved 9/27 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 15 (if pure_py:) evaluates to False.'
    var_1 = 'test_package'
    var_2 = 'module.pyi'
    var_3 = 'def foo(): pass'
    var_4 = 'test_module'
    var_5 = 'module'
    var_6 = False
    var_7 = 1
    var_8 = module_0.loader(var_0, var_1, var_6, var_7, var_6)
    assert var_8 == 'compiled'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_load_module_success. Retrieved 6/22 statements.
# Partially parsed test_load_module_import_error. Retrieved 5/14 statements.
# Partially parsed test_load_module_invalid_spec. Retrieved 6/20 statements.
# Partially parsed test_load_module_no_loader. Retrieved 6/21 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module.py'
    var_1 = '"""Test module docstring."""\ndef test_func():\n    """Test function."""\n    pass\n'
    var_2 = []
    var_3 = '__import__'
    var_4 = module_0.Parser()
    var_5 = 'test_module'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module.py'
    var_1 = '"""Test module."""\n'
    var_2 = '__import__'
    var_3 = module_0.Parser()
    var_4 = 'test_module'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module.py'
    var_1 = '"""Test module."""\n'
    var_2 = '__import__'
    var_3 = 'importlib.util.spec_from_file_location'
    var_4 = module_0.Parser()
    var_5 = 'test_module'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module.py'
    var_1 = '"""Test module."""\n'
    var_2 = '__import__'
    var_3 = 'importlib.util.spec_from_file_location'
    var_4 = module_0.Parser()
    var_5 = 'test_module'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_load_module_predicate_false_when_loader_not_instance. Retrieved 6/14 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 9 evaluates to False when s.loader is not a Loader instance.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = '/fake/path/test_module.py'
    var_4 = []
    var_5 = module_1._load_module(var_2, var_3, var_1)
    assert var_5 is False



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_load_module_returns_false_when_loader_not_instance_of_loader. Retrieved 4/10 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '/fake/path.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '/fake/path.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '/fake/path.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_loader_predicate_pure_py_false. Retrieved 10/22 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 15 evaluates to False when pure_py is False.'
    var_1 = 'test_module'
    var_2 = '/fake/path'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = '/root'
    var_6 = '/pwd'
    var_7 = False
    var_8 = 1
    var_9 = module_0.loader(var_5, var_6, var_7, var_8, var_7)
    assert var_9 == 'compiled'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_gen_api_basic. Retrieved 14/32 statements.
# Partially parsed test_gen_api_dry_run. Retrieved 14/30 statements.
# Partially parsed test_gen_api_empty_doc. Retrieved 11/26 statements.
# Partially parsed test_gen_api_multiple_packages. Retrieved 18/35 statements.
# Partially parsed test_gen_api_with_level. Retrieved 14/29 statements.
# Partially parsed test_gen_api_underscore_to_dash. Retrieved 14/31 statements.
# Partially parsed test_gen_api_with_sys_path. Retrieved 1/11 statements.


def test_case_0():
    var_0 = 'Test gen_api with basic functionality.'
    var_1 = 'apimd.loader.loader'
    var_2 = 'apimd.loader._site_path'
    var_3 = 'apimd.loader.isdir'
    var_4 = 'apimd.loader.mkdir'
    var_5 = None
    var_6 = lambda x: var_5
    var_7 = {}
    var_8 = 'apimd.loader._write'
    var_9 = 'Test Package'
    var_10 = 'test_pkg'
    var_11 = {var_9: var_10}
    var_12 = 'docs'
    var_13 = False
    var_14 = '# Test Package API'
    var_15 = '## Module'

def test_case_0():
    var_0 = 'Test gen_api with dry run mode.'
    var_1 = 'apimd.loader.loader'
    var_2 = 'apimd.loader._site_path'
    var_3 = 'apimd.loader.isdir'
    var_4 = 'apimd.loader.mkdir'
    var_5 = None
    var_6 = lambda x: var_5
    var_7 = 'apimd.loader._write'
    var_8 = lambda p, d: var_5
    var_9 = 'My API'
    var_10 = 'myapi'
    var_11 = {var_9: var_10}
    var_12 = 'docs'
    var_13 = True
    var_14 = '# My API API'

def test_case_0():
    var_0 = 'Test gen_api when loader returns empty documentation.'
    var_1 = 'apimd.loader.loader'
    var_2 = 'apimd.loader._site_path'
    var_3 = 'apimd.loader.isdir'
    var_4 = 'apimd.loader.mkdir'
    var_5 = None
    var_6 = lambda x: var_5
    var_7 = 'Empty'
    var_8 = 'empty_pkg'
    var_9 = {var_7: var_8}
    var_10 = 'docs'

def test_case_0():
    var_0 = 'Test gen_api with multiple packages.'
    var_1 = 'count'
    var_2 = 0
    var_3 = {var_1: var_2}
    var_4 = 'apimd.loader.loader'
    var_5 = 'apimd.loader._site_path'
    var_6 = 'apimd.loader.isdir'
    var_7 = 'apimd.loader.mkdir'
    var_8 = None
    var_9 = lambda x: var_8
    var_10 = 'apimd.loader._write'
    var_11 = lambda p, d: var_8
    var_12 = 'Package A'
    var_13 = 'Package B'
    var_14 = 'pkg_a'
    var_15 = 'pkg_b'
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = 'docs'
    var_18 = var_3['count']
    assert var_18 == 2
    var_19 = '# Package A API'
    var_20 = '# Package B API'

def test_case_0():
    var_0 = 'Test gen_api with custom heading level.'
    var_1 = 'apimd.loader.loader'
    var_2 = 'apimd.loader._site_path'
    var_3 = 'apimd.loader.isdir'
    var_4 = 'apimd.loader.mkdir'
    var_5 = None
    var_6 = lambda x: var_5
    var_7 = 'apimd.loader._write'
    var_8 = lambda p, d: var_5
    var_9 = 'API'
    var_10 = 'mymodule'
    var_11 = {var_9: var_10}
    var_12 = 'docs'
    var_13 = 2
    var_14 = '## API API'

def test_case_0():
    var_0 = 'Test gen_api converts underscores to dashes in filenames.'
    var_1 = []
    var_2 = 'apimd.loader.loader'
    var_3 = 'apimd.loader._site_path'
    var_4 = 'apimd.loader.isdir'
    var_5 = 'apimd.loader.mkdir'
    var_6 = None
    var_7 = lambda x: var_6
    var_8 = 'apimd.loader._write'
    var_9 = 'My Package'
    var_10 = 'my_package'
    var_11 = {var_9: var_10}
    var_12 = 'docs'
    var_13 = len(var_1)
    assert var_13 == 1
    var_14 = 'my-package-api.md'
    var_15 = bool('my-package-api.md' in var_1[0])
    assert var_15 is True

def test_case_0():
    var_0 = 'Test gen_api appends pwd to sys.path.'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_loader_pure_py_false. Retrieved 6/16 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 15 evaluates to False when pure_py is False.'
    var_1 = '/root'
    var_2 = '/pwd'
    var_3 = False
    var_4 = 1
    var_5 = module_0.loader(var_1, var_2, var_3, var_4, var_3)
    assert var_5 == 'compiled_output'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_loader_basic. Retrieved 6/15 statements.
# Partially parsed test_loader_with_submodule. Retrieved 8/17 statements.
# Partially parsed test_loader_with_toc. Retrieved 5/12 statements.
# Partially parsed test_loader_with_level. Retrieved 6/13 statements.
# Partially parsed test_loader_without_link. Retrieved 6/13 statements.
# Partially parsed test_loader_with_class. Retrieved 6/13 statements.
# Partially parsed test_loader_empty_package. Retrieved 6/14 statements.
# Partially parsed test_loader_with_stub_file. Retrieved 6/13 statements.
# Partially parsed test_loader_nested_packages. Retrieved 8/19 statements.


def test_case_0():
    var_0 = 'Test loader function with basic package structure.'
    var_1 = 'testpkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\ndef func1(): pass'
    var_4 = True
    var_5 = False
    var_6 = 'testpkg'
    var_7 = 'func1'

def test_case_0():
    var_0 = 'Test loader with submodules.'
    var_1 = 'mypkg'
    var_2 = '__init__.py'
    var_3 = '"""Main package."""\nVAR = 1'
    var_4 = 'submod.py'
    var_5 = '"""Submodule."""\ndef subfunc(): pass'
    var_6 = True
    var_7 = False
    var_8 = 'mypkg'
    var_9 = 'subfunc'

def test_case_0():
    var_0 = 'Test loader with table of contents enabled.'
    var_1 = 'docpkg'
    var_2 = '__init__.py'
    var_3 = '"""Package with docs."""\ndef method(): pass'
    var_4 = True
    var_5 = 'Table of contents'
    var_6 = 'docpkg'

def test_case_0():
    var_0 = 'Test loader with different heading level.'
    var_1 = 'lvlpkg'
    var_2 = '__init__.py'
    var_3 = '"""Level test."""\ndef test(): pass'
    var_4 = False
    var_5 = 2
    var_6 = 'lvlpkg'
    var_7 = 'test'

def test_case_0():
    var_0 = 'Test loader with link disabled.'
    var_1 = 'nolinkpkg'
    var_2 = '__init__.py'
    var_3 = '"""No link package."""\ndef func(): pass'
    var_4 = False
    var_5 = 1
    var_6 = 'nolinkpkg'
    var_7 = 'func'

def test_case_0():
    var_0 = 'Test loader with class definitions.'
    var_1 = 'classpkg'
    var_2 = '__init__.py'
    var_3 = '"""Class package."""\nclass MyClass:\n    """A class."""\n    def method(self): pass'
    var_4 = True
    var_5 = False
    var_6 = 'classpkg'
    var_7 = 'MyClass'
    var_8 = 'method'

def test_case_0():
    var_0 = 'Test loader with minimal package.'
    var_1 = 'emptypkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader with .pyi stub files.'
    var_1 = 'stubpkg'
    var_2 = '__init__.pyi'
    var_3 = '"""Stub package."""\ndef stub_func() -> int: ...'
    var_4 = True
    var_5 = False
    var_6 = 'stubpkg'
    var_7 = 'stub_func'

def test_case_0():
    var_0 = 'Test loader with nested package structure.'
    var_1 = 'parentpkg'
    var_2 = '__init__.py'
    var_3 = '"""Parent package."""\ndef parent_func(): pass'
    var_4 = 'child'
    var_5 = '"""Child package."""\ndef child_func(): pass'
    var_6 = True
    var_7 = False
    var_8 = 'parentpkg'
    var_9 = 'parent_func'
    var_10 = 'child_func'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = bool(var_0 == var_1)
    assert var_2 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_write_creates_file_and_writes_content. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'
    var_2 = bool(var_0 == var_1)
    assert var_2 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/7 statements.
# Partially parsed test_write_overwrites_existing_file. Retrieved 4/10 statements.
# Partially parsed test_write_empty_string. Retrieved 3/7 statements.
# Partially parsed test_write_multiline_content. Retrieved 3/7 statements.
# Partially parsed test_write_special_characters. Retrieved 3/7 statements.
# Partially parsed test_write_unicode_content. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'old content'
    var_2 = 'new content'
    var_3 = 'utf-8'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = ''
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'line1\nline2\nline3'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = "Special chars: !@#$%^&*()_+-=[]{}|;:',.<>?/~`"
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Unicode: 你好世界 🌍 Ñoño'
    var_2 = 'utf-8'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_read_returns_file_contents. Retrieved 2/6 statements.
# Partially parsed test_read_returns_empty_string_for_empty_file. Retrieved 2/6 statements.
# Partially parsed test_read_returns_multiline_content. Retrieved 2/6 statements.
# Partially parsed test_read_preserves_whitespace. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'

def test_case_0():
    var_0 = 'empty.txt'
    var_1 = ''

def test_case_0():
    var_0 = 'multiline.txt'
    var_1 = 'Line 1\nLine 2\nLine 3'

def test_case_0():
    var_0 = 'whitespace.txt'
    var_1 = '  spaces  \n\ttabs\t\n'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_write_predicate_false. Retrieved 2/12 statements.


def test_case_0():
    var_0 = 'w+'
    var_1 = 'utf-8'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_write_predicate_evaluates_to_false. Retrieved 3/22 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = []
    var_2 = 'test content'
    var_3 = var_1[0][1]
    assert var_3 == 'w+'
    var_4 = bool(not var_1[0][1] != 'w+')
    assert var_4 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_read_returns_file_content. Retrieved 2/6 statements.
# Partially parsed test_read_empty_file. Retrieved 2/6 statements.
# Partially parsed test_read_multiline_file. Retrieved 2/6 statements.
# Partially parsed test_read_file_with_special_characters. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'test_script.txt'
    var_1 = 'Hello, World!\nThis is a test file.'

def test_case_0():
    var_0 = 'empty_script.txt'
    var_1 = ''

def test_case_0():
    var_0 = 'multiline_script.txt'
    var_1 = 'Line 1\nLine 2\nLine 3\n'

def test_case_0():
    var_0 = 'special_chars.txt'
    var_1 = 'Special chars: !@#$%^&*()\nUnicode: café'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = bool(var_0 == var_1)
    assert var_2 is True



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_read_file_opens_in_read_mode. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 'test content'
    var_1 = bool(var_0)
    assert var_1 is True



# Parsed testcases at query #36
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 22 (for title, name in root_names.items()) evaluates to False when root_names is empty.'
    var_1 = {}
    var_2 = None
    var_3 = '/tmp/test_docs'
    var_4 = True
    var_5 = False
    var_6 = module_0.gen_api(var_1, var_2, prefix=var_3, link=var_4, level=var_4, toc=var_5, dry=var_4)
    var_7 = bool(var_6 == [])
    assert var_7 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_gen_api_iterates_root_names. Retrieved 9/18 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 22 evaluates to True by iterating root_names.'
    var_1 = 'Title1'
    var_2 = 'Title2'
    var_3 = 'module1'
    var_4 = 'module2'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = True
    var_7 = module_0.gen_api(var_5, dry=var_6)
    var_8 = len(var_7)
    assert var_8 == 2
    var_9 = 'Title1 API'
    var_10 = bool('Title1 API' in var_7[0])
    assert var_10 is True
    var_11 = 'Title2 API'
    var_12 = bool('Title2 API' in var_7[1])
    assert var_12 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_gen_api_basic. Retrieved 13/24 statements.
# Partially parsed test_gen_api_multiple_packages. Retrieved 14/23 statements.
# Partially parsed test_gen_api_empty_doc. Retrieved 12/22 statements.
# Partially parsed test_gen_api_dry_mode. Retrieved 12/22 statements.
# Partially parsed test_gen_api_custom_level. Retrieved 13/21 statements.
# Partially parsed test_gen_api_creates_directory. Retrieved 14/22 statements.
# Partially parsed test_gen_api_with_pwd. Retrieved 14/21 statements.
# Partially parsed test_gen_api_underscore_to_hyphen. Retrieved 13/22 statements.
# Partially parsed test_gen_api_with_toc. Retrieved 12/21 statements.


def test_case_0():
    var_0 = 'Test gen_api with basic parameters.'
    var_1 = 'apimd.loader.loader'
    var_2 = '# Test Doc'
    var_3 = 'apimd.loader._site_path'
    var_4 = '/fake/path'
    var_5 = 'apimd.loader._write'
    var_6 = 'apimd.loader.isdir'
    var_7 = True
    var_8 = 'docs'
    var_9 = 'TestLib'
    var_10 = 'test_lib'
    var_11 = {var_9: var_10}
    var_12 = '# TestLib API'
    var_13 = False

def test_case_0():
    var_0 = 'Test gen_api with multiple packages.'
    var_1 = 'apimd.loader.loader'
    var_2 = '# Doc'
    var_3 = 'apimd.loader._site_path'
    var_4 = '/path'
    var_5 = 'apimd.loader._write'
    var_6 = 'apimd.loader.isdir'
    var_7 = True
    var_8 = 'Lib1'
    var_9 = 'Lib2'
    var_10 = 'lib1'
    var_11 = 'lib2'
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = 'docs'

def test_case_0():
    var_0 = 'Test gen_api when loader returns empty doc.'
    var_1 = 'apimd.loader.loader'
    var_2 = '   '
    var_3 = 'apimd.loader._site_path'
    var_4 = '/path'
    var_5 = 'apimd.loader._write'
    var_6 = 'apimd.loader.isdir'
    var_7 = True
    var_8 = 'TestLib'
    var_9 = 'test_lib'
    var_10 = {var_8: var_9}
    var_11 = 'docs'

def test_case_0():
    var_0 = 'Test gen_api with dry run mode.'
    var_1 = 'apimd.loader.loader'
    var_2 = '# Test'
    var_3 = 'apimd.loader._site_path'
    var_4 = '/path'
    var_5 = 'apimd.loader._write'
    var_6 = 'apimd.loader.isdir'
    var_7 = True
    var_8 = 'TestLib'
    var_9 = 'test_lib'
    var_10 = {var_8: var_9}
    var_11 = 'docs'

def test_case_0():
    var_0 = 'Test gen_api with custom heading level.'
    var_1 = 'apimd.loader.loader'
    var_2 = 'content'
    var_3 = 'apimd.loader._site_path'
    var_4 = '/path'
    var_5 = 'apimd.loader._write'
    var_6 = 'apimd.loader.isdir'
    var_7 = True
    var_8 = 'TestLib'
    var_9 = 'test_lib'
    var_10 = {var_8: var_9}
    var_11 = 'docs'
    var_12 = 3
    var_13 = '### TestLib API'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api creates prefix directory if not exists.'
    var_1 = 'apimd.loader.loader'
    var_2 = '# Doc'
    var_3 = 'apimd.loader._site_path'
    var_4 = '/path'
    var_5 = 'apimd.loader._write'
    var_6 = 'apimd.loader.isdir'
    var_7 = False
    var_8 = 'apimd.loader.mkdir'
    var_9 = 'TestLib'
    var_10 = 'test_lib'
    var_11 = {var_9: var_10}
    var_12 = 'new_docs'
    var_13 = module_0.gen_api(var_11, prefix=var_12)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api with custom pwd parameter.'
    var_1 = 'apimd.loader.loader'
    var_2 = '# Doc'
    var_3 = 'apimd.loader._site_path'
    var_4 = '/path'
    var_5 = 'apimd.loader._write'
    var_6 = 'apimd.loader.isdir'
    var_7 = True
    var_8 = 'apimd.loader.sys_path.append'
    var_9 = 'TestLib'
    var_10 = 'test_lib'
    var_11 = {var_9: var_10}
    var_12 = '/custom/path'
    var_13 = module_0.gen_api(var_11, var_12)

def test_case_0():
    var_0 = 'Test gen_api converts underscores to hyphens in filename.'
    var_1 = 'apimd.loader.loader'
    var_2 = '# Doc'
    var_3 = 'apimd.loader._site_path'
    var_4 = '/path'
    var_5 = 'apimd.loader._write'
    var_6 = 'apimd.loader.isdir'
    var_7 = True
    var_8 = 'TestLib'
    var_9 = 'test_lib_name'
    var_10 = {var_8: var_9}
    var_11 = 'docs'
    var_12 = 0
    var_13 = 'test-lib-name-api.md'

def test_case_0():
    var_0 = 'Test gen_api with table of contents enabled.'
    var_1 = 'apimd.loader.loader'
    var_2 = '# Doc'
    var_3 = 'apimd.loader._site_path'
    var_4 = '/path'
    var_5 = 'apimd.loader._write'
    var_6 = 'apimd.loader.isdir'
    var_7 = True
    var_8 = 'TestLib'
    var_9 = 'test_lib'
    var_10 = {var_8: var_9}
    var_11 = 'docs'

def test_case_0():
    var_0 = 'Test gen_api with link disabled.'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_gen_api_creates_directory_and_writes_files. Retrieved 14/34 statements.
# Partially parsed test_gen_api_dry_run_does_not_write_files. Retrieved 10/22 statements.
# Partially parsed test_gen_api_skips_empty_documentation. Retrieved 10/22 statements.
# Partially parsed test_gen_api_multiple_packages. Retrieved 12/24 statements.
# Partially parsed test_gen_api_adds_title_header. Retrieved 11/22 statements.
# Partially parsed test_gen_api_appends_to_sys_path. Retrieved 11/27 statements.
# Partially parsed test_gen_api_converts_underscores_in_filename. Retrieved 13/31 statements.


def test_case_0():
    var_0 = 'Test gen_api creates prefix directory and writes API documentation files.'
    var_1 = 'docs'
    var_2 = 'apimd.loader.loader'
    var_3 = 'apimd.loader._site_path'
    var_4 = 'apimd.loader.isdir'
    var_5 = False
    var_6 = lambda x: var_5
    var_7 = 'apimd.loader.mkdir'
    var_8 = True
    var_9 = 'apimd.loader.isfile'
    var_10 = lambda x: var_5
    var_11 = 'Test Package'
    var_12 = 'test_pkg'
    var_13 = {var_11: var_12}
    var_14 = '# Sample API'

def test_case_0():
    var_0 = 'Test gen_api with dry=True does not write files.'
    var_1 = 'apimd.loader.loader'
    var_2 = 'apimd.loader._site_path'
    var_3 = 'apimd.loader.isdir'
    var_4 = True
    var_5 = lambda x: var_4
    var_6 = 'Test Package'
    var_7 = 'test_pkg'
    var_8 = {var_6: var_7}
    var_9 = 'docs'
    var_10 = '# Sample API'

def test_case_0():
    var_0 = 'Test gen_api skips packages with empty documentation.'
    var_1 = 'apimd.loader.loader'
    var_2 = 'apimd.loader._site_path'
    var_3 = 'apimd.loader.isdir'
    var_4 = True
    var_5 = lambda x: var_4
    var_6 = 'Test Package'
    var_7 = 'test_pkg'
    var_8 = {var_6: var_7}
    var_9 = 'docs'

def test_case_0():
    var_0 = 'Test gen_api handles multiple root packages.'
    var_1 = 'apimd.loader.loader'
    var_2 = 'apimd.loader._site_path'
    var_3 = 'apimd.loader.isdir'
    var_4 = True
    var_5 = lambda x: var_4
    var_6 = 'Package One'
    var_7 = 'Package Two'
    var_8 = 'pkg1'
    var_9 = 'pkg2'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 'docs'

def test_case_0():
    var_0 = 'Test gen_api adds title header with correct level.'
    var_1 = 'apimd.loader.loader'
    var_2 = 'apimd.loader._site_path'
    var_3 = 'apimd.loader.isdir'
    var_4 = True
    var_5 = lambda x: var_4
    var_6 = 'My Title'
    var_7 = 'my_pkg'
    var_8 = {var_6: var_7}
    var_9 = 'docs'
    var_10 = 2
    var_11 = '## My Title API'
    var_12 = 'Sample content'

def test_case_0():
    var_0 = 'Test gen_api appends pwd to sys.path when provided.'
    var_1 = 'apimd.loader.loader'
    var_2 = 'apimd.loader._site_path'
    var_3 = 'apimd.loader.isdir'
    var_4 = True
    var_5 = lambda x: var_4
    var_6 = 'apimd.loader.sys_path'
    var_7 = 'Test'
    var_8 = 'test'
    var_9 = {var_7: var_8}
    var_10 = 'docs'

def test_case_0():
    var_0 = 'Test gen_api converts underscores to hyphens in output filename.'
    var_1 = 'apimd.loader.loader'
    var_2 = 'apimd.loader._site_path'
    var_3 = 'apimd.loader.isdir'
    var_4 = True
    var_5 = lambda x: var_4
    var_6 = 'apimd.loader._write'
    var_7 = 'docs'
    var_8 = 'Title'
    var_9 = 'my_test_pkg'
    var_10 = {var_8: var_9}
    var_11 = False
    var_12 = 'my-test-pkg-api.md'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_read_file_opens_in_read_mode. Retrieved 4/16 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'test content'
    var_2 = 'w'
    var_3 = 'r'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_gen_api_basic. Retrieved 21/32 statements.
# Partially parsed test_gen_api_empty_doc. Retrieved 21/32 statements.
# Partially parsed test_gen_api_multiple_modules. Retrieved 23/32 statements.
# Partially parsed test_gen_api_with_custom_prefix. Retrieved 19/32 statements.
# Partially parsed test_gen_api_with_level_and_toc. Retrieved 22/33 statements.
# Partially parsed test_gen_api_with_pwd. Retrieved 23/35 statements.
# Partially parsed test_gen_api_link_parameter. Retrieved 20/31 statements.
# Partially parsed test_gen_api_dry_mode. Retrieved 20/31 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api with basic parameters.'
    var_1 = 'docs'
    var_2 = 'apimd.loader.mkdir'
    var_3 = True
    var_4 = 'apimd.loader.isdir'
    var_5 = False
    var_6 = lambda x: var_5
    var_7 = 'apimd.loader.loader'
    var_8 = '# Test API\n\nContent'
    var_9 = lambda *args, **kwargs: var_8
    var_10 = 'apimd.loader._site_path'
    var_11 = '/fake/path'
    var_12 = lambda x: var_11
    var_13 = 'apimd.loader._write'
    var_14 = None
    var_15 = lambda path, doc: var_14
    var_16 = 'Test'
    var_17 = 'test_module'
    var_18 = {var_16: var_17}
    var_19 = module_0.gen_api(var_18, dry=var_3)
    var_20 = len(var_19)
    var_21 = bool(var_20 > 0)
    assert var_21 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api when loader returns empty documentation.'
    var_1 = 'docs'
    var_2 = 'apimd.loader.mkdir'
    var_3 = True
    var_4 = 'apimd.loader.isdir'
    var_5 = False
    var_6 = lambda x: var_5
    var_7 = 'apimd.loader.loader'
    var_8 = '   \n  \n'
    var_9 = lambda *args, **kwargs: var_8
    var_10 = 'apimd.loader._site_path'
    var_11 = '/fake/path'
    var_12 = lambda x: var_11
    var_13 = 'apimd.loader._write'
    var_14 = None
    var_15 = lambda path, doc: var_14
    var_16 = 'Test'
    var_17 = 'test_module'
    var_18 = {var_16: var_17}
    var_19 = module_0.gen_api(var_18, dry=var_3)
    var_20 = len(var_19)
    assert var_20 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api with multiple root modules.'
    var_1 = 'docs'
    var_2 = 'apimd.loader.mkdir'
    var_3 = True
    var_4 = 'apimd.loader.isdir'
    var_5 = False
    var_6 = lambda x: var_5
    var_7 = 'apimd.loader.loader'
    var_8 = '# API\n\nContent'
    var_9 = lambda *args, **kwargs: var_8
    var_10 = 'apimd.loader._site_path'
    var_11 = '/fake/path'
    var_12 = lambda x: var_11
    var_13 = 'apimd.loader._write'
    var_14 = None
    var_15 = lambda path, doc: var_14
    var_16 = 'Module1'
    var_17 = 'Module2'
    var_18 = 'mod1'
    var_19 = 'mod2'
    var_20 = {var_16: var_18, var_17: var_19}
    var_21 = module_0.gen_api(var_20, dry=var_3)
    var_22 = len(var_21)
    assert var_22 == 2

def test_case_0():
    var_0 = 'Test gen_api with custom prefix parameter.'
    var_1 = 'custom_docs'
    var_2 = 'apimd.loader.mkdir'
    var_3 = True
    var_4 = 'apimd.loader.isdir'
    var_5 = False
    var_6 = lambda x: var_5
    var_7 = 'apimd.loader.loader'
    var_8 = '# API\n\nContent'
    var_9 = lambda *args, **kwargs: var_8
    var_10 = 'apimd.loader._site_path'
    var_11 = '/fake/path'
    var_12 = lambda x: var_11
    var_13 = 'apimd.loader._write'
    var_14 = None
    var_15 = lambda path, doc: var_14
    var_16 = 'Test'
    var_17 = 'test_module'
    var_18 = {var_16: var_17}

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api with level and toc parameters.'
    var_1 = 'docs'
    var_2 = 'apimd.loader.mkdir'
    var_3 = True
    var_4 = 'apimd.loader.isdir'
    var_5 = False
    var_6 = lambda x: var_5
    var_7 = 'apimd.loader.loader'
    var_8 = '## API\n\nContent'
    var_9 = lambda *args, **kwargs: var_8
    var_10 = 'apimd.loader._site_path'
    var_11 = '/fake/path'
    var_12 = lambda x: var_11
    var_13 = 'apimd.loader._write'
    var_14 = None
    var_15 = lambda path, doc: var_14
    var_16 = 'Test'
    var_17 = 'test_module'
    var_18 = {var_16: var_17}
    var_19 = 2
    var_20 = module_0.gen_api(var_18, level=var_19, toc=var_3, dry=var_3)
    var_21 = len(var_20)
    var_22 = bool(var_21 > 0)
    assert var_22 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api with pwd parameter.'
    var_1 = 'docs'
    var_2 = 'apimd.loader.mkdir'
    var_3 = True
    var_4 = 'apimd.loader.isdir'
    var_5 = False
    var_6 = lambda x: var_5
    var_7 = 'apimd.loader.loader'
    var_8 = '# API\n\nContent'
    var_9 = lambda *args, **kwargs: var_8
    var_10 = 'apimd.loader._site_path'
    var_11 = '/fake/path'
    var_12 = lambda x: var_11
    var_13 = 'apimd.loader._write'
    var_14 = None
    var_15 = lambda path, doc: var_14
    var_16 = 'sys'
    var_17 = __import__(var_16)
    var_18 = 'Test'
    var_19 = 'test_module'
    var_20 = {var_18: var_19}
    var_21 = '/custom/path'
    var_22 = module_0.gen_api(var_20, var_21, dry=var_3)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api with link parameter.'
    var_1 = 'docs'
    var_2 = 'apimd.loader.mkdir'
    var_3 = True
    var_4 = 'apimd.loader.isdir'
    var_5 = False
    var_6 = lambda x: var_5
    var_7 = 'apimd.loader.loader'
    var_8 = '# API\n\nContent'
    var_9 = lambda *args, **kwargs: var_8
    var_10 = 'apimd.loader._site_path'
    var_11 = '/fake/path'
    var_12 = lambda x: var_11
    var_13 = 'apimd.loader._write'
    var_14 = None
    var_15 = lambda path, doc: var_14
    var_16 = 'Test'
    var_17 = 'test_module'
    var_18 = {var_16: var_17}
    var_19 = module_0.gen_api(var_18, link=var_5, dry=var_3)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api in dry mode does not write files.'
    var_1 = []
    var_2 = 'docs'
    var_3 = 'apimd.loader.mkdir'
    var_4 = True
    var_5 = 'apimd.loader.isdir'
    var_6 = False
    var_7 = lambda x: var_6
    var_8 = 'apimd.loader.loader'
    var_9 = '# API\n\nContent'
    var_10 = lambda *args, **kwargs: var_9
    var_11 = 'apimd.loader._site_path'
    var_12 = '/fake/path'
    var_13 = lambda x: var_12
    var_14 = 'apimd.loader._write'
    var_15 = 'Test'
    var_16 = 'test_module'
    var_17 = {var_15: var_16}
    var_18 = module_0.gen_api(var_17, dry=var_4)
    var_19 = len(var_1)
    assert var_19 == 0



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_gen_api_basic. Retrieved 8/19 statements.
# Partially parsed test_gen_api_dry_run. Retrieved 9/19 statements.
# Partially parsed test_gen_api_multiple_packages. Retrieved 12/21 statements.
# Partially parsed test_gen_api_empty_doc. Retrieved 6/16 statements.
# Partially parsed test_gen_api_with_pwd. Retrieved 7/20 statements.
# Partially parsed test_gen_api_filename_conversion. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'Test gen_api with basic parameters.'
    var_1 = 'docs'
    var_2 = 'Test Module'
    var_3 = 'test_module'
    var_4 = {var_2: var_3}
    var_5 = None
    var_6 = True
    var_7 = False
    var_8 = '# Test Module API'
    var_9 = '# Module'

def test_case_0():
    var_0 = 'Test gen_api with dry run enabled.'
    var_1 = 'docs'
    var_2 = 'My Package'
    var_3 = 'my_pkg'
    var_4 = {var_2: var_3}
    var_5 = None
    var_6 = False
    var_7 = 2
    var_8 = True
    var_9 = '## My Package API'

def test_case_0():
    var_0 = 'Test gen_api with multiple packages.'
    var_1 = 'docs'
    var_2 = '# Pkg1'
    var_3 = '# Pkg2'
    var_4 = '/path1'
    var_5 = '/path2'
    var_6 = 'Package 1'
    var_7 = 'Package 2'
    var_8 = 'pkg1'
    var_9 = 'pkg2'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = False
    var_12 = '# Package 1 API'
    var_13 = '# Package 2 API'

def test_case_0():
    var_0 = 'Test gen_api when loader returns empty documentation.'
    var_1 = 'docs'
    var_2 = 'Empty Pkg'
    var_3 = 'empty_pkg'
    var_4 = {var_2: var_3}
    var_5 = False

def test_case_0():
    var_0 = 'Test gen_api with custom pwd parameter.'
    var_1 = 'docs'
    var_2 = 'site-packages'
    var_3 = 'Test'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = False

def test_case_0():
    var_0 = 'Test gen_api converts underscores to hyphens in filenames.'
    var_1 = 'docs'
    var_2 = 'Test'
    var_3 = 'test_module_name'
    var_4 = {var_2: var_3}
    var_5 = False
    var_6 = 'test-module-name-api.md'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_loader_basic. Retrieved 7/18 statements.
# Partially parsed test_loader_with_toc. Retrieved 4/12 statements.
# Partially parsed test_loader_without_link. Retrieved 5/13 statements.
# Partially parsed test_loader_with_different_level. Retrieved 6/14 statements.
# Partially parsed test_loader_nested_packages. Retrieved 7/19 statements.
# Partially parsed test_loader_with_stub_file. Retrieved 5/13 statements.
# Partially parsed test_loader_empty_package. Retrieved 5/15 statements.
# Partially parsed test_loader_with_class_definition. Retrieved 5/13 statements.
# Partially parsed test_loader_all_options_enabled. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\ndef hello():\n    """Say hello."""\n    pass\n'
    var_3 = 'module.py'
    var_4 = '"""Test module."""\ndef world():\n    """Say world."""\n    pass\n'
    var_5 = True
    var_6 = False
    var_7 = 'test_pkg'
    var_8 = 'hello'
    var_9 = 'world'

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\ndef func1():\n    """Function 1."""\n    pass\n'
    var_3 = True
    var_4 = '**Table of contents:**'
    var_5 = 'test_pkg'

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\ndef test_func():\n    """Test function."""\n    pass\n'
    var_3 = False
    var_4 = 1
    var_5 = 'test_pkg'
    var_6 = '<a id='

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\ndef my_func():\n    """My function."""\n    pass\n'
    var_3 = True
    var_4 = 2
    var_5 = False
    var_6 = 'test_pkg'

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'subpkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\n'
    var_4 = '"""Sub package."""\ndef sub_func():\n    """Sub function."""\n    pass\n'
    var_5 = True
    var_6 = False
    var_7 = 'test_pkg'

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.pyi'
    var_2 = '"""Test package stub."""\ndef stub_func() -> None: ...\n'
    var_3 = True
    var_4 = False
    var_5 = 'test_pkg'

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Empty test package."""\n'
    var_3 = True
    var_4 = False

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\nclass TestClass:\n    """Test class."""\n    pass\n'
    var_3 = True
    var_4 = False
    var_5 = 'TestClass'
    var_6 = 'test_pkg'

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\ndef func():\n    """Function."""\n    pass\n'
    var_3 = True
    var_4 = 3
    var_5 = '**Table of contents:**'
    var_6 = 'test_pkg'
    var_7 = '<a id='



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_walk_packages. Retrieved 8/32 statements.
# Partially parsed test_walk_packages_with_pep561_suffix. Retrieved 5/19 statements.
# Partially parsed test_walk_packages_ignores_non_python_files. Retrieved 7/23 statements.
# Partially parsed test_walk_packages_empty_package. Retrieved 4/14 statements.
# Partially parsed test_walk_packages_nested_structure. Retrieved 7/26 statements.


def test_case_0():
    var_0 = 'Test walk_packages function.'
    var_1 = 'testpkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'module1.py'
    var_5 = 'module2.pyi'
    var_6 = 'subpkg'
    var_7 = 'submodule.py'
    var_8 = 'testpkg'
    var_9 = 'testpkg.module1'
    var_10 = 'testpkg.module2'
    var_11 = 'testpkg.subpkg'
    var_12 = 'testpkg.subpkg.submodule'
    var_13 = bool(var_1)
    assert var_13 is True
    var_14 = bool(var_2)
    assert var_14 is True

def test_case_0():
    var_0 = 'Test walk_packages with PEP 561 suffix.'
    var_1 = '__init__.py'
    var_2 = ''
    var_3 = 'module.py'
    var_4 = 'testpkg'
    var_5 = 'testpkg'
    var_6 = 'testpkg.module'

def test_case_0():
    var_0 = 'Test walk_packages ignores non-Python files.'
    var_1 = 'testpkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'module.py'
    var_5 = 'readme.txt'
    var_6 = 'data.json'
    var_7 = 'testpkg'
    var_8 = 'testpkg.module'

def test_case_0():
    var_0 = 'Test walk_packages with empty package.'
    var_1 = 'emptypkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'emptypkg'

def test_case_0():
    var_0 = 'Test walk_packages with deeply nested package structure.'
    var_1 = 'root'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'level1'
    var_5 = 'level2'
    var_6 = 'deep_module.py'
    var_7 = 'root'
    var_8 = 'root.level1'
    var_9 = 'root.level1.level2'
    var_10 = 'root.level1.level2.deep_module'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_loader_creates_parser_with_correct_options. Retrieved 6/15 statements.
# Partially parsed test_loader_parses_python_files. Retrieved 6/18 statements.
# Partially parsed test_loader_parses_stub_files. Retrieved 7/19 statements.
# Partially parsed test_loader_handles_multiple_packages. Retrieved 6/17 statements.
# Partially parsed test_loader_returns_compiled_documentation. Retrieved 6/13 statements.
# Partially parsed test_loader_skips_non_python_files. Retrieved 5/14 statements.
# Partially parsed test_loader_with_different_link_and_toc_options. Retrieved 6/14 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_root'
    var_1 = '/test/pwd'
    var_2 = True
    var_3 = 2
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)

import apimd.loader as module_0

def test_case_0():
    var_0 = '.py'
    var_1 = 'test'
    var_2 = '/pwd'
    var_3 = True
    var_4 = False
    var_5 = module_0.loader(var_1, var_2, var_3, var_3, var_4)
    assert var_5 == '# result'

import apimd.loader as module_0

def test_case_0():
    var_0 = '.pyi'
    var_1 = 'test'
    var_2 = '/pwd'
    var_3 = False
    var_4 = 1
    var_5 = True
    var_6 = module_0.loader(var_1, var_2, var_3, var_4, var_5)

import apimd.loader as module_0

def test_case_0():
    var_0 = '.py'
    var_1 = 'root'
    var_2 = '/pwd'
    var_3 = True
    var_4 = False
    var_5 = module_0.loader(var_1, var_2, var_3, var_3, var_4)

import apimd.loader as module_0

def test_case_0():
    var_0 = '# Generated Documentation'
    var_1 = 'test'
    var_2 = '/pwd'
    var_3 = True
    var_4 = False
    var_5 = module_0.loader(var_1, var_2, var_3, var_3, var_4)
    var_6 = bool(var_5 == var_0)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = '/pwd'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = '/pwd'
    var_2 = False
    var_3 = 3
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_load_module_success. Retrieved 5/13 statements.
# Partially parsed test_load_module_import_error. Retrieved 6/13 statements.
# Partially parsed test_load_module_invalid_spec. Retrieved 6/13 statements.
# Partially parsed test_load_module_calls_load_docstring. Retrieved 8/22 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test _load_module successfully loads a module and docstring.'
    var_1 = 'test_module.py'
    var_2 = '"""Test module docstring."""\ndef func(): pass'
    var_3 = module_0.Parser()
    var_4 = 'test_module'

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module returns False when parent import fails.'
    var_1 = 'test_module.py'
    var_2 = '"""Test module."""'
    var_3 = module_0.Parser()
    var_4 = 'nonexistent.module'
    var_5 = module_1._load_module(var_4, var_1, var_3)
    assert var_5 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module returns False when spec is invalid.'
    var_1 = 'test_module.py'
    var_2 = '"""Test module."""'
    var_3 = module_0.Parser()
    var_4 = 'test_module'
    var_5 = module_1._load_module(var_4, var_1, var_3)
    assert var_5 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module calls load_docstring on the parser.'
    var_1 = 'test_module.py'
    var_2 = '"""Module docstring."""'
    var_3 = module_0.Parser()
    var_4 = []
    var_5 = 'test_module'
    var_6 = module_1._load_module(var_5, var_1, var_3)
    assert var_6 is True
    var_7 = len(var_4)
    assert var_7 == 1
    var_8 = var_4[0][0]
    assert var_8 == 'test_module'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_gen_api_creates_directory_when_prefix_not_exists. Retrieved 6/13 statements.


def test_case_0():
    var_0 = "Test that gen_api creates the prefix directory if it doesn't exist."
    var_1 = 'new_docs'
    var_2 = 'TestModule'
    var_3 = 'os'
    var_4 = {var_2: var_3}
    var_5 = True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_site_path_existing_package. Retrieved 2/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = module_0._site_path(var_0)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_package_xyz_12345'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'sys'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'json'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_loader_basic. Retrieved 6/15 statements.
# Partially parsed test_loader_with_submodule. Retrieved 8/18 statements.
# Partially parsed test_loader_with_toc. Retrieved 5/13 statements.
# Partially parsed test_loader_different_levels. Retrieved 7/15 statements.
# Partially parsed test_loader_without_link. Retrieved 6/14 statements.
# Partially parsed test_loader_with_class. Retrieved 6/14 statements.
# Partially parsed test_loader_with_stub_file. Retrieved 6/14 statements.
# Partially parsed test_loader_empty_package. Retrieved 6/15 statements.
# Partially parsed test_loader_nested_packages. Retrieved 8/20 statements.
# Partially parsed test_loader_with_docstring. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'Test loader with basic package structure.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = 'def hello(): pass'
    var_4 = True
    var_5 = False
    var_6 = 'test_pkg'
    var_7 = 'hello'

def test_case_0():
    var_0 = 'Test loader with submodules.'
    var_1 = 'myapp'
    var_2 = '__init__.py'
    var_3 = 'x = 1'
    var_4 = 'utils.py'
    var_5 = 'def util_func(): pass'
    var_6 = False
    var_7 = 1
    var_8 = 'myapp'
    var_9 = 'util_func'

def test_case_0():
    var_0 = 'Test loader with table of contents enabled.'
    var_1 = 'docs_pkg'
    var_2 = '__init__.py'
    var_3 = 'def func1(): pass'
    var_4 = True
    var_5 = '**Table of contents:**'
    var_6 = 'docs_pkg'

def test_case_0():
    var_0 = 'Test loader with different heading levels.'
    var_1 = 'level_pkg'
    var_2 = '__init__.py'
    var_3 = 'def test(): pass'
    var_4 = True
    var_5 = 2
    var_6 = False
    var_7 = 'level_pkg'

def test_case_0():
    var_0 = 'Test loader with link disabled.'
    var_1 = 'nolink_pkg'
    var_2 = '__init__.py'
    var_3 = 'def foo(): pass'
    var_4 = False
    var_5 = 1
    var_6 = 'nolink_pkg'
    var_7 = '<a id='

def test_case_0():
    var_0 = 'Test loader with class definitions.'
    var_1 = 'class_pkg'
    var_2 = '__init__.py'
    var_3 = 'class MyClass:\n    def method(self): pass'
    var_4 = True
    var_5 = False
    var_6 = 'class_pkg'
    var_7 = 'MyClass'

def test_case_0():
    var_0 = 'Test loader prefers .pyi stub files.'
    var_1 = 'stub_pkg'
    var_2 = '__init__.pyi'
    var_3 = 'def stub_func() -> int: ...'
    var_4 = True
    var_5 = False
    var_6 = 'stub_pkg'
    var_7 = 'stub_func'

def test_case_0():
    var_0 = 'Test loader with empty package.'
    var_1 = 'empty_pkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader with nested package structure.'
    var_1 = 'nested'
    var_2 = '__init__.py'
    var_3 = 'x = 1'
    var_4 = 'sub'
    var_5 = 'def nested_func(): pass'
    var_6 = True
    var_7 = False
    var_8 = 'nested'
    var_9 = 'nested_func'

def test_case_0():
    var_0 = 'Test loader preserves docstrings.'
    var_1 = 'doc_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Module docstring."""\ndef func(): """Function doc.""" pass'
    var_4 = True
    var_5 = False
    var_6 = 'doc_pkg'
    var_7 = 'func'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/7 statements.
# Partially parsed test_write_overwrites_existing_file. Retrieved 4/10 statements.
# Partially parsed test_write_empty_string. Retrieved 3/7 statements.
# Partially parsed test_write_multiline_content. Retrieved 3/7 statements.
# Partially parsed test_write_unicode_content. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'old content'
    var_2 = 'new content'
    var_3 = 'utf-8'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = ''
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'line1\nline2\nline3'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello 世界 🌍'
    var_2 = 'utf-8'



# Parsed testcases at query #10
#--------------------------




def test_case_0():
    var_0 = "Test that the predicate at line 13 evaluates to False when ext is not '.py'."
    var_1 = '.pyi'
    assert var_1 == '.py'
    var_2 = '.py'
    assert var_2 is False



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_loader_predicate_pure_py_false. Retrieved 11/27 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Ensure that the predicate at line 15 (if pure_py:) evaluates to False.'
    var_1 = 'test_module'
    var_2 = '/fake/path'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = '/root'
    var_6 = '/pwd'
    var_7 = True
    var_8 = module_0.loader(var_5, var_6, var_7, var_7, var_7)
    var_9 = 0
    var_10 = '.pyi'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_gen_api_basic. Retrieved 20/32 statements.
# Partially parsed test_gen_api_empty_doc. Retrieved 20/32 statements.
# Partially parsed test_gen_api_multiple_modules. Retrieved 23/35 statements.
# Partially parsed test_gen_api_dry_mode. Retrieved 22/32 statements.
# Partially parsed test_gen_api_with_pwd. Retrieved 23/35 statements.
# Partially parsed test_gen_api_level_parameter. Retrieved 21/32 statements.
# Partially parsed test_gen_api_underscore_to_dash. Retrieved 21/32 statements.


def test_case_0():
    var_0 = 'Test gen_api with basic functionality.'
    var_1 = 'docs'
    var_2 = 'apimd.loader.isdir'
    var_3 = False
    var_4 = lambda x: var_3
    var_5 = 'apimd.loader.mkdir'
    var_6 = None
    var_7 = lambda x: var_6
    var_8 = 'apimd.loader.loader'
    var_9 = '# Module docs\n'
    var_10 = lambda *args: var_9
    var_11 = 'apimd.loader._site_path'
    var_12 = '/fake/path'
    var_13 = lambda x: var_12
    var_14 = 'apimd.loader._write'
    var_15 = lambda *args: var_6
    var_16 = 'Test'
    var_17 = 'test_module'
    var_18 = {var_16: var_17}
    var_19 = True
    var_20 = 'Test API'

def test_case_0():
    var_0 = 'Test gen_api when loader returns empty documentation.'
    var_1 = 'docs'
    var_2 = 'apimd.loader.isdir'
    var_3 = False
    var_4 = lambda x: var_3
    var_5 = 'apimd.loader.mkdir'
    var_6 = None
    var_7 = lambda x: var_6
    var_8 = 'apimd.loader.loader'
    var_9 = '   \n  '
    var_10 = lambda *args: var_9
    var_11 = 'apimd.loader._site_path'
    var_12 = '/fake/path'
    var_13 = lambda x: var_12
    var_14 = 'apimd.loader._write'
    var_15 = lambda *args: var_6
    var_16 = 'Test'
    var_17 = 'test_module'
    var_18 = {var_16: var_17}
    var_19 = True

def test_case_0():
    var_0 = 'Test gen_api with multiple root modules.'
    var_1 = 'docs'
    var_2 = 'apimd.loader.isdir'
    var_3 = False
    var_4 = lambda x: var_3
    var_5 = 'apimd.loader.mkdir'
    var_6 = None
    var_7 = lambda x: var_6
    var_8 = 'apimd.loader.loader'
    var_9 = '# Module docs\n'
    var_10 = lambda *args: var_9
    var_11 = 'apimd.loader._site_path'
    var_12 = '/fake/path'
    var_13 = lambda x: var_12
    var_14 = 'apimd.loader._write'
    var_15 = lambda *args: var_6
    var_16 = 'Module A'
    var_17 = 'Module B'
    var_18 = 'module_a'
    var_19 = 'module_b'
    var_20 = {var_16: var_18, var_17: var_19}
    var_21 = True
    var_22 = 2
    var_23 = 'Module A API'
    var_24 = 'Module B API'

def test_case_0():
    var_0 = 'Test gen_api in dry mode.'
    var_1 = 'docs'
    var_2 = []
    var_3 = 'apimd.loader.isdir'
    var_4 = False
    var_5 = lambda x: var_4
    var_6 = 'apimd.loader.mkdir'
    var_7 = None
    var_8 = lambda x: var_7
    var_9 = 'apimd.loader.loader'
    var_10 = '# Module docs\n'
    var_11 = lambda *args: var_10
    var_12 = 'apimd.loader._site_path'
    var_13 = '/fake/path'
    var_14 = lambda x: var_13
    var_15 = 'apimd.loader._write'
    var_16 = lambda *args: var_2.append(args)
    var_17 = 'Test'
    var_18 = 'test_module'
    var_19 = {var_17: var_18}
    var_20 = True
    var_21 = len(var_2)
    assert var_21 == 0

def test_case_0():
    var_0 = 'Test gen_api with pwd parameter.'
    var_1 = 'docs'
    var_2 = []
    var_3 = 'apimd.loader.isdir'
    var_4 = False
    var_5 = lambda x: var_4
    var_6 = 'apimd.loader.mkdir'
    var_7 = None
    var_8 = lambda x: var_7
    var_9 = 'apimd.loader.loader'
    var_10 = '# Module docs\n'
    var_11 = lambda *args: var_10
    var_12 = 'apimd.loader._site_path'
    var_13 = '/fake/path'
    var_14 = lambda x: var_13
    var_15 = 'apimd.loader._write'
    var_16 = lambda *args: var_7
    var_17 = 'apimd.loader.sys_path'
    var_18 = 'Test'
    var_19 = 'test_module'
    var_20 = {var_18: var_19}
    var_21 = '/custom/path'
    var_22 = True
    var_23 = '/custom/path'
    var_24 = bool('/custom/path' in var_2)
    assert var_24 is True

def test_case_0():
    var_0 = 'Test gen_api respects level parameter.'
    var_1 = 'docs'
    var_2 = 'apimd.loader.isdir'
    var_3 = False
    var_4 = lambda x: var_3
    var_5 = 'apimd.loader.mkdir'
    var_6 = None
    var_7 = lambda x: var_6
    var_8 = 'apimd.loader.loader'
    var_9 = '# Module docs\n'
    var_10 = lambda *args: var_9
    var_11 = 'apimd.loader._site_path'
    var_12 = '/fake/path'
    var_13 = lambda x: var_12
    var_14 = 'apimd.loader._write'
    var_15 = lambda *args: var_6
    var_16 = 'Test'
    var_17 = 'test_module'
    var_18 = {var_16: var_17}
    var_19 = True
    var_20 = 3
    var_21 = '### Test API'

def test_case_0():
    var_0 = 'Test gen_api converts underscores to dashes in filename.'
    var_1 = 'docs'
    var_2 = []
    var_3 = 'apimd.loader.isdir'
    var_4 = False
    var_5 = lambda x: var_4
    var_6 = 'apimd.loader.mkdir'
    var_7 = None
    var_8 = lambda x: var_7
    var_9 = 'apimd.loader.loader'
    var_10 = '# Module docs\n'
    var_11 = lambda *args: var_10
    var_12 = 'apimd.loader._site_path'
    var_13 = '/fake/path'
    var_14 = lambda x: var_13
    var_15 = 'apimd.loader._write'
    var_16 = 'Test'
    var_17 = 'test_module_name'
    var_18 = {var_16: var_17}
    var_19 = True
    var_20 = len(var_2)
    assert var_20 == 1
    var_21 = 'test-module-name-api.md'
    var_22 = bool('test-module-name-api.md' in var_2[0])
    assert var_22 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_load_module_success. Retrieved 11/25 statements.
# Partially parsed test_load_module_invalid_spec. Retrieved 10/21 statements.
# Partially parsed test_load_module_calls_load_docstring. Retrieved 13/27 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module successfully loads a module.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'test_module.py'
    var_5 = "def test_func():\n    '''Test function'''\n    pass\n"
    var_6 = 0
    var_7 = module_0.Parser()
    var_8 = 'test_pkg.test_module'
    var_9 = module_1._load_module(var_8, var_1, var_7)
    assert var_9 is True
    var_10 = 0

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module returns False when parent import fails.'
    var_1 = module_0.Parser()
    var_2 = 'nonexistent_pkg.module'
    var_3 = '/fake/path.py'
    var_4 = module_1._load_module(var_2, var_3, var_1)
    assert var_4 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module returns False when spec is None.'
    var_1 = 'test_pkg2'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 0
    var_5 = module_0.Parser()
    var_6 = 'test_pkg2.nonexistent'
    var_7 = '/nonexistent/path.py'
    var_8 = module_1._load_module(var_6, var_7, var_5)
    assert var_8 is False
    var_9 = 0

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module calls parser.load_docstring.'
    var_1 = 'test_pkg3'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'test_module.py'
    var_5 = "'''Module docstring'''\ndef test_func():\n    '''Test function'''\n    pass\n"
    var_6 = 0
    var_7 = module_0.Parser()
    var_8 = 'test_pkg3.test_module'
    var_9 = "'''Module docstring'''"
    var_10 = var_7.parse(var_8, var_9)
    var_11 = module_1._load_module(var_8, var_2, var_7)
    assert var_11 is True
    var_12 = 'test_pkg3.test_module'
    var_13 = bool('test_pkg3.test_module' in var_7.docstring)
    assert var_13 is True
    var_14 = 0



# Parsed testcases at query #14
#--------------------------




def test_case_0():
    var_0 = 'Test that the predicate at line 13 (ext == ".py") evaluates to False.'
    var_1 = '.pyi'
    var_2 = bool(not var_1 == '.py')
    assert var_2 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_write_creates_file_with_correct_content. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_loader_basic. Retrieved 6/15 statements.
# Partially parsed test_loader_with_nested_modules. Retrieved 8/20 statements.
# Partially parsed test_loader_with_toc. Retrieved 5/13 statements.
# Partially parsed test_loader_without_link. Retrieved 6/14 statements.
# Partially parsed test_loader_with_different_heading_level. Retrieved 7/15 statements.
# Partially parsed test_loader_with_stub_file. Retrieved 6/14 statements.
# Partially parsed test_loader_empty_package. Retrieved 6/15 statements.
# Partially parsed test_loader_with_multiple_modules. Retrieved 10/22 statements.


def test_case_0():
    var_0 = 'Test loader with a simple package structure.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = "'''Test package.'''\ndef test_func():\n    '''Test function.'''\n    pass\n"
    var_4 = True
    var_5 = False
    var_6 = 'test_pkg'
    var_7 = 'test_func'

def test_case_0():
    var_0 = 'Test loader with nested module structure.'
    var_1 = 'my_pkg'
    var_2 = '__init__.py'
    var_3 = "'''Main package.'''\n"
    var_4 = 'sub'
    var_5 = "'''Sub package.'''\nclass MyClass:\n    '''A test class.'''\n    pass\n"
    var_6 = True
    var_7 = False
    var_8 = 'my_pkg'
    var_9 = 'MyClass'

def test_case_0():
    var_0 = 'Test loader with table of contents enabled.'
    var_1 = 'doc_pkg'
    var_2 = '__init__.py'
    var_3 = "'''Package with docs.'''\ndef func1():\n    '''Function 1.'''\n    pass\n"
    var_4 = True
    var_5 = 'Table of contents'
    var_6 = 'doc_pkg'

def test_case_0():
    var_0 = 'Test loader with link parameter set to False.'
    var_1 = 'nolink_pkg'
    var_2 = '__init__.py'
    var_3 = "'''Package without links.'''\ndef my_func():\n    '''My function.'''\n    pass\n"
    var_4 = False
    var_5 = 1
    var_6 = 'nolink_pkg'
    var_7 = 'my_func'

def test_case_0():
    var_0 = 'Test loader with different heading level.'
    var_1 = 'level_pkg'
    var_2 = '__init__.py'
    var_3 = "'''Package at level 2.'''\ndef level_func():\n    '''Function at level 2.'''\n    pass\n"
    var_4 = True
    var_5 = 2
    var_6 = False
    var_7 = 'level_pkg'
    var_8 = 'level_func'

def test_case_0():
    var_0 = 'Test loader with .pyi stub file.'
    var_1 = 'stub_pkg'
    var_2 = '__init__.pyi'
    var_3 = "'''Stub package.'''\ndef stub_func() -> None: ...\n"
    var_4 = True
    var_5 = False
    var_6 = 'stub_pkg'
    var_7 = 'stub_func'

def test_case_0():
    var_0 = 'Test loader with empty package.'
    var_1 = 'empty_pkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader with multiple modules in package.'
    var_1 = 'multi_pkg'
    var_2 = '__init__.py'
    var_3 = "'''Multi module package.'''\n"
    var_4 = 'module1.py'
    var_5 = "'''Module 1.'''\ndef func_a():\n    '''Function A.'''\n    pass\n"
    var_6 = 'module2.py'
    var_7 = "'''Module 2.'''\ndef func_b():\n    '''Function B.'''\n    pass\n"
    var_8 = True
    var_9 = False
    var_10 = 'multi_pkg'
    var_11 = 'func_a'
    var_12 = 'func_b'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_read_returns_file_content. Retrieved 2/6 statements.
# Partially parsed test_read_returns_empty_string_for_empty_file. Retrieved 2/6 statements.
# Partially parsed test_read_returns_multiline_content. Retrieved 2/6 statements.
# Partially parsed test_read_preserves_whitespace_and_special_characters. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'

def test_case_0():
    var_0 = 'empty.txt'
    var_1 = ''

def test_case_0():
    var_0 = 'multiline.txt'
    var_1 = 'Line 1\nLine 2\nLine 3'

def test_case_0():
    var_0 = 'special.txt'
    var_1 = '  spaces  \n\ttabs\t\n!@#$%^&*()'



# Parsed testcases at query #18
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 22 (for title, name in root_names.items()) evaluates to False.'
    var_1 = {}
    var_2 = '/tmp/test_docs'
    var_3 = True
    var_4 = module_0.gen_api(var_1, prefix=var_2, dry=var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 2/7 statements.
# Partially parsed test_write_overwrites_existing_file. Retrieved 3/10 statements.
# Partially parsed test_write_empty_string. Retrieved 2/7 statements.
# Partially parsed test_write_multiline_content. Retrieved 2/7 statements.
# Partially parsed test_write_unicode_content. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Original content'
    var_2 = 'New content'

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = ''

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Line 1\nLine 2\nLine 3'

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello 世界 🌍 Привет'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_read_file_opens_in_read_mode. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 'test content'
    var_1 = bool(var_0)
    assert var_1 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_loader_predicate_pure_py_false. Retrieved 10/26 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 15 evaluates to False when pure_py is False.'
    var_1 = 'test_module'
    var_2 = '/fake/path'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = '/fake/root'
    var_6 = '/fake/pwd'
    var_7 = False
    var_8 = 1
    var_9 = module_0.loader(var_5, var_6, var_7, var_8, var_7)
    assert var_9 == 'compiled'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_gen_api_basic. Retrieved 8/18 statements.
# Partially parsed test_gen_api_multiple_roots. Retrieved 10/20 statements.
# Partially parsed test_gen_api_empty_doc. Retrieved 6/16 statements.
# Partially parsed test_gen_api_creates_directory. Retrieved 6/15 statements.
# Partially parsed test_gen_api_with_pwd. Retrieved 8/21 statements.
# Partially parsed test_gen_api_write_file. Retrieved 6/17 statements.
# Partially parsed test_gen_api_with_level_parameter. Retrieved 7/16 statements.
# Partially parsed test_gen_api_with_link_parameter. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'Test gen_api with basic parameters.'
    var_1 = 'docs'
    var_2 = 'Test'
    var_3 = 'test_module'
    var_4 = {var_2: var_3}
    var_5 = None
    var_6 = True
    var_7 = False
    var_8 = '# Test API'
    var_9 = '## Module'

def test_case_0():
    var_0 = 'Test gen_api with multiple root modules.'
    var_1 = 'docs'
    var_2 = '## Module A'
    var_3 = '## Module B'
    var_4 = 'API A'
    var_5 = 'API B'
    var_6 = 'module_a'
    var_7 = 'module_b'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = True
    var_10 = '# API A API'
    var_11 = '# API B API'

def test_case_0():
    var_0 = 'Test gen_api when loader returns empty documentation.'
    var_1 = 'docs'
    var_2 = 'Empty'
    var_3 = 'empty_module'
    var_4 = {var_2: var_3}
    var_5 = True

def test_case_0():
    var_0 = "Test gen_api creates prefix directory if it doesn't exist."
    var_1 = 'docs'
    var_2 = 'Test'
    var_3 = 'test_module'
    var_4 = {var_2: var_3}
    var_5 = True

def test_case_0():
    var_0 = 'Test gen_api with custom pwd parameter.'
    var_1 = 'docs'
    var_2 = 'site_packages'
    var_3 = 'append'
    var_4 = 'Test'
    var_5 = 'test_module'
    var_6 = {var_4: var_5}
    var_7 = True

def test_case_0():
    var_0 = 'Test gen_api writes file when dry=False.'
    var_1 = 'docs'
    var_2 = 'Test'
    var_3 = 'test_module'
    var_4 = {var_2: var_3}
    var_5 = False

def test_case_0():
    var_0 = 'Test gen_api with different heading level.'
    var_1 = 'docs'
    var_2 = 'Test'
    var_3 = 'test_module'
    var_4 = {var_2: var_3}
    var_5 = 2
    var_6 = True
    var_7 = '## Test API'

def test_case_0():
    var_0 = 'Test gen_api passes link parameter to loader.'
    var_1 = 'docs'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_gen_api_predicate_line_25_evaluates_to_true. Retrieved 11/27 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 25 (not doc.strip()) evaluates to True when doc is empty/whitespace.'
    var_1 = 'apimd.loader.loader'
    var_2 = 'apimd.loader.isdir'
    var_3 = 'apimd.loader._site_path'
    var_4 = 'apimd.loader.logger.info'
    var_5 = 'apimd.loader.logger.warning'
    var_6 = 'Test'
    var_7 = 'test_module'
    var_8 = {var_6: var_7}
    var_9 = '/tmp/test_docs'
    var_10 = module_0.gen_api(var_8, prefix=var_9)
    var_11 = bool(var_10 == [])
    assert var_11 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_gen_api_predicate_line_25_evaluates_to_true. Retrieved 10/27 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 25 (not doc.strip()) evaluates to True when doc is empty/whitespace.'
    var_1 = 'apimd.loader.loader'
    var_2 = 'apimd.loader._site_path'
    var_3 = 'apimd.loader.isdir'
    var_4 = 'apimd.loader.logger'
    var_5 = 'Test'
    var_6 = 'test_module'
    var_7 = {var_5: var_6}
    var_8 = 'docs'
    var_9 = module_0.gen_api(var_7, prefix=var_8)
    var_10 = bool(var_9 == [])
    assert var_10 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_read_returns_file_content. Retrieved 1/9 statements.
# Partially parsed test_read_with_multiline_content. Retrieved 1/9 statements.
# Failed to parse test_read_empty_file.
# Partially parsed test_read_file_with_special_characters. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'Hello, World!'

def test_case_0():
    var_0 = 'Line 1\nLine 2\nLine 3'

def test_case_0():
    var_0 = "Special chars: !@#$%^&*()_+-=[]{}|;:',.<>?/~`"



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_gen_api_basic. Retrieved 8/20 statements.
# Partially parsed test_gen_api_dry_run. Retrieved 6/16 statements.
# Partially parsed test_gen_api_empty_doc. Retrieved 5/15 statements.
# Partially parsed test_gen_api_multiple_modules. Retrieved 7/16 statements.
# Partially parsed test_gen_api_with_pwd. Retrieved 6/17 statements.
# Partially parsed test_gen_api_level_parameter. Retrieved 6/14 statements.
# Partially parsed test_gen_api_file_naming. Retrieved 6/15 statements.


def test_case_0():
    var_0 = 'Test gen_api with basic parameters.'
    var_1 = 'docs'
    var_2 = 'TestModule'
    var_3 = 'test_module'
    var_4 = {var_2: var_3}
    var_5 = None
    var_6 = True
    var_7 = False
    var_8 = '# TestModule API'
    var_9 = '## Module'

def test_case_0():
    var_0 = 'Test gen_api with dry run enabled.'
    var_1 = 'API'
    var_2 = 'mymodule'
    var_3 = {var_1: var_2}
    var_4 = 'docs'
    var_5 = True

def test_case_0():
    var_0 = 'Test gen_api when loader returns empty documentation.'
    var_1 = 'Empty'
    var_2 = 'empty_module'
    var_3 = {var_1: var_2}
    var_4 = 'docs'

def test_case_0():
    var_0 = 'Test gen_api with multiple root modules.'
    var_1 = 'Module A'
    var_2 = 'Module B'
    var_3 = 'mod_a'
    var_4 = 'mod_b'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'docs'

def test_case_0():
    var_0 = 'Test gen_api with custom pwd parameter.'
    var_1 = 'site-packages'
    var_2 = 'Test'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = 'docs'

def test_case_0():
    var_0 = 'Test gen_api respects the level parameter.'
    var_1 = 'Title'
    var_2 = 'module'
    var_3 = {var_1: var_2}
    var_4 = 'docs'
    var_5 = 2
    var_6 = '## Title API'

def test_case_0():
    var_0 = 'Test gen_api creates correctly named files.'
    var_1 = 'Test'
    var_2 = 'test_module_name'
    var_3 = {var_1: var_2}
    var_4 = 'docs'
    var_5 = 0
    var_6 = 'test-module-name-api.md'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_load_module_predicate_true. Retrieved 6/22 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 9 evaluates to True.'
    var_1 = module_0.Parser()
    var_2 = 'test.module'
    var_3 = '/path/to/module.py'
    var_4 = module_1._load_module(var_2, var_3, var_1)
    assert var_4 is True
    var_5 = 'test.module'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = bool(var_0 == var_1)
    assert var_2 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_load_module_with_valid_loader. Retrieved 4/15 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module.py'
    var_1 = '"""Test module docstring."""\ndef test_func():\n    """Test function."""\n    pass\n'
    var_2 = module_0.Parser()
    var_3 = 'test_module'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_read_returns_file_contents. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_write_creates_file_and_writes_content. Retrieved 3/8 statements.
# Partially parsed test_write_overwrites_existing_file. Retrieved 4/9 statements.
# Partially parsed test_write_empty_string. Retrieved 3/8 statements.
# Partially parsed test_write_multiline_content. Retrieved 3/7 statements.
# Partially parsed test_write_special_characters. Retrieved 3/7 statements.
# Partially parsed test_write_unicode_content. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'old content'
    var_2 = 'utf-8'
    var_3 = 'new content'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = ''
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'line1\nline2\nline3'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Special chars: !@#$%^&*()_+-=[]{}|;\':",./<>?'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Unicode: 你好世界 🌍 مرحبا'
    var_2 = 'utf-8'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_load_module_success. Retrieved 5/19 statements.
# Partially parsed test_load_module_import_error. Retrieved 5/14 statements.
# Partially parsed test_load_module_invalid_spec. Retrieved 6/19 statements.
# Partially parsed test_load_module_no_loader. Retrieved 6/19 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module.py'
    var_2 = '"""Test module"""\ndef test_func():\n    pass\n'
    var_3 = '__import__'
    var_4 = 'test_module'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module.py'
    var_2 = 'def test_func():\n    pass\n'
    var_3 = '__import__'
    var_4 = 'nonexistent.test_module'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module.py'
    var_2 = 'def test_func():\n    pass\n'
    var_3 = '__import__'
    var_4 = 'importlib.util.spec_from_file_location'
    var_5 = 'test_module'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module.py'
    var_2 = 'def test_func():\n    pass\n'
    var_3 = '__import__'
    var_4 = 'importlib.util.spec_from_file_location'
    var_5 = 'test_module'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_read_returns_file_content. Retrieved 1/8 statements.
# Failed to parse test_read_empty_file.
# Partially parsed test_read_multiline_content. Retrieved 1/8 statements.
# Partially parsed test_read_with_special_characters. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 'Hello, World!'

def test_case_0():
    var_0 = 'Line 1\nLine 2\nLine 3'

def test_case_0():
    var_0 = 'Special chars: !@#$%^&*()'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_write_file_with_utf8_encoding. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_read_returns_file_contents. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'test_script.txt'
    var_1 = 'test script content'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_read_returns_file_content. Retrieved 2/9 statements.
# Partially parsed test_read_returns_empty_string_for_empty_file. Retrieved 2/9 statements.
# Partially parsed test_read_returns_multiline_content. Retrieved 2/9 statements.
# Partially parsed test_read_raises_error_for_nonexistent_file. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'

def test_case_0():
    var_0 = 'empty.txt'
    var_1 = ''

def test_case_0():
    var_0 = 'multiline.txt'
    var_1 = 'Line 1\nLine 2\nLine 3'

def test_case_0():
    var_0 = 'nonexistent.txt'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = bool(True)
    assert var_2 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_read_file_opens_in_read_mode. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 'test content'
    var_1 = bool(var_0)
    assert var_1 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_load_module_success. Retrieved 10/25 statements.
# Partially parsed test_load_module_invalid_spec. Retrieved 9/21 statements.
# Partially parsed test_load_module_with_docstring. Retrieved 10/26 statements.
# Partially parsed test_load_module_no_docstring. Retrieved 10/26 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module successfully loads a module and docstring.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'test_mod.py'
    var_5 = '"""Test module docstring."""\ndef func(): pass'
    var_6 = 0
    var_7 = module_0.Parser()
    var_8 = 'test_pkg.test_mod'
    var_9 = module_1._load_module(var_8, var_1, var_7)
    assert var_9 is True
    var_10 = 'test_pkg.test_mod'
    var_11 = bool('test_pkg.test_mod' in var_7.docstring)
    assert var_11 is True

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module returns False when parent module cannot be imported.'
    var_1 = module_0.Parser()
    var_2 = 'nonexistent.module'
    var_3 = '/fake/path.py'
    var_4 = module_1._load_module(var_2, var_3, var_1)
    assert var_4 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module returns False when spec cannot be created.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 0
    var_5 = module_0.Parser()
    var_6 = 'test_pkg.nonexistent'
    var_7 = '/nonexistent/path.py'
    var_8 = module_1._load_module(var_6, var_7, var_5)
    assert var_8 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module loads module with docstring correctly.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'test_mod.py'
    var_5 = '"""Module with docstring."""\nVAR = 42'
    var_6 = 0
    var_7 = module_0.Parser()
    var_8 = 'test_pkg.test_mod'
    var_9 = module_1._load_module(var_8, var_1, var_7)
    assert var_9 is True

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module handles module without docstring.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'test_mod.py'
    var_5 = 'VAR = 42'
    var_6 = 0
    var_7 = module_0.Parser()
    var_8 = 'test_pkg.test_mod'
    var_9 = module_1._load_module(var_8, var_1, var_7)
    assert var_9 is True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_write_predicate_evaluates_to_false. Retrieved 4/18 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'test content'
    var_2 = '/invalid/nonexistent/path/file.txt'
    var_3 = module_0._write(var_2, var_1)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_load_module_predicate_true. Retrieved 5/16 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 9 evaluates to True.'
    var_1 = 'test_module'
    var_2 = f'{var_1}.py'
    var_3 = '"""Test module docstring."""\n\ndef foo():\n    """Test function."""\n    pass\n'
    var_4 = module_0.Parser()



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/8 statements.
# Partially parsed test_write_overwrites_existing_file. Retrieved 4/9 statements.
# Partially parsed test_write_empty_string. Retrieved 3/8 statements.
# Partially parsed test_write_multiline_content. Retrieved 3/7 statements.
# Partially parsed test_write_special_characters. Retrieved 3/7 statements.
# Partially parsed test_write_unicode_content. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Old content'
    var_2 = 'utf-8'
    var_3 = 'New content'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = ''
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Line 1\nLine 2\nLine 3'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Special chars: !@#$%^&*()_+-=[]{}|;\':",./<>?'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Unicode: 你好世界 🌍 Привет'
    var_2 = 'utf-8'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_load_module_success. Retrieved 7/21 statements.
# Partially parsed test_load_module_import_parent_fails. Retrieved 6/11 statements.
# Partially parsed test_load_module_spec_none. Retrieved 7/18 statements.
# Partially parsed test_load_module_invalid_loader. Retrieved 7/21 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test _load_module successfully loads a module.'
    var_1 = 'test_module.py'
    var_2 = '"""Test module."""\ndef foo(): pass'
    var_3 = []
    var_4 = '__import__'
    var_5 = module_0.Parser()
    var_6 = 'test_module'

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module returns False when parent import fails.'
    var_1 = '__import__'
    var_2 = module_0.Parser()
    var_3 = 'nonexistent.module'
    var_4 = '/fake/path.py'
    var_5 = module_1._load_module(var_3, var_4, var_2)
    assert var_5 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module returns False when spec is None.'
    var_1 = '__import__'
    var_2 = 'spec_from_file_location'
    var_3 = module_0.Parser()
    var_4 = 'test_module'
    var_5 = '/nonexistent/path.py'
    var_6 = module_1._load_module(var_4, var_5, var_3)
    assert var_6 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module returns False when loader is not valid.'
    var_1 = '__import__'
    var_2 = 'spec_from_file_location'
    var_3 = module_0.Parser()
    var_4 = 'test_module'
    var_5 = '/fake/path.py'
    var_6 = module_1._load_module(var_4, var_5, var_3)
    assert var_6 is False



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_load_module_success. Retrieved 5/13 statements.
# Partially parsed test_load_module_with_docstring. Retrieved 6/12 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test _load_module successfully loads and processes a module.'
    var_1 = 'test_module.py'
    var_2 = '"""Test module."""\ndef foo():\n    """Test function."""\n    pass\n'
    var_3 = module_0.Parser()
    var_4 = 'test_module'

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module returns False when parent import fails.'
    var_1 = module_0.Parser()
    var_2 = 'nonexistent.module.test'
    var_3 = '/fake/path.py'
    var_4 = module_1._load_module(var_2, var_3, var_1)
    assert var_4 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module returns False when spec cannot be created.'
    var_1 = module_0.Parser()
    var_2 = 'os'
    var_3 = '/nonexistent/path.py'
    var_4 = module_1._load_module(var_2, var_3, var_1)
    assert var_4 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test _load_module loads module docstring into parser.'
    var_1 = 'documented_module.py'
    var_2 = '"""Module docstring."""\n\ndef bar():\n    """Function docstring."""\n    pass\n'
    var_3 = module_0.Parser()
    var_4 = 'documented_module'
    var_5 = var_3.parse(var_4, var_2)
    var_6 = bool('documented_module' in var_3.docstring or 'documented_module' in var_3.doc)
    assert var_6 is True



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_load_module_predicate_false_when_spec_is_none. Retrieved 4/11 statements.
# Partially parsed test_load_module_predicate_false_when_loader_not_instance. Retrieved 4/13 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '/fake/path.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '/fake/path.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_load_module_predicate_false. Retrieved 5/12 statements.
# Partially parsed test_load_module_predicate_false_not_loader. Retrieved 5/14 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 9 evaluates to False when s is None.'
    var_1 = module_0.Parser()
    var_2 = 'os.path'
    var_3 = '/fake/path.py'
    var_4 = module_1._load_module(var_2, var_3, var_1)
    assert var_4 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 9 evaluates to False when s.loader is not a Loader.'
    var_1 = module_0.Parser()
    var_2 = 'os.path'
    var_3 = '/fake/path.py'
    var_4 = module_1._load_module(var_2, var_3, var_1)
    assert var_4 is False



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_load_module_success. Retrieved 11/28 statements.
# Partially parsed test_load_module_with_docstring. Retrieved 11/29 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module successfully loads a module and updates parser.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'test_module.py'
    var_5 = '"""Test module docstring."""\ndef test_func(): pass'
    var_6 = 'PYTHONPATH'
    var_7 = 0
    var_8 = module_0.Parser()
    var_9 = 'test_pkg.test_module'
    var_10 = module_1._load_module(var_9, var_1, var_8)
    assert var_10 is True
    var_11 = 'test_pkg.test_module'
    var_12 = bool('test_pkg.test_module' in var_8.docstring)
    assert var_12 is True

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module returns False when parent module cannot be imported.'
    var_1 = module_0.Parser()
    var_2 = 'nonexistent.package.module'
    var_3 = '/fake/path.py'
    var_4 = module_1._load_module(var_2, var_3, var_1)
    assert var_4 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module returns False when spec_from_file_location returns None.'
    var_1 = module_0.Parser()
    var_2 = 'os.path'
    var_3 = '/nonexistent/fake/path.py'
    var_4 = module_1._load_module(var_2, var_3, var_1)
    assert var_4 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module correctly loads module docstring into parser.'
    var_1 = 'pkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'mod.py'
    var_5 = '"""Module with docstring."""\nVAR = 42'
    var_6 = 'PYTHONPATH'
    var_7 = 0
    var_8 = module_0.Parser()
    var_9 = 'pkg.mod'
    var_10 = module_1._load_module(var_9, var_1, var_8)
    assert var_10 is True
    var_11 = 'pkg.mod'
    var_12 = bool('pkg.mod' in var_8.docstring)
    assert var_12 is True



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_load_module_predicate_false. Retrieved 5/12 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 9 evaluates to False when s is None.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = '/fake/path.py'
    var_4 = module_1._load_module(var_2, var_3, var_1)
    assert var_4 is False



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_write_predicate_false. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'test content'



# Parsed testcases at query #49
#--------------------------




import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 9 evaluates to False when s is None.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = '/fake/path.py'
    var_4 = module_1._load_module(var_2, var_3, var_1)
    assert var_4 is False



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_load_module_predicate_false_when_loader_not_instance. Retrieved 4/12 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '/fake/path/test_module.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '/fake/path/test_module.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_read_file_successfully. Retrieved 1/12 statements.


def test_case_0():
    var_0 = 'test script content'



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_loader_pure_py_false_condition. Retrieved 12/25 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 15 (if pure_py:) evaluates to False.'
    var_1 = 'test_module'
    var_2 = '/fake/path'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = ''
    var_6 = False
    var_7 = '/root'
    var_8 = '/pwd'
    var_9 = False
    var_10 = 1
    var_11 = module_0.loader(var_7, var_8, var_9, var_10, var_9)
    assert var_11 == 'compiled'



# Parsed testcases at query #53
#--------------------------




def test_case_0():
    var_0 = 'Test that the predicate at line 13 (ext == ".py") evaluates to False.'
    var_1 = '.pyi'
    var_2 = bool(not var_1 == '.py')
    assert var_2 is True



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_read_file_opens_in_read_mode. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 'test content'
    var_1 = bool(var_0)
    assert var_1 is True



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_gen_api_predicate_line_25_true. Retrieved 8/15 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 25 evaluates to True when doc is empty or whitespace.'
    var_1 = 'TestModule'
    var_2 = 'test_module'
    var_3 = {var_1: var_2}
    var_4 = 'docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = 'can not be found'
    var_9 = bool(var_7 == [])
    assert var_9 is True



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = bool(var_0 == var_1)
    assert var_2 is True



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_loader_with_valid_package. Retrieved 8/19 statements.
# Partially parsed test_loader_with_no_matching_packages. Retrieved 7/15 statements.
# Partially parsed test_loader_with_toc_enabled. Retrieved 5/13 statements.
# Partially parsed test_loader_with_link_disabled. Retrieved 6/14 statements.
# Partially parsed test_loader_with_nested_modules. Retrieved 8/21 statements.
# Partially parsed test_loader_with_stub_file. Retrieved 6/14 statements.
# Partially parsed test_loader_with_different_levels. Retrieved 7/18 statements.
# Partially parsed test_loader_with_complex_docstring. Retrieved 6/14 statements.
# Partially parsed test_loader_with_class_definition. Retrieved 6/14 statements.
# Partially parsed test_loader_with_py_and_pyi_files. Retrieved 8/18 statements.


def test_case_0():
    var_0 = 'Test loader with a valid package structure.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = 'def hello(): pass'
    var_4 = 'module.py'
    var_5 = 'def world(): pass'
    var_6 = True
    var_7 = False

def test_case_0():
    var_0 = 'Test loader when no packages match the root name.'
    var_1 = 'other_pkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'nonexistent_pkg'
    var_5 = True
    var_6 = False

def test_case_0():
    var_0 = 'Test loader with table of contents enabled.'
    var_1 = 'my_pkg'
    var_2 = '__init__.py'
    var_3 = 'def func1(): pass\ndef func2(): pass'
    var_4 = True
    var_5 = '**Table of contents:**'

def test_case_0():
    var_0 = 'Test loader with link generation disabled.'
    var_1 = 'link_pkg'
    var_2 = '__init__.py'
    var_3 = 'def test(): pass'
    var_4 = False
    var_5 = 1

def test_case_0():
    var_0 = 'Test loader with nested module structure.'
    var_1 = 'nested_pkg'
    var_2 = '__init__.py'
    var_3 = 'def root_func(): pass'
    var_4 = 'submodule'
    var_5 = 'def sub_func(): pass'
    var_6 = True
    var_7 = False

def test_case_0():
    var_0 = 'Test loader with .pyi stub file.'
    var_1 = 'stub_pkg'
    var_2 = '__init__.pyi'
    var_3 = 'def stub_func() -> int: ...'
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader with different heading levels.'
    var_1 = 'level_pkg'
    var_2 = '__init__.py'
    var_3 = 'def func(): pass'
    var_4 = True
    var_5 = False
    var_6 = 2

def test_case_0():
    var_0 = 'Test loader with modules containing docstrings.'
    var_1 = 'doc_pkg'
    var_2 = '"""Module docstring."""\ndef documented_func():\n    """Function docstring."""\n    pass\n'
    var_3 = '__init__.py'
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader with class definitions.'
    var_1 = 'class_pkg'
    var_2 = 'class MyClass:\n    """Class docstring."""\n    def method(self):\n        """Method docstring."""\n        pass\n'
    var_3 = '__init__.py'
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader preferring .py over .pyi when both exist.'
    var_1 = 'dual_pkg'
    var_2 = '__init__.py'
    var_3 = 'def py_func(): pass'
    var_4 = '__init__.pyi'
    var_5 = 'def pyi_func() -> None: ...'
    var_6 = True
    var_7 = False



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_loader_predicate_false_when_pure_py_false. Retrieved 8/20 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Ensure that the predicate at line 15 evaluates to False when pure_py is False.'
    var_1 = 'test_module'
    var_2 = '/path/test_module'
    var_3 = (var_1, var_2)
    var_4 = '/root'
    var_5 = '/pwd'
    var_6 = True
    var_7 = module_0.loader(var_4, var_5, var_6, var_6, var_6)
    assert var_7 == 'compiled_output'



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_load_module_success. Retrieved 4/11 statements.
# Partially parsed test_load_module_parent_import_error. Retrieved 5/14 statements.
# Partially parsed test_load_module_invalid_spec. Retrieved 5/15 statements.
# Partially parsed test_load_module_no_loader. Retrieved 6/18 statements.
# Partially parsed test_load_module_calls_load_docstring. Retrieved 7/17 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module.py'
    var_1 = '"""Test module docstring."""\ndef test_func():\n    """Test function."""\n    pass\n'
    var_2 = module_0.Parser()
    var_3 = 'test_module'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module.py'
    var_1 = '"""Test module."""'
    var_2 = 'builtins.__import__'
    var_3 = module_0.Parser()
    var_4 = 'nonexistent.test_module'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module.py'
    var_1 = '"""Test module."""'
    var_2 = 'importlib.util.spec_from_file_location'
    var_3 = module_0.Parser()
    var_4 = 'test_module'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module.py'
    var_1 = '"""Test module."""'
    var_2 = None
    var_3 = 'importlib.util.spec_from_file_location'
    var_4 = module_0.Parser()
    var_5 = 'test_module'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module.py'
    var_1 = '"""Module docstring."""\ndef func():\n    """Function doc."""\n    pass\n'
    var_2 = module_0.Parser()
    var_3 = []
    var_4 = var_2.load_docstring
    var_5 = 'test_module'
    var_6 = len(var_3)
    assert var_6 == 1
    var_7 = var_3[0][0]
    assert var_7 == 'test_module'



# Parsed testcases at query #60
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 25 (if not doc.strip()) evaluates to True when doc is empty or whitespace.'
    var_1 = 'Test'
    var_2 = 'test_module'
    var_3 = {var_1: var_2}
    var_4 = 1
    var_5 = module_0.gen_api(var_3, level=var_4)
    var_6 = bool(var_5 == [])
    assert var_6 is True
    var_7 = 'Test'
    var_8 = 'test_module'
    var_9 = {var_7: var_8}
    var_10 = 1
    var_11 = module_0.gen_api(var_9, level=var_10)
    var_12 = bool(var_11 == [])
    assert var_12 is True



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_read_returns_file_content. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 2/17 statements.


def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_read_file_opens_in_read_mode. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'hello world'



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_write_file_opens_in_write_mode. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_loader_predicate_line_13_false. Retrieved 9/18 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 13 (ext == ".py") evaluates to False.'
    var_1 = 'test_module'
    var_2 = '/path/to/test_module'
    var_3 = (var_1, var_2)
    var_4 = False
    var_5 = True
    var_6 = '/root'
    var_7 = '/pwd'
    var_8 = module_0.loader(var_6, var_7, var_4, var_5, var_4)
    assert var_8 == 'compiled_output'



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_load_module_success. Retrieved 7/24 statements.
# Partially parsed test_load_module_import_error. Retrieved 5/14 statements.
# Partially parsed test_load_module_invalid_spec. Retrieved 5/15 statements.
# Partially parsed test_load_module_no_loader. Retrieved 6/17 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module.py'
    var_2 = '"""Test module docstring."""\ndef test_func():\n    """Test function."""\n    pass\n'
    var_3 = []
    var_4 = '__import__'
    var_5 = 'builtins.__import__'
    var_6 = 'test_module'
    var_7 = 'test_module'
    var_8 = bool('test_module' in var_0.docstring)
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module.py'
    var_2 = '"""Test module."""\n'
    var_3 = 'builtins.__import__'
    var_4 = 'nonexistent.module'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module.py'
    var_2 = '"""Test module."""\n'
    var_3 = 'importlib.util.spec_from_file_location'
    var_4 = 'test_module'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module.py'
    var_2 = '"""Test module."""\n'
    var_3 = 'test_module'
    var_4 = None
    var_5 = 'importlib.util.spec_from_file_location'



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_loader_with_valid_package. Retrieved 8/17 statements.
# Partially parsed test_loader_with_stub_files. Retrieved 6/13 statements.
# Partially parsed test_loader_with_toc_enabled. Retrieved 5/12 statements.
# Partially parsed test_loader_with_nested_modules. Retrieved 8/19 statements.
# Partially parsed test_loader_with_link_disabled. Retrieved 6/13 statements.
# Partially parsed test_loader_with_custom_level. Retrieved 7/14 statements.
# Partially parsed test_loader_nonexistent_package. Retrieved 4/7 statements.
# Partially parsed test_loader_with_multiple_modules. Retrieved 10/21 statements.
# Partially parsed test_loader_empty_package. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'Test loader with a valid package structure.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\ndef func(): pass'
    var_4 = 'module.py'
    var_5 = '"""Test module."""\ndef another_func(): pass'
    var_6 = True
    var_7 = False
    var_8 = 'test_pkg'
    var_9 = 'func'

def test_case_0():
    var_0 = 'Test loader with .pyi stub files.'
    var_1 = 'stub_pkg'
    var_2 = '__init__.pyi'
    var_3 = 'def stub_func() -> None: ...'
    var_4 = True
    var_5 = False
    var_6 = 'stub_pkg'

def test_case_0():
    var_0 = 'Test loader with table of contents enabled.'
    var_1 = 'toc_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Package with TOC."""\ndef test_func(): pass'
    var_4 = True
    var_5 = '**Table of contents:**'

def test_case_0():
    var_0 = 'Test loader with nested package structure.'
    var_1 = 'nested_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Root package."""\ndef root_func(): pass'
    var_4 = 'submodule'
    var_5 = '"""Sub package."""\ndef sub_func(): pass'
    var_6 = True
    var_7 = False
    var_8 = 'nested_pkg'

def test_case_0():
    var_0 = 'Test loader with link generation disabled.'
    var_1 = 'no_link_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\ndef test_func(): pass'
    var_4 = False
    var_5 = 1
    var_6 = 'no_link_pkg'

def test_case_0():
    var_0 = 'Test loader with custom heading level.'
    var_1 = 'level_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\ndef test_func(): pass'
    var_4 = True
    var_5 = 2
    var_6 = False
    var_7 = 'level_pkg'

def test_case_0():
    var_0 = 'Test loader with non-existent package.'
    var_1 = 'nonexistent_pkg'
    var_2 = True
    var_3 = False

def test_case_0():
    var_0 = 'Test loader with multiple module files.'
    var_1 = 'multi_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Main package."""\ndef main_func(): pass'
    var_4 = 'mod1.py'
    var_5 = '"""Module 1."""\ndef func1(): pass'
    var_6 = 'mod2.py'
    var_7 = '"""Module 2."""\ndef func2(): pass'
    var_8 = True
    var_9 = False
    var_10 = 'multi_pkg'

def test_case_0():
    var_0 = 'Test loader with empty package.'
    var_1 = 'empty_pkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = True
    var_5 = False



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/7 statements.
# Partially parsed test_write_overwrites_existing_file. Retrieved 4/10 statements.
# Partially parsed test_write_handles_empty_string. Retrieved 3/7 statements.
# Partially parsed test_write_handles_multiline_text. Retrieved 3/7 statements.
# Partially parsed test_write_handles_special_characters. Retrieved 3/7 statements.
# Partially parsed test_write_handles_unicode_characters. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'First content'
    var_2 = 'Second content'
    var_3 = 'utf-8'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = ''
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Line 1\nLine 2\nLine 3'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = "Special chars: !@#$%^&*()_+-=[]{}|;:',.<>?/\\"
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Unicode: 你好世界 🌍 Ñoño'
    var_2 = 'utf-8'



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_write_predicate_false. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'test content'



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_read_file_predicate_evaluates_to_false. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'test content'
    var_2 = 'r'
    var_3 = None



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_read_file_content. Retrieved 2/11 statements.
# Partially parsed test_read_empty_file. Retrieved 2/11 statements.
# Partially parsed test_read_multiline_file. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'test_script.txt'
    var_1 = 'test script content'

def test_case_0():
    var_0 = 'empty_script.txt'
    var_1 = ''

def test_case_0():
    var_0 = 'multiline_script.txt'
    var_1 = 'line1\nline2\nline3'



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_write_creates_file_with_correct_content. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_load_module_success. Retrieved 4/11 statements.
# Partially parsed test_load_module_import_error. Retrieved 4/10 statements.
# Partially parsed test_load_module_with_docstring. Retrieved 4/10 statements.
# Partially parsed test_load_module_returns_false_on_no_loader. Retrieved 5/18 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module.py'
    var_1 = '"""Test module."""\ndef test_func():\n    """Test function."""\n    pass\n'
    var_2 = module_0.Parser()
    var_3 = 'test_module'
    var_4 = 'test_module'
    var_5 = bool('test_module' in var_2.docstring)
    assert var_5 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module.py'
    var_1 = '"""Test module."""'
    var_2 = module_0.Parser()
    var_3 = 'nonexistent.submodule.test'

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'sys'
    var_2 = '/nonexistent/path/to/module.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = 'documented_module.py'
    var_1 = '"""Module docstring.\n\nThis is a detailed description.\n"""\ndef func():\n    """Function docstring."""\n    pass\n'
    var_2 = module_0.Parser()
    var_3 = 'documented_module'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module.py'
    var_1 = '"""Test."""'
    var_2 = module_0.Parser()
    var_3 = 'apimd.loader.spec_from_file_location'
    var_4 = 'test_module'



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_loader_basic. Retrieved 8/18 statements.
# Partially parsed test_loader_with_toc. Retrieved 7/17 statements.
# Partially parsed test_loader_nested_packages. Retrieved 10/24 statements.
# Partially parsed test_loader_with_pyi_stub. Retrieved 8/18 statements.
# Partially parsed test_loader_without_link. Retrieved 8/18 statements.
# Partially parsed test_loader_with_different_level. Retrieved 9/19 statements.
# Partially parsed test_loader_empty_package. Retrieved 6/14 statements.
# Partially parsed test_loader_with_constants. Retrieved 6/14 statements.
# Partially parsed test_loader_multiple_modules. Retrieved 10/22 statements.


def test_case_0():
    var_0 = 'Test loader function with basic package structure.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\n'
    var_4 = 'module.py'
    var_5 = '"""Test module."""\ndef func(): pass\n'
    var_6 = True
    var_7 = False
    var_8 = 'test_pkg'

def test_case_0():
    var_0 = 'Test loader function with table of contents enabled.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\n'
    var_4 = 'module.py'
    var_5 = '"""Test module."""\ndef func(): pass\n'
    var_6 = True
    var_7 = '**Table of contents:**'

def test_case_0():
    var_0 = 'Test loader with nested package structure.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\n'
    var_4 = 'sub'
    var_5 = '"""Sub package."""\n'
    var_6 = 'module.py'
    var_7 = '"""Sub module."""\nclass MyClass: pass\n'
    var_8 = True
    var_9 = False
    var_10 = 'test_pkg'

def test_case_0():
    var_0 = 'Test loader prioritizes .pyi stub files.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\n'
    var_4 = 'module.pyi'
    var_5 = 'def stub_func() -> int: ...\n'
    var_6 = False
    var_7 = 1

def test_case_0():
    var_0 = 'Test loader with link disabled.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\n'
    var_4 = 'module.py'
    var_5 = '"""Test module."""\n'
    var_6 = False
    var_7 = 1
    var_8 = '<a id='

def test_case_0():
    var_0 = 'Test loader with different heading level.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\n'
    var_4 = 'module.py'
    var_5 = '"""Test module."""\n'
    var_6 = True
    var_7 = 2
    var_8 = False
    var_9 = '###'

def test_case_0():
    var_0 = 'Test loader with empty package.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Empty package."""\n'
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader with module constants.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\nVERSION: str = \'1.0.0\'\n'
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader with multiple modules in package.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\n'
    var_4 = 'module1.py'
    var_5 = '"""Module 1."""\ndef func1(): pass\n'
    var_6 = 'module2.py'
    var_7 = '"""Module 2."""\ndef func2(): pass\n'
    var_8 = True
    var_9 = False



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_loader_basic. Retrieved 6/13 statements.
# Partially parsed test_loader_with_submodules. Retrieved 8/17 statements.
# Partially parsed test_loader_with_toc. Retrieved 5/12 statements.
# Partially parsed test_loader_with_stub_file. Retrieved 8/17 statements.
# Partially parsed test_loader_multiple_modules. Retrieved 10/21 statements.
# Partially parsed test_loader_with_different_base_level. Retrieved 6/13 statements.
# Partially parsed test_loader_without_link. Retrieved 6/13 statements.
# Partially parsed test_loader_nested_packages. Retrieved 8/19 statements.
# Partially parsed test_loader_with_class_definition. Retrieved 6/13 statements.
# Partially parsed test_loader_ignores_non_python_files. Retrieved 10/21 statements.


def test_case_0():
    var_0 = 'Test loader with a simple package structure.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\ndef foo(): pass'
    var_4 = True
    var_5 = False
    var_6 = 'test_pkg'
    var_7 = 'foo'

def test_case_0():
    var_0 = 'Test loader with submodules.'
    var_1 = 'mypackage'
    var_2 = '__init__.py'
    var_3 = '"""Main package."""\n'
    var_4 = 'submodule.py'
    var_5 = '"""Sub module."""\ndef bar(): pass'
    var_6 = False
    var_7 = 1
    var_8 = 'mypackage'
    var_9 = 'bar'

def test_case_0():
    var_0 = 'Test loader with table of contents enabled.'
    var_1 = 'tocpkg'
    var_2 = '__init__.py'
    var_3 = '"""Package with TOC."""\ndef func1(): pass'
    var_4 = True
    var_5 = 'Table of contents'
    var_6 = 'tocpkg'

def test_case_0():
    var_0 = 'Test loader prefers .pyi stub files.'
    var_1 = 'stubpkg'
    var_2 = '__init__.py'
    var_3 = '"""Python file."""\ndef py_func(): pass'
    var_4 = '__init__.pyi'
    var_5 = '"""Stub file."""\ndef stub_func(): pass'
    var_6 = False
    var_7 = 1
    var_8 = 'stubpkg'
    var_9 = 'stub_func'

def test_case_0():
    var_0 = 'Test loader with multiple modules in package.'
    var_1 = 'multipkg'
    var_2 = '__init__.py'
    var_3 = '"""Main module."""\n'
    var_4 = 'mod1.py'
    var_5 = '"""Module 1."""\ndef func1(): pass'
    var_6 = 'mod2.py'
    var_7 = '"""Module 2."""\ndef func2(): pass'
    var_8 = False
    var_9 = 2
    var_10 = 'multipkg'
    var_11 = 'func1'
    var_12 = 'func2'

def test_case_0():
    var_0 = 'Test loader respects base level parameter.'
    var_1 = 'levelpkg'
    var_2 = '__init__.py'
    var_3 = '"""Package."""\ndef test_func(): pass'
    var_4 = False
    var_5 = 3
    var_6 = 'levelpkg'

def test_case_0():
    var_0 = 'Test loader with link parameter disabled.'
    var_1 = 'nolinkpkg'
    var_2 = '__init__.py'
    var_3 = '"""No link package."""\ndef nolink_func(): pass'
    var_4 = False
    var_5 = 1
    var_6 = 'nolinkpkg'
    var_7 = '<a id='

def test_case_0():
    var_0 = 'Test loader with nested package structure.'
    var_1 = 'outer'
    var_2 = '__init__.py'
    var_3 = '"""Outer package."""\n'
    var_4 = 'inner'
    var_5 = '"""Inner package."""\ndef nested_func(): pass'
    var_6 = False
    var_7 = 1
    var_8 = 'outer'
    var_9 = 'nested_func'

def test_case_0():
    var_0 = 'Test loader handles class definitions.'
    var_1 = 'classpkg'
    var_2 = '__init__.py'
    var_3 = '"""Package with class."""\nclass MyClass:\n    """A test class."""\n    def method(self): pass'
    var_4 = False
    var_5 = 1
    var_6 = 'classpkg'
    var_7 = 'MyClass'

def test_case_0():
    var_0 = 'Test loader ignores non-Python files.'
    var_1 = 'filterpkg'
    var_2 = '__init__.py'
    var_3 = '"""Filter package."""\n'
    var_4 = 'readme.txt'
    var_5 = 'This should be ignored'
    var_6 = 'data.json'
    var_7 = '{}'
    var_8 = False
    var_9 = 1
    var_10 = 'filterpkg'
    var_11 = 'readme'
    var_12 = 'json'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_loader_basic. Retrieved 5/17 statements.
# Partially parsed test_loader_with_toc. Retrieved 4/15 statements.
# Partially parsed test_loader_multiple_files. Retrieved 9/25 statements.
# Partially parsed test_loader_no_link. Retrieved 5/17 statements.
# Partially parsed test_loader_with_class. Retrieved 5/17 statements.
# Partially parsed test_loader_with_constants. Retrieved 5/16 statements.
# Partially parsed test_loader_different_levels. Retrieved 6/19 statements.
# Partially parsed test_loader_with_stub_file. Retrieved 5/16 statements.


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\ndef func():\n    """A function."""\n    pass'
    var_3 = True
    var_4 = False

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\ndef func():\n    """A function."""\n    pass'
    var_3 = True
    var_4 = '**Table of contents:**'

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""'
    var_3 = 'module1.py'
    var_4 = '"""Module 1."""\ndef func1():\n    """Function 1."""\n    pass'
    var_5 = 'module2.py'
    var_6 = '"""Module 2."""\ndef func2():\n    """Function 2."""\n    pass'
    var_7 = True
    var_8 = False

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\ndef func():\n    """A function."""\n    pass'
    var_3 = False
    var_4 = 1

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\nclass MyClass:\n    """A class."""\n    def method(self):\n        """A method."""\n        pass'
    var_3 = True
    var_4 = False
    var_5 = 'class'

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\nCONSTANT: int = 42'
    var_3 = True
    var_4 = False

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\ndef func():\n    """A function."""\n    pass'
    var_3 = True
    var_4 = False
    var_5 = 2

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.pyi'
    var_2 = '"""Test package."""\ndef func() -> None: ...'
    var_3 = True
    var_4 = False



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_loader_predicate_line_9_false. Retrieved 11/22 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 9 (not isfile(path_ext)) evaluates to False.'
    var_1 = 'test_module'
    var_2 = '/fake/path/test_module'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = True
    var_6 = '/root'
    var_7 = '/pwd'
    var_8 = False
    var_9 = 1
    var_10 = module_0.loader(var_6, var_7, var_8, var_9, var_8)
    assert var_10 == 'compiled'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_loader_predicate_line_9_false. Retrieved 11/22 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 9 evaluates to False when file exists.'
    var_1 = 'test_module'
    var_2 = '/fake/path/test_module'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = True
    var_6 = '/root'
    var_7 = '/pwd'
    var_8 = False
    var_9 = 1
    var_10 = module_0.loader(var_6, var_7, var_8, var_9, var_8)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_loader_basic. Retrieved 5/17 statements.
# Partially parsed test_loader_with_toc. Retrieved 4/15 statements.
# Partially parsed test_loader_no_link. Retrieved 5/16 statements.
# Partially parsed test_loader_with_submodule. Retrieved 8/22 statements.
# Partially parsed test_loader_different_level. Retrieved 6/18 statements.
# Partially parsed test_loader_multiple_functions. Retrieved 4/15 statements.


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\ndef hello():\n    """Say hello."""\n    pass\n'
    var_3 = True
    var_4 = False
    var_5 = 'test_pkg'
    var_6 = 'hello'

def test_case_0():
    var_0 = 'test_pkg2'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\nclass MyClass:\n    """A class."""\n    pass\n'
    var_3 = True
    var_4 = 'Table of contents'

def test_case_0():
    var_0 = 'test_pkg3'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\ndef func():\n    """A function."""\n    pass\n'
    var_3 = False
    var_4 = 1
    var_5 = 'test_pkg3'

def test_case_0():
    var_0 = 'test_pkg4'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\n'
    var_3 = 'submodule.py'
    var_4 = '"""Submodule."""\ndef sub_func():\n    """Sub function."""\n    pass\n'
    var_5 = True
    var_6 = 2
    var_7 = False
    var_8 = 'test_pkg4'

def test_case_0():
    var_0 = 'test_pkg5'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\ndef test():\n    """Test."""\n    pass\n'
    var_3 = True
    var_4 = 3
    var_5 = False

def test_case_0():
    var_0 = 'test_pkg6'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\ndef func1():\n    """First function."""\n    pass\ndef func2():\n    """Second function."""\n    pass\n'
    var_3 = True
    var_4 = 'func1'
    var_5 = 'func2'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_gen_api_creates_directory. Retrieved 7/16 statements.
# Partially parsed test_gen_api_with_pwd. Retrieved 7/17 statements.
# Partially parsed test_gen_api_writes_file. Retrieved 6/14 statements.
# Partially parsed test_gen_api_level_parameter. Retrieved 8/18 statements.
# Partially parsed test_gen_api_toc_parameter. Retrieved 7/17 statements.
# Partially parsed test_gen_api_underscore_to_hyphen. Retrieved 6/16 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api with basic parameters.'
    var_1 = 'Test'
    var_2 = 'test_module'
    var_3 = {var_1: var_2}
    var_4 = True
    var_5 = module_0.gen_api(var_3, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = '# Test Doc'
    var_8 = bool('# Test Doc' in var_5[0])
    assert var_8 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api when loader returns empty string.'
    var_1 = 'Test'
    var_2 = 'test_module'
    var_3 = {var_1: var_2}
    var_4 = True
    var_5 = module_0.gen_api(var_3, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api with multiple modules.'
    var_1 = 'Module1'
    var_2 = 'Module2'
    var_3 = 'mod1'
    var_4 = 'mod2'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = True
    var_7 = module_0.gen_api(var_5, dry=var_6)
    var_8 = len(var_7)
    assert var_8 == 2

import apimd.loader as module_0

def test_case_0():
    var_0 = "Test gen_api creates directory when it doesn't exist."
    var_1 = 'Test'
    var_2 = 'test_module'
    var_3 = {var_1: var_2}
    var_4 = 'custom_docs'
    var_5 = True
    var_6 = module_0.gen_api(var_3, prefix=var_4, dry=var_5)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api appends pwd to sys.path.'
    var_1 = 'Test'
    var_2 = 'test_module'
    var_3 = {var_1: var_2}
    var_4 = '/custom/path'
    var_5 = True
    var_6 = module_0.gen_api(var_3, var_4, dry=var_5)
    var_7 = '/custom/path'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api writes to file when dry=False.'
    var_1 = 'Test'
    var_2 = 'test_module'
    var_3 = {var_1: var_2}
    var_4 = False
    var_5 = module_0.gen_api(var_3, dry=var_4)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api with custom level parameter.'
    var_1 = '# Test'
    var_2 = 'Test'
    var_3 = 'test_module'
    var_4 = {var_2: var_3}
    var_5 = 2
    var_6 = True
    var_7 = module_0.gen_api(var_4, level=var_5, dry=var_6)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api with toc parameter.'
    var_1 = '# Test'
    var_2 = 'Test'
    var_3 = 'test_module'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = module_0.gen_api(var_4, toc=var_5, dry=var_5)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api converts underscores to hyphens in filename.'
    var_1 = 'Test'
    var_2 = 'test_module_name'
    var_3 = {var_1: var_2}
    var_4 = False
    var_5 = module_0.gen_api(var_3, dry=var_4)
    var_6 = 'test-module-name-api.md'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_loader_predicate_line_9_evaluates_to_false. Retrieved 13/30 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 9 (not isfile(path_ext)) evaluates to False.'
    var_1 = 'test_module'
    var_2 = '/fake/path/test_module'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = '.py'
    var_6 = lambda path: path.endswith(var_5)
    var_7 = '/root'
    var_8 = '/pwd'
    var_9 = False
    var_10 = 1
    var_11 = module_0.loader(var_7, var_8, var_9, var_10, var_9)
    var_12 = 0



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_read_returns_file_content. Retrieved 2/6 statements.
# Partially parsed test_read_empty_file. Retrieved 2/6 statements.
# Partially parsed test_read_multiline_content. Retrieved 2/6 statements.
# Partially parsed test_read_with_special_characters. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'test_script.txt'
    var_1 = "print('Hello, World!')"

def test_case_0():
    var_0 = 'empty_script.txt'
    var_1 = ''

def test_case_0():
    var_0 = 'multiline_script.txt'
    var_1 = 'line1\nline2\nline3'

def test_case_0():
    var_0 = 'special_chars.txt'
    var_1 = 'special chars: !@#$%^&*()\n\ttab\n'

import apimd.loader as module_0

def test_case_0():
    var_0 = '/nonexistent/path/to/file.txt'
    var_1 = module_0._read(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #8
#--------------------------




def test_case_0():
    var_0 = "Test that the predicate at line 13 evaluates to False when ext is not '.py'"
    var_1 = '.pyi'
    assert var_1 == '.py'
    var_2 = '.py'
    assert var_2 is False



# Parsed testcases at query #9
#--------------------------




def test_case_0():
    var_0 = "Test that the predicate at line 13 evaluates to True when ext == '.py'."
    var_1 = '.py'
    var_2 = '.py'
    var_3 = var_1 == var_2
    assert var_3 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_loader_pure_py_false. Retrieved 6/17 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 15 evaluates to False when pure_py is False.'
    var_1 = '/fake/root'
    var_2 = '/fake/pwd'
    var_3 = False
    var_4 = 1
    var_5 = module_0.loader(var_1, var_2, var_3, var_4, var_3)
    assert var_5 == 'compiled_output'



# Parsed testcases at query #11
#--------------------------




def test_case_0():
    var_0 = 'Test that the predicate at line 13 (ext == ".py") evaluates to True.'
    var_1 = '.py'
    assert var_1 == '.py'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_loader_predicate_line_7_false. Retrieved 6/14 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 7 (for ext in [".py", ".pyi"]) evaluates to False when list is empty.'
    var_1 = 'root'
    var_2 = 'pwd'
    var_3 = False
    var_4 = 1
    var_5 = module_0.loader(var_1, var_2, var_3, var_4, var_3)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_loader_basic. Retrieved 5/17 statements.
# Partially parsed test_loader_with_toc. Retrieved 4/15 statements.
# Partially parsed test_loader_without_link. Retrieved 5/16 statements.
# Partially parsed test_loader_with_different_level. Retrieved 6/17 statements.
# Partially parsed test_loader_multiple_modules. Retrieved 7/21 statements.
# Partially parsed test_loader_nested_packages. Retrieved 7/23 statements.
# Partially parsed test_loader_with_constants. Retrieved 5/16 statements.
# Partially parsed test_loader_with_class. Retrieved 5/16 statements.


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\ndef foo(): pass'
    var_3 = True
    var_4 = False
    var_5 = 'test_pkg'
    var_6 = 'foo'

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\ndef bar(): pass'
    var_3 = True
    var_4 = 'Table of contents'

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\nclass MyClass: pass'
    var_3 = False
    var_4 = 1
    var_5 = 'MyClass'

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\ndef baz(): pass'
    var_3 = True
    var_4 = 2
    var_5 = False

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""'
    var_3 = 'module.py'
    var_4 = '"""Test module."""\ndef func(): pass'
    var_5 = True
    var_6 = False
    var_7 = 'test_pkg'

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""'
    var_3 = 'sub'
    var_4 = '"""Sub package."""\ndef sub_func(): pass'
    var_5 = True
    var_6 = False

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\nCONSTANT = 42'
    var_3 = True
    var_4 = False

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\nclass TestClass:\n    def method(self): pass'
    var_3 = True
    var_4 = False
    var_5 = 'TestClass'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_gen_api_creates_directory_when_not_exists. Retrieved 4/11 statements.


def test_case_0():
    var_0 = "Test that gen_api creates the prefix directory if it doesn't exist."
    var_1 = 'new_docs'
    var_2 = {}
    var_3 = True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_gen_api_creates_prefix_directory_when_not_exists. Retrieved 15/27 statements.


def test_case_0():
    var_0 = "Test that gen_api creates the prefix directory if it doesn't exist."
    var_1 = 'new_docs'
    var_2 = 'apimd.loader.isdir'
    var_3 = 'apimd.loader.mkdir'
    var_4 = None
    var_5 = lambda x: var_4
    var_6 = 'apimd.loader.loader'
    var_7 = ''
    var_8 = lambda *args, **kwargs: var_7
    var_9 = 'apimd.loader._site_path'
    var_10 = lambda x: var_4
    var_11 = 'apimd.loader.sys_path'
    var_12 = []
    var_13 = {}
    var_14 = True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_gen_api_creates_directory_when_prefix_does_not_exist. Retrieved 8/16 statements.


def test_case_0():
    var_0 = "Test that gen_api creates the prefix directory if it doesn't exist."
    var_1 = 'new_docs'
    var_2 = 'PYTHONPATH'
    var_3 = ''
    var_4 = 'TestModule'
    var_5 = 'nonexistent_module'
    var_6 = {var_4: var_5}
    var_7 = True



# Parsed testcases at query #17
#--------------------------




def test_case_0():
    var_0 = 'Test that the predicate at line 13 (ext == ".py") evaluates to False.'
    var_1 = '.pyi'
    var_2 = bool(not var_1 == '.py')
    assert var_2 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_loader_predicate_line_15_false. Retrieved 10/21 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 15 evaluates to False when .py file is not found.'
    var_1 = 'test_module'
    var_2 = '/fake/path'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = '/root'
    var_6 = '/pwd'
    var_7 = False
    var_8 = 1
    var_9 = module_0.loader(var_5, var_6, var_7, var_8, var_7)
    assert var_9 == 'compiled'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_site_path_returns_empty_string_when_submodule_search_locations_is_none. Retrieved 4/10 statements.
# Partially parsed test_site_path_returns_directory_when_spec_exists. Retrieved 7/14 statements.
# Partially parsed test_site_path_with_real_existing_module. Retrieved 5/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_module'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = None
    var_2 = 'test_module'
    var_3 = module_0._site_path(var_2)
    assert var_3 == ''

import apimd.loader as module_0
import posixpath as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = None
    var_2 = '/path/to/test_module'
    var_3 = 'test_module'
    var_4 = module_0._site_path(var_3)
    assert var_4 == '/path/to'
    var_5 = '/path/to/test_module'
    var_6 = module_1.dirname(var_5)
    var_7 = bool(var_4 == var_6)
    assert var_7 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = module_0._site_path(var_0)
    var_2 = len(var_1)
    var_3 = 0
    var_4 = var_2 > var_3
    var_5 = bool(var_4 or var_1 == '')
    assert var_5 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_gen_api_creates_directory_if_not_exists. Retrieved 20/29 statements.
# Partially parsed test_gen_api_dry_run_does_not_write. Retrieved 17/23 statements.
# Partially parsed test_gen_api_returns_list_of_docs. Retrieved 20/26 statements.
# Partially parsed test_gen_api_skips_empty_modules. Retrieved 22/28 statements.
# Partially parsed test_gen_api_adds_pwd_to_sys_path. Retrieved 20/27 statements.
# Partially parsed test_gen_api_generates_correct_filename. Retrieved 19/25 statements.
# Partially parsed test_gen_api_includes_title_in_output. Retrieved 17/21 statements.
# Partially parsed test_gen_api_respects_level_parameter. Retrieved 18/22 statements.
# Partially parsed test_gen_api_empty_root_names. Retrieved 9/11 statements.


def test_case_0():
    var_0 = "Test that gen_api creates prefix directory if it doesn't exist."
    var_1 = 'new_docs'
    var_2 = 'apimd.loader.isdir'
    var_3 = False
    var_4 = lambda x: var_3
    var_5 = 'apimd.loader.mkdir'
    var_6 = None
    var_7 = lambda x: var_6
    var_8 = 'apimd.loader.loader'
    var_9 = '# Test Doc'
    var_10 = lambda *args, **kwargs: var_9
    var_11 = 'apimd.loader._site_path'
    var_12 = '/fake/path'
    var_13 = lambda x: var_12
    var_14 = 'apimd.loader._write'
    var_15 = lambda *args: var_6
    var_16 = 'Test'
    var_17 = 'test_module'
    var_18 = {var_16: var_17}
    var_19 = True
    var_20 = '# Test Doc'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that gen_api with dry=True does not write files.'
    var_1 = []
    var_2 = 'apimd.loader.isdir'
    var_3 = True
    var_4 = lambda x: var_3
    var_5 = 'apimd.loader.loader'
    var_6 = '# Test'
    var_7 = lambda *args, **kwargs: var_6
    var_8 = 'apimd.loader._site_path'
    var_9 = '/fake/path'
    var_10 = lambda x: var_9
    var_11 = 'apimd.loader._write'
    var_12 = 'Test'
    var_13 = 'test_module'
    var_14 = {var_12: var_13}
    var_15 = module_0.gen_api(var_14, dry=var_3)
    var_16 = len(var_1)
    assert var_16 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that gen_api returns list of generated documents.'
    var_1 = 'apimd.loader.isdir'
    var_2 = True
    var_3 = lambda x: var_2
    var_4 = 'apimd.loader.loader'
    var_5 = '# Content'
    var_6 = lambda *args, **kwargs: var_5
    var_7 = 'apimd.loader._site_path'
    var_8 = '/fake/path'
    var_9 = lambda x: var_8
    var_10 = 'apimd.loader._write'
    var_11 = None
    var_12 = lambda *args: var_11
    var_13 = 'Module1'
    var_14 = 'Module2'
    var_15 = 'mod1'
    var_16 = 'mod2'
    var_17 = {var_13: var_15, var_14: var_16}
    var_18 = module_0.gen_api(var_17, dry=var_2)
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = '# Module1 API'
    var_21 = bool('# Module1 API' in var_18[0])
    assert var_21 is True
    var_22 = '# Module2 API'
    var_23 = bool('# Module2 API' in var_18[1])
    assert var_23 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that gen_api skips modules with empty documentation.'
    var_1 = []
    var_2 = 'apimd.loader.isdir'
    var_3 = True
    var_4 = lambda x: var_3
    var_5 = 'apimd.loader.loader'
    var_6 = 'empty_mod'
    var_7 = ''
    var_8 = '# Content'
    var_9 = lambda name, *args, **kwargs: var_7 if name == var_6 else var_8
    var_10 = 'apimd.loader._site_path'
    var_11 = '/fake/path'
    var_12 = lambda x: var_11
    var_13 = 'apimd.loader._write'
    var_14 = 'Empty'
    var_15 = 'Valid'
    var_16 = 'valid_mod'
    var_17 = {var_14: var_6, var_15: var_16}
    var_18 = False
    var_19 = module_0.gen_api(var_17, dry=var_18)
    var_20 = len(var_19)
    assert var_20 == 1
    var_21 = len(var_1)
    assert var_21 == 1

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that gen_api adds pwd to sys.path when provided.'
    var_1 = []
    var_2 = 'apimd.loader.isdir'
    var_3 = True
    var_4 = lambda x: var_3
    var_5 = 'apimd.loader.loader'
    var_6 = '# Test'
    var_7 = lambda *args, **kwargs: var_6
    var_8 = 'apimd.loader._site_path'
    var_9 = '/fake/path'
    var_10 = lambda x: var_9
    var_11 = 'apimd.loader._write'
    var_12 = None
    var_13 = lambda *args: var_12
    var_14 = 'apimd.loader.sys_path.append'
    var_15 = 'Test'
    var_16 = 'test_module'
    var_17 = {var_15: var_16}
    var_18 = '/custom/path'
    var_19 = module_0.gen_api(var_17, var_18, dry=var_3)
    var_20 = '/custom/path'
    var_21 = bool('/custom/path' in var_1)
    assert var_21 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that gen_api generates filenames with underscores replaced by dashes.'
    var_1 = []
    var_2 = 'apimd.loader.isdir'
    var_3 = True
    var_4 = lambda x: var_3
    var_5 = 'apimd.loader.loader'
    var_6 = '# Test'
    var_7 = lambda *args, **kwargs: var_6
    var_8 = 'apimd.loader._site_path'
    var_9 = '/fake/path'
    var_10 = lambda x: var_9
    var_11 = 'apimd.loader._write'
    var_12 = 'Test'
    var_13 = 'test_module_name'
    var_14 = {var_12: var_13}
    var_15 = 'docs'
    var_16 = False
    var_17 = module_0.gen_api(var_14, prefix=var_15, dry=var_16)
    var_18 = len(var_1)
    assert var_18 == 1
    var_19 = 'test-module-name-api.md'
    var_20 = bool('test-module-name-api.md' in var_1[0])
    assert var_20 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that gen_api includes title in generated documentation.'
    var_1 = 'apimd.loader.isdir'
    var_2 = True
    var_3 = lambda x: var_2
    var_4 = 'apimd.loader.loader'
    var_5 = 'content'
    var_6 = lambda *args, **kwargs: var_5
    var_7 = 'apimd.loader._site_path'
    var_8 = '/fake/path'
    var_9 = lambda x: var_8
    var_10 = 'apimd.loader._write'
    var_11 = None
    var_12 = lambda *args: var_11
    var_13 = 'MyTitle'
    var_14 = 'mymodule'
    var_15 = {var_13: var_14}
    var_16 = module_0.gen_api(var_15, level=var_2, dry=var_2)
    var_17 = '# MyTitle API'
    var_18 = bool('# MyTitle API' in var_16[0])
    assert var_18 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that gen_api respects the level parameter for heading.'
    var_1 = 'apimd.loader.isdir'
    var_2 = True
    var_3 = lambda x: var_2
    var_4 = 'apimd.loader.loader'
    var_5 = 'content'
    var_6 = lambda *args, **kwargs: var_5
    var_7 = 'apimd.loader._site_path'
    var_8 = '/fake/path'
    var_9 = lambda x: var_8
    var_10 = 'apimd.loader._write'
    var_11 = None
    var_12 = lambda *args: var_11
    var_13 = 'MyTitle'
    var_14 = 'mymodule'
    var_15 = {var_13: var_14}
    var_16 = 3
    var_17 = module_0.gen_api(var_15, level=var_16, dry=var_2)
    var_18 = '### MyTitle API'
    var_19 = bool('### MyTitle API' in var_17[0])
    assert var_19 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that gen_api handles empty root_names dictionary.'
    var_1 = 'apimd.loader.isdir'
    var_2 = True
    var_3 = lambda x: var_2
    var_4 = 'apimd.loader._write'
    var_5 = None
    var_6 = lambda *args: var_5
    var_7 = {}
    var_8 = module_0.gen_api(var_7, dry=var_2)
    var_9 = bool(var_8 == [])
    assert var_9 is True



# Parsed testcases at query #21
#--------------------------




def test_case_0():
    var_0 = "Test that the predicate at line 13 evaluates to False when ext is not '.py'."
    var_1 = '.pyi'
    var_2 = '.py'
    var_3 = var_1 == var_2
    assert var_3 is False



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_site_path_with_valid_package. Retrieved 2/5 statements.
# Partially parsed test_site_path_returns_string. Retrieved 2/3 statements.
# Partially parsed test_site_path_with_installed_package. Retrieved 2/3 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = module_0._site_path(var_0)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_package_xyz_123'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'sys'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'json'
    var_1 = module_0._site_path(var_0)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = module_0._site_path(var_0)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_loader_pure_py_false_condition. Retrieved 8/20 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 15 (if pure_py:) evaluates to False.'
    var_1 = 'test_module'
    var_2 = '/path/to/test_module'
    var_3 = (var_1, var_2)
    var_4 = '/root'
    var_5 = '/pwd'
    var_6 = True
    var_7 = module_0.loader(var_4, var_5, var_6, var_6, var_6)
    assert var_7 == 'compiled_output'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_loader_predicate_false_when_pure_py_false. Retrieved 10/24 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 15 evaluates to False when pure_py is False.'
    var_1 = 'test_module'
    var_2 = '/fake/path'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = '/fake/root'
    var_6 = '/fake/pwd'
    var_7 = False
    var_8 = 1
    var_9 = module_0.loader(var_5, var_6, var_7, var_8, var_7)
    assert var_9 == ''



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_read_returns_file_content. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/7 statements.
# Partially parsed test_write_overwrites_existing_file. Retrieved 4/10 statements.
# Partially parsed test_write_handles_empty_string. Retrieved 3/7 statements.
# Partially parsed test_write_handles_multiline_content. Retrieved 3/7 statements.
# Partially parsed test_write_handles_special_characters. Retrieved 3/7 statements.
# Partially parsed test_write_handles_unicode_content. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'First content'
    var_2 = 'Second content'
    var_3 = 'utf-8'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = ''
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Line 1\nLine 2\nLine 3'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = "Special chars: !@#$%^&*()_+-=[]{}|;:',.<>?/~`"
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Unicode: 你好世界 🌍 Ñoño'
    var_2 = 'utf-8'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_read_file_opens_in_read_mode. Retrieved 1/11 statements.


def test_case_0():
    var_0 = 'test content'
    var_1 = bool(var_0)
    assert var_1 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_load_module_success. Retrieved 8/14 statements.
# Partially parsed test_load_module_import_error. Retrieved 4/10 statements.
# Partially parsed test_load_module_with_docstring. Retrieved 4/10 statements.
# Partially parsed test_load_module_no_loader. Retrieved 5/15 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module.py'
    var_1 = '"""Test module docstring."""\ndef foo(): pass'
    var_2 = module_0.Parser()
    var_3 = 'test_module'
    var_4 = var_2.docstring
    var_5 = len(var_4)
    var_6 = 0
    var_7 = var_5 >= var_6
    var_8 = bool('test_module' in var_2.docstring or var_7)
    assert var_8 is True

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'nonexistent_module'
    var_2 = '/nonexistent/path.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_mod.py'
    var_1 = '"""Docstring."""'
    var_2 = module_0.Parser()
    var_3 = 'nonexistent.submodule'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'documented.py'
    var_1 = '"""Module with documentation."""\n\ndef func():\n    """Function doc."""\n    pass'
    var_2 = module_0.Parser()
    var_3 = 'documented'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'pass'
    var_2 = 'importlib.util.spec_from_file_location'
    var_3 = module_0.Parser()
    var_4 = 'test'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_write_creates_file_with_correct_content. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = bool(var_0 == var_1)
    assert var_2 is True



# Parsed testcases at query #30
#--------------------------




def test_case_0():
    var_0 = 'Test that the predicate at line 13 (ext == ".py") evaluates to False.'
    var_1 = '.pyi'
    var_2 = '.py'
    var_3 = var_1 == var_2
    assert var_3 is False



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_load_module_with_valid_spec_and_loader. Retrieved 4/15 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module.py'
    var_1 = '"""Test module docstring."""\ndef test_func():\n    """Test function."""\n    pass\n'
    var_2 = module_0.Parser()
    var_3 = 'test_module'



# Parsed testcases at query #32
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 22 evaluates to True by iterating root_names.'
    var_1 = 'Module A'
    var_2 = 'Module B'
    var_3 = 'module_a'
    var_4 = 'module_b'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = True
    var_7 = module_0.gen_api(var_5, dry=var_6)
    var_8 = len(var_7)
    assert var_8 == 2
    var_9 = '# Module A API'
    var_10 = bool('# Module A API' in var_7[0])
    assert var_10 is True
    var_11 = '# Module B API'
    var_12 = bool('# Module B API' in var_7[1])
    assert var_12 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_write_creates_file_with_correct_content. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'
    var_2 = "\ndef _write(path: str, doc: str) -> None:\n    with open(path, 'w+', encoding='utf-8') as f:\n        f.write(doc)\n"
    var_3 = exec(var_2)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_load_module_success. Retrieved 8/24 statements.
# Partially parsed test_load_module_parent_import_error. Retrieved 4/10 statements.
# Partially parsed test_load_module_invalid_spec. Retrieved 11/26 statements.
# Partially parsed test_load_module_with_docstring. Retrieved 8/24 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = ''
    var_3 = 'test_module.py'
    var_4 = '"""Test module."""\ndef test_func():\n    """Test function."""\n    pass'
    var_5 = 0
    var_6 = module_0.Parser()
    var_7 = 'test_pkg.test_module'
    var_8 = 'test_pkg.test_module'
    var_9 = bool('test_pkg.test_module' in var_6.docstring)
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module.py'
    var_1 = '"""Test module."""'
    var_2 = module_0.Parser()
    var_3 = 'nonexistent_parent.test_module'

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = ''
    var_3 = 'test_module.py'
    var_4 = '"""Test module."""'
    var_5 = 0
    var_6 = module_0.Parser()
    var_7 = 'test_pkg.nonexistent'
    var_8 = 'nonexistent.py'
    var_9 = str(var_1)
    var_10 = module_1._load_module(var_7, var_9, var_6)
    assert var_10 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = ''
    var_3 = 'test_module.py'
    var_4 = '"""Module docstring.\n\nThis is a test module.\n"""\n\ndef func():\n    """Function docstring."""\n    pass'
    var_5 = 0
    var_6 = module_0.Parser()
    var_7 = 'test_pkg.test_module'
    var_8 = 'test_pkg.test_module'
    var_9 = bool('test_pkg.test_module' in var_6.docstring)
    assert var_9 is True



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_gen_api_iterates_over_root_names. Retrieved 9/18 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Ensure that the predicate at line 22 evaluates to True by iterating over root_names.'
    var_1 = 'TestTitle'
    var_2 = 'AnotherTitle'
    var_3 = 'test_module'
    var_4 = 'another_module'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = True
    var_7 = module_0.gen_api(var_5, dry=var_6)
    var_8 = len(var_7)
    assert var_8 == 0



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_gen_api_iterates_over_root_names. Retrieved 7/15 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Title1'
    var_1 = 'Title2'
    var_2 = 'module1'
    var_3 = 'module2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True
    var_6 = module_0.gen_api(var_4, dry=var_5)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_write_predicate_false. Retrieved 1/12 statements.


def test_case_0():
    var_0 = 'test content'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_load_module_success. Retrieved 6/15 statements.
# Partially parsed test_load_module_import_error. Retrieved 6/15 statements.
# Partially parsed test_load_module_invalid_spec. Retrieved 9/19 statements.
# Partially parsed test_load_module_invalid_loader. Retrieved 8/22 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test _load_module successfully loads a module and loads docstring.'
    var_1 = 'test_module.py'
    var_2 = '"""Test module docstring."""\ndef test_func():\n    """Test function."""\n    pass\n'
    var_3 = module_0.Parser()
    var_4 = 'builtins.__import__'
    var_5 = 'test_module'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test _load_module returns False when parent import fails.'
    var_1 = 'nonexistent.py'
    var_2 = '"""Test."""\n'
    var_3 = module_0.Parser()
    var_4 = 'builtins.__import__'
    var_5 = 'nonexistent'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test _load_module returns False when spec is None.'
    var_1 = 'invalid.py'
    var_2 = '"""Test."""\n'
    var_3 = module_0.Parser()
    var_4 = 'builtins.__import__'
    var_5 = 'importlib.util.spec_from_file_location'
    var_6 = None
    var_7 = lambda n, p: var_6
    var_8 = 'invalid'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test _load_module returns False when loader is not a Loader instance.'
    var_1 = 'test.py'
    var_2 = '"""Test."""\n'
    var_3 = module_0.Parser()
    var_4 = None
    var_5 = 'builtins.__import__'
    var_6 = 'importlib.util.spec_from_file_location'
    var_7 = 'test'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_write_predicate_evaluates_to_false. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'test content'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_load_module_predicate_at_line_9_true. Retrieved 12/32 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '# test package\n'
    var_3 = 'test_module.py'
    var_4 = '"""Test module docstring."""\ndef test_func():\n    """Test function."""\n    pass\n'
    var_5 = 0
    var_6 = module_0.Parser()
    var_7 = 'test_pkg.test_module'
    var_8 = var_6.doc
    var_9 = len(var_8)
    var_10 = 0
    var_11 = var_9 > var_10
    var_12 = bool('test_pkg.test_module' in var_6.docstring or var_11)
    assert var_12 is True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/7 statements.
# Partially parsed test_write_overwrites_existing_file. Retrieved 4/9 statements.
# Partially parsed test_write_empty_string. Retrieved 3/7 statements.
# Partially parsed test_write_multiline_content. Retrieved 3/7 statements.
# Partially parsed test_write_special_characters. Retrieved 3/7 statements.
# Partially parsed test_write_unicode_content. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Old content'
    var_2 = 'utf-8'
    var_3 = 'New content'

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = ''
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Line 1\nLine 2\nLine 3'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = "Special chars: !@#$%^&*()_+-=[]{}|;:',.<>?/~`"
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Unicode: 你好世界 🌍 Привет'
    var_2 = 'utf-8'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_read_returns_file_content. Retrieved 2/6 statements.
# Partially parsed test_read_empty_file. Retrieved 2/6 statements.
# Partially parsed test_read_multiline_file. Retrieved 2/6 statements.
# Partially parsed test_read_file_with_special_characters. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'

def test_case_0():
    var_0 = 'empty.txt'
    var_1 = ''

def test_case_0():
    var_0 = 'multiline.txt'
    var_1 = 'Line 1\nLine 2\nLine 3\n'

def test_case_0():
    var_0 = 'special.txt'
    var_1 = "Special chars: !@#$%^&*()_+-=[]{}|;:',.<>?/"



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_load_module_predicate_at_line_9. Retrieved 5/22 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 9 evaluates to True.'
    var_1 = module_0.Parser()
    var_2 = 'test.module'
    var_3 = '/path/to/module.py'
    var_4 = module_1._load_module(var_2, var_3, var_1)
    assert var_4 is True



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_read_existing_file. Retrieved 2/6 statements.
# Partially parsed test_read_empty_file. Retrieved 2/6 statements.
# Partially parsed test_read_multiline_file. Retrieved 2/6 statements.
# Partially parsed test_read_file_with_special_characters. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'test_script.txt'
    var_1 = "print('Hello, World!')"

def test_case_0():
    var_0 = 'empty_script.txt'
    var_1 = ''

def test_case_0():
    var_0 = 'multiline_script.txt'
    var_1 = 'line1\nline2\nline3'

def test_case_0():
    var_0 = 'special_script.txt'
    var_1 = "x = 'special chars: !@#$%^&*()'"

import apimd.loader as module_0

def test_case_0():
    var_0 = '/nonexistent/path/to/file.txt'
    var_1 = module_0._read(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_read_returns_file_content. Retrieved 1/16 statements.


def test_case_0():
    var_0 = 'test script content'



# Parsed testcases at query #46
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 22 (for title, name in root_names.items()) evaluates to False when root_names is empty.'
    var_1 = {}
    var_2 = 'test_docs'
    var_3 = True
    var_4 = module_0.gen_api(var_1, prefix=var_2, dry=var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True



# Parsed testcases at query #47
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 22 (for title, name in root_names.items()) evaluates to False.'
    var_1 = {}
    var_2 = '/tmp/test_docs'
    var_3 = True
    var_4 = module_0.gen_api(var_1, prefix=var_2, dry=var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_gen_api_basic. Retrieved 9/21 statements.
# Partially parsed test_gen_api_multiple_modules. Retrieved 9/20 statements.
# Partially parsed test_gen_api_empty_doc. Retrieved 9/18 statements.
# Partially parsed test_gen_api_creates_directory. Retrieved 7/18 statements.
# Partially parsed test_gen_api_with_level. Retrieved 9/18 statements.
# Partially parsed test_gen_api_writes_file. Retrieved 8/20 statements.
# Partially parsed test_gen_api_sys_path_append. Retrieved 8/20 statements.


def test_case_0():
    var_0 = 'Test gen_api with basic parameters.'
    var_1 = 'docs'
    var_2 = '# Module\nDocumentation'
    var_3 = 'Test'
    var_4 = 'test_module'
    var_5 = {var_3: var_4}
    var_6 = None
    var_7 = True
    assert var_7 == 1
    var_8 = False
    var_9 = '# Test API'

def test_case_0():
    var_0 = 'Test gen_api with multiple root modules.'
    var_1 = 'docs'
    var_2 = '# Module\nDocumentation'
    var_3 = 'Module1'
    var_4 = 'Module2'
    var_5 = 'mod1'
    var_6 = 'mod2'
    var_7 = {var_3: var_5, var_4: var_6}
    assert var_7 == 2
    var_8 = True
    var_9 = '# Module1 API'
    var_10 = '# Module2 API'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api when loader returns empty documentation.'
    var_1 = 'docs'
    var_2 = '   \n  \n  '
    var_3 = 'Test'
    var_4 = 'test_module'
    var_5 = {var_3: var_4}
    var_6 = True
    var_7 = module_0.gen_api(var_5, prefix=var_2, dry=var_6)
    var_8 = len(var_7)
    assert var_8 == 0

def test_case_0():
    var_0 = "Test gen_api creates prefix directory if it doesn't exist."
    var_1 = 'new_docs'
    var_2 = '# Module\nDocumentation'
    var_3 = 'Test'
    var_4 = 'test_module'
    var_5 = {var_3: var_4}
    var_6 = True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api with different heading levels.'
    var_1 = 'docs'
    var_2 = 'Content'
    var_3 = 'Test'
    var_4 = 'test_module'
    var_5 = {var_3: var_4}
    var_6 = 3
    var_7 = True
    var_8 = module_0.gen_api(var_5, prefix=var_2, level=var_6, dry=var_7)
    var_9 = '### Test API'
    var_10 = bool('### Test API' in var_8[0])
    assert var_10 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api writes file when dry=False.'
    var_1 = 'docs'
    var_2 = '# Module\nDocumentation'
    var_3 = 'Test'
    var_4 = 'test_module'
    var_5 = {var_3: var_4}
    var_6 = False
    var_7 = module_0.gen_api(var_5, prefix=var_2, dry=var_6)
    var_8 = 'test-module-api.md'

def test_case_0():
    var_0 = 'Test gen_api appends pwd to sys.path when provided.'
    var_1 = 'docs'
    var_2 = '# Module\nDocumentation'
    var_3 = 'Test'
    var_4 = 'test_module'
    var_5 = {var_3: var_4}
    var_6 = '/custom/path'
    var_7 = True



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_write_creates_file_with_correct_content. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'
    var_2 = bool(var_0 == var_1)
    assert var_2 is True



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/8 statements.
# Partially parsed test_write_overwrites_existing_file. Retrieved 4/9 statements.
# Partially parsed test_write_empty_string. Retrieved 3/8 statements.
# Partially parsed test_write_multiline_content. Retrieved 3/7 statements.
# Partially parsed test_write_unicode_content. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'old content'
    var_2 = 'utf-8'
    var_3 = 'new content'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = ''
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'line1\nline2\nline3'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello 世界 🌍'
    var_2 = 'utf-8'



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_read_file_predicate. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 'test content'
    var_1 = bool(var_0)
    assert var_1 is True



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_load_module_returns_false_when_loader_not_instance_of_loader. Retrieved 4/10 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '/fake/path.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '/fake/path.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_read_file_predicate. Retrieved 1/12 statements.


def test_case_0():
    var_0 = 'test content'



# Parsed testcases at query #54
#--------------------------




import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 9 evaluates to False when s is None.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = '/fake/path.py'
    var_4 = module_1._load_module(var_2, var_3, var_1)
    assert var_4 is False



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_loader_predicate_line_15_false. Retrieved 12/23 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 15 evaluates to False when .py file is not found.'
    var_1 = 'test_module'
    var_2 = '/path/to/test_module'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = 'def foo(): pass'
    var_6 = True
    var_7 = '/root'
    var_8 = '/pwd'
    var_9 = False
    var_10 = 1
    var_11 = module_0.loader(var_7, var_8, var_9, var_10, var_9)
    assert var_11 == 'compiled_output'



# Parsed testcases at query #56
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 25 evaluates to True when doc.strip() is empty.'
    var_1 = 'Test'
    var_2 = 'test_module'
    var_3 = {var_1: var_2}
    var_4 = 'docs'
    var_5 = module_0.gen_api(var_3, prefix=var_4)
    var_6 = bool(var_5 == [])
    assert var_6 is True



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_read_returns_file_contents. Retrieved 2/6 statements.
# Partially parsed test_read_with_multiline_content. Retrieved 2/6 statements.
# Partially parsed test_read_with_empty_file. Retrieved 2/6 statements.
# Partially parsed test_read_with_special_characters. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'test_script.txt'
    var_1 = "print('Hello, World!')"

def test_case_0():
    var_0 = 'multiline_script.py'
    var_1 = 'line1\nline2\nline3'

def test_case_0():
    var_0 = 'empty_script.txt'
    var_1 = ''

def test_case_0():
    var_0 = 'special_chars.txt'
    var_1 = "Special chars: !@#$%^&*()_+-=[]{}|;:',.<>?/~`"



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_gen_api_predicate_line_25_evaluates_to_true. Retrieved 14/31 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 25 evaluates to True when doc.strip() returns empty string.'
    var_1 = 'apimd.loader.loader'
    var_2 = 'apimd.loader.isdir'
    var_3 = 'apimd.loader.mkdir'
    var_4 = 'apimd.loader._site_path'
    var_5 = 'apimd.loader._write'
    var_6 = 'apimd.loader.sys_path'
    var_7 = []
    var_8 = 'Test'
    var_9 = 'test_module'
    var_10 = {var_8: var_9}
    var_11 = 'docs'
    var_12 = False
    var_13 = module_0.gen_api(var_10, prefix=var_11, dry=var_12)
    var_14 = bool(var_13 == [])
    assert var_14 is True



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/7 statements.
# Partially parsed test_write_overwrites_existing_file. Retrieved 4/9 statements.
# Partially parsed test_write_empty_string. Retrieved 3/7 statements.
# Partially parsed test_write_multiline_content. Retrieved 3/7 statements.
# Partially parsed test_write_unicode_content. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'old content'
    var_2 = 'utf-8'
    var_3 = 'new content'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = ''
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Line 1\nLine 2\nLine 3'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello 世界 🌍'
    var_2 = 'utf-8'



# Parsed testcases at query #60
#--------------------------




def test_case_0():
    var_0 = "Test that the predicate at line 13 evaluates to False when ext is not '.py'"
    var_1 = '.pyi'
    assert var_1 == '.py'
    var_2 = '.py'
    assert var_2 is False



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_write_predicate_evaluates_to_false. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'test content'
    var_2 = 0



# Parsed testcases at query #62
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 25 evaluates to True when doc.strip() is empty.'
    var_1 = 'Test Module'
    var_2 = 'test_module'
    var_3 = {var_1: var_2}
    var_4 = 'docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = bool(var_7 == [])
    assert var_8 is True



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_gen_api_basic. Retrieved 8/16 statements.
# Partially parsed test_gen_api_creates_directory. Retrieved 8/14 statements.
# Partially parsed test_gen_api_dry_mode. Retrieved 9/17 statements.
# Partially parsed test_gen_api_multiple_roots. Retrieved 13/20 statements.
# Partially parsed test_gen_api_with_underscore_in_name. Retrieved 8/15 statements.
# Partially parsed test_gen_api_different_levels. Retrieved 8/16 statements.
# Partially parsed test_gen_api_returns_sequence. Retrieved 10/17 statements.


def test_case_0():
    var_0 = 'Test gen_api with basic parameters.'
    var_1 = 'docs'
    var_2 = 'Test Module'
    var_3 = 'nonexistent_module'
    var_4 = {var_2: var_3}
    var_5 = None
    var_6 = True
    var_7 = False

def test_case_0():
    var_0 = "Test gen_api creates prefix directory if it doesn't exist."
    var_1 = 'new_docs'
    var_2 = 'Test Module'
    var_3 = 'nonexistent_module'
    var_4 = {var_2: var_3}
    var_5 = None
    var_6 = True
    var_7 = False

def test_case_0():
    var_0 = "Test gen_api in dry mode doesn't write files."
    var_1 = 'docs'
    var_2 = 'Test Module'
    var_3 = 'nonexistent_module'
    var_4 = {var_2: var_3}
    var_5 = None
    var_6 = True
    var_7 = 2
    var_8 = '*.md'

def test_case_0():
    var_0 = 'Test gen_api with multiple root names.'
    var_1 = 'docs'
    var_2 = 'Module A'
    var_3 = 'Module B'
    var_4 = 'Module C'
    var_5 = 'nonexistent_a'
    var_6 = 'nonexistent_b'
    var_7 = 'nonexistent_c'
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = None
    var_10 = False
    var_11 = 1
    var_12 = True

def test_case_0():
    var_0 = 'Test gen_api converts underscores to hyphens in filenames.'
    var_1 = 'docs'
    var_2 = 'My Module'
    var_3 = 'my_test_module'
    var_4 = {var_2: var_3}
    var_5 = None
    var_6 = True
    var_7 = False

def test_case_0():
    var_0 = 'Test gen_api with different heading levels.'
    var_1 = 'docs'
    var_2 = 'Test'
    var_3 = 'nonexistent'
    var_4 = {var_2: var_3}
    var_5 = None
    var_6 = True
    var_7 = False

def test_case_0():
    var_0 = 'Test gen_api returns a sequence.'
    var_1 = 'docs'
    var_2 = 'Module'
    var_3 = 'nonexistent_module'
    var_4 = {var_2: var_3}
    var_5 = None
    var_6 = True
    var_7 = False
    var_8 = '__getitem__'
    var_9 = '__len__'



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_gen_api_predicate_line_25_evaluates_to_true. Retrieved 16/33 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 25 evaluates to True when doc is empty/whitespace.'
    var_1 = []
    var_2 = 'apimd.loader.loader'
    var_3 = 'apimd.loader._site_path'
    var_4 = 'apimd.loader.isdir'
    var_5 = 'apimd.loader.logger'
    var_6 = 'TestModule'
    var_7 = 'test_module'
    var_8 = {var_6: var_7}
    var_9 = 'docs'
    var_10 = True
    var_11 = module_0.gen_api(var_8, prefix=var_9, dry=var_10)
    var_12 = 0
    var_13 = 'warning'
    var_14 = [call for call in var_1 if call[var_12] == var_13]
    var_15 = len(var_14)
    var_16 = bool(var_15 > 0)
    assert var_16 is True
    var_17 = 'can not be found'
    var_18 = bool('can not be found' in var_14[0][1])
    assert var_18 is True
    var_19 = bool(var_11 == [])
    assert var_19 is True



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_read_file_opens_in_read_mode. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 'test content'
    var_1 = bool(var_0)
    assert var_1 is True



