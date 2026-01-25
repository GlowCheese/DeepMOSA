####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_gen_api_with_dry_run. Retrieved 16/29 statements.
# Partially parsed test_gen_api_without_dry_run. Retrieved 16/28 statements.
# Partially parsed test_gen_api_empty_documentation. Retrieved 16/28 statements.
# Partially parsed test_gen_api_multiple_modules. Retrieved 18/30 statements.
# Partially parsed test_gen_api_with_pwd. Retrieved 17/34 statements.


def test_case_0():
    var_0 = 'Test gen_api with dry run mode.'
    var_1 = 'docs'
    var_2 = 'apimd.loader._site_path'
    var_3 = 'apimd.loader.loader'
    var_4 = '## Module\n\nSome docs'
    var_5 = lambda root, pwd, link, level, toc: var_4
    var_6 = 'apimd.loader.mkdir'
    var_7 = None
    var_8 = lambda x: var_7
    var_9 = 'apimd.loader.isdir'
    var_10 = True
    var_11 = lambda x: var_10
    var_12 = 'Test Module'
    var_13 = 'test_module'
    var_14 = {var_12: var_13}
    var_15 = False
    var_16 = '# Test Module API'
    var_17 = '## Module'

def test_case_0():
    var_0 = 'Test gen_api without dry run mode (write to file).'
    var_1 = 'docs'
    var_2 = 'apimd.loader._site_path'
    var_3 = 'apimd.loader.loader'
    var_4 = '## Module\n\nSome docs'
    var_5 = lambda root, pwd, link, level, toc: var_4
    var_6 = 'apimd.loader.mkdir'
    var_7 = None
    var_8 = lambda x: var_7
    var_9 = 'apimd.loader.isdir'
    var_10 = True
    var_11 = lambda x: var_10
    var_12 = 'Test Module'
    var_13 = 'test_module'
    var_14 = {var_12: var_13}
    var_15 = False
    var_16 = '# Test Module API'

def test_case_0():
    var_0 = 'Test gen_api when loader returns empty documentation.'
    var_1 = 'docs'
    var_2 = 'apimd.loader._site_path'
    var_3 = 'apimd.loader.loader'
    var_4 = '   \n\n  '
    var_5 = lambda root, pwd, link, level, toc: var_4
    var_6 = 'apimd.loader.mkdir'
    var_7 = None
    var_8 = lambda x: var_7
    var_9 = 'apimd.loader.isdir'
    var_10 = True
    var_11 = lambda x: var_10
    var_12 = 'Test Module'
    var_13 = 'test_module'
    var_14 = {var_12: var_13}
    var_15 = False

def test_case_0():
    var_0 = 'Test gen_api with multiple modules.'
    var_1 = 'docs'
    var_2 = 'apimd.loader._site_path'
    var_3 = 'apimd.loader.loader'
    var_4 = '## Module\n\nSome docs'
    var_5 = lambda root, pwd, link, level, toc: var_4
    var_6 = 'apimd.loader.mkdir'
    var_7 = None
    var_8 = lambda x: var_7
    var_9 = 'apimd.loader.isdir'
    var_10 = True
    var_11 = lambda x: var_10
    var_12 = 'Module One'
    var_13 = 'Module Two'
    var_14 = 'mod1'
    var_15 = 'mod2'
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = 2
    var_18 = '## Module One API'
    var_19 = '## Module Two API'

def test_case_0():
    var_0 = 'Test gen_api with custom pwd parameter.'
    var_1 = 'docs'
    var_2 = 'custom_path'
    var_3 = 'apimd.loader._site_path'
    var_4 = 'apimd.loader.loader'
    var_5 = '## Module\n\nSome docs'
    var_6 = lambda root, pwd, link, level, toc: var_5
    var_7 = 'apimd.loader.mkdir'
    var_8 = None
    var_9 = lambda x: var_8
    var_10 = 'apimd.loader.isdir'
    var_11 = True
    var_12 = lambda x: var_11
    var_13 = 'Test Module'
    var_14 = 'test_module'
    var_15 = {var_13: var_14}
    var_16 = False



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_loader_with_valid_package. Retrieved 6/14 statements.
# Partially parsed test_loader_with_multiple_modules. Retrieved 8/18 statements.
# Partially parsed test_loader_with_toc_enabled. Retrieved 5/12 statements.
# Partially parsed test_loader_with_link_disabled. Retrieved 6/14 statements.
# Partially parsed test_loader_with_different_level. Retrieved 7/14 statements.
# Partially parsed test_loader_with_nested_modules. Retrieved 8/20 statements.
# Partially parsed test_loader_with_pyi_stub. Retrieved 6/13 statements.
# Partially parsed test_loader_with_class_definition. Retrieved 6/14 statements.
# Partially parsed test_loader_with_all_export. Retrieved 6/13 statements.
# Partially parsed test_loader_with_constants. Retrieved 6/13 statements.
# Partially parsed test_loader_returns_string. Retrieved 6/13 statements.
# Partially parsed test_loader_with_async_functions. Retrieved 6/13 statements.
# Partially parsed test_loader_with_decorators. Retrieved 6/13 statements.
# Partially parsed test_loader_with_type_annotations. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'Test loader with a valid package structure.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = 'def hello(): pass\n'
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader with multiple modules in a package.'
    var_1 = 'multi_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Main module"""\ndef main(): pass\n'
    var_4 = 'sub.py'
    var_5 = '"""Sub module"""\ndef sub_func(): pass\n'
    var_6 = True
    var_7 = False

def test_case_0():
    var_0 = 'Test loader with table of contents enabled.'
    var_1 = 'toc_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Package"""\ndef func(): pass\n'
    var_4 = True
    var_5 = 'Table of contents'

def test_case_0():
    var_0 = 'Test loader with link generation disabled.'
    var_1 = 'nolink_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Package"""\ndef func(): pass\n'
    var_4 = False
    var_5 = 1

def test_case_0():
    var_0 = 'Test loader with different heading level.'
    var_1 = 'level_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Package"""\ndef func(): pass\n'
    var_4 = True
    var_5 = 2
    var_6 = False
    var_7 = '###'

def test_case_0():
    var_0 = 'Test loader with nested module structure.'
    var_1 = 'nested_pkg'
    var_2 = 'sub'
    var_3 = '__init__.py'
    var_4 = '"""Main"""\ndef main(): pass\n'
    var_5 = '"""Sub"""\ndef sub(): pass\n'
    var_6 = True
    var_7 = False

def test_case_0():
    var_0 = 'Test loader with .pyi stub file.'
    var_1 = 'stub_pkg'
    var_2 = '__init__.pyi'
    var_3 = 'def stub_func() -> int: ...\n'
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader with class definitions.'
    var_1 = 'class_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Package"""\nclass MyClass:\n    """A class"""\n    def method(self): pass\n'
    var_4 = True
    var_5 = False
    var_6 = 'class'

def test_case_0():
    var_0 = 'Test loader with __all__ export list.'
    var_1 = 'all_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Package"""\n__all__ = ["public_func"]\ndef public_func(): pass\ndef _private(): pass\n'
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader with module constants.'
    var_1 = 'const_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Package"""\nVERSION = "1.0.0"\nDEBUG: bool = False\n'
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test that loader always returns a string.'
    var_1 = 'return_pkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader with async function definitions.'
    var_1 = 'async_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Package"""\nasync def async_func(): pass\n'
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader with decorated functions.'
    var_1 = 'deco_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Package"""\ndef decorator(f): return f\n@decorator\ndef decorated(): pass\n'
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader with type annotations.'
    var_1 = 'typed_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Package"""\ndef typed_func(x: int, y: str) -> bool: pass\n'
    var_4 = True
    var_5 = False



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/7 statements.
# Partially parsed test_write_overwrites_existing_file. Retrieved 4/9 statements.
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
    var_1 = "Special chars: !@#$%^&*()_+-=[]{}|;:',.<>?/~`"
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Unicode: 你好世界 🌍 Ñoño'
    var_2 = 'utf-8'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_read_returns_file_content. Retrieved 2/6 statements.
# Partially parsed test_read_returns_empty_string_for_empty_file. Retrieved 2/6 statements.
# Partially parsed test_read_returns_multiline_content. Retrieved 2/6 statements.
# Partially parsed test_read_preserves_special_characters. Retrieved 2/6 statements.


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
    var_1 = "Special chars: !@#$%^&*()_+-=[]{}|;:',.<>?/~`"



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_loader_pure_py_false_condition. Retrieved 10/22 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 15 (if pure_py) evaluates to False.'
    var_1 = 'test_module'
    var_2 = '/fake/path/test_module'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = '/root'
    var_6 = '/pwd'
    var_7 = False
    var_8 = 1
    var_9 = module_0.loader(var_5, var_6, var_7, var_8, var_7)
    assert var_9 == 'compiled_output'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_load_module_success. Retrieved 4/22 statements.
# Partially parsed test_load_module_import_error. Retrieved 3/10 statements.
# Partially parsed test_load_module_spec_none. Retrieved 3/10 statements.
# Partially parsed test_load_module_invalid_loader. Retrieved 3/14 statements.
# Partially parsed test_load_module_calls_load_docstring. Retrieved 3/18 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.py'
    var_3 = '"""Test module docstring"""\ndef test_func():\n    pass\n'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'nonexistent.module'
    var_2 = 'test.py'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test.py'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test.py'

import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test.py'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_loader_predicate_line_13_false. Retrieved 6/17 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = "Test that the predicate at line 13 evaluates to False when ext is not '.py'."
    var_1 = '/root'
    var_2 = '/pwd'
    var_3 = False
    var_4 = 1
    var_5 = module_0.loader(var_1, var_2, var_3, var_4, var_3)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_loader_predicate_line_15_false. Retrieved 10/23 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 15 (if pure_py:) evaluates to False.'
    var_1 = 'test_module'
    var_2 = '/fake/path'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = '/fake/root'
    var_6 = '/fake/pwd'
    var_7 = False
    var_8 = 1
    var_9 = module_0.loader(var_5, var_6, var_7, var_8, var_7)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_load_module_success. Retrieved 5/18 statements.
# Partially parsed test_load_module_import_error. Retrieved 5/14 statements.
# Partially parsed test_load_module_invalid_spec. Retrieved 6/19 statements.
# Partially parsed test_load_module_calls_load_docstring. Retrieved 8/21 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module.py'
    var_1 = '"""Test module docstring."""\ndef test_func():\n    """Test function."""\n    pass\n'
    var_2 = module_0.Parser()
    var_3 = 'builtins.__import__'
    var_4 = 'test_module'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module.py'
    var_1 = '"""Test module."""'
    var_2 = module_0.Parser()
    var_3 = 'builtins.__import__'
    var_4 = 'nonexistent.module'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module.py'
    var_1 = '"""Test module."""'
    var_2 = module_0.Parser()
    var_3 = 'apimd.loader.spec_from_file_location'
    var_4 = 'builtins.__import__'
    var_5 = 'test_module'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module.py'
    var_1 = '"""Module docstring."""\ndef func():\n    """Func doc."""\n    pass\n'
    var_2 = module_0.Parser()
    var_3 = []
    var_4 = var_2.load_docstring
    var_5 = 'builtins.__import__'
    var_6 = 'test_module'
    var_7 = len(var_3)
    assert var_7 == 1
    var_8 = var_3[0][0]
    assert var_8 == 'test_module'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_loader_predicate_line_13_false. Retrieved 6/19 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 13 (ext == ".py") evaluates to False.'
    var_1 = '/root'
    var_2 = '/pwd'
    var_3 = False
    var_4 = 1
    var_5 = module_0.loader(var_1, var_2, var_3, var_4, var_3)
    assert var_5 == 'compiled'



# Parsed testcases at query #11
#--------------------------




def test_case_0():
    var_0 = 'Ensure that the predicate at line 13 (ext == ".py") evaluates to False.'
    var_1 = '.pyi'
    var_2 = '.py'
    var_3 = var_1 == var_2
    assert var_3 is False



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_loader_pure_py_false. Retrieved 11/23 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 15 (if pure_py:) evaluates to False.'
    var_1 = 'test_module'
    var_2 = '/fake/path'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = ''
    var_6 = '/root'
    var_7 = '/pwd'
    var_8 = False
    var_9 = 1
    var_10 = module_0.loader(var_6, var_7, var_8, var_9, var_8)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_read_file_opens_in_read_mode. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'test content'



# Parsed testcases at query #14
#--------------------------




import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'nonexistent.module.that.does.not.exist'
    var_2 = '/fake/path.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_site_path_with_valid_module. Retrieved 2/5 statements.
# Partially parsed test_site_path_with_package. Retrieved 2/3 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = module_0._site_path(var_0)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_module_xyz_12345'
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



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_read_returns_file_contents. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_load_module_success. Retrieved 8/26 statements.
# Partially parsed test_load_module_invalid_loader. Retrieved 5/13 statements.
# Partially parsed test_load_module_calls_load_docstring. Retrieved 6/21 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test _load_module successfully loads a module and docstring.'
    var_1 = 'test_module.py'
    var_2 = '"""Test module docstring."""\ndef foo(): pass'
    var_3 = []
    var_4 = 'builtins'
    var_5 = '__import__'
    var_6 = module_0.Parser()
    var_7 = 'test_module'

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module returns False when parent import fails.'
    var_1 = module_0.Parser()
    var_2 = 'nonexistent.module'
    var_3 = '/fake/path.py'
    var_4 = module_1._load_module(var_2, var_3, var_1)
    assert var_4 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module returns False when spec_from_file_location returns None.'
    var_1 = module_0.Parser()
    var_2 = 'sys.fake'
    var_3 = '/nonexistent/path.py'
    var_4 = module_1._load_module(var_2, var_3, var_1)
    assert var_4 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module returns False when loader is not a Loader instance.'
    var_1 = module_0.Parser()
    var_2 = 'sys.fake'
    var_3 = '/nonexistent/path.py'
    var_4 = module_1._load_module(var_2, var_3, var_1)
    assert var_4 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module calls parser.load_docstring when successful.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = '/fake/path.py'
    var_4 = module_1._load_module(var_2, var_3, var_1)
    var_5 = 'test_module'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_load_module_returns_false_when_loader_is_not_instance_of_loader. Retrieved 4/13 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '/fake/path/test_module.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False



# Parsed testcases at query #19
#--------------------------




import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 9 evaluates to False when s is None.'
    var_1 = module_0.Parser()
    var_2 = 'os'
    var_3 = '/fake/path.py'
    var_4 = module_1._load_module(var_2, var_3, var_1)
    assert var_4 is False



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_loader_predicate_line_13_false. Retrieved 9/21 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = "Ensure that the predicate at line 13 evaluates to False when ext is '.pyi'."
    var_1 = 'test_module'
    var_2 = '/path/test_module'
    var_3 = (var_1, var_2)
    var_4 = '/root'
    var_5 = '/pwd'
    var_6 = False
    var_7 = 1
    var_8 = module_0.loader(var_4, var_5, var_6, var_7, var_6)



# Parsed testcases at query #21
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



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_read_existing_file. Retrieved 2/6 statements.
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
    var_1 = 'Line 1\nLine 2\nLine 3'

def test_case_0():
    var_0 = 'special.txt'
    var_1 = 'Special chars: !@#$%^&*()\n\t✓'



# Parsed testcases at query #23
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



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_load_module_success. Retrieved 7/20 statements.
# Partially parsed test_load_module_import_error. Retrieved 4/10 statements.
# Partially parsed test_load_module_invalid_spec. Retrieved 7/16 statements.
# Partially parsed test_load_module_with_docstring. Retrieved 7/19 statements.
# Partially parsed test_load_module_empty_module. Retrieved 6/18 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module.py'
    var_1 = '"""Test module docstring."""\ndef test_func():\n    """Test function."""\n    pass\n'
    var_2 = 'test_pkg'
    var_3 = '__init__.py'
    var_4 = ''
    var_5 = module_0.Parser()
    var_6 = 'test_pkg.test_module'
    var_7 = 'test_pkg.test_module'
    var_8 = bool('test_pkg.test_module' in var_5.docstring)
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'orphan_module.py'
    var_1 = '"""Orphan module."""'
    var_2 = module_0.Parser()
    var_3 = 'nonexistent.orphan_module'

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'valid_pkg'
    var_1 = '__init__.py'
    var_2 = ''
    var_3 = module_0.Parser()
    var_4 = 'valid_pkg.nonexistent'
    var_5 = '/nonexistent/path/file.py'
    var_6 = module_1._load_module(var_4, var_5, var_3)
    assert var_6 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = 'doc_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Package docstring."""'
    var_3 = 'doc_module.py'
    var_4 = '"""Module with documentation.\n\nThis is a test module.\n"""\n\nclass TestClass:\n    """Test class docstring."""\n    pass\n'
    var_5 = module_0.Parser()
    var_6 = 'doc_pkg.doc_module'
    var_7 = 'doc_pkg.doc_module'
    var_8 = bool('doc_pkg.doc_module' in var_5.docstring)
    assert var_8 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'empty_pkg'
    var_1 = '__init__.py'
    var_2 = ''
    var_3 = 'empty_module.py'
    var_4 = module_0.Parser()
    var_5 = 'empty_pkg.empty_module'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_gen_api_predicate_line_22. Retrieved 7/20 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 22 (for title, name in root_names.items()) evaluates to True.'
    var_1 = 'TestTitle'
    var_2 = 'AnotherTitle'
    assert var_2 == 2
    var_3 = 'test_module'
    var_4 = 'another_module'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = True
    var_7 = 'TestTitle API'
    var_8 = 'AnotherTitle API'



# Parsed testcases at query #26
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = False
    var_1 = 'nonexistent_file.txt'
    var_2 = module_0._read(var_1)
    var_3 = True
    assert var_3 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_read_returns_file_contents. Retrieved 2/6 statements.
# Partially parsed test_read_empty_file. Retrieved 2/6 statements.
# Partially parsed test_read_multiline_file. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'test_script.txt'
    var_1 = 'Hello, World!\nThis is a test.'

def test_case_0():
    var_0 = 'empty_script.txt'
    var_1 = ''

def test_case_0():
    var_0 = 'multiline_script.txt'
    var_1 = 'line1\nline2\nline3\n'

import apimd.loader as module_0

def test_case_0():
    var_0 = '/nonexistent/path/to/file.txt'
    var_1 = module_0._read(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/8 statements.
# Partially parsed test_write_overwrites_existing_file. Retrieved 4/10 statements.
# Partially parsed test_write_handles_empty_string. Retrieved 3/8 statements.
# Partially parsed test_write_handles_multiline_content. Retrieved 3/8 statements.
# Partially parsed test_write_handles_special_characters. Retrieved 3/8 statements.


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
    var_1 = 'Special chars: éàü 中文 🎉'
    var_2 = 'utf-8'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_read_returns_file_content. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'
    var_2 = 'r'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_write_creates_and_writes_file. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = bool(var_0 == var_1)
    assert var_2 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_write_file_predicate. Retrieved 1/12 statements.


def test_case_0():
    var_0 = 'test content'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_write_creates_file_and_writes_content. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = bool(var_0 == var_1)
    assert var_2 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_read_file_predicate. Retrieved 3/12 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'test content'
    var_2 = 'r'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 2/7 statements.
# Partially parsed test_write_overwrites_existing_file. Retrieved 3/9 statements.
# Partially parsed test_write_empty_string. Retrieved 2/7 statements.
# Partially parsed test_write_multiline_content. Retrieved 2/7 statements.
# Partially parsed test_write_unicode_content. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Old content'
    var_2 = 'New content'

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = ''

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Line 1\nLine 2\nLine 3'

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello 世界 🌍'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/8 statements.
# Partially parsed test_write_overwrites_existing_file. Retrieved 4/9 statements.
# Partially parsed test_write_handles_empty_string. Retrieved 3/8 statements.
# Partially parsed test_write_handles_multiline_content. Retrieved 3/7 statements.
# Partially parsed test_write_handles_special_characters. Retrieved 3/7 statements.
# Partially parsed test_write_handles_unicode_content. Retrieved 3/7 statements.


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
    var_1 = "Special chars: !@#$%^&*()_+-=[]{}|;:',.<>?/~`"
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Unicode: 你好世界 🌍 Привет'
    var_2 = 'utf-8'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_write_creates_file_with_correct_content. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = bool(var_0 == var_1)
    assert var_2 is True



# Parsed testcases at query #37
#--------------------------




def test_case_0():
    var_0 = 'Test that the predicate at line 13 (ext == ".py") evaluates to False.'
    var_1 = '.pyi'
    var_2 = bool(not var_1 == '.py')
    assert var_2 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_loader_predicate_false_when_no_py_file. Retrieved 12/27 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 15 evaluates to False when no .py file is found.'
    var_1 = 'test_module'
    var_2 = '/fake/path'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = ''
    var_6 = []
    var_7 = '/fake/root'
    var_8 = '/fake/pwd'
    var_9 = False
    var_10 = 1
    var_11 = module_0.loader(var_7, var_8, var_9, var_10, var_9)



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_write_file_with_valid_path_and_content. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'
    var_2 = 'utf-8'



# Parsed testcases at query #40
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



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_read_file_opens_in_read_mode. Retrieved 3/14 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'test content'
    var_2 = 'r'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_read_file_returns_string. Retrieved 1/15 statements.


def test_case_0():
    var_0 = 'test content'
    var_1 = bool(var_0)
    assert var_1 is True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_write_creates_and_writes_file. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'



# Parsed testcases at query #44
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



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_loader_creates_parser_with_correct_options. Retrieved 6/15 statements.
# Partially parsed test_loader_calls_walk_packages_with_correct_args. Retrieved 6/15 statements.
# Partially parsed test_loader_parses_py_files. Retrieved 10/22 statements.
# Partially parsed test_loader_tries_pyi_before_py. Retrieved 8/19 statements.
# Partially parsed test_loader_loads_extension_module_when_no_pure_py. Retrieved 9/23 statements.
# Partially parsed test_loader_skips_extension_module_when_pure_py_exists. Retrieved 9/22 statements.
# Partially parsed test_loader_returns_compiled_documentation. Retrieved 6/14 statements.
# Partially parsed test_loader_handles_multiple_packages. Retrieved 12/23 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_root'
    var_1 = '/test/pwd'
    var_2 = True
    var_3 = 2
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    assert var_5 == 'test_doc'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'my_root'
    var_1 = '/my/pwd'
    var_2 = False
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'def test(): pass'
    var_1 = 'pkg.module'
    var_2 = '/path/pkg/module'
    var_3 = (var_1, var_2)
    var_4 = '.py'
    var_5 = 'pkg'
    var_6 = '/path'
    var_7 = True
    var_8 = False
    var_9 = module_0.loader(var_5, var_6, var_7, var_7, var_8)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'pkg.mod'
    var_1 = '/path/pkg/mod'
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'pkg'
    var_5 = '/path'
    var_6 = False
    var_7 = module_0.loader(var_4, var_5, var_3, var_3, var_6)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'pkg.ext'
    var_1 = '/path/pkg/ext'
    var_2 = (var_0, var_1)
    var_3 = '.pyi'
    var_4 = 'pkg'
    var_5 = '/path'
    var_6 = True
    var_7 = False
    var_8 = module_0.loader(var_4, var_5, var_6, var_6, var_7)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'pkg.mod'
    var_1 = '/path/pkg/mod'
    var_2 = (var_0, var_1)
    var_3 = '.py'
    var_4 = 'pkg'
    var_5 = '/path'
    var_6 = True
    var_7 = False
    var_8 = module_0.loader(var_4, var_5, var_6, var_6, var_7)

import apimd.loader as module_0

def test_case_0():
    var_0 = '# Module documentation\n\nSome content'
    var_1 = 'pkg'
    var_2 = '/path'
    var_3 = True
    var_4 = False
    var_5 = module_0.loader(var_1, var_2, var_3, var_3, var_4)
    var_6 = bool(var_5 == var_0)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'pkg1.mod1'
    var_1 = '/path/pkg1/mod1'
    var_2 = (var_0, var_1)
    var_3 = 'pkg2.mod2'
    var_4 = '/path/pkg2/mod2'
    var_5 = (var_3, var_4)
    var_6 = '.py'
    var_7 = 'root'
    var_8 = '/path'
    var_9 = True
    var_10 = False
    var_11 = module_0.loader(var_7, var_8, var_9, var_9, var_10)



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_read_returns_file_content. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'
    var_2 = 'r'



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_loader_basic. Retrieved 8/17 statements.
# Partially parsed test_loader_with_multiple_modules. Retrieved 10/21 statements.
# Partially parsed test_loader_with_toc. Retrieved 7/16 statements.
# Partially parsed test_loader_with_nested_packages. Retrieved 10/23 statements.
# Partially parsed test_loader_with_class. Retrieved 8/17 statements.
# Partially parsed test_loader_no_link. Retrieved 8/17 statements.
# Partially parsed test_loader_different_levels. Retrieved 9/18 statements.
# Partially parsed test_loader_stub_files. Retrieved 8/17 statements.
# Partially parsed test_loader_empty_package. Retrieved 6/13 statements.
# Partially parsed test_loader_with_constants. Retrieved 8/17 statements.


def test_case_0():
    var_0 = 'Test loader with a basic package structure.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\n'
    var_4 = 'module.py'
    var_5 = '"""Test module."""\ndef func():\n    """Test function."""\n    pass\n'
    var_6 = True
    var_7 = False
    var_8 = 'test_pkg'
    var_9 = 'func'

def test_case_0():
    var_0 = 'Test loader with multiple modules.'
    var_1 = 'mypackage'
    var_2 = '__init__.py'
    var_3 = '"""My package."""\n'
    var_4 = 'mod1.py'
    var_5 = '"""Module 1."""\ndef foo():\n    """Foo function."""\n    pass\n'
    var_6 = 'mod2.py'
    var_7 = '"""Module 2."""\ndef bar():\n    """Bar function."""\n    pass\n'
    var_8 = True
    var_9 = False
    var_10 = 'mypackage'
    var_11 = 'foo'
    var_12 = 'bar'

def test_case_0():
    var_0 = 'Test loader with table of contents enabled.'
    var_1 = 'pkg_toc'
    var_2 = '__init__.py'
    var_3 = '"""Package with TOC."""\n'
    var_4 = 'func_mod.py'
    var_5 = '"""Module."""\ndef my_func():\n    """My function."""\n    pass\n'
    var_6 = True
    var_7 = 'Table of contents'
    var_8 = 'pkg_toc'

def test_case_0():
    var_0 = 'Test loader with nested package structure.'
    var_1 = 'parent_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Parent package."""\n'
    var_4 = 'sub_pkg'
    var_5 = '"""Sub package."""\n'
    var_6 = 'submod.py'
    var_7 = '"""Sub module."""\ndef nested_func():\n    """Nested function."""\n    pass\n'
    var_8 = True
    var_9 = False
    var_10 = 'parent_pkg'
    var_11 = 'nested_func'

def test_case_0():
    var_0 = 'Test loader with class definitions.'
    var_1 = 'class_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Class package."""\n'
    var_4 = 'classes.py'
    var_5 = '"""Classes module."""\nclass MyClass:\n    """My class."""\n    def method(self):\n        """My method."""\n        pass\n'
    var_6 = True
    var_7 = False
    var_8 = 'class_pkg'
    var_9 = 'MyClass'
    var_10 = 'method'

def test_case_0():
    var_0 = 'Test loader with link disabled.'
    var_1 = 'nolink_pkg'
    var_2 = '__init__.py'
    var_3 = '"""No link package."""\n'
    var_4 = 'mod.py'
    var_5 = '"""Module."""\ndef func():\n    """Function."""\n    pass\n'
    var_6 = False
    var_7 = 1
    var_8 = 'nolink_pkg'
    var_9 = 'func'

def test_case_0():
    var_0 = 'Test loader with different base levels.'
    var_1 = 'level_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Level package."""\n'
    var_4 = 'mod.py'
    var_5 = '"""Module."""\ndef func():\n    """Function."""\n    pass\n'
    var_6 = True
    var_7 = 2
    var_8 = False
    var_9 = 'level_pkg'
    var_10 = 'func'

def test_case_0():
    var_0 = 'Test loader with .pyi stub files.'
    var_1 = 'stub_pkg'
    var_2 = '__init__.pyi'
    var_3 = '"""Stub package."""\n'
    var_4 = 'mod.pyi'
    var_5 = '"""Stub module."""\ndef stub_func() -> None:\n    """Stub function."""\n    ...\n'
    var_6 = True
    var_7 = False
    var_8 = 'stub_pkg'
    var_9 = 'stub_func'

def test_case_0():
    var_0 = 'Test loader with empty package.'
    var_1 = 'empty_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Empty package."""\n'
    var_4 = True
    var_5 = False
    var_6 = 'empty_pkg'

def test_case_0():
    var_0 = 'Test loader with module constants.'
    var_1 = 'const_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Const package."""\n'
    var_4 = 'const_mod.py'
    var_5 = '"""Constants module."""\nVERSION: str = "1.0.0"\n"""Version constant."""\nMAX_SIZE = 100\n'
    var_6 = True
    var_7 = False
    var_8 = 'const_pkg'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_load_module_predicate_false_loader_not_instance. Retrieved 5/11 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 9 evaluates to False when s is None.'
    var_1 = module_0.Parser()
    var_2 = 'os'
    var_3 = '/fake/path.py'
    var_4 = module_1._load_module(var_2, var_3, var_1)
    assert var_4 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 9 evaluates to False when loader is not a Loader instance.'
    var_1 = module_0.Parser()
    var_2 = 'os'
    var_3 = '/fake/path.py'
    var_4 = module_1._load_module(var_2, var_3, var_1)
    assert var_4 is False



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_write_creates_file_and_writes_content. Retrieved 3/8 statements.
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



# Parsed testcases at query #50
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_file.txt'
    var_1 = module_0._read(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_loader_pure_py_false_continues_to_extension_loading. Retrieved 9/21 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that when pure_py is False, extension module loading is attempted.'
    var_1 = 'test_module'
    var_2 = '/path/test_module'
    var_3 = (var_1, var_2)
    var_4 = '/root'
    var_5 = '/pwd'
    var_6 = False
    var_7 = 1
    var_8 = module_0.loader(var_4, var_5, var_6, var_7, var_6)
    assert var_8 == 'compiled'



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_write_file_opens_with_correct_mode. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'test content'



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_loader_basic. Retrieved 6/13 statements.
# Partially parsed test_loader_with_submodules. Retrieved 8/17 statements.
# Partially parsed test_loader_with_toc. Retrieved 5/12 statements.
# Partially parsed test_loader_with_classes. Retrieved 6/13 statements.
# Partially parsed test_loader_link_disabled. Retrieved 6/13 statements.
# Partially parsed test_loader_different_base_level. Retrieved 7/14 statements.
# Partially parsed test_loader_with_constants. Retrieved 6/13 statements.
# Partially parsed test_loader_with_stub_file. Retrieved 6/13 statements.
# Partially parsed test_loader_nested_packages. Retrieved 8/19 statements.
# Partially parsed test_loader_with_docstrings. Retrieved 6/13 statements.
# Partially parsed test_loader_empty_package. Retrieved 6/14 statements.
# Partially parsed test_loader_with_all_export. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'Test loader with basic package structure.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = "'''Test package.'''\ndef func(): pass"
    var_4 = True
    var_5 = False
    var_6 = 'test_pkg'
    var_7 = 'func'

def test_case_0():
    var_0 = 'Test loader with submodules.'
    var_1 = 'mypackage'
    var_2 = '__init__.py'
    var_3 = "'''Main package.'''\n__all__ = ['submod']"
    var_4 = 'submod.py'
    var_5 = "'''Submodule.'''\ndef subfunc(): pass"
    var_6 = True
    var_7 = False
    var_8 = 'mypackage'
    var_9 = 'mypackage.submod'

def test_case_0():
    var_0 = 'Test loader with table of contents enabled.'
    var_1 = 'tocpkg'
    var_2 = '__init__.py'
    var_3 = "'''TOC package.'''\ndef func1(): pass\ndef func2(): pass"
    var_4 = True
    var_5 = '**Table of contents:**'
    var_6 = 'func1'
    var_7 = 'func2'

def test_case_0():
    var_0 = 'Test loader with class definitions.'
    var_1 = 'classpkg'
    var_2 = '__init__.py'
    var_3 = "'''Class package.'''\nclass MyClass:\n    '''A class.'''\n    def method(self): pass"
    var_4 = True
    var_5 = False
    var_6 = 'MyClass'
    var_7 = 'method'

def test_case_0():
    var_0 = 'Test loader with link disabled.'
    var_1 = 'nolinkpkg'
    var_2 = '__init__.py'
    var_3 = "'''No link package.'''\ndef test_func(): pass"
    var_4 = False
    var_5 = 1
    var_6 = 'test_func'
    var_7 = '<a id='

def test_case_0():
    var_0 = 'Test loader with different base level.'
    var_1 = 'levelpkg'
    var_2 = '__init__.py'
    var_3 = "'''Level package.'''\ndef func(): pass"
    var_4 = True
    var_5 = 2
    var_6 = False
    var_7 = 'levelpkg'

def test_case_0():
    var_0 = 'Test loader with constant definitions.'
    var_1 = 'constpkg'
    var_2 = '__init__.py'
    var_3 = "'''Constant package.'''\nVERSION = '1.0'\nDEBUG = True"
    var_4 = True
    var_5 = False
    var_6 = 'constpkg'

def test_case_0():
    var_0 = 'Test loader with stub file (.pyi).'
    var_1 = 'stubpkg'
    var_2 = '__init__.pyi'
    var_3 = "'''Stub package.'''\ndef stub_func() -> int: ..."
    var_4 = True
    var_5 = False
    var_6 = 'stubpkg'

def test_case_0():
    var_0 = 'Test loader with nested package structure.'
    var_1 = 'parent'
    var_2 = '__init__.py'
    var_3 = "'''Parent package.'''\ndef parent_func(): pass"
    var_4 = 'child'
    var_5 = "'''Child package.'''\ndef child_func(): pass"
    var_6 = True
    var_7 = False
    var_8 = 'parent'
    var_9 = 'parent.child'

def test_case_0():
    var_0 = 'Test loader preserves docstrings.'
    var_1 = 'docpkg'
    var_2 = '__init__.py'
    var_3 = "'''Main docstring.'''\ndef documented_func():\n    '''Function docstring.'''\n    pass"
    var_4 = True
    var_5 = False
    var_6 = 'Main docstring'
    var_7 = 'Function docstring'

def test_case_0():
    var_0 = 'Test loader with empty package.'
    var_1 = 'emptypkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader respects __all__ export.'
    var_1 = 'allpkg'
    var_2 = '__init__.py'
    var_3 = "'''All package.'''\n__all__ = ['public_func']\ndef public_func(): pass\ndef _private_func(): pass"
    var_4 = True
    var_5 = False
    var_6 = 'public_func'



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_load_module_success. Retrieved 10/25 statements.
# Partially parsed test_load_module_import_error. Retrieved 5/11 statements.
# Partially parsed test_load_module_with_docstring. Retrieved 10/26 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module successfully loads a module and extracts docstrings.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'test_module.py'
    var_5 = '"""Module docstring."""\ndef func():\n    """Function docstring."""\n    pass\n'
    var_6 = 0
    var_7 = module_0.Parser()
    var_8 = 'test_pkg.test_module'
    var_9 = module_1._load_module(var_8, var_1, var_7)
    assert var_9 is True
    var_10 = 'test_pkg.test_module'
    var_11 = bool('test_pkg.test_module' in var_7.docstring)
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test _load_module returns False when parent module cannot be imported.'
    var_1 = 'nonexistent_module.py'
    var_2 = '"""Module docstring."""\n'
    var_3 = module_0.Parser()
    var_4 = 'nonexistent.package.module'

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module returns False when spec_from_file_location fails.'
    var_1 = module_0.Parser()
    var_2 = 'sys.invalid'
    var_3 = '/nonexistent/path/file.py'
    var_4 = module_1._load_module(var_2, var_3, var_1)
    assert var_4 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module properly loads module docstrings.'
    var_1 = 'pkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'mod.py'
    var_5 = '"""Test module docstring."""\nVAR = 42\n'
    var_6 = 0
    var_7 = module_0.Parser()
    var_8 = 'pkg.mod'
    var_9 = module_1._load_module(var_8, var_1, var_7)
    assert var_9 is True



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_read_file_opens_in_read_mode. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'test content'



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_write_creates_file_and_writes_content. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'
    var_2 = bool(var_0 == var_1)
    assert var_2 is True



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_read_returns_file_contents. Retrieved 3/12 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'
    var_2 = 'r'



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_load_module_predicate_false_when_loader_not_loader_instance. Retrieved 4/10 statements.


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



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_loader. Retrieved 14/35 statements.
# Partially parsed test_loader_with_toc. Retrieved 9/26 statements.
# Partially parsed test_loader_no_link. Retrieved 10/27 statements.
# Partially parsed test_loader_different_level. Retrieved 11/28 statements.
# Partially parsed test_loader_nonexistent_package. Retrieved 7/20 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test loader function with a sample package structure.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\n\ndef test_func():\n    """Test function."""\n    pass'
    var_4 = 'subpkg'
    var_5 = '"""Test subpackage."""\n\nclass TestClass:\n    """Test class."""\n    pass'
    var_6 = 'logger'
    var_7 = 'test_pkg'
    var_8 = True
    var_9 = False
    var_10 = module_0.loader(var_7, var_1, var_8, var_8, var_9)
    var_11 = len(var_10)
    var_12 = var_11 > var_9
    var_13 = bool('test_pkg' in var_10 or var_12)
    assert var_13 is True
    var_14 = 'logger'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test loader function with table of contents enabled.'
    var_1 = 'sample_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Sample package."""\n\ndef sample():\n    """Sample function."""\n    pass'
    var_4 = 'logger'
    var_5 = 'sample_pkg'
    var_6 = True
    var_7 = module_0.loader(var_5, var_1, var_6, var_6, var_6)
    var_8 = bool(var_2)
    assert var_8 is True
    var_9 = '**Table of contents:**'
    var_10 = bool('**Table of contents:**' in var_7)
    assert var_10 is True
    var_11 = 'logger'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test loader function with link disabled.'
    var_1 = 'no_link_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Package without links."""\n\nVAR = 42'
    var_4 = 'logger'
    var_5 = 'no_link_pkg'
    var_6 = False
    var_7 = 1
    var_8 = module_0.loader(var_5, var_1, var_6, var_7, var_6)
    var_9 = 'logger'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test loader function with different heading level.'
    var_1 = 'level_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Package with different level."""\n\ndef func():\n    """A function."""\n    pass'
    var_4 = 'logger'
    var_5 = 'level_pkg'
    var_6 = True
    var_7 = 2
    var_8 = False
    var_9 = module_0.loader(var_5, var_1, var_6, var_7, var_8)
    var_10 = bool(var_3)
    assert var_10 is True
    var_11 = 'logger'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test loader function with nonexistent package.'
    var_1 = 'logger'
    var_2 = 'nonexistent'
    var_3 = True
    var_4 = False
    var_5 = module_0.loader(var_2, var_1, var_3, var_3, var_4)
    var_6 = 'logger'



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/8 statements.
# Partially parsed test_write_overwrites_existing_file. Retrieved 4/10 statements.
# Partially parsed test_write_empty_string. Retrieved 3/8 statements.
# Partially parsed test_write_multiline_content. Retrieved 3/7 statements.
# Partially parsed test_write_special_characters. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Initial content'
    var_2 = 'New content'
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
    var_1 = 'Special chars: ñ, é, ü, 中文, 🎉'
    var_2 = 'utf-8'



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_loader_pure_py_false_skips_extension_loading. Retrieved 18/38 statements.


def test_case_0():
    var_0 = 'Test that when pure_py is False, extension module loading is attempted.'
    var_1 = 'test_pkg'
    var_2 = '__init__.pyi'
    var_3 = ''
    var_4 = []
    var_5 = 'apimd.loader.walk_packages'
    var_6 = 'apimd.loader._read'
    var_7 = 'apimd.loader._load_module'
    var_8 = 'apimd.loader.isfile'
    var_9 = '.pyi'
    var_10 = lambda x: x.endswith(var_9)
    var_11 = 'apimd.loader.EXTENSION_SUFFIXES'
    var_12 = '.so'
    var_13 = '.pyd'
    var_14 = [var_12, var_13]
    var_15 = False
    var_16 = 1
    var_17 = len(var_4)
    var_18 = bool(var_17 > 0)
    assert var_18 is True



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_read_file_opens_in_read_mode. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 'test content'
    var_1 = bool(var_0 > 0)
    assert var_1 is True



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_read_returns_file_content. Retrieved 1/15 statements.


def test_case_0():
    var_0 = 'test script content'



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_write_predicate_false. Retrieved 2/13 statements.


def test_case_0():
    var_0 = 'test content'
    var_1 = 0
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_loader_basic. Retrieved 6/13 statements.
# Partially parsed test_loader_with_submodules. Retrieved 8/17 statements.
# Partially parsed test_loader_with_toc. Retrieved 5/12 statements.
# Partially parsed test_loader_without_link. Retrieved 6/13 statements.
# Partially parsed test_loader_with_different_level. Retrieved 7/14 statements.
# Partially parsed test_loader_nonexistent_package. Retrieved 4/7 statements.
# Partially parsed test_loader_with_stub_file. Retrieved 6/13 statements.
# Partially parsed test_loader_multiple_packages. Retrieved 8/19 statements.
# Partially parsed test_loader_with_class. Retrieved 6/13 statements.
# Partially parsed test_loader_with_constants. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'Test loader with basic package structure.'
    var_1 = 'testpkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\ndef foo(): pass'
    var_4 = True
    var_5 = False
    var_6 = 'testpkg'
    var_7 = 'foo'

def test_case_0():
    var_0 = 'Test loader with submodules.'
    var_1 = 'mypkg'
    var_2 = '__init__.py'
    var_3 = '"""Main package."""\nx = 1'
    var_4 = 'sub.py'
    var_5 = '"""Submodule."""\ndef bar(): pass'
    var_6 = True
    var_7 = False
    var_8 = 'mypkg'
    var_9 = 'bar'

def test_case_0():
    var_0 = 'Test loader with table of contents enabled.'
    var_1 = 'docpkg'
    var_2 = '__init__.py'
    var_3 = '"""Documentation package."""\ndef func1(): pass\ndef func2(): pass'
    var_4 = True
    var_5 = 'Table of contents'
    var_6 = 'func1'
    var_7 = 'func2'

def test_case_0():
    var_0 = 'Test loader with link disabled.'
    var_1 = 'nolinkpkg'
    var_2 = '__init__.py'
    var_3 = '"""No link package."""\ndef test(): pass'
    var_4 = False
    var_5 = 1
    var_6 = 'nolinkpkg'
    var_7 = '<a id='

def test_case_0():
    var_0 = 'Test loader with different heading level.'
    var_1 = 'levelpkg'
    var_2 = '__init__.py'
    var_3 = '"""Level package."""\ndef method(): pass'
    var_4 = True
    var_5 = 2
    var_6 = False
    var_7 = 'levelpkg'
    var_8 = 'method'

def test_case_0():
    var_0 = 'Test loader with nonexistent package.'
    var_1 = 'nonexistent'
    var_2 = True
    var_3 = False

def test_case_0():
    var_0 = 'Test loader preferring .pyi stub files.'
    var_1 = 'stubpkg'
    var_2 = '__init__.pyi'
    var_3 = '"""Stub file."""\ndef stub_func(): ...'
    var_4 = True
    var_5 = False
    var_6 = 'stubpkg'
    var_7 = 'stub_func'

def test_case_0():
    var_0 = 'Test loader with multiple packages.'
    var_1 = 'pkg1'
    var_2 = '__init__.py'
    var_3 = '"""Package 1."""\ndef func_a(): pass'
    var_4 = 'pkg2'
    var_5 = '"""Package 2."""\ndef func_b(): pass'
    var_6 = True
    var_7 = False
    var_8 = 'pkg1'
    var_9 = 'func_a'

def test_case_0():
    var_0 = 'Test loader with class definition.'
    var_1 = 'classpkg'
    var_2 = '__init__.py'
    var_3 = '"""Class package."""\nclass MyClass:\n    """A class."""\n    def method(self): pass'
    var_4 = True
    var_5 = False
    var_6 = 'MyClass'
    var_7 = 'method'

def test_case_0():
    var_0 = 'Test loader with constants.'
    var_1 = 'constpkg'
    var_2 = '__init__.py'
    var_3 = '"""Constants package."""\nVERSION = \'1.0.0\'\nDEBUG = True'
    var_4 = True
    var_5 = False
    var_6 = 'constpkg'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_loader_basic. Retrieved 6/16 statements.
# Partially parsed test_loader_with_submodules. Retrieved 8/19 statements.
# Partially parsed test_loader_with_toc. Retrieved 5/14 statements.
# Partially parsed test_loader_with_different_levels. Retrieved 7/16 statements.
# Partially parsed test_loader_with_pyi_stub. Retrieved 6/15 statements.
# Partially parsed test_loader_with_nested_packages. Retrieved 8/21 statements.
# Partially parsed test_loader_with_link_disabled. Retrieved 6/15 statements.
# Partially parsed test_loader_multiple_functions. Retrieved 6/15 statements.
# Partially parsed test_loader_with_constants. Retrieved 6/15 statements.
# Partially parsed test_loader_with_class_methods. Retrieved 6/15 statements.


def test_case_0():
    var_0 = 'Test loader with basic package structure.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\ndef foo(): pass'
    var_4 = True
    var_5 = False
    var_6 = 'test_pkg'

def test_case_0():
    var_0 = 'Test loader with submodules.'
    var_1 = 'mylib'
    var_2 = '__init__.py'
    var_3 = '"""Main package."""\nVERSION = \'1.0\''
    var_4 = 'utils.py'
    var_5 = '"""Utilities."""\ndef helper(): pass'
    var_6 = False
    var_7 = 1
    var_8 = 'mylib'

def test_case_0():
    var_0 = 'Test loader with table of contents enabled.'
    var_1 = 'testpkg'
    var_2 = '__init__.py'
    var_3 = '"""Test."""\ndef func1(): pass\ndef func2(): pass'
    var_4 = True
    var_5 = '**Table of contents:**'

def test_case_0():
    var_0 = 'Test loader with different heading levels.'
    var_1 = 'levelpkg'
    var_2 = '__init__.py'
    var_3 = '"""Package."""\nclass MyClass: pass'
    var_4 = True
    var_5 = 2
    var_6 = False

def test_case_0():
    var_0 = 'Test loader with .pyi stub file.'
    var_1 = 'stubpkg'
    var_2 = '__init__.pyi'
    var_3 = '"""Stub."""\ndef stub_func() -> int: ...'
    var_4 = False
    var_5 = 1

def test_case_0():
    var_0 = 'Test loader with nested package structure.'
    var_1 = 'parent'
    var_2 = '__init__.py'
    var_3 = '"""Parent package."""'
    var_4 = 'child'
    var_5 = '"""Child package."""\ndef child_func(): pass'
    var_6 = True
    var_7 = False

def test_case_0():
    var_0 = 'Test loader with link generation disabled.'
    var_1 = 'nolinkpkg'
    var_2 = '__init__.py'
    var_3 = '"""No link test."""\ndef test(): pass'
    var_4 = False
    var_5 = 1
    var_6 = '<a id='

def test_case_0():
    var_0 = 'Test loader with multiple functions in a module.'
    var_1 = 'multifunc'
    var_2 = '__init__.py'
    var_3 = '"""Multi function package."""\ndef func_a(): """Function A."""\ndef func_b(): """Function B."""\nclass ClassA: """Class A."""'
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader with module constants.'
    var_1 = 'constpkg'
    var_2 = '__init__.py'
    var_3 = '"""Constants package."""\nMAX_SIZE: int = 100\nVERSION = \'1.0\'\n'
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader with class containing methods.'
    var_1 = 'classpkg'
    var_2 = '__init__.py'
    var_3 = '"""Class package."""\nclass Handler:\n    """A handler class."""\n    def process(self): pass\n    @staticmethod\n    def static_method(): pass\n'
    var_4 = True
    var_5 = False



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_gen_api_with_dry_run. Retrieved 11/19 statements.
# Partially parsed test_gen_api_creates_directory. Retrieved 16/30 statements.
# Partially parsed test_gen_api_handles_empty_documentation. Retrieved 11/19 statements.
# Partially parsed test_gen_api_multiple_modules. Retrieved 18/27 statements.
# Partially parsed test_gen_api_with_pwd. Retrieved 10/24 statements.
# Partially parsed test_gen_api_file_naming. Retrieved 15/26 statements.
# Partially parsed test_gen_api_level_parameter. Retrieved 11/19 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api with dry run mode.'
    var_1 = 'apimd.loader.loader'
    var_2 = 'apimd.loader._site_path'
    var_3 = 'apimd.loader.isdir'
    var_4 = True
    var_5 = lambda x: var_4
    var_6 = 'Test Module'
    var_7 = 'test_module'
    var_8 = {var_6: var_7}
    var_9 = module_0.gen_api(var_8, level=var_4, dry=var_4)
    var_10 = len(var_9)
    assert var_10 == 1
    var_11 = '# Test Module API'
    var_12 = bool('# Test Module API' in var_9[0])
    assert var_12 is True
    var_13 = '# Module'
    var_14 = bool('# Module' in var_9[0])
    assert var_14 is True

def test_case_0():
    var_0 = "Test gen_api creates prefix directory if it doesn't exist."
    var_1 = 'docs'
    var_2 = 'apimd.loader.loader'
    var_3 = 'apimd.loader._site_path'
    var_4 = 'apimd.loader.isdir'
    var_5 = False
    var_6 = lambda x: var_5
    var_7 = 'apimd.loader.mkdir'
    var_8 = None
    var_9 = lambda x: var_8
    var_10 = 'apimd.loader._write'
    var_11 = lambda path, doc: var_8
    var_12 = 'Test'
    var_13 = 'test'
    var_14 = {var_12: var_13}
    var_15 = 1

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api skips modules with empty documentation.'
    var_1 = 'apimd.loader.loader'
    var_2 = 'apimd.loader._site_path'
    var_3 = 'apimd.loader.isdir'
    var_4 = True
    var_5 = lambda x: var_4
    var_6 = 'Empty Module'
    var_7 = 'empty'
    var_8 = {var_6: var_7}
    var_9 = module_0.gen_api(var_8, level=var_4, dry=var_4)
    var_10 = len(var_9)
    assert var_10 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api with multiple root modules.'
    var_1 = 'apimd.loader.loader'
    var_2 = 'apimd.loader._site_path'
    var_3 = 'apimd.loader.isdir'
    var_4 = True
    var_5 = lambda x: var_4
    var_6 = 'apimd.loader._write'
    var_7 = None
    var_8 = lambda path, doc: var_7
    var_9 = 'Module A'
    var_10 = 'Module B'
    var_11 = 'mod_a'
    var_12 = 'mod_b'
    var_13 = {var_9: var_11, var_10: var_12}
    var_14 = False
    var_15 = 2
    var_16 = module_0.gen_api(var_13, level=var_15, dry=var_14)
    var_17 = len(var_16)
    assert var_17 == 2
    var_18 = '## Module A API'
    var_19 = bool('## Module A API' in var_16[0])
    assert var_19 is True
    var_20 = '## Module B API'
    var_21 = bool('## Module B API' in var_16[1])
    assert var_21 is True

def test_case_0():
    var_0 = 'Test gen_api appends pwd to sys.path.'
    var_1 = 'custom_path'
    var_2 = 'apimd.loader.loader'
    var_3 = 'apimd.loader._site_path'
    var_4 = 'apimd.loader.isdir'
    var_5 = True
    var_6 = lambda x: var_5
    var_7 = 'Test'
    var_8 = 'test'
    var_9 = {var_7: var_8}

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api creates correctly named files.'
    var_1 = []
    var_2 = 'apimd.loader.loader'
    var_3 = 'apimd.loader._site_path'
    var_4 = 'apimd.loader.isdir'
    var_5 = True
    var_6 = lambda x: var_5
    var_7 = 'apimd.loader._write'
    var_8 = 'My Module'
    var_9 = 'my_module'
    var_10 = {var_8: var_9}
    var_11 = 'docs'
    var_12 = False
    var_13 = module_0.gen_api(var_10, prefix=var_11, level=var_5, dry=var_12)
    var_14 = len(var_1)
    assert var_14 == 1
    var_15 = 'my-module-api.md'
    var_16 = bool('my-module-api.md' in var_1[0])
    assert var_16 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api respects level parameter for heading.'
    var_1 = 'apimd.loader.loader'
    var_2 = 'apimd.loader._site_path'
    var_3 = 'apimd.loader.isdir'
    var_4 = True
    var_5 = lambda x: var_4
    var_6 = 'Test'
    var_7 = 'test'
    var_8 = {var_6: var_7}
    var_9 = 3
    var_10 = module_0.gen_api(var_8, level=var_9, dry=var_4)
    var_11 = '### Test API'
    var_12 = bool('### Test API' in var_10[0])
    assert var_12 is True
    var_13 = 'Content'
    var_14 = bool('Content' in var_10[0])
    assert var_14 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_gen_api_creates_directory_when_not_exists. Retrieved 6/13 statements.


def test_case_0():
    var_0 = "Test that gen_api creates the prefix directory if it doesn't exist."
    var_1 = 'new_docs'
    var_2 = 'Test'
    var_3 = 'os'
    var_4 = {var_2: var_3}
    var_5 = True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_loader_predicate_pure_py_false. Retrieved 9/17 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 15 evaluates to False when pure_py is False.'
    var_1 = 'test_module'
    var_2 = '/path/test_module'
    var_3 = (var_1, var_2)
    var_4 = '/root'
    var_5 = '/pwd'
    var_6 = False
    var_7 = 1
    var_8 = module_0.loader(var_4, var_5, var_6, var_7, var_6)
    assert var_8 == 'compiled_output'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_read_returns_file_content. Retrieved 2/9 statements.
# Partially parsed test_read_returns_empty_string_for_empty_file. Retrieved 2/9 statements.
# Partially parsed test_read_returns_multiline_content. Retrieved 2/9 statements.
# Partially parsed test_read_raises_file_not_found_error. Retrieved 1/7 statements.


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



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_loader_predicate_false_when_no_py_file. Retrieved 7/19 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 15 evaluates to False when no .py file is found.'
    var_1 = '.so'
    var_2 = '/root'
    var_3 = '/pwd'
    var_4 = False
    var_5 = 1
    var_6 = module_0.loader(var_2, var_3, var_4, var_5, var_4)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_read_file_opens_in_read_mode. Retrieved 3/13 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'test content'
    var_2 = 'r'



# Parsed testcases at query #9
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
    var_1 = 'こんにちは世界 🌍 Привет мир'
    var_2 = 'utf-8'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_loader_predicate_false_when_no_py_file. Retrieved 12/28 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 15 evaluates to False when no .py file is found.'
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



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_loader_predicate_line_13_false. Retrieved 6/16 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 13 (ext == ".py") evaluates to False.'
    var_1 = False
    var_2 = True
    var_3 = '/root'
    var_4 = '/pwd'
    var_5 = module_0.loader(var_3, var_4, var_1, var_2, var_1)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_gen_api_basic. Retrieved 22/29 statements.
# Partially parsed test_gen_api_multiple_modules. Retrieved 25/33 statements.
# Partially parsed test_gen_api_empty_doc. Retrieved 22/28 statements.
# Partially parsed test_gen_api_with_level. Retrieved 23/29 statements.
# Partially parsed test_gen_api_with_pwd. Retrieved 24/33 statements.
# Partially parsed test_gen_api_writes_file. Retrieved 21/29 statements.
# Partially parsed test_gen_api_underscore_to_dash_conversion. Retrieved 20/28 statements.
# Partially parsed test_gen_api_with_toc. Retrieved 20/29 statements.


import apimd.loader as module_0

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
    var_9 = '# Test\n\nTest content'
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
    var_20 = module_0.gen_api(var_18, prefix=var_1, dry=var_19)
    var_21 = len(var_20)
    assert var_21 == 1
    var_22 = 'Test API'
    var_23 = bool('Test API' in var_20[0])
    assert var_23 is True
    var_24 = 'Test content'
    var_25 = bool('Test content' in var_20[0])
    assert var_25 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api with multiple modules.'
    var_1 = 'apimd.loader.isdir'
    var_2 = False
    var_3 = lambda x: var_2
    var_4 = 'apimd.loader.mkdir'
    var_5 = None
    var_6 = lambda x: var_5
    var_7 = 'apimd.loader.loader'
    var_8 = '# Module\n\nContent'
    var_9 = lambda *args: var_8
    var_10 = 'apimd.loader._site_path'
    var_11 = '/fake/path'
    var_12 = lambda x: var_11
    var_13 = 'apimd.loader._write'
    var_14 = lambda *args: var_5
    var_15 = 'Module1'
    var_16 = 'Module2'
    var_17 = 'mod1'
    var_18 = 'mod2'
    var_19 = {var_15: var_17, var_16: var_18}
    var_20 = 'docs'
    var_21 = True
    var_22 = module_0.gen_api(var_19, prefix=var_20, dry=var_21)
    var_23 = len(var_22)
    assert var_23 == 2
    var_24 = 'API'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api when loader returns empty string.'
    var_1 = 'apimd.loader.isdir'
    var_2 = False
    var_3 = lambda x: var_2
    var_4 = 'apimd.loader.mkdir'
    var_5 = None
    var_6 = lambda x: var_5
    var_7 = 'apimd.loader.loader'
    var_8 = '   \n\n  '
    var_9 = lambda *args: var_8
    var_10 = 'apimd.loader._site_path'
    var_11 = '/fake/path'
    var_12 = lambda x: var_11
    var_13 = 'apimd.loader._write'
    var_14 = lambda *args: var_5
    var_15 = 'Test'
    var_16 = 'test_module'
    var_17 = {var_15: var_16}
    var_18 = 'docs'
    var_19 = True
    var_20 = module_0.gen_api(var_17, prefix=var_18, dry=var_19)
    var_21 = len(var_20)
    assert var_21 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api with different heading level.'
    var_1 = 'apimd.loader.isdir'
    var_2 = False
    var_3 = lambda x: var_2
    var_4 = 'apimd.loader.mkdir'
    var_5 = None
    var_6 = lambda x: var_5
    var_7 = 'apimd.loader.loader'
    var_8 = 'Content'
    var_9 = lambda *args: var_8
    var_10 = 'apimd.loader._site_path'
    var_11 = '/fake/path'
    var_12 = lambda x: var_11
    var_13 = 'apimd.loader._write'
    var_14 = lambda *args: var_5
    var_15 = 'Test'
    var_16 = 'test_module'
    var_17 = {var_15: var_16}
    var_18 = 'docs'
    var_19 = 2
    var_20 = True
    var_21 = module_0.gen_api(var_17, prefix=var_18, level=var_19, dry=var_20)
    var_22 = len(var_21)
    assert var_22 == 1
    var_23 = '## Test API'
    var_24 = bool('## Test API' in var_21[0])
    assert var_24 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api with custom pwd parameter.'
    var_1 = []
    var_2 = 'apimd.loader.isdir'
    var_3 = False
    var_4 = lambda x: var_3
    var_5 = 'apimd.loader.mkdir'
    var_6 = None
    var_7 = lambda x: var_6
    var_8 = 'apimd.loader.loader'
    var_9 = 'Content'
    var_10 = lambda *args: var_9
    var_11 = 'apimd.loader._site_path'
    var_12 = '/fake/path'
    var_13 = lambda x: var_12
    var_14 = 'apimd.loader._write'
    var_15 = lambda *args: var_6
    var_16 = 'apimd.loader.sys_path.append'
    var_17 = 'Test'
    var_18 = 'test_module'
    var_19 = {var_17: var_18}
    var_20 = '/custom/path'
    var_21 = 'docs'
    var_22 = True
    var_23 = module_0.gen_api(var_19, var_20, prefix=var_21, dry=var_22)
    var_24 = '/custom/path'
    var_25 = bool('/custom/path' in var_1)
    assert var_25 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api writes file when dry=False.'
    var_1 = []
    var_2 = 'apimd.loader.isdir'
    var_3 = False
    var_4 = lambda x: var_3
    var_5 = 'apimd.loader.mkdir'
    var_6 = None
    var_7 = lambda x: var_6
    var_8 = 'apimd.loader.loader'
    var_9 = 'Test content'
    var_10 = lambda *args: var_9
    var_11 = 'apimd.loader._site_path'
    var_12 = '/fake/path'
    var_13 = lambda x: var_12
    var_14 = 'apimd.loader._write'
    var_15 = 'MyModule'
    var_16 = 'my_module'
    var_17 = {var_15: var_16}
    var_18 = 'docs'
    var_19 = module_0.gen_api(var_17, prefix=var_18, dry=var_3)
    var_20 = len(var_1)
    assert var_20 == 1
    var_21 = 'my_module-api.md'
    var_22 = bool('my_module-api.md' in var_1[0][0])
    assert var_22 is True
    var_23 = 'MyModule API'
    var_24 = bool('MyModule API' in var_1[0][1])
    assert var_24 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api converts underscores to dashes in filename.'
    var_1 = []
    var_2 = 'apimd.loader.isdir'
    var_3 = False
    var_4 = lambda x: var_3
    var_5 = 'apimd.loader.mkdir'
    var_6 = None
    var_7 = lambda x: var_6
    var_8 = 'apimd.loader.loader'
    var_9 = 'Content'
    var_10 = lambda *args: var_9
    var_11 = 'apimd.loader._site_path'
    var_12 = '/fake/path'
    var_13 = lambda x: var_12
    var_14 = 'apimd.loader._write'
    var_15 = 'Test'
    var_16 = 'test_module_name'
    var_17 = {var_15: var_16}
    var_18 = 'docs'
    var_19 = module_0.gen_api(var_17, prefix=var_18, dry=var_3)
    var_20 = 'test-module-name-api.md'
    var_21 = bool('test-module-name-api.md' in var_1[0])
    assert var_21 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api with toc parameter.'
    var_1 = []
    var_2 = 'apimd.loader.isdir'
    var_3 = False
    var_4 = lambda x: var_3
    var_5 = 'apimd.loader.mkdir'
    var_6 = None
    var_7 = lambda x: var_6
    var_8 = 'apimd.loader.loader'
    var_9 = 'apimd.loader._site_path'
    var_10 = '/fake/path'
    var_11 = lambda x: var_10
    var_12 = 'apimd.loader._write'
    var_13 = lambda *args: var_6
    var_14 = 'Test'
    var_15 = 'test_module'
    var_16 = {var_14: var_15}
    var_17 = 'docs'
    var_18 = True
    var_19 = module_0.gen_api(var_16, prefix=var_17, toc=var_18, dry=var_18)
    var_20 = var_1[0][4]
    assert var_20 is True

def test_case_0():
    var_0 = 'Test gen_api returns sequence of strings.'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_load_module_success. Retrieved 4/25 statements.
# Partially parsed test_load_module_invalid_loader. Retrieved 4/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_pkg.test_module'
    var_2 = 'test_module.py'
    var_3 = '"""Test module docstring"""\ndef test_func():\n    """Test function"""\n    pass'

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'nonexistent.module'
    var_2 = '/fake/path.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_pkg.test_module'
    var_2 = '/fake/path.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_pkg.test_module'
    var_2 = '/fake/path.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_load_module_predicate_true. Retrieved 5/16 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 9 evaluates to True.'
    var_1 = 'test_module.py'
    var_2 = '"""Test module docstring."""\ndef test_func():\n    """Test function."""\n    pass\n'
    var_3 = module_0.Parser()
    var_4 = 'test_module'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_gen_api_iterates_over_root_names. Retrieved 10/15 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 22 evaluates to True by iterating over root_names.items().'
    var_1 = 'Module1'
    var_2 = 'Module2'
    var_3 = 'module1'
    var_4 = 'module2'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'docs'
    var_7 = True
    var_8 = module_0.gen_api(var_5, prefix=var_6, dry=var_7)
    var_9 = len(var_8)
    assert var_9 == 0



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_loader_predicate_line_13_evaluates_to_false. Retrieved 9/21 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 13 (ext == ".py") evaluates to False.'
    var_1 = 'test_module'
    var_2 = '/path/to/test_module'
    var_3 = (var_1, var_2)
    var_4 = '/root'
    var_5 = '/pwd'
    var_6 = False
    var_7 = 1
    var_8 = module_0.loader(var_4, var_5, var_6, var_7, var_6)
    assert var_8 == 'compiled output'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_read_file_opens_in_read_mode. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'test content'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_site_path_valid_package. Retrieved 2/5 statements.
# Partially parsed test_site_path_returns_string. Retrieved 2/3 statements.


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
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'collections'
    var_1 = module_0._site_path(var_0)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_load_module_predicate_true. Retrieved 4/15 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module.py'
    var_1 = 'def hello():\n    """Test function."""\n    pass\n'
    var_2 = module_0.Parser()
    var_3 = 'test_module'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_loader_basic. Retrieved 6/15 statements.
# Partially parsed test_loader_with_submodule. Retrieved 8/19 statements.
# Partially parsed test_loader_with_toc. Retrieved 5/12 statements.
# Partially parsed test_loader_with_link_and_level. Retrieved 7/14 statements.
# Partially parsed test_loader_multiple_modules. Retrieved 10/21 statements.
# Partially parsed test_loader_with_constants. Retrieved 6/13 statements.
# Partially parsed test_loader_pyi_stub. Retrieved 6/13 statements.
# Partially parsed test_loader_nested_packages. Retrieved 8/19 statements.
# Partially parsed test_loader_no_link. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'Test loader with basic package structure.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = "'''Test package.'''\ndef func(): pass"
    var_4 = True
    var_5 = False
    var_6 = 'test_pkg'
    var_7 = 'func'

def test_case_0():
    var_0 = 'Test loader with submodule.'
    var_1 = 'mylib'
    var_2 = '__init__.py'
    var_3 = "'''Main module.'''\n__all__ = ['sub']"
    var_4 = 'sub'
    var_5 = "'''Submodule.'''\ndef helper(): pass"
    var_6 = False
    var_7 = 1
    var_8 = 'mylib'
    var_9 = 'helper'

def test_case_0():
    var_0 = 'Test loader with table of contents enabled.'
    var_1 = 'doclib'
    var_2 = '__init__.py'
    var_3 = "'''Documentation library.'''\ndef api_func(): pass"
    var_4 = True
    var_5 = '**Table of contents:**'
    var_6 = 'doclib'

def test_case_0():
    var_0 = 'Test loader with link enabled and custom level.'
    var_1 = 'linkedlib'
    var_2 = '__init__.py'
    var_3 = "'''Linked library.'''\nclass MyClass: pass"
    var_4 = True
    var_5 = 2
    var_6 = False
    var_7 = 'linkedlib'
    var_8 = 'MyClass'

def test_case_0():
    var_0 = 'Test loader with multiple modules in package.'
    var_1 = 'multimod'
    var_2 = '__init__.py'
    var_3 = "'''Main.'''\ndef main_func(): pass"
    var_4 = 'utils.py'
    var_5 = "'''Utils.'''\ndef util_func(): pass"
    var_6 = 'helpers.py'
    var_7 = "'''Helpers.'''\ndef help_func(): pass"
    var_8 = False
    var_9 = 1
    var_10 = 'main_func'
    var_11 = 'util_func'
    var_12 = 'help_func'

def test_case_0():
    var_0 = 'Test loader with module constants.'
    var_1 = 'constlib'
    var_2 = '__init__.py'
    var_3 = "'''Constants library.'''\nVERSION: str = '1.0'\nMAX_SIZE: int = 100"
    var_4 = False
    var_5 = 1
    var_6 = 'constlib'

def test_case_0():
    var_0 = 'Test loader with .pyi stub file.'
    var_1 = 'stublib'
    var_2 = '__init__.pyi'
    var_3 = "'''Stub module.'''\ndef stub_func() -> int: ..."
    var_4 = False
    var_5 = 1
    var_6 = 'stublib'
    var_7 = 'stub_func'

def test_case_0():
    var_0 = 'Test loader with nested package structure.'
    var_1 = 'root'
    var_2 = '__init__.py'
    var_3 = "'''Root package.'''\ndef root_func(): pass"
    var_4 = 'nested'
    var_5 = "'''Nested package.'''\ndef nested_func(): pass"
    var_6 = False
    var_7 = 1
    var_8 = 'root'
    var_9 = 'nested_func'

def test_case_0():
    var_0 = 'Test loader with link disabled.'
    var_1 = 'nolinklib'
    var_2 = '__init__.py'
    var_3 = "'''No link library.'''\ndef func(): pass"
    var_4 = False
    var_5 = 1
    var_6 = 'nolinklib'
    var_7 = '<a id='



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_gen_api_basic. Retrieved 14/29 statements.
# Partially parsed test_gen_api_multiple_packages. Retrieved 15/29 statements.
# Partially parsed test_gen_api_empty_doc_warning. Retrieved 13/27 statements.
# Partially parsed test_gen_api_with_level. Retrieved 14/28 statements.
# Partially parsed test_gen_api_write_file. Retrieved 18/37 statements.
# Partially parsed test_gen_api_with_pwd. Retrieved 14/32 statements.
# Partially parsed test_gen_api_creates_directory. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'Test gen_api with basic parameters.'
    var_1 = 'docs'
    var_2 = 'apimd.loader.loader'
    var_3 = 'apimd.loader._site_path'
    var_4 = 'apimd.loader.isdir'
    var_5 = True
    var_6 = lambda x: var_5
    var_7 = 'apimd.loader.mkdir'
    var_8 = None
    var_9 = lambda x: var_8
    var_10 = 'Test Package'
    var_11 = 'test_pkg'
    var_12 = {var_10: var_11}
    var_13 = False
    var_14 = 'Test Package API'
    var_15 = 'Module'

def test_case_0():
    var_0 = 'Test gen_api with multiple packages.'
    var_1 = 'docs'
    var_2 = 'apimd.loader.loader'
    var_3 = 'apimd.loader._site_path'
    var_4 = 'apimd.loader.isdir'
    var_5 = True
    var_6 = lambda x: var_5
    var_7 = 'apimd.loader.mkdir'
    var_8 = None
    var_9 = lambda x: var_8
    var_10 = 'Package A'
    var_11 = 'Package B'
    var_12 = 'pkg_a'
    var_13 = 'pkg_b'
    var_14 = {var_10: var_12, var_11: var_13}
    var_15 = 'Package A API'
    var_16 = 'Package B API'

def test_case_0():
    var_0 = 'Test gen_api skips packages with empty documentation.'
    var_1 = 'docs'
    var_2 = 'apimd.loader.loader'
    var_3 = 'apimd.loader._site_path'
    var_4 = 'apimd.loader.isdir'
    var_5 = True
    var_6 = lambda x: var_5
    var_7 = 'apimd.loader.mkdir'
    var_8 = None
    var_9 = lambda x: var_8
    var_10 = 'Empty Package'
    var_11 = 'empty_pkg'
    var_12 = {var_10: var_11}

def test_case_0():
    var_0 = 'Test gen_api respects the level parameter.'
    var_1 = 'docs'
    var_2 = 'apimd.loader.loader'
    var_3 = 'apimd.loader._site_path'
    var_4 = 'apimd.loader.isdir'
    var_5 = True
    var_6 = lambda x: var_5
    var_7 = 'apimd.loader.mkdir'
    var_8 = None
    var_9 = lambda x: var_8
    var_10 = 'My Package'
    var_11 = 'my_pkg'
    var_12 = {var_10: var_11}
    var_13 = 2
    var_14 = '## My Package API'

def test_case_0():
    var_0 = 'Test gen_api writes files when dry=False.'
    var_1 = 'docs'
    var_2 = {}
    var_3 = 'apimd.loader.loader'
    var_4 = 'apimd.loader._site_path'
    var_5 = 'apimd.loader._write'
    var_6 = 'apimd.loader.isdir'
    var_7 = True
    var_8 = lambda x: var_7
    var_9 = 'apimd.loader.mkdir'
    var_10 = None
    var_11 = lambda x: var_10
    var_12 = 'Test'
    var_13 = 'test_pkg'
    var_14 = {var_12: var_13}
    var_15 = False
    var_16 = len(var_2)
    assert var_16 == 1
    var_17 = 'test-pkg-api.md'

def test_case_0():
    var_0 = 'Test gen_api appends pwd to sys.path when provided.'
    var_1 = 'docs'
    var_2 = 'custom'
    var_3 = 'apimd.loader.loader'
    var_4 = 'apimd.loader._site_path'
    var_5 = 'apimd.loader.isdir'
    var_6 = True
    var_7 = lambda x: var_6
    var_8 = 'apimd.loader.mkdir'
    var_9 = None
    var_10 = lambda x: var_9
    var_11 = 'Pkg'
    var_12 = 'pkg'
    var_13 = {var_11: var_12}

def test_case_0():
    var_0 = "Test gen_api creates prefix directory if it doesn't exist."
    var_1 = 'new_docs'
    var_2 = []
    var_3 = 'apimd.loader.loader'
    var_4 = 'apimd.loader._site_path'



# Parsed testcases at query #22
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



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_gen_api_iterates_root_names. Retrieved 10/14 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 22 (for loop) evaluates to True by iterating over root_names.items().'
    var_1 = 'TestTitle'
    var_2 = 'AnotherTitle'
    var_3 = 'test_module'
    var_4 = 'another_module'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = True
    var_7 = module_0.gen_api(var_5, dry=var_6)
    var_8 = isinstance(var_7, var_1)
    var_9 = bool(var_8)
    assert var_9 is True
    var_10 = len(var_7)
    assert var_10 == 2
    var_11 = '# TestTitle API'
    var_12 = bool('# TestTitle API' in var_7[0])
    assert var_12 is True
    var_13 = '# AnotherTitle API'
    var_14 = bool('# AnotherTitle API' in var_7[1])
    assert var_14 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_gen_api_iterates_root_names. Retrieved 9/17 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 22 (for title, name in root_names.items()) evaluates to True.'
    var_1 = 'TestTitle'
    var_2 = 'AnotherTitle'
    var_3 = 'test_module'
    var_4 = 'another_module'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = True
    var_7 = module_0.gen_api(var_5, dry=var_6)
    var_8 = len(var_7)
    assert var_8 == 0



# Parsed testcases at query #25
#--------------------------




import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 9 evaluates to False when s is None.'
    var_1 = module_0.Parser()
    var_2 = 'os'
    var_3 = '/fake/path.py'
    var_4 = module_1._load_module(var_2, var_3, var_1)
    assert var_4 is False



# Parsed testcases at query #26
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 25 evaluates to True when doc is empty or whitespace.'
    var_1 = 'Test'
    var_2 = 'test_module'
    var_3 = {var_1: var_2}
    var_4 = True
    var_5 = module_0.gen_api(var_3, dry=var_4)
    var_6 = bool(var_5 == [])
    assert var_6 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_load_module_success. Retrieved 8/18 statements.
# Partially parsed test_load_module_import_error. Retrieved 5/11 statements.
# Partially parsed test_load_module_invalid_spec. Retrieved 9/18 statements.
# Partially parsed test_load_module_with_docstring. Retrieved 8/18 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module successfully loads a module.'
    var_1 = 'test_module.py'
    var_2 = '"""Test module."""\ndef test_func():\n    """Test function."""\n    pass\n'
    var_3 = 0
    var_4 = module_0.Parser()
    var_5 = 'test_module'
    var_6 = module_1._load_module(var_5, var_1, var_4)
    assert var_6 is True
    var_7 = 0

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test _load_module returns False when parent import fails.'
    var_1 = 'test.py'
    var_2 = '"""Test."""\n'
    var_3 = module_0.Parser()
    var_4 = 'nonexistent.module.test'

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module returns False when spec is None.'
    var_1 = 'test_module.py'
    var_2 = '"""Test module."""\n'
    var_3 = 0
    var_4 = module_0.Parser()
    var_5 = 'test_module'
    var_6 = '/nonexistent/path/test_module.py'
    var_7 = module_1._load_module(var_5, var_6, var_4)
    assert var_7 is False
    var_8 = 0

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module loads module docstring correctly.'
    var_1 = 'test_module.py'
    var_2 = '"""Module docstring."""\n\ndef func():\n    """Function docstring."""\n    pass\n'
    var_3 = 0
    var_4 = module_0.Parser()
    var_5 = 'test_module'
    var_6 = module_1._load_module(var_5, var_1, var_4)
    assert var_6 is True
    var_7 = 'test_module'
    var_8 = bool('test_module' in var_4.docstring)
    assert var_8 is True
    var_9 = 0



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_gen_api_with_dry_run. Retrieved 11/20 statements.
# Partially parsed test_gen_api_creates_directory. Retrieved 17/29 statements.
# Partially parsed test_gen_api_writes_file. Retrieved 15/26 statements.
# Partially parsed test_gen_api_skips_empty_docs. Retrieved 14/20 statements.
# Partially parsed test_gen_api_multiple_modules. Retrieved 17/27 statements.
# Partially parsed test_gen_api_with_custom_level. Retrieved 13/19 statements.
# Partially parsed test_gen_api_appends_to_sys_path. Retrieved 15/24 statements.


def test_case_0():
    var_0 = 'Test gen_api with dry run mode.'
    var_1 = 'apimd.loader.loader'
    var_2 = 'apimd.loader._site_path'
    var_3 = ''
    var_4 = lambda x: var_3
    var_5 = 'apimd.loader.isdir'
    var_6 = True
    var_7 = lambda x: var_6
    var_8 = 'Test Module'
    var_9 = 'test_module'
    var_10 = {var_8: var_9}
    var_11 = 'Test Module API'
    var_12 = 'Module'

def test_case_0():
    var_0 = "Test gen_api creates prefix directory if it doesn't exist."
    var_1 = 'new_docs'
    var_2 = 'apimd.loader.loader'
    var_3 = 'apimd.loader._site_path'
    var_4 = ''
    var_5 = lambda x: var_4
    var_6 = 'apimd.loader.isdir'
    var_7 = False
    var_8 = lambda x: var_7
    var_9 = 'apimd.loader.mkdir'
    var_10 = None
    var_11 = lambda x: var_10
    var_12 = 'apimd.loader._write'
    var_13 = lambda path, doc: var_10
    var_14 = 'Test'
    var_15 = 'test'
    var_16 = {var_14: var_15}

def test_case_0():
    var_0 = 'Test gen_api writes documentation to file.'
    var_1 = []
    var_2 = 'apimd.loader.loader'
    var_3 = 'apimd.loader._site_path'
    var_4 = ''
    var_5 = lambda x: var_4
    var_6 = 'apimd.loader.isdir'
    var_7 = True
    var_8 = lambda x: var_7
    var_9 = 'apimd.loader._write'
    var_10 = 'My Module'
    var_11 = 'my_module'
    var_12 = {var_10: var_11}
    var_13 = False
    var_14 = len(var_1)
    assert var_14 == 1
    var_15 = 'my-module-api.md'
    var_16 = bool('my-module-api.md' in var_1[0][0])
    assert var_16 is True
    var_17 = 'My Module API'
    var_18 = bool('My Module API' in var_1[0][1])
    assert var_18 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api skips modules that produce empty documentation.'
    var_1 = 'apimd.loader.loader'
    var_2 = 'apimd.loader._site_path'
    var_3 = ''
    var_4 = lambda x: var_3
    var_5 = 'apimd.loader.isdir'
    var_6 = True
    var_7 = lambda x: var_6
    var_8 = 'Empty Module'
    var_9 = 'empty'
    var_10 = {var_8: var_9}
    var_11 = '/tmp'
    var_12 = module_0.gen_api(var_10, prefix=var_11, dry=var_6)
    var_13 = len(var_12)
    assert var_13 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api with multiple root modules.'
    var_1 = 'apimd.loader.loader'
    var_2 = 'apimd.loader._site_path'
    var_3 = ''
    var_4 = lambda x: var_3
    var_5 = 'apimd.loader.isdir'
    var_6 = True
    var_7 = lambda x: var_6
    var_8 = 'Module A'
    var_9 = 'Module B'
    var_10 = 'mod_a'
    var_11 = 'mod_b'
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = module_0.gen_api(var_12, dry=var_6)
    var_14 = len(var_13)
    assert var_14 == 2
    var_15 = 'Module A API'
    var_16 = 'Module B API'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api with custom heading level.'
    var_1 = 'apimd.loader.loader'
    var_2 = 'apimd.loader._site_path'
    var_3 = ''
    var_4 = lambda x: var_3
    var_5 = 'apimd.loader.isdir'
    var_6 = True
    var_7 = lambda x: var_6
    var_8 = 'Test'
    var_9 = 'test'
    var_10 = {var_8: var_9}
    var_11 = 3
    var_12 = module_0.gen_api(var_10, level=var_11, dry=var_6)
    var_13 = '### Test API'
    var_14 = bool('### Test API' in var_12[0])
    assert var_14 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test gen_api appends pwd to sys.path when provided.'
    var_1 = []
    var_2 = 'apimd.loader.loader'
    var_3 = 'apimd.loader._site_path'
    var_4 = ''
    var_5 = lambda x: var_4
    var_6 = 'apimd.loader.isdir'
    var_7 = True
    var_8 = lambda x: var_7
    var_9 = 'apimd.loader.sys_path.append'
    var_10 = 'Test'
    var_11 = 'test'
    var_12 = {var_10: var_11}
    var_13 = '/custom/path'
    var_14 = module_0.gen_api(var_12, var_13, dry=var_7)
    var_15 = '/custom/path'
    var_16 = bool('/custom/path' in var_1)
    assert var_16 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_gen_api_basic. Retrieved 20/30 statements.
# Partially parsed test_gen_api_empty_loader_result. Retrieved 18/27 statements.
# Partially parsed test_gen_api_multiple_modules. Retrieved 22/32 statements.
# Partially parsed test_gen_api_with_custom_level. Retrieved 21/30 statements.
# Partially parsed test_gen_api_dry_mode. Retrieved 21/32 statements.
# Partially parsed test_gen_api_write_mode. Retrieved 21/30 statements.
# Partially parsed test_gen_api_underscore_to_dash_conversion. Retrieved 20/29 statements.
# Partially parsed test_gen_api_with_pwd. Retrieved 21/30 statements.


def test_case_0():
    var_0 = 'Test gen_api with basic parameters.'
    var_1 = 'docs'
    var_2 = 'Test Module'
    var_3 = 'test_module'
    var_4 = {var_2: var_3}
    var_5 = 'apimd.loader.isdir'
    var_6 = False
    var_7 = lambda x: var_6
    var_8 = 'apimd.loader.mkdir'
    var_9 = None
    var_10 = lambda x: var_9
    var_11 = 'apimd.loader.loader'
    var_12 = '# Module content'
    var_13 = lambda *args, **kwargs: var_12
    var_14 = 'apimd.loader._site_path'
    var_15 = '/fake/path'
    var_16 = lambda x: var_15
    var_17 = 'apimd.loader._write'
    var_18 = lambda path, doc: var_9
    var_19 = True
    var_20 = 'Test Module API'
    var_21 = 'Module content'

def test_case_0():
    var_0 = 'Test gen_api when loader returns empty string.'
    var_1 = 'docs'
    var_2 = 'Empty Module'
    var_3 = 'empty_module'
    var_4 = {var_2: var_3}
    var_5 = 'apimd.loader.isdir'
    var_6 = False
    var_7 = lambda x: var_6
    var_8 = 'apimd.loader.mkdir'
    var_9 = None
    var_10 = lambda x: var_9
    var_11 = 'apimd.loader.loader'
    var_12 = '   '
    var_13 = lambda *args, **kwargs: var_12
    var_14 = 'apimd.loader._site_path'
    var_15 = '/fake/path'
    var_16 = lambda x: var_15
    var_17 = True

def test_case_0():
    var_0 = 'Test gen_api with multiple root modules.'
    var_1 = 'docs'
    var_2 = 'Module A'
    var_3 = 'Module B'
    var_4 = 'mod_a'
    var_5 = 'mod_b'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'apimd.loader.isdir'
    var_8 = False
    var_9 = lambda x: var_8
    var_10 = 'apimd.loader.mkdir'
    var_11 = None
    var_12 = lambda x: var_11
    var_13 = 'apimd.loader.loader'
    var_14 = '# Content'
    var_15 = lambda *args, **kwargs: var_14
    var_16 = 'apimd.loader._site_path'
    var_17 = '/fake/path'
    var_18 = lambda x: var_17
    var_19 = 'apimd.loader._write'
    var_20 = lambda path, doc: var_11
    var_21 = True
    var_22 = 'Module A API'
    var_23 = 'Module B API'

def test_case_0():
    var_0 = 'Test gen_api with custom heading level.'
    var_1 = 'docs'
    var_2 = 'Custom Level'
    var_3 = 'custom_mod'
    var_4 = {var_2: var_3}
    var_5 = 'apimd.loader.isdir'
    var_6 = False
    var_7 = lambda x: var_6
    var_8 = 'apimd.loader.mkdir'
    var_9 = None
    var_10 = lambda x: var_9
    var_11 = 'apimd.loader.loader'
    var_12 = 'content'
    var_13 = lambda *args, **kwargs: var_12
    var_14 = 'apimd.loader._site_path'
    var_15 = '/fake/path'
    var_16 = lambda x: var_15
    var_17 = 'apimd.loader._write'
    var_18 = lambda path, doc: var_9
    var_19 = 3
    var_20 = True
    var_21 = '### Custom Level API'

def test_case_0():
    var_0 = 'Test gen_api in dry mode does not write files.'
    var_1 = 'docs'
    var_2 = 'Test'
    var_3 = 'test_mod'
    var_4 = {var_2: var_3}
    var_5 = []
    var_6 = 'apimd.loader.isdir'
    var_7 = False
    var_8 = lambda x: var_7
    var_9 = 'apimd.loader.mkdir'
    var_10 = None
    var_11 = lambda x: var_10
    var_12 = 'apimd.loader.loader'
    var_13 = '# Content'
    var_14 = lambda *args, **kwargs: var_13
    var_15 = 'apimd.loader._site_path'
    var_16 = '/fake/path'
    var_17 = lambda x: var_16
    var_18 = 'apimd.loader._write'
    var_19 = True
    var_20 = len(var_5)
    assert var_20 == 0

def test_case_0():
    var_0 = 'Test gen_api writes files when not in dry mode.'
    var_1 = 'docs'
    var_2 = 'Test'
    var_3 = 'test_mod'
    var_4 = {var_2: var_3}
    var_5 = []
    var_6 = 'apimd.loader.isdir'
    var_7 = False
    var_8 = lambda x: var_7
    var_9 = 'apimd.loader.mkdir'
    var_10 = None
    var_11 = lambda x: var_10
    var_12 = 'apimd.loader.loader'
    var_13 = '# Content'
    var_14 = lambda *args, **kwargs: var_13
    var_15 = 'apimd.loader._site_path'
    var_16 = '/fake/path'
    var_17 = lambda x: var_16
    var_18 = 'apimd.loader._write'
    var_19 = lambda path, doc: var_5.append((path, doc))
    var_20 = len(var_5)
    assert var_20 == 1
    var_21 = 'test-mod-api.md'
    var_22 = bool('test-mod-api.md' in var_5[0][0])
    assert var_22 is True

def test_case_0():
    var_0 = 'Test gen_api converts underscores to dashes in filenames.'
    var_1 = 'docs'
    var_2 = 'Test'
    var_3 = 'test_module_name'
    var_4 = {var_2: var_3}
    var_5 = []
    var_6 = 'apimd.loader.isdir'
    var_7 = False
    var_8 = lambda x: var_7
    var_9 = 'apimd.loader.mkdir'
    var_10 = None
    var_11 = lambda x: var_10
    var_12 = 'apimd.loader.loader'
    var_13 = '# Content'
    var_14 = lambda *args, **kwargs: var_13
    var_15 = 'apimd.loader._site_path'
    var_16 = '/fake/path'
    var_17 = lambda x: var_16
    var_18 = 'apimd.loader._write'
    var_19 = lambda path, doc: var_5.append(path)
    var_20 = 'test-module-name-api.md'
    var_21 = bool('test-module-name-api.md' in var_5[0])
    assert var_21 is True

def test_case_0():
    var_0 = 'Test gen_api with custom pwd parameter.'
    var_1 = 'docs'
    var_2 = 'site-packages'
    var_3 = 'Test'
    var_4 = 'test_mod'
    var_5 = {var_3: var_4}
    var_6 = []
    var_7 = 'apimd.loader.isdir'
    var_8 = False
    var_9 = lambda x: var_8
    var_10 = 'apimd.loader.mkdir'
    var_11 = None
    var_12 = lambda x: var_11
    var_13 = 'apimd.loader.loader'
    var_14 = '# Content'
    var_15 = lambda *args, **kwargs: var_14
    var_16 = 'apimd.loader._site_path'
    var_17 = '/fake/path'
    var_18 = lambda x: var_17
    var_19 = 'apimd.loader._write'
    var_20 = lambda path, doc: var_11



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_read_returns_file_content. Retrieved 2/6 statements.
# Partially parsed test_read_returns_empty_string_for_empty_file. Retrieved 2/6 statements.
# Partially parsed test_read_returns_multiline_content. Retrieved 2/6 statements.
# Partially parsed test_read_preserves_special_characters. Retrieved 2/6 statements.


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
    var_1 = 'Special chars: !@#$%^&*()\t\n'



# Parsed testcases at query #31
#--------------------------




import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '/fake/path.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False



# Parsed testcases at query #32
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 25 evaluates to True when doc is empty or whitespace.'
    var_1 = 'Test'
    var_2 = 'test_module'
    var_3 = {var_1: var_2}
    var_4 = True
    var_5 = module_0.gen_api(var_3, dry=var_4)
    var_6 = bool(var_5 == [])
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 25 evaluates to True when doc contains only whitespace.'
    var_1 = 'Test'
    var_2 = 'test_module'
    var_3 = {var_1: var_2}
    var_4 = True
    var_5 = module_0.gen_api(var_3, dry=var_4)
    var_6 = bool(var_5 == [])
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 25 evaluates to False when doc has content.'
    var_1 = 'Test'
    var_2 = 'test_module'
    var_3 = {var_1: var_2}
    var_4 = True
    var_5 = module_0.gen_api(var_3, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 'Some documentation content'
    var_8 = bool('Some documentation content' in var_5[0])
    assert var_8 is True



# Parsed testcases at query #33
#--------------------------




def test_case_0():
    var_0 = 'Test that the predicate at line 13 (ext == ".py") evaluates to False.'
    var_1 = '.pyi'
    var_2 = bool(not var_1 == '.py')
    assert var_2 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_load_module_success. Retrieved 8/25 statements.
# Partially parsed test_load_module_parent_import_error. Retrieved 4/10 statements.
# Partially parsed test_load_module_invalid_spec. Retrieved 8/20 statements.
# Partially parsed test_load_module_with_docstring. Retrieved 8/24 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = ''
    var_3 = 'test_module.py'
    var_4 = '"""Test module."""\ndef test_func():\n    """Test function."""\n    pass\n'
    var_5 = 0
    var_6 = module_0.Parser()
    var_7 = 'test_pkg.test_module'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'nonexistent_module.py'
    var_1 = '"""Test module."""'
    var_2 = module_0.Parser()
    var_3 = 'nonexistent.pkg.module'

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'valid_pkg'
    var_1 = '__init__.py'
    var_2 = ''
    var_3 = 0
    var_4 = module_0.Parser()
    var_5 = 'valid_pkg.missing'
    var_6 = '/nonexistent/path/to/module.py'
    var_7 = module_1._load_module(var_5, var_6, var_4)
    assert var_7 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = 'pkg_with_doc'
    var_1 = '__init__.py'
    var_2 = ''
    var_3 = 'module_with_doc.py'
    var_4 = '"""Module docstring."""\n\ndef func():\n    """Function docstring."""\n    pass\n'
    var_5 = 0
    var_6 = module_0.Parser()
    var_7 = 'pkg_with_doc.module_with_doc'
    var_8 = 'pkg_with_doc.module_with_doc'
    var_9 = bool('pkg_with_doc.module_with_doc' in var_6.docstring)
    assert var_9 is True



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = bool(var_0 == var_1)
    assert var_2 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/7 statements.
# Partially parsed test_write_overwrites_existing_file. Retrieved 4/9 statements.
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
    var_1 = "Special chars: @#$%^&*()_+-=[]{}|;:',.<>?/~`"
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Unicode: 你好世界 🌍 Здравствуй'
    var_2 = 'utf-8'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_load_module_predicate_false_loader_type. Retrieved 5/13 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 9 evaluates to False when s is None.'
    var_1 = module_0.Parser()
    var_2 = 'test.module'
    var_3 = '/fake/path.py'
    var_4 = module_1._load_module(var_2, var_3, var_1)
    assert var_4 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 9 evaluates to False when loader is not Loader type.'
    var_1 = module_0.Parser()
    var_2 = 'test.module'
    var_3 = '/fake/path.py'
    var_4 = module_1._load_module(var_2, var_3, var_1)
    assert var_4 is False



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_read_file_opens_in_read_mode. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 'test content'
    var_1 = bool(var_0)
    assert var_1 is True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_write_predicate_evaluates_to_false. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'test content'
    var_2 = "\ndef _write(path: str, doc: str) -> None:\n    with open(path, 'w+', encoding='utf-8') as f:\n        f.write(doc)\n"
    var_3 = exec(var_2)



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_read_returns_file_contents. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'test_script.txt'
    var_1 = 'test script content'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_write_file_opens_with_correct_mode. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_read_file_opens_in_read_mode. Retrieved 2/14 statements.


def test_case_0():
    var_0 = 'test content'
    var_1 = False
    assert var_1 is False



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_write_file_opens_with_correct_mode. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/8 statements.
# Partially parsed test_write_overwrites_existing_file. Retrieved 4/10 statements.
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
    var_1 = 'Initial content'
    var_2 = 'New content'
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



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_read_returns_file_contents. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 'test content'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_write_creates_file_with_correct_content. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = bool(var_0 == var_1)
    assert var_2 is True



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_loader_predicate_pure_py_false. Retrieved 13/22 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 15 (if pure_py:) evaluates to False.'
    var_1 = 'test_module'
    var_2 = '/path/to/test_module'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = '.pyi'
    var_6 = lambda path: path.endswith(var_5)
    var_7 = 'def foo(): pass'
    var_8 = '/root'
    var_9 = '/pwd'
    var_10 = False
    var_11 = 1
    var_12 = module_0.loader(var_8, var_9, var_10, var_11, var_10)
    assert var_12 == 'compiled'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_gen_api_predicate_at_line_25. Retrieved 5/11 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = '   '
    var_1 = 'Test'
    var_2 = 'test_module'
    var_3 = {var_1: var_2}
    var_4 = module_0.gen_api(var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_load_module_success. Retrieved 10/25 statements.
# Partially parsed test_load_module_invalid_parent. Retrieved 5/11 statements.
# Partially parsed test_load_module_invalid_spec. Retrieved 9/21 statements.
# Partially parsed test_load_module_with_docstring. Retrieved 10/25 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module successfully loads and processes a module.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'test_module.py'
    var_5 = '"""Test module."""\ndef test_func():\n    """Test function."""\n    pass\n'
    var_6 = 0
    var_7 = module_0.Parser()
    var_8 = 'test_pkg.test_module'
    var_9 = module_1._load_module(var_8, var_1, var_7)
    assert var_9 is True
    var_10 = 'test_pkg.test_module'
    var_11 = bool('test_pkg.test_module' in var_7.docstring)
    assert var_11 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test _load_module returns False when parent import fails.'
    var_1 = 'nonexistent.py'
    var_2 = 'pass'
    var_3 = module_0.Parser()
    var_4 = 'nonexistent.module'

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module returns False when spec cannot be created.'
    var_1 = 'test_pkg2'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 0
    var_5 = module_0.Parser()
    var_6 = 'test_pkg2.nonexistent'
    var_7 = '/nonexistent/path/file.py'
    var_8 = module_1._load_module(var_6, var_7, var_5)
    assert var_8 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module loads module docstring correctly.'
    var_1 = 'test_pkg3'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'documented.py'
    var_5 = '"""Module with documentation."""\n\ndef func():\n    """Function doc."""\n    pass\n'
    var_6 = 0
    var_7 = module_0.Parser()
    var_8 = 'test_pkg3.documented'
    var_9 = module_1._load_module(var_8, var_1, var_7)
    assert var_9 is True
    var_10 = 'test_pkg3.documented'
    var_11 = bool('test_pkg3.documented' in var_7.docstring)
    assert var_11 is True



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_loader_basic. Retrieved 6/14 statements.
# Partially parsed test_loader_with_submodule. Retrieved 8/18 statements.
# Partially parsed test_loader_with_toc. Retrieved 5/13 statements.
# Partially parsed test_loader_with_level. Retrieved 7/15 statements.
# Partially parsed test_loader_without_link. Retrieved 6/14 statements.
# Partially parsed test_loader_nested_package. Retrieved 8/20 statements.
# Partially parsed test_loader_with_all. Retrieved 6/14 statements.
# Partially parsed test_loader_multiple_files. Retrieved 10/22 statements.
# Partially parsed test_loader_empty_package. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'Test loader with basic package structure.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\n\ndef func():\n    """Test function."""\n    pass\n'
    var_4 = True
    var_5 = False
    var_6 = 'Test package'
    var_7 = 'func'

def test_case_0():
    var_0 = 'Test loader with submodules.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Main package."""\n'
    var_4 = 'submodule.py'
    var_5 = '"""Submodule."""\n\nclass MyClass:\n    """A test class."""\n    pass\n'
    var_6 = True
    var_7 = False
    var_8 = 'Main package'
    var_9 = 'MyClass'

def test_case_0():
    var_0 = 'Test loader with table of contents enabled.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Package with TOC."""\n\ndef func1():\n    """Function 1."""\n    pass\n\ndef func2():\n    """Function 2."""\n    pass\n'
    var_4 = True
    var_5 = 'Table of contents'
    var_6 = 'func1'
    var_7 = 'func2'

def test_case_0():
    var_0 = 'Test loader with different heading level.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\n\ndef func():\n    """Test function."""\n    pass\n'
    var_4 = True
    var_5 = 2
    var_6 = False
    var_7 = '##'
    var_8 = 'func'

def test_case_0():
    var_0 = 'Test loader with link disabled.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\n\ndef func():\n    """Test function."""\n    pass\n'
    var_4 = False
    var_5 = 1
    var_6 = 'func'
    var_7 = '<a id='

def test_case_0():
    var_0 = 'Test loader with nested package structure.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Main package."""\n'
    var_4 = 'subpkg'
    var_5 = '"""Subpackage."""\n\nclass SubClass:\n    """A subpackage class."""\n    pass\n'
    var_6 = True
    var_7 = False
    var_8 = 'Main package'
    var_9 = 'SubClass'

def test_case_0():
    var_0 = 'Test loader respects __all__ definition.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Package with __all__."""\n\n__all__ = [\'public_func\']\n\ndef public_func():\n    """Public function."""\n    pass\n\ndef _private_func():\n    """Private function."""\n    pass\n'
    var_4 = True
    var_5 = False
    var_6 = 'public_func'

def test_case_0():
    var_0 = 'Test loader with multiple module files.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Main package."""\n'
    var_4 = 'module1.py'
    var_5 = '"""Module 1."""\n\ndef func1():\n    """Function in module 1."""\n    pass\n'
    var_6 = 'module2.py'
    var_7 = '"""Module 2."""\n\ndef func2():\n    """Function in module 2."""\n    pass\n'
    var_8 = True
    var_9 = False
    var_10 = 'Module 1'
    var_11 = 'Module 2'
    var_12 = 'func1'
    var_13 = 'func2'

def test_case_0():
    var_0 = 'Test loader with empty package.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Empty package."""\n'
    var_4 = True
    var_5 = False
    var_6 = 'Empty package'



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_write_file_predicate. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 2/12 statements.


def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = bool(var_0 == var_1)
    assert var_2 is True



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_read_file_opens_successfully. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 'test content'
    var_1 = bool(var_0)
    assert var_1 is True



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_read_returns_file_content. Retrieved 1/11 statements.


def test_case_0():
    var_0 = 'test content'
    var_1 = bool(var_0)
    assert var_1 is True



# Parsed testcases at query #55
#--------------------------




def test_case_0():
    var_0 = "Test that the predicate at line 13 evaluates to False when ext is not '.py'"
    var_1 = '.pyi'
    assert var_1 == '.py'
    var_2 = '.py'
    assert var_2 is False



