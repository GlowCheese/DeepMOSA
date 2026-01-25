####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_gen_api_dry_mode. Retrieved 9/18 statements.
# Partially parsed test_gen_api_creates_directory. Retrieved 8/14 statements.
# Partially parsed test_gen_api_returns_sequence. Retrieved 8/14 statements.
# Partially parsed test_gen_api_with_multiple_roots. Retrieved 10/16 statements.
# Partially parsed test_gen_api_with_custom_level. Retrieved 9/15 statements.
# Partially parsed test_gen_api_with_toc_enabled. Retrieved 7/13 statements.
# Partially parsed test_gen_api_with_link_disabled. Retrieved 9/15 statements.
# Partially parsed test_gen_api_with_pwd. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 'Test gen_api in dry mode without writing files.'
    var_1 = 'PYTHONPATH'
    var_2 = 'Test'
    var_3 = 'os'
    var_4 = {var_2: var_3}
    var_5 = None
    var_6 = 'docs'
    var_7 = True
    var_8 = False

def test_case_0():
    var_0 = "Test that gen_api creates the prefix directory if it doesn't exist."
    var_1 = 'new_docs'
    var_2 = 'Test'
    var_3 = 'os'
    var_4 = {var_2: var_3}
    var_5 = None
    var_6 = True
    var_7 = False

def test_case_0():
    var_0 = 'Test that gen_api returns a sequence.'
    var_1 = 'Test'
    var_2 = 'os'
    var_3 = {var_1: var_2}
    var_4 = None
    var_5 = 'docs'
    var_6 = True
    var_7 = False

def test_case_0():
    var_0 = 'Test gen_api with multiple root packages.'
    var_1 = 'OS'
    var_2 = 'SYS'
    var_3 = 'os'
    var_4 = 'sys'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = None
    var_7 = 'docs'
    var_8 = True
    var_9 = False

def test_case_0():
    var_0 = 'Test gen_api with custom heading level.'
    var_1 = 'Test'
    var_2 = 'os'
    var_3 = {var_1: var_2}
    var_4 = None
    var_5 = 'docs'
    var_6 = True
    var_7 = 2
    var_8 = False

def test_case_0():
    var_0 = 'Test gen_api with table of contents enabled.'
    var_1 = 'Test'
    var_2 = 'os'
    var_3 = {var_1: var_2}
    var_4 = None
    var_5 = 'docs'
    var_6 = True

def test_case_0():
    var_0 = 'Test gen_api with links disabled.'
    var_1 = 'Test'
    var_2 = 'os'
    var_3 = {var_1: var_2}
    var_4 = None
    var_5 = 'docs'
    var_6 = False
    var_7 = 1
    var_8 = True

def test_case_0():
    var_0 = 'Test gen_api with custom pwd parameter.'
    var_1 = 'Test'
    var_2 = 'os'
    var_3 = {var_1: var_2}
    var_4 = 'docs'
    var_5 = True
    var_6 = False



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_loader_basic. Retrieved 6/14 statements.
# Partially parsed test_loader_with_submodules. Retrieved 9/19 statements.
# Partially parsed test_loader_with_toc_enabled. Retrieved 5/12 statements.
# Partially parsed test_loader_with_different_link_levels. Retrieved 7/18 statements.
# Partially parsed test_loader_empty_package. Retrieved 6/14 statements.
# Partially parsed test_loader_nested_packages. Retrieved 8/20 statements.
# Partially parsed test_loader_with_pyi_files. Retrieved 6/14 statements.
# Partially parsed test_loader_multiple_levels. Retrieved 7/15 statements.
# Partially parsed test_loader_all_parameters_false. Retrieved 6/14 statements.
# Partially parsed test_loader_all_parameters_true. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 'Test loader with a basic package structure.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\ndef foo(): pass'
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader with submodules.'
    var_1 = 'mypackage'
    var_2 = '__init__.py'
    var_3 = '"""Main module."""'
    var_4 = 'submodule.py'
    var_5 = '"""Submodule."""\ndef bar(): pass'
    var_6 = False
    var_7 = 1
    var_8 = True

def test_case_0():
    var_0 = 'Test loader with table of contents enabled.'
    var_1 = 'pkg'
    var_2 = '__init__.py'
    var_3 = '"""Package."""\ndef func1(): pass'
    var_4 = True

def test_case_0():
    var_0 = 'Test loader with different link and level settings.'
    var_1 = 'test'
    var_2 = '__init__.py'
    var_3 = '"""Test."""'
    var_4 = True
    var_5 = False
    var_6 = 2

def test_case_0():
    var_0 = 'Test loader with empty package.'
    var_1 = 'empty_pkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader with nested package structure.'
    var_1 = 'outer'
    var_2 = '__init__.py'
    var_3 = '"""Outer package."""'
    var_4 = 'inner'
    var_5 = '"""Inner package."""\ndef nested_func(): pass'
    var_6 = True
    var_7 = False

def test_case_0():
    var_0 = 'Test loader with .pyi stub files.'
    var_1 = 'stub_pkg'
    var_2 = '__init__.pyi'
    var_3 = '"""Stub file."""\ndef stub_func() -> None: ...'
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader with multiple heading levels.'
    var_1 = 'multi_level'
    var_2 = '__init__.py'
    var_3 = '"""Multi level."""\nclass MyClass:\n    def method(self): pass'
    var_4 = True
    var_5 = 2
    var_6 = False

def test_case_0():
    var_0 = 'Test loader with all boolean parameters set to False.'
    var_1 = 'nolink'
    var_2 = '__init__.py'
    var_3 = '"""No link."""'
    var_4 = False
    var_5 = 1

def test_case_0():
    var_0 = 'Test loader with all boolean parameters set to True.'
    var_1 = 'withlink'
    var_2 = '__init__.py'
    var_3 = '"""With link."""'
    var_4 = True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_load_module_success. Retrieved 10/25 statements.
# Partially parsed test_load_module_parent_import_error. Retrieved 5/11 statements.
# Partially parsed test_load_module_invalid_spec. Retrieved 6/16 statements.
# Partially parsed test_load_module_invalid_loader. Retrieved 6/16 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module successfully loads a module.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'test_module.py'
    var_5 = '"""Test module."""\ndef test_func():\n    """Test function."""\n    pass'
    var_6 = 0
    var_7 = module_0.Parser()
    var_8 = 'test_pkg.test_module'
    var_9 = module_1._load_module(var_8, var_1, var_7)
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test _load_module returns False when parent import fails.'
    var_1 = 'nonexistent.py'
    var_2 = 'pass'
    var_3 = module_0.Parser()
    var_4 = 'nonexistent.module.submodule'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test _load_module returns False when spec is None.'
    var_1 = 'test_module.py'
    var_2 = 'pass'
    var_3 = 'apimd.loader.spec_from_file_location'
    var_4 = module_0.Parser()
    var_5 = 'sys.test_module'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test _load_module returns False when loader is not valid.'
    var_1 = 'test_module.py'
    var_2 = 'pass'
    var_3 = 'apimd.loader.spec_from_file_location'
    var_4 = module_0.Parser()
    var_5 = 'sys.test_module'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_load_module_success. Retrieved 5/12 statements.
# Partially parsed test_load_module_import_error. Retrieved 5/11 statements.
# Partially parsed test_load_module_invalid_spec. Retrieved 6/14 statements.
# Partially parsed test_load_module_invalid_loader. Retrieved 6/16 statements.
# Partially parsed test_load_module_calls_load_docstring. Retrieved 6/18 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test _load_module successfully loads a module.'
    var_1 = 'test_module.py'
    var_2 = '"""Test module docstring."""\ndef test_func():\n    """Test function."""\n    pass'
    var_3 = module_0.Parser()
    var_4 = 'test_module'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test _load_module returns False when parent import fails.'
    var_1 = 'nonexistent.py'
    var_2 = '"""Test."""'
    var_3 = module_0.Parser()
    var_4 = 'nonexistent_parent.nonexistent'

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module returns False when spec is None.'
    var_1 = 'test_mod.py'
    var_2 = '"""Test."""'
    var_3 = module_0.Parser()
    var_4 = 'test_mod'
    var_5 = module_1._load_module(var_4, var_1, var_3)
    assert var_5 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module returns False when loader is not valid.'
    var_1 = 'test_mod2.py'
    var_2 = '"""Test."""'
    var_3 = module_0.Parser()
    var_4 = 'test_mod2'
    var_5 = module_1._load_module(var_4, var_1, var_3)
    assert var_5 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module calls load_docstring on parser.'
    var_1 = 'test_mod3.py'
    var_2 = '"""Module docstring."""'
    var_3 = module_0.Parser()
    var_4 = 'test_mod3'
    var_5 = module_1._load_module(var_4, var_1, var_3)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_gen_api_creates_directory_when_prefix_not_exists. Retrieved 6/16 statements.


def test_case_0():
    var_0 = "Test that gen_api creates the prefix directory when it doesn't exist."
    var_1 = 'new_docs'
    var_2 = 'test'
    var_3 = 'nonexistent_module'
    var_4 = {var_2: var_3}
    var_5 = True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_loader_basic. Retrieved 5/16 statements.
# Partially parsed test_loader_with_module. Retrieved 7/21 statements.
# Partially parsed test_loader_with_toc. Retrieved 4/14 statements.
# Partially parsed test_loader_multiple_files. Retrieved 9/25 statements.
# Partially parsed test_loader_nested_package. Retrieved 7/21 statements.
# Partially parsed test_loader_with_class_and_methods. Retrieved 5/15 statements.
# Partially parsed test_loader_different_link_settings. Retrieved 5/19 statements.
# Partially parsed test_loader_different_level_settings. Retrieved 6/18 statements.


def test_case_0():
    var_0 = 'testpkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\ndef test_func():\n    """Test function."""\n    pass\n'
    var_3 = True
    var_4 = False

def test_case_0():
    var_0 = 'mypkg'
    var_1 = '__init__.py'
    var_2 = '"""My package."""\n'
    var_3 = 'module.py'
    var_4 = '"""Module docstring."""\ndef my_func():\n    """Function doc."""\n    pass\n'
    var_5 = False
    var_6 = 1

def test_case_0():
    var_0 = 'tocpkg'
    var_1 = '__init__.py'
    var_2 = '"""TOC package."""\nclass MyClass:\n    """Class doc."""\n    pass\n'
    var_3 = True

def test_case_0():
    var_0 = 'multipkg'
    var_1 = '__init__.py'
    var_2 = '"""Multi package."""\n'
    var_3 = 'mod1.py'
    var_4 = '"""Module 1."""\ndef func1():\n    """Func 1."""\n    pass\n'
    var_5 = 'mod2.py'
    var_6 = '"""Module 2."""\ndef func2():\n    """Func 2."""\n    pass\n'
    var_7 = True
    var_8 = False

def test_case_0():
    var_0 = 'nestpkg'
    var_1 = 'sub'
    var_2 = '__init__.py'
    var_3 = '"""Nested package."""\n'
    var_4 = '"""Sub package."""\ndef sub_func():\n    """Sub func."""\n    pass\n'
    var_5 = True
    var_6 = False

def test_case_0():
    var_0 = 'clspkg'
    var_1 = '__init__.py'
    var_2 = '"""Class package."""\nclass MyClass:\n    """My class."""\n    def method(self):\n        """My method."""\n        pass\n'
    var_3 = True
    var_4 = False

def test_case_0():
    var_0 = 'linkpkg'
    var_1 = '__init__.py'
    var_2 = '"""Link package."""\ndef func():\n    """Function."""\n    pass\n'
    var_3 = True
    var_4 = False

def test_case_0():
    var_0 = 'levelpkg'
    var_1 = '__init__.py'
    var_2 = '"""Level package."""\ndef func():\n    """Function."""\n    pass\n'
    var_3 = True
    var_4 = False
    var_5 = 2



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_gen_api_creates_directory_when_not_exists. Retrieved 4/11 statements.


def test_case_0():
    var_0 = "Test that gen_api creates the prefix directory if it doesn't exist."
    var_1 = 'new_docs'
    var_2 = {}
    var_3 = True



# Parsed testcases at query #8
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
    var_1 = 'Unicode: 你好世界 🌍 Привет'
    var_2 = 'utf-8'



# Parsed testcases at query #9
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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_gen_api_basic. Retrieved 20/32 statements.
# Partially parsed test_gen_api_dry_mode. Retrieved 21/30 statements.
# Partially parsed test_gen_api_multiple_modules. Retrieved 21/30 statements.
# Partially parsed test_gen_api_empty_doc. Retrieved 19/28 statements.
# Partially parsed test_gen_api_with_pwd. Retrieved 29/39 statements.
# Partially parsed test_gen_api_with_parameters. Retrieved 24/32 statements.


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
    var_9 = '# Test\nDocumentation'
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
    var_8 = '# Module\nTest doc'
    var_9 = lambda *args, **kwargs: var_8
    var_10 = 'apimd.loader._site_path'
    var_11 = '/path'
    var_12 = lambda x: var_11
    var_13 = []
    var_14 = 'apimd.loader._write'
    var_15 = lambda path, doc: write_called.append((path, doc))
    var_16 = 'API'
    var_17 = 'mymodule'
    var_18 = {var_16: var_17}
    var_19 = True
    var_20 = len(var_13)
    assert var_20 == 0

def test_case_0():
    var_0 = 'Test gen_api with multiple root modules.'
    var_1 = 'apimd.loader.isdir'
    var_2 = False
    var_3 = lambda x: var_2
    var_4 = 'apimd.loader.mkdir'
    var_5 = None
    var_6 = lambda x: var_5
    var_7 = 'apimd.loader.loader'
    var_8 = '# Doc\nContent'
    var_9 = lambda *args, **kwargs: var_8
    var_10 = 'apimd.loader._site_path'
    var_11 = '/path'
    var_12 = lambda x: var_11
    var_13 = 'apimd.loader._write'
    var_14 = lambda path, doc: var_5
    var_15 = 'Module1'
    var_16 = 'Module2'
    var_17 = 'mod1'
    var_18 = 'mod2'
    var_19 = {var_15: var_17, var_16: var_18}
    var_20 = True

def test_case_0():
    var_0 = 'Test gen_api when loader returns empty documentation.'
    var_1 = 'apimd.loader.isdir'
    var_2 = False
    var_3 = lambda x: var_2
    var_4 = 'apimd.loader.mkdir'
    var_5 = None
    var_6 = lambda x: var_5
    var_7 = 'apimd.loader.loader'
    var_8 = '   \n\n  '
    var_9 = lambda *args, **kwargs: var_8
    var_10 = 'apimd.loader._site_path'
    var_11 = '/path'
    var_12 = lambda x: var_11
    var_13 = 'apimd.loader._write'
    var_14 = lambda path, doc: var_5
    var_15 = 'Empty'
    var_16 = 'empty_module'
    var_17 = {var_15: var_16}
    var_18 = True

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
    var_9 = '# Doc'
    var_10 = lambda *args, **kwargs: var_9
    var_11 = 'apimd.loader._site_path'
    var_12 = '/path'
    var_13 = lambda x: var_12
    var_14 = 'apimd.loader._write'
    var_15 = lambda path, doc: var_6
    var_16 = 'apimd.loader.sys_path'
    var_17 = ''
    var_18 = ()
    var_19 = 'append'
    var_20 = lambda self, x: sys_path_append_called.append(x)
    var_21 = {var_19: var_20}
    var_22 = type(var_17, var_18, var_21)
    var_23 = 'Test'
    var_24 = 'test'
    var_25 = {var_23: var_24}
    var_26 = '/custom/pwd'
    var_27 = True
    var_28 = len(var_1)

def test_case_0():
    var_0 = 'Test gen_api with various parameters.'
    var_1 = []
    var_2 = 'apimd.loader.isdir'
    var_3 = False
    var_4 = lambda x: var_3
    var_5 = 'apimd.loader.mkdir'
    var_6 = None
    var_7 = lambda x: var_6
    var_8 = 'apimd.loader.loader'
    var_9 = 1
    var_10 = '# Doc'
    var_11 = lambda *args, **kwargs: (loader_calls.append((args, kwargs)), var_10)[var_9]
    var_12 = 'apimd.loader._site_path'
    var_13 = '/path'
    var_14 = lambda x: var_13
    var_15 = 'apimd.loader._write'
    var_16 = lambda path, doc: var_6
    var_17 = 'API'
    var_18 = 'mymodule'
    var_19 = {var_17: var_18}
    var_20 = 2
    var_21 = True
    var_22 = True
    var_23 = len(var_1)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_loader_predicate_pure_py_false. Retrieved 11/25 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Ensure that the predicate at line 15 evaluates to False when no .py file is found.'
    var_1 = 'test_module'
    var_2 = '/fake/path'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = 'stub content'
    var_6 = '/fake/root'
    var_7 = '/fake/pwd'
    var_8 = False
    var_9 = 1
    var_10 = module_0.loader(var_6, var_7, var_8, var_9, var_8)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_loader_predicate_line_15_false. Retrieved 8/17 statements.


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



# Parsed testcases at query #13
#--------------------------




def test_case_0():
    var_0 = 'Test that the predicate at line 13 (ext == ".py") evaluates to False.'
    var_1 = '.pyi'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_site_path_with_existing_package. Retrieved 2/5 statements.
# Partially parsed test_site_path_with_valid_installed_package. Retrieved 2/3 statements.


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
    var_0 = 'importlib'
    var_1 = module_0._site_path(var_0)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_loader_basic. Retrieved 6/15 statements.
# Partially parsed test_loader_with_multiple_modules. Retrieved 10/21 statements.
# Partially parsed test_loader_with_toc_enabled. Retrieved 5/12 statements.
# Partially parsed test_loader_with_nested_packages. Retrieved 8/19 statements.
# Partially parsed test_loader_with_link_disabled. Retrieved 6/13 statements.
# Partially parsed test_loader_with_custom_level. Retrieved 7/14 statements.
# Partially parsed test_loader_with_pyi_stub. Retrieved 6/13 statements.
# Partially parsed test_loader_empty_package. Retrieved 6/14 statements.
# Partially parsed test_loader_with_class_definitions. Retrieved 6/13 statements.
# Partially parsed test_loader_with_all_options_enabled. Retrieved 5/12 statements.


def test_case_0():
    var_0 = 'Test loader with a simple package structure.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\ndef hello(): pass'
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader with multiple modules in package.'
    var_1 = 'multi_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Main package."""'
    var_4 = 'module1.py'
    var_5 = '"""Module 1."""\ndef func1(): pass'
    var_6 = 'module2.py'
    var_7 = '"""Module 2."""\ndef func2(): pass'
    var_8 = True
    var_9 = False

def test_case_0():
    var_0 = 'Test loader with table of contents enabled.'
    var_1 = 'toc_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Package with TOC."""\ndef test_func(): pass'
    var_4 = True

def test_case_0():
    var_0 = 'Test loader with nested package structure.'
    var_1 = 'nested_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Nested package."""'
    var_4 = 'sub'
    var_5 = '"""Subpackage."""\ndef sub_func(): pass'
    var_6 = True
    var_7 = False

def test_case_0():
    var_0 = 'Test loader with link generation disabled.'
    var_1 = 'no_link_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Package without links."""\ndef func(): pass'
    var_4 = False
    var_5 = 1

def test_case_0():
    var_0 = 'Test loader with custom heading level.'
    var_1 = 'level_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Package with custom level."""\ndef func(): pass'
    var_4 = True
    var_5 = 2
    var_6 = False

def test_case_0():
    var_0 = 'Test loader with .pyi stub files.'
    var_1 = 'stub_pkg'
    var_2 = '__init__.pyi'
    var_3 = '"""Stub package."""\ndef stub_func() -> int: ...'
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader with empty package.'
    var_1 = 'empty_pkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader with class definitions in module.'
    var_1 = 'class_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Package with classes."""\nclass TestClass:\n    """A test class."""\n    pass'
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader with all options enabled.'
    var_1 = 'full_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Full featured package."""\ndef full_func(): """Function doc."""\n    pass'
    var_4 = True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_loader_predicate_line_15_false. Retrieved 6/18 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Ensure the predicate at line 15 (if pure_py:) evaluates to False.'
    var_1 = '/root'
    var_2 = '/pwd'
    var_3 = False
    var_4 = 1
    var_5 = module_0.loader(var_1, var_2, var_3, var_4, var_3)
    assert var_5 == 'compiled output'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/8 statements.
# Partially parsed test_write_overwrites_existing_file. Retrieved 4/10 statements.
# Partially parsed test_write_empty_string. Retrieved 3/8 statements.
# Partially parsed test_write_multiline_content. Retrieved 3/7 statements.
# Partially parsed test_write_unicode_content. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Original content'
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
    var_1 = 'Hello 世界 مرحبا мир'
    var_2 = 'utf-8'



# Parsed testcases at query #18
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
    var_0 = 'special_script.txt'
    var_1 = "Special chars: !@#$%^&*()_+-=[]{}|;:',.<>?/~`"



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_load_module_success. Retrieved 9/19 statements.
# Partially parsed test_load_module_import_error. Retrieved 6/15 statements.
# Partially parsed test_load_module_invalid_spec. Retrieved 9/15 statements.
# Partially parsed test_load_module_invalid_loader. Retrieved 8/17 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test _load_module successfully loads a module and docstring.'
    var_1 = 'test_module.py'
    var_2 = '"""Test module docstring."""\ndef func(): pass'
    var_3 = 'apimd.loader.parent'
    var_4 = ''
    var_5 = lambda x: var_4
    var_6 = module_0.Parser()
    var_7 = 'builtins.__import__'
    var_8 = 'test_module'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test _load_module returns False when parent import fails.'
    var_1 = 'test_module.py'
    var_2 = '"""Test module."""'
    var_3 = module_0.Parser()
    var_4 = 'builtins.__import__'
    var_5 = 'test_module'

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module returns False when spec is invalid.'
    var_1 = module_0.Parser()
    var_2 = 'builtins.__import__'
    var_3 = 'apimd.loader.spec_from_file_location'
    var_4 = None
    var_5 = lambda name, path: var_4
    var_6 = 'test_module'
    var_7 = '/nonexistent/path.py'
    var_8 = module_1._load_module(var_6, var_7, var_1)
    assert var_8 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module returns False when loader is not Loader type.'
    var_1 = module_0.Parser()
    var_2 = 'builtins.__import__'
    var_3 = 'test_module'
    var_4 = None
    var_5 = 'apimd.loader.spec_from_file_location'
    var_6 = '/nonexistent/path.py'
    var_7 = module_1._load_module(var_3, var_6, var_1)
    assert var_7 is False



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_load_module_predicate_true. Retrieved 4/15 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = f'{var_0}.py'
    var_2 = '"""Test module docstring."""\ndef foo():\n    """Foo function."""\n    pass\n'
    var_3 = module_0.Parser()



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_read_returns_file_content. Retrieved 3/12 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'
    var_2 = 'r'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_load_module_predicate_true. Retrieved 5/21 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 9 evaluates to True.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = '/path/to/module.py'
    var_4 = module_1._load_module(var_2, var_3, var_1)
    assert var_4 is True



# Parsed testcases at query #23
#--------------------------




def test_case_0():
    var_0 = 'Test that the predicate at line 13 (ext == ".py") evaluates to False.'
    var_1 = '.pyi'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_load_module_predicate_true. Retrieved 8/29 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\n'
    var_3 = 0
    var_4 = 'test_module.py'
    var_5 = '"""Test module docstring."""\ndef test_func():\n    """Test function."""\n    pass\n'
    var_6 = module_0.Parser()
    var_7 = 'test_pkg.test_module'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_load_module_success. Retrieved 8/18 statements.
# Partially parsed test_load_module_import_error. Retrieved 5/11 statements.
# Partially parsed test_load_module_no_loader. Retrieved 6/16 statements.
# Partially parsed test_load_module_calls_load_docstring. Retrieved 11/26 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module successfully loads a module.'
    var_1 = 'test_module.py'
    var_2 = '"""Test module docstring."""\ndef foo(): pass'
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
    var_2 = '"""Test."""'
    var_3 = module_0.Parser()
    var_4 = 'nonexistent_parent.test'

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module returns False when spec_from_file_location returns None.'
    var_1 = module_0.Parser()
    var_2 = 'sys'
    var_3 = '/nonexistent/path.py'
    var_4 = module_1._load_module(var_2, var_3, var_1)
    assert var_4 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module returns False when loader is not a Loader instance.'
    var_1 = 'test.py'
    var_2 = '"""Test."""'
    var_3 = module_0.Parser()
    var_4 = 'test'
    var_5 = module_1._load_module(var_4, var_1, var_3)
    assert var_5 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module calls p.load_docstring with correct arguments.'
    var_1 = 'test_mod.py'
    var_2 = '"""Module doc."""\nx = 1'
    var_3 = 0
    var_4 = module_0.Parser()
    var_5 = var_4.load_docstring
    var_6 = []
    var_7 = 'test_mod'
    var_8 = module_1._load_module(var_7, var_1, var_4)
    assert var_8 is True
    var_9 = len(var_6)
    assert var_9 == 1
    var_10 = 0



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_write_creates_file_and_writes_content. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'



# Parsed testcases at query #27
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



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_load_module_returns_false_when_loader_not_instance. Retrieved 4/11 statements.


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



# Parsed testcases at query #29
#--------------------------






# Parsed testcases at query #30
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
    var_1 = 'Line 1\nLine 2\nLine 3\n'

def test_case_0():
    var_0 = 'special.txt'
    var_1 = 'Special chars: !@#$%^&*()\nTabs:\t\t\nQuotes: "Hello"'



# Parsed testcases at query #31
#--------------------------




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
    assert var_8 == 2



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_load_module_predicate_false. Retrieved 4/11 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module'
    var_2 = '/path/to/module.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False



# Parsed testcases at query #33
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



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_read_returns_file_content. Retrieved 8/18 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'
    var_2 = 0
    var_3 = 'def _read'
    var_4 = var_3.split(var_3)[var_2]
    var_5 = '\ndef _read(path: str) -> str:\n    """Read the script from file."""\n    with open(path, \'r\') as f:\n        return f.read()\n'
    var_6 = var_4 + var_5
    var_7 = exec(var_6)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_gen_api_dry_mode. Retrieved 11/19 statements.
# Partially parsed test_gen_api_write_file. Retrieved 14/27 statements.
# Partially parsed test_gen_api_multiple_packages. Retrieved 14/22 statements.
# Partially parsed test_gen_api_empty_documentation. Retrieved 13/23 statements.
# Partially parsed test_gen_api_with_sys_path. Retrieved 10/23 statements.
# Partially parsed test_gen_api_underscore_to_dash_conversion. Retrieved 12/23 statements.
# Partially parsed test_gen_api_level_parameter. Retrieved 11/18 statements.


def test_case_0():
    var_0 = 'Test gen_api with dry mode enabled.'
    var_1 = 'apimd.loader.loader'
    var_2 = 'apimd.loader._site_path'
    var_3 = '/fake/path'
    var_4 = lambda x: var_3
    var_5 = 'Test Module'
    var_6 = 'test_module'
    var_7 = {var_5: var_6}
    var_8 = None
    var_9 = True
    var_10 = False

def test_case_0():
    var_0 = 'Test gen_api writes file when dry mode is disabled.'
    var_1 = 'apimd.loader.loader'
    var_2 = 'apimd.loader._site_path'
    var_3 = '/fake/path'
    var_4 = lambda x: var_3
    var_5 = 'docs'
    var_6 = 'My Package'
    var_7 = 'my_package'
    var_8 = {var_6: var_7}
    var_9 = None
    var_10 = True
    var_11 = False
    var_12 = 'my-package-api.md'
    var_13 = 'utf-8'

def test_case_0():
    var_0 = 'Test gen_api with multiple packages.'
    var_1 = 'apimd.loader.loader'
    var_2 = 'apimd.loader._site_path'
    var_3 = '/fake/path'
    var_4 = lambda x: var_3
    var_5 = 'Package A'
    var_6 = 'Package B'
    var_7 = 'pkg_a'
    var_8 = 'pkg_b'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = None
    var_11 = True
    var_12 = 2
    var_13 = False

def test_case_0():
    var_0 = 'Test gen_api skips packages with empty documentation.'
    var_1 = 'apimd.loader.loader'
    var_2 = 'apimd.loader._site_path'
    var_3 = '/fake/path'
    var_4 = lambda x: var_3
    var_5 = 'Valid'
    var_6 = 'Invalid'
    var_7 = 'valid_pkg'
    var_8 = 'invalid_pkg'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = None
    var_11 = True
    var_12 = False

def test_case_0():
    var_0 = 'Test gen_api appends pwd to sys.path when provided.'
    var_1 = 'apimd.loader.loader'
    var_2 = 'apimd.loader._site_path'
    var_3 = '/fake/path'
    var_4 = lambda x: var_3
    var_5 = 'custom_path'
    var_6 = 'Test'
    var_7 = 'test_pkg'
    var_8 = {var_6: var_7}
    var_9 = True

def test_case_0():
    var_0 = 'Test gen_api converts underscores to dashes in filenames.'
    var_1 = 'apimd.loader.loader'
    var_2 = 'apimd.loader._site_path'
    var_3 = '/fake/path'
    var_4 = lambda x: var_3
    var_5 = 'docs'
    var_6 = 'Test Package'
    var_7 = 'test_package_name'
    var_8 = {var_6: var_7}
    var_9 = None
    var_10 = False
    var_11 = 'test-package-name-api.md'

def test_case_0():
    var_0 = 'Test gen_api respects level parameter for heading.'
    var_1 = 'apimd.loader.loader'
    var_2 = 'apimd.loader._site_path'
    var_3 = '/fake/path'
    var_4 = lambda x: var_3
    var_5 = 'Title'
    var_6 = 'pkg'
    var_7 = {var_5: var_6}
    var_8 = None
    var_9 = 3
    var_10 = True



# Parsed testcases at query #36
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 31 evaluates to False.'
    var_1 = 'Test'
    var_2 = 'test_module'
    var_3 = {var_1: var_2}
    var_4 = False
    var_5 = module_0.gen_api(var_3, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 1



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_gen_api_dry_mode. Retrieved 26/39 statements.
# Partially parsed test_gen_api_write_mode. Retrieved 29/43 statements.
# Partially parsed test_gen_api_multiple_packages. Retrieved 31/45 statements.
# Partially parsed test_gen_api_empty_content. Retrieved 27/40 statements.
# Partially parsed test_gen_api_with_level. Retrieved 27/40 statements.
# Partially parsed test_gen_api_underscore_to_dash. Retrieved 30/45 statements.


def test_case_0():
    var_0 = 'Test gen_api in dry mode without writing files.'
    var_1 = 'docs'
    var_2 = 'apimd.loader.mkdir'
    var_3 = None
    var_4 = lambda x: var_3
    var_5 = 'apimd.loader.isdir'
    var_6 = True
    var_7 = lambda x: var_6
    var_8 = 'apimd.loader.logger'
    var_9 = 'obj'
    var_10 = 'info'
    var_11 = 'warning'
    var_12 = 'debug'
    var_13 = lambda x: var_3
    var_14 = lambda x: var_3
    var_15 = lambda x: var_3
    var_16 = {var_10: var_13, var_11: var_14, var_12: var_15}
    var_17 = 'apimd.loader.loader'
    var_18 = '## Module\n\nContent'
    var_19 = lambda *args: var_18
    var_20 = 'apimd.loader._site_path'
    var_21 = ''
    var_22 = lambda x: var_21
    var_23 = 'Test'
    var_24 = 'test_module'
    var_25 = {var_23: var_24}

def test_case_0():
    var_0 = 'Test gen_api in write mode to create files.'
    var_1 = 'docs'
    var_2 = 'apimd.loader.mkdir'
    var_3 = None
    var_4 = lambda x: var_3
    var_5 = 'apimd.loader.isdir'
    var_6 = True
    var_7 = lambda x: var_6
    var_8 = 'apimd.loader.logger'
    var_9 = 'obj'
    var_10 = 'info'
    var_11 = 'warning'
    var_12 = 'debug'
    var_13 = lambda x: var_3
    var_14 = lambda x: var_3
    var_15 = lambda x: var_3
    var_16 = {var_10: var_13, var_11: var_14, var_12: var_15}
    var_17 = 'apimd.loader.loader'
    var_18 = '## Module\n\nContent'
    var_19 = lambda *args: var_18
    var_20 = 'apimd.loader._site_path'
    var_21 = ''
    var_22 = lambda x: var_21
    var_23 = 'apimd.loader._write'
    var_24 = lambda path, doc: var_3
    var_25 = 'Test'
    var_26 = 'test_module'
    var_27 = {var_25: var_26}
    var_28 = False

def test_case_0():
    var_0 = 'Test gen_api with multiple packages.'
    var_1 = 'docs'
    var_2 = 'apimd.loader.mkdir'
    var_3 = None
    var_4 = lambda x: var_3
    var_5 = 'apimd.loader.isdir'
    var_6 = True
    var_7 = lambda x: var_6
    var_8 = 'apimd.loader.logger'
    var_9 = 'obj'
    var_10 = 'info'
    var_11 = 'warning'
    var_12 = 'debug'
    var_13 = lambda x: var_3
    var_14 = lambda x: var_3
    var_15 = lambda x: var_3
    var_16 = {var_10: var_13, var_11: var_14, var_12: var_15}
    var_17 = 'apimd.loader.loader'
    var_18 = '## Content'
    var_19 = lambda *args: var_18
    var_20 = 'apimd.loader._site_path'
    var_21 = ''
    var_22 = lambda x: var_21
    var_23 = 'apimd.loader._write'
    var_24 = lambda path, doc: var_3
    var_25 = 'API1'
    var_26 = 'API2'
    var_27 = 'pkg1'
    var_28 = 'pkg2'
    var_29 = {var_25: var_27, var_26: var_28}
    var_30 = False

def test_case_0():
    var_0 = 'Test gen_api skips packages with empty content.'
    var_1 = 'docs'
    var_2 = 'apimd.loader.mkdir'
    var_3 = None
    var_4 = lambda x: var_3
    var_5 = 'apimd.loader.isdir'
    var_6 = True
    var_7 = lambda x: var_6
    var_8 = 'apimd.loader.logger'
    var_9 = 'obj'
    var_10 = 'info'
    var_11 = 'warning'
    var_12 = 'debug'
    var_13 = lambda x: var_3
    var_14 = lambda x: var_3
    var_15 = lambda x: var_3
    var_16 = {var_10: var_13, var_11: var_14, var_12: var_15}
    var_17 = 'apimd.loader.loader'
    var_18 = '   \n\n   '
    var_19 = lambda *args: var_18
    var_20 = 'apimd.loader._site_path'
    var_21 = ''
    var_22 = lambda x: var_21
    var_23 = 'Test'
    var_24 = 'test_module'
    var_25 = {var_23: var_24}
    var_26 = False

def test_case_0():
    var_0 = 'Test gen_api with custom heading level.'
    var_1 = 'docs'
    var_2 = 'apimd.loader.mkdir'
    var_3 = None
    var_4 = lambda x: var_3
    var_5 = 'apimd.loader.isdir'
    var_6 = True
    var_7 = lambda x: var_6
    var_8 = 'apimd.loader.logger'
    var_9 = 'obj'
    var_10 = 'info'
    var_11 = 'warning'
    var_12 = 'debug'
    var_13 = lambda x: var_3
    var_14 = lambda x: var_3
    var_15 = lambda x: var_3
    var_16 = {var_10: var_13, var_11: var_14, var_12: var_15}
    var_17 = 'apimd.loader.loader'
    var_18 = '## Content'
    var_19 = lambda *args: var_18
    var_20 = 'apimd.loader._site_path'
    var_21 = ''
    var_22 = lambda x: var_21
    var_23 = 'Test'
    var_24 = 'test_module'
    var_25 = {var_23: var_24}
    var_26 = 2

def test_case_0():
    var_0 = 'Test gen_api converts underscores to dashes in filename.'
    var_1 = 'docs'
    var_2 = []
    var_3 = 'apimd.loader.mkdir'
    var_4 = None
    var_5 = lambda x: var_4
    var_6 = 'apimd.loader.isdir'
    var_7 = True
    var_8 = lambda x: var_7
    var_9 = 'apimd.loader.logger'
    var_10 = 'obj'
    var_11 = 'info'
    var_12 = 'warning'
    var_13 = 'debug'
    var_14 = lambda x: var_4
    var_15 = lambda x: var_4
    var_16 = lambda x: var_4
    var_17 = {var_11: var_14, var_12: var_15, var_13: var_16}
    var_18 = 'apimd.loader.loader'
    var_19 = '## Content'
    var_20 = lambda *args: var_19
    var_21 = 'apimd.loader._site_path'
    var_22 = ''
    var_23 = lambda x: var_22
    var_24 = 'apimd.loader._write'
    var_25 = 'Test'
    var_26 = 'test_module_name'
    var_27 = {var_25: var_26}
    var_28 = False
    var_29 = len(var_2)
    assert var_29 == 1



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_gen_api_dry_mode_false. Retrieved 7/14 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 31 (dry flag) evaluates to False.'
    var_1 = 'TestModule'
    var_2 = 'test_module'
    var_3 = {var_1: var_2}
    var_4 = False
    var_5 = module_0.gen_api(var_3, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 1



# Parsed testcases at query #39
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 31 (if dry:) evaluates to False.'
    var_1 = 'TestModule'
    var_2 = 'test_module'
    var_3 = {var_1: var_2}
    var_4 = 'docs'
    var_5 = False
    var_6 = module_0.gen_api(var_3, prefix=var_4, dry=var_5)
    var_7 = len(var_6)
    assert var_7 == 1



# Parsed testcases at query #40
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
    var_1 = 'old content'
    var_2 = 'new content'
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



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_read_returns_file_content. Retrieved 2/6 statements.
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
    var_1 = 'line1\nline2\nline3'

def test_case_0():
    var_0 = 'whitespace.txt'
    var_1 = '  indented\n\ttabbed\n'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_write_creates_file_and_writes_content. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_read_predicate_line_3_evaluates_to_false. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'test content'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_read_existing_file. Retrieved 2/6 statements.
# Partially parsed test_read_empty_file. Retrieved 2/6 statements.
# Partially parsed test_read_multiline_file. Retrieved 2/6 statements.
# Partially parsed test_read_file_with_special_characters. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'test_script.txt'
    var_1 = 'Hello, World!\nThis is a test.'

def test_case_0():
    var_0 = 'empty.txt'
    var_1 = ''

def test_case_0():
    var_0 = 'multiline.txt'
    var_1 = 'line1\nline2\nline3\n'

def test_case_0():
    var_0 = 'special.txt'
    var_1 = "Special chars: !@#$%^&*()_+-=[]{}|;:',.<>?/\n"

import apimd.loader as module_0

def test_case_0():
    var_0 = '/nonexistent/path/to/file.txt'
    var_1 = module_0._read(var_0)



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_read_file_opens_in_read_mode. Retrieved 1/11 statements.


def test_case_0():
    var_0 = 'test content'



# Parsed testcases at query #46
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
    var_1 = 'Old content'
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



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_write_file_predicate. Retrieved 1/17 statements.


def test_case_0():
    var_0 = 'test content'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_load_module_success. Retrieved 4/11 statements.
# Partially parsed test_load_module_import_error. Retrieved 5/14 statements.
# Partially parsed test_load_module_invalid_spec. Retrieved 5/14 statements.
# Partially parsed test_load_module_with_docstring. Retrieved 4/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module.py'
    var_1 = '"""Test module docstring."""\ndef test_func():\n    """Test function."""\n    pass\n'
    var_2 = module_0.Parser()
    var_3 = 'test_module'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'nonexistent.py'
    var_1 = 'pass'
    var_2 = module_0.Parser()
    var_3 = 'builtins.__import__'
    var_4 = 'nonexistent'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_invalid.py'
    var_1 = 'pass'
    var_2 = module_0.Parser()
    var_3 = 'importlib.util.spec_from_file_location'
    var_4 = 'test_invalid'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_doc.py'
    var_1 = '"""Module with docstring."""\n\nclass TestClass:\n    """Test class."""\n    pass\n'
    var_2 = module_0.Parser()
    var_3 = 'test_doc'



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_loader_basic. Retrieved 6/13 statements.
# Partially parsed test_loader_with_submodules. Retrieved 8/17 statements.
# Partially parsed test_loader_with_class. Retrieved 6/13 statements.
# Partially parsed test_loader_with_link_option. Retrieved 6/13 statements.
# Partially parsed test_loader_with_toc_option. Retrieved 5/12 statements.
# Partially parsed test_loader_with_level_option. Retrieved 6/13 statements.
# Partially parsed test_loader_nested_package. Retrieved 8/19 statements.
# Partially parsed test_loader_empty_package. Retrieved 6/14 statements.
# Partially parsed test_loader_with_constants. Retrieved 6/13 statements.
# Partially parsed test_loader_with_all. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'Test loader with a simple package structure.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\n\ndef foo():\n    """Test function."""\n    pass'
    var_4 = False
    var_5 = 1

def test_case_0():
    var_0 = 'Test loader with submodules.'
    var_1 = 'mypackage'
    var_2 = '__init__.py'
    var_3 = '"""Main package."""\n'
    var_4 = 'submodule.py'
    var_5 = '"""Sub module."""\n\ndef bar():\n    """Bar function."""\n    pass'
    var_6 = False
    var_7 = 1

def test_case_0():
    var_0 = 'Test loader with class definitions.'
    var_1 = 'clspkg'
    var_2 = '__init__.py'
    var_3 = '"""Package with class."""\n\nclass MyClass:\n    """A test class."""\n    pass'
    var_4 = False
    var_5 = 1

def test_case_0():
    var_0 = 'Test loader with link=True option.'
    var_1 = 'linkpkg'
    var_2 = '__init__.py'
    var_3 = '"""Package with links."""\n'
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader with toc=True option.'
    var_1 = 'tocpkg'
    var_2 = '__init__.py'
    var_3 = '"""Package with TOC."""\n\ndef func():\n    """A function."""\n    pass'
    var_4 = True

def test_case_0():
    var_0 = 'Test loader with different level option.'
    var_1 = 'lvlpkg'
    var_2 = '__init__.py'
    var_3 = '"""Package with level."""\n'
    var_4 = False
    var_5 = 2

def test_case_0():
    var_0 = 'Test loader with nested package structure.'
    var_1 = 'nested'
    var_2 = '__init__.py'
    var_3 = '"""Nested package."""\n'
    var_4 = 'inner'
    var_5 = '"""Inner package."""\n\ndef inner_func():\n    """Inner function."""\n    pass'
    var_6 = False
    var_7 = 1

def test_case_0():
    var_0 = 'Test loader with minimal package.'
    var_1 = 'emptypkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = False
    var_5 = 1

def test_case_0():
    var_0 = 'Test loader with module constants.'
    var_1 = 'constpkg'
    var_2 = '__init__.py'
    var_3 = '"""Package with constants."""\n\nVERSION = \'1.0.0\'\nDEBUG = True'
    var_4 = False
    var_5 = 1

def test_case_0():
    var_0 = 'Test loader with __all__ definition.'
    var_1 = 'allpkg'
    var_2 = '__init__.py'
    var_3 = '"""Package with __all__."""\n\n__all__ = [\'public_func\']\n\ndef public_func():\n    """Public function."""\n    pass\n\ndef _private_func():\n    """Private function."""\n    pass'
    var_4 = False
    var_5 = 1



# Parsed testcases at query #50
#--------------------------




def test_case_0():
    var_0 = 'Test that the predicate at line 13 (ext == ".py") evaluates to False.'
    var_1 = '.pyi'



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_write_predicate_false. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'test content'



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_read_returns_file_contents. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'
    var_2 = 'r'



# Parsed testcases at query #53
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
    var_1 = 'Special chars: !@#$%^&*()\n\tTab and newline'



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_write_creates_file_and_writes_content. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'



# Parsed testcases at query #55
#--------------------------




def test_case_0():
    var_0 = 'Ensure that the predicate at line 13 (ext == ".py") evaluates to False.'
    var_1 = '.pyi'



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_loader_basic. Retrieved 6/16 statements.
# Partially parsed test_loader_with_toc. Retrieved 5/14 statements.
# Partially parsed test_loader_without_link. Retrieved 6/15 statements.
# Partially parsed test_loader_different_levels. Retrieved 7/19 statements.
# Partially parsed test_loader_multiple_modules. Retrieved 8/19 statements.
# Partially parsed test_loader_stub_file. Retrieved 6/15 statements.
# Partially parsed test_loader_nested_package. Retrieved 8/21 statements.
# Partially parsed test_loader_with_all. Retrieved 6/15 statements.
# Partially parsed test_loader_empty_package. Retrieved 6/15 statements.


def test_case_0():
    var_0 = 'Test loader with basic package structure.'
    var_1 = 'testpkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\ndef test_func():\n    """Test function."""\n    pass\n'
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader with table of contents enabled.'
    var_1 = 'testpkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\nCONST = 42\n'
    var_4 = True

def test_case_0():
    var_0 = 'Test loader with link disabled.'
    var_1 = 'testpkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\nclass TestClass:\n    """Test class."""\n    pass\n'
    var_4 = False
    var_5 = 1

def test_case_0():
    var_0 = 'Test loader with different base heading levels.'
    var_1 = 'testpkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\ndef func():\n    """Function."""\n    pass\n'
    var_4 = True
    var_5 = False
    var_6 = 2

def test_case_0():
    var_0 = 'Test loader with multiple modules in package.'
    var_1 = 'testpkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\n'
    var_4 = 'module.py'
    var_5 = '"""Test module."""\ndef module_func():\n    """Module function."""\n    pass\n'
    var_6 = True
    var_7 = False

def test_case_0():
    var_0 = 'Test loader with .pyi stub files.'
    var_1 = 'testpkg'
    var_2 = '__init__.pyi'
    var_3 = '"""Test package stub."""\ndef stub_func() -> int: ...\n'
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader with nested package structure.'
    var_1 = 'testpkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\n'
    var_4 = 'subpkg'
    var_5 = '"""Sub package."""\ndef sub_func():\n    """Sub function."""\n    pass\n'
    var_6 = True
    var_7 = False

def test_case_0():
    var_0 = 'Test loader with __all__ defined.'
    var_1 = 'testpkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\n__all__ = ["public_func"]\ndef public_func():\n    """Public function."""\n    pass\ndef _private_func():\n    """Private function."""\n    pass\n'
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader with empty package.'
    var_1 = 'testpkg'
    var_2 = '__init__.py'
    var_3 = '"""Empty test package."""\n'
    var_4 = True
    var_5 = False



# Parsed testcases at query #57
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



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_write_file_opens_with_correct_mode. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'test content'



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_read_returns_file_contents. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = var_0 == var_1



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_read_returns_file_content. Retrieved 2/6 statements.
# Partially parsed test_read_with_multiline_content. Retrieved 2/6 statements.
# Partially parsed test_read_empty_file. Retrieved 2/6 statements.
# Partially parsed test_read_file_with_special_characters. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'

def test_case_0():
    var_0 = 'multiline.txt'
    var_1 = 'Line 1\nLine 2\nLine 3'

def test_case_0():
    var_0 = 'empty.txt'
    var_1 = ''

def test_case_0():
    var_0 = 'special.txt'
    var_1 = "Special chars: !@#$%^&*()_+-=[]{}|;:',.<>?/"

import apimd.loader as module_0

def test_case_0():
    var_0 = '/nonexistent/path/to/file.txt'
    var_1 = module_0._read(var_0)



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_loader_predicate_line_15_false. Retrieved 11/25 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 15 (if pure_py:) evaluates to False.'
    var_1 = 'test_module'
    var_2 = '/fake/path/test_module'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = False
    var_6 = '/fake/root'
    var_7 = '/fake/pwd'
    var_8 = False
    var_9 = 1
    var_10 = module_0.loader(var_6, var_7, var_8, var_9, var_8)
    assert var_10 == 'compiled_output'



# Parsed testcases at query #63
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



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_loader_basic. Retrieved 12/30 statements.
# Partially parsed test_loader_with_multiple_modules. Retrieved 17/23 statements.
# Partially parsed test_loader_with_extension_module. Retrieved 19/28 statements.
# Partially parsed test_loader_no_pure_py_with_extension. Retrieved 16/25 statements.
# Partially parsed test_loader_parser_creation_with_options. Retrieved 9/13 statements.
# Partially parsed test_loader_reads_py_before_pyi. Retrieved 15/25 statements.
# Partially parsed test_loader_extension_warning_when_no_module_found. Retrieved 17/27 statements.
# Partially parsed test_loader_empty_package. Retrieved 9/14 statements.


def test_case_0():
    var_0 = 'Test loader with basic package structure.'
    var_1 = 'testpkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\ndef foo(): pass'
    var_4 = 'apimd.loader.Parser.new'
    var_5 = 'apimd.loader.walk_packages'
    var_6 = '__init__'
    var_7 = 'apimd.loader._read'
    var_8 = 'def foo(): pass'
    var_9 = 'apimd.loader.isfile'
    var_10 = True
    var_11 = False

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test loader with multiple modules.'
    var_1 = 'apimd.loader.Parser.new'
    var_2 = 'apimd.loader.walk_packages'
    var_3 = 'pkg.mod1'
    var_4 = '/path/mod1'
    var_5 = (var_3, var_4)
    var_6 = 'pkg.mod2'
    var_7 = '/path/mod2'
    var_8 = (var_6, var_7)
    var_9 = [var_5, var_8]
    var_10 = 'apimd.loader._read'
    var_11 = 'code'
    var_12 = 'apimd.loader.isfile'
    var_13 = True
    var_14 = 'pkg'
    var_15 = '/root'
    var_16 = module_0.loader(var_14, var_15, var_13, var_13, var_13)
    assert var_16 == 'docs'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test loader with extension modules.'
    var_1 = 'apimd.loader.Parser.new'
    var_2 = 'apimd.loader.walk_packages'
    var_3 = 'pkg.ext'
    var_4 = '/path/ext'
    var_5 = (var_3, var_4)
    var_6 = [var_5]
    var_7 = 'apimd.loader._read'
    var_8 = ''
    var_9 = 'apimd.loader.isfile'
    var_10 = '.pyi'
    var_11 = '.so'
    var_12 = 'apimd.loader._load_module'
    var_13 = True
    var_14 = 'pkg'
    var_15 = '/root'
    var_16 = False
    var_17 = 2
    var_18 = module_0.loader(var_14, var_15, var_16, var_17, var_16)
    assert var_18 == 'compiled'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test loader skips extension loading when pure Python exists.'
    var_1 = 'apimd.loader.Parser.new'
    var_2 = 'apimd.loader.walk_packages'
    var_3 = 'pkg'
    var_4 = '/path'
    var_5 = (var_3, var_4)
    var_6 = [var_5]
    var_7 = 'apimd.loader._read'
    var_8 = 'python code'
    var_9 = 'apimd.loader.isfile'
    var_10 = '.py'
    var_11 = 'apimd.loader._load_module'
    var_12 = '/root'
    var_13 = True
    var_14 = False
    var_15 = module_0.loader(var_3, var_12, var_13, var_13, var_14)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test loader creates parser with correct options.'
    var_1 = 'apimd.loader.Parser.new'
    var_2 = 'apimd.loader.walk_packages'
    var_3 = []
    var_4 = 'pkg'
    var_5 = '/root'
    var_6 = True
    var_7 = 3
    var_8 = module_0.loader(var_4, var_5, var_6, var_7, var_6)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test loader prefers .py files over .pyi files.'
    var_1 = 'apimd.loader.Parser.new'
    var_2 = 'apimd.loader.walk_packages'
    var_3 = 'pkg'
    var_4 = '/path'
    var_5 = (var_3, var_4)
    var_6 = [var_5]
    var_7 = 'apimd.loader._read'
    var_8 = 'py_code'
    var_9 = 'pyi_code'
    var_10 = 'apimd.loader.isfile'
    var_11 = True
    var_12 = '/root'
    var_13 = False
    var_14 = module_0.loader(var_3, var_12, var_11, var_11, var_13)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test loader warns when extension module not found.'
    var_1 = 'apimd.loader.Parser.new'
    var_2 = 'apimd.loader.walk_packages'
    var_3 = 'pkg'
    var_4 = '/path'
    var_5 = (var_3, var_4)
    var_6 = [var_5]
    var_7 = 'apimd.loader._read'
    var_8 = ''
    var_9 = 'apimd.loader.isfile'
    var_10 = '.pyi'
    var_11 = 'apimd.loader._load_module'
    var_12 = False
    var_13 = 'apimd.loader.logger'
    var_14 = '/root'
    var_15 = True
    var_16 = module_0.loader(var_3, var_14, var_15, var_15, var_12)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test loader with empty package.'
    var_1 = 'apimd.loader.Parser.new'
    var_2 = 'apimd.loader.walk_packages'
    var_3 = []
    var_4 = 'empty'
    var_5 = '/root'
    var_6 = True
    var_7 = False
    var_8 = module_0.loader(var_4, var_5, var_6, var_6, var_7)
    assert var_8 == 'empty docs'



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_read_returns_file_content. Retrieved 2/13 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_write_file_opens_in_write_mode. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'test content'



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_write_creates_file_and_writes_content. Retrieved 2/10 statements.
# Partially parsed test_write_overwrites_existing_file. Retrieved 3/11 statements.
# Partially parsed test_write_handles_empty_string. Retrieved 2/10 statements.
# Partially parsed test_write_handles_multiline_content. Retrieved 2/7 statements.
# Partially parsed test_write_handles_unicode_content. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'

def test_case_0():
    var_0 = 'test.txt'
    assert var_0 == 'Second content'
    var_1 = 'First content'
    var_2 = 'Second content'

def test_case_0():
    var_0 = 'test.txt'
    assert var_0 == ''
    var_1 = ''

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Line 1\nLine 2\nLine 3'

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello 世界 🌍'



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_read_file_opens_in_read_mode. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'test content'



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_loader_basic. Retrieved 5/17 statements.
# Partially parsed test_loader_with_submodule. Retrieved 7/21 statements.
# Partially parsed test_loader_with_toc. Retrieved 4/15 statements.
# Partially parsed test_loader_without_link. Retrieved 5/16 statements.
# Partially parsed test_loader_with_different_level. Retrieved 6/17 statements.
# Partially parsed test_loader_empty_package. Retrieved 5/16 statements.
# Partially parsed test_loader_multiple_files. Retrieved 9/26 statements.
# Partially parsed test_loader_stub_file. Retrieved 5/16 statements.


def test_case_0():
    var_0 = 'testpkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\ndef test_func():\n    """Test function."""\n    pass\n'
    var_3 = True
    var_4 = False

def test_case_0():
    var_0 = 'mypkg'
    var_1 = '__init__.py'
    var_2 = '"""Main package."""\n'
    var_3 = 'submod.py'
    var_4 = '"""Submodule."""\nclass MyClass:\n    """A class."""\n    pass\n'
    var_5 = True
    var_6 = False

def test_case_0():
    var_0 = 'pkg'
    var_1 = '__init__.py'
    var_2 = '"""Package with TOC."""\ndef func():\n    """Function."""\n    pass\n'
    var_3 = True

def test_case_0():
    var_0 = 'nolinkpkg'
    var_1 = '__init__.py'
    var_2 = '"""No link package."""\n'
    var_3 = False
    var_4 = 1

def test_case_0():
    var_0 = 'levelpkg'
    var_1 = '__init__.py'
    var_2 = '"""Level package."""\n'
    var_3 = True
    var_4 = 2
    var_5 = False

def test_case_0():
    var_0 = 'emptypkg'
    var_1 = '__init__.py'
    var_2 = ''
    var_3 = True
    var_4 = False

def test_case_0():
    var_0 = 'multipkg'
    var_1 = '__init__.py'
    var_2 = '"""Multi package."""\n'
    var_3 = 'mod1.py'
    var_4 = '"""Module 1."""\ndef func1():\n    """Function 1."""\n    pass\n'
    var_5 = 'mod2.py'
    var_6 = '"""Module 2."""\ndef func2():\n    """Function 2."""\n    pass\n'
    var_7 = True
    var_8 = False

def test_case_0():
    var_0 = 'stubpkg'
    var_1 = '__init__.pyi'
    var_2 = '"""Stub package."""\ndef stub_func() -> None: ...\n'
    var_3 = True
    var_4 = False



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_load_module_predicate_false. Retrieved 5/12 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test that the predicate at line 9 evaluates to False when s is None.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = '/path/to/module.py'
    var_4 = module_1._load_module(var_2, var_3, var_1)
    assert var_4 is False



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_loader_basic. Retrieved 6/14 statements.
# Partially parsed test_loader_with_submodule. Retrieved 8/18 statements.
# Partially parsed test_loader_with_class. Retrieved 5/13 statements.
# Partially parsed test_loader_with_link_disabled. Retrieved 6/14 statements.
# Partially parsed test_loader_with_toc_enabled. Retrieved 5/12 statements.
# Partially parsed test_loader_with_different_level. Retrieved 6/14 statements.
# Partially parsed test_loader_nested_packages. Retrieved 8/20 statements.
# Partially parsed test_loader_with_stub_file. Retrieved 6/14 statements.
# Partially parsed test_loader_returns_string. Retrieved 6/15 statements.
# Partially parsed test_loader_with_constants. Retrieved 6/14 statements.
# Partially parsed test_loader_with_all_defined. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'Test loader with basic package structure.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\ndef func(): pass'
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader with submodule.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Package."""\nVAR = 1'
    var_4 = 'sub.py'
    var_5 = '"""Submodule."""\ndef subfunc(): pass'
    var_6 = False
    var_7 = 1

def test_case_0():
    var_0 = 'Test loader with class definition.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Package."""\nclass MyClass:\n    """A class."""\n    pass'
    var_4 = True

def test_case_0():
    var_0 = 'Test loader with link disabled.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test."""\nCONST = 42'
    var_4 = False
    var_5 = 1

def test_case_0():
    var_0 = 'Test loader with table of contents enabled.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test."""\ndef func1(): pass\ndef func2(): pass'
    var_4 = True

def test_case_0():
    var_0 = 'Test loader with different heading level.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test."""\ndef func(): pass'
    var_4 = False
    var_5 = 2

def test_case_0():
    var_0 = 'Test loader with nested packages.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Root package."""\nROOT = 1'
    var_4 = 'sub'
    var_5 = '"""Sub package."""\nSUB = 2'
    var_6 = True
    var_7 = False

def test_case_0():
    var_0 = 'Test loader with .pyi stub file.'
    var_1 = 'test_pkg'
    var_2 = '__init__.pyi'
    var_3 = '"""Stub."""\ndef stub_func() -> int: ...'
    var_4 = False
    var_5 = 1

def test_case_0():
    var_0 = 'Test that loader returns a string.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Package."""'
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader with module constants.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Package."""\nCONST1: int = 1\nCONST2: str = \'test\''
    var_4 = False
    var_5 = 1

def test_case_0():
    var_0 = 'Test loader with __all__ defined.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Package."""\n__all__ = [\'public_func\']\ndef public_func(): pass\ndef _private(): pass'
    var_4 = False
    var_5 = 1



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/8 statements.
# Partially parsed test_write_overwrites_existing_file. Retrieved 4/10 statements.
# Partially parsed test_write_empty_string. Retrieved 3/8 statements.
# Partially parsed test_write_multiline_content. Retrieved 3/7 statements.
# Partially parsed test_write_unicode_content. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = 'utf-8'

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Original content'
    var_2 = 'New content'
    var_3 = 'utf-8'

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
    var_1 = 'Hello 世界 🌍'
    var_2 = 'utf-8'



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_write_creates_and_writes_file. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_read_returns_file_contents. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_read_file_opens_in_read_mode. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'test content'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_loader_basic. Retrieved 7/18 statements.
# Partially parsed test_loader_with_toc. Retrieved 6/16 statements.
# Partially parsed test_loader_without_link. Retrieved 7/17 statements.
# Partially parsed test_loader_with_level. Retrieved 8/18 statements.
# Partially parsed test_loader_multiple_modules. Retrieved 9/21 statements.
# Partially parsed test_loader_stub_file. Retrieved 7/17 statements.
# Partially parsed test_loader_class_definition. Retrieved 7/17 statements.
# Partially parsed test_loader_nested_package. Retrieved 9/23 statements.


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\n'
    var_3 = 'module.py'
    var_4 = 'def func():\n    """Test function."""\n    pass\n'
    var_5 = True
    var_6 = False

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\n'
    var_3 = 'module.py'
    var_4 = 'def func():\n    """Test function."""\n    pass\n'
    var_5 = True

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\n'
    var_3 = 'module.py'
    var_4 = 'def func():\n    """Test function."""\n    pass\n'
    var_5 = False
    var_6 = 1

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\n'
    var_3 = 'module.py'
    var_4 = 'def func():\n    """Test function."""\n    pass\n'
    var_5 = True
    var_6 = 2
    var_7 = False

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\n'
    var_3 = 'module1.py'
    var_4 = 'def func1():\n    """Function 1."""\n    pass\n'
    var_5 = 'module2.py'
    var_6 = 'def func2():\n    """Function 2."""\n    pass\n'
    var_7 = True
    var_8 = False

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.pyi'
    var_2 = '"""Test package stub."""\n'
    var_3 = 'module.pyi'
    var_4 = 'def func() -> None: ...\n'
    var_5 = True
    var_6 = False

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\n'
    var_3 = 'module.py'
    var_4 = 'class MyClass:\n    """Test class."""\n    pass\n'
    var_5 = True
    var_6 = False

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\n'
    var_3 = 'sub'
    var_4 = '"""Sub package."""\n'
    var_5 = 'module.py'
    var_6 = 'def sub_func():\n    """Sub function."""\n    pass\n'
    var_7 = True
    var_8 = False



# Parsed testcases at query #2
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
    var_1 = 'Line 1\nLine 2\nLine 3'

def test_case_0():
    var_0 = 'special.txt'
    var_1 = 'Special chars: !@#$%^&*()\n\t'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_load_module_success. Retrieved 9/24 statements.
# Partially parsed test_load_module_import_error. Retrieved 4/10 statements.
# Partially parsed test_load_module_invalid_spec. Retrieved 8/17 statements.
# Partially parsed test_load_module_with_docstring. Retrieved 11/24 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = '__init__.py'
    var_2 = ''
    var_3 = 'test_module.py'
    var_4 = "def foo():\n    '''Test function'''\n    pass\n"
    var_5 = 'PYTHONPATH'
    var_6 = 0
    var_7 = module_0.Parser()
    var_8 = 'test_package.test_module'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'orphan_module.py'
    var_1 = 'def foo(): pass\n'
    var_2 = module_0.Parser()
    var_3 = 'nonexistent.orphan_module'

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'valid_package'
    var_1 = '__init__.py'
    var_2 = ''
    var_3 = 0
    var_4 = module_0.Parser()
    var_5 = 'valid_package.nonexistent'
    var_6 = '/nonexistent/path.py'
    var_7 = module_1._load_module(var_5, var_6, var_4)
    assert var_7 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = 'doc_package'
    var_1 = '__init__.py'
    var_2 = ''
    var_3 = 'doc_module.py'
    var_4 = '"""Module docstring"""\n\ndef bar():\n    """Bar function"""\n    pass\n'
    var_5 = 0
    var_6 = module_0.Parser()
    var_7 = 'doc_package.doc_module'
    var_8 = var_6.docstring
    var_9 = len(var_8)
    var_10 = var_9 >= var_5



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_gen_api_creates_prefix_directory. Retrieved 3/10 statements.
# Partially parsed test_gen_api_with_dry_run. Retrieved 8/16 statements.
# Partially parsed test_gen_api_with_valid_module. Retrieved 10/18 statements.
# Partially parsed test_gen_api_writes_file_when_not_dry. Retrieved 12/22 statements.
# Partially parsed test_gen_api_appends_to_sys_path. Retrieved 8/17 statements.
# Partially parsed test_gen_api_returns_sequence_of_strings. Retrieved 13/23 statements.
# Partially parsed test_gen_api_replaces_underscores_in_filename. Retrieved 12/21 statements.
# Partially parsed test_gen_api_includes_title_in_output. Retrieved 12/19 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.gen_api(var_0, dry=var_1)

def test_case_0():
    var_0 = 'test_docs'
    var_1 = {}
    var_2 = True

def test_case_0():
    var_0 = 'test_docs'
    var_1 = 'apimd.loader._site_path'
    var_2 = ''
    var_3 = lambda x: var_2
    var_4 = 'Test'
    var_5 = 'nonexistent_module'
    var_6 = {var_4: var_5}
    var_7 = True

def test_case_0():
    var_0 = 'test_docs'
    var_1 = 'apimd.loader._site_path'
    var_2 = ''
    var_3 = lambda x: var_2
    var_4 = 'apimd.loader.loader'
    var_5 = lambda *args: var_2
    var_6 = 'Test'
    var_7 = 'sys'
    var_8 = {var_6: var_7}
    var_9 = True

def test_case_0():
    var_0 = 'test_docs'
    var_1 = 'apimd.loader._site_path'
    var_2 = ''
    var_3 = lambda x: var_2
    var_4 = 'apimd.loader.loader'
    var_5 = 'test content'
    var_6 = lambda *args: var_5
    var_7 = 'TestModule'
    var_8 = 'test_mod'
    var_9 = {var_7: var_8}
    var_10 = False
    var_11 = 'test-mod-api.md'

def test_case_0():
    var_0 = 'apimd.loader._site_path'
    var_1 = ''
    var_2 = lambda x: var_1
    var_3 = 'apimd.loader.loader'
    var_4 = lambda *args: var_1
    var_5 = {}
    var_6 = 'docs'
    var_7 = True

def test_case_0():
    var_0 = 'test_docs'
    var_1 = 'apimd.loader._site_path'
    var_2 = ''
    var_3 = lambda x: var_2
    var_4 = 'apimd.loader.loader'
    var_5 = 'doc1\ndoc2'
    var_6 = lambda *args: var_5
    var_7 = 'Title1'
    var_8 = 'Title2'
    var_9 = 'mod1'
    var_10 = 'mod2'
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = True

def test_case_0():
    var_0 = 'test_docs'
    var_1 = 'apimd.loader._site_path'
    var_2 = ''
    var_3 = lambda x: var_2
    var_4 = 'apimd.loader.loader'
    var_5 = 'content'
    var_6 = lambda *args: var_5
    var_7 = 'Test'
    var_8 = 'my_test_module'
    var_9 = {var_7: var_8}
    var_10 = False
    var_11 = 'my-test-module-api.md'

def test_case_0():
    var_0 = 'test_docs'
    var_1 = 'apimd.loader._site_path'
    var_2 = ''
    var_3 = lambda x: var_2
    var_4 = 'apimd.loader.loader'
    var_5 = 'content'
    var_6 = lambda *args: var_5
    var_7 = 'MyTitle'
    var_8 = 'mymod'
    var_9 = {var_7: var_8}
    var_10 = 2
    var_11 = True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_gen_api_creates_prefix_directory. Retrieved 6/15 statements.
# Partially parsed test_gen_api_appends_to_sys_path. Retrieved 5/14 statements.
# Partially parsed test_gen_api_writes_file_when_not_dry. Retrieved 5/14 statements.
# Partially parsed test_gen_api_skips_empty_modules. Retrieved 4/12 statements.
# Partially parsed test_gen_api_with_custom_level. Retrieved 8/15 statements.
# Partially parsed test_gen_api_with_link_false. Retrieved 7/14 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.gen_api(var_0)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = len(var_4)
    assert var_5 == 1

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 'docs'
    var_4 = module_0.gen_api(var_2, prefix=var_3)
    var_5 = 1

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = '/custom/path'
    var_4 = module_0.gen_api(var_2, var_3)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Title'
    var_1 = 'module_name'
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = module_0.gen_api(var_2, dry=var_3)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'module'
    var_2 = {var_0: var_1}
    var_3 = module_0.gen_api(var_2)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'First'
    var_1 = 'Second'
    var_2 = 'mod1'
    var_3 = 'mod2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.gen_api(var_4)
    var_6 = len(var_5)
    assert var_6 == 2

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'mod'
    var_2 = {var_0: var_1}
    var_3 = 3
    var_4 = module_0.gen_api(var_2, level=var_3)
    var_5 = ''
    var_6 = True
    var_7 = False

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'mod'
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = module_0.gen_api(var_2, link=var_3)
    var_5 = ''
    var_6 = 1



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_loader_basic. Retrieved 6/15 statements.
# Partially parsed test_loader_with_submodule. Retrieved 8/17 statements.
# Partially parsed test_loader_with_toc. Retrieved 5/12 statements.
# Partially parsed test_loader_with_class. Retrieved 6/13 statements.
# Partially parsed test_loader_without_link. Retrieved 6/13 statements.
# Partially parsed test_loader_different_level. Retrieved 7/14 statements.
# Partially parsed test_loader_with_stub_file. Retrieved 6/13 statements.
# Partially parsed test_loader_with_multiple_modules. Retrieved 10/21 statements.
# Partially parsed test_loader_with_constants. Retrieved 6/13 statements.
# Partially parsed test_loader_nested_packages. Retrieved 8/19 statements.


def test_case_0():
    var_0 = 'Test loader with basic package structure.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\ndef func1(): pass'
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader with submodule.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""'
    var_4 = 'submod.py'
    var_5 = '"""Submodule."""\ndef sub_func(): pass'
    var_6 = True
    var_7 = False

def test_case_0():
    var_0 = 'Test loader with table of contents enabled.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\ndef func1(): pass'
    var_4 = True

def test_case_0():
    var_0 = 'Test loader with class definition.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\nclass MyClass:\n    """A test class."""\n    pass'
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader with link disabled.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""'
    var_4 = False
    var_5 = 1

def test_case_0():
    var_0 = 'Test loader with different heading level.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\ndef func1(): pass'
    var_4 = True
    var_5 = 2
    var_6 = False

def test_case_0():
    var_0 = 'Test loader with .pyi stub file.'
    var_1 = 'test_pkg'
    var_2 = '__init__.pyi'
    var_3 = '"""Test package."""\ndef func1() -> int: ...'
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader with multiple modules.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""'
    var_4 = 'mod1.py'
    var_5 = '"""Module 1."""\ndef func1(): pass'
    var_6 = 'mod2.py'
    var_7 = '"""Module 2."""\ndef func2(): pass'
    var_8 = True
    var_9 = False

def test_case_0():
    var_0 = 'Test loader with constants.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\nVERSION = \'1.0.0\''
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader with nested packages.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""'
    var_4 = 'sub'
    var_5 = '"""Subpackage."""\ndef sub_func(): pass'
    var_6 = True
    var_7 = False



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_loader. Retrieved 8/19 statements.
# Partially parsed test_loader_with_toc. Retrieved 5/13 statements.
# Partially parsed test_loader_no_link. Retrieved 6/15 statements.
# Partially parsed test_loader_with_different_level. Retrieved 7/16 statements.
# Partially parsed test_loader_nonexistent_package. Retrieved 4/8 statements.
# Partially parsed test_loader_multiple_modules. Retrieved 9/22 statements.


def test_case_0():
    var_0 = 'Test loader function with a simple package structure.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\n\ndef test_func():\n    """Test function."""\n    pass\n'
    var_4 = 'submod.py'
    var_5 = '"""Submodule."""\n\nclass TestClass:\n    """Test class."""\n    pass\n'
    var_6 = True
    var_7 = False

def test_case_0():
    var_0 = 'Test loader function with table of contents enabled.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\n'
    var_4 = True

def test_case_0():
    var_0 = 'Test loader function with link disabled.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\n\ndef foo():\n    """Foo function."""\n    pass\n'
    var_4 = False
    var_5 = 1

def test_case_0():
    var_0 = 'Test loader function with different heading level.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\n'
    var_4 = True
    var_5 = 2
    var_6 = False

def test_case_0():
    var_0 = 'Test loader with nonexistent package.'
    var_1 = 'nonexistent_pkg'
    var_2 = True
    var_3 = False

def test_case_0():
    var_0 = 'Test loader with multiple modules in package.'
    var_1 = 'multi_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Multi package."""\n'
    var_4 = 'mod1.py'
    var_5 = '"""Module 1."""\n\ndef func1():\n    """Function 1."""\n    pass\n'
    var_6 = 'mod2.py'
    var_7 = '"""Module 2."""\n\ndef func2():\n    """Function 2."""\n    pass\n'
    var_8 = True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_loader_basic. Retrieved 5/16 statements.
# Partially parsed test_loader_with_multiple_modules. Retrieved 7/18 statements.
# Partially parsed test_loader_with_toc_enabled. Retrieved 4/13 statements.
# Partially parsed test_loader_with_different_levels. Retrieved 6/18 statements.
# Partially parsed test_loader_without_link. Retrieved 5/14 statements.
# Partially parsed test_loader_nested_packages. Retrieved 7/20 statements.
# Partially parsed test_loader_with_class_definition. Retrieved 5/14 statements.
# Partially parsed test_loader_empty_package. Retrieved 5/14 statements.
# Partially parsed test_loader_with_stub_file. Retrieved 5/14 statements.
# Partially parsed test_loader_combined_py_and_pyi. Retrieved 7/18 statements.


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = 'def hello(): pass'
    var_3 = True
    var_4 = False

def test_case_0():
    var_0 = 'mypackage'
    var_1 = '__init__.py'
    var_2 = "\n'''Package docstring'''\ndef func1(): pass\n"
    var_3 = 'module1.py'
    var_4 = "\n'''Module docstring'''\ndef func2(): pass\n"
    var_5 = True
    var_6 = False

def test_case_0():
    var_0 = 'pkg_with_toc'
    var_1 = '__init__.py'
    var_2 = 'def example(): pass'
    var_3 = True

def test_case_0():
    var_0 = 'level_pkg'
    var_1 = '__init__.py'
    var_2 = 'def test(): pass'
    var_3 = True
    var_4 = False
    var_5 = 2

def test_case_0():
    var_0 = 'no_link_pkg'
    var_1 = '__init__.py'
    var_2 = 'def method(): pass'
    var_3 = False
    var_4 = 1

def test_case_0():
    var_0 = 'nested'
    var_1 = '__init__.py'
    var_2 = "'''Nested package'''"
    var_3 = 'submodule'
    var_4 = 'def nested_func(): pass'
    var_5 = True
    var_6 = False

def test_case_0():
    var_0 = 'class_pkg'
    var_1 = '__init__.py'
    var_2 = "\nclass MyClass:\n    '''A test class'''\n    def method(self): pass\n"
    var_3 = True
    var_4 = False

def test_case_0():
    var_0 = 'empty_pkg'
    var_1 = '__init__.py'
    var_2 = ''
    var_3 = True
    var_4 = False

def test_case_0():
    var_0 = 'stub_pkg'
    var_1 = '__init__.pyi'
    var_2 = 'def stub_func(): ...'
    var_3 = True
    var_4 = False

def test_case_0():
    var_0 = 'combined_pkg'
    var_1 = '__init__.py'
    var_2 = 'def py_func(): pass'
    var_3 = '__init__.pyi'
    var_4 = 'def stub_func(): ...'
    var_5 = True
    var_6 = False



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_loader_predicate_line_15_false. Retrieved 11/21 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 15 evaluates to False when .py file is not found.'
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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_loader_predicate_line_13_false. Retrieved 9/20 statements.


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
    assert var_8 == 'compiled_output'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_load_module_returns_false_when_loader_not_instance_of_loader. Retrieved 4/12 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '/fake/path/test_module.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False



# Parsed testcases at query #12
#--------------------------




def test_case_0():
    var_0 = 'Test that the predicate at line 13 (ext == ".py") evaluates to False.'
    var_1 = '.pyi'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_gen_api_predicate_line_25_true. Retrieved 8/14 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 25 evaluates to True when doc.strip() is empty.'
    var_1 = 'Test'
    var_2 = 'test_module'
    var_3 = {var_1: var_2}
    var_4 = 'docs'
    var_5 = True
    var_6 = module_0.gen_api(var_3, prefix=var_4, dry=var_5)
    var_7 = "'test_module' can not be found"



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_gen_api_basic. Retrieved 13/22 statements.
# Partially parsed test_gen_api_multiple_packages. Retrieved 12/21 statements.
# Partially parsed test_gen_api_empty_doc. Retrieved 10/19 statements.
# Partially parsed test_gen_api_creates_directory. Retrieved 10/19 statements.
# Partially parsed test_gen_api_with_level. Retrieved 11/19 statements.
# Partially parsed test_gen_api_with_sys_path. Retrieved 11/23 statements.
# Partially parsed test_gen_api_write_file. Retrieved 11/23 statements.


def test_case_0():
    var_0 = 'Test gen_api with basic parameters.'
    var_1 = '## Module\n\nDocumentation'
    var_2 = 'apimd.loader.loader'
    var_3 = 'apimd.loader._site_path'
    var_4 = '/fake/path'
    var_5 = lambda x: var_4
    var_6 = 'docs'
    var_7 = 'Test Module'
    var_8 = 'test_module'
    var_9 = {var_7: var_8}
    var_10 = None
    var_11 = True
    var_12 = False

def test_case_0():
    var_0 = 'Test gen_api with multiple packages.'
    var_1 = 'apimd.loader.loader'
    var_2 = 'apimd.loader._site_path'
    var_3 = '/fake/path'
    var_4 = lambda x: var_3
    var_5 = 'docs'
    var_6 = 'Package A'
    var_7 = 'Package B'
    var_8 = 'pkg_a'
    var_9 = 'pkg_b'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = True

def test_case_0():
    var_0 = 'Test gen_api when loader returns empty documentation.'
    var_1 = 'apimd.loader.loader'
    var_2 = 'apimd.loader._site_path'
    var_3 = '/fake/path'
    var_4 = lambda x: var_3
    var_5 = 'docs'
    var_6 = 'Empty Module'
    var_7 = 'empty_mod'
    var_8 = {var_6: var_7}
    var_9 = True

def test_case_0():
    var_0 = "Test gen_api creates prefix directory if it doesn't exist."
    var_1 = 'apimd.loader.loader'
    var_2 = 'apimd.loader._site_path'
    var_3 = '/fake/path'
    var_4 = lambda x: var_3
    var_5 = 'new_docs'
    var_6 = 'Test'
    var_7 = 'test'
    var_8 = {var_6: var_7}
    var_9 = True

def test_case_0():
    var_0 = 'Test gen_api respects the level parameter.'
    var_1 = 'apimd.loader.loader'
    var_2 = 'apimd.loader._site_path'
    var_3 = '/fake/path'
    var_4 = lambda x: var_3
    var_5 = 'docs'
    var_6 = 'Title'
    var_7 = 'module'
    var_8 = {var_6: var_7}
    var_9 = 2
    var_10 = True

def test_case_0():
    var_0 = 'Test gen_api appends pwd to sys.path.'
    var_1 = 'apimd.loader.loader'
    var_2 = 'apimd.loader._site_path'
    var_3 = '/fake/path'
    var_4 = lambda x: var_3
    var_5 = 'docs'
    var_6 = '/custom/path'
    var_7 = 'Test'
    var_8 = 'test'
    var_9 = {var_7: var_8}
    var_10 = True

def test_case_0():
    var_0 = 'Test gen_api writes files when dry=False.'
    var_1 = 'apimd.loader.loader'
    var_2 = 'apimd.loader._site_path'
    var_3 = '/fake/path'
    var_4 = lambda x: var_3
    var_5 = 'docs'
    var_6 = 'My Module'
    var_7 = 'my_module'
    var_8 = {var_6: var_7}
    var_9 = False
    var_10 = 'my-module-api.md'



# Parsed testcases at query #15
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
    var_1 = 'Hello, 世界! 🌍'
    var_2 = 'utf-8'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_loader_predicate_pure_py_false. Retrieved 10/22 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Ensure that the predicate at line 15 (if pure_py:) evaluates to False.'
    var_1 = 'test_module'
    var_2 = '/fake/path/test_module'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = '/root'
    var_6 = '/pwd'
    var_7 = False
    var_8 = 1
    var_9 = module_0.loader(var_5, var_6, var_7, var_8, var_7)
    assert var_9 == 'compiled'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_loader_predicate_pure_py_false. Retrieved 10/26 statements.


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
    assert var_9 == 'compiled'



# Parsed testcases at query #18
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

import apimd.loader as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = module_0._site_path(var_0)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_gen_api_predicate_line_25. Retrieved 17/37 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 25 (if not doc.strip()) evaluates to True when doc is empty.'
    var_1 = True
    var_2 = None
    var_3 = 'Logger'
    var_4 = ()
    var_5 = 'info'
    var_6 = 'warning'
    var_7 = lambda self, x: var_2
    var_8 = lambda self, x: var_2
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = type(var_3, var_4, var_9)
    var_11 = 'test'
    var_12 = 'test_module'
    var_13 = {var_11: var_12}
    var_14 = True
    var_15 = module_0.gen_api(var_13, dry=var_14)
    var_16 = len(var_15)
    assert var_16 == 0



# Parsed testcases at query #20
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



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_load_module_success. Retrieved 5/22 statements.
# Partially parsed test_load_module_import_error. Retrieved 5/12 statements.
# Partially parsed test_load_module_spec_none. Retrieved 5/12 statements.
# Partially parsed test_load_module_loader_not_instance. Retrieved 5/15 statements.
# Partially parsed test_load_module_calls_load_docstring. Retrieved 6/20 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module.py'
    var_1 = 'def test_func():\n    """Test function."""\n    pass\n'
    var_2 = module_0.Parser()
    var_3 = 'test_module'
    var_4 = module_1._load_module(var_3, var_1, var_2)
    assert var_4 is True

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module.py'
    var_1 = 'pass\n'
    var_2 = module_0.Parser()
    var_3 = 'nonexistent.module'
    var_4 = module_1._load_module(var_3, var_1, var_2)
    assert var_4 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module.py'
    var_1 = 'pass\n'
    var_2 = module_0.Parser()
    var_3 = 'test_module'
    var_4 = module_1._load_module(var_3, var_1, var_2)
    assert var_4 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module.py'
    var_1 = 'pass\n'
    var_2 = module_0.Parser()
    var_3 = 'test_module'
    var_4 = module_1._load_module(var_3, var_1, var_2)
    assert var_4 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module.py'
    var_1 = 'pass\n'
    var_2 = module_0.Parser()
    var_3 = var_2.load_docstring
    var_4 = 'test_module'
    var_5 = module_1._load_module(var_4, var_1, var_2)
    assert var_5 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_write_file_predicate. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_load_module_predicate_false_when_loader_not_loader_type. Retrieved 4/10 statements.


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



# Parsed testcases at query #24
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



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_load_module_returns_false_when_loader_is_not_instance_of_loader. Retrieved 4/14 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '/path/to/module.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_write_creates_file_with_correct_content. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_gen_api_predicate_line_25_evaluates_to_true. Retrieved 19/26 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 25 evaluates to True when doc.strip() is empty.'
    var_1 = 'apimd.loader.loader'
    var_2 = 'apimd.loader.isdir'
    var_3 = True
    var_4 = lambda x: var_3
    var_5 = 'apimd.loader.logger'
    var_6 = 'MockLogger'
    var_7 = ()
    var_8 = 'info'
    var_9 = 'warning'
    var_10 = None
    var_11 = lambda *args, **kwargs: var_10
    var_12 = lambda *args, **kwargs: var_10
    var_13 = {var_8: var_11, var_9: var_12}
    var_14 = type(var_6, var_7, var_13)
    var_15 = 'TestModule'
    var_16 = 'test_module'
    var_17 = {var_15: var_16}
    var_18 = module_0.gen_api(var_17)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_read_file_opens_in_read_mode. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'test content'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_read_returns_file_contents. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_read_returns_file_content. Retrieved 2/6 statements.
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



# Parsed testcases at query #31
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
    var_1 = "Special chars: !@#$%^&*()_+-=[]{}|;:',.<>?/\\"



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_write_predicate_evaluates_to_false. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'test content'
    var_2 = len(var_1)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_read_file_opens_in_read_mode. Retrieved 1/11 statements.


def test_case_0():
    var_0 = 'test content'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_write_predicate_false. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'test content'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_read_returns_non_empty_string. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'test content'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/8 statements.
# Partially parsed test_write_overwrites_existing_file. Retrieved 4/9 statements.
# Partially parsed test_write_empty_string. Retrieved 3/8 statements.
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
    var_1 = 'Unicode: 你好世界 🌍 Здравствуй'
    var_2 = 'utf-8'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_read_file_returns_content. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'test content'



# Parsed testcases at query #40
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



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_gen_api_dry_mode_predicate. Retrieved 11/21 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 31 (if dry:) evaluates to True.'
    var_1 = 'Test'
    var_2 = 'test_module'
    var_3 = {var_1: var_2}
    var_4 = True
    var_5 = module_0.gen_api(var_3, level=var_4, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = [str(call) for call in var_2]
    var_8 = '='
    var_9 = 12
    var_10 = var_8 * var_9



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_loader_predicate_line_15_false. Retrieved 11/22 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 15 (if pure_py:) evaluates to False.'
    var_1 = 'test_module'
    var_2 = '/fake/path'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = 'def test_func(): pass'
    var_6 = '/root'
    var_7 = '/pwd'
    var_8 = False
    var_9 = 1
    var_10 = module_0.loader(var_6, var_7, var_8, var_9, var_8)
    assert var_10 == 'compiled output'



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_loader_predicate_line_13_false. Retrieved 10/23 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 13 (ext == ".py") evaluates to False.'
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



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_gen_api_dry_mode_predicate. Retrieved 11/20 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 31 (if dry:) evaluates to True.'
    var_1 = 'Test'
    var_2 = 'test_module'
    var_3 = {var_1: var_2}
    var_4 = True
    var_5 = module_0.gen_api(var_3, dry=var_4)
    var_6 = [str(call) for call in var_1]
    var_7 = '='
    var_8 = 12
    var_9 = var_7 * var_8
    var_10 = len(var_5)



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_gen_api_dry_mode_predicate. Retrieved 8/11 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 31 (if dry:) evaluates to True.'
    var_1 = 'Test'
    var_2 = 'os'
    var_3 = {var_1: var_2}
    var_4 = 'test_docs_output'
    var_5 = True
    var_6 = module_0.gen_api(var_3, prefix=var_4, dry=var_5)
    var_7 = len(var_6)



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_gen_api_basic. Retrieved 14/27 statements.
# Partially parsed test_gen_api_multiple_roots. Retrieved 16/29 statements.
# Partially parsed test_gen_api_empty_content. Retrieved 14/27 statements.
# Partially parsed test_gen_api_with_pwd. Retrieved 17/35 statements.
# Partially parsed test_gen_api_write_file. Retrieved 12/27 statements.
# Partially parsed test_gen_api_with_custom_level. Retrieved 15/27 statements.
# Partially parsed test_gen_api_with_link_and_toc. Retrieved 15/28 statements.


def test_case_0():
    var_0 = 'Test gen_api with basic parameters.'
    var_1 = 'docs'
    var_2 = 'apimd.loader.loader'
    var_3 = 'apimd.loader._site_path'
    var_4 = 'apimd.loader.isdir'
    var_5 = False
    var_6 = lambda x: var_5
    var_7 = 'apimd.loader.mkdir'
    var_8 = None
    var_9 = lambda x: var_8
    var_10 = 'Test'
    var_11 = 'test_module'
    var_12 = {var_10: var_11}
    var_13 = True

def test_case_0():
    var_0 = 'Test gen_api with multiple root modules.'
    var_1 = 'docs'
    var_2 = 'apimd.loader.loader'
    var_3 = 'apimd.loader._site_path'
    var_4 = 'apimd.loader.isdir'
    var_5 = False
    var_6 = lambda x: var_5
    var_7 = 'apimd.loader.mkdir'
    var_8 = None
    var_9 = lambda x: var_8
    var_10 = 'Module A'
    var_11 = 'Module B'
    var_12 = 'mod_a'
    var_13 = 'mod_b'
    var_14 = {var_10: var_12, var_11: var_13}
    var_15 = True

def test_case_0():
    var_0 = 'Test gen_api when loader returns empty content.'
    var_1 = 'docs'
    var_2 = 'apimd.loader.loader'
    var_3 = 'apimd.loader._site_path'
    var_4 = 'apimd.loader.isdir'
    var_5 = False
    var_6 = lambda x: var_5
    var_7 = 'apimd.loader.mkdir'
    var_8 = None
    var_9 = lambda x: var_8
    var_10 = 'Empty'
    var_11 = 'empty_module'
    var_12 = {var_10: var_11}
    var_13 = True

def test_case_0():
    var_0 = 'Test gen_api with custom pwd parameter.'
    var_1 = 'docs'
    var_2 = 'site-packages'
    var_3 = []
    var_4 = 'apimd.loader.loader'
    var_5 = 'apimd.loader._site_path'
    var_6 = 'apimd.loader.isdir'
    var_7 = False
    var_8 = lambda x: var_7
    var_9 = 'apimd.loader.mkdir'
    var_10 = None
    var_11 = lambda x: var_10
    var_12 = 'apimd.loader.sys_path.append'
    var_13 = 'Test'
    var_14 = 'test_module'
    var_15 = {var_13: var_14}
    var_16 = True

def test_case_0():
    var_0 = 'Test gen_api writes file when dry=False.'
    var_1 = 'docs'
    var_2 = True
    var_3 = 'apimd.loader.loader'
    var_4 = 'apimd.loader._site_path'
    var_5 = 'apimd.loader.isdir'
    var_6 = lambda x: var_2
    var_7 = 'Test'
    var_8 = 'test_module'
    var_9 = {var_7: var_8}
    var_10 = False
    var_11 = 'test-module-api.md'

def test_case_0():
    var_0 = 'Test gen_api with custom heading level.'
    var_1 = 'docs'
    var_2 = 'apimd.loader.loader'
    var_3 = 'apimd.loader._site_path'
    var_4 = 'apimd.loader.isdir'
    var_5 = False
    var_6 = lambda x: var_5
    var_7 = 'apimd.loader.mkdir'
    var_8 = None
    var_9 = lambda x: var_8
    var_10 = 'Test'
    var_11 = 'test_module'
    var_12 = {var_10: var_11}
    var_13 = 3
    var_14 = True

def test_case_0():
    var_0 = 'Test gen_api passes link and toc parameters to loader.'
    var_1 = 'docs'
    var_2 = []
    var_3 = 'apimd.loader.loader'
    var_4 = 'apimd.loader._site_path'
    var_5 = 'apimd.loader.isdir'
    var_6 = False
    var_7 = lambda x: var_6
    var_8 = 'apimd.loader.mkdir'
    var_9 = None
    var_10 = lambda x: var_9
    var_11 = 'Test'
    var_12 = 'test_module'
    var_13 = {var_11: var_12}
    var_14 = True



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_gen_api_with_dry_run. Retrieved 9/20 statements.
# Partially parsed test_gen_api_creates_directory. Retrieved 14/27 statements.
# Partially parsed test_gen_api_skips_empty_documentation. Retrieved 9/20 statements.
# Partially parsed test_gen_api_multiple_packages. Retrieved 12/23 statements.
# Partially parsed test_gen_api_writes_file. Retrieved 13/26 statements.
# Partially parsed test_gen_api_with_pwd. Retrieved 10/26 statements.
# Partially parsed test_gen_api_different_levels. Retrieved 10/20 statements.


def test_case_0():
    var_0 = 'Test gen_api with dry run mode.'
    var_1 = 'apimd.loader.loader'
    var_2 = 'apimd.loader._site_path'
    var_3 = 'apimd.loader.isdir'
    var_4 = True
    var_5 = lambda x: var_4
    var_6 = 'Test Package'
    var_7 = 'test_pkg'
    var_8 = {var_6: var_7}

def test_case_0():
    var_0 = "Test gen_api creates prefix directory if it doesn't exist."
    var_1 = 'apimd.loader.loader'
    var_2 = 'apimd.loader._site_path'
    var_3 = 'apimd.loader.isdir'
    var_4 = False
    var_5 = lambda x: var_4
    var_6 = 'apimd.loader.mkdir'
    var_7 = None
    var_8 = lambda x: var_7
    var_9 = 'new_prefix'
    var_10 = 'Test'
    var_11 = 'test'
    var_12 = {var_10: var_11}
    var_13 = True

def test_case_0():
    var_0 = 'Test gen_api skips packages that produce empty documentation.'
    var_1 = 'apimd.loader.loader'
    var_2 = 'apimd.loader._site_path'
    var_3 = 'apimd.loader.isdir'
    var_4 = True
    var_5 = lambda x: var_4
    var_6 = 'Empty Package'
    var_7 = 'empty_pkg'
    var_8 = {var_6: var_7}

def test_case_0():
    var_0 = 'Test gen_api with multiple root packages.'
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
    var_11 = 2

def test_case_0():
    var_0 = 'Test gen_api writes documentation to file.'
    var_1 = []
    var_2 = 'apimd.loader.loader'
    var_3 = 'apimd.loader._site_path'
    var_4 = 'apimd.loader.isdir'
    var_5 = True
    var_6 = lambda x: var_5
    var_7 = 'apimd.loader._write'
    var_8 = 'My Pkg'
    var_9 = 'my_pkg'
    var_10 = {var_8: var_9}
    var_11 = False
    var_12 = len(var_1)
    assert var_12 == 1

def test_case_0():
    var_0 = 'Test gen_api appends pwd to sys.path.'
    var_1 = 'apimd.loader.loader'
    var_2 = 'apimd.loader._site_path'
    var_3 = 'apimd.loader.isdir'
    var_4 = True
    var_5 = lambda x: var_4
    var_6 = 'custom_path'
    var_7 = 'Test'
    var_8 = 'test'
    var_9 = {var_7: var_8}

def test_case_0():
    var_0 = 'Test gen_api with different heading levels.'
    var_1 = 'apimd.loader.loader'
    var_2 = 'apimd.loader._site_path'
    var_3 = 'apimd.loader.isdir'
    var_4 = True
    var_5 = lambda x: var_4
    var_6 = 'Test'
    var_7 = 'test'
    var_8 = {var_6: var_7}
    var_9 = 3



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_read_file_opens_in_read_mode. Retrieved 1/11 statements.


def test_case_0():
    var_0 = 'test content'



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_write_predicate_evaluates_to_false. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'test content'
    var_2 = "\ndef _write(path: str, doc: str) -> None:\n    with open(path, 'w+', encoding='utf-8') as f:\n        f.write(doc)\n"
    var_3 = exec(var_2)



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_read_returns_file_content. Retrieved 4/14 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'
    var_2 = 'r'
    var_3 = None



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_write_creates_and_writes_file. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_loader_predicate_false_when_no_py_file. Retrieved 8/20 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 15 evaluates to False when no .py file is found.'
    var_1 = 'test_module'
    var_2 = '/path/to/test_module'
    var_3 = (var_1, var_2)
    var_4 = '/root'
    var_5 = '/pwd'
    var_6 = True
    var_7 = module_0.loader(var_4, var_5, var_6, var_6, var_6)
    assert var_7 == 'compiled_result'



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_gen_api_dry_mode_predicate. Retrieved 11/20 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 31 (if dry:) evaluates to True.'
    var_1 = 'Test'
    var_2 = 'test_module'
    var_3 = {var_1: var_2}
    var_4 = True
    var_5 = module_0.gen_api(var_3, dry=var_4)
    var_6 = [str(call) for call in var_1]
    var_7 = '='
    var_8 = 12
    var_9 = var_7 * var_8
    var_10 = len(var_5)
    assert var_10 == 1



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_load_module_success. Retrieved 8/24 statements.
# Partially parsed test_load_module_parent_import_error. Retrieved 4/10 statements.
# Partially parsed test_load_module_invalid_spec. Retrieved 8/20 statements.
# Partially parsed test_load_module_with_docstring. Retrieved 8/24 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = ''
    var_3 = 'test_module.py'
    var_4 = 'def test_func():\n    """Test function."""\n    pass\n'
    var_5 = 0
    var_6 = module_0.Parser()
    var_7 = 'test_pkg.test_module'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module.py'
    var_1 = 'def test_func():\n    pass\n'
    var_2 = module_0.Parser()
    var_3 = 'nonexistent.parent.test_module'

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = ''
    var_3 = 0
    var_4 = module_0.Parser()
    var_5 = 'test_pkg.nonexistent'
    var_6 = '/nonexistent/path.py'
    var_7 = module_1._load_module(var_5, var_6, var_4)
    assert var_7 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = ''
    var_3 = 'documented.py'
    var_4 = '"""Module with docstring."""\n\ndef func():\n    """Function doc."""\n    pass\n'
    var_5 = 0
    var_6 = module_0.Parser()
    var_7 = 'test_pkg.documented'



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_read_returns_file_content. Retrieved 2/6 statements.
# Partially parsed test_read_returns_empty_string_for_empty_file. Retrieved 2/6 statements.
# Partially parsed test_read_returns_multiline_content. Retrieved 2/6 statements.
# Partially parsed test_read_preserves_special_characters. Retrieved 2/6 statements.
# Partially parsed test_read_preserves_whitespace. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'test_script.txt'
    var_1 = 'Hello, World!'

def test_case_0():
    var_0 = 'empty_script.txt'
    var_1 = ''

def test_case_0():
    var_0 = 'multiline_script.txt'
    var_1 = 'line1\nline2\nline3'

def test_case_0():
    var_0 = 'special_script.txt'
    var_1 = "Special chars: !@#$%^&*()_+-=[]{}|;:',.<>?/"

def test_case_0():
    var_0 = 'whitespace_script.txt'
    var_1 = '  leading\n\ttabs\n  spaces  '



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_loader_basic. Retrieved 5/17 statements.
# Partially parsed test_loader_with_toc. Retrieved 4/15 statements.
# Partially parsed test_loader_without_link. Retrieved 5/16 statements.
# Partially parsed test_loader_with_different_level. Retrieved 6/17 statements.
# Partially parsed test_loader_multiple_modules. Retrieved 7/21 statements.
# Partially parsed test_loader_empty_package. Retrieved 5/16 statements.


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\ndef test_func():\n    """Test function."""\n    pass\n'
    var_3 = True
    var_4 = False

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\ndef func():\n    """Function."""\n    pass\n'
    var_3 = True

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\nclass TestClass:\n    """Test class."""\n    pass\n'
    var_3 = False
    var_4 = 1

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package."""\ndef func():\n    """Function."""\n    pass\n'
    var_3 = True
    var_4 = 2
    var_5 = False

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Package."""\ndef pkg_func():\n    """Package function."""\n    pass\n'
    var_3 = 'module.py'
    var_4 = '"""Module."""\ndef mod_func():\n    """Module function."""\n    pass\n'
    var_5 = True
    var_6 = False

def test_case_0():
    var_0 = 'empty_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Empty package."""\n'
    var_3 = True
    var_4 = False



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_write_creates_file_with_correct_content. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_read_returns_file_content. Retrieved 2/6 statements.
# Partially parsed test_read_returns_empty_string_for_empty_file. Retrieved 2/6 statements.
# Partially parsed test_read_returns_multiline_content. Retrieved 2/6 statements.
# Partially parsed test_read_with_special_characters. Retrieved 2/6 statements.


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
    var_1 = "Special chars: !@#$%^&*()_+-=[]{}|;:',.<>?/"



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_write_file_predicate. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'test content'



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_gen_api_dry_mode_predicate. Retrieved 9/12 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 31 (if dry:) evaluates to True.'
    var_1 = 'Test'
    var_2 = 'os'
    var_3 = {var_1: var_2}
    var_4 = None
    var_5 = '/tmp/test_docs'
    var_6 = True
    var_7 = False
    var_8 = module_0.gen_api(var_3, var_4, prefix=var_5, link=var_6, level=var_6, toc=var_7, dry=var_6)



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_loader. Retrieved 8/19 statements.
# Partially parsed test_loader_with_toc. Retrieved 5/12 statements.
# Partially parsed test_loader_without_link. Retrieved 6/13 statements.
# Partially parsed test_loader_different_level. Retrieved 7/14 statements.
# Partially parsed test_loader_with_stub_file. Retrieved 6/13 statements.
# Partially parsed test_loader_empty_package. Retrieved 6/13 statements.
# Partially parsed test_loader_multiple_modules. Retrieved 10/21 statements.


def test_case_0():
    var_0 = 'Test loader function with a sample package structure.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\ndef foo(): pass'
    var_4 = 'subpkg'
    var_5 = '"""Subpackage."""\nclass Bar: pass'
    var_6 = True
    var_7 = False

def test_case_0():
    var_0 = 'Test loader function with table of contents enabled.'
    var_1 = 'toc_pkg'
    var_2 = '__init__.py'
    var_3 = '"""TOC test package."""\ndef func(): pass'
    var_4 = True

def test_case_0():
    var_0 = 'Test loader function with link disabled.'
    var_1 = 'no_link_pkg'
    var_2 = '__init__.py'
    var_3 = '"""No link test package."""\nFOO = 42'
    var_4 = False
    var_5 = 1

def test_case_0():
    var_0 = 'Test loader function with different heading level.'
    var_1 = 'level_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Level test package."""\nvar = 1'
    var_4 = True
    var_5 = 2
    var_6 = False

def test_case_0():
    var_0 = 'Test loader with .pyi stub file.'
    var_1 = 'stub_pkg'
    var_2 = '__init__.pyi'
    var_3 = '"""Stub package."""\ndef stub_func() -> int: ...'
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader with empty package.'
    var_1 = 'empty_pkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader with multiple modules in a package.'
    var_1 = 'multi_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Main package."""\n'
    var_4 = 'module1.py'
    var_5 = '"""Module 1."""\ndef func1(): pass'
    var_6 = 'module2.py'
    var_7 = '"""Module 2."""\ndef func2(): pass'
    var_8 = True
    var_9 = False



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_load_module_success. Retrieved 10/25 statements.
# Partially parsed test_load_module_import_error. Retrieved 5/11 statements.
# Partially parsed test_load_module_invalid_spec. Retrieved 11/31 statements.
# Partially parsed test_load_module_no_loader. Retrieved 11/32 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module successfully loads a module and docstring.'
    var_1 = 'test_package'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'test_module.py'
    var_5 = '"""Test module docstring."""\ndef func(): pass'
    var_6 = 0
    var_7 = module_0.Parser()
    var_8 = 'test_package.test_module'
    var_9 = module_1._load_module(var_8, var_1, var_7)
    assert var_9 is True

import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test _load_module returns False when parent import fails.'
    var_1 = 'nonexistent_module.py'
    var_2 = '"""Test."""'
    var_3 = module_0.Parser()
    var_4 = 'nonexistent.package.module'

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module returns False when spec is invalid.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'test_mod.py'
    var_5 = '"""Test."""'
    var_6 = 0
    var_7 = 'apimd.loader.spec_from_file_location'
    var_8 = module_0.Parser()
    var_9 = 'test_pkg.test_mod'
    var_10 = module_1._load_module(var_9, var_2, var_8)
    assert var_10 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test _load_module returns False when loader is not available.'
    var_1 = 'test_pkg2'
    var_2 = '__init__.py'
    var_3 = ''
    var_4 = 'test_mod2.py'
    var_5 = '"""Test."""'
    var_6 = 0
    var_7 = 'apimd.loader.spec_from_file_location'
    var_8 = module_0.Parser()
    var_9 = 'test_pkg2.test_mod2'
    var_10 = module_1._load_module(var_9, var_2, var_8)
    assert var_10 is False



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_loader_predicate_line_13_false. Retrieved 9/21 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 13 evaluates to False when ext is ".pyi".'
    var_1 = 'test_module'
    var_2 = '/path/to/test_module'
    var_3 = (var_1, var_2)
    var_4 = '/root'
    var_5 = '/pwd'
    var_6 = False
    var_7 = 1
    var_8 = module_0.loader(var_4, var_5, var_6, var_7, var_6)



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_read_returns_file_content. Retrieved 2/13 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'Hello, World!'



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_write_file_predicate. Retrieved 3/14 statements.


def test_case_0():
    var_0 = '/invalid/path/that/does/not/exist/file.txt'
    var_1 = False
    var_2 = True



# Parsed testcases at query #66
#--------------------------






# Parsed testcases at query #67
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_loader_with_valid_package. Retrieved 8/17 statements.
# Partially parsed test_loader_with_nonexistent_path. Retrieved 4/7 statements.
# Partially parsed test_loader_with_toc_enabled. Retrieved 5/12 statements.
# Partially parsed test_loader_with_link_disabled. Retrieved 6/13 statements.
# Partially parsed test_loader_with_different_level. Retrieved 7/14 statements.
# Partially parsed test_loader_with_stub_file. Retrieved 6/13 statements.
# Partially parsed test_loader_with_nested_modules. Retrieved 8/19 statements.
# Partially parsed test_loader_returns_string. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'Test loader with a valid package structure.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\n\ndef test_func():\n    """Test function."""\n    pass\n'
    var_4 = 'module.py'
    var_5 = '"""Test module."""\n\nclass TestClass:\n    """Test class."""\n    pass\n'
    var_6 = True
    var_7 = False

def test_case_0():
    var_0 = 'Test loader with a nonexistent package path.'
    var_1 = 'nonexistent'
    var_2 = True
    var_3 = False

def test_case_0():
    var_0 = 'Test loader with table of contents enabled.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\n'
    var_4 = True

def test_case_0():
    var_0 = 'Test loader with link disabled.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\n'
    var_4 = False
    var_5 = 1

def test_case_0():
    var_0 = 'Test loader with different heading level.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\n'
    var_4 = True
    var_5 = 2
    var_6 = False

def test_case_0():
    var_0 = 'Test loader with .pyi stub file.'
    var_1 = 'test_pkg'
    var_2 = '__init__.pyi'
    var_3 = '"""Test package stub."""\n\ndef stub_func() -> None: ...\n'
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = 'Test loader with nested package structure.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test package."""\n'
    var_4 = 'subpkg'
    var_5 = '"""Subpackage."""\n'
    var_6 = True
    var_7 = False

def test_case_0():
    var_0 = 'Test that loader always returns a string.'
    var_1 = 'test_pkg'
    var_2 = '__init__.py'
    var_3 = '"""Test."""\n'
    var_4 = True
    var_5 = False



