####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_gen_api_with_valid_root_names. Retrieved 10/11 statements.
# Partially parsed test_gen_api_with_multiple_root_names. Retrieved 14/16 statements.
# Partially parsed test_gen_api_with_custom_prefix. Retrieved 10/11 statements.
# Partially parsed test_gen_api_with_link_disabled. Retrieved 10/11 statements.
# Partially parsed test_gen_api_with_custom_level. Retrieved 11/12 statements.
# Partially parsed test_gen_api_with_toc_enabled. Retrieved 10/11 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'example'
    var_1 = 'example_module'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = True
    var_5 = module_0.gen_api(var_2, prefix=var_3, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 0
    var_8 = var_5[var_7]
    var_9 = '# example API'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = 'invalid_module'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = True
    var_5 = module_0.gen_api(var_2, prefix=var_3, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'example1'
    var_1 = 'example2'
    var_2 = 'example_module1'
    var_3 = 'example_module2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'test_docs'
    var_6 = True
    var_7 = module_0.gen_api(var_4, prefix=var_5, dry=var_6)
    var_8 = len(var_7)
    assert var_8 == 2
    var_9 = 0
    var_10 = var_7[var_9]
    var_11 = '# example1 API'
    var_12 = var_7[var_6]
    var_13 = '# example2 API'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'example'
    var_1 = 'example_module'
    var_2 = {var_0: var_1}
    var_3 = 'custom_prefix'
    var_4 = True
    var_5 = module_0.gen_api(var_2, prefix=var_3, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 0
    var_8 = var_5[var_7]
    var_9 = '# example API'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'example'
    var_1 = 'example_module'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = False
    var_5 = True
    var_6 = module_0.gen_api(var_2, prefix=var_3, link=var_4, dry=var_5)
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = var_6[var_4]
    var_9 = '# example API'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'example'
    var_1 = 'example_module'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = 2
    var_5 = True
    var_6 = module_0.gen_api(var_2, prefix=var_3, level=var_4, dry=var_5)
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 0
    var_9 = var_6[var_8]
    var_10 = '## example API'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'example'
    var_1 = 'example_module'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = True
    var_5 = module_0.gen_api(var_2, prefix=var_3, toc=var_4, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 0
    var_8 = var_5[var_7]
    var_9 = '# example API'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_loader_with_valid_package. Retrieved 6/7 statements.
# Partially parsed test_loader_with_invalid_package. Retrieved 6/7 statements.
# Partially parsed test_loader_with_nonexistent_path. Retrieved 6/7 statements.
# Partially parsed test_loader_with_empty_package. Retrieved 6/7 statements.
# Partially parsed test_loader_with_pure_python_package. Retrieved 6/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '/path/to/test_pkg'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = 'test_pkg'
    var_7 = bool('test_pkg' in var_5)
    assert var_7 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'invalid_pkg'
    var_1 = '/path/to/invalid_pkg'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = 'invalid_pkg'
    var_7 = bool('invalid_pkg' not in var_5)
    assert var_7 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '/nonexistent/path'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = 'test_pkg'
    var_7 = bool('test_pkg' not in var_5)
    assert var_7 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'empty_pkg'
    var_1 = '/path/to/empty_pkg'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = 'empty_pkg'
    var_7 = bool('empty_pkg' in var_5)
    assert var_7 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'pure_py_pkg'
    var_1 = '/path/to/pure_py_pkg'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = 'pure_py_pkg'
    var_7 = bool('pure_py_pkg' in var_5)
    assert var_7 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_loader_continue_not_executed_when_pure_py_is_false. Retrieved 6/10 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = 'pwd'
    var_2 = False
    var_3 = 1
    var_4 = module_0.loader(var_0, var_1, var_2, var_3, var_2)
    var_5 = 'loading extension module for fully documented:'



# Parsed testcases at query #4
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.gen_api(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_loader_with_valid_package. Retrieved 5/14 statements.
# Partially parsed test_loader_with_nonexistent_package. Retrieved 3/6 statements.
# Partially parsed test_loader_with_pyi_file. Retrieved 5/14 statements.
# Partially parsed test_loader_with_toc_enabled. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package docstring."""\n'
    var_3 = True
    var_4 = False
    var_5 = 'Test package docstring'
    var_6 = 'Module `test_pkg`'

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = True
    var_2 = False

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.pyi'
    var_2 = '"""Test package stub."""\n'
    var_3 = True
    var_4 = False
    var_5 = 'Test package stub'
    var_6 = 'Module `test_pkg`'

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package docstring."""\n'
    var_3 = True
    var_4 = '**Table of contents:**'
    var_5 = '+ [`test_pkg`]'



# Parsed testcases at query #6
#--------------------------




def test_case_0():
    var_0 = 'test_module'
    var_1 = '/path/to/test_module'
    var_2 = '.py'
    var_3 = var_1 + var_2
    var_4 = False
    var_5 = True
    assert var_5 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_loader_loads_python_files. Retrieved 6/7 statements.
# Partially parsed test_loader_handles_non_existent_path. Retrieved 6/7 statements.
# Partially parsed test_loader_processes_pure_py_files. Retrieved 6/7 statements.
# Partially parsed test_loader_handles_missing_module. Retrieved 6/7 statements.
# Partially parsed test_loader_handles_extension_modules. Retrieved 6/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = '/fake/path'
    var_1 = 'fake_module'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_1, var_0, var_2, var_3, var_4)

import apimd.loader as module_0

def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = 'non_existent_module'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_1, var_0, var_2, var_3, var_4)

import apimd.loader as module_0

def test_case_0():
    var_0 = '/fake/path'
    var_1 = 'pure_py_module'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_1, var_0, var_2, var_3, var_4)

import apimd.loader as module_0

def test_case_0():
    var_0 = '/fake/path'
    var_1 = 'missing_module'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_1, var_0, var_2, var_3, var_4)

import apimd.loader as module_0

def test_case_0():
    var_0 = '/fake/path'
    var_1 = 'extension_module'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_1, var_0, var_2, var_3, var_4)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_gen_api_basic_usage. Retrieved 10/11 statements.
# Partially parsed test_gen_api_with_dry_run. Retrieved 10/11 statements.
# Partially parsed test_gen_api_with_none_pwd. Retrieved 10/11 statements.
# Partially parsed test_gen_api_with_toc_enabled. Retrieved 10/11 statements.
# Partially parsed test_gen_api_with_custom_level. Retrieved 10/11 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = '/tmp'
    var_4 = 'docs'
    var_5 = True
    var_6 = 1
    var_7 = False
    var_8 = False
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = '/tmp'
    var_4 = 'docs'
    var_5 = True
    var_6 = 1
    var_7 = False
    var_8 = True
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)

import apimd.loader as module_0

def test_case_0():
    var_0 = {}
    var_1 = '/tmp'
    var_2 = 'docs'
    var_3 = True
    var_4 = 1
    var_5 = False
    var_6 = False
    var_7 = module_0.gen_api(var_0, var_1, prefix=var_2, link=var_3, level=var_4, toc=var_5, dry=var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'docs'
    var_5 = True
    var_6 = 1
    var_7 = False
    var_8 = False
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = '/tmp'
    var_4 = 'docs'
    var_5 = True
    var_6 = 1
    var_7 = True
    var_8 = False
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = '/tmp'
    var_4 = 'docs'
    var_5 = True
    var_6 = 2
    var_7 = False
    var_8 = False
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test__load_module_success. Retrieved 4/13 statements.
# Partially parsed test__load_module_no_loader. Retrieved 5/13 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '/path/to/test_module.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is True

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'nonexistent_module'
    var_2 = '/path/to/nonexistent_module.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module_no_loader'
    var_2 = '/path/to/test_module_no_loader.py'
    var_3 = None
    var_4 = module_1._load_module(var_1, var_2, var_0)
    assert var_4 is False



# Parsed testcases at query #10
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = module_0._site_path(var_0)
    var_2 = bool(var_1 != '')
    assert var_2 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existing_package_name'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'sys'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_site_path_with_valid_submodule. Retrieved 3/9 statements.
# Partially parsed test_site_path_with_none_spec. Retrieved 4/7 statements.
# Partially parsed test_site_path_with_none_submodule_locations. Retrieved 3/9 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = '/some/path'
    var_1 = [var_0]
    var_2 = 'valid_module'
    var_3 = module_0._site_path(var_2)
    assert var_3 == '/some/path'

import apimd.loader as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda _: var_0
    var_2 = 'nonexistent_module'
    var_3 = module_0._site_path(var_2)
    assert var_3 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = None
    var_1 = 'module_without_submodules'
    var_2 = module_0._site_path(var_1)
    assert var_2 == ''



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_site_path_predicate_evaluates_to_false. Retrieved 1/3 statements.


def test_case_0():
    var_0 = '/some/path'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_write_file_creates_file_with_content. Retrieved 3/5 statements.
# Partially parsed test_write_file_overwrites_existing_content. Retrieved 5/7 statements.
# Partially parsed test_write_file_with_empty_string. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = module_0._write(var_0, var_1)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Initial content'
    var_2 = 'New content'
    var_3 = module_0._write(var_0, var_1)
    var_4 = module_0._write(var_0, var_2)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = ''
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #14
#--------------------------




def test_case_0():
    var_0 = '.pyi'
    assert var_0 == '.py'
    var_1 = False
    var_2 = '.py'
    assert var_2 is False



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_site_path_predicate_evaluates_to_false. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'some_location'
    var_1 = [var_0]
    var_2 = None



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_load_module_predicate_evaluates_to_false. Retrieved 7/9 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_path'
    var_2 = module_0.Parser()
    var_3 = None
    var_4 = None
    var_5 = var_3 is not var_4
    var_6 = var_3.loader



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_load_module_with_valid_spec_and_loader. Retrieved 4/12 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'mock_path'
    var_2 = 'mock_name'
    var_3 = module_1._load_module(var_2, var_1, var_0)
    assert var_3 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, world!'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_loader. Retrieved 6/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'example'
    var_1 = '/path/to/module'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #20
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'some_root'
    var_1 = 'some_pwd'
    var_2 = False
    var_3 = 1
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test__site_path_with_existing_package. Retrieved 2/3 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = module_0._site_path(var_0)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_package_123'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'builtins'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_write_predicate_evaluates_to_false. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'sample text'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test__read_existing_file. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = module_0._read(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existing_file.txt'
    var_1 = module_0._read(var_0)
    var_2 = bool(True)
    assert var_2 is True
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_write_file_successfully. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_write_opens_file_with_correct_parameters. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'test content'
    var_2 = module_0._write(var_0, var_1)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True



# Parsed testcases at query #26
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = '/non/existent/directory/file.txt'
    var_1 = 'test content'
    var_2 = module_0._write(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test__read_predicate_evaluates_to_true. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'test content'
    var_2 = module_0._read(var_0)
    assert var_2 == 'test content'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/5 statements.
# Partially parsed test_write_overwrites_existing_file. Retrieved 5/7 statements.
# Partially parsed test_write_empty_string. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = module_0._write(var_0, var_1)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Initial Content'
    var_2 = 'New Content'
    var_3 = module_0._write(var_0, var_1)
    var_4 = module_0._write(var_0, var_2)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = ''
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_read_existing_file. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = module_0._read(var_0)
    assert var_2 == 'Hello, World!'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existing_file.txt'
    var_1 = module_0._read(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Test content'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_load_module_predicate_evaluates_to_false_when_s_loader_is_not_Loader. Retrieved 4/8 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_path'
    var_2 = module_0.Parser()
    var_3 = None
    var_4 = module_1._load_module(var_0, var_1, var_2)
    assert var_4 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_path'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is False



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_read_file_successfully. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'test content'
    var_2 = module_0._read(var_0)
    assert var_2 == 'test content'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_read_existing_file. Retrieved 3/7 statements.
# Partially parsed test_read_non_existing_file. Retrieved 2/6 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = module_0._read(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existing_file.txt'
    var_1 = module_0._read(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #34
#--------------------------




import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_path'
    var_2 = module_0.Parser()
    var_3 = None
    var_4 = module_1._load_module(var_0, var_1, var_2)
    assert var_4 is False



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_load_module_predicate_evaluates_to_true. Retrieved 7/9 statements.


import apimd.parser as module_0
import _frozen_importlib_external as module_1

def test_case_0():
    var_0 = 'example_module'
    var_1 = '/path/to/module.py'
    var_2 = module_0.Parser()
    var_3 = module_1.spec_from_file_location(var_0, var_1)
    var_4 = None
    var_5 = var_3 is not var_4
    var_6 = var_3.loader



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_loader_with_valid_root_and_pwd. Retrieved 6/7 statements.
# Partially parsed test_loader_with_invalid_root. Retrieved 6/7 statements.
# Partially parsed test_loader_with_invalid_pwd. Retrieved 6/7 statements.
# Partially parsed test_loader_with_link_false. Retrieved 6/7 statements.
# Partially parsed test_loader_with_toc_false. Retrieved 6/7 statements.
# Partially parsed test_loader_with_level_zero. Retrieved 6/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'example_pkg'
    var_1 = '/path/to/package'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_pkg'
    var_1 = '/path/to/package'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'example_pkg'
    var_1 = '/invalid/path'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'example_pkg'
    var_1 = '/path/to/package'
    var_2 = False
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'example_pkg'
    var_1 = '/path/to/package'
    var_2 = True
    var_3 = 1
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'example_pkg'
    var_1 = '/path/to/package'
    var_2 = True
    var_3 = 0
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #2
#--------------------------




import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'example.module'
    var_1 = '/path/to/module.py'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is True

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'example.module'
    var_1 = '/path/to/module.py'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'example.module'
    var_1 = '/path/to/module.py'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'example.module'
    var_1 = '/path/to/module.py'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is False



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_loader_with_valid_input. Retrieved 7/8 statements.
# Partially parsed test_loader_with_invalid_pwd. Retrieved 7/8 statements.
# Partially parsed test_loader_with_toc_enabled. Retrieved 6/7 statements.
# Partially parsed test_loader_with_link_disabled. Retrieved 6/7 statements.
# Partially parsed test_loader_with_no_modules_found. Retrieved 7/8 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'example'
    var_1 = '/path/to/package'
    var_2 = True
    var_3 = 1
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = len(var_5)
    var_7 = bool(var_6 > 0)
    assert var_7 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'example'
    var_1 = '/invalid/path'
    var_2 = True
    var_3 = 1
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = len(var_5)
    assert var_6 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'example'
    var_1 = '/path/to/package'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = '**Table of contents:**'
    var_7 = bool('**Table of contents:**' in var_5)
    assert var_7 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'example'
    var_1 = '/path/to/package'
    var_2 = False
    var_3 = 1
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = '<a id='
    var_7 = bool('<a id=' not in var_5)
    assert var_7 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = '/path/to/package'
    var_2 = True
    var_3 = 1
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = len(var_5)
    assert var_6 == 0



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_gen_api_basic_functionality. Retrieved 6/7 statements.
# Partially parsed test_gen_api_with_nonexistent_module. Retrieved 7/8 statements.
# Partially parsed test_gen_api_with_custom_pwd. Retrieved 6/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = {var_0: var_0}
    var_2 = 'test_docs'
    var_3 = True
    var_4 = module_0.gen_api(var_1, prefix=var_2, dry=var_3)
    var_5 = len(var_4)
    var_6 = bool(var_5 > 0)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = 'nonexistent_module'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = True
    var_5 = module_0.gen_api(var_2, prefix=var_3, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = {var_0: var_0}
    var_2 = '/custom/path'
    var_3 = 'test_docs'
    var_4 = True
    var_5 = module_0.gen_api(var_1, var_2, prefix=var_3, dry=var_4)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = {var_0: var_0}
    var_2 = 'test_docs'
    var_3 = True
    var_4 = module_0.gen_api(var_1, prefix=var_2, link=var_3, dry=var_3)
    var_5 = False
    var_6 = module_0.gen_api(var_1, prefix=var_2, link=var_5, dry=var_3)
    var_7 = bool(var_4 != var_6)
    assert var_7 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = {var_0: var_0}
    var_2 = 'test_docs'
    var_3 = 1
    var_4 = True
    var_5 = module_0.gen_api(var_1, prefix=var_2, level=var_3, dry=var_4)
    var_6 = 2
    var_7 = True
    var_8 = module_0.gen_api(var_1, prefix=var_2, level=var_6, dry=var_7)
    var_9 = bool(var_5 != var_8)
    assert var_9 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = {var_0: var_0}
    var_2 = 'test_docs'
    var_3 = True
    var_4 = module_0.gen_api(var_1, prefix=var_2, toc=var_3, dry=var_3)
    var_5 = False
    var_6 = module_0.gen_api(var_1, prefix=var_2, toc=var_5, dry=var_3)
    var_7 = bool(var_4 != var_6)
    assert var_7 is True



# Parsed testcases at query #5
#--------------------------




import apimd.loader as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = 'non_existent_directory'
    var_4 = module_0.gen_api(var_2, prefix=var_3)
    var_5 = module_1.isdir(var_3)
    var_6 = bool(var_5)
    assert var_6 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_gen_api_with_valid_root_names. Retrieved 11/12 statements.
# Partially parsed test_gen_api_with_empty_root_names. Retrieved 9/10 statements.
# Partially parsed test_gen_api_with_none_pwd. Retrieved 11/12 statements.
# Partially parsed test_gen_api_with_dry_run. Retrieved 11/12 statements.
# Partially parsed test_gen_api_with_invalid_root_names. Retrieved 11/12 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestModule'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = '/some/path'
    var_4 = 'docs'
    var_5 = True
    var_6 = 1
    var_7 = False
    var_8 = False
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)
    var_10 = len(var_9)
    assert var_10 == 1

import apimd.loader as module_0

def test_case_0():
    var_0 = {}
    var_1 = '/some/path'
    var_2 = 'docs'
    var_3 = True
    var_4 = 1
    var_5 = False
    var_6 = False
    var_7 = module_0.gen_api(var_0, var_1, prefix=var_2, link=var_3, level=var_4, toc=var_5, dry=var_6)
    var_8 = len(var_7)
    assert var_8 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestModule'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'docs'
    var_5 = True
    var_6 = 1
    var_7 = False
    var_8 = False
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)
    var_10 = len(var_9)
    assert var_10 == 1

import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestModule'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = '/some/path'
    var_4 = 'docs'
    var_5 = True
    var_6 = 1
    var_7 = False
    var_8 = True
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)
    var_10 = len(var_9)
    assert var_10 == 1

import apimd.loader as module_0

def test_case_0():
    var_0 = 'InvalidModule'
    var_1 = 'invalid_module'
    var_2 = {var_0: var_1}
    var_3 = '/some/path'
    var_4 = 'docs'
    var_5 = True
    var_6 = 1
    var_7 = False
    var_8 = False
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)
    var_10 = len(var_9)
    assert var_10 == 0



# Parsed testcases at query #7
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = 'module.submodule.class'
    var_1 = module_0.parent(var_0)
    assert var_1 == 'module'
    var_2 = 2
    var_3 = module_0.parent(var_0, level=var_2)
    assert var_3 == 'module.submodule'
    var_4 = 'single'
    var_5 = module_0.parent(var_4)
    assert var_5 == 'single'
    var_6 = 'a.b.c.d.e'
    var_7 = 3
    var_8 = module_0.parent(var_6, level=var_7)
    assert var_8 == 'a.b'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_loader. Retrieved 6/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '/tmp'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #9
#--------------------------




import apimd.loader as module_0
import importlib.util as module_1
import posixpath as module_2

def test_case_0():
    var_0 = 'os'
    var_1 = module_0._site_path(var_0)
    var_2 = 0
    var_3 = module_1.find_spec(var_0)
    var_4 = var_3.submodule_search_locations[var_2]
    var_5 = module_2.dirname(var_4)
    var_6 = bool(var_1 == var_5)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existing_module'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'builtins'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_loader_pure_py_evaluates_to_false. Retrieved 15/27 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'compiled_output'
    var_1 = 'module_name'
    var_2 = 'module_path'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = 'file_content'
    var_6 = True
    var_7 = 'module_path.py'
    var_8 = 'module_path.pyi'
    var_9 = lambda path: path != var_7 and path != var_8
    var_10 = 'root'
    var_11 = 'pwd'
    var_12 = False
    var_13 = 1
    var_14 = module_0.loader(var_10, var_11, var_12, var_13, var_12)
    assert var_14 == 'compiled_output'



# Parsed testcases at query #11
#--------------------------




def test_case_0():
    var_0 = 'test_module'
    var_1 = '/path/to/module'
    var_2 = '.pyi'
    assert var_2 == '.py'
    var_3 = False
    var_4 = '.py'
    assert var_4 is False



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_gen_api_with_valid_root_names. Retrieved 10/11 statements.
# Partially parsed test_gen_api_with_empty_root_names. Retrieved 9/10 statements.
# Partially parsed test_gen_api_with_none_pwd. Retrieved 10/11 statements.
# Partially parsed test_gen_api_with_dry_run. Retrieved 10/11 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'example'
    var_1 = 'example_package'
    var_2 = {var_0: var_1}
    var_3 = '/some/path'
    var_4 = 'docs'
    var_5 = True
    var_6 = 1
    var_7 = False
    var_8 = False
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)

import apimd.loader as module_0

def test_case_0():
    var_0 = {}
    var_1 = '/some/path'
    var_2 = 'docs'
    var_3 = True
    var_4 = 1
    var_5 = False
    var_6 = False
    var_7 = module_0.gen_api(var_0, var_1, prefix=var_2, link=var_3, level=var_4, toc=var_5, dry=var_6)
    var_8 = len(var_7)
    assert var_8 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'example'
    var_1 = 'example_package'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'docs'
    var_5 = True
    var_6 = 1
    var_7 = False
    var_8 = False
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'example'
    var_1 = 'example_package'
    var_2 = {var_0: var_1}
    var_3 = '/some/path'
    var_4 = 'docs'
    var_5 = True
    var_6 = 1
    var_7 = False
    var_8 = True
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)



# Parsed testcases at query #13
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'example_root'
    var_1 = 'example_pwd'
    var_2 = True
    var_3 = 2
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(not var_5 == '.py')
    assert var_6 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_loader. Retrieved 6/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '/tmp'
    var_2 = True
    var_3 = 1
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test__site_path_predicate_evaluates_to_false. Retrieved 3/8 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'some_path'
    var_1 = [var_0]
    var_2 = 'any_name'
    var_3 = module_0._site_path(var_2)
    var_4 = bool(var_3 != '')
    assert var_4 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test__site_path_with_existing_module. Retrieved 2/8 statements.
# Partially parsed test__site_path_with_module_without_submodule_search_locations. Retrieved 2/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'existing_module'
    var_1 = module_0._site_path(var_0)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_module'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'module_without_submodule_search_locations'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_gen_api_with_valid_root_names. Retrieved 8/9 statements.
# Partially parsed test_gen_api_with_empty_root_names. Retrieved 5/6 statements.
# Partially parsed test_gen_api_with_invalid_root_names. Retrieved 6/7 statements.
# Partially parsed test_gen_api_with_custom_prefix. Retrieved 7/8 statements.
# Partially parsed test_gen_api_with_link_false. Retrieved 8/9 statements.
# Partially parsed test_gen_api_with_level_2. Retrieved 8/9 statements.
# Partially parsed test_gen_api_with_toc_true. Retrieved 7/8 statements.
# Partially parsed test_gen_api_with_pwd. Retrieved 8/9 statements.
# Partially parsed test_gen_api_with_dry_false. Retrieved 7/8 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'module1'
    var_1 = 'module2'
    var_2 = {var_0: var_0, var_1: var_1}
    var_3 = 'test_docs'
    var_4 = True
    var_5 = module_0.gen_api(var_2, prefix=var_3, dry=var_4)
    var_6 = len(var_5)
    var_7 = len(var_2)
    var_8 = bool(var_6 == var_7)
    assert var_8 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'test_docs'
    var_2 = True
    var_3 = module_0.gen_api(var_0, prefix=var_1, dry=var_2)
    var_4 = len(var_3)
    assert var_4 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'invalid_module'
    var_1 = {var_0: var_0}
    var_2 = 'test_docs'
    var_3 = True
    var_4 = module_0.gen_api(var_1, prefix=var_2, dry=var_3)
    var_5 = len(var_4)
    assert var_5 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'module1'
    var_1 = {var_0: var_0}
    var_2 = 'custom_prefix'
    var_3 = True
    var_4 = module_0.gen_api(var_1, prefix=var_2, dry=var_3)
    var_5 = len(var_4)
    var_6 = len(var_1)
    var_7 = bool(var_5 == var_6)
    assert var_7 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'module1'
    var_1 = {var_0: var_0}
    var_2 = 'test_docs'
    var_3 = False
    var_4 = True
    var_5 = module_0.gen_api(var_1, prefix=var_2, link=var_3, dry=var_4)
    var_6 = len(var_5)
    var_7 = len(var_1)
    var_8 = bool(var_6 == var_7)
    assert var_8 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'module1'
    var_1 = {var_0: var_0}
    var_2 = 'test_docs'
    var_3 = 2
    var_4 = True
    var_5 = module_0.gen_api(var_1, prefix=var_2, level=var_3, dry=var_4)
    var_6 = len(var_5)
    var_7 = len(var_1)
    var_8 = bool(var_6 == var_7)
    assert var_8 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'module1'
    var_1 = {var_0: var_0}
    var_2 = 'test_docs'
    var_3 = True
    var_4 = module_0.gen_api(var_1, prefix=var_2, toc=var_3, dry=var_3)
    var_5 = len(var_4)
    var_6 = len(var_1)
    var_7 = bool(var_5 == var_6)
    assert var_7 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'module1'
    var_1 = {var_0: var_0}
    var_2 = '/some/path'
    var_3 = 'test_docs'
    var_4 = True
    var_5 = module_0.gen_api(var_1, var_2, prefix=var_3, dry=var_4)
    var_6 = len(var_5)
    var_7 = len(var_1)
    var_8 = bool(var_6 == var_7)
    assert var_8 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'module1'
    var_1 = {var_0: var_0}
    var_2 = 'test_docs'
    var_3 = False
    var_4 = module_0.gen_api(var_1, prefix=var_2, dry=var_3)
    var_5 = len(var_4)
    var_6 = len(var_1)
    var_7 = bool(var_5 == var_6)
    assert var_7 is True



# Parsed testcases at query #18
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'example_root'
    var_1 = 'example_pwd'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_site_path_with_valid_submodule. Retrieved 3/9 statements.
# Partially parsed test_site_path_with_none_spec. Retrieved 4/7 statements.
# Partially parsed test_site_path_with_none_submodule_locations. Retrieved 3/9 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = '/some/path'
    var_1 = [var_0]
    var_2 = 'valid_module'
    var_3 = module_0._site_path(var_2)
    assert var_3 == '/some/path'

import apimd.loader as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda _: var_0
    var_2 = 'nonexistent_module'
    var_3 = module_0._site_path(var_2)
    assert var_3 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = None
    var_1 = 'module_without_submodules'
    var_2 = module_0._site_path(var_1)
    assert var_2 == ''



# Parsed testcases at query #20
#--------------------------

# Partially parsed test__read_file_exists. Retrieved 3/7 statements.
# Partially parsed test__read_file_not_exists. Retrieved 2/6 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, world!'
    var_2 = module_0._read(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existent_file.txt'
    var_1 = module_0._read(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_site_path_existing_package. Retrieved 2/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = module_0._site_path(var_0)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existing_package'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'builtins'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_loader_pure_py_condition. Retrieved 6/36 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_root'
    var_1 = 'test_pwd'
    var_2 = False
    var_3 = 1
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    assert var_5 == 'compiled_result'



# Parsed testcases at query #23
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = 'module.submodule.class'
    var_1 = module_0.parent(var_0)
    assert var_1 == 'module'
    var_2 = 2
    var_3 = module_0.parent(var_0, level=var_2)
    assert var_3 == 'module.submodule'
    var_4 = 'single'
    var_5 = module_0.parent(var_4)
    assert var_5 == 'single'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_read_file_exists. Retrieved 3/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'test content'
    var_2 = module_0._read(var_0)
    assert var_2 == 'test content'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_file.txt'
    var_1 = module_0._read(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_write_to_file. Retrieved 3/5 statements.
# Partially parsed test_write_to_file_overwrite. Retrieved 5/7 statements.
# Partially parsed test_write_to_file_empty_string. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = module_0._write(var_0, var_1)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Initial content'
    var_2 = 'New content'
    var_3 = module_0._write(var_0, var_1)
    var_4 = module_0._write(var_0, var_2)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = ''
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #26
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_file.txt'
    var_1 = module_0._read(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/7 statements.
# Partially parsed test_write_overwrites_existing_file. Retrieved 5/9 statements.
# Partially parsed test_write_handles_unicode_characters. Retrieved 3/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, world!'
    var_2 = module_0._write(var_0, var_1)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Initial content'
    var_2 = 'New content'
    var_3 = module_0._write(var_0, var_1)
    var_4 = module_0._write(var_0, var_2)
    var_5 = bool(var_3 == var_2)
    assert var_5 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'こんにちは世界'
    var_2 = module_0._write(var_0, var_1)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_load_module_with_valid_spec_and_loader. Retrieved 4/12 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_path'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is True



# Parsed testcases at query #29
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existent_directory/test.txt'
    var_1 = 'sample text'
    var_2 = None
    var_3 = module_0._write(var_0, var_1)
    var_4 = False
    assert var_4 is False



# Parsed testcases at query #30
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = module_0._read(var_0)
    assert var_1 == 'Hello, World!'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existing_file.txt'
    var_1 = module_0._read(var_0)
    var_2 = bool(True)
    assert var_2 is True
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_read_script_from_file. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'test content'
    var_2 = module_0._read(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_write_file_opens_with_correct_encoding. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'test content'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_load_module_evaluates_to_true. Retrieved 5/6 statements.


import apimd.parser as module_0
import _frozen_importlib_external as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_path'
    var_2 = module_0.Parser()
    var_3 = module_1.spec_from_file_location(var_0, var_1)
    var_4 = var_3.loader



# Parsed testcases at query #34
#--------------------------




import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = '/path/to/module.py'
    var_2 = module_0.Parser()
    var_3 = None
    var_4 = module_1._load_module(var_0, var_1, var_2)
    assert var_4 is False



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_load_module_with_invalid_spec. Retrieved 4/8 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_path'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    var_4 = bool(not var_3)
    assert var_4 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_write_creates_file. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_write_file_content. Retrieved 3/5 statements.
# Partially parsed test_write_file_overwrites_existing_content. Retrieved 5/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_output.txt'
    var_1 = 'Hello, World!'
    var_2 = module_0._write(var_0, var_1)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_output.txt'
    var_1 = 'Initial content'
    var_2 = 'Updated content'
    var_3 = module_0._write(var_0, var_1)
    var_4 = module_0._write(var_0, var_2)



# Parsed testcases at query #38
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existent_file.txt'
    var_1 = False
    var_2 = module_0._read(var_0)
    var_3 = True
    var_4 = bool(var_3)
    assert var_4 is True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_write_file. Retrieved 3/5 statements.
# Partially parsed test_write_empty_string. Retrieved 3/5 statements.
# Partially parsed test_write_special_characters. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = module_0._write(var_0, var_1)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_empty_file.txt'
    var_1 = ''
    var_2 = module_0._write(var_0, var_1)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_special_characters.txt'
    var_1 = '!@#$%^&*()_+{}:"<>?[];\',./`~'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #40
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_file.txt'
    var_1 = module_0._read(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_read_returns_file_content. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'test content'
    var_2 = module_0._read(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_predicate_at_line_3_evaluates_to_false. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = '/nonexistent_directory/test.txt'
    var_1 = 'Sample text'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_load_module_with_valid_spec_and_loader. Retrieved 4/15 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_path'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True



# Parsed testcases at query #44
#--------------------------




import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module.py'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'non_existent_module'
    var_1 = 'non_existent_module.py'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    var_4 = bool(not var_3)
    assert var_4 is True

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module.py'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    var_4 = bool(not var_3)
    assert var_4 is True



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_load_module_predicate_evaluates_to_true. Retrieved 5/6 statements.


import apimd.parser as module_0
import _frozen_importlib_external as module_1
import apimd.loader as module_2

def test_case_0():
    var_0 = 'example_module'
    var_1 = '/path/to/example_module.py'
    var_2 = module_0.Parser()
    var_3 = module_1.spec_from_file_location(var_0, var_1)
    var_4 = []
    var_5 = module_2._load_module(var_0, var_1, var_2)
    assert var_5 is True



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_site_path_predicate_evaluates_to_false. Retrieved 2/9 statements.


def test_case_0():
    var_0 = '/some/path'
    var_1 = None



# Parsed testcases at query #47
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = module_0._read(var_0)
    assert var_1 == 'Expected file content'



# Parsed testcases at query #48
#--------------------------




import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = '/path/to/test_module.py'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is True

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'non_existent_module'
    var_1 = '/path/to/non_existent_module.py'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is False



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_write_file_exists. Retrieved 4/6 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = '/non/existent/directory/file.txt'
    var_1 = 'Sample text'
    var_2 = module_0._write(var_0, var_1)
    var_3 = 'The file should exist after writing'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_loader_pure_py_condition. Retrieved 8/34 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_root'
    var_1 = 'test_pwd'
    var_2 = False
    var_3 = 1
    var_4 = False
    var_5 = 'test_module'
    var_6 = 'test_path'
    var_7 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    assert var_7 == 'test_output'



# Parsed testcases at query #51
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = '/path/to/module'
    var_4 = 'docs'
    var_5 = True
    var_6 = 1
    var_7 = False
    var_8 = False
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)
    var_10 = len(var_9)
    assert var_10 == 0



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_pure_py_should_not_skip_extension_module_loading. Retrieved 5/22 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = 'pwd'
    var_2 = False
    var_3 = 1
    var_4 = module_0.loader(var_0, var_1, var_2, var_3, var_2)



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_load_module_with_valid_spec_and_loader. Retrieved 5/13 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = '/path/to/test_module.py'
    var_2 = 'test'
    var_3 = module_0.Parser()
    var_4 = module_1._load_module(var_0, var_1, var_3)
    assert var_4 is True



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_write_file_successfully. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'test content'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_read_file_exists. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'test content'



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 1/2 statements.


def test_case_0():
    var_0 = ''



# Parsed testcases at query #57
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'example_root'
    var_1 = 'example_pwd'
    var_2 = False
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_loader. Retrieved 9/12 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '/tmp'
    var_2 = True
    var_3 = module_0.loader(var_0, var_1, var_2, var_2, var_2)
    var_4 = 'Module `test_pkg`'
    var_5 = bool('Module `test_pkg`' in var_3)
    assert var_5 is True
    var_6 = 'Table of contents'
    var_7 = bool('Table of contents' in var_3)
    assert var_7 is True
    var_8 = 'nonexistent_pkg'
    var_9 = False
    var_10 = 2
    var_11 = module_0.loader(var_8, var_1, var_9, var_10, var_9)
    var_12 = 'Module `nonexistent_pkg`'
    var_13 = bool('Module `nonexistent_pkg`' not in var_11)
    assert var_13 is True
    var_14 = module_0.loader(var_0, var_1, var_2, var_2, var_9)
    var_15 = 'Module `test_pkg`'
    var_16 = bool('Module `test_pkg`' in var_14)
    assert var_16 is True
    var_17 = 'Table of contents'
    var_18 = bool('Table of contents' not in var_14)
    assert var_18 is True



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_predicate_evaluates_to_false_when_spec_is_none. Retrieved 4/6 statements.
# Partially parsed test_predicate_evaluates_to_false_when_loader_is_not_instance_of_Loader. Retrieved 1/9 statements.


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = var_0 is not var_1
    var_3 = var_0.loader

def test_case_0():
    var_0 = None



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_read_file_successfully. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'test content'



# Parsed testcases at query #61
#--------------------------

# Failed to parse test_load_module_with_valid_spec_and_loader.




# Parsed testcases at query #62
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = '/nonexistent/file/path'
    var_1 = module_0._read(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_predicate_at_line_3_evaluates_to_false. Retrieved 2/3 statements.


def test_case_0():
    var_0 = 'non_existent_directory/test.txt'
    var_1 = 'test content'



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_write_file_with_utf8_encoding. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, world!'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_predicate_at_line_25_evaluates_to_true. Retrieved 10/12 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'example'
    var_1 = 'non_empty_module'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'docs'
    var_5 = True
    var_6 = 1
    var_7 = False
    var_8 = False
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)



