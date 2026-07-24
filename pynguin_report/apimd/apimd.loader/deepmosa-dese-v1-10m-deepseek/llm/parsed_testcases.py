####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_gen_api_with_valid_root_names. Retrieved 5/8 statements.
# Partially parsed test_gen_api_with_empty_root_names. Retrieved 4/5 statements.
# Partially parsed test_gen_api_with_none_pwd. Retrieved 6/7 statements.
# Partially parsed test_gen_api_with_custom_prefix. Retrieved 6/7 statements.
# Partially parsed test_gen_api_with_link_false. Retrieved 6/7 statements.
# Partially parsed test_gen_api_with_level_2. Retrieved 6/7 statements.
# Partially parsed test_gen_api_with_toc_true. Retrieved 5/6 statements.
# Partially parsed test_gen_api_with_multiple_root_names. Retrieved 8/9 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)

import apimd.loader as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.gen_api(var_0, dry=var_1)
    var_3 = len(var_2)
    assert var_3 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = True
    var_5 = module_0.gen_api(var_2, var_3, dry=var_4)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = 'custom_docs'
    var_4 = True
    var_5 = module_0.gen_api(var_2, prefix=var_3, dry=var_4)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = True
    var_5 = module_0.gen_api(var_2, link=var_3, dry=var_4)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = True
    var_5 = module_0.gen_api(var_2, level=var_3, dry=var_4)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, toc=var_3, dry=var_3)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test1'
    var_1 = 'test2'
    var_2 = 'test_module1'
    var_3 = 'test_module2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True
    var_6 = module_0.gen_api(var_4, dry=var_5)
    var_7 = len(var_6)
    assert var_7 == 2



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_loader. Retrieved 6/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = '/fake/path'
    var_1 = 'fake.module'
    var_2 = True
    var_3 = 1
    var_4 = False
    var_5 = module_0.loader(var_1, var_0, var_2, var_3, var_4)



# Parsed testcases at query #3
#--------------------------




import apimd.loader as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'non_existent_directory'
    var_1 = 'Test'
    var_2 = 'test_module'
    var_3 = {var_1: var_2}
    var_4 = module_0.gen_api(var_3, prefix=var_0)
    var_5 = module_1.isdir(var_0)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test__load_module_success. Retrieved 5/14 statements.
# Partially parsed test__load_module_no_loader. Retrieved 6/11 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = '/path/to/test_module.py'
    var_2 = module_0.Parser()
    var_3 = module_0.parent(var_0)
    var_4 = module_1._load_module(var_0, var_1, var_2)
    assert var_4 is True

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'nonexistent_module'
    var_1 = '/path/to/nonexistent_module.py'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module_no_loader'
    var_1 = '/path/to/test_module_no_loader.py'
    var_2 = module_0.Parser()
    var_3 = None
    var_4 = module_0.parent(var_0)
    var_5 = module_1._load_module(var_0, var_1, var_2)
    assert var_5 is False



# Parsed testcases at query #5
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'example_root'
    var_1 = 'example_pwd'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #6
#--------------------------




import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'invalid_path'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is False



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



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_loader. Retrieved 6/7 statements.
# Partially parsed test_loader_with_toc. Retrieved 6/7 statements.
# Partially parsed test_loader_with_invalid_path. Retrieved 6/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '/path/to/test_pkg'
    var_2 = True
    var_3 = 1
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '/path/to/test_pkg'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'invalid_pkg'
    var_1 = '/path/to/invalid_pkg'
    var_2 = True
    var_3 = 1
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    assert var_5 == '\n'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_pure_py_is_false_when_ext_is_not_py. Retrieved 5/19 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = 'pwd'
    var_2 = False
    var_3 = 1
    var_4 = module_0.loader(var_0, var_1, var_2, var_3, var_2)



# Parsed testcases at query #10
#--------------------------




def test_case_0():
    var_0 = 'test_module'
    var_1 = '/path/to/module'
    var_2 = '/path'
    var_3 = '/path/to'
    var_4 = False
    var_5 = 1
    var_6 = False
    var_7 = False



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_loader_with_valid_package. Retrieved 7/19 statements.
# Partially parsed test_loader_with_non_existent_package. Retrieved 3/6 statements.
# Partially parsed test_loader_with_empty_package. Retrieved 5/14 statements.
# Partially parsed test_loader_with_toc_enabled. Retrieved 4/13 statements.
# Partially parsed test_loader_with_different_level. Retrieved 6/15 statements.


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Test package docstring."""\n'
    var_3 = 'module.py'
    var_4 = '"""Test module docstring."""\n'
    var_5 = True
    var_6 = False

def test_case_0():
    var_0 = 'non_existent'
    var_1 = True
    var_2 = False

def test_case_0():
    var_0 = 'empty_pkg'
    var_1 = '__init__.py'
    var_2 = ''
    var_3 = True
    var_4 = False

def test_case_0():
    var_0 = 'toc_pkg'
    var_1 = '__init__.py'
    var_2 = '"""TOC package docstring."""\n'
    var_3 = True

def test_case_0():
    var_0 = 'level_pkg'
    var_1 = '__init__.py'
    var_2 = '"""Level package docstring."""\n'
    var_3 = True
    var_4 = 2
    var_5 = False



# Parsed testcases at query #12
#--------------------------




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
    var_0 = 'math'
    var_1 = module_0._site_path(var_0)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_loader_pure_py_condition. Retrieved 8/33 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'isfile'
    var_1 = None
    var_2 = globals()
    var_3 = 'walk_packages'
    var_4 = ''
    var_5 = False
    var_6 = 1
    var_7 = module_0.loader(var_4, var_4, var_5, var_6, var_5)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_site_path_with_existing_module. Retrieved 3/4 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = module_0._site_path(var_0)
    var_2 = len(var_1)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_module_123'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'builtins'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_gen_api_with_valid_input. Retrieved 7/8 statements.
# Partially parsed test_gen_api_with_nonexistent_module. Retrieved 7/8 statements.
# Partially parsed test_gen_api_with_custom_pwd. Retrieved 7/8 statements.
# Partially parsed test_gen_api_with_link_false. Retrieved 8/9 statements.
# Partially parsed test_gen_api_with_level_2. Retrieved 8/9 statements.
# Partially parsed test_gen_api_with_toc_true. Retrieved 7/8 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = 'docs'
    var_4 = True
    var_5 = module_0.gen_api(var_2, prefix=var_3, dry=var_4)
    var_6 = len(var_5)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = 'nonexistent_module'
    var_2 = {var_0: var_1}
    var_3 = 'docs'
    var_4 = True
    var_5 = module_0.gen_api(var_2, prefix=var_3, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = '/custom/path'
    var_4 = 'docs'
    var_5 = True
    var_6 = module_0.gen_api(var_2, var_3, prefix=var_4, dry=var_5)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = 'docs'
    var_4 = False
    var_5 = True
    var_6 = module_0.gen_api(var_2, prefix=var_3, link=var_4, dry=var_5)
    var_7 = len(var_6)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = 'docs'
    var_4 = 2
    var_5 = True
    var_6 = module_0.gen_api(var_2, prefix=var_3, level=var_4, dry=var_5)
    var_7 = len(var_6)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = 'docs'
    var_4 = True
    var_5 = module_0.gen_api(var_2, prefix=var_3, toc=var_4, dry=var_4)
    var_6 = len(var_5)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/7 statements.
# Partially parsed test_write_overwrites_existing_file. Retrieved 5/9 statements.
# Partially parsed test_write_handles_unicode. Retrieved 3/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, world!'
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
    var_1 = 'こんにちは世界'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #17
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = '/non/existent/directory/file.txt'
    var_1 = 'test content'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #18
#--------------------------




def test_case_0():
    var_0 = False



# Parsed testcases at query #19
#--------------------------

# Partially parsed test__site_path_with_existing_package. Retrieved 3/4 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = module_0._site_path(var_0)
    var_2 = len(var_1)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existing_package_123'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'builtins'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_site_path_with_valid_module. Retrieved 3/9 statements.
# Partially parsed test_site_path_with_none_spec. Retrieved 4/7 statements.
# Partially parsed test_site_path_with_none_locations. Retrieved 3/9 statements.


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
    var_1 = lambda x: var_0
    var_2 = 'nonexistent_module'
    var_3 = module_0._site_path(var_2)
    assert var_3 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = None
    var_1 = 'module_with_none_locations'
    var_2 = module_0._site_path(var_1)
    assert var_2 == ''



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_load_module_predicate_evaluates_to_true. Retrieved 7/11 statements.


import _frozen_importlib_external as module_0

def test_case_0():
    var_0 = 'example_module'
    var_1 = '/path/to/module.py'
    var_2 = True
    var_3 = module_0.spec_from_file_location(var_0, var_1)
    var_4 = None
    var_5 = var_3 is not var_4
    var_6 = var_3.loader



# Parsed testcases at query #22
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'existing_file.txt'
    var_1 = module_0._read(var_0)
    assert var_1 == 'This is the content of the file.'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existing_file.txt'
    var_1 = module_0._read(var_0)



# Parsed testcases at query #23
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



# Parsed testcases at query #24
#--------------------------




import apimd.loader as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'Example'
    var_1 = 'example_module'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = module_0.gen_api(var_2, prefix=var_3)
    var_5 = module_1.isdir(var_3)



# Parsed testcases at query #25
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = module_0._site_path(var_0)
    var_2 = len(var_1)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existing_module_123'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'builtins'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''



# Parsed testcases at query #26
#--------------------------




import apimd.loader as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = module_0.gen_api(var_2, prefix=var_3)
    var_5 = module_1.isdir(var_3)



# Parsed testcases at query #27
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = 'test_module'
    var_4 = ''
    var_5 = True
    var_6 = module_0.gen_api(var_2, dry=var_5)
    var_7 = len(var_6)
    assert var_7 == 0



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_gen_api_with_valid_root_names. Retrieved 4/5 statements.
# Partially parsed test_gen_api_with_empty_root_names. Retrieved 3/4 statements.
# Partially parsed test_gen_api_with_none_pwd. Retrieved 5/6 statements.
# Partially parsed test_gen_api_with_custom_prefix. Retrieved 5/6 statements.
# Partially parsed test_gen_api_with_link_disabled. Retrieved 5/6 statements.
# Partially parsed test_gen_api_with_custom_level. Retrieved 5/6 statements.
# Partially parsed test_gen_api_with_toc_enabled. Retrieved 5/6 statements.
# Partially parsed test_gen_api_with_dry_run. Retrieved 5/6 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = module_0.gen_api(var_2)

import apimd.loader as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.gen_api(var_0)
    var_2 = len(var_1)
    assert var_2 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = module_0.gen_api(var_2, var_3)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = 'custom_prefix'
    var_4 = module_0.gen_api(var_2, prefix=var_3)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = module_0.gen_api(var_2, link=var_3)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = module_0.gen_api(var_2, level=var_3)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, toc=var_3)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)



# Parsed testcases at query #29
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = 'non_existent_path'
    var_4 = module_0.gen_api(var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 0



# Parsed testcases at query #30
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = '/nonexistent/path/to/file.txt'
    var_1 = 'Sample text'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #31
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = 'test_path'
    var_4 = True
    var_5 = module_0.gen_api(var_2, var_3, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 0



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_predicate_at_line_3_evaluates_to_false. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'temp_test_file.txt'
    var_1 = 'test content'
    var_2 = 'non_existent_file.txt'



# Parsed testcases at query #33
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = 'some_path'
    var_4 = True
    var_5 = False
    var_6 = module_0.gen_api(var_2, var_3, link=var_4, level=var_4, toc=var_5, dry=var_4)
    var_7 = len(var_6)
    assert var_7 == 0



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_gen_api_with_valid_input. Retrieved 11/13 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'module1'
    var_1 = 'module2'
    var_2 = {var_0: var_0, var_1: var_1}
    var_3 = 'tests/data'
    var_4 = 'docs'
    var_5 = True
    var_6 = 1
    var_7 = False
    var_8 = True
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)
    var_10 = len(var_9)
    assert var_10 == 2

import apimd.loader as module_0

def test_case_0():
    var_0 = 'module1'
    var_1 = {var_0: var_0}
    var_2 = 'invalid/path'
    var_3 = True
    var_4 = module_0.gen_api(var_1, var_2, dry=var_3)
    var_5 = len(var_4)
    assert var_5 == 1

import apimd.loader as module_0

def test_case_0():
    var_0 = 'module1'
    var_1 = {var_0: var_0}
    var_2 = None
    var_3 = True
    var_4 = module_0.gen_api(var_1, var_2, dry=var_3)
    var_5 = len(var_4)
    assert var_5 == 1

import apimd.loader as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.gen_api(var_0, dry=var_1)
    var_3 = len(var_2)
    assert var_3 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_module'
    var_1 = {var_0: var_0}
    var_2 = True
    var_3 = module_0.gen_api(var_1, dry=var_2)
    var_4 = len(var_3)
    assert var_4 == 1

import apimd.loader as module_0

def test_case_0():
    var_0 = 'module1'
    var_1 = {var_0: var_0}
    var_2 = 'custom_prefix'
    var_3 = True
    var_4 = module_0.gen_api(var_1, prefix=var_2, dry=var_3)
    var_5 = len(var_4)
    assert var_5 == 1

import apimd.loader as module_0

def test_case_0():
    var_0 = 'module1'
    var_1 = {var_0: var_0}
    var_2 = False
    var_3 = True
    var_4 = module_0.gen_api(var_1, link=var_2, dry=var_3)
    var_5 = len(var_4)
    assert var_5 == 1

import apimd.loader as module_0

def test_case_0():
    var_0 = 'module1'
    var_1 = {var_0: var_0}
    var_2 = 2
    var_3 = True
    var_4 = module_0.gen_api(var_1, level=var_2, dry=var_3)
    var_5 = len(var_4)
    assert var_5 == 1

import apimd.loader as module_0

def test_case_0():
    var_0 = 'module1'
    var_1 = {var_0: var_0}
    var_2 = True
    var_3 = True
    var_4 = module_0.gen_api(var_1, toc=var_2, dry=var_3)
    var_5 = len(var_4)
    assert var_5 == 1



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_read_file_successfully. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'test content'



# Parsed testcases at query #36
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existent_directory/test.txt'
    var_1 = 'sample text'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_load_module_with_non_loader. Retrieved 4/8 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = None
    var_1 = 'non_existent_path.py'
    var_2 = 'non_existent_module'
    var_3 = module_0.Parser()
    var_4 = module_1._load_module(var_2, var_1, var_3)
    assert var_4 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'non_existent_path.py'
    var_1 = 'non_existent_module'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_1, var_0, var_2)
    assert var_3 is False



# Parsed testcases at query #38
#--------------------------

# Partially parsed test__load_module_predicate_evaluates_to_false. Retrieved 4/6 statements.


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = var_0 is not var_1
    var_3 = var_0.loader



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_load_module_with_invalid_spec. Retrieved 4/10 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module'
    var_2 = '/path/to/module'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_write_text_to_file. Retrieved 2/6 statements.
# Partially parsed test_write_empty_text_to_file. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, world!'

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = ''



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_read_file_content. Retrieved 3/7 statements.
# Partially parsed test_read_empty_file. Retrieved 2/6 statements.
# Partially parsed test_read_nonexistent_file. Retrieved 2/6 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'test content'
    var_2 = module_0._read(var_0)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'empty_file.txt'
    var_1 = module_0._read(var_0)
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_file.txt'
    var_1 = module_0._read(var_0)



# Parsed testcases at query #42
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_file.txt'
    var_1 = module_0._read(var_0)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_read_file_successfully. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'This is a test file.'
    var_2 = module_0._read(var_0)



# Parsed testcases at query #44
#--------------------------

# Partially parsed test__read_file_exists. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, world!'
    var_2 = module_0._read(var_0)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_file.txt'
    var_1 = module_0._read(var_0)



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_write_file_successfully. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'test content'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_write_to_file. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_read_file_exists. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'test content'



# Parsed testcases at query #48
#--------------------------




import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module.py'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is True

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'non_existent_module'
    var_1 = 'non_existent_module.py'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module.py'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is False



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_write_file_successfully. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_load_module_with_valid_spec_and_loader. Retrieved 4/12 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_path.py'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is True



# Parsed testcases at query #51
#--------------------------




def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = False



# Parsed testcases at query #52
#--------------------------




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
    var_8 = True
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_gen_api_with_valid_input. Retrieved 11/12 statements.
# Partially parsed test_gen_api_with_none_pwd. Retrieved 10/11 statements.
# Partially parsed test_gen_api_with_dry_run. Retrieved 10/11 statements.
# Partially parsed test_gen_api_with_empty_root_names. Retrieved 9/10 statements.
# Partially parsed test_gen_api_with_invalid_module. Retrieved 11/12 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestModule'
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

import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestModule'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = 'docs'
    var_4 = True
    var_5 = 1
    var_6 = False
    var_7 = False
    var_8 = None
    var_9 = module_0.gen_api(var_2, var_8, prefix=var_3, link=var_4, level=var_5, toc=var_6, dry=var_7)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestModule'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = '/path/to/module'
    var_4 = 'docs'
    var_5 = True
    var_6 = 1
    var_7 = False
    var_8 = True
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)

import apimd.loader as module_0

def test_case_0():
    var_0 = {}
    var_1 = '/path/to/module'
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
    var_0 = 'InvalidModule'
    var_1 = 'invalid_module'
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



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_gen_api_dry_run. Retrieved 9/10 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestModule'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = '# TestModule API'



# Parsed testcases at query #55
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'example'
    var_1 = 'example_module'
    var_2 = {var_0: var_1}
    var_3 = '/some/path'
    var_4 = 'docs'
    var_5 = True
    var_6 = 1
    var_7 = False
    var_8 = False
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)



# Parsed testcases at query #56
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = '/some/path'
    var_4 = 'docs'
    var_5 = True
    var_6 = 1
    var_7 = False
    var_8 = False
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)



# Parsed testcases at query #57
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = len(var_4)
    assert var_5 == 1



# Parsed testcases at query #58
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = '.'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = module_0.gen_api(var_2, var_3, prefix=var_4, dry=var_5)
    var_7 = len(var_6)

import apimd.loader as module_0

def test_case_0():
    var_0 = {}
    var_1 = '.'
    var_2 = True
    var_3 = module_0.gen_api(var_0, var_1, dry=var_2)
    var_4 = len(var_3)
    assert var_4 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = True
    var_5 = module_0.gen_api(var_2, var_3, dry=var_4)
    var_6 = len(var_5)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Invalid'
    var_1 = 'nonexistent_module'
    var_2 = {var_0: var_1}
    var_3 = '.'
    var_4 = True
    var_5 = module_0.gen_api(var_2, var_3, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test1'
    var_1 = 'Test2'
    var_2 = 'test_module1'
    var_3 = 'test_module2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = '.'
    var_6 = True
    var_7 = module_0.gen_api(var_4, var_5, dry=var_6)
    var_8 = len(var_7)
    assert var_8 == 2

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = '.'
    var_4 = 'custom_prefix'
    var_5 = True
    var_6 = module_0.gen_api(var_2, var_3, prefix=var_4, dry=var_5)
    var_7 = len(var_6)



# Parsed testcases at query #59
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = '/some/path'
    var_4 = 'docs'
    var_5 = True
    var_6 = module_0.gen_api(var_2, var_3, prefix=var_4, dry=var_5)
    var_7 = len(var_6)
    assert var_7 == 1



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_loader_pure_py_condition. Retrieved 8/37 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_root'
    var_1 = 'test_pwd'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = 'test_module'
    var_6 = 'test_path'
    var_7 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    assert var_7 == 'compiled_result'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_loader_with_valid_package. Retrieved 6/7 statements.
# Partially parsed test_loader_with_invalid_package. Retrieved 6/8 statements.
# Partially parsed test_loader_with_mixed_package. Retrieved 6/7 statements.
# Partially parsed test_loader_with_pure_python_package. Retrieved 6/7 statements.
# Partially parsed test_loader_with_extension_module. Retrieved 6/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = '/path/to/package'
    var_1 = 'package'
    var_2 = True
    var_3 = 2
    var_4 = True
    var_5 = module_0.loader(var_1, var_0, var_2, var_3, var_4)

import apimd.loader as module_0

def test_case_0():
    var_0 = '/path/to/invalid/package'
    var_1 = 'invalid_package'
    var_2 = False
    var_3 = 1
    var_4 = False
    var_5 = module_0.loader(var_1, var_0, var_2, var_3, var_4)

import apimd.loader as module_0

def test_case_0():
    var_0 = '/path/to/mixed/package'
    var_1 = 'mixed_package'
    var_2 = True
    var_3 = 2
    var_4 = True
    var_5 = module_0.loader(var_1, var_0, var_2, var_3, var_4)

import apimd.loader as module_0

def test_case_0():
    var_0 = '/path/to/pure_python/package'
    var_1 = 'pure_python_package'
    var_2 = False
    var_3 = 1
    var_4 = False
    var_5 = module_0.loader(var_1, var_0, var_2, var_3, var_4)

import apimd.loader as module_0

def test_case_0():
    var_0 = '/path/to/extension/module'
    var_1 = 'extension_module'
    var_2 = True
    var_3 = 2
    var_4 = True
    var_5 = module_0.loader(var_1, var_0, var_2, var_3, var_4)



# Parsed testcases at query #2
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = 'valid_module'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = '/path/to/site-packages'
    var_5 = module_0.gen_api(var_2, var_4, prefix=var_3)
    var_6 = len(var_5)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = 'invalid_module'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = '/path/to/site-packages'
    var_5 = module_0.gen_api(var_2, var_4, prefix=var_3)
    var_6 = len(var_5)
    assert var_6 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = 'valid_module'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = '/path/to/site-packages'
    var_5 = True
    var_6 = module_0.gen_api(var_2, var_4, prefix=var_3, dry=var_5)
    var_7 = len(var_6)

import apimd.loader as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'test_docs'
    var_2 = '/path/to/site-packages'
    var_3 = module_0.gen_api(var_0, var_2, prefix=var_1)
    var_4 = len(var_3)
    assert var_4 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = 'valid_module'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = None
    var_5 = module_0.gen_api(var_2, var_4, prefix=var_3)
    var_6 = len(var_5)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = 'valid_module'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = '/invalid/path'
    var_5 = module_0.gen_api(var_2, var_4, prefix=var_3)
    var_6 = len(var_5)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = 'valid_module'
    var_2 = {var_0: var_1}
    var_3 = '/invalid/prefix'
    var_4 = '/path/to/site-packages'
    var_5 = module_0.gen_api(var_2, var_4, prefix=var_3)
    var_6 = len(var_5)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = 'valid_module'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = '/path/to/site-packages'
    var_5 = False
    var_6 = module_0.gen_api(var_2, var_4, prefix=var_3, link=var_5)
    var_7 = len(var_6)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = 'valid_module'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = '/path/to/site-packages'
    var_5 = 2
    var_6 = module_0.gen_api(var_2, var_4, prefix=var_3, level=var_5)
    var_7 = len(var_6)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = 'valid_module'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = '/path/to/site-packages'
    var_5 = False
    var_6 = module_0.gen_api(var_2, var_4, prefix=var_3, toc=var_5)
    var_7 = len(var_6)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_loader_with_valid_package. Retrieved 7/8 statements.
# Partially parsed test_loader_with_invalid_package. Retrieved 7/8 statements.
# Partially parsed test_loader_with_toc_enabled. Retrieved 6/7 statements.
# Partially parsed test_loader_with_toc_disabled. Retrieved 6/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'example_pkg'
    var_1 = '/path/to/package'
    var_2 = True
    var_3 = 1
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = len(var_5)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'invalid_pkg'
    var_1 = '/invalid/path'
    var_2 = True
    var_3 = 1
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = len(var_5)
    assert var_6 == 0

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
    var_0 = 'example_pkg'
    var_1 = '/path/to/package'
    var_2 = True
    var_3 = 1
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_loader_skips_pure_python_modules. Retrieved 28/45 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = ''
    var_1 = ''
    var_2 = False
    var_3 = 1
    var_4 = False
    var_5 = 'example'
    var_6 = '/path/to/module'
    var_7 = 'DummyLogger'
    var_8 = ()
    var_9 = 'debug'
    var_10 = 'warning'
    var_11 = None
    var_12 = lambda *args, **kwargs: var_11
    var_13 = lambda *args, **kwargs: var_11
    var_14 = {var_9: var_12, var_10: var_13}
    var_15 = type(var_7, var_8, var_14)
    var_16 = '.so'
    var_17 = '.pyd'
    var_18 = [var_16, var_17]
    var_19 = globals()
    var_20 = 'walk_packages'
    var_21 = 'isfile'
    var_22 = '_read'
    var_23 = '_load_module'
    var_24 = 'logger'
    var_25 = 'Parser'
    var_26 = 'EXTENSION_SUFFIXES'
    var_27 = module_0.loader(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #5
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'example'
    var_1 = 'example'
    var_2 = False
    var_3 = 1
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_write_to_file. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_read_valid_file. Retrieved 3/5 statements.
# Partially parsed test_read_empty_file. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Hello, World!'
    var_1 = 'test_file.txt'
    var_2 = module_0._read(var_1)

import apimd.loader as module_0

def test_case_0():
    var_0 = ''
    var_1 = 'test_file.txt'
    var_2 = module_0._read(var_1)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existent_file.txt'
    var_1 = module_0._read(var_0)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_loader_with_python_file. Retrieved 5/12 statements.
# Partially parsed test_loader_with_python_stub_file. Retrieved 5/12 statements.
# Partially parsed test_loader_with_non_python_file. Retrieved 5/12 statements.
# Partially parsed test_loader_with_empty_directory. Retrieved 3/6 statements.
# Partially parsed test_loader_with_toc_enabled. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'test_module.py'
    var_1 = 'def test_func():\n    pass\n'
    var_2 = 'test_module'
    var_3 = True
    var_4 = False

def test_case_0():
    var_0 = 'test_module.pyi'
    var_1 = 'def test_func() -> None: ...\n'
    var_2 = 'test_module'
    var_3 = True
    var_4 = False

def test_case_0():
    var_0 = 'test_module.txt'
    var_1 = 'This is not a Python file'
    var_2 = 'test_module'
    var_3 = True
    var_4 = False

def test_case_0():
    var_0 = 'test_module'
    var_1 = True
    var_2 = False

def test_case_0():
    var_0 = 'test_module.py'
    var_1 = 'def test_func():\n    pass\n'
    var_2 = 'test_module'
    var_3 = True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_loader_skips_pure_python_modules. Retrieved 3/14 statements.


def test_case_0():
    var_0 = '/'
    var_1 = False
    var_2 = 1



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_loader_should_not_skip_extension_module_when_pure_py_is_false. Retrieved 20/37 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = '.pyd'
    var_1 = [var_0]
    var_2 = 'MockLogger'
    var_3 = ()
    var_4 = 'debug'
    var_5 = 'warning'
    var_6 = None
    var_7 = lambda *args: var_6
    var_8 = lambda *args: var_6
    var_9 = {var_4: var_7, var_5: var_8}
    var_10 = type(var_2, var_3, var_9)
    var_11 = 'MockParser'
    var_12 = ()
    var_13 = 'new'
    var_14 = 'root'
    var_15 = 'pwd'
    var_16 = False
    var_17 = 1
    var_18 = True
    var_19 = module_0.loader(var_14, var_15, var_16, var_17, var_18)
    assert var_19 == 'compiled_output'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test__load_module_success. Retrieved 4/14 statements.


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
    var_0 = module_0.Parser()
    var_1 = 'nonexistent.module'
    var_2 = '/path/to/nonexistent.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'invalid'
    var_2 = '/path/to/invalid.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_load_module_predicate_evaluates_to_true. Retrieved 7/8 statements.


import apimd.parser as module_0
import _frozen_importlib_external as module_1
import _frozen_importlib as module_2
import apimd.loader as module_3

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_path'
    var_2 = module_0.Parser()
    var_3 = module_1.spec_from_file_location(var_0, var_1)
    var_4 = module_2.module_from_spec(var_3)
    var_5 = var_2.load_docstring(var_0, var_4)
    var_6 = module_3._load_module(var_0, var_1, var_2)
    assert var_6 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_load_module_predicate_evaluates_to_false. Retrieved 4/6 statements.


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = var_0 is not var_1
    var_3 = var_0.loader



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_gen_api. Retrieved 13/14 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = {var_0: var_0}
    var_2 = '/path/to/test_module'
    var_3 = 'docs'
    var_4 = True
    var_5 = 1
    var_6 = False
    var_7 = True
    var_8 = module_0.gen_api(var_1, var_2, prefix=var_3, link=var_4, level=var_5, toc=var_6, dry=var_7)
    var_9 = len(var_8)
    assert var_9 == 1
    var_10 = 0
    var_11 = var_8[var_10]
    var_12 = '# test_module API'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_pure_py_is_false_when_ext_is_not_py. Retrieved 15/35 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'MockLogger'
    var_1 = ()
    var_2 = 'debug'
    var_3 = 'warning'
    var_4 = None
    var_5 = lambda *args: var_4
    var_6 = lambda *args: var_4
    var_7 = {var_2: var_5, var_3: var_6}
    var_8 = '.so'
    var_9 = '.pyd'
    var_10 = 'root'
    var_11 = 'pwd'
    var_12 = False
    var_13 = 1
    var_14 = module_0.loader(var_10, var_11, var_12, var_13, var_12)
    assert var_14 == 'compiled'



# Parsed testcases at query #16
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = '/nonexistent_directory/test_file.txt'
    var_1 = 'test content'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_gen_api_with_valid_root_names. Retrieved 6/7 statements.
# Partially parsed test_gen_api_with_empty_root_names. Retrieved 4/5 statements.
# Partially parsed test_gen_api_with_nonexistent_module. Retrieved 6/7 statements.
# Partially parsed test_gen_api_with_multiple_modules. Retrieved 8/9 statements.
# Partially parsed test_gen_api_with_custom_prefix. Retrieved 7/8 statements.
# Partially parsed test_gen_api_with_link_disabled. Retrieved 7/8 statements.
# Partially parsed test_gen_api_with_custom_level. Retrieved 7/8 statements.
# Partially parsed test_gen_api_with_toc_enabled. Retrieved 6/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = len(var_4)
    assert var_5 == 1

import apimd.loader as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.gen_api(var_0, dry=var_1)
    var_3 = len(var_2)
    assert var_3 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = 'nonexistent_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = len(var_4)
    assert var_5 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test1'
    var_1 = 'test2'
    var_2 = 'test_module1'
    var_3 = 'test_module2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True
    var_6 = module_0.gen_api(var_4, dry=var_5)
    var_7 = len(var_6)
    assert var_7 == 2

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = 'custom_docs'
    var_4 = True
    var_5 = module_0.gen_api(var_2, prefix=var_3, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 1

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = True
    var_5 = module_0.gen_api(var_2, link=var_3, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 1

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = True
    var_5 = module_0.gen_api(var_2, level=var_3, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 1

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, toc=var_3, dry=var_3)
    var_5 = len(var_4)
    assert var_5 == 1



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_loader. Retrieved 6/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'example'
    var_1 = '/path/to/package'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #19
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

import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existing_package'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'pytest'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_predicate_at_line_3_evaluates_to_true. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'test content'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_read_file_content. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, world!'
    var_2 = module_0._read(var_0)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test__read_file_exists. Retrieved 3/7 statements.
# Partially parsed test__read_file_not_exists. Retrieved 2/6 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'test content'
    var_2 = module_0._read(var_0)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existent_file.txt'
    var_1 = module_0._read(var_0)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_load_module_predicate_evaluates_to_false. Retrieved 4/6 statements.


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = var_0 is not var_1
    var_3 = var_0.loader



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_read_file_content. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = module_0._read(var_0)



# Parsed testcases at query #25
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = 'module.submodule.class.method'
    var_1 = 1
    var_2 = module_0.parent(var_0, level=var_1)
    assert var_2 == 'module.submodule.class'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module.submodule.class.method'
    var_1 = 2
    var_2 = module_0.parent(var_0, level=var_1)
    assert var_2 == 'module.submodule'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module.submodule'
    var_1 = 3
    var_2 = module_0.parent(var_0, level=var_1)
    assert var_2 == 'module'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = 1
    var_2 = module_0.parent(var_0, level=var_1)
    assert var_2 == 'module'

import apimd.parser as module_0

def test_case_0():
    var_0 = ''
    var_1 = 1
    var_2 = module_0.parent(var_0, level=var_1)
    assert var_2 == ''



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_write_to_file. Retrieved 3/6 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_write.txt'
    var_1 = 'Hello, World!'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_read_file_exists. Retrieved 3/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'test content'
    var_2 = module_0._read(var_0)
    assert var_2 == 'test content'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 3/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'test content'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/7 statements.
# Partially parsed test_write_overwrites_existing_file. Retrieved 5/9 statements.
# Partially parsed test_write_handles_unicode_characters. Retrieved 3/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, world!'
    var_2 = module_0._write(var_0, var_1)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Initial content'
    var_2 = 'Updated content'
    var_3 = module_0._write(var_0, var_1)
    var_4 = module_0._write(var_0, var_2)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'こんにちは世界'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #30
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = '/path/to/non/existent/file'
    var_1 = module_0._read(var_0)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_read_file_content. Retrieved 3/5 statements.
# Partially parsed test_read_empty_file. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = module_0._read(var_0)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'empty_file.txt'
    var_1 = ''
    var_2 = module_0._read(var_0)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existent_file.txt'
    var_1 = module_0._read(var_0)



# Parsed testcases at query #32
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existent_file.txt'
    var_1 = module_0._read(var_0)



# Parsed testcases at query #33
#--------------------------




import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module_name'
    var_2 = 'module_path'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is True

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'invalid_module'
    var_2 = 'invalid_path'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module_name'
    var_2 = 'invalid_path'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False



# Parsed testcases at query #34
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = 'a.b.c.d'
    var_1 = module_0.parent(var_0)
    assert var_1 == 'a.b.c'
    var_2 = 2
    var_3 = module_0.parent(var_0, level=var_2)
    assert var_3 == 'a.b'
    var_4 = 'a.b'
    var_5 = module_0.parent(var_4)
    assert var_5 == 'a'
    var_6 = 'a'
    var_7 = module_0.parent(var_6)
    assert var_7 == 'a'
    var_8 = 3
    var_9 = module_0.parent(var_0, level=var_8)
    assert var_9 == 'a'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_write_file. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #36
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.gen_api(var_0)
    var_2 = len(var_1)
    assert var_2 == 0



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_load_module_with_non_loader. Retrieved 5/10 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_path'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'not_a_loader'
    var_1 = 'test_module'
    var_2 = 'test_path'
    var_3 = module_0.Parser()
    var_4 = module_1._load_module(var_1, var_2, var_3)
    assert var_4 is False



# Parsed testcases at query #38
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
    var_0 = 'nonexistent_module'
    var_1 = '/path/to/nonexistent_module.py'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'invalid_module'
    var_1 = '/invalid/path'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is False



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_load_module_with_valid_spec_and_loader. Retrieved 4/21 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = '/path/to/test_module.py'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test__load_module_correct_loader. Retrieved 5/6 statements.


import apimd.parser as module_0
import _frozen_importlib_external as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = '/path/to/test_module.py'
    var_2 = module_0.Parser()
    var_3 = module_1.spec_from_file_location(var_0, var_1)
    var_4 = var_3.loader



# Parsed testcases at query #41
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = {}
    var_1 = None
    var_2 = module_0.gen_api(var_0, var_1)



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_predicate_at_line_3_evaluates_to_true. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'test content'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #43
#--------------------------




import apimd.loader as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = 'temp_docs'
    var_4 = module_0.gen_api(var_2, prefix=var_3)
    var_5 = module_1.isdir(var_3)



# Parsed testcases at query #44
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'example'
    var_1 = 'example_module'
    var_2 = {var_0: var_1}
    var_3 = '/path/to/module'
    var_4 = 'docs'
    var_5 = True
    var_6 = 1
    var_7 = False
    var_8 = False
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)
    var_10 = len(var_9)



# Parsed testcases at query #45
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = 'pwd'
    var_2 = False
    var_3 = 1
    var_4 = module_0.loader(var_0, var_1, var_2, var_3, var_2)



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_predicate_at_line_3_evaluates_to_true. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'test content'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #47
#--------------------------




import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'module_name'
    var_2 = 'module_path'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False



# Parsed testcases at query #48
#--------------------------






# Parsed testcases at query #49
#--------------------------

# Partially parsed test_write_predicate_evaluates_to_false. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'sample text'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #50
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = module_0._read(var_0)



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_read_file_content. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'test content'
    var_2 = module_0._read(var_0)



# Parsed testcases at query #52
#--------------------------




import apimd.loader as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = 'non_existent_directory'
    var_4 = True
    var_5 = module_0.gen_api(var_2, prefix=var_3, dry=var_4)
    var_6 = module_1.isdir(var_3)



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_load_module_with_valid_spec_and_loader. Retrieved 4/9 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = '/path/to/module'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is True



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_gen_api_with_valid_inputs. Retrieved 11/12 statements.
# Partially parsed test_gen_api_with_invalid_module. Retrieved 11/12 statements.
# Partially parsed test_gen_api_with_no_pwd. Retrieved 11/12 statements.
# Partially parsed test_gen_api_with_multiple_modules. Retrieved 13/14 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestModule'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = '/path/to/module'
    var_4 = 'docs'
    var_5 = True
    var_6 = 1
    var_7 = False
    var_8 = True
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)
    var_10 = len(var_9)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'InvalidModule'
    var_1 = 'invalid_module'
    var_2 = {var_0: var_1}
    var_3 = '/path/to/invalid/module'
    var_4 = 'docs'
    var_5 = True
    var_6 = 1
    var_7 = False
    var_8 = True
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)
    var_10 = len(var_9)
    assert var_10 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'TestModule'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = 'docs'
    var_4 = True
    var_5 = 1
    var_6 = False
    var_7 = True
    var_8 = None
    var_9 = module_0.gen_api(var_2, var_8, prefix=var_3, link=var_4, level=var_5, toc=var_6, dry=var_7)
    var_10 = len(var_9)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Module1'
    var_1 = 'Module2'
    var_2 = 'module1'
    var_3 = 'module2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = '/path/to/modules'
    var_6 = 'docs'
    var_7 = True
    var_8 = 1
    var_9 = False
    var_10 = True
    var_11 = module_0.gen_api(var_4, var_5, prefix=var_6, link=var_7, level=var_8, toc=var_9, dry=var_10)
    var_12 = len(var_11)
    assert var_12 == 2



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_loader_with_valid_module. Retrieved 7/8 statements.
# Partially parsed test_loader_with_invalid_module. Retrieved 7/8 statements.
# Partially parsed test_loader_with_toc_disabled. Retrieved 6/7 statements.
# Partially parsed test_loader_with_no_link. Retrieved 6/7 statements.
# Partially parsed test_loader_with_different_level. Retrieved 6/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'example'
    var_1 = '/path/to/package'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = len(var_5)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = '/path/to/package'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = len(var_5)
    assert var_6 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'example'
    var_1 = '/path/to/package'
    var_2 = True
    var_3 = 1
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'example'
    var_1 = '/path/to/package'
    var_2 = False
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'example'
    var_1 = '/path/to/package'
    var_2 = True
    var_3 = 2
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_load_module_with_valid_spec_and_loader. Retrieved 5/7 statements.


import apimd.parser as module_0
import importlib._abc as module_1
import apimd.loader as module_2

def test_case_0():
    var_0 = 'test_module'
    var_1 = '/fake/path/to/module.py'
    var_2 = module_0.Parser()
    var_3 = module_1.Loader()
    var_4 = module_2._load_module(var_0, var_1, var_2)
    assert var_4 is True



# Parsed testcases at query #57
#--------------------------




def test_case_0():
    var_0 = 'example_root'
    var_1 = 'example_pwd'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = False
    assert var_5 is False



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_write_file_successfully. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'test content'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'test content'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_gen_api_with_valid_root_names. Retrieved 5/8 statements.
# Partially parsed test_gen_api_with_none_pwd. Retrieved 6/7 statements.
# Partially parsed test_gen_api_with_custom_prefix. Retrieved 6/7 statements.
# Partially parsed test_gen_api_with_link_disabled. Retrieved 6/7 statements.
# Partially parsed test_gen_api_with_custom_level. Retrieved 6/7 statements.
# Partially parsed test_gen_api_with_toc_enabled. Retrieved 5/6 statements.
# Partially parsed test_gen_api_with_nonexistent_module. Retrieved 5/6 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)

import apimd.loader as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.gen_api(var_0, dry=var_1)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = True
    var_5 = module_0.gen_api(var_2, var_3, dry=var_4)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = 'custom_docs'
    var_4 = True
    var_5 = module_0.gen_api(var_2, prefix=var_3, dry=var_4)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = True
    var_5 = module_0.gen_api(var_2, link=var_3, dry=var_4)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = True
    var_5 = module_0.gen_api(var_2, level=var_3, dry=var_4)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, toc=var_3, dry=var_3)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = 'nonexistent_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_read_file_content. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = module_0._read(var_0)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'empty_file.txt'
    var_1 = ''
    var_2 = module_0._read(var_0)



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_load_module_predicate_evaluates_to_false. Retrieved 7/9 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = '/path/to/module.py'
    var_2 = module_0.Parser()
    var_3 = None
    var_4 = None
    var_5 = var_3 is not var_4
    var_6 = var_3.loader



