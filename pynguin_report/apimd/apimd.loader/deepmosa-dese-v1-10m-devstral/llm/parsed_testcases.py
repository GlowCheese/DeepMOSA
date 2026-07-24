####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_gen_api. Retrieved 10/11 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_title'
    var_1 = 'test_name'
    var_2 = {var_0: var_1}
    var_3 = 'test_pwd'
    var_4 = 'test_prefix'
    var_5 = False
    var_6 = 2
    var_7 = True
    var_8 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_7)
    var_9 = len(var_8)
    assert var_9 == 1



# Parsed testcases at query #2
#--------------------------




import genericpath as module_0

def test_case_0():
    var_0 = 'docs'
    var_1 = module_0.isdir(var_0)



# Parsed testcases at query #3
#--------------------------




import genericpath as module_0

def test_case_0():
    var_0 = 'nonexistent_directory'
    var_1 = module_0.isdir(var_0)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_loader_basic. Retrieved 7/10 statements.
# Partially parsed test_loader_with_toc. Retrieved 5/6 statements.
# Partially parsed test_loader_different_level. Retrieved 8/10 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'tests/test_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = '#'
    var_6 = '**Table of contents:**'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'empty_pkg'
    var_1 = 'tests/empty_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    assert var_4 == '\n'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'tests/test_pkg'
    var_2 = True
    var_3 = module_0.loader(var_0, var_1, var_2, var_2, var_2)
    var_4 = '**Table of contents:**'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'tests/test_pkg'
    var_2 = False
    var_3 = 1
    var_4 = module_0.loader(var_0, var_1, var_2, var_3, var_2)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'tests/test_pkg'
    var_2 = True
    var_3 = 2
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = '##'
    var_7 = '**Table of contents:**'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existent'
    var_1 = 'tests/non_existent'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    assert var_4 == '\n'



# Parsed testcases at query #5
#--------------------------




import genericpath as module_0

def test_case_0():
    var_0 = 'docs'
    var_1 = module_0.isdir(var_0)



# Parsed testcases at query #6
#--------------------------




def test_case_0():
    var_0 = '.pyi'
    assert var_0 == '.py'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Test content'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #8
#--------------------------




def test_case_0():
    var_0 = False
    var_1 = '.pyi'
    var_2 = 'some_path.pyi'



# Parsed testcases at query #9
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = True
    var_2 = False
    var_3 = module_0.loader(var_0, var_0, var_1, var_1, var_2)
    var_4 = 2
    var_5 = module_0.loader(var_0, var_0, var_2, var_4, var_1)
    var_6 = 'non_existent_pkg'
    var_7 = 'non_existent_path'
    var_8 = module_0.loader(var_6, var_7, var_1, var_1, var_2)
    assert var_8 == ''
    var_9 = 3
    var_10 = module_0.loader(var_0, var_0, var_1, var_9, var_2)
    var_11 = module_0.loader(var_0, var_0, var_1, var_1, var_1)



# Parsed testcases at query #10
#--------------------------




def test_case_0():
    var_0 = '.pyi'
    assert var_0 == '.py'



# Parsed testcases at query #11
#--------------------------




def test_case_0():
    var_0 = '.pyi'
    assert var_0 == '.py'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_site_path_with_valid_module. Retrieved 5/6 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = module_0._site_path(var_0)
    var_2 = module_0._site_path(var_0)
    var_3 = ''
    var_4 = var_2 == var_3

import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_module'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'sys'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''



# Parsed testcases at query #13
#--------------------------




def test_case_0():
    var_0 = False



# Parsed testcases at query #14
#--------------------------




def test_case_0():
    var_0 = None



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_site_path_existing_package. Retrieved 5/6 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = module_0._site_path(var_0)
    var_2 = module_0._site_path(var_0)
    var_3 = ''
    var_4 = var_2 == var_3

import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_package_12345'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_site_path_with_none_spec. Retrieved 2/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''



# Parsed testcases at query #17
#--------------------------




def test_case_0():
    var_0 = False
    var_1 = '.py'



# Parsed testcases at query #18
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #19
#--------------------------




def test_case_0():
    var_0 = None



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_load_module_success. Retrieved 5/6 statements.
# Partially parsed test_load_module_failure. Retrieved 5/6 statements.
# Partially parsed test_load_module_with_parent_import_error. Retrieved 5/6 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test successful module loading.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'path/to/test_module.py'
    var_4 = module_1._load_module(var_2, var_3, var_1)
    assert var_4 is True

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test failed module loading.'
    var_1 = module_0.Parser()
    var_2 = 'nonexistent_module'
    var_3 = 'path/to/nonexistent.py'
    var_4 = module_1._load_module(var_2, var_3, var_1)
    assert var_4 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test module loading when parent import fails.'
    var_1 = module_0.Parser()
    var_2 = 'child_module'
    var_3 = 'path/to/child_module.py'
    var_4 = module_1._load_module(var_2, var_3, var_1)
    assert var_4 is False



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Test content'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #22
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = module_0._read(var_0)
    assert var_1 == 'Expected content of test_file.txt'



# Parsed testcases at query #23
#--------------------------




import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_path.py'
    var_2 = module_0.Parser()
    var_3 = None
    var_4 = module_1._load_module(var_0, var_1, var_2)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_gen_api_with_valid_inputs. Retrieved 12/14 statements.
# Partially parsed test_gen_api_with_none_pwd. Retrieved 9/10 statements.
# Partially parsed test_gen_api_with_dry_false. Retrieved 9/10 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = '/path/to/test'
    var_4 = 'test_docs'
    var_5 = False
    var_6 = 2
    var_7 = True
    var_8 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_7)
    var_9 = len(var_8)
    assert var_9 == 1
    var_10 = var_8[var_5]
    var_11 = '## Test API\n\n'

import apimd.loader as module_0

def test_case_0():
    var_0 = {}
    var_1 = '/path/to/test'
    var_2 = 'test_docs'
    var_3 = False
    var_4 = 2
    var_5 = True
    var_6 = module_0.gen_api(var_0, var_1, prefix=var_2, link=var_3, level=var_4, toc=var_5, dry=var_5)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'test_docs'
    var_5 = False
    var_6 = 2
    var_7 = True
    var_8 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_7)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Invalid'
    var_1 = 'nonexistent_package'
    var_2 = {var_0: var_1}
    var_3 = '/path/to/test'
    var_4 = 'test_docs'
    var_5 = False
    var_6 = 2
    var_7 = True
    var_8 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_7)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = '/path/to/test'
    var_4 = 'test_docs'
    var_5 = False
    var_6 = 2
    var_7 = True
    var_8 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_5)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_predicate_line_9_false. Retrieved 7/9 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'module.name'
    var_1 = 'path/to/module'
    var_2 = module_0.Parser()
    var_3 = None
    var_4 = None
    var_5 = var_3 is not var_4
    var_6 = var_3.loader



# Parsed testcases at query #26
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = '/invalid/path/that/does/not/exist/file.txt'
    var_1 = 'test content'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #27
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_file.txt'
    var_1 = module_0._read(var_0)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/8 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Test content'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/8 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Test content'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_load_module_with_valid_spec_and_loader. Retrieved 5/6 statements.


import apimd.parser as module_0
import _frozen_importlib_external as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module.py'
    var_2 = module_0.Parser()
    var_3 = module_1.spec_from_file_location(var_0, var_1)
    var_4 = var_3.loader



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_gen_api_basic. Retrieved 12/14 statements.
# Partially parsed test_gen_api_with_pwd. Retrieved 11/13 statements.
# Partially parsed test_gen_api_empty_root_names. Retrieved 7/8 statements.
# Partially parsed test_gen_api_nonexistent_package. Retrieved 9/10 statements.
# Partially parsed test_gen_api_custom_prefix. Retrieved 11/13 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'test_docs'
    var_5 = False
    var_6 = 2
    var_7 = True
    var_8 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_7)
    var_9 = len(var_8)
    assert var_9 == 1
    var_10 = var_8[var_5]
    var_11 = '## Test API\n\n'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 'some/path'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[var_6]
    var_10 = '# Test API\n\n'

import apimd.loader as module_0

def test_case_0():
    var_0 = {}
    var_1 = None
    var_2 = 'test_docs'
    var_3 = True
    var_4 = False
    var_5 = module_0.gen_api(var_0, var_1, prefix=var_2, link=var_3, level=var_3, toc=var_4, dry=var_3)
    var_6 = len(var_5)
    assert var_6 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'NonExistent'
    var_1 = 'nonexistent'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'test_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'custom_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[var_6]
    var_10 = '# Test API\n\n'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_gen_api_root_names_iteration. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'test_title'
    var_1 = 'test_name'
    var_2 = {var_0: var_1}



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_gen_api_with_valid_root_names. Retrieved 12/14 statements.
# Partially parsed test_gen_api_with_invalid_root_names. Retrieved 10/11 statements.
# Partially parsed test_gen_api_with_multiple_root_names. Retrieved 16/19 statements.
# Partially parsed test_gen_api_with_dry_false. Retrieved 11/13 statements.
# Partially parsed test_gen_api_with_custom_prefix. Retrieved 11/13 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = '/path/to/test'
    var_4 = 'test_docs'
    var_5 = False
    var_6 = 2
    var_7 = True
    var_8 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_7)
    var_9 = len(var_8)
    assert var_9 == 1
    var_10 = var_8[var_5]
    var_11 = '## Test API\n\n'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Invalid'
    var_1 = 'non_existent_package'
    var_2 = {var_0: var_1}
    var_3 = '/path/to/test'
    var_4 = 'test_docs'
    var_5 = False
    var_6 = 2
    var_7 = True
    var_8 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_7)
    var_9 = len(var_8)
    assert var_9 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test1'
    var_1 = 'Test2'
    var_2 = 'test_package1'
    var_3 = 'test_package2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = '/path/to/test'
    var_6 = 'test_docs'
    var_7 = False
    var_8 = 1
    var_9 = True
    var_10 = module_0.gen_api(var_4, var_5, prefix=var_6, link=var_7, level=var_8, toc=var_7, dry=var_9)
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = var_10[var_7]
    var_13 = '# Test1 API\n\n'
    var_14 = var_10[var_9]
    var_15 = '# Test2 API\n\n'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = '/path/to/test'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_6)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[var_6]
    var_10 = '# Test API\n\n'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = '/path/to/test'
    var_4 = 'custom_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[var_6]
    var_10 = '# Test API\n\n'



# Parsed testcases at query #34
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = 'a.b.c'
    var_1 = 1
    var_2 = module_0.parent(var_0, level=var_1)
    assert var_2 == 'a.b'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'a.b.c'
    var_1 = 2
    var_2 = module_0.parent(var_0, level=var_1)
    assert var_2 == 'a'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'a.b.c'
    var_1 = module_0.parent(var_0)
    assert var_1 == 'a.b'



# Parsed testcases at query #35
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_file.txt'
    var_1 = module_0._read(var_0)



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/8 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Test content'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #37
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'invalid_path.txt'
    var_1 = module_0._read(var_0)



# Parsed testcases at query #38
#--------------------------

# Partially parsed test__load_module_with_valid_spec_and_loader. Retrieved 4/19 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module.py'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_read_returns_file_content. Retrieved 5/6 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = 'w'
    var_3 = open(var_0, var_2)
    var_4 = module_0._read(var_0)



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_gen_api_with_valid_root_names. Retrieved 11/13 statements.
# Partially parsed test_gen_api_with_empty_root_names. Retrieved 8/9 statements.
# Partially parsed test_gen_api_with_nonexistent_package. Retrieved 9/10 statements.
# Partially parsed test_gen_api_with_dry_false. Retrieved 11/13 statements.
# Partially parsed test_gen_api_with_custom_prefix. Retrieved 11/13 statements.
# Partially parsed test_gen_api_with_different_level. Retrieved 11/13 statements.
# Partially parsed test_gen_api_with_toc_enabled. Retrieved 11/13 statements.
# Partially parsed test_gen_api_with_toc_disabled. Retrieved 11/13 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {var_0: var_0}
    var_2 = '.'
    var_3 = 'test_docs'
    var_4 = False
    var_5 = 2
    var_6 = True
    var_7 = module_0.gen_api(var_1, var_2, prefix=var_3, link=var_4, level=var_5, toc=var_6, dry=var_6)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[var_4]
    var_10 = '## test API\n\n'

import apimd.loader as module_0

def test_case_0():
    var_0 = {}
    var_1 = '.'
    var_2 = 'test_docs'
    var_3 = False
    var_4 = 2
    var_5 = True
    var_6 = module_0.gen_api(var_0, var_1, prefix=var_2, link=var_3, level=var_4, toc=var_5, dry=var_5)
    var_7 = len(var_6)
    assert var_7 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = {var_0: var_0}
    var_2 = '.'
    var_3 = 'test_docs'
    var_4 = False
    var_5 = 2
    var_6 = True
    var_7 = module_0.gen_api(var_1, var_2, prefix=var_3, link=var_4, level=var_5, toc=var_6, dry=var_6)
    var_8 = len(var_7)
    assert var_8 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {var_0: var_0}
    var_2 = '.'
    var_3 = 'test_docs'
    var_4 = False
    var_5 = 2
    var_6 = True
    var_7 = module_0.gen_api(var_1, var_2, prefix=var_3, link=var_4, level=var_5, toc=var_6, dry=var_4)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[var_4]
    var_10 = '## test API\n\n'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {var_0: var_0}
    var_2 = '.'
    var_3 = 'custom_docs'
    var_4 = False
    var_5 = 2
    var_6 = True
    var_7 = module_0.gen_api(var_1, var_2, prefix=var_3, link=var_4, level=var_5, toc=var_6, dry=var_6)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[var_4]
    var_10 = '## test API\n\n'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {var_0: var_0}
    var_2 = '.'
    var_3 = 'test_docs'
    var_4 = False
    var_5 = 3
    var_6 = True
    var_7 = module_0.gen_api(var_1, var_2, prefix=var_3, link=var_4, level=var_5, toc=var_6, dry=var_6)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[var_4]
    var_10 = '### test API\n\n'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {var_0: var_0}
    var_2 = '.'
    var_3 = 'test_docs'
    var_4 = True
    var_5 = 2
    var_6 = module_0.gen_api(var_1, var_2, prefix=var_3, link=var_4, level=var_5, toc=var_4, dry=var_4)
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 0
    var_9 = var_6[var_8]
    var_10 = '## test API\n\n'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = {var_0: var_0}
    var_2 = '.'
    var_3 = 'test_docs'
    var_4 = True
    var_5 = 2
    var_6 = False
    var_7 = module_0.gen_api(var_1, var_2, prefix=var_3, link=var_4, level=var_5, toc=var_6, dry=var_4)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[var_6]
    var_10 = '## test API\n\n'



# Parsed testcases at query #41
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #42
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



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/8 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Test content'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #44
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_dry_mode_prevents_file_writing. Retrieved 9/10 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = False
    var_5 = module_0.gen_api(var_2, link=var_4, level=var_3, toc=var_4, dry=var_3)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = var_5[var_4]
    var_8 = '# test_module API\n\n'



# Parsed testcases at query #46
#--------------------------




import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_path'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is False



# Parsed testcases at query #47
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = module_0._read(var_0)
    assert var_1 == 'expected content'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_load_module_success. Retrieved 4/5 statements.
# Partially parsed test_load_module_failure. Retrieved 4/5 statements.
# Partially parsed test_load_module_with_parent_import_error. Retrieved 4/5 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is True

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'nonexistent_module'
    var_2 = 'nonexistent_module.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'child_module'
    var_2 = 'child_module.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False



# Parsed testcases at query #49
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = module_0._read(var_0)
    assert var_1 == 'This is a test file.'



# Parsed testcases at query #50
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = module_0._read(var_0)
    assert var_1 == 'expected content'



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_load_module_success. Retrieved 4/5 statements.
# Partially parsed test_load_module_failure. Retrieved 4/5 statements.
# Partially parsed test_load_module_with_parent_import_error. Retrieved 4/5 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is True

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'nonexistent_module'
    var_2 = 'nonexistent_module.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'child_module'
    var_2 = 'child_module.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_predicate_at_line_9. Retrieved 4/8 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_path.py'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is True



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_site_path_returns_empty_string_when_spec_is_none. Retrieved 2/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_module'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_load_module_success. Retrieved 4/5 statements.
# Partially parsed test_load_module_failure. Retrieved 4/5 statements.
# Partially parsed test_load_module_with_parent_import_error. Retrieved 4/5 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is True

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'nonexistent_module'
    var_2 = 'nonexistent_module.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'child_module'
    var_2 = 'child_module.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/8 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Test content'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #56
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'test_pkg_path'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = 2
    var_6 = module_0.loader(var_0, var_1, var_3, var_5, var_2)
    var_7 = 'empty_pkg'
    var_8 = 'empty_pkg_path'
    var_9 = module_0.loader(var_7, var_8, var_2, var_2, var_3)
    assert var_9 == ''
    var_10 = 'invalid_pkg'
    var_11 = 'invalid_pkg_path'
    var_12 = module_0.loader(var_10, var_11, var_2, var_2, var_3)
    assert var_12 == ''



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_write_creates_file. Retrieved 3/6 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, world!'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #58
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existent_file.txt'
    var_1 = module_0._read(var_0)



# Parsed testcases at query #59
#--------------------------




def test_case_0():
    var_0 = True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_loader_basic. Retrieved 5/6 statements.
# Partially parsed test_loader_with_toc. Retrieved 4/5 statements.
# Partially parsed test_loader_no_link. Retrieved 5/6 statements.
# Partially parsed test_loader_different_level. Retrieved 6/7 statements.
# Partially parsed test_loader_empty_package. Retrieved 5/6 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'test_path'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'test_path'
    var_2 = True
    var_3 = module_0.loader(var_0, var_1, var_2, var_2, var_2)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'test_path'
    var_2 = False
    var_3 = 1
    var_4 = module_0.loader(var_0, var_1, var_2, var_3, var_2)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'test_path'
    var_2 = True
    var_3 = 2
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'empty_pkg'
    var_1 = 'empty_path'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_gen_api. Retrieved 9/10 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 0



# Parsed testcases at query #3
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = module_0._site_path(var_0)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_package'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'sys'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_pure_py_is_false_when_no_py_file. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = '/path/to/module'
    var_2 = False
    var_3 = True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_pure_py_false_when_no_py_file. Retrieved 3/7 statements.


def test_case_0():
    var_0 = False
    var_1 = 'nonexistent_path'
    var_2 = True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_gen_api_with_valid_root_names. Retrieved 9/10 statements.
# Partially parsed test_gen_api_with_empty_root_names. Retrieved 7/8 statements.
# Partially parsed test_gen_api_with_invalid_root_names. Retrieved 9/10 statements.
# Partially parsed test_gen_api_with_custom_prefix. Retrieved 9/10 statements.
# Partially parsed test_gen_api_with_dry_false. Retrieved 9/10 statements.
# Partially parsed test_gen_api_with_different_level. Retrieved 10/11 statements.
# Partially parsed test_gen_api_with_toc_enabled. Retrieved 8/9 statements.
# Partially parsed test_gen_api_with_link_disabled. Retrieved 10/11 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)

import apimd.loader as module_0

def test_case_0():
    var_0 = {}
    var_1 = None
    var_2 = 'docs'
    var_3 = True
    var_4 = False
    var_5 = module_0.gen_api(var_0, var_1, prefix=var_2, link=var_3, level=var_3, toc=var_4, dry=var_3)
    var_6 = len(var_5)
    assert var_6 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Invalid'
    var_1 = 'nonexistent_module'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'custom_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_6)
    var_8 = len(var_7)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'docs'
    var_5 = True
    var_6 = 2
    var_7 = False
    var_8 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_5)
    var_9 = len(var_8)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'docs'
    var_5 = True
    var_6 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_5, dry=var_5)
    var_7 = len(var_6)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'docs'
    var_5 = False
    var_6 = 1
    var_7 = True
    var_8 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_5, dry=var_7)
    var_9 = len(var_8)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Test content'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #8
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'existing_file.txt'
    var_1 = module_0._read(var_0)
    assert var_1 == 'Expected content'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existing_file.txt'
    var_1 = module_0._read(var_0)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_loader_basic. Retrieved 7/10 statements.
# Partially parsed test_loader_with_toc. Retrieved 5/6 statements.
# Partially parsed test_loader_different_level. Retrieved 8/10 statements.
# Partially parsed test_loader_with_stub_files. Retrieved 5/6 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'path/to/test_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = '#'
    var_6 = '**Table of contents:**'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'path/to/test_pkg'
    var_2 = True
    var_3 = module_0.loader(var_0, var_1, var_2, var_2, var_2)
    var_4 = '**Table of contents:**'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'path/to/test_pkg'
    var_2 = False
    var_3 = 1
    var_4 = module_0.loader(var_0, var_1, var_2, var_3, var_2)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'path/to/test_pkg'
    var_2 = True
    var_3 = 2
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = '##'
    var_7 = '**Table of contents:**'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'empty_pkg'
    var_1 = 'path/to/empty_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    assert var_4 == '\n'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'stub_pkg'
    var_1 = 'path/to/stub_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_site_path_returns_empty_string_when_spec_is_none. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = None
    var_1 = 'nonexistent_module'
    var_2 = module_0._site_path(var_1)
    assert var_2 == ''



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_load_module_success. Retrieved 4/5 statements.


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
    var_0 = 'nonexistent_module'
    var_1 = 'nonexistent_module.py'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is False



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Test content'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #13
#--------------------------




def test_case_0():
    var_0 = '.pyi'
    assert var_0 == '.py'



# Parsed testcases at query #14
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
    var_0 = 'nonexistent_module'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'sys'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''



# Parsed testcases at query #15
#--------------------------




import genericpath as module_0

def test_case_0():
    var_0 = 'docs'
    var_1 = module_0.isdir(var_0)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test__load_module_with_none_spec. Retrieved 8/10 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'Test that the predicate evaluates to False when spec is None.'
    var_1 = 'test_module'
    var_2 = 'test_path.py'
    var_3 = module_0.Parser()
    var_4 = None
    var_5 = None
    var_6 = var_4 is not var_5
    var_7 = var_4.loader



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 4/6 statements.


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = var_0 is not var_1
    var_3 = var_0.loader



# Parsed testcases at query #18
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'test_pkg_path'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = 2
    var_6 = module_0.loader(var_0, var_1, var_3, var_5, var_2)
    var_7 = 'empty_pkg'
    var_8 = 'empty_pkg_path'
    var_9 = module_0.loader(var_7, var_8, var_2, var_2, var_3)
    assert var_9 == ''
    var_10 = 'invalid_pkg'
    var_11 = 'invalid_pkg_path'
    var_12 = module_0.loader(var_10, var_11, var_2, var_2, var_3)
    assert var_12 == ''



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_gen_api_with_valid_root_names. Retrieved 11/13 statements.
# Partially parsed test_gen_api_with_invalid_root_names. Retrieved 9/10 statements.
# Partially parsed test_gen_api_with_custom_prefix. Retrieved 9/10 statements.
# Partially parsed test_gen_api_with_dry_false. Retrieved 9/10 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[var_6]
    var_10 = '# Test API\n\n'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Invalid'
    var_1 = 'invalid_module'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'custom_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 1

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_6)
    var_8 = len(var_7)
    assert var_8 == 1



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_loader_basic. Retrieved 7/10 statements.
# Partially parsed test_loader_no_toc. Retrieved 6/7 statements.
# Partially parsed test_loader_with_toc. Retrieved 5/6 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'test_pkg_path'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = '**Table of contents:**'
    var_6 = '#'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'test_pkg_path'
    var_2 = False
    var_3 = 1
    var_4 = module_0.loader(var_0, var_1, var_2, var_3, var_2)
    var_5 = '**Table of contents:**'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'test_pkg_path'
    var_2 = True
    var_3 = module_0.loader(var_0, var_1, var_2, var_2, var_2)
    var_4 = '**Table of contents:**'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'test_pkg_path'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = 2
    var_6 = module_0.loader(var_0, var_1, var_2, var_5, var_3)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'test_pkg_path'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'test_pkg_path'
    var_2 = False
    var_3 = 1
    var_4 = module_0.loader(var_0, var_1, var_2, var_3, var_2)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_site_path_returns_empty_string_when_spec_is_none. Retrieved 4/8 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda name: var_0
    var_2 = 'nonexistent_module'
    var_3 = module_0._site_path(var_2)
    assert var_3 == ''



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Test content'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_pure_py_remains_false_when_no_py_file. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = '/path/to/module'
    var_2 = False



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_site_path_with_valid_package. Retrieved 5/6 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = module_0._site_path(var_0)
    var_2 = module_0._site_path(var_0)
    var_3 = ''
    var_4 = var_2 == var_3

import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_package'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'sys'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''



# Parsed testcases at query #25
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_file.txt'
    var_1 = module_0._read(var_0)



# Parsed testcases at query #26
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'invalid_path.txt'
    var_1 = module_0._read(var_0)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_load_module_with_valid_spec_and_loader. Retrieved 4/7 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_path.py'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is True



# Parsed testcases at query #28
#--------------------------




def test_case_0():
    var_0 = 'test'
    var_1 = 'module'
    var_2 = {var_0: var_1}



# Parsed testcases at query #29
#--------------------------




def test_case_0():
    var_0 = False
    var_1 = '.pyi'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/8 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Test content'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #31
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'path/to/test/file.txt'
    var_1 = module_0._read(var_0)
    assert var_1 == 'expected content'



# Parsed testcases at query #32
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Test content'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_write_creates_file. Retrieved 3/6 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'test content'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #35
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #36
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'existing_file.txt'
    var_1 = module_0._read(var_0)
    assert var_1 == 'expected content'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_file.txt'
    var_1 = module_0._read(var_0)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'empty_file.txt'
    var_1 = module_0._read(var_0)
    assert var_1 == ''



# Parsed testcases at query #37
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = module_0._read(var_0)
    assert var_1 == 'expected content'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test__load_module_predicate. Retrieved 4/21 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_path'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)



# Parsed testcases at query #39
#--------------------------




import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'path/to/test_module.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is True

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'nonexistent_module'
    var_2 = 'path/to/nonexistent_module.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False



# Parsed testcases at query #40
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_file.txt'
    var_1 = module_0._read(var_0)



# Parsed testcases at query #41
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'existing_file.txt'
    var_1 = module_0._read(var_0)
    assert var_1 == 'Expected content'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existing_file.txt'
    var_1 = module_0._read(var_0)



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_gen_api_with_valid_root_names. Retrieved 9/10 statements.
# Partially parsed test_gen_api_with_invalid_root_names. Retrieved 9/10 statements.
# Partially parsed test_gen_api_with_custom_prefix. Retrieved 9/10 statements.
# Partially parsed test_gen_api_with_different_level. Retrieved 10/11 statements.
# Partially parsed test_gen_api_with_toc_enabled. Retrieved 8/9 statements.
# Partially parsed test_gen_api_with_dry_run_disabled. Retrieved 9/10 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Invalid'
    var_1 = 'nonexistent_package'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'custom_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'docs'
    var_5 = True
    var_6 = 2
    var_7 = False
    var_8 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_5)
    var_9 = len(var_8)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'docs'
    var_5 = True
    var_6 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_5, dry=var_5)
    var_7 = len(var_6)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_6)
    var_8 = len(var_7)



# Parsed testcases at query #43
#--------------------------




def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'w+'
    var_2 = 'utf-8'
    var_3 = open(var_0, var_1, encoding=var_2)
    var_4 = None
    var_5 = var_3 == var_4



# Parsed testcases at query #44
#--------------------------




def test_case_0():
    var_0 = {}



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 1/5 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #46
#--------------------------




def test_case_0():
    var_0 = 'test'
    var_1 = 'module'
    var_2 = {var_0: var_1}



# Parsed testcases at query #47
#--------------------------

# Partially parsed test__load_module_with_valid_spec_and_loader. Retrieved 4/9 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_path.py'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is True



# Parsed testcases at query #48
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.gen_api(var_0)



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_gen_api_with_valid_input. Retrieved 11/13 statements.
# Partially parsed test_gen_api_with_none_pwd. Retrieved 8/9 statements.
# Partially parsed test_gen_api_with_custom_prefix. Retrieved 8/9 statements.
# Partially parsed test_gen_api_with_different_level. Retrieved 11/12 statements.
# Partially parsed test_gen_api_with_toc_enabled. Retrieved 7/8 statements.
# Partially parsed test_gen_api_with_dry_false. Retrieved 8/9 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = '/path/to/test'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[var_6]
    var_10 = '# Test API\n\n'

import apimd.loader as module_0

def test_case_0():
    var_0 = {}
    var_1 = '/path/to/test'
    var_2 = 'test_docs'
    var_3 = True
    var_4 = False
    var_5 = module_0.gen_api(var_0, var_1, prefix=var_2, link=var_3, level=var_3, toc=var_4, dry=var_3)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'test_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Invalid'
    var_1 = 'nonexistent_package'
    var_2 = {var_0: var_1}
    var_3 = '/path/to/test'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = '/path/to/test'
    var_4 = 'custom_prefix'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = '/path/to/test'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = 2
    var_7 = False
    var_8 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_5)
    var_9 = var_8[var_7]
    var_10 = '## Test API\n\n'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = '/path/to/test'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_5, dry=var_5)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = '/path/to/test'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_6)



# Parsed testcases at query #50
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



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_load_module_with_valid_spec_and_loader. Retrieved 4/7 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_path.py'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is True



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_pure_py_false_when_no_py_file. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = '/path/to/module'
    var_2 = False



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_loader_basic. Retrieved 6/8 statements.
# Partially parsed test_loader_with_toc. Retrieved 5/6 statements.
# Partially parsed test_loader_different_level. Retrieved 7/8 statements.
# Partially parsed test_loader_empty_package. Retrieved 5/6 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'tests/test_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = '# Module `test_pkg`'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'tests/test_pkg'
    var_2 = True
    var_3 = module_0.loader(var_0, var_1, var_2, var_2, var_2)
    var_4 = '**Table of contents:**'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'tests/test_pkg'
    var_2 = True
    var_3 = 2
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = '## Module `test_pkg`'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'tests/test_pkg'
    var_2 = False
    var_3 = 1
    var_4 = module_0.loader(var_0, var_1, var_2, var_3, var_2)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'empty_pkg'
    var_1 = 'tests/empty_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'tests/test_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'ext_pkg'
    var_1 = 'tests/ext_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'mixed_pkg'
    var_1 = 'tests/mixed_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_site_path_returns_empty_string_when_spec_is_none. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = None
    var_1 = 'test_module'
    var_2 = module_0._site_path(var_1)
    assert var_2 == ''



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_write_creates_file. Retrieved 6/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'test content'
    var_2 = module_0._write(var_0, var_1)
    var_3 = 'r'
    var_4 = 'utf-8'
    var_5 = open(var_0, var_3, encoding=var_4)



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_gen_api. Retrieved 11/13 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[var_6]
    var_10 = '# Test API\n\n'



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_doc_not_empty. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'content'



# Parsed testcases at query #58
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = len(var_4)
    assert var_5 == 0



# Parsed testcases at query #59
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'nonexistent.module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)



# Parsed testcases at query #60
#--------------------------




import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_path.py'
    var_2 = module_0.Parser()
    var_3 = None
    var_4 = module_1._load_module(var_0, var_1, var_2)
    assert var_4 is False



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_write_creates_file. Retrieved 3/6 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'test content'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_gen_api_empty_doc. Retrieved 6/9 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = 1
    var_4 = module_0.gen_api(var_2, level=var_3)
    var_5 = "'test_module' can not be found"



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_read_returns_file_content. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test content'
    var_1 = 'test_file.txt'
    var_2 = module_0._read(var_1)



# Parsed testcases at query #64
#--------------------------

# Partially parsed test__read. Retrieved 3/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('Hello, World!')"
    var_2 = module_0._read(var_0)



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_write_creates_file. Retrieved 3/6 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'test content'
    var_2 = module_0._write(var_0, var_1)



