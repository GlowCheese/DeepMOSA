####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_gen_api_basic. Retrieved 11/13 statements.
# Partially parsed test_gen_api_multiple_packages. Retrieved 13/16 statements.
# Partially parsed test_gen_api_empty_package. Retrieved 9/10 statements.
# Partially parsed test_gen_api_custom_prefix. Retrieved 9/10 statements.


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
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[var_6]
    var_10 = '# Test API\n\n'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test1'
    var_1 = 'Test2'
    var_2 = 'test_package1'
    var_3 = 'test_package2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = 'test_docs'
    var_7 = False
    var_8 = 2
    var_9 = True
    var_10 = module_0.gen_api(var_4, var_5, prefix=var_6, link=var_7, level=var_8, toc=var_9, dry=var_9)
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = '##'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Empty'
    var_1 = 'nonexistent_package'
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
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'custom_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 1



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_loader_basic. Retrieved 5/7 statements.
# Partially parsed test_loader_with_toc. Retrieved 5/6 statements.


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
    var_4 = '**Table of contents:**'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'test_path'
    var_2 = False
    var_3 = 1
    var_4 = module_0.loader(var_0, var_1, var_2, var_3, var_2)
    var_5 = '<a id='
    var_6 = bool('<a id=' not in var_4)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'test_path'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = 2
    var_6 = module_0.loader(var_0, var_1, var_2, var_5, var_3)
    var_7 = bool(var_4 != var_6)
    assert var_7 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_gen_api_with_valid_root_names. Retrieved 8/10 statements.
# Partially parsed test_gen_api_with_custom_prefix. Retrieved 6/7 statements.
# Partially parsed test_gen_api_with_dry_run. Retrieved 6/7 statements.
# Partially parsed test_gen_api_with_custom_level. Retrieved 9/11 statements.
# Partially parsed test_gen_api_with_toc_enabled. Retrieved 6/7 statements.
# Partially parsed test_gen_api_with_link_disabled. Retrieved 6/7 statements.
# Partially parsed test_gen_api_with_custom_pwd. Retrieved 5/6 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.gen_api(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = module_0.gen_api(var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_3[var_5]
    var_7 = '# test_package API\n\n'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = 'nonexistent_package'
    var_2 = {var_0: var_1}
    var_3 = module_0.gen_api(var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'custom_docs'
    var_4 = module_0.gen_api(var_2, prefix=var_3)
    var_5 = len(var_4)
    assert var_5 == 1

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = len(var_4)
    assert var_5 == 1

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = module_0.gen_api(var_2, level=var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = '## test_package API\n\n'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, toc=var_3)
    var_5 = len(var_4)
    assert var_5 == 1

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = module_0.gen_api(var_2, link=var_3)
    var_5 = len(var_4)
    assert var_5 == 1

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = '/custom/path'
    var_4 = module_0.gen_api(var_2, var_3)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_site_path_with_valid_package. Retrieved 2/3 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = module_0._site_path(var_0)
    var_2 = bool(var_1 != '')
    assert var_2 is True

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



# Parsed testcases at query #5
#--------------------------




import genericpath as module_0

def test_case_0():
    var_0 = 'non_existent_directory'
    var_1 = module_0.isdir(var_0)
    var_2 = bool(not var_1)
    assert var_2 is True



# Parsed testcases at query #6
#--------------------------




def test_case_0():
    var_0 = 'test_module'
    var_1 = '/path/to/module'
    var_2 = False
    var_3 = '.pyi'
    var_4 = var_1 + var_3
    var_5 = True
    var_6 = bool(not var_2)
    assert var_6 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_loader_basic. Retrieved 7/10 statements.
# Partially parsed test_loader_with_toc. Retrieved 5/6 statements.
# Partially parsed test_loader_different_level. Retrieved 7/8 statements.


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
    var_2 = True
    var_3 = module_0.loader(var_0, var_1, var_2, var_2, var_2)
    var_4 = '**Table of contents:**'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'test_pkg_path'
    var_2 = False
    var_3 = 1
    var_4 = module_0.loader(var_0, var_1, var_2, var_3, var_2)
    var_5 = '<a id='
    var_6 = bool('<a id=' not in var_4)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'test_pkg_path'
    var_2 = True
    var_3 = 2
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = '#'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'empty_pkg'
    var_1 = 'empty_pkg_path'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    assert var_4 == '\n'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existent'
    var_1 = 'non_existent_path'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    assert var_4 == '\n'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_loader_basic. Retrieved 6/7 statements.
# Partially parsed test_loader_no_link. Retrieved 5/6 statements.
# Partially parsed test_loader_with_toc. Retrieved 4/5 statements.
# Partially parsed test_loader_different_level. Retrieved 8/11 statements.
# Partially parsed test_loader_empty_package. Retrieved 6/7 statements.
# Partially parsed test_loader_with_extensions. Retrieved 6/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'test_path'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = len(var_4)
    var_6 = bool(var_5 > 0)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'test_path'
    var_2 = False
    var_3 = 1
    var_4 = module_0.loader(var_0, var_1, var_2, var_3, var_2)
    var_5 = '<a id='
    var_6 = bool('<a id=' not in var_4)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'test_path'
    var_2 = True
    var_3 = module_0.loader(var_0, var_1, var_2, var_2, var_2)
    var_4 = '**Table of contents:**'
    var_5 = bool('**Table of contents:**' in var_3)
    assert var_5 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'test_path'
    var_2 = True
    var_3 = 2
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = '##'
    var_7 = '###'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'empty_pkg'
    var_1 = 'empty_path'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = len(var_4)
    var_6 = bool(var_5 > 0)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'ext_pkg'
    var_1 = 'ext_path'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = len(var_4)
    var_6 = bool(var_5 > 0)
    assert var_6 is True



# Parsed testcases at query #9
#--------------------------




def test_case_0():
    var_0 = '.pyi'
    assert var_0 == '.py'
    var_1 = '.py'
    assert var_1 is False



# Parsed testcases at query #10
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

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'invalid_module'
    var_2 = 'path/to/invalid_module.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'non_loader_module'
    var_2 = 'path/to/non_loader_module.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_site_path_returns_empty_string_when_spec_is_none. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = None
    var_1 = 'test'
    var_2 = module_0._site_path(var_1)
    assert var_2 == ''



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_site_path_returns_correct_path_for_existing_module. Retrieved 2/3 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_module'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'some_module_without_submodules'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = module_0._site_path(var_0)
    var_2 = bool(var_1 != '')
    assert var_2 is True
    var_3 = bool('site-packages' in var_1 or 'dist-packages' in var_1)
    assert var_3 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Test content'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #14
#--------------------------




def test_case_0():
    var_0 = bool(not '.py' == '.py')
    assert var_0 is True



# Parsed testcases at query #15
#--------------------------




import genericpath as module_0

def test_case_0():
    var_0 = 'docs'
    var_1 = module_0.isdir(var_0)
    assert var_1 is False



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_write_creates_file. Retrieved 3/8 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'test content'
    var_2 = module_0._write(var_0, var_1)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test__write_creates_file_with_content. Retrieved 3/8 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Test content'
    var_2 = module_0._write(var_0, var_1)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True



# Parsed testcases at query #18
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'existing_file.txt'
    var_1 = module_0._read(var_0)
    assert var_1 == 'content of existing_file.txt'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_file.txt'
    var_1 = module_0._read(var_0)
    assert var_1 == ''



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_site_path_existing_package. Retrieved 2/3 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = module_0._site_path(var_0)
    var_2 = bool(var_1 != '')
    assert var_2 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_package_name_12345'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'sys'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_load_module_success. Retrieved 4/5 statements.
# Partially parsed test_load_module_failure. Retrieved 4/5 statements.
# Partially parsed test_load_module_parent_import_error. Retrieved 4/5 statements.


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

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'child_module'
    var_2 = 'path/to/child_module.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_loader_basic_functionality. Retrieved 6/7 statements.
# Partially parsed test_loader_empty_package. Retrieved 5/6 statements.
# Partially parsed test_loader_nonexistent_package. Retrieved 5/6 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = 'path/to/package'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = len(var_4)
    var_6 = bool(var_5 > 0)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = 'path/to/package'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = module_0.loader(var_0, var_1, var_3, var_2, var_3)
    var_6 = bool(var_4 != var_5)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = 'path/to/package'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = 2
    var_6 = module_0.loader(var_0, var_1, var_2, var_5, var_3)
    var_7 = bool(var_4 != var_6)
    assert var_7 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_package'
    var_1 = 'path/to/package'
    var_2 = True
    var_3 = module_0.loader(var_0, var_1, var_2, var_2, var_2)
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_2, var_4)
    var_6 = bool(var_3 != var_5)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'empty_package'
    var_1 = 'path/to/empty'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_package'
    var_1 = 'path/to/nonexistent'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)



# Parsed testcases at query #22
#--------------------------




def test_case_0():
    var_0 = False
    var_1 = '.py'
    var_2 = 'some/path.py'
    var_3 = True
    var_4 = bool(not var_0)
    assert var_4 is True
    var_5 = '.py'
    var_6 = var_1 == var_5
    var_7 = var_6 and var_3
    var_8 = True
    var_9 = False
    var_10 = var_8 if var_7 else var_9
    var_11 = bool(var_10)
    assert var_11 is True



# Parsed testcases at query #23
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'valid_file_path.txt'
    var_1 = module_0._read(var_0)
    assert var_1 == 'expected file content'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_site_path_with_none_spec. Retrieved 4/8 statements.
# Partially parsed test_site_path_with_none_submodule_search_locations. Retrieved 3/10 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda name: var_0
    var_2 = 'test_module'
    var_3 = module_0._site_path(var_2)
    assert var_3 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = None
    var_1 = 'test_module'
    var_2 = module_0._site_path(var_1)
    assert var_2 == ''



# Parsed testcases at query #25
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



# Parsed testcases at query #26
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = module_0._read(var_0)
    assert var_1 == 'expected content'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_load_module_success. Retrieved 5/11 statements.
# Partially parsed test_load_module_import_error. Retrieved 5/7 statements.
# Partially parsed test_load_module_spec_none. Retrieved 5/8 statements.
# Partially parsed test_load_module_loader_not_instance. Retrieved 6/9 statements.


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
    var_0 = 'Test module loading with import error.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'path/to/test_module.py'
    var_4 = module_1._load_module(var_2, var_3, var_1)
    assert var_4 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test module loading with spec_from_file_location returning None.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'path/to/test_module.py'
    var_4 = module_1._load_module(var_2, var_3, var_1)
    assert var_4 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'Test module loading with loader not being an instance of Loader.'
    var_1 = module_0.Parser()
    var_2 = 'test_module'
    var_3 = 'path/to/test_module.py'
    var_4 = 'not a loader'
    var_5 = module_1._load_module(var_2, var_3, var_1)
    assert var_5 is False



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/8 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'test content'
    var_2 = module_0._write(var_0, var_1)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_gen_api_basic. Retrieved 12/14 statements.
# Partially parsed test_gen_api_multiple_modules. Retrieved 12/14 statements.
# Partially parsed test_gen_api_with_site_path. Retrieved 6/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_module'
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
    var_0 = 'Module1'
    var_1 = 'Module2'
    var_2 = 'mod1'
    var_3 = 'mod2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = 'test_docs'
    var_7 = True
    var_8 = False
    var_9 = module_0.gen_api(var_4, var_5, prefix=var_6, link=var_7, level=var_7, toc=var_8, dry=var_7)
    var_10 = len(var_9)
    assert var_10 == 2
    var_11 = '#'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Missing'
    var_1 = 'nonexistent_module'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'test_docs'
    var_5 = True
    var_6 = module_0.gen_api(var_2, var_3, prefix=var_4, dry=var_5)
    var_7 = len(var_6)
    assert var_7 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'custom_dir'
    var_5 = True
    var_6 = module_0.gen_api(var_2, var_3, prefix=var_4, dry=var_5)
    var_7 = len(var_6)
    assert var_7 == 1

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = '/custom/path'
    var_4 = True
    var_5 = module_0.gen_api(var_2, var_3, dry=var_4)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = len(var_4)
    assert var_5 == 1



# Parsed testcases at query #30
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = module_0._read(var_0)
    assert var_1 == 'expected content'



# Parsed testcases at query #31
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = module_0._read(var_0)
    assert var_1 == 'This is a test file.'



# Parsed testcases at query #32
#--------------------------




def test_case_0():
    var_0 = 'test'
    var_1 = 'module'
    var_2 = {var_0: var_1}
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True



# Parsed testcases at query #33
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'invalid_path.txt'
    var_1 = module_0._read(var_0)
    var_2 = bool(not var_1)
    assert var_2 is True



# Parsed testcases at query #34
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = module_0._read(var_0)
    assert var_1 == 'Expected content'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_load_module_with_valid_spec_and_loader. Retrieved 4/10 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module.py'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_write_creates_file. Retrieved 3/6 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'test content'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_gen_api_with_valid_root_names. Retrieved 11/12 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'test_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[var_6]
    var_10 = '# Test API\n\n'



# Parsed testcases at query #38
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existent_file.txt'
    var_1 = module_0._read(var_0)
    var_2 = bool(not var_1)
    assert var_2 is True



# Parsed testcases at query #39
#--------------------------




import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = 'test.py'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is False



# Parsed testcases at query #40
#--------------------------

# Partially parsed test__write_creates_file_with_content. Retrieved 3/8 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Test content'
    var_2 = module_0._write(var_0, var_1)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_load_module_success. Retrieved 4/5 statements.
# Partially parsed test_load_module_failure. Retrieved 4/5 statements.
# Partially parsed test_load_module_with_parent_import_error. Retrieved 4/5 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os.path'
    var_2 = 'path/to/os/path.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is True

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'nonexistent.module'
    var_2 = 'path/to/nonexistent.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'nonexistent.submodule'
    var_2 = 'path/to/submodule.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_loader_basic. Retrieved 6/8 statements.
# Partially parsed test_loader_with_toc. Retrieved 5/6 statements.
# Partially parsed test_loader_different_level. Retrieved 7/8 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'tests/test_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = '#'

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
    var_6 = '##'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'tests/test_pkg'
    var_2 = False
    var_3 = 1
    var_4 = module_0.loader(var_0, var_1, var_2, var_3, var_2)
    var_5 = '<a id='
    var_6 = bool('<a id=' not in var_4)
    assert var_6 is True

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
    var_0 = 'parent_pkg'
    var_1 = 'tests/parent_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = 'parent_pkg.submodule'
    var_6 = bool('parent_pkg.submodule' in var_4)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'doc_pkg'
    var_1 = 'tests/doc_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = 'This is a test module'
    var_6 = bool('This is a test module' in var_4)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'const_pkg'
    var_1 = 'tests/const_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = 'Constants'
    var_6 = bool('Constants' in var_4)
    assert var_6 is True
    var_7 = 'TEST_CONST'
    var_8 = bool('TEST_CONST' in var_4)
    assert var_8 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'class_pkg'
    var_1 = 'tests/class_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = 'class TestClass'
    var_6 = bool('class TestClass' in var_4)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'func_pkg'
    var_1 = 'tests/func_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = 'test_function()'
    var_6 = bool('test_function()' in var_4)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'import_pkg'
    var_1 = 'tests/import_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = 'imported_module'
    var_6 = bool('imported_module' in var_4)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'async_pkg'
    var_1 = 'tests/async_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = 'async test_async()'
    var_6 = bool('async test_async()' in var_4)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'decorator_pkg'
    var_1 = 'tests/decorator_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = 'Decorators'
    var_6 = bool('Decorators' in var_4)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'enum_pkg'
    var_1 = 'tests/enum_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = 'Enums'
    var_6 = bool('Enums' in var_4)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'member_pkg'
    var_1 = 'tests/member_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = 'Members'
    var_6 = bool('Members' in var_4)
    assert var_6 is True
    var_7 = 'Type'
    var_8 = bool('Type' in var_4)
    assert var_8 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'inherit_pkg'
    var_1 = 'tests/inherit_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = 'Bases'
    var_6 = bool('Bases' in var_4)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'alias_pkg'
    var_1 = 'tests/alias_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = 'TypeAlias'
    var_6 = bool('TypeAlias' in var_4)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'private_pkg'
    var_1 = 'tests/private_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = '_private'
    var_6 = bool('_private' not in var_4)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'all_pkg'
    var_1 = 'tests/all_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = 'included_module'
    var_6 = bool('included_module' in var_4)
    assert var_6 is True
    var_7 = 'excluded_module'
    var_8 = bool('excluded_module' not in var_4)
    assert var_8 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_pure_py_is_false_when_no_py_file. Retrieved 4/9 statements.


def test_case_0():
    var_0 = False
    var_1 = 'nonexistent_path'
    var_2 = False
    var_3 = True
    var_4 = bool(not var_3)
    assert var_4 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_gen_api. Retrieved 12/14 statements.


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



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_loader_basic. Retrieved 7/10 statements.
# Partially parsed test_loader_no_toc. Retrieved 6/8 statements.
# Partially parsed test_loader_with_level. Retrieved 8/11 statements.
# Partially parsed test_loader_empty_package. Retrieved 5/7 statements.
# Partially parsed test_loader_nonexistent_package. Retrieved 5/7 statements.
# Partially parsed test_loader_with_toc. Retrieved 5/7 statements.
# Partially parsed test_loader_link_disabled. Retrieved 5/6 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'test_path'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = '**Table of contents:**'
    var_6 = '#'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'test_path'
    var_2 = False
    var_3 = 1
    var_4 = module_0.loader(var_0, var_1, var_2, var_3, var_2)
    var_5 = '**Table of contents:**'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'test_path'
    var_2 = True
    var_3 = 2
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = '**Table of contents:**'
    var_7 = '##'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'empty_pkg'
    var_1 = 'empty_path'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_pkg'
    var_1 = 'nonexistent_path'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'test_path'
    var_2 = True
    var_3 = module_0.loader(var_0, var_1, var_2, var_2, var_2)
    var_4 = '**Table of contents:**'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'test_path'
    var_2 = False
    var_3 = 1
    var_4 = module_0.loader(var_0, var_1, var_2, var_3, var_2)
    var_5 = '<a id='
    var_6 = bool('<a id=' not in var_4)
    assert var_6 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_loader_basic. Retrieved 6/7 statements.
# Partially parsed test_loader_with_different_levels. Retrieved 7/8 statements.
# Partially parsed test_loader_with_toc_enabled. Retrieved 4/5 statements.
# Partially parsed test_loader_with_link_disabled. Retrieved 5/6 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'tests/test_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = len(var_4)
    var_6 = bool(var_5 > 0)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'tests/test_pkg'
    var_2 = True
    var_3 = 2
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = len(var_5)
    var_7 = bool(var_6 > 0)
    assert var_7 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'tests/test_pkg'
    var_2 = True
    var_3 = module_0.loader(var_0, var_1, var_2, var_2, var_2)
    var_4 = 'Table of contents'
    var_5 = bool('Table of contents' in var_3)
    assert var_5 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'tests/test_pkg'
    var_2 = False
    var_3 = 1
    var_4 = module_0.loader(var_0, var_1, var_2, var_3, var_2)
    var_5 = '<a id='
    var_6 = bool('<a id=' not in var_4)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existent'
    var_1 = 'tests/empty'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    assert var_4 == '\n'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'empty_pkg'
    var_1 = 'tests/empty_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    assert var_4 == '\n'



# Parsed testcases at query #6
#--------------------------




import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is True
    var_4 = 'test_module'
    var_5 = bool('test_module' in var_0.doc)
    assert var_5 is True
    var_6 = 'test_module'
    var_7 = bool('test_module' in var_0.root)
    assert var_7 is True

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'nonexistent_module'
    var_2 = 'nonexistent_module.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False
    var_4 = 'nonexistent_module'
    var_5 = bool('nonexistent_module' not in var_0.doc)
    assert var_5 is True
    var_6 = 'nonexistent_module'
    var_7 = bool('nonexistent_module' not in var_0.root)
    assert var_7 is True

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'child_module'
    var_2 = 'child_module.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False
    var_4 = 'child_module'
    var_5 = bool('child_module' not in var_0.doc)
    assert var_5 is True
    var_6 = 'child_module'
    var_7 = bool('child_module' not in var_0.root)
    assert var_7 is True



# Parsed testcases at query #7
#--------------------------




import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'nonexistent_module'
    var_1 = 'dummy_path'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is False



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_pure_py_remains_false_when_no_py_file. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = '/path/to/module'
    var_2 = False
    var_3 = '.py'
    var_4 = False
    var_5 = True
    var_6 = True
    var_7 = bool(not var_6)
    assert var_7 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_loader_basic. Retrieved 6/7 statements.
# Partially parsed test_loader_with_toc. Retrieved 4/5 statements.
# Partially parsed test_loader_with_different_level. Retrieved 6/7 statements.
# Partially parsed test_loader_nonexistent_package. Retrieved 6/7 statements.
# Partially parsed test_loader_empty_package. Retrieved 6/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'tests/test_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = len(var_4)
    var_6 = bool(var_5 > 0)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'tests/test_pkg'
    var_2 = True
    var_3 = module_0.loader(var_0, var_1, var_2, var_2, var_2)
    var_4 = 'Table of contents'
    var_5 = bool('Table of contents' in var_3)
    assert var_5 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'tests/test_pkg'
    var_2 = False
    var_3 = 2
    var_4 = module_0.loader(var_0, var_1, var_2, var_3, var_2)
    var_5 = len(var_4)
    var_6 = bool(var_5 > 0)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_pkg'
    var_1 = 'tests/nonexistent'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = len(var_4)
    var_6 = bool(var_5 > 0)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'empty_pkg'
    var_1 = 'tests/empty_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = len(var_4)
    var_6 = bool(var_5 > 0)
    assert var_6 is True



# Parsed testcases at query #10
#--------------------------




import genericpath as module_0

def test_case_0():
    var_0 = 'docs'
    var_1 = module_0.isdir(var_0)
    assert var_1 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_loader_basic. Retrieved 6/8 statements.
# Partially parsed test_loader_with_toc. Retrieved 5/6 statements.
# Partially parsed test_loader_different_level. Retrieved 7/8 statements.
# Partially parsed test_loader_empty_package. Retrieved 5/6 statements.
# Partially parsed test_loader_with_submodules. Retrieved 5/6 statements.


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
    var_5 = '<a id='
    var_6 = bool('<a id=' not in var_4)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'empty_pkg'
    var_1 = 'tests/empty_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'parent_pkg'
    var_1 = 'tests/parent_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = 'submodule'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_gen_api_basic. Retrieved 9/10 statements.
# Partially parsed test_gen_api_multiple_roots. Retrieved 13/15 statements.
# Partially parsed test_gen_api_with_prefix. Retrieved 10/11 statements.
# Partially parsed test_gen_api_with_custom_level. Retrieved 10/11 statements.
# Partially parsed test_gen_api_with_toc. Retrieved 9/10 statements.
# Partially parsed test_gen_api_with_link_false. Retrieved 9/10 statements.
# Partially parsed test_gen_api_with_pwd. Retrieved 10/11 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = '# Test API\n\n'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test1'
    var_1 = 'Test2'
    var_2 = 'test1'
    var_3 = 'test2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True
    var_6 = module_0.gen_api(var_4, dry=var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = 0
    var_9 = var_6[var_8]
    var_10 = '# Test1 API\n\n'
    var_11 = var_6[var_5]
    var_12 = '# Test2 API\n\n'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = True
    var_5 = module_0.gen_api(var_2, prefix=var_3, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 0
    var_8 = var_5[var_7]
    var_9 = '# Test API\n\n'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = True
    var_5 = module_0.gen_api(var_2, level=var_3, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 0
    var_8 = var_5[var_7]
    var_9 = '## Test API\n\n'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, toc=var_3, dry=var_3)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = '# Test API\n\n'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = True
    var_5 = module_0.gen_api(var_2, link=var_3, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = var_5[var_3]
    var_8 = '# Test API\n\n'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = '/tmp'
    var_4 = True
    var_5 = module_0.gen_api(var_2, var_3, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 0
    var_8 = var_5[var_7]
    var_9 = '# Test API\n\n'

import apimd.loader as module_0

def test_case_0():
    var_0 = {}
    var_1 = True
    var_2 = module_0.gen_api(var_0, dry=var_1)
    var_3 = len(var_2)
    assert var_3 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Nonexistent'
    var_1 = 'nonexistent_package'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = len(var_4)
    assert var_5 == 0



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_dry_mode_logs_documentation. Retrieved 11/18 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = 'Load root: test_module (Test)'
    var_6 = 'Write file: docs/test-module-api.md'
    var_7 = '='
    var_8 = 12
    var_9 = var_7 * var_8
    var_10 = '# Test API\n\nTest documentation'



# Parsed testcases at query #14
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



# Parsed testcases at query #15
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



# Parsed testcases at query #16
#--------------------------




import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'path/to/test_module.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is True
    var_4 = 'test_module'
    var_5 = bool('test_module' in var_0.doc)
    assert var_5 is True

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'nonexistent_module'
    var_2 = 'path/to/nonexistent.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False
    var_4 = 'nonexistent_module'
    var_5 = bool('nonexistent_module' not in var_0.doc)
    assert var_5 is True

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'sub.module'
    var_2 = 'path/to/sub/module.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False
    var_4 = 'sub.module'
    var_5 = bool('sub.module' not in var_0.doc)
    assert var_5 is True

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'invalid_module'
    var_2 = 'path/to/invalid.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False
    var_4 = 'invalid_module'
    var_5 = bool('invalid_module' not in var_0.doc)
    assert var_5 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_site_path_with_valid_package. Retrieved 3/4 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = module_0._site_path(var_0)
    var_2 = 'site-packages'

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



# Parsed testcases at query #18
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = module_0._read(var_0)
    assert var_1 == 'expected content'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test__load_module_predicate_false. Retrieved 1/5 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Test content'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_site_path_returns_empty_string_when_spec_is_none. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = None
    var_1 = 'nonexistent_module'
    var_2 = module_0._site_path(var_1)
    assert var_2 == ''



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_site_path_returns_empty_string_when_spec_is_none. Retrieved 3/6 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = None
    var_1 = 'non_existent_module'
    var_2 = module_0._site_path(var_1)
    assert var_2 == ''



# Parsed testcases at query #23
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_module'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_write_creates_file. Retrieved 3/8 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'test content'
    var_2 = module_0._write(var_0, var_1)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_write_creates_file_and_writes_content. Retrieved 3/8 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'test content'
    var_2 = module_0._write(var_0, var_1)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True



# Parsed testcases at query #26
#--------------------------




import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os.path'
    var_2 = 'path/to/os/path.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is True
    var_4 = var_0.docstring['os.path']
    var_5 = bool(var_0.docstring['os.path'] is not None)
    assert var_5 is True

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'nonexistent.module'
    var_2 = 'path/to/nonexistent.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False
    var_4 = var_0.docstring
    var_5 = bool(var_0.docstring == {})
    assert var_5 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_pure_py_remains_false_when_no_py_file. Retrieved 8/11 statements.


import apimd.loader as module_0
import ast as module_1

def test_case_0():
    var_0 = 'test'
    var_1 = '/path/to/'
    var_2 = '.pyi'
    var_3 = var_1 + var_2
    var_4 = False
    var_5 = 1
    var_6 = module_0._read(var_3)
    var_7 = module_1.parse(var_0, var_6)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test__load_module_predicate_false. Retrieved 1/5 statements.


def test_case_0():
    var_0 = None



# Parsed testcases at query #29
#--------------------------




def test_case_0():
    var_0 = True
    var_1 = bool(var_0)
    assert var_1 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_loader_basic. Retrieved 7/10 statements.
# Partially parsed test_loader_with_toc. Retrieved 5/6 statements.
# Partially parsed test_loader_different_level. Retrieved 8/10 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'test_pkg_path'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = '#'
    var_6 = '**'

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
    var_2 = False
    var_3 = 1
    var_4 = module_0.loader(var_0, var_1, var_2, var_3, var_2)
    var_5 = '<a id='
    var_6 = bool('<a id=' not in var_4)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'test_pkg_path'
    var_2 = True
    var_3 = 2
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = '##'
    var_7 = '**'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'empty_pkg'
    var_1 = 'empty_pkg_path'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    assert var_4 == '\n'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'parent_pkg'
    var_1 = 'parent_pkg_path'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = 'parent_pkg.submodule'
    var_6 = bool('parent_pkg.submodule' in var_4)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'ext_pkg'
    var_1 = 'ext_pkg_path'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = 'ext_pkg'
    var_6 = bool('ext_pkg' in var_4)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'mixed_pkg'
    var_1 = 'mixed_pkg_path'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = 'mixed_pkg.py_module'
    var_6 = bool('mixed_pkg.py_module' in var_4)
    assert var_6 is True
    var_7 = 'mixed_pkg.ext_module'
    var_8 = bool('mixed_pkg.ext_module' in var_4)
    assert var_8 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_load_module_success. Retrieved 4/5 statements.
# Partially parsed test_load_module_failure. Retrieved 4/5 statements.
# Partially parsed test_load_module_with_parent_import_error. Retrieved 4/5 statements.


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
    var_2 = 'path/to/nonexistent.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'child_module'
    var_2 = 'path/to/child_module.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/8 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'test content'
    var_2 = module_0._write(var_0, var_1)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True



# Parsed testcases at query #33
#--------------------------




import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_path'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is False



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_predicate_false. Retrieved 4/6 statements.


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = var_0 is not var_1
    var_3 = var_0.loader



# Parsed testcases at query #35
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = module_0._read(var_0)
    assert var_1 == 'Expected content'



# Parsed testcases at query #36
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_path.txt'
    var_1 = module_0._read(var_0)
    var_2 = bool(not var_1)
    assert var_2 is True



# Parsed testcases at query #37
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/8 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'test content'
    var_2 = module_0._write(var_0, var_1)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Test content'
    var_2 = module_0._write(var_0, var_1)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True



# Parsed testcases at query #40
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = module_0._read(var_0)
    assert var_1 == 'Expected content of test_file.txt'



# Parsed testcases at query #41
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_directory/file.txt'
    var_1 = 'test content'
    var_2 = module_0._write(var_0, var_1)
    var_3 = bool(not var_2)
    assert var_3 is True



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_read_existing_file. Retrieved 3/7 statements.


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



# Parsed testcases at query #43
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_file.txt'
    var_1 = module_0._read(var_0)
    var_2 = bool(not var_1)
    assert var_2 is True



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_read_returns_file_content. Retrieved 3/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = "print('Hello, World!')"
    var_2 = module_0._read(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True



# Parsed testcases at query #45
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'path/to/file'
    var_1 = module_0._read(var_0)
    assert var_1 == 'expected content'



