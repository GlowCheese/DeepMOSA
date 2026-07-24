####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_gen_api_with_valid_root_names. Retrieved 11/12 statements.
# Partially parsed test_gen_api_with_different_level. Retrieved 11/12 statements.


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
    var_0 = 'Invalid'
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

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'test_docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_6)
    var_8 = len(var_7)
    assert var_8 == 1

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = None
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
    var_3 = None
    var_4 = 'test_docs'
    var_5 = True
    var_6 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_5, dry=var_5)
    var_7 = len(var_6)
    assert var_7 == 1

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = None
    var_4 = 'test_docs'
    var_5 = False
    var_6 = 1
    var_7 = True
    var_8 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_5, dry=var_7)
    var_9 = len(var_8)
    assert var_9 == 1



# Parsed testcases at query #2
#--------------------------




import genericpath as module_0

def test_case_0():
    var_0 = 'docs'
    var_1 = module_0.isdir(var_0)
    var_2 = bool(not var_1)
    assert var_2 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_loader_basic. Retrieved 5/7 statements.
# Partially parsed test_loader_empty_package. Retrieved 5/6 statements.
# Partially parsed test_loader_with_stub_files. Retrieved 5/6 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'path/to/test_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'path/to/test_pkg'
    var_2 = True
    var_3 = module_0.loader(var_0, var_1, var_2, var_2, var_2)
    var_4 = 'Table of contents:'
    var_5 = bool('Table of contents:' in var_3)
    assert var_5 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'path/to/test_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = 2
    var_6 = module_0.loader(var_0, var_1, var_2, var_5, var_3)
    var_7 = bool(var_4 != var_6)
    assert var_7 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'path/to/test_pkg'
    var_2 = False
    var_3 = 1
    var_4 = module_0.loader(var_0, var_1, var_2, var_3, var_2)
    var_5 = '<a id='
    var_6 = bool('<a id=' not in var_4)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'empty_pkg'
    var_1 = 'path/to/empty_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'stub_pkg'
    var_1 = 'path/to/stub_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/8 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Test content'
    var_2 = module_0._write(var_0, var_1)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True



# Parsed testcases at query #5
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
    var_1 = 'invalid.parent.child'
    var_2 = 'path/to/child.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_pure_py_remains_false_when_no_py_file. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = '/path/to/module'
    var_2 = False
    assert var_2 is False
    var_3 = False
    var_4 = bool(not var_3)
    assert var_4 is True



# Parsed testcases at query #7
#--------------------------




import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_path.py'
    var_2 = module_0.Parser()
    var_3 = None
    var_4 = module_1._load_module(var_0, var_1, var_2)
    var_5 = bool(not var_4)
    assert var_5 is True



# Parsed testcases at query #8
#--------------------------




def test_case_0():
    var_0 = False
    var_1 = '.pyi'
    var_2 = bool(not var_1 == '.py')
    assert var_2 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_loader_no_toc. Retrieved 5/7 statements.
# Partially parsed test_loader_with_toc. Retrieved 4/6 statements.
# Partially parsed test_loader_no_link. Retrieved 5/7 statements.
# Partially parsed test_loader_different_level. Retrieved 6/8 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'path/to/test_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)

import apimd.loader as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'test_pkg'
    var_3 = 'path/to/test_pkg'
    var_4 = module_0.loader(var_2, var_3, var_0, var_0, var_1)

import apimd.loader as module_0

def test_case_0():
    var_0 = True
    var_1 = 'test_pkg'
    var_2 = 'path/to/test_pkg'
    var_3 = module_0.loader(var_1, var_2, var_0, var_0, var_0)

import apimd.loader as module_0

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'test_pkg'
    var_3 = 'path/to/test_pkg'
    var_4 = module_0.loader(var_2, var_3, var_0, var_1, var_0)

import apimd.loader as module_0

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = False
    var_3 = 'test_pkg'
    var_4 = 'path/to/test_pkg'
    var_5 = module_0.loader(var_3, var_4, var_0, var_1, var_2)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'empty_pkg'
    var_1 = 'path/to/empty_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    assert var_4 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'ext_pkg'
    var_1 = 'path/to/ext_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'mixed_pkg'
    var_1 = 'path/to/mixed_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_gen_api_calls_loader_and_writes_files. Retrieved 15/18 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = '/path/to/test'
    var_4 = 'docs'
    var_5 = True
    var_6 = 1
    var_7 = False
    var_8 = False
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)
    var_10 = len(var_9)
    assert var_10 == 1
    var_11 = 0
    var_12 = var_9[var_11]
    var_13 = '# Test API\n\n'
    var_14 = 'test-api.md'
    var_15 = [var_14]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_site_path_existing_package. Retrieved 2/3 statements.


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



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_loader_basic. Retrieved 6/8 statements.
# Partially parsed test_loader_with_toc. Retrieved 5/6 statements.
# Partially parsed test_loader_different_level. Retrieved 7/8 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'tests/data'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = '# Module `test_pkg`'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'tests/data'
    var_2 = True
    var_3 = module_0.loader(var_0, var_1, var_2, var_2, var_2)
    var_4 = '**Table of contents:**'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'tests/data'
    var_2 = False
    var_3 = 1
    var_4 = module_0.loader(var_0, var_1, var_2, var_3, var_2)
    var_5 = '<a id='
    var_6 = bool('<a id=' not in var_4)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'tests/data'
    var_2 = True
    var_3 = 2
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = '## Module `test_pkg`'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'empty_pkg'
    var_1 = 'tests/data'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = 'Missing documentation'
    var_6 = bool('Missing documentation' in var_4)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'parent_pkg'
    var_1 = 'tests/data'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = 'Module `parent_pkg.submodule`'
    var_6 = bool('Module `parent_pkg.submodule`' in var_4)
    assert var_6 is True



# Parsed testcases at query #13
#--------------------------




import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_path.py'
    var_2 = module_0.Parser()
    var_3 = None
    var_4 = module_1._load_module(var_0, var_1, var_2)
    var_5 = bool(not var_4)
    assert var_5 is True



# Parsed testcases at query #14
#--------------------------




def test_case_0():
    var_0 = bool(not False)
    assert var_0 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_load_module_success. Retrieved 4/5 statements.
# Partially parsed test_load_module_failure. Retrieved 4/5 statements.
# Partially parsed test_load_module_invalid_spec. Retrieved 4/5 statements.


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
    var_1 = 'invalid.spec'
    var_2 = 'path/to/invalid.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False



# Parsed testcases at query #16
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



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_pure_py_remains_false_when_no_py_file. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = '/path/to/module'
    var_2 = False
    var_3 = True
    var_4 = bool(not var_3)
    assert var_4 is True



# Parsed testcases at query #18
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_module'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''



# Parsed testcases at query #19
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = module_0._read(var_0)
    assert var_1 == 'expected content'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_pure_py_false_when_no_py_file. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 'Test that pure_py remains False when no .py file is found.'
    var_1 = 'test_module'
    var_2 = '/path/to/module'
    var_3 = False
    assert var_3 is False
    var_4 = False
    var_5 = bool(not var_4)
    assert var_5 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_site_path_with_valid_module. Retrieved 3/4 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = module_0._site_path(var_0)
    var_2 = 'site-packages'

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



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/8 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Test content'
    var_2 = module_0._write(var_0, var_1)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_load_module_predicate_false. Retrieved 7/9 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_path.py'
    var_2 = module_0.Parser()
    var_3 = None
    var_4 = None
    var_5 = var_3 is not var_4
    var_6 = var_3.loader



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_loader_basic_functionality. Retrieved 7/10 statements.
# Partially parsed test_loader_with_toc_enabled. Retrieved 5/6 statements.


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
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = module_0.loader(var_0, var_1, var_3, var_2, var_3)
    var_6 = bool(var_4 != var_5)
    assert var_6 is True

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
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = 2
    var_6 = module_0.loader(var_0, var_1, var_2, var_5, var_3)
    var_7 = bool(var_4 != var_6)
    assert var_7 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'empty_pkg'
    var_1 = 'empty_path'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    assert var_4 == '\n'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_pkg'
    var_1 = 'nonexistent_path'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    assert var_4 == '\n'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_site_path_with_none_submodule_search_locations. Retrieved 3/11 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = None
    var_2 = module_0._site_path(var_0)
    assert var_2 == ''



# Parsed testcases at query #26
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = module_0._read(var_0)
    assert var_1 == 'Expected content of test_file.txt'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_write_creates_file_and_writes_content. Retrieved 3/8 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Test content'
    var_2 = module_0._write(var_0, var_1)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_write_predicate_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'Write'



# Parsed testcases at query #29
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = '/invalid/path/that/does/not/exist/file.txt'
    var_1 = 'test content'
    var_2 = module_0._write(var_0, var_1)
    var_3 = bool(not var_2)
    assert var_3 is True



# Parsed testcases at query #30
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #31
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = module_0._read(var_0)
    assert var_1 == 'expected content'



# Parsed testcases at query #32
#--------------------------




def test_case_0():
    var_0 = 'test'
    var_1 = 'module'
    var_2 = {var_0: var_1}
    var_3 = bool(var_2)
    assert var_3 is True



# Parsed testcases at query #33
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.gen_api(var_0)
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




import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = module_0._read(var_0)
    assert var_1 == 'This is a test file.'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_write_creates_file. Retrieved 3/6 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Test content'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_write_creates_file. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'test content'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #38
#--------------------------




def test_case_0():
    var_0 = 'test_title'
    var_1 = 'test_name'
    var_2 = {var_0: var_1}



# Parsed testcases at query #39
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)



# Parsed testcases at query #40
#--------------------------

# Partially parsed test__read_returns_file_content. Retrieved 3/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = module_0._read(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True



# Parsed testcases at query #41
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
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_gen_api_with_valid_inputs. Retrieved 9/10 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_path'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = 2
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_5, dry=var_5)
    var_8 = len(var_7)
    var_9 = bool(var_8 >= 0)
    assert var_9 is True



# Parsed testcases at query #43
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = module_0._read(var_0)
    assert var_1 == 'Hello, World!'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_file.txt'
    var_1 = module_0._read(var_0)



# Parsed testcases at query #44
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.gen_api(var_0)
    var_2 = bool(not var_1)
    assert var_2 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_loader_basic. Retrieved 7/10 statements.
# Partially parsed test_loader_with_toc. Retrieved 5/6 statements.
# Partially parsed test_loader_different_level. Retrieved 5/6 statements.
# Partially parsed test_loader_no_link. Retrieved 5/6 statements.
# Partially parsed test_loader_empty_package. Retrieved 5/6 statements.
# Partially parsed test_loader_non_existent_package. Retrieved 5/6 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'path/to/test_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = '**Table of contents:**'
    var_6 = '#'

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
    var_3 = 2
    var_4 = module_0.loader(var_0, var_1, var_2, var_3, var_2)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'path/to/test_pkg'
    var_2 = False
    var_3 = 1
    var_4 = module_0.loader(var_0, var_1, var_2, var_3, var_2)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'empty_pkg'
    var_1 = 'path/to/empty_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existent'
    var_1 = 'path/to/non_existent'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_pure_py_remains_false_when_no_py_file. Retrieved 4/8 statements.


def test_case_0():
    var_0 = False
    var_1 = 'nonexistent_path'
    var_2 = False
    var_3 = True
    var_4 = bool(not var_3)
    assert var_4 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_isfile_returns_false. Retrieved 2/3 statements.


import genericpath as module_0

def test_case_0():
    var_0 = 'any_path.py'
    var_1 = module_0.isfile(var_0)
    var_2 = bool(not var_1)
    assert var_2 is True



# Parsed testcases at query #4
#--------------------------




def test_case_0():
    var_0 = '.py'
    assert var_0 == '.py'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_pure_py_remains_false_when_no_py_file. Retrieved 3/7 statements.


def test_case_0():
    var_0 = False
    var_1 = 'nonexistent_path'
    var_2 = True
    var_3 = bool(not var_2)
    assert var_3 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Test content'
    var_2 = module_0._write(var_0, var_1)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_pure_py_remains_false_when_no_py_file. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = '/path/to/module'
    var_2 = False
    assert var_2 is False
    var_3 = False
    var_4 = bool(not var_3)
    assert var_4 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_gen_api_basic. Retrieved 12/14 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 'test_path'
    var_4 = 'test_docs'
    var_5 = False
    var_6 = 2
    var_7 = True
    var_8 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_7)
    var_9 = len(var_8)
    assert var_9 == 1
    var_10 = var_8[var_5]
    var_11 = '## Test API\n\n'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_site_path_existing_package. Retrieved 2/3 statements.


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



# Parsed testcases at query #10
#--------------------------




import genericpath as module_0

def test_case_0():
    var_0 = 'docs'
    var_1 = module_0.isdir(var_0)
    var_2 = bool(not var_1)
    assert var_2 is True



# Parsed testcases at query #11
#--------------------------




def test_case_0():
    var_0 = False
    var_1 = '.pyi'
    var_2 = bool(not var_1 == '.py')
    assert var_2 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_loader_basic. Retrieved 6/8 statements.
# Partially parsed test_loader_with_toc. Retrieved 5/6 statements.
# Partially parsed test_loader_different_level. Retrieved 7/8 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'path/to/test_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = '# Module `test_pkg`'

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
    var_5 = '<a id='
    var_6 = bool('<a id=' not in var_4)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = 'path/to/test_pkg'
    var_2 = True
    var_3 = 2
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = '## Module `test_pkg`'

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
    var_0 = 'parent_pkg'
    var_1 = 'path/to/parent_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = 'parent_pkg.submodule'
    var_6 = bool('parent_pkg.submodule' in var_4)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'doc_pkg'
    var_1 = 'path/to/doc_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = 'This is a test package'
    var_6 = bool('This is a test package' in var_4)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'const_pkg'
    var_1 = 'path/to/const_pkg'
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
    var_1 = 'path/to/class_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = 'class TestClass'
    var_6 = bool('class TestClass' in var_4)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'func_pkg'
    var_1 = 'path/to/func_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = 'test_function()'
    var_6 = bool('test_function()' in var_4)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'import_pkg'
    var_1 = 'path/to/import_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = 'Imported module'
    var_6 = bool('Imported module' in var_4)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'decorator_pkg'
    var_1 = 'path/to/decorator_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = 'Decorators'
    var_6 = bool('Decorators' in var_4)
    assert var_6 is True
    var_7 = '@staticmethod'
    var_8 = bool('@staticmethod' in var_4)
    assert var_8 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'inherit_pkg'
    var_1 = 'path/to/inherit_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = 'Bases'
    var_6 = bool('Bases' in var_4)
    assert var_6 is True
    var_7 = 'ParentClass'
    var_8 = bool('ParentClass' in var_4)
    assert var_8 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'enum_pkg'
    var_1 = 'path/to/enum_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = 'Enums'
    var_6 = bool('Enums' in var_4)
    assert var_6 is True
    var_7 = 'RED'
    var_8 = bool('RED' in var_4)
    assert var_8 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'member_pkg'
    var_1 = 'path/to/member_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = 'Members'
    var_6 = bool('Members' in var_4)
    assert var_6 is True
    var_7 = 'public_attr'
    var_8 = bool('public_attr' in var_4)
    assert var_8 is True



# Parsed testcases at query #13
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
    var_1 = 'invalid.parent.child'
    var_2 = 'path/to/child.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False



# Parsed testcases at query #14
#--------------------------




import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_path'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is False



# Parsed testcases at query #15
#--------------------------




import genericpath as module_0

def test_case_0():
    var_0 = 'docs'
    var_1 = module_0.isdir(var_0)
    assert var_1 is True



# Parsed testcases at query #16
#--------------------------




import genericpath as module_0

def test_case_0():
    var_0 = 'docs'
    var_1 = module_0.isdir(var_0)
    var_2 = bool(var_1)
    assert var_2 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_loader_basic. Retrieved 6/8 statements.
# Partially parsed test_loader_with_toc. Retrieved 5/7 statements.
# Partially parsed test_loader_no_link. Retrieved 5/6 statements.
# Partially parsed test_loader_different_level. Retrieved 7/9 statements.
# Partially parsed test_loader_empty_package. Retrieved 5/6 statements.
# Partially parsed test_loader_extension_module. Retrieved 5/6 statements.


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
    var_2 = False
    var_3 = 1
    var_4 = module_0.loader(var_0, var_1, var_2, var_3, var_2)
    var_5 = '<a id='
    var_6 = bool('<a id=' not in var_4)
    assert var_6 is True

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
    var_0 = 'empty_pkg'
    var_1 = 'tests/empty_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = bool('no module' in var_4 or 'Missing documentation' in var_4)
    assert var_5 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'ext_pkg'
    var_1 = 'tests/ext_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    var_5 = bool('extension module' in var_4 or 'Missing documentation' in var_4)
    assert var_5 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test__load_module_with_none_spec. Retrieved 4/6 statements.


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = var_0 is not var_1
    var_3 = var_0.loader



# Parsed testcases at query #19
#--------------------------




import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_path.py'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is False



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_pure_py_is_false_when_no_py_file. Retrieved 3/8 statements.


def test_case_0():
    var_0 = False
    var_1 = 'nonexistent_path'
    var_2 = True
    var_3 = bool(not var_2)
    assert var_3 is True



# Parsed testcases at query #21
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
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Test content'
    var_2 = module_0._write(var_0, var_1)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True



# Parsed testcases at query #23
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



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_site_path_returns_empty_string_when_spec_is_none. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = None
    var_1 = 'nonexistent_module'
    var_2 = module_0._site_path(var_1)
    assert var_2 == ''



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_site_path_returns_empty_string_when_spec_is_none. Retrieved 2/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_module'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''



# Parsed testcases at query #26
#--------------------------




def test_case_0():
    var_0 = False
    var_1 = '.pyi'
    var_2 = True
    var_3 = bool(not var_2)
    assert var_3 is True



# Parsed testcases at query #27
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = module_0._read(var_0)
    assert var_1 == 'expected content'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_site_path_returns_empty_string_when_spec_is_none. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = None
    var_1 = 'nonexistent_module'
    var_2 = module_0._site_path(var_1)
    assert var_2 == ''



# Parsed testcases at query #29
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
    var_0 = 'nonexistent_package'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''



# Parsed testcases at query #30
#--------------------------




def test_case_0():
    var_0 = False
    var_1 = '.pyi'
    var_2 = bool(not var_1 == '.py')
    assert var_2 is True



# Parsed testcases at query #31
#--------------------------




import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_path'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    var_4 = bool(not var_3)
    assert var_4 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test__write_creates_file_with_content. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Test content'
    var_2 = module_0._write(var_0, var_1)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True



# Parsed testcases at query #33
#--------------------------




def test_case_0():
    var_0 = '.pyi'
    assert var_0 == '.py'
    var_1 = '.py'
    assert var_1 is False



# Parsed testcases at query #34
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = module_0._read(var_0)
    assert var_1 == 'expected content'



# Parsed testcases at query #35
#--------------------------




import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is True
    var_4 = var_0.docstring
    var_5 = bool(var_0.docstring == {'test_module': 'Test module docstring'})
    assert var_5 is True

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'nonexistent_module'
    var_2 = 'nonexistent_module.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False
    var_4 = var_0.docstring
    var_5 = bool(var_0.docstring == {})
    assert var_5 is True

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'child_module'
    var_2 = 'child_module.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False
    var_4 = var_0.docstring
    var_5 = bool(var_0.docstring == {})
    assert var_5 is True



