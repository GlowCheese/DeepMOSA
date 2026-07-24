####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_gen_api_with_valid_root_names. Retrieved 11/12 statements.
# Partially parsed test_gen_api_with_multiple_root_names. Retrieved 15/17 statements.
# Partially parsed test_gen_api_with_custom_level. Retrieved 11/12 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = '.'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = module_0.gen_api(var_2, var_3, prefix=var_4, dry=var_5)
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 0
    var_9 = var_6[var_8]
    var_10 = '# test API'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test1'
    var_1 = 'test2'
    var_2 = 'test_module1'
    var_3 = 'test_module2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = '.'
    var_6 = 'test_docs'
    var_7 = True
    var_8 = module_0.gen_api(var_4, var_5, prefix=var_6, dry=var_7)
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = 0
    var_11 = var_8[var_10]
    var_12 = '# test1 API'
    var_13 = var_8[var_7]
    var_14 = '# test2 API'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = 'nonexistent_module'
    var_2 = {var_0: var_1}
    var_3 = '.'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = module_0.gen_api(var_2, var_3, prefix=var_4, dry=var_5)
    var_7 = len(var_6)
    assert var_7 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = {}
    var_1 = '.'
    var_2 = 'test_docs'
    var_3 = True
    var_4 = module_0.gen_api(var_0, var_1, prefix=var_2, dry=var_3)
    var_5 = len(var_4)
    assert var_5 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = '.'
    var_4 = 'custom_prefix'
    var_5 = True
    var_6 = module_0.gen_api(var_2, var_3, prefix=var_4, dry=var_5)
    var_7 = len(var_6)
    assert var_7 == 1

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = '.'
    var_4 = False
    var_5 = True
    var_6 = module_0.gen_api(var_2, var_3, link=var_4, dry=var_5)
    var_7 = len(var_6)
    assert var_7 == 1

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = '.'
    var_4 = 2
    var_5 = True
    var_6 = module_0.gen_api(var_2, var_3, level=var_4, dry=var_5)
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 0
    var_9 = var_6[var_8]
    var_10 = '## test API'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = '.'
    var_4 = True
    var_5 = module_0.gen_api(var_2, var_3, toc=var_4, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 1



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_loader_basic. Retrieved 7/8 statements.
# Partially parsed test_loader_with_toc. Retrieved 6/7 statements.
# Partially parsed test_loader_with_non_existent_package. Retrieved 7/8 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '/tmp'
    var_2 = True
    var_3 = 1
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = len(var_5)
    var_7 = bool(var_6 >= 0)
    assert var_7 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '/tmp'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = '**Table of contents:**'
    var_7 = bool('**Table of contents:**' in var_5)
    assert var_7 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existent_pkg'
    var_1 = '/tmp'
    var_2 = True
    var_3 = 1
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = len(var_5)
    assert var_6 == 0



# Parsed testcases at query #3
#--------------------------




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
    var_11 = bool(var_10 > 0)
    assert var_11 is True

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
    var_10 = len(var_9)
    var_11 = bool(var_10 > 0)
    assert var_11 is True

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
    var_11 = bool(var_10 > 0)
    assert var_11 is True

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



# Parsed testcases at query #4
#--------------------------




import apimd.loader as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'os'
    var_1 = module_0._site_path(var_0)
    var_2 = module_1.isdir(var_1)
    var_3 = bool(var_2)
    assert var_3 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existing_package'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'sys'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_site_path_predicate_evaluates_to_false. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'some_location'
    var_1 = [var_0]



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_loader_pure_py_condition_false. Retrieved 6/36 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'example_root'
    var_1 = 'example_pwd'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_gen_api_basic. Retrieved 9/10 statements.
# Partially parsed test_gen_api_empty_root_names. Retrieved 3/4 statements.
# Partially parsed test_gen_api_multiple_packages. Retrieved 8/9 statements.
# Partially parsed test_gen_api_invalid_package. Retrieved 5/6 statements.
# Partially parsed test_gen_api_custom_prefix. Retrieved 5/6 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_dir'
    var_4 = 'docs'
    var_5 = True
    var_6 = False
    var_7 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_5, toc=var_6, dry=var_5)
    var_8 = len(var_7)
    assert var_8 == 1

import apimd.loader as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.gen_api(var_0)
    var_2 = len(var_1)
    assert var_2 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test1'
    var_1 = 'test2'
    var_2 = 'package1'
    var_3 = 'package2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True
    var_6 = module_0.gen_api(var_4, dry=var_5)
    var_7 = len(var_6)
    assert var_7 == 2

import apimd.loader as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = 'nonexistent_package'
    var_2 = {var_0: var_1}
    var_3 = module_0.gen_api(var_2)
    var_4 = len(var_3)
    assert var_4 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'custom_docs'
    var_4 = module_0.gen_api(var_2, prefix=var_3)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_write. Retrieved 1/9 statements.
# Partially parsed test_write_empty_string. Retrieved 1/9 statements.
# Partially parsed test_write_overwrite_existing_file. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'Hello, world!'

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = 'Initial content'
    var_1 = 'New content'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_gen_api_with_valid_root_names. Retrieved 11/12 statements.
# Partially parsed test_gen_api_with_multiple_packages. Retrieved 14/16 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'test_dir'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = module_0.gen_api(var_2, var_3, prefix=var_4, dry=var_5)
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 0
    var_9 = var_6[var_8]
    var_10 = '# test API'

import apimd.loader as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.gen_api(var_0)
    var_2 = len(var_1)
    assert var_2 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = 'nonexistent_package'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = len(var_4)
    assert var_5 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test1'
    var_1 = 'test2'
    var_2 = 'test_package1'
    var_3 = 'test_package2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'test_dir'
    var_6 = True
    var_7 = module_0.gen_api(var_4, var_5, dry=var_6)
    var_8 = len(var_7)
    assert var_8 == 2
    var_9 = 0
    var_10 = var_7[var_9]
    var_11 = '# test1 API'
    var_12 = var_7[var_6]
    var_13 = '# test2 API'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'custom_docs'
    var_4 = True
    var_5 = module_0.gen_api(var_2, prefix=var_3, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 1



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_load_module_success. Retrieved 3/12 statements.
# Partially parsed test_load_module_invalid_loader. Retrieved 3/9 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'def foo(): pass'
    var_2 = module_0.Parser()
    var_3 = bool(var_0 in var_2.docstring)
    assert var_3 is True

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'non_existent_module'
    var_1 = 'non_existent_path.py'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is False

import apimd.parser as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'def foo(): pass'
    var_2 = module_0.Parser()



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_loader_pure_py_condition_false. Retrieved 14/24 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = '/path/to/test_module'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = '.pyi'
    var_5 = lambda x: x.endswith(var_4)
    var_6 = 'source code'
    var_7 = True
    var_8 = '/root'
    var_9 = '/pwd'
    var_10 = True
    var_11 = module_0.loader(var_8, var_9, var_10, var_10, var_10)
    var_12 = 'test_module'
    var_13 = 'source code'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_gen_api_with_valid_input. Retrieved 11/12 statements.
# Partially parsed test_gen_api_with_none_pwd. Retrieved 10/11 statements.
# Partially parsed test_gen_api_with_invalid_module. Retrieved 11/12 statements.
# Partially parsed test_gen_api_with_dry_run. Retrieved 11/12 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
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
    var_11 = bool(var_10 > 0)
    assert var_11 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
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
    var_0 = 'invalid'
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

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
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
    var_11 = bool(var_10 > 0)
    assert var_11 is True



# Parsed testcases at query #13
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'existing_module'
    var_1 = module_0._site_path(var_0)
    var_2 = bool(var_1 != '')
    assert var_2 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existing_module'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'module_with_none_locations'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_read_file_content. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = module_0._read(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True



# Parsed testcases at query #15
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = '/path/to/test_module'
    var_2 = '/path/to'
    var_3 = '/path/to'
    var_4 = False
    var_5 = 1
    var_6 = False
    var_7 = module_0.loader(var_2, var_3, var_4, var_5, var_6)



# Parsed testcases at query #16
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = module_0._site_path(var_0)
    var_2 = bool(var_1 != '')
    assert var_2 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_module'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''



# Parsed testcases at query #17
#--------------------------




def test_case_0():
    var_0 = 'MockSpec'
    var_1 = ()
    var_2 = 'submodule_search_locations'
    var_3 = '/some/path'
    var_4 = [var_3]
    var_5 = {var_2: var_4}
    var_6 = type(var_0, var_1, var_5)
    var_7 = var_6()
    var_8 = bool(not (var_7 is None or var_7.submodule_search_locations is None))
    assert var_8 is True

def test_case_0():
    var_0 = None
    var_1 = var_0 is None or var_0.submodule_search_locations is None
    assert var_1 is True

def test_case_0():
    var_0 = 'MockSpec'
    var_1 = ()
    var_2 = 'submodule_search_locations'
    var_3 = None
    var_4 = {var_2: var_3}
    var_5 = type(var_0, var_1, var_4)
    var_6 = var_5()
    var_7 = var_6 is None or var_6.submodule_search_locations is None
    assert var_7 is True



# Parsed testcases at query #18
#--------------------------




def test_case_0():
    var_0 = '.pyi'
    var_1 = bool(var_0 != '.py')
    assert var_1 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_gen_api_dry_run. Retrieved 8/10 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = module_0.gen_api(var_2, dry=var_3)
    var_5 = len(var_4)
    var_6 = bool(var_5 > 0)
    assert var_6 is True
    var_7 = 0
    var_8 = var_4[var_7]



# Parsed testcases at query #20
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_file.txt'
    var_1 = module_0._read(var_0)
    assert var_1 is None



# Parsed testcases at query #21
#--------------------------




def test_case_0():
    var_0 = '.pyi'
    var_1 = '.py'
    var_2 = var_0 == var_1
    assert var_2 is False



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_loader_skips_pure_python_modules. Retrieved 5/33 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'root'
    var_1 = 'pwd'
    var_2 = False
    var_3 = 1
    var_4 = module_0.loader(var_0, var_1, var_2, var_3, var_2)
    assert var_4 == 'compiled_output'



# Parsed testcases at query #23
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



# Parsed testcases at query #24
#--------------------------




def test_case_0():
    var_0 = 'non_existent_directory/non_existent_file.txt'
    var_1 = 'test content'
    var_2 = 'w+'
    var_3 = 'utf-8'
    var_4 = open(var_0, var_2, encoding=var_3)
    var_5 = bool(var_4 is not None)
    assert var_5 is True



# Parsed testcases at query #25
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = '/path/to/nonexistent/file'
    var_1 = module_0._read(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_write_file. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = module_0._write(var_0, var_1)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True



# Parsed testcases at query #27
#--------------------------




import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_path'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is False



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_read_file_successfully. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'test content'



# Parsed testcases at query #29
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = '/nonexistent_directory/test.txt'
    var_1 = 'test content'
    var_2 = module_0._write(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_load_module_predicate_evaluates_to_true. Retrieved 4/11 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = []
    var_1 = 'test_module'
    var_2 = '/path/to/module'
    var_3 = module_0.Parser()
    var_4 = None



# Parsed testcases at query #31
#--------------------------

# Partially parsed test__load_module_success. Retrieved 5/16 statements.
# Partially parsed test__load_module_loader_failure. Retrieved 4/9 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = '/path/to/test_module.py'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is True
    var_4 = module_0.parent(var_0)

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = '/path/to/test_module.py'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = '/path/to/test_module.py'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = '/path/to/test_module.py'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is False



# Parsed testcases at query #32
#--------------------------




def test_case_0():
    var_0 = 'non_existent_file.txt'
    var_1 = 'r'
    var_2 = open(var_0, var_1)
    var_3 = bool(not var_2)
    assert var_3 is True



# Parsed testcases at query #33
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = ''
    var_1 = 'test'
    var_2 = module_0._write(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = None
    var_1 = 'test'
    var_2 = module_0._write(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test.txt'
    var_1 = None
    var_2 = module_0._write(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_write_file_creates_and_writes_content. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_load_module_successful. Retrieved 2/5 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'os'

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
    var_1 = 'invalid_spec'
    var_2 = '/path/to/invalid_spec.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_load_module_with_valid_spec_and_loader. Retrieved 3/12 statements.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test.module'
    var_2 = '/path/to/module'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_predicate_at_line_3_evaluates_to_true. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'test content'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #38
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existent_file.txt'
    var_1 = module_0._read(var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #39
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = '/non/existent/directory/file.txt'
    var_1 = 'test content'
    var_2 = module_0._write(var_0, var_1)
    var_3 = bool(not var_2)
    assert var_3 is True



# Parsed testcases at query #40
#--------------------------




import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = None
    var_1 = 'some/path'
    var_2 = 'some.module'
    var_3 = module_0.Parser()
    var_4 = module_1._load_module(var_2, var_1, var_3)
    assert var_4 is False



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_loader_with_valid_package. Retrieved 6/7 statements.
# Partially parsed test_loader_with_invalid_package. Retrieved 6/7 statements.
# Partially parsed test_loader_with_empty_package. Retrieved 6/7 statements.
# Partially parsed test_loader_with_different_levels. Retrieved 6/7 statements.
# Partially parsed test_loader_with_no_toc. Retrieved 6/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '/tmp'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'invalid_pkg'
    var_1 = '/tmp'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)

import apimd.loader as module_0

def test_case_0():
    var_0 = ''
    var_1 = '/tmp'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '/tmp'
    var_2 = True
    var_3 = 2
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '/tmp'
    var_2 = True
    var_3 = 1
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_loader. Retrieved 6/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'example_pkg'
    var_1 = '/path/to/package'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = 'Module `example_pkg`'
    var_7 = bool('Module `example_pkg`' in var_5)
    assert var_7 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_loader_with_valid_package. Retrieved 7/8 statements.
# Partially parsed test_loader_with_invalid_package. Retrieved 7/8 statements.
# Partially parsed test_loader_with_pure_py_package. Retrieved 7/8 statements.
# Partially parsed test_loader_with_extension_module. Retrieved 7/8 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'example_pkg'
    var_1 = '/path/to/package'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = len(var_5)
    var_7 = bool(var_6 > 0)
    assert var_7 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_pkg'
    var_1 = '/invalid/path'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = len(var_5)
    assert var_6 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'pure_py_pkg'
    var_1 = '/path/to/pure_py_package'
    var_2 = False
    var_3 = 2
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = len(var_5)
    var_7 = bool(var_6 > 0)
    assert var_7 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'ext_module_pkg'
    var_1 = '/path/to/extension_module'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = len(var_5)
    var_7 = bool(var_6 > 0)
    assert var_7 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_loader_should_not_skip_extension_module_when_pure_py_is_false. Retrieved 5/32 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_root'
    var_1 = 'test_pwd'
    var_2 = False
    var_3 = 1
    var_4 = module_0.loader(var_0, var_1, var_2, var_3, var_2)
    assert var_4 == 'mock_output'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_gen_api_with_valid_input. Retrieved 11/12 statements.
# Partially parsed test_gen_api_with_none_pwd. Retrieved 6/7 statements.
# Partially parsed test_gen_api_with_dry_run. Retrieved 6/7 statements.
# Partially parsed test_gen_api_with_invalid_package. Retrieved 5/6 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = '/path/to/test_package'
    var_4 = 'docs'
    var_5 = True
    var_6 = 1
    var_7 = False
    var_8 = False
    var_9 = module_0.gen_api(var_2, var_3, prefix=var_4, link=var_5, level=var_6, toc=var_7, dry=var_8)
    var_10 = len(var_9)
    var_11 = bool(var_10 > 0)
    assert var_11 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = 'docs'
    var_4 = None
    var_5 = module_0.gen_api(var_2, var_4, prefix=var_3)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_package'
    var_2 = {var_0: var_1}
    var_3 = '/path/to/test_package'
    var_4 = True
    var_5 = module_0.gen_api(var_2, var_3, dry=var_4)

import apimd.loader as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.gen_api(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = 'nonexistent_package'
    var_2 = {var_0: var_1}
    var_3 = module_0.gen_api(var_2)
    var_4 = len(var_3)
    assert var_4 == 0



# Parsed testcases at query #6
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'example_root'
    var_1 = 'example_pwd'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)



# Parsed testcases at query #7
#--------------------------




def test_case_0():
    var_0 = 'test_module'
    var_1 = '/path/to/test_module'
    var_2 = '.pyi'
    var_3 = False
    var_4 = True
    var_5 = bool(not var_4)
    assert var_5 is True



# Parsed testcases at query #8
#--------------------------




import apimd.loader as module_0
import genericpath as module_1

def test_case_0():
    var_0 = 'example'
    var_1 = 'example_module'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = module_0.gen_api(var_2, prefix=var_3)
    var_5 = module_1.isdir(var_3)
    var_6 = bool(var_5)
    assert var_6 is True



# Parsed testcases at query #9
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = 'module.submodule.class'
    var_1 = module_0.parent(var_0)
    assert var_1 == 'module.submodule'
    var_2 = 2
    var_3 = module_0.parent(var_0, level=var_2)
    assert var_3 == 'module'
    var_4 = 'single'
    var_5 = module_0.parent(var_4)
    assert var_5 == 'single'
    var_6 = 'a.b.c.d.e'
    var_7 = 3
    var_8 = module_0.parent(var_6, level=var_7)
    assert var_8 == 'a.b'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test__load_module_success. Retrieved 4/12 statements.
# Partially parsed test__load_module_loader_not_instance. Retrieved 4/7 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = '/path/to/module.py'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is True

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = '/path/to/module.py'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = '/path/to/module.py'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = '/path/to/module.py'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is False



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_loader_with_python_file. Retrieved 5/12 statements.
# Partially parsed test_loader_with_pyi_file. Retrieved 5/12 statements.
# Partially parsed test_loader_with_both_py_and_pyi. Retrieved 7/17 statements.
# Partially parsed test_loader_with_non_python_file. Retrieved 5/12 statements.
# Partially parsed test_loader_with_toc_enabled. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'test_module.py'
    var_1 = 'def test_func():\n    pass\n'
    var_2 = 'test_module'
    var_3 = True
    var_4 = False
    var_5 = 'test_func'

def test_case_0():
    var_0 = 'test_module.pyi'
    var_1 = 'def test_func() -> None: ...\n'
    var_2 = 'test_module'
    var_3 = True
    var_4 = False
    var_5 = 'test_func'

def test_case_0():
    var_0 = 'test_module.py'
    var_1 = 'test_module.pyi'
    var_2 = 'def test_func():\n    pass\n'
    var_3 = 'def test_func() -> None: ...\n'
    var_4 = 'test_module'
    var_5 = True
    var_6 = False
    var_7 = 'test_func'

def test_case_0():
    var_0 = 'test_module.txt'
    var_1 = 'This is not a Python file'
    var_2 = 'test_module'
    var_3 = True
    var_4 = False
    var_5 = 'test_func'

def test_case_0():
    var_0 = 'test_module.py'
    var_1 = 'def test_func():\n    pass\n'
    var_2 = 'test_module'
    var_3 = True
    var_4 = '**Table of contents:**'
    var_5 = '+ [test_func'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_write_file_success. Retrieved 3/5 statements.
# Partially parsed test_write_file_empty_string. Retrieved 3/5 statements.
# Partially parsed test_write_file_overwrite_existing_content. Retrieved 5/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = module_0._write(var_0, var_1)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = ''
    var_2 = module_0._write(var_0, var_1)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Initial content'
    var_2 = 'New content'
    var_3 = module_0._write(var_0, var_1)
    var_4 = module_0._write(var_0, var_2)



# Parsed testcases at query #13
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'root_dir'
    var_1 = 'current_dir'
    var_2 = False
    var_3 = 1
    var_4 = module_0.loader(var_0, var_1, var_2, var_3, var_2)



# Parsed testcases at query #14
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'example_root'
    var_1 = 'example_pwd'
    var_2 = False
    var_3 = 1
    var_4 = False
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = bool(var_5 is not None)
    assert var_6 is True



# Parsed testcases at query #15
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'example_root'
    var_1 = 'example_pwd'
    var_2 = True
    var_3 = 2
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    var_6 = 'loading extension module for fully documented:'
    var_7 = bool('loading extension module for fully documented:' in var_5)
    assert var_7 is True



# Parsed testcases at query #16
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = 'module.submodule.class.method'
    var_1 = 1
    var_2 = module_0.parent(var_0, level=var_1)
    assert var_2 == 'module.submodule.class'
    var_3 = 2
    var_4 = module_0.parent(var_0, level=var_3)
    assert var_4 == 'module.submodule'
    var_5 = 3
    var_6 = module_0.parent(var_0, level=var_5)
    assert var_6 == 'module'
    var_7 = 'single'
    var_8 = module_0.parent(var_7, level=var_1)
    assert var_8 == 'single'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_loader_skips_pure_python_modules. Retrieved 6/33 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_root'
    var_1 = 'test_pwd'
    var_2 = True
    var_3 = 1
    var_4 = True
    var_5 = module_0.loader(var_0, var_1, var_2, var_3, var_4)
    assert var_5 == 'compiled'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_site_path_with_existing_package. Retrieved 6/15 statements.
# Partially parsed test_site_path_with_non_existing_package. Retrieved 2/5 statements.
# Partially parsed test_site_path_with_package_without_submodule_search_locations. Retrieved 2/7 statements.


import apimd.loader as module_0
import importlib.util as module_1
import posixpath as module_2

def test_case_0():
    var_0 = 'some_existing_package'
    var_1 = module_0._site_path(var_0)
    var_2 = 0
    var_3 = module_1.find_spec(var_0)
    var_4 = var_3.submodule_search_locations[var_2]
    var_5 = module_2.dirname(var_4)
    var_6 = bool(var_1 == var_5)
    assert var_6 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existing_package'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'package_without_submodule_search_locations'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_write_file_content. Retrieved 3/5 statements.
# Partially parsed test_write_file_overwrite. Retrieved 5/7 statements.
# Partially parsed test_write_file_empty_string. Retrieved 3/5 statements.
# Partially parsed test_write_file_special_characters. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
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
    var_1 = ''
    var_2 = module_0._write(var_0, var_1)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'こんにちは, мир!'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #20
#--------------------------




import importlib.util as module_0
import posixpath as module_1
import apimd.loader as module_2

def test_case_0():
    var_0 = 'sys'
    var_1 = module_0.find_spec(var_0)
    var_2 = var_1.submodule_search_locations
    var_3 = var_1 and var_2
    var_4 = 0
    var_5 = var_1.submodule_search_locations[var_4]
    var_6 = module_1.dirname(var_5)
    var_7 = ''
    var_8 = var_6 if var_3 else var_7
    var_9 = module_2._site_path(var_0)
    var_10 = bool(var_9 == var_8)
    assert var_10 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'nonexistent_module'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'builtins'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_read_file_content. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'This is a test file content.'
    var_2 = module_0._read(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'empty_file.txt'
    var_1 = module_0._read(var_0)
    assert var_1 == ''



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 3/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'test content'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_site_path_with_submodule_search_locations. Retrieved 1/3 statements.


def test_case_0():
    var_0 = '/path/to/module'
    var_1 = [var_0]



# Parsed testcases at query #24
#--------------------------

# Partially parsed test__site_path_exists. Retrieved 3/4 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'os'
    var_1 = module_0._site_path(var_0)
    var_2 = 'lib/python3.x/site-packages'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existent_module'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'builtins'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_site_path_with_submodule_search_locations. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'some/path'
    var_1 = [var_0]



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_read_existing_file. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = module_0._read(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'empty_file.txt'
    var_1 = module_0._read(var_0)
    assert var_1 == ''



# Parsed testcases at query #27
#--------------------------




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
    var_11 = bool(var_10 > 0)
    assert var_11 is True

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
    var_11 = bool(var_10 > 0)
    assert var_11 is True

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
    var_11 = bool(var_10 > 0)
    assert var_11 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'NonExistentModule'
    var_1 = 'non_existent_module'
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



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_site_path_with_valid_module. Retrieved 3/9 statements.
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



# Parsed testcases at query #29
#--------------------------




import apimd.parser as module_0
import _frozen_importlib_external as module_1
import apimd.loader as module_2

def test_case_0():
    var_0 = 'test_module'
    var_1 = '/path/to/test_module.py'
    var_2 = module_0.Parser()
    var_3 = module_1.spec_from_file_location(var_0, var_1)
    var_4 = module_2._load_module(var_0, var_1, var_2)
    assert var_4 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_load_module_returns_false_when_loader_is_not_instance_of_loader. Retrieved 7/12 statements.


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
    var_0 = 'Spec'
    var_1 = ()
    var_2 = 'loader'
    var_3 = 'test_module'
    var_4 = 'test_path'
    var_5 = module_0.Parser()
    var_6 = module_1._load_module(var_3, var_4, var_5)
    assert var_6 is False



# Parsed testcases at query #31
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = 'module.submodule.class'
    var_1 = module_0.parent(var_0)
    assert var_1 == 'module.submodule'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module.submodule.class'
    var_1 = 2
    var_2 = module_0.parent(var_0, level=var_1)
    assert var_2 == 'module'

import apimd.parser as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = module_0.parent(var_0)
    assert var_1 == 'module'

import apimd.parser as module_0

def test_case_0():
    var_0 = ''
    var_1 = module_0.parent(var_0)
    assert var_1 == ''



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_predicate_at_line_3_evaluates_to_true. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'test content'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #33
#--------------------------




import apimd.parser as module_0

def test_case_0():
    var_0 = 'module.submodule.class.method'
    var_1 = 1
    var_2 = module_0.parent(var_0, level=var_1)
    assert var_2 == 'module.submodule.class'
    var_3 = 2
    var_4 = module_0.parent(var_0, level=var_3)
    assert var_4 == 'module.submodule'
    var_5 = 3
    var_6 = module_0.parent(var_0, level=var_5)
    assert var_6 == 'module'
    var_7 = 'single'
    var_8 = module_0.parent(var_7, level=var_1)
    assert var_8 == 'single'
    var_9 = 'a.b'
    var_10 = module_0.parent(var_9, level=var_1)
    assert var_10 == 'a'



# Parsed testcases at query #34
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existent_file.txt'
    var_1 = module_0._read(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_read_file_successfully. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'test content'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_predicate_at_line_3_evaluates_to_true. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'test content'



# Parsed testcases at query #37
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'existing_file.txt'
    var_1 = module_0._read(var_0)
    assert var_1 == 'expected content'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existing_file.txt'
    var_1 = module_0._read(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'empty_file.txt'
    var_1 = module_0._read(var_0)
    assert var_1 == ''



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_gen_api_with_valid_root_names. Retrieved 7/8 statements.
# Partially parsed test_gen_api_with_none_pwd. Retrieved 6/7 statements.
# Partially parsed test_gen_api_with_invalid_root_names. Retrieved 7/8 statements.
# Partially parsed test_gen_api_with_empty_root_names. Retrieved 6/7 statements.
# Partially parsed test_gen_api_with_non_existent_prefix. Retrieved 7/8 statements.
# Partially parsed test_gen_api_with_dry_run. Retrieved 7/8 statements.
# Partially parsed test_gen_api_with_link_disabled. Retrieved 8/9 statements.
# Partially parsed test_gen_api_with_toc_enabled. Retrieved 7/8 statements.
# Partially parsed test_gen_api_with_custom_level. Retrieved 8/9 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = {var_0: var_0}
    var_2 = 'test_path'
    var_3 = 'test_docs'
    var_4 = True
    var_5 = module_0.gen_api(var_1, var_2, prefix=var_3, dry=var_4)
    var_6 = len(var_5)
    var_7 = bool(var_6 > 0)
    assert var_7 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = {var_0: var_0}
    var_2 = 'test_docs'
    var_3 = None
    var_4 = True
    var_5 = module_0.gen_api(var_1, var_3, prefix=var_2, dry=var_4)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'invalid_module'
    var_1 = {var_0: var_0}
    var_2 = 'test_path'
    var_3 = 'test_docs'
    var_4 = True
    var_5 = module_0.gen_api(var_1, var_2, prefix=var_3, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'test_path'
    var_2 = 'test_docs'
    var_3 = True
    var_4 = module_0.gen_api(var_0, var_1, prefix=var_2, dry=var_3)
    var_5 = len(var_4)
    assert var_5 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = {var_0: var_0}
    var_2 = 'test_path'
    var_3 = 'non_existent_dir'
    var_4 = True
    var_5 = module_0.gen_api(var_1, var_2, prefix=var_3, dry=var_4)
    var_6 = len(var_5)
    var_7 = bool(var_6 > 0)
    assert var_7 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = {var_0: var_0}
    var_2 = 'test_path'
    var_3 = 'test_docs'
    var_4 = True
    var_5 = module_0.gen_api(var_1, var_2, prefix=var_3, dry=var_4)
    var_6 = len(var_5)
    var_7 = bool(var_6 > 0)
    assert var_7 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = {var_0: var_0}
    var_2 = 'test_path'
    var_3 = 'test_docs'
    var_4 = False
    var_5 = True
    var_6 = module_0.gen_api(var_1, var_2, prefix=var_3, link=var_4, dry=var_5)
    var_7 = len(var_6)
    var_8 = bool(var_7 > 0)
    assert var_8 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = {var_0: var_0}
    var_2 = 'test_path'
    var_3 = 'test_docs'
    var_4 = True
    var_5 = module_0.gen_api(var_1, var_2, prefix=var_3, toc=var_4, dry=var_4)
    var_6 = len(var_5)
    var_7 = bool(var_6 > 0)
    assert var_7 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = {var_0: var_0}
    var_2 = 'test_path'
    var_3 = 'test_docs'
    var_4 = 2
    var_5 = True
    var_6 = module_0.gen_api(var_1, var_2, prefix=var_3, level=var_4, dry=var_5)
    var_7 = len(var_6)
    var_8 = bool(var_7 > 0)
    assert var_8 is True



# Parsed testcases at query #39
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = '/nonexistent_directory/test.txt'
    var_1 = 'test content'
    var_2 = module_0._write(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_write_file_successfully. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_gen_api_with_valid_root_names. Retrieved 10/11 statements.
# Partially parsed test_gen_api_with_custom_level. Retrieved 11/12 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = True
    var_5 = module_0.gen_api(var_2, prefix=var_3, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 1
    var_7 = 0
    var_8 = var_5[var_7]
    var_9 = '# test API'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = 'non_existent_module'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = True
    var_5 = module_0.gen_api(var_2, prefix=var_3, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 0

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test1'
    var_1 = 'test2'
    var_2 = 'test_module1'
    var_3 = 'test_module2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'test_docs'
    var_6 = True
    var_7 = module_0.gen_api(var_4, prefix=var_5, dry=var_6)
    var_8 = len(var_7)
    assert var_8 == 2

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = '/tmp'
    var_4 = 'test_docs'
    var_5 = True
    var_6 = module_0.gen_api(var_2, var_3, prefix=var_4, dry=var_5)
    var_7 = len(var_6)
    assert var_7 == 1

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
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = 'custom_prefix'
    var_4 = True
    var_5 = module_0.gen_api(var_2, prefix=var_3, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 1

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = False
    var_5 = True
    var_6 = module_0.gen_api(var_2, prefix=var_3, link=var_4, dry=var_5)
    var_7 = len(var_6)
    assert var_7 == 1

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = 2
    var_5 = True
    var_6 = module_0.gen_api(var_2, prefix=var_3, level=var_4, dry=var_5)
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = 0
    var_9 = var_6[var_8]
    var_10 = '## test API'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_module'
    var_2 = {var_0: var_1}
    var_3 = 'test_docs'
    var_4 = True
    var_5 = module_0.gen_api(var_2, prefix=var_3, toc=var_4, dry=var_4)
    var_6 = len(var_5)
    assert var_6 == 1



# Parsed testcases at query #42
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = '/non/existent/directory/file.txt'
    var_1 = 'sample text'
    var_2 = module_0._write(var_0, var_1)
    var_3 = bool(True)
    assert var_3 is True
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_write_file_content. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'Hello, World!'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_load_module_predicate_evaluates_to_true. Retrieved 7/9 statements.


import apimd.parser as module_0
import _frozen_importlib_external as module_1

def test_case_0():
    var_0 = 'test_module'
    var_1 = '/path/to/test_module.py'
    var_2 = module_0.Parser()
    var_3 = module_1.spec_from_file_location(var_0, var_1)
    var_4 = None
    var_5 = var_3 is not var_4
    var_6 = var_3.loader



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_read_file_content. Retrieved 3/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = module_0._read(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True



