####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'pkg'
    var_1 = '/tmp/test'
    var_2 = module_0.walk_packages(var_0, var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_loader_basic_execution. Retrieved 10/15 statements.
# Partially parsed test_loader_extension_module_loading. Retrieved 10/16 statements.
# Partially parsed test_loader_skips_pure_python_modules. Retrieved 9/14 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test loader execution with mocked filesystem and walk_packages.'
    var_1 = 'pkg.sub'
    var_2 = '/path/to/pkg/sub.py'
    var_3 = (var_1, var_2)
    var_4 = '.py'
    var_5 = 'pkg'
    var_6 = '/path/to'
    var_7 = True
    var_8 = False
    var_9 = module_0.loader(var_5, var_6, var_7, var_7, var_8)
    assert var_9 == 'compiled_doc'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test loader when it finds and attempts to load an extension module.'
    var_1 = 'pkg.ext'
    var_2 = '/path/to/pkg/ext'
    var_3 = (var_1, var_2)
    var_4 = '.pyi'
    var_5 = 'pkg'
    var_6 = '/path/to'
    var_7 = True
    var_8 = False
    var_9 = module_0.loader(var_5, var_6, var_7, var_7, var_8)
    assert var_9 == 'extension_doc'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that loader skips extension loading if a .py file is found.'
    var_1 = 'pkg.pure'
    var_2 = '/path/to/pkg/pure'
    var_3 = (var_1, var_2)
    var_4 = 'pkg'
    var_5 = '/path/to'
    var_6 = True
    var_7 = False
    var_8 = module_0.loader(var_4, var_5, var_6, var_6, var_7)
    assert var_8 == 'pure_doc'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_read_success. Retrieved 2/6 statements.
# Partially parsed test_read_empty_file. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'hello world'
    var_1 = 'test.txt'

def test_case_0():
    var_0 = 'empty.txt'
    var_1 = ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existent_file_12345.txt'
    var_1 = module_0._read(var_0)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_gen_api_dry_run_returns_docs_without_writing_files. Retrieved 9/16 statements.
# Partially parsed test_gen_api_writes_file_when_dry_is_false. Retrieved 9/17 statements.
# Partially parsed test_gen_api_with_custom_level_and_prefix. Retrieved 7/12 statements.


import apimd.loader as module_0
import posixpath as module_1

def test_case_0():
    var_0 = 'test_docs_dry'
    var_1 = 'Test Module'
    var_2 = 'os'
    var_3 = {var_1: var_2}
    var_4 = True
    var_5 = module_0.gen_api(var_3, prefix=var_0, dry=var_4)
    var_6 = len(var_5)
    var_7 = bool(var_6 > 0)
    assert var_7 is True
    var_8 = 'Test Module API'
    var_9 = bool('Test Module API' in var_5[0])
    assert var_9 is True
    var_10 = 'os-api.md'
    var_11 = [var_10]
    var_12 = module_1.join(var_0, *var_11)

import apimd.loader as module_0
import posixpath as module_1

def test_case_0():
    var_0 = 'test_docs_write'
    var_1 = 'Test Module'
    var_2 = 'os'
    var_3 = {var_1: var_2}
    var_4 = False
    var_5 = module_0.gen_api(var_3, prefix=var_0, dry=var_4)
    var_6 = len(var_5)
    var_7 = bool(var_6 > 0)
    assert var_7 is True
    var_8 = 'os-api.md'
    var_9 = [var_8]
    var_10 = module_1.join(var_0, *var_9)
    var_11 = 'Test Module API'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_custom_prefix'
    var_1 = 'Test Module'
    var_2 = 'sys'
    var_3 = {var_1: var_2}
    var_4 = 2
    var_5 = True
    var_6 = module_0.gen_api(var_3, prefix=var_0, level=var_4, dry=var_5)
    var_7 = '## Test Module API'
    var_8 = bool('## Test Module API' in var_6[0])
    assert var_8 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_loader_predicate_at_line_9_is_false. Retrieved 7/14 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '/tmp/test_pkg'
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = False
    var_5 = '/tmp'
    var_6 = module_0.loader(var_5, var_5, var_3, var_3, var_3)
    var_7 = bool(not not True)
    assert var_7 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_site_path_returns_empty_string_when_submodule_search_locations_is_none. Retrieved 2/6 statements.
# Partially parsed test_site_path_returns_dirname_of_first_location_when_spec_is_valid. Retrieved 4/9 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existent_package'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'module_with_no_locations'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = '/usr/local/lib/python3.9/site-packages/my_pkg'
    var_1 = '/usr/local/lib/python3.9/site-packages'
    var_2 = 'my_pkg'
    var_3 = module_0._site_path(var_2)
    var_4 = bool(var_3 == var_1)
    assert var_4 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_loader_basic_flow. Retrieved 8/16 statements.
# Partially parsed test_loader_skips_pure_python_modules. Retrieved 9/16 statements.
# Partially parsed test_loader_with_extension_loading_failure. Retrieved 8/16 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'pkg.mod'
    var_1 = '/path/to/pkg/mod'
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = False
    var_5 = 'pkg'
    var_6 = '/path/to/pkg'
    var_7 = module_0.loader(var_5, var_6, var_3, var_3, var_4)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'pkg.mod'
    var_1 = '/path/to/pkg/mod'
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = 'pkg'
    var_5 = '/path/to/pkg'
    var_6 = False
    var_7 = module_0.loader(var_4, var_5, var_3, var_3, var_6)
    var_8 = 'x = 1'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'pkg.mod'
    var_1 = '/path/to/pkg/mod'
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = True
    var_5 = 'pkg'
    var_6 = '/path/to/pkg'
    var_7 = module_0.loader(var_5, var_6, var_4, var_4, var_3)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_loader_with_pure_python_package. Retrieved 9/15 statements.
# Partially parsed test_loader_with_extension_module_success. Retrieved 9/16 statements.
# Partially parsed test_loader_with_extension_module_failure. Retrieved 9/17 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'my_pkg'
    var_1 = '/tmp'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    assert var_4 == '\n'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'my_pkg.sub'
    var_1 = '/tmp/my_pkg/sub.py'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = 'my_pkg'
    var_5 = '/tmp'
    var_6 = True
    var_7 = False
    var_8 = module_0.loader(var_4, var_5, var_6, var_6, var_7)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'my_pkg.ext'
    var_1 = '/tmp/my_pkg/ext'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = 'my_pkg'
    var_5 = '/tmp'
    var_6 = True
    var_7 = False
    var_8 = module_0.loader(var_4, var_5, var_6, var_6, var_7)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'my_pkg.ext'
    var_1 = '/tmp/my_pkg/ext'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = 'my_pkg'
    var_5 = '/tmp'
    var_6 = True
    var_7 = False
    var_8 = module_0.loader(var_4, var_5, var_6, var_6, var_7)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_loader_skips_parse_when_file_does_not_exist. Retrieved 7/15 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '/tmp/test_pkg'
    var_2 = (var_0, var_1)
    var_3 = '/tmp'
    var_4 = False
    var_5 = 1
    var_6 = module_0.loader(var_3, var_3, var_4, var_5, var_4)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_loader_predicate_at_line_9_is_false. Retrieved 8/13 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '/tmp/test_pkg'
    var_2 = (var_0, var_1)
    var_3 = '/tmp'
    var_4 = False
    var_5 = 1
    var_6 = True
    var_7 = module_0.loader(var_3, var_3, var_4, var_5, var_6)
    assert var_7 == 'compiled_result'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_loader_sets_pure_py_true_on_py_extension. Retrieved 11/21 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = '/path/to/test_module'
    var_2 = (var_0, var_1)
    var_3 = '/path/to/test_module.py'
    var_4 = '/root'
    var_5 = '/pwd'
    var_6 = False
    var_7 = 1
    var_8 = module_0.loader(var_4, var_5, var_6, var_7, var_6)
    var_9 = 'content'
    var_10 = 'test_module <= /path/to/test_module.py'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_read_predicate_is_false. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'content'
    var_2 = 'non_existent_file_12345.txt'
    var_3 = bool(not var_1)
    assert var_3 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_loader_pure_py_is_false_when_only_pyi_exists. Retrieved 8/17 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'mypackage'
    var_1 = '/path/mypackage'
    var_2 = (var_0, var_1)
    var_3 = '/path/mypackage.pyi'
    var_4 = '/root'
    var_5 = '/pwd'
    var_6 = True
    var_7 = module_0.loader(var_4, var_5, var_6, var_6, var_6)
    assert var_7 == 'compiled_result'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_loader_line_13_predicate_true. Retrieved 10/16 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'mypackage'
    var_1 = '/path/to/mypackage'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = True
    var_5 = False
    var_6 = '/root'
    var_7 = '/pwd'
    var_8 = module_0.loader(var_6, var_7, var_4, var_4, var_4)
    var_9 = 'content'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_gen_api_empty_input. Retrieved 3/12 statements.
# Partially parsed test_gen_api_dry_run_output_format. Retrieved 6/15 statements.
# Partially parsed test_gen_api_file_creation_logic. Retrieved 6/17 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'docs_test'
    var_2 = True

def test_case_0():
    var_0 = 'MyModule'
    var_1 = 'non_existent_module_xyz'
    var_2 = {var_0: var_1}
    var_3 = 'docs_dry'
    var_4 = True
    var_5 = 2

def test_case_0():
    var_0 = 'OSModule'
    var_1 = 'os'
    var_2 = {var_0: var_1}
    var_3 = 'api_out'
    var_4 = False
    var_5 = 'os-api.md'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_loader_basic_execution. Retrieved 9/17 statements.
# Partially parsed test_loader_extension_module_loading. Retrieved 10/20 statements.
# Partially parsed test_loader_skips_pure_python_modules. Retrieved 10/18 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that loader initializes Parser and calls walk_packages and compile.'
    var_1 = 'mypkg.sub'
    var_2 = '/tmp/mypkg/sub'
    var_3 = (var_1, var_2)
    var_4 = '.py'
    var_5 = 'mypkg'
    var_6 = '/tmp'
    var_7 = True
    var_8 = module_0.loader(var_5, var_6, var_7, var_7, var_7)
    assert var_8 == 'compiled_output'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test the logic when an extension module (e.g. .so, .pyd) is found.'
    var_1 = 'mypkg.ext'
    var_2 = '/tmp/mypkg/ext'
    var_3 = (var_1, var_2)
    var_4 = '.so'
    var_5 = 'mypkg'
    var_6 = '/tmp'
    var_7 = True
    var_8 = False
    var_9 = module_0.loader(var_5, var_6, var_7, var_7, var_8)
    assert var_9 == 'extension_output'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'Test that if a .py file exists, the loader does not attempt to load extension modules.'
    var_1 = 'mypkg.pure'
    var_2 = '/tmp/mypkg/pure'
    var_3 = (var_1, var_2)
    var_4 = '.py'
    var_5 = 'mypkg'
    var_6 = '/tmp'
    var_7 = True
    var_8 = False
    var_9 = module_0.loader(var_5, var_6, var_7, var_7, var_8)
    assert var_9 == 'pure_output'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_loader_pure_py_evaluates_true. Retrieved 9/17 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = '/path/to/test_module'
    var_2 = (var_0, var_1)
    var_3 = '/path/to/test_module.py'
    var_4 = '/root'
    var_5 = '/pwd'
    var_6 = True
    var_7 = module_0.loader(var_4, var_5, var_6, var_6, var_6)
    var_8 = 'content'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_loader_basic_execution. Retrieved 7/18 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'pkg.sub'
    var_1 = '/fake/path/pkg/sub'
    var_2 = (var_0, var_1)
    var_3 = 'pkg'
    var_4 = '/fake/path'
    var_5 = True
    var_6 = module_0.loader(var_3, var_4, var_5, var_5, var_5)
    assert var_6 == 'compiled_output'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_loader_pure_py_true. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = '/path/to/test_module'
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = False



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_loader_pure_py_is_false_when_only_pyi_exists. Retrieved 8/17 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '/path/test_pkg'
    var_2 = (var_0, var_1)
    var_3 = '/path/test_pkg.pyi'
    var_4 = '/root'
    var_5 = '/pwd'
    var_6 = True
    var_7 = module_0.loader(var_4, var_5, var_6, var_6, var_6)
    assert var_7 == 'compiled_result'



# Parsed testcases at query #4
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #5
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = '/root'
    var_1 = '/pwd'
    var_2 = True
    var_3 = module_0.loader(var_0, var_1, var_2, var_2, var_2)
    var_4 = bool(var_3 is not None)
    assert var_4 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_loader_pure_py_evaluation. Retrieved 9/18 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_module'
    var_1 = '/path/to/test_module'
    var_2 = (var_0, var_1)
    var_3 = '/path/to/test_module.py'
    var_4 = '/root'
    var_5 = '/pwd'
    var_6 = True
    var_7 = module_0.loader(var_4, var_5, var_6, var_6, var_6)
    var_8 = 'content'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_gen_api_with_pwd_and_valid_roots. Retrieved 12/23 statements.
# Partially parsed test_gen_api_dry_run_does_not_write_file. Retrieved 14/25 statements.
# Partially parsed test_gen_api_empty_roots. Retrieved 4/8 statements.


import posixpath as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_api_gen_dir'
    var_1 = module_0.abspath(var_0)
    var_2 = True
    var_3 = 'dummy_pkg'
    var_4 = [var_3]
    var_5 = module_0.join(var_1, *var_4)
    var_6 = '"""Dummy Docstring"""\n'
    var_7 = 'Dummy Package'
    var_8 = {var_7: var_3}
    var_9 = 'docs_output'
    var_10 = [var_9]
    var_11 = module_0.join(var_1, *var_10)
    var_12 = True
    var_13 = module_1.gen_api(var_8, var_1, prefix=var_11, link=var_12, level=var_12, toc=var_12)
    var_14 = bool(var_2)
    assert var_14 is True

import posixpath as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_api_dry_run'
    var_1 = module_0.abspath(var_0)
    var_2 = True
    var_3 = 'dry_pkg'
    var_4 = [var_3]
    var_5 = module_0.join(var_1, *var_4)
    var_6 = '"""Dry Run Doc"""\n'
    var_7 = 'docs_dry'
    var_8 = [var_7]
    var_9 = module_0.join(var_1, *var_8)
    var_10 = 'Dry Package'
    var_11 = {var_10: var_3}
    var_12 = True
    var_13 = module_1.gen_api(var_11, var_1, prefix=var_9, dry=var_12)
    var_14 = bool(var_2)
    assert var_14 is True
    var_15 = 'dry-pkg-api.md'
    var_16 = [var_15]
    var_17 = module_0.join(var_9, *var_16)

import apimd.loader as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'empty_test_docs'
    var_2 = module_0.gen_api(var_0, prefix=var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True
    var_4 = 'empty_test_docs'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_loader_line_13_predicate_true. Retrieved 8/14 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '/tmp/test_pkg'
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = False
    var_5 = '/tmp'
    var_6 = module_0.loader(var_5, var_5, var_4, var_3, var_4)
    var_7 = 'content'



# Parsed testcases at query #9
#--------------------------




def test_case_0():
    var_0 = False
    assert var_0 is False



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_loader_skips_parse_when_pyi_extension_is_processed. Retrieved 17/29 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'apimd.loader.walk_packages'
    var_1 = 'mymodule'
    var_2 = '/tmp/mymodule'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = 'apimd.loader.isfile'
    var_6 = '.pyi'
    var_7 = lambda x: x.endswith(var_6)
    var_8 = 'apimd.loader._read'
    var_9 = 'content'
    var_10 = 'apimd.loader.Parser.new'
    var_11 = 'apimd.loader.EXTENSION_SUFFIXES'
    var_12 = []
    var_13 = 'apimd.loader.logger'
    var_14 = '/tmp'
    var_15 = True
    var_16 = module_0.loader(var_14, var_14, var_15, var_15, var_15)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_gen_api_prefix_directory_does_not_exist. Retrieved 6/11 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'title'
    var_1 = 'name'
    var_2 = {var_0: var_1}
    var_3 = 'non_existent_dir'
    var_4 = module_0.gen_api(var_2, prefix=var_3)
    var_5 = 'Create directory: non_existent_dir'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_load_module_success. Retrieved 2/15 statements.
# Partially parsed test_load_module_import_error. Retrieved 2/7 statements.
# Partially parsed test_load_module_no_spec. Retrieved 2/7 statements.
# Partially parsed test_load_module_invalid_loader. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'test_mod'
    var_1 = '/path/to/test_mod.py'

def test_case_0():
    var_0 = 'test_mod'
    var_1 = '/path/to/test_mod.py'

def test_case_0():
    var_0 = 'test_mod'
    var_1 = '/path/to/test_mod.py'

def test_case_0():
    var_0 = 'test_mod'
    var_1 = '/path/to/test_mod.py'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_gen_api_prefix_dir_does_not_exist. Retrieved 6/11 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'title'
    var_1 = 'name'
    var_2 = {var_0: var_1}
    var_3 = 'non_existent_dir'
    var_4 = module_0.gen_api(var_2, prefix=var_3)
    var_5 = 'Create directory: non_existent_dir'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_loader_predicate_false_when_file_exists. Retrieved 8/17 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '/tmp/test_pkg'
    var_2 = (var_0, var_1)
    var_3 = '/tmp'
    var_4 = False
    var_5 = 1
    var_6 = True
    var_7 = module_0.loader(var_3, var_3, var_4, var_5, var_6)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_gen_api_prefix_directory_does_not_exist. Retrieved 5/7 statements.


def test_case_0():
    var_0 = 'Ensure that the predicate at line 18 evaluates to True (prefix does not exist).'
    var_1 = 'non_existent_test_dir_12345'
    var_2 = 'Test'
    var_3 = 'test_module'
    var_4 = {var_2: var_3}



# Parsed testcases at query #16
#--------------------------




def test_case_0():
    var_0 = False
    assert var_0 is False



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_load_module_success_path. Retrieved 1/14 statements.
# Failed to parse test_load_module_predicate_logic_with_mock_loader.


import apimd.parser as module_0

def test_case_0():
    var_0 = module_0.Parser()



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_loader_predicate_false_on_pyi_extension. Retrieved 10/19 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '/root/test_pkg'
    var_2 = (var_0, var_1)
    var_3 = '/root/test_pkg.pyi'
    var_4 = '/root'
    var_5 = False
    var_6 = 1
    var_7 = True
    var_8 = module_0.loader(var_4, var_4, var_5, var_6, var_7)
    var_9 = 'content'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_loader_line_13_evaluates_to_true. Retrieved 8/17 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '/tmp/test_pkg'
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = False
    var_5 = '/tmp'
    var_6 = module_0.loader(var_5, var_5, var_3, var_3, var_3)
    var_7 = 'content'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_read_success. Retrieved 2/6 statements.


def test_case_0():
    var_0 = "print('hello world')"
    var_1 = 'script.py'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existent_file.txt'
    var_1 = module_0._read(var_0)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_loader_predicate_at_line_9_is_false. Retrieved 8/14 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '/path/to/pkg'
    var_2 = (var_0, var_1)
    var_3 = '/root'
    var_4 = '/pwd'
    var_5 = True
    var_6 = module_0.loader(var_3, var_4, var_5, var_5, var_5)
    var_7 = '/path/to/pkg.py'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_gen_api_with_dry_run_does_not_create_files. Retrieved 4/10 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.gen_api(var_0)
    var_2 = bool(var_1 == [])
    assert var_2 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_docs_dry'
    var_1 = {}
    var_2 = True
    var_3 = module_0.gen_api(var_1, prefix=var_0, dry=var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'test_prefix_empty'
    var_2 = module_0.gen_api(var_0, prefix=var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_loader_predicate_is_false_when_file_exists. Retrieved 7/15 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '/tmp/test_pkg'
    var_2 = (var_0, var_1)
    var_3 = '/tmp'
    var_4 = False
    var_5 = 1
    var_6 = module_0.loader(var_3, var_3, var_4, var_5, var_4)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_gen_api_predicate_false_when_doc_is_empty. Retrieved 6/10 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'Title'
    var_1 = 'module_name'
    var_2 = {var_0: var_1}
    var_3 = 'test_prefix'
    var_4 = module_0.gen_api(var_2, prefix=var_3)
    var_5 = "'module_name' can not be found"



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_site_path_returns_empty_string_when_spec_is_none. Retrieved 5/9 statements.
# Partially parsed test_site_path_returns_empty_string_when_submodule_search_locations_is_none. Retrieved 5/10 statements.
# Partially parsed test_site_path_returns_dirname_of_first_location. Retrieved 7/12 statements.
# Partially parsed test_site_path_with_multiple_locations_uses_first_one. Retrieved 8/13 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'builtins.find_spec'
    var_1 = None
    var_2 = lambda name: var_1
    var_3 = 'non_existent_module'
    var_4 = module_0._site_path(var_3)
    assert var_4 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'module'
    var_1 = None
    var_2 = 'builtins.find_spec'
    var_3 = 'module_without_locations'
    var_4 = module_0._site_path(var_3)
    assert var_4 == ''

import posixpath as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = None
    var_2 = '/fake/path'
    var_3 = 'package'
    var_4 = [var_3]
    var_5 = module_0.join(var_2, *var_4)
    var_6 = 'builtins.find_spec'
    var_7 = module_1._site_path(var_0)
    assert var_7 == '/fake/path'

import posixpath as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'module'
    var_1 = None
    var_2 = '/path/one'
    var_3 = []
    var_4 = module_0.join(var_2, *var_3)
    var_5 = '/path/two'
    var_6 = []
    var_7 = module_0.join(var_5, *var_6)
    var_8 = 'builtins.find_spec'
    var_9 = module_1._site_path(var_0)
    assert var_9 == '/path'



