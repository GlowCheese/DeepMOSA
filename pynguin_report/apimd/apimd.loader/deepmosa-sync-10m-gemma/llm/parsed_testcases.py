####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'pkg'
    var_1 = '/tmp'
    var_2 = module_0.walk_packages(var_0, var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'pkg'
    var_1 = '/tmp'
    var_2 = module_0.walk_packages(var_0, var_1)
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = var_3[0][0]
    assert var_5 == 'module'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'pkg'
    var_1 = '/tmp'
    var_2 = module_0.walk_packages(var_0, var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'pkg'
    var_1 = '/tmp'
    var_2 = module_0.walk_packages(var_0, var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'pkg'
    var_1 = '/tmp'
    var_2 = module_0.walk_packages(var_0, var_1)
    var_3 = list(var_2)



# Parsed testcases at query #2
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_loader_empty_package. Retrieved 5/7 statements.
# Partially parsed test_loader_with_python_files_skips_extension_loading. Retrieved 8/15 statements.
# Partially parsed test_loader_with_extension_module_triggers_load_module. Retrieved 8/15 statements.
# Partially parsed test_loader_handles_failed_extension_loading. Retrieved 8/15 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'my_pkg'
    var_1 = '/tmp'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    assert var_4 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'my_pkg.module'
    var_1 = '/tmp/my_pkg/module.py'
    var_2 = (var_0, var_1)
    var_3 = 'my_pkg'
    var_4 = '/tmp'
    var_5 = True
    var_6 = False
    var_7 = module_0.loader(var_3, var_4, var_5, var_5, var_6)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'my_pkg.ext'
    var_1 = '/tmp/my_pkg/ext.pyi'
    var_2 = (var_0, var_1)
    var_3 = 'my_pkg'
    var_4 = '/tmp'
    var_5 = True
    var_6 = False
    var_7 = module_0.loader(var_3, var_4, var_5, var_5, var_6)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'my_pkg.ext'
    var_1 = '/tmp/my_pkg/ext.pyi'
    var_2 = (var_0, var_1)
    var_3 = 'my_pkg'
    var_4 = '/tmp'
    var_5 = True
    var_6 = False
    var_7 = module_0.loader(var_3, var_4, var_5, var_5, var_6)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_site_path_evaluates_to_true_when_submodule_search_locations_is_none. Retrieved 2/6 statements.


def test_case_0():
    pass

import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existent_package'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'module_without_locations'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_walk_packages_predicate_evaluates_to_true. Retrieved 11/19 statements.


import posixpath as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = '/tmp/test_package'
    var_1 = module_0.abspath(var_0)
    var_2 = 'test_pkg'
    var_3 = 'module.py'
    var_4 = []
    var_5 = [var_3]
    var_6 = 'test_pkg'
    var_7 = '/tmp/test_package'
    var_8 = module_1.walk_packages(var_6, var_7)
    var_9 = list(var_8)
    var_10 = len(var_9)
    var_11 = bool(var_10 > 0)
    assert var_11 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_loader_basic_execution. Retrieved 5/20 statements.
# Partially parsed test_loader_with_toc. Retrieved 4/18 statements.
# Partially parsed test_loader_empty_package. Retrieved 5/20 statements.


def test_case_0():
    var_0 = 'test_pkg'
    var_1 = '__init__.py'
    var_2 = 'x = 1\n'
    var_3 = True
    var_4 = False
    var_5 = '# Module `test_pkg`'

def test_case_0():
    var_0 = 'test_pkg_toc'
    var_1 = '__init__.py'
    var_2 = 'Y = 2\n'
    var_3 = True
    var_4 = '**Table of contents:**'
    var_5 = '+ [test_pkg_toc](#test-pkg-toc)'

def test_case_0():
    var_0 = 'empty_pkg'
    var_1 = '__init__.py'
    var_2 = ''
    var_3 = False
    var_4 = 1



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_load_module_success. Retrieved 4/11 statements.
# Partially parsed test_load_module_invalid_loader. Retrieved 4/7 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.mod'
    var_2 = '/path/to/mod.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is True

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.mod'
    var_2 = '/path/to/mod.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.mod'
    var_2 = '/path/to/mod.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.mod'
    var_2 = '/path/to/mod.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_load_module_success. Retrieved 5/13 statements.
# Partially parsed test_load_module_invalid_loader. Retrieved 4/9 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module'
    var_3 = '/path/to/test_module.py'
    var_4 = module_1._load_module(var_2, var_3, var_0)
    assert var_4 is True

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'non_existent_package.module'
    var_2 = '/path/to/file.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '/path/to/test_module.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = '/path/to/test_module.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_site_path_returns_empty_string_when_submodule_search_locations_is_none. Retrieved 2/6 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existent_package'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'package_without_locations'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_walk_packages_predicate_evaluates_to_true. Retrieved 9/13 statements.


def test_case_0():
    var_0 = 'module.py'
    var_1 = '/tmp/dummy_path'
    var_2 = []
    var_3 = [var_0]
    var_4 = (var_1, var_2, var_3)
    assert var_4 is True
    var_5 = [var_4]
    var_6 = '.py'
    var_7 = '.pyi'
    var_8 = (var_6, var_7)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_load_module_success. Retrieved 3/15 statements.
# Partially parsed test_load_module_import_error. Retrieved 2/7 statements.
# Partially parsed test_load_module_failed_spec. Retrieved 2/7 statements.
# Partially parsed test_load_module_invalid_loader. Retrieved 3/12 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = 'test_module'
    var_2 = '/path/to/script.py'

def test_case_0():
    var_0 = 'non_existent'
    var_1 = '/path/to/script.py'

def test_case_0():
    var_0 = 'test_module'
    var_1 = '/path/to/script.py'

def test_case_0():
    var_0 = 'test_module'
    var_1 = []
    var_2 = {}
    var_3 = 'test_module'
    var_4 = '/path/to/script.py'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/7 statements.
# Partially parsed test_write_overwrites_existing_file. Retrieved 4/10 statements.
# Partially parsed test_write_handles_empty_string. Retrieved 3/7 statements.
# Partially parsed test_write_handles_special_characters. Retrieved 3/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_output.txt'
    var_1 = 'Hello, World!'
    var_2 = module_0._write(var_0, var_1)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_overwrite.txt'
    var_1 = 'Old Content'
    var_2 = 'New Content'
    var_3 = module_0._write(var_0, var_2)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_empty.txt'
    var_1 = ''
    var_2 = module_0._write(var_0, var_1)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_unicode.txt'
    var_1 = '🔥 Unicode Test: 漢字, 😊'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_write_success. Retrieved 3/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_walk_packages_predicate_evaluates_to_true. Retrieved 8/14 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_module.py'
    var_1 = 'test_package'
    var_2 = '/mock/path'
    var_3 = module_0.walk_packages(var_1, var_2)
    var_4 = next(var_3)
    var_5 = '.py'
    var_6 = '.pyi'
    var_7 = (var_5, var_6)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_write_success. Retrieved 3/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_loader_returns_string. Retrieved 7/14 statements.
# Partially parsed test_loader_skips_pure_py. Retrieved 7/16 statements.
# Partially parsed test_loader_handles_empty_package. Retrieved 5/10 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'pkg'
    var_1 = '/path/to/pkg'
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = False
    var_5 = '/path/to'
    var_6 = module_0.loader(var_0, var_5, var_3, var_3, var_3)
    assert var_6 == 'compiled_doc'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'pkg'
    var_1 = '/path/to/pkg'
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = '/path/to'
    var_5 = False
    var_6 = module_0.loader(var_0, var_4, var_3, var_3, var_5)
    assert var_6 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'pkg'
    var_1 = '/path/to'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    assert var_4 == 'empty'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_loader_integration_with_mocked_filesystem. Retrieved 8/14 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'my_pkg'
    var_1 = '/tmp/my_pkg'
    var_2 = (var_0, var_1)
    var_3 = '.py'
    var_4 = '/tmp'
    var_5 = True
    var_6 = False
    var_7 = module_0.loader(var_0, var_4, var_5, var_5, var_6)
    var_8 = '# Module `my_pkg`'
    var_9 = bool('# Module `my_pkg`' in var_7)
    assert var_9 is True
    var_10 = 'func()'
    var_11 = bool('func()' in var_7)
    assert var_11 is True
    var_12 = 'Docstring'
    var_13 = bool('Docstring' in var_7)
    assert var_13 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_loader_skip_parsing_if_file_not_exists. Retrieved 6/14 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'pkg'
    var_1 = '/root/pkg'
    var_2 = (var_0, var_1)
    var_3 = '/root'
    var_4 = True
    var_5 = module_0.loader(var_3, var_3, var_4, var_4, var_4)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_loader_empty_package. Retrieved 5/7 statements.
# Partially parsed test_loader_skips_pure_python_files. Retrieved 9/15 statements.
# Partially parsed test_loader_processes_extension_modules. Retrieved 8/21 statements.
# Partially parsed test_loader_handles_load_failure. Retrieved 8/19 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'my_pkg'
    var_1 = '/tmp/path'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    assert var_4 == '\n'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'my_pkg.module'
    var_1 = '/tmp/path/my_pkg/module'
    var_2 = (var_0, var_1)
    var_3 = '.py'
    var_4 = 'my_pkg'
    var_5 = '/tmp/path'
    var_6 = True
    var_7 = False
    var_8 = module_0.loader(var_4, var_5, var_6, var_6, var_7)
    assert var_8 == '\n'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'my_pkg.ext'
    var_1 = '/tmp/path/my_pkg/ext'
    var_2 = (var_0, var_1)
    var_3 = 'my_pkg'
    var_4 = '/tmp/path'
    var_5 = True
    var_6 = False
    var_7 = module_0.loader(var_3, var_4, var_5, var_5, var_6)
    assert var_7 == 'compiled_output'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'my_pkg.ext'
    var_1 = '/tmp/path/my_pkg/ext'
    var_2 = (var_0, var_1)
    var_3 = 'my_pkg'
    var_4 = '/tmp/path'
    var_5 = True
    var_6 = False
    var_7 = module_0.loader(var_3, var_4, var_5, var_5, var_6)
    assert var_7 == 'compiled_output'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_loader_pure_py_false_on_pyi. Retrieved 8/16 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'module_name'
    var_1 = '/path/to/module'
    var_2 = (var_0, var_1)
    var_3 = '.pyi'
    var_4 = '/root'
    var_5 = '/pwd'
    var_6 = True
    var_7 = module_0.loader(var_4, var_5, var_6, var_6, var_6)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_loader_pure_py_false. Retrieved 9/19 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'mypackage'
    var_1 = '/path/to/mypackage'
    var_2 = (var_0, var_1)
    var_3 = '/path/to/mypackage.pyi'
    var_4 = '/path'
    var_5 = '/pwd'
    var_6 = True
    var_7 = module_0.loader(var_4, var_5, var_6, var_6, var_6)
    assert var_7 == 'compiled_result'
    var_8 = 'content'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_loader_integration_with_mocked_filesystem. Retrieved 10/20 statements.
# Partially parsed test_loader_skips_pure_python_files. Retrieved 9/18 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'my_pkg.sub'
    var_1 = '/fake/path/my_pkg/sub'
    var_2 = (var_0, var_1)
    var_3 = '/fake/path/my_pkg/sub.pyi'
    var_4 = '.so'
    var_5 = 'my_pkg'
    var_6 = '/fake/path'
    var_7 = True
    var_8 = False
    var_9 = module_0.loader(var_5, var_6, var_7, var_7, var_8)
    assert var_9 == 'Generated Documentation'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'my_pkg.pure'
    var_1 = '/fake/path/my_pkg/pure'
    var_2 = (var_0, var_1)
    var_3 = '/fake/path/my_pkg/pure.py'
    var_4 = 'my_pkg'
    var_5 = '/fake/path'
    var_6 = True
    var_7 = False
    var_8 = module_0.loader(var_4, var_5, var_6, var_6, var_7)
    assert var_8 == 'Pure Doc'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_site_path_returns_empty_string_when_submodule_search_locations_is_none. Retrieved 2/6 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existent_module'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'module_without_locations'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_load_module_predicate_false_when_spec_is_none. Retrieved 4/24 statements.
# Partially parsed test_load_module_predicate_false_when_loader_is_not_correct_type. Retrieved 2/18 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = '/fake/path/to/module.py'
    var_2 = 'importlib.machinery'
    var_3 = None

def test_case_0():
    var_0 = 'test_module'
    var_1 = '/fake/path/to/module.py'
    var_2 = []
    var_3 = {}



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_loader_with_pure_python_modules. Retrieved 9/13 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'my_pkg'
    var_1 = '/tmp/my_pkg'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)
    assert var_4 == '\n'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'my_pkg.mod1'
    var_1 = '/tmp/my_pkg/mod1.py'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = 'my_pkg'
    var_5 = '/tmp/my_pkg'
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
    var_5 = '/tmp/my_pkg'
    var_6 = True
    var_7 = False
    var_8 = module_0.loader(var_4, var_5, var_6, var_6, var_7)
    assert var_8 == 'compiled_doc'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'my_pkg.ext'
    var_1 = '/tmp/my_pkg/ext'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = 'my_pkg'
    var_5 = '/tmp/my_pkg'
    var_6 = True
    var_7 = False
    var_8 = module_0.loader(var_4, var_5, var_6, var_6, var_7)
    assert var_8 == ''



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------






# Parsed testcases at query #2
#--------------------------

# Partially parsed test_load_module_success. Retrieved 4/10 statements.
# Partially parsed test_load_module_invalid_loader. Retrieved 4/7 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_mod'
    var_1 = '/fake/path.py'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is True

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'nonexistent_module'
    var_1 = '/fake/path.py'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_mod'
    var_1 = '/fake/path.py'
    var_2 = module_0.Parser()
    var_3 = module_1._load_module(var_0, var_1, var_2)
    assert var_3 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_mod'
    var_1 = '/fake/path.py'
    var_2 = module_0.Parser()
    var_3 = []
    var_4 = {}
    var_5 = module_1._load_module(var_0, var_1, var_2)
    assert var_5 is False



# Parsed testcases at query #3
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'pkg'
    var_1 = '/tmp'
    var_2 = module_0.walk_packages(var_0, var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'pkg'
    var_1 = '/tmp'
    var_2 = module_0.walk_packages(var_0, var_1)
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = var_3[0][0]
    assert var_5 == 'pkg.module'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'pkg'
    var_1 = '/tmp'
    var_2 = module_0.walk_packages(var_0, var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'pkg'
    var_1 = '/tmp'
    var_2 = module_0.walk_packages(var_0, var_1)
    var_3 = list(var_2)
    var_4 = bool(var_3 == [])
    assert var_4 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'pkg'
    var_1 = '/tmp'
    var_2 = module_0.walk_packages(var_0, var_1)
    var_3 = list(var_2)
    var_4 = len(var_3)
    var_5 = bool(var_4 >= 0)
    assert var_5 is True



# Parsed testcases at query #4
#--------------------------




import posixpath as module_0

def test_case_0():
    var_0 = '/tmp/test_pkg'
    var_1 = module_0.abspath(var_0)
    var_2 = 'my_package'
    var_3 = '.dist-info'
    var_4 = 'module.py'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_write_creates_file_with_correct_content. Retrieved 3/7 statements.
# Partially parsed test_write_overwrites_existing_file. Retrieved 5/9 statements.
# Partially parsed test_write_handles_empty_string. Retrieved 3/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_output.txt'
    var_1 = 'Hello, World!'
    var_2 = module_0._write(var_0, var_1)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_overwrite.txt'
    var_1 = 'initial content'
    var_2 = module_0._write(var_0, var_1)
    var_3 = 'new content'
    var_4 = module_0._write(var_0, var_3)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_empty.txt'
    var_1 = ''
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_read_success. Retrieved 3/6 statements.
# Partially parsed test_read_empty_file. Retrieved 3/6 statements.
# Partially parsed test_read_multiline_file. Retrieved 3/6 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_script.txt'
    var_1 = "print('hello world')"
    var_2 = module_0._read(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'empty_script.txt'
    var_1 = ''
    var_2 = module_0._read(var_0)
    assert var_2 == ''

def test_case_0():
    pass

import apimd.loader as module_0

def test_case_0():
    var_0 = 'multiline.txt'
    var_1 = 'line1\nline2\nline3'
    var_2 = module_0._read(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_loader_basic_flow. Retrieved 7/15 statements.
# Partially parsed test_loader_no_packages_found. Retrieved 5/10 statements.
# Partially parsed test_loader_skips_pure_py_files. Retrieved 8/13 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'my_package'
    var_1 = '/tmp/my_package'
    var_2 = (var_0, var_1)
    var_3 = '/tmp'
    var_4 = True
    var_5 = False
    var_6 = module_0.loader(var_0, var_3, var_4, var_4, var_5)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'empty_pkg'
    var_1 = '/tmp'
    var_2 = True
    var_3 = False
    var_4 = module_0.loader(var_0, var_1, var_2, var_2, var_3)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'my_pkg'
    var_1 = '/tmp/my_pkg'
    var_2 = (var_0, var_1)
    var_3 = '.py'
    var_4 = '/tmp'
    var_5 = True
    var_6 = False
    var_7 = module_0.loader(var_0, var_4, var_5, var_5, var_6)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_site_path_no_submodule_locations. Retrieved 2/6 statements.
# Partially parsed test_site_path_valid_package. Retrieved 6/11 statements.
# Failed to parse test_site_path_empty_list.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existent_module'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'package_without_locations'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''

import posixpath as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = '/usr/local/lib/python3.9/site-packages'
    var_1 = 'my_package'
    var_2 = [var_1]
    var_3 = module_0.join(var_0, *var_2)
    var_4 = module_0.dirname(var_3)
    var_5 = 'my_package'
    var_6 = module_1._site_path(var_5)
    var_7 = bool(var_6 == var_4)
    assert var_7 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_site_path_returns_empty_string_when_submodule_search_locations_is_none. Retrieved 2/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existent_module'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'module_without_locations'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_write_functionality. Retrieved 3/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_load_module_returns_false_when_spec_is_none. Retrieved 2/8 statements.
# Partially parsed test_load_module_returns_false_when_loader_is_not_Loader. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'pkg.module'
    var_1 = '/path/to/module.py'

def test_case_0():
    var_0 = 'pkg.module'
    var_1 = '/path/to/module.py'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_loader_predicate_evaluates_to_true. Retrieved 5/11 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = '/root'
    var_1 = '/pwd'
    var_2 = True
    var_3 = module_0.loader(var_0, var_1, var_2, var_2, var_2)
    var_4 = True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_load_module_predicate_false_due_invalid_loader. Retrieved 4/10 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_path.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_path.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_walk_packages_predicate_evaluates_to_true. Retrieved 5/6 statements.


def test_case_0():
    var_0 = 'module.py'
    var_1 = 'module.txt'
    var_2 = '.py'
    var_3 = '.pyi'
    var_4 = (var_2, var_3)



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_load_module_predicate_false_when_spec_is_none. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'test_module'
    var_1 = '/fake/path/test_module.py'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_load_module_returns_false_when_loader_is_not_Loader. Retrieved 4/12 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.mod'
    var_2 = '/path/to/mod.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'pkg.mod'
    var_2 = '/path/to/mod.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_write_success. Retrieved 3/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, world!'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_write_creates_file_with_correct_content. Retrieved 3/7 statements.
# Partially parsed test_write_overwrites_existing_file. Retrieved 5/9 statements.
# Partially parsed test_write_handles_empty_string. Retrieved 3/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_output.txt'
    var_1 = 'Hello, World!'
    var_2 = module_0._write(var_0, var_1)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_overwrite.txt'
    var_1 = 'Initial Content'
    var_2 = module_0._write(var_0, var_1)
    var_3 = 'New Content'
    var_4 = module_0._write(var_0, var_3)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_empty.txt'
    var_1 = ''
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_loader_predicate_evaluates_to_true. Retrieved 12/21 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'apimd.loader.Parser'
    var_1 = 'apimd.loader.walk_packages'
    var_2 = 'pkg'
    var_3 = 'path'
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = 'apimd.loader.isfile'
    var_7 = False
    var_8 = 'apimd.loader.logger'
    var_9 = '.'
    var_10 = True
    var_11 = module_0.loader(var_9, var_9, var_10, var_10, var_10)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_site_path_returns_empty_string_when_submodule_search_locations_is_none. Retrieved 2/6 statements.
# Partially parsed test_site_path_returns_dirname_of_first_location. Retrieved 4/9 statements.
# Partially parsed test_site_path_handles_single_location_correctly. Retrieved 6/11 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existent_module'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'module_without_locations'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''

import posixpath as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = '/path/to/site-packages/my_module'
    var_1 = module_0.dirname(var_0)
    var_2 = 'my_module'
    var_3 = module_1._site_path(var_2)
    var_4 = bool(var_3 == var_1)
    assert var_4 is True

import apimd.loader as module_0
import posixpath as module_1

def test_case_0():
    var_0 = '/usr/lib/python3.9/site-packages'
    var_1 = '/usr/lib/python3.9/site-packages'
    var_2 = 'some_package'
    var_3 = module_0._site_path(var_2)
    var_4 = '/usr/lib/python3.9/site-packages'
    var_5 = module_1.dirname(var_4)
    var_6 = bool(var_3 == var_5)
    assert var_6 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_site_path_returns_empty_string_when_submodule_search_locations_is_none. Retrieved 2/6 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existent_module'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'module_without_locations'
    var_1 = module_0._site_path(var_0)
    assert var_1 == ''



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_loader_returns_string. Retrieved 7/17 statements.
# Partially parsed test_loader_with_empty_package. Retrieved 9/18 statements.
# Partially parsed test_loader_with_toc_enabled. Retrieved 8/15 statements.


import posixpath as module_0

def test_case_0():
    var_0 = 'test_pkg_dir'
    var_1 = module_0.abspath(var_0)
    var_2 = True
    var_3 = '__init__.py'
    var_4 = [var_3]
    var_5 = module_0.join(var_1, *var_4)
    var_6 = 'x = 1\n'
    var_7 = False
    var_8 = '# Module `test_pkg_dir`'

import posixpath as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_empty_pkg'
    var_1 = module_0.abspath(var_0)
    var_2 = True
    var_3 = '__init__.py'
    var_4 = [var_3]
    var_5 = module_0.join(var_1, *var_4)
    var_6 = ''
    var_7 = 'empty_pkg'
    var_8 = False
    var_9 = module_1.loader(var_7, var_1, var_2, var_2, var_8)

import posixpath as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = 'test_toc_pkg'
    var_1 = module_0.abspath(var_0)
    var_2 = True
    var_3 = '__init__.py'
    var_4 = [var_3]
    var_5 = module_0.join(var_1, *var_4)
    var_6 = "VERSION = '1.0.0'\n"
    var_7 = 'toc_pkg'
    var_8 = module_1.loader(var_7, var_1, var_2, var_2, var_2)
    var_9 = '**Table of contents:**'
    var_10 = bool('**Table of contents:**' in var_8)
    assert var_10 is True
    var_11 = bool('+ [toc_pkg](#toc-pkg)' in var_8 or '+ [toc_pkg](#toc_pkg)' in var_8)
    assert var_11 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_read_predicate_is_false. Retrieved 6/13 statements.
# Partially parsed test_read_predicate_evaluates_to_false_via_mock. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'content'
    var_2 = 'r'
    var_3 = open(var_0, var_2)
    var_4 = None
    var_5 = bool(var_4)
    var_6 = bool(not var_5)
    assert var_6 is True

def test_case_0():
    var_0 = 'dummy.txt'
    var_1 = 'r'
    var_2 = open(var_0, var_1)



# Parsed testcases at query #24
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
    var_0 = 'non_existent_file.txt'
    var_1 = module_0._read(var_0)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_read_fails_on_nonexistent_file. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'non_existent_file_12345.txt'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_read_file_content. Retrieved 3/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_script.txt'
    var_1 = "print('hello')"
    var_2 = module_0._read(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_read_success. Retrieved 3/6 statements.
# Partially parsed test_read_empty_file. Retrieved 2/5 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_content.txt'
    var_1 = 'Hello, World!'
    var_2 = module_0._read(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True

import apimd.loader as module_0

def test_case_0():
    var_0 = 'empty.txt'
    var_1 = module_0._read(var_0)
    assert var_1 == ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existent_file_12345.txt'
    var_1 = module_0._read(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_read_success. Retrieved 3/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'hello world'
    var_2 = module_0._read(var_0)
    var_3 = bool(var_2 == var_1)
    assert var_3 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_read_success. Retrieved 2/6 statements.
# Partially parsed test_read_empty_file. Retrieved 2/6 statements.


def test_case_0():
    var_0 = "print('hello world')"
    var_1 = 'script.py'

def test_case_0():
    var_0 = 'empty.py'
    var_1 = ''

import apimd.loader as module_0

def test_case_0():
    var_0 = 'non_existent_file.txt'
    var_1 = module_0._read(var_0)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_write_creates_file_with_correct_content. Retrieved 3/7 statements.
# Partially parsed test_write_overwrites_existing_file. Retrieved 5/9 statements.
# Partially parsed test_write_handles_empty_string. Retrieved 3/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = module_0._write(var_0, var_1)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'overwrite_test.txt'
    var_1 = 'Initial Content'
    var_2 = module_0._write(var_0, var_1)
    var_3 = 'New Content'
    var_4 = module_0._write(var_0, var_3)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'empty_test.txt'
    var_1 = ''
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_loader_basic_execution. Retrieved 4/16 statements.
# Partially parsed test_loader_with_toc. Retrieved 3/15 statements.
# Partially parsed test_loader_empty_package. Retrieved 4/16 statements.


def test_case_0():
    var_0 = 'test_package'
    var_1 = '"""Module doc."""\n__all__ = ["func"]\ndef func():\n    """Func doc."""\n    pass'
    var_2 = True
    var_3 = False
    var_4 = '# Module `test_package`'
    var_5 = 'func()'
    var_6 = 'Func doc.'

def test_case_0():
    var_0 = 'toc_package'
    var_1 = '"""Doc."""\nclass MyClass:\n    """Class doc."""\n    pass'
    var_2 = True
    var_3 = '**Table of contents:**'
    var_4 = '+ [toc_package](#toc-package)'
    var_5 = '+ [toc_package.MyClass](#toc-package-myclass)'

def test_case_0():
    var_0 = 'empty_package'
    var_1 = '"""Empty."""'
    var_2 = True
    var_3 = False
    var_4 = '# Module `empty_package`'



# Parsed testcases at query #32
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_load_module_success. Retrieved 5/14 statements.
# Partially parsed test_load_module_failed_spec. Retrieved 5/10 statements.
# Partially parsed test_load_module_invalid_loader. Retrieved 5/12 statements.


import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module'
    var_3 = '/path/to/module.py'
    var_4 = module_1._load_module(var_2, var_3, var_0)
    assert var_4 is True

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'non_existent_module'
    var_2 = '/path/to/module.py'
    var_3 = module_1._load_module(var_1, var_2, var_0)
    assert var_3 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = 'test_module'
    var_3 = '/path/to/module.py'
    var_4 = module_1._load_module(var_2, var_3, var_0)
    assert var_4 is False

import apimd.parser as module_0
import apimd.loader as module_1

def test_case_0():
    var_0 = module_0.Parser()
    var_1 = 'test_module'
    var_2 = []
    var_3 = {}
    var_4 = 'test_module'
    var_5 = '/path/to/module.py'
    var_6 = module_1._load_module(var_4, var_5, var_0)
    assert var_6 is False



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_loader_predicate_at_line_13_is_false. Retrieved 11/22 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'package'
    var_1 = '/path/to/pkg'
    var_2 = (var_0, var_1)
    var_3 = [var_2]
    var_4 = True
    var_5 = 'content'
    var_6 = '.'
    var_7 = False
    var_8 = 1
    var_9 = True
    var_10 = module_0.loader(var_6, var_6, var_7, var_8, var_9)



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_loader_integration_with_mocked_parser. Retrieved 9/20 statements.
# Partially parsed test_loader_skips_pure_python_modules. Retrieved 10/20 statements.
# Partially parsed test_loader_extension_loading_logic. Retrieved 8/20 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'my_pkg.module'
    var_1 = '/fake/path/module'
    var_2 = (var_0, var_1)
    var_3 = '.py'
    var_4 = '.pyd'
    var_5 = 'my_pkg'
    var_6 = '/fake/path'
    var_7 = True
    var_8 = module_0.loader(var_5, var_6, var_7, var_7, var_7)
    assert var_8 == 'compiled_doc'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'my_pkg.module'
    var_1 = '/fake/path/module'
    var_2 = (var_0, var_1)
    var_3 = '.py'
    var_4 = 'my_pkg'
    var_5 = '/fake/path'
    var_6 = True
    var_7 = False
    var_8 = module_0.loader(var_4, var_5, var_6, var_6, var_7)
    assert var_8 == 'pure_python_doc'
    var_9 = "print('hello')"

import apimd.loader as module_0

def test_case_0():
    var_0 = 'my_pkg.ext'
    var_1 = '/fake/path/ext'
    var_2 = (var_0, var_1)
    var_3 = 'my_pkg'
    var_4 = '/fake/path'
    var_5 = True
    var_6 = False
    var_7 = module_0.loader(var_3, var_4, var_5, var_5, var_6)
    assert var_7 == 'extension_doc'



