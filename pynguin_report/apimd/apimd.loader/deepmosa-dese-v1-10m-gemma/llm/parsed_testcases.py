####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'my_pkg'
    var_1 = '/tmp/test'
    var_2 = module_0.walk_packages(var_0, var_1)
    var_3 = list(var_2)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'my_pkg'
    var_1 = '/tmp/test'
    var_2 = module_0.walk_packages(var_0, var_1)
    var_3 = list(var_2)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'my_pkg'
    var_1 = '/tmp/test'
    var_2 = module_0.walk_packages(var_0, var_1)
    var_3 = list(var_2)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'my_pkg'
    var_1 = '/tmp/test'
    var_2 = module_0.walk_packages(var_0, var_1)
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 1

import apimd.loader as module_0

def test_case_0():
    var_0 = 'my_pkg'
    var_1 = '/tmp/test'
    var_2 = module_0.walk_packages(var_0, var_1)
    var_3 = list(var_2)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'my_pkg'
    var_1 = '/tmp/test'
    var_2 = module_0.walk_packages(var_0, var_1)
    var_3 = list(var_2)



# Parsed testcases at query #2
#--------------------------




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
    var_0 = 'my_pkg.module'
    var_1 = '/tmp/my_pkg/module.py'
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
    assert var_8 == 'Compiled Doc'

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
    assert var_8 == ''



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_walk_packages_predicate_evaluates_true. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 'module.py'
    var_1 = '.py'
    var_2 = '.pyi'
    var_3 = (var_1, var_2)
    var_4 = False
    var_5 = 'interface.pyi'
    var_6 = (var_1, var_2)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_loader_integration_with_mock_parser. Retrieved 9/18 statements.
# Partially parsed test_loader_skips_pure_python_modules. Retrieved 9/15 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'pkg.sub'
    var_1 = '/path/to/pkg/sub'
    var_2 = (var_0, var_1)
    var_3 = '.py'
    var_4 = 'pkg'
    var_5 = '/path/to/pkg'
    var_6 = True
    var_7 = False
    var_8 = module_0.loader(var_4, var_5, var_6, var_6, var_7)
    assert var_8 == 'compiled_output'

import apimd.loader as module_0

def test_case_0():
    var_0 = 'pkg.pure'
    var_1 = '/path/to/pkg/pure'
    var_2 = (var_0, var_1)
    var_3 = '.py'
    var_4 = 'pkg'
    var_5 = '/path/to/pkg'
    var_6 = True
    var_7 = False
    var_8 = module_0.loader(var_4, var_5, var_6, var_6, var_7)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/7 statements.
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



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
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



# Parsed testcases at query #2
#--------------------------




import apimd.loader as module_0

def test_case_0():
    var_0 = 'mypkg'
    var_1 = '/tmp'
    var_2 = module_0.walk_packages(var_0, var_1)
    var_3 = list(var_2)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'mypkg'
    var_1 = '/tmp'
    var_2 = module_0.walk_packages(var_0, var_1)
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 1

import apimd.loader as module_0

def test_case_0():
    var_0 = 'mypkg'
    var_1 = '/tmp'
    var_2 = module_0.walk_packages(var_0, var_1)
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 1

import apimd.loader as module_0

def test_case_0():
    var_0 = 'mypkg'
    var_1 = '/tmp'
    var_2 = module_0.walk_packages(var_0, var_1)
    var_3 = list(var_2)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'mypkg'
    var_1 = '/tmp'
    var_2 = module_0.walk_packages(var_0, var_1)
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 1



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_write_creates_file_with_content. Retrieved 3/6 statements.
# Partially parsed test_write_overwrites_existing_file. Retrieved 5/8 statements.
# Partially parsed test_write_handles_empty_string. Retrieved 3/6 statements.
# Partially parsed test_write_handles_unicode_characters. Retrieved 3/6 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_output.txt'
    var_1 = 'Hello, World!'
    var_2 = module_0._write(var_0, var_1)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_overwrite.txt'
    var_1 = 'Initial content'
    var_2 = module_0._write(var_0, var_1)
    var_3 = 'New content'
    var_4 = module_0._write(var_0, var_3)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_empty.txt'
    var_1 = ''
    var_2 = module_0._write(var_0, var_1)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_unicode.txt'
    var_1 = 'こんにちは 🌍'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_write_success. Retrieved 3/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = module_0._write(var_0, var_1)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_write_creates_file_with_correct_content. Retrieved 3/7 statements.
# Partially parsed test_write_overwrites_existing_file. Retrieved 5/9 statements.
# Partially parsed test_write_handles_empty_string. Retrieved 3/7 statements.
# Partially parsed test_write_handles_special_characters. Retrieved 3/7 statements.


import apimd.loader as module_0

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = 'Hello, World!'
    var_2 = module_0._write(var_0, var_1)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'overwrite_test.txt'
    var_1 = 'initial content'
    var_2 = module_0._write(var_0, var_1)
    var_3 = 'new content'
    var_4 = module_0._write(var_0, var_3)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'empty_test.txt'
    var_1 = ''
    var_2 = module_0._write(var_0, var_1)

import apimd.loader as module_0

def test_case_0():
    var_0 = 'special_char_test.txt'
    var_1 = '🔥 Unicode Test: \n\t\r'
    var_2 = module_0._write(var_0, var_1)



