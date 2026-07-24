####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_find_with_python_files_in_directory. Retrieved 8/25 statements.
# Partially parsed test_find_with_skipped_files. Retrieved 6/20 statements.
# Partially parsed test_find_with_nonexistent_path. Retrieved 6/15 statements.
# Partially parsed test_find_with_single_file. Retrieved 5/20 statements.
# Partially parsed test_find_with_nested_directories. Retrieved 8/26 statements.
# Partially parsed test_find_with_unsupported_filetype. Retrieved 5/19 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 7/23 statements.


def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = False
    var_3 = []
    var_4 = []
    var_5 = 0
    var_6 = len(var_3)
    assert var_6 == 0
    var_7 = len(var_4)
    assert var_7 == 0

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = False
    var_3 = []
    var_4 = []
    var_5 = len(var_3)
    assert var_5 == 1

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = '/nonexistent/path'
    var_4 = [var_3]
    var_5 = len(var_2)
    assert var_5 == 1
    var_6 = var_2[0]
    assert var_6 == '/nonexistent/path'

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = False
    var_3 = []
    var_4 = []

def test_case_0():
    var_0 = 'subdir'
    var_1 = 'test1.py'
    var_2 = 'test2.py'
    var_3 = "print('1')"
    var_4 = "print('2')"
    var_5 = False
    var_6 = []
    var_7 = []

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'hello'
    var_2 = False
    var_3 = []
    var_4 = []

def test_case_0():
    var_0 = 'skip_me'
    var_1 = 'test.py'
    var_2 = "print('hello')"
    var_3 = False
    var_4 = []
    var_5 = []
    var_6 = len(var_4)
    assert var_6 == 1



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_find_with_directory_containing_python_files. Retrieved 13/37 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 9/29 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 10/28 statements.
# Partially parsed test_find_with_single_file_path. Retrieved 5/15 statements.
# Partially parsed test_find_with_nonexistent_path. Retrieved 5/9 statements.
# Partially parsed test_find_with_nested_directories. Retrieved 13/34 statements.
# Partially parsed test_find_with_multiple_paths. Retrieved 10/28 statements.


def test_case_0():
    var_0 = 'test_src'
    var_1 = 'file1.py'
    var_2 = '# python file 1'
    var_3 = 'file2.py'
    var_4 = '# python file 2'
    var_5 = 'file3.txt'
    var_6 = '# not python'
    var_7 = '.py'
    var_8 = False
    var_9 = []
    var_10 = []
    var_11 = len(var_9)
    assert var_11 == 0
    var_12 = len(var_10)
    assert var_12 == 0

def test_case_0():
    var_0 = 'test_src'
    var_1 = 'skip_me'
    var_2 = 'file1.py'
    var_3 = '# python file'
    var_4 = 'file2.py'
    var_5 = '.py'
    var_6 = []
    var_7 = []
    var_8 = len(var_6)
    assert var_8 == 1

def test_case_0():
    var_0 = 'test_src'
    var_1 = 'skip_file.py'
    var_2 = '# skip this'
    var_3 = 'keep_file.py'
    var_4 = '# keep this'
    var_5 = '.py'
    var_6 = 'skip_file'
    var_7 = []
    var_8 = []
    var_9 = len(var_7)
    assert var_9 == 1

def test_case_0():
    var_0 = 'single_file.py'
    var_1 = '# python file'
    var_2 = False
    var_3 = []
    var_4 = []

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '/nonexistent/path/file.py'
    var_3 = [var_2]
    var_4 = len(var_1)
    assert var_4 == 1
    var_5 = '/nonexistent/path/file.py'
    var_6 = bool('/nonexistent/path/file.py' in var_1)
    assert var_6 is True

def test_case_0():
    var_0 = 'test_src'
    var_1 = 'subdir1'
    var_2 = 'subdir2'
    var_3 = 'file1.py'
    var_4 = '# file 1'
    var_5 = 'file2.py'
    var_6 = '# file 2'
    var_7 = 'file3.py'
    var_8 = '# file 3'
    var_9 = '.py'
    var_10 = False
    var_11 = []
    var_12 = []

def test_case_0():
    var_0 = 'dir1'
    var_1 = 'dir2'
    var_2 = 'file1.py'
    var_3 = '# file 1'
    var_4 = 'file2.py'
    var_5 = '# file 2'
    var_6 = '.py'
    var_7 = False
    var_8 = []
    var_9 = []



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_find_predicate_line_7_evaluates_to_false. Retrieved 4/20 statements.


def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = len(var_2)
    assert var_3 == 0



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_predicate_line_7_evaluates_to_false. Retrieved 7/23 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 7 (os.path.isdir(path)) evaluates to False.'
    var_1 = False
    var_2 = True
    var_3 = 'test_file.py'
    var_4 = '# test content'
    var_5 = []
    var_6 = []



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_find_with_single_python_file. Retrieved 7/22 statements.
# Partially parsed test_find_with_directory_containing_python_files. Retrieved 10/33 statements.
# Partially parsed test_find_with_skipped_files. Retrieved 6/19 statements.
# Partially parsed test_find_with_nonexistent_path. Retrieved 6/15 statements.
# Partially parsed test_find_with_nested_directories. Retrieved 8/30 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 9/28 statements.
# Partially parsed test_find_with_unsupported_filetype. Retrieved 7/24 statements.
# Partially parsed test_find_with_multiple_paths. Retrieved 7/26 statements.


def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = False
    var_3 = []
    var_4 = []
    var_5 = len(var_3)
    assert var_5 == 0
    var_6 = len(var_4)
    assert var_6 == 0

def test_case_0():
    var_0 = 'file1.py'
    var_1 = 'file2.py'
    var_2 = 'file.txt'
    var_3 = "print('1')"
    var_4 = "print('2')"
    var_5 = 'text'
    var_6 = False
    var_7 = []
    var_8 = []
    var_9 = len(var_7)
    assert var_9 == 0

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = False
    var_3 = []
    var_4 = []
    var_5 = len(var_3)
    assert var_5 == 1

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = '/nonexistent/path.py'
    var_4 = [var_3]
    var_5 = len(var_2)
    assert var_5 == 1
    var_6 = var_2[0]
    assert var_6 == '/nonexistent/path.py'

def test_case_0():
    var_0 = 'subdir'
    var_1 = 'file1.py'
    var_2 = 'file2.py'
    var_3 = "print('1')"
    var_4 = "print('2')"
    var_5 = False
    var_6 = []
    var_7 = []

def test_case_0():
    var_0 = 'skipped_dir'
    var_1 = 'file1.py'
    var_2 = 'file2.py'
    var_3 = "print('1')"
    var_4 = "print('2')"
    var_5 = False
    var_6 = []
    var_7 = []
    var_8 = len(var_6)
    assert var_8 == 1

def test_case_0():
    var_0 = 'file.py'
    var_1 = 'file.txt'
    var_2 = "print('hello')"
    var_3 = 'text'
    var_4 = False
    var_5 = []
    var_6 = []

def test_case_0():
    var_0 = 'file1.py'
    var_1 = 'file2.py'
    var_2 = "print('1')"
    var_3 = "print('2')"
    var_4 = False
    var_5 = []
    var_6 = []



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_predicate_line_7_evaluates_to_false. Retrieved 4/25 statements.


def test_case_0():
    var_0 = 'test.py'
    var_1 = '# test'
    var_2 = []
    var_3 = []



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_find_with_single_python_file. Retrieved 6/14 statements.
# Partially parsed test_find_with_directory_containing_python_files. Retrieved 10/25 statements.
# Partially parsed test_find_with_nested_directories. Retrieved 9/24 statements.
# Partially parsed test_find_with_skipped_files. Retrieved 9/23 statements.
# Partially parsed test_find_with_skipped_directories. Retrieved 9/23 statements.
# Partially parsed test_find_with_multiple_paths. Retrieved 8/18 statements.
# Partially parsed test_find_with_unsupported_file_type. Retrieved 8/19 statements.
# Partially parsed test_find_empty_directory. Retrieved 5/12 statements.
# Partially parsed test_find_mixed_valid_and_invalid_paths. Retrieved 7/15 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test find with a single Python file path.'
    var_1 = 'test.py'
    var_2 = "print('hello')"
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = []
    var_6 = []

import isort.settings as module_0

def test_case_0():
    var_0 = 'Test find with a directory containing Python files.'
    var_1 = 'file1.py'
    var_2 = "print('1')"
    var_3 = 'file2.py'
    var_4 = "print('2')"
    var_5 = 'file3.txt'
    var_6 = 'not python'
    var_7 = {}
    var_8 = module_0.Config(**var_7)
    var_9 = []
    var_10 = []

import isort.settings as module_0

def test_case_0():
    var_0 = 'Test find with nested directory structure.'
    var_1 = 'subdir'
    var_2 = 'root.py'
    var_3 = "print('root')"
    var_4 = 'nested.py'
    var_5 = "print('nested')"
    var_6 = {}
    var_7 = module_0.Config(**var_6)
    var_8 = []
    var_9 = []

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'Test find with a path that does not exist.'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = []
    var_4 = []
    var_5 = '/nonexistent/path/to/file.py'
    var_6 = [var_5]
    var_7 = module_1.find(var_6, var_2, var_3, var_4)
    var_8 = list(var_7)
    var_9 = len(var_8)
    assert var_9 == 0
    var_10 = '/nonexistent/path/to/file.py'
    var_11 = bool('/nonexistent/path/to/file.py' in var_4)
    assert var_11 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'Test find respects config.is_skipped for files.'
    var_1 = 'include.py'
    var_2 = "print('include')"
    var_3 = 'skip.py'
    var_4 = "print('skip')"
    var_5 = {}
    var_6 = module_0.Config(**var_5)
    var_7 = 'skip'
    var_8 = []
    var_9 = []

import isort.settings as module_0

def test_case_0():
    var_0 = 'Test find respects config.is_skipped for directories.'
    var_1 = 'skip_dir'
    var_2 = 'file.py'
    var_3 = "print('skip')"
    var_4 = 'include.py'
    var_5 = "print('include')"
    var_6 = {}
    var_7 = module_0.Config(**var_6)
    var_8 = []
    var_9 = []

import isort.settings as module_0

def test_case_0():
    var_0 = 'Test find with multiple paths.'
    var_1 = 'file1.py'
    var_2 = 'file2.py'
    var_3 = "print('1')"
    var_4 = "print('2')"
    var_5 = {}
    var_6 = module_0.Config(**var_5)
    var_7 = []
    var_8 = []

import isort.settings as module_0

def test_case_0():
    var_0 = 'Test find ignores unsupported file types.'
    var_1 = 'file.txt'
    var_2 = 'not python'
    var_3 = 'file.py'
    var_4 = "print('python')"
    var_5 = {}
    var_6 = module_0.Config(**var_5)
    var_7 = []
    var_8 = []

import isort.settings as module_0

def test_case_0():
    var_0 = 'Test find with an empty directory.'
    var_1 = 'empty'
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []

import isort.settings as module_0

def test_case_0():
    var_0 = 'Test find with a mix of valid and invalid paths.'
    var_1 = 'valid.py'
    var_2 = "print('valid')"
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = []
    var_6 = []
    var_7 = '/invalid/path.py'
    var_8 = '/invalid/path.py'
    var_9 = bool('/invalid/path.py' in var_6)
    assert var_9 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_predicate_at_line_31_evaluates_to_true. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 'non_existent_file.py'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_find_with_directory_containing_python_files. Retrieved 8/20 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 8/22 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 8/20 statements.
# Partially parsed test_find_with_unsupported_filetype. Retrieved 6/17 statements.
# Partially parsed test_find_with_nonexistent_path. Retrieved 5/10 statements.
# Partially parsed test_find_with_direct_file_path. Retrieved 4/14 statements.
# Partially parsed test_find_with_nested_directories. Retrieved 9/24 statements.


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'test.py'
    var_3 = '# test'
    var_4 = []
    var_5 = []
    var_6 = 'test.py'
    var_7 = len(var_4)
    assert var_7 == 0
    var_8 = len(var_5)
    assert var_8 == 0

def test_case_0():
    var_0 = True
    var_1 = 'skip_dir'
    var_2 = lambda p: var_1 in str(p)
    var_3 = 'test.py'
    var_4 = '# test'
    var_5 = []
    var_6 = []
    var_7 = len(var_5)
    assert var_7 == 1

def test_case_0():
    var_0 = True
    var_1 = 'skip'
    var_2 = lambda p: var_1 in str(p)
    var_3 = 'skip_test.py'
    var_4 = '# test'
    var_5 = []
    var_6 = []
    var_7 = len(var_5)
    assert var_7 == 1

def test_case_0():
    var_0 = False
    var_1 = 'test.txt'
    var_2 = 'test'
    var_3 = []
    var_4 = []
    var_5 = len(var_3)
    assert var_5 == 0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '/nonexistent/path'
    var_3 = [var_2]
    var_4 = len(var_1)
    assert var_4 == 1
    var_5 = '/nonexistent/path'
    var_6 = bool('/nonexistent/path' in var_1)
    assert var_6 is True

def test_case_0():
    var_0 = 'test.py'
    var_1 = '# test'
    var_2 = []
    var_3 = []

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'subdir'
    var_3 = 'test1.py'
    var_4 = 'test2.py'
    var_5 = '# test1'
    var_6 = '# test2'
    var_7 = []
    var_8 = []



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_find_with_directory_containing_python_files. Retrieved 10/37 statements.
# Partially parsed test_find_with_skipped_files. Retrieved 9/31 statements.
# Partially parsed test_find_with_nested_directories. Retrieved 9/28 statements.
# Partially parsed test_find_with_nonexistent_path. Retrieved 6/15 statements.
# Partially parsed test_find_with_single_file. Retrieved 5/19 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 10/31 statements.


def test_case_0():
    var_0 = 'test_project'
    var_1 = 'file1.py'
    var_2 = '# python file'
    var_3 = 'file2.txt'
    var_4 = '# text file'
    var_5 = 'file3.py'
    var_6 = '# another python file'
    var_7 = False
    var_8 = []
    var_9 = []

def test_case_0():
    var_0 = 'test_project'
    var_1 = 'file1.py'
    var_2 = '# python file'
    var_3 = 'skip_me.py'
    var_4 = '# skip this'
    var_5 = False
    var_6 = []
    var_7 = []
    var_8 = len(var_6)
    assert var_8 == 1

def test_case_0():
    var_0 = 'test_project'
    var_1 = 'subdir'
    var_2 = 'file1.py'
    var_3 = '# root'
    var_4 = 'file2.py'
    var_5 = '# nested'
    var_6 = False
    var_7 = []
    var_8 = []

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = '/nonexistent/path'
    var_4 = [var_3]
    var_5 = len(var_2)
    assert var_5 == 1
    var_6 = '/nonexistent/path'
    var_7 = bool('/nonexistent/path' in var_2)
    assert var_7 is True

def test_case_0():
    var_0 = 'test.py'
    var_1 = '# python file'
    var_2 = False
    var_3 = []
    var_4 = []

def test_case_0():
    var_0 = 'test_project'
    var_1 = 'skip_dir'
    var_2 = 'file1.py'
    var_3 = '# root'
    var_4 = 'file2.py'
    var_5 = '# should skip'
    var_6 = False
    var_7 = []
    var_8 = []
    var_9 = len(var_7)
    assert var_9 == 1



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_predicate_at_line_9_evaluates_to_false. Retrieved 4/24 statements.


def test_case_0():
    var_0 = 'test.py'
    var_1 = '# test'
    var_2 = []
    var_3 = []



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_predicate_at_line_27_evaluates_to_true. Retrieved 3/21 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 27 evaluates to True when a file is skipped.'
    var_1 = 'test.py'
    var_2 = '# test file'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_predicate_at_line_27_evaluates_to_true. Retrieved 6/23 statements.


def test_case_0():
    var_0 = True
    var_1 = 'test.py'
    var_2 = '# test file'
    var_3 = []
    var_4 = []
    var_5 = len(var_3)
    var_6 = bool(var_5 > 0)
    assert var_6 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_predicate_at_line_27_evaluates_to_true. Retrieved 5/24 statements.


def test_case_0():
    var_0 = True
    var_1 = 'test.py'
    var_2 = '# test'
    var_3 = []
    var_4 = []



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_predicate_at_line_31_evaluates_to_true. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'non_existent_file.py'
    var_1 = True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_find_with_directory_containing_python_files. Retrieved 10/34 statements.
# Partially parsed test_find_with_single_file. Retrieved 6/18 statements.
# Partially parsed test_find_with_nonexistent_path. Retrieved 4/9 statements.
# Partially parsed test_find_with_skipped_files. Retrieved 10/25 statements.
# Partially parsed test_find_with_nested_directories. Retrieved 10/27 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 10/27 statements.
# Partially parsed test_find_with_multiple_paths. Retrieved 9/27 statements.


def test_case_0():
    var_0 = 'test_src'
    var_1 = 'file1.py'
    var_2 = '# test'
    var_3 = 'file2.py'
    var_4 = 'file3.txt'
    var_5 = False
    var_6 = '.py'
    var_7 = lambda x: x.endswith(var_6)
    var_8 = []
    var_9 = []

def test_case_0():
    var_0 = 'test.py'
    var_1 = '# test'
    var_2 = False
    var_3 = True
    var_4 = []
    var_5 = []

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '/nonexistent/path/file.py'
    var_3 = [var_2]
    var_4 = '/nonexistent/path/file.py'
    var_5 = bool('/nonexistent/path/file.py' in var_1)
    assert var_5 is True

def test_case_0():
    var_0 = 'test_src'
    var_1 = 'file1.py'
    var_2 = '# test'
    var_3 = 'file2.py'
    var_4 = 'file2'
    var_5 = lambda x: var_4 in str(x)
    var_6 = True
    var_7 = []
    var_8 = []
    var_9 = len(var_7)
    assert var_9 == 1
    var_10 = 'file2.py'
    var_11 = bool('file2.py' in var_7[0])
    assert var_11 is True

def test_case_0():
    var_0 = 'test_src'
    var_1 = 'file1.py'
    var_2 = '# test'
    var_3 = 'subdir'
    var_4 = 'file2.py'
    var_5 = False
    var_6 = '.py'
    var_7 = lambda x: x.endswith(var_6)
    var_8 = []
    var_9 = []

def test_case_0():
    var_0 = 'test_src'
    var_1 = 'file1.py'
    var_2 = '# test'
    var_3 = 'skip_me'
    var_4 = 'file2.py'
    var_5 = lambda x: var_3 in str(x)
    var_6 = True
    var_7 = []
    var_8 = []
    var_9 = len(var_7)
    assert var_9 == 1
    var_10 = 'skip_me'
    var_11 = bool('skip_me' in var_7[0])
    assert var_11 is True

def test_case_0():
    var_0 = 'dir1'
    var_1 = 'file1.py'
    var_2 = '# test'
    var_3 = 'dir2'
    var_4 = 'file2.py'
    var_5 = False
    var_6 = True
    var_7 = []
    var_8 = []



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_find_evaluates_for_loop_at_line_7. Retrieved 6/30 statements.


def test_case_0():
    var_0 = 'test_project'
    var_1 = 'test.py'
    var_2 = 'print("hello")'
    var_3 = []
    var_4 = []
    var_5 = len(var_4)
    assert var_5 == 0



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_find_evaluates_predicate_at_line_7. Retrieved 8/25 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 7 (for path in paths) evaluates to True.'
    var_1 = False
    var_2 = 'test.py'
    var_3 = '# test'
    var_4 = []
    var_5 = []
    var_6 = list(var_1)
    var_7 = 'paths should be iterable'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_find_evaluates_path_iteration_predicate. Retrieved 13/30 statements.


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = '/test/path'
    var_3 = [var_2]
    var_4 = []
    var_5 = []
    var_6 = '/test/path'
    var_7 = 'subdir'
    var_8 = [var_7]
    var_9 = 'file.py'
    var_10 = [var_9]
    var_11 = (var_6, var_8, var_10)
    var_12 = len(var_3)
    var_13 = bool(var_12 > 0)
    assert var_13 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_find_evaluates_predicate_at_line_7. Retrieved 10/24 statements.
# Partially parsed test_find_predicate_line_7_with_file_path. Retrieved 6/21 statements.
# Partially parsed test_find_predicate_line_7_empty_paths. Retrieved 3/9 statements.


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'test.py'
    var_3 = '# test'
    var_4 = []
    var_5 = []
    var_6 = list(var_1)
    var_7 = len(var_6)
    var_8 = 0
    var_9 = var_7 > var_8

def test_case_0():
    var_0 = False
    var_1 = '# test'
    var_2 = []
    var_3 = []
    var_4 = list(var_1)
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = var_4[0]

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_predicate_at_line_7_evaluates_to_false. Retrieved 4/14 statements.


def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = bool(var_2 == [])
    assert var_4 is True
    var_5 = bool(var_3 == [])
    assert var_5 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_at_line_7_evaluates_to_false. Retrieved 6/21 statements.


def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = '/nonexistent/path/that/does/not/exist'
    var_4 = [var_3]
    var_5 = list(var_0)
    var_6 = '/nonexistent/path/that/does/not/exist'
    var_7 = bool('/nonexistent/path/that/does/not/exist' in var_2)
    assert var_7 is True
    var_8 = bool(var_5 == [])
    assert var_8 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_predicate_at_line_7_evaluates_to_false. Retrieved 3/17 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = bool(var_1 == [])
    assert var_3 is True
    var_4 = bool(var_2 == [])
    assert var_4 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_find_evaluates_path_iteration_predicate. Retrieved 12/28 statements.


def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'test.py'
    var_3 = '# test'
    var_4 = []
    var_5 = []
    var_6 = list(var_1)
    var_7 = len(var_6)
    var_8 = 0
    var_9 = var_7 > var_8
    var_10 = len(var_4)
    var_11 = var_10 >= var_8
    var_12 = bool(var_9 or var_11)
    assert var_12 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_predicate_line_7_evaluates_to_false. Retrieved 8/22 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 7 (os.path.isdir(path)) evaluates to False.'
    var_1 = 'test_file.py'
    var_2 = '# test'
    var_3 = []
    var_4 = []
    var_5 = [var_0]
    var_6 = list(var_2)
    var_7 = len(var_4)
    assert var_7 == 0



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_predicate_at_line_7_evaluates_to_false. Retrieved 4/43 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '/this/path/does/not/exist/test_file_xyz.py'
    var_3 = [var_2]
    var_4 = bool(var_2 in var_1)
    assert var_4 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_find_with_single_python_file. Retrieved 5/17 statements.
# Partially parsed test_find_with_directory_containing_python_files. Retrieved 8/23 statements.
# Partially parsed test_find_with_skipped_files. Retrieved 10/25 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 8/25 statements.
# Partially parsed test_find_with_broken_path. Retrieved 6/11 statements.
# Partially parsed test_find_with_unsupported_filetype. Retrieved 9/24 statements.
# Partially parsed test_find_with_nested_directories. Retrieved 9/26 statements.
# Partially parsed test_find_with_multiple_paths. Retrieved 9/27 statements.


def test_case_0():
    var_0 = 'Test find with a single Python file path.'
    var_1 = 'test.py'
    var_2 = "print('hello')"
    var_3 = []
    var_4 = []
    var_5 = bool(var_3 == [])
    assert var_5 is True
    var_6 = bool(var_4 == [])
    assert var_6 is True

def test_case_0():
    var_0 = 'Test find with a directory containing Python files.'
    var_1 = 'src'
    var_2 = 'file1.py'
    var_3 = '# file1'
    var_4 = 'file2.py'
    var_5 = '# file2'
    var_6 = []
    var_7 = []
    var_8 = bool(var_6 == [])
    assert var_8 is True
    var_9 = bool(var_7 == [])
    assert var_9 is True

def test_case_0():
    var_0 = 'Test find with skipped Python files.'
    var_1 = 'src'
    var_2 = 'file1.py'
    var_3 = 'file2.py'
    var_4 = '# file1'
    var_5 = '# file2'
    var_6 = 'file1'
    var_7 = []
    var_8 = []
    var_9 = len(var_7)
    assert var_9 == 1
    var_10 = 'file1.py'
    var_11 = bool('file1.py' in var_7[0])
    assert var_11 is True

def test_case_0():
    var_0 = 'Test find with a skipped directory.'
    var_1 = 'src'
    var_2 = 'skip_me'
    var_3 = 'file.py'
    var_4 = '# file'
    var_5 = []
    var_6 = []
    var_7 = len(var_5)
    assert var_7 == 1

def test_case_0():
    var_0 = 'Test find with a non-existent path.'
    var_1 = []
    var_2 = []
    var_3 = '/nonexistent/path/file.py'
    var_4 = [var_3]
    var_5 = len(var_2)
    assert var_5 == 1
    var_6 = var_2[0]
    assert var_6 == '/nonexistent/path/file.py'

def test_case_0():
    var_0 = 'Test find with unsupported file types.'
    var_1 = 'src'
    var_2 = 'file.txt'
    var_3 = '# text file'
    var_4 = 'file.py'
    var_5 = '# python file'
    var_6 = '.py'
    var_7 = []
    var_8 = []
    var_9 = 'file.py'

def test_case_0():
    var_0 = 'Test find with nested directory structure.'
    var_1 = 'src'
    var_2 = 'subdir'
    var_3 = 'file1.py'
    var_4 = '# file1'
    var_5 = 'file2.py'
    var_6 = '# file2'
    var_7 = []
    var_8 = []

def test_case_0():
    var_0 = 'Test find with multiple input paths.'
    var_1 = 'dir1'
    var_2 = 'dir2'
    var_3 = 'file1.py'
    var_4 = '# file1'
    var_5 = 'file2.py'
    var_6 = '# file2'
    var_7 = []
    var_8 = []



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_find_with_single_python_file. Retrieved 5/21 statements.
# Partially parsed test_find_with_directory_containing_python_files. Retrieved 9/30 statements.
# Partially parsed test_find_with_skipped_files. Retrieved 8/26 statements.
# Partially parsed test_find_with_nonexistent_path. Retrieved 5/13 statements.
# Partially parsed test_find_with_nested_directories. Retrieved 8/29 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 9/28 statements.


def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = False
    var_3 = []
    var_4 = []
    var_5 = bool(var_3 == [])
    assert var_5 is True
    var_6 = bool(var_4 == [])
    assert var_6 is True

def test_case_0():
    var_0 = 'file1.py'
    var_1 = "print('file1')"
    var_2 = 'file2.py'
    var_3 = "print('file2')"
    var_4 = 'file.txt'
    var_5 = 'not python'
    var_6 = False
    var_7 = []
    var_8 = []
    var_9 = bool(var_7 == [])
    assert var_9 is True
    var_10 = bool(var_8 == [])
    assert var_10 is True

def test_case_0():
    var_0 = 'file1.py'
    var_1 = "print('file1')"
    var_2 = 'file2.py'
    var_3 = "print('file2')"
    var_4 = False
    var_5 = []
    var_6 = []
    var_7 = len(var_5)
    assert var_7 == 1

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = '/nonexistent/path/file.py'
    var_4 = [var_3]
    var_5 = bool(var_1 == [])
    assert var_5 is True
    var_6 = bool(var_2 == ['/nonexistent/path/file.py'])
    assert var_6 is True

def test_case_0():
    var_0 = 'subdir'
    var_1 = 'file1.py'
    var_2 = "print('file1')"
    var_3 = 'file2.py'
    var_4 = "print('file2')"
    var_5 = False
    var_6 = []
    var_7 = []

def test_case_0():
    var_0 = 'skip_me'
    var_1 = 'file1.py'
    var_2 = "print('file1')"
    var_3 = 'file2.py'
    var_4 = "print('file2')"
    var_5 = False
    var_6 = []
    var_7 = []
    var_8 = len(var_6)
    assert var_8 == 1
    var_9 = 'skip_me'
    var_10 = bool('skip_me' in var_6[0])
    assert var_10 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_find_predicate_line_7. Retrieved 6/26 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 7 (for path in paths) evaluates to True.'
    var_1 = 'test.py'
    var_2 = '# test file'
    var_3 = []
    var_4 = []
    var_5 = len(var_4)
    assert var_5 == 0



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_find_predicate_line_7_evaluates_to_true. Retrieved 6/23 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 7 (for path in paths:) evaluates to True with valid paths.'
    var_1 = False
    var_2 = 'test.py'
    var_3 = '# test file'
    var_4 = []
    var_5 = []



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_find_predicate_line_7_evaluates_to_false. Retrieved 6/18 statements.


def test_case_0():
    var_0 = 'nonexistent_file.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = 0
    var_5 = var_1[var_4]



