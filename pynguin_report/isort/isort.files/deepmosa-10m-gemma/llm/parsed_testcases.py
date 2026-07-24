####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_find_skipped_file_in_directory. Retrieved 7/31 statements.
# Partially parsed test_find_unsupported_filetype. Retrieved 5/24 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test_file.py'
    var_3 = [var_2]
    var_4 = []
    var_5 = []
    var_6 = [var_2]
    var_7 = module_1.find(var_6, var_1, var_4, var_5)
    var_8 = list(var_7)
    var_9 = bool(var_8 == ['test_file.py'])
    assert var_9 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'non_existent_path.py'
    var_3 = [var_2]
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_3, var_1, var_4, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True
    var_9 = 'non_existent_path.py'
    var_10 = bool('non_existent_path.py' in var_5)
    assert var_10 is True

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'valid.py'
    var_3 = 'skipped_file.py'
    var_4 = ''
    var_5 = ''
    var_6 = []
    var_7 = []

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test.txt'
    var_3 = ''
    var_4 = []
    var_5 = []



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_find_path_does_not_exist. Retrieved 4/13 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '/tmp/non_existent_directory_12345'
    var_3 = [var_2]
    var_4 = bool(var_2 in var_1)
    assert var_4 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_find_single_file_exists. Retrieved 4/15 statements.
# Partially parsed test_find_path_does_not_exist. Retrieved 4/11 statements.
# Partially parsed test_find_directory_traversal_with_skipping. Retrieved 23/34 statements.


def test_case_0():
    var_0 = '/mock/file.py'
    var_1 = []
    var_2 = []
    var_3 = [var_0]
    var_4 = bool(var_1 == [])
    assert var_4 is True
    var_5 = bool(var_2 == [])
    assert var_5 is True

def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = []
    var_2 = []
    var_3 = [var_0]
    var_4 = bool(var_2 == [var_0])
    assert var_4 is True

def test_case_0():
    var_0 = 'skip_me'
    var_1 = '.py'
    var_2 = []
    var_3 = []
    var_4 = '/root'
    var_5 = 'skip_me'
    var_6 = 'sub'
    var_7 = [var_5, var_6]
    var_8 = 'a.py'
    var_9 = 'ignore.txt'
    var_10 = [var_8, var_9]
    var_11 = (var_4, var_7, var_10)
    var_12 = '/root/sub'
    var_13 = []
    var_14 = 'b.py'
    var_15 = [var_14]
    var_16 = (var_12, var_13, var_15)
    var_17 = '/root/skip_me'
    var_18 = []
    var_19 = 'c.py'
    var_20 = [var_19]
    var_21 = (var_17, var_18, var_20)
    var_22 = [var_4]
    var_23 = '/root/a.py'
    var_24 = '/root/sub/b.py'
    var_25 = '/root/skip_me'
    var_26 = bool('/root/skip_me' in var_2)
    assert var_26 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_find_path_is_directory. Retrieved 2/14 statements.


def test_case_0():
    var_0 = []
    var_1 = []



# Parsed testcases at query #5
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '/tmp/non_existent_path_9999'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_0 in var_5)
    assert var_8 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_find_single_file_exists. Retrieved 10/23 statements.
# Partially parsed test_find_broken_path. Retrieved 5/10 statements.
# Partially parsed test_find_directory_with_supported_files. Retrieved 17/26 statements.
# Partially parsed test_find_with_real_temp_dir. Retrieved 6/29 statements.
# Partially parsed test_find_skipping_file. Retrieved 4/24 statements.


def test_case_0():
    var_0 = 'test_file.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = 'test_file.py'
    var_5 = var_0 / var_4
    var_6 = "print('hello')"
    var_7 = '/fake/path/file.py'
    var_8 = [var_7]
    var_9 = list(var_0)
    var_10 = bool(var_9 == ['/fake/path/file.py'])
    assert var_10 is True
    var_11 = bool(var_2 == [])
    assert var_11 is True
    var_12 = bool(var_3 == [])
    assert var_12 is True

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '/non/existent/path'
    var_3 = [var_2]
    var_4 = list(var_2)
    var_5 = bool(var_4 == [])
    assert var_5 is True
    var_6 = bool(var_1 == ['/non/existent/path'])
    assert var_6 is True

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '/fake/dir'
    var_3 = 'subdir'
    var_4 = [var_3]
    var_5 = 'file1.py'
    var_6 = 'file2.txt'
    var_7 = [var_5, var_6]
    var_8 = (var_2, var_4, var_7)
    var_9 = '/fake/dir/subdir'
    var_10 = []
    var_11 = 'file3.py'
    var_12 = [var_11]
    var_13 = (var_9, var_10, var_12)
    var_14 = [var_8, var_13]
    var_15 = '/fake/dir'
    var_16 = [var_15]

def test_case_0():
    var_0 = 'test1.py'
    var_1 = 'content'
    var_2 = 'test2.txt'
    var_3 = '.py'
    var_4 = []
    var_5 = []

def test_case_0():
    var_0 = 'skip.py'
    var_1 = 'content'
    var_2 = []
    var_3 = []



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_find_predicate_is_true_when_path_is_directory. Retrieved 4/18 statements.


def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = 0



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_find_path_is_directory. Retrieved 4/23 statements.


def test_case_0():
    var_0 = 'test_subdir'
    var_1 = []
    var_2 = []
    var_3 = 0



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_find_path_does_not_exist. Retrieved 7/20 statements.


def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = False
    var_5 = list(var_0)
    var_6 = len(var_5)
    assert var_6 == 0
    var_7 = '/non/existent/path'
    var_8 = bool('/non/existent/path' in var_3)
    assert var_8 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_find_evaluates_isdir_to_true. Retrieved 7/15 statements.


def test_case_0():
    var_0 = '/mock/directory'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = 'test_dir'
    var_5 = True
    var_6 = [var_4]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_find_broken_path_evaluates_true. Retrieved 5/11 statements.


def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = list(var_0)
    var_5 = var_1[0]
    var_6 = bool(var_1[0] in var_3)
    assert var_6 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_find_predicate_evaluates_to_true. Retrieved 4/20 statements.


def test_case_0():
    var_0 = 'test_directory'
    var_1 = False
    var_2 = []
    var_3 = []



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_find_broken_path_evaluation. Retrieved 4/13 statements.


def test_case_0():
    var_0 = '/tmp/non_existent_path_12345'
    var_1 = []
    var_2 = []
    var_3 = [var_0]
    var_4 = bool(var_0 in var_2)
    assert var_4 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_find_single_file_path. Retrieved 4/14 statements.
# Partially parsed test_find_non_existent_path. Retrieved 4/9 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 2/15 statements.
# Partially parsed test_find_unsupported_filetype. Retrieved 2/15 statements.


def test_case_0():
    var_0 = '/tmp/test_file.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = bool(var_2 == [])
    assert var_4 is True
    var_5 = bool(var_3 == [])
    assert var_5 is True

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '/tmp/definitely_not_exists_12345'
    var_3 = [var_2]
    var_4 = bool(var_1 == [var_2])
    assert var_4 is True

def test_case_0():
    var_0 = []
    var_1 = []

def test_case_0():
    var_0 = []
    var_1 = []



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_find_evaluates_isdir_true. Retrieved 8/15 statements.


import pathlib as module_0

def test_case_0():
    var_0 = 'test_directory_exists'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = str(var_3)
    var_7 = [var_6]
    var_8 = []
    var_9 = []



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_find_single_file_exists. Retrieved 6/16 statements.
# Partially parsed test_find_non_existent_path. Retrieved 4/8 statements.
# Partially parsed test_find_directory_with_files. Retrieved 16/33 statements.
# Partially parsed test_find_skipping_files. Retrieved 18/35 statements.


def test_case_0():
    var_0 = '/tmp/test_file.py'
    var_1 = ''
    var_2 = [var_0]
    var_3 = []
    var_4 = []
    var_5 = list(var_1)
    var_6 = bool(var_5 == [var_0])
    assert var_6 is True
    var_7 = bool(var_3 == [])
    assert var_7 is True
    var_8 = bool(var_4 == [])
    assert var_8 is True

def test_case_0():
    var_0 = '/tmp/non_existent_path_12345'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = bool(var_3 == [var_0])
    assert var_4 is True

import pathlib as module_0

def test_case_0():
    var_0 = '/tmp/test_dir_find'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(parents=var_4, exist_ok=var_4)
    var_6 = 'file1.py'
    var_7 = var_3 / var_6
    var_8 = 'file2.txt'
    var_9 = var_3 / var_8
    var_10 = 'content'
    var_11 = '.py'
    var_12 = str(var_3)
    var_13 = [var_12]
    var_14 = []
    var_15 = []
    var_16 = str(var_7)
    var_17 = str(var_9)

import pathlib as module_0

def test_case_0():
    var_0 = '/tmp/test_skip_dir'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(parents=var_4, exist_ok=var_4)
    var_6 = 'file1.py'
    var_7 = var_3 / var_6
    var_8 = 'content'
    var_9 = 'skip'
    var_10 = str(var_3)
    var_11 = [var_10]
    var_12 = []
    var_13 = []
    var_14 = 'skip_folder'
    var_15 = var_3 / var_14
    var_16 = 'file2.py'
    var_17 = var_15 / var_16
    var_18 = str(var_7)
    var_19 = str(var_15)
    var_20 = bool(var_19 in var_12)
    assert var_20 is True



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_find_single_file_exists. Retrieved 4/17 statements.
# Partially parsed test_find_non_existent_path. Retrieved 4/12 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 2/16 statements.
# Partially parsed test_find_with_unsupported_filetype. Retrieved 2/14 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '/tmp/test_file.py'
    var_3 = [var_2]
    var_4 = bool(var_0 == [])
    assert var_4 is True
    var_5 = bool(var_1 == [])
    assert var_5 is True

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '/non/existent/path/to/nothing'
    var_3 = [var_2]
    var_4 = bool(var_2 in var_1)
    assert var_4 is True

def test_case_0():
    var_0 = []
    var_1 = []

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = bool(var_0 == [])
    assert var_2 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_find_skips_iteration_when_paths_is_empty. Retrieved 3/9 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = bool(var_0 == [])
    assert var_3 is True
    var_4 = bool(var_1 == [])
    assert var_4 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_find_enters_loop_with_paths. Retrieved 8/27 statements.


import pathlib as module_0

def test_case_0():
    var_0 = '/tmp/test_dir'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Path(*var_4, **var_5)
    var_7 = True
    var_8 = var_6.mkdir(parents=var_7, exist_ok=var_7)
    var_9 = len(var_1)
    var_10 = bool(var_9 > 0)
    assert var_10 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_find_predicate_is_false_when_paths_is_empty. Retrieved 3/8 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = bool(var_1 == [])
    assert var_3 is True
    var_4 = bool(var_2 == [])
    assert var_4 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_find_enters_loop_when_paths_is_not_empty. Retrieved 4/16 statements.


def test_case_0():
    var_0 = '/some/path'
    var_1 = [var_0]
    var_2 = []
    var_3 = []



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_find_predicate_false_when_paths_is_empty. Retrieved 3/9 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = bool(var_1 == [])
    assert var_3 is True
    var_4 = bool(var_2 == [])
    assert var_4 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_find_single_file_exists. Retrieved 5/16 statements.
# Partially parsed test_find_broken_path. Retrieved 4/9 statements.
# Partially parsed test_find_skipped_file. Retrieved 5/18 statements.
# Partially parsed test_find_unsupported_filetype. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 'test_file.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = "print('hello')"
    var_5 = bool(var_2 == [])
    assert var_5 is True
    var_6 = bool(var_3 == [])
    assert var_6 is True

def test_case_0():
    var_0 = 'non_existent_path_12345.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = 'non_existent_path_12345.py'
    var_5 = bool('non_existent_path_12345.py' in var_3)
    assert var_5 is True

def test_case_0():
    var_0 = 'skip_me.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = ''

def test_case_0():
    var_0 = 'test.txt'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = 'not a python file'
    var_5 = 'test.txt'
    var_6 = bool('test.txt' not in var_2)
    assert var_6 is True



# Parsed testcases at query #8
#--------------------------

# Failed to parse test_find_predicate_evaluates_to_true.




# Parsed testcases at query #9
#--------------------------

# Partially parsed test_find_single_file_exists. Retrieved 7/17 statements.
# Partially parsed test_find_non_existent_path_adds_to_broken. Retrieved 4/12 statements.
# Partially parsed test_find_skips_file_if_config_says_so. Retrieved 9/19 statements.
# Partially parsed test_find_filters_by_filetype. Retrieved 11/21 statements.


import pathlib as module_0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'test_file.py'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Path(*var_3, **var_4)
    var_6 = var_5.touch()
    var_7 = [var_2]
    var_8 = bool(var_0 == [])
    assert var_8 is True
    var_9 = bool(var_1 == [])
    assert var_9 is True
    var_10 = var_5.unlink()

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'non_existent_path_12345'
    var_3 = [var_2]
    var_4 = bool(var_1 == [var_2])
    assert var_4 is True

import pathlib as module_0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'skip_me.py'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Path(*var_3, **var_4)
    var_6 = var_5.touch()
    var_7 = [var_2]
    var_8 = var_5.absolute()
    var_9 = str(var_8)
    var_10 = bool(var_9 in var_0)
    assert var_10 is True
    var_11 = var_5.unlink()

import pathlib as module_0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'valid.py'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Path(*var_3, **var_4)
    var_6 = 'invalid.txt'
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_0.Path(*var_7, **var_8)
    var_10 = var_5.touch()
    var_11 = var_9.touch()
    var_12 = [var_2, var_6]
    var_13 = 'valid.py'
    var_14 = 'invalid.txt'
    var_15 = var_5.unlink()
    var_16 = var_9.unlink()



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_find_predicate_is_false_when_paths_is_empty. Retrieved 3/9 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_find_predicate_false_with_empty_paths. Retrieved 3/9 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = bool(var_1 == [])
    assert var_3 is True
    var_4 = bool(var_2 == [])
    assert var_4 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_find_single_file_exists. Retrieved 7/18 statements.
# Partially parsed test_find_non_existent_path. Retrieved 4/15 statements.


import pathlib as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Path(*var_4, **var_5)
    var_7 = var_6.touch()
    var_8 = bool(var_2 == [])
    assert var_8 is True
    var_9 = bool(var_3 == [])
    assert var_9 is True
    var_10 = var_6.unlink()

def test_case_0():
    var_0 = 'non_existent_path_12345.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = bool(var_3 == ['non_existent_path_12345.py'])
    assert var_4 is True

def test_case_0():
    pass



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_find_skips_loop_iteration_when_paths_is_empty. Retrieved 3/9 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = bool(var_1 == [])
    assert var_3 is True
    var_4 = bool(var_2 == [])
    assert var_4 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_find_evaluates_predicate_true_with_directory. Retrieved 11/24 statements.


import pathlib as module_0

def test_case_0():
    var_0 = 'test_dir_exists'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = 'test_file.py'
    var_7 = var_3 / var_6
    var_8 = str(var_3)
    var_9 = [var_8]
    var_10 = []
    var_11 = []
    var_12 = str(var_3)
    var_13 = bool(var_12 in var_9)
    assert var_13 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_find_enters_loop_with_valid_path. Retrieved 4/18 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = None
    var_3 = 0



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_find_single_file_exists. Retrieved 5/16 statements.
# Partially parsed test_find_non_existent_path. Retrieved 5/16 statements.
# Partially parsed test_find_directory_with_files. Retrieved 16/29 statements.
# Partially parsed test_find_skipping_files. Retrieved 11/22 statements.


def test_case_0():
    var_0 = '/tmp/test_file.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = list(var_0)
    var_5 = bool(var_4 == ['/tmp/test_file.py'])
    assert var_5 is True
    var_6 = bool(var_2 == [])
    assert var_6 is True
    var_7 = bool(var_3 == [])
    assert var_7 is True

def test_case_0():
    var_0 = '/tmp/non_existent.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = list(var_0)
    var_5 = bool(var_4 == [])
    assert var_5 is True
    var_6 = bool(var_3 == ['/tmp/non_existent.py'])
    assert var_6 is True

def test_case_0():
    var_0 = '/tmp/src'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = 'subdir'
    var_5 = [var_4]
    var_6 = 'main.py'
    var_7 = 'README.md'
    var_8 = [var_6, var_7]
    var_9 = (var_0, var_5, var_8)
    var_10 = '/tmp/src/subdir'
    var_11 = []
    var_12 = 'utils.py'
    var_13 = [var_12]
    var_14 = (var_10, var_11, var_13)
    var_15 = [var_9, var_14]

def test_case_0():
    var_0 = '/tmp/src'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = 'main.py'
    var_6 = 'skipped.py'
    var_7 = [var_5, var_6]
    var_8 = (var_0, var_4, var_7)
    var_9 = [var_8]
    var_10 = list(var_0)
    var_11 = '/tmp/src/main.py'
    var_12 = bool('/tmp/src/main.py' in var_10)
    assert var_12 is True
    var_13 = '/tmp/src/skipped.py'
    var_14 = bool('/tmp/src/skipped.py' not in var_10)
    assert var_14 is True
    var_15 = '/tmp/src/skipped.py'
    var_16 = bool('/tmp/src/skipped.py' in var_2)
    assert var_16 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_find_single_file_path. Retrieved 4/16 statements.
# Partially parsed test_find_non_existent_path. Retrieved 4/9 statements.
# Partially parsed test_find_skipped_file. Retrieved 2/15 statements.
# Partially parsed test_find_unsupported_filetype. Retrieved 4/21 statements.


def test_case_0():
    var_0 = '/tmp/test_file.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = bool(var_2 == [])
    assert var_4 is True
    var_5 = bool(var_3 == [])
    assert var_5 is True

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '/non/existent/path/to/nothing'
    var_3 = [var_2]
    var_4 = '/non/existent/path/to/nothing'
    var_5 = bool('/non/existent/path/to/nothing' in var_1)
    assert var_5 is True

def test_case_0():
    var_0 = []
    var_1 = []

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'test.txt'
    var_3 = 'content'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_find_single_file_exists. Retrieved 5/13 statements.
# Partially parsed test_find_broken_path. Retrieved 5/11 statements.
# Partially parsed test_find_directory_walking_and_skipping. Retrieved 21/30 statements.


def test_case_0():
    var_0 = '/tmp/test_file.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = list(var_0)
    var_5 = bool(var_4 == ['/tmp/test_file.py'])
    assert var_5 is True
    var_6 = bool(var_2 == [])
    assert var_6 is True
    var_7 = bool(var_3 == [])
    assert var_7 is True

def test_case_0():
    var_0 = '/tmp/non_existent_path'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = list(var_0)
    var_5 = bool(var_4 == [])
    assert var_5 is True
    var_6 = bool(var_3 == ['/tmp/non_existent_path'])
    assert var_6 is True

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = '/tmp/test_dir'
    var_3 = [var_2]
    var_4 = []
    var_5 = []
    var_6 = 'subdir'
    var_7 = [var_6]
    var_8 = 'file1.py'
    var_9 = 'file2.txt'
    var_10 = [var_8, var_9]
    var_11 = (var_2, var_7, var_10)
    var_12 = '/tmp/test_dir/subdir'
    var_13 = []
    var_14 = 'file3.py'
    var_15 = [var_14]
    var_16 = (var_12, var_13, var_15)
    var_17 = [var_11, var_16]
    var_18 = False
    var_19 = True
    var_20 = list(var_2)
    var_21 = '/tmp/test_dir/file1.py'
    var_22 = bool('/tmp/test_dir/file1.py' in var_20)
    assert var_22 is True
    var_23 = '/tmp/test_dir/file2.txt'
    var_24 = bool('/tmp/test_dir/file2.txt' in var_4)
    assert var_24 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_find_path_is_directory. Retrieved 2/14 statements.


def test_case_0():
    var_0 = []
    var_1 = []



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_find_predicate_evaluates_to_true_when_path_is_directory. Retrieved 5/16 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = []
    var_4 = []
    var_5 = 0



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_find_single_file_exists. Retrieved 5/11 statements.
# Partially parsed test_find_broken_path. Retrieved 5/8 statements.
# Partially parsed test_find_directory_with_supported_files. Retrieved 17/22 statements.
# Partially parsed test_find_skips_files_and_directories. Retrieved 13/18 statements.


def test_case_0():
    var_0 = '/tmp/test_file.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = list(var_0)
    var_5 = bool(var_4 == ['/tmp/test_file.py'])
    assert var_5 is True

def test_case_0():
    var_0 = '/tmp/non_existent_path'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = list(var_0)
    var_5 = bool(var_4 == [])
    assert var_5 is True
    var_6 = '/tmp/non_existent_path'
    var_7 = bool('/tmp/non_existent_path' in var_3)
    assert var_7 is True

def test_case_0():
    var_0 = '/tmp/test_dir'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = 'subdir'
    var_5 = [var_4]
    var_6 = 'file1.py'
    var_7 = 'file2.txt'
    var_8 = [var_6, var_7]
    var_9 = (var_0, var_5, var_8)
    var_10 = '/tmp/test_dir/subdir'
    var_11 = []
    var_12 = 'file3.py'
    var_13 = [var_12]
    var_14 = (var_10, var_11, var_13)
    var_15 = [var_9, var_14]
    var_16 = list(var_0)
    var_17 = '/tmp/test_dir/file1.py'
    var_18 = bool('/tmp/test_dir/file1.py' in var_16)
    assert var_18 is True
    var_19 = '/tmp/test_dir/file3.py'
    var_20 = bool('/tmp/test_dir/file3.py' in var_16)
    assert var_20 is True

def test_case_0():
    var_0 = '/tmp/test_dir'
    var_1 = [var_0]
    var_2 = '/tmp/test_dir/skip_me'
    var_3 = []
    var_4 = []
    var_5 = 'skip_me'
    var_6 = 'keep_me'
    var_7 = [var_5, var_6]
    var_8 = 'file1.py'
    var_9 = [var_8]
    var_10 = (var_0, var_7, var_9)
    var_11 = [var_10]
    var_12 = list(var_0)
    var_13 = '/tmp/test_dir/file1.py'
    var_14 = bool('/tmp/test_dir/file1.py' in var_12)
    assert var_14 is True
    var_15 = '/tmp/test_dir/skip_me'
    var_16 = bool('/tmp/test_dir/skip_me' in var_3)
    assert var_16 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_find_path_is_directory_evaluates_true. Retrieved 9/17 statements.


import pathlib as module_0

def test_case_0():
    var_0 = 'test_dir_exists'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = str(var_3)
    var_7 = [var_6]
    var_8 = []
    var_9 = []
    var_10 = str(var_3)
    var_11 = bool(True)
    assert var_11 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_find_evaluates_isdir_true. Retrieved 11/18 statements.


import pathlib as module_0

def test_case_0():
    var_0 = 'test_dir_exists'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = str(var_3)
    var_7 = [var_6]
    var_8 = []
    var_9 = []
    var_10 = 0
    var_11 = var_7[var_10]
    var_12 = var_3.rmdir()



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_find_path_is_directory_evaluates_true. Retrieved 12/19 statements.


import pathlib as module_0

def test_case_0():
    var_0 = 'test_dir_tmp'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = str(var_3)
    var_7 = [var_6]
    var_8 = []
    var_9 = []
    var_10 = None
    var_11 = var_3.rmdir()
    var_12 = 0
    var_13 = var_7[var_12]



