####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_find_single_file. Retrieved 6/17 statements.
# Partially parsed test_find_broken_path. Retrieved 4/12 statements.
# Partially parsed test_find_with_skipping. Retrieved 12/31 statements.
# Partially parsed test_find_unsupported_filetype. Retrieved 10/26 statements.


def test_case_0():
    var_0 = 'test_file.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = ''
    var_5 = [var_4]
    var_6 = bool(var_2 == [])
    assert var_6 is True
    var_7 = bool(var_3 == [])
    assert var_7 is True

def test_case_0():
    var_0 = 'non_existent_file_12345.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = 'non_existent_file_12345.py'
    var_5 = bool('non_existent_file_12345.py' in var_3)
    assert var_5 is True

def test_case_0():
    var_0 = 'test_dir/skip_me'
    var_1 = True
    var_2 = ''
    var_3 = ''
    var_4 = 'test_dir'
    var_5 = [var_4]
    var_6 = []
    var_7 = []
    var_8 = [var_4]
    var_9 = 'test_dir/keep_me.py'
    var_10 = 'skip_me'
    var_11 = 'test_dir/keep_me.py'
    var_12 = 'test_dir/skip_me/ignore.py'

def test_case_0():
    var_0 = 'test_dir'
    var_1 = True
    var_2 = ''
    var_3 = ''
    var_4 = [var_3]
    var_5 = []
    var_6 = []
    var_7 = [var_3]
    var_8 = 'test_dir/valid.py'
    var_9 = 'test_dir/invalid.txt'
    var_10 = 'test_dir/valid.py'
    var_11 = 'test_dir/invalid.txt'



# Parsed testcases at query #2
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_0, var_2, var_3, var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [])
    assert var_7 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_find_with_single_valid_file. Retrieved 9/29 statements.


def test_case_0():
    var_0 = '/tmp/test_file.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = False
    var_5 = []
    var_6 = []
    var_7 = 'test_file.py'
    var_8 = [var_7]
    var_9 = 'test_file.py'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_find_iterates_over_paths. Retrieved 12/28 statements.


def test_case_0():
    var_0 = False
    var_1 = '/tmp/test_path_1'
    var_2 = '/tmp/test_path_2'
    var_3 = [var_1, var_2]
    var_4 = []
    var_5 = []
    var_6 = True
    var_7 = []
    var_8 = 'file.py'
    var_9 = [var_8]
    var_10 = (var_1, var_7, var_9)
    var_11 = [var_10]



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_find_iterates_over_paths. Retrieved 7/13 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '/fake/path'
    var_3 = [var_2]
    var_4 = '.'
    var_5 = [var_4]
    var_6 = None



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_find_returns_single_file_path. Retrieved 7/22 statements.
# Partially parsed test_find_adds_to_broken_when_path_does_not_exist. Retrieved 6/15 statements.
# Partially parsed test_find_skips_files_based_on_config. Retrieved 20/36 statements.


def test_case_0():
    var_0 = '/tmp/test_file.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = True
    var_5 = False
    var_6 = list(var_0)
    var_7 = bool(var_6 == ['/tmp/test_file.py'])
    assert var_7 is True

def test_case_0():
    var_0 = '/tmp/non_existent_path'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = False
    var_5 = list(var_0)
    var_6 = bool(var_5 == [])
    assert var_6 is True
    var_7 = '/tmp/non_existent_path'
    var_8 = bool('/tmp/non_existent_path' in var_3)
    assert var_8 is True

def test_case_0():
    var_0 = '/tmp/test_dir'
    var_1 = [var_0]
    var_2 = '/tmp/test_dir/skipped.py'
    var_3 = []
    var_4 = []
    var_5 = True
    var_6 = []
    var_7 = 'skipped.py'
    var_8 = 'valid.py'
    var_9 = [var_7, var_8]
    var_10 = (var_0, var_6, var_9)
    var_11 = [var_10]
    var_12 = list(var_0)
    var_13 = '/tmp/test_dir'
    var_14 = [var_13]
    var_15 = 'valid.py'
    var_16 = [var_15]
    var_17 = [os.path.join(p, f) for p in var_14 for f in var_16]
    var_18 = 'valid.py'
    var_19 = bool('valid.py' in var_17)
    assert var_19 is True
    var_20 = 'skipped.py'
    var_21 = any(var_10)
    var_22 = bool(var_21)
    assert var_22 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_find_evaluates_isdir_true. Retrieved 5/26 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = []
    var_4 = []
    var_5 = 0



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_find_broken_path_detection. Retrieved 6/12 statements.


def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = '/tmp/definitely_not_a_real_path_12345'
    var_5 = [var_4]
    var_6 = bool(var_4 in var_3)
    assert var_6 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_find_evaluates_true_for_directory. Retrieved 8/15 statements.


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
    var_10 = bool(True)
    assert var_10 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_find_broken_path_evaluation. Retrieved 5/19 statements.


def test_case_0():
    var_0 = '/non/existent/directory'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = bool(var_0 in var_3)
    assert var_4 is True
    var_5 = len(var_3)
    assert var_5 == 1



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_find_single_file. Retrieved 9/29 statements.
# Partially parsed test_find_non_existent_path. Retrieved 4/12 statements.
# Partially parsed test_find_with_skipping. Retrieved 15/31 statements.
# Partially parsed test_find_unsupported_filetype. Retrieved 11/25 statements.


import pathlib as module_0

def test_case_0():
    var_0 = 'test_script.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = var_3.absolute()
    var_5 = str(var_4)
    var_6 = [var_5]
    var_7 = []
    var_8 = []
    var_9 = len(var_7)
    assert var_9 == 0
    var_10 = len(var_8)
    assert var_10 == 0

def test_case_0():
    var_0 = '/tmp/non_existent_path_12345'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = '/tmp/non_existent_path_12345'
    var_5 = bool('/tmp/non_existent_path_12345' in var_3)
    assert var_5 is True

import pathlib as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = var_3.absolute()
    var_5 = True
    var_6 = 'valid.py'
    var_7 = var_4 / var_6
    var_8 = 'skip_me.py'
    var_9 = var_4 / var_8
    var_10 = str(var_4)
    var_11 = [var_10]
    var_12 = []
    var_13 = []
    var_14 = str(var_7)
    var_15 = str(var_9)
    var_16 = str(var_9)
    var_17 = bool(var_16 in var_12)
    assert var_17 is True

import pathlib as module_0

def test_case_0():
    var_0 = 'test_unsupported'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = var_3.absolute()
    var_5 = True
    var_6 = 'readme.txt'
    var_7 = var_4 / var_6
    var_8 = str(var_4)
    var_9 = [var_8]
    var_10 = []
    var_11 = []
    var_12 = len(var_10)
    assert var_12 == 0



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_find_single_file_exists. Retrieved 8/18 statements.
# Partially parsed test_find_directory_with_files_and_skipping. Retrieved 21/38 statements.
# Partially parsed test_find_with_broken_paths_mixed. Retrieved 9/17 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test_file.py'
    var_3 = [var_2]
    var_4 = ''
    var_5 = []
    var_6 = []
    var_7 = module_1.find(var_3, var_1, var_5, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == ['test_file.py'])
    assert var_9 is True
    var_10 = bool(var_5 == [])
    assert var_10 is True
    var_11 = bool(var_6 == [])
    assert var_11 is True

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
    var_9 = bool(var_5 == ['non_existent_path.py'])
    assert var_9 is True

import isort.settings as module_0
import pathlib as module_1
import isort.files as module_2

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'test_root'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_1.Path(*var_3, **var_4)
    var_6 = True
    var_7 = var_5.mkdir(exist_ok=var_6)
    var_8 = 'valid.py'
    var_9 = var_5 / var_8
    var_10 = 'ignored_dir'
    var_11 = var_5 / var_10
    var_12 = var_5 / var_10
    var_13 = 'file.py'
    var_14 = var_12 / var_13
    var_15 = 'invalid.txt'
    var_16 = var_5 / var_15
    var_17 = str(var_5)
    var_18 = [var_17]
    var_19 = []
    var_20 = []
    var_21 = module_2.find(var_18, var_1, var_19, var_20)
    var_22 = list(var_21)
    var_23 = len(var_22)
    assert var_23 == 1

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'exists.py'
    var_3 = 'does_not_exist.py'
    var_4 = [var_2, var_3]
    var_5 = ''
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_4, var_1, var_6, var_7)
    var_9 = list(var_8)
    var_10 = 'exists.py'
    var_11 = bool('exists.py' in var_9)
    assert var_11 is True
    var_12 = 'does_not_exist.py'
    var_13 = bool('does_not_exist.py' in var_7)
    assert var_13 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_find_adds_to_broken_when_path_does_not_exist. Retrieved 7/40 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '/tmp/this_path_should_not_exist_12345'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = []
    var_4 = []
    var_5 = [var_0]
    var_6 = module_1.find(var_5, var_2, var_3, var_4)
    var_7 = list(var_6)
    var_8 = bool(var_0 in var_4)
    assert var_8 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_find_single_file_exists. Retrieved 5/13 statements.
# Partially parsed test_find_broken_path. Retrieved 5/10 statements.
# Partially parsed test_find_directory_traversal_with_skipping. Retrieved 21/31 statements.


def test_case_0():
    var_0 = '/tmp/test_file.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = list(var_0)
    var_5 = bool(var_4 == ['/tmp/test_file.py'])
    assert var_5 is True

def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = list(var_0)
    var_5 = bool(var_4 == [])
    assert var_5 is True
    var_6 = '/non/existent/path'
    var_7 = bool('/non/existent/path' in var_3)
    assert var_7 is True

def test_case_0():
    var_0 = '/tmp/test_dir/skip_me'
    var_1 = '/tmp/test_dir'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = 'keep_me'
    var_6 = 'skip_me'
    var_7 = [var_5, var_6]
    var_8 = 'file1.py'
    var_9 = 'file2.txt'
    var_10 = [var_8, var_9]
    var_11 = (var_1, var_7, var_10)
    var_12 = '/tmp/test_dir/keep_me'
    var_13 = []
    var_14 = 'file3.py'
    var_15 = [var_14]
    var_16 = (var_12, var_13, var_15)
    var_17 = [var_11, var_16]
    var_18 = list(var_0)
    var_19 = '/tmp/test_dir/skip_me'
    var_20 = bool('/tmp/test_dir/skip_me' in var_3)
    assert var_20 is True
    var_21 = 'file1.py'
    var_22 = any(var_5)
    var_23 = bool(var_22)
    assert var_23 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_find_single_file_exists. Retrieved 13/24 statements.
# Partially parsed test_find_non_existent_path. Retrieved 4/16 statements.
# Partially parsed test_find_skipped_file. Retrieved 11/22 statements.
# Partially parsed test_find_unsupported_filetype. Retrieved 11/22 statements.


import pathlib as module_0

def test_case_0():
    var_0 = '/tmp/test_file.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = 'test_exists.py'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Path(*var_5, **var_6)
    var_8 = var_7.touch()
    var_9 = str(var_7)
    var_10 = [var_9]
    var_11 = str(var_7)
    var_12 = len(var_2)
    assert var_12 == 0
    var_13 = len(var_3)
    assert var_13 == 0
    var_14 = var_7.unlink()

def test_case_0():
    var_0 = '/tmp/non_existent_path_12345'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = '/tmp/non_existent_path_12345'
    var_5 = bool('/tmp/non_existent_path_12345' in var_3)
    assert var_5 is True

import pathlib as module_0

def test_case_0():
    var_0 = 'test_skip.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = var_3.touch()
    var_5 = var_3.absolute()
    var_6 = str(var_5)
    var_7 = [var_6]
    var_8 = []
    var_9 = []
    var_10 = var_3.absolute()
    var_11 = str(var_10)
    var_12 = bool(var_11 in var_8)
    assert var_12 is True
    var_13 = var_3.unlink()

import pathlib as module_0

def test_case_0():
    var_0 = 'test_unsupported.txt'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = var_3.touch()
    var_5 = var_3.absolute()
    var_6 = str(var_5)
    var_7 = [var_6]
    var_8 = []
    var_9 = []
    var_10 = len(var_8)
    assert var_10 == 0
    var_11 = len(var_9)
    assert var_11 == 0
    var_12 = var_3.unlink()



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_find_predicate_is_supported_and_not_skipped. Retrieved 4/21 statements.


def test_case_0():
    var_0 = 'test_file.py'
    var_1 = 'content'
    var_2 = []
    var_3 = []



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_find_predicate_true. Retrieved 4/23 statements.


def test_case_0():
    var_0 = 'test_file.py'
    var_1 = "print('hello')"
    var_2 = []
    var_3 = []



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_find_single_file_exists. Retrieved 7/13 statements.
# Partially parsed test_find_path_does_not_exist. Retrieved 5/8 statements.
# Partially parsed test_find_directory_traversal_and_skipping. Retrieved 19/28 statements.


def test_case_0():
    var_0 = '/tmp/test_file.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = list(var_0)
    var_5 = bool(var_4 == ['/tmp/test_file.py'])
    assert var_5 is True
    var_6 = len(var_2)
    assert var_6 == 0
    var_7 = len(var_3)
    assert var_7 == 0

def test_case_0():
    var_0 = '/tmp/non_existent.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = list(var_0)
    var_5 = bool(var_4 == [])
    assert var_5 is True
    var_6 = '/tmp/non_existent.py'
    var_7 = bool('/tmp/non_existent.py' in var_3)
    assert var_7 is True

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = 'dir_skipped'
    var_2 = 'skip_me.py'
    var_3 = '/tmp/root'
    var_4 = [var_3]
    var_5 = []
    var_6 = []
    var_7 = '/tmp/root'
    var_8 = 'dir_skipped'
    var_9 = [var_8]
    var_10 = 'test_file.py'
    var_11 = 'skip_me.py'
    var_12 = [var_10, var_11]
    var_13 = (var_7, var_9, var_12)
    var_14 = '/tmp/root/dir_skipped'
    var_15 = []
    var_16 = 'some_file.py'
    var_17 = [var_16]
    var_18 = (var_14, var_15, var_17)
    var_19 = '/tmp/root/test_file.py'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_find_predicate_true. Retrieved 4/20 statements.


def test_case_0():
    var_0 = 'test_file.py'
    var_1 = 'content'
    var_2 = []
    var_3 = []



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_find_detects_broken_path. Retrieved 4/13 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '/tmp/non_existent_directory_12345'
    var_3 = [var_2]
    var_4 = bool(var_2 in var_1)
    assert var_4 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_find_single_file_exists. Retrieved 4/21 statements.
# Partially parsed test_find_non_existent_path. Retrieved 4/13 statements.
# Partially parsed test_find_skipped_file. Retrieved 2/18 statements.
# Partially parsed test_find_unsupported_filetype. Retrieved 3/18 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = len(var_0)
    assert var_2 == 0
    var_3 = len(var_1)
    assert var_3 == 0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '/non/existent/path/at/all/12345'
    var_3 = [var_2]
    var_4 = '/non/existent/path/at/all/12345'
    var_5 = bool('/non/existent/path/at/all/12345' in var_1)
    assert var_5 is True

def test_case_0():
    var_0 = []
    var_1 = []

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = len(var_0)
    assert var_2 == 0



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_find_single_file_exists. Retrieved 5/13 statements.
# Partially parsed test_find_broken_path. Retrieved 5/11 statements.
# Partially parsed test_find_directory_with_files. Retrieved 17/28 statements.
# Partially parsed test_find_with_skipped_files. Retrieved 13/24 statements.


def test_case_0():
    var_0 = 'test.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = list(var_0)
    var_5 = bool(var_4 == ['test.py'])
    assert var_5 is True
    var_6 = bool(var_2 == [])
    assert var_6 is True
    var_7 = bool(var_3 == [])
    assert var_7 is True

def test_case_0():
    var_0 = 'non_existent.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = list(var_0)
    var_5 = bool(var_4 == [])
    assert var_5 is True
    var_6 = bool(var_3 == ['non_existent.py'])
    assert var_6 is True

def test_case_0():
    var_0 = '.py'
    var_1 = 'src'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = 'src'
    var_6 = 'sub'
    var_7 = [var_6]
    var_8 = 'main.py'
    var_9 = 'util.py'
    var_10 = [var_8, var_9]
    var_11 = (var_5, var_7, var_10)
    var_12 = 'src/sub'
    var_13 = []
    var_14 = 'sub.py'
    var_15 = [var_14]
    var_16 = (var_12, var_13, var_15)
    var_17 = 'src/main.py'
    var_18 = 'src/util.py'
    var_19 = 'src/sub/sub.py'

def test_case_0():
    var_0 = 'ignored.py'
    var_1 = 'src'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = 'src'
    var_6 = []
    var_7 = 'main.py'
    var_8 = 'ignored.py'
    var_9 = [var_7, var_8]
    var_10 = (var_5, var_6, var_9)
    var_11 = list(var_5)
    var_12 = 'src/main.py'
    var_13 = bool('src/main.py' in var_11)
    assert var_13 is True
    var_14 = 'src/ignored.py'
    var_15 = bool('src/ignored.py' not in var_11)
    assert var_15 is True
    var_16 = any(var_7)
    var_17 = bool(var_16)
    assert var_17 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_find_broken_path. Retrieved 6/20 statements.


def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = False
    var_5 = len(var_3)
    assert var_5 == 1
    var_6 = var_3[0]
    assert var_6 == '/non/existent/path'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_find_predicate_at_line_27_evaluates_to_true. Retrieved 5/16 statements.


def test_case_0():
    var_0 = '/tmp/test_dir'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = list(var_0)
    var_5 = '/tmp/test_dir/test_file.py'
    var_6 = bool('/tmp/test_dir/test_file.py' in var_2)
    assert var_6 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_find_broken_path. Retrieved 9/20 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '/non/existent/path'
    var_3 = [var_2]
    var_4 = []
    var_5 = []
    var_6 = '/tmp/this_path_should_not_exist_12345'
    var_7 = [var_6]
    var_8 = module_1.find(var_7, var_1, var_4, var_5)
    var_9 = list(var_8)
    var_10 = bool(var_6 in var_5)
    assert var_10 is True



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_find_single_file_exists. Retrieved 5/13 statements.
# Partially parsed test_find_broken_path. Retrieved 5/10 statements.
# Partially parsed test_find_directory_traversal_with_skipping. Retrieved 21/33 statements.
# Partially parsed test_find_unsupported_filetype. Retrieved 11/21 statements.


def test_case_0():
    var_0 = 'test_file.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = list(var_0)
    var_5 = bool(var_4 == ['test_file.py'])
    assert var_5 is True

def test_case_0():
    var_0 = 'non_existent_path.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = list(var_0)
    var_5 = bool(var_4 == [])
    assert var_5 is True
    var_6 = 'non_existent_path.py'
    var_7 = bool('non_existent_path.py' in var_3)
    assert var_7 is True

import pathlib as module_0

def test_case_0():
    var_0 = 'ignored_dir'
    var_1 = 'root_dir'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = 'root_dir'
    var_6 = 'ignored_dir'
    var_7 = [var_6]
    var_8 = 'file1.py'
    var_9 = [var_8]
    var_10 = (var_5, var_7, var_9)
    var_11 = 'root_dir/ignored_dir'
    var_12 = []
    var_13 = 'file2.py'
    var_14 = [var_13]
    var_15 = (var_11, var_12, var_14)
    var_16 = list(var_5)
    var_17 = 'file1.py'
    var_18 = bool('file1.py' in var_16)
    assert var_18 is True
    var_19 = 'root_dir/ignored_dir'
    var_20 = [var_19]
    var_21 = {}
    var_22 = module_0.Path(*var_20, **var_21)
    var_23 = str(var_22)
    var_24 = 'ignored_dir'
    var_25 = bool('ignored_dir' in var_23)
    assert var_25 is True
    var_26 = any(var_10)
    var_27 = bool(var_26)
    assert var_27 is True

def test_case_0():
    var_0 = 'root_dir'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = 'root_dir'
    var_5 = []
    var_6 = 'readme.txt'
    var_7 = 'script.py'
    var_8 = [var_6, var_7]
    var_9 = (var_4, var_5, var_8)
    var_10 = '.py'
    var_11 = 'root_dir/readme.txt'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_find_predicate_is_true. Retrieved 9/17 statements.


import pathlib as module_0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'test_dir_tmp'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Path(*var_3, **var_4)
    var_6 = True
    var_7 = var_5.mkdir(exist_ok=var_6)
    var_8 = str(var_5)
    var_9 = [var_8]
    var_10 = len(var_9)
    var_11 = bool(var_10 > 0)
    assert var_11 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_find_iterates_through_paths. Retrieved 12/22 statements.


import pathlib as module_0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'test_dir_tmp'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Path(*var_3, **var_4)
    var_6 = True
    var_7 = var_5.mkdir(exist_ok=var_6)
    var_8 = 'test_file.py'
    var_9 = var_5 / var_8
    var_10 = str(var_5)
    var_11 = [var_10]
    var_12 = str(var_5)
    var_13 = bool(var_12 in var_11)
    assert var_13 is True
    var_14 = var_5.rmdir()



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_find_single_file_exists. Retrieved 4/19 statements.
# Partially parsed test_find_non_existent_path. Retrieved 4/12 statements.
# Partially parsed test_find_skipping_file. Retrieved 2/17 statements.
# Partially parsed test_find_unsupported_filetype. Retrieved 5/25 statements.
# Partially parsed test_find_with_directory_and_files. Retrieved 7/29 statements.


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
    var_0 = '/non/existent/path/to/nothing'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = '/non/existent/path/to/nothing'
    var_5 = bool('/non/existent/path/to/nothing' in var_3)
    assert var_5 is True

def test_case_0():
    var_0 = []
    var_1 = []

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'test.txt'
    var_3 = ''
    var_4 = list(var_2)

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'script.py'
    var_3 = 'notes.txt'
    var_4 = ''
    var_5 = ''
    var_6 = list(var_3)



# Parsed testcases at query #5
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = []
    var_3 = []
    var_4 = '/tmp/test_dir_exists'
    var_5 = [var_4]
    var_6 = module_1.find(var_5, var_1, var_2, var_3)
    var_7 = list(var_6)
    var_8 = len(var_5)
    var_9 = bool(var_8 > 0)
    assert var_9 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_find_broken_path. Retrieved 6/16 statements.


def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = list(var_0)
    var_5 = len(var_4)
    assert var_5 == 0
    var_6 = '/non/existent/path'
    var_7 = bool('/non/existent/path' in var_3)
    assert var_7 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_find_predicate_true. Retrieved 13/26 statements.


import pathlib as module_0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'test_dir_tmp'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Path(*var_3, **var_4)
    var_6 = True
    var_7 = var_5.mkdir(exist_ok=var_6)
    var_8 = 'test_file.py'
    var_9 = var_5 / var_8
    var_10 = 'content'
    var_11 = str(var_5)
    var_12 = [var_11]
    var_13 = str(var_9)
    var_14 = var_5.rmdir()



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_find_single_file_exists. Retrieved 13/23 statements.
# Partially parsed test_find_directory_with_files_and_skipping. Retrieved 22/36 statements.
# Partially parsed test_find_unsupported_filetype. Retrieved 14/25 statements.


import pathlib as module_0
import isort.settings as module_1
import isort.files as module_2

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = var_3.absolute()
    var_5 = str(var_4)
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_1.Config(**var_7)
    var_9 = []
    var_10 = []
    var_11 = module_2.find(var_6, var_8, var_9, var_10)
    var_12 = list(var_11)
    var_13 = str(var_4)
    var_14 = [var_13]
    var_15 = bool(var_12 == var_14)
    assert var_15 is True
    var_16 = bool(var_9 == [])
    assert var_16 is True
    var_17 = bool(var_10 == [])
    assert var_17 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '/non/existent/path/to/nothing'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True
    var_9 = '/non/existent/path/to/nothing'
    var_10 = bool('/non/existent/path/to/nothing' in var_5)
    assert var_10 is True

import pathlib as module_0
import isort.settings as module_1
import isort.files as module_2

def test_case_0():
    var_0 = 'test_root'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = var_3.absolute()
    var_5 = 'sub_dir'
    var_6 = var_4 / var_5
    var_7 = 'skip_me'
    var_8 = var_4 / var_7
    var_9 = 'valid.py'
    var_10 = var_4 / var_9
    var_11 = 'ignored.py'
    var_12 = var_8 / var_11
    var_13 = True
    var_14 = str(var_4)
    var_15 = [var_14]
    var_16 = {}
    var_17 = module_1.Config(**var_16)
    var_18 = []
    var_19 = []
    var_20 = module_2.find(var_15, var_17, var_18, var_19)
    var_21 = list(var_20)
    var_22 = str(var_10)
    var_23 = bool(var_22 in var_21)
    assert var_23 is True
    var_24 = str(var_12)
    var_25 = bool(var_24 in var_18)
    assert var_25 is True
    var_26 = str(var_8)
    var_27 = bool(var_26 in var_18)
    assert var_27 is True

import pathlib as module_0
import isort.settings as module_1
import isort.files as module_2

def test_case_0():
    var_0 = 'test_dir_unsupported'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = var_3.absolute()
    var_5 = True
    var_6 = 'notes.txt'
    var_7 = var_4 / var_6
    var_8 = str(var_4)
    var_9 = [var_8]
    var_10 = {}
    var_11 = module_1.Config(**var_10)
    var_12 = []
    var_13 = []
    var_14 = module_2.find(var_9, var_11, var_12, var_13)
    var_15 = list(var_14)
    var_16 = bool(var_15 == [])
    assert var_16 is True
    var_17 = str(var_7)
    var_18 = bool(var_17 not in var_15)
    assert var_18 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_find_directory_traversal_and_skipping. Retrieved 32/42 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == ['test_file.py'])
    assert var_8 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'non_existent_path'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True
    var_9 = 'non_existent_path'
    var_10 = bool('non_existent_path' in var_5)
    assert var_10 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'src'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = 'main.py'
    var_7 = 'skip_me'
    var_8 = 'utils.py'
    var_9 = [var_6, var_7, var_8]
    var_10 = []
    var_11 = (var_0, var_9, var_10)
    var_12 = [var_11]
    var_13 = 'utils_dir'
    var_14 = [var_7, var_13]
    var_15 = 'other.txt'
    var_16 = [var_6, var_15]
    var_17 = (var_0, var_14, var_16)
    var_18 = 'src/skip_me'
    var_19 = []
    var_20 = 'hidden.py'
    var_21 = [var_20]
    var_22 = (var_18, var_19, var_21)
    var_23 = 'src/utils_dir'
    var_24 = []
    var_25 = 'helper.py'
    var_26 = [var_25]
    var_27 = (var_23, var_24, var_26)
    var_28 = [var_17, var_22, var_27]
    var_29 = module_1.find(var_1, var_3, var_4, var_5)
    var_30 = list(var_29)
    var_31 = 'src/main.py'
    var_32 = bool('src/main.py' in var_30)
    assert var_32 is True
    var_33 = 'skip_me'
    var_34 = any(var_7)
    var_35 = bool(var_34)
    assert var_35 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_find_single_file. Retrieved 8/19 statements.
# Partially parsed test_find_with_skipping. Retrieved 10/24 statements.
# Partially parsed test_find_unsupported_filetype. Retrieved 8/18 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = "print('hello')"
    var_7 = module_1.find(var_1, var_3, var_4, var_5)
    var_8 = list(var_7)
    var_9 = bool(var_8 == ['test_file.py'])
    assert var_9 is True
    var_10 = bool(var_4 == [])
    assert var_10 is True
    var_11 = bool(var_5 == [])
    assert var_11 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'non_existent_path_12345'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True
    var_9 = 'non_existent_path_12345'
    var_10 = bool('non_existent_path_12345' in var_5)
    assert var_10 is True
    var_11 = bool(var_4 == [])
    assert var_11 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = ''
    var_1 = ''
    var_2 = 'keep_me.py'
    var_3 = 'skip_me.py'
    var_4 = [var_2, var_3]
    var_5 = {}
    var_6 = module_0.Config(**var_5)
    var_7 = []
    var_8 = []
    var_9 = module_1.find(var_4, var_6, var_7, var_8)
    var_10 = list(var_9)
    var_11 = 'keep_me.py'
    var_12 = bool('keep_me.py' in var_10)
    assert var_12 is True
    var_13 = 'skip_me.py'
    var_14 = bool('skip_me.py' not in var_10)
    assert var_14 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = ''
    var_1 = 'test.txt'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = []
    var_6 = []
    var_7 = module_1.find(var_2, var_4, var_5, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [])
    assert var_9 is True
    var_10 = bool(var_5 == [])
    assert var_10 is True
    var_11 = bool(var_6 == [])
    assert var_11 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_find_single_file_exists. Retrieved 4/18 statements.
# Partially parsed test_find_broken_path. Retrieved 4/18 statements.
# Partially parsed test_find_directory_with_files. Retrieved 7/33 statements.
# Partially parsed test_find_skipping_files. Retrieved 6/35 statements.


def test_case_0():
    var_0 = '/tmp/test_file.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = bool(var_3 == [])
    assert var_4 is True
    var_5 = bool(var_2 == [])
    assert var_5 is True

def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = bool(var_3 == [var_0])
    assert var_4 is True

def test_case_0():
    var_0 = '/tmp/src'
    var_1 = 'script.py'
    var_2 = 'readme.txt'
    var_3 = ''
    var_4 = ''
    var_5 = []
    var_6 = []

def test_case_0():
    var_0 = 'valid.py'
    var_1 = 'ignored_file.py'
    var_2 = ''
    var_3 = ''
    var_4 = []
    var_5 = []



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_find_predicate_true. Retrieved 5/18 statements.


def test_case_0():
    var_0 = '/fake/dir'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = '/fake/dir/test.py'
    var_5 = '/fake/dir/test.py'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_find_single_file_exists. Retrieved 6/18 statements.
# Partially parsed test_find_broken_path. Retrieved 4/12 statements.
# Partially parsed test_find_directory_with_supported_files. Retrieved 14/31 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 13/30 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 14/31 statements.


import pathlib as module_0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'test_file.py'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Path(*var_3, **var_4)
    var_6 = var_5.absolute()
    var_7 = [var_2]
    var_8 = bool(var_0 == [])
    assert var_8 is True
    var_9 = bool(var_1 == [])
    assert var_9 is True

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
    var_2 = 'test_dir'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Path(*var_3, **var_4)
    var_6 = var_5.absolute()
    var_7 = True
    var_8 = 'script.py'
    var_9 = var_6 / var_8
    var_10 = 'readme.txt'
    var_11 = var_6 / var_10
    var_12 = str(var_6)
    var_13 = [var_12]
    var_14 = str(var_9)
    var_15 = str(var_11)

import pathlib as module_0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'parent_dir'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Path(*var_3, **var_4)
    var_6 = var_5.absolute()
    var_7 = 'skip_me_dir'
    var_8 = var_6 / var_7
    var_9 = 'file.py'
    var_10 = var_8 / var_9
    var_11 = True
    var_12 = str(var_6)
    var_13 = [var_12]
    var_14 = str(var_8)
    var_15 = bool(var_14 in var_0)
    assert var_15 is True

import pathlib as module_0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'test_root'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Path(*var_3, **var_4)
    var_6 = var_5.absolute()
    var_7 = True
    var_8 = 'good.py'
    var_9 = var_6 / var_8
    var_10 = 'ignored.py'
    var_11 = var_6 / var_10
    var_12 = str(var_6)
    var_13 = [var_12]
    var_14 = str(var_9)
    var_15 = str(var_11)
    var_16 = bool(var_15 in var_0)
    assert var_16 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_find_single_file_exists. Retrieved 7/22 statements.
# Partially parsed test_find_broken_path. Retrieved 6/17 statements.
# Partially parsed test_find_skipped_file. Retrieved 8/24 statements.
# Partially parsed test_find_directory_traversal. Retrieved 15/26 statements.


def test_case_0():
    var_0 = '/tmp/test_file.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = False
    var_5 = True
    var_6 = list(var_0)
    var_7 = bool(var_6 == ['/tmp/test_file.py'])
    assert var_7 is True

def test_case_0():
    var_0 = '/tmp/non_existent_path'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = False
    var_5 = list(var_0)
    var_6 = bool(var_5 == [])
    assert var_6 is True
    var_7 = '/tmp/non_existent_path'
    var_8 = bool('/tmp/non_existent_path' in var_3)
    assert var_8 is True

def test_case_0():
    var_0 = '/tmp/skip_me.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = False
    var_5 = True
    var_6 = list(var_0)
    var_7 = bool(var_6 == [])
    assert var_7 is True
    var_8 = '/tmp/skip_me.py'
    var_9 = bool(var_4 in var_2)
    assert var_9 is True

def test_case_0():
    var_0 = '/tmp/src'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = '/tmp/src'
    var_5 = 'subdir'
    var_6 = [var_5]
    var_7 = 'file1.py'
    var_8 = [var_7]
    var_9 = (var_4, var_6, var_8)
    var_10 = '/tmp/src/subdir'
    var_11 = []
    var_12 = 'file2.py'
    var_13 = [var_12]
    var_14 = (var_10, var_11, var_13)
    var_15 = '/tmp/src/file1.py'
    var_16 = '/tmp/src/subdir/file2.py'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_find_detects_broken_path. Retrieved 6/19 statements.


def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = False
    var_5 = list(var_0)
    var_6 = '/non/existent/path'
    var_7 = bool('/non/existent/path' in var_3)
    assert var_7 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_find_single_file. Retrieved 9/21 statements.
# Partially parsed test_find_broken_path. Retrieved 5/16 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 10/23 statements.
# Partially parsed test_find_unsupported_filetype. Retrieved 10/23 statements.


import pathlib as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = str(var_3)
    var_5 = [var_4]
    var_6 = []
    var_7 = []
    var_8 = str(var_3)
    var_9 = len(var_6)
    assert var_9 == 0
    var_10 = len(var_7)
    assert var_10 == 0

def test_case_0():
    var_0 = 'non_existent_path_999'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = bool(var_0 in var_3)
    assert var_4 is True
    var_5 = len(var_2)
    assert var_5 == 0

import pathlib as module_0

def test_case_0():
    var_0 = 'skipped_file.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'content'
    var_5 = var_3.write_text(var_4)
    var_6 = str(var_3)
    var_7 = [var_6]
    var_8 = []
    var_9 = []
    var_10 = var_3.absolute()
    var_11 = str(var_10)
    var_12 = bool(var_11 in var_8)
    assert var_12 is True

import pathlib as module_0

def test_case_0():
    var_0 = 'test.txt'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Path(*var_1, **var_2)
    var_4 = 'content'
    var_5 = var_3.write_text(var_4)
    var_6 = str(var_3)
    var_7 = [var_6]
    var_8 = []
    var_9 = []
    var_10 = len(var_8)
    assert var_10 == 0
    var_11 = len(var_9)
    assert var_11 == 0



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_find_predicate_true. Retrieved 4/20 statements.


def test_case_0():
    var_0 = 'test_file.py'
    var_1 = ''
    var_2 = []
    var_3 = []



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_find_single_file. Retrieved 4/18 statements.
# Partially parsed test_find_broken_path. Retrieved 4/14 statements.
# Partially parsed test_find_with_skipping. Retrieved 9/28 statements.
# Partially parsed test_find_unsupported_filetype. Retrieved 7/22 statements.


def test_case_0():
    var_0 = 'test_file.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = bool(var_2 == [])
    assert var_4 is True
    var_5 = bool(var_3 == [])
    assert var_5 is True

def test_case_0():
    var_0 = 'non_existent_path_12345'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = 'non_existent_path_12345'
    var_5 = bool('non_existent_path_12345' in var_3)
    assert var_5 is True

def test_case_0():
    var_0 = 'test_dir/sub_dir'
    var_1 = True
    var_2 = ''
    var_3 = ''
    var_4 = 'test_dir'
    var_5 = [var_4]
    var_6 = []
    var_7 = []
    var_8 = 'test_dir/valid.py'
    var_9 = 'test_dir/ignored.py'
    var_10 = 'ignored.py'

def test_case_0():
    var_0 = 'test_dir_unsupported'
    var_1 = True
    var_2 = ''
    var_3 = [var_2]
    var_4 = []
    var_5 = []
    var_6 = len(var_4)
    assert var_6 == 0



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_find_single_file_exists. Retrieved 10/18 statements.
# Partially parsed test_find_broken_path. Retrieved 4/10 statements.
# Partially parsed test_find_skipped_file. Retrieved 8/18 statements.
# Partially parsed test_find_unsupported_filetype. Retrieved 16/25 statements.


import pathlib as module_0

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = 'test_temp_exists.py'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Path(*var_5, **var_6)
    var_8 = "print('hello')"
    var_9 = var_7.write_text(var_8)
    var_10 = [var_4]
    var_11 = var_7.unlink()

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'non_existent_file_12345.py'
    var_3 = [var_2]
    var_4 = bool(var_1 == [var_2])
    assert var_4 is True

import pathlib as module_0

def test_case_0():
    var_0 = 'skip_me.py'
    var_1 = []
    var_2 = []
    var_3 = [var_0]
    var_4 = {}
    var_5 = module_0.Path(*var_3, **var_4)
    var_6 = 'pass'
    var_7 = var_5.write_text(var_6)
    var_8 = [var_0]
    var_9 = var_5.unlink()

import pathlib as module_0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'ignore_me.txt'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Path(*var_3, **var_4)
    var_6 = 'text content'
    var_7 = var_5.write_text(var_6)
    var_8 = 'test_dir_find'
    var_9 = [var_8]
    var_10 = {}
    var_11 = module_0.Path(*var_9, **var_10)
    var_12 = True
    var_13 = var_11.mkdir(exist_ok=var_12)
    var_14 = var_11 / var_2
    var_15 = 'content'
    var_16 = str(var_11)
    var_17 = [var_16]
    var_18 = var_11.rmdir()
    var_19 = var_5.unlink()



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_find_broken_path_adds_to_broken. Retrieved 6/21 statements.


def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = False
    var_5 = len(var_3)
    assert var_5 == 1
    var_6 = var_3[0]
    assert var_6 == '/non/existent/path'



