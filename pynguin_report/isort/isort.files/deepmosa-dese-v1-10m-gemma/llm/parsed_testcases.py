####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_find_direct_file_path. Retrieved 8/21 statements.
# Partially parsed test_find_skipped_file. Retrieved 4/24 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = 0
    var_6 = len(var_3)
    assert var_6 == 0
    var_7 = len(var_4)
    assert var_7 == 0

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'non_existent_file_12345.py'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_2, var_0, var_3, var_4)
    var_6 = list(var_5)
    var_7 = len(var_6)
    assert var_7 == 0

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 0
    var_2 = []
    var_3 = []



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_find_identifies_broken_path. Retrieved 4/14 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '/tmp/this_path_should_not_exist_12345'
    var_3 = [var_2]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_find_predicate_true_when_path_is_directory. Retrieved 5/10 statements.


def test_case_0():
    var_0 = '/tmp/test_dir'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_find_predicate_true. Retrieved 7/31 statements.


def test_case_0():
    var_0 = '/tmp/test_dir'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = 'test.py'
    var_5 = ''
    var_6 = 0



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_find_identifies_broken_path. Retrieved 10/21 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = '/non/existent/path'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = 0
    var_6 = var_2[var_5]
    var_7 = module_1.find(var_2, var_0, var_3, var_4)
    var_8 = list(var_7)
    var_9 = len(var_4)
    assert var_9 == 1



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_find_predicate_true. Retrieved 5/24 statements.


def test_case_0():
    var_0 = 'test_dir_temp'
    var_1 = True
    var_2 = 'test_file.py'
    var_3 = []
    var_4 = []



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_find_predicate_is_true. Retrieved 4/22 statements.


def test_case_0():
    var_0 = 'test_file.py'
    var_1 = 'content'
    var_2 = []
    var_3 = []



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_find_single_file. Retrieved 12/23 statements.
# Partially parsed test_find_with_skipping. Retrieved 10/23 statements.
# Partially parsed test_find_unsupported_filetype. Retrieved 10/21 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_file.py'
    var_2 = [var_1]
    var_3 = ''
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_2, var_0, var_4, var_5)
    var_7 = list(var_6)
    var_8 = False
    var_9 = 'test_all_file.py'
    var_10 = 'all_'
    var_11 = ''

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'non_existent_path_12345.py'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_2, var_0, var_3, var_4)
    var_6 = list(var_5)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = ''
    var_2 = ''
    var_3 = 'keep.py'
    var_4 = 'skip.py'
    var_5 = [var_3, var_4]
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_5, var_0, var_6, var_7)
    var_9 = list(var_8)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = ''
    var_2 = ''
    var_3 = 'test.txt'
    var_4 = 'test.py'
    var_5 = [var_3, var_4]
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_5, var_0, var_6, var_7)
    var_9 = list(var_8)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_find_is_supported_filetype_true. Retrieved 10/24 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = '/tmp/test_dir'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = True
    var_6 = 'test.py'
    var_7 = ''
    var_8 = module_1.find(var_2, var_0, var_3, var_4)
    var_9 = list(var_8)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_find_single_file_exists. Retrieved 7/22 statements.
# Partially parsed test_find_broken_path. Retrieved 6/18 statements.
# Partially parsed test_find_skipped_file. Retrieved 14/33 statements.
# Partially parsed test_find_unsupported_filetype. Retrieved 16/32 statements.


def test_case_0():
    var_0 = '/tmp/test_file.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = False
    var_5 = True
    var_6 = list(var_0)

def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = False
    var_5 = list(var_0)

def test_case_0():
    var_0 = 'skip.py'
    var_1 = '/tmp/dir'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = True
    var_6 = []
    var_7 = 'keep.py'
    var_8 = [var_0, var_7]
    var_9 = (var_1, var_6, var_8)
    var_10 = [var_9]
    var_11 = list(var_0)
    var_12 = 'skip.py'
    var_13 = any(var_5)

def test_case_0():
    var_0 = '.py'
    var_1 = '/tmp/dir'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = True
    var_6 = []
    var_7 = 'file.txt'
    var_8 = 'script.py'
    var_9 = [var_7, var_8]
    var_10 = (var_1, var_6, var_9)
    var_11 = [var_10]
    var_12 = list(var_0)
    var_13 = '/tmp/dir'
    var_14 = 'script.py'
    var_15 = [var_6]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_find_path_does_not_exist. Retrieved 7/44 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = '/tmp/this_path_definitely_does_not_exist_12345'
    var_4 = [var_3]
    var_5 = module_1.find(var_4, var_0, var_1, var_2)
    var_6 = list(var_5)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_find_predicate_true. Retrieved 8/31 statements.


def test_case_0():
    var_0 = 'test_root'
    var_1 = 'test_file.py'
    var_2 = True
    var_3 = []
    var_4 = [var_1]
    var_5 = [var_0]
    var_6 = []
    var_7 = []



# Parsed testcases at query #13
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = '/non/existent/path/to/file'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_2, var_0, var_3, var_4)
    var_6 = list(var_5)
    var_7 = len(var_6)
    assert var_7 == 0



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_find_is_supported_filetype_true. Retrieved 4/22 statements.


def test_case_0():
    var_0 = 'test_file.py'
    var_1 = ''
    var_2 = []
    var_3 = []



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_find_predicate_is_true. Retrieved 5/25 statements.


def test_case_0():
    var_0 = 'test_dir_temp'
    var_1 = tuple()
    var_2 = 'test_file.py'
    var_3 = []
    var_4 = []



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_find_adds_to_broken_when_path_does_not_exist. Retrieved 7/17 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = '/tmp/non_existent_directory_path_12345'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_2, var_0, var_3, var_4)
    var_6 = list(var_5)



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

# Partially parsed test_find_broken_path. Retrieved 5/15 statements.


def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = False



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_find_single_file_exists. Retrieved 5/19 statements.
# Partially parsed test_find_invalid_path_adds_to_broken. Retrieved 4/9 statements.
# Partially parsed test_find_file_is_skipped. Retrieved 4/20 statements.
# Partially parsed test_find_directory_traversal_with_supported_files. Retrieved 7/28 statements.
# Partially parsed test_find_directory_skipping_subdirs. Retrieved 8/34 statements.


def test_case_0():
    var_0 = '/tmp/test_file.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = b"print('hello')"

def test_case_0():
    var_0 = '/non/existent/path/at/all'
    var_1 = [var_0]
    var_2 = []
    var_3 = []

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = b"print('hello')"

def test_case_0():
    var_0 = '.py'
    var_1 = 'test1.py'
    var_2 = 'test2.txt'
    var_3 = ''
    var_4 = ''
    var_5 = []
    var_6 = []

def test_case_0():
    var_0 = 'skip_me'
    var_1 = 'keep'
    var_2 = 'keep.py'
    var_3 = 'skip.py'
    var_4 = ''
    var_5 = ''
    var_6 = []
    var_7 = []



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_find_single_file_exists. Retrieved 5/11 statements.
# Partially parsed test_find_broken_path. Retrieved 5/8 statements.
# Partially parsed test_find_directory_traversal_and_skipping. Retrieved 24/32 statements.
# Partially parsed test_find_supported_files_yielded. Retrieved 5/11 statements.
# Partially parsed test_find_skipping_logic_for_files. Retrieved 7/13 statements.


def test_case_0():
    var_0 = '/tmp/test_file.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = list(var_0)

def test_case_0():
    var_0 = '/tmp/non_existent_path'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = list(var_0)

def test_case_0():
    var_0 = '.py'
    var_1 = 'skip'
    var_2 = '/tmp/test_dir'
    var_3 = [var_2]
    var_4 = []
    var_5 = []
    var_6 = 'subdir'
    var_7 = 'skip_dir'
    var_8 = [var_6, var_7]
    var_9 = 'file1.py'
    var_10 = 'file2.py'
    var_11 = [var_9, var_10]
    var_12 = (var_2, var_8, var_11)
    var_13 = '/tmp/test_dir/subdir'
    var_14 = []
    var_15 = 'sub_file.py'
    var_16 = [var_15]
    var_17 = (var_13, var_14, var_16)
    var_18 = '/tmp/test_dir/skip_dir'
    var_19 = []
    var_20 = 'hidden.py'
    var_21 = [var_20]
    var_22 = (var_18, var_19, var_21)
    var_23 = [var_12, var_17, var_22]

def test_case_0():
    var_0 = '/tmp/test_file.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = list(var_0)

def test_case_0():
    var_0 = '/tmp/test_file.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = '/tmp/skip_me.py'
    var_5 = [var_4]
    var_6 = len(var_2)
    assert var_6 == 0



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_find_single_file_exists. Retrieved 5/11 statements.
# Partially parsed test_find_broken_path. Retrieved 5/8 statements.
# Partially parsed test_find_directory_traversal_and_skipping. Retrieved 23/31 statements.


def test_case_0():
    var_0 = '/tmp/test_file.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = list(var_0)

def test_case_0():
    var_0 = '/tmp/non_existent_path'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = list(var_0)

def test_case_0():
    var_0 = '/tmp/src'
    var_1 = [var_0]
    var_2 = 'ignored'
    var_3 = []
    var_4 = []
    var_5 = 'subdir'
    var_6 = [var_2, var_5]
    var_7 = 'file1.py'
    var_8 = [var_7]
    var_9 = (var_0, var_6, var_8)
    var_10 = '/tmp/src/subdir'
    var_11 = []
    var_12 = 'file2.py'
    var_13 = [var_12]
    var_14 = (var_10, var_11, var_13)
    var_15 = '/tmp/src/ignored'
    var_16 = []
    var_17 = 'hidden.py'
    var_18 = [var_17]
    var_19 = (var_15, var_16, var_18)
    var_20 = [var_9, var_14, var_19]
    var_21 = list(var_0)
    var_22 = [os.path.join(var_0, f) for f in var_21]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_find_single_file_exists. Retrieved 6/18 statements.
# Partially parsed test_find_broken_path. Retrieved 5/12 statements.
# Partially parsed test_find_directory_traversal_and_skipping. Retrieved 28/46 statements.


def test_case_0():
    var_0 = '/tmp/test_file.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = '/tmp/non_existent_path'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = False

def test_case_0():
    var_0 = 'unsupported'
    var_1 = 'skip'
    var_2 = '/tmp/test_dir'
    var_3 = [var_2]
    var_4 = []
    var_5 = []
    var_6 = True
    var_7 = 'skip_dir'
    var_8 = 'valid_dir'
    var_9 = [var_7, var_8]
    var_10 = 'file1.py'
    var_11 = 'unsupported.txt'
    var_12 = [var_10, var_11]
    var_13 = (var_2, var_9, var_12)
    var_14 = '/tmp/test_dir/valid_dir'
    var_15 = []
    var_16 = 'file2.py'
    var_17 = [var_16]
    var_18 = (var_14, var_15, var_17)
    var_19 = '/tmp/test_dir/skip_dir'
    var_20 = []
    var_21 = 'file3.py'
    var_22 = [var_21]
    var_23 = (var_19, var_20, var_22)
    var_24 = [var_13, var_18, var_23]
    var_25 = [var_2]
    var_26 = [var_10, var_11]
    var_27 = [os.path.join(p, f) for p in var_25 for f in var_26]



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_find_single_file. Retrieved 9/24 statements.
# Partially parsed test_find_broken_path. Retrieved 8/21 statements.
# Partially parsed test_find_skipped_file. Retrieved 12/31 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test.py'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = False
    var_6 = True
    var_7 = module_1.find(var_2, var_0, var_3, var_4)
    var_8 = list(var_7)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'non_existent.py'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = False
    var_6 = module_1.find(var_2, var_0, var_3, var_4)
    var_7 = list(var_6)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'dir/skip_me.py'
    var_2 = 'dir/keep_me.py'
    var_3 = [var_1, var_2]
    var_4 = []
    var_5 = []
    var_6 = 'keep_me.py'
    var_7 = 'skip_me.py'
    var_8 = ''
    var_9 = ''
    var_10 = module_1.find(var_3, var_0, var_4, var_5)
    var_11 = list(var_10)



# Parsed testcases at query #6
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = '/non/existent/path'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_2, var_0, var_3, var_4)
    var_6 = list(var_5)
    var_7 = len(var_6)
    assert var_7 == 0



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_find_single_file_exists. Retrieved 5/11 statements.
# Partially parsed test_find_non_existent_path. Retrieved 5/8 statements.
# Partially parsed test_find_directory_with_files. Retrieved 16/22 statements.
# Partially parsed test_find_skips_files. Retrieved 6/14 statements.


def test_case_0():
    var_0 = '/tmp/test_file.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = list(var_0)

def test_case_0():
    var_0 = '/tmp/non_existent.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = list(var_0)

def test_case_0():
    var_0 = '/tmp/test_dir'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = 'subdir'
    var_5 = [var_4]
    var_6 = 'file1.py'
    var_7 = [var_6]
    var_8 = (var_0, var_5, var_7)
    var_9 = '/tmp/test_dir/subdir'
    var_10 = []
    var_11 = 'file2.py'
    var_12 = [var_11]
    var_13 = (var_9, var_10, var_12)
    var_14 = [var_8, var_13]
    var_15 = list(var_0)

def test_case_0():
    var_0 = '/tmp/test_dir/file1.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = '/tmp/test_dir'
    var_5 = [var_4]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_find_predicate_is_true. Retrieved 5/15 statements.


def test_case_0():
    var_0 = '/tmp/test_dir'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = list(var_0)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_find_single_file. Retrieved 5/16 statements.
# Partially parsed test_find_broken_path. Retrieved 5/16 statements.
# Partially parsed test_find_directory_with_files. Retrieved 17/28 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 22/33 statements.


def test_case_0():
    var_0 = 'test_file.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = list(var_0)

def test_case_0():
    var_0 = 'non_existent.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = list(var_0)

def test_case_0():
    var_0 = 'src'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = 'subdir'
    var_5 = [var_4]
    var_6 = 'main.py'
    var_7 = 'README.md'
    var_8 = [var_6, var_7]
    var_9 = (var_0, var_5, var_8)
    var_10 = 'src/subdir'
    var_11 = []
    var_12 = 'utils.py'
    var_13 = [var_12]
    var_14 = (var_10, var_11, var_13)
    var_15 = [var_9, var_14]
    var_16 = list(var_0)

def test_case_0():
    var_0 = 'src'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = 'ignored_dir'
    var_5 = 'valid_dir'
    var_6 = [var_4, var_5]
    var_7 = 'main.py'
    var_8 = [var_7]
    var_9 = (var_0, var_6, var_8)
    var_10 = 'src/ignored_dir'
    var_11 = []
    var_12 = 'hidden.py'
    var_13 = [var_12]
    var_14 = (var_10, var_11, var_13)
    var_15 = 'src/valid_dir'
    var_16 = []
    var_17 = 'sub.py'
    var_18 = [var_17]
    var_19 = (var_15, var_16, var_18)
    var_20 = [var_9, var_14, var_19]
    var_21 = list(var_0)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_find_single_file_exists. Retrieved 6/22 statements.
# Partially parsed test_find_broken_path. Retrieved 5/18 statements.
# Partially parsed test_find_skipped_file. Retrieved 12/30 statements.


def test_case_0():
    var_0 = '/tmp/test_file.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = True
    var_5 = False

def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = False

def test_case_0():
    var_0 = '/tmp/test_file.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = True
    var_5 = '/tmp'
    var_6 = []
    var_7 = 'test_file.py'
    var_8 = [var_7]
    var_9 = (var_5, var_6, var_8)
    var_10 = [var_9]
    var_11 = lambda x: x



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_find_broken_path. Retrieved 5/16 statements.


def test_case_0():
    var_0 = '/non/existent/path'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = '/tmp/definitely_not_exists_'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_find_single_file_exists. Retrieved 5/11 statements.
# Partially parsed test_find_broken_path. Retrieved 5/9 statements.
# Partially parsed test_find_directory_traversal_and_skipping. Retrieved 24/36 statements.


def test_case_0():
    var_0 = '/tmp/test_file.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = list(var_0)

def test_case_0():
    var_0 = '/tmp/non_existent_path'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = list(var_0)

def test_case_0():
    var_0 = '.py'
    var_1 = 'skip'
    var_2 = '/tmp/root'
    var_3 = [var_2]
    var_4 = []
    var_5 = []
    var_6 = '/tmp/root'
    var_7 = 'skip_dir'
    var_8 = 'normal_dir'
    var_9 = [var_7, var_8]
    var_10 = 'valid.py'
    var_11 = 'skip_me.py'
    var_12 = [var_10, var_11]
    var_13 = (var_6, var_9, var_12)
    var_14 = '/tmp/root/normal_dir'
    var_15 = []
    var_16 = 'another_valid.py'
    var_17 = [var_16]
    var_18 = (var_14, var_15, var_17)
    var_19 = '/tmp/root/skip_dir'
    var_20 = []
    var_21 = 'hidden.py'
    var_22 = [var_21]
    var_23 = (var_19, var_20, var_22)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_find_single_file_exists. Retrieved 5/15 statements.
# Partially parsed test_find_broken_path. Retrieved 4/10 statements.
# Partially parsed test_find_directory_walking_with_skipping. Retrieved 19/28 statements.
# Partially parsed test_find_unsupported_filetype. Retrieved 4/12 statements.


def test_case_0():
    var_0 = '/tmp/test_file.py'
    var_1 = '/tmp/test_file.py'
    var_2 = [var_1]
    var_3 = []
    var_4 = []

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '/non/existent/path'
    var_3 = [var_2]

def test_case_0():
    var_0 = '/tmp/test/skip_me'
    var_1 = []
    var_2 = []
    var_3 = '/tmp/test'
    var_4 = 'keep_me'
    var_5 = 'skip_me'
    var_6 = [var_4, var_5]
    var_7 = 'file1.py'
    var_8 = [var_7]
    var_9 = (var_3, var_6, var_8)
    var_10 = '/tmp/test/keep_mock'
    var_11 = []
    var_12 = 'file2.py'
    var_13 = [var_12]
    var_14 = (var_10, var_11, var_13)
    var_15 = [var_9, var_14]
    var_16 = '/tmp/test'
    var_17 = [var_16]
    var_18 = list(var_4)

def test_case_0():
    var_0 = '/tmp/test'
    var_1 = [var_0]
    var_2 = []
    var_3 = []



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_find_not_exists_path. Retrieved 4/13 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '/tmp/non_existent_directory_abc_123'
    var_3 = [var_2]



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_find_predicate_is_true. Retrieved 6/25 statements.


def test_case_0():
    var_0 = '/tmp/test_dir'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = 'test.py'
    var_5 = ''



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_find_single_file_exists. Retrieved 4/14 statements.
# Partially parsed test_find_non_existent_path_adds_to_broken. Retrieved 5/11 statements.
# Partially parsed test_find_supported_file_yields_path. Retrieved 5/13 statements.
# Partially parsed test_find_skipped_file_adds_to_skipped. Retrieved 11/20 statements.


def test_case_0():
    var_0 = 'test_file.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []

def test_case_0():
    var_0 = 'non_existent.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = list(var_0)

def test_case_0():
    var_0 = 'existing_file.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = list(var_0)

def test_case_0():
    var_0 = 'some_dir'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = 'some_dir'
    var_5 = []
    var_6 = 'file1.py'
    var_7 = [var_6]
    var_8 = (var_4, var_5, var_7)
    var_9 = list(var_4)
    var_10 = len(var_9)
    assert var_10 == 0



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_find_single_file_exists. Retrieved 5/11 statements.
# Partially parsed test_find_non_existent_path. Retrieved 5/8 statements.
# Partially parsed test_find_directory_traversal_and_skipping. Retrieved 27/33 statements.


def test_case_0():
    var_0 = '/tmp/test_file.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = list(var_0)

def test_case_0():
    var_0 = '/tmp/non_existent.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = list(var_0)

def test_case_0():
    var_0 = '.py'
    var_1 = 'skip_me'
    var_2 = '/tmp/test_dir'
    var_3 = [var_2]
    var_4 = []
    var_5 = []
    var_6 = 'subdir'
    var_7 = 'skip_me_dir'
    var_8 = [var_6, var_7]
    var_9 = 'file1.py'
    var_10 = 'file2.txt'
    var_11 = [var_9, var_10]
    var_12 = (var_2, var_8, var_11)
    var_13 = '/tmp/test_dir/subdir'
    var_14 = []
    var_15 = 'file3.py'
    var_16 = [var_15]
    var_17 = (var_13, var_14, var_16)
    var_18 = '/tmp/test_dir/skip_me_dir'
    var_19 = []
    var_20 = 'ignored.py'
    var_21 = [var_20]
    var_22 = (var_18, var_19, var_21)
    var_23 = [var_12, var_17, var_22]
    var_24 = list(var_0)
    var_25 = 'skip_me'
    var_26 = any(var_2)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_find_path_does_not_exist. Retrieved 7/18 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '/tmp/this_path_should_not_exist_12345'
    var_3 = 'fallback'
    var_4 = [var_2]
    var_5 = list(var_3)
    var_6 = len(var_5)
    assert var_6 == 0



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_find_predicate_true. Retrieved 7/28 statements.


def test_case_0():
    var_0 = '/mock/dir'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = 'test_file.py'
    var_5 = ''
    var_6 = 'test_file.py'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_find_single_file_exists. Retrieved 7/19 statements.
# Partially parsed test_find_broken_path. Retrieved 6/16 statements.
# Partially parsed test_find_skipped_file_in_directory. Retrieved 7/17 statements.


def test_case_0():
    var_0 = '/tmp/test_file.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = False
    var_5 = True
    var_6 = list(var_0)

def test_case_0():
    var_0 = '/tmp/non_existent_path'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = False
    var_5 = list(var_0)

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = '/tmp/test_dir'
    var_3 = [var_2]
    var_4 = []
    var_5 = []
    var_6 = list(var_0)



