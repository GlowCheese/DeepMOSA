####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_find_with_single_file. Retrieved 8/11 statements.
# Partially parsed test_find_with_directory_containing_python_files. Retrieved 13/21 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 10/17 statements.
# Partially parsed test_find_with_broken_symlink. Retrieved 11/15 statements.


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

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '# test'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = 'test_file.py'
    var_4 = [var_3]
    var_5 = []
    var_6 = []
    var_7 = module_1.find(var_4, var_2, var_5, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == ['test_file.py'])
    assert var_9 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'non_existent_path.py'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Config(**var_4)
    var_6 = module_1.find(var_3, var_5, var_0, var_1)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True
    var_9 = bool(var_1 == ['non_existent_path.py'])
    assert var_9 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = True
    var_2 = '# test1'
    var_3 = '# test2'
    var_4 = {}
    var_5 = module_0.Config(**var_4)
    var_6 = [var_3]
    var_7 = []
    var_8 = []
    var_9 = module_1.find(var_6, var_5, var_7, var_8)
    var_10 = list(var_9)
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = 'test_dir/file1.py'
    var_13 = bool('test_dir/file1.py' in var_10)
    assert var_13 is True
    var_14 = 'test_dir/file2.py'
    var_15 = bool('test_dir/file2.py' in var_10)
    assert var_15 is True
    var_16 = 'test_dir/file1.py'
    var_17 = 'test_dir/file2.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = True
    var_2 = '# skipped'
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = []
    var_6 = [var_2]
    var_7 = []
    var_8 = module_1.find(var_6, var_4, var_5, var_7)
    var_9 = list(var_8)
    var_10 = bool(var_9 == [])
    assert var_10 is True
    var_11 = 'test_dir/skipped.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = True
    var_2 = 'non_existent.py'
    var_3 = 'test_dir/link.py'
    var_4 = 'follow_links'
    var_5 = {var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = []
    var_8 = []
    var_9 = [var_0]
    var_10 = module_1.find(var_9, var_6, var_7, var_8)
    var_11 = list(var_10)
    var_12 = bool(var_11 == [])
    assert var_12 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_find_with_single_file. Retrieved 8/11 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 9/15 statements.
# Partially parsed test_find_with_directory. Retrieved 11/19 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 11/20 statements.
# Partially parsed test_find_with_symlink. Retrieved 11/18 statements.
# Partially parsed test_find_with_broken_symlink. Retrieved 8/10 statements.


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

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = []
    var_1 = 'nonexistent_path.py'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = []
    var_6 = module_1.find(var_2, var_4, var_5, var_0)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True
    var_9 = bool(var_0 == ['nonexistent_path.py'])
    assert var_9 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '# test'
    var_1 = 'test_file.py'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = []
    var_6 = []
    var_7 = module_1.find(var_2, var_4, var_5, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == ['test_file.py'])
    assert var_9 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '# test'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = 'skipped_*'
    var_4 = []
    var_5 = 'skipped_file.py'
    var_6 = [var_5]
    var_7 = []
    var_8 = module_1.find(var_6, var_2, var_4, var_7)
    var_9 = list(var_8)
    var_10 = bool(var_9 == [])
    assert var_10 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = '# test'
    var_2 = '# test'
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = [var_2]
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_5, var_4, var_6, var_7)
    var_9 = list(var_8)
    var_10 = 'test_dir/file1.py'
    var_11 = bool('test_dir/file1.py' in var_9)
    assert var_11 is True
    var_12 = 'test_dir/file2.txt'
    var_13 = bool('test_dir/file2.txt' not in var_9)
    assert var_13 is True
    var_14 = 'test_dir/file1.py'
    var_15 = 'test_dir/file2.txt'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'skipped_dir/subdir'
    var_1 = '# test'
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = 'skipped_*'
    var_5 = []
    var_6 = 'skipped_dir'
    var_7 = [var_6]
    var_8 = []
    var_9 = module_1.find(var_7, var_3, var_5, var_8)
    var_10 = list(var_9)
    var_11 = bool(var_10 == [])
    assert var_11 is True
    var_12 = 'skipped_dir/file.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'real_dir'
    var_1 = '# test'
    var_2 = 'symlink_dir'
    var_3 = True
    var_4 = 'follow_links'
    var_5 = {var_4: var_3}
    var_6 = module_0.Config(**var_5)
    var_7 = [var_2]
    var_8 = []
    var_9 = []
    var_10 = module_1.find(var_7, var_6, var_8, var_9)
    var_11 = list(var_10)
    var_12 = 'symlink_dir/file.py'
    var_13 = bool('symlink_dir/file.py' in var_11)
    assert var_13 is True
    var_14 = 'real_dir/file.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = 'broken_link'
    var_2 = []
    var_3 = [var_1]
    var_4 = {}
    var_5 = module_0.Config(**var_4)
    var_6 = []
    var_7 = module_1.find(var_3, var_5, var_6, var_2)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [])
    assert var_9 is True
    var_10 = bool(var_2 == ['broken_link'])
    assert var_10 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_isdir_predicate_evaluates_to_true. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'test_directory'
    var_1 = True



# Parsed testcases at query #4
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_path'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_5 == ['nonexistent_path'])
    assert var_8 is True



# Parsed testcases at query #5
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_file.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = bool(var_5 == ['nonexistent_file.py'])
    assert var_7 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_find_with_single_file. Retrieved 8/11 statements.
# Partially parsed test_find_with_directory_containing_python_files. Retrieved 11/19 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 10/15 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 8/14 statements.
# Partially parsed test_find_with_follow_links_disabled. Retrieved 12/18 statements.
# Partially parsed test_find_with_follow_links_enabled. Retrieved 12/18 statements.
# Partially parsed test_find_with_unsupported_filetype. Retrieved 8/11 statements.


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

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = []
    var_1 = 'nonexistent_path'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = []
    var_6 = module_1.find(var_2, var_4, var_5, var_0)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True
    var_9 = bool(var_0 == ['nonexistent_path'])
    assert var_9 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '# test'
    var_1 = 'test_file.py'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = []
    var_6 = []
    var_7 = module_1.find(var_2, var_4, var_5, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == ['test_file.py'])
    assert var_9 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = '# test1'
    var_2 = '# test2'
    var_3 = 'text'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Config(**var_5)
    var_7 = []
    var_8 = []
    var_9 = module_1.find(var_4, var_6, var_7, var_8)
    var_10 = list(var_9)
    var_11 = set(var_10)
    var_12 = bool(var_11 == {'test_dir/file1.py', 'test_dir/file2.py'})
    assert var_12 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir/skip_dir'
    var_1 = '# test'
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = 'skip_dir'
    var_5 = []
    var_6 = 'test_dir'
    var_7 = [var_6]
    var_8 = []
    var_9 = module_1.find(var_7, var_3, var_5, var_8)
    var_10 = list(var_9)
    var_11 = bool(var_10 == [])
    assert var_11 is True
    var_12 = bool(var_5 == ['test_dir/skip_dir'])
    assert var_12 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '# test'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = 'test_file.py'
    var_4 = []
    var_5 = [var_3]
    var_6 = []
    var_7 = module_1.find(var_5, var_2, var_4, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [])
    assert var_9 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_dir/subdir'
    var_2 = 'subdir'
    var_3 = 'test_dir/link'
    var_4 = '# test'
    var_5 = False
    var_6 = 'follow_links'
    var_7 = {var_6: var_5}
    var_8 = module_0.Config(**var_7)
    var_9 = [var_4]
    var_10 = []
    var_11 = []
    var_12 = module_1.find(var_9, var_8, var_10, var_11)
    var_13 = list(var_12)
    var_14 = 'test_dir/link/file.py'
    var_15 = bool('test_dir/link/file.py' not in var_13)
    assert var_15 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_dir/subdir'
    var_2 = 'subdir'
    var_3 = 'test_dir/link'
    var_4 = '# test'
    var_5 = True
    var_6 = 'follow_links'
    var_7 = {var_6: var_5}
    var_8 = module_0.Config(**var_7)
    var_9 = [var_4]
    var_10 = []
    var_11 = []
    var_12 = module_1.find(var_9, var_8, var_10, var_11)
    var_13 = list(var_12)
    var_14 = 'test_dir/link/file.py'
    var_15 = bool('test_dir/link/file.py' in var_13)
    assert var_15 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'text'
    var_1 = 'test_file.txt'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = []
    var_6 = []
    var_7 = module_1.find(var_2, var_4, var_5, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [])
    assert var_9 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_isdir_predicate_evaluates_to_true. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'some_path'



# Parsed testcases at query #8
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = False
    var_3 = []
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = 'follow_links'
    var_7 = 'skipped_paths'
    var_8 = 'supported_filetypes'
    var_9 = {var_6: var_2, var_7: var_3, var_8: var_5}
    var_10 = module_0.Config(**var_9)
    var_11 = []
    var_12 = []
    var_13 = module_1.find(var_1, var_10, var_11, var_12)
    var_14 = list(var_13)
    var_15 = 'test_dir/file1.py'
    var_16 = bool('test_dir/file1.py' in var_14)
    assert var_16 is True
    var_17 = 'test_dir/subdir/file2.py'
    var_18 = bool('test_dir/subdir/file2.py' in var_14)
    assert var_18 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = False
    var_3 = 'test_dir/skip_dir'
    var_4 = 'test_dir/skip_file.py'
    var_5 = [var_3, var_4]
    var_6 = '.py'
    var_7 = [var_6]
    var_8 = 'follow_links'
    var_9 = 'skipped_paths'
    var_10 = 'supported_filetypes'
    var_11 = {var_8: var_2, var_9: var_5, var_10: var_7}
    var_12 = module_0.Config(**var_11)
    var_13 = []
    var_14 = []
    var_15 = module_1.find(var_1, var_12, var_13, var_14)
    var_16 = list(var_15)
    var_17 = 'test_dir/skip_file.py'
    var_18 = bool('test_dir/skip_file.py' not in var_16)
    assert var_18 is True
    var_19 = 'test_dir/skip_dir/file.py'
    var_20 = bool('test_dir/skip_dir/file.py' not in var_16)
    assert var_20 is True
    var_21 = 'test_dir/skip_dir'
    var_22 = bool('test_dir/skip_dir' in var_13)
    assert var_22 is True
    var_23 = 'test_dir/skip_file.py'
    var_24 = bool('test_dir/skip_file.py' in var_13)
    assert var_24 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_path'
    var_1 = 'test_dir'
    var_2 = [var_0, var_1]
    var_3 = False
    var_4 = []
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = 'follow_links'
    var_8 = 'skipped_paths'
    var_9 = 'supported_filetypes'
    var_10 = {var_7: var_3, var_8: var_4, var_9: var_6}
    var_11 = module_0.Config(**var_10)
    var_12 = []
    var_13 = []
    var_14 = module_1.find(var_2, var_11, var_12, var_13)
    var_15 = list(var_14)
    var_16 = 'nonexistent_path'
    var_17 = bool('nonexistent_path' in var_13)
    assert var_17 is True
    var_18 = 'test_dir/file1.py'
    var_19 = bool('test_dir/file1.py' in var_15)
    assert var_19 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = [var_0]
    var_2 = False
    var_3 = []
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = 'follow_links'
    var_7 = 'skipped_paths'
    var_8 = 'supported_filetypes'
    var_9 = {var_6: var_2, var_7: var_3, var_8: var_5}
    var_10 = module_0.Config(**var_9)
    var_11 = []
    var_12 = []
    var_13 = module_1.find(var_1, var_10, var_11, var_12)
    var_14 = list(var_13)
    var_15 = 'test_file.py'
    var_16 = bool('test_file.py' in var_14)
    assert var_16 is True



# Parsed testcases at query #9
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_path'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = len(var_5)
    assert var_8 == 1
    var_9 = var_5[0]
    assert var_9 == 'nonexistent_path'
    var_10 = len(var_4)
    assert var_10 == 0
    var_11 = len(var_7)
    assert var_11 == 0



# Parsed testcases at query #10
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_file.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = len(var_5)
    assert var_8 == 1
    var_9 = var_5[0]
    assert var_9 == 'nonexistent_file.py'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_path_is_directory. Retrieved 1/2 statements.


def test_case_0():
    var_0 = '/path/to/existing/directory'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_isdir_predicate_evaluates_to_true. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 'test_directory'
    var_1 = True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_find_with_skipped_file. Retrieved 8/9 statements.


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
    var_8 = bool(var_3 == [])
    assert var_8 is True
    var_9 = bool(var_4 == [])
    assert var_9 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_path.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True
    var_9 = bool(var_4 == [])
    assert var_9 is True
    var_10 = bool(var_5 == ['nonexistent_path.py'])
    assert var_10 is True

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
    var_9 = bool(var_4 == [])
    assert var_9 is True
    var_10 = bool(var_5 == [])
    assert var_10 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_directory'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == ['test_directory/file1.py', 'test_directory/file2.py'])
    assert var_8 is True
    var_9 = bool(var_4 == [])
    assert var_9 is True
    var_10 = bool(var_5 == [])
    assert var_10 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_directory'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = 'file1.py'
    var_5 = []
    var_6 = []
    var_7 = module_1.find(var_1, var_3, var_5, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == ['test_directory/file2.py'])
    assert var_9 is True
    var_10 = bool(var_5 == ['test_directory/file1.py'])
    assert var_10 is True
    var_11 = bool(var_6 == [])
    assert var_11 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'broken_symlink'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True
    var_9 = bool(var_4 == [])
    assert var_9 is True
    var_10 = bool(var_5 == ['broken_symlink'])
    assert var_10 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_isdir_predicate_evaluates_to_true. Retrieved 6/15 statements.


def test_case_0():
    var_0 = 'test_dir'
    var_1 = True
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = [var_0]



# Parsed testcases at query #15
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_path'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = 'nonexistent_path'
    var_8 = bool('nonexistent_path' in var_5)
    assert var_8 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_find_with_single_file. Retrieved 9/12 statements.
# Partially parsed test_find_with_directory_containing_python_files. Retrieved 16/27 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 13/20 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 14/22 statements.
# Partially parsed test_find_with_symlink. Retrieved 11/18 statements.
# Partially parsed test_find_with_broken_symlink. Retrieved 9/11 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = 'follow_links'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = []
    var_6 = []
    var_7 = module_1.find(var_0, var_4, var_5, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [])
    assert var_9 is True
    var_10 = bool(var_5 == [])
    assert var_10 is True
    var_11 = bool(var_6 == [])
    assert var_11 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_path.py'
    var_1 = [var_0]
    var_2 = False
    var_3 = 'follow_links'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_1, var_5, var_6, var_7)
    var_9 = list(var_8)
    var_10 = bool(var_9 == [])
    assert var_10 is True
    var_11 = bool(var_6 == [])
    assert var_11 is True
    var_12 = bool(var_7 == ['nonexistent_path.py'])
    assert var_12 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = [var_0]
    var_2 = False
    var_3 = 'follow_links'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = []
    var_7 = []
    var_8 = '# test'
    var_9 = module_1.find(var_1, var_5, var_6, var_7)
    var_10 = list(var_9)
    var_11 = bool(var_10 == ['test_file.py'])
    assert var_11 is True
    var_12 = bool(var_6 == [])
    assert var_12 is True
    var_13 = bool(var_7 == [])
    assert var_13 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = True
    var_2 = '# test1'
    var_3 = '# test2'
    var_4 = '# not python'
    var_5 = [var_4]
    var_6 = False
    var_7 = 'follow_links'
    var_8 = {var_7: var_6}
    var_9 = module_0.Config(**var_8)
    var_10 = []
    var_11 = []
    var_12 = module_1.find(var_5, var_9, var_10, var_11)
    var_13 = list(var_12)
    var_14 = set(var_13)
    var_15 = bool(var_14 == {'test_dir/file1.py', 'test_dir/file2.py'})
    assert var_15 is True
    var_16 = bool(var_10 == [])
    assert var_16 is True
    var_17 = bool(var_11 == [])
    assert var_17 is True
    var_18 = 'test_dir/file1.py'
    var_19 = 'test_dir/file2.py'
    var_20 = 'test_dir/file.txt'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = True
    var_2 = '# skipped'
    var_3 = [var_2]
    var_4 = False
    var_5 = 'skipped.py'
    var_6 = [var_5]
    var_7 = 'follow_links'
    var_8 = 'skip_patterns'
    var_9 = {var_7: var_4, var_8: var_6}
    var_10 = module_0.Config(**var_9)
    var_11 = []
    var_12 = []
    var_13 = module_1.find(var_3, var_10, var_11, var_12)
    var_14 = list(var_13)
    var_15 = bool(var_14 == [])
    assert var_15 is True
    var_16 = 'test_dir/skipped.py'
    var_17 = bool(var_12 == [])
    assert var_17 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir/skipped_dir'
    var_1 = True
    var_2 = '# skipped'
    var_3 = 'test_dir'
    var_4 = [var_3]
    var_5 = False
    var_6 = 'skipped_dir'
    var_7 = [var_6]
    var_8 = 'follow_links'
    var_9 = 'skip_patterns'
    var_10 = {var_8: var_5, var_9: var_7}
    var_11 = module_0.Config(**var_10)
    var_12 = []
    var_13 = []
    var_14 = module_1.find(var_4, var_11, var_12, var_13)
    var_15 = list(var_14)
    var_16 = bool(var_15 == [])
    assert var_16 is True
    var_17 = bool(var_13 == [])
    assert var_17 is True
    var_18 = 'test_dir/skipped_dir/file.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = True
    var_2 = '# test'
    var_3 = 'test_link'
    var_4 = [var_3]
    var_5 = 'follow_links'
    var_6 = {var_5: var_1}
    var_7 = module_0.Config(**var_6)
    var_8 = []
    var_9 = []
    var_10 = module_1.find(var_4, var_7, var_8, var_9)
    var_11 = list(var_10)
    var_12 = bool(var_11 == ['test_link/file.py'])
    assert var_12 is True
    var_13 = bool(var_8 == [])
    assert var_13 is True
    var_14 = bool(var_9 == [])
    assert var_14 is True
    var_15 = 'test_dir/file.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = 'broken_link'
    var_2 = [var_1]
    var_3 = True
    var_4 = 'follow_links'
    var_5 = {var_4: var_3}
    var_6 = module_0.Config(**var_5)
    var_7 = []
    var_8 = []
    var_9 = module_1.find(var_2, var_6, var_7, var_8)
    var_10 = list(var_9)
    var_11 = bool(var_10 == [])
    assert var_11 is True
    var_12 = bool(var_7 == [])
    assert var_12 is True
    var_13 = bool(var_8 == ['broken_link'])
    assert var_13 is True



# Parsed testcases at query #17
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '/existing_directory'
    var_1 = [var_0]
    var_2 = False
    var_3 = 'follow_links'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_1, var_5, var_6, var_7)
    var_9 = list(var_8)
    var_10 = bool(var_9 == [])
    assert var_10 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = False
    var_3 = []
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = 'follow_links'
    var_7 = 'skipped_dirs'
    var_8 = 'supported_extensions'
    var_9 = {var_6: var_2, var_7: var_3, var_8: var_5}
    var_10 = module_0.Config(**var_9)
    var_11 = []
    var_12 = []
    var_13 = module_1.find(var_1, var_10, var_11, var_12)
    var_14 = list(var_13)
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = 'test_dir/file1.py'
    var_17 = bool('test_dir/file1.py' in var_14)
    assert var_17 is True
    var_18 = 'test_dir/file2.py'
    var_19 = bool('test_dir/file2.py' in var_14)
    assert var_19 is True
    var_20 = bool(var_11 == [])
    assert var_20 is True
    var_21 = bool(var_12 == [])
    assert var_21 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = False
    var_3 = 'test_dir/skip_dir'
    var_4 = [var_3]
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = 'follow_links'
    var_8 = 'skipped_dirs'
    var_9 = 'supported_extensions'
    var_10 = {var_7: var_2, var_8: var_4, var_9: var_6}
    var_11 = module_0.Config(**var_10)
    var_12 = []
    var_13 = []
    var_14 = module_1.find(var_1, var_11, var_12, var_13)
    var_15 = list(var_14)
    var_16 = len(var_15)
    assert var_16 == 1
    var_17 = 'test_dir/file1.py'
    var_18 = bool('test_dir/file1.py' in var_15)
    assert var_18 is True
    var_19 = bool(var_12 == ['test_dir/skip_dir'])
    assert var_19 is True
    var_20 = bool(var_13 == [])
    assert var_20 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_path'
    var_1 = [var_0]
    var_2 = False
    var_3 = []
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = 'follow_links'
    var_7 = 'skipped_dirs'
    var_8 = 'supported_extensions'
    var_9 = {var_6: var_2, var_7: var_3, var_8: var_5}
    var_10 = module_0.Config(**var_9)
    var_11 = []
    var_12 = []
    var_13 = module_1.find(var_1, var_10, var_11, var_12)
    var_14 = list(var_13)
    var_15 = bool(var_14 == [])
    assert var_15 is True
    var_16 = bool(var_11 == [])
    assert var_16 is True
    var_17 = bool(var_12 == ['nonexistent_path'])
    assert var_17 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = [var_0]
    var_2 = False
    var_3 = []
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = 'follow_links'
    var_7 = 'skipped_dirs'
    var_8 = 'supported_extensions'
    var_9 = {var_6: var_2, var_7: var_3, var_8: var_5}
    var_10 = module_0.Config(**var_9)
    var_11 = []
    var_12 = []
    var_13 = module_1.find(var_1, var_10, var_11, var_12)
    var_14 = list(var_13)
    var_15 = bool(var_14 == ['test_file.py'])
    assert var_15 is True
    var_16 = bool(var_11 == [])
    assert var_16 is True
    var_17 = bool(var_12 == [])
    assert var_17 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = False
    var_3 = []
    var_4 = '.txt'
    var_5 = [var_4]
    var_6 = 'follow_links'
    var_7 = 'skipped_dirs'
    var_8 = 'supported_extensions'
    var_9 = {var_6: var_2, var_7: var_3, var_8: var_5}
    var_10 = module_0.Config(**var_9)
    var_11 = []
    var_12 = []
    var_13 = module_1.find(var_1, var_10, var_11, var_12)
    var_14 = list(var_13)
    var_15 = bool(var_14 == [])
    assert var_15 is True
    var_16 = bool(var_11 == [])
    assert var_16 is True
    var_17 = bool(var_12 == [])
    assert var_17 is True



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

# Partially parsed test_find_with_directory. Retrieved 8/10 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 8/10 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 11/17 statements.


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
    var_8 = bool(var_3 == [])
    assert var_8 is True
    var_9 = bool(var_4 == [])
    assert var_9 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == ['test.py'])
    assert var_8 is True
    var_9 = bool(var_4 == [])
    assert var_9 is True
    var_10 = bool(var_5 == [])
    assert var_10 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True
    var_9 = bool(var_4 == [])
    assert var_9 is True
    var_10 = bool(var_5 == ['nonexistent.py'])
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
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = len(var_7)
    var_9 = bool(var_8 > 0)
    assert var_9 is True
    var_10 = bool(var_4 == [])
    assert var_10 is True
    var_11 = bool(var_5 == [])
    assert var_11 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = [var_0]
    var_2 = [var_0]
    var_3 = 'skipped_files'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_1, var_5, var_6, var_7)
    var_9 = list(var_8)
    var_10 = bool(var_9 == [])
    assert var_10 is True
    var_11 = bool(var_7 == [])
    assert var_11 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'src'
    var_1 = [var_0]
    var_2 = 'src/skipped_dir'
    var_3 = [var_2]
    var_4 = 'skipped_dirs'
    var_5 = {var_4: var_3}
    var_6 = module_0.Config(**var_5)
    var_7 = []
    var_8 = []
    var_9 = module_1.find(var_1, var_6, var_7, var_8)
    var_10 = list(var_9)
    var_11 = len(var_10)
    var_12 = bool(var_11 > 0)
    assert var_12 is True
    var_13 = 'skipped_dir'
    var_14 = [var_2]
    var_15 = bool(var_8 == [])
    assert var_15 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'broken_link'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'follow_links'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_1, var_5, var_6, var_7)
    var_9 = list(var_8)
    var_10 = bool(var_9 == [])
    assert var_10 is True
    var_11 = bool(var_6 == [])
    assert var_11 is True
    var_12 = bool(var_7 == ['broken_link'])
    assert var_12 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'circular_link'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'follow_links'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_1, var_5, var_6, var_7)
    var_9 = list(var_8)
    var_10 = len(var_9)
    var_11 = bool(var_10 > 0)
    assert var_11 is True
    var_12 = bool(var_6 == [])
    assert var_12 is True
    var_13 = bool(var_7 == [])
    assert var_13 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_find_with_skipped_file. Retrieved 11/13 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = False
    var_3 = []
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = 'follow_links'
    var_7 = 'skipped_dirs'
    var_8 = 'supported_extensions'
    var_9 = {var_6: var_2, var_7: var_3, var_8: var_5}
    var_10 = module_0.Config(**var_9)
    var_11 = []
    var_12 = []
    var_13 = module_1.find(var_1, var_10, var_11, var_12)
    var_14 = list(var_13)
    var_15 = bool(var_14 == ['test_dir/file1.py', 'test_dir/subdir/file2.py'])
    assert var_15 is True
    var_16 = bool(var_11 == [])
    assert var_16 is True
    var_17 = bool(var_12 == [])
    assert var_17 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = False
    var_3 = 'test_dir/skip_me'
    var_4 = [var_3]
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = 'follow_links'
    var_8 = 'skipped_dirs'
    var_9 = 'supported_extensions'
    var_10 = {var_7: var_2, var_8: var_4, var_9: var_6}
    var_11 = module_0.Config(**var_10)
    var_12 = []
    var_13 = []
    var_14 = module_1.find(var_1, var_11, var_12, var_13)
    var_15 = list(var_14)
    var_16 = bool(var_15 == ['test_dir/file1.py'])
    assert var_16 is True
    var_17 = bool(var_12 == ['test_dir/skip_me'])
    assert var_17 is True
    var_18 = bool(var_13 == [])
    assert var_18 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'non_existent_dir'
    var_1 = [var_0]
    var_2 = False
    var_3 = []
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = 'follow_links'
    var_7 = 'skipped_dirs'
    var_8 = 'supported_extensions'
    var_9 = {var_6: var_2, var_7: var_3, var_8: var_5}
    var_10 = module_0.Config(**var_9)
    var_11 = []
    var_12 = []
    var_13 = module_1.find(var_1, var_10, var_11, var_12)
    var_14 = list(var_13)
    var_15 = bool(var_14 == [])
    assert var_15 is True
    var_16 = bool(var_11 == [])
    assert var_16 is True
    var_17 = bool(var_12 == ['non_existent_dir'])
    assert var_17 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = [var_0]
    var_2 = False
    var_3 = []
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = 'follow_links'
    var_7 = 'skipped_dirs'
    var_8 = 'supported_extensions'
    var_9 = {var_6: var_2, var_7: var_3, var_8: var_5}
    var_10 = module_0.Config(**var_9)
    var_11 = []
    var_12 = []
    var_13 = module_1.find(var_1, var_10, var_11, var_12)
    var_14 = list(var_13)
    var_15 = bool(var_14 == ['test_file.py'])
    assert var_15 is True
    var_16 = bool(var_11 == [])
    assert var_16 is True
    var_17 = bool(var_12 == [])
    assert var_17 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = False
    var_3 = []
    var_4 = '.txt'
    var_5 = [var_4]
    var_6 = 'follow_links'
    var_7 = 'skipped_dirs'
    var_8 = 'supported_extensions'
    var_9 = {var_6: var_2, var_7: var_3, var_8: var_5}
    var_10 = module_0.Config(**var_9)
    var_11 = []
    var_12 = []
    var_13 = module_1.find(var_1, var_10, var_11, var_12)
    var_14 = list(var_13)
    var_15 = bool(var_14 == [])
    assert var_15 is True
    var_16 = bool(var_11 == [])
    assert var_16 is True
    var_17 = bool(var_12 == [])
    assert var_17 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir/skip_me.py'
    var_1 = [var_0]
    var_2 = False
    var_3 = []
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = 'follow_links'
    var_7 = 'skipped_dirs'
    var_8 = 'supported_extensions'
    var_9 = {var_6: var_2, var_7: var_3, var_8: var_5}
    var_10 = module_0.Config(**var_9)
    var_11 = []
    var_12 = []
    var_13 = module_1.find(var_1, var_10, var_11, var_12)
    var_14 = list(var_13)
    var_15 = bool(var_14 == [])
    assert var_15 is True
    var_16 = bool(var_12 == [])
    assert var_16 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_isdir_predicate_evaluates_to_true. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'test_directory'
    var_1 = True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_find_with_single_file. Retrieved 4/13 statements.
# Partially parsed test_find_with_directory_containing_py_files. Retrieved 5/12 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 5/15 statements.
# Partially parsed test_find_with_broken_symlink. Retrieved 5/11 statements.


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

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = []
    var_1 = '/non/existent/path'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = []
    var_6 = module_1.find(var_2, var_4, var_5, var_0)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True
    var_9 = bool(var_0 == ['/non/existent/path'])
    assert var_9 is True

import isort.settings as module_0

def test_case_0():
    var_0 = b"print('hello')"
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = []
    var_4 = []

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []

import isort.settings as module_0

def test_case_0():
    var_0 = 'broken_link'
    var_1 = '/non/existent/target'
    var_2 = []
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = []



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_find_with_directory. Retrieved 7/9 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 7/10 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 7/12 statements.
# Partially parsed test_find_with_follow_links_disabled. Retrieved 8/10 statements.
# Partially parsed test_find_with_follow_links_enabled. Retrieved 8/10 statements.


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
    var_8 = bool(var_3 == [])
    assert var_8 is True
    var_9 = bool(var_4 == [])
    assert var_9 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_path'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True
    var_9 = bool(var_4 == [])
    assert var_9 is True
    var_10 = bool(var_5 == ['nonexistent_path'])
    assert var_10 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'existing_file.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == ['existing_file.py'])
    assert var_8 is True
    var_9 = bool(var_4 == [])
    assert var_9 is True
    var_10 = bool(var_5 == [])
    assert var_10 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'existing_dir'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_4 == [])
    assert var_8 is True
    var_9 = bool(var_5 == [])
    assert var_9 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'skipped_file.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True
    var_9 = bool(var_5 == [])
    assert var_9 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'skipped_dir'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True
    var_9 = [var_0]
    var_10 = bool(var_5 == [])
    assert var_10 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'dir_with_symlink'
    var_1 = [var_0]
    var_2 = False
    var_3 = 'follow_links'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_1, var_5, var_6, var_7)
    var_9 = list(var_8)
    var_10 = bool(var_6 == [])
    assert var_10 is True
    var_11 = bool(var_7 == [])
    assert var_11 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'dir_with_symlink'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'follow_links'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_1, var_5, var_6, var_7)
    var_9 = list(var_8)
    var_10 = bool(var_6 == [])
    assert var_10 is True
    var_11 = bool(var_7 == [])
    assert var_11 is True



# Parsed testcases at query #8
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_file.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = len(var_5)
    assert var_7 == 1
    var_8 = var_5[0]
    assert var_8 == 'nonexistent_file.py'



# Parsed testcases at query #9
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_file.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = len(var_5)
    assert var_8 == 1
    var_9 = var_5[0]
    assert var_9 == 'nonexistent_file.py'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_isdir_predicate_evaluates_to_true. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 'test_directory'
    var_1 = True



# Parsed testcases at query #11
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = False
    var_3 = 'follow_links'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_1, var_5, var_6, var_7)
    var_9 = list(var_8)
    var_10 = len(var_9)
    assert var_10 == 2
    var_11 = 'test_dir/file1.py'
    var_12 = bool('test_dir/file1.py' in var_9)
    assert var_12 is True
    var_13 = 'test_dir/subdir/file2.py'
    var_14 = bool('test_dir/subdir/file2.py' in var_9)
    assert var_14 is True
    var_15 = len(var_6)
    assert var_15 == 0
    var_16 = len(var_7)
    assert var_16 == 0

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = False
    var_3 = 'subdir'
    var_4 = [var_3]
    var_5 = 'follow_links'
    var_6 = 'skip_dirs'
    var_7 = {var_5: var_2, var_6: var_4}
    var_8 = module_0.Config(**var_7)
    var_9 = []
    var_10 = []
    var_11 = module_1.find(var_1, var_8, var_9, var_10)
    var_12 = list(var_11)
    var_13 = len(var_12)
    assert var_13 == 1
    var_14 = 'test_dir/file1.py'
    var_15 = bool('test_dir/file1.py' in var_12)
    assert var_15 is True
    var_16 = 'test_dir/subdir/file2.py'
    var_17 = bool('test_dir/subdir/file2.py' not in var_12)
    assert var_17 is True
    var_18 = len(var_9)
    assert var_18 == 1
    var_19 = 'test_dir/subdir'
    var_20 = bool('test_dir/subdir' in var_9[0])
    assert var_20 is True
    var_21 = len(var_10)
    assert var_21 == 0

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_path'
    var_1 = [var_0]
    var_2 = False
    var_3 = 'follow_links'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_1, var_5, var_6, var_7)
    var_9 = list(var_8)
    var_10 = len(var_9)
    assert var_10 == 0
    var_11 = len(var_6)
    assert var_11 == 0
    var_12 = len(var_7)
    assert var_12 == 1
    var_13 = 'nonexistent_path'
    var_14 = bool('nonexistent_path' in var_7)
    assert var_14 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir/file1.py'
    var_1 = [var_0]
    var_2 = False
    var_3 = 'follow_links'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_1, var_5, var_6, var_7)
    var_9 = list(var_8)
    var_10 = len(var_9)
    assert var_10 == 1
    var_11 = 'test_dir/file1.py'
    var_12 = bool('test_dir/file1.py' in var_9)
    assert var_12 is True
    var_13 = len(var_6)
    assert var_13 == 0
    var_14 = len(var_7)
    assert var_14 == 0

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = False
    var_3 = 'file1.py'
    var_4 = [var_3]
    var_5 = 'follow_links'
    var_6 = 'skip_files'
    var_7 = {var_5: var_2, var_6: var_4}
    var_8 = module_0.Config(**var_7)
    var_9 = []
    var_10 = []
    var_11 = module_1.find(var_1, var_8, var_9, var_10)
    var_12 = list(var_11)
    var_13 = len(var_12)
    assert var_13 == 1
    var_14 = 'test_dir/file1.py'
    var_15 = bool('test_dir/file1.py' not in var_12)
    assert var_15 is True
    var_16 = 'test_dir/subdir/file2.py'
    var_17 = bool('test_dir/subdir/file2.py' in var_12)
    assert var_17 is True
    var_18 = len(var_9)
    assert var_18 == 1
    var_19 = 'test_dir/file1.py'
    var_20 = bool('test_dir/file1.py' in var_9[0])
    assert var_20 is True
    var_21 = len(var_10)
    assert var_21 == 0



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_find_with_single_file. Retrieved 3/8 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 3/14 statements.
# Partially parsed test_find_with_directory_containing_python_files. Retrieved 7/18 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 6/19 statements.
# Partially parsed test_find_with_non_python_file. Retrieved 3/8 statements.
# Partially parsed test_find_with_follow_links_disabled. Retrieved 7/16 statements.
# Partially parsed test_find_with_circular_symlink. Retrieved 8/19 statements.


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

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = []
    var_1 = 'nonexistent_path.py'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = []
    var_6 = module_1.find(var_2, var_4, var_5, var_0)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True
    var_9 = bool(var_0 == ['nonexistent_path.py'])
    assert var_9 is True

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = []
    var_3 = []

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = []
    var_3 = []

import isort.settings as module_0

def test_case_0():
    var_0 = 'file1.py'
    var_1 = 'file2.py'
    var_2 = '# test'
    var_3 = '# test'
    var_4 = {}
    var_5 = module_0.Config(**var_4)
    var_6 = []
    var_7 = []

import isort.settings as module_0

def test_case_0():
    var_0 = 'subdir'
    var_1 = 'file.py'
    var_2 = '# test'
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = []
    var_6 = []

import isort.settings as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = []
    var_3 = []

import isort.settings as module_0

def test_case_0():
    var_0 = 'subdir'
    var_1 = 'file.py'
    var_2 = '# test'
    var_3 = False
    var_4 = 'follow_links'
    var_5 = {var_4: var_3}
    var_6 = module_0.Config(**var_5)
    var_7 = []
    var_8 = []

import isort.settings as module_0

def test_case_0():
    var_0 = 'subdir'
    var_1 = 'link'
    var_2 = 'file.py'
    var_3 = '# test'
    var_4 = True
    var_5 = 'follow_links'
    var_6 = {var_5: var_4}
    var_7 = module_0.Config(**var_6)
    var_8 = []
    var_9 = []



# Parsed testcases at query #13
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = False
    var_3 = []
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = 'follow_links'
    var_7 = 'skipped_dirs'
    var_8 = 'supported_extensions'
    var_9 = {var_6: var_2, var_7: var_3, var_8: var_5}
    var_10 = module_0.Config(**var_9)
    var_11 = []
    var_12 = []
    var_13 = module_1.find(var_1, var_10, var_11, var_12)
    var_14 = list(var_13)
    var_15 = bool(var_14 == ['test_dir/file1.py', 'test_dir/file2.py'])
    assert var_15 is True
    var_16 = bool(var_11 == [])
    assert var_16 is True
    var_17 = bool(var_12 == [])
    assert var_17 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = False
    var_3 = 'test_dir/skip_dir'
    var_4 = [var_3]
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = 'follow_links'
    var_8 = 'skipped_dirs'
    var_9 = 'supported_extensions'
    var_10 = {var_7: var_2, var_8: var_4, var_9: var_6}
    var_11 = module_0.Config(**var_10)
    var_12 = []
    var_13 = []
    var_14 = module_1.find(var_1, var_11, var_12, var_13)
    var_15 = list(var_14)
    var_16 = bool(var_15 == ['test_dir/file1.py'])
    assert var_16 is True
    var_17 = bool(var_12 == ['test_dir/skip_dir'])
    assert var_17 is True
    var_18 = bool(var_13 == [])
    assert var_18 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_path'
    var_1 = [var_0]
    var_2 = False
    var_3 = []
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = 'follow_links'
    var_7 = 'skipped_dirs'
    var_8 = 'supported_extensions'
    var_9 = {var_6: var_2, var_7: var_3, var_8: var_5}
    var_10 = module_0.Config(**var_9)
    var_11 = []
    var_12 = []
    var_13 = module_1.find(var_1, var_10, var_11, var_12)
    var_14 = list(var_13)
    var_15 = bool(var_14 == [])
    assert var_15 is True
    var_16 = bool(var_11 == [])
    assert var_16 is True
    var_17 = bool(var_12 == ['nonexistent_path'])
    assert var_17 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = [var_0]
    var_2 = False
    var_3 = []
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = 'follow_links'
    var_7 = 'skipped_dirs'
    var_8 = 'supported_extensions'
    var_9 = {var_6: var_2, var_7: var_3, var_8: var_5}
    var_10 = module_0.Config(**var_9)
    var_11 = []
    var_12 = []
    var_13 = module_1.find(var_1, var_10, var_11, var_12)
    var_14 = list(var_13)
    var_15 = bool(var_14 == ['test_file.py'])
    assert var_15 is True
    var_16 = bool(var_11 == [])
    assert var_16 is True
    var_17 = bool(var_12 == [])
    assert var_17 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = False
    var_3 = []
    var_4 = '.txt'
    var_5 = [var_4]
    var_6 = 'follow_links'
    var_7 = 'skipped_dirs'
    var_8 = 'supported_extensions'
    var_9 = {var_6: var_2, var_7: var_3, var_8: var_5}
    var_10 = module_0.Config(**var_9)
    var_11 = []
    var_12 = []
    var_13 = module_1.find(var_1, var_10, var_11, var_12)
    var_14 = list(var_13)
    var_15 = bool(var_14 == ['test_dir/file.txt'])
    assert var_15 is True
    var_16 = bool(var_11 == [])
    assert var_16 is True
    var_17 = bool(var_12 == [])
    assert var_17 is True



# Parsed testcases at query #14
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_path'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = 'nonexistent_path'
    var_8 = bool('nonexistent_path' in var_5)
    assert var_8 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_isdir_predicate_evaluates_to_true. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 'test_directory'
    var_1 = True



# Parsed testcases at query #16
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_file.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = len(var_5)
    assert var_7 == 1
    var_8 = var_5[0]
    assert var_8 == 'nonexistent_file.py'



# Parsed testcases at query #17
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_path'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = len(var_5)
    assert var_8 == 1
    var_9 = var_5[0]
    assert var_9 == 'nonexistent_path'
    var_10 = len(var_7)
    assert var_10 == 0
    var_11 = len(var_4)
    assert var_11 == 0



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_isdir_predicate_evaluates_to_true. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = True
    var_5 = []



# Parsed testcases at query #19
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '/existing_directory'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'follow_links'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_1, var_5, var_6, var_7)
    var_9 = list(var_8)
    var_10 = len(var_9)
    var_11 = bool(var_10 > 0)
    assert var_11 is True



# Parsed testcases at query #20
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = False
    var_3 = []
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = 'follow_links'
    var_7 = 'skipped_patterns'
    var_8 = 'supported_extensions'
    var_9 = {var_6: var_2, var_7: var_3, var_8: var_5}
    var_10 = module_0.Config(**var_9)
    var_11 = []
    var_12 = []
    var_13 = module_1.find(var_1, var_10, var_11, var_12)
    var_14 = list(var_13)
    var_15 = bool(var_14 == ['test_dir/file1.py', 'test_dir/subdir/file2.py'])
    assert var_15 is True
    var_16 = bool(var_11 == [])
    assert var_16 is True
    var_17 = bool(var_12 == [])
    assert var_17 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = False
    var_3 = '*skip*'
    var_4 = [var_3]
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = 'follow_links'
    var_8 = 'skipped_patterns'
    var_9 = 'supported_extensions'
    var_10 = {var_7: var_2, var_8: var_4, var_9: var_6}
    var_11 = module_0.Config(**var_10)
    var_12 = []
    var_13 = []
    var_14 = module_1.find(var_1, var_11, var_12, var_13)
    var_15 = list(var_14)
    var_16 = bool(var_15 == ['test_dir/file1.py'])
    assert var_16 is True
    var_17 = bool(var_12 == ['test_dir/skip_file.py'])
    assert var_17 is True
    var_18 = bool(var_13 == [])
    assert var_18 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_path'
    var_1 = [var_0]
    var_2 = False
    var_3 = []
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = 'follow_links'
    var_7 = 'skipped_patterns'
    var_8 = 'supported_extensions'
    var_9 = {var_6: var_2, var_7: var_3, var_8: var_5}
    var_10 = module_0.Config(**var_9)
    var_11 = []
    var_12 = []
    var_13 = module_1.find(var_1, var_10, var_11, var_12)
    var_14 = list(var_13)
    var_15 = bool(var_14 == [])
    assert var_15 is True
    var_16 = bool(var_11 == [])
    assert var_16 is True
    var_17 = bool(var_12 == ['nonexistent_path'])
    assert var_17 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir/file1.py'
    var_1 = [var_0]
    var_2 = False
    var_3 = []
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = 'follow_links'
    var_7 = 'skipped_patterns'
    var_8 = 'supported_extensions'
    var_9 = {var_6: var_2, var_7: var_3, var_8: var_5}
    var_10 = module_0.Config(**var_9)
    var_11 = []
    var_12 = []
    var_13 = module_1.find(var_1, var_10, var_11, var_12)
    var_14 = list(var_13)
    var_15 = bool(var_14 == ['test_dir/file1.py'])
    assert var_15 is True
    var_16 = bool(var_11 == [])
    assert var_16 is True
    var_17 = bool(var_12 == [])
    assert var_17 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = False
    var_3 = []
    var_4 = '.txt'
    var_5 = [var_4]
    var_6 = 'follow_links'
    var_7 = 'skipped_patterns'
    var_8 = 'supported_extensions'
    var_9 = {var_6: var_2, var_7: var_3, var_8: var_5}
    var_10 = module_0.Config(**var_9)
    var_11 = []
    var_12 = []
    var_13 = module_1.find(var_1, var_10, var_11, var_12)
    var_14 = list(var_13)
    var_15 = bool(var_14 == [])
    assert var_15 is True
    var_16 = bool(var_11 == [])
    assert var_16 is True
    var_17 = bool(var_12 == [])
    assert var_17 is True



# Parsed testcases at query #21
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_dir'
    var_1 = [var_0]
    var_2 = False
    var_3 = 'follow_links'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_1, var_5, var_6, var_7)
    var_9 = list(var_8)
    var_10 = bool(var_9 == [])
    assert var_10 is True
    var_11 = bool(var_6 == [])
    assert var_11 is True
    var_12 = bool(var_7 == ['nonexistent_dir'])
    assert var_12 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_predicate_at_line_9_evaluates_to_false. Retrieved 7/8 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_path'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)



# Parsed testcases at query #23
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
    var_8 = bool(var_3 == [])
    assert var_8 is True
    var_9 = bool(var_4 == [])
    assert var_9 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_path'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True
    var_9 = bool(var_4 == [])
    assert var_9 is True
    var_10 = bool(var_5 == ['nonexistent_path'])
    assert var_10 is True

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
    var_9 = bool(var_4 == [])
    assert var_9 is True
    var_10 = bool(var_5 == [])
    assert var_10 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == ['test_dir/file1.py', 'test_dir/file2.py'])
    assert var_8 is True
    var_9 = bool(var_4 == [])
    assert var_9 is True
    var_10 = bool(var_5 == [])
    assert var_10 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = 'test_dir/skip_me'
    var_3 = [var_2]
    var_4 = 'skip_dirs'
    var_5 = {var_4: var_3}
    var_6 = module_0.Config(**var_5)
    var_7 = []
    var_8 = []
    var_9 = module_1.find(var_1, var_6, var_7, var_8)
    var_10 = list(var_9)
    var_11 = bool(var_10 == ['test_dir/file1.py', 'test_dir/file2.py'])
    assert var_11 is True
    var_12 = bool(var_7 == ['test_dir/skip_me'])
    assert var_12 is True
    var_13 = bool(var_8 == [])
    assert var_13 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = 'test_dir/skip_me.py'
    var_3 = [var_2]
    var_4 = 'skip_files'
    var_5 = {var_4: var_3}
    var_6 = module_0.Config(**var_5)
    var_7 = []
    var_8 = []
    var_9 = module_1.find(var_1, var_6, var_7, var_8)
    var_10 = list(var_9)
    var_11 = bool(var_10 == ['test_dir/file1.py', 'test_dir/file2.py'])
    assert var_11 is True
    var_12 = bool(var_7 == ['test_dir/skip_me.py'])
    assert var_12 is True
    var_13 = bool(var_8 == [])
    assert var_13 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'broken_link'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'follow_links'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_1, var_5, var_6, var_7)
    var_9 = list(var_8)
    var_10 = bool(var_9 == [])
    assert var_10 is True
    var_11 = bool(var_6 == [])
    assert var_11 is True
    var_12 = bool(var_7 == ['broken_link'])
    assert var_12 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_find_with_single_file. Retrieved 8/11 statements.
# Partially parsed test_find_with_directory_containing_python_files. Retrieved 11/19 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 11/16 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 8/12 statements.
# Partially parsed test_find_with_symlink_loop. Retrieved 12/15 statements.


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

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = []
    var_1 = '/non/existent/path'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = []
    var_6 = module_1.find(var_2, var_4, var_5, var_0)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True
    var_9 = bool(var_0 == ['/non/existent/path'])
    assert var_9 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '# test'
    var_1 = 'test_file.py'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = []
    var_6 = []
    var_7 = module_1.find(var_2, var_4, var_5, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == ['test_file.py'])
    assert var_9 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = True
    var_2 = '# test1'
    var_3 = '# test2'
    var_4 = '# not python'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Config(**var_6)
    var_8 = []
    var_9 = []
    var_10 = module_1.find(var_5, var_7, var_8, var_9)
    var_11 = list(var_10)
    var_12 = 'test_dir/file1.py'
    var_13 = bool('test_dir/file1.py' in var_11)
    assert var_13 is True
    var_14 = 'test_dir/file2.py'
    var_15 = bool('test_dir/file2.py' in var_11)
    assert var_15 is True
    var_16 = 'test_dir/file3.txt'
    var_17 = bool('test_dir/file3.txt' not in var_11)
    assert var_17 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir/skip_me'
    var_1 = True
    var_2 = '# test'
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = 'skip_me'
    var_6 = []
    var_7 = 'test_dir'
    var_8 = [var_7]
    var_9 = []
    var_10 = module_1.find(var_8, var_4, var_6, var_9)
    var_11 = list(var_10)
    var_12 = bool(var_11 == [])
    assert var_12 is True
    var_13 = 'skip_me'
    var_14 = bool('skip_me' in var_6[0])
    assert var_14 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '# test'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = 'skip_me.py'
    var_4 = []
    var_5 = [var_3]
    var_6 = []
    var_7 = module_1.find(var_5, var_2, var_4, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [])
    assert var_9 is True
    var_10 = 'skip_me.py'
    var_11 = bool('skip_me.py' in var_4[0])
    assert var_11 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir/subdir'
    var_1 = True
    var_2 = '../test_dir'
    var_3 = 'test_dir/subdir/loop_link'
    var_4 = 'follow_links'
    var_5 = {var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = 'test_dir'
    var_8 = [var_7]
    var_9 = []
    var_10 = []
    var_11 = module_1.find(var_8, var_6, var_9, var_10)
    var_12 = list(var_11)
    var_13 = len(var_12)
    assert var_13 == 0



# Parsed testcases at query #25
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_path'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 0
    var_9 = len(var_4)
    assert var_9 == 0
    var_10 = len(var_5)
    assert var_10 == 1
    var_11 = var_5[0]
    assert var_11 == 'nonexistent_path'



# Parsed testcases at query #26
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7[0]
    assert var_9 == 'test_file.txt'



# Parsed testcases at query #27
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_dir'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 0
    var_9 = len(var_4)
    assert var_9 == 0
    var_10 = len(var_5)
    assert var_10 == 1
    var_11 = var_5[0]
    assert var_11 == 'nonexistent_dir'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_os_path_isdir_evaluates_to_true. Retrieved 1/2 statements.


def test_case_0():
    var_0 = '/some/directory'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_find_with_single_file. Retrieved 8/11 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 8/14 statements.
# Partially parsed test_find_with_directory_containing_python_files. Retrieved 15/26 statements.
# Partially parsed test_find_with_nested_directories. Retrieved 11/17 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 11/22 statements.
# Partially parsed test_find_with_symlink_loop. Retrieved 11/18 statements.


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

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = []
    var_1 = 'nonexistent_path.py'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = []
    var_6 = module_1.find(var_2, var_4, var_5, var_0)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True
    var_9 = bool(var_0 == ['nonexistent_path.py'])
    assert var_9 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '# test'
    var_1 = 'test_file.py'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = []
    var_6 = []
    var_7 = module_1.find(var_2, var_4, var_5, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == ['test_file.py'])
    assert var_9 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '# skip'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = 'skip_me.py'
    var_4 = []
    var_5 = [var_3]
    var_6 = []
    var_7 = module_1.find(var_5, var_2, var_4, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [])
    assert var_9 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = True
    var_2 = '# file1'
    var_3 = '# file2'
    var_4 = '# not python'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Config(**var_6)
    var_8 = []
    var_9 = []
    var_10 = module_1.find(var_5, var_7, var_8, var_9)
    var_11 = list(var_10)
    var_12 = sorted(var_11)
    var_13 = bool(var_12 == ['test_dir/file1.py', 'test_dir/file2.py'])
    assert var_13 is True
    var_14 = 'test_dir/file1.py'
    var_15 = 'test_dir/file2.py'
    var_16 = 'test_dir/readme.txt'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'parent/child'
    var_1 = True
    var_2 = '# nested'
    var_3 = 'parent'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Config(**var_5)
    var_7 = []
    var_8 = []
    var_9 = module_1.find(var_4, var_6, var_7, var_8)
    var_10 = list(var_9)
    var_11 = bool(var_10 == ['parent/child/nested.py'])
    assert var_11 is True
    var_12 = 'parent/child/nested.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'skip_dir/subdir'
    var_1 = True
    var_2 = '# should be skipped'
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = 'skip_dir'
    var_6 = []
    var_7 = [var_5]
    var_8 = []
    var_9 = module_1.find(var_7, var_4, var_6, var_8)
    var_10 = list(var_9)
    var_11 = bool(var_10 == [])
    assert var_11 is True
    var_12 = [var_5]
    var_13 = 'skip_dir/subdir/file.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'link_dir'
    var_1 = True
    var_2 = 'link_dir/link'
    var_3 = '# test'
    var_4 = 'follow_links'
    var_5 = {var_4: var_1}
    var_6 = module_0.Config(**var_5)
    var_7 = [var_3]
    var_8 = []
    var_9 = []
    var_10 = module_1.find(var_7, var_6, var_8, var_9)
    var_11 = list(var_10)
    var_12 = bool(var_11 == ['link_dir/file.py'])
    assert var_12 is True
    var_13 = 'link_dir/file.py'



# Parsed testcases at query #30
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '/valid/directory'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'follow_links'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_1, var_5, var_6, var_7)
    var_9 = list(var_8)
    var_10 = bool(var_9 == [])
    assert var_10 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_os_path_isdir_evaluates_to_true. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'test_directory'



# Parsed testcases at query #32
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'valid_directory'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'follow_links'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_1, var_5, var_6, var_7)
    var_9 = list(var_8)
    var_10 = bool(var_9 == ['valid_directory/file1.py', 'valid_directory/file2.py'])
    assert var_10 is True



# Parsed testcases at query #33
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_directory'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 0
    var_9 = len(var_4)
    assert var_9 == 0
    var_10 = len(var_5)
    assert var_10 == 1
    var_11 = var_5[0]
    assert var_11 == 'nonexistent_directory'



# Parsed testcases at query #34
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_directory'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 0
    var_9 = len(var_4)
    assert var_9 == 0
    var_10 = len(var_5)
    assert var_10 == 1
    var_11 = var_5[0]
    assert var_11 == 'nonexistent_directory'



