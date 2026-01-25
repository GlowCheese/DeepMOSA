####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_find_with_skipped_file. Retrieved 8/12 statements.
# Partially parsed test_find_with_directory. Retrieved 14/25 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 11/19 statements.
# Partially parsed test_find_with_symlink_directory. Retrieved 13/23 statements.
# Partially parsed test_find_with_duplicate_resolved_directory. Retrieved 14/24 statements.
# Partially parsed test_find_with_mixed_paths. Retrieved 17/32 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = []
    var_3 = []
    var_4 = 'test.py'
    var_5 = [var_4]
    var_6 = module_1.find(var_5, var_1, var_2, var_3)
    var_7 = list(var_6)
    var_8 = bool(var_7 == ['test.py'])
    assert var_8 is True
    var_9 = bool(var_2 == [])
    assert var_9 is True
    var_10 = bool(var_3 == [])
    assert var_10 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = []
    var_3 = []
    var_4 = 'nonexistent.py'
    var_5 = [var_4]
    var_6 = module_1.find(var_5, var_1, var_2, var_3)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True
    var_9 = bool(var_2 == [])
    assert var_9 is True
    var_10 = bool(var_3 == ['nonexistent.py'])
    assert var_10 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = True
    var_3 = []
    var_4 = []
    var_5 = 'skipped.py'
    var_6 = [var_5]
    var_7 = module_1.find(var_6, var_1, var_3, var_4)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [])
    assert var_9 is True
    var_10 = bool(var_4 == [])
    assert var_10 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '.py'
    var_3 = False
    var_4 = []
    var_5 = []
    var_6 = 'test_dir'
    var_7 = [var_6]
    var_8 = True
    var_9 = ''
    var_10 = ''
    var_11 = module_1.find(var_7, var_1, var_4, var_5)
    var_12 = list(var_11)
    var_13 = 'test_dir/file1.py'
    var_14 = bool('test_dir/file1.py' in var_12)
    assert var_14 is True
    var_15 = 'test_dir/file2.txt'
    var_16 = bool('test_dir/file2.txt' not in var_12)
    assert var_16 is True
    var_17 = bool(var_4 == [])
    assert var_17 is True
    var_18 = bool(var_5 == [])
    assert var_18 is True
    var_19 = 'test_dir/file1.py'
    var_20 = 'test_dir/file2.txt'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '.py'
    var_3 = 'test_dir'
    var_4 = []
    var_5 = []
    var_6 = [var_3]
    var_7 = True
    var_8 = ''
    var_9 = module_1.find(var_6, var_1, var_4, var_5)
    var_10 = list(var_9)
    var_11 = bool(var_10 == [])
    assert var_11 is True
    var_12 = bool(var_4 == ['test_dir'])
    assert var_12 is True
    var_13 = bool(var_5 == [])
    assert var_13 is True
    var_14 = 'test_dir/file.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '.py'
    var_3 = False
    var_4 = []
    var_5 = []
    var_6 = 'real_dir'
    var_7 = True
    var_8 = ''
    var_9 = 'link_dir'
    var_10 = [var_9]
    var_11 = module_1.find(var_10, var_1, var_4, var_5)
    var_12 = list(var_11)
    var_13 = 'link_dir/file.py'
    var_14 = bool('link_dir/file.py' in var_12)
    assert var_14 is True
    var_15 = bool(var_4 == [])
    assert var_15 is True
    var_16 = bool(var_5 == [])
    assert var_16 is True
    var_17 = 'real_dir/file.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '.py'
    var_3 = False
    var_4 = []
    var_5 = []
    var_6 = 'dir1'
    var_7 = True
    var_8 = 'dir2'
    var_9 = ''
    var_10 = [var_6, var_8]
    var_11 = module_1.find(var_10, var_1, var_4, var_5)
    var_12 = list(var_11)
    var_13 = len(var_12)
    assert var_13 == 1
    var_14 = bool(var_4 == [])
    assert var_14 is True
    var_15 = bool(var_5 == [])
    assert var_15 is True
    var_16 = 'dir1/file.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '.py'
    var_3 = 'skip.py'
    var_4 = []
    var_5 = []
    var_6 = 'mixed_dir'
    var_7 = True
    var_8 = ''
    var_9 = ''
    var_10 = ''
    var_11 = 'single.py'
    var_12 = 'ghost.py'
    var_13 = [var_6, var_11, var_12]
    var_14 = module_1.find(var_13, var_1, var_4, var_5)
    var_15 = list(var_14)
    var_16 = 'mixed_dir/valid.py'
    var_17 = bool('mixed_dir/valid.py' in var_15)
    assert var_17 is True
    var_18 = 'single.py'
    var_19 = bool('single.py' in var_15)
    assert var_19 is True
    var_20 = 'mixed_dir/skip.py'
    var_21 = bool('mixed_dir/skip.py' not in var_15)
    assert var_21 is True
    var_22 = 'mixed_dir/skip.py'
    var_23 = bool(var_5 == ['ghost.py'])
    assert var_23 is True
    var_24 = 'mixed_dir/valid.py'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_find_with_directory_and_skipped_path. Retrieved 13/36 statements.
# Partially parsed test_find_with_single_file. Retrieved 5/19 statements.
# Partially parsed test_find_with_non_existent_file. Retrieved 5/13 statements.
# Partially parsed test_find_with_unsupported_filetype. Retrieved 5/18 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 6/20 statements.
# Partially parsed test_find_with_follow_links_and_visited_dirs. Retrieved 9/29 statements.
# Partially parsed test_find_with_multiple_paths. Retrieved 14/40 statements.


def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = 'skip_me'
    var_4 = 'include'
    var_5 = ''
    var_6 = ''
    assert var_6 == 1
    var_7 = 0
    var_8 = 'include'
    var_9 = 'b.py'
    var_10 = len(var_1)
    assert var_10 == 1
    var_11 = var_1[var_7]
    var_12 = 'skip_me'
    var_13 = bool(var_2 == [])
    assert var_13 is True

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = 'test.py'
    var_4 = ''
    assert var_4 == 1
    var_5 = bool(var_1 == [])
    assert var_5 is True
    var_6 = bool(var_2 == [])
    assert var_6 is True

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = 'non_existent.py'
    var_4 = [var_3]
    var_5 = bool(var_1 == [])
    assert var_5 is True
    var_6 = bool(var_2 == ['non_existent.py'])
    assert var_6 is True

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = 'test.txt'
    var_4 = ''
    var_5 = bool(var_1 == [])
    assert var_5 is True
    var_6 = bool(var_2 == [])
    assert var_6 is True

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = 'skip.py'
    var_4 = ''
    var_5 = len(var_1)
    assert var_5 == 1
    var_6 = var_1[0]
    var_7 = bool(var_2 == [])
    assert var_7 is True

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = []
    var_3 = 'dir'
    var_4 = 'link'
    var_5 = ''
    assert var_5 == 1
    var_6 = 0
    var_7 = 'dir'
    var_8 = 'test.py'
    var_9 = bool(var_1 == [])
    assert var_9 is True
    var_10 = bool(var_2 == [])
    assert var_10 is True

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = 'dir1'
    var_4 = 'dir2'
    var_5 = 'a.py'
    var_6 = 'b.py'
    var_7 = ''
    var_8 = ''
    assert var_8 == 2
    var_9 = 'dir1'
    var_10 = 'a.py'
    var_11 = any(var_5)
    var_12 = bool(var_11)
    assert var_12 is True
    var_13 = 'dir2'
    var_14 = 'b.py'
    var_15 = bool(var_1 == [])
    assert var_15 is True
    var_16 = bool(var_2 == [])
    assert var_16 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_find_with_directory_and_skipped_path. Retrieved 7/40 statements.
# Partially parsed test_find_with_nonexistent_path. Retrieved 4/15 statements.
# Partially parsed test_find_with_single_file. Retrieved 4/23 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 5/28 statements.
# Partially parsed test_find_with_follow_links_and_visited_dirs. Retrieved 7/31 statements.
# Partially parsed test_find_with_unsupported_filetype. Retrieved 4/23 statements.
# Partially parsed test_find_with_multiple_paths. Retrieved 5/32 statements.


def test_case_0():
    var_0 = 'subdir'
    var_1 = 'file1.py'
    var_2 = 'file2.py'
    var_3 = 'skipped.py'
    var_4 = 'w'
    var_5 = []
    var_6 = []
    var_7 = bool(var_6 == [])
    assert var_7 is True

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '/nonexistent/path'
    var_3 = [var_2]
    var_4 = bool(var_0 == [])
    assert var_4 is True
    var_5 = bool(var_1 == ['/nonexistent/path'])
    assert var_5 is True

def test_case_0():
    var_0 = '.py'
    var_1 = False
    var_2 = []
    var_3 = []
    var_4 = bool(var_2 == [])
    assert var_4 is True
    var_5 = bool(var_3 == [])
    assert var_5 is True

def test_case_0():
    var_0 = 'skipped_dir'
    var_1 = 'file.py'
    var_2 = 'w'
    var_3 = []
    var_4 = []
    var_5 = bool(var_4 == [])
    assert var_5 is True

def test_case_0():
    var_0 = 'subdir'
    var_1 = 'linkdir'
    var_2 = 'file1.py'
    var_3 = 'w'
    var_4 = True
    var_5 = []
    var_6 = []
    var_7 = bool(var_5 == [])
    assert var_7 is True
    var_8 = bool(var_6 == [])
    assert var_8 is True

def test_case_0():
    var_0 = '.txt'
    var_1 = False
    var_2 = []
    var_3 = []
    var_4 = bool(var_2 == [])
    assert var_4 is True
    var_5 = bool(var_3 == [])
    assert var_5 is True

def test_case_0():
    var_0 = 'file1.py'
    var_1 = 'file2.py'
    var_2 = 'w'
    var_3 = []
    var_4 = []
    var_5 = bool(var_3 == [])
    assert var_5 is True
    var_6 = bool(var_4 == [])
    assert var_6 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_predicate_at_line_8_evaluates_true_for_directory. Retrieved 8/10 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '/tmp/test_dir'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = True
    var_7 = module_1.find(var_1, var_3, var_4, var_5)
    var_8 = list(var_7)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_predicate_at_line_8_evaluates_true_for_directory. Retrieved 8/10 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '/some/directory'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = True
    var_7 = module_1.find(var_1, var_3, var_4, var_5)
    var_8 = list(var_7)



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_find_with_skipped_file. Retrieved 8/12 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 8/10 statements.
# Partially parsed test_find_with_supported_file_in_directory. Retrieved 15/17 statements.
# Partially parsed test_find_with_unsupported_file_in_directory. Retrieved 8/10 statements.
# Partially parsed test_find_with_symlink_following. Retrieved 10/15 statements.
# Partially parsed test_find_without_symlink_following. Retrieved 9/12 statements.
# Partially parsed test_find_with_duplicate_resolved_directory. Retrieved 13/20 statements.
# Partially parsed test_find_with_mixed_paths. Retrieved 19/21 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = []
    var_3 = []
    var_4 = 'test.py'
    var_5 = [var_4]
    var_6 = module_1.find(var_5, var_1, var_2, var_3)
    var_7 = list(var_6)
    var_8 = bool(var_7 == ['test.py'])
    assert var_8 is True
    var_9 = bool(var_2 == [])
    assert var_9 is True
    var_10 = bool(var_3 == [])
    assert var_10 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = []
    var_3 = []
    var_4 = 'missing.py'
    var_5 = [var_4]
    var_6 = module_1.find(var_5, var_1, var_2, var_3)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True
    var_9 = bool(var_2 == [])
    assert var_9 is True
    var_10 = bool(var_3 == ['missing.py'])
    assert var_10 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = True
    var_3 = []
    var_4 = []
    var_5 = 'test.py'
    var_6 = [var_5]
    var_7 = module_1.find(var_6, var_1, var_3, var_4)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [])
    assert var_9 is True
    var_10 = bool(var_4 == [])
    assert var_10 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'skip_dir'
    var_3 = True
    var_4 = []
    var_5 = []
    var_6 = [var_2]
    var_7 = module_1.find(var_6, var_1, var_4, var_5)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [])
    assert var_9 is True
    var_10 = bool(var_4 == ['skip_dir'])
    assert var_10 is True
    var_11 = bool(var_5 == [])
    assert var_11 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = False
    var_3 = '.py'
    var_4 = []
    var_5 = []
    var_6 = 'dir'
    var_7 = [var_6]
    var_8 = module_1.find(var_7, var_1, var_4, var_5)
    var_9 = list(var_8)
    var_10 = 'a.py'
    var_11 = 'b.py'
    var_12 = [var_10, var_11]
    var_13 = [os.path.join(var_6, f) for f in var_12]
    var_14 = sorted(var_9)
    var_15 = sorted(var_13)
    var_16 = bool(var_14 == var_15)
    assert var_16 is True
    var_17 = bool(var_4 == [])
    assert var_17 is True
    var_18 = bool(var_5 == [])
    assert var_18 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = False
    var_3 = []
    var_4 = []
    var_5 = 'dir'
    var_6 = [var_5]
    var_7 = module_1.find(var_6, var_1, var_3, var_4)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [])
    assert var_9 is True
    var_10 = bool(var_3 == [])
    assert var_10 is True
    var_11 = bool(var_4 == [])
    assert var_11 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = False
    var_3 = '.py'
    var_4 = []
    var_5 = []
    var_6 = 'symlink_dir'
    var_7 = [var_6]
    var_8 = module_1.find(var_7, var_1, var_4, var_5)
    var_9 = list(var_8)
    var_10 = 'linked.py'
    var_11 = bool(var_4 == [])
    assert var_11 is True
    var_12 = bool(var_5 == [])
    assert var_12 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = False
    var_3 = '.py'
    var_4 = []
    var_5 = []
    var_6 = 'symlink_dir'
    var_7 = [var_6]
    var_8 = module_1.find(var_7, var_1, var_4, var_5)
    var_9 = list(var_8)
    var_10 = bool(var_9 == [])
    assert var_10 is True
    var_11 = bool(var_4 == [])
    assert var_11 is True
    var_12 = bool(var_5 == [])
    assert var_12 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = False
    var_3 = '.py'
    var_4 = []
    var_5 = []
    var_6 = 'dir'
    var_7 = 'symlink_to_dir'
    var_8 = [var_6, var_7]
    var_9 = module_1.find(var_8, var_1, var_4, var_5)
    var_10 = list(var_9)
    var_11 = 'a.py'
    var_12 = 'b.py'
    var_13 = sorted(var_10)
    var_14 = bool(var_4 == [])
    assert var_14 is True
    var_15 = bool(var_5 == [])
    assert var_15 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'skip'
    var_3 = '.py'
    var_4 = []
    var_5 = []
    var_6 = 'file1.py'
    var_7 = 'missing.py'
    var_8 = 'dir'
    var_9 = [var_6, var_2, var_7, var_8]
    var_10 = module_1.find(var_9, var_1, var_4, var_5)
    var_11 = list(var_10)
    var_12 = [var_6]
    var_13 = 'a.py'
    var_14 = 'b.py'
    var_15 = [var_13, var_14]
    var_16 = [os.path.join(var_8, f) for f in var_15]
    var_17 = var_12 + var_16
    var_18 = sorted(var_11)
    var_19 = sorted(var_17)
    var_20 = bool(var_18 == var_19)
    assert var_20 is True
    var_21 = bool(var_4 == ['skip'])
    assert var_21 is True
    var_22 = bool(var_5 == ['missing.py'])
    assert var_22 is True



# Parsed testcases at query #2
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '/fake/nonexistent/file.py'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True
    var_9 = bool(var_5 == ['/fake/nonexistent/file.py'])
    assert var_9 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_find_with_skipped_file. Retrieved 7/8 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 7/8 statements.
# Partially parsed test_find_with_follow_links. Retrieved 7/8 statements.
# Partially parsed test_find_without_follow_links. Retrieved 7/8 statements.
# Partially parsed test_find_with_unsupported_filetype. Retrieved 8/9 statements.
# Partially parsed test_find_with_supported_filetype_in_directory. Retrieved 8/9 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = []
    var_3 = []
    var_4 = 'test.py'
    var_5 = [var_4]
    var_6 = module_1.find(var_5, var_1, var_2, var_3)
    var_7 = list(var_6)
    var_8 = bool(var_7 == ['test.py'])
    assert var_8 is True
    var_9 = bool(var_2 == [])
    assert var_9 is True
    var_10 = bool(var_3 == [])
    assert var_10 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = []
    var_3 = []
    var_4 = 'nonexistent.py'
    var_5 = [var_4]
    var_6 = module_1.find(var_5, var_1, var_2, var_3)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True
    var_9 = bool(var_2 == [])
    assert var_9 is True
    var_10 = bool(var_3 == ['nonexistent.py'])
    assert var_10 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = []
    var_3 = []
    var_4 = 'test_dir'
    var_5 = [var_4]
    var_6 = module_1.find(var_5, var_1, var_2, var_3)
    var_7 = list(var_6)
    var_8 = 'test_dir/file1.py'
    var_9 = bool('test_dir/file1.py' in var_7)
    assert var_9 is True
    var_10 = 'test_dir/file2.py'
    var_11 = bool('test_dir/file2.py' in var_7)
    assert var_11 is True
    var_12 = bool(var_2 == [])
    assert var_12 is True
    var_13 = bool(var_3 == [])
    assert var_13 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'skipped.py'
    var_3 = []
    var_4 = []
    var_5 = [var_2]
    var_6 = module_1.find(var_5, var_1, var_3, var_4)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True
    var_9 = bool(var_3 == ['skipped.py'])
    assert var_9 is True
    var_10 = bool(var_4 == [])
    assert var_10 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'skipped_dir'
    var_3 = []
    var_4 = []
    var_5 = [var_2]
    var_6 = module_1.find(var_5, var_1, var_3, var_4)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True
    var_9 = bool(var_3 == ['skipped_dir'])
    assert var_9 is True
    var_10 = bool(var_4 == [])
    assert var_10 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = []
    var_3 = []
    var_4 = 'file1.py'
    var_5 = 'nonexistent.py'
    var_6 = 'test_dir'
    var_7 = [var_4, var_5, var_6]
    var_8 = module_1.find(var_7, var_1, var_2, var_3)
    var_9 = list(var_8)
    var_10 = 'file1.py'
    var_11 = bool('file1.py' in var_9)
    assert var_11 is True
    var_12 = 'test_dir/file1.py'
    var_13 = bool('test_dir/file1.py' in var_9)
    assert var_13 is True
    var_14 = 'test_dir/file2.py'
    var_15 = bool('test_dir/file2.py' in var_9)
    assert var_15 is True
    var_16 = bool(var_3 == ['nonexistent.py'])
    assert var_16 is True
    var_17 = bool(var_2 == [])
    assert var_17 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = []
    var_3 = []
    var_4 = 'linked_dir'
    var_5 = [var_4]
    var_6 = module_1.find(var_5, var_1, var_2, var_3)
    var_7 = list(var_6)
    var_8 = 'linked_dir/file.py'
    var_9 = bool('linked_dir/file.py' in var_7)
    assert var_9 is True
    var_10 = bool(var_2 == [])
    assert var_10 is True
    var_11 = bool(var_3 == [])
    assert var_11 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = []
    var_3 = []
    var_4 = 'linked_dir'
    var_5 = [var_4]
    var_6 = module_1.find(var_5, var_1, var_2, var_3)
    var_7 = list(var_6)
    var_8 = 'linked_dir/file.py'
    var_9 = bool('linked_dir/file.py' in var_7)
    assert var_9 is True
    var_10 = bool(var_2 == [])
    assert var_10 is True
    var_11 = bool(var_3 == [])
    assert var_11 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '.py'
    var_3 = []
    var_4 = []
    var_5 = 'test.txt'
    var_6 = [var_5]
    var_7 = module_1.find(var_6, var_1, var_3, var_4)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [])
    assert var_9 is True
    var_10 = bool(var_3 == [])
    assert var_10 is True
    var_11 = bool(var_4 == [])
    assert var_11 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '.py'
    var_3 = []
    var_4 = []
    var_5 = 'mixed_dir'
    var_6 = [var_5]
    var_7 = module_1.find(var_6, var_1, var_3, var_4)
    var_8 = list(var_7)
    var_9 = 'mixed_dir/script.py'
    var_10 = bool('mixed_dir/script.py' in var_8)
    assert var_10 is True
    var_11 = 'mixed_dir/data.txt'
    var_12 = bool('mixed_dir/data.txt' not in var_8)
    assert var_12 is True
    var_13 = bool(var_3 == [])
    assert var_13 is True
    var_14 = bool(var_4 == [])
    assert var_14 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_find_with_directory_and_skipped_path. Retrieved 12/38 statements.
# Partially parsed test_find_with_nonexistent_path. Retrieved 6/17 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 8/29 statements.
# Partially parsed test_find_with_direct_file_path. Retrieved 6/23 statements.
# Partially parsed test_find_with_unsupported_filetype. Retrieved 6/23 statements.
# Partially parsed test_find_with_follow_links_and_visited_dirs. Retrieved 9/30 statements.


def test_case_0():
    var_0 = 'subdir'
    var_1 = 'file1.py'
    var_2 = 'file2.py'
    var_3 = 'skipped.py'
    var_4 = ''
    var_5 = False
    var_6 = '.py'
    var_7 = [var_6]
    var_8 = []
    var_9 = []
    var_10 = len(var_8)
    assert var_10 == 1
    var_11 = var_8[0]
    var_12 = len(var_9)
    assert var_12 == 0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '/nonexistent/path'
    var_3 = [var_2]
    var_4 = len(var_0)
    assert var_4 == 0
    var_5 = len(var_1)
    assert var_5 == 1
    var_6 = var_1[0]
    assert var_6 == '/nonexistent/path'

def test_case_0():
    var_0 = 'skipped'
    var_1 = 'file.py'
    var_2 = ''
    var_3 = False
    var_4 = []
    var_5 = []
    var_6 = len(var_4)
    assert var_6 == 1
    var_7 = var_4[0]
    var_8 = len(var_5)
    assert var_8 == 0

def test_case_0():
    var_0 = 'test.txt'
    var_1 = ''
    var_2 = []
    var_3 = []
    var_4 = len(var_2)
    assert var_4 == 0
    var_5 = len(var_3)
    assert var_5 == 0

def test_case_0():
    var_0 = 'test.txt'
    var_1 = ''
    var_2 = []
    var_3 = []
    var_4 = len(var_2)
    assert var_4 == 0
    var_5 = len(var_3)
    assert var_5 == 0

def test_case_0():
    var_0 = 'link'
    var_1 = 'target'
    var_2 = 'file.py'
    var_3 = ''
    var_4 = True
    var_5 = []
    var_6 = []
    var_7 = len(var_5)
    assert var_7 == 0
    var_8 = len(var_6)
    assert var_8 == 0



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_find_with_directory_and_skipped_path. Retrieved 7/34 statements.
# Partially parsed test_find_with_nonexistent_path. Retrieved 4/15 statements.
# Partially parsed test_find_with_single_file. Retrieved 4/21 statements.
# Partially parsed test_find_with_follow_links_and_visited_dirs. Retrieved 7/29 statements.
# Partially parsed test_find_with_unsupported_filetype. Retrieved 4/21 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 4/26 statements.
# Partially parsed test_find_with_multiple_paths. Retrieved 7/32 statements.


def test_case_0():
    var_0 = 'subdir'
    var_1 = 'file1.py'
    var_2 = 'file2.py'
    var_3 = 'file3.txt'
    var_4 = 'w'
    var_5 = []
    var_6 = []
    var_7 = bool(var_6 == [])
    assert var_7 is True

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '/nonexistent/path'
    var_3 = [var_2]
    var_4 = bool(var_0 == [])
    assert var_4 is True
    var_5 = bool(var_1 == ['/nonexistent/path'])
    assert var_5 is True

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'w'
    var_2 = []
    var_3 = []
    var_4 = bool(var_2 == [])
    assert var_4 is True
    var_5 = bool(var_3 == [])
    assert var_5 is True

def test_case_0():
    var_0 = 'real'
    var_1 = 'link'
    var_2 = True
    var_3 = 'file.py'
    var_4 = 'w'
    var_5 = []
    var_6 = []
    var_7 = bool(var_5 == [])
    assert var_7 is True
    var_8 = bool(var_6 == [])
    assert var_8 is True

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'w'
    var_2 = []
    var_3 = []
    var_4 = bool(var_2 == [])
    assert var_4 is True
    var_5 = bool(var_3 == [])
    assert var_5 is True

def test_case_0():
    var_0 = 'skipped.py'
    var_1 = 'w'
    var_2 = []
    var_3 = []
    var_4 = bool(var_3 == [])
    assert var_4 is True

def test_case_0():
    var_0 = 'dir1'
    var_1 = 'dir2'
    var_2 = 'file1.py'
    var_3 = 'file2.py'
    var_4 = 'w'
    var_5 = []
    var_6 = []
    var_7 = bool(var_5 == [])
    assert var_7 is True
    var_8 = bool(var_6 == [])
    assert var_8 is True



# Parsed testcases at query #6
#--------------------------




import isort.files as module_0

def test_case_0():
    var_0 = '/fake/file.py'
    var_1 = [var_0]
    var_2 = 'Config'
    var_3 = ()
    var_4 = 'follow_links'
    var_5 = False
    var_6 = {var_4: var_5}
    var_7 = type(var_2, var_3, var_6)
    var_8 = var_7()
    var_9 = []
    var_10 = []
    var_11 = module_0.find(var_1, var_8, var_9, var_10)
    var_12 = list(var_11)
    var_13 = bool(var_12 == ['/fake/file.py'])
    assert var_13 is True
    var_14 = bool(var_9 == [])
    assert var_14 is True
    var_15 = bool(var_10 == [])
    assert var_15 is True



# Parsed testcases at query #7
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '/non/existent/path'
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
    assert var_9 == '/non/existent/path'
    var_10 = len(var_7)
    assert var_10 == 0



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_predicate_at_line_27_evaluates_to_true. Retrieved 15/37 statements.


def test_case_0():
    var_0 = '.py'
    var_1 = [var_0]
    var_2 = '/abs/path/to/skipped.py'
    var_3 = [var_2]
    var_4 = []
    var_5 = []
    var_6 = '/some/dir'
    var_7 = [var_6]
    var_8 = '/some/dir'
    var_9 = []
    var_10 = 'skipped.py'
    var_11 = [var_10]
    var_12 = (var_8, var_9, var_11)
    var_13 = list(var_8)
    var_14 = '/some/dir/skipped.py'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_predicate_at_line_27_evaluates_to_true. Retrieved 8/35 statements.


def test_case_0():
    var_0 = 'subdir'
    var_1 = 'test.py'
    var_2 = ''
    var_3 = False
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = []
    var_7 = []



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_find_with_directory_and_skipped_path. Retrieved 11/36 statements.
# Partially parsed test_find_with_single_file. Retrieved 4/18 statements.
# Partially parsed test_find_with_broken_path. Retrieved 5/13 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 5/19 statements.
# Partially parsed test_find_with_unsupported_filetype. Retrieved 4/17 statements.
# Partially parsed test_find_with_follow_links_and_visited_dirs. Retrieved 8/30 statements.
# Partially parsed test_find_with_multiple_paths. Retrieved 11/42 statements.


def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = 'skip_dir'
    assert var_3 == 1
    var_4 = 'normal_dir'
    var_5 = 0
    var_6 = 'normal_dir'
    var_7 = 'b.py'
    var_8 = len(var_1)
    assert var_8 == 1
    var_9 = var_1[var_5]
    var_10 = 'skip_dir'

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = 'test.py'
    assert var_3 == 1
    var_4 = bool(var_1 == [])
    assert var_4 is True
    var_5 = bool(var_2 == [])
    assert var_5 is True

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = '/non/existent/path'
    var_4 = [var_3]
    var_5 = bool(var_1 == [])
    assert var_5 is True
    var_6 = bool(var_2 == ['/non/existent/path'])
    assert var_6 is True

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = 'skip.py'
    var_4 = len(var_1)
    assert var_4 == 1
    var_5 = var_1[0]
    var_6 = bool(var_2 == [])
    assert var_6 is True

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = 'test.txt'
    var_4 = bool(var_1 == [])
    assert var_4 is True
    var_5 = bool(var_2 == [])
    assert var_5 is True

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = []
    var_3 = 'dir'
    assert var_3 == 1
    var_4 = 'link'
    var_5 = 0
    var_6 = 'dir'
    var_7 = 'a.py'
    var_8 = bool(var_1 == [])
    assert var_8 is True
    var_9 = bool(var_2 == [])
    assert var_9 is True

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = 'dir1'
    assert var_3 == 2
    var_4 = 'dir2'
    var_5 = 'a.py'
    var_6 = 'b.py'
    var_7 = 'dir1'
    var_8 = 'a.py'
    var_9 = 'dir2'
    var_10 = 'b.py'
    var_11 = bool(var_1 == [])
    assert var_11 is True
    var_12 = bool(var_2 == [])
    assert var_12 is True



