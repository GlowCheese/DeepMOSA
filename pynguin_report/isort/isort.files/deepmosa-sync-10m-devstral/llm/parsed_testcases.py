####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_find_with_single_file. Retrieved 8/11 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 8/14 statements.
# Partially parsed test_find_with_directory. Retrieved 12/20 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 9/19 statements.
# Partially parsed test_find_with_nested_directories. Retrieved 13/22 statements.
# Partially parsed test_find_with_symlink. Retrieved 11/18 statements.
# Partially parsed test_find_with_unsupported_filetype. Retrieved 8/11 statements.
# Partially parsed test_find_with_mixed_paths. Retrieved 12/20 statements.


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
    var_1 = '# test1'
    var_2 = '# test2'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Config(**var_4)
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_3, var_5, var_6, var_7)
    var_9 = list(var_8)
    var_10 = set(var_9)
    var_11 = bool(var_10 == {'test_dir/file1.py', 'test_dir/file2.py'})
    assert var_11 is True
    var_12 = 'test_dir/file1.py'
    var_13 = 'test_dir/file2.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'skip_dir'
    var_1 = '# skip'
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = [var_1]
    var_6 = []
    var_7 = module_1.find(var_5, var_3, var_4, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [])
    assert var_9 is True
    var_10 = [var_1]
    var_11 = 'skip_dir/file.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'parent/child'
    var_1 = '# parent'
    var_2 = '# child'
    var_3 = 'parent'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Config(**var_5)
    var_7 = []
    var_8 = []
    var_9 = module_1.find(var_4, var_6, var_7, var_8)
    var_10 = list(var_9)
    var_11 = set(var_10)
    var_12 = bool(var_11 == {'parent/file1.py', 'parent/child/file2.py'})
    assert var_12 is True
    var_13 = 'parent/file1.py'
    var_14 = 'parent/child/file2.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'real_dir'
    var_1 = '# real'
    var_2 = 'link_dir'
    var_3 = [var_2]
    var_4 = True
    var_5 = 'follow_links'
    var_6 = {var_5: var_4}
    var_7 = module_0.Config(**var_6)
    var_8 = []
    var_9 = []
    var_10 = module_1.find(var_3, var_7, var_8, var_9)
    var_11 = list(var_10)
    var_12 = bool(var_11 == ['link_dir/file.py'])
    assert var_12 is True
    var_13 = 'real_dir/file.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '# not python'
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

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '# file'
    var_1 = 'dir'
    var_2 = '# dir file'
    var_3 = 'file.py'
    var_4 = [var_3, var_2]
    var_5 = {}
    var_6 = module_0.Config(**var_5)
    var_7 = []
    var_8 = []
    var_9 = module_1.find(var_4, var_6, var_7, var_8)
    var_10 = list(var_9)
    var_11 = set(var_10)
    var_12 = bool(var_11 == {'file.py', 'dir/file.py'})
    assert var_12 is True
    var_13 = 'dir/file.py'



# Parsed testcases at query #2
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'valid_dir'
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
    var_16 = 'valid_dir/file1.py'
    var_17 = bool('valid_dir/file1.py' in var_14)
    assert var_17 is True
    var_18 = 'valid_dir/file2.py'
    var_19 = bool('valid_dir/file2.py' in var_14)
    assert var_19 is True
    var_20 = bool(var_11 == [])
    assert var_20 is True
    var_21 = bool(var_12 == [])
    assert var_21 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'dir_with_skipped'
    var_1 = [var_0]
    var_2 = False
    var_3 = 'dir_with_skipped/skipped_subdir'
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
    var_17 = 'dir_with_skipped/valid_file.py'
    var_18 = bool('dir_with_skipped/valid_file.py' in var_15)
    assert var_18 is True
    var_19 = 'dir_with_skipped/skipped_subdir'
    var_20 = bool('dir_with_skipped/skipped_subdir' in var_12)
    assert var_20 is True
    var_21 = bool(var_13 == [])
    assert var_21 is True

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
    var_0 = 'single_file.py'
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
    assert var_15 == 1
    var_16 = 'single_file.py'
    var_17 = bool('single_file.py' in var_14)
    assert var_17 is True
    var_18 = bool(var_11 == [])
    assert var_18 is True
    var_19 = bool(var_12 == [])
    assert var_19 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'dir_with_unsupported'
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
    assert var_15 == 1
    var_16 = 'dir_with_unsupported/file.py'
    var_17 = bool('dir_with_unsupported/file.py' in var_14)
    assert var_17 is True
    var_18 = bool(var_11 == [])
    assert var_18 is True
    var_19 = bool(var_12 == [])
    assert var_19 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'dir_with_symlink'
    var_1 = [var_0]
    var_2 = True
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
    assert var_15 == 1
    var_16 = 'dir_with_symlink/file.py'
    var_17 = bool('dir_with_symlink/file.py' in var_14)
    assert var_17 is True
    var_18 = bool(var_11 == [])
    assert var_18 is True
    var_19 = bool(var_12 == [])
    assert var_19 is True



# Parsed testcases at query #3
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
    var_9 = bool(var_7 == [])
    assert var_9 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_predicate_at_line_31_evaluates_to_true. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'non_existent_path'



# Parsed testcases at query #5
#--------------------------




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
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == ['test_dir/file1.py'])
    assert var_8 is True
    var_9 = bool(var_4 == ['test_dir/file2.py'])
    assert var_9 is True
    var_10 = bool(var_5 == [])
    assert var_10 is True

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
    var_0 = 'test_symlink'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'follow_links'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_1, var_5, var_6, var_7)
    var_9 = list(var_8)
    var_10 = bool(var_9 == ['test_symlink/file1.py', 'test_symlink/file2.py'])
    assert var_10 is True
    var_11 = bool(var_6 == [])
    assert var_11 is True
    var_12 = bool(var_7 == [])
    assert var_12 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_symlink'
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
    var_12 = bool(var_7 == [])
    assert var_12 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_find_with_single_file. Retrieved 8/11 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 8/14 statements.
# Partially parsed test_find_with_directory. Retrieved 12/20 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 11/22 statements.
# Partially parsed test_find_with_follow_links_disabled. Retrieved 12/21 statements.
# Partially parsed test_find_with_follow_links_enabled. Retrieved 12/21 statements.
# Partially parsed test_find_with_unsupported_filetype. Retrieved 8/11 statements.
# Partially parsed test_find_with_mixed_paths. Retrieved 12/20 statements.
# Partially parsed test_find_with_circular_symlink. Retrieved 9/13 statements.


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
    var_0 = '# test'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = 'skipped_file.py'
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
    var_1 = '# test1'
    var_2 = '# test2'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Config(**var_4)
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_3, var_5, var_6, var_7)
    var_9 = list(var_8)
    var_10 = set(var_9)
    var_11 = bool(var_10 == {'test_dir/file1.py', 'test_dir/file2.py'})
    assert var_11 is True
    var_12 = 'test_dir/file1.py'
    var_13 = 'test_dir/file2.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir/skipped_dir'
    var_1 = '# test'
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = 'skipped_dir'
    var_5 = []
    var_6 = 'test_dir'
    var_7 = [var_6]
    var_8 = []
    var_9 = module_1.find(var_7, var_3, var_5, var_8)
    var_10 = list(var_9)
    var_11 = bool(var_10 == [])
    assert var_11 is True
    var_12 = [var_1]
    var_13 = 'test_dir/skipped_dir/file.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_dir/subdir'
    var_2 = '# test'
    var_3 = 'test_dir/link'
    var_4 = False
    var_5 = 'follow_links'
    var_6 = {var_5: var_4}
    var_7 = module_0.Config(**var_6)
    var_8 = [var_2]
    var_9 = []
    var_10 = []
    var_11 = module_1.find(var_8, var_7, var_9, var_10)
    var_12 = list(var_11)
    var_13 = 'test_dir/link/file.py'
    var_14 = bool('test_dir/link/file.py' not in var_12)
    assert var_14 is True
    var_15 = 'test_dir/subdir/file.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_dir/subdir'
    var_2 = '# test'
    var_3 = 'test_dir/link'
    var_4 = True
    var_5 = 'follow_links'
    var_6 = {var_5: var_4}
    var_7 = module_0.Config(**var_6)
    var_8 = [var_2]
    var_9 = []
    var_10 = []
    var_11 = module_1.find(var_8, var_7, var_9, var_10)
    var_12 = list(var_11)
    var_13 = 'test_dir/link/file.py'
    var_14 = bool('test_dir/link/file.py' in var_12)
    assert var_14 is True
    var_15 = 'test_dir/subdir/file.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '# test'
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

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '# test'
    var_1 = 'dir'
    var_2 = '# test'
    var_3 = 'file.py'
    var_4 = [var_3, var_2]
    var_5 = {}
    var_6 = module_0.Config(**var_5)
    var_7 = []
    var_8 = []
    var_9 = module_1.find(var_4, var_6, var_7, var_8)
    var_10 = list(var_9)
    var_11 = set(var_10)
    var_12 = bool(var_11 == {'file.py', 'dir/file.py'})
    assert var_12 is True
    var_13 = 'dir/file.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_dir/link'
    var_2 = True
    var_3 = 'follow_links'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = [var_0]
    var_7 = []
    var_8 = []
    var_9 = module_1.find(var_6, var_5, var_7, var_8)
    var_10 = list(var_9)
    var_11 = bool(var_10 == [])
    assert var_11 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'nonexistent_path'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_isdir_predicate_evaluates_to_true. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'test_directory'
    var_1 = True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_isdir_predicate_evaluates_to_true. Retrieved 10/21 statements.


def test_case_0():
    var_0 = '/existing_directory'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = True
    var_5 = []
    var_6 = []
    var_7 = (var_0, var_5, var_6)
    var_8 = [var_7]
    var_9 = False



# Parsed testcases at query #10
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



# Parsed testcases at query #11
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
    var_16 = len(var_15)
    assert var_16 == 1
    var_17 = 'test_dir/file1.py'
    var_18 = bool('test_dir/file1.py' in var_15)
    assert var_18 is True
    var_19 = 'test_dir/file2.py'
    var_20 = bool('test_dir/file2.py' not in var_15)
    assert var_20 is True
    var_21 = bool(var_12 == ['test_dir/skip_me'])
    assert var_21 is True
    var_22 = bool(var_13 == [])
    assert var_22 is True

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
    var_0 = 'test_dir/file1.py'
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
    assert var_15 == 1
    var_16 = 'test_dir/file1.py'
    var_17 = bool('test_dir/file1.py' in var_14)
    assert var_17 is True
    var_18 = bool(var_11 == [])
    assert var_18 is True
    var_19 = bool(var_12 == [])
    assert var_19 is True

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



# Parsed testcases at query #12
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



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_isdir_predicate_evaluates_to_true. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'test_directory'
    var_1 = True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_find_with_single_file. Retrieved 8/11 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 8/14 statements.
# Partially parsed test_find_with_directory. Retrieved 9/14 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 9/19 statements.
# Partially parsed test_find_with_non_python_file. Retrieved 8/11 statements.
# Partially parsed test_find_with_follow_links_disabled. Retrieved 12/21 statements.
# Partially parsed test_find_with_circular_symlink. Retrieved 13/22 statements.


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
    var_1 = '# test'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = []
    var_6 = []
    var_7 = module_1.find(var_2, var_4, var_5, var_6)
    var_8 = set(var_7)
    var_9 = bool(var_8 == {'test_dir/test_file.py'})
    assert var_9 is True
    var_10 = 'test_dir/test_file.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = '# test'
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = [var_1]
    var_6 = []
    var_7 = module_1.find(var_5, var_3, var_4, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [])
    assert var_9 is True
    var_10 = [var_1]
    var_11 = 'test_dir/test_file.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '# test'
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

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_dir/subdir'
    var_2 = '# test'
    var_3 = 'test_dir/symlink'
    var_4 = False
    var_5 = 'follow_links'
    var_6 = {var_5: var_4}
    var_7 = module_0.Config(**var_6)
    var_8 = [var_2]
    var_9 = []
    var_10 = []
    var_11 = module_1.find(var_8, var_7, var_9, var_10)
    var_12 = set(var_11)
    var_13 = bool(var_12 == {'test_dir/subdir/test_file.py'})
    assert var_13 is True
    var_14 = 'test_dir/subdir/test_file.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_dir/subdir'
    var_2 = '# test'
    var_3 = '../test_dir'
    var_4 = 'test_dir/subdir/circular_link'
    var_5 = True
    var_6 = 'follow_links'
    var_7 = {var_6: var_5}
    var_8 = module_0.Config(**var_7)
    var_9 = [var_2]
    var_10 = []
    var_11 = []
    var_12 = module_1.find(var_9, var_8, var_10, var_11)
    var_13 = set(var_12)
    var_14 = bool(var_13 == {'test_dir/subdir/test_file.py'})
    assert var_14 is True
    var_15 = 'test_dir/subdir/test_file.py'



# Parsed testcases at query #15
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



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_find_with_single_file. Retrieved 8/11 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 8/14 statements.
# Partially parsed test_find_with_directory. Retrieved 12/20 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 9/19 statements.
# Partially parsed test_find_with_follow_links_disabled. Retrieved 10/15 statements.
# Partially parsed test_find_with_mixed_paths. Retrieved 11/19 statements.


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
    var_1 = '# test1'
    var_2 = '# test2'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Config(**var_4)
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_3, var_5, var_6, var_7)
    var_9 = list(var_8)
    var_10 = set(var_9)
    var_11 = bool(var_10 == {'test_dir/file1.py', 'test_dir/file2.py'})
    assert var_11 is True
    var_12 = 'test_dir/file1.py'
    var_13 = 'test_dir/file2.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'skip_dir'
    var_1 = '# skip'
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = [var_1]
    var_6 = []
    var_7 = module_1.find(var_5, var_3, var_4, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [])
    assert var_9 is True
    var_10 = [var_1]
    var_11 = 'skip_dir/file.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'link_dir'
    var_1 = '# test'
    var_2 = False
    var_3 = 'follow_links'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = [var_1]
    var_7 = []
    var_8 = []
    var_9 = module_1.find(var_6, var_5, var_7, var_8)
    var_10 = list(var_9)
    var_11 = bool(var_10 == ['link_dir/file.py'])
    assert var_11 is True
    var_12 = 'link_dir/file.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'mixed_dir'
    var_1 = '# valid'
    var_2 = '# invalid'
    var_3 = 'invalid.txt'
    var_4 = [var_2, var_3]
    var_5 = {}
    var_6 = module_0.Config(**var_5)
    var_7 = []
    var_8 = []
    var_9 = module_1.find(var_4, var_6, var_7, var_8)
    var_10 = list(var_9)
    var_11 = bool(var_10 == ['mixed_dir/valid.py'])
    assert var_11 is True
    var_12 = 'mixed_dir/valid.py'



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



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_find_with_skipped_file. Retrieved 8/11 statements.
# Partially parsed test_find_with_directory. Retrieved 10/16 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 12/19 statements.
# Partially parsed test_find_with_symlink_loop. Retrieved 11/17 statements.


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
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'skip_*'
    var_3 = []
    var_4 = 'skip_me.py'
    var_5 = [var_4]
    var_6 = []
    var_7 = module_1.find(var_5, var_1, var_3, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [])
    assert var_9 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = True
    var_2 = ''
    var_3 = ''
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Config(**var_5)
    var_7 = []
    var_8 = []
    var_9 = module_1.find(var_4, var_6, var_7, var_8)
    var_10 = list(var_9)
    var_11 = 'test_dir/file1.py'
    var_12 = bool('test_dir/file1.py' in var_10)
    assert var_12 is True
    var_13 = 'test_dir/file2.txt'
    var_14 = bool('test_dir/file2.txt' not in var_10)
    assert var_14 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir/skip_dir'
    var_1 = True
    var_2 = ''
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = 'skip_*'
    var_6 = []
    var_7 = 'test_dir'
    var_8 = [var_7]
    var_9 = []
    var_10 = module_1.find(var_8, var_4, var_6, var_9)
    var_11 = list(var_10)
    var_12 = bool(var_11 == [])
    assert var_12 is True
    var_13 = 'skip_dir'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'dir1'
    var_1 = True
    var_2 = 'dir2'
    var_3 = 'dir1/link_to_dir2'
    var_4 = 'dir2/link_to_dir1'
    var_5 = [var_0]
    var_6 = {}
    var_7 = module_0.Config(**var_6)
    var_8 = []
    var_9 = []
    var_10 = module_1.find(var_5, var_7, var_8, var_9)
    var_11 = list(var_10)
    var_12 = bool(var_11 == [])
    assert var_12 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_os_path_isdir_evaluates_to_true. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'test_directory'



# Parsed testcases at query #20
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'file.txt'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = bool(var_7 == ['file.txt'])
    assert var_8 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_predicate_at_line_9_evaluates_to_true. Retrieved 10/22 statements.


def test_case_0():
    var_0 = '/test_dir'
    var_1 = 'subdir'
    var_2 = [var_1]
    var_3 = 'file1.py'
    var_4 = [var_3]
    var_5 = (var_0, var_2, var_4)
    var_6 = [var_0]
    var_7 = []
    var_8 = []
    var_9 = True



# Parsed testcases at query #22
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



# Parsed testcases at query #23
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
    var_15 = bool(var_6 == [])
    assert var_15 is True
    var_16 = bool(var_7 == [])
    assert var_16 is True

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
    var_18 = bool(var_9 == ['test_dir/subdir'])
    assert var_18 is True
    var_19 = bool(var_10 == [])
    assert var_19 is True

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
    var_11 = bool(var_6 == [])
    assert var_11 is True
    var_12 = bool(var_7 == ['nonexistent_path'])
    assert var_12 is True

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
    var_13 = bool(var_6 == [])
    assert var_13 is True
    var_14 = bool(var_7 == [])
    assert var_14 is True

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
    var_18 = bool(var_9 == ['test_dir/file1.py'])
    assert var_18 is True
    var_19 = bool(var_10 == [])
    assert var_19 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = False
    var_3 = '.txt'
    var_4 = [var_3]
    var_5 = 'follow_links'
    var_6 = 'supported_filetypes'
    var_7 = {var_5: var_2, var_6: var_4}
    var_8 = module_0.Config(**var_7)
    var_9 = []
    var_10 = []
    var_11 = module_1.find(var_1, var_8, var_9, var_10)
    var_12 = list(var_11)
    var_13 = len(var_12)
    assert var_13 == 0
    var_14 = bool(var_9 == [])
    assert var_14 is True
    var_15 = bool(var_10 == [])
    assert var_15 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_find_with_single_file. Retrieved 8/11 statements.
# Partially parsed test_find_with_directory_containing_python_files. Retrieved 11/19 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 11/18 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 8/14 statements.
# Partially parsed test_find_with_symlink_loop. Retrieved 13/22 statements.


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
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Config(**var_4)
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_3, var_5, var_6, var_7)
    var_9 = sorted(var_8)
    var_10 = bool(var_9 == ['test_dir/file1.py'])
    assert var_10 is True
    var_11 = 'test_dir/file1.py'
    var_12 = 'test_dir/file2.txt'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir/skip_me'
    var_1 = '# test'
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = 'skip_me'
    var_5 = []
    var_6 = 'test_dir'
    var_7 = [var_6]
    var_8 = []
    var_9 = module_1.find(var_7, var_3, var_5, var_8)
    var_10 = list(var_9)
    var_11 = bool(var_10 == ['test_dir/file.py'])
    assert var_11 is True
    var_12 = bool(var_5 == ['test_dir/skip_me'])
    assert var_12 is True
    var_13 = 'test_dir/file.py'

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
    var_14 = bool(var_13 == ['test_dir/subdir/file.py'])
    assert var_14 is True
    var_15 = 'test_dir/subdir/file.py'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_os_path_isdir_returns_true. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'some_path'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_os_path_isdir_evaluates_to_true. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'test_directory'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_find_with_skipped_file. Retrieved 8/11 statements.
# Partially parsed test_find_with_directory_and_files. Retrieved 8/21 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 6/17 statements.
# Partially parsed test_find_with_broken_symlink. Retrieved 5/11 statements.
# Partially parsed test_find_with_follow_links_disabled. Retrieved 8/20 statements.
# Partially parsed test_find_with_unsupported_filetype. Retrieved 6/10 statements.


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
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'skip_*'
    var_3 = []
    var_4 = 'skip_me.py'
    var_5 = [var_4]
    var_6 = []
    var_7 = module_1.find(var_5, var_1, var_3, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [])
    assert var_9 is True

import isort.settings as module_0

def test_case_0():
    var_0 = 'subdir'
    var_1 = ''
    var_2 = ''
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = []
    var_6 = []
    var_7 = 'file1.py'
    var_8 = 'file2.py'

import isort.settings as module_0

def test_case_0():
    var_0 = 'skip_dir'
    var_1 = ''
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = 'skip_*'
    var_5 = []
    var_6 = []

import isort.settings as module_0

def test_case_0():
    var_0 = 'broken_link'
    var_1 = 'nonexistent'
    var_2 = []
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = []

import isort.settings as module_0

def test_case_0():
    var_0 = 'subdir'
    var_1 = ''
    var_2 = 'link'
    var_3 = False
    var_4 = 'follow_links'
    var_5 = {var_4: var_3}
    var_6 = module_0.Config(**var_5)
    var_7 = []
    var_8 = []
    var_9 = 'file.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_0, var_2, var_3, var_4)
    var_6 = list(var_5)
    var_7 = bool(var_6 == [])
    assert var_7 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_find_with_directory_skipped_files. Retrieved 8/9 statements.
# Partially parsed test_find_with_directory_broken_links. Retrieved 7/8 statements.


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
    var_0 = '/nonexistent/path'
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
    var_10 = bool(var_5 == ['/nonexistent/path'])
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
    var_4 = 'skip_*'
    var_5 = []
    var_6 = []
    var_7 = module_1.find(var_1, var_3, var_5, var_6)
    var_8 = list(var_7)
    var_9 = 'test_dir/skip_file.py'
    var_10 = bool('test_dir/skip_file.py' not in var_8)
    assert var_10 is True
    var_11 = 'skip_file.py'
    var_12 = bool('skip_file.py' in var_5)
    assert var_12 is True
    var_13 = 'test_dir/valid_file.py'
    var_14 = bool('test_dir/valid_file.py' in var_8)
    assert var_14 is True

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
    var_8 = 'test_dir/broken_link.py'
    var_9 = bool('test_dir/broken_link.py' not in var_7)
    assert var_9 is True
    var_10 = 'broken_link.py'
    var_11 = bool('broken_link.py' in var_5)
    assert var_11 is True
    var_12 = 'test_dir/valid_file.py'
    var_13 = bool('test_dir/valid_file.py' in var_7)
    assert var_13 is True

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
    var_8 = 'test_dir/subdir/file.py'
    var_9 = bool('test_dir/subdir/file.py' in var_7)
    assert var_9 is True
    var_10 = 'test_dir/file.py'
    var_11 = bool('test_dir/file.py' in var_7)
    assert var_11 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0, var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = len(var_7)
    var_9 = set(var_7)
    var_10 = len(var_9)
    var_11 = bool(var_8 == var_10)
    assert var_11 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_isdir_predicate_evaluates_to_true. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'test_dir'
    var_1 = True



# Parsed testcases at query #4
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



# Parsed testcases at query #5
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'valid_directory'
    var_1 = [var_0]
    var_2 = False
    var_3 = 'follow_links'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_1, var_5, var_6, var_7)
    var_9 = list(var_8)
    var_10 = bool(var_9 == ['valid_directory/file1.py', 'valid_directory/file2.py'])
    assert var_10 is True
    var_11 = bool(var_6 == [])
    assert var_11 is True
    var_12 = bool(var_7 == [])
    assert var_12 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'directory_with_skipped'
    var_1 = [var_0]
    var_2 = False
    var_3 = 'follow_links'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_1, var_5, var_6, var_7)
    var_9 = list(var_8)
    var_10 = bool(var_9 == ['directory_with_skipped/allowed.py'])
    assert var_10 is True
    var_11 = bool(var_6 == ['directory_with_skipped/skipped.py'])
    assert var_11 is True
    var_12 = bool(var_7 == [])
    assert var_12 is True

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
    var_10 = bool(var_9 == [])
    assert var_10 is True
    var_11 = bool(var_6 == [])
    assert var_11 is True
    var_12 = bool(var_7 == ['nonexistent_path'])
    assert var_12 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'single_file.py'
    var_1 = [var_0]
    var_2 = False
    var_3 = 'follow_links'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_1, var_5, var_6, var_7)
    var_9 = list(var_8)
    var_10 = bool(var_9 == ['single_file.py'])
    assert var_10 is True
    var_11 = bool(var_6 == [])
    assert var_11 is True
    var_12 = bool(var_7 == [])
    assert var_12 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'directory_with_symlinks'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'follow_links'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_1, var_5, var_6, var_7)
    var_9 = list(var_8)
    var_10 = bool(var_9 == ['directory_with_symlinks/file.py', 'directory_with_symlinks/link_target.py'])
    assert var_10 is True
    var_11 = bool(var_6 == [])
    assert var_11 is True
    var_12 = bool(var_7 == [])
    assert var_12 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'directory_with_symlinks'
    var_1 = [var_0]
    var_2 = False
    var_3 = 'follow_links'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_1, var_5, var_6, var_7)
    var_9 = list(var_8)
    var_10 = bool(var_9 == ['directory_with_symlinks/file.py'])
    assert var_10 is True
    var_11 = bool(var_6 == [])
    assert var_11 is True
    var_12 = bool(var_7 == [])
    assert var_12 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'circular_symlink_dir'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'follow_links'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_1, var_5, var_6, var_7)
    var_9 = list(var_8)
    var_10 = bool(var_9 == ['circular_symlink_dir/file.py'])
    assert var_10 is True
    var_11 = bool(var_6 == [])
    assert var_11 is True
    var_12 = bool(var_7 == [])
    assert var_12 is True



# Parsed testcases at query #6
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



# Parsed testcases at query #7
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
    var_10 = bool(var_9 == ['test_dir/file1.py', 'test_dir/subdir/file2.py'])
    assert var_10 is True
    var_11 = bool(var_6 == [])
    assert var_11 is True
    var_12 = bool(var_7 == [])
    assert var_12 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = False
    var_3 = '*skip*'
    var_4 = [var_3]
    var_5 = 'follow_links'
    var_6 = 'skip_patterns'
    var_7 = {var_5: var_2, var_6: var_4}
    var_8 = module_0.Config(**var_7)
    var_9 = []
    var_10 = []
    var_11 = module_1.find(var_1, var_8, var_9, var_10)
    var_12 = list(var_11)
    var_13 = bool(var_12 == ['test_dir/file1.py'])
    assert var_13 is True
    var_14 = bool(var_9 == ['test_dir/skip_file.py'])
    assert var_14 is True
    var_15 = bool(var_10 == [])
    assert var_15 is True

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
    var_10 = bool(var_9 == [])
    assert var_10 is True
    var_11 = bool(var_6 == [])
    assert var_11 is True
    var_12 = bool(var_7 == ['nonexistent_path'])
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
    var_8 = module_1.find(var_1, var_5, var_6, var_7)
    var_9 = list(var_8)
    var_10 = bool(var_9 == ['test_file.py'])
    assert var_10 is True
    var_11 = bool(var_6 == [])
    assert var_11 is True
    var_12 = bool(var_7 == [])
    assert var_12 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'symlink_dir'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'follow_links'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_1, var_5, var_6, var_7)
    var_9 = list(var_8)
    var_10 = bool(var_9 == ['symlink_dir/file1.py', 'symlink_dir/subdir/file2.py'])
    assert var_10 is True
    var_11 = bool(var_6 == [])
    assert var_11 is True
    var_12 = bool(var_7 == [])
    assert var_12 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'symlink_dir'
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
    var_12 = bool(var_7 == [])
    assert var_12 is True



# Parsed testcases at query #8
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



# Parsed testcases at query #9
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_path.txt'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = len(var_5)
    assert var_7 == 1
    var_8 = var_5[0]
    assert var_8 == 'nonexistent_path.txt'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_predicate_at_line_31_evaluates_to_true. Retrieved 7/8 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = 'non_existent_file.py'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Config(**var_4)
    var_6 = []
    var_7 = module_1.find(var_3, var_5, var_6, var_0)
    var_8 = bool(var_0 == ['non_existent_file.py'])
    assert var_8 is True



# Parsed testcases at query #11
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'valid_dir'
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
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = 'valid_dir/file1.py'
    var_17 = bool('valid_dir/file1.py' in var_14)
    assert var_17 is True
    var_18 = 'valid_dir/file2.py'
    var_19 = bool('valid_dir/file2.py' in var_14)
    assert var_19 is True
    var_20 = bool(var_11 == [])
    assert var_20 is True
    var_21 = bool(var_12 == [])
    assert var_21 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'dir_with_skipped'
    var_1 = [var_0]
    var_2 = False
    var_3 = 'skip_*'
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
    var_16 = len(var_15)
    assert var_16 == 1
    var_17 = 'dir_with_skipped/keep.py'
    var_18 = bool('dir_with_skipped/keep.py' in var_15)
    assert var_18 is True
    var_19 = bool(var_12 == ['dir_with_skipped/skip_me.py'])
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
    var_0 = 'single_file.py'
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
    var_15 = bool(var_14 == ['single_file.py'])
    assert var_15 is True
    var_16 = bool(var_11 == [])
    assert var_16 is True
    var_17 = bool(var_12 == [])
    assert var_17 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'symlink_dir'
    var_1 = [var_0]
    var_2 = True
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
    var_15 = len(var_14)
    assert var_15 == 1
    var_16 = 'symlink_dir/real_file.py'
    var_17 = bool('symlink_dir/real_file.py' in var_14)
    assert var_17 is True
    var_18 = bool(var_11 == [])
    assert var_18 is True
    var_19 = bool(var_12 == [])
    assert var_19 is True



# Parsed testcases at query #12
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'valid_dir'
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
    var_15 = bool(var_14 == ['valid_dir/file1.py', 'valid_dir/file2.py'])
    assert var_15 is True
    var_16 = bool(var_11 == [])
    assert var_16 is True
    var_17 = bool(var_12 == [])
    assert var_17 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'dir_with_skipped'
    var_1 = [var_0]
    var_2 = False
    var_3 = 'dir_with_skipped/skip_me'
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
    var_16 = bool(var_15 == ['dir_with_skipped/file.py'])
    assert var_16 is True
    var_17 = bool(var_12 == ['dir_with_skipped/skip_me'])
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
    var_0 = 'file.py'
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
    var_15 = bool(var_14 == ['file.py'])
    assert var_15 is True
    var_16 = bool(var_11 == [])
    assert var_16 is True
    var_17 = bool(var_12 == [])
    assert var_17 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'dir_with_unsupported'
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
    var_17 = bool(var_12 == [])
    assert var_17 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_isdir_predicate_evaluates_to_true. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'test_directory'
    var_1 = [var_0]
    var_2 = True



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
    var_7 = list(var_6)
    var_8 = len(var_5)
    assert var_8 == 1
    var_9 = var_5[0]
    assert var_9 == 'nonexistent_path'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_find_with_skipped_file. Retrieved 8/11 statements.
# Partially parsed test_find_with_directory. Retrieved 9/13 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 12/17 statements.
# Partially parsed test_find_with_follow_links_disabled. Retrieved 12/18 statements.
# Partially parsed test_find_with_broken_symlink. Retrieved 8/16 statements.


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
    var_1 = '/nonexistent/path'
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Config(**var_3)
    var_5 = []
    var_6 = module_1.find(var_2, var_4, var_5, var_0)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True
    var_9 = bool(var_0 == ['/nonexistent/path'])
    assert var_9 is True

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
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'skip_*'
    var_3 = []
    var_4 = 'skip_me.py'
    var_5 = [var_4]
    var_6 = []
    var_7 = module_1.find(var_5, var_1, var_3, var_6)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [])
    assert var_9 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = True
    var_2 = '# test'
    var_3 = [var_2]
    var_4 = {}
    var_5 = module_0.Config(**var_4)
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_3, var_5, var_6, var_7)
    var_9 = list(var_8)
    var_10 = 'test_dir/file1.py'
    var_11 = bool('test_dir/file1.py' in var_9)
    assert var_11 is True

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir/skip_dir'
    var_1 = True
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = 'skip_*'
    var_5 = []
    var_6 = 'test_dir'
    var_7 = [var_6]
    var_8 = []
    var_9 = module_1.find(var_7, var_3, var_5, var_8)
    var_10 = list(var_9)
    var_11 = str(var_10)
    var_12 = 'skip_dir'
    var_13 = bool('skip_dir' not in var_11)
    assert var_13 is True
    var_14 = 'skip_dir'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = True
    var_2 = 'test_dir/link_target'
    var_3 = 'test_dir/link'
    var_4 = False
    var_5 = 'follow_links'
    var_6 = {var_5: var_4}
    var_7 = module_0.Config(**var_6)
    var_8 = [var_0]
    var_9 = []
    var_10 = []
    var_11 = module_1.find(var_8, var_7, var_9, var_10)
    var_12 = list(var_11)
    var_13 = 'link'

def test_case_0():
    var_0 = 'test_dir'
    var_1 = True
    var_2 = 'nonexistent'
    var_3 = 'test_dir/broken_link'
    var_4 = []
    var_5 = [var_0]
    var_6 = []
    var_7 = 'broken_link'
    var_8 = 'broken_link'



# Parsed testcases at query #16
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
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'follow_links'
    var_4 = {var_3: var_2}
    var_5 = module_0.Config(**var_4)
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_1, var_5, var_6, var_7)
    var_9 = list(var_8)
    var_10 = bool(var_9 == ['test_dir/file1.py', 'test_dir/file2.py', 'test_dir/link_file.py'])
    assert var_10 is True
    var_11 = bool(var_6 == [])
    assert var_11 is True
    var_12 = bool(var_7 == [])
    assert var_12 is True

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
    var_10 = bool(var_9 == ['test_dir/file1.py', 'test_dir/file2.py'])
    assert var_10 is True
    var_11 = bool(var_6 == [])
    assert var_11 is True
    var_12 = bool(var_7 == [])
    assert var_12 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_isdir_predicate. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'test_directory'
    var_1 = True



