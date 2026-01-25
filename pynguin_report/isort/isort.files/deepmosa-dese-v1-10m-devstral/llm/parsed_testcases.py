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
    var_6 = module_0.Config()
    var_7 = []
    var_8 = []
    var_9 = module_1.find(var_1, var_6, var_7, var_8)
    var_10 = list(var_9)
    var_11 = len(var_10)
    assert var_11 == 2

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = False
    var_3 = 'subdir'
    var_4 = [var_3]
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = module_0.Config()
    var_8 = []
    var_9 = []
    var_10 = module_1.find(var_1, var_7, var_8, var_9)
    var_11 = list(var_10)
    var_12 = len(var_11)
    assert var_12 == 1

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_path'
    var_1 = [var_0]
    var_2 = False
    var_3 = []
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = module_0.Config()
    var_7 = []
    var_8 = []
    var_9 = module_1.find(var_1, var_6, var_7, var_8)
    var_10 = list(var_9)
    var_11 = len(var_10)
    assert var_11 == 0

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir/file1.py'
    var_1 = [var_0]
    var_2 = False
    var_3 = []
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = module_0.Config()
    var_7 = []
    var_8 = []
    var_9 = module_1.find(var_1, var_6, var_7, var_8)
    var_10 = list(var_9)
    var_11 = len(var_10)
    assert var_11 == 1

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = False
    var_3 = 'file2.py'
    var_4 = [var_3]
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = module_0.Config()
    var_8 = []
    var_9 = []
    var_10 = module_1.find(var_1, var_7, var_8, var_9)
    var_11 = list(var_10)
    var_12 = len(var_11)
    assert var_12 == 1



# Parsed testcases at query #2
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_path'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)
    var_7 = len(var_4)
    assert var_7 == 1



# Parsed testcases at query #3
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_path'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)
    var_7 = len(var_4)
    assert var_7 == 1
    var_8 = len(var_6)
    assert var_8 == 0



# Parsed testcases at query #4
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_path'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_find_with_single_file. Retrieved 8/11 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 8/14 statements.
# Partially parsed test_find_with_directory. Retrieved 9/14 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 11/20 statements.
# Partially parsed test_find_with_follow_links_disabled. Retrieved 12/21 statements.
# Partially parsed test_find_with_follow_links_enabled. Retrieved 12/21 statements.
# Partially parsed test_find_with_unsupported_filetype. Retrieved 8/11 statements.
# Partially parsed test_find_with_mixed_paths. Retrieved 11/19 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = []
    var_1 = module_0.Config()
    var_2 = []
    var_3 = []
    var_4 = module_1.find(var_0, var_1, var_2, var_3)
    var_5 = list(var_4)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = []
    var_1 = 'nonexistent_path'
    var_2 = [var_1]
    var_3 = module_0.Config()
    var_4 = []
    var_5 = module_1.find(var_2, var_3, var_4, var_0)
    var_6 = list(var_5)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '# test'
    var_1 = 'test_file.py'
    var_2 = [var_1]
    var_3 = module_0.Config()
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_2, var_3, var_4, var_5)
    var_7 = list(var_6)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '# test'
    var_1 = module_0.Config()
    var_2 = 'test_file.py'
    var_3 = []
    var_4 = [var_2]
    var_5 = []
    var_6 = module_1.find(var_4, var_1, var_3, var_5)
    var_7 = list(var_6)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = '# test'
    var_2 = [var_1]
    var_3 = module_0.Config()
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_2, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = 'test_dir/test_file.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir/skipped_dir'
    var_1 = '# test'
    var_2 = module_0.Config()
    var_3 = 'skipped_dir'
    var_4 = []
    var_5 = 'test_dir'
    var_6 = [var_5]
    var_7 = []
    var_8 = module_1.find(var_6, var_2, var_4, var_7)
    var_9 = list(var_8)
    var_10 = 'test_dir/skipped_dir/test_file.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_dir/subdir'
    var_2 = '# test'
    var_3 = 'test_dir/link'
    var_4 = False
    var_5 = module_0.Config()
    var_6 = [var_2]
    var_7 = []
    var_8 = []
    var_9 = module_1.find(var_6, var_5, var_7, var_8)
    var_10 = list(var_9)
    var_11 = 'test_dir/subdir/test_file.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_dir/subdir'
    var_2 = '# test'
    var_3 = 'test_dir/link'
    var_4 = True
    var_5 = module_0.Config()
    var_6 = [var_2]
    var_7 = []
    var_8 = []
    var_9 = module_1.find(var_6, var_5, var_7, var_8)
    var_10 = list(var_9)
    var_11 = 'test_dir/subdir/test_file.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '# test'
    var_1 = 'test_file.txt'
    var_2 = [var_1]
    var_3 = module_0.Config()
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_2, var_3, var_4, var_5)
    var_7 = list(var_6)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = '# test'
    var_2 = '# test'
    var_3 = 'test_file.py'
    var_4 = [var_2, var_3]
    var_5 = module_0.Config()
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_4, var_5, var_6, var_7)
    var_9 = list(var_8)
    var_10 = 'test_dir/test_file.py'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_find_with_single_file. Retrieved 8/11 statements.
# Partially parsed test_find_with_non_existent_path. Retrieved 5/8 statements.
# Partially parsed test_find_with_directory_containing_python_files. Retrieved 12/20 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 11/17 statements.
# Partially parsed test_find_with_broken_symlink. Retrieved 10/14 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = []
    var_1 = module_0.Config()
    var_2 = []
    var_3 = []
    var_4 = module_1.find(var_0, var_1, var_2, var_3)
    var_5 = list(var_4)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '# test'
    var_1 = 'test_file.py'
    var_2 = [var_1]
    var_3 = module_0.Config()
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_2, var_3, var_4, var_5)
    var_7 = list(var_6)

import isort.settings as module_0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'nonexistent_path.py'
    var_3 = [var_2]
    var_4 = module_0.Config()

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = True
    var_2 = '# test1'
    var_3 = '# test2'
    var_4 = [var_3]
    var_5 = module_0.Config()
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_4, var_5, var_6, var_7)
    var_9 = list(var_8)
    var_10 = 'test_dir/file1.py'
    var_11 = 'test_dir/file2.txt'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = True
    var_2 = '# skipped'
    var_3 = module_0.Config()
    var_4 = 'skipped.py'
    var_5 = []
    var_6 = [var_2]
    var_7 = []
    var_8 = module_1.find(var_6, var_3, var_5, var_7)
    var_9 = list(var_8)
    var_10 = 'test_dir/skipped.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = True
    var_2 = 'nonexistent.py'
    var_3 = 'test_dir/link.py'
    var_4 = []
    var_5 = [var_0]
    var_6 = module_0.Config()
    var_7 = []
    var_8 = module_1.find(var_5, var_6, var_7, var_4)
    var_9 = list(var_8)



# Parsed testcases at query #7
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_path'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)
    var_7 = len(var_4)
    assert var_7 == 1
    var_8 = len(var_6)
    assert var_8 == 0
    var_9 = len(var_3)
    assert var_9 == 0



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_isdir_predicate_evaluates_to_true. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 'test_directory'
    var_1 = True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_isdir_predicate_evaluates_to_true. Retrieved 2/4 statements.


def test_case_0():
    var_0 = '/path/to/directory'
    var_1 = True



# Parsed testcases at query #10
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_file.py'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_isdir_predicate_evaluates_to_true. Retrieved 5/12 statements.


def test_case_0():
    var_0 = '/existing_directory'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_os_path_isdirectory_returns_true. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'valid_directory_path'



# Parsed testcases at query #13
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_file.py'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_isdir_predicate_evaluates_to_true. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'test_directory'



# Parsed testcases at query #15
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_path'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_isdir_predicate_evaluates_to_true. Retrieved 2/4 statements.


def test_case_0():
    var_0 = '/existing_directory'
    var_1 = True



# Parsed testcases at query #17
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_file.py'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)
    var_7 = len(var_4)
    assert var_7 == 1



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_predicate_at_line_9_evaluates_to_true. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 'test_directory'
    var_1 = True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_predicate_at_line_9_evaluates_to_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = '/path/to/nonexistent/directory'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 8/12 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = 'test_dir/subdir'
    var_6 = True
    var_7 = module_1.find(var_1, var_2, var_3, var_4)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_find_with_skipped_file. Retrieved 7/10 statements.
# Partially parsed test_find_with_directory. Retrieved 8/10 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 8/9 statements.
# Partially parsed test_find_with_supported_and_unsupported_files. Retrieved 8/10 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = []
    var_1 = module_0.Config()
    var_2 = []
    var_3 = []
    var_4 = module_1.find(var_0, var_1, var_2, var_3)
    var_5 = list(var_4)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_path'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'existing_file.py'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'skipped_file.py'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)
    var_7 = len(var_6)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'skipped_dir'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)
    var_7 = len(var_3)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'broken_link'
    var_1 = [var_0]
    var_2 = True
    var_3 = module_0.Config()
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'mixed_dir'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)
    var_7 = '.py'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_predicate_at_line_9_evaluates_to_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'not_a_directory.txt'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_predicate_at_line_9_evaluates_to_true. Retrieved 17/21 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '/valid_directory'
    var_1 = [var_0]
    var_2 = False
    var_3 = module_0.Config()
    var_4 = []
    var_5 = []
    var_6 = 'subdir'
    var_7 = [var_6]
    var_8 = 'file.py'
    var_9 = [var_8]
    var_10 = (var_0, var_7, var_9)
    var_11 = [var_10]
    var_12 = iter(var_11)
    var_13 = True
    var_14 = module_1.find(var_1, var_3, var_4, var_5)
    var_15 = list(var_14)
    var_16 = len(var_15)
    assert var_16 == 1



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_predicate_at_line_9_evaluates_to_false. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'some_path'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 5/8 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = '/valid/directory'
    var_1 = True
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_os_path_isdir_returns_true. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'test_directory'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_predicate_at_line_9_evaluates_to_true. Retrieved 7/10 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = True
    var_6 = module_1.find(var_1, var_2, var_3, var_4)



# Parsed testcases at query #28
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '/valid/path'
    var_1 = [var_0]
    var_2 = True
    var_3 = module_0.Config()
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)



# Parsed testcases at query #29
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_path'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)
    var_7 = len(var_6)
    assert var_7 == 0
    var_8 = len(var_3)
    assert var_8 == 0
    var_9 = len(var_4)
    assert var_9 == 1



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_predicate_at_line_9_evaluates_to_false. Retrieved 9/10 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_path'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)
    var_7 = 0
    var_8 = var_1[var_7]



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_predicate_at_line_9_evaluates_to_true. Retrieved 8/9 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'valid_directory'
    var_1 = [var_0]
    var_2 = False
    var_3 = module_0.Config()
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_predicate_at_line_9_evaluates_to_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'non_existent_directory'



# Parsed testcases at query #33
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'non_existent_dir'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)
    var_7 = len(var_6)
    assert var_7 == 0
    var_8 = len(var_3)
    assert var_8 == 0
    var_9 = len(var_4)
    assert var_9 == 1



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_find_with_directory_and_valid_files. Retrieved 12/16 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 14/15 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 14/15 statements.
# Partially parsed test_find_with_mixed_paths. Retrieved 17/18 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'tests'
    var_1 = [var_0]
    var_2 = False
    var_3 = module_0.Config()
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = len(var_7)
    var_9 = '.py'
    var_10 = len(var_4)
    assert var_10 == 0
    var_11 = len(var_5)
    assert var_11 == 0

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_path'
    var_1 = [var_0]
    var_2 = False
    var_3 = module_0.Config()
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

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'tests/skipped_dir'
    var_1 = [var_0]
    var_2 = False
    var_3 = 'skipped_dir'
    var_4 = [var_3]
    var_5 = module_0.Config()
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_1, var_5, var_6, var_7)
    var_9 = list(var_8)
    var_10 = len(var_9)
    assert var_10 == 0
    var_11 = len(var_6)
    assert var_11 == 1
    var_12 = var_6[var_2]
    var_13 = len(var_7)
    assert var_13 == 0

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'tests/broken_symlink'
    var_1 = [var_0]
    var_2 = True
    var_3 = module_0.Config()
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

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'tests/circular_symlink'
    var_1 = [var_0]
    var_2 = True
    var_3 = module_0.Config()
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 0
    var_9 = len(var_4)
    assert var_9 == 0
    var_10 = len(var_5)
    assert var_10 == 0

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'tests/unsupported.txt'
    var_1 = [var_0]
    var_2 = False
    var_3 = module_0.Config()
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 0
    var_9 = len(var_4)
    assert var_9 == 0
    var_10 = len(var_5)
    assert var_10 == 0

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'tests/skipped_file.py'
    var_1 = [var_0]
    var_2 = False
    var_3 = 'skipped_file.py'
    var_4 = [var_3]
    var_5 = module_0.Config()
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_1, var_5, var_6, var_7)
    var_9 = list(var_8)
    var_10 = len(var_9)
    assert var_10 == 0
    var_11 = len(var_6)
    assert var_11 == 1
    var_12 = var_6[var_2]
    var_13 = len(var_7)
    assert var_13 == 0

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'tests'
    var_1 = 'nonexistent_path'
    var_2 = 'tests/skipped_dir'
    var_3 = 'tests/valid_file.py'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = False
    var_6 = 'skipped_dir'
    var_7 = [var_6]
    var_8 = module_0.Config()
    var_9 = []
    var_10 = []
    var_11 = module_1.find(var_4, var_8, var_9, var_10)
    var_12 = list(var_11)
    var_13 = len(var_12)
    var_14 = len(var_9)
    assert var_14 == 1
    var_15 = var_9[var_5]
    var_16 = len(var_10)
    assert var_16 == 1



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_find_with_single_file. Retrieved 8/11 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 9/15 statements.
# Partially parsed test_find_with_directory. Retrieved 12/20 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 10/20 statements.
# Partially parsed test_find_with_nested_directories. Retrieved 13/22 statements.
# Partially parsed test_find_with_follow_links_disabled. Retrieved 11/18 statements.
# Partially parsed test_find_with_mixed_files_and_directories. Retrieved 12/20 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = []
    var_1 = module_0.Config()
    var_2 = []
    var_3 = []
    var_4 = module_1.find(var_0, var_1, var_2, var_3)
    var_5 = list(var_4)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = []
    var_1 = 'nonexistent_path.py'
    var_2 = [var_1]
    var_3 = module_0.Config()
    var_4 = []
    var_5 = module_1.find(var_2, var_3, var_4, var_0)
    var_6 = list(var_5)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '# test'
    var_1 = 'test_file.py'
    var_2 = [var_1]
    var_3 = module_0.Config()
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_2, var_3, var_4, var_5)
    var_7 = list(var_6)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'skip_*'
    var_2 = []
    var_3 = '# test'
    var_4 = 'skip_test.py'
    var_5 = [var_4]
    var_6 = []
    var_7 = module_1.find(var_5, var_0, var_2, var_6)
    var_8 = list(var_7)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = '# test1'
    var_2 = '# test2'
    var_3 = [var_2]
    var_4 = module_0.Config()
    var_5 = []
    var_6 = []
    var_7 = module_1.find(var_3, var_4, var_5, var_6)
    var_8 = list(var_7)
    var_9 = set(var_8)
    var_10 = 'test_dir/file1.py'
    var_11 = 'test_dir/file2.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'skip_*'
    var_2 = []
    var_3 = 'skip_dir'
    var_4 = '# test'
    var_5 = [var_3]
    var_6 = []
    var_7 = module_1.find(var_5, var_0, var_2, var_6)
    var_8 = list(var_7)
    var_9 = 'skip_dir/file.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'parent/child'
    var_1 = '# test1'
    var_2 = '# test2'
    var_3 = 'parent'
    var_4 = [var_3]
    var_5 = module_0.Config()
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_4, var_5, var_6, var_7)
    var_9 = list(var_8)
    var_10 = set(var_9)
    var_11 = 'parent/file1.py'
    var_12 = 'parent/child/file2.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = False
    var_1 = module_0.Config()
    var_2 = 'real_dir'
    var_3 = '# test'
    var_4 = 'symlink_dir'
    var_5 = [var_4]
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_5, var_1, var_6, var_7)
    var_9 = list(var_8)
    var_10 = 'real_dir/file.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '# test'
    var_1 = 'dir'
    var_2 = '# test'
    var_3 = 'single_file.py'
    var_4 = [var_3, var_2]
    var_5 = module_0.Config()
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_4, var_5, var_6, var_7)
    var_9 = list(var_8)
    var_10 = set(var_9)
    var_11 = 'dir/file.py'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_isdir_predicate. Retrieved 1/2 statements.


def test_case_0():
    var_0 = '/path/to/existing/directory'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_find_with_single_file. Retrieved 8/11 statements.
# Partially parsed test_find_with_directory_containing_python_files. Retrieved 11/19 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 9/13 statements.
# Partially parsed test_find_with_broken_symlink. Retrieved 9/14 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = []
    var_1 = module_0.Config()
    var_2 = []
    var_3 = []
    var_4 = module_1.find(var_0, var_1, var_2, var_3)
    var_5 = list(var_4)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = []
    var_1 = 'nonexistent_path'
    var_2 = [var_1]
    var_3 = module_0.Config()
    var_4 = []
    var_5 = module_1.find(var_2, var_3, var_4, var_0)
    var_6 = list(var_5)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '# test'
    var_1 = 'test_file.py'
    var_2 = [var_1]
    var_3 = module_0.Config()
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_2, var_3, var_4, var_5)
    var_7 = list(var_6)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = '# test1'
    var_2 = '# test2'
    var_3 = [var_2]
    var_4 = module_0.Config()
    var_5 = []
    var_6 = []
    var_7 = module_1.find(var_3, var_4, var_5, var_6)
    var_8 = list(var_7)
    var_9 = 'test_dir/file1.py'
    var_10 = 'test_dir/file2.txt'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'skip_*'
    var_2 = []
    var_3 = '# skip'
    var_4 = 'test_skip_me.py'
    var_5 = [var_4]
    var_6 = []
    var_7 = module_1.find(var_5, var_0, var_2, var_6)
    var_8 = list(var_7)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = []
    var_1 = '# real'
    var_2 = 'real_file.py'
    var_3 = 'broken_link.py'
    var_4 = [var_3]
    var_5 = module_0.Config()
    var_6 = []
    var_7 = module_1.find(var_4, var_5, var_6, var_0)
    var_8 = list(var_7)



# Parsed testcases at query #5
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_path'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)
    var_7 = len(var_4)
    assert var_7 == 1



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_os_path_isdir_returns_true. Retrieved 2/4 statements.


def test_case_0():
    var_0 = '/some/existing/directory'
    var_1 = True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_isdir_predicate_evaluates_to_true. Retrieved 1/6 statements.


def test_case_0():
    var_0 = '/test/directory'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_find_with_single_file. Retrieved 8/11 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 8/14 statements.
# Partially parsed test_find_with_directory. Retrieved 12/20 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 10/21 statements.
# Partially parsed test_find_with_nested_directories. Retrieved 13/22 statements.
# Partially parsed test_find_with_symlink. Retrieved 11/18 statements.
# Partially parsed test_find_with_unsupported_filetype. Retrieved 8/11 statements.
# Partially parsed test_find_with_mixed_paths. Retrieved 12/20 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = []
    var_1 = module_0.Config()
    var_2 = []
    var_3 = []
    var_4 = module_1.find(var_0, var_1, var_2, var_3)
    var_5 = list(var_4)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = []
    var_1 = 'nonexistent_path.py'
    var_2 = [var_1]
    var_3 = module_0.Config()
    var_4 = []
    var_5 = module_1.find(var_2, var_3, var_4, var_0)
    var_6 = list(var_5)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '# test'
    var_1 = 'test_file.py'
    var_2 = [var_1]
    var_3 = module_0.Config()
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_2, var_3, var_4, var_5)
    var_7 = list(var_6)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '# skip'
    var_1 = module_0.Config()
    var_2 = 'skip_me.py'
    var_3 = []
    var_4 = [var_2]
    var_5 = []
    var_6 = module_1.find(var_4, var_1, var_3, var_5)
    var_7 = list(var_6)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = '# test1'
    var_2 = '# test2'
    var_3 = [var_2]
    var_4 = module_0.Config()
    var_5 = []
    var_6 = []
    var_7 = module_1.find(var_3, var_4, var_5, var_6)
    var_8 = list(var_7)
    var_9 = set(var_8)
    var_10 = 'test_dir/file1.py'
    var_11 = 'test_dir/file2.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'skip_dir/subdir'
    var_1 = '# skip'
    var_2 = module_0.Config()
    var_3 = 'skip_dir'
    var_4 = []
    var_5 = [var_3]
    var_6 = []
    var_7 = module_1.find(var_5, var_2, var_4, var_6)
    var_8 = list(var_7)
    var_9 = 'skip_dir/file.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'parent/child'
    var_1 = '# parent'
    var_2 = '# child'
    var_3 = 'parent'
    var_4 = [var_3]
    var_5 = module_0.Config()
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_4, var_5, var_6, var_7)
    var_9 = list(var_8)
    var_10 = set(var_9)
    var_11 = 'parent/file.py'
    var_12 = 'parent/child/file.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'real_dir'
    var_1 = '# real'
    var_2 = 'link_dir'
    var_3 = True
    var_4 = module_0.Config()
    var_5 = [var_2]
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_5, var_4, var_6, var_7)
    var_9 = list(var_8)
    var_10 = 'real_dir/file.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '# not python'
    var_1 = 'test.txt'
    var_2 = [var_1]
    var_3 = module_0.Config()
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_2, var_3, var_4, var_5)
    var_7 = list(var_6)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '# file'
    var_1 = 'dir'
    var_2 = '# dir file'
    var_3 = 'file.py'
    var_4 = [var_3, var_2]
    var_5 = module_0.Config()
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_4, var_5, var_6, var_7)
    var_9 = list(var_8)
    var_10 = set(var_9)
    var_11 = 'dir/file.py'



# Parsed testcases at query #9
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_path'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)
    var_7 = len(var_4)
    assert var_7 == 1
    var_8 = len(var_6)
    assert var_8 == 0



# Parsed testcases at query #10
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
    var_6 = module_0.Config()
    var_7 = []
    var_8 = []
    var_9 = module_1.find(var_1, var_6, var_7, var_8)
    var_10 = list(var_9)
    var_11 = len(var_10)
    assert var_11 == 2

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
    var_7 = module_0.Config()
    var_8 = []
    var_9 = []
    var_10 = module_1.find(var_1, var_7, var_8, var_9)
    var_11 = list(var_10)
    var_12 = len(var_11)
    assert var_12 == 1
    var_13 = len(var_8)
    assert var_13 == 1

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_path'
    var_1 = [var_0]
    var_2 = False
    var_3 = []
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = module_0.Config()
    var_7 = []
    var_8 = []
    var_9 = module_1.find(var_1, var_6, var_7, var_8)
    var_10 = list(var_9)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'single_file.py'
    var_1 = [var_0]
    var_2 = False
    var_3 = []
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = module_0.Config()
    var_7 = []
    var_8 = []
    var_9 = module_1.find(var_1, var_6, var_7, var_8)
    var_10 = list(var_9)
    var_11 = len(var_10)
    assert var_11 == 1

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'dir_with_unsupported'
    var_1 = [var_0]
    var_2 = False
    var_3 = []
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = module_0.Config()
    var_7 = []
    var_8 = []
    var_9 = module_1.find(var_1, var_6, var_7, var_8)
    var_10 = list(var_9)
    var_11 = len(var_10)
    assert var_11 == 1



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_isdir_predicate_evaluates_to_true. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 'test_directory'
    var_1 = True



# Parsed testcases at query #12
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_path'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)
    var_7 = len(var_4)
    assert var_7 == 1



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_isdir_predicate_evaluates_to_true. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 'test_directory'
    var_1 = True



# Parsed testcases at query #14
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_file.py'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)
    var_7 = len(var_4)
    assert var_7 == 1
    var_8 = len(var_6)
    assert var_8 == 0



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_isdir_predicate_evaluates_to_true. Retrieved 1/10 statements.


def test_case_0():
    var_0 = '/test/directory'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_isdir_predicate_evaluates_to_true. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'test_directory'
    var_1 = True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_find_with_directory. Retrieved 7/9 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 8/10 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 9/11 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = []
    var_1 = module_0.Config()
    var_2 = []
    var_3 = []
    var_4 = module_1.find(var_0, var_1, var_2, var_3)
    var_5 = list(var_4)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent.py'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'tests'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = [var_0]
    var_2 = [var_0]
    var_3 = module_0.Config()
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'tests'
    var_1 = [var_0]
    var_2 = 'tests/skip'
    var_3 = [var_2]
    var_4 = module_0.Config()
    var_5 = []
    var_6 = []
    var_7 = module_1.find(var_1, var_4, var_5, var_6)
    var_8 = list(var_7)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'broken_link'
    var_1 = [var_0]
    var_2 = True
    var_3 = module_0.Config()
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)



# Parsed testcases at query #2
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_path'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)
    var_7 = len(var_4)
    assert var_7 == 1
    var_8 = len(var_6)
    assert var_8 == 0



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_isdir_predicate_evaluates_to_true. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'test_directory'



# Parsed testcases at query #4
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_path'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)
    var_7 = len(var_4)
    assert var_7 == 1



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_isdir_predicate. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 'test_directory'
    var_1 = True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_isdir_predicate_evaluates_to_true. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'temp_test_dir'
    var_1 = True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_find_with_single_file. Retrieved 4/13 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 4/18 statements.
# Partially parsed test_find_with_directory_containing_python_files. Retrieved 7/17 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 6/18 statements.
# Partially parsed test_find_with_symlink_loop. Retrieved 8/19 statements.
# Partially parsed test_find_with_non_python_files. Retrieved 5/12 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = []
    var_1 = module_0.Config()
    var_2 = []
    var_3 = []
    var_4 = module_1.find(var_0, var_1, var_2, var_3)
    var_5 = list(var_4)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = []
    var_1 = 'nonexistent_path'
    var_2 = [var_1]
    var_3 = module_0.Config()
    var_4 = []
    var_5 = module_1.find(var_2, var_3, var_4, var_0)
    var_6 = list(var_5)

import isort.settings as module_0

def test_case_0():
    var_0 = b"print('hello')"
    var_1 = module_0.Config()
    var_2 = []
    var_3 = []

import isort.settings as module_0

def test_case_0():
    var_0 = b"print('hello')"
    var_1 = module_0.Config()
    var_2 = []
    var_3 = []

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = 'test.txt'
    var_3 = 'not python'
    var_4 = module_0.Config()
    var_5 = []
    var_6 = []

import isort.settings as module_0

def test_case_0():
    var_0 = 'subdir'
    var_1 = 'test.py'
    var_2 = "print('hello')"
    var_3 = module_0.Config()
    var_4 = []
    var_5 = []

import isort.settings as module_0

def test_case_0():
    var_0 = 'subdir'
    var_1 = 'link'
    var_2 = 'test.py'
    var_3 = "print('hello')"
    var_4 = True
    var_5 = module_0.Config()
    var_6 = []
    var_7 = []

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'not python'
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []



# Parsed testcases at query #8
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = []
    var_1 = 'nonexistent_path'
    var_2 = [var_1]
    var_3 = module_0.Config()
    var_4 = []
    var_5 = module_1.find(var_2, var_3, var_4, var_0)
    var_6 = len(var_0)
    assert var_6 == 1



# Parsed testcases at query #9
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_path'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)
    var_7 = len(var_4)
    assert var_7 == 1



# Parsed testcases at query #10
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = False
    var_3 = module_0.Config()
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 2
    var_9 = len(var_4)
    assert var_9 == 0
    var_10 = len(var_5)
    assert var_10 == 0

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = False
    var_3 = 'subdir'
    var_4 = [var_3]
    var_5 = module_0.Config()
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_1, var_5, var_6, var_7)
    var_9 = list(var_8)
    var_10 = len(var_9)
    assert var_10 == 1
    var_11 = len(var_7)
    assert var_11 == 0

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'non_existent_path'
    var_1 = [var_0]
    var_2 = False
    var_3 = module_0.Config()
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

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir/file1.py'
    var_1 = [var_0]
    var_2 = False
    var_3 = module_0.Config()
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = len(var_4)
    assert var_9 == 0
    var_10 = len(var_5)
    assert var_10 == 0

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = False
    var_3 = 'file1.py'
    var_4 = [var_3]
    var_5 = module_0.Config()
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_1, var_5, var_6, var_7)
    var_9 = list(var_8)
    var_10 = len(var_9)
    assert var_10 == 1
    var_11 = len(var_7)
    assert var_11 == 0



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_isdir_predicate_evaluates_to_true. Retrieved 7/25 statements.


def test_case_0():
    var_0 = '/test_dir'
    var_1 = True
    var_2 = []
    var_3 = []
    var_4 = (var_0, var_2, var_3)
    var_5 = []
    var_6 = []



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_find_with_single_file. Retrieved 3/10 statements.
# Partially parsed test_find_with_single_directory. Retrieved 5/12 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 5/15 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 6/18 statements.
# Partially parsed test_find_with_non_python_file. Retrieved 5/12 statements.
# Partially parsed test_find_with_mixed_files. Retrieved 7/17 statements.
# Partially parsed test_find_with_symlink. Retrieved 8/21 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = []
    var_1 = module_0.Config()
    var_2 = []
    var_3 = []
    var_4 = module_1.find(var_0, var_1, var_2, var_3)
    var_5 = list(var_4)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = []
    var_1 = 'nonexistent_path'
    var_2 = [var_1]
    var_3 = module_0.Config()
    var_4 = []
    var_5 = module_1.find(var_2, var_3, var_4, var_0)
    var_6 = list(var_5)

import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = '# test'
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = '# test'
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []

import isort.settings as module_0

def test_case_0():
    var_0 = 'skipme'
    var_1 = 'test.py'
    var_2 = '# test'
    var_3 = module_0.Config()
    var_4 = []
    var_5 = []

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.txt'
    var_1 = '# test'
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []

import isort.settings as module_0

def test_case_0():
    var_0 = 'test.py'
    var_1 = 'test.txt'
    var_2 = '# test'
    var_3 = '# test'
    var_4 = module_0.Config()
    var_5 = []
    var_6 = []

import isort.settings as module_0

def test_case_0():
    var_0 = 'subdir'
    var_1 = 'test.py'
    var_2 = '# test'
    var_3 = 'link'
    var_4 = True
    var_5 = module_0.Config()
    var_6 = []
    var_7 = []



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
    var_6 = module_0.Config()
    var_7 = []
    var_8 = []
    var_9 = module_1.find(var_1, var_6, var_7, var_8)
    var_10 = list(var_9)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = False
    var_3 = 'test_dir/subdir'
    var_4 = [var_3]
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = module_0.Config()
    var_8 = []
    var_9 = []
    var_10 = module_1.find(var_1, var_7, var_8, var_9)
    var_11 = list(var_10)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_dir'
    var_1 = [var_0]
    var_2 = False
    var_3 = []
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = module_0.Config()
    var_7 = []
    var_8 = []
    var_9 = module_1.find(var_1, var_6, var_7, var_8)
    var_10 = list(var_9)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir/file1.py'
    var_1 = [var_0]
    var_2 = False
    var_3 = []
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = module_0.Config()
    var_7 = []
    var_8 = []
    var_9 = module_1.find(var_1, var_6, var_7, var_8)
    var_10 = list(var_9)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = False
    var_3 = []
    var_4 = '.txt'
    var_5 = [var_4]
    var_6 = module_0.Config()
    var_7 = []
    var_8 = []
    var_9 = module_1.find(var_1, var_6, var_7, var_8)
    var_10 = list(var_9)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_isdir_predicate_evaluates_to_true. Retrieved 11/21 statements.


def test_case_0():
    var_0 = 'test_dir'
    var_1 = True
    var_2 = 'test_dir'
    var_3 = []
    var_4 = []
    var_5 = (var_2, var_3, var_4)
    var_6 = [var_5]
    var_7 = iter(var_6)
    var_8 = []
    var_9 = []
    var_10 = [var_0]



# Parsed testcases at query #15
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_path'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)
    var_7 = len(var_4)
    assert var_7 == 1



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_isdir_evaluates_to_true. Retrieved 2/4 statements.


def test_case_0():
    var_0 = '/valid/directory'
    var_1 = True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_find_with_skipped_file. Retrieved 8/11 statements.
# Partially parsed test_find_with_directory. Retrieved 10/16 statements.
# Partially parsed test_find_with_nested_skipped_directory. Retrieved 12/19 statements.
# Partially parsed test_find_with_symlink_and_follow_links_disabled. Retrieved 15/25 statements.
# Partially parsed test_find_with_broken_symlink. Retrieved 11/16 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = []
    var_1 = module_0.Config()
    var_2 = []
    var_3 = []
    var_4 = module_1.find(var_0, var_1, var_2, var_3)
    var_5 = list(var_4)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = []
    var_1 = 'nonexistent_path.py'
    var_2 = [var_1]
    var_3 = module_0.Config()
    var_4 = []
    var_5 = module_1.find(var_2, var_3, var_4, var_0)
    var_6 = list(var_5)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'skip_*'
    var_2 = []
    var_3 = 'skip_me.py'
    var_4 = [var_3]
    var_5 = []
    var_6 = module_1.find(var_4, var_0, var_2, var_5)
    var_7 = list(var_6)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = True
    var_2 = ''
    var_3 = ''
    var_4 = module_0.Config()
    var_5 = [var_3]
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_5, var_4, var_6, var_7)
    var_9 = list(var_8)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir/skip_dir'
    var_1 = True
    var_2 = ''
    var_3 = module_0.Config()
    var_4 = '*skip_dir'
    var_5 = []
    var_6 = 'test_dir'
    var_7 = [var_6]
    var_8 = []
    var_9 = module_1.find(var_7, var_3, var_5, var_8)
    var_10 = list(var_9)
    var_11 = 'skip_dir'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = True
    var_2 = 'test_dir/subdir'
    var_3 = ''
    var_4 = 'test_dir/link'
    var_5 = 'subdir'
    var_6 = False
    var_7 = module_0.Config()
    var_8 = [var_3]
    var_9 = []
    var_10 = []
    var_11 = module_1.find(var_8, var_7, var_9, var_10)
    var_12 = list(var_11)
    var_13 = 'subdir/file.py'
    var_14 = 'link/file.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = True
    var_2 = 'test_dir/broken_link'
    var_3 = 'nonexistent'
    var_4 = []
    var_5 = [var_0]
    var_6 = module_0.Config()
    var_7 = []
    var_8 = module_1.find(var_5, var_6, var_7, var_4)
    var_9 = list(var_8)
    var_10 = 'test_dir/broken_link'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_predicate_at_line_9_evaluates_to_false. Retrieved 8/12 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'file.txt'
    var_1 = False
    var_2 = [var_0]
    var_3 = module_0.Config()
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_2, var_3, var_4, var_5)
    var_7 = list(var_6)



# Parsed testcases at query #19
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_path'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)
    var_7 = len(var_6)
    assert var_7 == 0
    var_8 = len(var_3)
    assert var_8 == 0
    var_9 = len(var_4)
    assert var_9 == 1



# Parsed testcases at query #20
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = []
    var_1 = module_0.Config()
    var_2 = []
    var_3 = []
    var_4 = module_1.find(var_0, var_1, var_2, var_3)
    var_5 = list(var_4)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '/nonexistent/path'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test.py'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'src'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'src'
    var_1 = [var_0]
    var_2 = 'src/skipped.py'
    var_3 = [var_2]
    var_4 = module_0.Config()
    var_5 = []
    var_6 = []
    var_7 = module_1.find(var_1, var_4, var_5, var_6)
    var_8 = list(var_7)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'broken_link'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)



# Parsed testcases at query #21
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = []
    var_1 = module_0.Config()
    var_2 = []
    var_3 = []
    var_4 = module_1.find(var_0, var_1, var_2, var_3)
    var_5 = list(var_4)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '/nonexistent/path'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'broken_link'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_predicate_at_line_9_evaluates_to_false. Retrieved 7/8 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'file.txt'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = 0
    var_6 = var_1[var_5]



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 11/15 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'test_dir'
    var_2 = [var_1]
    var_3 = []
    var_4 = []
    var_5 = 'test_dir/subdir'
    var_6 = True
    var_7 = '# test'
    var_8 = module_1.find(var_2, var_0, var_3, var_4)
    var_9 = list(var_8)
    var_10 = len(var_9)
    assert var_10 == 1



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'some_directory'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_os_path_isdir_evaluates_to_true. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 'test_directory'
    var_1 = True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_predicate_at_line_9_evaluates_to_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'non_existent_directory'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_os_path_isdir_evaluates_to_true. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'test_directory'



# Parsed testcases at query #28
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'nonexistent_directory'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)
    var_7 = len(var_6)
    assert var_7 == 0
    var_8 = len(var_3)
    assert var_8 == 0
    var_9 = len(var_4)
    assert var_9 == 1



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_predicate_at_line_9_evaluates_to_true. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'test_dir'
    var_1 = True
    var_2 = []
    var_3 = []



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_predicate_at_line_9_evaluates_to_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 'non_directory_path'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_predicate_evaluates_to_true. Retrieved 5/7 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = '/valid/directory'
    var_1 = module_0.Config()
    var_2 = []
    var_3 = []
    var_4 = True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_predicate_at_line_9_evaluates_to_false. Retrieved 5/6 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = 'nonexistent_path'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []



