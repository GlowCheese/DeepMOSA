####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_find_with_directory. Retrieved 10/12 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 12/14 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'tests/test_data'
    var_4 = [var_3]
    var_5 = module_1.find(var_4, var_0, var_1, var_2)
    var_6 = list(var_5)
    var_7 = len(var_6)
    var_8 = len(var_1)
    assert var_8 == 0
    var_9 = len(var_2)
    assert var_9 == 0

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'tests/test_data/skip_this'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = 'tests/test_data'
    var_6 = [var_5]
    var_7 = module_1.find(var_6, var_2, var_3, var_4)
    var_8 = list(var_7)
    var_9 = len(var_8)
    var_10 = len(var_3)
    assert var_10 == 1
    var_11 = len(var_4)
    assert var_11 == 0

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'tests/non_existent_path'
    var_4 = [var_3]
    var_5 = module_1.find(var_4, var_0, var_1, var_2)
    var_6 = list(var_5)
    var_7 = len(var_6)
    assert var_7 == 0
    var_8 = len(var_1)
    assert var_8 == 0
    var_9 = len(var_2)
    assert var_9 == 1

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'tests/test_data/test_file.py'
    var_4 = [var_3]
    var_5 = module_1.find(var_4, var_0, var_1, var_2)
    var_6 = list(var_5)
    var_7 = len(var_6)
    assert var_7 == 1
    var_8 = len(var_1)
    assert var_8 == 0
    var_9 = len(var_2)
    assert var_9 == 0

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '.txt'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = 'tests/test_data'
    var_6 = [var_5]
    var_7 = module_1.find(var_6, var_2, var_3, var_4)
    var_8 = list(var_7)
    var_9 = len(var_8)
    assert var_9 == 0
    var_10 = len(var_3)
    assert var_10 == 0
    var_11 = len(var_4)
    assert var_11 == 0



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_find_is_dir_predicate. Retrieved 8/9 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '/some/directory'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = True
    var_6 = module_1.find(var_1, var_2, var_3, var_4)
    var_7 = list(var_6)



# Parsed testcases at query #3
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = False
    var_1 = '.py'
    var_2 = [var_1]
    var_3 = []
    var_4 = module_0.Config()
    var_5 = []
    var_6 = []
    var_7 = 'test_dir'
    var_8 = [var_7]
    var_9 = module_1.find(var_8, var_4, var_5, var_6)
    var_10 = list(var_9)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = False
    var_1 = '.py'
    var_2 = [var_1]
    var_3 = 'test_dir/skip_me'
    var_4 = [var_3]
    var_5 = module_0.Config()
    var_6 = []
    var_7 = []
    var_8 = 'test_dir'
    var_9 = [var_8]
    var_10 = module_1.find(var_9, var_5, var_6, var_7)
    var_11 = list(var_10)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = False
    var_1 = '.py'
    var_2 = [var_1]
    var_3 = []
    var_4 = module_0.Config()
    var_5 = []
    var_6 = []
    var_7 = 'nonexistent_path'
    var_8 = [var_7]
    var_9 = module_1.find(var_8, var_4, var_5, var_6)
    var_10 = list(var_9)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = False
    var_1 = '.py'
    var_2 = [var_1]
    var_3 = []
    var_4 = module_0.Config()
    var_5 = []
    var_6 = []
    var_7 = 'test_file.py'
    var_8 = [var_7]
    var_9 = module_1.find(var_8, var_4, var_5, var_6)
    var_10 = list(var_9)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = False
    var_1 = '.py'
    var_2 = [var_1]
    var_3 = 'test_file.py'
    var_4 = [var_3]
    var_5 = module_0.Config()
    var_6 = []
    var_7 = []
    var_8 = [var_3]
    var_9 = module_1.find(var_8, var_5, var_6, var_7)
    var_10 = list(var_9)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = False
    var_1 = '.py'
    var_2 = [var_1]
    var_3 = []
    var_4 = module_0.Config()
    var_5 = []
    var_6 = []
    var_7 = 'test_file.txt'
    var_8 = [var_7]
    var_9 = module_1.find(var_8, var_4, var_5, var_6)
    var_10 = list(var_9)



# Parsed testcases at query #4
#--------------------------




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
    var_7 = len(var_4)
    assert var_7 == 1



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_predicate_at_line27_evaluates_to_true. Retrieved 3/12 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'test_file.py'



# Parsed testcases at query #6
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '/non_existent_directory'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)
    var_7 = len(var_6)
    assert var_7 == 0
    var_8 = len(var_4)
    assert var_8 == 1



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_resolved_path_in_visited_dirs. Retrieved 12/16 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = set()
    var_6 = 'test_dir/subdir'
    var_7 = 'subdir'
    var_8 = [var_7]
    var_9 = 'test_dir'
    var_10 = module_1.find(var_1, var_2, var_3, var_4)
    var_11 = list(var_10)



# Parsed testcases at query #8
#--------------------------




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
    var_7 = len(var_4)
    assert var_7 == 1



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_predicate_at_line_9_evaluates_to_False. Retrieved 8/19 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '/nonexistent/directory'
    var_1 = [var_0]
    var_2 = False
    var_3 = module_0.Config(var_2)
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_find_with_directory. Retrieved 4/14 statements.
# Partially parsed test_find_with_nonexistent_path. Retrieved 4/12 statements.
# Partially parsed test_find_with_file. Retrieved 4/12 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 4/12 statements.
# Partially parsed test_find_with_unsupported_filetype. Retrieved 4/12 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'dir'
    var_3 = [var_2]

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'nonexistent'
    var_3 = [var_2]

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'file.py'
    var_3 = [var_2]

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'skipped_file.py'
    var_3 = [var_2]

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'file.txt'
    var_3 = [var_2]



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_find_with_directory. Retrieved 9/22 statements.
# Partially parsed test_find_with_nonexistent_path. Retrieved 8/11 statements.
# Partially parsed test_find_with_file. Retrieved 8/11 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 8/11 statements.
# Partially parsed test_find_with_unsupported_filetype. Retrieved 8/11 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = False
    var_3 = '/test/skip_me'
    var_4 = [var_3]
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = '/test'
    var_8 = [var_7]

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = False
    var_3 = []
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = '/nonexistent'
    var_7 = [var_6]

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = False
    var_3 = []
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = '/test/test_file.py'
    var_7 = [var_6]

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = False
    var_3 = '/test/skip_me.py'
    var_4 = [var_3]
    var_5 = '.py'
    var_6 = [var_5]
    var_7 = [var_3]

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = False
    var_3 = []
    var_4 = '.py'
    var_5 = [var_4]
    var_6 = '/test/test_file.txt'
    var_7 = [var_6]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_find_yields_files_in_directory. Retrieved 14/19 statements.
# Partially parsed test_find_skips_unsupported_files. Retrieved 14/19 statements.
# Partially parsed test_find_skips_skipped_directories. Retrieved 14/19 statements.
# Partially parsed test_find_yields_files_directly. Retrieved 12/15 statements.
# Partially parsed test_find_skips_skipped_files. Retrieved 12/17 statements.
# Partially parsed test_find_follows_symlinks. Retrieved 16/26 statements.
# Partially parsed test_find_does_not_follow_symlinks. Retrieved 17/27 statements.
# Partially parsed test_find_handles_duplicate_paths. Retrieved 14/19 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '.py'
    var_1 = {var_0}
    var_2 = []
    var_3 = False
    var_4 = module_0.Config()
    var_5 = []
    var_6 = []
    var_7 = 'test_dir'
    var_8 = [var_7]
    var_9 = True
    var_10 = ''
    var_11 = module_1.find(var_8, var_4, var_5, var_6)
    var_12 = list(var_11)
    var_13 = 'test_dir/test_file.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '.py'
    var_1 = {var_0}
    var_2 = []
    var_3 = False
    var_4 = module_0.Config()
    var_5 = []
    var_6 = []
    var_7 = 'test_dir'
    var_8 = [var_7]
    var_9 = True
    var_10 = ''
    var_11 = module_1.find(var_8, var_4, var_5, var_6)
    var_12 = list(var_11)
    var_13 = 'test_dir/test_file.txt'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '.py'
    var_1 = {var_0}
    var_2 = 'test_dir'
    var_3 = [var_2]
    var_4 = False
    var_5 = module_0.Config()
    var_6 = []
    var_7 = []
    var_8 = [var_2]
    var_9 = True
    var_10 = ''
    var_11 = module_1.find(var_8, var_5, var_6, var_7)
    var_12 = list(var_11)
    var_13 = 'test_dir/test_file.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '.py'
    var_1 = {var_0}
    var_2 = []
    var_3 = False
    var_4 = module_0.Config()
    var_5 = []
    var_6 = []
    var_7 = 'nonexistent_dir'
    var_8 = [var_7]
    var_9 = module_1.find(var_8, var_4, var_5, var_6)
    var_10 = list(var_9)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '.py'
    var_1 = {var_0}
    var_2 = []
    var_3 = False
    var_4 = module_0.Config()
    var_5 = []
    var_6 = []
    var_7 = 'test_file.py'
    var_8 = [var_7]
    var_9 = ''
    var_10 = module_1.find(var_8, var_4, var_5, var_6)
    var_11 = list(var_10)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '.py'
    var_1 = {var_0}
    var_2 = 'test_file.py'
    var_3 = [var_2]
    var_4 = False
    var_5 = module_0.Config()
    var_6 = []
    var_7 = []
    var_8 = [var_2]
    var_9 = ''
    var_10 = module_1.find(var_8, var_5, var_6, var_7)
    var_11 = list(var_10)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '.py'
    var_1 = {var_0}
    var_2 = []
    var_3 = True
    var_4 = module_0.Config()
    var_5 = []
    var_6 = []
    var_7 = 'test_dir'
    var_8 = [var_7]
    var_9 = 'test_dir/linked_dir'
    var_10 = ''
    var_11 = 'test_dir/symlink'
    var_12 = module_1.find(var_8, var_4, var_5, var_6)
    var_13 = list(var_12)
    var_14 = 'test_dir/symlink/test_file.py'
    var_15 = 'test_dir/linked_dir/test_file.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '.py'
    var_1 = {var_0}
    var_2 = []
    var_3 = False
    var_4 = module_0.Config()
    var_5 = []
    var_6 = []
    var_7 = 'test_dir'
    var_8 = [var_7]
    var_9 = True
    var_10 = 'test_dir/linked_dir'
    var_11 = ''
    var_12 = 'test_dir/symlink'
    var_13 = module_1.find(var_8, var_4, var_5, var_6)
    var_14 = list(var_13)
    var_15 = 'test_dir/symlink/test_file.py'
    var_16 = 'test_dir/linked_dir/test_file.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '.py'
    var_1 = {var_0}
    var_2 = []
    var_3 = False
    var_4 = module_0.Config()
    var_5 = []
    var_6 = []
    var_7 = 'test_dir'
    var_8 = [var_7, var_7]
    var_9 = True
    var_10 = ''
    var_11 = module_1.find(var_8, var_4, var_5, var_6)
    var_12 = list(var_11)
    var_13 = 'test_dir/test_file.py'

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '.py'
    var_1 = {var_0}
    var_2 = []
    var_3 = False
    var_4 = module_0.Config()
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_7, var_4, var_5, var_6)
    var_9 = list(var_8)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_find_with_directory_and_skipped_paths. Retrieved 11/25 statements.
# Partially parsed test_find_with_nonexistent_path. Retrieved 4/14 statements.
# Partially parsed test_find_with_supported_and_unsupported_files. Retrieved 10/23 statements.
# Partially parsed test_find_with_single_file. Retrieved 6/17 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '/test_dir'
    var_3 = [var_2]
    var_4 = True
    var_5 = '/test_dir/skip'
    var_6 = '/test_dir/file1.py'
    var_7 = 'w'
    var_8 = open(var_6, var_7)
    var_9 = '/test_dir/skip/file2.py'
    var_10 = open(var_9, var_7)

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '/nonexistent'
    var_3 = [var_2]

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '/test_dir'
    var_3 = [var_2]
    var_4 = True
    var_5 = '/test_dir/file1.py'
    var_6 = 'w'
    var_7 = open(var_5, var_6)
    var_8 = '/test_dir/file2.txt'
    var_9 = open(var_8, var_6)

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '/test_file.py'
    var_3 = [var_2]
    var_4 = 'w'
    var_5 = open(var_2, var_4)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_find_skips_directories. Retrieved 9/11 statements.
# Partially parsed test_find_yields_supported_files. Retrieved 10/12 statements.
# Partially parsed test_find_adds_broken_paths. Retrieved 9/11 statements.
# Partially parsed test_find_follows_links_when_configured. Retrieved 9/12 statements.
# Partially parsed test_find_skips_visited_directories. Retrieved 12/15 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = 'skip'
    var_2 = True
    var_3 = []
    var_4 = []
    var_5 = 'test_dir'
    var_6 = [var_5]
    var_7 = module_1.find(var_6, var_0, var_3, var_4)
    var_8 = list(var_7)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = False
    var_2 = '.py'
    var_3 = []
    var_4 = []
    var_5 = 'test_file.py'
    var_6 = 'test_file.txt'
    var_7 = [var_5, var_6]
    var_8 = module_1.find(var_7, var_0, var_3, var_4)
    var_9 = list(var_8)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = False
    var_2 = True
    var_3 = []
    var_4 = []
    var_5 = 'nonexistent'
    var_6 = [var_5]
    var_7 = module_1.find(var_6, var_0, var_3, var_4)
    var_8 = list(var_7)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = False
    var_2 = True
    var_3 = []
    var_4 = []
    var_5 = 'symlink_dir'
    var_6 = [var_5]
    var_7 = module_1.find(var_6, var_0, var_3, var_4)
    var_8 = list(var_7)

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = False
    var_2 = True
    var_3 = []
    var_4 = []
    var_5 = 'dir_with_loop'
    var_6 = [var_5]
    var_7 = module_1.find(var_6, var_0, var_3, var_4)
    var_8 = list(var_7)
    var_9 = 'loop_dir/file'
    var_10 = [p for p in var_8 if var_9 in p]
    var_11 = len(var_10)
    assert var_11 == 1



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_find_yields_files_in_directory. Retrieved 7/22 statements.
# Partially parsed test_find_skips_unsupported_files. Retrieved 7/22 statements.
# Partially parsed test_find_skips_skipped_paths. Retrieved 10/28 statements.
# Partially parsed test_find_reports_broken_paths. Retrieved 4/14 statements.
# Partially parsed test_find_yields_direct_files. Retrieved 5/17 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'test_dir'
    var_3 = [var_2]
    var_4 = True
    var_5 = ''
    var_6 = ''

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'test_dir'
    var_3 = [var_2]
    var_4 = True
    var_5 = ''
    var_6 = ''

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'test_dir'
    var_3 = [var_2]
    var_4 = True
    var_5 = 'test_dir/skip_dir'
    var_6 = ''
    var_7 = ''
    var_8 = set(var_0)
    var_9 = 'test_dir/skip_file.py'

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'nonexistent_dir'
    var_3 = [var_2]

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'file1.py'
    var_3 = [var_2]
    var_4 = ''



# Parsed testcases at query #6
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
    var_3 = 'test_dir/skip'
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
    var_0 = 'non_existent_dir'
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
    var_0 = 'test_file.py'
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
    var_0 = 'test_file.txt'
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



# Parsed testcases at query #7
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = True
    var_3 = module_0.Config()
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 1

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'non_existent_path'
    var_1 = [var_0]
    var_2 = True
    var_3 = module_0.Config()
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 0

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_file.py'
    var_1 = [var_0]
    var_2 = True
    var_3 = module_0.Config()
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 1

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_file.txt'
    var_1 = [var_0]
    var_2 = True
    var_3 = module_0.Config()
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 0

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
    assert var_8 == 1

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_file.py'
    var_2 = 'non_existent_path'
    var_3 = [var_0, var_1, var_2]
    var_4 = True
    var_5 = module_0.Config()
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_3, var_5, var_6, var_7)
    var_9 = list(var_8)
    var_10 = len(var_9)
    assert var_10 == 2



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_predicate_at_line_31_evaluates_to_true. Retrieved 9/10 statements.


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
    var_7 = 0
    var_8 = var_1[var_7]



# Parsed testcases at query #9
#--------------------------




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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_find_finds_files_in_directory. Retrieved 12/17 statements.
# Partially parsed test_find_handles_skipped_directories. Retrieved 13/17 statements.
# Partially parsed test_find_handles_direct_single_file. Retrieved 10/12 statements.
# Partially parsed test_find_skips_unsupported_filetypes. Retrieved 13/17 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'test_dir'
    var_4 = [var_3]
    var_5 = True
    var_6 = ''
    var_7 = ''
    var_8 = module_1.find(var_4, var_0, var_1, var_2)
    var_9 = list(var_8)
    var_10 = len(var_1)
    assert var_10 == 0
    var_11 = len(var_2)
    assert var_11 == 0

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'test_dir'
    var_4 = [var_3]
    var_5 = 'test_dir/skip_dir'
    var_6 = True
    var_7 = ''
    var_8 = 'skip_dir'
    var_9 = module_1.find(var_4, var_0, var_1, var_2)
    var_10 = list(var_9)
    var_11 = len(var_10)
    assert var_11 == 0
    var_12 = len(var_2)
    assert var_12 == 0

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'nonexistent_path'
    var_4 = [var_3]
    var_5 = module_1.find(var_4, var_0, var_1, var_2)
    var_6 = list(var_5)
    var_7 = len(var_6)
    assert var_7 == 0
    var_8 = len(var_1)
    assert var_8 == 0

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'test_file.py'
    var_4 = [var_3]
    var_5 = ''
    var_6 = module_1.find(var_4, var_0, var_1, var_2)
    var_7 = list(var_6)
    var_8 = len(var_1)
    assert var_8 == 0
    var_9 = len(var_2)
    assert var_9 == 0

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []
    var_3 = 'test_dir'
    var_4 = [var_3]
    var_5 = True
    var_6 = ''
    var_7 = '.py'
    var_8 = module_1.find(var_4, var_0, var_1, var_2)
    var_9 = list(var_8)
    var_10 = len(var_9)
    assert var_10 == 0
    var_11 = len(var_1)
    assert var_11 == 0
    var_12 = len(var_2)
    assert var_12 == 0



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_predicate_at_line_9_evaluates_to_false. Retrieved 7/8 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'non_existent_directory'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)



# Parsed testcases at query #12
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '/non-existent-directory'
    var_1 = [var_0]
    var_2 = False
    var_3 = module_0.Config()
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_find_skips_directories_based_on_config. Retrieved 5/17 statements.


def test_case_0():
    var_0 = '/test/skipped_dir'
    var_1 = []
    var_2 = []
    var_3 = '/test'
    var_4 = [var_3]



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_predicate_at_line_9_evaluates_to_false. Retrieved 9/10 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '/nonexistent/directory'
    var_1 = [var_0]
    var_2 = False
    var_3 = module_0.Config()
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = var_1[var_2]



# Parsed testcases at query #15
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'non_existent_file'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_predicate_at_line_9_evaluates_to_true. Retrieved 16/23 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '/some/directory'
    var_1 = [var_0]
    var_2 = True
    var_3 = module_0.Config()
    var_4 = []
    var_5 = []
    var_6 = 'root'
    var_7 = 'dir1'
    var_8 = [var_7]
    var_9 = 'file1.py'
    var_10 = [var_9]
    var_11 = (var_6, var_8, var_10)
    var_12 = [var_11]
    var_13 = iter(var_12)
    var_14 = module_1.find(var_1, var_3, var_4, var_5)
    var_15 = list(var_14)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_resolved_path_not_in_visited_dirs. Retrieved 13/17 statements.


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = module_0.Config()
    var_1 = False
    var_2 = True
    var_3 = []
    var_4 = []
    var_5 = set()
    var_6 = 'test_dir'
    var_7 = [var_6]
    var_8 = module_1.find(var_7, var_0, var_3, var_4)
    var_9 = list(var_8)
    var_10 = len(var_9)
    var_11 = len(var_3)
    assert var_11 == 0
    var_12 = len(var_4)
    assert var_12 == 0



# Parsed testcases at query #18
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '/some/existing/directory'
    var_1 = [var_0]
    var_2 = True
    var_3 = module_0.Config()
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = len(var_7)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_find_with_directory_and_skipped_files. Retrieved 21/41 statements.
# Partially parsed test_find_with_nonexistent_path. Retrieved 6/17 statements.
# Partially parsed test_find_with_single_file. Retrieved 7/21 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 19/39 statements.


def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = '/test_dir'
    var_4 = [var_3]
    var_5 = 'skip_me_dir'
    var_6 = 'normal_dir'
    var_7 = [var_5, var_6]
    var_8 = 'test.py'
    var_9 = 'skip_me.py'
    var_10 = 'ignore.txt'
    var_11 = [var_8, var_9, var_10]
    var_12 = (var_3, var_7, var_11)
    var_13 = '/test_dir/normal_dir'
    var_14 = []
    var_15 = 'normal.py'
    var_16 = [var_15]
    var_17 = (var_13, var_14, var_16)
    var_18 = [var_12, var_17]
    var_19 = iter(var_18)
    var_20 = True

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = '/nonexistent'
    var_4 = [var_3]
    var_5 = False

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = '/test_file.py'
    var_4 = [var_3]
    var_5 = False
    var_6 = True

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = '/test_dir'
    var_4 = [var_3]
    var_5 = 'skip_me_dir'
    var_6 = 'normal_dir'
    var_7 = [var_5, var_6]
    var_8 = 'test.py'
    var_9 = [var_8]
    var_10 = (var_3, var_7, var_9)
    var_11 = '/test_dir/normal_dir'
    var_12 = []
    var_13 = 'normal.py'
    var_14 = [var_13]
    var_15 = (var_11, var_12, var_14)
    var_16 = [var_10, var_15]
    var_17 = iter(var_16)
    var_18 = True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_predicate_at_line_9_evaluates_to_false. Retrieved 7/10 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = '/non-existent-directory'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = 0
    var_6 = var_1[var_5]



# Parsed testcases at query #21
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



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_find_with_directory. Retrieved 6/17 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 6/17 statements.
# Partially parsed test_find_with_broken_path. Retrieved 6/17 statements.
# Partially parsed test_find_with_unsupported_filetype. Retrieved 6/17 statements.
# Partially parsed test_find_with_follow_links. Retrieved 6/17 statements.


def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = len(var_2)
    assert var_4 == 0
    var_5 = len(var_3)
    assert var_5 == 0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = len(var_2)
    var_5 = len(var_3)
    assert var_5 == 0

def test_case_0():
    var_0 = 'non_existent_dir'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = len(var_2)
    assert var_4 == 0
    var_5 = len(var_3)

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = len(var_2)
    assert var_4 == 0
    var_5 = len(var_3)
    assert var_5 == 0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = len(var_2)
    assert var_4 == 0
    var_5 = len(var_3)
    assert var_5 == 0



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_predicate_at_line_9_evaluates_to_false. Retrieved 9/10 statements.


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
    var_7 = 0
    var_8 = var_1[var_7]



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_find_with_directory_and_skipped_files. Retrieved 6/17 statements.
# Partially parsed test_find_with_nonexistent_path. Retrieved 6/17 statements.
# Partially parsed test_find_with_supported_file. Retrieved 6/17 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 6/17 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '/test_dir'
    var_3 = [var_2]
    var_4 = len(var_0)
    assert var_4 == 0
    var_5 = len(var_1)
    assert var_5 == 0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '/nonexistent'
    var_3 = [var_2]
    var_4 = len(var_0)
    assert var_4 == 0
    var_5 = len(var_1)
    assert var_5 == 1

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '/test.py'
    var_3 = [var_2]
    var_4 = len(var_0)
    assert var_4 == 0
    var_5 = len(var_1)
    assert var_5 == 0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '/skip.py'
    var_3 = [var_2]
    var_4 = len(var_0)
    assert var_4 == 1
    var_5 = len(var_1)
    assert var_5 == 0



# Parsed testcases at query #25
#--------------------------




import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = '/nonexistent/path'
    var_1 = [var_0]
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_1, var_2, var_3, var_4)
    var_6 = len(var_4)
    assert var_6 == 1



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_find_skips_directories_based_on_config. Retrieved 6/17 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = False
    var_3 = '/path/to/skip'
    var_4 = [var_3]
    var_5 = [var_3]



# Parsed testcases at query #27
#--------------------------




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
    var_7 = len(var_4)
    assert var_7 == 1



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_find_returns_correct_files. Retrieved 8/11 statements.
# Partially parsed test_find_skips_directories. Retrieved 9/13 statements.
# Partially parsed test_find_reports_broken_paths. Retrieved 8/11 statements.
# Partially parsed test_find_follows_symlinks_when_enabled. Retrieved 8/11 statements.
# Partially parsed test_find_ignores_unsupported_filetypes. Retrieved 8/11 statements.
# Partially parsed test_find_handles_empty_paths. Retrieved 7/10 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = '.py'
    var_1 = [var_0]
    var_2 = False
    var_3 = module_0.Config()
    var_4 = []
    var_5 = []
    var_6 = 'test_dir'
    var_7 = [var_6]

import isort.settings as module_0

def test_case_0():
    var_0 = '.py'
    var_1 = [var_0]
    var_2 = False
    var_3 = module_0.Config()
    var_4 = []
    var_5 = []
    var_6 = 'test_dir'
    var_7 = [var_6]
    var_8 = 'skip'

import isort.settings as module_0

def test_case_0():
    var_0 = '.py'
    var_1 = [var_0]
    var_2 = False
    var_3 = module_0.Config()
    var_4 = []
    var_5 = []
    var_6 = 'nonexistent_dir'
    var_7 = [var_6]

import isort.settings as module_0

def test_case_0():
    var_0 = '.py'
    var_1 = [var_0]
    var_2 = True
    var_3 = module_0.Config()
    var_4 = []
    var_5 = []
    var_6 = 'test_dir'
    var_7 = [var_6]

import isort.settings as module_0

def test_case_0():
    var_0 = '.txt'
    var_1 = [var_0]
    var_2 = False
    var_3 = module_0.Config()
    var_4 = []
    var_5 = []
    var_6 = 'test_dir'
    var_7 = [var_6]

import isort.settings as module_0

def test_case_0():
    var_0 = '.py'
    var_1 = [var_0]
    var_2 = False
    var_3 = module_0.Config()
    var_4 = []
    var_5 = []
    var_6 = []



