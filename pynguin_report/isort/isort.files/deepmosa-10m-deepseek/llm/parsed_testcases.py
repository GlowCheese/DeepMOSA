####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_find_with_directory_and_skipped_path. Retrieved 7/34 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 8/37 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 7/35 statements.
# Partially parsed test_find_with_broken_path. Retrieved 5/22 statements.
# Partially parsed test_find_with_file_path. Retrieved 3/24 statements.
# Partially parsed test_find_with_unsupported_filetype. Retrieved 5/27 statements.
# Partially parsed test_find_with_follow_links. Retrieved 5/22 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = 'subdir'
    var_4 = 'file1.py'
    var_5 = 'file2.py'
    var_6 = 'w'
    var_7 = bool(var_1 == [])
    assert var_7 is True
    var_8 = bool(var_2 == [])
    assert var_8 is True

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = 'subdir'
    var_4 = 'file1.py'
    var_5 = 'file2.py'
    var_6 = 'w'
    var_7 = len(var_1)
    assert var_7 == 1
    var_8 = var_1[0]
    var_9 = bool(var_1[0] == var_5)
    assert var_9 is True
    var_10 = bool(var_2 == [])
    assert var_10 is True

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = 'file1.py'
    var_4 = 'file2.py'
    assert var_4 == 1
    var_5 = 'w'
    var_6 = len(var_1)
    assert var_6 == 1
    var_7 = var_1[0]
    var_8 = bool(var_2 == [])
    assert var_8 is True

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = '/nonexistent/path'
    var_4 = [var_3]
    var_5 = bool(var_1 == [])
    assert var_5 is True
    var_6 = bool(var_2 == ['/nonexistent/path'])
    assert var_6 is True

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = bool(var_1 == [])
    assert var_3 is True
    var_4 = bool(var_2 == [])
    assert var_4 is True

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = 'file1.txt'
    var_4 = 'w'
    assert var_4 == 0
    var_5 = bool(var_1 == [])
    assert var_5 is True
    var_6 = bool(var_2 == [])
    assert var_6 is True

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = 'subdir'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_predicate_at_line_7_evaluates_to_false. Retrieved 9/12 statements.


import isort.files as module_1
import isort.settings as module_0


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = False
    var_3 = True
    var_4 = '/non_existent_file.py'
    var_5 = [var_4]
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_5, var_1, var_6, var_7)
    var_9 = list(var_8)
    var_10 = bool(var_9 == [])
    assert var_10 is True
    var_11 = bool(var_7 == ['/non_existent_file.py'])
    assert var_11 is True
    var_12 = bool(var_6 == [])
    assert var_12 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_find_with_directory_and_skipped_path. Retrieved 8/31 statements.
# Partially parsed test_find_with_nonexistent_path. Retrieved 5/13 statements.
# Partially parsed test_find_with_single_file. Retrieved 3/16 statements.
# Partially parsed test_find_with_unsupported_filetype. Retrieved 3/16 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 3/18 statements.
# Partially parsed test_find_with_follow_links_and_cycle. Retrieved 4/16 statements.


def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = 'skip_dir'
    assert var_3 == 1
    var_4 = 'normal_dir'
    var_5 = 0
    var_6 = 'a.py'
    var_7 = 'skip_dir'
    var_8 = bool(var_2 == [])
    assert var_8 is True

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = '/nonexistent/path'
    var_4 = [var_3]
    var_5 = bool(var_1 == [])
    assert var_5 is True
    var_6 = bool(var_2 == ['/nonexistent/path'])
    assert var_6 is True

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = bool(var_1 == [])
    assert var_3 is True
    var_4 = bool(var_2 == [])
    assert var_4 is True

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = bool(var_1 == [])
    assert var_3 is True
    var_4 = bool(var_2 == [])
    assert var_4 is True

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = bool(var_2 == [])
    assert var_3 is True

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = []
    var_3 = 'link'
    var_4 = bool(var_2 == [])
    assert var_4 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_find_with_directory_and_skipped_path. Retrieved 8/40 statements.
# Partially parsed test_find_with_nonexistent_path. Retrieved 4/15 statements.
# Partially parsed test_find_with_single_file. Retrieved 4/22 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 5/26 statements.
# Partially parsed test_find_with_unsupported_filetype. Retrieved 4/22 statements.
# Partially parsed test_find_with_follow_links_and_visited_dirs. Retrieved 10/40 statements.


def test_case_0():
    var_0 = 'subdir'
    var_1 = 'file1.py'
    var_2 = 'w'
    var_3 = 'file2.py'
    var_4 = 'file3.txt'
    var_5 = []
    var_6 = []
    var_7 = len(var_5)
    assert var_7 == 1
    var_8 = bool(var_6 == [])
    assert var_8 is True

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
    var_0 = '.py'
    var_1 = False
    var_2 = []
    var_3 = []
    var_4 = len(var_2)
    assert var_4 == 1
    var_5 = bool(var_3 == [])
    assert var_5 is True

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
    var_0 = 'subdir'
    var_1 = 'linkdir'
    var_2 = 'file.py'
    var_3 = 'w'
    var_4 = True
    var_5 = []
    var_6 = []
    var_7 = 'file.py'
    var_8 = False
    var_9 = []
    var_10 = []
    var_11 = 'file.py'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_predicate_at_line_8_evaluates_true_for_directory. Retrieved 8/10 statements.



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



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_find_with_directory_and_skipped_path. Retrieved 6/35 statements.
# Partially parsed test_find_with_nonexistent_path. Retrieved 4/16 statements.
# Partially parsed test_find_with_single_file. Retrieved 4/23 statements.
# Partially parsed test_find_with_unsupported_filetype. Retrieved 4/23 statements.
# Partially parsed test_find_with_follow_links_and_cycle. Retrieved 4/23 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 4/29 statements.
# Partially parsed test_find_with_multiple_paths. Retrieved 5/33 statements.


def test_case_0():
    var_0 = 'subdir'
    var_1 = 'file1.py'
    var_2 = 'file2.py'
    var_3 = 'w'
    var_4 = []
    var_5 = []
    var_6 = bool(var_5 == [])
    assert var_6 is True

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
    var_0 = '.txt'
    var_1 = False
    var_2 = []
    var_3 = []
    var_4 = bool(var_2 == [])
    assert var_4 is True
    var_5 = bool(var_3 == [])
    assert var_5 is True

def test_case_0():
    var_0 = 'link'
    var_1 = True
    var_2 = []
    var_3 = []
    var_4 = bool(var_2 == [])
    assert var_4 is True
    var_5 = bool(var_3 == [])
    assert var_5 is True

def test_case_0():
    var_0 = '.py'
    var_1 = False
    var_2 = []
    var_3 = []
    var_4 = bool(var_3 == [])
    assert var_4 is True

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



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_at_line_8_evaluates_true_for_directory. Retrieved 8/10 statements.



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



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_find_with_single_file. Retrieved 9/12 statements.
# Partially parsed test_find_with_nonexistent_file. Retrieved 9/12 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 8/11 statements.
# Partially parsed test_find_with_directory. Retrieved 15/20 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 19/24 statements.
# Partially parsed test_find_with_symlink_following. Retrieved 14/19 statements.
# Partially parsed test_find_with_duplicate_resolved_path. Retrieved 16/24 statements.
# Partially parsed test_find_with_multiple_paths. Retrieved 16/21 statements.
# Partially parsed test_find_with_skipped_file_in_directory. Retrieved 15/20 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '.py'
    var_3 = False
    var_4 = []
    var_5 = []
    var_6 = '/tmp/test.py'
    var_7 = [var_6]
    var_8 = module_1.find(var_7, var_1, var_4, var_5)
    var_9 = list(var_8)
    var_10 = bool(var_9 == ['/tmp/test.py'])
    assert var_10 is True
    var_11 = bool(var_4 == [])
    assert var_11 is True
    var_12 = bool(var_5 == [])
    assert var_12 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '.py'
    var_3 = False
    var_4 = []
    var_5 = []
    var_6 = '/tmp/nonexistent.py'
    var_7 = [var_6]
    var_8 = module_1.find(var_7, var_1, var_4, var_5)
    var_9 = list(var_8)
    var_10 = bool(var_9 == [])
    assert var_10 is True
    var_11 = bool(var_4 == [])
    assert var_11 is True
    var_12 = bool(var_5 == ['/tmp/nonexistent.py'])
    assert var_12 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '.py'
    var_3 = '/tmp/skipped.py'
    var_4 = []
    var_5 = []
    var_6 = [var_3]
    var_7 = module_1.find(var_6, var_1, var_4, var_5)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [])
    assert var_9 is True
    var_10 = bool(var_4 == ['/tmp/skipped.py'])
    assert var_10 is True
    var_11 = bool(var_5 == [])
    assert var_11 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '.py'
    var_3 = False
    var_4 = []
    var_5 = []
    var_6 = '/tmp/dir'
    var_7 = [var_6]
    var_8 = '/tmp/dir'
    var_9 = []
    var_10 = 'file1.py'
    var_11 = 'file2.txt'
    var_12 = [var_10, var_11]
    var_13 = (var_8, var_9, var_12)
    var_14 = module_1.find(var_7, var_1, var_4, var_5)
    var_15 = list(var_14)
    var_16 = bool(var_15 == ['/tmp/dir/file1.py'])
    assert var_16 is True
    var_17 = bool(var_4 == [])
    assert var_17 is True
    var_18 = bool(var_5 == [])
    assert var_18 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '.py'
    var_3 = '/tmp/dir/skip'
    var_4 = []
    var_5 = []
    var_6 = '/tmp/dir'
    var_7 = [var_6]
    var_8 = '/tmp/dir'
    var_9 = 'skip'
    var_10 = [var_9]
    var_11 = []
    var_12 = (var_8, var_10, var_11)
    var_13 = '/tmp/dir/skip'
    var_14 = []
    var_15 = 'file.py'
    var_16 = [var_15]
    var_17 = (var_13, var_14, var_16)
    var_18 = module_1.find(var_7, var_1, var_4, var_5)
    var_19 = list(var_18)
    var_20 = bool(var_19 == [])
    assert var_20 is True
    var_21 = bool(var_4 == ['/tmp/dir/skip'])
    assert var_21 is True
    var_22 = bool(var_5 == [])
    assert var_22 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '.py'
    var_3 = False
    var_4 = []
    var_5 = []
    var_6 = '/tmp/dir'
    var_7 = [var_6]
    var_8 = '/tmp/dir'
    var_9 = []
    var_10 = 'link.py'
    var_11 = [var_10]
    var_12 = (var_8, var_9, var_11)
    var_13 = module_1.find(var_7, var_1, var_4, var_5)
    var_14 = list(var_13)
    var_15 = bool(var_14 == ['/tmp/dir/link.py'])
    assert var_15 is True
    var_16 = bool(var_4 == [])
    assert var_16 is True
    var_17 = bool(var_5 == [])
    assert var_17 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '.py'
    var_3 = False
    var_4 = []
    var_5 = []
    var_6 = '/tmp/dir'
    var_7 = [var_6]
    var_8 = set()
    var_9 = '/tmp/dir'
    var_10 = 'subdir'
    var_11 = [var_10]
    var_12 = []
    var_13 = (var_9, var_11, var_12)
    var_14 = '/tmp/real'
    var_15 = [var_14]
    var_16 = [var_14]
    var_17 = module_1.find(var_7, var_1, var_4, var_5)
    var_18 = list(var_17)
    var_19 = bool(var_18 == [])
    assert var_19 is True
    var_20 = bool(var_4 == [])
    assert var_20 is True
    var_21 = bool(var_5 == [])
    assert var_21 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '.py'
    var_3 = False
    var_4 = []
    var_5 = []
    var_6 = '/tmp/file1.py'
    var_7 = '/tmp/dir'
    var_8 = '/tmp/missing.py'
    var_9 = [var_6, var_7, var_8]
    var_10 = '/tmp/dir'
    var_11 = []
    var_12 = 'file2.py'
    var_13 = [var_12]
    var_14 = (var_10, var_11, var_13)
    var_15 = module_1.find(var_9, var_1, var_4, var_5)
    var_16 = list(var_15)
    var_17 = bool(var_16 == ['/tmp/file1.py', '/tmp/dir/file2.py'])
    assert var_17 is True
    var_18 = bool(var_4 == [])
    assert var_18 is True
    var_19 = bool(var_5 == ['/tmp/missing.py'])
    assert var_19 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '.py'
    var_3 = '/tmp/dir/skipped.py'
    var_4 = []
    var_5 = []
    var_6 = '/tmp/dir'
    var_7 = [var_6]
    var_8 = '/tmp/dir'
    var_9 = []
    var_10 = 'skipped.py'
    var_11 = 'included.py'
    var_12 = [var_10, var_11]
    var_13 = (var_8, var_9, var_12)
    var_14 = module_1.find(var_7, var_1, var_4, var_5)
    var_15 = list(var_14)
    var_16 = bool(var_15 == ['/tmp/dir/included.py'])
    assert var_16 is True
    var_17 = bool(var_4 == ['/tmp/dir/skipped.py'])
    assert var_17 is True
    var_18 = bool(var_5 == [])
    assert var_18 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_predicate_at_line_8_true_for_directory. Retrieved 9/14 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = False
    var_3 = True
    var_4 = []
    var_5 = []
    var_6 = '/tmp/test_dir'
    var_7 = [var_6]
    var_8 = module_1.find(var_7, var_1, var_4, var_5)
    var_9 = list(var_8)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_predicate_at_line_8_evaluates_to_true. Retrieved 11/22 statements.



def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = []
    var_4 = []
    var_5 = 'test_directory'
    var_6 = True
    var_7 = [var_5]
    var_8 = module_1.find(var_7, var_2, var_3, var_4)
    var_9 = list(var_8)
    var_10 = 0
    var_11 = var_7[var_10]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_predicate_at_line_8_evaluates_to_true_for_directory. Retrieved 8/10 statements.



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



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_find_with_skipped_file. Retrieved 8/11 statements.
# Partially parsed test_find_with_unsupported_filetype. Retrieved 8/9 statements.
# Partially parsed test_find_with_directory. Retrieved 8/21 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 8/19 statements.
# Partially parsed test_find_with_symlink_following. Retrieved 9/23 statements.
# Partially parsed test_find_with_duplicate_resolved_path. Retrieved 9/23 statements.



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


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = False
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


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '.py'
    var_3 = False
    var_4 = []
    var_5 = []
    var_6 = 'file1.py'
    var_7 = 'file2.py'
    var_8 = 'w'
    var_9 = bool(var_4 == [])
    assert var_9 is True
    var_10 = bool(var_5 == [])
    assert var_10 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '.py'
    var_3 = True
    var_4 = []
    var_5 = []
    var_6 = 'subdir'
    var_7 = 'file1.py'
    var_8 = 'w'
    var_9 = bool(var_5 == [])
    assert var_9 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '.py'
    var_3 = False
    var_4 = []
    var_5 = []
    var_6 = 'real'
    var_7 = 'link'
    var_8 = 'file1.py'
    var_9 = 'w'
    var_10 = bool(var_4 == [])
    assert var_10 is True
    var_11 = bool(var_5 == [])
    assert var_11 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '.py'
    var_3 = False
    var_4 = []
    var_5 = []
    var_6 = 'real'
    var_7 = 'link'
    var_8 = 'file1.py'
    var_9 = 'w'
    var_10 = bool(var_4 == [])
    assert var_10 is True
    var_11 = bool(var_5 == [])
    assert var_11 is True



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_find_with_directory_and_skipped_path. Retrieved 12/34 statements.
# Partially parsed test_find_with_nonexistent_path. Retrieved 5/13 statements.
# Partially parsed test_find_with_single_file. Retrieved 4/18 statements.
# Partially parsed test_find_with_unsupported_filetype. Retrieved 4/18 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 4/20 statements.
# Partially parsed test_find_with_follow_links_and_cycle. Retrieved 10/33 statements.


def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = 'skip_dir'
    var_4 = 'normal_dir'
    var_5 = ''
    var_6 = ''
    assert var_6 == 1
    var_7 = 0
    var_8 = 'normal_dir/a.py'
    var_9 = len(var_1)
    assert var_9 == 1
    var_10 = var_1[var_7]
    var_11 = 'skip_dir'
    var_12 = bool(var_2 == [])
    assert var_12 is True

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = '/nonexistent/path'
    var_4 = [var_3]
    var_5 = bool(var_1 == [])
    assert var_5 is True
    var_6 = bool(var_2 == ['/nonexistent/path'])
    assert var_6 is True

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = b''
    var_4 = bool(var_1 == [])
    assert var_4 is True
    var_5 = bool(var_2 == [])
    assert var_5 is True

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = b''
    var_4 = bool(var_1 == [])
    assert var_4 is True
    var_5 = bool(var_2 == [])
    assert var_5 is True

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = b''
    var_4 = bool(var_2 == [])
    assert var_4 is True

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = []
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'link_to_a'
    var_6 = 'link_to_b'
    var_7 = ''
    assert var_7 == 1
    var_8 = 0
    var_9 = 'file.py'
    var_10 = bool(var_5)
    assert var_10 is True
    var_11 = bool(var_1 == [])
    assert var_11 is True
    var_12 = bool(var_2 == [])
    assert var_12 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_predicate_at_line_7_evaluates_to_true. Retrieved 7/8 statements.



def test_case_0():
    var_0 = '/some/directory'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = module_1.find(var_1, var_3, var_4, var_5)
    var_7 = list(var_6)
    var_8 = bool(True)
    assert var_8 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_find_with_directory_and_skipped_path. Retrieved 5/18 statements.
# Partially parsed test_find_with_single_file. Retrieved 4/16 statements.
# Partially parsed test_find_with_nonexistent_path. Retrieved 4/16 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 4/17 statements.
# Partially parsed test_find_with_unsupported_filetype. Retrieved 4/16 statements.


def test_case_0():
    var_0 = '/test/skip_dir'
    var_1 = []
    var_2 = []
    var_3 = '/test'
    var_4 = [var_3]
    var_5 = bool(var_1 == ['/test/skip_dir'])
    assert var_5 is True
    var_6 = bool(var_2 == [])
    assert var_6 is True

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '/test/file.py'
    var_3 = [var_2]
    var_4 = bool(var_0 == [])
    assert var_4 is True
    var_5 = bool(var_1 == [])
    assert var_5 is True

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '/nonexistent'
    var_3 = [var_2]
    var_4 = bool(var_0 == [])
    assert var_4 is True
    var_5 = bool(var_1 == ['/nonexistent'])
    assert var_5 is True

def test_case_0():
    var_0 = '/test/skipped.py'
    var_1 = []
    var_2 = []
    var_3 = [var_0]
    var_4 = bool(var_1 == ['/test/skipped.py'])
    assert var_4 is True
    var_5 = bool(var_2 == [])
    assert var_5 is True

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '/test/file.txt'
    var_3 = [var_2]
    var_4 = bool(var_0 == [])
    assert var_4 is True
    var_5 = bool(var_1 == [])
    assert var_5 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_predicate_at_line_8_evaluates_to_true. Retrieved 4/51 statements.



def test_case_0():
    var_0 = 'test_dir'
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = []
    var_4 = []



# Parsed testcases at query #5
#--------------------------





def test_case_0():
    var_0 = '/fake/nonexistent/file.py'
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
    assert var_11 == '/fake/nonexistent/file.py'



# Parsed testcases at query #6
#--------------------------





def test_case_0():
    var_0 = '/non/existent/file.py'
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
    var_10 = bool(var_5 == ['/non/existent/file.py'])
    assert var_10 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_at_line_8_evaluates_to_true. Retrieved 8/50 statements.



def test_case_0():
    var_0 = 'test_directory'
    var_1 = True
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = [var_0]
    var_7 = module_1.find(var_6, var_3, var_4, var_5)
    var_8 = list(var_7)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_predicate_at_line_8_evaluates_true_for_directory. Retrieved 8/10 statements.



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



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_predicate_at_line_31_evaluates_to_true. Retrieved 9/46 statements.



def test_case_0():
    var_0 = '.py'
    var_1 = [var_0]
    var_2 = 'supported_extensions'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    var_5 = []
    var_6 = []
    var_7 = 'test_nonexistent_dir'
    var_8 = [var_7]
    var_9 = module_1.find(var_8, var_4, var_5, var_6)
    var_10 = list(var_9)
    var_11 = bool(var_6 == [var_7])
    assert var_11 is True
    var_12 = bool(var_10 == [])
    assert var_12 is True



# Parsed testcases at query #10
#--------------------------





def test_case_0():
    var_0 = '/non/existent/file.py'
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
    assert var_11 == '/non/existent/file.py'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_find_with_directory_and_skipped. Retrieved 7/39 statements.
# Partially parsed test_find_with_nonexistent_path. Retrieved 4/15 statements.
# Partially parsed test_find_with_single_file. Retrieved 4/22 statements.
# Partially parsed test_find_with_unsupported_filetype. Retrieved 4/22 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 5/27 statements.
# Partially parsed test_find_with_follow_links. Retrieved 7/34 statements.
# Partially parsed test_find_with_duplicate_resolved_path. Retrieved 8/39 statements.


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
    var_0 = '.txt'
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
    var_0 = 'link_dir'
    var_1 = 'target_dir'
    var_2 = 'file.py'
    var_3 = 'w'
    var_4 = True
    var_5 = []
    var_6 = []
    var_7 = bool(var_5 == [])
    assert var_7 is True
    var_8 = bool(var_6 == [])
    assert var_8 is True

def test_case_0():
    var_0 = 'dir1'
    var_1 = 'dir2'
    var_2 = 'file.py'
    var_3 = 'w'
    var_4 = 'link'
    var_5 = True
    var_6 = []
    var_7 = []
    var_8 = bool(var_6 == [])
    assert var_8 is True
    var_9 = bool(var_7 == [])
    assert var_9 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_predicate_at_line_9_evaluates_to_true. Retrieved 10/23 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = []
    var_3 = []
    var_4 = 'test_dir'
    var_5 = True
    var_6 = [var_4]
    var_7 = module_1.find(var_6, var_1, var_2, var_3)
    var_8 = list(var_7)
    var_9 = 0
    var_10 = var_6[var_9]



# Parsed testcases at query #13
#--------------------------





def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = []
    var_4 = []
    var_5 = '/non_existent_directory'
    var_6 = [var_5]
    var_7 = module_1.find(var_6, var_2, var_3, var_4)
    var_8 = list(var_7)
    var_9 = len(var_8)
    assert var_9 == 0
    var_10 = len(var_4)
    assert var_10 == 1
    var_11 = var_4[0]
    assert var_11 == '/non_existent_directory'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_find_with_directory_and_skipped_path. Retrieved 13/36 statements.
# Partially parsed test_find_with_nonexistent_path. Retrieved 5/13 statements.
# Partially parsed test_find_with_single_file. Retrieved 4/18 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 4/20 statements.
# Partially parsed test_find_with_unsupported_filetype. Retrieved 4/18 statements.
# Partially parsed test_find_with_follow_links_and_visited_dirs. Retrieved 9/29 statements.


def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = 'skip_dir'
    var_4 = 'keep_dir'
    var_5 = ''
    var_6 = ''
    assert var_6 == 1
    var_7 = 0
    var_8 = 'keep_dir'
    var_9 = 'b.py'
    var_10 = len(var_1)
    assert var_10 == 1
    var_11 = var_1[var_7]
    var_12 = 'skip_dir'
    var_13 = bool(var_2 == [])
    assert var_13 is True

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = '/nonexistent/path'
    var_4 = [var_3]
    var_5 = bool(var_1 == [])
    assert var_5 is True
    var_6 = bool(var_2 == ['/nonexistent/path'])
    assert var_6 is True

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = b''
    var_4 = bool(var_1 == [])
    assert var_4 is True
    var_5 = bool(var_2 == [])
    assert var_5 is True

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = b''
    var_4 = bool(var_2 == [])
    assert var_4 is True

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = b''
    var_4 = bool(var_1 == [])
    assert var_4 is True
    var_5 = bool(var_2 == [])
    assert var_5 is True

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = []
    var_3 = 'real'
    var_4 = 'link'
    var_5 = ''
    assert var_5 == 1
    var_6 = 0
    var_7 = 'real'
    var_8 = 'a.py'
    var_9 = bool(var_1 == [])
    assert var_9 is True
    var_10 = bool(var_2 == [])
    assert var_10 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_find_with_skipped_directory. Retrieved 8/9 statements.
# Partially parsed test_find_with_supported_file_in_directory. Retrieved 9/12 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 9/12 statements.
# Partially parsed test_find_with_mixed_paths. Retrieved 11/14 statements.



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


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = 'skip'
    var_3 = []
    var_4 = []
    var_5 = [var_2]
    var_6 = module_1.find(var_5, var_1, var_3, var_4)
    var_7 = list(var_6)
    var_8 = bool(var_7 == [])
    assert var_8 is True
    var_9 = len(var_3)
    assert var_9 == 1
    var_10 = bool(var_4 == [])
    assert var_10 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '.py'
    var_3 = False
    var_4 = []
    var_5 = []
    var_6 = 'dir'
    var_7 = [var_6]
    var_8 = module_1.find(var_7, var_1, var_4, var_5)
    var_9 = list(var_8)
    var_10 = 'dir/test.py'
    var_11 = bool('dir/test.py' in var_9)
    assert var_11 is True
    var_12 = bool(var_4 == [])
    assert var_12 is True
    var_13 = bool(var_5 == [])
    assert var_13 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '.py'
    var_3 = 'skip.py'
    var_4 = []
    var_5 = []
    var_6 = [var_3]
    var_7 = module_1.find(var_6, var_1, var_4, var_5)
    var_8 = list(var_7)
    var_9 = bool(var_8 == [])
    assert var_9 is True
    var_10 = len(var_4)
    assert var_10 == 1
    var_11 = bool(var_5 == [])
    assert var_11 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
    var_2 = '.py'
    var_3 = False
    var_4 = []
    var_5 = []
    var_6 = 'file.py'
    var_7 = 'nonexistent.py'
    var_8 = 'dir'
    var_9 = [var_6, var_7, var_8]
    var_10 = module_1.find(var_9, var_1, var_4, var_5)
    var_11 = list(var_10)
    var_12 = 'file.py'
    var_13 = bool('file.py' in var_11)
    assert var_13 is True
    var_14 = 'dir/test.py'
    var_15 = bool('dir/test.py' in var_11)
    assert var_15 is True
    var_16 = bool(var_5 == ['nonexistent.py'])
    assert var_16 is True
    var_17 = bool(var_4 == [])
    assert var_17 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_predicate_at_line_9_evaluates_to_true. Retrieved 8/50 statements.



def test_case_0():
    var_0 = 'test_directory'
    var_1 = True
    var_2 = {}
    var_3 = module_0.Config(**var_2)
    var_4 = []
    var_5 = []
    var_6 = [var_0]
    var_7 = module_1.find(var_6, var_3, var_4, var_5)
    var_8 = list(var_7)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_predicate_at_line_9_evaluates_to_false. Retrieved 10/20 statements.



def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Config(**var_1)
    var_3 = []
    var_4 = []
    var_5 = '/non_existent_directory'
    var_6 = [var_5]
    var_7 = module_1.find(var_6, var_2, var_3, var_4)
    var_8 = list(var_7)
    var_9 = 0
    var_10 = var_6[var_9]
    var_11 = 'non_existent_directory'
    var_12 = bool('non_existent_directory' in var_4)
    assert var_12 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_find_with_directory_and_skipped. Retrieved 13/36 statements.
# Partially parsed test_find_with_nonexistent_path. Retrieved 6/15 statements.
# Partially parsed test_find_with_single_file. Retrieved 6/21 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 6/22 statements.
# Partially parsed test_find_with_unsupported_filetype. Retrieved 6/21 statements.
# Partially parsed test_find_with_follow_links. Retrieved 9/29 statements.
# Partially parsed test_find_with_visited_dirs_loop. Retrieved 9/28 statements.


def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = 'skip_dir'
    var_4 = 'keep_dir'
    var_5 = ''
    var_6 = ''
    assert var_6 == 1
    var_7 = 0
    var_8 = 'keep_dir'
    var_9 = 'b.py'
    var_10 = len(var_1)
    assert var_10 == 1
    var_11 = var_1[var_7]
    var_12 = 'skip_dir'

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
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = b''
    var_4 = len(var_1)
    assert var_4 == 0
    var_5 = len(var_2)
    assert var_5 == 0

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = b''
    var_4 = len(var_1)
    assert var_4 == 1
    var_5 = var_1[0]
    var_6 = len(var_2)
    assert var_6 == 0

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = b''
    var_4 = len(var_1)
    assert var_4 == 0
    var_5 = len(var_2)
    assert var_5 == 0

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = []
    var_3 = 'real'
    var_4 = 'link'
    var_5 = ''
    assert var_5 == 1
    var_6 = 0
    var_7 = 'real'
    var_8 = 'a.py'

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = []
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'link_to_a'
    var_6 = ''
    var_7 = 'file.py'
    var_8 = len(var_4)
    assert var_8 == 1



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_find_with_directory_and_skipped_path. Retrieved 13/36 statements.
# Partially parsed test_find_with_nonexistent_path. Retrieved 5/13 statements.
# Partially parsed test_find_with_single_file. Retrieved 4/18 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 4/20 statements.
# Partially parsed test_find_with_unsupported_filetype. Retrieved 4/18 statements.
# Partially parsed test_find_with_follow_links_and_cycle. Retrieved 10/32 statements.
# Partially parsed test_find_with_multiple_paths. Retrieved 11/36 statements.
# Partially parsed test_find_with_empty_paths. Retrieved 4/12 statements.


def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = 'skip_dir'
    var_4 = 'normal_dir'
    var_5 = ''
    var_6 = ''
    assert var_6 == 1
    var_7 = 0
    var_8 = 'normal_dir'
    var_9 = 'a.py'
    var_10 = len(var_1)
    assert var_10 == 1
    var_11 = var_1[var_7]
    var_12 = 'skip_dir'
    var_13 = bool(var_2 == [])
    assert var_13 is True

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = '/nonexistent/path'
    var_4 = [var_3]
    var_5 = bool(var_1 == [])
    assert var_5 is True
    var_6 = bool(var_2 == ['/nonexistent/path'])
    assert var_6 is True

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = b''
    var_4 = bool(var_1 == [])
    assert var_4 is True
    var_5 = bool(var_2 == [])
    assert var_5 is True

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = b''
    var_4 = bool(var_2 == [])
    assert var_4 is True

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = b''
    var_4 = bool(var_1 == [])
    assert var_4 is True
    var_5 = bool(var_2 == [])
    assert var_5 is True

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = []
    var_3 = 'target'
    var_4 = 'link'
    var_5 = ''
    assert var_5 == 1
    var_6 = 'loop'
    var_7 = 0
    var_8 = 'target'
    var_9 = 'a.py'
    var_10 = bool(var_1 == [])
    assert var_10 is True
    var_11 = bool(var_2 == [])
    assert var_11 is True

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = 'dir1'
    var_4 = 'dir2'
    var_5 = ''
    var_6 = ''
    assert var_6 == 2
    var_7 = 'dir1'
    var_8 = 'a.py'
    var_9 = 'dir2'
    var_10 = 'b.py'
    var_11 = bool(var_1 == [])
    assert var_11 is True
    var_12 = bool(var_2 == [])
    assert var_12 is True

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = bool(var_1 == [])
    assert var_4 is True
    var_5 = bool(var_2 == [])
    assert var_5 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_find_with_directory_and_skipped_path. Retrieved 6/18 statements.
# Partially parsed test_find_with_supported_file. Retrieved 4/16 statements.
# Partially parsed test_find_with_broken_path. Retrieved 4/16 statements.
# Partially parsed test_find_with_unsupported_filetype. Retrieved 4/16 statements.
# Partially parsed test_find_with_follow_links_and_visited_dirs. Retrieved 5/17 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 5/17 statements.
# Partially parsed test_find_with_multiple_paths. Retrieved 5/17 statements.
# Partially parsed test_find_with_mixed_paths. Retrieved 7/19 statements.


def test_case_0():
    var_0 = '/test/skip_dir'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = '/test'
    var_5 = [var_4]
    var_6 = bool(var_2 == ['/test/skip_dir'])
    assert var_6 is True
    var_7 = bool(var_3 == [])
    assert var_7 is True

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '/test/file.py'
    var_3 = [var_2]
    var_4 = bool(var_0 == [])
    assert var_4 is True
    var_5 = bool(var_1 == [])
    assert var_5 is True

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '/nonexistent'
    var_3 = [var_2]
    var_4 = bool(var_0 == [])
    assert var_4 is True
    var_5 = bool(var_1 == ['/nonexistent'])
    assert var_5 is True

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '/test/file.txt'
    var_3 = [var_2]
    var_4 = bool(var_0 == [])
    assert var_4 is True
    var_5 = bool(var_1 == [])
    assert var_5 is True

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = []
    var_3 = '/test'
    var_4 = [var_3]
    var_5 = bool(var_1 == [])
    assert var_5 is True
    var_6 = bool(var_2 == [])
    assert var_6 is True

def test_case_0():
    var_0 = '/test/skip_file.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = [var_0]
    var_5 = bool(var_2 == ['/test/skip_file.py'])
    assert var_5 is True
    var_6 = bool(var_3 == [])
    assert var_6 is True

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '/test/file1.py'
    var_3 = '/test/file2.py'
    var_4 = [var_2, var_3]
    var_5 = bool(var_0 == [])
    assert var_5 is True
    var_6 = bool(var_1 == [])
    assert var_6 is True

def test_case_0():
    var_0 = '/test/skip.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = '/test/file.py'
    var_5 = '/nonexistent'
    var_6 = [var_4, var_0, var_5]
    var_7 = bool(var_2 == ['/test/skip.py'])
    assert var_7 is True
    var_8 = bool(var_3 == ['/nonexistent'])
    assert var_8 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_find_with_directory_and_skipped. Retrieved 7/14 statements.
# Partially parsed test_find_with_supported_file_in_directory. Retrieved 13/25 statements.
# Partially parsed test_find_with_skipped_file_in_directory. Retrieved 13/26 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 11/22 statements.
# Partially parsed test_find_with_broken_path. Retrieved 4/13 statements.
# Partially parsed test_find_with_direct_file_path. Retrieved 4/13 statements.
# Partially parsed test_find_with_unsupported_filetype. Retrieved 9/20 statements.
# Partially parsed test_find_with_follow_links_and_visited_dir. Retrieved 16/28 statements.
# Partially parsed test_find_with_multiple_paths. Retrieved 16/29 statements.


def test_case_0():
    var_0 = 'skip'
    var_1 = lambda p: str(p).endswith(var_0)
    var_2 = True
    var_3 = []
    var_4 = []
    var_5 = 'test_dir'
    var_6 = [var_5]
    var_7 = bool(var_3 == [])
    assert var_7 is True
    var_8 = bool(var_4 == [])
    assert var_8 is True

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = []
    var_3 = []
    var_4 = 'test_dir'
    var_5 = []
    var_6 = 'file.py'
    var_7 = [var_6]
    var_8 = (var_4, var_5, var_7)
    var_9 = [var_4]
    var_10 = 'test_dir'
    var_11 = 'file.py'
    var_12 = [var_8]
    var_13 = bool(var_2 == [])
    assert var_13 is True
    var_14 = bool(var_3 == [])
    assert var_14 is True

def test_case_0():
    var_0 = 'skip.py'
    var_1 = lambda p: str(p).endswith(var_0)
    var_2 = True
    var_3 = []
    var_4 = []
    var_5 = 'test_dir'
    var_6 = []
    var_7 = 'skip.py'
    var_8 = [var_7]
    var_9 = (var_5, var_6, var_8)
    var_10 = [var_5]
    var_11 = 'test_dir'
    var_12 = [var_10]
    var_13 = bool(var_3 == var_12)
    assert var_13 is True
    var_14 = bool(var_4 == [])
    assert var_14 is True

def test_case_0():
    var_0 = 'skip_dir'
    var_1 = lambda p: str(p).endswith(var_0)
    var_2 = True
    var_3 = []
    var_4 = []
    var_5 = 'test_dir'
    var_6 = 'skip_dir'
    var_7 = [var_6]
    var_8 = []
    var_9 = (var_5, var_7, var_8)
    var_10 = [var_5]
    var_11 = bool(var_3 == ['skip_dir'])
    assert var_11 is True
    var_12 = bool(var_4 == [])
    assert var_12 is True

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'broken_path'
    var_3 = [var_2]
    var_4 = bool(var_0 == [])
    assert var_4 is True
    var_5 = bool(var_1 == ['broken_path'])
    assert var_5 is True

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'file.py'
    var_3 = [var_2]
    var_4 = bool(var_0 == [])
    assert var_4 is True
    var_5 = bool(var_1 == [])
    assert var_5 is True

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = 'test_dir'
    var_4 = []
    var_5 = 'file.txt'
    var_6 = [var_5]
    var_7 = (var_3, var_4, var_6)
    var_8 = [var_3]
    var_9 = bool(var_1 == [])
    assert var_9 is True
    var_10 = bool(var_2 == [])
    assert var_10 is True

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = []
    var_3 = []
    var_4 = 'test_dir'
    var_5 = 'subdir'
    var_6 = [var_5]
    var_7 = 'file.py'
    var_8 = [var_7]
    var_9 = (var_4, var_6, var_8)
    var_10 = 'test_dir'
    var_11 = [var_10]
    var_12 = list(var_6)
    var_13 = 'test_dir'
    var_14 = 'file.py'
    var_15 = [var_8]
    var_16 = bool(var_12 == var_15)
    assert var_16 is True
    var_17 = bool(var_2 == [])
    assert var_17 is True
    var_18 = bool(var_3 == [])
    assert var_18 is True

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = []
    var_3 = []
    var_4 = 'dir1'
    var_5 = []
    var_6 = 'f1.py'
    var_7 = [var_6]
    var_8 = (var_4, var_5, var_7)
    var_9 = 'file2.py'
    var_10 = 'broken'
    var_11 = [var_4, var_9, var_10]
    var_12 = 'dir1'
    var_13 = 'f1.py'
    var_14 = 'file2.py'
    var_15 = [var_8, var_14]
    var_16 = bool(var_2 == [])
    assert var_16 is True
    var_17 = bool(var_3 == ['broken'])
    assert var_17 is True



