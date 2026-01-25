####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_find_with_directory_containing_python_files. Retrieved 18/29 statements.
# Partially parsed test_find_with_single_file. Retrieved 18/27 statements.
# Partially parsed test_find_with_nonexistent_path. Retrieved 17/21 statements.
# Partially parsed test_find_with_skipped_files. Retrieved 18/26 statements.
# Partially parsed test_find_with_unsupported_filetype. Retrieved 18/30 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 18/28 statements.
# Partially parsed test_find_with_multiple_paths. Retrieved 22/37 statements.


def test_case_0():
    var_0 = 'Test find function with a directory containing Python files.'
    var_1 = 'test.py'
    var_2 = "print('hello')"
    var_3 = 'Config'
    var_4 = ()
    var_5 = 'follow_links'
    var_6 = 'is_skipped'
    var_7 = 'is_supported_filetype'
    var_8 = False
    var_9 = lambda self, path: var_8
    var_10 = '.py'
    var_11 = lambda self, path: path.endswith(var_10)
    var_12 = {var_5: var_8, var_6: var_9, var_7: var_11}
    var_13 = type(var_3, var_4, var_12)
    var_14 = []
    var_15 = []
    var_16 = len(var_14)
    assert var_16 == 0
    var_17 = len(var_15)
    assert var_17 == 0

def test_case_0():
    var_0 = 'Test find function with a single file path.'
    var_1 = 'single.py'
    var_2 = 'x = 1'
    var_3 = 'Config'
    var_4 = ()
    var_5 = 'follow_links'
    var_6 = 'is_skipped'
    var_7 = 'is_supported_filetype'
    var_8 = False
    var_9 = lambda self, path: var_8
    var_10 = '.py'
    var_11 = lambda self, path: path.endswith(var_10)
    var_12 = {var_5: var_8, var_6: var_9, var_7: var_11}
    var_13 = type(var_3, var_4, var_12)
    var_14 = []
    var_15 = []
    var_16 = len(var_14)
    assert var_16 == 0
    var_17 = len(var_15)
    assert var_17 == 0

def test_case_0():
    var_0 = 'Test find function with a path that does not exist.'
    var_1 = 'Config'
    var_2 = ()
    var_3 = 'follow_links'
    var_4 = 'is_skipped'
    var_5 = 'is_supported_filetype'
    var_6 = False
    var_7 = lambda self, path: var_6
    var_8 = '.py'
    var_9 = lambda self, path: path.endswith(var_8)
    var_10 = {var_3: var_6, var_4: var_7, var_5: var_9}
    var_11 = type(var_1, var_2, var_10)
    var_12 = []
    var_13 = []
    var_14 = '/nonexistent/path.py'
    var_15 = [var_14]
    var_16 = len(var_13)
    assert var_16 == 1

def test_case_0():
    var_0 = 'Test find function skips files marked as skipped.'
    var_1 = 'skip_me.py'
    var_2 = 'x = 1'
    var_3 = 'Config'
    var_4 = ()
    var_5 = 'follow_links'
    var_6 = 'is_skipped'
    var_7 = 'is_supported_filetype'
    var_8 = False
    var_9 = 'skip_me'
    var_10 = lambda self, path: var_9 in str(path)
    var_11 = '.py'
    var_12 = lambda self, path: path.endswith(var_11)
    var_13 = {var_5: var_8, var_6: var_10, var_7: var_12}
    var_14 = type(var_3, var_4, var_13)
    var_15 = []
    var_16 = []
    var_17 = len(var_15)
    assert var_17 == 1

def test_case_0():
    var_0 = 'Test find function ignores unsupported file types.'
    var_1 = 'test.txt'
    var_2 = 'hello'
    var_3 = 'test.py'
    var_4 = 'x = 1'
    var_5 = 'Config'
    var_6 = ()
    var_7 = 'follow_links'
    var_8 = 'is_skipped'
    var_9 = 'is_supported_filetype'
    var_10 = False
    var_11 = lambda self, path: var_10
    var_12 = '.py'
    var_13 = lambda self, path: path.endswith(var_12)
    var_14 = {var_7: var_10, var_8: var_11, var_9: var_13}
    var_15 = type(var_5, var_6, var_14)
    var_16 = []
    var_17 = []

def test_case_0():
    var_0 = 'Test find function skips directories marked as skipped.'
    var_1 = 'skip_dir'
    var_2 = 'test.py'
    var_3 = 'x = 1'
    var_4 = 'Config'
    var_5 = ()
    var_6 = 'follow_links'
    var_7 = 'is_skipped'
    var_8 = 'is_supported_filetype'
    var_9 = False
    var_10 = lambda self, path: var_1 in str(path)
    var_11 = '.py'
    var_12 = lambda self, path: path.endswith(var_11)
    var_13 = {var_6: var_9, var_7: var_10, var_8: var_12}
    var_14 = type(var_4, var_5, var_13)
    var_15 = []
    var_16 = []
    var_17 = len(var_15)
    assert var_17 == 1

def test_case_0():
    var_0 = 'Test find function with multiple input paths.'
    var_1 = 'dir1'
    var_2 = 'test1.py'
    var_3 = 'x = 1'
    var_4 = 'dir2'
    var_5 = 'test2.py'
    var_6 = 'y = 2'
    var_7 = 'Config'
    var_8 = ()
    var_9 = 'follow_links'
    var_10 = 'is_skipped'
    var_11 = 'is_supported_filetype'
    var_12 = False
    var_13 = lambda self, path: var_12
    var_14 = '.py'
    var_15 = lambda self, path: path.endswith(var_14)
    var_16 = {var_9: var_12, var_10: var_13, var_11: var_15}
    var_17 = type(var_7, var_8, var_16)
    var_18 = []
    var_19 = []
    var_20 = len(var_18)
    assert var_20 == 0
    var_21 = len(var_19)
    assert var_21 == 0



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_find_with_python_files_in_directory. Retrieved 15/46 statements.
# Partially parsed test_find_with_skipped_files. Retrieved 9/29 statements.
# Partially parsed test_find_with_broken_path. Retrieved 6/15 statements.
# Partially parsed test_find_with_single_file. Retrieved 5/19 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 10/31 statements.
# Partially parsed test_find_with_unsupported_filetype. Retrieved 10/31 statements.
# Partially parsed test_find_with_multiple_paths. Retrieved 9/29 statements.


def test_case_0():
    var_0 = 'test_src'
    var_1 = 'file1.py'
    var_2 = '# python file 1'
    var_3 = 'file2.py'
    var_4 = '# python file 2'
    var_5 = 'file3.txt'
    var_6 = '# text file'
    var_7 = 'nested'
    var_8 = 'file4.py'
    var_9 = '# python file 4'
    var_10 = []
    var_11 = []
    var_12 = False
    var_13 = len(var_10)
    assert var_13 == 0
    var_14 = len(var_11)
    assert var_14 == 0

def test_case_0():
    var_0 = 'test_src'
    var_1 = 'file1.py'
    var_2 = '# python file 1'
    var_3 = 'file2.py'
    var_4 = '# python file 2'
    var_5 = []
    var_6 = []
    var_7 = False
    var_8 = len(var_5)
    assert var_8 == 1

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = False
    var_3 = '/nonexistent/path/to/file.py'
    var_4 = [var_3]
    var_5 = len(var_1)
    assert var_5 == 1

def test_case_0():
    var_0 = 'single_file.py'
    var_1 = '# single python file'
    var_2 = []
    var_3 = []
    var_4 = False

def test_case_0():
    var_0 = 'test_src'
    var_1 = 'file1.py'
    var_2 = '# python file 1'
    var_3 = 'skip_me'
    var_4 = 'file2.py'
    var_5 = '# python file 2'
    var_6 = []
    var_7 = []
    var_8 = False
    var_9 = len(var_6)
    assert var_9 == 1

def test_case_0():
    var_0 = 'test_src'
    var_1 = 'file1.py'
    var_2 = '# python file'
    var_3 = 'file2.txt'
    var_4 = '# text file'
    var_5 = 'file3.md'
    var_6 = '# markdown file'
    var_7 = []
    var_8 = []
    var_9 = False

def test_case_0():
    var_0 = 'dir1'
    var_1 = 'file1.py'
    var_2 = '# file 1'
    var_3 = 'dir2'
    var_4 = 'file2.py'
    var_5 = '# file 2'
    var_6 = []
    var_7 = []
    var_8 = False



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_predicate_line_7_evaluates_to_false. Retrieved 6/22 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 7 (os.path.isdir(path)) evaluates to False.'
    var_1 = False
    var_2 = 'test_file.txt'
    var_3 = 'test content'
    var_4 = []
    var_5 = []



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_find_evaluates_path_iteration_predicate. Retrieved 11/65 statements.


def test_case_0():
    var_0 = 'test.py'
    var_1 = '# test file'
    var_2 = 'test.txt'
    var_3 = '# not python'
    var_4 = []
    var_5 = []
    var_6 = 0
    var_7 = len(var_5)
    var_8 = var_7 == var_6
    var_9 = len(var_4)
    var_10 = var_9 == var_6



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_predicate_at_line_7_evaluates_to_false. Retrieved 6/20 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 7 (os.path.isdir(path)) evaluates to False.'
    var_1 = False
    var_2 = True
    var_3 = []
    var_4 = []
    var_5 = list(var_0)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_find_iterates_over_paths. Retrieved 10/21 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 7 (for path in paths:) evaluates to True by iterating over paths.'
    var_1 = False
    var_2 = 'path1'
    var_3 = 'path2'
    var_4 = 'path3'
    var_5 = [var_2, var_3, var_4]
    var_6 = []
    var_7 = []
    var_8 = list(var_0)
    var_9 = len(var_7)
    assert var_9 == 3



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_find_evaluates_path_iteration. Retrieved 13/28 statements.


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



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_predicate_at_line_7_evaluates_to_false. Retrieved 6/28 statements.


def test_case_0():
    var_0 = False
    var_1 = 'non_existent_file.py'
    var_2 = []
    var_3 = []
    var_4 = 'test.py'
    var_5 = '# test'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_find_yields_python_file_when_supported_and_not_skipped. Retrieved 7/23 statements.


def test_case_0():
    var_0 = "Test that find yields a filepath when it's a supported filetype and not skipped."
    var_1 = 'test.py'
    var_2 = "print('hello')"
    var_3 = True
    var_4 = False
    var_5 = []
    var_6 = []



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_predicate_line_7_evaluates_to_false. Retrieved 4/15 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 7 (os.path.isdir(path)) evaluates to False.'
    var_1 = 'non_existent_file.py'
    var_2 = []
    var_3 = []



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_find_with_python_files_in_directory. Retrieved 8/21 statements.
# Partially parsed test_find_skips_skipped_files. Retrieved 6/17 statements.
# Partially parsed test_find_with_nonexistent_path. Retrieved 6/10 statements.
# Partially parsed test_find_with_single_file. Retrieved 5/17 statements.
# Partially parsed test_find_skips_unsupported_filetypes. Retrieved 5/16 statements.
# Partially parsed test_find_with_nested_directories. Retrieved 8/23 statements.
# Partially parsed test_find_skips_directories. Retrieved 7/20 statements.
# Partially parsed test_find_with_multiple_paths. Retrieved 7/21 statements.


def test_case_0():
    var_0 = 'Test find yields Python files from a directory.'
    var_1 = 'test.py'
    var_2 = "print('hello')"
    var_3 = []
    var_4 = []
    var_5 = 0
    var_6 = len(var_3)
    assert var_6 == 0
    var_7 = len(var_4)
    assert var_7 == 0

def test_case_0():
    var_0 = 'Test find skips files marked as skipped.'
    var_1 = 'test.py'
    var_2 = "print('hello')"
    var_3 = []
    var_4 = []
    var_5 = len(var_3)
    assert var_5 == 1

def test_case_0():
    var_0 = 'Test find adds nonexistent paths to broken list.'
    var_1 = []
    var_2 = []
    var_3 = '/nonexistent/path'
    var_4 = [var_3]
    var_5 = len(var_2)
    assert var_5 == 1

def test_case_0():
    var_0 = 'Test find yields a single file when path is a file.'
    var_1 = 'test.py'
    var_2 = "print('hello')"
    var_3 = []
    var_4 = []

def test_case_0():
    var_0 = 'Test find skips files with unsupported types.'
    var_1 = 'test.py'
    var_2 = "print('hello')"
    var_3 = []
    var_4 = []

def test_case_0():
    var_0 = 'Test find recursively finds files in nested directories.'
    var_1 = 'subdir'
    var_2 = 'test1.py'
    var_3 = 'test2.py'
    var_4 = "print('hello')"
    var_5 = "print('world')"
    var_6 = []
    var_7 = []

def test_case_0():
    var_0 = 'Test find skips directories marked as skipped.'
    var_1 = 'skipped_dir'
    var_2 = 'test.py'
    var_3 = "print('hello')"
    var_4 = []
    var_5 = []
    var_6 = len(var_4)

def test_case_0():
    var_0 = 'Test find with multiple input paths.'
    var_1 = 'test1.py'
    var_2 = 'test2.py'
    var_3 = "print('1')"
    var_4 = "print('2')"
    var_5 = []
    var_6 = []



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_find_with_directory_containing_python_files. Retrieved 7/18 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 9/21 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 8/18 statements.
# Partially parsed test_find_with_unsupported_filetype. Retrieved 6/16 statements.
# Partially parsed test_find_with_broken_path. Retrieved 7/13 statements.
# Partially parsed test_find_with_single_file. Retrieved 7/19 statements.
# Partially parsed test_find_with_nested_directories. Retrieved 10/25 statements.
# Partially parsed test_find_with_multiple_paths. Retrieved 9/23 statements.


def test_case_0():
    var_0 = 'Test find yields Python files from a directory.'
    var_1 = 'test.py'
    var_2 = "print('hello')"
    var_3 = False
    var_4 = True
    var_5 = []
    var_6 = []

def test_case_0():
    var_0 = 'Test find skips directories marked as skipped.'
    var_1 = 'skip_me'
    var_2 = 'test.py'
    var_3 = "print('hello')"
    var_4 = lambda p: var_1 in str(p)
    var_5 = True
    var_6 = []
    var_7 = []
    var_8 = len(var_6)
    assert var_8 == 1

def test_case_0():
    var_0 = 'Test find skips files marked as skipped.'
    var_1 = 'test.py'
    var_2 = "print('hello')"
    var_3 = lambda p: str(p).endswith(var_1)
    var_4 = True
    var_5 = []
    var_6 = []
    var_7 = len(var_5)
    assert var_7 == 1

def test_case_0():
    var_0 = 'Test find ignores unsupported file types.'
    var_1 = 'test.txt'
    var_2 = 'hello'
    var_3 = False
    var_4 = []
    var_5 = []

def test_case_0():
    var_0 = 'Test find adds non-existent paths to broken list.'
    var_1 = False
    var_2 = True
    var_3 = []
    var_4 = []
    var_5 = '/nonexistent/path/to/file.py'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'Test find yields a single file when path is a file.'
    var_1 = 'test.py'
    var_2 = "print('hello')"
    var_3 = False
    var_4 = True
    var_5 = []
    var_6 = []

def test_case_0():
    var_0 = 'Test find recursively finds files in nested directories.'
    var_1 = 'subdir'
    var_2 = 'test1.py'
    var_3 = 'test2.py'
    var_4 = "print('1')"
    var_5 = "print('2')"
    var_6 = False
    var_7 = True
    var_8 = []
    var_9 = []

def test_case_0():
    var_0 = 'Test find processes multiple input paths.'
    var_1 = 'test1.py'
    var_2 = 'test2.py'
    var_3 = "print('1')"
    var_4 = "print('2')"
    var_5 = False
    var_6 = True
    var_7 = []
    var_8 = []



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_find_predicate_line_7_evaluates_to_true. Retrieved 9/32 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 7 (for path in paths:) evaluates to True with valid paths.'
    var_1 = 'test_source'
    var_2 = 'test.py'
    var_3 = '# test file'
    var_4 = []
    var_5 = []
    var_6 = 0
    var_7 = len(var_5)
    var_8 = var_7 == var_6



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_predicate_at_line_7_evaluates_to_false. Retrieved 6/19 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 7 (os.path.isdir(path)) evaluates to False.'
    var_1 = 'test_file.py'
    var_2 = '# test'
    var_3 = []
    var_4 = []
    var_5 = len(var_4)
    assert var_5 == 0



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_find_evaluates_predicate_at_line_7. Retrieved 5/29 statements.


def test_case_0():
    var_0 = 'test_project'
    var_1 = 'test.py'
    var_2 = '# test file'
    var_3 = []
    var_4 = []



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_find_with_empty_paths. Retrieved 14/19 statements.
# Partially parsed test_find_with_nonexistent_path. Retrieved 15/19 statements.
# Partially parsed test_find_with_single_file. Retrieved 14/23 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 15/26 statements.
# Partially parsed test_find_with_directory. Retrieved 18/31 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 18/31 statements.


def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'follow_links'
    var_3 = 'is_skipped'
    var_4 = 'is_supported_filetype'
    var_5 = False
    var_6 = lambda self, path: var_5
    var_7 = '.py'
    var_8 = lambda self, filepath: filepath.endswith(var_7)
    var_9 = {var_2: var_5, var_3: var_6, var_4: var_8}
    var_10 = type(var_0, var_1, var_9)
    var_11 = []
    var_12 = []
    var_13 = []

def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'follow_links'
    var_3 = 'is_skipped'
    var_4 = 'is_supported_filetype'
    var_5 = False
    var_6 = lambda self, path: var_5
    var_7 = '.py'
    var_8 = lambda self, filepath: filepath.endswith(var_7)
    var_9 = {var_2: var_5, var_3: var_6, var_4: var_8}
    var_10 = type(var_0, var_1, var_9)
    var_11 = []
    var_12 = []
    var_13 = '/nonexistent/path/to/file.py'
    var_14 = [var_13]

def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'follow_links'
    var_3 = 'is_skipped'
    var_4 = 'is_supported_filetype'
    var_5 = False
    var_6 = lambda self, path: var_5
    var_7 = '.py'
    var_8 = lambda self, filepath: filepath.endswith(var_7)
    var_9 = {var_2: var_5, var_3: var_6, var_4: var_8}
    var_10 = type(var_0, var_1, var_9)
    var_11 = []
    var_12 = []
    var_13 = list(var_1)

def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'follow_links'
    var_3 = 'is_skipped'
    var_4 = 'is_supported_filetype'
    var_5 = False
    var_6 = 'skip_me.py'
    var_7 = lambda self, path: str(path).endswith(var_6)
    var_8 = '.py'
    var_9 = lambda self, filepath: filepath.endswith(var_8)
    var_10 = {var_2: var_5, var_3: var_7, var_4: var_9}
    var_11 = type(var_0, var_1, var_10)
    var_12 = []
    var_13 = []
    var_14 = list(var_1)

def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'follow_links'
    var_3 = 'is_skipped'
    var_4 = 'is_supported_filetype'
    var_5 = False
    var_6 = lambda self, path: var_5
    var_7 = '.py'
    var_8 = lambda self, filepath: filepath.endswith(var_7)
    var_9 = {var_2: var_5, var_3: var_6, var_4: var_8}
    var_10 = type(var_0, var_1, var_9)
    var_11 = 'test.py'
    var_12 = '# test'
    var_13 = 'test.txt'
    var_14 = '# test'
    var_15 = []
    var_16 = []
    var_17 = list(var_3)

def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'follow_links'
    var_3 = 'is_skipped'
    var_4 = 'is_supported_filetype'
    var_5 = False
    var_6 = 'skip_dir'
    var_7 = lambda self, path: var_6 in str(path)
    var_8 = '.py'
    var_9 = lambda self, filepath: filepath.endswith(var_8)
    var_10 = {var_2: var_5, var_3: var_7, var_4: var_9}
    var_11 = type(var_0, var_1, var_10)
    var_12 = 'skip_dir'
    var_13 = 'test.py'
    var_14 = '# test'
    var_15 = []
    var_16 = []
    var_17 = list(var_4)



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_find_with_single_python_file. Retrieved 17/28 statements.
# Partially parsed test_find_with_directory_containing_python_files. Retrieved 19/30 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 17/28 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 16/25 statements.
# Partially parsed test_find_with_unsupported_filetype. Retrieved 15/24 statements.
# Partially parsed test_find_with_nonexistent_path. Retrieved 16/20 statements.
# Partially parsed test_find_with_nested_directories. Retrieved 19/32 statements.


def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = 'Config'
    var_3 = ()
    var_4 = 'follow_links'
    var_5 = 'is_skipped'
    var_6 = 'is_supported_filetype'
    var_7 = False
    var_8 = lambda self, path: var_7
    var_9 = '.py'
    var_10 = lambda self, path: path.endswith(var_9)
    var_11 = {var_4: var_7, var_5: var_8, var_6: var_10}
    var_12 = type(var_2, var_3, var_11)
    var_13 = []
    var_14 = []
    var_15 = len(var_13)
    assert var_15 == 0
    var_16 = len(var_14)
    assert var_16 == 0

def test_case_0():
    var_0 = 'test1.py'
    var_1 = "print('hello')"
    var_2 = 'test2.py'
    var_3 = "print('world')"
    var_4 = 'Config'
    var_5 = ()
    var_6 = 'follow_links'
    var_7 = 'is_skipped'
    var_8 = 'is_supported_filetype'
    var_9 = False
    var_10 = lambda self, path: var_9
    var_11 = '.py'
    var_12 = lambda self, path: path.endswith(var_11)
    var_13 = {var_6: var_9, var_7: var_10, var_8: var_12}
    var_14 = type(var_4, var_5, var_13)
    var_15 = []
    var_16 = []
    var_17 = len(var_15)
    assert var_17 == 0
    var_18 = len(var_16)
    assert var_18 == 0

def test_case_0():
    var_0 = 'skip_me'
    var_1 = 'test.py'
    var_2 = "print('hello')"
    var_3 = 'Config'
    var_4 = ()
    var_5 = 'follow_links'
    var_6 = 'is_skipped'
    var_7 = 'is_supported_filetype'
    var_8 = False
    var_9 = lambda self, path: var_0 in str(path)
    var_10 = '.py'
    var_11 = lambda self, path: path.endswith(var_10)
    var_12 = {var_5: var_8, var_6: var_9, var_7: var_11}
    var_13 = type(var_3, var_4, var_12)
    var_14 = []
    var_15 = []
    var_16 = len(var_14)
    assert var_16 == 1

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = 'Config'
    var_3 = ()
    var_4 = 'follow_links'
    var_5 = 'is_skipped'
    var_6 = 'is_supported_filetype'
    var_7 = False
    var_8 = lambda self, path: var_0 in str(path)
    var_9 = '.py'
    var_10 = lambda self, path: path.endswith(var_9)
    var_11 = {var_4: var_7, var_5: var_8, var_6: var_10}
    var_12 = type(var_2, var_3, var_11)
    var_13 = []
    var_14 = []
    var_15 = len(var_13)
    assert var_15 == 1

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'hello'
    var_2 = 'Config'
    var_3 = ()
    var_4 = 'follow_links'
    var_5 = 'is_skipped'
    var_6 = 'is_supported_filetype'
    var_7 = False
    var_8 = lambda self, path: var_7
    var_9 = '.py'
    var_10 = lambda self, path: path.endswith(var_9)
    var_11 = {var_4: var_7, var_5: var_8, var_6: var_10}
    var_12 = type(var_2, var_3, var_11)
    var_13 = []
    var_14 = []

def test_case_0():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'follow_links'
    var_3 = 'is_skipped'
    var_4 = 'is_supported_filetype'
    var_5 = False
    var_6 = lambda self, path: var_5
    var_7 = '.py'
    var_8 = lambda self, path: path.endswith(var_7)
    var_9 = {var_2: var_5, var_3: var_6, var_4: var_8}
    var_10 = type(var_0, var_1, var_9)
    var_11 = []
    var_12 = []
    var_13 = '/nonexistent/path.py'
    var_14 = [var_13]
    var_15 = len(var_12)
    assert var_15 == 1

def test_case_0():
    var_0 = 'subdir'
    var_1 = 'test1.py'
    var_2 = "print('hello')"
    var_3 = 'test2.py'
    var_4 = "print('world')"
    var_5 = 'Config'
    var_6 = ()
    var_7 = 'follow_links'
    var_8 = 'is_skipped'
    var_9 = 'is_supported_filetype'
    var_10 = False
    var_11 = lambda self, path: var_10
    var_12 = '.py'
    var_13 = lambda self, path: path.endswith(var_12)
    var_14 = {var_7: var_10, var_8: var_11, var_9: var_13}
    var_15 = type(var_5, var_6, var_14)
    var_16 = []
    var_17 = []
    var_18 = len(var_16)
    assert var_18 == 0



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_find_iterates_over_paths. Retrieved 4/27 statements.


def test_case_0():
    var_0 = 'test.py'
    var_1 = '# test file'
    var_2 = []
    var_3 = []



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_predicate_line_7_evaluates_to_false. Retrieved 8/23 statements.


def test_case_0():
    var_0 = False
    var_1 = 'test.py'
    var_2 = '# test'
    var_3 = 'os.path.isdir'
    var_4 = 'os.path.exists'
    var_5 = True
    var_6 = []
    var_7 = []



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_find_predicate_line_7_evaluates_to_true. Retrieved 4/26 statements.


def test_case_0():
    var_0 = 'test.py'
    var_1 = '# test file'
    var_2 = []
    var_3 = []



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_find_iterates_over_paths. Retrieved 5/26 statements.


def test_case_0():
    var_0 = False
    var_1 = 'test.py'
    var_2 = '# test'
    var_3 = []
    var_4 = []



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_find_with_directory_containing_python_files. Retrieved 7/25 statements.
# Partially parsed test_find_with_skipped_directory. Retrieved 6/24 statements.
# Partially parsed test_find_with_skipped_file. Retrieved 5/21 statements.
# Partially parsed test_find_with_non_python_files. Retrieved 5/20 statements.
# Partially parsed test_find_with_single_file. Retrieved 4/20 statements.
# Partially parsed test_find_with_broken_path. Retrieved 5/16 statements.
# Partially parsed test_find_with_nested_directories. Retrieved 7/26 statements.


def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = []
    var_3 = []
    var_4 = 0
    var_5 = len(var_2)
    assert var_5 == 0
    var_6 = len(var_3)
    assert var_6 == 0

def test_case_0():
    var_0 = 'skip_dir'
    var_1 = 'test.py'
    var_2 = "print('hello')"
    var_3 = []
    var_4 = []
    var_5 = len(var_3)
    assert var_5 == 1

def test_case_0():
    var_0 = 'skip_test.py'
    var_1 = "print('hello')"
    var_2 = []
    var_3 = []
    var_4 = len(var_2)
    assert var_4 == 1

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'hello'
    var_2 = []
    var_3 = []
    var_4 = len(var_2)
    assert var_4 == 0

def test_case_0():
    var_0 = 'test.py'
    var_1 = "print('hello')"
    var_2 = []
    var_3 = []

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '/nonexistent/path/file.py'
    var_3 = [var_2]
    var_4 = len(var_1)
    assert var_4 == 1

def test_case_0():
    var_0 = 'subdir'
    var_1 = 'test1.py'
    var_2 = "print('hello')"
    var_3 = 'test2.py'
    var_4 = "print('world')"
    var_5 = []
    var_6 = []



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_at_line_8_evaluates_to_false. Retrieved 3/20 statements.


import isort.settings as module_0

def test_case_0():
    var_0 = module_0.Config()
    var_1 = []
    var_2 = []



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_find_predicate_line_7_iterates_over_paths. Retrieved 4/26 statements.


def test_case_0():
    var_0 = 'test.py'
    var_1 = '# test file'
    var_2 = []
    var_3 = []



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_find_evaluates_path_iteration_predicate. Retrieved 12/44 statements.


def test_case_0():
    var_0 = 'test.py'
    var_1 = '# test'
    var_2 = 'subdir'
    var_3 = 'module.py'
    var_4 = '# module'
    var_5 = []
    var_6 = []
    var_7 = 'paths parameter should be Iterable'
    var_8 = 'find should return an iterator that can be converted to list'
    var_9 = 'Should find test.py'
    var_10 = 'Should find module.py in subdirectory'
    var_11 = len(var_6)
    assert var_11 == 0



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_find_predicate_line_7_false. Retrieved 2/22 statements.


def test_case_0():
    var_0 = []
    var_1 = []



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_find_predicate_line_7_evaluates_to_true. Retrieved 8/55 statements.


def test_case_0():
    var_0 = False
    var_1 = 'test_dir'
    var_2 = 'test.py'
    var_3 = "print('hello')"
    var_4 = []
    var_5 = []
    var_6 = 0
    var_7 = '.py'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_find_predicate_line_7_evaluates_to_false. Retrieved 7/26 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 7 (os.path.isdir(path)) evaluates to False.'
    var_1 = 'test.py'
    var_2 = '# test file'
    var_3 = []
    var_4 = []
    var_5 = len(var_4)
    assert var_5 == 0
    var_6 = len(var_3)
    assert var_6 == 0



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_find_yields_python_file_when_supported_and_not_skipped. Retrieved 6/28 statements.


def test_case_0():
    var_0 = 'test.py'
    var_1 = '# test file'
    var_2 = []
    var_3 = []
    var_4 = len(var_2)
    assert var_4 == 0
    var_5 = len(var_3)
    assert var_5 == 0



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_find_predicate_line_7_evaluates_to_false. Retrieved 5/22 statements.


def test_case_0():
    var_0 = False
    var_1 = 'test.py'
    var_2 = '# test'
    var_3 = []
    var_4 = []



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_find_iterates_over_paths. Retrieved 5/23 statements.


def test_case_0():
    var_0 = False
    var_1 = 'test.py'
    var_2 = '# test'
    var_3 = []
    var_4 = []



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_predicate_at_line_7_evaluates_to_false. Retrieved 4/19 statements.


def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []
    var_3 = len(var_2)
    assert var_3 == 0



