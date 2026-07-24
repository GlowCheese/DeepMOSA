####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'file.py'
    var_3 = [var_2]
    var_4 = 'non_existent'
    var_5 = [var_4]
    var_6 = []
    var_7 = []
    var_8 = 'src'
    var_9 = True
    var_10 = 'subdir'
    var_11 = [var_10]
    var_12 = 'module.py'
    var_13 = [var_12]
    var_14 = (var_8, var_11, var_13)
    var_15 = 'src/subdir'
    var_16 = []
    var_17 = 'inner.py'
    var_18 = [var_17]
    var_19 = (var_15, var_16, var_18)
    var_20 = [var_8]
    var_21 = False
    var_22 = 'src/module.txt'
    var_23 = []
    var_24 = 'module.txt'
    var_25 = [var_12, var_24]
    var_26 = (var_8, var_23, var_25)
    var_27 = [var_8]
    var_28 = 'src/module.py'
    var_29 = [var_8]



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'test_root'
    var_1 = 'file1.py'
    var_2 = 'content'
    var_3 = 'file2.py'
    var_4 = 'subdir'
    var_5 = 'file3.py'
    var_6 = 'skipped_dir'
    var_7 = 'file4.py'
    var_8 = 'non_existent_path'
    var_9 = []
    var_10 = []
    var_11 = []
    var_12 = []
    var_13 = 'test.txt'
    var_14 = 'text'
    var_15 = []
    var_16 = []



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'root'
    var_1 = 'main.py'
    var_2 = 'print(1)'
    var_3 = 'sub'
    var_4 = 'module.py'
    var_5 = 'print(2)'
    var_6 = 'skipped_dir'
    var_7 = 'ignored.py'
    var_8 = 'print(3)'
    var_9 = 'unsupported.txt'
    var_10 = 'not python'
    var_11 = '.py'
    var_12 = 'non_existent_path'
    var_13 = []
    var_14 = []

def test_case_0():
    var_0 = 'single.py'
    var_1 = 'pass'
    var_2 = []
    var_3 = []
    var_4 = len(var_2)
    assert var_4 == 0
    var_5 = len(var_3)
    assert var_5 == 0



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '/non/existent/path'
    var_3 = [var_2]
    var_4 = '/existing/file.py'
    var_5 = [var_4]
    var_6 = '/root'
    var_7 = 'subdir'
    var_8 = 'skipped_dir'
    var_9 = [var_7, var_8]
    var_10 = 'a.py'
    var_11 = [var_10]
    var_12 = (var_6, var_9, var_11)
    var_13 = '/root/subdir'
    var_14 = []
    var_15 = 'b.py'
    var_16 = [var_15]
    var_17 = (var_13, var_14, var_16)
    var_18 = '/root/skipped_dir'
    var_19 = []
    var_20 = 'c.py'
    var_21 = [var_20]
    var_22 = (var_18, var_19, var_21)
    var_23 = '/root'
    var_24 = [var_23]
    var_25 = list(var_7)
    var_26 = [os.path.basename(r) for r in var_25]
    var_27 = [os.path.basename(r) for r in var_25]
    var_28 = [os.path.basename(r) for r in var_25]
    var_29 = 'skipped_dir'
    var_30 = any(var_12)
    var_31 = '/root'
    var_32 = []
    var_33 = 'a.py'
    var_34 = 'ignore.txt'
    var_35 = [var_33, var_34]
    var_36 = (var_31, var_32, var_35)
    var_37 = '.txt'
    var_38 = [var_31]
    var_39 = list(var_30)
    var_40 = len(var_39)
    assert var_40 == 1
    var_41 = 0
    var_42 = var_39[var_41]



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'root'
    var_1 = 'file1.py'
    var_2 = 'print(1)'
    var_3 = 'file2.py'
    var_4 = 'print(2)'
    var_5 = 'subdir'
    var_6 = 'file3.py'
    var_7 = 'print(3)'
    var_8 = 'skipped_dir'
    var_9 = 'file4.py'
    var_10 = 'print(4)'
    var_11 = 'skipped_file.py'
    var_12 = 'print(5)'
    var_13 = 'notes.txt'
    var_14 = 'not python'
    var_15 = 'does_not_exist.py'
    var_16 = '.py'
    var_17 = []
    var_18 = []



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'root'
    var_1 = 'file1.py'
    var_2 = 'print(1)'
    var_3 = 'dir1'
    var_4 = 'file2.py'
    var_5 = 'print(2)'
    var_6 = 'skipped_dir'
    var_7 = 'file3.py'
    var_8 = 'print(3)'
    var_9 = 'dir2'



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'root'
    var_1 = 'file1.py'
    var_2 = 'print(1)'
    var_3 = 'dir1'
    var_4 = 'file2.py'
    var_5 = 'print(2)'
    var_6 = 'skipped_dir'
    var_7 = 'file3.py'
    var_8 = 'print(3)'
    var_9 = 'dir2'
    var_10 = 'file4.txt'
    var_11 = 'not python'
    var_12 = 'non_existent_path'
    var_13 = '.py'
    var_14 = []
    var_15 = []
    var_16 = [os.path.abspath(s) for s in var_14]



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'root'
    var_1 = 'file1.py'
    var_2 = 'print(1)'
    var_3 = 'subdir'
    var_4 = 'file2.py'
    var_5 = 'print(2)'
    var_6 = 'skipped_dir'
    var_7 = 'file3.py'
    var_8 = 'print(3)'
    var_9 = 'ignored.txt'
    var_10 = 'hello'
    var_11 = []
    var_12 = []
    var_13 = 'non_existent_path'

def test_case_0():
    var_0 = 'standalone.py'
    var_1 = 'pass'
    var_2 = []
    var_3 = []
    var_4 = len(var_2)
    assert var_4 == 0
    var_5 = len(var_3)
    assert var_5 == 0

def test_case_0():
    var_0 = 'base'
    var_1 = 'target'
    var_2 = 'target_file.py'
    var_3 = 'link_dir'
    var_4 = 'link_to_target'
    var_5 = []
    var_6 = []



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'test_root'
    var_1 = 'dir_a'
    var_2 = 'file_a.py'
    var_3 = "print('a')"
    var_4 = 'dir_b'
    var_5 = 'file_b.py'
    var_6 = "print('b')"
    var_7 = 'skipped_dir'
    var_8 = 'skipped.py'
    var_9 = "print('skip')"
    var_10 = 'non_existent'
    var_11 = 'standalone.py'
    var_12 = "print('standalone')"
    var_13 = []
    var_14 = []
    var_15 = 'test.txt'
    var_16 = 'text'
    var_17 = '.py'
    var_18 = []
    var_19 = []



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'src'
    var_1 = 'file1.py'
    var_2 = 'print(1)'
    var_3 = 'sub'
    var_4 = 'file2.py'
    var_5 = 'print(2)'
    var_6 = 'skipped_dir'
    var_7 = 'file3.py'
    var_8 = 'print(3)'
    var_9 = 'readme.txt'
    var_10 = 'text'
    var_11 = 'non_existent_path'
    var_12 = []
    var_13 = []
    var_14 = []
    var_15 = []

def test_case_0():
    var_0 = 'root'
    var_1 = 'target'
    var_2 = 'target.py'
    var_3 = 'link_to_target'
    var_4 = []
    var_5 = []



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'root'
    var_1 = 'file1.py'
    var_2 = 'print(1)'
    var_3 = 'file2.txt'
    var_4 = 'test'
    var_5 = 'subdir'
    var_6 = 'file3.py'
    var_7 = 'print(3)'
    var_8 = 'skipped_dir'
    var_9 = 'file4.py'
    var_10 = 'print(4)'
    var_11 = 'non_existent_path'
    var_12 = '.py'
    var_13 = []
    var_14 = []

def test_case_0():
    var_0 = 'single.py'
    var_1 = 'content'
    var_2 = []
    var_3 = []
    var_4 = len(var_2)
    assert var_4 == 0
    var_5 = len(var_3)
    assert var_5 == 0

def test_case_0():
    var_0 = 'root'
    var_1 = 'subdir'
    var_2 = 'sub.py'
    var_3 = 'content'
    var_4 = []
    var_5 = []



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'root'
    var_1 = 'file1.py'
    var_2 = 'print(1)'
    var_3 = 'skipped.py'
    var_4 = 'print(2)'
    var_5 = 'readme.txt'
    var_6 = 'text'
    var_7 = 'subdir'
    var_8 = 'file4.py'
    var_9 = 'print(4)'
    var_10 = 'non_existent_path'
    var_11 = []
    var_12 = []
    var_13 = 'skip_me'
    var_14 = 'hidden.py'
    var_15 = 'hidden'
    var_16 = []
    var_17 = []



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'root'
    var_1 = 'file1.py'
    var_2 = 'print(1)'
    var_3 = 'file2.txt'
    var_4 = 'hello'
    var_5 = 'subdir'
    var_6 = 'file3.py'
    var_7 = 'print(2)'
    var_8 = 'skipped_dir'
    var_9 = 'file4.py'
    var_10 = 'print(3)'
    var_11 = '.py'
    var_12 = 'skipped_directories'
    var_13 = 'non_existent_path'
    var_14 = []
    var_15 = []



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'project'
    var_1 = 'file1.py'
    var_2 = 'print(1)'
    var_3 = 'subdir'
    var_4 = 'file2.py'
    var_5 = 'print(2)'
    var_6 = 'skipped_dir'
    var_7 = 'file3.py'
    var_8 = 'print(3)'
    var_9 = 'non_existent_path'
    var_10 = []
    var_11 = []

def test_case_0():
    var_0 = 'standalone.py'
    var_1 = 'pass'
    var_2 = []
    var_3 = []
    var_4 = 0

def test_case_0():
    var_0 = '.py'
    var_1 = 'project'
    var_2 = 'test.py'
    var_3 = 'pass'
    var_4 = 'test.txt'
    var_5 = []
    var_6 = []
    var_7 = 0



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = '/valid/file.py'
    var_1 = '/valid/dir'
    var_2 = '/non/existent/path'
    var_3 = [var_0, var_1, var_2]
    var_4 = []
    var_5 = []
    var_6 = '/valid/dir'
    var_7 = 'sub'
    var_8 = 'skip_me'
    var_9 = [var_7, var_8]
    var_10 = 'file1.py'
    var_11 = [var_10]
    var_12 = (var_6, var_9, var_11)
    var_13 = '/valid/dir/sub'
    var_14 = []
    var_15 = 'file2.py'
    var_16 = [var_15]
    var_17 = (var_13, var_14, var_16)
    var_18 = '/valid/dir/skip_me'

def test_case_0():
    var_0 = '/invalid/path'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = list(var_0)

def test_case_0():
    var_0 = '/single/file.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = list(var_0)



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = '/root'
    var_1 = '/non_existent'
    var_2 = [var_0, var_1]
    var_3 = '/root/file1.py'
    var_4 = '/root/subdir/file2.py'
    var_5 = 'subdir'
    var_6 = 'skipped_dir'
    var_7 = [var_5, var_6]
    var_8 = 'file1.py'
    var_9 = [var_8]
    var_10 = (var_0, var_7, var_9)
    var_11 = '/root/subdir'
    var_12 = []
    var_13 = 'file2.py'
    var_14 = [var_13]
    var_15 = (var_11, var_12, var_14)
    var_16 = []
    var_17 = []



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = '/tmp/test_file.py'
    var_1 = '/tmp/test_dir'
    var_2 = '/tmp/does_not_exist'
    var_3 = '/tmp/skipped.py'
    var_4 = 'skipped_subdir'
    var_5 = 'normal_subdir'
    var_6 = [var_4, var_5]
    var_7 = 'file1.py'
    var_8 = 'file2.py'
    var_9 = [var_7, var_8]
    var_10 = (var_1, var_6, var_9)
    var_11 = []
    var_12 = 'file3.py'
    var_13 = [var_12]
    var_14 = '.py'
    var_15 = [var_0, var_1, var_2]
    var_16 = []
    var_17 = []



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'test_root'
    var_1 = 'valid.py'
    var_2 = "print('hello')"
    var_3 = 'skipped.py'
    var_4 = "print('skip me')"
    var_5 = 'unsupported.txt'
    var_6 = 'text content'
    var_7 = 'ignored_dir'
    var_8 = 'hidden.py'
    var_9 = 'hidden'
    var_10 = 'non_existent_path'
    var_11 = []
    var_12 = []
    var_13 = []
    var_14 = []



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'file.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = 'non_existent.py'
    var_5 = [var_4]
    var_6 = '/root'
    var_7 = 'subdir'
    var_8 = [var_7]
    var_9 = 'file1.py'
    var_10 = 'ignored.txt'
    var_11 = [var_9, var_10]
    var_12 = (var_6, var_8, var_11)
    var_13 = '/root/subdir'
    var_14 = []
    var_15 = 'file2.py'
    var_16 = [var_15]
    var_17 = (var_13, var_14, var_16)
    var_18 = [var_6]
    var_19 = []
    var_20 = []
    var_21 = 'skip_me'
    var_22 = [var_21]
    var_23 = [var_9]
    var_24 = (var_6, var_22, var_23)
    var_25 = '/root/skip_me'
    var_26 = []
    var_27 = [var_15]
    var_28 = (var_25, var_26, var_27)
    var_29 = []
    var_30 = [var_6]
    var_31 = []



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = '/root'
    var_1 = '/non_existent'
    var_2 = [var_0, var_1]
    var_3 = []
    var_4 = []
    var_5 = '/root'
    var_6 = 'subdir'
    var_7 = 'skipped_dir'
    var_8 = [var_6, var_7]
    var_9 = 'file1.py'
    var_10 = 'skipped_file.py'
    var_11 = [var_9, var_10]
    var_12 = (var_5, var_8, var_11)
    var_13 = '/root/subdir'
    var_14 = []
    var_15 = 'file2.py'
    var_16 = [var_15]
    var_17 = (var_13, var_14, var_16)
    var_18 = '/root/skipped_dir'
    var_19 = []
    var_20 = []
    var_21 = (var_18, var_19, var_20)



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'file.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = 'non_existent.py'
    var_5 = [var_4]
    var_6 = []
    var_7 = []
    var_8 = 'src'
    var_9 = [var_8]
    var_10 = []
    var_11 = []
    var_12 = 'venv'
    var_13 = 'utils'
    var_14 = [var_12, var_13]
    var_15 = 'main.py'
    var_16 = [var_15]
    var_17 = (var_8, var_14, var_16)
    var_18 = 'src/utils'
    var_19 = []
    var_20 = 'helper.py'
    var_21 = [var_20]
    var_22 = (var_18, var_19, var_21)
    var_23 = [var_12, var_13]
    var_24 = [var_15]
    var_25 = (var_8, var_23, var_24)
    var_26 = []
    var_27 = [var_20]
    var_28 = (var_18, var_26, var_27)



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'root'
    var_1 = 'file1.py'
    var_2 = 'print(1)'
    var_3 = 'subdir'
    var_4 = 'file2.py'
    var_5 = 'print(2)'
    var_6 = 'skipped_dir'
    var_7 = 'file3.py'
    var_8 = 'print(3)'
    var_9 = 'skipped_file.py'
    var_10 = 'print(4)'
    var_11 = 'non_existent_path_12345'
    var_12 = []
    var_13 = []

def test_case_0():
    var_0 = 'single.py'
    var_1 = 'content'
    var_2 = []
    var_3 = []
    var_4 = len(var_2)
    assert var_4 == 0
    var_5 = len(var_3)
    assert var_5 == 0

def test_case_0():
    var_0 = 'root'
    var_1 = 'subdir'
    var_2 = 'file.py'
    var_3 = 'content'
    var_4 = 'symlink_dir'
    var_5 = []
    var_6 = []
    var_7 = 'file.py'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = '/tmp/test_dir'
    var_1 = 'a.py'
    var_2 = 'sub'
    var_3 = 'b.py'
    var_4 = 'skipped.py'
    var_5 = '/tmp/non_existent'
    var_6 = 'sub'
    var_7 = [var_6]
    var_8 = 'a.py'
    var_9 = 'skipped.py'
    var_10 = [var_8, var_9]
    var_11 = (var_0, var_7, var_10)
    var_12 = str(var_4)
    var_13 = []
    var_14 = 'b.py'
    var_15 = [var_14]
    var_16 = (var_12, var_13, var_15)
    var_17 = []
    var_18 = []



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'root'
    var_1 = 'file1.py'
    var_2 = 'content'
    var_3 = 'file2.py'
    var_4 = 'subdir'
    var_5 = 'file3.py'
    var_6 = 'skipped_dir'
    var_7 = 'file4.py'
    var_8 = 'ignored.txt'
    var_9 = 'non_existent'
    var_10 = '.py'
    var_11 = []
    var_12 = []

def test_case_0():
    var_0 = 'standalone.py'
    var_1 = 'content'
    var_2 = []
    var_3 = []
    var_4 = len(var_2)
    assert var_4 == 0
    var_5 = len(var_3)
    assert var_5 == 0

def test_case_0():
    var_0 = 'ghost'
    var_1 = []
    var_2 = []



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'test_root'
    var_1 = 'subdir'
    var_2 = 'file1.py'
    var_3 = 'file2.py'
    var_4 = 'skipped_dir'
    var_5 = 'skipped.py'
    var_6 = 'non_existent_path'
    var_7 = True
    var_8 = 'print(1)'
    var_9 = 'print(2)'
    var_10 = 'print(3)'
    var_11 = str(var_6)
    var_12 = []
    var_13 = []
    var_14 = list(var_0)
    var_15 = 'skipped_dir'
    var_16 = any(var_3)
    var_17 = 'test.txt'
    var_18 = 'text'
    var_19 = [var_8]
    var_20 = list(var_9)



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'root'
    var_1 = 'file1.py'
    var_2 = 'print(1)'
    var_3 = 'subdir'
    var_4 = 'file2.py'
    var_5 = 'print(2)'
    var_6 = 'skipped_dir'
    var_7 = 'file3.py'
    var_8 = 'print(3)'
    var_9 = 'skipped_file.py'
    var_10 = 'print(4)'
    var_11 = 'non_existent_path'
    var_12 = []
    var_13 = []

def test_case_0():
    var_0 = 'standalone.py'
    var_1 = ''
    var_2 = []
    var_3 = []
    var_4 = len(var_2)
    assert var_4 == 0
    var_5 = len(var_3)
    assert var_5 == 0



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'tmp_test_root'
    var_1 = 'src'
    var_2 = 'sub'
    var_3 = 'skipped_dir'
    var_4 = True
    var_5 = 'file1.py'
    var_6 = 'file2.py'
    var_7 = 'file3.py'
    var_8 = 'print(1)'
    var_9 = 'print(2)'
    var_10 = 'print(3)'
    var_11 = 'non_existent_path'
    var_12 = []
    var_13 = []

def test_case_0():
    var_0 = 'single_test.py'
    var_1 = 'test'
    var_2 = []
    var_3 = []
    var_4 = len(var_3)
    assert var_4 == 0

def test_case_0():
    var_0 = 'skipped_file.py'
    var_1 = 'test'
    var_2 = []
    var_3 = []



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'test_root'
    var_1 = 'valid.py'
    var_2 = "print('hello')"
    var_3 = 'skipped.py'
    var_4 = "print('skip me')"
    var_5 = 'readme.txt'
    var_6 = 'text content'
    var_7 = '.py'
    var_8 = 'skipped_dir'
    var_9 = 'inside_skipped.py'
    var_10 = "print('hidden')"
    var_11 = 'does_not_exist.py'
    var_12 = []
    var_13 = []



