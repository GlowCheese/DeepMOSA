####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '/mock/dir'
    var_3 = '/mock/file.py'
    var_4 = '/non/existent'
    var_5 = [var_2, var_3, var_4]
    var_6 = '/mock/dir'
    var_7 = 'sub'
    var_8 = 'skip_me'
    var_9 = [var_7, var_8]
    var_10 = []
    var_11 = (var_6, var_9, var_10)
    var_12 = '/mock/dir/sub'
    var_13 = []
    var_14 = 'target.py'
    var_15 = [var_14]
    var_16 = (var_12, var_13, var_15)
    var_17 = '/mock/dir/skip_me'
    var_18 = []
    var_19 = 'hidden.py'
    var_20 = [var_19]
    var_21 = (var_17, var_18, var_20)



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'file.py'
    var_3 = [var_2]
    var_4 = 'nonexistent.py'
    var_5 = [var_4]
    var_6 = '/root'
    var_7 = 'skipped_dir'
    var_8 = 'subdir'
    var_9 = [var_7, var_8]
    var_10 = 'file1.py'
    var_11 = 'file3.txt'
    var_12 = [var_10, var_11]
    var_13 = (var_6, var_9, var_12)
    var_14 = '/root/skipped_dir'
    var_15 = []
    var_16 = 'file2.py'
    var_17 = [var_16]
    var_18 = (var_14, var_15, var_17)
    var_19 = '/root/subdir'
    var_20 = []
    var_21 = 'file4.py'
    var_22 = [var_21]
    var_23 = (var_19, var_20, var_22)
    var_24 = [var_6]



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'project'
    var_3 = 'src'
    var_4 = 'main.py'
    var_5 = "print('hello')"
    var_6 = 'README.md'
    var_7 = 'docs'
    var_8 = 'venv'
    var_9 = 'lib.py'
    var_10 = ''
    var_11 = 'non_existent_path'
    var_12 = 'src'
    var_13 = [var_12]
    var_14 = 'README.md'
    var_15 = [var_14]
    var_16 = (var_2, var_13, var_15)
    var_17 = 'venv'
    var_18 = [var_17]
    var_19 = 'main.py'
    var_20 = [var_19, var_14]
    var_21 = []
    var_22 = 'lib.py'
    var_23 = [var_22]
    var_24 = (var_9, var_21, var_23)
    var_25 = list(var_11)

def test_case_0():
    var_0 = 'single.py'
    var_1 = ''
    var_2 = []
    var_3 = []

def test_case_0():
    var_0 = 'skip_test'
    var_1 = 'test.py'
    var_2 = ''
    var_3 = []
    var_4 = []



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = '/valid/file.py'
    var_1 = '/valid/dir'
    var_2 = '/non/existent'
    var_3 = '/skip/me.py'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = []
    var_6 = []
    var_7 = '/valid/dir'
    var_8 = 'sub_dir'
    var_9 = [var_8]
    var_10 = 'inner.py'
    var_11 = [var_10]
    var_12 = (var_7, var_9, var_11)



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = '/valid/file.py'
    var_1 = '/non/existent/path'
    var_2 = '/valid/dir'
    var_3 = [var_0, var_1, var_2]
    var_4 = []
    var_5 = []
    var_6 = '/valid/dir'
    var_7 = 'subdir'
    var_8 = [var_7]
    var_9 = 'file1.py'
    var_10 = [var_9]
    var_11 = (var_6, var_8, var_10)
    var_12 = '/valid/dir/subdir'
    var_13 = []
    var_14 = 'file2.py'
    var_15 = [var_14]
    var_16 = (var_12, var_13, var_15)



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'src'
    var_1 = 'file1.py'
    var_2 = 'print(1)'
    var_3 = 'subdir'
    var_4 = 'file2.py'
    var_5 = 'print(2)'
    var_6 = 'skipped_dir'
    var_7 = 'hidden.py'
    var_8 = 'print(3)'
    var_9 = 'file3.txt'
    var_10 = 'not python'
    var_11 = 'non_existent_path'
    var_12 = []
    var_13 = []

def test_case_0():
    var_0 = 'root'
    var_1 = 'file1.py'
    var_2 = 'content'
    var_3 = 'link_dir'
    var_4 = []
    var_5 = []



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'test_dir'
    var_3 = 'file1.py'
    var_4 = 'print(1)'
    var_5 = 'subdir'
    var_6 = 'file2.py'
    var_7 = 'print(2)'
    var_8 = 'skip_me'
    var_9 = 'file3.py'
    var_10 = 'print(3)'
    var_11 = 'ignore.txt'
    var_12 = 'ignore me'
    var_13 = 'non_existent_path'
    var_14 = '.txt'



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'test_root'
    var_1 = True
    var_2 = 'subdir'
    var_3 = 'file1.py'
    var_4 = 'file2.py'
    var_5 = 'print(1)'
    var_6 = 'print(2)'
    var_7 = 'skipped.py'
    var_8 = 'print(3)'
    var_9 = [var_0]
    var_10 = []
    var_11 = []
    var_12 = list(var_1)
    var_13 = len(var_12)
    assert var_13 == 2
    var_14 = 'file1.py'
    var_15 = 'file2.py'
    var_16 = any(var_5)
    var_17 = 'non_existent_path'
    var_18 = [var_17]
    var_19 = []
    var_20 = []
    var_21 = [var_8]
    var_22 = []
    var_23 = []
    var_24 = 'skipped.py'
    var_25 = []
    var_26 = []
    var_27 = len(var_12)
    assert var_27 == 1
    var_28 = 'subdir'
    var_29 = []
    var_30 = []
    var_31 = len(var_12)
    assert var_31 == 1



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'root'
    var_1 = 'root/subdir'
    var_2 = 'non_existent'
    var_3 = 'subdir'
    var_4 = 'skipped_dir'
    var_5 = [var_3, var_4]
    var_6 = 'file1.py'
    var_7 = [var_6]
    var_8 = (var_0, var_5, var_7)
    var_9 = []
    var_10 = 'file2.py'
    var_11 = [var_10]
    var_12 = (var_1, var_9, var_11)
    var_13 = 'root/skipped_dir'
    var_14 = []
    var_15 = 'file3.py'
    var_16 = [var_15]
    var_17 = (var_13, var_14, var_16)
    var_18 = 'root/file1.py'
    var_19 = [var_0, var_18, var_2]
    var_20 = []
    var_21 = []



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'root'
    var_1 = 'file1.py'
    var_2 = 'content'
    var_3 = 'file2.txt'
    var_4 = 'subdir'
    var_5 = 'file3.py'
    var_6 = 'skipped_dir'
    var_7 = 'file4.py'
    var_8 = 'non_existent_path'
    var_9 = 'single.py'
    var_10 = []
    var_11 = []



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'non_existent'
    var_3 = [var_2]
    var_4 = []
    var_5 = []
    var_6 = 'file.py'
    var_7 = [var_6]
    var_8 = 'root'
    var_9 = 'subdir'
    var_10 = [var_9]
    var_11 = 'a.py'
    var_12 = 'b.txt'
    var_13 = [var_11, var_12]
    var_14 = (var_8, var_10, var_13)
    var_15 = 'root/subdir'
    var_16 = []
    var_17 = 'c.py'
    var_18 = [var_17]
    var_19 = (var_15, var_16, var_18)
    var_20 = '.py'
    var_21 = 'root'
    var_22 = [var_21]
    var_23 = list(var_10)
    var_24 = []
    var_25 = 'root'
    var_26 = 'subdir'
    var_27 = [var_26]
    var_28 = 'a.py'
    var_29 = [var_28]
    var_30 = (var_25, var_27, var_29)
    var_31 = 'root/subdir'
    var_32 = []
    var_33 = 'c.py'
    var_34 = [var_33]
    var_35 = (var_31, var_32, var_34)
    var_36 = 'root/subdir'
    var_37 = 'root'
    var_38 = [var_37]
    var_39 = list(var_32)



# Parsed testcases at query #12
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
    var_9 = 'not_here'
    var_10 = []
    var_11 = []

def test_case_0():
    var_0 = 'root'
    var_1 = 'target'
    var_2 = 'target.py'
    var_3 = 'link_dir'
    var_4 = 'link_to_target'
    var_5 = []
    var_6 = []

def test_case_0():
    var_0 = '.py'
    var_1 = 'root'
    var_2 = 'test.py'
    var_3 = 'content'
    var_4 = 'test.txt'
    var_5 = []
    var_6 = []



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'file.py'
    var_3 = 'dir_path'
    var_4 = [var_2]
    var_5 = 'non_existent.py'
    var_6 = [var_5]
    var_7 = False
    var_8 = []
    var_9 = []
    var_10 = True
    var_11 = 'subdir'
    var_12 = [var_11]
    var_13 = 'script1.py'
    var_14 = 'readme.txt'
    var_15 = [var_13, var_14]
    var_16 = (var_3, var_12, var_15)
    var_17 = '.py'
    var_18 = [var_3]
    var_19 = []
    var_20 = []
    var_21 = 'dir_path'
    var_22 = 'sub1'
    var_23 = 'sub2'
    var_24 = [var_22, var_23]
    var_25 = 'file.py'
    var_26 = [var_25]
    var_27 = (var_21, var_24, var_26)
    var_28 = 'dir_path/sub1'
    var_29 = []
    var_30 = 'file2.py'
    var_31 = [var_30]
    var_32 = (var_28, var_29, var_31)
    var_33 = 'dir_path'
    var_34 = [var_33]
    var_35 = list(var_23)



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = '/tmp/valid_file.py'
    var_1 = '/tmp/valid_dir'
    var_2 = '/tmp/non_existent'
    var_3 = [var_0, var_1, var_2]
    var_4 = []
    var_5 = []
    var_6 = '/tmp/dummy'
    var_7 = '/tmp/valid_dir'
    var_8 = 'sub'
    var_9 = 'skipped_dir'
    var_10 = [var_8, var_9]
    var_11 = 'ignored.txt'
    var_12 = [var_11]
    var_13 = (var_7, var_10, var_12)
    var_14 = '/tmp/valid_dir/sub'
    var_15 = []
    var_16 = 'file.py'
    var_17 = [var_16]
    var_18 = (var_14, var_15, var_17)
    var_19 = '/tmp/valid_dir/skipped_dir'
    var_20 = []
    var_21 = []
    var_22 = (var_19, var_20, var_21)



# Parsed testcases at query #15
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
    var_11 = 'non_existent_folder'
    var_12 = []
    var_13 = []
    var_14 = '.py'



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = 'test_root'
    var_1 = 'src'
    var_2 = 'sub'
    var_3 = 'skipped_folder'
    var_4 = 'file1.py'
    var_5 = 'file2.py'
    var_6 = 'file3.py'
    var_7 = 'non_existent_path'
    var_8 = True
    var_9 = 'print(1)'
    var_10 = 'single.py'
    var_11 = str(var_4)
    var_12 = [var_0, var_1, var_2, var_11, var_7]
    var_13 = []
    var_14 = []
    var_15 = list(var_8)
    var_16 = any(var_9)
    var_17 = 'skipped_folder'

def test_case_0():
    var_0 = 'test_visited'
    var_1 = 'dir1'
    var_2 = 'dir2'
    var_3 = True
    var_4 = 'file1.py'
    var_5 = []
    var_6 = []
    var_7 = [var_0]
    var_8 = list(var_2)
    var_9 = len(var_8)



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = '/valid/file.py'
    var_1 = '/valid/dir'
    var_2 = '/nonexistent'
    var_3 = '/skipped/file.py'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = []
    var_6 = []
    var_7 = '/valid/dir'
    var_8 = 'subdir'
    var_9 = [var_8]
    var_10 = 'file1.py'
    var_11 = [var_10]
    var_12 = (var_7, var_9, var_11)
    var_13 = '/valid/dir/subdir'
    var_14 = []
    var_15 = 'file2.py'
    var_16 = [var_15]
    var_17 = (var_13, var_14, var_16)
    var_18 = '/skipped/file.py'
    var_19 = []
    var_20 = 'subdir'
    var_21 = [var_20]
    var_22 = 'file1.py'
    var_23 = [var_22]
    var_24 = (var_8, var_21, var_23)
    var_25 = [var_8]
    var_26 = list(var_17)



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'test.py'
    var_3 = [var_2]
    var_4 = 'non_existent.py'
    var_5 = [var_4]
    var_6 = 'root'
    assert var_6 is True
    var_7 = 'subdir'
    var_8 = 'skipped_dir'
    var_9 = [var_7, var_8]
    var_10 = 'file1.py'
    var_11 = 'skipped_file.py'
    var_12 = [var_10, var_11]
    var_13 = (var_6, var_9, var_12)
    var_14 = 'root/subdir'
    var_15 = []
    var_16 = 'file2.py'
    var_17 = [var_16]
    var_18 = (var_14, var_15, var_17)
    var_19 = 'root/skipped_dir'
    var_20 = []
    var_21 = 'hidden.py'
    var_22 = [var_21]
    var_23 = (var_19, var_20, var_22)
    var_24 = [var_6]



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'non_existent_path'
    var_3 = [var_2]
    var_4 = 'existing_file.py'
    var_5 = [var_4]
    var_6 = '/root'
    var_7 = 'sub_dir'
    var_8 = 'skip_me'
    var_9 = [var_7, var_8]
    var_10 = 'file1.py'
    var_11 = 'file2.txt'
    var_12 = [var_10, var_11]
    var_13 = (var_6, var_9, var_12)
    var_14 = '/root'
    var_15 = [var_14]
    var_16 = list(var_7)
    var_17 = 'file2.txt'
    var_18 = any(var_9)
    var_19 = 'skip_me'
    var_20 = any(var_12)
    var_21 = []
    var_22 = []
    var_23 = 'bad1'
    var_24 = 'bad2'
    var_25 = [var_23, var_24]
    var_26 = list(var_17)



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'file.py'
    var_3 = [var_2]
    var_4 = 'non_existent.py'
    var_5 = [var_4]
    var_6 = '/root'
    var_7 = 'skipped_dir'
    var_8 = [var_7]
    var_9 = 'a.py'
    var_10 = 'b.txt'
    var_11 = [var_9, var_10]
    var_12 = (var_6, var_8, var_11)
    var_13 = '/root/skipped_dir'
    var_14 = []
    var_15 = 'c.py'
    var_16 = [var_15]
    var_17 = (var_13, var_14, var_16)
    var_18 = [var_6]
    var_19 = []
    var_20 = 'missing1.py'
    var_21 = 'missing2.py'
    var_22 = [var_20, var_21]
    var_23 = list(var_8)
    var_24 = 'test.py'
    var_25 = [var_24]
    var_26 = []
    var_27 = []
    var_28 = list(var_23)



# Parsed testcases at query #21
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
    var_11 = 'unsupported.txt'
    var_12 = [var_10, var_11]
    var_13 = (var_6, var_9, var_12)
    var_14 = '/root/subdir'
    var_15 = []
    var_16 = 'b.py'
    var_17 = [var_16]
    var_18 = (var_14, var_15, var_17)
    var_19 = '/root/skipped_dir'
    var_20 = []
    var_21 = 'hidden.py'
    var_22 = [var_21]
    var_23 = (var_19, var_20, var_22)
    var_24 = [var_6]
    var_25 = 'pathlib'
    var_26 = '/root/dir1'
    var_27 = '/root/dir2'
    var_28 = '/root'
    var_29 = 'dir1'
    var_30 = 'dir2'
    var_31 = [var_29, var_30]
    var_32 = 'file1.py'
    var_33 = [var_32]
    var_34 = (var_28, var_31, var_33)
    var_35 = []
    var_36 = [var_32]
    var_37 = (var_26, var_35, var_36)
    var_38 = []
    var_39 = 'file2.py'
    var_40 = [var_39]
    var_41 = (var_27, var_38, var_40)
    var_42 = [var_28]



# Parsed testcases at query #22
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

def test_case_0():
    var_0 = 'standalone.py'
    var_1 = 'pass'
    var_2 = []
    var_3 = []

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'file.py'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = 'nonexistent.py'
    var_5 = [var_4]
    var_6 = []
    var_7 = []
    var_8 = '/root'
    var_9 = 'subdir'
    var_10 = [var_9]
    var_11 = 'file1.py'
    var_12 = 'ignore.txt'
    var_13 = [var_11, var_12]
    var_14 = (var_8, var_10, var_13)
    var_15 = '/root/subdir'
    var_16 = []
    var_17 = 'file2.py'
    var_18 = [var_17]
    var_19 = (var_15, var_16, var_18)
    var_20 = 'skip_me'
    var_21 = [var_20]
    var_22 = [var_11]
    var_23 = (var_8, var_21, var_22)
    var_24 = '/root/skip_me'
    var_25 = []
    var_26 = 'hidden.py'
    var_27 = [var_26]
    var_28 = (var_24, var_25, var_27)
    var_29 = []
    var_30 = [var_17]
    var_31 = (var_15, var_29, var_30)
    var_32 = [var_8]
    var_33 = []
    var_34 = []
    var_35 = '/root'
    var_36 = []
    var_37 = 'file1.py'
    var_38 = 'image.png'
    var_39 = [var_37, var_38]
    var_40 = (var_35, var_36, var_39)
    var_41 = '.py'
    var_42 = [var_35]
    var_43 = []
    var_44 = []
    var_45 = list(var_11)
    var_46 = len(var_45)
    assert var_46 == 1
    var_47 = 0
    var_48 = var_45[var_47]

def test_case_0():
    var_0 = '/dir1'
    var_1 = '/dir2'
    var_2 = '/dir1'
    var_3 = 'sub'
    var_4 = [var_3]
    var_5 = 'f1.py'
    var_6 = [var_5]
    var_7 = (var_2, var_4, var_6)
    var_8 = '/dir2'
    var_9 = [var_3]
    var_10 = 'f2.py'
    var_11 = [var_10]
    var_12 = (var_8, var_9, var_11)
    var_13 = [var_0, var_1]
    var_14 = []
    var_15 = []



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'test_root'
    var_1 = True
    var_2 = 'subdir'
    var_3 = 'file1.py'
    var_4 = 'file2.py'
    var_5 = 'print(1)'
    var_6 = 'print(2)'
    var_7 = []
    var_8 = []
    var_9 = [var_0]
    var_10 = [var_2]
    var_11 = [var_3]
    var_12 = list(var_5)
    var_13 = 'non_existent_path'
    var_14 = [var_13]
    var_15 = False
    var_16 = 'py'
    var_17 = 'notes.txt'
    var_18 = 'hello'



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'dir1'
    var_1 = 'file1.py'
    var_2 = 'non_existent'
    var_3 = [var_0, var_1, var_2]
    var_4 = []
    var_5 = []
    var_6 = 'dir1'
    var_7 = 'subdir'
    var_8 = [var_7]
    var_9 = 'file2.py'
    var_10 = 'ignore.txt'
    var_11 = [var_9, var_10]
    var_12 = (var_6, var_8, var_11)
    var_13 = '.txt'



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = '/fake/dir'
    var_1 = '/fake/file.py'
    var_2 = '/non/existent'
    var_3 = [var_0, var_1, var_2]
    var_4 = []
    var_5 = []
    var_6 = '/fake/dir'
    var_7 = 'subdir'
    var_8 = [var_7]
    var_9 = 'file1.py'
    var_10 = 'ignore.txt'
    var_11 = [var_9, var_10]
    var_12 = (var_6, var_8, var_11)
    var_13 = '/fake/dir/subdir'



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'src'
    var_1 = 'subdir'
    var_2 = 'main.py'
    var_3 = "print('hello')"
    var_4 = 'utils.py'
    var_5 = 'def utils(): pass'
    var_6 = 'ignored.txt'
    var_7 = 'ignore me'
    var_8 = '/non/existent/path'
    var_9 = 'skipped_dir'
    var_10 = 'hidden.py'
    var_11 = 'secret'
    var_12 = '.py'
    var_13 = []
    var_14 = []
    var_15 = []
    var_16 = []
    var_17 = 0



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'src'
    var_1 = 'subdir'
    var_2 = 'main.py'
    var_3 = 'print(1)'
    var_4 = 'utils.py'
    var_5 = 'print(2)'
    var_6 = 'data.txt'
    var_7 = 'not python'
    var_8 = '.ignore_me'
    var_9 = 'secret.py'
    var_10 = 'secret'
    var_11 = '.py'
    var_12 = 'non_existent_path'
    var_13 = []
    var_14 = []



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'root'
    var_1 = 'file1.py'
    var_2 = 'content'
    var_3 = 'subdir'
    var_4 = 'file2.py'
    var_5 = 'skipped_dir'
    var_6 = 'file3.py'
    var_7 = 'not_a_python_file.txt'
    var_8 = 'non_existent_path'
    var_9 = []
    var_10 = []

def test_case_0():
    var_0 = '/tmp/test_file.py'
    var_1 = 'path'
    var_2 = 'isdir'
    var_3 = False
    var_4 = lambda x: var_3
    var_5 = 'exists'
    var_6 = True
    var_7 = lambda x: var_6
    var_8 = [var_0]
    var_9 = []
    var_10 = []

def test_case_0():
    var_0 = 'root'
    var_1 = 'subdir'
    var_2 = 'file1.py'
    var_3 = 'content'
    var_4 = []
    var_5 = []



# Parsed testcases at query #8
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
    var_8 = [var_7]
    var_9 = 'file1.py'
    var_10 = 'file2.txt'
    var_11 = [var_9, var_10]
    var_12 = (var_6, var_8, var_11)
    var_13 = '.py'
    var_14 = '/root'
    var_15 = [var_14]
    var_16 = list(var_7)
    var_17 = '/root'
    var_18 = 'skipped_dir'
    var_19 = [var_18]
    var_20 = 'file1.py'
    var_21 = [var_20]
    var_22 = (var_17, var_19, var_21)
    var_23 = [var_17]
    var_24 = list(var_12)
    var_25 = any(var_13)



# Parsed testcases at query #9
#--------------------------


def test_case_0():
    var_0 = 'project'
    var_1 = 'src'
    var_2 = 'main.py'
    var_3 = "print('hello')"
    var_4 = 'utils.py'
    var_5 = 'pass'
    var_6 = 'tests'
    var_7 = 'test_main.py'
    var_8 = 'test'
    var_9 = 'ignored'
    var_10 = 'secret.py'
    var_11 = 'hidden'
    var_12 = 'non_existent'
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
    var_0 = 'symlink_test'
    var_1 = 'target'
    var_2 = 'file.py'
    var_3 = 'data'
    var_4 = 'link'
    var_5 = []
    var_6 = []



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = '/root'
    var_1 = '/nonexistent'
    var_2 = '/file.py'
    var_3 = [var_0, var_1, var_2]
    var_4 = []
    var_5 = []
    var_6 = '/root'
    var_7 = '/nonexistent'
    var_8 = 'subdir'
    var_9 = 'ignored_dir'
    var_10 = [var_8, var_9]
    var_11 = 'script.py'
    var_12 = 'README.md'
    var_13 = [var_11, var_12]
    var_14 = (var_6, var_10, var_13)
    var_15 = '/root/subdir'
    var_16 = []
    var_17 = 'sub_script.py'
    var_18 = [var_17]
    var_19 = (var_15, var_16, var_18)



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'dir1'
    var_1 = 'file1.py'
    var_2 = 'non_existent'
    var_3 = [var_0, var_1, var_2]
    var_4 = []
    var_5 = []
    var_6 = 'dir1'
    var_7 = 'subdir'
    var_8 = [var_7]
    var_9 = 'src.py'
    var_10 = [var_9]
    var_11 = (var_6, var_8, var_10)
    var_12 = 'dir1/subdir'
    var_13 = []
    var_14 = 'module.py'
    var_15 = [var_14]
    var_16 = (var_12, var_13, var_15)
    var_17 = 'file1.py'
    var_18 = [var_17]
    var_19 = []
    var_20 = []
    var_21 = [var_6]
    var_22 = []
    var_23 = []
    var_24 = 'dir1/src.txt'
    var_25 = []
    var_26 = 'src.txt'
    var_27 = [var_9, var_26]
    var_28 = (var_6, var_25, var_27)
    var_29 = [var_6]
    var_30 = []
    var_31 = []
    var_32 = 'dir1/skip.py'
    var_33 = []
    var_34 = 'skip.py'
    var_35 = [var_34]
    var_36 = (var_6, var_33, var_35)
    var_37 = [var_6]



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'test_root'
    var_1 = True
    var_2 = 'subdir'
    var_3 = 'file1.py'
    var_4 = 'file2.py'
    var_5 = 'content'
    var_6 = 'skipped_dir'
    var_7 = 'skip.py'
    var_8 = 'non_existent_path'
    var_9 = 'single.py'
    var_10 = [var_0]
    var_11 = []
    var_12 = []
    var_13 = list(var_1)
    var_14 = len(var_13)
    var_15 = [var_4, var_8]
    var_16 = []
    var_17 = []
    var_18 = list(var_5)
    var_19 = 'skip.py'
    var_20 = []
    var_21 = []
    var_22 = [var_6]
    var_23 = list(var_7)
    var_24 = False
    var_25 = '.py'
    var_26 = 'test.txt'
    var_27 = 'content'
    var_28 = []
    var_29 = []



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = '/fake/path/dir'
    var_3 = '/fake/path/file.py'
    var_4 = '/non/existent/path'
    var_5 = [var_2, var_3, var_4]
    var_6 = '/fake/path/dir'
    var_7 = 'subdir'
    var_8 = [var_7]
    var_9 = 'file1.py'
    var_10 = 'ignore.txt'
    var_11 = [var_9, var_10]
    var_12 = (var_6, var_8, var_11)
    var_13 = '/fake/path/dir/subdir'
    var_14 = []
    var_15 = 'file2.py'
    var_16 = [var_15]
    var_17 = (var_13, var_14, var_16)



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'test_root'
    var_3 = 'file1.py'
    var_4 = 'content'
    var_5 = 'subdir'
    var_6 = 'file2.py'
    var_7 = 'skipped_dir'
    var_8 = 'file3.py'
    var_9 = 'file_unsupported.txt'
    var_10 = 'non_existent_path'
    var_11 = '.txt'



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = '/valid/dir'
    var_1 = '/single/file.py'
    var_2 = '/non/existent'
    var_3 = [var_0, var_1, var_2]
    var_4 = []
    var_5 = []
    var_6 = '/valid/dir'
    var_7 = 'subdir'
    var_8 = 'ignored_dir'
    var_9 = [var_7, var_8]
    var_10 = 'file1.py'
    var_11 = 'ignore.txt'
    var_12 = [var_10, var_11]
    var_13 = (var_6, var_9, var_12)
    var_14 = []
    var_15 = []
    var_16 = [var_6]
    var_17 = list(var_10)
    var_18 = len(var_17)
    assert var_18 == 0

def test_case_0():
    var_0 = '/root'
    var_1 = [var_0]
    var_2 = []
    var_3 = []
    var_4 = '/root'
    var_5 = 'dir1'
    var_6 = [var_5]
    var_7 = 'file1.py'
    var_8 = [var_7]
    var_9 = (var_4, var_6, var_8)
    var_10 = '/root/dir1'
    var_11 = 'dir2'
    var_12 = [var_11]
    var_13 = 'file2.py'
    var_14 = [var_13]
    var_15 = (var_10, var_12, var_14)



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'test_root'
    var_3 = True
    var_4 = 'subdir'
    var_5 = 'file1.py'
    var_6 = 'print(1)'
    var_7 = 'file2.py'
    var_8 = 'print(2)'
    var_9 = 'readme.txt'
    var_10 = 'text'
    var_11 = 'skip_me'
    var_12 = 'hidden.py'
    var_13 = 'hidden'
    var_14 = [var_2]
    var_15 = list(var_3)
    var_16 = 'skip_me'
    var_17 = any(var_6)
    assert var_17 is False
    var_18 = any(var_8)
    var_19 = []
    var_20 = []
    var_21 = [var_9]
    var_22 = [var_11]
    var_23 = []
    var_24 = []
    var_25 = '/non/existent/path'
    var_26 = [var_25]
    var_27 = []
    var_28 = []



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'src'
    var_1 = 'file1.py'
    var_2 = 'print(1)'
    var_3 = 'dir1'
    var_4 = 'file2.py'
    var_5 = 'print(2)'
    var_6 = 'skipped_dir'
    var_7 = 'file3.py'
    var_8 = 'print(3)'
    var_9 = 'readme.txt'
    var_10 = 'text'
    var_11 = 'non_existent'
    var_12 = '.py'
    var_13 = []
    var_14 = []
    var_15 = []
    var_16 = []

def test_case_0():
    var_0 = 'src'
    var_1 = 'sub'
    var_2 = 'sub.py'
    var_3 = 'content'
    var_4 = []
    var_5 = []



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'valid_file.py'
    var_3 = 'valid_dir'
    var_4 = 'non_existent'
    var_5 = [var_2, var_3, var_4]
    var_6 = 'subdir'
    var_7 = [var_6]
    var_8 = 'file1.py'
    var_9 = 'file2.txt'
    var_10 = [var_8, var_9]
    var_11 = (var_3, var_7, var_10)
    var_12 = 'valid_dir/subdir'
    var_13 = []
    var_14 = 'subfile.py'
    var_15 = [var_14]
    var_16 = (var_12, var_13, var_15)
    var_17 = [var_2, var_4, var_3]



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'test_file.py'
    var_3 = "print('hello')"
    var_4 = 'subdir'
    var_5 = 'sub_file.py'
    var_6 = "print('sub')"
    var_7 = 'skipped.py'
    var_8 = 'pass'
    var_9 = 'skip_me'
    var_10 = 'hidden.py'
    var_11 = len(var_1)
    assert var_11 == 0
    var_12 = 'does_not_exist.py'
    var_13 = 'unsupported.txt'
    var_14 = 'text'
    var_15 = [var_4]
    var_16 = []
    var_17 = []
    var_18 = list(var_7)
    var_19 = '.txt'



# Parsed testcases at query #20
#--------------------------


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'project'
    var_3 = 'dir_a'
    var_4 = 'file_a.py'
    var_5 = "print('hello')"
    var_6 = 'dir_b'
    var_7 = 'file_b.py'
    var_8 = "print('world')"
    var_9 = 'skipped_dir'
    var_10 = 'skip.py'
    var_11 = 'skip'
    var_12 = 'readme.txt'
    var_13 = 'text'
    var_14 = '.py'
    var_15 = 'non_existent_path'
    var_16 = []
    var_17 = []
    var_18 = []
    var_19 = 'invalid_path_123'
    var_20 = [var_19]
    var_21 = []
    var_22 = '.txt'
    var_23 = []
    var_24 = []



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'root'
    var_1 = 'file1.py'
    var_2 = "print('hello')"
    var_3 = 'subdir'
    var_4 = 'file2.py'
    var_5 = "print('world')"
    var_6 = 'skipped_dir'
    var_7 = 'file3.py'
    var_8 = "print('skip me')"
    var_9 = 'ignored_file.txt'
    var_10 = 'not python'
    var_11 = 'non_existent_dir'
    var_12 = '.py'
    var_13 = []
    var_14 = []



# Parsed testcases at query #22
#--------------------------


def test_case_0():
    var_0 = '/valid/dir'
    var_1 = '/valid/file.py'
    var_2 = '/non/existent'
    var_3 = [var_0, var_1, var_2]
    var_4 = []
    var_5 = []
    var_6 = '/valid/dir/subdir_normal'
    var_7 = 'dirnames'
    var_8 = 'filenames'
    var_9 = 'subdir_to_skip'
    var_10 = 'subdir_normal'
    var_11 = [var_9, var_10]
    var_12 = 'file2.py'
    var_13 = 'file3.txt'
    var_14 = [var_12, var_13]
    var_15 = {var_7: var_11, var_8: var_14}
    var_16 = []
    var_17 = 'file1.py'
    var_18 = [var_17]
    var_19 = {var_7: var_16, var_8: var_18}
    var_20 = {var_0: var_15, var_6: var_19}
    var_21 = '.py'
    var_22 = list(var_0)
    var_23 = len(var_22)
    assert var_23 == 3
    var_24 = 'subdir_to_skip'
    var_25 = any(var_6)



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'test_root'
    var_1 = True
    var_2 = 'subdir'
    var_3 = 'file1.py'
    var_4 = 'print(1)'
    var_5 = 'file2.py'
    var_6 = 'print(2)'
    var_7 = 'skipped_dir'
    var_8 = 'skip.py'
    var_9 = 'skip'
    var_10 = [var_0]
    var_11 = []
    var_12 = []
    var_13 = list(var_1)
    var_14 = len(var_12)
    assert var_14 == 0
    var_15 = 'non_existent_path'
    var_16 = []
    var_17 = []
    var_18 = list(var_6)
    var_19 = [var_7]
    var_20 = []
    var_21 = []
    var_22 = [var_8]
    var_23 = '.txt'
    var_24 = 'test.txt'
    var_25 = 'text'
    var_26 = []
    var_27 = []



