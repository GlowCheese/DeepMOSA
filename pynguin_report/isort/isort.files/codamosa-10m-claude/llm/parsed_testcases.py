####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'Test the find function with various scenarios.'
    var_1 = False
    var_2 = []
    var_3 = []
    var_4 = []

def test_case_0():
    var_0 = 'Test find function with a directory containing Python files.'
    var_1 = 'test.py'
    var_2 = 'test.txt'
    var_3 = False
    var_4 = '.py'
    var_5 = lambda x: x.endswith(var_4)
    var_6 = []
    var_7 = []

def test_case_0():
    var_0 = 'Test find function skips files marked as skipped.'
    var_1 = 'skip_me.py'
    var_2 = True
    var_3 = []
    var_4 = []
    var_5 = len(var_3)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_3[var_6]

def test_case_0():
    var_0 = 'Test find function skips directories marked as skipped.'
    var_1 = 'skip_dir'
    var_2 = 'test.py'
    var_3 = True
    var_4 = []
    var_5 = []
    var_6 = len(var_4)
    assert var_6 == 1

def test_case_0():
    var_0 = 'Test find function with non-existent path.'
    var_1 = []
    var_2 = []
    var_3 = '/nonexistent/path'
    var_4 = [var_3]
    var_5 = len(var_2)
    assert var_5 == 1

def test_case_0():
    var_0 = 'Test find function with a single file path.'
    var_1 = 'single.py'
    var_2 = []
    var_3 = []

def test_case_0():
    var_0 = 'Test find function with both file and directory paths.'
    var_1 = 'subdir'
    var_2 = 'dir_file.py'
    var_3 = 'standalone.py'
    var_4 = False
    var_5 = '.py'
    var_6 = lambda x: x.endswith(var_5)
    var_7 = []
    var_8 = []

def test_case_0():
    var_0 = 'Test find function respects follow_links configuration.'
    var_1 = 'test.py'
    var_2 = False
    var_3 = True
    var_4 = []
    var_5 = []

def test_case_0():
    var_0 = 'Test find function handles visited directories correctly.'
    var_1 = 'test.py'
    var_2 = False
    var_3 = True
    var_4 = []
    var_5 = []



# Parsed testcases at query #2
#--------------------------


import isort.settings as module_0

def test_case_0():
    var_0 = 'Test the find function with various scenarios.'
    var_1 = 'test.py'
    var_2 = 'test.txt'
    var_3 = 'subdir'
    var_4 = 'sub.py'
    var_5 = module_0.Config()
    var_6 = []
    var_7 = []

def test_case_0():
    var_0 = 'Test that skipped directories are properly excluded.'
    var_1 = 'include'
    var_2 = 'skip'
    var_3 = 'include.py'
    var_4 = 'skip.py'
    var_5 = '.py'
    var_6 = []
    var_7 = []
    var_8 = len(var_6)

import isort.settings as module_0

def test_case_0():
    var_0 = 'Test that a single file path is yielded directly.'
    var_1 = 'test.py'
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'Test that nonexistent paths are added to broken list.'
    var_1 = '/nonexistent/path/file.py'
    var_2 = module_0.Config()
    var_3 = []
    var_4 = []
    var_5 = [var_1]
    var_6 = module_1.find(var_5, var_2, var_3, var_4)
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 0

def test_case_0():
    var_0 = 'Test that skipped files are not yielded.'
    var_1 = 'skip.py'
    var_2 = 'include.py'
    var_3 = '.py'
    var_4 = 'skip'
    var_5 = []
    var_6 = []
    var_7 = len(var_5)
    assert var_7 == 1

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'Test finding files in an empty directory.'
    var_1 = module_0.Config()
    var_2 = []
    var_3 = []
    var_4 = module_1.find(var_0, var_1, var_2, var_3)
    var_5 = list(var_4)
    var_6 = len(var_5)
    assert var_6 == 0

def test_case_0():
    var_0 = 'Test that follow_links configuration is respected.'
    var_1 = 'real'
    var_2 = 'test.py'
    var_3 = '.py'
    var_4 = False
    var_5 = []
    var_6 = []

def test_case_0():
    var_0 = 'Test that visited directories prevent cycles.'
    var_1 = 'test.py'
    var_2 = '.py'
    var_3 = False
    var_4 = []
    var_5 = []



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'Test the find function with various directory and file scenarios.'
    var_1 = 'test1.py'
    var_2 = "print('test1')"
    var_3 = 'test2.py'
    var_4 = "print('test2')"
    var_5 = 'subdir'
    var_6 = 'test3.py'
    var_7 = "print('test3')"
    var_8 = 'readme.txt'
    var_9 = 'readme'
    var_10 = 'skipped'
    var_11 = 'test_skipped.py'
    var_12 = "print('skipped')"
    var_13 = []
    var_14 = []
    var_15 = 'nonexistent.py'
    var_16 = []
    var_17 = []

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'Test find function with empty directory.'
    var_1 = module_0.Config()
    var_2 = []
    var_3 = []
    var_4 = module_1.find(var_0, var_1, var_2, var_3)
    var_5 = list(var_4)
    var_6 = len(var_5)
    assert var_6 == 0

import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'Test find function respects follow_links configuration.'
    var_1 = 'test.py'
    var_2 = "print('test')"
    var_3 = 'link'
    var_4 = False
    var_5 = module_0.Config()
    var_6 = []
    var_7 = []
    var_8 = module_1.find(var_3, var_5, var_6, var_7)
    var_9 = list(var_8)
    var_10 = len(var_9)

import isort.settings as module_0

def test_case_0():
    var_0 = 'Test find function with multiple input paths.'
    var_1 = 'dir1'
    var_2 = 'test1.py'
    var_3 = "print('test1')"
    var_4 = 'dir2'
    var_5 = 'test2.py'
    var_6 = "print('test2')"
    var_7 = module_0.Config()
    var_8 = []
    var_9 = []



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'Test the find function with various scenarios.'
    var_1 = 'dir1'
    var_2 = 'dir2'
    var_3 = 'subdir'
    var_4 = 'test1.py'
    var_5 = 'test2.py'
    var_6 = 'test3.py'
    var_7 = 'test.txt'
    var_8 = False
    var_9 = '.py'
    var_10 = lambda x: x.endswith(var_9)
    var_11 = []
    var_12 = []

def test_case_0():
    var_0 = 'Test find function with direct file path.'
    var_1 = 'test.py'
    var_2 = False
    var_3 = True
    var_4 = []
    var_5 = []

def test_case_0():
    var_0 = 'Test find function with non-existent path.'
    var_1 = False
    var_2 = True
    var_3 = []
    var_4 = []
    var_5 = '/non/existent/path/to/file.py'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'Test find function skips files marked as skipped.'
    var_1 = 'include.py'
    var_2 = 'skip.py'
    var_3 = True
    var_4 = lambda x: var_2 in str(x)
    var_5 = []
    var_6 = []
    var_7 = len(var_5)
    assert var_7 == 1

def test_case_0():
    var_0 = 'Test find function skips directories marked as skipped.'
    var_1 = 'skip_dir'
    var_2 = 'include_dir'
    var_3 = 'test.py'
    var_4 = True
    var_5 = lambda x: var_1 in str(x)
    var_6 = []
    var_7 = []

def test_case_0():
    var_0 = 'Test find function filters unsupported file types.'
    var_1 = 'test.py'
    var_2 = 'test.txt'
    var_3 = False
    var_4 = '.py'
    var_5 = lambda x: x.endswith(var_4)
    var_6 = []
    var_7 = []

def test_case_0():
    var_0 = 'Test find function with follow_links enabled.'
    var_1 = 'real'
    var_2 = 'test.py'
    var_3 = False
    var_4 = True
    var_5 = []
    var_6 = []



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'Test the find function with various scenarios.'
    var_1 = False
    var_2 = True
    var_3 = []
    var_4 = []
    var_5 = []

def test_case_0():
    var_0 = 'Test find function with a single file path.'
    var_1 = False
    var_2 = True
    var_3 = []
    var_4 = []
    var_5 = 'test.py'
    var_6 = [var_5]
    var_7 = list(var_2)

def test_case_0():
    var_0 = 'Test find function with nonexistent file path.'
    var_1 = []
    var_2 = []
    var_3 = 'nonexistent.py'
    var_4 = [var_3]

def test_case_0():
    var_0 = 'Test find function with directory containing Python files.'
    var_1 = False
    var_2 = True
    var_3 = '/root'
    var_4 = 'subdir'
    var_5 = [var_4]
    var_6 = 'file1.py'
    var_7 = 'file2.py'
    var_8 = [var_6, var_7]
    var_9 = (var_3, var_5, var_8)
    var_10 = '/root/subdir'
    var_11 = []
    var_12 = 'file3.py'
    var_13 = [var_12]
    var_14 = (var_10, var_11, var_13)
    var_15 = [var_9, var_14]
    var_16 = []
    var_17 = []
    var_18 = '/root'
    var_19 = [var_18]
    var_20 = list(var_2)

def test_case_0():
    var_0 = 'Test find function skips files correctly.'
    var_1 = True
    var_2 = 'skip'
    var_3 = lambda x: var_2 in str(x)
    var_4 = '/root'
    var_5 = []
    var_6 = 'keep.py'
    var_7 = 'skip.py'
    var_8 = [var_6, var_7]
    var_9 = (var_4, var_5, var_8)
    var_10 = [var_9]
    var_11 = []
    var_12 = []
    var_13 = '/root'
    var_14 = [var_13]
    var_15 = list(var_2)
    var_16 = len(var_15)
    assert var_16 == 1

def test_case_0():
    var_0 = 'Test find function filters unsupported file types.'
    var_1 = False
    var_2 = '.py'
    var_3 = lambda x: x.endswith(var_2)
    var_4 = '/root'
    var_5 = []
    var_6 = 'file.py'
    var_7 = 'file.txt'
    var_8 = 'file.md'
    var_9 = [var_6, var_7, var_8]
    var_10 = (var_4, var_5, var_9)
    var_11 = [var_10]
    var_12 = []
    var_13 = []
    var_14 = '/root'
    var_15 = [var_14]
    var_16 = list(var_2)
    var_17 = len(var_16)
    assert var_17 == 1

def test_case_0():
    var_0 = 'Test find function handles visited directories to avoid cycles.'
    var_1 = False
    var_2 = True
    var_3 = '/root'
    var_4 = 'subdir'
    var_5 = [var_4]
    var_6 = 'file.py'
    var_7 = [var_6]
    var_8 = (var_3, var_5, var_7)
    var_9 = '/root/subdir'
    var_10 = []
    var_11 = 'file2.py'
    var_12 = [var_11]
    var_13 = (var_9, var_10, var_12)
    var_14 = [var_8, var_13]
    var_15 = []
    var_16 = []
    var_17 = '/root'
    var_18 = [var_17]
    var_19 = list(var_2)
    var_20 = len(var_19)

def test_case_0():
    var_0 = 'Test find function skips directories correctly.'
    var_1 = True
    var_2 = '/root'
    var_3 = 'skip_dir'
    var_4 = 'keep_dir'
    var_5 = [var_3, var_4]
    var_6 = 'file.py'
    var_7 = [var_6]
    var_8 = (var_2, var_5, var_7)
    var_9 = [var_8]
    var_10 = []
    var_11 = []
    var_12 = '/root'
    var_13 = [var_12]
    var_14 = list(var_2)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'Test the find function with various scenarios.'
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = '/nonexistent/path'
    var_7 = [var_6]
    var_8 = []
    var_9 = []
    var_10 = list(var_3)
    var_11 = 'test.py'
    var_12 = 'test.txt'
    var_13 = ''
    var_14 = ''
    var_15 = False
    var_16 = '.py'
    var_17 = lambda x: x.endswith(var_16)
    var_18 = []
    var_19 = []
    var_20 = 'test.py'
    var_21 = ''
    var_22 = True
    var_23 = []
    var_24 = []
    var_25 = list(var_16)
    var_26 = len(var_23)
    assert var_26 == 1
    var_27 = 'subdir'
    var_28 = 'test.py'
    var_29 = ''
    var_30 = lambda x: var_29 in str(x)
    var_31 = True
    var_32 = []
    var_33 = []
    var_34 = len(var_32)
    var_35 = 'test.py'
    var_36 = ''
    var_37 = False
    var_38 = True
    var_39 = []
    var_40 = []
    var_41 = '/nonexistent'
    var_42 = '/path'
    var_43 = []
    var_44 = []
    var_45 = (var_42, var_43, var_44)
    var_46 = False
    var_47 = True
    var_48 = []
    var_49 = []
    var_50 = [var_42]
    var_51 = list(var_34)



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'Test the find function with various scenarios.'
    var_1 = False
    var_2 = True
    var_3 = []
    var_4 = []
    var_5 = 'subdir'
    var_6 = 'file1.py'
    var_7 = 'file2.txt'
    var_8 = 'file3.py'

def test_case_0():
    var_0 = 'Test find function with skipped files.'
    var_1 = True
    var_2 = []
    var_3 = []
    var_4 = 'file1.py'
    var_5 = 'skip_me.py'
    var_6 = 'file2.py'
    var_7 = 'skip_me'
    var_8 = len(var_2)
    assert var_8 == 1

def test_case_0():
    var_0 = 'Test find function with skipped directories.'
    var_1 = True
    var_2 = []
    var_3 = []
    var_4 = 'skip_dir'
    var_5 = 'file1.py'
    var_6 = 'file2.py'

def test_case_0():
    var_0 = 'Test find function with a single file path.'
    var_1 = False
    var_2 = True
    var_3 = []
    var_4 = []
    var_5 = 'test.py'
    var_6 = var_0 / var_5
    var_7 = str(var_6)
    var_8 = [var_7]
    var_9 = str(var_6)

def test_case_0():
    var_0 = 'Test find function with nonexistent path.'
    var_1 = False
    var_2 = True
    var_3 = []
    var_4 = []
    var_5 = '/nonexistent/path/file.py'
    var_6 = [var_5]

def test_case_0():
    var_0 = 'Test find function with multiple paths.'
    var_1 = False
    var_2 = True
    var_3 = []
    var_4 = []
    var_5 = 'file1.py'
    var_6 = var_0 / var_5
    var_7 = 'file2.py'

def test_case_0():
    var_0 = 'Test find function filters out unsupported filetypes.'
    var_1 = False
    var_2 = '.py'
    var_3 = lambda x: x.endswith(var_2)
    var_4 = []
    var_5 = []
    var_6 = 'file1.py'
    var_7 = 'file2.txt'
    var_8 = 'file3.py'
    var_9 = '.py'



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'Test the find function with various scenarios.'
    var_1 = False
    var_2 = True
    var_3 = []
    var_4 = []
    var_5 = '/test/dir'
    var_6 = 'subdir'
    var_7 = [var_6]
    var_8 = 'file1.py'
    var_9 = 'file2.txt'
    var_10 = [var_8, var_9]
    var_11 = (var_5, var_7, var_10)
    var_12 = '/test/dir/subdir'
    var_13 = []
    var_14 = 'file3.py'
    var_15 = [var_14]
    var_16 = (var_12, var_13, var_15)
    var_17 = '/test/dir'
    var_18 = [var_17]
    var_19 = list(var_7)
    var_20 = len(var_19)
    var_21 = len(var_4)
    assert var_21 == 0
    var_22 = 'skip'
    var_23 = lambda x: var_22 in str(x)
    var_24 = []
    var_25 = []
    var_26 = '/test/dir'
    var_27 = 'skip_dir'
    var_28 = 'normal_dir'
    var_29 = [var_27, var_28]
    var_30 = 'file.py'
    var_31 = [var_30]
    var_32 = (var_26, var_29, var_31)
    var_33 = '/test/dir'
    var_34 = [var_33]
    var_35 = list(var_28)
    var_36 = any(var_12)
    var_37 = []
    var_38 = []
    var_39 = '/nonexistent/path'
    var_40 = [var_39]
    var_41 = list(var_28)
    var_42 = len(var_41)
    assert var_42 == 0
    var_43 = []
    var_44 = []
    var_45 = '/test/file.py'
    var_46 = [var_45]
    var_47 = list(var_28)
    var_48 = len(var_44)
    assert var_48 == 0
    var_49 = []
    var_50 = []
    var_51 = '/test/dir'
    var_52 = []
    var_53 = 'file.py'
    var_54 = [var_53]
    var_55 = (var_51, var_52, var_54)
    var_56 = 'file.py'
    var_57 = lambda x: var_56 in str(x)
    var_58 = '/test/dir'
    var_59 = [var_58]
    var_60 = list(var_55)
    var_61 = len(var_49)
    var_62 = []
    var_63 = []
    var_64 = '/test/dir'
    var_65 = 'subdir'
    var_66 = [var_65]
    var_67 = 'file.py'
    var_68 = [var_67]
    var_69 = (var_64, var_66, var_68)
    var_70 = '/test/dir/subdir'
    var_71 = [var_65]
    var_72 = []
    var_73 = (var_70, var_71, var_72)
    var_74 = '/test/dir'
    var_75 = [var_74]
    var_76 = list(var_66)



# Parsed testcases at query #4
#--------------------------


import isort.settings as module_0
import isort.files as module_1

def test_case_0():
    var_0 = 'Test the find function with various scenarios.'
    var_1 = module_0.Config()
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = module_1.find(var_4, var_1, var_2, var_3)
    var_6 = list(var_5)
    var_7 = 'test.py'
    var_8 = module_0.Config()
    var_9 = []
    var_10 = []
    var_11 = 'file1.py'
    var_12 = 'file2.py'
    var_13 = 'file.txt'
    var_14 = module_0.Config()
    var_15 = []
    var_16 = []
    var_17 = module_0.Config()
    var_18 = []
    var_19 = []
    var_20 = '/nonexistent/path/file.py'
    var_21 = [var_20]
    var_22 = module_1.find(var_21, var_17, var_18, var_19)
    var_23 = list(var_22)
    var_24 = 'subdir'
    var_25 = 'file1.py'
    var_26 = 'file2.py'
    var_27 = module_0.Config()
    var_28 = []
    var_29 = []
    var_30 = 'file.py'
    var_31 = []
    var_32 = []
    var_33 = module_1.find(var_26, var_27, var_31, var_32)
    var_34 = list(var_33)
    var_35 = len(var_31)
    var_36 = 'dir1'
    var_37 = 'file1.py'
    var_38 = 'file2.py'
    var_39 = 'file3.py'
    var_40 = module_0.Config()
    var_41 = []
    var_42 = []
    var_43 = 'file.py'
    var_44 = module_0.Config()
    var_45 = []
    var_46 = []
    var_47 = module_1.find(var_38, var_44, var_45, var_46)
    var_48 = list(var_47)



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'Test the find function with various scenarios.'
    var_1 = []
    var_2 = []
    var_3 = []

def test_case_0():
    var_0 = 'Test find with a single file path.'
    var_1 = []
    var_2 = []
    var_3 = 'test.py'
    var_4 = [var_3]

def test_case_0():
    var_0 = 'Test find with a nonexistent file path.'
    var_1 = []
    var_2 = []
    var_3 = 'nonexistent.py'
    var_4 = [var_3]

def test_case_0():
    var_0 = 'Test find with a directory path.'
    var_1 = []
    var_2 = []
    var_3 = '/root'
    var_4 = 'subdir'
    var_5 = [var_4]
    var_6 = 'file1.py'
    var_7 = 'file2.txt'
    var_8 = [var_6, var_7]
    var_9 = (var_3, var_5, var_8)
    var_10 = '/root/subdir'
    var_11 = []
    var_12 = 'file3.py'
    var_13 = [var_12]
    var_14 = (var_10, var_11, var_13)
    var_15 = [var_9, var_14]
    var_16 = '/root'
    var_17 = [var_16]
    var_18 = list(var_4)

def test_case_0():
    var_0 = 'Test find skips directories marked as skipped.'
    var_1 = []
    var_2 = []
    var_3 = '/root'
    var_4 = 'skip_dir'
    var_5 = 'keep_dir'
    var_6 = [var_4, var_5]
    var_7 = 'file1.py'
    var_8 = [var_7]
    var_9 = (var_3, var_6, var_8)
    var_10 = [var_9]
    var_11 = '/root'
    var_12 = [var_11]
    var_13 = list(var_4)

def test_case_0():
    var_0 = 'Test find filters out unsupported file types.'
    var_1 = []
    var_2 = []
    var_3 = '/root'
    var_4 = []
    var_5 = 'file1.py'
    var_6 = 'file2.txt'
    var_7 = 'file3.pyc'
    var_8 = [var_5, var_6, var_7]
    var_9 = (var_3, var_4, var_8)
    var_10 = [var_9]
    var_11 = '/root'
    var_12 = [var_11]
    var_13 = list(var_4)

def test_case_0():
    var_0 = 'Test find skips files marked as skipped.'
    var_1 = []
    var_2 = []
    var_3 = '/root'
    var_4 = []
    var_5 = 'file1.py'
    var_6 = 'skip_file.py'
    var_7 = [var_5, var_6]
    var_8 = (var_3, var_4, var_7)
    var_9 = [var_8]
    var_10 = '/root'
    var_11 = [var_10]
    var_12 = list(var_4)
    var_13 = 'skip_file'

def test_case_0():
    var_0 = 'Test find handles visited directories to prevent cycles.'
    var_1 = []
    var_2 = []
    var_3 = '/root'
    var_4 = 'subdir'
    var_5 = [var_4]
    var_6 = 'file1.py'
    var_7 = [var_6]
    var_8 = (var_3, var_5, var_7)
    var_9 = [var_8]
    var_10 = '/root'
    var_11 = [var_10]
    var_12 = list(var_4)

def test_case_0():
    var_0 = 'Test find with multiple input paths.'
    var_1 = []
    var_2 = []
    var_3 = 'file1.py'
    var_4 = 'file2.py'
    var_5 = 'file3.py'
    var_6 = [var_3, var_4, var_5]



